package main

import (
	"flag"
	"fmt"
	//"io/ioutil"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"runtime/pprof"
	"syscall"

	"net/http"
	_ "net/http/pprof"

	// "github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/examples/mnist"
	"gorgonia.org/tensor"

	"time"

	"gopkg.in/cheggaaa/pb.v1"
)

var (
	epochs     = flag.Int("epochs", 10, "Number of epochs to train for")
	dataset    = flag.String("dataset", "train", "Which dataset to train on? Valid options are \"train\" or \"test\"")
	dtype      = flag.String("dtype", "float64", "Which dtype to use")
	batchsize  = flag.Int("batchsize", 100, "Batch size")
	cpuprofile = flag.String("cpuprofile", "", "CPU profiling")
)

const loc = "../testdata/mnist/"

var dt tensor.Dtype

func parseDtype() {
	switch *dtype {
	case "float64":
		dt = tensor.Float64
	case "float32":
		dt = tensor.Float32
	default:
		log.Fatalf("Unknown dtype: %v", *dtype)
	}
}

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

type ffdnet struct {
	g                  *G.ExprGraph
	w0, w1 						 *G.Node // weights. the number at the back indicates which layer it's used for
	out								 *G.Node
}

// Build 2-layer feedforward network
func newFfdNet(g *G.ExprGraph) *ffdnet {
	w0 := G.NewMatrix(g, dt, G.WithShape(784, 128), G.WithName("w0"), G.WithInit(G.GlorotN(1.0)))
	w1 := G.NewMatrix(g, dt, G.WithShape(128, 10), G.WithName("w1"), G.WithInit(G.GlorotN(1.0)))
	return &ffdnet{
		g:  g,
		w0: w0,
		w1: w1,
	}
}

func (m *ffdnet) learnables() G.Nodes {
	return G.Nodes{m.w0, m.w1}
}

// Perform forward pass
func (m *ffdnet) fwd(x *G.Node) (err error) {

	// LAYER 0
	c0 := G.Must(G.Mul(x, m.w0))
	l0 := G.Must(G.Rectify(c0))

	// LAYER 1
	c1 := G.Must(G.Mul(l0, m.w1))

	//m.out = c1
	m.out = G.Must(G.SoftMax(c1))

	return
}

func main() {
	flag.Parse()
	parseDtype()
	rand.Seed(1337)

	// intercept Ctrl+C
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	doneChan := make(chan bool, 1)

	var inputs, targets tensor.Tensor
	var err error

	go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()

	trainOn := *dataset
	if inputs, targets, err = mnist.Load(trainOn, loc, dt); err != nil {
		log.Fatal(err)
	}

	// the data is in (numExamples, 784).
	numSamples := inputs.Shape()[0]
	bs := *batchsize
	// todo - check bs not 0

	log.Printf("numSamples = %d", numSamples)

	g := G.NewGraph()
	x := G.NewMatrix(g, dt, G.WithShape(bs, 784), G.WithName("x"))
	y := G.NewMatrix(g, dt, G.WithShape(bs, 10), G.WithName("y"))

	m := newFfdNet(g)
	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	one := G.NewConstant(1.0)

	// Here we create the nodes that contain the operations that
	// will calculate the cost function.
	// binary cross entropy: -y * log(prob) - (1-y)*log(1-prob)
	logh := G.Must(G.Log(m.out))
	fstTerm := G.Must(G.HadamardProd(G.Must(G.Neg(y)), logh))
	oneMinusY := G.Must(G.Sub(one, y))
	logOneMinusProb := G.Must(G.Log(G.Must(G.Sub(one, m.out))))
	sndTerm := G.Must(G.HadamardProd(oneMinusY, logOneMinusProb))
	losses := G.Must(G.Sub(fstTerm, sndTerm))
	cost := G.Must(G.Mean(losses))

 /*
	// In order to prevent overfitting, we add a L2 regularization term
	weightSq := G.Must(G.Square(w))
	sumSq := G.Must(G.Sum(weightSq))
	l2reg := G.NewConstant(0.01, G.WithName("l2reg"))
	regTerm := G.Must(G.Mul(l2reg, sumSq))

	// cost we want to minimize
	cost := G.Must(G.Add(loss, regTerm))
*/

	// we wanna track costs
	var costVal G.Value
	G.Read(cost, &costVal)

	if _, err = G.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	prog, locMap, _ := G.Compile(g)
	// This prints a long list of instructions...
	//log.Printf("%v", prog)

	vm := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap), G.BindDualValues(m.learnables()...))
	//solver := G.NewRMSPropSolver(G.WithBatchSize(float64(bs)))
	solver := G.NewVanillaSolver(G.WithBatchSize(float64(bs)), G.WithLearnRate(0.1))
	defer vm.Close()
	// pprof
	// handlePprof(sigChan, doneChan)

	var profiling bool
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		profiling = true
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	go cleanup(sigChan, doneChan, profiling)

	batches := numSamples / bs
	log.Printf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(80)

	for i := 0; i < *epochs; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numSamples {
				break
			}
			if end > numSamples {
				end = numSamples
			}

			var xVal, yVal tensor.Tensor
			if xVal, err = inputs.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice x")
			}

			if yVal, err = targets.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice y")
			}

			G.Let(x, xVal)
			G.Let(y, yVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d: %v", i, err)
			}
			solver.Step(G.NodesToValueGrads(m.learnables()))
			vm.Reset()
			bar.Increment()
		}
		log.Printf("Epoch %d | cost %v", i, costVal)
	}
	fmt.Print(m.out.Value())

}

func cleanup(sigChan chan os.Signal, doneChan chan bool, profiling bool) {
	select {
	case <-sigChan:
		log.Println("EMERGENCY EXIT!")
		if profiling {
			log.Println("Stop profiling")
			pprof.StopCPUProfile()
		}
		os.Exit(1)

	case <-doneChan:
		return
	}
}

func handlePprof(sigChan chan os.Signal, doneChan chan bool) {
	var profiling bool
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		profiling = true
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	go cleanup(sigChan, doneChan, profiling)
}
