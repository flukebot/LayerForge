package blueprint

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

// RunBenchmark initializes and runs the floating-point operation benchmarks within the Blueprint framework.
func (bp *Blueprint) RunBenchmark(duration time.Duration) (string, string, string, string, string, string, string, string) {
	bp.Duration = duration

	// Single-threaded benchmarks
	ops32Single := bp.runSingleThreadedBenchmark(true) / int(duration.Seconds())
	ops64Single := bp.runSingleThreadedBenchmark(false) / int(duration.Seconds())
	formattedOps32Single := bp.FormatNumber(ops32Single)
	formattedOps64Single := bp.FormatNumber(ops64Single)

	// Multi-threaded benchmarks
	ops32Multi := bp.runMultiThreadedBenchmark(true) / int(duration.Seconds())
	ops64Multi := bp.runMultiThreadedBenchmark(false) / int(duration.Seconds())
	formattedOps32Multi := bp.FormatNumber(ops32Multi)
	formattedOps64Multi := bp.FormatNumber(ops64Multi)

	// Estimation of max layers and nodes per layer
	maxLayers32Single, maxLayers64Single := bp.EstimateMaxLayersAndNodes(ops32Single, ops64Single)
	maxLayers32Multi, maxLayers64Multi := bp.EstimateMaxLayersAndNodes(ops32Multi, ops64Multi)

	return formattedOps32Single, formattedOps64Single, formattedOps32Multi, formattedOps64Multi, maxLayers32Single, maxLayers64Single, maxLayers32Multi, maxLayers64Multi
}

// runSingleThreadedBenchmark performs a single-threaded benchmark on float32 or float64 operations.
func (bp *Blueprint) runSingleThreadedBenchmark(isFloat32 bool) int {
	startTime := time.Now()
	ops := 0
	for time.Since(startTime) < bp.Duration {
		if isFloat32 {
			ops += bp.PerformFloat32Ops(1000)
		} else {
			ops += bp.PerformFloat64Ops(1000)
		}
	}
	return ops
}

// runMultiThreadedBenchmark performs a multi-threaded benchmark on float32 or float64 operations.
func (bp *Blueprint) runMultiThreadedBenchmark(isFloat32 bool) int {
	numCores := runtime.NumCPU()
	var wg sync.WaitGroup
	opsChan := make(chan int, numCores)

	for i := 0; i < numCores; i++ {
		wg.Add(1)
		if isFloat32 {
			go bp.workerBenchmark(bp.PerformFloat32Ops, opsChan, &wg)
		} else {
			go bp.workerBenchmark(bp.PerformFloat64Ops, opsChan, &wg)
		}
	}
	wg.Wait()
	close(opsChan)

	totalOps := 0
	for ops := range opsChan {
		totalOps += ops
	}
	return totalOps
}

// workerBenchmark performs operations for the multi-threaded benchmark.
func (bp *Blueprint) workerBenchmark(opFunc func(int) int, opsChan chan int, wg *sync.WaitGroup) {
	defer wg.Done()
	startTime := time.Now()
	ops := 0
	for time.Since(startTime) < bp.Duration {
		ops += opFunc(1000)
	}
	opsChan <- ops
}

// PerformFloat32Ops runs float32 multiply-add operations for benchmarking, returning the operation count.
func (bp *Blueprint) PerformFloat32Ops(count int) int {
	var a, b float32 = 1.1, 2.2
	var ops int
	for i := 0; i < count; i++ {
		a = a * b
		b = b + a
		ops++
	}
	return ops
}

// PerformFloat64Ops runs float64 multiply-add operations for benchmarking, returning the operation count.
func (bp *Blueprint) PerformFloat64Ops(count int) int {
	var a, b float64 = 1.1, 2.2
	var ops int
	for i := 0; i < count; i++ {
		a = a * b
		b = b + a
		ops++
	}
	return ops
}

// EstimateMaxLayersAndNodes estimates the maximum number of layers and nodes for a neural network based on operation count.
func (bp *Blueprint) EstimateMaxLayersAndNodes(ops32, ops64 int) (string, string) {
	const nodesPerLayer = 1000

	maxLayers32 := ops32 / (nodesPerLayer * nodesPerLayer)
	maxLayers64 := ops64 / (nodesPerLayer * nodesPerLayer)

	return bp.FormatNumber(maxLayers32), bp.FormatNumber(maxLayers64)
}

// FormatNumber formats large numbers into human-readable format with suffixes.
func (bp *Blueprint) FormatNumber(num int) string {
	switch {
	case float64(num) >= 1e12:
		return fmt.Sprintf("%.2f Trillion", float64(num)/1e12)
	case float64(num) >= 1e9:
		return fmt.Sprintf("%.2f Billion", float64(num)/1e9)
	case float64(num) >= 1e6:
		return fmt.Sprintf("%.2f Million", float64(num)/1e6)
	case float64(num) >= 1e3:
		return fmt.Sprintf("%.2f Thousand", float64(num)/1e3)
	default:
		return fmt.Sprintf("%d", num)
	}
}
