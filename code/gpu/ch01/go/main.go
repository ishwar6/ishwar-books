// Goroutines: concurrency vs parallelism, measured.
//
// Part 1 (concurrency): 10,000 goroutines that each "wait on I/O" (sleep 100 ms).
//   Wall time stays ~100 ms even with GOMAXPROCS=1, because a waiting goroutine
//   holds no OS thread and no CPU. The scheduler parks it and runs another.
//   This is latency hiding by switching tasks -- the software cousin of what a
//   GPU warp scheduler does in hardware every few nanoseconds (§1.8).
//
// Part 2 (parallelism): a CPU-bound computation split into chunks, run with
//   GOMAXPROCS = 1, 2, 4, ... NumCPU. Only here does "more goroutines on more
//   cores" buy throughput -- and it stops at the number of cores.
//
//   go run .                     (or: GOMAXPROCS=4 go run .)
package main

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

// CPU-bound work with no memory traffic: an integer hash chain per element.
// Returns a checksum so the work cannot be optimized away and can be verified.
func work(lo, hi uint64) uint64 {
	var acc uint64
	for i := lo; i < hi; i++ {
		x := i
		for r := 0; r < 16; r++ { // 16 rounds of a xorshift-multiply mix
			x ^= x >> 33
			x *= 0xff51afd7ed558ccd
		}
		acc += x
	}
	return acc
}

func parallelWork(n uint64, chunks int) uint64 {
	var wg sync.WaitGroup
	partial := make([]uint64, chunks)
	for c := 0; c < chunks; c++ {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			lo := n * uint64(c) / uint64(chunks)
			hi := n * uint64(c+1) / uint64(chunks)
			partial[c] = work(lo, hi) // each goroutine writes only its own slot
		}(c)
	}
	wg.Wait()
	var s uint64
	for _, p := range partial {
		s += p // uint64 addition wraps and is associative: order doesn't matter
	}
	return s
}

func main() {
	fmt.Printf("NumCPU=%d  GOMAXPROCS=%d (default)\n\n", runtime.NumCPU(), runtime.GOMAXPROCS(0))

	// ---- Part 1: concurrency
	fmt.Println("Part 1: 10,000 goroutines, each sleeps 100 ms (stand-in for an I/O wait)")
	for _, procs := range []int{1, runtime.NumCPU()} {
		runtime.GOMAXPROCS(procs)
		var wg sync.WaitGroup
		t0 := time.Now()
		for i := 0; i < 10000; i++ {
			wg.Add(1)
			go func() { defer wg.Done(); time.Sleep(100 * time.Millisecond) }()
		}
		wg.Wait()
		fmt.Printf("  GOMAXPROCS=%2d: wall %6.1f ms   (serial would be %d s)\n",
			procs, float64(time.Since(t0).Microseconds())/1e3, 10000/10)
	}

	// ---- Part 2: parallelism
	const n = 200_000_000
	runtime.GOMAXPROCS(1)
	t0 := time.Now()
	ref := work(0, n)
	tSerial := time.Since(t0)
	fmt.Printf("\nPart 2: CPU-bound hash of %d integers (serial reference %.0f ms)\n",
		n, float64(tSerial.Microseconds())/1e3)
	fmt.Printf("  %10s %10s %9s\n", "GOMAXPROCS", "time", "speedup")
	for _, procs := range []int{1, 2, 4, 6, 8, 12, 16, 18, 24, 32} {
		if procs > runtime.NumCPU() {
			break
		}
		runtime.GOMAXPROCS(procs)
		best := time.Duration(1 << 62)
		var got uint64
		for rep := 0; rep < 3; rep++ {
			t0 := time.Now()
			got = parallelWork(n, 4*runtime.NumCPU()) // more chunks than cores: fine
			if d := time.Since(t0); d < best {
				best = d
			}
		}
		status := ""
		if got != ref {
			status = "  <-- WRONG"
		}
		fmt.Printf("  %10d %8.0f ms %8.2fx%s\n", procs, float64(best.Microseconds())/1e3,
			float64(tSerial)/float64(best), status)
	}
}
