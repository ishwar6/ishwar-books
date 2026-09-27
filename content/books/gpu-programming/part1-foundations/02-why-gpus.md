# Chapter 2 · Why GPUs Exist

> **Goal:** by the end of this chapter you can look at a piece of code and predict, before writing
> a single line of CUDA, (a) whether it will go fast on a GPU, (b) what its ceiling is in
> GFLOP/s or GB/s, and (c) which resource will be the bottleneck.
>
> This is the most valuable skill in the book, and it requires no GPU.

---

## 2.1 Two different bets

GPUs were built to color pixels. Each pixel is independent of the others: thousands of identical
tiny programs. In the mid-2000s people noticed that a matrix multiply looks the same: thousands of
independent dot products. CUDA (2006) was NVIDIA betting that this pattern was bigger than
graphics. The rest of this chapter is the quantitative version of that bet.

A CPU and a GPU are built from the same transistors. They spend them on opposite bets.

**The CPU's bet: one thing, as fast as possible.** Most programs are sequential and
latency-sensitive. So spend transistors on making a *single* instruction stream finish sooner:
huge caches (so memory is rarely slow), out-of-order execution and register renaming (so an
independent instruction runs while another stalls), branch prediction and speculation (so control
flow doesn't stall the pipeline), deep pipelines at high clock. On a modern desktop CPU the ALUs are
a rounding error in die area. Nearly all of the chip exists to *keep the ALUs fed on one thread*.

**The GPU's bet: many things, total work as high as possible.** Some programs have thousands of
independent identical units of work: every pixel, every element, every dot product. For those, you
don't need to make one unit fast. Delete the machinery that accelerates single streams, and spend
the transistors on arithmetic units instead. Then handle memory latency not by avoiding it but by
having so many other units of work available that something is always ready to run.

That single design choice explains every property of the GPU:

| | CPU | GPU |
|---|---|---|
| Optimizes | latency of one stream | throughput of all streams |
| On-chip memory the design relies on | megabytes of cache, shared by a few threads | kilobytes of registers / shared memory *per thread* |
| Hides memory latency by | caching, prefetch, out-of-order | **switching to another thread** |
| Cost of a branch | predicted, ~free when right | both sides of a warp run (Ch. 10) |
| Threads to be efficient | 1–16 | 10,000–100,000 |
| Idle when under-parallel | no | catastrophically |

The last row is the one that bites beginners. A GPU with 1,000 threads of work is not "a bit
underutilized"; it can be *slower than the CPU*, because you paid the transistor budget for
throughput and then didn't supply throughput.

**Four words, used constantly from here on:**

- **FLOP**: one floating-point add, subtract, multiply, or divide. A fused multiply-add
  (`a*b+c`, an **FMA**) counts as **2 FLOPs**. NVIDIA's peak numbers assume the ALUs issue FMAs.
  That is why matrix multiply is `2N³` FLOPs, not `N³`.
- **Kernel**: the function you launch on the GPU. Each **thread** runs that function on a
  different piece of data. "Write a kernel" means "write the function + the launch."
- **DRAM / HBM**: the GPU's large off-chip memory. The T4 uses GDDR6; A100/H100/B200 use HBM.
  "Bytes moved" in this chapter means to or from this pool, not registers or cache. The interview
  answers say HBM; same thing.
- **Warp**: 32 threads that execute in lockstep. Details in Chapters 4 and 10; you only need the
  word. A branch does not "cost a mispredict" the way it does on a CPU: the warp walks *both*
  paths, so half the threads sit idle on each path.

### The number that makes it concrete

Take the T4 you have free access to (free on Google Colab):

- **8.1 TFLOP/s** FP32: 8.1 × 10¹² floating-point ops per second. This is 2,560 CUDA cores ×
  ~1.59 GHz × 2 (because a peak-rate instruction is an FMA).
- **320 GB/s** memory bandwidth: the theoretical GDDR6 rate (256-bit bus × 10 Gb/s). NVIDIA's
  datasheet sometimes prints 300 GB/s; we use 320. Your measured ceiling will be a bit under
  either number.

Divide:

```
8.1e12 FLOP/s ÷ 320e9 byte/s ≈ 25 FLOP per byte
```

**To run at full compute speed, this GPU must perform ~25 floating-point operations for every byte
it reads from memory.** In FP32, a number is 4 bytes, so that's ~100 FLOPs per element loaded.

Sit with that. One hundred operations per element. Almost nothing you write naturally does that.

This ratio is the **machine balance** (also called the ridge point). Over thirty years, arithmetic
has gotten cheaper faster than memory has gotten faster: the "memory wall." HBM bought a
temporary reprieve on *CUDA-core* FP32 (the A100's balance is actually *lower* than the T4's).
Tensor cores blew the gap back open:

| GPU | Peak | Bandwidth | FLOP/byte needed |
|---|---|---|---|
| T4 (Turing, 2018), FP32 CUDA cores | 8.1 TFLOP/s | 320 GB/s | ~25 |
| T4, FP16 tensor cores | 65 TFLOP/s | 320 GB/s | ~200 |
| A100 80GB (Ampere, 2020), FP32 | 19.5 TFLOP/s | 2039 GB/s | ~10 |
| H100 SXM (Hopper, 2022), FP32 | 67 TFLOP/s | 3350 GB/s | ~20 |
| H100 SXM, tensor core BF16 *dense* | 989 TFLOP/s | 3350 GB/s | **~295** |
| B200 (Blackwell), tensor FP4 *dense* | ~9 PFLOP/s | 8000 GB/s | **~1100** |

Dense, not sparse: NVIDIA slides often quote 2:4-sparse peaks (H100 BF16 ~1979 TFLOP/s, B200 FP4
~18 PFLOP/s). Those are real only if your weights happen to be 50% zeros in the required pattern.
This book uses dense numbers unless it says otherwise.

The tensor-core rows are the punchline of the modern era. Blackwell's matrix units need ~1,100
FLOPs per byte of bandwidth just to stay fed. Which means: **on modern AI hardware, almost every
kernel is memory bound, and the entire discipline of GPU performance engineering is the art of
moving fewer bytes.**

That is why FlashAttention (Ch. 22) is a landmark result. It does not reduce FLOPs. It reduces
bytes.

🎯 *"Is this kernel compute bound or memory bound, and how do you know?"* is the single most common
opening question in a GPU performance interview. By the end of §2.4 you'll answer it in ten seconds.

**Bound** just means "this is the resource that will run out first." Memory bound: you finish the
math while still waiting on DRAM. Compute bound: the bytes arrive in time and the ALUs are the
limit. A kernel is one or the other (or a third thing: §2.3); it is not a moral judgment.

---

## 2.2 Arithmetic intensity

**Arithmetic intensity** (AI) of a computation is:

```
AI = (useful FLOPs performed) / (bytes moved between DRAM and the chip)
```

Three rules that people get wrong:

1. **Bytes moved to and from DRAM, not bytes touched.** If a value is loaded once and reused from
   cache or shared memory nine more times, it counts *once*. Improving arithmetic intensity is
   almost always about increasing reuse, not reducing math.
2. **Stores count.** Writing `c[i]` costs 4 bytes, same as reading `a[i]`. Forgetting the write
   makes vector-add look twice as intense as it is.
3. **Useful FLOPs only.** The index arithmetic (`i = blockIdx * blockDim + threadIdx`) does not
   count. NVIDIA's peak also assumes FMAs, so a kernel of only adds cannot hit the FP32 roof even
   if it is "compute bound."

AI is a property of the *algorithm and how it moves data*, not of the GPU. The GPU only supplies
the number you compare against (the machine balance). Same matmul, two AIs, two different kernels.

Let's compute it for real kernels. Assume FP32 (4 bytes), size `N`, and no on-chip reuse (worst
case: also the realistic case for streaming kernels whose working set does not fit in L2).

### Vector add: `c[i] = a[i] + b[i]`

- FLOPs: `N` (one add per element: not an FMA)
- Bytes: read `a`, read `b`, write `c` = `12N`
- **AI = 1/12 ≈ 0.083 FLOP/byte**

Against a machine balance of 25, we are **300× below** the compute-bound threshold. This kernel,
*as a standalone launch*, is hopelessly memory bound, and no amount of cleverness inside the
kernel changes that: the arithmetic is one add. The only optimization available is to hit peak
bandwidth. Best case, vector add on a T4 runs at 320 GB/s, i.e. ~26 GFLOP/s, which is **0.3% of
peak FLOPs**.

(The cleverness that *does* help is not writing this kernel at all: fuse the add into whatever
produced `a` and `b`. §2.4.)

⚠️ A kernel at 0.3% of peak FLOPs is not a bad kernel. It is a *perfect* vector add. Percent-of-peak
FLOPs is a meaningless metric for memory-bound work. Use percent-of-peak *bandwidth*.

### SAXPY: `y[i] = a*x[i] + y[i]`

SAXPY is "scalar a times x plus y," the BLAS-1 workhorse. `a` is one value, not an array: ignore
it in the byte count.

- FLOPs: `2N` (one multiply, one add: this *is* an FMA)
- Bytes: read `x`, read `y`, write `y` = `12N`
- **AI = 2/12 = 1/6 ≈ 0.167**: still memory bound by ~150×.

Twice the math of vector add, same traffic, still nowhere near 25.

### Naive matrix multiply, `C = A·B`, all N×N

- FLOPs: `2N³`. Each of the `N²` outputs is a dot product of length `N`, and each step is an FMA
  (2 FLOPs).
- Bytes, if you re-read from DRAM every time: each output reads a row of A (`N` numbers) and a
  column of B (`N` numbers), so `2N` reads per output × `N²` outputs × 4 bytes = `8N³`. The
  `N²` writes of C are `4N²`, which is rounding error next to `8N³`. **AI = 2N³ / 8N³ = 0.25.**
  Memory bound. This is the naive kernel in Chapter 16, and it gets ~1–2% of cuBLAS.
- Bytes, if you achieve *perfect* reuse: each of A, B, and C touches DRAM exactly once:
  `3N² × 4 = 12N²`. **AI = 2N³ / 12N² = N/6.**

For `N = 4096`: AI = 4096/6 ≈ 683 FLOP/byte. Wildly compute bound.

**The entire distance between a 1%-of-peak matmul and a 90%-of-peak matmul is the distance between
those two lines.** Same FLOPs. Different byte traffic. Chapter 16 walks that distance in seven
measured steps. Everything in between (tiling, shared memory, register blocking) is a technique
for moving from `8N³` bytes toward `12N²` bytes.

### Reduction: `sum(a[0..N])`

- FLOPs: `N` (really `N−1`; call it `N`)
- Bytes: `4N` read (output is one scalar, negligible)
- **AI = 0.25.** Memory bound. A perfect reduction runs at memory bandwidth. Chapter 12.

### Attention, one head, sequence length `S`, head dim `d`

This is the same ratio, one more time, on the kernel that pays for most of modern AI.

Two matmuls: `QKᵀ` produces an `S×S` score matrix, then `PV` mixes `V` with the softmaxed scores.

- FLOPs: `2·S·S·d` for `QKᵀ` plus `2·S·S·d` for `PV` = **`4S²d`**. (Softmax is ~5 FLOPs per score,
  so `~5S²`. For `d = 64` that is ~2% of the matmul FLOPs; ignore it for intensity.)
- Naive bytes: you materialize the `S×S` scores in DRAM. Write them after `QKᵀ`, read them for
  softmax, write the probabilities, read them for `PV`. That is 4 trips × `S²` × 4 bytes = `16S²`,
  plus a little workspace: call it **`~20S²`**. The `Q, K, V, O` traffic is `16Sd`, which is
  smaller than `20S²` whenever `S` is several times `d`.
- **AI ≈ `4S²d / 20S²` = `d/5`.** Independent of sequence length. For `d = 64`, AI ≈ 13, which on
  an H100 tensor core (needs ~295) is **memory bound by ~20×**.

FlashAttention never writes the `S×S` matrix to DRAM. Bytes drop to the inputs and outputs:
`Q, K, V, O` ≈ `16Sd`. Then **AI ≈ `4S²d / 16Sd` = `S/4`.** Intensity now *grows with sequence
length*. At `S = 2048`, AI ≈ 512: above the H100 tensor-core ridge, so compute bound. At
`S = 256`, AI ≈ 64: still memory bound. The win is "we raised AI by a factor of ~`S/d`"; whether
that crosses the ridge depends on `S`.

That's the whole idea of FlashAttention, derivable from a ratio, before you know what an online
softmax is. This is why arithmetic intensity is worth internalizing.

### 🎯 Interview drill

Compute the arithmetic intensity of these, FP32, assuming no reuse. Try each on paper before
reading the answer.

1. `y[i] = x[i] * x[i]` → 1 FLOP, read `x` + write `y` = 8 bytes → **0.125**. Memory bound.
2. `y[i] = exp(x[i])` → careful: `exp` is a multi-instruction sequence on the special-function
   unit (~8–10 SFU ops), 8 bytes → AI ≈ 1.2. Still memory bound on the T4 (ridge 25), and this is
   why activation functions are always fused rather than launched alone. (The SFU is also slower
   than an FMA, so "10 ops" is not 10 FP32 FLOPs of roof: another reason not to stare at
   percent-of-peak-FLOPs.)
3. 3×3 convolution, **one input channel, one output channel**, stride 1. Each output is 9
   multiply-adds = **18 FLOPs**. Naive: 9 input loads + 1 store = 40 bytes → **AI = 18/40 = 0.45**.
   With shared-memory tiling, each input value is reused by ~9 outputs, so the 9 loads collapse to
   ~1 load (4 B) plus the store (4 B) ≈ 8 bytes → **AI ≈ 18/8 = 2.25**. Still memory bound on a
   T4, but 5× less traffic: and that *is* the win.
4. `C = A·B` where `A` is **16×4096** and `B` is **4096×4096**: a skinny GEMM. This is LLM
   *decode* with batch 16: 16 tokens, each a vector, times a square weight matrix.
   - FLOPs: `2 · 16 · 4096 · 4096`
   - Bytes at minimum: `4 · (16·4096 + 4096² + 16·4096)` (read A, read B, write C)
   - The `4096²` weights dominate, so AI ≈ `2·16·4096² / (4·4096²)` = **8**.
   - At **batch 1**, the 16 becomes 1 and AI ≈ **0.5**.

   Compare either number to an H100 tensor-core ridge of ~295. Decode is memory bound; training
   and prefill, where the batch×sequence dimension is thousands, are compute bound (Ch. 28).

Number 4 is the most commercially important fact in this book.

---

## 2.3 The roofline model

Plot achievable performance (FLOP/s) against arithmetic intensity (FLOP/byte), both log scale. Two
limits:

```
performance ≤ peak_FLOPs                        (the flat "compute roof")
performance ≤ AI × peak_bandwidth               (the sloped "memory roof")
```

So:

```
attainable = min(peak_FLOPs, AI × peak_bandwidth)
```

Worked on the T4, so you can see the arithmetic:

| Kernel | AI | Memory roof (`AI × 320 GB/s`) | Compute roof | Attainable |
|---|---|---|---|---|
| Vector add | 0.083 | ~26 GFLOP/s | 8.1 TFLOP/s | **26 GFLOP/s** |
| Naive N×N matmul | 0.25 | 80 GFLOP/s | 8.1 TFLOP/s | **80 GFLOP/s** |
| Tiled matmul, N=4096 | 683 | way above peak | 8.1 TFLOP/s | **8.1 TFLOP/s** |

```
  FLOP/s
    ▲
    │                         ┌──────────────  peak compute (8.1 TFLOP/s)
    │                        ╱
    │                       ╱   ← slope = peak bandwidth (320 GB/s)
    │                      ╱
    │                     ╱
    │                    ╱ · · · · · · · · ·   ridge point (AI = 25)
    │                   ╱  │
    │   vadd    naive  ╱   │           tiled
    │   (0.08)  matmul╱    │           matmul
    │     ●     (0.25)     │           (683)
    │              ●  ╱    │              ●
    └────────────────╱─────┴──────────────────▶  AI (FLOP/byte)
                          25
```

The **ridge point** is the machine balance, `peak_FLOPs / peak_BW` = 25 for the T4. Left of it,
memory bound: the slope is your ceiling, and raising AI (fewer bytes, same math) slides you *up*
the slope. Right of it, compute bound: the flat roof is your ceiling, and raising AI further does
nothing.

Naive matmul (AI = 0.25) sits on the slope, next to vector add, not on the flat roof. Tiling is
what carries it across the ridge.

**How to use it, practically:**

1. Compute your kernel's AI on paper.
2. Read off the roofline ceiling: `min(peak_FLOPs, AI × peak_BW)`.
3. Measure what you actually achieve.
4. The gap between (2) and (3) is your optimization budget. Chase that gap, not "make it faster."
5. If you're *at* the ceiling and still too slow, you must change the *algorithm* to raise AI. No
   kernel tuning will help.

Step 5 is the one people skip, and it's where the big wins are. FlashAttention is a step-5 move.

**Caveats worth knowing** (interviewers like these):

- The simple roofline assumes only two resources. Real kernels can be limited by L2 bandwidth,
  shared-memory bandwidth, instruction issue, or atomics. Nsight Compute draws a *hierarchical*
  roofline with a roof per memory level; that's Chapter 18.
- Peak FLOPs depends on which unit: FP32 CUDA cores, FP64 units, or tensor cores are three
  different roofs, differing by more than 100×. Say which one you mean. A kernel that never
  issues an `mma` does not get the tensor-core roof.
- Peak FP32 further assumes FMAs. A kernel of only `add` or only `exp` has a lower compute roof.
- Peak bandwidth is never fully achievable; ~85–90% of spec is a realistic ceiling for a
  well-written streaming kernel.

Reference: Williams, Waterman & Patterson, *"Roofline: An Insightful Visual Performance Model"*
(CACM 2009): [ACM](https://dl.acm.org/doi/10.1145/1498765.1498785). The paper calls it
*operational* intensity; everyone in GPU-land says arithmetic intensity. Same ratio.

---

## 2.4 The four-question triage

Before writing any kernel, answer these. It takes two minutes and saves days.

**1. How many independent units of work are there?**
Fewer than ~10,000 → the GPU will be underutilized; consider whether the launch overhead
(~3–10 µs) even pays for itself. Fewer than ~100,000 → you will not saturate a modern
datacenter GPU. "Independent" means the work items do not have to wait for each other. A
recurrence `x[i] = f(x[i-1])` is one item, not `N`.

**2. What is the arithmetic intensity?**
Compare to the machine balance. This tells you which roof you're under, and therefore which
optimizations are even relevant. Memory bound → work on access patterns, fusion, data types.
Compute bound → work on ILP, tensor cores, instruction mix.

**3. What is the minimum possible byte traffic?**
Not what your code does: the *information-theoretic* minimum. Every input read once, every output
written once. That number, divided by peak bandwidth, is your kernel's speed-of-light runtime.
For memory-bound kernels this is the target and it's usually achievable within 10%.

**4. Is the access pattern regular?**
Consecutive threads touching consecutive addresses → coalesced, full bandwidth (Ch. 8).
Random gather/scatter → possibly 8–32× less effective bandwidth. Data-dependent control flow →
warp divergence (Ch. 10). Irregular problems (sparse, graphs) are absolutely doable on GPUs but need
different techniques and have lower ceilings.

### Worked example: element-wise `y = gelu(x)` on 100M FP32 elements

GELU (`0.5 · x · (1 + tanh(…))`, the common approximation) is ~15 FLOPs of mixed SFU/FMA work.

1. **Parallelism:** 100M independent elements. Plenty. ✓
2. **AI:** ~15 FLOPs per element over 8 bytes (read `x`, write `y`) ≈ 1.9. Machine balance 25.
   **Memory bound by ~13×.**
3. **Min traffic:** 100M × 8 bytes = 800 MB. At 320 GB/s → **2.5 ms**. That is the floor.
4. **Pattern:** perfectly regular, coalesced. ✓

Conclusions, before writing anything:

- Target 2.5 ms. If you hit 2.8 ms you are done: stop optimizing.
- Making the GELU math cheaper is pointless. At AI 1.9 vs. balance 25, arithmetic is ~8% of the
  time even if the SFUs are free. You cannot see a 2× from a faster `tanh`.
- The only real win left is **not moving those bytes at all**: fuse GELU into the producing kernel
  (the matmul that made `x`). Then extra traffic → 0, extra time → 0.
- Second real win: **do it in BF16**. Bytes halve, time halves, to 1.25 ms.

That's the reasoning chain that gets you hired. Notice that two of the four conclusions are "don't
write this kernel," and both are correct.

---

## 2.5 Little's Law: why you need so many threads

Bandwidth is not free at any parallelism. Little's Law, from queueing theory:

```
concurrency = throughput × latency
```

Applied to memory: to sustain `B` bytes/sec when each request takes `L` seconds to come back, you
must have `B × L` bytes **in flight** at all times. In flight means "requested from DRAM, not yet
arrived." The GPU hides that wait by switching to another thread that is not waiting.

T4, roughly: `B` = 320 GB/s, DRAM latency `L` ≈ 400 ns.

```
bytes in flight needed = 320e9 × 400e-9 = 128,000 bytes ≈ 128 KB
```

If each thread has one outstanding 4-byte load, you need **32,000 threads resident and issuing
loads** to saturate bandwidth. On an H100 (3350 GB/s, ~600 ns) it's ~2 MB in flight: half a million
4-byte requests.

This is the quantitative answer to "why does a GPU need tens of thousands of threads?" It is not
because the ALUs need feeding. It's because **memory latency can only be hidden by having enough
requests outstanding**, and each thread can only have a few.

Three consequences you'll use constantly:

1. **Occupancy matters: but only until bandwidth saturates.** More resident threads = more requests
   in flight, up to the point where you're at peak bandwidth. After that, more occupancy buys
   nothing. This is why "maximize occupancy" is wrong advice (Ch. 11).
2. **Bytes per thread substitute for threads.** If each thread issues four 16-byte vectorized loads
   instead of one 4-byte load, you need 16× fewer threads for the same bytes in flight. This is
   instruction-level parallelism, and it's why `float4` loads help so much (Ch. 8) and why
   low-occupancy register-blocked GEMMs work (Ch. 16).
3. **Small kernels can't reach peak bandwidth, ever.** A kernel over 10,000 FP32 elements has 40 KB
   of reads: less than the 128 KB that needs to be in flight. It will finish in the latency-bound
   regime: you paid the 400 ns, you never filled the pipe. Batch it or fuse it.

---

## 2.6 When not to use a GPU

Being able to say "this shouldn't be on a GPU" is a senior signal. Bad fits:

- **Not enough parallelism.** < ~10⁴ independent work items. The launch alone is microseconds.
- **Sequential dependencies you can't restructure.** Long recurrences, Newton iterations on a single
  variable, most branch-heavy business logic. (Note: many *apparently* sequential things
  parallelize: prefix sum is the classic, Ch. 14. Check before giving up.)
- **Latency-critical single requests.** Kernel launch + PCIe round trip is tens of microseconds.
  If your SLA is 50 µs end-to-end, the GPU may lose to a CPU that never leaves L2.
- **Working set that doesn't fit and doesn't stream.** If you must random-access 500 GB, you're
  bound by PCIe (~32 GB/s for Gen4 x16, ~64 GB/s Gen5) not GPU bandwidth. Sometimes a CPU with
  1 TB of RAM wins.
- **Tiny data with heavy control flow.** Divergence + no parallelism = worst case.

Two traps that look like the above but aren't:

**Memory-bound on the CPU is not a reason to skip the GPU.** GPU bandwidth is often 5–20× a CPU's.
A streaming reduction can still be a clean win: just don't expect a 100×. The question is whether
that 5–20× pays for getting the data there.

**The transfer can cost more than the compute.** The GPU's fast memory and the CPU's RAM are
separate. Crossing that bus is PCIe (or NVLink-C2C on Grace-Hopper; same idea, faster bus).
Moving 1 GB over PCIe Gen4 takes ~31 ms. A kernel that then *reads* that 1 GB from HBM on an H100
takes 1 GB / 3.35 TB/s ≈ **0.3 ms**. You spent 100× the compute time on transport. Rule of thumb:
**if data has to make a round trip to the GPU for one cheap kernel, don't.** Keep data resident and
do many kernels, or fuse.

---

## 2.7 Summary

- CPUs minimize latency for one stream; GPUs maximize throughput for many. Everything else follows.
- **Machine balance** = peak FLOPs / peak bandwidth. ~25 FLOP/byte on a T4's CUDA cores; ~300 on
  H100 tensor cores (dense BF16); ~1100 on Blackwell dense FP4.
- **Arithmetic intensity** = useful FLOPs / DRAM-bytes-moved (stores count; on-chip reuse doesn't).
  Compare it to machine balance to learn which roof you're under.
- Almost all real kernels are **memory bound**. Performance engineering is mostly about moving
  fewer bytes: reuse, tiling, fusion, smaller dtypes.
- The **roofline** turns this into a number: `min(peak_FLOPs, AI × BW)`. Measure the gap to it.
- **Little's Law** explains the thread counts: you need `BW × latency` bytes in flight.
- Triage every problem with the four questions before writing code.

---

## Exercises

Do these on paper. A calculator is fine; a GPU is not required. If a number depends on which GPU,
the problem says so.

**2.1** Compute the arithmetic intensity of a 2D 5-point stencil,
`out[i][j] = 0.2*(in[i][j] + in[i±1][j] + in[i][j±1])`, in FP32: (a) naive, no reuse; (b) with
perfect reuse (each interior input loaded from DRAM once). On the T4, what speedup does the
roofline predict from (a) to (b)? Hint: count FLOPs in `0.2*(a+b+c+d+e)` first, then bytes, then
check which side of the ridge each AI sits on.

**2.2** Suppose a GPU has 900 GB/s bandwidth and 19.5 TFLOP/s FP32 (made-up bandwidth, so the
arithmetic stays pretty). What is the ridge point? A kernel has AI = 4 and achieves 2.1 TFLOP/s.
What is its roofline ceiling, what fraction of that ceiling is 2.1, and what should you work on?

**2.3** Layer-norm over a `[B=8192, H=4096]` FP32 tensor reads the input, computes mean and
variance per row, then writes a normalized output. A naive implementation makes three passes over
`x` (mean, then variance, then normalize). (a) Bytes moved for three passes? (b) For a two-pass
version that computes mean and variance together (Welford), then normalizes? (c) Predicted time at
900 GB/s for each. (d) A *true* single-pass fused kernel still needs the stats before it can
normalize: so it cannot actually drop to one read of `x` without approximation. Is the two-pass
Welford version worth it versus three-pass? Versus chasing a one-pass scheme?

**2.4** An H100 has 3.35 TB/s and ~600 ns DRAM latency. How many bytes must be in flight? If each
thread issues one `float4` (16 B) load, how many threads? Compare to the max resident threads on an
H100 SXM (132 SMs × 2048 threads/SM). What does the comparison tell you about vectorized loads?
What would change if each thread issued only a 4-byte load?

**2.5** Why is percent-of-peak-FLOPs a misleading metric for a reduction kernel? Propose the metric
you'd report instead, and state what value would mean "optimal."

**2.6** You must apply 30 element-wise ops in sequence to a 10 GB tensor. Estimate the runtime
(a) as 30 separate kernels, (b) fused into one. At 2 TB/s. Assume each standalone kernel reads the
whole tensor and writes it back. This ratio is the entire reason `torch.compile` exists.

---

## 🎯 Interview questions

<details>
<summary><b>Q1. Why does a GPU need thousands of threads when it only has a few thousand cores?</b></summary>

Two distinct reasons, and a strong answer gives both:

1. **Latency hiding.** A DRAM access is ~400–600 ns, which is ~600–1000 clock cycles. When a warp
   issues a load and stalls, the SM's warp schedulers switch to another *resident* warp with zero
   context-switch cost (registers are physically partitioned, not saved/restored). You need enough
   resident warps that at any instant one has a ready instruction. This is the whole reason for the
   large register file.
2. **Little's Law.** To sustain peak bandwidth you need `bandwidth × latency` bytes in flight:
~128 KB on a T4, ~2 MB on an H100. With small per-thread requests, that arithmetic alone demands
   tens or hundreds of thousands of concurrent requests.

The follow-up is usually "so should you always maximize occupancy?": no; see Ch. 11. Once
bandwidth is saturated, extra warps add nothing, and the registers they consume may force spills.
</details>

<details>
<summary><b>Q2. Is a kernel achieving 3% of peak FLOPs badly optimized?</b></summary>

Unanswerable without the arithmetic intensity. If it's a vector add (AI ≈ 0.08) on a machine with
balance 25, then 0.08/25 = 0.3% of peak FLOPs *is* the roofline ceiling: 3% would be impossible,
and hitting ~0.3% while at 95% of peak bandwidth is optimal.

The right move is to reframe: compute AI, compute the roofline ceiling, report achieved bandwidth as
a fraction of peak. For memory-bound kernels the only honest metric is % of peak bandwidth.
</details>

<details>
<summary><b>Q3. FlashAttention doesn't reduce the FLOP count. Why is it faster?</b></summary>

Standard attention materializes the `S×S` score matrix in HBM: write it after `QKᵀ`, read it for
softmax, write the probabilities, read them for `PV`. That's `O(S²)` HBM traffic with a large
constant, and since `S²` dominates for long sequences, the kernel is memory bound: AI ≈ `d/5`, far
below the tensor-core ridge point of ~300.

FlashAttention tiles over the sequence and fuses `QKᵀ` → softmax → `PV` into one kernel, keeping
tiles in SRAM (shared memory/registers) and never writing the score matrix to HBM. It uses the
online-softmax recurrence to compute a numerically correct softmax without seeing the whole row at
once. HBM traffic falls to `O(Sd)`: the inputs and outputs only. AI rises by a factor of ~`S/d`
(from `d/5` to `S/4`). For long sequences that crosses the ridge and the tensor cores can actually
be fed; for short sequences you still win on bytes, you just stay on the memory roof.

It also removes the `O(S²)` memory *footprint*, which is what makes long contexts feasible at all.
Reference: [Dao et al. 2022](https://arxiv.org/abs/2205.14135). Detail in Chapter 22.
</details>

<details>
<summary><b>Q4. Why is LLM inference memory bound during decoding but compute bound during training?</b></summary>

The operation is the same matmul; the shapes differ. For `C[M,N] = A[M,K] · B[K,N]` in a `b`-byte
dtype, AI ≈ `2MNK / b(MK + KN + MN)`. When the weights `B` dominate, that simplifies to **`~2M / b`**:
i.e. `M/2` in FP32 and **`M` in FP16**.

Training / prefill processes many tokens at once. A weight matrix `[K, N]` is multiplied by an
activation `[M, K]` with `M` in the thousands (batch × sequence). Each weight element loaded is
reused `M` times, so AI is hundreds of FLOP/byte, comfortably compute bound.

Decoding generates one token per sequence per step, so `M` = batch size, often 1–64. Every weight
must be read from HBM to be used only `M` times. At `M = 1` in FP16, AI ≈ 1: deeply memory bound.
Runtime is then essentially `model_bytes / bandwidth`: a 70B model in FP16 is 140 GB. That does not
fit on one 80 GB H100, so the floor is 140 GB streamed across however many GPUs hold the weights.
At 3.35 TB/s *per GPU that is doing useful work*, a *single* H100-shaped slice of 80 GB is ~24 ms;
the 140 GB figure at one GPU's bandwidth is ~42 ms: the right order of magnitude for "time to
read the model once." No tensor-core clock rate changes that floor.

That single ratio drives most of modern inference engineering: batching (raise `M`), quantization
(shrink `b`), speculative decoding (verify many tokens in one pass, raising effective `M`), and
KV-cache management. Chapter 28.
</details>

<details>
<summary><b>Q5. When would you tell a team not to port their workload to a GPU?</b></summary>

Cover the categories, with the reasoning rather than a list: insufficient parallelism (< ~10⁴
independent items: the launch overhead and the inability to fill the machine dominate);
irreducible sequential dependence; latency SLAs tighter than launch + transfer (tens of µs);
working sets that must be randomly accessed from host memory, making PCIe the real bottleneck;
and the common one: a single cheap kernel per data round trip, where 31 ms of PCIe transfer wraps
0.3 ms of compute.

Do *not* reject a port just because the CPU kernel is already memory bound. The GPU still has more
bandwidth; the question is whether the speedup survives the transfer.

The strong version of this answer includes the fix rather than just the verdict: batch many
requests to create parallelism, keep data resident across many kernels, or fuse the chain of cheap
ops so the transfer amortizes.
</details>

---

**Next:** [Chapter 3: The Hardware](03-hardware.md)

**Sources:**
[Roofline (CACM 2009)](https://dl.acm.org/doi/10.1145/1498765.1498785) ·
[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) ·
[NVIDIA T4 datasheet](https://www.nvidia.com/en-us/data-center/tesla-t4/) ·
[NVIDIA H100 datasheet](https://www.nvidia.com/en-us/data-center/h100/) ·
[FlashAttention](https://arxiv.org/abs/2205.14135)
