# Chapter 4 · The Programming Model

> **Goal:** know exactly what the abstraction promises, what it doesn't, and how each level maps onto
> the hardware from Chapter 3. Most CUDA bugs are violations of the "doesn't promise" list.

---

## 4.1 The decomposition

You write a function, called a **kernel**, that describes the work of *one* thread. You then launch
many thousands of copies of it.

```cpp
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // which element am I?
    if (i < n) c[i] = a[i] + b[i];
}

// launch: 1024 blocks of 256 threads = 262,144 threads
add<<<1024, 256>>>(a, b, c, n);
```

The hierarchy:

```
Grid                       - all threads of one kernel launch
└── Cluster (Hopper+, opt) - group of blocks co-scheduled on one GPC, sharing DSMEM
    └── Block              - group of threads on ONE SM, sharing shared memory, can sync
        └── Warp           - 32 threads, hardware scheduling unit (implicit; you don't declare it)
            └── Thread     - one lane, private registers
```

And the mapping to hardware:

| Software | Hardware | Key property |
|---|---|---|
| Thread | lane in a processing block | private registers |
| Warp | scheduling unit | lockstep issue; divergence granularity |
| Block | **one SM**, start to finish | shared memory; `__syncthreads()` |
| Cluster | one GPC | distributed shared memory; `cluster.sync()` |
| Grid | whole GPU | no cheap sync (see §4.5) |

**The block↔SM binding is the load-bearing fact.** A block is assigned to one SM and never migrates.
Therefore:
- Everything a block's threads share must fit in *one SM's* shared memory and register budget.
- Threads in a block can synchronize cheaply (`__syncthreads()`), because they're physically
  co-located.
- Threads in *different* blocks essentially cannot synchronize (they may not even be running at the
  same time), which is what makes the model scale to any number of SMs.

That last point is the design genius of CUDA: because blocks are independent, the same binary runs on
a 20-SM laptop GPU and a 148-SM B200, with the hardware just scheduling more blocks concurrently.
**Blocks are the unit of scalability.**

---

## 4.2 The built-in variables, precisely

Inside a kernel you get:

| Variable | Type | Meaning |
|---|---|---|
| `threadIdx.x/.y/.z` | `uint3` | my index within my block |
| `blockDim.x/.y/.z` | `dim3` | size of a block |
| `blockIdx.x/.y/.z` | `uint3` | my block's index within the grid |
| `gridDim.x/.y/.z` | `dim3` | size of the grid, in blocks |
| `warpSize` | `int` | 32 (always, on NVIDIA) |

The canonical global index, which you will type ten thousand times:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

For 2D:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;   // x → fastest-varying → columns
int row = blockIdx.y * blockDim.y + threadIdx.y;
if (row < H && col < W) out[row * W + col] = ...;  // row-major
```

⚠️ **`x` must map to the contiguous dimension.** `threadIdx.x` is what varies within a warp
(lane = `threadIdx.x % 32` when `blockDim.x` is a multiple of 32), so `x` must index the
fastest-varying memory dimension or you destroy coalescing. Swapping `x` and `y` in a 2D kernel is a
one-character change that can cost 8× (you'll measure it in Ch. 8).

Specifically, thread linearization within a block is:

```
tid_in_block = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y
lane_id      = tid_in_block % 32
warp_id      = tid_in_block / 32
```

**x is innermost.** Memorize it.

### Launch configuration

```cpp
dim3 block(16, 16);                              // 256 threads
dim3 grid((W + 15) / 16, (H + 15) / 16);         // ceil-divide, covers all of W×H
kernel<<<grid, block>>>(args);
kernel<<<grid, block, shmem_bytes, stream>>>(args);   // full form
```

Limits (see Ch. 3 table): ≤1024 threads per block; block dims ≤ (1024, 1024, 64); grid x up to
2³¹−1, y and z up to 65535.

The `(n + b - 1) / b` ceiling-division idiom appears in every CUDA program ever written. Combined
with the `if (i < n)` guard inside the kernel, it handles sizes that aren't multiples of the block
size. **Forgetting the guard is the #1 beginner bug**: the last block has threads past the end, and
they'll happily write out of bounds.

### Choosing the block size

Start with **256**. Reasons: multiple of 32 (no partial warps wasted), divides evenly into the
threads/SM limit on every architecture, gives the scheduler 8 warps to play with, and leaves room
for several blocks per SM. Then tune 128 / 256 / 512 empirically. Chapter 11 gives the real
occupancy-driven rule and the API (`cudaOccupancyMaxPotentialBlockSize`) that picks it for you.

Never use a non-multiple of 32. A block of 100 threads occupies 4 warps (128 lanes) with 28 lanes
permanently masked off: you pay for them in scheduling slots and get nothing.

---

## 4.3 Function and variable qualifiers

```cpp
__global__ void k(...)     // kernel: called from host with <<<>>>, runs on device, returns void
__device__ float f(...)    // device function: callable only from device code, inlined by default
__host__   float g(...)    // host function (the default)
__host__ __device__ float h(...)  // compiled twice, callable from both - very useful for shared math
```

```cpp
__shared__ float tile[32][33];   // per-block scratchpad, static size
extern __shared__ float dyn[];   // dynamic size, given as 3rd launch arg
__constant__ float coef[64];     // 64 KB total, broadcast-optimized read-only
__device__ int global_counter;   // module-scope device global
__managed__ int unified_var;     // unified memory, accessible from host and device
```

`__restrict__` and `const` on pointer parameters are not decoration: they let the compiler prove
non-aliasing and use the read-only data path (`__ldg`), which measurably improves some kernels:

```cpp
__global__ void k(const float* __restrict__ in, float* __restrict__ out);
```

Get in the habit. It's free.

---

## 4.4 Synchronization: what exists at each level

| Scope | Primitive | Cost | Notes |
|---|---|---|---|
| Warp | `__syncwarp(mask)` | ~free | reconvergence; needed on Volta+ for warp-synchronous code |
| Block | `__syncthreads()` | tens of cycles | barrier + memory fence for shared & global |
| Block | `__syncthreads_and/or/count(pred)` | same | barrier that also reduces a predicate |
| Cluster | `cluster.sync()` (cc 9.0+) | more | across blocks in a cluster |
| Grid | `grid.sync()` via cooperative groups | **expensive** | requires cooperative launch; grid must fit resident |
| Device | `cudaDeviceSynchronize()` (host) | µs | host waits for all work |
| Stream | `cudaStreamSynchronize(s)` | µs | host waits for one stream |
| - | kernel boundary | ~µs | **the normal way to sync the whole grid** |

### The `__syncthreads()` rules ⚠️

`__syncthreads()` is a **barrier for all threads in the block** plus a memory fence: writes to shared
and global memory made before it are visible to all block threads after it.

Three rules, all of which people break:

**Rule 1: every thread in the block must reach it.** This is illegal and hangs or corrupts:

```cpp
if (threadIdx.x < 32) {
    __syncthreads();        // ⚠️ UB: threads ≥32 never arrive
}
```

The fix is to hoist the barrier out of the conditional:

```cpp
if (threadIdx.x < 32) { /* work */ }
__syncthreads();            // ✓ all threads reach it
```

**Rule 2: you need one after writing shared memory and before reading what others wrote.** The
classic tiling loop needs *two*:

```cpp
for (int t = 0; t < numTiles; ++t) {
    tile[ty][tx] = A[...];      // write
    __syncthreads();            // (1) make writes visible before anyone reads
    for (int k = 0; k < TILE; ++k) acc += tile[ty][k] * ...;   // read
    __syncthreads();            // (2) don't overwrite the tile while others still read it
}
```

Dropping barrier (2) is the subtle one: fast warps race ahead to iteration `t+1` and overwrite `tile`
while slow warps are still reading iteration `t`. **It produces wrong answers only sometimes, and
usually only at certain block sizes.** 🎯 This is a very common interview question: see Ch. 9.

**Rule 3: it synchronizes a block, not the grid.** There is no `__syncgrid()` in a plain kernel. If
step 2 of your algorithm needs *all* of step 1 done, you need either a second kernel launch, a
cooperative-groups grid sync, or a lock-free scheme like decoupled lookback (Ch. 14).

---

## 4.5 What the model does NOT guarantee

This list is the source of most CUDA bugs, and interviewers probe it. 🎯

**1. No guarantee about block execution order.** Blocks may run in any order, and any number may run
concurrently. Do not write code whose correctness depends on block 0 finishing before block 1.

**2. No guarantee that all blocks are resident simultaneously.** Launch 1,000,000 blocks on a 40-SM
GPU and they're scheduled in waves. Therefore:

```cpp
// ⚠️ DEADLOCK: block N spins waiting for block N+1, which may not be scheduled yet
while (atomicAdd(&flag, 0) == 0) { /* spin */ }
```

This is the classic "global barrier by spinning" bug. It deadlocks not because of a race, but because
the waiting block occupies an SM slot that the block it's waiting for needs.

The exception: **cooperative launch** (`cudaLaunchCooperativeKernel`, cc 6.0+). That launch is
*atomic* (if the call succeeds, every block you asked for is resident) which is what makes
`grid.sync()` legal. Sizing the grid so it fits is your job
(`cudaOccupancyMaxActiveBlocksPerMultiprocessor`); an oversized grid fails the launch instead of
deadlocking.
[Cooperative Groups docs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html).

**3. No guarantee about warp scheduling order.** Two warps in the same block interleave arbitrarily.

**4. No forward-progress guarantee across blocks.** Related to (2). Spin-waiting on another block is
never safe without cooperative launch.

**5. Nothing about floating-point associativity.** A parallel reduction sums in a different order than
a sequential one, so `parallel_sum(x) != sequential_sum(x)` in floating point: legitimately, not as
a bug. Two *runs* of the same reduction kernel can also differ if it uses atomics (nondeterministic
order). If you need bit-reproducibility you must fix the order (deterministic tree reduction) or use
compensated/Kahan summation.
🎯 "Why does my GPU reduction give a slightly different answer than my CPU loop?": this.

**6. Memory ordering is weak.** Without a fence, one thread's global writes are not guaranteed
visible to another thread in another block in any particular order. You need
`__threadfence()` / `__threadfence_block()` / `__threadfence_system()`, or the C++-style atomics in
`cuda::atomic` with explicit memory orders. Details:
[CUDA C++ memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html).

**7. `printf` from a kernel is buffered.** Output appears after a sync, in nondeterministic order,
and can be dropped if the buffer overflows. Useful for debugging, never for output.

---

## 4.6 Thread block clusters (Hopper+)

New level in the hierarchy, cc 9.0 and later. A **cluster** is a group of up to 8 (portably) or 16
(non-portably) blocks guaranteed to be co-scheduled on the same GPC. Within a cluster:

- Blocks can **read and write each other's shared memory**: distributed shared memory (DSMEM):
over a fast intra-GPC network, much cheaper than a round trip to L2.
- `cluster.sync()` synchronizes all blocks in the cluster.

```cpp
// sketch
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2, 1, 1) k(float* out) {
    cg::cluster_group cluster = cg::this_cluster();
    __shared__ float smem[256];
    smem[threadIdx.x] = threadIdx.x;
    cluster.sync();
    // read block 1's shared memory from block 0:
    float* peer = cluster.map_shared_rank(smem, 1);
    out[threadIdx.x] = peer[threadIdx.x];
}
```

**Why it exists:** shared memory per SM stopped growing as fast as the tiles people want. Clusters
let a group of SMs pool their shared memory into a larger effective tile: critical for big GEMM and
attention tiles on Hopper/Blackwell. It's also the first mechanism for cheap inter-block
cooperation, which previously meant going through L2 or global memory.

You do not need this for Parts II–IV. It matters in Chapter 17 and 22.
Reference: [Programming Guide: thread block clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters).

---

## 4.7 The host side, minimally

```cpp
float *d_a;
cudaMalloc(&d_a, n * sizeof(float));                                  // allocate on device
cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);      // copy in
kernel<<<grid, block>>>(d_a, n);                                      // launch (ASYNCHRONOUS)
cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);      // copy out (SYNC, so it waits)
cudaFree(d_a);
```

⚠️ **Kernel launches are asynchronous.** `kernel<<<>>>()` returns immediately, before the kernel runs.
Two consequences that catch everyone:

1. **Timing with host clocks is wrong** unless you synchronize first. `clock()` around a launch
   measures launch overhead (~5 µs), not the kernel. Chapter 5 and 7 fix this.
2. **Errors are reported late.** A launch failure or an in-kernel fault surfaces at the *next*
   synchronizing call, which may be a completely unrelated `cudaMemcpy` fifty lines later. This is
   why the error-checking macro in Chapter 5 is not optional.

Also: `cudaMemcpy` (the synchronous one) implicitly synchronizes, which is why beginner code appears
to work without explicit syncs: and why it breaks the moment you switch to `cudaMemcpyAsync`.

**Unified memory** is the shortcut:

```cpp
float* a;
cudaMallocManaged(&a, n * sizeof(float));   // one pointer, valid on host and device
kernel<<<grid, block>>>(a, n);
cudaDeviceSynchronize();                     // must sync before host reads
printf("%f\n", a[0]);
```

Pages migrate on demand via hardware page faults. Excellent for prototyping and for oversubscribing
GPU memory. **Slower than explicit copies** when you know the access pattern, because faults are
expensive: use `cudaMemPrefetchAsync` to help it. Fine for learning; explicit for production.

---

## 4.8 SIMT vs SIMD, precisely

Worth being able to state crisply, because it's a common question.

**SIMD (CPU, AVX-512):** one instruction stream, one program counter, operating on a wide vector
register. *You* write vector code: `_mm512_add_ps`. Divergence is handled by you, manually, with
mask registers. The vector width is in your source code.

**SIMT (GPU):** you write scalar, per-thread code with ordinary `if`s and loops. The hardware groups
32 threads into a warp and issues one instruction to all of them, tracking an active mask
automatically. Divergence is handled by hardware. The warp width is *not* in your source code.

Same hardware idea; radically different programming experience. The trade:

- SIMT is far easier to write and lets one binary run on different widths.
- SIMT **hides** the cost of divergence, which means you can write code with a 32× slowdown and no
  compiler warning. SIMD makes the cost obvious because you'd have to write the masking yourself.

So the practical skill is: write scalar code, but *think* in warps. When you write

```cpp
if (data[i] > 0) expensive_a(); else expensive_b();
```

think "if this predicate splits any warp, that warp pays for both branches." Chapter 10.

---

## 4.9 Summary

- Grid → (cluster) → block → warp → thread. Blocks are the unit of scalability; a block lives on one
  SM for its whole life.
- `int i = blockIdx.x * blockDim.x + threadIdx.x;` plus `if (i < n)`. Always the guard.
- `threadIdx.x` is innermost and must map to the contiguous memory dimension.
- Block size: start at 256; always a multiple of 32.
- `__syncthreads()` synchronizes a *block*; all threads must reach it; you usually need two per
  tiling iteration.
- Not guaranteed: block order, block co-residency, warp order, FP associativity, memory ordering
  without fences. Most bugs live here.
- Kernel launches are async → time with events, check errors explicitly.
- Clusters (Hopper+) add cheap inter-block cooperation and pooled shared memory.

---

## Exercises

**4.1** Write the index arithmetic for a kernel over a 3D tensor `[D, H, W]` (row-major, `W`
contiguous) using a 3D block. Which of `x/y/z` maps to `W`? Now do it with a 1D grid and 1D block
using a flat index, recovering `d, h, w` by division and modulo. Which version coalesces better, and
why is the flat version usually preferred in practice?

**4.2** This kernel hangs. Explain exactly why, and fix it without changing what it computes.
```cpp
__global__ void k(float* x) {
    __shared__ float s[256];
    s[threadIdx.x] = x[threadIdx.x];
    if (threadIdx.x % 2 == 0) { __syncthreads(); s[threadIdx.x] *= 2.0f; }
    x[threadIdx.x] = s[threadIdx.x];
}
```

**4.3** This gives wrong answers ~1% of the time on some block sizes. Find the missing barrier.
```cpp
for (int t = 0; t < T; ++t) {
    tile[threadIdx.x] = in[t * blockDim.x + threadIdx.x];
    __syncthreads();
    acc += tile[blockDim.x - 1 - threadIdx.x];
}
```

**4.4** You need a grid-wide barrier between two phases. Give three implementations and state the
cost and the correctness precondition of each.

**4.5** Explain why launching 100 threads per block is wasteful, and compute exactly how many lanes
are idle per block.

**4.6** A colleague reports that their reduction returns `1000000.0` on CPU and `999999.94` on GPU and
files a bug. Write the reply.

---

## 🎯 Interview questions

<details>
<summary><b>Q1. What's the difference between a block and a warp, and why does the distinction matter?</b></summary>

A block is a *software* grouping you choose: up to 1024 threads that get placed on one SM, share
that SM's shared memory, and can synchronize with `__syncthreads()`. A warp is a *hardware*
grouping you don't control: exactly 32 threads that share instruction issue and execute in lockstep
under an active mask.

Why it matters: they're the granularity of different things. The block is the granularity of shared
memory allocation, of `__syncthreads()`, of occupancy accounting, and of scalability (blocks are
independent, which is why one binary scales across GPU sizes). The warp is the granularity of
divergence cost, of memory coalescing, and of the `_sync` intrinsics.

Concretely: a 256-thread block is 8 warps on one SM. An `if` that splits threads 0–15 from 16–31
diverges *within* warp 0 and costs both paths; an `if` on `threadIdx.x < 128` splits at a warp
boundary and costs nothing, because warps 0–3 take one path and warps 4–7 take the other with no
warp internally divided. Same source construct, completely different cost: and you can only see
that if you're thinking in warps.
</details>

<details>
<summary><b>Q2. Why is there no grid-wide barrier inside a kernel by default?</b></summary>

Because blocks are not guaranteed to be co-resident. The scheduler runs as many blocks as fit and
retires them to make room for more, so a grid of a million blocks on a 40-SM GPU executes in waves.
If block 0 waited for block 999,999, it would occupy an SM slot that block 999,999 needs in order to
ever start: deadlock, not a race.

That independence is deliberate: it's exactly what lets the same binary scale from a 20-SM laptop GPU
to a 148-SM B200 with no code change. The block is the unit of scalability precisely because blocks
don't talk to each other.

The escape hatches: end the kernel and launch another (the normal answer: a kernel boundary is a
grid-wide barrier costing a few microseconds); use a **cooperative launch**
(`cudaLaunchCooperativeKernel`), whose launch is atomic (it either places every block of the grid
on the device or fails outright, instead of running it in waves) which is what makes `grid.sync()`
safe, with you sizing the grid via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; or restructure to avoid the barrier
altogether, as single-pass scan does with decoupled lookback.
</details>

<details>
<summary><b>Q3. When do you need `__syncthreads()`, and what happens if a thread skips it?</b></summary>

You need it whenever a thread reads shared (or global) memory written by a *different* thread in the
same block: once after the writes and before the reads. In a tiling loop you need two: one after
loading the tile, and one after consuming it, before the next iteration overwrites it. Dropping the
second one is a classic bug: fast warps race ahead and overwrite the tile while slow warps are still
reading it, producing wrong results intermittently and block-size-dependently.

If a thread skips the barrier (because it's inside a divergent branch that thread didn't take) the
behavior is undefined. In practice you get a hang (the arrived threads wait forever for a thread
that will never come) or silent corruption. The rule is that `__syncthreads()` must be reached by
*every* thread in the block, so it must never sit inside data-dependent control flow that only some
threads enter. Hoist it out.

Worth adding: it is a barrier *and* a memory fence for the block, so it's what makes the shared-memory
writes visible, not just what makes the threads wait.
</details>

<details>
<summary><b>Q4. SIMT vs SIMD?</b></summary>

Both execute one instruction across many data elements. The difference is where the vectorization
lives.

With SIMD you write the vector explicitly: 512-bit intrinsics, a fixed width baked into your source,
and you handle divergence yourself with mask registers. With SIMT you write scalar per-thread code
with ordinary control flow, and the hardware bundles 32 threads into a warp, issues one instruction
to all of them, and maintains an active mask automatically to handle divergence.

The trade-off is the interesting part: SIMT is much easier to write and portable across widths, but
it *hides* the cost. A branch that splits a warp silently costs both paths (up to 32×) with no
compiler warning, whereas in SIMD you would have had to write the masking and would have seen the
cost. So the skill in CUDA is writing scalar code while reasoning in warps.

Since Volta, SIMT is also weaker than strict lockstep: each lane has its own program counter and can
make independent forward progress, which is why the warp intrinsics all take an explicit mask now.
</details>

<details>
<summary><b>Q5. Why doesn't my GPU reduction match my CPU loop bit-for-bit?</b></summary>

Floating-point addition isn't associative, and a parallel reduction necessarily sums in a different
order than a sequential loop. `(a+b)+c ≠ a+(b+c)` in IEEE-754 because each addition rounds. So the
GPU result differs by a few ULP: and it's often the *more* accurate one, since a tree reduction has
`O(log n)` error growth versus `O(n)` for a naive sequential loop.

If the kernel uses `atomicAdd` for the final combination, results can also differ **between runs of
the same binary**, because atomic ordering is nondeterministic. That's usually the more alarming
symptom and worth calling out explicitly.

Fixes, if you actually need reproducibility: use a fixed-order deterministic tree reduction (no
atomics) so the order is a function of the launch config only; use Kahan or pairwise compensated
summation to cut the error; accumulate in higher precision than the inputs (FP32 accumulate for FP16
data: which is exactly why tensor cores accumulate in FP32); or, if bitwise determinism is a hard
requirement, use a fixed-point or integer accumulator.
</details>

---

**Next:** [Chapter 5: The Toolchain](05-toolchain.md)

**Sources:**
[Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) ·
[Writing CUDA Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html) ·
[Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html) ·
[CUDA C++ memory model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html) ·
[Thread block clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)
