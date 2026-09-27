# Chapter 5 · The Toolchain

> **Goal:** compile deliberately rather than by copying flags; catch every error the moment it
> happens; and time kernels in a way that isn't a lie. Three unglamorous skills that determine
> whether the next twenty chapters teach you anything.

---

## 5.1 What `nvcc` actually does

`nvcc` is not a compiler. It's a driver that splits your `.cu` file in two and hands each half to a
different compiler.

```
        yourfile.cu
             │
      ┌──────┴──────┐  nvcc splits on __global__/__device__/<<<>>>
      │             │
  host C++      device code
      │             │
   g++/clang     cicc  (NVIDIA's frontend, LLVM-based)
      │             │
      │           PTX   ← virtual ISA, forward-compatible assembly-ish text
      │             │
      │          ptxas  ← the real optimizing assembler
      │             │
      │           SASS  ← actual machine code for one specific architecture
      │             │
      │          cubin
      └──────┬──────┘
       fatbinary embedded in the host object
             │
        linker → executable
```

Two levels of "assembly" is the part that confuses people:

- **PTX** (Parallel Thread eXecution) is a *virtual* ISA. It's stable across architectures, and it's
  what gets JIT-compiled by the driver if no matching binary is present. Think of it as portable
  bytecode. [PTX ISA reference](https://docs.nvidia.com/cuda/parallel-thread-execution/).
- **SASS** is the *real* per-architecture machine code. It's what actually executes. Register
  allocation, instruction scheduling, and most optimization happen in `ptxas`, going PTX → SASS.

⚠️ **Consequence that matters:** reading PTX tells you what the frontend produced, not what runs.
Register counts, scheduling, and predication are decided later. If you're doing serious optimization
you read SASS (`cuobjdump -sass`), not PTX. People waste days optimizing PTX that `ptxas` was going
to fix anyway.

---

## 5.2 Architecture flags: virtual vs real

The single most misunderstood part of the CUDA build.

```bash
nvcc -arch=sm_75 x.cu -o x        # shorthand, most common
# CUDA 13 nvcc docs: -arch=sm_75  ==  -arch=compute_75 -code=sm_75,compute_75
# i.e. SASS for Turing PLUS PTX as a fallback. This is what you want for Colab.
```

- `compute_XX` = **virtual** architecture → generate PTX for this capability level.
- `sm_XX` = **real** architecture → generate SASS/cubin for this exact chip.

Four cases:

```bash
# 1. SASS only for Turing. Fastest to build, smallest binary. FAILS on anything else.
-gencode arch=compute_75,code=sm_75

# 2. PTX only. Runs on cc ≥ 7.5 forever via driver JIT, but pays JIT cost on first launch.
-gencode arch=compute_75,code=compute_75

# 3. Both: SASS for the GPUs you care about + PTX as a forward-compatible fallback.
#    This is what you ship.
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_90,code=sm_90 \
-gencode arch=compute_90,code=compute_90

# 4. Let nvcc pick for the GPU in this machine (CUDA 11.5+). Emits cubin only, no PTX -
#    great for local dev, not for shipping.
-arch=native
```

**Compute capability is not a version number you can round.** Cubin (SASS) compatibility:

- Same major, **equal or higher** minor: usually runs. `sm_80` cubin runs on `sm_86` / `sm_89`.
- Same major, **lower** minor: no. `sm_86` cubin does **not** run on `sm_80`.
- Different major: no. `sm_75` cubin does not run on `sm_80`; `sm_90` cubin does not run on `sm_100`.

Across majors you need embedded PTX (the driver JIT-compiles it) or a rebuild. That is why
`-arch=sm_75` embedding `compute_75` PTX is the right Colab default: an A100 can JIT it.

From Hopper (cc 9.0) on, there are extra suffixes:
- `sm_90` / `compute_90`: portable baseline, forward-compatible via PTX.
- `sm_90a`: architecture-specific (TMA, `wgmma`, …). Runs **only** on that exact cc.
- `sm_100f`: *family* target: the features common to cc 10.0 and 10.3. See
  [compute-capability appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
  and the [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/).

Also: `-arch=all` embeds cubin for every supported `sm_*` plus PTX for the newest major;
`-arch=all-major` does one cubin per major (`sm_80`, `sm_90`, `sm_100`, …) plus PTX. Useful for
shipping; slow to compile.

Also note CUDA 13.x **dropped** Maxwell, Pascal, and Volta. Supported: Turing (7.5) through Blackwell.
If you find a tutorial using `sm_35`, it won't build.

**For this book on Colab: `-arch=sm_75`.**

> ⚠️ **CUDA 13 breaking change you will hit immediately.** CUDA 13.0 **removed**
> `cudaDeviceProp::clockRate` and `cudaDeviceProp::memoryClockRate`. Nearly every `device_query`
> tutorial online still uses them, so nearly every one of them now fails to compile with
> `error: class "cudaDeviceProp" has no member "memoryClockRate"`. The replacement is
> `cudaDeviceGetAttribute`:
>
> ```cpp
> int mem_clock_khz = 0, bus_bits = 0;
> cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
> cudaDeviceGetAttribute(&bus_bits,      cudaDevAttrGlobalMemoryBusWidth, dev);
> double gbps = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
> ```
>
> This book's code uses the attribute API throughout (see the `dev_attr()` helper in
> [`code/common/cuda_utils.cuh`](https://github.com/ishwar6/ishwar-books/blob/main/code/gpu/common/cuda_utils.cuh)). Good general lesson: when a CUDA sample from a blog post doesn't
> compile, check the [release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/) for
> removals before assuming you did something wrong.

### The flags you should always pass

```bash
nvcc -O3 -arch=sm_75 -lineinfo -Xptxas -v prog.cu -o prog
```

| Flag | Why |
|---|---|
| `-O3` | host optimization. Device code is optimized by default at `-O3`-equivalent; `-G` disables it. |
| `-arch=sm_XX` | CUDA 13's default *is* `sm_75` (T4), which happens to be right for Colab. On any other GPU, set it. |
| `-lineinfo` | maps SASS back to source lines so the profiler can show you *which line* stalls. Negligible cost. **Always on.** |
| `-Xptxas -v` | prints registers/thread, shared memory/block, and **spills**. Read this every build. |
| `-use_fast_math` | ⚠️ opt-in. Enables `--ffast-math`-style reassociation, flushes denormals, and swaps `sin/exp/div` for lower-precision intrinsics. Big speedups, real accuracy loss. Never set it without checking your tolerance. |
| `-G` | full device debug info for `cuda-gdb`. **Disables optimization**: a `-G` build can be 10× slower. Never benchmark with it. |
| `-Xptxas -dlcm=ca` / `-dlcm=cg` | force L1 caching on / off for global loads. Occasionally worth trying. |
| `--maxrregcount=N` | cap registers per thread. Blunt instrument; prefer `__launch_bounds__` (Ch. 11). |
| `-std=c++17` | you'll want it. `-std=c++20` works in CUDA 12+. |

### Reading `-Xptxas -v` output

This is the highest-value 30 seconds in a build. Example:

```
ptxas info    : Compiling entry function '_Z6sgemmPfS_S_iii' for 'sm_75'
ptxas info    : Function properties for _Z6sgemmPfS_S_iii
    24 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 62 registers, 8192 bytes smem, 388 bytes cmem[0]
```

What to look at, in order:

1. **`spill stores` / `spill loads` ≠ 0** → the compiler ran out of registers and pushed variables to
   *local memory*, which is DRAM (Ch. 3). This is usually a 2–10× regression. Fix it: reduce the
   per-thread tile, or add `__launch_bounds__`, or find the dynamically-indexed local array causing
   it.
2. **`Used N registers`** → `65536 / N` is the *thread* ceiling before the architecture cap, ignoring
   allocation granularity (registers are handed out per warp, rounded; Ch. 11). 62 registers on a
   T4 → 65536/62 ≈ 1057, capped at 1024 → full occupancy is still available. 62 on an H100 →
   ~1056 of 2048 → ~51% max occupancy. If `N` is just under a rounding boundary (63 vs 64) the
   real number can drop a full warp; treat this as a first cut.
3. **`bytes smem`** → check it against the per-SM budget to see how many blocks fit. 8192 B/block on
   a T4 (64 KB/SM) → 8 blocks; not the limiter.
4. **`stack frame` > 0** → often benign (ABI), but a large one means local memory use.

---

## 5.3 Error handling that actually works

Every CUDA API call returns a `cudaError_t`. Kernel launches don't return anything. Both can fail,
and by default **nothing tells you**.

Worse: because launches are asynchronous, an in-kernel fault (bad address, misaligned access) is
reported at whatever *later* synchronizing call happens to run. You get
`cudaErrorIllegalAddress` from an innocent `cudaMemcpy` fifty lines away from the real bug.

Use this. Put it in a header, include it everywhere, no exceptions:

```cpp
// code/common/cuda_utils.cuh
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s (%s)\n", __FILE__, __LINE__,    \
                    cudaGetErrorName(err_), cudaGetErrorString(err_));            \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Call after a kernel launch. Catches BOTH launch-config errors (sync) and
// execution errors (async) -- the sync is what makes the second kind visible.
#define CUDA_CHECK_KERNEL()                                                       \
    do {                                                                          \
        CUDA_CHECK(cudaGetLastError());       /* launch-time errors */            \
        CUDA_CHECK(cudaDeviceSynchronize());  /* execution errors  */             \
    } while (0)
```

Usage:

```cpp
CUDA_CHECK(cudaMalloc(&d_a, bytes));
kernel<<<grid, block>>>(d_a, n);
CUDA_CHECK_KERNEL();
```

⚠️ `CUDA_CHECK_KERNEL()` contains a `cudaDeviceSynchronize()`, which **serializes**. Use it during
development and in debug builds; guard it out of release builds and out of timing loops, or your
benchmark measures nothing but syncs. A common pattern:

```cpp
#ifdef NDEBUG
  #define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#else
  #define CUDA_CHECK_KERNEL() do {                                 \
      CUDA_CHECK(cudaGetLastError());                              \
      CUDA_CHECK(cudaDeviceSynchronize());                         \
  } while (0)
#endif
```

### The errors you'll actually hit, and what they mean

| Error | Real cause, 90% of the time |
|---|---|
| `cudaErrorIllegalAddress` | out-of-bounds global access. You forgot `if (i < n)`. |
| `cudaErrorMisalignedAddress` | casting `float*` to `float4*` at an offset not divisible by 16 (Ch. 8). |
| `cudaErrorInvalidConfiguration` | > 1024 threads per block, or requesting more shared memory than the SM has |
| `cudaErrorLaunchOutOfResources` | too many registers × threads for the SM. Reduce block size or registers. |
| `cudaErrorNoKernelImageForDevice` | compiled for the wrong `sm_XX`. |
| `cudaErrorLaunchTimeout` | kernel exceeded the watchdog (only on display-attached GPUs; irrelevant on Colab) |
| silent wrong answers | missing `__syncthreads()`, or a race. No error will ever be raised. |

That last row is why the rest of this book emphasizes verifying against a CPU reference **every
time**. The GPU will not tell you that you have a race.

### `compute-sanitizer`: use it

The single most useful debugging tool in CUDA, and most people don't know it exists. It's the
CUDA equivalent of Valgrind + ThreadSanitizer, ships with the toolkit, and needs no rebuild:

```bash
compute-sanitizer ./prog                       # memcheck: OOB and misaligned accesses
compute-sanitizer --tool racecheck ./prog      # shared-memory data races  ← the missing-barrier bug
compute-sanitizer --tool synccheck ./prog      # illegal __syncthreads() usage
compute-sanitizer --tool initcheck ./prog      # reads of uninitialized device memory
```

`racecheck` finds the missing-`__syncthreads()` class of bug that produces intermittently wrong
answers. When a kernel is "wrong 1% of the time," run racecheck before you do anything else.
It's slow (10–100×): run it on a small input.
[Docs](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

---

## 5.4 Timing kernels correctly

**Wrong:**

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
kernel<<<g,b>>>(...);
auto t1 = std::chrono::high_resolution_clock::now();   // ⚠️ measures ~5 µs of launch overhead
```

The launch is asynchronous. You timed the enqueue.

**Also wrong:**

```cpp
auto t0 = clock();
kernel<<<g,b>>>(...);
cudaDeviceSynchronize();
auto t1 = clock();     // better, but includes sync overhead and host jitter; no warmup
```

**Right: CUDA events**, which are timestamps recorded *in the stream* by the GPU itself:

```cpp
cudaEvent_t start, stop;
CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));

// 1. WARM UP. The first launch pays context creation, module load, and possibly PTX JIT.
for (int i = 0; i < 5; ++i) kernel<<<g,b>>>(args);
CUDA_CHECK(cudaDeviceSynchronize());

// 2. Time many iterations, not one.
const int iters = 100;
CUDA_CHECK(cudaEventRecord(start));
for (int i = 0; i < iters; ++i) kernel<<<g,b>>>(args);
CUDA_CHECK(cudaEventRecord(stop));
CUDA_CHECK(cudaEventSynchronize(stop));

float ms;
CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));   // resolution ~0.5 µs
printf("%.4f ms per iteration\n", ms / iters);
```

The five rules of GPU benchmarking:

1. **Warm up.** First launch includes context init and JIT: often 10–100× the steady-state time.
2. **Repeat.** Single measurements have 10%+ variance. 100 iterations, report the mean; report the
   *minimum* if you want the "clean run" number and say which you did.
3. **Sync before reading the timer.** `cudaEventSynchronize(stop)`, not just `cudaEventRecord`.
4. **Watch for clock throttling.** GPUs downclock under sustained load. If iteration 100 is 20%
   slower than iteration 5, you're thermally/power limited. `nvidia-smi -q -d CLOCK` and
   `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv -l 1` show it. For
   stable numbers you can lock clocks with `nvidia-smi -lgc <freq>` (needs privileges; not available
   on Colab).
5. **Don't let the compiler delete your kernel.** If the result is never read, `ptxas` can eliminate
   the work. Always copy something back, or write to a `volatile` sink.

⚠️ **The back-to-back launch trap.** The loop above launches 100 kernels into the same stream, so
they serialize: good, that's what you want. But it also *hides launch overhead* by pipelining: the
host enqueues launch `i+1` while `i` runs. For kernels shorter than ~10 µs this makes them look
faster than they'd be in isolation. If launch overhead is what you care about (Ch. 13), measure a
single launch with a sync on each side, and compare.

Convert time to the metric that matters (Ch. 7):

```cpp
double sec   = (ms / iters) / 1000.0;
double gbps  = total_bytes_moved / sec / 1e9;
double gflops = total_flops / sec / 1e9;
```

---

## 5.5 Know your device: `device_query`

Never optimize for a GPU whose limits you haven't printed. [`code/ch05/device_query.cu`](https://github.com/ishwar6/ishwar-books/blob/main/code/gpu/ch05/device_query.cu) prints
everything from the Chapter 3 tables for whatever GPU you're on, plus the derived quantities that
actually matter (peak bandwidth, machine balance, max resident threads).

```bash
nvcc -O3 -arch=sm_75 ch05/device_query.cu -o bin/device_query && ./bin/device_query
```

Abbreviated output on a Colab T4:

```
Device 0: Tesla T4
  Compute capability            : 7.5
  SMs                           : 40
  Max threads / SM              : 1024
  Max threads / block           : 1024
  Warp size                     : 32
  Registers / SM                : 65536
  Shared memory / SM            : 65536 bytes (64 KB)
  Shared memory / block (opt-in): 65536 bytes
  L2 cache                      : 4194304 bytes (4 MB)
  Global memory                 : 15360 MB
  Memory clock                  : 5001 MHz
  Memory bus width              : 256 bits

-- DERIVED (the numbers that matter) -----------------------
  peak bandwidth                : 320.1 GB/s
      = 2 (DDR) x 5001 MHz x 256 bits / 8
  max resident threads          : 40960
  max resident warps            : 1280
  registers/thread @ full occ.  : 64   <-- your register budget
  shared mem/thread @ full occ. : 64.0 B
  L2 bytes / resident thread    : 102.4 B   <-- why caches don't save you
  FP32 cores / SM               : 64
  peak FP32 (CUDA cores)        : 8.14 TFLOP/s
  MACHINE BALANCE               : 25.4 FLOP/byte
      ^ arithmetic intensity above this = compute bound,
        below it = memory bound. (FP32 CUDA cores only;
        the tensor-core roof is 10-100x higher.)
```

Three of those derived lines are the ones you'll keep coming back to:
**registers/thread at full occupancy** (your budget in Ch. 11), **peak bandwidth** (the denominator
of every memory-bound claim), and **machine balance** (which side of the roofline you're on, Ch. 2).

Note `peak FP32` is computed from the core count and the *boost* clock, so it's the theoretical
ceiling: Chapter 7 measures what you can actually reach (~95–98% of it), and that measured number is
what you should use.

Peak bandwidth from the clock and bus width:

```
BW = memoryClockRate(kHz) × 1000 × (busWidth/8) × 2   // ×2 for DDR
   = 5.001e9 Hz × 32 bytes × 2 = 320.06e9 B/s = 320.06 GB/s
```

Write the output down for your GPU. Every "% of peak" claim in this book is relative to your numbers.

---

## 5.6 Inspecting the generated code

Occasionally necessary, and knowing how marks you as someone who's done this for real.

```bash
# PTX (virtual ISA, human-readable-ish)
nvcc -arch=sm_75 -ptx kernel.cu -o kernel.ptx

# SASS (actual machine code) -- what really runs
nvcc -arch=sm_75 -cubin kernel.cu -o kernel.cubin
cuobjdump -sass kernel.cubin

# or in one step from the binary
cuobjdump -sass ./prog | less

# resource usage per kernel, without recompiling
cuobjdump -res-usage ./prog

# richer disassembly with control flow
nvdisasm -c kernel.cubin
```

What to look for in SASS, practically:

| Pattern | Meaning |
|---|---|
| `LDG.E.128` | 128-bit (16-byte) vectorized global load. **You want to see this** (Ch. 8). |
| `LDG.E` / `LDG.E.64` | 32- and 64-bit loads. Fine, but check if you could vectorize. |
| `LDS` / `STS` | shared memory load/store. |
| `LDL` / `STL` | ⚠️ **local** memory load/store: a register spill. Something to fix. |
| `HFMA2` / `FFMA` | fused multiply-add. In a GEMM inner loop you want these to dominate. |
| `HMMA` / `IMMA` | tensor-core matrix instruction (Ch. 17). |
| `BAR.SYNC` | `__syncthreads()`. Count them; each one costs. |
| `@!P0 BRA` | predicated branch: the compiler handled a small `if` without divergence. Good. |

The single most useful use of `cuobjdump`: after a GEMM optimization, check that your inner loop is
mostly `FFMA` with `LDS` and not a pile of `IADD3`/`SHF` address arithmetic. If half your
instructions are computing addresses, that's your bottleneck (Ch. 16).

`godbolt.org` supports `nvcc` and shows PTX/SASS side by side with your source: the fastest way to
answer "does the compiler vectorize this?" without a GPU. Works on a Mac.

---

## 5.7 The rest of the toolbox

| Tool | Use it for |
|---|---|
| `nvidia-smi` | is there a GPU, what is it, is someone else using it, clocks/temp/power |
| `nsys` (Nsight Systems) | **timeline**: where does wall-clock go across CPU, kernels, memcpys, streams. Start here. |
| `ncu` (Nsight Compute) | **one kernel, deeply**: counters, roofline, stall reasons, source-level attribution |
| `compute-sanitizer` | correctness: OOB, races, sync errors, uninitialized reads |
| `cuda-gdb` | step through device code. Needs `-G`. Rarely worth it vs. `printf` + sanitizer. |
| `cuobjdump` / `nvdisasm` | see the SASS |
| `nvcc --dryrun` | print every sub-command nvcc would run. Great for understanding the pipeline. |

Chapter 18 is a full treatment of `nsys` and `ncu`. The rule for now: **`nsys` to find *which* kernel
matters, `ncu` to find *why* that kernel is slow.** Doing it in the other order is how people spend a
week optimizing a kernel that was 2% of runtime.

---

## 5.8 Summary

- `nvcc` splits host/device; device goes `.cu` → PTX (virtual, portable) → SASS (real, per-arch).
  Optimization happens in `ptxas`, so **read SASS, not PTX**.
- `compute_XX` = PTX, `sm_XX` = SASS. Ship both. Use `-arch=sm_75` on Colab.
- Always: `-O3 -arch=sm_XX -lineinfo -Xptxas -v`. Read the spill numbers every build.
- Wrap every API call in `CUDA_CHECK`; after every launch, `cudaGetLastError()` + a sync in debug.
  Async errors surface late and blame the wrong line.
- `compute-sanitizer --tool racecheck` finds missing-barrier bugs. Use it before guessing.
- Time with **CUDA events**, after a **warmup**, over **many iterations**, and make sure the result is
  consumed.
- Run `device_query` before optimizing anything. Write down your peak bandwidth and
  registers-per-thread-at-full-occupancy.

---

## Exercises

**5.1** Compile [`code/ch05/device_query.cu`](https://github.com/ishwar6/ishwar-books/blob/main/code/gpu/ch05/device_query.cu) on Colab and record: SMs, max threads/SM, shared
memory/SM, peak bandwidth, machine balance (peak FP32 ÷ peak BW). Keep this file open for the rest
of the book.

**5.2** Compile any kernel with `-Xptxas -v`. Now add a `float scratch[64]` local array indexed by a
runtime variable. Recompile. What changed in the report? Explain in terms of Chapter 3.

**5.3** Write a kernel whose result is never read by the host. Compile at `-O3` and inspect
`cuobjdump -sass`. Is the work still there? Now write the result to a `volatile` global. Compare.

**5.4** Three parts, all runnable on a Colab T4 (cc 7.5):
(a) Compile a working kernel with `-gencode arch=compute_80,code=sm_80` (SASS for Ampere only, no
PTX) and run it on the T4. What error, and why can the driver not fall back?
(b) Now compile with only `-gencode arch=compute_75,code=compute_75` (PTX only, no SASS) and time
the *first* launch against the second. Where did the extra time go?
(c) Compile with `-gencode arch=compute_75,code=[sm_75,compute_75]` and confirm the first-launch
penalty is gone. Explain what each of the three binaries contains.
*(Note the direction of compatibility: you cannot run an `sm_80` cubin on `sm_75` hardware. The
forward-compatibility rule only goes up within a major family: an `sm_80` cubin runs on `sm_86`,
not the reverse.)*

**5.5** Time a trivial kernel three ways: (a) `chrono` with no sync, (b) `chrono` with
`cudaDeviceSynchronize`, (c) CUDA events with warmup and 100 iterations. Report all three. The spread
is the reason for this section.

**5.6** Write a kernel with a deliberate off-by-one out-of-bounds write. Run it normally: does it
crash? Where does the error surface? Now run it under `compute-sanitizer`. Compare the diagnostics.

**5.7** Compile a kernel that reads `float4` from a pointer offset by one `float`. Predict the error
before running it. Verify.

---

## 🎯 Interview questions

<details>
<summary><b>Q1. What's the difference between PTX and SASS?</b></summary>

PTX is a virtual instruction set: a stable, forward-compatible intermediate representation that
`nvcc` emits for a *virtual* architecture (`compute_75`). SASS is the real machine ISA for a
*specific* chip (`sm_75`), produced from PTX by `ptxas`, and it's what actually executes.

The practical implications are what the question is really after. First, a binary can embed SASS for
several architectures plus PTX as a fallback; if you run on a GPU with no matching SASS, the driver
JIT-compiles the PTX at first launch, which costs real time on startup. Second, and more important
for performance work: **register allocation and instruction scheduling happen in PTX → SASS**, so
PTX tells you what the frontend produced, not what runs. If you want to know whether your loads got
vectorized or whether you're spilling, you read SASS via `cuobjdump -sass`: looking at PTX
register counts will mislead you.
</details>

<details>
<summary><b>Q2. Your kernel produces the right answer on 256 threads per block and the wrong answer on 512. What's your first hypothesis?</b></summary>

A missing or misplaced `__syncthreads()`: a shared-memory race whose visibility depends on how
warps happen to interleave, and therefore on the block size.

The reasoning: at 256 threads (8 warps) the scheduler may happen to keep warps close enough together
that the race doesn't manifest; at 512 (16 warps) the spread widens and a fast warp overwrites data
a slow warp is still reading. The classic instance is a tiling loop with a barrier after the load
but none after the consume, so iteration *t+1*'s writes clobber iteration *t*'s reads.

I'd confirm it in one command rather than by reading code: `compute-sanitizer --tool racecheck` on a
small input will point at the exact shared-memory access pair. Second hypothesis if that comes back
clean: a barrier inside divergent control flow that not all threads reach, which `--tool synccheck`
catches. Third: an assumption of warp-synchronous behavior without `__syncwarp()`, which is
undefined on Volta and later.
</details>

<details>
<summary><b>Q3. How do you time a kernel?</b></summary>

CUDA events, because kernel launches are asynchronous and any host-side clock around a bare launch
measures only the ~5 µs enqueue. `cudaEventRecord` inserts a timestamp into the stream that the GPU
itself fills in, so it measures device time.

The full recipe matters more than the API: warm up first (the first launch pays context creation,
module load, and possibly PTX JIT: often 10–100× steady state); loop many iterations and divide,
because single-shot variance is >10%; `cudaEventSynchronize(stop)` before reading the elapsed time;
make sure the result is consumed so `ptxas` doesn't delete the work; and watch for clock throttling
by checking whether late iterations are slower than early ones.

One subtlety worth volunteering: back-to-back launches in a loop pipeline with each other, so launch
overhead is hidden. For kernels under ~10 µs that makes the per-iteration number optimistic relative
to running the kernel in isolation, and if launch overhead is the thing you care about you have to
measure a single launch bracketed by syncs.
</details>

<details>
<summary><b>Q4. `-Xptxas -v` reports "48 bytes spill stores". Should you care?</b></summary>

Yes, almost always. Spills mean the compiler ran out of registers and moved variables to *local*
memory: which despite the name is off-chip DRAM, cached in L1/L2. A spilled value that's read in an
inner loop can turn a ~1-cycle register access into hundreds of cycles, so spills in a hot loop are
routinely a 2–10× regression.

Causes, in the order I'd check: a per-thread register tile that's too large (a register-blocked GEMM
asking for 8×8 accumulators plus operands); a local array indexed by a runtime value, which forces
it to local memory whether or not registers were tight; heavy function inlining; or an aggressive
`__launch_bounds__` / `--maxrregcount` that capped registers too low.

Fixes: shrink the per-thread tile, restructure so local arrays are indexed by compile-time constants
(`#pragma unroll` on the indexing loop often does this), relax `__launch_bounds__` and accept lower
occupancy, or move the data to shared memory deliberately rather than letting it spill accidentally.

The exception where spills are acceptable: cold code: setup, epilogue, an error path. A few spill
bytes outside the hot loop are noise.
</details>

<details>
<summary><b>Q5. What does `-use_fast_math` do and when would you use it?</b></summary>

It bundles several precision-for-speed trades: it enables contraction and reassociation, denormals
are flushed to zero, `1/x` and `sqrt` become fast approximate instructions rather than
IEEE-correctly-rounded, and transcendentals like `sin`, `exp`, `log` are replaced with hardware SFU
intrinsics (`__sinf`, `__expf`) that have a few ULP of error instead of sub-ULP.

When to use it: inference workloads and graphics, where you're already in FP16/BF16 and a few ULP is
far below your noise floor, and where the transcendentals are actually hot. The speedups on
transcendental-heavy kernels are substantial.

When not to: anything with an accuracy requirement you have to defend: scientific simulation,
iterative solvers where reassociation changes convergence, or a kernel you're validating bit-wise
against a reference. It's also a blunt instrument: it applies to the whole compilation unit. The
better practice is usually to leave it off and opt in per-call-site with the explicit intrinsics
(`__expf` instead of `expf`, `__fdividef` instead of `/`), so the precision trade is visible in the
source at the exact place it's made.
</details>

---


**Next:** Part II, Chapter 6: Your First Kernels (coming soon).
**Sources:**
[nvcc documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/) ·
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) ·
[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) ·
[Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/) ·
[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
