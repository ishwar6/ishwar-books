# Chapter 1 · CPU Parallelism: What GPUs Are Reacting Against

> **Goal:** before you meet a single GPU concept, know every kind of parallelism a CPU already
> gives you (pipelining, superscalar and out-of-order issue, SIMD lanes, threads, and green
> threads / async tasks) well enough to *measure* each one on your own machine, compute a
> CPU's peak FLOP/s from first principles, and explain in one sentence which of these ideas the
> GPU keeps, which it throws away, and why.
>
> Every number in this chapter was measured on the book's reference machine, with the
> programs in `code/ch01/`. Run them on yours.

---

## 1.1 The machine, and the four kinds of CPU parallelism

The reference machine for this chapter is an **Apple M5 Pro** laptop (MacBook Pro,
`Mac17,8`, macOS 26.5.1, 64 GB). Apple's published facts: an 18-core CPU made of **6 "super"
cores and 12 "performance" cores**, up to **307 GB/s** unified-memory bandwidth, up to 64 GB
([Apple newsroom, March 2026](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)).
Apple publishes **no clock speed, no pipe counts, no reorder-buffer size**. Where we need those,
we measure them (§1.9) and say so. `sysctl` on the machine reports 128 KB L1d and 16 MB L2 for
the super-core cluster, 64 KB L1d and 8 MB L2 for each performance cluster, and a **128-byte
cache line**: worth remembering, because x86 lines are 64 bytes.

A CPU core running your code has four independent ways to do more than one thing at a time.
They multiply:

| Level | What runs in parallel | Who finds the parallelism | §  |
|---|---|---|---|
| Instruction-level (ILP) | independent instructions of one thread | the hardware (pipeline, superscalar, out-of-order) | 1.2–1.3 |
| Data-level (SIMD) | the same operation on 4 / 8 / 16 lanes | you (intrinsics) or the compiler (auto-vectorizer) | 1.4–1.6 |
| Thread-level | separate instruction streams on separate cores | you (pthreads, OpenMP) | 1.7 |
| Task-level concurrency | many logical tasks multiplexed onto few threads | a runtime (Go, Java, Tokio) | 1.8 |

The first three are **parallelism**: more work per second. The fourth is mostly
**concurrency**: more tasks *in progress* at once, which is not the same thing, and the
difference is the single most useful idea to carry into GPUs (§1.9).

---

## 1.2 One core, many instructions: pipelining and superscalar issue

**Pipelining.** An instruction goes through stages: fetch, decode, rename, schedule, execute,
write back, retire. A pipelined core works on a different instruction in each stage at once,
like a car assembly line, so a new instruction can *start* every cycle even though each one
takes many cycles to *finish*. Two numbers describe any instruction:

- **Latency**: cycles from inputs ready to result ready.
- **Throughput**: how many can start per cycle (across all the pipes that can execute it).

**Superscalar** means more than one pipe: the core can start several instructions per cycle if
they are independent. `peak_flops.c` measures both numbers for the vector FMA on the M5 Pro
(single thread, run by the OS on whichever core it chose: almost certainly a super core):

| Measurement (Apple M5 Pro, 1 thread) | Result |
|---|---|
| Dependent integer adds per second (1-cycle latency ⇒ ≈ clock) | **4.59 G/s ⇒ ~4.59 GHz** |
| Latency of vector `fadd` (4 × FP32) | **2.05 cycles** |
| Latency of vector `fmla` (fused multiply-add, 4 × FP32) | **3.07 cycles** |
| Best FMA throughput | **31.0 FLOP/cycle ⇒ ~4 FMA pipes × 4 lanes × 2** |

(The clock estimate assumes a dependent `add` takes one cycle, true of every mainstream core;
the fact that the two FP latencies then come out as near-integers is a consistency check.)

**The consequence that matters:** a loop whose every iteration depends on the previous one runs
at *one instruction per latency*, no matter how many pipes exist. The dot product
`s += x[i]*y[i]` is exactly that: each FMA needs the previous `s`. Split the sum into four
independent accumulators and the pipes can overlap them. From `simd_dot.c`, one thread,
vectorization disabled so this is purely scalar:

| Scalar dot product, n = 4096 (L1-resident) | GFLOP/s | speedup |
|---|---|---|
| 1 accumulator (`dot_scalar`) | 2.75 | 1.00× |
| 4 accumulators (`dot_scalar_ilp4`) | 10.42 | **3.79×** |

Same instructions, same count: only the *dependency structure* changed. This is
**instruction-level parallelism (ILP)**, and the rule "latency × throughput = independent
operations you need in flight" is Little's law (Chapter 2 §2.5) applied inside a single core.
The FMA sweep in `peak_flops.c` shows the same law directly: with 3-cycle latency and ~4 pipes you
need ~12–16 independent chains before the pipes are full (1 chain: 11.9 GFLOP/s; 16 chains:
142.1 GFLOP/s). **GPUs use the identical rule: Chapter 11 calls it "latency hiding = warps ×
ILP".** The CPU just has to find all of its ILP inside one thread.

---

## 1.3 Out-of-order execution: what it buys, and the bill

Real code does not come conveniently pre-split into independent chains. An **out-of-order (OoO)**
core finds the independence itself, at run time:

1. **Register renaming.** The ISA has a few dozen architectural registers (32 general + 32 vector
   on AArch64), so compilers reuse them constantly: `x3` holds one value, then an unrelated one.
   Those reuses create *false* dependencies (write-after-read, write-after-write). The renamer
   maps every write to a fresh **physical** register from a much larger pool, so only *true*
   (read-after-write) dependencies remain.
2. **Scheduling window.** Renamed instructions wait in schedulers; any whose inputs are ready can
   issue to a free pipe, in any order. A stalled load does not block the independent work
   behind it.
3. **Reorder buffer (ROB).** Every in-flight instruction holds a ROB slot, and results are
   *retired* (made architecturally visible) strictly in program order. That is what makes
   exceptions precise and branch mispredictions recoverable: everything after a mispredicted
   branch is simply discarded. Combined with **branch prediction and speculation**, the core
   runs hundreds of instructions ahead of the oldest unfinished one.

How far ahead? Intel disclosed a **512-entry ROB** (up from 352) for its Golden Cove core
([Intel Architecture Day 2021, as reported by Tom's Hardware](https://www.tomshardware.com/features/intel-architecture-day-2021-intel-unveils-alder-lake-golden-cove-and-gracemont-cores/4)).
Apple publishes nothing; sizes quoted online for Apple cores come from reverse-engineering.

**What it buys:** single-thread speed on ordinary, branchy, pointer-chasing code, without the
programmer doing anything. It is why the CPU is the right machine for most software.

**What it cannot buy.** `mem_latency.c` chases pointers through a random cycle so that no
load can start before the previous one finishes:

| Working set (Apple M5 Pro, 1 thread) | 16 KB | 1 MB | 8 MB | 64 MB | 256 MB | 1 GB |
|---|---|---|---|---|---|---|
| ns per dependent load | 0.89 | 3.04 | 7.73 | 77.6 | 104.6 | 113.0 |

113 ns at ~4.6 GHz is **~520 cycles**. A 512-entry window filled at several instructions per
cycle is exhausted in ~100 cycles. OoO hides L1 and L2 latency; it cannot hide DRAM on its own.
The CPU's answer is *more cache* and *prefetchers*.

**The bill.** Renaming, wakeup/select logic in the schedulers, the ROB, a physical register file
with many read/write ports, the branch predictors: none of it does arithmetic. The comparison
logic in the scheduler grows roughly with (window size × issue width), register-file ports with
issue width, and all of it is paid **per thread** and burns power on every instruction. That is
the transistor and energy budget Chapter 2 §2.1 describes the GPU as "deleting". A GPU SM issues
instructions **in order** within each warp; when a warp stalls, it does not look ahead in that
warp: it switches to another warp (§1.9). Latency hiding moves from *speculation within one
thread* to *switching between many threads*.

---

## 1.4 SIMD: one instruction, many lanes

A **SIMD** (single instruction, multiple data) instruction operates on a wide register holding
several values: **lanes**. One `fmla v0.4s, v1.4s, v2.4s` does four FP32 FMAs.

### x86: SSE → AVX → AVX2 → AVX-512 → AVX10

| Extension | Width | First shipped | Notes |
|---|---|---|---|
| SSE / SSE2 | 128-bit (4 × FP32) | 1999 / 2000 (Pentium III / 4) | SSE2 is baseline for every x86-64 CPU |
| AVX | 256-bit FP | 2011 (Sandy Bridge) | three-operand encoding (VEX) |
| AVX2 + FMA3 | 256-bit int + FP FMA | 2013 (Haswell) | the practical "modern baseline" (`x86-64-v3`) |
| AVX-512 | 512-bit, 32 regs, **mask registers** | 2016 (Knights Landing), 2017 (Skylake-SP) | fragmented: Intel servers yes; Intel hybrid clients (Alder Lake on) disabled it; AMD since Zen 4 (2022) |
| AVX10 | converged AVX-512 feature set | Intel's Nov 2025 ISA-extensions manual lists AVX10.2 for Nova Lake ([Phoronix](https://www.phoronix.com/news/Nova-Lake-Does-AVX10.2-APX)) | Intel's route out of the fragmentation |

The mask registers of AVX-512 are worth noticing: they let each lane be switched on or off per
instruction, which is how SIMD code handles `if`: exactly what a GPU does *for you* with its
warp active mask (Chapter 4 §4.8).

### Arm: Neon, SVE, SVE2 · and what Apple actually ships

- **Neon** (officially *Advanced SIMD*): fixed **128-bit** registers (4 × FP32), 32 of them,
  mandatory in every AArch64 CPU. This is what the M5 Pro runs.
- **SVE / SVE2** ([Arm](https://developer.arm.com/Architectures/Scalable%20Vector%20Extensions)):
  *length-agnostic*: the hardware vector length is anything from 128 to 2048 bits in 128-bit
  steps, and the **same binary** runs on all of them. Code asks the hardware "how many lanes?"
  at run time and uses **predicate registers** for loop tails and `if`s. Fujitsu's A64FX
  implements 512-bit SVE. NVIDIA's Grace (the CPU in GH200/GB200) uses Neoverse V2 cores with
  **SVE2 in a 4 × 128-bit configuration**, alongside Neon on the same four units
  ([NVIDIA Grace benchmarking guide](https://nvidia.github.io/grace-cpu-benchmarking-guide/developer/vectorization.html)).
- **Apple**: Neon yes; **ordinary SVE/SVE2 no**. Instead, M4 and later implement **SME/SME2**
  (Scalable *Matrix* Extension): a separate matrix unit entered via a "streaming mode" that
  supports only a subset of SVE instructions. On this M5 Pro, `sysctl hw.optional.arm` reports
  `FEAT_SME=1`, `FEAT_SME2=1`, `FEAT_SME2p1=1`, `sme_max_svl_b=64` (a **512-bit** streaming vector
  length), and **no `FEAT_SVE` entry at all**. Independent reverse-engineering of the M4 reports
  one SME block per CPU cluster, shared by its cores, sustaining ~2 TFLOP/s FP32 on the M4
  P-cluster ([tzakharko/m4-sme-exploration](https://github.com/tzakharko/m4-sme-exploration/blob/main/reports/01-sme-overview.md):
reported, not measured here). This chapter's code sticks to Neon.

### Intrinsics: SIMD by hand

Intrinsics are C functions that map (nearly) one-to-one onto SIMD instructions. The Neon dot
product from `simd_dot.c`:

```c
#include <arm_neon.h>
float dot_neon(const float* x, const float* y, size_t n) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;   // 4 accumulators (ILP, §1.2)
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(x + i +  0), vld1q_f32(y + i +  0));  // a0 += x*y, 4 lanes
        a1 = vfmaq_f32(a1, vld1q_f32(x + i +  4), vld1q_f32(y + i +  4));
        a2 = vfmaq_f32(a2, vld1q_f32(x + i +  8), vld1q_f32(y + i +  8));
        a3 = vfmaq_f32(a3, vld1q_f32(x + i + 12), vld1q_f32(y + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));  // horizontal add
    for (; i < n; ++i) s += x[i] * y[i];                                     // scalar tail
    return s;
}
```

The AVX2 version is the same shape with `__m256`, `_mm256_loadu_ps`, `_mm256_fmadd_ps` and a
longer horizontal sum; it lives in the same file behind `__attribute__((target("avx2,fma")))`
and a run-time `__builtin_cpu_supports("avx2")` check. Three things you had to do by hand that a
GPU kernel never mentions: pick the **width** (4 vs 8 vs 16), write the **tail**, and do the
**horizontal reduction** across lanes.

**Measured** (`simd_dot`, Apple M5 Pro, Apple clang 21.0.0 `-O3`, one thread, best of 5):

| n (resident in) | scalar | scalar-ilp4 | autovec-plain | autovec-simd | Neon intrinsics | Highway |
|---|---|---|---|---|---|---|
| 4 096 (L1) | 2.75 GFLOP/s | 10.42 (3.8×) | 4.43 (1.6×) | 42.54 (15.5×) | **43.36 (15.8×)** | 41.4 (15.7×) |
| 65 536 (L2) | 2.69 | 9.95 (3.7×) | 4.25 (1.6×) | 36.29 (13.5×) | 36.36 (13.5×) | 34.4 (13.3×) |
| 33.5 M (DRAM) | 2.68 | 9.95 (3.7×) | 4.24 (1.6×) | 29.08 (10.9×) | 29.27 (10.9×) | 27.7 (10.9×) |

All variants are first verified against a double-precision reference at n = 1, 37, 4099 and
1,000,003 (every tail path exercised). Highway's speedup is against its own scalar loop in
`hwy_dot.cc`. Two lessons hide in the table. **4 lanes gave ~16×, not 4×**: SIMD multiplied
the ILP win (four vectors in flight × four lanes). And **the speedup shrinks as data moves
off-chip**: §1.10 explains why with a roofline. The "autovec-plain" column is §1.6's story.

(x86/AVX2: the plan was to run the same program under `docker --platform linux/amd64 gcc:15`.
See `code/ch01/README.md` for whether that emulated CPU exposes AVX2; timings under emulation
say nothing about real x86 hardware and are not reported here.)

---

## 1.5 Portable SIMD: Google Highway

Intrinsics tie you to one ISA and one width. **[Highway](https://github.com/google/highway)**
(Google, Apache-2.0; tested here at v1.4.0, commit `32ae733`) is a C++ library of portable SIMD
operations that compiles one source for many targets (Neon, SVE, SVE2, SSE4, AVX2, AVX-512,
RISC-V V, WebAssembly) and **dispatches at run time** to the best target the CPU has. The core
of `hwy_dot.cc`:

```cpp
namespace hn = hwy::HWY_NAMESPACE;
float DotHwy(const float* HWY_RESTRICT x, const float* HWY_RESTRICT y, size_t n) {
    const hn::ScalableTag<float> d;       // "a full vector of floats, whatever that is here"
    const size_t N = hn::Lanes(d);        // 4 on Neon, 8 on AVX2, 16 on AVX-512; varies on SVE
    auto a0 = hn::Zero(d), a1 = hn::Zero(d), a2 = hn::Zero(d), a3 = hn::Zero(d);
    size_t i = 0;
    for (; i + 4 * N <= n; i += 4 * N) {
        a0 = hn::MulAdd(hn::LoadU(d, x + i + 0 * N), hn::LoadU(d, y + i + 0 * N), a0);
        // ... a1, a2, a3 likewise
    }
    float s = hn::ReduceSum(d, hn::Add(hn::Add(a0, a1), hn::Add(a2, a3)));
    for (; i < n; ++i) s += x[i] * y[i];
    return s;
}
```

`ScalableTag<float>` names "a full native vector of `float`"; `Lanes(d)` is a *run-time* value
(it is only `constexpr` on fixed-width targets); `LoadU` needs only element alignment while
`Load` needs full-vector alignment; `MulAdd(a, b, c)` is `a*b + c`. Dynamic dispatch is
boilerplate: define `HWY_TARGET_INCLUDE` as the file's own name and include
`hwy/foreach_target.h` (which re-includes the file once per target, each in a different
`HWY_NAMESPACE`), then outside that region `HWY_EXPORT(DotHwy)` builds a table of per-target
pointers and `HWY_DYNAMIC_DISPATCH(DotHwy)(x, y, n)` calls the best one
([quick reference](https://github.com/google/highway/blob/master/g3doc/quick_reference.md)).

On the M5 Pro it reports `Highway dispatched to target NEON_BF16 (4 float lanes)`: not SVE,
consistent with the `sysctl` flags in §1.4: and matches hand-written Neon within a few percent
(table above). Highway is a real production dependency (it underlies, e.g., the JPEG XL
reference codec), and "the width is a run-time value, not a constant in my source" is the
halfway point between intrinsics and the GPU, where the source never mentions the width at all.

---

## 1.6 Auto-vectorization, and what defeats it

The compiler will write the SIMD for you: when it can prove that doing so does not change your
program's meaning. `vec_remarks.c` is a zoo of loops designed to make the vectorizer explain
itself. Ask it:

```bash
# clang (Apple clang on the Mac)
cc -O3 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize \
   -c vec_remarks.c -o /dev/null
# gcc
gcc -O3 -fopt-info-vec-all -c vec_remarks.c -o /dev/null      # or -fopt-info-vec-missed
```

What Apple clang 21 says (abridged, real output):

```
vec_remarks.c:15:5: remark: vectorized loop (vectorization width: 4, interleaved count: 4)   saxpy_alias
vec_remarks.c:21:5: remark: vectorized loop (vectorization width: 4, interleaved count: 4)   saxpy_restrict
vec_remarks.c:29:5: remark: vectorized loop (vectorization width: 4, interleaved count: 4)   sum_float  (!)
vec_remarks.c:44:41: remark: loop not vectorized: unsafe dependent memory operations in loop ...
                     Backward loop carried data dependence.                                  prefix_sum
vec_remarks.c:49:5: remark: loop not vectorized: Cannot vectorize potentially faulting early exit loop
vec_remarks.c:58:36: remark: loop not vectorized: cannot identify array bounds               histogram
```

The things that defeat the vectorizer, in the order you will meet them:

**Aliasing.** In `saxpy_alias(float a, const float* x, float* y, n)`, the compiler cannot
know whether `x` and `y` overlap; if they do, vectorizing changes the answer. Clang's response
here is to emit a **run-time overlap check** and two copies of the loop (you can see the
`cmp`/`b.hs` pair ahead of the vector loop in `cc -O3 -S`). Declaring the pointers
**`restrict`** (your promise that they do not overlap) removes the check. Break the promise
and the behaviour is undefined: wrong answers, not errors. (Every CUDA kernel in this book writes
`__restrict__` for the same reason.)

**Floating-point reductions and associativity.** Float addition is not associative:
`(a + b) + c` and `a + (b + c)` round differently. Vectorizing a sum means four partial sums
combined at the end (a *different order*) so the compiler **may not do it** under the default
strict IEEE semantics. Here is the trap: clang *did* print "vectorized loop" for `sum_float`.
Read the assembly (`cc -O3 -S`) and the loop body is vector loads, then **16 scalar `fadd s0,
s0, …` in the original order**: an *ordered* ("strict in-order") reduction: loads vectorized,
the dependency chain intact. That is the "autovec-plain" column in §1.4: 1.6× from shortening the
chain from an FMA (~3 cycles) to an `fadd` (~2 cycles), not 16×. **A "vectorized" remark is not
proof of speed; the assembly and the timing are.** Three ways to grant permission to reorder:

- `#pragma omp simd reduction(+:s)` on that one loop, compiled with **`-fopenmp-simd`** (no
  OpenMP runtime needed, so it works with stock Apple clang). That is the "autovec-simd" column:
  15.5×, matching hand-written Neon.
- `-fassociative-math` (with `-fno-signed-zeros -fno-trapping-math`), for the whole file.
- **`-ffast-math`**: with it, clang's `sum_float` becomes eight `fadd.4s` vector accumulators.
  But `-ffast-math` also enables `-ffinite-math-only`: the compiler may assume **no NaNs or
  infinities exist**, so your `isnan()` checks can be deleted: and, per the
  [GCC manual](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html), "when used at link
  time, it may include libraries or startup files that change the default FPU control word"
  (on x86 this typically turns on flush-to-zero for denormals, *for the whole process*). Use the
  narrow tool, not the sledgehammer.

The integer version, `sum_int`, vectorizes cleanly: integer `+` is associative.

**True loop-carried dependences.** `a[i] = a[i-1] + a[i]` (a running prefix sum) needs
iteration `i-1`'s result. No flag fixes that; it needs a different *algorithm*: the parallel
scan of Chapter 14.

**Data-dependent control flow and scatters.** An early `return` inside the loop (unknown trip
count) and `bins[idx[i]] += 1` (two lanes may hit the same bin) both stop it. The histogram is
exactly the GPU problem of Chapter 15, solved there with atomics and privatization.

GPU compilers face none of these decisions: in SIMT you write the code for *one* element and
the hardware runs 32 of them (Chapter 4 §4.8). The reduction-order problem does not go away:
it becomes the determinism discussion of Chapter 12.

---

## 1.7 Threads: pthreads and OpenMP

SIMD uses more lanes of one core; **threads** use more cores. An OS thread is an independent
instruction stream with its own registers and stack, sharing the process's memory with other
threads, scheduled onto cores by the kernel.

**pthreads** is the raw API. `threads_sum.c` sums 128 M floats (537 MB) by giving each thread
a contiguous chunk and combining per-thread partial sums in a fixed order:

```c
for (int t = 0; t < nt; ++t) {
    jobs[t] = (sum_job){ .a = a, .lo = n*t/nt, .hi = n*(t+1)/nt };
    pthread_create(&th[t], NULL, sum_worker, &jobs[t]);   // worker: vectorized sum of [lo, hi)
}
for (int t = 0; t < nt; ++t) { pthread_join(th[t], NULL); s += jobs[t].result; }
```

**OpenMP** writes all of that (chunking, thread team, private partials, combine) as one
pragma, and the `simd` clause vectorizes each chunk:

```c
#pragma omp parallel for simd reduction(+ : s) schedule(static)
for (size_t i = 0; i < n; ++i) s += a[i];
```

It needs `-fopenmp` and a runtime library (libgomp for gcc, libomp for clang). **Stock Apple
clang ships no OpenMP runtime**, so on the Mac this part is compiled out; it builds and runs in
the `gcc:15` container (`code/ch01/README.md`). Note that `reduction` leaves the combine order
to the runtime, so the last bits of a float result can vary between runs; the pthreads version
above is deterministic.

**Measured** (`threads_sum`, pthreads, Apple M5 Pro, best of 5; other light background load on
the machine, load average ~1.5):

| Threads | 1 | 2 | 4 | 6 | 8 | 12 | 16 | 18 |
|---|---|---|---|---|---|---|---|---|
| GB/s | 88.9 | 151.4 | 156.8 | 147.4 | 187.1 | 207.4 | 227.1 | **236.6** |
| speedup | 1.00× | 1.70× | 1.76× | 1.66× | 2.10× | 2.33× | 2.55× | 2.66× |

A sum does 1 FLOP per 4 bytes. It is **memory bound**, and the curve shows it: 18 cores buy only
2.7×, flattening at **236.6 GB/s = 77% of Apple's 307 GB/s**. More threads cannot beat the
memory roof: the whole of Chapter 2 in one table. (The dip at 6 threads is where the OS places
threads across core clusters; it moved between runs.)

Two other things every threaded program meets:

- **Amdahl's law.** If a fraction `p` of the work parallelizes, the speedup on `N` cores is
  `1 / ((1−p) + p/N)`. At `p = 0.95` the limit is 20× no matter how many cores. GPUs obey the
  same law: the serial part is usually "copy to the device and launch".
- **False sharing.** Coherence works on whole cache lines. Eight threads each doing 20 M relaxed
  atomic increments of *their own* counter (Part 3 of `threads_sum`):

  | Counter spacing | Time | vs padded |
  |---|---|---|
  | 8 B (all in one 128 B line) | 1047.0 ms | **31.6×** |
  | 64 B (x86 line size: two per Apple line) | 58.9 ms | 1.8× |
  | 128 B (one Apple line each) | 33.1 ms | 1.0× |

  No data is shared, yet the packed version is 31× slower, because the line bounces between
  cores on every increment. And padding to 64 B (the right answer on x86) is still 1.8× slow on
  a 128 B-line Apple core. (With plain non-atomic stores the effect on this Mac was only ~1.2×,
  because stores are buffered and merged; atomics are where it hurts.)

---

## 1.8 Concurrency is not parallelism: goroutines, virtual threads, Tokio

**Parallelism** is doing more work per second: more cores, more lanes. **Concurrency** is having
many tasks *in progress*: structuring a program so that when one task waits (for the network, a
disk, a timer), another runs. It hides **latency**; it does not add **throughput**. Rob Pike's
talk title says it: [*Concurrency is not parallelism*](https://go.dev/blog/waza-talk).

The problem concurrency runtimes solve is that OS threads are expensive to *switch*. Blocking in
the kernel and waking another thread costs a system call, a trip through the scheduler, and cache
disruption. `ctx_switch.c` bounces a byte between two threads through pipes: **1.9 µs per
one-way hand-off** on the M5 Pro: about 8,800 cycles. Tens of thousands of threads, each mostly
waiting, would spend their lives switching. So language runtimes multiplex many cheap
user-space tasks onto a few OS threads (**M:N scheduling**, "green threads").

All three demos below run the same two experiments: 10,000 tasks that each wait 100 ms
(concurrency), and a CPU-bound integer hash of 200 M values split into chunks (parallelism). All
measured on the M5 Pro host (18 cores); versions in each subsection.

### Go: goroutines and GOMAXPROCS

The Go runtime scheduler (see the comments at the top of
[`runtime/proc.go`](https://github.com/golang/go/blob/master/src/runtime/proc.go)) has **G**s
(goroutines: a small, growable stack plus a saved register context), **M**s (OS threads) and
**P**s (processors: the right to run Go code). `GOMAXPROCS` = the number of Ps = the maximum
number of goroutines *executing Go code simultaneously*; it defaults to the number of logical
CPUs. Since **Go 1.25** the default also respects a Linux cgroup CPU limit and is updated
periodically if the CPU count or limit changes
([Go 1.25 release notes](https://go.dev/doc/go1.25)). A goroutine blocked on a channel, timer or
network poll is parked and costs no M; one blocked in a system call has its P handed to another M.

[`code/ch01/go`](https://github.com/ishwar6/ishwar-books/tree/main/code/gpu/ch01/go) (run with go 1.27.1 on the host):

| Experiment | Result |
|---|---|
| 10,000 goroutines × `time.Sleep(100ms)`, `GOMAXPROCS=1` | **112 ms** (serially: 1,000 s) |
| same, `GOMAXPROCS=18` | 104 ms |
| CPU-bound hash, `GOMAXPROCS` = 1 / 4 / 8 / 18 | 978 / 252 / 143 / 83 ms → **1.0 / 3.9 / 6.9 / 11.8×** |

One P was enough for 10,000 concurrent waits; only cores bought CPU speed. (18 cores giving
11.8×, not 18×: the 12 performance cores are not as fast as the 6 super cores, and all-core
clocks are lower than single-core: see §1.10.)

### Java: virtual threads (JEP 444, JDK 21)

[JEP 444](https://openjdk.org/jeps/444) made **virtual threads** final in JDK 21. A virtual
thread is *mounted* on a **carrier** (a platform/OS thread in a work-stealing `ForkJoinPool`
whose parallelism defaults to the number of processors) and *unmounted* when it blocks, so the
carrier can run another. The JEP is blunt: "Virtual threads are not faster threads … They exist
to provide scale (higher throughput), not speed (lower latency)." A virtual thread is **pinned**
(it cannot unmount) inside a `synchronized` block and inside native code. [JEP 491](https://openjdk.org/jeps/491)
(**JDK 24**) removed the `synchronized` case; pinning remains for native frames, class loading
and class initializers, and the `jdk.tracePinnedThreads` property was removed (the JFR event
`jdk.VirtualThreadPinned` remains).

[`code/ch01/java/VirtualThreads.java`](https://github.com/ishwar6/ishwar-books/blob/main/code/gpu/ch01/java/VirtualThreads.java), run on Temurin **25.0.4.1** and **21.0.12.1**:

| Experiment | JDK 25 | JDK 21 |
|---|---|---|
| 10,000 × `Thread.sleep(100)`, pool of 200 platform threads | 5,386 ms | 5,384 ms |
| same, one virtual thread per task | **133 ms** | 141 ms |
| CPU-bound hash, serial → 18 platform threads | 922 → 79 ms (11.7×) | 739 → 62 ms (11.9×) |
| CPU-bound hash, virtual threads | 80 ms (11.5×): **no better** | 55 ms (13.4×) |
| 2,000 virtual threads sleeping **inside `synchronized`** | **103 ms**, 2,000 asleep at once | **11,935 ms, max 18 asleep at once** |

The last row is JEP 491 made visible: on JDK 21 each sleeper pinned its carrier, so only 18
(= carriers) could wait at once: 2,000/18 × 100 ms ≈ 11 s. On JDK 25 the same code needs no
change. (The CPU-bound rows vary ±15% run to run with JIT and core placement; the point is that
virtual threads are not faster than a platform pool of NumCPU threads.)

### Rust: Tokio async tasks

An `async fn` in Rust compiles to a **state machine** (a `Future`); `.await` is a point where it
may return "not ready" so the executor can poll a different task. Scheduling is
**cooperative**: a task that never awaits never yields. Tokio's multi-thread runtime runs tasks
on worker threads (default: one per core) with **work stealing**; its tutorial says plainly that
for "speeding up CPU-bound computations by running them in parallel on several threads … you
should be using rayon" ([Tokio tutorial](https://tokio.rs/tokio/tutorial)). For blocking calls,
`spawn_blocking` moves the work to a separate blocking pool (default cap 512 threads,
[`Builder`](https://docs.rs/tokio/latest/tokio/runtime/struct.Builder.html)); the docs advise
bounding CPU-heavy use with a semaphore
([`spawn_blocking`](https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html)).

[`code/ch01/rust`](https://github.com/ishwar6/ishwar-books/tree/main/code/gpu/ch01/rust) (tokio 1.53.1, rayon 1.12.0, rustc/cargo 1.98.1, `--release`):

| Experiment | Result |
|---|---|
| 10,000 tasks × `tokio::time::sleep(100ms)` on a **one-thread** runtime | **108.5 ms** |
| CPU-bound hash, serial | 632 ms |
| (a) `tokio::spawn` chunks, `current_thread` runtime | 610 ms: **1.04×** |
| (b) `tokio::spawn` chunks, `multi_thread` (18 workers) | 53 ms: 11.9× |
| (c) `spawn_blocking` chunks | 52 ms: 12.2× |
| (d) `rayon` `par_iter` | 54 ms: 11.7× |
| 10 tasks calling **`std::thread::sleep(100ms)`** on 2 workers | **549.7 ms** (blocked the workers) |
| 10 tasks awaiting `tokio::time::sleep(100ms)` on 2 workers | 103.6 ms |

Async gave 10,000-way concurrency on one thread and exactly **zero** CPU speedup (row a). CPU
speed came only from more OS threads ((b), (c), (d)) and option (b) works *only because nothing
else on those workers needed to run*: a CPU-bound task that never awaits starves every I/O task
queued behind it. The last two rows are the classic async bug: a blocking call inside a task
freezes its worker thread.

---

## 1.9 The bridge: a GPU is green threads in hardware

Put the three latency-hiding strategies side by side:

| | OS threads | Green threads / async tasks | GPU warps (Chapter 3) |
|---|---|---|---|
| Who switches | kernel scheduler | language runtime | warp scheduler, in hardware, every cycle |
| Switch cost | ~2 µs measured here (~9,000 cycles) | tens–hundreds of ns (save a few registers, jump) | **zero cycles** |
| State on a switch | registers saved to memory | a few registers + stack pointer saved | **nothing saved**: every resident warp's registers stay in the register file |
| Typical count | tens to thousands | thousands to millions | up to 64 warps (2,048 threads) resident per SM on recent parts; 32 warps (1,024 threads) on a T4 |
| What it hides | blocking syscalls | I/O waits (ms) | DRAM latency (~hundreds of ns) and pipeline latency (a few cycles) |

A GPU streaming multiprocessor keeps the full execution context of every resident warp on-chip
for the warp's whole lifetime, so "switching from one execution context to another has no cost"
([CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)).
Each cycle, each scheduler picks some warp whose next instruction is ready. When a warp issues a
load that will take 500 cycles, the scheduler simply stops picking it.

That is Go's scheduler, done in silicon, at nanosecond granularity, with the context-switch cost
driven to zero by a trade: **registers are partitioned among resident threads rather than saved
and restored**. The cost has not disappeared; it has moved from *time* to *capacity*. A T4 SM has
a 64 K-entry (256 KB) register file; the more registers each thread uses, the fewer warps fit,
and the less latency can be hidden. That trade is the whole of Chapter 11 (occupancy).

So the GPU's design, read against this chapter:

- **Out-of-order execution → thrown away.** In-order issue per warp; the *other warps* are the
  independent work.
- **ILP → kept.** Independent instructions within one thread still overlap; "latency hiding =
  warps × ILP".
- **SIMD → kept, but hidden.** A warp *is* a 32-wide SIMD instruction stream, with the lane mask
  managed by hardware (Chapter 4 §4.8 draws the SIMT/SIMD line precisely).
- **Threads → multiplied.** Tens of thousands instead of eighteen.
- **Green-thread latency hiding → moved into hardware.** Zero-cost switch, but only among warps
  already resident.

---

## 1.10 Counting FLOP/s: the CPU's roofline, previewed

The peak floating-point rate of any processor is a product of the things this chapter measured:

```
peak FLOP/s = cores × SIMD lanes × FMA pipes per core × 2 (FLOP per FMA) × clock
```

For one M5 Pro super core, FP32 Neon, with the numbers from §1.2 (measured, since Apple publishes
none of them): `4 lanes × 4 FMA pipes × 2 × 4.59 GHz = 146.9 GFLOP/s`. `peak_flops` achieved
**142.1 GFLOP/s** on one thread (97%). For the whole chip we cannot simply multiply by 18: the
12 performance cores' clock and pipe count are unpublished, and all-core clocks sit below the
single-core clock. So measure it: **1,914.8 GFLOP/s** with 18 threads (`peak_flops` part 4):
72% of the naive 18 × 146.9. That is the Neon peak; the SME matrix units (§1.4) are a separate,
larger roof this chapter does not use.

The other axis is bytes. Chip bandwidth: 307 GB/s spec, **236.6 GB/s** measured in §1.7. Their
ratio is the **machine balance**:

```
1,915 GFLOP/s ÷ 236.6 GB/s ≈ 8 FLOP/byte      (vs ~25 for a T4, Chapter 2)
```

A dot product does 2 FLOPs per 8 bytes read: **arithmetic intensity 0.25 FLOP/byte**, 32× below
that balance. Whole-chip ceiling for a DRAM-sized dot product: `0.25 × 236.6 ≈ 59 GFLOP/s`: 3%
of peak compute, and that is *optimal*. Now re-read the §1.4 table:

- **In L1**, SIMD gave 15.8× (43.4 GFLOP/s). Even there it is not FMA bound (43 ≪ 142): each FMA
  needs two fresh 16-byte loads, and the load units, not the FMA pipes, are the limit.
- **From DRAM**, the Neon kernel moved **117 GB/s on one core**: half the chip's measured
  bandwidth from 1/18 of its cores. Adding cores (§1.7) topped out at 237 GB/s. The kernel is on
  the memory roof; more SIMD, more threads, or a better compiler can no longer help.

That is the whole roofline argument, arrived at from the CPU side: **compute peak comes from
cores × lanes × pipes × clock; achievable performance is capped by `AI × bandwidth`; and almost
every simple kernel is on the bandwidth side of the ridge.** Chapter 2 formalizes it.

It also tells you what a GPU is *for*. This laptop's CPU has about the same DRAM bandwidth as a
T4 (237 GB/s measured vs 320 GB/s spec): Apple's unified memory is unusually fast for a CPU;
Chapter 2 §2.6's "5–20× a CPU's bandwidth" is the typical desktop/server case. Its ~1.9 TFLOP/s
of Neon is about a quarter of a T4's 8.1 TFLOP/s FP32. A T4 would not beat it by much on this dot
product; it would beat it by far on anything with enough arithmetic intensity to be compute
bound: and on the datacenter GPUs of Parts VI–VII, with 3–8 TB/s HBM and tensor cores, by far on
both axes.

---

## 1.11 Summary

- A CPU core already exploits four kinds of parallelism: **ILP** (pipelining, superscalar,
  out-of-order), **SIMD** lanes, **threads** on cores, and **concurrency** runtimes that
  multiplex tasks onto threads. The first three add throughput; the fourth hides waiting.
- A dependent chain runs at one op per *latency*; independent chains run at one per *pipe*.
  Needed independent work = latency × throughput (Little's law): for FMAs on the M5 Pro,
  ~3 cycles × ~4 pipes ≈ 12–16 chains.
- **Out-of-order** execution finds ILP automatically via renaming and a reorder buffer, at a
  large per-thread cost in area and power, and still cannot cover a ~113 ns (~520-cycle) DRAM
  miss. GPUs delete it and switch warps instead.
- **SIMD**: SSE/AVX/AVX2/AVX-512 on x86, Neon (128-bit) and SVE/SVE2 (length-agnostic) on Arm.
  Apple M-series: Neon, no SVE, SME/SME2 from M4 (verified on this M5 Pro via `sysctl`).
  Intrinsics, Highway (`ScalableTag`, `Lanes`, `LoadU`, `MulAdd`, `HWY_DYNAMIC_DISPATCH`), or
  the auto-vectorizer: which is blocked by aliasing (`restrict`), FP reassociation
  (`omp simd reduction`, not `-ffast-math`), loop-carried dependences and data-dependent control.
- **Read the assembly**: clang reported "vectorized" for a float sum that kept a serial `fadd`
  chain (1.6×, not 16×).
- **Threads**: pthreads by hand, OpenMP by pragma (`parallel for simd reduction`); memory-bound
  work saturates bandwidth long before cores (2.7× on 18 cores, 77% of spec bandwidth); pad
  per-thread data to the **cache-line size of the machine** (128 B on Apple).
- **Concurrency is not parallelism.** Goroutines, virtual threads and Tokio tasks gave
  10,000-way waiting on one thread and zero CPU speedup on their own; CPU speed came only from
  more OS threads on more cores. Pinning (pre-JDK 24 `synchronized`) and blocking inside async
  tasks are the ways to lose the concurrency.
- **The GPU is green threads in hardware**: zero-cost warp switching because registers are
  partitioned, not saved: trading switch *time* for register *capacity* (Chapter 11).
- **Peak = cores × lanes × FMA pipes × 2 × clock**; achievable = `min(peak, AI × bandwidth)`.
  A dot product (AI 0.25) is memory bound on any machine. Chapter 2 builds on exactly this.

---

## Exercises

**1.1** Run `peak_flops` on your machine. From its clock estimate and best FLOP/cycle, compute
the single-core FP32 peak with the formula of §1.10 and compare with the measured best. Then
explain, using Little's law, why the throughput stops rising at the K where it does. What K
would you predict for a core with 5-cycle FMA latency and 2 FMA pipes?

**1.2** A server core has two 512-bit FMA units and runs at 3.0 GHz under AVX-512 load; the chip
has 56 such cores and 8 channels of DDR5-4800 (8 bytes per transfer per channel). Compute the
FP32 peak, the FP64 peak, the theoretical bandwidth, and the machine balance in FLOP/byte. Is a
dot product compute or memory bound on it? By how much?

**1.3** Change `dot_neon` in `simd_dot.c` to use one accumulator instead of four. Predict the L1
GFLOP/s from the measured `fmla` latency and clock before you run it, then measure. Repeat with
eight accumulators. Explain both results.

**1.4** In `vec_remarks.c`, compile `sum_float` with `-O3`, then with `-O3 -ffast-math`, and
count vector vs scalar `fadd` instructions in each (`cc -S`). Then add `#pragma omp simd
reduction(+:s)` and compile with `-O3 -fopenmp-simd` only. Which of the three would you ship, and
why? Write a three-line test that shows a program whose output `-ffast-math` changes.

**1.5** Run `threads_sum` and plot GB/s against thread count. At what thread count does it reach
90% of its final value? Using the single-thread GB/s and Little's law, estimate how many bytes
one core keeps in flight, given the ~113 ns DRAM latency from `mem_latency`.

**1.6** Modify the Go program so each goroutine in Part 1 does 100 ms of *CPU work* instead of
sleeping. Predict the wall time with `GOMAXPROCS=1` and with `GOMAXPROCS=18`, then measure. In
one sentence, what does this show about goroutines?

**1.7** In the Rust demo, give the multi-thread runtime 18 workers, spawn the CPU-bound chunks,
and at the same time spawn a task that awaits a 10 ms timer in a loop and prints how late each
wake-up is. What happens to the timer task while the chunks run? Fix it two different ways.

**1.8** A GPU SM on a T4 has 65,536 32-bit registers and supports at most 1,024 resident threads.
How many registers per thread can a kernel use and still keep all 1,024 threads resident? If a
kernel needs 128 registers per thread, how many warps fit, and what does that do to the SM's
ability to hide a 500-cycle load (§1.9)? (Preview of Chapter 11.)

---

## 🎯 Interview questions

<details>
<summary><b>Q1. What is the difference between concurrency and parallelism, and which one does a GPU rely on?</b></summary>

Parallelism is doing more work per unit time by executing several things at the same instant:
more cores, more SIMD lanes, more pipes. Concurrency is having several tasks in progress at once,
so that when one waits another can run; it hides latency but by itself adds no throughput. A
10,000-goroutine program with `GOMAXPROCS=1` is highly concurrent and not parallel at all, and in
this chapter's measurements it still finished 10,000 100-ms waits in about 110 ms, while the same
runtime gave zero speedup on CPU-bound work until more cores were allowed.

A GPU relies on both, and the strong answer says how. Its throughput comes from parallelism:
thousands of lanes executing at once. But it keeps those lanes busy through concurrency: each SM
holds many more warps than it can issue from in a cycle, and when one warp waits hundreds of
cycles on memory the scheduler issues from another. That is the concurrency trick of green
threads, implemented in hardware, and it is why a GPU needs far more threads than it has ALUs.
</details>

<details>
<summary><b>Q2. Why can a GPU switch between warps for free when an OS thread switch costs microseconds?</b></summary>

An OS context switch has to save the running thread's registers to memory, enter the kernel, run
the scheduler, restore another thread's registers, and usually suffers cache and TLB disruption
afterwards; a pipe ping-pong measured about 1.9 µs per hand-off on the M5 Pro. A GPU SM never
saves anything: the register file is partitioned among all resident warps, each warp's program
counter and registers stay on-chip for its whole lifetime, and each cycle the warp scheduler
simply chooses which ready warp to issue from. There is no state to move, so there is no cost.

The price is capacity rather than time. Because every resident thread's registers must fit in
the register file simultaneously, a kernel that uses more registers per thread allows fewer
resident warps, and fewer warps means less latency can be hidden. That trade-off is what
occupancy analysis is about, and it is the direct hardware analogue of a green-thread runtime
running out of memory for stacks.
</details>

<details>
<summary><b>Q3. What does out-of-order execution buy a CPU, what does it cost, and why do GPUs mostly skip it?</b></summary>

It buys single-thread performance on code nobody tuned. Register renaming removes false
dependencies caused by register reuse; a scheduling window lets any instruction whose inputs are
ready issue ahead of stalled older ones; a reorder buffer retires results in program order so
exceptions stay precise and mispredicted speculation can be discarded. Together with branch
prediction the core runs hundreds of instructions ahead, hiding L1 and L2 latency and much of the
control-flow cost of ordinary programs.

It costs area and power that do no arithmetic: large physical register files with many ports,
wakeup and select logic whose cost grows with window size and issue width, the ROB, and
predictors, all per thread and exercised on every instruction. And it still cannot cover a DRAM
miss: about 520 cycles on the machine measured here, versus a window that fills in on the order
of a hundred cycles. A GPU targets workloads that already come with abundant independent
threads, so it issues in order within each warp and spends the transistors on more ALUs and a
huge register file, getting its latency tolerance from switching warps instead of looking ahead
in one.
</details>

<details>
<summary><b>Q4. Your float sum loop didn't vectorize at -O3. Why, and how do you fix it properly?</b></summary>

Vectorizing a sum means keeping several partial sums and combining them at the end, which adds
the numbers in a different order. Floating-point addition is not associative, so the result can
change in its last bits, and under default IEEE semantics the compiler is not allowed to change
your answer. Some compilers refuse outright; clang on AArch64 may vectorize the loads and keep
an ordered chain of scalar adds, and report "vectorized" even though the dependency chain (the
actual bottleneck) is intact. The fix is to check the assembly and the timing, not the remark.

The proper fix is to grant permission narrowly: `#pragma omp simd reduction(+:s)` on that loop
with `-fopenmp-simd`, which needs no OpenMP runtime, or reassociation flags on that translation
unit. Hand-written multiple accumulators also work. `-ffast-math` is the blunt tool: it also
assumes no NaNs or infinities, which can delete your NaN checks, and on some toolchains linking
with it changes the floating-point control state of the whole process, such as flush-to-zero for
denormals. Also point out that the reordered sum is not less accurate in general (pairwise
partial sums are often more accurate) it is just different, which matters for reproducibility.
</details>

<details>
<summary><b>Q5. What is `restrict` and why does it matter for vectorization: and for CUDA?</b></summary>

`restrict` on a pointer is the programmer's promise that, within its scope, the memory it
points to is accessed only through that pointer, so it does not overlap with other restrict
pointers. Without it, in `y[i] = a*x[i] + y[i]` the compiler must allow for `x` and `y`
overlapping, in which case loading eight elements of `x` before storing earlier elements of `y`
would change the result. It must then either not vectorize or emit a run-time overlap check and
two versions of the loop, which clang did in this chapter's example.

With `restrict` the check disappears and the compiler can also keep values in registers across
stores. Breaking the promise is undefined behavior: silent wrong answers. The same reasoning
applies in CUDA, where `const float* __restrict__` lets the compiler reorder and batch loads and
use the read-only data path, which is why the kernels in this book mark their pointers that way.
</details>

<details>
<summary><b>Q6. Compute the peak FP32 FLOP/s of a CPU. What goes into the formula, and why is the real number lower?</b></summary>

Peak equals cores times SIMD lanes per register times FMA pipes per core times two FLOPs per
fused multiply-add times the clock. For one core of the M5 Pro used here, with the pipe count
and clock measured rather than published, that is 4 FP32 lanes of Neon, about 4 FMA pipes, 2,
and about 4.59 GHz: roughly 147 GFLOP/s, of which a microbenchmark with enough independent
accumulators reached 142. For an x86 core with two 512-bit FMA units the per-cycle figure is 16
lanes times 2 pipes times 2, or 64 FP32 FLOPs per cycle.

Real code falls short for several reasons, and naming them is the point of the question. The
peak assumes every instruction is an FMA, which only matrix-like kernels approach. It requires
enough independent chains to cover FMA latency. It requires feeding operands: a dot product
needs two loads per FMA and is limited by load bandwidth even in L1. Clocks drop when all cores
are busy (here 18 threads reached 72% of 18 times the single-core peak) and heterogeneous
cores differ. And above all, most kernels are memory bound: at 0.25 FLOP per byte a dot product
is capped at intensity times bandwidth, a few percent of peak, however perfect the code.
</details>

<details>
<summary><b>Q7. Why doesn't a CPU-bound computation get faster when you spread it over async tasks?</b></summary>

Because async tasks are a concurrency mechanism, not a parallelism mechanism. An async task is a
state machine polled by an executor, and it yields only at await points. On a single-threaded
runtime, any number of CPU-bound tasks simply run one after another on that thread; in the
chapter's Tokio measurement that gave 1.04 times the serial speed. Speed comes only from
spreading the work across OS threads on multiple cores.

The follow-up is what to do instead. You can run the tasks on a multi-threaded runtime, which
does parallelize them, but a task that computes without awaiting monopolizes its worker thread
and starves every I/O task queued behind it; blocking calls like `std::thread::sleep` inside
async code do the same, as the demo showed with ten sleeps on two workers taking five rounds.
The idiomatic answer is to hand CPU work to a dedicated pool: rayon for data parallelism, or
`spawn_blocking` for bounded blocking work: and keep the async runtime for waiting. Java's
virtual threads and Go's goroutines tell the same story, with the difference that Go preempts
long-running goroutines, so starvation is less severe, but the core count still caps speedup.
</details>

<details>
<summary><b>Q8. What is false sharing, and how does its GPU counterpart differ?</b></summary>

False sharing happens when threads write different variables that live in the same cache line.
Coherence protocols track ownership per line, so each write forces the line to move between
cores even though no data is logically shared. In this chapter eight threads incrementing their
own atomic counters ran 31 times slower when the counters were packed into one line than when
each had its own 128-byte line, and padding to 64 bytes (correct for x86) still left a 1.8
times penalty on Apple's 128-byte lines. The fix is to pad or align per-thread data to the line
size, or to accumulate in registers and write once.

On a GPU the per-core private caches are not kept coherent with each other in the same way, so
the classic ping-pong does not arise, but the underlying lesson (contention for the same memory
unit serializes independent work) reappears as atomic contention on a single address, as shared
memory bank conflicts, and as partition camping. The remedies are the same in spirit:
privatize, then combine once, which is exactly how the reduction and histogram kernels of
Chapters 12 and 15 are built.
</details>

---

**Sources:**
[Apple M5 Pro/Max announcement](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/) ·
[Intel Golden Cove (Architecture Day 2021)](https://www.tomshardware.com/features/intel-architecture-day-2021-intel-unveils-alder-lake-golden-cove-and-gracemont-cores/4) ·
[Arm SVE](https://developer.arm.com/Architectures/Scalable%20Vector%20Extensions) ·
[Arm Neon intrinsics reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/) ·
[Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) ·
[NVIDIA Grace vectorization guide](https://nvidia.github.io/grace-cpu-benchmarking-guide/developer/vectorization.html) ·
[M4 SME exploration (reverse-engineered)](https://github.com/tzakharko/m4-sme-exploration/blob/main/reports/01-sme-overview.md) ·
[Nova Lake AVX10.2 (Phoronix)](https://www.phoronix.com/news/Nova-Lake-Does-AVX10.2-APX) ·
[Google Highway](https://github.com/google/highway) and its [quick reference](https://github.com/google/highway/blob/master/g3doc/quick_reference.md) ·
[GCC optimize options (`-ffast-math`)](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) ·
[Clang optimization remarks](https://clang.llvm.org/docs/UsersManual.html#options-to-emit-optimization-reports) ·
[OpenMP specifications](https://www.openmp.org/specifications/) ·
[Go `runtime/proc.go`](https://github.com/golang/go/blob/master/src/runtime/proc.go) ·
[Go 1.25 release notes](https://go.dev/doc/go1.25) ·
[Concurrency is not parallelism](https://go.dev/blog/waza-talk) ·
[JEP 444](https://openjdk.org/jeps/444) · [JEP 491](https://openjdk.org/jeps/491) ·
[Tokio tutorial](https://tokio.rs/tokio/tutorial) ·
[`tokio::task::spawn_blocking`](https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html) ·
[CUDA Programming Guide: advanced kernel programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)

---

**Next:** [Chapter 2: Why GPUs Exist](02-why-gpus.md)
