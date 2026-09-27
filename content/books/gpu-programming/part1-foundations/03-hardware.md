# Chapter 3 · The Hardware

> **Goal:** a mental model of the physical machine accurate enough to predict performance from.
> Not trivia: every number here shows up later as a constraint you have to design around.

---

## 3.1 The hierarchy, top down

```
GPU (one die, or two glued together on a B200)
├── L2 cache                      ← shared by everything: 4 MB on a T4, 40–126 MB on datacenter parts
├── Memory controllers → DRAM     ← GDDR6 / HBM2e / HBM3e, "global memory"
└── GPC ×N   (Graphics Processing Cluster)
    └── TPC ×M  (Texture Processing Cluster)
        └── SM ×1–2  ← Streaming Multiprocessor: THE unit that matters
            ├── 4 × processing block (a.k.a. SM sub-partition)
            │   ├── warp scheduler + dispatch
            │   ├── register file slice
            │   ├── CUDA cores (FP32/INT32 ALUs)
            │   ├── 1 Tensor Core
            │   ├── LSU (load/store), SFU (special function), FP64 unit
            │   └── (Blackwell) 64 KB Tensor Memory slice
            ├── unified L1 data cache / shared memory  (96–256 KB)
            ├── read-only constant cache
            └── (Hopper+) Tensor Memory Accelerator (TMA)
```

You can ignore GPCs and TPCs: they're a graphics-era grouping (and the number of SMs per TPC
is not always 2). **The SM is the unit you program against.**

A **block** of threads is assigned to exactly one SM and lives there until it finishes. One SM
holds **several blocks at once**: as many as fit in its registers, shared memory, and thread
slots. That fraction of the SM's thread slots that are filled is **occupancy**. A T4 SM has 1024
thread slots; two 256-thread blocks on it is 50% occupancy. Chapter 11 is the full calculation;
you only need the word.

Counts, for calibration:

| GPU | SMs | Threads/SM | Max resident threads |
|---|---|---|---|
| T4 (cc 7.5) | 40 | 1024 | 40,960 |
| A100 (cc 8.0) | 108 | 2048 | 221,184 |
| H100 SXM (cc 9.0) | 132 | 2048 | 270,336 |
| B200 (cc 10.0) | 148 | 2048 | 303,104 |

Read that last column as: *a B200 can have 303,104 threads mid-flight simultaneously.* Nothing about
CPU intuition prepares you for this.

---

## 3.2 The SM in detail

An SM is a small, weird, extremely wide processor. Blackwell's, as an example
([Cornell VW](https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/b200_sm),
[Jarmusch & Chandrasekaran, arXiv:2512.02189](https://arxiv.org/abs/2512.02189)):

- **128 FP32 CUDA cores**, 128 INT32, 64 FP64
- **4 fifth-gen Tensor Cores** (one per processing block)
- **256 KB register file** (64K × 32-bit registers)
- **256 KB unified L1/shared block**, of which up to **228 KB** can be carved out as shared memory
- **256 KB Tensor Memory (TMEM)**: new in Blackwell, a dedicated scratchpad for tensor-core operands
- **64 resident warps max** (= 2048 threads)

### The four processing blocks

The SM is divided into four **processing blocks** (NVIDIA also says "sub-partitions"). Each has its
own warp scheduler and its own slice of the register file. **A warp is assigned to one processing
block for its lifetime.** Its registers come from that block's slice; its instructions issue from
that block's scheduler.

This matters more than it sounds:

- Each scheduler issues **at most one instruction per cycle, to one ready warp**: not "every
  warp issues every cycle." The other resident warps wait their turn or sit stalled on memory.
- **Issue width is not the same on every GPU.** Blackwell (and Hopper, Ada, consumer Ampere) has
  32 FP32 cores per processing block, so one FP32 instruction from a 32-thread warp fills the
  pipe for exactly one cycle. The T4 you actually have is Turing: **64 FP32 cores per SM ÷ 4 =
  16 per processing block**, so that same warp instruction takes **two cycles**. Same programming
  model, half the issue rate. Peak FLOP/s already folds this in; you do not have to schedule
  around it. You *do* have to stop copying Blackwell SM diagrams onto a T4.
- The scheduler picks among *its* resident warps only. If a processing block's warps are all
  stalled while another processing block has spare work, the first one idles. You can't control
  this: but it's why occupancy is measured per SM and why very uneven work between warps hurts.

### Warp scheduling, concretely

Each cycle, per processing block:

1. Look at all resident warps assigned to me.
2. Find the ones whose next instruction has all operands ready and whose functional unit is free.
3. Pick one (greedy-then-oldest, roughly). Issue it. Done.

Zero-cost context switching, because there *is* no context switch: every resident warp's registers
are simultaneously live in the register file. That is what the enormous register file buys you: not
capacity for one thread, but **many threads' state resident at once.** This is the hardware
mechanism behind everything Chapter 2 said about latency hiding.

⚠️ The corollary that trips people up: **registers are the scarcest resource on the SM.** 64K
registers per SM, 2048 threads max → 32 registers per thread if you want full occupancy. Ask for 64
and your occupancy halves. Ask for more than 255 and the compiler spills to (slow, local) memory.
Chapter 11.

---

## 3.3 The memory hierarchy and what it costs

From fastest to slowest. **Latencies are approximate and architecture-dependent**: treat them as
orders of magnitude, and measure your own with a pointer-chase microbenchmark if you need real
numbers (see [arXiv:2512.02189](https://arxiv.org/abs/2512.02189) for careful Blackwell/Hopper
measurements).

| Level | Scope | Size | Latency (≈ cycles) | Bandwidth | Managed by |
|---|---|---|---|---|---|
| **Registers** | one thread | 255/thread, 64K/SM | ~0 (1, pipelined) | ~100s TB/s aggregate | compiler |
| **Shared memory** | one block | 64–228 KB/SM | ~20–30 | ~10s TB/s aggregate | **you** |
| **L1 / unified cache** | one SM | 96–256 KB/SM | ~30–40 | high | hardware |
| **L2 cache** | whole GPU | 4–126 MB | ~200–300 | ~5–10 TB/s | hardware |
| **Global (DRAM)** | whole GPU | 16 GB – 192 GB | **~400–800** | 320 GB/s – 8 TB/s | you (allocation) |
| **Host RAM over PCIe** | system | ~TB | ~10,000+ | 32–64 GB/s | you |

The two numbers to burn in: **shared memory is ~20× closer than DRAM, and DRAM costs ~500 cycles.**
Five hundred cycles is ~500 FP32 instructions you could have executed instead. That is why tiling
into shared memory (Ch. 9) is *the* fundamental GPU optimization.

### The address spaces you actually name in code

| Space | Declaration | Lifetime | Notes |
|---|---|---|---|
| Register | local variable | thread | The default. Fast. Finite. |
| Local | local array too big / spilled / dynamically indexed | thread | ⚠️ **Physically lives in DRAM** (up to 512 KB/thread). "Local" is about *scope*, not location. Cached in L1/L2. Spills land here. |
| Shared | `__shared__ float s[256];` | block | Scratchpad. You manage it. Banked (Ch. 9). Static `__shared__` is still capped at 48 KB; anything larger must be *dynamic* shared memory with an explicit opt-in (`cudaFuncSetAttribute` / `cudaFuncAttributeMaxDynamicSharedMemorySize`). The "227 KB/block" in the table below is the opt-in ceiling, not the default. |
| Global | `cudaMalloc` | application | DRAM. Coalescing matters (Ch. 8). |
| Constant | `__constant__ float c[64];` | application | 64 KB total, 8 KB cache/SM (the programming-guide number). Fast **only** when all threads in a warp read the same address; otherwise serializes. |
| Read-only / `__ldg` | `const __restrict__` pointer | application | Asks the compiler to emit `LDG`. Still the right default for inputs you will not write. |
| Texture | texture object / `tex1D` | application | Separate sampler path: 2D spatial locality, hardware interpolation. Rarely needed for compute. Not the same thing as `__ldg`. |

⚠️ **The single most expensive beginner mistake in this table** is not knowing that "local memory" is
DRAM. Write

```cpp
__global__ void oops(float* out, int n) {
    float scratch[64];          // 64 registers? No.
    for (int i = 0; i < n; ++i) scratch[i % 64] = ...;   // dynamic index → forced to local memory
    ...
}
```

and you have just put a 64-element array per thread in *global memory*. With 1024 threads per SM
that's 256 KB per SM of DRAM traffic that looks, in your source, like a local variable. Kernels
regress 10× this way. Chapter 11 shows how to spot it (`nvcc -Xptxas -v` reports spills and local
memory usage: always read that output).

### Caches are small and shared: do not think like a CPU

L2 on an H100 is 50 MB, shared across 132 SMs and up to 270,000 threads. That's ~190 bytes per
thread. On a CPU you rely on caches; **on a GPU you rely on coalescing and shared memory, and treat
caches as a bonus.** A kernel designed around cache reuse per thread will not work.

The exception: L2 is genuinely useful for data reused *across blocks*: e.g. the same weight tile
read by many blocks. Ampere+ (cc 8.0 and later) even lets you reserve a portion of L2 for persistent data
([L2 cache control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)).

---

## 3.4 How memory requests actually work

This is the mechanism underneath Chapter 8, and it's worth understanding once, properly.

When a warp executes a load instruction, it produces **32 addresses**. The memory subsystem does not
service 32 requests. It:

1. Breaks the 32 addresses into the set of distinct **32-byte sectors** they fall into.
   (A 128-byte cache line is four sectors; the sector is the DRAM transaction granularity on
   Maxwell and later.)
2. Issues one transaction per distinct sector.
3. Distributes the returned bytes back to the lanes that wanted them.

So the cost of a warp's load is **the number of distinct sectors it touches**, not the number of
threads.

**Best case.** 32 threads read 32 consecutive `float`s = 128 contiguous bytes = **4 sectors**.
128 bytes requested, 128 delivered. 100% efficiency.

**Worst case.** 32 threads read `float`s strided 32 apart. Each lands in its own sector →
**32 sectors** = 1024 bytes moved to deliver 128 useful bytes. **12.5% efficiency, 8× the traffic.**

**Pathological case.** 32 threads read `float`s strided 1024 bytes apart, from a 4 GB array. 32
sectors *and* 32 different DRAM pages, so you also lose row-buffer locality. Worse than 8×.

This single mechanism is responsible for most of the difference between a beginner kernel and a good
one. Interviewers ask about it constantly (🎯) because it separates people who have read about GPUs
from people who have profiled one.

---

## 3.5 SIMT: how 32 threads share one instruction pointer

NVIDIA calls the execution model **SIMT**: Single Instruction, Multiple Thread. It is SIMD hardware
presented as scalar threads.

- The **warp** is 32 threads. It has been 32 on every CUDA-capable NVIDIA GPU. (AMD's equivalent, the
  wavefront, is 32 or 64.) The 32-byte DRAM sector came later (Maxwell, 2014): warp size is not
  "because sectors are 32 bytes." A fully coalesced `float` load just happens to be one 128-byte
  cache line = four sectors = one warp.
- Threads in a warp execute **in lockstep** on shared instruction-issue hardware.
- The programming model lets you write per-thread scalar code with `if`s. The hardware handles
  divergence with an **active mask**: when a warp hits a branch that some lanes take and others
  don't, it executes *both* paths, masking off the inactive lanes in each. The cost is the sum of
  both paths (Ch. 10).

**Volta changed this** (2017), and it's a favorite interview topic. Pre-Volta, a warp had one program
counter; lanes could not make independent forward progress, so a spin-lock where lane 0 holds a lock
and lane 1 waits for it **deadlocks**. Volta introduced **independent thread scheduling**: per-lane
program counters, so lanes can diverge, make progress independently, and reconverge.

Consequences you must know:
- Warp-synchronous programming without explicit sync is **broken** on Volta+. Code like
  `s[tid] += s[tid+16];` inside a warp with no `__syncwarp()` used to work by accident. It is now
  undefined.
- All warp intrinsics gained a mandatory mask argument: `__shfl_sync`, `__ballot_sync`,
  `__any_sync`. The old `__shfl` / `__ballot` / `__any` without `_sync` are **deprecated**: do
  not write them. If you see them in a tutorial, the tutorial predates 2017.
- `__syncwarp()` exists to force reconvergence explicitly.

Chapter 10 covers this properly. Reference:
[Programming Guide: independent thread scheduling](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#independent-thread-scheduling).

---

## 3.6 The architecture timeline, and what each generation actually changed

This is the "what changed over the past decade" question directly. For each generation: the one thing
that mattered.

| Arch | Year | CC | Flagship | The one thing that changed |
|---|---|---|---|---|
| **Kepler** | 2012 | 3.x | K20 / K80 | Dynamic parallelism; Hyper-Q (multiple hardware work queues); warp shuffle introduced |
| **Maxwell** | 2014 | 5.x | M40 | Big efficiency rework; **shared memory split from L1**; 32-byte sectors |
| **Pascal** | 2016 | 6.x | P100 | **HBM2** (732 GB/s, ~2.5× Maxwell's 288 GB/s GDDR5); NVLink 1.0; unified memory with page faulting; FP16 arithmetic |
| **Volta** | 2017 | 7.0 | V100 | **Tensor Cores** (1st gen, FP16); **independent thread scheduling**; unified L1/shared; separate INT32 pipe |
| **Turing** | 2018 | 7.5 | T4, RTX 20 | 2nd-gen Tensor Cores + **INT8/INT4**; RT cores; cheap inference GPU (your T4) |
| **Ampere** | 2020 | 8.0/8.6 | A100 | **`cp.async`** (async global→shared copy, no register round trip); **TF32**; BF16; 2:4 structured sparsity; MIG; 3rd-gen tensor cores; L2 residency control |
| **Ada** | 2022 | 8.9 | L4, RTX 40 | **FP8** tensor cores; 4th-gen; big L2 (up to 96 MB) |
| **Hopper** | 2022 | 9.0 | H100/H200 | **The big one.** `wgmma` (warpgroup-level async matmul, 128 threads); **TMA** (Tensor Memory Accelerator: hardware DMA engine for tiles); **thread block clusters** + **distributed shared memory**; async barriers; FP8; 228 KB shared/SM |
| **Blackwell** | 2024–25 | 10.0 / 10.3 | B200 / B300 | **`tcgen05`**: a *single thread* now issues the whole matmul (TMEM allocation is warp-level); dedicated **256 KB TMEM** for operands; **FP6 and FP4**; 2nd-gen Transformer Engine; NVLink 5 |
| **Blackwell (consumer)** | 2025 | 12.x | RTX 50 | Same ISA family, smaller SM budget (100 KB shared, 48 warps) |
| *(Jetson Thor)* | 2025 | 11.0 | Thor | Embedded Blackwell derivative: note CC 11.0 is *not* a newer datacenter part |

Verified against
[Programming Guide: Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
and [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/).

### The three inflection points that matter

If you only remember three transitions:

**1. Volta (2017): matmul became a hardware instruction.** Before, matmul was FMA in a loop. After,
a tensor core does a whole small matrix product per instruction. This is why FP32 peak stopped being
the interesting number and why deep learning performance decoupled from graphics performance.

**2. Ampere → Hopper: data movement became asynchronous and hardware-managed.**
- Ampere's `cp.async`: copy global→shared without going through registers, and without the thread
  waiting.
- Hopper's **TMA**: describe a multidimensional tile once, and a hardware DMA engine handles
  addressing, bounds, and the transfer. One thread launches it; the whole block consumes it.

  The point of both: **overlap.** A well-written Hopper GEMM is a software pipeline where TMA fetches
  tile *k+1* while tensor cores multiply tile *k*. Getting near peak on Hopper/Blackwell is mostly
  about pipelining, not about the math.

**3. Hopper → Blackwell: the matmul issue scope moved, twice.** WMMA (Volta, warp-level, synchronous)
→ `wgmma` (Hopper, 128-thread warpgroup, asynchronous) → `tcgen05.mma` (Blackwell, *single-thread
semantics* (one thread's instruction initiates the whole MMA) with the accumulator, and optionally
A, living in dedicated TMEM while B comes from shared memory). The trend is that the
*data* is increasingly moved by dedicated hardware and held in dedicated storage, while threads just
issue descriptors. Chapter 17.

### The two levels the model gained

Worth flagging because they change the programming model, not just the speed:

- **Thread block clusters** (Hopper): a new level between block and grid. A cluster of blocks is
  co-scheduled on the same GPC and can read each other's shared memory: **distributed shared
  memory (DSMEM)**. This is the first time blocks can cooperate cheaply.
- **Tensor Memory** (Blackwell): a fourth on-chip storage class alongside registers, shared, and
  local: 256 KB/SM, addressable only by `tcgen05` instructions, with a *microbenchmarked* read
  bandwidth of ~16 TB/s per SM ([arXiv:2512.02189](https://arxiv.org/abs/2512.02189)).

---

## 3.7 Reference: hardware limits by compute capability

Straight from the
[Programming Guide appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
You will need these for occupancy math in Chapter 11, so bookmark it.

| Limit | 7.5 (T4) | 8.0 (A100) | 8.6 (A10) | 8.9 (L4) | 9.0 (H100) | 10.x (B200) | 12.x (RTX 50) |
|---|---|---|---|---|---|---|---|
| Max threads / block | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 | 1024 |
| Max warps / SM | **32** | 64 | 48 | 48 | 64 | 64 | 48 |
| Max threads / SM | **1024** | 2048 | 1536 | 1536 | 2048 | 2048 | 1536 |
| Max blocks / SM | 16 | 32 | 16 | 24 | 32 | 32 | 24 |
| 32-bit registers / SM | 64 K | 64 K | 64 K | 64 K | 64 K | 64 K | 64 K |
| Max registers / thread | 255 | 255 | 255 | 255 | 255 | 255 | 255 |
| Max shared mem / SM | 64 KB | 164 KB | 100 KB | 100 KB | **228 KB** | 228 KB | 100 KB |
| Max shared mem / block | 64 KB | 163 KB | 99 KB | 99 KB | 227 KB | 227 KB | 99 KB |
| Shared memory banks | 32 | 32 | 32 | 32 | 32 | 32 | 32 |
| Unified L1+shared / SM | 96 KB | 192 KB | 128 KB | 128 KB | 256 KB | 256 KB | 128 KB |
| FP32:FP64 throughput | 32:1 | **2:1** | 64:1 | 64:1 | **2:1** | **2:1** (10.0) / 64:1 (10.3) | 64:1 |

Things to notice, because they're all interview-relevant:

- **Registers per SM never changed.** 64 K since Kepler. But max threads/SM doubled on some parts:
so the *registers available per thread at full occupancy* went **down**. On a T4: 64K/1024 = 64
  registers/thread at full occupancy. On an H100: 64K/2048 = **32**. Modern GPUs make register
  pressure a harder constraint than old ones.
- **Shared memory per SM grew 3.5×** (64 KB → 228 KB) while registers didn't. Hopper wants you to
  keep working sets in shared memory, not registers. That shapes modern kernel design.
- **Consumer Blackwell (12.x) is not a small datacenter Blackwell.** Unified L1+shared is 128 KB
  (max shared carveout **100 KB**) vs 256 KB / 228 KB on a B200, and 48 warps vs 64. A kernel that
  asks for 227 KB of shared memory per block will not even launch on an RTX 5090. The 48 KB static
  default still applies on both: large tiles need the opt-in on Hopper *and* on Blackwell.
- **FP64 is a product-segmentation lever, not a technology one.** 2:1 on A100, H100, and B200
  (cc 10.0); 64:1 on Ada, the consumer parts, and B300 (cc 10.3): a 32× difference, and NVIDIA
  moved it *within* one architecture generation. Never benchmark scientific FP64 code on a gaming
  card and extrapolate.

[`code/ch05/device_query.cu`](https://github.com/ishwar6/ishwar-books/blob/main/code/gpu/ch05/device_query.cu) (Chapter 5) prints all of this for whatever GPU you're on. Run it
before you optimize anything, ever.

---

## 3.8 Summary

- The **SM** is the unit of scheduling and resources: 4 processing blocks, each with a warp
  scheduler, register-file slice, CUDA cores, and a tensor core; plus per-SM shared memory/L1.
  Several blocks reside on one SM at once; that filling fraction is occupancy.
- A **warp** is 32 threads sharing instruction issue. Divergence costs the sum of both paths.
  On a T4, one FP32 warp instruction takes two cycles (16-wide pipe); on Hopper/Blackwell, one.
- The register file is huge so that **many warps are resident at once**: that's the latency-hiding
  mechanism.
- Memory requests are serviced in **32-byte sectors**. A warp's load costs the number of distinct
  sectors it touches. This is coalescing.
- **"Local memory" is DRAM.** Spills and dynamically-indexed local arrays are silent 10× regressions.
- Since Volta, matmul is a hardware instruction; since Hopper, data movement is asynchronous and
  hardware-managed; since Blackwell, tensor operands live in dedicated TMEM.
- Registers/SM has been flat at 64 K while threads/SM doubled: register pressure got *harder*, not
  easier.

---

## Exercises

**3.1** A T4 has 40 SMs, max 1024 threads/SM, max 16 blocks/SM. You launch 100 blocks × 256 threads.
(a) Ignoring registers and shared memory, how many 256-thread blocks *can* one SM hold? (b) 100
blocks do not divide evenly across 40 SMs: some SMs get 3 blocks, some get 2. Threads resident
on each kind of SM? Occupancy in each case? (c) What happens if you launch 100 blocks × 1024
threads? Think about the tail wave: the last 20 blocks leave 20 SMs idle.

**3.2** Your kernel uses 40 registers per thread on an H100. What is the maximum occupancy (warps/SM)
achievable? What if it uses 32? 65? (Registers are allocated in granularity: assume 8 for now;
Ch. 11 gives the real rule.)

**3.3** A warp reads `float`s from a `float* p` with `p[threadIdx.x * stride]`. For stride = 1, 2, 4,
8, 32: how many 32-byte sectors does the warp touch, and what is the bandwidth efficiency? (Assume
`p` is 128-byte aligned.) Plot it. You'll measure this exact curve in Chapter 8.

**3.4** Why does a Hopper GEMM tuned for 227 KB of shared memory per block fail to launch on an
RTX 5090? What are the two ways to write one kernel that runs well on both?

**3.5** On an H100, full occupancy allows 32 registers/thread. A register-blocked GEMM wants 8×8 = 64
accumulators per thread, plus operands and addresses: say 100 registers. What occupancy does that
force? Why might that kernel still be the fastest one? (Answer in Ch. 11 and 16; reason it out now.)

**3.6** Read `nvcc -Xptxas -v` output for any kernel you have. Identify: registers per thread, shared
memory per block, spill stores, spill loads, local memory. Which of these numbers would make you
stop and rewrite the kernel?

---

## 🎯 Interview questions

<details>
<summary><b>Q1. What is a warp, and why is 32 the number?</b></summary>

A warp is the hardware's unit of scheduling: 32 threads that share instruction issue and execute in
lockstep under an active mask. It has been 32 since the first CUDA GPU (G80, 2006). The datapaths
were built around that choice: 32 lanes × 4-byte `float`s = 128 bytes, which later (Maxwell) became
exactly one cache line / four 32-byte sectors. Don't reverse the history: sectors are 32 bytes
*because* a warp of floats is 128 bytes that they decided to split, not the other way around. And
on a T4 the FP32 pipe is only 16-wide per scheduler, so one warp instruction takes two cycles;
the "32-wide issue" story is Hopper/Blackwell.

The programmer-visible consequences are the important part of the answer: the warp is the
granularity of divergence (an `if` splitting a warp costs both paths), of coalescing (the 32
addresses are what get merged into sectors), of occupancy accounting (resources are allocated per
warp), and of the warp intrinsics (`__shfl_sync`, `__ballot_sync`) which let lanes exchange data
without touching memory at all.
</details>

<details>
<summary><b>Q2. Walk me through the GPU memory hierarchy and the cost of each level.</b></summary>

Registers (per thread, ~1 cycle, 64 K per SM total, allocated at compile time: the scarce resource);
shared memory (per block, ~20–30 cycles, 64–228 KB per SM, programmer-managed, banked into 32 banks);
L1/unified cache (per SM, ~30–40 cycles, physically the same SRAM as shared on Volta+, split
configurably); L2 (whole GPU, ~200–300 cycles, 4–126 MB); DRAM (~400–800 cycles, 320 GB/s to 8 TB/s);
host memory over PCIe (10,000+ cycles, 32–64 GB/s).

The insight to volunteer: because DRAM is ~500 cycles and shared memory is ~25, the fundamental
optimization is to load a tile into shared memory once and reuse it many times: that is what
tiling is, and it's why arithmetic intensity is the metric that matters. And the trap to mention:
"local memory" is not a fast level; it's DRAM, and register spills go there.
</details>

<details>
<summary><b>Q3. What changed in Volta, and why does it break old CUDA code?</b></summary>

Two things: tensor cores were introduced, and (the one that breaks code) **independent thread
scheduling**. Before Volta a warp had a single program counter, so lanes were guaranteed to execute
in lockstep and reconverge at the immediate post-dominator. A lot of code exploited that guarantee:
warp-synchronous reductions like `if (tid < 32) { s[tid] += s[tid+16]; ... }` with no
synchronization, relying on lockstep for correctness.

Volta gave each lane its own program counter. Lanes can now diverge and progress independently, which
enables things that used to deadlock (a spin-lock among lanes of the same warp) but removes the
implicit-lockstep guarantee. Warp-synchronous code without explicit sync became undefined behavior.
Hence the `_sync` suffix on all warp intrinsics with a mandatory participation mask, and
`__syncwarp()` for explicit reconvergence. `volatile` on the shared array was the *pre-Volta*
hack to stop the compiler eliding reads; it is **not sufficient** on Volta+. The modern fix is
`__syncwarp()`, not `volatile`.
</details>

<details>
<summary><b>Q4. Registers per SM has been 64 K since Kepler while threads per SM doubled. What does that imply for how you write kernels today?</b></summary>

Register budget per thread at full occupancy fell: 64 registers/thread on a T4 (64K/1024) vs 32 on
an H100 (64K/2048). So the constraint tightened exactly as kernels got more complex.

Practically: you can no longer assume "keep everything in registers" scales. Modern kernels are
designed around the *shared memory* budget, which grew 3.5× (64 KB → 228 KB per SM), and around
explicitly accepting lower occupancy in exchange for large per-thread register tiles when the kernel
has enough ILP to hide latency without many warps. That's precisely the design of a modern
register-blocked GEMM: ~50% or even 25% occupancy, huge accumulator tiles, latency hidden by
instruction-level parallelism and asynchronous copies rather than by thread count.

Concretely it also means always reading `-Xptxas -v` and using `__launch_bounds__` to tell the
compiler your target occupancy so it budgets registers instead of spilling.
</details>

<details>
<summary><b>Q5. What is the TMA and what problem does it solve?</b></summary>

The Tensor Memory Accelerator (Hopper, cc 9.0) is a hardware DMA engine for multidimensional tiles.
You build a descriptor on the host describing a tensor's shape, strides, and tile size; then a
*single* thread issues one instruction and the engine copies a tile between global and shared memory,
handling all address generation, bounds/out-of-range padding, and optional swizzling.

The problems it solves: (1) address arithmetic used to consume a meaningful fraction of instruction
issue in a GEMM inner loop: TMA removes it entirely; (2) the copy is fully asynchronous and tracked
by a barrier, so it composes into a software pipeline where the tensor cores work on tile *k* while
TMA fetches tile *k+1*; (3) it needs no registers for staging, unlike a manual load-to-register,
store-to-shared sequence, and unlike even Ampere's `cp.async` it needs no per-element addressing.

Ampere's `cp.async` was the halfway step: async, bypasses registers, but still per-thread
addressing. TMA is the full version. On Hopper and Blackwell, reaching near-peak matmul is mostly
about building that pipeline correctly.
</details>

<details>
<summary><b>Q6. Why can't caches save you on a GPU the way they do on a CPU?</b></summary>

Capacity per thread. An H100 has 50 MB of L2 shared by up to 270,000 resident threads (under 200
bytes each) and 256 KB of L1/shared per SM shared by 2048 threads. A CPU core has hundreds of KB of
private L2 for one or two threads.

So GPU caches cannot hold a per-thread working set, and a kernel designed around temporal locality
per thread will thrash. GPUs instead get their memory efficiency from *spatial* coalescing across the
warp (making each DRAM transaction fully useful) and from *explicit* staging in shared memory (where
you, not the replacement policy, decide what stays). Caches do help for data reused across many
blocks (a weight tile, a small lookup table) and Ampere+ lets you pin such data in a reserved L2
partition. But the default assumption should be: if you want reuse, you arrange it yourself.
</details>

---

**Next:** [Chapter 4: The Programming Model](04-programming-model.md)

**Sources:**
[CUDA Programming Guide 13.3](https://docs.nvidia.com/cuda/cuda-programming-guide/) ·
[Compute Capabilities appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) ·
[Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/) ·
[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) ·
[Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) ·
[Microbenchmarking Blackwell (arXiv:2512.02189)](https://arxiv.org/abs/2512.02189) ·
[Cornell VW: B200 SM](https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/b200_sm)
