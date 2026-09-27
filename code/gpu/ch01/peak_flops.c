// peak_flops.c -- measure the numbers that go into
//
//     peak FP32 FLOP/s = cores x SIMD lanes x FMA pipes x 2 x clock
//
// on THIS machine, instead of trusting a spec sheet (Apple publishes no CPU
// clock speeds or pipe counts at all). arm64 (Neon) only.
//
//   1. clock      : a chain of dependent integer adds. Every modern core does a
//                   dependent add in 1 cycle, so adds/second ~= cycles/second.
//   2. latency    : dependent chains of fadd / fmla (vector FMA), in cycles.
//   3. throughput : K independent FMA chains, K = 1..24. With too few chains the
//                   core waits on latency; with enough chains it is limited by
//                   the number of FMA pipes. Latency x pipes = chains needed --
//                   Little's law again (Chapter 2), inside one core.
//   4. all cores  : the best K on 1..ncpu threads.
//
//   cc -O3 peak_flops.c -lpthread -o peak_flops
#include "bench.h"

#if !defined(__aarch64__)
int main(void) {
    printf("peak_flops.c is arm64-only (Neon inline asm). Run it on Apple Silicon or\n"
           "an arm64 Linux container; see README.md.\n");
    return 0;
}
#else
#include <arm_neon.h>
#include <pthread.h>
#include <unistd.h>

#define R4(s) s s s s
#define R16(s) R4(R4(s))
#define R64(s) R4(R16(s))

// ---------------------------------------------------------------- 1. clock
static double adds_per_sec(void) {
    const long iters = 2000000;  // x 64 adds = 1.28e8 dependent adds
    double best = 0;
    for (int t = 0; t < 5; ++t) {
        long x = 0;
        double t0 = now_sec();
        for (long i = 0; i < iters; ++i) __asm__ volatile(R64("add %0, %0, #1\n\t") : "+r"(x));
        double dt = now_sec() - t0;
        g_sink += (double)x;
        double r = (double)iters * 64 / dt;
        if (r > best) best = r;
    }
    return best;
}

// ---------------------------------------------------------------- 2. latency
static double fadd_chain_per_sec(void) {
    const long iters = 1000000;
    double best = 0;
    for (int t = 0; t < 5; ++t) {
        float32x4_t a = vdupq_n_f32(0.0f), b = vdupq_n_f32(1e-7f);
        double t0 = now_sec();
        for (long i = 0; i < iters; ++i)
            __asm__ volatile(R64("fadd %0.4s, %0.4s, %1.4s\n\t") : "+w"(a) : "w"(b));
        double dt = now_sec() - t0;
        g_sink += vgetq_lane_f32(a, 0);
        double r = (double)iters * 64 / dt;
        if (r > best) best = r;
    }
    return best;
}
static double fmla_chain_per_sec(void) {
    const long iters = 1000000;
    double best = 0;
    for (int t = 0; t < 5; ++t) {
        float32x4_t a = vdupq_n_f32(0.0f), b = vdupq_n_f32(1e-4f), c = vdupq_n_f32(1e-3f);
        double t0 = now_sec();
        for (long i = 0; i < iters; ++i)
            __asm__ volatile(R64("fmla %0.4s, %1.4s, %2.4s\n\t") : "+w"(a) : "w"(b), "w"(c));
        double dt = now_sec() - t0;
        g_sink += vgetq_lane_f32(a, 0);
        double r = (double)iters * 64 / dt;
        if (r > best) best = r;
    }
    return best;
}

// ---------------------------------------------------------------- 3. throughput
// K independent accumulators. Each loop iteration issues 8*K fmla, each doing 4
// lanes x 2 FLOPs = 8 FLOPs. Every fmla is its own `asm volatile` statement:
// with plain intrinsics the optimizer is allowed to (and does) prove that some
// accumulators never change value or are never used, and deletes them -- a
// classic microbenchmark trap. asm volatile cannot be deleted or merged; the
// out-of-order hardware is still free to overlap the K chains, which is the
// thing we are measuring.
static volatile float g_init = 1e-3f;
#define DEF_FMA(K)                                                                    \
    static double fma_k##K(long iters) {                                              \
        float32x4_t acc[K];                                                           \
        for (int k = 0; k < K; ++k) acc[k] = vdupq_n_f32(g_init * (float)k);          \
        const float32x4_t b = vdupq_n_f32(g_init), c = vdupq_n_f32(g_init);           \
        for (long i = 0; i < iters; ++i) {                                            \
            for (int u = 0; u < 8; ++u)                                               \
                for (int k = 0; k < K; ++k)                                           \
                    __asm__ volatile("fmla %0.4s, %1.4s, %2.4s" : "+w"(acc[k]) : "w"(b), "w"(c)); \
        }                                                                             \
        float32x4_t s = acc[0];                                                       \
        for (int k = 1; k < K; ++k) s = vaddq_f32(s, acc[k]);                         \
        g_sink += vaddvq_f32(s);                                                      \
        return (double)iters * 8 * K * 8; /* FLOPs */                                 \
    }
DEF_FMA(1) DEF_FMA(2) DEF_FMA(4) DEF_FMA(8) DEF_FMA(12) DEF_FMA(16) DEF_FMA(24)

typedef double (*fma_fn)(long);
static const fma_fn FMA_FNS[] = {fma_k1, fma_k2, fma_k4, fma_k8, fma_k12, fma_k16, fma_k24};
static const int FMA_KS[] = {1, 2, 4, 8, 12, 16, 24};
#define NFMA 7

static double gflops_single(fma_fn f) {
    double best = 0;
    for (int t = 0; t < 5; ++t) {
        double t0 = now_sec();
        double flops = f(2000000);
        double r = flops / (now_sec() - t0) * 1e-9;
        if (r > best) best = r;
    }
    return best;
}

// ---------------------------------------------------------------- 4. all cores
typedef struct {
    fma_fn f;
    double flops, secs;
} job_t;
static void* worker(void* p) {
    job_t* j = (job_t*)p;
    double t0 = now_sec();
    j->flops = j->f(20000000);  // ~0.3-0.5 s per thread
    j->secs = now_sec() - t0;
    return NULL;
}
static double gflops_threads(fma_fn f, int nt) {
    pthread_t th[256];
    job_t jobs[256];
    double t0 = now_sec();
    for (int i = 0; i < nt; ++i) {
        jobs[i].f = f;
        pthread_create(&th[i], NULL, worker, &jobs[i]);
    }
    double total = 0;
    for (int i = 0; i < nt; ++i) {
        pthread_join(th[i], NULL);
        total += jobs[i].flops;
    }
    return total / (now_sec() - t0) * 1e-9;  // wall clock: the slowest thread counts
}

int main(void) {
    double hz = adds_per_sec();
    printf("1. dependent integer adds: %.2f G/s  => estimated clock %.2f GHz\n", hz * 1e-9,
           hz * 1e-9);
    printf("   (assumes 1-cycle add latency; the OS decides which core type this ran on)\n\n");

    double fadd = fadd_chain_per_sec(), fmla = fmla_chain_per_sec();
    printf("2. latency: vector fadd %.2f cycles, vector fmla %.2f cycles\n\n", hz / fadd,
           hz / fmla);

    printf("3. one thread, K independent 4-lane FMA chains:\n");
    printf("   %4s %10s %12s\n", "K", "GFLOP/s", "FLOP/cycle");
    double best = 0;
    int best_i = 0;
    for (int i = 0; i < NFMA; ++i) {
        double g = gflops_single(FMA_FNS[i]);
        printf("   %4d %10.1f %12.2f\n", FMA_KS[i], g, g * 1e9 / hz);
        if (g > best) {
            best = g;
            best_i = i;
        }
    }
    double fpc = best * 1e9 / hz;
    printf("   best: %.1f FLOP/cycle = 4 lanes x 2 FLOP x %.2f FMA pipes\n\n", fpc, fpc / 8.0);

    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    printf("4. K=%d on T threads (wall-clock aggregate):\n", FMA_KS[best_i]);
    int ts[8], nts = 0;
    int cand[] = {1, 2, 4, 6, 8, 12, 16, 18};
    for (int i = 0; i < 8; ++i)
        if (cand[i] <= ncpu) ts[nts++] = cand[i];
    if (ts[nts - 1] != ncpu && nts < 8) ts[nts++] = ncpu;
    for (int i = 0; i < nts; ++i) {
        double g = gflops_threads(FMA_FNS[best_i], ts[i]);
        printf("   T=%2d  %8.1f GFLOP/s\n", ts[i], g);
    }
    return 0;
}
#endif
