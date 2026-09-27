// threads_sum.c -- the same parallel sum with pthreads and with OpenMP, plus
// the classic false-sharing trap.
//
//   Part 1: sum of 128M floats (512 MB, far bigger than any cache) on
//           T = 1, 2, 4, ... threads with pthreads. Reports GB/s. The curve
//           flattens when DRAM bandwidth, not cores, becomes the limit --
//           a memory-bound kernel (AI = 1 FLOP / 4 bytes) hitting the memory roof.
//   Part 2: the identical computation as one OpenMP line
//           (`#pragma omp parallel for simd reduction(+:s)`), if built with
//           -fopenmp. Apple clang has no OpenMP runtime, so on a stock Mac
//           this part is compiled out and says so.
//   Part 3: T threads each increment their OWN counter. Packed counters share
//           a cache line; padded ones do not. Same work, very different time.
//
//   Mac:   cc -O3 -fopenmp-simd threads_sum.c -lpthread -lm -o threads_sum
//   Linux: gcc -O3 -fopenmp     threads_sum.c -lpthread -lm -o threads_sum_omp
#include "bench.h"

#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define N ((size_t)1 << 27)  // 128M floats = 512 MB

// Per-thread inner loop. The simd pragma gives the compiler permission to
// reorder the float adds (see §1.5), so each thread runs a vectorized loop.
static double sum_range(const float* a, size_t lo, size_t hi) {
    float s = 0.0f;
#pragma omp simd reduction(+ : s)
    for (size_t i = lo; i < hi; ++i) s += a[i];
    return (double)s;
}

// ------------------------------------------------------------ Part 1: pthreads
typedef struct {
    const float* a;
    size_t lo, hi;
    double result;
} sum_job;

static void* sum_worker(void* p) {
    sum_job* j = (sum_job*)p;
    j->result = sum_range(j->a, j->lo, j->hi);
    return NULL;
}

static double sum_pthreads(const float* a, size_t n, int nt) {
    pthread_t th[64];
    sum_job jobs[64];
    for (int t = 0; t < nt; ++t) {
        // Contiguous chunks: each thread streams its own region, which is what
        // hardware prefetchers like. (Compare the GPU, Ch. 8, where
        // *interleaved* per-thread access is the fast pattern.)
        jobs[t].a = a;
        jobs[t].lo = n * (size_t)t / (size_t)nt;
        jobs[t].hi = n * (size_t)(t + 1) / (size_t)nt;
        pthread_create(&th[t], NULL, sum_worker, &jobs[t]);
    }
    double s = 0.0;
    for (int t = 0; t < nt; ++t) {
        pthread_join(th[t], NULL);
        s += jobs[t].result;  // combine partials in a fixed order: deterministic
    }
    return s;
}

// ------------------------------------------------------------ Part 2: OpenMP
#ifdef _OPENMP
static double sum_openmp(const float* a, size_t n, int nt) {
    double s = 0.0;
    // Everything Part 1 did by hand -- chunking, thread creation, per-thread
    // partials, the combine -- in one pragma. `simd` vectorizes each chunk.
    // Note the combine order is up to the runtime, so results may differ in the
    // last bits between runs; the pthreads version above is deterministic.
#pragma omp parallel for simd reduction(+ : s) num_threads(nt) schedule(static)
    for (size_t i = 0; i < n; ++i) s += a[i];
    return s;
}
#endif

// ------------------------------------------------------------ Part 3: false sharing
// Each thread does ITERS relaxed atomic increments of ITS OWN counter -- the
// shape of a per-thread statistics array. No data is shared, but when counters
// sit in the same cache line, every increment must pull the line away from the
// core that last wrote it (coherence traffic). Padding each counter to its own
// line removes that. (With plain non-atomic volatile stores the effect on this
// Mac was only ~1.2x -- stores are buffered and merged -- which is why the demo
// uses atomics: that is where false sharing hurts in real code.)
#define ITERS 20000000L
#define LINE 128  // Apple M-series cache line is 128 B (sysctl hw.cachelinesize); x86 is 64

typedef struct {
    _Atomic long* slot;
} fs_job;
static void* fs_worker(void* p) {
    _Atomic long* c = ((fs_job*)p)->slot;
    for (long i = 0; i < ITERS; ++i) atomic_fetch_add_explicit(c, 1, memory_order_relaxed);
    return NULL;
}
static double false_sharing(int nt, size_t stride_bytes) {
    char* buf = NULL;
    if (posix_memalign((void**)&buf, LINE, (size_t)nt * stride_bytes + LINE) != 0) exit(1);
    pthread_t th[64];
    fs_job jobs[64];
    for (int t = 0; t < nt; ++t) {
        jobs[t].slot = (_Atomic long*)(buf + (size_t)t * stride_bytes);
        atomic_store(jobs[t].slot, 0);
    }
    double t0 = now_sec();
    for (int t = 0; t < nt; ++t) pthread_create(&th[t], NULL, fs_worker, &jobs[t]);
    for (int t = 0; t < nt; ++t) pthread_join(th[t], NULL);
    double dt = now_sec() - t0;
    for (int t = 0; t < nt; ++t)
        if (atomic_load(jobs[t].slot) != ITERS) {
            printf("false_sharing: WRONG count\n");
            exit(1);
        }
    free(buf);
    return dt;
}

static double best_of(double (*f)(const float*, size_t, int), const float* a, int nt,
                      double* result) {
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        double t0 = now_sec();
        *result = f(a, N, nt);
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

int main(void) {
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    float* a = alloc_floats(N);
    fill_random(a, N, 4242u);

    double ref = 0.0, scale = 0.0;
    for (size_t i = 0; i < N; ++i) {
        ref += (double)a[i];
        scale += fabs((double)a[i]);
    }
    const double tol = 1e-6 * scale;  // loose enough for any summation order, see simd_dot.c

    printf("sum of %zu floats (%.0f MB), %d logical CPUs\n\n", N, N * 4.0 / 1e6, ncpu);
    printf("Part 1: pthreads\n  %3s %9s %9s %8s\n", "T", "time", "GB/s", "speedup");
    int ts[] = {1, 2, 4, 6, 8, 12, 16, 18, 24, 32};
    double t1 = 0;
    int ok = 1;
    for (int k = 0; k < 10 && ts[k] <= ncpu; ++k) {
        double s, t = best_of(sum_pthreads, a, ts[k], &s);
        if (k == 0) t1 = t;
        int good = fabs(s - ref) <= tol;
        ok &= good;
        printf("  %3d %7.2f ms %9.1f %7.2fx %s\n", ts[k], t * 1e3, N * 4.0 / t * 1e-9, t1 / t,
               good ? "" : "WRONG");
    }

#ifdef _OPENMP
    printf("\nPart 2: OpenMP (omp_get_max_threads() = %d)\n  %3s %9s %9s %8s\n",
           omp_get_max_threads(), "T", "time", "GB/s", "speedup");
    for (int k = 0; k < 10 && ts[k] <= ncpu; ++k) {
        double s, t = best_of(sum_openmp, a, ts[k], &s);
        int good = fabs(s - ref) <= tol;
        ok &= good;
        printf("  %3d %7.2f ms %9.1f %7.2fx %s\n", ts[k], t * 1e3, N * 4.0 / t * 1e-9, t1 / t,
               good ? "" : "WRONG");
    }
#else
    printf("\nPart 2: OpenMP -- not built with -fopenmp (stock Apple clang has no OpenMP\n"
           "        runtime). Build with gcc -fopenmp in the gcc:15 container; see README.\n");
#endif

    int nt = ncpu < 8 ? ncpu : 8;
    printf("\nPart 3: false sharing, %d threads x %ld relaxed atomic increments each\n", nt,
           ITERS);
    const size_t strides[3] = {sizeof(long), 64, LINE};
    const char* what[3] = {"packed (8 B apart, same line)", "64 B apart (x86 line size)",
                           "128 B apart (own Apple line)"};
    double t_pad = false_sharing(nt, LINE);
    for (int k = 0; k < 3; ++k) {
        double t = false_sharing(nt, strides[k]);
        printf("  %-32s %8.1f ms  (%.1fx the padded time)\n", what[k], t * 1e3, t / t_pad);
    }
    printf("\n%s\n", ok ? "all sums match the double-precision reference" : "SUM MISMATCH");
    free(a);
    return ok ? 0 : 1;
}
