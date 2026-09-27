// simd_dot.c -- one dot product, six ways, on one CPU core.
//
//   scalar        1 accumulator, vectorization OFF      (dot_scalar.c)
//   scalar-ilp4   4 accumulators, vectorization OFF     (dot_scalar.c)
//   autovec-plain the plain loop at -O3                 (expect: NOT vectorized)
//   autovec-simd  the plain loop + `#pragma omp simd reduction(+:s)`
//   neon          Arm Neon intrinsics, 4 x float32x4 accumulators   (arm64 only)
//   avx2          x86 AVX2+FMA intrinsics, 4 x __m256 accumulators  (x86-64 only)
//
// Each is verified against a double-precision reference, then timed at three
// sizes: one that fits in L1, one that fits in L2, and one that must stream
// from DRAM. The point of the three sizes is the roofline preview in §1.9: SIMD
// speedups are real when the data is on-chip and collapse when it is not.
//
// Build (see Makefile / README.md for the exact flags per compiler):
//   Mac:    make simd_dot            (Apple clang, Neon)
//   Linux:  make CC=gcc simd_dot     (gcc:15 arm64 = Neon, amd64 = AVX2)
#include "bench.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define HAVE_NEON 1
#endif
#if defined(__x86_64__)
#include <immintrin.h>
#define HAVE_X86 1
#endif

float dot_scalar(const float* x, const float* y, size_t n);       // dot_scalar.c
float dot_scalar_ilp4(const float* x, const float* y, size_t n);  // dot_scalar.c

// The loop a Python programmer would write. At -O3 WITHOUT -ffast-math the
// compiler may not vectorize it: vectorizing means summing in a different
// order, and float addition is not associative, so the answer would change.
// The compiler is not allowed to change your answer. (See vec_remarks.c.)
static float dot_autovec_plain(const float* restrict x, const float* restrict y, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}

// Same loop, but the pragma tells the compiler "reordering this reduction is
// fine by me". Needs -fopenmp-simd (NOT full OpenMP: no runtime library, which
// is why it works with Apple clang out of the box).
static float dot_autovec_simd(const float* restrict x, const float* restrict y, size_t n) {
    float s = 0.0f;
#pragma omp simd reduction(+ : s)
    for (size_t i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}

#ifdef HAVE_NEON
// 128-bit Neon = 4 floats per register. Four accumulators = 16 floats in
// flight, so four independent FMA chains can overlap (same idea as ilp4, but
// each "lane" is now a 4-wide vector). vfmaq_f32(a, b, c) = a + b*c, fused.
static float dot_neon(const float* x, const float* y, size_t n) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(x + i + 0), vld1q_f32(y + i + 0));
        a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4), vld1q_f32(y + i + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(x + i + 8), vld1q_f32(y + i + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(x + i + 12), vld1q_f32(y + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));  // horizontal add
    for (; i < n; ++i) s += x[i] * y[i];                                     // scalar tail
    return s;
}
#endif

#ifdef HAVE_X86
// 256-bit AVX2 = 8 floats per register. The target attribute lets this one
// function use AVX2/FMA while the rest of the file stays baseline x86-64, and
// main() only calls it after __builtin_cpu_supports() says the CPU has them.
__attribute__((target("avx2,fma"))) static float dot_avx2(const float* x, const float* y,
                                                          size_t n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 0), _mm256_loadu_ps(y + i + 0), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(y + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(y + i + 24), a3);
    }
    __m256 v = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    // Horizontal sum of 8 lanes: fold 256 -> 128 -> 64 -> 32 bits.
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x1));
    float s = _mm_cvtss_f32(h);
    for (; i < n; ++i) s += x[i] * y[i];
    return s;
}
#endif

typedef float (*dot_fn)(const float*, const float*, size_t);

// Reference in double, plus the scale sum|x_i*y_i| used for the tolerance:
// different summation orders legitimately differ by ~n * eps * sum|x*y|.
static double dot_ref(const float* x, const float* y, size_t n, double* scale) {
    double s = 0.0, a = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s += (double)x[i] * (double)y[i];
        a += fabs((double)x[i] * (double)y[i]);
    }
    *scale = a;
    return s;
}

// Best-of-5 seconds per call. Each trial repeats the call enough times to
// last ~50 ms so the clock resolution does not matter.
static double time_dot(dot_fn f, const float* x, const float* y, size_t n) {
    size_t reps = 1;
    for (;;) {
        double t0 = now_sec();
        for (size_t r = 0; r < reps; ++r) g_sink += f(x, y, n);
        if (now_sec() - t0 > 0.05) break;
        reps *= 2;
    }
    double best = 1e30;
    for (int t = 0; t < 5; ++t) {
        double t0 = now_sec();
        for (size_t r = 0; r < reps; ++r) g_sink += f(x, y, n);
        double dt = (now_sec() - t0) / (double)reps;
        if (dt < best) best = dt;
    }
    return best;
}

int main(void) {
    struct {
        const char* name;
        dot_fn f;
    } impls[8];
    int m = 0;
    impls[m].name = "scalar";        impls[m++].f = dot_scalar;
    impls[m].name = "scalar-ilp4";   impls[m++].f = dot_scalar_ilp4;
    impls[m].name = "autovec-plain"; impls[m++].f = dot_autovec_plain;
    impls[m].name = "autovec-simd";  impls[m++].f = dot_autovec_simd;
#ifdef HAVE_NEON
    impls[m].name = "neon";          impls[m++].f = dot_neon;
#endif
#ifdef HAVE_X86
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        impls[m].name = "avx2";      impls[m++].f = dot_avx2;
    } else {
        printf("(CPU reports no AVX2+FMA: skipping the avx2 variant)\n");
    }
#endif

    // 4K floats x 2 arrays = 32 KB   -> L1-resident on Apple P-cores (128 KB L1d)
    // 64K floats x 2       = 512 KB  -> L2-resident (16 MB L2 on M5 Pro super cluster)
    // 32M floats x 2       = 256 MB  -> DRAM
    const size_t sizes[3] = {4096, 65536, (size_t)1 << 25};
    const char* where[3] = {"L1", "L2", "DRAM"};

    float* x = alloc_floats(sizes[2]);
    float* y = alloc_floats(sizes[2]);
    fill_random(x, sizes[2], 12345u);
    fill_random(y, sizes[2], 67890u);

    // Verification pass on awkward sizes first, so every scalar tail and every
    // partial vector is exercised (the timing sizes are all multiples of 32).
    // Tolerance 1e-6 * sum|x*y|: far above the rounding differences between
    // summation orders at these n, far below a single dropped or doubled term
    // (each term is ~0.25 on average, the tolerance at n=4099 is ~0.001).
    int all_ok = 1;
    const size_t vsizes[4] = {1, 37, 4099, 1000003};
    for (int vi = 0; vi < 4; ++vi) {
        double scale, ref = dot_ref(x, y, vsizes[vi], &scale);
        for (int k = 0; k < m; ++k) {
            double err = fabs((double)impls[k].f(x, y, vsizes[vi]) - ref);
            if (err > 1e-6 * scale + 1e-6) {
                printf("VERIFY FAIL: %s at n=%zu: err %.3g\n", impls[k].name, vsizes[vi], err);
                all_ok = 0;
            }
        }
    }
    printf("verification at n = 1, 37, 4099, 1000003: %s\n", all_ok ? "PASS" : "FAIL");

    for (int si = 0; si < 3; ++si) {
        size_t n = sizes[si];
        double scale, ref = dot_ref(x, y, n, &scale);
        printf("\nn = %zu floats (%s-resident, %.1f KB touched per call)\n", n, where[si],
               2.0 * n * sizeof(float) / 1024.0);
        printf("  %-14s %10s %10s %10s %9s\n", "variant", "time/call", "GFLOP/s", "GB/s", "speedup");
        double t_scalar = 0.0;
        for (int k = 0; k < m; ++k) {
            // 1) verify
            double got = impls[k].f(x, y, n);
            double err = fabs(got - ref);
            double tol = 1e-6 * scale + 1e-6;  // same rule as the verification pass
            int ok = err <= tol;
            all_ok &= ok;
            // 2) time
            double t = time_dot(impls[k].f, x, y, n);
            if (k == 0) t_scalar = t;
            double flops = 2.0 * (double)n;               // one mul + one add per element
            double bytes = 2.0 * (double)n * sizeof(float);  // read x and y once
            printf("  %-14s %8.2f us %10.2f %10.2f %8.2fx %s\n", impls[k].name, t * 1e6,
                   flops / t * 1e-9, bytes / t * 1e-9, t_scalar / t, ok ? "" : "  <-- WRONG");
        }
    }
    printf("\n%s\n", all_ok ? "all variants match the double-precision reference"
                            : "SOME VARIANT FAILED VERIFICATION");
    free(x);
    free(y);
    return all_ok ? 0 : 1;
}
