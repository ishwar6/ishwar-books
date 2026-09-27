// bench.h -- tiny timing / data helpers shared by the Chapter 1 C programs.
//
// Deliberately plain C so it compiles with Apple clang, gcc, and g++ alike.
// Why a helper at all: every program in this chapter follows the same rule as
// the CUDA code later in the book -- verify against a reference FIRST, then
// time, and report the best of several repetitions (the minimum is the run
// least disturbed by the OS, which is what we want when comparing code paths).
#ifndef CH01_BENCH_H
#define CH01_BENCH_H

// With -std=c11, glibc hides POSIX functions (clock_gettime, posix_memalign)
// unless asked. _DEFAULT_SOURCE asks; macOS ignores it. Must come before any
// system header, which is why bench.h is the first include in every file.
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// Deterministic xorshift so every run (and every machine) sees the same data.
static inline uint32_t xorshift32(uint32_t* s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *s = x;
}

// Uniform in [-1, 1).
static inline void fill_random(float* p, size_t n, uint32_t seed) {
    uint32_t s = seed ? seed : 1u;
    for (size_t i = 0; i < n; ++i)
        p[i] = (float)((double)xorshift32(&s) / 2147483648.0 - 1.0);
}

// 64-byte aligned allocation: a cache line on the machines we care about, and
// enough for any SIMD width up to AVX-512. Exits on failure (these are demos).
static inline float* alloc_floats(size_t n) {
    void* p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(float)) != 0) {
        fprintf(stderr, "allocation of %zu floats failed\n", n);
        exit(1);
    }
    return (float*)p;
}

// Results go through a volatile so the compiler cannot delete a timed loop
// whose answer is otherwise unused.
static volatile double g_sink;

#endif
