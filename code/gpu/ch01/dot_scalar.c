// dot_scalar.c -- the scalar reference kernels for simd_dot.c.
//
// This file is compiled SEPARATELY with vectorization switched off
// (clang: -fno-vectorize -fno-slp-vectorize, gcc: -fno-tree-vectorize), so the
// two functions below really are one-lane-at-a-time code. Without that, clang's
// SLP vectorizer would happily pack dot_scalar_ilp4's four accumulators into
// one 4-lane Neon register and the "scalar" baseline would secretly be SIMD.
#include <stddef.h>

// One accumulator: every iteration's add depends on the previous one, so the
// loop runs at one FMA per FMA *latency* (~4 cycles on current big cores), not
// one per cycle. This is the "latency-bound single stream" of Chapter 2.
float dot_scalar(const float* x, const float* y, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}

// Four independent accumulators: four dependency chains in flight, so the
// out-of-order core can overlap them. Same instruction count, ~up to 4x faster.
// This is instruction-level parallelism (ILP) -- the CPU's version of the
// "warps x ILP" latency-hiding budget you will meet in Chapter 11.
// NOTE: it sums in a different order, so the float result differs slightly.
float dot_scalar_ilp4(const float* x, const float* y, size_t n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += x[i + 0] * y[i + 0];
        s1 += x[i + 1] * y[i + 1];
        s2 += x[i + 2] * y[i + 2];
        s3 += x[i + 3] * y[i + 3];
    }
    for (; i < n; ++i) s0 += x[i] * y[i];
    return (s0 + s1) + (s2 + s3);
}
