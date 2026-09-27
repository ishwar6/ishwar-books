// vec_remarks.c -- loops chosen to make the auto-vectorizer explain itself.
// Nothing here is run; compile it and READ THE REMARKS:
//
//   clang: cc  -O3 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
//              -Rpass-analysis=loop-vectorize -c vec_remarks.c -o /dev/null
//   gcc:   gcc -O3 -fopt-info-vec-all -c vec_remarks.c -o /dev/null
//
// then again with -ffast-math (or -fassociative-math) and compare what changes.
// (`make remarks` in this directory runs the right command for $(CC).)
#include <stddef.h>

// 1. May x and y overlap? The compiler cannot know, so it either gives up or
//    emits a runtime overlap check plus two versions of the loop.
void saxpy_alias(float a, const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

// 2. `restrict` is your promise that they do not overlap: no runtime check.
//    (Break the promise and the behaviour is undefined -- wrong answers, not errors.)
void saxpy_restrict(float a, const float* restrict x, float* restrict y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

// 3. A float reduction. Vectorizing it means summing in a different order,
//    which changes the rounding, which the compiler may not do unless you allow
//    it (-ffast-math, -fassociative-math, or `#pragma omp simd reduction`).
float sum_float(const float* x, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += x[i];
    return s;
}

// 4. The same reduction on integers: integer + IS associative, so it vectorizes.
int sum_int(const int* x, size_t n) {
    int s = 0;
    for (size_t i = 0; i < n; ++i) s += x[i];
    return s;
}

// 5. A true loop-carried dependence (a running prefix sum): element i needs
//    element i-1 of THIS loop. No reordering permission can fix that; it needs
//    a different algorithm (parallel scan -- Chapter 14).
void prefix_sum(float* a, size_t n) {
    for (size_t i = 1; i < n; ++i) a[i] = a[i - 1] + a[i];
}

// 6. An early exit: the trip count depends on the data.
long find_first_negative(const float* x, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (x[i] < 0.0f) return (long)i;
    return -1;
}

// 7. A scatter with possible index collisions: two lanes could hit the same
//    bin in one vector step. (GPUs have the same problem; Chapter 15 solves
//    it with atomics and privatization.)
void histogram(const int* idx, int* bins, size_t n) {
    for (size_t i = 0; i < n; ++i) bins[idx[i]] += 1;
}
