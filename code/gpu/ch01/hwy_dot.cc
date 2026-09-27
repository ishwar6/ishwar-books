// hwy_dot.cc -- the dot product once, in Google Highway, compiled for many
// SIMD targets in one binary and dispatched at run time to the best one the
// CPU actually has (Neon / SVE / SVE2 on Arm; SSE4 / AVX2 / AVX-512 on x86).
//
// The same source that runs as 4-lane Neon on an Apple M5 runs as 8-lane AVX2
// or 16-lane AVX-512 on x86: the width is `hn::Lanes(d)`, a run-time value,
// never a constant in the source. That is the idea behind "length-agnostic"
// SIMD -- and a first step toward the GPU model, where the source never
// mentions the width at all.
//
// Build (needs a checkout of https://github.com/google/highway; tested with
// Highway 1.4.0 at commit 32ae733):
//   git clone --depth 1 https://github.com/google/highway.git
//   c++ -O3 -std=c++17 -I. -I highway hwy_dot.cc highway/hwy/targets.cc \
//       highway/hwy/abort.cc highway/hwy/per_target.cc highway/hwy/print.cc -o hwy_dot
// (or: make hwy HWY=highway)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <vector>

// ---- Highway's multi-target boilerplate. foreach_target.h re-includes THIS
// file once per enabled target, each time with HWY_NAMESPACE set to a
// different name (N_NEON, N_SVE, N_AVX2, ...), so the code between
// HWY_BEFORE_NAMESPACE / HWY_AFTER_NAMESPACE is compiled once per target.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy_dot.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace dot {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

float DotHwy(const float* HWY_RESTRICT x, const float* HWY_RESTRICT y, size_t n) {
    const hn::ScalableTag<float> d;  // "a full vector of floats, whatever that is here"
    const size_t N = hn::Lanes(d);   // 4 on Neon, 8 on AVX2, 16 on AVX-512, varies on SVE
    // Four accumulators for the same ILP reason as the Neon version.
    auto a0 = hn::Zero(d), a1 = hn::Zero(d), a2 = hn::Zero(d), a3 = hn::Zero(d);
    size_t i = 0;
    for (; i + 4 * N <= n; i += 4 * N) {
        // LoadU = unaligned load (only element alignment required).
        // MulAdd(a, b, c) = a*b + c, fused where the target has FMA.
        a0 = hn::MulAdd(hn::LoadU(d, x + i + 0 * N), hn::LoadU(d, y + i + 0 * N), a0);
        a1 = hn::MulAdd(hn::LoadU(d, x + i + 1 * N), hn::LoadU(d, y + i + 1 * N), a1);
        a2 = hn::MulAdd(hn::LoadU(d, x + i + 2 * N), hn::LoadU(d, y + i + 2 * N), a2);
        a3 = hn::MulAdd(hn::LoadU(d, x + i + 3 * N), hn::LoadU(d, y + i + 3 * N), a3);
    }
    float s = hn::ReduceSum(d, hn::Add(hn::Add(a0, a1), hn::Add(a2, a3)));
    for (; i < n; ++i) s += x[i] * y[i];  // scalar tail
    return s;
}

// Report which target the dispatcher picked and its lane count.
const char* TargetInfo(size_t* lanes) {
    *lanes = hn::Lanes(hn::ScalableTag<float>());
    return hwy::TargetName(HWY_TARGET);
}

}  // namespace HWY_NAMESPACE
}  // namespace dot
HWY_AFTER_NAMESPACE();

// ---- Code below is compiled only once (not per target).
#if HWY_ONCE
namespace dot {
HWY_EXPORT(DotHwy);      // builds a table of per-target function pointers
HWY_EXPORT(TargetInfo);
}  // namespace dot

static double now_sec() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

// Scalar reference, deliberately a single dependency chain (same as
// dot_scalar in simd_dot.c) so the speedup column is comparable.
static float DotScalar(const float* x, const float* y, size_t n) {
    float s = 0.0f;
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (size_t i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}

static volatile double g_sink;

template <class F>
static double TimeIt(F f, size_t n) {
    size_t reps = 1;
    for (;;) {
        double t0 = now_sec();
        for (size_t r = 0; r < reps; ++r) g_sink += f();
        if (now_sec() - t0 > 0.05) break;
        reps *= 2;
    }
    double best = 1e30;
    for (int t = 0; t < 5; ++t) {
        double t0 = now_sec();
        for (size_t r = 0; r < reps; ++r) g_sink += f();
        best = fmin(best, (now_sec() - t0) / reps);
    }
    (void)n;
    return best;
}

int main() {
    size_t lanes = 0;
    const char* target = HWY_DYNAMIC_DISPATCH(dot::TargetInfo)(&lanes);
    printf("Highway dispatched to target %s (%zu float lanes)\n", target, lanes);

    const size_t nmax = size_t(1) << 25;
    std::vector<float> x(nmax), y(nmax);
    unsigned s = 12345u;
    for (size_t i = 0; i < nmax; ++i) {  // same xorshift stream idea as bench.h
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        x[i] = float(double(s) / 2147483648.0 - 1.0);
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        y[i] = float(double(s) / 2147483648.0 - 1.0);
    }

    // Verify at awkward sizes first (tails, partial vectors).
    bool ok = true;
    for (size_t n : {size_t(1), size_t(37), size_t(4099), size_t(1000003)}) {
        double ref = 0, scale = 0;
        for (size_t i = 0; i < n; ++i) {
            ref += double(x[i]) * y[i];
            scale += fabs(double(x[i]) * y[i]);
        }
        double got = HWY_DYNAMIC_DISPATCH(dot::DotHwy)(x.data(), y.data(), n);
        if (fabs(got - ref) > 1e-6 * scale + 1e-6) {
            printf("VERIFY FAIL at n=%zu: got %.6f ref %.6f\n", n, got, ref);
            ok = false;
        }
    }
    printf("verification at n = 1, 37, 4099, 1000003: %s\n", ok ? "PASS" : "FAIL");

    const size_t sizes[3] = {4096, 65536, nmax};
    const char* where[3] = {"L1", "L2", "DRAM"};
    printf("\n  %-6s %10s %12s %12s %9s\n", "size", "n", "scalar us", "highway us", "speedup");
    for (int k = 0; k < 3; ++k) {
        size_t n = sizes[k];
        double ts = TimeIt([&] { return DotScalar(x.data(), y.data(), n); }, n);
        double th = TimeIt([&] { return HWY_DYNAMIC_DISPATCH(dot::DotHwy)(x.data(), y.data(), n); }, n);
        printf("  %-6s %10zu %12.2f %12.2f %8.2fx  (%.1f GFLOP/s)\n", where[k], n, ts * 1e6,
               th * 1e6, ts / th, 2.0 * n / th * 1e-9);
    }
    return ok ? 0 : 1;
}
#endif  // HWY_ONCE
