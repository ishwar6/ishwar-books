// cuda_utils.cuh -- shared helpers for every program in this book.
//
// Include this first. It gives you:
//   CUDA_CHECK(call)        -- abort with file:line on any CUDA API error
//   CUDA_CHECK_KERNEL()     -- catch launch + execution errors after a <<<>>>
//   time_kernel(...)        -- correct event-based timing with warmup
//   Bench                   -- reports ms / GB/s / GFLOP/s / % of peak
//   verify(...)             -- compare against a host reference with tolerance
//   DeviceInfo              -- peak bandwidth, machine balance, occupancy limits
//
// Deliberately header-only and dependency-free so every .cu file compiles standalone:
//   nvcc -O3 -arch=sm_75 -lineinfo -I common file.cu -o prog

#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ---------------------------------------------------------------- error checks

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "\nCUDA error at %s:%d\n  %s\n  %s\n", __FILE__, \
                    __LINE__, cudaGetErrorName(err_),                        \
                    cudaGetErrorString(err_));                               \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Launch errors are reported synchronously by cudaGetLastError(); execution
// errors (bad address, misaligned access) only surface at a synchronization
// point, which is why the sync is here. It serializes -- keep it out of timing
// loops. Define NDEBUG to drop the sync in release builds.
#ifdef NDEBUG
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#else
#define CUDA_CHECK_KERNEL()                   \
    do {                                      \
        CUDA_CHECK(cudaGetLastError());       \
        CUDA_CHECK(cudaDeviceSynchronize());  \
    } while (0)
#endif

#define CEIL_DIV(n, d) (((n) + (d) - 1) / (d))

// ---------------------------------------------------------------- device info

// CUDA 13.0 REMOVED cudaDeviceProp::clockRate and ::memoryClockRate. They are
// only reachable through cudaDeviceGetAttribute now. Colab ships CUDA 13.x, so
// touching the old fields is a compile error there. Always go through this.
inline int dev_attr(cudaDeviceAttr attr, int dev = 0) {
    int v = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&v, attr, dev));
    return v;
}

struct DeviceInfo {
    cudaDeviceProp prop{};
    int            device = 0;

    int sm_clock_khz  = 0;   // cudaDevAttrClockRate
    int mem_clock_khz = 0;   // cudaDevAttrMemoryClockRate
    int mem_bus_bits  = 0;   // cudaDevAttrGlobalMemoryBusWidth

    // Derived quantities -- these are the numbers you actually reason with.
    double peak_bandwidth_gbps = 0.0;  // memory clock x bus width x 2 (DDR)
    int    max_resident_threads = 0;
    int    regs_per_thread_at_full_occupancy = 0;

    static DeviceInfo query(int dev = 0) {
        DeviceInfo d;
        d.device = dev;
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaGetDeviceProperties(&d.prop, dev));

        d.sm_clock_khz  = dev_attr(cudaDevAttrClockRate, dev);
        d.mem_clock_khz = dev_attr(cudaDevAttrMemoryClockRate, dev);
        d.mem_bus_bits  = dev_attr(cudaDevAttrGlobalMemoryBusWidth, dev);

        // clock in kHz; bus in bits; x2 because DDR.
        d.peak_bandwidth_gbps =
            2.0 * d.mem_clock_khz * 1e3 * (d.mem_bus_bits / 8.0) / 1e9;

        d.max_resident_threads =
            d.prop.multiProcessorCount * d.prop.maxThreadsPerMultiProcessor;
        d.regs_per_thread_at_full_occupancy =
            d.prop.regsPerMultiprocessor / d.prop.maxThreadsPerMultiProcessor;
        return d;
    }

    void print() const {
        printf("Device %d: %s\n", device, prop.name);
        printf("  compute capability            : %d.%d\n", prop.major, prop.minor);
        printf("  SMs                           : %d\n", prop.multiProcessorCount);
        printf("  peak bandwidth                : %.1f GB/s\n", peak_bandwidth_gbps);
        printf("  max resident threads          : %d\n", max_resident_threads);
        printf("  regs/thread @ full occupancy  : %d\n",
               regs_per_thread_at_full_occupancy);
        printf("  shared memory / SM            : %zu KB\n",
               prop.sharedMemPerMultiprocessor / 1024);
        printf("\n");
    }
};

// ---------------------------------------------------------------- timing

// Correct kernel timing: warm up, then time `iters` back-to-back launches with
// CUDA events and divide. Pass any callable that performs one launch.
//
//   float ms = time_kernel([&]{ my_kernel<<<g,b>>>(args); });
//
template <typename Launch>
float time_kernel(Launch&& launch, int iters = 100, int warmup = 10) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

// ---------------------------------------------------------------- reporting

// A benchmark row. Set bytes and/or flops to whatever the kernel *must* move or
// compute (the theoretical minimum, not what a bad implementation does) and this
// prints the numbers that matter, including % of the memory roof.
struct Bench {
    const DeviceInfo* dev  = nullptr;
    double            bytes = 0;   // bytes that MUST move between DRAM and chip
    double            flops = 0;   // useful FLOPs
    double            baseline_ms = 0;  // set to compare speedups; 0 to skip

    void header() const {
        printf("%-34s %10s %10s %10s %8s %8s\n", "kernel", "ms", "GB/s",
               "GFLOP/s", "%peakBW", "vs base");
        printf("%-34s %10s %10s %10s %8s %8s\n", "------", "--", "----",
               "-------", "-------", "-------");
    }

    void row(const char* name, float ms) const {
        double sec    = ms / 1e3;
        double gbps   = bytes ? bytes / sec / 1e9 : 0.0;
        double gflops = flops ? flops / sec / 1e9 : 0.0;
        double pct    = (dev && bytes) ? 100.0 * gbps / dev->peak_bandwidth_gbps : 0.0;

        printf("%-34s %10.4f ", name, ms);
        if (bytes) printf("%10.1f ", gbps); else printf("%10s ", "-");
        if (flops) printf("%10.1f ", gflops); else printf("%10s ", "-");
        if (pct)   printf("%7.1f%% ", pct);  else printf("%8s ", "-");
        if (baseline_ms > 0) printf("%7.2fx", baseline_ms / ms);
        printf("\n");
    }
};

// ---------------------------------------------------------------- verification

// Relative-error comparison. Never trust a GPU kernel you have not checked
// against a host reference -- races and missing barriers raise no error.
inline bool verify(const float* got, const float* want, size_t n,
                   float rtol = 1e-4f, const char* label = "result",
                   int max_report = 5) {
    int    bad       = 0;
    double worst     = 0.0;
    size_t worst_idx = 0;

    // A PURELY relative test is meaningless wherever the reference is ~0. In a
    // dot product over random signs (any GEMM in this book) a handful of the
    // outputs cancel down to ~1e-5, and there a single-ulp difference between
    // host FP32 and device FP32-with-FMA reads as a 100% "error" -- a spurious
    // FAILED on a kernel that is correct. So the denominator gets a floor tied
    // to the scale of the whole result, which is what BLAS test suites do.
    double scale = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double m = std::fabs((double)want[i]);
        if (m > scale) scale = m;
    }
    const double floor_denom = (scale * 1e-3 > 1e-6) ? scale * 1e-3 : 1e-6;

    for (size_t i = 0; i < n; ++i) {
        double denom = std::fabs((double)want[i]);
        double rel   = std::fabs((double)got[i] - (double)want[i]) /
                     (denom > floor_denom ? denom : floor_denom);
        if (rel > worst) { worst = rel; worst_idx = i; }
        if (rel > rtol) {
            if (bad < max_report)
                fprintf(stderr, "  [%s] mismatch at %zu: got %.7g want %.7g (rel %.2e)\n",
                        label, i, got[i], want[i], rel);
            ++bad;
        }
    }
    if (bad) {
        fprintf(stderr, "  [%s] FAILED: %d/%zu elements exceed rtol=%.1e "
                        "(worst %.2e at %zu)\n",
                label, bad, n, rtol, worst, worst_idx);
        return false;
    }
    printf("  [%s] OK (max rel err %.2e)\n", label, worst);
    return true;
}

// ---------------------------------------------------------------- host helpers

// Deterministic pseudo-random fill. Fixed seed so runs are comparable; no
// <random> so the header stays cheap to include.
inline void fill_random(std::vector<float>& v, unsigned seed = 1234) {
    unsigned s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;                 // LCG
        x = (float)((s >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;  // [-1, 1)
    }
}

// RAII device buffer, so the examples don't leak or need cudaFree everywhere.
template <typename T>
struct DeviceBuffer {
    T*     ptr = nullptr;
    size_t n   = 0;

    explicit DeviceBuffer(size_t count) : n(count) {
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    DeviceBuffer(const std::vector<T>& host) : n(host.size()) {
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
        upload(host);
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void upload(const std::vector<T>& host) {
        CUDA_CHECK(cudaMemcpy(ptr, host.data(), n * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
    void download(std::vector<T>& host) const {
        host.resize(n);
        CUDA_CHECK(cudaMemcpy(host.data(), ptr, n * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }
    void zero() { CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T))); }

    operator T*() { return ptr; }
    operator const T*() const { return ptr; }
};
