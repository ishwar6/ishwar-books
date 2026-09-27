// mem_latency.c -- how long does one load take when the CPU cannot overlap it?
//
// Pointer chasing: next = buf[next] over a random cyclic permutation. Each load's
// address depends on the previous load's value, so neither out-of-order
// execution nor the prefetchers can run ahead. Time per step = load-to-use
// latency at that working-set size: L1, L2, then DRAM (plus TLB misses at the
// largest sizes, which is part of the real cost of a random DRAM access).
//
//   cc -O2 mem_latency.c -o mem_latency
#include "bench.h"

int main(void) {
    const size_t kb_sizes[] = {16, 64, 1024, 8192, 65536, 262144, 1048576};  // 16 KB .. 1 GB
    const size_t max_n = kb_sizes[6] * 1024 / sizeof(size_t);
    size_t* buf = (size_t*)malloc(max_n * sizeof(size_t));
    size_t* perm = (size_t*)malloc(max_n * sizeof(size_t));
    if (!buf || !perm) return 1;
    uint32_t seed = 7u;

    printf("%12s %12s\n", "working set", "ns/load");
    for (int k = 0; k < 7; ++k) {
        size_t n = kb_sizes[k] * 1024 / sizeof(size_t);
        // Random cyclic permutation (Sattolo's algorithm): one cycle through all n slots.
        for (size_t i = 0; i < n; ++i) perm[i] = i;
        for (size_t i = n - 1; i > 0; --i) {
            size_t j = ((size_t)xorshift32(&seed) * 65536u + xorshift32(&seed)) % i;
            size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        }
        for (size_t i = 0; i < n; ++i) buf[perm[i]] = perm[(i + 1) % n];

        // Verify it is one cycle of length n (otherwise we would measure a small loop).
        size_t p = 0, len = 0;
        do { p = buf[p]; ++len; } while (p != 0 && len <= n);
        if (len != n) { printf("permutation is not a single cycle!\n"); return 1; }

        const size_t steps = 20000000;
        double best = 1e30;
        for (int r = 0; r < 3; ++r) {
            p = 0;
            double t0 = now_sec();
            for (size_t s = 0; s < steps; ++s) p = buf[p];
            double dt = (now_sec() - t0) / (double)steps;
            g_sink += (double)p;
            if (dt < best) best = dt;
        }
        printf("%9zu KB %12.2f\n", kb_sizes[k], best * 1e9);
    }
    printf("(multiply ns by your clock in GHz -- peak_flops estimates it -- for cycles)\n");
    free(buf);
    free(perm);
    return 0;
}
