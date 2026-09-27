// device_query.cu -- print everything about the GPU you are about to optimize for.
//
// Run this FIRST, on every new machine, and write down the output. Every
// "% of peak" claim in this book is relative to these numbers.
//
//   nvcc -O3 -arch=sm_75 -I ../common device_query.cu -o device_query
//   ./device_query

#include "cuda_utils.cuh"

// Cores per SM is not exposed by the runtime API -- it depends on the
// architecture, so it has to be a table. Source: CUDA Programming Guide,
// "Arithmetic Instructions" throughput table.
static int fp32_cores_per_sm(int major, int minor) {
    switch (major) {
        case 7:  return 64;                        // Volta 7.0, Turing 7.5
        case 8:  return (minor == 0) ? 64 : 128;   // A100 64; GA10x/Ada 128
        case 9:  return 128;                       // Hopper
        case 10: return 128;                       // Blackwell datacenter
        case 11: return 128;                       // Jetson Thor
        case 12: return 128;                       // Blackwell consumer
        default: return -1;                        // unknown; skip the estimate
    }
}

static const char* arch_name(int major, int minor) {
    if (major == 7 && minor == 0) return "Volta";
    if (major == 7) return "Turing";
    if (major == 8 && minor == 0) return "Ampere (GA100)";
    if (major == 8 && minor == 9) return "Ada Lovelace";
    if (major == 8) return "Ampere (GA10x)";
    if (major == 9) return "Hopper";
    if (major == 10) return "Blackwell (datacenter)";
    if (major == 11) return "Blackwell (Jetson Thor)";
    if (major == 12) return "Blackwell (consumer)";
    return "unknown";
}

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        printf("No CUDA devices found.\n");
        return 1;
    }
    printf("Found %d CUDA device(s).\n", count);

    int rt = 0, drv = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    printf("CUDA runtime %d.%d, driver %d.%d\n\n", rt / 1000, (rt % 1000) / 10,
           drv / 1000, (drv % 1000) / 10);

    for (int dev = 0; dev < count; ++dev) {
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

        printf("================================================================\n");
        printf("Device %d: %s  (%s)\n", dev, p.name, arch_name(p.major, p.minor));
        printf("================================================================\n");

        // CUDA 13 removed p.clockRate / p.memoryClockRate / (and we avoid
        // p.memoryBusWidth for symmetry). Query attributes instead.
        int sm_clock_khz  = dev_attr(cudaDevAttrClockRate, dev);
        int mem_clock_khz = dev_attr(cudaDevAttrMemoryClockRate, dev);
        int mem_bus_bits  = dev_attr(cudaDevAttrGlobalMemoryBusWidth, dev);

        printf("\n-- identity ------------------------------------------------\n");
        printf("  compute capability            : %d.%d  (compile with -arch=sm_%d%d)\n",
               p.major, p.minor, p.major, p.minor);
        printf("  SM clock rate                 : %.0f MHz\n", sm_clock_khz / 1e3);

        printf("\n-- parallelism ---------------------------------------------\n");
        printf("  SMs                           : %d\n", p.multiProcessorCount);
        printf("  warp size                     : %d\n", p.warpSize);
        printf("  max threads / block           : %d\n", p.maxThreadsPerBlock);
        printf("  max threads / SM              : %d\n", p.maxThreadsPerMultiProcessor);
        printf("  max warps / SM                : %d\n",
               p.maxThreadsPerMultiProcessor / p.warpSize);
        printf("  max block dim                 : (%d, %d, %d)\n",
               p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        printf("  max grid dim                  : (%d, %d, %d)\n",
               p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);

        printf("\n-- on-chip memory ------------------------------------------\n");
        printf("  registers / SM                : %d\n", p.regsPerMultiprocessor);
        printf("  registers / block             : %d\n", p.regsPerBlock);
        printf("  shared memory / SM            : %zu B (%zu KB)\n",
               p.sharedMemPerMultiprocessor, p.sharedMemPerMultiprocessor / 1024);
        printf("  shared memory / block         : %zu B (%zu KB)  [default]\n",
               p.sharedMemPerBlock, p.sharedMemPerBlock / 1024);
        printf("  shared memory / block, opt-in : %zu B (%zu KB)  "
               "[via cudaFuncAttributeMaxDynamicSharedMemorySize]\n",
               p.sharedMemPerBlockOptin, p.sharedMemPerBlockOptin / 1024);
        printf("  L2 cache                      : %d B (%.1f MB)\n", p.l2CacheSize,
               p.l2CacheSize / 1048576.0);
        printf("  constant memory               : %zu B\n", p.totalConstMem);

        printf("\n-- off-chip memory -----------------------------------------\n");
        printf("  global memory                 : %.0f MB\n",
               p.totalGlobalMem / 1048576.0);
        printf("  memory clock                  : %.0f MHz\n", mem_clock_khz / 1e3);
        printf("  memory bus width              : %d bits\n", mem_bus_bits);
        printf("  ECC enabled                   : %s\n", p.ECCEnabled ? "yes" : "no");

        printf("\n-- capabilities --------------------------------------------\n");
        printf("  unified addressing            : %s\n", p.unifiedAddressing ? "yes" : "no");
        printf("  managed memory                : %s\n", p.managedMemory ? "yes" : "no");
        printf("  concurrent kernels            : %s\n", p.concurrentKernels ? "yes" : "no");
        printf("  async engines (copy)          : %d\n", p.asyncEngineCount);
        printf("  cooperative launch            : %s\n", p.cooperativeLaunch ? "yes" : "no");
#if CUDART_VERSION >= 12000
        printf("  thread block clusters         : %s\n", p.clusterLaunch ? "yes" : "no");
#endif

        // ---- the derived numbers you actually reason with -------------------
        double bw_gbps = 2.0 * mem_clock_khz * 1e3 * (mem_bus_bits / 8.0) / 1e9;
        int    cores   = fp32_cores_per_sm(p.major, p.minor);

        printf("\n-- DERIVED (the numbers that matter) -----------------------\n");
        printf("  peak bandwidth                : %.1f GB/s\n", bw_gbps);
        printf("      = 2 (DDR) x %.0f MHz x %d bits / 8\n", mem_clock_khz / 1e3,
               mem_bus_bits);
        printf("  max resident threads          : %d\n",
               p.multiProcessorCount * p.maxThreadsPerMultiProcessor);
        printf("  max resident warps            : %d\n",
               p.multiProcessorCount * p.maxThreadsPerMultiProcessor / p.warpSize);
        printf("  registers/thread @ full occ.  : %d   <-- your register budget\n",
               p.regsPerMultiprocessor / p.maxThreadsPerMultiProcessor);
        printf("  shared mem/thread @ full occ. : %.1f B\n",
               (double)p.sharedMemPerMultiprocessor / p.maxThreadsPerMultiProcessor);
        printf("  L2 bytes / resident thread    : %.1f B   <-- why caches don't save you\n",
               (double)p.l2CacheSize /
                   (p.multiProcessorCount * p.maxThreadsPerMultiProcessor));

        if (cores > 0) {
            // 2 FLOPs per FMA, at the boost clock. Theoretical FP32 CUDA-core peak;
            // tensor-core peak is much higher and is not derivable from cudaDeviceProp.
            double tflops =
                2.0 * cores * p.multiProcessorCount * sm_clock_khz * 1e3 / 1e12;
            printf("  FP32 cores / SM               : %d\n", cores);
            printf("  peak FP32 (CUDA cores)        : %.2f TFLOP/s\n", tflops);
            printf("  MACHINE BALANCE               : %.1f FLOP/byte\n",
                   tflops * 1e12 / (bw_gbps * 1e9));
            printf("      ^ arithmetic intensity above this = compute bound,\n");
            printf("        below it = memory bound. (FP32 CUDA cores only;\n");
            printf("        the tensor-core roof is 10-100x higher.)\n");
        }

        // ---- occupancy table -----------------------------------------------
        printf("\n-- occupancy vs block size (threads only, ignoring regs/smem) --\n");
        printf("  %8s %8s %10s %12s\n", "block", "blocks/SM", "warps/SM", "occupancy");
        for (int bs : {32, 64, 128, 256, 512, 1024}) {
            if (bs > p.maxThreadsPerBlock) continue;
            int by_threads = p.maxThreadsPerMultiProcessor / bs;
            int blocks     = by_threads;  // real limit also involves regs/smem: Ch. 11
            int warps      = blocks * bs / p.warpSize;
            printf("  %8d %8d %10d %11.0f%%\n", bs, blocks, warps,
                   100.0 * warps * p.warpSize / p.maxThreadsPerMultiProcessor);
        }
        printf("\n  (This is the CEILING. Registers and shared memory can only\n");
        printf("   lower it -- see Chapter 11 for the full calculation.)\n\n");
    }
    return 0;
}
