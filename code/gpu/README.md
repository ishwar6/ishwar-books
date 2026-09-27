# GPU Programming (code)

Code for Part I of the book at https://ishwarj.com/books/gpu-programming

| Path | What it is |
|---|---|
| `ch01/` | Chapter 1 CPU programs (SIMD, threads, peak FLOP/s, memory latency), plus Go, Java and Rust comparisons. Runs on any laptop. |
| `ch05/device_query.cu` | Chapter 5: print the properties of the GPU you are on |
| `common/cuda_utils.cuh` | error-checking and timing helpers shared by every CUDA program |
| `Makefile` | builds every `.cu` file; `ARCH=sm_75` targets a Colab T4 |

```bash
# Chapter 1, on a laptop
cd ch01 && make && ./bin/peak_flops

# CUDA chapters, e.g. on Google Colab
make ARCH=sm_75 && ./bin/device_query
```
