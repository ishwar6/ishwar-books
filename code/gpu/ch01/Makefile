# Chapter 1 CPU programs. Not part of the top-level CUDA Makefile.
#
#   make                    # Mac (Apple clang): simd_dot peak_flops threads_sum ctx_switch mem_latency
#   make CC=gcc all omp     # Linux gcc (e.g. docker gcc:15): adds the OpenMP build
#   make remarks            # print the vectorizer's reasons (clang or gcc)
#   make hwy HWY=/path/to/highway   # Highway example, needs a checkout of google/highway
#
CC     ?= cc
CXX    ?= c++
CFLAGS ?= -O3 -std=c11 -Wall -Wextra
BIN    := bin

# Does $(CC) identify as clang? Vectorizer flag names differ between compilers.
IS_CLANG := $(shell $(CC) --version 2>/dev/null | grep -qi clang && echo 1)
ifeq ($(IS_CLANG),1)
  NOVEC   := -fno-vectorize -fno-slp-vectorize
  REMARKS := -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
else
  NOVEC   := -fno-tree-vectorize
  REMARKS := -fopt-info-vec-all
endif

all: $(BIN)/simd_dot $(BIN)/peak_flops $(BIN)/threads_sum $(BIN)/ctx_switch $(BIN)/mem_latency

$(BIN):
	mkdir -p $(BIN)

# Scalar reference kernels: vectorization OFF so "scalar" really means scalar.
$(BIN)/dot_scalar.o: dot_scalar.c | $(BIN)
	$(CC) $(CFLAGS) $(NOVEC) -c $< -o $@

$(BIN)/simd_dot: simd_dot.c $(BIN)/dot_scalar.o bench.h | $(BIN)
	$(CC) $(CFLAGS) -fopenmp-simd simd_dot.c $(BIN)/dot_scalar.o -lm -o $@

$(BIN)/peak_flops: peak_flops.c bench.h | $(BIN)
	$(CC) $(CFLAGS) peak_flops.c -lpthread -o $@

$(BIN)/threads_sum: threads_sum.c bench.h | $(BIN)
	$(CC) $(CFLAGS) -fopenmp-simd threads_sum.c -lpthread -lm -o $@

$(BIN)/ctx_switch: ctx_switch.c bench.h | $(BIN)
	$(CC) $(CFLAGS) ctx_switch.c -lpthread -o $@

$(BIN)/mem_latency: mem_latency.c bench.h | $(BIN)
	$(CC) $(CFLAGS) mem_latency.c -o $@

# Full OpenMP (needs libgomp/libomp: fine on Linux gcc, NOT on stock Apple clang).
$(BIN)/threads_sum_omp: threads_sum.c bench.h | $(BIN)
	$(CC) $(CFLAGS) -fopenmp threads_sum.c -lpthread -lm -o $@
omp: $(BIN)/threads_sum_omp

remarks:
	$(CC) -O3 -std=c11 $(REMARKS) -c vec_remarks.c -o /dev/null

hwy: $(BIN)/hwy_dot
$(BIN)/hwy_dot: hwy_dot.cc | $(BIN)
	@test -n "$(HWY)" || (echo "set HWY=/path/to/highway checkout"; exit 1)
	$(CXX) -O3 -std=c++17 -I. -I$(HWY) hwy_dot.cc $(HWY)/hwy/targets.cc $(HWY)/hwy/abort.cc \
	  $(HWY)/hwy/per_target.cc $(HWY)/hwy/print.cc -o $@

clean:
	rm -rf $(BIN)

.PHONY: all omp remarks hwy clean
