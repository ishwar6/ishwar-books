# Makefile for all book code.
#
#   make                    # build everything for the GPU in this machine
#   make ARCH=sm_75         # build for a specific arch (Colab T4 = sm_75)
#   make ch08               # build only chapter 8's programs
#   make run-ch08           # build and run them in order
#   make DEBUG=1 ch09       # -G -O0, for cuda-gdb
#   make sanitize-ch09      # run chapter 9 under compute-sanitizer racecheck
#   make ptx-ch16           # dump PTX
#   make sass-ch16          # dump SASS  (what actually runs)
#   make clean
#
# On Colab:
#   !cd code && make ARCH=sm_75 && ./bin/device_query

# ---------------------------------------------------------------- config

# sm_75 = Turing/T4 (Colab free tier). Override on the command line.
# "native" asks nvcc to detect the local GPU (CUDA 11.8+).
ARCH ?= native

NVCC   ?= nvcc
STD    ?= c++17
BIN    := bin

# -lineinfo is always on: it costs nothing and without it the profiler cannot
# attribute anything to a source line, which makes Ch. 18 impossible.
NVCCFLAGS := -std=$(STD) -arch=$(ARCH) -lineinfo -I common -Xptxas -v

ifdef DEBUG
  # -G disables device optimization. NEVER benchmark a -G build.
  #
  # ⚠️ -G also inflates register usage, so kernels launched with 1024 threads
  # per block (ch09's 32x32 blocks, ch12's BLOCK=1024 variants) can fail at
  # launch with cudaErrorLaunchOutOfResources: 1024 threads x >64 registers
  # exceeds the 64K register file on a T4. If that happens, debug with a smaller
  # block size rather than fighting it -- or drop -G and use
  # compute-sanitizer, which needs no rebuild.
  NVCCFLAGS += -G -O0 -DDEBUG
else
  NVCCFLAGS += -O3 -DNDEBUG
endif

ifdef FASTMATH
  # Opt-in only: real accuracy loss. See Ch. 5.
  NVCCFLAGS += -use_fast_math
endif

# ---------------------------------------------------------------- targets

CHAPTERS := ch05 ch06 ch07 ch08 ch09 ch10 ch11 ch12 ch13 ch14 ch15 ch16 ch17

# Every .cu file is a standalone program named after itself.
SOURCES := $(wildcard ch*/*.cu)
TARGETS := $(patsubst %.cu,$(BIN)/%,$(notdir $(SOURCES)))

.PHONY: all clean list $(CHAPTERS) $(addprefix run-,$(CHAPTERS))

all: $(TARGETS)

# Pattern rule per chapter directory. vpath lets us keep flat binary names.
vpath %.cu $(CHAPTERS)

# Per-target link libraries. Most programs need none; the ones that compare
# against a library set this. GNU make target-specific variables work with the
# pattern rule below.
NVCC_LIBS =
$(BIN)/sgemm:    NVCC_LIBS = -lcublas
$(BIN)/wmma:     NVCC_LIBS = -lcublas
$(BIN)/tensor:   NVCC_LIBS = -lcublas

$(BIN)/%: %.cu common/cuda_utils.cuh | $(BIN)
	@echo "=== $< -> $@"
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(NVCC_LIBS)

$(BIN):
	@mkdir -p $(BIN)

# ---- per-chapter convenience targets --------------------------------------
# make ch08  builds just chapter 8
define CHAPTER_RULE
$(1): $$(patsubst %.cu,$$(BIN)/%,$$(notdir $$(wildcard $(1)/*.cu)))
run-$(1): $(1)
	@for p in $$(patsubst %.cu,$$(BIN)/%,$$(notdir $$(sort $$(wildcard $(1)/*.cu)))); do \
	    echo ""; echo "########## $$$$p ##########"; ./$$$$p || exit 1; \
	done
sanitize-$(1): $(1)
	@for p in $$(patsubst %.cu,$$(BIN)/%,$$(notdir $$(sort $$(wildcard $(1)/*.cu)))); do \
	    echo ""; echo "########## racecheck $$$$p ##########"; \
	    compute-sanitizer --tool racecheck ./$$$$p || exit 1; \
	done
endef
$(foreach c,$(CHAPTERS),$(eval $(call CHAPTER_RULE,$(c))))

# ---- inspection ------------------------------------------------------------
# make ptx-vector_add / make sass-sgemm   (by program name, no extension)
ptx-%: | $(BIN)
	$(NVCC) $(NVCCFLAGS) -ptx $(firstword $(wildcard ch*/$*.cu)) -o $(BIN)/$*.ptx
	@echo "--> $(BIN)/$*.ptx"

sass-%: $(BIN)/%
	@cuobjdump -sass $(BIN)/$* | sed -n '1,200p'

res-%: $(BIN)/%
	@cuobjdump -res-usage $(BIN)/$*

list:
	@echo "ARCH=$(ARCH)"
	@echo "programs:"; for t in $(sort $(TARGETS)); do echo "  $$t"; done

clean:
	rm -rf $(BIN)
