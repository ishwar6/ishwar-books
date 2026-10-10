---
title: "Beyond One GPU: Tensor, Pipeline and Expert Parallelism, and Split Prefill and Decode"
description: "Why big models need many GPUs, what the wires between them cost, and how tensor, pipeline and expert parallelism and disaggregated serving split the work: a real two-process tensor-parallel run, measured MoE routing, and worked arithmetic."
date: 2026-10-11
tags: [inference, distributed, parallelism, llm]
series: "LLM Inference from the Ground Up"
series_part: 7
motif: grid
accent: "#ff8a65"
---

Llama 3.1 70B stores its weights in 141 GB. A widely used data-centre GPU, the NVIDIA H100, has 80 GB of memory. The model does not fit, and nothing clever inside one GPU changes that.

So we split the model across several GPUs. The moment we do, a new cost appears that the first six parts of this series could ignore: **the GPUs have to talk to each other**, many times for every single token. How often they talk, how much they send, and how fast the wire between them is, decides almost everything about how a model should be spread out.

This part builds the whole picture from the ground up:

1. **Why one GPU is not enough**: the memory arithmetic for an 8B, a 70B, a 405B and a 671B model, and the speed limit on decode.
2. **The wires**: how fast GPUs can talk inside one server and between servers, and a simple model of what a message costs.
3. **Tensor parallelism**: split every layer across GPUs. We run a real model this way, in two separate processes that only share numbers through a sum, and check it gives the same answer.
4. **Collectives**: all-reduce, all-gather and all-to-all, the few ways a group of GPUs exchange data, with the cost formula worked out.
5. **Pipeline parallelism**: give each GPU a slice of the layers instead.
6. **Expert parallelism** for mixture-of-experts models, with **real routing decisions** recorded from a 64-expert model.
7. **Disaggregated prefill and decode**: run the two phases from [Part 1](llm-inference-1-prefill-and-decode.md) on different GPUs, and move the KV cache between them.
8. **The real systems** that do this today, and the flags you would set.
9. **How to choose** a layout, with worked examples.

> [!NOTE] Three kinds of numbers in this article
> Every number carries its source.
> **Measured:** run on the laptop used throughout this series (Apple M5 Pro, PyTorch). That includes a real tensor-parallel run of Qwen2.5-0.5B across separate processes on the CPU, timings of matrix shards on the Apple GPU (`mps`), and the expert choices of the real OLMoE-1B-7B model on real text. There is no NVIDIA GPU here, so **no multi-GPU speed was measured**. The laptop was also shared with other heavy jobs while it ran, so its timings are noisy; they are used for shapes and ratios, never as H100 numbers.
> **Arithmetic and simulation:** model sizes, bytes and times computed from published shapes and NVIDIA's spec-sheet peaks, plus small simulators in plain Python. Real systems are slower than peak; where an input is an assumption (like the fixed cost of one message), it is named and varied.
> **Reported:** numbers from papers (Megatron-LM, GPipe, the Llama 3 report, DeepSeek-V3, DistServe, Splitwise, Mooncake) and from official documentation, each with its hardware.
> Engine flags were checked in the vLLM and SGLang documentation on 11 October 2026 (vLLM `187a0eb`, SGLang `f9cee8d`).

This part builds on [Part 6](llm-inference-6-serving-in-production.md), which covered serving on one GPU per copy of the model: latency targets, goodput, queueing, capacity planning and routing between copies. Here we look inside a copy that is too big for one GPU.

## 1. Why one GPU is not enough

### Memory: weights plus the KV cache

A GPU must hold two big things while it serves a model:

- the **weights**, read on every forward pass ([Part 1](llm-inference-1-prefill-and-decode.md));
- the **KV cache**, the saved keys and values of every token of every conversation in flight ([Part 2](llm-inference-2-kv-cache.md)).

> [!DEFINITION] HBM (GPU memory)
> High Bandwidth Memory: the stacked memory chips packaged right next to a data-centre GPU. An H100 SXM has 80 GB of it, read at up to 3.35 TB/s. Everything the GPU computes on must come from here.

The weight size is the number of parameters times the bytes per parameter. The KV cache size is the formula from Part 2:

$$
M = N\,b_w \;+\; \underbrace{2\,L\,H_{\mathrm{KV}}\,d_h\,b_{kv}}_{\text{bytes per token}}\;\times\;S\times C
$$

where:

- $$N$$ is the number of parameters and $$b_w$$ the bytes per parameter (2 for BF16, 1 for FP8);
- $$L$$ is the number of layers, $$H_{\mathrm{KV}}$$ the number of key-value heads and $$d_h$$ the size of each head;
- the 2 counts keys and values, and $$b_{kv}$$ is the bytes per stored number (2 for BF16);
- $$S$$ is the number of conversations being served at once and $$C$$ the tokens in each one.

**Worked example: Llama 3.1 70B.** The Llama 3 report gives 80 layers, 8 KV heads and a model width of 8,192 (64 heads of 128). Counting every matrix gives $$N = 70.6$$ billion, so the weights take $$70.6\times10^9\times 2 = 141$$ GB. One token of KV cache is $$2\times80\times8\times128\times2 = 327{,}680$$ bytes, 320 KiB. Serve 32 conversations of 8,192 tokens and the cache is $$320\ \text{KiB}\times32\times8192 = 80$$ GiB. Together that is 227 GB. With about 90% of each 80 GB GPU usable (the rest goes to the CUDA runtime, activations and fragmentation), that needs **four** H100s.

The same arithmetic for the other models:

{{FIG:fit|Weights (BF16, or FP8 for DeepSeek-V3) plus the KV cache for 32 conversations of 8,192 tokens each. Arithmetic from the published model shapes; 90% of each 80 GB GPU counted as usable.}}

[![Terminal output of memory_math.py: Llama 3.1 8B 8.0B parameters, 16 GB weights, 128 KiB per token, 32 GiB of KV for 32 x 8K, 50 GB total, 1 H100; Llama 3.1 70B 70.6B, 141 GB, 320 KiB, 80 GiB, 227 GB, 4 H100s; Llama 3.1 405B 405.9B, 812 GB, 504 KiB, 126 GiB, 947 GB, 14 H100s; DeepSeek-V3 671B, 671 GB in FP8, 68.6 KiB, 17 GiB, 689 GB, 10 H100s. Decode floor table: 8B 4.8 ms on 1 GPU down to 0.3 ms on 16; 70B 21.1 ms on 2 GPUs, 10.5 on 4, 5.3 on 8; 405B 15.1 ms on 16; DeepSeek-V3 active weights 11.0 ms](/img/multigpu/memory_math-run.png)](/img/multigpu/memory_math-run.png)

Three things stand out.

**The 8B model fits on one GPU with room to spare.** For it, more GPUs mean more copies of the model, not a split model. That is the world of Part 6.

**The 405B model does not even fit on one 8-GPU server.** In BF16 its weights alone are 812 GB, more than the 640 GB of eight H100s. Meta hit exactly this wall:

> [!PAPER] Llama Team, The Llama 3 Herd of Models · Section 6.1 · page 51
> [![Llama 3 report: When using a BF16 number representation for the model parameters, Llama 3 405B does not fit in the GPU memory of a single machine with 8 Nvidia H100 GPUs. To address this issue, we parallelize model inference using BF16 precision across 16 GPUs on two machines. Within each machine, the high NVLink bandwidth](/img/multigpu/paper-llama3-pp1.png)](/img/multigpu/paper-llama3-pp1.png)
>
> **Context:** the start of the report's inference section, about serving the 405B model.
>
> **What it says:** 405B in BF16 does not fit in one 8-GPU H100 machine, so Meta spread it over 16 GPUs in two machines. We will come back to *how* they split it in Section 5.
>
> **Why it matters:** our arithmetic (812 GB of weights against 640 GB) predicts exactly this, before a single GPU is bought.
>
> [Read Section 6.1](https://arxiv.org/pdf/2407.21783#page=51)

**DeepSeek-V3 is bigger but its cache is smaller.** It has 671 billion parameters, stored in FP8 (one byte each), so 671 GB. But it uses multi-head latent attention (MLA, mentioned in [Part 2](llm-inference-2-kv-cache.md)): each layer caches one compressed vector of 512 numbers plus a 64-number position key, shared by all heads. That is $$61\times(512+64)\times2 = 70{,}272$$ bytes per token, **68.6 KiB, against 504 KiB for Llama 3.1 405B**. Its problem is the weights, not the cache.

> [!DEFINITION] Mixture of experts (MoE)
> A model whose feed-forward layers are split into many smaller "experts". A small router picks a few experts for each token, so each token uses only part of the model. DeepSeek-V3 has 256 routed experts per layer and picks 8, plus one shared expert: 671 billion parameters in total, but only 37 billion are used for any one token.

### Speed: decode is limited by reading the weights

Even when a model fits, there is a second reason to spread it: speed. [Part 1](llm-inference-1-prefill-and-decode.md) showed that a decode step for a small batch is **memory-bound**: the GPU spends its time reading every weight once, not multiplying. That gives a hard floor on the time per token:

$$
t_{\text{step}} \;\ge\; \frac{N\,b_w}{p\,B_{\text{HBM}}}
$$

where:

- $$N\,b_w$$ is the size of the weights in bytes;
- $$p$$ is the number of GPUs the weights are split over, each reading only its own share;
- $$B_{\text{HBM}}$$ is one GPU's memory bandwidth (3.35 TB/s for an H100).

**Worked example.** Llama 3.1 70B on 2 H100s: $$141\ \text{GB} / (2\times3.35\ \text{TB/s}) = 21.1$$ ms, at best 47 tokens per second for one user. On 8 H100s each GPU reads only 17.6 GB, so the floor drops to **5.3 ms**. Spreading the weights does not just make the model fit; it multiplies the memory bandwidth working on each token.

{{FIG:floor|The lowest possible time per decode step at batch 1, if the weights are split perfectly and communication were free. Arithmetic: BF16 weights, 3.35 TB/s per H100.}}

That last condition, "if communication were free", is the whole subject of this article. It is not free.

## 2. The wires between GPUs

### Inside a server and between servers

A typical AI server (NVIDIA calls it a node) has eight GPUs. They are connected to each other in two very different ways, depending on whether the other GPU is in the same box.

> [!DEFINITION] NVLink and NVSwitch
> NVLink is NVIDIA's direct GPU-to-GPU connection. On H100 it carries up to 900 GB/s per GPU, counting both directions. NVSwitch is a switch chip inside the server that connects all eight GPUs' NVLinks, so any GPU can talk to any other at full speed.

> [!DEFINITION] PCIe
> The general-purpose connection between a GPU and the rest of the computer (the CPU, network cards, disks). PCIe Gen5 with 16 lanes carries 128 GB/s, both directions together. GPUs without NVLink must talk to each other over PCIe.

> [!DEFINITION] InfiniBand and RDMA
> InfiniBand is a fast network for connecting servers in a cluster. A 400 Gb/s card (NVIDIA ConnectX-7) moves 50 GB/s each way. RDMA (remote direct memory access) lets one machine write into another's memory without either CPU copying the data; with GPUDirect RDMA the network card reads and writes GPU memory directly. Some clusters use RDMA over fast Ethernet instead (RoCE).

{{FIG:node|One 8-GPU server, as in NVIDIA's DGX H100: NVLink through NVSwitch inside, one 400 Gb/s network card per GPU to the rest of the cluster.}}

Put the numbers side by side:

{{FIG:links|Peak bandwidth of each link, from NVIDIA's H100 and ConnectX-7 pages. Note the log scale: each step down is several times slower.}}

| Link | Peak bandwidth | Compared with NVLink |
|---|---|---|
| HBM, inside one H100 | 3,350 GB/s | (the GPU's own memory) |
| NVLink 4, GPU to GPU in a server | 900 GB/s both ways (450 each way) | 1x |
| PCIe Gen5 x16 | 128 GB/s both ways (64 each way) | 7x slower |
| InfiniBand NDR, one 400 Gb/s card | 50 GB/s each way | 9x slower |
| Ethernet, 100 Gb/s | 12.5 GB/s each way | 36x slower |

These are NVIDIA's published peaks (the [H100 page](https://www.nvidia.com/en-us/data-center/h100/) lists "NVIDIA NVLink: 900GB/s" and "PCIe Gen5: 128GB/s"; the [ConnectX-7 page](https://www.nvidia.com/en-us/networking/infiniband-adapters/) lists 400 Gb/s). The newer Blackwell generation doubles NVLink to 1,800 GB/s per GPU and can join 72 GPUs into one NVLink domain (the "NVL72" racks), which moves the line between "inside" and "outside" further out. The shape of the problem stays the same.

Real clusters see less than peak. DeepSeek reports what their H800 cluster (a version of the H100 with reduced NVLink bandwidth, made for export to China) delivers in practice:

> [!PAPER] DeepSeek-AI, DeepSeek-V3 Technical Report · Section 3.2.2 · page 13
> [![DeepSeek-V3 report: in our cluster, cross-node GPUs are fully interconnected with IB, and intra-node communications are handled via NVLink. NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s). To effectively leverage the different bandwidths of IB and NVLink, we limit each token to be dispatched to at most 4 nodes, thereby reducing IB traffic.](/img/multigpu/paper-dsv3-links.png)](/img/multigpu/paper-dsv3-links.png)
>
> **Context:** how DeepSeek built the communication kernels for their mixture-of-experts model.
>
> **What it says:** in their cluster, NVLink gives about 160 GB/s and InfiniBand 50 GB/s, so they design the model's routing to send each token to at most 4 servers.
>
> **Why it matters:** the network shaped the model itself. When communication is the bottleneck, people change the algorithm to fit the wires, not the other way round.
>
> [Read Section 3.2.2](https://arxiv.org/pdf/2412.19437#page=13)

### What one message costs: latency plus size over bandwidth

Every message between GPUs costs a fixed amount of time before any data moves (setting up, synchronising, crossing switches), plus time proportional to its size. This is called the **alpha-beta model**:

$$
T(n) = \alpha + \frac{n}{\beta}
$$

where:

- $$n$$ is the message size in bytes;
- $$\alpha$$ (alpha) is the fixed cost per message, in seconds;
- $$\beta$$ (beta) is the bandwidth in bytes per second.

**Worked example.** Send 16 KiB (16,384 bytes) over NVLink at 450 GB/s each way. The size term is $$16{,}384 / (450\times10^9) = 36$$ ns. If the fixed cost is a few microseconds (an assumed figure; NVIDIA does not publish one), it is **about 100 times bigger** than the data term. Small messages are all alpha; large messages are all beta.

NVIDIA does not publish an alpha for NVLink or InfiniBand, and I cannot measure one without the hardware. So I measured the *shape* on what I have: separate processes on this laptop's CPU, exchanging data with `torch.distributed` (the same library NVIDIA GPUs use, but with its CPU backend, called gloo).

{{FIG:alpha_beta|Measured on this laptop: the time of one all-reduce (defined in Section 4) between 2 or 4 CPU processes, against message size. The absolute numbers describe this laptop's shared-memory path while it was busy with other jobs, not any GPU link.}}

[![Terminal output of tp_demo.py part C: gloo all_reduce timings for 2 and 4 processes from 4 bytes to 64 MiB, with the fitted alpha and beta](/img/multigpu/tp_demo_c-run.png)](/img/multigpu/tp_demo_c-run.png)

The curve is flat for small messages and then rises in step with size: exactly $$\alpha + n/\beta$$. On this laptop the flat part is about 0.25 ms with 2 processes and 1.3 ms with 4: more participants means more steps and more waiting for the slowest one, a first hint of Section 4. (These alphas are hundreds of times larger than a GPU's, because they are CPU processes on a busy machine; only the shape carries over.) The flat part is why the next sections keep asking two separate questions about every design: **how many messages** per token, and **how many bytes** per message.

## 3. Tensor parallelism: split every layer

### The idea

> [!DEFINITION] Tensor parallelism (TP)
> Every layer's weight matrices are cut into pieces, and each GPU holds one piece of **every** layer. All GPUs work on the same tokens at the same time, each computing part of each layer, and they combine their partial results after each block. "TP=8" means the layers are split eight ways.

Tensor parallelism attacks both problems from Section 1 at once: each GPU stores $$1/p$$ of the weights, and each GPU reads only $$1/p$$ of them per step. The price is communication inside every layer. The trick, from the Megatron-LM paper (Shoeybi et al., 2019), is to cut the matrices so that the communication happens as rarely as possible.

### Splitting the MLP: columns, then rows

A transformer's MLP block is two matrix multiplications with a nonlinearity between them:

$$
Y = \mathrm{GeLU}(XA), \qquad Z = YB
$$

where:

- $$X$$ is the input, one row per token, $$d$$ columns wide (the model width);
- $$A$$ is a $$d\times f$$ weight matrix that widens each token to $$f$$ numbers (the FFN width);
- GeLU is the nonlinear function applied to every number (Llama uses a close cousin, SwiGLU);
- $$B$$ is an $$f\times d$$ matrix that narrows each token back to $$d$$ numbers.

There are two ways to cut $$A$$ in half. Megatron's authors explain why only one of them works well:

> [!PAPER] Shoeybi et al., Megatron-LM · Section 3 · page 4
> [![Megatron-LM: We start by detailing the MLP block. The first part of the block is a GEMM followed by a GeLU nonlinearity, Y = GeLU(XA). One option is to split A along its rows and X along its columns; since GeLU is a nonlinear function, GeLU(X1A1+X2A2) is not GeLU(X1A1)+GeLU(X2A2) and this approach will require a synchronization point before the GeLU function. Another option is to split A along its columns, which allows the GeLU nonlinearity to be independently applied to the output of each partitioned GEMM. Hence, we partition the first GEMM in this column parallel fashion and split the second GEMM along its rows so it takes the output of the GeLU layer directly without requiring any communication. The output of the second GEMM is then reduced across the GPUs. This requires only a single all-reduce operation in the forward pass and a single all-reduce in the backward pass.](/img/multigpu/paper-megatron-mlp.png)](/img/multigpu/paper-megatron-mlp.png)
>
> **Context:** Section 3 of the paper, which introduced the tensor-parallel layout every inference engine uses today.
>
> **What it says:** cut $$A$$ by **columns**, so each GPU computes complete GeLU outputs for its own share of the $$f$$ hidden units; cut $$B$$ by **rows**, so each GPU can multiply its own hidden units straight away. Only at the very end are the partial results added together, once.
>
> **Why it matters:** inference only runs the forward pass, so the MLP costs **exactly one all-reduce** per layer, no matter how many GPUs share it.
>
> [Read Section 3](https://arxiv.org/pdf/1909.08053#page=4)

Written out for two GPUs, with $$A = [A_1, A_2]$$ split by columns and $$B = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}$$ split by rows:

$$
Z = \mathrm{GeLU}(X[A_1, A_2])\begin{bmatrix} B_1 \\ B_2 \end{bmatrix} = \underbrace{\mathrm{GeLU}(XA_1)\,B_1}_{\text{GPU 0}} + \underbrace{\mathrm{GeLU}(XA_2)\,B_2}_{\text{GPU 1}}
$$

The GeLU is applied to each column of $$XA$$ separately, so it does not care how the columns are grouped. The only cross-GPU step is the final **+**.

{{FIG:tp_mlp|The Megatron MLP split on two GPUs. Each GPU computes a complete partial output with no communication; one all-reduce adds them.}}

### Check it: the sum of the parts equals the whole

This is easy to verify. The first part of `tp_demo.py` builds an MLP with Qwen2.5-0.5B's sizes ($$d = 896$$, $$f = 4{,}864$$), splits it two, four and eight ways, and compares the sum of the pieces with the unsplit result:

```python
full = F.gelu(X @ A) @ B
for p in [2, 4, 8]:
    A_parts = A.chunk(p, dim=1)              # column split: each GPU gets ffn/p output columns
    B_parts = B.chunk(p, dim=0)              # row split: each GPU gets the matching ffn/p rows
    partial = [F.gelu(X @ Ai) @ Bi for Ai, Bi in zip(A_parts, B_parts)]   # no communication needed here
    Y = sum(partial)                         # the all-reduce: add the p partial outputs
    res[p] = float((Y - full).abs().max())
```

`chunk(p, dim=1)` cuts $$A$$ into $$p$$ blocks of columns; `chunk(p, dim=0)` cuts $$B$$ into the matching blocks of rows. Each list entry in `partial` is what one GPU would compute on its own. `sum(partial)` is the all-reduce. Then the wrong way, splitting $$A$$ by rows and applying GeLU before adding:

```python
Xs, As = X.chunk(2, dim=1), A.chunk(2, dim=0)
wrong = sum(F.gelu(Xi @ Ai) for Xi, Ai in zip(Xs, As)) @ B
```

```text
PART A: split matrix multiplies (float64, CPU)
p=2: each part A_i (896, 2432), B_i (2432, 896); max |sum of parts - unsplit| = 1.1e-14
p=4: each part A_i (896, 1216), B_i (1216, 896); max |sum of parts - unsplit| = 1.3e-14
p=8: each part A_i (896, 608), B_i (608, 896); max |sum of parts - unsplit| = 1.2e-14
row-split first matrix, GeLU applied before adding: max error 0.909 (output values are about 0.533 on average) -> needs a sync before GeLU
```

The column-then-row split matches the unsplit result to $$10^{-14}$$, which is just the rounding of 64-bit floats. The row split is wrong by more than the size of the answer itself, because $$\mathrm{GeLU}(a+b) \ne \mathrm{GeLU}(a)+\mathrm{GeLU}(b)$$. To use it you would have to add the halves *before* the GeLU: a second all-reduce per layer.

### Splitting attention: whole heads per GPU

Attention splits even more naturally. Its heads are already independent: head 3 never looks at head 5's numbers until the output projection mixes them. So each GPU takes a whole group of heads, with their query, key and value weights (a column split), and the matching rows of the output projection $$W_O$$ (a row split). Again one all-reduce adds the partial outputs.

{{FIG:tp_attn|Attention split by heads. Each GPU computes its own heads, keeps only its own key-value heads in its KV cache, and produces a partial output for the all-reduce.}}

There is a bonus: **each GPU only caches the keys and values of its own heads**. The KV cache is split $$p$$ ways along with the weights, so the room for conversations grows with every GPU you add.

There is also a limit. With grouped-query attention (Part 2), Llama 3.1 70B has 64 query heads but only **8 KV heads**. At TP=8 each GPU gets exactly one KV head. At TP=16 two GPUs would need the same KV head, so it must be copied, and the cache stops shrinking. That is one reason tensor parallelism usually stops at the 8 GPUs of one server.

The paper's own picture shows both blocks. The boxes marked $$f$$ and $$g$$ are where communication happens; in the forward pass $$f$$ does nothing and $$g$$ is the all-reduce:

> [!PAPER] Shoeybi et al., Megatron-LM · Figure 3 · page 4
> [![Megatron-LM Figure 3: Blocks of Transformer with Model Parallelism. (a) MLP: X goes through f, is multiplied by A1 and A2 on two GPUs, GeLU, then by B1 and B2, then g combines them before dropout. (b) Self-attention: queries, keys and values are split by heads into two groups, each with softmax and dropout, then multiplied by B1 and B2 and combined by g. f and g are conjugate: f is an identity operator in the forward pass and all reduce in the backward pass while g is an all reduce in the forward pass and identity in the backward pass.](/img/multigpu/paper-megatron-fig3.png)](/img/multigpu/paper-megatron-fig3.png)
>
> **Context:** Figure 3, the paper's diagram of a tensor-parallel transformer layer on two GPUs.
>
> **What it says:** in the forward pass, each block starts with no communication ($$f$$ is the identity) and ends with one all-reduce ($$g$$).
>
> **Why it matters:** a whole transformer layer costs **two all-reduces** in inference: one after attention and one after the MLP.
>
> [Read Section 3](https://arxiv.org/pdf/1909.08053#page=4)

{{FIG:ar_layer|Two all-reduces per layer, in every layer, for every forward pass. The GPUs cannot start the next block until the sum is complete.}}

### A real tensor-parallel run, in two processes

Splitting one MLP is a toy. To be sure the whole recipe works, `tp_demo.py` runs **all 24 layers of Qwen2.5-0.5B** with tensor parallelism across two separate operating-system processes. Each process loads the model, keeps only its own half of every layer, and the two talk only through `torch.distributed.all_reduce`, the same call vLLM and SGLang make on NVIDIA GPUs (here with the CPU backend, gloo, since there is no NVIDIA GPU). Qwen2.5-0.5B has 14 query heads and 2 KV heads, so with TP=2 each process gets 7 query heads and 1 KV head.

Each process slices its share of the weights once, at load time:

```python
qs = slice(rank * self.hq * hd, (rank + 1) * self.hq * hd)     # this rank's query heads
ks = slice(rank * self.hkv * hd, (rank + 1) * self.hkv * hd)   # this rank's KV heads
fs = slice(rank * ffn, (rank + 1) * ffn)                       # this rank's MLP hidden units
self.layers.append(dict(
    wq=g('self_attn.q_proj.weight')[qs], bq=g('self_attn.q_proj.bias')[qs],      # column split
    wk=g('self_attn.k_proj.weight')[ks], bk=g('self_attn.k_proj.bias')[ks],
    wv=g('self_attn.v_proj.weight')[ks], bv=g('self_attn.v_proj.bias')[ks],
    wo=g('self_attn.o_proj.weight')[:, qs],                                        # row split
    wg=g('mlp.gate_proj.weight')[fs], wu=g('mlp.up_proj.weight')[fs],              # column split
    wd=g('mlp.down_proj.weight')[:, fs]))                                          # row split
```

PyTorch stores a linear layer's weight as (output, input), so a **column** split of the maths is a slice of the weight's **first** axis (`[qs]`), and a **row** split is a slice of its second axis (`[:, qs]`). The gate and up projections of Qwen's SwiGLU MLP are both column-split the same way, so their elementwise product stays local.

Then each layer of the forward pass is ordinary code, with exactly two collective calls:

```python
a = F.scaled_dot_product_attention(q, K, V, is_causal=ids.shape[0] > 1)   # my 7 heads only
a = a.transpose(0, 1).reshape(ids.shape[0], -1)
h = h + self.all_reduce(a @ L['wo'].T)       # all-reduce 1 of 2 in this layer
x = rms(h, L['ln2'], c.rms_norm_eps)
m = F.silu(x @ L['wg'].T) * (x @ L['wu'].T)  # my half of the MLP
h = h + self.all_reduce(m @ L['wd'].T)       # all-reduce 2 of 2 in this layer
```

`self.all_reduce` wraps `dist.all_reduce`, which replaces each process's tensor with the sum over both processes, and counts the calls and bytes. Everything outside the two calls (embeddings, normalisation, the residual additions, the output head) is simply done on both processes, which is how Megatron handles them too (it can also split the vocabulary, which this demo skips).

The script compares the last-token logits with the unsplit Hugging Face model and greedily generates 24 tokens both ways:

[![Terminal output of tp_demo.py part B: TP=1 holds 100% of the layer weights and 912 KiB of KV cache, max logit difference 4.9e-05 against Hugging Face, same 24 greedy tokens; TP=2 rank 0 holds 50% of the layer weights and 456 KiB of KV cache, max logit difference 5.0e-05, same 24 tokens; 48 all-reduce calls per token carrying 168 KiB, and the generated continuation](/img/multigpu/tp_demo_b-run.png)](/img/multigpu/tp_demo_b-run.png)

The split model gives the **same 24 tokens** as the original, and the logits agree to about $$5\times10^{-5}$$, the same small difference that the unsplit re-implementation has (float32 sums in a different order). Each process holds exactly half of the layer weights and half of the KV cache. And every decode token costs **48 all-reduces**: 24 layers times two.

The timings in that run are not a speed-up: both "GPUs" are processes on one CPU, sharing the same cores and memory, so splitting gives each process half the work but no extra hardware to do it with. This demo checks correctness and counts messages; the next pages estimate speed.

### How many bytes per token?

Each all-reduce carries one activation vector per token: $$d$$ numbers. Per decode step:

$$
\text{calls} = 2L, \qquad \text{bytes per call} = b\times d\times s
$$

where:

- $$L$$ is the number of layers (two all-reduces each);
- $$b$$ is the number of tokens in the step (the batch size during decode);
- $$d$$ is the model width and $$s$$ the bytes per number (2 for BF16).

**Worked example, Qwen2.5-0.5B in the demo:** $$2\times24 = 48$$ calls, each $$1\times896\times4 = 3{,}584$$ bytes in float32, so $$48\times3{,}584 = 172{,}032$$ bytes = 168 KiB per token, exactly the 168.0 KiB the script counted.

**Worked example, Llama 3.1 70B in BF16:** $$2\times80 = 160$$ all-reduces per step. At batch 1 each carries $$8{,}192\times2 = 16$$ KiB; at batch 64, 1 MiB. Compare that with what each GPU reads from its own memory at TP=8: 17.6 GB of weights. The bytes sent are tiny next to the bytes read. But there are **160 separate messages**, and each must finish before the layer can go on. From Section 2: when messages are small, it is their count, through $$\alpha$$, that costs time.

### What each GPU's share costs: a measurement

To see how the work per GPU shrinks, `measure_mps.py` times one real Llama 3.1 8B MLP matrix (14,336 by 4,096, BF16, 117 MB) on this laptop's Apple GPU, whole and cut to the half and quarter that TP=2 and TP=4 would give each GPU:

{{FIG:mps_shard|Measured on the Apple GPU (mps): one 8B MLP matrix, whole and as TP=2 and TP=4 shards, for 1 to 4,096 tokens. Fastest of 40 runs.}}

[![Terminal output of measure_mps.py: GPU copy bandwidth, then for 1 to 4,096 tokens the time, TFLOP/s, GB/s and speedup of the whole matrix and of TP=2 and TP=4 shards, then the all-reduce bytes each shard would send](/img/multigpu/measure_mps-run.png)](/img/multigpu/measure_mps-run.png)

Two regimes show up. With **4,096 tokens** the work is arithmetic, and the shards behave as hoped: the half takes half the time (2.03x faster) and the quarter close to a quarter (3.63x). With **one token**, the whole matrix is a 117 MB read at 134 GB/s, 0.88 ms. But the half and quarter shards take 0.69 and 0.73 ms, barely faster. Below a certain size, every GPU operation has a fixed cost of its own (launching the kernel and getting it going), and splitting the work cannot go under that floor. It is the same shape as the alpha of Section 2, inside one GPU. Engines fight it by capturing a whole decode step as one CUDA graph, so that hundreds of small launches cost about one; the principle stays: cutting small work very finely gives diminishing returns, even before any communication.

The last block of the output adds the communication those shards would need: the partial outputs are only 8 KiB per token, which takes nanoseconds on any link. The bytes are not the problem; the number of separate exchanges is. (The copy bandwidth here, 153 GB/s, is lower than the 262 GB/s measured in Part 1 because this run shared the laptop with other jobs; the fastest of 40 runs is reported for every number.)

### Putting it together: TP on 70B

Now add the communication back. `layout_model.py` estimates one Llama 3.1 70B decode step on H100s as the time to read the weights and the KV cache (or to do the arithmetic, whichever is longer) plus 160 all-reduces:

$$
t_{\text{step}} = \max\!\left(\frac{W + K}{p\,B_{\text{HBM}}},\ \frac{2Nb}{p\,F}\right) + 2L\left(2\alpha + \frac{2(p-1)}{p}\cdot\frac{b\,d\,s}{\beta}\right)
$$

where:

- $$W$$ is the weight bytes and $$K$$ the KV-cache bytes read in the step ($$b$$ sequences of 4,096 tokens each);
- $$F$$ is one GPU's peak BF16 rate (989 TFLOP/s for an H100) and $$2Nb$$ the arithmetic for $$b$$ tokens;
- the last term is $$2L$$ all-reduces, each costing two message delays ($$2\alpha$$) plus its bytes, using the two-step all-reduce explained in Section 4, with $$\beta$$ = 450 GB/s;
- $$\alpha = 5\ \mu$$s is an **assumption** (NVIDIA publishes none); it is varied below.

{{FIG:tp_scaling|Modelled Llama 3.1 70B decode step on H100s, split into reading memory and all-reduces. Peak numbers throughout, so real systems are slower.}}

[![Terminal output of layout_model.py: tensor parallel scaling for Llama 3.1 70B at batch 1 and 64, the alpha sensitivity, the one-node comparison of TP=8, TP=4 x 2 and TP=2 x 4, the 405B two-node comparison and the pipeline schedule](/img/multigpu/layout_model-run.png)](/img/multigpu/layout_model-run.png)

**Worked example at batch 1.** TP=2: reading 141 GB at $$2\times3.35$$ TB/s takes 21.3 ms (with the cache); 160 all-reduces at $$2\times5\ \mu\text{s}$$ each add 1.6 ms; total **22.9 ms**. TP=8: memory 5.3 ms, all-reduces still **1.6 ms**, total **6.9 ms**. Going from 2 to 8 GPUs cut the memory time by 4x but left the communication exactly where it was, so the step is only 3.3x faster, and communication grew from 7% to 23% of it. With $$\alpha = 2\ \mu$$s the TP=8 step is 6.0 ms; with $$\alpha = 10\ \mu$$s it is 8.5 ms.

This is a general law, and the Google team that scaled PaLM inference put it plainly:

> [!PAPER] Pope et al., Efficiently Scaling Transformer Inference · Section 3.2.1 · page 4
> [![Pope et al.: As we parallelize the computation across more chips, the memory latency and compute latency does decrease, often near-linearly. However, the communication latency remains roughly constant independent of the number of chips used, since the entire activation matrix is aggregated across chips for every pair of matrix multiplications. As the number of chips grows larger, communication becomes a bottleneck.](/img/multigpu/paper-pope-constant.png)](/img/multigpu/paper-pope-constant.png)
>
> **Context:** the paper's analysis of the Megatron-style "1D weight-stationary" layout for feed-forward layers, on TPU v4 chips.
>
> **What it says:** splitting wider shrinks memory and compute time almost in proportion, but the communication stays about the same size, because every chip must see the whole activation after every pair of matrix multiplications.
>
> **Why it matters:** tensor parallelism has diminishing returns built in. Past a point, more GPUs per copy of the model buy almost nothing, and the GPUs are better used as more copies.
>
> [Read Section 3.2.1](https://arxiv.org/pdf/2211.05102#page=4)

Two practical consequences follow. **Tensor parallelism belongs where $$\alpha$$ is small and links are fast**: inside one NVLink server. And because it needs that, inference engines put real effort into making all-reduce fast: vLLM and TensorRT-LLM ship their own all-reduce kernels for small messages on NVLink, rather than relying only on the general NCCL library. Section 4 shows why that helps.

## 4. Collectives: how a group of GPUs exchanges data

### The four you need

When a group of GPUs exchange data in a fixed pattern, the operation is called a **collective**. NVIDIA's NCCL library (pronounced "nickel") implements them on GPUs; `torch.distributed` calls it. Four collectives cover almost everything in this article.

> [!DEFINITION] All-reduce
> Every GPU starts with a vector of the same length. Every GPU ends with the element-by-element sum of all of them. Tensor parallelism uses it after each attention and MLP block.

> [!DEFINITION] All-gather
> Every GPU starts with one piece. Every GPU ends with all the pieces, in order.

> [!DEFINITION] Reduce-scatter
> Every GPU starts with a full vector. GPU $$r$$ ends with the sum of piece $$r$$ only. An all-reduce is exactly a reduce-scatter followed by an all-gather.

> [!DEFINITION] All-to-all
> Every GPU starts with one piece addressed to each GPU. Piece $$r$$ of every GPU goes to GPU $$r$$: like transposing a table. Expert parallelism uses it to send tokens to their experts.

`collectives.py` runs all four on four simulated GPUs (list entries in NumPy), where rank $$r$$ starts with $$[10r, 10r+1, 10r+2, 10r+3]$$:

{{FIG:collectives|The four collectives on four ranks, with the numbers from collectives.py. All-to-all keeps piece r on rank r (highlighted) and swaps the rest.}}

### Ring all-reduce, step by step

How do $$p$$ GPUs compute a sum without one GPU becoming a bottleneck? The classic answer is the **ring**. Arrange the GPUs in a circle and cut each vector into $$p$$ chunks.

1. **Reduce-scatter, $$p-1$$ steps.** At each step every GPU sends one chunk to its right-hand neighbour, which adds it to its own copy of that chunk. After $$p-1$$ steps, each GPU holds one chunk that contains the complete sum.
2. **All-gather, $$p-1$$ more steps.** Each GPU passes its finished chunk to the right, and the finished chunks travel round the ring until everyone has all of them.

Here is the trace for four ranks holding $$[1,2,3,4]$$ times 1, 2, 3 and 4:

{{FIG:ring|Ring all-reduce traced by collectives.py. Highlighted cells already hold their final sum. Every rank ends with [10, 20, 30, 40].}}

[![Terminal output of collectives.py: the ring all-reduce trace with each rank's four numbers after every reduce-scatter and all-gather step, the check that every rank ends with the plain sum, the bytes each rank sends for p = 2, 4 and 8 against the formula 2(p-1)/p, and the other three collectives](/img/multigpu/collectives-run.png)](/img/multigpu/collectives-run.png)

### What the ring costs

Each step moves one chunk, $$n/p$$ bytes, from every GPU at the same time. There are $$2(p-1)$$ steps. So, with the alpha-beta model from Section 2:

$$
T_{\text{ring}} = 2(p-1)\left(\alpha + \frac{n}{p\,\beta}\right) = \underbrace{2(p-1)\,\alpha}_{\text{grows with } p} + \underbrace{\frac{2(p-1)}{p}\cdot\frac{n}{\beta}}_{\text{almost constant}}
$$

where:

- $$p$$ is the number of GPUs and $$n$$ the size of the vector being summed, in bytes;
- $$\alpha$$ is the fixed cost of one step and $$\beta$$ each GPU's sending bandwidth.

The byte term is the famous property of the ring: each GPU sends $$2(p-1)/p$$ times the vector, which is never more than **2x** the vector however many GPUs there are. The simulation counts the bytes and agrees: 1.000, 1.500 and 1.750 times $$n$$ for 2, 4 and 8 ranks. (NCCL's [performance notes](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) use the same $$2(p-1)/p$$ factor to turn measured times into "bus bandwidth".)

The latency term is the catch. It grows with $$p$$: $$2(p-1) = 14$$ steps on 8 GPUs, and every GPU must wait for its neighbour at every step.

**Worked example: Llama 3.1 70B at batch 1, TP=8, NVLink.** Each all-reduce carries $$n = 16$$ KiB. Bytes: $$\frac{14}{8}\times\frac{16{,}384}{450\times10^9} = 64$$ ns. Latency: $$14\times\alpha$$, so $$28$$ to $$140\ \mu$$s for $$\alpha$$ between 2 and 10 $$\mu$$s. Times 160 all-reduces: **4.5 to 22.4 ms per token**, while the bytes alone would take 0.01 ms. A ring is the wrong shape for tiny messages.

### Fewer steps for small messages

That is why there are other algorithms. With NVSwitch every GPU can reach every other directly, so a GPU can send each peer its chunk in one step (a one-hop reduce-scatter) and then gather the sums in one more step: **two steps, whatever $$p$$ is**. NVIDIA describes this for TensorRT-LLM (their "MultiShot" all-reduce, which uses the switch to multicast):

> "This process is repeated 2N-2 times where N is the number of GPUs working together ... This increases latency, as all GPUs need to stay synchronized at every step of the ring." ([NVIDIA technical blog, *3x Faster AllReduce with NVSwitch and TensorRT-LLM MultiShot*, November 2024](https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/))

The two-step version costs about

$$
T_{\text{two-step}} \approx 2\alpha + \frac{2(p-1)}{p}\cdot\frac{n}{\beta}
$$

with the same symbols as before. Same bytes, but only two delays. For the 70B example: $$160\times2\times\alpha$$ = **0.65 to 3.2 ms per token**, about 7x less than the ring at TP=8. (NCCL itself also switches algorithm and protocol by message size; the point is that small all-reduces are a latency problem, and good implementations attack the step count.)

| Llama 3.1 70B, all 160 all-reduces of one decode step | Ring | Two-step | Bytes only |
|---|---|---|---|
| TP=2, batch 1, NVLink | 0.65 to 3.21 ms | 0.65 to 3.21 ms | 0.006 ms |
| TP=8, batch 1, NVLink | 4.49 to 22.41 ms | 0.65 to 3.21 ms | 0.010 ms |
| TP=8, batch 64, NVLink | 5.13 to 23.05 ms | 1.29 to 3.85 ms | 0.65 ms |
| TP=8, batch 64, PCIe Gen5 | 9.07 to 26.99 ms | 5.23 to 7.79 ms | 4.59 ms |
| TP=8, batch 64, one 400 Gb/s NIC | 10.35 to 28.27 ms | 6.51 to 9.07 ms | 5.87 ms |

Ranges are for $$\alpha$$ = 2 to 10 $$\mu$$s (assumed); bandwidths are per-direction peaks. Read the table by rows:

- At batch 1 the **bytes never matter**; only the number of steps does.
- At batch 64 the bytes start to count, and on **PCIe or the network** they alone cost 4.6 to 5.9 ms per token, as much as reading the weights. That is why the vLLM documentation tells you to avoid tensor parallelism on GPUs without NVLink (Section 8).

## 5. Pipeline parallelism: split the layers instead

### The idea, from training

> [!DEFINITION] Pipeline parallelism (PP)
> The model's layers are divided into consecutive groups called stages, and each GPU (or group of GPUs) holds one stage. A token's activations flow from stage 1 to stage 2 and onwards, like an assembly line. "PP=2" means two stages.

Pipeline parallelism communicates far less than tensor parallelism. Between two stages it sends the activations once ($$b\times d\times s$$ bytes, point to point), instead of two all-reduces in every layer. The cost is idle time. GPipe (Huang et al., 2019) introduced the standard picture:

> [!PAPER] Huang et al., GPipe · Figure 2 · page 3
> [![GPipe Figure 2: (a) An example neural network with sequential layers is partitioned across four accelerators. (b) The naive model parallelism strategy leads to severe under-utilization due to the sequential dependency of the network: only one device works at a time. (c) Pipeline parallelism divides the input mini-batch into smaller micro-batches, enabling different accelerators to work on different micro-batches simultaneously; the idle region is labelled Bubble.](/img/multigpu/paper-gpipe-fig2.png)](/img/multigpu/paper-gpipe-fig2.png)
>
> **Context:** GPipe was a training system; panel (b) shows forward passes F and backward passes B on four devices.
>
> **What it says:** with one batch, only one device works at a time. Cutting the batch into micro-batches lets the devices overlap, leaving a smaller idle "bubble".
>
> **Why it matters:** the same two pictures apply to inference, minus the backward pass.
>
> [Read Section 2](https://arxiv.org/pdf/1811.06965#page=3)

> [!DEFINITION] Micro-batch and bubble
> A micro-batch is a slice of the batch that moves through the pipeline on its own, so that different stages can work on different slices at the same time. The bubble is the time a stage sits idle because no micro-batch has reached it yet, or because the next one is not ready.

GPipe gives the size of the bubble:

> [!PAPER] Huang et al., GPipe · Section 2.3 · page 4
> [![GPipe: As illustrated in Figure 2c, partitioning introduces some idle time per accelerator, which we refer to as the bubble overhead. This bubble time is O((K-1)/(M+K-1)) amortized over the number of micro-steps M. In our experiments, we found the bubble overhead to be negligible when M is at least 4 x K.](/img/multigpu/paper-gpipe-bubble.png)](/img/multigpu/paper-gpipe-bubble.png)
>
> **Context:** GPipe's performance analysis.
>
> **What it says:** with $$K$$ stages and $$M$$ micro-batches, the idle fraction is about $$\frac{K-1}{M+K-1}$$, small once $$M \ge 4K$$.
>
> [Read Section 2.3](https://arxiv.org/pdf/1811.06965#page=4)

**Worked example.** $$K = 4$$ stages. With $$M = 1$$: $$\frac{3}{4} = 75\%$$ idle. With $$M = 16$$: $$\frac{3}{19} = 16\%$$.

### Pipelines during decode

Decode adds a twist: the next token of a sequence cannot start until its previous token has left the **last** stage. One sequence alone keeps only one stage busy at a time. Several independent groups of sequences (micro-batches) fill the gaps. `layout_model.py` lays out the schedule for 4 stages, each micro-batch generating 3 tokens:

{{FIG:pp_schedule|Pipeline schedule from layout_model.py: 4 GPUs, each stage taking one time slot. One sequence keeps each GPU busy a quarter of the time; four micro-batches keep them 80% busy.}}

The figure shows the two faces of pipelining:

- **Throughput improves** with micro-batches: 12 tokens in 15 slots instead of 3 tokens in 12.
- **Latency does not.** Each token still passes through all 4 stages one after another, each stage running only its quarter of the layers. A pipeline never makes one sequence's token faster than reading all its layers on one GPU would, and it adds a network hop between stages.

Meta's choice for Llama 3 405B, from the same section as the box in Section 1, follows from this:

> [!PAPER] Llama Team, The Llama 3 Herd of Models · Section 6.1 · page 52
> [![Llama 3 report: Within each machine the high NVLink bandwidth enables the use of tensor parallelism. Across nodes, however, connectivity has lower bandwidth and higher latency, so we use pipeline parallelism instead. During training with pipeline parallelism, bubbles are a major efficiency concern. However, they are not an issue during inference, since inference does not involve a backward pass that requires a pipeline flush. Therefore, we use micro-batching to improve inference throughput with pipeline parallelism. We evaluate the effect of using two micro-batches in inference workloads of 4,096 input tokens and 256 output tokens. The additional synchronization points due to micro-batching also increase latency but, overall, micro-batching still leads to a better throughput-latency trade-off.](/img/multigpu/paper-llama3-pp2.png)](/img/multigpu/paper-llama3-pp2.png)
>
> **Context:** how Meta served Llama 3 405B in BF16 on two 8-GPU machines.
>
> **What it says:** tensor parallelism inside each machine, pipeline parallelism between the two machines, with two micro-batches. Micro-batching raised throughput and also raised latency, and was still worth it.
>
> **Why it matters:** this is the standard recipe for a model too big for one node: TP where the wires are fast, PP where they are slow.
>
> [Read Section 6.1](https://arxiv.org/pdf/2407.21783#page=51)

### Worked example: 405B on two nodes

`layout_model.py` compares two ways to use 16 H100s in two nodes for Llama 3.1 405B:

- **TP=16**: every layer split 16 ways, so 252 all-reduces per step, all crossing InfiniBand ($$\alpha$$ assumed 10 $$\mu$$s, $$\beta$$ = 50 GB/s per GPU).
- **TP=8 x PP=2**: each node holds 63 layers, split 8 ways over NVLink; one hop over InfiniBand per step.

{{FIG:pp_vs_tp|Modelled Llama 3.1 405B decode on two 8-GPU nodes. Pipelining does not lower latency, but with two micro-batches in flight it gives the highest throughput.}}

| Batch 32 | ms per token | Tokens per second |
|---|---|---|
| TP=16 across both nodes | 31.4 (15.0 of it all-reduce over InfiniBand) | 1,020 |
| TP=8 x PP=2, one micro-batch | 36.4 | 879 |
| TP=8 x PP=2, two micro-batches | 36.4 | 1,760 |

Per token, the pipeline is *slower*: each token reads 406 GB on 8 GPUs (15.1 ms) in the first node, then again in the second, plus the hop. TP=16 reads everything in parallel on 16 GPUs but spends almost half its time in all-reduces over the network. Fill the pipeline with two micro-batches and it delivers **1.7x the throughput** of TP=16 while sending a few kilobytes per token over InfiniBand instead of hundreds of messages. If latency matters most and the network is excellent, TP across nodes can win; for throughput per GPU, the pipeline does. Both conclusions depend on the assumed $$\alpha$$, which is why the code keeps it as a named input.

## 6. Expert parallelism for mixture-of-experts models

### Why MoE models need their own kind of parallelism

A mixture-of-experts layer replaces one big MLP with many small ones. DeepSeek-V3 has, in each of its 58 MoE layers, 256 routed experts (each an MLP with a hidden width of 2,048) plus one shared expert that every token uses. A small router scores the experts for each token and sends the token to its top 8.

You could split every expert with tensor parallelism. But each expert is small, and an all-reduce per expert per layer would be pure overhead. The natural move is the opposite: **keep each expert whole and give different experts to different GPUs.**

> [!DEFINITION] Expert parallelism (EP)
> The experts of each MoE layer are distributed across GPUs, each GPU holding a few whole experts. Tokens are sent to the GPUs that hold their chosen experts and the results are sent back. "EP=32" means the experts are spread over 32 GPUs.

> [!DEFINITION] Data-parallel attention
> The attention part of the model is copied onto every GPU (or every small TP group), and each copy handles different sequences. It is often combined with EP: attention is cheap to copy, especially with MLA's small KV cache, while the experts are too large to copy. SGLang calls this "DP attention"; DeepSeek's report writes it as DP8 or DP80.

Each MoE layer then runs in four stages:

{{FIG:moe_dispatch|Expert parallelism on four GPUs. Dispatch sends each token's activations to the GPUs holding its chosen experts; combine sends the expert outputs back to be added up.}}

1. Each GPU runs attention for its own tokens, and the router picks each token's experts.
2. **Dispatch:** an all-to-all sends each token's activations to the GPUs that hold its experts.
3. Each GPU runs its experts on whatever tokens it received.
4. **Combine:** a second all-to-all sends the outputs back, where they are added with the router's weights.

### How many bytes

Each token is sent once per chosen expert, and the answers come back the same way.

$$
\text{bytes per token per MoE layer} = k\,d\,(s_{\text{dispatch}} + s_{\text{combine}})
$$

where:

- $$k$$ is the number of experts chosen per token (8 for DeepSeek-V3);
- $$d$$ is the model width (7,168);
- $$s_{\text{dispatch}}$$ and $$s_{\text{combine}}$$ are the bytes per number on the way out and back: DeepSeek sends 1 byte (FP8) out and 2 bytes (BF16) back.

**Worked example.** $$8\times7{,}168\times(1+2) = 172{,}032$$ bytes per token per layer; over 58 MoE layers, **10.0 MB per token**. A decode batch of 128 tokens per GPU dispatches $$128\times8\times7{,}168 = 7.34$$ MB per layer.

DeepSeek's open-source all-to-all library, [DeepEP](https://github.com/deepseek-ai/DeepEP), published timings for exactly this setting (H800s with 400 Gb/s InfiniBand, 128 tokens per batch, hidden 7,168, top-8, FP8 dispatch, BF16 combine; [README of release v1.2.1](https://github.com/deepseek-ai/DeepEP/tree/v1.2.1)). Our byte count reproduces them: 7.34 MB at their measured 98 GB/s is 75 $$\mu$$s, against the **77 $$\mu$$s** they report for EP8 dispatch; 14.68 MB of combine at 127 GB/s is 116 $$\mu$$s against **114 $$\mu$$s** reported. Across all 58 MoE layers that is **11.1 ms per decode step** at EP8 and 32.1 ms at EP256 (194 and 360 $$\mu$$s per layer), unless it is hidden behind computation. That is why DeepSeek runs two micro-batches and overlaps one's communication with the other's compute.

[![Terminal output of collectives.py section 3 and 4: the all-reduce cost table for Llama 3.1 70B, ring against two-step, and the DeepSeek-V3 all-to-all byte arithmetic: 57,344 B dispatch and 114,688 B combine per token per layer, 10.0 MB per token, 75 and 116 microseconds for a 128-token batch against DeepEP's reported 77 and 114, 11.1 ms per decode step at EP8 and 32.1 ms at EP256](/img/multigpu/collectives_cost-run.png)](/img/multigpu/collectives_cost-run.png)

### The real problem: load imbalance

All-to-all has a second cost that the bytes do not show. A layer is finished only when **the busiest GPU** is finished. If the router sends twice the average number of tokens to the experts on one GPU, that GPU takes twice as long and every other GPU waits.

How uneven is real routing? Rather than guess, `moe_routing.py` records it. It runs the real **OLMoE-1B-7B** model (64 experts per layer, 8 chosen per token, 16 layers, 6.9 billion parameters) on this laptop's GPU over 12,288 tokens of WikiText and 12,288 tokens of Python code, and saves which 8 experts the router chose for every token in every layer. That is about 3 million real routing decisions. The routing is the measurement; nothing is timed.

The recording is a forward hook on each layer's router, which returns the chosen expert ids as its third output:

```python
for i, layer in enumerate(model.model.layers):
    hooks.append(layer.mlp.gate.register_forward_hook(
        lambda m, inp, out, i=i: buf.setdefault(i, []).append(out[2].cpu())))
with torch.no_grad():
    for s in seqs:
        model(torch.tensor(s, device=dev)[None])
```

Then plain Python asks what those choices would do to expert parallelism. Experts $$0$$ to $$63$$ are placed in order, $$64/G$$ per GPU, and for random batches of tokens we compute the load on the busiest GPU divided by the average load:

```python
counts = np.bincount(picks[l, idx].ravel(), minlength=E)   # tokens sent to each expert in this batch
load = np.zeros(n_gpus)
np.add.at(load, expert_to_gpu, counts)                       # tokens each GPU must process
ratios.append(load.max() / load.mean())                      # 1.00 = perfectly even
```

The same calculation on random routing (each token picks 8 experts uniformly) separates bad luck from real preference.

[![Terminal output of moe_routing.py: for WikiText and Python code, how much more than its fair share the hottest expert gets, the busiest-GPU-over-average ratio for EP = 8, 16 and 64 at batches of 64, 256 and 4,096 tokens against random routing, the effect of balanced placement and redundant experts, and how many hot experts the two kinds of text share](/img/multigpu/moe_routing-run.png)](/img/multigpu/moe_routing-run.png)

First, the experts themselves. In a typical layer the busiest expert receives **3.1 times** its fair share of WikiText tokens (2.2x to 4.2x across layers), and on code **6.9 times**; the idlest receive almost nothing.

{{FIG:moe_share|Measured: share of all expert slots each of the 64 experts receives in one layer of OLMoE-1B-7B, sorted. The dashed line is a perfectly fair 1/64.}}

Then what that does to GPUs:

{{FIG:moe_imbalance|Busiest GPU divided by the average GPU, from OLMoE's real routing on WikiText against random routing. Experts placed in order, 64/EP per GPU.}}

| Busiest GPU / average, batches of 4,096 tokens | EP=8 | EP=16 | EP=64 |
|---|---|---|---|
| Random routing (chance only) | 1.02 | 1.04 | 1.10 |
| Real routing, WikiText | 1.32 | 1.57 | 3.18 |
| Real routing, Python code | 1.80 | 2.61 | 6.77 |

Three lessons:

- **Big batches do not save you.** Random imbalance fades as batches grow (1.17 at 64 tokens, 1.02 at 4,096 for EP=8). Real imbalance does not (1.35 and 1.32): it comes from the router's genuine preferences, not from small numbers.
- **Wider EP makes it worse.** At EP=64 each GPU holds one expert, so the busiest GPU is simply the hottest expert: 3.2x the average on WikiText, 6.8x on code. Two thirds or more of the GPU time in the layer is waiting.
- **The hot experts depend on the traffic.** WikiText and code share a median of **1** of their 8 hottest experts per layer. A placement tuned on yesterday's chat traffic can be wrong for today's coding traffic.

(OLMoE was trained with a load-balancing loss, like most MoE models, and DeepSeek-V3 adds its own auxiliary-loss-free balancing. Training reduces the skew, but as these numbers show, it does not remove it at serving time, when the mix of text is whatever users send.)

### The fix: place experts by load, and copy the hot ones

DeepSeek's report describes what they do in production:

> [!PAPER] DeepSeek-AI, DeepSeek-V3 Technical Report · Section 3.4.1 · page 19
> [![DeepSeek-V3 report, 3.4.1 Prefilling: The minimum deployment unit of the prefilling stage consists of 4 nodes with 32 GPUs. The attention part employs 4-way Tensor Parallelism (TP4) with Sequence Parallelism (SP), combined with 8-way Data Parallelism (DP8). Its small TP size of 4 limits the overhead of TP communication. For the MoE part, we use 32-way Expert Parallelism (EP32), which ensures that each expert processes a sufficiently large batch size. To achieve load balancing among different experts, we introduce a deployment strategy of redundant experts, which duplicates high-load experts and deploys them redundantly. The high-load experts are detected based on statistics collected during the online deployment and are adjusted periodically (e.g., every 10 minutes). For the deployment of DeepSeek-V3, we set 32 redundant experts for the prefilling stage. For each GPU, besides the original 8 experts it hosts, it will also host one additional redundant expert.](/img/multigpu/paper-dsv3-prefill.png)](/img/multigpu/paper-dsv3-prefill.png)
>
> **Context:** Section 3.4, "Inference and Deployment", which describes how DeepSeek serves the model.
>
> **What it says:** attention runs with small tensor parallelism (TP4) copied 8 ways (DP8); the experts are spread 32 ways (EP32). The busiest experts are found from live statistics and **copied**, with the set refreshed about every 10 minutes; every GPU holds its 8 experts plus one extra copy.
>
> **Why it matters:** this is the production answer to the table above: measure the load, then copy hot experts so their tokens can be split.
>
> [Read Section 3.4](https://arxiv.org/pdf/2412.19437#page=18)

`moe_routing.py` tries the same two ideas on OLMoE's routing. It learns from the **first half** of the recorded tokens and is tested on the **second half**, just as a server must use past load to plan for future load:

1. **Placement by load:** put the heaviest experts first, each on the least-loaded GPU that still has room.
2. **Redundant copies:** each GPU gets a few spare slots; repeatedly copy the busiest expert on the busiest GPU to the least-loaded GPU with a free slot (keeping a copy only if it does not create a new busiest GPU), and split that expert's tokens evenly between its copies.

{{FIG:moe_redundant|Simulated on measured routing (WikiText, batches of 4,096 tokens): experts in order, placed by past load, and with redundant copies. Learned on the first half of the text, tested on the second.}}

| Busiest / average, tested on unseen tokens | In order | Placed by load | Plus redundant copies |
|---|---|---|---|
| WikiText, EP=8 (8 copies) | 1.34 | 1.28 | 1.29 |
| WikiText, EP=16 (16 copies) | 1.58 | 1.47 | 1.47 |
| WikiText, EP=64 (64 copies) | 3.19 | 3.19 | 2.77 |
| Code, EP=8 (8 copies) | 1.81 | 1.12 | 1.12 |
| Code, EP=16 (16 copies) | 2.62 | 1.74 | 1.37 |
| Code, EP=64 (64 copies) | 6.75 | 6.75 | 1.67 |

When a GPU holds several experts, **placing them by load** does most of the work (code at EP=8: 1.81 to 1.12). When a GPU holds only one expert, placement cannot help at all, and **only copies help** (code at EP=64: 6.75 to 1.67). The leftover imbalance on WikiText comes from the traffic changing between the two halves of the text, which is exactly why DeepSeek refreshes its choice every 10 minutes and is "exploring a dynamic redundancy strategy" (same section) that re-plans for every batch.

### DeepSeek-V3 in production

Putting the pieces together, here is how DeepSeek serves V3, with prefill and decode on **separate** groups of machines (Section 7 explains why):

{{FIG:dsv3|DeepSeek-V3's deployment units as described in its technical report. Prefill uses 32 GPUs per unit; decode uses 320, with one expert per GPU.}}

> [!PAPER] DeepSeek-AI, DeepSeek-V3 Technical Report · Section 3.4.2 · page 19
> [![DeepSeek-V3 report, 3.4.2 Decoding: During decoding, we treat the shared expert as a routed one, so each token will select 9 experts. The minimum deployment unit of the decoding stage consists of 40 nodes with 320 GPUs. The attention part employs TP4 with SP, combined with DP80, while the MoE part uses EP320. For the MoE part, each GPU hosts only one expert, and 64 GPUs are responsible for hosting redundant experts and shared experts. All-to-all communication of the dispatch and combine parts is performed via direct point-to-point transfers over IB to achieve low latency. Additionally, we leverage the IBGDA technology to further minimize latency and enhance communication efficiency.](/img/multigpu/paper-dsv3-decode.png)](/img/multigpu/paper-dsv3-decode.png)
>
> **Context:** the decoding half of Section 3.4.
>
> **What it says:** decode uses a much wider unit than prefill, 320 GPUs, with **one expert per GPU** and 64 GPUs for copies and the shared expert. Tokens move by direct point-to-point InfiniBand writes, issued by the GPU itself (IBGDA), to keep latency low.
>
> **Why it matters:** decode processes few tokens per step, so it needs many GPUs' worth of memory bandwidth and very low-latency all-to-all. One expert per GPU is exactly the case where our simulation says only redundant copies can fix imbalance.
>
> [Read Section 3.4.2](https://arxiv.org/pdf/2412.19437#page=19)

DeepSeek later published what this looks like in daily operation. Their [inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) (February 2025) describes a slightly different production layout than the paper, **EP32 for prefill over 4 nodes and EP144 for decode over 18 nodes**, each with 32 redundant experts, and reports the results over one day:

[![DeepSeek inference system overview: Prefilling Phase [Routed Expert EP32, MLA/Shared Expert DP32]: Each deployment unit spans 4 nodes with 32 redundant routed experts, where each GPU handles 9 routed experts and 1 shared expert. Decoding Phase [Routed Expert EP144, MLA/Shared Expert DP144]: Each deployment unit spans 18 nodes with 32 redundant routed experts, where each GPU manages 2 routed experts and 1 shared expert.](/img/multigpu/deepseek-day6-units.png)](/img/multigpu/deepseek-day6-units.png)

[![DeepSeek inference system overview statistics: total input tokens 608B, of which 342B tokens (56.3%) hit the on-disk KV cache; total output tokens 168B; average output speed 20 to 22 tokens per second; each H800 node delivers an average throughput of about 73.7k tokens/s input including cache hits during prefilling or about 14.8k tokens/s output during decoding.](/img/multigpu/deepseek-day6-stats.png)](/img/multigpu/deepseek-day6-stats.png)

Each H800 node delivered **about 73.7 thousand input tokens per second** in prefill (including cache hits) or **about 14.8 thousand output tokens per second** in decode. Prefill nodes move five times more tokens, which is the compute-bound against memory-bound gap of Part 1 showing up at cluster scale, and one more reason to give the two phases different machines.

## 7. Disaggregated prefill and decode

### Why the two phases get in each other's way

[Part 1](llm-inference-1-prefill-and-decode.md) showed that prefill and decode are different kinds of work. Prefill processes thousands of prompt tokens at once and is limited by arithmetic (compute-bound). Decode produces one token per sequence per step and is limited by reading memory (memory-bound). [Part 3](llm-inference-3-vllm.md) showed what happens when one GPU does both: a long prompt arriving in the middle of other users' answers either **stalls** those answers while it is prefilled, or, with chunked prefill, is cut into slices that **slow every step a little** for longer.

> [!DEFINITION] Disaggregated serving
> Prefill and decode run on different GPUs (often different machines). A prefill instance processes the prompt and produces the first token and the KV cache; the KV cache is then transferred to a decode instance, which generates the rest of the answer. Also called prefill/decode (PD) disaggregation or phase splitting.

The cost model of the simulator below makes the interference concrete. A user is one of 32 sequences being decoded on an H100 running Llama 3.1 8B when another user's 2,000-token prompt arrives:

{{FIG:interference|What one decoding user sees when someone else's long prompt arrives, from the simulator's cost model (roofline with assumed efficiencies, Llama 3.1 8B on an H100).}}

With prefill first, the user's stream freezes for the whole 67.8 ms prefill, six normal steps long. With 512-token chunks, four of the user's steps grow from 11.4 ms to 19.3 ms. On a decode-only GPU nothing happens at all.

### The papers: Splitwise and DistServe

Two papers in 2024 made the case for splitting the phases onto separate GPUs. **Splitwise** (Patel et al., Microsoft and the University of Washington) started from production traces and the observation that decode does not need the newest, most compute-heavy GPUs; it reported clusters with "up to 1.4x higher throughput at 20% lower cost". **DistServe** (Zhong et al., OSDI 2024) framed the goal as **goodput**, defined in [Part 6](llm-inference-6-serving-in-production.md): the highest request rate at which a target share of requests still meets both the time-to-first-token (TTFT) and time-per-output-token (TPOT) objectives. Its Figure 1 is the clearest picture of the problem:

> [!PAPER] Zhong et al., DistServe · Figure 1 · page 1
> [![DistServe Figure 1: Performance when serving an LLM with 13B parameters under a synthetic workload with input length 512 and output length 64 on one NVIDIA 80GB A100. Upper: P90 time-to-first-token for existing systems against a prefill-only system, with the SLA line crossed at about 1.6 and 5.6 requests per second. Lower: P90 time-per-output-token for existing systems against a decode-only system, with the SLA crossed at about 1.6 and 10 requests per second.](/img/multigpu/paper-distserve-fig1.png)](/img/multigpu/paper-distserve-fig1.png)
>
> **Context:** the first figure of the paper, a 13B model on one A100 with 512-token prompts and 64-token answers.
>
> **What it says:** a GPU doing both phases meets the latency objectives up to about 1.6 requests per second. A GPU doing only prefill handles 5.6, and one doing only decode handles 10.
>
> [Read the introduction](https://arxiv.org/pdf/2401.09670#page=1)

> [!PAPER] Zhong et al., DistServe · Section 1 · page 2
> [![DistServe: Under the SLO attainment of 90%, the maximum achievable goodput on a single A100 GPU, constrained by the more stringent one of TTFT and TPOT requirements, is about 1.6 requests per second. The performance contrasts sharply when each phase is served independently on a separate GPU, which achieve per-GPU goodput of 5.6 rps for the prefill phase and 10 rps for decoding. Ideally, by allocating 2 GPUs for prefill and 1 GPU for decoding, we can effectively serve the model with an overall goodput of 10 rps, or equally 3.3 rps per GPU, which is 2.1x higher than existing systems.](/img/multigpu/paper-distserve-goodput.png)](/img/multigpu/paper-distserve-goodput.png)
>
> **What it says:** two prefill GPUs (2 x 5.6 = 11.2 rps of prefill) feeding one decode GPU (10 rps) serve 10 rps on 3 GPUs: 3.3 per GPU, **2.1x** the 1.6 of a GPU doing both.
>
> **Why it matters:** the gain comes from two things together: no interference, and the freedom to choose the **ratio** of prefill to decode GPUs (and, in the full paper, a different parallel layout for each). Across their workloads the authors report up to "7.4x more requests or 12.6x tighter SLO" than the systems they compared with.
>
> [Read Section 1](https://arxiv.org/pdf/2401.09670#page=2)

Notice what the baseline was: "existing systems" in early 2024 meant vLLM without chunked prefill. That matters for our own simulation below.

### The cost: moving the KV cache

Splitting the phases adds one job: the prompt's KV cache, built on the prefill GPU, must be copied to the decode GPU before the second token. Its size is the bytes-per-token of Section 1 times the prompt length, and its transfer time is size over bandwidth:

$$
t_{\text{transfer}} = \frac{2\,L\,H_{\mathrm{KV}}\,d_h\,b_{kv}\times T}{\beta}
$$

where:

- $$T$$ is the number of prompt tokens and the numerator is the KV cache in bytes (Section 1);
- $$\beta$$ is the bandwidth of the path between the two GPUs.

DistServe works one example; our script reproduces it first, to check the formula:

> [!PAPER] Zhong et al., DistServe · Section 3.3 · page 6
> [![DistServe: Communication overhead. Transferring KV caches from prefill to decoding instances incurs notable overheads. For example, the KV cache size of a single 512-token request on OPT-66B is approximately 1.13GB. Assuming an average arrival rate of 10 rps, we need to transfer 11.3GB data per second, or equivalently 90Gbps bandwidth to render the overhead invisible. While many modern GPU clusters for LLMs are equipped with InfiniBand (e.g., 800 Gbps), in cases where cross-node bandwidth is limited, DistServe relies on the commonly available intra-node NVLINK, where the peak bandwidth between A100 GPUs is 600 GB/s, again rendering the transmission overhead negligible.](/img/multigpu/paper-distserve-kv.png)](/img/multigpu/paper-distserve-kv.png)
>
> **What it says:** 1.13 GB per 512-token request for OPT-66B; at 10 requests per second that is about 90 Gbps of sustained traffic, easy for InfiniBand or NVLink, hard for ordinary networks.
>
> [Read Section 3.3](https://arxiv.org/pdf/2401.09670#page=6)

**Worked example (check).** OPT-66B has 64 layers and full multi-head attention with width 9,216, so $$2\times64\times9{,}216\times2 = 2{,}359{,}296$$ bytes per token; times 512 tokens is $$1{,}207{,}959{,}552$$ bytes = **1.125 GiB**, and at 10 per second, 90 Gibit/s. The paper's 1.13GB and 90Gbps are the same numbers counted in powers of two.

**Worked example (today).** Llama 3.1 70B, an 8,192-token prompt: $$320\ \text{KiB}\times8{,}192 = 2.5$$ GiB. Over one 400 Gb/s card (50 GB/s): **54 ms**. Over NVLink in the same server: 6 ms. If the prefill side runs TP=8 and the decode side runs TP=8, each GPU holds one eighth of the cache and can send it over **its own** network card, so the 8 cards together move it in 6.7 ms.

{{FIG:kv_transfer|Moving the KV cache of one 8,192-token prompt, against the time to prefill it. Arithmetic with peak link bandwidths; prefill assumed at 50% of H100 peak.}}

[![Terminal output of kv_transfer.py: the DistServe check (1.125 GiB, 90 Gibit/s), then KV size, prefill time and transfer time over NVLink, eight 400G NICs, one 400G NIC and 100G Ethernet for Llama 3.1 8B, Llama 3.1 70B and DeepSeek-V3 at 1,024, 8,192 and 32,768 tokens, and the extra delay before the second token when sending at the end or layer by layer](/img/multigpu/kv_transfer-run.png)](/img/multigpu/kv_transfer-run.png)

Two things make this cheaper than it looks. First, MLA: DeepSeek-V3's 8,192-token cache is only 0.54 GiB, a fifth of 70B's. Second, the transfer does not have to wait for the prefill to finish. Splitwise sends each layer's KV as soon as that layer is done:

> [!PAPER] Patel et al., Splitwise · Section IV-C · page 7
> [![Splitwise: The time required for the transfer depends on the size of the KV cache and on the bandwidth of the interconnect between the prompt and the token machines. Even when using fast InfiniBand links, the transfer overhead for large prompt sizes could become a significant fraction of the TBT. In Splitwise, we optimize the KV-cache transfer by overlapping it with the computation in the prompt phase. As each layer in the LLM gets calculated in the prompt machine, the KV cache corresponding to that layer is also generated. At the end of each layer, we trigger an asynchronous transfer of the KV-cache for that layer while the prompt computation continues to the next layer.](/img/multigpu/paper-splitwise-layerwise.png)](/img/multigpu/paper-splitwise-layerwise.png)
>
> **Context:** Splitwise's "prompt machine" does prefill and its "token machine" does decode; TBT is time between tokens.
>
> **What it says:** each layer's KV leaves while the next layer is still computing, so most of the transfer hides behind the prefill.
>
> [Read Section IV-C](https://arxiv.org/pdf/2311.18677#page=7)

With layer-by-layer sending, what is left after the prefill is roughly the larger of one layer's share and the part of the transfer that did not fit under the computation:

$$
t_{\text{exposed}} \approx \max\!\left(\frac{t_{\text{transfer}}}{L},\ t_{\text{transfer}} - t_{\text{prefill}}\cdot\frac{L-1}{L}\right)
$$

where $$L$$ is the number of layers and $$t_{\text{prefill}}$$ the prefill time. For the 70B example over one card: the transfer (54 ms) is far shorter than the prefill (about 314 ms at the assumed efficiency), so only one layer's share is left: $$54/80 = 0.67$$ ms.

{{FIG:layerwise|The same 2.5 GiB transfer sent after the prefill or layer by layer. Arithmetic from kv_transfer.py.}}

Splitwise measured the effect on real hardware:

> [!PAPER] Patel et al., Splitwise · Section VI-A · page 9
> [![Splitwise: End-to-end impact. The latency impact of serially transferring the KV-cache grows up to 3% of the E2E with large prompts. However, Splitwise only incurs 0.8% of E2E. In a user-facing inference, the only visible impact of KV-cache transfer overhead is the latency for the second token. Splitwise adds a 16.5% latency to the second token, as compared to the 64% overhead from a serialized transfer. Overall, the transfer impact in Splitwise is hardly perceivable even in a user-facing inference.](/img/multigpu/paper-splitwise-result.png)](/img/multigpu/paper-splitwise-result.png)
>
> **What it says:** layer-wise transfer cut the cost from up to 3% of end-to-end latency to 0.8%, and the delay of the second token from +64% to +16.5%.
>
> [Read Section VI-A](https://arxiv.org/pdf/2311.18677#page=9)

### Mooncake: build the system around the KV cache

Moonshot AI's **Mooncake**, the serving platform behind their Kimi assistant, takes the idea one step further. If KV caches are going to travel between machines anyway, treat them as the central object: keep them in a pool spread across the cluster's spare CPU memory and SSDs, and schedule every request by where its cache already is.

> [!PAPER] Qin et al., Mooncake · Figure 1 · page 2
> [![Mooncake Figure 1, Mooncake Architecture: a KVCache-centric Conductor with a cache-aware prefill scheduler, a KVCache balance scheduler and a load-balance decoding scheduler; a prefill pool of instances with chunked prefill and paged KV cache in GPU memory; a distributed KVCache pool in CPU memory, DRAM and SSD connected by RDMA; and a decoding pool. The prefill stage optimises for maximum cache reuse subject to the TTFT SLO and MFU; the decoding stage optimises for maximum throughput subject to the TBT SLO.](/img/multigpu/paper-mooncake-fig1.png)](/img/multigpu/paper-mooncake-fig1.png)
>
> **Context:** the overview figure of the paper.
>
> **What it says:** a global scheduler ("Conductor") picks a prefill instance that already holds as much of the prompt's KV as possible, sends the result over RDMA to a decode instance with room, and balances the KV pool across machines.
>
> **Why it matters:** disaggregation and prefix caching (Parts 3 and 5) combine: the KV cache becomes a cluster-wide resource. Mooncake reports up to 525% more throughput in simulated long-context scenarios and 75% more requests handled under real Kimi workloads.
>
> [Read the paper](https://arxiv.org/abs/2407.00079)

DeepSeek's production numbers from Section 6 show the same pattern at scale: 56.3% of their input tokens hit an on-disk KV cache.

### A simulation: when is it worth it?

The papers report big wins against the systems of their time. Engines have improved since, especially with chunked prefill. So `disagg_sim.py` compares the options on equal hardware, under clearly stated assumptions:

- **4 H100s serving Llama 3.1 8B.** Each iteration costs $$\max(\text{bytes}/(0.7 \times 3.35\ \text{TB/s}),\ \text{FLOPs}/(0.5\times989\ \text{TFLOP/s})) + 1$$ ms, with attention FLOPs counted for prompts. The 70% and 50% efficiencies and the 1 ms overhead are assumptions.
- **Layouts:** four colocated GPUs with prefill first; four colocated GPUs with 512-token chunked prefill; and disaggregated 1+3, 2+2 and 3+1 prefill and decode GPUs, with the KV cache moved over NVLink.
- **Workloads:** "chat" with 1,000 to 3,000-token prompts and 100 to 400-token answers; "long" with 8,000 to 16,000-token prompts and 50 to 200-token answers. Poisson arrivals for 120 simulated seconds.
- **SLOs:** loose (TTFT 1 s, TPOT 40 ms), tight (TTFT 0.5 s, TPOT 15 ms) and strict (TTFT 1 s, TPOT 12 ms). Goodput is the highest rate at which 90% of requests meet both.

The heart of the simulator is the cost of one iteration and the three kinds of iteration a GPU can run:

```python
def iter_time(decode_reqs, prefill_tokens, extra_flops=0.0):
    kv = sum(r['ctx'] for r in decode_reqs) * KV_TOK          # every running sequence's cache is read
    toks = len(decode_reqs) + prefill_tokens
    return max((W_BYTES + kv) / HBM, (2 * P * toks + extra_flops) / PEAK) + OVH
```

A prefill-first GPU runs a whole-prompt iteration whenever prompts are waiting, so everyone decoding waits; a chunked GPU adds up to 512 prompt tokens to every decode iteration; a disaggregated prefill GPU only prefills, then schedules a hand-off event $$T \times 128\ \text{KiB} / 450\ \text{GB/s}$$ later, when the request joins the least-loaded decode GPU.

{{FIG:goodput|Simulated share of requests meeting the SLO as load grows, on 4 GPUs. Left: chat workload, tight SLO. Right: long-prompt workload, strict SLO. Goodput is where a curve crosses 90%.}}

[![Terminal output of disagg_sim.py: for the chat and long workloads, the goodput of each layout under the loose, tight and strict SLOs, with p90 TTFT, p90 TPOT, p99 gap between tokens and SLO attainment at several request rates](/img/multigpu/disagg_sim-run.png)](/img/multigpu/disagg_sim-run.png)

| Goodput, requests/s on 4 GPUs (simulated) | Chat, loose | Chat, tight | Chat, strict | Long, loose | Long, strict |
|---|---|---|---|---|---|
| Colocated, prefill first | 30 | 16 | 10 | 4 | 1 |
| Colocated, chunked prefill | **48** | **28** | 18 | **6** | 2 |
| Disaggregated 1P + 3D | 12 | 10 | 12 | 1 | 1 |
| Disaggregated 2P + 2D | 28 | 26 | **20** | 3 | **3** |
| Disaggregated 3P + 1D | 26 | 14 | 8 | 5 | 2 |

(No layout meets the tight SLO on the long workload: a 12,000-token prompt alone takes about half a second to prefill on one GPU.)

What the simulation says, honestly:

- **Disaggregation removes the stalls.** In the chat workload at 16 requests per second, prefill-first has a 99th-percentile gap between tokens of 99 ms (worst 236 ms); chunked prefill 19 ms; 2P + 2D 12 ms. On long prompts, prefill-first freezes some users for up to 1.4 seconds.
- **But on a handful of GPUs, chunked prefill usually wins on goodput.** With only 4 GPUs, the split must be 1+3, 2+2 or 3+1, and every ratio wastes some capacity. Chunked prefill uses all four GPUs for whatever work exists. Under the loose and tight SLOs it serves the most requests on both workloads.
- **Disaggregation wins when the time-per-token target is strict.** Under the strict 12 ms TPOT, 2P + 2D serves 20 chat requests per second against 18, and 3 long-prompt requests against 2, because chunked steps that carry prompt slices are slower than pure decode steps.
- **The ratio is everything.** The same four GPUs serve 12, 28 or 26 chat requests per second (loose SLO) depending on the split. Production systems pick and adjust the ratio continuously; DistServe searches for it automatically.

The vLLM documentation states the same conclusion in one line:

> [!PAPER] vLLM documentation · Disaggregated Prefilling (experimental) · Why disaggregated prefilling?
> [![vLLM docs: Why disaggregated prefilling? Two main reasons. Tuning time-to-first-token and inter-token-latency separately: disaggregated prefilling puts prefill and decode phase of LLM inference inside different vLLM instances, which gives you the flexibility to assign different parallel strategies (e.g. tp and pp) to tune TTFT without affecting ITL, or to tune ITL without affecting TTFT. Controlling tail ITL: without disaggregated prefilling, vLLM may insert some prefill jobs during the decoding of one request, which results in higher tail latency; chunked prefill with a proper chunk size also can achieve the same goal, but in practice it is hard to figure out the correct chunk size value. Note: Disaggregated prefill DOES NOT improve throughput.](/img/multigpu/vllm-disagg-doc.png)](/img/multigpu/vllm-disagg-doc.png)
>
> **Context:** the feature page of vLLM's disaggregated prefill, still marked experimental.
>
> **What it says:** the two reasons to disaggregate are tuning TTFT and inter-token latency separately (including different parallelism per phase) and controlling the tail of inter-token latency. It does not raise throughput.
>
> [Read the page](https://docs.vllm.ai/en/latest/features/disagg_prefill/)

So disaggregation is worth it when: prompts are long; the per-token latency target is strict; the deployment is large enough to choose the prefill-to-decode ratio finely (DeepSeek runs thousands of GPUs); the two phases benefit from different parallel layouts (DeepSeek's EP32 prefill against EP144 or EP320 decode); and there is fast networking for the KV cache. For one or two servers of a mid-sized dense model, chunked prefill on every GPU is the simpler and often better choice.

## 8. Real systems today

Everything above is available as flags in the open-source engines. Here is how each idea maps to them, checked against the current documentation (11 October 2026). Flags change between releases, so check the docs for your version.

### vLLM

The vLLM docs give the same rule of thumb this article arrived at:

> [!PAPER] vLLM documentation · Parallelism and Scaling
> [![vLLM docs: To choose a distributed inference strategy for a single-model replica: Single GPU (no distributed inference) if the model fits on a single GPU. Single-node multi-GPU using tensor parallel inference if the model is too large for a single GPU but fits on a single node with multiple GPUs, for example tensor_parallel_size=4 on a node with 4 GPUs. Multi-node multi-GPU using tensor parallel and pipeline parallel inference if the model is too large for a single node: set tensor_parallel_size to the number of GPUs per node and pipeline_parallel_size to the number of nodes, for example 8 and 2 for 2 nodes with 8 GPUs. Edge case: if the GPUs on the node do not have NVLINK interconnect (e.g. L40S), leverage pipeline parallelism instead of tensor parallelism for higher throughput and lower communication overhead.](/img/multigpu/vllm-parallelism-doc.png)](/img/multigpu/vllm-parallelism-doc.png)
>
> **Context:** the guide for choosing a layout for one copy of a model.
>
> **What it says:** one GPU if it fits; tensor parallel inside a node; tensor parallel inside nodes and pipeline parallel across them; and pipeline instead of tensor parallel on GPUs without NVLink.
>
> **Why it matters:** each rule is one of our results. TP needs fast links (Section 4); PP sends little across slow links (Section 5).
>
> [Read the page](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)

| Idea | vLLM flag | Note |
|---|---|---|
| Tensor parallel | `--tensor-parallel-size 8` | Inside one NVLink node |
| Pipeline parallel | `--pipeline-parallel-size 2` | Across nodes, with TP inside each |
| Data parallel (copies) | `--data-parallel-size 8` | With EP: attention copied, experts spread |
| Expert parallel | `--enable-expert-parallel` | EP size = TP size x DP size |
| All-to-all kernels | `--all2all-backend deepep_low_latency` | Also `deepep_high_throughput`, `allgather_reducescatter` (default) |
| Expert load balancing | `--enable-eplb`, `--eplb-config '{...}'` | Redundant experts, as in Section 6 |
| Disaggregated prefill | `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'` | Connectors include NIXL, Mooncake, LMCache; marked experimental |

Typical launches, from the documentation:

```bash
# Llama 3.1 70B on one 8-GPU node
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 8

# A model too big for one node: TP inside each of 2 nodes, PP across them (after joining the nodes as the docs describe)
vllm serve <model> --tensor-parallel-size 8 --pipeline-parallel-size 2

# DeepSeek-V3 on one node: attention copied 8 ways, experts spread over 8 GPUs
vllm serve deepseek-ai/DeepSeek-V3-0324 --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel
```

The expert-parallel page lists the all-to-all backends, split by the two phases exactly as DeepSeek splits them:

[![vLLM docs: vLLM provides multiple communication backends for EP, selected with --all2all-backend: allgather_reducescatter, the default, standard all2all using allgather and reducescatter primitives, general purpose; deepep_high_throughput for multi-node prefill, grouped GEMM with continuous layout, for prefill-dominated high-throughput workloads; deepep_low_latency for multi-node decode, CUDA graph support and masked layout, for decode-dominated low-latency workloads.](/img/multigpu/vllm-ep-doc.png)](/img/multigpu/vllm-ep-doc.png)

### SGLang

| Idea | SGLang flag | Note |
|---|---|---|
| Tensor parallel | `--tp-size` (or `--tensor-parallel-size`) | |
| Pipeline parallel | `--pp-size` | |
| Data parallel copies | `--dp-size` | |
| Data-parallel attention | `--attn-dp-size` | Replaces the older `--enable-dp-attention`, now deprecated |
| Expert parallel | `--ep-size` | |
| All-to-all kernels | `--moe-a2a-backend deepep` | Also mooncake, nixl, flashinfer and others |
| Expert load balancing | `--enable-eplb` | |
| Disaggregation | `--disaggregation-mode prefill` or `decode` | Transfer with `--disaggregation-transfer-backend mooncake` (default) or `nixl` |

SGLang's PD disaggregation page explains the motivation in the terms of Section 7, including a problem specific to data-parallel attention:

[![SGLang docs: Large Language Model inference comprises two distinct phases: Prefill and Decode. The Prefill phase is computation-intensive, processing the entire input sequence, while the Decode phase is memory-intensive, managing the KV cache for token generation. Traditionally these phases are handled within a unified engine, where combined scheduling of prefill and decode batches introduces inefficiencies. Issues with unified scheduling: 1. Prefill interruption: incoming prefill batches frequently interrupt ongoing decode batches, causing substantial delays in token generation. 2. DP attention imbalance: in data-parallel attention, one DP worker may process a prefill batch while another handles a decode batch simultaneously, leading to increased decode latency. PD Disaggregation resolves these by separating the two stages, enabling tailored optimizations for each. Currently, we support Mooncake and NIXL as the transfer engine.](/img/multigpu/sglang-pd-doc.png)](/img/multigpu/sglang-pd-doc.png)

A single-node disaggregated setup from that page (Llama 3.1 8B, one GPU each, Mooncake transfer) is three commands:

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill --port 30000 --disaggregation-ib-device mlx5_roce0
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode --port 30001 --base-gpu-id 1 --disaggregation-ib-device mlx5_roce0
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000
```

The router receives each request, sends it to a prefill server, and has the decode server pick up the KV cache over RDMA (the `mlx5_roce0` device is the RDMA network card). The page's DeepSeek-V3 example uses two prefill nodes with `--tp-size 16 --attn-dp-size 8 --moe-a2a-backend deepep`: attention in data-parallel groups, experts over all 16 GPUs with DeepEP.

### The orchestration layer: Dynamo and llm-d

Running disaggregation and wide expert parallelism in production needs more than one engine process: routers that know where each KV cache lives, a way to move KV between machines, and autoscaling of the prefill and decode pools separately. Two open-source projects package this around the engines:

- **[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)** describes itself as "the open-source, datacenter-scale inference stack". It runs on top of SGLang, TensorRT-LLM or vLLM ("it doesn't replace" them) and adds disaggregated prefill and decode pools that scale independently, KV-aware routing ("based on worker load and KV cache overlap"), and a KV block manager that offloads cache from GPU to CPU, SSD and remote storage. Its transfer library, [NIXL](https://github.com/ai-dynamo/nixl) (NVIDIA Inference Xfer Library), is also one of vLLM's KV connectors and SGLang's transfer backends. Latest release at the time of writing: v1.5.1, 7 October 2026.
- **[llm-d](https://github.com/llm-d/llm-d)**, a Cloud Native Computing Foundation sandbox project started by Red Hat, Google Cloud, IBM Research, CoreWeave and NVIDIA, is "a high-performance distributed inference serving stack optimized for production deployments on Kubernetes". Its documented ["well-lit paths"](https://llm-d.ai/docs/well-lit-paths) include prefix-cache-aware routing, prefill/decode disaggregation, and "wide expert parallelism" for large MoE models. Latest release: v0.10.0, 29 September 2026.

Both build on vLLM (and in Dynamo's case SGLang and TensorRT-LLM as well); neither replaces the parallelism inside an engine. They decide which engine instance a request goes to and where its KV cache moves, the cluster-level version of the routing in [Part 6](llm-inference-6-serving-in-production.md).

## 9. Choosing a layout

### A decision guide

{{FIG:decision|A first decision guide. Every branch is a rule from Sections 1 to 7; the last step is always to measure.}}

### Worked example 1: Llama 3.1 8B

Weights 16 GB, 128 KiB of KV per token. One H100 holds the model plus $$(72 - 16)\ \text{GB} / 128\ \text{KiB} \approx 427{,}000$$ tokens of KV cache: 52 conversations of 8,192 tokens. The decode floor is 4.8 ms per step on one GPU. **Use one GPU per copy, and add copies for more traffic.** Tensor parallelism would only add all-reduces to a model that already fits; at TP=2 the step floor halves to 2.4 ms, which is worth it only if a per-token latency target cannot be met otherwise. Disaggregation, by our simulation, does not raise goodput at this size unless the per-token target is very strict. Everything else about scaling this model is in [Part 6](llm-inference-6-serving-in-production.md).

### Worked example 2: Llama 3.1 70B on one 8-GPU node

The weights (141 GB) need at least two GPUs, and realistically more to leave room for KV. With 8 GPUs, three layouts are possible: one copy at TP=8, two copies at TP=4, or four copies at TP=2. The model in `layout_model.py` compares them for 4,096-token conversations:

{{FIG:node70b|Modelled Llama 3.1 70B on one 8-GPU H100 node: time per token against tokens per second for the whole node, as the batch per copy grows. Each curve ends where that layout runs out of KV-cache memory.}}

| Layout | KV room (4,096-token conversations) | Batch 1: ms per token | Batch 64 per copy: ms per token, node tokens/s | Most the node can do |
|---|---|---|---|---|
| TP=8, 1 copy | 324 | 6.9 | 10.7 ms, 5,969 (batch 128: 14.6 ms, 8,779) | 11,482 tokens/s at 22.3 ms (batch 256) |
| TP=4, 2 copies | 109 each, 218 total | 12.2 | 19.1 ms, 6,702 | 6,702 tokens/s at 19.1 ms (batch 64) |
| TP=2, 4 copies | 2 each | 22.9 | (does not fit) | 347 tokens/s |

**Worked numbers for TP=4.** Each copy has $$4\times72 = 288$$ GB usable; the weights take 141 GB, leaving 147 GB, which at $$320\ \text{KiB}\times4{,}096 = 1.25$$ GiB per conversation holds 109 conversations. For TP=8 the free memory is $$576 - 141 = 435$$ GB: **324** conversations, three times as many, because the weights are stored once instead of twice.

You may have heard "prefer more copies over wider tensor parallelism". For a 70B model on 80 GB GPUs this model says the opposite. At the same 128 sequences in flight on the node, one TP=8 copy makes **8,779 tokens per second at 14.6 ms per token**, while two TP=4 copies (64 each) make **6,702 at 19.1 ms**. Decode is memory-bound, and two copies read the 141 GB of weights twice per step where one copy reads them once. The extra all-reduces of TP=8 cost less than that second read. TP=8 also has three times the KV room, so it can go on to 11,482 tokens per second at batch 256, and it gives the lowest latency for a single user (6.9 against 12.2 ms). TP=2 barely fits the weights and is useless.

More copies do win in other conditions: when the weights are a small part of each step's memory traffic (long contexts, where the KV cache dominates), when the links are slow (no NVLink, so all-reduce is expensive), and for reasons a roofline ignores, such as isolating failures and scheduling requests independently. So the lesson is a method, not a rule: **count the KV room for each layout first, then compare latency and throughput at the batch sizes your traffic needs**, and confirm with a benchmark as in Part 6.

### Worked example 3: a large MoE across nodes

DeepSeek-V3 in FP8 needs 671 GB for weights: at least nine 80 GB GPUs, so more than one node. Its attention is small and its KV cache tiny (68.6 KiB per token), so copying attention is cheap; its experts are huge, so spreading them is the only option. The layouts from Section 6 follow:

- **Small deployment (2 nodes, 16 GPUs):** attention in data-parallel groups, experts spread over all 16 GPUs with EP, DeepEP for the all-to-all. This is SGLang's documented two-node example.
- **Large deployment (dozens of nodes):** separate prefill and decode units, with wider EP for decode (EP144 to EP320 in DeepSeek's case), redundant copies of hot experts, and two micro-batches to hide the all-to-all.

The all-to-all cost from Section 6 sets the limits. At about 11 ms of dispatch and combine per decode step at EP8 (more at wider EP), the communication must be overlapped with computation, and the network matters as much as the GPUs. That is why DeepSeek limits each token to 4 nodes, and why the decode units are so large: many GPUs reading their own experts in parallel is what makes a 37-billion-active-parameter step fast.

## 10. Summary

### The whole part, on one page

| Question | Answer | Where the number comes from |
|---|---|---|
| Why more than one GPU? | 70B needs 227 GB with KV for 32 x 8K conversations; 405B BF16 weights (812 GB) exceed one 8-GPU node | Arithmetic from published shapes |
| What does spreading buy? | Each GPU reads 1/p of the weights: 70B decode floor 21.1 ms on 2 GPUs, 5.3 ms on 8 | Arithmetic, H100 peak bandwidth |
| How fast are the wires? | HBM 3,350, NVLink 900, PCIe 128, one 400G NIC 50 GB/s | NVIDIA spec pages |
| What is a message's cost? | $$\alpha + n/\beta$$: small messages cost a fixed delay | Measured shape on this laptop; NVIDIA publishes no alpha |
| Tensor parallelism | Columns then rows: 2 all-reduces per layer; exact (same 24 tokens, logits within $$5\times10^{-5}$$) | Measured: 2-process Qwen2.5-0.5B run |
| Its cost | 160 all-reduces per 70B step; 23% of a TP=8 step at batch 1 | Model, $$\alpha$$ assumed 5 $$\mu$$s |
| Ring all-reduce | Bytes $$\frac{2(p-1)}{p}n$$, but $$2(p-1)$$ steps; two-step algorithms cut latency 7x at TP=8 | Simulation and formula |
| Pipeline parallelism | Little traffic, no latency gain; with micro-batches 1.7x the throughput of TP=16 across two nodes | Model, 405B |
| Expert parallelism | 10 MB of all-to-all per token for DeepSeek-V3; real routing makes the busiest GPU 1.3x to 6.8x the average | DeepEP report checked by arithmetic; measured OLMoE routing |
| Fixing MoE imbalance | Place by load (1.81 to 1.12 at EP=8 on code); copy hot experts (6.75 to 1.67 at EP=64) | Simulation on measured routing |
| Disaggregation | Moves 2.5 GiB per 8K-token 70B prompt (54 ms on one NIC, under 1 ms exposed layer by layer); wins under strict TPOT and at scale, not by default | Arithmetic; simulation |

Return to where we started. A 70B model does not fit on one GPU, so it is cut into pieces that must talk. Inside a server, where talking is cheap, every layer is split (tensor parallelism) and the GPUs add up their partial results 160 times per token. Between servers, where talking is slow, the model is cut into stages (pipeline parallelism) that pass one message per step. Mixture-of-experts models send each token to the GPUs holding its experts (expert parallelism), and the real difficulty is that some experts are far more popular than others. Finally, the two phases of every request can live on different GPUs (disaggregation), at the price of moving the KV cache, which pays off for long prompts, strict per-token targets and large fleets. In every case the design follows one question: **how often must the GPUs talk, and how fast is the wire?**

### The series so far

| Part | Topic | The one idea |
|---|---|---|
| [1](llm-inference-1-prefill-and-decode.md) | Prefill and decode | Decode is limited by reading the weights, not by arithmetic |
| [2](llm-inference-2-kv-cache.md) | The KV cache | Saving keys and values avoids recomputation, and their size limits the batch |
| [3](llm-inference-3-vllm.md) | vLLM | Paged memory, continuous batching and chunked prefill keep one GPU busy |
| [4](llm-inference-4-speculative-decoding.md) | Speculative decoding | Checking several guessed tokens costs about as much as writing one |
| [5](llm-inference-5-sglang-vs-vllm.md) | SGLang and vLLM | Reuse the saved state of shared prompt beginnings |
| [6](llm-inference-6-serving-in-production.md) | Serving in production | Measure goodput against SLOs, plan capacity, route between copies |
| 7 | Beyond one GPU | Split the model where the wires are fast; the cost of talking decides the layout |

## Try it yourself

All the code is in [`code/multigpu`](https://github.com/ishwar6/ishwar-books/tree/main/code/multigpu). Nothing needs an NVIDIA GPU.

```bash
python code/multigpu/memory_math.py     # Section 1: memory and the decode floor
python code/multigpu/tp_demo.py         # Section 3: split MLP, 2-process Qwen2.5-0.5B, all-reduce timings
python code/multigpu/measure_mps.py     # Section 3: shard timings (needs an Apple GPU; edit dev for CUDA)
python code/multigpu/collectives.py     # Section 4: ring trace, collectives, cost tables, MoE bytes
python code/multigpu/layout_model.py    # Sections 3, 5 and 9: layouts for 70B and 405B
python code/multigpu/moe_routing.py     # Section 6: records OLMoE routing (14 GB download), then simulates EP
python code/multigpu/kv_transfer.py     # Section 7: KV transfer arithmetic
python code/multigpu/disagg_sim.py      # Section 7: colocated vs disaggregated goodput
```

If you have a machine with two or more NVIDIA GPUs, change the backend in `tp_demo.py` from `gloo` to `nccl` and put each rank's tensors on `cuda:<rank>`; the same code then runs real tensor parallelism over NVLink or PCIe, and the all-reduce timings in part C become your own $$\alpha$$ and $$\beta$$.

## References

**Papers**

1. M. Shoeybi et al. [*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*](https://arxiv.org/abs/1909.08053). arXiv 1909.08053, 2019.
2. Y. Huang et al. [*GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*](https://arxiv.org/abs/1811.06965). NeurIPS 2019.
3. R. Pope et al. [*Efficiently Scaling Transformer Inference*](https://arxiv.org/abs/2211.05102). MLSys 2023.
4. Llama Team, AI @ Meta. [*The Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783). 2024. Section 6, Inference.
5. DeepSeek-AI. [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437). 2024. Sections 3.2.2 and 3.4.
6. N. Shazeer et al. [*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538). ICLR 2017.
7. D. Lepikhin et al. [*GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*](https://arxiv.org/abs/2006.16668). 2020.
8. P. Patel et al. [*Splitwise: Efficient Generative LLM Inference Using Phase Splitting*](https://arxiv.org/abs/2311.18677). ISCA 2024.
9. Y. Zhong et al. [*DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*](https://arxiv.org/abs/2401.09670). OSDI 2024.
10. R. Qin et al. [*Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*](https://arxiv.org/abs/2407.00079). 2024.
11. A. Agrawal et al. [*SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*](https://arxiv.org/abs/2308.16369). 2023.
12. N. Muennighoff et al. [*OLMoE: Open Mixture-of-Experts Language Models*](https://arxiv.org/abs/2409.02060). 2024. The model used in Section 6: [OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924).

**Documentation, specifications and engineering posts**

13. NVIDIA. [H100](https://www.nvidia.com/en-us/data-center/h100/), [H200](https://www.nvidia.com/en-us/data-center/h200/), [NVLink and NVLink Switch](https://www.nvidia.com/en-us/data-center/nvlink/), [DGX H100 user guide](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html), [InfiniBand adapters (ConnectX-7)](https://www.nvidia.com/en-us/networking/infiniband-adapters/).
14. NVIDIA. [*3x Faster AllReduce with NVSwitch and TensorRT-LLM MultiShot*](https://developer.nvidia.com/blog/3x-faster-allreduce-with-nvswitch-and-tensorrt-llm-multishot/). Technical blog, November 2024.
15. NVIDIA. [nccl-tests performance notes](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) (algorithm and bus bandwidth).
16. vLLM. [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/), [Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/), [Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/).
17. SGLang. [Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments), [PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation).
18. DeepSeek. [DeepEP](https://github.com/deepseek-ai/DeepEP) (performance tables in the [v1.2.1 README](https://github.com/deepseek-ai/DeepEP/tree/v1.2.1)); [DeepSeek-V3/R1 inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md), February 2025.
19. [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) and [documentation](https://docs.nvidia.com/dynamo/); [llm-d](https://github.com/llm-d/llm-d) and its [well-lit paths](https://llm-d.ai/docs/well-lit-paths).

**Companion results**

20. [Memory arithmetic](https://github.com/ishwar6/ishwar-books/blob/main/code/multigpu/results/memory_math.json), [tensor-parallel demo](https://github.com/ishwar6/ishwar-books/blob/main/code/multigpu/results/tp_demo.json), [shard timings](https://github.com/ishwar6/ishwar-books/blob/main/code/multigpu/results/measure_mps.json), [collectives](https://github.com/ishwar6/ishwar-books/blob/main/code/multigpu/results/collectives.json), [layouts](https://github.com/ishwar6/ishwar-books/blob/main/code/multigpu/results/layout_model.json), [MoE routing](https://github.com/ishwar6/ishwar-books/blob/main/code/multigpu/results/moe_routing.json), [KV transfer](https://github.com/ishwar6/ishwar-books/blob/main/code/multigpu/results/kv_transfer.json), [disaggregation](https://github.com/ishwar6/ishwar-books/blob/main/code/multigpu/results/disagg_sim.json).
