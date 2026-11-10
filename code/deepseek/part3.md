---
title: "How DeepSeek-V3 Was Trained on 2,048 GPUs"
description: "How DeepSeek-V3 was trained: 14.8T tokens on 2,048 GPUs, a schedule that hides network time, 8-bit numbers, and what the $5.576M covers."
date: 2026-11-10
tags: [deepseek, training, fp8, parallelism, llm]
series: "DeepSeek, From the Ground Up"
series_part: 3
motif: grid
accent: "#5b9cf5"
---

In 2024, a cluster of 2,048 GPUs read 14.8 trillion tokens of text, one batch at a time, for less than two months. At the end of it stood DeepSeek-V3, the base of R1, V3.1 and V3.2. The report gives one more number, and it travelled further than the model. The run took 2.788 million GPU hours, or **$5.576M** at $2 per hour.

[Part 2](deepseek-2-the-v3-architecture.md) took the machine apart: 671B parameters, 37B of them active for each token, experts, MLA and multi-token prediction. This part is about how that machine was trained. We follow the run from the data to the bill. On the way we meet the hardware and the plan that splits one model over 2,048 GPUs. Then come a schedule that hides network time, and 8-bit numbers that do not break.

We also run two small experiments on a laptop CPU. One shows why DeepSeek scales 8-bit numbers in groups of 128. The other shows why it moves its running sums to 32-bit every 128 terms.

> [!TIP] The whole idea in one minute
> V3 was pre-trained on **14.8T tokens** with **4K-token** sequences, then taught to read **128K** tokens in two short extra phases. It ran on **2,048 NVIDIA H800** GPUs, a version of the H100 with slower links between GPUs. DeepSeek cut the model into **16 pipeline stages** and spread the experts of each layer over **64 GPUs**. It never split single matrices across GPUs. A schedule called **DualPipe** lets the network move data while the GPU computes.
>
> Most matrix products ran in **FP8**, an 8-bit number format, with one scale per small group of numbers. The report states no loss spikes that needed a restart. The whole run cost **2.788M GPU hours**. The $5.576M is that number times an assumed $2 rental price, and it leaves out research, hardware, staff and data.

{{FIG:timeline|The whole training of DeepSeek-V3. Pre-training on 4K-token sequences takes 95.6% of the GPU hours. Two short phases then stretch the context to 32K and 128K tokens, and post-training (Part 4) takes only 5K GPU hours.}}

Pre-training is next-token prediction with a cross-entropy loss; the [training overview](how-models-are-trained-today.md) on this site walks through the loop. What matters here is the scale. V3 took about 243,000 steps (our estimate, below), and each step read up to 63 million tokens.

## The data: 14.8 trillion tokens

DeepSeek says little about its data, but what it says is specific. The report changes three things compared with V2: more maths and code, more languages, and less repetition.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §4.1 · pages 21 and 22
> [![The opening of section 4.1 on page 21, with enhancing the ratio of mathematical and programming samples and expanding multilingual coverage highlighted](/img/deepseek/papers/p3-data.png)](/img/deepseek/papers/p3-data.png)
> [![Page 22: the rest of the data paragraph, the FIM paragraph with the PSM format and the tokenizer paragraph, with do not incorporate cross-sample attention masking, 14.8T high-quality and diverse, the FIM rate of 0.1, Byte-level BPE and 128K tokens highlighted](/img/deepseek/papers/p3-data2.png)](/img/deepseek/papers/p3-data2.png)
>
> **What it says:** the corpus raises "the ratio of mathematical and programming samples" and widens languages beyond English and Chinese. Documents are packed together, but the authors "do not incorporate cross-sample attention masking". The corpus holds "14.8T high-quality and diverse tokens". Fill-in-the-middle is "applied at a rate of 0.1". The tokenizer is "Byte-level BPE" with "128K tokens".
>
> **What it means:**
>
> A training sequence is 4,096 tokens long, but most documents are shorter. So the pipeline packs several documents end to end into one sequence, like suitcases in a car boot. Without a mask, a token at the start of document two can still look back at document one. DeepSeek accepts that small leak. V4 changed its mind: its report says it now masks between documents.
>
> The report gives no table of sources and no percentages. It does not say how many tokens are code or maths, or how often a document repeats. We know the direction of the changes, not their size.

> [!DEFINITION] Fill-in-the-middle (FIM)
> A way to teach a left-to-right model to fill a gap. Cut a document into prefix, middle and suffix, then reorder it as prefix, suffix, middle, with special marker tokens. The model still only predicts the next token, but now it learns to write a middle that fits both sides. Code editors need exactly this.

{{FIG:fim|Fill-in-the-middle in the PSM order of the V3 report. The file is cut in three, and the middle moves to the end, so ordinary next-token training learns to complete a gap. V3 does this to 10% of documents.}}

The tokenizer paragraph hides a small fix. Some new tokens join punctuation with a line break. That compresses text, but a prompt that ends without a line break then looks unusual to the model. So during training, DeepSeek randomly splits some joined tokens back into two, and the model learns both forms.

<details><summary>Research notes: what the R1 paper later added about the data</summary>

The Nature version of the R1 paper (arXiv 2501.12948 v2, Supplementary A.1, page 13) adds a disclosure about the V3 base: "we exclusively use plain web pages and e-books, without incorporating any synthetic data." It then admits that "some web pages contain a significant number of OpenAI-model-generated answers", so the base model may learn from other models indirectly. The same document gives the knowledge cutoff of V3-Base as July 2024 and describes 10-gram decontamination that removed about six million potential maths texts (page 40). None of this is in the V3 report itself.

</details>

## The schedule: learning rate and batch size

A training run does not use one learning rate from start to end. V3's schedule has four parts, and the report gives each one in tokens.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §4.2 · pages 22 and 23
> [![The training hyper-parameters paragraph on page 22, with the warm-up to 2.2e-4 in 2K steps, the constant rate until 10T tokens, the cosine decay to 2.2e-5 in 4.3T tokens and the final 500B tokens highlighted](/img/deepseek/papers/p3-lr.png)](/img/deepseek/papers/p3-lr.png)
> [![Page 23: the batch size grows from 3072 to 15360 in the first 469B tokens, and routed experts are spread over 64 GPUs on 8 nodes, highlighted](/img/deepseek/papers/p3-batch.png)](/img/deepseek/papers/p3-batch.png)
>
> **What it says:** the learning rate rises "linearly" from 0 to 2.2 × 10⁻⁴ "during the first 2K steps". It stays constant "until the model consumes 10T" tokens, then decays to 2.2 × 10⁻⁵ "in 4.3T tokens, following a cosine decay curve". The "final 500B tokens" use 2.2 × 10⁻⁵ for 333B tokens, then 7.3 × 10⁻⁶ for 167B. The "batch size is gradually increased from 3072 to 15360 in the training of the first 469B tokens".
>
> **What it means:**
>
> Think of learning to park a car. At first you make big turns of the wheel, because you are far from the spot. Near the end you make tiny corrections. A learning rate schedule does the same.
>
> The long flat part covers two thirds of the run. The cosine part slowly brakes. The last two small steps are the final, careful touches.
>
> The batch ramp follows the same logic from the other side. Early on, the weights are bad and almost any batch points in a useful direction, so small, cheap batches are enough. Later, the useful signal is subtle and noisy, and a bigger batch averages away more noise. The report does not say how the batch grows: in steps or in a straight line.

The cosine part has a standard formula. The report names the curve but does not print it, so this is the usual form:

$$
\eta(t) = \eta_{\min} + \frac{1}{2}\left(\eta_{\max} - \eta_{\min}\right)\left(1 + \cos\frac{\pi t}{T_{d}}\right)
$$

Here $$\eta(t)$$ is the learning rate after $$t$$ tokens of the decay phase. $$\eta_{\max} = 2.2 \times 10^{-4}$$ is where it starts and $$\eta_{\min} = 2.2 \times 10^{-5}$$ is where it ends. $$T_d = 4.3\text{T}$$ is the length of the decay. At $$t = 0$$ the cosine is 1 and $$\eta = \eta_{\max}$$; at $$t = T_d$$ it is $$-1$$ and $$\eta = \eta_{\min}$$.

Check the middle by hand. At $$t = T_d / 2$$ the cosine is 0, so $$\eta = 2.2 \times 10^{-5} + 0.5 \times 1.98 \times 10^{-4} = 1.21 \times 10^{-4}$$. Our script prints the same value at 12.15T tokens. The four parts also add up: 10 + 4.3 + 0.333 + 0.167 = 14.8T tokens.

{{FIG:schedule|The two schedules of V3's pre-training, drawn from the numbers on pages 22 and 23. The learning rate is flat for 10T tokens, then brakes along a cosine. The batch grows five times over the first 469B tokens; the report does not give the shape of that ramp.}}

How big is a step? At the full batch, $$15{,}360 \times 4{,}096 = 62{,}914{,}560$$ tokens. If the ramp is a straight line (our assumption), the first 469B tokens take about 15,000 steps and the rest take about 227,800. That gives about **243,000 steps** for the whole run.

```
2. Tokens per optimizer step
   start: 3,072 x 4,096 = 12,582,912 tokens   full: 15,360 x 4,096 = 62,914,560 tokens
   steps in the 469B-token ramp (linear ramp assumed): 14,997
   steps after the ramp: (14.8T - 469B) / 62,914,560 = 227,785
   total: about 242,782 steps
```

The same paragraph sets the other knobs. AdamW with $$\beta_1 = 0.9$$, $$\beta_2 = 0.95$$ and weight decay 0.1, gradient clipping at 1.0, initial weights with standard deviation 0.006. Two knobs from Part 2 also change during the run. The multi-token prediction loss weight drops from 0.3 to 0.1 after 10T tokens. The bias update speed of the aux-loss-free balancer is 0.001 until 14.3T tokens, then 0. So the balancing biases stop moving for the last 500B tokens.

## Stretching the context to 128K tokens

Pre-training used 4K-token sequences, because attention costs grow with sequence length. A chat model must read much longer inputs. So after pre-training, DeepSeek ran two short phases on longer sequences.

A model trained at 4K positions has never seen the RoPE angles of position 100,000 ([Part 2](deepseek-2-the-v3-architecture.md) explains RoPE).

> [!DEFINITION] YaRN
> A recipe to stretch RoPE to longer inputs. It slows down the rotations, so long positions map to angles the model already knows. It treats the slow and fast parts of the rotation differently, and scales attention up a little to keep it sharp. A short training phase then lets the model adapt.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §4.3 · page 23
> [![Section 4.3 on page 23, with two additional training phases each comprising 1000 steps, from 4K to 32K and then to 128K, 32K with batch size 1920 and 128K with batch size 480 highlighted](/img/deepseek/papers/p3-yarn.png)](/img/deepseek/papers/p3-yarn.png)
>
> **What it says:** after pre-training, DeepSeek applies YaRN and runs "two additional training phases, each comprising 1000 steps", growing the window "from 4K to 32K and then to 128K". Phase 1 uses "32K, and the batch size is 1920". Phase 2 uses 128K, and "the batch size is reduced to 480". YaRN acts only on the decoupled RoPE key, with scale $$s = 40$$, and the learning rate is $$7.3 \times 10^{-6}$$ in both phases.
>
> **What it means:**
>
> Look at the two batch sizes. Four times longer sequences, four times fewer of them. So each step holds the same number of tokens: $$1{,}920 \times 32{,}768$$ and $$480 \times 131{,}072$$ are both about 62.9M. Each phase reads about 62.9B tokens, less than half a percent of pre-training.
>
> "Only on the decoupled key" ties back to MLA in [Part 2](deepseek-2-the-v3-architecture.md). In MLA, position lives in a small separate 64-dimensional part of each key. Only that part rotates, so only that part needs stretching. The learning rate is the final one of pre-training, so the model changes gently.

The YaRN attention factor in the report is $$\sqrt{t} = 0.1 \ln s + 1$$. With $$s = 40$$, that is $$0.1 \times 3.689 + 1 = 1.369$$: attention scores get about 37% sharper to make up for the longer, flatter spread over many tokens.

These two phases cost 119K GPU hours for about 126B tokens. That is about 946K GPU hours per trillion tokens, 5.3 times the pre-training rate, because long sequences make attention expensive. The V3.1 model card, months later, gives a useful cross-check. It says the 32K phase "has been increased 10-fold to 630B tokens" and the 128K phase "extended by 3.3x to 209B tokens". Our numbers give 10.0 and 3.32 times the V3 phases.

## The hardware: 2,048 H800 GPUs

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §3.1 · page 11
> [![Section 3.1 Compute Clusters, with 2048 NVIDIA H800 GPUs, 8 GPUs connected by NVLink and NVSwitch within nodes and InfiniBand interconnects highlighted](/img/deepseek/papers/p3-cluster.png)](/img/deepseek/papers/p3-cluster.png)
>
> **What it says:** V3 is trained on a cluster with "2048 NVIDIA H800 GPUs". Each node has "8 GPUs connected by NVLink and NVSwitch within nodes", and nodes talk over "InfiniBand (IB) interconnects".
>
> **What it means:**
>
> 2,048 GPUs in nodes of 8 is 256 nodes. Inside a node, GPUs share a fast private network. Between nodes, data goes through network cards and switches, which are slower. Almost every design choice in the rest of this part follows from that gap.

What is an H800? DeepSeek's later hardware paper, shown at the ISCA 2025 conference (we call it the ISCA paper), answers it.

> [!PAPER] Zhao et al. (2025), Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures · §4.1 and §4.2 · page 7
> [![Section 4.1 of the ISCA paper: the H800 is similar to the H100 but has reduced FP64 performance and NVLink bandwidth for regulatory compliance, from 900 GB/s to 400 GB/s, with eight 400G InfiniBand CX7 NICs per node, and section 4.2 avoidance of tensor parallelism because of limited NVLink bandwidth, highlighted](/img/deepseek/papers/p3-isca-h800.png)](/img/deepseek/papers/p3-isca-h800.png)
>
> **What it says:** the H800 is "built on the Hopper architecture, similar to the H100", but with "reduced FP64 computational performance and NVLink bandwidth for regulatory compliance". NVLink drops "from 900 GB/s to 400 GB/s". To make up for it, each node has "eight 400G Infiniband (IB) CX7 NICs". The first design rule that follows is "Avoidance of Tensor Parallelism (TP)", because of its "inefficiency under limited NVLink bandwidth".
>
> **What it means:**
>
> The H800 is the H100 made for export to China under US rules. The calculating part is the same Hopper chip; FP64 is the 64-bit maths of scientific computing, which language models do not use. What hurts is NVLink: under half the bandwidth inside a node.
>
> So DeepSeek had a cluster where the link inside a node was weakened, and the links between nodes were normal. Usable speeds were about 160 GB/s over NVLink and 50 GB/s per InfiniBand card (V3 report, page 13). A plan built for H100s would be slow on this machine. DeepSeek built a plan for this machine.

{{FIG:node|One H800 node. Eight GPUs share an NVSwitch, and each GPU has its own 400 Gb/s InfiniBand card to the rest of the cluster. The NVLink figure is 400 GB/s per GPU on the H800, against 900 GB/s on the H100.}}

This is the same habit as the Fire-Flyer 2 cluster in [Part 1](deepseek-1-every-model-and-what-is-open.md): accept a limited link, then hide it with software and scheduling.

## The parallelism plan

One GPU cannot train V3. In BF16, the weights alone are about 1.3 TB; with gradients and optimizer state, our estimate below is about 9.6 TB. The work must be split. There are four standard ways to split it.

> [!DEFINITION] Pipeline parallelism (PP)
> The layers are cut into consecutive **stages**, one per GPU group, like stations on an assembly line. A batch is cut into **micro-batches** that flow through the stages one after another.

> [!DEFINITION] Tensor parallelism (TP)
> Single weight matrices are cut across GPUs. Every layer then needs the GPUs to swap partial results several times, so TP needs a very fast link and usually stays inside one node.

> [!DEFINITION] ZeRO-1
> A memory saver for data parallelism. The optimizer state is the AdamW moments plus the 32-bit master weights. With ZeRO-1, each of N copies keeps only 1/N of it and updates only its own slice.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §3.2 · page 12
> [![Section 3.2 Training Framework, with 16-way Pipeline Parallelism, 64-way Expert Parallelism spanning 8 nodes and ZeRO-1 Data Parallelism highlighted](/img/deepseek/papers/p3-framework.png)](/img/deepseek/papers/p3-framework.png)
>
> **What it says:** training runs on HAI-LLM, DeepSeek's own framework. V3 "applies 16-way Pipeline Parallelism (PP), 64-way Expert Parallelism (EP) spanning 8 nodes, and ZeRO-1 Data Parallelism (DP)". The same page says careful memory work lets them train "without using costly Tensor Parallelism (TP)".
>
> **What it means:**
>
> Notice what is missing: tensor parallelism. It is the standard tool for big dense models, and it is exactly the one that needs fast NVLink. On the H800, DeepSeek dropped it.
>
> The three that remain fit the hardware. Pipeline parallelism sends only one activation tensor from stage to stage, which is cheap. Expert parallelism is heavy, but its traffic can be planned. Data parallelism only averages gradients, and that can overlap with the backward pass.

The report gives the three degrees but not the full layout. Here is our reading, using only arithmetic. 2,048 GPUs over 16 stages is 128 GPUs per stage, or 16 nodes. Expert parallelism spans 64 GPUs on 8 nodes, so each stage holds two expert-parallel groups. Each layer has 256 routed experts, and 256 / 64 = 4 experts per GPU; the ISCA paper confirms "4 experts per GPU" for this 8-node setup. The two groups hold the same experts and read different data, so they are data-parallel copies of each other.

{{FIG:parallel|Our reading of V3's layout. Sixteen pipeline stages of about four layers each. Each stage runs on 128 GPUs: two expert-parallel groups of 64 GPUs on 8 nodes. Inside a group, each GPU holds 4 of the 256 routed experts of every layer in its stage.}}

Does a stage fit in memory? Our script makes a rough budget. It assumes even stages and DualPipe's two stages per GPU (next section). It uses the formats the report states: BF16 weights, 32-bit gradients and master weights, BF16 AdamW moments.

```
7. Rough memory budget per GPU (DERIVED; our reading of the plan, not a number from the paper)
   parameters outside the routed experts: 17.5B   routed experts: 665.2B   (main model 671.0B + MTP module)
   per GPU: 2.18B non-expert + 1.30B routed-expert parameters = 3.48B (4 routed experts per layer per GPU)
   BF16 weights + FP32 gradients: 20.9 GB
   FP32 master weights + BF16 Adam moments (8 bytes per parameter): 27.8 GB without sharing, 5.3 GB with ZeRO-1
   total for the model state: 26.2 GB of 80 GB; the rest is for activations and buffers
   the whole model state on ONE GPU would be 9.6 TB
```

The surprise is how small the expert share is. Routed experts are 97% of the parameters, but a GPU holds only 4 of 256 per layer, so they shrink 64 times. What stays large is the attention and shared part, which every GPU of a stage keeps whole. ZeRO-1 cuts its optimizer state 128 ways. The 80 GB is our assumption: the papers do not state the H800's memory, and the H100 SXM has 80 GB.

## DualPipe: hiding the network behind the compute

### The bubble problem

A pipeline has a well-known flaw. At the start, stage 2 waits for stage 1 to finish the first micro-batch. At the end, stage 1 is done while the last micro-batch is still in stage 4. These idle periods are **bubbles**.

Our toy simulator shows it with 4 stages and 8 micro-batches. A forward pass takes 1 time unit and a backward pass takes 2 (backward does about twice the arithmetic). The simulator runs two classic orders: GPipe, all forwards first and then all backwards, and 1F1B, which alternates as soon as it can.

{{FIG:bubbles|Our toy pipeline simulation with 4 stages and 8 micro-batches. In both orders, each GPU sits idle for 9 of 33 time units. 1F1B does not shrink the bubble; it starts backward passes earlier, so each GPU stores fewer activations at once.}}

```
6b. Toy simulation: 4 pipeline stages, 8 micro-batches, F = 1, B = 2
   GPipe  total time 33, idle time per device 9 = 27% of the run (formula (S-1)(F+B) = 9)
   1F1B   total time 33, idle time per device 9 = 27% of the run (formula (S-1)(F+B) = 9)
   the difference between them is memory: at its peak, stage 0 holds activations of 8 micro-batches in GPipe and 4 in 1F1B
```

The idle time follows a simple rule. The first micro-batch must cross every stage before the last stage starts, and the last one must cross back. So each stage waits for $$PP - 1$$ forward and backward passes in total:

$$
\text{bubble}_{\text{1F1B}} = (PP - 1)(F + B)
$$

$$PP$$ is the number of pipeline stages, $$F$$ the time of one forward chunk and $$B$$ the time of one backward chunk. With 4 stages, $$F = 1$$ and $$B = 2$$, that is $$3 \times 3 = 9$$ units, exactly what the simulator measured. More micro-batches make the useful work longer, so the bubble becomes a smaller share, but it never goes away.

### The communication problem

V3 has a second, bigger problem. Each MoE layer sends every token to its experts and back, across nodes. The report measures how bad that is.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §3.2.1 · page 12
> [![The start of section 3.2.1, with an inefficient computation-to-communication ratio of approximately 1:1 highlighted](/img/deepseek/papers/p3-dualpipe-ratio.png)](/img/deepseek/papers/p3-dualpipe-ratio.png)
>
> **What it says:** cross-node expert parallelism gives "an inefficient computation-to-communication ratio of approximately 1:1". The key idea of DualPipe "is to overlap the computation and communication within a pair of individual forward and backward chunks". Each chunk is split into "attention, all-to-all dispatch, MLP, and all-to-all combine".
>
> **What it means:**
>
> A 1:1 ratio means the GPU spends as long waiting for the network as it spends computing. Run naively, half the cluster's time is waiting. That is far worse than the bubble.
>
> The way out is an old one: do two things at once. A GPU can compute on some of its cores while other cores drive the network. The trick is to always have something to compute while the network is busy. A forward chunk of one micro-batch and a backward chunk of another are independent, so one can send while the other computes.

{{FIG:overlap|A toy timing of the idea, with sending as slow as computing. Run one after the other, two forward and backward pairs take 12 units. Overlapped, the network sends one chunk while the GPU computes the next, and they take 8. With long runs, each pair costs half.}}

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · Figure 4 · page 12
> [![Figure 4: a computation row with MLP backward, MLP weights, MLP forward, attention backward, attention weights and attention forward, above a communication row with dispatch and combine for forward and backward and a PP block](/img/deepseek/papers/p3-fig4.png)](/img/deepseek/papers/p3-fig4.png)
>
> **What it says:** the figure shows one forward chunk and one backward chunk woven together. The top row is computation, the bottom row is communication. "Both all-to-all and PP communication can be fully hidden."
>
> **What it means:**
>
> Read it column by column. While the GPU computes the MLP backward of one micro-batch, the network dispatches tokens for the forward of the other. While it computes attention, the network combines results. No column has the network working alone.
>
> The backward is split in two: "backward for input" (green), which the previous stage needs soon, and "backward for weights" (blue), which can wait. That trick comes from the ZeroBubble paper. The weight part becomes filler that plugs holes in the schedule.

### Feeding the pipeline from both ends

The name DualPipe comes from the full schedule. Micro-batches enter from both ends of the pipeline at once.

{{FIG:dualdir|A sketch of DualPipe's two directions on four GPUs. One copy of the model runs left to right, another right to left. Each GPU holds one stage of each copy, so it stores twice the parameters but rarely runs out of work.}}

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · Figure 5 and Table 2 · page 13
> [![Figure 5: the DualPipe schedule for 8 devices and 20 micro-batches, with forward cells in orange, backward in green and blue, and overlapped pairs outlined](/img/deepseek/papers/p3-fig5.png)](/img/deepseek/papers/p3-fig5.png)
> [![Table 2: bubble and memory of 1F1B, ZB1P and DualPipe. DualPipe has bubble (PP/2 - 1)(F&B + B - 3W), 2x parameters and PP + 1 activations](/img/deepseek/papers/p3-table2.png)](/img/deepseek/papers/p3-table2.png)
>
> **What it says:** an example schedule "for 8 PP ranks and 20 micro-batches in two directions". Table 2 compares bubbles and memory: 1F1B has $$(PP-1)(F+B)$$, ZB1P has $$(PP-1)(F+B-2W)$$, DualPipe has $$(\tfrac{PP}{2}-1)(F\&B+B-3W)$$, with "2×" the parameters and $$PP+1$$ activations.
>
> **What it means:**
>
> In Figure 5, device 0 starts with micro-batches 0 to 6 of the forward direction, while device 7 starts the reverse direction. They meet in the middle, and from then on almost every cell is a double cell: a forward and a backward overlapped. The white gaps are the remaining bubbles, and there are few.
>
> The price is in the table: two copies of the parameters. The report says this "does not significantly increase the memory consumption since we use a large EP size". Our budget above agrees: expert weights shrink 64 times, so a second stage on each GPU costs little.

Put toy numbers into Table 2. Take $$PP = 16$$, $$F = 1$$, $$B = 2$$ with $$W = 1$$ of it for weights, and assume no gain from overlapping compute, so $$F\&B = 3$$. Then 1F1B has 45 units of bubble, ZB1P has 15 and DualPipe 14. The formula gain looks small. The large gain is the communication that Figure 4 hides, and the bubble formulas do not show it.

DeepSeek open-sourced DualPipe in February 2025. The repository also has DualPipeV, a "cut-in-half" variant credited to Sea AI Lab that needs half the devices.

### Sending tokens across nodes

Overlap only helps if the network part is fast enough to hide. The report's all-to-all design is a direct answer to the H800's links.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §3.2.2 · pages 13 and 14
> [![Page 13: NVLink offers 160 GB/s, about 3.2 times IB at 50 GB/s; each token goes to at most 4 nodes, first over IB to the GPU with the same in-node index, then forwarded via NVLink, highlighted](/img/deepseek/papers/p3-alltoall.png)](/img/deepseek/papers/p3-alltoall.png)
> [![Page 14: the routing could scale to a maximum of 13 experts, and only 20 SMs are enough to use the full IB and NVLink bandwidth, highlighted](/img/deepseek/papers/p3-sms.png)](/img/deepseek/papers/p3-sms.png)
>
> **What it says:** "NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)". So each token goes to "at most 4 nodes". It travels "via IB to the GPUs with the same in-node index on its target nodes", then is "forwarded via NVLink". With this design "only 20 SMs are sufficient to fully utilize the bandwidths of IB and NVLink".
>
> **What it means:**
>
> The rule "at most 4 nodes" is the node-limited routing of [Part 2](deepseek-2-the-v3-architecture.md). It caps the slow traffic per token at 4 sends, whatever experts it picks. The report adds that a token could pick up to 13 experts (4 nodes × 3.2 experts per node) at the same network cost; V3 uses 8.

{{FIG:route|The path of one token. It crosses InfiniBand once per target node, to the GPU with the same number, then NVLink copies it to the GPUs that hold its experts.}}

An SM (streaming multiprocessor) is one of the independent processing units of an NVIDIA GPU. DeepSeek reserves 20 of them to move data and leaves the rest to compute, so the two run at the same time.

The custom kernels behind this became DeepEP, an open-source library for expert-parallel communication, released in February 2025.

## Memory-saving tricks

Without tensor parallelism, each GPU must hold more of the model. So DeepSeek also cut memory in three small ways.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §3.2.3 · page 14
> [![Section 3.2.3 with three techniques: recompute all RMSNorm operations and MLA up-projections during back-propagation, EMA stored in CPU memory and updated asynchronously, and the shallowest and deepest layers on the same PP rank, highlighted](/img/deepseek/papers/p3-memsave.png)](/img/deepseek/papers/p3-memsave.png)
>
> **What it says:** they "recompute all RMSNorm operations and MLA up-projections during back-propagation". The exponential moving average of the weights is "stored in CPU memory and are updated asynchronously". And DualPipe puts "the shallowest layers (including the embedding layer) and deepest layers (including the output head) of the model on the same PP rank".
>
> **What it means:**
>
> The backward pass needs the values that the forward pass computed. Storing all of them costs memory; computing some again costs time. Normalisation and MLA's up-projections are cheap to compute and large to store, so DeepSeek throws them away and recomputes them. It trades a little arithmetic for a lot of memory.
>
> The moving average is a smoothed copy of the weights, used to judge how the model would do after the learning rate decays. It is useful but not urgent, so it lives in ordinary CPU memory and updates in the background. The last trick uses DualPipe's layout. GPU 0 holds the first and the last stage, so the embedding and the output head, shared with the MTP module, sit together.

## FP8: training with 8-bit numbers

This is the part of the report that other labs studied most closely. V3 does most of its matrix products in **FP8**, numbers that take one byte. It was, the report says, the first time FP8 training was validated "on an extremely large-scale model".

### What a floating-point number is

A floating-point number is scientific notation in binary: a sign, an exponent and a mantissa. $$-1.5 \times 2^{3} = -12$$ has sign minus, mantissa 1.5 and exponent 3. More exponent bits give a wider range, from tiny to huge. More mantissa bits give finer steps between neighbouring numbers.

> [!DEFINITION] FP32, BF16 and FP8
> **FP32** uses 32 bits: 8 for the exponent, 23 for the mantissa. **BF16** uses 16 bits: the same 8-bit exponent, so the same range, but only 7 mantissa bits. **FP8** uses 8 bits in two common forms. **E4M3** has 4 exponent and 3 mantissa bits, and its largest number is 448. **E5M2** has 5 and 2: more range, less precision.

{{FIG:formats|The bits of four number formats. Every bit of exponent buys range and every bit of mantissa buys precision. FP8 E4M3 can hold numbers up to 448, with only 8 steps between 1 and 2.}}

Why bother? An H800's Tensor Cores do FP8 products about twice as fast as BF16, and FP8 numbers take half the memory and half the network. The report says FP8 "theoretically doubles the computational speed compared with the original BF16 method".

Our script shows how coarse E4M3 is. These are real PyTorch casts to `torch.float8_e4m3fn`, run on a CPU:

```
A. FP8 E4M3 (4 exponent bits, 3 mantissa bits)
   largest number: 448   smallest normal: 0.015625   smallest above zero: 0.00195312
          1 -> stored as 1
        1.1 -> stored as 1.125
        0.3 -> stored as 0.3125
        300 -> stored as 288
      0.004 -> stored as 0.00390625
     0.0001 -> stored as 0
   between 1 and 2 there are only 8 steps: 1, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2
```

Two kinds of damage show up. Every number gets rounded to a step that is about 6% to 12% of its size: 300 becomes 288. And very small numbers fall off the bottom: 0.0001 becomes 0.

### Why naive FP8 breaks

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · Figure 7 and §3.3.2 · page 16
> [![Figure 7 of the report: (a) fine-grained quantization, with one scaling factor per 1 x NC strip of the input and per NC x NC block of the weight, applied on CUDA cores after the Tensor Core product; (b) accumulation in Tensor Cores for 4 WGMMAs, then promotion to FP32 registers on CUDA cores every NC elements](/img/deepseek/papers/p3-fig7.png)](/img/deepseek/papers/p3-fig7.png)
> [![The fine-grained quantization paragraph on page 16, with highly sensitive to activation outliers, 1x128 tile basis per token per 128 channels and 128x128 block basis highlighted](/img/deepseek/papers/p3-tiles.png)](/img/deepseek/papers/p3-tiles.png)
>
> **What it says:** standard practice scales "the maximum absolute value of the input tensor to the maximum representable value of FP8". This makes training "highly sensitive to activation outliers". DeepSeek instead groups "on a 1x128 tile basis (i.e., per token per 128 channels)" for activations and "on a 128x128 block basis" for weights.
>
> **What it means:**
>
> FP8 has a narrow window of sizes. So before a tensor is cast, it is multiplied by a scale that moves its largest value to 448, the top of the window. One scale for a whole tensor works if the numbers are of similar size. But transformer activations have **outliers**: a few values hundreds or thousands of times larger than the rest. The outlier sets the scale, and the ordinary numbers get pushed down towards the bottom of the window, where steps are coarse or zero.
>
> The fix is to give small groups their own scale. An outlier then damages only its own group of 128 numbers. Everyone else keeps a scale that fits them. In Figure 7 (a), the small coloured squares are those scales, one per strip of the input and one per block of the weight.

Here is the scaling rule as an equation:

$$
s = \frac{448}{\max_{i \in G} |x_i|}, \qquad q_i = \mathrm{fp8}(s \, x_i), \qquad \hat{x}_i = \frac{q_i}{s}
$$

$$G$$ is the group that shares one scale: the whole tensor, a 1 × 128 tile or a 128 × 128 block. $$s$$ is the scale, chosen so that the largest value in the group lands on 448. $$q_i$$ is the stored one-byte number, and $$\hat{x}_i$$ is what comes back when we divide the scale out.

Work it for a group of four numbers, once without and once with an outlier. Our script prints both:

```
A2. Worked example: scale, round, scale back
   group [0.5, -1.2, 2.0, 0.01]: scale = 448 / 2 = 224
      0.01 x scale = 2.24 -> FP8 2.25 -> back 0.010045  (error 0.4%)
   group [0.5, -1.2, 1000.0, 0.01]: scale = 448 / 1000 = 0.448
      0.01 x scale = 0.00448 -> FP8 0.0039062 -> back 0.0087193  (error 12.8%)
```

Same number, 0.01, same format. With a friendly group, it comes back with 0.4% error. With an outlier of 1,000 in its group, it is pushed to 0.00448, below FP8's normal range, where steps are 0.00195 wide. It comes back 12.8% wrong.

{{FIG:scaling|Who shares a scale with an outlier. With one scale per tensor, every number does. With 1 × 128 tiles, only the numbers in the outlier's row-piece do; with 128 × 128 blocks, its whole block does.}}

### Our experiment: one scale against small groups

Now the real test, on a full matrix. Our script builds a 128 × 1,024 activation matrix of random normal numbers. It quantizes it three ways: one scale, 1 × 128 tiles and 128 × 128 blocks. The tile function is short:

```python
def per_tile(x, n=128):
    """One scale per row-piece of n channels (1 x n tiles): per token, per 128 channels."""
    r, c = x.shape
    t = x.reshape(r, c // n, n)
    scale = FMAX / t.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)
    return to_fp8_and_back(t, scale).reshape(r, c)
```

The reshape cuts each row into pieces of 128. `amax` finds the largest absolute value in each piece. Each piece gets its own scale, then the shared function clamps, casts to FP8 and divides the scale back out.

First, one outlier of growing size, put at row 5, column 300. We measure the error of all the other numbers:

```
B. Activation matrix 128 x 1024, random normal (typical |x| about 0.8), one outlier at row 5, column 300
    outlier |      error of the ordinary numbers       |   ordinary numbers turned into 0  
            |       tensor    1x128 tile   128x128 blk |     tensor       tile      block
          0 |       2.65%         2.57%         2.63% |      0.00%      0.00%      0.00%
        100 |       2.67%         2.57%         2.63% |      0.01%      0.00%      0.00%
      1,000 |       2.65%         2.57%         2.63% |      0.19%      0.00%      0.03%
     10,000 |       2.72%         2.57%         2.65% |      1.78%      0.00%      0.24%
    100,000 |      12.56%         2.60%         5.07% |     17.28%      0.02%      2.14%
   for comparison: BF16 (7 stored mantissa bits) gives 0.17% on the same numbers
```

The result surprised us a little, and it is worth reading slowly. E4M3 has a floating exponent, so moderate outliers do almost no harm even with one scale: up to 10,000, the error stays near 2.7%. Only when the outlier is 100,000 times the typical value does one scale break: 12.56% error, and 17% of the numbers become zero. Tiles stay at 2.6% throughout. Blocks are in between, because the outlier's block is 128 rows tall.

Second, a harder and more realistic case. In gradients, whole rows (tokens) can differ in size by many orders of magnitude. Appendix B.2 of the report calls these "token-correlated outliers". We multiply each row by $$10^{u}$$, with $$u$$ random between $$-S$$ and 0:

```
C. Rows (tokens) of very different sizes: row i is multiplied by 10**u, u uniform in [-S, 0]
   (activation gradients look like this: some tokens carry far larger values than others)
   error = mean over rows of each row's relative error, so small rows count as much as large ones
     S |              error of G              |            error of G W^T           
       |     tensor         tile        block |     tensor         tile        block
     0 |      2.64%        2.56%        2.66% |      3.74%        3.67%        3.73%
     2 |      2.64%        2.58%        2.64% |      3.74%        3.70%        3.72%
     4 |      2.75%        2.57%        2.68% |      3.82%        3.70%        3.78%
     6 |     17.52%        2.57%       16.31% |     18.30%        3.68%       17.14%
     8 |     31.53%        2.56%       30.85% |     32.20%        3.69%       31.53%
```

{{FIG:fp8err|Our FP8 test on random data. A: one outlier in the matrix; one scale fails only at an outlier of 100,000. B: rows of very different sizes; one scale and 128 × 128 blocks fail once rows differ by a million times, while 1 × 128 tiles stay at 2.6%.}}

Once rows differ by a million times ($$S = 6$$), both one scale and 128 × 128 blocks lose the small rows: about 17% error, rising to 31%. The 1 × 128 tiles do not care, because every token gets its own scale. The right half of the table shows the damage survives a matrix product with a weight matrix. This is why V3 uses tiles for activations and blocks only for weights, which are smooth.

The report learned the same lesson the hard way, in an appendix:

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · Appendix B.2 · pages 47 and 48
> [![Appendix B.2 on page 47: an experiment quantizing all Dgrad tensors block-wise, with block-wise quantization of activation gradients leads to highlighted](/img/deepseek/papers/p3-blockwise.png)](/img/deepseek/papers/p3-blockwise.png)
> [![Page 48: model divergence on an MoE model comprising approximately 16B total parameters, and token-correlated outliers, highlighted](/img/deepseek/papers/p3-blockwise2.png)](/img/deepseek/papers/p3-blockwise2.png)
>
> **What it says:** quantizing activation gradients with 128 × 128 blocks, like weights, "leads to model divergence on an MoE model comprising approximately 16B total parameters", trained for about 300B tokens. The authors guess the cause: gradients are imbalanced among tokens, "resulting in token-correlated outliers".
>
> **What it means:**
>
> Blocks would have been simpler. A block can be transposed for the backward pass without re-scaling, while a 1 × 128 tile must become a 128 × 1 tile. DeepSeek tried the simple way, and a 16B model diverged: its loss blew up and never came back. Our part C shows the mechanism on random numbers. One huge token row sets the scale for 127 quiet rows, and the quiet rows drop to zero.

### The second problem: adding up

A matrix product is a long sum. Each output is a dot product: multiply pairs, then add up thousands of products. Products of FP8 numbers are fine. The running sum needs more care.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §3.3.2 · page 17
> [![The Increasing Accumulation Precision paragraph, with limited to retaining around 14 bits, maximum relative error of nearly 2% and copied to FP32 registers on CUDA Cores highlighted](/img/deepseek/papers/p3-accum.png)](/img/deepseek/papers/p3-accum.png)
> [![The end of the paragraph: setting NC = 128 elements, equivalent to 4 WGMMAs, is the minimal interval that improves precision without substantial overhead; and E4M3 on all tensors, highlighted](/img/deepseek/papers/p3-nc128.png)](/img/deepseek/papers/p3-nc128.png)
>
> **What it says:** the FP8 matrix units of the H800 add up with a precision "limited to retaining around 14 bits", far below FP32. For two random matrices with inner size 4,096, this gives "a maximum relative error of nearly 2%". The fix: every so often, the partial sums are "copied to FP32 registers on CUDA Cores". "Setting $$N_C$$ = 128 elements, equivalent to 4 WGMMAs" is the smallest interval that helps without much slowdown. The page also says V3 uses E4M3 "on all tensors".
>
> **What it means:**
>
> Picture adding up a long shopping list on a calculator that shows only four digits. The first items are fine. But once the total is in the thousands, small items like 0.35 vanish in the rounding, and the error grows with every line. The fix is to add 128 lines on the small calculator, write the subtotal on paper in full, clear the calculator, and continue.
>
> The cost is that Tensor Cores must stop and hand partial sums to slower CUDA Cores. DeepSeek hides that too: on the H800, two groups of threads take turns, so one copies while the other multiplies. The per-group scales from the last section are divided out in the same step, almost for free.

We cannot run the H800's Tensor Cores on a laptop, so we built a model of the idea. Our accumulator keeps only 14 significant bits and cuts off the rest after every addition. We add 4,096 products of FP8-rounded random numbers, for 4,096 dot products, two ways:

```python
for k in range(K):
    low = keep_bits(low + prod[:, k])                 # (1) everything in the narrow accumulator
    part = keep_bits(part + prod[:, k])               # (2) narrow accumulator for 128 terms...
    if (k + 1) % 128 == 0:
        total = (total + part).float().double()       # ...then moved into an FP32 total
        part.zero_()
```

```
D. SIMULATION: 4,096 dot products of length K = 4,096 (FP8-rounded random inputs)
   narrow accumulator for all 4,096 terms   max error 0.4638%   mean error 0.0643%
   narrow for 128 terms, then FP32          max error 0.0045%   mean error 0.0009%
   (error = |result - exact sum| / sum of |terms|, so it does not blow up when the sum is near zero)
```

{{FIG:accum|Our simulation of a 14-bit accumulator. Adding all 4,096 products in it gives a worst-case error of 0.46%; moving the subtotal to FP32 every 128 products cuts it about 100 times. The real H800 measured nearly 2%, so this is a model of the effect, not of the chip.}}

Promotion every 128 terms cuts the worst error about 100 times in our model. Our worst case, 0.46%, is smaller than the report's "nearly 2%". The report does not describe the real circuit, and it does not behave like our simple cut-off. Only the direction of the result carries over.

### What stays in higher precision

FP8 is not used everywhere. Figure 6 shows the plan for one linear layer.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · Figure 6 and §3.3.1 · page 15
> [![Figure 6: the three GEMMs of a linear layer, Fprop, Dgrad and Wgrad, each taking FP8 inputs and accumulating in FP32, with BF16 inputs and outputs and FP32 master weights, weight gradients and optimizer states](/img/deepseek/papers/p3-fig6.png)](/img/deepseek/papers/p3-fig6.png)
> [![Page 15: original precision kept for the embedding module, the output head, MoE gating modules, normalization operators and attention operators, highlighted](/img/deepseek/papers/p3-fp8-keep.png)](/img/deepseek/papers/p3-fp8-keep.png)
>
> **What it says:** all three GEMMs of a linear layer run in FP8: Fprop (forward), Dgrad (gradient for the input) and Wgrad (gradient for the weights). Each one sums in FP32. The original precision, BF16 or FP32, is kept for "the embedding module, the output head, MoE gating modules, normalization operators, and attention operators".
>
> **What it means:**
>
> The rule is: FP8 where the arithmetic is, high precision where the risk is. The linear layers hold almost all the FLOPs, so they go to FP8. The parts that are cheap or touchy stay wide. The gate picks experts, and a small rounding error there would send a token to the wrong expert. Normalisation divides by a computed size. The output head turns numbers into probabilities over 128K tokens.
>
> Note the converters in the figure, marked "To FP8" and "To BF16". Every arrow is a cast with its own fine-grained scale. The FP8 copies exist only for the product, and the master copy of every weight stays in FP32.

The rest of the precision plan, from §3.3.3 (page 18), fits in a table:

| Item | Format in V3 | Why |
|---|---|---|
| Linear layer products (Fprop, Dgrad, Wgrad) | FP8 E4M3 inputs, FP32 sums | Most of the arithmetic |
| Master weights, weight gradients | FP32 | Small updates must not round away |
| AdamW first and second moments | BF16 | Report: no "observable performance degradation" |
| Activations saved for backward | FP8, tile-scaled | Memory |
| Input of the linear after attention | Custom E5M6 | Attention's backward needs it; precision-sensitive |
| SwiGLU input in MoE | FP8, output recomputed | Memory |
| Token dispatch to experts | FP8 | Halves network traffic |
| Combine back from experts | BF16 | Report: kept to "preserve training precision" |
| Embedding, output head, gate, norms, attention | BF16 or FP32 | Sensitive or cheap |

Did it work? The report checked FP8 against BF16 before trusting it.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §3.3 · page 15
> [![The FP8 introduction on page 15, with tile-wise grouping with 1 x Nc elements, block-wise grouping with Nc x Nc, and relative loss error remains consistently below 0.25% highlighted](/img/deepseek/papers/p3-fp8-025.png)](/img/deepseek/papers/p3-fp8-025.png)
>
> **What it says:** on two smaller models similar to V2-Lite and V2, each trained for about 1T tokens, "the relative loss error of our FP8-training model remains consistently below 0.25%" compared with BF16, "a level well within the acceptable range of training randomness".
>
> **What it means:**
>
> Before betting a two-month run on FP8, DeepSeek ran the same models twice, once in each format, and compared the loss curves. Appendix B.1 gives the sizes: about 16B parameters on 1.33T tokens, and about 230B on 0.9T tokens. A 0.25% gap in loss is about the gap between two BF16 runs with different random seeds.
>
> This is a test at a smaller scale. The 671B run itself has no BF16 twin. The evidence for full size is the model's results and the stability claim below.

The FP8 kernels became DeepGEMM, released in February 2025. V3.1 later switched its scale format to UE8M0, a power-of-two scale that suits the newer "microscaling" hardware formats; its model card says so. The report itself says its tile-and-block design is "highly consistent with the idea of microscaling formats" that NVIDIA's Blackwell GPUs support in hardware.

> [!TIP] In production
> The main training libraries behind V3 are public, from DeepSeek's Open-Source Week in February 2025. They are **DualPipe** (the schedule), **DeepEP** (expert-parallel all-to-all), **DeepGEMM** (FP8 GEMMs) and **3FS** (the file system). The released V3 checkpoint also ships in FP8 E4M3 with 128 × 128 block scales, the same block size that training used for weights.

## No loss spikes, no rollbacks

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §1 · page 4
> [![Page 4: during pre-training on 14.8T high-quality and diverse tokens, we did not encounter any irrecoverable loss spikes or have to roll back, highlighted](/img/deepseek/papers/p3-stable.png)](/img/deepseek/papers/p3-stable.png)
>
> **What it says:** "The pre-training process is remarkably stable. Throughout the entire training process, we did not encounter any irrecoverable loss spikes or have to roll back."
>
> **What it means:**
>
> In a long run, the loss sometimes jumps up suddenly. Usually it recovers. Sometimes it does not, and the team must reload a checkpoint from hours or days before, change something and try again. That is a **rollback**, and each one burns GPU time that a cost table may or may not show.
>
> The word "irrecoverable" matters. The sentence does not say the loss never jumped. It says no jump needed a rollback. The report shows no loss curve for the 14.8T-token run, so this is a statement we cannot check.

The claim matters because FP8 was new at this scale, and outliers are what cause spikes. A stable run also supports the cost table: 2,664K hours for one clean pass.

V4 makes a useful contrast. Its report is open about trouble:

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §4.2.3 · page 26
> [![Section 4.2.3 of the V4 report: we encountered notable instability challenges during training; simple rollbacks could temporarily restore the training state; the occurrence of spikes is consistently tied to outliers in the MoE layers, highlighted](/img/deepseek/papers/p3-v4-instability.png)](/img/deepseek/papers/p3-v4-instability.png)
>
> **What it says:** "We encountered notable instability challenges during training." Rollbacks "could temporarily restore the training state" but did not stop spikes from coming back. The spikes are "consistently tied to outliers in the MoE layers".
>
> **What it means:**
>
> V4 is bigger: 1.6T parameters for V4-Pro. It also trains with a new optimizer and new attention. Spikes came back, and DeepSeek fixed them with two tricks, described in the V4 report. One computes the routing from slightly older weights only when a spike starts; the other clamps SwiGLU values. The lesson for V3: its stability was real, but not automatic. The next, larger model did not get it for free.

## The cost: $5.576M, and what it leaves out

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · Table 1 and §1 · page 5
> [![Table 1: training costs in H800 GPU hours, pre-training 2664K, context extension 119K, post-training 5K, total 2788K, and in USD 5.328M, 0.238M, 0.01M and 5.576M, assuming 2 dollars per GPU hour](/img/deepseek/papers/p3-cost.png)](/img/deepseek/papers/p3-cost.png)
> [![The cost paragraph on page 5, with 180K H800 GPU hours, 3.7 days on our cluster with 2048 H800 GPUs, and excluding the costs associated with prior research and ablation experiments highlighted](/img/deepseek/papers/p3-cost-text.png)](/img/deepseek/papers/p3-cost-text.png)
>
> **What it says:** each trillion tokens takes "180K H800 GPU hours, i.e., 3.7 days on our cluster with 2048 H800 GPUs". Pre-training costs 2,664K GPU hours, context extension 119K and post-training 5K, so 2.788M in total. At "$2 per GPU hour", that is $5.576M. The costs "include only the official training of DeepSeek-V3, excluding the costs associated with prior research and ablation experiments on architectures, algorithms, or data".
>
> **What it means:**
>
> The table is GPU hours, priced as if rented. DeepSeek owns its GPUs, so no one paid $5.576M to a cloud. The number answers one narrow question: what would the final run cost on rented H800s at $2 per hour?
>
> The last sentence is the important one, and many headlines dropped it. Ablations are the many smaller test runs that decide the design: MLA or not, how many experts, FP8 or BF16. The FP8 test runs above, on 16B and 230B models, are examples. None of that is in the $5.576M.

The numbers are consistent, and the check is a good habit:

```
4. The cost paragraph (V3 p.5), checked
   180K GPU hours per trillion tokens x 14.8 = 2,664K GPU hours (paper: 2,664K)
   180K GPU hours / 2,048 GPUs / 24 = 3.66 days per trillion tokens (paper: 3.7)
   pre-training        2,664K GPU hours =  54.2 days on 2,048 GPUs = $5.328M at $2/hour
   context extension     119K GPU hours =   2.4 days on 2,048 GPUs = $0.238M at $2/hour
   post-training           5K GPU hours =   0.1 days on 2,048 GPUs = $0.010M at $2/hour
   total               2,788K GPU hours =  56.7 days on 2,048 GPUs = $5.576M at $2/hour
```

Pre-training is 54 days on the full cluster, which matches "less than two months". The whole run is about eight weeks of wall-clock time on one large cluster.

{{FIG:cost|What the $5.576M counts and what it does not. Only the GPU hours of the final run, at an assumed rental price, are inside. The paper names research and ablations as excluded; hardware, staff, power and data work are outside any rental-price total by definition.}}

Two exclusions are less obvious. First, V3's reasoning training data came from "an internal DeepSeek-R1 model" (report, pages 28 and 29), and that model's cost is not in the 5K post-training hours. Second, the R1 paper later gave its own cost: $294K for R1-Zero and R1 on top of the V3 base. [Part 4](deepseek-4-r1-and-reasoning-by-rl.md) covers that number and what it excludes.

> [!WARNING] Third-party estimates are not DeepSeek's numbers
> The analysts at SemiAnalysis (31 January 2025) estimated that DeepSeek has access to about 50,000 Hopper GPUs and server spending of about $1.6B. These are **estimates by an outside firm**, not confirmed by DeepSeek. The hardware is also shared with other work, so the estimates are not a training cost either. Anthropic's CEO Dario Amodei, who runs a competitor, wrote in January 2025 that V3 was "an expected point on an ongoing cost reduction curve". Both are opinions about the context. Neither changes what the table measures.

## A sanity check: does the compute add up?

Can we check 2,664K GPU hours from first principles? There is a well-known rule of thumb for the arithmetic of training.

**The intuition.** Every parameter that a token touches does one multiply and one add in the forward pass: 2 floating-point operations (FLOPs). The backward pass does about twice the forward work: one product for the gradient of the input, one for the gradient of the weight. So each parameter costs about 6 FLOPs per token.

$$
C \approx 6 \, N \, D
$$

$$C$$ is the total compute in FLOPs. $$N$$ is the number of parameters each token uses: for an MoE model, the **active** parameters, not the total. $$D$$ is the number of training tokens. The rule ignores attention's own arithmetic, which grows with sequence length, and small extras.

A tiny example: a 1B-parameter dense model on 1B tokens needs about $$6 \times 10^9 \times 10^9 = 6 \times 10^{18}$$ FLOPs. For V3, with 37B active parameters and 14.8T tokens:

$$
C \approx 6 \times 37 \times 10^{9} \times 14.8 \times 10^{12} \approx 3.3 \times 10^{24} \text{ FLOPs}
$$

The ISCA paper gives a more careful number: about 250 GFLOPs per token at sequence length 4,096, which includes attention. That gives $$3.7 \times 10^{24}$$ FLOPs.

> [!PAPER] Zhao et al. (2025), Insights into DeepSeek-V3 · Table 2 · page 4
> [![Table 2 of the ISCA paper: training cost per token at sequence length 4096, DeepSeek-V2 MoE 155 GFLOPS per token, DeepSeek-V3 MoE 250, Qwen-72B dense 394, LLaMA-405B dense 2448, with V3 highlighted](/img/deepseek/papers/p3-isca-flops.png)](/img/deepseek/papers/p3-isca-flops.png)
>
> **What it says:** at "sequence length of 4096", training costs "250 GFLOPS/Token" for V3 with 671B parameters, against 394 for the dense Qwen-72B and 2,448 for the dense LLaMA-405B.
>
> **What it means:**
>
> 6 × 37B is 222 GFLOPs, so the extra 28 is roughly attention and the other parts our rule skips. The comparison is the point of the table. V3 has nine times the parameters of Qwen-72B, yet each token costs less to train. That is the MoE bargain from [Part 2](deepseek-2-the-v3-architecture.md), measured in FLOPs.

Now divide by the time. 2,664K GPU hours is $$9.59 \times 10^{9}$$ GPU-seconds:

```
5. Compute estimate (pre-training only)
   6 x N x D = 6 x 37e9 x 14.8e12 = 3.286e+24 FLOPs
   ISCA paper, 250 GFLOPs per token x 14.8e12 = 3.700e+24 FLOPs
   GPU-seconds: 2,664,000 x 3,600 = 9.5904e+09
   6ND : 343 TFLOPS per GPU on average = 35% of dense BF16 peak (990), 17% of dense FP8 peak (1979)
   ISCA: 386 TFLOPS per GPU on average = 39% of dense BF16 peak (990), 19% of dense FP8 peak (1979)
```

So each GPU did about 340 to 390 trillion useful operations per second, averaged over the run. The peaks come from NVIDIA's H100 SXM page, which lists 1,979 BF16 and 3,958 FP8 TFLOPS "with sparsity"; dense is half. We assume the H800 has the same Tensor Cores, as the ISCA paper says it is "similar to the H100" apart from FP64 and NVLink.

Is 19% of the FP8 peak low? Not for this job. Attention, normalisation, the gate and the output head run in BF16 or FP32. Scaling and promotion take time. Twenty of each GPU's SMs drive the network. And real models never reach the peak.

The useful conclusion is the reverse one: the claimed GPU hours are physically reasonable. They need no hidden extra compute, and they are not so small that they need a miracle.

> [!WARNING] This check has error bars
> 6ND is an approximation, and the ISCA figure is DeepSeek's own estimate. The check says the cost table is plausible; it cannot prove it.

## How V3.1, V3.2 and V4 were trained differently

DeepSeek did not train every later model from scratch. The table collects what each source states; where a cell is empty, the source does not say.

| Model | Starting point | New training tokens | Context | Precision | Hardware and cost |
|---|---|---|---|---|---|
| V3 (Dec 2024) | Random weights | 14.8T at 4K, plus 2 × 62.9B for long context | 128K | FP8, 1 × 128 and 128 × 128 scales | 2,048 H800; 2.788M GPU hours |
| V3.1 (Aug 2025) | The V3 base checkpoint | 630B at 32K and 209B at 128K | 128K | FP8 with UE8M0 scales | Not stated |
| V3.2 (Dec 2025) | V3.1-Terminus | 2.1B indexer warm-up, then 943.7B with sparse attention | 128K | Not restated | Not stated |
| V4-Flash / V4-Pro (Apr 2026) | Random weights | 32T / 33T | 4K growing to 1M during pre-training | Routed experts in FP4 | Not stated |
| V4.1-Flash (Sep 2026) | Random weights | 45T multimodal tokens | 64K, then 1M | FP4 experts, FP8 elsewhere (card) | Not stated |

Sources: V3.1 model card; V3.2 report (arXiv 2512.02556, page 4); V4 report (arXiv 2606.19348, pages 5 and 25); V4.1-Flash report (arXiv 2609.19969, page 22).

Three changes stand out. First, V3.1 and V3.2 continue training V3's weights; they are not new pre-training runs. V3.2 added sparse attention ([Part 5](deepseek-5-sparse-attention-and-v3-2.md)) in about 946B more tokens.

Second, V4 is a new run, and the recipe changed a lot. It trains on twice the data and uses the Muon optimizer for most weights instead of AdamW alone. It grows the sequence length to 1M tokens during pre-training. Its routed experts use FP4, 4-bit numbers. The report puts FP4 training in the post-training stage, as quantization-aware training, which can "fully reuse the existing FP8 training framework". V4.1-Flash keeps a fixed batch of 100.6 million tokens and reports training "with no instability".

Third, from V3.1 on, DeepSeek stopped publishing GPU hours. V3 remains the only DeepSeek model with a full public cost table. Any cost you read for V4 is somebody's estimate.

> [!TAKEAWAYS]
> - V3 was pre-trained on **14.8T tokens** at 4K tokens per sequence. YaRN then stretched it to **128K** in two 1,000-step phases of about 62.9B tokens each.
> - The learning rate held at **2.2 × 10⁻⁴** for 10T tokens, fell along a cosine to 2.2 × 10⁻⁵, and ended at 7.3 × 10⁻⁶. The batch grew from 3,072 to **15,360** sequences (about 63M tokens).
> - The cluster had **2,048 H800 GPUs**: H100-class chips whose NVLink was cut from 900 to **400 GB/s**. Usable speeds were about 160 GB/s inside a node and 50 GB/s per InfiniBand card.
> - The plan used **16-way pipeline**, **64-way expert** and **ZeRO-1 data** parallelism, and **no tensor parallelism**, which the weak NVLink would have made slow.
> - **DualPipe** pairs a forward and a backward chunk, so the 1:1 network traffic hides under compute. It feeds micro-batches from both ends, at the cost of two parameter copies.
> - Most GEMMs ran in **FP8 E4M3** with one scale per **1 × 128** activation tile and **128 × 128** weight block. Partial sums moved to **FP32 every 128** products. In our CPU test, tiles kept 2.6% error where one scale reached 31%.
> - The report claims **no irrecoverable loss spikes and no rollbacks**. V4's report, by contrast, describes spikes and rollbacks.
> - The cost is **2.788M H800 GPU hours**, or **$5.576M** at an assumed $2 per hour. It excludes research, ablations, hardware, staff, data work and the R1 model that made V3's reasoning data.
> - 6ND gives about **3.3 × 10²⁴** FLOPs, about 340 TFLOPS per GPU over the run: plausible for this hardware.

**Next.** V3 as described here is a base model with 5K GPU hours of post-training. The model that made DeepSeek famous a month later was R1, built on this base with reinforcement learning. [Part 4](deepseek-4-r1-and-reasoning-by-rl.md) explains GRPO, R1-Zero, the four-stage R1 pipeline, the $294K figure and how later models changed post-training.

## References

- DeepSeek-AI (2024). DeepSeek-V3 Technical Report. [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- Zhao et al. (2025). Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures (ISCA 2025). [arXiv 2505.09343](https://arxiv.org/abs/2505.09343)
- An et al. (2024). Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning (SC24). [arXiv 2408.14158](https://arxiv.org/abs/2408.14158)
- DeepSeek-AI (2025). DeepSeek-V3.2. [arXiv 2512.02556](https://arxiv.org/abs/2512.02556)
- DeepSeek-AI (2026). DeepSeek-V4. [arXiv 2606.19348](https://arxiv.org/abs/2606.19348)
- DeepSeek-AI (2026). DeepSeek-V4.1-Flash. [arXiv 2609.19969](https://arxiv.org/abs/2609.19969)
- Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models. [arXiv 2309.00071](https://arxiv.org/abs/2309.00071)
- Qi et al. (2023). Zero Bubble Pipeline Parallelism. [arXiv 2401.10241](https://arxiv.org/abs/2401.10241)
- DeepSeek-V3.1 model card: [huggingface.co/deepseek-ai/DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)
- Code: [DualPipe](https://github.com/deepseek-ai/DualPipe), [DeepEP](https://github.com/deepseek-ai/DeepEP), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [3FS](https://github.com/deepseek-ai/3FS), [open-infra-index](https://github.com/deepseek-ai/open-infra-index)
- NVIDIA H100 specifications: [nvidia.com/en-us/data-center/h100](https://www.nvidia.com/en-us/data-center/h100/)
- Third-party views: [SemiAnalysis, DeepSeek Debates](https://newsletter.semianalysis.com/p/deepseek-debates); [Dario Amodei, On DeepSeek and Export Controls](https://www.darioamodei.com/post/on-deepseek-and-export-controls)
- Our scripts: `code/deepseek/part3_fp8.py`, `part3_numbers.py`, `figs_part3.py`, `shots_part3.py`
