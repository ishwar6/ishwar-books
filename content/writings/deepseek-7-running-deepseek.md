---
title: "Running DeepSeek on Your Own Hardware, and How DeepSeek Serves It"
description: "How much memory each DeepSeek model needs, which GPUs, Macs and CPU servers run it at what reported speed, and how DeepSeek serves it at scale."
date: 2026-11-10
tags: [deepseek, inference, hardware, gpu, quantization, moe, serving]
series: "DeepSeek, From the Ground Up"
series_part: 7
motif: grid
accent: "#4fb3a9"
---

On one day in February 2025, DeepSeek served its chat models on an average of 226.75 server nodes, each with eight H800 GPUs. In that one day the fleet read 608 billion input tokens and wrote 168 billion output tokens. A few weeks later, a reviewer ran the same 671-billion-parameter model on one Mac Studio. It wrote about 17 to 18 tokens per second.

Both facts are true, and both come down to one question: where do the bytes go? A model is a very large file of numbers. Before it can answer, those numbers must sit in fast memory, and a few of them must be read for every word it writes. This part does that arithmetic for every major DeepSeek model, from the 1.5B distill on a laptop to the 1.6-trillion-parameter V4-Pro. Then it shows the real recipes people use, and how DeepSeek itself runs the service.

> [!TIP] The whole idea in one minute
> Memory needed is about **parameters x bytes per weight**, plus the **KV cache** (the notes the model keeps about the text so far). DeepSeek's big models are **mixtures of experts**: all experts must be stored, but each token reads only a few of them. So total size decides **if a model fits**, and active size decides **how fast it writes**. The 671B V3 is a 688.6 GB file in its native format, 404 GB at 4 bits, and 140 GB in the smallest community version. It needs a node of 8 H200 GPUs, or similar, to serve properly. But it also runs, slowly, on a big Mac or a CPU server with one consumer GPU.
>
> DeepSeek serves it on hundreds of GPUs with separate machines for reading prompts and for writing answers.

[Part 6](deepseek-6-v4-and-v4-1.md) ended with the V4 and V4.1 architectures and their tiny KV caches. Here those design choices meet real machines.

## Memory arithmetic

### Bytes per weight

The weights must sit somewhere the processor can reach quickly, so the first number is their size:

$$
M_{\text{weights}} = N \times \frac{b}{8}
$$

Here $$M_{\text{weights}}$$ is the memory in bytes, $$N$$ is the number of weights, and $$b$$ is the number of bits per weight. The division by 8 turns bits into bytes.

Now the real model. The script `part7_fits.py` starts with the rule and then reads the real file sizes:

```python
for fmt, bits in [('BF16', 16), ('FP8', 8), ('INT4 / FP4', 4)]:
    log(f'  671B in {fmt:10s}: 671e9 x {bits:2d} / 8 = {671e9 * bits / 8 / GB:7,.1f} GB')
...
for m, f, gb, n, src in FILES:
    bits = gb * GB * 8 / n          # effective bits per weight of the real file
```

```text
== 1. Bytes per weight: the rule ==
memory = parameters x bits per weight / 8
  671B in BF16      : 671e9 x 16 / 8 = 1,342.0 GB
  671B in FP8       : 671e9 x  8 / 8 =   671.0 GB
  671B in INT4 / FP4: 671e9 x  4 / 8 =   335.5 GB

== Real files: size and effective bits per weight ==
model                    format                  GB      GiB bits/weight
V3 / R1 (671B)           BF16 GGUF          1,342.3  1,250.1       16.00
V3 / R1 (671B)           GGUF Q8_0            713.3    664.3        8.50
V3 / R1 (671B)           official FP8         688.6    641.3        8.05
V3 / R1 (671B)           GGUF Q4_K_M          404.4    376.6        4.82
V3 / R1 (671B)           UD-Q2_K_XL           226.6    211.0        2.70
V3 / R1 (671B)           UD-IQ1_S             140.2    130.6        1.67
V3.2 (671B)              official FP8         689.5    642.1        8.06
V3.2 (671B)              UD-IQ1_S             184.1    171.5        2.19
V2 (236B)                official BF16        471.5    439.1       16.00
V2-Lite (16B)            official BF16         31.4     29.2       16.00
V4-Flash (284B)          official FP4+FP8     159.6    148.6        4.50
V4-Flash (284B)          UD-Q2_K_XL            96.8     90.2        2.73
V4-Pro (1.6T)            official FP4+FP8     864.7    805.3        4.32
V4.1-Flash (552B+196B)   official FP4+FP8     510.3    475.3        5.46
R1-Distill-Llama-70B     official BF16        141.1    131.4       16.00
R1-Distill-Llama-70B     GGUF Q4_K_M           42.5     39.6        4.82
```

The sizes are the sums of the real file listings on Hugging Face, checked on 10 November 2026. Three things in this table are worth a second look.

First, the official V3 file is 688.6 GB, not 671 GB. The repository holds 684.5 billion weights: the 671B main model plus a 14B module for **multi-token prediction** (MTP, explained below). About 3.9 billion of those weights are small tensors kept in BF16, and every 128 x 128 block of FP8 weights carries a scale. That is why the official file averages 8.05 bits, not exactly 8.

Second, a "4-bit" file is not 4.00 bits. The common GGUF format Q4_K_M averages 4.82 bits on V3, because some tensors keep more bits and every block stores its scales.

Third, the V4 family already ships its experts in FP4. The official V4-Pro file averages 4.32 bits per weight. So the usual trick of "download a 4-bit version" barely helps for V4: the official file is already about that small.

<figure class="fig"><svg viewBox="0 0 760 306" role="img" aria-label="File sizes of the same 671B model in six formats, with three memory sizes marked."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">One 671B model, six file sizes (DeepSeek-V3 / R1)</text><line class="edge-dim" x1="393.4" y1="36" x2="393.4" y2="248"/><text class="t-muted" x="393.4" y="264.0" text-anchor="middle">512 GB Mac</text><line class="edge-dim" x1="429.2" y1="36" x2="429.2" y2="260"/><text class="t-muted" x="429.2" y="278.0" text-anchor="middle">8x H100</text><line class="edge-dim" x1="565.8" y1="36" x2="565.8" y2="248"/><text class="t-muted" x="565.8" y="264.0" text-anchor="middle">8x H200</text><text class="t-tick" x="240.0" y="61.0" text-anchor="end">BF16 GGUF (16.00 bits)</text><g class="mark"><title>BF16 GGUF (16.00 bits): 1,342.3 GB</title><rect class="s2" x="250" y="44" width="375.8" height="24" rx="4"/></g><text class="t-val" x="633.8" y="61.0" text-anchor="start">1,342.3 GB</text><text class="t-tick" x="240.0" y="95.0" text-anchor="end">GGUF Q8_0 (8.50 bits)</text><g class="mark"><title>GGUF Q8_0 (8.50 bits): 713.3 GB</title><rect class="s2" x="250" y="78" width="199.7" height="24" rx="4"/></g><text class="t-val" x="457.7" y="95.0" text-anchor="start">713.3 GB</text><text class="t-tick" x="240.0" y="129.0" text-anchor="end">official FP8 (8.05 bits)</text><g class="mark"><title>official FP8 (8.05 bits): 688.6 GB</title><rect class="s1" x="250" y="112" width="192.8" height="24" rx="4"/></g><text class="t-val" x="450.8" y="129.0" text-anchor="start">688.6 GB</text><text class="t-tick" x="240.0" y="163.0" text-anchor="end">GGUF Q4_K_M (4.82 bits)</text><g class="mark"><title>GGUF Q4_K_M (4.82 bits): 404.4 GB</title><rect class="s2" x="250" y="146" width="113.2" height="24" rx="4"/></g><text class="t-val" x="371.2" y="163.0" text-anchor="start">404.4 GB</text><text class="t-tick" x="240.0" y="197.0" text-anchor="end">UD-Q2_K_XL (2.70 bits)</text><g class="mark"><title>UD-Q2_K_XL (2.70 bits): 226.6 GB</title><rect class="s2" x="250" y="180" width="63.4" height="24" rx="4"/></g><text class="t-val" x="321.4" y="197.0" text-anchor="start">226.6 GB</text><text class="t-tick" x="240.0" y="231.0" text-anchor="end">UD-IQ1_S (1.67 bits)</text><g class="mark"><title>UD-IQ1_S (1.67 bits): 140.2 GB</title><rect class="s2" x="250" y="214" width="39.3" height="24" rx="4"/></g><text class="t-val" x="297.3" y="231.0" text-anchor="start">140.2 GB</text><line class="axis" x1="250" y1="243.0" x2="670" y2="243.0"/><text class="t-muted" x="250.0" y="296.0" text-anchor="start">GB on disk (decimal). Dashed lines: memory of common machines.</text></svg><figcaption>The same 671B DeepSeek-V3/R1 model as six files. The three smallest files fit a 512 GB Mac; the native FP8 file needs more memory than eight H100 GPUs have.</figcaption></figure>

> [!DEFINITION] GGUF
> A single-file model format used by llama.cpp, Ollama and similar tools that run models on ordinary computers. A GGUF file holds the weights in one of many quantized types (Q8_0, Q4_K_M, IQ1_S and so on) plus the tokenizer and settings.

### The GB versus GiB trap

In January 2025, the company Unsloth published a "1.58-bit" version of R1 and said it was 131 GB. The folder on Hugging Face adds up to 140.2 GB. Both numbers are right, because they use different units.

> [!DEFINITION] GB and GiB
> A **gigabyte (GB)** is $$10^9$$ bytes. A **gibibyte (GiB)** is $$2^{30}$$ bytes, which is 1,073,741,824 bytes, about 7.4% more. Disk makers and Hugging Face use GB. Many tools and operating systems print GiB but label it "GB". Memory chips come in powers of two, so a "512 GB" Mac really has 512 GiB.

```text
== 2. The GB versus GiB trap ==
R1 UD-IQ1_S ("131 GB" in the blog): 140.2 GB = 140,200,000,000 bytes = 130.6 GiB
R1 UD-Q2_K_XL ("212 GB" in the blog): 226.6 GB = 226,600,000,000 bytes = 211.0 GiB
average bits per weight of the "1.58-bit" file: 140.2e9 x 8 / 671e9 = 1.67
```

So the "131 GB" file is 131 GiB. On a machine where every gigabyte counts, a 7% mistake is the difference between "fits" and "does not fit". The output also shows a second trap: the "1.58-bit" label describes most expert weights, not the file. The file averages 1.67 bits, because Unsloth kept the first dense layers and other sensitive tensors at higher precision.

> [!DEFINITION] Dynamic quantization (Unsloth "UD" files)
> A quantized file where different tensors get different bit widths. The many routed-expert weights get the fewest bits; attention, the first dense layers and other sensitive parts get more. Unsloth names these files with a "UD-" prefix, such as UD-IQ1_S or UD-Q2_K_XL.

> [!WARNING]
> Do not reuse "131 GB" for newer models. The UD-IQ1_S files for R1-0528, V3.1 and V3.2 are 185.5, 192.3 and 184.1 GB. Always add up the real files of the exact model you want.

### What all those bytes are

DeepSeek-V3 is a **mixture of experts** ([Part 2](deepseek-2-the-v3-architecture.md) builds it from zero). Each of its 58 MoE layers holds 256 small feed-forward networks, the routed experts. A router picks 8 of them for each token. The same script counts where the weights live, straight from the model's `config.json`:

```text
== 3. DeepSeek-V3: where the 671B weights live (from config.json) ==
one expert = 3 x 7168 x 2048 = 44,040,192 parameters
routed experts: 58 layers x 256 x 44,040,192 = 653.91B (97.4%)
everything else (attention, dense layers, shared experts, routers, embedding, head) = 17.12B (2.6%)
total = 671.03B
per token: 8 of 256 routed experts in each of 58 layers = 20.43B parameters
at Q4_K_M (4.82 bits): one expert = 26.5 MB; routed experts read per token = 12.3 GB out of 394 GB; the rest = 10.3 GB
```

This is the most important table in the part. Of 671 billion weights, 97.4% are routed experts. A token reads only 20.4 billion of them, plus the 17.1 billion that every token uses. That gives the "37B active" in the V3 paper.

<figure class="fig"><svg viewBox="0 0 760 244" role="img" aria-label="Bar chart: 97.4 percent of DeepSeek-V3 weights are routed experts; one token reads about 20 billion expert weights plus 17 billion other weights."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">DeepSeek-V3's 671B weights: stored versus used for one token</text><text class="t-note" x="20.0" y="50.0" text-anchor="start">stored (all weights must be somewhere)</text><rect class="box-2" x="20.0" y="60.0" width="701.6" height="40.0" rx="4"/><rect class="box-on" x="721.6" y="60.0" width="18.4" height="40.0" rx="4"/><text class="t-note" x="32.0" y="85.0" text-anchor="start">routed experts: 653.9B (97.4%)</text><text class="t-tick" x="740.0" y="122.0" text-anchor="end">everything else: 17.1B (2.6%), always used</text><text class="t-note" x="20.0" y="140.0" text-anchor="start">read for one token (37B)</text><rect class="box-ghost" x="20.0" y="150.0" width="720.0" height="40.0" rx="4"/><rect class="box-2" x="20.0" y="150.0" width="21.9" height="40.0" rx="4"/><rect class="box-on" x="41.9" y="150.0" width="18.4" height="40.0" rx="4"/><text class="t-note" x="72.3" y="175.0" text-anchor="start">20.4B of experts (8 of 256 per layer) + 17.1B always</text><text class="t-muted" x="20.0" y="228.0" text-anchor="start">Memory size follows the top bar. Decode speed follows the bottom bar.</text></svg><figcaption>DeepSeek-V3 stores 653.9B weights of routed experts but one token reads only 20.4B of them, plus 17.1B weights that every token uses. Storage follows the top bar; work per token follows the bottom bar.</figcaption></figure>

Hold on to the two bars. Memory size follows the top bar: every expert must be stored, because the next token may pick any of them. Speed follows the bottom bar: each new token reads about 37 billion weights, not 671 billion. A dense model of 671B would read all of them for every token.

## The other memory bill: the KV cache

Weights are not the whole bill. While the model reads a prompt and writes an answer, it keeps notes about every token so far. These notes are the KV cache. [LLM Inference Part 2](llm-inference-2-kv-cache.md) explains the cache from zero; here we only need its size.

The size is "bytes per token x number of tokens". Bytes per token depend on the attention design. DeepSeek changed that design three times, and each change shrank the cache.

V2 and V3 use **Multi-head Latent Attention** (MLA). Per layer, it stores one compressed vector of 512 numbers plus a 64-number position key, shared by all 128 heads. [Attention Part 2](attention-2-mqa-gqa-mla.md) explains MLA, and [Part 2](deepseek-2-the-v3-architecture.md) of this series shows it inside V3.

$$
B_{\text{token}} = (d_c + d_R) \times L \times s
$$

Read it left to right. $$B_{\text{token}}$$ is the KV cache in bytes for one token. $$d_c$$ is the size of the compressed vector (512). $$d_R$$ is the size of the shared position key (64). $$L$$ is the number of layers. $$s$$ is the bytes per stored number (2 for BF16).

Tiny example: a toy model with $$d_c = 4$$, $$d_R = 2$$, 3 layers and BF16 stores $$(4+2) \times 3 \times 2 = 36$$ bytes per token. A prompt of 10 tokens needs 360 bytes. For real V3: $$(512 + 64) \times 61 \times 2 = 70{,}272$$ bytes per token, about 69 KiB.

> [!PAPER] Zhao et al. (2025), Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures (ISCA 2025) · §2.1.2, Table 1 · page 4
> [![Table 1 of the ISCA 2025 paper: KV cache per token in BF16, with 70.272 KB for DeepSeek-V3, 327.680 KB for Qwen-2.5 72B and 516.096 KB for LLaMA-3.1 405B highlighted](/img/deepseek/papers/p7-isca-kv.png)](/img/deepseek/papers/p7-isca-kv.png)
>
> **What it says:** DeepSeek-V3 with MLA needs "70.272 KB" of KV cache per token in BF16. Two dense models that use grouped-query attention need "327.680 KB" (Qwen-2.5 72B) and "516.096 KB" (LLaMA-3.1 405B).
>
> **What it means:**
>
> Our formula gives exactly 70,272 bytes, so the table uses decimal kilobytes. The comparison is striking because V3 is the largest model in the table. A bigger model usually has a bigger cache, since it has more layers and wider vectors. MLA breaks that link.
>
> The practical effect is on the second machine you buy, not the first. Weights are a fixed cost. The cache grows with every user and every token. A cache five times smaller means about five times more users, or five times longer prompts, in the same memory left over after the weights.

### Every model, from its config

The script reads each model's `config.json` and computes bytes per token with the formula that fits its attention type. The distills are normal Qwen and Llama models with grouped-query attention. Their formula is $$2 \times L \times h_{kv} \times d_h \times 2$$: keys and values, layers, key-value heads, head size and 2 bytes. For V3.2, V4 and V4.1, the script uses the FP8 and FP4 cache layouts from DeepSeek's FlashMLA kernels.

```python
def mla_bf16(c):
    return (c['kv_lora_rank'] + c['qk_rope_head_dim']) * c['num_hidden_layers'] * 2

def v32(c):     # FlashMLA FP8 row 656 B (latent + scales + RoPE) + FP8 indexer key 128 B + 4 B scale
    return c['num_hidden_layers'] * (656 + c['index_head_dim'] + 4)

def v4(c):      # V4 row 584 B per compressed entry; FP4 indexer key 64 B + 4 B scale on CSA layers
    r = c['compress_ratios'][:c['num_hidden_layers']]
    csa = sum(1 for x in r if x == 4)
    hca = sum(1 for x in r if x == 128)
    return csa * (584 + 68) / 4 + hca * 584 / 128, csa, hca
```

```text
== 4. KV cache per token, from each config.json ==
naive multi-head cache for V3 (for contrast): 128 heads x (128 + 64 + 128) x 61 x 2 B = 4,997,120 B per token
model                B/token       32K      128K        1M   how
V2-Lite               31,104     1.02G       n/a       n/a   MLA BF16: (512 + 64) x 27 layers x 2 B
V2                    69,120     2.26G     9.06G       n/a   MLA BF16: (512 + 64) x 60 layers x 2 B
V3 / R1 / V3.1        70,272     2.30G     9.21G       n/a   MLA BF16: (512 + 64) x 61 layers x 2 B
V3.2                  48,068     1.58G     6.30G       n/a   FP8 MLA row 656 B + indexer 132 B, x 61 layers
V4-Flash               3,514     0.12G     0.46G     3.68G   21 CSA layers x 652 B / 4 + 20 HCA layers x 584 B / 128
V4-Pro                 5,031     0.16G     0.66G     5.28G   30 CSA layers x 652 B / 4 + 31 HCA layers x 584 B / 128
V4.1-Flash               890     0.03G     0.12G     0.93G   published: 890 B per token (V4.1 report, abstract)
Distill-Qwen-1.5B     28,672     0.94G     3.76G       n/a   GQA BF16: 2 x 28 layers x 2 KV heads x 128 x 2 B
Distill-Qwen-7B       57,344     1.88G     7.52G       n/a   GQA BF16: 2 x 28 layers x 4 KV heads x 128 x 2 B
Distill-Llama-8B     131,072     4.29G    17.18G       n/a   GQA BF16: 2 x 32 layers x 8 KV heads x 128 x 2 B
Distill-Qwen-14B     196,608     6.44G    25.77G       n/a   GQA BF16: 2 x 48 layers x 8 KV heads x 128 x 2 B
Distill-Qwen-32B     262,144     8.59G    34.36G       n/a   GQA BF16: 2 x 64 layers x 8 KV heads x 128 x 2 B
Distill-Llama-70B    327,680    10.74G    42.95G       n/a   GQA BF16: 2 x 80 layers x 8 KV heads x 128 x 2 B
checks against published numbers:
  V3 70,272 B = ISCA 2025 Table 1 "70.272 KB"
  V3.2 48,068 B, V4-Flash 3,514 B: V4.1 report Figure 1(b) gives 48,068 and 3,514
  V4-Pro / V3.2 = 10.5% (V4 report: "10% of KV cache"); V4-Flash / V3.2 = 7.3% (report: "7%")
  V3 MLA versus naive multi-head: 71x smaller; Distill-Llama-8B needs 1.87x more per token than the 671B V3
```

"n/a" means the model does not support that context length. The checks at the bottom matter. Our numbers for V3, V3.2 and V4-Flash match the numbers DeepSeek published, byte for byte. V4-Pro has no published per-token figure, so 5,031 bytes is our estimate. It gives 10.5% of V3.2, and the V4 report says "10%". The estimate leaves out the small sliding-window cache, which has a fixed size of 128 tokens per layer and does not grow with the prompt.

<figure class="fig"><svg viewBox="0 0 760 374" role="img" aria-label="KV cache bytes per token on a log scale, from about 5 MB for a naive cache down to 890 bytes for V4.1-Flash."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">KV cache per token (log scale)</text><line class="edge-dim" x1="295.5" y1="36" x2="295.5" y2="316"/><text class="t-muted" x="295.5" y="332.0" text-anchor="middle">1 KB</text><line class="edge-dim" x1="382.6" y1="36" x2="382.6" y2="328"/><text class="t-muted" x="382.6" y="346.0" text-anchor="middle">10 KB</text><line class="edge-dim" x1="469.7" y1="36" x2="469.7" y2="316"/><text class="t-muted" x="469.7" y="332.0" text-anchor="middle">100 KB</text><line class="edge-dim" x1="556.7" y1="36" x2="556.7" y2="328"/><text class="t-muted" x="556.7" y="346.0" text-anchor="middle">1 MB</text><text class="t-tick" x="240.0" y="61.0" text-anchor="end">naive multi-head (V3 shape)</text><g class="mark"><title>naive multi-head (V3 shape): 4,997,120 B</title><rect class="s3" x="250" y="44" width="367.6" height="24" rx="4"/></g><text class="t-val" x="625.6" y="61.0" text-anchor="start">4,997,120 B</text><text class="t-tick" x="240.0" y="95.0" text-anchor="end">R1-Distill-Llama-70B (GQA)</text><g class="mark"><title>R1-Distill-Llama-70B (GQA): 327,680 B</title><rect class="s3" x="250" y="78" width="264.5" height="24" rx="4"/></g><text class="t-val" x="522.5" y="95.0" text-anchor="start">327,680 B</text><text class="t-tick" x="240.0" y="129.0" text-anchor="end">R1-Distill-Llama-8B (GQA)</text><g class="mark"><title>R1-Distill-Llama-8B (GQA): 131,072 B</title><rect class="s3" x="250" y="112" width="229.9" height="24" rx="4"/></g><text class="t-val" x="487.9" y="129.0" text-anchor="start">131,072 B</text><text class="t-tick" x="240.0" y="163.0" text-anchor="end">V3 / R1 / V3.1 (MLA, BF16)</text><g class="mark"><title>V3 / R1 / V3.1 (MLA, BF16): 70,272 B</title><rect class="s1" x="250" y="146" width="206.3" height="24" rx="4"/></g><text class="t-val" x="464.3" y="163.0" text-anchor="start">70,272 B</text><text class="t-tick" x="240.0" y="197.0" text-anchor="end">V3.2 (FP8 MLA + indexer)</text><g class="mark"><title>V3.2 (FP8 MLA + indexer): 48,068 B</title><rect class="s1" x="250" y="180" width="192.0" height="24" rx="4"/></g><text class="t-val" x="450.0" y="197.0" text-anchor="start">48,068 B</text><text class="t-tick" x="240.0" y="231.0" text-anchor="end">V4-Pro (CSA + HCA), estimate</text><g class="mark"><title>V4-Pro (CSA + HCA), estimate: 5,031 B</title><rect class="s2" x="250" y="214" width="106.6" height="24" rx="4"/></g><text class="t-val" x="364.6" y="231.0" text-anchor="start">5,031 B</text><text class="t-tick" x="240.0" y="265.0" text-anchor="end">V4-Flash (CSA + HCA)</text><g class="mark"><title>V4-Flash (CSA + HCA): 3,514 B</title><rect class="s2" x="250" y="248" width="93.0" height="24" rx="4"/></g><text class="t-val" x="351.0" y="265.0" text-anchor="start">3,514 B</text><text class="t-tick" x="240.0" y="299.0" text-anchor="end">V4.1-Flash (CSA2, FP4)</text><g class="mark"><title>V4.1-Flash (CSA2, FP4): 890 B</title><rect class="s2" x="250" y="282" width="41.1" height="24" rx="4"/></g><text class="t-val" x="299.1" y="299.0" text-anchor="start">890 B</text><line class="axis" x1="250" y1="311.0" x2="670" y2="311.0"/><text class="t-muted" x="250.0" y="364.0" text-anchor="start">bytes per token, log scale: each grid step is 10x</text></svg><figcaption>KV cache bytes per token on a log scale. A naive multi-head cache for V3's shape would need about 5 MB per token; MLA brings it to 70 KB, V3.2 to 48 KB, V4 to 3.5 to 5 KB, and V4.1-Flash to 890 bytes.</figcaption></figure>

Two lessons sit in this table. The small 8B Llama distill needs 1.87 times more cache per token than the 671B V3. And V4.1-Flash keeps a one-million-token conversation in 0.93 GB, less than V3 needs for 16K tokens.

> [!WARNING]
> A tool must support MLA to get MLA's small cache. Before llama.cpp added MLA in April 2025, it stored a full per-head cache for V3. One independent test on a Mac with V3 Q4_K_M measured 157,380 MiB of KV cache at 32K context. MLA in BF16 needs 2.3 GB for the same 32K tokens.

## What fits where

Now we can answer the question people actually ask: will it run on my machine? The rule we use is an estimate, and it is strict on purpose. A setup fits if the weights plus the KV cache of one 32K-token conversation take at most 90% of the memory. The other 10% goes to activations, engine buffers and the operating system.

```python
for label, w, kvm in ROWS:
    need = w + MODELS[kvm][0] * CTX / GB          # weights + KV cache of one 32K-token sequence
    row = [cap * USABLE >= need for _, cap in MACHINES]
```

```text
== 5. What fits where: weights + KV for one 32,768-token sequence <= 90% of memory ==
(capacities taken as decimal GB; RAM sold as "512 GB" is really 512 GiB, so this errs on the safe side)
setup                           need GB  16GB lap  24GB GPU  32GB GPU 128GB box 512GB Mac   8x H100   8x H200  2x8 H100 8x MI300X
V3/R1 FP8 official                690.9         -         -         -         -         -         -       yes       yes       yes
V3/R1 Q4_K_M                      406.7         -         -         -         -       yes       yes       yes       yes       yes
R1 UD-Q2_K_XL                     228.9         -         -         -         -       yes       yes       yes       yes       yes
R1 UD-IQ1_S                       142.5         -         -         -         -       yes       yes       yes       yes       yes
V3.2 FP8 official                 691.1         -         -         -         -         -         -       yes       yes       yes
V2 BF16 official                  473.8         -         -         -         -         -       yes       yes       yes       yes
V2 4-bit (Ollama)                 135.3         -         -         -         -       yes       yes       yes       yes       yes
V2-Lite 4-bit (Ollama)              9.9       yes       yes       yes       yes       yes       yes       yes       yes       yes
V4-Pro-0813 official              892.9         -         -         -         -         -         -       yes       yes       yes
V4-Flash official                 159.7         -         -         -         -       yes       yes       yes       yes       yes
V4-Flash UD-Q2_K_XL                96.9         -         -         -       yes       yes       yes       yes       yes       yes
V4.1-Flash-1017 official          510.3         -         -         -         -         -       yes       yes       yes       yes
V4.1 UD-IQ1_S, Engram on SSD      122.4         -         -         -         -       yes       yes       yes       yes       yes
Distill-Llama-70B Q4_K_M           53.2         -         -         -       yes       yes       yes       yes       yes       yes
Distill-Qwen-32B Q4_K_M            28.5         -         -       yes       yes       yes       yes       yes       yes       yes
Distill-Qwen-14B Q4_K_M            15.4         -       yes       yes       yes       yes       yes       yes       yes       yes
Distill-Qwen-7B Q4_K_M              6.6       yes       yes       yes       yes       yes       yes       yes       yes       yes
```

<figure class="fig"><svg viewBox="0 0 742 568" role="img" aria-label="A grid of 17 DeepSeek setups against 9 machines showing which fit in memory."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Weights + one 32K-token KV cache within 90% of memory</text><text class="t-tick" x="263.0" y="86.0" text-anchor="start" transform="rotate(-40 263.0 86.0)">16GB lap</text><text class="t-tick" x="317.0" y="86.0" text-anchor="start" transform="rotate(-40 317.0 86.0)">24GB GPU</text><text class="t-tick" x="371.0" y="86.0" text-anchor="start" transform="rotate(-40 371.0 86.0)">32GB GPU</text><text class="t-tick" x="425.0" y="86.0" text-anchor="start" transform="rotate(-40 425.0 86.0)">128GB box</text><text class="t-tick" x="479.0" y="86.0" text-anchor="start" transform="rotate(-40 479.0 86.0)">512GB Mac</text><text class="t-tick" x="533.0" y="86.0" text-anchor="start" transform="rotate(-40 533.0 86.0)">8x H100</text><text class="t-tick" x="587.0" y="86.0" text-anchor="start" transform="rotate(-40 587.0 86.0)">8x H200</text><text class="t-tick" x="641.0" y="86.0" text-anchor="start" transform="rotate(-40 641.0 86.0)">2x8 H100</text><text class="t-tick" x="695.0" y="86.0" text-anchor="start" transform="rotate(-40 695.0 86.0)">8x MI300X</text><text class="t-tick" x="226.0" y="113.0" text-anchor="end">V3/R1 FP8 official (691)</text><rect class="box-ghost" x="239.0" y="99.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="99.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="347.0" y="99.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="401.0" y="99.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="455.0" y="99.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="509.0" y="99.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="563.0" y="99.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="113.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="99.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="113.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="99.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="113.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="139.0" text-anchor="end">V3/R1 Q4_K_M (407)</text><rect class="box-ghost" x="239.0" y="125.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="125.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="347.0" y="125.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="401.0" y="125.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="455.0" y="125.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="479.0" y="139.0" text-anchor="middle">fits</text><rect class="box-1" x="509.0" y="125.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="139.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="125.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="139.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="125.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="139.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="125.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="139.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="165.0" text-anchor="end">R1 UD-Q2_K_XL (229)</text><rect class="box-ghost" x="239.0" y="151.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="151.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="347.0" y="151.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="401.0" y="151.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="455.0" y="151.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="479.0" y="165.0" text-anchor="middle">fits</text><rect class="box-1" x="509.0" y="151.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="165.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="151.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="165.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="151.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="165.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="151.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="165.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="191.0" text-anchor="end">R1 UD-IQ1_S (142)</text><rect class="box-ghost" x="239.0" y="177.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="177.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="347.0" y="177.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="401.0" y="177.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="455.0" y="177.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="479.0" y="191.0" text-anchor="middle">fits</text><rect class="box-1" x="509.0" y="177.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="191.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="177.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="191.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="177.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="191.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="177.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="191.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="217.0" text-anchor="end">V3.2 FP8 official (691)</text><rect class="box-ghost" x="239.0" y="203.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="203.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="347.0" y="203.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="401.0" y="203.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="455.0" y="203.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="509.0" y="203.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="563.0" y="203.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="217.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="203.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="217.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="203.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="217.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="243.0" text-anchor="end">V2 BF16 official (474)</text><rect class="box-ghost" x="239.0" y="229.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="229.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="347.0" y="229.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="401.0" y="229.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="455.0" y="229.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="509.0" y="229.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="243.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="229.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="243.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="229.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="243.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="229.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="243.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="269.0" text-anchor="end">V2 4-bit (Ollama) (135)</text><rect class="box-ghost" x="239.0" y="255.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="255.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="347.0" y="255.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="401.0" y="255.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="455.0" y="255.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="479.0" y="269.0" text-anchor="middle">fits</text><rect class="box-1" x="509.0" y="255.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="269.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="255.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="269.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="255.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="269.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="255.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="269.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="295.0" text-anchor="end">V2-Lite 4-bit (Ollama) (10)</text><rect class="box-1" x="239.0" y="281.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="263.0" y="295.0" text-anchor="middle">fits</text><rect class="box-1" x="293.0" y="281.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="317.0" y="295.0" text-anchor="middle">fits</text><rect class="box-1" x="347.0" y="281.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="371.0" y="295.0" text-anchor="middle">fits</text><rect class="box-1" x="401.0" y="281.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="425.0" y="295.0" text-anchor="middle">fits</text><rect class="box-1" x="455.0" y="281.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="479.0" y="295.0" text-anchor="middle">fits</text><rect class="box-1" x="509.0" y="281.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="295.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="281.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="295.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="281.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="295.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="281.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="295.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="321.0" text-anchor="end">V4-Pro-0813 official (893)</text><rect class="box-ghost" x="239.0" y="307.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="307.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="347.0" y="307.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="401.0" y="307.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="455.0" y="307.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="509.0" y="307.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="563.0" y="307.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="321.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="307.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="321.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="307.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="321.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="347.0" text-anchor="end">V4-Flash official (160)</text><rect class="box-ghost" x="239.0" y="333.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="333.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="347.0" y="333.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="401.0" y="333.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="455.0" y="333.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="479.0" y="347.0" text-anchor="middle">fits</text><rect class="box-1" x="509.0" y="333.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="347.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="333.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="347.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="333.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="347.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="333.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="347.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="373.0" text-anchor="end">V4-Flash UD-Q2_K_XL (97)</text><rect class="box-ghost" x="239.0" y="359.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="359.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="347.0" y="359.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="401.0" y="359.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="425.0" y="373.0" text-anchor="middle">fits</text><rect class="box-1" x="455.0" y="359.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="479.0" y="373.0" text-anchor="middle">fits</text><rect class="box-1" x="509.0" y="359.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="373.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="359.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="373.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="359.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="373.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="359.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="373.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="399.0" text-anchor="end">V4.1-Flash-1017 official (510)</text><rect class="box-ghost" x="239.0" y="385.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="385.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="347.0" y="385.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="401.0" y="385.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="455.0" y="385.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="509.0" y="385.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="399.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="385.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="399.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="385.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="399.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="385.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="399.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="425.0" text-anchor="end">V4.1 UD-IQ1_S, Engram on SSD (122)</text><rect class="box-ghost" x="239.0" y="411.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="411.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="347.0" y="411.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="401.0" y="411.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="455.0" y="411.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="479.0" y="425.0" text-anchor="middle">fits</text><rect class="box-1" x="509.0" y="411.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="425.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="411.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="425.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="411.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="425.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="411.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="425.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="451.0" text-anchor="end">Distill-Llama-70B Q4_K_M (53)</text><rect class="box-ghost" x="239.0" y="437.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="437.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="347.0" y="437.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="401.0" y="437.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="425.0" y="451.0" text-anchor="middle">fits</text><rect class="box-1" x="455.0" y="437.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="479.0" y="451.0" text-anchor="middle">fits</text><rect class="box-1" x="509.0" y="437.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="451.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="437.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="451.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="437.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="451.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="437.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="451.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="477.0" text-anchor="end">Distill-Qwen-32B Q4_K_M (28)</text><rect class="box-ghost" x="239.0" y="463.0" width="48.0" height="20.0" rx="3"/><rect class="box-ghost" x="293.0" y="463.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="347.0" y="463.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="371.0" y="477.0" text-anchor="middle">fits</text><rect class="box-1" x="401.0" y="463.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="425.0" y="477.0" text-anchor="middle">fits</text><rect class="box-1" x="455.0" y="463.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="479.0" y="477.0" text-anchor="middle">fits</text><rect class="box-1" x="509.0" y="463.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="477.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="463.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="477.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="463.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="477.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="463.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="477.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="503.0" text-anchor="end">Distill-Qwen-14B Q4_K_M (15)</text><rect class="box-ghost" x="239.0" y="489.0" width="48.0" height="20.0" rx="3"/><rect class="box-1" x="293.0" y="489.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="317.0" y="503.0" text-anchor="middle">fits</text><rect class="box-1" x="347.0" y="489.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="371.0" y="503.0" text-anchor="middle">fits</text><rect class="box-1" x="401.0" y="489.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="425.0" y="503.0" text-anchor="middle">fits</text><rect class="box-1" x="455.0" y="489.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="479.0" y="503.0" text-anchor="middle">fits</text><rect class="box-1" x="509.0" y="489.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="503.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="489.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="503.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="489.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="503.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="489.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="503.0" text-anchor="middle">fits</text><text class="t-tick" x="226.0" y="529.0" text-anchor="end">Distill-Qwen-7B Q4_K_M (7)</text><rect class="box-1" x="239.0" y="515.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="263.0" y="529.0" text-anchor="middle">fits</text><rect class="box-1" x="293.0" y="515.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="317.0" y="529.0" text-anchor="middle">fits</text><rect class="box-1" x="347.0" y="515.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="371.0" y="529.0" text-anchor="middle">fits</text><rect class="box-1" x="401.0" y="515.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="425.0" y="529.0" text-anchor="middle">fits</text><rect class="box-1" x="455.0" y="515.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="479.0" y="529.0" text-anchor="middle">fits</text><rect class="box-1" x="509.0" y="515.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="533.0" y="529.0" text-anchor="middle">fits</text><rect class="box-1" x="563.0" y="515.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="587.0" y="529.0" text-anchor="middle">fits</text><rect class="box-1" x="617.0" y="515.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="641.0" y="529.0" text-anchor="middle">fits</text><rect class="box-1" x="671.0" y="515.0" width="48.0" height="20.0" rx="3"/><text class="t-tick" x="695.0" y="529.0" text-anchor="middle">fits</text><text class="t-muted" x="236.0" y="560.0" text-anchor="start">Number in brackets: GB needed. Filled cell: fits by memory alone (not a speed claim).</text></svg><figcaption>Which DeepSeek setups fit which machines by memory alone. The native FP8 V3 and the official V4-Pro need at least eight H200 GPUs; low-bit V3 files need a 512 GB machine; V4-Flash at 2 bits and the 70B distill fit a 128 GB box.</figcaption></figure>

The grid answers "does it fit", not "is it fast" or "does the software support it". Three notes keep it honest.

The 8x H100 column is the famous gap. Eight H100 GPUs have 8 x 80 = 640 GB, and the native V3 file is 688.6 GB. That is why every guide says V3 in FP8 needs either 8 H200s (1,128 GB) or two H100 nodes. The column also says V4.1-Flash fits 8x H100 by memory. In practice, DeepSeek's own V4.1 engine targets newer NVIDIA SM100 GPUs and Huawei Ascend 950 chips, so memory is not the only gate.

The 128 GB column is where V4-Flash at 2 bits appears, and where V4.1-Flash with its Engram tables on SSD comes close. V4.1-Flash stores a large "Engram" memory of n-gram tables ([Part 6](deepseek-6-v4-and-v4-1.md)). It reads only a few rows per token, so the tables can stay on an SSD. Unsloth's table splits each V4.1 file into "Engram" and "Rest".

For UD-IQ1_S, 110.6 GB is Engram and 122.4 GB is the rest, so the rest is what must sit in RAM or GPU memory. By our 90% rule, 122.4 GB is just too much for a 128 GB box. Unsloth itself lists 128 to 145 GB of RAM plus VRAM for its 1-bit V4.1 files.

The distills are a different kind of model. They are Qwen 2.5 and Llama 3 models, fine-tuned on R1's answers. They have no MLA and no experts. They fit laptops because they are small, not because of DeepSeek's architecture.

## Data-centre recipes: GPUs, flags and why

### Splitting one model over many GPUs

When a model does not fit one GPU, engines split it. There are three basic ways, and real engines mix them. [LLM Inference Part 7](llm-inference-7-beyond-one-gpu.md) covers them in depth.

<figure class="fig"><svg viewBox="0 0 760 324" role="img" aria-label="Three panels: tensor parallel splits every matrix, data parallel copies the model, expert parallel splits the experts."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Three ways to spread one model over 4 GPUs</text><text class="t-note" x="20.0" y="58.0" text-anchor="start">Tensor parallel (TP)</text><rect class="box" x="20.0" y="72.0" width="220.0" height="38.0" rx="6"/><text class="t-tick" x="28.0" y="96.0" text-anchor="start">GPU 0</text><rect class="box-1" x="82.0" y="80.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="97.0" y="96.0" text-anchor="middle">L1</text><rect class="box-1" x="120.0" y="80.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="135.0" y="96.0" text-anchor="middle">L2</text><rect class="box-1" x="158.0" y="80.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="173.0" y="96.0" text-anchor="middle">L3</text><rect class="box-1" x="196.0" y="80.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="211.0" y="96.0" text-anchor="middle">L4</text><rect class="box" x="20.0" y="118.0" width="220.0" height="38.0" rx="6"/><text class="t-tick" x="28.0" y="142.0" text-anchor="start">GPU 1</text><rect class="box-1" x="82.0" y="126.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="97.0" y="142.0" text-anchor="middle">L1</text><rect class="box-1" x="120.0" y="126.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="135.0" y="142.0" text-anchor="middle">L2</text><rect class="box-1" x="158.0" y="126.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="173.0" y="142.0" text-anchor="middle">L3</text><rect class="box-1" x="196.0" y="126.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="211.0" y="142.0" text-anchor="middle">L4</text><rect class="box" x="20.0" y="164.0" width="220.0" height="38.0" rx="6"/><text class="t-tick" x="28.0" y="188.0" text-anchor="start">GPU 2</text><rect class="box-1" x="82.0" y="172.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="97.0" y="188.0" text-anchor="middle">L1</text><rect class="box-1" x="120.0" y="172.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="135.0" y="188.0" text-anchor="middle">L2</text><rect class="box-1" x="158.0" y="172.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="173.0" y="188.0" text-anchor="middle">L3</text><rect class="box-1" x="196.0" y="172.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="211.0" y="188.0" text-anchor="middle">L4</text><rect class="box" x="20.0" y="210.0" width="220.0" height="38.0" rx="6"/><text class="t-tick" x="28.0" y="234.0" text-anchor="start">GPU 3</text><rect class="box-1" x="82.0" y="218.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="97.0" y="234.0" text-anchor="middle">L1</text><rect class="box-1" x="120.0" y="218.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="135.0" y="234.0" text-anchor="middle">L2</text><rect class="box-1" x="158.0" y="218.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="173.0" y="234.0" text-anchor="middle">L3</text><rect class="box-1" x="196.0" y="218.0" width="30.0" height="22.0" rx="3"/><text class="t-tick" x="211.0" y="234.0" text-anchor="middle">L4</text><text class="t-muted" x="20.0" y="274.0" text-anchor="start">every GPU holds 1/4 of every matrix;</text><text class="t-muted" x="20.0" y="290.0" text-anchor="start">they talk at every layer</text><text class="t-note" x="268.0" y="58.0" text-anchor="start">Data parallel (DP)</text><rect class="box" x="268.0" y="72.0" width="220.0" height="38.0" rx="6"/><text class="t-tick" x="276.0" y="96.0" text-anchor="start">GPU 0</text><rect class="box-1" x="330.0" y="80.0" width="146.0" height="22.0" rx="3"/><text class="t-tick" x="403.0" y="96.0" text-anchor="middle">full model, requests A</text><rect class="box" x="268.0" y="118.0" width="220.0" height="38.0" rx="6"/><text class="t-tick" x="276.0" y="142.0" text-anchor="start">GPU 1</text><rect class="box-1" x="330.0" y="126.0" width="146.0" height="22.0" rx="3"/><text class="t-tick" x="403.0" y="142.0" text-anchor="middle">full model, requests B</text><rect class="box" x="268.0" y="164.0" width="220.0" height="38.0" rx="6"/><text class="t-tick" x="276.0" y="188.0" text-anchor="start">GPU 2</text><rect class="box-1" x="330.0" y="172.0" width="146.0" height="22.0" rx="3"/><text class="t-tick" x="403.0" y="188.0" text-anchor="middle">full model, requests C</text><rect class="box" x="268.0" y="210.0" width="220.0" height="38.0" rx="6"/><text class="t-tick" x="276.0" y="234.0" text-anchor="start">GPU 3</text><rect class="box-1" x="330.0" y="218.0" width="146.0" height="22.0" rx="3"/><text class="t-tick" x="403.0" y="234.0" text-anchor="middle">full model, requests D</text><text class="t-muted" x="268.0" y="274.0" text-anchor="start">every GPU holds a full copy;</text><text class="t-muted" x="268.0" y="290.0" text-anchor="start">each serves different requests</text><text class="t-note" x="516.0" y="58.0" text-anchor="start">Expert parallel (EP)</text><rect class="box" x="516.0" y="72.0" width="220.0" height="38.0" rx="6"/><text class="t-tick" x="524.0" y="96.0" text-anchor="start">GPU 0</text><rect class="box-2" x="578.0" y="80.0" width="146.0" height="22.0" rx="3"/><text class="t-tick" x="651.0" y="96.0" text-anchor="middle">experts 0-63</text><rect class="box" x="516.0" y="118.0" width="220.0" height="38.0" rx="6"/><text class="t-tick" x="524.0" y="142.0" text-anchor="start">GPU 1</text><rect class="box-2" x="578.0" y="126.0" width="146.0" height="22.0" rx="3"/><text class="t-tick" x="651.0" y="142.0" text-anchor="middle">experts 64-127</text><rect class="box" x="516.0" y="164.0" width="220.0" height="38.0" rx="6"/><text class="t-tick" x="524.0" y="188.0" text-anchor="start">GPU 2</text><rect class="box-2" x="578.0" y="172.0" width="146.0" height="22.0" rx="3"/><text class="t-tick" x="651.0" y="188.0" text-anchor="middle">experts 128-191</text><rect class="box" x="516.0" y="210.0" width="220.0" height="38.0" rx="6"/><text class="t-tick" x="524.0" y="234.0" text-anchor="start">GPU 3</text><rect class="box-2" x="578.0" y="218.0" width="146.0" height="22.0" rx="3"/><text class="t-tick" x="651.0" y="234.0" text-anchor="middle">experts 192-255</text><text class="t-muted" x="516.0" y="274.0" text-anchor="start">each GPU holds different experts;</text><text class="t-muted" x="516.0" y="290.0" text-anchor="start">tokens travel to their experts</text><text class="t-muted" x="20.0" y="312.0" text-anchor="start">TP: box L1 on GPU 0 is one quarter of layer 1. Real engines mix all three: DP for attention, EP for experts.</text></svg><figcaption>Three ways to spread one model over four GPUs. Tensor parallelism slices every matrix; data parallelism copies the model and splits the requests; expert parallelism gives each GPU different experts.</figcaption></figure>

DeepSeek's models suit a mix. The attention part (MLA) is small, so engines often copy it on every GPU with DP. The experts are 97% of the weights, so engines spread them with EP. SGLang calls this "DP attention".

> [!DEFINITION] NVLink and InfiniBand
> Two kinds of wire between GPUs. **NVLink** connects the 8 GPUs inside one server and is very fast. **InfiniBand** (IB) connects servers. For the H800 GPUs DeepSeek used, the DeepEP README gives about 160 GB/s over NVLink and about 50 GB/s over IB per GPU.

### Speculative decoding with MTP and DSpark

One more idea appears in every recipe. During training, V3 also learned to predict the token after the next one. That extra head is the **MTP module**. At serving time, it can propose a second token for free, and the main model checks it. [LLM Inference Part 4](llm-inference-4-speculative-decoding.md) explains why checking is cheap.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §5.4.3 · page 35
> [![Section 5.4.3 of the V3 paper with between 85% and 90% and delivering 1.8 times TPS (Tokens Per Second) highlighted](/img/deepseek/papers/p7-v3-mtp.png)](/img/deepseek/papers/p7-v3-mtp.png)
>
> **What it says:** the acceptance rate of the second predicted token "ranges between 85% and 90%", and this gives "1.8 times TPS (Tokens Per Second)".
>
> **What it means:**
>
> Think of a student who writes one word and also pencils in a guess for the next word. If the teacher agrees with the pencil guess 85 to 90 times out of 100, most steps produce two words instead of one. The 1.8x is close to the best case of 2x.
>
> The catch is hidden in "TPS". Speculation helps most when the GPU has spare compute, which means small batches and one or few users. In a full server with large batches, the GPU is already busy, so the gain shrinks. Engine docs show exactly that pattern below.

V4 replaced the single MTP head with **DSpark**, a drafter that proposes several tokens at once ([Part 6](deepseek-6-v4-and-v4-1.md)). The V4-Pro-0813 checkpoint ships with its DSpark module attached, which is one reason it is 892.7 GB, against 864.7 GB for the April preview.

### The recipes, model by model

All flags below are copied from the engine docs or DeepSeek's model cards. Numbers are vendor claims unless the text says otherwise.

**V3, R1, V3.1 on one node of 8x H200 or 8x MI300X.** The native FP8 file fits, with about 440 GB left for KV cache and buffers. SGLang's launch line is short:

```bash
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8
#   add --reasoning-parser deepseek-v3 for R1-style thinking output
#   add --attn-dp-size 8 for high-throughput, large-batch serving (DP attention)
#   add --speculative-algorithm EAGLE for MTP speculation
#       (defaults: --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
```

The SGLang page warns not to add `--quantization fp8`, because the checkpoint is already FP8. It reports MTP "up to 1.8x decoding speed at batch size 1 and 1.5x at batch size 32" on 8x H200. On 8x MI300X with SGLang 0.5.7 and 1K-token inputs and outputs, it reports 34.26 output tokens per second at one request with EAGLE. At 100 concurrent requests with `--ep 8 --attn-dp-size 8`, the total is 463.80 output tokens per second. The gain for one user is small, the gain for the whole server is large, exactly as the MTP box predicted.

vLLM's recipe uses the same hardware:

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528 --trust-remote-code --tensor-parallel-size 8 --enable-expert-parallel
#   or: --data-parallel-size 8 --enable-expert-parallel
#   AMD ROCm: VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1
```

**V3 on H100, H800, H20 or A100.** The SGLang hardware table lists FP8 V3 on "2x8 H100/H800/H20", that is, two nodes with tensor parallelism across both. If you upcast to BF16 (1.34 TB), you need 2x8 H200, 4x8 H100 or 4x8 A100. There is also an INT8 path on 16 A100s, 32 L40S cards, a Xeon 6980P CPU server, or 4x Huawei Ascend A3. With 4-bit weights (W4A8, AWQ, NVFP4 and similar), one node of 8x H20 or 8x H100 is enough.

> [!DEFINITION] H20 and H800
> GPUs that NVIDIA made for the Chinese market under US export rules. The **H800** is close to an H100 with a slower link between GPUs; DeepSeek trained and served V3 on it. The **H20** has a lot of memory but much less compute than an H100.

**V3.2 and V3.2-Exp.** DeepSeek's own README gives SGLang images for H200, AMD MI350 and Huawei Ascend, launched with `--tp 8 --dp 8 --enable-dp-attention`. The vLLM recipe recommends `-dp 8 --enable-expert-parallel` on 8x H200, 8x H20 or 8x B200, and says the sparse-attention kernels are "mainly optimized for TP=1 (DP=8, EP=8)". It uses an FP8 KV cache by default. Some H20 users found `VLLM_USE_DEEP_GEMM=0` faster. [Part 5](deepseek-5-sparse-attention-and-v3-2.md) explains the sparse attention these kernels run.

**V4-Pro and V4-Flash.** DeepSeek's V4-Pro-0813 model card gives two launch lines. Each turns on DSpark, an FP8 KV cache and expert parallelism:

```bash
# vLLM, one node of 4x GB300
vllm serve deepseek-ai/DeepSeek-V4-Pro-0813 --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
  --data-parallel-size 4 --enable-expert-parallel --moe-backend deep_gemm_mega_moe \
  --attention-config '{"use_fp4_indexer_cache": true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'

# SGLang
sglang serve --model-path deepseek-ai/DeepSeek-V4-Pro-0813 --tp 4 --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algorithm DSPARK --mem-fraction-static 0.90 --chunked-prefill-size 4096 --swa-full-tokens-ratio 0.1
```

The vLLM recipe defaults V4-Pro to 8x H200 and caps the context at 800,000 tokens to keep room for the KV cache.

SGLang's cookbook puts V4-Flash on 4 B200, B300, GB200, GB300 or H200 GPUs. It also fits 2 RTX PRO 6000 cards (96 GB each) or 8 H100s. V4-Pro needs 8 B200s or B300s, 4 GB300s, or one 8x H200 node with FP4 weights. With FP8 weights on H200, or on H100, it needs two nodes. The cookbook says V4-Pro "does not fit on 8x 96 GB RTX PRO 6000", and that "a single 128 GB GB10 cannot hold any of these checkpoints". Neither recipe page publishes tokens-per-second results for V4 that we could read.

**V4.1-Flash.** DeepSeek's announcement says it "will work closely with the open-source community on V4.1-Flash inference support". The reference code splits the weights for 8-way tensor parallelism. The newest option is DeepSeek's own engine, DeepInfer, covered below.

> [!TIP] In production
> Start from the engine's own recipe page for your exact model and GPU, not from a blog post. Flags change between releases, and DeepSeek's checkpoints often need a matching kernel backend (DeepGEMM, FlashMLA, FlashInfer). Turn on DP attention and EP for throughput; turn on MTP or DSpark when one user's speed matters more.

## Why MoE models suit CPU and GPU together

### The speed rule for writing tokens

Writing one token means reading every active weight once from memory. On a single user's machine, the processor spends most of its time waiting for those bytes, not doing maths. [LLM Inference Part 1](llm-inference-1-prefill-and-decode.md) shows why.

> [!DEFINITION] Memory bandwidth
> How many bytes per second the memory can deliver to the processor. An Apple M3 Ultra reaches "over 800GB/s". One CPU socket with 8 channels of DDR5-4800 memory has about 8 x 38.4 = 307 GB/s.

This gives a simple ceiling for one user:

$$
\text{tokens per second} \le \frac{\text{bandwidth}}{\text{bytes read per token}}
$$

The bytes read per token are the active weights times the bytes per weight. It is a ceiling, not a forecast: real systems also read the KV cache, run attention, and never reach full bandwidth.

Tiny example: memory that delivers 100 GB/s, and a model that reads 10 GB per token, can write at most 10 tokens per second. Now the real numbers from `part7_serving.py`:

```text
== 1. Decode ceiling: tokens/s <= bandwidth / bytes read per token ==
M3 Ultra, "over 800GB/s"                         V3/R1 Q4_K_M, 37B active                      22.3 GB/token -> at most  35.9 tok/s
M3 Ultra, "over 800GB/s"                         R1-Distill-Llama-70B Q4_K_M, 70.55B dense     42.5 GB/token -> at most  18.8 tok/s
2 sockets x 8 ch DDR5-4800 (2 x 8 x 38.4 GB/s)   V3/R1 Q4_K_M, 37B active                      22.3 GB/token -> at most  27.6 tok/s
2 sockets x 8 ch DDR5-4800 (2 x 8 x 38.4 GB/s)   R1-Distill-Llama-70B Q4_K_M, 70.55B dense     42.5 GB/token -> at most  14.5 tok/s
measured (third party): R1 4-bit on M3 Ultra 512 GB about 17-18 tok/s (short prompts);
V3-0324 Q4_K_M: 10.89 tok/s after a 4.8K prompt, 6.67 tok/s after a 12.7K prompt
share of the ceiling reached at 17.5 tok/s: 49%
```

Look at the first two lines. The 671B MoE model has a higher speed ceiling than the 70B dense distill, because it reads fewer bytes per token. That is the whole reason a 671B model is usable on one desk at all.

<figure class="fig"><svg viewBox="0 0 760 258" role="img" aria-label="On an M3 Ultra, the bandwidth ceiling for V3 is 35.9 tokens per second; reports measure 17-18 for short prompts and less for long ones."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">M3 Ultra (over 800 GB/s): ceiling versus reported speed</text><text class="t-tick" x="290.0" y="61.0" text-anchor="end">ceiling, V3/R1 Q4_K_M (37B active)</text><g class="mark"><title>ceiling, V3/R1 Q4_K_M (37B active): at most 35.9 tok/s</title><rect class="s1" x="300" y="44" width="332.1" height="24" rx="4"/></g><text class="t-val" x="640.1" y="61.0" text-anchor="start">at most 35.9 tok/s</text><text class="t-tick" x="290.0" y="95.0" text-anchor="end">measured, R1 4-bit, short prompt</text><g class="mark"><title>measured, R1 4-bit, short prompt: about 17-18 tok/s</title><rect class="s2" x="300" y="78" width="161.9" height="24" rx="4"/></g><text class="t-val" x="469.9" y="95.0" text-anchor="start">about 17-18 tok/s</text><text class="t-tick" x="290.0" y="129.0" text-anchor="end">measured, V3-0324 Q4_K_M, 4.8K prompt</text><g class="mark"><title>measured, V3-0324 Q4_K_M, 4.8K prompt: 10.89 tok/s</title><rect class="s2" x="300" y="112" width="100.7" height="24" rx="4"/></g><text class="t-val" x="408.7" y="129.0" text-anchor="start">10.89 tok/s</text><text class="t-tick" x="290.0" y="163.0" text-anchor="end">measured, V3-0324 Q4_K_M, 12.7K prompt</text><g class="mark"><title>measured, V3-0324 Q4_K_M, 12.7K prompt: 6.67 tok/s</title><rect class="s2" x="300" y="146" width="61.7" height="24" rx="4"/></g><text class="t-val" x="369.7" y="163.0" text-anchor="start">6.67 tok/s</text><text class="t-tick" x="290.0" y="197.0" text-anchor="end">ceiling, dense 70B distill Q4_K_M</text><g class="mark"><title>ceiling, dense 70B distill Q4_K_M: at most 18.8 tok/s</title><rect class="s3" x="300" y="180" width="173.9" height="24" rx="4"/></g><text class="t-val" x="481.9" y="197.0" text-anchor="start">at most 18.8 tok/s</text><line class="axis" x1="300" y1="209.0" x2="670" y2="209.0"/><text class="t-muted" x="300.0" y="248.0" text-anchor="start">output tokens per second (one user)</text></svg><figcaption>On an M3 Ultra, the bandwidth ceiling for V3/R1 at Q4_K_M is 35.9 tokens per second. Reports measure about 17-18 for short prompts and 6.67 to 10.89 after longer prompts. A dense 70B model has a lower ceiling, 18.8.</figcaption></figure>

> [!PAPER] Zhao et al. (2025), Insights into DeepSeek-V3 (ISCA 2025) · §2.2.2 · page 4
> [![Section 2.2.2 of the ISCA paper with activates just 21B parameters, nearly 20 tokens per second (TPS), single-digit TPS, costing approximately $10,000 and still achieving nearly 20 TPS highlighted](/img/deepseek/papers/p7-isca-personal.png)](/img/deepseek/papers/p7-isca-personal.png)
>
> **What it says:** because V2 "activates just 21B parameters", PCs with AI chips reach "nearly 20 tokens per second (TPS)", while dense models of similar quality reach "single-digit TPS". KTransformers runs the full V3 on a server with a consumer GPU "costing approximately $10,000", "still achieving nearly 20 TPS".
>
> **What it means:**
>
> DeepSeek's own engineers make the same argument as the bandwidth rule. A dense model must read all its weights for every token. A sparse model reads only the chosen experts. On a machine with lots of slow memory and little fast memory, that difference decides whether the model is usable.
>
> Read the claim with care. "Nearly 20 TPS" for V3 on a $10,000 server is DeepSeek quoting a third-party engine. And the best KTransformers numbers below use 6 experts instead of 8. It is a fair picture of what is possible, not a promise for your machine.

### Experts can live in different places

The bandwidth rule has a second consequence. Only 8 of 256 experts are read per token. The other 248 can sit in slower, cheaper memory, as long as the chosen ones arrive in time. This turns memory into tiers, like a kitchen. The things you use every minute sit on the counter, the weekly things in the cupboard, the rare things in the basement.

<figure class="fig"><svg viewBox="0 0 760 336" role="img" aria-label="Three memory tiers, GPU, RAM and SSD, each holding experts; one token reads four chosen experts spread over the tiers."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Where the experts can live: three tiers, one token</text><rect class="box" x="20.0" y="54.0" width="600.0" height="72.0" rx="8"/><text class="t-note" x="32.0" y="76.0" text-anchor="start">GPU memory (fast, small)</text><text class="t-muted" x="32.0" y="94.0" text-anchor="start">attention, shared expert, KV cache, hot experts</text><rect class="box-ghost" x="330.0" y="84.0" width="18.0" height="26.0" rx="3"/><rect class="box-2" x="353.0" y="84.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="376.0" y="84.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="399.0" y="84.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="422.0" y="84.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="445.0" y="84.0" width="18.0" height="26.0" rx="3"/><rect class="box" x="20.0" y="140.0" width="600.0" height="72.0" rx="8"/><text class="t-note" x="32.0" y="162.0" text-anchor="start">RAM (slower, big)</text><text class="t-muted" x="32.0" y="180.0" text-anchor="start">most routed experts</text><rect class="box-ghost" x="330.0" y="170.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="353.0" y="170.0" width="18.0" height="26.0" rx="3"/><rect class="box-2" x="376.0" y="170.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="399.0" y="170.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="422.0" y="170.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="445.0" y="170.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="468.0" y="170.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="491.0" y="170.0" width="18.0" height="26.0" rx="3"/><rect class="box-2" x="514.0" y="170.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="537.0" y="170.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="560.0" y="170.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="583.0" y="170.0" width="18.0" height="26.0" rx="3"/><rect class="box" x="20.0" y="226.0" width="600.0" height="72.0" rx="8"/><text class="t-note" x="32.0" y="248.0" text-anchor="start">SSD (slowest, huge)</text><text class="t-muted" x="32.0" y="266.0" text-anchor="start">cold experts or Engram tables, read on demand</text><rect class="box-ghost" x="330.0" y="256.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="353.0" y="256.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="376.0" y="256.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="399.0" y="256.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="422.0" y="256.0" width="18.0" height="26.0" rx="3"/><rect class="box-2" x="445.0" y="256.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="468.0" y="256.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="491.0" y="256.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="514.0" y="256.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="537.0" y="256.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="560.0" y="256.0" width="18.0" height="26.0" rx="3"/><rect class="box-ghost" x="583.0" y="256.0" width="18.0" height="26.0" rx="3"/><rect class="box-on" x="660.0" y="156.0" width="80.0" height="30.0" rx="6"/><text class="t-note" x="700.0" y="176.0" text-anchor="middle">token</text><line class="path" x1="660.0" y1="171.0" x2="375.0" y2="97.0" marker-end="url(#ah-on)"/><line class="path" x1="660.0" y1="171.0" x2="398.0" y2="183.0" marker-end="url(#ah-on)"/><line class="path" x1="660.0" y1="171.0" x2="536.0" y2="183.0" marker-end="url(#ah-on)"/><line class="path" x1="660.0" y1="171.0" x2="467.0" y2="269.0" marker-end="url(#ah-on)"/><text class="t-muted" x="20.0" y="322.0" text-anchor="start">A token reads only its chosen experts (filled boxes). The tier that holds each one sets how long the read takes.</text></svg><figcaption>Experts spread over three memory tiers. Attention, the shared expert, the KV cache and the most-used experts sit in GPU memory; most routed experts sit in RAM; cold experts or Engram tables can stay on an SSD. A token reads only its chosen experts.</figcaption></figure>

The always-used part of V3 belongs on the counter. From the script: at Q4_K_M, everything except the routed experts is about 10.3 GB. Add a KV cache and buffers, and you get close to the "at least 14GB VRAM" that KTransformers asks for. The routed experts, 394 GB at Q4_K_M, go to RAM. For one token, the CPU reads about 12.3 GB of them.

The SSD tier is the newest. V4.1-Flash's Engram tables are made for it: the model reads only a few rows per token, so llama.cpp builds can map them from disk. Unsloth's guide uses `--load-mode mmap --lazy-mode on` for this. The same idea can be pushed onto experts, with a cache of recently used experts in RAM and the rest on an SSD. The catch is speed. An SSD delivers a few GB per second, far below RAM, so every miss costs a long wait.

> [!TIP] In production
> This repository includes a small local tool, the **MoE viewer** (`code/deepseek/moe-viewer`). It runs a real MoE model (DeepSeek-V2-Lite-Chat, 64 experts and top-6 per layer, or OLMoE) and records which experts each token picks. It then replays those real picks through a pretend machine with a GPU tier for the most-used experts, a least-recently-used RAM cache and an SSD. Sliders set the tier sizes and the SSD speed. The tiers are simulated, the expert picks are real, and the page says so.

## Running on consumer machines

### llama.cpp and GGUF files

> [!DEFINITION] llama.cpp
> An open-source engine in C and C++ that runs GGUF models on CPUs, Apple GPUs and NVIDIA or AMD GPUs. It can keep some layers on the GPU and the rest in system RAM. Ollama and several desktop apps build on it.

llama.cpp has run DeepSeek V2 and V3 for a long time. Proper MLA support, with the small cache, arrived in pull request #12801, merged on 15 April 2025. V4 support landed in pull request #24162 on 29 June 2026. For V4.1-Flash, Unsloth's guide says to build from pull request #30817, which was closed without merging. As of 10 November 2026 we could not confirm V4.1 support in the main llama.cpp branch.

For the 1.58-bit R1 file, Unsloth reported (January 2025) "around 140 tokens per second for throughput and 14 tokens/s for single user inference" on two H100 80 GB GPUs. Unsloth staff wrote that one 24 GB RTX 4090 with the rest in RAM gives "at least 2-3 tokens/s". We read that second claim through a mirror, so treat it as unconfirmed.

### KTransformers: experts on the CPU, the rest on the GPU

> [!DEFINITION] KTransformers
> An open-source engine from the kvcache-ai project, built for exactly the tiers above. Attention, the shared expert and the KV cache run on one GPU. The routed experts run on the CPU with fast vector instructions (AVX-512 and Intel AMX).

Its tutorial for V3 and R1 at Q4_K_M uses two Intel Xeon Gold 6454S CPUs with 32 cores each. The memory is DDR5-4800 with 8 channels per socket, and the GPU is one RTX 4090D with 24 GB. With one socket it needs 382 GB of RAM and at least 14 GB of GPU memory. With two sockets it needs 1 TB, because it copies the weights to both sockets' memory. The reported numbers, all from the KTransformers team for a 500-token prompt:

| Setup (V3/R1 Q4_K_M) | Prefill tok/s | Decode tok/s |
|---|---|---|
| llama.cpp, same 2 x 32 cores | 10.31 | 4.51 |
| KTransformers V0.2, 2 sockets, 8 experts | 82.94 | 12.21 |
| KTransformers V0.2, 2 sockets, 6 experts | 97.32 | 13.69 |
| KTransformers V0.3 preview (AMX), 8 / 6 experts | 255.26 / 286.55 | not given |
| KTransformers V0.2.1, 6 experts, short / 8K prompt | not given | 16.8 / 13.9 |

> [!WARNING]
> "6 experts" means the engine uses 6 experts per token instead of the 8 the model was trained with. That is a change to the model's computation, a trade of some quality for speed. Compare 8-expert numbers with 8-expert numbers.

KTransformers also runs V4-Flash on one RTX 5090 (32 GB) with an x86 CPU and at least 200 GB of RAM, and reports "20+ tok/s" decode. On a single Huawei Ascend die (61.3 GB) with 229 GB of host RAM, it reports 18.32 to 20.48 tokens per second. It notes that "Decode is host-memory-bandwidth bound". For R1-0528 with SGLang on a server with 8 L20 GPUs and a Xeon, it reports 87.58 output tokens per second at 8 parallel requests.

### Apple Silicon

A Mac with Apple Silicon has one pool of **unified memory** shared by the CPU and GPU. A Mac Studio with an M3 Ultra can have up to 512 GB of it, at "over 800GB/s". Apple's own launch text says it can run "LLMs with over 600 billion parameters entirely in memory".

The reported speeds are good for short prompts and much worse for long ones. A reviewer measured R1 at 4 bits at "approximately 17-18 tokens per second". An independent llama.cpp-family test of V3-0324 Q4_K_M measured these numbers:

| Prompt length | Prompt reading (prefill) | Writing (decode) |
|---|---|---|
| 4.8K tokens | 64.02 tok/s, about 75 s | 10.89 tok/s |
| 12.7K tokens | 58.82 tok/s, about 216 s | 6.67 tok/s |

The waiting times are our arithmetic: 4,800 / 64.02 and 12,700 / 58.82. They show the weak spot. A Mac has the bandwidth for decode but much less raw compute than a data-centre GPU. So a long document takes minutes before the first word appears. Short demos hide this.

For V4-Flash, the project DwarfStar (`ds4`, by antirez) publishes its own Metal and CUDA measurements of a 2-bit file:

| Machine | Context | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| M5 Max, 128 GB | 2,048 | 790.18 | 39.35 |
| M5 Max, 128 GB | 65,536 | 398.50 | 27.64 |
| DGX Spark, 128 GB | 2,048 | 825.76 | 18.05 |
| DGX Spark, 128 GB | 65,536 | 822.98 | 13.84 |

V4-Flash has 13B active weights against V3's 37B, so a 128 GB laptop decodes it faster than a 512 GB Mac Studio decodes V3. For V4.1-Flash, ds4 says the 2-bit file runs on one 128 GB Mac or Spark with SSD streaming, and "Q4 needs SSD streaming or a 512 GB Mac". It publishes no speed for V4.1.

<figure class="fig"><svg viewBox="0 0 760 446" role="img" aria-label="Reported decode speeds of DeepSeek models on home and workstation hardware, from 2-3 to about 39 tokens per second."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Reported decode speeds outside the data centre</text><text class="t-tick" x="290.0" y="61.0" text-anchor="end">V4-Flash Q2, M5 Max 128 GB, 2K ctx</text><g class="mark"><title>V4-Flash Q2, M5 Max 128 GB, 2K ctx: 39.35 (ds4)</title><rect class="s2" x="300" y="44" width="323.5" height="24" rx="4"/></g><text class="t-val" x="631.5" y="61.0" text-anchor="start">39.35 (ds4)</text><text class="t-tick" x="290.0" y="95.0" text-anchor="end">V4-Flash Q2, M5 Max 128 GB, 65K ctx</text><g class="mark"><title>V4-Flash Q2, M5 Max 128 GB, 65K ctx: 27.64 (ds4)</title><rect class="s2" x="300" y="78" width="227.3" height="24" rx="4"/></g><text class="t-val" x="535.3" y="95.0" text-anchor="start">27.64 (ds4)</text><text class="t-tick" x="290.0" y="129.0" text-anchor="end">V4-Flash, 1x RTX 5090 + CPU</text><g class="mark"><title>V4-Flash, 1x RTX 5090 + CPU: "20+" (KTransformers)</title><rect class="s2" x="300" y="112" width="164.4" height="24" rx="4"/></g><text class="t-val" x="472.4" y="129.0" text-anchor="start">"20+" (KTransformers)</text><text class="t-tick" x="290.0" y="163.0" text-anchor="end">V4-Flash Q2, DGX Spark, 2K ctx</text><g class="mark"><title>V4-Flash Q2, DGX Spark, 2K ctx: 18.05 (ds4)</title><rect class="s2" x="300" y="146" width="148.4" height="24" rx="4"/></g><text class="t-val" x="456.4" y="163.0" text-anchor="start">18.05 (ds4)</text><text class="t-tick" x="290.0" y="197.0" text-anchor="end">R1 4-bit, M3 Ultra 512 GB</text><g class="mark"><title>R1 4-bit, M3 Ultra 512 GB: about 17-18 (review)</title><rect class="s1" x="300" y="180" width="143.9" height="24" rx="4"/></g><text class="t-val" x="451.9" y="197.0" text-anchor="start">about 17-18 (review)</text><text class="t-tick" x="290.0" y="231.0" text-anchor="end">V3 Q4_K_M, 2x Xeon + 4090D, 6 experts</text><g class="mark"><title>V3 Q4_K_M, 2x Xeon + 4090D, 6 experts: 13.69 (KTransformers)</title><rect class="s1" x="300" y="214" width="112.6" height="24" rx="4"/></g><text class="t-val" x="420.6" y="231.0" text-anchor="start">13.69 (KTransformers)</text><text class="t-tick" x="290.0" y="265.0" text-anchor="end">V3 Q4_K_M, 2x Xeon + 4090D, 8 experts</text><g class="mark"><title>V3 Q4_K_M, 2x Xeon + 4090D, 8 experts: 12.21 (KTransformers)</title><rect class="s1" x="300" y="248" width="100.4" height="24" rx="4"/></g><text class="t-val" x="408.4" y="265.0" text-anchor="start">12.21 (KTransformers)</text><text class="t-tick" x="290.0" y="299.0" text-anchor="end">V3-0324 Q4_K_M, M3 Ultra, 4.8K prompt</text><g class="mark"><title>V3-0324 Q4_K_M, M3 Ultra, 4.8K prompt: 10.89 (third party)</title><rect class="s1" x="300" y="282" width="89.5" height="24" rx="4"/></g><text class="t-val" x="397.5" y="299.0" text-anchor="start">10.89 (third party)</text><text class="t-tick" x="290.0" y="333.0" text-anchor="end">V3 Q4_K_M, same Xeon box, llama.cpp</text><g class="mark"><title>V3 Q4_K_M, same Xeon box, llama.cpp: 4.51 (KTransformers)</title><rect class="s3" x="300" y="316" width="37.1" height="24" rx="4"/></g><text class="t-val" x="345.1" y="333.0" text-anchor="start">4.51 (KTransformers)</text><text class="t-tick" x="290.0" y="367.0" text-anchor="end">R1 1.58-bit, one RTX 4090 + RAM</text><g class="mark"><title>R1 1.58-bit, one RTX 4090 + RAM: "2-3" (Unsloth staff)</title><rect class="s3" x="300" y="350" width="20.6" height="24" rx="4"/></g><text class="t-val" x="328.6" y="367.0" text-anchor="start">"2-3" (Unsloth staff)</text><line class="axis" x1="300" y1="379.0" x2="670" y2="379.0"/><text class="t-muted" x="300.0" y="418.0" text-anchor="start">output tokens per second, one user; source in brackets</text><text class="t-muted" x="300.0" y="436.0" text-anchor="start">Top four: V4-Flash (13B active). Rest: V3/R1 (37B active). Each from a different test.</text></svg><figcaption>Reported single-user decode speeds outside the data centre, from different tests and sources. V4-Flash (13B active) runs at 18 to 39 tokens per second on 128 GB machines; the 671B V3/R1 runs at about 11 to 18 on an M3 Ultra or a CPU server with one GPU.</figcaption></figure>

### Ollama for the distills

> [!DEFINITION] Ollama
> A desktop tool that downloads and runs GGUF models with one command, such as `ollama run deepseek-r1:7b`. It uses llama.cpp inside.

The R1 distills are the practical way to try DeepSeek-style reasoning on a laptop. Ollama's default tags are 4-bit files. The 1.5b tag is 1.1 GB, 7b is 4.7 GB and 8b is 5.2 GB. The 14b tag is 9.0 GB, 32b is 20 GB and 70b is 43 GB. The full `deepseek-r1:671b` tag is 404 GB.

> [!WARNING]
> `ollama run deepseek-r1` (the `latest` tag) and `deepseek-r1:8b` now pull DeepSeek-R1-0528-Qwen3-8B, not the original Llama-8B distill. Say which model you ran. V4-Pro and V4.1-Flash exist on Ollama only as `:cloud` tags, which run on Ollama's servers, not on your machine.

The fits table gives a rough rule. A 16 GB laptop runs the 7B and 8B distills well, and a 24 GB GPU or a 32 GB Mac fits the 14B. The 32B at 4 bits needs about 28 GB with a 32K cache. The 70B needs 64 GB or more. We found no single trustworthy speed table for laptops, so measure your own.

## How DeepSeek serves its own models

### One model, two kinds of machine

A chat service does two different jobs. Reading a long prompt is one big parallel computation. Writing the answer is a long series of small steps, one token at a time, for thousands of users. On the same GPU, the two jobs get in each other's way: a big prefill pauses everybody's decode.

> [!DEFINITION] Prefill/decode disaggregation
> Running prefill and decode on different groups of machines. The prefill group reads a prompt, builds its KV cache, and sends that cache to the decode group, which writes the answer. Each group gets the parallel layout that suits its job.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §3.4 · page 18
> [![The start of section 3.4 of the V3 paper with deploy DeepSeek-V3 on the H800 cluster and separates the prefilling and decoding stages highlighted](/img/deepseek/papers/p7-v3-deploy.png)](/img/deepseek/papers/p7-v3-deploy.png)
>
> **What it says:** DeepSeek runs V3 "on the H800 cluster", with NVLink inside each node and InfiniBand between nodes, and uses a strategy that "separates the prefilling and decoding stages".
>
> **What it means:**
>
> The paper gives the reason in one phrase. It wants to meet a service-level objective (a promised response speed) and get high throughput at the same time. A restaurant does the same with a prep kitchen and a line kitchen. The prep cooks chop in big batches; the line cooks plate one dish at a time without being interrupted.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §3.4.1 · page 19
> [![Section 3.4.1 of the V3 paper: 4 nodes with 32 GPUs, 32-way Expert Parallelism (EP32), redundant experts adjusted every 10 minutes, and one additional redundant expert per GPU highlighted](/img/deepseek/papers/p7-v3-prefill.png)](/img/deepseek/papers/p7-v3-prefill.png)
>
> **What it says:** the smallest prefill unit is "4 nodes with 32 GPUs". Attention uses TP4 with DP8; the experts use "32-way Expert Parallelism (EP32)". DeepSeek "duplicates high-load experts", picks them from live statistics "every 10 minutes", and each GPU "will also host one additional redundant expert".
>
> **What it means:**
>
> Experts are not equally popular. If one GPU holds a popular expert, it gets more tokens than the others, and everybody waits for it. The fix is to copy the popular experts. With 32 spare slots, 256 + 32 = 288 experts spread as 9 per GPU over 32 GPUs.
>
> "Every 10 minutes" tells you the popularity moves. Code, maths and chat use different experts, and the mix of users changes through the day. The open-source EPLB library, described below, is the planner that decides which experts to copy.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §3.4.2 · page 19
> [![Section 3.4.2 of the V3 paper with 40 nodes with 320 GPUs, EP320, each GPU hosts only one expert and direct point-to-point transfers over IB highlighted](/img/deepseek/papers/p7-v3-decode.png)](/img/deepseek/papers/p7-v3-decode.png)
>
> **What it says:** in the paper, the smallest decode unit is "40 nodes with 320 GPUs", with "EP320". "Each GPU hosts only one expert", and 64 GPUs hold the redundant and shared experts. Dispatch and combine use "direct point-to-point transfers over IB".
>
> **What it means:**
>
> Decode is the opposite of prefill. Each step handles one token per user, so each expert sees only a few tokens. Spreading experts as widely as possible means each GPU reads only one expert's weights per step, which keeps every step short.
>
> Two months later, DeepSeek published the production layout, and it was smaller. The script below checks both.

### The February 2025 overview

On the last day of its "Open Source Week", DeepSeek published an overview of its V3/R1 serving system in the `open-infra-index` repository. It is a GitHub page, so we quote its text instead of a screenshot:

> [!PAPER] DeepSeek-AI (2025), Day 6: DeepSeek-V3/R1 Inference System Overview · System Design Principles
> **What it says:** "Prefilling Phase [Routed Expert EP32, MLA/Shared Expert DP32]: Each deployment unit spans 4 nodes with 32 redundant routed experts, where each GPU handles 9 routed experts and 1 shared expert." And: "Decoding Phase [Routed Expert EP144, MLA/Shared Expert DP144]: Each deployment unit spans 18 nodes with 32 redundant routed experts, where each GPU manages 2 routed experts and 1 shared expert."
>
> **What it means:**
>
> Production chose 144 decode GPUs with 2 experts each, not the paper's 320 GPUs with one. A smaller unit is easier to fill with enough users and easier to replace when a node fails. The attention part runs as pure DP on every GPU, so each GPU serves its own set of conversations.
>
> The overview also explains how both stages hide communication. Each batch is split into two micro-batches. While one micro-batch computes, the other one sends or receives tokens. In decode, the attention layer is split into two steps to make a 5-stage pipeline.

```text
== 3. Where the 256 routed experts go ==
V3 paper, prefill unit (4 nodes)  : 256 + 32 redundant = 288 = 32 GPUs x 9 routed experts
Feb 2025, prefill unit (4 nodes)  : 256 + 32 redundant = 288 = 32 GPUs x 9 routed experts
Feb 2025, decode unit (18 nodes)  : 256 + 32 redundant = 288 = 144 GPUs x 2 routed experts
V3 paper, decode unit (40 nodes)  : 320 GPUs, one expert each: 256 routed + 64 for redundant and shared experts = 320
```

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="A prefill unit of 32 GPUs with 9 experts each sends the KV cache to a decode unit of 144 GPUs with 2 experts each."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">DeepSeek's serving split (Feb 2025, V3/R1 on H800)</text><rect class="box-1" x="20.0" y="50.0" width="300.0" height="210.0" rx="10"/><text class="t-note" x="36.0" y="74.0" text-anchor="start">Prefill unit: 4 nodes, 32 GPUs</text><text class="t-muted" x="36.0" y="94.0" text-anchor="start">reads the whole prompt at once</text><rect class="box" x="36.0" y="110.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="50.0" y="126.0" text-anchor="middle">9</text><rect class="box" x="70.0" y="110.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="84.0" y="126.0" text-anchor="middle">9</text><rect class="box" x="104.0" y="110.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="118.0" y="126.0" text-anchor="middle">9</text><rect class="box" x="138.0" y="110.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="152.0" y="126.0" text-anchor="middle">9</text><rect class="box" x="172.0" y="110.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="186.0" y="126.0" text-anchor="middle">9</text><rect class="box" x="206.0" y="110.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="220.0" y="126.0" text-anchor="middle">9</text><rect class="box" x="240.0" y="110.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="254.0" y="126.0" text-anchor="middle">9</text><rect class="box" x="274.0" y="110.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="288.0" y="126.0" text-anchor="middle">9</text><rect class="box" x="36.0" y="140.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="50.0" y="156.0" text-anchor="middle">9</text><rect class="box" x="70.0" y="140.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="84.0" y="156.0" text-anchor="middle">9</text><rect class="box" x="104.0" y="140.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="118.0" y="156.0" text-anchor="middle">9</text><rect class="box" x="138.0" y="140.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="152.0" y="156.0" text-anchor="middle">9</text><rect class="box" x="172.0" y="140.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="186.0" y="156.0" text-anchor="middle">9</text><rect class="box" x="206.0" y="140.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="220.0" y="156.0" text-anchor="middle">9</text><rect class="box" x="240.0" y="140.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="254.0" y="156.0" text-anchor="middle">9</text><rect class="box" x="274.0" y="140.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="288.0" y="156.0" text-anchor="middle">9</text><rect class="box" x="36.0" y="170.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="50.0" y="186.0" text-anchor="middle">9</text><rect class="box" x="70.0" y="170.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="84.0" y="186.0" text-anchor="middle">9</text><rect class="box" x="104.0" y="170.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="118.0" y="186.0" text-anchor="middle">9</text><rect class="box" x="138.0" y="170.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="152.0" y="186.0" text-anchor="middle">9</text><rect class="box" x="172.0" y="170.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="186.0" y="186.0" text-anchor="middle">9</text><rect class="box" x="206.0" y="170.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="220.0" y="186.0" text-anchor="middle">9</text><rect class="box" x="240.0" y="170.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="254.0" y="186.0" text-anchor="middle">9</text><rect class="box" x="274.0" y="170.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="288.0" y="186.0" text-anchor="middle">9</text><rect class="box" x="36.0" y="200.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="50.0" y="216.0" text-anchor="middle">9</text><rect class="box" x="70.0" y="200.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="84.0" y="216.0" text-anchor="middle">9</text><rect class="box" x="104.0" y="200.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="118.0" y="216.0" text-anchor="middle">9</text><rect class="box" x="138.0" y="200.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="152.0" y="216.0" text-anchor="middle">9</text><rect class="box" x="172.0" y="200.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="186.0" y="216.0" text-anchor="middle">9</text><rect class="box" x="206.0" y="200.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="220.0" y="216.0" text-anchor="middle">9</text><rect class="box" x="240.0" y="200.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="254.0" y="216.0" text-anchor="middle">9</text><rect class="box" x="274.0" y="200.0" width="28.0" height="24.0" rx="3"/><text class="t-tick" x="288.0" y="216.0" text-anchor="middle">9</text><text class="t-tick" x="36.0" y="250.0" text-anchor="start">each GPU: 9 routed + 1 shared expert (EP32)</text><line class="path" x1="328.0" y1="155.0" x2="412.0" y2="155.0" marker-end="url(#ah-on)"/><text class="t-tick" x="370.0" y="142.0" text-anchor="middle">KV cache</text><rect class="box-2" x="420.0" y="50.0" width="320.0" height="210.0" rx="10"/><text class="t-note" x="436.0" y="74.0" text-anchor="start">Decode unit: 18 nodes, 144 GPUs</text><text class="t-muted" x="436.0" y="94.0" text-anchor="start">writes one token per step for every user</text><rect class="box" x="436.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="448.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="460.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="472.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="484.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="496.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="508.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="520.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="532.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="544.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="556.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="568.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="580.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="592.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="604.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="616.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="628.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="640.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="652.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="664.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="676.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="688.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="700.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="712.0" y="110.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="436.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="448.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="460.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="472.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="484.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="496.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="508.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="520.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="532.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="544.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="556.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="568.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="580.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="592.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="604.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="616.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="628.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="640.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="652.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="664.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="676.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="688.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="700.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="712.0" y="130.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="436.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="448.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="460.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="472.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="484.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="496.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="508.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="520.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="532.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="544.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="556.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="568.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="580.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="592.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="604.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="616.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="628.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="640.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="652.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="664.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="676.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="688.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="700.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="712.0" y="150.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="436.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="448.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="460.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="472.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="484.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="496.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="508.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="520.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="532.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="544.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="556.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="568.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="580.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="592.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="604.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="616.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="628.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="640.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="652.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="664.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="676.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="688.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="700.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="712.0" y="170.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="436.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="448.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="460.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="472.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="484.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="496.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="508.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="520.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="532.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="544.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="556.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="568.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="580.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="592.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="604.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="616.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="628.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="640.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="652.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="664.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="676.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="688.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="700.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="712.0" y="190.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="436.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="448.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="460.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="472.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="484.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="496.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="508.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="520.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="532.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="544.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="556.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="568.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="580.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="592.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="604.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="616.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="628.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="640.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="652.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="664.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="676.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="688.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="700.0" y="210.0" width="9.0" height="15.0" rx="2"/><rect class="box" x="712.0" y="210.0" width="9.0" height="15.0" rx="2"/><text class="t-tick" x="436.0" y="250.0" text-anchor="start">each GPU: 2 routed + 1 shared expert (EP144)</text><text class="t-muted" x="20.0" y="286.0" text-anchor="start">Both units: 256 routed experts + 32 redundant copies = 288 slots (32 x 9 = 144 x 2).</text></svg><figcaption>DeepSeek's production split in February 2025: a 32-GPU prefill unit with 9 routed experts per GPU hands the KV cache to a 144-GPU decode unit with 2 routed experts per GPU.</figcaption></figure>

Precision follows training: matrix multiplications and dispatch in FP8, core MLA and combine in BF16. And the fleet breathes with demand. DeepSeek runs inference on all nodes during the day and moves nodes to research and training at night.

### How fast can expert parallelism be?

The ISCA 2025 paper does a back-of-the-envelope limit for this design. Each GPU holds one expert and handles about 32 tokens per step. Each token is sent to 9 experts (8 routed and 1 shared) in FP8 (1 byte per number) and returned in BF16 (2 bytes). The hidden size is about 7,000 numbers.

> [!PAPER] Zhao et al. (2025), Insights into DeepSeek-V3 (ISCA 2025) · §2.3.2 · page 5
> [![The ISCA paper's speed-limit calculation: 120.96 microseconds of communication, 241.92 microseconds per layer, 14.76 ms over 61 layers, with 14.76 ms TPOT, equivalent to 67 tokens per second highlighted](/img/deepseek/papers/p7-isca-limit.png)](/img/deepseek/papers/p7-isca-limit.png)
>
> **What it says:** one dispatch-and-combine takes $$(1 + 2) \times 32 \times 9 \times 7\text{K} / 50\ \text{GB/s} = 120.96$$ microseconds on 400 Gbps InfiniBand. Two per layer and 61 layers give "14.76 ms TPOT, equivalent to 67 tokens per second".
>
> **What it means:**
>
> Even with infinitely fast GPUs, a token cannot come out faster than the wires can carry it between experts. With 50 GB/s per GPU, that limit is about 67 tokens per second for one user. Real systems are slower, because the GPUs also compute and the network never runs at full speed.
>
> The authors then redo the sum for NVIDIA's GB200 NVL72, where 72 GPUs share a 900 GB/s fabric. The limit moves to more than 1,000 tokens per second. This is DeepSeek telling hardware makers, in numbers, that the wire between GPUs is now the bottleneck.

```text
== 2. Expert-parallel communication limit (ISCA 2025, section 2.3.2) ==
CX7 400Gbps InfiniBand, 50 GB/s   : one all-to-all pair  120.96 us, x2 per layer x 61 layers =  14.76 ms ->    67.8 tok/s
GB200 NVL72, 900 GB/s             : one all-to-all pair    6.72 us, x2 per layer x 61 layers =   0.82 ms ->  1219.8 tok/s
```

The paper rounds 67.8 down to 67. In production, DeepSeek's users got an average of 20 to 22 tokens per second, about a third of this ideal.

### One day of numbers, and the 545%

The overview ends with 24 hours of statistics, from 12:00 on 27 February 2025 to 12:00 the next day, Beijing time. V3 and R1 together peaked at 278 nodes and averaged 226.75 nodes of 8 H800s. 56.3% of input tokens hit DeepSeek's on-disk KV cache, so their prefill was skipped. Each node delivered about 73.7k input tokens per second in prefill (counting cache hits) or about 14.8k output tokens per second in decode.

Then it does a famous sum. If every token had been billed at R1's API prices, the day's revenue would have been $562,027 against a cost of $87,072. The overview calls this a "cost profit margin of 545%".

$$
\text{margin} = \frac{R - C}{C}
$$

$$R$$ is the theoretical revenue for the day and $$C$$ is the cost. A tiny example: a lemonade stand that spends $10 and takes $30 has a margin of $$(30 - 10)/10 = 200\%$$. The real inputs, recomputed:

```text
== 4. Feb 27-28 2025, 24 hours of V3 + R1 on H800 ==
cost    = 226.75 nodes x 8 GPUs x 24 h x $2 = $87,072
revenue = 342B hit x $0.14/M + 266B miss x $0.55/M + 168B out x $2.19/M
        = $47,880 + $146,300 + $367,920 = $562,100  (reported: $562,027)
margin  = (562,027 - 87,072) / 87,072 = 545%
cache-hit share of input: 56.25%
output share of the theoretical revenue: 65%
decode: 14.8k output tok/s per node = 1,850 per H800; at 20-22 tok/s per user, 673 to 740 streams per node
```

Our revenue is $73 higher than DeepSeek's because the token counts in the overview are rounded to billions. The 545% follows exactly from the published inputs.

<figure class="fig"><svg viewBox="0 0 760 236" role="img" aria-label="Bar chart: a daily cost of 87,072 dollars against a theoretical revenue of 562,027 dollars split into cache-hit input, cache-miss input and output."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">One day of V3 + R1 (27-28 Feb 2025): cost versus theoretical revenue</text><text class="t-tick" x="150.0" y="76.0" text-anchor="end">cost</text><rect class="s3" x="160" y="58" width="75.5" height="30" rx="4"/><text class="t-val" x="243.5" y="78.0" text-anchor="start">$87,072 (226.75 nodes x 8 x 24 h x $2)</text><text class="t-tick" x="150.0" y="136.0" text-anchor="end">if all billed</text><text class="t-tick" x="150.0" y="152.0" text-anchor="end">at R1 prices</text><g class="mark"><title>cache-hit input: $47,880</title><rect class="s1" x="160.0" y="120" width="41.5" height="30" rx="2"/></g><g class="mark"><title>cache-miss input: $146,300</title><rect class="s2" x="201.5" y="120" width="126.8" height="30" rx="2"/></g><g class="mark"><title>output: $367,920</title><rect class="s3" x="328.3" y="120" width="318.9" height="30" rx="2"/></g><rect class="s1" x="160" y="166" width="12" height="12" rx="2"/><text class="t-tick" x="178.0" y="177.0" text-anchor="start">cache-hit input: $47,880</text><rect class="s2" x="340" y="166" width="12" height="12" rx="2"/><text class="t-tick" x="358.0" y="177.0" text-anchor="start">cache-miss input: $146,300</text><rect class="s3" x="520" y="166" width="12" height="12" rx="2"/><text class="t-tick" x="538.0" y="177.0" text-anchor="start">output: $367,920</text><text class="t-val" x="655.2" y="140.0" text-anchor="start">$562,027</text><text class="t-muted" x="20.0" y="222.0" text-anchor="start">(562,027 - 87,072) / 87,072 = 5.45, the "545%". Real revenue is lower: web and app are free, V3 is cheaper, night discounts.</text></svg><figcaption>The 545% sum: a daily cost of $87,072 against a theoretical revenue of $562,027, of which $367,920 would come from output tokens at R1's price.</figcaption></figure>

> [!WARNING]
> The 545% is a theoretical number, and DeepSeek says so. Its own caveats: V3 is priced well below R1, web and app use are free, and night-time discounts apply, so "actual revenue is substantially lower". Our own caveats: $2 per GPU-hour is an assumed rental price, not DeepSeek's real cost. And the sum leaves out training, research, staff, storage and the network. Read it as "serving at this scale is cheap per token", not as a company profit.

For anyone planning a service, the most useful number is per GPU. Decode gave about 1,850 output tokens per second per H800, while each user still saw 20 to 22 tokens per second. That is roughly 700 conversations being written at once on every 8-GPU node.

Others have rebuilt this design in the open. The SGLang team ran V3 on 96 H100 GPUs with prefill/decode disaggregation and large EP. It reported 52.3k input and 22.3k output tokens per second per node for 2,000-token inputs. That is up to 5 times the output throughput of plain TP16 on the same hardware. The vLLM team reported "2.2k tokens/s per H200 GPU" in multi-node wide-EP deployments in December 2025. Both are engine-team claims.

### V4 and V4.1: the cache moves to disk

The V4 report says DeepSeek's inference framework "largely inherits" from V3's, with changes to KV cache management. The big change is that the cache now lives on disk between turns.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence · §3.5.2 · page 23
> [![Section 3.5.2 of the V4 report with on-disk KV cache storage mechanism to eliminate repeated prefilling for shared-prefix requests and approximately 8 times larger highlighted](/img/deepseek/papers/p7-v4-ondisk.png)](/img/deepseek/papers/p7-v4-ondisk.png)
>
> **What it says:** V4 uses "an on-disk KV cache storage mechanism to eliminate repeated prefilling for shared-prefix requests". Compressed CSA and HCA entries go to disk whole. The sliding-window entries are "approximately 8 times larger", so they get three separate strategies.
>
> **What it means:**
>
> In a chat, each new message resends the whole conversation so far. Without a stored cache, the server would read it all again. With the cache on disk, it reads only the new message. The 56.3% cache-hit rate in February 2025 shows how much work that saves.
>
> V4's compressed cache is small enough to store cheaply. The awkward part is the uncompressed sliding window of recent tokens, which is 8 times larger than the compressed entries. Storing it all wastes SSD space; storing none of it means recomputing.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression · §3.2.1 · page 19
> [![Section 3.2.1 of the V4.1 report with Encoder-Prefill-Decode (EPD) disaggregation, reduces the persistent KV cache footprint to 1/8 and remain resident for over 72 hours highlighted](/img/deepseek/papers/p7-v41-persistent.png)](/img/deepseek/papers/p7-v41-persistent.png)
>
> **What it says:** V4.1 adds "Encoder–Prefill–Decode (EPD) disaggregation", with vision encoding as its own stage. It "reduces the persistent KV cache footprint to 1/8" of V4's. DeepSeek sized the SSD cache so that cached conversations "remain resident for over 72 hours".
>
> **What it means:**
>
> Two changes multiply. The sliding-window entries leave the SSD cache, which almost halves it. They move to a short-lived pool in each machine's RAM, about 10% of host memory, kept for minutes. The global entries also shrink to a quarter, through the V4.1 architecture and FP4. Half times a quarter gives the 1/8.
>
> The 72 hours is a design target. A user can come back to a long conversation three days later, and the server still skips the prefill. That is a cost decision as much as a speed one.

### DeepInfer: the production engine, opened

In April 2025, DeepSeek wrote that its internal engine was "an early fork of vLLM" and that it would not release it. It would contribute pieces to vLLM and SGLang instead. In October 2026 it changed course and released **DeepInfer** under the Apache 2.0 license. The README calls it "the high-performance inference framework that powers DeepSeek's large-scale production services".

The open release is limited. It supports one model, V4.1-Flash, and one mode, offline inference. It runs on NVIDIA SM100 GPUs and Huawei Ascend 950 NPUs. It includes attention DP, MoE EP with EPLB, DSpark speculation, "SWA bounded replay" and batch-invariant inference. The README lists what a complete service still needs and the release leaves out: "A KV cache system", "VE/Prefill/Decode disaggregation", and load balancing across replicas. Its benchmark charts were measured on 4 SM100 GPUs for prefill and 64 for decode.

## The open infrastructure libraries

DeepSeek released the building blocks of its serving and training stack during Open Source Week in February 2025, and kept updating them. Each solves one bottleneck from this part. All numbers below are DeepSeek's own.

| Library | What it does | Reported numbers |
|---|---|---|
| FlashMLA | Attention kernels for MLA, DSA, V4 and V4.1 caches | Feb 2025: 3,000 GB/s memory-bound, 580 TFLOPS compute-bound on H800; Apr 2025: up to 660 TFLOPS; Sep 2025 sparse kernels: 640 TFLOPS prefill, 410 decode on H800; Sep 2026 V4.1 on B200: 1,350 sparse prefill, 1,024 sparse decode; Ascend 950: 410 and 360 TFLOPS |
| DeepEP | Dispatch and combine for expert parallelism over NVLink and RDMA | normal kernels 153 to 158 GB/s inside a node, 43 to 58 GB/s across nodes; low-latency dispatch 163 to 194 us from EP8 to EP256 |
| DeepGEMM | FP8 (later FP4) matrix multiplication for dense and MoE layers | "1350+ FP8 TFLOPS" at launch, "up to 1550 TFLOPS on H800" in April 2025; Mega MoE kernel (Apr 2026) |
| EPLB | Plans which experts to copy and where to put them | two policies: hierarchical for prefill, global for decode |
| 3FS | Distributed file system; holds the on-disk KV cache | 6.6 TiB/s aggregate read on 180 nodes; "40+ GiB/s peak throughput per client node for KVCache lookup" |

> [!DEFINITION] RDMA
> Remote direct memory access. One machine's network card writes straight into another machine's memory, without the other CPU's help. InfiniBand uses it; DeepEP depends on it for low latency.

FlashMLA's 2026 updates come with a warning for older models. The main branch dropped Hopper GPUs (H100, H800, H200) and the V3, V3.2 and V4.0 kernels. Those live on at commit `ba89a34`.

The DeepEP numbers explain how DeepSeek could afford 144-way expert parallelism in decode.

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="DeepEP low-latency kernels: dispatch latency grows from 163 to 194 microseconds and combine from 318 to 360 as EP grows from 8 to 256."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="610" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0</text><line class="grid" x1="64" y1="195.0" x2="610" y2="195.0"/><text class="t-tick" x="56.0" y="199.0" text-anchor="end">100</text><line class="grid" x1="64" y1="140.0" x2="610" y2="140.0"/><text class="t-tick" x="56.0" y="144.0" text-anchor="end">200</text><line class="grid" x1="64" y1="85.0" x2="610" y2="85.0"/><text class="t-tick" x="56.0" y="89.0" text-anchor="end">300</text><line class="grid" x1="64" y1="30.0" x2="610" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">400</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">8</text><text class="t-tick" x="173.2" y="270.0" text-anchor="middle">16</text><text class="t-tick" x="282.4" y="270.0" text-anchor="middle">32</text><text class="t-tick" x="391.6" y="270.0" text-anchor="middle">64</text><text class="t-tick" x="500.8" y="270.0" text-anchor="middle">128</text><text class="t-tick" x="610.0" y="270.0" text-anchor="middle">256</text><line class="axis" x1="64" y1="250" x2="610" y2="250"/><polyline class="l2" points="64.0,75.1 173.2,69.1 282.4,57.5 391.6,55.9 500.8,47.1 610.0,52.0"/><g class="mark"><title>combine, 8: 318</title><circle class="s2 ring" cx="64.0" cy="75.1" r="5"/></g><g class="mark"><title>combine, 16: 329</title><circle class="s2 ring" cx="173.2" cy="69.1" r="5"/></g><g class="mark"><title>combine, 32: 350</title><circle class="s2 ring" cx="282.4" cy="57.5" r="5"/></g><g class="mark"><title>combine, 64: 353</title><circle class="s2 ring" cx="391.6" cy="55.9" r="5"/></g><g class="mark"><title>combine, 128: 369</title><circle class="s2 ring" cx="500.8" cy="47.1" r="5"/></g><g class="mark"><title>combine, 256: 360</title><circle class="s2 ring" cx="610.0" cy="52.0" r="5"/></g><polyline class="l1" points="64.0,160.4 173.2,154.8 282.4,149.9 391.6,147.7 500.8,144.4 610.0,143.3"/><g class="mark"><title>dispatch, 8: 163</title><circle class="s1 ring" cx="64.0" cy="160.4" r="5"/></g><g class="mark"><title>dispatch, 16: 173</title><circle class="s1 ring" cx="173.2" cy="154.8" r="5"/></g><g class="mark"><title>dispatch, 32: 182</title><circle class="s1 ring" cx="282.4" cy="149.9" r="5"/></g><g class="mark"><title>dispatch, 64: 186</title><circle class="s1 ring" cx="391.6" cy="147.7" r="5"/></g><g class="mark"><title>dispatch, 128: 192</title><circle class="s1 ring" cx="500.8" cy="144.4" r="5"/></g><g class="mark"><title>dispatch, 256: 194</title><circle class="s1 ring" cx="610.0" cy="143.3" r="5"/></g><line class="grid" x1="616.0" y1="52.0" x2="626.0" y2="52.0"/><rect class="s2" x="630.0" y="47.0" width="10" height="10" rx="2"/><text class="t-note" x="646.0" y="56.0" text-anchor="start">combine: 360</text><line class="grid" x1="616.0" y1="143.3" x2="626.0" y2="143.3"/><rect class="s1" x="630.0" y="138.3" width="10" height="10" rx="2"/><text class="t-note" x="646.0" y="147.3" text-anchor="start">dispatch: 194</text><text class="t-tick" x="337.0" y="290.0" text-anchor="middle">expert-parallel group size (GPUs), log scale</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">microseconds per call (H800, 128 tokens)</text></svg><figcaption>DeepEP's low-latency kernels on H800 with 128 tokens per batch: dispatch latency grows only from 163 to 194 microseconds and combine from 318 to 360 microseconds as the expert-parallel group grows from 8 to 256 GPUs.</figcaption></figure>

Going from 8 to 256 GPUs multiplies the number of partners by 32, but the latency grows by less than a fifth. So a wider spread of experts costs little extra waiting, and each GPU can hold fewer experts.

V4 brought a new kernel that fuses dispatch, expert computation and combine into one.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §3.1 · page 16
> [![The V4 report's paragraph on the MegaMoE kernel with both NVIDIA GPUs and HUAWEI Ascend NPUs, 1.50 to 1.73x speedup, up to 1.96x and as a component of DeepGEMM highlighted](/img/deepseek/papers/p7-v4-megamoe.png)](/img/deepseek/papers/p7-v4-megamoe.png)
>
> **What it says:** the fine-grained expert-parallel scheme was validated "on both NVIDIA GPUs and HUAWEI Ascend NPUs". Against non-fused baselines it gives a "1.50 ~ 1.73× speedup" for general inference and "up to 1.96×" for latency-sensitive cases. The CUDA version, MegaMoE, is open "as a component of DeepGEMM".
>
> **What it means:**
>
> A non-fused MoE layer runs as separate steps: send tokens, wait, compute, wait, send back. A fused kernel starts computing on the first tokens that arrive while the rest are still in flight. The waiting turns into overlap.
>
> The first highlight is the political one. This is the first DeepSeek report to say, in its own words, that a core kernel was validated on Huawei's chips.

## Chinese hardware

DeepSeek trained and served V3 on NVIDIA H800s. Its move toward Chinese accelerators can be traced through primary sources, step by step.

| When | What DeepSeek published |
|---|---|
| Dec 2024, V3 | Huawei Ascend listed as a target through Huawei's MindIE engine, in BF16 at first |
| Aug 2025, V3.1 | Weights and activations use "UE8M0 FP8 scale" format, for "compatibility with microscaling data formats" |
| Sep 2025, V3.2-Exp | Key kernels released in TileLang as well as CUDA; SGLang images for Ascend A2 and A3 on day one |
| Apr 2026, V4 | MegaMoE kernel validated on NVIDIA GPUs and Huawei Ascend NPUs |
| Sep 2026, V4.1 | FlashMLA "powering inference of the DeepSeek-V4.1 model on NVIDIA GPUs and Huawei Ascend NPUs"; Ascend versions of DeepGEMM and DeepEP |
| Oct 2026 | DeepInfer supports Huawei Ascend 950 |

> [!DEFINITION] UE8M0 scale
> A scale factor stored as an 8-bit exponent with no sign and no fraction. Every scale is a power of two. Some new chips handle such scales natively in "microscaling" number formats.

> [!DEFINITION] TileLang
> A programming language for GPU and NPU kernels. A kernel written once in TileLang can be compiled for different chips, which makes porting easier than hand-written CUDA.

What is not confirmed matters as much. DeepSeek's WeChat post on V3.1 said UE8M0 was for "soon-to-be-released next-generation domestic chips", as reported by Reuters; we did not check the original. Press reports said V4 was trained "entirely" on Ascend; the V4 report and model card do not say that. Whether DeepSeek's public API runs on Ascend, on NVIDIA, or on both is not public.

Outside DeepSeek, Huawei's CloudMatrix384 system serves R1 in INT8 on 384 Ascend NPUs and 192 Kunpeng CPUs. Its paper reports 6,688 prefill tokens per second per NPU. In decode it reports 1,943 tokens per second per NPU, at under 50 ms per token. The vLLM-Ascend project fits V3.2 at 8-bit weights on one Atlas 800 A3 node with 16 NPUs.

## Which setup for which goal

All speeds are reported numbers from the sources above, on the hardware named. None of them is our own measurement.

| Goal | Setup | Memory | Reported speed |
|---|---|---|---|
| Try current DeepSeek | API: `deepseek-flash` (V4.1-Flash-1017) | none | n/a; peak price $0.006 / $0.30 / $1.20 per 1M tokens (cache-hit input / input / output), half off-peak |
| Try the biggest model | API: `deepseek-v4-pro` | none | n/a; peak $0.044 / $1.32 / $3.96 per 1M |
| Learn reasoning on a laptop | R1 distill 7B or 8B, Ollama | about 5 GB + cache | measure it yourself |
| Better laptop reasoning | R1 distill 14B or 32B, 4-bit | 9 or 20 GB + cache | measure it yourself |
| Modern DeepSeek at home | V4-Flash at 2 bits with ds4, 128 GB Mac or DGX Spark | 96.8 GB file | 39.35 tok/s (M5 Max), 18.05 (Spark), at 2K context |
| Modern DeepSeek, GPU + CPU | V4-Flash with KTransformers, 1x RTX 5090 | 32 GB GPU + 200 GB RAM | "20+ tok/s" |
| Full 671B V3/R1, cheapest | UD-IQ1_S with llama.cpp, one 24 GB GPU + RAM | 140.2 GB file | "at least 2-3 tokens/s" (unconfirmed) |
| Full 671B V3/R1, usable | KTransformers, 1 GPU + Xeon (Gold 6454S) | 382 GB RAM (1 socket) or 1 TB (2) + 14 GB GPU | 12.21 tok/s decode (8 experts); 16.8 with 6 experts |
| Full 671B V3/R1 on a Mac | M3 Ultra 512 GB, 4-bit | 404 GB file | 17-18 tok/s short prompts; 6.67 to 10.89 after 5K to 13K prompts |
| V4.1-Flash at home | Unsloth GGUF with Engram on SSD, or ds4 | 122 to 306 GB RAM + VRAM, plus disk | none published |
| Serve V3/R1, one node | 8x H200 or 8x MI300X, FP8, SGLang or vLLM, DP attention + EP + MTP | 688.6 GB weights | MI300X: 34.26 tok/s at 1 request, 463.80 total at 100 |
| Serve V3/R1 on H100 | two 8x H100 nodes, TP16, or 4-bit on one node | 688.6 GB / about 400 GB | not given |
| Serve V4-Pro | 8x H200 (FP4), 8x B200, 4x GB300, or 2 nodes of H100 | 865 to 893 GB weights | not published |
| Serve at large scale | PD disaggregation + wide EP + EPLB, tens of nodes | n/a | DeepSeek: 73.7k in / 14.8k out tok/s per H800 node |
| Serve on Chinese hardware | Ascend A3 with vLLM-Ascend or SGLang images; DeepInfer on Ascend 950 | V3.2 W8A8: 16 NPUs | CloudMatrix384: 1,943 decode tok/s per NPU (Huawei) |

> [!TIP] In production
> Pick by the job, not by the model's fame. For a product, the API is cheaper than any hardware you can buy for most volumes. For privacy at home, V4-Flash on a 128 GB machine is the best current balance. For learning how the full V3 works, a quantized file on a big Mac or a CPU server is enough. For a real service, start with one 8-GPU node and the engine's recipe, and move to disaggregation only when one node is full.

<details><summary>Research notes: what we could not verify</summary>

- The SGLang V4 cookbook shows throughput tables in an interactive widget that our tools could not read. The vLLM V4-Pro recipe publishes no tokens-per-second figures.
- No tokens-per-second results for V4.1-Flash on consumer hardware were published by Unsloth or ds4 as of 10 November 2026.
- Unsloth's January 2025 blog post and the TweakTown article (448 GB of RAM, under 200 W for R1 on an M3 Ultra) block automated readers. We quote Unsloth's speeds through secondary pages and leave out the TweakTown numbers.
- The exact bytes of the V3.2 and V4 indexer caches in production engines are not documented. Our formulas assume an FP8 key with a 4-byte scale for V3.2 and an FP4 key with a 4-byte scale for V4. Both reproduce DeepSeek's published per-token totals exactly, which is strong but indirect evidence.
- Huawei's 6,688 prefill tokens per second per NPU assumes ideal expert balancing, according to secondary summaries we could not confirm.
- Which model answers `deepseek-v4-pro` API calls today is not fully clear from public pages: an announcement said V4-Pro would be routed to V4.1-Flash, then DeepSeek said it would keep serving V4-Pro.

</details>

> [!TAKEAWAYS]
> - Weight memory is parameters x bits / 8. The 671B V3 is 1,342 GB in BF16, 688.6 GB as the official FP8 file, 404.4 GB at Q4_K_M and 140.2 GB at UD-IQ1_S.
> - "131 GB" is 131 GiB: 140.2 GB divided by 2^30 bytes. Always check which unit a size uses.
> - 97.4% of V3's weights are routed experts. Total size decides whether a model fits; the 37B active weights decide how fast it writes.
> - MLA keeps V3's KV cache at 70,272 bytes per token, less than the 8B Llama distill. V4.1-Flash needs 890 bytes, so 1M tokens fit in 0.93 GB.
> - The native V3 needs 8x H200, 8x MI300X or two 8x H100 nodes. V4-Pro needs 8x H200 with FP4 or newer GPUs. V4-Flash fits 4 H200s or a 128 GB box at 2 bits.
> - Decode speed is capped by bandwidth divided by bytes read per token. An M3 Ultra's ceiling for V3 at Q4_K_M is 35.9 tok/s; reviews report 17-18.
> - MoE models suit tiered memory: attention and the shared expert on the GPU, routed experts in RAM, and Engram tables or cold experts on SSD.
> - DeepSeek serves with separate prefill (EP32, 32 GPUs) and decode (EP144, 144 GPUs) units, redundant experts, and an on-disk KV cache.
> - The 545% margin assumes every token was billed at R1 prices and $2 per GPU-hour. DeepSeek itself says real revenue was "substantially lower".
> - FlashMLA, DeepEP, DeepGEMM, EPLB, 3FS and, since October 2026, DeepInfer are open; their newest versions also target Huawei Ascend.

## Next

[Part 8](deepseek-8-the-specialised-models.md) leaves the general models and turns to the specialists. It covers Coder and Coder-V2, Math and Math-V2, the Prover models, the vision-language models, Janus, DeepSeek-OCR and the reward models. Many of them are small enough to run with the arithmetic from this part.

## References

- DeepSeek-AI (2024), DeepSeek-V3 Technical Report: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- Zhao et al. (2025), Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures: [arXiv:2505.09343](https://arxiv.org/abs/2505.09343)
- DeepSeek-AI (2026), DeepSeek-V4: [arXiv:2606.19348](https://arxiv.org/abs/2606.19348); DeepSeek-V4.1-Flash: [arXiv:2609.19969](https://arxiv.org/abs/2609.19969)
- Huawei (2025), Serving Large Language Models on Huawei CloudMatrix384: [arXiv:2506.12708](https://arxiv.org/abs/2506.12708)
- DeepSeek, Day 6 inference system overview: [open-infra-index](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md); [The Path to Open-Sourcing the DeepSeek Inference Engine](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md); [DeepInfer](https://github.com/deepseek-ai/DeepInfer)
- DeepSeek libraries: [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [DeepEP](https://github.com/deepseek-ai/DeepEP), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [EPLB](https://github.com/deepseek-ai/EPLB), [3FS](https://github.com/deepseek-ai/3FS)
- Model cards and file listings: [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3), [DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813), [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [Unsloth R1 GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF), [Unsloth V4.1-Flash guide](https://unsloth.ai/docs/models/deepseek-v4.1-flash)
- Engine recipes: [SGLang DeepSeek V3](https://docs.sglang.io/basic_usage/deepseek_v3.html), [SGLang DeepSeek V4](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4), [vLLM DeepSeek V3](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3.html), [vLLM V3.2-Exp](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html), [vLLM V4-Pro](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro), [vLLM-Ascend V3.2](https://docs.vllm.ai/projects/ascend/en/v0.13.0/tutorials/DeepSeek-V3.2.html)
- Large-scale reproductions: [SGLang, 96 H100s](https://lmsys.org/blog/2025-05-05-large-scale-ep/); [vLLM wide-EP](https://vllm.ai/blog/large-scale-serving)
- Consumer engines: [llama.cpp MLA PR #12801](https://github.com/ggml-org/llama.cpp/pull/12801), [V4 PR #24162](https://github.com/ggml-org/llama.cpp/pull/24162), [KTransformers V3/R1 tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md), [KTransformers V4-Flash](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md), [ds4 performance](https://github.com/antirez/ds4/blob/main/docs/PERFORMANCE.md), [Ollama deepseek-r1 tags](https://ollama.com/library/deepseek-r1/tags)
- Apple Silicon reports: [Apple Mac Studio announcement](https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/), [MacRumors on the R1 review](https://www.macrumors.com/2025/03/17/apples-m3-ultra-runs-deepseek-r1-efficiently/), [M3 Ultra prompt and write speeds](https://dev.to/someoddcodeguy/m3-ultra-mac-studio-512gb-prompt-and-write-speeds-for-deepseek-v3-0-671b-gguf-q4km-for-those-47hi)
- API prices: [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing)
