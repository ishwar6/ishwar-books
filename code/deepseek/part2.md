---
title: "The DeepSeek-V3 Architecture, Piece by Piece"
description: "How DeepSeek-V3 holds 671 billion parameters but uses 37 billion per token: many small experts, a fair router, a compressed memory and a second prediction head."
date: 2026-11-10
tags: [deepseek, mixture-of-experts, mla, attention, llm, architecture]
series: "DeepSeek, From the Ground Up"
series_part: 2
motif: blocks
accent: "#4f8cff"
---

You ask DeepSeek-V3 a question, and it starts to write. For every word of the answer, the model reads about 37 billion of its numbers. The other 634 billion sit still in memory and wait for a different word.

That one fact is the key to the whole design. A model with the knowledge of 671 billion parameters does the work of a model about 18 times smaller, for each token. The same design runs inside R1 and V3.1, and with one addition inside V3.2, all listed in [Part 1](deepseek-1-every-model-and-what-is-open.md). Part 2 takes it apart, piece by piece, and counts every parameter from the real `config.json`.

> [!TIP] The whole idea in one minute
> DeepSeek-V3 is a stack of 61 transformer layers. In 58 of them, the usual feed-forward block becomes **256 small experts plus 1 shared expert**. A router sends each token to only **8** of the 256. A small **bias** per expert, nudged after every training step, keeps all experts equally busy without an extra loss term. Each token may use experts on at most **4 of the 8 machines** that hold them, to limit network traffic.
>
> Attention uses **MLA**: each token leaves only **576 numbers per layer** in memory. That is 70,272 bytes for all 61 layers, about 71 times less than plain multi-head attention with the same heads. A small extra layer learns to predict the token **after** the next one, and later serves as a fast drafter. Counted from the config: **671.03 billion** parameters in the main model, **37.55 billion** used per token.

## A map of the model

The frame is the standard transformer block: attention, then a feed-forward network, each with an RMSNorm before it and a residual connection around it. [Inside a Modern Language Model](inside-a-modern-language-model.md) covers that frame, and [Attention, Part 1](attention-1-self-attention.md) builds attention from scratch.

> [!DEFINITION] FFN and SwiGLU
> The feed-forward network takes one hidden state of width $$d$$ and expands it to a wider middle layer of width $$w$$. It applies a non-linear function, then shrinks the result back to $$d$$. Modern models use SwiGLU, a version with **three** weight matrices (gate, up, down), so one FFN holds $$3 \times d \times w$$ parameters.

DeepSeek-V3 keeps this frame and changes what is inside it. The figure is the map for this whole page.

{{FIG:block|The map of DeepSeek-V3 drawn from its config.json. Left: the whole model, from token ids to the next token, with 3 dense layers, 58 MoE layers and one multi-token prediction module. Right: one MoE layer. The two highlighted boxes, MLA and DeepSeekMoE, are the two big changes this part explains.}}

The paper draws the same picture with more detail.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §2.1, Figure 2 · page 7
> [![Figure 2 of the DeepSeek-V3 report: on the left a transformer block with RMSNorm, attention and a feed-forward network; on the right the DeepSeekMoE layer with shared and routed experts and a router, and below it Multi-Head Latent Attention with the cached latent vectors marked](/img/deepseek/papers/p2-v3-figure2.png)](/img/deepseek/papers/p2-v3-figure2.png)
>
> **What it says:** "Following DeepSeek-V2, we adopt MLA and DeepSeekMoE for efficient inference and economical training."
>
> **What it means:**
>
> Look at the left column first. It is an ordinary transformer block: RMSNorm, attention, add; RMSNorm, feed-forward, add. Nothing in the frame is new.
>
> The two dashed boxes on the right zoom into the two parts DeepSeek replaced. The top box swaps one big feed-forward network for many small "experts" (blue) and a few "shared" ones (green). A router picks a few experts per token. The bottom box is attention that stores a small striped "latent" vector instead of full keys and values. The stripes mark what is "Cached During Inference".
>
> The caption gives the motive in a few words each. MLA is for **efficient inference** (small memory while writing). DeepSeekMoE is for **economical training** (less arithmetic per token). The rest of this page follows those two boxes.

The exact sizes are in the paper's list of hyper-parameters, and they match the released `config.json` field for field.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §4.2 · page 22
> [![The model hyper-parameters paragraph of the DeepSeek-V3 report, with the MoE layout and the 671B total, 37B activated parameter count highlighted](/img/deepseek/papers/p2-v3-hyper.png)](/img/deepseek/papers/p2-v3-hyper.png)
>
> **What it says:** 61 layers, hidden size 7,168, 128 heads. "We substitute all FFNs except for the first three layers with MoE layers." Each MoE layer has "1 shared expert and 256 routed experts", "8 experts will be activated for each token", and the model "comprises 671B total parameters, of which 37B are activated for each token."
>
> **What it means:**
>
> Every number on the map comes from this paragraph. Three layers keep a normal FFN; 58 get experts. Each expert is narrow (width 2,048 against 18,432 for a normal FFN). The last sentence is the headline, and we will check it ourselves at the end of this page with a script.
>
> Note the phrase "except for the first three layers". The earlier DeepSeekMoE paper kept only the first layer dense, because "the load balance status converges especially slower for the first layer" (Dai et al., 2024, page 15). The V3 report does not say why it chose three.

## Why mixture of experts

A dense model has a hard rule: every parameter works on every token. If you want a model that knows more, you add parameters, and every token pays for all of them. Knowledge and cost grow together.

But a token does not need all the knowledge at once. The word "mitochondria" needs biology. The token `}` in a Python file needs code. A model could keep many specialist sub-networks and run only the few that matter for each token.

> [!DEFINITION] Router (gate)
> A small learned layer that gives each expert a score for the current token. In DeepSeek-V3 it is one vector per expert, called a **centroid**. The score of expert $$i$$ is the dot product of the token's hidden state with centroid $$i$$, passed through a sigmoid.

The idea is old. Shazeer and colleagues trained a sparsely-gated MoE in 2017, and Google's GShard (Lepikhin et al., 2021) scaled it up. DeepSeek's change was about the **shape** of the experts, which the next section explains.

First, look at what MoE buys in DeepSeek-V3's own numbers. A dense V3 layer has one SwiGLU FFN of width 18,432. An MoE layer has 257 experts of width 2,048. Per token, 9 of them run: 8 routed plus 1 shared. And $$9 \times 2{,}048 = 18{,}432$$.

$$
\underbrace{3 \times 7{,}168 \times 18{,}432}_{\text{dense FFN}} = 396{,}361{,}728 = 9 \times \underbrace{3 \times 7{,}168 \times 2{,}048}_{\text{one expert}}
$$

Read it left to right: the dense FFN has $$3 \times d \times w$$ parameters with $$d = 7{,}168$$ and $$w = 18{,}432$$, which equals nine experts of width 2,048. So a token does **exactly the same arithmetic** in a dense layer and in an MoE layer. The MoE layer just holds 257 experts instead of the 9 a token uses.

{{FIG:dense_vs_moe|A dense FFN and a DeepSeekMoE layer from DeepSeek-V3, drawn to the same budget. Both cost 396.36 million multiply-adds per token. The MoE layer stores 11.32 billion parameters, 28.6 times more, and each token uses 9 of its 257 experts (highlighted).}}

That is the trade in one picture: same work per token, 28.6 times the parameters to learn with. The price is memory (all 257 experts must sit somewhere) and a new problem: someone has to decide which 9.

## DeepSeekMoE: many small experts, plus a shared one

Earlier MoE designs such as GShard used a few big experts and sent each token to the best 2. DeepSeek's January 2024 paper argued that this is the wrong shape, and changed it in two steps.

> [!PAPER] Dai et al. (2024), DeepSeekMoE · §3, Figure 2 · page 5
> [![Figure 2 of the DeepSeekMoE paper: (a) conventional top-2 routing over N experts, (b) the same experts split into 2N smaller ones with top-4 routing, (c) one of them turned into an always-on shared expert with top-3 routing](/img/deepseek/papers/p2-moe-figure2.png)](/img/deepseek/papers/p2-moe-figure2.png)
>
> **What it says:** Three versions of one MoE layer: (a) conventional top-2 routing, (b) fine-grained expert segmentation, (c) shared expert isolation. "Across these three architectures, the number of expert parameters and computational costs remain constant."
>
> **What it means:**
>
> Read the three panels left to right as one experiment with a fixed budget. In (a), a router picks 2 of $$N$$ big experts. In (b), each expert is cut in half, so there are $$2N$$ experts of half the size, and the router picks 4. The total parameters and the work per token have not changed.
>
> In (c), one of those small experts (green) is taken out of the competition. Every token always goes through it, and the router now picks only 3. Again, nothing about the budget changed. Only the **arrangement** did. The paper's claim is that arrangement alone makes the model better.

### Step 1: cut each expert into smaller pieces

> [!PAPER] Dai et al. (2024), DeepSeekMoE · §3.1 · page 5
> [![The fine-grained expert segmentation paragraph: each expert FFN is split into m smaller experts and the number of activated experts is raised m times](/img/deepseek/papers/p2-moe-segment.png)](/img/deepseek/papers/p2-moe-segment.png)
>
> **What it says:** "we segment each expert FFN into $$m$$ smaller experts by reducing the FFN intermediate hidden dimension to $$1/m$$ times its original size", and "increase the number of activated experts to $$m$$ times to keep the same computation cost."
>
> **What it means:**
>
> Think of a hospital with 16 general doctors, where each patient sees 2. Each doctor must know a bit of everything, because any patient may come in. Now replace them with 64 narrower specialists and let each patient see 8. The total staff hours are the same, but each specialist can focus.
>
> The paragraph before this one gives the reason in the authors' words: with few experts, one expert "will intend to learn vastly different types of knowledge in its parameters". Smaller experts can each hold one narrow kind of knowledge, and the router combines them per token.

The second reason is combinations. A token's "team" of experts is a set, and more experts give many more possible sets.

> [!PAPER] Dai et al. (2024), DeepSeekMoE · §3.1 · page 6
> [![The combinatorial argument: top-2 of 16 experts gives 120 combinations; top-8 of 64 smaller experts gives 4,426,165,368](/img/deepseek/papers/p2-moe-combos.png)](/img/deepseek/papers/p2-moe-combos.png)
>
> **What it says:** "A typical top-2 routing strategy can yield $$\binom{16}{2} = 120$$ possible combinations." With each expert split into 4, "the fine-grained routing strategy can yield $$\binom{64}{8} = 4{,}426{,}165{,}368$$ potential combinations."
>
> **What it means:**
>
> $$\binom{n}{k}$$ counts the ways to choose $$k$$ things from $$n$$ when order does not matter. Choosing 2 of 16 gives 120 teams. Choosing 8 of 64 gives about 4.4 billion. Same budget, 37 million times more ways to mix knowledge for each token.
>
> DeepSeek-V3 goes much further: 8 of 256 routed experts. Our script prints the count: $$\binom{256}{8} = 409{,}663{,}695{,}276{,}000$$, about 410 trillion possible teams in every MoE layer.

### Step 2: one expert that every token uses

If every expert is small and specialised, where does common knowledge go? Grammar, frequent words and basic facts are useful for almost every token.

> [!PAPER] Dai et al. (2024), DeepSeekMoE · §3.2 · page 6
> [![The shared expert isolation paragraph: routed experts may all learn the same common knowledge, so some experts are made shared and every token is assigned to them](/img/deepseek/papers/p2-moe-shared.png)](/img/deepseek/papers/p2-moe-shared.png)
>
> **What it says:** Without shared experts, "multiple experts may converge in acquiring shared knowledge", which wastes parameters. So "each token will be deterministically assigned to these shared experts", and the router picks that many fewer routed experts.
>
> **What it means:**
>
> Back to the hospital. If every specialist must also know basic first aid, you pay for first aid 64 times. It is cheaper to hire one general nurse whom every patient sees, and let the specialists drop the basics.
>
> The shared expert is that nurse. It is not routed, so it can hold what every token needs. The routed experts are then free to differ from each other. The paper credits the first version of this idea to earlier systems work (Rajbhandari et al., 2022) and says DeepSeek approached it "from an algorithmic standpoint".

> [!DEFINITION] Routed expert and shared expert
> A **routed** expert runs only when the router picks it for a token. A **shared** expert runs for every token, with no router involved. DeepSeek-V3 has 256 routed experts and 1 shared expert in each MoE layer; its predecessor V2 had 160 and 2.

### The equations of one V3 MoE layer

Here is the whole layer as DeepSeek-V3 computes it.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §2.1.2 · pages 8 and 9
> [![The DeepSeekMoE equations in the V3 report: the output is the input plus the shared experts plus the gated routed experts; gates are normalised top-K sigmoid affinities](/img/deepseek/papers/p2-v3-moe-eqs.png)](/img/deepseek/papers/p2-v3-moe-eqs.png)
> [![The paragraph after the equations, which defines every symbol and says V3 uses the sigmoid function and normalises the selected affinity scores](/img/deepseek/papers/p2-v3-sigmoid.png)](/img/deepseek/papers/p2-v3-sigmoid.png)
>
> **What it says:** Equations 12 to 15. "Slightly different from DeepSeek-V2, DeepSeek-V3 uses the sigmoid function to compute the affinity scores, and applies a normalization among all selected affinity scores to produce the gating values."
>
> **What it means:**
>
> Four lines, four jobs. Equation 15 scores every routed expert for this token. Equation 14 keeps the top $$K_r = 8$$ scores and sets the rest to zero. Equation 13 rescales the 8 kept scores so they add up to 1. Equation 12 runs the shared expert and the 8 chosen experts, weights each routed output by its gate, and adds everything to the input.
>
> V2 used softmax for the scores, which makes all 160 scores compete for a total of 1. Sigmoid gives each expert an independent score between 0 and 1. The report does not explain the switch. Our reading is that it fits the bias trick later on this page, because one expert's score no longer depends on all the others.

Written out, with the symbols in words:

$$
h'_t = u_t + \sum_{i=1}^{N_s} \mathrm{FFN}^{(s)}_i(u_t) + \sum_{i=1}^{N_r} g_{i,t}\, \mathrm{FFN}^{(r)}_i(u_t)
$$

$$
g_{i,t} = \frac{g'_{i,t}}{\sum_{j=1}^{N_r} g'_{j,t}}, \qquad
g'_{i,t} = \begin{cases} s_{i,t} & s_{i,t} \in \mathrm{Topk}(\{s_{j,t}\}, K_r) \\ 0 & \text{otherwise} \end{cases}, \qquad
s_{i,t} = \mathrm{Sigmoid}(u_t^{\top} e_i)
$$

- $$u_t$$ is the hidden state of token $$t$$ as it enters the layer (7,168 numbers), and $$h'_t$$ is what leaves it.
- $$N_s = 1$$ shared expert and $$N_r = 256$$ routed experts; $$\mathrm{FFN}_i$$ is expert $$i$$, a small SwiGLU of width 2,048.
- $$e_i$$ is the centroid of routed expert $$i$$, and $$s_{i,t}$$ is the **affinity**: how well token $$t$$ matches expert $$i$$.
- $$K_r = 8$$ is how many routed experts each token uses; $$g_{i,t}$$ is the **gate**, the weight of expert $$i$$ in the sum.

> [!DEFINITION] Affinity and gate
> The **affinity** $$s_{i,t}$$ is the router's raw score of expert $$i$$ for token $$t$$. The **gate** $$g_{i,t}$$ is the weight actually used to mix that expert's output. It is zero for experts that were not picked, and the normalised affinity for the 8 that were.

A tiny worked example. Suppose a token's four best affinities are 0.90, 0.60, 0.55 and 0.30, and $$K_r = 4$$. Their sum is 2.35, so the gates are $$0.90/2.35 = 0.383$$, then 0.255, 0.234 and 0.128. They add up to 1. The layer output is the input, plus the shared expert, plus 0.383 times the first expert's output, and so on.

One detail is only in the code. The released model multiplies the normalised gates by `routed_scaling_factor = 2.5` from the config. So in practice the 8 gates of a token add up to 2.5, not 1. The paper hints at this when it says V3 "multiplies additional scaling factors at the width bottlenecks" (page 22).

## Routing in V3: top-8 of 256, on at most 4 machines

So far the router looks free to pick any 8 of 256 experts. In real training it is not, because of where the experts live.

> [!DEFINITION] Node, NVLink and InfiniBand
> A **node** is one server, here with 8 GPUs. GPUs inside a node talk over **NVLink**, a fast direct link. GPUs in different nodes talk over **InfiniBand (IB)**, a network that is slower per GPU.

During training, each layer's 256 experts are spread over 64 GPUs in 8 nodes, 4 experts per GPU. A token whose 8 experts sit on 8 different nodes sends its hidden state over the slow network 8 times.

{{FIG:cluster|Where the 256 routed experts of one MoE layer lived during DeepSeek-V3's training: 8 nodes of 8 GPUs, 4 experts per GPU, 32 experts per node. The highlighted nodes are the 4 that the example token below was allowed to use.}}

The V3 report gives the numbers that made this a problem.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §3.2.2 · page 13
> [![The cross-node communication paragraph: NVLink offers 160 GB/s, about 3.2 times InfiniBand's 50 GB/s, so each token is dispatched to at most 4 nodes; each token can use an average of 3.2 experts per node](/img/deepseek/papers/p2-v3-ib.png)](/img/deepseek/papers/p2-v3-ib.png)
>
> **What it says:** "NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)." So "we limit each token to be dispatched to at most 4 nodes". A token first crosses IB once per target node, then fans out inside the node over NVLink.
>
> **What it means:**
>
> Picture a parcel service with slow trucks between cities and fast bikes inside each city. Sending one parcel per expert by truck is wasteful. It is better to send one truck to each city and let the bikes deliver to the right street.
>
> That is exactly the design. The slow hop (IB) happens once per node, not once per expert. The fast hop (NVLink) handles the rest. Capping the number of nodes caps the slow traffic. The paragraph goes on, on page 14, with a nice consequence: with this scheme, V3 "can scale up this number to a maximum of 13 experts (4 nodes × 3.2 experts/node) while preserving the same communication cost".

### How the 4 nodes are chosen

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §2.1.2 · page 10
> [![The node-limited routing and no token-dropping paragraphs: each token is sent to at most M nodes, chosen by the sum of the highest K_r/M affinity scores on each node; V3 drops no tokens in training or inference](/img/deepseek/papers/p2-v3-node.png)](/img/deepseek/papers/p2-v3-node.png)
>
> **What it says:** "each token will be sent to at most $$M$$ nodes, which are selected according to the sum of the highest $$K_r/M$$ affinity scores of the experts distributed on each node." And V3 "does not drop any tokens during training".
>
> **What it means:**
>
> The rule has two rounds, like choosing restaurants before dishes. First, give each node a score: the sum of its best $$K_r/M = 8/4 = 2$$ expert affinities. Keep the 4 best nodes. Second, pick the top 8 experts, but only from those 4 nodes.
>
> Why the **sum of the top 2**, not the single best expert? Because the token will take about 2 experts from each kept node. A node with one star and nothing else is a worse trip than a node with two good experts. V2 used the single best score per device; V3 switched to the sum.

> [!DEFINITION] Token dropping
> In many MoE systems, each expert has a fixed capacity per batch. When too many tokens pick the same expert, the extra tokens skip it ("are dropped") and pass through unchanged. DeepSeek-V3 says it never drops tokens, in training or in serving, because its balancing keeps loads even.

Here is the rule on a case small enough to check by hand. There are 12 experts on 4 nodes (3 per node), $$K = 4$$ and $$M = 2$$. So each node is scored by its best 2 experts.

{{FIG:hand_route|Node-limited routing by hand. Node scores are the sums of each node's 2 best affinities: 0.90 + 0.30 = 1.20, 0.60 + 0.55 = 1.15, 0.25 + 0.20 = 0.45 and 0.85 + 0.10 = 0.95. Nodes 0 and 1 win. The top 4 experts inside them are e0, e3, e4 and e2. Expert e9, the second-best overall, is lost because its node scores too low.}}

The interesting loss is e9. With an affinity of 0.85 it is the second-best expert for this token. But its node has only one good expert, so the node loses to nodes 0 and 1. The router takes e2 (0.30) instead. This is the price of the rule: a small loss in choice for a large saving in traffic. The router is trained under this rule from the start, so it learns to work within it.

### The router, from scratch, against the library

Here is our NumPy version of the whole V3 router. Each line maps to an equation; the bias line is explained in the next section.

```python
def route_v3(u, centroids, bias, n_group, topk_group, top_k, scale):
    """u: (T, d) token vectors; centroids: (E, d); bias: (E,). Returns expert ids (T, top_k) and gates (T, top_k)."""
    s = 1 / (1 + np.exp(-(u @ centroids.T)))            # Eq. 15: affinity s_i,t = sigmoid(u_t . e_i), shape (T, E)
    choice = s + bias                                    # Eq. 16: the bias is added for choosing only
    T, E = s.shape
    per = choice.reshape(T, n_group, E // n_group)       # experts grouped by node: (T, 8 nodes, 32 experts)
    k_per_node = top_k // topk_group                     # K_r / M = 8 / 4 = 2
    node_score = np.sort(per, axis=-1)[:, :, -k_per_node:].sum(-1)   # sum of the 2 best experts on each node
    nodes = np.argsort(-node_score, axis=-1)[:, :topk_group]          # keep the 4 best nodes
    allowed = np.zeros((T, n_group), bool)
    np.put_along_axis(allowed, nodes, True, axis=1)
    masked = np.where(np.repeat(allowed, E // n_group, axis=1), choice, -np.inf)
    ids = np.argsort(-masked, axis=-1)[:, :top_k]       # Eq. 14/16: top-8 among the allowed experts
    g = np.take_along_axis(s, ids, axis=1)               # gates come from s, NOT from s + b
    g = g / g.sum(-1, keepdims=True)                     # Eq. 13: normalise the 8 gates to sum to 1
    return ids, g * scale, dict(s=s, node_score=node_score, nodes=nodes)
```

The config names map one to one: `n_group = 8` is the number of nodes, `topk_group = 4` is $$M$$, `num_experts_per_tok = 8` is $$K_r$$ and `routed_scaling_factor = 2.5` is `scale`. The `topk_method` in V3's config is `noaux_tc`, the name for this bias-based, node-limited rule.

We then built transformers' own `DeepseekV3TopkRouter` with V3's routing shape: hidden size 7,168, 256 experts, 8 groups, 4 kept, top-8. We gave it random weights and a random bias, and fed 4,096 random tokens through both. We did the same for V2's older rule with `DeepseekV2TopkRouter` on a tiny config. The output of `part2_router.py`:

```text
== 2. V3 router (256 experts, 8 nodes, 4 nodes kept, top-8, scale 2.5) vs transformers ==
  4096 random tokens: same 8 experts for every token: True; largest gate difference: 5.3e-07
  gates sum to 2.5000 (= 2.5, the routed_scaling_factor)
  nodes touched per token: max 4, mean 3.97 (limit 4)
  same tokens with no node limit: mean 5.30 nodes, 83% of tokens need more than 4

== 3. V2 router on a tiny random config (16 experts, 4 devices, 2 devices kept, top-4) vs transformers ==
  1000 random tokens: same 4 experts for every token: True; largest gate difference: 6.4e-07
```

Both routers pick the same experts for every token, and the gates agree to about one part in two million (float32 rounding). The last V3 line shows why the limit matters. With random weights and no limit, a token's 8 experts land on 5.3 nodes on average, and 83% of tokens would cross more than 4. A trained router is not random, so treat this as an illustration of the mechanism, not a measurement of V3.

{{FIG:route_token|One random token through a V3-shaped router (random weights). Each row is a node of 32 experts; darker cells have higher affinity. Nodes 0, 3, 4 and 7 have the highest sums of their 2 best affinities and are kept. The 8 boxed cells are the chosen experts, and their gates add up to 2.5.}}

## Load balancing: keeping 256 experts busy

A learned router has a bad habit. Early in training, a few experts happen to be slightly better. The router sends them more tokens, so they train more and get better still. The others get few tokens and stay weak.

> [!PAPER] Dai et al. (2024), DeepSeekMoE · §3.3 · page 7
> [![The load balance paragraph: learned routing risks routing collapse, where the model always selects only a few experts, and load imbalance across devices; it introduces an expert-level balance loss](/img/deepseek/papers/p2-moe-collapse.png)](/img/deepseek/papers/p2-moe-collapse.png)
>
> **What it says:** Learned routing risks "routing collapse (Shazeer et al., 2017), i.e., the model always selects only a few experts, preventing other experts from sufficient training." Also, across devices, "load imbalance can exacerbate computation bottlenecks."
>
> **What it means:**
>
> There are two separate harms. The first is about learning: unused experts are wasted parameters, so the model behaves like a much smaller one. The second is about speed: if one GPU holds the popular experts, every other GPU waits for it at every layer. One slow GPU sets the pace for 64.

> [!DEFINITION] Expert load and routing collapse
> An expert's **load** is how many tokens it receives in a batch. With 4,096 tokens and 8 picks each over 256 experts, the fair share is $$4{,}096 \times 8 / 256 = 128$$ tokens per expert. **Routing collapse** is the end state where a few experts take almost all tokens.

### The classic fix: an auxiliary loss

The usual remedy adds a second term to the training loss that is small when the load is even. DeepSeek-V2 used this, and V3 keeps a very weak version, so it is worth understanding.

Here is V3's version, the sequence-wise balance loss:

$$
\mathcal{L}_{\mathrm{Bal}} = \alpha \sum_{i=1}^{N_r} f_i P_i, \qquad
f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathbb{1}\big(\text{token } t \text{ picks expert } i\big), \qquad
P_i = \frac{1}{T} \sum_{t=1}^{T} \frac{s_{i,t}}{\sum_{j} s_{j,t}}
$$

- $$T$$ is the number of tokens in one sequence; $$N_r$$ and $$K_r$$ are 256 and 8 as before.
- $$f_i$$ is expert $$i$$'s **load**, scaled so that the fair share is exactly 1.0.
- $$P_i$$ is expert $$i$$'s average **normalised affinity**: how much the router likes it, on average.
- $$\alpha$$ is the balance factor, the strength of this term.

The trick is the product $$f_i P_i$$. The load $$f_i$$ is a count, so training cannot push on it directly. The affinity $$P_i$$ is smooth, so it can. An expert with a high load gets a large $$f_i$$, which makes its $$P_i$$ costly, so training lowers that expert's affinities. Over time it gets fewer tokens.

A tiny worked example, from `part2_balance.py`: 4 experts, each token picks 1, 4 tokens with the same affinities in both cases.

```text
== Worked example of the balance loss: 4 experts, each token picks 1, 4 tokens ==
  collapsed (picks 0,0,0,3): f = [3.0, 0.0, 0.0, 1.0]  P = [0.48, 0.111, 0.185, 0.223]  sum f*P = 1.664
  balanced  (picks 0,1,2,3): f = [1.0, 1.0, 1.0, 1.0]  P = [0.48, 0.111, 0.185, 0.223]  sum f*P = 1.000
```

When expert 0 takes 3 of 4 tokens, its load is $$3 \times 4/4 = 3.0$$, and the sum is $$3.0 \times 0.48 + 1.0 \times 0.223 = 1.664$$. When every expert takes one token, every $$f_i$$ is 1 and the sum is just $$\sum P_i = 1.000$$. So the collapsed case costs 66% more, and training is pushed away from it.

The paper sets the problem with this fix in one sentence.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §2.1.2 · page 9
> [![The auxiliary-loss-free load balancing paragraph: an auxiliary loss that is too large impairs performance; V3 adds a bias term b_i to each affinity for top-K routing only, and moves it by gamma after each step depending on the expert's load](/img/deepseek/papers/p2-v3-auxfree.png)](/img/deepseek/papers/p2-v3-auxfree.png)
>
> **What it says:** "too large an auxiliary loss will impair the model performance". So "we introduce a bias term $$b_i$$ for each expert" and add it to the affinities to choose the top-K. "Note that the bias term is only used for routing." At the end of each step, "we will decrease the bias term by $$\gamma$$ if its corresponding expert is overloaded, and increase it by $$\gamma$$ if its corresponding expert is underloaded".
>
> **What it means:**
>
> The auxiliary loss has a hidden cost. Its gradient flows into the router, and from there into the model. Part of the model's learning effort goes into "be balanced" instead of "predict the next token". Turn it up and balance improves, but quality drops. Turn it down and experts collapse.
>
> V3's idea is to take balance out of the loss altogether. Each expert gets one extra number, a bias, that is **not learned** by gradient descent. A plain rule moves it, like a thermostat. Too many tokens last step means a slightly lower bias; too few means a slightly higher one.
>
> The bias changes **which** experts are picked, never **how much** each picked expert counts. The gate still uses the original $$s_{i,t}$$. So the model's output for a given choice of experts is untouched by balancing.

### The bias rule, written out

$$
g'_{i,t} = \begin{cases} s_{i,t} & s_{i,t} + b_i \in \mathrm{Topk}(\{s_{j,t} + b_j\}, K_r) \\ 0 & \text{otherwise} \end{cases}
\qquad\quad
b_i \leftarrow b_i - \gamma \cdot \mathrm{sign}\big(\mathrm{load}_i - \overline{\mathrm{load}}\big)
$$

- $$b_i$$ is expert $$i$$'s bias. It starts at 0 and is stored in the released weights as `e_score_correction_bias`.
- $$\mathrm{load}_i$$ is the number of tokens expert $$i$$ got in the whole batch of this step; $$\overline{\mathrm{load}}$$ is the average over experts.
- $$\gamma$$ is the **bias update speed**. $$\mathrm{sign}(x)$$ is $$+1$$ if $$x > 0$$, $$-1$$ if $$x < 0$$ and 0 if $$x = 0$$.

The left half is the choosing rule: rank experts by $$s + b$$, but keep $$s$$ as the gate. The right half is the thermostat, written as one line. The paper describes it in words; the sign form is our compact way to write the same rule.

A tiny example with two experts, A and B. A token scores A at 0.62 and B at 0.60, so A wins. After a few steps in which A was overloaded, $$b_A = -0.003$$ and $$b_B = +0.003$$. Now the token sees $$0.617$$ for A and $$0.603$$ for B, so A still wins.

A token that scored A at 0.604 and B at 0.600 would now go to B (0.601 against 0.603). Only close calls flip. Those are tokens for which both experts were almost equally good.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §4.2 · page 23
> [![The training hyper-parameters: routed experts of each layer on 64 GPUs in 8 nodes; at most 4 nodes per token; bias update speed 0.001 for the first 14.3T tokens and 0.0 for the last 500B; balance loss alpha 0.0001](/img/deepseek/papers/p2-v3-gamma.png)](/img/deepseek/papers/p2-v3-gamma.png)
>
> **What it says:** The routed experts of each layer are "uniformly deployed on 64 GPUs belonging to 8 nodes", each token goes to "at most 4 nodes", and the "bias update speed $$\gamma$$" is 0.001 for the first 14.3T tokens and 0.0 for the remaining 500B. The balance loss uses $$\alpha = 0.0001$$.
>
> **What it means:**
>
> A step of 0.001 is tiny next to affinities between 0 and 1. That is on purpose: the bias moves a little every step and only tips close calls. Over thousands of steps it can still move a lot.
>
> Setting $$\gamma$$ to zero for the last 500 billion tokens freezes the biases. The model finishes training with a fixed routing rule, the same rule it will use when serving. The value $$\alpha = 0.0001$$ shows how little V3 still relies on the auxiliary loss.

### A simulation: watch the bias even out the load

We cannot train V3, but we can simulate its router. `part2_balance.py` builds a V3-shaped router (256 experts, top-8) with random centroids. To make some experts "popular", it adds a fixed random offset to each expert's score, as a trained router might have. Each step, a fresh batch of 4,096 random tokens arrives. Here is the core loop:

```python
def step(offset, bias):
    u = rng.normal(0, 1, (T, D))
    s = 1 / (1 + np.exp(-(u @ centroids.T + offset)))    # affinities s_i,t (sigmoid)
    ids = np.argpartition(-(s + bias), K, axis=1)[:, :K]  # top-8 of s + b
    load = np.bincount(ids.ravel(), minlength=E)         # tokens each expert received
    return s, ids, load
```

and the update, applied after each step exactly as the paper describes it, with $$\gamma = 0.001$$:

```python
        if balance:                                       # V3's rule, applied at the end of every step
            bias -= GAMMA * np.sign(load - load.mean())
```

We measure imbalance with **MaxVio**, the measure used in the paper that introduced this method (Wang et al., 2024).

$$
\mathrm{MaxVio} = \frac{\max_i \mathrm{load}_i - \overline{\mathrm{load}}}{\overline{\mathrm{load}}}
$$

It says how far the busiest expert is above the fair share, as a fraction of the fair share. MaxVio = 0 is perfect balance; MaxVio = 1 means the busiest expert does twice its share of work, so its GPU takes twice as long.

```text
== A no balancing ==
  MaxVio: step 1 3.99, step 100 4.10, step 400 4.30
  tokens per expert at the last step: fair share 128, busiest 678, idlest 0, experts with no token 6
  busiest 8 experts take 13.0% of all assignments (fair: 3.1%)
  balance loss L_Bal (alpha = 1e-4): first step 1.114e-04, last step 1.114e-04

== B bias rule ==
  MaxVio: step 1 3.88, step 100 0.35, step 400 0.33
  tokens per expert at the last step: fair share 128, busiest 170, idlest 89, experts with no token 0
  busiest 8 experts take 3.8% of all assignments (fair: 3.1%)
  balance loss L_Bal (alpha = 1e-4): first step 1.115e-04, last step 1.002e-04
  bias range after 400 steps: -0.104 to +0.245; correlation with the popularity offset: -0.87
```

Without balancing, the busiest expert gets 678 tokens against a fair share of 128, and 6 experts get none. With the bias rule, MaxVio falls from 3.88 to 0.35 within 100 steps and stays there. The busiest expert drops to 170 and no expert is idle.

{{FIG:load_bars|Tokens per expert in one batch, all 256 experts sorted from busiest to idlest, from our simulation. Top: no balancing; the busiest expert gets 678 tokens and 6 get none. Bottom: after 400 steps of the bias rule with gamma = 0.001; every expert gets between 89 and 170 tokens, near the fair share of 128.}}

{{FIG:maxvio|MaxVio in our simulation over 400 steps. Without balancing it stays near 4: the busiest expert does about five times its fair share. With the bias rule it falls below 0.4 in about 100 steps.}}

Two lines of the output deserve a closer look. The biases end up strongly anti-correlated with the hidden popularity offsets (−0.87). The rule found the popular experts and pushed them down, without ever seeing the offsets. And MaxVio does not reach 0, because each batch is random. With 128 tokens expected per expert, chance alone spreads loads by about ±11 tokens. The busiest of 256 experts always sits a few of those spreads above the mean.

We also ran a harsher toy. We added a "rich get richer" feedback: after each step, every expert's popularity grows with its load, as if busy experts learned faster. This is our assumption, made to show collapse, not a model of real training.

```text
== C feedback, no balancing ==
  MaxVio: step 1 4.29, step 100 31.00, step 400 31.00
  tokens per expert at the last step: fair share 128, busiest 4096, idlest 0, experts with no token 248
  busiest 8 experts take 100.0% of all assignments (fair: 3.1%)

== D feedback + bias rule ==
  MaxVio: step 1 4.27, step 100 20.03, step 400 0.34
  tokens per expert at the last step: fair share 128, busiest 171, idlest 4, experts with no token 0
```

Without balancing, the router collapses completely: 8 experts take every token, and 248 are dead. MaxVio reaches 31, its maximum, because $$(4{,}096 - 128)/128 = 31$$. With the bias rule, the same feedback still causes a bad start (MaxVio 20 at step 100). But the biases catch up, and by step 400 every expert is back in use.

> [!WARNING] What the simulation does not show
> Our router never learns; only the bias moves (and, in runs C and D, our toy popularity). It shows the balancing mechanism, not its effect on model quality. For quality we need DeepSeek's own experiments, below.

### What balance costs in quality

The bias rule balances over the whole **batch**. The auxiliary loss in V2 balanced each **sequence**. That difference turns out to matter.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §4.5.3 · page 27
> [![The batch-wise versus sequence-wise load balance discussion: batch-wise balancing is more flexible and lets experts specialise by domain; validation losses of 2.258 with a sequence-wise auxiliary loss against 2.253 with the auxiliary-loss-free method on 1B MoE models](/img/deepseek/papers/p2-v3-batchwise.png)](/img/deepseek/papers/p2-v3-batchwise.png)
>
> **What it says:** Batch-wise balancing "does not enforce in-domain balance on each sequence", and this flexibility "allows experts to better specialize in different domains". On 1B MoE models, validation loss is 2.258 with a sequence-wise auxiliary loss, 2.253 with the auxiliary-loss-free method, and 2.253 with a batch-wise auxiliary loss.
>
> **What it means:**
>
> Think of a sequence of Python code. A per-sequence rule forces it to spread its tokens over all experts, including those that learned poetry. A per-batch rule only asks that the whole batch, code and poetry together, comes out even. Then the code experts can take most of a code sequence.
>
> The last result is the honest part. A batch-wise auxiliary loss does as well as the bias method (2.253 both). So the gain comes mainly from balancing over the batch. The bias method is one clean way to do that, with no gradient into the model.

The report's ablation (Table 5, page 27) compares the two methods on larger models. At 228.7B total parameters and 578B training tokens, the auxiliary-loss-free model scores higher on most benchmarks. For example, HumanEval goes from 40.2 to 46.3 and GSM8K from 70.7 to 74.5. It is not better everywhere: MMLU drops from 68.3 to 67.2.

V3 still keeps a very weak sequence-wise loss as a safety net.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §2.1.2 · page 9
> [![The complementary sequence-wise auxiliary loss, equations 17 to 20, used to prevent extreme imbalance within any single sequence, with an extremely small balance factor](/img/deepseek/papers/p2-v3-seqloss.png)](/img/deepseek/papers/p2-v3-seqloss.png)
>
> **What it says:** "to prevent extreme imbalance within any single sequence, we also employ a complementary sequence-wise balance loss", with a balance factor "assigned an extremely small value".
>
> **What it means:**
>
> This is the same $$f_i P_i$$ loss we worked through above, with two V3 changes. $$T$$ is the length of one sequence. And the affinities are first divided by their sum (Equation 19), because sigmoid scores do not add up to 1 on their own.
>
> With $$\alpha = 0.0001$$ it hardly pulls on the model. It only matters if one sequence sends nearly all its tokens to one expert, which could stall a GPU in serving. The bias does the real work.

> [!TIP] In production
> Balance during training does not guarantee balance while serving: real traffic can be all code, or all one language. DeepSeek serves V3 with redundant copies of hot experts, and its open EPLB library decides which experts to copy and where to place them. [LLM Inference, Part 7](llm-inference-7-beyond-one-gpu.md) walks through this; [Part 7](deepseek-7-running-deepseek.md) of this series covers DeepSeek's own serving.

## MLA: a small memory for long conversations

The second dashed box in Figure 2 is attention. Its problem is memory, not arithmetic.

With plain multi-head attention, every head in every layer stores one key and one value per token. [Attention, Part 2](attention-2-mqa-gqa-mla.md) builds MHA, GQA, MQA and MLA from scratch and compares them; here we focus on V3's real numbers.

### Compress, then cache the compressed version

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V2 · §2.1.2 · page 7
> [![The low-rank key-value joint compression section of the DeepSeek-V2 paper: equations 9 to 11 compress the hidden state into a latent vector c and rebuild keys and values from it; MLA only needs to cache c, and the up-projections can be absorbed](/img/deepseek/papers/p2-v2-mla-cache.png)](/img/deepseek/papers/p2-v2-mla-cache.png)
>
> **What it says:** $$c^{KV}_t = W^{DKV} h_t$$, $$k^C_t = W^{UK} c^{KV}_t$$, $$v^C_t = W^{UV} c^{KV}_t$$. "During inference, MLA only needs to cache $$c^{KV}_t$$". Also, since $$W^{UK}$$ and $$W^{UV}$$ "can be absorbed" into the query and output matrices, "we even do not need to compute keys and values out for attention."
>
> **What it means:**
>
> Instead of storing 128 keys and 128 values for each token, MLA stores one short vector $$c$$. It rebuilds the keys and values from it when needed. It is like keeping a zip file instead of the folder: smaller on disk, and the folder can be rebuilt.
>
> The second sentence is the clever part. You do not even need to unzip. The rebuild matrices can be moved into the query side and the output side, so attention works directly on the compressed vectors. We check that below with real V3 shapes.

> [!DEFINITION] Latent vector
> A short vector that is not a key or a value itself, but from which keys and values can be computed by a fixed matrix. "Low-rank" means it is much shorter than what it replaces: 512 numbers in V3, against 128 heads × 128 numbers × 2 for full keys and values.

In V3's sizes: the hidden state has 7,168 numbers. $$W^{DKV}$$ maps it to $$d_c = 512$$ (`kv_lora_rank`). $$W^{UK}$$ and $$W^{UV}$$ map those 512 numbers back to 128 keys and 128 values of 128 numbers each. Queries get their own compression to 1,536 numbers (`q_lora_rank`). That does not shrink the cache; the paper says it saves activation memory in training.

{{FIG:mla_flow|MLA in one DeepSeek-V3 layer, with every size from config.json. The hidden state is compressed into a 512-number KV latent and a 64-number RoPE key. Only those two (highlighted) are cached. Keys and values for 128 heads are computed from the latent, and queries come from their own 1,536-number latent.}}

### Absorption: never rebuild the keys at all

Why can the rebuild be skipped? An attention score is a dot product of a query $$q$$ with a key $$k = W^{UK} c$$. Matrix products can be regrouped:

$$
q^{\top} \big(W^{UK} c\big) \;=\; \big(W^{UK\top} q\big)^{\top} c
$$

- $$q$$ is one head's query (128 numbers) for the token being written.
- $$c$$ is one cached latent (512 numbers) of an earlier token.
- $$W^{UK}$$ is that head's slice of the key up-projection, a 128 × 512 matrix.

The left side rebuilds the key for **every** cached token, then compares. The right side moves the query into the 512-number latent space **once**, then compares it with every cached $$c$$ directly. Same number, very different cost. The value side works the same way: take the weighted sum of the latents first, then apply $$W^{UV}$$ once.

{{FIG:absorb|Two orders for the same attention score. Naive: rebuild a 128-number key from each cached latent, then compare it with the query. Absorbed: move the query into the 512-number latent space once per step, then compare it with each cached latent directly.}}

A tiny example with numbers you can check. Let $$W^{UK} = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}$$, $$c = (3, 1)$$ and $$q = (1, 1)$$. Naive: the key is $$(1 \cdot 3 + 2 \cdot 1,\; 0 \cdot 3 + 1 \cdot 1) = (5, 1)$$, and the score is $$5 + 1 = 6$$. Absorbed: $$W^{UK\top} q = (1, 3)$$, and the score is $$1 \cdot 3 + 3 \cdot 1 = 6$$.

`part2_kv.py` runs both orders with V3's real per-head shapes, random weights and 64 cached tokens. The shapes are 128 heads, latent 512, content head 128, RoPE part 64 and value head 128. The core of both methods:

```python
# Method 1: naive. Rebuild keys and values of every head for every cached token.
k_c = torch.einsum('hdc,tc->htd', W_uk, c_kv)                    # (128, T, 128)
v_c = torch.einsum('hdc,tc->htd', W_uv, c_kv)                    # (128, T, 128)
scores1 = (torch.einsum('hd,htd->ht', q_c, k_c) + q_r @ k_r.T) * scale
o1 = torch.einsum('ht,htd->hd', scores1.softmax(-1), v_c)        # (128, 128)

# Method 2: absorbed. Move W^UK onto the query side and W^UV after the weighted sum.
q_lat = torch.einsum('hd,hdc->hc', q_c, W_uk)                    # (128, 512): query seen in latent space
scores2 = (q_lat @ c_kv.T + q_r @ k_r.T) * scale                 # score straight against the cache
x = scores2.softmax(-1) @ c_kv                                   # (128, 512): weighted sum of latents
o2 = torch.einsum('hc,hdc->hd', x, W_uv)                         # apply W^UV once, at the end
```

```text
== MLA, naive vs absorbed, V3 shapes, random weights ==
  cache: c_kv (64, 512) + k_r (64, 64) = 576 numbers per token
  naive  : rebuilt keys + values hold 2,097,152 numbers for 64 tokens
  absorbed: reads only the cache, 36,864 numbers for 64 tokens
  largest score difference 4.9e-15; largest output difference (relative) 1.5e-15
  multiply-adds per cached token per layer: naive 16,818,176, absorbed 139,264
```

The two methods agree to 15 decimal places (float64), so absorption is exact, not an approximation. The cost line is the reason V3's inference code uses it (`attn_impl = "absorb"`). Rebuilding keys and values costs about 16.8 million multiply-adds per cached token per layer. Working in latent space costs 139,264, about 120 times less.

### Why position needs its own small key

There is one catch, and it explains the strange "+ 64" in V3's cache.

> [!DEFINITION] RoPE (rotary position embedding)
> The way most modern models tell attention where each token sits. Before the query and key are compared, each pair of their numbers is rotated by an angle that depends on the token's position. The score then depends on how far apart the two tokens are.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V2 · §2.1.3 · page 8
> [![The decoupled RoPE section: RoPE is incompatible with low-rank KV compression because a position-dependent rotation would sit between W^Q and W^UK, so keys for all prefix tokens would have to be recomputed; the fix uses extra multi-head queries and a shared key to carry RoPE](/img/deepseek/papers/p2-v2-rope.png)](/img/deepseek/papers/p2-v2-rope.png)
>
> **What it says:** "RoPE is incompatible with low-rank KV compression." If keys were rotated, a position-dependent matrix would sit between $$W^Q$$ and $$W^{UK}$$, so "we must recompute the keys for all the prefix tokens during inference". The fix is a "decoupled RoPE strategy" with extra query parts and one shared key that carry position.
>
> **What it means:**
>
> Absorption works because $$W^{UK}$$ is the **same** matrix for every cached token, so it can be moved to the query side once. A rotation that depends on each token's position breaks that: each cached token would need its own rotated matrix, and the regrouping fails.
>
> So V3 splits each key into two parts. The **content** part (128 numbers per head) comes from the latent and has no rotation, so it can be absorbed. The **position** part is a separate 64-number key $$k^R$$, rotated by RoPE, and shared by all 128 heads. That small key must be cached too, because it cannot be rebuilt from $$c$$.

That gives the full cache entry per token per layer:

$$
\text{MLA cache per token} = (d_c + d^R_h) \times l \times b = (512 + 64) \times 61 \times 2 = 70{,}272 \text{ bytes}
$$

- $$d_c = 512$$ is the latent size and $$d^R_h = 64$$ the shared RoPE key size, both from the config.
- $$l = 61$$ is the number of layers, and $$b = 2$$ bytes per number in BF16.

### How small is 70,272 bytes?

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V2 · §2.1.4, Table 1 · page 9
> [![Table 1 of DeepSeek-V2: KV cache per token in elements for MHA (2 n_h d_h l), GQA (2 n_g d_h l), MQA (2 d_h l) and MLA ((d_c + d_h^R) l, about 9/2 d_h l), with MLA marked stronger; the caption says it equals GQA with only 2.25 groups](/img/deepseek/papers/p2-v2-table1.png)](/img/deepseek/papers/p2-v2-table1.png)
>
> **What it says:** MHA stores $$2 n_h d_h l$$ numbers per token, GQA $$2 n_g d_h l$$, MQA $$2 d_h l$$, and MLA $$(d_c + d^R_h) l \approx \tfrac{9}{2} d_h l$$. MLA's cache is "equal to GQA with only 2.25 groups", "but its performance is stronger than MHA."
>
> **What it means:**
>
> With $$d_h = 128$$, MLA's $$512 + 64 = 576$$ numbers equal $$4.5 \times 128$$. A GQA model with $$n_g$$ groups stores $$2 n_g \times 128$$. Set them equal and $$n_g = 2.25$$. So MLA costs about as much memory as GQA with two and a quarter key/value heads.
>
> The bold claim is the last column: MLA beats full MHA in their tests (Appendix D.2 of the V2 paper). Keep in mind that this comparison comes from DeepSeek's own small and large MoE models, not an independent study.

Here are the numbers for V3's shapes, from `part2_kv.py`. "MHA, V3 head shapes" is what V3 would store if it kept full keys (192 numbers: 128 content + 64 RoPE) and values (128) for every head.

```text
== KV cache per token, 61 layers ==
  MLA stores d_c + d_R = 512 + 64 = 576 numbers per layer; x 61 layers x 2 bytes = 70,272 bytes
  MHA, V3 head shapes (K 192, V 128 per head)           4,997,120 B/token    4880.0 KiB   71.11x MLA   128K tokens:  610.00 GiB
  MHA, textbook (2 x 128 heads x 128)                   3,997,696 B/token    3904.0 KiB   56.89x MLA   128K tokens:  488.00 GiB
  GQA, 8 KV groups x 128                                  249,856 B/token     244.0 KiB    3.56x MLA   128K tokens:   30.50 GiB
  MQA, 1 KV head x 128                                     31,232 B/token      30.5 KiB    0.44x MLA   128K tokens:    3.81 GiB
  MLA, BF16 (512 latent + 64 RoPE key)                     70,272 B/token      68.6 KiB    1.00x MLA   128K tokens:    8.58 GiB
  MLA, FP8 layout (512 B + 16 B scales + 128 B RoPE)       40,016 B/token      39.1 KiB    0.57x MLA   128K tokens:    4.88 GiB
```

{{FIG:kv_bars|KV cache per token for attention types with DeepSeek-V3's 61 layers and 128 heads, and the cache for one 128K-token conversation. MLA in BF16 needs 68.6 KiB per token and 8.6 GiB at 128K; full MHA with V3's head shapes would need 4,880 KiB and 610 GiB.}}

Read the 128K column. One full-length conversation with plain MHA would need 610 GiB, more than eight 80 GB GPUs hold together (640 GB, or 596 GiB). With MLA it needs 8.6 GiB. That is the difference between serving one long chat per server and serving dozens.

DeepSeek's later paper on V3's hardware lists the same 70.272 KB for V3 (2505.09343, Table 1). The same table gives 327.680 KB for Qwen-2.5 72B and 516.096 KB for LLaMA-3.1 405B, both GQA models, all in BF16. The FP8 row uses the cache layout documented in DeepSeek's FlashMLA kernels. It keeps 512 bytes of FP8 latent, 16 bytes of scales and the 64-number RoPE key in BF16: 656 bytes per layer.

> [!WARNING] The cache is small, but attention is not free
> MLA shrinks memory, not the number of tokens each new token attends to. Every step still reads the whole cache, so cost grows with context length. Cutting that growth is the job of DeepSeek Sparse Attention, in [Part 5](deepseek-5-sparse-attention-and-v3-2.md).

<details><summary>Research notes: where V2's "93.3% smaller KV cache" comes from</summary>

The V2 abstract says V2 "reduces the KV cache by 93.3%" against DeepSeek 67B. The arithmetic only works with mixed precisions. V2 caches $$(512 + 64) \times 60 = 34{,}560$$ numbers per token, and the paper deploys it with the cache quantised to about 6 bits on average, so 25,920 bytes. DeepSeek 67B uses GQA with 8 KV heads: $$2 \times 8 \times 128 \times 95 = 194{,}560$$ numbers, 389,120 bytes in BF16. And $$1 - 25{,}920 / 389{,}120 = 93.3\%$$. At equal BF16 precision the saving is $$1 - 69{,}120 / 389{,}120 = 82.2\%$$. The paper does not spell this out; the derivation is from the research dossier for this series, and it matches to the decimal.

</details>

## Multi-token prediction: learning to see two tokens ahead

The last new piece sits outside the main stack. It changes how V3 is trained, and it can make writing faster.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §2.2, Figure 3 · page 10
> [![Figure 3 of the DeepSeek-V3 report: the main model predicts the next token; MTP module 1 takes the main model's representation plus the embedding of the next input token, passes them through RMSNorms, a linear projection and one transformer block, and predicts the token after next; embedding and output head are shared](/img/deepseek/papers/p2-v3-mtp-figure.png)](/img/deepseek/papers/p2-v3-mtp-figure.png)
> [![The MTP modules paragraph: each module has a shared embedding layer, a shared output head, a transformer block and a projection matrix](/img/deepseek/papers/p2-v3-mtp-modules.png)](/img/deepseek/papers/p2-v3-mtp-modules.png)
>
> **What it says:** "We keep the complete causal chain for the prediction of each token at each depth." The $$k$$-th MTP module has "a shared embedding layer", "a shared output head", "a Transformer block" and "a projection matrix" $$M_k$$ of size $$d \times 2d$$.
>
> **What it means:**
>
> Look at one column of the figure. The main model reads $$t_1$$ and predicts $$t_2$$, as usual. The MTP module takes the main model's last hidden state at $$t_1$$ **plus** the embedding of the real $$t_2$$. It mixes them, runs one more transformer block, and predicts $$t_3$$.
>
> "Causal chain" is the key phrase. An earlier design (Gloeckle et al., 2024) predicted $$t_2$$ and $$t_3$$ with independent heads from the same state. DeepSeek's module first learns what $$t_2$$ actually is, then guesses $$t_3$$. It is the difference between guessing two words blind and guessing the second after hearing the first.

In symbols, for depth $$k$$ (V3 uses only $$k = 1$$):

$$
h'^{k}_i = M_k\big[\mathrm{RMSNorm}(h^{k-1}_i);\ \mathrm{RMSNorm}(\mathrm{Emb}(t_{i+k}))\big], \qquad
P^{k}_{i+k+1} = \mathrm{OutHead}\big(\mathrm{TRM}_k(h'^{k}_i)\big)
$$

$$
\mathcal{L} = \mathcal{L}_{\mathrm{main}} + \frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}^{k}_{\mathrm{MTP}}
$$

- $$h^{k-1}_i$$ is the hidden state at position $$i$$ from the level below; for $$k = 1$$ it is the main model's output.
- $$\mathrm{Emb}(t_{i+k})$$ is the embedding of the real token $$k$$ steps ahead; $$[\,\cdot\,;\,\cdot\,]$$ joins two vectors of 7,168 into one of 14,336.
- $$M_k$$ maps 14,336 numbers back to 7,168; $$\mathrm{TRM}_k$$ is one full transformer block (MLA plus an MoE layer in V3).
- $$P^{k}_{i+k+1}$$ is the predicted distribution for the token $$k+1$$ steps ahead; $$\mathcal{L}^{k}_{\mathrm{MTP}}$$ is its cross-entropy loss.
- $$D = 1$$ is the number of MTP modules, and $$\lambda$$ is their weight in the total loss.

Our toy run prints which targets each part learns, on a 6-token sentence:

```text
== Which token each depth predicts (T = 6 tokens, D = 1) ==
  position 1 (The): main model predicts t2 = 'cat'  MTP module (also given the embedding of t2) predicts t3 = 'sat'
  position 2 (cat): main model predicts t3 = 'sat'  MTP module (also given the embedding of t3) predicts t4 = 'on'
  position 3 (sat): main model predicts t4 = 'on'   MTP module (also given the embedding of t4) predicts t5 = 'the'
  position 4 ( on): main model predicts t5 = 'the'  MTP module (also given the embedding of t5) predicts t6 = 'mat'
```

{{FIG:mtp|Multi-token prediction on a short sentence. At each position the main model predicts the next token. The MTP module takes the main model's hidden state and the embedding of that next token, and predicts the token after it.}}

The weight $$\lambda$$ was 0.3 for the first 10T training tokens and 0.1 for the last 4.8T (page 23). Take an illustrative main loss of 2.0 and MTP loss of 2.6; we chose these, they are not from the paper. The total is $$2.0 + 0.3 \times 2.6 = 2.78$$ early in training and $$2.0 + 0.1 \times 2.6 = 2.26$$ late. The extra signal is strong early and fades as training ends.

Why bother? The paper says MTP "densifies the training signals" and "may enable the model to pre-plan its representations". Its ablation (Table 4, page 26) adds one MTP module to a 228.7B-parameter MoE model trained on 540B tokens. Most scores go up, for example HumanEval from 44.5 to 53.7 and GSM8K from 72.3 to 74.0. A few go down slightly, such as MMLU from 67.5 to 66.6.

### At inference: drop it, or use it as a drafter

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §2.2 and §5.4.3 · pages 11 and 35
> [![The MTP in inference paragraph: the MTP modules can be discarded, or repurposed for speculative decoding](/img/deepseek/papers/p2-v3-mtp-inference.png)](/img/deepseek/papers/p2-v3-mtp-inference.png)
> [![The MTP evaluation paragraph: the acceptance rate of the second token prediction ranges between 85% and 90% across generation topics, giving 1.8 times tokens per second](/img/deepseek/papers/p2-v3-mtp-accept.png)](/img/deepseek/papers/p2-v3-mtp-accept.png)
>
> **What it says:** "we can directly discard the MTP modules and the main model can function independently", or "repurpose these MTP modules for speculative decoding". The "acceptance rate of the second token prediction ranges between 85% and 90%", "delivering 1.8 times TPS (Tokens Per Second)."
>
> **What it means:**
>
> The module was built to improve training. Throwing it away costs nothing: the main model is complete without it. But it already predicts the token after next, which is exactly what a **draft** model in speculative decoding does ([LLM Inference, Part 4](llm-inference-4-speculative-decoding.md)).
>
> So at each step, the main model writes token $$n$$ and the MTP module guesses token $$n+1$$. In the next pass, the main model checks that guess while it computes the following token. If the guess was right, two tokens come out of one pass.

With one draft token accepted with probability $$p$$, each step yields $$1 + p$$ tokens on average:

```text
== Tokens per decoding step with one draft token ==
  acceptance 85%: 1 + 0.85 = 1.85 tokens per step (upper bound on the speed-up)
  acceptance 90%: 1 + 0.90 = 1.90 tokens per step (upper bound on the speed-up)
  reported by DeepSeek: 1.8x tokens per second; the gap is the cost of running the MTP layer and verifying
```

The reported 1.8× sits just under our 1.85 to 1.90 ceiling, which is what you would expect once the extra layer's cost is paid. In the V4 releases from July 2026, DeepSeek replaced this drafter with a new one called DSpark; [Part 6](deepseek-6-v4-and-v4-1.md) covers it.

## The 671B and 37B, counted from config.json

Now we can count everything. `part2_params.py` reads the real V3 `config.json` and counts each weight matrix with a formula (Method 1). Then it builds `DeepseekV3ForCausalLM` from the same config on PyTorch's "meta" device, which creates every tensor's shape but no memory. It counts what transformers made (Method 2). The heart of Method 1 is one line for attention:

```python
    q = d * nh * (nope + rope) if qr is None else d * qr + qr + qr * nh * (nope + rope)
    attn = q + d * (kvr + rope) + kvr + kvr * nh * (nope + vd) + nh * vd * d
```

Read it as the MLA flow figure, box by box. First the query down-projection (7,168 × 1,536) and its norm, then the query up-projection to 128 heads of 192. Next the KV down-projection to 512 + 64 and its norm, and the KV up-projection to 128 heads of 128 + 128. Last comes the output matrix. That is 187,107,328 parameters per layer. The output:

```text
== Per-piece sizes ==
  MLA attention, one layer         187,107,328
  dense SwiGLU FFN, one layer      396,361,728
  one expert (SwiGLU)               44,040,192
  router, one MoE layer              1,835,264
  dense layers: 3, MoE layers: 58

== Method 1: our formula, total parameters ==
  embedding                         0.927B    0.14%
  output head                       0.927B    0.14%
  attention (MLA), all layers      11.414B    1.70%
  dense FFN, first layers           1.189B    0.18%
  routed experts                  653.909B   97.45%
  shared experts                    2.554B    0.38%
  routers (incl. bias)              0.106B    0.02%
  RMSNorm weights                   0.001B    0.00%
  TOTAL main model                671.026B   (671,026,419,200)

== Method 2: transformers DeepseekV3ForCausalLM on the meta device ==
  parameters + balancing bias    671.026B   (671,026,419,200)
  equal to Method 1: True
```

The two counts agree to the last parameter: 671,026,419,200. The paper's "671B" is exact, not a rounding of something larger. And 97.45% of the model is routed experts. Attention, the part most explanations of transformers spend their time on, is 1.7%.

Now count what one token touches: everything except the 248 routed experts it did not pick in each MoE layer.

```text
== Active parameters for one token ==
  embedding                         0.927B
  output head                       0.927B
  attention (MLA), all layers      11.414B
  dense FFN, first layers           1.189B
  routed experts                   20.435B
  shared experts                    2.554B
  routers (incl. bias)              0.106B
  RMSNorm weights                   0.001B
  TOTAL active                     37.552B
  without the input embedding (a table lookup, not a matrix multiply):   36.626B
  share of all parameters used per token: 5.6%
```

{{FIG:params|Where DeepSeek-V3's parameters are, from our count of config.json. Top: all 671.0 billion; routed experts are 97.4 percent. Bottom: the 37.6 billion one token uses; routed experts are 54 percent, attention 30 percent.}}

Per token, the picture flips. Routed experts drop to 20.4B (8 of 256 in 58 layers), and attention becomes the second-largest share at 30%. So the paper's "37B" checks out: 37.55B if you count the input embedding, 36.63B if you do not. DeepSeek's weight notes say 36.7B "including 0.9B for Embedding and 0.9B for the output Head"; we could not reproduce that exact figure: with both counted, we get 37.55B, a gap of about 0.85B.

### Why Hugging Face says 685B

```text
== The MTP module (one extra layer, num_nextn_predict_layers = 1) ==
  enorm + hnorm + eh_proj (7168 x 14336) + one full MoE layer + final norm =   11.610B
  plus its stored copies of the embedding and output head            =    1.853B
  checkpoint on Hugging Face:  671.026B +   11.610B +    1.853B =  684.490B
```

The MTP module is a full MoE layer (11.5B of its 11.61B) plus the $$M_k$$ projection and norms. In the checkpoint it also stores its own copies of the embedding and output head. That sums to 684.49B. DeepSeek's README rounds it to "685B, which includes 671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) Module weights." The 685B on the model page is the same model plus its drafter.

Finally, a second real config as a check. The cached DeepSeek-V2-Lite config gives:

```text
== Check on a second real config: DeepSeek-V2-Lite-Chat (published 15.7B total, 2.4B active) ==
  Method 1 total   15.706B, Method 2 total   15.706B, equal: True
  Method 1 active    2.661B, without the input embedding    2.451B
```

The total matches the paper's 15.7B. The active count matches its 2.4B when the input embedding is left out, which suggests that is the convention DeepSeek used for V2-Lite.

> [!TIP] In production
> Parameter count drives memory, active count drives speed. All 671B must be loaded somewhere. V3 was released in FP8, one byte per parameter, so that is about 671 GB of weights. But each token reads only about 37B of them. That is why V3 needs many GPUs, yet writes fast once loaded. [Part 7](deepseek-7-running-deepseek.md) does the full memory arithmetic.

## What this design costs

None of these ideas is free. Many experts mean a large memory footprint: the paper's own limits section says "the recommended deployment unit for DeepSeek-V3 is relatively large, which might pose a burden for small-sized teams" (page 35). Expert parallelism needs fast networks and careful kernels. Node-limited routing gives up some choice of experts for less traffic. MLA saves memory but needs absorption tricks and special kernels to be fast.

The payoff is a model that does the work of a roughly 37B model per token, while it holds the knowledge of a 671B one. Its cache is small enough for 128K-token conversations. The same skeleton, with small changes, carries every DeepSeek model through V3.2.

> [!TAKEAWAYS]
> - DeepSeek-V3 has 61 layers: 3 with a dense FFN and 58 with DeepSeekMoE. Counted from its config.json, the main model has exactly **671,026,419,200** parameters; one token uses **37.55B** (36.63B without the input embedding).
> - An MoE layer gives each token **1 shared + 8 of 256 routed** experts. Since $$9 \times 2{,}048 = 18{,}432$$, a token does the same arithmetic as in a dense layer, with 28.6 times the parameters available.
> - **Fine-grained experts** give many more possible teams: $$\binom{256}{8} \approx 4.1 \times 10^{14}$$. The **shared expert** holds common knowledge so routed experts can specialise.
> - V3 scores experts with a **sigmoid**, normalises the 8 chosen gates, and scales them by 2.5 (`routed_scaling_factor`).
> - **Node-limited routing**: each token uses at most **4 of 8 nodes**, chosen by the sum of each node's 2 best scores. The reason: NVLink (160 GB/s) is about 3.2 times faster than InfiniBand (50 GB/s).
> - **Auxiliary-loss-free balancing**: a per-expert bias, moved by $$\gamma = 0.001$$ each step, changes which experts are chosen but never the gate values. In our simulation it cut MaxVio from 3.88 to 0.35 in 100 steps.
> - **MLA** caches a 512-number latent plus a 64-number RoPE key per layer: **70,272 bytes per token** in BF16. That is 71 times less than MHA with V3's head shapes, and 8.6 GiB for a 128K-token conversation.
> - **Absorption** is exact: scoring against the latent gives the same result as rebuilding keys (our check: 4.9e-15). It needs about 120 times fewer multiply-adds per cached token.
> - **MTP** trains one extra layer to predict the token after next. At inference it drafts tokens with **85% to 90%** acceptance for **1.8 times** tokens per second. It is why the checkpoint is **685B**.

Next, [Part 3](deepseek-3-training-v3.md) follows this architecture through training. It covers the 14.8 trillion tokens, FP8 arithmetic and the DualPipe schedule that hides the all-to-all traffic described here. It also covers the cluster, and what the famous training cost does and does not include.

## References

- Dai et al. (2024), *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
- DeepSeek-AI (2024), *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- DeepSeek-AI (2024), *DeepSeek-V3 Technical Report*. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- Wang et al. (2024), *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*. [arXiv:2408.15664](https://arxiv.org/abs/2408.15664)
- DeepSeek-AI (2025), *Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures*. [arXiv:2505.09343](https://arxiv.org/abs/2505.09343)
- Shazeer et al. (2017), *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- DeepSeek-V3 model card, config.json and weight notes: [huggingface.co/deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3), [README_WEIGHTS.md](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md)
- DeepSeek-V3 reference inference code: [inference/model.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)
- FlashMLA (KV cache layouts): [github.com/deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- Our scripts: `code/deepseek/part2_params.py`, `part2_router.py`, `part2_balance.py`, `part2_kv.py`, `part2_mtp.py`
