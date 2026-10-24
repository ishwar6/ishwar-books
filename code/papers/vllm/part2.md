---
title: "Why Memory Is the Bottleneck"
description: "How a language model writes text, why one GPU must serve many requests at once, and where the memory for those requests goes. We read §2, §3 and Figure 3, with runs on a real model."
part: 2
covers: "§2, §3, Figure 3"
date: 2026-10-20
tags: [llm-serving, inference, kv-cache, gpu-memory, batching]
---

Part 1 showed the problem in one picture. A GPU that serves a language model runs out of memory long before it runs out of speed. This part explains why. We follow the paper's §2 and §3, which describe how a model writes text, how a server groups requests, and where the memory goes.

Each piece below shows the exact lines of the paper, then explains them.

> [!DEFINITION] Definition boxes
> A yellow box like this explains one technical word, the first time that the word occurs. If you know the word, skip the box.

Here is the whole vLLM system first, so that you can see where this part fits.

{{FIG:p2_arch|The vLLM system, redrawn from the paper's Figure 4. Requests come in on the left. The scheduler, highlighted, picks the requests that run together in each step. Its KV cache manager decides where each request's memory lives. The workers run the model on the GPUs.}}

This part explains why the scheduler wants many requests in every step, and why memory stops it. Part 3 opens the box inside it, the KV cache manager.

## How a language model scores a sentence {§2.1}

A language model does one job. It reads some text and says how likely each possible next word is. Everything else, from chat to code completion, is built on that one job. The paper starts there, because the way the model does this job decides how much memory it needs later.

> [!PAPER] · §2.1 · page 3
> [![The opening of §2.1 with Equation 1, with product of conditional probabilities and autoregressive decomposition highlighted](/img/papers/vllm/p2-eq1.png)](/img/papers/vllm/p2-eq1.png)
>
> **What it says:** language modeling is "to model the probability of a list of tokens". Because language has a natural order, the probability of the whole list is written as the "product of conditional probabilities", also called the "autoregressive decomposition" (Eq. 1).
>
> **What it means:**
>
> Think of a guessing game. A friend reads a sentence to you one word at a time, and before each word you bet on what comes next. "Four ..." could go anywhere. "Four score and seven years ..." almost surely continues with "ago". Each bet uses all the words you heard so far, never the words still to come.
>
> Eq. 1 turns that game into arithmetic. The chance of the whole sentence is the chance of the first word, times the chance of the second word given the first. Then you multiply by the chance of the third word given the first two, and so on. This rule is exact; it is not an approximation. Any probability of a sequence can be written this way.
>
> The rule has a cost that the rest of the paper pays. To write new text, the model must pick one word, add it, and only then score the next word. The order of the factors becomes the order of the work.

> [!DEFINITION] Token
> The unit of text that a model reads and writes. A token is often a whole word, sometimes a piece of a word or a punctuation mark. The OPT models in this paper use about 50,000 different tokens.

> [!DEFINITION] Conditional probability
> The chance of one event when you already know another. $$P(B \mid A)$$ reads "the probability of B given A". The chance that a card is a heart is 1 in 4. Given that it is red, the chance is 1 in 2.

Here is Eq. 1 as the paper writes it:

$$
P(x) = P(x_1) \cdot P(x_2 \mid x_1) \cdots P(x_n \mid x_1, \ldots, x_{n-1})
$$

Read it from left to right. The symbol $$x$$ is the whole list of tokens, and $$x_1$$ to $$x_n$$ are the tokens in order. The left side, $$P(x)$$, is the probability of the whole list. Each factor $$P(x_i \mid x_1, \ldots, x_{i-1})$$ is the probability of token $$i$$, given all the tokens before it. The dots mean "continue the same pattern up to token $$n$$".

A tiny example with numbers you can check by hand. Suppose a model gives "The" a probability of 0.1 as a first word. It gives "cat" 0.02 after "The", and "sat" 0.3 after "The cat". The sentence "The cat sat" then has probability $$0.1 \times 0.02 \times 0.3 = 0.0006$$. Three tokens, three factors.

Now the real thing. We give OPT-125M, the smallest model of the family that the paper serves, the sentence "Four score and seven years ago our fathers". We compute the factors in two ways. Method 1 runs the model once for each token and reads one probability each time. Method 2 runs the model once on the whole sentence and reads every probability from that one pass:

```python
@torch.no_grad()
def chain_rule_one_by_one(ids):
    """Method 1: for each position i, run the model on x_1..x_{i-1} and read P(x_i | x_1..x_{i-1})."""
    probs = []
    for i in range(1, ids.shape[1]):
        logits = model(ids[:, :i]).logits[0, -1]           # the model sees only the tokens before position i
        probs.append(torch.softmax(logits, -1)[ids[0, i]].item())
    return probs


@torch.no_grad()
def chain_rule_one_pass(ids):
    """Method 2: one pass over the whole sentence. Thanks to the causal mask, row i of the output already
    holds P(x_{i+1} | x_1..x_i), so every conditional comes out at once (this is the prompt phase)."""
    logits = model(ids).logits[0, :-1]                      # (n - 1, vocabulary)
    return torch.softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0].tolist()
```

```text
== 1. Eq. 1: the probability of a sentence is a product of next-token probabilities
token      P(token | earlier)
'Four'               1.23e-05
' score'             5.54e-05
' and'               2.99e-02
' seven'             1.69e-02
' years'             1.21e-03
' ago'               7.47e-02
' our'               1.68e-03
' fathers'           5.16e-04
product, Method 1 (8 passes): 2.710e-23
product, Method 2 (1 pass):     2.710e-23
library loss 6.4953 -> exp(-loss x 8) = 2.710e-23; max difference of the conditionals 2.5e-07
```

{{FIG:p2_chain|Eq. 1 on our sentence. Each token gets one factor: its probability given the tokens to its left. The product of the eight factors is the probability of the whole sentence.}}

Both methods give the same eight factors, to within $$2.5 \times 10^{-7}$$, and the same product. The library's own training loss gives that product too. The small model is not sure of "ago" (0.0747) because it saw few old speeches, but that does not matter here.

What matters is the difference in work. Method 2 got all eight factors from one pass, because every token of the sentence was already known. That is how the model reads a prompt. When the model writes, the next token is not known yet, so it must work like Method 1: one pass per token. This difference between reading and writing is the root of everything in this part.

## Queries, keys and values {§2.1}

> [!PAPER] · §2.1 · page 3
> [![The paragraph on self-attention with Equation 2, with self-attention layers and query, key, and value vectors highlighted](/img/papers/vllm/p2-qkv.png)](/img/papers/vllm/p2-qkv.png)
>
> **What it says:** the most important part of a Transformer language model is its "self-attention layers". Each layer first turns the vector of each position into "query, key, and value vectors" with three weight matrices (Eq. 2).
>
> **What it means:**
>
> Inside the model, each token is a list of numbers, a vector. For OPT-13B the list has 5,120 numbers. Eq. 2 says that each attention layer makes three new vectors from it, by multiplying with three learned matrices. The **query** says what this token looks for. The **key** says what this token offers to others. The **value** is what it hands over when another token picks it.
>
> A library catalogue is a fair picture. Your question is the query. The label on the spine of each book is its key. The text inside is its value. You compare your question with every label, then read mostly from the books that match.

> [!DEFINITION] Vector and hidden state
> A vector is an ordered list of numbers. A **hidden state** is the vector that the model keeps for one token between its layers. In the paper, $$x_i \in \mathbb{R}^{d}$$ means "the hidden state of token $$i$$ is a list of $$d$$ real numbers".

The equation is three matrix products, one per vector:

$$
q_i = W_q x_i, \qquad k_i = W_k x_i, \qquad v_i = W_v x_i
$$

Read it this way. The input $$x_i$$ is the hidden state of token $$i$$. The three matrices $$W_q$$, $$W_k$$ and $$W_v$$ are weights that the model learned in training. They are the same for every token and every request. The outputs $$q_i$$, $$k_i$$ and $$v_i$$ are the query, key and value of token $$i$$.

A tiny example. Take $$d = 2$$, the hidden state $$x_i = (1, 2)$$, and the key matrix with rows $$(1, 0)$$ and $$(1, 1)$$. Then $$k_i = (1 \cdot 1 + 0 \cdot 2,\ 1 \cdot 1 + 1 \cdot 2) = (1, 3)$$. In OPT-13B each matrix is 5,120 by 5,120, and every layer of the 40 does this for every token.

> [!PAPER] · §2.1 · page 3
> [![The paragraph on attention scores with Equation 3, with the key vectors before it and weighted average over the value vectors highlighted](/img/papers/vllm/p2-eq3.png)](/img/papers/vllm/p2-eq3.png)
>
> **What it says:** the layer multiplies the query at one position with "the key vectors before it" to get attention scores, and computes the output as the "weighted average over the value vectors" (Eq. 3).
>
> **What it means:**
>
> Token $$i$$ compares its query with the key of every token up to itself. A good match gives a large score. The scores are turned into weights that are positive and add up to 1. The output is the values mixed with those weights. So each token gathers information mostly from the earlier tokens that matter to it.
>
> Look at the words "before it". Token $$i$$ reads the keys and values of tokens 1 to $$i$$, and of no later token. This is why the keys and values can be kept. Once token 5 has its key and value, no later token will ever change them. They only get read again, at every later step. That stored set is the **KV cache**.

$$
a_{ij} = \frac{\exp\left(q_i^\top k_j / \sqrt{d}\right)}{\sum_{t=1}^{i} \exp\left(q_i^\top k_t / \sqrt{d}\right)}, \qquad o_i = \sum_{j=1}^{i} a_{ij} v_j
$$

Read the left half first. The term $$q_i^\top k_j$$ is the dot product of the query of token $$i$$ with the key of token $$j$$. You multiply the matching numbers and add them up.

Dividing by $$\sqrt{d}$$ keeps the numbers in a good range when vectors are long. The exponential makes every score positive. Dividing by the sum over $$t = 1$$ to $$i$$ makes the weights $$a_{ij}$$ add up to 1. This step is called a **softmax**. The right half, $$o_i$$, adds up the values $$v_j$$, each one multiplied by its weight.

The [Background page](background-the-kv-cache-and-paging.md) works one example by hand. A query $$(1, 0)$$ meets three keys, $$(1, 0)$$, $$(0, 1)$$ and $$(1, 1)$$. The scores are 0.707, 0 and 0.707, the weights 0.401, 0.198 and 0.401, and the output is $$(0.802, 0.599)$$. The two keys that match the query equally get equal weight.

The sums in Eq. 3 run up to $$i$$, and that is what fills memory. To write token 1,000, the model needs the keys and values of all 999 tokens before it, in every layer. They must sit in GPU memory, ready to read, for the whole life of the request.

> [!PAPER] · §2.1 · page 3
> [![The paragraph that says all other components are applied position-wise, with applied independently and position-wise highlighted](/img/papers/vllm/p2-positionwise.png)](/img/papers/vllm/p2-positionwise.png)
>
> **What it says:** apart from attention, every other part of the model, such as the embedding, the feed-forward layer and layer normalization, is "applied independently position-wise", in the form $$y_i = f(x_i)$$.
>
> **What it means:**
>
> Most of the model treats each token on its own. A feed-forward layer takes the vector of token 5 and gives back a new vector for token 5, without looking at token 4. Attention is the only place where tokens meet. (The text says "Eq. 4" here; it means the attention of Eq. 3.)
>
> This fact is good news for a server. Work that each token does on its own can be done for tokens of different requests side by side, in one big matrix product. Only attention needs each request's own keys and values. So the serving problem comes down to one question: where do the keys and values of each request live?

## Prompts, outputs and sequences {§2.2}

> [!PAPER] · §2.2 · page 3
> [![The opening of §2.2, with input prompt, output tokens and lists as sequence highlighted](/img/papers/vllm/p2-service.png)](/img/papers/vllm/p2-service.png)
>
> **What it says:** a request gives the service "input prompt" tokens $$(x_1, \ldots, x_n)$$, and the service writes "output tokens" $$(x_{n+1}, \ldots, x_{n+T})$$. The prompt and the output together are called a "sequence".
>
> **What it means:**
>
> This is the vocabulary for the rest of the paper. The **prompt** is what the user sends: a question, a document, a chat so far. The **output** is what the model writes back. $$n$$ is the length of the prompt and $$T$$ the length of the output. The **sequence** is both together, and its KV cache grows from $$n$$ tokens to $$n + T$$ tokens while the request runs.
>
> The server knows $$n$$ as soon as the request arrives. It does not know $$T$$. That one unknown number causes most of the trouble in §3.

> [!PAPER] · §2.2 · page 3
> [![The paragraph about the KV cache, with one by one, cached for generating future tokens, KV cache of one token depends on all its previous tokens, and will be different highlighted](/img/papers/vllm/p2-kvcache.png)](/img/papers/vllm/p2-kvcache.png)
>
> **What it says:** the model can only generate new tokens "one by one". The keys and values of existing tokens are "cached for generating future tokens", the KV cache. "The KV cache of one token depends on all its previous tokens", so the cache of the same token at different positions "will be different".
>
> **What it means:**
>
> The first two sentences repeat what Eq. 1 and Eq. 3 told us. The last two are new and easy to miss. A key is not a fixed property of a word. The key of "the" depends on where "the" stands and on the words before it.
>
> This rules out a tempting shortcut. You cannot keep one key per word of the vocabulary and reuse it everywhere. The cache belongs to one sequence. Two requests can share cache only if they share the same tokens from the very start, such as the same prompt. Part 4 builds on exactly that.

We test the claim on OPT-125M. First, the word "the" twice in one sentence. Then the word "the" at the same position after two different openings:

```text
== 2. The same token gets a different KV cache at a different position or after different words
'the cat sat on the mat': "the" at token positions 1 and 5
  layer  0: keys of the two "the" are equal: False; cosine similarity 0.638
  layer 11: keys of the two "the" are equal: False; cosine similarity 0.843
'I saw the' against 'We ate the': "the" is token 3 in both
  layer  0: max difference of the keys of "the" 0.00e+00; cosine similarity 1.000
  layer  1: max difference of the keys of "the" 4.74e-01; cosine similarity 0.997
  layer 11: max difference of the keys of "the" 2.68e+00; cosine similarity 0.922
layer 0 sees only the token and its position; from layer 1 on, the key carries the earlier words
```

The two copies of "the" in one sentence have clearly different keys, even in the first layer. The reason is that OPT adds a position vector to every token. At the same position after different words, the first layer gives identical keys: layer 0 sees only the token and its position. From layer 1 on, attention has mixed in the earlier words, and the keys drift apart. By the last layer the similarity is down to 0.922. So the paper is exact: the cache of a token depends on its position and on every token before it.

> [!DEFINITION] Cosine similarity
> A measure of how much two vectors point the same way, from −1 (opposite) to 1 (same direction). It is the dot product divided by the lengths of the two vectors. A value of 1.000 with a difference of 0 means the vectors are identical.

## Two phases: reading and writing {§2.2}

> [!PAPER] · §2.2 · page 3
> [![The paragraph on the prompt phase, with The prompt phase and parallelized using matrix-matrix multiplication operations highlighted](/img/papers/vllm/p2-prompt-phase.png)](/img/papers/vllm/p2-prompt-phase.png)
>
> **What it says:** "The prompt phase" takes the whole prompt and computes the probability of the first new token. It also makes the keys and values of all prompt tokens. Because all prompt tokens are known, the work "can be parallelized using matrix-matrix multiplication operations", which uses the GPU well.
>
> **What it means:**
>
> This is Method 2 of our first run. The prompt is known, so the model processes all its tokens in one pass. Each weight matrix multiplies a whole block of token vectors at once. A GPU is built for exactly that kind of work.
>
> The prompt phase also fills the KV cache with the keys and values of the $$n$$ prompt tokens. From then on, the cache is the request's memory of its own past.

> [!PAPER] · §2.2 · page 3
> [![The paragraph on the autoregressive generation phase, with only the new key and value vector, maximum length, end-of-sequence token, matrix-vector multiplication and becomes memory-bound highlighted](/img/papers/vllm/p2-gen-phase.png)](/img/papers/vllm/p2-gen-phase.png)
>
> **What it says:** the generation phase writes the remaining tokens one at a time. At each step, "only the new key and value vector" are computed; the earlier ones come from the cache. It stops at a "maximum length" or at an "end-of-sequence (<eos>) token". The steps cannot run in parallel and use "matrix-vector multiplication", so this phase "becomes memory-bound".
>
> **What it means:**
>
> Now the model writes. At each step it feeds in one token, the one it just wrote, and gets out the next one. It computes one new key and one new value per layer and adds them to the cache. The request ends when the model writes a special stop token, or when it reaches a length limit.
>
> One token at a time means each weight matrix multiplies a single vector, not a block of vectors. The GPU must still read the whole matrix from memory, but it does very little arithmetic with it. The paper says this phase is "responsible for most portion of the latency". A prompt of 500 tokens takes one pass. An answer of 500 tokens takes 500 passes.

> [!DEFINITION] End-of-sequence token
> A special token that means "the text ends here". OPT writes it as `</s>`, the paper as `<eos>`. When the model picks it, the request is done. Nobody knows in advance when that will happen.

> [!DEFINITION] Matrix-vector and matrix-matrix multiplication
> Multiplying a matrix by one vector gives one vector. Multiplying it by a matrix of many vectors side by side gives many vectors in one operation. The second does the work of many of the first, but it reads the big matrix only once.

> [!DEFINITION] Latency
> The time a user waits for an answer, from sending the request to getting the last token.

Follow one request through both phases. Its prompt is the 7 tokens "Four score and seven years ago our". The first pass of the model reads all 7 tokens together. It stores their 7 keys and values in the KV cache, and it writes "fathers".

Now the generation phase starts. The second pass feeds in only "fathers". It computes one new key and value, reads the 7 stored ones, and writes "brought". The third pass does the same for "brought" and writes "forth". The fourth writes `<eos>`, and the request ends.

{{FIG:p2_life|The life of one request. The prompt phase computes the keys and values of all 7 prompt tokens in one pass. Each generation pass computes one new slot, reads every earlier slot from the KV cache, and writes one token.}}

The prompt phase is one wide pass. The generation phase is a stack of thin passes, one per output token, each reading the whole cache. A real answer has hundreds of them, so they set the latency.

> [!NOTE] Where this idea comes from: the memory wall
> For decades, processors got faster at arithmetic more quickly than memory got faster at delivering data. Computer architects call this gap the "memory wall". A processor that waits for data sits idle, however fast its arithmetic is.
>
> In 2009, Williams, Waterman and Patterson gave a simple way to reason about it, the roofline model. A program that does little arithmetic per byte it reads is limited by memory speed. A program that does a lot is limited by arithmetic speed. A GPU has the same two limits, only larger.
>
> Think of a cook whose recipe book sits in the basement. If she cooks one dish per trip downstairs, most of her time goes on the stairs. If she cooks eight dishes from the same page per trip, the stairs cost the same, and she does eight times the work.

> [!DEFINITION] Memory bandwidth
> The number of bytes per second that memory can deliver to the processor. Arithmetic speed is counted in FLOPs per second, memory speed in bytes per second.

{{FIG:p2_mv|Matrix-vector against matrix-matrix work. In both cases the GPU reads the big weight matrix W once from memory. With one token, that read buys one output. With a batch of 8 tokens, the same read buys 8 outputs.}}

In the generation phase, the weight matrix is the recipe book and each token is a dish. Why is one vector at a time so wasteful? A GPU can do arithmetic much faster than it can read memory. What counts is how much arithmetic you do for each byte you read. This ratio is called the **arithmetic intensity**:

$$
\text{intensity} = \frac{\text{FLOPs}}{\text{bytes moved}} = \frac{2 d^2 n}{2\,(d^2 + 2 d n)} \approx n \quad \text{when } n \ll d
$$

Read it from left to right. One weight matrix is $$d \times d$$; for OPT-13B, $$d = 5{,}120$$. We multiply it by $$n$$ token vectors. Each weight does one multiplication and one addition per token, so the work is $$2 d^2 n$$ FLOPs. The bytes moved are the matrix ($$d^2$$ numbers) plus the input and output vectors ($$2 d n$$ numbers). Each number takes 2 bytes in FP16.

While $$n$$ is much smaller than $$d$$, the matrix dominates the bytes, and the ratio is about $$n$$.

> [!DEFINITION] FLOP
> One floating-point operation, such as one multiplication or one addition of two decimal numbers. FLOPS (with a capital S) means FLOPs per second, the speed of a processor. A TFLOP is a trillion FLOPs.

A tiny example. Take $$d = 4$$ and $$n = 1$$. The work is $$2 \cdot 16 \cdot 1 = 32$$ FLOPs, the bytes are $$2\,(16 + 8) = 48$$, and the intensity is 0.67. With $$n = 2$$, it is $$64 / 64 = 1$$. Twice the tokens give twice the work for only a few more bytes.

Now on real hardware. We multiply one 5,120 by 5,120 matrix by 1 to 512 token vectors, on the GPU of an Apple laptop. The matrix has the size of one layer of OPT-13B. Each time is the fastest of 7 runs of 20 products:

```python
for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    X = torch.randn(D, n, dtype=torch.float16, device=GPU)
    t = fastest(lambda: [W @ X for _ in range(20)]) / 20
    flops = 2 * D * D * n                                    # one multiply and one add per weight per token
    nbytes = 2 * (D * D + 2 * D * n)                         # read W and X, write the result, 2 bytes each
```

```text
== 3. One 5120 x 5120 FP16 weight matrix (OPT-13B size) times n token vectors, on the laptop GPU
n tokens  time ms  TFLOP/s  FLOPs per byte
       1    0.196     0.27             1.0
       2    0.216     0.49             2.0
       4    0.211     0.99             4.0
       8    0.218     1.92             8.0
      16    0.210     4.00            15.9
      32    0.226     7.42            31.6
      64    0.233    14.43            62.4
     128    0.272    24.70           121.9
     256    0.595    22.55           232.7
     512    1.782    15.06           426.7
from 1 to 64 tokens: time x1.2 for 64x the work; from 1 to 512: time x9.1 for 512x the work
```

{{FIG:p2_matmul|Our run on a laptop GPU. Multiplying one 5,120 by 5,120 matrix by 1 token or by 64 tokens takes almost the same time, because the time goes into reading the matrix. Only past about 128 tokens does the arithmetic start to cost time.}}

This table is the heart of §2. One token takes 0.196 ms. Sixty-four tokens take 0.233 ms, only 1.2 times longer for 64 times the work. In that range the GPU waits for the matrix to arrive from memory, and the arithmetic is almost free. Above 128 tokens the time starts to grow, because now the arithmetic is the limit. (At 512 the speed drops again; a laptop GPU is not tuned for every size, and timings on a shared laptop are noisy.)

In roofline words, the time stops being flat between 64 and 256 tokens, where the intensity is between 62.4 and 232.7 FLOPs per byte. In that range, this GPU's arithmetic catches up with its memory.

So the generation phase, with $$n = 1$$, runs this GPU at 0.27 TFLOP/s, while it can reach 24.7. It uses about 1% of the speed it has. The obvious fix is to give the matrix more tokens per pass. If one request can give only one token per step, then many requests must share each step.

## Batching, and why it is hard {§2.3}

> [!NOTE] Where this idea comes from: batching
> Batching is older than computers. A bank collects the day's cheques and sends one van to the city, not one van per cheque.
>
> Model servers used the same idea before large language models. A server waits a short time, collects the requests that came in, and runs them through the model as one batch. The paper's first baseline works this way. FasterTransformer has no scheduler, so the authors add one like that of NVIDIA's Triton server. It takes up to $$B$$ of the earliest requests and runs them together (§6.1). This is **request-level batching**: a batch is formed once and stays the same until it is done.

> [!PAPER] · §2.3 · page 3
> [![The paragraph on batching, with amortized across the requests in a batch, arrive at different times, queueing delays, pad the inputs and wasting highlighted](/img/papers/vllm/p2-batching.png)](/img/papers/vllm/p2-batching.png)
>
> **What it says:** batching helps because requests share the same weights, so the cost of reading the weights is "amortized across the requests in a batch". But it is hard for two reasons. Requests "arrive at different times", so a simple method causes "queueing delays". And their lengths differ, so a simple method must "pad the inputs" and outputs, "wasting" computation and memory.
>
> **What it means:**
>
> The first sentence is our table in words. Eight requests in one step read the weights once and do eight tokens of work. The cost of the read is split eight ways.
>
> The two problems come from real life. Think of a minibus that leaves only when it is full and comes back only when the last passenger has reached home. Passengers who arrive just after it leaves wait a long time. And the passenger who gets off at the first stop leaves an empty seat for the whole trip. The empty seat is **padding**; the wait is **queueing delay**.

> [!DEFINITION] Padding
> Filler that makes items of different lengths the same length, so they fit in one rectangular table. A batch with answers of 10 and 300 tokens is padded to 300 for both. Then 290 slots of the short answer do no useful work.

> [!DEFINITION] Queueing delay
> The time a request waits before the server starts it. Users feel it as a pause before the first word appears.

> [!PAPER] · §2.3 · page 3
> [![The paragraph on fine-grained batching, with the iteration level, completed requests are removed from the batch and new ones are added, waiting for a single iteration and eliminate the need to pad highlighted](/img/papers/vllm/p2-fine-batching.png)](/img/papers/vllm/p2-fine-batching.png)
>
> **What it says:** newer methods, such as cellular batching and iteration-level scheduling, work at "the iteration level". After each step, "completed requests are removed from the batch, and new ones are added". A new request waits only for "a single iteration", and special GPU kernels "eliminate the need to pad".
>
> **What it means:**
>
> Here the minibus becomes a city bus. It stops after every step. Anyone who has arrived gets off, anyone who is waiting gets on, and the bus goes on. No seat stays empty for long, and nobody waits for a whole trip.
>
> Iteration-level scheduling comes from Orca, one of the two systems that vLLM is compared with. vLLM keeps this idea and does not claim it. The paper's point is the next step: once batching is this flexible, the only thing that still limits the batch is memory.

> [!DEFINITION] Iteration
> One step of the generation phase: one pass of the model that writes one new token for every request in the batch.

> [!DEFINITION] GPU kernel
> One program that runs on the GPU, started by the CPU, such as "multiply these two matrices" or "compute attention". A model step runs many kernels, one after another.

> [!NOTE] Where this idea comes from: Orca
> Cellular batching came first, in 2018, for recurrent neural networks, which also run one step at a time. Orca (OSDI 2022) brought the idea to Transformer language models and called it iteration-level scheduling.
>
> Orca had to solve one problem to make it work. Requests in a batch have different lengths, so their attention cannot run as one rectangular matrix product. Orca's answer, **selective batching**, uses the fact from the position-wise box above. It joins the tokens of all requests for the position-wise layers, and runs attention for each request on its own. The vLLM system batches in the same way, and builds its memory system under it.

Before the real simulation, here is a tiny one you can check by hand. Two slots, four requests. A needs 2 steps and B needs 5; both arrive at step 0. C needs 2 steps and arrives at step 1. D needs 3 steps and arrives at step 2.

{{FIG:p2_batch_toy|Four requests on a server with two slots. Top: request-level batching keeps A's slot as padding until B ends, so C and D wait until step 5. Bottom: iteration-level batching gives A's slot to C at step 2 and to D at step 4, and finishes one step sooner.}}

With request-level batching, A and B start together, and the batch lasts as long as B, 5 steps. A is done after 2 steps, so its slot is padding for 3 steps. C and D wait until step 5, then run as the next batch for 3 steps. That is 8 steps in all.

With iteration-level batching, A's slot goes to C at step 2, the step after A ends. When C ends, D takes the slot at step 4. There is no padding, C waits 1 step instead of 4, and the work ends at step 7. The same rules, applied to 300 real requests, give the simulation below.

We simulate both methods with real answer lengths. We take 300 answers from the Alpaca dataset of instructions and count their tokens with the OPT tokenizer. One request arrives every 12 steps, at a server with room for 8 requests at once. The simulation counts steps, not seconds:

```python
def request_level():
    """Method 1: take the waiting requests (up to 8) as one batch, run it until its longest answer ends, repeat.
    A request that ends early keeps its slot as padding; a request that arrives meanwhile waits."""
    t, i, waits, pad, busy = 0, 0, [], 0, 0
    batches = []
    while i < len(OUTLEN):
        t = max(t, ARRIVE[i])
        batch = [j for j in range(i, len(OUTLEN)) if ARRIVE[j] <= t][:SLOTS]
        i += len(batch)
        steps = max(OUTLEN[j] for j in batch)
        waits += [t - ARRIVE[j] for j in batch]
        pad += sum(steps - OUTLEN[j] for j in batch)
        busy += sum(OUTLEN[j] for j in batch)
        batches.append([OUTLEN[j] for j in batch])
        t += steps
    return {'steps': t, 'wait': statistics.mean(waits), 'pad': pad, 'useful': busy, 'batches': batches}


def iteration_level():
    """Method 1: after every step, finished requests leave and waiting requests take their slots at once."""
    t, i, running, waits, useful = 0, 0, [], [], 0
    while i < len(OUTLEN) or running:
        while i < len(OUTLEN) and ARRIVE[i] <= t and len(running) < SLOTS:
            waits.append(t - ARRIVE[i]); running.append(OUTLEN[i]); i += 1
        if not running:                                     # nothing to do until the next request arrives
            t = ARRIVE[i]; continue
        useful += len(running)
        running = [r - 1 for r in running if r > 1]         # one step: every running request writes one token
        t += 1
    return {'steps': t, 'wait': statistics.mean(waits), 'pad': 0, 'useful': useful}
```

The padding of a batch also has a short formula, which we use as Method 2. A batch of $$b$$ requests with lengths $$\ell_1, \ldots, \ell_b$$ runs for $$\max_j \ell_j$$ steps, so its padding is:

$$
\text{padding} = b \cdot \max_j \ell_j - \sum_{j=1}^{b} \ell_j
$$

For example, a batch with lengths 10, 40 and 50 runs 50 steps in 3 slots. That is 150 slot-steps, of which 100 write a token, so 50 are padding. Here is the result of both methods:

```text
== 4. Request-level batching (pad to the longest) against iteration-level batching (Alpaca output lengths)
300 requests, one every 12 steps; output lengths: mean 52.1, shortest 1, longest 306; batch of 8
                 steps to finish  mean wait (steps)  padding slot-steps
request-level              5,519              927.3              24,784
iteration-level            3,697                0.2                   0
useful slot-steps (one token each): 15,643 in both; padding share of request-level batches 61.3%; closed form for padding agrees: True
```

{{FIG:p2_batching|The first 520 steps of our simulation. Top: each batch runs until its longest answer ends; the dashed parts are slots held by requests that are already done. Bottom: a slot is refilled as soon as its request ends. By step 520, 27 requests have started at the top and 44 at the bottom.}}

The two servers do the same useful work, 15,643 tokens. With request-level batching, 61.3% of the slot-steps in its batches are padding. That waste makes the server too slow for the traffic, so a queue builds up. A request waits 927 steps on average before it starts.

The picture shows why. From step 224 to step 474, a batch of one long answer of 250 tokens runs alone. Meanwhile, 21 new requests arrive and wait.

With iteration-level batching, a request waits 0.2 steps on average, and the work ends at step 3,697, soon after the last request arrives.

The demand is light, too: on average only about 4.3 of the 8 slots are needed (52.1 tokens per request, one request every 12 steps). Request-level batching fails anyway, because padding eats the free room. This is why every modern server, vLLM included, schedules at the iteration level. It also shows the pattern of the whole paper: the same hardware serves far more when no slot is held for nothing.

## The real limit is memory {§3}

> [!PAPER] · §3 · page 4
> [![The opening of §3, with constrained by GPU memory capacity and memory-bound highlighted](/img/papers/vllm/p2-s3-intro.png)](/img/papers/vllm/p2-s3-intro.png)
>
> **What it says:** fine-grained batching reduces wasted computation, but the number of requests in a batch is "still constrained by GPU memory capacity", in particular the room for the KV cache. The serving system's throughput is "memory-bound".
>
> **What it means:**
>
> Iteration-level batching removed padding, so the next limit shows. Each running request keeps its KV cache in GPU memory. A ninth request can join the batch only if its cache fits too. So the batch size is set by memory, and the batch size sets the throughput.
>
> Note that "memory-bound" has a second meaning here. In §2 it meant that one step waits on reading memory. Here it means that the amount of memory limits how many requests run. The fix for the first, bigger batches, runs straight into the second.

> [!DEFINITION] Throughput
> The number of requests, or tokens, that a server finishes per second. A server with high throughput serves many users with the same GPUs.

The paper then lists three challenges. We take them in order.

### Challenge 1: the cache is large {§3}

> [!PAPER] · §3 · page 4
> [![The paragraph on the large KV cache, with 800 KB of space, as much as 1.6 GB and only a few tens of requests could highlighted](/img/papers/vllm/p2-large-kv.png)](/img/papers/vllm/p2-large-kv.png)
>
> **What it says:** for OPT-13B, the KV cache of one token takes "800 KB of space": 2 (key and value) × 5,120 (hidden size) × 40 (layers) × 2 (bytes per FP16 number). A request can reach 2,048 tokens, so its cache can be "as much as 1.6 GB". Even if all GPU memory went to the cache, "only a few tens of requests could" fit.
>
> **What it means:**
>
> Each token keeps one key and one value in each of the 40 layers, and each is 5,120 numbers of 2 bytes. That is 819,200 bytes per token, more than the text of a long novel. A full conversation of 2,048 tokens needs 1.68 GB.
>
> Part 1 checked this formula against the real tensors of a running model, to the byte. Here we apply it to the real configuration files of four OPT models.

```text
== 5. KV cache size from the real configs (FP16, 2 bytes per number)
model      layers hidden KB per token GB per 2048 tokens weights GB
opt-125m       12    768           36               0.08       0.25
opt-1.3b       24   2048          192               0.40       2.60
opt-13b        40   5120          800               1.68      26.00
opt-66b        64   9216        2,304               4.83     132.00
OPT-13B on an 80 GB GPU: 54 GB left after the weights = 32 requests of 2048 tokens
```

The cache per token grows with both the width and the depth of the model. OPT-66B needs 2.3 MB for every token, and 4.83 GB for a full request. Even on the largest A100, with 80 GB, OPT-13B leaves room for only 32 full requests after its 26 GB of weights. That is the paper's "a few tens". If a server books the full 2,048 slots for every request, 32 is the hard limit, no matter how fast the GPU is.

> [!PAPER] · §3 · page 4
> [![The paragraph on hardware trends, with grows faster than the memory capacity, The FLOPS increases by more than 2x and stays at 80GB maximum highlighted](/img/papers/vllm/p2-trend.png)](/img/papers/vllm/p2-trend.png)
>
> **What it says:** GPU computation speed "grows faster than the memory capacity". From the NVIDIA A100 to the H100, "The FLOPS increases by more than 2x", but the memory "stays at 80GB maximum". So memory will be "an increasingly significant bottleneck".
>
> **What it means:**
>
> A new GPU generation can do more than twice the arithmetic in the same time, but holds the same amount of data. A faster GPU wants larger batches to stay busy, as our matrix table showed. Larger batches need more KV cache. So each new GPU makes the memory problem worse, not better.
>
> This is the authors' argument that their work will last. Saving memory is not a trick for one card. It grows more valuable with every card that computes faster without holding more.

### Challenge 2: decoding methods that could share {§3}

> [!PAPER] · §3 · page 4
> [![The paragraph on complex decoding algorithms, with multiple random samples, accounts for 12% of the total KV cache memory, can be shared and should remain unshared highlighted](/img/papers/vllm/p2-decoding.png)](/img/papers/vllm/p2-decoding.png)
> [![The end of the paragraph, with the sharing pattern evolves as the decoding process advances highlighted](/img/papers/vllm/p2-decoding-2.png)](/img/papers/vllm/p2-decoding-2.png)
>
> **What it says:** when a user asks for "multiple random samples" from one prompt, the cache of the prompt, which "accounts for 12% of the total KV cache memory" in the authors' experiment, "can be shared". The cache of the generated parts "should remain unshared", because the samples differ. In beam search, beams can share more (up to 55% saving), and "the sharing pattern evolves as the decoding process advances".
>
> **What it means:**
>
> A code assistant often shows a user several suggestions for one request. All of them start from the same prompt, so the keys and values of the prompt are the same for all of them. We saw why above: the cache of a token depends only on the tokens before it, and here those are the same. Storing the prompt once would be enough.
>
> Beam search is harder. It keeps several candidate answers, called beams, and at each step keeps the best ones and drops the others. Beams often start the same way and split later. Which beams share which tokens changes at every step. A memory system for beam search must handle sharing that comes and goes.

> [!DEFINITION] Parallel sampling and beam search
> Two ways to get several answers from one prompt. **Parallel sampling** draws several answers at random, each one on its own. **Beam search** keeps the $$k$$ most likely partial answers (the beams) at each step and extends them, so it explores several good paths at once.

We ask OPT-125M for 4 random answers to one prompt with the library's `model.generate`, and look at the cache it keeps. Then we run beam search with 4 beams and count how many of the new tokens the beams have in common:

```text
== 6. Parallel sampling and beam search with model.generate (OPT-125M)
prompt of 16 tokens, 4 samples of up to 40 new tokens; cache shape of layer 0: (4, 12, 55, 64)
the 4 copies of the prompt keys are identical: True
cache held: 16,220,160 bytes; the prompt stored 4 times = 4,718,592 bytes, of which 3 copies (3,538,944 bytes, 22% of the cache) are duplicates
  sample 1: '\nWrite the code to get the sum of the numbers for each varia'
  sample 2: 'This also looks pretty cool.  Thanks for sharing that :)\nTha'
  sample 3: '\nIf a function returns zero from the table and the return am'
  sample 4: '\nWrite your function to return a function that returns an ar'
beam search, 4 beams, 30 new tokens: all 4 share their first 3 new tokens
new tokens stored once per beam: 120; stored once per distinct prefix: 86 (28% less)
  beam 1: '\nWrite a function that returns the sum of a list of numbers.\n\nWrite a '
  beam 2: '\nWrite a Python function that returns the sum of a list of numbers.\n\nW'
  beam 3: '\nWrite a Python function that returns the sum of a list of numbers.\n\nW'
  beam 4: '\nWrite a function that returns the sum of a list of integers.\n\nWrite a'
```

{{FIG:p2_sharing|Parallel sampling. The library keeps a full copy of the prompt cache for each of the four samples. All four copies are identical, so one shared copy would be enough.}}

The cache has a first dimension of 4: one full copy per sample. The four copies of the prompt keys are identical, bit for bit. Three of them, 22% of the whole cache, are pure duplicates. Our share is larger than the paper's 12% because our prompt is long next to our 40-token answers; the share depends on the lengths.

The beams tell the same story with a twist. All four begin with the same 3 tokens. Beams 2 and 3 agree for much longer, and beams 1 and 4 for a while. If each distinct prefix were stored once, 86 token slots would hold what now takes 120, which is 28% less. (A small model often repeats the prompt; the text is not the point.)

To share like this, the memory system must let two sequences point at the same stored tokens. One contiguous chunk per sequence cannot do that.

### Challenge 3: lengths nobody knows in advance {§3}

> [!PAPER] · §3 · page 4
> [![The paragraph on unknown input and output lengths, with variability in their input and output lengths, exhaust available memory and deleting or swapping out the highlighted](/img/papers/vllm/p2-lengths.png)](/img/papers/vllm/p2-lengths.png)
>
> **What it says:** requests show "variability in their input and output lengths". As an output grows, its cache grows and may "exhaust available memory". The system must then make choices, such as "deleting or swapping out the" KV cache of some requests.
>
> **What it means:**
>
> A server must accept a short question and a long document. Worse, it cannot see how long the answer will be. A request that looked small can keep writing until memory is full, and then something has to give.
>
> The paper names the two ways out. The server can delete a request's cache and compute it again later. Or it can copy the cache to the CPU's memory and bring it back when there is room. Part 4 covers how vLLM does both.

> [!DEFINITION] Swapping
> Moving data out of fast memory into a larger, slower memory to make room, and moving it back when it is needed. Operating systems swap pages to disk; vLLM swaps blocks of KV cache from GPU memory to CPU memory.

How unpredictable are answer lengths? We ask OPT-125M the same question 20 times with random sampling, and stop at its end token or at 200 tokens. We also count the lengths of 2,000 answers in the Alpaca dataset:

```text
== 7. The output length is unknown until the model stops
20 sampled answers to one question (stop at </s> or 200 tokens): lengths [2, 3, 3, 4, 5, 6, 15, 19, 23, 27, 28, 29, 35, 48, 73, 88, 151, 200, 200, 200]
2000 Alpaca answers: shortest 1, median 43, 90th percentile 127, longest 499 tokens
```

The same question gives answers from 2 to 200 tokens, and three answers hit the limit. Nothing about the prompt tells the server which case it will get. Across Alpaca, the median answer has 43 tokens, but one in ten has more than 127 and the longest has 499. A server that books memory in advance has two bad options. It can book for the long case and waste memory on the short ones, or book less and risk running out.

## How earlier systems stored the cache {§3.1}

> [!PAPER] · §3.1 · page 4
> [![The opening of §3.1, with contiguous memory, contiguous tensor, statically allocate a chunk of memory and maximum possible sequence highlighted](/img/papers/vllm/p2-existing.png)](/img/papers/vllm/p2-existing.png)
>
> **What it says:** deep learning frameworks require tensors to be stored in "contiguous memory", so earlier serving systems stored the KV cache of each request as one "contiguous tensor". Because output lengths are unpredictable, they "statically allocate a chunk of memory" based on the request's "maximum possible sequence" length.
>
> **What it means:**
>
> A tensor in PyTorch is one block of memory with a shape on top. To read element 500 of the cache, the code computes "start address plus 500 times the size of one token" and jumps there. That only works if all tokens sit next to each other. So a cache that grows must have room to grow into, right behind it.
>
> Nobody knows how far the cache will grow, so the earlier systems took the safe choice. They booked room for the longest possible request, 2,048 tokens, the moment it arrived. It is like a restaurant that seats every guest at a table for twelve, because any guest might bring eleven friends.

> [!DEFINITION] Tensor and contiguous memory
> A **tensor** is an array of numbers with any number of dimensions. An example is the keys of one layer, with shape (batch, heads, tokens, head size). **Contiguous** means that its numbers fill one unbroken run of addresses in memory, with no gaps.

> [!NOTE] Where this idea comes from: allocators and fragmentation
> Operating systems faced this problem long before GPUs. Programs ask for memory of different sizes, at different times, and give it back in a different order. The part of the system that hands out memory is the **allocator**; in C it is behind `malloc`. Once programs of mixed sizes come and go, free memory breaks into gaps between the chunks still in use.
>
> Kenneth Knowlton described the buddy system in 1965. It rounds every request up to a power of two, which makes free pieces easy to find and to merge. The Linux kernel still uses a buddy allocator for its pages. The price is the space lost by rounding up. Think of parking cars of mixed lengths along a kerb: when some leave, the gaps rarely fit the next car.

{{FIG:p2_alloc|Fragmentation on a memory of 16 units. After B and D leave, 7 units are free, but no gap holds the 5 units that E needs in a row: external fragmentation. A buddy allocator rounds E's 5 units up to 8, and 3 units sit unused inside the chunk: internal fragmentation.}}

The [Background page](background-the-kv-cache-and-paging.md) runs a first-fit allocator and a buddy allocator on real request sizes. Here it is enough to see the two kinds of waste. The KV cache of earlier serving systems got both, and one more.

## Figure 3: three kinds of waste {§Figure 3}

> [!PAPER] · Figure 3 · page 4
> [![Figure 3 of the paper: the memory of two requests. Request A has a 7-token prompt, 2 generated tokens, 2 reserved slots and 2038 slots never used. Request B has a 3-token prompt, 1 generated token, 1 reserved slot and 507 slots never used. A gap between them is external fragmentation](/img/papers/vllm/p2-figure3.png)](/img/papers/vllm/p2-figure3.png)
>
> **What it says:** "Three types of memory wastes", reserved, internal fragmentation and external fragmentation, keep other requests from fitting. Each slot holds the KV cache of one token, and "same tokens can have different KV cache" at different positions.
>
> **What it means:**
>
> The figure is a strip of GPU memory, one box per token slot. Request A has a chunk of 2,048 slots. Its prompt "Four score and seven years ago our" fills 7. It has written "fathers" and is writing "brought" in the current step. It will still write "forth" and `<eos>`: those 2 slots are **reserved**, booked for tokens that will come. The rest of the chunk, about 2,038 slots, will never hold anything: that is **internal fragmentation**.
>
> Request B has a chunk of 512 slots. Its prompt "You only live" fills 3, "once" is being written now, `<eos>` is reserved, and 507 slots will stay empty. Between the two chunks sits a gap, too small for another request. That gap is **external fragmentation**.
>
> The last sentence of the caption repeats the point we tested above. The two `<eos>` tokens look the same, but their caches differ, because they follow different words.

> [!DEFINITION] Internal and external fragmentation
> **Internal fragmentation** is space inside a booked chunk that the owner never uses. **External fragmentation** is free space between chunks that is too small or too broken up to give to anyone. The [Background page](background-the-kv-cache-and-paging.md) shows both on a 32-slot memory.

We count the slots of the figure with a small function. For each chunk it splits the slots into those that hold a token now, those reserved for later tokens, and those never used:

```python
def slots(max_len, prompt, generated, final):
    """Split one pre-allocated chunk into: slots that hold a token now, slots that a later token of this request
    will fill (reserved) and slots that no token will ever fill (internal fragmentation)."""
    now = prompt + generated
    return {'token states': now, 'reserved': final - now, 'never used': max_len - final}


A = dict(max_len=2048, prompt=7, generated=2, final=7 + 4)   # Four score ... our | fathers brought | forth <eos>
B = dict(max_len=512, prompt=3, generated=1, final=3 + 2)    # You only live | once | <eos>
```

```text
== 8. Figure 3: the slots of two pre-allocated requests at the current step
request A: chunk of 2048: token states 9, reserved 2, never used 2,037; useful share 0.44%
request B: chunk of 512: token states 4, reserved 1, never used 507; useful share 0.78%
both chunks: 2,560 slots; token states 13, reserved 3, never used 2,544; even at the end only 16 slots (0.62%) ever hold a token
```

Request B matches the figure exactly: 3 + 1 + 1 + 507 = 512. For request A, the figure's tokens give 2,037 slots never used, one fewer than its label of 2,038. It is a small slip in the drawing and changes nothing. The big number is the share: even when both requests finish, only 16 of 2,560 booked slots, 0.62%, ever hold a token.

{{FIG:p2_fig3|Figure 3 drawn to scale. The thin coloured sliver at the left of each bar is all the memory that request ever uses. The rest of each chunk is booked and never filled.}}

The paper's toy example is extreme on purpose, with very short requests. Real requests are longer. Part 1 simulated 2,000 Alpaca requests that each book 2,048 slots, and only 3.6% of the booked memory held tokens. The pattern is the same: almost all the booked memory sits empty.

> [!PAPER] · §3.1 · page 4
> [![The paragraph that names the three sources of waste, with three primary sources of memory wastes, reserved slots for future tokens, internal fragmentation due to and external fragmentation from the memory allocator highlighted](/img/papers/vllm/p2-wastes.png)](/img/papers/vllm/p2-wastes.png)
> [![The rest of the paragraph, with pure memory waste, eventually used, could otherwise be used to and as low as 20.4% highlighted](/img/papers/vllm/p2-wastes-2.png)](/img/papers/vllm/p2-wastes-2.png)
>
> **What it says:** the "three primary sources of memory wastes" are "reserved slots for future tokens", "internal fragmentation due to" booking for the maximum length, and "external fragmentation from the memory allocator", such as the buddy allocator. Internal and external fragmentation are "pure memory waste". Reserved memory is "eventually used", but while it waits it takes space that "could otherwise be used to" serve other requests. The effective memory can be "as low as 20.4%".
>
> **What it means:**
>
> The three wastes differ in a useful way. External fragmentation is waste from the start: the gap between chunks is known before any token is written. Internal fragmentation is only known at the end, when the request stops and the empty slots stay empty. Reserved slots are not wasted forever, but they are wasted for now. A slot booked for token 1,000 sits empty for 999 steps while another request could have used it.
>
> The buddy allocator, from the Background page, rounds every chunk up to a power of two. That adds more internal waste and leaves gaps of awkward sizes, which is external fragmentation.
>
> The figure of 20.4% is from the paper's Figure 2, measured on Orca with the maximum booking. Only about one fifth of the memory set aside for the KV cache held real keys and values. Four fifths of the most valuable memory on the card did nothing.

> [!PAPER] · §3.1 · page 5
> [![The paragraph on compaction, with compaction, impractical, massive KV cache, prevents memory and sharing highlighted](/img/papers/vllm/p2-compaction.png)](/img/papers/vllm/p2-compaction.png)
>
> **What it says:** "compaction" has been proposed against fragmentation, but it is "impractical" in a fast serving system because of the "massive KV cache". And even with compaction, one chunk per request still "prevents memory sharing".
>
> **What it means:**
>
> Compaction means sliding all the chunks together to close the gaps. It is like pushing the books on a shelf to one end to make one big free space. It cures external fragmentation, but it costs a copy of everything that moves. It does nothing for internal fragmentation or for reserved slots, and it does nothing for sharing.
>
> How much does the copy cost? We time a copy of 1 GiB of GPU memory on our laptop:

> [!DEFINITION] Compaction
> Moving the blocks of memory that are in use so that they sit next to each other. The free gaps then merge into one large free area. All data that moves must be copied.

```text
== 9. Compaction means copying the KV cache: how fast does this laptop copy GPU memory?
copy 1 GiB: 8.1 ms = 132.7 GB copied per second
moving 12 GB of KV cache at that speed: 90 ms, during which no request writes a token
```

Part 1's budget for a 13B model on a 40 GB card was 12 GB of KV cache. At 132.7 GB per second, moving all of it takes 90 ms. Our laptop needs about 0.2 ms to multiply one large matrix by a token, so 90 ms is hundreds of steps of such work. A server GPU copies faster, but it also writes tokens faster, so the ratio stays bad. And the gaps come back as soon as the next requests finish. This is the paper's reason to avoid gaps in the first place, rather than clean them up.

## What this part sets up

§2 and §3 build one chain of reasons. Writing text is one token at a time, so a single request leaves the GPU idle. Batching many requests fixes that, and iteration-level batching makes batching flexible. Then the KV cache limits the batch size. The cache is large, it could be shared but is not, and it grows to a length nobody knows. Earlier systems answered with one big chunk per request, and most of that memory sat empty.

{{FIG:p2_before_after|Before and with vLLM, on a toy memory of 40 slots. Top: each request books one chunk of 12 slots at the start, so most slots are reserved or never used, and small gaps stay useless. Bottom: each request takes blocks of 4 slots only when it needs them, so 24 slots stay free for new requests.}}

Both rows hold the same 12 tokens. Before vLLM, they lock up all 40 slots, and a fourth request cannot start. With blocks of 4, the only waste is the empty end of each last block, and six blocks stay free.

Part 3 gives the paper's answer. PagedAttention computes Eq. 3 over keys and values stored in small blocks that can sit anywhere in memory. A block table per request, like an operating system's page table, keeps track of where they are.

## What our small run cannot show

Our runs used OPT-125M on an Apple laptop, not OPT-13B on an A100. The probabilities, keys and cache sizes are exact. The timings are the fastest of 7 runs on a laptop that was doing other work, so treat them as shapes, not server numbers. Our batching run is a simulation in steps. It assumes that every step takes the same time, which is close to true only while the batch stays small.

Our sampling run used the library's `model.generate`, which copies the prompt cache once per sample. The serving systems of 2023 did the same, but we did not run them.

> [!TAKEAWAYS] Key takeaways
> - **Eq. 1** writes the probability of a sentence as a product of next-token probabilities. On a real sentence, eight separate passes and one single pass gave the same factors, to within $$2.5 \times 10^{-7}$$.
> - **The KV cache of a token depends on its position and on every earlier token.** In OPT-125M, the same word after different openings has the same key in layer 0 but different keys from layer 1 on.
> - **The prompt phase** reads all prompt tokens in one pass. **The generation phase** writes one token per pass, so each weight matrix multiplies a single vector.
> - **Matrix-vector work wastes the GPU.** On our laptop, one 5,120 by 5,120 matrix took 0.196 ms for 1 token and 0.233 ms for 64 tokens.
> - **Iteration-level batching** lets requests join and leave after every step. In our simulation, padding took 61.3% of the request-level batches, and the mean wait fell from 927 steps to 0.2.
> - **The cache is large:** 800 KB per token and 1.68 GB per full request for OPT-13B. After the weights, an 80 GB GPU holds only 32 full requests.
> - **Sharing is possible but not used.** Four samples from one prompt kept four identical copies of the prompt cache, 22% of the cache in our run.
> - **Output lengths are unknown.** One question gave answers from 2 to 200 tokens.
> - **Figure 3:** booking one chunk per request causes reserved slots, internal fragmentation and external fragmentation. Earlier systems used as little as 20.4% of the cache memory for real tokens.

**Next, in Part 3:** §4.1 to §4.3. PagedAttention, the block table and the KV cache manager, step by step.

<details>
<summary>Run it yourself</summary>

The script behind this part is [`code/papers/vllm/vllm_part2.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vllm/vllm_part2.py). It scores a sentence with Eq. 1 two ways and compares the keys of the same word in different places. It times one large matrix against a growing number of tokens. It then simulates two kinds of batching on Alpaca answers and reads the configs of four OPT models. It also inspects the cache of parallel sampling and beam search.

Last, it counts the slots of Figure 3 and times a memory copy. Every number is saved in `results/part2.json`. On an Apple M-series laptop it takes about two minutes.

```bash
pip install torch transformers datasets
python vllm_part2.py   # downloads facebook/opt-125m (about 250 MB), four OPT configs, and tatsu-lab/alpaca
```

<figure><img src="/img/papers/vllm/part2-run.png" alt="Terminal output of vllm_part2.py: the chain rule, the keys of the word the, the matrix timings, the batching simulation, the cache sizes, the sharing run, the answer lengths, the Figure 3 count and the memory copy" loading="lazy" /><figcaption>The full output of vllm_part2.py, as it ran for this article.</figcaption></figure>

</details>

## The paper and the code

[vLLM on arXiv](https://arxiv.org/abs/2309.06180) · [vLLM on GitHub](https://github.com/vllm-project/vllm) · [our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vllm)
