---
title: "Build It from Scratch: The Whole Transformer in PyTorch"
description: "We write the whole Transformer of the paper in readable PyTorch, one small class per block, and check each block against PyTorch's own layers. Then we load a released translation model into our code, get the same translations, and train a small model on a laptop."
part: 7
covers: "§3, §5.3, §5.4, §6.1"
date: 2026-10-18
tags: [transformer, pytorch, from-scratch, attention, training, beam-search]
---

Parts 1 to 6 explained the Transformer one idea at a time. In this part we write all of it ourselves, in about 300 lines of Python code, and show that it works.

The proof has three steps. First, each block gives the same numbers as PyTorch's own version of that block. Second, the weights of a released English to German model, copied into our code, give the same translations as the official library. Third, our code trains a small model on a laptop, and the model learns a task that needs attention.

## The package: Figure 1 as code {§3}

The code lives in one folder, [code/papers/transformer/scratch](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/transformer/scratch). The main file, `model.py`, has one small class for each block of Figure 1. It uses only the basic parts of PyTorch: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm` and `nn.Dropout`. It does not use `nn.MultiheadAttention` or `nn.Transformer`, because those would hide the parts we want to see.

> [!DEFINITION] PyTorch module
> A class that holds learned weights and has a `forward` function. You call the module like a function, and PyTorch runs `forward`. Modules can hold other modules, so a whole model is a tree of small modules.

<figure class="fig"><svg viewBox="0 0 760 440" role="img" aria-label="The classes of model.py arranged like Figure 1 of the paper. Left: TokenEmbedding, PositionalEncoding and N EncoderLayers, each with MultiHeadAttention and FeedForward wrapped in AddAndNorm, giving the memory. Right: the same embedding table, PositionalEncoding and N DecoderLayers with masked self-attention, cross-attention to the memory and FeedForward, then the generator tied to the embedding."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="40.0" y="382.0" width="280.0" height="48.0" rx="8"/><text class="t-title" x="180.0" y="402.0" text-anchor="middle">TokenEmbedding × √d_model</text><text class="t-tick" x="180.0" y="420.0" text-anchor="middle">source ids → (b, n, d_model)</text><rect class="box" x="40.0" y="316.0" width="280.0" height="48.0" rx="8"/><text class="t-title" x="180.0" y="336.0" text-anchor="middle">PositionalEncoding</text><text class="t-tick" x="180.0" y="354.0" text-anchor="middle">+ fixed sinusoids (§3.5)</text><rect class="box-1" x="30.0" y="112.0" width="300.0" height="186.0" rx="10"/><text class="t-title" x="180.0" y="132.0" text-anchor="middle">EncoderLayer  × N</text><rect class="box" x="50.0" y="146.0" width="260.0" height="48.0" rx="8"/><text class="t-title" x="180.0" y="166.0" text-anchor="middle">MultiHeadAttention + AddAndNorm</text><text class="t-tick" x="180.0" y="184.0" text-anchor="middle">self-attention, padding mask</text><rect class="box" x="50.0" y="222.0" width="260.0" height="48.0" rx="8"/><text class="t-title" x="180.0" y="242.0" text-anchor="middle">FeedForward + AddAndNorm</text><text class="t-tick" x="180.0" y="260.0" text-anchor="middle">Eq. 2, each position alone</text><rect class="box" x="40.0" y="34.0" width="280.0" height="48.0" rx="8"/><text class="t-title" x="180.0" y="54.0" text-anchor="middle">memory</text><text class="t-tick" x="180.0" y="72.0" text-anchor="middle">encoder output (b, n, d_model)</text><rect class="box" x="420.0" y="382.0" width="300.0" height="48.0" rx="8"/><text class="t-title" x="570.0" y="402.0" text-anchor="middle">TokenEmbedding (the same table)</text><text class="t-tick" x="570.0" y="420.0" text-anchor="middle">target ids, shifted right</text><rect class="box" x="420.0" y="316.0" width="300.0" height="48.0" rx="8"/><text class="t-title" x="570.0" y="336.0" text-anchor="middle">PositionalEncoding</text><text class="t-tick" x="570.0" y="354.0" text-anchor="middle">+ fixed sinusoids</text><rect class="box-2" x="410.0" y="84.0" width="320.0" height="214.0" rx="10"/><text class="t-title" x="570.0" y="104.0" text-anchor="middle">DecoderLayer  × N</text><rect class="box" x="430.0" y="238.0" width="280.0" height="48.0" rx="8"/><text class="t-title" x="570.0" y="258.0" text-anchor="middle">masked self-attention + AddAndNorm</text><text class="t-tick" x="570.0" y="276.0" text-anchor="middle">causal mask</text><rect class="box" x="430.0" y="178.0" width="280.0" height="48.0" rx="8"/><text class="t-title" x="570.0" y="198.0" text-anchor="middle">cross-attention + AddAndNorm</text><text class="t-tick" x="570.0" y="216.0" text-anchor="middle">queries: decoder · keys, values: memory</text><rect class="box" x="430.0" y="118.0" width="280.0" height="48.0" rx="8"/><text class="t-title" x="570.0" y="138.0" text-anchor="middle">FeedForward + AddAndNorm</text><text class="t-tick" x="570.0" y="156.0" text-anchor="middle">Eq. 2</text><rect class="box" x="420.0" y="20.0" width="300.0" height="48.0" rx="8"/><text class="t-title" x="570.0" y="40.0" text-anchor="middle">generator: y · Eᵀ (+ bias)</text><text class="t-tick" x="570.0" y="58.0" text-anchor="middle">logits (b, t, vocab), tied to the embedding</text><line class="edge" x1="180.0" y1="382.0" x2="180.0" y2="366.0" marker-end="url(#ah)"/><line class="edge" x1="180.0" y1="316.0" x2="180.0" y2="300.0" marker-end="url(#ah)"/><line class="edge" x1="180.0" y1="112.0" x2="180.0" y2="84.0" marker-end="url(#ah)"/><line class="edge" x1="570.0" y1="382.0" x2="570.0" y2="366.0" marker-end="url(#ah)"/><line class="edge" x1="570.0" y1="316.0" x2="570.0" y2="300.0" marker-end="url(#ah)"/><line class="edge" x1="570.0" y1="84.0" x2="570.0" y2="70.0" marker-end="url(#ah)"/><line class="path" x1="320.0" y1="58.0" x2="428.0" y2="200.0" marker-end="url(#ah-on)"/></svg><figcaption>The classes of model.py, arranged like Figure 1 of the paper. The encoder (left) and the decoder (right) use the same embedding table, and the final layer reuses that table too.</figcaption></figure>

| File | What it holds |
|---|---|
| `model.py` | Every block of the model, the masks, greedy decoding and beam search |
| `load_marian.py` | Copies the weights of a released translation model into `model.py` |
| `toy_task.py` | A small task: turn `2021-03-12` into `12 March 2021` |
| `train_toy.py` | Trains a small model on that task, on a laptop CPU, in about 4 minutes |

The comments in the code give the shape of each tensor. The same letters appear everywhere: `b` sentences in a batch, `n` source tokens, `t` target tokens, `h` heads, and the paper's `d_model`, `d_k` and `d_ff`.

> [!DEFINITION] Tensor and shape
> A tensor is a table of numbers with any number of axes. Its shape lists the size of each axis. A batch of 2 sentences with 10 tokens each, 512 numbers per token, has the shape (2, 10, 512).

For each block below, you see the lines of the paper, our code, and a check. The check runs the same input through our block and through PyTorch's block with the same weights. Then it prints the largest difference between the two outputs.

## Token embeddings {§3.4}

> [!PAPER] · §3.4
> [![Section 3.4 of the paper: the same weight matrix is shared between the two embedding layers and the pre-softmax linear transformation, and in the embedding layers the weights are multiplied by the square root of d_model](/img/papers/transformer/p7-embed.png)](/img/papers/transformer/p7-embed.png)
>
> **What it says:** "we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation". "In the embedding layers, we multiply those weights by $$\sqrt{d_{\text{model}}}$$."
>
> **What it means:**
>
> One table does three jobs. It turns each source token into a vector, and it does the same for each target token. At the end, it turns the decoder's output vector back into a score for each token. In code, that is one `nn.Embedding` that three places use.
>
> The multiplication by $$\sqrt{d_{\text{model}}}$$ sets the size of the numbers. We start the table with small values, about $$1/\sqrt{d_{\text{model}}}$$, which suit the final layer. The embedding layers then scale them up to about 1, the same size as the position vectors that get added next. [Part 2](part-2-the-architecture.md) explains this choice in detail.

```python
class TokenEmbedding(nn.Module):
    """Look up one learned vector per token, then multiply it by sqrt(d_model) (§3.4)."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.table = nn.Embedding(vocab_size, d_model)  # (vocab_size, d_model): one row per token
        self.scale = math.sqrt(d_model)                 # 22.6 for d_model = 512
        # Small start values (std 1/sqrt(d_model)), so that after the scale each number has std about 1.
        nn.init.normal_(self.table.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, ids):                     # ids: (b, n) integers
        return self.table(ids) * self.scale     # (b, n, d_model)
```

The check makes a table of 1,000 tokens with $$d_{\text{model}} = 512$$ and looks up three of them:

```text
== 1. Token embedding (§3.4)
ids (1, 3) -> vectors (1, 3, 512)
std of a table row: 0.0441   after x sqrt(512) = x 22.63: 0.9599
```

The table starts with a standard deviation of 0.0441, which is close to $$1/\sqrt{512} = 0.0442$$. After the scale, the vectors have a standard deviation of 0.96, close to 1. Here the library and our code are the same thing, because `nn.Embedding` is only a lookup table.

## Positional encoding {§3.5}

> [!PAPER] · §3.5
> [![The positional encoding formula of section 3.5: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))](/img/papers/transformer/p7-pe.png)](/img/papers/transformer/p7-pe.png)
>
> **What it says:** "we use sine and cosine functions of different frequencies", with $$PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{\text{model}}})$$ and $$PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{\text{model}}})$$.
>
> **What it means:**
>
> Attention alone does not know the order of the tokens. If you shuffle the input, each token gets the same output, only shuffled. So the model adds a fixed vector to each token, and the vector is different for each position.
>
> Think of a row of clocks that run at different speeds. The first hand turns once every 6.3 positions, and the last hand needs about 63,000 positions for one turn. Read all the hands together and you know the position. The even columns hold the sines and the odd columns hold the cosines. [Part 2](part-2-the-architecture.md) shows why a shift by $$k$$ positions is a rotation of each pair of columns.

The code computes the table once. Note the `layout` option. The paper puts the sine and the cosine of each frequency side by side. Marian, the released model we load later, puts all the sines first and all the cosines after them.

```python
def sinusoid_table(max_len, d_model, layout='interleaved'):
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)        # (max_len, 1): 0, 1, 2, ...
    i2 = torch.arange(0, d_model, 2, dtype=torch.float32)                # (d_model/2,): 0, 2, 4, ... (this is 2i)
    freq = 1.0 / (10000.0 ** (i2 / d_model))                             # (d_model/2,): from 1 down to about 1/10000
    angle = pos * freq                                                   # (max_len, d_model/2)
    pe = torch.zeros(max_len, d_model)
    if layout == 'interleaved':
        pe[:, 0::2] = torch.sin(angle)                                   # even columns 0, 2, 4, ...
        pe[:, 1::2] = torch.cos(angle)                                   # odd columns 1, 3, 5, ...
    elif layout == 'sin_then_cos':
        half = d_model // 2
        pe[:, :half] = torch.sin(angle)                                  # first half: all the sines
        pe[:, half:] = torch.cos(angle)                                  # second half: all the cosines
    else:
        raise ValueError(layout)
    return pe                                                            # (max_len, d_model)


class PositionalEncoding(nn.Module):
    """Add the fixed sinusoid vector of each position to the token vectors (§3.5). Nothing here is learned."""

    def __init__(self, d_model, max_len=512, layout='interleaved'):
        super().__init__()
        # A buffer is saved with the model but is not a trained parameter.
        self.register_buffer('pe', sinusoid_table(max_len, d_model, layout), persistent=False)

    def forward(self, x):                       # x: (b, n, d_model)
        n = x.size(1)
        return x + self.pe[:n].unsqueeze(0)     # (1, n, d_model) is added to every sentence in the batch
```

With a small $$d_{\text{model}} = 8$$ we can print the whole vector of position 1 in both layouts:

```text
== 2. Positional encoding (§3.5)
position 1, interleaved  (sin, cos, sin, cos, ...): tensor([0.8415, 0.5403, 0.0998, 0.9950, 0.0100, 0.9999, 0.0010, 1.0000])
position 1, sin_then_cos (sin, sin, ..., cos, cos): tensor([0.8415, 0.0998, 0.0100, 0.0010, 0.5403, 0.9950, 0.9999, 1.0000])
same numbers, other column order: True
```

The first number is $$\sin(1) = 0.8415$$ and the second is $$\cos(1) = 0.5403$$. The slower clocks barely move at position 1: 0.0998, then 0.0100, then 0.0010. Both layouts hold the same eight numbers. A model trained with one layout only works with that layout, so the loader must pick the right one.

## Scaled dot-product attention {§3.2.1}

> [!PAPER] · §3.2.1
> [![Section 3.2.1: compute the dot products of the query with all keys, divide each by the square root of d_k and apply a softmax to obtain the weights on the values, and Equation 1](/img/papers/transformer/p7-eq1.png)](/img/papers/transformer/p7-eq1.png)
>
> **What it says:** "We compute the dot products of the query with all keys, divide each by $$\sqrt{d_k}$$, and apply a softmax function to obtain the weights on the values." Then Equation 1: $$\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d_k})V$$.
>
> **What it means:**
>
> Each token asks a question (its query) and every token offers a label (its key) and a content (its value). The dot product of a query and a key says how well they match. The softmax turns the match scores of one query into weights that add up to 1. The output is the weighted average of the values.
>
> The division by $$\sqrt{d_k}$$ keeps the scores at a size where the softmax still learns. Without it, long vectors give large scores, the softmax puts almost all weight on one key, and the gradients become very small. [Part 3](part-3-attention.md) works through this with numbers.

The code follows the equation line by line. One function, four steps:

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (..., n_q, n_k): how well each query matches each key
    if mask is not None:
        # A very negative score becomes 0 after softmax: the hidden positions get no weight.
        scores = scores.masked_fill(~mask, -1e9)
    weights = torch.softmax(scores, dim=-1)              # (..., n_q, n_k): each row sums to 1
    output = weights @ V                                 # (..., n_q, d_v): weighted average of the values
    return output, weights
```

The `...` in the shapes means "any axes in front". The same function works for one sentence, for a batch, and for a batch with eight heads. The `@` sign is matrix multiplication, and it works on the last two axes.

We test it on the example of [Part 3](part-3-attention.md): three queries $$(1, 0)$$, $$(0, 1)$$, $$(1, 1)$$, three keys $$(2, 0)$$, $$(0, 2)$$, $$(1, 1)$$, and three values $$(10, 0)$$, $$(0, 10)$$, $$(5, 5)$$. Then we compare with PyTorch's own function, `torch.nn.functional.scaled_dot_product_attention`, on random tensors of the real size: 2 sentences, 8 heads, 10 tokens, $$d_k = 64$$.

```text
== 3. Scaled dot-product attention (Eq. 1)
weights (each row sums to 1):
tensor([[0.5760, 0.1400, 0.2840],
        [0.1400, 0.5760, 0.2840],
        [0.3333, 0.3333, 0.3333]])
output:
tensor([[7.1797, 2.8203],
        [2.8203, 7.1797],
        [5.0000, 5.0000]])
ours vs torch.nn.functional.scaled_dot_product_attention, max abs difference: 3.58e-07
with the causal mask, max abs difference: 3.58e-07
```

The first row is the hand calculation of Part 3: weights 0.576, 0.140 and 0.284, output (7.18, 2.82). The third query matches all keys equally, so it gets one third of each value. Against the library, the largest difference is 0.000000358. A 32-bit float holds about seven digits, so this is rounding noise. The library adds the numbers in a different order, nothing more.

## Multi-head attention {§3.2.2}

> [!PAPER] · §3.2.2
> [![Section 3.2.2: the head outputs are concatenated and once again projected; MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O with head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V); h = 8 heads with d_k = d_v = d_model / h = 64](/img/papers/transformer/p7-multihead.png)](/img/papers/transformer/p7-multihead.png)
>
> **What it says:** "These are concatenated and once again projected". $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$, where $$\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$. "we employ $$h = 8$$ parallel attention layers", with $$d_k = d_v = d_{\text{model}}/h = 64$$.
>
> **What it means:**
>
> One attention can only form one weighted average per token. Eight heads form eight averages, each with its own projections. One head can follow the previous word while another follows the subject of the verb. Each head works in a narrow space of 64 numbers, so eight heads cost about as much as one wide head.
>
> The paper writes eight small matrices $$W^Q_i$$, each 512 by 64. In code we put them side by side in one 512 by 512 matrix. One multiplication then gives all eight projections at once, and we cut the result into eight pieces of 64. The maths is the same; the cut is the only new step.

Here is our class. The two helper functions, `split_heads` and `merge_heads`, do the cutting and the gluing. They only move numbers around; they do not compute anything.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True, attn_dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be a multiple of the number of heads'
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=bias)   # all h query projections W_i^Q, side by side
        self.W_k = nn.Linear(d_model, d_model, bias=bias)   # all h key projections W_i^K
        self.W_v = nn.Linear(d_model, d_model, bias=bias)   # all h value projections W_i^V
        self.W_o = nn.Linear(d_model, d_model, bias=bias)   # W^O mixes the concatenated heads
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.last_weights = None                            # kept after each call, for drawing attention maps

    def split_heads(self, x):
        """(b, n, d_model) -> (b, h, n, d_k): cut each vector into h pieces, one per head."""
        b, n, _ = x.shape
        x = x.view(b, n, self.h, self.d_k)                  # (b, n, h, d_k)
        return x.transpose(1, 2)                            # (b, h, n, d_k): heads become a batch dimension

    def merge_heads(self, x):
        """(b, h, n, d_k) -> (b, n, d_model): put the h head outputs back side by side (the Concat)."""
        b, h, n, d_k = x.shape
        x = x.transpose(1, 2)                               # (b, n, h, d_k)
        return x.reshape(b, n, h * d_k)                     # (b, n, d_model)

    def forward(self, query, key, value, mask=None):
        # query: (b, n_q, d_model); key and value: (b, n_k, d_model)
        # In self-attention all three are the same tensor. In cross-attention key = value = encoder output.
        Q = self.split_heads(self.W_q(query))               # (b, h, n_q, d_k)
        K = self.split_heads(self.W_k(key))                 # (b, h, n_k, d_k)
        V = self.split_heads(self.W_v(value))               # (b, h, n_k, d_k)
        out, weights = scaled_dot_product_attention(Q, K, V, mask)   # out: (b, h, n_q, d_k)
        self.last_weights = weights                         # (b, h, n_q, n_k)
        out = self.merge_heads(out)                         # (b, n_q, d_model)
        return self.W_o(out)                                # (b, n_q, d_model)
```

Follow one token through `split_heads`. Its vector of 512 numbers becomes 8 rows of 64 numbers. Numbers 0 to 63 go to head 0, numbers 64 to 127 go to head 1, and so on. The `transpose` moves the head axis to the front, so the attention function sees 8 independent problems and solves them in one call. `merge_heads` reverses both steps. That reverse is the Concat of the paper.

> [!DEFINITION] Bias
> A learned vector added after a matrix multiplication: $$xW + b$$. The paper's attention formulas have no bias, but `nn.Linear` has one by default, and most released models keep it. Our class has a `bias` switch for both cases.

**Method 2, with the library.** PyTorch's `nn.MultiheadAttention` does the same job. It keeps the three input matrices stacked in one tensor of shape (1536, 512) called `in_proj_weight`. We copy our weights into it and run both:

```python
lib = nn.MultiheadAttention(d_model, h, batch_first=True).eval()
with torch.no_grad():
    # PyTorch keeps W_q, W_k, W_v stacked in one (3 * d_model, d_model) matrix called in_proj_weight.
    lib.in_proj_weight.copy_(torch.cat([ours.W_q.weight, ours.W_k.weight, ours.W_v.weight], dim=0))
    lib.in_proj_bias.copy_(torch.cat([ours.W_q.bias, ours.W_k.bias, ours.W_v.bias], dim=0))
    lib.out_proj.weight.copy_(ours.W_o.weight)
    lib.out_proj.bias.copy_(ours.W_o.bias)
x = torch.randn(2, 10, d_model)                          # (b=2, n=10, d_model)
ids = torch.tensor([[3] * 10, [3] * 7 + [0] * 3])        # the second sentence has 3 pad tokens at the end
with torch.no_grad():
    a = ours(x, x, x, padding_mask(ids, 0))              # our mask: True = may look
    b, wb = lib(x, x, x, key_padding_mask=(ids == 0),    # PyTorch's mask: True = pad, do NOT look
                average_attn_weights=False)
```

Note the two mask conventions. In our code `True` means "may look". In PyTorch's `key_padding_mask`, `True` means "pad, do not look". Mixing them up is a common bug, and the model still runs, only wrong.

```text
== 4. Multi-head attention (§3.2.2) vs torch.nn.MultiheadAttention
output (2, 10, 512) weights (2, 8, 10, 10)
self-attention with padding, max abs difference of outputs: 2.53e-07
max abs difference of the 8 heads' weights: 7.45e-08
cross-attention (queries from another sequence), max abs difference: 3.28e-07
```

The outputs agree to 0.00000025, and all 8 heads give the same weights to 0.00000007. The cross-attention case uses 6 queries from another sequence against the 10 keys, the shape the decoder needs. It agrees too.

## The masks {§3.2.3}

> [!PAPER] · §3.2.3
> [![Section 3.2.3: self-attention in the decoder lets each position attend to all positions up to and including that position, implemented by masking out (setting to minus infinity) the illegal connections](/img/papers/transformer/p7-mask.png)](/img/papers/transformer/p7-mask.png)
>
> **What it says:** each decoder position may attend to "all positions in the decoder up to and including that position". This is done "by masking out (setting to $$-\infty$$) all values in the input of the softmax which correspond to illegal connections."
>
> **What it means:**
>
> During training, the decoder sees the whole correct translation at once. Without a mask, position 3 could read word 4 and copy it, and the model would learn nothing useful. The mask hides every later position, so each position predicts the next word from earlier words only.
>
> The trick is in the softmax. A score of $$-\infty$$ becomes $$e^{-\infty} = 0$$, so a hidden position gets weight 0. The visible positions share all the weight. The same trick hides padding tokens, which only fill short sentences up to the length of the longest one.

Our code builds both masks as tables of `True` and `False`:

```python
def padding_mask(ids, pad_id):
    """(b, n) ids -> (b, 1, 1, n) bool. False at padding, so no query looks at a pad token."""
    return (ids != pad_id).unsqueeze(1).unsqueeze(2)


def causal_mask(t, device=None):
    """(1, 1, t, t) bool. Row i is True for columns 0..i: position i may look at itself and earlier positions only."""
    i = torch.arange(t, device=device).unsqueeze(1)    # (t, 1) query positions
    j = torch.arange(t, device=device).unsqueeze(0)    # (1, t) key positions
    return (j <= i).unsqueeze(0).unsqueeze(0)          # lower triangle, including the diagonal
```

```text
== 5. Masks (True = may look)
causal_mask(4):
tensor([[1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]], dtype=torch.int32)
padding_mask for ids [7, 8, 9, 0, 0] with pad id 0: tensor([1, 1, 1, 0, 0], dtype=torch.int32)
```

Row $$i$$ of the causal mask is the list of positions that token $$i$$ may read. The extra axes of size 1 let one mask serve every sentence and every head. PyTorch copies the mask along those axes without using memory; this is broadcasting.

Two details in our code differ from the paper's words, and both are deliberate.

First, we write $$-10^9$$, not $$-\infty$$. After the softmax both give exactly 0. But with $$-\infty$$, a row where every position is hidden gives $$0/0$$, and the model outputs "not a number".

Second, the decoder uses only the causal mask, not a padding mask. Targets are padded on the right, so the causal mask already hides every pad from every real token. Marian, the model we load later, also starts each output with its pad token, so a padding mask would hide that start token.

## The feed-forward network {§3.3}

> [!PAPER] · §3.3
> [![Section 3.3: a fully connected feed-forward network applied to each position separately and identically, two linear transformations with a ReLU activation in between, FFN(x) = max(0, x W1 + b1) W2 + b2, with d_model = 512 and d_ff = 2048](/img/papers/transformer/p7-ffn.png)](/img/papers/transformer/p7-ffn.png)
>
> **What it says:** a feed-forward network "applied to each position separately and identically", made of "two linear transformations with a ReLU activation in between": $$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$, with an inner size $$d_{ff} = 2048$$.
>
> **What it means:**
>
> Attention moves information between tokens. The feed-forward network then works on each token alone. It widens the vector from 512 to 2048 numbers, keeps the positive ones, and narrows it back to 512. The same weights serve every position, but each layer has its own weights.
>
> "Separately" is the key word. In code, `nn.Linear` acts on the last axis only. A tensor of shape (b, n, 512) becomes (b, n, 2048), with no mixing between tokens. We do not need a loop over positions.

```python
class FeedForward(nn.Module):
    """§3.3: FFN(x) = max(0, x W_1 + b_1) W_2 + b_2, applied to each position on its own."""

    def __init__(self, d_model, d_ff, activation='relu', dropout=0.0):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)    # 512 -> 2048: widen
        self.W_2 = nn.Linear(d_ff, d_model)    # 2048 -> 512: back to the model width
        self.activation = activation
        self.dropout = nn.Dropout(dropout)     # 0 in the paper; some code drops here too

    def forward(self, x):                      # x: (b, n, d_model)
        hidden = self.W_1(x)                   # (b, n, d_ff)
        if self.activation == 'relu':          # this `if` reads a fixed setting, not tensor data: export is fine
            hidden = F.relu(hidden)            # max(0, .) as in the paper
        else:
            hidden = F.silu(hidden)            # swish(x) = x * sigmoid(x), used by Marian
        return self.W_2(self.dropout(hidden))  # (b, n, d_model)
```

> [!DEFINITION] Swish
> The function $$\text{swish}(x) = x \cdot \sigma(x)$$, where $$\sigma$$ is the sigmoid. For large positive $$x$$ it is close to $$x$$, and for large negative $$x$$ it is close to 0, like ReLU. Near 0 it is smooth. Marian uses swish in place of ReLU; PyTorch calls it `silu`.

The check for this block comes with the encoder layer below, because PyTorch has no separate feed-forward class.

## Residual connection and layer norm {§3.1}

> [!PAPER] · §3.1
> [![Section 3.1, the encoder: a residual connection around each of the two sub-layers, followed by layer normalization, so the output of each sub-layer is LayerNorm(x + Sublayer(x))](/img/papers/transformer/p7-residual.png)](/img/papers/transformer/p7-residual.png)
>
> **What it says:** "We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is $$\text{LayerNorm}(x + \text{Sublayer}(x))$$".
>
> **What it means:**
>
> The residual connection adds the input of a sub-layer to its output. The sub-layer only has to learn a change to the vector, not a whole new vector. If a layer has nothing useful to add, it can output values near zero and the token passes through unchanged.
>
> Layer norm then rescales each token vector to mean 0 and standard deviation 1, and applies a learned scale and shift. This keeps the numbers at a steady size through 12 layers. The norm comes after the addition, so this layout is called "post-norm". Many later models moved the norm before the sub-layer ("pre-norm"), because it trains more stably in deep stacks. [Part 2](part-2-the-architecture.md) shows both with numbers.

> [!PAPER] · §5.4
> [![Section 5.4, residual dropout: dropout is applied to the output of each sub-layer before it is added to the sub-layer input and normalized, and to the sums of the embeddings and the positional encodings, with a rate of 0.1 for the base model](/img/papers/transformer/p7-dropout.png)](/img/papers/transformer/p7-dropout.png)
>
> **What it says:** "We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings". The rate is $$P_{drop} = 0.1$$.
>
> **What it means:**
>
> This line from the training section tells us where dropout goes inside the residual block. It acts on the sub-layer's output only, never on the residual path. So the full rule is $$\text{LayerNorm}(x + \text{Dropout}(\text{Sublayer}(x)))$$, and that is one line of code.
>
> During training, dropout sets 10% of the numbers to zero at random and scales the rest up by $$1/0.9$$. In evaluation mode it does nothing. That is why every check on this page calls `.eval()` first, or sets the dropout rate to 0.

```python
class AddAndNorm(nn.Module):
    """§3.1 and §5.4: output = LayerNorm(x + Dropout(Sublayer(x))). This is "post-norm", as in the paper."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)      # eps 1e-5, the same as Marian and nn.TransformerEncoderLayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_out):        # both: (b, n, d_model)
        return self.norm(x + self.dropout(sublayer_out))
```

The paper does not give the small number $$\epsilon$$ that layer norm adds before it divides by the standard deviation. We use $$10^{-5}$$, the value in PyTorch and in Marian.

## The encoder layer {§3.1}

Now the blocks combine. An encoder layer is self-attention, then the feed-forward network, each wrapped in `AddAndNorm`. The `forward` function is two lines, and each line is one sub-layer of Figure 1.

```python
class EncoderLayer(nn.Module):
    """Self-attention, then feed-forward. Each one is wrapped in AddAndNorm."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.bias)
        self.add_norm_1 = AddAndNorm(cfg.d_model, cfg.dropout)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.activation)
        self.add_norm_2 = AddAndNorm(cfg.d_model, cfg.dropout)

    def forward(self, x, src_mask):            # x: (b, n, d_model); src_mask: (b, 1, 1, n)
        x = self.add_norm_1(x, self.self_attn(x, x, x, src_mask))   # every word looks at every (non-pad) word
        x = self.add_norm_2(x, self.ffn(x))                         # then each position is processed alone
        return x                                                    # (b, n, d_model)
```

**Method 2, with the library.** PyTorch has a ready encoder layer, `nn.TransformerEncoderLayer`. With `norm_first=False` and `activation='relu'` it follows the paper. We copy our weights into it, one tensor at a time. Before that, we give our layer norms random scales and shifts, so that the check also tests them.

```python
lib = nn.TransformerEncoderLayer(512, 8, dim_feedforward=2048, dropout=0.0, activation='relu',
                                 batch_first=True, norm_first=False)
```

```python
    copy_attn(lib.self_attn, ours.self_attn)
    copy_linear_norm(lib.linear1, ours.ffn.W_1)          # FFN first matrix W_1, b_1
    copy_linear_norm(lib.linear2, ours.ffn.W_2)          # FFN second matrix W_2, b_2
    copy_linear_norm(lib.norm1, ours.add_norm_1.norm)    # norm after self-attention
    copy_linear_norm(lib.norm2, ours.add_norm_2.norm)    # norm after the FFN
```

```text
== 6. Encoder layer vs nn.TransformerEncoderLayer (post-norm, ReLU)
output (2, 10, 512)  max abs difference: 1.91e-06
```

The difference is 0.0000019. It is a little larger than for attention alone, because the rounding noise passes through more steps and two layer norms. It is still noise: a real bug, such as a wrong mask or a missing residual, gives differences near 1.

## The decoder layer {§3.1}

> [!PAPER] · §3.1
> [![Section 3.1, the decoder: a third sub-layer performs multi-head attention over the output of the encoder stack, the self-attention is masked, and the output embeddings are offset by one position](/img/papers/transformer/p7-decoder.png)](/img/papers/transformer/p7-decoder.png)
>
> **What it says:** "the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack." The self-attention is masked, and "the output embeddings are offset by one position".
>
> **What it means:**
>
> The decoder layer has three sub-layers. Masked self-attention lets each target position read the earlier target words. Cross-attention lets it read the whole source sentence: its queries come from the decoder, and its keys and values come from the encoder output. The feed-forward network then works on each position.
>
> "Offset by one position" is the shift that makes training work. The decoder input is the target sentence moved one step to the right, with a start token in front. So at position $$i$$ the decoder sees the words before $$i$$ and must predict word $$i$$. We meet this shift again in the training loop.

```python
class DecoderLayer(nn.Module):
    """Masked self-attention, then cross-attention to the encoder output, then feed-forward."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.bias)
        self.add_norm_1 = AddAndNorm(cfg.d_model, cfg.dropout)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.bias)
        self.add_norm_2 = AddAndNorm(cfg.d_model, cfg.dropout)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.activation)
        self.add_norm_3 = AddAndNorm(cfg.d_model, cfg.dropout)

    def forward(self, y, memory, tgt_mask, src_mask):
        # y: (b, t, d_model) target so far; memory: (b, n, d_model) encoder output
        # tgt_mask: (1, 1, t, t) causal; src_mask: (b, 1, 1, n)
        y = self.add_norm_1(y, self.self_attn(y, y, y, tgt_mask))            # look only at earlier target words
        y = self.add_norm_2(y, self.cross_attn(y, memory, memory, src_mask)) # queries from y, keys/values from source
        y = self.add_norm_3(y, self.ffn(y))
        return y                                                             # (b, t, d_model)
```

Look at the call `self.cross_attn(y, memory, memory, src_mask)`. The only difference from self-attention is which tensors go in. The same `MultiHeadAttention` class serves all three uses of attention in the paper.

The check against `nn.TransformerDecoderLayer` works the same way. PyTorch calls the cross-attention `multihead_attn`, and its causal mask uses $$-\infty$$ where ours uses `False`:

```text
== 7. Decoder layer vs nn.TransformerDecoderLayer (post-norm, ReLU)
output (2, 6, 512)  max abs difference: 9.54e-07
```

## The whole model {§3.4}

The two stacks repeat their layer $$N$$ times. Each stack first embeds the tokens, adds the positions and applies dropout. The `Transformer` class holds one embedding table and gives it to both stacks.

```python
class Encoder(nn.Module):
    def __init__(self, cfg: Config, embedding: TokenEmbedding):
        super().__init__()
        self.embed = embedding                                  # shared with the decoder (§3.4)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.pos_layout)
        self.dropout = nn.Dropout(cfg.dropout)                  # §5.4: dropout on embeddings + positions
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_encoder_layers)])

    def forward(self, src_ids, src_mask):                       # src_ids: (b, n)
        x = self.dropout(self.pos(self.embed(src_ids)))         # (b, n, d_model)
        for layer in self.layers:                               # N layers, each with its own weights
            x = layer(x, src_mask)
        return x                                                # (b, n, d_model): the "memory"
```

The `Decoder` class is the same with `DecoderLayer`. The full model then has three steps: encode the source, decode the target, and score every vocabulary token.

```python
    def encode(self, src_ids):
        """(b, n) ids -> memory (b, n, d_model) and the source padding mask (b, 1, 1, n)."""
        src_mask = padding_mask(src_ids, self.cfg.pad_id)
        return self.encoder(src_ids, src_mask), src_mask

    def decode(self, tgt_ids, memory, src_mask):
        """Target so far (b, t) -> logits (b, t, vocab_size). Row k predicts token k+1."""
        t = tgt_ids.size(1)
        # Only the causal mask. Targets are padded on the right, so a real token never sees a later pad anyway.
        # (And Marian starts the decoder with the pad id itself: a padding mask here would hide that start token.)
        tgt_mask = causal_mask(t, tgt_ids.device)               # (1, 1, t, t), broadcast over the batch
        y = self.decoder(tgt_ids, memory, tgt_mask, src_mask)   # (b, t, d_model)
        return self.generator(y)

    def generator(self, y):
        """The final Linear of Figure 1, tied to the embedding table: logits = y E^T (+ bias)."""
        return y @ self.embedding.table.weight.T + self.logits_bias   # (b, t, vocab_size)
```

> [!DEFINITION] Logits
> The raw scores the model gives to every token of the vocabulary, before the softmax. A higher logit means a more likely token. The softmax turns them into probabilities; the loss function does that step itself.

The `generator` is the shared matrix at work. The embedding table $$E$$ has one row per token. Multiplying the decoder output by $$E^\top$$ gives the dot product of the output vector with every token's vector. The token whose vector points the same way gets the highest logit.

We keep `encode` and `decode` apart on purpose. When the model writes a sentence, the source is encoded once and the decoder runs once per new token. The split also lets a tool such as ONNX export the encoder and one decoder step as two separate graphs. No `forward` function has a Python `if` that depends on the numbers inside a tensor, so the export records the same steps for every input.

### How many parameters?

Table 3 of the paper says the base model has 65 million parameters and the big model 213 million. We build both configurations from our code with a vocabulary of 37,000 tokens, the size §5.1 gives, and count:

```text
== 8. Parameter count from our code (paper, Table 3: base 65M, big 213M)
base: d_model 512, h 8, d_ff 2048, vocabulary 37,000
  shared embedding 18,944,000 | one encoder layer 3,152,384 | one decoder layer 4,204,032
  total 63,082,496 = 63.1M
  by hand, one encoder layer: attention 1,050,624 + FFN 2,099,712 + 2 norms 2,048 = 3,152,384
  share of the shared embedding table: 30.0%
big: d_model 1024, h 16, d_ff 4096, vocabulary 37,000
  shared embedding 37,888,000 | one encoder layer 12,596,224 | one decoder layer 16,796,672
  total 214,245,376 = 214.2M
```

You can check the encoder layer by hand. Attention has four 512 by 512 matrices with biases: $$4 \times (512 \times 512 + 512) = 1{,}050{,}624$$. The feed-forward network has $$512 \times 2048 + 2048 + 2048 \times 512 + 512 = 2{,}099{,}712$$. Two layer norms add $$2 \times 1{,}024 = 2{,}048$$. The sum is 3,152,384. A decoder layer adds one more attention and one more norm.

Our counts are 63.1M and 214.2M, against 65M and 213M in the paper. The paper only says "about 37000" tokens, and it does not list exactly what it counted, so we cannot close the last few percent. The order of size matches, and so does the split: in the base model, the shared table alone is 30% of all weights.

## Loading a released model: Marian

A from-scratch model that only matches single layers could still be wrong as a whole. The strongest test is to load real trained weights and get real translations. We use `Helsinki-NLP/opus-mt-en-de`, an English to German model with the paper's base shape: 6 + 6 layers, $$d_{\text{model}} = 512$$, 8 heads, $$d_{ff} = 2048$$. It was trained with the Marian toolkit; the `transformers` library runs it as `MarianMTModel`.

Marian differs from the paper in four small ways. Each one is a switch in our `Config`, so the same code runs both.

| Difference | The paper | Marian | Our switch |
|---|---|---|---|
| Biases in $$W^Q, W^K, W^V, W^O$$ | none in the formulas | yes | `bias=True` |
| Extra vector added to the logits | none | `final_logits_bias` | `final_logits_bias=True` |
| Feed-forward activation | ReLU | swish | `activation='swish'` |
| Position table layout | sin, cos, sin, cos, ... | all sines, then all cosines | `pos_layout='sin_then_cos'` |

The loader maps every tensor by name. Marian calls the query matrix of layer 3 `model.encoder.layers.3.self_attn.q_proj.weight`; we call it `encoder.layers.3.self_attn.W_q.weight`. A table holds the part names and a few lines handle the layer norms:

```python
LAYER_PARTS = {
    'self_attn.q_proj': 'self_attn.W_q',
    'self_attn.k_proj': 'self_attn.W_k',
    'self_attn.v_proj': 'self_attn.W_v',
    'self_attn.out_proj': 'self_attn.W_o',
    'encoder_attn.q_proj': 'cross_attn.W_q',          # decoder only: queries from the decoder
    'encoder_attn.k_proj': 'cross_attn.W_k',          # keys and values from the encoder output
    'encoder_attn.v_proj': 'cross_attn.W_v',
    'encoder_attn.out_proj': 'cross_attn.W_o',
    'self_attn_layer_norm': 'add_norm_1.norm',        # after self-attention
    'fc1': 'ffn.W_1',
    'fc2': 'ffn.W_2',
}
```

```python
def our_name(marian_name):
    """Map one Marian tensor name to our tensor name. Returns None for tensors we do not copy (with the reason)."""
    if marian_name == 'model.shared.weight':
        return 'embedding.table.weight'                       # the one shared table (§3.4)
    if marian_name in ('model.encoder.embed_tokens.weight', 'model.decoder.embed_tokens.weight', 'lm_head.weight'):
        return None                                           # the same table again (tied); checked below
    if marian_name.endswith('embed_positions.weight'):
        return None                                           # fixed sinusoids, not learned; checked below
    if marian_name == 'final_logits_bias':
        return 'logits_bias'                                  # difference 2
    m = re.match(r'model\.(encoder|decoder)\.layers\.(\d+)\.(.+)\.(weight|bias)$', marian_name)
    if not m:
        raise KeyError(marian_name)
    stack, i, part, kind = m.groups()
    if part == 'final_layer_norm':                            # the norm after the feed-forward network
        part_ours = 'add_norm_2.norm' if stack == 'encoder' else 'add_norm_3.norm'
    elif part == 'encoder_attn_layer_norm':                   # decoder only: the norm after cross-attention
        part_ours = 'add_norm_2.norm'
    else:
        part_ours = LAYER_PARTS[part]
    return f'{stack}.layers.{i}.{part_ours}.{kind}'
```

Both libraries store an `nn.Linear` weight as (output size, input size), so each tensor copies with no transpose. After the copy, the loader checks that every one of our learned tensors was filled. It also checks that the three tied copies in Marian are really one table, and that Marian's position table equals ours.

```text
== 1. Copy the weights
Marian tensors: 259, copied into our model: 254
not copied: encoder/decoder embed_tokens and lm_head (the same table as model.shared),
            encoder/decoder embed_positions (fixed sinusoids, not learned)
our sin_then_cos table vs Marian's position table, max abs difference: 3.4e-05
parameters: ours 73,886,208, Marian 74,410,496
gap: 524,288 = 2 x 512 x 512, the two fixed position tables that Marian stores as (frozen) parameters
```

Of the 259 tensors, 254 are copied. The other five are explained: three are the shared table again, and two are position tables that our code computes. Our table differs from Marian's by at most 0.000034, because the two compute the sines with different float precision. The parameter counts differ by exactly the two position tables.

Now the real test. We send five English sentences through both models. First the encoder output, then the logits for a whole German sentence, then complete translations:

```text
== 2. Encoder output
src (5, 15) -> memory (5, 15, 512); max abs difference on real tokens: 1.91e-06
size of the numbers: max abs value 9.03

== 3. Decoder logits (teacher forcing with Marian's own German translation)
target (5, 15) -> logits (5, 15, 58101); max abs difference: 2.67e-05
same top token at 100.0% of 55 positions

== 4. Greedy translation: ours vs MarianMTModel.generate(num_beams=1)
EN    The cat sat on the mat.
ours  Die Katze saß auf der Matte.
        (same as MarianMTModel)
EN    Attention is all you need.
ours  Achtung ist alles, was du brauchst.
        (same as MarianMTModel)
EN    The weather is nice today, so we will walk to the station.
ours  Das Wetter ist heute schön, so dass wir zum Bahnhof gehen.
        (same as MarianMTModel)
EN    I have read this book twice.
ours  Ich habe dieses Buch zweimal gelesen.
        (same as MarianMTModel)
EN    Neural networks learn from examples.
ours  Neuronale Netzwerke lernen aus Beispielen.
        (same as MarianMTModel)
identical: 5 of 5
```

After six encoder layers, the outputs agree to 0.0000019, on numbers as large as 9. After six more decoder layers and a projection onto 58,101 tokens, the logits agree to 0.000027. The top token is the same at all 55 real positions, so all five translations are identical, word for word.

"Achtung ist alles, was du brauchst" is a fun error. "Achtung" means attention in the sense of "watch out!". Both models make it, because both are the same model. Our code reproduces the trained network, including its mistakes.

## Writing a sentence: greedy decoding and beam search {§6.1}

The model predicts one token at a time. To write a whole sentence, we give it the start token and pick a next token. We add that token to the input and repeat until the model writes the end token.

> [!DEFINITION] Greedy decoding
> At each step, take the single most likely next token. It is fast, but an early choice can never be undone, even if it leads to a poor sentence.

```python
@torch.no_grad()
def greedy_decode(model, src_ids, bos_id, eos_id, max_len=64):
    """Encode once, then add the most probable next token, one step at a time, until every row has eos.
    src_ids: (b, n). Returns (b, <= max_len + 1) ids that start with bos_id."""
    model.eval()
    memory, src_mask = model.encode(src_ids)
    b = src_ids.size(0)
    out = torch.full((b, 1), bos_id, dtype=torch.long, device=src_ids.device)   # (b, 1): start token
    done = torch.zeros(b, dtype=torch.bool, device=src_ids.device)
    for _ in range(max_len):
        logits = model.decode(out, memory, src_mask)          # (b, t, vocab): we re-run the whole prefix
        next_id = logits[:, -1].argmax(-1)                    # (b,): best token after the last position
        next_id = torch.where(done, torch.full_like(next_id, model.cfg.pad_id), next_id)  # pad finished rows
        out = torch.cat([out, next_id.unsqueeze(1)], dim=1)
        done = done | (next_id == eos_id)
        if bool(done.all()):
            break
    return out
```

Our loop runs the decoder on the whole prefix at each step. That repeats work: real systems keep the keys and values of earlier positions in a cache. We keep the slow version because it has fewer moving parts, and the outputs are the same.

> [!PAPER] · §6.1
> [![Section 6.1: beam search with a beam size of 4 and length penalty alpha = 0.6, and a maximum output length of input length + 50](/img/papers/transformer/p7-beam.png)](/img/papers/transformer/p7-beam.png)
>
> **What it says:** "We used beam search with a beam size of 4 and length penalty $$\alpha = 0.6$$". The maximum output length is "input length + 50".
>
> **What it means:**
>
> Beam search keeps the 4 best partial sentences, not only one. At each step it extends each of them by every possible token, scores all the new candidates, and keeps the best 4 again. A choice that looks second best now can win two words later.
>
> The score of a sentence is the sum of the log-probabilities of its tokens. Each token adds a negative number, so long sentences always score lower. The length penalty divides the score by $$((5 + |Y|)/6)^{\alpha}$$, where $$|Y|$$ is the length, so that a good long sentence can beat a short one.

> [!DEFINITION] Beam search
> A search that keeps the $$k$$ best partial outputs at every step (the beam). With $$k = 1$$ it is greedy decoding. The paper uses $$k = 4$$.

Our `beam_search` in `model.py` follows that description for one sentence. On two of our sentences:

```text
== 5. Our beam search (beam 4, length penalty alpha 0.6, as in §6.1)
EN    The weather is nice today, so we will walk to the station.
beam  Das Wetter ist heute schön, also gehen wir zum Bahnhof.
EN    I have read this book twice.
beam  Ich habe dieses Buch zweimal gelesen.
```

For the short sentence, beam search finds the greedy answer again. For the long one, it finds a different sentence: "also gehen wir zum Bahnhof" ("so we go to the station") in place of "so dass wir zum Bahnhof gehen" ("so that we go to the station"). The beam version is closer to the English meaning. One example proves nothing about quality, but it shows that the two searches can disagree.

## Training a small Transformer

Matching a trained model proves that our forward pass is right. It does not prove that our code can learn. For that we train a model from random weights.

The task must need attention and run on a laptop. We chose dates. The input is an ISO date such as `1965-05-10`, and the output is the same date for people: `10 May 1965`.

| Input | Output |
|---|---|
| `1965-05-10` | `10 May 1965` |
| `1963-09-01` | `1 September 1963` |

The task is small but not trivial. The day, the month and the year change places, so each output character must find the right input character. The two month digits must become a word. The day loses its leading zero, so outputs have different lengths. There are 73,049 dates from 1900 to 2099; we keep 2,000 aside for testing and never train on them.

Both sides use characters as tokens, from one shared vocabulary of 41 tokens. It holds three special tokens (pad, start, end), the digits, the dash, the space and the letters of the month names. The model is the paper's design at a smaller size: $$d_{\text{model}} = 128$$, 4 heads, $$d_{ff} = 512$$, 2 encoder and 2 decoder layers, dropout 0.1. It has 930,944 parameters, and the file is 3.8 MB.

### The training loop, line by line

Here is the core of `train_toy.py`. Everything in it comes from our own `model.py` and from the paper.

```python
def make_batch(dates):
    src = pad_batch([encode_src(iso(d)) for d in dates])      # (b, 11): 10 characters + </s>
    tgt = pad_batch([encode_tgt(human(d)) for d in dates])    # (b, t+1): <s> ... </s>
    return src, tgt


def lrate(step):
    """Eq. 3: lrate = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5). Step counts from 1."""
    step = max(step, 1)
    return cfg.d_model ** -0.5 * min(step ** -0.5, step * WARMUP ** -1.5)
```

```python
    # Adam with the paper's settings (§5.3): beta1 = 0.9, beta2 = 0.98, eps = 1e-9. lr=1 so LambdaLR gives Eq. 3.
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: lrate(i + 1))

    log, snapshots, t0 = [], {}, time.time()
    for step in range(1, STEPS + 1):
        model.train()                                          # dropout on
        src, tgt = make_batch(random.sample(train_dates, BATCH))
        dec_in = tgt[:, :-1]                                   # "shifted right": <s> 1 2   M a r c h ...
        labels = tgt[:, 1:]                                    # what must come next: 1 2   M a r c h ... </s>
        logits = model(src, dec_in)                            # (b, t, vocab): teacher forcing, all positions at once
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labels.reshape(-1),
                               ignore_index=PAD, label_smoothing=SMOOTHING)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
```

**The batch.** Each step takes 128 random training dates. The source is the 10 characters plus an end token. The target is a start token, the characters, and an end token. Shorter targets get pad tokens on the right.

**Shifting the target right.** This is the "offset by one position" of §3.1. Take the target `<s> 1 0 ␣ M a y ... </s>`. The decoder input `dec_in` drops the last token, and the labels drop the first token:

| position | 0 | 1 | 2 | 3 | 4 | ... |
|---|---|---|---|---|---|---|
| decoder input | `<s>` | `1` | `0` | `␣` | `M` | ... |
| label (must predict) | `1` | `0` | `␣` | `M` | `a` | ... |

At each position, the label is the next character after the input. The causal mask inside `model.decode` makes sure that position 3 sees only `<s> 1 0 ␣`, so it cannot copy the `M` from the input.

> [!DEFINITION] Teacher forcing
> During training, the decoder reads the correct previous tokens, not its own guesses. All positions are then trained in one pass, in parallel. At test time there is no correct answer, so the model reads its own output, one token at a time.

**One forward pass for all positions.** `model(src, dec_in)` returns logits of shape (128, t, 41): one score for every vocabulary token, at every position of every date. Thanks to teacher forcing and the causal mask, this one call trains all positions at once. An RNN would need a loop over positions here.

**The loss.** `F.cross_entropy` compares the logits with the labels. `ignore_index=PAD` skips the padded positions, so the model is not graded on filler. `label_smoothing=0.1` is the paper's $$\epsilon_{ls}$$.

> [!PAPER] · §5.4
> [![Section 5.4, label smoothing: during training, label smoothing of value 0.1 is used; this hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score](/img/papers/transformer/p7-smoothing.png)](/img/papers/transformer/p7-smoothing.png)
>
> **What it says:** "we employed label smoothing of value $$\epsilon_{ls} = 0.1$$". "This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score."
>
> **What it means:**
>
> Without smoothing, the target says "the right token has probability 1, every other token 0". With smoothing, the target moves 10% of the probability away from the right token and spreads it over the vocabulary. The model is never asked to be fully certain.
>
> This has a visible effect on the loss: it can never reach 0. The best the model can do is to match the smoothed target exactly, and that target has a fixed amount of uncertainty. Our training curve below hits this floor. [Part 5](part-5-training.md) works through label smoothing with a five-word example.

**The learning rate.** `lrate` is Equation 3 of the paper, word for word. PyTorch's `LambdaLR` multiplies the base rate by the function's value, so we set the base rate to 1.

> [!PAPER] · §5.3
> [![Section 5.3: the Adam optimizer with beta1 = 0.9, beta2 = 0.98 and epsilon = 10^-9, the learning rate formula of Equation 3, and warmup_steps = 4000](/img/papers/transformer/p7-eq3.png)](/img/papers/transformer/p7-eq3.png)
>
> **What it says:** Adam with "$$\beta_1 = 0.9$$, $$\beta_2 = 0.98$$ and $$\epsilon = 10^{-9}$$", and $$lrate = d_{\text{model}}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})$$, with "$$warmup\_steps = 4000$$".
>
> **What it means:**
>
> The rate climbs in a straight line for the first warm-up steps, then falls slowly, with one over the square root of the step. At the start, the weights are random and Adam's running averages are still poor, so large steps would be risky. The climb gives Adam time to settle.
>
> The two parts of the $$\min$$ are equal at the end of the warm-up. Before that the straight line is smaller; after it, the falling curve is smaller. So the peak is $$d_{\text{model}}^{-0.5} \cdot warmup\_steps^{-0.5}$$. [Part 5](part-5-training.md) plots the curve for the paper's settings.

Our model is small and our run is short, so we use 400 warm-up steps in place of 4,000. With $$d_{\text{model}} = 128$$, the peak is $$128^{-0.5} \times 400^{-0.5} = 0.0884 \times 0.05 = 0.00442$$. The training log shows the rate for the next step, so at step 400 it prints $$128^{-0.5} \times 401^{-0.5}$$:

```text
{"step": 400, "loss": 0.7041, "token_acc": 0.9958, "lr": 0.004414, "seconds": 53.5}
```

**The update.** `zero_grad` clears the old gradients, `backward` computes new ones, and `opt.step` changes the weights. Then `sched.step` moves the learning rate to the next value of Equation 3. That is one training step. We run 3,000 of them.

## What the small model learned

The run took 249 seconds on a laptop CPU. Every 50 steps the script logs the loss and the share of correct characters. Every 250 steps it also writes out 500 held-out dates with greedy decoding and counts the ones that are exactly right.

```text
== 2. Training (3,000 steps of 128 dates, CPU)
step     1  loss 4.383  token accuracy 0.021  exact match on 500 test dates 0.000
step   100  loss 0.998  token accuracy 0.907
step   200  loss 0.715  token accuracy 0.995
step   300  loss 0.699  token accuracy 0.996
step   500  loss 0.704  token accuracy 0.995  exact match on 500 test dates 0.958
step  1000  loss 0.686  token accuracy 0.999  exact match on 500 test dates 1.000
step  2000  loss 0.685  token accuracy 0.999  exact match on 500 test dates 1.000
step  3000  loss 0.684  token accuracy 0.999  exact match on 500 test dates 1.000
training time: 249 s
uniform guess over 41 tokens: ln 41 = 3.714
smoothed target: right token 0.9024, each other token 0.00244
entropy terms: 0.0926 + 0.5869
lowest possible loss with label smoothing 0.1 and 41 classes: 0.680
Eq. 3 with d_model 128, warmup 400: 128^-0.5 = 0.0884, 400^-0.5 = 0.05, peak 0.00442
```

<figure class="fig"><svg viewBox="0 0 780 320" role="img" aria-label="Training loss of the date model over 3,000 steps of 128 dates. It falls from 4.38 at step 1 to about 1.0 at step 100 and 0.70 by step 300, then stays just above the lowest value label smoothing allows, 0.680. The step axis is logarithmic."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="270.0" x2="500" y2="270.0"/><text class="t-tick" x="56.0" y="274.0" text-anchor="end">0</text><line class="grid" x1="64" y1="216.7" x2="500" y2="216.7"/><text class="t-tick" x="56.0" y="220.7" text-anchor="end">1</text><line class="grid" x1="64" y1="163.3" x2="500" y2="163.3"/><text class="t-tick" x="56.0" y="167.3" text-anchor="end">2</text><line class="grid" x1="64" y1="110.0" x2="500" y2="110.0"/><text class="t-tick" x="56.0" y="114.0" text-anchor="end">3</text><line class="grid" x1="64" y1="56.7" x2="500" y2="56.7"/><text class="t-tick" x="56.0" y="60.7" text-anchor="end">4</text><text class="t-tick" x="64.0" y="290.0" text-anchor="middle">1</text><text class="t-tick" x="277.0" y="290.0" text-anchor="middle">50</text><text class="t-tick" x="314.8" y="290.0" text-anchor="middle">100</text><text class="t-tick" x="374.6" y="290.0" text-anchor="middle">300</text><text class="t-tick" x="440.2" y="290.0" text-anchor="middle">1,000</text><text class="t-tick" x="500.0" y="290.0" text-anchor="middle">3,000</text><line class="axis" x1="64" y1="270" x2="500" y2="270"/><polyline class="l1" points="64.0,36.2 277.0,152.8 314.8,216.8 336.9,229.4 352.5,231.9 364.7,231.4 374.6,232.7 390.3,232.4 402.4,232.5 424.5,233.0 440.2,233.4 462.3,233.0 477.9,233.5 490.1,233.6 500.0,233.5"/><g class="mark"><title>training loss, 1: 4.383</title><circle class="s1 ring" cx="64.0" cy="36.2" r="5"/></g><g class="mark"><title>training loss, 50: 2.198</title><circle class="s1 ring" cx="277.0" cy="152.8" r="5"/></g><g class="mark"><title>training loss, 100: 0.998</title><circle class="s1 ring" cx="314.8" cy="216.8" r="5"/></g><g class="mark"><title>training loss, 150: 0.761</title><circle class="s1 ring" cx="336.9" cy="229.4" r="5"/></g><g class="mark"><title>training loss, 200: 0.715</title><circle class="s1 ring" cx="352.5" cy="231.9" r="5"/></g><g class="mark"><title>training loss, 250: 0.724</title><circle class="s1 ring" cx="364.7" cy="231.4" r="5"/></g><g class="mark"><title>training loss, 300: 0.699</title><circle class="s1 ring" cx="374.6" cy="232.7" r="5"/></g><g class="mark"><title>training loss, 400: 0.704</title><circle class="s1 ring" cx="390.3" cy="232.4" r="5"/></g><g class="mark"><title>training loss, 500: 0.704</title><circle class="s1 ring" cx="402.4" cy="232.5" r="5"/></g><g class="mark"><title>training loss, 750: 0.693</title><circle class="s1 ring" cx="424.5" cy="233.0" r="5"/></g><g class="mark"><title>training loss, 1,000: 0.686</title><circle class="s1 ring" cx="440.2" cy="233.4" r="5"/></g><g class="mark"><title>training loss, 1,500: 0.694</title><circle class="s1 ring" cx="462.3" cy="233.0" r="5"/></g><g class="mark"><title>training loss, 2,000: 0.685</title><circle class="s1 ring" cx="477.9" cy="233.5" r="5"/></g><g class="mark"><title>training loss, 2,500: 0.682</title><circle class="s1 ring" cx="490.1" cy="233.6" r="5"/></g><g class="mark"><title>training loss, 3,000: 0.684</title><circle class="s1 ring" cx="500.0" cy="233.5" r="5"/></g><polyline class="l2" points="64.0,233.8 277.0,233.8 314.8,233.8 336.9,233.8 352.5,233.8 364.7,233.8 374.6,233.8 390.3,233.8 402.4,233.8 424.5,233.8 440.2,233.8 462.3,233.8 477.9,233.8 490.1,233.8 500.0,233.8"/><g class="mark"><title>lowest possible (label smoothing), 1: 0.680</title><circle class="s2 ring" cx="64.0" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 50: 0.680</title><circle class="s2 ring" cx="277.0" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 100: 0.680</title><circle class="s2 ring" cx="314.8" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 150: 0.680</title><circle class="s2 ring" cx="336.9" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 200: 0.680</title><circle class="s2 ring" cx="352.5" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 250: 0.680</title><circle class="s2 ring" cx="364.7" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 300: 0.680</title><circle class="s2 ring" cx="374.6" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 400: 0.680</title><circle class="s2 ring" cx="390.3" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 500: 0.680</title><circle class="s2 ring" cx="402.4" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 750: 0.680</title><circle class="s2 ring" cx="424.5" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 1,000: 0.680</title><circle class="s2 ring" cx="440.2" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 1,500: 0.680</title><circle class="s2 ring" cx="462.3" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 2,000: 0.680</title><circle class="s2 ring" cx="477.9" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 2,500: 0.680</title><circle class="s2 ring" cx="490.1" cy="233.8" r="5"/></g><g class="mark"><title>lowest possible (label smoothing), 3,000: 0.680</title><circle class="s2 ring" cx="500.0" cy="233.8" r="5"/></g><line class="grid" x1="506.0" y1="233.5" x2="516.0" y2="233.5"/><rect class="s1" x="520.0" y="228.5" width="10" height="10" rx="2"/><text class="t-note" x="536.0" y="237.5" text-anchor="start">training loss: 0.684</text><line class="grid" x1="506.0" y1="233.8" x2="516.0" y2="250.5"/><rect class="s2" x="520.0" y="245.5" width="10" height="10" rx="2"/><text class="t-note" x="536.0" y="254.5" text-anchor="start">lowest possible (label smoothing): 0.680</text><text class="t-tick" x="282.0" y="310.0" text-anchor="middle">training step (log scale)</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">loss</text></svg><figcaption>The training loss of the date model, with a logarithmic step axis. It falls from 4.38 to about 1.0 in 100 steps, then flattens just above 0.680, the lowest loss that label smoothing allows.</figcaption></figure>

At step 1 the loss is 4.38, a little above $$\ln 41 = 3.714$$, the loss of a model that guesses uniformly over 41 tokens. By step 100 it is below 1, and from step 300 it stays near 0.70. It never goes below 0.68, and that is the label smoothing floor at work.

We can compute the floor. PyTorch spreads $$\epsilon = 0.1$$ over all 41 tokens, so the smoothed target gives the right token $$0.9 + 0.1/41 = 0.9024$$ and each other token $$0.1/41 = 0.00244$$. The lowest loss is the entropy of that target:

$$
-\left(0.9024 \ln 0.9024 + 40 \times 0.00244 \ln 0.00244\right) = 0.0926 + 0.5869 = 0.680
$$

Our final loss is 0.684. The model has matched the smoothed target almost exactly; there is nothing left to learn from this loss.

<figure class="fig"><svg viewBox="0 0 780 280" role="img" aria-label="Exact match on 500 held-out dates, measured every 250 steps with greedy decoding. 0 at step 1, 0.896 at step 250, 0.998 at step 750, 1.000 at step 1,000 and at the end, with dips to 0.954 at step 1,250 and 0.978 at step 2,750."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="230.0" x2="520" y2="230.0"/><text class="t-tick" x="56.0" y="234.0" text-anchor="end">0</text><line class="grid" x1="64" y1="180.0" x2="520" y2="180.0"/><text class="t-tick" x="56.0" y="184.0" text-anchor="end">0.25</text><line class="grid" x1="64" y1="130.0" x2="520" y2="130.0"/><text class="t-tick" x="56.0" y="134.0" text-anchor="end">0.50</text><line class="grid" x1="64" y1="80.0" x2="520" y2="80.0"/><text class="t-tick" x="56.0" y="84.0" text-anchor="end">0.75</text><line class="grid" x1="64" y1="30.0" x2="520" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">1.00</text><text class="t-tick" x="64.0" y="250.0" text-anchor="middle">1</text><text class="t-tick" x="139.9" y="250.0" text-anchor="middle">500</text><text class="t-tick" x="215.9" y="250.0" text-anchor="middle">1,000</text><text class="t-tick" x="291.9" y="250.0" text-anchor="middle">1,500</text><text class="t-tick" x="367.9" y="250.0" text-anchor="middle">2,000</text><text class="t-tick" x="444.0" y="250.0" text-anchor="middle">2,500</text><text class="t-tick" x="520.0" y="250.0" text-anchor="middle">3,000</text><line class="axis" x1="64" y1="230" x2="520" y2="230"/><polyline class="l3" points="64.0,230.0 101.9,50.8 139.9,38.4 177.9,30.4 215.9,30.0 253.9,39.2 291.9,32.0 329.9,30.0 367.9,30.0 406.0,30.0 444.0,30.0 482.0,34.4 520.0,30.0"/><g class="mark"><title>exact match, held-out dates, 1: 0.00</title><circle class="s3 ring" cx="64.0" cy="230.0" r="5"/></g><g class="mark"><title>exact match, held-out dates, 250: 0.90</title><circle class="s3 ring" cx="101.9" cy="50.8" r="5"/></g><g class="mark"><title>exact match, held-out dates, 500: 0.96</title><circle class="s3 ring" cx="139.9" cy="38.4" r="5"/></g><g class="mark"><title>exact match, held-out dates, 750: 1.00</title><circle class="s3 ring" cx="177.9" cy="30.4" r="5"/></g><g class="mark"><title>exact match, held-out dates, 1,000: 1.00</title><circle class="s3 ring" cx="215.9" cy="30.0" r="5"/></g><g class="mark"><title>exact match, held-out dates, 1,250: 0.95</title><circle class="s3 ring" cx="253.9" cy="39.2" r="5"/></g><g class="mark"><title>exact match, held-out dates, 1,500: 0.99</title><circle class="s3 ring" cx="291.9" cy="32.0" r="5"/></g><g class="mark"><title>exact match, held-out dates, 1,750: 1.00</title><circle class="s3 ring" cx="329.9" cy="30.0" r="5"/></g><g class="mark"><title>exact match, held-out dates, 2,000: 1.00</title><circle class="s3 ring" cx="367.9" cy="30.0" r="5"/></g><g class="mark"><title>exact match, held-out dates, 2,250: 1.00</title><circle class="s3 ring" cx="406.0" cy="30.0" r="5"/></g><g class="mark"><title>exact match, held-out dates, 2,500: 1.00</title><circle class="s3 ring" cx="444.0" cy="30.0" r="5"/></g><g class="mark"><title>exact match, held-out dates, 2,750: 0.98</title><circle class="s3 ring" cx="482.0" cy="34.4" r="5"/></g><g class="mark"><title>exact match, held-out dates, 3,000: 1.00</title><circle class="s3 ring" cx="520.0" cy="30.0" r="5"/></g><line class="grid" x1="526.0" y1="30.0" x2="536.0" y2="30.0"/><rect class="s3" x="540.0" y="25.0" width="10" height="10" rx="2"/><text class="t-note" x="556.0" y="34.0" text-anchor="start">exact match, held-out dates: 1.00</text><text class="t-tick" x="292.0" y="270.0" text-anchor="middle">training step</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">share of dates right</text></svg><figcaption>Exact match on 500 held-out dates, checked every 250 steps with greedy decoding. It reaches 1.00 by step 1,000, with two short dips while the learning rate is still high.</figcaption></figure>

The exact match is strict: one wrong character makes the whole date wrong. It reaches 0.896 at step 250 and 1.000 at step 1,000. Twice it dips, to 0.954 at step 1,250 and to 0.978 at step 2,750, and then it recovers. The learning rate is still above 0.0016 at the end, so each step still moves the weights a little.

### Right outputs, wrong outputs

At the end, the model writes all 2,000 held-out dates correctly. The first six:

```text
dates 1900 to 2099: 73,049; training 71,049, held out 2,000
held-out test dates (never seen in training): exact match 100.0% of 2,000
  1965-05-10  ->  10 May 1965
  1915-06-20  ->  20 June 1915
  1977-03-18  ->  18 March 1977
  1968-04-22  ->  22 April 1968
  1956-07-26  ->  26 July 1956
  1963-09-01  ->  1 September 1963
```

The early checkpoints show how it got there. We saved the outputs for the first three test dates at a few steps:

```text
early checkpoints, same first three test dates:
  step    1: 'OOOOOOOOOOOOOOOOOOOOOOOO'   'O'                    'OOOOOOOOOOOOOOOOOOOOOOOO'
  step  100: '10 May 1965'          '20 June 19555'        '18 March 1977'     
  step  200: '10 May 1965'          '20 June 1915'         '18 March 1977'     
  step  300: '10 May 1965'          '20 June 1915'         '18 March 1977'     
  step  500: '10 May 1965'          '20 June 1915'         '18 March 1977'     
  step 1000: '10 May 1965'          '20 June 1915'         '18 March 1977'     
```

At step 1 the random model repeats one letter until it hits the length limit. By step 100 it already knows the format, but it writes `19555` for the year 1915. It has learned where the year is, but not yet when to stop copying it. By step 200 these three are right.

The most useful failures come from inputs unlike the training data:

```text
outside what it was trained on:
  2150-07-04  ->  4 July 1950
  1850-11-23  ->  23 November 1950
  2021-02-30  ->  30 February 2021
  2021-13-01  ->  1 March 2021
  0001-01-01  ->  1 January 2001
```

The model never saw a year outside 1900 to 2099, so it forces 2150 and 1850 into that range: both become 1950. It writes "30 February" without complaint, because it copies digits and never learned a calendar. For month 13 it writes "March", the month whose second digit is 3. And year 0001 becomes 2001. A model learns the patterns in its data, and nothing more.

### Where the model looks

The task needs attention, so attention should show the alignment. After training, each `MultiHeadAttention` keeps its last weights in `last_weights`. We feed the correct output with teacher forcing and read the cross-attention of both decoder layers, for all 4 heads.

<figure class="fig"><svg viewBox="0 0 770 400" role="img" aria-label="Cross-attention of the trained date model for 1965-05-10 → 10 May 1965. Rows are the output characters, columns the input characters, darker means more weight. Head 2 of decoder layer 2 sends each output digit to the input digit it copies, including the right one of the two ones. Head 3 sends the first month letter to the month digits."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="198.0" y="28.0" text-anchor="middle">layer 2, head 2: copy the digit</text><text class="t-tick" x="58.0" y="56.0" text-anchor="end">1</text><g class="mark"><title>1 → 1: 0.00</title><rect class="cell" x="67.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 9: 0.00</title><rect class="cell" x="91.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 6: 0.00</title><rect class="cell" x="115.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 5: 0.00</title><rect class="cell" x="139.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → -: 0.00</title><rect class="cell" x="163.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 0: 0.00</title><rect class="cell" x="187.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 5: 0.00</title><rect class="cell" x="211.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → -: 0.00</title><rect class="cell" x="235.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 1: 1.00</title><rect class="cell" x="259.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:1.000"/></g><g class="mark"><title>1 → 0: 0.00</title><rect class="cell" x="283.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → &lt;/s&gt;: 0.00</title><rect class="cell" x="307.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="58.0" y="80.0" text-anchor="end">0</text><g class="mark"><title>0 → 1: 0.00</title><rect class="cell" x="67.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 9: 0.00</title><rect class="cell" x="91.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 6: 0.00</title><rect class="cell" x="115.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 5: 0.00</title><rect class="cell" x="139.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → -: 0.00</title><rect class="cell" x="163.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 0: 0.00</title><rect class="cell" x="187.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 5: 0.00</title><rect class="cell" x="211.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → -: 0.00</title><rect class="cell" x="235.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 1: 0.00</title><rect class="cell" x="259.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 0: 1.00</title><rect class="cell" x="283.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:1.000"/></g><g class="mark"><title>0 → &lt;/s&gt;: 0.00</title><rect class="cell" x="307.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="58.0" y="104.0" text-anchor="end">␣</text><g class="mark"><title>␣ → 1: 0.00</title><rect class="cell" x="67.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 9: 0.00</title><rect class="cell" x="91.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 6: 0.00</title><rect class="cell" x="115.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 5: 0.00</title><rect class="cell" x="139.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → -: 0.03</title><rect class="cell" x="163.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 0: 0.08</title><rect class="cell" x="187.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.077"/></g><g class="mark"><title>␣ → 5: 0.00</title><rect class="cell" x="211.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → -: 0.89</title><rect class="cell" x="235.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.894"/></g><g class="mark"><title>␣ → 1: 0.00</title><rect class="cell" x="259.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 0: 0.00</title><rect class="cell" x="283.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → &lt;/s&gt;: 0.00</title><rect class="cell" x="307.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="58.0" y="128.0" text-anchor="end">M</text><g class="mark"><title>M → 1: 0.00</title><rect class="cell" x="67.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → 9: 0.00</title><rect class="cell" x="91.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → 6: 0.00</title><rect class="cell" x="115.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → 5: 0.00</title><rect class="cell" x="139.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → -: 0.14</title><rect class="cell" x="163.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.143"/></g><g class="mark"><title>M → 0: 0.01</title><rect class="cell" x="187.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → 5: 0.00</title><rect class="cell" x="211.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → -: 0.85</title><rect class="cell" x="235.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.850"/></g><g class="mark"><title>M → 1: 0.00</title><rect class="cell" x="259.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → 0: 0.00</title><rect class="cell" x="283.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → &lt;/s&gt;: 0.00</title><rect class="cell" x="307.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="58.0" y="152.0" text-anchor="end">a</text><g class="mark"><title>a → 1: 0.00</title><rect class="cell" x="67.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → 9: 0.00</title><rect class="cell" x="91.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → 6: 0.00</title><rect class="cell" x="115.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → 5: 0.00</title><rect class="cell" x="139.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → -: 0.02</title><rect class="cell" x="163.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → 0: 0.03</title><rect class="cell" x="187.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → 5: 0.00</title><rect class="cell" x="211.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → -: 0.95</title><rect class="cell" x="235.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.949"/></g><g class="mark"><title>a → 1: 0.00</title><rect class="cell" x="259.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → 0: 0.00</title><rect class="cell" x="283.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → &lt;/s&gt;: 0.00</title><rect class="cell" x="307.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="58.0" y="176.0" text-anchor="end">y</text><g class="mark"><title>y → 1: 0.00</title><rect class="cell" x="67.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → 9: 0.00</title><rect class="cell" x="91.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → 6: 0.00</title><rect class="cell" x="115.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → 5: 0.00</title><rect class="cell" x="139.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → -: 0.01</title><rect class="cell" x="163.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → 0: 0.01</title><rect class="cell" x="187.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → 5: 0.00</title><rect class="cell" x="211.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → -: 0.97</title><rect class="cell" x="235.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.974"/></g><g class="mark"><title>y → 1: 0.00</title><rect class="cell" x="259.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → 0: 0.00</title><rect class="cell" x="283.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → &lt;/s&gt;: 0.01</title><rect class="cell" x="307.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="58.0" y="200.0" text-anchor="end">␣</text><g class="mark"><title>␣ → 1: 0.00</title><rect class="cell" x="67.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 9: 0.00</title><rect class="cell" x="91.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 6: 0.00</title><rect class="cell" x="115.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 5: 0.00</title><rect class="cell" x="139.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → -: 0.02</title><rect class="cell" x="163.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 0: 0.09</title><rect class="cell" x="187.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.093"/></g><g class="mark"><title>␣ → 5: 0.00</title><rect class="cell" x="211.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → -: 0.89</title><rect class="cell" x="235.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.887"/></g><g class="mark"><title>␣ → 1: 0.00</title><rect class="cell" x="259.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 0: 0.00</title><rect class="cell" x="283.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → &lt;/s&gt;: 0.00</title><rect class="cell" x="307.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="58.0" y="224.0" text-anchor="end">1</text><g class="mark"><title>1 → 1: 1.00</title><rect class="cell" x="67.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.997"/></g><g class="mark"><title>1 → 9: 0.00</title><rect class="cell" x="91.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 6: 0.00</title><rect class="cell" x="115.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 5: 0.00</title><rect class="cell" x="139.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → -: 0.00</title><rect class="cell" x="163.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 0: 0.00</title><rect class="cell" x="187.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 5: 0.00</title><rect class="cell" x="211.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → -: 0.00</title><rect class="cell" x="235.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 1: 0.00</title><rect class="cell" x="259.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 0: 0.00</title><rect class="cell" x="283.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → &lt;/s&gt;: 0.00</title><rect class="cell" x="307.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="58.0" y="248.0" text-anchor="end">9</text><g class="mark"><title>9 → 1: 0.03</title><rect class="cell" x="67.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → 9: 0.92</title><rect class="cell" x="91.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.922"/></g><g class="mark"><title>9 → 6: 0.00</title><rect class="cell" x="115.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → 5: 0.00</title><rect class="cell" x="139.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → -: 0.00</title><rect class="cell" x="163.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → 0: 0.04</title><rect class="cell" x="187.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → 5: 0.00</title><rect class="cell" x="211.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → -: 0.00</title><rect class="cell" x="235.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → 1: 0.00</title><rect class="cell" x="259.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → 0: 0.00</title><rect class="cell" x="283.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → &lt;/s&gt;: 0.00</title><rect class="cell" x="307.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="58.0" y="272.0" text-anchor="end">6</text><g class="mark"><title>6 → 1: 0.00</title><rect class="cell" x="67.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → 9: 0.00</title><rect class="cell" x="91.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → 6: 1.00</title><rect class="cell" x="115.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:1.000"/></g><g class="mark"><title>6 → 5: 0.00</title><rect class="cell" x="139.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → -: 0.00</title><rect class="cell" x="163.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → 0: 0.00</title><rect class="cell" x="187.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → 5: 0.00</title><rect class="cell" x="211.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → -: 0.00</title><rect class="cell" x="235.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → 1: 0.00</title><rect class="cell" x="259.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → 0: 0.00</title><rect class="cell" x="283.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → &lt;/s&gt;: 0.00</title><rect class="cell" x="307.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="58.0" y="296.0" text-anchor="end">5</text><g class="mark"><title>5 → 1: 0.00</title><rect class="cell" x="67.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 9: 0.00</title><rect class="cell" x="91.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 6: 0.00</title><rect class="cell" x="115.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 5: 1.00</title><rect class="cell" x="139.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:1.000"/></g><g class="mark"><title>5 → -: 0.00</title><rect class="cell" x="163.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 0: 0.00</title><rect class="cell" x="187.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 5: 0.00</title><rect class="cell" x="211.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → -: 0.00</title><rect class="cell" x="235.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 1: 0.00</title><rect class="cell" x="259.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 0: 0.00</title><rect class="cell" x="283.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → &lt;/s&gt;: 0.00</title><rect class="cell" x="307.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="58.0" y="320.0" text-anchor="end">&lt;/s&gt;</text><g class="mark"><title>&lt;/s&gt; → 1: 0.00</title><rect class="cell" x="67.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → 9: 0.00</title><rect class="cell" x="91.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → 6: 0.00</title><rect class="cell" x="115.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → 5: 0.08</title><rect class="cell" x="139.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.080"/></g><g class="mark"><title>&lt;/s&gt; → -: 0.75</title><rect class="cell" x="163.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.747"/></g><g class="mark"><title>&lt;/s&gt; → 0: 0.02</title><rect class="cell" x="187.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → 5: 0.00</title><rect class="cell" x="211.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → -: 0.14</title><rect class="cell" x="235.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.140"/></g><g class="mark"><title>&lt;/s&gt; → 1: 0.00</title><rect class="cell" x="259.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → 0: 0.00</title><rect class="cell" x="283.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → &lt;/s&gt;: 0.02</title><rect class="cell" x="307.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="78.0" y="338.0" text-anchor="end" transform="rotate(-50 78.0 338.0)">1</text><text class="t-tick" x="102.0" y="338.0" text-anchor="end" transform="rotate(-50 102.0 338.0)">9</text><text class="t-tick" x="126.0" y="338.0" text-anchor="end" transform="rotate(-50 126.0 338.0)">6</text><text class="t-tick" x="150.0" y="338.0" text-anchor="end" transform="rotate(-50 150.0 338.0)">5</text><text class="t-tick" x="174.0" y="338.0" text-anchor="end" transform="rotate(-50 174.0 338.0)">-</text><text class="t-tick" x="198.0" y="338.0" text-anchor="end" transform="rotate(-50 198.0 338.0)">0</text><text class="t-tick" x="222.0" y="338.0" text-anchor="end" transform="rotate(-50 222.0 338.0)">5</text><text class="t-tick" x="246.0" y="338.0" text-anchor="end" transform="rotate(-50 246.0 338.0)">-</text><text class="t-tick" x="270.0" y="338.0" text-anchor="end" transform="rotate(-50 270.0 338.0)">1</text><text class="t-tick" x="294.0" y="338.0" text-anchor="end" transform="rotate(-50 294.0 338.0)">0</text><text class="t-tick" x="318.0" y="338.0" text-anchor="end" transform="rotate(-50 318.0 338.0)">&lt;/s&gt;</text><text class="t-title" x="578.0" y="28.0" text-anchor="middle">layer 2, head 3: find the month</text><text class="t-tick" x="438.0" y="56.0" text-anchor="end">1</text><g class="mark"><title>1 → 1: 0.00</title><rect class="cell" x="447.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 9: 0.00</title><rect class="cell" x="471.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 6: 0.00</title><rect class="cell" x="495.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 5: 0.00</title><rect class="cell" x="519.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → -: 0.16</title><rect class="cell" x="543.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.160"/></g><g class="mark"><title>1 → 0: 0.06</title><rect class="cell" x="567.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.059"/></g><g class="mark"><title>1 → 5: 0.00</title><rect class="cell" x="591.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → -: 0.21</title><rect class="cell" x="615.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.214"/></g><g class="mark"><title>1 → 1: 0.13</title><rect class="cell" x="639.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.130"/></g><g class="mark"><title>1 → 0: 0.00</title><rect class="cell" x="663.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → &lt;/s&gt;: 0.44</title><rect class="cell" x="687.0" y="41.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.437"/></g><text class="t-tick" x="438.0" y="80.0" text-anchor="end">0</text><g class="mark"><title>0 → 1: 0.00</title><rect class="cell" x="447.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 9: 0.00</title><rect class="cell" x="471.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 6: 0.00</title><rect class="cell" x="495.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 5: 0.00</title><rect class="cell" x="519.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → -: 0.00</title><rect class="cell" x="543.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 0: 0.00</title><rect class="cell" x="567.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 5: 0.00</title><rect class="cell" x="591.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → -: 0.00</title><rect class="cell" x="615.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 1: 0.00</title><rect class="cell" x="639.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>0 → 0: 1.00</title><rect class="cell" x="663.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:1.000"/></g><g class="mark"><title>0 → &lt;/s&gt;: 0.00</title><rect class="cell" x="687.0" y="65.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="438.0" y="104.0" text-anchor="end">␣</text><g class="mark"><title>␣ → 1: 0.00</title><rect class="cell" x="447.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 9: 0.00</title><rect class="cell" x="471.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 6: 0.00</title><rect class="cell" x="495.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 5: 0.00</title><rect class="cell" x="519.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → -: 0.18</title><rect class="cell" x="543.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.177"/></g><g class="mark"><title>␣ → 0: 0.05</title><rect class="cell" x="567.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.051"/></g><g class="mark"><title>␣ → 5: 0.00</title><rect class="cell" x="591.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → -: 0.77</title><rect class="cell" x="615.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.772"/></g><g class="mark"><title>␣ → 1: 0.00</title><rect class="cell" x="639.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 0: 0.00</title><rect class="cell" x="663.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → &lt;/s&gt;: 0.00</title><rect class="cell" x="687.0" y="89.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="438.0" y="128.0" text-anchor="end">M</text><g class="mark"><title>M → 1: 0.00</title><rect class="cell" x="447.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → 9: 0.00</title><rect class="cell" x="471.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → 6: 0.00</title><rect class="cell" x="495.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → 5: 0.00</title><rect class="cell" x="519.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → -: 0.00</title><rect class="cell" x="543.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → 0: 0.00</title><rect class="cell" x="567.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → 5: 1.00</title><rect class="cell" x="591.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:1.000"/></g><g class="mark"><title>M → -: 0.00</title><rect class="cell" x="615.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → 1: 0.00</title><rect class="cell" x="639.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → 0: 0.00</title><rect class="cell" x="663.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>M → &lt;/s&gt;: 0.00</title><rect class="cell" x="687.0" y="113.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="438.0" y="152.0" text-anchor="end">a</text><g class="mark"><title>a → 1: 0.00</title><rect class="cell" x="447.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → 9: 0.00</title><rect class="cell" x="471.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → 6: 0.00</title><rect class="cell" x="495.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → 5: 0.00</title><rect class="cell" x="519.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → -: 0.10</title><rect class="cell" x="543.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.101"/></g><g class="mark"><title>a → 0: 0.08</title><rect class="cell" x="567.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.077"/></g><g class="mark"><title>a → 5: 0.00</title><rect class="cell" x="591.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → -: 0.82</title><rect class="cell" x="615.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.822"/></g><g class="mark"><title>a → 1: 0.00</title><rect class="cell" x="639.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → 0: 0.00</title><rect class="cell" x="663.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>a → &lt;/s&gt;: 0.00</title><rect class="cell" x="687.0" y="137.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="438.0" y="176.0" text-anchor="end">y</text><g class="mark"><title>y → 1: 0.00</title><rect class="cell" x="447.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → 9: 0.00</title><rect class="cell" x="471.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → 6: 0.00</title><rect class="cell" x="495.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → 5: 0.00</title><rect class="cell" x="519.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → -: 0.00</title><rect class="cell" x="543.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → 0: 0.00</title><rect class="cell" x="567.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → 5: 1.00</title><rect class="cell" x="591.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:1.000"/></g><g class="mark"><title>y → -: 0.00</title><rect class="cell" x="615.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → 1: 0.00</title><rect class="cell" x="639.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → 0: 0.00</title><rect class="cell" x="663.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>y → &lt;/s&gt;: 0.00</title><rect class="cell" x="687.0" y="161.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="438.0" y="200.0" text-anchor="end">␣</text><g class="mark"><title>␣ → 1: 0.00</title><rect class="cell" x="447.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 9: 0.00</title><rect class="cell" x="471.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 6: 0.00</title><rect class="cell" x="495.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 5: 0.00</title><rect class="cell" x="519.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → -: 0.14</title><rect class="cell" x="543.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.140"/></g><g class="mark"><title>␣ → 0: 0.06</title><rect class="cell" x="567.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.062"/></g><g class="mark"><title>␣ → 5: 0.00</title><rect class="cell" x="591.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → -: 0.80</title><rect class="cell" x="615.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.797"/></g><g class="mark"><title>␣ → 1: 0.00</title><rect class="cell" x="639.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → 0: 0.00</title><rect class="cell" x="663.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>␣ → &lt;/s&gt;: 0.00</title><rect class="cell" x="687.0" y="185.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="438.0" y="224.0" text-anchor="end">1</text><g class="mark"><title>1 → 1: 0.00</title><rect class="cell" x="447.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 9: 0.00</title><rect class="cell" x="471.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 6: 0.00</title><rect class="cell" x="495.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 5: 0.00</title><rect class="cell" x="519.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → -: 0.00</title><rect class="cell" x="543.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 0: 0.00</title><rect class="cell" x="567.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 5: 0.00</title><rect class="cell" x="591.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → -: 0.00</title><rect class="cell" x="615.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 1: 0.00</title><rect class="cell" x="639.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → 0: 0.00</title><rect class="cell" x="663.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>1 → &lt;/s&gt;: 1.00</title><rect class="cell" x="687.0" y="209.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.998"/></g><text class="t-tick" x="438.0" y="248.0" text-anchor="end">9</text><g class="mark"><title>9 → 1: 0.11</title><rect class="cell" x="447.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.114"/></g><g class="mark"><title>9 → 9: 0.89</title><rect class="cell" x="471.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.886"/></g><g class="mark"><title>9 → 6: 0.00</title><rect class="cell" x="495.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → 5: 0.00</title><rect class="cell" x="519.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → -: 0.00</title><rect class="cell" x="543.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → 0: 0.00</title><rect class="cell" x="567.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → 5: 0.00</title><rect class="cell" x="591.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → -: 0.00</title><rect class="cell" x="615.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → 1: 0.00</title><rect class="cell" x="639.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → 0: 0.00</title><rect class="cell" x="663.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>9 → &lt;/s&gt;: 0.00</title><rect class="cell" x="687.0" y="233.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="438.0" y="272.0" text-anchor="end">6</text><g class="mark"><title>6 → 1: 0.00</title><rect class="cell" x="447.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → 9: 0.00</title><rect class="cell" x="471.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → 6: 1.00</title><rect class="cell" x="495.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:1.000"/></g><g class="mark"><title>6 → 5: 0.00</title><rect class="cell" x="519.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → -: 0.00</title><rect class="cell" x="543.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → 0: 0.00</title><rect class="cell" x="567.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → 5: 0.00</title><rect class="cell" x="591.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → -: 0.00</title><rect class="cell" x="615.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → 1: 0.00</title><rect class="cell" x="639.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → 0: 0.00</title><rect class="cell" x="663.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>6 → &lt;/s&gt;: 0.00</title><rect class="cell" x="687.0" y="257.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="438.0" y="296.0" text-anchor="end">5</text><g class="mark"><title>5 → 1: 0.00</title><rect class="cell" x="447.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 9: 0.00</title><rect class="cell" x="471.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 6: 0.00</title><rect class="cell" x="495.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 5: 1.00</title><rect class="cell" x="519.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:1.000"/></g><g class="mark"><title>5 → -: 0.00</title><rect class="cell" x="543.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 0: 0.00</title><rect class="cell" x="567.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 5: 0.00</title><rect class="cell" x="591.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → -: 0.00</title><rect class="cell" x="615.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 1: 0.00</title><rect class="cell" x="639.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → 0: 0.00</title><rect class="cell" x="663.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>5 → &lt;/s&gt;: 0.00</title><rect class="cell" x="687.0" y="281.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="438.0" y="320.0" text-anchor="end">&lt;/s&gt;</text><g class="mark"><title>&lt;/s&gt; → 1: 0.00</title><rect class="cell" x="447.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → 9: 0.00</title><rect class="cell" x="471.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → 6: 0.00</title><rect class="cell" x="495.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → 5: 0.00</title><rect class="cell" x="519.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → -: 0.25</title><rect class="cell" x="543.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.247"/></g><g class="mark"><title>&lt;/s&gt; → 0: 0.02</title><rect class="cell" x="567.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → 5: 0.00</title><rect class="cell" x="591.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → -: 0.73</title><rect class="cell" x="615.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.728"/></g><g class="mark"><title>&lt;/s&gt; → 1: 0.00</title><rect class="cell" x="639.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → 0: 0.00</title><rect class="cell" x="663.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>&lt;/s&gt; → &lt;/s&gt;: 0.00</title><rect class="cell" x="687.0" y="305.0" width="22.0" height="22.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="458.0" y="338.0" text-anchor="end" transform="rotate(-50 458.0 338.0)">1</text><text class="t-tick" x="482.0" y="338.0" text-anchor="end" transform="rotate(-50 482.0 338.0)">9</text><text class="t-tick" x="506.0" y="338.0" text-anchor="end" transform="rotate(-50 506.0 338.0)">6</text><text class="t-tick" x="530.0" y="338.0" text-anchor="end" transform="rotate(-50 530.0 338.0)">5</text><text class="t-tick" x="554.0" y="338.0" text-anchor="end" transform="rotate(-50 554.0 338.0)">-</text><text class="t-tick" x="578.0" y="338.0" text-anchor="end" transform="rotate(-50 578.0 338.0)">0</text><text class="t-tick" x="602.0" y="338.0" text-anchor="end" transform="rotate(-50 602.0 338.0)">5</text><text class="t-tick" x="626.0" y="338.0" text-anchor="end" transform="rotate(-50 626.0 338.0)">-</text><text class="t-tick" x="650.0" y="338.0" text-anchor="end" transform="rotate(-50 650.0 338.0)">1</text><text class="t-tick" x="674.0" y="338.0" text-anchor="end" transform="rotate(-50 674.0 338.0)">0</text><text class="t-tick" x="698.0" y="338.0" text-anchor="end" transform="rotate(-50 698.0 338.0)">&lt;/s&gt;</text></svg><figcaption>Cross-attention in decoder layer 2 for 1965-05-10 to 10 May 1965. Rows are the output characters being predicted, columns are the input characters. Head 2 (left) sends each output digit to the input digit it copies. Head 3 (right) sends the first month letter to the month digit 5.</figcaption></figure>

Head 2 of layer 2 draws the alignment of the task. To write the day `10` it looks at the last two digits of the input. To write the year `1965` it looks at the first four, in order. The date has two `1`s, at positions 0 and 8, and the head picks the right one each time. Head 3 does a different job: when the model must write `M`, it looks at the month digit `5`.

The letters `a` and `y` look mostly at a dash. Once the model has written `M` after the month `05`, the spelling `ay` follows without reading the input. The heads park their attention on a fixed landmark when they need nothing.

One example could be luck, so we measured it on all 2,000 held-out dates. For each head we count how often its strongest weight lands on the expected place:

```text
2,000 held-out dates: 11,408 digits, 2,000 first month letters, 10,433 other month letters.
A = digit: the strongest look is at the exact input digit it copies
B = first month letter: the strongest look is at a month digit
C = other month letters: the strongest look is at a dash or </s>
  decoder layer 1, head 0:  A   1.7%   B   2.5%   C  95.7%
  decoder layer 1, head 1:  A  29.9%   B   8.8%   C  68.0%
  decoder layer 1, head 2:  A   3.0%   B   0.0%   C  92.0%
  decoder layer 1, head 3:  A  12.3%   B  91.8%   C   9.5%
  decoder layer 2, head 0:  A  82.5%   B  16.6%   C  93.9%
  decoder layer 2, head 1:  A  82.2%   B  95.0%   C 100.0%
  decoder layer 2, head 2:  A  99.4%   B  25.6%   C 100.0%
  decoder layer 2, head 3:  A  70.8%   B 100.0%   C  67.9%
```

Layer 2, head 2 looks at the exact digit it copies 99.4% of the time. Layer 2, head 3 finds the month digits for the first month letter 100% of the time. Layer 1, head 3 already does so 91.8% of the time. Most heads of layer 1 do little cross-attention work: they look at landmarks. The heads split the job without being told to. The paper's appendix reports the same kind of division of labour in its full-size model (see [Part 6](part-6-results-and-impact.md)).

## Run it yourself

The code needs Python, `torch`, and for the Marian part `transformers` and `sentencepiece`. Everything runs on a laptop CPU.

```bash
git clone https://github.com/ishwar6/ishwar-books
cd ishwar-books/code/papers/transformer/scratch
python train_toy.py           # trains the date model: about 4 minutes, writes toy.pt and toy_log.json
python load_marian.py         # downloads opus-mt-en-de once (about 300 MB) and translates two sentences
cd ..
python tx_part7.py            # each block against PyTorch's layers, and the parameter counts
python tx_part7_marian.py     # our model with Marian's weights against MarianMTModel
python tx_part7_toy.py        # outputs, failures and cross-attention of the date model
```

Every printed block on this page comes from these scripts, and the logs are saved in `code/papers/transformer/results/`. To try your own date after training:

```python
import torch
from model import Config, Transformer, greedy_decode
from toy_task import encode_src, decode_ids, BOS, EOS

ck = torch.load('toy.pt')
model = Transformer(Config(**ck['config']))
model.load_state_dict(ck['state_dict'])
out = greedy_decode(model, torch.tensor([encode_src('2021-03-12')]), BOS, EOS, max_len=24)
print(decode_ids(out[0].tolist()))   # 12 March 2021
```

Some good next experiments:

- Train with `label_smoothing=0` and watch the loss go below 0.68.
- Set `n_heads=1` and look at the attention map again.
- Reverse the task and turn `12 March 2021` back into `2021-03-12`.

## What this code does not show

Our code is the paper's architecture, but not its full training system. We do not batch sentences by length, average checkpoints, or train for 100,000 steps on 8 GPUs. We never trained a full-size model, so we report no BLEU score of our own. The Marian weights come from a different training setup, with swish, biases and another vocabulary. Matching them proves our forward pass, not the paper's training recipe.

The date task is far easier than translation. It has fixed lengths and one correct answer, and the model reaches 100% on held-out dates. A translation model never does that. The task shows that the code learns and that attention finds the alignment, which is all we ask of it.

> [!TAKEAWAYS] Key takeaways
> - The whole Transformer fits in about 300 lines of PyTorch code, one small class per block of Figure 1. It uses only `nn.Linear`, `nn.Embedding`, `nn.LayerNorm` and `nn.Dropout`.
> - Multi-head attention is one projection per role, cut into $$h$$ pieces of $$d_k$$ numbers, attention on each piece, and a merge. Our version equals `nn.MultiheadAttention` to $$3 \times 10^{-7}$$.
> - Our encoder and decoder layers equal PyTorch's `nn.TransformerEncoderLayer` and `nn.TransformerDecoderLayer` (post-norm, ReLU) to about $$2 \times 10^{-6}$$.
> - From our code, the base configuration has 63.1M parameters and the big one 214.2M, against 65M and 213M in the paper.
> - With the weights of opus-mt-en-de, our model matches MarianMTModel: encoder outputs to $$1.9 \times 10^{-6}$$, logits to $$2.7 \times 10^{-5}$$, and 5 of 5 identical translations.
> - Training uses teacher forcing: the target shifted right is the decoder input, and the causal mask lets one pass train every position.
> - A model of 930,944 parameters learns to turn `1965-05-10` into `10 May 1965` in 4 minutes on a CPU, with 100% on 2,000 held-out dates.
- Its loss stops at 0.684, just above the floor of 0.680 that label smoothing sets.
> - In the trained model, one cross-attention head looks at the exact digit it copies 99.4% of the time. Another finds the month digits 100% of the time.

## The paper and the code

[Paper (arXiv 1706.03762)](https://arxiv.org/abs/1706.03762) · [Official code (tensor2tensor)](https://github.com/tensorflow/tensor2tensor) · [Our scripts and the from-scratch package](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/transformer)
