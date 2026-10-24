---
title: "Sharing, Scheduling and Many GPUs"
description: "How vLLM lets several answers share one prompt's memory, what it does when memory runs out, and how it runs one model on many GPUs. We read §4.4 to §4.6 and §5, and build each piece in Python."
part: 4
covers: "§4.4 to §4.6, §5, Figures 8 to 10"
date: 2026-10-24
tags: [llm-serving, inference, kv-cache, beam-search, tensor-parallelism]
---

Part 3 cut the KV cache into small blocks that can sit anywhere in GPU memory. This part shows what that freedom buys. Several answers can share one prompt's blocks. A full GPU can push requests out and bring them back. And one model can run on many GPUs with a single memory manager.

Each piece below shows the exact lines of the paper, then explains them. We build every mechanism in Python and check it against the library.

The map below is the paper's Figure 4, redrawn. Requests come in on the left. The scheduler in the middle decides which requests run, and its KV cache manager keeps the block tables. The workers on the right are the GPUs that run the model. Part 3 explained the block tables and the GPU block allocator.

This part adds the three highlighted pieces. They are the reference counts in the block tables (§4.4), the CPU block allocator (§4.5) and the model shards on several GPUs (§4.6).

{{FIG:p4_map|The parts of vLLM, after the paper's Figure 4. The highlighted boxes are the subject of this part: the reference counts that let block tables share blocks, the CPU block allocator that holds preempted requests, and the model shards that split one model over several GPUs.}}

## Blocks and block tables, in brief

Three words carry this whole part. Here they are again, in one line each.

> [!DEFINITION] KV block
> A fixed-size piece of the KV cache. It holds the keys and values of $$B$$ tokens, for example 4 or 16. When vLLM starts, it cuts all the cache memory of a GPU into such blocks.

> [!DEFINITION] Logical and physical block
> A **logical block** is a place in one request's own list: its first block, its second block, and so on. A **physical block** is a real block in GPU memory, with a number. Logical block 1 of a request can live in physical block 26.

> [!DEFINITION] Block table
> The list that maps each logical block of a request to its physical block. The table `[370, 26]` means: logical block 0 is physical block 370, and logical block 1 is physical block 26.

Two requests whose tables hold the same number point at the same memory. That one fact makes everything in §4.4 possible.

## Many answers from one prompt {§4.4}

> [!PAPER] · §4.4 · page 6
> [![The opening of §4.4, with greedy decoding and sampling, more complex decoding scenarios and more opportunities for memory sharing highlighted](/img/papers/vllm/p4-intro.png)](/img/papers/vllm/p4-intro.png)
>
> **What it says:** §4.3 covered "greedy decoding and sampling", where one prompt gives one answer. Real services must offer "more complex decoding scenarios", and these give "more opportunities for memory sharing".
>
> **What it means:**
>
> Until now, one request meant one sequence: one prompt, one answer, one block table. Many products break that rule. A code assistant shows three suggestions. A translator searches several candidate translations and keeps the best. A company puts the same long instructions in front of every user's question.
>
> In all these cases, several sequences start with the same tokens. The cache of a token depends only on the tokens before it, as Part 2 showed. So these sequences have identical caches for their common start. The question of this section is how to store that common part once.

> [!DEFINITION] Greedy decoding and sampling
> Two ways to pick the next token. **Greedy decoding** always takes the most likely token. **Sampling** draws the token at random, with the model's probabilities as weights, so the same prompt can give different answers.

### Parallel sampling {§4.4}

Start with a tiny example. A user sends the prompt "The best pizza topping is" and asks for three answers. The model samples three times and might write " mushrooms", " pineapple" and " extra cheese". This is **parallel sampling**: one prompt, several answers drawn at random, all made at the same time.

Each answer is its own sequence: the same prompt tokens, then its own new tokens. All three need the keys and values of the prompt tokens at every step. A system that gives each sequence its own contiguous cache stores the keys and values of the prompt three times. The keys and values are identical in all three copies, because they depend only on the prompt.

Where is this used? A code assistant shows several suggestions and lets the programmer choose. A chat service can show two answers and ask which one the user likes. Some systems draw many answers and keep the best one by some score. In all of these, the prompt is shared and the answers are not.

> [!PAPER] · §4.4 · pages 6 to 7
> [![The paragraph on parallel sampling, with multiple sampled outputs for a single input prompt highlighted](/img/papers/vllm/p4-sampling-1.png)](/img/papers/vllm/p4-sampling-1.png)
> [![Its continuation on page 7, with a request generates multiple sequences and the KV cache of the prompt to be shared highlighted](/img/papers/vllm/p4-sampling-2.png)](/img/papers/vllm/p4-sampling-2.png)
>
> **What it says:** in program assistants, a model makes "multiple sampled outputs for a single input prompt", and the user picks one. From here on, the paper assumes that "a request generates multiple sequences". With paged memory, this lets "the KV cache of the prompt to be shared".
>
> **What it means:**
>
> Note the change of words. A **request** is now what the user sent. A **sequence** is one prompt-plus-answer inside it. One request can hold four sequences, all with the same prompt and four different answers.
>
> The prompt is computed once, in the prompt phase, so its keys and values exist once. The question is whether the system copies them four times or lets four sequences point at one copy.

> [!PAPER] · Figure 8 · page 7
> [![Figure 8 of the paper (left of the crop): samples A1 and A2 each have logical blocks 0 and 1. Both map logical block 0 to physical block 7, holding Four score and seven. A1's logical block 1 points to a new physical block 3, a copy-on-write copy holding years ago our fathers; A2's points to physical block 1, holding years ago our mothers. A label says Ref count 2 to 1](/img/papers/vllm/p4-figure8.png)](/img/papers/vllm/p4-figure8.png)
>
> **What it says:** "Parallel sampling example." Two samples, A1 and A2, share the prompt "Four score and seven years ago our". A1 writes "fathers" and A2 writes "mothers". A red arrow marks "Copy-on-write", and a label says "Ref count: 2 → 1".
>
> **What it means:**
>
> Read the middle column first. It is GPU memory, nine physical blocks of four slots. The prompt sits in physical block 7 ("Four score and seven") and physical block 1 ("years ago our" plus one free slot). On both sides, each sample has its own block table with two logical blocks. Both tables point at blocks 7 and 1.
>
> Then the samples write their first new token. A2 writes "mothers" into the free slot of block 1. A1 cannot, because block 1 is shared. So block 1 is first copied to block 3, and A1 writes "fathers" there. After the copy, block 1 has one owner left, so its count drops from 2 to 1.

> [!NOTE] Where this idea comes from: fork() in Unix
> The two tools of the next boxes, reference counts and copy-on-write, are older than GPUs. They come from operating systems. Since the first versions of Unix, around 1970, a running program can call `fork()`. The call makes a second program, the child, that starts as an exact copy of the first, the parent.
>
> The early systems really copied all the parent's memory at every fork. That was slow and mostly wasted, because a child often replaces itself with another program at once. Later Unix systems stopped copying. Parent and child get page tables that point at the same physical pages, and each page counts how many page tables point at it.
>
> The pages are marked read-only. When either program writes to one, the processor stops it, and the operating system copies that one page and lets the write go on. Pages that nobody writes stay shared forever. vLLM does exactly this, with blocks of tokens in the place of pages of bytes. The paper says so itself: "similar to the copy-on-write technique in OS virtual memory (e.g., when forking a process)".

> [!PAPER] · §4.4 · page 7
> [![The paragraph that explains Figure 8, with reserve space for one copy, mapped to the same physical blocks, reference count and are both 2 highlighted](/img/papers/vllm/p4-refcount.png)](/img/papers/vllm/p4-refcount.png)
>
> **What it says:** both outputs share the prompt, so vLLM will "reserve space for one copy" of it. The logical blocks of both sequences are "mapped to the same physical blocks". Each physical block gets a "reference count"; for blocks 7 and 1 the counts "are both 2".
>
> **What it means:**
>
> A reference count is a tally on each physical block: how many block tables point here right now. It answers one question. Is anyone else using this block?
>
> The count makes two things safe. A block goes back to the free list only when its count reaches 0, so nobody loses data that another sequence still reads. And a sequence may write into a block only when the count is 1, so nobody changes data that another sequence still reads.

> [!DEFINITION] Reference count
> A number kept for each shared object: how many users point at it. Each new user adds 1; each user that leaves takes 1 away. At 0 the object can be thrown away. Operating systems use it for shared memory pages and open files.

> [!PAPER] · §4.4 · page 7
> [![The rest of the paragraph, with copy-on-write mechanism at the block granularity, is greater than 1, allocates a new physical block, decreases the reference count to 1 and directly writes highlighted](/img/papers/vllm/p4-cow.png)](/img/papers/vllm/p4-cow.png)
>
> **What it says:** vLLM uses a "copy-on-write mechanism at the block granularity". When A1 wants to write into physical block 1, its count "is greater than 1", so vLLM "allocates a new physical block" (block 3), copies block 1 into it and "decreases the reference count to 1". When A2 then writes to block 1, the count is already 1, so A2 "directly writes" there.
>
> **What it means:**
>
> Think of two students who share one printed worksheet. While they only read it, one copy is enough. The moment one of them wants to write an answer, that student photocopies the page and writes on the copy. The other keeps the original. Only the page with the writing is copied, not the whole workbook.
>
> Here the "page" is one block of four tokens. Copy-on-write costs one block copy per sample, at the moment the samples split. The blocks that are full stay shared for the whole life of the request, because nobody ever writes into a full block again.

> [!DEFINITION] Copy-on-write
> Share data until someone writes to it; copy only the piece that is written, and only at that moment. The [Background page](background-the-kv-cache-and-paging.md) shows how an operating system does this when a program forks.

The picture shows the whole of Figure 8 as two moments. Follow the reference count on each block. In the first moment, both block tables say "logical 0 is block 7, logical 1 is block 1", so both blocks have a count of 2. In the second moment, A1 has written "fathers". Its table now says "logical 1 is block 3", a fresh copy, and the count of block 1 has fallen to 1.

{{FIG:p4_cow|Copy-on-write in two moments. Left: A1 and A2 map the same two physical blocks, each with a reference count of 2. Right: A1 wanted to write into the shared block 1, so it got a copy (block 3) and the count of block 1 fell to 1. A2 then writes into block 1 directly. Block 7 is full and stays shared.}}

Block 7 never changes. It is full, so no sample ever writes into it again, and it stays shared until both samples finish. Only the one block where the two answers start to differ is copied.

We build this ourselves. Our block manager keeps a pool of physical blocks, a block table and a length per sequence, and a reference count per block. It has three calls, and §5.2 of the paper will name the same three. `fork` makes a new sequence that points at its parent's blocks. `append` stores one new token, with copy-on-write inside it. `free` drops a sequence:

```python
    def fork(self, parent, child):
        """fork: a new sequence that maps every block of its parent. Nothing is copied; reference counts go up."""
        self.table[child], self.length[child] = list(self.table[parent]), self.length[parent]
        for b in self.table[child]:
            self.ref[b] += 1

    def append(self, seq, kv):
        """append: store the keys and values of one new token. kv: (layers, 2, heads, head size)."""
        t, pos = self.table[seq], self.length[seq]
        if pos % self.B == 0:                               # the last block is full (or there is none): take a new one
            t.append(self._allocate())
        elif self.ref[t[-1]] > 1:                           # the last block is shared: copy-on-write
            new = self._allocate()
            self.pool[new].copy_(self.pool[t[-1]])          # copy the one block, then write into the copy
            self.ref[t[-1]] -= 1
            t[-1] = new
            self.cow_copies += 1
        self.pool[t[-1], :, :, :, pos % self.B] = kv
        self.length[seq] = pos + 1

    def free(self, seq):
        """free: drop a sequence. A block goes back to the free list when no sequence maps it any more."""
        for b in self.table.pop(seq):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free_blocks.append(b)
        del self.length[seq]
```

The `append` call has three cases. If the last block is full, it takes a fresh block. If the last block is shared, it copies it first. Otherwise it writes in place. The pool is a real tensor that holds the keys and values of OPT-125M, for all 12 layers.

To run the model on this cache, a helper called `extend` reads each sequence's blocks in table order. It hands them to the model as the past keys and values, and stores the new ones with `append`. Parallel sampling is then short. We run the prompt phase once, fork four samples, and drop the prompt's own sequence:

```python
    mgr = BlockManager(512, B)
    mgr.new('prompt')
    logits = extend(mgr, ['prompt'], [prompt])
    seqs = [f's{i}' for i in range(n)]
    for s in seqs:
        mgr.fork('prompt', s)
    mgr.free('prompt')
```

After that, each step draws one token per sample and calls `extend` for all four. We use the paper's prompt, "Four score and seven years ago", and a block size of 4, as in Figure 8. Method 2 is the library: `model.generate` with `num_return_sequences=4` and the same random seed. It keeps one full copy of the cache per sample. If our shared blocks or our copies were wrong, the tokens would differ.

```text
== 1. Parallel sampling (Figure 8): 4 samples share the blocks of one prompt, block size 4
prompt of 7 tokens (with the start token </s>) in 2 blocks: </s> Four score and | seven years ago
after fork: every sample maps blocks [370, 26]; reference counts {370: 4, 26: 4}
after the first new token: block tables [[370, 153], [370, 116], [370, 366], [370, 26]]; copy-on-write copies: 3
  sample 1: ', as the Los Angeles Dodgers amassed a 54-55 record, the question was: Will the Dodgers compete for the National'
  sample 2: ' this song crossed your mind...I just jokingly said (again) it had the power to be the best instrumental your'
  sample 3: ', Dave wanted the quintessential "I am her Unicorns" phase, but he had to think ahead.\nLast'
  sample 4: ", Bellus had a massive eye tear on one of Justin Matthews' passing screens. Being in the 80s, the"
Method 1 (paged, shared prompt) and Method 2 (model.generate, same seed): same 96 tokens: True
max difference of the logits against one plain pass per sample: 2.2e-05
at the end: 29 physical blocks with sharing, 32 without; copy-on-write copies in total: 3
```

{{FIG:p4_fig8|Our own Figure 8, from the run above. Right after fork, all four block tables point at the same two physical blocks. After the first new token, three samples have their own copy of the second block, and the fourth writes into the original.}}

This is Figure 8 with four samples instead of two. With the start token, the prompt has 7 tokens: 4 in physical block 370 and 3 in block 26, with one free slot. After fork, both blocks have a reference count of 4.

Then all four samples write their first token. The first three find block 26 shared, so each gets a copy: blocks 153, 116 and 366. The fourth finds a count of 1 and writes into block 26 directly. That is 3 copies in all, exactly the paper's rule. Block 370 was full, so it stays shared to the end.

The check passes. All 96 tokens are the same as the library's, and the logits match one plain pass per sample to $$2.2 \times 10^{-5}$$. (The samples are random text from a small model; only the agreement matters.)

> [!PAPER] · §4.4 · page 7
> [![The summary of parallel sampling, with with the exception of the final logical block and long input prompts highlighted](/img/papers/vllm/p4-sampling-sum.png)](/img/papers/vllm/p4-sampling-sum.png)
>
> **What it says:** vLLM shares most of the prompt's KV cache across the samples, "with the exception of the final logical block", which copy-on-write handles. The saving is largest "especially for long input prompts".
>
> **What it means:**
>
> Only the last, partly filled block of the prompt is ever copied. Every full block of the prompt is stored once, however many samples there are. So the saving grows with the length of the prompt.
>
> Our run shows the other side. The prompt has only 7 tokens, so sharing saves just 3 blocks: 29 instead of 32 at the end. The four answers are 24 tokens each and cannot be shared. A prompt of 2,000 tokens with short answers would save almost everything.

### Beam search {§4.4}

Parallel sampling wants variety. Beam search wants the opposite: the single answer that the model finds most likely as a whole. A translator wants one good translation, not five random ones.

Greedy decoding does not give that answer. It picks the best next word and never looks back. Suppose the best first word has probability 0.5, but all its continuations are weak, at 0.2 each. A second-best first word of 0.4, followed by a word of 0.9, gives a better pair: 0.4 × 0.9 = 0.36 against 0.5 × 0.2 = 0.10. Greedy decoding never finds it, because it threw that first word away.

Beam search keeps a few candidates alive, say 4. They are called **beams**, like the beam of a torch: a narrow light that searches part of a large dark space. Each step grows every beam by one token and keeps the 4 best results. Weak beams die, and a strong beam can have several children.

> [!NOTE] Where this idea comes from
> Beam search is far older than language models. It comes from speech recognition systems of the 1970s, which had to find the most likely sentence among very many possible ones. Machine translation then used it for decades. When neural networks took over translation, beam search stayed, and today's LLM servers still offer it.

> [!PAPER] · §4.4 · pages 7 to 8
> [![The paragraph that introduces beam search, with Beam search and the most probable output sequence highlighted](/img/papers/vllm/p4-beam-1.png)](/img/papers/vllm/p4-beam-1.png)
> [![Its continuation, with beam width, number of top candidates, retained at every step and is the vocabulary size highlighted](/img/papers/vllm/p4-beam-2.png)](/img/papers/vllm/p4-beam-2.png)
>
> **What it says:** beam search finds "the most probable output sequence" without trying every possible answer. Its "beam width" $$k$$ is the "number of top candidates retained at every step". Each step extends every candidate by every token and keeps the top $$k$$ of the $$k \cdot |V|$$ candidates, where $$|V|$$ "is the vocabulary size".
>
> **What it means:**
>
> The example above, 0.36 against 0.10, showed why greedy decoding can miss the best answer. Trying every answer is impossible too: with 50,000 tokens, two steps already give 2.5 billion pairs.
>
> Beam search sits in between. It keeps $$k$$ partial answers, the **beams**. At each step it tries every next token on every beam and keeps the $$k$$ best results. A beam can die, and a good beam can have several children.

The score of a beam is the sum of the log-probabilities of its tokens:

$$
\text{score}(y_1, \ldots, y_t) = \sum_{i=1}^{t} \log P(y_i \mid x, y_1, \ldots, y_{i-1})
$$

Read it from left to right. The tokens $$y_1$$ to $$y_t$$ are the answer so far, and $$x$$ is the prompt. Each term is the log of the probability that the model gives token $$i$$, given the prompt and the earlier tokens. Adding logs is the same as multiplying probabilities, which is Eq. 1 from Part 2. Logs are used because a product of many small numbers gets too small for a computer.

A tiny example with $$k = 2$$ and three tokens a, b, c. The two beams are "a" with score −0.5 and "b" with score −1.0. The model gives "a" the next-token log-probabilities −0.2, −2.0 and −3.0, and "b" gives −0.1, −0.3 and −4.0. The six candidates score −0.7, −2.5, −3.5, −1.1, −1.3 and −5.0. The top two are "aa" (−0.7) and "ba" (−1.1).

Both beams survive here, each with one child. If "ab" had scored above "ba", beam "a" would have both children and beam "b" would die.

{{FIG:p4_beamtree|The tiny example as a tree. Two beams, "a" and "b", each try all three tokens. Of the six candidates, only the two best scores survive: "aa" at −0.7 and "ba" at −1.1. The other four are dropped.}}

Now look at the tree with memory in mind. Both survivors start with the prompt, so they share its cache. "aa" also reuses the cache of "a", its parent. In a real search, beams share long common starts and split late. That is the sharing Figure 9 shows.

> [!DEFINITION] Log-probability
> The natural logarithm of a probability. It is 0 for a certain event and more negative for a rarer one: $$\log 0.5 \approx -0.69$$, $$\log 0.01 \approx -4.6$$. Adding log-probabilities gives the log of their product.

> [!PAPER] · Figure 9 · page 7
> [![Figure 9 of the paper (right of the crop): four beam candidates as chains of blocks. All start at block 0; candidates 0 to 2 share blocks 1 and 3; candidate 3 has blocks 2, 4 and 8. Blocks 2, 4, 5 and 8 are crossed out. After a dotted line, new blocks 9 to 12 extend the survivors](/img/papers/vllm/p4-figure9.png)](/img/papers/vllm/p4-figure9.png)
>
> **What it says:** "Beam search example." Four beam candidates, drawn as chains of blocks, before and after one iteration (the dotted line).
>
> **What it means:**
>
> Each row is one beam, and each box is one physical block. Before the dotted line, all four beams share block 0, the prompt. Beams 0 to 2 also share blocks 1 and 3, then split into blocks 5, 6 and 7. Beam 3 left the others early and has its own blocks 2, 4 and 8.
>
> After the dotted line, all four new beams descend from old beams 1 and 2. Old beams 0 and 3 die. Their blocks 5, 2, 4 and 8 have no owner left, so they are crossed out and freed. The survivors get new blocks 9 to 12.

> [!PAPER] · §4.4 · page 7
> [![The paragraph that explains Figure 9, with sharing patterns dynamically, process tree in the OS created by compound forks, whose reference counts reach and all candidates share blocks 0, 1, 3 highlighted](/img/papers/vllm/p4-beam-share.png)](/img/papers/vllm/p4-beam-share.png)
>
> **What it says:** in beam search the "sharing patterns dynamically" change, "similar to the process tree in the OS created by compound forks". When beams 0 and 3 drop out, vLLM frees all blocks "whose reference counts reach" 0 (blocks 2, 4, 5 and 8). Afterwards "all candidates share blocks 0, 1, 3".
>
> **What it means:**
>
> In parallel sampling the sharing is fixed: the prompt is shared and the answers never are. In beam search, which beams share which blocks changes at every step. A beam is born as a copy of its parent: a fork. It dies when it falls out of the top $$k$$: a free.
>
> The beams form a family tree, like processes that fork other processes. Reference counts do all the bookkeeping. A block lives exactly as long as some living beam still descends through it.

Our beam search uses the same three calls. At each step, every new beam is a `fork` of its parent, and then all old beams are freed. Then the new token of each beam is stored with `append`:

```python
        top = torch.topk((scores[:, None] + logp).flatten(), k)
        parents, words = (top.indices // V).tolist(), (top.indices % V).tolist()
        new = []
        for i, (p, w) in enumerate(zip(parents, words)):
            nid += 1
            c = f'b{nid}'
            mgr.fork(beams[p], c)
            toks[c] = toks[beams[p]] + [w]
            new.append(c)
            if p != i:                                      # a contiguous cache would copy the parent's whole cache
                contiguous_copy += mgr.length[beams[p]]
        for b in beams:
            mgr.free(b)
```

The first line is the score equation: the beam scores plus the log-probabilities of every next token, all $$k \cdot |V|$$ of them, then the top $$k$$. Fork comes before free, so a block that a child still needs never reaches a count of 0. Method 2 is `model.generate` with `num_beams=4`:

```text
== 2. Beam search (Figure 9): 4 beams fork, append and free blocks at every step, block size 4
  beam 1: ', I was in my early 20s.\nI was in my early 20s.\nI was in my early'
  beam 2: ', I was in my early 20s.\n\nI was in my early 20s.\n\nI was in'
  beam 3: ', I was in my early 20s.\nI was in my early 20s.\nI was in my late'
  beam 4: ', I was in my early 20s.\nI was in my early 20s.\nI was in my mid'
Method 1 (paged beam search) and Method 2 (model.generate, num_beams=4): same 96 tokens: True
final block tables: [[782, 921, 628, 975, 950, 147, 92, 201], [782, 921, 628, 975, 904, 521, 772, 907], [782, 921, 628, 975, 950, 147, 92, 201], [782, 921, 628, 975, 950, 147, 92, 201]]
blocks mapped by all 4 beams: [782, 921, 628, 975]; physical blocks in use 14, without sharing 32
copied during the search: 22 blocks by copy-on-write = 88 token slots; a contiguous cache that keeps beam i in slot i copies 1029 token slots
```

{{FIG:p4_fig9|Our own Figure 9. The block tables of the four beams at the end. All four map the same first four blocks; beams 1, 3 and 4 map the same eight. Only beam 2 has its own last four blocks.}}

All 96 tokens match the library. The block tables tell the story of Figure 9 in numbers. All four beams share their first four blocks. Beams 1, 3 and 4 share all eight. They differ only in the token just chosen ("early", "late", "mid"), and its keys and values are not stored yet. Beam 2 split off earlier, at the double line break.

So 14 physical blocks hold what four separate caches would store in 32 blocks.

{{FIG:p4_blocks|Physical blocks in use at each step of our two runs. Without sharing, four beams need four full caches (top). With sharing, beam search stays far lower. The four samples share only their short prompt, so they need almost as much as four separate caches.}}

> [!PAPER] · §4.4 · page 7
> [![The paragraph on memory copies in earlier systems, with frequent memory copies, copy a large portion and only copying one block of data highlighted](/img/papers/vllm/p4-beam-copy.png)](/img/papers/vllm/p4-beam-copy.png)
>
> **What it says:** earlier systems needed "frequent memory copies" of the KV cache across beams. In Figure 9, beam 3 "would need to copy a large portion" of beam 2's cache. In vLLM, copy-on-write applies only when a new token lands in an old shared block, which means "only copying one block of data".
>
> **What it means:**
>
> With one contiguous chunk per beam, a beam that takes over another beam's place must copy that beam's whole cache into its own chunk. That happens again and again, because beams die and split at almost every step.
>
> With blocks, taking over a cache means copying a block table, a short list of numbers. The only real data copy is the last, half-full block, when the new beam first writes into it.

We counted both in the run above. A contiguous cache that keeps beam $$i$$ in slot $$i$$ would copy 1,029 token slots over the 24 steps. Our block manager copied 22 blocks of 4, which is 88 token slots. (The library itself copies even more: it reorders the whole cache of all beams at every step.)

How much memory does sharing save in a larger setting? We run both methods again with vLLM's default block size of 16. The prompt is the 41-token translation prefix of Figure 10 below, and every answer has 48 tokens. The saving is the paper's measure: blocks saved by sharing, divided by the blocks needed without sharing.

```text
== 3. Memory saved by sharing blocks, block size 16 (the vLLM default), prompt of the Figure 10 prefix
prompt of 41 tokens, 48 new tokens; saving = blocks saved by sharing / blocks without sharing, averaged over the steps
parallel sampling, 2 / 4 / 6 samples: 23.1% / 34.7% / 38.5%
beam search, width 2 / 4 / 6:          31.9% / 43.8% / 48.2%
```

{{FIG:p4_saving|Memory saved by sharing in our run, block size 16. More samples or wider beams share more. Beam search saves more than parallel sampling because beams also share parts of their answers.}}

Both grow with the number of sequences, and beam search saves more than sampling. The paper reports 6.1% to 9.8% for sampling and 37.6% to 55.2% for beam search, on OPT-13B with Alpaca requests (§6.3, Part 5). Our sampling numbers are higher because our prompt is long next to our answers. As the paper said, the saving depends on the prompt length.

### Shared prefix {§4.4}

The third case crosses the line between users. Most LLM products send the model more than the user typed. A support chatbot for a bank might put a hidden text in front of every question: "You are the assistant of Example Bank. Answer politely. Never give investment advice. Here are three sample answers..." This hidden text is the **system prompt**. It can be hundreds or thousands of tokens long, and it is the same for every user.

So a thousand users who each ask a short question send a thousand prompts that start with the same long text. Without sharing, the server computes the keys and values of that text a thousand times and stores a thousand copies. With paged memory, the text can be computed once, kept in a few blocks, and mapped by every request.

> [!DEFINITION] System prompt
> Instructions that a service puts in front of every user's message: the task, the rules, and often a few worked examples. The user usually does not see it. Many requests share it word for word.

> [!PAPER] · §4.4 · pages 7 to 8
> [![The paragraph on shared prefixes, with instructions and example inputs and outputs and system prompt highlighted](/img/papers/vllm/p4-prefix-1.png)](/img/papers/vllm/p4-prefix-1.png)
> [![Figure 10 of the paper: two sequences whose prompts start with the same shared prefix, Translate English to French with three examples, followed by different task inputs, cheese and I love you, and their outputs fromage and Je t'amie](/img/papers/vllm/p4-figure10.png)](/img/papers/vllm/p4-figure10.png)
>
> **What it says:** users often give a long description of the task with "instructions and example inputs and outputs", also known as a "system prompt". Figure 10 shows two requests whose prompts start with the same translation prefix and end with different task inputs.
>
> **What it means:**
>
> A translation service might put the same instruction and three worked examples in front of every word a user sends. Here every request starts with "Translate English to French:" and three example pairs. Only the last line differs: "cheese =>" in one request, "I love you =>" in the other.
>
> This is sharing across different requests from different users, not inside one request. The shared part is often much longer than the user's own text.

> [!PAPER] · §4.4 · page 8
> [![The paragraph on storing the prefix, with store the KV cache of the prefix in advance, reserving a set of physical blocks, shared library across processes, last block marked copy-on-write and task input highlighted](/img/papers/vllm/p4-prefix-2.png)](/img/papers/vllm/p4-prefix-2.png)
>
> **What it says:** the service can "store the KV cache of the prefix in advance", by "reserving a set of physical blocks" for known prefixes, like an OS handles a "shared library across processes". A request maps its logical blocks to these cached blocks, with the "last block marked copy-on-write". The prompt phase then runs only on the user's "task input".
>
> **What it means:**
>
> On a computer, many programs use the same library code. The operating system keeps one copy of it in memory for all of them. vLLM does the same with a popular prefix. Its keys and values are computed once and kept in a few blocks that no request owns.
>
> A new request starts as a fork of the prefix. Its block table already points at the cached blocks, so the prompt phase skips those tokens and only reads them. This saves memory and computation: the prefix is neither stored twice nor computed twice.

We store the 41-token prefix once, in blocks of 4. Each request is a fork of it. Our `extend` then runs the prompt phase on the task tokens only. We check the answer against `model.generate` on the full prompt, computed from scratch:

```text
== 4. Shared prefix (Figure 10): the prefix is computed once and two requests map its blocks
  'cheese =>': 3 task tokens computed instead of 44; maps 10 prefix blocks, owns 3; greedy output ' pouvoir\npouv'; same as model.generate on the full prompt: True
  'I love you =>': 4 task tokens computed instead of 45; maps 10 prefix blocks, owns 4; greedy output ' le le le le le le le le'; same as model.generate on the full prompt: True
prefix of 41 tokens in 11 blocks; its last block is partly full, so each request copied it once: 2 copy-on-write copies
```

The prompt phase computed 3 tokens instead of 44 for the first request, and 4 instead of 45 for the second. Both answers equal the library's answers on the full prompt. The prefix fills 10 blocks and 1 token of an eleventh. That last block is the one "marked copy-on-write": each request copied it once, before writing its own tokens.

{{FIG:p4_prefix|A shared prefix before and with vLLM. Top: each request computes and stores the same 41-token prefix again. Bottom: the prefix is computed once and stored in 11 blocks; each request maps those blocks and computes only its own task tokens, 3 and 4 of them.}}

OPT-125M is far too small to translate. It writes nonsense French, so do not read the outputs as translations. The point is that the shared cache gives exactly the same tokens as computing everything again.

### Mixed decoding methods {§4.4}

> [!PAPER] · §4.4 · page 8
> [![The paragraph on mixed decoding methods, with simultaneous processing of requests with different decoding preferences, common mapping layer, list of physical block IDs and broadens the batching opportunities highlighted](/img/papers/vllm/p4-mixed.png)](/img/papers/vllm/p4-mixed.png)
>
> **What it says:** vLLM allows the "simultaneous processing of requests with different decoding preferences". It hides all sharing behind a "common mapping layer" from logical to physical blocks. The model and its kernel only see a "list of physical block IDs" for each sequence. This "broadens the batching opportunities".
>
> **What it means:**
>
> In one step, a batch can hold a greedy request, a request with four samples and a beam search with six beams. They share memory in completely different patterns. The attention kernel does not care. For each sequence, it gets a block table and reads those blocks, whether one sequence or five point at them.
>
> Our `extend` works the same way. It never asks why two tables hold the same number; it just reads the blocks. All the decoding logic lives above it, in fork, append and free. That split is what lets one batch mix any decoding methods.

## When memory runs out {§4.5}

Sharing makes memory go further, but it cannot make memory endless. Answers grow one token at a time, and nobody knows how long they will get. Sooner or later, a running request needs a new block and none is free.

A tiny example shows the problem. A GPU has 6 free blocks of 4 tokens. Three requests run, and each holds 2 blocks. Now each writes its next token, and the last block of each is full. Each needs one more block, and there are none. Somebody must give up memory, or nobody can take a step.

> [!NOTE] Where this idea comes from: scheduling in operating systems
> A computer runs more programs than it has processors. The part of the operating system that picks which program runs next is the **scheduler**. Its oldest rule is **first-come-first-served**: run the jobs in the order they arrived, like a queue at a counter. Modern schedulers can also **preempt**: stop a running program in the middle, give its turn to another, and resume it later.
>
> Memory has the same problem. Sometimes programs want more memory than the machine has. The operating system then moves some of their memory out to disk, and brings it back when it is needed. This is **swapping**. The early Unix systems swapped whole programs at once; later systems with paging swap single pages.
>
> vLLM borrows all three ideas. Its scheduler serves requests first-come-first-served and preempts a request when blocks run out. It can swap that request's blocks out to CPU memory, which plays the part of the disk.

> [!PAPER] · §4.5 · page 8
> [![The opening of §4.5, with first-come-first-serve (FCFS), earliest arrived requests are served first and latest requests are preempted first highlighted](/img/papers/vllm/p4-fcfs.png)](/img/papers/vllm/p4-fcfs.png)
>
> **What it says:** when traffic is more than the system can serve, vLLM uses a "first-come-first-serve (FCFS)" policy, for fairness and against starvation. The "earliest arrived requests are served first" and the "latest requests are preempted first".
>
> **What it means:**
>
> This is the rule of a queue at a bakery: whoever came first is served first. When the shop gets too full and someone must step outside for a while, it is the person who came in last.
>
> The rule is simple on purpose. A server could try to be clever, for example by serving short requests first. But it does not know which requests are short, and a long request could then wait forever. That is **starvation**. FCFS cannot starve anyone, because every request only waits for the ones that came before it.

> [!DEFINITION] Preemption
> Stopping a running job before it is done, to give its resources to someone else, and resuming it later. Here the resource is the request's KV cache blocks.

> [!PAPER] · §4.5 · page 8
> [![The paragraph on running out of blocks, with input prompts for an LLM can vary significantly in length, not known a priori and run out of the GPU's physical blocks highlighted](/img/papers/vllm/p4-evict-1.png)](/img/papers/vllm/p4-evict-1.png)
> [![Its continuation in the right column, with Which blocks should it evict?, How to recover evicted blocks, all-or-nothing eviction policy and gang-scheduled as a sequence group highlighted](/img/papers/vllm/p4-evict-2.png)](/img/papers/vllm/p4-evict-2.png)
>
> **What it says:** prompt lengths vary a lot and output lengths are "not known a priori", so vLLM can "run out of the GPU's physical blocks". It must answer two questions: "Which blocks should it evict?" and "How to recover evicted blocks". All blocks of a sequence are used together, so vLLM uses an "all-or-nothing eviction policy". The sequences of one request are "gang-scheduled as a sequence group".
>
> **What it means:**
>
> An operating system evicts single pages. It tries to guess which page nobody will need for the longest time. vLLM does not need to guess. At every step, attention reads every block of a running sequence, from the first to the last. Evicting half of a sequence's blocks saves nothing, because the sequence cannot take a single step without them. So either all its blocks stay, or all go.
>
> The same holds one level up. Four samples or six beams of one request share blocks. Evicting one beam would not free the shared blocks, and the beams must stay together to compare scores. So a whole request, its **sequence group**, is stopped and resumed as one unit.

> [!DEFINITION] Eviction
> Removing data from fast memory to make room. The data is either thrown away or moved to slower memory, and must be brought back or rebuilt before it is used again.

Go back to the tiny example: 6 blocks, three requests with 2 blocks each, and all three need a third block. Request 3 arrived last, so it is preempted and gives up both of its blocks. Now 2 blocks are free. Requests 1 and 2 take one each and go on. Request 3 waits until one of the others finishes.

Why not take only one block from request 3, so that it could keep going? Because the next step of request 3 reads all of its blocks. With one block missing, it cannot run at all, and the block it kept is wasted space. This is the old Unix choice too: a program that cannot run gives up all its memory at once.

> [!PAPER] · §4.5 · page 8
> [![The paragraph on swapping, with copy evicted blocks to the CPU memory, CPU block allocator, stops accepting new requests until all preempted sequences and never exceeds the number of total physical highlighted](/img/papers/vllm/p4-swap.png)](/img/papers/vllm/p4-swap.png)
>
> **What it says:** the first way to recover is swapping. vLLM will "copy evicted blocks to the CPU memory", managed by a "CPU block allocator". After a preemption, it "stops accepting new requests until all preempted sequences" are done. So the blocks in CPU memory "never exceeds the number of total physical" blocks on the GPU.
>
> **What it means:**
>
> The CPU of a server has its own memory, often several times larger than the GPU's. Swapping parks a preempted request's blocks there. When room appears on the GPU, the blocks are copied back and the request goes on as if nothing happened.
>
> The rule about new requests is what keeps this bounded. While any request is parked, no newcomer may start, so the parked requests always come back first. The total parked can never exceed what fit on the GPU in the first place. A server can size its CPU swap space in advance.

> [!PAPER] · §4.5 · page 8
> [![The paragraph on recomputation, with significantly lower than the original latency, one prompt phase iteration, bandwidth between CPU RAM and GPU memory and computation power of the GPU highlighted](/img/papers/vllm/p4-recompute.png)](/img/papers/vllm/p4-recompute.png)
>
> **What it says:** the second way is recomputation: throw the cache away and compute it again on resume. This can be "significantly lower than the original latency", because the prompt and the written tokens form a new prompt, whose cache comes out in "one prompt phase iteration". Which way is faster depends on the "bandwidth between CPU RAM and GPU memory" and the "computation power of the GPU".
>
> **What it means:**
>
> Recomputation sounds wasteful, but there is a trick. The request first built its cache slowly, one token per step, because each token had to be written before the next. On resume, all those tokens are already known. They can go through the model in one pass, as one long prompt, using the GPU at full speed (Part 2).
>
> An operating system can never do this. A page of a program's memory cannot be rebuilt from nothing. A KV cache can, because it is a pure function of the tokens. That makes recomputation a choice that only an LLM server has.

> [!DEFINITION] Bandwidth
> How many bytes a connection moves per second. Between CPU memory and a server GPU, data moves over **PCIe**, the standard slot that connects a GPU card to the rest of the computer. PCIe is much slower than the memory inside the GPU.

> [!DEFINITION] Latency
> The time one request waits from the moment it is sent until its answer is complete. A preempted request has a longer latency, because it spends some steps waiting.

The picture puts both ways of recovering on one timeline. Three requests run. At step 5 no block is free, and request 3, the last to arrive, is preempted. At step 8 request 1 ends, and its blocks become free. With swapping, request 3's blocks waited in CPU memory and are copied back. With recomputation, they were thrown away and one prompt pass builds them again.

Either way, request 3 then runs on from the token where it stopped. Its answer is the same; only its latency is longer.

{{FIG:p4_preempt|One preemption, recovered in two ways (a sketch, not a measurement). Top: with swapping, the preempted request's blocks wait in CPU memory and are copied back when room appears. Bottom: with recomputation, the blocks are thrown away and rebuilt in one prompt pass. In both, the request that arrived last is the one preempted.}}

We build the scheduler as a simulation in steps, with the memory rules above and no model. The requests are 200 real Alpaca prompts and answers, counted in OPT tokens; one arrives per step. The GPU has room for 64 blocks of 16 tokens. Here is the part that preempts:

```python
        # 2. every running request writes one token; it may need one more block. If none is free, preempt the
        #    request that arrived last (all of its blocks), as often as needed.
        for r in list(running):
            if r not in running:
                continue
            n = REQ[r][0] + gen[r]                          # tokens stored so far
            if n % BS == 0:                                 # the last block is full
                while free == 0:
                    victim = running.pop()                   # the latest arrival
                    hold = blocks(REQ[victim][0] + gen[victim])
                    free += hold
                    st['preemptions'] += 1
                    st['preempt_steps'].append(t)
                    if policy == 'swap':
                        st['swap_out'] += hold; cpu += hold; cpu_max = max(cpu_max, cpu)
                    preempted.append(victim); preempted.sort()
                    if victim == r:
                        break
                if r not in running:
                    continue
                free -= 1
```

The running list is sorted by arrival, so `running.pop()` takes the request that came last. It loses all its blocks at once. At the start of each step, the scheduler brings back preempted requests, oldest first. It admits no new request while any is still out. We run the same traffic with swapping and with recomputation. In the output, look for two things: whether every request still gets all its tokens, and what each method pays for its preemptions.

```text
== 5. First-come-first-served scheduling with preemption (simulation in steps, Alpaca request lengths)
200 requests, one per step; prompt mean 19.9 tokens, output mean 53.8; GPU memory: 64 blocks of 16 tokens
swap     : 873 steps, mean batch 12.3, 356 preemptions; every request got all its tokens: True
recompute: 873 steps, mean batch 12.3, 356 preemptions; every request got all its tokens: True
swap     : 705 blocks out, 705 in; at most 24 blocks in CPU memory (never more than the 64 GPU blocks: True)
           bytes moved, OPT-13B: 18.48 GB; OPT-125M: 831.7 MB
recompute: 9,115 tokens computed again, in 356 prompt passes
```

{{FIG:p4_sched|Our scheduler over 873 steps. The free blocks (green) soon fall to zero and stay near zero while the queue is long; each tick under the axis is one preemption. Only when the queue empties near the end do blocks come free.}}

The traffic is far more than 64 blocks can serve, on purpose. Memory is full almost all the time, and the scheduler preempts 356 times. Still, every request gets every one of its tokens. In a simulation in steps, both methods give the same timeline. What differs is the cost of each preemption.

Swapping moved 705 blocks out and 705 back in. For OPT-13B, with 800 KB of cache per token, that is 18.48 GB over the CPU-GPU connection. At no time did CPU memory hold more than 24 blocks, well under the bound of 64 that the paper promises. Recomputation moved nothing, but computed 9,115 tokens again, in 356 prompt passes.

Which is cheaper? A simple model of the two costs helps:

$$
t_{\text{swap}} \approx \frac{\text{bytes}}{\text{bandwidth}} + n_{\text{blocks}} \cdot t_{\text{copy}}, \qquad t_{\text{recompute}} \approx t_{\text{prompt pass}}(n_{\text{tokens}})
$$

Read the left part first. Swapping pays for the bytes it moves, divided by the bandwidth. It also pays a fixed start-up cost $$t_{\text{copy}}$$ for every separate copy, and there is one copy per block. The right part says that recomputation costs one prompt pass over all the tokens, whatever the block size.

A tiny example: say the connection moves 32 GB/s. Then 64 MB takes 2 ms. As 4 blocks with a fixed cost of 0.1 ms each, it takes 2.4 ms. As 4,096 tiny blocks, the fixed costs alone add 409.6 ms. Small blocks make swapping slow.

Now the real thing. We take a request with a 14-token prompt and 200 written tokens and do both on OPT-125M. First we check that recomputation gives back the same cache. Then we time the three ways to get that cache on the laptop GPU:

```text
== 6. Swapping against recomputation on the real model (OPT-125M)
a request with a 14-token prompt and 200 written tokens: 213 tokens of KV cache
recomputed in one pass against built one token at a time: max difference 1.8e-05; next token after recomputation the same: True
laptop GPU (mps), FP16, fastest of several runs; 213 tokens = 7.85 MB of KV cache
  building it the first time, one pass per token:  1342.8 ms
  recomputation, one prompt pass:                     9.5 ms
  swap out + swap in, blocks of 16 tokens:            5.9 ms; round trip exact: True
  recomputation took 142 times less time than building the cache one token at a time
swap time against block size (the same 213 tokens; one copy per block):
  block size      1      2      4      8     16     32     64    128    256
  out ms      40.14  21.50   9.65   4.81   2.77   1.48   1.02   0.52   0.33
  in ms       46.75  19.37   9.18   4.99   2.84   1.49   0.92   0.52   0.33
  recompute    9.45 ms at every block size (it does not read the blocks)
  block size 1 means 213 copies each way: 0.19 ms per copy out; block size 256 means 1 copy
```

{{FIG:p4_swap|Time to swap the same 213-token cache out and back in, against the block size, on our laptop. One copy per block makes small blocks expensive. Recomputation does not read the blocks, so its cost is the same at every block size.}}

Recomputation is exact: the cache from one pass equals the cache built token by token to $$1.8 \times 10^{-5}$$, and the next token is the same. Swapping is exact too, bit for bit.

The paper's claim about speed holds by a wide margin. Building the cache one token at a time took 1,342.8 ms. Recomputing it in one pass took 9.5 ms, 142 times less.

The block-size table shows the start-up cost. At block size 1, each direction makes 213 small copies, about 0.19 ms each, and the swap takes over 40 ms. At block size 16 it takes under 3 ms each way, and at 256 one copy does it. Recomputation sits at 9.45 ms whatever the block size. So recomputation wins for small blocks and swapping for large ones, the pattern the paper measures in §7.3 (Part 5).

Our laptop's numbers are not a server's. An Apple laptop's CPU and GPU share one memory, so "swapping" here copies within the same chips. A server moves the blocks over PCIe, which is slower. A server GPU also computes much faster. These timings come from a laptop shared with other jobs, so treat them as shapes.

## Many GPUs {§4.6}

Every number of a model's weights must sit in GPU memory while the model runs. In 16-bit numbers, each parameter takes 2 bytes. So a model of 13 billion parameters needs 26 GB for its weights alone, and the largest A100 holds 80 GB. The paper's Table 1 lists its three models: 26 GB, 132 GB and 346 GB of weights.

{{FIG:p4_gpus|Why one model needs several GPUs. The weights of the paper's three models (Table 1) against the memory of one A100. Only OPT-13B fits on one card; the paper runs OPT-66B on 4 GPUs and OPT-175B on 8.}}

The weights are not all. The KV cache also needs room, and that room decides how many requests can run at once. On 4 A100s with 160 GB together, OPT-66B leaves 21 GB for the cache (Table 1). So the question is not only "does the model fit?" but "how do several GPUs work as one?".

> [!NOTE] Where this idea comes from: Megatron-LM
> In 2019 a team at NVIDIA published Megatron-LM. They wanted to train language models too big for one GPU, and showed a simple way to cut each layer into slices, one per GPU. With it they trained a model of 8.3 billion parameters. Their method is called **tensor parallelism**, and most LLM servers use it today, vLLM among them.
>
> Think of two cooks and one long recipe. One way to share the work: the first cook does the first half of the steps, then passes the dish to the second. That is **pipeline parallelism**. The other way: at every step, both cooks work at once, each on half of the ingredients, and they join the results. That is tensor parallelism, and Megatron-LM does it at every layer.

> [!DEFINITION] Shard
> One piece of something that is cut up and spread over several machines. A **model shard** is the part of the model's weights that one GPU holds.

> [!PAPER] · §4.6 · pages 8 to 9
> [![The opening of §4.6, with exceeding the capacity of a single GPU, Megatron-LM style tensor model parallelism and SPMD (Single Program Multiple Data) highlighted](/img/papers/vllm/p4-dist-1.png)](/img/papers/vllm/p4-dist-1.png)
> [![Its continuation on page 9, with all-reduce operation, split on the attention head dimension and a subset of attention heads highlighted](/img/papers/vllm/p4-dist-2.png)](/img/papers/vllm/p4-dist-2.png)
>
> **What it says:** many models are "exceeding the capacity of a single GPU". vLLM supports "Megatron-LM style tensor model parallelism", with an "SPMD (Single Program Multiple Data)" schedule. The linear layers are split, and the GPUs combine their results with an "all-reduce operation". Attention is "split on the attention head dimension": each GPU computes "a subset of attention heads".
>
> **What it means:**
>
> As the chart above showed, OPT-66B cannot fit on one A100. So the paper serves it on 4 GPUs and OPT-175B on 8 (Table 1, Part 5). The question is how to cut one model into pieces that work together at every step.
>
> Tensor parallelism cuts every layer, not the list of layers. Every GPU runs the same program on every token, which is what SPMD means. Each GPU holds a slice of every weight matrix. For attention the cut is natural: a layer with 12 heads on 2 GPUs gives 6 heads to each. Heads do not talk to each other until the very end, so each GPU can do its own heads alone.

> [!DEFINITION] Tensor parallelism
> Splitting each weight matrix of a model across several GPUs, so that every GPU does part of every layer. The other common split, pipeline parallelism, gives each GPU a different group of whole layers.

> [!DEFINITION] All-reduce
> A group operation: each GPU has a tensor of the same shape, and afterwards every GPU holds the sum of all of them. A library such as NCCL, NVIDIA's library for sending data between GPUs, does it over the links between the cards.

> [!DEFINITION] Attention head
> One of several small attention units that work side by side in one layer. OPT-125M has 12 heads per layer, each with vectors of 64 numbers. Each head makes its own query, key and value and its own output; the layer then joins the 12 outputs.

> [!DEFINITION] SPMD
> Single Program, Multiple Data. Every GPU runs the same code at the same time, each on its own share of the data. Here the "share" is a share of the weights: GPU 0 has heads 0 to 5, GPU 1 has heads 6 to 11.

Why is a sum enough? The last step of attention multiplies the outputs of all heads, side by side, by an output matrix $$W_o$$. Split the head outputs into GPU 0's part $$h^{(0)}$$ and GPU 1's part $$h^{(1)}$$, and split the rows of $$W_o$$ the same way:

$$
o = \begin{bmatrix} h^{(0)} & h^{(1)} \end{bmatrix} \begin{bmatrix} W_o^{(0)} \\ W_o^{(1)} \end{bmatrix} = h^{(0)} W_o^{(0)} + h^{(1)} W_o^{(1)}
$$

Read it from left to right. The row vector of all head outputs times the full matrix equals two smaller products added together. GPU 0 can compute $$h^{(0)} W_o^{(0)}$$ with its own heads and its own rows of $$W_o$$, and GPU 1 the other term. One all-reduce adds them. This is just how matrix multiplication works, with no approximation.

A tiny example: $$h = (1 \mid 2)$$ and $$W_o$$ with rows 3 and 4. The full product is $$1 \cdot 3 + 2 \cdot 4 = 11$$. GPU 0 computes $$1 \cdot 3 = 3$$, GPU 1 computes $$2 \cdot 4 = 8$$, and the all-reduce gives $$3 + 8 = 11$$.

{{FIG:p4_heads|Attention split by heads over two GPUs. GPU 0 holds heads 0 to 5 and GPU 1 heads 6 to 11. Each multiplies its head outputs by its own rows of the output matrix and gets a partial output. The all-reduce adds the partial outputs, 3 + 8 = 11 in the tiny example, and gives the sum to both GPUs.}}

The heads themselves need no talk between GPUs. Each head reads only its own queries, keys and values. Only the last multiplication mixes the heads, and that is the one place where the GPUs must add their results.

Here is one GPU's work, for the first attention layer of OPT-125M. It takes the query, key and value rows of its own heads and runs attention for those heads. Then it multiplies by its own columns of the output matrix. The second block is the all-reduce, here just a sum of the two results. If the split is right, the sum must equal the layer of the library:

```python
def attention_shard(x, heads):
    """One GPU's work: the query, key and value rows of its own heads (column split), attention for those heads,
    and its columns of the output matrix (row split). The result is a partial sum, without the output bias."""
    rows = slice(heads.start * D, heads.stop * D)
    q = (x @ attn.q_proj.weight[rows].T + attn.q_proj.bias[rows]) * attn.scaling
    k = x @ attn.k_proj.weight[rows].T + attn.k_proj.bias[rows]
    v = x @ attn.v_proj.weight[rows].T + attn.v_proj.bias[rows]
    h = len(heads)
    q, k, v = (t.view(-1, h, D).transpose(0, 1) for t in (q, k, v))          # (heads, tokens, head size)
    s = q @ k.transpose(1, 2) + torch.full((Tn, Tn), -float('inf')).triu(1)   # causal mask
    o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(-1, h * D)
    return o @ attn.out_proj.weight[:, rows].T, k                              # partial output, this GPU's keys
```

```python
parts = [attention_shard(x, range(0, 6)), attention_shard(x, range(6, 12))]
allreduce = parts[0][0] + parts[1][0] + attn.out_proj.bias                    # all-reduce = sum over the GPUs
```

(PyTorch stores a weight matrix transposed. So the "rows of $$W_o$$" in the equation are columns of `out_proj.weight` here.) The bias is added once, after the sum, or it would be counted twice. Method 2 is the unsplit attention layer of the library, which we capture while the model runs. We also split the feed-forward layer the same way and compare the whole layer:

```text
== 7. Tensor parallelism (§4.6): one OPT-125M layer split over 2 simulated GPUs, 6 heads each
11 tokens; GPU 0 has heads 0-5, GPU 1 has heads 6-11; each computes a partial output of shape (11, 768)
all-reduce (sum of the 2 partial outputs) against the unsplit attention of the library: max difference 4.2e-07
the whole layer (attention + feed-forward, 2 all-reduces) against the library layer: max difference 2.3e-05
KV cache per GPU: keys of shape (6, 11, 64) (6 heads, all 11 positions); the two halves together equal the full keys: True
per GPU and per token: 18,432 bytes of KV cache instead of 36,864 (FP16); the same block table serves both
data sent by each all-reduce: 11 x 768 numbers; 2 all-reduces per layer, 24 per pass
```

{{FIG:p4_tp|One attention layer split over two GPUs, as in our run. Both GPUs get the same input. Each computes six heads and a partial output; the all-reduce adds the two partial outputs into the layer's output.}}

The two partial outputs add up to the library's attention output, to $$4.2 \times 10^{-7}$$. The whole layer, with a second all-reduce after the feed-forward part, matches to $$2.3 \times 10^{-5}$$. Each layer needs two all-reduces, so OPT-125M on 2 GPUs would need 24 per pass. Our two "GPUs" are two function calls on one laptop; a real setup sends these sums over fast links between the cards.

Look at the keys. Each GPU keeps keys for its 6 heads, but for all 11 positions. Each token's cache is cut in half, 18,432 bytes per GPU instead of 36,864, but every position is on every GPU. That fact is the point of the next box.

> [!PAPER] · §4.6 · page 9
> [![The paragraph on the single KV cache manager, with the same set of input tokens, single KV cache manager, share the manager and only stores a portion of the KV cache highlighted](/img/papers/vllm/p4-one-manager.png)](/img/papers/vllm/p4-one-manager.png)
>
> **What it says:** with model parallelism, each shard still processes "the same set of input tokens", so it needs the KV cache for the same positions. vLLM therefore has a "single KV cache manager" in the central scheduler, and all GPU workers "share the manager". Each worker uses the same physical block IDs but "only stores a portion of the KV cache": the part for its own heads.
>
> **What it means:**
>
> Picture a library whose books are so wide that each book is cut down the middle. The left halves stand on shelf A, the right halves on shelf B. A single catalogue says "your book is at position 26", and both halves stand at position 26 on their shelves.
>
> Block 26 on GPU 0 holds heads 0 to 5 of some 16 tokens. Block 26 on GPU 1 holds heads 6 to 11 of the same 16 tokens. So one block table is right for every GPU. The scheduler can do all the memory bookkeeping in one place, and no GPU ever needs to ask another where a block is.

> [!PAPER] · §4.6 · page 9
> [![The paragraph on one step, with broadcasts this control message, all-reduce communication primitive and do not need to synchronize on memory management highlighted](/img/papers/vllm/p4-control.png)](/img/papers/vllm/p4-control.png)
>
> **What it says:** at each step, the scheduler prepares the token IDs and block tables of the batch and "broadcasts this control message" to the workers. The workers run the model and combine results with the "all-reduce communication primitive", without the scheduler. They send back the sampled tokens. So workers "do not need to synchronize on memory management".
>
> **What it means:**
>
> One step has a simple rhythm. The scheduler decides who runs, does fork, append and free, and sends one message with the token IDs and block tables. The GPUs then compute, talking only to each other through all-reduces. At the end they report the new tokens.
>
> All memory decisions happen before the step starts. During the step, the GPUs only read and write the blocks they were told about. That keeps the fast path, the model on the GPUs, free of any memory logic.

## The implementation {§5}

> [!PAPER] · §5 · page 9
> [![The opening of §5, with 8.5K lines of Python and 2K lines of C++/CUDA code and custom CUDA kernels highlighted](/img/papers/vllm/p4-impl.png)](/img/papers/vllm/p4-impl.png)
>
> **What it says:** vLLM has a FastAPI frontend that extends the OpenAI API, and an engine of "8.5K lines of Python and 2K lines of C++/CUDA code". The scheduler and block manager are Python; "custom CUDA kernels" handle key operations such as PagedAttention.
>
> **What it means:**
>
> The split follows the cost. The scheduler and the block manager run once per step and deal with lists and counters. Python is fast enough for that, and easy to change. The work that touches every number of the cache, thousands of times per step, runs on the GPU in hand-written kernels.
>
> Our block manager is the same idea in one short Python class. What we cannot write in Python is the fast GPU side. The next two boxes show why it needs special kernels.

> [!DEFINITION] Kernel and kernel launch
> A **kernel** is a function that runs on the GPU, on many threads at once. A **kernel launch** is the CPU telling the GPU to start one. Each launch has a small fixed cost, so a thousand tiny kernels can cost more than the work they do.

> [!NOTE] Where this idea comes from: GPU kernels and kernel fusion
> A GPU has thousands of small cores. It does not run a whole program by itself. The CPU runs the program and hands the GPU one job at a time: "multiply these two matrices", "copy these bytes". Each job is a kernel. Since 2007, NVIDIA's **CUDA** has let programmers write their own kernels in a form of C++. A **thread** is one of the many copies of the kernel that run at once, each on its own piece of the data.
>
> Every launch has a fixed cost, a few microseconds on a server GPU, before any real work starts. Think of a trip to the shop. Buying ten things in ten trips costs ten journeys; buying them in one trip costs one. **Kernel fusion** is the one trip. Several small steps are joined into one kernel, so the fixed cost is paid once. The data also stays on the chip between the steps.
>
> Fast inference libraries are built on fused kernels. One of them is NVIDIA's FasterTransformer, which the paper uses as a baseline in its experiments. The vLLM authors took FasterTransformer's attention kernel and changed it to read keys and values through a block table.

> [!PAPER] · §5.1 · page 9
> [![The first part of §5.1, with Fused reshape and block write, into a single kernel., Fusing block read and attention and assign a GPU warp to read each block highlighted](/img/papers/vllm/p4-kern-1.png)](/img/papers/vllm/p4-kern-1.png)
>
> **What it says:** (1) "Fused reshape and block write": the new KV cache is split into blocks, reshaped and saved at the places that the block table gives, all "into a single kernel". (2) "Fusing block read and attention": the attention kernel reads the cache through the block table on the fly, and they "assign a GPU warp to read each block".
>
> **What it means:**
>
> At every step, every layer writes one new key and value per sequence into some slot of some block. Done naively, that is one small copy per token per layer. The fused kernel gets a list of target slots and writes them all in one launch.
>
> The second kernel is PagedAttention itself. It reads keys and values block by block, following the table, inside the attention computation. A **warp** is a group of 32 GPU threads that run in lockstep. Giving each block to one warp makes those threads read neighbouring addresses at the same time, which is the fast way to read GPU memory.

> [!PAPER] · §5.1 · page 9
> [![The end of §5.1, with Fused block copy, numerous invocations of small data movements and into a single kernel launch highlighted](/img/papers/vllm/p4-kern-2.png)](/img/papers/vllm/p4-kern-2.png)
>
> **What it says:** (3) "Fused block copy": the copies that copy-on-write asks for may touch scattered blocks. With one cudaMemcpyAsync call per block, that means "numerous invocations of small data movements". vLLM batches all of them "into a single kernel launch".
>
> **What it means:**
>
> In one step of a large beam search, dozens of blocks may need copy-on-write. Each copy is small, a few kilobytes per layer. Paid one launch at a time, the fixed costs add up, as with our small-block swaps. One kernel that gets the full list of (source, target) pairs does all of them in one go.

We try both fused writes in PyTorch on the laptop GPU. For the block write, 256 sequences each store one new key. The slot of a token is its physical block times 16 plus its offset inside the block:

```python
slots = (tables[torch.arange(256), lens // B9] * B9 + lens % B9).to(GPU)       # where each new token goes
```

```python
def write_loop():
    for i in range(256):
        cache[slots[i]] = new[i]


def write_fused():
    cache.index_copy_(0, slots, new)
```

The copy-on-write test copies 128 scattered blocks, once with one copy per block and once with one batched copy. In the output, compare the two times on each line, and check that both ways give the same result:

```text
== 8. Kernels of §5.1 in PyTorch, on the laptop GPU (FP16, one layer's key cache: blocks x 16 x 12 x 64)
block write of 256 new tokens: one copy per token 62.15 ms (256 launches), one scatter 0.155 ms (1 launch); same cache: True; 401x
copy-on-write of 128 scattered blocks: one copy per block 1.10 ms, one batched copy 0.405 ms; same result: True; 2.7x
```

{{FIG:p4_fusion|Kernel fusion in our run. Writing 256 new tokens with one launch each pays the fixed launch cost 256 times; one fused launch pays it once and does the same work. The bars are a sketch; the two times are from our laptop GPU.}}

Both versions give the same cache, bit for bit. The single scatter is 401 times faster than 256 separate writes. Batching the 128 block copies makes them 2.7 times faster. A PyTorch call on an Apple GPU has much more overhead than a CUDA kernel launch, so the ratios on an A100 are different. The direction is the same: many small operations lose to one large one.

> [!PAPER] · §5.2 · page 9
> [![§5.2, with fork, append, and free, creates a new sequence from an existing one, appends a new token and deletes the sequence highlighted](/img/papers/vllm/p4-api.png)](/img/papers/vllm/p4-api.png)
>
> **What it says:** vLLM builds all its decoding algorithms from three methods: "fork, append, and free". Fork "creates a new sequence from an existing one", append "appends a new token", and free "deletes the sequence". Parallel sampling forks the samples, appends tokens each step and frees finished ones; beam search and prefix sharing use the same calls.
>
> **What it means:**
>
> This is the block manager we built at the start of this part. Parallel sampling was one fork per sample, then appends. Beam search was fork the winners, free the losers, append. A shared prefix was a fork of a stored sequence. Copy-on-write and reference counts hide inside the three calls, and the decoding code never touches a block.
>
> The authors add that future decoding methods can be built from the same three calls. The memory side would not change at all: a new method only decides when to fork, append and free.

## What our small run cannot show

Our model was OPT-125M on an Apple laptop, not OPT-13B on an A100. The block tables, reference counts, copies and the agreement with `model.generate` are exact and do not depend on the hardware. The memory savings depend on our prompt and answer lengths, which are not the paper's workload.

The scheduler is a simulation in steps: it counts blocks and tokens but does not time anything. Our timings come from a laptop GPU that shares its memory with the CPU and ran other jobs at the same time. We took the fastest of several runs, but treat the numbers as shapes, not server numbers. Our two "GPUs" for tensor parallelism are two function calls; we did not measure the cost of a real all-reduce. Part 5 reads the paper's own measurements.

> [!TAKEAWAYS] Key takeaways
> - **Sharing blocks needs only two tools:** a reference count per physical block, and copy-on-write when a sequence writes into a shared block.
> - **Parallel sampling** stores the prompt once. In our run, four samples copied only the last, half-full prompt block (3 copies) and matched `model.generate` on all 96 tokens.
> - **Beam search** shares more, and the sharing changes at every step. Our 4 beams used 14 physical blocks instead of 32 and copied 88 token slots instead of 1,029.
> - **A shared prefix** is computed once and mapped by every request. Our two requests ran the prompt phase on 3 and 4 tokens instead of 44 and 45.
> - **The kernel only sees block tables.** That is why one batch can mix greedy decoding, sampling and beam search.
> - **When blocks run out,** vLLM preempts the latest request, all of its blocks at once, and serves requests first-come-first-served.
> - **Swapping** copies the blocks to CPU memory; **recomputation** runs prompt plus answer as one prompt. Recomputing our 213-token cache took 142 times less time than building it token by token.
> - **Tensor parallelism** gives each GPU a share of the attention heads. One all-reduce adds the partial outputs, which matched the unsplit layer to $$4.2 \times 10^{-7}$$.
> - **One block table serves every GPU,** because every GPU stores the same token positions for different heads.
> - **fork, append and free** are the whole interface between decoding algorithms and memory.

**Next, in Part 5:** §6 to §10. The experiments: throughput against latency, the measured savings of sharing, swapping against recomputation on an A100, and the limits.

<details>
<summary>Run it yourself</summary>

The script behind this part is [`code/papers/vllm/vllm_part4.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vllm/vllm_part4.py). It builds the block manager and runs parallel sampling, beam search and a shared prefix on it, each checked against `model.generate`. It measures the memory saved by sharing. It then simulates the scheduler with swapping and recomputation, and times both on the real model.

Last, it splits one layer over two simulated GPUs and tries two of the fused writes of §5.1. Every number is saved in `results/part4.json`. On an Apple M-series laptop it takes under a minute.

```bash
pip install torch transformers datasets
python vllm_part4.py   # downloads facebook/opt-125m (about 250 MB) and tatsu-lab/alpaca
```

<figure><img src="/img/papers/vllm/part4-run.png" alt="Terminal output of vllm_part4.py: parallel sampling and beam search on the block manager, the memory saved by sharing, the shared prefix, the scheduler simulation, swapping against recomputation, tensor parallelism and the fused kernels" loading="lazy" /><figcaption>The full output of vllm_part4.py, as it ran for this article.</figcaption></figure>

</details>

## The paper and the code

[vLLM on arXiv](https://arxiv.org/abs/2309.06180) · [vLLM on GitHub](https://github.com/vllm-project/vllm) · [our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vllm)
