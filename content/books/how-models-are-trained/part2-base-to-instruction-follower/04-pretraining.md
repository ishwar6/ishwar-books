---
description: "Pretraining from the inside: how web text is collected, filtered and deduplicated, how it is tokenized, what training costs (C = 6ND), what scaling laws say about model size and data, the learning-rate schedule, batch size and mixed precision, and a real 12M-parameter GPT pretrained on a laptop GPU."
---
# Chapter 4 · Pretraining: learning from the whole internet

> **Goal:** by the end of this chapter you can explain every stage that turns raw web pages into a base model: how the text is collected, cleaned, deduplicated and scored; how it is cut into tokens; how much compute a run needs and how to calculate it; how scaling laws decide the size of the model and the amount of data; why the learning rate rises and then falls, why batches are large, and why training runs in 16-bit numbers. You will also pretrain a small GPT yourself on a laptop GPU, watch its loss fall and its stories improve, and then probe a real base model to see what pretraining alone puts inside it.

---

## 4.1 What pretraining is

Chapter 1 described a language model as a machine that reads some tokens and gives a probability to every possible next token. It showed the loss used to train it (cross-entropy), how that loss turns into perplexity, and a tiny training loop. It also drew the whole pipeline, from a **base model** to an assistant. This chapter zooms into the first and by far the most expensive box of that pipeline: **pretraining**.

> [!DEFINITION] Pretraining
> The first training stage of a language model. A network with random weights is trained to predict the next token on a very large amount of general text (today, trillions of tokens). Nothing in the data says "this is a question" or "this is a good answer"; the only signal is the next token.

> [!DEFINITION] Base model
> The model that comes out of pretraining, before any fine-tuning. It continues text well, it knows many facts, and it can often do a task if the prompt shows a few examples, but it does not reliably follow instructions or stop when it has answered.

The objective is the same next-token cross-entropy you met in Chapter 1. What makes pretraining a field of its own is everything around that objective, and the scale at which it runs. A modern run has to answer questions like these:

- Where do ten trillion tokens of text come from, and how do you remove the spam, the duplicates and the junk?
- How is text cut into tokens?
- How large should the model be, and how many tokens should it see, for a given budget of compute?
- How fast should it learn at each moment, how many examples should be in each step, and with how many bits per number?
- What does the model actually know when it is done?

<figure class="fig"><svg viewBox="0 0 760 305" role="img" aria-label="The pretraining recipe: raw text is cleaned and mixed, tokenized, and fed to a Transformer through a training loop, producing a base model."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="380.0" y="24.0" text-anchor="middle">Pretraining in one picture: five ingredients and one loop</text><rect class="box-1" x="20.0" y="50.0" width="125.0" height="70.0" rx="10"/><text class="t-note" x="82.5" y="72.0" text-anchor="middle">raw text</text><text class="t-tick" x="82.5" y="91.0" text-anchor="middle">web, books, code</text><text class="t-tick" x="82.5" y="108.0" text-anchor="middle">trillions of words</text><rect class="box-1" x="170.0" y="50.0" width="125.0" height="70.0" rx="10"/><text class="t-note" x="232.5" y="72.0" text-anchor="middle">clean + mix</text><text class="t-tick" x="232.5" y="91.0" text-anchor="middle">filter, dedup,</text><text class="t-tick" x="232.5" y="108.0" text-anchor="middle">choose proportions</text><rect class="box-3" x="320.0" y="50.0" width="125.0" height="70.0" rx="10"/><text class="t-note" x="382.5" y="72.0" text-anchor="middle">tokenize</text><text class="t-tick" x="382.5" y="91.0" text-anchor="middle">BPE: text to ids</text><text class="t-tick" x="382.5" y="108.0" text-anchor="middle">one long stream</text><rect class="box-2" x="470.0" y="50.0" width="125.0" height="70.0" rx="10"/><text class="t-note" x="532.5" y="72.0" text-anchor="middle">Transformer</text><text class="t-tick" x="532.5" y="91.0" text-anchor="middle">random weights</text><text class="t-tick" x="532.5" y="108.0" text-anchor="middle">N parameters</text><rect class="box-4" x="620.0" y="50.0" width="125.0" height="70.0" rx="10"/><text class="t-note" x="682.5" y="72.0" text-anchor="middle">base model</text><text class="t-tick" x="682.5" y="91.0" text-anchor="middle">predicts the</text><text class="t-tick" x="682.5" y="108.0" text-anchor="middle">next token well</text><line class="edge" x1="145.0" y1="85.0" x2="190.0" y2="85.0" marker-end="url(#ah)"/><line class="edge" x1="295.0" y1="85.0" x2="340.0" y2="85.0" marker-end="url(#ah)"/><line class="edge" x1="445.0" y1="85.0" x2="490.0" y2="85.0" marker-end="url(#ah)"/><line class="edge" x1="595.0" y1="85.0" x2="640.0" y2="85.0" marker-end="url(#ah)"/><rect class="box-ghost" x="200.0" y="150.0" width="400.0" height="112.0" rx="12"/><text class="t-note" x="400.0" y="172.0" text-anchor="middle">the training loop, repeated for S steps</text><text class="t-tick" x="225.0" y="196.0" text-anchor="start">1. take a batch of B token windows</text><text class="t-tick" x="225.0" y="213.0" text-anchor="start">2. predict every next token, cross-entropy loss</text><text class="t-tick" x="225.0" y="230.0" text-anchor="start">3. backpropagate: gradient of the loss</text><text class="t-tick" x="225.0" y="247.0" text-anchor="start">4. AdamW update, learning rate from the schedule</text><line class="edge" x1="532.0" y1="120.0" x2="532.0" y2="150.0" marker-end="url(#ah)"/><line class="edge" x1="470.0" y1="150.0" x2="470.0" y2="120.0" marker-end="url(#ah)"/><text class="t-muted" x="400.0" y="290.0" text-anchor="middle">Chapter 1 showed the loss and one small training loop. This chapter is about everything around it, at scale.</text></svg><figcaption>The pretraining recipe. Raw text is cleaned and mixed, cut into tokens and fed, a batch at a time, to a Transformer with random weights. The loop (predict, compute the loss, backpropagate, update) runs for many steps, and what comes out is a base model.</figcaption></figure>

We will take these in order. Every number in this chapter comes either from a paper (shown in a PAPER box, with the excerpt) or from a script in `code/training/` that you can run yourself. The hands-on run in Section 4.11 pretrains a 12-million-parameter GPT on 12 million tokens in about eight minutes on a laptop's Apple GPU. It is a toy, tens of billions of times smaller in compute than a frontier model, but it has every piece a real run has.

## 4.2 The data: where trillions of words come from

### 4.2.1 Common Crawl: a copy of the public web

Almost every large pretraining corpus starts from **Common Crawl**, a nonprofit that has crawled the public web since 2008 and gives the results away. Each crawl ("snapshot") holds a few billion pages and is published in three file formats.

> [!DEFINITION] Common Crawl
> A free, public archive of web pages, collected by a crawler several times a year. Each snapshot is named like `CC-MAIN-2024-33` (year 2024, week 33) and is split into tens of thousands of files.

> [!DEFINITION] WARC, WAT and WET files
> Common Crawl's three formats. **WARC** holds the raw HTTP responses (full HTML). **WAT** holds metadata (links, headers). **WET** holds plain text that Common Crawl has already extracted from the HTML. The text is ready to use, but the extraction keeps menus, footers and other boilerplate.

To see what this raw material looks like, the script `ch4_data.py` downloads the first 12 MB of one WET file of the August 2024 crawl and reads every page in it. The first numbers are a good reality check:

```text
raw documents: 3,964   characters: 23,632,096
top languages (Common Crawl header, CLD2): eng 1405, zho 727, rus 296, jpn 246, deu 152, fra 148, spa 144, pol 94
```

Only 35% of the pages are English. Here are the first few English pages the filters threw away, as the script recorded them (the text is collapsed to one line):

```text
word count       | http://16beavergroup.org/articles/...     | One moment, please... Please wait while your request is being verified...
word count       | http://200pluswinegrapes.com/synonym/...  | One moment, please... Please wait while your request is being verified...
mean word length | http://4.ff1213.com/sitemap.xml           | https://www.algomachristian.net/parents.html 2024-04-11T12:25:57+00:00 https://...
```

Bot-check pages, sitemaps full of URLs and timestamps, cookie banners, shop listings, navigation menus: a large share of the raw web is text that no one would want a model to learn to write. The rest of this section is about removing it.

Not every corpus is pure web text. **The Pile** (Gao et al., 2020) was an early open corpus that deliberately mixed 22 sources: a filtered slice of Common Crawl, but also academic papers (arXiv, PubMed), books, GitHub code, Stack Exchange, Wikipedia, legal text and more. Its treemap shows how much each source contributed after the authors chose how many times to repeat each one.

> [!PAPER] Gao et al. (2020), The Pile · Figure 1 · page 2
> [![The treemap of the Pile: large blocks for Pile-CC, ArXiv, OpenWebText2, StackExchange and Wikipedia, coloured by category: academic, internet, prose, dialogue, misc](/img/training/ch4-pile-treemap.png)](/img/training/ch4-pile-treemap.png)
>
> **Context:** the first figure of the paper, before any detail about how each source was processed.
>
> **What it says:** the Pile is a *mixture*. Web text (green) is the largest category, but academic writing (blue), books (orange) and code and other sources (grey) take a large share by design. The sizes are "effective": a high-quality source such as Wikipedia was repeated several times, so it counts for more than its raw size.
>
> **Why it matters:** the idea that the *mixture* of sources is a design choice, with high-quality sources sampled more often, is now standard. Section 4.13 returns to it with Llama 3.

### 4.2.2 Filtering with rules

The first line of defence is a set of cheap **heuristic filters**: rules computed from simple statistics of a page, each with a threshold.

> [!DEFINITION] Heuristic filter
> A hand-written rule that keeps or removes a document based on a simple statistic, such as its word count, the share of lines that end with punctuation, or the share of repeated lines. Rules are cheap enough to run on billions of pages.

Three rule sets appear again and again.

- **Gopher rules** (Rae et al., 2021, the MassiveText corpus): keep a page only if it has between 50 and 100,000 words, a mean word length between 3 and 10 characters, at most one `#` or `...` per ten words, at least 80% of words containing a letter, and at least two of the stop words *the, be, to, of, and, that, have, with*. A second group of **repetition** rules removes pages where many lines, paragraphs or n-grams are repeated.
- **C4 rules** (Raffel et al., 2019, the T5 paper): drop lines that mention `javascript` or cookie and privacy policies, drop pages with "lorem ipsum" or a curly bracket `{` (a strong sign of code or broken markup), drop pages with too few sentences, and keep only lines that end with terminal punctuation.
- **FineWeb rules** (Penedo et al., 2024), found by comparing statistics of good and bad data, described in the box below.

The FineWeb paper is the most complete public description of a modern web pipeline, and it is worth reading in full. Its "base filtering" step combines a URL blocklist, a language classifier and the Gopher rules:

> [!PAPER] Penedo et al. (2024), The FineWeb Datasets · Section 3.3 · page 5
> [![A paragraph of the FineWeb paper describing base filtering, with highlights on URL filtering using a blocklist, keep only English text with a score >= 0.65, and quality and repetition filters from MassiveText](/img/training/ch4-fineweb-base-filtering.png)](/img/training/ch4-fineweb-base-filtering.png)
>
> **Context:** the first filtering step of FineWeb, applied to the text extracted from 96 Common Crawl snapshots.
>
> **What it says:** a blocklist removes adult sites by URL; a fastText language classifier keeps only pages that are English with a score of at least 0.65; and the MassiveText (Gopher) rules are applied with their original thresholds. What survives is "roughly 36 trillion tokens".
>
> **Why it matters:** these three cheap steps remove most of the raw web before anything expensive runs.

The FineWeb team then looked for new rules in a systematic way: they computed over 50 statistics on a high-quality and a low-quality version of the same crawl and picked thresholds where the low-quality data was over-represented. They also tested the C4 rules one by one and kept all of them except the terminal-punctuation rule, which helped the most but removed about 30% of all tokens. Three new rules survived their ablations:

> [!PAPER] Penedo et al. (2024), The FineWeb Datasets · Section 3.6 · page 8
> [![The FineWeb paper's list of three custom filters, with highlights on fraction of lines ending with punctuation, fraction of characters in duplicated lines, and fraction of lines shorter than 30 characters is >= 0.67](/img/training/ch4-fineweb-custom-filters.png)](/img/training/ch4-fineweb-custom-filters.png)
>
> **Context:** the end of the section on "developing additional heuristic filters", after the authors tested 16 candidate rules.
>
> **What it says:** remove a document if at most 12% of its lines end with punctuation, if at least 10% of its characters sit in duplicated lines, or if at least 67% of its lines are shorter than 30 characters. Together these remove about 22% of tokens and raise the benchmark score of a small test model by about 1%.
>
> **Why it matters:** every rule is tested by *training a model* on the filtered data and measuring it. Good data is defined by what it does to a model, not by how it looks.

`ch4_data.py` implements simplified versions of all three rule sets and runs them on the 3,964 pages, in the order FineWeb uses. The rules are short. Here are the Gopher quality rules (simplified from the script):

```python
STOP = {'the', 'be', 'to', 'of', 'and', 'that', 'have', 'with'}

def gopher_quality(t):
    words = t.split()
    n = len(words)
    if not 50 <= n <= 100_000: return 'word count'
    if not 3 <= sum(len(w) for w in words) / n <= 10: return 'mean word length'
    if (t.count('#') + t.count('...')) / n > 0.1: return 'symbol ratio'
    lines = [l for l in t.split('\n') if l.strip()]
    if sum(l.lstrip().startswith(('•', '-', '*')) for l in lines) / len(lines) > 0.9: return 'bullet lines'
    if sum(l.rstrip().endswith('...') for l in lines) / len(lines) > 0.3: return 'ellipsis lines'
    if sum(bool(re.search('[a-zA-Z]', w)) for w in words) / n < 0.8: return 'non-alphabetic words'
    if len(STOP & {w.lower() for w in words}) < 2: return 'stop words'
    return None
```

Each `if` is one rule. The function returns the *name* of the first rule that fires, or `None` if the page passes, so the script can count why pages were removed. `words = t.split()` splits on whitespace and `n` is the word count. The mean word length catches pages of very short tokens (tables of numbers) or very long ones (URLs, encoded blobs). The symbol ratio catches hashtag spam. The bullet and ellipsis rules catch pages that are only lists or only teasers. The alphabetic-word rule catches pages of numbers and symbols. The stop-word rule is a surprisingly strong test of "is this real English prose?", because any real English paragraph uses at least two of those eight words.

The FineWeb rules look the same:

```python
def fineweb_rules(t):
    lines = [l for l in t.split('\n') if l.strip()]
    if sum(l.rstrip().endswith(('.', '!', '?', '"', "'")) for l in lines) / len(lines) <= 0.12:
        return 'lines ending in punctuation <= 0.12'
    c = collections.Counter(lines)
    if sum(len(l) * v for l, v in c.items() if v > 1) / len(t) >= 0.1:
        return 'chars in duplicated lines >= 0.1'
    if sum(len(l) < 30 for l in lines) / len(lines) >= 0.67:
        return 'short lines >= 0.67'
    return None
```

The first rule computes the share of non-empty lines that end like a sentence. The second counts every line that appears more than once, weighted by its length and number of copies, as a share of all characters. The third is the share of lines shorter than 30 characters, which is high on menus and lists of links.

Running the whole pipeline gives this funnel:

<figure class="fig"><svg viewBox="0 0 760 310" role="img" aria-label="A funnel of document counts through language, quality, deduplication and classifier filters."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">3,964 real Common Crawl pages through a FineWeb-style pipeline (ch4_data.py)</text><text class="t-tick" x="230.0" y="64.0" text-anchor="end">raw pages (one WET file)</text><g class="mark"><title>raw WET records: 3964</title><rect class="s1" x="240" y="50" width="400.0" height="20" rx="3"/></g><text class="t-val" x="648.0" y="64.0" text-anchor="start">3,964  (100.0%)</text><text class="t-tick" x="230.0" y="98.0" text-anchor="end">English only</text><g class="mark"><title>English: 1405</title><rect class="s1" x="240" y="84" width="141.8" height="20" rx="3"/></g><text class="t-val" x="389.8" y="98.0" text-anchor="start">1,405  (35.4%)</text><text class="t-tick" x="230.0" y="132.0" text-anchor="end">+ Gopher quality/repetition</text><g class="mark"><title>Gopher rules: 587</title><rect class="s1" x="240" y="118" width="59.2" height="20" rx="3"/></g><text class="t-val" x="307.2" y="132.0" text-anchor="start">587  (14.8%)</text><text class="t-tick" x="230.0" y="166.0" text-anchor="end">+ C4 rules</text><g class="mark"><title>C4 rules: 459</title><rect class="s1" x="240" y="152" width="46.3" height="20" rx="3"/></g><text class="t-val" x="294.3" y="166.0" text-anchor="start">459  (11.6%)</text><text class="t-tick" x="230.0" y="200.0" text-anchor="end">+ FineWeb custom rules</text><g class="mark"><title>FineWeb rules: 295</title><rect class="s1" x="240" y="186" width="29.8" height="20" rx="3"/></g><text class="t-val" x="277.8" y="200.0" text-anchor="start">295  (7.4%)</text><text class="t-tick" x="230.0" y="234.0" text-anchor="end">MinHash dedup</text><g class="mark"><title>MinHash dedup: 291</title><rect class="s3" x="240" y="220" width="29.4" height="20" rx="3"/></g><text class="t-val" x="277.4" y="234.0" text-anchor="start">291  (7.3%)</text><text class="t-tick" x="230.0" y="268.0" text-anchor="end">FineWeb-Edu score &gt;= 3</text><g class="mark"><title>FineWeb-Edu score &gt;= 3: 11</title><rect class="s4" x="240" y="254" width="3.0" height="20" rx="3"/></g><text class="t-val" x="251.0" y="268.0" text-anchor="start">11  (0.3%)</text><text class="t-muted" x="20.0" y="298.0" text-anchor="start">Each bar is the number of pages still alive after that step. Most raw pages are not English; most English pages fail a quality rule.</text></svg><figcaption>Real Common Crawl pages through a FineWeb-style pipeline. Of 3,964 raw pages in our sample, 1,405 are English, 295 survive all the rules, 291 survive near-duplicate removal, and 11 would be kept by the FineWeb-Edu classifier at its threshold of 3. The last two steps are explained below.</figcaption></figure>

Only 7.4% of the raw pages, and 21% of the English pages, survive the rules. The reasons are spread across many rules:

```text
why documents were removed:
   not English                                     2559
   Gopher: duplicate lines                          340
   Gopher: word count                               229
   Gopher: non-alphabetic words                     184
   FineWeb: lines ending in punctuation <= 0.12     131
   C4: too few sentences                             85
   C4: curly bracket                                 40
   Gopher: stop words                                33
   Gopher: mean word length                          30
   FineWeb: chars in duplicated lines >= 0.1         21
   FineWeb: short lines >= 0.67                      12
   C4: lorem ipsum                                    3
```

The single most common problem is repetition: pages whose lines repeat (menus, product grids, "Add to cart" forty times). Next come pages that are too short (bot-check pages, error pages) and pages made mostly of numbers and symbols. Real FineWeb would differ in two ways: it extracts text from the raw HTML with a better tool (trafilatura) instead of using the WET text, and it removes adult sites with a URL blocklist, which we do not have.

> [!NOTE]
> These are simplified versions of the published rules, run on one small sample. The exact percentages will differ on other crawls. What carries over is the shape: most raw pages are not usable, and no single rule does most of the work.

### 4.2.3 Deduplication

The web repeats itself. The same news story is syndicated to hundreds of sites, the same template fills thousands of product pages, and the same legal footer appears on millions of pages. Training on duplicates wastes compute, and it does something worse: it teaches the model to **memorize**.

> [!DEFINITION] Deduplication
> Removing documents (or long passages) that are copies or near-copies of others. **Exact** deduplication compares hashes of the full text; **fuzzy** (near-duplicate) deduplication also catches pages that differ in a few words, such as the same article with a different header.

> [!PAPER] Lee et al. (2022), Deduplicating Training Data Makes Language Models Better · Abstract · page 1
> [![The abstract of Lee et al., with highlights on over 1% of the unprompted output, a single 61 word English sentence that is repeated over 60,000 times, and emit memorized text ten times less frequently](/img/training/ch4-dedup-abstract.png)](/img/training/ch4-dedup-abstract.png)
>
> **Context:** the abstract.
>
> **What it says:** in the C4 corpus a single 61-word sentence appears over 60,000 times. Models trained on such data copy their training data verbatim in over 1% of what they generate unprompted. After deduplication, models emit memorized text ten times less often and need fewer training steps for the same accuracy.
>
> **Why it matters:** deduplication is not just about saving compute. It changes what the model learns.

Comparing every pair of documents is impossible at web scale: a billion documents make about $$5 \times 10^{17}$$ pairs. The standard tool is **MinHash**, which gives each document a short "signature" so that similar documents get similar signatures, and then only compares documents whose signatures collide.

> [!DEFINITION] Shingles and Jaccard similarity
> A document's **shingles** are its overlapping n-grams, for example every run of 5 consecutive words. The **Jaccard similarity** of two documents is the number of shingles they share divided by the number of distinct shingles they have in total. It is 1 for identical documents and 0 for documents with nothing in common.

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

where:

- $$A$$ and $$B$$ are the sets of shingles of the two documents,
- $$|A \cap B|$$ is the number of shingles in both,
- $$|A \cup B|$$ is the number of distinct shingles in either.

**Worked example** (from `ch4_worked.py`, using 3-word shingles to keep it small). Take two sentences that differ in one word:

```text
A: the cat sat on the mat and looked out of the window at the rain
B: the cat sat on the mat and looked out of the door at the rain
A has 13 shingles, B has 13, shared 10, union 16: Jaccard = 0.625
```

Changing "window" to "door" breaks the three shingles that contain that word, so 10 of the 13 shingles are shared, and the union has $$13 + 13 - 10 = 16$$ distinct shingles. $$J = 10/16 = 0.625$$.

MinHash rests on one neat fact. Apply a random hash function $$h$$ to every shingle of a document and keep only the smallest value. For two documents,

$$P\left[\min_{s \in A} h(s) = \min_{s \in B} h(s)\right] = J(A, B)$$

where:

- $$h$$ is a hash function chosen at random, which behaves like a random ordering of all possible shingles,
- $$\min_{s \in A} h(s)$$ is the smallest hash value over the shingles of $$A$$,
- the probability is over the random choice of $$h$$.

The reason: under a random ordering, the first shingle of $$A \cup B$$ is equally likely to be any of its $$|A \cup B|$$ shingles, and the two minimums agree exactly when that first shingle lies in $$A \cap B$$. So if you compute many different hash minimums for each document (its **signature**), the share of positions where two signatures agree estimates their Jaccard similarity. With 64 hashes on the two sentences above, 38 of 64 positions agreed: an estimate of 0.594 against the true 0.625.

> [!DEFINITION] MinHash signature
> A short list of numbers, one per hash function, each being the smallest hash value over all of a document's shingles. Two documents agree at each position with probability equal to their Jaccard similarity.

Comparing signatures still means comparing pairs. The final trick is **banding** (a form of locality-sensitive hashing): cut the signature into bands, and put two documents in the same bucket if *all the numbers of any one band* are equal. Only documents that share a bucket are compared. FineWeb's settings:

> [!PAPER] Penedo et al. (2024), The FineWeb Datasets · Section 3.4 · page 6
> [![The FineWeb paper's description of its MinHash setup, with highlights on computed MinHashes using 112 hash functions in total, split into 14 buckets of 8 hashes each, and targeting documents that are at least 75% similar](/img/training/ch4-fineweb-minhash.png)](/img/training/ch4-fineweb-minhash.png)
>
> **Context:** the deduplication section, after the base filtering.
>
> **What it says:** each document's word 5-grams are hashed with 112 hash functions, split into 14 buckets of 8, "targeting documents that are at least 75% similar". Documents with the same 8 MinHashes in any bucket are treated as duplicates.
>
> **Why it matters:** the numbers 14 and 8 are chosen to put the cut-off near 75% similarity, as the next formula shows.

If two documents have Jaccard similarity $$J$$, one band of 8 matches with probability $$J^8$$, so the chance that at least one of the 14 bands matches is

$$P(\text{flagged}) = 1 - \left(1 - J^{8}\right)^{14}$$

where:

- $$J$$ is the true Jaccard similarity of the two documents,
- $$J^8$$ is the chance that all 8 hashes of one band agree,
- $$(1 - J^8)^{14}$$ is the chance that none of the 14 bands agrees.

**Worked example** (`ch4_data.py`). For $$J = 0.75$$: $$0.75^8 = 0.100$$, $$(1 - 0.100)^{14} = 0.228$$, so $$P = 0.772$$. For $$J = 0.6$$ it is only 0.211, and for $$J = 0.9$$ it is 1.000. The curve is a soft step around 0.75: near-duplicates are almost always caught, and documents that only share some phrases are almost never touched.

<figure class="fig"><svg viewBox="0 0 820 334" role="img" aria-label="The probability that MinHash with 14 buckets of 8 hashes flags a pair, as a function of their Jaccard similarity."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">MinHash with 112 hashes in 14 buckets of 8: a soft threshold near 75% similarity</text><line class="grid" x1="70" y1="284.0" x2="630" y2="284.0"/><text class="t-tick" x="62.0" y="288.0" text-anchor="end">0</text><line class="grid" x1="70" y1="229.0" x2="630" y2="229.0"/><text class="t-tick" x="62.0" y="233.0" text-anchor="end">0.25</text><line class="grid" x1="70" y1="174.0" x2="630" y2="174.0"/><text class="t-tick" x="62.0" y="178.0" text-anchor="end">0.5</text><line class="grid" x1="70" y1="119.0" x2="630" y2="119.0"/><text class="t-tick" x="62.0" y="123.0" text-anchor="end">0.75</text><line class="grid" x1="70" y1="64.0" x2="630" y2="64.0"/><text class="t-tick" x="62.0" y="68.0" text-anchor="end">1</text><text class="t-tick" x="70.0" y="302.0" text-anchor="middle">0</text><text class="t-tick" x="182.0" y="302.0" text-anchor="middle">0.2</text><text class="t-tick" x="294.0" y="302.0" text-anchor="middle">0.4</text><text class="t-tick" x="406.0" y="302.0" text-anchor="middle">0.6</text><text class="t-tick" x="490.0" y="302.0" text-anchor="middle">0.75</text><text class="t-tick" x="630.0" y="302.0" text-anchor="middle">1</text><line class="axis" x1="70" y1="284" x2="630" y2="284"/><polyline class="l1" points="70.0,284.0 81.2,284.0 92.4,284.0 103.6,284.0 114.8,284.0 126.0,284.0 137.2,284.0 148.4,284.0 159.6,284.0 170.8,284.0 182.0,284.0 193.2,284.0 204.4,284.0 215.6,283.9 226.8,283.9 238.0,283.8 249.2,283.7 260.4,283.5 271.6,283.1 282.8,282.7 294.0,282.0 305.2,281.0 316.4,279.7 327.6,277.9 338.8,275.5 350.0,272.3 361.2,268.1 372.4,262.7 383.6,256.0 394.8,247.7 406.0,237.6 417.2,225.5 428.4,211.5 439.6,195.7 450.8,178.3 462.0,159.8 473.2,141.0 484.4,122.8 495.6,106.2 506.8,92.0 518.0,80.8 529.2,73.0 540.4,68.1 551.6,65.5 562.8,64.4 574.0,64.1 585.2,64.0 596.4,64.0 607.6,64.0 618.8,64.0 630.0,64.0"/><text class="t-muted" x="350.0" y="322.0" text-anchor="middle">true Jaccard similarity of two documents (shared word 5-grams / all word 5-grams)</text><text class="t-muted" x="62.0" y="48.0" text-anchor="start">probability the pair is flagged as duplicate</text><circle class="s2 ring" cx="406.0" cy="237.6" r="5"/><text class="t-val" x="416.0" y="231.6" text-anchor="start">J=0.6: 0.21</text><circle class="s2 ring" cx="490.0" cy="114.2" r="5"/><text class="t-val" x="500.0" y="108.2" text-anchor="start">J=0.75: 0.77</text><circle class="s2 ring" cx="574.0" cy="64.1" r="5"/><text class="t-val" x="564.0" y="80.1" text-anchor="end">J=0.9: 1.00</text><text class="t-math" x="650.0" y="120.0" text-anchor="start">P = 1 - (1 - J^8)^14</text><text class="t-muted" x="650.0" y="142.0" text-anchor="start">all 8 hashes of one</text><text class="t-muted" x="650.0" y="158.0" text-anchor="start">bucket must match</text></svg><figcaption>The probability that a pair of documents is flagged by MinHash with 14 bands of 8 hashes, as a function of their true Jaccard similarity. The curve is a soft threshold that rises steeply between 0.6 and 0.9.</figcaption></figure>

On our 295 surviving pages, MinHash found 4 candidate pairs: two weather-satellite pages from the same site (Jaccard 1.00, the same page with different URL parameters), two pairs of library catalogue search results (0.88 and 0.87), and two laptop-battery product pages (true Jaccard 0.72, estimate 0.72). Removing one of each pair left 291 pages. In a single 12 MB sample duplicates are rare; across a whole crawl, and across many crawls, they are everywhere.

> [!WARNING]
> More deduplication is not always better. FineWeb found that deduplicating *across* all 96 snapshots at once removed so much that what was left was, on average, *lower* quality: good pages tend to be copied, so the survivors of an aggressive global pass were the pages no one wanted to copy. They deduplicated each snapshot on its own instead. Every choice in a data pipeline has to be checked by training a model on the result.

### 4.2.4 Quality classifiers

Rules catch pages that are obviously broken. They cannot tell a careful explanation from a well-formed sales page. For that, recent pipelines train a **quality classifier**.

> [!DEFINITION] Quality classifier
> A small model that reads a document and outputs a score for how useful it is likely to be as training data. The pipeline keeps documents above a threshold. GPT-3 used a classifier trained to tell curated text (books, Wikipedia, WebText) from raw web text; FineWeb-Edu and DCLM train classifiers on labels produced by a large language model or on hand-picked examples of good text.

FineWeb-Edu is the clearest example. The authors asked Llama-3-70B-Instruct to rate 460,000 web pages for "educational value" on a scale from 0 to 5, then trained a small, fast model to predict that rating, and ran it over all of FineWeb:

> [!PAPER] Penedo et al. (2024), The FineWeb Datasets · Section 4 · page 9
> [![The FineWeb-Edu paragraphs, with highlights on score 460,000 randomly sampled, on a scale from 0 to 5, and minimum threshold of 3](/img/training/ch4-fineweb-edu.png)](/img/training/ch4-fineweb-edu.png)
>
> **Context:** the section that builds FineWeb-Edu from FineWeb.
>
> **What it says:** a large model labels 460,000 pages; a linear regression head on top of a small embedding model (Snowflake-arctic-embed-m) is trained on 410,000 of those labels; its output is rounded to an integer from 0 to 5; pages scoring at least 3 are kept. Running it over 15 trillion tokens took 6,000 H100 GPU hours.
>
> **Why it matters:** the expensive judge (a 70B model) is used once, to make labels; the cheap student (about 100M parameters) does the work at scale. The resulting 1.3T-token FineWeb-Edu beat every other open web dataset on knowledge and reasoning benchmarks.

The released classifier is on the Hugging Face Hub, so `ch4_data.py` runs it on our 291 surviving pages:

```python
name = 'HuggingFaceFW/fineweb-edu-classifier'
tk = AutoTokenizer.from_pretrained(name)
clf = AutoModelForSequenceClassification.from_pretrained(name).to('mps').eval()
with torch.no_grad():
    for i in range(0, len(keep2), 16):
        enc = tk([t for _, t in keep2[i:i + 16]], return_tensors='pt', padding='longest',
                 truncation=True, max_length=512).to('mps')
        scores += clf(**enc).logits.squeeze(-1).float().cpu().tolist()
ints = [int(round(max(0, min(s, 5)))) for s in scores]
```

The first three lines download the tokenizer and the model (a BERT-like encoder with a one-number regression head) and move it to the Apple GPU (`'mps'`). The loop sends pages in groups of 16, cut to their first 512 tokens; `logits` holds a single number per page, the predicted score. The last line clips it to the range 0 to 5 and rounds it, as the paper does.

<figure class="fig"><svg viewBox="0 0 760 296" role="img" aria-label="Histogram of FineWeb-Edu classifier scores on the pages that survived the rule-based filters."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">FineWeb-Edu classifier on our 291 surviving pages: most web text is not "educational"</text><g class="mark"><title>score 0: 35</title><rect class="s1" x="90" y="208.0" width="70" height="32.0" rx="3"/></g><text class="t-val" x="125.0" y="202.0" text-anchor="middle">35</text><text class="t-tick" x="125.0" y="258.0" text-anchor="middle">score 0</text><g class="mark"><title>score 1: 197</title><rect class="s1" x="190" y="60.0" width="70" height="180.0" rx="3"/></g><text class="t-val" x="225.0" y="54.0" text-anchor="middle">197</text><text class="t-tick" x="225.0" y="258.0" text-anchor="middle">score 1</text><g class="mark"><title>score 2: 48</title><rect class="s1" x="290" y="196.1" width="70" height="43.9" rx="3"/></g><text class="t-val" x="325.0" y="190.1" text-anchor="middle">48</text><text class="t-tick" x="325.0" y="258.0" text-anchor="middle">score 2</text><g class="mark"><title>score 3: 10</title><rect class="s3" x="390" y="230.9" width="70" height="9.1" rx="3"/></g><text class="t-val" x="425.0" y="224.9" text-anchor="middle">10</text><text class="t-tick" x="425.0" y="258.0" text-anchor="middle">score 3</text><g class="mark"><title>score 4: 1</title><rect class="s3" x="490" y="239.1" width="70" height="0.9" rx="3"/></g><text class="t-val" x="525.0" y="233.1" text-anchor="middle">1</text><text class="t-tick" x="525.0" y="258.0" text-anchor="middle">score 4</text><g class="mark"><title>score 5: 0</title><rect class="s3" x="590" y="240.0" width="70" height="0.0" rx="3"/></g><text class="t-val" x="625.0" y="234.0" text-anchor="middle">0</text><text class="t-tick" x="625.0" y="258.0" text-anchor="middle">score 5</text><line class="axis" x1="80" y1="240" x2="690" y2="240"/><line class="edge-dim" x1="375.0" y1="50.0" x2="375.0" y2="250.0"/><text class="t-note" x="385.0" y="64.0" text-anchor="start">threshold 3: keep 11 of 291 (3.8%)</text><text class="t-muted" x="20.0" y="284.0" text-anchor="start">Rounded regression score, 0 = no educational value, 5 = excellent for teaching. FineWeb-Edu keeps scores of 3 and above.</text></svg><figcaption>Scores of the FineWeb-Edu classifier on our 291 surviving pages. Most web pages get a 1. Only 11 pages (3.8%) reach the threshold of 3.</figcaption></figure>

```text
FineWeb-Edu classifier scores (rounded 0..5): 0: 35, 1: 197, 2: 48, 3: 10, 4: 1, 5: 0
kept by FineWeb-Edu (score >= 3): 11 of 291 (3.8%)
   score -0.58  (adult site, text not shown)
   score 3.47  http://mkwc2.ifa.hawaii.edu/satellite/anim.cgi?chnl=08&anim=off&domain
   score 4.03  http://sites.cde.state.co.us/comath/researchandpracticeguides
```

Two things stand out. First, the lowest scores went to adult pages that the rules had let through. Real pipelines remove those earlier with a URL blocklist, and this shows why a blocklist is needed: such pages are well-formed text and pass every rule. Second, the classifier is not perfect. A guide to teaching mathematics scores 4.03, which is right, but two navigation pages of a weather-satellite site score above 3.3, probably because of their scientific vocabulary. At scale, a few errors do not matter; what matters is that the *average* quality of what is kept goes up. In FineWeb's own experiments, every stage of the pipeline improved the benchmark score of a model trained on the result:

> [!PAPER] Penedo et al. (2024), The FineWeb Datasets · Figures 9 and 10 · page 9
> [![Two plots of aggregate benchmark accuracy against training tokens. Left: base filtering is lowest, then individual MinHash, then C4 filters, then the full FineWeb pipeline. Right: FineWeb-Edu is above all other public datasets, including C4, RefinedWeb, Dolma and the Pile](/img/training/ch4-fineweb-fig-pipeline.png)](/img/training/ch4-fineweb-fig-pipeline.png)
>
> **Context:** the summary of the FineWeb pipeline and the comparison with other public datasets.
>
> **What it says:** on the left, each processing step gives a small model trained on the result a higher average benchmark score. On the right, FineWeb-Edu is the best open dataset by a clear margin, and FineWeb itself is near the top; the Pile and C4 are lower.
>
> **Why it matters:** better data is worth as much as more compute. Every line on the right was trained with the same model and the same number of tokens; only the data differs.

Real pipelines add a few more steps that we skip here: removing personal information (email addresses, phone numbers), removing text that overlaps with benchmark test sets (**decontamination**, see Chapter 3), and sometimes toxicity filters. The output of all of this is a set of clean documents, which now has to be turned into numbers.

## 4.3 Tokenization, briefly

A Transformer reads integers, not characters. A **tokenizer** turns text into a sequence of integer ids from a fixed **vocabulary**, and back. Chapter 1 used the Qwen2.5 tokenizer without looking inside it; here is how such a vocabulary is built.

> [!DEFINITION] Byte-pair encoding (BPE)
> A way to build a subword vocabulary. Start with single characters (or single bytes). Count every pair of adjacent symbols in the training text, merge the most frequent pair into one new symbol, and repeat until the vocabulary reaches the size you want. Frequent words end up as one token; rare words are split into pieces.

BPE came to language modelling from machine translation. The original paper by Sennrich, Haddow and Birch fits the whole algorithm into a few lines of Python:

> [!PAPER] Sennrich et al. (2016), Neural Machine Translation of Rare Words with Subword Units · Algorithm 1 · page 4
> [![Algorithm 1 of the BPE paper: a short Python listing with get_stats counting symbol pairs, merge_vocab replacing a pair by its concatenation, a toy vocabulary of low, lower, newest and widest, and a loop of ten merges with best = max(pairs, key=pairs.get) highlighted](/img/training/ch4-bpe-algorithm.png)](/img/training/ch4-bpe-algorithm.png)
>
> **Context:** Section 3.2 of the paper, where BPE is introduced for splitting words.
>
> **What it says:** words are written as space-separated characters with an end-of-word marker `</w>`. `get_stats` counts every adjacent pair, weighted by word frequency. The highlighted line picks the most frequent pair, and `merge_vocab` glues it together everywhere.
>
> **Why it matters:** this is the algorithm behind the GPT-2, Llama and Qwen tokenizers, with one change: modern tokenizers start from the 256 possible *bytes* instead of characters, so any text, in any language, can be encoded.

`ch4_tokenizer.py` runs the same algorithm on the same toy vocabulary (5 times "low", 2 times "lower", 6 times "newest", 3 times "widest") and prints each merge:

```text
merge  1: 'e' + 's'  (seen 9 times)  ->  ['l o w </w>', 'l o w e r </w>', 'n e w es t </w>', 'w i d es t </w>']
merge  2: 'es' + 't'  (seen 9 times)  ->  ['l o w </w>', 'l o w e r </w>', 'n e w est </w>', 'w i d est </w>']
merge  3: 'est' + '</w>'  (seen 9 times)  ->  [..., 'n e w est</w>', 'w i d est</w>']
merge  4: 'l' + 'o'  (seen 7 times)  ->  ['lo w </w>', 'lo w e r </w>', ...]
merge  5: 'lo' + 'w'  (seen 7 times)  ->  ['low </w>', 'low e r </w>', ...]
```

"e s" appears 6 times in "newest" and 3 times in "widest", 9 in total, more than any other pair, so it is merged first. (Several pairs tie at 9; Python's `max` keeps the first one it sees.) After three merges the suffix "est" with its end marker is one symbol; after five, "low" is. That is how BPE discovers pieces of words without being told: frequent pieces become tokens.

<figure class="fig"><svg viewBox="0 0 800 426" role="img" aria-label="Five frames showing byte-pair encoding merging the most frequent adjacent pair of symbols step by step."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Byte-pair encoding, merge by merge (the toy example of Sennrich et al., run by ch4_tokenizer.py)</text><rect class="box-ghost" x="20.0" y="44.0" width="760.0" height="62.0" rx="12"/><circle class="node on" cx="38.0" cy="62.0" r="11"/><text class="t-note" x="38.0" y="66.5" text-anchor="middle">1</text><text class="t-title" x="56.0" y="67.0" text-anchor="start">start: every word split into characters</text><rect class="box" x="56.0" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="65.0" y="93.5" text-anchor="middle">l</text><rect class="box" x="76.0" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="85.0" y="93.5" text-anchor="middle">o</text><rect class="box" x="96.0" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="105.0" y="93.5" text-anchor="middle">w</text><rect class="box" x="116.0" y="78.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="135.2" y="93.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="158.4" y="93.5" text-anchor="start">x5</text><rect class="box" x="180.4" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="189.4" y="93.5" text-anchor="middle">l</text><rect class="box" x="200.4" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="209.4" y="93.5" text-anchor="middle">o</text><rect class="box" x="220.4" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="229.4" y="93.5" text-anchor="middle">w</text><rect class="box" x="240.4" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="249.4" y="93.5" text-anchor="middle">e</text><rect class="box" x="260.4" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="269.4" y="93.5" text-anchor="middle">r</text><rect class="box" x="280.4" y="78.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="299.6" y="93.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="322.8" y="93.5" text-anchor="start">x2</text><rect class="box" x="344.8" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="353.8" y="93.5" text-anchor="middle">n</text><rect class="box" x="364.8" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="373.8" y="93.5" text-anchor="middle">e</text><rect class="box" x="384.8" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="393.8" y="93.5" text-anchor="middle">w</text><rect class="box" x="404.8" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="413.8" y="93.5" text-anchor="middle">e</text><rect class="box" x="424.8" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="433.8" y="93.5" text-anchor="middle">s</text><rect class="box" x="444.8" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="453.8" y="93.5" text-anchor="middle">t</text><rect class="box" x="464.8" y="78.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="484.0" y="93.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="507.2" y="93.5" text-anchor="start">x6</text><rect class="box" x="529.2" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="538.2" y="93.5" text-anchor="middle">w</text><rect class="box" x="549.2" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="558.2" y="93.5" text-anchor="middle">i</text><rect class="box" x="569.2" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="578.2" y="93.5" text-anchor="middle">d</text><rect class="box" x="589.2" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="598.2" y="93.5" text-anchor="middle">e</text><rect class="box" x="609.2" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="618.2" y="93.5" text-anchor="middle">s</text><rect class="box" x="629.2" y="78.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="638.2" y="93.5" text-anchor="middle">t</text><rect class="box" x="649.2" y="78.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="668.4" y="93.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="691.6" y="93.5" text-anchor="start">x3</text><rect class="box" x="20.0" y="116.0" width="760.0" height="62.0" rx="12"/><circle class="node on" cx="38.0" cy="134.0" r="11"/><text class="t-note" x="38.0" y="138.5" text-anchor="middle">2</text><text class="t-title" x="56.0" y="139.0" text-anchor="start">after merge 1: "e" + "s" (seen 9 times)</text><rect class="box" x="56.0" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="65.0" y="165.5" text-anchor="middle">l</text><rect class="box" x="76.0" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="85.0" y="165.5" text-anchor="middle">o</text><rect class="box" x="96.0" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="105.0" y="165.5" text-anchor="middle">w</text><rect class="box" x="116.0" y="150.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="135.2" y="165.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="158.4" y="165.5" text-anchor="start">x5</text><rect class="box" x="180.4" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="189.4" y="165.5" text-anchor="middle">l</text><rect class="box" x="200.4" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="209.4" y="165.5" text-anchor="middle">o</text><rect class="box" x="220.4" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="229.4" y="165.5" text-anchor="middle">w</text><rect class="box" x="240.4" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="249.4" y="165.5" text-anchor="middle">e</text><rect class="box" x="260.4" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="269.4" y="165.5" text-anchor="middle">r</text><rect class="box" x="280.4" y="150.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="299.6" y="165.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="322.8" y="165.5" text-anchor="start">x2</text><rect class="box" x="344.8" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="353.8" y="165.5" text-anchor="middle">n</text><rect class="box" x="364.8" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="373.8" y="165.5" text-anchor="middle">e</text><rect class="box" x="384.8" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="393.8" y="165.5" text-anchor="middle">w</text><rect class="box-1" x="404.8" y="150.0" width="23.2" height="22.0" rx="4"/><text class="t-tick" x="416.4" y="165.5" text-anchor="middle">es</text><rect class="box" x="430.0" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="439.0" y="165.5" text-anchor="middle">t</text><rect class="box" x="450.0" y="150.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="469.2" y="165.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="492.4" y="165.5" text-anchor="start">x6</text><rect class="box" x="514.4" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="523.4" y="165.5" text-anchor="middle">w</text><rect class="box" x="534.4" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="543.4" y="165.5" text-anchor="middle">i</text><rect class="box" x="554.4" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="563.4" y="165.5" text-anchor="middle">d</text><rect class="box-1" x="574.4" y="150.0" width="23.2" height="22.0" rx="4"/><text class="t-tick" x="586.0" y="165.5" text-anchor="middle">es</text><rect class="box" x="599.6" y="150.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="608.6" y="165.5" text-anchor="middle">t</text><rect class="box" x="619.6" y="150.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="638.8" y="165.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="662.0" y="165.5" text-anchor="start">x3</text><rect class="box" x="20.0" y="188.0" width="760.0" height="62.0" rx="12"/><circle class="node on" cx="38.0" cy="206.0" r="11"/><text class="t-note" x="38.0" y="210.5" text-anchor="middle">3</text><text class="t-title" x="56.0" y="211.0" text-anchor="start">after merge 3: "est" + "&lt;/w&gt;" (seen 9 times)</text><rect class="box" x="56.0" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="65.0" y="237.5" text-anchor="middle">l</text><rect class="box" x="76.0" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="85.0" y="237.5" text-anchor="middle">o</text><rect class="box" x="96.0" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="105.0" y="237.5" text-anchor="middle">w</text><rect class="box" x="116.0" y="222.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="135.2" y="237.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="158.4" y="237.5" text-anchor="start">x5</text><rect class="box" x="180.4" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="189.4" y="237.5" text-anchor="middle">l</text><rect class="box" x="200.4" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="209.4" y="237.5" text-anchor="middle">o</text><rect class="box" x="220.4" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="229.4" y="237.5" text-anchor="middle">w</text><rect class="box" x="240.4" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="249.4" y="237.5" text-anchor="middle">e</text><rect class="box" x="260.4" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="269.4" y="237.5" text-anchor="middle">r</text><rect class="box" x="280.4" y="222.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="299.6" y="237.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="322.8" y="237.5" text-anchor="start">x2</text><rect class="box" x="344.8" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="353.8" y="237.5" text-anchor="middle">n</text><rect class="box" x="364.8" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="373.8" y="237.5" text-anchor="middle">e</text><rect class="box" x="384.8" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="393.8" y="237.5" text-anchor="middle">w</text><rect class="box-1" x="404.8" y="222.0" width="61.2" height="22.0" rx="4"/><text class="t-tick" x="435.4" y="237.5" text-anchor="middle">est&lt;/w&gt;</text><text class="t-muted" x="470.0" y="237.5" text-anchor="start">x6</text><rect class="box" x="492.0" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="501.0" y="237.5" text-anchor="middle">w</text><rect class="box" x="512.0" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="521.0" y="237.5" text-anchor="middle">i</text><rect class="box" x="532.0" y="222.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="541.0" y="237.5" text-anchor="middle">d</text><rect class="box-1" x="552.0" y="222.0" width="61.2" height="22.0" rx="4"/><text class="t-tick" x="582.6" y="237.5" text-anchor="middle">est&lt;/w&gt;</text><text class="t-muted" x="617.2" y="237.5" text-anchor="start">x3</text><rect class="box" x="20.0" y="260.0" width="760.0" height="62.0" rx="12"/><circle class="node on" cx="38.0" cy="278.0" r="11"/><text class="t-note" x="38.0" y="282.5" text-anchor="middle">4</text><text class="t-title" x="56.0" y="283.0" text-anchor="start">after merge 5: "lo" + "w" (seen 7 times)</text><rect class="box-1" x="56.0" y="294.0" width="30.8" height="22.0" rx="4"/><text class="t-tick" x="71.4" y="309.5" text-anchor="middle">low</text><rect class="box" x="88.8" y="294.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="108.0" y="309.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="131.2" y="309.5" text-anchor="start">x5</text><rect class="box-1" x="153.2" y="294.0" width="30.8" height="22.0" rx="4"/><text class="t-tick" x="168.6" y="309.5" text-anchor="middle">low</text><rect class="box" x="186.0" y="294.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="195.0" y="309.5" text-anchor="middle">e</text><rect class="box" x="206.0" y="294.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="215.0" y="309.5" text-anchor="middle">r</text><rect class="box" x="226.0" y="294.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="245.2" y="309.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="268.4" y="309.5" text-anchor="start">x2</text><rect class="box" x="290.4" y="294.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="299.4" y="309.5" text-anchor="middle">n</text><rect class="box" x="310.4" y="294.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="319.4" y="309.5" text-anchor="middle">e</text><rect class="box" x="330.4" y="294.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="339.4" y="309.5" text-anchor="middle">w</text><rect class="box-1" x="350.4" y="294.0" width="61.2" height="22.0" rx="4"/><text class="t-tick" x="381.0" y="309.5" text-anchor="middle">est&lt;/w&gt;</text><text class="t-muted" x="415.6" y="309.5" text-anchor="start">x6</text><rect class="box" x="437.6" y="294.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="446.6" y="309.5" text-anchor="middle">w</text><rect class="box" x="457.6" y="294.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="466.6" y="309.5" text-anchor="middle">i</text><rect class="box" x="477.6" y="294.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="486.6" y="309.5" text-anchor="middle">d</text><rect class="box-1" x="497.6" y="294.0" width="61.2" height="22.0" rx="4"/><text class="t-tick" x="528.2" y="309.5" text-anchor="middle">est&lt;/w&gt;</text><text class="t-muted" x="562.8" y="309.5" text-anchor="start">x3</text><rect class="box" x="20.0" y="332.0" width="760.0" height="62.0" rx="12"/><circle class="node on" cx="38.0" cy="350.0" r="11"/><text class="t-note" x="38.0" y="354.5" text-anchor="middle">5</text><text class="t-title" x="56.0" y="355.0" text-anchor="start">after merge 8: "new" + "est&lt;/w&gt;" (seen 6 times)</text><rect class="box-1" x="56.0" y="366.0" width="30.8" height="22.0" rx="4"/><text class="t-tick" x="71.4" y="381.5" text-anchor="middle">low</text><rect class="box" x="88.8" y="366.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="108.0" y="381.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="131.2" y="381.5" text-anchor="start">x5</text><rect class="box-1" x="153.2" y="366.0" width="30.8" height="22.0" rx="4"/><text class="t-tick" x="168.6" y="381.5" text-anchor="middle">low</text><rect class="box" x="186.0" y="366.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="195.0" y="381.5" text-anchor="middle">e</text><rect class="box" x="206.0" y="366.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="215.0" y="381.5" text-anchor="middle">r</text><rect class="box" x="226.0" y="366.0" width="38.4" height="22.0" rx="4"/><text class="t-tick" x="245.2" y="381.5" text-anchor="middle">&lt;/w&gt;</text><text class="t-muted" x="268.4" y="381.5" text-anchor="start">x2</text><rect class="box-1" x="290.4" y="366.0" width="84.0" height="22.0" rx="4"/><text class="t-tick" x="332.4" y="381.5" text-anchor="middle">newest&lt;/w&gt;</text><text class="t-muted" x="378.4" y="381.5" text-anchor="start">x6</text><rect class="box" x="400.4" y="366.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="409.4" y="381.5" text-anchor="middle">w</text><rect class="box" x="420.4" y="366.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="429.4" y="381.5" text-anchor="middle">i</text><rect class="box" x="440.4" y="366.0" width="18.0" height="22.0" rx="4"/><text class="t-tick" x="449.4" y="381.5" text-anchor="middle">d</text><rect class="box-1" x="460.4" y="366.0" width="61.2" height="22.0" rx="4"/><text class="t-tick" x="491.0" y="381.5" text-anchor="middle">est&lt;/w&gt;</text><text class="t-muted" x="525.6" y="381.5" text-anchor="start">x3</text><text class="t-muted" x="20.0" y="412.0" text-anchor="start">Coloured blocks are merged symbols. After 10 merges, "low", "newest" and "est" are single tokens; real tokenizers do 50,000 to 150,000 merges.</text></svg><figcaption>Byte-pair encoding on the toy vocabulary of the BPE paper. Each frame shows the four words after one more merge; coloured blocks are symbols created by merges. The counts on the right of each word are word frequencies.</figcaption></figure>

Real tokenizers do the same thing tens of thousands of times on gigabytes of text. For the hands-on run, `ch4_gpt.py` trains a 4,096-token byte-level BPE on the TinyStories text with the Hugging Face `tokenizers` library. Comparing it with two real tokenizers shows what the vocabulary size buys:

```text
text: 'Pretraining is unbelievably expensive.'  (38 characters)
   GPT-2 (50,257)       7 tokens: P | ret | raining |  is |  unbelievably |  expensive | .
   Qwen2.5 (151,665)    7 tokens: Pre | training |  is |  unbelie | vably |  expensive | .
   TinyStories (4,096)  15 tokens: P | ret | ra | in | ing |  is |  un | b | el | ie | v | ab | ly |  expensive | .
text: 'The year 2024 had 366 days.'  (27 characters)
   GPT-2 (50,257)       7 tokens: The |  year |  2024 |  had |  366 |  days | .
   Qwen2.5 (151,665)   14 tokens: The |  year |   | 2 | 0 | 2 | 4 |  had |   | 3 | 6 | 6 |  days | .
```

<figure class="fig"><svg viewBox="0 0 760 428" role="img" aria-label="Token counts for six strings under the GPT-2, Qwen2.5 and a 4,096-token TinyStories tokenizer."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">How many tokens does the same text become? (ch4_tokenizer.py)</text><text class="t-tick" x="170.0" y="72.0" text-anchor="end">English sentence</text><text class="t-muted" x="170.0" y="87.0" text-anchor="end">23 characters</text><g class="mark"><title>GPT-2 (50,257): 7 tokens</title><rect class="s1" x="180" y="50.0" width="63.0" height="12" rx="2"/></g><text class="t-tick" x="249.0" y="60.0" text-anchor="start">7</text><g class="mark"><title>Qwen2.5 (151,665): 7 tokens</title><rect class="s2" x="180" y="65.0" width="63.0" height="12" rx="2"/></g><text class="t-tick" x="249.0" y="75.0" text-anchor="start">7</text><g class="mark"><title>TinyStories (4,096): 7 tokens</title><rect class="s3" x="180" y="80.0" width="63.0" height="12" rx="2"/></g><text class="t-tick" x="249.0" y="90.0" text-anchor="start">7</text><text class="t-tick" x="170.0" y="130.0" text-anchor="end">rare word</text><text class="t-muted" x="170.0" y="145.0" text-anchor="end">38 characters</text><g class="mark"><title>GPT-2 (50,257): 7 tokens</title><rect class="s1" x="180" y="108.0" width="63.0" height="12" rx="2"/></g><text class="t-tick" x="249.0" y="118.0" text-anchor="start">7</text><g class="mark"><title>Qwen2.5 (151,665): 7 tokens</title><rect class="s2" x="180" y="123.0" width="63.0" height="12" rx="2"/></g><text class="t-tick" x="249.0" y="133.0" text-anchor="start">7</text><g class="mark"><title>TinyStories (4,096): 15 tokens</title><rect class="s3" x="180" y="138.0" width="135.0" height="12" rx="2"/></g><text class="t-tick" x="321.0" y="148.0" text-anchor="start">15</text><text class="t-tick" x="170.0" y="188.0" text-anchor="end">Python code</text><text class="t-muted" x="170.0" y="203.0" text-anchor="end">27 characters</text><g class="mark"><title>GPT-2 (50,257): 11 tokens</title><rect class="s1" x="180" y="166.0" width="99.0" height="12" rx="2"/></g><text class="t-tick" x="285.0" y="176.0" text-anchor="start">11</text><g class="mark"><title>Qwen2.5 (151,665): 10 tokens</title><rect class="s2" x="180" y="181.0" width="90.0" height="12" rx="2"/></g><text class="t-tick" x="276.0" y="191.0" text-anchor="start">10</text><g class="mark"><title>TinyStories (4,096): 14 tokens</title><rect class="s3" x="180" y="196.0" width="126.0" height="12" rx="2"/></g><text class="t-tick" x="312.0" y="206.0" text-anchor="start">14</text><text class="t-tick" x="170.0" y="246.0" text-anchor="end">numbers</text><text class="t-muted" x="170.0" y="261.0" text-anchor="end">27 characters</text><g class="mark"><title>GPT-2 (50,257): 7 tokens</title><rect class="s1" x="180" y="224.0" width="63.0" height="12" rx="2"/></g><text class="t-tick" x="249.0" y="234.0" text-anchor="start">7</text><g class="mark"><title>Qwen2.5 (151,665): 14 tokens</title><rect class="s2" x="180" y="239.0" width="126.0" height="12" rx="2"/></g><text class="t-tick" x="312.0" y="249.0" text-anchor="start">14</text><g class="mark"><title>TinyStories (4,096): 13 tokens</title><rect class="s3" x="180" y="254.0" width="117.0" height="12" rx="2"/></g><text class="t-tick" x="303.0" y="264.0" text-anchor="start">13</text><text class="t-tick" x="170.0" y="304.0" text-anchor="end">Hindi</text><text class="t-muted" x="170.0" y="319.0" text-anchor="end">20 characters</text><g class="mark"><title>GPT-2 (50,257): 32 tokens</title><rect class="s1" x="180" y="282.0" width="288.0" height="12" rx="2"/></g><text class="t-tick" x="474.0" y="292.0" text-anchor="start">32</text><g class="mark"><title>Qwen2.5 (151,665): 20 tokens</title><rect class="s2" x="180" y="297.0" width="180.0" height="12" rx="2"/></g><text class="t-tick" x="366.0" y="307.0" text-anchor="start">20</text><g class="mark"><title>TinyStories (4,096): 50 tokens</title><rect class="s3" x="180" y="312.0" width="450.0" height="12" rx="2"/></g><text class="t-tick" x="636.0" y="322.0" text-anchor="start">50</text><text class="t-tick" x="170.0" y="362.0" text-anchor="end">TinyStories style</text><text class="t-muted" x="170.0" y="377.0" text-anchor="end">42 characters</text><g class="mark"><title>GPT-2 (50,257): 10 tokens</title><rect class="s1" x="180" y="340.0" width="90.0" height="12" rx="2"/></g><text class="t-tick" x="276.0" y="350.0" text-anchor="start">10</text><g class="mark"><title>Qwen2.5 (151,665): 10 tokens</title><rect class="s2" x="180" y="355.0" width="90.0" height="12" rx="2"/></g><text class="t-tick" x="276.0" y="365.0" text-anchor="start">10</text><g class="mark"><title>TinyStories (4,096): 10 tokens</title><rect class="s3" x="180" y="370.0" width="90.0" height="12" rx="2"/></g><text class="t-tick" x="276.0" y="380.0" text-anchor="start">10</text><rect class="s1" x="200" y="402" width="10" height="10" rx="2"/><text class="t-tick" x="216.0" y="411.0" text-anchor="start">GPT-2 (50,257)</text><rect class="s2" x="390" y="402" width="10" height="10" rx="2"/><text class="t-tick" x="406.0" y="411.0" text-anchor="start">Qwen2.5 (151,665)</text><rect class="s3" x="580" y="402" width="10" height="10" rx="2"/><text class="t-tick" x="596.0" y="411.0" text-anchor="start">TinyStories (4,096)</text></svg><figcaption>Token counts for six strings under three tokenizers. A small vocabulary trained on children's stories handles children's stories as well as the big ones, but needs about twice as many tokens for a rare word and two and a half times as many as Qwen2.5 for Hindi.</figcaption></figure>

Three lessons from these lines:

1. **A tokenizer reflects its training text.** The 4,096-token TinyStories vocabulary encodes "Once upon a time, a little girl named Lily" in 10 tokens, exactly like GPT-2 and Qwen, because those are the words it saw. "unbelievably" costs it 9 tokens.
2. **Numbers are a design choice.** GPT-2 learned "2024" and "366" as single tokens, because they were common. Qwen2.5 deliberately splits every number into single digits, so the model sees a consistent representation for arithmetic.
3. **Languages are not treated equally.** The Hindi greeting costs Qwen2.5 20 tokens, GPT-2 32 and the TinyStories tokenizer 50 (it falls back to raw bytes, several per character). More tokens per word means less text fits in the context and more compute per sentence.

> [!DEFINITION] Compression rate
> The average number of characters per token on some text. On our TinyStories training text, the 4,096-token BPE averages 4.03 characters per token (`ch4_prep.py`). Larger vocabularies compress better but make the embedding and output layers larger.

## 4.4 The objective at scale

Pretraining uses exactly the loss from Chapter 1: at every position, the cross-entropy between the model's predicted distribution and the actual next token. Two practical details turn it into a pretraining objective.

**Documents become one long stream.** All documents are tokenized, separated by a special end-of-text token, and concatenated. Our TinyStories training text becomes a single array of 14,880,999 token ids, with `<|endoftext|>` between stories.

**Training examples are windows cut from the stream.** Each example is a window of $$T$$ tokens (the context length); the target is the same window shifted left by one token, exactly the token shift of Chapter 1. Here is the function from `ch4_gpt.py`:

```python
def batch(data, B, T, gen, device):
    """B random windows of T+1 tokens: x is the window, y is the same window shifted left by one."""
    ix = torch.randint(len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + T + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
```

`ix` picks `B` random start positions in the token stream. For each start `i`, `x` holds tokens `i` to `i+T-1` and `y` holds tokens `i+1` to `i+T`: the right answer for position `t` of `x` is position `t` of `y`. The windows ignore document boundaries, so a window can hold the end of one story, an end-of-text token and the start of the next; the model learns that what follows `<|endoftext|>` has nothing to do with what came before.

With a batch of $$B$$ windows of $$T$$ tokens, the loss of one step is the average over all $$B \times T$$ predictions:

$$\mathcal{L} = -\frac{1}{BT} \sum_{b=1}^{B} \sum_{t=1}^{T} \log p_\theta\left(x_{b,t+1} \mid x_{b,1}, \ldots, x_{b,t}\right)$$

where:

- $$B$$ is the number of windows in the batch and $$T$$ the number of tokens in each,
- $$x_{b,t}$$ is the token at position $$t$$ of window $$b$$,
- $$p_\theta(\cdot \mid \ldots)$$ is the probability the model with weights $$\theta$$ gives to the true next token,
- the minus sign turns "high probability" into "low loss".

**Worked example.** Our run uses $$B = 32$$ and $$T = 256$$, so each step averages $$32 \times 256 = 8{,}192$$ predictions. Before training, the weights are random and the model's distribution over the 4,096 tokens is close to uniform, so each prediction should cost about $$\ln 4096 = 8.318$$ nats. The first logged loss of our run was 8.374, within 1% of that. This is a useful sanity check for any new training setup: if the initial loss is far from $$\ln V$$ (with $$V$$ the vocabulary size), something in the initialization or the loss is wrong.

> [!TIP]
> Every one of those 8,192 positions is a training example. That is why pretraining extracts so much from its data: a single 256-token window teaches 256 lessons at once, and the causal mask (Section 4.5) keeps all of them honest.

## 4.5 The Transformer forward pass, at a glance

Chapter 2 told the history of the Transformer, and this book does not re-derive attention. What matters for pretraining is the overall flow and the shapes, because they decide the parameter count and the compute. Here is one forward pass of the model we will train, with its real shapes:

<figure class="fig"><svg viewBox="0 0 780 330" role="img" aria-label="The forward pass of a small GPT: token ids become embeddings, pass through six blocks, become logits and then one loss number."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">One forward pass of the tiny GPT, with the real shapes (B = 32 windows, T = 256 tokens, d = 384, V = 4,096)</text><rect class="box" x="20.0" y="60.0" width="132.0" height="76.0" rx="10"/><text class="t-note" x="86.0" y="82.0" text-anchor="middle">token ids</text><text class="t-tick" x="86.0" y="101.0" text-anchor="middle">32 x 256</text><text class="t-tick" x="86.0" y="118.0" text-anchor="middle">integers 0..4095</text><line class="edge" x1="152.0" y1="98.0" x2="170.0" y2="98.0" marker-end="url(#ah)"/><rect class="box-1" x="170.0" y="60.0" width="132.0" height="76.0" rx="10"/><text class="t-note" x="236.0" y="82.0" text-anchor="middle">embeddings</text><text class="t-tick" x="236.0" y="101.0" text-anchor="middle">32 x 256 x 384</text><text class="t-tick" x="236.0" y="118.0" text-anchor="middle">token + position</text><line class="edge" x1="302.0" y1="98.0" x2="320.0" y2="98.0" marker-end="url(#ah)"/><rect class="box-2" x="320.0" y="60.0" width="132.0" height="76.0" rx="10"/><text class="t-note" x="386.0" y="82.0" text-anchor="middle">6 blocks</text><text class="t-tick" x="386.0" y="101.0" text-anchor="middle">32 x 256 x 384</text><text class="t-tick" x="386.0" y="118.0" text-anchor="middle">attention + MLP</text><line class="edge" x1="452.0" y1="98.0" x2="470.0" y2="98.0" marker-end="url(#ah)"/><rect class="box-3" x="470.0" y="60.0" width="132.0" height="76.0" rx="10"/><text class="t-note" x="536.0" y="82.0" text-anchor="middle">logits</text><text class="t-tick" x="536.0" y="101.0" text-anchor="middle">32 x 256 x 4096</text><text class="t-tick" x="536.0" y="118.0" text-anchor="middle">a score per token</text><line class="edge" x1="602.0" y1="98.0" x2="620.0" y2="98.0" marker-end="url(#ah)"/><rect class="box-4" x="620.0" y="60.0" width="132.0" height="76.0" rx="10"/><text class="t-note" x="686.0" y="82.0" text-anchor="middle">loss</text><text class="t-tick" x="686.0" y="101.0" text-anchor="middle">1 number</text><text class="t-tick" x="686.0" y="118.0" text-anchor="middle">mean cross-entropy</text><rect class="box-ghost" x="170.0" y="170.0" width="420.0" height="120.0" rx="12"/><text class="t-note" x="380.0" y="192.0" text-anchor="middle">inside each block (pre-norm, residual)</text><rect class="box-2" x="190.0" y="205.0" width="180.0" height="70.0" rx="10"/><text class="t-tick" x="280.0" y="227.0" text-anchor="middle">x + Attention(LN(x))</text><text class="t-muted" x="280.0" y="246.0" text-anchor="middle">mixes information</text><text class="t-muted" x="280.0" y="263.0" text-anchor="middle">across positions (causal)</text><rect class="box-2" x="390.0" y="205.0" width="180.0" height="70.0" rx="10"/><text class="t-tick" x="480.0" y="227.0" text-anchor="middle">x + MLP(LN(x))</text><text class="t-muted" x="480.0" y="246.0" text-anchor="middle">transforms each position</text><text class="t-muted" x="480.0" y="263.0" text-anchor="middle">on its own (384 to 1536)</text><line class="edge" x1="370.0" y1="240.0" x2="390.0" y2="240.0" marker-end="url(#ah)"/><line class="edge-dim" x1="390.0" y1="136.0" x2="380.0" y2="170.0"/><text class="t-muted" x="380.0" y="318.0" text-anchor="middle">Chapter 1 explained logits, softmax and cross-entropy. Pretraining just does this on trillions of tokens.</text></svg><figcaption>One forward pass of the tiny GPT with real shapes. Token ids become 384-number vectors, pass through six identical blocks, and become 4,096 scores per position. The loss compares those scores with the true next tokens. Each block adds the output of attention and of an MLP back onto its input (residual connections).</figcaption></figure>

The code of one block, from `ch4_gpt.py`:

```python
class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)              # queries, keys and values in one matrix
        self.proj = nn.Linear(d, d)                 # mixes the heads back together
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n_head = n_head

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (t.view(B, T, self.n_head, d // self.n_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # each position sees only the past
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))          # residual connection 1
        x = x + self.mlp(self.ln2(x))                                  # residual connection 2
        return x
```

Block by block:

- `__init__` creates the learnable parts: two LayerNorms (which rescale each vector to a standard size), one linear layer that produces queries, keys and values for all heads at once, one output projection, and the MLP, which widens each vector from 384 to 1,536 numbers, applies the GELU non-linearity and narrows it back.
- In `forward`, the input `x` has shape `(B, T, d)`. The `qkv` layer produces three tensors of the same shape, which are reshaped into `n_head` heads of `d / n_head = 64` numbers each.
- `scaled_dot_product_attention(..., is_causal=True)` is the whole attention computation in one call: scores between every query and every key, a mask that hides future positions, a softmax, and a weighted sum of values. The causal mask is what makes all $$T$$ predictions in a window honest: position $$t$$ cannot see token $$t+1$$.
- The two `x = x + ...` lines are the residual connections. Each sub-layer only adds a correction to its input, which keeps gradients healthy through many layers.

The full model (`GPT` in the same file) adds a token embedding table and a position embedding table before the blocks, a final LayerNorm after them, and an output layer that turns each 384-number vector into 4,096 scores. That output layer *shares* its weight matrix with the token embedding (**weight tying**), which saves 1.6 million parameters here.

### Counting parameters

Most of a Transformer's parameters live in its blocks, and a block's count has a simple form. Ignoring the small bias and LayerNorm vectors,

$$N_{\text{block}} \approx \underbrace{4d^2}_{\text{attention}} + \underbrace{8d^2}_{\text{MLP}} = 12 d^2, \qquad N \approx 12\, L\, d^2$$

where:

- $$d$$ is the model width (the length of each token's vector),
- $$4d^2$$ counts the query, key, value and output matrices, each $$d \times d$$,
- $$8d^2$$ counts the MLP's two matrices, $$d \times 4d$$ and $$4d \times d$$,
- $$L$$ is the number of blocks and $$N$$ the **non-embedding** parameter count.

**Worked example** (`ch4_worked.py`). With $$d = 384$$: $$12 \times 384^2 = 1{,}769{,}472$$ per block. The exact count, with biases and LayerNorms, is 1,774,464. Six blocks plus the final LayerNorm give 10,647,552 non-embedding parameters. The token embedding adds $$4096 \times 384 = 1{,}572{,}864$$ and the position embedding $$256 \times 384 = 98{,}304$$, for a total of 12,318,720, the same number PyTorch reports. Scaling-law papers usually count $$N$$ without embeddings, because a table lookup costs almost no compute.

## 4.6 How much compute: C ≈ 6ND

The cost of training is measured in **FLOPs**, floating-point operations (one multiply or one add). There is a famous rule of thumb for it.

> [!DEFINITION] FLOP and FLOP/s
> A FLOP is one floating-point operation. Training compute is counted in total FLOPs (a number like $$10^{24}$$); hardware speed is counted in FLOPs per second (FLOP/s). An H100 GPU does about 989 TFLOP/s ($$9.89 \times 10^{14}$$) in 16-bit arithmetic at its theoretical peak.

$$C \approx 6\, N\, D$$

where:

- $$C$$ is the total training compute in FLOPs,
- $$N$$ is the number of parameters,
- $$D$$ is the number of training tokens,
- 6 is the number of FLOPs per parameter per token.

Where does the 6 come from? In the forward pass, each parameter takes part in one multiply and one add per token: 2 FLOPs, so $$2N$$ per token. The backward pass has to compute two things for every layer: the gradient with respect to the layer's *input* (to pass the error further back) and the gradient with respect to its *weights* (to update them). Each costs about as much as the forward pass, so the backward pass costs about $$4N$$. The total is $$6N$$ per token. Kaplan et al. wrote it down in their scaling-law paper:

> [!PAPER] Kaplan et al. (2020), Scaling Laws for Neural Language Models · Section 2.1 · page 7
> [![A paragraph from Kaplan et al. ending with the highlighted phrase: compute as C approximately 6N floating point operators per training token](/img/training/ch4-kaplan-6n.png)](/img/training/ch4-kaplan-6n.png)
>
> **Context:** the section that counts parameters and compute for a Transformer, next to a table of per-operation costs.
>
> **What it says:** the forward pass costs about $$2N$$ FLOPs per token plus a term for attention over the context, which is small when the model width is much larger than the context length divided by 12. "Accounting for the backwards pass (approximately twice the compute as the forwards pass)" gives $$C \approx 6N$$ per training token.
>
> **Why it matters:** with one multiplication you can estimate the cost of any training run from two numbers that papers always report.

<figure class="fig"><svg viewBox="0 0 760 225" role="img" aria-label="The forward pass costs 2N FLOPs per token and the backward pass about 4N, for a total of 6N."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Why training costs about 6 N FLOPs per token</text><rect class="box-1" x="20.0" y="50.0" width="230.0" height="70.0" rx="10"/><text class="t-note" x="135.0" y="74.0" text-anchor="middle">forward: 2N</text><text class="t-muted" x="135.0" y="96.0" text-anchor="middle">each weight: one multiply + one add</text><rect class="box-2" x="262.0" y="50.0" width="230.0" height="70.0" rx="10"/><text class="t-note" x="377.0" y="74.0" text-anchor="middle">backward: gradient for inputs: 2N</text><text class="t-muted" x="377.0" y="96.0" text-anchor="middle">pass the error back through each layer</text><rect class="box-2" x="504.0" y="50.0" width="230.0" height="70.0" rx="10"/><text class="t-note" x="619.0" y="74.0" text-anchor="middle">backward: gradient for weights: 2N</text><text class="t-muted" x="619.0" y="96.0" text-anchor="middle">how each weight should change</text><path class="edge" d="M20.0,128.0 L20.0,136.0 L734.0,136.0 L734.0,128.0"/><text class="t-note" x="377.0" y="154.0" text-anchor="middle">total per token: 2N + 2N + 2N = 6N FLOPs</text><text class="t-tick" x="20.0" y="190.0" text-anchor="start">Worked: our tiny GPT has N = 12,318,720 parameters and saw D = 12,288,000 tokens.</text><text class="t-tick" x="20.0" y="210.0" text-anchor="start">C = 6 x 12,318,720 x 12,288,000 = 9.08e+14 FLOPs, about 5.5 minutes at the 2.76 TFLOP/s the laptop GPU reached.</text></svg><figcaption>Why training costs about 6N FLOPs per token: 2N for the forward pass, 2N to send the error back through the layers, and 2N to compute the gradient of every weight.</figcaption></figure>

**Worked examples** (`ch4_compute.py`):

```text
our tiny GPT (ch4_pretrain.py)   N=1.23e+07  D=1.23e+07  C=6ND= 9.08e+14 FLOPs  tokens/param=       1
GPT-3 175B                       N=1.75e+11  D=   3e+11  C=6ND= 3.15e+23 FLOPs  tokens/param=       2
Chinchilla 70B                   N=   7e+10  D= 1.4e+12  C=6ND= 5.88e+23 FLOPs  tokens/param=      20
Llama 3 8B                       N=   8e+09  D= 1.5e+13  C=6ND=  7.2e+23 FLOPs  tokens/param=   1,875
Llama 3 405B                     N=4.05e+11  D=1.56e+13  C=6ND= 3.79e+25 FLOPs  tokens/param=      39
Qwen2.5-0.5B                     N= 4.9e+08  D= 1.8e+13  C=6ND= 5.29e+22 FLOPs  tokens/param=  36,735
```

The formula is accurate: the GPT-3 paper reports $$3.14 \times 10^{23}$$ FLOPs for its largest model, and the Llama 3 paper reports $$3.8 \times 10^{25}$$ for the 405B model. Our run is $$6 \times 12{,}318{,}720 \times 12{,}288{,}000 = 9.08 \times 10^{14}$$ FLOPs.

<figure class="fig"><svg viewBox="0 0 760 276" role="img" aria-label="Bars of training compute for our tiny model and five well-known models, on a log scale."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Training compute C = 6ND, on a log scale (each grid line is 1,000x more)</text><line class="grid" x1="260.0" y1="40" x2="260.0" y2="228"/><text class="t-tick" x="260.0" y="240.0" text-anchor="middle">1e14</text><line class="grid" x1="367.5" y1="40" x2="367.5" y2="228"/><text class="t-tick" x="367.5" y="240.0" text-anchor="middle">1e17</text><line class="grid" x1="475.0" y1="40" x2="475.0" y2="228"/><text class="t-tick" x="475.0" y="240.0" text-anchor="middle">1e20</text><line class="grid" x1="582.5" y1="40" x2="582.5" y2="228"/><text class="t-tick" x="582.5" y="240.0" text-anchor="middle">1e23</text><line class="grid" x1="690.0" y1="40" x2="690.0" y2="228"/><text class="t-tick" x="690.0" y="240.0" text-anchor="middle">1e26</text><text class="t-tick" x="250.0" y="63.0" text-anchor="end">our tiny GPT</text><g class="mark"><title>our tiny GPT (ch4_pretrain.py): 9.08e+14 FLOPs</title><rect class="s3" x="260" y="51" width="34.3" height="17" rx="3"/></g><text class="t-val" x="300.3" y="64.0" text-anchor="start">9.1e+14</text><text class="t-tick" x="250.0" y="93.0" text-anchor="end">GPT-3 175B</text><g class="mark"><title>GPT-3 175B: 3.15e+23 FLOPs</title><rect class="s1" x="260" y="81" width="340.4" height="17" rx="3"/></g><text class="t-val" x="606.4" y="94.0" text-anchor="start">3.2e+23</text><text class="t-tick" x="250.0" y="123.0" text-anchor="end">Chinchilla 70B</text><g class="mark"><title>Chinchilla 70B: 5.88e+23 FLOPs</title><rect class="s1" x="260" y="111" width="350.1" height="17" rx="3"/></g><text class="t-val" x="616.1" y="124.0" text-anchor="start">5.9e+23</text><text class="t-tick" x="250.0" y="153.0" text-anchor="end">Llama 3 8B</text><g class="mark"><title>Llama 3 8B: 7.2e+23 FLOPs</title><rect class="s1" x="260" y="141" width="353.2" height="17" rx="3"/></g><text class="t-val" x="619.2" y="154.0" text-anchor="start">7.2e+23</text><text class="t-tick" x="250.0" y="183.0" text-anchor="end">Llama 3 405B</text><g class="mark"><title>Llama 3 405B: 3.79e+25 FLOPs</title><rect class="s1" x="260" y="171" width="414.9" height="17" rx="3"/></g><text class="t-val" x="680.9" y="184.0" text-anchor="start">3.8e+25</text><text class="t-tick" x="250.0" y="213.0" text-anchor="end">Qwen2.5-0.5B</text><g class="mark"><title>Qwen2.5-0.5B: 5.29e+22 FLOPs</title><rect class="s1" x="260" y="201" width="312.6" height="17" rx="3"/></g><text class="t-val" x="578.6" y="214.0" text-anchor="start">5.3e+22</text><text class="t-muted" x="20.0" y="264.0" text-anchor="start">Llama 3 405B used about 42 billion times the compute of our laptop run.</text></svg><figcaption>Training compute of six runs on a log scale. Each grid line is a factor of 1,000. The largest Llama 3 model used about 42 billion times the compute of our laptop run.</figcaption></figure>

Turning FLOPs into time needs the speed of the hardware and how much of that speed a real run gets.

> [!DEFINITION] Model FLOPs utilization (MFU)
> The share of the hardware's peak FLOP/s that a training run actually turns into useful model computation. Large runs on thousands of GPUs typically reach 35 to 45%; the rest is lost to memory traffic, communication between GPUs and waiting.

$$\text{time} = \frac{C}{n_{\text{GPU}} \times \text{peak FLOP/s} \times \text{MFU}}$$

where:

- $$C$$ is the training compute in FLOPs,
- $$n_{\text{GPU}}$$ is the number of GPUs,
- peak FLOP/s is one GPU's theoretical speed and MFU the share of it actually used.

**Worked example.** Llama 3 405B with 16,384 H100s at 989 TFLOP/s and 40% MFU: $$3.79 \times 10^{25} / (16{,}384 \times 9.89 \times 10^{14} \times 0.40) \approx 5.85 \times 10^{6}$$ seconds, or 67.7 days of pure computation (the paper reports MFU between 38% and 43%). On the laptop, the benchmark in `ch4_prep.py` measured 37,367 tokens per second in bf16, which is $$6 \times 12{,}318{,}720 \times 37{,}367 = 2.76$$ TFLOP/s. At that speed our $$9.08 \times 10^{14}$$ FLOPs take 5.5 minutes of pure training; the real run took 8.2 minutes, including evaluation and sampling, while other jobs shared the GPU. Llama 3 405B at laptop speed would take about 435,000 years.

## 4.7 Scaling laws: how big, and for how long?

Suppose you have a fixed compute budget $$C$$. Because $$C \approx 6ND$$, you can spend it on a big model trained on few tokens or a small model trained on many. Which is better? Two papers, two years apart, gave two different answers, and the difference shaped every model since.

> [!DEFINITION] Scaling law
> An empirical formula that predicts a model's loss from the resources used to train it: parameters $$N$$, training tokens $$D$$, or compute $$C$$. Scaling laws are fitted on many small runs and then used to plan one large run.

### 4.7.1 Kaplan et al. (2020): smooth power laws

Researchers at OpenAI trained hundreds of Transformers, from under a thousand to about 1.5 billion non-embedding parameters, and found that the test loss falls as a straight line on log-log axes in each resource, as long as the other two are not the bottleneck:

> [!PAPER] Kaplan et al. (2020), Scaling Laws for Neural Language Models · Figure 1 · page 3
> [![Three log-log plots of test loss. Left: loss against compute in PF-days, with many thin training curves and a dashed frontier L = (Cmin/2.3e8)^-0.050. Middle: loss against dataset size, a straight line L = (D/5.4e13)^-0.095. Right: loss against non-embedding parameters, a straight line L = (N/8.8e13)^-0.076](/img/training/ch4-kaplan-fig1.png)](/img/training/ch4-kaplan-fig1.png)
>
> **Context:** the first figure of the paper; the rest of the paper explains each fit.
>
> **What it says:** test loss is a power law in compute, in dataset size and in parameters, over six or more orders of magnitude, with no sign of bending. "For optimal performance all three factors must be scaled up in tandem."
>
> **Why it matters:** a power law means you can measure small models cheaply and extrapolate to a large one before you spend the money. This is how every large run is planned today.

The right panel's law is

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad \alpha_N = 0.076, \quad N_c = 8.8 \times 10^{13}$$

where:

- $$L$$ is the test loss in nats per token (on their WebText2 data, with their tokenizer),
- $$N$$ is the number of non-embedding parameters,
- $$N_c$$ is a fitted constant with the units of parameters,
- $$\alpha_N$$ is the fitted exponent; a small exponent means slow but steady improvement.

**Worked example** (`ch4_compute.py`). At $$N = 10^8$$: $$L = (8.8 \times 10^{13} / 10^8)^{0.076} = (8.8 \times 10^5)^{0.076} = 2.830$$. At $$N = 10^9$$ it is 2.376 and at $$N = 10^{10}$$ it is 1.994. Each factor of 10 in parameters multiplies the loss by $$10^{-0.076} = 0.839$$, a 16.1% drop. Note what a power law implies: the second 16% costs ten times as much as the first.

Kaplan et al. also concluded that, for a fixed compute budget, the model size should grow much faster than the data: roughly $$N_{\text{opt}} \propto C^{0.73}$$. Following that advice, GPT-3 (175B parameters) was trained on only 300B tokens, under 2 tokens per parameter.

### 4.7.2 Scaling laws on a laptop

You can see the same kind of law on a laptop. `ch4_scaling_mini.py` trains five GPTs, from 0.1M to 10.6M non-embedding parameters, with exactly the same data, recipe and number of steps (800 steps, 6.6M tokens each), and records the validation loss every 50 steps.

```text
d= 64 layers=2 heads=2:    100,096 non-embedding (   378,624 total) params, final val loss 3.396, C=1.49e+13 FLOPs, 45s
d=128 layers=2 heads=4:    396,800 non-embedding (   953,856 total) params, final val loss 2.880, C=3.75e+13 FLOPs, 66s
d=192 layers=4 heads=6:  1,779,840 non-embedding ( 2,615,424 total) params, final val loss 2.629, C=1.03e+14 FLOPs, 110s
d=256 layers=4 heads=8:  3,159,552 non-embedding ( 4,273,664 total) params, final val loss 2.489, C=1.68e+14 FLOPs, 135s
d=384 layers=6 heads=6: 10,647,552 non-embedding (12,318,720 total) params, final val loss 2.383, C=4.84e+14 FLOPs, 305s
```

Fitting $$\log L = \log(N_c^{\alpha}) - \alpha \log N$$ by least squares on these five points (`ch4_fit.py`) gives:

```text
fit on 5 runs (6.6M tokens each): log L = 2.064 -0.0757 log N
  -> L(N) = (Nc / N)^alpha with alpha = 0.076, Nc = 7.01e+11
  every 10x more parameters multiplies the loss by 0.840 here (Kaplan et al.: 0.839)
```

<figure class="fig"><svg viewBox="0 0 780 354" role="img" aria-label="Left: validation loss against training compute for five model sizes. Right: final loss against parameter count with a fitted power law."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Five model sizes, same data and recipe (ch4_scaling_mini.py)</text><line class="grid" x1="70" y1="276.8" x2="370" y2="276.8"/><text class="t-tick" x="62.0" y="280.8" text-anchor="end">2.5</text><line class="grid" x1="70" y1="248.0" x2="370" y2="248.0"/><text class="t-tick" x="62.0" y="252.0" text-anchor="end">3</text><line class="grid" x1="70" y1="219.2" x2="370" y2="219.2"/><text class="t-tick" x="62.0" y="223.2" text-anchor="end">3.5</text><line class="grid" x1="70" y1="190.5" x2="370" y2="190.5"/><text class="t-tick" x="62.0" y="194.5" text-anchor="end">4</text><line class="grid" x1="70" y1="161.8" x2="370" y2="161.8"/><text class="t-tick" x="62.0" y="165.8" text-anchor="end">4.5</text><line class="grid" x1="70" y1="133.0" x2="370" y2="133.0"/><text class="t-tick" x="62.0" y="137.0" text-anchor="end">5</text><line class="grid" x1="70" y1="104.2" x2="370" y2="104.2"/><text class="t-tick" x="62.0" y="108.2" text-anchor="end">5.5</text><line class="grid" x1="70" y1="75.5" x2="370" y2="75.5"/><text class="t-tick" x="62.0" y="79.5" text-anchor="end">6</text><text class="t-tick" x="80.1" y="312.0" text-anchor="middle">1e12</text><text class="t-tick" x="184.5" y="312.0" text-anchor="middle">1e13</text><text class="t-tick" x="288.8" y="312.0" text-anchor="middle">1e14</text><line class="axis" x1="70" y1="294" x2="370" y2="294"/><polyline class="l1" points="76.8,79.8 108.3,148.9 126.6,178.2 139.7,188.0 149.8,194.5 158.0,200.0 165.0,205.4 171.1,210.4 176.4,213.6 181.2,216.8 185.5,218.8 189.5,221.1 193.1,222.6 196.4,223.7 199.6,224.6 202.5,225.2"/><polyline class="l2" points="118.7,116.6 150.1,181.8 168.5,195.7 181.5,207.4 191.7,217.9 199.9,224.8 206.9,231.0 213.0,236.9 218.3,241.1 223.1,244.6 227.4,247.4 231.3,249.8 235.0,251.6 238.3,253.2 241.4,254.2 244.4,254.9"/><polyline class="l3" points="164.4,145.6 195.8,188.6 214.2,202.8 227.3,216.7 237.4,228.1 245.6,236.9 252.6,243.6 258.7,249.6 264.0,253.7 268.8,257.7 273.1,260.6 277.0,263.4 280.7,265.4 284.0,267.2 287.1,268.5 290.1,269.3"/><polyline class="l4" points="186.7,154.3 218.1,192.2 236.5,208.9 249.5,224.0 259.6,235.7 267.9,244.8 274.9,250.7 280.9,256.6 286.3,260.7 291.0,264.8 295.3,268.0 299.3,270.9 302.9,273.1 306.3,275.0 309.4,276.4 312.3,277.4"/><polyline class="edge-on" points="234.7,143.9 266.1,186.6 284.4,206.5 297.5,220.9 307.6,234.4 315.9,244.3 322.8,251.9 328.9,258.5 334.2,263.8 339.0,268.4 343.3,272.2 347.3,275.5 350.9,278.2 354.2,280.7 357.4,282.3 360.3,283.5"/><rect class="s1" x="214.5" y="220.2" width="10" height="10" rx="2"/><text class="t-tick" x="229.5" y="229.2" text-anchor="start">0.1M</text><rect class="s2" x="256.4" y="249.9" width="10" height="10" rx="2"/><text class="t-tick" x="271.4" y="258.9" text-anchor="start">0.4M</text><rect class="s3" x="302.1" y="265.9" width="10" height="10" rx="2"/><text class="t-tick" x="317.1" y="274.9" text-anchor="start">1.8M</text><rect class="s4" x="324.3" y="281.9" width="10" height="10" rx="2"/><text class="t-tick" x="339.3" y="290.9" text-anchor="start">3.2M</text><rect class="box-on" x="372.3" y="297.9" width="10" height="10" rx="2"/><text class="t-tick" x="387.3" y="306.9" text-anchor="start">10.6M</text><text class="t-muted" x="220.0" y="332.0" text-anchor="middle">training compute C = 6ND (FLOPs, log)</text><text class="t-muted" x="62.0" y="48.0" text-anchor="start">validation loss</text><line class="grid" x1="500" y1="274.8" x2="730" y2="274.8"/><text class="t-tick" x="492.0" y="278.8" text-anchor="end">2.4</text><line class="grid" x1="500" y1="198.2" x2="730" y2="198.2"/><text class="t-tick" x="492.0" y="202.2" text-anchor="end">2.8</text><line class="grid" x1="500" y1="121.5" x2="730" y2="121.5"/><text class="t-tick" x="492.0" y="125.5" text-anchor="end">3.2</text><text class="t-tick" x="509.8" y="312.0" text-anchor="middle">1e5</text><text class="t-tick" x="611.0" y="312.0" text-anchor="middle">1e6</text><text class="t-tick" x="712.2" y="312.0" text-anchor="middle">1e7</text><line class="axis" x1="500" y1="294" x2="730" y2="294"/><polyline class="l2" points="509.8,83.9 570.4,182.8 636.3,231.0 661.5,257.7 714.9,278.1"/><g class="mark"><title>measured: 1e5, 3.39639</title><circle class="s2 ring" cx="509.8" cy="83.9" r="4"/></g><g class="mark"><title>measured: 1e6, 2.88026</title><circle class="s2 ring" cx="570.4" cy="182.8" r="4"/></g><g class="mark"><title>measured: 1e6, 2.62872</title><circle class="s2 ring" cx="636.3" cy="231.0" r="4"/></g><g class="mark"><title>measured: 1e6, 2.48938</title><circle class="s2 ring" cx="661.5" cy="257.7" r="4"/></g><g class="mark"><title>measured: 1e7, 2.38287</title><circle class="s2 ring" cx="714.9" cy="278.1" r="4"/></g><text class="t-muted" x="615.0" y="332.0" text-anchor="middle">non-embedding parameters N (log)</text><text class="t-muted" x="492.0" y="48.0" text-anchor="start">final loss after 6.6M tokens</text><polyline class="edge-dim" points="509.8,103.0 519.9,113.9 530.0,124.7 540.2,135.2 550.3,145.6 560.4,155.7 570.5,165.7 580.6,175.6 590.8,185.2 600.9,194.7 611.0,204.0 621.1,213.2 631.2,222.2 641.4,231.1 651.5,239.8 661.6,248.3 671.7,256.7 681.8,265.0 691.9,273.1 702.1,281.1 712.2,288.9 722.3,296.6 732.4,304.2"/><text class="t-muted" x="730.0" y="84.0" text-anchor="end">dashed fit: L = (Nc/N)^0.076</text></svg><figcaption>Left: validation loss against training compute for five model sizes trained the same way. Each larger model costs more per token but ends lower. Right: the final losses against parameter count on a log axis, with a fitted power law (dashed).</figcaption></figure>

The exponent, 0.076, matches Kaplan's to three decimals. That is a coincidence: our data (children's stories), tokenizer (4,096 tokens) and training length (6.6M tokens, the same for all sizes) are all different, and a fit on five points is fragile. Do not read more into it than this: *the loss falls smoothly and predictably as the model grows*, even at a scale a laptop can afford. And the warning from the fit itself applies: extrapolating it to 100M parameters predicts a loss of 1.95, but our 14.9M training tokens would run out long before such a model could reach it. The data is the other half of the law.

### 4.7.3 Chinchilla (2022): grow data as fast as the model

Hoffmann et al. at DeepMind revisited the question with one important change. Kaplan's runs had used the same learning-rate schedule length for every run, which made shorter runs look worse than they were. Training over 400 models from 70M to over 16B parameters on 5B to 500B tokens, and matching the schedule to each run's length, they reached a different conclusion: **parameters and tokens should grow in equal proportion**.

> [!PAPER] Hoffmann et al. (2022), Training Compute-Optimal Large Language Models · Figure 1 · page 2
> [![Log-log plot of parameters against FLOPs, with three coloured lines from the paper's three approaches and the Kaplan et al. line above them, and markers for Chinchilla 70B, Gopher 280B, GPT-3 175B and Megatron-Turing NLG 530B](/img/training/ch4-chinchilla-fig1.png)](/img/training/ch4-chinchilla-fig1.png)
>
> **Context:** the headline figure.
>
> **What it says:** the three approaches of the paper (coloured lines) all predict much smaller compute-optimal models than the Kaplan line. At Gopher's budget, the optimum is not 280B parameters but around 70B, trained on more tokens.
>
> **Why it matters:** the large models of 2020 and 2021 (GPT-3, Gopher, Megatron-Turing NLG) sit far above the optimum: too big, trained on too little data.

The cleanest of their three methods is the **IsoFLOP** experiment.

> [!DEFINITION] IsoFLOP curve
> Fix one compute budget. Train several model sizes, each on exactly as many tokens as that budget allows ($$D = C / 6N$$), and plot the final loss against model size. The curve has a minimum: the compute-optimal model size for that budget.

> [!PAPER] Hoffmann et al. (2022), Training Compute-Optimal Large Language Models · Figure 3 · page 6
> [![Left: U-shaped curves of training loss against parameters, one per FLOP budget from 6e18 to 3e21. Centre and right: the minimum of each curve plotted against FLOPs gives straight lines for optimal parameters and optimal tokens](/img/training/ch4-chinchilla-fig3.png)](/img/training/ch4-chinchilla-fig3.png)
>
> **Context:** Approach 2 of the paper.
>
> **What it says:** each budget gives a valley. A model to the left of the valley is too small (it sees many tokens but cannot absorb them); a model to the right is too big (it is undertrained). The valleys move right as the budget grows, along straight lines in log-log space.
>
> **Why it matters:** the slopes of those lines are the answer to "how should I split my compute?". Both come out at about 0.5.

Their third approach fits one formula to all runs:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

where:

- $$E$$ is the irreducible loss: the entropy of natural text, which no model can beat,
- $$A / N^{\alpha}$$ is the extra loss from having a finite model,
- $$B / D^{\beta}$$ is the extra loss from seeing finite data,
- $$A$$, $$B$$, $$\alpha$$, $$\beta$$ are fitted constants.

> [!PAPER] Hoffmann et al. (2022), Training Compute-Optimal Large Language Models · Appendix D.2, Equation 10 · page 25
> [![Equation 10 of the Chinchilla paper: L(N, D) = E + A / N^0.34 + B / D^0.28, with E = 1.69, A = 406.4, B = 410.7 highlighted](/img/training/ch4-chinchilla-fit.png)](/img/training/ch4-chinchilla-fit.png)
>
> **Context:** the appendix that gives the fitted values for Approach 3.
>
> **What it says:** $$E = 1.69$$, $$A = 406.4$$, $$B = 410.7$$, $$\alpha = 0.34$$, $$\beta = 0.28$$, for loss in nats per token on their MassiveText data.
>
> **Why it matters:** with these five numbers you can predict the loss of any $$(N, D)$$ pair, and compare plans before training anything.

**Worked example** (`ch4_compute.py`). Compare Gopher (280B parameters, 300B tokens) with Chinchilla (70B parameters, 1.4T tokens), which used a similar budget:

```text
  Gopher 280B, 300B tokens: 6ND = 5.04e+23;  L = 1.69 + 0.052 + 0.251 = 1.993
  Chinchilla 70B, 1.4T tokens: 6ND = 5.88e+23;  L = 1.69 + 0.083 + 0.163 = 1.937
```

For Gopher the model term is small ($$406.4 / (2.8 \times 10^{11})^{0.34} = 0.052$$) but the data term is large ($$410.7 / (3 \times 10^{11})^{0.28} = 0.251$$): it is starved of data. Chinchilla trades a little model term (0.083) for a much smaller data term (0.163) and ends 0.056 nats lower. In the real experiment, Chinchilla, at a quarter of Gopher's size, beat it on almost every benchmark, and being four times smaller it was also much cheaper to use.

### 4.7.4 The compute-optimal split, derived

With the formula, the best split of a budget is a calculus exercise: minimize $$L(N, D)$$ subject to $$6ND = C$$. Substitute $$D = C / 6N$$, set the derivative with respect to $$N$$ to zero, and solve. The result is

$$N_{\text{opt}}(C) = G \left(\frac{C}{6}\right)^{a}, \qquad D_{\text{opt}}(C) = G^{-1} \left(\frac{C}{6}\right)^{b}$$

$$a = \frac{\beta}{\alpha + \beta}, \qquad b = \frac{\alpha}{\alpha + \beta}, \qquad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}}$$

where:

- $$N_{\text{opt}}$$ and $$D_{\text{opt}}$$ are the loss-minimizing model size and token count for budget $$C$$,
- $$a$$ and $$b$$ are the growth exponents (they add up to 1, because $$N \times D$$ must grow like $$C$$),
- $$G$$ is a constant built from the fitted values.

**Worked example** (`ch4_compute.py`). With the fitted values, $$a = 0.28 / 0.62 = 0.452$$, $$b = 0.548$$ and $$G = 1.345$$:

```text
  C =    1e+18:  N_opt =  8.06e+07  D_opt =  2.07e+09  tokens/param =   25.7  L = 3.535
  C =    1e+21:  N_opt =  1.82e+09  D_opt =  9.14e+10  tokens/param =   50.1  L = 2.329
  C = 5.76e+23:  N_opt =  3.22e+10  D_opt =  2.98e+12  tokens/param =   92.6  L = 1.931
  C =  3.8e+25:  N_opt =  2.13e+11  D_opt =  2.97e+13  tokens/param =  139.0  L = 1.817
```

<figure class="fig"><svg viewBox="0 0 800 344" role="img" aria-label="Compute-optimal parameters and tokens grow together as the compute budget grows."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Grow the model and the data together (Chinchilla Approach 3 fit)</text><line class="grid" x1="80" y1="275.6" x2="480" y2="275.6"/><text class="t-tick" x="72.0" y="279.6" text-anchor="end">1e8</text><line class="grid" x1="80" y1="205.0" x2="480" y2="205.0"/><text class="t-tick" x="72.0" y="209.0" text-anchor="end">1e10</text><line class="grid" x1="80" y1="134.5" x2="480" y2="134.5"/><text class="t-tick" x="72.0" y="138.5" text-anchor="end">1e12</text><line class="grid" x1="80" y1="64.0" x2="480" y2="64.0"/><text class="t-tick" x="72.0" y="68.0" text-anchor="end">1e14</text><text class="t-tick" x="94.7" y="312.0" text-anchor="middle">1e18</text><text class="t-tick" x="192.2" y="312.0" text-anchor="middle">1e20</text><text class="t-tick" x="289.7" y="312.0" text-anchor="middle">1e22</text><text class="t-tick" x="387.2" y="312.0" text-anchor="middle">1e24</text><line class="axis" x1="80" y1="294" x2="480" y2="294"/><polyline class="l1" points="94.7,278.9 192.2,247.0 240.9,231.1 375.5,187.1 387.2,183.3 464.2,158.2"/><g class="mark"><title>N_opt (parameters): 1e18, 1e8</title><circle class="s1 ring" cx="94.7" cy="278.9" r="4"/></g><g class="mark"><title>N_opt (parameters): 1e20, 1e9</title><circle class="s1 ring" cx="192.2" cy="247.0" r="4"/></g><g class="mark"><title>N_opt (parameters): 1e21, 1e9</title><circle class="s1 ring" cx="240.9" cy="231.1" r="4"/></g><g class="mark"><title>N_opt (parameters): 1e24, 1e11</title><circle class="s1 ring" cx="375.5" cy="187.1" r="4"/></g><g class="mark"><title>N_opt (parameters): 1e24, 1e11</title><circle class="s1 ring" cx="387.2" cy="183.3" r="4"/></g><g class="mark"><title>N_opt (parameters): 1e26, 1e11</title><circle class="s1 ring" cx="464.2" cy="158.2" r="4"/></g><polyline class="l2" points="94.7,229.2 192.2,190.5 240.9,171.2 375.5,117.8 387.2,113.2 464.2,82.6"/><g class="mark"><title>D_opt (tokens): 1e18, 1e9</title><circle class="s2 ring" cx="94.7" cy="229.2" r="4"/></g><g class="mark"><title>D_opt (tokens): 1e20, 1e10</title><circle class="s2 ring" cx="192.2" cy="190.5" r="4"/></g><g class="mark"><title>D_opt (tokens): 1e21, 1e11</title><circle class="s2 ring" cx="240.9" cy="171.2" r="4"/></g><g class="mark"><title>D_opt (tokens): 1e24, 1e12</title><circle class="s2 ring" cx="375.5" cy="117.8" r="4"/></g><g class="mark"><title>D_opt (tokens): 1e24, 1e13</title><circle class="s2 ring" cx="387.2" cy="113.2" r="4"/></g><g class="mark"><title>D_opt (tokens): 1e26, 1e13</title><circle class="s2 ring" cx="464.2" cy="82.6" r="4"/></g><rect class="s2" x="476.2" y="77.6" width="10" height="10" rx="2"/><text class="t-tick" x="491.2" y="86.6" text-anchor="start">D_opt (tokens)</text><rect class="s1" x="476.2" y="153.2" width="10" height="10" rx="2"/><text class="t-tick" x="491.2" y="162.2" text-anchor="start">N_opt (parameters)</text><text class="t-muted" x="280.0" y="332.0" text-anchor="middle">compute budget C (FLOPs)</text><text class="t-muted" x="72.0" y="48.0" text-anchor="start">compute-optimal size (log)</text><text class="t-muted" x="94.7" y="217.2" text-anchor="middle">26 tok/param</text><text class="t-muted" x="375.5" y="105.8" text-anchor="middle">93 tok/param</text><text class="t-muted" x="464.2" y="70.6" text-anchor="middle">139 tok/param</text><text class="t-muted" x="610.0" y="150.0" text-anchor="start">N grows as C^0.45</text><text class="t-muted" x="610.0" y="167.0" text-anchor="start">D grows as C^0.55</text><text class="t-muted" x="610.0" y="184.0" text-anchor="start"></text><text class="t-muted" x="610.0" y="201.0" text-anchor="start">Approaches 1 and 2 in the</text><text class="t-muted" x="610.0" y="218.0" text-anchor="start">paper give C^0.50 for both:</text><text class="t-muted" x="610.0" y="235.0" text-anchor="start">about 20 tokens per</text><text class="t-muted" x="610.0" y="252.0" text-anchor="start">parameter at every scale.</text></svg><figcaption>Compute-optimal model size and token count from the Chinchilla Approach 3 fit. Both grow steadily with the budget; tokens grow a little faster here. The paper's other two approaches give exponents of 0.50 for both, which means a constant ratio of about 20 tokens per parameter.</figcaption></figure>

Here is a subtlety worth knowing. Approach 3's fitted exponents make the tokens-per-parameter ratio drift upward with scale (from 26 to 139 in the table). The paper's Approaches 1 and 2 give $$a \approx b \approx 0.5$$, a constant ratio of about **20 tokens per parameter**, and that is the number everyone quotes. A 2024 replication (Besiroglu et al.) found that the Approach 3 fit in the paper was imprecise and that a corrected fit agrees with about 20. The paper's own Table 3, built from Approach 1, shows the rule directly:

> [!PAPER] Hoffmann et al. (2022), Training Compute-Optimal Large Language Models · Table 3 · page 8
> [![Table 3: estimated optimal FLOPs and tokens for model sizes from 400 million (1.92e19 FLOPs, 8.0 billion tokens) through 1 billion (20.2 billion tokens), 10 billion (205.1 billion tokens), 67 billion (1.5 trillion tokens) to 175 billion parameters (3.7 trillion tokens)](/img/training/ch4-chinchilla-table3.png)](/img/training/ch4-chinchilla-table3.png)
>
> **Context:** the projections of Approach 1 for common model sizes.
>
> **What it says:** a 1B model should see 20.2B tokens; a 10B model 205.1B; a 175B model (GPT-3's size) 3.7 trillion, twelve times what GPT-3 actually saw.
>
> **Why it matters:** divide the last column by the first and you get about 20 every time. That ratio became the default plan for 2022 and 2023.

**Worked example: the 20-tokens rule.** With $$D = 20N$$, the budget is $$C = 6N \cdot 20N = 120 N^2$$, so $$N = \sqrt{C / 120}$$. For $$C = 10^{21}$$: $$N = \sqrt{10^{21} / 120} = 2.89 \times 10^9$$ and $$D = 5.77 \times 10^{10}$$. To see how forgiving the optimum is, `ch4_compute.py` also evaluates the Approach 3 formula along that whole budget:

```text
  IsoFLOP slice at C = 1e21 with the Approach 3 fit:
    N =    2e+08  D = 8.33e+11  ( 4166.7 tok/param)  L = 2.4905
    N =    7e+08  D = 2.38e+11  (  340.1 tok/param)  L = 2.3575
    N =  1.5e+09  D = 1.11e+11  (   74.1 tok/param)  L = 2.3301   <- lowest
    N =  2.9e+09  D = 5.75e+10  (   19.8 tok/param)  L = 2.3354
    N =    1e+10  D = 1.67e+10  (    1.7 tok/param)  L = 2.4160
```

<figure class="fig"><svg viewBox="0 0 760 334" role="img" aria-label="An isoFLOP curve: predicted loss for different model sizes at a fixed compute budget of 1e21 FLOPs, with a minimum in the middle."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">One compute budget, many ways to spend it (Chinchilla Approach 3 fit, ch4_compute.py)</text><line class="grid" x1="80" y1="264.0" x2="600" y2="264.0"/><text class="t-tick" x="72.0" y="268.0" text-anchor="end">2.32</text><line class="grid" x1="80" y1="224.0" x2="600" y2="224.0"/><text class="t-tick" x="72.0" y="228.0" text-anchor="end">2.36</text><line class="grid" x1="80" y1="184.0" x2="600" y2="184.0"/><text class="t-tick" x="72.0" y="188.0" text-anchor="end">2.40</text><line class="grid" x1="80" y1="144.0" x2="600" y2="144.0"/><text class="t-tick" x="72.0" y="148.0" text-anchor="end">2.44</text><line class="grid" x1="80" y1="104.0" x2="600" y2="104.0"/><text class="t-tick" x="72.0" y="108.0" text-anchor="end">2.48</text><line class="grid" x1="80" y1="64.0" x2="600" y2="64.0"/><text class="t-tick" x="72.0" y="68.0" text-anchor="end">2.52</text><text class="t-tick" x="113.5" y="302.0" text-anchor="middle">0.2B</text><text class="t-tick" x="220.3" y="302.0" text-anchor="middle">0.5B</text><text class="t-tick" x="301.1" y="302.0" text-anchor="middle">1B</text><text class="t-tick" x="381.9" y="302.0" text-anchor="middle">2B</text><text class="t-tick" x="488.6" y="302.0" text-anchor="middle">5B</text><text class="t-tick" x="569.4" y="302.0" text-anchor="middle">10B</text><line class="axis" x1="80" y1="284" x2="600" y2="284"/><polyline class="l1" points="113.5,93.5 194.3,181.6 259.5,226.5 301.1,244.0 348.3,253.9 425.2,248.6 488.6,224.6 569.4,168.0"/><g class="mark"><title>C = 1e21 FLOPs: 0.2B, 2.49</title><circle class="s1 ring" cx="113.5" cy="93.5" r="4"/></g><g class="mark"><title>C = 1e21 FLOPs: 0.4B, 2.40</title><circle class="s1 ring" cx="194.3" cy="181.6" r="4"/></g><g class="mark"><title>C = 1e21 FLOPs: 0.7B, 2.36</title><circle class="s1 ring" cx="259.5" cy="226.5" r="4"/></g><g class="mark"><title>C = 1e21 FLOPs: 1B, 2.34</title><circle class="s1 ring" cx="301.1" cy="244.0" r="4"/></g><g class="mark"><title>C = 1e21 FLOPs: 1.5B, 2.33</title><circle class="s1 ring" cx="348.3" cy="253.9" r="4"/></g><g class="mark"><title>C = 1e21 FLOPs: 2.9B, 2.34</title><circle class="s1 ring" cx="425.2" cy="248.6" r="4"/></g><g class="mark"><title>C = 1e21 FLOPs: 5B, 2.36</title><circle class="s1 ring" cx="488.6" cy="224.6" r="4"/></g><g class="mark"><title>C = 1e21 FLOPs: 10B, 2.42</title><circle class="s1 ring" cx="569.4" cy="168.0" r="4"/></g><text class="t-muted" x="340.0" y="322.0" text-anchor="middle">model size N (log scale); D = C / 6N is whatever the budget leaves</text><text class="t-muted" x="72.0" y="48.0" text-anchor="start">predicted loss L(N, D)</text><circle class="s2 ring" cx="348.3" cy="253.9" r="7"/><text class="t-val" x="348.3" y="279.9" text-anchor="middle">lowest: N = 1.5B, D = 111B</text><text class="t-muted" x="124.6" y="81.5" text-anchor="start">too small, too many tokens</text><text class="t-muted" x="569.4" y="156.0" text-anchor="end">too big, too few tokens</text></svg><figcaption>An isoFLOP slice computed from the Chinchilla formula at a budget of 1e21 FLOPs. Every point costs the same; the loss is lowest in the middle. The valley is flat: models between about 1B and 3B parameters are all within 0.01 nats of the best.</figcaption></figure>

The valley is flat near the bottom: the 20-tokens choice (2.9B parameters) is only 0.005 nats worse than the formula's own optimum (1.5B). Being off by a factor of two in model size costs little; being off by a factor of ten (2B parameters with 4,000 tokens each, or 10B with under 2) costs a lot.

### 4.7.5 Beyond compute-optimal: over-training on purpose

"Compute-optimal" answers one question: the lowest loss *for a training budget*. But a model is trained once and then used millions of times, and the cost of using it grows with its size. If you plan to serve a model heavily, it pays to train a *smaller* model for *longer* than Chinchilla suggests: you spend more on training to get a model that is cheaper at every use.

<figure class="fig"><svg viewBox="0 0 760 246" role="img" aria-label="Tokens per parameter for GPT-3, Chinchilla, Llama 3 and Qwen2.5, on a log scale."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Tokens seen per parameter: from "Kaplan-style" to "Chinchilla-optimal" to "over-trained"</text><line class="grid" x1="220.0" y1="40" x2="220.0" y2="196"/><text class="t-tick" x="220.0" y="210.0" text-anchor="middle">1</text><line class="grid" x1="316.0" y1="40" x2="316.0" y2="196"/><text class="t-tick" x="316.0" y="210.0" text-anchor="middle">10</text><line class="grid" x1="412.0" y1="40" x2="412.0" y2="196"/><text class="t-tick" x="412.0" y="210.0" text-anchor="middle">100</text><line class="grid" x1="508.0" y1="40" x2="508.0" y2="196"/><text class="t-tick" x="508.0" y="210.0" text-anchor="middle">1,000</text><line class="grid" x1="604.0" y1="40" x2="604.0" y2="196"/><text class="t-tick" x="604.0" y="210.0" text-anchor="middle">10,000</text><line class="grid" x1="700.0" y1="40" x2="700.0" y2="196"/><text class="t-tick" x="700.0" y="210.0" text-anchor="middle">100,000</text><line class="key-line" x1="344.9" y1="36.0" x2="344.9" y2="196.0"/><text class="t-tick" x="350.9" y="44.0" text-anchor="start">about 20 (Chinchilla)</text><text class="t-tick" x="210.0" y="65.0" text-anchor="end">GPT-3 175B</text><g class="mark"><title>GPT-3 175B: 2</title><rect class="s1" x="220" y="53" width="22.5" height="17" rx="3"/></g><text class="t-val" x="248.5" y="66.0" text-anchor="start">2</text><text class="t-tick" x="210.0" y="95.0" text-anchor="end">Chinchilla 70B</text><g class="mark"><title>Chinchilla 70B: 20</title><rect class="s1" x="220" y="83" width="124.9" height="17" rx="3"/></g><text class="t-val" x="350.9" y="96.0" text-anchor="start">20</text><text class="t-tick" x="210.0" y="125.0" text-anchor="end">Llama 3 8B</text><g class="mark"><title>Llama 3 8B: 1,875</title><rect class="s1" x="220" y="113" width="314.2" height="17" rx="3"/></g><text class="t-val" x="540.2" y="126.0" text-anchor="start">1,875</text><text class="t-tick" x="210.0" y="155.0" text-anchor="end">Llama 3 405B</text><g class="mark"><title>Llama 3 405B: 39</title><rect class="s1" x="220" y="143" width="152.2" height="17" rx="3"/></g><text class="t-val" x="378.2" y="156.0" text-anchor="start">39</text><text class="t-tick" x="210.0" y="185.0" text-anchor="end">Qwen2.5-0.5B</text><g class="mark"><title>Qwen2.5-0.5B: 36,735</title><rect class="s1" x="220" y="173" width="438.2" height="17" rx="3"/></g><text class="t-val" x="664.2" y="186.0" text-anchor="start">36,735</text><text class="t-muted" x="20.0" y="234.0" text-anchor="start">Small models meant to be run cheaply are trained far past 20 tokens per parameter: it costs more training, but less at use time.</text></svg><figcaption>Tokens per parameter for five well-known models, on a log scale. GPT-3 was undertrained by Chinchilla's standard; Chinchilla sits at 20; Llama 3 405B is close to compute-optimal; the small Llama 3 8B and Qwen2.5-0.5B are trained far beyond it.</figcaption></figure>

The numbers are striking. Llama 3 8B saw 15 trillion tokens: 1,875 per parameter, almost 100 times the Chinchilla ratio. Qwen2.5-0.5B, the model we use throughout this book, saw 18 trillion tokens: about 36,700 per parameter. Its loss keeps falling far past the Chinchilla point, just more slowly. The Llama 3 paper also shows that the flagship's size was chosen with a scaling law of its own: its isoFLOP experiments, extrapolated to its $$3.8 \times 10^{25}$$ FLOP budget, pointed to a model of about 402B parameters trained on 16.55T tokens, close to what they built.

> [!NOTE]
> Scaling laws predict *loss*, not abilities. A lower loss reliably comes with better benchmark scores on average, but specific skills (multi-step arithmetic, following a format) can appear abruptly at some size. Chapter 3 discussed how to measure those.

## 4.8 The learning rate: warm up, then decay

The optimizer for nearly every pretraining run is **AdamW**, which Chapter 1's training loop used. What changes at scale is that the learning rate is not a constant: it follows a **schedule**.

> [!DEFINITION] Learning-rate schedule
> A rule that sets the learning rate at each step. The usual pretraining schedule has a **warmup** (the rate grows linearly from near zero to its peak over the first few hundred or thousand steps) followed by a **decay** (the rate falls, usually along a cosine curve, to a small final value).

Why warm up? At step 0 the weights are random and the gradients are large and erratic. AdamW also needs a few steps for its running estimates of gradient size to settle. A full learning rate at that moment can throw the weights into a bad region and make the loss explode. A short ramp avoids that. Why decay? Late in training, the model is close to a good solution, and a large step size makes it bounce around the bottom of the valley instead of settling in. Lowering the rate lets it settle, and the loss usually drops visibly as the rate falls.

The schedule used in our run, from `ch4_gpt.py`:

```python
def lr_at(step, peak, warmup, total, floor=0.1):
    """Linear warmup to `peak`, then cosine decay down to floor * peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))
```

In formula form, for step $$s$$ after warmup:

$$\eta(s) = \eta_{\max}\left[f + (1 - f)\,\frac{1 + \cos(\pi p)}{2}\right], \qquad p = \frac{s - s_w}{S - s_w}$$

where:

- $$\eta_{\max}$$ is the peak learning rate ($$10^{-3}$$ in our run),
- $$s_w$$ is the number of warmup steps (100) and $$S$$ the total number of steps (1,500),
- $$p$$ is the fraction of the decay phase completed, from 0 to 1,
- $$f$$ is the final rate as a fraction of the peak (0.1),
- $$\frac{1 + \cos(\pi p)}{2}$$ falls smoothly from 1 to 0 as $$p$$ goes from 0 to 1.

**Worked example** (`ch4_worked.py`). At step 50, still in warmup: $$10^{-3} \times 51 / 100 = 5.1 \times 10^{-4}$$. At step 800: $$p = 700 / 1400 = 0.5$$, $$\cos(\pi/2) = 0$$, so $$\eta = 10^{-3} \times (0.1 + 0.9 \times 0.5) = 5.5 \times 10^{-4}$$. At the last step, 1,499, the rate is $$1.0 \times 10^{-4}$$, a tenth of the peak.

<figure class="fig"><svg viewBox="0 0 760 360" role="img" aria-label="The learning rate of our run rises linearly for 100 steps and then follows a cosine down to 10% of its peak; a warmup-stable-decay schedule is drawn for comparison."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Learning-rate schedules: warm up, hold high, come down</text><line class="grid" x1="80" y1="264.0" x2="640" y2="264.0"/><text class="t-tick" x="72.0" y="268.0" text-anchor="end">0</text><line class="grid" x1="80" y1="218.5" x2="640" y2="218.5"/><text class="t-tick" x="72.0" y="222.5" text-anchor="end">2.5e-04</text><line class="grid" x1="80" y1="173.1" x2="640" y2="173.1"/><text class="t-tick" x="72.0" y="177.1" text-anchor="end">5.0e-04</text><line class="grid" x1="80" y1="127.6" x2="640" y2="127.6"/><text class="t-tick" x="72.0" y="131.6" text-anchor="end">7.5e-04</text><line class="grid" x1="80" y1="82.2" x2="640" y2="82.2"/><text class="t-tick" x="72.0" y="86.2" text-anchor="end">1.0e-03</text><text class="t-tick" x="80.0" y="282.0" text-anchor="middle">0</text><text class="t-tick" x="117.3" y="282.0" text-anchor="middle">100</text><text class="t-tick" x="192.0" y="282.0" text-anchor="middle">300</text><text class="t-tick" x="304.0" y="282.0" text-anchor="middle">600</text><text class="t-tick" x="416.0" y="282.0" text-anchor="middle">900</text><text class="t-tick" x="528.0" y="282.0" text-anchor="middle">1,200</text><text class="t-tick" x="640.0" y="282.0" text-anchor="middle">1,500</text><line class="axis" x1="80" y1="264" x2="640" y2="264"/><polyline class="l1" points="80.0,262.2 83.7,244.0 87.5,225.8 91.2,207.6 94.9,189.5 98.7,171.3 102.4,153.1 106.1,134.9 109.9,116.7 113.6,98.5 117.3,82.2 121.1,82.2 124.8,82.3 128.5,82.4 132.3,82.5 136.0,82.7 139.7,82.9 143.5,83.2 147.2,83.5 150.9,83.8 154.7,84.2 158.4,84.7 162.1,85.1 165.9,85.6 169.6,86.2 173.3,86.8 177.1,87.4 180.8,88.1 184.5,88.8 188.3,89.5 192.0,90.3 195.7,91.1 199.5,92.0 203.2,92.8 206.9,93.8 210.7,94.7 214.4,95.7 218.1,96.7 221.9,97.8 225.6,98.9 229.3,100.0 233.1,101.2 236.8,102.4 240.5,103.6 244.3,104.9 248.0,106.1 251.7,107.5 255.5,108.8 259.2,110.2 262.9,111.6 266.7,113.0 270.4,114.4 274.1,115.9 277.9,117.4 281.6,118.9 285.3,120.5 289.1,122.0 292.8,123.6 296.5,125.2 300.3,126.9 304.0,128.5 307.7,130.2 311.5,131.8 315.2,133.5 318.9,135.3 322.7,137.0 326.4,138.7 330.1,140.5 333.9,142.2 337.6,144.0 341.3,145.8 345.1,147.6 348.8,149.4 352.5,151.2 356.3,153.0 360.0,154.8 363.7,156.7 367.5,158.5 371.2,160.3 374.9,162.2 378.7,164.0 382.4,165.8 386.1,167.7 389.9,169.5 393.6,171.3 397.3,173.2 401.1,175.0 404.8,176.8 408.5,178.6 412.3,180.4 416.0,182.2 419.7,184.0 423.5,185.8 427.2,187.5 430.9,189.3 434.7,191.0 438.4,192.7 442.1,194.5 445.9,196.2 449.6,197.8 453.3,199.5 457.1,201.1 460.8,202.8 464.5,204.4 468.3,206.0 472.0,207.5 475.7,209.1 479.5,210.6 483.2,212.1 486.9,213.6 490.7,215.0 494.4,216.4 498.1,217.8 501.9,219.2 505.6,220.5 509.3,221.9 513.1,223.1 516.8,224.4 520.5,225.6 524.3,226.8 528.0,228.0 531.7,229.1 535.5,230.2 539.2,231.3 542.9,232.3 546.7,233.3 550.4,234.2 554.1,235.2 557.9,236.0 561.6,236.9 565.3,237.7 569.1,238.5 572.8,239.2 576.5,239.9 580.3,240.6 584.0,241.2 587.7,241.8 591.5,242.4 595.2,242.9 598.9,243.3 602.7,243.8 606.4,244.2 610.1,244.5 613.9,244.8 617.6,245.1 621.3,245.3 625.1,245.5 628.8,245.6 632.5,245.7 636.3,245.8"/><text class="t-muted" x="360.0" y="302.0" text-anchor="middle">training step</text><text class="t-muted" x="72.0" y="48.0" text-anchor="start">learning rate</text><polyline class="l2" style="stroke-dasharray:6 4" points="80.0,262.2 83.7,244.0 87.5,225.8 91.2,207.6 94.9,189.5 98.7,171.3 102.4,153.1 106.1,134.9 109.9,116.7 113.6,98.5 117.3,82.2 121.1,82.2 124.8,82.2 128.5,82.2 132.3,82.2 136.0,82.2 139.7,82.2 143.5,82.2 147.2,82.2 150.9,82.2 154.7,82.2 158.4,82.2 162.1,82.2 165.9,82.2 169.6,82.2 173.3,82.2 177.1,82.2 180.8,82.2 184.5,82.2 188.3,82.2 192.0,82.2 195.7,82.2 199.5,82.2 203.2,82.2 206.9,82.2 210.7,82.2 214.4,82.2 218.1,82.2 221.9,82.2 225.6,82.2 229.3,82.2 233.1,82.2 236.8,82.2 240.5,82.2 244.3,82.2 248.0,82.2 251.7,82.2 255.5,82.2 259.2,82.2 262.9,82.2 266.7,82.2 270.4,82.2 274.1,82.2 277.9,82.2 281.6,82.2 285.3,82.2 289.1,82.2 292.8,82.2 296.5,82.2 300.3,82.2 304.0,82.2 307.7,82.2 311.5,82.2 315.2,82.2 318.9,82.2 322.7,82.2 326.4,82.2 330.1,82.2 333.9,82.2 337.6,82.2 341.3,82.2 345.1,82.2 348.8,82.2 352.5,82.2 356.3,82.2 360.0,82.2 363.7,82.2 367.5,82.2 371.2,82.2 374.9,82.2 378.7,82.2 382.4,82.2 386.1,82.2 389.9,82.2 393.6,82.2 397.3,82.2 401.1,82.2 404.8,82.2 408.5,82.2 412.3,82.2 416.0,82.2 419.7,82.2 423.5,82.2 427.2,82.2 430.9,82.2 434.7,82.2 438.4,82.2 442.1,82.2 445.9,82.2 449.6,82.2 453.3,82.2 457.1,82.2 460.8,82.2 464.5,82.2 468.3,82.2 472.0,82.2 475.7,82.2 479.5,82.2 483.2,82.2 486.9,82.2 490.7,82.2 494.4,82.2 498.1,82.2 501.9,82.2 505.6,82.2 509.3,82.2 513.1,82.2 516.8,82.2 520.5,82.2 524.3,82.2 528.0,82.2 531.7,87.6 535.5,93.1 539.2,98.5 542.9,104.0 546.7,109.5 550.4,114.9 554.1,120.4 557.9,125.8 561.6,131.3 565.3,136.7 569.1,142.2 572.8,147.6 576.5,153.1 580.3,158.5 584.0,164.0 587.7,169.5 591.5,174.9 595.2,180.4 598.9,185.8 602.7,191.3 606.4,196.7 610.1,202.2 613.9,207.6 617.6,213.1 621.3,218.5 625.1,224.0 628.8,229.5 632.5,234.9 636.3,240.4"/><rect class="s1" x="80" y="318" width="10" height="10" rx="2"/><text class="t-tick" x="96.0" y="327.0" text-anchor="start">warmup + cosine decay (used in ch4_pretrain.py, GPT-3, Llama 3)</text><rect class="s2" x="80" y="338" width="10" height="10" rx="2"/><text class="t-tick" x="96.0" y="347.0" text-anchor="start">warmup-stable-decay (MiniCPM): flat, then a short final decay (drawn for comparison)</text><text class="t-muted" x="125.3" y="173.1" text-anchor="start">&lt;- warmup: 100 steps</text></svg><figcaption>The learning rate of our run (blue): 100 warmup steps up to 1e-3, then a cosine curve down to 1e-4. The dashed orange line is a warmup-stable-decay schedule, which holds the peak and decays only at the end, drawn for comparison.</figcaption></figure>

The real Llama 3 recipe has the same shape at a vastly larger scale:

> [!PAPER] Grattafiori et al. (2024), The Llama 3 Herd of Models · Section 3.4.1 · page 14
> [![The Llama 3 pre-training recipe paragraph with highlights on peak learning rate of 8 x 10^-5, linear warm up of 8,000 steps, cosine learning rate schedule decaying to 8 x 10^-7, initial batch size of 4M tokens, and double the batch size again to 16M](/img/training/ch4-llama3-recipe.png)](/img/training/ch4-llama3-recipe.png)
>
> **Context:** the initial pre-training stage of the 405B model.
>
> **What it says:** AdamW with a peak learning rate of $$8 \times 10^{-5}$$, 8,000 warmup steps, and a cosine decay to $$8 \times 10^{-7}$$ over 1,200,000 steps. The batch starts at 4M tokens and doubles twice, to 16M (Section 4.9).
>
> **Why it matters:** the peak rate of a 405B model is more than ten times smaller than ours. Bigger models need smaller learning rates: each update moves many more weights at once.

A newer alternative is the **warmup-stable-decay** (WSD) schedule, used by MiniCPM and others: hold the peak rate for most of training and decay quickly in the last 10 to 20%. Its advantage is practical: you can stop the stable phase at any point, branch off, decay, and get a finished model, without fixing the run length in advance as a cosine schedule requires. Section 4.13 shows that the final decay is also where data quality matters most.

Two other settings in our training loop are standard and worth naming:

- **Weight decay** (0.1 in our run), applied only to weight matrices, not to biases and LayerNorm gains. It pulls weights slightly towards zero at every step, a mild regularizer.
- **Gradient clipping** at a norm of 1.0. If the gradient vector is longer than 1.0, it is scaled down to length 1.0 before the update. It protects against the occasional huge gradient from an unusual batch, one cause of **loss spikes**.

> [!DEFINITION] Loss spike
> A sudden jump in the training loss, sometimes recovering by itself and sometimes not. Large runs watch the loss and the gradient norm closely and, after a spike that does not recover, restart from an earlier checkpoint, sometimes skipping the batches that caused it.

## 4.9 Batch size

Each step averages the gradient over a batch. A bigger batch gives a more accurate gradient, which allows larger steps and fewer of them; it also keeps thousands of GPUs busy. But there are diminishing returns: past some size, doubling the batch no longer halves the number of steps needed, and the extra data per step is wasted.

> [!DEFINITION] Gradient noise and critical batch size
> The gradient from one batch is a noisy estimate of the "true" gradient over all the data. Its noise shrinks as the batch grows. The **critical batch size** is roughly where the noise stops being the main limit: below it, doubling the batch nearly halves the steps needed; above it, it barely helps. It grows as the loss falls, so later stages of training can use bigger batches.

`ch4_batch.py` makes the noise visible. It takes our trained tiny GPT, computes the gradient on two *independent* random batches of the same size, and measures how much the two gradients point in the same direction (their cosine similarity). If the gradient were noise-free, the cosine would be 1.

```python
def grad(B, g):
    m.zero_grad(set_to_none=True)
    x, y = batch(train, B, 256, g, dev)
    m(x, y)[1].backward()
    return torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]).clone()

for B in [1, 2, 4, 8, 16, 32, 64, 128]:
    cos = [torch.nn.functional.cosine_similarity(grad(B, g), grad(B, g), dim=0).item() for _ in range(6)]
```

`grad` draws a batch of `B` windows, runs the forward and backward pass, and flattens all 12.3 million gradient numbers into one long vector. For each batch size, the loop compares two such vectors from different batches, six times, and averages.

```text
batch of    1 windows (   256 tokens): cosine(grad1, grad2) = 0.007
batch of    8 windows ( 2,048 tokens): cosine(grad1, grad2) = 0.016
batch of   32 windows ( 8,192 tokens): cosine(grad1, grad2) = 0.069
batch of   64 windows (16,384 tokens): cosine(grad1, grad2) = 0.138
batch of  128 windows (32,768 tokens): cosine(grad1, grad2) = 0.245
```

<figure class="fig"><svg viewBox="0 0 760 324" role="img" aria-label="Cosine similarity between gradients from two independent batches grows with the batch size."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Bigger batches give gradients that agree more (trained tiny GPT, ch4_batch.py)</text><line class="grid" x1="80" y1="274.0" x2="600" y2="274.0"/><text class="t-tick" x="72.0" y="278.0" text-anchor="end">0.00</text><line class="grid" x1="80" y1="233.6" x2="600" y2="233.6"/><text class="t-tick" x="72.0" y="237.6" text-anchor="end">0.05</text><line class="grid" x1="80" y1="193.2" x2="600" y2="193.2"/><text class="t-tick" x="72.0" y="197.2" text-anchor="end">0.10</text><line class="grid" x1="80" y1="152.8" x2="600" y2="152.8"/><text class="t-tick" x="72.0" y="156.8" text-anchor="end">0.15</text><line class="grid" x1="80" y1="112.5" x2="600" y2="112.5"/><text class="t-tick" x="72.0" y="116.5" text-anchor="end">0.20</text><line class="grid" x1="80" y1="72.1" x2="600" y2="72.1"/><text class="t-tick" x="72.0" y="76.1" text-anchor="end">0.25</text><text class="t-tick" x="104.2" y="292.0" text-anchor="middle">256</text><text class="t-tick" x="240.3" y="292.0" text-anchor="middle">1,024</text><text class="t-tick" x="376.3" y="292.0" text-anchor="middle">4,096</text><text class="t-tick" x="512.4" y="292.0" text-anchor="middle">16,384</text><text class="t-tick" x="580.4" y="292.0" text-anchor="middle">32,768</text><line class="axis" x1="80" y1="274" x2="600" y2="274"/><polyline class="l1" points="104.2,268.3 172.3,270.3 240.3,265.3 308.3,261.3 376.3,236.9 444.4,218.2 512.4,162.7 580.4,75.8"/><g class="mark"><title>cosine: 256, 0.01</title><circle class="s1 ring" cx="104.2" cy="268.3" r="4"/></g><g class="mark"><title>cosine: 512, 0.00</title><circle class="s1 ring" cx="172.3" cy="270.3" r="4"/></g><g class="mark"><title>cosine: 1,024, 0.01</title><circle class="s1 ring" cx="240.3" cy="265.3" r="4"/></g><g class="mark"><title>cosine: 2,048, 0.02</title><circle class="s1 ring" cx="308.3" cy="261.3" r="4"/></g><g class="mark"><title>cosine: 4,096, 0.05</title><circle class="s1 ring" cx="376.3" cy="236.9" r="4"/></g><g class="mark"><title>cosine: 8,192, 0.07</title><circle class="s1 ring" cx="444.4" cy="218.2" r="4"/></g><g class="mark"><title>cosine: 16,384, 0.14</title><circle class="s1 ring" cx="512.4" cy="162.7" r="4"/></g><g class="mark"><title>cosine: 32,768, 0.25</title><circle class="s1 ring" cx="580.4" cy="75.8" r="4"/></g><text class="t-muted" x="340.0" y="312.0" text-anchor="middle">tokens per batch (log scale)</text><text class="t-muted" x="72.0" y="48.0" text-anchor="start">cosine similarity of two independent batch gradients</text><line class="edge-dim" x1="444.4" y1="64.0" x2="444.4" y2="274.0"/><text class="t-muted" x="438.4" y="80.0" text-anchor="end">our training batch</text><text class="t-val" x="444.4" y="208.2" text-anchor="middle">0.069</text><text class="t-val" x="512.4" y="152.7" text-anchor="middle">0.138</text><text class="t-val" x="580.4" y="65.8" text-anchor="middle">0.245</text></svg><figcaption>How much two gradients from independent batches agree, for the trained tiny GPT, as the batch grows. With one window they are almost unrelated; doubling the batch roughly doubles the agreement, until it starts to flatten.</figcaption></figure>

At the end of training, the gradient from one 256-token window is almost pure noise (cosine 0.007). Even our training batch of 8,192 tokens gives gradients that agree only weakly (0.069). Every doubling of the batch roughly doubles the agreement, which is what you expect when noise dominates. This is why real runs use batches of millions of tokens, and why they *increase* the batch during training: Llama 3 405B started at 4M tokens per batch, doubled to 8M after 252M tokens and to 16M after 2.87T tokens. Early on, when the loss is high, the true gradient is large compared with the noise and a small batch is enough; later the true gradient shrinks, and more tokens are needed to see it.

## 4.10 Mixed precision: fewer bits per number

Every weight, gradient and activation is a floating-point number, and the format decides how much memory it takes and how fast the hardware can process it.

> [!DEFINITION] Floating-point format
> A way to store a real number in bits: one **sign** bit, some **exponent** bits (which set the range, from tiny to huge) and some **mantissa** bits (which set the precision, how many significant digits). fp32 uses 32 bits; fp16 and bf16 use 16.

`ch4_precision.py` asks PyTorch about the three formats and tries a few numbers:

```text
format bits exponent mantissa    largest  smallest normal  step after 1.0
fp32     32        8       23    3.4e+38         1.18e-38        1.19e-07
fp16     16        5       10   6.55e+04          6.1e-05        0.000977
bf16     16        8        7   3.39e+38         1.18e-38        0.00781

    3.14159265 ->  fp32: 3.1415927   fp16: 3.140625   bf16: 3.140625
         1e-08 ->  fp32: 9.9999999e-09   fp16: 0   bf16: 1.0011718e-08
       70000.0 ->  fp32: 70000   fp16: inf   bf16: 70144
         1.001 ->  fp32: 1.001   fp16: 1.0009766   bf16: 1

fp32: start at 1.0, add 1e-4 a thousand times -> 1.100017  (exact answer 1.1)
bf16: start at 1.0, add 1e-4 a thousand times -> 1.000000  (exact answer 1.1)
```

<figure class="fig"><svg viewBox="0 0 820 256" role="img" aria-label="Bit layouts of fp32, fp16 and bf16 with their range and precision."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Three ways to store a number in bits: sign, exponent (range), mantissa (precision)</text><text class="t-note" x="70.0" y="66.0" text-anchor="end">fp32</text><rect class="box-4" x="80.0" y="50.0" width="14.0" height="24.0" rx="3"/><text class="t-tick" x="87.0" y="66.0" text-anchor="middle">1</text><rect class="box-1" x="96.0" y="50.0" width="112.0" height="24.0" rx="3"/><text class="t-tick" x="152.0" y="66.0" text-anchor="middle">exponent 8</text><rect class="box-2" x="210.0" y="50.0" width="322.0" height="24.0" rx="3"/><text class="t-tick" x="371.0" y="66.0" text-anchor="middle">mantissa 23</text><text class="t-tick" x="546.0" y="60.0" text-anchor="start">max 3.4e+38</text><text class="t-muted" x="546.0" y="76.0" text-anchor="start">step after 1.0: 1.19e-07</text><text class="t-note" x="70.0" y="116.0" text-anchor="end">fp16</text><rect class="box-4" x="80.0" y="100.0" width="14.0" height="24.0" rx="3"/><text class="t-tick" x="87.0" y="116.0" text-anchor="middle">1</text><rect class="box-1" x="96.0" y="100.0" width="70.0" height="24.0" rx="3"/><text class="t-tick" x="131.0" y="116.0" text-anchor="middle">exponent 5</text><rect class="box-2" x="168.0" y="100.0" width="140.0" height="24.0" rx="3"/><text class="t-tick" x="238.0" y="116.0" text-anchor="middle">mantissa 10</text><text class="t-tick" x="322.0" y="110.0" text-anchor="start">max 6.55e+04</text><text class="t-muted" x="322.0" y="126.0" text-anchor="start">step after 1.0: 0.000977</text><text class="t-note" x="70.0" y="166.0" text-anchor="end">bf16</text><rect class="box-4" x="80.0" y="150.0" width="14.0" height="24.0" rx="3"/><text class="t-tick" x="87.0" y="166.0" text-anchor="middle">1</text><rect class="box-1" x="96.0" y="150.0" width="112.0" height="24.0" rx="3"/><text class="t-tick" x="152.0" y="166.0" text-anchor="middle">exponent 8</text><rect class="box-2" x="210.0" y="150.0" width="98.0" height="24.0" rx="3"/><text class="t-tick" x="259.0" y="166.0" text-anchor="middle">mantissa 7</text><text class="t-tick" x="322.0" y="160.0" text-anchor="start">max 3.39e+38</text><text class="t-muted" x="322.0" y="176.0" text-anchor="start">step after 1.0: 0.00781</text><text class="t-tick" x="20.0" y="202.0" text-anchor="start">fp16 has precision but little range: 70,000 overflows to inf and 1e-8 rounds to 0.</text><text class="t-tick" x="20.0" y="219.0" text-anchor="start">bf16 keeps fp32's range with less precision: 1 + 0.001 rounds to exactly 1.</text><text class="t-tick" x="20.0" y="236.0" text-anchor="start">So adding a 1e-4 update to a weight of 1.0 a thousand times gives 1.0000 in bf16 and 1.1000 in fp32.</text></svg><figcaption>Bit layouts of the three formats. bf16 keeps fp32's 8 exponent bits (the same range) and gives up mantissa bits (precision); fp16 keeps more precision but has a much smaller range.</figcaption></figure>

Read the table as two different trade-offs:

- **fp16** has a decent mantissa but only 5 exponent bits. Its largest value is 65,504, so 70,000 becomes infinity; and small gradients like $$10^{-8}$$ become 0. Training in fp16 needs **loss scaling** (multiply the loss by a large factor so gradients stay in range, then divide back).
- **bf16** ("brain float") keeps fp32's 8-bit exponent, so it has the same huge range and never needs loss scaling, but with only 7 mantissa bits it has about 2 to 3 significant digits. The step from 1.0 to the next number is 0.0078, so $$1 + 0.001$$ rounds to exactly 1.

The last two lines show why that matters for training. A weight of 1.0 receiving an update of $$10^{-4}$$ a thousand times should end at 1.1. In fp32 it does. In bf16 *every single update is rounded away* and the weight never moves. Typical updates are tiny compared with the weights, so storing the weights themselves in bf16 would stall learning.

The solution is **mixed-precision training**, introduced by Micikevicius et al. in 2017:

> [!PAPER] Micikevicius et al. (2018), Mixed Precision Training · Figure 1 · page 3
> [![Diagram of one mixed-precision iteration: fp32 master weights are converted to half precision for the forward pass, the backward pass for activations and the backward pass for weights, all in F16; the weight gradient flows into an fp32 weight update that writes back the updated master weights](/img/training/ch4-mixed-precision-fig1.png)](/img/training/ch4-mixed-precision-fig1.png)
>
> **Context:** the method section of the paper.
>
> **What it says:** keep a "master" copy of the weights in fp32. At each step, convert them to 16 bits, run the forward and backward passes in 16 bits (fast), then apply the update to the fp32 master copy (precise).
>
> **Why it matters:** the slow, memory-hungry part (the matrix multiplications over activations) runs in 16 bits, and the part that needs precision (adding tiny updates to weights) stays in 32 bits. Every large run since uses some version of this, today with bf16 instead of fp16.

In PyTorch this is one context manager. Our training loop wraps the forward pass in `torch.autocast('mps', dtype=torch.bfloat16)`, which runs the matrix multiplications in bf16 while the weights and the optimizer stay in fp32. On the laptop GPU, `ch4_prep.py` measured 367 ms per step in fp32 and 219 ms in bf16 autocast: 1.68 times faster, with the same results.

### What training costs in memory

Mixed precision also explains a number every practitioner knows: training with AdamW needs about **16 bytes per parameter**, before counting activations.

$$\underbrace{2}_{\text{bf16 weights}} + \underbrace{2}_{\text{bf16 gradients}} + \underbrace{4}_{\text{fp32 master}} + \underbrace{4}_{\text{Adam } m} + \underbrace{4}_{\text{Adam } v} = 16 \text{ bytes}$$

where $$m$$ and $$v$$ are AdamW's running averages of the gradient and of the squared gradient, one of each for every parameter, kept in fp32.

**Worked example** (`ch4_compute.py`): Qwen2.5-0.5B needs $$4.94 \times 10^8 \times 16 = 7.36$$ GiB just for this state; Llama 3 8B needs 119 GiB, more than one 80 GB GPU holds; Llama 3 405B needs about 6,035 GiB. That is why large runs split the weights, gradients and optimizer state across many GPUs, and why Chapter 5 will reach for LoRA, which trains only a tiny fraction of the parameters.

## 4.11 Hands-on: pretrain a tiny GPT on a laptop

Now all the pieces come together. The goal: pretrain a GPT from random weights on a laptop's Apple GPU in a few minutes, and watch it learn to write.

### The data: TinyStories

A 12M-parameter model trained on web text for a few minutes would produce word salad, because the web is far too diverse for it. **TinyStories** (Eldan and Li, 2023) is a dataset designed for exactly our situation:

> [!PAPER] Eldan and Li (2023), TinyStories · Abstract · page 1
> [![The TinyStories abstract with the highlighted phrase: a synthetic dataset of short stories that only contain words that a typical 3 to 4-year-olds usually understand](/img/training/ch4-tinystories-abstract.png)](/img/training/ch4-tinystories-abstract.png)
>
> **Context:** the abstract.
>
> **What it says:** short stories generated by GPT-3.5 and GPT-4 using only words a 3 to 4-year-old understands. Models "below 10 million total parameters" trained on it produce fluent, consistent stories with almost perfect grammar.
>
> **Why it matters:** by shrinking the *world* the model has to learn, TinyStories lets a tiny model show the same stages of learning a large model goes through on the web.

The script uses the first 60 MB of the TinyStories (V2) training file (73,000 stories) for training and the official validation file for evaluation. `ch4_prep.py` trains the 4,096-token BPE tokenizer of Section 4.3 on the training text and encodes both files:

```text
vocab size: 4096
train tokens: 14,880,999   val tokens: 5,580,195   train characters: 59,975,054   chars/token: 4.03
example: Once | Ġupon | Ġa | Ġtime | , | Ġa | Ġlittle | Ġgirl | Ġnamed | ĠLily | Ġfound | Ġa | Ġshiny | Ġred | Ġball | .
```

The `Ġ` character is how byte-level BPE writes a leading space: " upon" is one token, different from "upon" at the start of a line.

### The model and the plan

The model is the GPT of Section 4.5: 6 blocks, width 384, 6 heads, context of 256 tokens, 12,318,720 parameters. The plan: 1,500 steps of 32 windows of 256 tokens, which is 12.3M tokens, a bit less than one pass over the training text. Every token is new to the model, as in real pretraining, where data is rarely repeated.

### The training loop

Here is the core of `ch4_pretrain.py`, slightly simplified:

```python
B, T = 32, 256                      # 32 windows of 256 tokens = 8,192 tokens per step
STEPS, WARMUP, PEAK = 1500, 100, 1e-3

tok = get_tokenizer(4096)
train, val = get_tokens('train', tok), get_tokens('val', tok)
torch.manual_seed(0)
model = GPT(tok.get_vocab_size(), d=384, n_layer=6, n_head=6, ctx=T).to('mps')
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]       # weight matrices
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]     # biases, LayerNorm gains
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1},
                         {'params': no_decay, 'weight_decay': 0.0}], lr=PEAK, betas=(0.9, 0.95))

for step in range(STEPS):
    lr = lr_at(step, PEAK, WARMUP, STEPS)
    for gr in opt.param_groups:
        gr['lr'] = lr
    x, y = batch(train, B, T, g, 'mps')
    with torch.autocast('mps', dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
```

Block by block:

- **Setup.** `B, T` fix the batch shape; `STEPS, WARMUP, PEAK` fix the schedule of Section 4.8. `get_tokenizer` and `get_tokens` load the tokenizer and the two token arrays (they are built once and cached). `torch.manual_seed(0)` makes the random initial weights reproducible. The model is created and moved to the Apple GPU (`'mps'`).
- **Optimizer.** The parameters are split into two groups: matrices (2 or more dimensions) get weight decay 0.1, vectors (biases, LayerNorm gains) get none, because shrinking them towards zero has no useful regularizing effect. `betas=(0.9, 0.95)` are AdamW's averaging rates for the gradient and the squared gradient; 0.95 instead of the default 0.999 makes the second average react faster, a common choice for large-model pretraining (GPT-3 and LLaMA use it).
- **Learning rate.** At each step, `lr_at` computes the scheduled rate and writes it into every parameter group.
- **Forward.** `batch` draws 32 random windows and their shifted targets. Inside `autocast`, the forward pass runs in bf16 and returns the mean cross-entropy over all 8,192 positions.
- **Backward and update.** `loss.backward()` fills every parameter's `.grad`. `clip_grad_norm_` measures the total gradient length, rescales it to at most 1.0 and returns the length *before* clipping, which we log. `opt.step()` applies AdamW to the fp32 weights, and `zero_grad` clears the gradients for the next step.

The full script also evaluates on 40 fixed validation batches every 100 steps and generates a sample from the prompt "Once upon a time" at chosen steps, always with the same random seed, so the samples differ only because the weights do.

### The run

[![Terminal output of ch4_pretrain.py: the model size and data, then every 100 steps a validation loss and a training loss, learning rate and gradient norm, with text samples at steps 0, 50, 150, 300, 600, 1000 and 1500, ending with the total time](/img/training/ch4-pretrain-run.png)](/img/training/ch4-pretrain-run.png)

<figure class="fig"><svg viewBox="0 0 780 374" role="img" aria-label="Training and validation loss of the tiny GPT over 1,500 steps, starting near ln(4096) and ending near 2.07."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Pretraining the 12M-parameter GPT on TinyStories: 8.2 minutes on the laptop GPU</text><line class="grid" x1="80" y1="316.4" x2="640" y2="316.4"/><text class="t-tick" x="72.0" y="320.4" text-anchor="end">2</text><line class="grid" x1="80" y1="278.1" x2="640" y2="278.1"/><text class="t-tick" x="72.0" y="282.1" text-anchor="end">3</text><line class="grid" x1="80" y1="239.9" x2="640" y2="239.9"/><text class="t-tick" x="72.0" y="243.9" text-anchor="end">4</text><line class="grid" x1="80" y1="201.6" x2="640" y2="201.6"/><text class="t-tick" x="72.0" y="205.6" text-anchor="end">5</text><line class="grid" x1="80" y1="163.4" x2="640" y2="163.4"/><text class="t-tick" x="72.0" y="167.4" text-anchor="end">6</text><line class="grid" x1="80" y1="125.2" x2="640" y2="125.2"/><text class="t-tick" x="72.0" y="129.2" text-anchor="end">7</text><line class="grid" x1="80" y1="86.9" x2="640" y2="86.9"/><text class="t-tick" x="72.0" y="90.9" text-anchor="end">8</text><text class="t-tick" x="80.0" y="342.0" text-anchor="middle">0</text><text class="t-tick" x="192.0" y="342.0" text-anchor="middle">300</text><text class="t-tick" x="304.0" y="342.0" text-anchor="middle">600</text><text class="t-tick" x="416.0" y="342.0" text-anchor="middle">900</text><text class="t-tick" x="528.0" y="342.0" text-anchor="middle">1,200</text><text class="t-tick" x="640.0" y="342.0" text-anchor="middle">1,500</text><line class="axis" x1="80" y1="324" x2="640" y2="324"/><polyline class="l1" points="80.0,72.6 83.7,109.1 87.5,131.3 91.2,161.3 94.9,181.7 98.7,203.8 102.4,211.1 106.1,220.3 109.9,231.5 113.6,233.2 117.3,237.7 121.1,242.8 124.8,243.1 128.5,249.6 132.3,251.5 136.0,254.5 139.7,256.5 143.5,258.9 147.2,256.2 150.9,261.5 154.7,264.4 158.4,264.2 162.1,264.8 165.9,267.5 169.6,265.1 173.3,272.9 177.1,270.2 180.8,272.7 184.5,279.4 188.3,276.9 192.0,274.9 195.7,277.6 199.5,274.8 203.2,284.9 206.9,284.2 210.7,282.4 214.4,288.4 218.1,283.6 221.9,283.3 225.6,281.9 229.3,283.7 233.1,285.9 236.8,288.6 240.5,291.1 244.3,288.1 248.0,291.2 251.7,290.3 255.5,291.1 259.2,292.1 262.9,289.1 266.7,287.5 270.4,289.3 274.1,292.7 277.9,291.1 281.6,292.4 285.3,295.1 289.1,292.0 292.8,296.3 296.5,295.8 300.3,297.0 304.0,294.0 307.7,299.0 311.5,296.4 315.2,298.6 318.9,301.6 322.7,299.9 326.4,295.7 330.1,301.1 333.9,299.7 337.6,301.7 341.3,301.8 345.1,302.8 348.8,307.2 352.5,297.3 356.3,302.1 360.0,297.9 363.7,302.4 367.5,297.0 371.2,303.2 374.9,301.1 378.7,302.0 382.4,305.8 386.1,304.5 389.9,303.6 393.6,301.3 397.3,309.2 401.1,298.7 404.8,305.2 408.5,303.0 412.3,303.6 416.0,304.4 419.7,302.9 423.5,303.4 427.2,306.0 430.9,303.9 434.7,310.2 438.4,310.2 442.1,305.7 445.9,309.0 449.6,303.6 453.3,305.7 457.1,309.9 460.8,311.7 464.5,307.5 468.3,308.9 472.0,311.4 475.7,310.4 479.5,312.1 483.2,308.3 486.9,309.8 490.7,309.4 494.4,312.0 498.1,312.7 501.9,310.7 505.6,312.5 509.3,312.9 513.1,309.5 516.8,312.8 520.5,312.4 524.3,314.4 528.0,312.2 531.7,310.0 535.5,312.3 539.2,309.7 542.9,310.0 546.7,313.0 550.4,314.6 554.1,314.6 557.9,314.3 561.6,320.1 565.3,313.2 569.1,316.0 572.8,312.1 576.5,311.9 580.3,312.5 584.0,315.7 587.7,316.2 591.5,312.3 595.2,317.8 598.9,313.0 602.7,313.0 606.4,311.2 610.1,311.5 613.9,313.3 617.6,311.0 621.3,312.5 625.1,314.0 628.8,310.3 632.5,314.2 636.3,315.8"/><polyline class="l2" points="80.0,72.6 117.3,238.7 154.7,262.7 192.0,276.5 229.3,285.0 266.7,290.8 304.0,295.2 341.3,299.0 378.7,302.4 416.0,305.0 453.3,307.4 490.7,309.4 528.0,310.9 565.3,312.1 602.7,312.9 640.0,313.6"/><g class="mark"><title>validation loss: 0, 8.37415</title><circle class="s2 ring" cx="80.0" cy="72.6" r="4"/></g><g class="mark"><title>validation loss: 100, 4.03141</title><circle class="s2 ring" cx="117.3" cy="238.7" r="4"/></g><g class="mark"><title>validation loss: 200, 3.40212</title><circle class="s2 ring" cx="154.7" cy="262.7" r="4"/></g><g class="mark"><title>validation loss: 300, 3.04345</title><circle class="s2 ring" cx="192.0" cy="276.5" r="4"/></g><g class="mark"><title>validation loss: 400, 2.81932</title><circle class="s2 ring" cx="229.3" cy="285.0" r="4"/></g><g class="mark"><title>validation loss: 500, 2.66872</title><circle class="s2 ring" cx="266.7" cy="290.8" r="4"/></g><g class="mark"><title>validation loss: 600, 2.55286</title><circle class="s2 ring" cx="304.0" cy="295.2" r="4"/></g><g class="mark"><title>validation loss: 700, 2.45304</title><circle class="s2 ring" cx="341.3" cy="299.0" r="4"/></g><g class="mark"><title>validation loss: 800, 2.36491</title><circle class="s2 ring" cx="378.7" cy="302.4" r="4"/></g><g class="mark"><title>validation loss: 900, 2.29608</title><circle class="s2 ring" cx="416.0" cy="305.0" r="4"/></g><g class="mark"><title>validation loss: 1,000, 2.23472</title><circle class="s2 ring" cx="453.3" cy="307.4" r="4"/></g><g class="mark"><title>validation loss: 1,100, 2.18097</title><circle class="s2 ring" cx="490.7" cy="309.4" r="4"/></g><g class="mark"><title>validation loss: 1,200, 2.14154</title><circle class="s2 ring" cx="528.0" cy="310.9" r="4"/></g><g class="mark"><title>validation loss: 1,300, 2.1121</title><circle class="s2 ring" cx="565.3" cy="312.1" r="4"/></g><g class="mark"><title>validation loss: 1,400, 2.08925</title><circle class="s2 ring" cx="602.7" cy="312.9" r="4"/></g><g class="mark"><title>validation loss: 1,500, 2.07325</title><circle class="s2 ring" cx="640.0" cy="313.6" r="4"/></g><rect class="s2" x="652.0" y="308.6" width="10" height="10" rx="2"/><text class="t-tick" x="667.0" y="317.6" text-anchor="start">validation loss</text><rect class="s1" x="648.3" y="324.6" width="10" height="10" rx="2"/><text class="t-tick" x="663.3" y="333.6" text-anchor="start">train loss</text><text class="t-muted" x="360.0" y="362.0" text-anchor="middle">training step (each step = 8,192 tokens; 1,500 steps = 12.3M tokens)</text><text class="t-muted" x="72.0" y="48.0" text-anchor="start">cross-entropy loss (nats per token)</text><line class="base-line" x1="80" y1="74.8" x2="640" y2="74.8"/><text class="t-muted" x="300.0" y="68.8" text-anchor="start">ln(4096) = 8.32: a uniform guess over the vocabulary</text><text class="t-val" x="636.0" y="301.6" text-anchor="end">2.07</text></svg><figcaption>Training loss (blue, every 10 steps) and validation loss (orange, every 100 steps) of the tiny GPT. The loss starts at the uniform-guess value ln(4096) = 8.32, falls below 4 within 100 steps, and ends at 2.07 on the validation set.</figcaption></figure>

The curve has the shape of every pretraining curve:

1. **A cliff in the first 100 steps** (8.37 to 4.03). The model learns the cheapest lessons first: which tokens are common ("the", ".", " a") and which are never used. Just knowing the frequency of each token takes the loss from 8.3 to roughly the entropy of single tokens.
2. **A long, bending slope** (4.0 to 2.5 by step 600). The model learns word order, short phrases, then grammar.
3. **A slow tail** (2.5 to 2.07). Each further gain costs more steps, as the power laws of Section 4.7 predict. The decaying learning rate helps squeeze out the last part.

The training and validation curves lie on top of each other. That is expected when every token is seen once: the model has never seen the validation stories, but it has never seen most training stories more than once either, so there is nothing to overfit. In perplexity terms (Chapter 1), the final validation loss of 2.073 is $$e^{2.073} = 7.95$$: at each position the model is, on average, as unsure as if it were choosing uniformly among about 8 tokens, down from 4,096.

The samples tell the same story in words:

<figure class="fig"><svg viewBox="0 0 780 511" role="img" aria-label="Five text samples from the tiny GPT at steps 0, 50, 150, 600 and 1,500, from random tokens to simple stories."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">The same prompt, "Once upon a time", at five checkpoints (temperature 0.8, same random seed)</text><rect class="box-ghost" x="20.0" y="42.0" width="740.0" height="85.0" rx="12"/><circle class="node on" cx="38.0" cy="60.0" r="11"/><text class="t-note" x="38.0" y="64.5" text-anchor="middle">1</text><text class="t-title" x="56.0" y="65.0" text-anchor="start">step 0   (validation loss 8.37)</text><text class="t-tick" x="40.0" y="86.0" text-anchor="start">Once upon a time word tid spaghettiudden squ fairy slidistilly ele mean slid steak cloud cloud</text><text class="t-tick" x="40.0" y="103.0" text-anchor="start">goat caterpill grumpy clay Olive hall eleist Rex pat vo caterpill beachstic climb junk have ele</text><text class="t-tick" x="40.0" y="120.0" text-anchor="start">curtain polish rugstic slid listened?lf listened slid straw</text><rect class="box" x="20.0" y="135.0" width="740.0" height="85.0" rx="12"/><circle class="node on" cx="38.0" cy="153.0" r="11"/><text class="t-note" x="38.0" y="157.5" text-anchor="middle">2</text><text class="t-title" x="56.0" y="158.0" text-anchor="start">step 50</text><text class="t-tick" x="40.0" y="179.0" text-anchor="start">Once upon a time, very, Tim was to see his day he was a mom's not day, but he said, " day The big,</text><text class="t-tick" x="40.0" y="196.0" text-anchor="start">" day.  The and the new the tree, little fun and the toys. She was not play not tree. The, the</text><text class="t-tick" x="40.0" y="213.0" text-anchor="start">little dog.</text><rect class="box" x="20.0" y="228.0" width="740.0" height="85.0" rx="12"/><circle class="node on" cx="38.0" cy="246.0" r="11"/><text class="t-note" x="38.0" y="250.5" text-anchor="middle">3</text><text class="t-title" x="56.0" y="251.0" text-anchor="start">step 150</text><text class="t-tick" x="40.0" y="272.0" text-anchor="start">Once upon a time, there was a small blue little boy named Tim. Tim went to play with his red car</text><text class="t-tick" x="40.0" y="289.0" text-anchor="start">in the box. One day, Tim said, "Thank you and you?" They were playing, "Mom, Tim." Tim was a big,</text><text class="t-tick" x="40.0" y="306.0" text-anchor="start">Tim saw the park, "What are very sad</text><rect class="box" x="20.0" y="321.0" width="740.0" height="85.0" rx="12"/><circle class="node on" cx="38.0" cy="339.0" r="11"/><text class="t-note" x="38.0" y="343.5" text-anchor="middle">4</text><text class="t-title" x="56.0" y="344.0" text-anchor="start">step 600   (validation loss 2.55)</text><text class="t-tick" x="40.0" y="365.0" text-anchor="start">Once upon a time, there was a humble dog named Max. Max was a very small girl who loved to play in</text><text class="t-tick" x="40.0" y="382.0" text-anchor="start">the yard. One sunny day, Max was playing in the park. He saw a little girl named Sue. Tim was also</text><text class="t-tick" x="40.0" y="399.0" text-anchor="start">sad because she was a very cold, but she knew it was</text><rect class="box-1" x="20.0" y="414.0" width="740.0" height="85.0" rx="12"/><circle class="node on" cx="38.0" cy="432.0" r="11"/><text class="t-note" x="38.0" y="436.5" text-anchor="middle">5</text><text class="t-title" x="56.0" y="437.0" text-anchor="start">step 1,500   (validation loss 2.07)</text><text class="t-tick" x="40.0" y="458.0" text-anchor="start">Once upon a time, there was a girl named Lily. She liked to create things with her toys. One day,</text><text class="t-tick" x="40.0" y="475.0" text-anchor="start">she found a big, shiny thing in her room. Lily and her mom went to the store to buy her toys. They</text><text class="t-tick" x="40.0" y="492.0" text-anchor="start">saw a big, colorful toy car. Lily was so happy. She</text></svg><figcaption>The same prompt, sampled with the same random seed at five checkpoints. At step 0 the output is random tokens; at step 50 it is English-looking word soup; at step 150 the sentences have a shape; at step 600 the story is grammatical but forgets who is who; at step 1,500 it is a short, consistent story.</figcaption></figure>

At step 600, "there was a humble dog named Max. Max was a very small girl" shows a model that has learned grammar and story templates but not yet to keep track of a character. At step 1,500, "there was a girl named Lily. She liked to create things with her toys. One day, she found a big, shiny thing in her room" keeps the same character and the right pronoun across sentences. Nobody told the model what a pronoun is: it is just the cheapest way to predict the next token in millions of stories.

> [!TIP]
> To run this yourself: `python ch4_prep.py` (downloads nothing; it expects the TinyStories text in `~/.cache/tinystories`, which you can get from the dataset page on the Hugging Face Hub), then `python ch4_pretrain.py`. On a recent laptop GPU it takes 5 to 10 minutes. On a CPU it will take roughly ten times longer; cut `STEPS` to 300 to see the first stages.

## 4.12 What a base model learns

Our tiny model learned a tiny world. A real base model, trained on trillions of tokens of web, books and code, learns a great deal more. `ch4_probe.py` probes Qwen2.5-0.5B, the base model of Chapter 1, without any fine-tuning, in three ways.

### Facts

The simplest probe gives the model the start of a factual sentence and looks at its five most likely next tokens:

```text
'The capital of France is'                    ' Paris' 0.32  ' ______' 0.11  ' ____' 0.06  ' __' 0.06  ':\n' 0.05
'The chemical symbol for gold is'             ' ____' 0.37  ' __' 0.20  ' ______' 0.07  ' Au' 0.07  '\n' 0.04
'Romeo and Juliet was written by'             ' William' 0.33  ' Shakespeare' 0.17  ' the' 0.06  ' which' 0.05  ' a' 0.04
'The largest planet in the solar system is'   ' Jupiter' 0.45  ' the' 0.05  ' ' 0.04  ' Mercury' 0.03  ' __' 0.03
```

The facts are there: Paris, William (Shakespeare), Jupiter, Au. But look at the competitors. For "The chemical symbol for gold is", the most likely continuation is a blank, `____`, with probability 0.37, and the right answer gets only 0.07. The model is not "trying to answer"; it is predicting what comes next *in the kind of document that contains this sentence*, and on the web a sentence like that most often appears in a fill-in-the-blank worksheet. A base model's knowledge and its behaviour are separate things. Fine-tuning (Chapter 5) changes the behaviour; it adds very little knowledge.

### Grammar

The second probe compares the log-probability of a right and a wrong continuation:

```text
'The keys to the cabinet'            ' are'    -1.99   ' is'     -5.41   prefers the right one: True  (ratio 30.4x)
'The author of the books'            ' is'     -7.75   ' are'    -9.16   prefers the right one: True  (ratio 4.1x)
'Yesterday she'                      ' went'   -2.59   ' goes'   -9.33   prefers the right one: True  (ratio 841.8x)
'The children who live next door'    ' are'    -2.55   ' is'     -8.75   prefers the right one: True  (ratio 495.3x)
'I have never'                       ' seen'   -2.22   ' saw'    -8.01   prefers the right one: True  (ratio 324.8x)
'Each of the students'               ' has'    -5.22   ' have'   -9.69   prefers the right one: True  (ratio 86.6x)
```

All six are right, including the classic traps: "The keys to the cabinet" has a singular noun ("cabinet") right before the verb, but the subject is "keys", and the model prefers "are" 30 to 1. "Yesterday she" prefers the past tense "went" over "goes" by more than 800 to 1. The ratio is $$e^{\log p_{\text{right}} - \log p_{\text{wrong}}}$$; for the first line, $$e^{-1.99 + 5.41} = e^{3.42} \approx 30$$.

Our tiny TinyStories model learned the same kind of thing inside its small world:

```text
'Lily and her mom went to the'                   ' park' 0.37  ' store' 0.16  ' kitchen' 0.07  ' beach' 0.02  ' shop' 0.02
'Tom was very sad because he lost his'           ' toy' 0.35  ' ball' 0.10  ' favorite' 0.05  ' friend' 0.04  ' red' 0.02
'The little girl smiled because she was very'    ' happy' 0.26  ' brave' 0.06  ' kind' 0.04  ' excited' 0.03  ' proud' 0.03
```

Every top-5 token is a noun after "the" or "his", and an adjective after "very"; and the meanings fit (sad because he lost his *toy*, smiled because she was *happy*).

### In-context learning

The most surprising thing pretraining produces was described in the GPT-3 paper: a large base model can learn a new task *from examples in its prompt*, with no change to its weights.

> [!DEFINITION] In-context learning
> Doing a task from a description and a few solved examples placed in the prompt ("few-shot"), with no gradient update. **Zero-shot** means only a description, **one-shot** one example, **few-shot** several.

> [!PAPER] Brown et al. (2020), Language Models are Few-Shot Learners · Figure 2.1 · page 7
> [![Figure 2.1 of the GPT-3 paper: four panels comparing zero-shot (a task description and a prompt), one-shot (one example), few-shot (a few examples) for English to French translation, with traditional fine-tuning (gradient updates on many examples) shown separately](/img/training/ch4-gpt3-fig2-1.png)](/img/training/ch4-gpt3-fig2-1.png)
>
> **Context:** the section that defines the evaluation settings of the paper.
>
> **What it says:** in zero-, one- and few-shot learning, the task is given *in the prompt*: a description such as "Translate English to French:", then zero, one or a few examples like "sea otter => loutre de mer", then the new input. No weights change. Fine-tuning, by contrast, updates the weights on many examples.
>
> **Why it matters:** this is how base models were used before instruction tuning existed, and it is still how you test what pretraining alone put into a model.

> [!PAPER] Brown et al. (2020), Language Models are Few-Shot Learners · Figure 1.2 · page 4
> [![Accuracy against the number of examples in context for a word-unscrambling task, for 1.3B, 13B and 175B models, with and without a natural-language prompt; the 175B model climbs steeply with more examples while the small model stays low](/img/training/ch4-gpt3-fig1-2.png)](/img/training/ch4-gpt3-fig1-2.png)
>
> **Context:** an early figure that summarizes the paper's main finding on a simple task: removing random symbols from a word.
>
> **What it says:** "Larger models make increasingly efficient use of in-context information." The 175B model goes from under 10% with no examples to over 60% with many; the 1.3B model stays below 5%.
>
> **Why it matters:** in-context learning is something models *get better at* with scale, a capability that appears from the next-token objective alone.

`ch4_probe.py` runs a small version of this experiment on Qwen2.5-0.5B with three tasks, 28 test items each, and 0, 1, 2, 4 or 8 examples in the prompt:

```python
TASKS = {'English to French': (fr, ' ->', 'Translate English to French:\n'),
         'antonyms': (ant, ' ->', 'Write the opposite of each word:\n'),
         'sentiment with made-up labels (blue/red)': (senti, ' :', 'Label each review:\n')}
for name, (data, sep, head) in TASKS.items():
    pool, test = data[:8], data[8:]
    for k in [0, 1, 2, 4, 8]:
        shots = head + ''.join(f'{x}{sep} {y}\n' for x, y in pool[:k])
        correct = sum(answer(shots + f'{x}{sep}').lower().startswith(y) for x, y in test)
```

Each task has a one-line description, a separator, and a list of `(input, answer)` pairs. The first 8 pairs are the pool of examples, the rest are test items. For each `k`, the prompt is the description, `k` solved examples on separate lines, and then the new input followed by the separator; `answer` generates up to 6 tokens greedily and keeps the first line. The third task is the interesting one: reviews are labelled **blue** (positive) or **red** (negative). Those labels mean nothing; the model can only get them right by reading the examples.

```text
English to French                          k=0: 43%  k=1: 86%  k=2: 82%  k=4: 86%  k=8: 82%   (28 test items)
antonyms                                   k=0: 36%  k=1: 93%  k=2: 89%  k=4: 82%  k=8: 89%   (28 test items)
sentiment with made-up labels (blue/red)   k=0: 0%  k=1: 54%  k=2: 4%  k=4: 100%  k=8: 100%   (28 test items)
```

<figure class="fig"><svg viewBox="0 0 780 334" role="img" aria-label="Accuracy of the base model on three tasks as the number of examples in the prompt grows from 0 to 8."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">In-context learning in Qwen2.5-0.5B base: no training, only examples in the prompt (ch4_probe.py)</text><line class="grid" x1="80" y1="284.0" x2="520" y2="284.0"/><text class="t-tick" x="72.0" y="288.0" text-anchor="end">0%</text><line class="grid" x1="80" y1="229.0" x2="520" y2="229.0"/><text class="t-tick" x="72.0" y="233.0" text-anchor="end">25%</text><line class="grid" x1="80" y1="174.0" x2="520" y2="174.0"/><text class="t-tick" x="72.0" y="178.0" text-anchor="end">50%</text><line class="grid" x1="80" y1="119.0" x2="520" y2="119.0"/><text class="t-tick" x="72.0" y="123.0" text-anchor="end">75%</text><line class="grid" x1="80" y1="64.0" x2="520" y2="64.0"/><text class="t-tick" x="72.0" y="68.0" text-anchor="end">100%</text><text class="t-tick" x="80.0" y="302.0" text-anchor="middle">0</text><text class="t-tick" x="190.0" y="302.0" text-anchor="middle">1</text><text class="t-tick" x="300.0" y="302.0" text-anchor="middle">2</text><text class="t-tick" x="410.0" y="302.0" text-anchor="middle">4</text><text class="t-tick" x="520.0" y="302.0" text-anchor="middle">8</text><line class="axis" x1="80" y1="284" x2="520" y2="284"/><polyline class="l1" points="80.0,189.7 190.0,95.4 300.0,103.3 410.0,95.4 520.0,103.3"/><g class="mark"><title>English to French: 0, 43%</title><circle class="s1 ring" cx="80.0" cy="189.7" r="4"/></g><g class="mark"><title>English to French: 1, 86%</title><circle class="s1 ring" cx="190.0" cy="95.4" r="4"/></g><g class="mark"><title>English to French: 2, 82%</title><circle class="s1 ring" cx="300.0" cy="103.3" r="4"/></g><g class="mark"><title>English to French: 4, 86%</title><circle class="s1 ring" cx="410.0" cy="95.4" r="4"/></g><g class="mark"><title>English to French: 8, 82%</title><circle class="s1 ring" cx="520.0" cy="103.3" r="4"/></g><polyline class="l2" points="80.0,205.4 190.0,79.7 300.0,87.6 410.0,103.3 520.0,87.6"/><g class="mark"><title>antonyms: 0, 36%</title><circle class="s2 ring" cx="80.0" cy="205.4" r="4"/></g><g class="mark"><title>antonyms: 1, 93%</title><circle class="s2 ring" cx="190.0" cy="79.7" r="4"/></g><g class="mark"><title>antonyms: 2, 89%</title><circle class="s2 ring" cx="300.0" cy="87.6" r="4"/></g><g class="mark"><title>antonyms: 4, 82%</title><circle class="s2 ring" cx="410.0" cy="103.3" r="4"/></g><g class="mark"><title>antonyms: 8, 89%</title><circle class="s2 ring" cx="520.0" cy="87.6" r="4"/></g><polyline class="l3" points="80.0,284.0 190.0,166.1 300.0,276.1 410.0,64.0 520.0,64.0"/><g class="mark"><title>made-up labels (blue/red): 0, 0%</title><circle class="s3 ring" cx="80.0" cy="284.0" r="4"/></g><g class="mark"><title>made-up labels (blue/red): 1, 54%</title><circle class="s3 ring" cx="190.0" cy="166.1" r="4"/></g><g class="mark"><title>made-up labels (blue/red): 2, 4%</title><circle class="s3 ring" cx="300.0" cy="276.1" r="4"/></g><g class="mark"><title>made-up labels (blue/red): 4, 100%</title><circle class="s3 ring" cx="410.0" cy="64.0" r="4"/></g><g class="mark"><title>made-up labels (blue/red): 8, 100%</title><circle class="s3 ring" cx="520.0" cy="64.0" r="4"/></g><rect class="s3" x="532.0" y="59.0" width="10" height="10" rx="2"/><text class="t-tick" x="547.0" y="68.0" text-anchor="start">made-up labels (blue/red)</text><rect class="s2" x="532.0" y="82.6" width="10" height="10" rx="2"/><text class="t-tick" x="547.0" y="91.6" text-anchor="start">antonyms</text><rect class="s1" x="532.0" y="98.6" width="10" height="10" rx="2"/><text class="t-tick" x="547.0" y="107.6" text-anchor="start">English to French</text><text class="t-muted" x="300.0" y="322.0" text-anchor="middle">number of solved examples in the prompt (k)</text><text class="t-muted" x="72.0" y="48.0" text-anchor="start">accuracy on 28 held-out items</text></svg><figcaption>In-context learning in Qwen2.5-0.5B base. Translation and antonyms jump from about 40% with only a description to about 85% with a single example. The made-up label task is impossible with no examples and perfect with four.</figcaption></figure>

For translation and antonyms, one example does most of the work: it shows the *format*, and the knowledge was already there. The made-up labels show genuine learning from the prompt: 0% with no examples (the model has never seen "blue" mean "positive"), and 100% with four. The steps in between are revealing. With one example (a positive review labelled "blue") the model answers "blue" for everything, which is right for the 15 positive reviews out of 28: 54%. With two examples, one "blue" and one "red", it got only 1 of 28 right. Asking it what it answers shows why:

```text
what the model answers with only two labelled examples:
   'The view was beautiful' -> 'green'
   'The room was dirty' -> 'yellow'
   'Superb quality' -> 'green'
```

With two colours in the prompt, the 0.5B model decides it is reading *a list of colours* and continues the list. With four examples the pattern "positive review means blue" becomes the most likely reading. Small base models learn in context, but fragilely, which is one more reason to fine-tune them.

> [!NOTE]
> These are 28 items per task with one fixed choice of examples. Change the examples or their order and the numbers move by several points, which is itself a known property of few-shot prompting. The qualitative pattern (description alone is weak, a few examples help a lot, arbitrary labels need several) is robust.

## 4.13 Data mixtures, mid-training and annealing

The last ingredient is the one most model reports now spend the most pages on: *which data, in what proportion, at which point of training*.

> [!DEFINITION] Data mixture
> The proportions in which different sources (web text, code, mathematics, books, multilingual text, and so on) are sampled during training. A source can be shown more often than its size suggests (**upsampled**) or less (**downsampled**).

Llama 3 describes its final pretraining mixture in one line, and then a technique that has become standard:

> [!PAPER] Grattafiori et al. (2024), The Llama 3 Herd of Models · Sections 3.1.2 and 3.1.3 · page 6
> [![The Llama 3 data mix summary and the annealing data section, with highlights on roughly 50% of tokens corresponding to general knowledge, 25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens, annealing improved the performance, and by 24.0% and 6.4%, respectively](/img/training/ch4-llama3-mix-anneal.png)](/img/training/ch4-llama3-mix-anneal.png)
>
> **Context:** the end of the pre-training data section.
>
> **What it says:** the final mix is about 50% general knowledge, 25% mathematics and reasoning, 17% code and 8% multilingual text. Separately, "annealing" on small amounts of high-quality code and mathematics data raised a Llama 3 8B model's scores on GSM8K and MATH by 24.0% and 6.4%, while the gain for the 405B model was negligible.
>
> **Why it matters:** half of the tokens are not general web text. Code and mathematics are deliberately over-represented, because they improve reasoning far beyond their share of the web.

How do teams pick those proportions? The same way as everything else in this chapter: by training small models on candidate mixtures, measuring them, and using scaling laws to predict which mixture will be best at full scale. Llama 3 describes exactly that process for its data mix.

> [!DEFINITION] Annealing
> The final phase of pretraining, in which the learning rate is decayed to (near) zero while the data mixture is shifted towards the highest-quality sources. Because the model is "settling" during this phase, what it sees then has an outsized effect on the final model.

> [!PAPER] Grattafiori et al. (2024), The Llama 3 Herd of Models · Section 3.4.3 · page 15
> [![The Llama 3 annealing paragraph with highlights on linearly annealed the learning rate to 0, upsample data sources, and average of model checkpoints](/img/training/ch4-llama3-annealing.png)](/img/training/ch4-llama3-annealing.png)
>
> **Context:** the last stage of the 405B model's pre-training.
>
> **What it says:** during the final 40M tokens, the learning rate is decayed linearly to 0 while the mix upsamples very high-quality sources; the final base model is the average of several checkpoints from this phase.
>
> **Why it matters:** 40M tokens is about 0.0003% of the 15.6T total, yet this is where the model gets its finishing touches.

The Llama 3 paper also uses annealing as a cheap *test*: to judge whether a new small dataset is valuable, anneal a partly trained 8B model with 30% of the new data and 70% of the usual mix, and compare benchmark scores. That is far cheaper than a full training run per candidate dataset.

Between the main phase and the annealing, many recent models add a stage often called **mid-training**.

> [!DEFINITION] Mid-training
> A stage between general pretraining and fine-tuning, still trained with the next-token objective on large amounts of data, but with a different mixture or setup: much more code, mathematics or synthetic reasoning text, longer documents to extend the context window, or instruction-like data. It sits in a grey zone: pretraining in method, closer to fine-tuning in purpose.

For Llama 3 405B, the long-context stage is an example: after the main phase, the context window was raised from 8K to 128K tokens in six steps, using about 800B tokens of training. Putting it together:

<figure class="fig"><svg viewBox="0 0 760 270" role="img" aria-label="The Llama 3 data mix and the three stages of its pretraining: initial pretraining, long-context training and annealing."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">The phases of a modern pretraining run (numbers from the Llama 3 paper)</text><text class="t-muted" x="20.0" y="50.0" text-anchor="start">final data mix (share of tokens)</text><g class="mark"><title>general knowledge: 50%</title><rect class="s1" x="20.0" y="58" width="358.0" height="26" rx="3" style="fill-opacity:0.8"/></g><text class="t-tick" x="200.0" y="100.0" text-anchor="middle">general knowledge 50%</text><g class="mark"><title>math + reasoning: 25%</title><rect class="s2" x="380.0" y="58" width="178.0" height="26" rx="3" style="fill-opacity:0.8"/></g><text class="t-tick" x="470.0" y="100.0" text-anchor="middle">math + reasoning 25%</text><g class="mark"><title>code: 17%</title><rect class="s3" x="560.0" y="58" width="120.4" height="26" rx="3" style="fill-opacity:0.8"/></g><text class="t-tick" x="621.2" y="100.0" text-anchor="middle">code 17%</text><g class="mark"><title>multilingual: 8%</title><rect class="s4" x="682.4" y="58" width="55.6" height="26" rx="3" style="fill-opacity:0.8"/></g><text class="t-tick" x="711.2" y="100.0" text-anchor="middle">multilingual 8%</text><rect class="box-1" x="20.0" y="130.0" width="470.0" height="40.0" rx="8"/><text class="t-tick" x="28.0" y="155.0" text-anchor="start">1. initial pretraining</text><rect class="box-3" x="495.0" y="130.0" width="150.0" height="40.0" rx="8"/><text class="t-tick" x="503.0" y="155.0" text-anchor="start">2. long-context</text><rect class="box-2" x="650.0" y="130.0" width="100.0" height="40.0" rx="8"/><text class="t-tick" x="658.0" y="155.0" text-anchor="start">3. annealing</text><text class="t-tick" x="20.0" y="196.0" text-anchor="start">1: about 15T tokens, cosine LR, batch 4M to 16M tokens, context 8K</text><text class="t-tick" x="20.0" y="213.0" text-anchor="start">2: context grown to 128K in steps, ~800B tokens</text><text class="t-tick" x="20.0" y="230.0" text-anchor="start">3: last 40M tokens: LR to 0, high-quality data upsampled, then average the checkpoints</text><text class="t-muted" x="20.0" y="256.0" text-anchor="start">Widths are not to scale: annealing is a tiny fraction of the tokens but has an outsized effect on benchmarks.</text></svg><figcaption>The phases of a modern pretraining run, with the numbers Llama 3 reports: a data mix where half the tokens are not general web text; a long initial phase with a cosine schedule and a growing batch; a long-context stage; and a short annealing phase on the best data. The widths are not to scale.</figcaption></figure>

## 4.14 Where this leaves us

At the end of pretraining you have a base model: a network that has compressed trillions of tokens into a few billion numbers, that knows facts and grammar, and that can pick up a task from a few examples. You have also seen its limits. It completes "The chemical symbol for gold is" with a blank because worksheets do. Asked a question, it might answer, continue with three more questions, or start a quiz. It has no notion of a user, of a turn, or of when to stop.

Turning this into a model that follows instructions does not need anything like the compute of pretraining. It needs a much smaller amount of the *right* data, and a few careful changes to the same loss. That is the subject of Chapter 5.

> [!TAKEAWAYS]
> - Pretraining is next-token prediction on trillions of tokens of general text. The loss is the same as in Chapter 1; the difficulty is in the data and the scale.
> - Raw web text is mostly unusable. In a real Common Crawl sample, 35% of pages were English and only 7.4% of all pages survived Gopher, C4 and FineWeb-style rules. MinHash removes near-duplicates by comparing signatures whose agreement estimates Jaccard similarity, and quality classifiers such as FineWeb-Edu's keep the most useful pages.
> - BPE builds a vocabulary by repeatedly merging the most frequent pair of symbols. The vocabulary reflects its training text: numbers, rare words and other languages can cost several times more tokens.
> - Training compute is about C = 6ND FLOPs. Our laptop run was 9.1e14 FLOPs; Llama 3 405B was 3.8e25.
> - Loss falls as a power law in parameters, data and compute. Chinchilla showed that parameters and tokens should grow together, about 20 tokens per parameter for a fixed training budget; models meant for heavy use are deliberately trained far longer (Llama 3 8B: 1,875 tokens per parameter, Qwen2.5-0.5B: about 36,700).
> - The learning rate warms up, then decays (cosine or warmup-stable-decay). Batches are large, because a single batch's gradient is mostly noise; our two-batch test showed agreement roughly doubling with each doubling of the batch.
> - Mixed precision runs the heavy matrix work in bf16 and keeps fp32 master weights, because bf16 rounds small updates away. Training with AdamW costs about 16 bytes per parameter.
> - A 12M-parameter GPT pretrained for 8 minutes on a laptop went from random tokens (loss 8.37) to short consistent stories (validation loss 2.07, perplexity 7.95).
> - A base model knows facts and grammar and can learn in context, but its behaviour follows the documents it was trained on, not the user. Data mixtures, mid-training and a final annealing phase on the best data shape the last part of pretraining.

## References

### Papers

- Penedo, G. et al. (2024). *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.* [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
- Gao, L. et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling.* [arXiv:2101.00027](https://arxiv.org/abs/2101.00027)
- Rae, J. W. et al. (2021). *Scaling Language Models: Methods, Analysis & Insights from Training Gopher.* [arXiv:2112.11446](https://arxiv.org/abs/2112.11446)
- Raffel, C. et al. (2019). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (C4). [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
- Penedo, G. et al. (2023). *The RefinedWeb Dataset for Falcon LLM.* [arXiv:2306.01116](https://arxiv.org/abs/2306.01116)
- Li, J. et al. (2024). *DataComp-LM: In search of the next generation of training sets for language models.* [arXiv:2406.11794](https://arxiv.org/abs/2406.11794)
- Lee, K. et al. (2022). *Deduplicating Training Data Makes Language Models Better.* [arXiv:2107.06499](https://arxiv.org/abs/2107.06499)
- Sennrich, R., Haddow, B. and Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- Vaswani, A. et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models.* [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Besiroglu, T. et al. (2024). *Chinchilla Scaling: A replication attempt.* [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners* (GPT-3). [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Grattafiori, A. et al. (2024). *The Llama 3 Herd of Models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- Qwen Team (2024). *Qwen2.5 Technical Report.* [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
- Hu, S. et al. (2024). *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (warmup-stable-decay). [arXiv:2404.06395](https://arxiv.org/abs/2404.06395)
- Loshchilov, I. and Hutter, F. (2019). *Decoupled Weight Decay Regularization* (AdamW). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Micikevicius, P. et al. (2018). *Mixed Precision Training.* [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Eldan, R. and Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)

### Other sources

- Common Crawl, the open web archive used in Section 4.2. [commoncrawl.org](https://commoncrawl.org/)
- The FineWeb-Edu classifier used in `ch4_data.py`. [huggingface.co/HuggingFaceFW/fineweb-edu-classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier)
- The TinyStories dataset used in Section 4.11. [huggingface.co/datasets/roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- The scripts behind every number in this chapter: `code/training/ch4_*.py`, with their saved output in `code/training/results/`.
