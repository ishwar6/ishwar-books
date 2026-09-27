---
title: How this site renders Markdown
description: "A tour of every element the reader supports (headings, code, callouts, tables, math and diagrams), so writing a new chapter is just dropping a .md file into content/."
date: 2026-09-27
tags: [meta, writing]
---

Everything on this site starts life as a plain Markdown file. Drop a file into `content/writings/` and it becomes a page; drop a folder into `content/books/` and it becomes a book with a chapter sidebar. This page doubles as the style guide: if it looks right here, it looks right everywhere.

## Headings carry the structure

Second-level headings open a major section and show up in the **On this page** panel. Third-level headings are sub-sections, and get their own colour so the hierarchy reads at a glance.

### A sub-section

Body text is set large with generous line height, because these pages are meant to be *read*, not skimmed. Links look like [this one](https://github.com/ishwar6), and `inline code` stands out without shouting.

#### A minor heading

Use these sparingly, for small groups inside a sub-section.

## Code

Fenced code blocks are highlighted at build time, so there is no flash of unstyled code. Each block shows its language and has a copy button.

```python
import torch

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention, the one formula every transformer is built on."""
    scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    return scores.softmax(dim=-1) @ v
```

```cuda
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];   // one thread, one element
}
```

```bash
# write a new post
cp content/writings/how-this-site-renders-markdown.md content/writings/my-new-post.md
npm run dev
```

## Callouts

Use GitHub-style alerts to pull an idea out of the flow of text.

> [!NOTE]
> A RAG system is only as good as the **worst** of its three stages: ingestion, retrieval and generation.

> [!TIP] Measure before you tune
> You cannot fix a stage you have not measured. Put recall@k on a dashboard before touching the chunker.

> [!WARNING]
> Tensor cores only help when your matrix dimensions are multiples of 8 (FP16) or 16 (INT8).

> [!IMPORTANT] Interview angle
> When they ask *why* reranking helps, the answer is that bi-encoders compress a document into one vector before seeing the query; a cross-encoder reads both together.

> A plain blockquote still works for quotations.

## Tables

| Bottleneck | You are limited by | Typical fix |
|---|---|---|
| Memory bandwidth | bytes/sec from DRAM | fuse kernels, reuse data in shared memory |
| Compute throughput | FLOP/sec of the ALUs | tensor cores, lower precision |
| Launch overhead | kernels too small to matter | CUDA graphs, batching |

## Lists

1. Read the chapter start to finish.
2. Run the code and write down the numbers.
3. Answer the interview questions without looking.

- [x] Headings, code, tables
- [x] Callouts and math
- [ ] Your next chapter

## Math

Display math uses double dollars:

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Inline math also uses double dollars, like $$O(n^2 d)$$, so prices such as $5 are never mistaken for maths.

## Diagrams

Mermaid code blocks turn into diagrams:

```mermaid
flowchart LR
    Q[Question] --> R[Retriever]
    R --> K[Top-k chunks]
    K --> G[LLM]
    Q --> G
    G --> A[Grounded answer]
```

## Frontmatter

Every file can start with optional frontmatter. Anything left out is inferred: the title from the first `#` heading, the description from the first paragraph.

```yaml
---
title: Why RAG exists
description: One line shown on cards and in search results.
date: 2026-09-27
tags: [rag, retrieval]
order: 1          # books only: chapter order (otherwise sorted by file name)
draft: true       # hidden from the production build
---
```

Links between Markdown files, like `[next](02-embeddings.md)`, are rewritten to the right page automatically, and Obsidian `[[wikilinks]]` work too.
