# Chapter 28 · Documents in the Real World: PDFs, Tables, Diagrams

> **Goal:** you can say exactly what happens to every part of a PDF on its way into a vector
> index (which parts survive, which are silently dropped, and which arrive *wrong*) and you
> can choose and defend a representation for tables and figures. By the end you will have
> measured all three: a table that loses its columns, a diagram that contributes nothing, and
> an OCR error rate that costs lexical search 18% of its MRR while dense search barely notices.
>
> Files: [`code/ch28/make_sample_pdf.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch28/make_sample_pdf.py), `parser_shootout.py`, `tables_and_representations.py`,
> `figures_to_text.py`, `ocr_noise.py`.

[Chapter 5](../part2-building/05-chunking.md) gave this one section and the honest summary
"parsing is where the battle is". This chapter fights the battle.

---

## 28.1 A PDF has no structure. It has coordinates.

Start here, because every wrong answer downstream comes from getting this wrong.

A PDF is a page-description program. Its content stream says *put this glyph at x=321.5,
y=91.7 in this font at this size*, then *stroke a line from here to here*, then *paint this
image in this rectangle*. That is all. There is no paragraph, no heading, no table, no
figure, no reading order, and (outside of optional, usually-absent tagging) no semantics at
all. HTML has `<table>` and `<h1>`; PDF has ink.

```
   what the author saw            what the file contains
   ------------------            ----------------------
   ┌──────────────────┐          BT 47.6 512 Td (Maximum payload) Tj ET
   │ Maximum payload  │          BT 268.0 512 Td (250 kg) Tj ET
   │           250 kg │   ==>    BT 416.7 512 Td (120 kg) Tj ET
   │           120 kg │          47.6 508 380 0.6 re f   (a hairline)
   └──────────────────┘          ... no row, no column, no cell, no table
```

(Content streams are postfix: operands first, operator last. `Td` sets the text position,
`Tj` paints a string with the current font, `re` appends a rectangle and `f` fills it.)

Everything a parser gives you above that level ("this is a paragraph", "these cells form a
row", "this is Figure 2") is **reconstruction**, inferred from coordinates, font sizes and
whitespace. Different parsers infer differently. That is why two libraries give two different
answers on the same file and neither is lying.

Three consequences that you will meet on your first real corpus:

- **Reading order is a guess.** The content stream order is whatever the producing program
  emitted. Word, LaTeX, Chrome's print-to-PDF and a scanner all emit differently.
- **Columns interleave.** Any extractor that sorts text by vertical position welds
  side-by-side columns into single lines, because line 1 of the left column and line 1 of the
  right column have the same `y`.
- **A figure is not text.** A vector diagram is line and curve operators; a photo or a
  screenshot is a single image object. Unless you do something deliberate, neither one puts a
  single searchable word into your index.

[`code/ch28/make_sample_pdf.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch28/make_sample_pdf.py) builds a six-page fixture that isolates each failure: a title
page, a two-column page, a table, a vector chart, a vector tree diagram, and the same chart
pasted as a raster image. It is generated rather than shipped so you can read exactly what
went in and compare it with what comes out.

> **The rcParam that decides whether your PDF is readable at all.** matplotlib defaults to
> Type 3 fonts in PDFs, which many extractors read as mojibake because Type 3 glyph programs
> carry no reliable `ToUnicode` map. The fixture sets `pdf.fonttype = 42` (TrueType). When
> extraction returns garbage from a real file, this class of producer decision: fonts,
> encodings, missing `ToUnicode`: is the first thing to check, before you blame the parser.
> Ligatures are the same problem in miniature: a typesetter emits one `ﬁ` glyph, and without a
> `ToUnicode` map that maps it back to `f` + `i`, "office" and "find" arrive in your index as
> tokens no user will ever type. Normalise ligatures at ingest.

## 28.2 The parser landscape

Two parsers on the same six pages, measured (`parser_shootout.py`):

```
parser          ms   chars  gives you
pypdf           16    1373  text in stream order, no coordinates
pymupdf          6    1380  blocks/spans + bbox, images, drawing ops
```

Both recover essentially the same characters. The difference that matters is not speed or
volume, it is **what else** you get: `pypdf` hands you a string, `pymupdf` hands you every
span with its bounding box, plus the image objects and the vector drawing operators. You
cannot fix reading order, detect a table, or crop a figure without coordinates. A string is a
dead end.

| Tool | What it returns | Runs where | Cost model | Reach for it when |
|---|---|---|---|---|
| **pypdf** | text, per page; form fields; page ops | pure Python, in-process | free | simple single-column text, metadata, splitting/merging |
| **pymupdf** | spans + bboxes, images, drawings, page→PNG rendering | C library, in-process, fast | free (AGPL; commercial licence otherwise) | the default when you need layout, figures, or rendering |
| **pdfplumber** | words/lines + ruling-line based table extraction | pure Python, slower | free | tables that are drawn with visible rules |
| **Docling** | document model: headings, lists, tables, figures, reading order | local ML models, GPU helps | free, compute | structured output from complex reports, offline |
| **Unstructured** | typed elements (Title, NarrativeText, Table, Image) | local or hosted API | free / per-page | mixed corpora and many file types, fast to adopt |
| **Marker** | PDF → clean markdown, good with equations | local ML models, GPU | GPL-3.0 code, weights free only below a revenue threshold; commercial licence otherwise | scientific papers and books |
| **LlamaParse** | layout+LLM parsing to markdown/JSON | hosted | per page | complex tables where local parsers fail |
| **AWS Textract / Azure Document Intelligence / Google Document AI** | OCR, layout, tables, key–value, forms, per-element confidence | hosted | ~$1–65 per 1k pages by feature | scanned documents, forms, compliance, SLAs |
| **Mistral OCR and similar VLM OCR** | markdown incl. figure descriptions | hosted | per page | scanned or visually complex documents |

**What I actually ran here:** `pypdf` and `pymupdf`, with the numbers above. Everything else
in that table is characterised from its documentation, not benchmarked on this fixture: treat
those rows as a map, and benchmark the two or three candidates that fit your corpus on *your*
pages before choosing. That benchmark is cheap and it is the only opinion worth having. Check
the licence and the per-page price at the time you read this: both columns move, and two of
these tools are free for you and not free for your employer.

The decision rule is about your documents, not about features:

- Digital, single-column, text-dominant → `pypdf` or `pymupdf`. Do not overbuy.
- Multi-column, tables that matter, figures that matter → a layout parser (Docling,
  Unstructured) or `pymupdf` plus the 25 lines of column logic in the next section.
- Scanned, handwritten, forms, or "my answers depend on the numbers inside tables in
  scanned pages" → a hosted document-AI service, and budget per page.

## 28.3 Rebuilding reading order

Here is the trap, measured on page 2 of the fixture. Sort every span top-to-bottom then
left-to-right: the obvious thing, and what a hand-rolled script does:

```
naive y-then-x sort - the two columns are welded together:
   Lumora Robotics deployed 6,000 Atlas
   Field service costs fell 18 percent
   robots across 120 warehouses in 14
   after predictive battery scheduling
   countries during the 2025 financial
```

Every second line comes from the other column. Chunk that and you embed sentences that no
human wrote. The fix is to find the **gutter** before sorting (the vertical band no text
crosses) and read column by column:

```
column-aware order - left column first, as written:
   Lumora Robotics deployed 6,000 Atlas
   robots across 120 warehouses in 14
   countries during the 2025 financial
   year. The largest single site, near
   Rotterdam, runs 410 robots on one
```

The implementation is a projection profile, and it is short enough to read in one sitting:

```python
def find_gutter(spans, page_width):
    intervals = sorted((sp[1], 2 * sp[2] - sp[1]) for sp in spans)   # (x0, x1)
    merged = []
    for x0, x1 in intervals:
        if x1 - x0 > 0.55 * page_width:              # spans the whole width: ignore
            continue
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    best, best_w = None, 0.04 * page_width           # a gutter is at least 4% wide
    for (_, end), (start, _) in zip(merged, merged[1:]):
        centre = (end + start) / 2
        if start - end > best_w and 0.25 * page_width < centre < 0.75 * page_width:
            best, best_w = centre, start - end
    return best
```

Two details in there are the difference between working and not:

- **Full-width spans are excluded from the profile.** A heading or a footer that crosses the
  gutter would paint over it and the page would look single-column. Layout parsers do the same
  thing, usually per horizontal band.
- **Column assignment uses the span's own extent, not its midpoint.** My first version tested
  midpoints against a middle band, and a single short line: `on every measured site.`, centre
  382 in a 595-point page: fell inside the band and collapsed the whole page to one column.
  Geometric heuristics fail on one line out of twenty; that is why real pipelines eventually
  move to a trained layout model.

Headers and footers deserve the same treatment before chunking: they repeat on every page, so
they inflate term frequencies, waste chunk budget, and produce near-duplicate chunks. Strip
text whose bounding box repeats at the same coordinates across pages.

## 28.4 Tables: the grid is the meaning

`250 kg` means nothing on its own. It means something because it sits in the row
*Maximum payload* and the column *Atlas A2*. Flatten the table and the number survives while
its coordinates in the grid (its actual meaning) do not.

`tables_and_representations.py` indexes the same two tables six ways, adds 48 distractor
chunks from the rest of the handbook, and asks nine table-lookup questions, several of which
require the *column* (`... of the Atlas A2 Lite?`). It scores retrieval (did the value appear
in the top-k at all) separately from the answer (did the model then say it):

```
9 table questions, k=2, 48 distractor chunks from the rest of the handbook

=== no chunk limit - the whole table fits in one chunk ===
representation       tbl ch  retrieval   answer
flattened (naive)         4       8/9     8/9
markdown                  2       9/9     9/9
html                      2       9/9     9/9
csv                       2       9/9     9/9
row-per-chunk            15       9/9     9/9
verbalised               15       9/9     9/9

=== chunk limit 300 chars - a big table gets cut up ===
representation       tbl ch  retrieval   answer
flattened (naive)         4       8/9     7/9
markdown                  4       8/9     8/9
html                      5       8/9     8/9
csv                       3       9/9     9/9
row-per-chunk            15       9/9     9/9
verbalised               15       9/9     9/9

the interesting cases - value was retrieved, answer still wrong:
  [flattened (naive)] Which support tier includes an on-site spares kit?
      -> Gold
```

That is **one run**, and the honest reading matters more than the digits. Nine questions means
a one-question difference is one question. Re-running moves the split-regime rows by about a
question in either direction (flattened 7–8/9, markdown 8–9/9), and the wrong-answer example
printed below appears in some runs and not others, because the generator is stochastic. What
repeats across runs is the *ordering* (self-contained chunks ≥ whole-table formats > naive
flatten) and the mechanism behind it:

- **While the table fits in one chunk, every structured representation ties.** Markdown, HTML
  and CSV all keep the header adjacent to the rows, and the model reads the grid. This is the
  regime most demos are in, which is why most demos conclude "markdown is fine".
- **The moment a chunker cuts the table, representations separate.** Markdown and HTML can lose
  a question because the header row ends up in one chunk and the later rows in another: the
  model receives `| Gold | 24x7 | 30 minutes | ...` with no idea what the columns are.
- **The naive flatten can produce a confidently wrong answer**, not a refusal. In the run above,
  asked which tier includes an on-site spares kit, it answered **Gold**: the value is real, it
  belongs to Platinum, and the row boundary that said so was destroyed by the chunker. A wrong
  answer is much worse than a missing one, and separating header from rows is one of the
  cleanest ways to manufacture them.
- **Self-contained chunks are immune.** Row-per-chunk (title + header + one row) and
  verbalised rows ("In the Atlas A2 specification, for Maximum payload: Atlas A2 is 250 kg;
  Atlas A2 Lite is 120 kg.") score 9/9 in both regimes, because no chunk ever depends on a
  neighbour it might not be retrieved with.

The rules that follow, in priority order:

1. **Never let a splitter cut a table.** Detect tables at parse time and treat each one as an
   atomic unit your chunker is not allowed to break.
2. **If a table is bigger than your chunk budget, split it by rows and repeat the title and
   header into every chunk.** Repetition costs a few tokens and buys independence.
3. **Store the whole table as the payload, even when you embed a row or a summary.** Retrieve
   a row, give the model the table. Embedding text and context text do not have to be equal:
this is the same small-to-big idea as parent/child retrieval in
   [Chapter 9](../part3-retrieval/09-advanced-retrieval.md).
4. **Verbalise when queries are natural-language and the table is small.** It embeds closest
   to how people ask, at the cost of tokens at ingest and an LLM pass to generate.
5. **Keep numbers with their units and their row label in the same chunk.** Most "the RAG got
   the number wrong" bugs are this.

## 28.5 Diagrams: the tree-diagram question, answered

This is the question that exposed the gap. *A PDF contains a tree diagram. What happens to it?*

The answer has two halves, and the first half surprises people:

**A vector diagram's labels are real text and they extract. Its structure does not.** Here
is page 5 of the fixture (an org chart with nine boxes and eight connectors) as the parser
sees it:

```
page 5 (tree diagram) text - every node, not one edge:
   Lumora Robotics | Robot Platform | Fleet Platform | Field & Support |
   Navigation | Firmware | Beacon | Compass | EU Service |
   5.1  Engineering organisation | Figure 2: engineering reporting lines. | 5
```

Nine node labels, in drawing order, with **no edges**. The parent-child relationships exist
only as line coordinates: `18 drawing operators` on that page. So the chunk you index says
that these nine names appeared near each other, and nothing about who reports to whom. A
question like *who does Navigation report to?* cannot be answered from it, and the index gives
you no signal that anything is missing.

The same page as a chart shows the numeric version of the loss:

```
page 4 (vector chart) text - axis vocabulary, no data:
   Pune | Bengaluru | Rotterdam | Austin | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | robots
```

The categories and the axis ticks are there. Which bar is 410 high is not: bar heights are
path-fill operators.

**A raster figure contributes nothing at all.** The same chart pasted as an image:

```
page 6 (same chart, rasterised) text - the caption and nothing else:
   6.1  Deployment by site (exported) | Figure 3: the same chart, pasted as an image. | 6
```

Per page, what reaches the indexer:

```
page kind                    chars  draw ops  imgs
   1 title                      67         1     1
   2 two-column prose          597         1     1
   3 table                     274         8     1
   4 vector chart              180        21     1
   5 vector tree diagram       179        18     1
   6 raster chart               83         1     1
```

### The four architectures

**(a) Ignore the figure.** The default, and it is a *silent* failure: nothing errors, the
document indexes, and the questions that depended on the figure get answered from whatever
else was nearby. Measured below, this produced the worst outcome in the chapter.

**(b) OCR the image.** Correct for text *inside* an image: a scanned page, a screenshot of a
table. Useless for structure: OCR on the org chart returns the same nine labels with no edges.
OCR reads glyphs; it does not read arrows.

**(c) Have a vision model describe the figure, index the description.** Render the page (or
crop the figure's bbox) to PNG, send it to a VLM, and index the returned text with a pointer
back to the page image. This is the workhorse today: it turns pixels and vectors into words
your existing text pipeline already handles.

**(d) Skip text entirely: multimodal or late-interaction embeddings.** CLIP-style and newer
multimodal embedders (Cohere embed-v4, Voyage multimodal) put images and text in one vector
space, so a text query can retrieve an image directly. ColPali/ColQwen go further and embed
*page images* as multi-vector patch representations with MaxSim scoring, no OCR at all: see
[Chapter 19 §19.5](../part7-frontier/19-whats-new-in-rag.md). Strong on visually dense pages,
and expensive: many vectors per page, a heavier index, and a retrieval stack that must support
multivectors.

### A caption contains only what you asked for

This is the part to internalise. The VLM does not extract "the figure"; it answers your
prompt. `figures_to_text.py` captions the same two figures twice: once with
`"Describe this figure."` and once with a prompt that demands structure: and the difference
is not stylistic:

```
--- structured caption, page 5 ---
Engineering organisation reporting lines.
Lumora Robotics -> Robot Platform
Lumora Robotics -> Fleet Platform
Lumora Robotics -> Field & Support
Robot Platform -> Navigation
Robot Platform -> Firmware
Fleet Platform -> Beacon
Fleet Platform -> Compass
Field & Support -> EU Service

--- structured caption, page 4 ---
Figure 1: Robots deployed per site, 2025.
Pune: 210
Bengaluru: 95
Rotterdam: 410
Austin: 160
```

The generic prompt produced good prose (nested bullets naming the same hierarchy) but prose
about a hierarchy is not the same artefact as an edge list. Indexed alongside 51 handbook
chunks and asked four questions that only the figures can answer:

```
index        q1  q2  q3  q4   score
extracted    -   -   -   OK   1/4
generic      OK  -   OK  OK   3/4
structured   OK  OK  OK  OK   4/4

  Q2: Who does the Navigation team report to?
     extracted   I don't know.
     generic     I don't know.
     structured  Robot Platform

  Q3: How many robots are deployed at the Rotterdam site?
     extracted   47
     generic     410
     structured  410
```

Q3 is the line to remember. The text-only index answered **47**: a real number, from the
Rotterdam outage postmortem, which has nothing to do with deployment counts. The figure
contributed no data, so retrieval found the nearest plausible number in the corpus and the
model used it. **Dropping a figure does not produce "I don't know"; it produces a confident
wrong answer sourced from somewhere else.**

Two honest caveats on those numbers. The company-overview document states the org structure
and the Rotterdam figure in prose, so it is excluded from the distractors: otherwise every
index would score well by reading the text and the experiment would measure nothing. And
captions are LLM output: rerun it and the generic caption may score 4/4 or 2/4. The
*structured* caption is stable across runs, which is itself the argument for it.

### Prompt the structure you intend to query

- Hierarchy or flow chart → *"list every edge as `parent -> child`, one per line, covering
  every box"*.
- Chart → *"one line per bar as `label: value`, using the axis scale"*.
- Architecture diagram → components, then connections, then the labels on the connections.
- Screenshot of a table → *"return the table as markdown"*: then treat it as §28.4 says.

Store the structured caption **and** a pointer to the rendered image (`doc_id`, `page`,
`bbox`), so an answer can show the figure it used. And keep the caption short: it is a
retrieval surface, not documentation.

### Choosing, in practice

| Situation | Do this |
|---|---|
| Figures are decorative; answers never depend on them | Ignore them, and say so explicitly in your design notes so it is a decision, not an accident |
| A few hundred documents with meaningful charts/diagrams | VLM captions at ingest (c). One cheap call per figure, one time |
| Text inside images (scans, screenshots of text) | OCR (b), plus a VLM pass for anything OCR mangles |
| Visually dense corpus: slides, financial reports, catalogues | Evaluate ColPali-style page embeddings (d) against captions; budget for the index |
| Regulated, audited answers | Captions plus the image and page provenance, so a human can verify the source |

Costs, so the decision is arithmetic: a caption is roughly 1k–2k input tokens and ~100 output
tokens per figure, one time at ingest. For 50,000 figures on a cheap vision-capable model that
is tens of dollars: far less than the meetings you will spend deciding.

## 28.6 Scanned pages and OCR error

If pages are images, OCR is not optional: but OCR output is *noisy text*, and noise lands
differently on your two retrievers. `ocr_noise.py` corrupts the corpus with realistic glyph
confusions (`rn`→`m`, `l`→`1`, `O`→`0`), lost word spaces and spurious hyphen breaks, leaves
the queries clean, and measures with exact search so nothing but the noise is moving:

```
 noise | dense h@1   h@5   MRR |  BM25 h@1   h@5   MRR
    0% |      1.00  1.00  1.00 |      0.71  0.98  0.83
    2% |      1.00  1.00  1.00 |      0.69  0.93  0.81
    5% |      1.00  1.00  1.00 |      0.67  0.95  0.78
   10% |      0.98  1.00  0.99 |      0.62  0.95  0.75
   20% |      0.90  1.00  0.95 |      0.52  0.93  0.67
   30% |      0.93  1.00  0.96 |      0.52  0.95  0.68

what the corpus looks like at each rate:
   10%  Full charge- time 90 rninutes, in-gress protcetion IP54.
   30%  Full chargc t-i-me 9O m-inutes, jnqrcss p-rotection I-PS4.

from clean to 30% noise: dense MRR 1.00 -> 0.96 (-4%), BM25 MRR 0.83 -> 0.68 (-18%)
```

Dense retrieval loses 4% of MRR; BM25 loses 18%, and its hit@1 falls from 0.71 to 0.52. The
reason is mechanical: BM25 matches tokens, and `rninutes` is simply not the token `minutes`:
a corrupted term contributes nothing and its IDF weight is lost. Subword embeddings degrade
smoothly instead, because a damaged word still shares most of its pieces with the original.

Note the methodology as much as the result. My first version of this experiment reported
hit@5 and every row read 1.00: over a 14-document corpus, hit@5 is saturated and cannot move.
Choose a metric with headroom for the effect you are trying to see, *before* you run the
experiment ([Chapter 21](21-retrieval-metrics-deep.md)).

What follows for a scanned corpus:

- **Hybrid search leans dense** when OCR quality is poor. If you rely on BM25 for
  identifiers and part numbers, OCR errors hit precisely the tokens you were relying on.
- **Keep per-element confidence** if your OCR provides it (Textract and Document Intelligence
  do). Route low-confidence pages to review or to a second engine; flag answers whose evidence
  came from low-confidence text.
- **Normalise the predictable damage** at ingest: `0`↔`O` and `1`↔`l` inside words,
  de-hyphenate across line breaks, collapse whitespace. Cheap, and it recovers lexical matches.
- **Known traps:** rotated or skewed scans, multi-column pages (OCR reading order is as
  unreliable as §28.3's), tables reduced to a stream of numbers, handwriting, stamps and
  watermarks bleeding into text, and non-Latin scripts needing the right language pack.
- **Re-OCR is a re-ingest.** Engines improve; keep the page images so you can reprocess
  without going back to the source system.

## 28.7 Formulas, code, captions and provenance

- **Formulas** break into gibberish under plain extraction (`x2 + y2 = z2`). If equations
  carry meaning, use a parser that emits LaTeX (Marker, Mistral OCR, or a VLM pass) and index
  both the LaTeX and a verbalisation ("the Pythagorean identity relating x, y and z").
- **Code blocks** must not be reflowed; preserve line breaks and indentation, and keep a whole
  block in one chunk. Code answers are worthless if the indentation is gone.
- **Captions belong to their figure.** "Figure 2: engineering reporting lines" is often the
  only natural-language description of the image in the file, and it is frequently the best
  retrieval handle you have. Bind caption text to the figure element at parse time, by
  proximity, rather than letting it float into the surrounding prose chunk.
- **Cross-references** ("see Section 4.2", "as shown in Figure 2") are dangling pointers after
  chunking. Either resolve them at ingest or accept that a chunk containing one is incomplete.
- **Provenance is not optional.** Carry `doc_id`, `page`, `bbox` and `element_type` on every
  chunk. It is what lets a citation say "page 5" and a reviewer open the page and check:
and it is what lets you answer "where did this number come from?" six months later.

## 28.8 The ingestion contract

The way to keep all of this from collapsing into per-document special cases is to put one
boundary between parsing and everything else:

```python
@dataclass
class Element:
    kind: str          # "text" | "table" | "figure" | "caption" | "formula"
    text: str          # what gets embedded: prose, markdown table, VLM caption, LaTeX
    page: int
    bbox: tuple[float, float, float, float]
    meta: dict         # doc_id, element_id, image_path, ocr_confidence, parser, ...

def parse(path: Path) -> list[Element]: ...
```

Then the rules from this chapter become chunker policy over a typed stream, not a pile of
conditionals in a loop:

```
parse() -> [Element]
     |
     +-- kind=text     -> normal recursive chunking (Chapter 5)
     +-- kind=table    -> atomic; if too big, row-chunks with header repeated (28.4)
     +-- kind=figure   -> VLM caption is the text; image_path + bbox in metadata (28.5)
     +-- kind=caption  -> merged into its figure's element, never orphaned (28.7)
     +-- kind=formula  -> LaTeX + verbalisation, never split (28.7)
                |
                v
         chunks with provenance: doc_id, page, bbox, element_type, parser version
```

Two practical benefits. You can swap `pymupdf` for Docling for one document class without
touching the chunker. And when retrieval quality drops after an ingest change, `parser` and
`parser_version` in the metadata tell you which documents to re-examine: otherwise you are
guessing.

## The interview answer

**Asked: "A PDF has a tree diagram in it. What happens to it: how is it represented and
indexed?"**

"By default, almost nothing happens to it, and that is the dangerous part. A PDF has no
semantic structure (it is positioned glyphs and drawing operators) so the parser
reconstructs everything. For a vector diagram, the box *labels* are real text and they come
out fine: I measured our org chart and got all nine node names. The *edges* are line-drawing
operators, so they contribute nothing. The chunk I index says these nine names appeared near
each other and nothing about who reports to whom. If the figure is a raster image (a
screenshot pasted into the report) I get literally nothing but the caption.

The failure is silent. In my measurement, the text-only index answered 'how many robots are in
Rotterdam' with 47, which it picked up from an unrelated outage postmortem, because the chart
contributed no data and retrieval found the nearest plausible number. So a dropped figure
shows up as a confident wrong answer, not as 'I don't know'.

What I do about it depends on the corpus. The workhorse is: detect figures at parse time,
render the region to PNG, have a vision model caption it, index the caption, and keep the image
path and page in metadata. The key detail is that the caption only contains what you asked
for: so for a hierarchy I prompt for an explicit `parent -> child` edge list rather than
'describe this figure'. With a generic prompt I scored three out of four on structure
questions; with the edge-list prompt, four out of four, and it is stable across runs. If the
corpus is genuinely visual (slides, financial reports) I'd evaluate ColPali-style page
embeddings that skip OCR entirely, against captions, and pick on measured recall and index
cost."

**Asked: "And how do you index tables?"**

"The grid is the meaning: 250 kg means nothing until you know it's the payload row and the A2
column. So the rule is that a table is atomic: the chunker is not allowed to cut it. If it's
bigger than the chunk budget, I split by rows and repeat the title and header into every row
chunk, so each chunk is self-contained. I measured this: while the table fits in one chunk,
markdown, HTML and CSV all tie; the moment a 300-character chunker cuts them, markdown and
HTML lose questions because the header is in another chunk, and a naive flattened table
answered 'which tier includes an on-site spares kit' with Gold instead of Platinum: the value
was retrieved, the row boundary was gone. Row-per-chunk and verbalised rows held at nine out
of nine in both regimes. I also keep the full table as the payload even when I embed a single
row, so the model reads the whole grid."

## Run it

```bash
uv run python code/ch28/make_sample_pdf.py                              # build the fixture
uv run python code/ch28/parser_shootout.py                              # free, no API calls
QDRANT_MODE=memory uv run python code/ch28/tables_and_representations.py   # ~110 LLM calls
QDRANT_MODE=memory uv run python code/ch28/figures_to_text.py           # 4 vision + 12 calls
QDRANT_MODE=memory uv run python code/ch28/ocr_noise.py                 # embeddings only
```

Expected: a six-page PDF in `data/pdfs/`; the parser table and the two reading orders; the
table-representation grid in both chunking regimes; the two caption styles with the 1/4, 3/4,
4/4 scores; and the OCR noise curves. Anything a model generates varies run to run: expect the
split-regime table rows to move by a question, and the caption wording to differ every time.
The *shape* (self-contained chunks ≥ whole-table formats > naive flatten; extracted ≪ generic
≤ structured; dense ≫ BM25 under noise) is what reproduces. Total cost is a few cents.

## Exercises

1. Set `matplotlib.rcParams["pdf.fonttype"] = 3` in `make_sample_pdf.py`, rebuild, and re-run
   `parser_shootout.py`. How much text survives? This is the single most common cause of
   "the parser returns garbage" in the wild.
2. Add a third column to the Atlas table (an "Atlas A3") whose values differ from the A2's,
   and add two questions that require distinguishing them. Which representations still hold at
   a 300-character chunk limit?
3. Crop just the figure's bounding box with `pymupdf` instead of rendering the whole page, and
   re-run `figures_to_text.py`. Does the caption improve, and how many input tokens do you save?
4. Write the `parse() -> list[Element]` function from §28.8 over the fixture PDF, emitting
   `text`, `table` and `figure` elements with page and bbox, then feed it through
   `ragbook.chunk_documents` with a table-aware rule. Compare retrieval with §28.4's winner.
5. Push `ocr_noise.py` to a corpus of 500+ chunks (use `data/gutenberg/` from Chapter 7) and
   re-run. Does dense retrieval still degrade as gracefully when there are more near-miss
   candidates to confuse it?
6. Take a real PDF from your own work. Run `parser_shootout.py` against it and count, by hand
   on three pages, how many facts a text-only index would miss. That number is your business
   case for a figure pipeline.

## Interview questions

**Q: What actually happens to a tree diagram in a PDF when you index it?**
Its labels extract as ordinary text; its structure does not. A vector diagram is line and
curve operators, so parent-child relationships exist only as coordinates: I measured nine node
labels and zero edges from a nine-box org chart. A raster figure yields nothing but the
caption. Unless you deliberately caption or embed the image, the figure contributes no
retrievable facts, and the failure is silent: questions about it get answered from whatever
else was nearby.

**Q: A PDF has text, images and hyperlinks. How do you chunk it?**
Chunking is the last step; parsing decides everything. Use a layout-aware parser to produce
typed elements (text, table, figure, caption) with page and bbox. Chunk prose recursively on
headings; keep tables atomic and repeat the header if you must split by rows; replace figures
with a structured VLM caption plus a pointer to the image; keep anchor text in the chunk and
the URL in metadata. Carry `doc_id`, `page` and `bbox` on every chunk so citations can point at
a page.

**Q: Why not just use `pypdf` for everything?**
Because it returns a string. Without coordinates you cannot detect columns, so multi-column
pages interleave; you cannot detect tables, so grids flatten; you cannot crop or render
figures, so images are invisible. `pypdf` is right for simple single-column digital PDFs and
metadata work. The moment layout carries meaning, you need bounding boxes.

**Q: How do you handle a table that is larger than your chunk size?**
Split by rows, repeating the table title and the header row in every chunk, so each chunk is
self-contained and can be retrieved alone. Store the full table as the payload the model reads,
even though you embedded a row. Never let a generic character splitter cut it: I measured a
flattened table answering "which tier includes an on-site spares kit" with the wrong tier,
because the row boundary was destroyed while the value survived.

**Q: When is OCR necessary, and what does it cost you in retrieval quality?**
Whenever pages are images: scans, photos, screenshots of text. It costs you lexical matching
first: at 30% character noise I measured BM25 MRR down 18% while dense MRR dropped 4%, because
a corrupted token simply fails to match while subword embeddings degrade smoothly. Mitigate
with normalisation of common confusions, de-hyphenation, per-element confidence routing, and by
leaning on the dense side of hybrid for scanned corpora.

**Q: How would you make charts and diagrams searchable?**
Four options: ignore them (the silent default), OCR them (good for text in images, useless for
structure), caption them with a vision model and index the caption (the workhorse), or use
multimodal/late-interaction page embeddings like ColPali that skip OCR. I default to captions
with a prompt that demands the structure I intend to query (an explicit edge list for
hierarchies, label-value lines for charts) and I keep the image path and page in metadata so
answers can show their source.

**Q: Why does the caption prompt matter so much?**
Because a VLM answers your prompt; it does not "extract the figure". "Describe this figure"
produced fluent prose that scored 3/4 on structure questions; a prompt demanding
`parent -> child` lines scored 4/4 and produced identical output across runs. Generic captions
are also longer and vaguer, which hurts embedding precision. Prompt for the shape you will
query, and validate the caption against the figure for a sample.

**Q: How do you know your parser is losing information?**
Measure, do not eyeball. Build a small golden set whose answers live in tables and figures
specifically, and compare retrieval with and without each element type. Also track cheap
signals per document: characters extracted per page, ratio of drawing operators to text,
number of image objects, and pages with near-zero text: a text-free page with twenty drawing
ops is a figure you are dropping.

**Q: A user says an answer cites the wrong page. How do you debug it?**
Provenance first: every chunk should carry `doc_id`, `page`, `bbox` and the parser version, so
you can render the exact region the chunk came from and look at it. Most such bugs are reading
order (a chunk spans two columns or two sections) or a header/footer merged into the body, both
visible immediately once you can draw the bbox on the page.

**Q: What would you use for a corpus of scanned financial reports with lots of tables?**
A hosted document-AI service (Textract, Document Intelligence, Document AI) or a VLM OCR
pipeline, because they give table structure and per-element confidence on scanned pages, and
budget per page accordingly. Then row-level chunks with repeated headers, the full table as
payload, hybrid retrieval leaning dense because OCR noise damages lexical matching, and a
figure-caption pass for the charts. I'd validate on a golden set of table lookups before
committing, since this is the corpus type where parser choice dominates every other decision.

## Key takeaways

- A PDF is positioned glyphs and drawing operators. Paragraphs, reading order, tables and
  figures are reconstructions: different parsers reconstruct differently, and a parser without
  coordinates cannot reconstruct at all.
- A vector diagram gives you its labels and hides its structure; a raster figure gives you
  nothing. Dropping a figure produces confident wrong answers, not refusals: measured: "47"
  for a number that only the chart contained.
- Tables mean nothing without the grid. Keep them atomic; if you must split, repeat the header
  into every row chunk and keep the full table as payload.
- A VLM caption contains exactly what you prompted for. Ask for the structure you intend to
  query (edge lists for hierarchies, label-value lines for charts) and it becomes stable and
  retrievable.
- OCR noise punishes lexical retrieval far more than dense (−18% vs −4% MRR at 30% noise), and
  a saturated metric such as hit@5 on a small corpus will hide the whole effect.

## Next

→ [Chapter 29: Context Construction](29-context-construction.md)

Parsing decided what exists and chunking decided what can be retrieved. Chapter 29 takes the
chunks that survived and asks the last question before generation: what actually goes into the
prompt, in what order, and how much of it.
