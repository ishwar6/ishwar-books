"""Cut highlighted excerpts out of research papers (arXiv PDFs) for the attention series.

For each job: download the PDF, find the given phrases on the page, highlight them in yellow,
and render a crop around them (a paragraph, an equation, a figure or a table) as a PNG.
Usage: python paper_shots.py [part]   ->  public/img/attention/papers/<name>.png"""
import os, sys, urllib.request
import re, unicodedata
import pymupdf as fitz

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'public', 'img', 'attention', 'papers')
CACHE = os.path.expanduser('~/.cache/papers')
YELLOW = (1.0, 0.86, 0.2)


def pdf(arxiv_id):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, arxiv_id.replace('/', '_') + '.pdf')
    if not os.path.exists(path):
        req = urllib.request.Request(f'https://arxiv.org/pdf/{arxiv_id}', headers={'User-Agent': 'Mozilla/5.0'})
        open(path, 'wb').write(urllib.request.urlopen(req).read())
    return fitz.open(path)


def norm(w):
    w = unicodedata.normalize('NFKC', w).lower()
    return re.sub(r'[^0-9a-z%]+', '', w)


def find(pg, phrase, nth=0):
    """Find a phrase in the page's word list, across line breaks. Returns one rectangle per line it covers."""
    words = [(*w[:4], part, *w[5:]) for w in pg.get_text('words')     # x0, y0, x1, y1, word, block, line, word_no
             for part in re.split(r'[-\u2013\u2014/]', w[4]) if part]  # split "encoder-decoder" into two words
    toks = [norm(w[4]) for w in words]
    want = [t for t in (norm(x) for x in re.split(r'[\s\u2013\u2014/-]+', phrase)) if t]
    for i in range(len(toks)):
        j, k, used = i, 0, []
        while j < len(toks) and k < len(want):
            if not toks[j]:
                j += 1; continue
            if toks[j] == want[k]:
                used.append(j); j += 1; k += 1
            elif want[k].startswith(toks[j]) and j + 1 < len(toks) and want[k] == toks[j] + toks[j + 1]:
                used += [j, j + 1]; j += 2; k += 1     # a word hyphenated across lines
            else:
                break
        if k == len(want) and nth > 0:
            nth -= 1
            continue
        if k == len(want):
            lines = {}
            for u in used:
                key = (words[u][5], words[u][6])
                r = fitz.Rect(words[u][:4])
                lines[key] = lines[key] | r if key in lines else r
            return list(lines.values())
    return []


def shot(name, arxiv_id, highlight=(), anchor=None, page=None, above=40, below=40, dpi=190, snap=True, band=None,
         figure=False, column=None, nth=0, hl_nth=None, fig_top=None):
    """Render one highlighted excerpt.
    anchor: phrase that positions the crop (defaults to the first highlight); nth picks a later match on the page.
    above/below: points of context kept around the anchor (text mode).
    figure=True: the anchor is a caption; the crop is the figure graphics above it plus the whole caption.
    column: None = automatic (one column of a two-column paper if the anchor sits in one), or 'full'.
    band: (dy0, dy1) points below the anchor's bottom to mark as a highlighted band (equations, table rows)."""
    doc = pdf(arxiv_id)
    anchor = anchor or highlight[0]
    pages = [page] if page is not None else range(len(doc))
    for pno in pages:
        pg = doc[pno]
        hits = find(pg, anchor, nth)
        if hits:
            break
    else:
        raise SystemExit(f'{name}: anchor not found: {anchor!r}')
    box = fitz.Rect(hits[0])
    for h in hits[1:]:
        box |= h
    W, Hh = pg.rect.width, pg.rect.height
    blocks = [fitz.Rect(b[:4]) for b in pg.get_text('blocks')]
    blk = next((b for b in blocks if b.intersects(fitz.Rect(hits[0]))), box)
    left = [b for b in blocks if b.x1 < 0.53 * W and b.width > 0.3 * W]
    right = [b for b in blocks if b.x0 > 0.47 * W and b.width > 0.3 * W]
    two_col = len(left) >= 2 and len(right) >= 2
    one_col = column != 'full' and two_col and blk.width < 0.6 * W
    if one_col:
        x0, x1 = blk.x0 - 10, blk.x1 + 10
    else:                                             # the page's text width, without the empty side margins
        ws = [w for w in pg.get_text('words') if (w[3] - w[1]) < 3 * (w[2] - w[0])]   # skip arXiv's rotated side stamp
        x0, x1 = max(20, min(w[0] for w in ws) - 10), min(W - 20, max(w[2] for w in ws) + 10)

    found = []
    for phrase in highlight:
        n = (hl_nth or {}).get(phrase, 0)
        rects = find(pg, phrase, n)
        if not rects:
            raise SystemExit(f'{name}: highlight not found on page {pno + 1}: {phrase!r}')
        for r in rects:
            pg.add_highlight_annot(r).set_colors(stroke=YELLOW); found.append(r)
            pg.annots
    if band:
        y0b, y1b = box.y1 + band[0], box.y1 + band[1]
        xs = [w for w in pg.get_text('words') if w[2] > x0 and w[0] < x1 and w[3] > y0b and w[1] < y1b]
        xs = xs or [w for w in pg.get_text('words') if w[2] > x0 and w[0] < x1]
        bx0, bx1 = min(w[0] for w in xs) - 8, max(w[2] for w in xs) + 8
        r = fitz.Rect(bx0, box.y1 + band[0], bx1, box.y1 + band[1])
        pg.draw_rect(r, color=(0.95, 0.7, 0.0), fill=YELLOW, fill_opacity=0.28, width=1.2, overlay=False)

    if figure:
        cap = blk
        art = [fitz.Rect(d['rect']) for d in pg.get_drawings()] + [fitz.Rect(i['bbox']) for i in pg.get_image_info()]
        art = sorted((r for r in art if r.y1 <= cap.y0 + 3 and r.y0 > 60 and r.width > 2 and r.x1 > x0 and r.x0 < x1),
                     key=lambda r: -r.y1)
        top = cap.y0
        for r in (art if fig_top is None else []):
            if r.y1 >= top - 40:
                top = min(top, r.y0)
        # figure labels are text: include text blocks that sit inside the figure area
        for b in (blocks if fig_top is None else []):
            if b.y0 >= top - 2 and b.y1 <= cap.y0 + 1 and b.x1 > x0 and b.x0 < x1:
                top = min(top, b.y0)
        if fig_top is not None:
            top = cap.y0 - fig_top
        changed = fig_top is None
        while changed:                                # short text lines just above the art are figure titles
            changed = False
            for b in blocks:
                if b.y0 < top and b.y1 >= top - 14 and b.height < 30 and b.y1 <= cap.y0:
                    top, changed = b.y0, True
        clip = fitz.Rect(x0, max(0, top - 6), x1, min(Hh, cap.y1 + 4 + below))
    else:
        top, bot = max(0, box.y0 - above), min(Hh, box.y1 + below)
        if snap:
            lines = sorted({(round(r[1], 1), round(r[3], 1)) for r in pg.get_text('words') if r[2] > x0 and r[0] < x1})
            tops = [a for a, b in lines if a >= top - 0.5]
            bots = [b for a, b in lines if b <= bot + 0.5]
            top = min(tops) - 4 if tops else top
            bot = max(bots) + 4 if bots else bot
        clip = fitz.Rect(x0, top, x1, bot)
    os.makedirs(OUT, exist_ok=True)
    pix = pg.get_pixmap(clip=clip, dpi=dpi, annots=True)
    pix.save(os.path.join(OUT, name + '.png'))
    print(f'{name}: page {pno + 1}, {pix.width}x{pix.height}, highlights: {len(found)}')


JOBS = {
    '1': [
        dict(name='bahdanau-bottleneck', arxiv_id='1409.0473', anchor='A potential issue with this encoder',
             highlight=['A potential issue with this encoder', 'compress all the necessary information of a source sentence into a fixed-length vector'],
             above=30, below=60),
        dict(name='bahdanau-weights', arxiv_id='1409.0473', anchor='of each annotation',
             highlight=['of each annotation', 'is computed by'], above=6, below=110, band=(3, 40)),
        dict(name='bahdanau-alignments', arxiv_id='1409.0473', anchor='Four sample alignments found by',
             highlight=['Four sample alignments found by'], figure=True, below=0),
        dict(name='vaswani-scaled-dot', arxiv_id='1706.03762', anchor='We call our particular attention',
             highlight=['We call our particular attention', 'divide each by'], above=6, below=121, band=(86, 118)),
        dict(name='vaswani-why-sqrt', arxiv_id='1706.03762', anchor='We suspect that for large values of',
             highlight=['We suspect that for large values of', 'extremely small gradients'], above=20, below=30),
        dict(name='vaswani-footnote', arxiv_id='1706.03762', anchor='illustrate why the dot products get large',
             highlight=['illustrate why the dot products get large', 'has mean 0 and variance'], above=8, below=22),
        dict(name='vaswani-figure2', arxiv_id='1706.03762', anchor='Figure 2: (left) Scaled Dot-Product Attention',
             highlight=['Figure 2: (left) Scaled Dot-Product Attention'], figure=True, below=0),
        dict(name='vaswani-multihead', arxiv_id='1706.03762', anchor='Multi-head attention allows the model to jointly attend',
             highlight=['Multi-head attention allows the model to jointly attend'], above=8, below=100, band=(34, 80)),
        dict(name='vaswani-table1', arxiv_id='1706.03762', anchor='Table 1: Maximum path lengths', highlight=[],
             above=6, below=110, band=(58, 71)),
        dict(name='vaswani-bleu', arxiv_id='1706.03762', anchor='Table 2: The Transformer achieves better BLEU scores',
             highlight=['Transformer (big)'], above=6, below=165),
        dict(name='vaswani-abstract-result', arxiv_id='1706.03762', page=0, anchor='On the WMT 2014 English-to-French translation task',
             highlight=['establishes a new single-model state-of-the-art BLEU score of 41.8', 'after training for 3.5 days on eight GPUs'], above=30, below=40),
        dict(name='xiao-sinks-figure', arxiv_id='2309.17453', anchor='Visualization of the average attention logits in Llama-2-7B over 256 sentences, each with a length of 16',
             highlight=['Visualization of the average attention logits in Llama-2-7B'], figure=True, below=0),
        dict(name='clark-coref', arxiv_id='1906.04341', anchor='Coreferent mentions attend to their antecedents',
             highlight=['Coreferent mentions attend to their antecedents', 'BERT attention heads that correspond to linguistic phenomena'],
             column='full', above=24, below=275, snap=False),
    ],
    '2': [
        dict(name='shazeer-abstract', arxiv_id='1911.02150', page=0, anchor='We propose a variant called multi-query attention',
             highlight=['often slow, due to the memory-bandwidth cost of repeatedly loading the large',
                        'We propose a variant called multi-query attention, where the keys and values are shared across all of the different attention'],
             above=60, below=40),
        dict(name='shazeer-definition', arxiv_id='1911.02150', anchor='identical except that the different heads share a single set of keys and values',
             highlight=['identical except that the different heads share a single set of keys and values'], above=24, below=40),
        dict(name='shazeer-table1', arxiv_id='1911.02150', anchor='Table 1: WMT14 EN-DE Results', highlight=['Table 1: WMT14 EN-DE Results'], above=6, below=88, band=(56, 69)),
        dict(name='shazeer-table2', arxiv_id='1911.02150', anchor='Table 2: Amortized training and inference costs', highlight=['Table 2: Amortized training and inference costs'], above=6, below=120, band=(70, 83)),
        dict(name='gqa-abstract', arxiv_id='2305.13245', page=0, anchor='We (1) propose a recipe for uptraining',
             highlight=['uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute',
                        'grouped-query attention (GQA), a generalization of multi-query attention which uses an intermediate'], above=10, below=60),
        dict(name='gqa-figure2', arxiv_id='2305.13245', anchor='Overview of grouped-query method', highlight=['Overview of grouped-query method'], figure=True, below=0),
        dict(name='gqa-figure3', arxiv_id='2305.13245', anchor='Uptrained MQA yields a favorable tradeoff', highlight=['Uptrained MQA yields a favorable tradeoff'], figure=True, below=0, fig_top=205),
        dict(name='deepseek-abstract', arxiv_id='2405.04434', page=0, anchor='reduces the KV cache by 93.3%',
             highlight=['saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times'],
             above=40, below=30),
        dict(name='deepseek-figure1', arxiv_id='2405.04434', page=0, anchor='(a) MMLU accuracy vs. activated parameters', highlight=['Training costs and inference efficiency'], figure=True, below=0),
        dict(name='deepseek-figure3', arxiv_id='2405.04434', anchor='Simplified illustration of Multi-Head Attention (MHA)',
             highlight=['Simplified illustration of Multi-Head Attention (MHA)'], figure=True, below=0),
        dict(name='deepseek-mla-equations', arxiv_id='2405.04434', anchor='The core of MLA is the low-rank joint compression for keys and values',
             highlight=['The core of MLA is the low-rank joint compression for keys and values'], above=6, below=112),
        dict(name='deepseek-rope', arxiv_id='2405.04434', anchor='RoPE is incompatible with low-rank KV compression',
             highlight=['RoPE is incompatible with low-rank KV compression', 'we propose the decoupled RoPE strategy'], above=30, below=110),
        dict(name='deepseek-table1', arxiv_id='2405.04434', page=8, anchor='Comparison of the KV cache per token among different attention mechanisms',
             highlight=['MLA (Ours)'], figure=True, below=26, fig_top=118),
        dict(name='roformer-figure1', arxiv_id='2104.09864', anchor='Figure 1: Implementation of Rotary Position', highlight=['Implementation of Rotary Position'], figure=True, below=0),
        dict(name='llama2-gqa', arxiv_id='2307.09288', anchor='used grouped-query attention (GQA) to improve inference scalability',
             highlight=['used grouped-query attention (GQA) to improve inference scalability for our larger models'], above=40, below=30),
    ],
    '3': [
        dict(name='mistral-figure1', arxiv_id='2310.06825', anchor='Sliding Window Attention. The number of operations in vanilla attention',
             highlight=['Hence, after k attention layers, information can move forward by up to k'], figure=True, below=0),
        dict(name='mistral-span', arxiv_id='2310.06825', anchor='we have a theoretical attention span of approximately 131K tokens',
             highlight=['we have a theoretical attention span of approximately 131K tokens'], above=110, below=40),
        dict(name='mistral-rolling', arxiv_id='2310.06825', anchor='Rolling buffer cache. The cache has a fixed size of W = 4',
             highlight=['are stored in position i mod W of the cache'], figure=True, below=0),
        dict(name='mistral-8x', arxiv_id='2310.06825', anchor='this reduces the cache memory usage by 8x',
             highlight=['this reduces the cache memory usage by 8x, without impacting the model quality'], above=56, below=6),
        dict(name='longformer-figure2', arxiv_id='2004.05150', anchor='Comparing the full self-attention pattern and the configuration of attention patterns',
             highlight=['Comparing the full self-attention pattern and the configuration of attention patterns'], figure=True, column='full', below=0),
        dict(name='longformer-receptive', arxiv_id='2004.05150', anchor='the receptive field size at the top layer is',
             highlight=['the receptive field size at the top layer is'], above=40, below=40),
        dict(name='streaming-figure1', arxiv_id='2309.17453', anchor='Illustration of StreamingLLM vs. existing methods',
             highlight=['Illustration of StreamingLLM vs. existing methods'], figure=True, below=0),
        dict(name='streaming-figure3', arxiv_id='2309.17453', anchor='Language modeling perplexity on texts with 20K tokens across various LLM',
             highlight=['Dense attention fails once the input length'], figure=True, below=0),
        dict(name='streaming-four', arxiv_id='2309.17453', anchor='a threshold of four initial tokens appears enough',
             highlight=['a threshold of four initial tokens appears enough'], above=40, below=30),
        dict(name='streaming-figure10', arxiv_id='2309.17453', anchor='Comparison of per-token decoding latency and memory usage',
             highlight=['Comparison of per-token decoding latency and memory usage'], figure=True, below=0),
        dict(name='gemma3-memory', arxiv_id='2503.19786', page=0, anchor='long context is the memory explosion of the KV cache during inference',
             highlight=['long context is the memory explosion of the KV cache during inference'], above=30, below=24),
        dict(name='gemma3-rope', arxiv_id='2503.19786', anchor='We increase RoPE base frequency from 10k to 1M on global self-attention layers',
             highlight=['We increase RoPE base frequency from 10k to 1M on global self-attention layers, and keep the frequency of the local'], above=40, below=30),
        dict(name='gemma3-figure3', arxiv_id='2503.19786', anchor='Impact of Local:Global ratio on the perplexity',
             highlight=['The impact is minimal, even with 7-to-1 local to global'], figure=True, below=0),
        dict(name='gemma3-figure6', arxiv_id='2503.19786', anchor='KV cache memory versus context length. We show the memory usage',
             highlight=['KV cache memory versus context length'], figure=True, below=0, fig_top=106),
        dict(name='gemma2-alternate', arxiv_id='2408.00118', anchor='in every other layer. The sliding window size of local attention layers is set to 4096 tokens',
             highlight=['in every other layer. The sliding window size of local attention layers is set to 4096 tokens'], above=40, below=20),
        dict(name='dsv32-indexer', arxiv_id='2512.02556', anchor='The lightning indexer computes the index score',
             highlight=['The lightning indexer computes the index score'], above=30, below=160),
        dict(name='dsv32-figure2', arxiv_id='2512.02556', anchor='Attention architecture of DeepSeek-V3.2, where DSA is instantiated under MLA',
             highlight=['The green part illustrates how DSA selects the top-k key-value entries'], figure=True, below=32),
        dict(name='dsv32-complexity', arxiv_id='2512.02556', anchor='still has a complexity of',
             highlight=['still has a complexity of'], above=50, below=40),
        dict(name='dsv32-figure3', arxiv_id='2512.02556', anchor='Inference costs of DeepSeek-V3.1-Terminus and DeepSeek-V3.2 on H800 clusters',
             highlight=['Inference costs of DeepSeek-V3.1-Terminus and DeepSeek-V3.2'], figure=True, below=0),
    ],
}

if __name__ == '__main__':
    parts = sys.argv[1:] or list(JOBS)
    only = os.environ.get('ONLY')
    for p in parts:
        for job in JOBS[p]:
            if not only or job['name'] in only.split(','):
                shot(**job)
