"""Figures for Part 2 (why memory is the bottleneck). Writes results/figs_part2.json. CSS classes only.
Numbers come from results/part2.json (our run) and from the paper's Figure 3."""
import json, os
from figlib import svg, text, box, arrow, lines_chart
from samfig import token

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part2.json')))
F = {}


def chain():
    """Eq. 1 on our sentence: each token with its probability given the tokens before it."""
    words, probs = R['chain']['words'], R['chain']['probs']
    W, H, x0, w, gap = 720, 210, 20, 80, 6
    b = [text(20, 26, 'Eq. 1 on a real sentence (OPT-125M): one factor per token', 't-title')]
    for i, (s, p) in enumerate(zip(words, probs)):
        x = x0 + i * (w + gap)
        b += token(x, 50, w, s.strip(), 'box-1', 32)
        b.append(text(x + w / 2, 104, f'{p:.1e}', 't-mono', 'middle'))
    b += [text(x0, 84 + 50, 'probability of each token, given all the tokens to its left', 't-tick'),
          text(x0, 168, f'P(sentence) = product of the 8 numbers = {R["chain"]["product"]:.2e}', 't-mono t-strong'),
          text(x0, 192, 'One pass over the sentence gives all 8 numbers at once; writing new text gives one per pass.', 't-tick')]
    return svg(W, H, 'Eight tokens of the sentence Four score and seven years ago our fathers, each with the probability '
               'that OPT-125M gives it after the tokens before it. Their product is the probability of the sentence.', b)


def matmul():
    ns = sorted(int(k) for k in R['matmul'])
    ms = [R['matmul'][str(n)]['ms'] for n in ns]
    return lines_chart(720, 300, 'Time of one 5120 by 5120 matrix times n token vectors on a laptop GPU: '
                       + ', '.join(f'{v:.2f} ms' for v in ms) + ' for n = ' + ', '.join(map(str, ns)) + '.',
                       ns, [('time', ms, 'l1', 's1')], 'tokens multiplied together, n (log scale)',
                       'milliseconds', 0, 2.0, [0, 0.5, 1.0, 1.5, 2.0], xlog=True, fmt=lambda v: f'{v:.1f}')


def batching():
    """The first 520 steps of our batching simulation: 8 slots, one row each, for both schemes."""
    L, gap, slots, last = R['batching']['lengths'], R['batching']['gap'], R['batching']['slots'], 520
    arrive = [i * gap for i in range(len(L))]
    W, H, x0, rh = 720, 424, 110, 13
    sx = (W - x0 - 20) / last
    b = [text(20, 26, 'Eight slots over the first 520 steps (our simulation, Alpaca output lengths)', 't-title')]
    leg = [('s1', 'writes a token'), ('waste', 'padding: request done, slot still held')]
    lx = 20
    for cls, lab in leg:
        b += [f'<rect class="{cls}" x="{lx}" y="40" width="12" height="12" rx="2"/>', text(lx + 18, 50, lab, 't-tick')]
        lx += 40 + 7 * len(lab)

    def bar(row, y0, t0, t1, cls):
        t0, t1 = max(t0, 0), min(t1, last)
        if t1 > t0:
            b.append(box(x0 + t0 * sx, y0 + row * rh, (t1 - t0) * sx, rh - 3, cls, 2))

    # request-level: take the waiting requests as one batch, run until the longest ends
    y0, t, i, started = 80, 0, 0, 0
    b.append(text(x0 - 10, y0 + 4 * rh, 'request-level', 't-tick t-strong', 'end'))
    while i < len(L) and t < last:
        t = max(t, arrive[i])
        batch = [j for j in range(i, len(L)) if arrive[j] <= t][:slots]
        i += len(batch)
        steps = max(L[j] for j in batch)
        started += len(batch)
        for row, j in enumerate(batch):
            bar(row, y0, t, t + L[j], 's1')
            bar(row, y0, t + L[j], t + steps, 'waste')
        t += steps
    # iteration-level: a finished request frees its slot at once; the next waiting request takes it
    b.append(text(x0, y0 + slots * rh + 16, f'{started} requests started by step {last}', 't-tick'))
    y1 = 80 + slots * rh + 56
    b.append(text(x0 - 10, y1 + 4 * rh, 'iteration-level', 't-tick t-strong', 'end'))
    free, started = [0] * slots, 0
    for j in range(len(L)):
        row = min(range(slots), key=lambda s: free[s])
        start = max(free[row], arrive[j])
        if start >= last:
            break
        bar(row, y1, start, start + L[j], 's1')
        free[row] = start + L[j]
        started += 1
    b.append(text(x0, y1 + slots * rh + 16, f'{started} requests started by step {last}', 't-tick'))
    ya = y1 + slots * rh + 44
    for s in range(0, last + 1, 100):
        b.append(text(x0 + s * sx, ya, str(s), 't-tick', 'middle'))
    b.append(text(x0 + last * sx / 2, ya + 20, 'step', 't-tick', 'middle'))
    return svg(W, H, 'Two panels of eight rows. In request-level batching, each batch runs until its longest request '
               'ends, so many slots sit idle as padding. In iteration-level batching, a slot is refilled as soon as its '
               'request ends.', b)


def figure3():
    """The two chunks of Figure 3 drawn to scale: the slots that ever hold a token are a sliver."""
    W, H, x0, bw = 720, 230, 20, 680
    sx = bw / 2048
    b = [text(20, 26, 'Figure 3 to scale: the two chunks, one slot = one token', 't-title')]
    for k, (name, y) in enumerate((('A', 60), ('B', 140))):
        s = R['fig3'][name]
        n = sum(s.values())
        b.append(text(x0, y - 8, f'request {name}: chunk of {n:,} slots', 't-tick t-strong'))
        x = x0
        for cls, key in (('s1', 'token states'), ('s2', 'reserved'), ('waste', 'never used')):
            w = s[key] * sx
            b.append(box(x, y, max(w, 1.5), 28, cls, 2))
            x += w
        b.append(text(x0, y + 46, f'{s["token states"]} hold a token, {s["reserved"]} reserved, '
                      f'{s["never used"]:,} never used', 't-tick'))
    b.append(text(x0, 222, 'The coloured sliver at the left of each bar is all the memory these requests ever need.', 't-tick'))
    return svg(W, H, 'Two bars drawn to scale. Request A books 2,048 slots and request B books 512. The slots that hold '
               'or will hold a token are a thin sliver at the left; the rest is never used.', b)


def sharing():
    """Parallel sampling: the library keeps one copy of the prompt cache per sample; sharing would keep one."""
    P, T = R['sampling']['prompt'], R['sampling']['cache_tokens']
    W, H, x0 = 720, 250, 20
    pw, ow = 120, 200
    b = [text(20, 26, f'Four samples from one prompt of {P} tokens', 't-title'),
         text(x0, 54, 'what model.generate keeps', 't-tick t-strong'),
         text(400, 54, 'what sharing would keep', 't-tick t-strong')]
    for s in range(4):
        y = 66 + s * 40
        b += token(x0, y, pw, 'prompt', 'box-1', 30)
        b += token(x0 + pw + 4, y, ow - 60, f'sample {s + 1}', 'box-2', 30)
        b += token(600, y, ow - 60, f'sample {s + 1}', 'box-2', 30)
        b.append(arrow(522, 141, 596, y + 15))
    b += token(400, 126, pw, 'prompt', 'box-1', 30)
    b.append(text(x0, 240, f'4 copies of the prompt: {100 * R["sampling"]["dup_share"]:.0f}% of the cache holds '
                  'duplicates. One shared copy is enough.', 't-tick'))
    return svg(W, H, 'Left: four rows, each with its own copy of the prompt cache and its own sample. Right: one prompt '
               'cache that all four samples point to.', b)



def arch():
    """The vLLM system (paper Figure 4) as a "you are here" map. Part 2 lights up the scheduler."""
    W, H = 720, 300
    b = [text(20, 26, 'vLLM, the whole system (paper Figure 4); this part is about the scheduler', 't-title')]
    # requests on the left
    b.append(box(20, 70, 110, 180, 'box', 10))
    b.append(text(75, 92, 'Requests', 't-title', 'middle'))
    for k in range(3):
        b += token(32, 108 + k * 44, 86, f'request {k + 1}', 'box-1', 30)
    b.append(arrow(132, 160, 166, 160))
    # scheduler with the KV cache manager inside
    b.append(box(170, 50, 280, 210, 'box-on', 12))
    b.append(text(310, 74, 'Scheduler', 't-title', 'middle'))
    b.append(text(310, 92, 'picks which requests run in each step', 't-tick', 'middle'))
    b.append(box(188, 104, 244, 144, 'box', 10))
    b.append(text(310, 124, 'KV cache manager', 't-title', 'middle'))
    for k, lab in enumerate(('Block tables', 'GPU block allocator', 'CPU block allocator')):
        b += token(200, 136 + k * 36, 220, lab, 'box-2', 28)
    # workers on the right
    b.append(arrow(452, 150, 488, 96))
    b.append(arrow(452, 170, 488, 222))
    for y, name in ((50, 'Worker 0'), (180, 'Worker N-1')):
        b.append(box(490, y, 210, 80, 'box', 10))
        b.append(text(595, y + 22, name, 't-title', 'middle'))
        b += token(500, y + 36, 92, 'Cache engine', 'box-3', 30)
        b += token(598, y + 36, 92, 'Model shard', 'box-3', 30)
    b.append(text(595, 160, '...', 't-title', 'middle'))
    b.append(text(20, 288, 'Part 2: why every step must batch many requests, and why KV cache memory limits that batch.', 't-tick'))
    return svg(W, H, 'The vLLM system. Requests enter on the left. In the centre, the scheduler, highlighted for this part, '
               'holds the KV cache manager with block tables, a GPU block allocator and a CPU block allocator. On the right, '
               'workers 0 to N-1 each have a cache engine and a model shard.', b)


def life():
    """A request's life: one prompt pass fills 7 cache slots, then each generation step adds one slot."""
    prompt = ['Four', 'score', 'and', 'seven', 'years', 'ago', 'our']
    outs = ['fathers', 'brought', 'forth', '<eos>']
    W, H, x0, cw, gap = 720, 300, 128, 45, 3
    b = [text(20, 26, 'One request, from prompt to <eos>: the KV cache grows by one slot per step', 't-title')]
    b += [box(20, 40, 12, 12, 'box-on', 2), text(38, 50, 'computed in this pass', 't-tick'),
          box(190, 40, 12, 12, 'box-1', 2), text(208, 50, 'read from the KV cache', 't-tick'),
          text(x0 + 10 * (cw + gap) + 8, 50, 'writes', 't-tick t-strong')]
    rows = [('prompt phase', '1 pass, 7 tokens', 7, 0)] + [(f'generation {k}', '1 pass, 1 token', 7 + k, 7 + k - 1)
                                                         for k in (1, 2, 3)]
    words = prompt + outs[:3]
    for r, (lab, sub, n, first_new) in enumerate(rows):
        y = 70 + r * 50
        b.append(text(20, y + 13, lab, 't-tick t-strong'))
        b.append(text(20, y + 28, sub, 't-tick'))
        for c in range(10):
            x = x0 + c * (cw + gap)
            if c < n:
                b += token(x, y, cw, words[c], 'box-on' if c >= first_new else 'box-1', 30)
            else:
                b.append(box(x, y, cw, 30, 'box-ghost', 6))
        xo = x0 + 10 * (cw + gap) + 8
        b.append(text(xo, y + 20, outs[r] + (', stop' if r == 3 else ''), 't-mono'))
    b.append(text(20, 280, 'The prompt phase computes 7 slots at once. Each generation step computes 1 new slot and '
                  're-reads all the earlier ones.', 't-tick'))
    return svg(W, H, 'Four rows, one per pass of the model. The prompt phase computes the keys and values of all 7 prompt '
               'tokens at once and writes fathers. Each of the next three passes computes one new slot, reads the earlier '
               'slots, and writes brought, forth and then the end token.', b)


def mv():
    """Matrix-vector against matrix-matrix: the same weight read, 1 or 8 tokens of work."""
    W, H = 720, 250
    b = [text(20, 26, 'The weight matrix is read once per pass, however many tokens use it', 't-title')]
    for k, (x, n, lab) in enumerate(((20, 1, 'generation, 1 request'), (360, 8, 'a batch of 8 requests'))):
        b.append(text(x, 54, lab, 't-tick t-strong'))
        b.append(box(x, 66, 120, 120, 'box-2', 4))
        b.append(text(x + 60, 130, 'W', 't-big', 'middle'))
        b.append(text(x + 60, 204, 'd by d weights', 't-tick', 'middle'))
        b.append(text(x + 136, 130, 'x', 't-title', 'middle'))
        for j in range(n):
            b.append(box(x + 150 + j * 10, 66, 7, 120, 'box-1', 2))
        xe = x + 150 + n * 10 + 6
        b.append(text(xe + 4, 130, '=', 't-title'))
        for j in range(n):
            b.append(box(xe + 20 + j * 10, 66, 7, 120, 'box-on', 2))
        b.append(text(x, 230, f'reads W once, does {n} token{"s" if n > 1 else ""} of work', 't-tick'))
    return svg(W, H, 'Left: the weight matrix W times one token vector gives one output vector. Right: the same matrix '
               'times eight token vectors side by side gives eight outputs. Both read W once from memory; the right does '
               'eight times the work.', b)


def batch_toy():
    """Four requests, two slots: request-level against iteration-level batching, step by step."""
    W, H, x0, cw, rh = 720, 340, 140, 60, 30
    # (start, length) per slot row, worked out by hand from the rules in the text
    req = [('A', 0, 2), ('B', 0, 5), ('C', 1, 2), ('D', 2, 3)]
    b = [text(20, 26, 'Four requests, two slots: who runs at each step', 't-title')]
    b.append(text(20, 58, 'step', 't-tick t-strong'))
    b.append(text(20, 80, 'arrives', 't-tick t-strong'))
    for s in range(8):
        cx = x0 + s * cw + cw / 2
        b.append(text(cx, 58, str(s), 't-tick', 'middle'))
    for s, who in ((0, 'A, B'), (1, 'C'), (2, 'D')):
        b.append(text(x0 + s * cw + cw / 2, 80, who, 't-on', 'middle'))
    panels = [('request-level', [[('A', 0, 2), ('pad', 2, 5), ('C', 5, 7), ('pad', 7, 8)],
                                 [('B', 0, 5), ('D', 5, 8)]],
               '8 steps; 4 slot-steps of padding; C waits 4 steps, D waits 3'),
              ('iteration-level', [[('A', 0, 2), ('C', 2, 4), ('D', 4, 7)], [('B', 0, 5)]],
               '7 steps; no padding; C waits 1 step, D waits 2')]
    cls = {'A': 'box-1', 'B': 'box-2', 'C': 'box-3', 'D': 'box-4', 'pad': 'waste'}
    for p, (name, rows, summary) in enumerate(panels):
        y0 = 104 + p * 122
        b.append(text(20, y0 + 12, name, 't-tick t-strong'))
        for r, segs in enumerate(rows):
            y = y0 + 20 + r * (rh + 4)
            b.append(text(x0 - 8, y + 19, f'slot {r + 1}', 't-tick', 'end'))
            for s in range(8):
                b.append(box(x0 + s * cw + 2, y, cw - 4, rh, 'box-ghost', 5))
            for who, t0, t1 in segs:
                for s in range(t0, t1):
                    b += token(x0 + s * cw + 2, y, cw - 4, who, cls[who], rh, rx=5)
        b.append(text(x0, y0 + 108, summary, 't-tick'))
    return svg(W, H, 'Two grids of two slots over eight steps. Request-level: A and B run together until B ends at step 5, '
               'so A leaves three padding steps; C and D then wait until step 5. Iteration-level: C takes A\'s slot at '
               'step 2 and D takes it at step 4, so the work ends one step sooner with no padding.', b)


def alloc():
    """A 16-unit memory under a first-fit allocator, then the buddy allocator rounding 5 up to 8."""
    W, H, x0, cw = 720, 334, 150, 34
    cls = {'A': 'box-1', 'B': 'box-2', 'C': 'box-3', 'D': 'box-4', 'E': 'box-on', '.': 'box-ghost', 'x': 'waste'}
    b = [text(20, 26, 'A 16-unit memory under a simple allocator', 't-title')]

    def strip(y, cells, label):
        b.append(text(20, y + 19, label, 't-tick t-strong'))
        for c, ch in enumerate(cells):
            b.extend(token(x0 + c * cw, y, cw - 3, '' if ch in '.x' else ch, cls[ch], 28, rx=4))

    strip(48, 'AAAABBBCCCCCDD..', '1. A to D get room')
    strip(104, 'AAAA...CCCCC....', '2. B and D leave')
    b.append(text(x0 + 5.5 * cw, 150, '3 free', 't-tick', 'middle'))
    b.append(text(x0 + 13.5 * cw, 150, '4 free', 't-tick', 'middle'))
    strip(172, 'EEEEE', '3. E asks for 5')
    b.append(text(x0 + 5 * cw + 10, 191, 'no gap holds 5 in a row', 't-bad'))
    b.append(text(x0, 220, '7 units are free, but E does not fit: external fragmentation', 't-tick'))
    strip(242, 'EEEEExxx', '4. buddy: 5 -> 8')
    b.append(text(x0 + 8 * cw + 10, 261, '3 units never used', 't-bad'))
    b.append(text(x0, 290, 'the chunk is rounded up to a power of two: internal fragmentation', 't-tick'))
    b.append(text(20, 318, 'Each unit is one slot of memory. Gaps appear when programs of different sizes come and go.',
                  't-tick'))
    return svg(W, H, 'Row 1: programs A to D fill 14 of 16 units. Row 2: B and D leave, which leaves gaps of 3 and 4 units. '
               'Row 3: E needs 5 units in a row and does not fit, although 7 units are free. Row 4: a buddy allocator '
               'rounds a request of 5 up to 8, so 3 units sit unused inside the chunk.', b)


def before_after():
    """Three requests in 40 slots: one booked chunk each, against blocks of 4 given on demand."""
    W, H, x0, cw = 720, 320, 40, 15
    pos = lambda c: x0 + c * cw + (c // 4) * 4                       # a small gap after every 4 slots
    b = [text(20, 26, 'Three requests in 40 slots of KV cache memory', 't-title')]
    leg = [('s1', 'holds a token'), ('s2', 'reserved for a later token'), ('waste', 'never used or unusable'),
           ('box-ghost', 'free')]
    lx = 20
    for c, lab in leg:
        b += [box(lx, 40, 12, 12, c, 2), text(lx + 18, 50, lab, 't-tick')]
        lx += 36 + 7 * len(lab)
    # before: chunks of 12 for A, B, C and two gaps of 2 left by earlier requests
    before = 'TTTTTRRNNNNN' + 'GG' + 'TTTRRRRNNNNN' + 'GG' + 'TTTTRNNNNNNN'
    names = [('A', 0), ('B', 14), ('C', 28)]
    after = 'AAAA' + 'BBBe' + '....' + 'Aeee' + '....' + 'CCCC' + '.' * 16
    assert len(before) == len(after) == 40
    cmap = {'T': 's1', 'R': 's2', 'N': 'waste', 'G': 'waste', 'e': 'waste', '.': 'box-ghost', 'A': 's1', 'B': 's1',
            'C': 's1'}
    for k, (title, cells, y) in enumerate((('before vLLM: one chunk of 12 slots per request, booked at the start', before, 98),
                                           ('with vLLM: blocks of 4 slots, taken one at a time, anywhere', after, 212))):
        b.append(text(20, y - 10, title, 't-tick t-strong'))
        for c, ch in enumerate(cells):
            b.append(box(pos(c), y, cw - 2, 30, cmap[ch], 3))
        if k == 0:
            for n, c in names:
                b.append(text(pos(c) + 2, y + 48, n, 't-on'))
            b.append(text(pos(12), y + 48, 'gap', 't-tick'))
            b.append(text(pos(26), y + 48, 'gap', 't-tick'))
        else:
            for n, c in (('A', 0), ('B', 4), ('A', 12), ('C', 20)):
                b.append(text(pos(c) + 2, y + 48, n, 't-on'))
    b.append(text(20, 170, 'tokens 12, reserved 7, never used 17, gaps 4: a request of 12 more slots does not fit', 't-tick'))
    b.append(text(20, 284, 'tokens 12, empty slots in last blocks 4, free 24 (6 blocks): new requests can join', 't-tick'))
    b.append(text(20, 308, 'Part 3 explains how vLLM finds the scattered blocks of one request again.', 't-tick'))
    return svg(W, H, 'Top: before vLLM, requests A, B and C each book a chunk of 12 slots; most slots are reserved or never '
               'used, and two small gaps stay free but useless. Bottom: with vLLM, each request takes blocks of 4 slots '
               'only when it needs them, so 24 slots stay free for new requests.', b)

F['p2_chain'] = chain()
F['p2_matmul'] = matmul()
F['p2_batching'] = batching()
F['p2_fig3'] = figure3()
F['p2_sharing'] = sharing()
F['p2_arch'] = arch()
F['p2_life'] = life()
F['p2_mv'] = mv()
F['p2_batch_toy'] = batch_toy()
F['p2_alloc'] = alloc()
F['p2_before_after'] = before_after()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part2.json'), 'w'), indent=1)
print('figures:', ', '.join(F))
