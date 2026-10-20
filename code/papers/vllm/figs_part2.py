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


F['p2_chain'] = chain()
F['p2_matmul'] = matmul()
F['p2_batching'] = batching()
F['p2_fig3'] = figure3()
F['p2_sharing'] = sharing()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part2.json'), 'w'), indent=1)
print('figures:', ', '.join(F))
