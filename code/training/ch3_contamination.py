"""Chapter 3, experiment 5: contamination (test questions inside the training data).
A. The GPT-3 style n-gram check. "Training corpus" = the WikiText-103 training split (about 100 million words).
   Normalise words (lower case, no punctuation), take every 13-word window of each benchmark item, and look for it
   in the corpus. Then plant 20 GSM8K test questions in the corpus (10 copied exactly, 10 lightly reworded) and check
   which ones the method finds.
B. A memorisation probe on the real model: give Qwen2.5-0.5B base the first half of a text and let it continue greedily.
   If it reproduces the next 8 words exactly, it has very probably seen that text. Compared on several sources."""
import json, re, time
import torch
from common import Log, save
from ch3_common import BASE, DEV, data, load, sample_rows, generate, chat
from huggingface_hub import hf_hub_download
import pandas as pd

log = Log('ch3_contamination')
N_GRAM = 13


def words(s):
    return [w for w in re.sub(r'[^0-9a-z ]+', ' ', s.lower()).split() if w]


def grams(ws, n=N_GRAM):
    return {' '.join(ws[i:i + n]) for i in range(len(ws) - n + 1)} if len(ws) >= n else {' '.join(ws)}


# ---------------------------------------------------------------- A. n-gram overlap
gsm = sample_rows(data('gsm8k_test'), 300, seed=0)
mmlu = sample_rows(data('mmlu_test'), 400, seed=0)
items = {('gsm8k', i): q for i, q in enumerate(gsm['question'])}
items.update({('mmlu', i): q + ' ' + ' '.join(map(str, c)) for i, (q, c) in enumerate(zip(mmlu['question'], mmlu['choices']))})
wiki2 = [t for t in data('wikitext2_test')['text'] if len(t.split()) > 40]
wiki_items = sample_rows(pd.DataFrame({'t': wiki2}), 300, seed=0)['t']
items.update({('wikitext2-test', i): t for i, t in enumerate(wiki_items)})

PLANT_EXACT = list(gsm['question'][:10])
SWAPS = [('How many', 'What number of'), ('how many', 'what number of'), (' each ', ' every '), ('total', 'overall'),
         (' buys ', ' purchases '), (' has ', ' owns '), ('week', 'seven-day period'), (' makes ', ' earns ')]


def reword(q):
    """A light paraphrase: a few word swaps and one number written as a word. Meaning unchanged."""
    for a, b in SWAPS:
        q = q.replace(a, b)
    return re.sub(r'\b2\b', 'two', q, count=1)


PLANT_REWORD = [reword(q) for q in gsm['question'][10:20]]


def paraphrases(qs):
    """A real paraphrase, written by Qwen2.5-1.5B-Instruct (greedy), cached in results/ch3_paraphrases.json."""
    path = 'results/ch3_paraphrases.json'
    try:
        return json.load(open(path))
    except FileNotFoundError:
        tok, model = load('Qwen/Qwen2.5-1.5B-Instruct')
        ask = ('Rewrite the following maths word problem in completely different words and sentence structure. '
               'Keep every number, every fact and the question the same. Output only the rewritten problem.\n\n')
        out = generate(tok, model, [chat(tok, ask + q) for q in qs], max_new_tokens=200, batch=10)
        json.dump(out, open(path, 'w'), indent=1)
        del model
        return out


PLANT_PARA = paraphrases(list(gsm['question'][20:30]))
log('== A. 13-gram overlap with the WikiText-103 training split ==')
want = {}
for key, text in items.items():
    for g in grams(words(text)):
        want.setdefault(g, set()).add(key)
log(f'benchmark items: {len(items)} ({sum(k[0] == "gsm8k" for k in items)} GSM8K, {sum(k[0] == "mmlu" for k in items)} MMLU, '
    f'{sum(k[0] == "wikitext2-test" for k in items)} WikiText-2 test paragraphs); distinct 13-grams: {len(want):,}')
t0 = time.time()
hits = {}
corpus_words = 0
files = [f'wikitext-103-raw-v1/train-0000{i}-of-00002.parquet' for i in range(2)]
docs = []
for f in files:
    docs += list(pd.read_parquet(hf_hub_download('Salesforce/wikitext', f, repo_type='dataset'))['text'])
docs += PLANT_EXACT + PLANT_REWORD + PLANT_PARA                       # the planted leak
buf = []
for line in docs:
    ws = words(line)
    corpus_words += len(ws)
    for i in range(len(ws) - N_GRAM + 1):
        g = ' '.join(ws[i:i + N_GRAM])
        if g in want:
            for key in want[g]:
                hits.setdefault(key, set()).add(g)
log(f'corpus: {corpus_words:,} words (WikiText-103 train + 30 planted lines), scanned in {time.time() - t0:.0f}s')
res = {'corpus_words': corpus_words, 'A': {}}
PLANTED = set(range(30))                                             # GSM8K items 0-29 were planted (below)
for src in ['gsm8k', 'mmlu', 'wikitext2-test']:
    keys = [k for k in items if k[0] == src and not (src == 'gsm8k' and k[1] in PLANTED)]
    flagged = [k for k in keys if k in hits]
    res['A'][src] = dict(n=len(keys), flagged=len(flagged), examples=[sorted(hits[k])[0] for k in flagged[:3]])
    log(f'{src:15s} {len(flagged):3d} of {len(keys)} items share at least one 13-gram with the corpus' +
        (' (the 30 planted questions not counted)' if src == 'gsm8k' else ''))
    for k in flagged[:2]:
        log(f'    e.g. {src} #{k[1]}: "{sorted(hits[k])[0]}"')


def surviving(orig, variant):
    go = grams(words(orig))
    gv = grams(words(variant))
    return len(go & gv), len(go)


for name, rng_, texts in [('copied exactly', range(0, 10), PLANT_EXACT), ('lightly reworded', range(10, 20), PLANT_REWORD),
                          ('paraphrased by a model', range(20, 30), PLANT_PARA)]:
    found = sum(('gsm8k', i) in hits for i in rng_)
    sv = [surviving(gsm['question'][i], t) for i, t in zip(rng_, texts)]
    frac = sum(a for a, b in sv) / sum(b for a, b in sv)
    res['A'][name] = dict(found=found, n=10, windows_kept=frac)
    log(f'planted, {name:23s}: found {found} of 10   (share of the original 13-grams still present: {100 * frac:.0f}%)')
log(f'    original   : {gsm["question"][10]}')
log(f'    reworded   : {PLANT_REWORD[0]}')
res['A']['reword_example'] = [gsm['question'][10], PLANT_REWORD[0]]
nums = lambda t: sorted(re.findall(r'\d+(?:\.\d+)?', t))
faithful = [j for j in range(10) if nums(gsm['question'][20 + j]) == nums(PLANT_PARA[j])]
found_f = sum(('gsm8k', 20 + j) in hits for j in faithful)
res['A']['para_faithful'] = dict(n=len(faithful), found=found_f)
log(f'the paraphrasing model kept exactly the same numbers in {len(faithful)} of 10; of those, {found_f} were found')
j = faithful[0]
log(f'    original   : {gsm["question"][20 + j]}')
log(f'    paraphrase : {PLANT_PARA[j]}')
res['A']['para_example'] = [gsm['question'][20 + j], PLANT_PARA[j]]

# ---------------------------------------------------------------- B. memorisation probe
log('')
log('== B. memorisation probe: first half in, does the model write the next 8 words exactly? ==')
tok, model = load(BASE)


@torch.no_grad()
def probe(texts, n_words=8):
    exact, tot = 0, 0
    examples = []
    for t in texts:
        ws = t.split()
        if len(ws) < 24:
            continue
        cut = len(ws) // 2
        prefix, target = ' '.join(ws[:cut]), ws[cut:cut + n_words]
        x = tok(prefix, return_tensors='pt')['input_ids'].to(DEV)
        g = model.generate(x, max_new_tokens=24, do_sample=False, temperature=None, top_p=None, top_k=None,
                           pad_token_id=tok.eos_token_id)
        cont = tok.decode(g[0, x.shape[1]:]).split()[:n_words]
        ok = [w.strip('.,;:') for w in cont] == [w.strip('.,;:') for w in target]
        exact += ok; tot += 1
        if ok and len(examples) < 2:
            examples.append((prefix[-80:], ' '.join(target)))
    return exact, tot, examples


wiki_long = [t.strip() for t in data('wikitext2_test')['text'] if len(t.split()) > 60]
sources = {
    'GSM8K test questions': list(sample_rows(data('gsm8k_test'), 300, seed=2)['question']),
    'GSM8K train questions': list(sample_rows(data('gsm8k_train'), 300, seed=2)['question']),
    'MMLU test questions': list(sample_rows(data('mmlu_test'), 1500, seed=2)['question']),
    'WikiText-2 test paragraphs': list(sample_rows(pd.DataFrame({'t': wiki_long}), 300, seed=2)['t']),
    'HumanEval prompts': list(data('humaneval')['prompt']),
}
res['B'] = {}
try:                                    # part B is slow and deterministic (greedy): reuse an earlier run if there is one
    old = json.load(open('results/ch3_contamination.json')).get('B', {})
except FileNotFoundError:
    old = {}
for name, texts in sources.items():
    e, n, ex = (old[name]['exact'], old[name]['n'], old[name]['examples']) if name in old else probe(texts)
    res['B'][name] = dict(exact=e, n=n, examples=ex)
    log(f'{name:28s} {e:3d} of {n:3d} continued exactly ({100 * e / max(n, 1):.1f}%)')
    for p, t in ex[:1]:
        log(f'    ...{p!r} -> {t!r}')
save('ch3_contamination', res)
