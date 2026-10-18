"""Part 3: the two text-classification harnesses that Meta-Harness discovered (Appendix B.1), rebuilt around a small
local model and run on Symptom2Disease, one of the paper's three datasets.

What it shows (each block writes its own log in results/):
  part3_data    the dataset, the memory / search / test split, and the 22 labels
  part3_tfidf   TF-IDF retrieval two ways: Method 1 from scratch, Method 2 scikit-learn, and a check that they agree
  part3_greedy  one model call two ways: Method 1 our own greedy decoding loop, Method 2 model.generate
  part3_prompts one real prompt from each harness (zero-shot, few-shot, draft-verification, label-primed query)
  part3_eval    accuracy and context size of every harness on the search split and on the held-out test split
  part3_pareto  the accuracy / context Pareto frontier two ways: a sweep from scratch and a brute-force check
  part3_verify  what the verification call of draft-verification changed
  part3_labels  test accuracy per disease, zero-shot against label-primed query

Model: Qwen/Qwen2.5-1.5B-Instruct, frozen, greedy decoding, Apple-silicon GPU (mps). Predictions are cached in
results/part3_preds.json so a second run only re-prints the numbers."""
import collections, difflib, json, math, os, random, re, sys, time
import numpy as np

if os.environ.get('MHLIB'):   # a folder with scikit-learn, if it is not installed in your environment
    sys.path.append(os.environ['MHLIB'])
from common import Log, save, RESULTS

MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
SEED = 0
N_SEARCH = 100            # labeled training examples held out as the "search set"
CACHE = os.path.join(RESULTS, 'part3_preds.json')


# ----------------------------------------------------------------------------------------------- data
def load_data():
    from datasets import load_dataset
    d = load_dataset('gretelai/symptom_to_diagnosis')
    test = [(r['input_text'], r['output_text']) for r in d['test']]
    test_texts = {t for t, _ in test}
    train = [(r['input_text'], r['output_text']) for r in d['train'] if r['input_text'] not in test_texts]
    random.Random(SEED).shuffle(train)
    search, memory = train[:N_SEARCH], train[N_SEARCH:]
    labels = sorted({y for _, y in memory})
    return memory, search, test, labels, len(d['train']) - len(train)


# ----------------------------------------------------------------------------------------------- TF-IDF
TOKEN = re.compile(r'(?u)\b\w\w+\b')           # the same token rule as scikit-learn's default


def tokenize(text):
    return TOKEN.findall(text.lower())


def tfidf_fit(docs):
    """Method 1: TF-IDF from scratch. Returns the vocabulary, the idf weights and the document matrix."""
    vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in tokenize(d)}))}
    df = np.zeros(len(vocab))                               # in how many documents each word appears
    for d in docs:
        for w in set(tokenize(d)):
            df[vocab[w]] += 1
    n = len(docs)
    idf = np.log((1 + n) / (1 + df)) + 1                    # rare words get a large weight, common words a small one
    return vocab, idf, tfidf_transform(docs, vocab, idf)


def tfidf_transform(docs, vocab, idf):
    X = np.zeros((len(docs), len(vocab)))                  # one row per document, one column per word
    for r, d in enumerate(docs):
        for w in tokenize(d):
            if w in vocab:
                X[r, vocab[w]] += 1                        # tf: how often the word appears in this document
    X *= idf                                               # tf x idf
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1, norms)              # unit length, so a dot product is the cosine similarity


class Memory:
    """The labeled examples a harness has seen, with TF-IDF retrieval over their texts."""
    def __init__(self, examples):
        self.texts = [t for t, _ in examples]
        self.labels = [y for _, y in examples]
        self.vocab, self.idf, self.X = tfidf_fit(self.texts)

    def sims(self, query):
        q = tfidf_transform([query], self.vocab, self.idf)[0]
        return self.X @ q                                  # cosine similarity of the query with every memory text

    def top(self, query, k, keep=lambda i: True):
        order = np.argsort(-self.sims(query), kind='stable')
        return [int(i) for i in order if keep(i)][:k]


# ----------------------------------------------------------------------------------------------- the model call
class Model:
    def __init__(self, name=MODEL, dtype='bfloat16'):   # 16-bit numbers: half the memory of float32
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.tok = AutoTokenizer.from_pretrained(name)
        self.lm = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype)).to(self.dev).eval()

    def ids(self, user):
        msgs = [{'role': 'user', 'content': user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors='pt').input_ids.to(self.dev)

    def greedy(self, user, max_new=16):
        """Method 1: our own greedy decoding loop. Run the model, take the most likely next token, append, repeat."""
        torch = self.torch
        ids = self.ids(user)
        out, past = [], None
        with torch.no_grad():
            x = ids
            for _ in range(max_new):
                r = self.lm(input_ids=x, past_key_values=past, use_cache=True)
                past = r.past_key_values                   # keep the keys and values, so each step reads one token
                nxt = int(r.logits[0, -1].argmax())        # the most likely next token
                if nxt == self.tok.eos_token_id or nxt == self.tok.convert_tokens_to_ids('<|im_end|>'):
                    break
                out.append(nxt)
                x = torch.tensor([[nxt]], device=self.dev)
        return self.tok.decode(out, skip_special_tokens=True)

    def generate(self, user, max_new=16, repetition_penalty=1.0):
        """Method 2: the library way, model.generate with sampling turned off.
        Qwen's generation_config.json also sets repetition_penalty=1.1; 1.0 turns it off, as in our loop."""
        ids = self.ids(user)
        with self.torch.no_grad():
            g = self.lm.generate(ids, max_new_tokens=max_new, do_sample=False, top_p=None, top_k=None,
                                 temperature=None, repetition_penalty=repetition_penalty,
                                 pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)


def parse(answer, labels):
    """Map the model's text to one valid label: an exact label inside the answer, else the closest spelling."""
    a = answer.lower().strip()
    hits = [l for l in labels if l in a]
    if hits:
        return max(hits, key=len)
    close = difflib.get_close_matches(a.split('\n')[0], labels, n=1, cutoff=0.6)
    return close[0] if close else None


# ----------------------------------------------------------------------------------------------- the harnesses
def label_list(labels):
    return ', '.join(labels)


def shots(mem, idx):
    return '\n\n'.join(f'Description: {mem.texts[i]}\nDisease: {mem.labels[i]}' for i in idx)


ASK = 'Answer with the disease name only.'


def zero_shot(q, mem, labels, call):
    p = f'Classify the patient description as one of these diseases: {label_list(labels)}.\n\nDescription: {q}\n{ASK}'
    return call(p), [p]


def few_shot(q, mem, labels, call, n=8):
    idx = random.Random(SEED).sample(range(len(mem.texts)), n)      # the same n random examples for every query
    p = (f'Classify the patient description as one of these diseases: {label_list(labels)}.\n\n'
         f'Examples:\n\n{shots(mem, idx)}\n\nDescription: {q}\n{ASK}')
    return call(p), [p]


def draft_verification(q, mem, labels, call, k=5):
    """Figure 5. Call 1 drafts a label from the 5 nearest examples. Call 2 shows 5 examples that agree with the draft
    (confirmers) and 5 that do not (challengers), and keeps or revises the draft."""
    if len(mem.texts) < k:                                          # cold start: a single few-shot call
        return few_shot(q, mem, labels, call, n=len(mem.texts))
    near = mem.top(q, k)
    p1 = (f'Classify the patient description as one of these diseases: {label_list(labels)}.\n\n'
          f'Similar solved cases:\n\n{shots(mem, near)}\n\nDescription: {q}\n{ASK}')
    draft = call(p1)
    if draft is None:
        return None, [p1]
    confirm = mem.top(q, k, keep=lambda i: mem.labels[i] == draft)  # the second retrieval depends on the draft
    challenge = mem.top(q, k, keep=lambda i: mem.labels[i] != draft)
    p2 = (f'Classify the patient description as one of these diseases: {label_list(labels)}.\n\n'
          f'A first guess was: {draft}.\n\nCases WITH the label {draft}:\n\n{shots(mem, confirm)}\n\n'
          f'Similar cases with OTHER labels:\n\n{shots(mem, challenge)}\n\n'
          f'Description: {q}\nKeep the first guess or change it. {ASK}')
    return call(p2), [p1, p2]


def label_primed_query(q, mem, labels, call, n_pairs=4, pool=20):
    """Figure 6. One prompt in three blocks: the label primer, one query-relevant example per label (coverage),
    and pairs of close examples with different labels taken from the query's neighbourhood (contrastive pairs)."""
    s = mem.sims(q)
    coverage = []
    for lab in labels:                                              # the best example of every label
        own = [i for i in range(len(mem.texts)) if mem.labels[i] == lab]
        coverage.append(max(own, key=lambda i: s[i]))
    near = [int(i) for i in np.argsort(-s, kind='stable')[:pool]]  # the query's neighbourhood
    pairs, used = [], set()
    for a in near:                                                 # pair each close example with the closest one
        if a in used:                                              # that carries a different label
            continue
        b = next((j for j in near if j not in used and j != a and mem.labels[j] != mem.labels[a]), None)
        if b is not None:
            pairs.append((a, b)); used |= {a, b}
        if len(pairs) == n_pairs:
            break
    pair_text = '\n\n'.join(f'Pair {n + 1}:\n{shots(mem, [a])}\n{shots(mem, [b])}' for n, (a, b) in enumerate(pairs))
    p = (f'Valid labels: {label_list(labels)}.\n\n'
         f'One example for each label:\n\n{shots(mem, coverage)}\n\n'
         f'Close cases with different labels:\n\n{pair_text}\n\n'
         f'Description: {q}\nClassify the description as one of the valid labels. {ASK}')
    return call(p), [p]


def knn_vote(q, mem, labels, call, k=5):
    """No model at all: the most common label among the 5 nearest memory examples (a retrieval-only check)."""
    near = mem.top(q, k)
    votes = collections.Counter(mem.labels[i] for i in near)
    return votes.most_common(1)[0][0], []


HARNESSES = {'zero-shot': zero_shot, 'few-shot (8)': few_shot, 'draft-verification': draft_verification,
             'label-primed query': label_primed_query, 'TF-IDF 5-NN vote (no model)': knn_vote}


# ----------------------------------------------------------------------------------------------- Pareto frontier
def pareto_sweep(points):
    """Method 1: sort by context (small first); a point is on the frontier if it beats every cheaper point."""
    best, front = -1.0, []
    for name, ctx, acc in sorted(points, key=lambda p: (p[1], -p[2])):
        if acc > best:
            front.append(name); best = acc
    return front


def pareto_brute(points):
    """Method 2: the definition itself. Drop a point if another point is at least as good on both and better on one."""
    def dominates(p, q):
        return p[1] <= q[1] and p[2] >= q[2] and (p[1] < q[1] or p[2] > q[2])
    return [q[0] for q in points if not any(dominates(p, q) for p in points)]


# ----------------------------------------------------------------------------------------------- main
def main():
    memory, search, test, labels, dropped = load_data()
    L = Log('part3_data')
    L(f'Symptom2Disease (gretelai/symptom_to_diagnosis): {len(labels)} labels')
    L(f'memory: {len(memory)} labeled examples the harness may read')
    L(f'search: {len(search)} held-out training examples (to pick a harness)')
    L(f'test:   {len(test)} examples of the official test split, never used to choose anything')
    L(f'training texts dropped because the same text is in the test split: {dropped}')
    L(f'labels: {label_list(labels)}')
    L(f'example: "{test[0][0]}" -> {test[0][1]}')

    # ---- TF-IDF two ways
    mem = Memory(memory)
    from sklearn.feature_extraction.text import TfidfVectorizer
    T = Log('part3_tfidf')
    vec = TfidfVectorizer()
    Xs = vec.fit_transform(mem.texts).toarray()                     # Method 2: the library
    cols = [vec.vocabulary_[w] for w in sorted(mem.vocab, key=mem.vocab.get)]
    T(f'vocabulary: ours {len(mem.vocab)} words, scikit-learn {len(vec.vocabulary_)} words')
    T(f'largest difference between the two matrices: {np.abs(mem.X - Xs[:, cols]).max():.2e}')
    for w in ['and', 'stomach', 'burning', 'heartburn']:
        df = sum(w in set(tokenize(t)) for t in mem.texts)
        T(f'  word {w!r:<12} in {df:>3} of {len(mem.texts)} texts, idf = {mem.idf[mem.vocab[w]]:.3f}')
    agree = 0
    for q, _ in test:
        ours = mem.top(q, 5)
        lib = list(np.argsort(-(Xs @ vec.transform([q]).toarray()[0]), kind='stable')[:5])
        agree += ours == [int(i) for i in lib]
    T(f'test queries with the same top-5 neighbours: {agree} of {len(test)}')
    q0 = test[0][0]
    T(f'query: "{q0[:90]}..."')
    s0 = mem.sims(q0)
    for i in mem.top(q0, 5):
        T(f'  {s0[i]:.3f}  {mem.labels[i]:<24} "{mem.texts[i][:60]}..."')

    # ---- the model: greedy decoding two ways
    model = Model()
    G = Log('part3_greedy')
    for pen in [1.1, 1.0]:                           # 1.1 is the default in Qwen's generation_config.json
        same = same_label = 0
        G(f'--- model.generate with repetition_penalty={pen}')
        for q, y in test[:5]:
            p = zero_shot(q, mem, labels, lambda p: p)[1][0]
            a, b = model.greedy(p), model.generate(p, repetition_penalty=pen)
            same += a == b
            same_label += parse(a, labels) == parse(b, labels)
            G(f'ours: {a!r:<42} generate: {b!r}')
        G(f'same text: {same} of 5, same label after parse(): {same_label} of 5')

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    def run(hname, split, rows):
        key = f'{hname}|{split}'
        if key in cache and len(cache[key]) == len(rows):
            return cache[key]
        out, t0 = [], time.time()
        for n, (q, y) in enumerate(rows):
            raw = []

            def call(p):
                a = model.greedy(p)
                raw.append(a)
                return parse(a, labels)
            pred, prompts = HARNESSES[hname](q, mem, labels, call)
            base = len(zero_shot(q, mem, labels, lambda p: p)[1][0])
            extra = sum(len(p) for p in prompts) - (base if prompts else 0)
            out.append({'pred': pred, 'true': y, 'raw': raw, 'chars': sum(len(p) for p in prompts),
                        'extra_chars': max(0, extra), 'calls': len(prompts)})
        cache[key] = out
        print(f'{key}: {len(rows)} rows in {time.time() - t0:.0f} s', flush=True)
        json.dump(cache, open(CACHE, 'w'))
        return out

    P = Log('part3_prompts')
    q, y = test[0]
    for hname in ['zero-shot', 'few-shot (8)', 'draft-verification', 'label-primed query']:
        prompts, answers = [], []

        def call(p):
            prompts.append(p); answers.append(model.greedy(p))
            return parse(answers[-1], labels)
        HARNESSES[hname](q, mem, labels, call)
        P(f'===== {hname}: {len(prompts)} call(s), {sum(len(p) for p in prompts)} characters')
        for p, a in zip(prompts, answers):
            P(p)
            P(f'----- model answer: {a!r}')
    P(f'true label: {y}')

    E = Log('part3_eval')
    E(f'model: {MODEL}, greedy decoding, device {model.dev}')
    E(f'{"harness":<30}{"search acc":>11}{"test acc":>10}{"calls":>7}{"extra chars":>13}{"no label":>10}')
    rows = {}
    for hname in HARNESSES:
        r = {}
        for split, data in [('search', search), ('test', test)]:
            out = run(hname, split, data)
            r[split] = 100 * sum(o['pred'] == o['true'] for o in out) / len(out)
            r[split + '_n'] = len(out)
            if split == 'test':
                r['extra'] = sum(o['extra_chars'] for o in out) / len(out)
                r['calls'] = sum(o['calls'] for o in out) / len(out)
                r['invalid'] = sum(o['pred'] is None for o in out)
        rows[hname] = r
        E(f'{hname:<30}{r["search"]:>10.1f}%{r["test"]:>9.1f}%{r["calls"]:>7.0f}{r["extra"]:>13,.0f}{r["invalid"]:>10}')
    from sklearn.metrics import accuracy_score
    lpq = cache['label-primed query|test']
    E(f'check with scikit-learn accuracy_score (label-primed query, test): '
      f'{100 * accuracy_score([o["true"] for o in lpq], [str(o["pred"]) for o in lpq]):.1f}%')
    se = lambda p, n: 100 * math.sqrt(p / 100 * (1 - p / 100) / n)
    E(f'one standard error of a test accuracy near {rows["label-primed query"]["test"]:.0f}% on {len(test)} '
      f'examples: about {se(rows["label-primed query"]["test"], len(test)):.1f} points')

    # ---- Pareto frontier on the test split, two ways
    F = Log('part3_pareto')
    pts = [(h, rows[h]['extra'], rows[h]['test']) for h in HARNESSES if h != 'TF-IDF 5-NN vote (no model)']
    for h, c, a in sorted(pts, key=lambda p: p[1]):
        F(f'{h:<22} extra context {c:>7,.0f} chars   test accuracy {a:.1f}%')
    f1, f2 = pareto_sweep(pts), pareto_brute(pts)
    F(f'Method 1 (sweep):       frontier = {f1}')
    F(f'Method 2 (definition):  frontier = {sorted(f2, key=[p[0] for p in sorted(pts, key=lambda p: p[1])].index)}')
    F(f'same frontier: {sorted(f1) == sorted(f2)}')

    # ---- what the verification call changed
    V = Log('part3_verify')
    for split in ['search', 'test']:
        out = cache[f'draft-verification|{split}']
        drafts = [parse(o['raw'][0], labels) for o in out]
        kept = sum(d == o['pred'] for d, o in zip(drafts, out))
        fixed = sum(d != o['true'] and o['pred'] == o['true'] for d, o in zip(drafts, out))
        broke = sum(d == o['true'] and o['pred'] != o['true'] for d, o in zip(drafts, out))
        dacc = 100 * sum(d == o['true'] for d, o in zip(drafts, out)) / len(out)
        V(f'{split}: draft accuracy {dacc:.1f}%, final {rows["draft-verification"][split]:.1f}%; '
          f'kept {kept} of {len(out)}, changed {len(out) - kept}: fixed {fixed}, broke {broke}')
    out = cache['draft-verification|test']
    for o, (q, y) in zip(out, test):
        d = parse(o['raw'][0], labels)
        if d != y and o['pred'] == y:
            V(f'fixed: "{q[:100]}..."')
            V(f'  draft: {d}   final: {o["pred"]}   true: {y}')
            break
    for o, (q, y) in zip(out, test):
        d = parse(o['raw'][0], labels)
        if d == y and o['pred'] != y:
            V(f'broke: "{q[:100]}..."')
            V(f'  draft: {d}   final: {o["pred"]}   true: {y}')
            break

    lab_acc = {}
    for h in ['zero-shot', 'label-primed query']:
        c = collections.Counter(); n = collections.Counter()
        for o in cache[f'{h}|test']:
            n[o['true']] += 1; c[o['true']] += o['pred'] == o['true']
        lab_acc[h] = {l: 100 * c[l] / n[l] for l in n}
    A = Log('part3_labels')
    A(f'{"disease":<32}{"n":>3}{"zero-shot":>11}{"label-primed":>14}')
    for l in sorted(labels, key=lambda l: lab_acc['label-primed query'][l] - lab_acc['zero-shot'][l], reverse=True):
        A(f'{l:<32}{n[l]:>3}{lab_acc["zero-shot"][l]:>10.0f}%{lab_acc["label-primed query"][l]:>13.0f}%')
    save('part3', {'rows': rows, 'labels': labels, 'per_label': lab_acc, 'frontier': f1,
                   'sizes': {'memory': len(memory), 'search': len(search), 'test': len(test)}})


if __name__ == '__main__':
    main()
