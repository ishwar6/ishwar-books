"""Part 6 checks: each piece of the from-scratch package (scratch/) next to a second way of doing the same thing.

  1. the task: sizes of the three sets
  2. the frozen model: our greedy loop vs transformers' generate()      -> same replies
  3. TF-IDF retrieval: hlib.TfIdf vs sklearn TfidfVectorizer            -> same scores, same top 5
  4. interface validation on four harness files (one good, three broken)
  5. the objective as a mean of rewards, on a tiny example and on a saved candidate
  6. the Pareto frontier: by the definition vs by a sort-and-sweep      -> same frontier
Writes results/part6_checks_stdout.txt."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'scratch'))
from common import Log, save
import task, hlib
from harness_api import validate
from store import Store, pareto_front, pareto_front_sweep

log = Log('part6_checks')
labels, memory, search, test = task.load()

log('== 1. The task')
log(f'{len(labels)} labels: {", ".join(labels[:4])}, ...')
log(f'memory {len(memory)} (labeled stream), search {len(search)}, test {len(test)}')
log(f"example: {search[2]['text']!r} -> {search[2]['label']}")

log('== 2. The frozen model: greedy decoding two ways')
from llm import LocalLLM, greedy_from_scratch
m = LocalLLM()
same = 0
for ex in search[:3]:
    p = ('Classify the bank customer message into exactly one of the labels.\n\nLabels:\n' + '\n'.join('- ' + l for l in labels)
         + f"\n\nMessage: {ex['text']}\nAnswer with the label only.\nLabel:")
    a, b = greedy_from_scratch(m, p), m(p)
    same += a == b
    log(f"from scratch: {a!r:30s} generate(): {b!r:30s} gold: {ex['label']}")
log(f'same reply in {same} of 3 prompts')

log('== 3. TF-IDF retrieval two ways')
docs = [ex['text'] for ex in memory]
ours = hlib.TfIdf(docs)
from sklearn.feature_extraction.text import TfidfVectorizer
sk = TfidfVectorizer(tokenizer=hlib.words, lowercase=False, token_pattern=None)   # same words as ours
D = sk.fit_transform(docs)
max_diff, same_top = 0.0, 0
for ex in search[:5]:
    s1 = ours.scores(ex['text'])
    s2 = (sk.transform([ex['text']]) @ D.T).toarray()[0]
    max_diff = max(max_diff, max(abs(x - y) for x, y in zip(s1, s2)))
    top1 = ours.top(ex['text'], 5)
    top2 = sorted(range(len(docs)), key=lambda i: -s2[i])[:5]
    same_top += top1 == top2
log(f'vocabulary: ours {len(ours.idf)} words, sklearn {len(sk.vocabulary_)} words')
log(f'largest difference in cosine scores over 5 queries x {len(docs)} documents: {max_diff:.1e}')
log(f'same top-5 neighbours in {same_top} of 5 queries')
q = search[2]['text']
log(f'query: {q!r}')
for i in ours.top(q, 3):
    log(f"  {ours.scores(q)[i]:.3f}  {memory[i]['label']:26s} {memory[i]['text'][:60]!r}")

log('== 4. Interface validation (Algorithm 1, line 11)')
GOOD = open(os.path.join(HERE, 'scratch', 'README.md')).read().split('```python\n')[1].split('```')[0]
cases = {
    'the README harness': GOOD,
    'reads a file': GOOD.replace("self.memory = llm, labels, []", "self.memory = llm, labels, []\n        self.cheat = open('data/task.json').read()"),
    'returns a made-up label': GOOD.replace("return parse_label(self.llm(prompt), self.labels) or self.labels[0]",
                                           "return 'other'"),
    'has no Harness class': GOOD.replace('class Harness', 'class Classifier'),
}
for name, src in cases.items():
    t0 = time.time()
    ok, err = validate(src, labels, memory)
    log(f"{name:26s} {'passed' if ok else 'rejected'} in {1000 * (time.time() - t0):.1f} ms  {err[:70]}")

log('== 5. The objective: the mean reward over the search set')
rewards = [1, 0, 1, 1, 0]
log(f'tiny example: rewards {rewards} -> mean {sum(rewards) / len(rewards):.2f}')
run = os.path.join(HERE, 'scratch', 'runs', 'scripted-1')
if os.path.exists(os.path.join(run, 'summary.json')):
    st = Store(run)
    c = st.best()
    tr = st.traces(c['folder'])
    r = [int(t['correct']) for t in tr]
    log(f"{c['folder']}: {sum(r)} rewards of 1 out of {len(r)} -> {100 * sum(r) / len(r):.1f}%  (scores.json: {c['scores']['accuracy']}%)")

    log('== 6. The Pareto frontier two ways')
    cands = st.candidates()
    pts = [(x['scores']['accuracy'], x['scores']['ctx_chars']) for x in cands]
    f1, f2 = pareto_front(pts), pareto_front_sweep(pts)
    log(f'{len(pts)} candidates; by definition: {f1}; by sort and sweep: {f2}; same: {f1 == f2}')
    for i in f1:
        log(f"  {cands[i]['folder']:44s} {pts[i][0]:5.1f}%  {pts[i][1]:5d} chars")
save('part6_checks', {'greedy_same': same, 'tfidf_max_diff': max_diff, 'tfidf_same_top5': same_top})
