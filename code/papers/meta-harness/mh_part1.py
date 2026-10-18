"""Part 1 of Meta-Harness, explained: the big idea, run on a laptop.

What it shows (every number is printed and saved to results/part1_stdout.txt and results/part1.json):
  1. One frozen model, one task, four harnesses: the score moves a lot when only the harness changes.
     Model: Qwen/Qwen2.5-1.5B-Instruct (greedy decoding). Task: Symptom2Disease (gretelai/symptom_to_diagnosis),
     one of the paper's three text-classification datasets, 22 diseases. 60 fixed test examples.
  2. What the model sees: the chat prompt built by hand (Method 1) and by the tokenizer's chat template (Method 2).
  3. Accuracy counted by hand (Method 1) and with scikit-learn (Method 2).
  4. Feedback three ways (a score, a short summary, the full traces): how many tokens each one is, and what the
     score hides that the traces show.
  5. The traces as files: searched with our own Python (Method 1) and with the real grep (Method 2).
  6. The paper's headline numbers, checked with arithmetic.
Run: PYTHONPATH=<dir with scikit-learn> python mh_part1.py   (about 3 minutes on an M-series Mac)"""
import json, os, re, shutil, subprocess, random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from common import Log, save, timer, RESULTS

log = Log('part1')
MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
N_EVAL, K_SHOTS = 60, 5
FS = os.path.join(RESULTS, 'part1_fs')          # the "filesystem of experience": one folder per harness

# ---------------------------------------------------------------- the task and the frozen model
data = load_dataset('gretelai/symptom_to_diagnosis')
train, test = data['train'], data['test']
LABELS = sorted(set(train['output_text']))
rng = random.Random(0)
idx = rng.sample(range(len(test)), N_EVAL)
EVAL = [(test[i]['input_text'], test[i]['output_text']) for i in idx]
MEMORY = [(r['input_text'], r['output_text']) for r in train]     # labeled examples a harness may retrieve

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()

log('== 1. One frozen model, one task, four harnesses')
log(f'model: {MODEL} on {DEV}, greedy decoding, weights never change')
log(f'task: Symptom2Disease, {len(LABELS)} labels, memory {len(MEMORY)} labeled examples, eval {N_EVAL} test examples')


# ---------------------------------------------------------------- the model call (the same for every harness)
def call_model(messages, max_new_tokens=12):
    """The only place where the model runs. A harness decides what goes into `messages`."""
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEV)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), ids.shape[1]


# ---------------------------------------------------------------- four harnesses: what to show, how to read the answer
def norm(s):
    return re.sub(r'[^a-z ]+', '', s.lower()).strip()


def parse_strict(output):
    """Accept the answer only if the whole output is exactly one label."""
    return norm(output) if norm(output) in LABELS else None


def parse_lenient(output):
    """Accept the first label that occurs anywhere in the output (longest labels first)."""
    o = norm(output)
    for lab in sorted(LABELS, key=len, reverse=True):
        if lab in o:
            return lab
    return None


LABEL_LIST = 'Possible diagnoses: ' + ', '.join(LABELS) + '.'
vec = TfidfVectorizer().fit([t for t, _ in MEMORY])
MEM_M = vec.transform([t for t, _ in MEMORY])


def retrieve(text, k):
    """The k most similar labeled examples in memory (TF-IDF cosine similarity)."""
    sims = (vec.transform([text]) @ MEM_M.T).toarray()[0]
    return [MEMORY[i] for i in sims.argsort()[::-1][:k]]


def h_bare(text):
    return [{'role': 'user', 'content': f'Patient: {text}\nWhat is the diagnosis?'}], parse_strict


def h_labels(text):
    return [{'role': 'system', 'content': LABEL_LIST + ' Answer with one diagnosis from the list only.'},
            {'role': 'user', 'content': f'Patient: {text}\nDiagnosis:'}], parse_strict


def h_labels_lenient(text):
    msgs, _ = h_labels(text)
    return msgs, parse_lenient


def h_retrieval(text):
    shots = ''.join(f'Patient: {t}\nDiagnosis: {y}\n\n' for t, y in retrieve(text, K_SHOTS))
    return [{'role': 'system', 'content': LABEL_LIST + ' Answer with one diagnosis from the list only.'},
            {'role': 'user', 'content': f'{shots}Patient: {text}\nDiagnosis:'}], parse_lenient


HARNESSES = {'A_bare': h_bare, 'B_labels': h_labels, 'C_labels_lenient': h_labels_lenient,
             'D_labels_retrieval': h_retrieval}


def evaluate(name, harness):
    """Run one harness on the eval set. Write its traces and score to FS/<name>/, as Meta-Harness does."""
    d = os.path.join(FS, name)
    os.makedirs(d, exist_ok=True)
    preds, golds, ctx, traces = [], [], [], []
    for i, (text, gold) in enumerate(EVAL):
        msgs, parse = harness(text)
        out, n_in = call_model(msgs)
        pred = parse(out)
        preds.append(pred or 'NONE'); golds.append(gold); ctx.append(n_in)
        traces.append({'id': i, 'prompt': msgs, 'output': out, 'parsed': pred, 'gold': gold, 'correct': pred == gold})
    with open(os.path.join(d, 'traces.jsonl'), 'w') as f:
        for t in traces:
            f.write(json.dumps(t) + '\n')
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)                 # Method 1: count by hand
    json.dump({'accuracy': acc, 'mean_input_tokens': sum(ctx) / len(ctx)}, open(os.path.join(d, 'scores.json'), 'w'))
    return acc, preds, golds, sum(ctx) / len(ctx), traces


shutil.rmtree(FS, ignore_errors=True)
R, T = {}, {}
clock = timer()
log(f'{"harness":<20} {"accuracy":>9} {"right":>6} {"input tokens":>13}')
for name, h in HARNESSES.items():
    acc, preds, golds, ctx, traces = evaluate(name, h)
    R[name] = {'accuracy': acc, 'right': round(acc * N_EVAL), 'mean_input_tokens': round(ctx, 1),
               'sklearn_accuracy': accuracy_score(golds, preds)}
    T[name] = traces
    log(f'{name:<20} {acc:>9.3f} {round(acc * N_EVAL):>3}/{N_EVAL} {ctx:>13.1f}')
log(f'time for 4 x {N_EVAL} model calls: {clock():.0f} s')
log('the first outputs of harness A (12 new tokens at most):')
for t in T['A_bare'][:2]:
    log(f"  {t['output']!r}")
accs = [r['accuracy'] for r in R.values()]
best, worst = max(accs), min(accs)
log(f'best {best:.3f}, worst {worst:.3f}, gap {100 * (best - worst):.1f} points, '
    + (f'ratio {best / worst:.1f}x' if worst > 0 else 'ratio undefined (worst is 0)'))

# ---------------------------------------------------------------- 2. what the model sees, two ways
log('')
log('== 2. The prompt the model sees: built by hand (Method 1) and by the chat template (Method 2)')
msgs, _ = h_labels(EVAL[0][0])


def chatml_by_hand(messages):
    """Qwen2.5's chat format: each turn is <|im_start|>role, newline, text, <|im_end|>, newline."""
    s = ''
    if messages[0]['role'] != 'system':                                  # Qwen adds its default system turn
        s += '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'
    for m in messages:
        s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return s + '<|im_start|>assistant\n'                                  # the model writes from here


ours = chatml_by_hand(msgs)
lib = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
log(f'Method 1 length {len(ours)} chars, Method 2 length {len(lib)} chars, identical: {ours == lib}')
ours_bare = chatml_by_hand(h_bare(EVAL[0][0])[0])
lib_bare = tok.apply_chat_template(h_bare(EVAL[0][0])[0], add_generation_prompt=True, tokenize=False)
log(f'harness A (no system turn): identical: {ours_bare == lib_bare}')
log('the start of the prompt of harness B, as the model reads it:')
for line in lib.split('\n')[:3]:
    log('  | ' + line[:110] + (' ...' if len(line) > 110 else ''))

# ---------------------------------------------------------------- 3. accuracy, two ways
log('')
log('== 3. Accuracy counted by hand (Method 1) and by scikit-learn (Method 2)')
for name, r in R.items():
    log(f"{name:<20} ours {r['accuracy']:.4f}  sklearn {r['sklearn_accuracy']:.4f}  "
        f"agree: {abs(r['accuracy'] - r['sklearn_accuracy']) < 1e-12}")

# ---------------------------------------------------------------- 4. feedback three ways
log('')
log('== 4. Feedback about harness B, three ways')
tb = T['B_labels']
wrong = [t for t in tb if not t['correct']]
unparsed = [t for t in tb if t['parsed'] is None]
fb_score = f"B_labels: accuracy {R['B_labels']['accuracy']:.3f}"
fb_summary = (f"B_labels: accuracy {R['B_labels']['accuracy']:.3f}. {len(wrong)} of {N_EVAL} wrong. "
              f"The model often gives answers that do not match a label.")
fb_traces = open(os.path.join(FS, 'B_labels', 'traces.jsonl')).read()
n_tok = lambda s: len(tok(s)['input_ids'])
sizes = {'score': n_tok(fb_score), 'summary': n_tok(fb_summary), 'traces': n_tok(fb_traces)}
log(f"a score:          {sizes['score']:>7} tokens  -> {fb_score}")
log(f"a short summary:  {sizes['summary']:>7} tokens")
log(f"the full traces:  {sizes['traces']:>7} tokens  ({N_EVAL} prompts, outputs and verdicts)")
log(f"traces / score = {sizes['traces'] / sizes['score']:.0f}x")
log(f'what only the traces show: {len(wrong)} wrong answers, and {len(unparsed)} of them are outputs that match no label')
shown = 0
for t in unparsed:
    if parse_lenient(t['output']) == t['gold'] and shown < 3:
        log(f"  output {t['output']!r:<34} gold {t['gold']!r}")
        shown += 1
right_but_rejected = sum(parse_lenient(t['output']) == t['gold'] for t in unparsed)
log(f'{right_but_rejected} of the {len(unparsed)} rejected outputs name the right disease inside a longer answer')

# ---------------------------------------------------------------- 5. the traces as files: grep, two ways
log('')
log('== 5. Search the trace files: our Python (Method 1) and grep (Method 2)')
PATTERN = '"correct": false'


def grep_count(root, pattern):
    """Count matching lines in every traces.jsonl under root, like grep -rc."""
    counts = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn == 'traces.jsonl':
                p = os.path.join(dirpath, fn)
                counts[os.path.relpath(p, root)] = sum(pattern in line for line in open(p))
    return dict(sorted(counts.items()))


ours = grep_count(FS, PATTERN)
out = subprocess.run(['grep', '-rc', '--include=traces.jsonl', PATTERN, '.'], cwd=FS, capture_output=True, text=True).stdout
lib = dict(sorted((l.rsplit(':', 1)[0][2:], int(l.rsplit(':', 1)[1])) for l in out.strip().split('\n')))
for k in ours:
    log(f'{k:<32} Method 1 {ours[k]:>3}   grep {lib[k]:>3}')
log(f'Method 1 and grep agree: {ours == lib}')
files = sorted(os.path.relpath(os.path.join(dp, f), FS) for dp, _, fs in os.walk(FS) for f in fs)
log(f'{len(files)} files in the store: ' + ', '.join(files))

# ---------------------------------------------------------------- 6. the paper's numbers, checked
log('')
log("== 6. The paper's headline numbers, checked")
checks = [
    ('text classification: Meta-Harness 48.6 vs ACE 40.9 (points)', 48.6 - 40.9),
    ('context: ACE 50.8K vs Meta-Harness 11.4K tokens (times fewer)', 50.8 / 11.4),
    ('TerminalBench-2, Opus 4.6: 76.4 vs Terminus-KIRA 74.7 (points)', 76.4 - 74.7),
    ('TerminalBench-2, Haiku 4.5: 37.6 vs Goose 35.5 (points)', 37.6 - 35.5),
    ('the 6x source: same model, 12% vs 2% of 50 tasks (ratio)', 12 / 2),
    ('12% and 2% of 50 tasks, as task counts', (0.12 * 50, 0.02 * 50)),
]
for name, v in checks:
    log(f'{name:<66} {v if isinstance(v, tuple) else round(v, 2)}')

save('part1', {'model': MODEL, 'device': DEV, 'n_eval': N_EVAL, 'labels': len(LABELS), 'harnesses': R,
               'feedback_tokens': sizes, 'unparsed_B': len(unparsed), 'wrong_B': len(wrong),
               'right_but_rejected_B': right_but_rejected, 'grep_agree': ours == lib})
