"""Background page experiments: what a harness is, on a tiny task with a small local model.
1. The task: 6 card intents of Banking77 (PolyAI). Memory = the labelled training questions, test = 10 per intent.
2. TF-IDF retrieval two ways: Method 1 our own code, Method 2 scikit-learn's TfidfVectorizer; check they agree.
3. Prompt assembly two ways: Method 1 our own chat-format string, Method 2 the tokenizer's chat template; check equal.
4. Three harnesses around the same frozen Qwen2.5 model: zero-shot, few-shot (1 fixed example per intent),
   retrieval (the 6 nearest questions from memory). Accuracy and prompt tokens (context cost) for each.
5. A ReAct-style tool loop: Method 1 our own Thought/Action/Observation loop with a calculator,
   Method 2 the model's native tool calling through the chat template; both against the model with no tool.
Every printed line goes to results/partb_stdout.txt; numbers to results/partb.json; traces to results/partb_traces.jsonl.
Data: https://github.com/PolyAI-LDN/task-specific-datasets (banking_data/train.csv, test.csv), cached in ~/.cache/papers.
Run: PYTHONPATH=<dir with scikit-learn> ~/.venvs/attn/bin/python mh_partb.py   (about 3 minutes on an M-series Mac)"""
import ast, csv, json, math, operator, os, re, time, urllib.request
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import Log, save, RESULTS

log = Log('partb')
R = {}
DATA = os.path.expanduser('~/.cache/papers/banking77')
URL = 'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}.csv'
INTENTS = ['activate_my_card', 'card_arrival', 'card_delivery_estimate',
           'card_not_working', 'declined_card_payment', 'lost_or_stolen_card']
N_TEST = 10                     # test questions per intent
K = 6                           # retrieved examples per question
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'


def load(split):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, split + '.csv')
    if not os.path.exists(path):
        open(path, 'wb').write(urllib.request.urlopen(URL.format(split)).read())
    return [(r['text'].strip(), r['category']) for r in csv.DictReader(open(path)) if r['category'] in INTENTS]


train, test_all = load('train'), load('test')
test = [x for lab in INTENTS for x in [t for t in test_all if t[1] == lab][:N_TEST]]
log('$ python mh_partb.py')
log(f'== 1. Task: {len(INTENTS)} card intents of Banking77')
log(f'  memory (labelled training questions): {len(train)}   test questions: {len(test)} ({N_TEST} per intent)')
log(f'  example: {test[0][0]!r} -> {test[0][1]}')
R['task'] = dict(intents=INTENTS, memory=len(train), test=len(test))


# ---------------------------------------------------------------- 2. TF-IDF retrieval, two ways
def tokens(text):
    """Lower-case words of 2+ letters or digits: the same rule as scikit-learn's default token_pattern."""
    return re.findall(r'(?u)\b\w\w+\b', text.lower())


class TfidfMemory:
    """Method 1: TF-IDF from scratch. idf(w) = ln((1 + n) / (1 + df(w))) + 1, then each vector is scaled to length 1."""
    def __init__(self, texts):
        self.n = len(texts)
        df = Counter(w for t in texts for w in set(tokens(t)))          # in how many questions each word appears
        self.idf = {w: math.log((1 + self.n) / (1 + d)) + 1 for w, d in df.items()}
        self.vecs = [self.vec(t) for t in texts]                        # one sparse vector (a dict) per question

    def vec(self, text):
        tf = Counter(w for w in tokens(text) if w in self.idf)          # raw counts; unknown words are dropped
        v = {w: c * self.idf[w] for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def scores(self, query):
        q = self.vec(query)                                             # cosine = dot product of unit vectors
        return [sum(q[w] * d.get(w, 0.0) for w in q) for d in self.vecs]

    def top(self, query, k):
        s = self.scores(query)
        return sorted(range(self.n), key=lambda i: -s[i])[:k]


from sklearn.feature_extraction.text import TfidfVectorizer          # Method 2: the library
mem_texts = [t for t, _ in train]
ours = TfidfMemory(mem_texts)
lib = TfidfVectorizer().fit(mem_texts)
M = lib.transform(mem_texts)
max_diff, same_top = 0.0, 0
for q, _ in test:
    s1 = torch.tensor(ours.scores(q))
    s2 = torch.tensor((M @ lib.transform([q]).T).toarray().ravel())
    max_diff = max(max_diff, (s1 - s2).abs().max().item())
    same_top += sorted(range(len(s1)), key=lambda i: -s1[i])[:K] == sorted(range(len(s2)), key=lambda i: -s2[i])[:K]
log('')
log('== 2. TF-IDF retrieval: our code against scikit-learn TfidfVectorizer')
log(f'  vocabulary: ours {len(ours.idf)} words, sklearn {len(lib.vocabulary_)} words')
log(f'  max |ours - sklearn| over {len(test)} x {len(train)} cosine scores: {max_diff:.1e}')
log(f'  same top-{K} list for {same_top} of {len(test)} test questions')
q0 = test[0][0]
log(f'  nearest to {q0!r}:')
for i in ours.top(q0, 3):
    log(f'    {ours.scores(q0)[i]:.3f}  {train[i][0]!r} -> {train[i][1]}')
R['tfidf'] = dict(max_diff=max_diff, same_top=same_top, vocab=len(ours.idf))

# ---------------------------------------------------------------- 3. prompt assembly, two ways
SYSTEM = 'You classify bank customer questions. Answer with one label from the list and nothing else.'
LABELS = 'Labels: ' + ', '.join(INTENTS)
FEWSHOT = [next(x for x in train if x[1] == lab) for lab in INTENTS]   # the first training question of each intent


def user_text(query, examples):
    """The harness decides what the model sees: the label list, some examples, then the question."""
    parts = [LABELS]
    if examples:
        parts.append('Examples:\n' + '\n'.join(f'Question: {t}\nLabel: {lab}' for t, lab in examples))
    parts.append(f'Question: {query}\nLabel:')
    return '\n\n'.join(parts)


def chat_string(system, user):
    """Method 1: the Qwen2.5 chat format written by hand."""
    return (f'<|im_start|>system\n{system}<|im_end|>\n'
            f'<|im_start|>user\n{user}<|im_end|>\n'
            f'<|im_start|>assistant\n')


HARNESSES = {
    'zero-shot': lambda q: [],
    'few-shot': lambda q: FEWSHOT,
    'retrieval': lambda q: [train[i] for i in ours.top(q, K)][::-1],   # the most similar question goes last
}

MODELS = ['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct']
tok = AutoTokenizer.from_pretrained(MODELS[1])
msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user_text(q0, HARNESSES['retrieval'](q0))}]
lib_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)   # Method 2
our_str = chat_string(SYSTEM, msgs[1]['content'])
log('')
log('== 3. Prompt assembly: our chat string against tokenizer.apply_chat_template')
log(f'  identical strings: {our_str == lib_str}   ({len(tok(our_str).input_ids)} tokens)')
R['template_equal'] = our_str == lib_str


def parse_label(out):
    s = out.strip().lower().replace(' ', '_')
    for lab in sorted(INTENTS, key=len, reverse=True):                 # longest first: card_arrival vs card_...
        if lab in s:
            return lab
    return 'invalid'


@torch.no_grad()
def generate(model, prompt, max_new=12):
    ids = tok(prompt, return_tensors='pt').to(DEV)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True), ids.input_ids.shape[1]


# ---------------------------------------------------------------- 4. three harnesses, one frozen model
log('')
log('== 4. Three harnesses around the same frozen model (greedy decoding, ' + DEV + ')')
R['harness'], preds = {}, {}
traces = open(os.path.join(RESULTS, 'partb_traces.jsonl'), 'w')
for name in MODELS:
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to(DEV).eval()
    short = name.split('/')[1]
    for h, pick in HARNESSES.items():
        t0, right, invalid, ntok, preds[short, h] = time.time(), 0, 0, [], []
        for q, gold in test:
            prompt = chat_string(SYSTEM, user_text(q, pick(q)))
            out, n = generate(model, prompt)
            pred = parse_label(out)
            right += pred == gold
            invalid += pred == 'invalid'
            ntok.append(n)
            preds[short, h].append(pred)
            traces.write(json.dumps(dict(model=short, harness=h, question=q, gold=gold, output=out, pred=pred,
                                         prompt_tokens=n)) + '\n')
        acc = 100 * right / len(test)
        R['harness'][f'{short}/{h}'] = dict(acc=acc, right=right, invalid=invalid, tokens=sum(ntok) / len(ntok))
        log(f'  {short:22} {h:10} accuracy {right:2}/{len(test)} = {acc:5.1f}%   '
            f'invalid {invalid:2}   prompt tokens {sum(ntok) / len(ntok):5.0f}   {time.time() - t0:4.1f} s')
    if name != MODELS[-1]:
        del model
traces.close()

log('')
log(f'== 4b. Which questions change between zero-shot and retrieval ({short})')
z, r = preds[short, 'zero-shot'], preds[short, 'retrieval']
fixed = [i for i in range(len(test)) if z[i] != test[i][1] and r[i] == test[i][1]]
broken = [i for i in range(len(test)) if z[i] == test[i][1] and r[i] != test[i][1]]
log(f'  wrong -> right: {len(fixed)}   right -> wrong: {len(broken)}')
for i in fixed[:2]:
    q, gold = test[i]
    log(f'  {q!r}  (gold {gold})')
    log(f'     zero-shot said {z[i]}; retrieval said {r[i]}')
    log(f'     labels of the {K} retrieved questions: {[train[j][1] for j in ours.top(q, K)]}')
R['flips'] = dict(fixed=len(fixed), broken=len(broken))


# ---------------------------------------------------------------- 5. a ReAct-style tool loop
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.USub: operator.neg}


def calculator(expr):
    """The tool: evaluate + - * / and brackets only (no eval of arbitrary code)."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        raise ValueError('not allowed')
    expr = expr.replace('%', '/100').replace('x', '*').replace(',', '')
    return round(ev(ast.parse(expr, mode='eval').body), 4)


QUESTIONS = [
    ('I paid 3 bills of 47.85 EUR and 2 bills of 118.4 EUR. How much did I pay in total?', 3 * 47.85 + 2 * 118.4),
    ('My balance was 1250.75 EUR. I sent 3 transfers of 212.6 EUR each. What is my balance now?', 1250.75 - 3 * 212.6),
    ('The card fee is 2.9% of a 845 EUR payment. How much is the fee?', 0.029 * 845),
]
REACT = ('Answer the question. You can use a calculator.\n'
         'Use this format:\n'
         'Thought: what to do next\n'
         'Action: calculator[an arithmetic expression]\n'
         'Observation: the result (the system writes this line)\n'
         '... (Thought, Action, Observation can repeat)\n'
         'Final Answer: the number\n\n'
         'Question: {q}\n')


def first_number(s):
    m = re.findall(r'-?\d[\d,]*\.?\d*', s or '')
    return float(m[-1].replace(',', '')) if m else None


def close(a, b):
    return a is not None and abs(a - b) < 0.01


STRICT = r'Action:\s*calculator\[(.+?)\]'                       # only the exact format we asked for
LOOSE = r'Action:\s*(?:calculator)?\s*\[?([-\d.(][^\]\n]*)\]?\s*$'            # also "Action: 1 + 2", "calculator(1 + 2)"


@torch.no_grad()
def react_loop(model, question, pattern, max_steps=4):
    """Method 1: our own harness loop. The model writes Thought and Action; the harness runs the tool
    and writes the Observation; the loop stops at 'Final Answer:'."""
    text, steps = REACT.format(q=question), []
    for _ in range(max_steps):
        out, _ = generate(model, chat_string('You are a careful assistant.', text), max_new=80)
        if not steps:
            react_loop.first_turn = out                                 # kept to show what the model wrote
        out = out.split('Observation:')[0]                              # the model must not invent the tool result
        text += out
        if 'Final Answer:' in out:
            return first_number(out.split('Final Answer:')[1]), steps
        m = re.search(pattern, out, re.M)
        if not m:
            return first_number(out), steps
        try:
            obs = calculator(m.group(1))
        except Exception as e:
            obs = f'error: {e}'
        steps.append((m.group(1), obs))
        text += f'\nObservation: {obs}\n'
    return None, steps


TOOLS = [{'type': 'function', 'function': {
    'name': 'calculator', 'description': 'Evaluate an arithmetic expression with + - * / and brackets.',
    'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression']}}}]


@torch.no_grad()
def native_tools(model, question, max_steps=4):
    """Method 2: the same loop with the model's own tool-call format (<tool_call>{json}</tool_call>),
    built by the tokenizer's chat template."""
    msgs, steps = [{'role': 'user', 'content': question + ' Use the calculator, then give the number.'}], []
    for _ in range(max_steps):
        prompt = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True)
        out, _ = generate(model, prompt, max_new=80)
        calls = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', out, re.S)
        if not calls:
            return first_number(out), steps
        msgs.append({'role': 'assistant', 'content': out})
        for c in calls:
            try:
                expr = json.loads(c)['arguments']['expression']
                obs = calculator(expr)
            except Exception as e:
                expr, obs = c, f'error: {e}'
            steps.append((expr, obs))
            msgs.append({'role': 'tool', 'content': str(obs)})
    return None, steps


log('')
log(f'== 5. A tool loop ({short}): no tool, our ReAct loop (strict, loose parser), the native tool-call format')
R['react'] = []
for q, gold in QUESTIONS:
    bare, _ = generate(model, chat_string('You are a careful assistant.', q + ' Give only the number.'), max_new=20)
    a0, s0 = react_loop(model, q, STRICT)
    raw = react_loop.first_turn
    a1, s1 = react_loop(model, q, LOOSE)
    a2, s2 = native_tools(model, q)
    log(f'  Q: {q}')
    log(f'     first model turn, raw: {raw!r}')
    log(f'     exact answer        {round(gold, 4)}')
    log(f'     model alone         {first_number(bare)}   {"right" if close(first_number(bare), gold) else "wrong"}')
    log(f'     ours, strict parser {a0}   {"right" if close(a0, gold) else "wrong"}   tool calls: {s0}')
    log(f'     ours, loose parser  {a1}   {"right" if close(a1, gold) else "wrong"}   tool calls: {s1}')
    log(f'     native tool calls   {a2}   {"right" if close(a2, gold) else "wrong"}   tool calls: {s2}')
    R['react'].append(dict(q=q, gold=gold, bare=first_number(bare), strict=a0, ours=a1, ours_steps=s1, native=a2, native_steps=s2))
save('partb', R)
