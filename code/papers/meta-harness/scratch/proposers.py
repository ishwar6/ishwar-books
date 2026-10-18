"""Proposers: the P of Algorithm 1. Each one reads the filesystem of earlier candidates and returns new harness files.

  ScriptedProposer   hand-written rules over the design space of design_space.py. It reads scores and traces,
                     makes a diagnosis with fixed rules, and changes one or two knobs. The honest baseline.
  LocalLLMProposer   a small local chat model (Qwen2.5-1.5B-Instruct). It sees the list of candidates, picks the
                     files it wants to read, reads them, and writes a whole new harness file. No API, no cost.
  ChatAPIProposer    the same two steps, sent to any OpenAI-compatible chat endpoint (needs a key). Not run here.
  AgentCLIProposer   hands the run folder to a coding agent on the command line, as the paper does. Not run here.

propose(store, t, k) returns a list of k dicts: {'name', 'source', 'proposal'} where 'proposal' is the text
that goes into proposal.md (the proposer's reasoning)."""
import json, os, random, re, urllib.request
import design_space as ds
from harness_api import validate
from hlib import show_label

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = open(os.path.join(HERE, 'skill.md')).read()


# ================================================================================================ scripted
class ScriptedProposer:
    kind = 'scripted'

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def diagnose(self, store, cand):
        """Read the traces of one candidate and count three kinds of failure."""
        traces = store.traces(cand['folder'])
        wrong = [t for t in traces if not t['correct']]
        n = len(traces)

        def gold_shown(t):                      # did any example in the prompt carry the gold label?
            g = t['gold']
            return any(f'Label: {g}\n' in c['prompt'] or f'Label: {show_label(g)}\n' in c['prompt'] for c in t['calls'])
        pairs = {}
        for t in wrong:
            pairs[(t['gold'], t['pred'])] = pairs.get((t['gold'], t['pred']), 0) + 1
        top = max(pairs.items(), key=lambda kv: kv[1]) if pairs else (None, 0)
        return {
            'n': n, 'wrong': len(wrong),
            'invalid': sum(t['pred'] is None for t in traces),
            'gold_not_shown': sum(not gold_shown(t) for t in wrong),
            'top_pair': top[0], 'top_pair_count': top[1],
            'ctx_chars': cand['scores']['ctx_chars'],
        }

    def rules(self, cfg, d):
        """Fixed if-then rules: a diagnosis -> an edit, with the reason in words. Most urgent first."""
        out = []
        w = max(d['wrong'], 1)
        if d['invalid'] > 0.05 * d['n']:
            out.append(({'label_style': 'words' if cfg['label_style'] == 'snake' else 'snake'},
                        f"{d['invalid']} of {d['n']} replies were not a valid label; change how labels are written"))
        if cfg['shots'] == 0:
            out.append(({'shots': 8, 'select': 'tfidf'},
                        'the prompt has no examples; add the 8 stored examples most similar to the message'))
        if cfg['shots'] > 0 and cfg['select'] != 'tfidf':
            out.append(({'select': 'tfidf'},
                        f"{d['gold_not_shown']} of {d['wrong']} wrong answers had no example of the gold label in the "
                        'prompt; pick the examples by similarity to the message'))
        if d['gold_not_shown'] > 0.3 * w and not cfg['coverage']:
            out.append(({'coverage': True},
                        f"{d['gold_not_shown']} of {d['wrong']} wrong answers never saw the gold label in an example; "
                        'add one example of every label'))
        if d['top_pair_count'] >= 2 and not cfg['contrastive']:
            g, p = d['top_pair']
            out.append(({'contrastive': True},
                        f"the most common mistake is {g} -> {p} ({d['top_pair_count']} times); show close examples "
                        'with different labels side by side'))
        if d['top_pair_count'] >= 2 and not cfg['verify']:
            out.append(({'verify': True},
                        'mistakes repeat between the same labels; check the first answer against confirmers and '
                        'challengers'))
        if cfg['shots'] in (4, 8) and cfg['select'] == 'tfidf':
            out.append(({'shots': cfg['shots'] * 2}, 'more similar examples may separate close labels'))
        if cfg['shots'] >= 8 and d['ctx_chars'] > 1500:
            out.append(({'shots': cfg['shots'] // 2 if cfg['shots'] < 60 else 16},
                        f"the prompt is long ({d['ctx_chars']} characters); try fewer examples for a cheaper point"))
        if cfg['instruction'] == 'short':
            out.append(({'instruction': 'careful'}, 'tell the model that several labels are close'))
        return [({**cfg, **edit}, reason) for edit, reason in out]

    def propose(self, store, t, k):
        cands = [c for c in store.candidates() if ds.read_config(c['source'])]
        tried = {ds.key(ds.read_config(c['source'])) for c in cands}
        ranked = sorted(cands, key=lambda c: (-c['scores']['accuracy'], c['scores']['ctx_chars']))
        picks = []
        for parent in ranked:                    # start from the best; if all its edits were tried, use the next one
            cfg = ds.read_config(parent['source'])
            d = self.diagnose(store, parent)
            options = self.rules(cfg, d)
            rest = [(c, f'explore: set {knob} = {v!r}') for knob, v, c in ds.neighbours(cfg)]
            self.rng.shuffle(rest)
            for new, reason in options + rest:
                if ds.key(new) in tried or ds.key(new) in {ds.key(p[1]) for p in picks}:
                    continue
                picks.append((parent, new, reason, d))
                if len(picks) == k:
                    break
            if len(picks) == k:
                break
        out = []
        for parent, new, reason, d in picks:
            name = ds.describe(new).replace(', ', '_').replace(' ', '_')
            text = (f"# Proposal (scripted proposer, iteration {t})\n\nParent: {parent['folder']} "
                    f"(search accuracy {parent['scores']['accuracy']}%, {parent['scores']['ctx_chars']} prompt characters)\n\n"
                    f"Diagnosis from the parent's traces: {json.dumps(d)}\n\nEdit: {reason}\n\nNew config: {new}\n")
            out.append({'name': name, 'source': ds.render(name, new), 'proposal': text})
        return out


# ================================================================================================ local LLM
def extract_code(reply):
    m = re.findall(r'```(?:python)?\s*\n(.*?)```', reply, re.S)
    if m:
        return max(m, key=len)
    m = re.search(r'```(?:python)?\s*\n(.*)$', reply, re.S)      # an unclosed block (the model ran out of tokens)
    return m.group(1) if m else ''


class LocalLLMProposer:
    """A small chat model as the proposer, in two steps per proposal: choose files to read, then write a harness."""
    kind = 'local-llm'
    FILES = ('harness.py', 'scores.json', 'traces.jsonl', 'proposal.md')

    def __init__(self, llm=None, model_name='Qwen/Qwen2.5-1.5B-Instruct', temperature=0.7, max_new_tokens=1500,
                 retries=1, labels=None, examples=None):
        if llm is None:
            from llm import LocalLLM
            llm = LocalLLM(model_name)
        self.llm, self.temperature, self.max_new_tokens, self.retries = llm, temperature, max_new_tokens, retries
        self.labels, self.examples = labels, examples
        self.reads = []                                          # every file read, for statistics like Table 8

    def chat(self, prompt, max_new_tokens, seed):
        return self.llm(prompt, max_new_tokens=max_new_tokens, system=SKILL, temperature=self.temperature, seed=seed)

    @staticmethod
    def listing(store):
        rows = ['folder | search accuracy % | prompt chars per message | idea']
        for c in store.candidates():
            idea = open(os.path.join(c['dir'], 'proposal.md')).read().split('\n')
            idea = next((l for l in idea if l.startswith(('IDEA:', 'Edit:', 'Seed:'))), '').strip()[:90]
            rows.append(f"{c['folder']} | {c['scores']['accuracy']} | {c['scores']['ctx_chars']} | {idea}")
        return '\n'.join(rows)

    @staticmethod
    def read(store, rel):
        """Return the text of one file under candidates/, shortened so it fits the small model's prompt."""
        folder, fname = rel.split('/')[-2:]
        path = os.path.join(store.cand_dir, folder, fname)
        if fname != 'traces.jsonl':
            return open(path).read()[:6000]
        wrong = [t for t in store.traces(folder) if not t['correct']][:5]
        lines = [f"{len(wrong)} wrong answers shown (of the search set). For each: message, gold, predicted, "
                 'and the end of the last prompt with the reply.']
        for t in wrong:
            last = t['calls'][-1] if t['calls'] else {'prompt': '', 'reply': ''}
            lines.append(json.dumps({'text': t['text'], 'gold': t['gold'], 'pred': t['pred'],
                                     'prompt_end': last['prompt'][-500:], 'reply': last['reply']}))
        return '\n'.join(lines)

    def choose_files(self, store, seed):
        names = [c['folder'] for c in store.candidates()]
        ask = (f"## Candidates so far\n{self.listing(store)}\n\nEach folder has: {', '.join(self.FILES)}.\n"
               'Which files do you want to read before you write a new harness? Answer with 1 to 3 paths like '
               f'candidates/{names[-1]}/traces.jsonl, one per line, and nothing else.')
        reply = self.chat(ask, 80, seed)
        paths = []
        for folder, fname in re.findall(r'(\d{3}_[a-z0-9_]+)/(harness\.py|scores\.json|traces\.jsonl|proposal\.md)', reply):
            if folder in names and f'{folder}/{fname}' not in paths:
                paths.append(f'{folder}/{fname}')
        fallback = not paths
        if fallback:                                             # the model named no real file: read the best one
            best = store.best()['folder']
            paths = [f'{best}/harness.py', f'{best}/traces.jsonl']
        return paths[:3], reply, fallback

    def propose(self, store, t, k):
        out = []
        for j in range(k):
            seed = 1000 * t + j
            paths, reply1, fallback = self.choose_files(store, seed)
            self.reads.append({'t': t, 'j': j, 'paths': paths, 'fallback': fallback})
            best = f"{store.best()['folder']}/harness.py"           # always shown: a working file to start from
            shown = paths if best in paths else [best] + paths
            files = '\n\n'.join(f'### candidates/{p}\n{self.read(store, p)}' for p in shown)
            ask = (f"## Candidates so far\n{self.listing(store)}\n\n## Files you chose to read\n{files}\n\n"
                   f'Now write the new harness. Start from {best}, which works, and keep its class Harness and its '
                   'three methods. First one line that starts with IDEA:, then the complete file in one ```python block.')
            reply = self.chat(ask, self.max_new_tokens, seed)
            code, attempts = extract_code(reply), []
            for r in range(self.retries + 1):
                ok, err = validate(code, self.labels, self.examples) if code else (False, 'no ```python block')
                attempts.append(err or 'passed')
                if ok or r == self.retries:
                    break
                fix = (f"{ask}\n\nYour file was rejected: {err}\nWrite the complete corrected file in one ```python "
                       'block.')
                reply = self.chat(fix, self.max_new_tokens, seed + 500)
                code = extract_code(reply) or code
            idea = next((l.strip() for l in reply.split('\n') if l.strip().startswith('IDEA:')), 'IDEA: (none given)')
            text = (f"# Proposal (local LLM proposer {self.llm.name}, iteration {t}, sample {j})\n\n{idea}\n\n"
                    f"Files it asked for: {reply1.strip()[:300]!r}\n"
                    f"Files read: {paths}{' (fallback: it named no real file)' if fallback else ''}; "
                    f"also shown: {best}\n"
                    f"Self-check attempts: {attempts}\n\n## Full reply\n\n{reply}\n")
            out.append({'name': f'llm_t{t}_{j}', 'source': code, 'proposal': text})
        return out


# ================================================================================================ API (not run)
class ChatAPIProposer(LocalLLMProposer):
    """The same two steps, sent to an OpenAI-compatible /chat/completions endpoint. Not run for this page.

    export MH_API_BASE=https://<provider>/v1   MH_API_KEY=...   MH_API_MODEL=<model name>
    python run.py --proposer api --iters 10 --k 2 --name api-1
    A local server that speaks the same protocol (for example ollama at http://localhost:11434/v1) also works."""
    kind = 'api'

    def __init__(self, labels=None, examples=None, temperature=0.7, max_new_tokens=4000, retries=1):
        self.base = os.environ['MH_API_BASE'].rstrip('/')
        self.key = os.environ.get('MH_API_KEY', '')
        self.model = os.environ['MH_API_MODEL']
        self.temperature, self.max_new_tokens, self.retries = temperature, max_new_tokens, retries
        self.labels, self.examples, self.reads = labels, examples, []
        self.llm = type('Named', (), {'name': self.model})()

    def chat(self, prompt, max_new_tokens, seed):
        body = {'model': self.model, 'temperature': self.temperature, 'max_tokens': max_new_tokens, 'seed': seed,
                'messages': [{'role': 'system', 'content': SKILL}, {'role': 'user', 'content': prompt}]}
        req = urllib.request.Request(f'{self.base}/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json',
                                              'Authorization': f'Bearer {self.key}'})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)['choices'][0]['message']['content']


class AgentCLIProposer:
    """The paper's setup: a coding agent works inside the run folder with the skill, using its own file tools.
    Not run for this page. MH_AGENT_CMD is a shell command with {prompt}, for your agent's non-interactive mode.
    The agent must write pending/<name>/harness.py and pending/<name>/proposal.md; we collect them after it exits.
    Keep heldout/ out of its reach (run the search before writing any test results, as run.py does)."""
    kind = 'agent-cli'

    def propose(self, store, t, k):
        import shlex, subprocess
        pending = os.path.join(store.root, 'pending')
        os.makedirs(pending, exist_ok=True)
        open(os.path.join(store.root, 'SKILL.md'), 'w').write(SKILL)
        prompt = (f'Read SKILL.md. Iteration {t}: write {k} new harnesses, each in its own folder pending/<name>/ '
                  'with harness.py and proposal.md (start proposal.md with a line IDEA: ...).')
        subprocess.run(os.environ['MH_AGENT_CMD'].format(prompt=shlex.quote(prompt)), shell=True, cwd=store.root,
                       timeout=3600)
        out = []
        for name in sorted(os.listdir(pending)):
            d = os.path.join(pending, name)
            src = open(os.path.join(d, 'harness.py')).read() if os.path.exists(os.path.join(d, 'harness.py')) else ''
            why = open(os.path.join(d, 'proposal.md')).read() if os.path.exists(os.path.join(d, 'proposal.md')) else ''
            out.append({'name': name, 'source': src, 'proposal': why})
            os.rename(d, os.path.join(store.root, 'rejected', f'pending_{t}_{name}'))   # keep the raw folder
        return out[:k]
