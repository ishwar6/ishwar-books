"""Chapter 1, experiment 6: a first, tiny evaluation (Chapter 3 does this properly).
20 short questions with one known answer. For each model setting we check (a) does the expected answer appear in the
output, and (b) did the model end its turn by itself within 60 new tokens. Greedy decoding."""
from common import Log, save
from ch1_models import load, greedy, BASE, INSTRUCT

log = Log('ch1_eval_preview')
QA = [('What is the capital of Japan?', 'Tokyo'), ('What is the capital of Canada?', 'Ottawa'),
      ('What is the capital of Kenya?', 'Nairobi'), ('What is the capital of Egypt?', 'Cairo'),
      ('Who wrote Romeo and Juliet?', 'Shakespeare'), ('Who developed the theory of relativity?', 'Einstein'),
      ('What planet is known as the Red Planet?', 'Mars'), ('What is the chemical symbol for gold?', 'Au'),
      ('How many continents are there?', '7'), ('What is the largest ocean on Earth?', 'Pacific'),
      ('What is 12 plus 30?', '42'), ('What is 9 times 8?', '72'), ('What is 100 minus 37?', '63'),
      ('What is half of 64?', '32'), ('What is 15 squared?', '225'), ('How many minutes are in two hours?', '120'),
      ('What gas do plants take in from the air?', 'carbon dioxide'), ('What is the freezing point of water in Celsius?', '0'),
      ('In which country are the pyramids of Giza?', 'Egypt'), ('What is the longest river in Africa?', 'Nile')]
MAXN = 60

tok_i, inst = load(INSTRUCT)
tok_b, base = load(BASE)


def chat(tok, q):
    ids = tok.apply_chat_template([{'role': 'user', 'content': q}], add_generation_prompt=True, tokenize=True, return_dict=False)
    return ids['input_ids'] if isinstance(ids, dict) else ids


settings = [('base, raw question', tok_b, base, lambda q: tok_b(q)['input_ids']),
            ('base, chat template', tok_b, base, lambda q: chat(tok_b, q)),
            ('instruct, chat template', tok_i, inst, lambda q: chat(tok_i, q))]
summary, examples = {}, []
for name, tok, model, enc in settings:
    correct = stopped_n = 0
    for q, a in QA:
        new, stopped = greedy(tok, model, enc(q), MAXN)
        text = tok.decode(new, skip_special_tokens=True)
        ok = a.lower() in text.lower()
        correct += ok
        stopped_n += stopped
        examples.append(dict(setting=name, q=q, a=a, out=text, ok=ok, stopped=stopped))
    summary[name] = dict(correct=correct, stopped=stopped_n, n=len(QA))
    log(f'{name:<24} answer found: {correct:>2}/{len(QA)}   ended its turn by itself: {stopped_n:>2}/{len(QA)}')
log('\nfirst outputs per setting (q = "What is the capital of Kenya?"):')
for e in examples:
    if e['q'] == 'What is the capital of Kenya?':
        log(f'  {e["setting"]:<24} {e["out"][:110]!r}')
log('\nmisses:')
for e in examples:
    if not e['ok']:
        log(f'  {e["setting"]:<24} {e["q"]!r}: {e["out"][:90]!r}')
save('ch1_eval_preview', dict(summary=summary, examples=examples, max_new_tokens=MAXN))
