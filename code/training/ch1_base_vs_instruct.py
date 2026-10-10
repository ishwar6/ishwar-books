"""Chapter 1, experiment 2: the same questions to Qwen2.5-0.5B (base) and Qwen2.5-0.5B-Instruct.
Greedy decoding (no randomness), seed fixed anyway. The base model gets the raw question text, and also the question
wrapped in the chat template; the instruct model gets the chat template, as it was trained. Also prints the chat template
token by token."""
import torch
from common import Log, save
from ch1_models import load, greedy, BASE, INSTRUCT

torch.manual_seed(0)
log = Log('ch1_base_vs_instruct')
PROMPTS = ['What is the capital of France?',
           'Write a haiku about rain.',
           'A shop sells pencils at 3 for 45 cents. How much do 7 pencils cost?',
           'Give me three tips for sleeping better.',
           'Who are you?']
MAXN = 200

tok_i, inst = load(INSTRUCT)
tok_b, base = load(BASE)
assert tok_i.get_vocab() == tok_b.get_vocab(), 'the two tokenizers differ'


def chat_ids(tok, q):
    msgs = [{'role': 'user', 'content': q}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)


# ---- 1. the chat template, token by token
q = PROMPTS[0]
text = tok_i.apply_chat_template([{'role': 'user', 'content': q}], add_generation_prompt=True, tokenize=False)
ids = chat_ids(tok_i, q)
if isinstance(ids, dict):
    ids = ids['input_ids']
log('=== the chat template (what the instruct model actually reads) ===')
log(text.rstrip('\n') + '\n')
log(f'{len(ids)} tokens:')
special = set(tok_i.all_special_ids)
template_tokens = []
for i in ids:
    s = tok_i.convert_ids_to_tokens(i)
    template_tokens.append(dict(id=i, tok=tok_i.decode([i]), special=i in special))
    log(f'  {i:>6}  {tok_i.decode([i])!r}' + ('   <- special token' if i in special else ''))

# ---- 2. the same prompts to both models
results = []
for q in PROMPTS:
    row = dict(prompt=q)
    for label, tok, model, ids in [('base, raw text', tok_b, base, tok_b(q)['input_ids']),
                                   ('base, chat template', tok_b, base, chat_ids(tok_b, q)),
                                   ('instruct, chat template', tok_i, inst, chat_ids(tok_i, q))]:
        if isinstance(ids, dict):
            ids = ids['input_ids']
        new, stopped = greedy(tok, model, ids, MAXN)
        out = tok.decode(new)
        row[label] = dict(text=out, n_tokens=len(new), stopped=stopped)
        log(f'\n=== {label} | prompt: {q!r} | {len(new)} new tokens, ' + ('stopped by itself' if stopped else f'cut at {MAXN}') + ' ===')
        log(out)
    results.append(row)

save('ch1_base_vs_instruct', dict(template_text=text, template_tokens=template_tokens, results=results, max_new_tokens=MAXN))
