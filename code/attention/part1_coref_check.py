"""Is layer 5 / head 5 really linking pronouns to their subject, or was one sentence a coincidence?"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
name = 'Qwen/Qwen2.5-0.5B'
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation='eager').eval()
L, H = 5, 5
cases = [
    ('The dog chased the ball because it was bored.', ' it', ' dog'),
    ('The trophy did not fit in the suitcase because it was too big.', ' it', ' trophy'),
    ('My sister bought a new car and she loves driving it.', ' she', ' sister'),
    ('The engineers fixed the server after it crashed twice.', ' it', ' server'),
    ('The scientist published the paper because she was proud of it.', ' she', ' scientist'),
    ('The children ate the cake because they were hungry.', ' they', ' children'),
]
rows = []
for text, pron, target in cases:
    ids = tok(text, return_tensors='pt').input_ids
    words = [tok.decode([i]) for i in ids[0]]
    with torch.no_grad():
        a = model(ids, output_attentions=True).attentions[L][0, H]
    p = words.index(pron)
    row = a[p, :p + 1]
    top = words[int(row.argmax())]
    rows.append({'text': text, 'pronoun': pron.strip(), 'expected': target.strip(), 'top': top.strip(),
                 'weight_on_expected': float(row[words.index(target)]), 'correct': top == target})
    print(f"{'OK ' if top == target else 'MISS'} {text!r}: '{pron.strip()}' -> '{top.strip()}' "
          f"(weight on '{target.strip()}' = {float(row[words.index(target)]):.2f})")
print(f"{sum(r['correct'] for r in rows)}/{len(rows)} correct")
json.dump(rows, open('results/part1_coref.json', 'w'), indent=1)
