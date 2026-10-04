"""Part 1: why reading in both directions matters.

A left-to-right model (GPT-2) must guess the next word from the left side only.
BERT fills in a blank using the words on BOTH sides. Same sentences, real models, top-5 guesses.
Runs on the CPU, so the numbers are the same on every machine (up to tiny float differences)."""
import torch
from transformers import AutoTokenizer, BertForMaskedLM, GPT2LMHeadModel
from common import Log, save

torch.manual_seed(0)
log = Log('part1')

SENTENCES = [  # (left part, the hidden word, right part)
    ('i went to the', 'bank', 'to deposit my paycheck.'),
    ("my neighbor's", 'dog', 'barked at the mailman all morning.'),
    ('she picked up her', 'guitar', 'and started to play a song.'),
]

btok = AutoTokenizer.from_pretrained('bert-base-uncased')
bert = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
gtok = AutoTokenizer.from_pretrained('openai-community/gpt2')
gpt2 = GPT2LMHeadModel.from_pretrained('openai-community/gpt2').eval()


def top5(logits, tok):
    p = torch.softmax(logits, -1)
    v, i = p.topk(5)
    return [(tok.decode([int(t)]).strip(), round(float(x), 3)) for x, t in zip(v, i)]


out = []
for left, word, right in SENTENCES:
    # GPT-2 sees only the left part and predicts the next token
    ids = gtok(left[0].upper() + left[1:], return_tensors='pt').input_ids
    with torch.no_grad():
        g = top5(gpt2(ids).logits[0, -1], gtok)
    # BERT sees the whole sentence with the word replaced by [MASK]
    text = f'{left} [MASK] {right}'
    enc = btok(text, return_tensors='pt')
    pos = (enc.input_ids[0] == btok.mask_token_id).nonzero().item()
    with torch.no_grad():
        b = top5(bert(**enc).logits[0, pos], btok)
    out.append(dict(left=left, word=word, right=right, gpt2=g, bert=b))
    log(f'sentence: {left} ____ {right}   (hidden word: {word})')
    log('  GPT-2, left side only: ' + ', '.join(f'{w} {p:.3f}' for w, p in g))
    log('  BERT, both sides:      ' + ', '.join(f'{w} {p:.3f}' for w, p in b))
    log('')

save('part1', dict(fill=out))
