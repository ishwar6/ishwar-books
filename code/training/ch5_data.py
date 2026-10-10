"""Chapter 5: a look at the instruction data used in the hands-on run (Alpaca-cleaned, 51,760 rows): what a row looks like,
how instructions start, how long the answers are, and how one row becomes a chat."""
import collections, re
import numpy as np
from datasets import load_dataset
from common import Log, save
from ch5_common import tok, messages, encode

log = Log('ch5_data')
d = load_dataset('yahma/alpaca-cleaned', split='train')
log(f'rows: {len(d):,}; rows with a non-empty input field: {sum(bool(x.strip()) for x in d["input"]):,}')
first = collections.Counter(re.findall(r'[A-Za-z]+', s)[0].lower() if re.findall(r'[A-Za-z]+', s) else '' for s in d['instruction'])
log('most common first words of the instruction: ' + ', '.join(f'{w} {c:,}' for w, c in first.most_common(12)))
sample = d.select(range(0, len(d), 50))
out_tok = [len(tok(x)['input_ids']) for x in sample['output']]
log(f'reply length (every 50th row, {len(out_tok)} rows): median {np.median(out_tok):.0f} tokens, 90th percentile {np.percentile(out_tok, 90):.0f}, max {max(out_tok)}')
R = {'rows': len(d), 'with_input': sum(bool(x.strip()) for x in d['input']), 'first': first.most_common(12),
     'reply_median': float(np.median(out_tok)), 'reply_p90': float(np.percentile(out_tok, 90))}
ex = d[1]
log('')
log('one row:')
for k in ['instruction', 'input', 'output']:
    log(f'  {k}: {ex[k][:200]!r}')
msgs, reply = messages(ex)
e = encode(ex)
log('')
log('as a chat, in the Qwen2.5 template:')
log(tok.decode(e['input_ids']))
log(f'{len(e["input_ids"])} tokens, of which {sum(l != -100 for l in e["labels"])} are learned (the reply and <|im_end|>)')
R['example'] = {k: ex[k] for k in ['instruction', 'input', 'output']}
save('ch5_data', R)
