"""Chapter 7: what is in our preference data? Sizes, rating gaps, lengths, and how far "pick the longer answer" gets you."""
import collections
import numpy as np
from common import Log, save
from ch7_common import load_pairs, parquet, tok, chat_text

log = Log('ch7_data')
full = parquet('train_prefs')
ties = (full.score_chosen == full.score_rejected).mean()
log(f'UltraFeedback binarized train_prefs: {len(full):,} rows; ties (same rating) {100 * ties:.1f}%')

train = load_pairs('train_prefs', 4000, seed=0)
test = load_pairs('test_prefs', 600, seed=123)
for nm, D in [('train', train), ('test', test)]:
    gap = np.array([p['sc'] - p['sr'] for p in D])
    lc, lr = np.array([p['ntok_c'] for p in D]), np.array([p['ntok_r'] for p in D])
    longer = (lc > lr).mean()
    log(f'{nm}: {len(D)} pairs | rating gap mean {gap.mean():.2f}, median {np.median(gap):.2f} | '
        f'response tokens chosen {lc.mean():.0f}, rejected {lr.mean():.0f} | chosen is longer in {100 * longer:.1f}%')

# example
p = test[544]
log('\nexample pair (test #544):')
log('prompt:  ', p['prompt'][:160].replace('\n', ' '))
log(f'chosen   (rating {p["sc"]}):', p['chosen'][:160].replace('\n', ' '))
log(f'rejected (rating {p["sr"]}):', p['rejected'][:160].replace('\n', ' '))
log('\nthe reward model reads this exact text (chosen):')
log(repr(chat_text(p['prompt'], p['chosen'])[:260]))
log(f'... {len(p["ids_c"])} tokens; last token id {p["ids_c"][-1]} = {tok.convert_ids_to_tokens(p["ids_c"][-1])}')

# gap buckets on train
gaps = np.array([p['sc'] - p['sr'] for p in train])
edges = [0, 1, 2, 4, 10]
hist = collections.Counter(np.digitize(gaps, edges[1:-1]))
log('\nrating gap buckets (train):', {f'[{edges[i]},{edges[i+1]})': int(hist[i]) for i in range(4)})

# length-only baselines on test
lc, lr = np.array([p['ntok_c'] for p in test]), np.array([p['ntok_r'] for p in test])
acc_long = ((lc > lr) + 0.5 * (lc == lr)).mean()
log(f'\n"always pick the longer answer" on the test set: accuracy {acc_long:.3f}')

save('ch7_data', dict(n_full=len(full), ties=float(ties), train_gaps=gaps.tolist(),
                      train_len_c=[p['ntok_c'] for p in train], train_len_r=[p['ntok_r'] for p in train],
                      test_len_c=lc.tolist(), test_len_r=lr.tolist(), test_gaps=[p['sc'] - p['sr'] for p in test],
                      acc_longer=float(acc_long), longer_train=float(np.mean([p['ntok_c'] > p['ntok_r'] for p in train])),
                      example=dict(prompt=p['prompt'], chosen=p['chosen'], rejected=p['rejected'], sc=p['sc'], sr=p['sr'],
                                   ntok=len(p['ids_c']))))
