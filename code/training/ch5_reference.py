"""Chapter 5: the same evaluation for the two reference models, Qwen2.5-0.5B (base, before SFT) and Qwen2.5-0.5B-Instruct
(after Qwen's own post-training): held-out loss on our 100 held-out Alpaca examples, the checks, and the forgetting probes."""
import torch
from common import Log, save
from ch5_common import load, BASE, INSTRUCT, load_split, heldout_loss, run_checks, wikitext_ppl, fewshot_antonyms, chat

log = Log('ch5_reference')
train, val = load_split()
R = {}
for name in [BASE, INSTRUCT]:
    m = load(name).eval()
    r = {'heldout': heldout_loss(m, val)}
    log(f'== {name}: held-out loss {r["heldout"]:.4f}')
    r['checks'] = run_checks(m, log)
    r['pass'] = sum(x['passed'] for x in r['checks']); r['stops'] = sum(x['stopped'] for x in r['checks'])
    r['wikitext_ppl'], r['antonyms_4shot'] = wikitext_ppl(m), fewshot_antonyms(m)
    log(f'passed {r["pass"]}/12, ended its turn {r["stops"]}/12, wikitext-2 ppl {r["wikitext_ppl"]:.2f}, 4-shot antonyms {r["antonyms_4shot"]:.0%}')
    r['samples'] = {}
    for q in ["Give three tips for a good night's sleep.", 'Write a short, polite email asking a colleague to send the meeting notes.',
              'What is the difference between weather and climate?']:
        t, s = chat(m, q, 200)
        r['samples'][q] = {'reply': t, 'stopped': s}
        log(f'Q: {q}\nA: {t[:400]}\n   (ended its turn: {s})')
    R[name] = r
    del m
save('ch5_reference', R)
