"""Chapter 3: one MMLU question scored three ways, printed in full (the worked example behind the scoring figure).
Uses the same prompt formats as ch3_mmlu.py."""
import torch
from common import Log, save
from ch3_common import BASE, INSTRUCT, data, load, chat, generate, sample_rows
from ch3_mmlu_fmt import fmt, header, L, next_probs, letter_ids, option_logprob

log = Log('ch3_mc_demo')
test = sample_rows(data('mmlu_test'), 400, seed=0)
r = next(r for r in test.itertuples() if len(r.question) < 160 and all(len(str(c)) < 40 for c in r.choices)
         and r.subject not in ('high_school_mathematics', 'elementary_mathematics'))
q, ch, ans = r.question, list(r.choices), int(r.answer)
prompt = header(r.subject) + fmt(q, ch)
log('== the question (MMLU, ' + r.subject.replace('_', ' ') + '), correct answer: ' + L[ans] + ' ==')
for line in prompt.split('\n'):
    log('  ' + line)
res = {'subject': r.subject, 'question': q, 'choices': ch, 'answer': ans, 'models': {}}
for name, label in [(BASE, 'base'), (INSTRUCT, 'instruct')]:
    tok, model = load(name)
    pr, _ = next_probs(tok, model, prompt)
    sp = letter_ids(tok, True)
    p4 = [pr[i].item() for i in sp]
    top = torch.topk(pr, 5)
    cl = [option_logprob(tok, model, f'Question: {q}\nAnswer:', str(c)) for c in ch]
    gen_prompt = prompt if label == 'base' else chat(tok, prompt[:-len('Answer:')] + 'Answer with the letter of the correct option.')
    out = generate(tok, model, [gen_prompt], max_new_tokens=40)[0]
    res['models'][label] = dict(p_letters=p4, top5=[(tok.decode([int(i)]), v.item()) for v, i in zip(top.values, top.indices)],
                                cloze=cl, cloze_norm=[v / len(str(c)) for v, c in zip(cl, ch)], generated=out)
    log(f'\n-- {label} --')
    log('1. letter probabilities after "Answer:":  ' + '  '.join(f'P(" {L[i]}") = {p4[i]:.3f}' for i in range(4))
        + f'   (sum {sum(p4):.3f})')
    log('   top 5 next tokens: ' + ', '.join(f'{t!r} {v:.3f}' for t, v in res['models'][label]['top5']))
    log('2. log-probability of each option text: ' + '  '.join(f'{L[i]} {cl[i]:.2f}' for i in range(4)))
    log('   per character:                       ' + '  '.join(f'{L[i]} {cl[i] / len(str(ch[i])):.3f}' for i in range(4)))
    log(f'3. generated: {out[:120]!r}')
    del model
save('ch3_mc_demo', res)
