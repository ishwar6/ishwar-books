"""Chapter 3, experiment 2: MMLU on a random 400-question subset, scored in several ways, for Qwen2.5-0.5B base and Instruct.
Ways of scoring a multiple-choice question:
  letter-5shot   5 solved examples from the same subject, then the question, ending "Answer:"; compare P(" A") .. P(" D")
  letter-0shot   the same without the examples (the original MMLU method, Hendrycks et al. 2021)
  cloze          "Question: ...\nAnswer:" then the log-probability of each option's TEXT; pick the highest
  cloze-norm     the same, divided by the option's length in characters (long options are not punished)
  chat-letter    (Instruct) the question in the chat template; compare P("A") .. P("D") for the first answer token
  generate       let the model write; take the first standalone letter A to D it produces (no letter = wrong)
Also: prompt-format variants, and the correct answer moved to each of the four positions.
Every per-question result is saved, so ch3_stats.py can compute bootstrap confidence intervals."""
import re, time
import torch
import torch.nn.functional as Fn
from common import Log, save
from ch3_common import BASE, INSTRUCT, DEV, data, load, chat, generate, sample_rows
from ch3_mmlu_fmt import L, subj, fmt, header, next_probs, letter_ids, option_logprob

log = Log('ch3_mmlu')
N = 400
test = sample_rows(data('mmlu_test'), N, seed=0)
dev = data('mmlu_dev')


def shots(s):
    d = dev[dev['subject'] == s]
    return ''.join(fmt(r.question, list(r.choices)) + f' {L[r.answer]}\n\n' for r in d.itertuples())


def parse_letter(text):
    m = re.search(r'\b([ABCD])\b', text)
    return L.index(m.group(1)) if m else -1


R = {'n': N, 'subjects': test['subject'].tolist(), 'answer': test['answer'].tolist(), 'models': {}}
log(f'MMLU subset: {N} questions from {test["subject"].nunique()} of 57 subjects (random, seed 0)')
log(f'correct letters in the subset: ' + ', '.join(f'{c}={sum(test["answer"] == i)}' for i, c in enumerate(L)))
log('')
for name, label in [(BASE, 'base'), (INSTRUCT, 'instruct')]:
    tok, model = load(name)
    sp, ns = letter_ids(tok, True), letter_ids(tok, False)
    M = {}
    t0 = time.time()

    def letter_run(key, make_prompt, ids):
        preds, mass, pcorr = [], [], []
        for r in test.itertuples():
            pr, _ = next_probs(tok, model, make_prompt(r))
            p4 = pr[ids]
            preds.append(int(p4.argmax())); mass.append(p4.sum().item()); pcorr.append(p4[r.answer].item())
        ok = [int(p == a) for p, a in zip(preds, test['answer'])]
        M[key] = dict(correct=ok, pred=preds, mass=mass, p_correct=pcorr)
        log(f'{label:8s} {key:22s} acc {sum(ok) / N:.3f}   mean P(A..D) {sum(mass) / N:.3f}   ({time.time() - t0:.0f}s)')

    letter_run('letter-5shot', lambda r: header(r.subject) + shots(r.subject) + fmt(r.question, list(r.choices)), sp)
    letter_run('letter-0shot', lambda r: header(r.subject) + fmt(r.question, list(r.choices)), sp)
    for style in ['paren', 'qa', 'inline']:
        letter_run(f'format-{style}', lambda r, s=style: header(r.subject) + fmt(r.question, list(r.choices), s), sp)
    letter_run('format-noheader', lambda r: fmt(r.question, list(r.choices)), sp)
    if label == 'instruct':
        letter_run('chat-letter', lambda r: chat(tok, header(r.subject) + fmt(r.question, list(r.choices))[:-len('Answer:')]
                                              + 'Answer with the letter of the correct option only.'), ns)

    # cloze: score each option's text
    cl, cn = [], []
    for r in test.itertuples():
        prompt = f'Question: {r.question}\nAnswer:'
        lps = [option_logprob(tok, model, prompt, str(c)) for c in r.choices]
        norm = [lp / max(1, len(str(c))) for lp, c in zip(lps, r.choices)]
        cl.append(int(max(range(4), key=lambda i: lps[i]) == r.answer))
        cn.append(int(max(range(4), key=lambda i: norm[i]) == r.answer))
    M['cloze'] = dict(correct=cl); M['cloze-norm'] = dict(correct=cn)
    log(f'{label:8s} {"cloze":22s} acc {sum(cl) / N:.3f}')
    log(f'{label:8s} {"cloze-norm":22s} acc {sum(cn) / N:.3f}   ({time.time() - t0:.0f}s)')

    # the answer moved to each position (the other options keep their order)
    for pos in range(4):
        def moved(r, pos=pos):
            ch = [c for i, c in enumerate(r.choices) if i != r.answer]
            ch.insert(pos, r.choices[r.answer])
            return header(r.subject) + fmt(r.question, ch)
        preds = []
        for r in test.itertuples():
            pr, _ = next_probs(tok, model, moved(r))
            preds.append(int(pr[sp].argmax()))
        ok = [int(p == pos) for p in preds]
        M[f'answer-at-{L[pos]}'] = dict(correct=ok, pred=preds)
        log(f'{label:8s} {"answer always at " + L[pos]:22s} acc {sum(ok) / N:.3f}   predicted letters: '
            + ' '.join(f'{c}={preds.count(i)}' for i, c in enumerate(L)))

    # generation: the model writes, we parse a letter
    if label == 'base':
        prompts = [header(r.subject) + shots(r.subject) + fmt(r.question, list(r.choices)) for r in test.itertuples()]
    else:
        prompts = [chat(tok, header(r.subject) + fmt(r.question, list(r.choices))[:-len('Answer:')]
                        + 'Answer with the letter of the correct option.') for r in test.itertuples()]
    outs = generate(tok, model, prompts, max_new_tokens=48, batch=16)
    preds = [parse_letter(o) for o in outs]
    ok = [int(p == a) for p, a in zip(preds, test['answer'])]
    M['generate'] = dict(correct=ok, pred=preds, text=outs)
    log(f'{label:8s} {"generate":22s} acc {sum(ok) / N:.3f}   no letter found: {preds.count(-1)}   ({time.time() - t0:.0f}s)')
    for o in outs[:3]:
        log(f'         sample output: {o[:90]!r}')
    log('')
    R['models'][label] = M
    del model
    if DEV == 'mps':
        torch.mps.empty_cache()
save('ch3_mmlu', R)
