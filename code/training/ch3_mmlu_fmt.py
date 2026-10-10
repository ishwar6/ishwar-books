"""Prompt formats and scoring helpers for the Chapter 3 MMLU experiments (shared by ch3_mmlu.py and ch3_mc_demo.py)."""
import torch
import torch.nn.functional as Fn
from ch3_common import DEV

L = 'ABCD'


def subj(s):
    return s.replace('_', ' ')


def fmt(q, choices, style='plain'):
    if style == 'plain':      # the original MMLU layout
        return q + '\n' + ''.join(f'{L[i]}. {c}\n' for i, c in enumerate(choices)) + 'Answer:'
    if style == 'paren':
        return q + '\n' + ''.join(f'({L[i]}) {c}\n' for i, c in enumerate(choices)) + 'Answer:'
    if style == 'qa':
        return 'Question: ' + q + '\nChoices:\n' + ''.join(f'{L[i]}: {c}\n' for i, c in enumerate(choices)) + 'Answer:'
    if style == 'inline':
        return q + ' ' + ' '.join(f'{L[i]}) {c}' for i, c in enumerate(choices)) + ' The correct answer is'
    raise ValueError(style)


def header(s):
    return f'The following are multiple choice questions (with answers) about {subj(s)}.\n\n'


@torch.no_grad()
def next_probs(tok, model, prompt):
    x = tok(prompt, return_tensors='pt')['input_ids'].to(DEV)
    return torch.softmax(model(x).logits[0, -1].float(), -1), x.shape[1]


def letter_ids(tok, space):
    return [tok.encode((' ' if space else '') + c)[0] for c in L]


@torch.no_grad()
def option_logprob(tok, model, prompt, option):
    p = tok(prompt)['input_ids']
    o = tok(' ' + option)['input_ids']
    x = torch.tensor([p + o], device=DEV)
    lp = Fn.log_softmax(model(x).logits[0, len(p) - 1:-1].float(), -1)
    return lp.gather(1, x[0, len(p):, None]).sum().item()


