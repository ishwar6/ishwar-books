"""Chapter 2, experiment 3: the DPO loss (Rafailov et al. 2023, eq. 7) on real models.

Part A: one preference pair, scored by a policy and a reference model. DPO needs exactly four numbers per pair:
log pi(chosen), log pi_ref(chosen), log pi(rejected), log pi_ref(rejected). No reward model, no sampling.
Part B: a tiny DPO run. Start from Qwen2.5-0.5B-Instruct (policy and frozen reference are the same model at step 0),
train on 16 pairs where the chosen answer starts with "In short:" and the rejected one has the same content without it.
Then check 8 held-out questions the model never saw: does it now start its answers with "In short:"?"""
import copy, random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import Log, save, timer

log = Log('ch2_dpo')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
INST = 'Qwen/Qwen2.5-0.5B-Instruct'
BETA = 0.1                                   # the DPO paper's default
torch.manual_seed(0); random.seed(0)
tok = AutoTokenizer.from_pretrained(INST)
policy = AutoModelForCausalLM.from_pretrained(INST, dtype=torch.float32).to(DEV)
reference = copy.deepcopy(policy).eval()     # frozen copy: pi_ref
for p in reference.parameters():
    p.requires_grad_(False)

# (question, the one-line summary, the longer explanation). chosen = "In short: " + summary + " " + detail; rejected = detail only
DATA = [
    ('What is the capital of Japan?', 'Tokyo.', 'Tokyo has been the seat of government since 1868 and is the largest city in the country.'),
    ('Why do we have seasons?', "the Earth's tilt.", "The Earth's axis is tilted about 23 degrees, so each hemisphere gets more direct sunlight for part of the year."),
    ('How many legs does a spider have?', 'eight.', 'Spiders are arachnids, and all arachnids have eight legs, unlike insects, which have six.'),
    ('What does a thermometer measure?', 'temperature.', 'A thermometer shows how hot or cold something is, usually in degrees Celsius or Fahrenheit.'),
    ('Why does ice float on water?', 'ice is less dense than water.', 'When water freezes its molecules form an open crystal that takes up more space, so ice is lighter for its size.'),
    ('What is photosynthesis?', 'plants making food from light.', 'Plants use sunlight, water and carbon dioxide to make sugar, and they release oxygen as a by-product.'),
    ('Who wrote Romeo and Juliet?', 'William Shakespeare.', 'Shakespeare wrote the play in the 1590s, and it is one of the most performed plays in the world.'),
    ('What is the boiling point of water at sea level?', '100 degrees Celsius.', 'At normal air pressure pure water boils at 100 degrees Celsius, which is 212 degrees Fahrenheit.'),
    ('Why is exercise good for you?', 'it keeps the heart and body strong.', 'Regular exercise strengthens the heart and muscles, helps control weight and improves mood and sleep.'),
    ('What is a prime number?', 'a number with exactly two divisors.', 'A prime number can only be divided evenly by 1 and by itself, like 2, 3, 5, 7 and 11.'),
    ('How do vaccines work?', 'they train the immune system.', 'A vaccine shows the immune system a harmless piece of a germ, so it can react quickly if the real germ arrives.'),
    ('What causes rain?', 'water vapour cooling into drops.', 'Warm air carries water vapour upward, where it cools and condenses into droplets that grow heavy and fall.'),
    ('What is the largest ocean?', 'the Pacific Ocean.', 'The Pacific covers about a third of the Earth and is larger than all the land on the planet put together.'),
    ('Why do cats purr?', 'mostly to show contentment.', 'Cats usually purr when they are relaxed, but they can also purr to calm themselves when hurt or stressed.'),
    ('What is gravity?', 'the pull between masses.', 'Gravity is the force that pulls objects with mass toward each other, which is why things fall to the ground.'),
    ('How does a bicycle stay upright?', 'steering and forward motion.', 'As the bicycle moves, small steering corrections keep the wheels under the rider, so it does not tip over.'),
]
HELD_OUT = ['What is the capital of Italy?', 'Why do leaves change colour in autumn?', 'How many days are in a leap year?',
            'What is the speed of light?', 'Why do we need sleep?', 'What is an atom?', 'Who painted the Mona Lisa?',
            'Why is the sea salty?']


def chat_ids(q):
    return tok.apply_chat_template([{'role': 'user', 'content': q}], add_generation_prompt=True, tokenize=True, return_dict=False)


def seq_logp(model, q, answer):
    """log pi(answer | question) = sum of the log-probabilities of the answer's tokens."""
    p = chat_ids(q)
    a = tok(answer + '<|im_end|>', add_special_tokens=False)['input_ids']
    ids = torch.tensor([p + a], device=DEV)
    logits = model(ids).logits[0, len(p) - 1:-1]
    return torch.log_softmax(logits.float(), -1).gather(1, torch.tensor(a, device=DEV)[:, None]).sum()


def pair(q, s, d):
    return q, f'In short: {s} {d}', d


def dpo_terms(q, yw, yl):
    lw, ll = seq_logp(policy, q, yw), seq_logp(policy, q, yl)
    with torch.no_grad():
        rw, rl = seq_logp(reference, q, yw), seq_logp(reference, q, yl)
    margin = BETA * ((lw - rw) - (ll - rl))
    return -F.logsigmoid(margin), margin, (lw, rw, ll, rl)


# ---------------------------------------------------------------- Part A: worked numbers on one pair
log('== A. the DPO loss on one pair, before any training ==')
q, yw, yl = pair(*DATA[0])
log(f'prompt   x  : {q}')
log(f'chosen   y_w: {yw}')
log(f'rejected y_l: {yl}')
with torch.no_grad():
    loss, margin, (lw, rw, ll, rl) = dpo_terms(q, yw, yl)
log(f'log pi(y_w|x) = {lw:.3f}   log pi_ref(y_w|x) = {rw:.3f}   (identical: the policy starts as a copy of the reference)')
log(f'log pi(y_l|x) = {ll:.3f}   log pi_ref(y_l|x) = {rl:.3f}')
log(f'margin = beta*[(log pi(y_w)-log pi_ref(y_w)) - (log pi(y_l)-log pi_ref(y_l))] = {margin:.4f}; loss = -log sigmoid(margin) = {loss:.4f} = ln 2')
A = dict(q=q, yw=yw, yl=yl, before=dict(lw=float(lw), rw=float(rw), ll=float(ll), rl=float(rl), margin=float(margin), loss=float(loss)))


def eval_pairs(qs_pairs):
    out = []
    with torch.no_grad():
        for q, yw, yl in qs_pairs:
            loss, m, _ = dpo_terms(q, yw, yl)
            out.append((float(loss), float(m)))
    return float(np.mean([o[0] for o in out])), float(np.mean([(o[1] > 0) + 0.5 * (o[1] == 0) for o in out])), float(np.mean([o[1] for o in out]))


@torch.no_grad()
def answers(model, qs, n=40):
    res = []
    for q in qs:
        ids = torch.tensor([chat_ids(q)], device=DEV)
        g = model.generate(ids, max_new_tokens=n, do_sample=False, temperature=None, top_p=None, top_k=None,
                           pad_token_id=tok.convert_tokens_to_ids('<|endoftext|>'), eos_token_id=tok.convert_tokens_to_ids('<|im_end|>'))
        res.append(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
    return res


# ---------------------------------------------------------------- Part B: train
train = [pair(*d) for d in DATA]
log('\n== B. a tiny DPO run: 16 pairs, chosen = "In short: ..." + explanation, rejected = explanation only ==')
before = answers(policy, HELD_OUT)
n0 = sum(a.startswith('In short') for a in before)
log(f'held-out questions answered with "In short:" BEFORE training: {n0}/8')
log(f'  e.g. {HELD_OUT[0]!r} -> {before[0][:70]!r}')
opt = torch.optim.AdamW(policy.parameters(), lr=1e-6, weight_decay=0.0)
clock = timer()
hist = []
STEPS = 40
for step in range(STEPS + 1):
    if step % 5 == 0:
        policy.eval()
        l_tr, acc_tr, m_tr = eval_pairs(train)
        with torch.no_grad():
            _, _, (lw, rw, ll, rl) = dpo_terms(*train[0])
        hist.append(dict(step=step, loss=l_tr, acc=acc_tr, margin=m_tr, lw=float(lw), ll=float(ll)))
        log(f'step {step:>2}  train loss {l_tr:.4f}  pairs ranked right {acc_tr:.2f}  mean margin {m_tr:+.3f}   '
            f'pair 0: log pi(y_w) {lw:.2f}, log pi(y_l) {ll:.2f}   [{clock():.0f}s]')
    if step == STEPS:
        break
    policy.train()
    batch = random.sample(train, 4)
    loss = sum(dpo_terms(*b)[0] for b in batch) / len(batch)
    opt.zero_grad(); loss.backward(); opt.step()

policy.eval()
with torch.no_grad():
    loss, margin, (lw, rw, ll, rl) = dpo_terms(q, yw, yl)
A['after'] = dict(lw=float(lw), rw=float(rw), ll=float(ll), rl=float(rl), margin=float(margin), loss=float(loss))
log(f'\npair 0 after training: log pi(y_w) {lw:.3f} (ref {rw:.3f}), log pi(y_l) {ll:.3f} (ref {rl:.3f}), '
    f'margin {margin:.3f}, loss {loss:.4f}')
log(f'implicit rewards: beta*log(pi/pi_ref) chosen = {BETA * (lw - rw):+.3f}, rejected = {BETA * (ll - rl):+.3f}')
after = answers(policy, HELD_OUT)
n1 = sum(a.startswith('In short') for a in after)
log(f'\nheld-out questions answered with "In short:" AFTER training: {n1}/8')
for qq, a in zip(HELD_OUT, after):
    log(f'  {qq:<42} -> {a[:72]!r}')
save('ch2_dpo', dict(A=A, hist=hist, before=before, after=after, held_out=HELD_OUT, n_before=n0, n_after=n1, beta=BETA))
