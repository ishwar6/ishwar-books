"""Chapter 4: what a base model learned from pretraining alone. Probes Qwen2.5-0.5B (base, no fine-tuning) for facts,
grammar and in-context learning, and our own 12M TinyStories GPT for grammar inside its tiny world."""
import random, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import Log, save
from ch4_gpt import GPT, get_tokenizer

log = Log('ch4_probe')
dev = 'mps'
tk = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
lm = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', dtype=torch.float32).to(dev).eval()
R = {}


@torch.no_grad()
def next_probs(prompt, k=5):
    ids = tk(prompt, return_tensors='pt').input_ids.to(dev)
    p = F.softmax(lm(ids).logits[0, -1].float(), -1)
    v, i = p.topk(k)
    return [(tk.decode([j]), float(q)) for q, j in zip(v.tolist(), i.tolist())]


@torch.no_grad()
def logprob(prompt, cont):
    """Sum of log-probabilities of the continuation tokens given the prompt."""
    a = tk(prompt).input_ids; b = tk(cont).input_ids
    ids = torch.tensor([a + b], device=dev)
    lp = F.log_softmax(lm(ids).logits[0].float(), -1)
    return sum(lp[len(a) + i - 1, t].item() for i, t in enumerate(b))


log('== 1. facts: top-5 next tokens ==')
R['facts'] = {}
for p in ['The capital of France is', 'Water freezes at a temperature of', 'The chemical symbol for gold is',
          'Romeo and Juliet was written by', 'The largest planet in the solar system is']:
    top = next_probs(p)
    R['facts'][p] = top
    log(f'{p!r:45s} ' + '  '.join(f'{t!r} {q:.2f}' for t, q in top))

log('')
log('== 2. grammar: log P(right) vs log P(wrong) ==')
pairs = [('The keys to the cabinet', ' are', ' is'), ('The author of the books', ' is', ' are'),
         ('Yesterday she', ' went', ' goes'), ('The children who live next door', ' are', ' is'),
         ('I have never', ' seen', ' saw'), ('Each of the students', ' has', ' have')]
R['grammar'] = []
for p, good, bad in pairs:
    a, b = logprob(p, good), logprob(p, bad)
    log(f'{p!r:36s} {good!r:7s} {a:7.2f}   {bad!r:7s} {b:7.2f}   prefers the right one: {a > b}  (ratio {torch.tensor(a - b).exp().item():,.1f}x)')
    R['grammar'].append({'prompt': p, 'good': good, 'bad': bad, 'lp_good': a, 'lp_bad': b})

log('')
log('== 3. in-context learning: accuracy vs number of examples in the prompt ==')
fr = [('dog', 'chien'), ('cat', 'chat'), ('house', 'maison'), ('water', 'eau'), ('book', 'livre'), ('red', 'rouge'),
      ('apple', 'pomme'), ('car', 'voiture'), ('tree', 'arbre'), ('sun', 'soleil'), ('moon', 'lune'), ('bread', 'pain'),
      ('cheese', 'fromage'), ('school', 'école'), ('friend', 'ami'), ('window', 'fenêtre'), ('milk', 'lait'), ('green', 'vert'),
      ('city', 'ville'), ('sea', 'mer'), ('horse', 'cheval'), ('bird', 'oiseau'), ('flower', 'fleur'), ('door', 'porte'),
      ('night', 'nuit'), ('day', 'jour'), ('king', 'roi'), ('hand', 'main'), ('black', 'noir'), ('white', 'blanc'),
      ('table', 'table'), ('chair', 'chaise'), ('fish', 'poisson'), ('rain', 'pluie'), ('snow', 'neige'), ('book shop', 'librairie')]
ant = [('hot', 'cold'), ('big', 'small'), ('fast', 'slow'), ('happy', 'sad'), ('up', 'down'), ('light', 'dark'),
       ('old', 'new'), ('early', 'late'), ('rich', 'poor'), ('strong', 'weak'), ('open', 'closed'), ('full', 'empty'),
       ('hard', 'soft'), ('wet', 'dry'), ('tall', 'short'), ('good', 'bad'), ('loud', 'quiet'), ('first', 'last'),
       ('true', 'false'), ('win', 'lose'), ('push', 'pull'), ('give', 'take'), ('buy', 'sell'), ('love', 'hate'),
       ('clean', 'dirty'), ('thick', 'thin'), ('young', 'old'), ('high', 'low'), ('cheap', 'expensive'), ('inside', 'outside'),
       ('always', 'never'), ('remember', 'forget'), ('arrive', 'leave'), ('safe', 'dangerous'), ('heavy', 'light'), ('wide', 'narrow')]
pos = ['I loved this film', 'What a wonderful day', 'The food was delicious', 'Great service and kind staff', 'This book is brilliant',
       'I am so happy with it', 'An excellent and moving story', 'Best purchase I ever made', 'The view was beautiful',
       'Fantastic acting', 'It works perfectly', 'A joy to read', 'Superb quality', 'They were very helpful', 'I would buy it again',
       'Everything was perfect', 'Such a fun evening', 'Highly recommended']
neg = ['I hated this film', 'What an awful day', 'The food was cold and bland', 'Rude staff and slow service', 'This book is boring',
       'I am very disappointed', 'A dull and pointless story', 'Worst purchase I ever made', 'The room was dirty', 'Terrible acting',
       'It broke after one day', 'A chore to read', 'Cheap and flimsy', 'They ignored us', 'I want my money back',
       'Everything went wrong', 'Such a waste of time', 'Avoid at all costs']
senti = [(s, 'blue') for s in pos] + [(s, 'red') for s in neg]          # made-up labels: blue = positive, red = negative
TASKS = {'English to French': (fr, ' ->', 'Translate English to French:\n'),
         'antonyms': (ant, ' ->', 'Write the opposite of each word:\n'),
         'sentiment with made-up labels (blue/red)': (senti, ' :', 'Label each review:\n')}


@torch.no_grad()
def answer(prompt):
    ids = tk(prompt, return_tensors='pt').input_ids.to(dev)
    out = lm.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=tk.eos_token_id)
    return tk.decode(out[0, ids.shape[1]:]).split('\n')[0].strip()


R['icl'] = {}
for name, (data, sep, head) in TASKS.items():
    rng = random.Random(0)
    data = data[:]
    rng.shuffle(data)
    pool, test = data[:8], data[8:]
    row = {}
    for k in [0, 1, 2, 4, 8]:
        shots = head + ''.join(f'{x}{sep} {y}\n' for x, y in pool[:k])     # the prompt ends at the separator, no space
        correct = sum(answer(shots + f'{x}{sep}').lower().startswith(y) for x, y in test)
        row[k] = correct / len(test)
    R['icl'][name] = row
    log(f'{name:42s} ' + '  '.join(f'k={k}: {v:.0%}' for k, v in row.items()) + f'   ({len(test)} test items)')
ex = 'Label each review:\n' + ''.join(f'{x} : {y}\n' for x, y in senti[:2] + senti[18:20])
log('example 4-shot prompt for the made-up labels task:')
for line in (ex + 'The view was beautiful :').split('\n'):
    log('   ' + line)

log('what the model answers with only two labelled examples:')
R['icl_k2'] = []
for q in ['The view was beautiful', 'The room was dirty', 'Superb quality']:
    a = answer('Label each review:\nThe food was delicious : blue\nTerrible acting : red\n' + q + ' :')
    log(f'   {q!r} -> {a!r}')
    R['icl_k2'].append([q, a])
log('')
log('== 4. our 12M TinyStories GPT: grammar inside its tiny world ==')
ts = get_tokenizer(4096)
m = GPT(4096).to(dev)
m.load_state_dict(torch.load('/Users/admin/.cache/tinystories/ch4_tinygpt.pt'))
m.eval()
R['tiny'] = {}
for p in ['Lily and her mom went to the', 'Tom was very sad because he lost his', 'The little girl smiled because she was very']:
    ids = torch.tensor([ts.encode(p).ids], device=dev)
    with torch.no_grad():
        pr = F.softmax(m(ids)[0, -1].float(), -1)
    v, i = pr.topk(5)
    top = [(ts.decode([j]), float(q)) for q, j in zip(v.tolist(), i.tolist())]
    R['tiny'][p] = top
    log(f'{p!r:48s} ' + '  '.join(f'{t!r} {q:.2f}' for t, q in top))
save('ch4_probe', R)
