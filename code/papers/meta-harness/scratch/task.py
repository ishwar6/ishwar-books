"""The task: classify a bank customer's message into one of 10 card intents (a slice of Banking77).

Banking77 (Casanueva et al., 2020) is one of the nine out-of-distribution datasets of the paper (Table 5).
We keep 10 of its 77 intents, all about cards, so the labels are close to each other and a small model makes mistakes.

Three disjoint sets, as in the paper's online text classification setup (§4.1):
  memory  60 labeled messages (6 per intent). The harness sees them one at a time through update(text, label).
  search  50 messages (5 per intent). Each candidate harness is scored on them; the proposer sees these scores and traces.
  test   100 messages (10 per intent), from Banking77's own test split. The proposer never sees them.

The data comes from the PolyAI GitHub CSV files. The first call downloads them (about 1 MB) and writes data/task.json,
a small file with only the 210 messages we use. Every later call reads data/task.json."""
import csv, json, os, random, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_FILE = os.path.join(HERE, 'data', 'task.json')
CACHE = os.path.expanduser('~/.cache/mh_banking77')
URL = 'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}.csv'

LABELS = [
    'activate_my_card', 'card_arrival', 'card_delivery_estimate', 'card_not_working', 'card_payment_fee_charged',
    'card_payment_not_recognised', 'compromised_card', 'declined_card_payment', 'lost_or_stolen_card',
    'pending_card_payment',
]
N_MEMORY, N_SEARCH, N_TEST = 6, 5, 10      # per intent
SEED = 0


def _csv_rows(split):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f'banking77_{split}.csv')
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL.format(split), path)
    return [(r['text'].strip(), r['category']) for r in csv.DictReader(open(path, encoding='utf-8'))]


def build():
    """Sample the three sets once, with a fixed seed, and save them."""
    rng = random.Random(SEED)
    train, test = _csv_rows('train'), _csv_rows('test')
    sets = {'memory': [], 'search': [], 'test': []}
    for label in LABELS:
        pool = [t for t, c in train if c == label]
        rng.shuffle(pool)
        sets['memory'] += [{'text': t, 'label': label} for t in pool[:N_MEMORY]]
        sets['search'] += [{'text': t, 'label': label} for t in pool[N_MEMORY:N_MEMORY + N_SEARCH]]
        held = [t for t, c in test if c == label]
        rng.shuffle(held)
        sets['test'] += [{'text': t, 'label': label} for t in held[:N_TEST]]
    for name in sets:                       # mix the intents, so the stream is not sorted by label
        rng.shuffle(sets[name])
    os.makedirs(os.path.dirname(TASK_FILE), exist_ok=True)
    json.dump({'labels': LABELS, **sets}, open(TASK_FILE, 'w'), indent=1)


def load():
    """Return (labels, memory, search, test). Each set is a list of {'text', 'label'} dicts."""
    if not os.path.exists(TASK_FILE):
        build()
    d = json.load(open(TASK_FILE))
    if os.environ.get('MH_SMOKE'):                 # a tiny version of every set, to test the code in seconds
        return d['labels'], d['memory'], d['search'][:6], d['test'][:6]
    return d['labels'], d['memory'], d['search'], d['test']


if __name__ == '__main__':
    labels, memory, search, test = load()
    print(f'{len(labels)} labels, memory {len(memory)}, search {len(search)}, test {len(test)}')
    for ex in search[:3]:
        print(f"  {ex['label']:28s} {ex['text']}")
