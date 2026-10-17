"""The toy task: turn an ISO date into a date for people.

    "2021-03-12"  ->  "12 March 2021"

The day, the month and the year change places, and the two month digits become a word. So each output character must
find the right input characters: the model has to learn where to look. That is the job of attention.

Both sides use one small character vocabulary (shared, as in §3.4 of the paper).
"""
import datetime
import random

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']

SPECIALS = ['<pad>', '<s>', '</s>']                     # ids 0, 1, 2
CHARS = sorted(set('0123456789- ' + ''.join(MONTHS)))  # digits, '-', space and the letters of the month names
VOCAB = SPECIALS + CHARS
STOI = {s: i for i, s in enumerate(VOCAB)}
PAD, BOS, EOS = 0, 1, 2

FIRST = datetime.date(1900, 1, 1)
LAST = datetime.date(2099, 12, 31)


def iso(d):
    return d.strftime('%Y-%m-%d')                       # "2021-03-12"


def human(d):
    return f'{d.day} {MONTHS[d.month - 1]} {d.year}'   # "12 March 2021" (no leading zero on the day)


def encode_src(text):
    """Source ids: the characters, then </s>."""
    return [STOI[c] for c in text] + [EOS]


def encode_tgt(text):
    """Target ids: <s>, the characters, then </s>."""
    return [BOS] + [STOI[c] for c in text] + [EOS]


def decode_ids(ids):
    """ids -> text; stop at </s>, skip <s> and <pad>."""
    out = []
    for i in ids:
        if i == EOS:
            break
        if i in (PAD, BOS):
            continue
        out.append(VOCAB[i])
    return ''.join(out)


def all_dates():
    n = (LAST - FIRST).days + 1
    return [FIRST + datetime.timedelta(days=k) for k in range(n)]


def split_dates(n_test=2000, seed=0):
    """Every date from 1900 to 2099 (73,049 dates). n_test of them are held out and never used in training."""
    dates = all_dates()
    rng = random.Random(seed)
    rng.shuffle(dates)
    return dates[n_test:], dates[:n_test]
