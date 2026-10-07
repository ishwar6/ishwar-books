"""Chapter 4: typed, exact graders shared by ch4_router.py and ch4_parallel.py, and a self-test on adversarial strings.
A grader must never give credit because the right answer appears somewhere inside a wrong one ("184" contains "84"; "not 84,
the answer is 74" contains "84"). So:
  numbers: the reply must contain EXACTLY ONE number (after removing thousands separators and a trailing unit word); it is compared
           numerically with the gold value (absolute tolerance 1e-6, or 0.005 for money written with two decimals).
  times:   exactly one HH:MM token, compared as minutes.
  words:   the reply, lower-cased, stripped of punctuation and of a leading "the answer is", must EQUAL the gold word (or a listed alias).
Run this file to print the self-test (results/ch4_grade_stdout.txt)."""
import re

NUM = re.compile(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.]*\d)')


def _clean(s):
    s = (s or '').strip().lower()
    s = re.sub(r'^(the )?(final )?answer( is)?[:\s]*', '', s)
    s = s.replace(',', '')                               # thousands separators: "1,200" -> "1200"
    return s.strip().strip('.').strip()


def numbers_in(s):
    s = _clean(s)
    s = re.sub(r'(\d)\s*(km/h|km|kg|g|grams|litres|minutes|metres|m|%|degrees|°)', r'\1', s)
    s = re.sub(r'[£$€]', '', s)
    return [float(x) for x in NUM.findall(s)]


def grade_number(reply, gold, tol=1e-6):
    """Correct iff the reply contains exactly one number and it equals gold within tol."""
    ns = numbers_in(reply)
    return len(ns) == 1 and abs(ns[0] - float(gold)) <= tol


def grade_time(reply, gold):
    ts = re.findall(r'\b(\d{1,2}):(\d{2})\b', reply or '')
    if len(ts) != 1:
        return False
    gh, gm = (int(x) for x in gold.split(':'))
    return int(ts[0][0]) * 60 + int(ts[0][1]) == gh * 60 + gm


def grade_word(reply, gold, aliases=()):
    w = re.sub(r'[^a-z\- ]', '', _clean(reply)).strip()
    return w in {gold.lower(), *[a.lower() for a in aliases]}


def grade(reply, gold):
    """Dispatch on the type of the gold answer: HH:MM, a number, or a word."""
    if re.fullmatch(r'\d{1,2}:\d{2}', gold):
        return grade_time(reply, gold)
    if re.fullmatch(r'-?\d+(\.\d+)?', gold):
        return grade_number(reply, gold, tol=0.005 if '.' in gold else 1e-6)
    return grade_word(reply, gold)


CASES = [  # (reply, gold, expected verdict, why)
    ('84', '84', True, 'plain match'),
    ('184', '84', False, 'gold is a substring of a wrong number'),
    ('13', '3', False, 'gold digit inside a wrong number'),
    ('It is not 84; the answer is 74.', '84', False, 'two numbers: refuse to guess'),
    ('The answer is 84.', '84', True, 'leading phrase and full stop'),
    ('84.0', '84', True, 'numeric, not string, comparison'),
    ('1,200', '1200', True, 'thousands separator'),
    ('1,200 g', '1200', True, 'trailing unit'),
    ('14.2', '14.20', True, 'money written with one decimal'),
    ('14.25', '14.20', False, 'money off by 5p'),
    ('-6', '6', False, 'sign matters'),
    ('6 trailing zeros', '6', True, 'number with a unit phrase'),
    ('2496 (48 x 52)', '2496', False, 'working shown: three numbers, refuse'),
    ('14:05', '14:05', True, 'time'),
    ('2:05 pm', '14:05', False, 'time in another format is not accepted'),
    ('Saturday', 'saturday', True, 'case'),
    ('Saturday or Sunday', 'saturday', False, 'hedged answer'),
    ('lisbon.', 'lisbon', True, 'punctuation'),
    ('Lisbon, Portugal', 'lisbon', False, 'extra entity'),
]


if __name__ == '__main__':
    from common import Log
    log = Log('ch4_grade')
    log('== Grader self-test: typed comparison on adversarial strings ==')
    log(f'{"reply":<34} {"gold":<9} {"expected":<9} {"got":<6} why')
    bad = 0
    for reply, gold, exp, why in CASES:
        got = grade(reply, gold)
        bad += got != exp
        log(f'{reply!r:<34} {gold:<9} {str(exp):<9} {str(got):<6} {why}' + ('   <-- MISMATCH' if got != exp else ''))
    log(f'{len(CASES) - bad}/{len(CASES)} cases behave as specified' + ('' if not bad else '  (FIX THE GRADER)'))
