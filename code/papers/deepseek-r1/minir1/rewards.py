"""The two rule-based rewards of §2.2.2. No neural reward model: a few lines of Python decide each reward."""
import re
from fractions import Fraction

ANSWER_RE = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
TAG_RE = re.compile(r'</?(think|answer)>')
# One number: '-3', '12.5', '1/2', '-7/4', '1e3'. The lookarounds stop it from starting or ending inside a word.
NUMBER_RE = re.compile(r'(?<![\w.])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?(?!\w|\.\d)')
OPERATOR_RE = re.compile(r'\d\s*(?:[-+*×÷^])\s*\(?-?\d')     # '10 - 2', '3 * 4', '2^3': an expression, not a number


def extract_answer(completion):
    """The text inside the last <answer> ... </answer> pair, or None when there is no such pair."""
    found = ANSWER_RE.findall(completion)
    return found[-1].strip() if found else None


def extract_boxed(text):
    """What is inside the last \\boxed{...}, counting braces so that '\\boxed{\\frac{1}{2}}' works. None if unclosed."""
    start = text.rfind('\\boxed')
    i = text.find('{', start) if start >= 0 else -1
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        if depth == 0:                                   # the brace that closes the box
            return text[i + 1:j]
    return None


def to_number(text):
    """The final value of an answer as an exact Fraction, or None when it is not one number.
    '$1,250.00' -> 1250, 'x = -3' -> -3, '3 + 4 = 7' -> 7, '1/2' -> 1/2, '\\boxed{36/2}' -> 18, '1e3' -> 1000,
    '10 - 2' -> None (an unworked expression), '2^3' -> None, 'none' -> None."""
    if '\\boxed' in text:
        text = extract_boxed(text)                       # the box is the final answer
        if text is None:
            return None
    text = text.split('=')[-1]                           # the final expression: what comes after the last '='
    text = re.sub(r'\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}', r'\1/\2', text)     # \frac{a}{b} -> a/b
    text = re.sub(r'\\text\{[^{}]*\}|\\[$%,!]|[$%]', ' ', text)               # unit text, dollar and per cent signs
    while re.search(r'\d,\d{3}(?!\d)', text):
        text = re.sub(r'(\d),(\d{3})(?!\d)', r'\1\2', text)                    # 1,250,000 -> 1250000
    if OPERATOR_RE.search(text):
        return None                                      # a sum or a power left unworked: no single number
    found = NUMBER_RE.findall(text)
    if not found:
        return None
    a, b = found[-1]                                     # the last number of the final expression
    value = Fraction(float(a)) if 'e' in a.lower() else Fraction(a)
    return None if b and float(b) == 0 else (value / Fraction(b) if b else value)


def accuracy_reward(completion, gold):
    """1.0 when the number in the <answer> tags equals the known answer, else 0.0 (the 'accuracy reward')."""
    answer = extract_answer(completion)
    if answer is None:
        return 0.0                                   # no answer tags: nothing to check
    value = to_number(answer)
    return float(value is not None and value == Fraction(gold))


def format_reward(completion):
    """1.0 when the completion is exactly <think> ... </think> <answer> ... </answer>: each tag once, in this order,
    both blocks non-empty, nothing before, between or after them (the 'format reward'). Part 3's tag parser."""
    tags = list(TAG_RE.finditer(completion))
    if [t.group(0) for t in tags] != ['<think>', '</think>', '<answer>', '</answer>']:
        return 0.0                                   # a tag is missing, repeated or out of order
    t1, t2, a1, a2 = tags
    if completion[:t1.start()].strip() or completion[t2.end():a1.start()].strip() or completion[a2.end():].strip():
        return 0.0                                   # text before, between or after the two blocks
    if not completion[t1.end():t2.start()].strip() or not completion[a1.end():a2.start()].strip():
        return 0.0                                   # an empty block
    return 1.0


def total_reward(completion, gold):
    """The reward GRPO sees. Version 1 of the paper names the two rules but not how they combine; the later version
    adds them with the same weight (its Eq. 4). So one answer earns 0, 1 or 2."""
    return accuracy_reward(completion, gold) + format_reward(completion)
