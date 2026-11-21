"""Part 5, §2.3.2: the language consistency reward, written two ways.

Reward_language = Num(Words_target) / Num(Words)   (the proportion of target-language words in the chain of thought)

Method 1: our own scanner. It walks the text one character at a time and asks the Unicode database
          (unicodedata.name) which script each letter belongs to: LATIN -> English, CJK -> Chinese.
Method 2: the `regex` library, which knows Unicode script properties (\\p{Latin}, \\p{Han}).
Then: a printed check that both agree on every string, and the final reward of §2.3.2 (accuracy + language)
for a small group of answers, with the GRPO advantage of Part 2.
No model is loaded here."""
import unicodedata
import regex
import common
import torch
from r1_part2_grpo import group_advantages

# ---------------------------------------------------------------- Method 1: our own word splitter
def script_of(ch):
    """'en' for a Latin letter, 'zh' for a CJK ideograph, None for digits, spaces, symbols and punctuation."""
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, '')
    if name.startswith('LATIN'):
        return 'en'
    if name.startswith('CJK UNIFIED IDEOGRAPH') or name.startswith('CJK COMPATIBILITY IDEOGRAPH'):
        return 'zh'
    return 'other'                                   # Greek, Cyrillic, ... count as words of another language


def words_scratch(text):
    """Split text into (word, language) pairs. A run of Latin letters is one English word.
    Chinese is written without spaces, so each Han character counts as one word."""
    out, cur, cur_lang = [], '', None
    for ch in text:
        s = script_of(ch)
        if s in ('en', 'other'):                     # letters that build a word
            if cur and cur_lang != s:                 # a change of script ends the current word
                out.append((cur, cur_lang)); cur = ''
            cur, cur_lang = cur + ch, s
            continue
        if cur:                                       # a space, digit, symbol or Han character ends it
            out.append((cur, cur_lang)); cur = ''
        if s == 'zh':
            out.append((ch, 'zh'))                    # one Han character = one word
    if cur:
        out.append((cur, cur_lang))
    return out


def lang_reward_scratch(cot, target):
    """Reward_language = Num(Words_target) / Num(Words). A text with no words at all gets 0."""
    ws = words_scratch(cot)
    if not ws:
        return 0.0, 0, 0
    n_target = sum(1 for _, lang in ws if lang == target)
    return n_target / len(ws), n_target, len(ws)


# ---------------------------------------------------------------- Method 2: the regex library and Unicode scripts
WORD = regex.compile(r'\p{Han}|\p{Latin}+|[^\P{L}\p{Latin}\p{Han}]+')   # one Han char, a Latin run, or a run of other letters


def lang_reward_regex(cot, target):
    ws = WORD.findall(cot)
    if not ws:
        return 0.0, 0, 0
    pat = {'en': r'\p{Latin}+', 'zh': r'\p{Han}'}[target]
    n_target = sum(1 for w in ws if regex.fullmatch(pat, w))
    return n_target / len(ws), n_target, len(ws)


def target_language(question):
    """The target language is the language of the question: the script with more words in it."""
    ws = words_scratch(question)
    zh = sum(1 for _, l in ws if l == 'zh')
    en = sum(1 for _, l in ws if l == 'en')
    return 'zh' if zh > en else 'en'


def main():
    log = common.Log('r1_part5_lang')
    # ---------------------------------------------------------------- the test strings
    CASES = [
        ('A: English question, English thinking',
         'What is 15% of 80?',
         'First, 10% of 80 is 8. Then 5% is half of that, so 4. Together 8 + 4 = 12. Let me check: 0.15 * 80 = 12.'),
        ('B: English question, thinking drifts into Chinese',
         'What is 15% of 80?',
         'First, 10% of 80 is 8. 然后 5% 是 4, 所以 8 + 4 = 12. Let me check: 0.15 * 80 = 12.'),
        ('C: Chinese question, Chinese thinking with two English terms',
         '80的15%是多少?',
         '先算 10%: 80 的 10% 是 8. 再算 5%: 是 4. 所以答案是 12. Double check: 0.15 * 80 = 12.'),
        ('D: Chinese question, thinking all in English',
         '80的15%是多少?',
         'First, 10% of 80 is 8. Then 5% is 4. So the answer is 12.'),
    ]

    log('Method 1 (our scanner + unicodedata) and Method 2 (regex + Unicode scripts)')
    log('')
    log('tiny example: how Method 1 splits case B')
    ws = words_scratch(CASES[1][2])
    log('  words:', ' '.join(f'{w}/{l}' for w, l in ws))
    log('')
    rows = []
    all_agree = True
    for name, q, cot in CASES:
        t = target_language(q)
        r1, n1, d1 = lang_reward_scratch(cot, t)
        r2, n2, d2 = lang_reward_regex(cot, t)
        agree = (n1, d1) == (n2, d2)
        all_agree &= agree
        log(f'{name}')
        log(f'  target={t}  method 1: {n1}/{d1} = {r1:.3f}   method 2: {n2}/{d2} = {r2:.3f}   agree: {agree}')
        rows.append(dict(case=name, target=t, n_target=n1, n_words=d1, reward=round(r1, 4)))
    log('')
    log(f'both methods agree on all {len(CASES)} strings: {all_agree}')

    # the unicodedata names behind Method 1, for three characters
    log('')
    for ch in ['a', '然', '%']:
        log(f'  unicodedata.name({ch!r}) = {unicodedata.name(ch)!r}  -> {script_of(ch)}')

    # ---------------------------------------------------------------- §2.3.2: final reward = accuracy + language
    log('')
    log('final reward of §2.3.2 for a group of 4 answers to the English question (accuracy 1 or 0, plus language)')
    group = [
        ('right, English', 1, CASES[0][2]),
        ('right, mixed', 1, CASES[1][2]),
        ('wrong, English', 0, 'First, 10% of 80 is 8. Then 5% is 5. So 8 + 5 = 13.'),
        ('wrong, mixed', 0, 'First, 10% of 80 is 8. 然后 5% 是 5, 所以 13.'),
    ]
    rewards = []
    for name, acc, cot in group:
        lc, _, _ = lang_reward_scratch(cot, 'en')
        rewards.append(acc + lc)
    # Equation 3 with Part 2's own helper: std divides by G - 1 (as torch.std and TRL do), plus eps = 1e-4
    adv = group_advantages(torch.tensor(rewards), G=len(rewards)).tolist()
    grp = []
    for (name, acc, cot), r, a in zip(group, rewards, adv):
        lc = r - acc
        log(f'  {name:15s} accuracy={acc}  language={lc:.3f}  reward={r:.3f}  advantage={a:+.3f}')
        grp.append(dict(name=name, accuracy=acc, language=round(lc, 4), reward=round(r, 4), advantage=round(a, 4)))
    # and the same group with the accuracy reward alone
    acc_only = [g[1] for g in group]
    adv2 = group_advantages(torch.tensor(acc_only, dtype=torch.float), G=4).tolist()
    log('  accuracy alone gives advantages ' + ', '.join(f'{a:+.3f}' for a in adv2)
        + ' : the two right answers look the same')
    log('  (advantages from group_advantages() of r1_part2_grpo.py: std over G - 1, eps 1e-4)')

    common.save('r1_part5_lang', dict(cases=rows, group=grp, agree=all_agree, split_B=[list(x) for x in ws]))


if __name__ == '__main__':
    main()
