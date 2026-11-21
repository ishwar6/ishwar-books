---
title: "Rewards and the Template"
description: "How R1-Zero scores its own answers with two simple rules, why the authors refused a learned reward model, and the one-paragraph prompt that starts training. We write each reward two ways, game a weak reward on purpose, and test two base models in the real template."
part: 3
covers: "§2.2.2, §2.2.3, Table 1"
date: 2026-11-21
tags: [deepseek, reasoning, reinforcement-learning, reward-hacking, rewards]
---

Reinforcement learning only sees one number per answer, so whoever writes that number decides what the model becomes. R1-Zero's number comes from two short rules and one paragraph of prompt, and this part takes both apart line by line.

{{FIG:p3_here|The training loop of R1-Zero. Part 2 explained the GRPO update. This part covers the two highlighted boxes: the template that wraps each question, and the rules that turn each sampled output into a reward.}}

## Reward modeling {§2.2.2}

> [!PAPER] · §2.2.2 · page 6
> [![The first lines of §2.2.2, with The reward is the source of the training signal, decides the optimization direction of RL and rule-based reward system highlighted](/img/papers/deepseek-r1/p3-reward-source.png)](/img/papers/deepseek-r1/p3-reward-source.png)
>
> **What it says:** "The reward is the source of the training signal, which decides the optimization direction of RL." For R1-Zero the authors "adopt a rule-based reward system" with two kinds of reward.
>
> **What it means:**
>
> Think of a dog trainer with a bag of treats. The dog does not understand words like "sit nicely". It only learns which actions earn a treat. If the trainer gives a treat for barking by mistake, the dog learns to bark. The treat is the whole lesson.
>
> RL works the same way. The model writes an answer, a reward scores it, and GRPO (see [Part 2](part-2-grpo.md)) pushes the model toward the answers that scored above their group. Nothing else reaches the weights. So every choice inside the reward is a choice about what the model will learn.
>
> The key words are "rule-based". The reward is a short program that a person can read, not a neural network. The rest of this section explains what the two rules check, and why the authors chose rules over a network.

> [!DEFINITION] Rule-based reward
> A reward computed by a fixed program from the output and a known right answer, with no trained weights. It can only check what a program can check: a final number, a test that passes, a tag in the right place. R1-Zero uses two: an accuracy reward and a format reward.

### Background: the reward model of RLHF

To see why "rule-based" is a choice, look at what came before. The assistants of 2022 to 2024 were trained with **reinforcement learning from human feedback** (RLHF). There is no answer key for "write a kind email", so people compare two answers and pick the better one. A second network, the **reward model**, learns from many such comparisons. Then RL trains the language model to get high scores from the reward model.

The reward model learns with one equation. Give it a prompt $$x$$, the answer people chose $$y_w$$ and the answer they rejected $$y_l$$. The network $$r_\phi$$ gives each answer a score, and the loss pushes the chosen score above the rejected one:

$$
\mathcal{L}(\phi) = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)
$$

Read it from the inside. $$r_\phi(x, y)$$ is the score that the network with weights $$\phi$$ gives answer $$y$$. The difference is the margin between the chosen and the rejected answer. The sigmoid $$\sigma(z) = 1/(1+e^{-z})$$ turns the margin into a probability that a person prefers $$y_w$$. The minus log makes the loss small when that probability is near 1.

A tiny example. The network scores the chosen answer 1.2 and the rejected one 0.4. The margin is 0.8, and $$\sigma(0.8) = 0.690$$. The loss is $$-\log 0.690 = 0.371$$. If the scores were the other way round, the loss would be 1.171, and training would move the weights to fix the order. Our script checks this by hand and with PyTorch's `logsigmoid`:

```text
== 0. reward model loss for one human comparison (Bradley-Terry)
r(chosen)=+1.2  r(rejected)=+0.4  P(chosen preferred)=0.690  loss ours=0.371  torch=0.371
r(chosen)=+0.4  r(rejected)=+1.2  P(chosen preferred)=0.310  loss ours=1.171  torch=1.171
r(chosen)=+3.0  r(rejected)=-3.0  P(chosen preferred)=0.998  loss ours=0.002  torch=0.002
```

> [!DEFINITION] Reward model
> A neural network trained to give a score to a (prompt, answer) pair. In RLHF it learns from human comparisons with the loss above, and it then stands in for the humans during RL. It is a guess about what people want, learned from a finite set of examples.

Note the last sentence of that box. The reward model is a stand-in. RL then pushes hard on that stand-in for thousands of steps. That is where the trouble starts, as the section on reward hacking shows.

DeepSeek's own earlier work used this kind of network for maths too. GRPO comes from the DeepSeekMath paper, and there the score came from a trained reward model.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.1.2 · page 14
> [![The start of §4.1.2 of DeepSeekMath, Outcome Supervision RL with GRPO, with A reward model is then used to score the outputs highlighted](/img/papers/deepseek-r1/p3-dsm-outcome.png)](/img/papers/deepseek-r1/p3-dsm-outcome.png)
>
> **What it says:** in outcome supervision with GRPO, a group of outputs is sampled for each question, and "A reward model is then used to score the outputs".
>
> **What it means:**
>
> This is the same group recipe that R1 uses, with one difference: the scorer. In DeepSeekMath a trained network read each answer and gave it a score. The same paper also tried process supervision, where the network scores every step of the reasoning.
>
> So the R1 authors knew both kinds of neural reward from their own work. Dropping them was not a shortcut taken by people who had never tried. It was a decision, and the paper gives its reasons at the end of §2.2.2.

> [!DEFINITION] Outcome and process reward models
> An **outcome reward model** (ORM) is a network that scores a whole answer once, at the end. A **process reward model** (PRM) is a network that scores each step of the reasoning. Both are "neural" reward models: trained networks, not rules.

### Accuracy rewards

> [!PAPER] · §2.2.2 · page 6
> [![The Accuracy rewards bullet of §2.2.2, with final answer in a specified format (e.g., within a box), reliable rule-based verification of correctness, a compiler can be used to generate feedback and predefined test cases highlighted](/img/papers/deepseek-r1/p3-accuracy.png)](/img/papers/deepseek-r1/p3-accuracy.png)
>
> **What it says:** for maths problems "with deterministic results", the model must give its "final answer in a specified format (e.g., within a box)", which enables "reliable rule-based verification of correctness". For LeetCode problems, "a compiler can be used to generate feedback based on predefined test cases".
>
> **What it means:**
>
> This is a teacher with an answer key. The teacher does not grade the working or the handwriting. The teacher looks at the final answer in the box and ticks it or crosses it. That is fast, cheap and fair, and it works on millions of answers.
>
> The box solves a practical problem. A reasoning model may write 2,000 tokens with dozens of numbers in them. A program cannot guess which number is the answer. So the model must put its final answer in a fixed place, `\boxed{...}` in LaTeX, and the checker reads only that place.
>
> "Deterministic results" is the condition. The rule works only when each question has one right answer that a program can compare: a number, a fraction, an expression. For code, the answer key is a set of tests. The program runs, and either it passes them or it does not.

As an equation, the accuracy reward of an output $$o$$ for a question with the right answer $$a^\star$$ is

$$
r_{\text{acc}}(o) = \begin{cases} 1 & \text{if } \operatorname{norm}\big(\operatorname{box}(o)\big) = \operatorname{norm}(a^\star) \\ 0 & \text{otherwise.} \end{cases}
$$

Read it from the inside. The function $$\operatorname{box}(o)$$ takes the text inside the last `\boxed{...}` of the output. The function $$\operatorname{norm}$$ rewrites an answer into one standard spelling, so that $$0.5$$ and $$\tfrac{1}{2}$$ look the same. The reward is 1 when the two spellings match and 0 otherwise. There is no partial credit and no score for nice working.

The equation hides two hard parts, and both are about strings. First, find the box. Second, decide when two different strings are the same number.

### Method 1: our own maths checker

Finding the box looks like a one-line regular expression. It is not. LaTeX answers contain their own braces, as in `\boxed{\frac{\sqrt{3}}{2}}`. A pattern that stops at the first `}` cuts the answer in half. Our function counts braces instead, like a person checking brackets in a long formula:

```python
def extract_boxed(text):
    """Return what is inside the LAST \\boxed{...} of text, or None. Counts braces, so nested braces work:
    '\\boxed{\\frac{\\sqrt{3}}{2}}' -> '\\frac{\\sqrt{3}}{2}'."""
    start = text.rfind('\\boxed')                  # the last box is the final answer
    if start < 0:
        return None
    i = text.find('{', start)                      # the opening brace of the box
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == '{':
            depth += 1                             # one level deeper
        elif text[j] == '}':
            depth -= 1                             # one level up
            if depth == 0:                         # back to the level of the box: this brace closes it
                return text[i + 1:j]
    return None                                    # the box never closes: no answer


def naive_boxed(text):
    """The one-line regex many people write first. It stops at the FIRST closing brace."""
    m = re.findall(r'\\boxed\{(.*?)\}', text)
    return m[-1] if m else None
```

Here are both on four outputs:

```text
== 1a. extract the last box: a naive regex vs our brace counter
'so the answer is \\boxed{18}.'                  naive regex: '18'                   ours: '18'
'Thus $\\boxed{\\frac{\\sqrt{3}}{2}}$.'          naive regex: '\\frac{\\sqrt{3'      ours: '\\frac{\\sqrt{3}}{2}'
'First \\boxed{4}, no wait: \\boxed{\\dfrac{3}{4}}' naive regex: '\\dfrac{3'            ours: '\\dfrac{3}{4}'
'The answer is \\boxed{\\frac{1}{2}'             naive regex: '\\frac{1'             ours: None
```

The regex fails on every answer with a fraction. In RL that bug is not a small error. Every right fraction would get reward 0, and the model would learn to avoid fractions. The third line shows why we take the last box: the model changed its mind, and the last box holds its final answer. The fourth line has a box that never closes, so our checker gives no answer at all.

Next, the spelling. The function `normalise` removes one harmless difference per line. Then `to_number` reads the result as an exact fraction when it can, so $$0.5$$ and $$\tfrac{1}{2}$$ become the same `Fraction(1, 2)`:

```python
def normalise(s):
    """Turn an answer string into one standard spelling. Each line removes one harmless difference."""
    s = s.strip()
    s = re.sub(r'\\text\{\s*([^{}]*?)\s*\}', r' \1', s)        # \text{ cm} -> ' cm' (keep the words, drop the macro)
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')   # display and text fractions are fractions
    s = s.replace('\\left', '').replace('\\right', '')         # bracket sizes do not change the value
    s = s.replace('\\$', '').replace('$', '')                  # dollar signs
    s = s.replace('\\%', '').replace('%', '')                  # per cent signs (see the table: this is a choice)
    s = s.replace('\\!', '').replace('\\,', '').replace('{,}', ',')   # LaTeX spacing
    s = re.sub(r'^\s*[a-zA-Z]\s*=\s*', '', s)                  # 'x = 3' -> '3'
    s = re.sub(r'\s*\b' + UNITS + r'\s*\.?$', '', s)           # a trailing unit word: '18 dollars' -> '18'
    s = re.sub(r'(\d),(\d\d\d)\b', r'\1\2', s)                 # thousands separator: 1,000 -> 1000
    s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)   # \frac{a}{b} -> (a)/(b)
    s = re.sub(r'\\frac(\d)(\d)', r'(\1)/(\2)', s)             # \frac12 -> (1)/(2)
    s = s.replace(' ', '').rstrip('.')
    return s


def to_number(s):
    """'(3)/(4)' -> Fraction(3, 4); '0.5' -> Fraction(1, 2); anything else -> None. Exact, no rounding."""
    sign = -1 if s.startswith('-(') else 1         # '-(1)/(2)' -> minus sign outside the brackets
    m = NUM.match(s[1:] if sign < 0 else s)
    if not m:
        return None
    a = Fraction(m.group(1))
    return sign * (a / Fraction(m.group(2)) if m.group(2) else a)


def same_answer(pred, gold):
    """Method 1 compare: equal as exact numbers if both are numbers, else equal as normalised strings."""
    p, g = normalise(pred), normalise(gold)
    pn, gn = to_number(p), to_number(g)
    if pn is not None and gn is not None:
        return pn == gn
    return p == g


def accuracy_reward_ours(output, gold):
    """Method 1: 1.0 if the last boxed answer equals the gold answer, else 0.0."""
    pred = extract_boxed(output)
    return float(pred is not None and same_answer(pred, gold))
```

`UNITS` is a list of common unit words, and `NUM` is a pattern for an integer, a decimal or a fraction of the two. Python's `Fraction` keeps numbers exact, so $$0.333$$ never equals $$\tfrac{1}{3}$$. That is on purpose: a checker that rounds would pay for a wrong answer that happens to be close.

### Method 2: the math-verify library

A hand-written checker grows one rule at a time, and it never covers everything. The `math-verify` library takes the other road. It parses LaTeX into SymPy objects, the symbolic maths library of Python, and asks SymPy whether the two are equal as maths:

```python
def accuracy_reward_lib(output, gold):
    """Method 2: math-verify parses both sides into SymPy objects and compares them as maths."""
    g = parse('$' + gold + '$')          # the gold answer, read as LaTeX
    p = parse(output)                    # the model output: math-verify finds the boxed answer itself
    return float(bool(g) and bool(p) and verify(g, p))
```

The later long version of the R1 paper says that its own maths grader works the same way.

> [!PAPER] DeepSeek-AI (2025), long version · Appendix, maths evaluation · page 78
> [![The evaluation note of the long version of the R1 paper, with use a rule-based grader and to parse expressions highlighted](/img/papers/deepseek-r1/p3-long-grader.png)](/img/papers/deepseek-r1/p3-long-grader.png)
>
> **What it says:** "Parse the final answer within \boxed{} and use a rule-based grader to determine if it equals the ground truth. Round numerical values as needed, and use 'SymPy' to parse expressions."
>
> **What it means:**
>
> This note comes from the later 86-page version of the paper, written for the evaluation of maths benchmarks. It confirms the picture of v1. The answer is read from the box, and a program, helped by SymPy, decides if it equals the right answer.
>
> It also shows that the authors face the same spelling problem we face. "Round numerical values as needed" is a rule about when 0.333 may count as one third. Every grader has to pick such rules, and v1 does not list them.

### Where the two checkers disagree

We gave both methods 15 tricky pairs. Each model output is `The answer is \boxed{<model answer>}.`, and 1 means "reward 1":

```text
== 1b. accuracy reward on tricky pairs (the model output is "The answer is \boxed{<pred>}.")
gold                 model answer            ours   lib  note
\frac{1}{2}          0.5                        1     1  
\dfrac{3}{4}         3/4                        1     1  
-\frac{1}{2}         -0.5                       1     1  
1000                 1,000                      1     1  
18                   18 \text{ dollars}         1     1  
18                   18 dollars                 1     0  DISAGREE
3                    x = 3                      1     1  
5                    \frac{10}{2}               1     1  
1000                 10^{3}                     0     1  DISAGREE
\frac{\sqrt{2}}{2}   \sqrt{2}/2                 0     1  DISAGREE
50\%                 0.5                        0     1  DISAGREE
\{1,2\}              \{2,1\}                    0     1  DISAGREE
\frac{1}{3}          0.333                      0     0  
2\pi                 6.28                       0     0  
(1,2)                (2,1)                      0     0  
the two methods agree on 10 of 15 pairs
```

The first eight rows are easy for both. Fractions, decimals, thousands separators, `x = 3` and a unit inside `\text{}` all count as right. The last three rows are right to fail. The decimal $$0.333$$ is not $$\tfrac{1}{3}$$, and $$6.28$$ is not $$2\pi$$. The point $$(2, 1)$$ is not the point $$(1, 2)$$.

The five disagreements teach the most:

| Pair | Ours | math-verify | Why |
|---|---|---|---|
| `18` vs `18 dollars` | 1 | 0 | We strip unit words. math-verify reads "dollars" as letters multiplied by 18. |
| `1000` vs `10^{3}` | 0 | 1 | We compare strings and simple fractions. SymPy computes the power. |
| `\frac{\sqrt{2}}{2}` vs `\sqrt{2}/2` | 0 | 1 | Both are the same expression. Our checker cannot read a square root. |
| `50\%` vs `0.5` | 0 | 1 | We drop the % sign, so we compare 50 with 0.5. SymPy reads 50% as 50/100. |
| `\{1,2\}` vs `\{2,1\}` | 0 | 1 | A set has no order. Only SymPy knows that these braces mean a set. |

Neither checker is "the right one". Ours is short and easy to audit, and it is strict. The library knows much more maths, and it trusts its parser. Each disagreement is a place where a right answer could get 0 or a wrong answer could get 1. In RL, both mistakes are paid back by the model: it learns to avoid spellings that the checker misreads. This is why real systems spend so much care on the grader, and why the paper stresses that it must be "reliable".

### Code: a compiler and test cases

For code, the answer key is a set of tests. The checker must run code that a model wrote, so it runs it in a separate process with a time limit. An endless loop then costs two seconds, not the whole training run:

```python
def run_python(program, stdin='', timeout=2.0):
    """Run a Python program in a fresh process. Returns (exit code, stdout, stderr) or None on a timeout."""
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(program)
    try:
        r = subprocess.run([sys.executable, '-I', f.name], input=stdin, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return None                       # an endless loop gets no reward and does not hang the trainer
    finally:
        os.unlink(f.name)


HARNESS = """
import sys, importlib.util
nonce = sys.stdin.readline().strip()           # the secret is read BEFORE the submission runs
tests = TESTS
spec = importlib.util.spec_from_file_location('submission', PATH)
sub = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(sub)               # run the submission as a module, inside a try
except BaseException as e:                     # BaseException also catches sys.exit() and SystemExit
    print(f'{type(e).__name__}: {e}', file=sys.stderr); sys.exit(1)
for call, want in tests:
    try:
        got = eval(call, dict(vars(sub)))      # call the submitted function
    except BaseException as e:
        print(f'{type(e).__name__} in {call}', file=sys.stderr); sys.exit(1)
    if type(got) is not type(want) or got != want:   # same type, so an object that claims to equal anything fails
        print(f'wrong: {call} == {want!r}', file=sys.stderr); sys.exit(1)
print(nonce)                                   # reached only after every test passed
"""


def code_reward_asserts(code, tests, timeout=2.0):
    """Method 1: a trusted harness imports the submission and checks each test (call, expected value) itself.
    It prints a fresh random nonce only after every test passes, so a submission that prints a success marker
    or exits early gets nothing. (Not a full sandbox: real systems also lock the process down.)"""
    nonce = secrets.token_hex(16)                                    # new for every call; the submission never sees it
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(code)                                                # the submission, in its own file
    harness = HARNESS.replace('TESTS', repr(list(tests))).replace('PATH', repr(f.name))
    try:
        r = run_python(harness, stdin=nonce + '\n', timeout=timeout)
    finally:
        os.unlink(f.name)
    if r is None:
        return 0.0, 'timeout'
    code_, out, err = r
    if code_ == 0 and out.strip().splitlines()[-1:] == [nonce]:
        return 1.0, 'all tests pass'
    last = (err.strip().splitlines() or ['no nonce printed'])[-1]
    last = re.sub(r' \(tmp\w+\.py, line \d+\)', '', last)           # drop the temporary file name
    return 0.0, last[:60]


def code_reward_judge(program, cases, timeout=2.0):
    """Method 2: the way a contest judge works. Feed each input on stdin, compare stdout with the expected output."""
    passed = 0
    for inp, want in cases:
        r = run_python(program, stdin=inp, timeout=timeout)
        if r is not None and r[0] == 0 and r[1].strip() == want.strip():
            passed += 1
    return float(passed == len(cases)), f'{passed}/{len(cases)} cases'
```

Method 1 works like a unit test, but the test code is ours, not the model's. A trusted harness loads the submission as a module, calls the function on each test input, and compares the result with the expected value. It prints a secret, a random string made fresh for every call, only after the last test passes. The grader pays 1 only when that secret comes back.

Method 2 works like a contest site such as Codeforces: it feeds an input on standard input and compares what the program prints. The task is `digit_sum(n)`, the sum of the digits of `n`, with five tests. We wrote six "model answers": one correct, and five that fail in different ways.

Why the secret? Our first version pasted the tests after the model's code and paid 1 when the output contained "ALL PASS". A submission can print those words itself and stop before any test runs. The last line of the run below shows that this two-line program, `print("ALL PASS")` then `raise SystemExit(0)`, would have earned full reward from that grader. The harness catches the early exit, and the submission cannot print a secret it never sees.

The harness is still not a full sandbox: a program that really wanted to could inspect the harness process. Real training systems also run the code in a locked-down container.

```text
== 2. code reward: digit_sum(n), 5 hidden tests, 2 s limit
correct            Method 1 (harness): reward 1  (all tests pass)
drops a digit      Method 1 (harness): reward 0  (wrong: digit_sum(7) == 7)
endless loop       Method 1 (harness): reward 0  (timeout)
syntax error       Method 1 (harness): reward 0  (SyntaxError: expected ':')
learned the tests  Method 1 (harness): reward 0  (wrong: digit_sum(99999) == 45)
fakes the marker   Method 1 (harness): reward 0  (SystemExit: 0)
correct            Method 2 (judge)  : reward 1  (5/5 cases)
drops a digit      Method 2 (judge)  : reward 0  (1/5 cases)
endless loop       Method 2 (judge)  : reward 0  (0/5 cases)
syntax error       Method 2 (judge)  : reward 0  (0/5 cases)
learned the tests  Method 2 (judge)  : reward 0  (3/5 cases)
fakes the marker   Method 2 (judge)  : reward 0  (0/5 cases)
the two code rewards agree on all 6 programs
a grader that trusts a printed "ALL PASS": fakes the marker gets reward 1
```

Look at the "learned the tests" answer. It is a lookup table, `{0: 0, 7: 7, 1234: 10}`, that knows three of the five tests and nothing else. It passes 3 of 5 cases. With partial credit it would score 0.6 for code that cannot add digits. With all-or-nothing scoring and tests the model never sees, it gets 0. This is the first small case of a theme of this part: a reward is only as good as what it actually checks.

v1 of the paper does not say whether code rewards give partial credit, nor how many tests each problem has. The later long version describes how the team built test cases for thousands of contest problems. A model wrote programs that generate inputs, and known correct solutions filtered the bad tests.

### Format rewards

> [!PAPER] · §2.2.2 · page 6
> [![The Format rewards bullet of §2.2.2, with format reward and put its thinking process between highlighted](/img/papers/deepseek-r1/p3-format.png)](/img/papers/deepseek-r1/p3-format.png)
>
> **What it says:** "we employ a format reward model that enforces the model to put its thinking process between '<think>' and '</think>' tags."
>
> **What it means:**
>
> The accuracy reward cares only about the final answer. The format reward cares only about the shape. It asks: did the model write its reasoning inside think tags, and then its answer?
>
> Why reward a shape? Because the shape is what lets people study the model. With the thinking in one fenced block, you can measure its length and read it. You can also see new behaviour appear, such as the "aha moment" of [Part 4](part-4-r1-zero-and-the-aha-moment.md). The tags also split the answer from the working, so a user can be shown only the answer.
>
> Note the word "model" in "format reward model". It is still a rule, a pattern match. The paper uses "reward model" for any scorer, learned or not.

v1 does not say how the two rewards combine. The later long version gives the formula.

> [!PAPER] DeepSeek-AI (2025), long version · §2.2 · page 4
> [![The format rewards paragraph of the long version, with enforcing specific formatting requirements and combined with the same weight highlighted, and Equation 4, Reward rule equals Reward acc plus Reward format](/img/papers/deepseek-r1/p3-long-eq4.png)](/img/papers/deepseek-r1/p3-long-eq4.png)
>
> **What it says:** the format reward works "by enforcing specific formatting requirements", and Equation 4 adds the two rewards: $$Reward_{\text{rule}} = Reward_{\text{acc}} + Reward_{\text{format}}$$. They are "combined with the same weight".
>
> **What it means:**
>
> The sum is the simplest possible way to join two rules. Each output gets 0, 1 or 2 points. A right answer in the right shape gets 2. A right answer without tags gets 1, and so does a wrong answer in perfect tags.
>
> For GRPO only the differences inside a group matter (see [Part 2](part-2-grpo.md)). So in a group where every answer is wrong, the format reward still separates the tidy answers from the messy ones. Early in training, before the model can solve much, that is the only signal it gets.

Written as one equation:

$$
r(o) = r_{\text{acc}}(o) + r_{\text{format}}(o), \qquad r_{\text{format}}(o) = \begin{cases} 1 & \text{if } o \text{ has the shape } \texttt{<think>}\dots\texttt{</think> <answer>}\dots\texttt{</answer>} \\ 0 & \text{otherwise.} \end{cases}
$$

Read it left to right. The symbol $$r(o)$$ is the reward of output $$o$$. The term $$r_{\text{acc}}$$ is the accuracy reward from above. The term $$r_{\text{format}}$$ is 1 when the output has the shape that Table 1 asks for, below, and 0 when it does not.

{{FIG:p3_pipeline|The reward of one output in four steps. The format check and the accuracy check are separate, so a right answer without tags still scores 1, and a tidy wrong answer also scores 1.}}

The paper names only the think tags in §2.2.2, but the template also asks for answer tags. Our format check asks for both, in order, with nothing outside them.

### Method 1 and Method 2: a regex and a parser

The usual way to check a shape is one regular expression:

```python
FORMAT_RE = re.compile(r'^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$', re.DOTALL)


def format_reward_regex(output):
    """Method 1: one regular expression for the whole output."""
    return float(FORMAT_RE.match(output) is not None)
```

The second way reads the tags in order and checks each rule by name. It is longer, but every rule is a line you can read:

```python
TAG = re.compile(r'</?(think|answer)>')


def format_reward_parser(output):
    """Method 2: read the tags in order and check every rule by name."""
    tags = [(m.group(0), m.start(), m.end()) for m in TAG.finditer(output)]
    if [t[0] for t in tags] != ['<think>', '</think>', '<answer>', '</answer>']:
        return 0.0, 'tags: ' + ' '.join(t[0] for t in tags) if tags else 'no tags'
    (_, _, a), (_, b, c), (_, d, e), (_, f, g) = tags
    if output[:tags[0][1]].strip():
        return 0.0, 'text before <think>'
    if output[g:].strip():
        return 0.0, 'text after </answer>'
    if output[c:d].strip():
        return 0.0, 'text between the blocks'
    if not output[a:b].strip():
        return 0.0, 'empty think'
    if not output[e:f].strip():
        return 0.0, 'empty answer'
    return 1.0, 'ok'
```

We ran both on eight outputs for the duck-egg question (the right answer is 18):

```text
== 3. format reward: regex (Method 1) vs parser (Method 2)
good                 regex 1  parser 1  (ok)
no tags              regex 0  parser 0  (no tags)
no answer tags       regex 0  parser 0  (tags: <think> </think>)
text before          regex 0  parser 0  (text before <think>)
text after           regex 0  parser 0  (text after </answer>)
empty think          regex 1  parser 0  (empty think)   DISAGREE
two think blocks     regex 1  parser 0  (tags: <think> </think> <think> </think> <answer> </answer>)   DISAGREE
answer inside think  regex 1  parser 0  (tags: <think> <answer> </answer> </think> <answer> </answer>)   DISAGREE
```

The regex looks strict, but `.*?` can match anything, including nothing and including more tags. So an empty think block passes. So does an output with a second think block, or a fake answer inside the thinking. The parser rejects all three.

The empty think block matters most. A model that writes `<think></think>` and then the answer gets the full format reward for no thinking at all. Under RL, a hole like this is not rare. The model tries millions of outputs, and it finds every hole that pays.

## Why not a neural reward model {§2.2.2}

> [!PAPER] · §2.2.2 · page 6
> [![The last paragraph of §2.2.2, with outcome or process neural reward model, reward hacking, retraining the reward model needs additional training resources and complicates the whole training pipeline highlighted](/img/papers/deepseek-r1/p3-no-neural.png)](/img/papers/deepseek-r1/p3-no-neural.png)
>
> **What it says:** "We do not apply the outcome or process neural reward model" for R1-Zero, because "the neural reward model may suffer from reward hacking" in large-scale RL, and "retraining the reward model needs additional training resources and it complicates the whole training pipeline."
>
> **What it means:**
>
> There are three reasons, and the first is the deep one. A neural reward model is a learned guess. RL is a search that runs for thousands of steps and tries millions of outputs. Give a strong search a guess with flaws, and the search finds the flaws.
>
> The second and third reasons are about cost. When the policy finds a flaw, you must collect new data and retrain the reward model. That needs more machines and a second training loop that must stay in step with the first. Every extra network is one more thing that can break in a long run.
>
> Rules avoid all three. A rule has no weights, so there is nothing to retrain. And a rule that checks a fact, such as "is the boxed answer 18?", has very few flaws for the search to find.

### Background: reward hacking and Goodhart's law

The idea behind the first reason is old. The economist Charles Goodhart noticed it in the 1970s. It is usually said in one line: **when a measure becomes a target, it ceases to be a good measure.** A school that is paid by exam scores starts to teach the exam. A factory paid by the number of nails makes many tiny, useless nails.

> [!DEFINITION] Reward hacking
> When a model gets a high reward by exploiting a flaw in the reward, not by doing the task the reward was meant to measure. One example is an answer that lists every possible number, so the right one is "in" it. Another is a summary that a reward model likes because it is long, not because it is correct.

The R1 paper points to one study for this effect, from 2022.

> [!PAPER] Gao, Schulman and Hilton (2022), Scaling Laws for Reward Model Overoptimization · Abstract and §1 · page 1
> [![The abstract of Gao et al., with reward model trained to predict human preferences, imperfect proxy, optimizing its value too much can hinder ground truth performance and Goodhart's law highlighted](/img/papers/deepseek-r1/p3-gao-abstract.png)](/img/papers/deepseek-r1/p3-gao-abstract.png)
> [![The first lines of the introduction of Gao et al., with When a measure becomes a target, it ceases to be a good measure highlighted](/img/papers/deepseek-r1/p3-gao-goodhart.png)](/img/papers/deepseek-r1/p3-gao-goodhart.png)
>
> **What it says:** in RLHF the model is optimised against "a reward model trained to predict human preferences". Because the reward model "is an imperfect proxy, optimizing its value too much can hinder ground truth performance, in accordance with Goodhart's law", the adage "When a measure becomes a target, it ceases to be a good measure."
>
> **What it means:**
>
> Real human labels are too expensive for a careful test. So the authors play a trick. A large "gold" reward model plays the part of the humans. A smaller "proxy" reward model learns from the gold model's labels, the way a normal reward model learns from people. Then RL trains a policy against the proxy, and they watch both scores.
>
> The proxy score is what the policy sees. The gold score is the truth the proxy was meant to track. If the proxy were perfect, the two would rise together forever.

> [!PAPER] Gao, Schulman and Hilton (2022) · Figure 1 · page 3
> [![Panel b of Figure 1 of Gao et al.: RM score against the KL distance between the RL policy and the initial policy, for reward models from 3M to 3B parameters. Dashed proxy curves keep rising; solid gold curves rise, peak and then fall, and the smaller the reward model, the earlier and lower the peak. The caption phrase the gold reward initially increases and later decreases is highlighted](/img/papers/deepseek-r1/p3-gao-figure1.png)](/img/papers/deepseek-r1/p3-gao-figure1.png)
>
> **What it says:** "when we optimize for a learned proxy of the gold reward, the gold reward initially increases and later decreases."
>
> **What it means:**
>
> Read the chart from left to right. The x-axis is how far RL has moved the policy from where it started. The dashed lines are the proxy score: they keep climbing, so the policy believes it is improving. The solid lines are the gold score: they climb, reach a peak, then fall.
>
> After the peak, every step of RL makes the model worse at the real task while its reward still goes up. That is reward hacking, measured. The colours add a second lesson: a smaller reward model (darker lines, 3M parameters) peaks earlier and lower than a bigger one (3B). A bigger reward model delays the problem. It does not remove it.
>
> Now picture R1-Zero's run: thousands of RL steps on a very large model. That is far to the right on this chart. A learned reward would almost surely pass its peak long before the end.

The R1 paper comes back to this in §4.2, where it lists the attempts that failed. One of them was a process reward model.

> [!PAPER] · §4.2 · page 15
> [![A sentence of §4.2 about process reward models, with it inevitably leads to reward hacking (Gao et al., 2022) highlighted](/img/papers/deepseek-r1/p3-prm-hacking.png)](/img/papers/deepseek-r1/p3-prm-hacking.png)
>
> **What it says:** "once a model-based PRM is introduced, it inevitably leads to reward hacking (Gao et al., 2022), and retraining the reward model needs additional training resources and it complicates the whole training pipeline."
>
> **What it means:**
>
> Here the word is "inevitably", stronger than the "may suffer" of §2.2.2. The authors had tried it. This is the same sentence as in §2.2.2, now as a report of a failure, not only as a reason. [Part 7](part-7-results-failures-and-limits.md) reads the full list of failed attempts.

### Hacking a weak reward on purpose

Rules can be hacked too, if they are written carelessly. Here is a rule that sounds fair: "reward 1 if the right answer appears in the output". And here is the strict rule we built above: right format, then exactly one box inside the answer tags, then the accuracy check.

```python
def naive_reward(output, gold):
    """'The gold answer appears somewhere in the output.' Sounds fair; it is not."""
    return float(re.search(r'(?<![\d.])' + re.escape(gold) + r'(?![\d.])', output) is not None)


def strict_reward(output, gold):
    """The strict format parser first, then exactly one boxed answer inside <answer>, then the accuracy check."""
    if format_reward_parser(output)[0] != 1.0:                       # the parser, not the loose regex
        return 0.0
    ans = answer_block(output)
    if ans is None or ans.count('\\boxed') != 1:
        return 0.0
    return accuracy_reward_ours(ans, gold)
```

The strict rule runs the format parser (Method 2) first, because we saw above that the regex lets broken outputs through. Only then does `answer_block` take the text between the answer tags. We scored three outputs for the duck-egg question. The honest one solves it. The "number spray" writes every whole number from 0 to 1000 inside the answer tags. The third writes 30 boxes, one for each number from 0 to 29.

```text
== 4. reward hacking on the duck-egg question (gold 18)
honest answer          naive reward 1   strict reward 1   (75 characters)
number spray 0..1000   naive reward 1   strict reward 0   (3947 characters)
30 boxes 0..29         naive reward 1   strict reward 0   (331 characters)
empty think            regex format 1   strict reward 0   (the strict reward runs the parser first)
two think blocks       regex format 1   strict reward 0   (the strict reward runs the parser first)
answer inside think    regex format 1   strict reward 0   (the strict reward runs the parser first)
number spray on all 1319 GSM8K test questions: naive reward 1 on 1190 (90.2%), strict reward 1 on 0
```

The number spray needs no maths at all. Still, the naive rule pays it on 90.2% of the GSM8K test set: every question whose answer is a whole number up to 1000. A model trained with that rule would not learn arithmetic. It would learn to print long lists. The strict rule gives the same output 0 on every question.

The three middle rows are the broken shapes from the format section, each with a correct boxed 18. The regex would accept all three. The strict rule rejects them, because it asks the parser first.

This is the real difference between rules and a learned reward. Both can have holes. A rule's holes sit in a few lines of code that you can read, test and fix, as we just did. A network's holes are spread over millions of weights, and you only find them when the policy has already learned to use them.

## Training template {§2.2.3}

> [!PAPER] · Table 1 · page 6
> [![Table 1 of the paper: the template for DeepSeek-R1-Zero, with think tags, answer tags and User: prompt. Assistant: highlighted, and the caption saying that prompt is replaced with the specific reasoning question during training](/img/papers/deepseek-r1/p3-table1.png)](/img/papers/deepseek-r1/p3-table1.png)
>
> **What it says:** "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: prompt. Assistant:" The caption: "prompt will be replaced with the specific reasoning question during training."
>
> **What it means:**
>
> This paragraph is the whole interface between the people and the base model. A base model is not a chat model. It has never been taught to follow instructions. It only continues text. So the template does not give orders. It writes the beginning of a document that a sensible continuation would finish in the right shape.
>
> Read it as a document. It says that what follows is a conversation, that the assistant thinks first, and where the thinking and the answer go. It even shows a tiny example of the tags. Then it ends with "Assistant:", which leaves the model one natural move: write the assistant's reply.
>
> The template asks for no box. The accuracy reward of §2.2.2 reads the answer from a box, but nothing in Table 1 says so. The paper does not explain this gap. Maths questions in the training data may have carried the box instruction themselves; v1 does not say.

> [!PAPER] · §2.2.3 · page 6
> [![All of §2.2.3, with straightforward template, first produce a reasoning process, followed by the final answer, We intentionally limit our constraints to this structural format, avoiding any content-specific biases, mandating reflective reasoning, promoting particular problem-solving strategies and accurately observe the model's natural progression highlighted](/img/papers/deepseek-r1/p3-template.png)](/img/papers/deepseek-r1/p3-template.png)
>
> **What it says:** the "straightforward template" requires the model "to first produce a reasoning process, followed by the final answer." Then: "We intentionally limit our constraints to this structural format, avoiding any content-specific biases", such as "mandating reflective reasoning or promoting particular problem-solving strategies", so that they "can accurately observe the model's natural progression during the RL process."
>
> **What it means:**
>
> This is how a scientist designs a clean experiment. To learn whether a plant grows toward light by itself, you do not tie it to a stick that points at the window. The authors wanted to know what RL alone teaches. So they kept every other hand off the model.
>
> Look at what they left out. The template does not say "check your work", "think step by step", or "try another method if stuck". It gives no worked examples. Any of these would likely help early on. But then, if the model later checked its own work, no one could tell whether RL taught it or the prompt asked for it.
>
> This choice is what makes Part 4 possible. When R1-Zero's answers grow longer and it writes "Wait, wait. Wait." in the middle of a solution (Table 3), the template cannot be the reason. The template only fixed the shape. The content grew from the reward alone.

The paper's structural rule is just as important for the reward. The template asks for exactly the shape that the format reward checks. So the format reward is not a surprise for the model. It is the same request, now paid in points.

### Our run: what a raw base model does with Table 1

Before any RL, how often does a base model already follow this template, and how often is it right? This is the starting point that R1-Zero's RL climbs from. We cannot run DeepSeek-V3-Base, R1-Zero's real base model, on a laptop. So we use two small base models: Qwen2.5-Math-1.5B, the base of the 1.5B distilled model of §2.4, and the general Qwen2.5-0.5B.

We put four questions into the exact Table 1 text: two from GSM8K (school word problems) and two from MATH-500 (competition maths). Each model samples 8 answers per question. The sampling temperature is 1.0, the rollout temperature that the later long version reports. Each answer may have at most 1,024 new tokens, and generation stops at `</answer>`.

```python
TEMPLATE = ('A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
            'The assistant first thinks about the reasoning process in the mind and then provides the user with the '
            'answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, '
            'respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. '
            'User: {prompt}. Assistant:')      # Table 1, word for word; "prompt" is replaced by the question
```

```python
            prompt = TEMPLATE.format(prompt=q.strip().rstrip('.'))
            ids = tok(prompt, return_tensors='pt').to(device)
            torch.manual_seed(0)
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**ids, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=MAX_NEW,
                                     num_return_sequences=N, stop_strings=['</answer>'], tokenizer=tok,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
```

Each answer then goes through the rewards of this page. The format reward is the strict parser (Method 2). The accuracy reward reads the answer tags if they exist, and takes the last box inside them if there is one. Otherwise it falls back to the last box anywhere:

```python
def final_answer(output):
    """What the accuracy reward reads: the <answer> block if there is one (its last box if it has a box),
    otherwise the last \\boxed{} anywhere. None if neither exists."""
    m = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
    if m:
        return extract_boxed(m.group(1)) or m.group(1).strip()
    return extract_boxed(output)
```

The run took a few minutes on a laptop GPU (Apple `mps`). Here is the full log:

[![Terminal output of r1_part3_run.py: counts per question, the eight rewards, their GRPO advantages, and the totals](/img/papers/deepseek-r1/r1_part3_run-run.png)](/img/papers/deepseek-r1/r1_part3_run-run.png)

The two totals lines are the summary:

```text
== totals over 4 questions x 8 samples = 32 answers per model
Qwen/Qwen2.5-Math-1.5B   think  2/32  answer tags  2/32  strict format  1/32  correct  8/32  format and correct  1/32  mean reward 0.281  groups with different rewards 4/4
Qwen/Qwen2.5-0.5B        think  6/32  answer tags  8/32  strict format  1/32  correct  0/32  format and correct  0/32  mean reward 0.031  groups with different rewards 1/4
```

{{FIG:p3_counts|What two raw base models do with the Table 1 template, before any RL: 32 answers each (4 questions, 8 samples). Almost no answer follows the format, and only the maths model ever gets an answer right.}}

Three things stand out. First, the format is almost never right. Each model wrote the strict shape once in 32 tries. Most answers ignore the tags and solve the problem in ordinary prose, the way the maths model was trained to write. The template asks politely, and a base model does not have to obey.

Second, the maths model already knows some maths. It got 8 of 32 answers right, and our checker and math-verify agreed on every one of the 64 answers. Most of those right answers sit in a `\boxed{}` with no tags at all, so they score 1, not 2. The 0.5B model got none right.

The log above prints only counts. The full text of every answer is saved in `results/part3_run.json`, and the answers quoted in the rest of this section are copied from that file.

Third, look at what the reward does with a near miss. One maths-model answer to $$1 - 2 + 3 - \dots - 100$$ ends with "the required sum is equal to $$-\underline{\underline{50}}$$". The number is right, but it sits in no box and no tag, so the reward is 0. Another boxes $$4^{\frac{3}{2}}$$ for "what power of 4 is 8?". It found $$\tfrac{3}{2}$$ in its working and then boxed the wrong thing, so it scores 0 too. The rules pay only for an answer stated where the checker looks.

Here is the one answer from the maths model that earned the full reward of 2:

```text
<think>
  Let's solve the problem step by step.
  We can pair the terms as follows: $(1-2) + (3-4) + (5-6) + \dots + (99-100)$.
  Each pair sums to $-1$.
  There are 50 such pairs because there are 100 terms in total.
  Therefore, the sum of each pair is $-1$, and there are 50 pairs, so the total sum is $-1 \times 50 = -50$.
</think>

<answer>
  The answer is \boxed{-50}.
</answer>
```

And here is the one answer from the 0.5B model in the right format. It scores 1 for the shape and 0 for the content, because the house question asks for the profit, 70000:

```text
<think> First we need to determine the actual value of the house, which is $80,000</think> <answer> So, $80,000 + $50,000 = $130,000</answer>
```

### Why this starting point is enough for RL

GRPO learns from the differences inside a group, not from the rewards alone. Recall Equation 3 from [Part 2](part-2-grpo.md): each reward minus the group mean, divided by the group's standard deviation. Take the maths model on MATH-500 question 16. Its eight rewards were 0, 0, 1, 0, 2, 0, 0, 0. The mean is 0.375, and the advantages are:

```text
MATH-500 16  gold -50          think tags 1/8  answer tags 2/8  strict format 1/8  correct 2/8 (math-verify 2/8)  hit 1024 tokens 1/8  38 s
   rewards (accuracy + format): [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0]
   GRPO advantages (Eq. 3)    : [-0.5, -0.5, 0.84, -0.5, 2.18, -0.5, -0.5, -0.5]
```

The tidy, correct answer gets the largest push, +2.18. The right answer without the format gets a smaller one, +0.84. The six others are pushed down a little. One update of this kind makes the format slightly more likely. Thousands of them turn a rare shape into the habit of every answer.

Now compare the 0.5B model. On three of its four questions, all eight rewards were 0. Then every advantage is 0, and GRPO learns nothing from that question. Only 1 of its 4 groups had any signal, against 4 of 4 for the maths model. This is the practical meaning of "a good base model".

RL cannot reward a behaviour that the model never shows; it can only make a rare good answer common. A base model with no right answers gives RL nothing to climb from.

Our run is tiny: two small models, four questions, eight samples, and answers capped at 1,024 tokens. DeepSeek-V3-Base is far larger, and the paper does not report how often it followed the template at step 0. Our numbers show the shape of the starting point, not R1-Zero's real one. [Part 4](part-4-r1-zero-and-the-aha-moment.md) shows what RL built from there.

> [!TAKEAWAYS] Key takeaways
> - The reward is the only training signal of RL, so it decides what the model learns. R1-Zero uses two rule-based rewards, not a neural network.
> - The accuracy reward checks the final answer. For maths it reads the answer from a box and compares it with the known answer; for code it runs predefined tests.
> - The format reward checks the shape: reasoning inside think tags, then the answer. The later long version adds the two rewards with equal weight, so each output scores 0, 1 or 2.
> - A maths checker is harder than it looks. A one-line regex breaks on nested braces, and our checker and the math-verify library disagreed on 5 of 15 tricky answer pairs.
> - The authors refused outcome and process reward models for three reasons: reward hacking in large-scale RL, the cost of retraining, and a more complex pipeline.
> - Reward hacking is Goodhart's law in RL: optimise a proxy too hard and the true goal falls. Gao et al. (2022) measured the gold reward rising, then falling.
> - Rules can be hacked too: a "the answer appears somewhere" rule pays a list of numbers on 90.2% of GSM8K. But a rule's holes are visible and easy to close.
> - Table 1 fixes only the structure. It does not ask for reflection or any strategy, so whatever reasoning appears later must come from RL itself.
> - Before any RL, our two small base models wrote the strict format in 1 of 32 answers each. The maths model was right in 8 of 32. GRPO had a signal on all 4 of its questions, but on only 1 of 4 for the 0.5B model.

## The paper and the code

Paper: [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) · Official repo: [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) · Our scripts: [code/papers/deepseek-r1](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/deepseek-r1)
