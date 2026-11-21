---
title: "The R1 Pipeline: Four Stages"
description: "How R1-Zero's raw skill became DeepSeek-R1: a cold start, reasoning RL with a language reward, rejection sampling into 800k examples, and a last RL stage. We run each idea in Python."
part: 5
covers: "§2.3, §2.3.1 to §2.3.4"
date: 2026-11-21
tags: [deepseek, reasoning, reinforcement-learning, rejection-sampling, sft, reward-model]
---

R1-Zero could reason, but its thinking was often hard to read, and it switched between English and Chinese in the middle of a thought. Section 2.3 is the recipe that kept the skill and fixed the rest, in four stages and two pages of text.

{{FIG:p5_pipeline|The whole recipe of §2.3. DeepSeek-V3-Base enters twice: at stage 1, and again at stage 3. The dashed arrow carries data, not weights.}}

## The pipeline in one paragraph {§1}

The introduction gives the whole recipe before §2.3 explains it. Read it once now, so the four stages below have a place to go.

> [!PAPER] · §1 · page 3
> [![The introduction: R1 begins with thousands of cold-start data to fine-tune DeepSeek-V3-Base, then reasoning-oriented RL like R1-Zero, then new SFT data through rejection sampling on the RL checkpoint, combined with supervised data from DeepSeek-V3, to retrain DeepSeek-V3-Base, then an additional RL process with prompts from all scenarios](/img/papers/deepseek-r1/p5-intro-pipeline.png)](/img/papers/deepseek-r1/p5-intro-pipeline.png)
>
> **What it says:** "we begin by collecting thousands of cold-start data to fine-tune the DeepSeek-V3-Base model." Then "reasoning-oriented RL like DeepSeek-R1-Zero." Near convergence, "we create new SFT data through rejection sampling on the RL checkpoint", add data from DeepSeek-V3, "and then retrain the DeepSeek-V3-Base model." Last, "an additional RL process, taking into account prompts from all scenarios."
>
> **What it means:**
>
> Think of a music student. First a teacher shows a few clean examples of how a good solo sounds. Then the student practises alone for months, with a tuner that says only right or wrong. Then the student records hundreds of takes, keeps the best ones, and a new student learns from that tape. Last, the new student plays in front of an audience and learns what people like.
>
> Those are the four stages. Two of them are supervised fine-tuning (SFT): the model copies examples. Two of them are reinforcement learning (RL): the model tries, gets a score and changes. The order matters, and one word in the paragraph matters most: "retrain". The third stage does not continue from the RL model. It starts again from the base model.

> [!PAPER] · §1.1 · page 4
> [![The contributions: the pipeline incorporates two RL stages aimed at discovering improved reasoning patterns and aligning with human preferences, as well as two SFT stages that serve as the seed for the model's reasoning and non-reasoning capabilities](/img/papers/deepseek-r1/p5-two-and-two.png)](/img/papers/deepseek-r1/p5-two-and-two.png)
>
> **What it says:** "two RL stages aimed at discovering improved reasoning patterns and aligning with human preferences, as well as two SFT stages that serve as the seed for the model's reasoning and non-reasoning capabilities."
>
> **What it means:**
>
> The sentence gives each stage a job. The SFT stages plant seeds: they put a behaviour into the model by example. The RL stages grow what was planted: one grows reasoning, the other grows good manners with people.
>
> This division is the main design idea of the section. SFT is cheap and fast but can only copy what is in the data. RL can find new behaviour, but it needs a starting point where good answers already happen sometimes. So the recipe switches between the two, and each one prepares the ground for the next.

## Two questions after R1-Zero {§2.3}

[Part 4](part-4-r1-zero-and-the-aha-moment.md) ended with R1-Zero's two problems: poor readability and language mixing. Section 2.3 opens with two questions that come straight from them.

> [!PAPER] · §2.3 · page 9
> [![Section 2.3: two natural questions arise: can reasoning performance be further improved or convergence accelerated by incorporating a small amount of high-quality data as a cold start, and how can we train a user-friendly model that produces clear and coherent Chains of Thought and strong general capabilities; the pipeline consists of four stages](/img/papers/deepseek-r1/p5-two-questions.png)](/img/papers/deepseek-r1/p5-two-questions.png)
>
> **What it says:** Question 1: "Can reasoning performance be further improved or convergence accelerated by incorporating a small amount of high-quality data as a cold start?" Question 2: "How can we train a user-friendly model that not only produces clear and coherent Chains of Thought (CoT) but also demonstrates strong general capabilities?" The answer is a pipeline of "four stages".
>
> **What it means:**
>
> The first question is about speed and score. R1-Zero needed thousands of RL steps to climb from 15.6% to 71.0% on AIME (Part 4). Could a small push at the start make the climb shorter or higher?
>
> The second question is about people. A model that thinks well but writes a mix of two languages, with no structure, is a research result, not a product. People also ask for emails, stories and translations, not only maths. R1-Zero never trained on those.
>
> Note the order of the answer. Stages 1 and 2 answer the first question. Stages 3 and 4 answer the second. Keep this split in mind; it explains why stage 3 changes so much.

The later long version of the paper (arXiv v2, January 2026, the text behind the Nature article) draws the same pipeline as a picture. It also names the checkpoints between the stages: Dev-1, Dev-2 and Dev-3. We use these names in this part.

> [!PAPER] DeepSeek-AI, arXiv v2 (2026, long version) · Figure 2 · page 6
> [![Figure 2 of the long version: R1-Zero samples reasoning data that is filtered and refined by DeepSeek-V3 and humans; SFT of V3-Base on cold-start long CoT gives R1 Dev-1; RL with rule-based reward and language consistency gives Dev-2; Dev-2 samples reasoning data and DeepSeek-V3 gives non-reasoning data; SFT of a fresh V3-Base gives Dev-3; RL with rule-based and preference reward on diverse prompts gives R1](/img/papers/deepseek-r1/p5-long-figure2.png)](/img/papers/deepseek-r1/p5-long-figure2.png)
>
> **What it says:** "The multi-stage pipeline of DeepSeek-R1." The models Dev1, Dev2 and Dev3 "represent intermediate checkpoints within this pipeline."
>
> **What it means:**
>
> Follow the purple boxes; they are models. DeepSeek-V3-Base appears three times. The first copy becomes R1-Zero. The second copy is fine-tuned on cold-start data and becomes Dev-1, and RL turns Dev-1 into Dev-2. The third copy is fine-tuned on data that Dev-2 and DeepSeek-V3 wrote, and becomes Dev-3. The last RL stage turns Dev-3 into R1.
>
> The figure also shows where the cold-start data came from: R1-Zero itself, filtered and then refined by DeepSeek-V3 and people. So R1-Zero was not a dead end. It was the first teacher of R1.

## Stage 1: Cold start {§2.3.1}

{{FIG:p5_pipeline_1|Stage 1 fine-tunes DeepSeek-V3-Base on a few thousand long, readable chains of thought. There is no reward yet; the model copies examples.}}

### Why RL from a raw base model starts badly

The paper says R1 avoids "the early unstable cold start phase of RL training from the base model". To see why that phase is unstable, recall how GRPO learns ([Part 2](part-2-grpo.md)). For each question it samples a group of answers and scores them. Each answer's advantage is its reward minus the group mean, divided by the group's standard deviation.

Now think of a base model at step 0. It was trained to continue web text, not to answer in a format. Most of its answers to a hard question are wrong, or they never close the `</think>` tag. If all eight answers in a group get reward 0, then every answer equals the mean. Every advantage is 0, and the group teaches nothing. The model learns only from the rare groups where at least one answer is right.

So the first part of training is slow and noisy. The few useful signals come from lucky answers, and the model's style drifts in whatever direction those lucky answers point. R1-Zero lived through this phase, and its odd style (mixed languages, no summary) is partly a memory of it.

The name comes from engines. A cold engine is hard to start, so old diesel engines have a glow plug that heats the chamber first. A gas heater has a pilot light: a small flame that is always on, so the main burner lights at once. The cold-start data is R1's pilot light. A few thousand good examples make sure that, from the first RL step, some answers in each group are already readable and right.

> [!DEFINITION] Cold-start data
> A small set of high-quality examples used to fine-tune a model before reinforcement learning starts. For R1 it is "thousands" of long chains of thought in a readable format. The term is borrowed from engines (and from recommender systems, where a new user with no history is a "cold start").

> [!PAPER] · §2.3.1 · page 9
> [![Section 2.3.1: unlike R1-Zero, to prevent the early unstable cold start phase of RL training from the base model, for R1 we construct and collect a small amount of long CoT data to fine-tune the model as the initial RL actor](/img/papers/deepseek-r1/p5-cold-why.png)](/img/papers/deepseek-r1/p5-cold-why.png)
>
> **What it says:** "to prevent the early unstable cold start phase of RL training from the base model, for DeepSeek-R1 we construct and collect a small amount of long CoT data to fine-tune the model as the initial RL actor."
>
> **What it means:**
>
> "Actor" is the RL word for the model that acts: the policy that writes answers. "Initial RL actor" means the model that RL starts from. For R1-Zero, that was the raw base model. For R1, it is the base model after a short SFT on the cold-start data.
>
> Note the word "long". The examples are not short worked solutions from a textbook. They are long chains of thought, with checks and second tries, like the ones R1-Zero learned to write. The goal is to keep R1-Zero's way of thinking and give it a clean shape.

> [!PAPER] · §2.3.1 · page 9
> [![The four ways to collect cold-start data: few-shot prompting with a long CoT as an example, directly prompting models to generate detailed answers with reflection and verification, gathering DeepSeek-R1-Zero outputs in a readable format, and refining the results through post-processing by human annotators; thousands of cold-start data fine-tune DeepSeek-V3-Base](/img/papers/deepseek-r1/p5-cold-ways.png)](/img/papers/deepseek-r1/p5-cold-ways.png)
>
> **What it says:** "we have explored several approaches: using few-shot prompting with a long CoT as an example, directly prompting models to generate detailed answers with reflection and verification, gathering DeepSeek-R1-Zero outputs in a readable format, and refining the results through post-processing by human annotators." Then: "we collect thousands of cold-start data to fine-tune the DeepSeek-V3-Base".
>
> **What it means:**
>
> The four ways are four sources of examples, from cheap to expensive:
>
> 1. **Few-shot prompting.** Show a model one long chain of thought, then ask a new question. The model copies the style.
> 2. **Direct prompting.** Ask a model in words to answer in detail, to reflect and to verify its steps.
> 3. **R1-Zero's own outputs.** Take the answers R1-Zero wrote, and keep the ones that are already readable.
> 4. **Human post-processing.** People clean up the results: fix the language, cut repeats, add a summary.
>
> "Thousands" is a small number for a model of 671 billion parameters. DeepSeek-V3's own SFT set had 1.5 million examples (DeepSeek-V3 report, §5.1). This stage does not teach reasoning. It teaches a shape for reasoning that the model can already do.

The later long version describes the recipe in more detail. The words below are from that version, not from v1.

> [!PAPER] DeepSeek-AI, arXiv v2 (2026, long version) · §B.3.2 · page 21
> [![The cold-start recipe of the long version: thousands of prompts, multiple reasoning trajectories from R1-Zero at temperature 1.0, keep only those with correct final answers and a readable format, sympy for maths, repetition detection and language-mixing filtering, then DeepSeek-V3 refines the reasoning and summaries, with the instruction to translate the thinking process to the same language as the question](/img/papers/deepseek-r1/p5-long-cold-recipe.png)](/img/papers/deepseek-r1/p5-long-cold-recipe.png)
>
> **What it says:** "we generate multiple reasoning trajectories using DeepSeek-R1-Zero with a relatively high temperature of 1.0." They "retain only those with correct final answers and a readable format", use "sympy" to compare maths answers, and apply "repetition detection and language-mixing filtering". DeepSeek-V3 then refines the text, with the instruction "Translate the thinking process to the same language as the question."
>
> **What it means:**
>
> This is a small version of stage 3, run before stage 1. R1-Zero writes many answers; a program keeps the correct ones; filters throw out the messy ones; a strong chat model rewrites what is left. The cold start is mostly R1-Zero's own thinking, cleaned and translated.
>
> The same appendix adds a reason that v1 does not give. The motivation is "primarily product-driven": people find a first-person thinking style ("I") easier to follow than R1-Zero's "we". The authors also warn that this human style "may elicit unwarranted trust" and that it reflects their design, not human-like intelligence.

### The readable format

> [!PAPER] · §2.3.1 · page 10
> [![The readability advantage: R1-Zero responses may mix multiple languages or lack markdown formatting; the cold-start data includes a summary at the end of each response and filters out responses that are not reader-friendly; the output format is |special_token|<reasoning_process>|special_token|<summary>](/img/papers/deepseek-r1/p5-readability.png)](/img/papers/deepseek-r1/p5-readability.png)
>
> **What it says:** R1-Zero's responses "may mix multiple languages or lack markdown formatting". The cold-start data has "a summary at the end of each response" and the authors "filters out responses that are not reader-friendly". The format is "|special_token|<reasoning_process>|special_token|<summary>".
>
> **What it means:**
>
> The format has two parts with a marker before each. The first part is the reasoning: long, with false starts and checks. The second part is the summary: short, clean, written for the person who asked. A reader can skip the reasoning and read only the summary.
>
> The paper hides the name of the special token. The released models show it: it is `<think>` before the reasoning and `</think>` after it. R1-Zero's template (Part 3) had `<think>` and `<answer>` tags written in plain text. R1 keeps the thinking tag and drops the answer tag: the summary is simply everything after `</think>`.

We can see this format in the released model. The chat template of DeepSeek-R1-Distill-Qwen-1.5B is a short Jinja program inside its tokenizer. We rewrote it in Python and compared the two on a two-turn chat.

**Method 1: our own function.**

```python
def r1_prompt(messages, add_generation_prompt=False):
    """Build the prompt text the way the released template does (system, user and assistant turns; no tools)."""
    system = ''.join(m['content'] for m in messages if m['role'] == 'system')
    out = BOS + system                                       # the system text goes first, with no tag
    for m in messages:
        if m['role'] == 'user':
            out += USER + m['content']
        elif m['role'] == 'assistant':
            content = m['content'].split('</think>')[-1]    # an old answer keeps only its summary
            out += ASSISTANT + content + EOS
    if add_generation_prompt:
        out += ASSISTANT + '<think>\n'                       # the model's turn opens its thinking
    return out
```

**Method 2: the tokenizer's template.** We call `tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)` on the same chat. The first answer in the chat has a thinking part and a summary. The output (`results/r1_part5_template_stdout.txt`):

```text
Method 1 (our function):
'<｜begin▁of▁sentence｜><｜User｜>What is 15% of 80?<｜Assistant｜>\n\nThe answer is **12**.<｜end▁of▁sentence｜><｜User｜>And 15% of 60?<｜Assistant｜><think>\n'
Method 2 (tokenizer.apply_chat_template):
'<｜begin▁of▁sentence｜><｜User｜>What is 15% of 80?<｜Assistant｜>\n\nThe answer is **12**.<｜end▁of▁sentence｜><｜User｜>And 15% of 60?<｜Assistant｜><think>\n'
same string: True

the thinking of the first answer is in the prompt: False

  '<think>'                    -> token ids [151648]
  '</think>'                   -> token ids [151649]
  '<｜User｜>'                   -> token ids [151644]
  '<｜Assistant｜>'              -> token ids [151645]
  '<｜end▁of▁sentence｜>'        -> token ids [151643]
```

Three facts come out of this. First, `<think>` and `</think>` are single tokens (151648 and 151649), so they are the paper's "|special_token|". Second, the template itself writes `<think>\n` at the start of the model's turn. The model never has to decide whether to think; it is already inside the thinking part when it writes its first token. Third, an old answer in the chat history keeps only its summary. The template cuts everything up to `</think>`, so the next turn never sees old reasoning.

So the summary is the part meant for people, and also the part that stays in the conversation.

### Better than R1-Zero

> [!PAPER] · §2.3.1 · page 10
> [![The potential advantage: by carefully designing the pattern for cold-start data with human priors, we observe better performance against DeepSeek-R1-Zero; we believe the iterative training is a better way for reasoning models](/img/papers/deepseek-r1/p5-potential.png)](/img/papers/deepseek-r1/p5-potential.png)
>
> **What it says:** "By carefully designing the pattern for cold-start data with human priors, we observe better performance against DeepSeek-R1-Zero. We believe the iterative training is a better way for reasoning models."
>
> **What it means:**
>
> A "human prior" is what people already know about good answers: understand the question first, check the result, end with a clear summary. The cold-start data puts that knowledge into the model before RL starts. RL then improves a model that already writes in a good shape.
>
> "Iterative training" is the switching between SFT and RL. Note what v1 does not give here: a table of scores after each stage. It says "better performance" and moves on. The long version added that table, below.

> [!PAPER] DeepSeek-AI, arXiv v2 (2026, long version) · Table 3 · page 9
> [![Table 3 of the long version: benchmark results of R1-Zero, R1-Dev1, R1-Dev2, R1-Dev3 and R1 on English, code, maths and Chinese benchmarks; for example AIME 2024 is 77.9, 59.0, 74.0, 78.1 and 79.8, IF-Eval is 46.6, 71.7, 72.0, 78.1 and 83.3, and AlpacaEval 2.0 is 24.7, 50.1, 55.8, 62.1 and 87.6](/img/papers/deepseek-r1/p5-long-table3.png)](/img/papers/deepseek-r1/p5-long-table3.png)
>
> **What it says:** "Experimental results at each stage of DeepSeek-R1." AIME 2024 goes 77.9 (R1-Zero), 59.0 (Dev-1), 74.0 (Dev-2), 78.1 (Dev-3), 79.8 (R1). IF-Eval goes 46.6, 71.7, 72.0, 78.1, 83.3.
>
> **What it means:**
>
> Read the AIME row first, and remember the figure above: Dev-1 is not R1-Zero plus a cold start. Both start from DeepSeek-V3-Base, on two branches: thousands of RL steps for R1-Zero, only SFT on a few thousand cold-start examples for Dev-1. So Dev-1's 59.0 sits below R1-Zero's 77.9, but this is not a before-and-after measurement. It does not show that the cold start took away ability the model already had. It shows that a short SFT on a small set reaches less reasoning than long RL. The text of the long version describes Dev-1 the same way: "due to the limited size of the cold-start dataset, Dev1 exhibits a partial degradation in reasoning performance".
>
> Now read IF-Eval, a test of following format instructions. Dev-1 scores 71.7, against 46.6 for R1-Zero. So the cold-start branch starts out readable and obedient, but behind on reasoning. Stage 2 exists to add the reasoning: RL on top of Dev-1 gives Dev-2, which reaches 74.0 on AIME. The final R1 reaches 79.8.
>
> One caution. In v1, R1-Zero's AIME pass@1 is 71.0 (Table 2). In this long-version table it is 77.9. The two versions report different R1-Zero numbers; compare numbers only inside one version.

## Stage 2: Reasoning-oriented RL {§2.3.2}

{{FIG:p5_pipeline_2|Stage 2 runs the R1-Zero recipe on the cold-started model, with one new reward for language consistency.}}

> [!PAPER] · §2.3.2 · page 10
> [![Section 2.3.2: after fine-tuning DeepSeek-V3-Base on the cold start data, we apply the same large-scale reinforcement learning training process as employed in DeepSeek-R1-Zero, focused on coding, mathematics, science, and logic reasoning, which involve well-defined problems with clear solutions](/img/papers/deepseek-r1/p5-rl-reasoning.png)](/img/papers/deepseek-r1/p5-rl-reasoning.png)
>
> **What it says:** "we apply the same large-scale reinforcement learning training process as employed in DeepSeek-R1-Zero." The focus is "coding, mathematics, science, and logic reasoning, which involve well-defined problems with clear solutions."
>
> **What it means:**
>
> Nothing new in the algorithm: GRPO from [Part 2](part-2-grpo.md), with the rule-based rewards from [Part 3](part-3-rewards-and-the-template.md). Only the starting point changed. Instead of the raw base model, RL starts from Dev-1, which already writes readable chains of thought.
>
> "Well-defined problems with clear solutions" is the condition for rule-based rewards. A maths problem has one right number; a program passes its tests or fails. A program can check the answer, so no neural judge is needed and there is nothing for the model to trick. An essay has no such check, so essays wait until stage 4.

### The language consistency reward

> [!PAPER] · §2.3.2 · page 10
> [![The language problem: CoT often exhibits language mixing, particularly when RL prompts involve multiple languages; to mitigate it, a language consistency reward is introduced, calculated as the proportion of target language words in the CoT](/img/papers/deepseek-r1/p5-lang-reward.png)](/img/papers/deepseek-r1/p5-lang-reward.png)
>
> **What it says:** "CoT often exhibits language mixing, particularly when RL prompts involve multiple languages." The fix is "a language consistency reward during RL training, which is calculated as the proportion of target language words in the CoT."
>
> **What it means:**
>
> Why would a model mix languages at all? DeepSeek-V3-Base read a great deal of English and Chinese. The accuracy reward checks only the final answer. It does not care which language the thinking uses. If a Chinese phrase helps the model reach the right number, RL rewards it. Over thousands of steps the thinking drifts toward whatever mix works, and the mix is unreadable for most people.
>
> The fix is to pay for consistency directly. The "target language" is the language of the question. The reward is the share of the thinking's words that are in that language. A thinking part all in English, for an English question, gets 1. Half English and half Chinese gets 0.5.

The v1 text gives this reward in words. The long version writes it as Equation 7:

$$
R_{\text{language}} = \frac{\text{Num}(\text{Words}_{\text{target}})}{\text{Num}(\text{Words})}
$$

Read it left to right. $$R_{\text{language}}$$ is the language reward, a number from 0 to 1. The top, $$\text{Num}(\text{Words}_{\text{target}})$$, counts the words of the chain of thought that are in the target language. The bottom, $$\text{Num}(\text{Words})$$, counts all its words. Only the thinking part is counted, not the summary.

The paper does not say how it counts words. English has spaces between words; Chinese has none. We made one choice and wrote it down: a run of Latin letters is one English word, and each Chinese character is one word. Digits, symbols and punctuation are not words. Here is a tiny example: an English question, "What is 15% of 80?", and a thinking part that drifts into Chinese.

{{FIG:p5_lang_words|Our word count on a thinking part that drifts into Chinese. Six of the eleven words are English, so the language reward is 6 / 11 = 0.545.}}

**Method 1: our own word splitter.** Python's `unicodedata` module knows the official Unicode name of every character. A Latin letter's name starts with "LATIN"; a Chinese character's name starts with "CJK UNIFIED IDEOGRAPH".

```python
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
```

**Method 2: the `regex` library.** It understands Unicode script classes: `\p{Han}` matches a Chinese character and `\p{Latin}` a Latin letter. One pattern finds all the words at once.

```python
WORD = regex.compile(r'\p{Han}|\p{Latin}+|[^\P{L}\p{Latin}\p{Han}]+')   # one Han char, a Latin run, or a run of other letters


def lang_reward_regex(cot, target):
    ws = WORD.findall(cot)
    if not ws:
        return 0.0, 0, 0
    pat = {'en': r'\p{Latin}+', 'zh': r'\p{Han}'}[target]
    n_target = sum(1 for w in ws if regex.fullmatch(pat, w))
    return n_target / len(ws), n_target, len(ws)
```

We ran both on four thinking parts: English for English, English drifting into Chinese, Chinese with two English words, and English for a Chinese question. The output (`results/r1_part5_lang_stdout.txt`):

```text
tiny example: how Method 1 splits case B
  words: First/en of/en is/en 然/zh 后/zh 是/zh 所/zh 以/zh Let/en me/en check/en

A: English question, English thinking
  target=en  method 1: 13/13 = 1.000   method 2: 13/13 = 1.000   agree: True
B: English question, thinking drifts into Chinese
  target=en  method 1: 6/11 = 0.545   method 2: 6/11 = 0.545   agree: True
C: Chinese question, Chinese thinking with two English terms
  target=zh  method 1: 12/14 = 0.857   method 2: 12/14 = 0.857   agree: True
D: Chinese question, thinking all in English
  target=zh  method 1: 0/9 = 0.000   method 2: 0/9 = 0.000   agree: True

both methods agree on all 4 strings: True
```

![The terminal output of r1_part5_lang.py: both methods agree on all four strings, and the summed rewards and advantages of a group of four answers](/img/papers/deepseek-r1/p5-lang-run.png)

Case D is the interesting one. The thinking is perfect English and fully correct, but the question was in Chinese, so the reward is 0. The reward is not "prefer English". It is "answer in the language you were asked in". The paper's later limitations section notes that R1 is tuned for Chinese and English only. A question in a third language can still get English thinking.

### Summing the two rewards

> [!PAPER] · §2.3.2 · page 10
> [![The cost of the language reward: ablation experiments show a slight degradation in the model's performance, but this reward aligns with human preferences and makes the output more readable; the accuracy reward and the language consistency reward are directly summed to form the final reward, and RL runs until convergence on reasoning tasks](/img/papers/deepseek-r1/p5-lang-cost.png)](/img/papers/deepseek-r1/p5-lang-cost.png)
>
> **What it says:** "ablation experiments show that such alignment results in a slight degradation in the model's performance, this reward aligns with human preferences, making it more readable." The two rewards are combined "by directly summing them to form the final reward", and RL runs "until it achieves convergence on reasoning tasks."
>
> **What it means:**
>
> This is an honest trade. Mixing languages helped the model a little, maybe because some ideas are shorter or more common in one language. Forcing one language takes that help away, and scores drop a little. The authors pay this price for readability.
>
> "Directly summing" is the simplest way to join two rewards. An answer's reward becomes accuracy (0 or 1) plus language (0 to 1). No weights, no tuning. The paper does not say whether the format reward of R1-Zero is still added; the phrase "the same large-scale reinforcement learning training process" suggests it is.

Here is the sum as an equation, as v1 describes it:

$$
r_i = r_i^{\text{acc}} + R_{\text{language}}(o_i)
$$

Here $$r_i$$ is the reward of answer $$o_i$$, the $$i$$-th answer in a group. The accuracy part $$r_i^{\text{acc}}$$ is 1 if its final answer is right and 0 if not. The last term is the language reward of its thinking. So the reward now ranges from 0 to 2.

Why does a small extra number change anything? Because GRPO compares answers inside a group. We took four answers to "What is 15% of 80?": right in English, right but mixed, wrong in English, wrong and mixed. Then we computed the GRPO advantage of Part 2 twice: with the summed reward and with accuracy alone.

```text
final reward of §2.3.2 for a group of 4 answers to the English question (accuracy 1 or 0, plus language)
  right, English  accuracy=1  language=1.000  reward=2.000  advantage=+1.097
  right, mixed    accuracy=1  language=0.545  reward=1.545  advantage=+0.449
  wrong, English  accuracy=0  language=1.000  reward=1.000  advantage=-0.328
  wrong, mixed    accuracy=0  language=0.375  reward=0.375  advantage=-1.219
  accuracy alone gives advantages +0.866, +0.866, -0.866, -0.866 : the two right answers look the same
  (advantages from group_advantages() of r1_part2_grpo.py: std over G - 1, eps 1e-4)
```

The advantages come from the same `group_advantages()` function as in [Part 2](part-2-grpo.md), so the standard deviation divides by $$G - 1$$. That is why accuracy alone gives $$\pm 0.866$$ here and not $$\pm 1$$.

With accuracy alone, the clean answer and the mixed answer get the same push, +0.866. With the sum, the clean one gets +1.097 and the mixed one only +0.449. In this group both are still pushed up, because both are above the group's mean reward. That is not guaranteed. In a group where most answers are right and clean, a right but mixed answer can fall below the mean and get a negative advantage. The model now learns a little more from the clean one, step after step.

Note also what the sum does to right and wrong answers. Accuracy adds 1, and the language reward adds at most 1. So a right answer never gets a lower reward than a wrong one, but the two can tie. A right answer with language reward 0 gets a reward of 1. So does a wrong answer with language reward 1.

The long version gives the settings of this stage. They are not in v1.

> [!PAPER] DeepSeek-AI, arXiv v2 (2026, long version) · §3.2.1 · page 7
> [![Training details of the first RL stage in the long version: learning rate 3e-6, KL coefficient 0.001, GRPO clip ratio 10, temperature 1, 16 outputs per question with a maximum length of 32,768, 32 unique questions per step, and the language consistency reward written as Equation 7, the number of target-language words over the number of words](/img/papers/deepseek-r1/p5-long-rl1.png)](/img/papers/deepseek-r1/p5-long-rl1.png)
>
> **What it says:** "we set the learning rate to 3e-6, the KL coefficient to 0.001", the clip ratio $$\varepsilon$$ to 10 and the temperature to 1. "For each question, we sample 16 outputs with a maximum length of 32,768." The language reward is "the proportion of target language words in the CoT", Equation 7.
>
> **What it means:**
>
> These are the numbers v1 leaves out of Part 2's GRPO equation. The group size is 16: each question gets 16 answers, and each answer is compared with the other 15. The KL coefficient $$\beta$$ is 0.001, a very light leash to the reference model. Each answer may run to 32,768 tokens, room for very long thinking.
>
> The clip ratio of 10 is unusual. The PPO paper used 0.2, and most GRPO code keeps that value. A clip of 10 almost turns the clip off. The text after the crop says only that the value "plays a crucial role in training": too low cuts the gradient of many tokens, too high makes training unstable.

## Stage 3: Rejection sampling and SFT {§2.3.3}

{{FIG:p5_pipeline_3|Stage 3 turns the RL checkpoint into a writer of data, then fine-tunes a fresh copy of DeepSeek-V3-Base on 800k samples.}}

### What rejection sampling means

The name comes from statistics. Suppose you want random points from a strange shape, but you only know how to draw points from a simple box around it. Throw points into the box at random. Keep the points that land inside the shape, and reject the rest. What you keep is a fair sample of the shape. John von Neumann described this "acceptance-rejection" method in 1951, and it is still a basic tool of Monte Carlo methods.

Language models use the same idea with answers instead of points. The model is the box: it is easy to sample from. "Good answers" are the shape: we cannot sample them directly, but we can check them. So sample many answers, keep the ones that pass the check, and throw away the rest. Then fine-tune on what you kept, and the model learns from its own best work.

Meta used this for Llama 2 in 2023, and DeepSeek's own maths work used it too.

> [!DEFINITION] Rejection sampling (for language models)
> Generate several answers to each prompt and score each one with a check: a rule, a program or a judge model. Keep only those that pass, and use them as training data for SFT. It turns a model that is right some of the time into a dataset that is right all of the time.

> [!PAPER] · §2.3.3 · page 10
> [![Section 2.3.3: when reasoning-oriented RL converges, the resulting checkpoint is used to collect SFT data for the subsequent round; unlike the cold-start data, which primarily focuses on reasoning, this stage incorporates data from other domains to enhance the model's capabilities in writing, role-playing, and other general-purpose tasks](/img/papers/deepseek-r1/p5-rs-intro.png)](/img/papers/deepseek-r1/p5-rs-intro.png)
>
> **What it says:** "When reasoning-oriented RL converges, we utilize the resulting checkpoint to collect SFT (Supervised Fine-Tuning) data for the subsequent round." This stage adds "data from other domains" for "writing, role-playing, and other general-purpose tasks."
>
> **What it means:**
>
> Stage 2 made a strong reasoner, Dev-2. But it trained only on problems with checkable answers. Ask it for a cover letter and it has nothing new to offer. Stage 3 is where the model becomes general.
>
> Note what Dev-2 is used for: to "collect SFT data". It becomes a tool that writes examples. Its weights are not the starting point of the next step.

> [!PAPER] · §2.3.3 · page 10
> [![Reasoning data: reasoning prompts and trajectories from rejection sampling on the RL checkpoint; the previous stage only included data that could be evaluated using rule-based rewards; this stage adds data that uses a generative reward model, feeding the ground-truth and model predictions into DeepSeek-V3 for judgment](/img/papers/deepseek-r1/p5-rs-reasoning.png)](/img/papers/deepseek-r1/p5-rs-reasoning.png)
>
> **What it says:** "We curate reasoning prompts and generate reasoning trajectories by performing rejection sampling from the checkpoint". Stage 2 used only "rule-based rewards", but this stage adds data that uses "a generative reward model by feeding the ground-truth and model predictions into DeepSeek-V3 for judgment."
>
> **What it means:**
>
> RL could only use problems that a program can grade. Rejection sampling can be looser, because it only has to decide keep or drop, once, offline. So a strong model can be the judge. DeepSeek-V3 reads the question, the reference answer and the model's answer, and says correct or incorrect.
>
> This is safer here than in RL, but not risk-free. In RL, the model trains against the judge for thousands of steps and can learn to fool it. That is reward hacking (Part 3). In rejection sampling, the judge filters data once, offline, so the model cannot keep adapting to the judge's weak spots. But the filter still keeps whatever the judge wrongly marks as correct, and SFT then teaches the model those answers. Picking the best of many answers by a judge's score is a known way to select the judge's errors.

> [!DEFINITION] Generative reward model
> A language model used as a judge. It reads a prompt, a reference and an answer, and writes a verdict in words (here, "correct" or "incorrect"). It differs from a classic reward model, which outputs one number from a special head. The long version prints the judge prompt (Listing 4): it asks DeepSeek-V3 for an analysis, then a JSON verdict.

> [!PAPER] · §2.3.3 · page 10
> [![The filters: because the model output is sometimes chaotic and difficult to read, chain-of-thought with mixed languages, long paragraphs, and code blocks is filtered out; for each prompt, multiple responses are sampled and only the correct ones are retained, for about 600k reasoning related training samples](/img/papers/deepseek-r1/p5-rs-filters.png)](/img/papers/deepseek-r1/p5-rs-filters.png)
>
> **What it says:** "we have filtered out chain-of-thought with mixed languages, long parapraphs, and code blocks. For each prompt, we sample multiple responses and retain only the correct ones. In total, we collect about 600k reasoning related training samples."
>
> **What it means:**
>
> Two sieves, one after the other. The first sieve is correctness: wrong answers go. The second sieve is readability: three kinds of messy thinking go, even when the answer is right. Mixed languages you know. A long paragraph is a wall of text with no breaks. A code block inside the thinking of a maths problem is often the model pasting a program instead of reasoning.
>
> The paper gives no thresholds. How long is "long"? How much mixing is too much? It does not say. It also does not say how many answers "multiple" means, or how many answers it sampled to keep 600k. Our experiment below has to choose these numbers itself.

### Our rejection sampling run

We ran the whole stage-3 recipe at toy size on the released DeepSeek-R1-Distill-Qwen-1.5B. It is the smallest model that R1's own 800k samples trained (Part 6), so its answers have R1's shape. We took 10 problems: 5 from the GSM8K test set and 5 from MATH-500, one of each difficulty level, picked with a fixed seed. For each problem we sampled 8 answers with the model card's advice: temperature 0.6 and top-p 0.95. The prompt ends with the line "Please reason step by step, and put your final answer within \boxed{}." Each answer could use at most 2,048 new tokens. The run used a Mac with an Apple GPU (MPS).

```python
with torch.no_grad():
    out = model.generate(ids, do_sample=True, temperature=TEMP, top_p=TOP_P, max_new_tokens=MAX_NEW,
                         num_return_sequences=N_SAMPLES, pad_token_id=tok.eos_token_id)
```

First, keep only the correct answers. **Method 1** is our own check: find the last `\boxed{...}`, match its braces by hand, clean both answers, and compare.

```python
def last_boxed(text):
    """The content of the last \\boxed{...}, with nested braces matched by hand."""
    k = text.rfind('\\boxed{')
    if k < 0:
        return None
    i, depth, start = k + len('\\boxed{'), 1, k + len('\\boxed{')
    while i < len(text) and depth:
        depth += {'{': 1, '}': -1}.get(text[i], 0)
        i += 1
    return text[start:i - 1] if depth == 0 else None


def correct_scratch(text, gold):
    pred = last_boxed(text)
    if pred is None:
        return None, False
    a, b = norm(pred), norm(gold)
    if a == b:
        return pred, True
    try:
        return pred, abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return pred, False
```

**Method 2** is the `math-verify` library, which parses both answers into maths and compares them as expressions. The funnel uses Method 2, and the script prints every answer where the two disagree.

```python
def correct_mv(text, gold):
    from math_verify import parse, verify
    try:
        return bool(verify(parse('$' + gold + '$') or parse(gold), parse(text)))
    except Exception:
        return False
```

Then the three filters, on the thinking part only. The mixed-language filter reuses the language reward from stage 2. The paper gives no thresholds, so these are ours. A language reward below 0.99 is "mixed", and a paragraph of more than 120 words is "long".

```python
def paragraph_words(thinking):
    return [len(p.split()) for p in re.split(r'\n\s*\n', thinking) if p.strip()]


def filters(thinking):
    lc, _, _ = lang_reward_scratch(thinking, 'en')
    longest = max(paragraph_words(thinking) or [0])
    return dict(lang=round(lc, 4), longest_para=longest, code='```' in thinking)
```

The run took 11.6 minutes for 80 answers. Here is the score of each group (`results/r1_part5_reject_stdout.txt`):

```text
per problem: correct answers out of 8 (math-verify), mean length in tokens
  gsm8k-1275             correct 8/8   finished 8/8   mean tokens 979
  gsm8k-523              correct 7/8   finished 8/8   mean tokens 1258
  gsm8k-734              correct 4/8   finished 4/8   mean tokens 1695
  gsm8k-1085             correct 8/8   finished 8/8   mean tokens 1429
  gsm8k-59               correct 7/8   finished 7/8   mean tokens 1099
  math500-L1-362         correct 7/8   finished 7/8   mean tokens 1313
  math500-L2-120         correct 0/8   finished 0/8   mean tokens 2048
  math500-L3-396         correct 0/8   finished 0/8   mean tokens 2048
  math500-L4-71          correct 0/8   finished 0/8   mean tokens 2048
  math500-L5-166         correct 0/8   finished 0/8   mean tokens 2048
groups where all 8 rewards are equal (GRPO advantage 0 for every answer): 6 of 10
```

Our limit of 2,048 tokens is the main filter here. On the four harder MATH-500 problems, not one of the 32 answers closed its thinking in time. That is not a fault of the model: the paper lets R1 think for up to 32,768 tokens. It is the price of a toy run on a laptop.

The last line connects back to the cold start. In 6 of our 10 groups, all eight answers got the same reward: two groups all right, four all wrong. GRPO learns nothing from such a group. A raw base model at step 0 is in the "all wrong" case for almost every hard question, which is why a cold start helps.

Now the funnel. Each line is the number of answers left after one more sieve.

```text
the funnel
   80  sampled   (10 problems)
   42  finished (</think> and an end token)   (6 problems)
   41  correct (math-verify)   (6 problems)
   41  no mixed language (language reward >= 0.99)   (6 problems)
   40  no long paragraph (<= 120 words)   (6 problems)
   39  no code block   (6 problems)

among the 41 correct answers: language reward min 1.000; longest paragraph max 122 words; code blocks 1
paragraphs in their thinking: 923, median 26 words, over 120 words: 1
```

{{FIG:p5_funnel|Our rejection-sampling funnel on DeepSeek-R1-Distill-Qwen-1.5B: 80 answers sampled, 42 finished their thinking, 41 were correct, and 39 passed the three readability filters.}}

The two answer checks did not fully agree: they agree on 35 of the 42 finished answers. All seven differences are on one problem:

```text
  methods disagree on math500-L1-362 #0: ours=False math-verify=True pred='\\dfrac{1}{4}' gold='\\frac14'
```

The model wrote `\dfrac{1}{4}` and the reference says `\frac14`. Both mean one quarter. Our string cleaner turns `\dfrac` into `\frac`, but it does not know that `\frac14` is short for `\frac{1}{4}`. The library math-verify parses both into the number 0.25 and sees they are equal. This is why real pipelines use a parser; the long version names sympy. It is also why the paper adds DeepSeek-V3 as a judge for answers that rules cannot compare.

The readability filters removed only two answers. Every correct answer had a language reward of 1.000: this distilled model was trained on R1's cleaned data, so it already thinks in one language. One answer had a paragraph of 122 words, and one had a code block. The paper's filters bite much harder on a raw RL checkpoint like Dev-2 than on a model that already learned from filtered data.

The kept answers become a tiny SFT file, one JSON record per line, in the chat format that the template of stage 1 reads. Here is the shortest record, with the middle of the thinking cut for space:

```text
{
 "id": "gsm8k-59-7",
 "messages": [
  {
   "role": "user",
   "content": "A raspberry bush has 6 clusters of 20 fruit each and 67 individual fruit scattered across the bush. How many raspberries are there total?\nPlease reason step by step, and put your final answer within \\boxed{}."
  },
  {
   "role": "assistant",
   "content": "<think>\nOkay, so I have this problem about a raspberry bush, and I need to figure out how many raspberries there are in total. Let me read it again: \"A raspberry bush has 6 clusters of 20 fruit each and 67 individual fruit scattered across the bush. How many r [...]  \\times 20 \\text{ raspberries per cluster} = 120 \\text{ raspberries} \\]\n\nAdditionally, there are 67 individual fruits scattered across the bush, each of which is a raspberry. Therefore, the number of raspberries from the scattered fruits is:\n\n\\[ 67 \\text{ individual fruits} \\]\n\nAdding these together gives the total number of raspberries:\n\n\\[ 120 + 67 = 187 \\]\n\nThus, the total number of raspberries is \\(\\boxed{187}\\)."
  }
 ]
}
```

Look at its shape. The thinking opens with `<think>`, reads the problem again, and works in short paragraphs. After `</think>` comes a clean summary with the boxed answer. That is the format of stage 1, written by the model itself. DeepSeek ran this same loop at a scale of about 600k kept answers; we kept 39.

![The terminal output of r1_part5_reject.py: the per-problem scores, the disagreements of the two answer checks, and the funnel](/img/papers/deepseek-r1/p5-reject-run.png)

## The data mix: 600k + 200k = 800k {§2.3.3}

> [!PAPER] · §2.3.3 · page 11
> [![Non-reasoning data: for writing, factual QA, self-cognition, and translation, the DeepSeek-V3 pipeline is adopted and portions of the SFT dataset of DeepSeek-V3 are reused; for some tasks DeepSeek-V3 generates a potential chain-of-thought before answering; for simpler queries such as hello there is no CoT; about 200k samples unrelated to reasoning](/img/papers/deepseek-r1/p5-nonreasoning.png)](/img/papers/deepseek-r1/p5-nonreasoning.png)
>
> **What it says:** For "writing, factual QA, self-cognition, and translation", they "reuse portions of the SFT dataset of DeepSeek-V3." For some tasks, DeepSeek-V3 is asked to "generate a potential chain-of-thought before answering". But "for simpler queries, such as 'hello' we do not provide a CoT in response." In total, "approximately 200k" samples.
>
> **What it means:**
>
> R1 must still write, translate and chat, so it gets examples of those from the data that made DeepSeek-V3 good at them. "Self-cognition" is a small set about the model itself: what is your name, who made you.
>
> The "hello" rule is a lesson in taste. A model that thinks for 500 tokens before saying hello is slow and silly. So some examples teach a short or empty thinking part. The model learns that thinking length should depend on the question. Easy questions get an answer at once; hard ones get long reasoning.

> [!PAPER] · §2.3.3 · page 11
> [![We fine-tune DeepSeek-V3-Base for two epochs using the above curated dataset of about 800k samples](/img/papers/deepseek-r1/p5-800k.png)](/img/papers/deepseek-r1/p5-800k.png)
>
> **What it says:** "We fine-tune DeepSeek-V3-Base for two epochs using the above curated dataset of about 800k samples."
>
> **What it means:**
>
> 600k reasoning samples plus 200k other samples make about 800k. Two epochs means the model reads every sample twice. Note the model name once more: DeepSeek-V3-Base, not Dev-2. The section below explains why.
>
> These 800k samples have a second life. [Part 6](part-6-distillation.md) shows how they fine-tune small Qwen and Llama models into the distilled R1 models. One of them is the 1.5B model we use on this site.

The long version gives the split of the 800k by domain. Maths and code are three quarters of it.

> [!PAPER] DeepSeek-AI, arXiv v2 (2026, long version) · Table 5 · page 27
> [![Table 5 of the long version, data statistics of SFT data: Math 395,285 samples with 6,094.2 average tokens, Code 211,129 with 7,435.7, STEM 10,124, Logic 10,395, General 177,812 with 1,419.8, total 804,745 samples with 5,355.3 average tokens](/img/papers/deepseek-r1/p5-long-table5.png)](/img/papers/deepseek-r1/p5-long-table5.png)
>
> **What it says:** Math 395,285 samples, Code 211,129, STEM 10,124, Logic 10,395, General 177,812. Total 804,745 samples, with 5,355.3 tokens on average.
>
> **What it means:**
>
> The "about 800k" of v1 is 804,745 exactly. A general sample is short: 1,419.8 tokens on average, against more than 6,000 for maths and code. Long reasoning is most of what the model reads in this stage.
>
> The same appendix says that almost every sample is a single turn (about 1.0 rounds). The authors note that this may limit R1 in long conversations, a weakness the v1 conclusion also lists.

### Why stage 3 starts again from V3-Base

{{FIG:p5_restart|The weights of the RL checkpoint Dev-2 stop at stage 2. Dev-2 only writes data. A fresh copy of DeepSeek-V3-Base learns from that data, together with 200k samples from DeepSeek-V3. Checkpoint names are from the long version.}}

This is the most surprising choice in the section. Dev-2 is the best reasoner so far. Why not fine-tune Dev-2 on the 800k samples and continue? The paper does not say. Here is our reading, built from facts in the paper.

**The data carries the skill.** The 600k reasoning samples are Dev-2's best answers, checked and cleaned. A model that learns to write them has learned what Dev-2 can do, without Dev-2's weights. Part 6 proves that this works: the same data turns a 1.5B Qwen model into a strong reasoner with SFT alone. If the data can carry the skill into a different model, it can carry it into a fresh copy of the same base.

**The weights carry side effects.** Dev-2 spent thousands of RL steps on maths, code, science and logic prompts only. RL also pushes a model's style in odd directions, and stage 2 paid "a slight degradation" for the language reward. Fine-tuning Dev-2 would keep all of that and then add writing on top. Starting from the base gives one clean training run in which reasoning and general skills learn side by side, from data chosen on purpose.

**A clean mix is easier to control.** With one SFT run on 800k known samples, the authors decide the exact mix. They choose how much maths, how much writing, and which answers have no thinking at all. Each fine-tuning stage on top of another can erase earlier skills. One run from the base avoids that chain of stages.

A teacher analogy helps. Dev-2 is a brilliant but odd student who has only done exam problems for a year. Instead of sending that student to writing classes, you ask the student to write a clean answer book. Then a new student learns from the answer book and from a writing course at the same time. The second student ends up with the skill and none of the odd habits. Table 3 above supports this: Dev-3 scores 78.1 on AIME, above Dev-2's 74.0, and 62.1 on AlpacaEval 2.0, above Dev-2's 55.8.

## Stage 4: RL for all scenarios {§2.3.4}

{{FIG:p5_pipeline_4|Stage 4 runs RL on a mix of prompts: rules grade the reasoning prompts, and reward models grade the general ones.}}

> [!PAPER] · §2.3.4 · page 11
> [![Section 2.3.4: a secondary reinforcement learning stage to improve helpfulness and harmlessness while refining reasoning; rule-based rewards for math, code and logic; reward models to capture human preferences for general data, building on the DeepSeek-V3 pipeline with a similar distribution of preference pairs and training prompts](/img/papers/deepseek-r1/p5-rl-all.png)](/img/papers/deepseek-r1/p5-rl-all.png)
>
> **What it says:** "a secondary reinforcement learning stage aimed at improving the model's helpfulness and harmlessness while simultaneously refining its reasoning capabilities." Reasoning data uses "rule-based rewards". General data uses "reward models to capture human preferences", with "a similar distribution of preference pairs and training prompts" as DeepSeek-V3.
>
> **What it means:**
>
> Stage 3 taught the model to write like a helpful assistant by copying examples. Stage 4 lets it practise with feedback. Each training batch mixes two kinds of prompts, and each kind gets its own judge.
>
> A maths prompt still gets a rule: is the boxed answer right. A prompt like "write a polite reply to my landlord" has no rule. For it, a reward model gives a score that predicts which answer people would prefer. This is the first time in the whole paper that a neural reward model drives RL. Section 2.2.2 avoided it for R1-Zero because of reward hacking, and that risk is why it comes only now, at the very end.

### What a preference reward model is

A preference reward model is a language model with one extra output: a single number, the score of an answer. It is trained on pairs. People (or a strong model) see two answers to one prompt and pick the better one.

The classic training rule is the Bradley-Terry model from 1952. The chance that answer A beats answer B is $$\sigma(r_A - r_B)$$. Here $$\sigma$$ is the sigmoid, and $$r_A, r_B$$ are the two scores. The [Background page](background-the-road-to-r1.md) shows where this came from in RLHF.

> [!DEFINITION] Preference reward model
> A model that reads a prompt and an answer and outputs one number. It is trained so that preferred answers get higher numbers than rejected ones. The Bradley-Terry rule is $$P(A \succ B) = \sigma(r_A - r_B)$$. A score gap of 0 means a coin flip; a large gap means near certainty.

> [!PAPER] · §2.3.4 · page 11
> [![Helpfulness focuses exclusively on the final summary, minimizing interference with the underlying reasoning process; harmlessness evaluates the entire response of the model, including both the reasoning process and the summary](/img/papers/deepseek-r1/p5-helpful-harmless.png)](/img/papers/deepseek-r1/p5-helpful-harmless.png)
>
> **What it says:** "For helpfulness, we focus exclusively on the final summary", "minimizing interference with the underlying reasoning process." "For harmlessness, we evaluate the entire response of the model, including both the reasoning process and the summary".
>
> **What it means:**
>
> Two judges look at two different parts of the answer, and the choice of part is the clever bit. The helpfulness judge reads only the summary. If it also read the thinking, it might reward thinking that looks polished, and RL would push the reasoning toward style over substance.
>
> Keeping the thinking out of view means the judge never scores the reasoning style directly. It does not cut every link. The reward still flows back to all the tokens of the answer, thinking included. And the summary depends on the thinking before it. The paper's own word is "minimizing" the interference, not removing it.
>
> The harmlessness judge reads everything. Harmful content in the thinking is still harmful, because a user can see the thinking. A dangerous plan does not become safe because the summary refuses. So safety checks the whole text.
>
> This matches the chat template we read in stage 1. The summary is the part for people, the part kept in the chat history, and the part the helpfulness judge scores. The thinking is the model's workspace, checked only for safety.

The long version tells how these reward models were built. The details below are from that version.

> [!PAPER] DeepSeek-AI, arXiv v2 (2026, long version) · §3.1 · page 7
> [![The helpful and safety reward models of the long version: preference pairs judged by querying DeepSeek-V3 four times with random A and B positions to mitigate positional bias, 66,000 data pairs, a reward head on the DeepSeek-R1 architecture; the safety reward model uses 106,000 prompts labelled safe or unsafe and a point-wise method](/img/papers/deepseek-r1/p5-long-rm.png)](/img/papers/deepseek-r1/p5-long-rm.png)
>
> **What it says:** For each pair, "we query DeepSeek-V3 four times, randomly assigning the responses as either Response A or Response B to mitigate positional bias." They "curated 66,000 data pairs". The model has "a reward head". The safety model uses "106,000" prompts labelled safe or unsafe and a "point-wise" method.
>
> **What it means:**
>
> The helpfulness labels came from DeepSeek-V3 as a judge, not from people. A judge model tends to prefer whichever answer it sees first. Asking four times with the order shuffled, and averaging, cancels that. They also kept only pairs with a clear winner. Chosen and rejected answers have about the same length, so the reward model cannot learn "longer is better".
>
> The two models learn in two ways. Helpfulness is pairwise, the Bradley-Terry way: which of two answers is better. Safety is point-wise: is this one answer safe or not. A safety question has an absolute answer; helpfulness is easier to judge by comparison.

> [!PAPER] DeepSeek-AI, arXiv v2 (2026, long version) · §3.2.2 · page 8
> [![The second RL stage of the long version: the reward is the sum of a reasoning reward, a general reward and a language reward; the reasoning reward is the rule reward; the general reward is the reward model score plus the format reward; a reduced temperature of 0.7; 1,700 training steps, with general instruction data and preference rewards only in the final 400 steps, because more steps with the model-based reward may lead to reward hacking](/img/papers/deepseek-r1/p5-long-rl2.png)](/img/papers/deepseek-r1/p5-long-rl2.png)
>
> **What it says:** $$\text{Reward} = \text{Reward}_{\text{reasoning}} + \text{Reward}_{\text{general}} + \text{Reward}_{\text{language}}$$. The stage uses "a reduced temperature of 0.7" and has "1,700 training steps", with preference rewards used "exclusively in the final 400 steps", because more steps "may lead to reward hacking".
>
> **What it means:**
>
> The reward is again a plain sum, now of three parts. A reasoning prompt gets the rule reward. A general prompt gets the reward model's score plus a format reward. Every answer also gets the language reward of stage 2.
>
> The schedule shows how much the authors fear reward hacking. The reward models act for only the last 400 of 1,700 steps. The long version's appendix shows why: with longer training on the helpfulness reward model, the reward kept rising while the real Codeforces score fell. The model was learning to please the judge, not to help. A short, late dose of the reward model gives the polish without the damage.

What did the last stage buy? The long version's Table 3 (above) shows it. From Dev-3 to R1, AIME moves only from 78.1 to 79.8, but AlpacaEval 2.0 goes from 62.1 to 87.6 and ArenaHard from 75.6 to 92.3. Reasoning was mostly done by stage 3. Stage 4 made R1 pleasant to talk to.

> [!TAKEAWAYS]
> - R1 is built in four stages: cold-start SFT, reasoning RL, rejection sampling with SFT, and RL for all scenarios. Two SFT stages plant behaviour; two RL stages grow it.
> - The cold start is "thousands" of long, readable chains of thought, mostly R1-Zero's own answers, filtered and cleaned. It avoids RL's slow, noisy start from a raw base model.
> - The readable format is `|special_token|<reasoning_process>|special_token|<summary>`. In the released models the special tokens are `<think>` and `</think>`, and the chat template opens every answer with `<think>`.
> - Stage 2 adds a language consistency reward: the share of thinking words in the question's language. It is summed with the accuracy reward and costs a little accuracy for readability.
> - Stage 3 samples many answers from the RL checkpoint and keeps the correct ones, graded by rules or by DeepSeek-V3 as a judge. Filters remove mixed languages, long paragraphs and code blocks: about 600k samples.
> - About 200k non-reasoning samples come from the DeepSeek-V3 pipeline. Simple queries like "hello" get no chain of thought. DeepSeek-V3-Base is fine-tuned on the 800k for two epochs.
> - Stage 3 restarts from DeepSeek-V3-Base, not from the RL checkpoint. The data carries the skill; the same 800k later trains the distilled models.
> - Stage 4 uses rule rewards for reasoning and reward models for general prompts. Helpfulness is judged on the summary only, harmlessness on the whole response.
> - Our toy run of stage 3 on the 1.5B distilled model kept 39 of 80 answers. Our 2,048-token limit removed most; the three filters removed only two, because the model already learned from filtered data.

## The paper and the code

[Paper (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948) · [Official repository](https://github.com/deepseek-ai/DeepSeek-R1) · [Our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/deepseek-r1)
