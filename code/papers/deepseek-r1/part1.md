---
title: "The Big Idea: Reasoning from a Reward"
description: "A model that learned to think step by step from one signal: was the final answer right. We read the title, the abstract, Figure 1 and §1, and run the released 1.5B model next to its base model."
part: 1
covers: "Title, Abstract, Figure 1, §1, §1.1, §1.2"
date: 2026-11-21
tags: [deepseek, reasoning, reinforcement-learning, chain-of-thought, distillation, llm]
---

In January 2025, a lab in Hangzhou released a model that solved competition maths as well as the best closed model of the time. It also published how, and the method was the surprise. Its first model, DeepSeek-R1-Zero, got no supervised fine-tuning on worked reasoning before reinforcement learning. Two rules scored it: was the final answer right, and was the reasoning written inside the right tags. From that alone it found long, careful reasoning by itself. The released model, DeepSeek-R1, then adds thousands of worked examples (the paper's "cold-start data") and more training stages on top.

Each piece below shows the exact lines of the paper, then explains them.

## The title {§Title}

> [!PAPER] · Title · page 1
> [![The title of the paper: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](/img/papers/deepseek-r1/p1-title.png)](/img/papers/deepseek-r1/p1-title.png)
>
> **What it says:** "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning."
>
> **What it means:**
>
> The key word is "incentivizing". There are two ways to teach a student long division. You can show the steps on the board and ask the student to copy them. Or you can give a worksheet, mark only the final answers, and give a point for each right one. The second student must find a method that works, because nothing else earns points.
>
> Before this paper, the usual way to teach a language model to reason was the first way. You show it many worked solutions and train it to copy them. The title says that the authors did the second. They did not teach the steps. They rewarded the result, and the steps came by themselves.
>
> "Reasoning capability" means the skill to solve a problem in several dependent steps, where step 3 needs the result of step 2. Maths, code and logic puzzles are the main tests. "LLMs" are large language models, and "R1" is the name of the model: the first reasoning model of the company DeepSeek.

> [!DEFINITION] Reinforcement learning (RL)
> Learning from a score instead of from examples. The model tries something, and a reward rule gives a number: here, 1 for a right final answer and 0 for a wrong one. Training then makes the tries with high rewards more likely. Nobody shows the model a correct attempt. A chess program that learns only from wins and losses learns by RL.

## The abstract {§Abstract}

> [!PAPER] · Abstract · page 1
> [![The first two sentences of the abstract, with first-generation reasoning models and large-scale reinforcement learning without supervised fine-tuning as a preliminary step highlighted](/img/papers/deepseek-r1/p1-abstract-1.png)](/img/papers/deepseek-r1/p1-abstract-1.png)
>
> **What it says:** "DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrates remarkable reasoning capabilities."
>
> **What it means:**
>
> The paper has two models, and this sentence names the stranger one. **DeepSeek-R1-Zero** starts from a **base model**: a model that has only read a very large amount of text and learned to predict the next token. Such a model can continue a text, but nobody has trained it to answer questions or to follow instructions.
>
> The normal next step is **supervised fine-tuning** (SFT): train the model on many examples of good answers, written by people or by a stronger model. Then RL comes after, to polish. R1-Zero skips the SFT step completely. RL starts on the raw base model. The "Zero" in the name means zero examples of how to answer.
>
> Why is this bold? A base model is a bad student for RL. RL can only reward what the model already does now and then. If the model never writes a sensible solution, there is nothing to reward. The paper bets that a strong base model sometimes writes a right solution, and that RL can grow that rare success into a habit.

> [!DEFINITION] Base model
> A language model after pre-training only. It has read trillions of tokens (web pages, books, code and maths among them) and predicts the next token. It has not been fine-tuned on question-and-answer examples. Give it the start of a text and it writes a plausible continuation. Here the base model is DeepSeek-V3-Base. Our run below uses a small one, Qwen2.5-Math-1.5B.

> [!DEFINITION] Supervised fine-tuning (SFT)
> More training of a pre-trained model on chosen examples of input and desired output. The loss is the same next-token loss as in pre-training, but only on the desired outputs. The model learns to copy the style and the steps of the examples. SFT is learning from examples; RL is learning from a score.

> [!PAPER] · Abstract · page 1
> [![The middle of the abstract, with naturally emerges and poor readability, and language mixing highlighted](/img/papers/deepseek-r1/p1-abstract-2.png)](/img/papers/deepseek-r1/p1-abstract-2.png)
>
> **What it says:** "Through RL, DeepSeek-R1-Zero naturally emerges with numerous powerful and intriguing reasoning behaviors. However, it encounters challenges such as poor readability, and language mixing."
>
> **What it means:**
>
> "Naturally emerges" is the claim that made the paper famous. During RL, R1-Zero began to write longer solutions. It checked its own steps, and it wrote things like "wait, let me check this again" and then tried another approach. Nobody wrote a rule for these habits. They appeared because they led to more right answers, and right answers were rewarded. [Part 4](part-4-r1-zero-and-the-aha-moment.md) shows the curves and the famous "aha moment".
>
> The second sentence is the price. The rewards checked only the final answer and the tags around the reasoning. They did not care if the reasoning was pleasant to read. So R1-Zero's reasoning was often hard to follow, and it could switch between English and Chinese in the middle of a thought. For a score, this does not matter. For a person who wants to read the reasoning, it does.

> [!PAPER] · Abstract · page 1
> [![The next sentences of the abstract, with multi-stage training and cold-start data before RL and comparable to OpenAI-o1-1217 highlighted](/img/papers/deepseek-r1/p1-abstract-3.png)](/img/papers/deepseek-r1/p1-abstract-3.png)
>
> **What it says:** "we introduce DeepSeek-R1, which incorporates multi-stage training and cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1-1217 on reasoning tasks."
>
> **What it means:**
>
> **DeepSeek-R1** is the fix, and the model that people use. "Cold-start data" is a small set of good, readable worked solutions. The model first learns from them with SFT, and only then starts RL. Think of a new driver who gets a few lessons before practice on the open road. The lessons do not teach everything, but they set good habits early.
>
> "Multi-stage" means that SFT and RL alternate: four stages in all. [Part 5](part-5-the-r1-pipeline.md) explains each one. The result matched **OpenAI-o1-1217**, the version of OpenAI's o1 reasoning model from 17 December 2024. At that time, o1 was the strongest reasoning model, and OpenAI had not said how it was trained.

> [!PAPER] · Abstract · page 1
> [![The last sentence of the abstract, with six dense models, the sizes 1.5B to 70B distilled from DeepSeek-R1, and Qwen and Llama highlighted](/img/papers/deepseek-r1/p1-abstract-4.png)](/img/papers/deepseek-r1/p1-abstract-4.png)
>
> **What it says:** "we open-source DeepSeek-R1-Zero, DeepSeek-R1, and six dense models (1.5B, 7B, 8B, 14B, 32B, 70B) distilled from DeepSeek-R1 based on Qwen and Llama."
>
> **What it means:**
>
> The paper releases the weights of both big models and of six small ones. The small ones are not new designs. They are existing open models, Qwen2.5 from Alibaba and Llama from Meta, fine-tuned on the answers that DeepSeek-R1 wrote. This copying is **distillation**: a big teacher writes worked solutions, and a small student learns them with SFT.
>
> "Dense" is the opposite of DeepSeek-V3's design. V3 and R1 are mixture-of-experts models: 671 billion parameters in all, but only 37 billion of them work on each token. In a dense model, every parameter works on every token. The smallest student has 1.5 billion parameters and runs on a laptop. We run it below.

> [!DEFINITION] Distillation
> Training a small "student" model on outputs written by a big "teacher" model. Here the teacher is DeepSeek-R1 and the training is plain SFT on its answers, reasoning included. The student learns the teacher's habits without the teacher's size.

These three sentences describe the whole paper. The picture below puts them on one map. Each later part of this series zooms into one box.

{{FIG:p1_map|The paper in one picture. Road 1: reinforcement learning straight on the base model gives DeepSeek-R1-Zero. Road 2: four stages, two of SFT and two of RL, give DeepSeek-R1. Road 3: about 800,000 samples from R1 fine-tune six small models. The part of this series that explains each step is under it.}}

## Figure 1 {§Figure 1}

> [!PAPER] · Figure 1 · page 1
> [![Figure 1 of the paper: grouped bars for AIME 2024, Codeforces, GPQA Diamond, MATH-500, MMLU and SWE-bench Verified, for DeepSeek-R1, OpenAI-o1-1217, DeepSeek-R1-32B, OpenAI-o1-mini and DeepSeek-V3](/img/papers/deepseek-r1/p1-figure1.png)](/img/papers/deepseek-r1/p1-figure1.png)
>
> **What it says:** "Benchmark performance of DeepSeek-R1." Six groups of bars, one per benchmark, and five models in each group.
>
> **What it means:**
>
> Read the figure group by group. The striped bar is DeepSeek-R1, and the dark grey bar next to it is OpenAI-o1-1217. The two bars are close in every group: R1 is ahead by 0.6 points on AIME and by 0.9 on MATH-500. The o1 model is ahead by 4.2 points on GPQA Diamond and by 1.0 on MMLU. That closeness is the headline.
>
> Now look at the last bar of each group, DeepSeek-V3. V3 and R1 share the same base model. They differ only in post-training.
>
> On AIME, V3 gets 39.2 and R1 gets 79.8. On Codeforces, 58.7 against 96.3. But on MMLU, a test of knowledge, the gap is only 88.5 against 90.8. Reasoning training changed the maths and the code a lot, and the facts only a little.
>
> The third bar is the surprise of the figure. DeepSeek-R1-32B is a distilled student, about twenty times smaller than R1 in total parameters. It beats o1-mini on four of the six benchmarks.

The six benchmarks test different skills. **AIME 2024** is the American Invitational Mathematics Examination: 30 hard competition problems, each with a whole-number answer from 0 to 999. **MATH-500** is 500 problems from a large set of school competition maths. **GPQA Diamond** has 198 multiple-choice science questions written by experts so that a web search does not help. **MMLU** is multiple-choice general knowledge in 57 subjects.

**Codeforces** is a site for programming contests. **SWE-bench Verified** asks the model to fix real bugs in real Python projects.

> [!DEFINITION] Pass@1 and percentile
> **Pass@1** is the chance that one answer is right. The paper samples $$k$$ answers to each question, with $$k$$ from 4 to 64, and checks each one. Then it averages: $$\text{pass@1} = \frac{1}{k}\sum_{i=1}^{k} p_i$$, with $$p_i = 1$$ for a right answer and 0 for a wrong one. The score is the mean of this over all questions. **Percentile** on Codeforces is the share of human contest players with a lower rating than the model. [Part 7](part-7-results-failures-and-limits.md) explains every score in full.

Here is the same figure, redrawn from the paper's numbers, so you can hover each bar:

{{FIG:p1_fig1|Figure 1 of the paper, redrawn from its numbers. R1 and o1-1217 are within about 4 points of each other on every benchmark. DeepSeek-V3, from the same base model, is far behind on AIME 2024 (39.2) and Codeforces (58.7), but close on MMLU (88.5).}}

The weak group is SWE-bench Verified. R1 fixes 49.2% of the bugs, and V3 already fixed 42.0%. Section 5 of the paper gives the reason. RL on software tasks is slow, because each test of a fix takes a long time. So R1 got little RL on this kind of task.

## Introduction {§1}

> [!PAPER] · §1 · page 3
> [![The second paragraph of the Introduction, with post-training, enhance accuracy on reasoning tasks, align with social values, adapt, and relatively minimal computational resources against pre-training highlighted](/img/papers/deepseek-r1/p1-intro-posttraining.png)](/img/papers/deepseek-r1/p1-intro-posttraining.png)
>
> **What it says:** "post-training has emerged as an important component of the full training pipeline. It has been shown to enhance accuracy on reasoning tasks, align with social values, and adapt to user preferences, all while requiring relatively minimal computational resources against pre-training."
>
> **What it means:**
>
> A large language model is trained in two phases. In **pre-training**, it reads trillions of tokens and learns to predict the next one. This phase costs almost all of the compute. In **post-training**, the model learns how to behave. It learns to answer a question instead of continuing it, to refuse harmful requests, and to reason with care.
>
> Think of a medical student. Pre-training is the years in the library: a large amount of knowledge, but no practice with patients. Post-training is the residency: much shorter, but it turns the knowledge into a skill. The paper's point is that the residency is cheap compared with the library, and that it can change a lot.
>
> This framing matters for the whole paper. R1 does not change the model's architecture or its pre-training. Everything new happens in post-training.

> [!DEFINITION] Post-training
> All the training that comes after pre-training: SFT on examples, RL from a reward, or both. It uses far less compute than pre-training. It turns a base model, which only continues text, into an assistant that answers, follows instructions and, here, reasons.

> [!PAPER] · §1 · page 3
> [![The Introduction lines on OpenAI's o1, with the first to introduce inference-time scaling by increasing the length of the Chain-of-Thought reasoning process and effective test-time scaling remains an open question highlighted](/img/papers/deepseek-r1/p1-intro-o1.png)](/img/papers/deepseek-r1/p1-intro-o1.png)
>
> **What it says:** OpenAI's o1 models "were the first to introduce inference-time scaling by increasing the length of the Chain-of-Thought reasoning process". But "the challenge of effective test-time scaling remains an open question for the research community."
>
> **What it means:**
>
> A **chain of thought** is the working that a model writes before its answer, like the working that a student writes on scrap paper. In 2022, researchers found that a model gets more maths problems right when it writes its steps first. Each written step is text that the next step can read. The model uses its own output as working memory.
>
> o1 took this further. It writes a very long chain of thought, often thousands of tokens, before it answers. More tokens of thinking give better answers. This is **inference-time scaling**, or test-time scaling. You get a better answer from more compute when you ask the question, not only when you train the model. A chess player with more time on the clock plays better moves, with the same brain.
>
> The second sentence is the gap that this paper fills. OpenAI showed that long thinking works, but did not say how to train a model to think long and think well. "Open question" is a polite way to say: nobody outside OpenAI knew.

> [!DEFINITION] Chain-of-thought (CoT)
> The intermediate steps that a model writes before its final answer. Instead of "18", the model writes "16 - 3 - 4 = 9 eggs to sell, 9 × 2 = 18 dollars, so the answer is 18". A long CoT, as in o1 and R1, can include checks, mistakes, corrections and second attempts.

> [!DEFINITION] Test-time scaling
> Better answers from more computation at the moment of use ("test time" or "inference time"), not from a bigger model or more training. The two main ways are a longer chain of thought for one answer, and many answers to the same question with a vote.

> [!PAPER] · §1 · page 3
> [![The Introduction lines on earlier work, with process-based reward models, reinforcement learning, Monte Carlo Tree Search and Beam Search, and none of these methods has achieved general reasoning performance comparable highlighted](/img/papers/deepseek-r1/p1-intro-prior.png)](/img/papers/deepseek-r1/p1-intro-prior.png)
>
> **What it says:** prior work explored "process-based reward models", "reinforcement learning", and "search algorithms such as Monte Carlo Tree Search and Beam Search". "However, none of these methods has achieved general reasoning performance comparable to OpenAI's o1 series models."
>
> **What it means:**
>
> In 2024, many groups tried to rebuild o1. The paper names three families of attempts. A **process reward model** (PRM) is a second model that scores each step of a solution, not only the final answer. It is like a teacher who marks every line of the working.
>
> **Search** methods write many partial solutions, score them, and keep growing the best ones, as a chess program explores a tree of moves. Monte Carlo Tree Search is the method of AlphaGo; beam search keeps the best few partial answers at each step. Other groups tried RL.
>
> The phrase "general reasoning" is the key. Some of these methods worked on one narrow test. None reached o1 across maths, code and science together. The paper itself tried PRM and tree search and reports both as failures in §4.2. [Part 7](part-7-results-failures-and-limits.md) reads why: it is hard to say what one "step" is, and a learned scorer can be fooled.

> [!PAPER] · §1 · page 3
> [![The Introduction paragraph on pure RL, with pure reinforcement learning, without any supervised data, self-evolution through, DeepSeek-V3-Base as the base model and GRPO highlighted](/img/papers/deepseek-r1/p1-intro-purerl.png)](/img/papers/deepseek-r1/p1-intro-purerl.png)
>
> **What it says:** "we take the first step toward improving language model reasoning capabilities using pure reinforcement learning (RL). Our goal is to explore the potential of LLMs to develop reasoning capabilities without any supervised data, focusing on their self-evolution through a pure RL process. Specifically, we use DeepSeek-V3-Base as the base model and employ GRPO (Shao et al., 2024) as the RL framework".
>
> **What it means:**
>
> "Pure" means no SFT and no process reward model. There is only RL with two rule-based rewards: one for a right final answer, one for the right format. Compared with the methods of the last paragraph, this is the simplest design. There is no second model that marks the steps and no search tree. There are only the base model, a set of questions with known answers, and a rule that checks each answer.
>
> "Self-evolution" says what the authors want to watch. Without examples, any reasoning style that appears was found by the model itself. So R1-Zero is also an experiment: what does a model invent when nothing tells it how to think?
>
> The two named parts are the base and the algorithm. **DeepSeek-V3-Base** is DeepSeek's large pre-trained model; [DeepSeek, from the Ground Up, Part 2](/writings/deepseek-2-the-v3-architecture/) explains its architecture. **GRPO** (Group Relative Policy Optimization) is the RL algorithm, from DeepSeek's earlier DeepSeekMath paper. For each question it samples a group of answers and compares each answer with the average of its group. [Part 2](part-2-grpo.md) writes it from scratch.

> [!DEFINITION] GRPO, in one paragraph
> For each question, the model writes a group of several answers. Each answer gets a reward. An answer that scores above the group's average is made more likely, and an answer below the average is made less likely. The group average replaces the separate "critic" model that older RL methods train, a second model about as large as the first. Part 2 gives the equations.

> [!PAPER] · §1 · page 3
> [![The Introduction lines on R1-Zero's results, with thousands of RL steps, pass@1 score on AIME 2024 increases from 15.6% to 71.0%, with majority voting, 86.7% and OpenAI-o1-0912 highlighted](/img/papers/deepseek-r1/p1-intro-aime.png)](/img/papers/deepseek-r1/p1-intro-aime.png)
>
> **What it says:** "After thousands of RL steps, DeepSeek-R1-Zero exhibits super performance on reasoning benchmarks. For instance, the pass@1 score on AIME 2024 increases from 15.6% to 71.0%, and with majority voting, the score further improves to 86.7%, matching the performance of OpenAI-o1-0912."
>
> **What it means:**
>
> These three numbers are the evidence for the title. At the start of RL, the base model solved 15.6% of AIME problems in one try. That is about 5 problems out of 30. After RL, it solved 71.0%, about 21 of 30. Nothing else changed: the same model size, the same pre-training, no supervised fine-tuning on worked reasoning. Only the rewards.
>
> The third number uses a different way to answer. The model writes 64 answers to each problem, and the most common final answer counts. With this vote, R1-Zero gets 86.7%. Table 2 of the paper puts o1-0912, the September 2024 version of o1, at 74.4% with one answer and 83.3% with the vote. So R1-Zero is close with one answer and ahead with the vote.
>
> Why does a vote help? Wrong answers tend to be different from each other, and right answers tend to agree. If a model is right 60% of the time, and its wrong answers scatter over many values, the right value usually wins the vote. We count a real vote below.

> [!DEFINITION] Majority voting (cons@64)
> Sample many answers to the same question, read the final answer of each, and return the most common one. The paper uses 64 samples and calls the score "cons@64" (consensus of 64). It is a form of test-time scaling: 64 times the compute, one better answer. Wang et al. (2022) named it self-consistency.

{{FIG:p1_aime|The AIME 2024 numbers of the introduction next to the models it compares with. Reinforcement learning alone takes R1-Zero from 15.6% to 71.0%, and a vote of 64 answers takes it to 86.7%, above the 83.3% of o1-0912 with the same vote.}}

> [!PAPER] · §1 · page 3
> [![The Introduction lines on R1-Zero's problems, with poor readability, and language mixing, small amount of cold-start data and multi-stage training pipeline highlighted](/img/papers/deepseek-r1/p1-intro-r1-1.png)](/img/papers/deepseek-r1/p1-intro-r1-1.png)
>
> **What it says:** "However, DeepSeek-R1-Zero encounters challenges such as poor readability, and language mixing. To address these issues and further enhance reasoning performance, we introduce DeepSeek-R1, which incorporates a small amount of cold-start data and a multi-stage training pipeline."
>
> **What it means:**
>
> Note the two goals in the second sentence. One is to fix the readability. The other is to "further enhance reasoning performance". So the cold start is not only cosmetic. A model that starts RL with good habits learns faster and ends higher. DeepSeek-R1 gets 79.8% pass@1 on AIME, against R1-Zero's 71.0%.
>
> Why would a few thousand examples help so much? At the start of pure RL, the base model writes mostly bad attempts, so the reward is rare and learning is slow and unstable. A few thousand clear worked solutions give the model a good starting format. RL then improves a model that already writes sensible solutions.

> [!PAPER] · §1 · page 3
> [![The Introduction paragraph on the R1 pipeline, with thousands of cold-start data, reasoning-oriented RL, rejection sampling on the RL checkpoint, retrain the DeepSeek-V3-Base model, additional RL process, prompts from all scenarios and on par with OpenAI-o1-1217 highlighted](/img/papers/deepseek-r1/p1-intro-r1-2.png)](/img/papers/deepseek-r1/p1-intro-r1-2.png)
>
> **What it says:** "we begin by collecting thousands of cold-start data to fine-tune the DeepSeek-V3-Base model. Following this, we perform reasoning-oriented RL like DeepSeek-R1-Zero. Upon nearing convergence in the RL process, we create new SFT data through rejection sampling on the RL checkpoint, combined with supervised data from DeepSeek-V3 [...], and then retrain the DeepSeek-V3-Base model. After fine-tuning with the new data, the checkpoint undergoes an additional RL process, taking into account prompts from all scenarios."
>
> **What it means:**
>
> This paragraph is the whole R1 recipe in four steps. It is road 2 on the map above.
>
> 1. **Cold start.** SFT of V3-Base on thousands of long, readable worked solutions.
> 2. **Reasoning RL.** The same RL as R1-Zero, on maths, code, science and logic.
> 3. **Rejection sampling and SFT.** The RL model writes many answers to each question; only the right, readable ones are kept. With writing and general data from DeepSeek-V3, this makes about 800,000 examples (§2.3.3). A fresh copy of V3-Base is fine-tuned on them.
> 4. **RL for all scenarios.** A last RL stage on all kinds of prompts, for helpfulness and safety as well as reasoning.
>
> Step 3 has a detail that is easy to miss: "retrain the DeepSeek-V3-Base model". They do not continue from the RL model. They go back to the base and fine-tune it on the best outputs of the RL model. The RL model is used as a writer of training data, a teacher, not as the next starting point. [Part 5](part-5-the-r1-pipeline.md) explains why.

> [!DEFINITION] Rejection sampling
> Generate many answers, test each one, throw away the ones that fail, and keep the rest as training data. Here "fail" means a wrong final answer, mixed languages, or a messy format. It turns a model that is right only some of the time into a clean data set of right answers.

> [!PAPER] · §1 · page 3
> [![The Introduction paragraph on distillation, with distillation, direct distillation from DeepSeek-R1 outperforms applying RL on it and reasoning patterns discovered by larger base models highlighted](/img/papers/deepseek-r1/p1-intro-distill.png)](/img/papers/deepseek-r1/p1-intro-distill.png)
>
> **What it says:** "Using Qwen2.5-32B (Qwen, 2024b) as the base model, direct distillation from DeepSeek-R1 outperforms applying RL on it. This demonstrates that the reasoning patterns discovered by larger base models are crucial for improving reasoning capabilities."
>
> **What it means:**
>
> This is a fair test with a clear result. Take one 32-billion-parameter base model and train it two ways. Way 1: give it the same large-scale RL as R1-Zero. Way 2: fine-tune it on R1's answers. Way 2 wins clearly. Table 6 of the paper has 72.6% against 47.0% on AIME.
>
> The reading: a small model has trouble finding good reasoning by itself, by trial and reward. A big model can find it. Once the big model has found it, a small model can learn it cheaply by copying. A student who reads a master's worked solutions learns faster than a student who must find every method alone. [Part 6](part-6-distillation.md) reads this experiment in detail.

> [!PAPER] · §1 · page 3
> [![The last lines of the Introduction, with distilled 14B model outperforms, QwQ-32B-Preview and new record on the reasoning benchmarks among dense models highlighted](/img/papers/deepseek-r1/p1-intro-opensource.png)](/img/papers/deepseek-r1/p1-intro-opensource.png)
>
> **What it says:** "our distilled 14B model outperforms state-of-the-art open-source QwQ-32B-Preview (Qwen, 2024a) by a large margin, and the distilled 32B and 70B models set a new record on the reasoning benchmarks among dense models."
>
> **What it means:**
>
> QwQ-32B-Preview was the best open reasoning model before R1, from the Qwen team. A distilled model less than half its size beats it. The point is not the 14B model itself. The point is that the reasoning of R1 moves into other models with plain SFT, the cheapest kind of training. Anyone with R1's outputs and a GPU can make a reasoning model.

### Running a distilled model

The smallest student, DeepSeek-R1-Distill-Qwen-1.5B, is small enough for a laptop. Its base model is Qwen2.5-Math-1.5B (§2.4). So we can do something that is hard with the 671B models. We ask the same question to the base model and to the reasoning model, on one machine, and look at every token. The script is `r1_part1.py`; everything runs on an Apple laptop GPU (`mps`) in float32.

```text
== 1. The two models (float32 on mps)
  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  params 1,777,088,000  layers 28  d_model 1536  heads 12  vocab 151,936
  Qwen/Qwen2.5-Math-1.5B                     params 1,543,714,304  layers 28  d_model 1536  heads 12  vocab 151,936
  same shapes: True; mean |difference| of one weight matrix (layer 0 up_proj): 0.00406
```

The two models have the same 28 layers, the same width and the same vocabulary. Every weight matrix has the same shape. The parameter counts differ only because the base model shares one matrix between its input and output layers, and the distilled release stores two copies. The weights themselves moved only a little: in one matrix, the mean change is 0.004. All the difference in behaviour below comes from these small changes, made by SFT on R1's answers.

#### One decoding step, by hand

A language model writes one token at a time. At each step it gives a score, a **logit**, to every token of its vocabulary: 151,936 scores here. A decoding rule turns the scores into one chosen token. The paper's evaluation (§3, page 12) samples with "a temperature of 0.6 and a top-$$p$$ value of 0.95". We use the same rule everywhere on this page.

**Temperature** divides every logit by a number $$T$$ before the softmax:

$$
p_i = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}
$$

Read it from left to right. $$z_i$$ is the logit of token $$i$$, and $$p_i$$ is its probability. The sum in the denominator runs over all tokens, so the probabilities add up to 1. With $$T = 1$$ this is the plain softmax. With $$T = 0.6$$, the gaps between the logits grow by a factor of $$1 / 0.6 \approx 1.67$$. So the top token gets more likely, and the rare ones get rarer.

**Top-$$p$$** then sorts the tokens by probability and keeps the smallest group at the top whose probabilities add up to at least $$p$$. It throws away the rest and rescales the kept ones so that they add up to 1 again. A long tail of thousands of unlikely tokens can still hold a few percent in total. Top-$$p$$ cuts that tail, so the model does not pick a strange token by bad luck.

Here is Method 1, our own Python. Each function does one step:

```python
def softmax(z):
    """z: (V,) logits -> (V,) probabilities that add up to 1."""
    e = torch.exp(z - z.max())           # subtract the max so exp never overflows
    return e / e.sum()


def temperature_probs(logits, temperature):
    """Divide the logits by the temperature, then softmax. T < 1 makes the top token more likely."""
    return softmax(logits / temperature)                                  # (V,)


def top_p_filter(p, top_p):
    """Keep the smallest set of top tokens whose probabilities add up to at least top_p; renormalise. p: (V,)."""
    order = torch.argsort(p, descending=True)                             # (V,) token ids, most likely first
    sorted_p = p[order]
    before = torch.cumsum(sorted_p, 0) - sorted_p                         # mass of the tokens ranked above each one
    keep_sorted = before < top_p                                          # keep a token while the set is still short of top_p
    keep = torch.zeros_like(p, dtype=torch.bool)
    keep[order[keep_sorted]] = True
    q = torch.where(keep, p, torch.zeros_like(p))
    return q / q.sum(), int(keep.sum())
```

The line with `before` is the one to read twice. For each token in the sorted list, it is the total probability of the tokens ranked above it. A token stays while that total is still below 0.95. So the token that crosses 0.95 stays too, and every token after it goes.

Method 2 is the library. The `transformers` package has the same two steps as `TemperatureLogitsWarper` and `TopPLogitsWarper`, the classes that `model.generate` uses. We run both on four toy logits, and then on the real logits of the distilled model for its first token:

```text
== 2. Temperature and top-p on four toy logits
  token   logit   p (T=1)   p (T=0.6)   after top-p 0.95   library
  "18"     2.00     0.609       0.783              0.787     0.787
  "9"      1.00     0.224       0.148              0.149     0.149
  "16"     0.50     0.136       0.064              0.065     0.065
  "2"     -1.00     0.030       0.005              0.000     0.000
  top-p keeps 3 of 4 tokens; max |ours - library| = 6.0e-08

  real logits: the first token the distilled model writes after the GSM8K prompt (vocabulary 151,936)
    'Okay'         p (T=0.6) = 0.7923
    'Alright'      p (T=0.6) = 0.2069
    'First'        p (T=0.6) = 0.0006
    'Hmm'          p (T=0.6) = 0.0001
    'To'           p (T=0.6) = 0.0000
  top-p 0.95 keeps 2 tokens of 151,936; max |ours - library| = 0.0e+00
```

Check the toy row by hand.

With $$T = 1$$, the weight of "18" is $$e^{2} = 7.39$$, and the four weights add up to $$7.39 + 2.72 + 1.65 + 0.37 = 12.13$$. So $$p = 7.39 / 12.13 = 0.609$$, as printed. With $$T = 0.6$$ the same token gets 0.783. Then top-$$p$$ adds up 0.783, 0.148 and 0.064. The sum 0.995 is past 0.95, so the fourth token goes, and the three that stay are scaled up a little. Our numbers and the library's agree to $$10^{-7}$$.

The real step is more extreme. Of 151,936 tokens, only two survive top-$$p$$: "Okay" and "Alright". The distilled model almost always opens its thinking with one of these two words. It learned this habit from R1's answers.

{{FIG:p1_sampling|One decoding step on four toy tokens, with the numbers printed by our script. The softmax turns logits into probabilities; temperature 0.6 sharpens them; top-p 0.95 cuts the last token and rescales the other three.}}

#### Greedy decoding: our loop against the library

Greedy decoding has no randomness: it always takes the token with the highest logit. So it is a clean test of our own generation loop. If our loop and `model.generate` both decode greedily, they must write exactly the same tokens.

```python
@torch.no_grad()
def our_generate(model, ids, max_new, eos_id, greedy=False, seed=0):
    """Method 1: our own token-by-token loop. The first call reads the whole prompt and fills the KV cache;
    every later call feeds one token and reuses the cache. ids: (1, n) prompt token ids."""
    gen = torch.Generator().manual_seed(seed)
    out = model(input_ids=ids, use_cache=True)                            # prefill: logits (1, n, V)
    past, logits = out.past_key_values, out.logits[0, -1]
    new = []
    for _ in range(max_new):
        tok = pick(logits, gen, greedy)
        new.append(tok)
        if tok == eos_id:
            break
        out = model(input_ids=torch.tensor([[tok]], device=ids.device), past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1]               # (V,) scores for the next token
    return new
```

The first call reads the whole prompt at once and keeps the keys and values of every layer. This is the KV cache, which [LLM Inference, Part 2](/writings/llm-inference-2-kv-cache/) explains. Each later call feeds only the newest token. `pick` is either `argmax` or our temperature and top-$$p$$ functions followed by one random draw.

```text
== 3. Greedy decoding, 128 new tokens: our loop (KV cache) against model.generate
  our loop 128 tokens, generate 128 tokens, equal at 128 of 128 positions: True
  first 12 tokens: ['Okay', ',', 'Ġso', 'ĠI', 'Ġneed', 'Ġto', 'Ġfigure', 'Ġout', 'Ġhow', 'Ġmuch', 'Ġmoney', 'ĠJanet']
```

All 128 tokens are the same. (The mark "Ġ" means "a space comes before this token".) With sampling, the two methods write different tokens, even with the same seed. They draw their random numbers in a different order. They draw from the same distribution, which the check in step 2 shows.

#### The distilled model thinks

Now the real question, the first problem of the GSM8K test set, a set of school word problems. We add the instruction that the official repository recommends for maths, "Please reason step by step, and put your final answer within \boxed{}."

The model's chat format wraps the question in special tokens and ends the prompt with `<think>`, so the model starts inside its thinking. Our own sampling loop writes at most 2,048 new tokens. Here is the start of what it wrote (the full text is in `results/r1_part1_stdout.txt`):

```text
== 4. GSM8K test question 0, distilled model, our own sampling loop (T 0.6, top-p 0.95, seed 0, max 2,048 new tokens)
  | Okay, so Janet has ducks that lay 16 eggs every day. She eats three eggs for breakfast each morning and bakes muffins for her friends with four eggs each day. Then, she sells the remaining eggs at the farmers' market for $2 each. I need to figure out how much money she makes every day from selling these eggs.
  ...
  | So, she eats 3 eggs every morning. There are 24 hours in a day, but since she eats breakfast every morning, which is 12 times a day, right? Because a day has 24 hours, so 24 divided by 2 is 12. So, she eats 3 eggs 12 times a day. Let me calculate how many eggs she eats for breakfast in a day.
  | That would be 3 eggs/morning * 12 mornings/day = 36 eggs.
  | Wait, hold on. That can't be right because she only has 16 eggs a day. If she eats 36 eggs for breakfast, that's way more than she has. That must be a mistake in my reasoning.
  | Let me rethink that. She eats 3 eggs for breakfast every morning. Since a day has 24 hours, but she only eats breakfast once a day. So, she eats 3 eggs once a day. So, that's 3 eggs.
  ...
  ...
  | Then, she sells these 9 eggs at the farmers' market for $2 each. So, she makes 9 * $2 = $18 per day.
  | 
  | Wait, that seems like a lot. Let me confirm each step again.
```

This is a small model, and it makes a silly mistake: twelve breakfasts a day. But look at what it does next. "Wait, hold on. That can't be right." It checks the result against the problem, 36 eggs eaten from 16 laid, sees the contradiction, and starts again. Nobody wrote this check into the model. It came from R1's answers, and R1 got it from RL. This is the self-verification of §1.1, in a 1.5B model on a laptop.

Then the model keeps checking. It finds 18 dollars early, and then it checks again, and again. The end of its thinking and the start of its answer:

```text
  | **Final Answer**
  | \boxed{18}
  | </think>
  | 
  | Janet’s ducks lay 16 eggs per day. She eats 3 eggs for breakfast every morning and 4 eggs for muffins every day. The remaining eggs are sold at the farmers' market for $2 each. 
  ...
  new tokens 2048 in 134.3 s (15.2 tokens/s); thinking tokens 1907; answer tokens 140; closed </think>: True
  boxed answer None, gold '18'; correct (our checker) False; correct (math-verify) False
```

The thinking took 1,907 tokens for a problem that needs two subtractions and one product.

It ended with `\boxed{18}` inside the thinking. Then the model began a clean answer for the reader. Our limit of 2,048 tokens cut it off after 140 tokens, before its box. So the code marks it wrong. The reasoning was right; the budget was too small. The paper allows 32,768 new tokens for every benchmark (§3).

{{FIG:p1_anatomy|The answer of our own sampling loop, split at the closing think tag: 1,907 thinking tokens, then 140 tokens of the answer before the 2,048-token limit cut it off.}}

> [!DEFINITION] The think tags
> R1-Zero was trained to put its reasoning between `<think>` and `</think>`, and its answer after it. R1 and the distilled models keep this format. We split the output at `</think>` to count the thinking tokens. The text after the tag is a short, clean answer for the reader. [Part 3](part-3-rewards-and-the-template.md) shows the template that started the habit.

How does the code check an answer? It takes the content of the last `\boxed{...}` in the answer and compares it with the right answer. Method 1 is our own checker: remove spaces, dollar signs and units, then compare the strings. Method 2 is `math-verify`, a library that parses both answers as maths, so "0.5" and "\frac{1}{2}" count as equal. A rule that a program can check, with no human and no judge model, is exactly the kind of reward that R1-Zero trained on.

```python
def check(pred, gold):
    """Return (ours, math_verify): is the boxed answer equal to the gold answer?"""
    ours = pred is not None and norm(pred) == norm(gold)
    mv = pred is not None and bool(verify(parse('$' + gold + '$'), parse('$\\boxed{' + pred + '}$')))
    return ours, mv
```

Method 2 for generation is `model.generate` with the same settings: `do_sample=True, temperature=0.6, top_p=0.95`. Its random draws differ from ours, so it writes a different sample:

```text
== 5. The same question with model.generate (do_sample, T 0.6, top-p 0.95, seed 0, max 2,048 new tokens)
  new tokens 872; thinking tokens 732; answer tokens 139; closed </think>: True
  boxed answer '18'; correct (our checker) True; correct (math-verify) True
```

This sample thought for 732 tokens, closed the tag, and gave a boxed 18 in 139 more tokens. Both checkers agree that it is right. Two samples of the same model with the same rule took 872 and more than 2,048 tokens. Thinking length varies a lot from sample to sample.

#### The base model, same words

Now the same words go to the base model, Qwen2.5-Math-1.5B. A base model has no chat format, so it gets the question and the instruction as plain text. We use our own sampling loop with the same settings and at most 1,024 new tokens.

```text
== 6. The same question, the same words, base model Qwen2.5-Math-1.5B (no chat format; T 0.6, top-p 0.95, seed 0, max 1,024)
  | To determine how much Janet makes every day at the farmers' market, we need to follow these steps:
  ...
  | 4. The number of eggs remaining to be sold is \(16 - 3 - 4 = 9\) eggs per day.
  | 5. Each egg is sold for $2, so the revenue per day is \(9 \times 2 = 18\) dollars.
  ...
  | The final answer is \(\boxed{18}\). Janet makes $18 every day at the farmers' market.<|endoftext|>
  new tokens 377; stopped by end-of-text: True; contains <think>: False; boxed answers found: ['18']
  first boxed answer '18'; correct (our checker) True; correct (math-verify) True
```

The base model is right in 377 tokens. It writes a short plan, does the arithmetic, then writes a Python program to check it and prints the program's "output". There is no `<think>`, no "Wait", and no second attempt.

This is worth a pause, because it is not the naive story. The base model is not a blank page. Qwen2.5-Math was pre-trained and tuned on a great deal of maths, and it already writes short chains of thought. What the distilled model added is the long, self-checking style. On an easy problem, that style costs many tokens and buys nothing.

#### Pass@1 and majority voting, by code

The last test uses a harder problem from MATH-500: "What is the least positive integer multiple of 30 that can be written with only the digits 0 and 2?" A brute-force loop over the multiples of 30 confirms the answer 2220. Each model writes 8 samples. The distilled model may use up to 6,144 new tokens, and the base model up to 1,024.

```text
  -- distilled model, max 6,144 new tokens, 701 s for the batch of 8
  sample 0:  6144 tokens, thinking     -, "Wait" x80  answer None         correct False (math-verify False)
  sample 1:  6144 tokens, thinking     -, "Wait" x41  answer None         correct False (math-verify False)
  sample 2:  5197 tokens, thinking  4759, "Wait" x33  answer '2220'       correct True (math-verify True)
  sample 3:  6144 tokens, thinking     -, "Wait" x62  answer None         correct False (math-verify False)
  sample 4:  3465 tokens, thinking  3202, "Wait" x16  answer '2220'       correct True (math-verify True)
  sample 5:  2202 tokens, thinking  1969, "Wait" x5   answer '2220'       correct True (math-verify True)
  sample 6:  6144 tokens, thinking     -, "Wait" x19  answer None         correct False (math-verify False)
  sample 7:  3624 tokens, thinking  3287, "Wait" x17  answer '2220'       correct True (math-verify True)
  pass@1 = 4/8 = 0.500; votes {'2220': 4}; majority answer '2220'; majority correct: True
  -- base model, max 1,024 new tokens, 48 s for the batch of 8
  ...
  pass@1 = 7/8 = 0.875; votes {'2220': 7, '210': 1}; majority answer '2220'; majority correct: True
```

Every finished sample of the distilled model is right: four of four. The other four were still thinking at 6,144 tokens, with up to 80 "Wait"s. So pass@1 is $$\frac{1}{8}(0+0+1+0+1+1+0+1) = 0.5$$, the formula of the definition box with $$k = 8$$. The vote counts only the samples that gave an answer, and all four say 2220. So the majority answer is right, although only half of the samples were.

The base model is right in 7 of 8 samples, with 196 to 589 tokens each. One sample said 210, a multiple of 30 with a 1 in it. Its vote is 7 to 1 for 2220.

{{FIG:p1_samples|The eight samples of each model on one MATH-500 problem, from our run. The distilled model's bars are long: four samples finish with the right answer and four reach the 6,144-token limit. The base model's bars are short, and seven of eight are right.}}

So, on our laptop and with these limits, the base model did as well or better on both questions. This does not contradict the paper. Table 5 gives the distilled 1.5B model 83.9% on MATH-500, with up to 32,768 tokens per answer. Our 6,144 tokens cut half of its answers.

The run also says something real about this model: the smallest student copies R1's long, careful style, but it checks too often and too long. Test-time scaling is not free. It buys accuracy on hard problems with tokens, and a small model does not always know when to stop. Part 6 compares the distilled models with their base models on full benchmarks, and Part 7 reads the scoring rules in detail.

## Contributions {§1.1}

> [!PAPER] · §1.1 · page 4
> [![The first contribution, with directly apply RL to the base model, explore chain-of-thought (CoT), self-verification, reflection, and generating long CoTs, and incentivized purely through RL, without the need for SFT highlighted](/img/papers/deepseek-r1/p1-contrib-zero.png)](/img/papers/deepseek-r1/p1-contrib-zero.png)
>
> **What it says:** "We directly apply RL to the base model without relying on supervised fine-tuning (SFT) as a preliminary step. This approach allows the model to explore chain-of-thought (CoT) [...]. DeepSeek-R1-Zero demonstrates capabilities such as self-verification, reflection, and generating long CoTs [...]. Notably, it is the first open research to validate that reasoning capabilities of LLMs can be incentivized purely through RL, without the need for SFT."
>
> **What it means:**
>
> "Explore" is the right verb for RL. The model writes many different chains of thought; the reward tells it which ones led somewhere. Over thousands of steps, the styles that work grow and the others fade. **Self-verification** means checking a result, for example by putting the answer back into the equation. **Reflection** means going back to an earlier step and judging it again.
>
> Read "first open research" with care. The claim is not that nobody had done this. OpenAI may have done something similar for o1, but it published no method. The claim is that this is the first public, checkable evidence that pure RL can create these habits. With open weights, anyone can test it.

> [!PAPER] · §1.1 · page 4
> [![The second contribution, with two RL stages, two SFT stages and seed highlighted](/img/papers/deepseek-r1/p1-contrib-pipeline.png)](/img/papers/deepseek-r1/p1-contrib-pipeline.png)
>
> **What it says:** "The pipeline incorporates two RL stages aimed at discovering improved reasoning patterns and aligning with human preferences, as well as two SFT stages that serve as the seed for the model's reasoning and non-reasoning capabilities."
>
> **What it means:**
>
> This sentence gives each stage a job. The two SFT stages are "seeds": they plant a behaviour. The first plants a readable reasoning style; the second plants general skills such as writing and answering facts. The two RL stages make the plants grow. The first finds better reasoning; the second makes the model helpful and safe for people.
>
> A rule of thumb hides here. SFT is good at teaching a format and a style quickly, because the model sees the target directly. RL is good at improving a skill past the level of the examples, because it rewards results, not imitation. R1 alternates the two to use both strengths.

> [!PAPER] · §1.1 · page 4
> [![The first distillation contribution, with reasoning patterns of larger models can be distilled into smaller models and better performance compared to the reasoning patterns discovered through RL on small models highlighted](/img/papers/deepseek-r1/p1-contrib-distill-1.png)](/img/papers/deepseek-r1/p1-contrib-distill-1.png)
>
> **What it says:** "the reasoning patterns of larger models can be distilled into smaller models, resulting in better performance compared to the reasoning patterns discovered through RL on small models."
>
> **What it means:**
>
> This is the practical lesson for anyone without a giant model. If you want a small reasoning model, do not run large-scale RL on the small model. Fine-tune it on a strong reasoner's outputs. The paper adds that RL on top of the distilled model gives more gains (§3.2), but it leaves that step to others.

> [!PAPER] · §1.1 · page 4
> [![The second distillation contribution, with 55.5% on AIME 2024, 72.6% on AIME 2024, 94.3% on MATH-500, 57.2% on LiveCodeBench, comparable to o1-mini and the sizes 1.5B to 70B highlighted](/img/papers/deepseek-r1/p1-contrib-distill-2.png)](/img/papers/deepseek-r1/p1-contrib-distill-2.png)
>
> **What it says:** "DeepSeek-R1-Distill-Qwen-7B achieves 55.5% on AIME 2024, surpassing QwQ-32B-Preview. Additionally, DeepSeek-R1-Distill-Qwen-32B scores 72.6% on AIME 2024, 94.3% on MATH-500, and 57.2% on LiveCodeBench. [...] We open-source distilled 1.5B, 7B, 8B, 14B, 32B, and 70B checkpoints based on Qwen2.5 and Llama3 series".
>
> **What it means:**
>
> Put the 32B student next to its teacher. On AIME, DeepSeek-R1 gets 79.8% and the 32B student gets 72.6%. The student keeps most of the teacher's skill with about one twentieth of the parameters. **LiveCodeBench** is a set of programming problems from recent contests, here from August 2024 to January 2025. So the model is less likely to have seen them in training.
>
> Our 1.5B model is the smallest of the six. Table 5 gives it 28.9% pass@1 on AIME and 83.9% on MATH-500. That is weak against R1, but higher than GPT-4o on both (9.3% and 74.6%).

## Summary of evaluation results {§1.2}

> [!PAPER] · §1.2 · page 4
> [![The reasoning results, with 79.8% Pass@1 on AIME 2024, 97.3%, 2,029 Elo rating on Codeforces, 96.3% human participants and slightly better than highlighted](/img/papers/deepseek-r1/p1-results-reasoning.png)](/img/papers/deepseek-r1/p1-results-reasoning.png)
>
> **What it says:** "DeepSeek-R1 achieves a score of 79.8% Pass@1 on AIME 2024, slightly surpassing OpenAI-o1-1217. On MATH-500, it attains an impressive score of 97.3% [...] it achieves 2,029 Elo rating on Codeforces outperforming 96.3% human participants in the competition. For engineering-related tasks, DeepSeek-R1 performs slightly better than DeepSeek-V3".
>
> **What it means:**
>
> An **Elo rating** is a skill number from chess, also used by Codeforces. A player who beats strong players gains many points, and one who loses to weak players drops many. A rating of 2,029 puts R1 above 96.3% of the human contest players. These are people who choose to do programming contests, not average programmers.
>
> The last sentence is honest. On engineering tasks, such as fixing bugs in a real code base, R1 is only "slightly better" than V3. Contest problems are short and have tests that a program can run quickly. Real software tasks are long and slow to check, so RL had little data there.

> [!PAPER] · §1.2 · page 4
> [![The knowledge results, with 90.8% on MMLU, 84.0% on MMLU-Pro, and 71.5% on GPQA Diamond, slightly below that of OpenAI-o1-1217, and SimpleQA highlighted](/img/papers/deepseek-r1/p1-results-knowledge.png)](/img/papers/deepseek-r1/p1-results-knowledge.png)
>
> **What it says:** DeepSeek-R1 scores "90.8% on MMLU, 84.0% on MMLU-Pro, and 71.5% on GPQA Diamond. While its performance is slightly below that of OpenAI-o1-1217 on these benchmarks, DeepSeek-R1 surpasses other closed-source models". On "the factual benchmark SimpleQA, DeepSeek-R1 outperforms DeepSeek-V3".
>
> **What it means:**
>
> Knowledge tests ask what the model knows, not only how well it thinks. Most knowledge comes from pre-training, so post-training should change these scores less. It still helps: R1 beats V3 on all three. Table 4 shows that most of the gain is on science and maths questions, where careful reasoning helps.
>
> **SimpleQA** asks short factual questions, such as a date or a name. R1 gets 30.1% and V3 gets 24.9% (Table 4). Here R1 is behind o1-1217, which gets 47.0%. Thinking longer does not help much if a fact was never learned.

> [!PAPER] · §1.2 · page 5
> [![The other results, with 87.6% on AlpacaEval 2.0, 92.3% on ArenaHard and long-context understanding highlighted](/img/papers/deepseek-r1/p1-results-others.png)](/img/papers/deepseek-r1/p1-results-others.png)
>
> **What it says:** "It achieves an impressive length-controlled win-rate of 87.6% on AlpacaEval 2.0 and a win-rate of 92.3% on ArenaHard [...]. Additionally, DeepSeek-R1 demonstrates outstanding performance on tasks requiring long-context understanding".
>
> **What it means:**
>
> **AlpacaEval 2.0** and **ArenaHard** have no right answers. They are open requests, such as "write a short story" or "explain this code". A strong judge model, GPT-4-Turbo, compares the model's answer with a fixed reference answer and picks the better one. The **win rate** is the share of comparisons that the model wins. "Length-controlled" corrects for a known bias: judges like long answers.
>
> These scores show that reasoning training did not make R1 a narrow maths machine. The last RL stage, on prompts of all kinds, kept it a good general assistant. [Part 7](part-7-results-failures-and-limits.md) reads all of Table 4.

> [!TAKEAWAYS] Key takeaways
> - **The title's idea:** reward the result, not the steps. R1-Zero got RL with a reward for a right final answer and a right format, and no supervised fine-tuning on worked reasoning before it.
> - **Two models.** R1-Zero is RL straight on DeepSeek-V3-Base. R1 adds a cold start and alternates two SFT and two RL stages, to fix readability and language mixing and to reason better.
> - **The evidence:** R1-Zero's AIME 2024 pass@1 rose from 15.6% to 71.0% with RL alone, and to 86.7% with a vote of 64 answers. R1 reaches 79.8%, next to o1-1217's 79.2%.
> - **Same base, different post-training:** DeepSeek-V3 gets 39.2% on AIME and R1 gets 79.8%. On MMLU, a knowledge test, the gap is only 88.5% against 90.8%.
> - **Test-time scaling:** a longer chain of thought, or many answers and a vote, buys accuracy with compute at answer time.
> - **Distillation beats small-model RL:** a 32B model fine-tuned on R1's answers gets 72.6% on AIME; the same base with large-scale RL gets 47.0%.
> - **Our run:** the released 1.5B student showed real self-checking ("Wait, hold on. That can't be right"). It also overthought: 1,907 thinking tokens on a two-step problem, and 4 of 8 MATH-500 samples still thinking at 6,144 tokens.
> - **Pass@1 and the vote, by code:** the student's pass@1 was 4/8 = 0.500, but its majority answer, 2220, was right. The base model got 7/8 in 196 to 589 tokens each.
> - **Two methods agree:** our temperature and top-p match the transformers warpers to 6.0e-08, and our greedy loop writes the same 128 tokens as model.generate.

**Next, in Part 2:** §2.1 and §2.2.1, the RL algorithm. Part 2 reads Equations 1 to 3 of GRPO term by term, writes the objective from scratch in Python, and checks it against a library.

<details>
<summary>Run it yourself</summary>

The script behind this part is [`code/papers/deepseek-r1/r1_part1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/deepseek-r1/r1_part1.py). It loads both 1.5B models and runs every step above. The limits are 128 new tokens for the greedy check and 2,048 for the distilled model's GSM8K samples. The base model gets 1,024, and the distilled model's MATH-500 samples get 6,144.

It uses float32, seed 0, and the Apple laptop GPU (`mps`), or the CPU. The longest step, 8 samples of up to 6,144 tokens, took 701 s. The numbers are saved in `results/r1_part1.json`.

```bash
pip install torch transformers datasets math-verify
python r1_part1.py     # downloads DeepSeek-R1-Distill-Qwen-1.5B and Qwen2.5-Math-1.5B (about 3.5 GB and 3.1 GB)
```

<figure><img src="/img/papers/deepseek-r1/r1_part1-run.png" alt="Terminal output of r1_part1.py: the two models, the sampling table, the greedy check and the start of the distilled model's thinking" loading="lazy" /><figcaption>The first lines of the output of r1_part1.py, as it ran for this article.</figcaption></figure>

</details>

## The paper and the code

[DeepSeek-R1 on arXiv](https://arxiv.org/abs/2501.12948) · [official repository](https://github.com/deepseek-ai/DeepSeek-R1) · [our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/deepseek-r1)
