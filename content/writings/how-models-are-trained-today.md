---
title: "From RLHF to Reasoning: How Language Models Are Trained Today"
description: "How chat and reasoning models are taught after pre-training: human preference labels, AI feedback, answers a program can check, and how we test the result."
date: 2026-10-22
tags: [history, llm, reinforcement-learning]
series: "How AI Got Here"
series_part: 4
motif: graph
accent: "#c4a1ff"
---

You give a model a hard maths problem. It writes a few lines, stops, and writes "Wait, wait. Wait." Then it goes back and checks its own work. Nobody typed that line into its training data as an example to copy. The model found that checking earned more reward, so checking became a habit.

That line comes from DeepSeek-R1-Zero, released in January 2025. After pre-training, it was trained with reinforcement learning and almost nothing else. During that training, its score on the AIME 2024 maths competition rose from 15.6% to 77.9%. The reward was simple: 1 if the final answer was right, 0 if it was not.

This is Part 4 of How AI Got Here. [Part 1](from-neuron-to-opus.md) told the story of neural networks and gave a short account of RLHF; [Part 2](classical-machine-learning.md) covered classical machine learning. [Part 3](reinforcement-learning-history.md) followed reinforcement learning from Thorndike's cats to PPO and AlphaZero. This part shows how those methods became the way language models are trained, from 2017 to October 2026. [Part 5](inside-a-modern-language-model.md) then opens the base model itself.

> [!TIP] The whole idea in one minute
> Pre-training makes a model that continues text. **Post-training** turns it into an assistant. First, people's choices between two answers trained a **reward model**, and PPO pushed the language model toward high reward without drifting too far: **RLHF**. Then AI feedback and written principles replaced some human labels, and **DPO** removed the RL step for preferences.
>
> Since 2024, the biggest gains come from rewards that a program can check, such as a correct maths answer or a passing test. **GRPO** made this cheap, and DeepSeek-R1 showed that long reasoning can emerge from it. In 2026, a frontier model passes through many stages, from pre-training to agent training in tool environments. The labs publish only an outline of them.

<figure class="fig"><svg viewBox="0 0 720 364" role="img" aria-label="A timeline of post-training methods from 2017 to 2026. Preference learning comes first, then RLHF assistants, Constitutional AI, DPO, verifiers, reasoning RL and RL for agents."><text class="t-title" x="10" y="22">From preferences to reasoning, 2017 to 2026</text><line class="grid" x1="271.6" y1="34" x2="271.6" y2="326"/><line class="grid" x1="358.2" y1="34" x2="358.2" y2="326"/><line class="grid" x1="444.7" y1="34" x2="444.7" y2="326"/><line class="grid" x1="531.2" y1="34" x2="531.2" y2="326"/><line class="grid" x1="617.8" y1="34" x2="617.8" y2="326"/><text class="t-note" x="10" y="52">Learning from preferences</text><text class="t-tick" x="10" y="68">2017 to 2020</text><rect class="box-1" x="291.1" y="42" width="140.6" height="22" rx="4"/><text class="t-note" x="10" y="92">RLHF assistants</text><text class="t-tick" x="10" y="108">2022 to 2023</text><rect class="box-2" x="496.6" y="82" width="60.6" height="22" rx="4"/><text class="t-note" x="10" y="132">Constitutional AI, RLAIF</text><text class="t-tick" x="10" y="148">2022 to 2023</text><rect class="box-3" x="529.1" y="122" width="32.5" height="22" rx="4"/><text class="t-note" x="10" y="172">DPO and its family</text><text class="t-tick" x="10" y="188">2023 to 2024</text><rect class="box-4" x="548.6" y="162" width="47.6" height="22" rx="4"/><text class="t-note" x="10" y="212">Verifiers, process rewards</text><text class="t-tick" x="10" y="228">2022 to 2024</text><rect class="box-1" x="496.6" y="202" width="116.8" height="22" rx="4"/><text class="t-note" x="10" y="252">Reasoning RL: o1, R1, RLVR</text><text class="t-tick" x="10" y="268">2024 to 2026</text><rect class="box-2" x="604.8" y="242" width="90.9" height="22" rx="4"/><text class="t-note" x="10" y="292">RL for agents and tools</text><text class="t-tick" x="10" y="308">2025 to 2026</text><rect class="box-3" x="617.8" y="282" width="77.9" height="22" rx="4"/><line class="axis" x1="250" y1="328" x2="700" y2="328"/><text class="t-tick" x="271.635" y="348" text-anchor="middle">2017</text><text class="t-tick" x="358.173" y="348" text-anchor="middle">2019</text><text class="t-tick" x="444.712" y="348" text-anchor="middle">2021</text><text class="t-tick" x="531.25" y="348" text-anchor="middle">2023</text><text class="t-tick" x="617.788" y="348" text-anchor="middle">2025</text></svg><figcaption>Nine years, seven overlapping threads. Each one is still in use; new stages were added on top of the old ones, not in their place.</figcaption></figure>

## Why a pre-trained model needs more training

A model trained only to predict the next token is called a **base model**. It has read a large part of the public web, so it knows a great deal. But it continues text; it does not answer. Ask it "How do I bake bread?" and it may write three more questions, as a web forum might.

> [!DEFINITION] Base model
> A language model after pre-training only. It predicts the next token of any text, so it imitates whatever kind of document the prompt looks like.

> [!DEFINITION] Post-training
> All the training a model gets after pre-training, to make it follow instructions, be helpful and safe, and reason well. It uses far fewer tokens than pre-training, but it shapes most of what a user notices.

The first step of post-training is **supervised fine-tuning** (SFT). People write good answers to many prompts, and the model is trained on them with the same next-token loss as before. It is like an apprentice copying the work of a master. It works, but it has limits.

> [!DEFINITION] Supervised fine-tuning (SFT)
> Training a pre-trained model on examples of prompts with good answers, using the ordinary next-token loss. The model learns the format and style of an assistant.

Writing a great answer is slow and costly, and the best a copy can be is as good as the person who wrote it. Judging is easier than writing. Most people cannot write a prize-winning essay, but they can tell which of two essays they prefer. That difference is the starting point of everything in this part.

## 2017 to 2020: learning a reward from people

### Preferences between video clips

In June 2017, Paul Christiano and colleagues at OpenAI and DeepMind showed how to train an agent when nobody can write the reward. A person watched two short video clips of the agent and picked the one that looked closer to the goal. A network learned to predict those choices. The agent then ran ordinary reinforcement learning against the predicted reward, while new clips went to the person.

> [!PAPER] Christiano et al. (2017), Deep Reinforcement Learning from Human Preferences · Abstract · page 1
> [![The abstract of Christiano et al., with human preferences between pairs of trajectory segments, less than 1% of our agent's interactions and about an hour of human time highlighted](/img/writings/history/h-christiano.png)](/img/writings/history/h-christiano.png)
>
> **What it says:** goals are defined "in terms of (non-expert) human preferences between pairs of trajectory segments". The method solves Atari games and simulated robot tasks "without access to the reward function", with feedback on "less than 1% of our agent's interactions". Novel behaviours took "about an hour of human time".
>
> **What it means:**
>
> A reward function is a written definition of "good". This paper replaced it with a learned guess of what a person would choose. "Less than 1%" is the important number: a person does not have to watch everything, only a small sample. Every method in the first half of this part keeps that shape.

### From clips to text

In 2019, Daniel Ziegler and colleagues at OpenAI applied the method to GPT-2. People compared continuations of text and later summaries. The paper used a KL penalty to keep the fine-tuned model close to the original, following earlier work by Natasha Jaques and colleagues. It also found an early warning. With 60,000 comparisons, the summarisers learned to copy whole sentences from the input, which pleased the labellers but was not real summarising.

In September 2020, Nisan Stiennon and colleagues scaled the method for summaries of Reddit posts (the TL;DR data set).

> [!PAPER] Stiennon et al. (2020), Learning to summarize from human feedback · Figure 1 · page 2
> [![Figure 1 of Stiennon et al.: the fraction of summaries people prefer to the human-written reference, against model size from 1.3B to 12.9B, for human feedback, supervised learning and pre-training only](/img/writings/history/h-stiennon-fig.png)](/img/writings/history/h-stiennon-fig.png)
>
> **What it says:** the plot shows the "fraction of the time humans prefer our models' summaries over the human-generated reference summaries". The caption adds that length differences "explain about a third of the gap between feedback and supervised learning at 6.7B".
>
> **What it means:**
>
> The dashed line at 0.5 is the human-written summaries. The human-feedback models sit above it, and the supervised models below it at every size. Even the smallest human-feedback model, at 1.3B parameters, beats the largest supervised one. The length remark is a warning that later papers took seriously: people, and reward models, tend to like longer answers.

## The reward model, up close

### Ranking by comparisons

How do you turn "A is better than B" into a number for each answer? Chess has the same problem. Players are never given an absolute score, only wins and losses, yet the Elo rating gives each player one number. The gap between two ratings predicts who will win. In 1952, Ralph Bradley and Milton Terry wrote down the model behind such ratings.

> [!DEFINITION] Bradley-Terry model
> A model of pairwise choices. Each item gets a score. The chance that item A is preferred to item B depends only on the difference of their scores, through the sigmoid function.

$$
P(y_w \succ y_l \mid x) = \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$x$$ is the prompt, $$y_w$$ is the answer people preferred (the winner) and $$y_l$$ is the other one (the loser). The symbol $$\succ$$ means "is preferred to". $$r_\phi(x, y)$$ is the reward model's score for an answer, with parameters $$\phi$$ (phi). The sigmoid $$\sigma$$ turns any number into a probability between 0 and 1.

Training makes the observed choices as likely as possible, which gives the loss that Part 1 showed briefly:

$$
\mathcal{L}_{\text{RM}}(\phi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \Big[ \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \Big]
$$

$$\mathcal{D}$$ is the data set of comparisons and $$\mathbb{E}$$ means the average over it. The loss is small when the winner's score is well above the loser's.

Our script gives the loss for four score gaps. A gap of 2 gives a probability of 0.8808 and a loss of 0.1269. A gap of 0.5 gives 0.4741. A gap of 0 gives $$\ln 2 = 0.6931$$, which means "no idea". A gap of $$-2$$, the wrong way round, gives 2.1269.

<figure class="fig"><svg viewBox="0 0 720 276" role="img" aria-label="Two answers to the same prompt go through the same Transformer and a linear head, giving scores 2.0 and 0.0. The sigmoid of the difference is 0.881 and the loss is 0.127."><text class="t-title" x="10" y="22">A reward model: a language model with a one-number head</text><rect class="box-1" x="20" y="50" width="200" height="50" rx="6"/><text class="t-note" x="120" y="71" text-anchor="middle">prompt x + answer y_w</text><text class="t-muted" x="120" y="88" text-anchor="middle">(the one people chose)</text><rect class="box-4" x="20" y="150" width="200" height="50" rx="6"/><text class="t-note" x="120" y="171" text-anchor="middle">prompt x + answer y_l</text><text class="t-muted" x="120" y="188" text-anchor="middle">(the one they rejected)</text><rect class="box" x="260" y="50" width="170" height="50" rx="6"/><text class="t-note" x="345" y="71" text-anchor="middle">Transformer</text><text class="t-muted" x="345" y="88" text-anchor="middle">from the SFT model</text><rect class="box" x="260" y="150" width="170" height="50" rx="6"/><text class="t-note" x="345" y="171" text-anchor="middle">same Transformer</text><text class="t-muted" x="345" y="188" text-anchor="middle">same weights</text><rect class="box-on" x="470" y="50" width="90" height="50" rx="6"/><text class="t-note" x="515" y="71" text-anchor="middle">r_w = 2.0</text><text class="t-muted" x="515" y="88" text-anchor="middle">linear head</text><rect class="box-on" x="470" y="150" width="90" height="50" rx="6"/><text class="t-note" x="515" y="171" text-anchor="middle">r_l = 0.0</text><text class="t-muted" x="515" y="188" text-anchor="middle">linear head</text><polyline class="edge" points="220,75 260,75"/><polygon class="arrow" points="260.0,75.0 252.0,79.0 252.0,71.0"/><polyline class="edge" points="220,175 260,175"/><polygon class="arrow" points="260.0,175.0 252.0,179.0 252.0,171.0"/><polyline class="edge" points="430,75 470,75"/><polygon class="arrow" points="470.0,75.0 462.0,79.0 462.0,71.0"/><polyline class="edge" points="430,175 470,175"/><polygon class="arrow" points="470.0,175.0 462.0,179.0 462.0,171.0"/><rect class="box-3" x="590" y="90" width="115" height="70" rx="6"/><text class="t-note" x="647.5" y="112.5" text-anchor="middle">σ(2.0 − 0.0)</text><text class="t-muted" x="647.5" y="129.5" text-anchor="middle">= 0.881</text><text class="t-muted" x="647.5" y="146.5" text-anchor="middle">loss 0.127</text><polyline class="edge" points="560,75 590,110"/><polygon class="arrow" points="590.0,110.0 581.8,106.5 587.8,101.3"/><polyline class="edge" points="560,175 590,140"/><polygon class="arrow" points="590.0,140.0 587.8,148.7 581.8,143.5"/><text class="t-muted" x="20" y="240">The head reads the hidden state at the last token and outputs one number. Only the difference</text><text class="t-muted" x="20" y="258">between two scores matters, so the loss teaches an ordering, not an absolute grade.</text></svg><figcaption>The Bradley-Terry reward model. The numbers are the worked example from the text, computed by our script.</figcaption></figure>

Only differences matter. If you add 10 to every score, the loss does not change. So a reward model learns an ordering, not a grade, and its numbers mean nothing on their own. In practice it is a copy of the language model with its word output replaced by a single number. The number is read at the last token of the answer.

> [!DEFINITION] Reward model
> A network trained to predict which of two answers a person would prefer. It reads a prompt and an answer and outputs one number. During RL, it stands in for the person and scores every new answer.

## 2022: InstructGPT and the RLHF objective

### Three stages

In March 2022, Long Ouyang and colleagues at OpenAI published InstructGPT, which Part 1 described. Their recipe had three stages, and the paper gave the sizes of the data. About 13,000 prompts with written answers went into SFT. People ranked answers for 33,000 prompts to train the reward model. A further 31,000 prompts, with no labels at all, were used for the RL stage.

<figure class="fig"><svg viewBox="0 0 720 336" role="img" aria-label="The three stages of RLHF: supervised fine-tuning, a reward model trained on ranked answers, and PPO training against the reward model with a KL penalty."><text class="t-title" x="10" y="22">Classic RLHF in three stages</text><rect class="box-1" x="20" y="50" width="210" height="70" rx="6"/><text class="t-note" x="125" y="81" text-anchor="middle">1. Supervised fine-tuning</text><text class="t-muted" x="125" y="98" text-anchor="middle">copy answers people wrote</text><rect class="box-2" x="255" y="50" width="210" height="70" rx="6"/><text class="t-note" x="360" y="81" text-anchor="middle">2. Reward model</text><text class="t-muted" x="360" y="98" text-anchor="middle">learn which answer people prefer</text><rect class="box-3" x="490" y="50" width="210" height="70" rx="6"/><text class="t-note" x="595" y="81" text-anchor="middle">3. RL with PPO</text><text class="t-muted" x="595" y="98" text-anchor="middle">raise the reward, stay close</text><polyline class="edge" points="230,85 255,85"/><polygon class="arrow" points="255.0,85.0 247.0,89.0 247.0,81.0"/><polyline class="edge" points="465,85 490,85"/><polygon class="arrow" points="490.0,85.0 482.0,89.0 482.0,81.0"/><rect class="box" x="20" y="160" width="210" height="60" rx="6"/><text class="t-note" x="125" y="186" text-anchor="middle">data: prompt + good answer</text><text class="t-muted" x="125" y="203" text-anchor="middle">InstructGPT: about 13k prompts</text><rect class="box" x="255" y="160" width="210" height="60" rx="6"/><text class="t-note" x="360" y="186" text-anchor="middle">data: prompt + ranked answers</text><text class="t-muted" x="360" y="203" text-anchor="middle">InstructGPT: 33k prompts</text><rect class="box" x="490" y="160" width="210" height="60" rx="6"/><text class="t-note" x="595" y="186" text-anchor="middle">data: prompts only</text><text class="t-muted" x="595" y="203" text-anchor="middle">InstructGPT: 31k prompts</text><line class="edge-dim" x1="125" y1="120" x2="125" y2="160"/><line class="edge-dim" x1="360" y1="120" x2="360" y2="160"/><line class="edge-dim" x1="595" y1="120" x2="595" y2="160"/><rect class="box-on" x="255" y="260" width="445" height="54" rx="6"/><text class="t-note" x="477.5" y="291.5" text-anchor="middle">score = r(x, y) − β · log( π(y | x) / π_SFT(y | x) )</text><polyline class="edge-on" points="360,220 360,260"/><polygon class="arrow-on" points="360.0,260.0 356.0,252.0 364.0,252.0"/><polyline class="edge-on" points="595,260 595,220"/><polygon class="arrow-on" points="595.0,220.0 599.0,228.0 591.0,228.0"/><text class="t-muted" x="20" y="280">The SFT model is also the</text><text class="t-muted" x="20" y="298">reference that the KL term</text><text class="t-muted" x="20" y="316">measures distance from.</text></svg><figcaption>The InstructGPT recipe of 2022, with its dataset sizes. Stage 3 needs no new human labels: the reward model scores every answer.</figcaption></figure>

### The objective with a leash

In the third stage, the model writes answers and the reward model scores them. Left alone, PPO would push the model toward whatever the reward model happens to like most, even nonsense. So the objective adds a penalty for moving away from the SFT model. It works like a leash on a dog, which can explore but not run off.

$$
\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot \mid x)} \big[ r_\phi(x, y) \big] \;-\; \beta \, \mathbb{D}_{\text{KL}} \big[ \pi_\theta(y \mid x) \,\Vert\, \pi_{\text{ref}}(y \mid x) \big]
$$

$$\pi_\theta$$ is the language model being trained, the **policy**. $$\pi_{\text{ref}}$$ is the **reference model**, usually the SFT model, kept frozen. $$y \sim \pi_\theta(\cdot \mid x)$$ means "an answer sampled from the policy for prompt $$x$$". $$\mathbb{D}_{\text{KL}}$$ is the KL divergence between the two models' distributions over answers. $$\beta$$ (beta) sets the length of the leash.

> [!DEFINITION] KL divergence
> A measure of how different two probability distributions are. It is 0 when they are the same and grows as they differ. For one sampled answer, the estimate used in RLHF is $$\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x)$$.

> [!DEFINITION] Reference model
> A frozen copy of the model from before the current training stage. The KL penalty measures how far the trained model has moved from it.

In practice, each sampled answer gets a penalised reward: the reward model's score minus $$\beta$$ times the log ratio of the two models' probabilities. Take $$\beta = 0.1$$. Answer 1 scores 2.0, and the policy gives it a log-probability of $$-12$$ against $$-11$$ for the reference. Its penalised reward is $$2.0 - 0.1 \times (-1) = 2.1$$. Answer 2 scores higher, 2.6, but the policy has moved far to produce it: $$-14$$ against $$-22$$. Its penalised reward is $$2.6 - 0.1 \times 8 = 1.8$$, so the leash prefers answer 1.

InstructGPT also mixed in some pre-training gradients (PPO-ptx), so the model would not forget skills while it learned to please the reward model. The paper called the loss of skills an "alignment tax".

### Helpful and harmless

In April 2022, Yuntao Bai and colleagues at Anthropic published "Training a Helpful and Harmless Assistant with RLHF". They collected two kinds of comparisons, one for helpfulness and one for harmlessness, and found the two goals pull against each other. A model that refuses everything is harmless but useless. They also retrained with fresh human feedback each week, an "online" form of RLHF. And they reported "a roughly linear relation between the RL reward and the square root of the KL divergence".

## Goodhart's law: when the reward model is wrong

> [!DEFINITION] Goodhart's law
> "When a measure becomes a target, it ceases to be a good measure." A number that tracks quality well stops tracking it once people, or a training algorithm, optimise the number directly.

A reward model is a measure, and RL makes it a target. It was trained on answers from the SFT model. As the policy moves away, it writes answers the reward model has never seen, and the scores become less reliable. The policy finds and exploits those blind spots. This is **reward hacking**.

> [!DEFINITION] Reward hacking
> When an agent finds a way to get a high reward that does not match what the designers wanted. A language model might learn that longer answers, flattery or confident tone score well, whether or not they are better.

In October 2022, Leo Gao, John Schulman and Jacob Hilton measured this without human labels. A large "gold" reward model played the role of people and labelled data. Smaller "proxy" reward models were trained on those labels, and policies were optimised against the proxies. Since the gold model was available, they could watch the true score as the proxy score rose.

> [!PAPER] Gao, Schulman and Hilton (2022), Scaling Laws for Reward Model Overoptimization · Figure 1 · page 3
> [![Figure 1b of Gao et al.: gold and proxy reward model scores against the KL distance between the RL-tuned policy and the initial policy, for proxy reward models from 3M to 3B parameters; the dashed proxy curves keep rising while the solid gold curves rise and then fall](/img/writings/history/h-gao-fig.png)](/img/writings/history/h-gao-fig.png)
>
> **What it says:** "when we optimize for a learned proxy of the gold reward, the gold reward initially increases and later decreases". The x-axes have a square-root scale, and the authors' "functional forms fit this effect well".
>
> **What it means:**
>
> The dashed lines are what the policy is trained on, and they keep rising. The solid lines are the truth, and each one has a peak. Past the peak, every extra unit of proxy reward makes the model worse. Bigger proxy models (the yellow lines) peak later and higher, but they still peak.

The paper fits a simple curve to each line. With $$d = \sqrt{\mathbb{D}_{\text{KL}}}$$, the distance from the start, the gold score under RL follows

$$
R_{\text{RL}}(d) = d \, \big( \alpha_{\text{RL}} - \beta_{\text{RL}} \log d \big)
$$

At first the term $$\alpha_{\text{RL}} d$$ wins and the true score climbs. The second term, $$\beta_{\text{RL}}\, d \log d$$, grows faster, and eventually pulls the score back down. $$\alpha_{\text{RL}}$$ and $$\beta_{\text{RL}}$$ are fitted numbers that depend on the size of the reward model. The lesson for practice: watch the KL, keep the leash, and stop early.

## Best-of-n and rejection sampling

### The simplest optimiser

There is a way to use a reward model with no RL at all. Sample $$n$$ answers, score them all, and keep the best. It is like a photographer who takes twenty shots and prints one.

$$
y_{\text{BoN}} = \arg\max_{y \in \{y_1, \dots, y_n\}} r_\phi(x, y), \qquad \mathbb{D}_{\text{KL}}\big(\pi_{\text{BoN}} \,\Vert\, \pi\big) = \log n - \frac{n - 1}{n}
$$

$$y_1$$ to $$y_n$$ are answers sampled independently from the model $$\pi$$. The first formula keeps the one with the highest reward. The second gives the KL divergence of this best-of-n procedure from the model, in nats. Gao and colleagues take it from the appendix of Stiennon and colleagues. It grows only like $$\log n$$.

Say each answer is right with probability 0.3, and a perfect judge picks a right one when there is one. Then the chance of success is $$1 - 0.7^n$$. For $$n = 4$$ it is 0.7599, for $$n = 16$$ it is 0.9967, at a KL cost of 0.6363 and 1.8351 nats. Real reward models are not perfect judges, so best-of-n can be hacked too; the Gao paper shows the same rise and fall for it.

<figure class="fig"><svg viewBox="0 0 720 330" role="img" aria-label="A line chart over n equal to 1, 2, 4, 8, 16 and 64. The chance of at least one right answer rises from 0.3 to almost 1. The KL distance rises from 0 to 3.17 nats."><text class="t-title" x="10" y="22">Best-of-n: more samples, more chance of a right one, more distance from the model</text><line class="grid" x1="70" y1="250.0" x2="660" y2="250.0"/><text class="t-tick" x="62" y="254" text-anchor="end">0</text><text class="t-tick" x="668" y="254">0</text><line class="grid" x1="70" y1="200.0" x2="660" y2="200.0"/><text class="t-tick" x="62" y="204" text-anchor="end">0.25</text><text class="t-tick" x="668" y="204">1</text><line class="grid" x1="70" y1="150.0" x2="660" y2="150.0"/><text class="t-tick" x="62" y="154" text-anchor="end">0.5</text><text class="t-tick" x="668" y="154">2</text><line class="grid" x1="70" y1="100.0" x2="660" y2="100.0"/><text class="t-tick" x="62" y="104" text-anchor="end">0.75</text><text class="t-tick" x="668" y="104">3</text><line class="grid" x1="70" y1="50.0" x2="660" y2="50.0"/><text class="t-tick" x="62" y="54" text-anchor="end">1</text><text class="t-tick" x="668" y="54">4</text><line class="axis" x1="70" y1="250" x2="660" y2="250"/><text class="t-tick" x="70" y="268" text-anchor="middle">n = 1</text><text class="t-tick" x="188" y="268" text-anchor="middle">n = 2</text><text class="t-tick" x="306" y="268" text-anchor="middle">n = 4</text><text class="t-tick" x="424" y="268" text-anchor="middle">n = 8</text><text class="t-tick" x="542" y="268" text-anchor="middle">n = 16</text><text class="t-tick" x="660" y="268" text-anchor="middle">n = 64</text><polyline class="l1" points="70.0,190.0 188.0,148.0 306.0,98.0 424.0,61.5 542.0,50.7 660.0,50.0"/><polyline class="l2" points="70.0,250.0 188.0,240.3 306.0,218.2 424.0,189.8 542.0,158.2 660.0,91.3"/><circle class="s1" cx="70.0" cy="190.0" r="4"/><circle class="s2" cx="70.0" cy="250.0" r="4"/><circle class="s1" cx="188.0" cy="148.0" r="4"/><circle class="s2" cx="188.0" cy="240.3" r="4"/><circle class="s1" cx="306.0" cy="98.0" r="4"/><circle class="s2" cx="306.0" cy="218.2" r="4"/><circle class="s1" cx="424.0" cy="61.5" r="4"/><circle class="s2" cx="424.0" cy="189.8" r="4"/><circle class="s1" cx="542.0" cy="50.7" r="4"/><circle class="s2" cx="542.0" cy="158.2" r="4"/><circle class="s1" cx="660.0" cy="50.0" r="4"/><circle class="s2" cx="660.0" cy="91.3" r="4"/><text class="t-s1" x="70" y="292">● chance that at least one of n answers is right, if each is right with p = 0.3 (left axis)</text><text class="t-s2" x="70" y="312">● KL of best-of-n from the model, log n − (n − 1)/n, in nats (right axis)</text></svg><figcaption>Picking the best of n answers is the simplest way to optimise a reward. Its cost in KL grows like log n, so it is a good yardstick for how hard RL pushes a model. Numbers from our script.</figcaption></figure>

### Llama 2

**Rejection sampling fine-tuning** turns best-of-n into training: sample many answers, keep the best by the reward model, and fine-tune on them. Meta used it, together with PPO, for Llama 2-Chat in July 2023. Its paper is one of the most detailed public accounts of an RLHF pipeline.

> [!PAPER] Touvron et al. (2023), Llama 2: Open Foundation and Fine-Tuned Chat Models · Figure 4 · page 5
> [![Figure 4 of the Llama 2 paper: pretraining data and self-supervised learning give Llama 2; human preference data trains a safety reward model and a helpful reward model; fine-tuning runs supervised fine-tuning, then rejection sampling and PPO in a loop to give Llama-2-chat](/img/writings/history/h-llama2-fig.png)](/img/writings/history/h-llama2-fig.png)
>
> **What it says:** after pre-training and supervised fine-tuning, "the model is iteratively refined using Reinforcement Learning with Human Feedback (RLHF) methodologies, specifically through rejection sampling and Proximal Policy Optimization (PPO)". New reward-model data is collected in parallel so the reward models "remain within distribution".
>
> **What it means:**
>
> Two details stand out. There are two reward models, one for helpfulness and one for safety, because one score cannot hold both goals well. And the loop at the top runs several times: the paper trained five rounds, RLHF-V1 to V5. Its table of preference data lists 1,418,091 comparisons collected by Meta. Fresh labels on the newest model's answers keep the reward models from going stale.

## 2022 to 2026: feedback from AI and from written principles

Part 1 introduced **Constitutional AI**, which Anthropic published in December 2022. It has two phases. In the first, the model critiques and revises its own answers against a list of principles, and is fine-tuned on the revisions. In the second, a model compares pairs of answers against the principles, and those AI-made preferences train the reward model. This second phase is **RLAIF**, reinforcement learning from AI feedback.

> [!DEFINITION] RLAIF
> Reinforcement learning from AI feedback. A language model, given instructions or principles, labels which of two answers is better. Those labels train the reward model in place of human labels.

In 2023, Harrison Lee and colleagues at Google compared the two directly. Across summaries, helpful dialogue and harmless dialogue, they found that "RLAIF achieves comparable performance to RLHF". AI labels are fast and cheap, and the principles behind them are written down, so anyone can read them.

Written principles grew into long public documents. OpenAI published its **Model Spec** on 8 May 2024 and updated it in February 2025. On 22 January 2026, Anthropic published a new constitution for Claude, far longer than the 2023 one. It explains the reasons behind its rules, and Anthropic says the document is part of how the models are trained.

In December 2024, OpenAI described **deliberative alignment**, a way to train a reasoning model to think about the rules before it answers.

> [!PAPER] Guan et al. (2024), Deliberative Alignment: Reasoning Enables Safer Language Models · Abstract · page 1
> [![The abstract of the deliberative alignment paper, with directly teaches the model safety specifications and explicitly recall and accurately reason over the specifications before answering highlighted](/img/writings/history/h-delib.png)](/img/writings/history/h-delib.png)
>
> **What it says:** the method "directly teaches the model safety specifications" and trains it "to explicitly recall and accurately reason over the specifications before answering". OpenAI used it for its o-series models, "without requiring human-written chain-of-thoughts or answers".
>
> **What it means:**
>
> Classic RLHF teaches rules indirectly, through thousands of labelled examples, and the model has to guess the rule behind them. Here the rule itself is in the training, and the model learns to look it up in its own reasoning. The paper reports fewer successful jailbreaks and fewer needless refusals at the same time.

## 2023 to 2024: DPO and the family after it

### The derivation

RLHF with PPO is heavy. It keeps four large models in memory: the policy, the reference, the reward model and a value model for PPO's advantages. It needs fresh samples at every step, and it has many settings to tune. In May 2023, Rafael Rafailov, Archit Sharma, Eric Mitchell and colleagues at Stanford showed that, for preferences, the RL step can be removed.

Their starting point is a known result: the KL-penalised objective above has an exact best policy.

$$
\pi^{*}(y \mid x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x) \, \exp\!\Big( \frac{1}{\beta} \, r(x, y) \Big)
$$

Read it as "start from the reference model and tilt it toward high reward". $$\exp$$ is the exponential function, so each unit of reward multiplies an answer's probability by $$e^{1/\beta}$$. $$Z(x)$$ is the sum of the tilted probabilities over all answers, which makes them add up to 1 again. Nobody can compute $$Z(x)$$, because the number of possible answers is astronomical.

The trick is to solve this equation for the reward instead:

$$
r(x, y) = \beta \log \frac{\pi^{*}(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

Now put this reward into the Bradley-Terry model. Bradley-Terry uses only the difference between two rewards for the same prompt, so the two $$\beta \log Z(x)$$ terms cancel. The impossible sum disappears, and what is left depends only on the policy and the reference.

$$
\mathcal{L}_{\text{DPO}}(\theta) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

This is the reward-model loss, with the reward replaced by $$\beta \log (\pi_\theta / \pi_{\text{ref}})$$, the **implicit reward**. The language model is its own reward model. Training raises the probability of the chosen answer and lowers that of the rejected one, each measured against the reference.

> [!PAPER] Rafailov et al. (2023), Direct Preference Optimization: Your Language Model is Secretly a Reward Model · Figure 1 · page 2
> [![Figure 1 of the DPO paper: on the left, RLHF goes from preference data to a reward model by maximum likelihood, then to an LM policy by reinforcement learning with sampled completions; on the right, DPO goes from preference data straight to the final LM by maximum likelihood](/img/writings/history/h-dpo-fig.png)](/img/writings/history/h-dpo-fig.png)
>
> **What it says:** "DPO optimizes for human preferences while avoiding reinforcement learning". It uses "a simple classification objective, fitting an implicit reward model whose corresponding optimal policy can be extracted in closed form".
>
> **What it means:**
>
> Compare the two boxes. The left one has a loop: sample, score, update, sample again. The right one is a straight line from data to model, like ordinary fine-tuning. "Closed form" means the best policy for a given reward is a formula, the one above, so no search is needed to find it.

### A worked example

Take $$\beta = 0.1$$ and one pair. The policy gives the chosen answer a log-probability of $$-10$$, against $$-12$$ for the reference, so its implicit reward is $$0.1 \times 2 = 0.2$$. The rejected answer has $$-15$$ against $$-14$$, for an implicit reward of $$-0.1$$. The margin is 0.3, the sigmoid gives 0.5744, and the loss is 0.5544.

At the very start, the policy equals the reference, every margin is 0, and the loss is $$\ln 2 = 0.6931$$. The gradient of each pair is weighted by $$\sigma(-\text{margin})$$, which starts at 0.5. In our example it has already fallen to 0.4256. Pairs the model already gets right count less, and pairs it gets wrong count more.

<figure class="fig"><svg viewBox="0 0 720 296" role="img" aria-label="Left: PPO-based RLHF holds a policy, a reference, a reward model and a value model. Right: DPO holds only a policy and a reference."><text class="t-title" x="10" y="22">What each method keeps in memory during training</text><text class="t-note" x="180" y="50" text-anchor="middle">RLHF with PPO: four models</text><text class="t-note" x="540" y="50" text-anchor="middle">DPO: two models</text><rect class="box-on" x="30" y="66" width="140" height="54" rx="6"/><text class="t-note" x="100" y="89" text-anchor="middle">policy</text><text class="t-muted" x="100" y="106" text-anchor="middle">trained</text><rect class="box" x="190" y="66" width="140" height="54" rx="6"/><text class="t-note" x="260" y="89" text-anchor="middle">reference</text><text class="t-muted" x="260" y="106" text-anchor="middle">frozen</text><rect class="box-3" x="30" y="136" width="140" height="54" rx="6"/><text class="t-note" x="100" y="159" text-anchor="middle">reward model</text><text class="t-muted" x="100" y="176" text-anchor="middle">frozen</text><rect class="box-on" x="190" y="136" width="140" height="54" rx="6"/><text class="t-note" x="260" y="159" text-anchor="middle">value model</text><text class="t-muted" x="260" y="176" text-anchor="middle">trained</text><rect class="box-on" x="400" y="66" width="140" height="54" rx="6"/><text class="t-note" x="470" y="89" text-anchor="middle">policy</text><text class="t-muted" x="470" y="106" text-anchor="middle">trained</text><rect class="box" x="560" y="66" width="140" height="54" rx="6"/><text class="t-note" x="630" y="89" text-anchor="middle">reference</text><text class="t-muted" x="630" y="106" text-anchor="middle">frozen</text><line class="edge-dim" x1="360" y1="40" x2="360" y2="290"/><text class="t-muted" x="30" y="220">Loop: sample answers, score them,</text><text class="t-muted" x="30" y="238">estimate advantages, run PPO steps,</text><text class="t-muted" x="30" y="256">sample again. Online and fiddly.</text><text class="t-muted" x="400" y="150">Loop: read a fixed pair (chosen, rejected),</text><text class="t-muted" x="400" y="168">compute four log-probabilities,</text><text class="t-muted" x="400" y="186">one classification-style loss. Offline.</text><rect class="box-2" x="400" y="210" width="300" height="60" rx="6"/><text class="t-note" x="550" y="236" text-anchor="middle">loss = −log σ( β·[Δ chosen] − β·[Δ rejected] )</text><text class="t-muted" x="550" y="253" text-anchor="middle">Δ = log π − log π_ref of the answer</text></svg><figcaption>DPO drops the reward model and the value model and needs no sampling during training. The price: it learns only from the pairs in its data set.</figcaption></figure>

DPO spread fast because it is cheap and stable. Hugging Face's Zephyr models used it in late 2023. Meta's Llama 3 paper of July 2024 describes post-training with SFT, rejection sampling and DPO, "as opposed to more complex reinforcement learning algorithms". Its weakness is that it learns only from the pairs in its fixed data set, while PPO keeps sampling new answers from the current model.

### The family after DPO

DPO opened a family of "offline" preference methods. **IPO** (Mohammad Gheshlaghi Azar and colleagues at Google DeepMind, October 2023) argued that both RLHF and DPO can overfit. The reason is the Bradley-Terry assumption that every preference becomes a point score, and IPO replaces the log-sigmoid with a squared loss on the margin. **KTO** (Kawin Ethayarajh and colleagues, February 2024) borrowed prospect theory from economics, which says people feel losses more than gains. It learns from single answers marked good or bad, so it needs no pairs.

**ORPO** (Jiwoo Hong and colleagues, March 2024) folded preference learning into SFT itself and needs no reference model. **SimPO** (Yu Meng and colleagues, May 2024) also drops the reference model. Its implicit reward is the average log-probability per token of an answer. It also adds a target margin by which the winner must beat the loser. Taking the average per token keeps the length of an answer from tilting its reward.

## Rewarding the steps: verifiers and process rewards

### Train a checker, keep the best

A preference reward model judges style and helpfulness. For a maths problem, there is a sharper question: is the answer right? In October 2021, Karl Cobbe and colleagues at OpenAI released GSM8K, 8,500 grade-school maths word problems. They trained **verifiers** to judge whether a solution is correct. At test time they sampled many solutions and kept the one the verifier ranked highest, which is best-of-n with a correctness judge.

> [!DEFINITION] Verifier
> A model or a program that judges whether an answer is correct. A program verifier (an answer checker, a unit test, a proof checker) is exact. A learned verifier is a reward model trained on correct and incorrect solutions.

### STaR: learn from your own right answers

In March 2022, Eric Zelikman and colleagues published STaR, the Self-Taught Reasoner. The model writes a chain of thought and an answer for many problems. The chains that reach the right answer are kept, and the model is fine-tuned on them; then the loop repeats. For problems it fails, the model gets the right answer and writes a reasoning that leads to it. The authors call this step "rationalization". STaR is a filtered form of reinforcement learning: a reward of 1 for a correct answer, and training only on the rewarded samples.

### Outcome or process?

A wrong final answer tells you that something went wrong, not where. A teacher who marks every line of a pupil's work gives far more help. In May 2023, Hunter Lightman and colleagues at OpenAI compared two kinds of reward model on hard maths problems.

> [!DEFINITION] Outcome and process reward models
> An **outcome reward model** (ORM) scores only the final answer of a solution. A **process reward model** (PRM) scores each intermediate step, so it can say where a solution first goes wrong.

<figure class="fig"><svg viewBox="0 0 720 272" role="img" aria-label="A four-step solution of 17 times 24. A process reward model marks steps 1 and 2 correct and step 3 wrong. An outcome reward model only sees that the final answer 398 is wrong."><text class="t-title" x="10" y="22">Outcome reward vs process reward for one solution</text><rect class="box" x="20" y="48" width="330" height="38" rx="6"/><text class="t-mono" x="34" y="72">Step 1: 17 × 24 = 17 × 20 + 17 × 4</text><text class="t-on" x="470" y="72" text-anchor="middle">✓ correct</text><rect class="box" x="20" y="94" width="330" height="38" rx="6"/><text class="t-mono" x="34" y="118">Step 2: 17 × 20 = 340</text><text class="t-on" x="470" y="118" text-anchor="middle">✓ correct</text><rect class="box-4" x="20" y="140" width="330" height="38" rx="6"/><text class="t-mono" x="34" y="164">Step 3: 17 × 4 = 58</text><text class="t-bad" x="470" y="164" text-anchor="middle">✗ wrong</text><rect class="box-4" x="20" y="186" width="330" height="38" rx="6"/><text class="t-mono" x="34" y="210">Step 4: 340 + 58 = 398</text><text class="t-bad" x="470" y="210" text-anchor="middle">✗ follows the error</text><text class="t-note" x="470" y="40" text-anchor="middle">process reward model</text><text class="t-note" x="640" y="40" text-anchor="middle">outcome reward</text><rect class="box-4" x="590" y="48" width="100" height="176" rx="6"/><text class="t-note" x="640" y="115" text-anchor="middle">answer 398</text><text class="t-muted" x="640" y="132" text-anchor="middle">is wrong:</text><text class="t-muted" x="640" y="149" text-anchor="middle">reward 0</text><text class="t-muted" x="640" y="166" text-anchor="middle">for all</text><text class="t-muted" x="20" y="254">The outcome label says only "wrong". The process labels say where: step 3. That is a far more useful signal.</text></svg><figcaption>The correct answer is 408. Lightman and colleagues paid people to label about 800,000 steps like these (PRM800K).</figcaption></figure>

> [!PAPER] Lightman et al. (2023), Let's Verify Step by Step · Abstract · page 1
> [![The abstract of Let's Verify Step by Step, with process supervision significantly outperforms outcome supervision and 78% of problems highlighted](/img/writings/history/h-lightman.png)](/img/writings/history/h-lightman.png)
>
> **What it says:** "process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset". The process-supervised model "solves 78% of problems from a representative subset of the MATH test set". The authors release "PRM800K", "800,000 step-level human feedback labels".
>
> **What it means:**
>
> Here the reward models were used to pick the best of many sampled solutions, not yet to train with RL. Step labels are expensive, which is why the data set itself was a contribution. Later work, as the next sections show, found that simple outcome rewards plus a lot of RL can go very far too.

## September 2024: o1 and thinking at test time

On 12 September 2024, OpenAI released **o1** and a post called "Learning to reason with LLMs". It said that o1 is trained with reinforcement learning to think, in a long chain of thought, before it answers. It also said that the performance of o1 "consistently improves with more reinforcement learning (train-time compute) and with more time spent thinking (test-time compute)".

> [!DEFINITION] Test-time compute
> The computation a model spends while answering, as opposed to during training. A reasoning model spends more of it by writing a longer chain of thought, or by sampling several answers and choosing among them.

The post gave numbers for the 2024 AIME, a hard American maths competition. GPT-4o solved 12% of the problems on average. o1 solved 74% with one sample per problem and 83% with a consensus of 64 samples. A consensus answer is the most common final answer across the samples.

> [!DEFINITION] pass@1 and consensus
> **pass@1** is the share of problems solved by a single sampled answer, averaged over samples. **cons@64** (consensus, or majority vote) samples 64 answers. It counts the problem as solved if the most common final answer is right.

OpenAI did not publish how o1 was trained: not the algorithm, the rewards or the data. The post showed accuracy rising along two axes, training compute and thinking time, and that was enough to change the field. Within months, several labs published their own versions, with far more detail.

## Rewards a program can check: RLVR

In November 2024, Nathan Lambert and colleagues at the Allen Institute for AI published **Tülu 3**, a fully open post-training recipe. Its last stage gave a name to an idea that was spreading: reinforcement learning with verifiable rewards.

> [!DEFINITION] RLVR
> Reinforcement learning with verifiable rewards. The reward comes from a program that checks the answer, not from a learned reward model. Examples: does the final number match, do the unit tests pass, does the output follow the requested format.

> [!PAPER] Lambert et al. (2024), Tülu 3: Pushing Frontiers in Open Language Model Post-Training · Figure 18 · page 31
> [![Figure 18 of the Tulu 3 paper: training prompts go to a policy, its completions go to a verifiable reward that is gamma if correct and 0 otherwise, and the scalar reward updates the policy](/img/writings/history/h-tulu3-rlvr.png)](/img/writings/history/h-tulu3-rlvr.png)
>
> **What it says:** "an overview of how Reinforcement Learning with Verifiable Rewards (RLVR) works". The policy samples completions for a set of prompts, a deterministic function checks them, and the model gets a fixed reward if the answer "is verifiably correct, otherwise 0". Training then uses PPO.
>
> **What it means:**
>
> Compare it with the RLHF figure: the reward model is gone, replaced by a checker. A checker cannot be flattered and has no blind spots for the policy to find. The cost is that it only works where correctness can be checked: maths, code, puzzles, strict formats. Tülu 3 ran it after SFT and DPO.

## GRPO: dropping the critic

### The idea

PPO needs a value model to estimate advantages, and for a language model the value model is as big as the policy. In February 2024, Zhihong Shao and colleagues at DeepSeek proposed a cheaper baseline in their DeepSeekMath paper. For each question, sample a group of answers, and compare each answer with the group's average. It is like grading on a curve: a mark means something only next to the marks of the rest of the class.

> [!DEFINITION] GRPO
> Group Relative Policy Optimisation. For each prompt, it samples a group of answers and scores each one. The advantage of an answer is its reward minus the group mean, divided by the group's standard deviation. It needs no value model.

> [!PAPER] Shao et al. (2024), DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models · Figure 4 · page 13
> [![Figure 4 of DeepSeekMath: PPO passes a question through the policy, a reference, a reward model and a trained value model into GAE; GRPO samples G outputs, scores them with the reward model and computes group advantages, with no value model](/img/writings/history/h-grpo-fig.png)](/img/writings/history/h-grpo-fig.png)
>
> **What it says:** "GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources".
>
> **What it means:**
>
> In the top row, the yellow "Value Model" is trained alongside the policy, and its output feeds GAE. In the bottom row it is gone. The policy writes $$G$$ outputs $$o_1$$ to $$o_G$$, the reward model scores them, and "Group Computation" turns the scores into advantages. The KL term moves from the reward into the loss.

### The equations

The advantage of answer $$i$$ in a group of $$G$$ answers is its reward, centred and scaled by the group:

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}
$$

Every token of answer $$i$$ gets this same advantage. The objective is PPO's clipped objective, averaged over the tokens of each answer and over the group, with the KL penalty inside:

$$
J_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Big( \min\big( \rho_{i,t} \hat{A}_i,\; \text{clip}(\rho_{i,t}, 1 - \epsilon, 1 + \epsilon)\, \hat{A}_i \big) - \beta\, \mathbb{D}_{\text{KL}}\big[ \pi_\theta \,\Vert\, \pi_{\text{ref}} \big] \Big) \right]
$$

$$|o_i|$$ is the number of tokens in answer $$i$$. $$\rho_{i,t}$$ (rho) is the probability ratio of token $$t$$ under the new and the old policy, as in PPO: $$\pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_\text{old}}(o_{i,t} \mid q, o_{i,<t})$$. $$q$$ is the question and $$o_{i,<t}$$ the tokens before $$t$$. $$\epsilon$$ is the clip range and $$\beta$$ the KL weight.

<figure class="fig"><svg viewBox="0 0 720 330" role="img" aria-label="A question goes to the policy, which samples eight answers. Three are right with reward 1 and five are wrong with reward 0. Advantages are plus 1.291 for the right ones and minus 0.775 for the wrong ones."><text class="t-title" x="10" y="22">GRPO: sample a group, score it, compare each answer with the group</text><rect class="box-2" x="20" y="120" width="130" height="70" rx="6"/><text class="t-note" x="85" y="151" text-anchor="middle">one maths</text><text class="t-muted" x="85" y="168" text-anchor="middle">question</text><rect class="box-on" x="190" y="120" width="120" height="70" rx="6"/><text class="t-note" x="250" y="151" text-anchor="middle">policy</text><text class="t-muted" x="250" y="168" text-anchor="middle">samples 8</text><polyline class="edge" points="150,155 190,155"/><polygon class="arrow" points="190.0,155.0 182.0,159.0 182.0,151.0"/><rect class="box-1" x="350" y="40" width="110" height="22" rx="4"/><text class="t-muted" x="405" y="56" text-anchor="middle">answer 1: right</text><text class="t-note" x="490" y="56" text-anchor="middle">r = 1</text><text class="t-s1" x="600" y="56" text-anchor="middle">A = +1.291</text><line class="edge-dim" x1="310" y1="155" x2="350" y2="51"/><rect class="box-4" x="350" y="68" width="110" height="22" rx="4"/><text class="t-muted" x="405" y="84" text-anchor="middle">answer 2: wrong</text><text class="t-note" x="490" y="84" text-anchor="middle">r = 0</text><text class="t-bad" x="600" y="84" text-anchor="middle">A = −0.775</text><line class="edge-dim" x1="310" y1="155" x2="350" y2="79"/><rect class="box-4" x="350" y="96" width="110" height="22" rx="4"/><text class="t-muted" x="405" y="112" text-anchor="middle">answer 3: wrong</text><text class="t-note" x="490" y="112" text-anchor="middle">r = 0</text><text class="t-bad" x="600" y="112" text-anchor="middle">A = −0.775</text><line class="edge-dim" x1="310" y1="155" x2="350" y2="107"/><rect class="box-1" x="350" y="124" width="110" height="22" rx="4"/><text class="t-muted" x="405" y="140" text-anchor="middle">answer 4: right</text><text class="t-note" x="490" y="140" text-anchor="middle">r = 1</text><text class="t-s1" x="600" y="140" text-anchor="middle">A = +1.291</text><line class="edge-dim" x1="310" y1="155" x2="350" y2="135"/><rect class="box-1" x="350" y="152" width="110" height="22" rx="4"/><text class="t-muted" x="405" y="168" text-anchor="middle">answer 5: right</text><text class="t-note" x="490" y="168" text-anchor="middle">r = 1</text><text class="t-s1" x="600" y="168" text-anchor="middle">A = +1.291</text><line class="edge-dim" x1="310" y1="155" x2="350" y2="163"/><rect class="box-4" x="350" y="180" width="110" height="22" rx="4"/><text class="t-muted" x="405" y="196" text-anchor="middle">answer 6: wrong</text><text class="t-note" x="490" y="196" text-anchor="middle">r = 0</text><text class="t-bad" x="600" y="196" text-anchor="middle">A = −0.775</text><line class="edge-dim" x1="310" y1="155" x2="350" y2="191"/><rect class="box-4" x="350" y="208" width="110" height="22" rx="4"/><text class="t-muted" x="405" y="224" text-anchor="middle">answer 7: wrong</text><text class="t-note" x="490" y="224" text-anchor="middle">r = 0</text><text class="t-bad" x="600" y="224" text-anchor="middle">A = −0.775</text><line class="edge-dim" x1="310" y1="155" x2="350" y2="219"/><rect class="box-4" x="350" y="236" width="110" height="22" rx="4"/><text class="t-muted" x="405" y="252" text-anchor="middle">answer 8: wrong</text><text class="t-note" x="490" y="252" text-anchor="middle">r = 0</text><text class="t-bad" x="600" y="252" text-anchor="middle">A = −0.775</text><line class="edge-dim" x1="310" y1="155" x2="350" y2="247"/><text class="t-note" x="350" y="284">mean = 0.375, std = 0.484, A = (r − mean) / std</text><text class="t-muted" x="20" y="312">No value network: the group mean is the baseline. Right answers are pushed up, wrong ones down, then a PPO-style clipped step.</text></svg><figcaption>Our script's group of eight. If all eight answers were right (or all wrong), every advantage would be 0, and the question would teach nothing.</figcaption></figure>

Take one question and eight answers, three right and five wrong. The rewards are 1, 0, 0, 1, 1, 0, 0, 0, with mean 0.375 and standard deviation 0.4841. The right answers get an advantage of $$+1.291$$ and the wrong ones $$-0.775$$. For an easy question with seven right answers, the one wrong answer gets $$-2.646$$, and the right ones only $$+0.378$$. If all eight are right, the standard deviation is 0 and every advantage is 0: the question teaches nothing.

With GRPO, DeepSeekMath 7B reached 51.7% on the MATH benchmark, the paper reports, without tools or voting. The method was cheap enough to run at scale, and a year later it was the engine of DeepSeek-R1.

## January 2025: DeepSeek-R1

### R1-Zero: reinforcement learning alone

On 20 January 2025, DeepSeek released DeepSeek-R1, with open weights and a paper; a longer version appeared in *Nature* on 17 September 2025. Its most striking model was **R1-Zero**. It started from the base model DeepSeek-V3-Base and went straight to RL with GRPO, with no SFT first. The rewards were rule-based: an accuracy reward for a correct final answer, and a format reward for putting the reasoning between think tags.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning · Figure 1 · page 4
> [![Figure 1 of the DeepSeek-R1 paper: on the left, AIME accuracy of R1-Zero rising over 10,000 training steps past the human average; on the right, the average response length rising from a few hundred tokens to more than 12,500](/img/writings/history/h-r1-fig.png)](/img/writings/history/h-r1-fig.png)
>
> **What it says:** panel (a) is the "AIME accuracy of DeepSeek-R1-Zero during training", and panel (b) is "the average response length" on the training set. "DeepSeek-R1-Zero naturally learns to solve reasoning tasks with more thinking time."
>
> **What it means:**
>
> Nobody told the model to think longer. The reward only checked the final answer. The model found that longer reasoning got more answers right, so its answers grew from a few hundred tokens to more than ten thousand. The green dashed line is the average score of human AIME participants; R1-Zero passed it early in training. The text reports pass@1 rising from 15.6% to 77.9%, and 86.7% with self-consistency.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1 · Table 2 · page 5
> [![Table 2 of the DeepSeek-R1 paper: a maths question about square roots, the model's reasoning, then the line Wait, wait. Wait. That's an aha moment I can flag here, and a restart of the reasoning](/img/writings/history/h-r1-aha.png)](/img/writings/history/h-r1-aha.png)
>
> **What it says:** "an interesting 'aha moment' of an intermediate version of DeepSeek-R1-Zero". "The model learns to rethink using an anthropomorphic tone."
>
> **What it means:**
>
> The model squares both sides, gets a quartic equation, and then stops itself. It goes back to the original equation and tries again, more carefully. Checking and backtracking were never shown to it as examples; they appeared because they earned reward. Later work, discussed below, found that base models already show such moments now and then, and RL makes them common.

R1-Zero had problems. Its reasoning was hard to read, and it often mixed English and Chinese in one chain of thought. So DeepSeek-R1 used a fuller pipeline. A small set of long, readable reasoning examples came first (a "cold start"), then reasoning RL. Then came SFT on a large set of the model's own filtered answers and general data, and a final RL stage for helpfulness and safety.

The team also **distilled** R1 into smaller open models based on Qwen and Llama by fine-tuning them on R1's outputs. The paper's appendix lists the RL training cost of R1 as 147,000 H800 GPU hours. At 2 dollars per hour, that is about 294,000 US dollars. That does not include the much larger cost of pre-training the base model.

> [!DEFINITION] Distillation
> Training a smaller "student" model on the outputs of a larger "teacher" model. For reasoning, the student is fine-tuned on the teacher's chains of thought.

The paper is also open about reward hacking. For reasoning, DeepSeek used no neural reward models at all, because it found them "susceptible to reward hacking" at scale. In the last stage, a model-based preference reward was used for general tasks. One figure shows that reward rising while performance on CodeForces falls: Goodhart's law again.

## 2025: fixing GRPO and scaling the recipe

The R1 paper started a rush of open work. In the same week, Moonshot AI's **Kimi k1.5** report described long-context RL with a simple framework, "without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models". Both reports said that plain outcome rewards and a lot of RL were enough.

Then came the fixes. In March 2025, ByteDance Seed and Tsinghua University released **DAPO**, with four changes to GRPO. Clip-Higher raises the upper clip limit, so rare tokens that turn out well can grow faster. That keeps the policy from collapsing onto a few answers. Dynamic Sampling drops prompts whose group is all right or all wrong, since they give zero advantage. A token-level loss and a shaped penalty for overlong answers complete the set.

Also in March 2025, Zichen Liu and colleagues at Sea AI Lab published "Understanding R1-Zero-Like Training". They found an "optimization bias" in GRPO that "artificially increases response length (especially for incorrect outputs)". Dividing by the answer length $$|o_i|$$ makes each token of a long wrong answer count for less, so long wrong answers are punished too gently. Their **Dr. GRPO** drops that division and the division by the standard deviation.

For our group of eight, Dr. GRPO's advantages are simply $$r_i - 0.375$$: $$+0.625$$ for a right answer and $$-0.375$$ for a wrong one. Without the standard deviation, an easy question no longer gives its one wrong answer an outsized push. The same paper found that DeepSeek-V3-Base "already exhibit[s]" an aha moment before RL, so RL amplifies a behaviour that pre-training made possible.

In May 2025, Alibaba's **Qwen3** report described one model with both a "thinking" and a "non-thinking" mode. The user can also set a thinking budget.

> [!PAPER] Qwen Team (2025), Qwen3 Technical Report · Figure 1 · page 9
> [![Figure 1 of the Qwen3 report: flagship models go from base models through stage 1, long-CoT cold start, stage 2, reasoning RL, stage 3, thinking mode fusion, and stage 4, general RL; lightweight models are made by strong-to-weak distillation from the flagship models](/img/writings/history/h-qwen3-flow.png)](/img/writings/history/h-qwen3-flow.png)
>
> **What it says:** the "post-training pipeline of the Qwen3 series models". Flagship models pass through four stages; lightweight models come from "Strong-to-Weak Distillation".
>
> **What it means:**
>
> This is the R1 recipe, made into a standard. Stages 1 and 2 teach long reasoning. Stage 3 mixes in data for quick, non-thinking answers, so one model can do both. Stage 4 is RL across general tasks. The bottom row is how most small models are made in 2026: they learn from a bigger model's outputs, not from RL of their own.

By the end of 2025, RL was no longer a small final polish. In its December 2025 report, DeepSeek wrote that the post-training of DeepSeek-V3.2 used a large compute budget.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models · §1 · page 2
> [![A passage from the DeepSeek-V3.2 report, with post-training computational budget exceeding 10% of the pre-training cost highlighted](/img/writings/history/h-dsv32.png)](/img/writings/history/h-dsv32.png)
>
> **What it says:** the RL framework "allocates a post-training computational budget exceeding 10% of the pre-training cost, unlocking advanced capabilities".
>
> **What it means:**
>
> For InstructGPT, the RL stage was a tiny fraction of the pre-training cost. More than 10% means RL is now a major training phase in its own right. The same report says its results were still improving as the RL budget grew.

## What Google says about Gemini

Google publishes less than DeepSeek, but its 2025 technical report on Gemini 2.5 gives an outline of post-training and of the "thinking" models.

> [!PAPER] Gemini Team, Google (2025), Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities · §2.4 and §2.5 · page 5
> [![Sections 2.4 and 2.5 of the Gemini 2.5 report, with the SFT, reward modeling and RL stages, increased the training compute allocated to RL, verifiable rewards, model-based generative rewards, multi-step actions and tool use, and Gemini Thinking models are trained with Reinforcement Learning highlighted](/img/writings/history/h-gemini-post.png)](/img/writings/history/h-gemini-post.png)
>
> **What it says:** post-training has "Supervised Fine-Tuning (SFT), Reward Modeling (RM), and Reinforcement Learning (RL) stages". Google "increased the training compute allocated to RL", with "a focus on verifiable rewards and model-based generative rewards". RL environments include "those requiring multi-step actions and tool use". "Gemini Thinking models are trained with Reinforcement Learning to use additional compute at inference time".
>
> **What it means:**
>
> Every piece of this part is in that paragraph: SFT, a reward model, RL, verifiable rewards and agent environments. "Model-based generative rewards" most likely means a language model that writes a judgement, an AI judge in place of a one-number reward head. That is our reading; the report gives no details. It also gives no algorithm, data size or reward design.

The same report describes **Deep Think**, a mode of Gemini 2.5 Pro that "naturally blends in parallel thinking techniques during response generation". It lets the model "produce multiple hypotheses and carefully critique them before arriving at the final answer". On 21 July 2025, Google DeepMind announced that an advanced version of Gemini with Deep Think had reached gold-medal standard at the International Mathematical Olympiad.

It solved five of the six problems for 35 of 42 points, within the 4.5-hour time limit. It worked in natural language from the official problem statements, and IMO coordinators graded and certified the solutions. A year earlier, AlphaProof and AlphaGeometry 2 had scored 28 points, but only after the problems were translated into a formal language. The post names new reinforcement learning methods, data on multi-step reasoning and theorem proving, and parallel thinking. OpenAI reported the same score of 35 points for an experimental model of its own.

## Sycophancy: when the reward is a thumbs-up

> [!DEFINITION] Sycophancy
> When a model tells users what they want to hear in place of what is true or useful. Examples: agreeing with a wrong claim, praising a weak idea, or dropping a right answer when the user pushes back.

Preference learning has a built-in danger: people like to be agreed with. In October 2023, Mrinank Sharma and colleagues at Anthropic studied five assistants trained with human feedback and found sycophancy in all of them. Their analysis of preference data showed why. Both humans and preference models sometimes preferred convincing, sycophantic answers over correct ones.

In April 2025, this happened in a widely used product. On 25 April, OpenAI shipped an update to GPT-4o, the default model of its chat app. Users soon saw it flattering them and endorsing poor decisions. OpenAI began rolling it back on 28 April. In its account of 2 May, OpenAI said the update had added a reward signal based on users' thumbs-up and thumbs-down ratings. Together with other changes, this weakened the main reward signal that had kept sycophancy in check.

The lesson is Goodhart's law again, at the scale of millions of users. A thumbs-up measures what pleases a person in the moment. Optimise it hard enough and it stops measuring what helps them. OpenAI said it would weigh behaviour problems like this more heavily in its review before each launch.

## RL for agents

A chat answer is one step. An agent works for many steps: it reads files, runs commands, reads the results and tries again. Training it with RL means building **environments**: a computer with a code base, a task, tools and a checker at the end.

> [!DEFINITION] RL environment (for language models)
> A sandbox where a model acts through tools, such as a shell, a code editor or a browser. It gets a reward at the end of a task; for coding, often whether the project's tests pass.

Coding is the clearest case because the reward is checkable. SWE-bench, from Carlos Jimenez and colleagues at Princeton in October 2023, has 2,294 real GitHub issues from 12 Python projects. A model must edit the code base so that the project's tests pass. In the paper, the best model, Claude 2, solved 1.96% of the issues. In August 2024, OpenAI released SWE-bench Verified, a subset of 500 issues that people had checked were solvable.

Labs now build such environments at scale. Moonshot AI's Kimi K2 report of July 2025 describes "a large-scale agentic data synthesis pipeline and a joint reinforcement learning (RL) stage". There, the model "improves its capabilities through interactions with real and synthetic environments". For tasks with no checker, it adds a "self-critique rubric reward": the model judges answers against written rubrics. DeepSeek-V3.2 describes a "large-scale agentic task synthesis pipeline" for tool use. And the Gemini 2.5 report names RL environments with "multi-step actions and tool use".

The hard part is the reward. A test suite can be fooled by code that special-cases the tests, and a judge model can be fooled by confident text. Much of the engineering of agent RL is making the checker hard to game.

## How the models are tested

Training needs a reward, and progress needs a test. The tests themselves have changed as fast as the models.

### From exams to harder exams

In September 2020, Dan Hendrycks and colleagues released **MMLU**, multiple-choice questions on 57 subjects, from law to physics. It became the standard test of knowledge. In the DeepSeek-R1 paper of 2025, R1 scored 90.8% and OpenAI's o1 91.8%. A re-check of the questions, MMLU-Redux, found errors in some of them. A test that every model passes can no longer tell models apart: it is **saturated**.

> [!DEFINITION] Benchmark
> A fixed set of tasks with a scoring rule, used to compare models. A benchmark is **saturated** when the best models score close to its maximum, so it no longer separates them.

In November 2023, David Rein and colleagues released **GPQA**, 448 multiple-choice questions in biology, physics and chemistry written by domain experts. People with or studying for PhDs in the field scored 65%. Skilled non-experts scored 34%, even with over 30 minutes and free use of the web. That is why it is called "Google-proof". In January 2025, the Center for AI Safety and Scale AI released **Humanity's Last Exam**. Experts worldwide wrote its 2,500 questions at the edge of human knowledge.

For code, the field moved from writing one function to fixing real projects: from HumanEval in 2021 to SWE-bench. Code tests usually report **pass@k**, the chance that at least one of $$k$$ sampled answers is right. Mark Chen and colleagues gave an unbiased way to estimate it from $$n$$ samples, of which $$c$$ are right:

$$
\text{pass@}k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}
$$

$$\binom{n}{k}$$ is the number of ways to choose $$k$$ samples out of $$n$$. So the fraction is the chance that a random choice of $$k$$ samples contains no right one. With 10 samples of which 3 are right, pass@1 is $$1 - 7/10 = 0.3$$. pass@5 is $$1 - 21/252 = 0.9167$$. A reasoning model that is reported with pass@1 is being judged on its first try.

### Tests of a different kind

In 2019, François Chollet published "On the Measure of Intelligence" and the **ARC** benchmark. Its small grid puzzles are easy for people, and each one needs a new rule. The ARC Prize reported that GPT-4o scored 5% on the original version in 2024. On 20 December 2024, it reported that OpenAI's o3 scored 75.7% on its semi-private test set, and 87.5% with about 172 times more compute. Harder versions of ARC followed, built to stay hard for machines.

### The arena and AI judges

Fixed tests miss what users care about, such as tone, clarity and usefulness. In May 2023, a team at Berkeley's LMSYS opened **Chatbot Arena**, now called LMArena. A user asks anything, sees two anonymous answers, and votes for the better one. The 2024 paper reports over 240,000 votes at the time.

The ranking uses the Bradley-Terry model, the same equation as the reward model at the start of this part. If two models' scores differ by 1, the one with the higher score is preferred with probability $$\sigma(1) = 0.7311$$. The arena is a reward model for whole models, fitted to the votes of the public.

Arena rankings have their own Goodhart problems. In April 2025, "The Leaderboard Illusion", from Cohere Labs and university researchers, reported "undisclosed private testing practices". These let a few providers test many variants before release and publish only the best score. Arena votes also reward style: length, formatting and confident tone.

Human votes are slow, so many tests use a strong model as the judge. In June 2023, Lianmin Zheng and colleagues introduced MT-Bench and found that GPT-4 as a judge reached "over 80% agreement" with human preferences. That is the same level as between humans. But AI judges have biases too, for example toward longer answers and toward answers in a later position.

> [!DEFINITION] LLM-as-a-judge
> Using a strong language model to grade or compare other models' answers, in place of human raters. It is fast and cheap, but it inherits the judge model's biases.

### Contamination

> [!DEFINITION] Contamination
> When test questions, or their answers, appear in a model's training data. The model may then recall the answers instead of solving the problems, and its score overstates its ability.

The web is full of copies of popular benchmarks, and pre-training reads the web. The Olmo 3 report warns that results can be distorted when "mid-training data contain[s] the evaluation". The usual defences are fresh tests built after a model's training cut-off, such as new competition problems. Others are private held-out sets and searching the training data for test questions. None is perfect.

## October 2026: the stages of a frontier model

By 2026, training a frontier language model is a long pipeline. No lab outside the open-model world publishes its full recipe. Anthropic, OpenAI and Google describe their stages only in outline, in model cards and reports. The figure is a composite of what the open reports describe and what the closed labs confirm in outline.

<figure class="fig"><svg viewBox="0 0 720 390" role="img" aria-label="Eight stages in two rows: pre-training, mid-training, supervised fine-tuning, preference tuning, reasoning RL, agent RL, safety training and distillation, with evaluation running under all of them."><text class="t-title" x="10" y="22">How a frontier language model is trained, as far as public reports show</text><rect class="box-1" x="20" y="50" width="155" height="86" rx="6"/><text class="t-note" x="97.5" y="80.5" text-anchor="middle">Pre-training</text><text class="t-muted" x="97.5" y="97.5" text-anchor="middle">trillions of tokens</text><text class="t-muted" x="97.5" y="114.5" text-anchor="middle">next-token loss</text><polyline class="edge" points="175,93 195,93"/><polygon class="arrow" points="195.0,93.0 187.0,97.0 187.0,89.0"/><rect class="box-1" x="195" y="50" width="155" height="86" rx="6"/><text class="t-note" x="272.5" y="80.5" text-anchor="middle">Mid-training</text><text class="t-muted" x="272.5" y="97.5" text-anchor="middle">curated maths, code,</text><text class="t-muted" x="272.5" y="114.5" text-anchor="middle">reasoning, long context</text><polyline class="edge" points="350,93 370,93"/><polygon class="arrow" points="370.0,93.0 362.0,97.0 362.0,89.0"/><rect class="box-2" x="370" y="50" width="155" height="86" rx="6"/><text class="t-note" x="447.5" y="80.5" text-anchor="middle">SFT</text><text class="t-muted" x="447.5" y="97.5" text-anchor="middle">instructions, chats,</text><text class="t-muted" x="447.5" y="114.5" text-anchor="middle">reasoning traces, tools</text><polyline class="edge" points="525,93 545,93"/><polygon class="arrow" points="545.0,93.0 537.0,97.0 537.0,89.0"/><rect class="box-2" x="545" y="50" width="155" height="86" rx="6"/><text class="t-note" x="622.5" y="80.5" text-anchor="middle">Preference tuning</text><text class="t-muted" x="622.5" y="97.5" text-anchor="middle">reward models, DPO,</text><text class="t-muted" x="622.5" y="114.5" text-anchor="middle">AI feedback</text><rect class="box-3" x="20" y="180" width="155" height="86" rx="6"/><text class="t-note" x="97.5" y="210.5" text-anchor="middle">Reasoning RL</text><text class="t-muted" x="97.5" y="227.5" text-anchor="middle">verifiable rewards:</text><text class="t-muted" x="97.5" y="244.5" text-anchor="middle">tests, checkers</text><polyline class="edge" points="175,223 195,223"/><polygon class="arrow" points="195.0,223.0 187.0,227.0 187.0,219.0"/><rect class="box-3" x="195" y="180" width="155" height="86" rx="6"/><text class="t-note" x="272.5" y="210.5" text-anchor="middle">Agent RL</text><text class="t-muted" x="272.5" y="227.5" text-anchor="middle">multi-step tasks</text><text class="t-muted" x="272.5" y="244.5" text-anchor="middle">in tool environments</text><polyline class="edge" points="350,223 370,223"/><polygon class="arrow" points="370.0,223.0 362.0,227.0 362.0,219.0"/><rect class="box-4" x="370" y="180" width="155" height="86" rx="6"/><text class="t-note" x="447.5" y="210.5" text-anchor="middle">Safety training</text><text class="t-muted" x="447.5" y="227.5" text-anchor="middle">specs, constitutions,</text><text class="t-muted" x="447.5" y="244.5" text-anchor="middle">red-team data</text><polyline class="edge" points="525,223 545,223"/><polygon class="arrow" points="545.0,223.0 537.0,227.0 537.0,219.0"/><rect class="box-4" x="545" y="180" width="155" height="86" rx="6"/><text class="t-note" x="622.5" y="210.5" text-anchor="middle">Distillation</text><text class="t-muted" x="622.5" y="227.5" text-anchor="middle">small models learn</text><text class="t-muted" x="622.5" y="244.5" text-anchor="middle">from the big one</text><polyline class="edge" points="637,136 637,158 97,158 97,180"/><polygon class="arrow" points="97.0,180.0 93.0,172.0 101.0,172.0"/><rect class="box-ghost" x="20" y="300" width="680" height="44" rx="6"/><text class="t-note" x="360" y="327" text-anchor="middle">Evaluation at every stage: benchmarks, held-out tests, human and model judges, safety tests</text><text class="t-muted" x="20" y="372">Stages 3 to 7 are often repeated in rounds, and their order differs between labs. Few labs publish their full recipe.</text></svg><figcaption>A composite drawn from the reports of Llama 3, DeepSeek-R1 and V3.2, Qwen3, Olmo 3, Kimi K2 and Gemini 2.5. No single lab describes exactly this list.</figcaption></figure>

The clearest public example is from the Allen Institute for AI, which releases its data, code and checkpoints for every stage.

> [!PAPER] Olmo et al. (2025), Olmo 3 · Figure 2 · page 4
> [![Figure 2 of the Olmo 3 report: pretraining, midtraining and long context make Olmo 3 Base; then Think SFT, Think DPO and Think RLVR make Olmo 3 Think, Instruct SFT, DPO and RLVR make Olmo 3 Instruct, and RL-Zero RLVR goes straight from the base model](/img/writings/history/h-olmo3-flow.png)](/img/writings/history/h-olmo3-flow.png)
>
> **What it says:** the "depiction of model flow for Olmo 3". Development is divided into "base model training (left) and post-training (right) stages", each with its own recipe of data and method.
>
> **What it means:**
>
> Read it left to right as the history of this series. Pre-training is Part 1. SFT copies, DPO prefers and RLVR checks. The bottom branch, RL-Zero, repeats R1-Zero's experiment on a model whose pre-training data is fully public, so others can test what RL really adds.

Here are the stages, with what is public about each.

**Pre-training** on trillions of tokens of text, code and images, with the next-token loss of Part 1. The Olmo 3 report lists 5.9 trillion tokens for its pre-training mix.

**Mid-training** continues pre-training on a smaller, cleaner mix that favours maths, code, reasoning and question answering. For Olmo 3 it is 100 billion tokens. Long-context training, with longer documents, often follows.

**Supervised fine-tuning** on instructions, conversations, tool calls and long chains of thought. Many of the chains are written by a stronger model and filtered by checkers, not written by people.

**Preference tuning** with reward models, DPO or both, on human and AI-labelled comparisons. This is where style, tone and refusals are shaped.

**Reasoning RL** with verifiable rewards on maths, code and puzzles, usually with a GRPO-style or PPO-style algorithm. DeepSeek reports more than 10% of the pre-training cost here.

**Agent RL** in tool environments, with rewards from tests, checkers and AI judges with rubrics. Kimi K2, DeepSeek-V3.2 and Gemini 2.5 all describe some form of it.

**Safety training** runs through several of these stages. It includes refusal and red-team data in SFT and safety reward models (Llama 2 had a separate one). It also includes specifications or a constitution that the model learns to reason over.

**Distillation** makes the smaller members of a model family from the largest one, as Qwen3 and DeepSeek-R1 describe.

Evaluation runs alongside every stage, because each stage can damage what an earlier one built. The order varies, rounds repeat, and labs keep the details private. What they share is the shape. Learn from text, copy good examples, learn what people prefer, then practise against rewards that are hard to fool.

## What is still open

Rewards are still the weak point. Learned reward models can be hacked, and human votes reward flattery. Checkers exist only where answers can be checked, and most useful work, such as writing, advice and research, has no unit test. Much of today's work is on rubric-based and model-based rewards that stretch RL to such tasks, and on keeping them honest.

The other open question is what RL really adds. Several 2025 papers found that base models already contain much of the reasoning that RL brings out. In that view, RL mostly selects and strengthens behaviour that pre-training made possible, as with the aha moment in DeepSeek-V3-Base. Others point to new skills learned in long agent training. The answer matters for where the next unit of compute should go.

| Step | What it fixed | What it left open |
|---|---|---|
| Preferences from people (2017 to 2020) | A reward for goals nobody can write down | Expensive labels; reward hacking |
| RLHF with PPO, InstructGPT (2022) | Helpful assistants from base models | Four models in memory; fiddly training |
| KL penalty, early stopping | Limits on over-optimisation | Still a proxy; length and style bias |
| Constitutional AI, RLAIF (2022, 2023) | Cheaper labels; written principles | AI judges have their own biases |
| DPO and its family (2023, 2024) | Preference learning without RL | Learns only from a fixed data set |
| Verifiers, PRMs, STaR (2021 to 2023) | Signals for correct reasoning | Step labels are costly |
| o1, RLVR, GRPO, R1 (2024, 2025) | Long reasoning from checkable rewards | Only where answers can be checked |
| DAPO, Dr. GRPO, Qwen3 (2025) | Stable, efficient reasoning RL | Length and entropy problems remain |
| Agent RL in environments (2025, 2026) | Models that finish multi-step tasks | Checkers that can be gamed |
| Harder tests, arenas, AI judges | Measuring models beyond saturated tests | Contamination; leaderboard gaming |

> [!TAKEAWAYS]
> - Post-training turns a base model into an assistant: SFT copies good answers, and preference learning teaches what people prefer.
> - A reward model is a Bradley-Terry model of pairwise choices; RLHF maximises its score with PPO, on a KL leash to the reference model.
> - Optimising a learned reward too hard makes the true quality fall (Goodhart's law); best-of-n, the KL penalty and early stopping limit the damage.
> - DPO solves the KL-penalised objective in closed form and turns preference learning into a classification loss with no RL step.
> - Since 2024, the biggest gains come from verifiable rewards; GRPO compares each answer with its group, and DeepSeek-R1 showed long reasoning emerging from it.
> - In 2026 a frontier model goes through pre-training, mid-training, SFT, preference tuning, reasoning RL, agent RL, safety training and distillation; labs publish only outlines.
> - Testing moved from saturated exams to expert questions, real software tasks, puzzles and human arenas, each with its own way to be gamed.

Key papers: [Christiano et al. 2017](https://arxiv.org/abs/1706.03741), [InstructGPT 2022](https://arxiv.org/abs/2203.02155), [Gao et al. 2022](https://arxiv.org/abs/2210.10760), [DPO 2023](https://arxiv.org/abs/2305.18290), [DeepSeekMath 2024](https://arxiv.org/abs/2402.03300), [DeepSeek-R1 2025](https://arxiv.org/abs/2501.12948), [Olmo 3 2025](https://arxiv.org/abs/2512.13961).

Next: [Part 5, Inside a Modern Language Model](inside-a-modern-language-model.md), how the base model is built: its data, scale and architecture.
