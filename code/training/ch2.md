---
description: "The history of how language models learned to follow instructions, told as one story where each step fixes a problem left by the step before: from Shannon's n-grams and the Transformer, through pretraining, learning from human preferences and instruction tuning, to InstructGPT, ChatGPT, DPO, GRPO and DeepSeek-R1. With the papers' own figures, the four key equations worked on real numbers, and small experiments you can run."
---
# Chapter 2 · How it started, and how it evolved

> **Goal:** by the end of this chapter you can tell the story of how a next-word predictor became an assistant, step by step, and say for each step *which problem it solved* and *which new problem it left behind*. You will know the papers behind each step, you will have seen their key figures and sentences, and you will be able to work four equations by hand: the Bradley-Terry preference model, the RLHF objective with its KL penalty, the DPO loss, and GRPO's group-relative advantage. Each equation comes with a small script that runs on a laptop.

---

## 2.1 The story in one picture

Chapter 1 showed the end result: a base model that only continues text, and an instruct model that answers your question. This chapter is about *how people got from one to the other*. It is a short history, about eight years of fast work, with a longer prehistory behind it.

The useful way to read this history is as a chain of problems and fixes. Nobody planned "SFT, then a reward model, then PPO" from the start. Each method was invented because the previous one did something annoying, and each fix brought a new annoyance of its own. If you remember the chain, you remember why each part of a modern training pipeline exists.

Here is every event in this chapter on one timeline. The colour of each dot says which of four threads it belongs to.

{{FIG:ch2_timeline|The events in this chapter, from Shannon's work on predicting English text to DeepSeek-R1. Blue: pretraining and scale. Orange: learning from human preferences with reinforcement learning. Green: instruction data for supervised fine-tuning. Yellow: reinforcement learning for reasoning, the newest thread. Months are the first public version, usually the arXiv submission.}}

For most of the time, three threads ran side by side, often in different labs. One thread made models bigger and trained them on more text. Another learned to turn "this answer is better than that one" into a training signal. A third collected tasks written as instructions and fine-tuned on them. In early 2022 the three met in one paper, InstructGPT, and later that year the result became ChatGPT.

{{FIG:ch2_threads|Three threads that met in 2022. Pretraining gives the model its knowledge and fluency; instruction data teaches it the habit of doing what was asked; preference learning gives a number for "better" that reinforcement learning can increase.}}

And here is the chain of problems and fixes that the rest of the chapter walks through, one row at a time.

{{FIG:ch2_chain|The whole chapter as nine problems and nine fixes. Read down the left column for the problems and down the right for the fixes. Each fix creates the problem on the next row.}}

Two words will come up again and again, so let us define them now.

> [!DEFINITION] Base model
> A language model trained only to predict the next token on a very large amount of text. It has absorbed a lot of knowledge and writes fluent text, but it was never taught to answer questions or follow instructions. Give it a question and it may continue with more questions, because that is what often comes next on a web page.

> [!DEFINITION] Post-training
> Everything done to a base model after pretraining to make it a useful assistant: supervised fine-tuning on example answers, learning from preferences, reinforcement learning. Most of this chapter is the history of post-training.

## 2.2 Before 2017: learning to predict the next word

The idea that a machine could model language by predicting what comes next is older than computers that could do it well. This section is short on purpose. It sets up the one idea everything else builds on.

**1948 and 1951: Shannon.** In *A Mathematical Theory of Communication* (1948), Shannon built "approximations to English" by choosing each next letter or word with the probability it has after the previous ones. With one word of context the output already looks a little like English; with more context it looks more like it. In *Prediction and Entropy of Printed English* (1951) he asked people to guess the next letter of a text, one letter at a time, and used how often they were right to estimate how predictable English is. Predicting the next symbol, and measuring how surprised you are, is still exactly the training objective of every model in this book.

> [!DEFINITION] Language model
> A model that gives a probability to every possible next token, given the tokens so far. Writing text means repeatedly picking a next token from that probability and appending it. "Training a language model" means adjusting it so the real next token gets a high probability.

> [!DEFINITION] n-gram model
> The simplest language model: the probability of the next word depends only on the previous $$n - 1$$ words, and is estimated by counting how often each sequence of $$n$$ words appears in a text. A 2-gram (bigram) model looks at one previous word. It cannot generalise: a word pair it never counted gets probability zero.

**2003: neural language models.** Yoshua Bengio and colleagues (*A Neural Probabilistic Language Model*, JMLR 2003) replaced the counting with a small neural network. Each word became a learned vector of numbers, and the network predicted the next word from the vectors of the previous words. Similar words got similar vectors, so the model could generalise to word sequences it had never seen.

> [!DEFINITION] Embedding
> A list of numbers (a vector) that represents a token. The numbers are learned during training, so tokens used in similar ways end up with similar vectors.

**2013 to 2014: word vectors, seq2seq and attention.** Word2vec (Mikolov et al., January 2013) showed that very cheap training on a lot of text gives word vectors with useful structure. In September 2014, Sutskever, Vinyals and Le trained a recurrent network to read a sentence and write its translation (*sequence to sequence*), and Bahdanau, Cho and Bengio added **attention**: when writing each output word, the model looks back at all input words and decides which ones matter now.

**2017: the Transformer.** Vaswani et al. (*Attention Is All You Need*, June 2017) built a model from attention alone, without the recurrent network. Because every position can be computed at the same time, Transformers train efficiently on modern hardware, and they kept getting better as they got bigger. Every model in the rest of this chapter is a Transformer.

> [!DEFINITION] Transformer
> A neural network made of stacked layers, each with an attention step (every token gathers information from the other tokens) and a small feed-forward network applied to each token. The "decoder-only" kind used by GPT models lets each token attend only to the tokens before it, which is exactly what next-token prediction needs.

The same month, June 2017, a paper appeared that had nothing to do with language: a robot learning to do a backflip from a person's clicks. It will matter a great deal in Section 2.4.

## 2.3 Pretrain, then fine-tune, then prompt

**2018: GPT-1 and BERT.** OpenAI's report *Improving Language Understanding by Generative Pre-Training* (Radford et al., June 2018) trained "a 12-layer decoder-only transformer" on BooksCorpus, a collection of "over 7,000 unique unpublished books", to predict the next token. Then, for each task (classifying a sentence, answering a multiple-choice question), they added a small output layer and trained the whole model a little more on that task's labelled examples. The pretrained model needed far fewer labelled examples than a model trained from scratch. In October 2018, BERT (Devlin et al.) did the same with a different pretraining game (filling in masked words) and beat the state of the art on many benchmarks.

> [!DEFINITION] Pretraining
> The first, largest stage of training: next-token prediction (or a similar game) on a huge amount of unlabelled text. It is where the model learns grammar, facts and patterns. It uses almost all of the compute.

> [!DEFINITION] Fine-tuning
> Continuing to train an already trained model on a smaller, more specific dataset, so it gets good at one task or one style. It starts from the pretrained weights instead of random ones, which is why it needs so much less data.

The recipe "pretrain once, fine-tune per task" worked, but it had a cost: one separate fine-tuned model per task, and a labelled dataset for each.

**2019: GPT-2, one model and many tasks.** *Language Models are Unsupervised Multitask Learners* (Radford et al., February 2019) trained "a 1.5B parameter Transformer" on WebText, "slightly over 8 million documents for a total of 40 GB of text". The surprise was that some tasks could be done with *no* fine-tuning at all, just by writing the input so that the answer is the natural continuation. To get a summary of a news article, the authors added the text `TL;DR:` after the article and let the model continue, because on the web "TL;DR:" is usually followed by a short summary. The summaries were weak, but the method was new: the task is specified in the text.

**2020: GPT-3 and few-shot prompting.** *Language Models are Few-Shot Learners* (Brown et al., May 2020) scaled the same idea to 175 billion parameters and showed that you can teach a task inside the prompt by giving a few examples.

> [!PAPER] Brown et al. (2020), GPT-3 · Figure 2.1 · page 7
> [![The GPT-3 paper's Figure 2.1. Three panels on the left show zero-shot, one-shot and few-shot prompts for translating English to French, with a task description, zero to three examples like sea otter to loutre de mer, and the prompt cheese. A panel on the right shows traditional fine-tuning with a gradient update after each example](/img/training/ch2-gpt3-fig21.png)](/img/training/ch2-gpt3-fig21.png)
>
> **Context:** Section 2 of the paper, where the authors define the settings they test.
>
> **What it says:** "Zero-shot, one-shot and few-shot" prompting put a task description and zero, one or a few examples *in the prompt*, and "no gradient updates are performed". Fine-tuning, on the right, changes the weights with one gradient update per example.
>
> **Why it matters:** with a large enough model, a task could be specified in text alone. This made one model useful for thousands of tasks, and it made writing prompts a skill.

> [!DEFINITION] Few-shot prompting
> Putting a few input-output examples of a task in the prompt, followed by a new input, and letting the model continue. The model's weights do not change. "Zero-shot" means giving only the instruction, with no examples.

**Scaling laws.** Why 175 billion parameters? In January 2020, Kaplan et al. (*Scaling Laws for Neural Language Models*) measured that the pretraining loss falls smoothly, as a power law, as you increase model size, data and compute together. That made "bigger" a predictable investment. In March 2022, Hoffmann et al. (*Training Compute-Optimal Large Language Models*, the Chinchilla paper) corrected the recipe: for a fixed compute budget, "for every doubling of model size the number of training tokens should also be doubled". Their 70B model Chinchilla, trained on 1.4 trillion tokens (20 tokens per parameter), beat the 280B Gopher with the same compute. Later open models such as LLaMA (2023) followed this lesson and trained smaller models on many more tokens.

### The problem a base model leaves

By 2021 base models were knowledgeable and fluent. But they were trained to continue internet text, and the internet is not an assistant. Ask a base model a question and it might answer it, or it might write five more questions, or a forum post arguing about the question, or something rude it once read. The InstructGPT paper (2022) put the problem in one sentence.

> [!PAPER] Ouyang et al. (2022), InstructGPT · Section 1 · page 2
> [![A sentence from the InstructGPT introduction, highlighted: the objective used for many recent large LMs, predicting the next token on a webpage from the internet, is different from the objective follow the user's instructions helpfully and safely. Thus, we say that the language modeling objective is misaligned](/img/training/ch2-instructgpt-misaligned.png)](/img/training/ch2-instructgpt-misaligned.png)
>
> **Context:** the opening paragraph of the paper. Just before this sentence, the authors list what goes wrong: models "making up facts, generating biased or toxic text, or simply not following user instructions".
>
> **What it says:** "predicting the next token on a webpage from the internet" is a different goal from "follow the user's instructions helpfully and safely", so "the language modeling objective is *misaligned*".
>
> **Why it matters:** this is the problem the whole rest of the chapter is about. More pretraining makes the model know more; it does not, by itself, make it do what you asked.

> [!DEFINITION] Alignment
> Making a model's behaviour match what its users and developers intend: follow the instruction, be truthful, avoid harm. In this chapter "aligned" is used in this practical sense, the way the papers use it.

Few-shot prompts helped, but they were fragile (change the examples and the answer changes), they cost context length, and they could not express things like "be honest when you do not know". Two fixes were being developed at the same time. One said: *show the model what you want*, with many instructions and good answers (Section 2.6). The other said: *let people judge the model's outputs*, and train the model toward what they prefer (Sections 2.4 and 2.5). The second one started with a robot.

## 2.4 Learning what people prefer (2017)

> [!DEFINITION] Reinforcement learning (RL)
> A way of training where an agent acts, receives a number called a **reward** that says how good the outcome was, and is updated so that actions leading to high reward become more likely. Nobody shows it the right action; it finds out by trying.

> [!DEFINITION] Policy
> In RL, the thing that chooses actions: given what it sees, it gives a probability to each possible action. For a language model the policy is the model itself: given the prompt and the tokens so far, it gives a probability to each next token. "Updating the policy" means changing the model's weights.

RL had beaten Atari games and Go by 2017, but in every case someone could write the reward as code: the game score, or win or lose. Many things we want cannot be written as code. What is the reward for "a good summary", "a polite answer", or "a graceful backflip"?

*Deep Reinforcement Learning from Human Preferences* (Christiano, Leike, Brown, Martic, Legg and Amodei, June 2017; authors from OpenAI and DeepMind) answered: do not write the reward, *learn* it from people. And do not ask people for a score, which is hard to give consistently. Show them two short video clips of the agent and ask which one is closer to what they want.

> [!PAPER] Christiano et al. (2017), Deep RL from Human Preferences · Figure 1 · page 2
> [![Figure 1 of the paper: a loop between an RL algorithm and an environment, with observation and action arrows, plus a reward predictor that receives human feedback and sends predicted reward to the RL algorithm. Caption highlighted: the reward predictor is trained asynchronously from comparisons of trajectory segments](/img/training/ch2-christiano-fig1.png)](/img/training/ch2-christiano-fig1.png)
>
> **Context:** the introduction, where the method is first described.
>
> **What it says:** the agent acts in its environment as usual, but its reward comes from a **reward predictor**, a neural network, and "the reward predictor is trained asynchronously from comparisons of trajectory segments", that is, from people's choices between pairs of short clips.
>
> **Why it matters:** this is the RLHF loop, five years before ChatGPT. Replace "clips of a robot" with "two answers from a language model" and you have the method behind InstructGPT.

> [!DEFINITION] Reward model
> A neural network that reads an output (a clip, an answer) and returns one number: how much a person would like it. It is trained on human comparisons and then used in place of the person, because it can score millions of outputs for free.

They showed it works with very little human time. Their most famous demonstration was a simulated robot taught to do backflips.

> [!PAPER] Christiano et al. (2017) · Section 3.2 · page 8
> [![A paragraph from the paper: The Hopper robot performing a sequence of backflips (see Figure 4). Highlighted: This behavior was trained using 900 queries in less than an hour. The agent learns to consistently perform a backflip, land upright, and repeat](/img/training/ch2-christiano-backflip.png)](/img/training/ch2-christiano-backflip.png)
>
> **Context:** the section on "novel behaviors", tasks with no reward function in the simulator at all.
>
> **What it says:** "This behavior was trained using 900 queries in less than an hour." Each query was one comparison of two clips by a person.
>
> **Why it matters:** nobody knew how to write a reward for "a good backflip", but a person can recognise one. About 900 clicks were enough to learn a reward that the RL algorithm could optimise.

### The Bradley-Terry model: from choices to scores

How do you train a network that outputs a *score* from data that only says "*this* one is better"? Christiano et al. used a model from 1952, built for ranking things from pairwise comparisons (and the same idea behind Elo ratings in chess).

> [!PAPER] Christiano et al. (2017) · Section 2.2.3, equation 1 · page 5
> [![Equation 1 of the paper: the predicted probability that segment one is preferred over segment two equals exp of the summed reward of segment one divided by the sum of exp of the summed rewards of both segments. Below it, the cross-entropy loss over the human labels. Highlighted: This follows the Bradley-Terry model](/img/training/ch2-christiano-eq1.png)](/img/training/ch2-christiano-eq1.png)
>
> **Context:** "Fitting the reward function", the core of the method.
>
> **What it says:** the chance that a person prefers clip 1 is $$e^{r_1}$$ divided by $$e^{r_1} + e^{r_2}$$, where $$r_1$$ and $$r_2$$ are the summed rewards of the two clips; the reward predictor is trained with a cross-entropy loss against the human labels. "This follows the Bradley-Terry model."
>
> **Why it matters:** every reward model in this chapter, including InstructGPT's and Llama 2's, is trained with this exact formula.

Dividing the top and bottom of that fraction by $$e^{r_A}$$ gives the form used everywhere since:

$$
P(A \succ B) = \frac{e^{r_A}}{e^{r_A} + e^{r_B}} = \frac{1}{1 + e^{-(r_A - r_B)}} = \sigma(r_A - r_B)
$$

where:

- $$A$$ and $$B$$ are two outputs for the same input (two clips, or two answers to the same prompt);
- $$A \succ B$$ (read "A is preferred to B") is the event that a person picks $$A$$;
- $$r_A$$ and $$r_B$$ are the reward model's scores for $$A$$ and $$B$$, any real numbers;
- $$\sigma$$ (sigma) is the **sigmoid** function, $$\sigma(z) = 1/(1 + e^{-z})$$, which squeezes any number into the range 0 to 1;
- $$e \approx 2.718$$ is the base of the natural logarithm.

> [!DEFINITION] Sigmoid
> The S-shaped function $$\sigma(z) = 1 / (1 + e^{-z})$$. It turns any number into a probability: $$\sigma(0) = 0.5$$, large positive numbers give almost 1, large negative numbers give almost 0.

Only the *difference* of the two rewards appears. That has a consequence you should remember: you can add the same constant to every reward and nothing changes. A reward model's numbers have no absolute meaning; only gaps between outputs for the same prompt do.

**Worked example.** Say the reward model gives answer A a score of 1.2 and answer B a score of 0.4. The gap is 0.8, so the model predicts that a person picks A with probability $$\sigma(0.8) = 1/(1 + e^{-0.8}) = 1/(1 + 0.449) = 0.690$$. Add 10 to both scores and the gap is still 0.8, so the probability is still 0.690. The first lines of `ch2_bradley_terry.py` print exactly this:

```text
== 1. Bradley-Terry on two responses ==
r_A = 1.2, r_B = 0.4  ->  P(A preferred) = sigmoid(1.2 - 0.4) = sigmoid(0.8) = 0.6900
add 10 to both: sigmoid(11.2 - 10.4) = 0.6900   (only the difference matters)
  reward gap   0:  P(better one preferred) = 0.500
  reward gap 0.5:  P(better one preferred) = 0.622
  reward gap   1:  P(better one preferred) = 0.731
  reward gap   2:  P(better one preferred) = 0.881
  reward gap   3:  P(better one preferred) = 0.953
  reward gap   5:  P(better one preferred) = 0.993
```

{{FIG:ch2_bt_curve|The Bradley-Terry curve. The horizontal axis is the reward gap between A and B, the vertical axis the predicted chance that A is preferred. The orange dots are the values printed by ch2_bradley_terry.py. A gap of 3 already means 95%.}}

To *train* the reward model, we turn this probability into a loss. For each human comparison we know which answer was chosen ($$y_w$$, "winner") and which was rejected ($$y_l$$, "loser"):

$$
\mathcal{L}(\theta) = -\log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)
$$

where:

- $$x$$ is the prompt, $$y_w$$ the chosen output and $$y_l$$ the rejected one;
- $$r_\theta(x, y)$$ is the reward model's score, and $$\theta$$ (theta) stands for all its weights;
- $$\log$$ is the natural logarithm;
- $$\mathcal{L}$$ is the loss for one comparison; training averages it over all comparisons and lowers it with gradient descent.

This is the cross-entropy loss from Christiano's equation, for the case where the person chose one side. If the model already gives the winner a much higher score, $$\sigma$$ is close to 1 and the loss is close to 0. If it ranks them the wrong way round, the loss is large. With the numbers above, the loss is $$-\log 0.690 = 0.371$$. If the model had the scores the other way round (A 0.4, B 1.2) the loss would be $$-\log \sigma(-0.8) = -\log 0.310 = 1.171$$.

{{FIG:ch2_pref_pair|One human comparison becomes one training step. The reward model scores both answers; the gap goes through the sigmoid to a probability; the loss is minus its log; the gradient raises the chosen answer's score and lowers the rejected one's.}}

### Experiment: can a reward model learn a hidden taste from choices alone?

Let us check that this works, in a setting small enough to verify. In `ch2_bradley_terry.py`, each "response" is described by four numbers: is it correct (0 or 1), how detailed, how polite, and how long (each between 0 and 1). A simulated labeler has a hidden taste, a true reward of $$3 \cdot \text{correct} + 1.5 \cdot \text{detail} + 1.0 \cdot \text{polite} - 0.5 \cdot \text{length}$$. Shown two responses, the labeler prefers A with probability $$\sigma(r_A - r_B)$$, so the labels are noisy, like real people. The reward model never sees the hidden weights; it only sees which response won each comparison.

```python
# simplified from ch2_bradley_terry.py
TRUE_W = np.array([3.0, 1.5, 1.0, -0.5])                  # the labeler's hidden taste
X = np.column_stack([rng.integers(0, 2, 200),             # 200 responses: correct 0/1,
                     rng.random(200), rng.random(200), rng.random(200)])   # detail, polite, length
r_true = X @ TRUE_W

def make_pairs(n):
    i, j = rng.integers(0, 200, n), rng.integers(0, 200, n)
    a_wins = rng.random(n) < sigmoid(r_true[i] - r_true[j])   # a noisy human choice
    return np.where(a_wins, i, j), np.where(a_wins, j, i)    # (winner, loser)

w = torch.zeros(4, requires_grad=True)                     # the reward model: r(x) = w . x
opt = torch.optim.Adam([w], lr=0.05)
for step in range(401):
    r = Xt @ w
    loss = -F.logsigmoid(r[win] - r[lose]).mean()           # the Bradley-Terry loss
    opt.zero_grad(); loss.backward(); opt.step()
```

Block by block:

- `TRUE_W` and `r_true` are the hidden taste and the true reward of each of the 200 responses. Only the simulator knows them.
- `make_pairs` picks two random responses, lets the simulated labeler choose with probability $$\sigma(r_A - r_B)$$, and returns the winner and loser indices. We make about 1,000 training comparisons and 2,000 separate test comparisons.
- The reward model is as simple as possible: four weights $$w$$, and a score $$w \cdot x$$ for each response. It starts at zero, so at step 0 every pair is a coin flip.
- The loop computes all scores, takes the Bradley-Terry loss over all training pairs, and takes one Adam step. `F.logsigmoid` computes $$\log \sigma$$ in a numerically safe way.

[![Terminal output of ch2_bradley_terry.py: the Bradley-Terry table, the data summary with an example comparison, the training log from loss 0.6931 to 0.4206 with test accuracy rising to 0.775, the learned weights next to the true weights, and accuracy for different numbers of comparisons](/img/training/ch2_bradley_terry-run.png)](/img/training/ch2_bradley_terry-run.png)

The key lines:

```text
test accuracy of the TRUE reward (the ceiling, because labels are noisy): 0.777
step   0  loss 0.6931  test acc 0.500  w = [0. 0. 0. 0.]
step  50  loss 0.4364  test acc 0.775  w = [ 1.94  1.39  0.79 -0.24]
step 400  loss 0.4206  test acc 0.775  w = [ 2.93  1.53  0.95 -0.23]
 correct: true weight +3.00   learned +2.93
  detail: true weight +1.50   learned +1.53
  polite: true weight +1.00   learned +0.95
  length: true weight -0.50   learned -0.23
correlation between learned and true reward over the 200 responses: 0.9989
```

Four things to notice.

1. The loss starts at $$\ln 2 = 0.693$$. With all scores equal, every prediction is 0.5, and $$-\log 0.5 = 0.693$$. You will see this number again in DPO.
2. The test accuracy, 0.775, is almost exactly the **ceiling** of 0.777: even the true reward cannot predict more than 77.7% of these noisy choices, because the simulated people sometimes pick the worse answer, as real people do. A reward model that scores 100% on human labels would be memorising noise.
3. The learned weights are close to the hidden ones, and the learned scores correlate 0.9989 with the true reward. The weight on length is the least accurate (-0.23 against -0.50): it has the smallest effect on choices, so the data says least about it.
4. With few comparisons the weights are unreliable. The last part of the script refits with fewer pairs; with 25 comparisons it learned a weight of -1.34 for politeness, the wrong sign. Real reward models are trained on tens of thousands to millions of comparisons for this reason.

{{FIG:ch2_bt_weights|True and learned weights of the toy reward model after training on 994 noisy comparisons. Choices alone were enough to recover what the labeler cares about, in the right order.}}

> [!NOTE]
> Real reward models are not four weights on hand-made features. They are a full language model (often the same size as the model being trained, or a bit smaller) with its last layer replaced by a single number. The loss is the same.

## 2.5 From robots to language (2019 to 2021)

### Ziegler et al. (2019): the KL penalty

*Fine-Tuning Language Models from Human Preferences* (Ziegler, Stiennon, Wu, Brown, Radford, Amodei, Christiano and Irving, OpenAI, September 2019) moved the method to text. They fine-tuned GPT-2 to continue text in a given style (for example, with positive sentiment) and to summarise. For the style tasks "we achieve good results with only 5,000 comparisons evaluated by humans".

They also introduced the piece that every RLHF system since has kept: a penalty for drifting too far from the starting model.

> [!PAPER] Ziegler et al. (2019) · Section 2, equation 2 · page 3
> [![Equation 2 of the paper: R of x and y equals r of x and y minus beta times log of pi of y given x over rho of y given x. Highlighted: To keep pi from moving too far from rho, we add a penalty. The paragraph after it says the term plays the role of an entropy bonus, prevents the policy from moving too far from the range where r is valid, and encourages coherence and topicality](/img/training/ch2-ziegler-eq2.png)](/img/training/ch2-ziegler-eq2.png)
>
> **Context:** the method section, right after the reward model is trained.
>
> **What it says:** "To keep $$\pi$$ from moving too far from $$\rho$$, we add a penalty". The model $$\pi$$ is trained on a modified reward: the reward model's score minus $$\beta$$ times the log-ratio of the new model's probability to the original model's.
>
> **Why it matters:** the reward model was trained on outputs that look like the original model's. Far from those, its scores mean little. The penalty keeps the policy "within the range where $$r$$ is valid", and keeps the text coherent.

We will work through this equation properly in Section 2.7, where InstructGPT uses it in the same form. For now, the idea: a learned reward is only trustworthy near the data it was trained on, so you pay a price for every step away from the original model.

The paper also contains a cautionary tale that shows how literally RL optimises whatever you give it.

> [!PAPER] Ziegler et al. (2019) · Section 4.4 · page 12
> [![Section 4.4, Bugs can optimize for bad behavior. One of our code refactors introduced a bug which flipped the sign of the reward. Because the same bug also flipped the sign of the KL penalty, the result was a model that optimized for negative sentiment while staying fluent, and since labelers were told to rate sexually explicit text very low, the model learned to output only that. Highlighted: not gibberish but maximally bad output](/img/training/ch2-ziegler-bug.png)](/img/training/ch2-ziegler-bug.png)
>
> **Context:** the "lessons" section at the end of the paper.
>
> **What it says:** a code change "introduced a bug which flipped the sign of the reward", and of the KL penalty too. The model optimised for what the labelers hated most while still writing fluent English: "not gibberish but maximally bad output". The sentence continues on the next page: the authors were asleep "during the training process, so the problem was noticed only once training had finished".
>
> **Why it matters:** RL is a very strong optimiser of the number you give it, including a wrong number. Every later safeguard (KL penalties, checking samples, evaluating with people) exists because of this.

> [!DEFINITION] PPO (Proximal Policy Optimization)
> The RL algorithm used by almost all RLHF work from 2019 to 2023 (Schulman et al., July 2017). It updates the policy to make high-reward outputs more likely, but limits ("clips") how much the probability of any output can change in one update, which keeps training stable. It needs a second network, a **value model**, that predicts the expected reward, to judge whether an output was better or worse than expected.

### Stiennon et al. (2020): human feedback beats a metric

*Learning to Summarize from Human Feedback* (Stiennon, Ouyang, Wu, Ziegler, Lowe, Voss, Radford, Amodei and Christiano, OpenAI, September 2020) scaled this up on summaries of Reddit posts, with models up to 6.7 billion parameters, and found that it beat both the human-written reference summaries and much larger supervised models.

> [!PAPER] Stiennon et al. (2020), Learning to Summarize · Figure 1 · page 2
> [![Figure 1 of the paper: fraction of the time humans prefer each model's summaries over the human reference summaries, against model size from 1.3B to 12.9B. Human feedback models, highlighted, are at about 0.6 to 0.7, above the dashed line at 0.5 for the reference summaries. Supervised learning is around 0.4, pretrain only around 0.2 to 0.35](/img/training/ch2-stiennon-fig1.png)](/img/training/ch2-stiennon-fig1.png)
>
> **Context:** the first figure of the paper, its headline result.
>
> **What it says:** summaries from models trained with "Human feedback" were preferred to the human-written reference summaries more than half the time, at every size tested. Supervised models trained to imitate the references stayed below 0.5.
>
> **Why it matters:** training on *what people prefer* beat training on *what people wrote*. Imitation can only copy the references, including their flaws; preferences can push past them.

The paper also drew the three-step diagram that InstructGPT would reuse a year and a half later.

> [!PAPER] Stiennon et al. (2020) · Figure 2 · page 4
> [![Figure 2 of the paper in three columns. 1: Collect human feedback: a Reddit post, several summaries, two are shown to a human who judges which is better. 2: Train reward model: both summaries go through the reward model, giving r_j and r_k, and the loss is log of sigma of r_j minus r_k. 3: Train policy with PPO: a new post, the policy writes a summary, the reward model scores it, and the reward updates the policy](/img/training/ch2-stiennon-fig2.png)](/img/training/ch2-stiennon-fig2.png)
>
> **Context:** the overview of the method.
>
> **What it says:** collect comparisons, train a reward model with the loss "log($$\sigma$$($$r_j - r_k$$))", then train the policy with PPO against that reward model.
>
> **Why it matters:** the middle panel is the Bradley-Terry loss you just ran. The whole RLHF recipe was in place in 2020; what changed in 2022 was the task (any instruction instead of summaries).

They also measured what happens if you optimise the reward model too hard, and the answer became one of the most important figures in the field.

> [!PAPER] Stiennon et al. (2020) · Figure 5 · page 8
> [![Figure 5 of the paper: the horizontal axis is KL from the supervised baseline, from 0 to 250; a dashed line for RM prediction rises steadily to nearly 1.0, while a solid line for actual human preference rises to about 0.47 at KL around 10 and then falls to under 0.1 at KL 250. Caption highlighted: initially improves summaries, but eventually overfits](/img/training/ch2-stiennon-fig5.png)](/img/training/ch2-stiennon-fig5.png)
>
> **Context:** Section 4.3, where the authors deliberately let the policy move further and further from the supervised model.
>
> **What it says:** as the policy drifts (larger KL), the reward model keeps predicting better and better summaries (dashed line), but real people's preference (solid line) first rises and then collapses. "Optimizing against the reward model initially improves summaries, but eventually overfits, giving worse summaries."
>
> **Why it matters:** this is **reward over-optimisation**, the core weakness of any learned reward. It is the reason for the KL penalty, and later one of the reasons people turned to rewards that can be checked (Section 2.9). A 2022 paper by Gao, Schulman and Hilton (*Scaling Laws for Reward Model Overoptimization*) measured this effect systematically.

> [!DEFINITION] Reward hacking (over-optimisation)
> When a policy finds outputs that get a high score from the reward model without being good, because the reward model is only an imperfect copy of human judgement. The harder you optimise, the more likely you find its blind spots.

Section 2.7 reproduces this effect in miniature with numbers you can check.

### WebGPT and HHH (December 2021)

Two papers from the end of 2021 show how the idea spread. *WebGPT* (Nakano et al., OpenAI, December 2021) fine-tuned GPT-3 to answer questions by browsing the web and citing sources, and found that a simpler use of the reward model also worked well.

> [!PAPER] Nakano et al. (2021), WebGPT · Abstract · page 1
> [![The WebGPT abstract. Highlighted: Our best model is obtained by fine-tuning GPT-3 using behavior cloning, and then performing rejection sampling against a reward model trained to predict human preferences. This model's answers are preferred by humans 56% of the time to those of our human demonstrators, and 69% of the time to the highest-voted answer from Reddit](/img/training/ch2-webgpt-abstract.png)](/img/training/ch2-webgpt-abstract.png)
>
> **Context:** the abstract.
>
> **What it says:** the best model used "rejection sampling against a reward model": generate several answers, keep the one the reward model scores highest. Its answers were "preferred by humans 56% of the time" to answers written by the human demonstrators.
>
> **Why it matters:** you do not always need RL to use a reward model. Picking the best of N samples is simple and strong; Llama 2 would use it as a training step in 2023.

> [!DEFINITION] Best-of-N (rejection sampling)
> Generate N answers, score each with the reward model, keep the best. Used at answer time it costs N times the compute; used in training, the kept answers become new fine-tuning data.

At Anthropic, *A General Language Assistant as a Laboratory for Alignment* (Askell et al., December 2021) set the goal in three words.

> [!PAPER] Askell et al. (2021), Anthropic · Section 1 · page 3
> [![A sentence from the introduction: We will define an AI as aligned if it is, in three words, helpful, honest, and harmless or HHH, with helpful, honest, and harmless highlighted](/img/training/ch2-hhh-definition.png)](/img/training/ch2-hhh-definition.png)
>
> **Context:** the introduction, defining what the authors mean by an aligned assistant.
>
> **What it says:** an AI is aligned "if it is, in three words, helpful, honest, and harmless or 'HHH'".
>
> **Why it matters:** "helpful, honest and harmless" became the standard way to describe what post-training aims for. The paper also found that "ranked preference modeling performs much better than imitation learning" for learning these judgements.

## 2.6 The other thread: show the model many instructions (2021 to 2022)

While one group of researchers was learning rewards from comparisons, others attacked the same problem from the data side. The problem with a base model is that it was never shown "an instruction, then a good response". So: collect many tasks, write each one as a natural-language instruction, and fine-tune on all of them together. If the model learns the *general* habit of following instructions, it should handle new instructions it never saw.

> [!DEFINITION] Instruction tuning
> Supervised fine-tuning on a large mix of tasks, each written as an instruction plus input, with the desired output as the target. The goal is not to learn those tasks but to learn to follow instructions in general, so the model does better on unseen tasks without examples (zero-shot).

> [!DEFINITION] Supervised fine-tuning (SFT)
> Fine-tuning with the ordinary next-token loss on example inputs and the outputs you want. In post-training, SFT data is usually "prompt, ideal response" pairs, and the loss is computed only on the response tokens.

The thread in four steps:

- **Natural Instructions** (Mishra et al., April 2021): "a dataset of 61 distinct tasks, their human-authored instructions, and 193k task instances", built from the instructions originally written for crowd workers. Its 2022 successor, **Super-NaturalInstructions** (Wang et al., April 2022), grew to more than 1,600 tasks.
- **FLAN** (Wei et al., Google, September 2021) took "a 137B parameter pretrained language model", fine-tuned it on more than 60 NLP datasets rewritten as instructions, and tested it on task types held out from training. FLAN "surpasses zero-shot 175B GPT-3 on 20 of 25 datasets".
- **T0** (Sanh et al., BigScience, October 2021) did the same with an 11-billion-parameter model and crowd-written prompts, "often outperforming models up to 16x its size".
- **Flan-PaLM** (Chung et al., Google, October 2022) scaled the idea to 540B parameters and "1.8K tasks", which "outperforms PaLM 540B by a large margin (+9.4% on average)".

FLAN's Figure 2 is the clearest picture of where instruction tuning sits between the two older recipes.

> [!PAPER] Wei et al. (2021), FLAN · Figure 2 · page 2
> [![Figure 2 of the FLAN paper with three panels. A, Pretrain-finetune (BERT, T5): a pretrained LM is fine-tuned on task A and run on task A, needing many task-specific examples and one specialised model per task. B, Prompting (GPT-3): a pretrained LM is run on task A, improved via few-shot prompting or prompt engineering. C, Instruction tuning (FLAN), highlighted: a pretrained LM is instruction-tuned on many tasks B, C, D and then run on an unseen task A](/img/training/ch2-flan-fig2.png)](/img/training/ch2-flan-fig2.png)
>
> **Context:** the introduction, comparing the paper's idea with BERT-style fine-tuning and GPT-3-style prompting.
>
> **What it says:** in (C), the model is tuned on many tasks B, C, D and then used on a task A it has not seen. "Model learns to perform many tasks via natural language instructions."
>
> **Why it matters:** this is the "SFT" step of every modern pipeline in its first form. InstructGPT's step 1 is the same idea with one change: real user prompts and answers written by hired labelers, instead of academic datasets.

What did instruction tuning leave unsolved? The datasets were mostly academic tasks (classify this, translate that) with short, single correct answers. Real users ask for open-ended things ("write a cover letter", "explain this bug") where there are many good answers, and where "good" includes tone, honesty and safety. A dataset of fixed answers cannot easily say "this answer is better than that one". The preference thread could. In early 2022 one paper combined them.

## 2.7 InstructGPT: the threads meet (2022)

*Training Language Models to Follow Instructions with Human Feedback* (Ouyang et al., OpenAI, March 2022) took prompts that real users had sent to the OpenAI API, hired "a team of about 40 contractors", and ran three steps.

> [!PAPER] Ouyang et al. (2022), InstructGPT · Figure 2 · page 3
> [![Figure 2 of InstructGPT in three columns. Step 1: collect demonstration data and train a supervised policy: a prompt like explain the moon landing to a 6 year old is sampled, a labeler demonstrates the desired output, and the data fine-tunes GPT-3. Step 2: collect comparison data and train a reward model: several outputs are sampled, a labeler ranks them D better than C better than A equals B, and the ranking trains the reward model. Step 3: optimise a policy against the reward model with PPO: a new prompt, the policy writes an output, the reward model scores it, and the reward updates the policy. Caption highlighted: supervised fine-tuning, reward model training, and reinforcement learning via proximal policy optimization](/img/training/ch2-instructgpt-fig2.png)](/img/training/ch2-instructgpt-fig2.png)
>
> **Context:** the introduction; this diagram became the standard picture of RLHF.
>
> **What it says:** "(1) supervised fine-tuning (SFT), (2) reward model (RM) training, and (3) reinforcement learning via proximal policy optimization (PPO) on this reward model."
>
> **Why it matters:** step 1 is the instruction-tuning thread; steps 2 and 3 are the preference thread, exactly as in Stiennon's Figure 2. This three-step recipe defined post-training until 2023.

{{FIG:ch2_instructgpt_steps|The three steps with the paper's dataset sizes: about 13k prompts for SFT, 33k for the reward model and 31k for PPO (Section 3.2). Labelers ranked between 4 and 9 answers per prompt, which gives many comparisons per prompt.}}

A detail about step 2: labelers did not compare just two answers. They ranked "between K = 4 and K = 9 responses", and every pair in the ranking became one Bradley-Terry comparison, so a ranking of 9 answers gives $$\binom{9}{2} = 36$$ pairs. The loss is the one from Section 2.4.

### The result

> [!PAPER] Ouyang et al. (2022) · Figure 1 · page 2
> [![Figure 1 of InstructGPT: win rate against the 175B SFT model on the vertical axis, model size 1.3B, 6B and 175B on the horizontal axis. Lines from top: PPO-ptx and PPO around 0.5 to 0.65, SFT from about 0.35 to 0.5, GPT prompted from about 0.2 to 0.4, GPT from about 0.15 to 0.25. Caption highlighted: how often outputs from each model were preferred to those from the 175B SFT model](/img/training/ch2-instructgpt-fig1.png)](/img/training/ch2-instructgpt-fig1.png)
>
> **Context:** the first figure, the headline result.
>
> **What it says:** labelers compared each model's outputs against the 175B SFT model. Plain GPT-3 lost most of the time even when prompted carefully; the RLHF models (PPO and PPO-ptx) won more often than any other. The caption ends: "outputs from our 1.3B PPO-ptx model are preferred to those from the 175B GPT-3".
>
> **Why it matters:** a model with more than 100 times fewer parameters was preferred, because it was trained to do what people asked. Post-training was not a small polish; on this task it mattered more than size.

### The RLHF objective, symbol by symbol

Step 3 maximises this objective (the paper's equation 2):

> [!PAPER] Ouyang et al. (2022) · Section 3.5, equation 2 · page 9
> [![Equation 2 of InstructGPT: objective of phi equals the expectation over x and y drawn from the RL policy of r theta of x and y minus beta times log of pi RL of y given x over pi SFT of y given x, plus gamma times the expectation over x from the pretraining distribution of log pi RL of x. Highlighted: PPO-ptx, and the KL reward coefficient beta and pretraining loss coefficient gamma control the strength of the KL penalty and pretraining gradients](/img/training/ch2-instructgpt-eq2.png)](/img/training/ch2-instructgpt-eq2.png)
>
> **Context:** the "Models" section, describing the RL step.
>
> **What it says:** maximise the reward minus a KL penalty, plus (optionally) a term that keeps the model good at ordinary next-token prediction. Models trained with that last term are called "PPO-ptx". $$\beta$$ and $$\gamma$$ "control the strength of the KL penalty and pretraining gradients respectively".
>
> **Why it matters:** this is Ziegler's 2019 modified reward, plus one new term. The appendix gives the value used: $$\beta = 0.02$$.

Written out:

$$
\begin{aligned}\text{objective}(\phi) = {} & \mathbb{E}_{(x, y) \sim D_{\pi_\phi^{\text{RL}}}}\Big[ r_\theta(x, y) - \beta \log \frac{\pi_\phi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)} \Big] \\ & + \gamma\, \mathbb{E}_{x \sim D_{\text{pretrain}}}\big[\log \pi_\phi^{\text{RL}}(x)\big]\end{aligned}
$$

where:

- $$x$$ is a prompt from the dataset and $$y$$ is an answer **sampled from the policy being trained**, so the data changes as the model changes;
- $$\pi_\phi^{RL}$$ is the policy being trained (the language model), with weights $$\phi$$ (phi); $$\pi_\phi^{RL}(y \mid x)$$ is the probability it gives to the whole answer $$y$$, the product of its next-token probabilities;
- $$\pi^{SFT}$$ is the frozen model from step 1, the starting point. It is often called the **reference model**, $$\pi_{\text{ref}}$$;
- $$r_\theta(x, y)$$ is the reward model's score from step 2, frozen during this step;
- $$\log \frac{\pi^{RL}(y|x)}{\pi^{SFT}(y|x)}$$ is the **log-ratio**: positive if the trained model now likes this answer more than the starting model did, negative if less;
- $$\beta$$ (beta) is the KL coefficient, how much each unit of drift costs; InstructGPT used 0.02;
- $$\mathbb{E}$$ means "average over"; the first average is over prompts and sampled answers, the second over ordinary pretraining text $$x$$;
- $$\gamma$$ (gamma) weights the pretraining term $$\log \pi^{RL}(x)$$, which rewards the model for still predicting normal text well. $$\gamma = 0$$ gives the "PPO" models; $$\gamma > 0$$ gives "PPO-ptx".

> [!DEFINITION] KL divergence
> A measure of how different two probability distributions are: $$\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}}) = \sum_y \pi(y) \log \frac{\pi(y)}{\pi_{\text{ref}}(y)}$$. It is 0 when they are identical and grows as $$\pi$$ puts probability where $$\pi_{\text{ref}}$$ does not. Its unit is the **nat** (natural-log units). Averaging the log-ratio over answers sampled from $$\pi$$ gives exactly this sum, which is why the penalty in the objective is "a KL penalty".

Why the pretraining term? Plain PPO made the model worse on some public benchmarks (the paper names SQuAD, DROP, HellaSwag and WMT French to English translation). The authors call this an "alignment tax", and mixing in pretraining gradients removed most of it.

> [!DEFINITION] Alignment tax
> A drop in some capability caused by alignment training. InstructGPT argued that a high tax would push people to deploy unaligned models, so alignment methods should keep it low.

### Worked example: reward minus beta times KL

To see what the penalty does, take a toy that is small enough to solve exactly. One prompt has four possible answers. The SFT model writes them with probabilities $$\pi_{\text{ref}}$$ = 0.40, 0.35, 0.2499 and 0.0001, and the reward model scores them 1.0, 0.6, -0.5 and 2.5. Answer D is the trap: an odd piece of text the SFT model almost never writes, which the reward model happens to over-rate. A careful person would give it -3.0.

For this objective (without the $$\gamma$$ term) the best possible policy is known in closed form (Ziegler et al. 2019 use it; the DPO paper derives it as its equation 4):

$$
\pi^*(y) = \frac{1}{Z}\, \pi_{\text{ref}}(y)\, e^{\,r(y)/\beta}, \qquad Z = \sum_{y'} \pi_{\text{ref}}(y')\, e^{\,r(y')/\beta}
$$

where $$Z$$ is just the number that makes the probabilities add up to 1. In words: start from the reference model's probabilities and multiply each by $$e^{r/\beta}$$. A small $$\beta$$ makes that multiplier enormous for high-reward answers, a large $$\beta$$ keeps it near 1. `ch2_kl.py` computes $$\pi^*$$ for several values of $$\beta$$:

```python
# simplified from ch2_kl.py, part A
pi_ref = np.array([0.40, 0.35, 0.2499, 0.0001])
r      = np.array([1.0, 0.6, -0.5, 2.5])        # what the reward model says
true_q = np.array([1.0, 0.6, -0.5, -3.0])       # what a careful person would say
for beta in [10, 2, 1, 0.5, 0.25, 0.1, 0.05]:
    w = pi_ref * np.exp(r / beta)                # reference probability times exp(reward / beta)
    pi = w / w.sum()                             # divide by Z
    print(beta, pi, pi @ r, kl(pi, pi_ref), pi @ r - beta * kl(pi, pi_ref), pi @ true_q)
```

```text
 beta | pi*(A)  pi*(B)  pi*(C)  pi*(D) | E[reward]  KL(nats)  objective | E[true quality]
   10 | 0.420  0.353  0.226  0.000 |   +0.520    0.002     +0.503   |   +0.519
    2 | 0.497  0.356  0.147  0.000 |   +0.638    0.036     +0.566   |   +0.637
    1 | 0.579  0.340  0.081  0.001 |   +0.744    0.114     +0.630   |   +0.740
  0.5 | 0.700  0.275  0.022  0.004 |   +0.863    0.284     +0.720   |   +0.843
 0.25 | 0.782  0.138  0.001  0.079 |   +1.061    0.915     +0.832   |   +0.628
  0.1 | 0.001  0.000  0.000  0.999 |   +2.498    9.190     +1.579   |   -2.995
 0.05 | 0.000  0.000  0.000  1.000 |   +2.500    9.210     +2.039   |   -3.000
reference policy itself: E[reward] +0.485, KL 0, E[true quality] +0.485
```

Read it from the top. With a large $$\beta$$ (10) the policy barely moves (KL 0.002 nats). As $$\beta$$ shrinks, probability shifts toward the good answer A and away from the vague answer C, and both the reward model's score and the true quality rise together: at $$\beta = 0.5$$ the reward is +0.863 and the true quality +0.843, from +0.485 for the reference. Then, at $$\beta = 0.1$$, the policy jumps almost entirely onto D. The reward model's score climbs to +2.498, its maximum, and the true quality falls to -2.995. Moving all the probability from 0.0001 to nearly 1 costs 9.19 nats of KL, and with a weak penalty that price is worth paying *according to the reward model*.

{{FIG:ch2_kl_tradeoff|The toy from ch2_kl.py. As the KL coefficient beta gets smaller (left to right), the reward model's score keeps rising, but the true quality rises, then collapses once the policy finds the over-rated answer D. This is Stiennon's Figure 5 in miniature.}}

Now the per-answer arithmetic that PPO actually uses. During RL, each sampled answer gets the penalised reward $$R = r - \beta \log(\pi / \pi_{\text{ref}})$$. Take the policy that is optimal for $$\beta = 0.5$$:

```text
A: correct, helpful   log(pi/pi_ref) = log(0.6996/0.4000) = +0.559   r - beta*logratio = +1.00 - 0.5*(+0.559) = +0.720
B: correct, terse     log(pi/pi_ref) = log(0.2751/0.3500) = -0.241   r - beta*logratio = +0.60 - 0.5*(-0.241) = +0.720
C: vague              log(pi/pi_ref) = log(0.0218/0.2499) = -2.441   r - beta*logratio = -0.50 - 0.5*(-2.441) = +0.720
D: odd text           log(pi/pi_ref) = log(0.0035/0.0001) = +3.559   r - beta*logratio = +2.50 - 0.5*(+3.559) = +0.720
```

Look at D: its reward of 2.5 is the highest, but the policy already gives it 35 times its reference probability, so the penalty of $$0.5 \times 3.559 = 1.78$$ brings it down to 0.720. Answer C has a negative reward, but the policy has already made it 11 times less likely than the reference did, which earns it a *bonus* of 1.22. At the optimum every answer ends up with the same penalised reward (0.720, which is $$\beta \log Z$$), so there is nothing left to gain by moving probability around. That balance is what "the KL term keeps the policy near the reference" means in numbers. Keep this identity in mind; it is the key step behind DPO in Section 2.8.

### The same quantity on a real model

How big are these log-ratios for real language models? `ch2_kl.py` part B measures how far Qwen2.5-0.5B-Instruct (a post-trained model) is from Qwen2.5-0.5B (its base model), the same way the penalty would. It samples 8 answers from the instruct model to one prompt and, for each, sums the token-by-token difference of log-probabilities under the two models.

```python
# simplified from ch2_kl.py, part B
@torch.no_grad()
def token_logps(model, ids, start):
    logits = model(torch.tensor([ids])).logits[0, :-1]          # prediction for every next position
    lp = torch.log_softmax(logits, -1)                          # log-probabilities over the vocabulary
    return lp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])][start - 1:]   # the ones for the actual tokens

lp_pol = token_logps(pol, prompt_ids + answer_ids, len(prompt_ids))    # instruct model
lp_ref = token_logps(ref, prompt_ids + answer_ids, len(prompt_ids))    # base model
log_ratio = (lp_pol - lp_ref).sum()                             # log pi(y|x) - log pi_ref(y|x)
```

`token_logps` runs the model once over prompt plus answer, takes the log-probability it gave to each actual next token, and keeps only the answer's tokens. The log-probability of the whole answer is the sum over its tokens, so the log-ratio of the answer is the sum of per-token differences.

```text
prompt: 'Explain in two sentences why the sky is blue.' (40 tokens with the chat template)
sample 0: 52 tokens  log pi =   -90.50  log pi_ref =  -119.91  log-ratio =  29.42   "The sky appears blue because it reflects sunlight from the E..."
sample 1: 80 tokens  log pi =  -156.64  log pi_ref =  -185.80  log-ratio =  29.16   "The sky appears blue because it is covered with a large conc..."
...
average log-ratio over 8 samples (a Monte Carlo estimate of KL(pi || pi_ref) for this prompt): 29.91 nats
with beta = 0.02 (InstructGPT): penalty = 0.02 * 29.91 = 0.598 reward points per answer
```

{{FIG:ch2_kl_tokens|The first ten tokens of sample 0, with log pi minus log pi_ref for each. Most tokens are about as likely under both models; a few, like the opening "The" and "appears", are much more likely after post-training. The sum over all 52 tokens is 29.42 nats.}}

About 30 nats per answer: the instruct model finds its own answers about $$e^{30}$$ times more likely than the base model does, even though most individual tokens differ by less than 0.5. Small per-token shifts add up over a long answer. With InstructGPT's $$\beta = 0.02$$, that distance would cost 0.598 reward points per answer. (Two honest caveats: this Qwen model was not trained with exactly this objective, so this is only a measurement of distance; and these are samples at temperature 1.0 from a 0.5B model, so the physics in them is shaky. The experiment measures distance, not quality.)

{{FIG:ch2_rlhf_loop|The RLHF loop as InstructGPT ran it. The policy samples an answer; the reward model scores it; the frozen reference model gives the log-ratio for the KL penalty; PPO updates the policy toward answers with a high penalised reward. New human comparisons can be collected along the way to retrain the reward model.}}

### ChatGPT (November 30, 2022)

Eight months later OpenAI released ChatGPT as a free research preview. The announcement described its training in one paragraph.

> [!PAPER] OpenAI blog, "Introducing ChatGPT" (November 30, 2022) · Methods
> [![A paragraph from the ChatGPT announcement, highlighted: We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as InstructGPT, but with slight differences in the data collection setup. It continues: an initial model was trained with supervised fine-tuning on conversations in which human AI trainers played both sides, the user and an AI assistant, and this new dialogue dataset was mixed with the InstructGPT dataset transformed into a dialogue format](/img/training/ch2-chatgpt-blog.png)](/img/training/ch2-chatgpt-blog.png)
>
> **Context:** the "Methods" section of the announcement post.
>
> **What it says:** "We trained this model using Reinforcement Learning from Human Feedback (RLHF), using the same methods as InstructGPT", with dialogue data in which trainers "played both sides".
>
> **Why it matters:** the method was not new. What was new was the format (a conversation), the scale of public use, and how much the public saw at once what post-training does. After this, every lab needed a post-training pipeline.

### Anthropic's HH-RLHF and DeepMind's Sparrow (2022)

Two other 2022 papers ran the same recipe for dialogue assistants and added useful findings.

*Training a Helpful and Harmless Assistant with RLHF* (Bai et al., Anthropic, April 2022) trained dialogue models with separate data for helpfulness and harmlessness, and found that the two goals pull against each other: a model that refuses everything is harmless but useless. They ran "an iterated online mode of training, where preference models and RL policies are updated on a weekly cadence with fresh human feedback data", which fixes a problem from Section 2.5: as the policy improves, its outputs drift away from what the reward model was trained on, so the reward model needs fresh comparisons of the *new* outputs. They also reported "a roughly linear relation between the RL reward and the square root of the KL divergence" between the policy and its starting point. Their dataset, HH-RLHF, was released publicly and became a standard benchmark for later methods, including DPO.

*Improving Alignment of Dialogue Agents via Targeted Human Judgements* (Glaese et al., DeepMind, September 2022), the Sparrow paper, broke "good dialogue" into "natural language rules the agent should follow" and asked raters about each rule separately, which gave more targeted feedback. Sparrow also showed evidence from web search for factual claims; "evidence provided by Sparrow supports the sampled response 78% of the time", and it broke its rules "only 8% of the time" under adversarial probing.

The cost of all this was people. Every comparison was a person reading two long answers. Harmlessness data in particular meant people reading harmful text. And PPO itself was hard: four models in memory, sampling inside the training loop, and many settings to tune. The next year attacked both problems.

## 2.8 2023: cheaper labels, open models, and RLHF without RL

### Constitutional AI: feedback from a model (December 2022)

*Constitutional AI: Harmlessness from AI Feedback* (Bai et al., Anthropic, December 2022) asked: if the hard, unpleasant part of labelling is judging harmful outputs, can a model do that judging, guided by a short written list of principles? "The only human oversight is provided through a list of rules or principles, and so we refer to the method as 'Constitutional AI'."

> [!PAPER] Bai et al. (2022), Constitutional AI · Figure 1 · page 2
> [![Figure 1 of the Constitutional AI paper. Top row, the supervised stage: a helpful RLHF model generates responses to red-teaming prompts, then each response goes through critique and revision, giving a fine-tuned SL-CAI model. Bottom row, the RL stage: the SL-CAI model generates pairs of responses to red-teaming prompts, constitutional AI feedback for self-improvement chooses between them, a preference model is fine-tuned on these choices, and RLAIF training with the preference model gives the final RL-CAI model. Highlighted: supervised learning (SL) stage, Reinforcement Learning (RL) stage](/img/training/ch2-cai-fig1.png)](/img/training/ch2-cai-fig1.png)
>
> **Context:** the overview figure on page 2.
>
> **What it says:** in the supervised stage, the model critiques and revises its own harmful answers, and is fine-tuned on the revisions. In the RL stage, a model compares pairs of answers according to the principles, a preference model is trained on those AI comparisons, and RL proceeds as in RLHF. The paper calls this "RL from AI Feedback" (RLAIF).
>
> **Why it matters:** the pipeline did not change; the labeler did. Feedback became cheap and fast, and the rules the model follows became an explicit, readable document instead of the hidden average of many raters' judgements.

{{FIG:ch2_rlaif|RLHF and RLAIF side by side: the same steps, with a model following written principles in place of the human comparing two answers.}}

### Open base models and cheap SFT data

Until 2023 almost all of this happened inside a few companies. Then **LLaMA** (Touvron et al., Meta, February 2023) released strong base models from 7B to 65B parameters, trained only on public data. The paper reports that "LLaMA-13B outperforms GPT-3 (175B) on most benchmarks", following the Chinchilla lesson of training smaller models on more tokens. Suddenly anyone with a few GPUs could do post-training research.

The first thing people did was SFT with cheap data. **Self-Instruct** (Wang et al., December 2022) showed that a model can write its own training data: start from 175 seed tasks, ask the model to write new instructions, inputs and outputs, filter out bad or duplicate ones, and fine-tune on the result. On GPT-3 this produced "about 52k instructions" and "a 33% absolute improvement over the original model on Super-NaturalInstructions". In March 2023, Stanford's **Alpaca** fine-tuned LLaMA 7B "on 52K instruction-following demonstrations generated in the style of self-instruct using text-davinci-003" (an OpenAI instruct model), and the result behaved, in the authors' words, "qualitatively similarly" to that model on single-turn instructions.

How much SFT data do you actually need? **LIMA** (Zhou et al., Meta, May 2023) fine-tuned the 65B LLaMA on "only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling".

> [!PAPER] Zhou et al. (2023), LIMA: Less Is More for Alignment · Abstract · page 1
> [![The end of the LIMA abstract: in a controlled human study, responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43% of cases, 58% compared to Bard and 65% versus DaVinci003. Highlighted: almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary to teach models to produce high quality output](/img/training/ch2-lima-abstract.png)](/img/training/ch2-lima-abstract.png)
>
> **Context:** the abstract, after the human evaluation results.
>
> **What it says:** "almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary". The paper calls this the *Superficial Alignment Hypothesis*: SFT mostly teaches the format and style of answering.
>
> **Why it matters:** it changed where effort goes. A thousand excellent examples can beat fifty thousand mediocre ones. Quality of SFT data matters far more than quantity.

### Llama 2: RLHF in the open (July 2023)

*Llama 2: Open Foundation and Fine-Tuned Chat Models* (Touvron et al., Meta, July 2023) published the most detailed open RLHF recipe of the time. Meta collected "over 1 million binary comparisons" (Table 6 lists 1,418,091 Meta comparisons), trained *two* reward models, one for helpfulness and one for safety, and ran five rounds of RLHF.

> [!PAPER] Touvron et al. (2023), Llama 2 · Figure 4 · page 5
> [![Figure 4 of the Llama 2 paper: pretraining data leads to Llama 2 through self-supervised learning, then supervised fine-tuning gives Llama-2-chat; human preference data trains a safety reward model and a helpful reward model; in the fine-tuning box, rejection sampling and proximal policy optimization form the RLHF loop that improves Llama-2-chat, with feedback returning to human preference collection. Highlighted: rejection sampling and Proximal Policy Optimization](/img/training/ch2-llama2-fig4.png)](/img/training/ch2-llama2-fig4.png)
>
> **Context:** the overview of the chat model's training.
>
> **What it says:** after SFT, the model is "iteratively refined using Reinforcement Learning with Human Feedback (RLHF) methodologies, specifically through rejection sampling and Proximal Policy Optimization (PPO)", with reward modelling data collected in parallel.
>
> **Why it matters:** it combined WebGPT's best-of-N (as a training step: sample K answers, fine-tune on the best) with InstructGPT's PPO, and showed the iteration of Anthropic's weekly updates at scale. It is also a reminder of the cost: two reward models, many rounds, a million comparisons.

### DPO: the reward model was hiding in the policy (May 2023)

PPO-based RLHF worked, but it was heavy. During step 3 you hold four models in memory (policy, reference, reward model, value model), you generate text inside the training loop, and the result is sensitive to many settings. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (Rafailov, Sharma, Mitchell, Ermon, Manning and Finn, Stanford, May 2023) showed that you can skip the reward model and the RL entirely.

> [!PAPER] Rafailov et al. (2023), DPO · Figure 1 · page 2
> [![Figure 1 of the DPO paper. Left, RLHF: preference data, for example two poems about the history of jazz with one preferred, trains a reward model by maximum likelihood; the reward model labels rewards for completions sampled from the LM policy, which is trained by reinforcement learning. Right, DPO: the same preference data trains the final LM directly by maximum likelihood. Caption highlighted: DPO optimizes for human preferences while avoiding reinforcement learning](/img/training/ch2-dpo-fig1.png)](/img/training/ch2-dpo-fig1.png)
>
> **Context:** the first figure of the paper.
>
> **What it says:** "DPO optimizes for human preferences while avoiding reinforcement learning." Instead of fitting a reward model and then running RL, it fits the policy directly with "a simple classification objective".
>
> **Why it matters:** RLHF's step 2 and step 3 collapse into one supervised training run on a fixed dataset of pairs.

**Where DPO comes from, in three steps.** You already have all the pieces.

1. Section 2.7 gave the best policy for "reward minus $$\beta$$ times KL": $$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x)\, e^{r(x,y)/\beta}$$.
2. Take the log of both sides and solve for the reward: $$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$. Any policy defines an "implicit reward" this way. (This is the identity the toy showed: at the optimum every answer had the same penalised reward, $$\beta \log Z$$.)
3. Put this reward into the Bradley-Terry formula from Section 2.4. Bradley-Terry only uses the *difference* $$r(x, y_w) - r(x, y_l)$$, and both answers share the same prompt, so the awkward $$\beta \log Z(x)$$ cancels. What is left contains only the policy and the reference model.

> [!PAPER] Rafailov et al. (2023), DPO · Section 4, equation 7 · page 4
> [![Equation 7 of the DPO paper: L DPO of pi theta and pi ref equals minus the expectation over x, y w, y l drawn from D of log sigma of beta log pi theta of y w given x over pi ref of y w given x, minus beta log pi theta of y l given x over pi ref of y l given x. Highlighted: our policy objective becomes. Below: this way, we fit an implicit reward using an alternative parameterization, whose optimal policy is simply pi theta](/img/training/ch2-dpo-eq7.png)](/img/training/ch2-dpo-eq7.png)
>
> **Context:** Section 4, right after the derivation above.
>
> **What it says:** the DPO loss. "We fit an implicit reward using an alternative parameterization, whose optimal policy is simply $$\pi_\theta$$."
>
> **Why it matters:** this one line replaces the reward model training and the PPO run.

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\,\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \Big[ \log \sigma \Big( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \Big) \Big]
$$

where:

- $$\mathcal{D}$$ is a fixed dataset of preference triples: a prompt $$x$$, a chosen answer $$y_w$$ and a rejected answer $$y_l$$;
- $$\pi_\theta$$ is the model being trained, with weights $$\theta$$, and $$\pi_\theta(y \mid x)$$ is the probability it gives to the whole answer;
- $$\pi_{\text{ref}}$$ is a frozen copy of the starting model (usually the SFT model);
- $$\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$ is the **implicit reward** of answer $$y$$: how much more likely the trained model makes it than the reference did, scaled by $$\beta$$;
- the difference of the two implicit rewards is the **margin**; $$\sigma$$ and $$-\log$$ are exactly the Bradley-Terry loss of Section 2.4;
- $$\beta$$ plays the role of the KL coefficient; the DPO paper's default is 0.1.

So DPO is the reward-model loss from Section 2.4, with the reward model replaced by "log-probability under the policy minus log-probability under the reference". Per pair, it needs only four numbers: $$\log \pi_\theta(y_w|x)$$, $$\log \pi_{\text{ref}}(y_w|x)$$, $$\log \pi_\theta(y_l|x)$$ and $$\log \pi_{\text{ref}}(y_l|x)$$. No sampling, no reward model, no value model.

{{FIG:ch2_ppo_vs_dpo|What each method needs during training. PPO-based RLHF keeps four networks and generates text at every step; DPO keeps two and reads a fixed file of pairs, like ordinary supervised fine-tuning.}}

### Experiment: a tiny DPO run on a real model

`ch2_dpo.py` runs DPO on Qwen2.5-0.5B-Instruct. To make the effect easy to see, the preference is a pure style preference: in each of 16 training pairs, the chosen answer starts with "In short:" and a one-line summary, and the rejected answer has the same explanation without it. The content is identical, so the only thing to learn is the style. Then we ask 8 *new* questions and count how many answers start with "In short:".

```python
# simplified from ch2_dpo.py
policy = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
reference = copy.deepcopy(policy).eval()                 # frozen copy: pi_ref
BETA = 0.1

def seq_logp(model, q, answer):
    p = chat_ids(q)                                      # the question in the chat template
    a = tok(answer + '<|im_end|>')['input_ids']          # the answer tokens, plus the end-of-turn token
    logits = model(torch.tensor([p + a])).logits[0, len(p) - 1:-1]
    return torch.log_softmax(logits, -1).gather(1, torch.tensor(a)[:, None]).sum()   # log pi(answer | question)

def dpo_terms(q, yw, yl):
    lw, ll = seq_logp(policy, q, yw), seq_logp(policy, q, yl)
    with torch.no_grad():
        rw, rl = seq_logp(reference, q, yw), seq_logp(reference, q, yl)
    margin = BETA * ((lw - rw) - (ll - rl))              # implicit reward of chosen minus rejected
    return -F.logsigmoid(margin), margin

opt = torch.optim.AdamW(policy.parameters(), lr=1e-6)
for step in range(40):
    batch = random.sample(train, 4)                      # 4 pairs per step
    loss = sum(dpo_terms(*b)[0] for b in batch) / 4
    opt.zero_grad(); loss.backward(); opt.step()
```

Block by block:

- The reference is an exact copy of the starting model, frozen. At step 0 the policy and reference are identical, so every log-ratio is 0, the margin is 0 and the loss is $$-\log \sigma(0) = \ln 2 = 0.693$$, the same coin-flip value as the untrained reward model in Section 2.4.
- `seq_logp` scores an answer: one forward pass over question plus answer, then the sum of the log-probabilities of the answer's tokens. This is the only thing DPO asks of a model.
- `dpo_terms` is equation 7 for one pair. The reference model runs under `torch.no_grad()` because it is never trained.
- The loop is an ordinary supervised training loop: a batch of pairs, a loss, a step. There is no generation anywhere.

[![Terminal output of ch2_dpo.py: the four log-probabilities before training, then the training log for steps 0 to 40 with loss falling from 0.6931 to 0.0012 and the mean margin rising to 7.36, the final numbers for pair 0, and the eight held-out answers, all now starting with In short](/img/training/ch2_dpo-run.png)](/img/training/ch2_dpo-run.png)

```text
held-out questions answered with "In short:" BEFORE training: 0/8
  e.g. 'What is the capital of Italy?' -> 'The capital of Italy is Rome.'
step  0  train loss 0.6931  pairs ranked right 0.50  mean margin +0.000   pair 0: log pi(y_w) -57.33, log pi(y_l) -40.67
step  5  train loss 0.1039  pairs ranked right 1.00  mean margin +2.754   pair 0: log pi(y_w) -42.73, log pi(y_l) -70.66
step 20  train loss 0.0058  pairs ranked right 1.00  mean margin +5.727   pair 0: log pi(y_w) -32.12, log pi(y_l) -79.34
step 40  train loss 0.0012  pairs ranked right 1.00  mean margin +7.356   pair 0: log pi(y_w) -30.04, log pi(y_l) -89.94
held-out questions answered with "In short:" AFTER training: 8/8
```

**Worked numbers for one pair.** Take pair 0, "What is the capital of Japan?", after training:

- chosen: $$\log \pi_\theta(y_w|x) = -30.045$$, $$\log \pi_{\text{ref}}(y_w|x) = -57.331$$, so the implicit reward is $$0.1 \times (-30.045 + 57.331) = 0.1 \times 27.286 = +2.729$$;
- rejected: $$\log \pi_\theta(y_l|x) = -89.942$$, $$\log \pi_{\text{ref}}(y_l|x) = -40.671$$, so the implicit reward is $$0.1 \times (-49.271) = -4.927$$;
- margin: $$2.729 - (-4.927) = 7.656$$; $$\sigma(7.656) = 0.99953$$; loss $$= -\log 0.99953 = 0.0005$$.

{{FIG:ch2_dpo_numbers|Equation 7 on pair 0 after training. Only four log-probabilities are needed. The policy now finds the chosen answer e to the 27.3 times more likely than the reference does, and the rejected answer e to the 49.3 times less likely.}}

Notice something in the "before" numbers: the reference model gave the chosen answer a *lower* log-probability (-57.33) than the rejected one (-40.67), simply because it is longer. DPO does not care. It never compares the two answers' raw probabilities; it compares how much each one *moved* relative to the reference.

{{FIG:ch2_dpo_run|Log-probabilities of the chosen and rejected answer of pair 0 during the 40 steps. Most of the change happens in the first 5 steps. The loss falls from 0.693 to 0.0012, and all 8 held-out answers now start with "In short:".}}

And the held-out answers:

```text
  What is the capital of Italy?              -> 'In short: Rome. In detail: The capital city of Italy is Rome (or "Lucca"'
  How many days are in a leap year?          -> 'In short: 366. In detail: A common year has 365 days (28 or 29 days in F'
  Who painted the Mona Lisa?                 -> 'In short: Leonardo da Vinci. In detail: The painting was done in 1503-15'
```

The preference generalised: 0 of 8 new questions before, 8 of 8 after, from 16 pairs and 40 small steps. But look closer and the weaknesses of a tiny DPO run are visible too. The model invented an "In detail:" label that is not in any training pair, and the first answer now contains a made-up claim ("or Lucca"). 40 steps of 4 pairs is 10 passes over 16 pairs, which is heavy over-training on a tiny dataset, and nothing in the loss checks facts. DPO learns *what distinguishes* chosen from rejected, and it can pick up side effects along the way. Real DPO runs use tens of thousands of varied pairs, one or a few passes, and careful evaluation.

> [!WARNING]
> A DPO loss near zero does not mean a good model. It means the training pairs are separated. In our run the loss reached 0.0012 while the model started to invent text. Always read samples and evaluate on held-out prompts (Chapter 3 is about this).

**DPO goes open.** Within months DPO became the default for open models. *Zephyr* (Tunstall et al., Hugging Face, October 2023) used preference data ranked by a stronger model (AI feedback again) and "distilled direct preference optimization (dDPO)"; the 7B result "surpasses Llama2-Chat-70B, the best open-access RLHF-based model" on the MT-Bench chat benchmark, and "requires no human annotation". *Tulu 2* (Ivison et al., Allen Institute for AI, November 2023) released "the largest DPO-trained model to date", a 70B model.

What did DPO leave unsolved? It learns from a *fixed* set of pairs, usually written by other models, so it never gets feedback on its own new outputs the way online RL does. And like RLHF, it learns from preferences, which are fuzzy. For some tasks there is something better than a preference: a right answer.

## 2.9 2024 to 2025: rewards you can check, and models that think longer

### Outcome or process?

If the task is a maths problem with a known answer, you do not need a person or a reward model to say whether the final answer is right: you can check it. But a solution can reach the right answer by luck, or go wrong in step 3 of 10. *Let's Verify Step by Step* (Lightman et al., OpenAI, May 2023) compared two kinds of reward models for maths: one that judges only the final answer (outcome supervision) and one that judges every step (process supervision).

> [!PAPER] Lightman et al. (2023), Let's Verify Step by Step · Abstract · page 1
> [![Part of the abstract: We conduct our own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset. Our process-supervised model solves 78% of problems from a representative subset of the MATH test set. To support related research, we also release PRM800K, the complete dataset of 800,000 step-level human feedback labels used to train our best reward model. The phrases process supervision significantly outperforms outcome supervision and PRM800K are highlighted](/img/training/ch2-lightman-abstract.png)](/img/training/ch2-lightman-abstract.png)
>
> **Context:** the abstract.
>
> **What it says:** "process supervision significantly outperforms outcome supervision" when the reward model is used to pick the best of many solutions to MATH problems; the step labels were released as "PRM800K", 800,000 human labels on individual steps.
>
> **Why it matters:** it established that the *kind* of reward signal matters for reasoning. Process labels are expensive, though, and the next papers went back to the cheapest signal of all: is the final answer right?

> [!DEFINITION] Outcome reward and process reward
> An outcome reward scores only the final result (right or wrong). A process reward scores each intermediate step. Process rewards give more detailed feedback but need step-level labels or a model that can judge steps.

### DeepSeekMath and GRPO (February 2024)

*DeepSeekMath* (Shao et al., DeepSeek, February 2024) trained a 7B maths model ("51.7% on the competition-level MATH benchmark") and introduced a lighter RL algorithm. PPO needs a value model, usually as large as the policy, to estimate how good an answer is *expected* to be, so that it can tell whether a particular answer was better or worse than expected. GRPO gets that baseline in a simpler way: sample several answers to the same question and compare each one to the others.

> [!PAPER] Shao et al. (2024), DeepSeekMath · Figure 4 · page 13
> [![Figure 4 of DeepSeekMath. Top, PPO: a question q goes to the policy model, which produces an output o; a reference model, reward model and value model process it; the KL term and reward give r, the value model gives v, and GAE computes the advantage A. Bottom, GRPO: the policy produces a group of outputs o1 to oG; the reference and reward models give rewards r1 to rG; group computation turns them into advantages A1 to AG. Trained models are marked in yellow, frozen models in blue. Caption highlighted: GRPO foregoes the value model](/img/training/ch2-grpo-fig4.png)](/img/training/ch2-grpo-fig4.png)
>
> **Context:** Section 4.1, where GRPO is introduced.
>
> **What it says:** "GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources."
>
> **Why it matters:** one fewer large network to train and hold in memory. The cost moves to sampling: several answers per question (64 in the paper).

> [!DEFINITION] Advantage
> How much better an action turned out than expected: reward minus a baseline. RL makes actions with a positive advantage more likely and actions with a negative advantage less likely. Using an advantage instead of the raw reward tells the model which answers were *better than usual*, not just which ones got points.

For outcome rewards, the paper's advantage is one line:

> [!PAPER] Shao et al. (2024), DeepSeekMath · Section 4.1.2 · page 14
> [![A paragraph from Section 4.1.2: a reward model scores the G outputs, giving rewards r1 to rG. Highlighted: these rewards are normalized by subtracting the group average and dividing by the group standard deviation. Outcome supervision sets the advantages of all tokens in output i to r i minus mean of r, divided by std of r](/img/training/ch2-grpo-adv.png)](/img/training/ch2-grpo-adv.png)
>
> **Context:** "Outcome Supervision RL with GRPO".
>
> **What it says:** rewards "are normalized by subtracting the group average and dividing by the group standard deviation", and every token of output $$i$$ gets that normalised reward as its advantage.
>
> **Why it matters:** the baseline is no longer a network's guess; it is simply how the other answers to the same question did.

$$
\hat{A}_i = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G)}
$$

where:

- $$G$$ is the group size, the number of answers sampled for one question (64 in DeepSeekMath, 4 in our example);
- $$r_i$$ is the reward of answer $$i$$; with a checker it is 1 if the final answer is right and 0 if not;
- $$\operatorname{mean}$$ and $$\operatorname{std}$$ are the average and the standard deviation of the $$G$$ rewards;
- $$\hat{A}_i$$ is the advantage given to every token of answer $$i$$.

The policy is then updated with a PPO-style clipped objective using these advantages, plus a KL penalty to a reference model (coefficient 0.04 in the paper). Simplified, the update raises the log-probability of each answer in proportion to its advantage.

**Worked example.** Four answers to one question, two right and two wrong: rewards $$(1, 0, 0, 1)$$. The mean is 0.5. The standard deviation is $$\sqrt{\tfrac{1}{4}(0.25 + 0.25 + 0.25 + 0.25)} = 0.5$$. The advantages are $$(1 - 0.5)/0.5 = +1$$ for the right answers and $$(0 - 0.5)/0.5 = -1$$ for the wrong ones. The script checks this and some other groups:

```python
# from ch2_grpo.py
def advantages(r, eps=1e-4):
    r = np.asarray(r, dtype=float)
    return (r - r.mean()) / (r.std() + eps)        # population std (divide by G); eps avoids 0/0
```

```text
2 right, 2 wrong  rewards [1, 0, 0, 1]  mean 0.500  std 0.500  ->  advantages [1.0, -1.0, -1.0, 1.0]
1 right, 3 wrong  rewards [1, 0, 0, 0]  mean 0.250  std 0.433  ->  advantages [1.732, -0.577, -0.577, -0.577]
3 right, 1 wrong  rewards [1, 1, 0, 1]  mean 0.750  std 0.433  ->  advantages [0.577, 0.577, -1.732, 0.577]
all right         rewards [1, 1, 1, 1]  mean 1.000  std 0.000  ->  advantages [0.0, 0.0, 0.0, 0.0]
all wrong         rewards [0, 0, 0, 0]  mean 0.000  std 0.000  ->  advantages [0.0, 0.0, 0.0, 0.0]
graded rewards    rewards [0.9, 0.2, 0.5, 0.0]  mean 0.400  std 0.339  ->  advantages [1.474, -0.59, 0.295, -1.179]
```

Three lessons hide in these lines. First, a rare success is rewarded strongly: when only 1 of 4 is right, it gets +1.732, while each wrong answer gets only -0.577. Second, the advantages in a group always add up to zero, so the group pushes some answers up exactly as much as it pushes others down. Third, **if every answer gets the same reward, every advantage is 0 and the model learns nothing from that question**. Questions that are too easy or too hard are wasted compute. (Implementations differ in small ways: with the sample standard deviation, dividing by $$G - 1$$, the 2-right case gives $$\pm 0.866$$ instead of $$\pm 1$$.)

### Experiment: real groups and one gradient step

`ch2_grpo.py` part B does this with a real model. For 4 maths word problems, it samples $$G = 4$$ answers from Qwen2.5-0.5B-Instruct (temperature 0.8), checks the final number with a regular expression (the content of the last `\boxed{}`, or else the last number in the text), and computes the advantages. Then it takes one gradient step that raises each answer's average token log-probability in proportion to its advantage.

```python
# simplified from ch2_grpo.py
def final_number(text):                       # the checker: no reward model, just a rule
    box = re.findall(r'\\boxed\{([^}]*)\}', text)
    nums = re.findall(r'-?\d[\d,]*\.?\d*', box[-1] if box else text)
    return float(nums[-1].replace(',', '')) if nums else None

out = model.generate(prompt_ids_repeated_4_times, do_sample=True, temperature=0.8, max_new_tokens=300)
rewards = [float(final_number(text) == gold) for text in decoded]
adv = advantages(rewards)

loss = -sum(a * mean_logp(q, answer) for a, answer in zip(adv, answers) if a != 0) / n   # policy-gradient step
loss.backward(); opt.step()
```

`final_number` is the whole "reward model": a few lines of code that cannot be flattered or fooled by style. The loss is the simplest policy-gradient form: on the first step after sampling, PPO's probability ratio is exactly 1, so its clipping does nothing, and the KL to the reference is 0, so both are left out of this one-step demo.

[![Terminal output of ch2_grpo.py: the advantage table for hand-picked rewards, then four questions with four sampled answers each, their final answers, rewards and advantages, then the change in mean log-probability per token after one gradient step for all sixteen answers](/img/training/ch2_grpo-run.png)](/img/training/ch2_grpo-run.png)

```text
Q: Tom has 3 boxes with 12 apples in each box. He gives away 10 apples. ...       rewards [1.0, 1.0, 1.0, 1.0]  mean 1.00  std 0.00
Q: A train travels at 60 km per hour for 2.5 hours. ...                            rewards [1.0, 1.0, 1.0, 1.0]  mean 1.00  std 0.00
Q: What is 17 times 23?                                                            rewards [1.0, 1.0, 1.0, 1.0]  mean 1.00  std 0.00
Q: A book costs 8 dollars and a pen costs 3 dollars. How much do 4 books and 5 pens cost in total?   (correct answer 47)
  o1: 154 tokens  final answer 47.0  reward 1  advantage +0.577
  o2: 197 tokens  final answer 47.0  reward 1  advantage +0.577
  o3: 228 tokens  final answer 47.0  reward 1  advantage +0.577
  o4: 210 tokens  final answer 23.0  reward 0  advantage -1.732   "...the total cost of 4 books and 5 pens is: \[ \boxed{23} \]"
```

(The lines are shortened here; the full log is in the screenshot.) Three of the four questions were too easy: all four answers were right, every advantage was 0, and those 12 answers contribute nothing to the update. Only the books-and-pens question produced a mixed group.

{{FIG:ch2_grpo_group|The one mixed group from ch2_grpo.py. Four answers, checked against the key 47: three right, one wrong (23). Mean reward 0.75, standard deviation 0.43, so the right answers get +0.577 and the wrong one gets -1.732.}}

After one step with learning rate $$2 \times 10^{-6}$$:

```text
average change: positive-advantage answers +0.0434, negative-advantage answers -0.2560, zero-advantage answers -0.0049
```

{{FIG:ch2_grpo_step|What one gradient step did to each answer's average log-probability per token. The three right answers became more likely, the wrong answer much less likely. The 12 answers with zero advantage barely moved: they only shift because all answers share the same weights.}}

The wrong answer's average log-probability per token fell from -0.171 to -0.427, so over its 210 tokens it became about $$e^{0.256 \times 210} \approx e^{54}$$ times less likely. That is a big change from one step of a tiny learning rate; real runs use smaller steps and many more questions, and the clipping and KL terms we left out are there to keep such jumps in check.

> [!TIP]
> When you read about GRPO-style training, look for how the authors chose the questions. Since all-right and all-wrong groups teach nothing, a good RL dataset has questions the current model solves *sometimes*. Many later recipes filter questions by the model's current pass rate for this reason.

### OpenAI o1 (September 2024)

On September 12, 2024, OpenAI announced o1, "a new large language model trained with reinforcement learning to perform complex reasoning", which "thinks before it answers" by producing a long internal chain of thought.

> [!PAPER] OpenAI blog, "Learning to reason with LLMs" (September 12, 2024)
> [![A paragraph from the post, highlighted: Our large-scale reinforcement learning algorithm teaches the model how to think productively using its chain of thought in a highly data-efficient training process. It continues: the performance of o1 consistently improves with more reinforcement learning (train-time compute) and with more time spent thinking (test-time compute)](/img/training/ch2-o1-blog.png)](/img/training/ch2-o1-blog.png)
>
> **Context:** the opening section of the announcement.
>
> **What it says:** RL "teaches the model how to think productively using its chain of thought", and performance "consistently improves with more reinforcement learning (train-time compute) and with more time spent thinking (test-time compute)".
>
> **Why it matters:** it named a new axis for scaling: not only bigger pretraining, but more RL and longer thinking at answer time. The post did not describe the algorithm or the rewards, so the research community did not know exactly how it was done.

> [!DEFINITION] Chain of thought
> Intermediate reasoning written out as text before the final answer ("first compute 4 times 8, then ..."). Reasoning models are trained so that writing a longer, more careful chain of thought leads to more correct answers.

### Tulu 3 and RLVR (November 2024)

The open *Tulu 3* recipe (Lambert et al., Allen Institute for AI, November 2024) gave the checkable-reward idea a name: Reinforcement Learning with Verifiable Rewards.

> [!PAPER] Lambert et al. (2024), Tulu 3 · Figure 18 · page 31
> [![Figure 18 of Tulu 3: training data provides prompts to the policy, the policy produces completions, a verifiable reward box gives r equal to a constant if correct and 0 otherwise, and the scalar reward drives a policy update. Caption highlighted: If the answer is verifiably correct, we provide reward of alpha, otherwise 0](/img/training/ch2-tulu3-rlvr.png)](/img/training/ch2-tulu3-rlvr.png)
>
> **Context:** Section 6, which introduces RLVR as the last stage of the Tulu 3 pipeline, after SFT and DPO.
>
> **What it says:** completions are checked "using a deterministic function. If the answer is verifiably correct, we provide reward of $$\alpha$$, otherwise 0" ($$\alpha = 10$$ in the paper), and the policy is trained against this reward with PPO, still with a KL penalty.
>
> **Why it matters:** the reward model of Section 2.4 is replaced by a program. It cannot be over-optimised in the way of Stiennon's Figure 5 because a wrong answer never scores, though the model can still exploit a loose checker (for example one that only reads the last number).

{{FIG:ch2_judge_vs_checker|Two sources of reward. A learned reward model works for any task but is an imperfect copy of people and can be gamed; a checking program is exact and cheap but only exists where answers can be verified.}}

### DeepSeek-R1 and R1-Zero (January 2025)

*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (DeepSeek-AI, January 22, 2025) put the pieces together in the open. Its first model, **DeepSeek-R1-Zero**, "applies RL directly to the base model without any SFT data". The algorithm is GRPO. The reward is "a rule-based reward system" with two parts: accuracy rewards (is the final answer right, for example "within a box", or do the tests pass for code) and format rewards (put the reasoning between `<think>` and `</think>` tags). The authors did not use a neural reward model because "the neural reward model may suffer from reward hacking in the large-scale reinforcement learning process".

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1 (arXiv v1) · Section 2.2.4 · page 6
> [![A paragraph from the R1 paper: Figure 2 depicts the performance trajectory of DeepSeek-R1-Zero on the AIME 2024 benchmark throughout the RL training process. Highlighted: jumping from an initial 15.6% to an impressive 71.0%, reaching performance levels comparable to OpenAI-o1-0912](/img/training/ch2-r1-aime.png)](/img/training/ch2-r1-aime.png)
>
> **Context:** the results of RL on the base model alone.
>
> **What it says:** on the AIME 2024 maths competition, the average pass@1 score rose "from an initial 15.6% to an impressive 71.0%" during RL; with majority voting over 64 samples (cons@64) it reached 86.7%.
>
> **Why it matters:** a base model, a checker and GRPO, with no human-written reasoning examples, produced a model at the level of o1-0912 on this benchmark.

The most striking result was *how* the model got there. Nobody told it to think longer. The reward only checked the final answer. But the answers grew longer and longer during training.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1 (arXiv v1) · Figure 3 · page 8
> [![Figure 3 of the R1 paper: average length per response of DeepSeek-R1-Zero during training, on the vertical axis from 0 to 12,000 tokens, against training steps from 0 to about 8,000. The line rises steadily from a few hundred tokens to nearly 10,000, with noisy variation. Caption highlighted: naturally learns to solve reasoning tasks with more thinking time](/img/training/ch2-r1-length.png)](/img/training/ch2-r1-length.png)
>
> **Context:** the "self-evolution" part of Section 2.2.4.
>
> **What it says:** the average response grew from a few hundred tokens to nearly 10,000 over about 8,000 RL steps. "DeepSeek-R1-Zero naturally learns to solve reasoning tasks with more thinking time."
>
> **Why it matters:** longer reasoning was not taught; it was *discovered*, because on hard problems longer, more careful chains of thought reached the right answer more often, and GRPO rewards whatever reaches the right answer.

They also reported a moment where the model, in the middle of a solution, stopped and re-checked its own work.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1 (arXiv v1) · Table 3 · page 9
> [![Table 3 of the R1 paper: a question about the sum of the real solutions of the square root of a minus the square root of a plus x equals x. The response starts squaring both sides, then writes, highlighted: Wait, wait. Wait. That's an aha moment I can flag here. Then: let's reevaluate this step-by-step. Caption highlighted: An interesting aha moment of an intermediate version of DeepSeek-R1-Zero](/img/training/ch2-r1-aha.png)](/img/training/ch2-r1-aha.png)
>
> **Context:** the "Aha Moment of DeepSeek-R1-Zero" paragraph.
>
> **What it says:** an intermediate checkpoint writes "Wait, wait. Wait. That's an aha moment I can flag here." and re-evaluates its steps. The authors: "rather than explicitly teaching the model on how to solve a problem, we simply provide it with the right incentives".
>
> **Why it matters:** self-checking behaviour appeared under a reward that only checked final answers. Treat the single example with some care, though: later work (for example the "There may not be aha moment in R1-Zero-like training" study the authors themselves cite in a later version) found that such "wait" moments can already appear in base models, so how much is new and how much is amplified is still discussed.

R1-Zero had problems a user would notice: the paper lists "poor readability, and language mixing". The released **DeepSeek-R1** fixed them with a multi-stage recipe: a small amount of "cold-start" SFT data with readable long reasoning, then reasoning RL as for R1-Zero, then new SFT data made partly by rejection sampling from that RL model, then a final RL stage for all kinds of prompts. In other words, the 2025 pipeline is SFT, RL, SFT, RL: the old pieces, rearranged around a new kind of reward.

> [!NOTE]
> The R1 paper was later expanded (arXiv v2, January 2026, matching the version published in Nature in 2025) with more training details and somewhat different numbers, for example 77.9% AIME pass@1 for R1-Zero. The screenshots here are from the January 2025 version, the one that started the discussion.

## 2.10 What changed, and what stayed the same

Step back and the eight years look like one long refinement of a single question: *where does the training signal come from?*

| Year | Method | Training signal | Human data | Models in memory |
|---|---|---|---|---|
| 2018 to 2020 | pretraining (GPT-1 to GPT-3) | next token of web text | none (raw text) | 1 |
| 2021 | instruction tuning (FLAN, T0) | next token of a target answer | instructions and answers | 1 |
| 2017 to 2022 | RLHF with PPO (Christiano to InstructGPT) | reward model minus KL penalty | demonstrations and comparisons | 4 (policy, reference, reward, value) |
| 2022 | RLAIF (Constitutional AI) | preference model from AI comparisons | written principles, some human labels | 4 |
| 2023 | DPO | the implicit reward $$\beta \log(\pi / \pi_{\text{ref}})$$ on fixed pairs | comparisons (often model-generated) | 2 (policy, reference) |
| 2024 to 2025 | RLVR, GRPO (Tulu 3, R1) | a program that checks the answer | questions with checkable answers | 2 (policy, reference) |

Three things stayed constant through all of it.

1. **Pretraining does the heavy lifting.** LIMA's finding and R1-Zero both depend on a strong base model. Post-training draws out and shapes what pretraining put there.
2. **The Bradley-Terry formula never went away.** It is the reward-model loss of Christiano, InstructGPT and Llama 2, and it is the outer shell of DPO.
3. **The leash stayed.** Ziegler's 2019 KL penalty is in InstructGPT's equation 2, it is hidden inside DPO's log-ratios, and it is still a term in GRPO and RLVR. Every method that pushes a model toward a reward also pulls it back toward where it started.

And one thing changed fundamentally: the *meaning* of the reward. It went from "what the next word was" (pretraining), to "what a person preferred" (RLHF), to "what a model following written rules preferred" (RLAIF), to "whether the answer is actually right" (RLVR). Each change made the signal cheaper or more trustworthy, and each worked only for some tasks. Modern pipelines use all of them, in sequence.

The next chapter asks the question this history keeps raising: when a model changes, how do you know it got *better*?

> [!TAKEAWAYS]
> - A base model is trained to continue web text, which is "misaligned" with following instructions (InstructGPT). Every post-training method since is a way to supply a different training signal.
> - Three threads met in 2022: pretraining at scale (GPT-1 to GPT-3), learning a reward from human comparisons (Christiano 2017, Ziegler 2019, Stiennon 2020), and instruction tuning (FLAN, T0). InstructGPT combined them as SFT, reward model, PPO; ChatGPT used "the same methods".
> - The Bradley-Terry model $$P(A \succ B) = \sigma(r_A - r_B)$$ turns choices into scores; only reward differences matter. Our toy reward model recovered a hidden taste (correlation 0.9989) from 994 noisy choices, and reached the 0.777 accuracy ceiling set by label noise.
> - RLHF maximises reward minus $$\beta$$ times KL to the starting model. Without enough penalty, the policy finds answers the reward model over-rates: in our toy, at $$\beta = 0.1$$ the reward model's score went to +2.50 while true quality fell to -3.00 (Stiennon's Figure 5 in miniature).
> - Constitutional AI replaced human harmlessness labels with AI feedback guided by written principles; LIMA showed 1,000 excellent SFT examples go a long way; Llama 2 combined rejection sampling with PPO at scale.
> - DPO removes the reward model and the RL loop: the loss needs only four log-probabilities per pair. Our 40-step run on Qwen2.5-0.5B-Instruct took a style preference from 0 of 8 to 8 of 8 held-out answers, and also showed side effects that a near-zero loss does not reveal.
> - GRPO replaces PPO's value model with the group: advantage = (reward minus group mean) / group std. Groups where all answers get the same reward teach nothing; in our run 3 of 4 questions were like that.
> - Verifiable rewards (Tulu 3's RLVR, DeepSeek-R1) use a checking program instead of a learned judge. With GRPO on a base model, R1-Zero went from 15.6% to 71.0% on AIME 2024 and learned on its own to write much longer reasoning.

## References

**Papers**

1. C. E. Shannon. [*A Mathematical Theory of Communication*](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf). Bell System Technical Journal, 1948.
2. C. E. Shannon. [*Prediction and Entropy of Printed English*](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf). Bell System Technical Journal, 1951.
3. Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin. [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). JMLR 2003.
4. Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean. [*Efficient Estimation of Word Representations in Vector Space*](https://arxiv.org/abs/1301.3781). arXiv January 2013.
5. Ilya Sutskever, Oriol Vinyals, Quoc V. Le. [*Sequence to Sequence Learning with Neural Networks*](https://arxiv.org/abs/1409.3215). NeurIPS 2014 (arXiv September 2014).
6. Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473). ICLR 2015 (arXiv September 2014).
7. Ashish Vaswani et al. [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). NeurIPS 2017 (arXiv June 2017).
8. Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei. [*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741). NeurIPS 2017 (arXiv June 2017).
9. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347). arXiv July 2017.
10. Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever. [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). OpenAI report, June 2018.
11. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805). NAACL 2019 (arXiv October 2018).
12. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever. [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). OpenAI report, February 2019.
13. Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, Geoffrey Irving. [*Fine-Tuning Language Models from Human Preferences*](https://arxiv.org/abs/1909.08593). arXiv September 2019.
14. Jared Kaplan et al. [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361). arXiv January 2020.
15. Tom B. Brown et al. [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165). NeurIPS 2020 (arXiv May 2020).
16. Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano. [*Learning to Summarize from Human Feedback*](https://arxiv.org/abs/2009.01325). NeurIPS 2020 (arXiv September 2020).
17. Swaroop Mishra, Daniel Khashabi, Chitta Baral, Hannaneh Hajishirzi. [*Cross-Task Generalization via Natural Language Crowdsourcing Instructions*](https://arxiv.org/abs/2104.08773). ACL 2022 (arXiv April 2021).
18. Jason Wei et al. [*Finetuned Language Models Are Zero-Shot Learners*](https://arxiv.org/abs/2109.01652) (FLAN). ICLR 2022 (arXiv September 2021).
19. Victor Sanh et al. [*Multitask Prompted Training Enables Zero-Shot Task Generalization*](https://arxiv.org/abs/2110.08207) (T0). ICLR 2022 (arXiv October 2021).
20. Amanda Askell et al. [*A General Language Assistant as a Laboratory for Alignment*](https://arxiv.org/abs/2112.00861). Anthropic, arXiv December 2021.
21. Reiichiro Nakano et al. [*WebGPT: Browser-assisted Question-Answering with Human Feedback*](https://arxiv.org/abs/2112.09332). arXiv December 2021.
22. Long Ouyang et al. [*Training Language Models to Follow Instructions with Human Feedback*](https://arxiv.org/abs/2203.02155) (InstructGPT). NeurIPS 2022 (arXiv March 2022).
23. Jordan Hoffmann et al. [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) (Chinchilla). arXiv March 2022.
24. Yuntao Bai et al. [*Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*](https://arxiv.org/abs/2204.05862). Anthropic, arXiv April 2022.
25. Yizhong Wang et al. [*Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks*](https://arxiv.org/abs/2204.07705). EMNLP 2022 (arXiv April 2022).
26. Amelia Glaese et al. [*Improving Alignment of Dialogue Agents via Targeted Human Judgements*](https://arxiv.org/abs/2209.14375) (Sparrow). DeepMind, arXiv September 2022.
27. Leo Gao, John Schulman, Jacob Hilton. [*Scaling Laws for Reward Model Overoptimization*](https://arxiv.org/abs/2210.10760). arXiv October 2022.
28. Hyung Won Chung et al. [*Scaling Instruction-Finetuned Language Models*](https://arxiv.org/abs/2210.11416) (Flan-PaLM, Flan-T5). arXiv October 2022.
29. Yuntao Bai et al. [*Constitutional AI: Harmlessness from AI Feedback*](https://arxiv.org/abs/2212.08073). Anthropic, arXiv December 2022.
30. Yizhong Wang et al. [*Self-Instruct: Aligning Language Models with Self-Generated Instructions*](https://arxiv.org/abs/2212.10560). ACL 2023 (arXiv December 2022).
31. Hugo Touvron et al. [*LLaMA: Open and Efficient Foundation Language Models*](https://arxiv.org/abs/2302.13971). arXiv February 2023.
32. Chunting Zhou et al. [*LIMA: Less Is More for Alignment*](https://arxiv.org/abs/2305.11206). NeurIPS 2023 (arXiv May 2023).
33. Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn. [*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290). NeurIPS 2023 (arXiv May 2023).
34. Hunter Lightman et al. [*Let's Verify Step by Step*](https://arxiv.org/abs/2305.20050). ICLR 2024 (arXiv May 2023).
35. Hugo Touvron et al. [*Llama 2: Open Foundation and Fine-Tuned Chat Models*](https://arxiv.org/abs/2307.09288). arXiv July 2023.
36. Lewis Tunstall et al. [*Zephyr: Direct Distillation of LM Alignment*](https://arxiv.org/abs/2310.16944). arXiv October 2023.
37. Hamish Ivison et al. [*Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2*](https://arxiv.org/abs/2311.10702). arXiv November 2023.
38. Zhihong Shao et al. [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300). arXiv February 2024.
39. Nathan Lambert et al. [*Tulu 3: Pushing Frontiers in Open Language Model Post-Training*](https://arxiv.org/abs/2411.15124). arXiv November 2024.
40. DeepSeek-AI. [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/abs/2501.12948). arXiv January 2025 ([v1](https://arxiv.org/abs/2501.12948v1) is the version shown here); published in Nature 645 (2025).

**Other sources**

1. OpenAI. [*Introducing ChatGPT*](https://openai.com/index/chatgpt/). Blog post, November 30, 2022.
2. Stanford CRFM. [*Alpaca: A Strong, Replicable Instruction-Following Model*](https://crfm.stanford.edu/2023/03/13/alpaca.html). Blog post, March 13, 2023.
3. OpenAI. [*Learning to Reason with LLMs*](https://openai.com/index/learning-to-reason-with-llms/). Blog post, September 12, 2024.
4. The code for this chapter: `code/training/ch2_bradley_terry.py`, `ch2_kl.py`, `ch2_dpo.py`, `ch2_grpo.py`, with their outputs in `code/training/results/`.
