---
description: "What a language model computes (tokens, next-token probabilities, the chain rule, softmax, cross-entropy, perplexity), the full training pipeline from pretraining to RL with verifiable rewards, and hands-on experiments with Qwen2.5-0.5B base and instruct: the chat template, how few tokens post-training really changes, and a tiny SFT loop."
---
# Chapter 1 · From next word to assistant

> **Goal:** by the end of this chapter you can explain what a language model computes, write down the probability of a sentence and the training loss with every symbol named, and work them out with real numbers. You can draw the full training pipeline (pretraining, mid-training, supervised fine-tuning, preference tuning, RL with verifiable rewards) and say, for each stage, what data it uses, what it optimises, how big it is and what it changes. You will have seen a base model and its instruct version answer the same questions, looked at the chat template token by token, measured how many tokens post-training actually changes, and run a small training loop yourself.

---

## 1.1 Two models, one question

Here are two small models from the same family. Both have 494 million weights, the same architecture and the same vocabulary. The first, **Qwen2.5-0.5B**, is a **base model**: it has only ever been trained to continue text. The second, **Qwen2.5-0.5B-Instruct**, started as an exact copy of the first and was then trained further to act as an assistant. We ask both the same question, "Who are you?", with **greedy decoding** (the model always takes its single most likely next token, so there is no randomness and anyone who runs the script gets the same text).

The base model, given the question as plain text, writes:

```text
 What is your purpose? What is your purpose in life? What is your purpose in life? What is your purpose in life? ...
```

and keeps going until we cut it off at 200 tokens. The instruct model, given the same question in the format it was trained on, writes:

```text
I am Qwen, a large language model created by Alibaba Cloud. I am a language model that can generate human-like
text based on the input I receive. ...
```

and then stops by itself after 87 tokens.

The base model did nothing wrong. It was trained to continue text the way text on the internet continues, and on the internet the line "Who are you?" is often followed by more questions: a list of journaling prompts, a song lyric, a quiz. The instruct model was trained to treat the text as a message from a person and to write one reply, then end its turn.

This chapter is about the distance between those two outputs. What is a language model actually computing? What training turns the first model into the second? How much of the model does that training really change? And how could you do a small piece of that training yourself?

> [!DEFINITION] Base model (pretrained model)
> A language model trained only to predict the next token of ordinary text, at very large scale. It knows a lot about language and the world, but it has no notion of "a user asked me something, I should answer and stop". Also called a **pretrained** or **foundation** model.

> [!DEFINITION] Instruct model (chat model, aligned model)
> A base model that has been trained further, on conversations and on human or automatic feedback, so that it answers questions, follows instructions and ends its turn. The extra training is called **post-training**. Qwen2.5-0.5B-Instruct is the instruct version of Qwen2.5-0.5B.

> [!DEFINITION] Post-training
> Everything done to a model after pretraining to make it useful and safe to talk to: supervised fine-tuning, preference tuning (RLHF, DPO), and reinforcement learning on tasks with checkable answers. Older papers call all of it **alignment** or **alignment tuning**.

All the experiments in this chapter use these two models because they are small enough to run on a laptop, the base and the instruct version are both published, and the team that made them describes their training in a technical report (Qwen Team, 2024) that we can quote. Every number in the text that does not come from a paper comes from a script in the book's repository, and the script's name is given.

## 1.2 What a language model is

Before we can say what training changes, we need to be exact about what the thing being trained is. A language model has three parts: a **tokenizer** that turns text into numbers, a large set of **weights** that does the computing, and an output that is a **probability for every possible next token**.

### Tokens

A model does not read letters or words. It reads **tokens**: pieces of text from a fixed list, each with a number (its **id**). Common words are one token; rare words are split into several pieces; spaces and punctuation are part of tokens too.

> [!DEFINITION] Token
> One unit of text the model reads or writes: a whole word, part of a word, a punctuation mark, or a special marker. Each token has an integer id.

> [!DEFINITION] Tokenizer and vocabulary
> The **tokenizer** is the fixed procedure that splits text into tokens and maps each to its id. The **vocabulary** is the full list of tokens it can produce. Qwen2.5 uses byte-level byte-pair encoding (BPE) with 151,643 regular tokens plus 22 special **control tokens** (Qwen Team, 2024, Section 2).

Here is the sentence we will use for the next few sections, as the Qwen2.5 tokenizer splits it:

{{FIG:ch1_tokens|The sentence "The capital of France is Paris." becomes 7 tokens. Each token has an id, and the ids are all the model ever sees. Output of ch1_nexttoken.py.}}

Notice that most tokens start with a space: `" capital"`, `" of"`, `" Paris"`. The token for `" Paris"` (with a space, id 12095) is a different token from `"Paris"` (no space). This matters later, when we look at what the model wants to write first in an answer.

### Weights

> [!DEFINITION] Weights (parameters)
> The numbers inside the model that training changes. Qwen2.5-0.5B has 494,032,768 of them, stored as a few hundred matrices. When people say a model "learned" something, they mean these numbers were changed so the model's outputs got better on its training data.

The weights are organised as a **transformer**: a stack of 24 identical layers, each of which mixes information between positions (attention) and then transforms each position separately (a small feed-forward network). Inside the model every token is a list of 896 numbers (the **hidden size**). This book does not need the inside of the transformer; for us the model is a function with a very large number of adjustable knobs.

> [!DEFINITION] Transformer
> The network architecture used by nearly all modern language models (Vaswani et al., 2017). It processes all tokens of the input in parallel, and in a language model each position can only look at itself and earlier positions, never at later ones.

### The output: one probability for every possible next token

The model's job is narrow and precise. Given the tokens so far (the **context**), it outputs a score for every one of the 151,936 entries in its output table, and those scores become probabilities that sum to 1. That list of probabilities is the model's prediction of what comes next.

{{FIG:ch1_model|A language model as a function. Token ids go in; the 494 million weights compute; out comes one probability for every possible next token. Shown: the five most likely tokens after "The capital of France is", with the real probabilities from Qwen2.5-0.5B.}}

> [!DEFINITION] Context
> The tokens the model has seen so far, which it uses to predict the next one. For a chat model, the context includes the system message, every earlier turn and the start of the current answer.

> [!NOTE]
> The output table has 151,936 rows, a little more than the 151,665 tokens the tokenizer can actually produce. The extra rows are padding (a round size is faster on GPUs). They are never the right answer, so the model learns to give them almost zero probability.

To generate text, you take that distribution, pick one token (the most likely, for greedy decoding, or a random draw, for sampling), append it to the context, and run the model again. One new token per run. A 200-token answer is 200 runs of the same model, each one seeing one more token than the last. Everything a chat model says, it says this way.

> [!DEFINITION] Decoding (generation)
> Turning next-token distributions into text, one token at a time. **Greedy decoding** always picks the most probable token. **Sampling** draws a token at random according to the probabilities, often after sharpening or flattening them with a **temperature**.

## 1.3 Next-token prediction, one position at a time

Let us watch the base model read our sentence. At every position it sees the tokens so far and produces a distribution. We can look up the probability it gave to the token that really came next.

{{FIG:ch1_frames|Six frames, one per position. Each shows the context, the probability the model gave to the actual next token (upper bar) and the model's own favourite (lower bar). Real numbers from Qwen2.5-0.5B, ch1_nexttoken.py.}}

Read the frames one at a time; they are a small lesson in what the model has learned.

1. After `"The"`, the model gives `" capital"` a probability of only 0.00009. It ranks it 521st. Its favourite is `" following"` (0.684), because "The following ..." starts a great many documents, lists and exam questions. With one word of context, `" capital"` is a genuinely unlikely continuation.
2. After `"The capital"`, `" of"` is the favourite (0.541). Grammar.
3. After `"The capital of"`, the model spreads its bets across countries and cities. `" France"` gets 0.042 and is ranked 4th; the favourite is `" Ber"` (the first piece of "Berlin", 0.186). This is world knowledge in the form of a distribution: many capitals are plausible here.
4. After `"The capital of France"`, `" is"` is the favourite at 0.721.
5. After `"The capital of France is"`, `" Paris"` is the favourite, but at only 0.316. This surprises most people. The softmax example below shows where the other 68% went.
6. After `"... is Paris"`, the full stop is the favourite at 0.465.

No probability is given for the first token, `"The"`. The model needs at least one token of context, and Qwen2.5 does not put a special "beginning of text" token in front of ordinary text, so the first token is simply given.

### From scores to probabilities: softmax

The last layer of the model does not output probabilities directly. It outputs one raw score per vocabulary entry, called a **logit**. A function called **softmax** turns the scores into probabilities.

> [!DEFINITION] Logits
> The raw scores a model outputs before softmax, one per vocabulary entry. They can be any real number. Only differences between logits matter: a logit 1 higher means about 2.72 times (the number $$e$$) more probable.

$$
P(v \mid x_{<t}) = \frac{\exp(z_v)}{\sum_{u=1}^{V} \exp(z_u)}
$$

where:

- $$x_{<t}$$ is the context: all tokens before position $$t$$ (read "x before t");
- $$v$$ is one candidate token from the vocabulary;
- $$z_v$$ is the logit the model computed for $$v$$ at this position;
- $$\exp(z) = e^z$$, with $$e \approx 2.718$$, which turns any number into a positive one and turns differences into ratios;
- $$V$$ is the size of the output table, 151,936 here, and the sum runs over every entry $$u$$;
- $$P(v \mid x_{<t})$$ is the probability of $$v$$ being the next token, given the context. The vertical bar is read "given".

Dividing by the sum makes all the probabilities add up to exactly 1. Because $$\exp$$ grows so fast, the token with the highest logit gets the largest share, but every token keeps a small positive probability.

**Worked example.** After `"The capital of France is"`, the five highest logits are:

{{FIG:ch1_softmax|Softmax by hand on the top five candidates. Left to right: the logits, the exponentials after subtracting the largest logit, the true softmax probabilities over all 151,936 entries, and what softmax would give if these five were the only options. ch1_nexttoken.py.}}

Take `" Paris"` (logit 17.843) and `" ______"` (logit 16.830, a fill-in-the-blank line). The difference is 1.013, so `" Paris"` should be $$e^{1.013} = 2.75$$ times more probable. The true probabilities are 0.3156 and 0.1146, and 0.3156 / 0.1146 = 2.75. Softmax over all entries does exactly what the formula says.

If the vocabulary contained only these five tokens, the denominator would be $$1 + 0.363 + 0.201 + 0.175 + 0.163 = 1.902$$ (after subtracting the largest logit, which does not change the result), and `" Paris"` would get $$1 / 1.902 = 0.526$$. In the real model it gets 0.316, because the other 151,931 entries together hold 40% of the probability. Each of them is tiny, but there are a lot of them.

Now look at what the model's top five are: `" Paris"`, then three lengths of blank line (`" ______"`, `" ____"`, `" __"`), then a colon and a line break. A base model is a mirror of its training data, and on the web the words "The capital of France is" appear very often in **quizzes and worksheets**, followed by a blank to fill in. Together the blanks get 23% of the probability. The model knows the answer is Paris; it is also, correctly, unsure whether this document wants the answer or the question. Keep this in mind: it is a big part of what post-training will change.

Here is the code that produced these numbers, simplified from `ch1_nexttoken.py`:

```python
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B').to('mps').eval()

ids = tok('The capital of France is Paris.')['input_ids']        # [785, 6722, 315, 9625, 374, 12095, 13]
with torch.no_grad():
    logits = model(torch.tensor([ids], device='mps')).logits[0]  # shape [7, 151936]: one row of scores per position
probs = torch.softmax(logits, dim=-1)                            # each row now sums to 1

for t in range(len(ids) - 1):
    p = probs[t, ids[t + 1]]          # row t predicts the token at position t+1
    print(tok.decode(ids[:t + 1]), '->', tok.decode([ids[t + 1]]), float(p))
```

Line by line: the tokenizer turns the sentence into 7 ids. One forward pass of the model (`model(...)`) computes logits for **all 7 positions at once**: row 0 is the prediction after `"The"`, row 1 after `"The capital"`, and so on. This is the trick that makes training efficient: one pass over a sequence gives a prediction, and therefore a training signal, at every position. `torch.softmax(..., dim=-1)` applies the softmax formula to each row. Finally, row $$t$$ is the prediction for position $$t+1$$, so we look up the probability of the token that actually sits there. The `torch.no_grad()` block tells PyTorch we are only reading, not training, so it does not keep the extra bookkeeping needed for gradients.

## 1.4 The probability of a whole sentence: the chain rule

A language model gives probabilities for one next token. But we often want the probability of a whole sentence: to compare two answers, to score how well the model fits some text, or to train it. Probability theory gives a rule for exactly this.

> [!DEFINITION] Chain rule of probability
> The probability of several things happening in order equals the probability of the first, times the probability of the second given the first, times the probability of the third given the first two, and so on. It is always true; it is not an approximation.

For a sequence of tokens $$x_1, x_2, \dots, x_T$$:

$$
P(x_1, \dots, x_T) = P(x_1) \prod_{t=2}^{T} P(x_t \mid x_1, \dots, x_{t-1})
$$

where:

- $$x_t$$ is the token at position $$t$$, and $$T$$ is the number of tokens;
- $$P(x_1)$$ is the probability of the first token on its own;
- $$\prod$$ (capital Greek "pi") means "multiply together", here for every position from 2 to $$T$$;
- $$P(x_t \mid x_1, \dots, x_{t-1})$$ is the model's next-token probability at position $$t$$, exactly the number in the frames above.

A language model is called **autoregressive** because it uses this rule: it only ever predicts one token given the earlier ones, and the probability of anything longer is the product of those predictions.

**Worked example.** Our sentence has 7 tokens. Since the model does not score the first one, we compute the probability of the other six given `"The"`:

{{FIG:ch1_chain|The chain rule on real numbers: six next-token probabilities multiply to 2.1 × 10⁻⁷. The logs add up to −15.355, which divided by 6 gives the average loss of 2.559. ch1_nexttoken.py.}}

$$
0.00009 \times 0.541 \times 0.0416 \times 0.721 \times 0.316 \times 0.465 = 2.1 \times 10^{-7}
$$

About two in ten million. That sounds absurdly small for a true and ordinary sentence, but it is the right scale: there are billions of possible six-token continuations of `"The"`, and this is one of them. Probabilities of whole sentences are always tiny, which is why nobody works with them directly. They work with **logarithms**.

### Why logs

The log of a product is the sum of the logs: $$\log(a \times b) = \log a + \log b$$. So:

$$
\log P(x_2, \dots, x_T \mid x_1) = \sum_{t=2}^{T} \log P(x_t \mid x_{<t})
$$

where $$\log$$ is the natural logarithm (base $$e$$), and every term is negative or zero because every probability is at most 1.

For our sentence the six logs are −9.316, −0.614, −3.179, −0.327, −1.153 and −0.765, and their sum is **−15.355**. Taking $$e^{-15.355}$$ gives back $$2.1 \times 10^{-7}$$. Sums are easier to work with than products, and computers cannot store numbers like $$10^{-5000}$$ (the probability of a long document) without special tricks, while a sum like −11,500 is no problem.

## 1.5 Cross-entropy loss and perplexity

Training needs one number that says how badly the model did, so that it can change the weights to make that number smaller. For language models the number is the average negative log-probability of the actual tokens.

$$
\mathcal{L}(\theta) = -\frac{1}{T-1} \sum_{t=2}^{T} \log P_\theta(x_t \mid x_{<t})
$$

where:

- $$\theta$$ (Greek "theta") stands for all the weights of the model, and $$P_\theta$$ is the model's probability with those weights;
- $$\mathcal{L}$$ (a curly L) is the **loss**: the number training tries to make small;
- the minus sign turns the negative logs into a positive number;
- dividing by $$T-1$$ (the number of predictions) makes it an average per token, so long and short texts are comparable.

> [!DEFINITION] Cross-entropy loss (negative log-likelihood)
> Minus the log of the probability the model gave to the right answer, averaged over all positions. If the model gives the right token probability 1, the loss there is 0. If it gives 0.5, the loss is 0.69. If it gives 0.01, the loss is 4.6. Confident mistakes are punished very hard.

> [!DEFINITION] Nats
> The unit of a loss computed with natural logarithms. A loss of 1 nat per token means the model gave the right tokens, on average (geometrically), a probability of $$e^{-1} = 0.37$$. With base-2 logs the same quantity is in **bits**.

The curve below is $$-\log P$$ for every possible $$P$$, with our six real tokens marked on it.

{{FIG:ch1_nll|The per-token loss −log P. The six tokens of our sentence sit on the curve. The loss is flat and near zero when the model is confident and right, and rises steeply as P falls. One token, " capital", supplies most of the total. ch1_nexttoken.py.}}

**Worked example.** The six losses are 9.316, 0.614, 3.179, 0.327, 1.153 and 0.765. Their sum is 15.355 and their average is 15.355 / 6 = **2.559 nats per token**. The library computes the same number in one call; the script checks this:

```python
out = model(torch.tensor([ids]), labels=torch.tensor([ids]))
print(out.loss)      # 2.559, identical to our hand computation
```

When you pass `labels`, the library shifts them by one position for you (the label for row $$t$$ is the token at $$t+1$$), computes the softmax and the negative log at every position, and averages. That one line is the pretraining objective. Everything in Section 1.10 is built on it.

Notice how unequal the six terms are. `" capital"` alone is 9.3 of the 15.4. A single surprising token dominates the loss of a short sentence. Over billions of tokens these surprises average out, and the loss becomes a smooth measure of how well the model predicts text.

### Perplexity

The loss in nats is hard to picture. **Perplexity** turns it back into something countable.

$$
\text{PPL} = \exp(\mathcal{L})
$$

where $$\mathcal{L}$$ is the average cross-entropy loss in nats, and $$\exp$$ undoes the log.

> [!DEFINITION] Perplexity
> The exponential of the average loss. It can be read as "the model was, on average, as unsure as if it were choosing uniformly among this many tokens". A perplexity of 1 means perfect prediction. A perplexity of 151,936 would mean the model knows nothing and guesses uniformly over the whole vocabulary.

For our sentence, $$e^{2.559} = 12.9$$: on average, the model was as unsure as if it were picking among about 13 equally likely tokens at each step. The script also scores two more sentences:

| Sentence | Loss (nats/token) | Perplexity |
|---|---|---|
| The capital of France is Paris. | 2.559 | 12.9 |
| The cat sat on the mat. | 2.957 | 19.2 |
| Purple ideas sleep furiously under quiet spoons. | 7.668 | 2,139 |

The first two are ordinary English and get low perplexity. The third is grammatical nonsense, and the model is very surprised by almost every word. Perplexity on held-out text is the main number people watch during pretraining, and Chapter 3 discusses its limits (in short: it tells you how well a model predicts text, not how good its answers are).

> [!TIP]
> When you see "loss 2.0" in a training log, compute $$e^{2.0} = 7.4$$. That is the perplexity, and it is easier to feel: "about seven equally likely choices per token". A loss that drops from 2.0 to 1.9 is a 10% drop in perplexity (7.4 to 6.7).

## 1.6 The training pipeline: from base model to assistant

Everything above describes a single model and a single loss. Real models are trained in **stages**. Each stage starts from the weights the previous stage produced, uses different data, and often a different objective. Here is the whole pipeline as it looks in the technical reports of 2024 and 2025 (Qwen2.5, Llama 3, OLMo 2, Tulu 3, DeepSeek-R1). The numbers in the "scale" boxes are quoted from those reports.

{{FIG:ch1_pipeline|The training pipeline. Pretraining and mid-training produce a base model; supervised fine-tuning, preference tuning and RL with verifiable rewards turn it into an assistant. For each stage: its data, its objective, its published scale, and what it changes. Sources: Qwen Team (2024), Grattafiori et al. (2024), Team OLMo (2024), Ouyang et al. (2022), DeepSeek-AI (2025).}}

The Llama 3 report describes the two halves in two short paragraphs that are worth reading in the original:

> [!PAPER] Grattafiori et al. (2024), The Llama 3 Herd of Models · Section 1 · page 3
> [![Two bullet points from the Llama 3 introduction. Language model pre-training: the model is pre-trained with 405B parameters on 15.6T tokens with an 8K context window, then a continued pre-training stage extends context to 128K. Language model post-training: the pre-trained model does not yet follow instructions or behave as an assistant; it is aligned in several rounds of supervised finetuning and Direct Preference Optimization.](/img/training/ch1-llama3-stages.png)](/img/training/ch1-llama3-stages.png)
>
> **Context:** the overview at the start of the report, before the details of data, architecture and training.
>
> **What it says:** pretraining is "next-token prediction" at "massive scale" (405 billion parameters, 15.6 trillion tokens). After it, the model "has a rich understanding of language but it does not yet follow instructions or behave in the way we would expect an assistant to". Post-training fixes that with rounds of SFT and DPO.
>
> **Why it matters:** this is the split this whole book is organised around. Pretraining builds the knowledge; post-training shapes the behaviour.

The rest of this section walks through the five stages. For each one: the data, the objective (as an equation preview; later parts of the book derive each one properly), the scale, and what it changes. Part 1 of this book (this chapter and the next two) is the big picture. Later parts take supervised fine-tuning, RLHF, DPO and RL for reasoning one at a time, in depth, with code.

### Stage 1: pretraining

**Data.** As much good text as can be found: web pages (filtered and de-duplicated), books, scientific papers, code, maths, in many languages. The Llama 3 report gives its final mix as roughly 50% general knowledge, 25% maths and reasoning, 17% code and 8% multilingual tokens (Grattafiori et al., 2024, Section 3.1.2).

**Objective.** Exactly the loss of Section 1.5, averaged over every position of every document in the training set:

$$
\mathcal{L}_{\text{pretrain}}(\theta) = -\,\mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t}) \right]
$$

where:

- $$\mathcal{D}$$ is the training corpus, and $$x \sim \mathcal{D}$$ means "a document $$x$$ drawn from it";
- $$\mathbb{E}$$ means "the average over" (the **expectation**), here over documents;
- the inside of the brackets is the per-token average log-probability of one document, as before.

Nothing is labelled by a person. The text is its own label: every token is the answer to the question "what comes next?" for the tokens before it. This is called **self-supervised learning**, and it is why pretraining can use trillions of tokens.

> [!DEFINITION] Self-supervised learning
> Training where the labels come from the data itself rather than from people. In next-token prediction the label for each position is simply the token that comes next in the text.

**Scale.** This stage is where nearly all the compute goes.

| Model | Parameters | Pretraining tokens | Source |
|---|---|---|---|
| GPT-3 (2020) | 175B | 300B | Brown et al. (2020), Table 2.1 |
| Llama 3 (2024) | 405B | 15.6T | Grattafiori et al. (2024), Section 1 |
| Qwen2.5 (2024) | 0.5B to 72B | 18T | Qwen Team (2024), Section 3 |
| OLMo 2 7B (2024) | 7B | 3.9T (plus mid-training) | Team OLMo (2024), Section 2 |

The Llama 3 flagship used $$3.8 \times 10^{25}$$ floating-point operations for pretraining, "almost 50× more than the largest version of Llama 2" (Grattafiori et al., 2024, Section 1).

**What it changes.** Everything, from random numbers to a model that knows grammar, facts, styles, code, some arithmetic and a great deal of the structure of the world as described in text. The output is a base model. It can already do many tasks if you phrase them as text to be continued, as GPT-3 showed:

> [!PAPER] Brown et al. (2020), GPT-3: Language Models are Few-Shot Learners · Figure 2.1 · page 7
> [![Figure 2.1 of the GPT-3 paper. Left: zero-shot, one-shot and few-shot prompts for translating English to French, such as "sea otter => loutre de mer", followed by "cheese =>", with no gradient updates. Right: traditional fine-tuning, where each example causes a gradient update.](/img/training/ch1-gpt3-fig21.png)](/img/training/ch1-gpt3-fig21.png)
>
> **Context:** Section 2 of the GPT-3 paper, which defines the settings in which the 175-billion-parameter base model is evaluated.
>
> **What it says:** instead of fine-tuning, you can put a task description and a few worked examples into the context ("sea otter => loutre de mer") and let the base model continue the pattern ("cheese =>"). "No gradient updates are performed." The left side is **in-context learning**; the right side is the traditional alternative of training on each example.
>
> **Why it matters:** a base model can already do tasks, but only if you disguise the task as a document it wants to continue. Post-training removes the need for the disguise.

> [!DEFINITION] In-context learning (few-shot prompting)
> Getting a model to do a task by showing examples of the task inside the prompt, with no change to its weights. "Zero-shot" means only an instruction, "few-shot" means an instruction plus a few examples.

### Stage 2: mid-training

A newer name for an old idea. Near the end of pretraining, the data mix is changed: the share of high-quality text, maths, code and long documents goes up, and the learning rate is lowered towards zero (**annealing**). The objective stays the same next-token loss.

> [!PAPER] Team OLMo (2024), 2 OLMo 2 Furious · Section 1 · page 4
> [![A bullet point from the OLMo 2 introduction titled Mid-training Recipe: earlier models demonstrated the usefulness of data curricula for pretraining; the paper splits pretraining into two stages, with the latter mid-training stage used to infuse new knowledge and patch deficiencies in capabilities.](/img/training/ch1-olmo2-midtraining.png)](/img/training/ch1-olmo2-midtraining.png)
>
> **Context:** the list of contributions at the start of the OLMo 2 report, whose models, data and code are fully open.
>
> **What it says:** pretraining is split "into two stages, with the latter mid-training stage being used to infuse new knowledge and patch deficiencies in capabilities".
>
> **Why it matters:** the base model you download is often not "pure web text". Its last tens or hundreds of billions of tokens were chosen with care, and some of them look a lot like post-training data.

> [!DEFINITION] Mid-training (annealing)
> A final phase of pretraining on a smaller, higher-quality and more targeted data mix (maths, code, long documents, synthetic question-answer data), usually while the learning rate decays to zero. Same loss as pretraining, different data.

**Scale.** OLMo 2 7B is pretrained on 3.90 trillion tokens, then mid-trained three separate times on 50 billion tokens each, and the three results are averaged: 4.05 trillion tokens in total (OLMo et al., 2024, Section 2.3). So mid-training is about 3.7% of the tokens. Llama 3 anneals on its final 40 million tokens, and reports that annealing on high-quality maths and code improved the 8B model on the GSM8k and MATH benchmarks by 24.0% and 6.4% (Grattafiori et al., 2024, Section 3.1.3).

**What it changes.** Sharper maths and code, longer context, and, often, some familiarity with question-and-answer formats. This matters for our experiments: the Qwen2.5 report says its pretraining data includes synthetic data "particularly in mathematics, code, and knowledge domains" generated with Qwen2-72B-Instruct and Qwen2-Math-72B-Instruct (Qwen Team, 2024, Section 3.1). So the Qwen2.5 base model has already read a lot of text written by an assistant. You will see the effect in Section 1.7.

### Stage 3: supervised fine-tuning (SFT)

**Data.** Pairs of a **prompt** (an instruction or question, possibly with earlier turns of a conversation) and a **response** written the way the final assistant should write. The responses are written by people, generated by a stronger model and filtered, or both. Every example is put into the **chat template** (Section 1.8) so the model learns where a turn begins and ends.

**Objective.** The same next-token loss, with one change: only the tokens of the response count. The prompt is in the context but is not trained on.

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\,\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{SFT}}} \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} \log P_\theta(y_t \mid x, y_{<t}) \right]
$$

where:

- $$x$$ is the prompt (with the chat template around it) and $$y$$ is the response, $$y_1 \dots y_{|y|}$$;
- $$|y|$$ is the number of tokens in the response;
- $$P_\theta(y_t \mid x, y_{<t})$$ is the probability of the $$t$$-th response token given the whole prompt and the response so far;
- $$\mathcal{D}_{\text{SFT}}$$ is the set of prompt-response pairs.

Compare it with the pretraining loss: it is the same formula with the sum restricted to the response. In code, this restriction is a single line that sets the label of every prompt token to −100 (Section 1.10).

> [!DEFINITION] Supervised fine-tuning (SFT), instruction tuning
> Continuing to train a pretrained model with the next-token loss on examples of the desired behaviour (prompt plus ideal response), with the loss computed only on the response. **Instruction tuning** is the same thing with an emphasis on many different tasks phrased as instructions (Wei et al., 2021; Chung et al., 2022).

**Scale.** Tiny compared with pretraining. InstructGPT's SFT set had about 13,000 training prompts (Ouyang et al., 2022, Section 3.2). Qwen2.5's has "over 1 million" examples, trained for two epochs with sequences up to 32,768 tokens (Qwen Team, 2024, Section 4.1). LIMA trained a 65B model on just 1,000 carefully chosen examples (Zhou et al., 2023). Even a million examples of a thousand tokens each is about a billion tokens: less than 0.01% of an 18-trillion-token pretraining run.

**What it changes.** Format and behaviour: answering instead of continuing, the turn structure, the tone, when to stop, when to refuse, how long to be. Whether it also teaches new knowledge is exactly the question of Section 1.9.

### Stage 4: preference tuning (RLHF, DPO)

SFT can only show the model good answers. It cannot tell the model which of two decent answers is better, or punish a bad habit directly. Preference tuning does this.

**Data.** For each prompt, two or more responses (usually sampled from the model itself) and a judgement of which one is better, made by a person or by another model. A **preference pair** is (prompt, chosen response, rejected response).

**Objective, version 1: RLHF.** First train a **reward model** $$r_\phi(x, y)$$ that gives a score to a response, using the preference pairs. Then improve the language model (now called the **policy** $$\pi_\theta$$) with reinforcement learning to get high reward, while staying close to the SFT model:

$$
\max_{\theta} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \Big[ r_\phi(x, y) \Big] \;-\; \beta \, \mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\text{SFT}}\big)
$$

where:

- $$\pi_\theta(y \mid x)$$ is the language model being trained, seen as a **policy**: a rule for choosing outputs; $$y \sim \pi_\theta(\cdot \mid x)$$ means a response sampled from it;
- $$r_\phi(x, y)$$ is the reward model's score, with its own weights $$\phi$$ (Greek "phi");
- $$\pi_{\text{SFT}}$$ is the frozen SFT model, the starting point;
- $$\mathrm{KL}(\cdot \,\|\, \cdot)$$ is the **KL divergence**, a measure of how different two distributions are (we define it properly and compute it in Section 1.9);
- $$\beta$$ (Greek "beta") is a number that sets how strongly the model is held near the SFT model.

The KL term is the leash: without it the model finds strange outputs that fool the reward model. InstructGPT is the paper that made this recipe famous:

> [!PAPER] Ouyang et al. (2022), Training language models to follow instructions with human feedback (InstructGPT) · Figure 2 · page 3
> [![Figure 2 of the InstructGPT paper, three columns. Step 1: collect demonstration data and train a supervised policy, where a labeler writes the desired output and GPT-3 is fine-tuned on it. Step 2: collect comparison data and train a reward model, where a labeler ranks several model outputs from best to worst. Step 3: optimize a policy against the reward model using PPO, where the reward for each new output updates the policy.](/img/training/ch1-instructgpt-fig2.png)](/img/training/ch1-instructgpt-fig2.png)
>
> **Context:** the method overview at the start of the paper.
>
> **What it says:** three steps: "(1) supervised fine-tuning (SFT), (2) reward model (RM) training, and (3) reinforcement learning via proximal policy optimization (PPO) on this reward model." Labelers write demonstrations for step 1 and rank model outputs for step 2; step 3 needs no new human labels.
>
> **Why it matters:** this is the template every later pipeline varies. Chapter 2 tells the story of how it came about; a later part of the book builds each step.

> [!DEFINITION] RLHF (reinforcement learning from human feedback)
> Training a language model with reinforcement learning to maximise the score of a reward model that was itself trained on human preference judgements, usually with a penalty for drifting too far from the starting model.

**Objective, version 2: DPO.** Rafailov et al. (2023) showed that the same goal can be reached without a separate reward model and without reinforcement learning, with a loss computed directly on preference pairs:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

where:

- $$y_w$$ is the chosen ("winning") response and $$y_l$$ the rejected ("losing") one;
- $$\pi_{\text{ref}}$$ is a frozen reference model, usually the SFT model;
- each fraction compares how likely the trained model makes a response with how likely the reference model makes it;
- $$\sigma$$ is the **sigmoid** function, $$\sigma(z) = 1 / (1 + e^{-z})$$, which squashes any number into the range 0 to 1;
- $$\beta$$ again controls how far the model may move from the reference.

In words: raise the probability of the chosen answer and lower the probability of the rejected one, both measured relative to where the model started. No sampling, no reward model; just a loss on log-probabilities, like SFT.

> [!DEFINITION] DPO (direct preference optimization)
> A preference-tuning method that trains directly on (prompt, chosen, rejected) triples with a classification-style loss, instead of first training a reward model and then running reinforcement learning.

**Scale.** InstructGPT's reward model used 33,000 training prompts and its PPO stage 31,000 prompts (Ouyang et al., 2022, Section 3.2). Qwen2.5's offline stage used about 150,000 DPO training pairs for one epoch (Qwen Team, 2024, Section 4.2), followed by online RL with GRPO against a reward model (Section 4.3). Llama 3 ran SFT and DPO in six rounds, each round collecting new preference data with the best model so far:

> [!PAPER] Grattafiori et al. (2024), The Llama 3 Herd of Models · Figure 7 · page 15
> [![Figure 7 of the Llama 3 paper: collected prompts lead to K generations per prompt; a reward model scores them in rejection sampling; the best become SFT data; an SFT model is trained, then a DPO model; the best models from each round feed the next round. Pairwise preference data trains both the reward model and DPO.](/img/training/ch1-llama3-fig7.png)](/img/training/ch1-llama3-fig7.png)
>
> **Context:** Section 4, post-training. The figure shows one round; "we apply the above methods in six rounds" (Section 4.1.6).
>
> **What it says:** the strategy "involves rejection sampling, supervised finetuning, and direct preference optimization". The model generates several answers per prompt, the reward model picks the best ones as new SFT data, and preference pairs train DPO.
>
> **Why it matters:** in practice SFT and preference tuning are not two clean steps but a loop, and the model's own outputs become its training data.

> [!DEFINITION] Rejection sampling (best-of-K)
> Generate K answers for a prompt, score them with a reward model or a checker, and keep the best one as a training example.

**What it changes.** Which of the answers the model could already give it actually prefers: more helpful, better organised, more careful, less harmful. InstructGPT's labelers preferred outputs of the 175B InstructGPT over the 175B GPT-3 85 ± 3% of the time, and preferred the 1.3B InstructGPT over the 175B GPT-3, "despite having over 100x fewer parameters" (Ouyang et al., 2022, Section 1).

### Stage 5: RL with verifiable rewards (RLVR)

**Data.** Problems whose answer a program can check: maths with a known final number, code with unit tests, puzzles with a unique solution, instructions with checkable constraints ("answer in exactly three sentences").

**Objective.** The same reinforcement-learning idea as RLHF, but the reward is not a learned model of human taste. It is a rule:

$$
r(x, y) = \begin{cases} 1 & \text{if the final answer in } y \text{ is correct for problem } x \\ 0 & \text{otherwise} \end{cases}
$$

where $$x$$ is the problem, $$y$$ is the model's full response (reasoning and answer), and "correct" is decided by a program: compare the boxed number with the reference, run the tests.

> [!DEFINITION] RLVR (reinforcement learning with verifiable rewards)
> Reinforcement learning in which the reward comes from an automatic check of the answer (exact match, unit tests, a constraint checker) rather than from a learned reward model. The name comes from the Tulu 3 report (Lambert et al., 2024).

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1 · Section 2.2 · page 3
> [![Section 2.2 Reward Design of the DeepSeek-R1 paper: for DeepSeek-R1-Zero, rule-based rewards deliver precise feedback for mathematical, coding and logical reasoning data; the system has two types of reward, accuracy rewards and format rewards. Accuracy rewards check whether the response is correct; for maths, the final answer must be in a specified format such as a box, enabling reliable rule-based verification of correctness.](/img/training/ch1-r1-reward.png)](/img/training/ch1-r1-reward.png)
>
> **Context:** the reward design for DeepSeek-R1-Zero, a model trained with reinforcement learning directly from the base model, with no SFT first.
>
> **What it says:** "we employ rule-based rewards", of two kinds: "accuracy rewards and format rewards". For maths, a final answer in a box allows "reliable rule-based verification of correctness".
>
> **Why it matters:** with a reward that cannot be fooled the way a learned reward model can, RL can run much longer. The current version of the paper (arXiv v2, January 2026; the first version reported 71.0%, see Chapter 2) reports that DeepSeek-R1-Zero's average pass@1 on the AIME 2024 maths competition rose "from an initial 15.6% to 77.9%" during RL training (Section 2.3 of the current arXiv version).

**What it changes.** Longer and more careful step-by-step reasoning, self-checking, and accuracy on exactly the kinds of problems that can be checked. Qwen2.5 uses the same kind of automatic checking ("execution feedback and answer matching") to build the preference pairs for maths and code in its offline RL stage (Qwen Team, 2024, Section 4.2). A later part of the book is devoted to RL for reasoning.

### The whole picture in one table

| Stage | Data | Loss or reward | Typical size | What it changes | Covered in depth |
|---|---|---|---|---|---|
| Pretraining | trillions of tokens of text | next-token, all tokens | 15T to 18T tokens | knowledge, language, skills | Sections 1.2 to 1.5 |
| Mid-training | curated high-quality and synthetic text | next-token, all tokens | tens to hundreds of billions of tokens | maths, code, long context, Q&A familiarity | this section |
| SFT | prompt + ideal response | next-token, response only | 1,000 to a few million examples | format, turn-taking, stopping, tone | a later part on SFT |
| Preference tuning | prompt + chosen + rejected | reward model + RL (RLHF), or DPO loss | tens to hundreds of thousands of pairs | which answer it prefers | later parts on RLHF and DPO |
| RLVR | problems with checkable answers | 1 if correct, else 0 | thousands to hundreds of thousands of problems | reasoning, accuracy on checkable tasks | a later part on RL for reasoning |

### What post-training costs

If you remember one fact about the pipeline, make it this one. Post-training is a small fraction of the cost of pretraining, and yet it is what turns a text continuer into an assistant.

> [!PAPER] Ouyang et al. (2022), InstructGPT · Section 5.1 · page 17
> [![A paragraph from the InstructGPT discussion headed "The cost of increasing model alignment is modest relative to pretraining": training the 175B SFT model requires 4.9 petaflops/s-days and the 175B PPO-ptx model 60 petaflops/s-days, compared to 3,640 petaflops/s-days for GPT-3; RLHF is more effective at making models helpful than a 100x model size increase.](/img/training/ch1-instructgpt-cost.png)](/img/training/ch1-instructgpt-cost.png)
>
> **Context:** the discussion section, "Implications for alignment research".
>
> **What it says:** "training our 175B SFT model requires 4.9 petaflops/s-days and training our 175B PPO-ptx model requires 60 petaflops/s-days, compared to 3,640 petaflops/s-days for GPT-3".
>
> **Why it matters:** SFT cost about 0.13% of pretraining, and the whole RLHF stage about 1.6%. Post-training is cheap in compute; what it is expensive in is careful data.

> [!DEFINITION] Petaflop/s-day
> A unit of compute: a machine doing $$10^{15}$$ floating-point operations per second, running for one day, which is about $$8.64 \times 10^{19}$$ operations.

{{FIG:ch1_compute|Compute for the 175B models in petaflop/s-days, on a log scale, from Ouyang et al. (2022), Section 5.1. Each gridline is ten times the previous one.}}

That imbalance raises the question at the heart of this chapter. If post-training uses so little compute and so little data, what can it possibly be changing? We will answer it by measurement. But first, let us see the change with our own eyes.

## 1.7 Hands-on: the base model and the instruct model, side by side

The script `ch1_base_vs_instruct.py` gives five prompts to the two models in three settings:

1. **base, raw text:** the base model gets just the question, as plain text, the way you would test a pretrained model;
2. **base, chat template:** the base model gets the question wrapped in the chat format that the instruct model was trained on (Section 1.8 shows it token by token);
3. **instruct, chat template:** the instruct model gets the chat format, as its makers intended.

Decoding is greedy, at most 200 new tokens, and both models stop if they produce either of the two end tokens, `<|endoftext|>` or `<|im_end|>`. The core of the script, simplified:

```python
def greedy(tok, model, ids, max_new_tokens=200):
    stop_ids = [tok.convert_tokens_to_ids(t) for t in ['<|endoftext|>', '<|im_end|>']]  # same stop rule for both
    out = model.generate(torch.tensor([ids]), max_new_tokens=max_new_tokens, do_sample=False,
                         eos_token_id=stop_ids)
    return out[0, len(ids):].tolist()                     # only the new tokens

raw  = tok_b(q)['input_ids']                               # setting 1: the question as plain text
chat = tok_i.apply_chat_template([{'role': 'user', 'content': q}],
                                 add_generation_prompt=True, tokenize=True)   # settings 2 and 3
```

`do_sample=False` is greedy decoding: no randomness, so the outputs below are exactly reproducible. `apply_chat_template` builds the chat format from a list of messages; `add_generation_prompt=True` adds the line that opens the assistant's turn, so the model's next token is the first token of its answer. The two models share one tokenizer (the script checks this), so the same text becomes the same ids for both.

{{FIG:ch1_side|Two of the five prompts, in all three settings. Red badges mark outputs that never ended and were cut at 200 tokens. Greedy decoding, ch1_base_vs_instruct.py.}}

Here is everything the script found, prompt by prompt.

**"What is the capital of France?"** The base model, given the raw question, answers `" The capital of France is Paris."` and stops. The instruct model gives the same sentence and stops. Inside the chat template, the base model also starts with "The capital of France is Paris." but then cannot end its turn: it produces odd tokens (`" navigationOptions"`, `"icode"`) and starts repeating "You are a helpful assistant." and the question, over and over, until it is cut off.

**"Write a haiku about rain."** Raw, the base model writes three short lines and stops: "Raindrops fall, / Soft and gentle, / Nature's gift to us all." The instruct model writes "Raindrops dance on the ground, / A symphony of nature's music, / Peace and tranquility." and stops. (Neither has the 5-7-5 syllable pattern of a real haiku; a model of half a billion weights does not count syllables well.) In the chat template the base model writes two lines and then falls into a loop of `"reeze"` and the prompt.

**"A shop sells pencils at 3 for 45 cents. How much do 7 pencils cost?"** All three settings work it out step by step: 45 / 3 = 15 cents per pencil, 7 × 15 = 105 cents = 1.05 dollars. Both base settings are still talking at 200 tokens. The instruct model writes its working in LaTeX and finishes after 197 tokens with the answer in a box, `\boxed{1.05}`, then ends its turn. The box is a habit from post-training on maths: as Section 1.6 showed, a boxed final answer is what an automatic checker looks for.

**"Give me three tips for sleeping better."** All three give three sensible tips. The instruct model opens with "Certainly! Here are three tips for improving your sleep quality:", gives each tip a bold heading, adds a closing summary, and is still going at 200 tokens; the base model in the chat template opens with "Sure, here are three tips for sleeping better:" and stops by itself; the raw base model just starts with "1." and also stops.

**"Who are you?"** The raw base model repeats "What is your purpose in life?" until cut off. In the chat template, it says "I am a system that can help you with various tasks." and then invents a whole conversation, writing the user's lines as well as its own. The instruct model says "I am Qwen, a large language model created by Alibaba Cloud..." and stops after 87 tokens.

### What this shows, honestly

This is not the dramatic "base model is useless, instruct model is brilliant" picture you may have expected, and it is worth understanding why.

1. **The base model already knows the answers.** Paris, 105 cents, sensible sleep tips: the knowledge is all there before any post-training. This is the first piece of evidence for the claim we will test properly in Section 1.9.
2. **Qwen2.5-0.5B is not a "pure" base model.** On three of the five prompts, the raw base model answered the question instead of continuing the text in some other way. In the chat template, it even opened with "Sure, here are three tips". As Section 1.6 noted, the Qwen2.5 pretraining data includes synthetic data written by Qwen2 instruct models. Older base models, such as GPT-3 in 2020, were much less willing to answer a bare question. Today's line between "base" and "instruct" is blurrier than the names suggest.
3. **The clearest difference is turn-taking: knowing when to stop.** The instruct model ended its turn by itself on four of five prompts (the fifth was the long, formatted sleep-tips answer, which hit the limit). The base model, in the chat format, ended only the sleep-tips answer, and only with `<|endoftext|>`, never with `<|im_end|>`. Everywhere else it rambled, looped, or wrote the user's next message for them.
4. **Identity comes from post-training.** "I am Qwen, created by Alibaba Cloud" is not something the base model says. It comes from the default system prompt ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") together with the post-training that taught the model to take that message seriously.
5. **Post-training did not fix everything.** Neither model writes a correct haiku. It is a small model.

Point 3 deserves a closer look, because the reason is visible in the tokens themselves.

## 1.8 The chat template, token by token

A chat model does not see "messages". It sees one long sequence of tokens, exactly like a base model does. The conversation structure is written into that sequence with a few **special tokens**. Here is our first prompt exactly as the instruct model reads it:

{{FIG:ch1_template|The chat template for "What is the capital of France?", all 36 tokens. Highlighted blocks are special tokens; each marks the start or end of a turn. ch1_base_vs_instruct.py.}}

As text, the same thing looks like this:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```

> [!DEFINITION] Chat template
> The exact text format that turns a list of messages (system, user, assistant) into one token sequence for the model. Each model family has its own. Qwen uses the **ChatML** style: every message is `<|im_start|>` + role name + line break + content + `<|im_end|>`.

> [!DEFINITION] Special token (control token)
> A token that is not produced by ordinary text but added on purpose, to mark structure: the start and end of a turn, the end of a document, a tool call. `<|im_start|>` (id 151644) and `<|im_end|>` (id 151645) are single tokens, even though they look like 12 characters.

> [!DEFINITION] System prompt
> The first message in a conversation, written by the developer rather than the user, which sets the assistant's identity, rules or style. If you give Qwen2.5-Instruct no system message, its template inserts the default one shown above.

Reading the 36 tokens in order:

- `<|im_start|>` `system` `↵` opens the system turn. The role name is ordinary text (`"system"` is token 8948, a normal word).
- 16 tokens of the default system prompt follow, then `<|im_end|>` `↵` closes the turn.
- `<|im_start|>` `user` `↵`, the 7 tokens of the question, `<|im_end|>` `↵`: the user turn.
- `<|im_start|>` `assistant` `↵` opens the assistant turn, and the sequence ends there. The next token the model writes is the first token of its answer.

When the answer is finished, a well-trained model writes `<|im_end|>`. The software that runs the model watches for that token, stops generating, and shows you the answer. If the model never writes it, the software keeps asking for more tokens, and the model keeps going: it closes the turn with something else, opens a new "user" turn, and writes a conversation with itself. That is exactly what the base model did in Section 1.7.

### Why the template matters

**The base model never learned to end a turn.** The `<|im_end|>` token exists in the base model's vocabulary, but the base model gives it essentially zero probability where the instruct model writes it. Section 1.9 measures this: across 24 answers that the instruct model ended with `<|im_end|>`, the instruct model gave that token an average probability of 0.81, and the base model, reading exactly the same context, gave it at most 0.00003. Where the base model wanted to stop, it put its probability on `<|endoftext|>`, the end-of-document token it saw at the end of every pretraining document. Teaching the model this one token, and the turn structure around it, is a large part of what SFT does.

**Use the template the model was trained with.** If you send an instruct model text without its template, or with another family's template, it is reading a format it rarely saw in training, and quality drops in ways that are hard to spot. Libraries ship the template with the tokenizer (`apply_chat_template`) precisely so you do not have to type it by hand. Getting it subtly wrong (a missing line break, a missing `add_generation_prompt`) is one of the most common bugs when people fine-tune or serve models.

**The template is part of the training data format.** In SFT (Section 1.10) every training example is written in this template, and the loss is computed only on the tokens of the assistant's turn, including its closing `<|im_end|>`. That is how the model learns both what to say and when to stop.

> [!WARNING]
> A base model inside a chat template is not a broken chat model; it is a text continuer reading a strange document. If you want to test a base model on questions, either give it a plain-text prompt, or use few-shot prompting (a few example question-answer pairs before the real question), as GPT-3 did. Section 1.9 shows that, with the right context, a base model's next-token choices are already very close to the instruct model's.

## 1.9 How much does post-training really change?

Section 1.6 ended with a puzzle: post-training uses a tiny fraction of the compute and data of pretraining, yet it turns a text continuer into an assistant. Section 1.7 added a clue: the base model already knew the answers; what it lacked was the manners. Two papers from 2023 turned this clue into a hypothesis and a measurement.

### The superficial alignment hypothesis

> [!PAPER] Zhou et al. (2023), LIMA: Less Is More for Alignment · Section 2 · page 2
> [![The paragraph from LIMA defining the Superficial Alignment Hypothesis: a model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users. If this is correct, one could sufficiently tune a pretrained language model with a rather small set of examples.](/img/training/ch1-lima-hypothesis.png)](/img/training/ch1-lima-hypothesis.png)
>
> **Context:** the start of Section 2, where the authors explain why they fine-tuned a 65B LLaMa model on only 1,000 examples.
>
> **What it says:** "A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users." If so, a small set of good examples should be enough.
>
> **Why it matters:** it is a precise, testable claim about what post-training changes: style and format, not knowledge.

LIMA's evidence was indirect: a model fine-tuned on 1,000 examples, with no preference tuning at all, produced answers that human raters judged equivalent to or better than GPT-4's in 43% of cases (Zhou et al., 2023, Section 1). That shows a little SFT goes a long way. It does not show what changed inside the model.

> [!DEFINITION] Superficial alignment hypothesis
> The claim (Zhou et al., 2023) that almost all of a chat model's knowledge and abilities come from pretraining, and that alignment mainly selects the style and format of answers. "Superficial" here means "on the surface", not "unimportant".

### Measuring the change token by token

Lin et al. (2023) found a direct way to look. Take the instruct model's answer. At every position, ask the base model what it would have written next, given exactly the same context. If post-training changed everything, the base model would disagree often. If it only changed the surface, the base model would mostly agree.

> [!PAPER] Lin et al. (2023), The Unlocking Spell on Base LLMs (URIAL) · Figure 2 · page 3
> [![Figure 2 of the URIAL paper. An aligned model answers "What breed dog is the smallest?" and its answer is shown with each token coloured: blue for unshifted, brown for marginal, red for shifted. At each position the base model is decoded on the same context and the rank of the aligned token in the base model's list decides the colour. A word cloud shows the common shifted tokens such as Hello, However, Thank, and, must, and a ring chart shows 77.7% unshifted, 14.5% marginal and 7.8% shifted.](/img/training/ch1-urial-fig2.png)](/img/training/ch1-urial-fig2.png)
>
> **Context:** Section 2, "Alignment as token distribution shift", the analysis at the core of the paper.
>
> **What it says:** an aligned model (Llama-2-7b-chat) writes an answer with greedy decoding. At each position the base model (Llama-2-7b) sees the same context, and the aligned token is classed by its rank in the base model's list: "unshifted" if it is the base model's top-1, "marginal" if it is 2nd or 3rd, "shifted" otherwise. "On average, 77.7% of tokens are also ranked top 1 by the base LLM", "92.2% are within the top 3", and "knowledge-intensive tokens are predominantly found in unshifted positions."
>
> **Why it matters:** this turns the superficial alignment hypothesis into a number that anyone can measure on any base and instruct pair. We do exactly that next.

{{FIG:ch1_shift_method|The measurement at one position t. The instruct model's greedy token oₜ is compared with the base model's ranking for the same context, and the position is classed as unshifted, marginal or shifted. The KL divergence between the two full distributions is recorded too.}}

In symbols, at position $$t$$ of the instruct model's answer $$o = o_1 \dots o_T$$ to a query $$q$$:

$$
\eta_t = \text{rank of } o_t \text{ in } P_{\text{base}}(\cdot \mid q, o_{<t}), \qquad \text{class}_t = \begin{cases} \text{unshifted} & \eta_t = 1 \\ \text{marginal} & \eta_t \in \{2, 3\} \\ \text{shifted} & \eta_t > 3 \end{cases}
$$

where:

- $$q$$ is the user's query in the chat template, and $$o_{<t}$$ is the instruct model's answer up to position $$t-1$$;
- $$o_t$$ is the token the instruct model chose at position $$t$$ (with greedy decoding, its own top-1);
- $$P_{\text{base}}(\cdot \mid q, o_{<t})$$ is the base model's next-token distribution for the same context, the dot meaning "over all tokens";
- $$\eta_t$$ (Greek "eta") is the **base rank**: 1 if the base model would also have chosen $$o_t$$, 2 if it was its second choice, and so on.

### Our experiment

The script `ch1_token_shift.py` runs this on Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct, with 40 prompts of six kinds that we wrote for this chapter: knowledge questions ("Why is the sky blue?"), how-to and advice ("How do I make a cup of tea?"), creative writing ("Write a short poem about the ocean."), maths ("Is 97 a prime number?"), code ("Write a Python function that reverses a string.") and conversation ("Hello! How are you today?", "Tell me a joke."). The instruct model answers each with greedy decoding (up to 200 tokens). That gives 5,223 answer tokens.

```python
for q in PROMPTS:
    ctx = chat(q)                                   # the chat-template tokens of the query
    ans, stopped = greedy(tok, inst, ctx, 200)      # the instruct model's answer o_1 ... o_T
    full = ctx + ans
    li = log_softmax(inst(full).logits)[len(ctx)-1 : len(full)-1]   # row t predicts ans[t]
    lb = log_softmax(base(full).logits)[len(ctx)-1 : len(full)-1]   # same rows, from the base model
    for t, a in enumerate(ans):
        rank = (lb[t] > lb[t, a]).sum() + 1                          # 1 = base model's top-1
        kl = (li[t].exp() * (li[t] - lb[t])).sum()                   # KL(P_instruct || P_base), Section below
```

(simplified from `ch1_token_shift.py`). The trick is the same as in Section 1.3: a single forward pass over prompt plus answer gives the distribution at every position at once, so each model runs once per prompt, not once per token. The slice `[len(ctx)-1 : len(full)-1]` picks the rows that predict the answer tokens. The rank is computed by counting how many tokens the base model scored higher than the instruct model's choice.

{{FIG:ch1_shift_bars|Unshifted, marginal and shifted tokens. Top two rows: our measurement on Qwen2.5-0.5B, first with the identical chat-template context, then with the base model reading a plain "Question: ... Answer:" context instead. Bottom three rows: the three 7B pairs reported by Lin et al. (2023), Figure 3. ch1_token_shift.py.}}

```text
=== 40 prompts, 5223 answer tokens (format A: identical chat-template context) ===
unshifted (instruct token is base top-1): 86.7%
marginal  (base rank 2 or 3):              10.5%
shifted   (base rank > 3):                 2.8%
within base top-3:                         97.2%
format B (base sees "Question: ... Answer:"), 5200 tokens: unshifted 86.9%, within top-3 97.9%
```

**The result reproduces, and then some.** On 86.7% of the tokens of the instruct model's answers, the base model's own first choice was the very same token. On 97.2%, it was among the base model's top three. Only 2.8% of tokens are shifted. Lin et al. found 77.7% and 92.2% for Llama-2-7b-chat, a model 14 times larger from a different family trained with full RLHF, and about 82% unshifted for their two SFT-only pairs. Our pair agrees even more, which fits the point of Section 1.7: the Qwen2.5 base model has already seen a lot of assistant-style text.

We also checked that this is not an artefact of feeding the base model the chat template. In "format B", the base model reads `Question: <q>\nAnswer:` followed by the same answer text, with no special tokens at all. The numbers barely move (86.9% unshifted, 97.9% in the top three).

> [!WARNING]
> Decoding settings change this measurement. Qwen2.5-0.5B-Instruct ships with a **repetition penalty** of 1.1 in its generation settings, a rule that makes tokens already used a little less likely. Our first run of the script left it switched on by accident, so the instruct "greedy" answers were not quite the instruct model's own top choices. That run (saved as `results/ch1_token_shift_reppen1.1_stdout.txt`) gave 78.6% unshifted, 94.9% in the top three and 5.1% shifted. The numbers above use pure greedy decoding (`repetition_penalty=1.0`), as the method requires. The picture is the same, but the exact percentage depends on details like this, so always report how the answers were generated.

> [!PAPER] Lin et al. (2023), URIAL · Figure 3 · page 4
> [![Figure 3 of the URIAL paper: three bars of unshifted, marginal and shifted ratios for Llama-2-7b to Llama-2-7b-chat (77.7%, 14.5%, 7.8%), Llama-2-7b to Vicuna-7b-v1.5 (82.4%, 12.8%, 4.8%) and Mistral-7b to Mistral-7b-instruct (82.2%, 12.5%, 5.2%), each with a box listing frequently shifted tokens such as end-of-sequence, Thank, Hello, However, cannot, Sure, As.](/img/training/ch1-urial-fig3.png)](/img/training/ch1-urial-fig3.png)
>
> **Context:** the comparison across three base-and-aligned pairs at 7B scale, one trained with RLHF and two with SFT only.
>
> **What it says:** "The shifted token ratios are all very low (5%-7%)", and the pairs "share similar frequently shifted tokens, such as 'However', 'cannot', 'Here', 'To'". The first token in every list is `</s>`, Llama's end-of-sequence token.
>
> **Why it matters:** the pattern does not depend on the model family or on whether RLHF was used. Our end-of-turn finding below matches the `</s>` at the top of their lists.

### Which tokens shift

The script lists every shifted position with the tokens before it and the base model's preferred token. `ch1_shift_analysis.py` then sorts the 145 shifted tokens into groups:

| Kind of token | Shifted tokens | Share of shifted | Share of all answer tokens | Shift rate inside the group |
|---|---|---|---|---|
| end-of-turn `<\|im_end\|>` | 24 | 16.6% | 0.5% | 100% |
| first token of the answer | 11 | 7.6% | 0.8% | 27.5% |
| punctuation, spacing, line breaks | 8 | 5.5% | 18.5% | 0.8% |
| other words | 102 | 70.3% | 80.2% | 2.4% |

The last column is the one to read. The end-of-turn token is shifted every time, the first token of an answer more than a quarter of the time, and an ordinary word only about one time in forty.

Four patterns stand out.

**1. The end of the turn.** The single most shifted token is `<|im_end|>`. It ends 24 of the 40 answers (the other 16 hit the 200-token limit), and it is shifted in all 24. The instruct model gives it an average probability of 0.81; the base model, on the same context, essentially zero (at most 0.00003). The base model's favourite at those positions was most often `<|endoftext|>`: it, too, thinks the text is over, but it uses the end-of-document marker from pretraining. This is the turn-taking behaviour of Section 1.8, measured.

**2. The first token of the answer.** How to open an answer is a style decision, and the two models often disagree. Of the 40 first tokens, 11 are shifted and 9 more are marginal. The base model's favourite openers are "The", "Sure", "To", "You" and "Hi"; the instruct model prefers "Certainly" (for the sleep tips and the Python function), "To" for maths ("To find..."), a quotation mark for a slogan or a bakery name, and "Dear" rather than "Hi" for a thank-you note. Notice that the base model's top choice was "Sure" on six prompts: it already knows how an assistant opens a reply, again a sign of the assistant-style data in its pretraining.

**3. Formatting, less than you might expect.** A few shifts are paragraph breaks (`".\n\n"` where the base model wanted `"."`). But punctuation and spacing tokens as a group shift less than 1% of the time, less than words do. Both models format answers in much the same way.

**4. Word choice.** Most shifted tokens (70%) are ordinary words, and many are synonyms or near-synonyms: `" marks"` where the base model wanted `" occurs"` (a sunset "marks" the end of the day), `" display"` for `" sight"`, `" engaging"` for `" clear"`, `" Certainly"` for `" Sure"`. Creative writing and chat have the most of these, because there is no single right next word in a poem or a greeting.

And on the other side, look at what does **not** shift. The facts in the answers are unshifted: `" Canberra"` for the capital of Australia, `" Leonardo"` `" da"` `" Vinci"` for the Mona Lisa, `" Aust"` `"en"` for Pride and Prejudice. Qwen's tokenizer writes every digit as its own token, so the 206 bones of the human body are three tokens, `"2"` `"0"` `"6"`, all unshifted. Over all answers, 245 tokens are digits (dates, quantities, results of sums): 98.8% of them are unshifted and none is shifted. The base model would have written the same facts.

{{FIG:ch1_strip|The beginning of one instruct answer, every token coloured by its class. Most tokens are ones the base model would also have chosen; the few shifted ones are the opener and some word choices. Hover a token to see its base rank. ch1_token_shift.py.}}

We should also be honest about what the list shows that the paper's summary does not. A few shifted tokens change **what is said**, not only how. Asked "How do I pick a lock?", the instruct model begins "Picking a lock can be a fun ..." where the base model wanted "a bit ...". Asked "Hello! How are you today?", it says "I'm just a large language model" where the base model wanted "a system". In "How do I stay focused while studying?" the list of tips itself differs ("Set" a goal where the base model wanted "Take" a break). These are small, but they are not style: post-training changed the model's view of what it is and what it should say. At 0.5B parameters, neither model is reliable on facts, and our 40 prompts are too few to find factual changes reliably.

### KL divergence: how different are two distributions?

The rank tells us whether the models agree on the top choice. It does not tell us how much the whole distribution moved. For that we use the **Kullback-Leibler divergence**.

$$
\mathrm{KL}(P \,\|\, Q) = \sum_{v=1}^{V} P(v) \log \frac{P(v)}{Q(v)}
$$

where:

- $$P$$ and $$Q$$ are two probability distributions over the same vocabulary, here $$P = P_{\text{instruct}}$$ and $$Q = P_{\text{base}}$$ at one position;
- the sum runs over all $$V$$ tokens $$v$$;
- $$\log \frac{P(v)}{Q(v)}$$ is positive where $$P$$ gives a token more probability than $$Q$$, and negative where it gives less;
- each term is weighted by $$P(v)$$, so tokens the instruct model considers likely count most.

> [!DEFINITION] KL divergence
> A measure of how different a distribution $$P$$ is from a reference distribution $$Q$$, in nats. It is 0 only when the two are identical and positive otherwise. It is not symmetric: $$\mathrm{KL}(P \,\|\, Q)$$ and $$\mathrm{KL}(Q \,\|\, P)$$ are usually different numbers. It can be read as the extra loss you would pay, on average, if the text really came from $$P$$ but you predicted it with $$Q$$.

This is the same KL that appeared in the RLHF objective of Section 1.6, where it keeps the trained model close to the starting one. Here we use it as a ruler.

**Worked example.** `ch1_kl_example.py` computes the KL at two real positions. The first is the very first token of the answer to "Give me three tips for sleeping better." (a style decision). The second is the token after "The capital of Australia is" in an answer about Australia (a fact).

{{FIG:ch1_kl|KL divergence at two real positions, the three most likely instruct tokens at each. Left: how to open an answer; the instruct model moved most of its probability to "Certainly". Right: a fact; both models put about 90% or more on " Canberra". ch1_kl_example.py.}}

For the style position, the four biggest terms of the sum are:

| Token | $$P_{\text{instruct}}$$ | $$P_{\text{base}}$$ | $$P \log(P/Q)$$ |
|---|---|---|---|
| Certainly | 0.5666 | 0.0884 | +1.052 |
| Sure | 0.1761 | 0.3533 | −0.123 |
| 1 | 0.1115 | 0.0736 | +0.046 |
| Here | 0.0232 | 0.0583 | −0.021 |
| all other 151,932 tokens | | | +0.025 |
| **total** | | | **0.980 nats** |

Check the first term by hand: $$0.5666 \times \log(0.5666 / 0.0884) = 0.5666 \times \log 6.41 = 0.5666 \times 1.858 = 1.052$$. The instruct model made "Certainly" 6.4 times more likely than the base model did, and since it gives that token 57% of its probability, this one term dominates. "Sure" contributes a negative term because the instruct model gives it less probability than the base model (the total is still positive; KL always is).

For the fact position, the KL is only **0.046 nats**. The base model already put 0.893 on `" Canberra"`; the instruct model puts 0.963. The two models agree, with the instruct model a little more confident. Twenty times less divergence than at the style position.

### Across all 5,223 tokens

```text
KL(P_instruct || P_base) per token: mean 0.216 nats, median 0.041, 90th percentile 0.323, max 28.56
  mean KL at unshifted positions: 0.085  (4530 tokens)
  mean KL at marginal  positions: 0.371  (548 tokens)
  mean KL at shifted   positions: 3.743  (145 tokens)
  shifted positions are 2.8% of tokens but carry 48.1% of the total KL
```

The median token has a KL of 0.04 nats, about the same as the Canberra example: the two models have nearly the same distribution. The mean is five times larger than the median because a few positions are wildly different: the largest values (up to 28.6 nats) are at `<|im_end|>`, where the instruct model is sure and the base model gives almost nothing. **The 2.8% of shifted positions carry almost half (48%) of all the divergence.** Post-training changed a few positions a lot and most positions hardly at all.

The change is also concentrated at the start of answers:

{{FIG:ch1_bypos|Shifted tokens and mean KL by position in the answer, over all 40 answers. The first five tokens are shifted three to six times as often as tokens after the 50th, and their mean KL is about four times higher. ch1_token_shift.py.}}

Once the answer is under way, the instruct model's own earlier tokens are in the context, and the base model, reading them, is pulled along into the same style. Lin et al. report the same decline (their Figure 4). It is also why the URIAL paper's main proposal works: if you give a base model a good start, with a few carefully written example answers in its prompt (URIAL: "Untuned LLMs with Restyled In-context ALignment", with as few as three examples and a system prompt), it can answer like a chat model with no training at all.

The kind of prompt matters too. `ch1_shift_analysis.py` splits the result by the six groups of prompts:

| Kind of prompt | Answer tokens | Unshifted | Shifted | Mean KL (nats) |
|---|---|---|---|---|
| knowledge | 1,041 | 86.2% | 3.4% | 0.271 |
| how-to, advice | 1,407 | 86.4% | 2.3% | 0.130 |
| creative writing | 663 | 79.5% | 4.8% | 0.318 |
| maths | 726 | 94.5% | 1.0% | 0.214 |
| code | 800 | 89.8% | 1.2% | 0.108 |
| chat, identity | 586 | 83.1% | 4.8% | 0.361 |

Maths and code shift least: there is usually one right next token in a calculation or a line of code, and the base model already knows it. Creative writing and chat shift most: there are many good next words, and post-training has taught the instruct model its own preferences among them. These are small groups (about 600 to 1,400 tokens each), so treat the differences as indications, not precise measurements.

### What the experiment shows, and what it does not

What it shows: for this pair of models, on these prompts, post-training moved the chosen token out of the base model's top three at fewer than 1 position in 30, and the changes are the end of the turn (always), the opening of the answer (often), and word choice here and there, most of all in creative writing. The factual tokens are the same ones the base model would have produced. This is strong support for the superficial alignment hypothesis, measured on our own models, and it matches the 7B results of Lin et al.

What it does **not** show:

- **It measures agreement on the instruct model's path.** At every position the base model is given the instruct model's previous tokens. It is never allowed to go its own way. The base model's first choice differs from the instruct model's at 13% of positions (all the marginal and shifted ones), so a base model decoding on its own leaves the instruct model's path within a few tokens, and then the contexts are no longer the same. Section 1.7 showed where it can end up. Small local differences add up to a large difference in behaviour.
- **A few tokens can matter a lot.** Refusing a harmful request, saying "I don't know", or stopping at the right moment are single tokens. 3% of tokens is not 3% of the value.
- **Greedy answers to easy prompts.** Our 40 prompts are short and ordinary. Hard reasoning, long conversations, tool use and safety-critical prompts are where RL stages make their biggest changes, and they are not tested here. Section 1.6 noted that RLVR changes reasoning substantially; this experiment cannot see that.
- **Qwen2.5 base is not a pure base model.** Its pretraining included synthetic assistant-written data (Section 1.6), so part of "alignment" may have happened before post-training even began. Our 86.7% may be higher than it would be for a model pretrained only on human-written text.
- **One small pair, 40 prompts, one decoding method.** The numbers have sampling noise; another 40 prompts would give somewhat different percentages, and the repetition-penalty warning above shows that a small change in decoding moves them by several points.

"Superficial" is not "unimportant". The superficial part is what makes the model usable. But the evidence says that what post-training mostly does is select a style the base model already had available, rather than teach it new things to say.

## 1.10 A tiny training loop: SFT by hand

Everything so far has been measurement. Now we change some weights. The script `ch1_sft_tiny.py` takes the base model, Qwen2.5-0.5B, and fine-tunes all 494 million weights on 8 short chat examples for 30 steps. It is the SFT stage of Section 1.6 at the smallest possible scale, written out by hand so that every line is visible. It runs in a few minutes on a laptop GPU.

### The data, and the masking that makes it SFT

Eight prompt-and-answer pairs such as ("What is the capital of Japan?", "The capital of Japan is Tokyo.") and ("Give me one tip for staying hydrated.", "Carry a water bottle and sip from it throughout the day."), plus three held-out pairs the model never trains on, to see whether what it learns carries over. Each pair becomes one token sequence and one list of labels:

```python
def encode(q, a):
    msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': q}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)   # prompt, ends with "<|im_start|>assistant\n"
    r = tok(a)['input_ids'] + [END]                                               # answer tokens + <|im_end|>
    return p + r, [-100] * len(p) + r                                             # input ids, labels
```

Line by line: the prompt is built with the chat template, exactly as at inference time, and ends by opening the assistant's turn. The response is the answer's tokens followed by `<|im_end|>`, the token that ends the turn. The input is the two joined together. The labels are the same tokens, except that every prompt position is replaced by **−100**. PyTorch's cross-entropy function skips any position whose label is −100. So the model reads the whole prompt, but is only graded on the answer and the closing `<|im_end|>`. That one line is the difference between the SFT loss and the pretraining loss of Section 1.6.

> [!DEFINITION] Label masking (loss masking)
> Excluding some positions from the loss by giving them a special label (−100 in PyTorch and the Hugging Face libraries). In SFT the prompt is masked, so the model is not trained to write user messages or system prompts, only the assistant's reply.

{{FIG:ch1_mask|The first training example, token by token. Grey tokens (the system prompt, the user turn and the assistant header) have label −100 and do not count. Only the 7 answer tokens and <|im_end|> are trained on. ch1_sft_tiny.py.}}

Of the 34 tokens in this example, only 8 are trained on. The script prints the labels so you can check:

```text
  '<|im_start|>'   label -100
  'system'         label -100
  ...
  'assistant'      label -100
  '\n'             label -100
  'The'            label = 785
  ' capital'       label = 6722
  ' of'            label = 315
  ' Japan'         label = 6323
  ' is'            label = 374
  ' Tokyo'         label = 26194
  '.'              label = 13
  '<|im_end|>'     label = 151645
34 tokens, 8 of them are trained on
```

Why mask the prompt? Three reasons. First, we want to teach the model to answer, not to write questions; training on the prompt would spend effort predicting the user's words. Second, the system prompt and template are identical in every example, so without masking a large share of the loss would be spent on the same easy tokens. Third, it matches what the model will be asked to do at inference time: continue after `<|im_start|>assistant\n`.

The eight examples have different lengths, so they are padded to the same length with `<|endoftext|>`, and the padding gets label −100 too, plus an **attention mask** of 0 so that no real token pays attention to it.

### The loop

{{FIG:ch1_step|One training step: forward pass, loss, backward pass, optimizer update. Then the gradients are cleared and the loop repeats.}}

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
model.train()
for step in range(1, 31):
    out = model(input_ids=ids, attention_mask=att, labels=lab)   # 1. forward pass, 2. loss
    out.loss.backward()                                          # 3. backward pass: gradients
    opt.step()                                                   # 4. update the weights
    opt.zero_grad()                                              #    clear the gradients
```

(simplified from `ch1_sft_tiny.py`, which also evaluates the held-out examples after every step). Every line, in order:

- `torch.optim.AdamW(model.parameters(), lr=1e-5)` creates the **optimizer**, the rule that changes the weights. It is told which numbers it may change (all of them) and the **learning rate**, the size of each step. 1e-5 (0.00001) is a typical learning rate for fine-tuning a whole model; Qwen2.5's own SFT used 7e-6 decaying to 7e-7 (Qwen Team, 2024, Section 4.1).
- `model.train()` switches the model into training mode (for models with dropout this turns dropout on; Qwen2.5 has none, but it is a good habit).
- `model(input_ids=..., attention_mask=..., labels=...)` is the **forward pass**: it computes the logits at every position of all 8 sequences, then the cross-entropy loss of Section 1.5, averaged over every position whose label is not −100. `out.loss` is one number.
- `out.loss.backward()` is the **backward pass**. PyTorch works backwards through every operation of the forward pass and computes, for each of the 494 million weights, the **gradient**: how much the loss would go up if that weight went up a little. This is **backpropagation**.
- `opt.step()` changes every weight a little in the direction that makes the loss go down. AdamW scales each weight's step using running averages of its past gradients, which makes training much more stable than plain gradient descent.
- `opt.zero_grad()` resets the gradients to zero. PyTorch adds new gradients to old ones by default, so without this line each step would use the sum of all previous gradients.

> [!DEFINITION] Gradient
> For each weight, the rate at which the loss changes when that weight changes. A positive gradient means "increasing this weight increases the loss", so the update moves the weight down. The full list of gradients has the same size as the model: 494 million numbers here.

> [!DEFINITION] Learning rate
> How big a step the optimizer takes. Too large and training becomes unstable or forgets what the model knew; too small and nothing changes. Fine-tuning uses much smaller learning rates than pretraining.

> [!DEFINITION] AdamW
> The standard optimizer for training transformers. It keeps, for every weight, a running average of its gradients and of their squares, and uses them to choose a separate step size per weight. The "W" is for decoupled weight decay, a gentle pull of all weights towards zero (switched off here).

### The output

```text
step  0: train loss    -     held-out loss 2.252   P(<|im_end|>) on held-out 0.0000
step  1: train loss 2.442   held-out loss 1.691   P(<|im_end|>) on held-out 0.0001
step  2: train loss 1.791   held-out loss 1.409   P(<|im_end|>) on held-out 0.0001
step  3: train loss 1.248   held-out loss 1.387   P(<|im_end|>) on held-out 0.0001
step  4: train loss 1.035   held-out loss 1.342   P(<|im_end|>) on held-out 0.0001
step  5: train loss 0.894   held-out loss 1.342   P(<|im_end|>) on held-out 0.0002
step 10: train loss 0.784   held-out loss 1.566   P(<|im_end|>) on held-out 0.0003
step 20: train loss 0.683   held-out loss 1.642   P(<|im_end|>) on held-out 0.0009
step 30: train loss 0.575   held-out loss 1.593   P(<|im_end|>) on held-out 0.0027

held-out prompt: 'What is the capital of Italy?'
  before (stopped): 'The capital of Italy is Rome. Roma is the largest city in Italy and is also the capital of the country.<|endoftext|>'
  after  (stopped): 'The capital of Italy is Rome.<|im_end|>'
held-out prompt: 'How many legs does a spider have?'
  before (stopped): "A spider has 8 legs.\n戥user\nThat's correct! A spider has 8 legs.<|endoftext|>"
  after  (stopped): 'A spider has 8 legs.<|im_end|>'
held-out prompt: 'Give me one tip for better sleep.'
  before (did not stop): 'Sure, here are some tips for better sleep:\n\n1. Establish a regular sleep schedule: ...'
  after  (stopped): 'Carry a water bottle and sip from it throughout the day.<|im_end|>'
```

The "held-out loss" is the same SFT loss computed on the three examples the model never trains on, and "P(<|im_end|>)" is the probability the model gives to the end-of-turn token right after each held-out answer, averaged over the three.

{{FIG:ch1_sft|Two tiny SFT runs. Left and middle: training loss on the 8 examples (falls) and held-out loss on 3 new examples (falls, then rises: overfitting), at learning rates 1e-5 and 1e-4. Right: the probability of <|im_end|> after the held-out answers; only the larger learning rate learns it. ch1_sft_tiny.py.}}

Read the log from top to bottom.

**The training loss goes down, as it must.** From 2.442 at step 1 to 0.575 at step 30. Gradient descent on 8 fixed examples will always make their loss smaller.

**The held-out loss goes down, then up.** It falls from 2.252 to 1.342 in four steps: the model is learning something general about the format (answer briefly, in the assistant's turn). After that it climbs back to 1.593. The model has started to learn these 8 particular answers rather than the general habit. This is **overfitting**, and with 8 examples it starts almost at once.

> [!DEFINITION] Overfitting
> When a model gets better on its training examples but worse on new ones, because it is memorising the specific examples instead of learning the general pattern. Watching a held-out loss is the standard way to catch it.

**The answers change shape.** Before training, in the chat template, the base model answered "Rome" and then kept talking, or wrote a fake user turn ("戥user That's correct!"), and ended with `<|endoftext|>`, never `<|im_end|>`. After 30 steps, all three answers are one short sentence ending in `<|im_end|>`. The style of the 8 examples (one sentence, then stop) has been copied.

**But one answer is the wrong answer.** Asked for one tip for better sleep, the trained model replies "Carry a water bottle and sip from it throughout the day.", word for word the training answer for staying hydrated. That is memorisation in plain sight: the prompt "Give me one tip for ..." has been tied to one specific answer.

**And it stops only by a hair.** The log says the probability of `<|im_end|>` after the held-out answers is only 0.0027. How can greedy decoding pick a token with probability 0.27%? The script's check prints the top five tokens at that position:

```text
  top-5 after the answer: '<|im_end|>' 0.0027, '<|im_start|>' 0.0010, 'oubted' 0.0006, 'powiedzieć' 0.0005, '看查看' 0.0005   (entropy 8.72 nats)
```

The distribution is almost flat: an entropy of 8.72 nats means the model is as unsure as if it were choosing among about $$e^{8.72} \approx 6{,}000$$ tokens. `<|im_end|>` is the most likely, but only just. With sampling instead of greedy decoding, this model would almost never stop.

The reason is in the base model's weights. The script also measures the rows of the output table for `<|im_start|>`, `<|im_end|>` and the 290 unused rows after them:

```text
base model output rows from <|im_start|> (151644) to 151935: 292 rows, cosine similarity of <|im_end|> to the others: min 1.0000, mean 1.0000
```

A cosine similarity of 1 means the vectors point in exactly the same direction. In the base model, the `<|im_end|>` row is indistinguishable from 291 rows that are never used. The base model never trained it, so it is not a "stop" token yet; it is a placeholder. SFT has to build that meaning from scratch, and 30 small steps are not enough.

**A larger learning rate.** Running the same script with `python ch1_sft_tiny.py 1e-4` (ten times larger steps) shows the other side of the trade-off:

```text
step 10: train loss 0.344   held-out loss 1.677   P(<|im_end|>) on held-out 0.0646
step 15: train loss 0.034   held-out loss 1.805   P(<|im_end|>) on held-out 0.8395
step 30: train loss 0.000   held-out loss 2.298   P(<|im_end|>) on held-out 0.9985

held-out prompt: 'How many legs does a spider have?'
  after  (stopped): 'Carry a spider and sip from it throughout the day.<|im_end|>'
```

Now `<|im_end|>` is learned properly: probability 0.9985, higher than the instruct model's 0.81 average from Section 1.9. But the training loss is 0.000 (every training answer is memorised exactly), the held-out loss has risen to 2.298, above where it started, and the spider question now gets "Carry a spider and sip from it throughout the day." The model learned when to stop and forgot how to answer.

This is the whole problem of SFT in miniature. The format, including the end-of-turn token, has to be learned, and that takes enough training. But with too few and too similar examples, the same training memorises instead of generalising. Real SFT solves it with data: hundreds of thousands of varied examples (Qwen2.5 used over a million), so that the only thing common to all of them is the format and the helpful style. LIMA's 1,000 examples worked because they were carefully varied, and because the base model was 130 times larger than ours. A later part of the book covers SFT data, learning-rate schedules and how to tell when to stop.

> [!TIP]
> Run `ch1_sft_tiny.py` and change one thing at a time: the learning rate (1e-6, 1e-4), the number of steps, or remove the masking (set the labels equal to the input ids). Watch what happens to the held-out loss and to the "after" answers. Removing the mask, for example, makes the model spend its training on the system prompt it already knows by heart.

## 1.11 How do we know it got better? A first look

Every stage of the pipeline claims to make the model better. How would you check? Chapter 3 is about evaluation in depth; here is a first taste, with a tiny test anyone can run.

The script `ch1_eval_preview.py` asks 20 short questions that each have one clear answer: four capitals, two authors and scientists, six arithmetic questions, and some general knowledge ("What is the largest ocean on Earth?", "What is the longest river in Africa?"). For each model setting it checks two things: does the expected answer (for example "Nairobi", "72", "Pacific") appear anywhere in the output, and did the model end its turn by itself within 60 new tokens? Greedy decoding again.

{{FIG:ch1_eval|A first evaluation on 20 short questions. Left: how often the expected answer appears in the first 60 new tokens. Right: how often the model ended its turn within those 60 tokens. ch1_eval_preview.py.}}

```text
base, raw question       answer found: 12/20   ended its turn by itself:  9/20
base, chat template      answer found: 14/20   ended its turn by itself:  5/20
instruct, chat template  answer found: 18/20   ended its turn by itself: 16/20
```

At first sight the instruct model is much better: 18 of 20 against 12, and it ends its turn on 16 of 20 questions against 9. But read the misses before you believe a score. The script prints every one:

- **All six arithmetic misses of the raw base model are cut-offs, not wrong answers.** For "What is 9 times 8?" it begins "To find the product of 9 and 8, we can use the standard multiplication method. Here are t..." and runs out of its 60-token budget before reaching 72. With a larger budget it would probably have scored more. The test, not only the model, decided that result.
- **Two raw base misses are "wrong genre".** "Who wrote Romeo and Juliet?" is continued with more questions ("What is the name of the play? What is the name of the author?..."), and "How many continents are there?" becomes a multiple-choice exam ("A. 1 B. 2 C. 3 D. 4 Answer: C"). The knowledge may be there; the format is not.
- **The base model in the chat template fails differently:** for four questions it falls straight into a loop of junk tokens (`" Comey"`, `" Cherokee"`, `"ounces"` repeated), and once it states a wrong fact ("plants take in oxygen from the air").
- **The instruct model's two misses are real errors, stated confidently.** "There are currently 5 continents: Asia, Africa, North America, South America, and Europe." and "There are 60 minutes in two hours." Post-training made the answers short, polite and well-formed; it did not make a 0.5B model right. The base model, reading the same question as plain text, at least started to work out the minutes step by step.
- **"Ended its turn" is also budget-bound.** The four instruct answers that did not stop within 60 tokens were explanations that kept going (about relativity, the Pacific, photosynthesis and the Nile). Whether that counts as a failure depends on whether you wanted a one-word answer.

Four lessons, which Chapter 3 develops properly:

1. **A score is a measurement procedure, not a property of the model.** The prompt format, the token budget and the matching rule changed our numbers as much as the model did.
2. **Read the outputs.** A substring check counts "Answer: C" as a miss and would count "not 72" as a hit. Every automatic metric needs a look at real examples.
3. **Measure what you care about.** If you want an assistant, measure answers in the assistant format and also how it behaves (stopping, length, refusals), not only whether a fact appears.
4. **Twenty questions is a demonstration, not an evaluation.** With 20 items, one question is 5 percentage points. Real benchmarks use thousands of items, careful answer checking, and guard against test questions having leaked into the training data.

The published reports do this at scale. The InstructGPT result quoted in Section 1.6 (labelers preferred the 175B InstructGPT 85% of the time) comes from human comparisons of outputs on held-out prompts; the Qwen2.5 and Llama 3 reports each list dozens of benchmarks for knowledge, maths, code and instruction following before and after post-training. Chapter 3 explains how such numbers are made and how far to trust them.

## Takeaways

> [!TAKEAWAYS]
> - A language model is a function from a context of tokens to one probability for every possible next token. Text is generated one token at a time by picking from that distribution and running the model again.
> - Softmax turns raw scores (logits) into probabilities; a logit 1 higher means about 2.72 times more probable. After "The capital of France is", Qwen2.5-0.5B gives " Paris" 0.316 and spends 23% on quiz-style blanks.
> - The chain rule makes the probability of a sentence the product of next-token probabilities. Training minimises the average negative log of those probabilities, the cross-entropy loss; perplexity is its exponential. Our example sentence: loss 2.559 nats per token, perplexity 12.9.
> - Training happens in stages: pretraining (trillions of tokens, next-token loss), mid-training (curated data, same loss), SFT (prompt and answer, loss on the answer only), preference tuning (RLHF with a reward model, or DPO on chosen and rejected pairs) and RL with verifiable rewards (reward 1 for a checked correct answer).
> - Post-training is cheap in compute compared with pretraining: for InstructGPT, SFT took 4.9 and RLHF 60 petaflop/s-days, against 3,640 for pretraining GPT-3.
> - The chat template writes the conversation into one token sequence with special tokens. The base model never learned to write `<|im_end|>`, which is why it cannot end its turn; teaching that is a big part of SFT.
> - Measured token by token, post-training changed little: on 86.7% of the instruct model's answer tokens the base model's first choice was the same token, on 97.2% it was in the top three. The shifted 2.8% (the end-of-turn token every time, answer openers often, word choices now and then) carry 48% of the KL divergence; digits and named facts stay unshifted. This supports the superficial alignment hypothesis, with real limits: the base model is never allowed off the instruct model's path, a few tokens can matter a lot, and decoding settings move the numbers.
> - SFT is the pretraining loop with the prompt masked out (label −100). With 8 examples and 30 steps, a small learning rate copies the short-answer format but barely learns `<|im_end|>` (an untrained row in the base model); a larger one learns it (probability 0.9985) and memorises the training answers. More, and more varied, data is the real fix.
> - "Better" needs a test you define in advance and that matches what you care about. Chapter 3 is about building such tests.

## References

**Papers**

- Brown, T. et al. (2020). *Language Models are Few-Shot Learners* (GPT-3). [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Chung, H. W. et al. (2022). *Scaling Instruction-Finetuned Language Models* (Flan-T5, Flan-PaLM). [arXiv:2210.11416](https://arxiv.org/abs/2210.11416)
- DeepSeek-AI (Guo, D. et al.) (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- Grattafiori, A. et al. (2024). *The Llama 3 Herd of Models*. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- Lambert, N. et al. (2024). *Tulu 3: Pushing Frontiers in Open Language Model Post-Training*. [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)
- Lin, B. Y. et al. (2023). *The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning* (URIAL). [arXiv:2312.01552](https://arxiv.org/abs/2312.01552)
- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback* (InstructGPT). [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Qwen Team (2024). *Qwen2.5 Technical Report*. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
- Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (introduces GRPO). [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- Team OLMo (Walsh, P. et al.) (2024). *2 OLMo 2 Furious*. [arXiv:2501.00656](https://arxiv.org/abs/2501.00656)
- Vaswani, A. et al. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Wei, J. et al. (2021). *Finetuned Language Models Are Zero-Shot Learners* (FLAN). [arXiv:2109.01652](https://arxiv.org/abs/2109.01652)
- Zhou, C. et al. (2023). *LIMA: Less Is More for Alignment*. [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)

**Other sources**

- Qwen2.5-0.5B model card. [huggingface.co/Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- Qwen2.5-0.5B-Instruct model card. [huggingface.co/Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- Hugging Face Transformers documentation, *Chat templates*. [huggingface.co/docs/transformers/chat_templating](https://huggingface.co/docs/transformers/chat_templating)
- PyTorch documentation, *CrossEntropyLoss* (the `ignore_index=-100` used for label masking). [pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- Code for this chapter: `code/training/ch1_nexttoken.py`, `ch1_base_vs_instruct.py`, `ch1_token_shift.py`, `ch1_shift_analysis.py`, `ch1_kl_example.py`, `ch1_sft_tiny.py`, `ch1_eval_preview.py`, with their outputs in `code/training/results/`.
