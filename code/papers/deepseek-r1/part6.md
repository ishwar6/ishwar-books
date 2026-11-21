---
title: "Distillation: Small Models That Reason"
description: "How 800k answers written by R1 teach six small open models to reason with plain fine-tuning, and why this beat RL on the same model. We write the loss by hand and test a distilled 1.5B model."
part: 6
covers: "§2.4, §3.2, Table 5, §4.1, Table 6"
date: 2026-11-21
tags: [deepseek, distillation, knowledge-distillation, sft, reasoning]
---

DeepSeek-R1 has 671 billion parameters, and few people can run it. Yet its reasoning can be copied into a 1.5-billion-parameter model that runs on a laptop. The small model needs no reinforcement learning: it only reads R1's written answers.

## The promise {§1}

> [!PAPER] · §1 · page 3
> [![The paragraph of the introduction about distillation, with direct distillation from DeepSeek-R1 outperforms applying RL on it, reasoning patterns discovered by larger base models are crucial, and distilled 14B model outperforms state-of-the-art open-source QwQ-32B-Preview highlighted](/img/papers/deepseek-r1/p6-intro.png)](/img/papers/deepseek-r1/p6-intro.png)
>
> **What it says:** on Qwen2.5-32B, "direct distillation from DeepSeek-R1 outperforms applying RL on it". The authors conclude that "the reasoning patterns discovered by larger base models are crucial". The distilled 14B model beats QwQ-32B-Preview, and the distilled 32B and 70B models "set a new record" among dense models.
>
> **What it means:**
>
> The introduction gives the whole result of this part in four sentences. There are two ways to give a 32-billion-parameter model the skill to reason. The first way is the R1-Zero way: reinforcement learning on the model itself. The second way is to let it copy the answers of a bigger model that already reasons. The second way won.
>
> The reason the authors give is in the phrase "reasoning patterns discovered by larger base models". A pattern here is a habit of thought: try a method, check the result, notice an error, go back. R1 found these habits through RL on a 671B model. A small model that copies them does not have to find them again.
>
> "Dense" means that every parameter works on every token. R1 is a mixture-of-experts model, so only a part of it runs for each token. The students are ordinary dense models from the Qwen2.5 and Llama 3 families.

> [!PAPER] · §1.1 · page 4
> [![The Distillation contribution bullets, with reasoning patterns of larger models can be distilled into smaller, better performance compared to the reasoning patterns discovered, and DeepSeek-R1-Distill-Qwen-7B achieves 55.5% on AIME 2024 highlighted](/img/papers/deepseek-r1/p6-contrib.png)](/img/papers/deepseek-r1/p6-contrib.png)
>
> **What it says:** "the reasoning patterns of larger models can be distilled into smaller models, resulting in better performance compared to the reasoning patterns discovered through RL on small models." The distilled 7B gets 55.5% on AIME 2024; the distilled 32B gets 72.6% on AIME 2024 and 94.3% on MATH-500. Six checkpoints are released: 1.5B, 7B, 8B, 14B, 32B and 70B.
>
> **What it means:**
>
> The first bullet is a comparison, not only a claim that distillation works. It puts two sources of reasoning side by side: patterns that a large model found, and patterns that a small model finds by itself. The large model's patterns win. Section 4.1 is the experiment behind this sentence, and the end of this page reads it.
>
> The second bullet is about weights you can download. The six models are on Hugging Face under names such as DeepSeek-R1-Distill-Qwen-1.5B. That is the model we run on every page of this series, and it is the reason a laptop can show what R1 does.

## Background: what distillation is, and where it came from

The word comes from chemistry. You boil a liquid, keep the vapour and cool it again, and you get the essence without the bulk. In machine learning, the "liquid" is a large model, and the "essence" is the behaviour we want a small model to copy.

> [!DEFINITION] Knowledge distillation
> Training a small model (the **student**) to behave like a large trained model (the **teacher**). The student learns from the teacher's outputs, not only from the original labels. The teacher stays frozen; only the student changes.

### Soft labels: the first form

> [!PAPER] Hinton et al. (2015), Distilling the Knowledge in a Neural Network · §1 · page 2
> [![A paragraph of Hinton, Vinyals and Dean, with use the class probabilities produced by the cumbersome model as soft targets highlighted](/img/papers/deepseek-r1/p6-hinton.png)](/img/papers/deepseek-r1/p6-hinton.png)
>
> **What it says:** to move what a large ("cumbersome") model knows into a small one, "use the class probabilities produced by the cumbersome model as 'soft targets'" for the small model. Soft targets carry "much more information per training case" than the hard labels.
>
> **What it means:**
>
> A classifier ends with a list of probabilities, one for each class. The normal label of a photo of a dog says "dog: 1, everything else: 0". That is a **hard label**. The teacher's output says something like "dog: 0.9, wolf: 0.08, cat: 0.02". That is a **soft label**, and the student is trained to match the whole list.
>
> The paper also uses a temperature $$T$$ in the softmax to make the list flatter, so that the small numbers become visible. We do not need the temperature here. The key idea is that the student copies a distribution, not a single answer.

> [!PAPER] Hinton et al. (2015), Distilling the Knowledge in a Neural Network · §1 · page 2
> [![The BMW example of Hinton, Vinyals and Dean, with relative probabilities of incorrect answers, An image of a BMW and mistaken for a garbage truck highlighted](/img/papers/deepseek-r1/p6-hinton-bmw.png)](/img/papers/deepseek-r1/p6-hinton-bmw.png)
>
> **What it says:** "The relative probabilities of incorrect answers" show how the teacher generalises. "An image of a BMW" has a very small chance of being "mistaken for a garbage truck", but that mistake is many times more probable than a mistake for a carrot.
>
> **What it means:**
>
> This is why soft labels help. A hard label says only "BMW". The teacher's probabilities also say "this looks a little like a truck, and not at all like a vegetable". That is knowledge about how the classes relate, and the student gets it for free.
>
> Keep this picture in mind. In R1's case the "classes" are the next tokens of a text. The question is whether the student copies the teacher's full list of probabilities at each token, or only the text that the teacher wrote.

### Sequence-level distillation: copy the text

> [!PAPER] Kim and Rush (2016), Sequence-Level Knowledge Distillation · §1 · page 2
> [![A paragraph of Kim and Rush, with sequence-level (as opposed to word-level) and trained on a newly generated dataset that is the result of running beam search with the teacher network highlighted](/img/papers/deepseek-r1/p6-kimrush.png)](/img/papers/deepseek-r1/p6-kimrush.png)
>
> **What it says:** for translation, the student can match the teacher at the "sequence-level (as opposed to word-level)". In practice the student is "trained on a newly generated dataset that is the result of running beam search with the teacher network".
>
> **What it means:**
>
> A translation model makes a sentence one word at a time. Word-level distillation applies Hinton's idea at every word: the student matches the teacher's probabilities for each next word. Sequence-level distillation is simpler. The teacher writes whole translations, and the student is trained on them as if they were the correct answers.
>
> This turns distillation into ordinary fine-tuning on a new dataset. The student never sees the teacher's probabilities, only its text. Kim and Rush found that this works well, and that the student even needs less search when it generates.

> [!DEFINITION] Sequence-level distillation
> The teacher generates complete outputs for many inputs, and the student is fine-tuned on these outputs with the normal next-token loss. The teacher's text replaces the human-written answers. R1's distillation is of this kind, with sampling and a correctness filter in place of beam search.

### The two losses, side by side

Look at one position of a text. Let $$p_T(v)$$ be the teacher's probability that the next token is $$v$$, and $$p_S(v)$$ the student's. Word-level distillation uses the cross-entropy between the two lists:

$$
\mathcal{L}_{\text{word}} = - \sum_{v} p_T(v) \, \log p_S(v)
$$

Sequence-level distillation keeps only the token $$\hat{y}$$ that the teacher actually wrote at that position:

$$
\mathcal{L}_{\text{seq}} = - \log p_S(\hat{y})
$$

Read them from left to right. The sum runs over every token of the vocabulary. Each term is the teacher's probability for that token times the student's log-probability for it. The second loss has no sum: one token, one log-probability. Both are summed (or averaged) over all positions of the text.

A tiny example. Take a vocabulary of three tokens, A, B and C. The teacher gives (0.7, 0.2, 0.1) and the student gives (0.5, 0.3, 0.2). The word-level loss is

$$
-(0.7 \ln 0.5 + 0.2 \ln 0.3 + 0.1 \ln 0.2) = 0.485 + 0.241 + 0.161 = 0.887
$$

Suppose the teacher sampled A. The sequence-level loss is $$-\ln 0.5 = 0.693$$. If it had sampled C, the loss would be $$-\ln 0.2 = 1.609$$. Now sample many times from the teacher: A comes 70% of the time, B 20% and C 10%. The average sequence-level loss is then exactly 0.887, the word-level loss. Sampled text is a noisy but honest copy of the teacher's distribution.

R1's data is not plain samples either. For each prompt the authors sample several answers and keep only the correct ones ([Part 5](part-5-the-r1-pipeline.md) runs this rejection sampling). Version 1 does not give the sampling temperature for this step; the 0.6 of §3 is for evaluation. So the student copies the teacher's good answers, not its full distribution. That is a choice, and it fits the goal: we want the student to reason well, not to copy R1's mistakes.

{{FIG:p6_kd_kinds|One real position of our duck trace, with R1-Distill-Qwen-1.5B as the teacher, seen both ways. On the left, the teacher's top five next tokens and their probabilities, which a word-level student would match. On the right, the one token that the sampled text keeps, which is all that an R1 student sees.}}

Why did DeepSeek choose the text and not the probabilities? Three reasons follow from the setup, although the paper does not list them. None of the six students uses R1's tokenizer, so R1's probabilities over its own tokens do not line up with a student's tokens. Storing a full probability list for every token of 800k long answers would be huge, while text is small. And the 800k samples already existed: they were made for R1's own training.

## The data: 800k samples {§2.3.3}

> [!PAPER] · §2.3.3 · page 11
> [![The end of the Non-Reasoning data paragraph, with approximately 200k training samples and about 800k samples highlighted](/img/papers/deepseek-r1/p6-800k.png)](/img/papers/deepseek-r1/p6-800k.png)
>
> **What it says:** besides about 600k reasoning samples, the authors collect "approximately 200k training samples" for writing, factual questions, self-cognition and translation. R1's stage-3 SFT uses the whole set of "about 800k samples".
>
> **What it means:**
>
> [Part 5](part-5-the-r1-pipeline.md) explains how this set was built. In short: an RL checkpoint of R1 wrote many answers for each reasoning prompt, and only the correct and readable ones were kept. The other 200k are a mixture. Some are reused portions of DeepSeek-V3's own fine-tuning data. For some non-reasoning tasks, DeepSeek-V3 was prompted to write new answers, with a short chain of thought first.
>
> The important fact for this part is that the data was made once and used twice. It fine-tunes DeepSeek-V3-Base on the way to R1, and it fine-tunes the six small students. The same text teaches a 671B model and a 1.5B model.

## Distillation: fine-tune small models on R1's answers {§2.4}

{{FIG:p6_pipeline|You are here. Part 5 turns DeepSeek-V3-Base into DeepSeek-R1, and its stage 3 builds 800k samples. Section 2.4 fine-tunes six open models on those samples, with no RL stage.}}

> [!PAPER] · §2.4 · page 11
> [![The first lines of Section 2.4, with directly fine-tuned open-source models like Qwen, the 800k samples curated with DeepSeek-R1 and this straightforward distillation method significantly enhances the reasoning abilities of smaller highlighted](/img/papers/deepseek-r1/p6-s24-a.png)](/img/papers/deepseek-r1/p6-s24-a.png)
>
> **What it says:** the authors "directly fine-tuned open-source models like Qwen and Llama using the 800k samples curated with DeepSeek-R1". They find that "this straightforward distillation method significantly enhances the reasoning abilities of smaller models".
>
> **What it means:**
>
> "Directly" is the important word. There is no cold start, no reward and no RL here. The recipe is the most ordinary one in machine learning: take a pretrained model and fine-tune it on examples of the behaviour you want. The only unusual part is where the examples came from.
>
> Each example is a prompt and an answer. For a maths problem, the answer is R1's full reasoning between the thinking tags, then a short summary with the final result. The student learns to write that whole text, token by token.

> [!PAPER] · §2.4 · page 11
> [![The list of base models, with Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B and Llama-3.3-70B-Instruct highlighted, and the reason for Llama-3.3](/img/papers/deepseek-r1/p6-s24-b.png)](/img/papers/deepseek-r1/p6-s24-b.png)
>
> **What it says:** the students are "Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, and Llama-3.3-70B-Instruct". Llama-3.3 was chosen because "its reasoning capability is slightly better than that of Llama-3.1".
>
> **What it means:**
>
> Look at the list closely, because it is not uniform. The two smallest Qwen students are maths specialists: Qwen2.5-Math models were trained on extra maths text before any of this. The 14B and 32B Qwen students are general base models. The 70B Llama student is an instruct model, already fine-tuned to chat, while the 8B Llama is a base model.
>
> So the students start from different places. Keep this in mind when you read Table 5. The 1.5B student had a head start in maths. The 70B student had a head start in following instructions.

> [!PAPER] · §2.4 · page 11
> [![The second paragraph of Section 2.4, with we apply only SFT and do not include an RL stage, could substantially boost model performance and demonstrate the effectiveness of the distillation technique highlighted](/img/papers/deepseek-r1/p6-s24-c.png)](/img/papers/deepseek-r1/p6-s24-c.png)
>
> **What it says:** "we apply only SFT and do not include an RL stage", even though RL "could substantially boost model performance". The goal is to "demonstrate the effectiveness of the distillation technique".
>
> **What it means:**
>
> This is a choice made for science, not for the best scores. If the students also got RL, a reader could not tell which gains came from R1's text and which from the RL. With SFT only, every point in Table 5 comes from the copied answers.
>
> It also makes the recipe cheap to repeat. Anyone with the data and a few GPUs can run SFT. RL needs a reward, many samples per prompt and a careful setup, as Parts 2 to 4 showed.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, later long version · Appendix B.4.3 · page 35
> [![Appendix B.4.3 of the later version, with 2 to 3 epochs, cosine decay learning rate scheduler, one-tenth of its initial value, batch size is 64, Qwen2.5-Math-1.5B and 1 times 10 to the minus 4 highlighted](/img/papers/deepseek-r1/p6-hparams.png)](/img/papers/deepseek-r1/p6-hparams.png)
>
> **What it says:** each base model is fine-tuned "for 2–3 epochs" on the 800k data, with a "cosine decay learning rate scheduler" that ends at "one-tenth of its initial value". The maximum context is 32,768 tokens and the "batch size is 64". The 1.5B student starts at a learning rate of $$1 \times 10^{-4}$$; the 70B student starts at $$2 \times 10^{-5}$$.
>
> **What it means:**
>
> Version 1 of the paper gives none of these numbers; the later long version adds them. They are ordinary fine-tuning settings. The smaller the model, the larger its learning rate, which is the usual rule.
>
> A quick count shows the size of the job. With 800k samples and a batch of 64, one epoch is 800,000 / 64 = 12,500 steps. Two to three epochs make 25,000 to 37,500 steps. Each step is one forward and one backward pass, with no sampling.

### The loss of distillation, written out

Every step of this SFT uses one loss. The intuition is a teacher who reads a student's essay aloud and asks, at each word, "how surprised were you by this word?". The student is graded only on the answer part, never on the question, because the question is given.

$$
\mathcal{L}_{\text{SFT}} = - \frac{1}{N} \sum_{t \in \text{response}} \log p_\theta\left(y_t \mid y_{<t}\right)
$$

Read it from left to right. $$N$$ is the number of response tokens. The sum runs only over positions in the response, the teacher's answer. $$y_t$$ is the token that the teacher wrote at position $$t$$, and $$y_{<t}$$ is everything before it, the prompt included. $$p_\theta$$ is the student's probability, with its weights $$\theta$$. The minus sign turns "high probability" into "low loss".

This is exactly the sequence-level loss of the last section, averaged over a whole answer. A tiny example: a response of three tokens that the student gives probabilities 0.5, 0.25 and 0.8. The loss is $$-(\ln 0.5 + \ln 0.25 + \ln 0.8)/3 = (0.693 + 1.386 + 0.223)/3 = 0.767$$. The prompt tokens add nothing, however unlikely they are.

> [!DEFINITION] Perplexity
> $$e^{\mathcal{L}}$$, the exponential of the mean loss per token. It reads as "how many tokens the model is choosing between, on average". A perplexity of 2 means the model was as unsure as a fair coin at each token. Our three-token example has a perplexity of $$e^{0.767} = 2.15$$.

How does a program know which tokens are the response? Libraries mark every prompt token with the label $$-100$$, a number that no real token has, and skip those positions in the sum. The student must also predict token $$t+1$$ from position $$t$$, so the labels are shifted by one. These two details are where SFT code usually goes wrong, so we write them ourselves.

#### Method 1: the loss from scratch

We use a real R1-style example. The distilled 1.5B model solved the first problem of the GSM8K test set, about a farmer's ducks and eggs:

```text
question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
gold answer: 18
generated tokens: 872, finished with end token: True
trace starts: "Okay, so I have this problem about Janet and her ducks. Let me try to break it down step by step. I need to figure out how much money Janet makes every day at the farmers' market. Hmm, let's see.\n\nFirst, the problem says"
trace ends:   "ells for $2. Therefore, the total money she makes per day is:\n\n\\[\n9 \\times 2 = 18\n\\]\n\nThus, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market."
```

Then we tokenise it the way SFT does. The prompt comes from the chat template and ends with the thinking tag; the response is the teacher's text plus the end token.

```python
def build_example(tok, question, response):
    """Tokenise one distillation example the way SFT sees it.
    prompt: the chat template up to where the assistant starts (it ends with '<think>\\n' for the R1 template).
    response: the teacher's text plus the end-of-sequence token. Only the response is trained on."""
    msgs = [{'role': 'user', 'content': question + SUFFIX}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)   # list of P ids
    if not isinstance(prompt_ids, list):
        prompt_ids = prompt_ids['input_ids']
    response_ids = tok(response + tok.eos_token, add_special_tokens=False)['input_ids']  # list of N ids
    input_ids = prompt_ids + response_ids                                                 # P + N ids
    labels = [-100] * len(prompt_ids) + response_ids                                      # -100 = "no loss here"
    return input_ids, labels, len(prompt_ids)
```

{{FIG:p6_mask|The tokens of our real example around the boundary between prompt and response. Prompt tokens carry the label minus 100; response tokens carry their own token id.}}

```text
tokens: 957 in total, 85 prompt tokens (label -100), 872 response tokens
  position  82  token '<｜Assistant｜>'        label -100
  position  83  token '<think>'              label -100
  position  84  token '\n'                   label -100
  position  85  token 'Okay'                 label 32313
  position  86  token ','                    label 11
  position  87  token ' so'                  label 773
```

Now the loss itself, with the log-softmax written by hand. A model returns one row of scores (logits) for each position. The softmax turns a row into probabilities, and we need their log.

```python
def sft_loss_from_scratch(logits, labels):
    """Method 1. logits: (T, V) floats, one row per position. labels: (T,) ids, -100 where there is no loss.
    The model at position t predicts the token at t + 1, so we shift by one."""
    logits = logits[:-1]                                      # (T-1, V): predictions made at positions 0 .. T-2
    target = labels[1:]                                       # (T-1,):   the tokens at positions 1 .. T-1
    keep = target != -100                                     # (T-1,):   True only on response tokens
    m = logits.max(dim=-1, keepdim=True).values               # (T-1, 1): row maximum, for a stable log-sum-exp
    logsumexp = m + torch.log(torch.exp(logits - m).sum(dim=-1, keepdim=True))   # (T-1, 1)
    logprobs = logits - logsumexp                             # (T-1, V): log-softmax by hand
    picked = logprobs[keep].gather(1, target[keep].unsqueeze(1)).squeeze(1)      # (N,): log p(correct token)
    nll = -picked                                             # (N,): loss of each response token
    return nll.mean(), nll                                    # mean over response tokens only
```

#### Method 2: the same loss from the libraries

The student here is Qwen2.5-Math-1.5B, the base model that became R1-Distill-Qwen-1.5B. We give it the same example three more ways. transformers computes the loss itself when it gets `labels`. PyTorch's `cross_entropy` skips the $$-100$$ labels with `ignore_index`. And TRL's `SFTTrainer`, the trainer most people use for this job, tokenises the prompt and completion, builds its own labels and computes its own loss.

```python
loss1, nll1 = sft_loss_from_scratch(logits, lab[0])                  # Method 1, our function

# Method 2a: transformers computes the loss itself when we pass labels
loss2 = bmodel(ids, labels=lab).loss.float()
# Method 2b: PyTorch's cross_entropy with ignore_index=-100
loss2b = F.cross_entropy(logits[:-1], lab[0, 1:], ignore_index=-100)
# Method 2c: TRL's SFTTrainer builds the example (tokenise, mask the prompt) and computes the loss
ds = Dataset.from_list([{'prompt': prompt_text, 'completion': trace + dtok.eos_token}])
cfg = SFTConfig(output_dir=os.path.join(RESULTS, '.part6_trl_tmp'), report_to=[], completion_only_loss=True,
                max_length=None, per_device_eval_batch_size=1)
trainer = SFTTrainer(model=bmodel, args=cfg, train_dataset=ds, processing_class=dtok)
ex = trainer.train_dataset[0]
same_ids = list(ex['input_ids']) == input_ids
same_labels = list(ex['labels']) == labels
batch = trainer.data_collator([ex])
loss3 = trainer.compute_loss(bmodel, {k: v.to(DEV) for k, v in batch.items()}).float()
```

```text
logits shape: (957, 151936)
Method 1 (from scratch)        loss = 0.927018  over 872 tokens
Method 2a (model(labels=...))  loss = 0.927019
Method 2b (F.cross_entropy)    loss = 0.927019
Method 2c (TRL SFTTrainer.compute_loss) loss = 0.927019
TRL tokenised example: same input ids as ours: True, same labels (prompt masked): True
largest difference between methods: 1.25e-06  (agree to 1e-5: True)
first 12 response tokens and their loss under the student (nats):
  'Okay':12.07  ',':0.17  ' so':3.67  ' I':4.01  ' have':2.57  ' this':3.09  ' problem':0.67  ' about':3.20  ' Janet':0.93  ' and':1.18  ' her':0.08  ' ducks':0.14
```

All four numbers are 0.927 nats per token. Our function and the libraries differ by $$1.25 \times 10^{-6}$$, which is rounding in 32-bit floats, well inside $$10^{-5}$$. TRL also built exactly our token ids and exactly our labels, with the 85 prompt tokens masked. So the whole of SFT distillation fits in the two short functions above: build the labels, then average the log-probabilities of the response.

The per-token losses are worth a look. The base model gives the first word, "Okay", a loss of 12.07 nats: it almost never starts an answer like that. A few tokens later, " her" and " ducks" cost under 0.15 nats, because the question already said them. Distillation must teach the student the first kind of token, the voice and habits of R1. The second kind it already knows.

[![Terminal output of r1_part6.py: the trace, the mask, the four losses and the base against distilled comparison](/img/papers/deepseek-r1/part6_loss-run.png)](/img/papers/deepseek-r1/part6_loss-run.png)

## How the distilled models are tested {§3}

> [!PAPER] · §3 · page 12
> [![The sentence on distilled models in the Benchmarks paragraph, with report representative results on AIME 2024, MATH-500, GPQA Diamond, Codeforces, and LiveCodeBench highlighted](/img/papers/deepseek-r1/p6-bench.png)](/img/papers/deepseek-r1/p6-bench.png)
>
> **What it says:** "For distilled models, we report representative results on AIME 2024, MATH-500, GPQA Diamond, Codeforces, and LiveCodeBench."
>
> **What it means:**
>
> Five tests, in three groups. AIME 2024 and MATH-500 are maths: 30 hard competition problems and 500 school and competition problems. GPQA Diamond is science: graduate-level multiple-choice questions in biology, physics and chemistry. LiveCodeBench and Codeforces are code: write a program that passes hidden tests.
>
> Remember these groups. The headline about the smallest model is true for one group and false for another.

> [!PAPER] · §3 · page 12
> [![The Baselines paragraph, with GPT-4o-0513, OpenAI-o1-mini and For distilled models, we also compare the open-source model QwQ-32B-Preview highlighted](/img/papers/deepseek-r1/p6-baselines.png)](/img/papers/deepseek-r1/p6-baselines.png)
>
> **What it says:** the baselines include GPT-4o-0513, Anthropic's Sonnet 3.5 and OpenAI-o1-mini. "For distilled models, we also compare the open-source model QwQ-32B-Preview."
>
> **What it means:**
>
> The four comparison models have two kinds. GPT-4o and Sonnet 3.5 are strong general chat models. They answer without a long phase of reasoning first. o1-mini and QwQ-32B-Preview are reasoning models: they think at length before they answer. QwQ-32B-Preview was the best open reasoning model of late 2024, from the same team that makes Qwen.
>
> So Table 5 asks two questions. Can a small distilled model beat big models that do not reason? And can it match models that were built to reason?

> [!PAPER] · §3 · page 12
> [![The Evaluation Setup paragraph, with maximum generation length to 32,768 tokens, sampling temperature of 0.6, top-p value of 0.95, consensus (majority vote) results and cons@64 highlighted](/img/papers/deepseek-r1/p6-setup.png)](/img/papers/deepseek-r1/p6-setup.png)
>
> **What it says:** answers may be up to "32,768 tokens" long. Each question gets $$k$$ samples with a "sampling temperature of 0.6" and top-p 0.95, and pass@1 is the mean correctness of those samples. For AIME there is also a majority vote over 64 samples, "cons@64".
>
> **What it means:**
>
> [Part 4](part-4-r1-zero-and-the-aha-moment.md) met these two scores in Table 2, and [Part 7](part-7-results-failures-and-limits.md) reads the whole setup. In one line: pass@1 is "how often is one answer right", and cons@64 is "is the most common answer of 64 right".
>
> One number matters for our own run below: 32,768 tokens. A distilled model often thinks for thousands of tokens. On a laptop we must stop much earlier, and that costs accuracy.

## Table 5, row by row {§3.2}

> [!PAPER] · Table 5 · page 14
> [![Table 5 of the paper: GPT-4o-0513, Anthropic's Sonnet 3.5, OpenAI-o1-mini and QwQ-32B-Preview above a line, then the six DeepSeek-R1-Distill models, with AIME 2024 pass@1 and cons@64, MATH-500, GPQA Diamond, LiveCodeBench and CodeForces rating](/img/papers/deepseek-r1/p6-table5.png)](/img/papers/deepseek-r1/p6-table5.png)
>
> **What it says:** ten models, five benchmarks. Above the line are four models that did not learn from R1. Below the line are the six distilled students. The bold numbers are the best of each column.
>
> **What it means:**
>
> The table is dense, so the figure below draws its two maths columns. The sections after it read the table row by row.

{{FIG:p6_table5|Table 5 drawn as bars: AIME 2024 pass@1 and MATH-500 pass@1 for the four comparison models (top) and the six distilled students (bottom).}}

**R1-Distill-Qwen-1.5B.** AIME 28.9, MATH-500 83.9, GPQA 33.8, LiveCodeBench 16.9, Codeforces 954. On the two maths tests it beats GPT-4o (9.3 and 74.6) and Sonnet 3.5 (16.0 and 78.3). On GPQA and LiveCodeBench it is far below both. GPQA Diamond has four choices per question, so 33.8% is not far above a guess.

On the Codeforces rating, the 1.5B student is above both chat models: 954 against 759 and 717. But that column measures competition problems, which are close to maths puzzles.

**R1-Distill-Qwen-7B.** AIME 55.5, MATH-500 92.8. On maths it is already near o1-mini (63.6 and 90.0) and above QwQ-32B-Preview on AIME (50.0). It beats GPT-4o on every column but one: GPQA, 49.1 against 49.9. The paper's words "across the board" round that away.

**R1-Distill-Qwen-14B.** AIME 69.7, cons@64 80.0, MATH-500 93.9, GPQA 59.1, LiveCodeBench 53.1, Codeforces 1481. Every one of these numbers is above QwQ-32B-Preview's, a model more than twice its size that was built to reason.

**R1-Distill-Qwen-32B.** AIME 72.6, MATH-500 94.3, GPQA 62.1, LiveCodeBench 57.2. It beats o1-mini on five of six columns. The exception is the Codeforces rating, 1691 against 1820.

**R1-Distill-Llama-8B.** AIME 50.4, MATH-500 89.1. It is a little weaker on maths than the Qwen 7B student, which started from a maths model. On code it is a little stronger: LiveCodeBench 39.6 against 37.6.

**R1-Distill-Llama-70B.** AIME 70.0, cons@64 86.7, MATH-500 94.5, GPQA 65.2, LiveCodeBench 57.5. It has the best value in four columns. Yet it is more than twice the size of the Qwen 32B student, and only level with it. AIME is 70.0 against 72.6, and MATH-500 is 94.5 against 94.3.

Two patterns run through the rows. First, size helps, but less and less: AIME goes 28.9, 55.5, 69.7, 72.6 from 1.5B to 32B. Second, the start matters. The maths models are strong on maths for their size. The Llama students are not better per parameter than the Qwen students.

> [!PAPER] · §5 · page 16
> [![The distillation paragraph of the conclusion, with 800K training samples and on math benchmarks with 28.9% on AIME and 83.9% on MATH highlighted](/img/papers/deepseek-r1/p6-conclusion.png)](/img/papers/deepseek-r1/p6-conclusion.png)
>
> **What it says:** R1 is the teacher for "800K training samples". The 1.5B student outperforms GPT-4o and Anthropic's Sonnet 3.5 "on math benchmarks with 28.9% on AIME and 83.9% on MATH".
>
> **What it means:**
>
> Read the words "on math benchmarks" with care. This is the sentence most often repeated about the paper, usually without them. A 1.5B model is not better than GPT-4o in general. It is better at two maths tests, after it learned from 800k answers that are mostly maths and code reasoning.
>
> The next figure shows the same three models on the other two tests. The order turns upside down.

{{FIG:p6_small_vs_big|GPT-4o, Sonnet 3.5 and the distilled 1.5B model on four benchmarks of Table 5. The 1.5B model is first on both maths tests and last on science and code.}}

Why the split? A small model has little room for facts. GPQA asks for deep knowledge of biology, physics and chemistry, and reasoning cannot make up for facts the model does not know. Maths problems need fewer facts and more steps, and steps are what R1's text teaches.

> [!PAPER] · §3.2 · page 14
> [![The paragraph after Table 5, with outperform non-reasoning models like GPT-4o-0513 across the board, DeepSeek-R1-14B surpasses QwQ-32B-Preview on all evaluation metrics and exceed o1-mini on most benchmarks highlighted](/img/papers/deepseek-r1/p6-t5-text.png)](/img/papers/deepseek-r1/p6-t5-text.png)
>
> **What it says:** distillation lets the 7B student "outperform non-reasoning models like GPT-4o-0513 across the board". The 14B student "surpasses QwQ-32B-Preview on all evaluation metrics", and the 32B and 70B students "exceed o1-mini on most benchmarks".
>
> **What it means:**
>
> We checked each claim against the table above. The 14B claim holds on all six columns. The 32B and 70B claims hold on five of six; both lose on the Codeforces rating. The 7B claim has one small exception, GPQA, where it trails GPT-4o by 0.8 points.
>
> None of these exceptions changes the story. They are a reminder to read a table, not only the sentence about it.

> [!PAPER] · §3.2 · page 14
> [![The last sentences after Table 5, with applying RL to these distilled models yields significant further gains and only the results of the simple SFT-distilled models highlighted](/img/papers/deepseek-r1/p6-t5-rl.png)](/img/papers/deepseek-r1/p6-t5-rl.png)
>
> **What it says:** "applying RL to these distilled models yields significant further gains", but the paper presents "only the results of the simple SFT-distilled models".
>
> **What it means:**
>
> Distillation and RL are not rivals here. They stack: first copy a strong teacher, then let RL polish the student. The paper gives no numbers for this, so we cannot say how large the gain is.
>
> Many later open reasoning models followed exactly this order. They start from a distilled R1 model and add RL on top.

## Why the copy works: R1's text through two models

Section 1 says that the student copies R1's "reasoning patterns". We can measure that idea directly. Take text that R1 wrote, and ask two models how surprised they are by each token. One is the base model Qwen2.5-Math-1.5B, before distillation. The other is R1-Distill-Qwen-1.5B, after it. The loss is the same $$\mathcal{L}_{\text{SFT}}$$ as above, so a lower number means "this is the kind of text I write".

We use five texts. One is the trace from above. Three are real traces of the full DeepSeek-R1, from the public s1K-1.1 dataset. We took the three shortest correct ones among its first 300 rows, cut at 1,024 tokens. The last one is a control: an answer that the base model wrote itself to the duck problem.

```text
our trace (R1-Distill-1.5B)      732 tokens | base: loss 0.973, perplexity   2.64 | distilled: loss 0.361, perplexity   1.43
real R1 trace 1 (s1K-1.1)       1024 tokens | base: loss 0.480, perplexity   1.62 | distilled: loss 0.259, perplexity   1.30
real R1 trace 2 (s1K-1.1)       1024 tokens | base: loss 1.476, perplexity   4.38 | distilled: loss 0.998, perplexity   2.71
real R1 trace 3 (s1K-1.1)       1024 tokens | base: loss 0.701, perplexity   2.02 | distilled: loss 0.458, perplexity   1.58
control: base model answer       107 tokens | base: loss 0.250, perplexity   1.28 | distilled: loss 0.809, perplexity   2.25
same response token ids under both tokenizers: True (checked for every text)
```

{{FIG:p6_our_loss|The mean loss per token of five texts under the base model and the distilled model. Lower means the model finds the text more likely.}}

The pattern is clear. On every R1-style text, the distilled model has the lower loss. On our trace it falls from 0.973 to 0.361 nats per token, so the perplexity falls from 2.64 to 1.43. On the three real R1 traces it falls from 0.480 to 0.259, from 1.476 to 0.998, and from 0.701 to 0.458.

The control row turns the other way. On the base model's own answer, the base model has a loss of 0.250 and the distilled model 0.809. So the distilled model is not simply "better at text". It has moved toward one style of text, R1's, and away from the plain answers the base model used to write.

Two notes keep the test honest. Our trace was written by the distilled model itself, so this model must find it likely. That is why the three real R1 traces matter. Also, we used only the thinking part of our trace, 732 tokens. The end-of-thinking tag is one token for the R1 tokenizer and several for the base one. Every text gave identical token ids under both tokenizers, which the script checks.

{{FIG:p6_loss_curve|The loss of every token of a real R1 trace, smoothed, under the base model and the distilled model.}}

Which tokens moved most? The script averages the loss of each token over the three real R1 traces. It keeps tokens seen at least five times and sorts them by the drop:

```text
tokens of the real R1 traces with the largest gap (seen at least 5 times; mean loss base -> distilled):
  'Okay'         x  5    6.36 ->  0.90
  'First'        x  5    4.04 ->  0.34
  ' have'        x  6    3.82 ->  0.97
  ' right'       x  6    3.47 ->  1.13
  ' Let'         x 10    3.39 ->  1.09
  ' me'          x  8    2.74 ->  0.50
  ' susceptible' x  6    2.22 ->  0.45
  ' here'        x  5    3.70 ->  2.35
  ' so'          x 10    1.90 ->  0.59
  ' *'           x 11    1.38 ->  0.08
  ' \n\n'        x  5    2.30 ->  1.00
  ' that'        x 15    1.85 ->  0.61
  word 'Wait': x2, base 13.71, distilled 4.03 nats
  word ' wait': x1, base 4.02, distilled 0.79 nats
  word ' Hmm': x3, base 11.75, distilled 1.00 nats
  word 'Okay': x5, base 6.36, distilled 0.90 nats
  word ' Let': x10, base 3.39, distilled 1.09 nats
```

The top of the list is not maths. It is the voice of a thinker talking to itself: "Okay", "First", " Let", " me", " right". The words that mark R1's self-checks move the most of all. "Wait" costs the base model 13.71 nats, which is a probability of about one in a million, and the distilled model 4.03. " Hmm" goes from 11.75 to 1.00. These are the "reasoning patterns" of §1, visible as numbers: the student learned when to stop, doubt and look again.

## Our Table 5 on a laptop

The paper scores six students on five benchmarks with up to 32,768 tokens per answer. We can do a small part of that: the same base model and its released student on the same maths problems. We drew 16 problems of levels 1 to 3 from MATH-500, with a fixed seed. Each model wrote 4 answers per problem, with temperature 0.6 and top-p 0.95 as in §3. Each answer may use at most 2,048 new tokens, 16 times less than the paper allows.

We grade with one policy, the same as in [Part 4](part-4-r1-zero-and-the-aha-moment.md). An answer that hits the 2,048-token limit has no final answer, so it counts as wrong. For the distilled model, we grade only the text after `</think>`: the summary it writes once it has finished thinking. An answer that never closes `</think>` counts as wrong. The base model writes no thinking block, so we grade its whole finished text. math-verify then compares the last boxed answer with the reference.

The generation script saved every answer, and `r1_part6_regrade.py` grades the saved answers with this policy. No model runs again:

```text
$ python r1_part6_regrade.py
== 128 saved answers (16 problems x 4 samples x 2 models), max 2048 new tokens
== policy: cut off at the limit = wrong; distilled: grade only the text after </think>, no </think> = wrong
base      pass@1 0.328 (21/64) | problems solved at least once 10/16 | level 1: 0.25, level 2: 0.29, level 3: 0.45
          finished inside the limit 34/64 | closed </think> 0/64 | mean tokens 1263 (correct answers only: 538)
          old grading (whole text, cut-off answers included): 24/64 = 0.375
          kinder: finished answers graded at their first box: 21/64 = 0.328
distilled pass@1 0.484 (31/64) | problems solved at least once 11/16 | level 1: 0.70, level 2: 0.38, level 3: 0.40
          finished inside the limit 31/64 | closed </think> 36/64 | mean tokens 1693 (correct answers only: 1316)
          old grading (whole text, cut-off answers included): 38/64 = 0.594
distilled answers that finished: 31, of which correct: 31/31
```

[![Terminal output of r1_part6_regrade.py: the saved answers of both models regraded with one policy](/img/papers/deepseek-r1/part6_regrade-run.png)](/img/papers/deepseek-r1/part6_regrade-run.png)

{{FIG:p6_our_eval|Our run on 16 MATH-500 problems: accuracy under one grading policy (a cut-off answer is wrong, the distilled model is graded after its thinking), mean answer length, and answers that hit the 2,048-token limit, for the base model and the distilled model.}}

The distilled model is right on 31 of 64 answers, a pass@1 of 0.484. The base model is right on 21 of 64, or 0.328. The distilled model solves 11 of the 16 problems at least once; the base model solves 10. That is the direction of Table 5, on a much smaller test.

Look at the "old grading" lines. Our first grading read the whole text of every answer, cut off or not, and gave 38 and 24. That was too kind. A cut-off answer can hold a boxed number from the middle of its thinking, a step the model had not yet checked. The old grader counted it. The policy above does not.

The lengths tell the other half. The distilled model's correct answers average 1,316 tokens; the base model's correct answers average 538. The student has learned to think for a long time, as R1 does. With a cap of 2,048 tokens, 33 of its 64 answers were cut off before they ended.

Every one of the 31 answers that did finish was right. So the cap, not the model, decides most of this score. With the paper's 32,768 tokens, this model would likely score higher than 0.484.

The base model also hit the limit, 30 times, but for a different reason. We read its long answers. It often gives an answer and then keeps going. It writes a new problem of its own and solves that, or it repeats one phrase until the cap. A base model was never taught where an answer ends. Under our policy these answers are wrong, because they never end.

The "kinder" line grades each finished base answer at its first box instead of its last; it changes nothing (21 either way).

Sixteen problems with four samples each is a small test, so read these numbers as a direction, not as a benchmark score.

## Distillation or RL? {§4.1}

> [!PAPER] · §4.1 · page 14
> [![The start of Section 4.1, with can the model achieve comparable performance through the large-scale RL training without distillation, large-scale RL training on Qwen-32B-Base using math, code, and STEM data, training for over 10K steps, and DeepSeek-R1-Zero-Qwen-32B highlighted](/img/papers/deepseek-r1/p6-s41-q.png)](/img/papers/deepseek-r1/p6-s41-q.png)
>
> **What it says:** the question is whether a small model "can achieve comparable performance through the large-scale RL training discussed in the paper without distillation". To answer it, the authors run "large-scale RL training on Qwen-32B-Base using math, code, and STEM data, training for over 10K steps". The result is DeepSeek-R1-Zero-Qwen-32B.
>
> **What it means:**
>
> This is the control experiment that makes the distillation result mean something. Take the same base model as the 32B student, Qwen2.5-32B. Give it the R1-Zero treatment of [Part 4](part-4-r1-zero-and-the-aha-moment.md): RL with rule-based rewards, from the base model, with no copied answers. Then compare.
>
> "Over 10K steps" is a lot of RL. Each RL step samples a group of answers for every prompt in the batch, scores them and updates the model. The paper does not give the batch size or the group size for this run.

> [!PAPER] · Table 6 · page 15
> [![Table 6 of the paper: QwQ-32B-Preview, DeepSeek-R1-Zero-Qwen-32B and DeepSeek-R1-Distill-Qwen-32B on AIME 2024 pass@1 and cons@64, MATH-500, GPQA Diamond and LiveCodeBench](/img/papers/deepseek-r1/p6-table6.png)](/img/papers/deepseek-r1/p6-table6.png)
>
> **What it says:** QwQ-32B-Preview gets 50.0, 60.0, 90.6, 54.5 and 41.9. R1-Zero-Qwen-32B gets 47.0, 60.0, 91.6, 55.0 and 40.2. R1-Distill-Qwen-32B gets 72.6, 83.3, 94.3, 62.1 and 57.2.
>
> **What it means:**
>
> The RL model and QwQ-32B-Preview are almost twins: within 3 points on every column, and equal on cons@64. So RL on a 32B base works. It produces a real reasoning model, as good as the best open one of its time.
>
> The distilled model is in another class. It is 25.6 points higher on AIME pass@1, 23.3 higher on cons@64, 7.1 higher on GPQA and 17.0 higher on LiveCodeBench. Only MATH-500 is close, 94.3 against 91.6, because both are near the top of that test.

{{FIG:p6_table6|Table 6 as bars. On the same Qwen2.5-32B base, over 10K steps of RL give about the level of QwQ-32B-Preview; SFT on R1's 800k samples gives much more on every column.}}

> [!PAPER] · §4.1 · page 15
> [![The text after Table 6, with achieves performance on par with QwQ-32B-Preview, performs significantly better than and across all benchmarks highlighted](/img/papers/deepseek-r1/p6-t6-text.png)](/img/papers/deepseek-r1/p6-t6-text.png)
>
> **What it says:** the RL model "achieves performance on par with QwQ-32B-Preview", while the distilled model "performs significantly better than" the RL model "across all benchmarks".
>
> **What it means:**
>
> Why should copying beat learning by trial and error on the same model? Think about how RL learns. It can only make more likely the answers that the model already writes now and then. If a good line of thought is very rare for the model, RL almost never sees it, and so it almost never gets rewarded.
>
> R1's patterns came from RL on a 671B model, which could find them because it was much stronger at the start. The 32B model alone finds weaker patterns. Distillation skips the search: it hands the student the patterns that the bigger model found. Our loss experiment above showed the same gap from the other side. The base 1.5B model finds R1's text very unlikely, so it would rarely write such text on its own.

> [!PAPER] · §4.1 · page 15
> [![The two conclusions of Section 4.1, with distilling more powerful models into smaller ones yields excellent results, require enormous computational power and may not even achieve the performance of distillation, and advancing beyond the boundaries of intelligence may still require more powerful base models and larger-scale reinforcement learning highlighted](/img/papers/deepseek-r1/p6-conclusions.png)](/img/papers/deepseek-r1/p6-conclusions.png)
>
> **What it says:** first, "distilling more powerful models into smaller ones yields excellent results", while small models that rely on large-scale RL "require enormous computational power and may not even achieve the performance of distillation". Second, "advancing beyond the boundaries of intelligence may still require more powerful base models and larger-scale reinforcement learning."
>
> **What it means:**
>
> The first conclusion is practical. If a strong teacher exists, copy it: it is cheaper and it works better. SFT on 800k samples costs one pass of the data two or three times. RL costs sampling, scoring and updating, for over 10K steps, and still ends lower.
>
> The second conclusion is the limit of the first. In this paper, every student scores below its teacher, R1, and the students learn only from text that R1 wrote and that passed a filter. That is an observation about these experiments, not a law: in other work, students have sometimes beaten their teachers. Still, someone must build a strong teacher first, and the paper says that pushing further "may still require" a bigger base model and more RL. In this paper, distillation spreads what the frontier model can do, and RL on a large model is what moved the frontier.

## A tiny distillation of our own

Can we watch the recipe work at toy size? We tried §2.3.3 and §2.4 in miniature. The teacher was R1-Distill-Qwen-1.5B, standing in for R1. The student was Qwen2.5-0.5B-Instruct.

The teacher wrote one answer for each of 128 GSM8K training problems, with at most 1,024 new tokens. We kept only the answers that ended and were right. Then the student got SFT on the kept answers with our Method 1 loss. We used 3 epochs, batch 1, AdamW, and a learning rate of $$10^{-5}$$ with cosine decay to one tenth.

```text
== 1. teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B writes one trace per problem (128 GSM8K train problems, max 1024 tokens)
== 0. teacher distribution at response position 444 (after " $18 every day at the farmers' market.\n\n"):
   'Wait' 0.900, 'Let' 0.074, 'Hmm' 0.011, 'But' 0.005, 'Hold' 0.005; the trace has 'Wait' (p = 0.900)
traces written 128, finished and correct (kept) 35, mean length of kept 835 tokens, 391 s
== 3a. student Qwen/Qwen2.5-0.5B-Instruct before distillation (30 held-out GSM8K test problems x 2 samples)
before  accuracy 0.450 (27/60) | mean tokens 328 | with </think> 0/60 | with "Wait" 0/60 | hit the limit 0/60
== 2. SFT on the kept traces: 3 epochs, batch 1, AdamW, lr 1e-05 cosine to 1.0000000000000002e-06
  step    1/105  loss 0.8382  (mean of last 1)  3 s
  step   25/105  loss 0.7023  (mean of last 25)  52 s
  step   50/105  loss 0.4550  (mean of last 25)  162 s
  step   75/105  loss 0.3267  (mean of last 25)  476 s
  step  100/105  loss 0.1719  (mean of last 25)  712 s
```

The run did not finish. Our session was restarted at step 100 of 105, before the last evaluation, and the shared machine was then busy with a long training job. So we have no "after" numbers, and we do not guess them. What the log does show is still worth reading.

First, rejection sampling is expensive. Of 128 answers, only 35 both ended within 1,024 tokens and were right. Most of the rest were cut off while still thinking. The kept answers average 835 tokens, against 328 tokens for the student's own answers before training.

Second, the student starts far from the teacher's style. Before training it is right on 27 of 60 answers, and none of its 60 answers contains "Wait" or a thinking tag. Its training loss on the teacher's text then falls from 0.84 at the first step to 0.17 over steps 76 to 100. That fall is the same effect as in our comparison of the two 1.5B models: the student learns to find the teacher's text likely.

Third, the probe at the top of the log is the left half of the figure of the two losses. At one position of our duck trace, after the final answer, the teacher gives "Wait" a probability of 0.900, "Let" 0.074 and "Hmm" 0.011. The model checks its work even after it has the answer.

> [!TAKEAWAYS] Key takeaways
> - Distillation in R1 is plain SFT: six open models are fine-tuned on the 800k samples of R1's stage 3, with no RL stage, on purpose.
> - It is sequence-level distillation (Kim and Rush, 2016): the student learns R1's text, not R1's probabilities as in Hinton et al. (2015). On average over samples, the two losses are the same.
> - The loss is the next-token cross-entropy on the response tokens only; prompt tokens get the label -100. Our from-scratch loss matched transformers, PyTorch and TRL to $$1.25 \times 10^{-6}$$.
> - The students are Qwen2.5-Math-1.5B and 7B, Qwen2.5-14B and 32B, Llama-3.1-8B and Llama-3.3-70B-Instruct. The later version adds 2 to 3 epochs, batch 64 and a learning rate of 1e-4 for the 1.5B.
> - The 1.5B student beats GPT-4o and Sonnet 3.5 on maths only. AIME is 28.9 against 9.3 and 16.0; GPQA Diamond is 33.8 against 49.9 and 65.0.
> - The 14B student beats QwQ-32B-Preview on every column; the 32B and 70B students beat o1-mini on five of six.
> - On Qwen2.5-32B, over 10K steps of RL reach 47.0 on AIME, level with QwQ-32B-Preview; distillation reaches 72.6.
> - R1-Distill-Qwen-1.5B finds real R1 text far more likely than its base model does (loss 0.259 against 0.480 on one trace); "Wait" drops from 13.71 to 4.03 nats.
> - On 16 MATH-500 problems with a 2,048-token cap (cut-off = wrong), the distilled 1.5B scored 0.484 pass@1 and its base 0.328. All 31 distilled answers that finished were right, and its correct answers were more than twice as long.
> - Copying a stronger teacher is cheaper and better for small models; moving the frontier still needs bigger base models and more RL.

## The paper and the code

[Paper (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948) · [Official repository](https://github.com/deepseek-ai/DeepSeek-R1) · [Our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/deepseek-r1)
