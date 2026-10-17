---
title: "The Task: Promptable Segmentation"
description: "What SAM must learn: a valid mask for any prompt. We read §2 and Figure 3 of the paper, and we test ambiguous clicks, extra points, boxes and the paper's training game on the cat photo."
part: 2
covers: "§2, Figure 3"
date: 2026-10-17
tags: [sam, segmentation, prompts, ambiguity, computer-vision]
---

GPT-3 learned one task: predict the next word. That one task let it do many others, and the paper's §2 looks for a task with the same power for masks.

## The idea: a task like "predict the next word" {§2}

> [!PAPER] · §2 · page 4
> [![The start of Section 2 of the paper, Segment Anything Task: next token prediction serves for pre-training and for downstream tasks, and the team wants a task with analogous capabilities](/img/papers/sam/p2-intro.png)](/img/papers/sam/p2-intro.png)
>
> **What it says:** in NLP, one task, next token prediction, serves for pre-training *and* for new tasks via prompts. The team wants "a task with analogous capabilities".
>
> **What it means:**
>
> Look at the word *and* (the paper prints it in italics). The task of a language model does two jobs. First, it is the training job: read a text and guess the next word, billions of times. Second, it is the way to use the model after training. You write the start of a text, and the model continues it. A question, a translation and a summary are all "continue this text".
>
> So one task is both the school and the exam. The authors want the same two-in-one for pictures. The task must train a model on a very large set of masks. It must also be the handle that people and programs use later, for jobs that nobody planned.

> [!DEFINITION] Token
> One piece of text that a language model reads or writes: a word, a part of a word or a punctuation mark. "Next token prediction" means: from the tokens so far, guess the next one.

> [!DEFINITION] Pre-training
> The first, large training of a model, on a broad set of data, before any special job. A pre-trained model is the starting point. You then use it with prompts, or you train it a little more for one job. The task that the model learns in pre-training is its **pre-training objective**.

## What a prompt can be {§2}

> [!PAPER] · §2 · page 4
> [![The Task paragraph of Section 2: a prompt can be points, a box, a mask, text or any information indicating what to segment, and the task is to return a valid mask for any prompt](/img/papers/sam/p2-task-1.png)](/img/papers/sam/p2-task-1.png)
>
> **What it says:** a prompt is "any information indicating what to segment". The task is "to return a *valid* segmentation mask given any *prompt*".
>
> **What it means:**
>
> Imagine that you sit next to a friend with a photo between you. You want your friend to cut out one thing. You can put a finger on it, or say "not this part" with a finger somewhere else. You can draw a rough frame around it, or a quick outline. Or you can say "the cat". Each of these is a prompt, and each one gives a different kind of hint.
>
> The words "in general, any information" are important. The authors do not want a closed list of prompt types. A prompt is anything that tells the model *which* thing, as a text prompt can be any words. The task itself is one line: prompt in, valid mask out.

> [!DEFINITION] Foreground and background point
> A **foreground point** (a positive point, label 1) says "this pixel is part of the object". A **background point** (a negative point, label 0) says "this pixel is not part of the object". One prompt can hold many points of both kinds.

> [!DEFINITION] Bounding box
> The smallest rectangle around an object, with sides parallel to the edges of the picture. Four numbers give it: the left, top, right and bottom edges, $$(x_0, y_0, x_1, y_1)$$, in pixels. A detector gives its answers as bounding boxes.

{{FIG:p2_prompt_kinds|Five kinds of prompt for the same object. A point says "here". A negative point says "not here". A box says "inside this frame". A rough mask gives the approximate shape. Text gives a name.}}

The released SAM accepts points, boxes and masks. Text was a separate experiment in §7.5 of the paper, and the released weights do not include a text encoder. In the Hugging Face version of SAM, each kind of prompt has its own input. Our script [`sam_part2.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/sam/sam_part2.py) uses all of them:

```python
emb = sam.get_image_embeddings(proc(images=image, return_tensors='pt')['pixel_values'])    # once per picture

# points: (x, y) in pixels of the photo; label 1 = foreground, 0 = background
p = proc(images=image, input_points=[[[[180, 305], [100, 200]]]], input_labels=[[[1, 0]]], return_tensors='pt')
# a box: (x0, y0, x1, y1)
p = proc(images=image, input_boxes=[[[140, 285, 220, 345]]], return_tensors='pt')
# a mask: the 256 x 256 logits of an earlier answer
out = sam(image_embeddings=emb, input_points=..., input_labels=..., input_masks=prev_logits.reshape(1, 1, 256, 256),
          multimask_output=True)       # True: 3 masks and 3 scores; False: 1 mask and 1 score
```

## "Valid": one sensible meaning is enough {§2}

> [!PAPER] · §2 · page 4
> [![The rest of the Task paragraph: for an ambiguous prompt, the output should be a reasonable mask for at least one of the objects, like a coherent answer of a language model](/img/papers/sam/p2-task-2.png)](/img/papers/sam/p2-task-2.png)
>
> **What it says:** for an ambiguous prompt, the output "should be a reasonable mask for at least *one* of those objects", as a language model gives "a coherent response to an ambiguous prompt".
>
> **What it means:**
>
> Put your finger on a shirt in a photo and say "cut this out". Your friend can cut out the shirt or the whole person. Both answers are fair, because your finger was on both. A cut that takes half the shirt and a piece of the wall is not fair. It matches nothing that your finger could mean.
>
> The language model comparison makes this clear. Ask a chat model "Tell me about Java". It can talk about the island or about the programming language, and both answers are coherent. A text that mixes coffee, islands and code in every sentence is not.
>
> Between the lines, this rule also solves a practical problem. For an ambiguous prompt, there is no single correct mask to compare against, and a strict test would punish good answers. So the evaluation in §7 also reports an "oracle" score: of SAM's three masks, it takes the one that best matches the ground truth.

The paper gives the rule in words. We can write it as a formula. Think of a prompt as a finger on the photo, and list all the things that the finger can mean. A mask is valid if it matches at least one item of that list well.

$$
M \text{ is valid for } P \quad \Longleftrightarrow \quad \max_{O \in \mathcal{O}(P)} \text{IoU}(M, O) \ge \tau
$$

Read it from left to right:

- $$M$$ is the mask that the model gives, and $$P$$ is the prompt.
- $$\Longleftrightarrow$$ means "exactly when".
- $$\mathcal{O}(P)$$ is the set of objects that the prompt can mean. For a finger on a shirt, $$\mathcal{O}(P) = \{\text{shirt}, \text{person}\}$$.
- $$\max_{O \in \mathcal{O}(P)} \text{IoU}(M, O)$$ computes the IoU of the mask with each possible object ([Part 1](part-1-the-big-idea.md) explains IoU), and keeps the largest value.
- $$\tau$$ (the Greek letter tau) is a threshold: the lowest IoU that we accept as "a reasonable mask". The paper gives no number. We use $$\tau = 0.5$$ here.

A small example. A mask with IoU 0.20 with the shirt and 0.90 with the person is valid, because the maximum, 0.90, is more than 0.5. It is a good mask of the person. A mask of half the shirt and part of the wall has 0.30 and 0.10. Its maximum is 0.30, so it is not valid.

Now a real click. We clicked once on the head of the right cat, at pixel (400, 325). The click can mean the head or the whole cat. As the reference for the head, we took SAM's best mask for a box around the head. For the cat, we took the Part 1 mask from a box around the whole cat:

```text
reference "head": box [350, 262, 475, 362], the best of its 3 masks, 8,633 pixels (predicted IoU 0.904)
reference "cat":  Part 1 box mask of the right cat, 58,209 pixels
click at (400, 325):
  mask 1:   8,803 pixels, predicted IoU 0.740   IoU with head 0.739, with cat 0.151   -> best 0.739 (head)
  mask 2:  58,124 pixels, predicted IoU 1.006   IoU with head 0.149, with cat 0.996   -> best 0.996 (cat)
  mask 3: 144,150 pixels, predicted IoU 0.877   IoU with head 0.060, with cat 0.401   -> best 0.401 (cat)
  mask 3 covers 46.9% of the picture; 99.6% of the cat is inside it, and 86,200 of its pixels are not the cat
```

<figure><img src="/img/papers/sam/p2-valid.png" alt="Three copies of the photo with the click on the right cat's head. Mask 1 in yellow covers the head, score 0.740. Mask 2 in light blue covers the whole right cat, score 1.006. Mask 3 in dark blue covers the cat and large parts of the blanket and the sofa with many holes, score 0.877" loading="lazy" /><figcaption>One click on the right cat's head. Mask 1 is the head and mask 2 is the whole cat: both are valid. Mask 3 covers the cat and also 86,200 pixels of blanket and sofa, so it matches no sensible meaning of the click.</figcaption></figure>

Apply the formula with $$\tau = 0.5$$:

$$
\begin{aligned}
\text{mask 1:}\quad & \max(0.739,\ 0.151) = 0.739 \ge 0.5 && \text{valid (the head)} \\
\text{mask 2:}\quad & \max(0.149,\ 0.996) = 0.996 \ge 0.5 && \text{valid (the cat)} \\
\text{mask 3:}\quad & \max(0.060,\ 0.401) = 0.401 < 0.5 && \text{not valid}
\end{aligned}
$$

Mask 3 contains almost all of the cat, but IoU punishes its 86,200 extra pixels. So "valid" asks for one good mask among the answers, not for three. SAM ranks mask 2 first, so an application that keeps the top mask gets a valid answer here. Its score, 1.006, is a little above 1, because the score is only an *estimate* of the IoU.

## Figure 3: one point, three valid masks {§Figure 3}

> [!PAPER] · Figure 3 · page 4
> [![Figure 3 of the paper: four photos (an emu, a person with a backpack, an arm with a bottle, a sign with the word Zurich), each with one green point and three masks from whole to part to subpart](/img/papers/sam/p2-fig3.png)](/img/papers/sam/p2-fig3.png)
>
> **What it says:** "Each column shows 3 valid masks generated by SAM from a single ambiguous point prompt (green circle)."
>
> **What it means:**
>
> Read each column from top to bottom. The point stays in the same place, but the mask becomes smaller. In the second column, the point is on a backpack. The top mask is the whole person, the middle one is the backpack, and the bottom one is a small pocket of the backpack. In the fourth column, the point is on a sign: the whole board, then the word "ZURICH", then only the letter "Z".
>
> So the three masks are not three random guesses. They are the same object at three levels: the whole thing, a part of it and a part of that part. A person who clicked on the pocket can want any of the three. Note also what is missing: no mask of "the pocket and the ground next to it".

We did the same on our photo with three new clicks. They are on the dark tail of the left cat, a back paw of the right cat and the right remote control. This is the log of the tail click:

```text
click at (180, 305) on the dark tail of the left cat:
  mask 1 (tail              )   2,347 pixels =  0.76%   predicted IoU 0.980
  mask 2 (tail + back leg   )  12,024 pixels =  3.91%   predicted IoU 0.874
  mask 3 (whole cat         )  50,290 pixels = 16.37%   predicted IoU 0.962
  nested: 99.7% of mask 1 is inside mask 2, 99.4% of mask 2 is inside mask 3
  IoU between the masks: 1-2: 0.194, 1-3: 0.045, 2-3: 0.237
```

<figure><img src="/img/papers/sam/p2-fig3-ours.png" alt="A grid of nine crops in three columns and three rows, like Figure 3 of the paper. Column 1: the tail click gives the tail, the tail with the back leg, and the whole left cat. Column 2: the paw click gives the paw, the paw with the lower leg, and the whole right cat. Column 3: the remote click gives a few buttons, part of the remote, and the whole remote" loading="lazy" /><figcaption>Our version of Figure 3. Each column is one click (green dot), and each row is one of SAM's three masks. The masks grow from subpart to part to whole, and each one sits inside the next.</figcaption></figure>

The numbers show the structure of Figure 3. More than 99% of each mask lies inside the next one, and the size grows 5.1 times and then 4.2 times. The IoU between mask 1 and mask 3 is only 0.045, so these are really different answers.

{{FIG:p2_nested|The tail click as three nested regions. The area of each circle is in proportion to the number of pixels: 2,347 for the tail, 12,024 for the tail and back leg, 50,290 for the whole cat.}}

The ranking changes from click to click. On the tail, SAM puts the tail first (0.980). On the paw, it puts the whole cat first (0.998). So SAM has learned which level people usually mean for each kind of click.

### Why three masks, and not one?

§2 only asks for a valid mask. Why does SAM give three? One paragraph of §3 answers, and it explains why Figure 3 exists.

> [!PAPER] · §3 · page 5
> [![The Resolving ambiguity paragraph of Section 3: one output averages valid masks, so SAM predicts three masks (whole, part, subpart), trains only the one with the minimum loss and ranks them by estimated IoU](/img/papers/sam/p2-resolving.png)](/img/papers/sam/p2-resolving.png)
>
> **What it says:** "With one output, the model will average multiple valid masks". So SAM gives three masks ("whole, part, and subpart"), trains only the best one, and ranks them by an "estimated IoU".
>
> **What it means:**
>
> Imagine a student who sees the same exam question many times. Half of the time the answer key says "tail", and half of the time it says "cat". The safest single answer is a blur: half tail, half cat. That answer is never right. A model with one output is that student, because in training the same kind of click comes with different masks.
>
> Three outputs remove the problem. Each output can learn one level. Only the output closest to the ground truth gets the penalty, so the other outputs keep their own answers. Part 3 explains this loss.

The averaging shows clearly in the numbers of one pixel. Take a pixel on the cat's back: it is part of the cat, but not part of the tail. A fraction $$q$$ of the people want the whole cat, so for them the correct answer at this pixel is 1. For the other people it is 0. A single output gives the pixel a probability $$p$$. Its average penalty is the pixel loss of the [Background page](background-the-road-to-sam.md), weighted by how often each answer occurs:

$$
L(p) = q \cdot (-\ln p) \;+\; (1 - q) \cdot \big(-\ln (1 - p)\big)
$$

Read it from left to right:

- $$q \cdot (-\ln p)$$ is the penalty when the person wants the cat (correct answer 1), times how often that happens.
- $$(1 - q) \cdot (-\ln(1 - p))$$ is the penalty when the person wants only the tail (correct answer 0), times how often that happens.
- $$\ln$$ is the natural logarithm. $$-\ln$$ of a number near 1 is near 0, and $$-\ln$$ of a small number is large.

With $$q = 0.5$$, a confident "yes" ($$p = 0.99$$) costs $$0.5 \times 0.010 + 0.5 \times 4.605 = 2.308$$. A confident "no" ($$p = 0.01$$) costs the same 2.308. A shrug ($$p = 0.5$$) costs only 0.693. With $$q = 0.8$$, the best value is $$p = 0.8$$ with a loss of 0.500. In general, the smallest loss is always at $$p = q$$. The output learns the average answer.

{{FIG:p2_averaging|The loss of one pixel for a single output. When half of the people want the cat (q = 0.5), the best probability is 0.5, exactly on the threshold. When 80% want the cat, the best probability is 0.8. One output learns the average of the answers.}}

The released SAM has a fourth output for one mask (`multimask_output=False`). Training used it only for prompts with more than one point or box, so one click is new to it. We tried it anyway. We counted the pixels of its 256 × 256 output with a probability between 0.25 and 0.75: the pixels where the model hedges.

```text
tail    1 mask:   2,369 pixels, predicted IoU 0.999; IoU with masks 1/2/3: 0.991 / 0.195 / 0.045; low-res pixels with probability 0.25-0.75: 11 of 65,536
paw     1 mask:   1,851 pixels, predicted IoU 0.658; IoU with masks 1/2/3: 0.589 / 0.895 / 0.032; low-res pixels with probability 0.25-0.75: 2,746 of 65,536
paw     3 masks: low-res pixels with probability 0.25-0.75: 31 / 55 / 325
```

For the tail, the single output picks one level and is confident. For the paw, it hedges: 2,746 unsure pixels, against 325 or fewer in each normal mask, and a score of only 0.658. That is the blurred student. In §7.1, people gave a SAM trained with one output clearly lower ratings (Part 5).

## More prompts remove the ambiguity

Ambiguity is a property of the *prompt*, not of the model. Add information, and the ambiguity goes away. We gave the tail click a second point on the body of the left cat. We tried it once as a foreground point and once as a background point:

```text
tail (180, 305) + positive (foreground) point at (100, 200): 1 mask, 50,567 pixels, predicted IoU 0.983; IoU with tail mask 1 / 2 / 3: 0.045 / 0.236 / 0.991
tail (180, 305) + negative (background) point at (100, 200): 1 mask, 2,379 pixels, predicted IoU 0.970; IoU with tail mask 1 / 2 / 3: 0.987 / 0.195 / 0.045
```

<figure><img src="/img/papers/sam/p2-second-point.png" alt="Three copies of the left cat. Left: one green dot on the tail. Middle: a second green dot on the body gives a blue mask of the whole cat, 50,567 pixels, score 0.983. Right: a red dot on the body gives a yellow mask of only the tail, 2,379 pixels, score 0.970" loading="lazy" /><figcaption>The same tail click with a second point on the body. A foreground point (green) says "the body is part of it", and SAM gives the whole cat. A background point (red) says "the body is not part of it", and SAM gives only the tail.</figcaption></figure>

The second point picks the level: the whole cat (IoU 0.991 with mask 3 of the first click) or the tail (0.987 with mask 1).

A box also removes most of the ambiguity, because it tells the size of the object. We drew a box around the tail only:

```text
point (180, 305), 3 masks: 2,347 px (0.980), 12,024 px (0.874), 50,290 px (0.962)
box [140, 285, 220, 345], 3 masks: 2,315 px (0.989), 2,347 px (0.990), 2,402 px (0.986)
largest / smallest mask: point 21.4x, box 1.04x; IoU of box masks 1-2 0.986, 1-3 0.964, 2-3 0.977
```

<figure><img src="/img/papers/sam/p2-point-box.png" alt="Two rows of three crops of the left cat's tail. Top row, a point: the tail, the tail with the back leg, and the whole cat. Bottom row, a green box around the tail: three almost identical tail masks of 2,315, 2,347 and 2,402 pixels" loading="lazy" /><figcaption>Top: a point gives three levels, from 2,347 to 50,290 pixels. Bottom: a box around the tail gives three almost identical tail masks. The largest of them is only 1.04 times the smallest.</figcaption></figure>

With a point, the largest mask is 21.4 times the smallest. With a box, it is 1.04 times, and the three masks overlap with an IoU of 0.96 to 0.99. The box already answered the question "how large?". This is why SAM gives one mask for prompts with more than one element.

## Pre-training: a game of prompts {§2}

Now the first of the two jobs: how does this task train a model?

> [!PAPER] · §2 · page 4
> [![The start of the Pre-training paragraph: the task suggests a pre-training algorithm that simulates a sequence of prompts for each sample and compares the masks with the ground truth](/img/papers/sam/p2-pretrain-1.png)](/img/papers/sam/p2-pretrain-1.png)
>
> **What it says:** pre-training "simulates a sequence of prompts" for each mask and "compares the model's mask predictions against the ground truth".
>
> **What it means:**
>
> For a language model, the training data writes its own questions: hide the next word, and ask the model for it. Masks can do the same. A program pretends to be a person who wants one object of the dataset. It clicks on it, looks at the model's answer, and clicks again where the answer is wrong. Each answer is compared with the true mask, and the difference trains the model. So the model learns from a short conversation: a first, vague hint, then corrections.

[Part 1](part-1-the-big-idea.md) defined the **ground truth**: the correct mask of a training example. Appendix A of the paper gives the exact rules of the game.

> [!PAPER] · Appendix A · page 17
> [![The Training algorithm paragraph of Appendix A: the first prompt is a random foreground point or a noisy box, with box noise of 10% of the side length, at most 20 pixels](/img/papers/sam/p2-algo-1.png)](/img/papers/sam/p2-algo-1.png)
>
> **What it says:** the first prompt is a random point in the mask or a noisy box, with equal probability. The box noise has a "standard deviation equal to 10% of the box sidelength, to a maximum of 20 pixels".
>
> **What it means:**
>
> The first move of the game is a coin toss: a point or a box. A point lands on a random pixel of the object, so it can land on the tail as often as on the body. This is how ambiguity enters the training.
>
> A box is never perfect in real life. A person drags it by hand, and a detector makes small errors. So the training box shakes a little around the true box. A large object gets larger shifts, but the shift has a limit.

> [!DEFINITION] Standard deviation and normal noise
> **Normal noise** is a random number from the bell-shaped normal distribution, centred on 0. The **standard deviation** $$\sigma$$ (sigma) gives its usual size. About two thirds of the values fall between $$-\sigma$$ and $$+\sigma$$, and about 95% between $$-2\sigma$$ and $$+2\sigma$$.

The box noise in symbols, for the left edge $$x_0$$ of a box of width $$w$$:

$$
\tilde{x}_0 = x_0 + \varepsilon, \qquad \varepsilon \sim \mathcal{N}\big(0,\ (0.1\, w)^2\big), \qquad |\varepsilon| \le 20 \text{ pixels}
$$

Read it from left to right:

- $$\tilde{x}_0$$ (say "x-nought tilde") is the noisy left edge, and $$x_0$$ is the true left edge.
- $$\varepsilon$$ (epsilon) is the random shift. $$\sim \mathcal{N}(0, \sigma^2)$$ means "is drawn from a normal distribution with centre 0 and standard deviation $$\sigma$$". Here $$\sigma = 0.1\,w$$.
- $$|\varepsilon| \le 20$$ is the limit. The same rule applies to the other three edges, with the height for $$y_0$$ and $$y_1$$.

A small example: a box 100 pixels wide gets $$\sigma = 10$$. Its left edge usually moves by up to 10 pixels, and rarely by up to 20. Our right cat has a tight box of 291 × 343 pixels, so $$\sigma = 29.1$$ for the x edges and $$34.3$$ for the y edges. Here the 20-pixel limit cuts the larger shifts. (The paper's sentence can also mean a limit on $$\sigma$$ itself. Our script limits each shift.) We test these boxes in the zero-shot section below.

> [!PAPER] · Appendix A · page 17
> [![The rest of the Training algorithm: later points come from the error region, the previous mask logits go back in as a prompt, and there are 11 iterations in total](/img/papers/sam/p2-algo-2.png)](/img/papers/sam/p2-algo-2.png)
>
> **What it says:** each next point comes "uniformly from the error region". The last mask goes back in as "unthresholded mask logits", and with three masks the game keeps "the one with the highest predicted IoU". There are "11 total iterations".
>
> **What it means:**
>
> Think of a teacher who corrects a drawing. The teacher puts one finger on a mistake: "add this" where the student missed a piece, "remove this" where the student coloured too much. The student draws again, with the old drawing still on the desk. The old drawing is the mask prompt. It goes back in as raw scores, not 0s and 1s, so "very sure" and "only just yes" stay different.
>
> The 11 rounds are one first prompt, 8 corrections, and 2 rounds with no new point. In those 2 rounds, the model only sees its own last mask, so it learns to clean up its own work.

> [!DEFINITION] False negative and false positive
> A **false negative** is a pixel of the object that the mask missed. A **false positive** is a pixel outside the object that the mask included. The **error region** is all pixels of both kinds.

> [!DEFINITION] Logits
> The raw scores of a network before the last step changes them into probabilities. A logit of 0 means a probability of 0.5. Large positive logits mean "almost surely yes", and large negative logits mean "almost surely no". SAM's mask logits form a grid of 256 × 256 numbers.

How likely is the next point to be a foreground point? The point is a random pixel of the error region, so the answer is the share of missed pixels in that region:

$$
E = \underbrace{(Y \setminus M)}_{\text{missed}} \;\cup\; \underbrace{(M \setminus Y)}_{\text{extra}}, \qquad \Pr(\text{next point is foreground}) = \frac{|Y \setminus M|}{|E|}
$$

Read it from left to right:

- $$Y$$ is the ground truth mask and $$M$$ is the model's mask.
- $$Y \setminus M$$ (say "Y without M") is the set of pixels in the truth but not in the mask: the false negatives. $$M \setminus Y$$ is the false positives.
- $$\cup$$ joins the two sets into the error region $$E$$, and $$|\ldots|$$ counts pixels, as in Part 1.

A small example: the mask missed 30 pixels and added 10 extra pixels. Then $$|E| = 40$$, and the next point is a foreground point with probability $$30 / 40 = 0.75$$. Large errors attract more points, because they hold more pixels.

{{FIG:p2_sim_loop|The training game for one mask. The first prompt goes to SAM. The mask is compared with the ground truth. A point from the error region, plus the last mask logits, makes the next prompt. Only the light decoder runs again in each round.}}

### We played the game

We cannot train SAM again, but we can play the same game with the trained model and watch the IoU after each round. We have no hand-drawn mask for our photo, so the target was the whole left cat from Part 1 (49,520 pixels). The first point was the tail click, and all later points came from the error region.

```text
target: the whole left cat (Part 1, ear click, mask 3), 49,520 pixels; its tight box [14, 56, 314, 472]
  round  1: first prompt: point (180, 305)                                                   ->   2,347 px, IoU 0.027 (score 0.980)
  round  2: + point (308, 214); error region 48,170 missed + 997 extra, P(+) = 0.980         ->  50,826 px, IoU 0.972 (score 0.998)
  round  3: - point (196, 293); error region 65 missed + 1,371 extra, P(+) = 0.045           ->  50,341 px, IoU 0.977 (score 0.996)
  round  4: - point (209, 294); error region 160 missed + 981 extra, P(+) = 0.140            ->  50,216 px, IoU 0.979 (score 0.993)
  ...
  round  8: no new point, only the previous mask                                             ->  50,318 px, IoU 0.978 (score 0.997)
  ...
  round 11: no new point, only the previous mask                                             ->  50,052 px, IoU 0.981 (score 0.993)
```

<figure><img src="/img/papers/sam/p2-sim-rounds.png" alt="Four crops of the left cat with a yellow outline of the target. Round 1: one green dot on the tail, only the tail is blue, IoU 0.027. Round 2: a second green dot on a front paw, the whole cat is blue, IoU 0.972. Round 3: a red dot near the tail, IoU 0.977. Round 11: nine dots, green and red, IoU 0.981" loading="lazy" /><figcaption>Rounds 1, 2, 3 and 11 of the game. The yellow line is the target, the whole left cat. Green dots are foreground points and red dots are background points. One foreground point in round 2 changes the tail into the whole cat.</figcaption></figure>

Use the formula on round 2. After round 1, the mask is the tail: it missed 48,170 pixels of the cat and added 997 extra pixels. So $$|E| = 49{,}167$$, and the next point is a foreground point with probability $$48{,}170 / 49{,}167 = 0.980$$. It was: a point on the front paw. In round 3, almost all of the error is extra pixels. The probability falls to 0.045, and the game sent a background point.

After round 2, the IoU stays near 0.98. The error that remains is a thin band of about 1,100 pixels along the edge of the fur. A point in that band adds little. The paper saw the same effect: "diminishing returns after 8 iteratively sampled points".

We also played 20 games with random first prompts, as the paper does: 7 started with a point and 13 with a noisy box. We played each game twice, with and without the mask prompt.

```text
  round:                  1      2      3      4      5      6      7      8      9     10     11
  with mask prompt:   0.974  0.971  0.973  0.975  0.975  0.976  0.979  0.979  0.979  0.979  0.979
  without:            0.974  0.969  0.970  0.954  0.965  0.972  0.965  0.965  0.964  0.967  0.967
```

{{FIG:p2_sim_curve|The IoU with the target after each round. The run from the tail click jumps from 0.027 to 0.972 in round 2. The mean of 20 games ends at 0.979 with the mask prompt and at 0.967 without it.}}

A random first prompt on the whole cat is rarely ambiguous, so the mean starts high, at 0.974. The mask prompt keeps the answer steady. Without it, each new point on the fuzzy edge can push the mask the wrong way. The mean then ends at 0.967 instead of 0.979. Our target came from SAM itself, so this game cannot show how close SAM gets to a human mask. Part 5 does that.

## Promptable is not the same as interactive {§2}

> [!PAPER] · §2 · page 4
> [![The rest of the Pre-training paragraph: unlike interactive segmentation, the aim is to always predict a valid mask for any prompt, which automatic annotation needs](/img/papers/sam/p2-pretrain-2.png)](/img/papers/sam/p2-pretrain-2.png)
>
> **What it says:** interactive segmentation wants a valid mask "after enough user input". This task wants "to *always* predict a *valid mask* for *any prompt*", which "automatic annotation" needs.
>
> **What it means:**
>
> **Interactive segmentation** is the older field of click-and-correct tools, such as the "select object" tool in a photo editor. A person clicks, looks and clicks again until the mask is right. The first answer can be poor, because the person is there to fix it.
>
> Promptable segmentation removes that safety net: the first answer must already be valid. Why so strict? In the last stage of the data engine (Part 4), a program puts a grid of points on each picture and keeps SAM's answers. Nobody clicks again, so a bad first answer goes straight into the dataset.

Our game shows both views. As an interactive tool, SAM was poor in round 1 (IoU 0.027) and good from round 2 (0.972). As a promptable model, SAM was right in every round. The tail was a valid answer to a click on the tail, and SAM was sure of it (0.980).

## Zero-shot transfer: the prompt is the connection {§2}

Now the second job of the task: use the model for new jobs.

> [!PAPER] · §2 · page 4
> [![The Zero-shot transfer paragraph: downstream tasks are solved by engineering prompts, for example a cat detector box as the prompt for cat instance segmentation](/img/papers/sam/p2-zeroshot.png)](/img/papers/sam/p2-zeroshot.png)
>
> **What it says:** new tasks "can be solved by engineering appropriate prompts". For example, a cat detector's boxes, given to SAM as prompts, solve cat instance segmentation.
>
> **What it means:**
>
> Think of a standard power socket. The maker of a lamp does not need to know the power station, because both follow the socket. For SAM, the prompt is the socket. Any program that can make points or boxes can use SAM.
>
> The cat example shows the recipe. A detector knows how to *find* cats, but it only gives boxes. SAM knows how to *cut out* a thing, but not how to find it. Connect the two, and you get a mask for each cat, although neither model had training for "cat masks". The [Background page](background-the-road-to-sam.md) does this with YOLO and SAM on our photo. Part 5 shows the five tasks of §7.

> [!DEFINITION] Inference time
> The time when you use a trained model, after training has ended. At inference time, the weights do not change. "Test time" means the same.

A detector's box is never perfect. This is where the noisy boxes of the training game pay off. We took the tight box of the right cat and shook it 30 times with the training noise, and 30 times each with larger noise. We compared each answer with the mask of the tight box:

```text
tight box of the Part 1 cat mask: [348, 26, 639, 369] (width 291, height 343); its mask: 58,087 px, score 0.991, IoU with the Part 1 mask 0.998
  noise sd 10% of the side, at most 20 px (training)    mean corner shift  15.6 px: mean IoU 0.995, worst 0.970, 30/30 above 0.9
  noise sd 10% of the side, no limit                    mean corner shift  25.2 px: mean IoU 0.889, worst 0.000, 22/30 above 0.9
  noise sd 20% of the side, no limit                    mean corner shift  50.3 px: mean IoU 0.514, worst 0.000, 8/30 above 0.9
  noise sd 40% of the side, no limit                    mean corner shift 100.7 px: mean IoU 0.225, worst 0.000, 3/30 above 0.9
```

{{FIG:p2_box_noise|Noisy boxes around the right cat. With the noise of training, the masks hardly change (mean IoU 0.995). With larger noise, the boxes cut off parts of the cat or become very loose, and the mean IoU falls to 0.889, 0.514 and 0.225.}}

With the training noise, the corners moved 15.6 pixels on average, and all 30 masks stayed above IoU 0.9 (mean 0.995). SAM ignores a small error in the box, and a detector box needs exactly that. With larger noise, the boxes cut off the head or the legs, or become much too loose. Then the box no longer says "this cat".

The worst box without the limit, (377, 5, 692, 324), cuts off the cat's head. Its mask has no pixel of the cat (IoU 0.000), and SAM's score is low, 0.474. But a loose box with 20% noise gave a wrong 2,035-pixel mask with a score of 0.824. The training made SAM robust to the errors that it saw, not to all errors.

## Related tasks: many, but not all {§2}

> [!PAPER] · §2 · page 4
> [![The start of the Related tasks paragraph: a list of segmentation tasks, from interactive segmentation and edge detection to semantic, instance and panoptic segmentation](/img/papers/sam/p2-related-1.png)](/img/papers/sam/p2-related-1.png)
>
> **What it says:** "Segmentation is a broad field", and the paper names eight of its tasks.
>
> **What it means:**
>
> Each name in this list was a separate research field, with its own datasets, models and scores. The list is a quiet claim: one promptable model can stand in for many of these specialist models.

The table gives each task in one line, and the prompt that can turn it into a job for SAM. Part 1 defined semantic and instance segmentation.

| Task | What it gives | How SAM can do it |
|---|---|---|
| Interactive segmentation | a mask from a person's clicks | the clicks are the prompts |
| Edge detection | the lines where objects meet | a grid of points, then the edges of all the masks (§7.2) |
| Superpixelization | small groups of similar neighbour pixels | not a goal of the paper |
| Object proposal generation | about 1,000 masks of things that may be objects | a dense grid of points (§7.3) |
| Foreground segmentation | the main object against the background | a point or box on the main object |
| Semantic segmentation | a class name for each pixel | not alone: SAM gives no class names |
| Instance segmentation | a mask and a class for each object | detector boxes as prompts (§7.4) |
| Panoptic segmentation | semantic and instance segmentation together | not alone: it needs class names |

> [!PAPER] · §2 · page 4
> [![The rest of the Related tasks paragraph: the model adapts to many but not all tasks, and unlike a multi-task system it can do a new task as a component in a larger system](/img/papers/sam/p2-related-2.png)](/img/papers/sam/p2-related-2.png)
>
> **What it says:** the model adapts "to *many* (though not all)" tasks. Unlike a multi-task system, it can do "a new, different task at inference time" as "a *component* in a larger system".
>
> **What it means:**
>
> "Though not all" is an honest phrase. SAM does not know names. Semantic and panoptic segmentation need a name for each pixel, so SAM alone cannot do them. It needs a partner that knows names.
>
> A multi-task model is like a cook with three fixed recipes: it makes those three dishes very well, and nothing else. SAM is like a sharp knife. It makes no dish by itself, but every new recipe can use it.

> [!DEFINITION] Multi-task model
> One model that learns several fixed tasks at the same time, for example semantic, instance and panoptic segmentation. It is tested on the same tasks that it learned.

> [!DEFINITION] Task generalization
> The ability to do a task that was not part of the training. For SAM, the new task comes from a new way to make prompts, or from a new system around the model.

{{FIG:p2_multitask|Left: a multi-task model learns three fixed tasks and is tested on the same three. Right: SAM learns one task, and each new task comes from the part that makes its prompts: a detector, a grid of points or a person.}}

## Discussion: composition {§2}

> [!PAPER] · §2 · page 4
> [![The Discussion paragraph: prompting and composition allow tasks unknown at design time, as CLIP is a component of DALL-E, and a promptable model can be part of a larger algorithmic system](/img/papers/sam/p2-discussion.png)](/img/papers/sam/p2-discussion.png)
>
> **What it says:** prompting and composition allow "tasks unknown at the time of model design", as CLIP is a component of DALL·E. A promptable model can be "composed into a larger algorithmic system".
>
> **What it means:**
>
> Think of a building brick with a standard stud on top. The maker of the brick does not know what you will build, but every brick has the same connection. A foundation model with a prompt is such a brick. CLIP is the example from Part 1. The first DALL·E made many candidate pictures for a text, and CLIP ranked them by how well they matched it.
>
> The last sentence makes a fine point. An interactive tool is designed for a person who reads the screen. A promptable model can also take its prompts from a program, millions of times, with nobody watching. The data engine of Part 4 and the experiments of Part 5 are such programs.

> [!DEFINITION] Composition
> Building a system from separate models, where the output of one model becomes the input of the next. For SAM, a detector's boxes or a grid of points become SAM's prompts, and SAM's masks become the output of the system.

{{FIG:p2_composition|Two examples of composition. CLIP is one block in the DALL·E image generator: it ranks the candidate pictures. SAM is one block in a cat instance segmenter: a cat detector gives the boxes, and SAM gives one mask for each box.}}

> [!TAKEAWAYS] Key takeaways
> - **The task:** return a valid mask for any prompt: points, a box, a rough mask or text. Like next-word prediction, the one task is both the training job and the way to use the model.
> - **Valid** means a good mask for at least one sensible meaning of the prompt. On the head click, the head (IoU 0.739) and the cat (0.996) are valid. A mask with 86,200 pixels of blanket has a best IoU of 0.401, so it is not.
> - **Figure 3:** one point gives three nested masks: whole, part and subpart. Our tail click gave 2,347, 12,024 and 50,290 pixels, each more than 99% inside the next.
> - **Why three outputs:** one output learns the average answer. For one pixel, the loss is smallest at $$p = q$$, so a 50:50 pixel sits on the threshold.
> - **More prompts remove ambiguity:** a second point picks the level, and a box tells the size.
> - **Pre-training is a game** of 11 rounds: a first point or noisy box, 8 points from the error region, and 2 empty rounds. The last mask logits go back in. In our game, one point took the IoU from 0.027 to 0.972.
> - **Promptable, not only interactive:** every answer must be valid, because programs also send the prompts.
> - **Zero-shot transfer by composition:** a detector's boxes make SAM an instance segmenter. With the training box noise, our masks kept a mean IoU of 0.995. Tasks that need class names need a partner model.

**Next, in Part 3:** §3 of the paper, the model. Part 3 opens the image encoder, the prompt encoder and the mask decoder, and explains the training loss with the minimum over the three masks.

<details>
<summary>Run it yourself</summary>

The script [`code/papers/sam/sam_part2.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/sam/sam_part2.py) runs SAM ViT-H on the cat photo. It makes the three masks of each click, checks which masks are valid, and tries one output, a second point and a box. It plays the training game of Appendix A and tests noisy boxes. Every number goes to `results/part2.json`.

```bash
pip install torch transformers datasets pillow
python sam_part1.py        # Part 2 uses two masks that Part 1 saves
python sam_part2.py        # downloads facebook/sam-vit-huge (2.6 GB) if it is not cached
```

<figure><img src="/img/papers/sam/part2-run.png" alt="Terminal output of sam_part2.py: the three masks of the tail, paw and remote clicks, the validity check of the head click, the single-output test, the second point, the point and box on the tail, the 11 rounds of the training game and the noisy boxes" loading="lazy" /><figcaption>The first 60 lines of the output of sam_part2.py, as it ran for this article.</figcaption></figure>

</details>

## The paper and the code

[Segment Anything](https://arxiv.org/abs/2304.02643) (ICCV 2023) · [code and weights](https://github.com/facebookresearch/segment-anything) · [our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/sam)
