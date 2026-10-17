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

<figure class="fig"><svg viewBox="0 0 760 244" role="img" aria-label="Five kinds of prompt: a foreground point, a background point, a box, a rough mask, and text."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box-ghost" x="10.0" y="10.0" width="140.0" height="150.0" rx="10"/><g transform="translate(14,30) scale(0.95)" style="fill-opacity:0.35"><ellipse class="s4" cx="60" cy="70" rx="48" ry="26"/><circle class="s4" cx="112" cy="46" r="20"/><path class="s4" d="M98,32 L102,14 L112,28 Z"/><path class="s4" d="M114,28 L124,12 L128,32 Z"/><path class="s4" d="M14,66 C-4,60 -6,36 6,26 C10,38 14,52 22,58 Z"/></g><circle class="s3 ring" cx="72" cy="96" r="7"/><text class="t-note" x="80.0" y="182.0" text-anchor="middle">foreground point</text><text class="t-muted" x="80.0" y="200.0" text-anchor="middle">"this is part of it"</text><rect class="box-ghost" x="160.0" y="10.0" width="140.0" height="150.0" rx="10"/><g transform="translate(164,30) scale(0.95)" style="fill-opacity:0.35"><ellipse class="s4" cx="60" cy="70" rx="48" ry="26"/><circle class="s4" cx="112" cy="46" r="20"/><path class="s4" d="M98,32 L102,14 L112,28 Z"/><path class="s4" d="M114,28 L124,12 L128,32 Z"/><path class="s4" d="M14,66 C-4,60 -6,36 6,26 C10,38 14,52 22,58 Z"/></g><circle class="s3 ring" cx="222" cy="96" r="7"/><circle class="s2 ring" cx="280" cy="132" r="7"/><text class="t-tick" x="280.0" y="136.0" text-anchor="middle">−</text><text class="t-note" x="230.0" y="182.0" text-anchor="middle">background point</text><text class="t-muted" x="230.0" y="200.0" text-anchor="middle">"this is not part of it"</text><rect class="box-ghost" x="310.0" y="10.0" width="140.0" height="150.0" rx="10"/><g transform="translate(314,30) scale(0.95)" style="fill-opacity:0.35"><ellipse class="s4" cx="60" cy="70" rx="48" ry="26"/><circle class="s4" cx="112" cy="46" r="20"/><path class="s4" d="M98,32 L102,14 L112,28 Z"/><path class="s4" d="M114,28 L124,12 L128,32 Z"/><path class="s4" d="M14,66 C-4,60 -6,36 6,26 C10,38 14,52 22,58 Z"/></g><rect class="edge-on" x="316" y="36" width="128" height="78" rx="2" style="fill:none"/><text class="t-note" x="380.0" y="182.0" text-anchor="middle">box</text><text class="t-muted" x="380.0" y="200.0" text-anchor="middle">"it is inside here"</text><rect class="box-ghost" x="460.0" y="10.0" width="140.0" height="150.0" rx="10"/><g transform="translate(464,30) scale(0.95)" style="fill-opacity:0.35"><ellipse class="s4" cx="60" cy="70" rx="48" ry="26"/><circle class="s4" cx="112" cy="46" r="20"/><path class="s4" d="M98,32 L102,14 L112,28 Z"/><path class="s4" d="M114,28 L124,12 L128,32 Z"/><path class="s4" d="M14,66 C-4,60 -6,36 6,26 C10,38 14,52 22,58 Z"/></g><ellipse class="s1" cx="530" cy="89" rx="58" ry="32" style="fill-opacity:0.55"/><text class="t-note" x="530.0" y="182.0" text-anchor="middle">rough mask</text><text class="t-muted" x="530.0" y="200.0" text-anchor="middle">"about this shape"</text><rect class="box-ghost" x="610.0" y="10.0" width="140.0" height="150.0" rx="10"/><g transform="translate(614,30) scale(0.95)" style="fill-opacity:0.35"><ellipse class="s4" cx="60" cy="70" rx="48" ry="26"/><circle class="s4" cx="112" cy="46" r="20"/><path class="s4" d="M98,32 L102,14 L112,28 Z"/><path class="s4" d="M114,28 L124,12 L128,32 Z"/><path class="s4" d="M14,66 C-4,60 -6,36 6,26 C10,38 14,52 22,58 Z"/></g><text class="t-note" x="680.0" y="138.0" text-anchor="middle">"the cat"</text><text class="t-note" x="680.0" y="182.0" text-anchor="middle">text</text><text class="t-muted" x="680.0" y="200.0" text-anchor="middle">"the cat"</text><text class="t-muted" x="10.0" y="232.0" text-anchor="start">Each prompt gives the model a different kind of hint. Points can be mixed: two foreground points and one background point are one prompt.</text></svg><figcaption>Five kinds of prompt for the same object. A point says "here". A negative point says "not here". A box says "inside this frame". A rough mask gives the approximate shape. Text gives a name.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 290" role="img" aria-label="The three masks of the tail click are nested: the tail inside the tail and back leg inside the whole cat, with 2,347, 12,024 and 50,290 pixels."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">One click on the tail: three masks, each inside the next</text><g class="mark"><title>mask 3: 50,290 pixels</title><circle class="s3" cx="190.0" cy="160.0" r="110.0" style="fill-opacity:0.55"/></g><g class="mark"><title>mask 2: 12,024 pixels</title><circle class="s1" cx="220.9" cy="185.3" r="53.8" style="fill-opacity:0.55"/></g><g class="mark"><title>mask 1: 2,347 pixels</title><circle class="s2" cx="237.4" cy="198.8" r="23.8" style="fill-opacity:0.55"/></g><circle class="s4 ring" cx="237.4" cy="198.8" r="4"/><rect class="s2" x="380" y="59" width="12" height="12" rx="2"/><text class="t-note" x="400.0" y="70.0" text-anchor="start">subpart: mask 1: tail</text><text class="t-tick" x="400.0" y="90.0" text-anchor="start">2,347 pixels, predicted IoU 0.980</text><rect class="s1" x="380" y="117" width="12" height="12" rx="2"/><text class="t-note" x="400.0" y="128.0" text-anchor="start">part: mask 2: tail + back leg</text><text class="t-tick" x="400.0" y="148.0" text-anchor="start">12,024 pixels, predicted IoU 0.874</text><rect class="s3" x="380" y="175" width="12" height="12" rx="2"/><text class="t-note" x="400.0" y="186.0" text-anchor="start">whole: mask 3: whole cat</text><text class="t-tick" x="400.0" y="206.0" text-anchor="start">50,290 pixels, predicted IoU 0.962</text><text class="t-muted" x="380.0" y="252.0" text-anchor="start">99.7% of mask 1 is inside mask 2; 99.4% of mask 2 is inside mask 3</text><text class="t-muted" x="380.0" y="272.0" text-anchor="start">circle area is in proportion to the number of pixels</text></svg><figcaption>The tail click as three nested regions. The area of each circle is in proportion to the number of pixels: 2,347 for the tail, 12,024 for the tail and back leg, 50,290 for the whole cat.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 316" role="img" aria-label="The loss of one pixel for a single output, when a fraction q of people want the cat and the rest want only the tail. The smallest loss is at p = q: 0.5 for q = 0.5 and 0.8 for q = 0.8."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Loss of one pixel when people disagree: L(p) = q·(−ln p) + (1 − q)·(−ln(1 − p))</text><line class="grid" x1="70.0" y1="262.0" x2="490.0" y2="262.0"/><text class="t-tick" x="62.0" y="266.0" text-anchor="end">0</text><line class="grid" x1="70.0" y1="194.0" x2="490.0" y2="194.0"/><text class="t-tick" x="62.0" y="198.0" text-anchor="end">1</text><line class="grid" x1="70.0" y1="126.0" x2="490.0" y2="126.0"/><text class="t-tick" x="62.0" y="130.0" text-anchor="end">2</text><line class="grid" x1="70.0" y1="58.0" x2="490.0" y2="58.0"/><text class="t-tick" x="62.0" y="62.0" text-anchor="end">3</text><text class="t-tick" x="70.0" y="280.0" text-anchor="middle">0</text><text class="t-tick" x="175.0" y="280.0" text-anchor="middle">0.25</text><text class="t-tick" x="280.0" y="280.0" text-anchor="middle">0.5</text><text class="t-tick" x="385.0" y="280.0" text-anchor="middle">0.75</text><text class="t-tick" x="490.0" y="280.0" text-anchor="middle">1</text><line class="axis" x1="70" y1="262" x2="490" y2="262"/><polyline class="l1" points="74.2,105.1 76.3,118.7 78.4,128.3 80.5,135.7 82.6,141.7 84.7,146.8 86.8,151.2 88.9,155.0 91.0,158.4 93.1,161.5 95.2,164.2 97.3,166.8 99.4,169.1 101.5,171.3 103.6,173.3 105.7,175.2 107.8,176.9 109.9,178.6 112.0,180.1 114.1,181.6 116.2,183.0 118.3,184.3 120.4,185.6 122.5,186.8 124.6,187.9 126.7,189.0 128.8,190.0 130.9,191.0 133.0,192.0 135.1,192.9 137.2,193.8 139.3,194.6 141.4,195.4 143.5,196.2 145.6,196.9 147.7,197.7 149.8,198.4 151.9,199.0 154.0,199.7 156.1,200.3 158.2,200.9 160.3,201.5 162.4,202.1 164.5,202.6 166.6,203.1 168.7,203.7 170.8,204.1 172.9,204.6 175.0,205.1 177.1,205.5 179.2,206.0 181.3,206.4 183.4,206.8 185.5,207.2 187.6,207.6 189.7,207.9 191.8,208.3 193.9,208.6 196.0,208.9 198.1,209.3 200.2,209.6 202.3,209.9 204.4,210.1 206.5,210.4 208.6,210.7 210.7,210.9 212.8,211.2 214.9,211.4 217.0,211.7 219.1,211.9 221.2,212.1 223.3,212.3 225.4,212.5 227.5,212.7 229.6,212.8 231.7,213.0 233.8,213.2 235.9,213.3 238.0,213.5 240.1,213.6 242.2,213.7 244.3,213.9 246.4,214.0 248.5,214.1 250.6,214.2 252.7,214.3 254.8,214.4 256.9,214.5 259.0,214.5 261.1,214.6 263.2,214.6 265.3,214.7 267.4,214.7 269.5,214.8 271.6,214.8 273.7,214.8 275.8,214.9 277.9,214.9 280.0,214.9 282.1,214.9 284.2,214.9 286.3,214.8 288.4,214.8 290.5,214.8 292.6,214.7 294.7,214.7 296.8,214.6 298.9,214.6 301.0,214.5 303.1,214.5 305.2,214.4 307.3,214.3 309.4,214.2 311.5,214.1 313.6,214.0 315.7,213.9 317.8,213.7 319.9,213.6 322.0,213.5 324.1,213.3 326.2,213.2 328.3,213.0 330.4,212.8 332.5,212.7 334.6,212.5 336.7,212.3 338.8,212.1 340.9,211.9 343.0,211.7 345.1,211.4 347.2,211.2 349.3,210.9 351.4,210.7 353.5,210.4 355.6,210.1 357.7,209.9 359.8,209.6 361.9,209.3 364.0,208.9 366.1,208.6 368.2,208.3 370.3,207.9 372.4,207.6 374.5,207.2 376.6,206.8 378.7,206.4 380.8,206.0 382.9,205.5 385.0,205.1 387.1,204.6 389.2,204.1 391.3,203.7 393.4,203.1 395.5,202.6 397.6,202.1 399.7,201.5 401.8,200.9 403.9,200.3 406.0,199.7 408.1,199.0 410.2,198.4 412.3,197.7 414.4,196.9 416.5,196.2 418.6,195.4 420.7,194.6 422.8,193.8 424.9,192.9 427.0,192.0 429.1,191.0 431.2,190.0 433.3,189.0 435.4,187.9 437.5,186.8 439.6,185.6 441.7,184.3 443.8,183.0 445.9,181.6 448.0,180.1 450.1,178.6 452.2,176.9 454.3,175.2 456.4,173.3 458.5,171.3 460.6,169.1 462.7,166.8 464.8,164.2 466.9,161.5 469.0,158.4 471.1,155.0 473.2,151.2 475.3,146.8 477.4,141.7 479.5,135.7 481.6,128.3 483.7,118.7 485.8,105.1" style="fill:none"/><g class="mark"><title>q = 0.5: smallest loss 0.693 at p = 0.5</title><circle class="s1 ring" cx="280.0" cy="214.9" r="6"/></g><polyline class="l3" points="74.2,58.0 76.3,58.0 78.4,58.0 80.5,61.0 82.6,70.8 84.7,79.1 86.8,86.3 88.9,92.7 91.0,98.3 93.1,103.4 95.2,108.1 97.3,112.4 99.4,116.3 101.5,120.0 103.6,123.5 105.7,126.7 107.8,129.7 109.9,132.6 112.0,135.3 114.1,137.9 116.2,140.3 118.3,142.7 120.4,144.9 122.5,147.1 124.6,149.1 126.7,151.1 128.8,153.0 130.9,154.8 133.0,156.6 135.1,158.3 137.2,159.9 139.3,161.5 141.4,163.1 143.5,164.6 145.6,166.0 147.7,167.4 149.8,168.8 151.9,170.1 154.0,171.4 156.1,172.7 158.2,173.9 160.3,175.1 162.4,176.3 164.5,177.4 166.6,178.5 168.7,179.6 170.8,180.6 172.9,181.7 175.0,182.7 177.1,183.7 179.2,184.6 181.3,185.6 183.4,186.5 185.5,187.4 187.6,188.3 189.7,189.2 191.8,190.0 193.9,190.8 196.0,191.7 198.1,192.5 200.2,193.2 202.3,194.0 204.4,194.8 206.5,195.5 208.6,196.2 210.7,197.0 212.8,197.7 214.9,198.4 217.0,199.0 219.1,199.7 221.2,200.4 223.3,201.0 225.4,201.6 227.5,202.3 229.6,202.9 231.7,203.5 233.8,204.1 235.9,204.6 238.0,205.2 240.1,205.8 242.2,206.3 244.3,206.9 246.4,207.4 248.5,207.9 250.6,208.4 252.7,209.0 254.8,209.5 256.9,209.9 259.0,210.4 261.1,210.9 263.2,211.4 265.3,211.8 267.4,212.3 269.5,212.7 271.6,213.2 273.7,213.6 275.8,214.0 277.9,214.5 280.0,214.9 282.1,215.3 284.2,215.7 286.3,216.1 288.4,216.4 290.5,216.8 292.6,217.2 294.7,217.6 296.8,217.9 298.9,218.3 301.0,218.6 303.1,219.0 305.2,219.3 307.3,219.6 309.4,219.9 311.5,220.3 313.6,220.6 315.7,220.9 317.8,221.2 319.9,221.5 322.0,221.7 324.1,222.0 326.2,222.3 328.3,222.6 330.4,222.8 332.5,223.1 334.6,223.3 336.7,223.6 338.8,223.8 340.9,224.1 343.0,224.3 345.1,224.5 347.2,224.7 349.3,224.9 351.4,225.1 353.5,225.3 355.6,225.5 357.7,225.7 359.8,225.9 361.9,226.1 364.0,226.2 366.1,226.4 368.2,226.5 370.3,226.7 372.4,226.8 374.5,226.9 376.6,227.1 378.7,227.2 380.8,227.3 382.9,227.4 385.0,227.5 387.1,227.6 389.2,227.7 391.3,227.7 393.4,227.8 395.5,227.8 397.6,227.9 399.7,227.9 401.8,228.0 403.9,228.0 406.0,228.0 408.1,228.0 410.2,228.0 412.3,227.9 414.4,227.9 416.5,227.8 418.6,227.8 420.7,227.7 422.8,227.6 424.9,227.5 427.0,227.4 429.1,227.2 431.2,227.1 433.3,226.9 435.4,226.7 437.5,226.5 439.6,226.2 441.7,225.9 443.8,225.6 445.9,225.3 448.0,225.0 450.1,224.6 452.2,224.1 454.3,223.6 456.4,223.1 458.5,222.5 460.6,221.9 462.7,221.2 464.8,220.4 466.9,219.5 469.0,218.5 471.1,217.3 473.2,216.0 475.3,214.5 477.4,212.7 479.5,210.5 481.6,207.7 483.7,204.1 485.8,198.8" style="fill:none"/><g class="mark"><title>q = 0.8: smallest loss 0.500 at p = 0.8</title><circle class="s3 ring" cx="406.0" cy="228.0" r="6"/></g><text class="t-tick" x="280.0" y="304.0" text-anchor="middle">p: the probability that the single output gives to "this pixel is part of the mask"</text><text class="t-tick" x="20.0" y="48.0" text-anchor="start">loss</text><text class="t-note" x="520.0" y="70.0" text-anchor="start">A pixel on the cat, not on the tail.</text><text class="t-muted" x="520.0" y="90.0" text-anchor="start">q = fraction of people who want the cat.</text><rect class="s1" x="520" y="113" width="10" height="10" rx="2"/><text class="t-note" x="536.0" y="122.0" text-anchor="start">q = 0.5: best p = 0.50</text><rect class="s3" x="520" y="139" width="10" height="10" rx="2"/><text class="t-note" x="536.0" y="148.0" text-anchor="start">q = 0.8: best p = 0.80</text><text class="t-muted" x="520.0" y="186.0" text-anchor="start">The best single answer is the</text><text class="t-muted" x="520.0" y="204.0" text-anchor="start">average of the answers, not one</text><text class="t-muted" x="520.0" y="222.0" text-anchor="start">of them. At p = 0.5 the pixel sits</text><text class="t-muted" x="520.0" y="240.0" text-anchor="start">on the threshold: a blurred mask.</text></svg><figcaption>The loss of one pixel for a single output. When half of the people want the cat (q = 0.5), the best probability is 0.5, exactly on the threshold. When 80% want the cat, the best probability is 0.8. One output learns the average of the answers.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 296" role="img" aria-label="The training loop of the paper: a first prompt goes to SAM, the mask is compared with the ground truth, a new point is taken from the error region, and the points and the previous mask logits go back in as the next prompt."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box-1" x="20.0" y="40.0" width="150.0" height="64.0" rx="10"/><text class="t-note" x="95.0" y="66.0" text-anchor="middle">first prompt</text><text class="t-tick" x="95.0" y="86.0" text-anchor="middle">point or noisy box</text><line class="edge" x1="174.0" y1="72.0" x2="216.0" y2="72.0" marker-end="url(#ah)"/><rect class="box-on" x="220.0" y="40.0" width="140.0" height="64.0" rx="10"/><text class="t-title" x="290.0" y="66.0" text-anchor="middle">SAM</text><text class="t-tick" x="290.0" y="86.0" text-anchor="middle">decoder only</text><line class="edge" x1="364.0" y1="72.0" x2="406.0" y2="72.0" marker-end="url(#ah)"/><rect class="box-2" x="410.0" y="40.0" width="150.0" height="64.0" rx="10"/><text class="t-note" x="485.0" y="66.0" text-anchor="middle">predicted mask</text><text class="t-tick" x="485.0" y="86.0" text-anchor="middle">best score if 3</text><line class="edge" x1="564.0" y1="72.0" x2="600.0" y2="72.0" marker-end="url(#ah)"/><rect class="box-3" x="604.0" y="40.0" width="140.0" height="64.0" rx="10"/><text class="t-note" x="674.0" y="66.0" text-anchor="middle">compare with</text><text class="t-note" x="674.0" y="86.0" text-anchor="middle">ground truth</text><line class="edge" x1="674.0" y1="108.0" x2="674.0" y2="150.0" marker-end="url(#ah)"/><rect class="box-ghost" x="560.0" y="154.0" width="184.0" height="78.0" rx="10"/><text class="t-note" x="652.0" y="176.0" text-anchor="middle">error region</text><text class="t-tick" x="652.0" y="196.0" text-anchor="middle">missed pixels → + point</text><text class="t-tick" x="652.0" y="214.0" text-anchor="middle">extra pixels → − point</text><line class="edge" x1="556.0" y1="193.0" x2="440.0" y2="193.0" marker-end="url(#ah)"/><rect class="box-1" x="220.0" y="160.0" width="216.0" height="66.0" rx="10"/><text class="t-note" x="328.0" y="184.0" text-anchor="middle">next prompt</text><text class="t-tick" x="328.0" y="204.0" text-anchor="middle">all points so far</text><text class="t-tick" x="328.0" y="220.0" text-anchor="middle">+ previous mask logits</text><path class="edge" d="M290,156 L290,108" marker-end="url(#ah)"/><text class="t-muted" x="20.0" y="264.0" text-anchor="start">11 rounds for each mask: the first prompt, 8 new points, and 2 rounds with no new point (only the previous mask).</text><text class="t-muted" x="20.0" y="284.0" text-anchor="start">The loss is computed after each round. The image encoder runs once; only the light decoder runs 11 times.</text></svg><figcaption>The training game for one mask. The first prompt goes to SAM. The mask is compared with the ground truth. A point from the error region, plus the last mask logits, makes the next prompt. Only the light decoder runs again in each round.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="IoU with the target after each of the 11 rounds. The run that starts on the tail goes from 0.027 to 0.972 after one more point. The mean of 20 runs goes from 0.974 to 0.979 with the mask prompt and to 0.967 without it."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="510" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0.00</text><line class="grid" x1="64" y1="195.0" x2="510" y2="195.0"/><text class="t-tick" x="56.0" y="199.0" text-anchor="end">0.25</text><line class="grid" x1="64" y1="140.0" x2="510" y2="140.0"/><text class="t-tick" x="56.0" y="144.0" text-anchor="end">0.50</text><line class="grid" x1="64" y1="85.0" x2="510" y2="85.0"/><text class="t-tick" x="56.0" y="89.0" text-anchor="end">0.75</text><line class="grid" x1="64" y1="30.0" x2="510" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">1.00</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">1</text><text class="t-tick" x="108.6" y="270.0" text-anchor="middle">2</text><text class="t-tick" x="153.2" y="270.0" text-anchor="middle">3</text><text class="t-tick" x="197.8" y="270.0" text-anchor="middle">4</text><text class="t-tick" x="242.4" y="270.0" text-anchor="middle">5</text><text class="t-tick" x="287.0" y="270.0" text-anchor="middle">6</text><text class="t-tick" x="331.6" y="270.0" text-anchor="middle">7</text><text class="t-tick" x="376.2" y="270.0" text-anchor="middle">8</text><text class="t-tick" x="420.8" y="270.0" text-anchor="middle">9</text><text class="t-tick" x="465.4" y="270.0" text-anchor="middle">10</text><text class="t-tick" x="510.0" y="270.0" text-anchor="middle">11</text><line class="axis" x1="64" y1="250" x2="510" y2="250"/><polyline class="l2" points="64.0,244.1 108.6,36.2 153.2,35.0 197.8,34.5 242.4,34.8 287.0,34.9 331.6,34.9 376.2,34.9 420.8,34.9 465.4,34.3 510.0,34.2"/><g class="mark"><title>run from the tail click, 1: 0.03</title><circle class="s2 ring" cx="64.0" cy="244.1" r="5"/></g><g class="mark"><title>run from the tail click, 2: 0.97</title><circle class="s2 ring" cx="108.6" cy="36.2" r="5"/></g><g class="mark"><title>run from the tail click, 3: 0.98</title><circle class="s2 ring" cx="153.2" cy="35.0" r="5"/></g><g class="mark"><title>run from the tail click, 4: 0.98</title><circle class="s2 ring" cx="197.8" cy="34.5" r="5"/></g><g class="mark"><title>run from the tail click, 5: 0.98</title><circle class="s2 ring" cx="242.4" cy="34.8" r="5"/></g><g class="mark"><title>run from the tail click, 6: 0.98</title><circle class="s2 ring" cx="287.0" cy="34.9" r="5"/></g><g class="mark"><title>run from the tail click, 7: 0.98</title><circle class="s2 ring" cx="331.6" cy="34.9" r="5"/></g><g class="mark"><title>run from the tail click, 8: 0.98</title><circle class="s2 ring" cx="376.2" cy="34.9" r="5"/></g><g class="mark"><title>run from the tail click, 9: 0.98</title><circle class="s2 ring" cx="420.8" cy="34.9" r="5"/></g><g class="mark"><title>run from the tail click, 10: 0.98</title><circle class="s2 ring" cx="465.4" cy="34.3" r="5"/></g><g class="mark"><title>run from the tail click, 11: 0.98</title><circle class="s2 ring" cx="510.0" cy="34.2" r="5"/></g><polyline class="l1" points="64.0,35.6 108.6,36.4 153.2,36.0 197.8,35.6 242.4,35.4 287.0,35.2 331.6,34.7 376.2,34.7 420.8,34.7 465.4,34.6 510.0,34.6"/><g class="mark"><title>mean of 20 runs, with mask prompt, 1: 0.97</title><circle class="s1 ring" cx="64.0" cy="35.6" r="5"/></g><g class="mark"><title>mean of 20 runs, with mask prompt, 2: 0.97</title><circle class="s1 ring" cx="108.6" cy="36.4" r="5"/></g><g class="mark"><title>mean of 20 runs, with mask prompt, 3: 0.97</title><circle class="s1 ring" cx="153.2" cy="36.0" r="5"/></g><g class="mark"><title>mean of 20 runs, with mask prompt, 4: 0.97</title><circle class="s1 ring" cx="197.8" cy="35.6" r="5"/></g><g class="mark"><title>mean of 20 runs, with mask prompt, 5: 0.98</title><circle class="s1 ring" cx="242.4" cy="35.4" r="5"/></g><g class="mark"><title>mean of 20 runs, with mask prompt, 6: 0.98</title><circle class="s1 ring" cx="287.0" cy="35.2" r="5"/></g><g class="mark"><title>mean of 20 runs, with mask prompt, 7: 0.98</title><circle class="s1 ring" cx="331.6" cy="34.7" r="5"/></g><g class="mark"><title>mean of 20 runs, with mask prompt, 8: 0.98</title><circle class="s1 ring" cx="376.2" cy="34.7" r="5"/></g><g class="mark"><title>mean of 20 runs, with mask prompt, 9: 0.98</title><circle class="s1 ring" cx="420.8" cy="34.7" r="5"/></g><g class="mark"><title>mean of 20 runs, with mask prompt, 10: 0.98</title><circle class="s1 ring" cx="465.4" cy="34.6" r="5"/></g><g class="mark"><title>mean of 20 runs, with mask prompt, 11: 0.98</title><circle class="s1 ring" cx="510.0" cy="34.6" r="5"/></g><polyline class="l3" points="64.0,35.6 108.6,36.8 153.2,36.5 197.8,40.1 242.4,37.7 287.0,36.1 331.6,37.7 376.2,37.7 420.8,38.0 465.4,37.4 510.0,37.4"/><g class="mark"><title>mean of 20 runs, no mask prompt, 1: 0.97</title><circle class="s3 ring" cx="64.0" cy="35.6" r="5"/></g><g class="mark"><title>mean of 20 runs, no mask prompt, 2: 0.97</title><circle class="s3 ring" cx="108.6" cy="36.8" r="5"/></g><g class="mark"><title>mean of 20 runs, no mask prompt, 3: 0.97</title><circle class="s3 ring" cx="153.2" cy="36.5" r="5"/></g><g class="mark"><title>mean of 20 runs, no mask prompt, 4: 0.95</title><circle class="s3 ring" cx="197.8" cy="40.1" r="5"/></g><g class="mark"><title>mean of 20 runs, no mask prompt, 5: 0.96</title><circle class="s3 ring" cx="242.4" cy="37.7" r="5"/></g><g class="mark"><title>mean of 20 runs, no mask prompt, 6: 0.97</title><circle class="s3 ring" cx="287.0" cy="36.1" r="5"/></g><g class="mark"><title>mean of 20 runs, no mask prompt, 7: 0.97</title><circle class="s3 ring" cx="331.6" cy="37.7" r="5"/></g><g class="mark"><title>mean of 20 runs, no mask prompt, 8: 0.97</title><circle class="s3 ring" cx="376.2" cy="37.7" r="5"/></g><g class="mark"><title>mean of 20 runs, no mask prompt, 9: 0.96</title><circle class="s3 ring" cx="420.8" cy="38.0" r="5"/></g><g class="mark"><title>mean of 20 runs, no mask prompt, 10: 0.97</title><circle class="s3 ring" cx="465.4" cy="37.4" r="5"/></g><g class="mark"><title>mean of 20 runs, no mask prompt, 11: 0.97</title><circle class="s3 ring" cx="510.0" cy="37.4" r="5"/></g><line class="grid" x1="516.0" y1="34.2" x2="526.0" y2="34.2"/><rect class="s2" x="530.0" y="29.2" width="10" height="10" rx="2"/><text class="t-note" x="546.0" y="38.2" text-anchor="start">run from the tail click: 0.98</text><line class="grid" x1="516.0" y1="34.6" x2="526.0" y2="51.2"/><rect class="s1" x="530.0" y="46.2" width="10" height="10" rx="2"/><text class="t-note" x="546.0" y="55.2" text-anchor="start">mean of 20 runs, with mask prompt: 0.98</text><line class="grid" x1="516.0" y1="37.4" x2="526.0" y2="68.2"/><rect class="s3" x="530.0" y="63.2" width="10" height="10" rx="2"/><text class="t-note" x="546.0" y="72.2" text-anchor="start">mean of 20 runs, no mask prompt: 0.97</text><text class="t-tick" x="287.0" y="290.0" text-anchor="middle">round</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">IoU with the target</text></svg><figcaption>The IoU with the target after each round. The run from the tail click jumps from 0.027 to 0.972 in round 2. The mean of 20 games ends at 0.979 with the mask prompt and at 0.967 without it.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 250" role="img" aria-label="Mean IoU with the tight-box mask for noisy boxes: 0.995 with the training noise, 0.889, 0.514 and 0.225 with larger noise."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Noisy boxes around the right cat: mean IoU with the tight-box mask (30 boxes each)</text><line class="grid" x1="250.0" y1="44.0" x2="250.0" y2="220.0"/><text class="t-muted" x="250.0" y="236.0" text-anchor="middle">0.0</text><line class="grid" x1="415.0" y1="44.0" x2="415.0" y2="220.0"/><text class="t-muted" x="415.0" y="236.0" text-anchor="middle">0.5</text><line class="grid" x1="580.0" y1="44.0" x2="580.0" y2="220.0"/><text class="t-muted" x="580.0" y="236.0" text-anchor="middle">1.0</text><text class="t-tick t-strong" x="240.0" y="65.0" text-anchor="end">10%, max 20 px (training)</text><text class="t-muted" x="240.0" y="81.0" text-anchor="end">corners move 16 px on average</text><g class="mark"><title>10%, max 20 px (training): mean IoU 0.995, worst 0.970</title><rect class="s3" x="250" y="54" width="328.2" height="22" rx="3"/></g><text class="t-val" x="586.2" y="70.0" text-anchor="start">0.995  (30/30 above 0.9)</text><text class="t-tick" x="240.0" y="109.0" text-anchor="end">10%, no limit</text><text class="t-muted" x="240.0" y="125.0" text-anchor="end">corners move 25 px on average</text><g class="mark"><title>10%, no limit: mean IoU 0.889, worst 0.000</title><rect class="s1" x="250" y="98" width="293.4" height="22" rx="3"/></g><text class="t-val" x="551.4" y="114.0" text-anchor="start">0.889  (22/30 above 0.9)</text><text class="t-tick" x="240.0" y="153.0" text-anchor="end">20%, no limit</text><text class="t-muted" x="240.0" y="169.0" text-anchor="end">corners move 50 px on average</text><g class="mark"><title>20%, no limit: mean IoU 0.514, worst 0.000</title><rect class="s1" x="250" y="142" width="169.5" height="22" rx="3"/></g><text class="t-val" x="427.5" y="158.0" text-anchor="start">0.514  (8/30 above 0.9)</text><text class="t-tick" x="240.0" y="197.0" text-anchor="end">40%, no limit</text><text class="t-muted" x="240.0" y="213.0" text-anchor="end">corners move 101 px on average</text><g class="mark"><title>40%, no limit: mean IoU 0.225, worst 0.000</title><rect class="s1" x="250" y="186" width="74.2" height="22" rx="3"/></g><text class="t-val" x="332.2" y="202.0" text-anchor="start">0.225  (3/30 above 0.9)</text></svg><figcaption>Noisy boxes around the right cat. With the noise of training, the masks hardly change (mean IoU 0.995). With larger noise, the boxes cut off parts of the cat or become very loose, and the mean IoU falls to 0.889, 0.514 and 0.225.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 234" role="img" aria-label="Left: a multi-task model learns a fixed set of tasks and is tested on the same tasks. Right: one promptable model becomes a part of different systems: a detector gives boxes for instance masks, a grid of points gives object proposals and edges, a person gives clicks."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Multi-task model</text><text class="t-title" x="400.0" y="24.0" text-anchor="start">Promptable model, used as a component</text><rect class="box-1" x="20.0" y="44.0" width="110.0" height="30.0" rx="6"/><text class="t-tick" x="75.0" y="64.0" text-anchor="middle">semantic</text><line class="edge" x1="134.0" y1="59.0" x2="170.0" y2="100.0" marker-end="url(#ah)"/><line class="edge" x1="290.0" y1="100.0" x2="326.0" y2="59.0" marker-end="url(#ah)"/><rect class="box-1" x="330.0" y="44.0" width="50.0" height="30.0" rx="6"/><text class="t-tick" x="355.0" y="64.0" text-anchor="middle">sema.</text><rect class="box-1" x="20.0" y="84.0" width="110.0" height="30.0" rx="6"/><text class="t-tick" x="75.0" y="104.0" text-anchor="middle">instance</text><line class="edge" x1="134.0" y1="99.0" x2="170.0" y2="100.0" marker-end="url(#ah)"/><line class="edge" x1="290.0" y1="100.0" x2="326.0" y2="99.0" marker-end="url(#ah)"/><rect class="box-1" x="330.0" y="84.0" width="50.0" height="30.0" rx="6"/><text class="t-tick" x="355.0" y="104.0" text-anchor="middle">inst.</text><rect class="box-1" x="20.0" y="124.0" width="110.0" height="30.0" rx="6"/><text class="t-tick" x="75.0" y="144.0" text-anchor="middle">panoptic</text><line class="edge" x1="134.0" y1="139.0" x2="170.0" y2="100.0" marker-end="url(#ah)"/><line class="edge" x1="290.0" y1="100.0" x2="326.0" y2="139.0" marker-end="url(#ah)"/><rect class="box-1" x="330.0" y="124.0" width="50.0" height="30.0" rx="6"/><text class="t-tick" x="355.0" y="144.0" text-anchor="middle">pano.</text><rect class="box-on" x="174.0" y="70.0" width="112.0" height="60.0" rx="8"/><text class="t-note" x="230.0" y="96.0" text-anchor="middle">one model</text><text class="t-tick" x="230.0" y="114.0" text-anchor="middle">3 fixed tasks</text><text class="t-muted" x="20.0" y="184.0" text-anchor="start">train: these 3 tasks</text><text class="t-muted" x="250.0" y="184.0" text-anchor="start">test: the same 3</text><rect class="box-2" x="400.0" y="44.0" width="124.0" height="28.0" rx="6"/><text class="t-tick" x="462.0" y="63.0" text-anchor="middle">detector</text><line class="edge" x1="528.0" y1="58.0" x2="566.0" y2="58.0" marker-end="url(#ah)"/><text class="t-muted" x="547.0" y="49.0" text-anchor="middle">boxes</text><rect class="box-on" x="570.0" y="44.0" width="54.0" height="28.0" rx="6"/><text class="t-tick" x="597.0" y="63.0" text-anchor="middle">SAM</text><line class="edge" x1="628.0" y1="58.0" x2="650.0" y2="58.0" marker-end="url(#ah)"/><text class="t-tick" x="654.0" y="63.0" text-anchor="start">instance masks</text><rect class="box-2" x="400.0" y="80.0" width="124.0" height="28.0" rx="6"/><text class="t-tick" x="462.0" y="99.0" text-anchor="middle">grid of points</text><line class="edge" x1="528.0" y1="94.0" x2="566.0" y2="94.0" marker-end="url(#ah)"/><text class="t-muted" x="547.0" y="85.0" text-anchor="middle">points</text><rect class="box-on" x="570.0" y="80.0" width="54.0" height="28.0" rx="6"/><text class="t-tick" x="597.0" y="99.0" text-anchor="middle">SAM</text><line class="edge" x1="628.0" y1="94.0" x2="650.0" y2="94.0" marker-end="url(#ah)"/><text class="t-tick" x="654.0" y="99.0" text-anchor="start">object proposals</text><rect class="box-2" x="400.0" y="116.0" width="124.0" height="28.0" rx="6"/><text class="t-tick" x="462.0" y="135.0" text-anchor="middle">grid + edge filter</text><line class="edge" x1="528.0" y1="130.0" x2="566.0" y2="130.0" marker-end="url(#ah)"/><text class="t-muted" x="547.0" y="121.0" text-anchor="middle">points</text><rect class="box-on" x="570.0" y="116.0" width="54.0" height="28.0" rx="6"/><text class="t-tick" x="597.0" y="135.0" text-anchor="middle">SAM</text><line class="edge" x1="628.0" y1="130.0" x2="650.0" y2="130.0" marker-end="url(#ah)"/><text class="t-tick" x="654.0" y="135.0" text-anchor="start">edges</text><rect class="box-2" x="400.0" y="152.0" width="124.0" height="28.0" rx="6"/><text class="t-tick" x="462.0" y="171.0" text-anchor="middle">a person</text><line class="edge" x1="528.0" y1="166.0" x2="566.0" y2="166.0" marker-end="url(#ah)"/><text class="t-muted" x="547.0" y="157.0" text-anchor="middle">clicks</text><rect class="box-on" x="570.0" y="152.0" width="54.0" height="28.0" rx="6"/><text class="t-tick" x="597.0" y="171.0" text-anchor="middle">SAM</text><line class="edge" x1="628.0" y1="166.0" x2="650.0" y2="166.0" marker-end="url(#ah)"/><text class="t-tick" x="654.0" y="171.0" text-anchor="start">interactive masks</text><text class="t-muted" x="400.0" y="202.0" text-anchor="start">train: one task (prompt → valid mask)</text><text class="t-muted" x="400.0" y="220.0" text-anchor="start">test: new tasks, by what you connect to it</text></svg><figcaption>Left: a multi-task model learns three fixed tasks and is tested on the same three. Right: SAM learns one task, and each new task comes from the part that makes its prompts: a detector, a grid of points or a person.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 236" role="img" aria-label="Two examples of composition. CLIP is one block inside the DALL·E image generator. SAM is one block inside a cat instance segmenter, after a cat detector that gives the boxes."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Composition: a foundation model as one block of a larger system</text><rect class="box-1" x="20.0" y="46.0" width="140.0" height="46.0" rx="8"/><text class="t-note" x="90.0" y="74.0" text-anchor="middle">text</text><line class="edge" x1="164.0" y1="69.0" x2="196.0" y2="69.0" marker-end="url(#ah)"/><rect class="box-ghost" x="200.0" y="40.0" width="330.0" height="58.0" rx="10"/><text class="t-tick" x="212.0" y="58.0" text-anchor="start">DALL·E (an image generator)</text><rect class="box-on" x="300.0" y="62.0" width="90.0" height="30.0" rx="6"/><text class="t-note" x="345.0" y="82.0" text-anchor="middle">CLIP</text><text class="t-muted" x="400.0" y="82.0" text-anchor="start">ranks the pictures</text><line class="edge" x1="534.0" y1="69.0" x2="566.0" y2="69.0" marker-end="url(#ah)"/><rect class="box-3" x="570.0" y="46.0" width="170.0" height="46.0" rx="8"/><text class="t-note" x="655.0" y="74.0" text-anchor="middle">pictures</text><rect class="box-1" x="20.0" y="136.0" width="140.0" height="46.0" rx="8"/><text class="t-note" x="90.0" y="164.0" text-anchor="middle">photo</text><line class="edge" x1="164.0" y1="159.0" x2="196.0" y2="159.0" marker-end="url(#ah)"/><rect class="box-ghost" x="200.0" y="124.0" width="330.0" height="70.0" rx="10"/><text class="t-tick" x="212.0" y="142.0" text-anchor="start">a cat instance segmenter</text><rect class="box-2" x="212.0" y="152.0" width="110.0" height="30.0" rx="6"/><text class="t-tick" x="267.0" y="172.0" text-anchor="middle">cat detector</text><line class="edge" x1="326.0" y1="167.0" x2="360.0" y2="167.0" marker-end="url(#ah)"/><text class="t-muted" x="343.0" y="157.0" text-anchor="middle">box</text><rect class="box-on" x="364.0" y="152.0" width="70.0" height="30.0" rx="6"/><text class="t-note" x="399.0" y="172.0" text-anchor="middle">SAM</text><line class="edge" x1="534.0" y1="159.0" x2="566.0" y2="159.0" marker-end="url(#ah)"/><rect class="box-3" x="570.0" y="136.0" width="170.0" height="46.0" rx="8"/><text class="t-note" x="655.0" y="164.0" text-anchor="middle">one mask per cat</text><text class="t-muted" x="20.0" y="222.0" text-anchor="start">Neither block was trained for the whole system. The prompt is the connection between them.</text></svg><figcaption>Two examples of composition. CLIP is one block in the DALL·E image generator: it ranks the candidate pictures. SAM is one block in a cat instance segmenter: a cat detector gives the boxes, and SAM gives one mask for each box.</figcaption></figure>

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
