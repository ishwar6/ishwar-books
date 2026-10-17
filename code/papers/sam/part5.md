---
title: "Zero-Shot Experiments: One Model, Five Tasks"
description: "The paper tests SAM on pictures and jobs that it never saw in training. We explain each score, and we repeat the tests in small on 40 objects drawn by people."
part: 5
covers: "§7, Figures 8 to 13, Tables 3 to 5, Appendix D, Appendix E"
date: 2026-10-17
tags: [sam, segmentation, evaluation, zero-shot, computer-vision]
---

A model that only works on its own training pictures is a good student of one textbook. Section 7 gives SAM an exam with questions from other books, and we sit a small copy of that exam with the released model.

## Five tasks, one model {§7}

> [!PAPER] · §7 · page 8
> [![The first sentences of Section 7: zero-shot transfer experiments with SAM on five tasks, four of which differ significantly from the promptable segmentation task](/img/papers/sam/p5-intro-1a.png)](/img/papers/sam/p5-intro-1a.png)
>
> [![The end of the paragraph: zero-shot transfer follows its usage in CLIP, and the datasets include underwater or ego-centric images that do not appear in SA-1B](/img/papers/sam/p5-intro-1b.png)](/img/papers/sam/p5-intro-1b.png)
>
> **What it says:** the team runs "zero-shot transfer experiments" on "five tasks, four of which differ significantly" from the task that trained SAM. The datasets and tasks were "not seen during training". The term follows its "usage in CLIP [82]". Some pictures, such as "underwater or ego-centric images", do not appear in SA-1B.
>
> **What it means:**
>
> Part 1 defined zero-shot transfer: use a trained model on a new job with no more training. [Part 2](part-2-the-task.md) showed the key that makes it possible: the prompt. Here the team puts the claim to work. They take the one SAM model, freeze it, and point it at datasets made by other people for other purposes.
>
> "Ego-centric" means pictures from a camera on a person's head, so the hands and the tools fill the view. Underwater pictures are blue and blurred. SA-1B has neither kind. So a good score here cannot come from memory of similar training pictures.
>
> Why does "five tasks, four of which differ" matter? The first task is the training task itself, on new pictures. The other four are new jobs. For them, the team must invent a prompt that turns the new job into "give me a mask".

> [!DEFINITION] Zero-shot transfer
> Use a trained model on a new dataset or a new task with no more training and no examples of the new task. For SAM, the only tool is the prompt: [Part 2](part-2-the-task.md#zero-shot-transfer-the-prompt-is-the-connection) explains why the prompt is the connection.

> [!DEFINITION] Benchmark
> A fixed test set plus a fixed way to score it. Everybody runs their model on the same pictures with the same rules, so the scores can be compared. A **baseline** is an earlier method that the new model is compared with.

> [!PAPER] · §7 · page 8
> [![The second paragraph of Section 7: a valid mask from any prompt, a single foreground point, then edge detection, segment everything, segment detected objects and segment objects from free-form text, implemented via prompt engineering](/img/papers/sam/p5-intro-2.png)](/img/papers/sam/p5-intro-2.png)
>
> **What it says:** the tests start with the core goal, "producing a valid mask from any prompt", with "a single foreground point prompt". Then SAM must "(1) perform edge detection, (2) segment everything", "(3) segment detected objects", and "segment objects from free-form text". All four are "implemented via prompt engineering".
>
> **What it means:**
>
> The order is a short history of computer vision. Edge detection is the oldest and "lowest" job: it only asks where one region ends and the next one starts. Object proposals ask "where could there be objects?". Instance segmentation asks "where is each person, each car?". Text asks "where is the thing with this name?". Each step needs more understanding of the scene.
>
> "Prompt engineering" is the method for every step. The model never changes. A small program around it makes the prompts: a grid of points, a detector's boxes, a sentence. The figure below shows the plan.

{{FIG:p5_ladder|The five experiments of §7. The core test gives SAM one point and expects one valid mask. The four new tasks climb from low-level to high-level vision. In every row, the weights are the same; only the prompts change.}}

## One point, one mask {§7.1}

> [!PAPER] · §7.1 · page 8
> [![The Task paragraph of Section 7.1: the task is ill-posed, ground truth does not enumerate all masks, the mIoU metric, a human study with ratings from 1 to 10, points from the center of the mask, the most confident mask, and RITM as the main baseline](/img/papers/sam/p5-71-task.png)](/img/papers/sam/p5-71-task.png)
>
> **What it says:** "This task is ill-posed as one point can refer to multiple objects", and datasets "do not enumerate all possible masks". So the team adds to "the mean of all IoUs between predicted and ground truth masks" a human study, with ratings "from 1 (nonsense) to 10 (pixel-perfect)". The point sits at the "maximal value of the mask's interior distance transform". SAM is judged on its "most confident mask". The main baseline is "RITM [92]".
>
> **What it means:**
>
> "Ill-posed" means that the question has more than one right answer. A point on a shirt can mean the shirt or the whole person, as [Part 2](part-2-the-task.md) showed. But each test dataset stores only one answer: the mask that its annotator drew. If SAM gives the person and the dataset holds the shirt, SAM loses points for a valid answer.
>
> The team handles this in two ways. First, they keep the automatic score, mIoU, because it is cheap and standard. Second, they ask people to look at the masks and grade them, with no reference mask at all. We come back to the human study below.
>
> RITM is a model built for interactive segmentation: a person clicks, the model draws a mask, the person clicks again to fix it. It was trained on COCO and LVIS. It is the strongest of the earlier click-to-mask models on this benchmark, so it is the fair opponent.

### The score: mean IoU

Part 1 explained IoU with two sheets of tracing paper. One IoU grades one mask. A benchmark has thousands of masks, so it needs one number for all of them. The simplest choice is the average, as a teacher averages the marks of a class.

$$
\text{mIoU} = \frac{1}{N} \sum_{i=1}^{N} \text{IoU}(M_i, G_i), \qquad \text{IoU}(M, G) = \frac{|M \cap G|}{|M \cup G|}
$$

Read it from the right. $$G_i$$ is the ground-truth mask of object $$i$$, drawn by a person. $$M_i$$ is the mask that the model gives for it. $$|M \cap G|$$ counts the pixels in both masks, and $$|M \cup G|$$ counts the pixels in at least one. $$N$$ is the number of objects, and the sum adds their IoUs.

A tiny example with three objects. SAM gets IoU 0.9 on a cup, 0.8 on a dog and 0.1 on a shirt, where it gave the whole person. The mIoU is $$(0.9 + 0.8 + 0.1) / 3 = 0.600$$. One ambiguous click pulls the class average down by a third. This is why the paper says "low absolute mIoU at 1 point is the result of ambiguity".

The paper averages twice: first over the objects of one dataset, then over the 23 datasets. So a small dataset counts as much as a large one.

> [!DEFINITION] mIoU (mean intersection over union)
> The average of the IoUs of many masks, each against its ground-truth mask. It goes from 0 (no overlap anywhere) to 1 (every mask exact). In §7.1, it is the average over the objects of each dataset, then over the 23 datasets.

### Twenty-three new test sets

> [!PAPER] · Figure 8 · page 9
> [![Figure 8 of the paper: 23 small sample pictures with coloured masks, one from each dataset: a theatre hall, microscope cells, a street, doors, a painting, hands from a head camera, a drone view of houses, toys, dolphins underwater, parked cars, plants seen from above, deer in an infrared picture, logs, an X-ray picture of a bag, a kitchen and waste on a conveyor belt](/img/papers/sam/p5-figure8.png)](/img/papers/sam/p5-figure8.png)
>
> **What it says:** "Samples from the 23 diverse segmentation datasets used to evaluate SAM's zero-shot transfer capabilities."
>
> **What it means:**
>
> Look at the variety. There are cells under a microscope (BBBC038v1), an X-ray of a bag (PIDRay), dolphins under water (NDD20), and paintings (DRAM). There are first-person kitchen views (EgoHOS, VISOR, GTEA), and a fish-eye car camera (WoodScape). A model that learned "what an object looks like in holiday photos" would fail on most of them.
>
> Appendix D.1 adds two practical details. The team cut each large dataset down to about 10,000 masks, so that one dataset does not cost days. And they blurred the faces of people in all of them. One of the 23 is LVIS, which uses the photos of COCO. Our own small test below uses COCO photos too.

### Where to click: the center of the object

The paper does not let a person click. A person clicks in a different place each time, so two runs of the benchmark would differ. Instead, a fixed rule chooses the point, the same rule as earlier papers on interactive segmentation:

> [!PAPER] · §D.1 · page 19
> [![Appendix D.1, point sampling and evaluation: the first point is the point farthest from the object boundary, each later point is the farthest from the boundary of the error region, N is 1, 2, 3, 5 or 9, and the per-dataset mIoUs are averaged across all 23 datasets](/img/papers/sam/p5-d1-points.png)](/img/papers/sam/p5-d1-points.png)
>
> **What it says:** "The first point is chosen deterministically as the point farthest from the object boundary. Each subsequent point is the farthest from the boundary of the error region between ground truth and the previous prediction." The scores use "N = {1, 2, 3, 5, 9}" points, and the top-line number averages "the per-dataset mIoUs across all 23 datasets".
>
> **What it means:**
>
> Picture the mask as a lake on a map. Walk inland from every point of the shore, and ask how far you are from the nearest shore. The point that is farthest from all shores is the "center" of the lake. For a round object, it is the middle. For a letter C, it is somewhere in the thick part of the stroke, never in the empty gap.
>
> The later points copy a careful annotator. After each answer, compare it with the true mask. Some pixels were missed, and some were added by mistake: together they are the **error region**. Find the largest blob of error and click in its own center. If the blob is missed object, the click is a foreground point. If it is extra background, the click is a background point.
>
> The paper scores SAM after 1, 2, 3, 5 and 9 such points. The usual protocol in interactive segmentation counts how many clicks a model needs to reach a target IoU, with up to 20 clicks. SAM's use cases have one click, or a few, so the paper scores those.

> [!DEFINITION] Distance transform
> A picture in which each pixel of a mask stores its distance to the nearest pixel outside the mask. The edge of the mask gets small values; the deep inside gets large values. The pixel with the largest value is the point farthest from the boundary.

The rule fits in one line:

$$
p_1 = \arg\max_{p \in G} \; d(p, \partial G)
$$

$$G$$ is the ground-truth mask, $$\partial G$$ is its boundary, and $$d(p, \partial G)$$ is the distance from pixel $$p$$ to the nearest boundary pixel. $$\arg\max$$ returns the pixel, not the distance. For the later points, replace $$G$$ by the larger of the two error blobs.

A tiny example: a mask of 5 rows and 7 columns. The distance transform counts the steps to the outside. The outer ring of pixels gets 1, the next ring gets 2, and the middle row gets 3:

```text
1 1 1 1 1 1 1
1 2 2 2 2 2 1
1 2 3 3 3 2 1
1 2 2 2 2 2 1
1 1 1 1 1 1 1
```

The largest value, 3, sits in the middle row. Our code takes the first of the tied pixels; the real distance transform uses straight-line distances, so ties are rare.

### Our small copy of the test

We cannot run the paper's 23 datasets. We built a small honest copy. The photos and masks come from COCO, a public dataset of everyday scenes with masks drawn by people. We read it from Hugging Face with no login. We took the first 16 pictures, and in each picture up to 4 objects at random, all larger than 24 × 24 pixels:

```text
COCO panoptic val2017 (Hugging Face nielsr/coco-panoptic-val2017), the first 16 images
objects: 40 from 13 images (up to 4 per image, chosen at random, area >= 576 px = 24 x 24)
object size: smallest 664 px, median 7,343 px, largest 275,827 px
categories: person 14, chair 3, teddy bear 3, refrigerator 2, potted plant 2, tennis racket 2, backpack 2, tv 1, bear 1, book 1, stop sign 1, bed 1, skis 1, oven 1, baseball glove 1, bird 1, boat 1, handbag 1, cell phone 1
```

That is 40 objects, against about 10,000 masks for each of the paper's datasets. Our numbers can show the effects that the paper describes. They cannot rank SAM against other models, and the error of a mean over 40 objects is large. COCO is also not new to every model: RITM trained on it, and LVIS uses its photos. SAM never trained on it.

For each object, we placed the center point, gave it to SAM ViT-H, and kept all three masks. Here are a few rows of the output:

```text
  # object            area       center  IoU mask 1  mask 2  mask 3   top oracle    1 pt
  1 tv              13,247    (54, 215)       0.025   0.026   0.951     3      3   0.951
  9 stop sign       16,446   (187, 149)       0.050   0.181   0.965     1      3   0.049
 35 cell phone       7,653   (335, 179)       0.012   0.062   0.936     1      3   0.012
mIoU over 40 objects, 1 center point: top-scored mask 0.601, oracle (best of 3) 0.789
the top-scored mask was also the best of the 3 for 17 of 40 objects
```

For the TV, SAM ranked mask 3 first, and mask 3 matches the TV (IoU 0.951). For the stop sign and the cell phone, SAM ranked mask 1 first: a small part. For the stop sign, it is the letter O under the click. Its IoU with the full object is 0.049 for the sign and 0.012 for the phone. Mask 3 was the whole object both times, with IoU 0.965 and 0.936.

### The oracle: best of three

> [!PAPER] · Figure 9 · page 9
> [![Figure 9 of the paper. (a) Horizontal bars: the IoU difference between SAM and RITM on each of 23 datasets, from +46.9 on PPDLS to -21.4 on GTEA, with light circles for the oracle further right. (b) Mean human ratings for 7 datasets: ground truth highest, then SAM, then single-output SAM, then RITM. (c) and (d) mIoU against the number of points, for center points and random points, for SAM, RITM, SimpleClick and FocalClick, with a cross for the SAM oracle at one point](/img/papers/sam/p5-figure9.png)](/img/papers/sam/p5-figure9.png)
>
> **What it says:** "(a) Mean IoU of SAM and the strongest single point segmenter, RITM." "Due to ambiguity, a single mask may not match ground truth; circles show 'oracle' results of the most relevant of SAM's 3 predictions." Panel (b) has ratings "from 1 (worst) to 10 (best)". "Low absolute mIoU at 1 point is the result of ambiguity."
>
> **What it means:**
>
> Panel (a) is the main result. Each bar is one dataset: SAM's mIoU minus RITM's mIoU, with one center point. Orange bars are wins for SAM, up to +46.9 on PPDLS, a dataset of plants seen from above. Blue bars are losses, down to −21.4 on GTEA, a dataset of hands in kitchen videos.
>
> The light circles at the end of each bar are the oracle. For every dataset, even GTEA, the circle is to the right of zero. So SAM always had a mask that beats RITM; on some datasets it ranked another mask first. Panels (c) and (d) show more points, and panel (b) the human study. We take them one at a time.

> [!PAPER] · §7.1 · page 9
> [![The Results paragraph of 7.1: SAM is higher on 16 of the 23 datasets, by as much as 47 IoU, and with the oracle, SAM outperforms RITM on all datasets](/img/papers/sam/p5-71-results.png)](/img/papers/sam/p5-71-results.png)
>
> [![Appendix D.1: the oracle evaluates the predicted mask that best matches ground truth from amongst SAM's three predictions, which relaxes the requirement to guess the one right mask](/img/papers/sam/p5-d1-oracle.png)](/img/papers/sam/p5-d1-oracle.png)
>
> **What it says:** SAM gives "higher results on 16 of the 23 datasets, by as much as ∼47 IoU". The oracle takes "the most relevant of SAM's 3 masks", the one that "best matches ground truth". "With the oracle to perform ambiguity resolution, SAM outperforms RITM on all datasets." It relaxes "the requirement to guess the one right mask among several valid objects".
>
> **What it means:**
>
> "Oracle" is an old word for a voice that knows the answer. Here the oracle is allowed to look at the ground truth and pick the best of SAM's three masks. A real user does the same with their eyes: SAM shows three masks, and the user picks the one they meant.
>
> So the two numbers answer two questions. The top-scored mask answers "how good is SAM if nobody helps it choose?". The oracle answers "does SAM know the right mask at all?". The gap between them is the cost of ambiguity, and it is large.

The oracle score of one object is the best of three IoUs:

$$
\text{IoU}_{\text{oracle}} = \max_{k \in \{1, 2, 3\}} \text{IoU}(M_k, G), \qquad \text{IoU}_{\text{top}} = \text{IoU}(M_{k^*}, G), \quad k^* = \arg\max_k \hat{s}_k
$$

$$M_1, M_2, M_3$$ are SAM's three masks and $$\hat{s}_k$$ is SAM's own score for mask $$k$$, the predicted IoU of [Part 3](part-3-the-model.md). The top mask is the one with the highest score; the oracle mask is the one with the highest true IoU. They are the same mask only when SAM's own ranking agrees with the ground truth. For the stop sign, the three IoUs are 0.050, 0.181 and 0.965. SAM ranked mask 1 first, so the top IoU is 0.050 and the oracle IoU is 0.965.

On our 40 objects, the oracle lifts the mIoU from 0.601 to 0.789. The top-scored mask was also the best mask for only 17 objects. The figure shows every object:

{{FIG:p5_oracle|Our 40 COCO objects with one center point and SAM ViT-H. For each object, orange is the IoU of the mask with the highest score and green is the best of the three masks. For many objects, the best mask is near 0.9 while the top-scored mask is far lower. The means are 0.601 and 0.789.}}

The pictures show three objects. In the first two rows, SAM ranked a small part first. In the last row, the top mask and the oracle are the same mask:

<figure><img src="/img/papers/sam/p5-one-point.png" alt="Three rows of three crops from COCO photos. Each row shows the object that people drew in yellow with the green center point, then SAM's top-scored mask in blue with the people's outline in red, then the oracle mask" loading="lazy" /><figcaption>One center point (green) per object. Left: the mask that people drew. Middle: SAM's top-scored mask (blue; the red line is the people's outline). Right: the oracle, the best of the three masks. The top-scored mask can be a valid part of the object, but the dataset wanted the whole object.</figcaption></figure>

Does this mean that SAM is wrong? Not always. The letter O is a valid answer to a click on the letter O. COCO's annotators chose the whole phone. The automatic score cannot tell "wrong" from "a different valid answer". The human study can.

### The human study

> [!PAPER] · §7.1 · page 9
> [![The human study paragraph of 7.1, first part: annotators rate SAM's masks substantially higher than RITM, and single-output SAM consistently lower](/img/papers/sam/p5-71-human-a.png)](/img/papers/sam/p5-71-human-a.png)
>
> [![The human study paragraph, second part: SAM's mean ratings fall between 7 and 9, and on DRAM and IBD SAM gets higher ratings although it is worse on automatic metrics](/img/papers/sam/p5-71-human-b.png)](/img/papers/sam/p5-71-human-b.png)
>
> **What it says:** annotators rate SAM's masks "substantially higher than the strongest baseline, RITM". A version of SAM with a "single output mask has consistently lower ratings, though still higher than RITM". SAM's means "fall between 7 and 9". On "DRAM and IBD", where SAM loses on mIoU, "it receives consistently higher ratings in the human study".
>
> **What it means:**
>
> A rating of 7 to 9 means "the object is identifiable and errors are small and rare" in the guidelines. So a person looking at SAM's mask usually sees a good mask of a real object.
>
> The single-output version is the "ambiguity-unaware" SAM of Part 3: one mask token and no best-of-three training. It has to blend the possible answers, and people notice. Its masks get lower ratings than SAM's on every dataset in panel (b) but one (see below).
>
> DRAM and IBD are the most telling. On paintings (DRAM) and drone pictures (IBD), mIoU says that RITM is better. People say that SAM is better. The two measures disagree because the measure with a reference mask punishes valid answers that differ from the reference.

> [!DEFINITION] Human study and rating scale
> An experiment in which people judge the outputs of models. Here each annotator sees a picture, the prompt and one mask, and gives a score from 1 to 10. The score does not use a reference mask, so a valid mask of a different object is not punished.

Appendix E explains why the team paid for this study, and how they ran it:

> [!PAPER] · §E · page 23
> [![Appendix E, first paragraph: IoU with ground truth has two limitations: a model may be penalized for a valid mask of a different object, and ground truth masks may include biases that an in-domain model can learn](/img/papers/sam/p5-e-purpose.png)](/img/papers/sam/p5-e-purpose.png)
>
> **What it says:** IoU has "two limitations". A model may be "strongly penalized for returning a valid mask of a different object". And "ground truth masks may include various biases", so a model trained on the same dataset can "obtain a higher IoU without necessarily producing better masks".
>
> **What it means:**
>
> The first limitation is ambiguity, as above. The second is subtler. Every dataset has habits: some annotators trace edges loosely, some never cut holes, some include the hidden part of an object. A model that trained on that dataset learns its habits and copies them. It gets a higher IoU because it agrees with the habits, not because its masks are better.
>
> SAM never saw these datasets, so it cannot learn their habits. Human ratings do not reward the habits. This is the fair test for a zero-shot model.

> [!PAPER] · §E · page 23
> [![Appendix E, Methodology, first lines: masks are shown to professional annotators](/img/papers/sam/p5-e-method-a.png)](/img/papers/sam/p5-e-method-a.png)
>
> [![Appendix E, Methodology, continued: three questions about valid object, clean boundary and match with the input, and the meaning of each score from 1 to 10](/img/papers/sam/p5-e-method-b.png)](/img/papers/sam/p5-e-method-b.png)
>
> [![Appendix E: the standard deviation over five annotators was 0.83, each job had a single annotator, and a job took 90 seconds on average](/img/papers/sam/p5-e-jobs.png)](/img/papers/sam/p5-e-jobs.png)
>
> **What it says:** "professional annotators" answer three questions: "Does the mask correspond to a valid object? Does the mask have a clean boundary? Does the mask correspond to the input?" "A score of 1 indicates a mask that corresponds to no object at all", 2 to 4 means huge errors, 5 to 6 "mostly sensible" with significant errors, 7 to 9 "only minor boundary errors", and "a score of 10 is for masks with no visible errors". With five annotators per job, "the average standard deviation in score over the five annotators was 0.83". After that, each job had "only a single annotator", and a job took "90 seconds" on average.
>
> **What it means:**
>
> The three questions separate three kinds of failure. A mask can show nothing, it can have a ragged edge, or it can be a good mask of the wrong thing. The scale then gives each band a written meaning, so that two annotators mean the same thing by "7".
>
> The 0.83 is the check on that. If five people rate the same mask, their scores differ by less than one point on average. That is consistent enough to trust one rater per mask. The team then rated 1,000 masks per dataset for each of four sources: RITM, single-output SAM, SAM and the ground truth. That is 4,000 ratings per dataset, at 90 seconds each.

Figure 18 in the appendix gives the mean ratings for each dataset. We drew them in one chart:

{{FIG:p5_ratings|The mean ratings of the human study (Figure 18 of the paper). On every dataset, SAM is above RITM, by 1.1 to 2.2 points. The masks drawn by people are still higher. On OVIS, the legend gives single-output SAM a higher mean (7.7) than SAM (7.2).}}

SAM sits between RITM and the masks drawn by people. On LVIS, VISOR, IBD and NDD20, it is within 0.4 of the people. Look at OVIS: the legend of Figure 18 gives 7.2 for SAM and 7.7 for single-output SAM. Table 8 of the same appendix reports that SAM is significantly *above* single-output SAM on OVIS. The paper does not explain the difference, so we report both numbers as printed.

The error bars in panel (b) are **95% confidence intervals**. Each one is the range that holds the true mean with 95% confidence, given the 1,000 sampled masks. With 1,000 ratings per bar, the intervals are narrow, from ±0.06 to ±0.17 in the legends. So differences of a whole point are not luck.

### More points

> [!PAPER] · §7.1 · page 9
> [![The last paragraph of 7.1: as the number of points increases, the gap between methods decreases; SAM is not optimized for the very high IoU regime; with random first points the gap grows](/img/papers/sam/p5-71-points.png)](/img/papers/sam/p5-71-points.png)
>
> **What it says:** "As the number of points increases from 1 to 9, we observe that the gap between methods decreases." SAM "is not optimized for the very high IoU regime". With "random point sampling" for the first point, "the gap between SAM and the baselines grows".
>
> **What it means:**
>
> With nine clicks, the object is no longer ambiguous, and every decent model gets close to the truth. The remaining errors are at the edge: a few pixels of hair or a thin strap. SAM's decoder works at 256 × 256 and its training stopped after 11 rounds of clicks (Part 3). So SAM is quick and good, not pixel-perfect after many clicks.
>
> A random first point is harder than the center: it can land near the edge or on a thin part. RITM drops much more than SAM when that happens. A likely reason: SAM's training game placed its first points anywhere inside the mask (Part 3). A real user, or a point from eye tracking, behaves more like the random point.

We ran the same protocol on our 40 objects. The paper does not say whether the earlier mask goes back in as a mask prompt, so we ran both ways:

```text
mIoU after N points (error-region rule): 1 -> 0.601, 2 -> 0.718, 3 -> 0.786, 5 -> 0.804, 9 -> 0.845
variant: the previous mask logits also go back in as a mask prompt: 1 -> 0.601, 2 -> 0.774, 3 -> 0.844, 5 -> 0.866, 9 -> 0.884
random first point (Figure 9d): 1 -> 0.604, 2 -> 0.704, 3 -> 0.793, 5 -> 0.860, 9 -> 0.841, oracle at 1 point 0.809
```

{{FIG:p5_points|Mean IoU on our 40 objects as the points accumulate. With the previous mask as an extra prompt (green), one more point lifts the mIoU from 0.601 to 0.774, and nine points reach 0.884. With points only (blue), the curve is lower. A random first point (orange, points only) gives about the same curve as the center point on our small sample.}}

The second point does most of the work: it fixes the wrong level of the nest. The cell phone is a clear case. Its first mask was a tiny piece of the print, so almost the whole phone was error region. The second point landed in the middle of that region, at (341, 216), and the IoU went from 0.012 to 0.623.

With points only, the curve is not smooth for every object. The large bear went from 0.935 after one point to 0.334 after three, then back to 0.805 after nine. Each extra point is a new question, and without its last answer SAM can change its mind. With the previous mask as a prompt, SAM keeps its last answer and corrects it, as in the training game of Part 3. This is why the mask-prompt curve is higher.

<figure><img src="/img/papers/sam/p5-more-points.png" alt="Three crops of the photo of a woman holding a cell phone. With one point, SAM's mask is a tiny piece of the print. With two points on the phone, the mask covers most of the phone. With three points, it covers the whole phone" loading="lazy" /><figcaption>The error-region rule on the cell phone, with points only. Each new point (green) lands in the center of the largest missed region. The IoU goes from 0.012 to 0.623 to 0.956. The red line is the mask in COCO.</figcaption></figure>

Our random first point did not start lower: 0.604 against 0.601 for the center point. The paper's effect is clear on 23 datasets, with about 10,000 masks each. On 40 objects, the luck of 40 random clicks is as large as the effect. COCO's objects are also mostly compact, so a random point inside them is rarely near a thin part. This is the limit of a small sample, and we report it as it came out.

## Edges {§7.2}

> [!PAPER] · §7.2 · page 10
> [![The Approach paragraph of 7.2: a 16 by 16 grid of foreground points gives 768 masks, NMS removes duplicates, and edge maps come from Sobel filtering of unthresholded mask probability maps, with edge NMS](/img/papers/sam/p5-72-approach.png)](/img/papers/sam/p5-72-approach.png)
>
> [![Appendix D.2: BSDS500 with annotations from five subjects, the metrics ODS, OIS, AP and R50, and the method: no filter, NMS, a Sobel filter, values kept only on the outer boundary pixels of a mask, a pixel-wise max, normalization and edge NMS](/img/papers/sam/p5-d2.png)](/img/papers/sam/p5-d2.png)
>
> **What it says:** SAM gets "a 16×16 regular grid of foreground points", which gives "768 predicted masks (3 per point)". "Redundant masks are removed by NMS". Edges come from "Sobel filtering of unthresholded mask probability maps", kept only where they meet the "outer boundary pixels" of a mask, then "a pixel-wise max" over all masks, and "edge NMS" to thin them. The test set is BSDS500, with edges from "five different subjects", scored by ODS, OIS, AP and R50.
>
> **What it means:**
>
> SAM has no "edge" output. But a mask has an edge by definition: the line where it stops. So the trick is to make many masks, everywhere in the picture, and collect their outlines. A grid of 16 × 16 points covers the picture evenly, and each point gives three masks at three levels of the nest.
>
> Many masks repeat: two neighbouring points on the same wall give the same wall. NMS removes the copies. Then each remaining mask gives its outline. The outlines of all masks are stacked, and at each pixel the strongest one wins.
>
> "Unthresholded probability maps" is a detail with a reason. A mask cut at 0.5 has a hard, jagged step. The probability itself rises smoothly from 0 to 1 across the edge. The Sobel filter measures the slope of that rise, which gives a soft line with a strength, as a real edge detector does.

> [!DEFINITION] Edge detection
> A task that marks the pixels where one region of a picture ends and another begins: the outline of objects and the borders between surfaces. The output is an **edge map**: a grey picture where bright means "edge here".

> [!DEFINITION] Non-maximum suppression (NMS)
> A clean-up step for many overlapping answers. Sort the answers by score; keep the best; remove every other answer that overlaps it more than a threshold; repeat. For masks and boxes, the overlap is the IoU of their boxes. **Edge NMS** is a version for edge maps: across an edge, keep only the brightest pixel, so a thick line becomes one pixel wide.

The Sobel filter is from 1968, the oldest row of Table 3. It measures how fast the brightness changes. It slides two small tables of weights over the picture:

$$
S_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \qquad S_y = S_x^{\top}, \qquad E = \sqrt{(S_x * I)^2 + (S_y * I)^2}
$$

$$I$$ is the picture (here, one mask's probability map), and $$*$$ means "slide the table over the picture and take the weighted sum at each place". $$S_x * I$$ is the change from left to right; $$S_y * I$$ is the change from top to bottom. $$E$$ combines them into one strength, whatever the direction of the edge.

A tiny example. Three columns of a probability map read 0.0, 0.5, 1.0 in all three rows. $$S_x$$ gives $$(-1 - 2 - 1) \times 0.0 + (1 + 2 + 1) \times 1.0 = 4.0$$ at the middle column. $$S_y$$ gives 0, because nothing changes from top to bottom. So $$E = 4.0$$ there: a strong vertical edge. On a flat region of all 1.0, both are 0.

We built this pipeline on the cat photo of the earlier parts. We have no BSDS500 here, so there is no score, only a picture. We compared it with the Sobel filter applied to the photo itself, the 1968 baseline:

```text
16 x 16 grid -> 768 masks (3 per point), no filter by predicted IoU or stability; 57 left after box NMS 0.7
SAM edge map: 17,338 edge pixels above 0.1 after thinning; Sobel on the photo: 33,837
```

Only 57 of the 768 masks survive NMS. The cat photo has few objects, so most points repeat a mask of a cat, a remote or the blanket. After thinning, the Sobel filter on the photo marks 33,837 pixels above 0.1, and SAM's map marks 17,338, about half.

<figure><img src="/img/papers/sam/p5-edges.png" alt="Three panels: the photo of two cats on a pink blanket with two remotes; the Sobel filter on the photo, full of fine lines from the blanket's texture and the fur; and SAM's edge map, with clean outlines of the cats, the remotes and the blanket folds" loading="lazy" /><figcaption>Left: the photo. Middle: the Sobel filter on the photo, thinned. It marks every change of brightness, including the texture of the blanket and the fur. Right: the edges of SAM's masks, thinned. It marks the borders between things. The dotted patches on the fur come from masks with ragged edges there.</figcaption></figure>

The difference is the whole point of the experiment. The Sobel filter on the photo sees brightness, so every stripe of fur is an edge. The Sobel filter on SAM's masks sees *things*, because each mask is a thing. Our code follows Appendix D.2; the box-NMS threshold (0.7) and the exact edge-NMS settings are our choices, because the paper does not give them.

### How edges are scored

> [!PAPER] · Table 3 · page 10
> [![Figure 10 and Table 3 of the paper: two BSDS500 photos, their human edges and SAM's edges; then a table of ODS, OIS, AP and R50 for HED, EDETR, the Sobel filter, Canny, Felz-Hutt and SAM](/img/papers/sam/p5-fig10-table3.png)](/img/papers/sam/p5-fig10-table3.png)
>
> [![The Results paragraph of 7.2: SAM predicts more edges than the ground truth, R50 is high at the cost of precision, and SAM lags methods that learn which edges to suppress](/img/papers/sam/p5-72-results.png)](/img/papers/sam/p5-72-results.png)
>
> **What it says:** "SAM was not trained to predict edge maps". In Table 3, SAM gets ODS .768, OIS .786, AP .794 and R50 .928. The Sobel filter gets .539 and Canny .600. HED, trained on BSDS500, gets .788, and EDETR .840. "SAM predicts more edges", so "recall at 50% precision (R50) is high, at the cost of precision". It lags methods that learn "which edges to suppress".
>
> **What it means:**
>
> Look at Figure 10. The people who drew BSDS500 drew only the main outlines. SAM also draws the windows of the building and the small leaves. Many of these edges are real, but the people did not draw them, so the score counts them as false alarms.
>
> The numbers make this exact. SAM is far above every older zero-shot method (.768 against .539 to .610) and close to HED (.788), a 2015 network trained on BSDS500 itself. EDETR, trained on BSDS500 in 2022, is clearly better (.840). It has learned the taste of the BSDS annotators: which edges they leave out. SAM's R50 (.928) is as high as EDETR's (.930): when SAM is allowed to be half right, it finds almost every edge.

The four metrics come from two counts, precision and recall:

$$
P = \frac{\text{true edge pixels found}}{\text{edge pixels predicted}}, \qquad R = \frac{\text{true edge pixels found}}{\text{true edge pixels}}, \qquad F = \frac{2 P R}{P + R}
$$

Precision $$P$$ asks "of what I marked, how much was right?". Recall $$R$$ asks "of what was there, how much did I find?". The F-score combines them, and it is high only when both are high. A predicted pixel counts as "found" if it lies within a small distance of a human edge.

A tiny example. The people drew 100 edge pixels. A model marks 150 pixels, and 90 of them match. Then $$P = 90 / 150 = 0.60$$, $$R = 90 / 100 = 0.90$$, and $$F = 2 \times 0.60 \times 0.90 / 1.50 = 0.72$$. A model that marks more edges raises $$R$$ and lowers $$P$$. That is SAM's profile.

An edge map is grey, not black and white, so a threshold decides which pixels count as "marked". Each metric handles the threshold differently:

- **ODS** (optimal dataset scale): one threshold for the whole dataset, the one that gives the best F. Report that F.
- **OIS** (optimal image scale): the best threshold for each picture separately. It is always at least the ODS.
- **AP** (average precision): sweep all thresholds, draw precision against recall, and take the area under the curve.
- **R50**: the recall at the threshold where precision is 50%.

So R50 rewards a model that finds every edge even with many false alarms, and ODS punishes those false alarms. SAM's high R50 and lower ODS tell the same story as the picture: SAM finds more edges than people draw.

> [!DEFINITION] Precision and recall
> **Precision** is the share of the model's answers that are correct. **Recall** is the share of the correct answers that the model found. A model that answers "everything" has perfect recall and poor precision; a model that answers only when it is sure has the opposite.

## Object proposals {§7.3}

> [!PAPER] · §7.3 · page 10
> [![Section 7.3: the standard average recall metric on LVIS v1, chosen for its large number of categories, against a ViTDet detector, a baseline shown to game AR](/img/papers/sam/p5-73-ar.png)](/img/papers/sam/p5-73-ar.png)
>
> **What it says:** the team computes "the standard average recall (AR) metric on LVIS v1", because its "large number of categories" makes a hard test. The baseline is a ViTDet detector. It corresponds to a method "that was shown to game AR", so it is a demanding comparison.
>
> **What it means:**
>
> An object proposal is a guess: "there may be an object here". Old detection systems worked in two steps. First, a cheap method listed about a thousand guesses per picture. Then a slow classifier looked at each guess and named it. If the first step missed an object, the second step could never find it. So a proposal method is graded on one thing: did it miss anything?
>
> That is recall. Precision does not matter much here, because the classifier will throw away the bad guesses. LVIS has 1,203 categories, from "person" to "mushroom" and "pocket watch". A proposal method must catch all of them without knowing any names.
>
> "Game AR" is honest language. A trained detector, used as a proposal method, already knows the categories of LVIS. Its guesses are excellent on exactly the objects that the test counts. So this baseline is hard to beat.

> [!DEFINITION] Object proposal
> A mask or box that marks a place where an object may be, with no name attached. A proposal method returns many of them per picture, here up to 1,000, and is judged by how many real objects they cover.

Average recall is recall, averaged over strictness. An object counts as "found" if some proposal overlaps it with an IoU of at least $$t$$. Then:

$$
\text{AR@1000} = \frac{1}{10} \sum_{t \in \{0.50, 0.55, \ldots, 0.95\}} \text{Recall}(t), \qquad \text{Recall}(t) = \frac{\#\{\text{objects with a proposal of IoU} \ge t\}}{\#\{\text{objects}\}}
$$

"@1000" means that only the top 1,000 proposals of each picture count. $$t$$ takes 10 values, from loose (0.50: half the overlap is enough) to strict (0.95: almost exact). The average over $$t$$ rewards a proposal method that finds objects *and* outlines them well.

A tiny example with 4 objects. The best proposals have IoUs of 0.97, 0.82, 0.66 and 0.40 with them. From $$t = 0.50$$ to $$0.65$$, three objects are found: recall 0.75. From $$t = 0.70$$ to $$0.80$$: two, recall 0.50. From $$t = 0.85$$: one, recall 0.25. The ten recalls are 0.75, 0.75, 0.75, 0.75, 0.50, 0.50, 0.50, 0.25, 0.25, 0.25, and their mean is AR = 0.525.

> [!PAPER] · §D.3 · page 21
> [![Appendix D.3, method: no image crops, no filtering by predicted IoU and stability, a 64 by 64 point grid and an NMS threshold of 0.9, about 900 masks per image, ranked by the average of confidence and stability](/img/papers/sam/p5-d3.png)](/img/papers/sam/p5-d3.png)
>
> **What it says:** the team uses "a 64×64 point grid and an NMS threshold of 0.9", which "produces ∼900 masks per image on average". There is no filter by predicted IoU or stability. With more than 1,000 masks, they rank them by "the average of their confidence and stability scores" and keep the top 1,000.
>
> **What it means:**
>
> 64 × 64 points with 3 masks each is 12,288 masks per picture. The NMS threshold of 0.9 is very loose: it removes a mask only if its box is almost identical to a better mask's box. Proposals want many guesses, so the team keeps almost everything.
>
> "Stability" is a score that SAM's automatic mask generator uses (Part 4). Cut the mask's probability map at a slightly higher and a slightly lower level, and compare the two masks. A good mask has a sharp edge, so both cuts give almost the same mask.

> [!DEFINITION] Stability score
> The IoU between the mask cut at logit +1 and the mask cut at logit −1 (the logit is the raw score before the sigmoid). A crisp mask barely changes between the two cuts and scores near 1. A blurry, unsure mask changes a lot and scores lower.

> [!PAPER] · Table 4 · page 10
> [![Table 4: mask AR@1000 on LVIS v1 for ViTDet-H, single-output SAM and SAM, overall, by size and by category frequency](/img/papers/sam/p5-table4.png)](/img/papers/sam/p5-table4.png)
>
> [![The Results paragraph of 7.3: SAM outperforms ViTDet-H on medium and large objects and on rare and common objects; the single-output SAM is worse on all AR metrics](/img/papers/sam/p5-73-results.png)](/img/papers/sam/p5-73-results.png)
>
> **What it says:** ViTDet-H gets AR@1000 63.0 overall and SAM 59.3. SAM "outperforms ViTDet-H on medium and large objects" (81.6 against 80.8) and on "rare and common objects" (65.8 against 58.3, 63.9 against 63.3). It is behind on small objects (45.5 against 51.7) and frequent ones (59.1 against 63.1). The single-output SAM, at 54.9, is "significantly worse than SAM on all AR metrics".
>
> **What it means:**
>
> The pattern fits the story of habits. ViTDet trained on LVIS. Frequent categories have thousands of training examples, so ViTDet knows them well; it also learned how LVIS annotators draw small things. On rare categories, ViTDet saw only a few examples, and SAM, which knows no categories at all, wins by 7.5 points.
>
> The single-output row is the cleanest result in the table. Three masks per point give proposals at three scales from one click: the button, the shirt and the person. One mask per point gives one scale. Even with a denser grid of 128 × 128 points, the single-output model loses 4.4 points overall.

> [!DEFINITION] Small, medium and large objects
> The size classes of COCO and LVIS. Small objects have an area below 32 × 32 = 1,024 pixels, large objects above 96 × 96 = 9,216 pixels, and medium objects are in between.

We ran a small version on all objects of our 13 COCO pictures. To save time, we used a grid of 32 × 32 points, not 64 × 64, with the same NMS threshold, ranking and limit:

```text
  image 139: 3,072 masks -> 430 after box NMS 0.9; 20 objects drawn by people
  image 285: 3,072 masks -> 165 after box NMS 0.9; 1 objects drawn by people
  ...
  image 1503: 3,072 masks -> 293 after box NMS 0.9; 5 objects drawn by people
ground-truth objects: 111 (small 41, medium 40, large 30); proposals per image: mean 306
recall at each IoU threshold (top 1000): 0.50 0.820, 0.55 0.793, 0.60 0.748, 0.65 0.730, 0.70 0.667, 0.75 0.622, 0.80 0.568, 0.85 0.441, 0.90 0.234, 0.95 0.072
AR@1000 = mean of the 10 recalls = 0.569   (AR@100 = 0.458)
AR@1000 by size: small 0.329, medium 0.647, large 0.793
```

{{FIG:p5_recall|Recall of our SAM proposals at each of the ten IoU thresholds. At 0.50, almost every object has a matching proposal. At 0.95, few proposals are almost exact. Average recall is the mean of the ten bars.}}

Our grid gives 3,072 masks per picture, and NMS leaves between 143 and 560 (306 on average). So our "AR@1000" never reaches its limit of 1,000; the paper's denser grid gives about 900. Even so, 82% of the 111 objects have a proposal with IoU at least 0.5. At the strict end, only 7.2% have a proposal with IoU at least 0.95. Part of that is COCO itself: its polygons are coarse, and a crisp SAM mask cannot match them pixel for pixel.

Our AR@1000 is 0.569, and the pattern by size is the paper's pattern: small objects 0.329, medium 0.647, large 0.793. The paper reports 59.3 on LVIS, in percent. Our 13 pictures, our sparser grid and COCO's masks make the two numbers different experiments, so we do not compare them directly.

<figure><img src="/img/papers/sam/p5-proposals.png" alt="Two copies of one COCO photo. Left: the outlines of the objects that people drew, each in a colour. Right: for each object, the best proposal of SAM, filled in the same colour, with its IoU written on it" loading="lazy" /><figcaption>Left: the objects that people drew in one of our COCO pictures. Right: for each object, the best of SAM's proposals, with its IoU. SAM was told nothing about these objects; the grid of points found them.</figcaption></figure>

## Instance segmentation {§7.4}

> [!PAPER] · §7.4 · page 10
> [![Section 7.4: SAM is prompted with the output boxes of a detector; on mask AP it is reasonably close but behind ViTDet; its masks have crisper boundaries; in a human study SAM outperforms ViTDet](/img/papers/sam/p5-74.png)](/img/papers/sam/p5-74.png)
>
> [![Table 5: mask AP on COCO and LVIS v1, overall and by size, for ViTDet-H and SAM prompted with ViTDet boxes; the gap shrinks on the higher-quality LVIS masks](/img/papers/sam/p5-table5.png)](/img/papers/sam/p5-table5.png)
>
> **What it says:** the team runs a detector and will "prompt SAM with its output boxes". On mask AP, SAM is "reasonably close, though certainly behind ViTDet": 46.5 against 51.0 on COCO, and 44.7 against 46.6 on LVIS. Yet SAM's masks have "crisper boundaries", and "SAM consistently outperforms ViTDet in the human study". The caption adds that "the gap shrinks on the higher-quality LVIS masks".
>
> **What it means:**
>
> This is the composition of [Part 2](part-2-the-task.md): a detector finds each object and draws a box, and SAM turns each box into a mask. The detector is the same ViTDet as in §7.3. It also makes its own masks, so the comparison is exact: same boxes, two ways to draw the mask.
>
> The gap is 4.5 points on COCO but only 1.9 on LVIS. LVIS masks were drawn more carefully than COCO masks. The worse the reference masks, the more a model gains by copying their habits, and the more SAM loses by drawing what it sees.

> [!DEFINITION] Instance segmentation
> Find every object of known categories in a picture, and give each one its own mask and name. Two people get two masks. [Part 1](part-1-the-big-idea.md) compares it with semantic segmentation.

Mask AP is the standard score of instance segmentation. It uses the precision and recall of the edge section, but for objects. The detector gives each mask a confidence, and we sort all masks of the dataset by it.

Now walk down the list. A mask is a "hit" if its IoU with a true object of its category is at least $$t$$. That object must not have a hit yet. Otherwise it is a false alarm. At each step, compute precision and recall so far. The **average precision** is the area under the precision-recall curve:

$$
\text{AP}_t = \int_0^1 P_t(R) \, dR, \qquad \text{AP} = \frac{1}{10} \sum_{t \in \{0.50, \ldots, 0.95\}} \text{AP}_t
$$

$$P_t(R)$$ is the precision at recall $$R$$ with IoU threshold $$t$$. The integral is the area under that curve, a number from 0 to 1. COCO then averages over the same ten thresholds as AR, and over the categories, and reports it in percent. **AP$$^S$$**, **AP$$^M$$** and **AP$$^L$$** are the same score for small, medium and large objects only.

A tiny example at $$t = 0.5$$, with 2 true objects and 3 predicted masks in order of confidence. The first is a hit: precision 1/1, recall 1/2. The second is a false alarm: precision 1/2, recall 1/2. The third is a hit: precision 2/3, recall 2/2.

The usual rule takes, at each recall, the best precision at that recall or higher. So the curve is 1.0 up to recall 0.5, then 2/3 up to recall 1.0. The area is $$0.5 \times 1.0 + 0.5 \times 0.667 = 0.833$$.

The mask's IoU enters through $$t$$. At $$t = 0.95$$, a mask must match the reference almost pixel for pixel. If the reference has loose polygon edges, a crisp mask fails this test, and a mask that copies the loose style passes.

> [!PAPER] · Figure 11 · page 11
> [![Figure 11: histograms of mask quality ratings: LVIS ground truth 8.6, SAM 8.1, ViTDet-H 7.9, COCO ground truth 7.6](/img/papers/sam/p5-figure11.png)](/img/papers/sam/p5-figure11.png)
>
> [![The paragraph after Table 5: COCO ground truth quality is relatively low, SAM is unable to exploit these biases, and LVIS masks do not contain holes](/img/papers/sam/p5-74-biases.png)](/img/papers/sam/p5-74-biases.png)
>
> **What it says:** with LVIS ground-truth boxes, people rate SAM's masks "8.1 ± 0.07" and ViTDet's "7.9 ± 0.08". The LVIS masks get 8.6 and the COCO masks 7.6. "ViTDet exploits biases in the COCO and LVIS training data." On COCO, "ground truth quality is relatively low", and SAM is "unable to exploit these (generally undesirable) biases". LVIS masks also have habits: "masks do not contain holes".
>
> **What it means:**
>
> The surprising line is the last one in the legend. People rate COCO's own reference masks at 7.6, *below* both models. So on COCO, the "correct answer" in AP is often worse than SAM's answer. A model that agrees with it gets a better AP and a worse mask.
>
> "No holes" is an example of a habit. LVIS masks are simple polygons, so a plate with food on it is one solid shape. SAM sees the food on top and cuts it out. That is the next box.

> [!PAPER] · Figure 16 · page 22
> [![Figure 16: four examples from LVIS with the ground truth, ViTDet and SAM side by side: a table, a plate with orange slices, an airplane and an elephant; SAM's masks follow the edges more closely, and for the plate SAM cuts out the fruit](/img/papers/sam/p5-figure16.png)](/img/papers/sam/p5-figure16.png)
>
> [![Appendix D.4: an additional mask refinement iteration feeds the most confident mask with the box back to the decoder; SAM makes a valid modal prediction for the plate, which LVIS annotates amodally](/img/papers/sam/p5-d4.png)](/img/papers/sam/p5-d4.png)
>
> **What it says:** SAM gets "an additional mask refinement iteration": the most confident mask goes back to the decoder "together with the box prompt". For the plate, "SAM makes a valid modal prediction", whereas LVIS is "annotated amodally".
>
> **What it means:**
>
> Look at the elephant and the airplane in Figure 16. ViTDet's edges are wavy, as if drawn with a thick pen. SAM's edges follow the trunk and the wings. This is the "crisper boundaries" of §7.4, and the 256 × 256 masks of SAM's decoder help here.
>
> The refinement is the training game of Part 3 in one round. SAM draws a mask from the box, then gets its own mask back with the same box and draws again. It costs one more decoder pass, a fraction of a second.

> [!DEFINITION] Modal and amodal masks
> A **modal** mask covers only the visible pixels of an object. An **amodal** mask also covers the hidden parts, as you imagine them: the whole plate under the fruit. Both are valid; a dataset chooses one style.

We could not run ViTDet, so we took the true COCO boxes as prompts. The human study of Figure 11 also used true boxes, from LVIS. We scored SAM's masks with IoU against COCO's masks, before and after the refinement pass:

```text
mIoU with the box prompt: 0.808; after one refinement pass (mask + box back in): 0.823
refinement helped 26 objects, hurt 13
lowest box IoUs: #11 bed 0.003; #14 skis 0.528; #2 chair 0.651; #6 chair 0.660
```

A box is a much clearer prompt than a point. The mIoU rises from 0.601 with one center point to 0.808 with the box, and the refinement pass adds 0.015, to 0.823. It helped 26 objects and hurt 13: a small, real gain on average, not a gain on every object.

<figure><img src="/img/papers/sam/p5-boxes.png" alt="Three crops from COCO photos. Each has a green box prompt, SAM's mask in blue and the COCO mask as a red line" loading="lazy" /><figcaption>The true box (green) as the prompt. Blue: SAM's mask after the refinement pass. Red line: the mask in COCO. Left and middle: our two lowest IoUs. Right: an object with the median IoU.</figcaption></figure>

The lowest score is a lesson in ambiguity of a different kind. The "bed" in this COCO picture is almost hidden under teddy bears, so its box is full of bears. SAM gave this mask a low score, 0.569, and the refinement moved the mask further away (IoU 0.077, then 0.003). The skis are thin and cross each other, so a small shift at the edge costs much IoU. On the refrigerator, at the median, look at the red line: COCO's polygon cuts a straight edge, and SAM follows the real one.

## Text to mask {§7.5}

> [!PAPER] · §7.5 · page 11
> [![Section 7.5: a proof-of-concept text prompt; training with CLIP image embeddings of masks larger than 100 squared pixels, which needs no new text annotations; at inference, CLIP text embeddings are the prompt](/img/papers/sam/p5-75.png)](/img/papers/sam/p5-75.png)
>
> [![Figure 12: a car photo with masks from text prompts: a wheel and beaver tooth grille are correct; a wiper and wipers fail, and succeed with one added point](/img/papers/sam/p5-figure12.png)](/img/papers/sam/p5-figure12.png)
>
> **What it says:** this is "a proof-of-concept". The training changes "in a way that does not require new text annotations". For each mask larger than $$100^2$$ pixels, the team extracts "the CLIP image embedding" of the masked object. Because CLIP aligns pictures and text, "we can train with image embeddings, but use text embeddings for inference". In Figure 12, "a wheel" and "beaver tooth grille" work; "a wiper" fails, and "an additional point prompt can help".
>
> **What it means:**
>
> SA-1B has masks but no words. How can SAM learn text without text? The trick uses CLIP, which places a picture and its caption at almost the same point in one shared space. During training, SAM's prompt is the CLIP vector of the *picture* of the object. At test time, the prompt is the CLIP vector of a *sentence*. If CLIP's alignment is good, the two are close enough, and SAM never notices the swap.
>
> Appendix D.5 gives the recipe. It uses the largest public CLIP model (ViT-L/14 at 336 pixels). The crops around each object are enlarged 1 to 2 times, and half of the time the pixels outside the mask are set to zero. "Proof-of-concept" is honest: the paper shows pictures, not a score.

This is the one experiment that we cannot repeat. The released SAM weights have no text prompt: the model was trained without this CLIP path, and the text-trained version was not published. So we show no text result. Part 6 returns to text as one of SAM's limits.

## Ablations {§7.6}

> [!DEFINITION] Ablation
> An experiment that removes or changes one part of a system and measures the effect. If the score drops, that part mattered. The word comes from surgery: the removal of tissue.

> [!PAPER] · §7.6 · page 11
> [![The first paragraphs of 7.6: ablations on the 23 datasets with one center point and an oracle; each data engine stage increases mIoU; manual and semi-automatic masks oversampled 10 times; automatic masks only are marginally lower, about 0.5 mIoU](/img/papers/sam/p5-76-stages.png)](/img/papers/sam/p5-76-stages.png)
>
> [![The data volume paragraph: at 0.1M images a large mIoU decline, at 1M images, about 10% of the dataset, results comparable to the full dataset, still about 100M masks](/img/papers/sam/p5-76-volume.png)](/img/papers/sam/p5-76-volume.png)
>
> **What it says:** "each stage increases mIoU". With all three stages, the team found that "oversampling the manual and semi-automatic masks" 10 times worked best. But automatic masks alone are "only marginally lower than using all data (∼0.5 mIoU)", so SAM trains on automatic masks only. "At 0.1M images, we observe a large mIoU decline", but "with 1M images, about 10% of the full dataset", the results are comparable, with "approximately 100M" masks.
>
> **What it means:**
>
> These are tests of Part 4's data engine. Each stage adds data, and each stage helps. The surprise is the last bar: the masks that SAM made by itself are almost as good a teacher as all the masks together. The team chose the simpler recipe, and gave up about half a point.
>
> The data-volume result is practical news. You do not need all 11 million pictures to train a good SAM. One million pictures, still about 100 million masks, gets close. One hundred thousand pictures is too few.

> [!PAPER] · Figure 13 · page 12
> [![Figure 13: three plots of mIoU on the 23 datasets: training data stages (manual, plus semi-automatic, plus automatic, automatic only), training images (0.1M, 1M, 11M) and image encoder size (ViT-B 91M, ViT-L 308M, ViT-H 636M parameters)](/img/papers/sam/p5-figure13.png)](/img/papers/sam/p5-figure13.png)
>
> [![The last paragraph of 7.6: ViT-H improves substantially over ViT-B but has only marginal gains over ViT-L; further image encoder scaling does not appear fruitful](/img/papers/sam/p5-76-encoder.png)](/img/papers/sam/p5-76-encoder.png)
>
> **What it says:** the encoders have "91M", "308M" and "636M" parameters. "ViT-H improves substantially over ViT-B, but has only marginal gains over ViT-L." The caption calls the gains "meaningful, yet saturating", and "further image encoder scaling does not appear fruitful at this time".
>
> **What it means:**
>
> The paper gives these three plots without a table, so we read the trends, not exact values. In the right plot, the one-point line climbs from ViT-B to ViT-L, then flattens to ViT-H. The oracle line does the same. Doubling the encoder from ViT-L to ViT-H buys very little.
>
> "Saturating" is the important word. In language models of the same years, a bigger model was nearly always better. Here, the encoder seems large enough, and the limit is somewhere else: in the data, the decoder, or the ambiguity of the task itself.

We ran our 40-object test with all three released encoders:

```text
ViT-B: 1 point 0.505, oracle 0.746, 3 points 0.819, 5 points 0.797
ViT-L: 1 point 0.555, oracle 0.791, 3 points 0.825, 5 points 0.821
ViT-H: 1 point 0.601, oracle 0.789, 3 points 0.786, 5 points 0.804
```

{{FIG:p5_encoders|Our mini test with the three released image encoders, one center point on our 40 COCO objects. Orange: the top-scored mask. Green: the oracle.}}

The oracle column repeats the paper's shape. It rises from ViT-B to ViT-L (0.746 to 0.791) and then stays flat (0.789 for ViT-H). So the larger encoders already *have* the right mask. The top-scored column keeps rising (0.505, 0.555, 0.601): on our sample, the larger encoder ranks its three masks better. With three or five points, the order is mixed, and the differences are smaller than the noise of 40 objects. The paper's 23 datasets settle what our sample cannot.

> [!TAKEAWAYS] Key takeaways
> - **Zero-shot means frozen.** One SAM model, trained only on SA-1B, faces 23 datasets and four new tasks. Only the prompts change.
> - **mIoU with one center point.** The point is the farthest from the mask's edge, found by a distance transform. SAM beats RITM on 16 of 23 datasets, by up to 46.9 IoU.
> - **The oracle shows the cost of ambiguity.** With the best of its three masks, SAM beats RITM on all 23 datasets. On our 40 COCO objects, the oracle lifts the mIoU from 0.601 to 0.789.
> - **People like SAM's masks.** On a 1 to 10 scale, SAM's mean ratings are 7.1 to 8.6, above RITM on every dataset, even where mIoU says otherwise.
> - **Edges from masks.** Sobel filters on SAM's mask probability maps give ODS .768 on BSDS500. The Sobel filter on the photo gets .539, and HED gets .788.
> - **Proposals: AR@1000 59.3 against 63.0** for a detector trained on LVIS. SAM wins on medium, large, common and rare objects.
> - **Instance segmentation: lower AP, higher ratings.** SAM's AP is 46.5 against 51.0 on COCO. But people rate its masks higher (8.1 against 7.9), because AP rewards the habits of the reference masks.
> - **Text worked as a proof of concept** and was not released, so we could not test it.
> - **Ablations:** each data-engine stage helps; automatic masks alone lose about 0.5 mIoU; 1M pictures are almost as good as 11M; ViT-H gains little over ViT-L.

**Next, in Part 6:** the paper's discussion and limits. What SAM changed, where it fails, and what came after it.

<details>
<summary>Run it yourself</summary>

The script [`code/papers/sam/sam_part5.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/sam/sam_part5.py) reads 16 COCO pictures from Hugging Face. It places the center point of each of 40 objects by a distance transform and scores SAM's top-scored and oracle masks. It adds points by the error-region rule, prompts with the true boxes, makes proposals from a grid of points and computes the average recall. It also draws edge maps from SAM's masks, and repeats the point test with ViT-B and ViT-L. Every number of this part is in `results/part5.json` and `results/part5_stdout.txt`.

```bash
pip install torch transformers datasets pillow scipy opencv-python
python sam_part5.py        # uses facebook/sam-vit-base, -large and -huge, and about 140 MB of COCO data
```

<figure><img src="/img/papers/sam/part5-run.png" alt="Terminal output of sam_part5.py: the sample of 40 COCO objects and, for each object, the center point, the IoU of the three masks, the top-scored and oracle mask, and the IoU after 1 to 9 points" loading="lazy" /><figcaption>The first part of the output of sam_part5.py, as it ran for this article.</figcaption></figure>

</details>

## The paper and the code

[Segment Anything](https://arxiv.org/abs/2304.02643) (ICCV 2023) · [code and weights](https://github.com/facebookresearch/segment-anything) · [our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/sam)
