---
title: "After SAM: Impact, Limits and Summary"
description: "Where SAM fails, measured with the released model on a bicycle, a drawn wheel and an edge with teeth. Then five later papers that fixed those limits, and the whole paper in one page."
part: 6
covers: "§8, Conclusion, and five later papers: SAM 2, MobileSAM, HQ-SAM, MedSAM, SAM 3"
date: 2026-10-17
tags: [sam, segmentation, limitations, sam-2, computer-vision]
---

The last page of the paper is honest: it lists what SAM cannot do. We test each limit on the released model, then follow five later papers that took up those limits one by one.

## Foundation models, once more {§8}

> [!PAPER] · §8 · page 12
> [![The Foundation models paragraph of Section 8, with the definition trained on broad data at scale and adaptable to a wide range of downstream tasks, inherently limited scope, and the vast majority of its capabilities come from large-scale supervised training highlighted](/img/papers/sam/p6-foundation.png)](/img/papers/sam/p6-foundation.png)
>
> **What it says:** a foundation model is "trained on broad data at scale and are adaptable to a wide range of downstream tasks". SAM fits, but segmentation is "an inherently limited scope". Also, "the vast majority of its capabilities come from large-scale supervised training", not from self-supervised learning.
>
> **What it means:**
>
> Part 1 defined a foundation model with the same words. Here the authors check SAM against the definition, and they add two careful notes.
>
> The first note is modesty. A language model can write, translate and answer questions. SAM only cuts objects out of pictures. It is a foundation for one floor of the building, not for the whole building.
>
> The second note is about how SAM learned. The usual story says that foundation models learn from raw data with no labels. SAM starts that way (the MAE pre-training of Part 3), but most of its skill comes from 1.1 billion masks with a correct answer. The data engine of [Part 4](part-4-data-engine-and-sa-1b.md) made those labels cheap, so labelled training became possible at scale again.

> [!DEFINITION] Supervised and self-supervised learning
> In **supervised** learning, each training example comes with the correct answer, for example a photo and its mask. In **self-supervised** learning, the data makes its own questions, for example "fill in the hidden patches" (MAE). No person labels anything.

## Composition: SAM as a part of a bigger system {§8}

> [!PAPER] · §8 · page 12
> [![The Compositionality paragraph of Section 8, with make this kind of composition straightforward, reliable interface between SAM and other components, prompted with gaze points detected by a wearable device, and without need for additional training highlighted](/img/papers/sam/p6-composition.png)](/img/papers/sam/p6-composition.png)
>
> **What it says:** the goal is to "make this kind of composition straightforward with SAM". A valid mask for any prompt creates "a reliable interface between SAM and other components". For example, SAM can be "prompted with gaze points detected by a wearable device", and such systems work "without need for additional training".
>
> **What it means:**
>
> Think of a power socket. Any lamp, phone charger or kettle works with it, because the socket keeps one promise: the same voltage, the same shape. SAM's promise is "give me a point or a box, and I give you a valid mask". Any other program that can produce a point can use SAM.
>
> The gaze example shows this well. Smart glasses track where the wearer looks. That point goes to SAM as a prompt, and out comes the object the person looks at. Nobody trained SAM for glasses. The prompt is the only connection. [Part 2](part-2-the-task.md) explained why the task was chosen to make this possible.

The five later papers below are also compositions, in a way. Each one keeps most of SAM and changes one piece.

## The limits {§8}

> [!PAPER] · §8 · page 12
> [![The start of the Limitations paragraph: it can miss fine structures, hallucinates small disconnected components at times, and does not produce boundaries as crisply as more computationally intensive methods that zoom-in, highlighted](/img/papers/sam/p6-limits-1.png)](/img/papers/sam/p6-limits-1.png)
>
> **What it says:** "While SAM performs well in general, it is not perfect. It can miss fine structures, hallucinates small disconnected components at times, and does not produce boundaries as crisply as more computationally intensive methods that 'zoom-in'."
>
> **What it means:**
>
> Three faults in one sentence. "Fine structures" are thin parts: spokes, wires, legs of an insect, strands of hair. "Hallucinates" means the mask contains small islands of pixels where no object is. "Boundaries as crisply" is about the exact line of the edge.
>
> The last phrase names the cure. A "zoom-in" method looks at the picture once and finds the rough edge. Then it cuts out a small window around the edge at high resolution and looks again. That costs more computation, and SAM does not do it. SAM looks once.

All three faults have one cause that we met in [Part 3](part-3-the-model.md). The mask decoder predicts the mask on a grid of 256 × 256 cells, and the final mask is that grid, enlarged by interpolation. For a 1024 × 1024 picture, one cell covers 4 × 4 pixels. Anything thinner than a cell has no cell of its own.

> [!DEFINITION] Interpolation
> A way to fill the pixels between known values. **Bilinear** interpolation gives each new pixel a weighted average of the four nearest known values, with more weight on the nearer ones. It makes smooth slopes, never new detail.

We tested the three faults with the released ViT-H and ViT-B models.

### Fault 1: thin structures

A bicycle is the classic test: a wheel is mostly empty space with thin spokes. We took a photo from the COCO dataset. SAM got two prompts: a box around the whole bicycle, and one point on the rear tyre.

```text
ViT-H: box [195, 8, 497, 388] -> 1 mask, predicted IoU 0.972, 61,124 pixels
ViT-H: point (482, 270) on the rear tyre -> mask 2 (the wheel), predicted IoU 1.003, 34,976 pixels; holes inside it: 6 (77 pixels)
```

<figure><img src="/img/papers/sam/p6-bike.png" alt="Two copies of a photo of a blue bicycle on a city sidewalk, both overlaid with a pink mask. Left: a box around the bicycle; the mask covers the frame and both wheels as solid discs, including the road visible between the spokes. Right: one green point on the rear tyre; the mask is the whole rear wheel as a solid disc" loading="lazy" /><figcaption>SAM ViT-H on a COCO photo. Both masks paint each wheel as a solid disc: the road seen between the spokes becomes "bicycle". The wheel mask has only 6 tiny holes (77 pixels in total).</figcaption></figure>

The predicted IoU is 0.972 for the box and 1.003 for the wheel: SAM is very sure. But look at the wheels. The road between the spokes is pink. To SAM, the wheel is a disc.

A photo cannot tell us how wrong this is, because nobody drew the exact correct mask for every spoke. So we drew a wheel ourselves, where we know every pixel. The picture is 1024 × 1024 pixels with a dark rim, a hub and 24 spokes. We drew it eight times, with spokes 1 to 16 pixels wide, and gave SAM a box around the wheel each time.

Two numbers grade each mask. **Spoke recall** is the share of spoke pixels that the mask covers. **Gap fill** is the share of the empty pixels between the spokes that the mask covers. The correct mask has a spoke recall of 1 and a gap fill of 0.

> [!DEFINITION] Recall
> Of all the pixels that should be in the mask, the share that really is in it. A recall of 1 means nothing was missed. Recall does not care about extra pixels; that is why we also measure the gap fill.

```text
model   spoke width  spoke px   score  spoke recall  gap fill    IoU  best of 3 IoU
ViT-H          1 px     7,907   0.917         0.000     0.000  0.835          0.839
ViT-H          2 px    16,188   0.733         0.208     0.014  0.745          0.771
ViT-H          3 px    26,062   0.756         0.620     0.534  0.263          0.709
ViT-H          4 px    34,234   0.893         0.988     0.991  0.193          0.656
ViT-H          6 px    52,289   0.989         1.000     1.000  0.222          0.569
ViT-H          8 px    69,115   0.994         1.000     1.000  0.249          0.586
ViT-B          3 px    26,062   0.871         0.506     0.046  0.681          0.708
ViT-B          4 px    34,234   0.915         0.888     0.402  0.358          0.681
ViT-H: widths with spoke recall > 0.9 and gap fill < 0.1 (the correct shape): none
ViT-B: widths with spoke recall > 0.9 and gap fill < 0.1 (the correct shape): none
```

<figure><img src="/img/papers/sam/p6-wheels.png" alt="Ten drawn wheels in two rows, ViT-H on top and ViT-B below, with spokes 1, 2, 3, 4 and 8 pixels wide. With 1 and 2 pixel spokes only the rim is pink. With 3 and 4 pixel spokes the inside is partly pink in a blotchy pattern. With 8 pixel spokes the whole disc is pink" loading="lazy" /><figcaption>The drawn wheel. Thin spokes vanish (only the rim is masked). Thick spokes turn the wheel into a solid disc. In between, the mask is a blotchy mix. No width gives spokes without gaps.</figcaption></figure>

{{FIG:p6_wheel|Spoke recall and gap fill against the spoke width, for ViT-H and ViT-B. All four lines stay near 0 for thin spokes and jump to near 1 between 3 and 8 pixels. The correct answer (recall 1, gap fill 0) never appears.}}

The result has two sides, and both are wrong. With spokes of 1 or 2 pixels, SAM does not see them: recall 0.000 and 0.208. With spokes of 6 pixels or more, SAM sees them and fills the whole disc: gap fill 1.000. The switch happens at 3 to 4 pixels, which is one cell of the 256 × 256 grid.

That is the grid at work. A gap between two spokes is wide near the rim and narrow near the hub. Near the hub, a cell holds spoke and gap together, and it can only say "in" or "out". The decoder learned from SA-1B that a wheel is usually one object, so it says "in" everywhere.

The IoU column hides all this. For 1-pixel spokes, the IoU is 0.835, a fine score, because the rim and the hub are most of the pixels. A good average can hide a total failure on the thin parts. The column "best of 3" gives the best of the three ambiguity masks; none of them is correct either.

ViT-B is closer at 3 pixels: recall 0.506 with a gap fill of only 0.046. That is not a general win for the small model. It only shows that the two models switch at slightly different widths.

What this test cannot show: a drawn wheel on a flat grey ground is not a photo. Real spokes have shadows, blur and a busy background. But the bicycle photo shows the same solid disc.

### Fault 2: small islands

The second fault is "small disconnected components". A **connected component** is a group of mask pixels that touch each other ([Part 4](part-4-data-engine-and-sa-1b.md) defines it fully). A mask of one cat should be one component. Extra small islands are pixels that SAM marks as "object" far from the object.

We clicked a regular grid of 16 × 16 = 256 points on each of two photos, the bicycle and the two cats of Part 1. For each click we kept the mask with the highest predicted IoU and counted its components. An island under 100 pixels that is not the main piece is a "stray piece".

```text
bike ViT-H: 256 prompts, 152 masks with more than one piece (59%), 145 with stray pieces under 100 pixels (57%), 1051 such pieces in total, median size 5 pixels
cats ViT-H: 256 prompts, 154 masks with more than one piece (60%), 87 with stray pieces under 100 pixels (34%), 1195 such pieces in total, median size 4 pixels
cats ViT-B: 256 prompts, 210 masks with more than one piece (82%), 209 with stray pieces under 100 pixels (82%), 9212 such pieces in total, median size 4 pixels
```

<figure><img src="/img/papers/sam/p6-specks.png" alt="Two photos with pink masks and yellow rings. Top: the cats photo, a click on the blanket at the left; the blanket mask is pink, and a dense row of yellow rings marks hundreds of tiny islands on the red cloth at the top. Bottom: the bicycle photo, a click on the sidewalk; the sidewalk mask is pink, and many yellow rings mark small islands on the bricks around the tree" loading="lazy" /><figcaption>Stray pieces in confident ViT-H masks. Top: a click on the blanket gives a mask with 395 islands under 100 pixels on the red cloth (predicted IoU 0.959). Bottom: a click on the sidewalk gives 57 islands on the bricks (predicted IoU 0.924).</figcaption></figure>

The pictures show where the islands come from. They sit on textures that look a little like the clicked surface: the red cloth beside the blanket, the bricks beside the sidewalk. The logit there is just above 0, and the threshold turns each lucky pixel into an island. The median island has 4 or 5 pixels.

The paper knew this. Appendix B estimates that "4% of masks include small, spurious components" in the automatic stage of the data engine. It removes islands under 100 pixels and fills holes under 100 pixels. It also keeps only masks with a predicted IoU above 0.88 and a high **stability score** (Part 4 explains both). We applied the two filters to our masks:

```text
bike ViT-H: only masks with predicted IoU > 0.88: 231 masks, 125 with stray pieces under 100 pixels (54%)
   and also stable (IoU of the masks at logit +1 and -1 >= 0.95): 153 masks, 57 with stray pieces under 100 pixels (37%)
cats ViT-B: only masks with predicted IoU > 0.88: 224 masks, 182 with stray pieces under 100 pixels (81%)
   and also stable (IoU of the masks at logit +1 and -1 >= 0.95): 37 masks, 3 with stray pieces under 100 pixels (8%)
```

{{FIG:p6_specks|The share of masks with stray pieces under 100 pixels, for both photos and both models: all masks, then masks with predicted IoU above 0.88, then masks that are also stable. The filters lower the share, most for ViT-B, but they never remove it.}}

The stability filter helps most, because an island of 4 pixels with a logit near 0 is exactly what "unstable" means. For ViT-B on the cats it removes most masks: 37 of 224 survive. The predicted IoU alone barely helps. SAM gives a mask full of islands a score of 0.959.

Our 37% to 8% is far above the paper's 4%, and the two numbers are not comparable. The paper counted on the original photos, about 1500 pixels on the short side, with a special SAM trained for automatic generation. We counted on photos of 640 pixels, with raw single clicks and no crops. The lesson is the same: islands are common in raw masks, and a short clean-up step removes them.

### Fault 3: edges that are not crisp

The third fault is the exact line of the edge. We zoomed into the tyre of the bicycle, 64 × 64 pixels, and drew four views of the same place:

<figure><img src="/img/papers/sam/p6-boundary.png" alt="Four zoomed views of the same 64 by 64 pixel patch at the edge of the bicycle tyre. First: the photo, dark tyre with knobbly tread on a grey road. Second: the 256 by 256 low-resolution mask as square blocks. Third: the final mask, almost the same shape with slightly smoother steps. Fourth: the probability map, white inside and black outside, with a thin grey edge" loading="lazy" /><figcaption>The tyre edge in four views. The final mask (third) is the 256 × 256 grid (second) with its steps smoothed; it adds no new detail. The probability (fourth) is uncertain only in a thin band at the edge.</figcaption></figure>

```text
ViT-H: low-resolution logits (256, 256); one cell covers 2.50 x 2.50 photo pixels
ViT-H: wheel mask edge 1,060 pixels long; pixels with probability between 0.1 and 0.9: 3,768; mean width of the soft band 3.55 pixels
ViT-H: the final mask and the 256 x 256 cell mask (blown up without interpolation) differ in 648 pixels, 1.9% of the mask area
```

The photo is 640 pixels wide, so one cell covers 2.5 × 2.5 photo pixels. The final mask and the blocky cell mask differ in only 1.9% of the mask's pixels. Interpolation rounds the corners of the blocks, but it cannot add a knob of the tread that the grid did not have.

The "soft band" is the strip where SAM is unsure: the probability is between 0.1 and 0.9. We divide its area by the length of the edge to get its average width. It is 3.55 pixels for ViT-H and 8.54 for ViT-B. Inside that band, the exact position of the line is a guess.

> [!DEFINITION] Logit and probability
> The decoder outputs a **logit** for each pixel: any number, positive for "in" and negative for "out" ([Part 2](part-2-the-task.md)). The sigmoid function turns it into a **probability** between 0 and 1. A logit of 0 is a probability of 0.5, the threshold of the mask.

To measure crispness with a known answer, we drew a dark block with teeth on its top edge, 16 pixels high. The teeth and the gaps between them have the same width $$t$$. We graded tooth recall and gap fill, as for the spokes:

```text
model       t  tooth recall  gap fill   edge error (no teeth)
ViT-H    none                                         2.27 px
ViT-H     1 px         0.968     0.968
ViT-H     2 px         0.986     0.989
ViT-H     4 px         0.993     0.985
ViT-H     8 px         0.916     0.469
ViT-H    16 px         0.384     0.011
ViT-H    32 px         0.807     0.004
```

<figure><img src="/img/papers/sam/p6-teeth.png" alt="Four zoomed crops of a dark block with teeth on top, overlaid with the ViT-H mask in pink. Teeth of 2 and 4 pixels: the mask is a straight band that covers teeth and gaps alike. Teeth of 8 pixels: the mask follows the teeth with rounded corners and partly fills the gaps. Teeth of 16 pixels: the mask covers only a low rounded bump inside each tooth" loading="lazy" /><figcaption>The edge with teeth, ViT-H. Up to 4 pixels the mask draws a straight line over the tips. At 8 pixels it starts to follow the teeth, with round corners. At 16 pixels it cuts the teeth down to rounded bumps.</figcaption></figure>

{{FIG:p6_teeth|Tooth recall and gap fill against the tooth width. Up to 4 pixels both are near 1: the mask is a straight line over the tips. At 8 pixels the mask starts to follow the teeth. At 16 pixels it keeps only 38% of the tooth pixels (ViT-H).}}

Up to 4 pixels, the mask is a straight line over the tips: recall and gap fill both near 1. At 8 pixels, two cells per tooth, the mask starts to follow the teeth. Even at 32 pixels, the mask misses 19% of the tooth pixels, mostly at the sharp corners. Even a straight edge with no teeth is off by 2.27 pixels on average.

A zoom-in method would fix this. It would cut a window around the edge from the full photo and run a second, sharper pass on it. SAM chose one pass for speed.

## Many clicks, and the heavy encoder {§8}

> [!PAPER] · §8 · page 12
> [![The middle of the Limitations paragraph: dedicated interactive methods outperform SAM when many points are provided; SAM is designed for generality and breadth of use rather than high IoU interactive segmentation](/img/papers/sam/p6-limits-2.png)](/img/papers/sam/p6-limits-2.png)
>
> **What it says:** "we expect dedicated interactive segmentation methods to outperform SAM when many points are provided". SAM "is designed for generality and breadth of use rather than high IoU interactive segmentation".
>
> **What it means:**
>
> SAM is a pocket knife. It cuts many things well enough, the first time. A tool built for one job, such as the SimpleClick model the paper cites, is a scalpel. After many careful clicks on one object, the scalpel wins.
>
> [Part 5](part-5-zero-shot-experiments.md) showed this in Figure 9: as the points go from 1 to 9, "the gap between methods decreases". Our three faults explain part of the reason. More clicks cannot add detail that the 256 × 256 grid does not have.

> [!PAPER] · §8 · page 12
> [![The sentence: SAM can process prompts in real-time, but nevertheless SAM's overall performance is not real-time when using a heavy image encoder](/img/papers/sam/p6-limits-3.png)](/img/papers/sam/p6-limits-3.png)
>
> **What it says:** "SAM can process prompts in real-time, but nevertheless SAM's overall performance is not real-time when using a heavy image encoder."
>
> **What it means:**
>
> Part 1 called SAM's speed "amortized": a slow encoder once per picture, then a fast decoder per click. That works for one photo and many clicks. It fails for video, where every frame is a new picture, and for a phone, which cannot run 637 million weights quickly.

Our script counted the weights of the two released models:

```text
ViT-H: 641,090,864 weights, of which image encoder 637,026,048, prompt encoder + mask decoder 4,064,816
ViT-B: 93,735,728 weights, of which image encoder 89,670,912, prompt encoder + mask decoder 4,064,816
ViT-H has 7.1 x the encoder weights of ViT-B
```

The decoder is the same size in both. Only the encoder changes. So does the small encoder give the same masks? We compared ViT-B with ViT-H on the same 256 clicks per photo, with ViT-H's top mask as the reference:

```text
bike, ViT-B top mask: 256 prompts, mean IoU with the ViT-H top mask 0.753, median 0.915, 54% above 0.9, 18% below 0.5
bike, closest of 3 ViT-B masks: 256 prompts, mean IoU with the ViT-H top mask 0.884, median 0.944, 69% above 0.9, 5% below 0.5
cats, ViT-B top mask: 256 prompts, mean IoU with the ViT-H top mask 0.810, median 0.936, 58% above 0.9, 11% below 0.5
```

The median is high: the typical click gets nearly the same mask. The mean is lower, because 18% of the bicycle clicks get a very different mask. Most of those are ambiguity, not failure. When we let ViT-B pick the closest of its three masks, only 5% stay below 0.5. The small model often finds the same object, but ranks the part and the whole in another order.

[Part 5](part-5-zero-shot-experiments.md) reads Figure 13 of the paper: ViT-H gains a lot over ViT-B, but little over ViT-L. Its caption adds that "smaller image encoders may be preferred in certain settings". Those settings are the phone and the video, and two later papers took them up.

## Text, semantics and special domains {§8}

> [!PAPER] · §8 · page 12
> [![The end of the Limitations paragraph: the text-to-mask task is exploratory and not entirely robust; it is unclear how to design simple prompts that implement semantic and panoptic segmentation; domain-specific tools are expected to outperform SAM in their domains](/img/papers/sam/p6-limits-4.png)](/img/papers/sam/p6-limits-4.png)
>
> **What it says:** the text-to-mask work "is exploratory and not entirely robust". "It is unclear how to design simple prompts that implement semantic and panoptic segmentation." And there are "domain-specific tools", such as a tool for microscopy, "that we expect to outperform SAM in their respective domains".
>
> **What it means:**
>
> Three more limits. The text prompt worked as a proof of concept in Part 5, and it was not released. Semantic segmentation needs a class name for every pixel, but SAM never names anything: a prompt asks for "the thing here", not "all the sky".
>
> The last limit is the honest one for science. A tool trained for years on cell pictures knows things about cells that SA-1B never showed. SAM is a generalist, and a generalist loses to a specialist on the specialist's home ground.

> [!DEFINITION] Panoptic segmentation
> Semantic and instance segmentation together. Every pixel gets a class, and each countable object ("thing") also gets its own mask. The sky and the road get one region each; each car gets its own mask. Part 1 defined the two parts.

## The conclusion {§8}

> [!PAPER] · §8 · page 12
> [![The Conclusion paragraph: an attempt to lift image segmentation into the era of foundation models; a new task (promptable segmentation), model (SAM), and dataset (SA-1B); whether SAM achieves the status of a foundation model remains to be seen by how it is used in the community](/img/papers/sam/p6-conclusion.png)](/img/papers/sam/p6-conclusion.png)
>
> **What it says:** the project "is an attempt to lift image segmentation into the era of foundation models". Its contributions are "a new task (promptable segmentation), model (SAM), and dataset (SA-1B)". Whether SAM becomes a foundation model "remains to be seen by how it is used in the community".
>
> **What it means:**
>
> The authors do not claim victory. A foundation model is defined by what others build on it, and in 2023 nobody had built anything yet. The test is the use, not the score.
>
> So the fair way to end this series is to look at what came next. We picked five papers. Each one takes one limit of Section 8 and answers it, and each keeps a large part of SAM.

{{FIG:p6_family|Five limits of Section 8 and five later papers. SAM 2 adds a memory for video. MobileSAM trains a small encoder to copy ViT-H. HQ-SAM adds one output token for sharper masks. MedSAM fine-tunes SAM on medical images. SAM 3 finds all instances of a noun phrase.}}

## After SAM: five papers

### Video: SAM 2

> [!PAPER] Ravi et al. (2024), SAM 2 · Abstract · page 1
> [![The abstract of SAM 2: a foundation model towards solving promptable visual segmentation in images and videos, a simple transformer architecture with streaming memory for real-time video processing, and in image segmentation more accurate and 6x faster than SAM](/img/papers/sam/p6-sam2-abstract.png)](/img/papers/sam/p6-sam2-abstract.png)
>
> **What it says:** SAM 2 is "a foundation model towards solving promptable visual segmentation in images and videos". It is "a simple transformer architecture with streaming memory for real-time video processing". On images, it is "more accurate and 6× faster than the Segment Anything Model (SAM)".
>
> **What it means:**
>
> A video is a stack of pictures. SAM could cut an object out of each frame, but it would need a new click on every frame. It would also forget which object you meant. SAM 2 adds memory: you click once, and the model follows the object through the video.
>
> Notice the same recipe as SAM: a task (prompts on any frame), a model, and a data engine that built a large video dataset. The SAM 2 paper says that a single image is treated as "a single-frame video". So SAM 2 also replaces SAM on photos.

> [!PAPER] Ravi et al. (2024), SAM 2 · §4 · page 5
> [![The Memory bank paragraph of SAM 2: a FIFO queue of memories of up to N recent frames, up to M prompted frames, and object pointers](/img/papers/sam/p6-sam2-memory.png)](/img/papers/sam/p6-sam2-memory.png)
>
> **What it says:** the memory bank keeps "a FIFO queue of memories of up to N recent frames" and a queue of "up to M prompted frames". It also stores "object pointers", small vectors that summarise the object.
>
> **What it means:**
>
> Think of a guard who watches one person in a crowd. The guard remembers the last few seconds (the recent frames) and the photo on the poster (the frames where you clicked). On each new frame, the model looks at the new picture and asks the memory: "which of these regions looks like what I followed so far?"
>
> SAM's decoder already used cross-attention to read the image embedding (Part 3). SAM 2 adds memory attention before the decoder, so the frame's embedding first reads the memory. The same paper adds a head that predicts whether the object is visible at all. In a video, the object can walk out of the picture.

> [!DEFINITION] FIFO queue
> "First in, first out": a list with a fixed size. A new item enters at one end, and when the list is full, the oldest item leaves at the other end. It keeps the memory to the most recent frames.

> [!PAPER] Ravi et al. (2024), SAM 2 · §6.3 · page 10
> [![SAM 2, Section 6.3: on the 23 datasets of SAM, SAM 2 gets 58.9 mIoU with 1 click against 58.1 for SAM, mainly because of the smaller but more effective Hiera image encoder](/img/papers/sam/p6-sam2-image.png)](/img/papers/sam/p6-sam2-image.png)
>
> **What it says:** on the 23 datasets of SAM, SAM 2 reaches "58.9 mIoU with 1 click" against "58.1 mIoU with 1 click" for SAM, with no extra data and 6 times faster. The gain is "mainly attributed to the smaller but more effective Hiera image encoder".
>
> **What it means:**
>
> This is the same exam as Part 5, so the numbers compare directly. The accuracy gain on photos is small: 0.8 points of mIoU. The speed gain is large. Table 5 of SAM 2 gives 21.7 frames per second for SAM and 130.1 for SAM 2, on one A100 GPU.
>
> The lesson matches Figure 13 of SAM: a bigger encoder was not the way forward. A better-designed encoder was. Hiera is a hierarchical encoder: it works at several resolutions, and the decoder gets its sharper early features.

### A small encoder: MobileSAM

> [!PAPER] Zhang et al. (2023), MobileSAM · Abstract · page 1
> [![The MobileSAM abstract: replacing the heavyweight image encoder with a lightweight one, decoupled distillation, training on a single GPU within less than one day, more than 60 times smaller](/img/papers/sam/p6-mobilesam-abstract.png)](/img/papers/sam/p6-mobilesam-abstract.png)
>
> **What it says:** MobileSAM makes SAM mobile-friendly "by replacing the heavyweight image encoder with a lightweight one". The method is "decoupled distillation". Training takes a "single GPU within less than one day", and the model is "more than 60 times smaller".
>
> **What it means:**
>
> Our weight count showed that 99% of SAM is the encoder. So MobileSAM keeps SAM's decoder and changes only the encoder. The trick is in how the small encoder learns.
>
> It does not learn from masks. It learns to copy ViT-H's embedding directly. Show both encoders the same photo. Make the small one's 256 × 64 × 64 output as close as possible to the big one's. If the copy is good, SAM's own decoder cannot tell the difference.

> [!DEFINITION] Knowledge distillation
> Training a small model (the student) to copy the outputs of a large trained model (the teacher). The student learns from the teacher's answers, not from human labels. MobileSAM's student copies the image embedding, with a mean squared error loss.

> [!PAPER] Zhang et al. (2023), MobileSAM · Table 3 · page 5
> [![Table 3 of MobileSAM: the image encoder of the original SAM has 632M parameters and takes 452ms; the MobileSAM encoder has 5.78M parameters and takes 8ms, on a single GPU](/img/papers/sam/p6-mobilesam-table.png)](/img/papers/sam/p6-mobilesam-table.png)
>
> **What it says:** the image encoder of the original SAM has "632M" parameters and takes "452ms" on one GPU. MobileSAM's encoder has "5.78M" parameters and takes "8ms".
>
> **What it means:**
>
> 632 divided by 5.78 is about 109: the encoder is over 100 times smaller. The time drops by a factor of about 56. With 8 ms for the encoder, the whole model fits in a video frame. The "not real-time" limit of Section 8 is gone for this setting.

> [!PAPER] Zhang et al. (2023), MobileSAM · §4.1 · page 6
> [![MobileSAM, Section 4.1: we compute the mIoU between the masks predicted by the original SAM and our MobileSAM](/img/papers/sam/p6-mobilesam-eval.png)](/img/papers/sam/p6-mobilesam-eval.png)
>
> **What it says:** "we compute the mIoU between the masks predicted by the original SAM and our MobileSAM".
>
> **What it means:**
>
> Read this sentence before you read any claim of "on par". The reference is not a human mask. It is SAM's own mask. A perfect score would mean "exactly as good as SAM, including SAM's mistakes". The MobileSAM paper reports values from 0.706 to 0.745 in its ablation (Table 4).
>
> Our ViT-B against ViT-H test above used the same kind of measure. It gave a median of 0.915 on the bicycle, but a mean of 0.753 because of ambiguity. A single average can hide very different situations, so a careful reader asks what the reference was.

### Sharper masks: HQ-SAM

> [!PAPER] Ke et al. (2023), HQ-SAM · §1 · page 1
> [![The HQ-SAM introduction: SAM suffers from two key problems, coarse mask boundaries, often even neglecting the segmentation of thin object structures, and incorrect predictions, broken masks](/img/papers/sam/p6-hq-problem.png)](/img/papers/sam/p6-hq-problem.png)
>
> **What it says:** SAM has two key problems: "Coarse mask boundaries, often even neglecting the segmentation of thin object structures", and "Incorrect predictions, broken masks".
>
> **What it means:**
>
> These are faults 1 and 3 of our tests, named by another team. Their Figure 1 uses kite lines, our spokes in another form.

> [!PAPER] Ke et al. (2023), HQ-SAM · Abstract · page 1
> [![The HQ-SAM abstract: a learnable High-Quality Output Token injected into SAM's mask decoder, fused with early and final ViT features, trained on 44K fine-grained masks in 4 hours on 8 GPUs](/img/papers/sam/p6-hq-token.png)](/img/papers/sam/p6-hq-token.png)
>
> **What it says:** HQ-SAM adds "a learnable High-Quality Output Token" to SAM's mask decoder. It fuses the decoder's features "with early and final ViT features". It trains on "44K fine-grained masks", in "4 hours on 8 GPUs".
>
> **What it means:**
>
> Part 3 showed that each mask comes from one output token. The token becomes 32 numbers, and a dot product with each pixel's 32 numbers gives the mask. HQ-SAM adds one more token of the same kind. SAM's weights stay frozen; only the new token and a few small layers learn.
>
> The "early ViT features" are the important part for thin structures. Early blocks of the encoder still see edges and small details; late blocks see whole objects. HQ-SAM brings the early features back into the mask, at the same 256 × 256 grid as before. On the DIS dataset, the paper reports that the boundary quality rises from 52.8 to 70.4 mBIoU. That is the IoU measured only in a thin band along the edges.

The 44,000 training masks are tiny next to SA-1B's 1.1 billion. But the paper calls them "extremely accurate", taken from six existing datasets. A small set of very exact masks taught what a billion good masks did not.

### Medicine: MedSAM

> [!PAPER] Ma et al. (2024), MedSAM · Abstract · page 1
> [![The MedSAM abstract: a large-scale medical image dataset with 1,570,263 image-mask pairs covering 10 imaging modalities, evaluated on 86 internal validation tasks and 60 external validation tasks](/img/papers/sam/p6-medsam-abstract.png)](/img/papers/sam/p6-medsam-abstract.png)
>
> **What it says:** MedSAM trains on "1,570,263 image-mask pairs" from "10 imaging modalities". The tests cover "86 internal validation tasks and 60 external validation tasks".
>
> **What it means:**
>
> Section 8 expected domain tools to beat SAM in their domain. MedSAM shows the other path: take SAM and make it a domain tool. The MedSAM paper cites studies that found SAM good on targets with distinct boundaries, and weak on targets with weak boundaries or low contrast. Scans are full of such targets.

> [!DEFINITION] Imaging modality
> A kind of medical picture, made with a different physical method. Examples are CT (X-rays from many angles), MRI (magnetic fields), ultrasound and endoscopy (a camera inside the body). Each looks very different from a phone photo, and from each other.

> [!PAPER] Ma et al. (2024), MedSAM · Methods · page 12
> [![MedSAM methods: the model was initialized with the pre-trained SAM ViT-Base model; the trainable parameters of the image encoder and mask decoder are 89,670,912 and 4,058,340; the bounding box prompt was simulated with a random perturbation of 0-20 pixels](/img/papers/sam/p6-medsam-train.png)](/img/papers/sam/p6-medsam-train.png)
>
> **What it says:** MedSAM starts from "the pre-trained SAM model with the ViT-Base model". The trainable parameters of the image encoder and mask decoder are "89,670,912 and 4,058,340". The box prompt gets "a random perturbation of 0-20 pixels".
>
> **What it means:**
>
> Compare the first number with our weight count above: ViT-B's image encoder has 89,670,912 weights. It is the same network. MedSAM chose the small encoder, and it uses box prompts. Its paper argues that a box gives a less ambiguous target than a point.
>
> The random shift of the box copies SAM's own training game from Part 3. There, boxes got noise of 10% of the side, at most 20 pixels. **Fine-tuning** means to continue the training of a trained model on new data: here, SAM's weights are the start, and medical images teach the rest.

> [!PAPER] Ma et al. (2024), MedSAM · Discussion · page 9
> [![MedSAM limitations: modality imbalance in the training set, and difficulty with vessel-like branching structures, because arteries and veins share the same bounding box](/img/papers/sam/p6-medsam-limits.png)](/img/papers/sam/p6-medsam-limits.png)
>
> **What it says:** MedSAM has its own limits: "modality imbalance in the training set", and difficulty with "vessel-like branching structures", because "arteries and veins share the same bounding box" in eye pictures.
>
> **What it means:**
>
> The second limit is our spoked wheel again, in an eye. Blood vessels are thin and branch like a tree. A box around one vessel tree is also a box around the other tree, so the prompt is ambiguous. Fine-tuning on a new domain does not remove the thin-structure problem that comes from the design.

### Concepts: SAM 3

> [!PAPER] Carion et al. (2025), SAM 3 · Abstract · page 1
> [![The SAM 3 abstract: a unified model that detects, segments and tracks objects based on concept prompts, short noun phrases or image exemplars, returning masks for all matching object instances, with a dataset of 4M unique concept labels](/img/papers/sam/p6-sam3-abstract.png)](/img/papers/sam/p6-sam3-abstract.png)
>
> **What it says:** SAM 3 segments objects "based on concept prompts", which are "short noun phrases" such as "yellow school bus", image examples, or both. It returns masks for "all matching object instances". Its data engine produced "4M unique concept labels".
>
> **What it means:**
>
> SAM answers "what is at this point?" SAM 3 answers "where are all the yellow school buses?" One click gives one object. One phrase gives every object that matches it, each with its own mask.
>
> This is the answer to two limits of Section 8 at once. Text prompts are now a main feature, trained on millions of phrases, not a proof of concept. And "all instances of a concept" is close to the semantic and instance segmentation that SAM could not express with a click. The SAM 3 paper says that it predicts "instance and semantic masks".

> [!DEFINITION] Noun phrase
> A noun with optional words that describe it: "cat", "striped cat", "red apple". SAM 3 accepts short noun phrases, not long sentences. Its paper shows that a multimodal large language model, combined with SAM 3, can handle longer requests.

> [!PAPER] Carion et al. (2025), SAM 3 · Appendix B · page 18
> [![SAM 3 limitations: SAM 3 struggles to generalize to fine-grained out-of-domain concepts such as aircraft types and medical terms; concept generalization is inherently more challenging than the class-agnostic generalization of SAM and SAM 2](/img/papers/sam/p6-sam3-limits.png)](/img/papers/sam/p6-sam3-limits.png)
>
> **What it says:** "SAM 3 struggles to generalize to fine-grained out-of-domain concepts", such as aircraft types and medical terms. Concepts are "inherently more challenging than the class-agnostic generalization" of SAM and SAM 2.
>
> **What it means:**
>
> This closes a circle. SAM never named anything, and Part 1 saw that as a weakness. Here we see the strength in it: a model that only cuts, with no names, transfers to X-rays and paintings with no training. A model that must know what an "Airbus A320" is fails where it never saw one. Every name is one more thing to learn.

## The whole paper on one page

Six parts ago, we started with a photo, a click and three masks. Here is the paper again, in one picture and a few paragraphs.

{{FIG:p6_summary|The paper in one picture: the task (Part 2), the model (Part 3), the data engine and SA-1B (Part 4) and the zero-shot experiments (Part 5). The model makes the data, the data trains the model, and the prompt lets one model do many tasks.}}

**The task** ([Part 2](part-2-the-task.md)). Give the model any prompt, a point, a box, a rough mask or text. It must return a valid mask. If the prompt is ambiguous, any sensible object counts, so the model returns three masks: a part, a larger part and the whole. Language models learned from "predict the next word"; SAM learns from "predict the mask for this prompt".

**The model** ([Part 3](part-3-the-model.md)). A heavy ViT-H encoder reads the picture once and writes a 256 × 64 × 64 embedding. A light prompt encoder and mask decoder read that embedding for each click, fast enough for a person to click and see. The split into "once per picture" and "once per click" is the economic idea of the whole design.

**The data engine and the dataset** ([Part 4](part-4-data-engine-and-sa-1b.md)). No dataset of the right size existed, so the team built one with the model in the loop. Annotators first clicked with SAM's help, then corrected SAM's automatic masks, and at the end SAM made masks alone. The result, SA-1B, has 11 million photos and 1.1 billion masks.

**The experiments** ([Part 5](part-5-zero-shot-experiments.md)). With no extra training, SAM was tested on 23 datasets and on other tasks: edges, object proposals, instance segmentation and text. It does not win every score, but people rate its masks highly, and one model covers all the tasks through prompts.

**The limits** (this part). SAM misses thin structures, makes small islands, draws edges at the resolution of a 256 × 256 grid and needs a heavy encoder. The later papers kept SAM's idea and fixed one limit each.

And the [Background](background-the-road-to-sam.md) told where all this came from. Before SAM, there were two roads. Some models found and cut out a fixed list of things. Interactive tools cut out whatever you clicked. SAM joined the two roads, and it is still a central reference for the work that followed.

> [!TAKEAWAYS] Key takeaways
> - **SAM is a foundation model with a limited scope.** It learned most of its skill from 1.1 billion labelled masks, not from self-supervision; the data engine made that possible.
> - **Composition is the point of the prompt.** Any system that can produce a point or a box can use SAM, with no training: glasses that track the eye, 3D tools, later models.
> - **Thin structures fail both ways.** On our drawn wheel, ViT-H misses spokes of 1 and 2 pixels (recall 0.000 and 0.208). For spokes of 6 pixels or more, it fills the whole disc (gap fill 1.000). No width gave the correct shape.
> - **Islands are common in raw masks.** 34% to 82% of our raw single-click masks had stray pieces under 100 pixels. The paper's filters and clean-up exist for this reason.
> - **Edges follow a 256 × 256 grid.** Teeth up to 4 pixels wide vanish under a straight line; teeth of 16 pixels keep only 38% of their pixels. The final mask differs from the blocky grid mask in 1.9% of its pixels.
> - **The heavy encoder is the bottleneck.** The decoder has 4.06 million weights in both models; the encoder has 637 million (ViT-H) or 89.7 million (ViT-B).
> - **SAM 2 adds memory for video** and is 6 times faster on images; **MobileSAM** copies ViT-H into a 5.78M encoder; **HQ-SAM** adds one output token for sharper masks; **MedSAM** fine-tunes ViT-B SAM on 1.57 million medical image-mask pairs; **SAM 3** finds all instances of a noun phrase.
> - **The paper's own test still holds:** a foundation model is defined by its use. The later papers above are that use.

<details>
<summary>Run it yourself</summary>

The script [`code/papers/sam/sam_part6.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/sam/sam_part6.py) runs ViT-H and ViT-B on the bicycle photo, which it downloads from COCO. It draws the spoked wheel and the edge with teeth. It counts the islands in 512 masks per model, measures the soft band at the tyre edge and compares the two models. Every number of this part is in `results/part6.json`. We ran it on a laptop GPU (Apple MPS); no timing numbers are used, because other jobs ran at the same time.

```bash
pip install torch transformers datasets pillow scipy
python sam_part6.py        # uses facebook/sam-vit-huge and facebook/sam-vit-base
```

<figure><img src="/img/papers/sam/part6-run.png" alt="Terminal output of sam_part6.py: the weights of the two models, the bicycle masks, the table of the drawn wheel with spoke recall and gap fill for each width, and the counts of stray pieces" loading="lazy" /><figcaption>The first part of the output of sam_part6.py, as it ran for this article.</figcaption></figure>

</details>

## The paper and the code

[Segment Anything](https://arxiv.org/abs/2304.02643) (ICCV 2023) · [code and weights](https://github.com/facebookresearch/segment-anything) · [our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/sam)
