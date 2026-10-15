---
title: "The Road to SAM"
label: Background
description: "Before SAM, each segmentation model knew a fixed list of objects. How FCN, YOLO and Mask R-CNN got there, how GPT-3 and CLIP brought the prompt, and how SAM joins the two roads. With YOLO and SAM on the same photo."
date: 2026-10-15
tags: [sam, yolo, segmentation, object-detection, history]
---

A new idea is always an answer to the limits of the ideas before it. If you do not know those limits, a paper looks like a list of design decisions with no reason. So before we read the Segment Anything paper, we look at the road that led to it.

There are two roads. On the first road, computers learned to find objects and to cut them out of pictures. On the second road, large models learned to do a new task when you only *ask* for it. SAM is the place where the two roads meet.

{{FIG:road_timeline|Two roads. FCN (2015), YOLO (2016) and Mask R-CNN (2017) find and cut out objects, but only the objects of a fixed list. GPT-3 (2020) and CLIP (2021) do new tasks from a prompt. SAM (2023) brings the prompt to segmentation.}}

## The first road: find the object, then cut it out

### Before 2015: rules made by hand

The first programs used rules that people wrote: "an edge is where the colour changes quickly", "a region is a group of pixels with similar colour". In 2004, **GrabCut** added a person to the loop. You drew a box around the object, and the program separated the object from the background by colour.

Look at what GrabCut asked of you: a box. This is the same hint that SAM accepts today. But GrabCut knew only colours. A brown cat on a brown sofa was a problem, because the program did not know what a cat is.

### 2015, FCN: a class for every pixel

In 2012, deep networks became very good at **classification**: one label for one picture, "cat". But a label does not tell you *where* the cat is. The **fully convolutional network** (FCN) changed the end of a classification network. In place of one answer for the whole picture, it gives one answer for each pixel. This is **semantic segmentation**.

FCN had two limits, and the rest of the road is an answer to them:

1. **A fixed list.** FCN learned 21 classes (20 object types and "background"). A pixel of a dog's collar gets the class "dog" or "background", because "collar" is not on the list.
2. **No separate objects.** Two cats side by side become one region of "cat" pixels. FCN does not know that there are two cats.

**The equation of FCN.** How does a network learn to give a class to each pixel? It gets a score for each class at each pixel, and it changes the scores into probabilities. Then it gets a penalty for each pixel where the probability of the correct class is low. The total penalty is the **loss**, and training makes it small:

$$
L = -\frac{1}{N} \sum_{\text{pixels } n=1}^{N} \log p_n(y_n)
$$

Read it from the inside out. $$y_n$$ is the correct class of pixel $$n$$, for example "cat". $$p_n(y_n)$$ is the probability that the network gives to that correct class, a number from 0 to 1. The $$\log$$ of a number near 1 is near 0, and the $$\log$$ of a small number is a large negative number. The minus sign makes the penalty positive, and $$\frac{1}{N} \sum$$ takes the average over all $$N$$ pixels.

Two pixels show how it works. If the network gives "cat" a probability of 0.8 on a cat pixel, the penalty is $$-\log 0.8 = 0.22$$. If it gives only 0.1, the penalty is $$-\log 0.1 = 2.30$$, ten times larger. So training pushes hardest on the pixels where the network is most wrong. This loss is **cross-entropy**, and almost every segmentation model after FCN uses it, or a form of it.

### 2016, YOLO: find the objects, fast

**YOLO** ("You Only Look Once") solved a different problem: **detection**. Detection means a box around each object, and a name for each box.

Before YOLO, a detector looked at a picture many times. It first proposed thousands of possible regions, and then it classified each region. This was slow. YOLO looks at the picture once. It puts a grid of 7 × 7 cells over the picture. Each cell predicts some boxes, a score for each box and a class.

One pass of one network gives all the boxes. The first YOLO ran at 45 pictures each second, and a small version ran at 155. That is fast enough for live video.

Today, YOLO is a family of models. The versions from the company Ultralytics, for example YOLO11, learn the 80 object types of the COCO dataset. A "-seg" version also gives a mask for each box. But the main limit stays: the list of 80 types is fixed when you train the model.

> [!DEFINITION] COCO
> A dataset from 2014 with about 118,000 training pictures of everyday scenes. People drew the masks of objects of 80 types: person, car, cat, cup, and others. Most detection and segmentation models of this time learn and test on COCO.

**The equation of YOLO.** The whole answer of YOLO is one block of numbers. The first YOLO used a grid of $$S \times S = 7 \times 7$$ cells. Each cell predicts $$B = 2$$ boxes with 5 numbers each, and one probability for each of the $$C = 20$$ classes:

$$
S \times S \times (B \cdot 5 + C) = 7 \times 7 \times (2 \cdot 5 + 20) = 7 \times 7 \times 30 = 1{,}470 \text{ numbers}
$$

The 5 numbers of a box are its centre $$(x, y)$$, its width $$w$$, its height $$h$$ and a **confidence**. The confidence answers two questions at the same time: is there an object in this box, and how well does the box fit it?

$$
\text{confidence} = \Pr(\text{object}) \times \text{IoU}(\text{predicted box}, \text{true box})
$$

You know IoU from Part 1: the overlap divided by the total area. So a box with no object gets 0. A box on an object, but in the wrong place, gets a low IoU and a low confidence.

One more detail shows how carefully YOLO was designed.

In its loss, YOLO compares $$\sqrt{w}$$ and $$\sqrt{h}$$, not $$w$$ and $$h$$. Why? An error of 10 pixels is small for a large box and large for a small box. For a box of width 100 that the network predicts as 110, the error is $$\sqrt{110} - \sqrt{100} = 0.49$$. For a box of width 10 that it predicts as 20, the error is $$\sqrt{20} - \sqrt{10} = 1.31$$. The same 10 pixels cost almost three times more on the small box.

### 2017, Mask R-CNN: a mask for every object

**Mask R-CNN** joined detection and segmentation. It first finds a box for each object, as a detector does. Then a small extra branch draws the mask inside each box. Now the two cats are two masks. This is **instance segmentation**, and Mask R-CNN was the standard method for years.

But look at what did not change. Mask R-CNN learns the 80 types of COCO. For a new type of object, you collect new pictures, people draw new masks, and you train again. Each model is a specialist with a closed list.

**The equation of Mask R-CNN.** Mask R-CNN learns three jobs at the same time: the class, the box and the mask. Its loss is the sum of three losses, one for each job:

$$
L = L_{\text{cls}} + L_{\text{box}} + L_{\text{mask}}
$$

The new part is $$L_{\text{mask}}$$. Inside each box, the network makes a small mask of $$28 \times 28$$ pixels. For each pixel, it gives a probability $$\hat{y}$$ that the pixel is part of the object. The correct value $$y$$ is 1 or 0. The penalty is **binary cross-entropy**, the average over the $$m \times m$$ pixels:

$$
L_{\text{mask}} = -\frac{1}{m^2} \sum_{i,j} \Big[ y_{ij} \log \hat{y}_{ij} + (1 - y_{ij}) \log (1 - \hat{y}_{ij}) \Big]
$$

Only one of the two terms is active for each pixel. When $$y = 1$$, the penalty is $$-\log \hat{y}$$: a pixel of the object with $$\hat{y} = 0.9$$ costs 0.11. When $$y = 0$$, the penalty is $$-\log(1 - \hat{y})$$: a background pixel with $$\hat{y} = 0.7$$ costs 1.20.

The clever part: Mask R-CNN makes one mask for each class, but it uses only the mask of the correct class in the loss. So the classes do not compete for the pixels. The box branch says "this is a cat", and the mask branch only has to answer "which pixels?". SAM uses the same split, and goes further: it has no class at all.

{{FIG:road_questions|The question that each type of model answers. Each step gives a more exact answer, but every model on the first road answers only for its own list of object types.}}

## The second road: ask, and the model does it

The second road starts in language. **GPT-3** (2020) showed that one very large model can do many tasks with no more training. You write the task as text, and the model does it. The text is a **prompt**, and the use of a task with no training examples is **zero-shot**.

**The equation of GPT-3.** GPT-3 has one job in training: predict the next word. The probability of a whole text is the product of the probabilities of each word, given all the words before it:

$$
p(x_1, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1}), \qquad L = -\sum_{t=1}^{T} \log p(x_t \mid x_1, \ldots, x_{t-1})
$$

$$\prod$$ means "multiply all of these", and the bar $$\mid$$ means "given". So $$p(x_t \mid x_1, \ldots, x_{t-1})$$ reads "the probability of word $$t$$, given the words before it". The loss $$L$$ is the same cross-entropy as in FCN, but for words in place of pixels. After "The cat sat on the", a good model gives "mat" a probability of perhaps 0.6, a penalty of $$-\log 0.6 = 0.51$$. A weak model gives 0.05, a penalty of 3.00.

Now see why a prompt works. A prompt is only the first words $$x_1, \ldots, x_{t-1}$$. The model continues the text, so the prompt *conditions* the answer. The weights do not change. SAM copies this idea: the click is the condition, and the mask is the continuation.

**CLIP** (2021) brought this idea to pictures. CLIP learned from 400 million pictures with their captions from the web. After training, you can write any name, "a photo of a dog collar", and CLIP can tell you if a picture matches it. There is no fixed list. But CLIP gives one answer for the whole picture. It does not tell you which pixels are the collar.

**The equation of CLIP.** CLIP has two encoders: one changes a picture into a vector $$I$$, and one changes a caption into a vector $$T$$. In training, it gets $$N$$ pictures with their $$N$$ captions. It computes a score for each picture-caption pair, the **cosine similarity**, divided by a small number $$\tau$$:

$$
s_{ij} = \frac{\cos(I_i, T_j)}{\tau}, \qquad L_{\text{picture } i} = -\log \frac{e^{s_{ii}}}{\sum_{j=1}^{N} e^{s_{ij}}}
$$

The cosine is 1 when two vectors point the same way, and 0 when they have nothing in common. The fraction is a **softmax**: it changes the $$N$$ scores of picture $$i$$ into probabilities that add up to 1. $$e^{s_{ii}}$$ is the score of the correct caption, so the fraction is "the probability that CLIP chooses the correct caption". The loss is again cross-entropy. CLIP adds the same loss in the other direction, from each caption to the pictures.

An example with three captions and $$\tau = 0.1$$. A cat picture has the cosines 0.9 with its own caption, and 0.2 and 0.1 with two wrong captions. The softmax gives the correct caption a probability of 0.999, so the loss is almost 0. Now say that one wrong caption, "a kitten", has a cosine of 0.85. The probability falls to 0.62, and the loss rises to 0.47. CLIP learns from such near misses.

After training, zero-shot classification is one line. Write a caption for each class, "a photo of a {class}", and choose the class whose caption has the largest cosine with the picture. Any word can be a class, so there is no fixed list.

Part 1 reads these two papers in the words of the SAM paper.

## SAM: the two roads meet

Now put the two roads side by side. The first road knows how to cut out objects, but only from a fixed list. The second road has no fixed list, but it does not cut out anything.

SAM takes the best part of each road. From the first road, it takes the mask. From the second road, it takes the prompt. The result is a model that answers a new question. The question is not "where are the cats?" but "which pixels belong to the thing that I show you?".

To answer that question, SAM does not need a list of object types at all. It does not know the word "cat". It knows only what an object *looks like*: a region with a clear edge, a part of a larger thing. This is why one click can give the ear, the head or the whole cat. The paper calls this property **class-agnostic**: the mask has no class name.

**The equation of SAM.** SAM's training loss joins the ideas of this page. For each prompt, SAM makes three masks $$M_1, M_2, M_3$$, and the correct mask is $$Y$$:

$$
L = \min_{k \in \{1,2,3\}} \Big[ 20 \cdot L_{\text{focal}}(M_k, Y) + L_{\text{dice}}(M_k, Y) \Big] + \big( \widehat{\text{IoU}} - \text{IoU}(M, Y) \big)^2
$$

Read it in three pieces.

- **The minimum over $$k$$.** Only the best of the three masks gets the penalty. This is how SAM learns ambiguity. A click on the ear can give "ear" in one output and "cat" in another. Neither output gets a penalty for the answer of the other.
- **Focal loss and dice loss, in a ratio of 20 to 1.** The focal loss is the pixel cross-entropy of FCN with one change: it multiplies the penalty by $$(1 - p)^2$$. An easy pixel with $$p = 0.9$$ costs 0.105 in cross-entropy, but only 0.001 in focal loss. A hard pixel with $$p = 0.3$$ keeps most of its penalty: 0.59 in place of 1.20. So the training spends its effort on the hard pixels, usually at the edges.

  The dice loss looks at the whole mask: $$L_{\text{dice}} = 1 - \frac{2 |M \cap Y|}{|M| + |Y|}$$. It is a close relative of IoU. For the two ear masks of Part 1, the dice score is $$\frac{2 \times 424}{427 + 470} = 0.945$$ and the IoU is 0.896.
- **The squared error of the score.** $$\widehat{\text{IoU}}$$ is the score that SAM predicts for its mask, and $$\text{IoU}(M, Y)$$ is the real IoU. The square of the difference teaches SAM to estimate the quality of its own masks. These are the scores that you saw in Part 1.

Part 3 reads this loss in the words of the paper.

> [!DEFINITION] Class-agnostic
> A model is class-agnostic when its output does not depend on a list of object types. SAM gives a mask for a thing, but it never says what the thing is.

## YOLO and SAM on the same photo

The best way to see the difference is to run both models on the same picture. We used the cat photo of Part 1 (640 × 480 pixels), YOLO11n-seg, and SAM with the ViT-H encoder. The script is [`sam_road.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/sam/sam_road.py).

```python
yolo = YOLO('yolo11n-seg.pt')
res = yolo.predict(image, conf=0.25)[0]                          # one pass: boxes, names, scores, masks

for box in res.boxes.xyxy:                                        # each YOLO box becomes a SAM prompt
    p = proc(images=image, input_boxes=[[box.tolist()]], return_tensors='pt')
    out = sam(image_embeddings=emb, input_boxes=p['input_boxes'], multimask_output=False)
```

```text
== YOLO11n-seg ==
parameters: 2,876,848   classes it knows: 80
is "ear" a class? False   is "button" a class? False
object      score   box (x1, y1, x2, y2)     mask pixels
cat         0.860   (5, 53, 317, 470)             54,371
cat         0.846   (339, 27, 640, 374)           61,884
remote      0.821   (40, 73, 176, 118)             4,564
couch       0.555   (1, 1, 637, 473)             174,666
time for the whole picture (all objects): 10.1 ms (mps)   11.8 ms (cpu)

== SAM ViT-H, prompted with the YOLO boxes ==
parameters: 641,090,864 (223 times YOLO11n-seg)
object     SAM score  SAM pixels  YOLO pixels  IoU(YOLO, SAM)
cat            0.964      50,508       54,371           0.928
cat            0.992      58,097       61,884           0.923
remote         0.972       4,105        4,564           0.899
couch          0.991     183,830      174,666           0.944
time: SAM image encoder 2947.5 ms (mps), then about 6 ms for each box (Part 1)
```

<figure><img src="/img/papers/sam/road-yolo-sam.png" alt="Two copies of the cat photo. Left: YOLO's result, with boxes labelled cat 0.86, cat 0.85, remote 0.82 and couch 0.56, and a coloured mask in each box. Right: SAM's masks for the same four boxes, with no labels" loading="lazy" /><figcaption>Left: YOLO finds four objects and names them. Right: SAM gets YOLO's four boxes as prompts and gives one mask for each box. SAM gives no names. Neither model shows the second remote control, next to the right cat: YOLO did not find it, so SAM got no box for it.</figcaption></figure>

Read the output slowly, because each line teaches one difference.

**YOLO knows names, and SAM does not.** YOLO says "cat", "remote" and "couch". SAM says nothing about what the objects are. It only gives the pixels and a score for the quality of the mask.

**YOLO has a list, and SAM does not.** YOLO has no class "ear" and no class "button". It cannot give you the cat's ear, because nobody trained it on ears. In Part 1, one click on the ear gave SAM a mask of the ear (427 pixels). SAM can cut out a thing that has no name in any list.

**YOLO is very fast, and SAM is slow for each new picture.** YOLO finds all four objects in 10 ms. SAM's image encoder needs 2,948 ms before it can answer the first prompt. YOLO is about 290 times faster on this picture. It is also 223 times smaller. YOLO11n is the smallest YOLO, but even the largest YOLO11 has about 60 million weights, ten times fewer than SAM ViT-H.

**SAM's masks are more exact.** The masks of the two models agree well: the IoU is 0.90 to 0.94. But YOLO's masks for the cats and the remote are 7% to 11% larger than SAM's. YOLO makes its masks at a low resolution and scales them up, so its edges are less exact. SAM was trained on 1.1 billion masks of every type of object, and the quality of the edge is its main job.

**SAM needs someone to show it the object.** YOLO did not find the second remote control, which lies between the two cats. So SAM did not get a box for it, and it made no mask. SAM never searches the picture for objects by itself. A person, a grid of points or another model must tell it *where* to look.

{{FIG:road_yolo_sam|The two models side by side, with our numbers. YOLO takes only the picture and gives boxes, names and masks for 80 fixed types. SAM takes the picture and a prompt, and gives a mask and a score with no name.}}

| | YOLO11n-seg | SAM ViT-H |
|---|---|---|
| Question | Which objects of my list are in the picture, and where? | Which pixels belong to the thing that you show me? |
| Input | the picture | the picture and a prompt (point, box, mask) |
| Output | boxes, class names, scores, masks | masks and a quality score, no names |
| Object types | 80, fixed in training | no list |
| Training data | COCO: about 118,000 pictures | SA-1B: 11 million pictures, 1.1 billion masks |
| Weights | 2.9 million | 641 million |
| Time on our laptop | 10 ms for the whole picture | 2,948 ms for each picture, then about 6 ms for each prompt |
| Good for | live video, counting, tracking known objects | exact cut-outs, new object types, labelling tools |

## The two together

So which model is better? Neither. They answer different questions, and they work well together. YOLO finds the objects and names them. SAM turns each YOLO box into an exact mask. You get the speed and the names of YOLO, and the edges of SAM.

The SAM paper does the same thing in one of its experiments (Part 5). A detector called ViTDet gives the boxes, and SAM gives the masks. SAM had no training for this job, but people rated its masks higher than the masks of the detector itself. Later tools such as Grounded-SAM add a detector that takes text. You write "the dog's collar", the detector finds a box, and SAM cuts it out.

## Why this matters for reading the paper

Keep three facts from this page. Each one explains a decision in the paper.

1. **"Anything" is the answer to the fixed list.** FCN, YOLO and Mask R-CNN each learned a closed list of object types. SAM is class-agnostic, so it has no list. This is why the paper defines its task with prompts, not with classes (Part 2).
2. **The words "prompt", "zero-shot" and "foundation model" come from the second road.** The paper copies the recipe of GPT-3 and CLIP: one large training, then a prompt for each new task (Part 1).
3. **SAM is fast for each click, not for each picture.** YOLO is fast for the whole picture. SAM puts almost all its work in the image encoder, which runs once. Then each click takes only milliseconds. The paper calls this "amortized real-time" (Parts 1 and 3).

> [!TAKEAWAYS] Key takeaways
> - **The first road** (FCN 2015, YOLO 2016, Mask R-CNN 2017) taught computers to find objects and to cut them out. But each model knew only a fixed list of object types.
> - **The second road** (GPT-3 2020, CLIP 2021) gave models that do a new task from a prompt, with no more training. CLIP has no fixed list, but it gives no pixels.
> - **SAM joins the two roads:** a mask from the first, a prompt from the second. It is class-agnostic: it gives the pixels of a thing, but not its name.
> - **YOLO and SAM answer different questions.** On our photo, YOLO found and named four objects in 10 ms with 2.9 million weights. SAM needed 2,948 ms and 641 million weights. But its masks are more exact, and it can cut out things with no name, such as an ear.
> - **Together they are strong:** a detector gives boxes and names, and SAM turns each box into an exact mask.

**Next:** [Part 1](part-1-the-big-idea.md) reads the title, Figure 1, the abstract and the introduction of the paper.

<details>
<summary>Run it yourself</summary>

The script [`code/papers/sam/sam_road.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/sam/sam_road.py) runs YOLO11n-seg and SAM ViT-H on the same photo. It saves the numbers in `results/road.json` and the picture above.

```bash
pip install torch transformers datasets pillow ultralytics
python sam_road.py        # downloads yolo11n-seg.pt (6 MB) and facebook/sam-vit-huge (2.6 GB)
```

</details>

## The papers

[FCN](https://arxiv.org/abs/1411.4038) (2015) · [YOLO](https://arxiv.org/abs/1506.02640) (2016) · [Mask R-CNN](https://arxiv.org/abs/1703.06870) (2017) · [GPT-3](https://arxiv.org/abs/2005.14165) (2020) · [CLIP](https://arxiv.org/abs/2103.00020) (2021) · [SAM](https://arxiv.org/abs/2304.02643) (2023)
