---
title: "Fine-tuning, Higher Resolution and the Setup"
description: "How a pre-trained ViT is adapted to a new task, why it is fine-tuned on bigger pictures, and the full training recipe behind every result. We swap the head, stretch the position embeddings, fine-tune on ten classes and solve the few-shot metric by hand."
part: 3
covers: "§3.2, §4 (opening), §4.1, Tables 1, 3 and 4, Appendix B.1 and B.1.1"
date: 2026-10-17
tags: [vit, vision-transformer, fine-tuning, transfer-learning, computer-vision]
---

[Part 2](part-2-the-model.md) built ViT from pixels to a class label. A model is only half of a paper, though. The other half is how it is trained, adapted and measured, and that half decides every number in the results.

## Fine-tuning: a new head for a new task {§3.2}

> [!PAPER] · §3.2
> [![The opening of Section 3.2, Fine-tuning and higher resolution: typically, we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks. For this, we remove the pre-trained prediction head and attach a zero-initialized D by K feedforward layer, where K is the number of downstream classes](/img/papers/vit/p3-finetune-head.png)](/img/papers/vit/p3-finetune-head.png)
>
> **What it says:** "we pre-train ViT on large datasets, and fine-tune to (smaller) downstream tasks". To do so, "we remove the pre-trained prediction head and attach a zero-initialized $$D \times K$$ feedforward layer, where $$K$$ is the number of downstream classes."
>
> **What it means:**
>
> Think of a doctor who trained for years in a big hospital and then joins a small clinic. The clinic does not retrain the doctor from zero. It keeps everything the doctor knows and only teaches the clinic's own forms. The pre-trained body of ViT is the doctor; the head is the form.
>
> The old head answered one question: which of the 21,843 ImageNet-21k classes (or 18,000 JFT classes) is this? A new task asks a different question, with $$K$$ answers, so the old head is useless and is thrown away. The new head is one matrix of $$D \times K$$ numbers, 768 × 10 for ViT-B on a 10-class task, and every one of them starts at zero.
>
> The word "zero-initialized" looks like a detail, but it has a clean effect that we can measure. At the first step every class gets the same score, and the body of the network receives no gradient at all. We check this below.

> [!DEFINITION] Downstream task
> The task you actually care about, after pre-training: classifying pets, flowers or medical scans. "Downstream" because it comes after the big pre-training run, and usually has far fewer labelled pictures. [Part 1](part-1-the-big-idea.md) defines pre-training and fine-tuning.

> [!DEFINITION] Zero initialisation
> Starting a layer with every weight and bias equal to 0, instead of small random numbers. A zero layer outputs 0 for every input, so at the start it says nothing and adds no noise.

The released ViT-B/16 checkpoint pre-trained on ImageNet-21k shows the old head. It ends in a "pre-logits" layer, a $$768 \times 768$$ matrix with a tanh (Part 2 met it as the library's "pooler"). The 21,843-class layer on top of it was not released. Appendix B.1.1 says what is removed:

> [!PAPER] · Appendix B.1.1
> [![Appendix B.1.1: when transferring ViT models to another dataset, we remove the whole head (two linear layers) and replace it by a single, zero-initialized linear layer outputting the number of classes required by the target dataset. We found this to be a little more robust than simply re-initializing the very last layer](/img/papers/vit/p3-b11-head.png)](/img/papers/vit/p3-b11-head.png)
>
> **What it says:** "we remove the whole head (two linear layers) and replace it by a single, zero-initialized linear layer". This was "a little more robust than simply re-initializing the very last layer."
>
> **What it means:**
>
> The pre-training head has two layers: the pre-logits layer and the class layer. A common habit is to keep the first and replace only the second. The paper removes both, so the new task reads the [class] token's representation directly, as Equation 4 of Part 2 defines it.
>
> "A little more robust" is an honest phrase. The authors do not claim it is much better, only that it worked a little more reliably. One way to read it: the pre-logits layer was shaped to serve 21,843 classes. For a new task, it is one more layer that must unlearn that shape.

`vit_part3_finetune.py` loads the ImageNet-21k checkpoint into a 10-class classifier, sets the new head to zero and runs one batch of 32 Imagenette pictures through it. (Imagenette is a small public set of 10 ImageNet classes, described in the fine-tuning run below.)

```python
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=10)
torch.nn.init.zeros_(model.classifier.weight); torch.nn.init.zeros_(model.classifier.bias)
loss = F.cross_entropy(model(pixel_values=x).logits, y); loss.backward()
```

```text
== 0. data: Imagenette, 9469 training and 3925 validation pictures, 10 classes; device mps ==
  loaded google/vit-base-patch16-224-in21k into a 10-class classifier
  the pre-logits layer (768 x 768 + tanh) is dropped: model.vit.pooler is None
  new head: Linear(in_features=768, out_features=10, bias=True) with every weight and bias set to 0 (7,690 numbers)

== 1. step 0 with the zero head ==
  logits of the first picture: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  -> softmax gives 0.1 to each of the 10 classes
  loss = -ln(0.1) = ln 10 = 2.302585;  measured 2.302585
  gradient of the head weight by hand, (softmax - one_hot)^T y / batch: norm 1.2152; autograd: 1.2152; max |difference| 0.0e+00
  gradient reaching the body (all 85.8M other numbers): norm 0.0  (W_head^T (softmax - one_hot) = 0 because W_head = 0)
  for comparison, a random head (normal, std 0.02): loss 2.3175, gradient reaching the body: norm 0.561, logits of picture 1 range -0.13 to 0.13
```

The loss of the first step is exactly $$\ln 10$$, because ten equal scores give each class a probability of 0.1. The more surprising line is the last one: the body gets a gradient of exactly 0. The chain rule explains it. The gradient that flows back into the [class] token is the head's weight matrix times the error of each class:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{y}} = W_{\text{head}}^\top\,(\mathbf{p} - \mathbf{t}) = \mathbf{0} \quad \text{when } W_{\text{head}} = \mathbf{0}
$$

Here $$\mathbf{y}$$ is the image representation (768 numbers) and $$W_{\text{head}}$$ the new $$10 \times 768$$ head. $$\mathbf{p}$$ holds the 10 predicted probabilities, and $$\mathbf{t}$$ is the one-hot target (1 for the right class, 0 for the rest). The head itself does get a gradient, $$(\mathbf{p} - \mathbf{t})\,\mathbf{y}^\top$$ averaged over the batch, because $$\mathbf{y}$$ is not zero. So step 1 changes only the head. From step 2 on, the head is no longer zero and the body starts to move.

A random head does the opposite. Its scores are noise (−0.13 to 0.13 here), and its loss starts above $$\ln 10$$, at 2.3175. That noise sends a gradient of size 0.561 into all 85.8 million pre-trained numbers on the very first step. The zero head lets the pre-trained body wait until the head has learned something sensible.

{{FIG:p3_head_swap|The head swap. Left: the ImageNet-21k checkpoint ends in a pre-logits layer (768 × 768, tanh) and a 21,843-class layer. Both are removed. Right: one new 768 × 10 layer of zeros reads the [class] representation. At step 0 all ten scores are 0, the loss is ln 10 = 2.3026, the head gets a gradient and the body gets none.}}

## Fine-tuning at a higher resolution {§3.2}

> [!PAPER] · §3.2
> [![Section 3.2 continued: it is often beneficial to fine-tune at higher resolution than pre-training (Touvron et al., 2019; Kolesnikov et al., 2020). When feeding images of higher resolution, we keep the patch size the same, which results in a larger effective sequence length. The Vision Transformer can handle arbitrary sequence lengths (up to memory constraints), however, the pre-trained position embeddings may no longer be meaningful](/img/papers/vit/p3-higher-res.png)](/img/papers/vit/p3-higher-res.png)
>
> **What it says:** "It is often beneficial to fine-tune at higher resolution than pre-training". The patch size stays the same, "which results in a larger effective sequence length". ViT "can handle arbitrary sequence lengths", but "the pre-trained position embeddings may no longer be meaningful."
>
> **What it means:**
>
> Pre-training runs at 224 × 224 pixels because it is long and expensive. Fine-tuning is short, so it can afford bigger pictures, and bigger pictures show finer detail. Table 4 sets the usual fine-tuning size at 384 × 384.
>
> Keep the patch at 16 × 16 pixels and the picture at 384, and you get $$384/16 = 24$$ patches per side, so 576 patches instead of 196. The weights of the encoder do not care: the same matrices act on every token, however many there are. Only one part of the model has a fixed size, the table of position embeddings. It has 197 rows, and the longer sequence needs 577.

> [!DEFINITION] Resolution
> The size of the picture in pixels, here width × height: 224 × 224 or 384 × 384. More pixels show smaller details, and for ViT they also mean more patches, so more tokens and more compute.

The arithmetic, with the symbols of Part 2 ($$N = HW/P^2$$):

$$
N_{224} = \frac{224 \times 224}{16^2} = 14 \times 14 = 196, \qquad N_{384} = \frac{384 \times 384}{16^2} = 24 \times 24 = 576
$$

So the sequence grows from 197 to 577 tokens, about 2.9 times longer. Attention compares every pair of tokens, so its pairs grow from 38,809 to 332,929, about 8.6 times. The whole forward pass of ViT-B/16 grows from about 34.9 to 110.3 billion operations. That is our estimate from the shapes, printed in the Model variants section below.

The paper cites two sources for "often beneficial". One is BiT, the ResNet baseline we meet below. The other is a paper about exactly this effect:

> [!PAPER] Touvron et al. (2019), Fixing the train-test resolution discrepancy · Abstract
> [![The abstract of Touvron et al. (2019): data-augmentation is key; existing augmentations induce a significant discrepancy between the size of the objects seen by the classifier at train and test time: in fact, a lower train resolution improves the classification at test time. We propose a simple strategy that employs different train and test resolutions, relying on a computationally cheap fine-tuning of the network at the test resolution. A ResNet-50 trained on 128 by 128 images reaches 77.1% top-1 accuracy on ImageNet, and 79.8% when trained at 224 by 224](/img/papers/vit/p3-ext-touvron.png)](/img/papers/vit/p3-ext-touvron.png)
>
> **What it says:** common augmentations create "a significant discrepancy between the size of the objects seen by the classifier at train and test time"; "a lower train resolution improves the classification at test time"; the fix is "a computationally cheap fine-tuning of the network at the test resolution".
>
> **What it means:**
>
> During training, pictures are randomly cropped and zoomed, so objects look larger than in the plain test pictures. Testing at a higher resolution makes test objects larger too, so the two match again. A short fine-tuning at the test resolution then repairs the rest.
>
> This is the habit ViT follows: train long at a small size, fine-tune briefly at a larger one. For a CNN the change is easy, because a convolution slides over any size of grid. For ViT one piece does not stretch by itself, and the next sentence of the paper fixes it.

> [!PAPER] · §3.2
> [![The end of Section 3.2: we therefore perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image. Note that this resolution adjustment and patch extraction are the only points at which an inductive bias about the 2D structure of the images is manually injected into the Vision Transformer](/img/papers/vit/p3-interpolation.png)](/img/papers/vit/p3-interpolation.png)
>
> **What it says:** "We therefore perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image." This step and the patch cutting "are the only points at which an inductive bias about the 2D structure of the images is manually injected".
>
> **What it means:**
>
> The 196 position rows were learned for a 14 × 14 grid. Lay them out on that grid, like 196 pins on a map. Now stretch the map to 24 × 24. Each new pin gets a value made from the old pins nearest to its place, and the stretched map has 576 pins.
>
> "2D" is the key word. The rows are not stretched as one long list, which would mix the end of one grid row with the start of the next. They are stretched as a grid, rows and columns separately. Together with the cut into patches, it is the only moment when the model is told that its patches form a picture.

> [!DEFINITION] Interpolation
> Filling in values between known values. If a thermometer reads 10° at noon and 14° at 2 pm, linear interpolation guesses 12° at 1 pm. **Bilinear** interpolation does the same on a grid: first along the rows, then along the columns. **Bicubic** uses 4 × 4 neighbours and a smoother curve.

**A tiny example first.** The official code resizes the grid with `scipy.ndimage.zoom(..., order=1)`, which is linear interpolation in each direction with the corner values kept. On a 2 × 2 grid of single numbers, resized to 3 × 3 ([`vit_part3_posemb.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part3_posemb.py)):

```text
== 1. resize a 2x2 grid to 3x3 the way the official code does (scipy.ndimage.zoom, order=1 = linear) ==
  grid 2x2: [[0.0, 1.0], [2.0, 3.0]]
    0.00  0.50  1.00
    1.00  1.50  2.00
    2.00  2.50  3.00
  corners stay, the new middle row and column are averages of their neighbours: (0+1)/2 = 0.5, (0+2)/2 = 1, (0+1+2+3)/4 = 1.5
```

Check it by hand. The four corners keep their values 0, 1, 2 and 3. The new top-middle value sits halfway between 0 and 1, so it is 0.5. The centre sits halfway between all four, so it is $$(0 + 1 + 2 + 3)/4 = 1.5$$.

{{FIG:p3_interp_tiny|Bilinear interpolation on the smallest grid. Left: a 2×2 grid of values 0, 1, 2, 3. Right: the same grid stretched to 3×3. The corners keep their values; each new cell is the average of the old cells around its place (0.5, 1, 1.5, 2, 2.5).}}

**The formula.** For one new grid row $$i$$ out of $$n_{\text{new}}$$, the place on the old grid of $$n_{\text{old}}$$ rows is

$$
y = i \cdot \frac{n_{\text{old}} - 1}{n_{\text{new}} - 1}, \qquad \text{row}_{\text{new}}(i) = (1 - \delta)\,\text{row}_{\text{old}}(\lfloor y \rfloor) + \delta\,\text{row}_{\text{old}}(\lfloor y \rfloor + 1), \quad \delta = y - \lfloor y \rfloor
$$

where $$\lfloor y \rfloor$$ is $$y$$ rounded down, and $$\delta$$ is how far the new row sits past that old row. The same rule then runs along the columns. Each "row" here is a whole vector of 768 numbers, so the mix is done number by number. For ViT-B/16 at 384, $$n_{\text{old}} = 14$$ and $$n_{\text{new}} = 24$$. New row 5 sits at $$5 \times 13/23 = 2.8261$$, so it takes 0.1739 of old row 2 and 0.8261 of old row 3.

**The real table.** We wrote this rule as a plain loop and ran it on the 14 × 14 × 768 position grid of `google/vit-base-patch16-224`. Then we compared it with SciPy and with the library:

```text
== 2. the position table of google/vit-base-patch16-224 ==
  E_pos (197, 768) = 1 row for [class] + 196 rows = a 14x14 grid of 768 numbers
  at 384 pixels: 384/16 = 24 patches per side, N = 576, so 577 rows are needed: 380 rows do not exist
  our bilinear 14x14 -> 24x24: (24, 24, 768); max |difference| vs scipy.ndimage.zoom(order=1): 2.3e-15
  new row 5 sits at old position 5 x 13/23 = 2.8261: 0.1739 of old row 2 + 0.8261 of old row 3
  library interpolate_pos_encoding: bicubic, align_corners=False; our bicubic call matches it: max |difference| 0.0e+00
  bilinear (official) vs bicubic (library) position rows: mean |difference| 0.0131, mean |value| 0.0784
  corner row (0,0) kept by bilinear: max |difference| 0.0e+00
```

Our loop equals SciPy's resize to $$2 \times 10^{-15}$$, which is the rounding of 64-bit numbers. The Hugging Face library does something slightly different: it uses bicubic interpolation, not bilinear. The two tables differ by 0.0131 on average, against an average size of 0.0784 for the numbers themselves. So the official and the library versions are close but not the same.

To see what the stretch keeps, take one position row and compare it, by cosine similarity, with every other row. In the 14 × 14 grid, row (3, 10) is most like its four neighbours (0.79 to 0.81) and unlike the far corner (−0.29). After the stretch, the same place on the 24 × 24 grid shows the same pattern, only smoother. Its neighbours are now closer (0.92 to 0.95), because they are made from the same old rows.

{{FIG:p3_posemb_sim|Cosine similarity of one position row with all others, before and after the 2D interpolation. Left: the 14×14 grid of google/vit-base-patch16-224, row (3, 10), bright near itself and dark far away. Right: the 24×24 grid after the bilinear stretch, the same place (5, 18): the same map, stretched. Part 5 reads these maps in full.}}

**Does it work?** We ran the ImageNet-fine-tuned ViT-B/16 on the 3,925 validation pictures of Imagenette. We used 224 and 384 pixels, and five versions of the position table. There is no training here at all; only the position table changes.

```text
== 4. accuracy on the Imagenette validation set: 3925 pictures (160 px on the short side), device mps ==
  the ImageNet head of the checkpoint is read at the 10 Imagenette classes; no training; "of 1000" = the top class of all 1000 must be right
  224, the training resolution (196 patches)     accuracy 99.54% (of 10)   83.13% (of 1000)   [65 s]
  384, 2D bilinear (the official code)           accuracy 99.69% (of 10)   84.76% (of 1000)   [196 s]
  384, 2D bicubic (the library)                  accuracy 99.69% (of 10)   84.97% (of 1000)   [1582 s]
  384, 1D: the 196 rows stretched as one list    accuracy 92.00% (of 10)   48.38% (of 1000)   [1583 s]
  384, no position rows at all (zeros)           accuracy 52.64% (of 10)   13.99% (of 1000)   [297 s]
```

The two 2D stretches keep the model working at 384. Over all 1,000 ImageNet classes, accuracy is 84.76% with the official bilinear table and 84.97% with the library's bicubic one, against 83.13% at 224. Read at only the 10 Imagenette classes, all three are almost perfect (99.54% to 99.69%).

The wrong stretch is a disaster. Stretching the 196 rows as one list of 576 drops the accuracy to 48.38%. A 1D stretch treats the last patch of one grid row and the first patch of the next as neighbours. So most new rows land in the wrong place. With no position rows at all the model falls to 13.99%; [Part 2](part-2-the-model.md) showed why, since attention alone sees only a bag of patches.

Two honest limits. This model was never fine-tuned at 384, and the paper always fine-tunes after the stretch. Our numbers show only that the stretched table is a good start. And Imagenette stores its pictures at 160 pixels, so at 384 they are only upscaled, with no new detail. The small gain at 384 cannot come from detail. It fits the explanation of Touvron et al. that objects simply look bigger.

The times in brackets include waiting for another job on the same GPU, so they say nothing about speed.

{{FIG:p3_res_acc|Accuracy of the released ViT-B/16 (ImageNet fine-tuned) on the 3,925 Imagenette validation pictures, scored over all 1,000 ImageNet classes, with five versions of the position table and no training. The 2D stretches keep the model working at 384; a 1D stretch of the list or no position rows at all break it.}}

## The experiments: three model families {§4}

> [!PAPER] · §4
> [![The opening of Section 4, Experiments: we evaluate the representation learning capabilities of ResNet, Vision Transformer (ViT), and the hybrid. To understand the data requirements of each model, we pre-train on datasets of varying size and evaluate many benchmark tasks. When considering the computational cost of pre-training the model, ViT performs very favourably, attaining state of the art on most recognition benchmarks at a lower pre-training cost. Lastly, we perform a small experiment using self-supervision](/img/papers/vit/p3-experiments-intro.png)](/img/papers/vit/p3-experiments-intro.png)
>
> **What it says:** the paper evaluates "the representation learning capabilities of ResNet, Vision Transformer (ViT), and the hybrid", pre-trained on "datasets of varying size". ViT reaches the state of the art "at a lower pre-training cost". A small experiment uses "self-supervision".
>
> **What it means:**
>
> This paragraph is the table of contents of the results. Three model families, three dataset sizes, many tasks, and the cost of each run. [Part 4](part-4-results.md) reads the results, and Part 5 reads the cost study and self-supervision.
>
> "Representation learning" is the frame for everything. The real product of pre-training is not the head, which is thrown away. It is the body, which turns any picture into a useful vector. Every test that follows asks how good that vector is for a new task.

> [!DEFINITION] Representation
> The vector a model produces for an input, before any task-specific head: for ViT, the 768 numbers of $$\mathbf{y}$$ (Equation 4). A good representation puts pictures of the same kind close together, so that a simple head can separate the classes.

## Datasets {§4.1}

> [!PAPER] · §4.1
> [![Section 4.1, Datasets: to explore model scalability, we use the ILSVRC-2012 ImageNet dataset with 1k classes and 1.3M images, its superset ImageNet-21k with 21k classes and 14M images, and JFT with 18k classes and 303M high-resolution images. We de-duplicate the pre-training datasets with respect to the test sets of the downstream tasks following Kolesnikov et al. (2020)](/img/papers/vit/p3-datasets.png)](/img/papers/vit/p3-datasets.png)
>
> **What it says:** three pre-training sets of growing size: ImageNet ("1k classes and 1.3M images"), ImageNet-21k ("21k classes and 14M images") and JFT ("18k classes and 303M high-resolution images"). "We de-duplicate the pre-training datasets w.r.t. the test sets of the downstream tasks."
>
> **What it means:**
>
> Each step up is 10 to 20 times bigger in pictures: 1.3 million, 14 million, 303 million. That ladder is the paper's main experiment. Train the same model on each rung and see when ViT starts to beat ResNets ([Part 1](part-1-the-big-idea.md) defines the three sets).
>
> De-duplication is a quiet but important step. A collection of 303 million pictures can easily hold copies of test pictures of CIFAR or Pets. A model that saw the test picture in pre-training would get an unfair head start. So near-copies of every downstream test picture are removed from the pre-training data first.

> [!DEFINITION] De-duplication
> Removing near-copies of the test pictures from the training data, so that a test measures what the model learned, not what it memorised. The method follows the BiT paper (Kolesnikov et al., 2020).

{{FIG:p3_datasets|The three pre-training datasets of the paper on a log scale, with our fine-tuning set for comparison. ImageNet: 1.3M pictures, 1k classes. ImageNet-21k: 14M, 21k classes. JFT-300M: 303M, 18k classes. Imagenette, which we use: 9,469 training pictures, 10 classes, 31,999 times fewer than JFT.}}

> [!PAPER] · §4.1
> [![Section 4.1 continued: we transfer the models trained on these dataset to several benchmark tasks: ImageNet on the original validation labels and the cleaned-up ReaL labels (Beyer et al., 2020), CIFAR-10/100 (Krizhevsky, 2009), Oxford-IIIT Pets (Parkhi et al., 2012), and Oxford Flowers-102 (Nilsback and Zisserman, 2008). For these datasets, pre-processing follows Kolesnikov et al. (2020)](/img/papers/vit/p3-transfer-tasks.png)](/img/papers/vit/p3-transfer-tasks.png)
>
> **What it says:** the pre-trained models are transferred to ImageNet (with its original labels and "the cleaned-up ReaL labels"), "CIFAR-10/100", "Oxford-IIIT Pets" and "Oxford Flowers-102". The "pre-processing follows Kolesnikov et al. (2020)".
>
> **What it means:**
>
> These are the downstream tasks of Table 2. There is a big one (ImageNet), two with tiny 32 × 32 pictures (CIFAR) and two fine-grained ones (pet breeds and flower species). Together they test whether one pre-trained body helps on very different problems.
>
> Copying the pre-processing of BiT matters for a fair fight. BiT is the ResNet baseline, so both families see the same crops and the same picture sizes.

> [!DEFINITION] ImageNet ReaL
> "Reassessed Labels" (Beyer et al., 2020): new labels for the ImageNet validation pictures, collected again by people. Some original labels were wrong or too narrow (a picture with a dog and a ball has one label). ReaL allows several correct labels, so it rewards models that are right and were marked wrong.

> [!DEFINITION] CIFAR-10, Oxford-IIIT Pets, Oxford Flowers-102
> Three standard benchmarks. CIFAR-10 has small pictures of 32 × 32 pixels in 10 classes; CIFAR-100 ([Part 1](part-1-the-big-idea.md)) has the same pictures sorted into 100 classes. Pets has 37 breeds of cats and dogs, and Flowers-102 has 102 flower species. Both are small, fine-grained sets.

> [!PAPER] · §4.1
> [![Section 4.1 continued: we also evaluate on the 19-task VTAB classification suite (Zhai et al., 2019b). VTAB evaluates low-data transfer to diverse tasks, using 1 000 training examples per task. The tasks are divided into three groups: Natural, tasks like the above, Pets, CIFAR, etc.; Specialized, medical and satellite imagery; and Structured, tasks that require geometric understanding like localization](/img/papers/vit/p3-vtab.png)](/img/papers/vit/p3-vtab.png)
>
> **What it says:** "the 19-task VTAB classification suite" tests "low-data transfer to diverse tasks, using 1 000 training examples per task", in three groups: "Natural", "Specialized" (medical and satellite pictures) and "Structured" ("geometric understanding like localization").
>
> **What it means:**
>
> VTAB ([Part 1](part-1-the-big-idea.md)) is the hard exam. Each task gives only 1,000 labelled pictures. Some tasks look nothing like photos: a microscope slide, a satellite tile, or a rendered scene that asks how far away an object is.
>
> The three groups matter because a model can be good at one and poor at another. Natural tasks reward knowing objects. Structured tasks reward knowing where things are and how many there are, which a classifier of photos never practised. Figure 2 in Part 4 breaks the score down by group.

## Model variants and the "/16" in the name {§4.1}

> [!PAPER] · §4.1, Model Variants
> [![The Model Variants paragraph: we base ViT configurations on those used for BERT, as summarized in Table 1. The Base and Large models are directly adopted from BERT and we add the larger Huge model. We use brief notation to indicate the model size and the input patch size: for instance, ViT-L/16 means the Large variant with 16 by 16 input patch size. Note that the Transformer's sequence length is inversely proportional to the square of the patch size, thus models with smaller patch size are computationally more expensive](/img/papers/vit/p3-variants.png)](/img/papers/vit/p3-variants.png)
> [![Table 1, details of Vision Transformer model variants: ViT-Base 12 layers, hidden size 768, MLP size 3072, 12 heads, 86M parameters; ViT-Large 24, 1024, 4096, 16, 307M; ViT-Huge 32, 1280, 5120, 16, 632M](/img/papers/vit/p3-table1.png)](/img/papers/vit/p3-table1.png)
>
> **What it says:** "Base" and "Large" are "directly adopted from BERT", and the paper adds "the larger 'Huge' model". "ViT-L/16 means the 'Large' variant with 16×16 input patch size." "The Transformer's sequence length is inversely proportional to the square of the patch size, thus models with smaller patch size are computationally more expensive."
>
> **What it means:**
>
> A ViT name has two parts. The letter is the size of the Transformer (Table 1: Base, Large, Huge). The number is the side of a patch in pixels. ViT-B/32 and ViT-B/16 are the same network, fed differently cut pictures.
>
> "Inversely proportional to the square" is the key to reading every result table. Halve the patch side, and each side of the picture holds twice as many patches, so there are four times as many patches. The weights barely change, but the work grows about four times. A smaller patch is a more expensive model, not a bigger one.

[Part 2](part-2-the-model.md) counted the parameters of each size from a formula. Here we add the patch size and the resolution. `vit_part3_setup.py` computes the sequence length and an estimate of the forward compute from the shapes, and counts the numbers in each released weight file:

```text
== 1. Table 1 and the patch-size notation: ViT-<size>/<patch> ==
  N = (resolution / patch)^2 patches, plus 1 [class] token; attention compares every pair: (N+1)^2 pairs per head
  ViT-B/32 at 224: grid  7x7  N =   49  sequence   50  pairs per head     2,500  forward compute about    8.6 GFLOPs
  ViT-B/32 at 384: grid 12x12 N =  144  sequence  145  pairs per head    21,025  forward compute about   25.4 GFLOPs
  ViT-B/16 at 224: grid 14x14 N =  196  sequence  197  pairs per head    38,809  forward compute about   34.9 GFLOPs
  ViT-B/16 at 384: grid 24x24 N =  576  sequence  577  pairs per head   332,929  forward compute about  110.3 GFLOPs
  ViT-L/32 at 224: grid  7x7  N =   49  sequence   50  pairs per head     2,500  forward compute about   30.4 GFLOPs
  ViT-L/32 at 384: grid 12x12 N =  144  sequence  145  pairs per head    21,025  forward compute about   89.6 GFLOPs
  ViT-L/16 at 224: grid 14x14 N =  196  sequence  197  pairs per head    38,809  forward compute about  122.8 GFLOPs
  ViT-L/16 at 384: grid 24x24 N =  576  sequence  577  pairs per head   332,929  forward compute about  381.2 GFLOPs
  ViT-H/14 at 224: grid 16x16 N =  256  sequence  257  pairs per head    66,049  forward compute about  334.2 GFLOPs
  ViT-H/14 at 384: grid 27x27 N =  729  sequence  730  pairs per head   532,900  forward compute about 1005.9 GFLOPs
  ViT-B/16 vs ViT-B/32 at 224: 196 vs 49 patches = x4 (patch side halved, so (1/2)^-2 = 4); compute x4.06
  Table 2 fine-tuning resolution: ViT-L/16 at 512: grid 32x32, N = 1024, sequence 1025
  Table 2 fine-tuning resolution: ViT-H/14 at 518: grid 37x37, N = 1369, sequence 1370

-- parameters of the released checkpoints, read from the weight files (no head, no pre-logits layer) --
  ViT-B/32 (vit-base-patch32-224-in21k, weight file): backbone  87,455,232  =   87.5M   pre-logits layer   590,592   Table 1: 86M
  ViT-B/16 (vit-base-patch16-224-in21k, weight file): backbone  85,798,656  =   85.8M   pre-logits layer   590,592   Table 1: 86M
  ViT-L/32 (vit-large-patch32-224-in21k, weight file): backbone 305,510,400  =  305.5M   pre-logits layer 1,049,600   Table 1: 307M
  ViT-L/16 (vit-large-patch16-224, weight file): backbone 303,301,632  =  303.3M   pre-logits layer         0   Table 1: 307M
  ViT-H/14 (vit-huge-patch14-224-in21k, weight file): backbone 630,764,800  =  630.8M   pre-logits layer 1,639,680   Table 1: 632M
  ViT-B/32 minus ViT-B/16: 1,656,576 parameters (1.9% more), because E grows from 768x768 to 3072x768 (+1,769,472) and the position table shrinks from 197 to 50 rows (-112,896)
  ViT-L/32 with its pre-logits layer: 306.6M; ViT-L/16 has 303.3M without one
```

Three things stand out. First, ViT-B/16 needs 4.06 times the compute of ViT-B/32 at 224 pixels, almost exactly the factor 4 of the paper's sentence. The extra 0.06 comes from attention, whose cost grows with the square of the sequence.

Second, ViT-B/32 has 1.9% *more* parameters than ViT-B/16, not fewer: its patch embedding is $$3{,}072 \times 768$$ instead of $$768 \times 768$$. Parameters and compute are separate dials.

Third, the 307M of ViT-Large in Table 1 is close to ViT-L/32 with its pre-logits layer (306.6M). ViT-L/16 has 303.3M without one; its only cached file is the ImageNet fine-tuned one, which has no pre-logits layer. Part 2 already flagged this gap. The table may count a different version; we cannot tell which.

{{FIG:p3_seq_len|Patch size and resolution set the work, not the size of the model. For each variant, the bar is the forward compute of one picture in billions of operations (our estimate from the shapes) at 224 and at 384 pixels, with the sequence length written beside it. ViT-B/16 costs about 4 times ViT-B/32; going from 224 to 384 costs about 3 times more again.}}

> [!DEFINITION] FLOPs
> Floating-point operations: the count of additions and multiplications a model performs. One multiply-add counts as 2. "GFLOPs" means billions. Our estimate counts the large matrix products of each layer: $$2 \times (12ND^2 + 2N^2D)$$ for $$N$$ tokens of width $$D$$ with an MLP of $$4D$$.

## The CNN baselines: ResNet (BiT) and Noisy Student {§4.1}

> [!PAPER] · §4.1, Model Variants
> [![For the baseline CNNs, we use ResNet (He et al., 2016), but replace the Batch Normalization layers (Ioffe and Szegedy, 2015) with Group Normalization (Wu and He, 2018), and used standardized convolutions (Qiao et al., 2019). These modifications improve transfer (Kolesnikov et al., 2020), and we denote the modified model ResNet (BiT)](/img/papers/vit/p3-bit-baseline.png)](/img/papers/vit/p3-bit-baseline.png)
>
> **What it says:** the baseline is a ResNet in which the authors "replace the Batch Normalization layers" with "Group Normalization" and use "standardized convolutions". "These modifications improve transfer", and the result is called "ResNet (BiT)".
>
> **What it means:**
>
> A fair race needs a strong opponent. A plain ResNet ([Part 1](part-1-the-big-idea.md)) is not the best CNN for transfer. So the paper uses the version that the BiT paper tuned for exactly this job: pre-train big, fine-tune small.
>
> The two changes are both about normalisation, the step that keeps numbers inside a network in a sane range. The standard version, Batch Norm, depends on the other pictures in the batch. The BiT version depends only on the picture itself. The tiny example below shows the difference.

> [!DEFINITION] Batch Normalization (BN)
> A layer that, for each channel, subtracts the mean and divides by the spread computed over all pictures in the current batch and all pixels. During training, a picture's output depends on which other pictures share its batch. At test time it uses running averages saved during training.

> [!DEFINITION] Group Normalization (GN)
> The same idea, but the mean and spread are computed inside one picture, over a group of channels. A picture's output never depends on the batch, so small batches and fine-tuning do not upset it.

> [!DEFINITION] Weight Standardization (WS)
> Normalising the weights instead of the activations: before a convolution uses a filter, its weights are shifted to mean 0 and scaled to spread 1. The paper calls it "standardized convolutions". It works well together with Group Norm.

The difference fits on a few numbers. Picture A has 4 channels of 2 pixels each. We normalise it once together with one partner picture and once with another:

```text
== 3. Batch Norm vs Group Norm on a tiny batch (4 channels, 2 pixels per channel, 2 groups) ==
  picture A, channel 1 = [1.0, 3.0]
  Batch Norm, A batched with partner 1: A channel 1 -> [-1.260, -0.700]
  Batch Norm, A batched with partner 2: A channel 1 -> [ 0.632,  1.265]   (same picture, different output)
  Group Norm, group 1 = channels 1 and 2 of A = [1.0, 3.0, 2.0, 2.0]: mean 2.000, std 0.707
  Group Norm: A channel 1 -> [-1.414,  1.414] with any partner (it never looks at the batch); torch GroupNorm agrees: True
  Weight Standardization: filter 2 = [1.0, 2.0, 3.0, 2.0]: mean 2.00, std 0.7071 -> [-1.414,  0.000,  1.414,  0.000]
  filter 1 = [0.2, -0.1, 0.5, 0.0] -> [ 0.218, -1.091,  1.528, -0.655]   (every filter now has mean 0 and std 1 before the convolution uses it)
```

With Batch Norm, the same picture A gives $$[-1.260, -0.700]$$ in one batch and $$[0.632, 1.265]$$ in another. Its output depends on a stranger. With Group Norm, channels 1 and 2 of A form a group with values $$1, 3, 2, 2$$: mean 2, spread $$\sqrt{(1 + 1 + 0 + 0)/4} = 0.707$$. So channel 1 becomes $$(1 - 2)/0.707 = -1.414$$ and $$(3 - 2)/0.707 = 1.414$$, whatever the batch. The released `microsoft/resnet-50` that Part 1 ran is a standard ResNet with 53 Batch Norm layers; ResNet (BiT) swaps each of them for Group Norm.

{{FIG:p3_bn_gn|Which numbers are averaged. A batch of two pictures, each with four channels. Batch Norm (left) takes the mean of one channel across both pictures, so picture A's result depends on picture B. Group Norm (right) takes the mean of a group of channels inside one picture. The numbers are the tiny example: A's first channel becomes −1.260, −0.700 or 0.632, 1.265 with Batch Norm, and always −1.414, 1.414 with Group Norm.}}

The BiT paper gives the reason in its own words:

> [!PAPER] Kolesnikov et al. (2020), Big Transfer (BiT) · §2.1
> [![The BiT paper, Section 2.1: the second component is Group Normalization (GN) and Weight Standardization (WS). Batch Normalization (BN) is used in most state-of-the-art vision models to stabilize training. However, we find that BN is detrimental to Big Transfer for two reasons. First, when training large models with small per-device batches, BN performs poorly or incurs inter-device synchronization cost. Second, due to the requirement to update running statistics, BN is detrimental for transfer. GN, when combined with WS, has been shown to improve performance on small-batch training](/img/papers/vit/p3-ext-bit-gnws.png)](/img/papers/vit/p3-ext-bit-gnws.png)
>
> **What it says:** "BN is detrimental to Big Transfer for two reasons." With "small per-device batches" it "performs poorly or incurs inter-device synchronization cost", and the "requirement to update running statistics" hurts transfer. GN with WS fixes both.
>
> **What it means:**
>
> A big model trained on many chips sees only a few pictures per chip. Batch Norm then either averages over too few pictures, or must stop and share its averages across chips, which is slow.
>
> The second reason is about fine-tuning. Batch Norm carries running averages from pre-training, and a new task with different pictures makes them wrong. Group Norm has no running averages to go stale. ViT has the same property: its LayerNorm (Part 2) also works inside one token at a time.

> [!DEFINITION] BiT (Big Transfer)
> A recipe from the same Google team (Kolesnikov et al., 2020). Large ResNets with Group Norm and Weight Standardization are pre-trained on ImageNet-21k (BiT-M) or JFT-300M (BiT-L), then fine-tuned with one fixed rule. BiT-L is the main opponent in Table 2. The paper names ResNets by depth and width: R152x4 is 152 layers, four times wider than usual.

The second opponent appears in Section 4.2 ([Part 4](part-4-results.md)): Noisy Student, which the paper calls the state of the art on ImageNet at the time. It is a different kind of CNN trained in a different way:

> [!PAPER] Xie et al. (2020), Self-training with Noisy Student · Abstract
> [![The abstract of Noisy Student: we present Noisy Student Training, a semi-supervised learning approach that works well even when labeled data is abundant; it achieves 88.4% top-1 accuracy on ImageNet. We first train an EfficientNet model on labeled images and use it as a teacher to generate pseudo labels for 300M unlabeled images. We then train a larger EfficientNet as a student model on the combination of labeled and pseudo labeled images, and iterate. During the learning of the student, we inject noise such as dropout, stochastic depth, and data augmentation via RandAugment](/img/papers/vit/p3-ext-noisy.png)](/img/papers/vit/p3-ext-noisy.png)
>
> **What it says:** "a semi-supervised learning approach" that reaches 88.4% on ImageNet. A teacher EfficientNet "generate[s] pseudo labels for 300M unlabeled images"; a "larger EfficientNet as a student model" learns from both, with noise such as "dropout, stochastic depth" and RandAugment; the student then becomes the next teacher.
>
> **What it means:**
>
> Noisy Student uses the same 300 million JFT pictures as ViT, but without their labels. A teacher model labels them itself, and a bigger student learns from those guesses while being made to work under noise. Repeat a few times, and the student passes the teacher.
>
> So the two opponents of ViT are both strong in different ways. BiT is the fair comparison: same team, same data, same transfer recipe, only the architecture differs. Noisy Student is the record to beat on ImageNet, with a very different method.

> [!DEFINITION] Semi-supervised learning and pseudo labels
> Training with a few labelled pictures and many unlabelled ones. A **pseudo label** is a label that a trained model guesses for an unlabelled picture and that is then used as if it were true.

> [!DEFINITION] EfficientNet
> A family of CNNs (Tan and Le, 2019) whose depth, width and picture size are scaled together by one rule. Noisy Student's largest model is EfficientNet-L2.

## The hybrids: which ResNet stage feeds ViT {§4.1}

> [!PAPER] · §4.1, Model Variants
> [![For the hybrids, we feed the intermediate feature maps into ViT with patch size of one pixel. To experiment with different sequence lengths, we either (i) take the output of stage 4 of a regular ResNet50 or (ii) remove stage 4, place the same number of layers in stage 3 (keeping the total number of layers), and take the output of this extended stage 3. Option (ii) results in a 4x longer sequence length, and a more expensive ViT model](/img/papers/vit/p3-hybrid-stages.png)](/img/papers/vit/p3-hybrid-stages.png)
>
> **What it says:** the hybrids read a CNN feature map "with patch size of one 'pixel'". Either "(i) take the output of stage 4 of a regular ResNet50" or "(ii) remove stage 4, place the same number of layers in stage 3", and "take the output of this extended stage 3". Option (ii) "results in a 4x longer sequence length".
>
> **What it means:**
>
> [Part 2](part-2-the-model.md) defined the hybrid: a ResNet does the first steps of seeing, and each position of its feature map becomes a token. This paragraph says which feature map. A ResNet halves its grid at each stage, so the later the stage, the shorter the sequence.
>
> Option (ii) is a clever trade. Moving stage 4's layers into stage 3 keeps the CNN's depth but skips the last halving. The grid stays at 14 × 14, so the hybrid gets 196 tokens, the same as ViT-B/16. The Transformer part then costs about four times as much as in option (i).

The real ResNet-50 shows the four grids on one Imagenette picture:

```text
== 2. the hybrid: which ResNet-50 stage feeds ViT ==
  stage 1 output:  256 channels x 56 x 56 -> as 1x1 patches:  3136 tokens of 256 numbers
  stage 2 output:  512 channels x 28 x 28 -> as 1x1 patches:   784 tokens of 512 numbers
  stage 3 output: 1024 channels x 14 x 14 -> as 1x1 patches:   196 tokens of 1024 numbers
  stage 4 output: 2048 channels x  7 x 7  -> as 1x1 patches:    49 tokens of 2048 numbers
  option (i), stage 4: 49 tokens; option (ii), extended stage 3: 196 tokens = x4 (the "4x longer sequence")
  microsoft/resnet-50 is a standard ResNet: it has 53 BatchNorm2d layers (BiT replaces each with Group Norm)
```

{{FIG:p3_hybrid|The four stages of ResNet-50 on a 224×224 picture. Each stage halves the grid: 56×56, 28×28, 14×14, 7×7. Option (i) reads stage 4: 49 tokens. Option (ii) moves stage 4's layers into stage 3 and reads its 14×14 grid: 196 tokens, four times longer, as many as ViT-B/16.}}

## Pre-training: Adam, a big batch and strong weight decay {§4.1}

> [!PAPER] · §4.1, Training and Fine-tuning
> [![Training and Fine-tuning: we train all models, including ResNets, using Adam (Kingma and Ba, 2015) with beta1 = 0.9, beta2 = 0.999, a batch size of 4096 and apply a high weight decay of 0.1, which we found to be useful for transfer of all models (Appendix D.1 shows that, in contrast to common practices, Adam works slightly better than SGD for ResNets in our setting). We use a linear learning rate warmup and decay, see Appendix B.1 for details](/img/papers/vit/p3-training.png)](/img/papers/vit/p3-training.png)
>
> **What it says:** all models, "including ResNets", are trained "using Adam" with "$$\beta_1 = 0.9, \beta_2 = 0.999$$, a batch size of 4096" and "a high weight decay of 0.1", which was "useful for transfer of all models". The learning rate follows "a linear learning rate warmup and decay".
>
> **What it means:**
>
> Every model in the race gets the same training recipe, ResNets included. That is unusual: ResNets are normally trained with plain SGD. The authors checked that Adam suits their ResNets slightly better (Appendix D.1, in Part 4), so the baseline is not handicapped.
>
> Each piece of the sentence is a dial. Adam decides how each weight moves. The batch says how many pictures each step averages over. Weight decay pulls every weight a little towards zero, and the warm-up and decay set the step size over time. The boxes below define them, and a tiny example shows Adam's arithmetic.

> [!DEFINITION] Batch size
> How many pictures the model sees before it takes one step. The gradient of each step is the average over the batch. A batch of 4,096 gives a smooth, reliable direction, and spreads well over many chips.

> [!DEFINITION] Learning rate, warm-up and decay
> The **learning rate** scales every step. A **warm-up** starts it at 0 and raises it in a straight line over the first steps. The first gradients of a fresh model are wild. **Decay** lowers it again towards the end, so the last steps are small and careful. "Linear" decay is a straight line down; "cosine" follows half a cosine wave.

> [!DEFINITION] Weight decay
> At every step, shrink each weight a little towards zero. A model must then "pay" to keep a weight large, so it keeps only the weights that really help. It is a form of regularization ([Part 1](part-1-the-big-idea.md)). The paper calls its value of 0.1 "high".

> [!DEFINITION] Adam
> An optimizer (Kingma and Ba, 2015). For every weight it keeps a running average of the gradients (the direction) and of the squared gradients (the size). It divides the first by the square root of the second. Each weight gets its own step size.

**Adam, by hand.** The intuition: a hiker in fog keeps a running sense of which way is downhill ($$m$$), and of how bumpy the ground has been ($$v$$). Steps are long on smooth ground and short on bumpy ground. For one weight $$w$$ with gradient $$g_t$$ at step $$t$$:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\,g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2)\,g_t^2
$$

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}, \qquad w_t = w_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

Here $$m_t$$ is the running average of the gradient with memory $$\beta_1 = 0.9$$, and $$v_t$$ the running average of the squared gradient with memory $$\beta_2 = 0.999$$. $$\hat m_t$$ and $$\hat v_t$$ are the same averages, corrected for starting at 0. $$\eta$$ is the learning rate, and $$\epsilon$$ a tiny number ($$10^{-8}$$) that avoids a division by zero. With three gradients $$0.5, -0.1, 0.3$$ and $$\eta = 0.001$$:

```text
  Adam t=1: g= 0.50  m=0.0500  v=0.000250  m_hat=0.5000  v_hat=0.2500  step = lr*m_hat/sqrt(v_hat) = 0.001000  w=0.999000
  Adam t=2: g=-0.10  m=0.0350  v=0.000260  m_hat=0.1842  v_hat=0.1299  step = lr*m_hat/sqrt(v_hat) = 0.000511  w=0.998489
  Adam t=3: g= 0.30  m=0.0615  v=0.000349  m_hat=0.2269  v_hat=0.1166  step = lr*m_hat/sqrt(v_hat) = 0.000665  w=0.997824
  torch.optim.Adam after the same three gradients: w=0.997824  (ours 0.997824)
```

Check step 1 with a pencil: $$m_1 = 0.1 \times 0.5 = 0.05$$ and $$v_1 = 0.001 \times 0.25 = 0.00025$$. The corrections divide by $$1 - 0.9 = 0.1$$ and $$1 - 0.999 = 0.001$$, giving $$\hat m_1 = 0.5$$ and $$\hat v_1 = 0.25$$. The step is $$0.001 \times 0.5 / \sqrt{0.25} = 0.001$$. The first step of Adam is always exactly $$\eta$$ long, whatever the size of the gradient. That is why a warm-up matters: without it, every weight jumps by the full learning rate at once.

> [!PAPER] · Table 3
> [![Table 3, hyperparameters for training: ViT-B/16 and B/32 on JFT-300M 7 epochs, base LR 8 times 10 to the minus 4, linear decay, weight decay 0.1, dropout 0.0; ViT-L/32 6e-4; ViT-L/16 7 or 14 epochs, 4e-4; ViT-H/14 14 epochs, 3e-4; ResNets and hybrids similar; ViT-B on ImageNet-21k 90 epochs, 1e-3, weight decay 0.03, dropout 0.1; ViT-L on ImageNet-21k 30 or 90 epochs; ViT on ImageNet 300 epochs, 3e-3, cosine decay, weight decay 0.3, dropout 0.1. Caption: all models are trained with a batch size of 4096 and learning rate warmup of 10k steps. For ImageNet we found it beneficial to additionally apply gradient clipping at global norm 1. Training resolution is 224](/img/papers/vit/p3-table3.png)](/img/papers/vit/p3-table3.png)
>
> **What it says:** on JFT-300M every model trains for 7 or 14 epochs with linear decay, weight decay 0.1 and no dropout; the base learning rate falls as models grow ($$8 \cdot 10^{-4}$$ for ViT-B, $$3 \cdot 10^{-4}$$ for ViT-H). On ImageNet-21k: 30 or 90 epochs, weight decay 0.03, dropout 0.1. On ImageNet: 300 epochs, cosine decay, weight decay 0.3, dropout 0.1. All use "a batch size of 4096 and learning rate warmup of 10k steps"; ImageNet adds "gradient clipping at global norm 1".
>
> **What it means:**
>
> Read the table from top to bottom and a pattern appears. The smaller the dataset, the more epochs and the more regularization. JFT is so big that 7 passes are enough and nothing needs to be held back. ImageNet is small for ViT, so the model sees it 300 times, and needs dropout and three times the weight decay to avoid memorising it.
>
> The learning rate shrinks as the model grows. The paper gives no reason. A common one: a bigger model has more weights moving at once, so the same step shakes it more. Note also the single line "Training resolution is 224": all pre-training uses small pictures, and the bigger ones come only at fine-tuning.

> [!DEFINITION] Epoch
> One full pass over the training set. Seven epochs of JFT-300M means each of its 303 million pictures is seen about seven times.

Turned into steps, the table reads like this (`vit_part3_setup.py`; the ImageNet training set is 1.28 million pictures):

```text
  pre-training ViT-B/16 on JFT-300M      :   7 epochs x 303.00M images / batch 4096 =   517,822 steps; warm-up 10,000 =  1.9% of them; base LR 0.0008, linear decay
  pre-training ViT-H/14 on JFT-300M      :  14 epochs x 303.00M images / batch 4096 = 1,035,645 steps; warm-up 10,000 =  1.0% of them; base LR 0.0003, linear decay
  pre-training ViT-B/16 on ImageNet-21k  :  90 epochs x  14.00M images / batch 4096 =   307,617 steps; warm-up 10,000 =  3.3% of them; base LR 0.001, linear decay
  pre-training ViT on ImageNet           : 300 epochs x   1.28M images / batch 4096 =    93,750 steps; warm-up 10,000 = 10.7% of them; base LR 0.003, cosine decay
  ViT-B/16 on JFT: LR at step 5,000 = 4.0e-04 (half way up the warm-up), at 10,000 = 8.0e-04, at the middle of the decay = 4.0e-04
```

The 10,000 warm-up steps are under 2% of a JFT run, but almost 11% of an ImageNet run. One rule fits very different runs. The learning rate itself is a ramp up and a long line down:

{{FIG:p3_lr_schedules|The learning-rate schedules of Table 3 and Table 4, computed from their numbers. Left: pre-training ViT-B/16 on JFT-300M, a linear warm-up to 8·10⁻⁴ over 10,000 steps, then a straight line down over the remaining 507,822 steps (we assume it ends at 0; the paper does not say). Middle: ViT on ImageNet, the same warm-up to 3·10⁻³ and a cosine decay over 93,750 steps. Right: fine-tuning on ImageNet, 20,000 steps of cosine decay from a base of 0.03.}}

> [!PAPER] · Appendix B.1
> [![Appendix B.1, Training: Table 3 summarizes our training setups for our different models. We found strong regularization to be key when training models from scratch on ImageNet. Dropout, when used, is applied after every dense layer except for the qkv-projections and directly after adding positional- to patch embeddings. Hybrid models are trained with the exact setup as their ViT counterparts. Finally, all training is done on resolution 224](/img/papers/vit/p3-b1-training.png)](/img/papers/vit/p3-b1-training.png)
>
> **What it says:** "strong regularization" is "key when training models from scratch on ImageNet". Dropout is "applied after every dense layer except for the the qkv-projections and directly after adding positional- to patch embeddings". Hybrids use "the exact setup as their ViT counterparts", and "all training is done on resolution 224".
>
> **What it means:**
>
> "From scratch on ImageNet" is the case where ViT struggles. With only 1.3 million pictures and few built-in assumptions, it can memorise the training set. Dropout and weight decay make memorising harder.
>
> The exact placement of dropout is given because it matters for reproduction. Every dense layer gets it, except the one that makes queries, keys and values. The input sequence also gets it, right after the position embeddings are added. The released checkpoints set dropout to 0.0 in their configuration, which is what fine-tuning and inference use.

> [!DEFINITION] Dropout
> During training, set a random share of the numbers in a layer's output to zero (10% for a rate of 0.1). Scale the rest up to keep the average. The network cannot rely on any single number, so it learns more robust patterns. At test time dropout is off.

## Fine-tuning: SGD with momentum, cosine decay, clipping {§4.1}

> [!PAPER] · §4.1, Training and Fine-tuning
> [![For fine-tuning we use SGD with momentum, batch size 512, for all models, see Appendix B.1.1. For ImageNet results in Table 2, we fine-tuned at higher resolution: 512 for ViT-L/16 and 518 for ViT-H/14, and also used Polyak and Juditsky (1992) averaging with a factor of 0.9999](/img/papers/vit/p3-finetune-recipe.png)](/img/papers/vit/p3-finetune-recipe.png)
>
> **What it says:** "For fine-tuning we use SGD with momentum, batch size 512, for all models". For the ImageNet results of Table 2, the models are fine-tuned at "512 for ViT-L/16 and 518 for ViT-H/14", with "Polyak & Juditsky (1992) averaging with a factor of 0.9999".
>
> **What it means:**
>
> Pre-training and fine-tuning use different optimizers. Adam is good at finding its way from a random start. Fine-tuning starts from a good place and only needs small, steady steps, and plain SGD with momentum does that with less memory.
>
> The two odd resolutions follow from the patch sizes. 512 is $$32 \times 16$$, so ViT-L/16 gets a 32 × 32 grid of 1,024 patches. 518 is $$37 \times 14$$, so ViT-H/14 gets 37 × 37 = 1,369 patches. Both are chosen to divide evenly by the patch. Polyak averaging, defined below, is the last polish for the headline numbers.

> [!DEFINITION] SGD with momentum
> Stochastic gradient descent: step against the gradient of the current batch. **Momentum** keeps a running "velocity" that adds up past gradients, like a ball rolling downhill. It smooths out the noise of single batches and speeds up travel along steady slopes.

> [!DEFINITION] Polyak averaging
> Keeping a running average of the weights during training and using the average, not the last weights, at the end. The last weights jitter from step to step; their average sits in the middle of the jitter. A factor of 0.9999 means each step moves the average by 0.0001 of the way to the current weights.

**SGD with momentum and Polyak averaging, by hand.** For one weight with velocity $$u$$, momentum $$\mu = 0.9$$ and learning rate $$\eta$$:

$$
u_t = \mu\,u_{t-1} + g_t, \qquad w_t = w_{t-1} - \eta\,u_t, \qquad \bar w_t = \beta\,\bar w_{t-1} + (1 - \beta)\,w_t
$$

where $$u_t$$ is the velocity, a running sum of gradients that forgets 10% per step, and $$\bar w_t$$ is the Polyak average with factor $$\beta$$. The same three gradients as before, with $$\eta = 0.01$$, and an average with $$\beta = 0.9$$ of five weight values:

```text
  SGD momentum 0.9, lr 0.01, t=1: g= 0.50  velocity = 0.9*velocity + g = 0.5000  w = w - lr*velocity = 0.99500
  SGD momentum 0.9, lr 0.01, t=2: g=-0.10  velocity = 0.9*velocity + g = 0.3500  w = w - lr*velocity = 0.99150
  SGD momentum 0.9, lr 0.01, t=3: g= 0.30  velocity = 0.9*velocity + g = 0.6150  w = w - lr*velocity = 0.98535
  torch.optim.SGD(momentum=0.9) after the same gradients: w=0.98535  (ours 0.98535)
  Polyak (exponential) averaging, factor 0.9: weights [1.0, 1.2, 0.9, 1.1, 1.05] -> averages [1.0, 1.02, 1.008, 1.0172, 1.0205]
  a factor of 0.9999 averages over roughly the last 1/(1 - 0.9999) = 10,000 steps
```

At step 2 the gradient is $$-0.1$$, uphill, but the velocity is still $$0.9 \times 0.5 - 0.1 = 0.35$$, so the weight keeps moving down. That is momentum: one contrary batch does not reverse the direction. The Polyak average of the jittering weights $$1.00, 1.20, 0.90, 1.10, 1.05$$ moves only from 1.0 to 1.0205. A factor of 0.9999 averages over roughly the last 10,000 steps, half of an ImageNet fine-tuning run of 20,000 steps.

> [!PAPER] · Table 4
> [![Table 4, hyperparameters for fine-tuning: ImageNet 20 000 steps, base LR in 0.003, 0.01, 0.03, 0.06; CIFAR100 and CIFAR10 10 000 steps, base LR in 0.001, 0.003, 0.01, 0.03; Oxford-IIIT Pets and Oxford Flowers-102 500 steps, same range; VTAB (19 tasks) 2 500 steps, 0.01. Caption: all models are fine-tuned with cosine learning rate decay, a batch size of 512, no weight decay, and grad clipping at global norm 1. If not mentioned otherwise, fine-tuning resolution is 384](/img/papers/vit/p3-table4.png)](/img/papers/vit/p3-table4.png)
>
> **What it says:** ImageNet is fine-tuned for 20,000 steps, CIFAR for 10,000, Pets and Flowers for 500, each VTAB task for 2,500 at a learning rate of 0.01. All runs use "cosine learning rate decay, a batch size of 512, no weight decay, and grad clipping at global norm 1". The usual "fine-tuning resolution is 384".
>
> **What it means:**
>
> Fine-tuning is short. Pre-training ViT-B/16 on JFT took about 518,000 steps of 4,096 pictures; fine-tuning on Pets takes 500 steps of 512. The body already knows how to see. Fine-tuning only points it at a new question.
>
> "No weight decay" is a deliberate change from pre-training. Weight decay pulls weights towards zero, and in fine-tuning that means pulling them away from the pre-trained values that make the model good. Clipping, the last item, guards against single bad steps.

> [!DEFINITION] Gradient clipping at global norm 1
> Before each step, measure the length of the whole gradient, all weights together, as one long vector. If it is longer than 1, shrink every number by the same factor so the length becomes 1. The direction stays; only an unusually large step is cut down.

**Cosine decay and clipping, in numbers.** With base learning rate $$\eta_0$$ and $$T$$ steps, cosine decay sets step $$t$$ to

$$
\eta_t = \eta_0 \cdot \tfrac{1}{2}\left(1 + \cos\frac{\pi t}{T}\right)
$$

which is $$\eta_0$$ at $$t = 0$$, $$\eta_0/2$$ halfway, and 0 at the end. For ImageNet with $$\eta_0 = 0.03$$ and $$T = 20{,}000$$: 0.03, then 0.015 at step 10,000, 0.0044 at step 15,000, and 0 at step 20,000. Clipping uses the global norm $$\lVert g \rVert = \sqrt{\sum_i g_i^2}$$ over all weights, and multiplies the gradient by $$\min(1, 1/\lVert g \rVert)$$. With two tensors of gradients $$[3, 0]$$ and $$[4]$$:

```text
  clipping at global norm 1: gradients of two tensors [3, 0] and [4]; global norm sqrt(9 + 0 + 16) = 5.0 > 1, so every gradient is multiplied by 1/5: [0.6, 0.0] and [0.8]; direction kept, length 1
  torch.nn.utils.clip_grad_norm_: [0.6, 0.0] and [0.8]
```

The length is $$\sqrt{9 + 0 + 16} = 5$$, so every number is divided by 5: $$[0.6, 0]$$ and $$[0.8]$$, a vector of length 1 that points the same way. Table 4 in steps and passes over the data:

```text
  fine-tuning ImageNet           : 20,000 steps x batch 512 =  10.24M pictures seen =     8.0 passes over 1,280,000 training pictures
  fine-tuning CIFAR-100          : 10,000 steps x batch 512 =   5.12M pictures seen =   102.4 passes over 50,000 training pictures
  fine-tuning CIFAR-10           : 10,000 steps x batch 512 =   5.12M pictures seen =   102.4 passes over 50,000 training pictures
  fine-tuning Oxford-IIIT Pets   :    500 steps x batch 512 =   0.26M pictures seen =    69.6 passes over 3,680 training pictures
  fine-tuning Oxford Flowers-102 :    500 steps x batch 512 =   0.26M pictures seen =   125.5 passes over 2,040 training pictures
  fine-tuning VTAB task          :  2,500 steps x batch 512 =   1.28M pictures seen =  1280.0 passes over 1,000 training pictures
  ImageNet fine-tuning at base LR 0.03, cosine: step 0 -> 0.03, step 10,000 -> 0.015, step 15,000 -> 0.0044, step 20,000 -> 0
```

These counts use the standard training-set sizes of each benchmark. Pets is seen about 70 times, and each 1,000-example VTAB task 1,280 times. Small tasks get many passes but few steps.

> [!PAPER] · Appendix B.1.1
> [![Appendix B.1.1, Fine-tuning: we fine-tune all ViT models using SGD with a momentum of 0.9. We run a small grid search over learning rates, see learning rate ranges in Table 4. To do so, we use small sub-splits from the training set (10% for Pets and Flowers, 2% for CIFAR, 1% ImageNet) as development set and train on the remaining data. For final results we train on the entire training set and evaluate on the respective test data. For fine-tuning ResNets and hybrid models we use the exact same setup, with the only exception of ImageNet where we add another value 0.06 to the learning rate sweep](/img/papers/vit/p3-b11-sweep.png)](/img/papers/vit/p3-b11-sweep.png)
>
> **What it says:** "SGD with a momentum of 0.9", and "a small grid search over learning rates". To choose, they hold out "10% for Pets and Flowers, 2% for CIFAR, 1% ImageNet" of the training set as a development set. "For final results we train on the entire training set". ResNets and hybrids get the same setup, plus a learning rate of 0.06 on ImageNet.
>
> **What it means:**
>
> Four learning rates are tried for each task, and the winner is chosen on held-out training pictures, never on the test set. Picking on the test set would be like a student who sees the exam before choosing how to study.
>
> After the choice, the held-out pictures go back in, and the final model trains on everything. The extra 0.06 for ResNets on ImageNet is a small gift to the baseline. If anything, it tilts the race against ViT.

> [!DEFINITION] Development set and grid search
> A **development set** (or validation set) is a slice of the training data kept aside to choose settings. A **grid search** tries every value in a short list, here four learning rates, and keeps the one that does best on the development set.

> [!PAPER] · Appendix B.1.1
> [![Appendix B.1.1 continued: for ResNets we also run the setup of Kolesnikov et al. (2020) and select the best results across this run and our sweep. Finally, if not mentioned otherwise, all fine-tuning experiments run at 384 resolution (running fine-tuning at different resolution than training is common practice (Kolesnikov et al., 2020))](/img/papers/vit/p3-b11-res.png)](/img/papers/vit/p3-b11-res.png)
>
> **What it says:** ResNets get BiT's own fine-tuning setup as well, and "the best results across this run and our sweep" count. "All fine-tuning experiments run at 384 resolution", which "is common practice".
>
> **What it means:**
>
> The baseline gets two chances: the paper's recipe and its own original recipe, whichever is better. This is how a careful paper protects itself from the charge "you tuned your model and not the opponent".
>
> The 384 rule ties back to Section 3.2. Unless the paper says otherwise, every ViT fine-tuning number comes from a model whose position table was stretched first. For a /16 model, that is 14 × 14 to 24 × 24.

> [!PAPER] · Appendix B.1.1
> [![Appendix B.1.1, VTAB: for VTAB we follow the protocol in Kolesnikov et al. (2020), and use the same hyperparameter setting for all tasks. We use a learning rate of 0.01 and train for 2500 steps. We chose this setting by running a small sweep over two learning rates and two schedules, and selecting the setting with the highest VTAB score on the 200-example validation sets. We follow the pre-processing used in Kolesnikov et al. (2020), except that we do not use task-specific input resolutions. Instead we find that Vision Transformer benefits most from a high resolution (384 by 384) for all tasks](/img/papers/vit/p3-b11-vtab.png)](/img/papers/vit/p3-b11-vtab.png)
>
> **What it says:** VTAB uses "the same hyperparameter setting for all tasks": "a learning rate of 0.01 and train for 2500 steps", chosen on "200-example validation sets". Unlike BiT, no task-specific picture sizes: ViT "benefits most from a high resolution (384 × 384) for all tasks".
>
> **What it means:**
>
> On VTAB, tuning each of 19 tasks separately would be both costly and unfair, since each task has only 1,000 pictures. So one setting serves all 19, chosen once on small validation slices of 200 pictures.
>
> The last sentence is a finding, not only a setting. BiT picks the picture size task by task. ViT does best at 384 on every task, so it needs no such rule.

### Our fine-tuning run

The paper's fine-tuning runs are far beyond one computer: 20,000 steps of 512 pictures at 384 pixels. We kept the pieces of the recipe and shrank the scale. We start from the ImageNet-21k checkpoint of ViT-B/16 with the zero head shown above. We fine-tune it on **Imagenette**, a public set of 10 easy ImageNet classes. They are tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball and parachute. It has 9,469 training and 3,925 validation pictures, stored at 160 pixels on the short side.

`vit_part3_finetune.py` runs 300 steps of 32 pictures at 224 pixels, so the model sees 9,600 pictures, about one pass over the training set. The pieces follow Table 4: SGD with momentum 0.9, no weight decay, cosine decay and clipping at global norm 1. We also keep a running average of the weights.

Three choices are ours. The batch is 32 instead of 512. There is a short linear warm-up of 30 steps, which Table 4 does not list. And the learning rate is 0.003, a value from Table 4's grid, picked without a search.

```python
opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0)
for step in range(1, STEPS + 1):
    for g in opt.param_groups:
        g['lr'] = lr_at(step - 1)                               # warm-up, then cosine to 0
    loss = F.cross_entropy(model(pixel_values=x).logits, y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # global norm 1
    opt.step()
    for pe, pm in zip(ema.parameters(), model.parameters()):   # Polyak average, factor 0.99
        pe.mul_(0.99).add_(pm.detach(), alpha=0.01)
```

```text
== 2. fine-tuning: 300 steps, batch 32, SGD momentum 0.9, no weight decay, base LR 0.003, linear warm-up 30 steps, cosine decay to 0, clip at global norm 1.0, weight average factor 0.99, random horizontal flips ==
  step   0: validation accuracy (1,000 pictures, 100 per class) 10.0%  (zero head: every class ties, the first wins)
  step   5: lr 0.00040  loss 2.3014  gradient norm before clipping   1.151  validation  98.3%   [39 s]
  step  10: lr 0.00090  loss 2.2914  gradient norm before clipping   1.197  validation  99.0%   [76 s]
  step  25: lr 0.00240  loss 2.1605  gradient norm before clipping   1.123  validation  99.0%   [137 s]
  step  50: lr 0.00296  loss 1.6761  gradient norm before clipping   1.280  validation  99.1%   [220 s]
  step 100: lr 0.00254  loss 0.9975  gradient norm before clipping   1.655  validation  99.4%   [359 s]
  step 150: lr 0.00178  loss 0.6214  gradient norm before clipping   1.523  validation  99.6%   [494 s]
  step 200: lr 0.00092  loss 0.4379  gradient norm before clipping   1.217  validation  99.7%   [599 s]
  step 250: lr 0.00026  loss 0.4833  gradient norm before clipping   1.825  validation  99.7%   [718 s]
  step 300: lr 0.00000  loss 0.3019  gradient norm before clipping   1.104  validation  99.7%   [850 s]
  clipping was active in 292 of 300 steps; median gradient norm: first 50 steps 1.197, last 50 steps 1.393
  mean training loss: steps 1 to 10 2.299, steps 291 to 300 0.350; pictures seen 9,600 = 1.01 passes over the training set
```

Read the loss and the accuracy side by side. After only 5 steps, validation accuracy jumps from 10.0% to 98.3%, while the loss has barely moved from $$\ln 10$$ (2.3014). Accuracy only asks whether the right class has the highest score. The loss asks how sure the model is. After 5 small steps the head already points the right way, but its scores are still small, since the loss is still close to $$\ln 10$$. The other 295 steps mostly make the model confident: the mean loss of the last 10 steps is 0.350.

{{FIG:p3_finetune_curve|Our fine-tuning run of ViT-B/16 (ImageNet-21k weights) on Imagenette, 300 steps of 32 pictures. Left: the training loss of every step, from ln 10 down to about 0.3, with the learning rate dashed (30 warm-up steps, then cosine to 0). Right: accuracy on 1,000 validation pictures, 10.0% at step 0, 98.3% at step 5 and 99.7% at the end.}}

Two numbers deserve a comment. Clipping was active in 292 of the 300 steps. The median gradient norm was 1.197 over the first 50 steps and 1.393 over the last 50, above the cap of 1. At our batch of 32, the cap of 1 works more like a fixed step length than like a rare guard. A batch of 512 averages more pictures, so its gradient is less noisy and may hit the cap less often; we did not measure that.

The weight average with factor 0.99 ended at 99.52%, against 99.57% for the last weights. On a task this easy, the average has no jitter to remove.

What this run cannot show: Imagenette is tiny and easy. Its ten classes are among the labels of ImageNet-21k, so the body has seen them before. We used one run, one learning rate and 224 pixels instead of 384. The paper's fine-tuning numbers come from much harder tasks, many runs and a learning-rate search. Our run shows the mechanics of the recipe, not its value.

## Metrics: fine-tuning accuracy and few-shot accuracy {§4.1}

> [!PAPER] · §4.1, Metrics
> [![Metrics: we report results on downstream datasets either through few-shot or fine-tuning accuracy. Fine-tuning accuracies capture the performance of each model after fine-tuning it on the respective dataset. Few-shot accuracies are obtained by solving a regularized least-squares regression problem that maps the (frozen) representation of a subset of training images to minus one, one to the K target vectors. This formulation allows us to recover the exact solution in closed form. Though we mainly focus on fine-tuning performance, we sometimes use linear few-shot accuracies for fast on-the-fly evaluation where fine-tuning would be too costly](/img/papers/vit/p3-metrics.png)](/img/papers/vit/p3-metrics.png)
>
> **What it says:** results are "few-shot or fine-tuning accuracy". Few-shot accuracy comes from "a regularized least-squares regression problem that maps the (frozen) representation of a subset of training images to $$\{-1, 1\}^K$$ target vectors", which has an "exact solution in closed form". It is used "for fast on-the-fly evaluation where fine-tuning would be too costly".
>
> **What it means:**
>
> Fine-tuning accuracy is the real exam: adapt the whole model, then test it. It costs a full training run per model and per task. Few-shot accuracy is the quick quiz. Freeze the model, take a handful of labelled pictures per class, and fit one linear layer on their vectors.
>
> The clever part is how the layer is fitted. There are no gradient steps. The paper asks for scores of +1 for the right class and −1 for the others, and solves for the best layer in one formula. No learning rate, no epochs, just one solve of a linear system. That is why the paper can use it "on the fly", for the many models of Figure 4 and the appendix.

> [!DEFINITION] Few-shot
> Learning a task from very few labelled examples. "5-shot" means five labelled pictures per class. With 10 classes, a 5-shot task has 50 training pictures in total.

> [!DEFINITION] Frozen
> Not changed by training. A frozen model is only run forward; its weights stay as they were after pre-training.

> [!DEFINITION] Least squares and regularization (ridge regression)
> Least squares picks the weights that make the squared errors between predictions and targets as small as possible. Adding $$\lambda$$ times the squared size of the weights keeps the weights small ("regularized", also called ridge regression). That matters when there are fewer examples than features.

**The closed form.** The intuition: draw the best straight line through the training points. "Best" means the smallest total squared miss, plus a small fine for steep lines. For the few-shot layer:

$$
W^{*} = \arg\min_{W}\ \lVert XW - Y \rVert^2 + \lambda \lVert W \rVert^2 = (X^\top X + \lambda I)^{-1} X^\top Y
$$

where:

- $$X$$ has one row per training picture: its frozen representation of $$d$$ numbers, plus a 1 at the end for a bias. So it is $$n \times (d + 1)$$;
- $$Y$$ is $$n \times K$$: row $$i$$ holds +1 in the column of picture $$i$$'s class and −1 in the other $$K - 1$$ columns;
- $$W$$ is the linear layer, $$(d + 1) \times K$$; $$XW$$ are the predicted scores;
- $$\lambda \geq 0$$ sets the fine for large weights, and $$I$$ is the identity matrix;
- $$\lVert \cdot \rVert^2$$ is the sum of squares of all entries.

A new picture with representation $$x$$ gets the $$K$$ scores $$xW$$, and the largest score names its class. The formula comes from one line of calculus: set the gradient $$2X^\top(XW - Y) + 2\lambda W$$ to zero, and you get $$(X^\top X + \lambda I)\,W = X^\top Y$$. Solving that linear system is all the "training" there is.

**A tiny example by hand** ([`vit_part3_fewshot.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part3_fewshot.py)). Four pictures have two features each, two of class A and two of class B. We use $$\lambda = 1$$ and no bias column, to keep the sums short.

```text
== 1. the closed form by hand: 4 pictures, 2 features, K = 2 classes, lambda = 1 ==
  X (rows = pictures A1, A2, B1, B2) = [[2.0, 1.0], [1.0, 0.0], [0.0, 2.0], [1.0, 3.0]]
  Y (targets in {-1, 1}: class A = [1, -1], class B = [-1, 1]) = [[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]]
  X^T X = [[6.0, 5.0], [5.0, 14.0]]   + lambda I = [[7.0, 5.0], [5.0, 15.0]]   det = 80
  (X^T X + I)^-1 = [[0.1875, -0.0625], [-0.0625, 0.0875]]
  X^T Y = [[2.0, -2.0], [-4.0, 4.0]]
  W = (X^T X + I)^-1 X^T Y = [[0.625, -0.625], [-0.475, 0.475]]
  new picture x = [1.5, 0.5]: scores x W = [0.7, -0.7] -> class A
  training scores X W = [[0.775, -0.775], [0.625, -0.625], [-0.95, 0.95], [-0.8, 0.8]] (all on the right side of 0)
```

Check it with a pencil. The top-left entry of $$X^\top X$$ is $$2^2 + 1^2 + 0^2 + 1^2 = 6$$, and the off-diagonal entry is $$2 \cdot 1 + 1 \cdot 0 + 0 \cdot 2 + 1 \cdot 3 = 5$$. Adding $$\lambda = 1$$ on the diagonal gives $$\begin{bmatrix} 7 & 5 \\ 5 & 15 \end{bmatrix}$$, with determinant $$7 \cdot 15 - 5 \cdot 5 = 80$$, so its inverse is $$\tfrac{1}{80}\begin{bmatrix} 15 & -5 \\ -5 & 7 \end{bmatrix}$$. The first column of $$X^\top Y$$ is $$(2 + 1 - 0 - 1,\ 1 + 0 - 2 - 3) = (2, -4)$$. So the first column of $$W$$ is

$$
\tfrac{1}{80}\,(15 \cdot 2 - 5 \cdot (-4),\ -5 \cdot 2 + 7 \cdot (-4)) = \tfrac{1}{80}\,(50, -38) = (0.625, -0.475)
$$

The new picture $$(1.5, 0.5)$$ scores $$1.5 \times 0.625 + 0.5 \times (-0.475) = 0.7$$ for class A and $$-0.7$$ for class B, so it is class A. With two classes, the second column is just the first with its sign flipped. With $$K$$ classes, each column is its own "this class against all the others" line.

{{FIG:p3_ridge_tiny|The tiny example as a picture. Two class-A pictures (blue) and two class-B pictures (green) in a plane of two features. The orange line is where the two class scores of W are equal; the new picture (1.5, 0.5) falls on the class-A side with scores 0.7 and −0.7.}}

**The real thing.** We froze ViT-B/16 with its ImageNet-21k weights, never trained on Imagenette. We computed the representation of 500 training pictures (a pool of 50 per class) and of all 3,925 validation pictures. Then we drew 1, 5 or 10 pictures per class from the pool, five times each, and solved the formula above:

```python
X = np.hstack([features[idx], np.ones((len(idx), 1))])     # n x 769: 768 features and a bias
Y = -np.ones((len(idx), 10)); Y[np.arange(len(idx)), labels[idx]] = 1
W = np.linalg.solve(X.T @ X + lam * np.eye(769), X.T @ Y)  # the closed form
pred = (np.hstack([val_features, ones]) @ W).argmax(1)
```

```text
== 2. frozen ViT-B/16 (ImageNet-21k weights), device mps: features of 500 training pictures (a pool of 50 per class) and 3925 validation pictures ==
  representation "pre-logits": the tanh layer on the [class] token, 768 numbers;  "class token": the final LayerNorm output (Eq. 4)
  accuracy on the 3,925 validation pictures, mean +- std over 5 random draws of the training pictures
  pre-logits  lambda   0.01:  1-shot  98.0 +-  0.6%   5-shot  99.0 +-  0.2%   10-shot  99.0 +-  0.1%
  pre-logits  lambda      1:  1-shot  98.0 +-  0.6%   5-shot  99.1 +-  0.1%   10-shot  99.0 +-  0.1%
  pre-logits  lambda    100:  1-shot  98.2 +-  0.5%   5-shot  99.1 +-  0.2%   10-shot  99.2 +-  0.1%
  class token lambda   0.01:  1-shot  98.1 +-  0.5%   5-shot  99.0 +-  0.1%   10-shot  99.0 +-  0.0%
  class token lambda      1:  1-shot  98.1 +-  0.5%   5-shot  99.1 +-  0.1%   10-shot  99.0 +-  0.1%
  class token lambda    100:  1-shot  98.3 +-  0.3%   5-shot  99.2 +-  0.1%   10-shot  99.2 +-  0.1%
  chance is 10%; 1-shot means 10 training pictures in total, one per class
  5-shot, draw 0, pre-logits, lambda 1: X is (50, 769) (50 pictures x 768 features + bias), Y is (50, 10), W is (769, 10); one solve of a 769x769 system
  per-class accuracy (5-shot, draw 0): tench 100%, springer 100%, cassette player 99%, chain saw 96%, church 99%, French horn 99%, garbage truck 99%, gas pump 99%, golf ball 99%, parachute 99%
  scores of the first validation picture (a tench): +0.92 -0.96 -1.01 -1.04 -1.02 -0.97 -0.97 -0.94 -0.95 -1.02  -> class tench
```

One picture per class, ten pictures in all, already gives 98.0% on 3,925 new pictures. Five per class gives 99.1%. The value of $$\lambda$$ hardly matters here (98.0% to 98.3% at 1-shot), and the two choices of representation agree within 0.2 points. The scores of one tench picture show the targets at work: +0.92 for tench and about −1 for every other class.

{{FIG:p3_fewshot_scores|The ten scores that the 5-shot layer gives one new tench picture. It was fitted to targets of +1 for the right class and −1 for the others, and on this unseen picture it gives +0.92 for tench and between −0.94 and −1.04 for the other nine classes.}}

Do not read these numbers as "ViT learns anything from one picture". Imagenette is easy: ten very different objects, and all ten are ImageNet classes, so the ImageNet-21k pre-training already saw labelled pictures of each of them. The test shows something narrower and still useful.

A frozen representation that already separates the classes needs only a linear layer and a handful of examples. The closed form finds that layer in one solve. The paper uses the same metric on harder tasks, where models differ more. [Part 4](part-4-results.md) runs it across model sizes.

Few-shot and fine-tuning measure different things. Fine-tuning asks how good the model can become on a task; few-shot asks how good its frozen vectors already are. The paper reports fine-tuning in its main tables. It uses few-shot where it must test many models, because one solve of a linear system is far cheaper than thousands of training steps.

> [!TAKEAWAYS] Key takeaways
> - Fine-tuning removes the whole pre-training head (the pre-logits layer and the class layer) and attaches one **zero-initialised** $$D \times K$$ layer. At step 0 every score is 0 and the loss is exactly $$\ln 10$$ for 10 classes. The body gets a gradient of exactly 0, so only the head learns at first.
> - ViT is fine-tuned at a **higher resolution** (usually 384) with the same patch size. For ViT-B/16 the sequence grows from 197 to 577 tokens, and the forward compute from about 34.9 to 110.3 billion operations.
> - The 196 learned position rows are **stretched in 2D** from 14 × 14 to 24 × 24 (the official code: linear interpolation; the library: bicubic). On Imagenette at 384 with no training, the 2D tables kept 84.76% and 84.97%, a 1D stretch fell to 48.38%, and no positions to 13.99%.
> - Patch cutting and this stretch are the **only places** where the paper hands ViT the 2D structure of pictures.
> - Pre-training data: ImageNet (1.3M pictures), ImageNet-21k (14M), JFT-300M (303M), each de-duplicated against the test sets. Downstream: ImageNet and ReaL, CIFAR-10/100, Pets, Flowers-102 and the 19 VTAB tasks.
> - **ViT-L/16** means the Large size with 16 × 16 patches. Halving the patch side gives 4 times the tokens: ViT-B/16 costs 4.06 times ViT-B/32 while having 1.9% fewer parameters.
> - The CNN baseline is **ResNet (BiT)**: Group Norm and Weight Standardization instead of Batch Norm, so a picture's output never depends on its batch. Noisy Student, a semi-supervised EfficientNet, held the ImageNet state of the art.
> - Pre-training: **Adam** (β₁ = 0.9, β₂ = 0.999), batch 4,096, weight decay 0.1 on JFT, 10,000 warm-up steps, linear decay, all at 224 pixels. Fine-tuning: **SGD with momentum** 0.9, batch 512, cosine decay, no weight decay, clipping at global norm 1, a learning rate picked on a held-out slice.
> - **Few-shot accuracy** fits one linear layer to targets in $$\{-1, 1\}^K$$ with the closed form $$W = (X^\top X + \lambda I)^{-1} X^\top Y$$. On frozen ViT-B/16 features, one Imagenette picture per class gave 98.0%, an easy task that the pre-training already knew.

**Next, in Part 4:** the results. Table 2 pits ViT-H/14 and ViT-L/16 against BiT-L and Noisy Student, and Figure 2 breaks down VTAB by group. Figures 3 and 4 answer the central question: how much data does ViT need to beat a ResNet? Continue to [Part 4](part-4-results.md).

<details>
<summary>Run it yourself</summary>

Every number in this part comes from four scripts in [`code/papers/vit/`](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vit). `vit_part3_setup.py` does the arithmetic of Tables 1, 3 and 4, the optimizer steps and the normalisation example. It also runs ResNet-50 once (about a minute on a CPU). `vit_part3_posemb.py` stretches the position table and scores five versions of it on Imagenette. `vit_part3_fewshot.py` solves the few-shot problem on frozen features.

`vit_part3_finetune.py` fine-tunes ViT-B/16 for 300 steps. The last three use the Imagenette set (imagenette2-160, from the fast.ai datasets) and a GPU if there is one.

```bash
pip install torch transformers safetensors scipy pillow
python vit_part3_setup.py
python vit_part3_posemb.py
python vit_part3_fewshot.py
python vit_part3_finetune.py
```

<figure><img src="/img/papers/vit/part3_setup-run.png" alt="Terminal output of vit_part3_setup.py: sequence lengths and compute for each ViT variant, parameter counts from the weight files, the ResNet-50 stage shapes, the Batch Norm and Group Norm example, the schedules of Tables 3 and 4, and Adam, SGD, clipping and Polyak averaging by hand" loading="lazy" /><figcaption>The real output of vit_part3_setup.py (first 60 lines).</figcaption></figure>

<figure><img src="/img/papers/vit/part3_posemb-run.png" alt="Terminal output of vit_part3_posemb.py: the 2 by 2 to 3 by 3 example, the 14 by 14 to 24 by 24 stretch checked against SciPy and the library, the similarity numbers and the accuracy of five position tables on Imagenette" loading="lazy" /><figcaption>The real output of vit_part3_posemb.py.</figcaption></figure>

<figure><img src="/img/papers/vit/part3_fewshot-run.png" alt="Terminal output of vit_part3_fewshot.py: the closed-form example by hand, and the 1, 5 and 10-shot accuracies of frozen ViT-B/16 for three values of lambda and two representations" loading="lazy" /><figcaption>The real output of vit_part3_fewshot.py.</figcaption></figure>

<figure><img src="/img/papers/vit/part3_finetune-run.png" alt="Terminal output of vit_part3_finetune.py: the zero head at step 0 with the loss ln 10 and no gradient in the body, the random head for comparison, and the 300-step fine-tuning run with loss, learning rate, gradient norm and validation accuracy" loading="lazy" /><figcaption>The real output of vit_part3_finetune.py.</figcaption></figure>

</details>

## The paper and the code

[The paper (arXiv 2010.11929)](https://arxiv.org/abs/2010.11929) · [Official code and checkpoints](https://github.com/google-research/vision_transformer) · [Our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vit)
