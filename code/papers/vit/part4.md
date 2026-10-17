---
title: "Results: Beating Big CNNs, and How Much Data It Takes"
description: "Sections 4.2 and 4.3 with their tables and appendix: how ViT compares with the best convolutional networks of 2020, and how much pre-training data it needs before it wins. We recompute every table and test seven released models on two of the paper's datasets."
part: 4
covers: "§4.2, §4.3, Table 2, Figures 2 to 4, Appendix C (Table 5), Appendix D.1, D.9, D.10"
date: 2026-10-17
tags: [vit, vision-transformer, transformers, computer-vision, transfer-learning, benchmarks]
---

[Part 3](part-3-fine-tuning-and-setup.md) set up the experiments: the datasets, the model sizes, the ResNet baselines and the training recipe. Now the paper puts ViT against the strongest convolutional networks of 2020. Then it asks the question behind the whole paper: how much data does a Transformer need before it wins?

## Comparison to the state of the art {§4.2}

> [!PAPER] · §4.2
> [![The opening of Section 4.2, Comparison to State of the Art: we first compare our largest models, ViT-H/14 and ViT-L/16, to state-of-the-art CNNs from the literature. The first comparison point is Big Transfer (BiT), which performs supervised transfer learning with large ResNets. The second is Noisy Student, which is a large EfficientNet trained using semi-supervised learning on ImageNet and JFT-300M with the labels removed. Currently, Noisy Student is the state of the art on ImageNet](/img/papers/vit/p4-sota-intro.png)](/img/papers/vit/p4-sota-intro.png)
>
> **What it says:** the paper compares "our largest models, ViT-H/14 and ViT-L/16" with "state-of-the-art CNNs from the literature". The first rival is "Big Transfer (BiT)", which "performs supervised transfer learning with large ResNets". The second is "Noisy Student", "a large EfficientNet trained using semi-supervised learning on ImageNet and JFT-300M with the labels removed". Noisy Student holds "the state of the art on ImageNet", and BiT-L holds it on the other datasets.
>
> **What it means:**
>
> A new design has to beat the record holders, not an average network. So the paper picks the two strongest networks of 2020, one for each kind of record. Noisy Student holds the ImageNet record. BiT-L holds the records on the smaller transfer benchmarks.
>
> The two rivals play different roles. BiT comes from the same team, uses the same JFT-300M pictures and the same "pre-train, then fine-tune" recipe. Only the network differs: a ResNet against a Transformer. That makes BiT the closest thing to a controlled experiment in this table.
>
> Noisy Student is a fair rival in a different way. It learns with a trick that ViT does not use: labels that a teacher network invents. So it shows where the field stood with every trick available. If ViT matches it with plain supervised training, the result is strong.

The model names come from Table 1 and [Part 3](part-3-fine-tuning-and-setup.md). **ViT-L/16** is the Large model (24 layers, 307M parameters in Table 1) with 16×16 patches. **ViT-H/14** is the Huge model (32 layers, 632M parameters) with 14×14 patches. [Part 1](part-1-the-big-idea.md) defined state of the art and the two pre-training sets, ImageNet-21k (14 million pictures) and JFT-300M (303 million).

> [!DEFINITION] BiT (Big Transfer)
> A family of ResNets pre-trained on a large labelled set and then fine-tuned on each task (Kolesnikov et al., 2020). **BiT-S** is pre-trained on ImageNet, **BiT-M** on ImageNet-21k and **BiT-L** on JFT-300M. A name like **ResNet152x4** means 152 layers, with every layer 4 times wider than in the standard ResNet-152.

> [!DEFINITION] EfficientNet
> A family of CNNs (Tan and Le, 2019) whose depth, width and input resolution grow together by a fixed rule. EfficientNet-L2 is a very large member of the family, used by Noisy Student.

> [!DEFINITION] Semi-supervised learning
> Learning from a few labelled pictures plus many unlabelled ones. Noisy Student uses the 300 million JFT pictures without their labels. A trained network (the "teacher") guesses a label for each one, and those guesses become training targets.

Here is how each rival describes itself. First BiT, in its own abstract:

> [!PAPER] Kolesnikov et al. (2020), Big Transfer (BiT) · Abstract
> [![The abstract of the Big Transfer paper: we revisit the paradigm of pre-training on large supervised datasets and fine-tuning the model on a target task. BiT performs well across a surprisingly wide range of data regimes, from 1 example per class to 1 M total examples. BiT achieves 87.5% top-1 accuracy on ILSVRC-2012, 99.4% on CIFAR-10, and 76.3% on the 19 task Visual Task Adaptation Benchmark (VTAB)](/img/papers/vit/p4-ext-bit.png)](/img/papers/vit/p4-ext-bit.png)
>
> **What it says:** BiT revisits "pre-training on large supervised datasets and fine-tuning the model on a target task". It works "from 1 example per class to 1 M total examples", and reaches "87.5% top-1 accuracy on ILSVRC-2012" (ImageNet) and "76.3% on the 19 task Visual Task Adaptation Benchmark (VTAB)".
>
> **What it means:**
>
> BiT is the recipe that ViT borrows, applied to ResNets. The two numbers in the abstract are the BiT-L column of the ViT paper's Table 2, rounded: 87.54 on ImageNet and 76.29 on VTAB.
>
> Note the phrase "from 1 example per class". BiT already tested models with very few labelled examples. The ViT paper reuses that test as its fast "few-shot" measure, which we meet in Section 4.3 and run ourselves.

And Noisy Student:

> [!PAPER] Xie et al. (2020), Self-training with Noisy Student · Abstract
> [![The abstract of Noisy Student: Noisy Student Training achieves 88.4% top-1 accuracy on ImageNet. On ImageNet, we first train an EfficientNet model on labeled images and use it as a teacher to generate pseudo labels for 300M unlabeled images. We then train a larger EfficientNet as a student model on the combination of labeled and pseudo labeled images. We iterate this process by putting back the student as the teacher, and inject noise such as dropout, stochastic depth and data augmentation via RandAugment to the student](/img/papers/vit/p4-ext-noisy.png)](/img/papers/vit/p4-ext-noisy.png)
>
> **What it says:** Noisy Student reaches "88.4% top-1 accuracy on ImageNet". A first EfficientNet is used "as a teacher to generate pseudo labels for 300M unlabeled images", then the authors "train a larger EfficientNet as a student model" on real and invented labels, and repeat with the student as the new teacher.
>
> **What it means:**
>
> Think of a teacher who marks a huge pile of unmarked homework, then trains a student on the marked pile. The student gets noise in training (dropout, distorted pictures), so it must learn more robust rules than the teacher had. Then the student becomes the next teacher.
>
> The "300M unlabeled images" are the JFT-300M pictures with their labels thrown away. So Noisy Student and ViT-H/14 see the same pictures. The difference is in what they learn from: ViT uses JFT's own labels, Noisy Student uses its teacher's guesses.

### The cost unit: TPUv3-core-days

> [!PAPER] · §4.2
> [![Section 4.2 continued: all models were trained on TPUv3 hardware, and we report the number of TPUv3-core-days taken to pre-train each of them, that is, the number of TPU v3 cores (2 per chip) used for training multiplied by the training time in days](/img/papers/vit/p4-core-days.png)](/img/papers/vit/p4-core-days.png)
>
> **What it says:** "All models were trained on TPUv3 hardware", and the paper reports "the number of TPUv3-core-days taken to pre-train each of them", which is "the number of TPU v3 cores (2 per chip) used for training multiplied by the training time in days".
>
> **What it means:**
>
> This is a unit of work, like "person-days" on a building site. Ten workers for three days is 30 person-days, and so is one worker for 30 days. The unit ignores how the work was split, and counts only how much was done.
>
> All five models in the table trained on the same kind of chip. So core-days compare their cost directly, without the question of whose hardware was faster. The count covers pre-training only. Fine-tuning is cheap next to it, and the paper leaves it out.

> [!DEFINITION] TPUv3 core
> A TPU (Tensor Processing Unit) is Google's chip for neural-network arithmetic, the role a GPU plays elsewhere ([Part 1](part-1-the-big-idea.md) introduced hardware accelerators). One TPUv3 chip holds 2 cores. A "TPUv3-8" machine that anyone can rent in Google's cloud has 4 chips, so 8 cores.

The unit is one multiplication:

$$
\text{core-days} = \text{number of cores} \times \text{days of training}
$$

Read it as: count the cores that worked in parallel, and multiply by the number of days they worked. A tiny example: 4 cores for 10 days is $$4 \times 10 = 40$$ core-days, the same as 1 core for 40 days.

The paper gives one real check of the unit. It says that ViT-L/16 on ImageNet-21k "could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days". That is $$8 \times 30 = 240$$ core-days, or 0.24k, and Table 2 lists 0.23k for that model. The script [`vit_part4_tables.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part4_tables.py) turns every column into the same units:

```text
== 1. TPUv3-core-days: cores x days ==
  one TPUv3 chip has 2 cores; "core-days" = (number of cores) x (days of training)
  8 cores x 30 days = 240 core-days = 0.24k   (Table 2 says 0.23k for ViT-L/16, I21k)
  ViT-H/14 (JFT)               2.50k core-days =  1,250 chip-days =   312 days on one 8-core TPUv3
  ViT-L/16 (JFT)               0.68k core-days =    340 chip-days =    85 days on one 8-core TPUv3
  ViT-L/16 (I21k)              0.23k core-days =    115 chip-days =    29 days on one 8-core TPUv3
  BiT-L (R152x4)               9.90k core-days =  4,950 chip-days = 1,238 days on one 8-core TPUv3
  Noisy Student (EffNet-L2)   12.30k core-days =  6,150 chip-days = 1,538 days on one 8-core TPUv3
  BiT-L / ViT-L/16 (JFT) = 9.9k / 0.68k = 14.6x more pre-training compute for BiT-L
  BiT-L / ViT-H/14       = 9.9k / 2.5k  = 4.0x
  Noisy Student / ViT-H/14 = 12.3k / 2.5k = 4.9x
```

On one rented 8-core machine, BiT-L would take 1,238 days, more than three years. ViT-L/16 on JFT would take 85 days. Large runs are spread over many chips at once, so the real waiting time is much shorter. The total work stays the same.

## Table 2: ViT against the record holders {§4.2}

> [!PAPER] · Table 2
> [![Table 2 of the ViT paper: five columns, Ours-JFT ViT-H/14, Ours-JFT ViT-L/16, Ours-I21k ViT-L/16, BiT-L ResNet152x4 and Noisy Student EfficientNet-L2. Rows: ImageNet 88.55, 87.76, 85.30, 87.54, 88.4/88.5; ImageNet ReaL 90.72, 90.54, 88.62, 90.54, 90.55; CIFAR-10 99.50, 99.42, 99.15, 99.37; CIFAR-100 94.55, 93.90, 93.25, 93.51; Oxford-IIIT Pets 97.56, 97.32, 94.67, 96.62; Oxford Flowers-102 99.68, 99.74, 99.61, 99.63; VTAB 77.63, 76.28, 72.72, 76.29; TPUv3-core-days 2.5k, 0.68k, 0.23k, 9.9k, 12.3k. The caption: mean and standard deviation of the accuracies, averaged over three fine-tuning runs](/img/papers/vit/p4-table2.png)](/img/papers/vit/p4-table2.png)
>
> **What it says:** a "Comparison with state of the art on popular image classification benchmarks". The numbers are the "mean and standard deviation of the accuracies, averaged over three fine-tuning runs". ViT models pre-trained on JFT-300M "outperform ResNet-based baselines on all datasets, while taking substantially less computational resources to pre-train". The ImageNet-21k model "performs well too".
>
> **What it means:**
>
> This is the paper's main result in one table. Read it by columns first. Three ViT columns (two pre-trained on JFT-300M, one on the public ImageNet-21k) face two CNN columns. The last row is the price: core-days of pre-training.
>
> Then read it by rows. Seven benchmarks, from ImageNet's 1,000 classes to the 19 small tasks of VTAB. The bold number is the best in its row. ViT-H/14 holds six of the seven; ViT-L/16 holds Flowers.
>
> Now look at the price row together with the ImageNet row. ViT-H/14 beats both CNNs for a quarter to a fifth of their cost. ViT-L/16 beats BiT-L for a fifteenth of its cost. That pairing, more accuracy for less compute, is the claim. Accuracy alone would not have been news.

A few words in the table need a note. [Part 3](part-3-fine-tuning-and-setup.md) describes the datasets in full.

> [!DEFINITION] Mean ± standard deviation over runs
> Each pre-trained model was fine-tuned three times on each benchmark, with different random seeds. "88.55 ± 0.04" means the three results averaged 88.55%, and they typically sat about 0.04 points from that average. A difference much larger than the ± values is not luck of the seed.

> [!DEFINITION] CIFAR-10, Oxford-IIIT Pets, Oxford Flowers-102
> Three small benchmarks. **CIFAR-10**: 60,000 pictures of 32×32 pixels in 10 classes (planes, cats, trucks...). **Oxford-IIIT Pets**: about 7,400 photos of 37 cat and dog breeds, about 100 training photos per breed. **Oxford Flowers-102**: 102 kinds of flowers, only 10 training photos per kind.

> [!DEFINITION] Error rate
> The share of test pictures the model gets wrong: $$100\% - \text{accuracy}$$. At high accuracy it shows differences that accuracy hides. 99.37% and 99.50% look the same, but the errors are 0.63% and 0.50%.

**Accuracy hides progress near 100%.** When two accuracies are both near 100, count the mistakes instead. The relative error reduction asks: of the old mistakes, which share has gone?

$$
\text{error cut} = \frac{e_{\text{old}} - e_{\text{new}}}{e_{\text{old}}}, \qquad e = 100 - \text{accuracy}
$$

Read it as: the old error rate $$e_{\text{old}}$$ minus the new one $$e_{\text{new}}$$, divided by the old one. A tiny example: on 1,000 test pictures, the old model makes 63 mistakes and the new one 50. The cut is $$(63 - 50)/63 = 0.21$$, so 21% of the mistakes are gone. That is CIFAR-10 in Table 2: BiT-L at 99.37% (0.63% errors) and ViT-H/14 at 99.50% (0.50%). Every row:

```text
== 2. Table 2 as error rates (100 - accuracy) ==
  dataset                    ViT-H/14        ViT-L/16        ViT-L/16           BiT-L   H/14 vs BiT-L: error cut
  ImageNet                      11.45           12.24           14.70           12.46   12.46 -> 11.45: 8% fewer errors
  ImageNet ReaL                  9.28            9.46           11.38            9.46   9.46 -> 9.28: 2% fewer errors
  CIFAR-10                       0.50            0.58            0.85            0.63   0.63 -> 0.50: 21% fewer errors
  CIFAR-100                      5.45            6.10            6.75            6.49   6.49 -> 5.45: 16% fewer errors
  Oxford-IIIT Pets               2.44            2.68            5.33            3.38   3.38 -> 2.44: 28% fewer errors
  Oxford Flowers-102             0.32            0.26            0.39            0.37   0.37 -> 0.32: 14% fewer errors
  VTAB (19 tasks)               22.37           23.72           27.28           23.71   23.71 -> 22.37: 6% fewer errors
```

{{FIG:p4_errors|Table 2 as error rates, ViT-H/14 (JFT) against BiT-L. Each row has its own scale. ViT-H/14 makes fewer mistakes on every benchmark: 8% fewer on ImageNet, 16% on CIFAR-100, 21% on CIFAR-10 and 28% on Oxford-IIIT Pets.}}

{{FIG:p4_compute|Pre-training compute against ImageNet accuracy for the five columns of Table 2. The JFT ViTs sit up and to the left of the two CNNs. ViT-L/16 (JFT) is 0.22 points above BiT-L for 14.6 times less compute.}}

> [!PAPER] · §4.2
> [![The paragraph after the table, part 1: Table 2 shows the results. The smaller ViT-L/16 model pre-trained on JFT-300M outperforms BiT-L, which is pre-trained on the same dataset, on all tasks, while requiring substantially less computational resources to train. The larger model, ViT-H/14, further improves the performance, especially on the more challenging datasets: ImageNet, CIFAR-100, and the VTAB suite](/img/papers/vit/p4-table2-text.png)](/img/papers/vit/p4-table2-text.png)
> [![The paragraph continues on the next page: this model still took substantially less compute to pre-train than prior state of the art. However, pre-training efficiency may be affected not only by the architecture choice, but also other parameters, such as training schedule, optimizer, weight decay. A controlled study follows in Section 4.4. Finally, the ViT-L/16 model pre-trained on the public ImageNet-21k dataset performs well on most datasets too, while taking fewer resources to pre-train: it could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days](/img/papers/vit/p4-table2-text2.png)](/img/papers/vit/p4-table2-text2.png)
>
> **What it says:** ViT-L/16 on JFT-300M "outperforms BiT-L (which is pre-trained on the same dataset) on all tasks, while requiring substantially less computational resources to train". ViT-H/14 "further improves the performance, especially on the more challenging datasets". Then a warning: "pre-training efficiency may be affected not only by the architecture choice, but also other parameters, such as training schedule, optimizer, weight decay". Last, the ImageNet-21k model "could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days".
>
> **What it means:**
>
> The paragraph makes three claims, and each one deserves a check. The first is the strongest: same data, a smaller ViT, better everywhere. We check "on all tasks" row by row below.
>
> The second claim is about where the Huge model helps. "Challenging" means the datasets where accuracy is still far from 100%, so there is room to improve. On CIFAR-10 or Flowers both models are already near the ceiling.
>
> The warning is honest and important. The compute numbers come from each team's own training runs, with their own schedules and settings. So "14.6 times cheaper" compares two recipes, not only two designs. Section 4.4 (Part 5, coming soon) runs the controlled version of this comparison.

**"On all tasks", checked.** The script compares ViT-L/16 (JFT) with BiT-L row by row, and divides each gap by the larger of the two ± values:

```text
== 3. ViT-L/16 (JFT) against BiT-L, row by row (gap in points, and in units of the larger std) ==
  ImageNet             87.76 -  87.54 = +0.22   (larger std 0.03, gap = 7.3 std)   ViT-L/16 higher
  ImageNet ReaL        90.54 -  90.54 = +0.00   (larger std 0.03, gap = 0.0 std)   equal
  CIFAR-10             99.42 -  99.37 = +0.05   (larger std 0.06, gap = 0.8 std)   ViT-L/16 higher
  CIFAR-100            93.90 -  93.51 = +0.39   (larger std 0.08, gap = 4.9 std)   ViT-L/16 higher
  Oxford-IIIT Pets     97.32 -  96.62 = +0.70   (larger std 0.23, gap = 3.0 std)   ViT-L/16 higher
  Oxford Flowers-102   99.74 -  99.63 = +0.11   (larger std 0.03, gap = 3.7 std)   ViT-L/16 higher
  VTAB (19 tasks)      76.28 -  76.29 = -0.01   (larger std 1.70, gap = -0.0 std)   BiT-L higher
  ViT-L/16 (JFT) is strictly higher on 5 of 7 rows; ImageNet ReaL is a tie and VTAB is 0.01 lower, inside BiT-L's std of 1.70
```

So the exact count is five wins, one tie and one row where BiT-L is 0.01 higher. The tie and the 0.01 are far inside the noise: BiT-L's VTAB results spread by 1.70 points between runs. A fair reading of "outperforms on all tasks" is "is never measurably worse, and is clearly better on four rows".

One more caution about the ± values. They measure only the luck of fine-tuning, because each model was pre-trained once. A second pre-training run of the same model could land a little higher or lower, and the table cannot show that spread. So a gap of 7.3 standard deviations on ImageNet is very unlikely to be fine-tuning luck for these two checkpoints. It is a weaker statement about the two designs in general.

The Huge model's gains over ViT-L/16 match the "challenging datasets" sentence. They are largest on VTAB (+1.35), ImageNet (+0.79) and CIFAR-100 (+0.65), and near zero on CIFAR-10 (+0.08) and Flowers (−0.06):

```text
  ViT-H/14 minus ViT-L/16 (both JFT), row by row: ImageNet +0.79, ImageNet ReaL +0.18, CIFAR-10 +0.08, CIFAR-100 +0.65, Oxford-IIIT Pets +0.24, Oxford Flowers-102 -0.06, VTAB (19 tasks) +1.35
```

The public model deserves a sentence too. ViT-L/16 pre-trained on ImageNet-21k scores 85.30% on ImageNet. That is 2.24 points below BiT-L, for 0.23k core-days against 9.9k:

```text
  ViT-L/16 (I21k) against BiT-L on ImageNet: -2.24 points, for 0.23k against 9.9k core-days
```

Anyone with a rented TPU and a month could repeat it. The released weights we use in this series come from this kind of run.

### Top-1, top-5 and ImageNet ReaL

The second row of Table 2, ImageNet ReaL, uses the same 50,000 ImageNet test pictures as the first row, with new labels. Here is why the labels were redone:

> [!PAPER] Beyer et al. (2020), Are we done with ImageNet? · Abstract
> [![The abstract of Are we done with ImageNet: we ask whether recent progress on the ImageNet classification benchmark continues to represent meaningful generalization, or whether the community has started to overfit to the idiosyncrasies of its labeling procedure. We therefore develop a significantly more robust procedure for collecting human annotations of the ImageNet validation set. Using these new labels, we reassess the accuracy of recently proposed ImageNet classifiers, and find their gains to be substantially smaller than those reported on the original labels](/img/papers/vit/p4-ext-real.png)](/img/papers/vit/p4-ext-real.png)
>
> **What it says:** the authors ask whether progress on ImageNet still means real progress, or whether models now "overfit to the idiosyncrasies of its labeling procedure". They build "a more robust procedure for collecting human annotations" of the test pictures, and find the recent "gains to be substantially smaller than those reported on the original labels".
>
> **What it means:**
>
> Each ImageNet test picture has one label, picked by one process years ago. Many pictures show two objects, and some labels are simply wrong. A model that answers "keyboard" for a photo of a keyboard and a mouse, labelled "mouse", is marked wrong.
>
> ReaL ("Reassessed Labels") lets a picture have several correct labels, chosen with more care. A model is right if its top answer is any of them. That is why every ReaL number in Table 2 is higher than the ImageNet number above it.

> [!DEFINITION] Top-5 accuracy
> The share of test pictures whose correct label is among the model's five highest-scoring classes. It is always at least the top-1 accuracy ([Part 1](part-1-the-big-idea.md)). It forgives a model that ranks a second plausible class first.

We can see the labelling problem with released models. The script [`vit_part4_top5.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part4_top5.py) runs three ImageNet classifiers on the 3,925 validation pictures of Imagenette, a 10-class subset of ImageNet. Each model still chooses among all 1,000 ImageNet classes:

```text
== top-1 and top-5 accuracy over all 1000 ImageNet classes, on the 3,925 Imagenette validation pictures ==
  ViT-B/16     top-1 83.13%   top-5 96.66%   top-1 wrong on 662 pictures, of which 531 have the label in the top 5
  ViT-L/16     top-1 84.92%   top-5 97.63%   top-1 wrong on 592 pictures, of which 499 have the label in the top 5
  ResNet-50    top-1 79.36%   top-5 95.95%   top-1 wrong on 810 pictures, of which 651 have the label in the top 5
```

Top-5 forgives most mistakes: of ViT-L/16's 592 top-1 "errors", 499 have the correct label among its five best guesses. Are those real errors? Here are eight of them, picked at random:

```text
  ViT-L/16, pictures whose top-1 is not the label but whose top-5 contains it (8 picked at random, seed 3):
    n02102040_2700.JPEG          label: English springer top-1: Boston bull            label at rank 3
    n02102040_8260.JPEG          label: English springer top-1: German short-haired pointer label at rank 3
    n02102040_880.JPEG           label: English springer top-1: Sussex spaniel         label at rank 2
    n02979186_15511.JPEG         label: cassette player  top-1: cassette               label at rank 2
    n03028079_26940.JPEG         label: church           top-1: bell cote              label at rank 2
    n03028079_9481.JPEG          label: church           top-1: monastery              label at rank 2
    n03028079_9560.JPEG          label: church           top-1: vault                  label at rank 3
    n03394916_38702.JPEG         label: French horn      top-1: cornet                 label at rank 2
```

<figure><img src="/img/papers/vit/p4-top5-misses.png" alt="Eight photos with their ImageNet label and ViT-L/16's top answer. Three English springer spaniels called Boston bull, German short-haired pointer and Sussex spaniel. A hand holding a cassette inside a cassette player, called cassette. A church tower called bell cote, a small church called monastery, a church interior called vault, and two musicians with a French horn and a large brass instrument called cornet" loading="lazy" /><figcaption>Eight of ViT-L/16's top-1 "errors" on Imagenette, picked at random. The label is in black, the model's first answer in orange. In each case the correct label is the model's second or third answer.</figcaption></figure>

Look at the pictures. The cassette is in the photo, inside the cassette player. A bell cote is the small tower that holds a church's bell, and the church photo shows one. The third church photo is taken inside, under its vault. The musicians hold a French horn and a second, larger brass instrument.

Only the three dog pictures are plain mistakes, and two of them name a similar hunting breed. Under ImageNet's single-label rule, all eight count as errors. Several of them are exactly what ReaL fixes.

So treat a single ImageNet number with care, especially near the top of the table. The ReaL paper found that part of the recent gains came from learning ImageNet's labelling habits. The ReaL row of Table 2 is the check on that, and ViT-H/14 still leads it (90.72%).

What this run cannot show: Imagenette's 10 classes are far easier than ImageNet's 1,000, and its pictures are stored at 160 pixels. Also, these ImageNet pictures may have been seen in pre-training. So the accuracies above are a lesson about the metrics, not a benchmark result.

### VTAB and Figure 2

[Part 1](part-1-the-big-idea.md) defined VTAB as 19 tasks with 1,000 training examples each. Here is what it set out to measure, in its own words:

> [!PAPER] Zhai et al. (2019), The Visual Task Adaptation Benchmark · Abstract
> [![The abstract of the VTAB paper: representation learning promises to unlock deep learning for the long tail of vision tasks without expensive labelled datasets. We present the Visual Task Adaptation Benchmark (VTAB), which defines good representations as those that adapt to diverse, unseen tasks with few examples](/img/papers/vit/p4-ext-vtab.png)](/img/papers/vit/p4-ext-vtab.png)
>
> **What it says:** VTAB "defines good representations as those that adapt to diverse, unseen tasks with few examples".
>
> **What it means:**
>
> ImageNet tests one task with a million labels. VTAB tests the opposite: many unlike tasks, each with only 1,000 labels. Some tasks are ordinary photos. Others are satellite images, tissue slides, or synthetic scenes where the question is "how many objects?" or "how far away?".
>
> A model that only memorised ImageNet's classes does poorly here. A model that learned general visual features adapts. So VTAB is the paper's test of how general ViT's features are.

> [!PAPER] · Figure 2
> [![Figure 2 of the ViT paper: four bar charts, VTAB (19 tasks), Natural (7 tasks), Specialized (4 tasks) and Structured (8 tasks), each with four bars for ViT-H/14, BiT-L (R152x4), VIVI-Ex-100% (R50x3) and S4L (R50x1). ViT-H/14 is the tallest bar in every group](/img/papers/vit/p4-figure2.png)](/img/papers/vit/p4-figure2.png)
> [![The text on Figure 2: Figure 2 decomposes the VTAB tasks into their respective groups, and compares to previous SOTA methods on this benchmark: BiT, VIVI, a ResNet co-trained on ImageNet and online videos, and S4L, supervised plus semi-supervised learning on ImageNet. ViT-H/14 outperforms BiT-R152x4, and other methods, on the Natural and Structured tasks. On the Specialized the performance of the top two models is similar](/img/papers/vit/p4-figure2-text.png)](/img/papers/vit/p4-figure2-text.png)
>
> **What it says:** Figure 2 splits VTAB into its three groups and compares ViT-H/14 with BiT, "VIVI" (a ResNet "co-trained on ImageNet" and on online videos) and "S4L" ("supervised plus semi-supervised learning on ImageNet"). ViT-H/14 "outperforms BiT-R152x4, and other methods, on the Natural and Structured tasks". On the Specialized tasks "the performance of the top two models is similar".
>
> **What it means:**
>
> Read the four panels left to right. The first is the total VTAB score; the other three split it into Natural, Specialized and Structured tasks. Note that each panel has its own vertical scale. The bars look very different in height, but the y-axis starts at 65, 70, 80 or 50.
>
> The Structured group is where ViT's lead over BiT looks largest. These tasks ask about geometry: counting objects, judging distance, reading an angle. A CNN pools its features into a summary of "what" is in the picture. Global attention can relate any two patches directly, which may suit "where" questions. That is our reading; the paper does not test the reason.

> [!DEFINITION] VIVI and S4L
> Two earlier VTAB leaders. **VIVI** (Tschannen et al., 2020) trains a ResNet on ImageNet labels plus unlabelled online videos. The order of the video frames gives it a free training signal. **S4L** (Zhai et al., 2019) combines ImageNet labels with a self-supervised task, such as guessing how much a picture was rotated.

The paper prints Figure 2 only as bars. Its Appendix D.10 (read at the end of this part) gives all 19 task scores for three ViT models. So we can recompute the ViT groups ourselves:

```text
== 6. Table 9: the 19 VTAB-1k tasks, grouped as in Figure 2 ==
  ViT-H/14 (JFT)   mean of 19 = 77.63 (table: 77.6)   Natural 86.7   Specialized 87.0   Structured 65.0   lowest task: sNORB-Azim 33.2
  ViT-L/16 (JFT)   mean of 19 = 76.29 (table: 76.3)   Natural 85.6   Specialized 86.7   Structured 63.0   lowest task: sNORB-Azim 36.2
  ViT-L/16 (I21k)  mean of 19 = 72.71 (table: 72.7)   Natural 83.3   Specialized 84.7   Structured 57.5   lowest task: sNORB-Azim 30.5
```

Our ViT-H/14 values (86.7, 87.0 and 65.0) agree with the blue bars of Figure 2, as far as a bar chart can be read. The mean of the 19 tasks gives back Table 2's 77.63 exactly.

{{FIG:p4_vtab_groups|The 19 VTAB tasks of Table 9 averaged into Figure 2's groups, for the three ViT models of Table 2. Natural and Specialized tasks score in the 80s. Structured tasks are much harder (57.5 to 65.0) and gain the most from JFT-300M and a larger model.}}

## Pre-training data requirements {§4.3}

[Part 1](part-1-the-big-idea.md) read two sentences of this section as the evidence behind "large scale training trumps inductive bias". Here is the whole section, with the experiments behind it.

> [!PAPER] · §4.3
> [![The opening of Section 4.3, Pre-training Data Requirements: the Vision Transformer performs well when pre-trained on a large JFT-300M dataset. With fewer inductive biases for vision than ResNets, how crucial is the dataset size? We perform two series of experiments. First, we pre-train ViT models on datasets of increasing size: ImageNet, ImageNet-21k, and JFT-300M. To boost the performance on the smaller datasets, we optimize three basic regularization parameters: weight decay, dropout, and label smoothing. When pre-trained on the smallest dataset, ImageNet, ViT-Large models underperform compared to ViT-Base models, despite moderate regularization. With ImageNet-21k pre-training, their performances are similar. Only with JFT-300M, do we see the full benefit of larger models](/img/papers/vit/p4-data-intro.png)](/img/papers/vit/p4-data-intro.png)
>
> **What it says:** "With fewer inductive biases for vision than ResNets, how crucial is the dataset size?" Two series of experiments answer it. In the first, ViT models are pre-trained "on datasets of increasing size", and three regularization settings ("weight decay, dropout, and label smoothing") are tuned to help on the smaller ones. After ImageNet pre-training, "ViT-Large models underperform compared to ViT-Base models". After ImageNet-21k, "their performances are similar". "Only with JFT-300M, do we see the full benefit of larger models."
>
> **What it means:**
>
> This section turns the paper's thesis into an experiment. [Part 2](part-2-the-model.md) showed that ViT builds in almost no knowledge about pictures. If that is true, it must learn everything from data, so the amount of data should decide how well it does.
>
> The first series is generous to the small datasets. Small data invites memorising, and the three tuned settings fight memorising. So when Large still loses to Base on ImageNet, the loss is not for lack of effort. It is the data.
>
> The pattern has three steps: Large loses (1.3 million pictures), ties (14 million), wins clearly (303 million). Bigger models are hungrier, and only the largest set feeds them.

[Part 1](part-1-the-big-idea.md) defined regularization. The three settings tuned here work in different ways:

> [!DEFINITION] Weight decay
> A pull of every weight towards zero during training: at each step, each weight shrinks by a small fraction of itself. Large weights can encode details of single training pictures, so keeping weights small favours simple, general rules. The paper pre-trains with a weight decay of 0.1 ([Part 3](part-3-fine-tuning-and-setup.md)).

> [!DEFINITION] Dropout
> During training, each number passed between layers is set to zero with some probability, for example 0.1, and the rest are scaled up to compensate. The network cannot rely on any single path, so it learns several overlapping ways to reach the answer. At test time nothing is dropped.

> [!DEFINITION] Label smoothing
> Training against a slightly softened target instead of a hard "100% this class". With $$K$$ classes and a smoothing value $$\varepsilon$$, the correct class gets $$1 - \varepsilon + \varepsilon/K$$ and every other class gets $$\varepsilon/K$$. With $$K = 4$$ and $$\varepsilon = 0.1$$, the target is $$(0.925, 0.025, 0.025, 0.025)$$. The model is never pushed to total certainty, which makes it less likely to memorise.

A footnote on the same page explains one detail of the first series:

> [!PAPER] · Footnote 2
> [![Footnote 2: note that the ImageNet pre-trained models are also fine-tuned, but again on ImageNet. This is because the resolution increase during fine-tuning improves the performance](/img/papers/vit/p4-footnote2.png)](/img/papers/vit/p4-footnote2.png)
>
> **What it says:** "the ImageNet pre-trained models are also fine-tuned, but again on ImageNet", because "the resolution increase during fine-tuning improves the performance".
>
> **What it means:**
>
> A model pre-trained on ImageNet and tested on ImageNet seems to need no fine-tuning. It gets one anyway, at a higher resolution, for the same reason every other model does.
>
> [Part 3](part-3-fine-tuning-and-setup.md) showed the step: pre-train at 224 pixels, fine-tune at 384, with the position embeddings stretched in 2D. Doing it for every row keeps the comparison fair. Each ImageNet number then comes from the same two-step recipe, whatever the pre-training set.

> [!PAPER] · Figure 3
> [![Figure 3 of the ViT paper: ImageNet top-1 accuracy against the pre-training dataset, ImageNet, ImageNet-21k and JFT-300M. A grey band shows the range of BiT ResNets. Circles for ViT-B/32, ViT-B/16, ViT-L/32, ViT-L/16 and ViT-H/14, larger circles for larger models. Caption: transfer to ImageNet. While large ViT models perform worse than BiT ResNets (shaded area) when pre-trained on small datasets, they shine when pre-trained on larger datasets. Similarly, larger ViT variants overtake smaller ones as the dataset grows](/img/papers/vit/p4-figure3.png)](/img/papers/vit/p4-figure3.png)
> [![The text on Figure 3: Figure 3 also shows the performance region spanned by BiT models of different sizes. The BiT CNNs outperform ViT on ImageNet, but with the larger datasets, ViT overtakes](/img/papers/vit/p4-figure3-text.png)](/img/papers/vit/p4-figure3-text.png)
>
> **What it says:** "While large ViT models perform worse than BiT ResNets (shaded area) when pre-trained on small datasets, they shine when pre-trained on larger datasets." Also, "larger ViT variants overtake smaller ones as the dataset grows". The grey band is the "region spanned by BiT models of different sizes".
>
> **What it means:**
>
> Read the figure as three columns, one per pre-training set. In each column, the grey band shows where the BiT ResNets landed, from the smallest to the largest. The coloured circles are the five ViTs, and a bigger circle means a bigger model.
>
> In the ImageNet column every ViT sits inside or below the band: the CNNs are as good or better. In the JFT-300M column the picture flips. ViT-H/14 reaches the top edge of the band, and ViT-L/16 sits just under it. The small ViT-B/32 is still far below. Size only pays when the data is there.

Appendix C holds the numbers behind Figure 3 (Table 5, read at the end of this part). [Part 1](part-1-the-big-idea.md) drew two of its models on ImageNet. Here are all five models and three of the six datasets:

{{FIG:p4_table5|Table 5 as three small charts, one per fine-tuning dataset. Each line is one ViT size across the three pre-training sets; dashed lines are the /32 models. On ImageNet, ViT-B/16 leads after ImageNet pre-training (77.91), and ViT-H/14 leads after JFT-300M (88.04). On CIFAR-100, the /32 models keep up with the /16 models.}}

The ranking of the five models on ImageNet changes with the data:

```text
== 5. Table 5: ImageNet top-1 after fine-tuning, by pre-training dataset (the numbers behind Figure 3) ==
  pre-trained on ImageNet      ViT-B/16 77.91  ViT-B/32 73.38  ViT-L/16 76.53  ViT-L/32 71.16
      best to worst: ViT-B/16 > ViT-L/16 > ViT-B/32 > ViT-L/32;   ViT-L/16 - ViT-B/16 = -1.38
  pre-trained on ImageNet-21k  ViT-B/16 83.97  ViT-B/32 81.28  ViT-L/16 85.15  ViT-L/32 80.99  ViT-H/14 85.13
      best to worst: ViT-L/16 > ViT-H/14 > ViT-B/16 > ViT-B/32 > ViT-L/32;   ViT-L/16 - ViT-B/16 = +1.18
  pre-trained on JFT-300M      ViT-B/16 84.15  ViT-B/32 80.73  ViT-L/16 87.12  ViT-L/32 84.37  ViT-H/14 88.04
      best to worst: ViT-H/14 > ViT-L/16 > ViT-L/32 > ViT-B/16 > ViT-B/32;   ViT-L/16 - ViT-B/16 = +2.97
  ImageNet -> JFT-300M pre-training: ViT-B/16 gains +6.24 points, ViT-L/16 gains +10.59 points
```

Three things stand out. First, the winner changes. ViT-B/16 leads after ImageNet pre-training, and ViT-H/14 leads after JFT-300M. On ImageNet-21k the Huge model only ties ViT-L/16 (85.13 against 85.15), and the paper did not train it on ImageNet alone.

Second, more data helps the bigger model more: from ImageNet to JFT-300M, ViT-B/16 gains 6.24 points and ViT-L/16 gains 10.59. Third, ViT-L/32 jumps from last place to third. With a 32-pixel patch it has 4 times fewer tokens than ViT-L/16, so it sees the picture more coarsely. With enough data, its size starts to pay anyway.

> [!PAPER] · §4.3
> [![The second series of Section 4.3: second, we train our models on random subsets of 9M, 30M, and 90M as well as the full JFT-300M dataset. We do not perform additional regularization on the smaller subsets and use the same hyper-parameters for all settings. This way, we assess the intrinsic model properties, and not the effect of regularization. We do, however, use early-stopping, and report the best validation accuracy achieved during training. To save compute, we report few-shot linear accuracy instead of full fine-tuning accuracy. Vision Transformers overfit more than ResNets with comparable computational cost on smaller datasets. For example, ViT-B/32 is slightly faster than ResNet50; it performs much worse on the 9M subset, but better on 90M+ subsets. The same is true for ResNet152x2 and ViT-L/16. This result reinforces the intuition that the convolutional inductive bias is useful for smaller datasets, but for larger ones, learning the relevant patterns directly from data is sufficient, even beneficial](/img/papers/vit/p4-subsets.png)](/img/papers/vit/p4-subsets.png)
>
> **What it says:** the second series trains on "random subsets of 9M, 30M, and 90M" pictures and on the full JFT-300M. It does "not perform additional regularization" and keeps the same settings everywhere, to measure "intrinsic model properties". It uses "early-stopping" and reports "few-shot linear accuracy". The finding: "Vision Transformers overfit more than ResNets with comparable computational cost on smaller datasets". "ViT-B/32 is slightly faster than ResNet50; it performs much worse on the 9M subset, but better on 90M+ subsets." The conclusion: "the convolutional inductive bias is useful for smaller datasets, but for larger ones, learning the relevant patterns directly from data is sufficient, even beneficial."
>
> **What it means:**
>
> The first series compared datasets that differ in more than size: ImageNet, ImageNet-21k and JFT have different pictures and different labels. The second series removes that doubt. All four training sets are random samples of one dataset, so only the amount changes.
>
> It also removes the tuning. The first series helped small data with regularization; this one uses one fixed recipe. What is left is the bare tendency of each design. Think of two students given the same amount of study time and the same textbook, only with more or fewer pages.
>
> The pairs are matched by cost, not by size: ViT-B/32 against ResNet-50, ViT-L/16 against ResNet152x2. In each pair the ResNet wins with 9 million pictures and loses with 90 million or more. "Even beneficial" is the bold part. With enough data, the CNN's built-in assumptions stop helping and start to hold it back.

> [!DEFINITION] Overfitting
> Learning the training pictures too well. The model memorises details that are true only of those pictures, such as a watermark or a background. It scores high on them and lower on new ones. A model with more freedom and less data overfits more easily.

> [!DEFINITION] Early stopping
> Checking the model on held-out validation pictures during training, and keeping the best checkpoint instead of the last one. If the model starts to overfit late in training, its validation score falls, and early stopping ignores that fall.

The second series reports a "few-shot linear accuracy" instead of fine-tuning. [Part 3](part-3-fine-tuning-and-setup.md) works the method out in full; here is what we need.

> [!DEFINITION] Few-shot linear evaluation
> Freeze the pre-trained model. Take $$k$$ labelled pictures of each of the $$K$$ classes ("$$k$$-shot"), and compute their representations. Fit one linear layer from representation to class by regularised least squares, which has an exact formula, so no training loop is needed. Then score test pictures with that layer. "5-shot ImageNet" means 5 pictures of each of the 1,000 classes.

The idea: we want a weight matrix $$W$$ that maps each representation $$\mathbf{x}$$ to a row of scores, one score per class. The target for a picture of class $$c$$ is $$+1$$ at position $$c$$ and $$-1$$ everywhere else (the paper's "$$\{-1, 1\}^K$$ target vectors"). Least squares picks the $$W$$ that makes the scores closest to the targets, with a penalty on large weights:

$$
W = \left(X^\top X + \lambda I\right)^{-1} X^\top Y
$$

Read it left to right. $$X$$ is the $$n \times d$$ matrix of representations, one row per training picture ($$n = kK$$ rows, $$d$$ numbers each). $$Y$$ is the $$n \times K$$ matrix of $$\pm 1$$ targets. $$X^\top X$$ is $$d \times d$$ and measures how the features vary together. $$\lambda I$$ adds $$\lambda$$ to its diagonal: the penalty on large weights, which also keeps the inverse well defined when $$n$$ is small. $$X^\top Y$$ is $$d \times K$$ and measures how each feature goes with each class.

A new picture gets the scores $$\mathbf{x} W$$, and the largest score is the predicted class.

A tiny example, small enough for a pencil: four pictures with two features each, two classes, $$\lambda = 1$$. The first two pictures are class 0, the last two class 1:

```text
== 1. the few-shot closed form on a tiny example: 4 images, 2 features, 2 classes, lambda = 1 ==
  X (one row per image) = [[1.0, 0.0], [2.0, 1.0], [0.0, 2.0], [1.0, 3.0]]   classes [0, 0, 1, 1]
  Y (targets in {-1, 1}) = [[1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]]
  X^T X = [[6.0, 5.0], [5.0, 14.0]]   + lambda I = [[7.0, 5.0], [5.0, 15.0]]
  X^T Y = [[2.0, -2.0], [-4.0, 4.0]]
  W = (X^T X + I)^-1 X^T Y = [[0.625, -0.625], [-0.475, 0.475]]
  new image x = [1, 2]: scores x W = [-0.325, 0.325] -> class 1
```

Check one entry by hand. $$X^\top X$$ has top-left entry $$1^2 + 2^2 + 0^2 + 1^2 = 6$$; add $$\lambda = 1$$ to get 7. The inverse of $$\begin{pmatrix} 7 & 5 \\ 5 & 15 \end{pmatrix}$$ is $$\tfrac{1}{80}\begin{pmatrix} 15 & -5 \\ -5 & 7 \end{pmatrix}$$, because $$7 \cdot 15 - 5 \cdot 5 = 80$$. Its first row times the first column of $$X^\top Y$$, $$(2, -4)$$, gives $$(15 \cdot 2 + 5 \cdot 4)/80 = 50/80 = 0.625$$. The new picture $$(1, 2)$$ scores $$1 \cdot 0.625 + 2 \cdot (-0.475) = -0.325$$ for class 0 and $$+0.325$$ for class 1, so it is class 1.

{{FIG:p4_ridge|The tiny example as a picture. The class-0 pictures (squares) and class-1 pictures (circles) sit on either side of the line where both scores are equal. The new picture (1, 2) lands on the class-1 side.}}

Why use this instead of fine-tuning? One matrix inverse of size $$d \times d$$ replaces thousands of training steps. For ViT-L, $$d = 1{,}024$$, so it is one $$1{,}024 \times 1{,}024$$ system of equations, tiny next to a training run. So the authors could score many checkpoints cheaply, and Figure 4 is made of these scores.

> [!PAPER] · Figure 4
> [![Figure 4 of the ViT paper: linear 5-shot ImageNet top-1 accuracy against the number of JFT pre-training samples, 10M, 30M, 100M and 300M, for ViT-L/16, ViT-L/32, ViT-B/32, ViT-b/32, ResNet50x1 (BiT) and ResNet152x2 (BiT). Caption: linear few-shot evaluation on ImageNet versus pre-training size. ResNets perform better with smaller pre-training datasets but plateau sooner than ViT, which performs better with larger pre-training. ViT-b is ViT-B with all hidden dimensions halved](/img/papers/vit/p4-figure4.png)](/img/papers/vit/p4-figure4.png)
>
> **What it says:** "Linear few-shot evaluation on ImageNet versus pre-training size. ResNets perform better with smaller pre-training datasets but plateau sooner than ViT, which performs better with larger pre-training." "ViT-b is ViT-B with all hidden dimensions halved."
>
> **What it means:**
>
> The x-axis is the size of the JFT subset, on a log scale, from about 10 million to 300 million pictures. The y-axis is 5-shot ImageNet accuracy of the frozen model. Each line is one model trained from scratch on that much data.
>
> Look at the slopes. The two grey ResNet lines are nearly flat after 30 million pictures: more data barely helps them. The ViT-L/16 and ViT-L/32 lines climb all the way. On the smallest subset, the ResNet152x2 line is the highest of all. At 300 million, ViT-L/16 is above it.
>
> "Plateau" is the key word. A built-in assumption is like a ceiling as well as a head start. The ResNets reach their ceiling early. The two large ViTs are still rising at 300 million pictures.

### Section 4.3 in miniature: training from scratch on growing data

The paper's two series need hundreds of millions of pictures. We can still ask its question at a size a laptop can afford. We train both kinds of network from nothing, on more and more pictures, with one fixed recipe. The script [`vit_part4_scratch.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part4_scratch.py) does it on CIFAR-10:

- **the data:** random subsets of the 50,000 CIFAR-10 training pictures (1,000, 5,000 and all 50,000), each inside the next;
- **two models of matched cost:** a small ViT from the same `transformers` code as the real one, and a small ResNet-18-style network. The ViT cuts the 32×32 picture into 4×4 patches, so it reads 64 patch tokens plus the class token;
- **one recipe for all six runs:** AdamW, one learning-rate schedule, 2,500 steps of 128 pictures, random crops and flips. As in Section 4.3, early stopping keeps the best score on 5,000 held-out pictures.

```text
== Section 4.3 in miniature: CIFAR-10 from scratch, same recipe for both models and every data size ==
  ViT    (4x4 patches -> 64 tokens + [class], D 128, 6 layers, 4 heads, MLP 512): 1.21M parameters, 77M multiply-adds per image
  ResNet (ResNet-18 layout, widths 24-48-96-192, BatchNorm):                     1.58M parameters, 79M multiply-adds per image
  recipe: AdamW lr 0.001, warm-up 250 steps then cosine, weight decay 0.05, batch 128, 2500 steps, crop + flip;
          validation on 5,000 CIFAR-10 test images every 250 steps, best value kept (early stopping)
    images  per class  epochs   ViT best  ResNet best  ResNet - ViT
     1,000        100     320      39.0%        54.6%        +15.5
     5,000        500      64      57.4%        76.5%        +19.1
    50,000      5,000       6      63.8%        84.2%        +20.5
  going from 1,000 to 50,000 images: ViT 39.0% -> 63.8% (+24.7), ResNet 54.6% -> 84.2% (+29.7)
```

The two models cost the same to run: 77 million and 79 million multiply-adds per picture. That matches the paper's pairing of "ViT-B/32" with "ResNet50" by speed rather than by size.

{{FIG:p4_scratch|Best validation accuracy of the small ViT and the small ResNet, trained from scratch on 1,000, 5,000 and 50,000 CIFAR-10 pictures with one recipe. The ResNet is ahead at every size we can afford, by 15.5 to 20.5 points.}}

**The ResNet wins at every size, by 15.5 to 20.5 points.** This is the left edge of Figure 4, where the ResNets lead. Our largest set has 50,000 pictures. The paper's smallest subset has 9 million, 180 times more, and even there the ResNet-50 beats ViT-B/32. We are far below the region where the paper sees the lines cross.

**The small ViT memorises.** The training curves show what "overfit" means in practice:

```text
== the training curves of vit_part4_scratch.py: validation accuracy (%) and smoothed training loss ==
   1,000 images  ViT    step 500: 36.0% / loss 1.272  step 1500: 37.7% / loss 0.080  step 2500: 38.6% / loss 0.004   best 39.0% at step 2000
   1,000 images  ResNet step 500: 47.5% / loss 0.360  step 1500: 53.3% / loss 0.003  step 2500: 54.1% / loss 0.001   best 54.6% at step 2000
  50,000 images  ViT    step 500: 41.5% / loss 1.657  step 1500: 57.7% / loss 1.230  step 2500: 63.8% / loss 1.059   best 63.8% at step 2500
  50,000 images  ResNet step 500: 55.0% / loss 1.047  step 1500: 79.7% / loss 0.594  step 2500: 84.2% / loss 0.432   best 84.2% at step 2500
```

With 1,000 pictures, both models drive the training loss to almost zero: they know every training picture by heart. But the ViT's validation accuracy stops near 38% to 39%, while the ResNet's reaches 54%. Both memorise; the ResNet's built-in assumptions make what it memorises more useful for new pictures. That is "the convolutional inductive bias is useful for smaller datasets", measured.

**More data helps both, and the gap does not close here.** From 1,000 to 50,000 pictures, the ViT gains 24.7 points and the ResNet 29.7. With 50,000 pictures, 2,500 steps are only 6 passes over the data, and both curves are still rising at the end. The fixed recipe now limits the result, and it limits the slower learner, the ViT, more. The paper does not say how many passes it made over each JFT subset. We could not afford longer runs on a shared laptop GPU.

What this experiment cannot show: the cross-over itself, which the paper finds between 9 million and 90 million pictures. Each point is one run, so differences of a point or two are noise. And CIFAR-10 pictures are 32×32, far from the 224×224 photos of JFT. What it does show is the starting point of the paper's story. At small scale, with the same recipe and the same cost, a ResNet beats a ViT clearly.

### Few-shot on the released models: our small analogue of Figure 4

We cannot pre-train on JFT subsets. But we can run Figure 4's measuring stick, the few-shot linear test, on models that Google released. The script [`vit_part4_fewshot.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part4_fewshot.py) freezes seven models and tests them on two datasets of Table 2:

- **the five ViT sizes of Figure 3** (B/32, B/16, L/32, L/16, H/14), all released after ImageNet-21k pre-training, with no fine-tuning;
- **two ResNet-50s** as reference points: `google/bit-50`, a BiT R50x1 that ends in an ImageNet-1k head, and `microsoft/resnet-50`, trained on ImageNet-1k only;
- **CIFAR-100** (100 classes, 1,000 test pictures, 10 per class) and **Oxford-IIIT Pets** (37 breeds, 1,110 test pictures, 30 per class).

Every picture is resized to 224×224. For a ViT, the representation is its "pre-logits": the dense + tanh layer on the class token, which feeds the pre-training head. For a ResNet, it is the 2,048 pooled numbers.

For each $$k$$, we draw 5 random sets of $$k$$ pictures per class from a pool of 20 per class. Each set gets the closed form above. The penalty $$\lambda$$ is chosen once per model and dataset, from nine values, on 5 other training pictures per class. No test picture is ever used to choose anything.

Here is what "1-shot" means on Pets: the whole training set of the first 1-shot draw, one picture per breed.

<figure><img src="/img/papers/vit/p4-pets-oneshot.png" alt="A grid of 37 small photos, one per breed of the Oxford-IIIT Pets dataset, each labelled with its breed: abyssinian, american bulldog, american pit bull terrier, basset hound, beagle, bengal, birman, bombay, boxer, british shorthair, chihuahua, egyptian mau, english cocker spaniel, english setter, german shorthaired, great pyrenees, havanese, japanese chin, keeshond, leonberger, maine coon, miniature pinscher, newfoundland, persian, pomeranian, pug, ragdoll, russian blue, saint bernard, samoyed, scottish terrier, shiba inu, siamese, sphynx, staffordshire bull terrier, wheaten terrier and yorkshire terrier" loading="lazy" /><figcaption>The complete training set of one 1-shot draw on Oxford-IIIT Pets: 37 pictures, one per breed. The frozen model never sees any other labelled picture of these breeds.</figcaption></figure>

The results, mean ± spread over the 5 draws:

```text
-- CIFAR-100 (100 classes, chance = 1.0%) --
  model                      dim  lambda        1-shot        2-shot        5-shot       10-shot
  ViT-B/32                   768       1    53.9 ± 2.6    67.1 ± 1.1    77.8 ± 0.8    81.3 ± 0.5
  ViT-B/16                   768       1    54.4 ± 2.0    68.1 ± 0.3    77.2 ± 0.4    80.6 ± 0.7
  ViT-L/32                  1024       1    55.1 ± 2.8    68.7 ± 0.8    78.3 ± 0.9    81.1 ± 0.3
  ViT-L/16                  1024     0.1    64.0 ± 3.3    74.9 ± 1.5    82.2 ± 0.9    84.3 ± 0.3
  ViT-H/14                  1280       1    46.8 ± 2.6    59.8 ± 1.7    69.8 ± 0.7    74.9 ± 0.7
  BiT R50x1                 2048     0.1    42.2 ± 2.6    54.0 ± 1.2    64.0 ± 1.3    69.4 ± 0.4
  ResNet-50 (ImageNet-1k)   2048     0.1    26.1 ± 2.5    37.0 ± 2.2    49.2 ± 1.2    56.1 ± 1.0

-- Oxford-IIIT Pets (37 classes, chance = 2.7%) --
  model                      dim  lambda        1-shot        2-shot        5-shot       10-shot
  ViT-B/32                   768     100    67.5 ± 6.1    77.1 ± 2.5    82.0 ± 0.6    83.3 ± 0.7
  ViT-B/16                   768     0.1    75.0 ± 4.6    81.3 ± 1.9    85.7 ± 0.6    87.6 ± 0.6
  ViT-L/32                  1024       1    65.7 ± 3.5    74.6 ± 2.1    79.5 ± 0.7    81.9 ± 0.2
  ViT-L/16                  1024     0.1    74.6 ± 3.0    80.7 ± 1.9    84.6 ± 0.5    86.9 ± 0.2
  ViT-H/14                  1280     0.1    66.6 ± 5.1    77.6 ± 1.7    83.0 ± 0.7    85.2 ± 0.7
  BiT R50x1                 2048     0.1    66.8 ± 4.5    77.7 ± 1.4    84.7 ± 1.2    87.6 ± 1.2
  ResNet-50 (ImageNet-1k)   2048       1    76.6 ± 5.2    82.8 ± 2.1    86.6 ± 1.1    88.9 ± 0.6
```

{{FIG:p4_fewshot|Our few-shot linear accuracy on frozen features, against labelled pictures per class (log scale). On CIFAR-100 every ViT pre-trained on ImageNet-21k is above both ResNet-50s at every k. On Pets the ResNet-50 trained on ImageNet-1k is on top at 10-shot (88.9%), and ViT-B/16 and the BiT ResNet share second place (87.6%).}}

**First, few-shot works.** With one picture of each of 100 classes, the frozen ViT-L/16 names 64.0% of CIFAR-100 test pictures correctly; chance is 1%. With 10 pictures per class it reaches 84.3%. For comparison, Table 2 reports 93.25% for the same kind of model after full fine-tuning on all 50,000 CIFAR-100 training pictures. Our number uses 1,000 labels and no training loop. That is the "very low-data transfer" of the paper's last paragraph.

**Second, on CIFAR-100 the ImageNet-21k ViTs win clearly.** At 10-shot, the five ViTs score 74.9% to 84.3%, the two ResNet-50s 69.4% and 56.1%. The closest to a fair pair is the one the paper names: "ViT-B/32 is slightly faster than ResNet50". Here ViT-B/32 scores 81.3% against 69.4% for the BiT ResNet. Two warnings: we cannot confirm which data `google/bit-50` was pre-trained on, and `microsoft/resnet-50` saw only 1.3 million pictures. So this is a comparison of released checkpoints, not of designs.

**Third, Pets turns the order around, and the reason is the labels the models saw.** At least 20 of the 37 Pets breeds are ImageNet-1k classes with the same name, such as "beagle", "pug" and "Persian cat". Both ResNet-50s were trained with exactly those labels, so their features already separate the breeds:

```text
  20 of 37 breeds have an ImageNet-1k class of the same name: american pit bull terrier, basset hound, beagle, boxer, chihuahua, english cocker spaniel, english setter, great pyrenees, keeshond, miniature pinscher, newfoundland, persian, pomeranian, pug, saint bernard, samoyed, scottish terrier, siamese, staffordshire bull terrier, yorkshire terrier
```

The ViTs saw ImageNet-21k's 21,000 classes, a much broader set that contains the ImageNet-1k classes among many others. The ImageNet-1k ResNet-50 is on top (88.9% at 10-shot), with ViT-B/16 (87.6%) and the BiT ResNet (87.6%) next. On CIFAR-100, whose classes are broad ones such as "apple", "cloud" or "bridge", the same ResNet-50 is last. A few-shot score measures how well the pre-training labels match the new task, as well as the model.

**Fourth, bigger is not always better on ImageNet-21k.** ViT-H/14 is below ViT-L/16 on both datasets (74.9% against 84.3% on CIFAR-100 at 10-shot). This matches the paper's own numbers for this pre-training set. In Table 5, after ImageNet-21k, ViT-H/14 only ties ViT-L/16 on ImageNet (85.13 against 85.15). The paper says that larger models show their full benefit "only with JFT-300M". Our run fits that pattern: 14 million pictures seem too few for a 632-million-parameter model.

We checked that our 16-bit arithmetic does not explain it. On four test pictures, ViT-H/14's features in 16-bit and 32-bit arithmetic point the same way:

```text
== 4. does float16 change the features? ViT-H/14 on 4 CIFAR-100 test pictures, float16 on the GPU against float32 on the CPU ==
  cosine similarity float16 vs float32, per picture: 1.000000, 0.999999, 1.000000, 1.000000
```

A cosine similarity of 1 means the two feature vectors point in exactly the same direction. The ranking of the models is real for this protocol.

**Fifth, the patch size matters less on CIFAR-100.** ViT-B/32 (81.3%) is level with ViT-B/16 (80.6%) at 10-shot. The paper's Table 5 shows the same thing after ImageNet-21k pre-training: ViT-B/32 91.97 against ViT-B/16 91.67 on CIFAR-100. One likely reason: a CIFAR picture has only 32×32 real pixels, scaled up to 224, so a smaller patch finds no finer detail to read. That is our reading, not a claim of the paper.

What this experiment cannot show: it uses no JFT-300M, no fine-tuning and small test sets, and the checkpoints are not the paper's exact runs. The ± values are the spread over 5 draws of the training pictures (at $$k = 10$$ they are below 1.3 points), not over test sets. The one thing it can show is the paper's practical point: a large pre-trained ViT gives useful features from a handful of labelled pictures.

> [!PAPER] · §4.3
> [![The last paragraph of Section 4.3: overall, the few-shot results on ImageNet (Figure 4), as well as the low-data results on VTAB (Table 2) seem promising for very low-data transfer. Further analysis of few-shot properties of ViT is an exciting direction of future work](/img/papers/vit/p4-outlook.png)](/img/papers/vit/p4-outlook.png)
>
> **What it says:** the few-shot results and the VTAB results "seem promising for very low-data transfer", and "further analysis of few-shot properties of ViT is an exciting direction of future work".
>
> **What it means:**
>
> Note the turn in this sentence. The whole section said that ViT needs a lot of data. That is true of pre-training. Once pre-trained on a lot of data, the model needs very little data for a new task. VTAB gives it 1,000 examples, and Figure 4 gives it 5 per class.
>
> So the data cost is paid once, up front, and every later task is cheap. This is the same trade that language models made with BERT. Our own few-shot runs above show the same pattern on released checkpoints.

## Appendix C: the numbers behind Figure 3 {§C}

> [!PAPER] · Appendix C
> [![The start of Appendix C, Additional Results: we report detailed results corresponding to the figures presented in the paper. Table 5 corresponds to Figure 3 from the paper and shows transfer performance of different ViT models pre-trained on datasets of increasing size: ImageNet, ImageNet-21k, and JFT-300M](/img/papers/vit/p4-appc.png)](/img/papers/vit/p4-appc.png)
> [![Table 5 of the ViT paper: top-1 accuracy in percent of ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32 and ViT-H/14 on CIFAR-10, CIFAR-100, ImageNet, ImageNet ReaL, Oxford Flowers-102 and Oxford-IIIT Pets, after pre-training on ImageNet, ImageNet-21k or JFT-300M. Caption: these values correspond to Figure 3 in the main text. Models are fine-tuned at 384 resolution. Note that the ImageNet results are computed without additional techniques (Polyak averaging and 512 resolution images) used to achieve results in Table 2](/img/papers/vit/p4-table5.png)](/img/papers/vit/p4-table5.png)
>
> **What it says:** "Table 5 corresponds to Figure 3." Its models "are fine-tuned at 384 resolution", and its ImageNet results are "computed without additional techniques (Polyak averaging and 512 resolution images) used to achieve results in Table 2".
>
> **What it means:**
>
> Table 5 is Figure 3 as numbers, plus five more fine-tuning datasets. We drew three of them above. The caption also explains a small puzzle: why ViT-H/14 on JFT scores 88.04 here but 88.55 in Table 2.
>
> The Table 2 runs used two extra tricks for ImageNet: a higher fine-tuning resolution (512 for ViT-L/16, 518 for ViT-H/14) and Polyak averaging. Table 5 uses neither. So the gap between the two tables measures what the tricks are worth.

> [!DEFINITION] Polyak averaging
> Keeping a running average of the model's weights during fine-tuning and using the average, not the last weights, at test time. With a factor of 0.9999, each step does $$\bar\theta \leftarrow 0.9999\,\bar\theta + 0.0001\,\theta$$, where $$\theta$$ are the current weights and $$\bar\theta$$ the average. The average mostly covers the last $$1/(1 - 0.9999) = 10{,}000$$ steps, which smooths out the noise of single steps.

```text
== 4. Table 2 versus Table 5: ImageNet with and without Polyak averaging and 512/518 px fine-tuning ==
  ViT-H/14 (JFT)   Table 2: 88.55   Table 5 (384 px, no averaging): 88.04   difference +0.51
  ViT-L/16 (JFT)   Table 2: 87.76   Table 5 (384 px, no averaging): 87.12   difference +0.64
  ViT-L/16 (I21k)  Table 2: 85.30   Table 5 (384 px, no averaging): 85.15   difference +0.15
```

The two tricks are worth half a point for the JFT models, and 0.15 for the ImageNet-21k model. So part of the headline 88.55% is fine-tuning craft, not architecture. ViT-H/14 would still beat BiT-L's 87.54 without the tricks (88.04), but it would no longer pass Noisy Student's 88.4.

## Appendix D.1: Adam or SGD for the ResNets? {§D.1}

> [!PAPER] · Appendix D.1
> [![Appendix D.1, SGD vs. Adam for ResNets: ResNets are typically trained with SGD and our use of Adam as optimizer is quite unconventional. Here we show the experiments that motivated this choice](/img/papers/vit/p4-d1.png)](/img/papers/vit/p4-d1.png)
> [![Appendix D.1 continued: we compare the fine-tuning performance of two ResNets, 50x1 and 152x2, pre-trained on JFT with SGD and Adam. For SGD, we use the hyperparameters recommended by Kolesnikov et al. Results are presented in Table 7. Adam pre-training outperforms SGD pre-training on most datasets and on average. Note that the absolute numbers are lower than those reported by Kolesnikov et al., since we pre-train only for 7 epochs, not 30](/img/papers/vit/p4-d1-text.png)](/img/papers/vit/p4-d1-text.png)
>
> **What it says:** "ResNets are typically trained with SGD and our use of Adam as optimizer is quite unconventional." Two ResNets were pre-trained on JFT with each optimizer and then fine-tuned. "Adam pre-training outperforms SGD pre-training on most datasets and on average." The numbers are lower than BiT's own "since we pre-train only for 7 epochs, not 30".
>
> **What it means:**
>
> This appendix answers a reviewer's natural worry. If the ResNet baselines were trained with an unusual optimizer, maybe they were handicapped, and ViT's win is partly a weak opponent.
>
> The test is direct: train the same ResNet both ways and compare. Adam comes out slightly ahead, so the authors kept one optimizer, Adam, for every model. That keeps the comparison simple: the only thing that changes between rows is the network.

> [!DEFINITION] SGD and Adam
> Two rules for updating weights from their gradients. **SGD** (with momentum) moves every weight by the same learning rate times a smoothed gradient. **Adam** also keeps a running size of each weight's gradients and divides by it, so each weight gets its own step size. ResNets were traditionally trained with SGD; Transformers with Adam. [Part 3](part-3-fine-tuning-and-setup.md) gives the paper's settings.

> [!PAPER] · Table 7
> [![Table 7: fine-tuning ResNet models pre-trained with Adam and SGD. ResNet50 Adam and SGD, ResNet152x2 Adam and SGD, on ImageNet 77.54, 78.24, 84.97, 84.37; CIFAR10 97.67, 97.46, 99.06, 99.07; CIFAR100 86.07, 85.17, 92.05, 91.06; Oxford-IIIT Pets 91.11, 91.00, 95.37, 94.79; Oxford Flowers-102 94.26, 92.06, 98.62, 99.32; Average 89.33, 88.79, 94.01, 93.72](/img/papers/vit/p4-table7.png)](/img/papers/vit/p4-table7.png)
>
> **What it says:** "Fine-tuning ResNet models pre-trained with Adam and SGD": five datasets, two ResNets, two optimizers, and an average row.
>
> **What it means:**
>
> Read across each row: the left pair is ResNet-50, the right pair ResNet152x2. Adam is ahead on the average row for both models, by about half a point and a third of a point.
>
> The rows disagree with each other, though. With SGD, ResNet-50 is better on ImageNet and ResNet152x2 is better on Flowers. "On most datasets" is the honest phrase. Adam wins 7 of the 10 comparisons, as the check below shows.

```text
== 7. Table 7 (Appendix D.1): ResNets pre-trained on JFT with Adam or SGD, then fine-tuned ==
                       R50 Adam   R50 SGD  Adam-SGD  R152x2 Adam  R152x2 SGD  Adam-SGD
  ImageNet                77.54     78.24     -0.70        84.97       84.37     +0.60
  CIFAR10                 97.67     97.46     +0.21        99.06       99.07     -0.01
  CIFAR100                86.07     85.17     +0.90        92.05       91.06     +0.99
  Oxford-IIIT Pets        91.11     91.00     +0.11        95.37       94.79     +0.58
  Oxford Flowers-102      94.26     92.06     +2.20        98.62       99.32     -0.70
  average (ours)          89.33     88.79     +0.54        94.01       93.72     +0.29
  average (table)         89.33     88.79                  94.01       93.72
  Adam is higher in 7 of 10 dataset-model pairs
```

Our averages match the table to the last digit. One limit is worth stating: the test uses short pre-training (7 epochs). It shows that Adam is not a handicap in the paper's setting. It does not show that Adam beats SGD for ResNets in general.

## Appendix D.9: ObjectNet {§D.9}

> [!PAPER] · Appendix D.9
> [![Appendix D.9, ObjectNet Results: we also evaluate our flagship ViT-H/14 model on the ObjectNet benchmark following the evaluation setup in Kolesnikov et al. (2020), resulting in 82.1% top-5 accuracy and 61.7% top-1 accuracy](/img/papers/vit/p4-d9.png)](/img/papers/vit/p4-d9.png)
>
> **What it says:** ViT-H/14 was evaluated on "the ObjectNet benchmark", with BiT's evaluation setup, "resulting in 82.1% top-5 accuracy and 61.7% top-1 accuracy".
>
> **What it means:**
>
> One sentence, and a sobering number. The same model that scores 88.55% on ImageNet scores 61.7% top-1 on ObjectNet. That is 26.85 points lower, on pictures of the same kinds of everyday objects.
>
> The drop is not special to ViT; every ImageNet model falls on ObjectNet. The point of the test is robustness: does the model recognise a chair when it lies on its side, on a bed, photographed from below?

> [!PAPER] Barbu et al. (2019), ObjectNet · Abstract
> [![The abstract of ObjectNet: we collect a large real-world test set, ObjectNet, for object recognition with controls where object backgrounds, rotations, and imaging viewpoints are random. When tested on ObjectNet, object detectors show a 40-45% drop in performance, with respect to their performance on other benchmarks, due to the controls for biases. ObjectNet is the same size as the ImageNet test set, 50,000 images, and by design does not come paired with a training set](/img/papers/vit/p4-ext-objectnet.png)](/img/papers/vit/p4-ext-objectnet.png)
>
> **What it says:** ObjectNet is a test set where "backgrounds, rotations, and imaging viewpoints are random". Models show "a 40-45% drop in performance" against other benchmarks. It is as large as the ImageNet test set and "does not come paired with a training set".
>
> **What it means:**
>
> Photos on the web are posed: mugs stand upright on tables, chairs face the camera. Models learn these habits along with the objects. ObjectNet's photographers were told to place each object at a random angle, on a random background, from a random viewpoint, so the habits stop working.
>
> No training set means no fine-tuning. A model must arrive with whatever robustness it has. ObjectNet shares 113 of its classes with ImageNet, and models are scored on those classes only.

How does ViT-H/14 compare with BiT-L on the same test? The ViT paper does not say, but the BiT paper lists BiT-L's score:

> [!PAPER] Kolesnikov et al. (2020), Big Transfer (BiT) · Table 6
> [![Table 6 of the BiT paper: results in percent on the ObjectNet test set, top-1 and top-5 accuracy, resize and crop or bounding box, for BiT-S, BiT-M and BiT-L ResNets from R50x1 to R152x4. BiT-L R152x4 reaches 58.7 top-1 and 80.0 top-5 with resize and crop](/img/papers/vit/p4-ext-bit-objectnet.png)](/img/papers/vit/p4-ext-bit-objectnet.png)
>
> **What it says:** "Results (%) on the ObjectNet test set", for BiT models of five sizes and three pre-training sets. With the standard "resize and crop" setting, BiT-L R152x4 gets 58.7% top-1 and 80.0% top-5.
>
> **What it means:**
>
> The ViT paper used "the evaluation setup in Kolesnikov et al. (2020)", which is this table's resize-and-crop setting. So the two numbers can be compared directly.
>
> ViT-H/14 is ahead by 3.0 points top-1 (61.7 against 58.7) and 2.1 points top-5 (82.1 against 80.0). The table also shows the paper's theme from the CNN side. More pre-training data helps a big model far more than a small one. From BiT-S to BiT-L, R50x1 gains 6.8 points top-1 and R152x4 gains 22.7.

```text
== 8. ObjectNet (Appendix D.9) ==
  ViT-H/14: ObjectNet top-1 61.7%, top-5 82.1%;  ImageNet top-1 (Table 2) 88.55%
  drop from ImageNet to ObjectNet, top-1: 26.85 points
  BiT-L (R152x4, JFT), the BiT paper, Table 6, resize and crop: ObjectNet top-1 58.7%, top-5 80.0%
  ViT-H/14 minus BiT-L on ObjectNet: top-1 +3.0, top-5 +2.1 points
  BiT paper, Table 6, top-1, BiT-S -> BiT-L: R50x1 30.8 -> 37.6 (+6.8), R152x4 36.0 -> 58.7 (+22.7)
```

## Appendix D.10: the VTAB breakdown {§D.10}

> [!PAPER] · Table 9
> [![Table 9 of the ViT paper: breakdown of VTAB-1k performance across tasks, for ViT-H/14 (JFT), ViT-L/16 (JFT) and ViT-L/16 (I21k). Columns: Caltech101, CIFAR-100, DTD, Flowers102, Pets, Sun397, SVHN (Natural, red dots), Camelyon, EuroSAT, Resisc45, Retinopathy (Specialized, green dots), Clevr-Count, Clevr-Dist, DMLab, dSpr-Loc, dSpr-Ori, KITTI-Dist, sNORB-Azim, sNORB-Elev (Structured, blue dots) and the Mean: 77.6, 76.3 and 72.7](/img/papers/vit/p4-table9.png)](/img/papers/vit/p4-table9.png)
>
> **What it says:** a "Breakdown of VTAB-1k performance across tasks": the 19 task scores and their mean for ViT-H/14 (JFT), ViT-L/16 (JFT) and ViT-L/16 (I21k).
>
> **What it means:**
>
> The coloured dots under the task names mark the groups of Figure 2: red for Natural, green for Specialized, blue for Structured. We used them above to rebuild Figure 2's ViT bars.
>
> The table shows how uneven VTAB is. Flowers102 is at 99.7 for both JFT models, while sNORB-Azim (the angle of a toy, photographed from different sides) is at 33.2 to 36.2. One average over 19 such tasks hides a lot. The per-task view shows where the extra data went.

From ViT-L/16 on ImageNet-21k to ViT-H/14 on JFT, the largest gains are on Structured tasks. The smallest are on tasks that were already easy:

```text
  ViT-H/14 (JFT) minus ViT-L/16 (I21k), largest gaps: Clevr-Count +21.4, DMLab +11.2, sNORB-Elev +9.5, SVHN +8.0
  ... smallest: Camelyon +0.8, Flowers102 +0.4, dSpr-Ori -1.6
```

Clevr-Count (count the objects in a synthetic scene) gains 21.4 points. Flowers102 gains 0.4, because it was already at 99.3. Bigger data and a bigger model mostly help where the task is far from photos of named objects. Note that these gaps mix two changes, the model size and the pre-training set, so they cannot be split between the two.

> [!TAKEAWAYS] Key takeaways
> - **Table 2:** ViT-H/14 pre-trained on JFT-300M is best on 6 of 7 benchmarks: 88.55% ImageNet, 94.55% CIFAR-100, 77.63% VTAB. It costs 2.5k TPUv3-core-days, 4.0 times less than BiT-L and 4.9 times less than Noisy Student.
> - **Same data, different network:** ViT-L/16 (JFT) against BiT-L gives 5 wins, 1 tie and one row 0.01 lower, for 14.6 times less compute. That row is VTAB, well inside BiT-L's spread of 1.70. The compute numbers come from different recipes; Section 4.4 controls for that.
> - **Count errors, not accuracy, near 100%:** ViT-H/14 makes 2% (ImageNet ReaL) to 28% (Pets) fewer mistakes than BiT-L.
> - **The public model:** ViT-L/16 on ImageNet-21k reaches 85.30% on ImageNet for 0.23k core-days, about one rented 8-core TPU for 30 days.
> - **VTAB:** for ViT-H/14 the groups average 86.7 (Natural), 87.0 (Specialized) and 65.0 (Structured). Structured tasks are the hardest and gain the most from more data and a bigger model (Clevr-Count +21.4).
> - **Section 4.3:** ViT-L/16 is 1.38 points below ViT-B/16 after ImageNet pre-training, 1.18 above after ImageNet-21k and 2.97 above after JFT-300M. On JFT subsets, ResNets win at 9M pictures but plateau; ViTs win at 90M and more.
> - **Few-shot linear evaluation** is one formula, $$W = (X^\top X + \lambda I)^{-1} X^\top Y$$, on frozen features. On released ImageNet-21k models, ViT-L/16 names 64.0% of CIFAR-100 test pictures from one example per class, 84.3% from ten.
> - **Our from-scratch runs on CIFAR-10** (1,000 to 50,000 pictures, matched cost) show the small-data side of the story. The ResNet beats the ViT at every size, by 15.5 to 20.5 points. With 1,000 pictures both memorise the training set, but the ViT gets only 39.0% of new pictures right, against 54.6%.
> - **Appendix:** Polyak averaging and 512/518 px fine-tuning add 0.51 to 0.64 points to Table 2; Adam beats SGD for the ResNet baselines on average (+0.54 and +0.29). On ObjectNet ViT-H/14 gets 61.7% top-1 and 82.1% top-5, against 58.7% and 80.0% for BiT-L.

**Next, in Part 5 (coming soon):** Section 4.4 compares ViT, ResNets and hybrids at equal pre-training compute (Figure 5, Table 6). Then Section 4.5 opens the trained model: its patch filters, the position embeddings and how far its attention reaches, recomputed on the released weights. Last, Section 4.6 tries pre-training without labels. Part 5 is coming soon.

<details>
<summary>Run it yourself</summary>

Every number in this part that is not from the paper comes from the scripts in [`code/papers/vit/`](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vit). `vit_part4_tables.py` runs no model; it recomputes the paper's tables. The other three download released checkpoints from Hugging Face. We ran them on an Apple M5 Pro GPU that other jobs shared, so we make no timing claims.

```bash
pip install torch transformers pyarrow pillow
python vit_part4_tables.py     # Table 2, 5, 7, 9 arithmetic and ObjectNet; no model
python vit_part4_fewshot.py    # the closed form by hand, then 7 frozen models on CIFAR-100 and Pets
python vit_part4_top5.py       # top-1 and top-5 of 4 ImageNet classifiers on Imagenette
python vit_part4_scratch.py    # a small ViT and a small ResNet from scratch on CIFAR-10 subsets
python vit_part4_scratch_curves.py   # prints the training curves saved by the run above
python vit_part4_pictures.py   # the 1-shot Pets picture
```

<figure><img src="/img/papers/vit/part4_tables-run.png" alt="Terminal output of vit_part4_tables.py: TPUv3-core-days in other units and the compute ratios, Table 2 as error rates, ViT-L/16 against BiT-L row by row, and the start of the Table 2 versus Table 5 comparison" loading="lazy" /><figcaption>The real output of vit_part4_tables.py (first 60 lines).</figcaption></figure>

<figure><img src="/img/papers/vit/part4_fewshot-run.png" alt="Terminal output of vit_part4_fewshot.py: the tiny closed-form example, the dataset sizes, the few-shot accuracy tables for CIFAR-100 and Oxford-IIIT Pets, the breed overlap with ImageNet-1k and the float16 check" loading="lazy" /><figcaption>The real output of vit_part4_fewshot.py.</figcaption></figure>

<figure><img src="/img/papers/vit/part4_scratch-run.png" alt="Terminal output of vit_part4_scratch.py: the two models with their parameters and multiply-adds, the training recipe, and the best validation accuracy of each model for each CIFAR-10 subset size" loading="lazy" /><figcaption>The real output of vit_part4_scratch.py.</figcaption></figure>

</details>

## The paper and the code

[The paper (arXiv 2010.11929)](https://arxiv.org/abs/2010.11929) · [Official code and checkpoints](https://github.com/google-research/vision_transformer) · [Our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vit)
