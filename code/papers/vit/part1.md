---
title: "The Big Idea: An Image Is Worth 16×16 Words"
description: "The title, the abstract and the introduction of the Vision Transformer paper, line by line: what it means to cut a picture into 16×16 patches and read them like words, why convolutional networks ruled computer vision, what inductive bias, locality and translation equivariance are, and the claim that enough data beats built-in assumptions. With a real run of ViT-B/16 and ResNet-50 on the same picture and the quadratic cost of attention worked out by hand."
part: 1
covers: "Title, Abstract, §1"
date: 2026-10-06
tags: [vit, vision-transformer, transformers, computer-vision]
---

On 22 October 2020, twelve researchers at Google posted a paper with an odd title: *An Image is Worth 16×16 Words*. It took the Transformer, the network design that had taken over language processing, cut photographs into small squares, and fed the squares to the Transformer as if they were words. On the standard photo benchmark, ImageNet, the model matched or beat the best convolutional networks of the day, once it had been trained on enough pictures. The paper was accepted at ICLR 2021, and the model it introduced, the Vision Transformer (ViT), is today the base of most image models and of the image half of models that read both text and pictures.

This series reads the paper slowly, in the paper's own order. Each piece follows the same pattern:

1. **The exact lines from the paper**, as a highlighted screenshot in a teal box like the one below.
2. **A plain-English explanation**, with a yellow box for every new word.
3. **A picture**, and **real code** when it helps, with its real output.
4. **Why it matters**, and only then the next piece.

The small `Paper §1` tag above each heading tells you which section of the paper you are reading. The screenshots come from the paper's second arXiv version (3 June 2021), the one most people read today. We run every experiment on the model weights Google released, through the Hugging Face `transformers` library, on an ordinary laptop.

> [!DEFINITION] Definition boxes
> A yellow box like this explains one technical word in plain language, the first time it appears. If you already know the word, skip the box.

## The title {§Title}

> [!PAPER] Dosovitskiy et al. (2020), ViT · Title · page 1
> [![The title of the ViT paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, with 16x16 Words and Image Recognition at Scale highlighted](/img/papers/vit/p1-title.png)](/img/papers/vit/p1-title.png)
>
> **Context:** the first thing on page 1. Above it, in small print, the page says "Published as a conference paper at ICLR 2021". ICLR is one of the three big machine-learning conferences.
>
> **What it says:** "An Image is Worth **16×16 Words**: Transformers for **Image Recognition at Scale**."
>
> **Why it matters:** the title is a joke on the proverb "a picture is worth a thousand words", and the joke is exactly the method. A picture is turned into a sequence of words, and each "word" is a 16×16 square of pixels.

Let us take the title apart, one piece at a time.

- **An image.** A digital picture. For this paper, a photograph of one main thing: a cat, a car, a bird.
- **16×16 words.** The picture is cut into squares of 16 pixels by 16 pixels. Each square is treated as one "word". A 224×224 picture gives 14×14 = 196 such squares. The paper calls them **patches**.
- **Transformers.** The kind of neural network used. It was introduced in 2017 for translating sentences, and by 2020 it was the standard network for text. The paper uses it with almost no changes.
- **Image recognition.** The task: look at a picture and say what is in it. The paper's version of the task is **image classification**: choose one label out of a fixed list (for ImageNet, one of 1,000 labels).
- **At scale.** The key words. The method only works well when the model is trained on very many pictures: 14 million to 300 million. On the "ordinary" 1.3 million pictures of ImageNet, it loses to convolutional networks. This is the paper's main finding, and most of the paper is about it.

> [!DEFINITION] Image recognition and image classification
> **Image recognition** is the general job of understanding what a picture shows. **Image classification** is the simplest version: the model gets a picture and must pick one label from a fixed list ("tabby cat", "sports car", "goldfinch"). It does not say *where* the thing is, only *what* it is.

> [!DEFINITION] Pixel
> One dot of a digital picture. A 224×224 picture has 224 rows of 224 dots, 50,176 in all. Each dot stores a colour as numbers.

> [!DEFINITION] Channel (RGB)
> A colour pixel is stored as three numbers: how much red, how much green and how much blue. Each of the three is a **channel**. So a 224×224 colour picture is really 3 × 224 × 224 = 150,528 numbers.

> [!DEFINITION] Patch
> A small square cut out of the picture, here 16 pixels by 16 pixels by 3 channels, that is 768 numbers. ViT treats each patch as one token.

> [!DEFINITION] Token
> One unit of input to a Transformer. In language, a token is a word or a piece of a word. In ViT, a token is a patch. The model never sees words or pictures directly: it sees a list of tokens, each turned into a list of numbers.

> [!DEFINITION] Transformer
> A neural network design built around **attention**: every token computes how much it should "look at" every other token, and mixes in information from the tokens it looks at. If attention is new to you, the [attention series](/writings/attention-1-self-attention/) explains it from zero. For this paper you only need the idea: tokens look at other tokens. ViT uses the same **encoder** half of the Transformer as [BERT](/papers/bert/).

{{FIG:p1_timeline|Two lines of research meet. In vision, AlexNet (2012) and ResNet (2015) made convolutional networks the standard. In language, the Transformer (2017) and BERT (2018) replaced older designs and introduced the pre-train-then-fine-tune recipe. ViT (2020) takes the language design, unchanged, to pictures.}}

The title's analogy is worth drawing out, because the whole paper rests on it. A language model reads a sentence as a sequence of word tokens, each turned into a vector of numbers. ViT reads a picture as a sequence of patch tokens, each turned into a vector of numbers of the same size. After that step, the Transformer cannot tell whether it is reading a sentence or a photo.

{{FIG:p1_analogy|The analogy of the title. Left: a sentence becomes six word tokens, and each token becomes a vector of 768 numbers. Right: a picture is cut into nine patches, each patch becomes a token, and each token becomes a vector of 768 numbers. The same Transformer reads both.}}

Here is the paper's own drawing of the model, which Part 2 explains block by block. We show it now because it makes the title concrete.

> [!PAPER] Dosovitskiy et al. (2020), ViT · Figure 1 · page 3
> [![Figure 1 of the ViT paper: a picture is cut into nine patches, the patches are flattened and passed through a linear projection, a class token and position embeddings are added, the sequence goes through a Transformer encoder, and an MLP head on the class token outputs the class. On the right, the Transformer encoder block with layer norm, multi-head attention, an MLP and residual connections](/img/papers/vit/p1-figure1.png)](/img/papers/vit/p1-figure1.png)
>
> **Context:** Figure 1, at the top of Section 3 (page 3). It is the only diagram of the model in the paper.
>
> **What it says:** "We split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder." For classification, an extra learnable "classification token" is added to the sequence, "the standard approach" from BERT.
>
> **Why it matters:** look at the bottom row of the left panel: nine small photo squares, numbered 1 to 9, going into a box. That row *is* the title. Everything above the box is a Transformer encoder copied from the language world.

## The authors and the footnotes {§Title}

> [!PAPER] Dosovitskiy et al. (2020), ViT · Authors · page 1
> [![The author list of the ViT paper: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby, with footnotes for equal technical contribution and equal advising, and the affiliation Google Research, Brain Team](/img/papers/vit/p1-authors.png)](/img/papers/vit/p1-authors.png)
>
> **Context:** under the title. Twelve authors, one affiliation, two kinds of footnote mark.
>
> **What it says:** the asterisk marks "equal technical contribution": five people (Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai) plus Houlsby share the credit for building and running the experiments. The dagger marks "equal advising": Dosovitskiy and Houlsby led the project. The affiliation is "Google Research, Brain Team", and the two contact e-mails belong to the two leads.
>
> **Why it matters:** this is a large-team industrial paper. Several of the authors (Kolesnikov, Beyer, Zhai, Houlsby) had just written the Big Transfer paper on scaling up convolutional networks with the same giant datasets, and that paper's models are the main baseline here. One author, Jakob Uszkoreit, is also an author of the original Transformer paper.

The order of the names matters less than usual: six of the twelve are marked as equal contributors. When this series says "the authors" it means the team.

There is one footnote on page 1, attached to the last sentence of the abstract.

> [!PAPER] Dosovitskiy et al. (2020), ViT · Footnote 1 · page 1
> [![Footnote 1 of the ViT paper: Fine-tuning code and pre-trained models are available at https://github.com/google-research/vision_transformer](/img/papers/vit/p1-footnote.png)](/img/papers/vit/p1-footnote.png)
>
> **Context:** footnote 1, at the bottom of page 1.
>
> **What it says:** "Fine-tuning code and pre-trained models are available at" the `google-research/vision_transformer` repository on GitHub.
>
> **Why it matters:** the released weights are the reason this series can run the real model. The Hugging Face checkpoints we use (`google/vit-base-patch16-224` and its relatives) are conversions of those files. Note the wording: *fine-tuning* code. The pre-training itself ran on Google's private JFT-300M dataset and is not something you can repeat at home.

## The abstract, sentence by sentence {§Abstract}

The abstract is one paragraph of four sentences. We read it in two halves.

> [!PAPER] Dosovitskiy et al. (2020), ViT · Abstract · page 1
> [![The first half of the ViT abstract: while the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place](/img/papers/vit/p1-abstract-1.png)](/img/papers/vit/p1-abstract-1.png)
>
> **Context:** the first two sentences of the abstract: the state of the field in 2020.
>
> **What it says:** the Transformer is "the de-facto standard" for language, but "its applications to computer vision remain limited". Where vision does use attention, it is either "in conjunction with convolutional networks" or used to "replace certain components of convolutional networks while keeping their overall structure in place".
>
> **Why it matters:** the two sentences set up the gap. Vision had borrowed attention, but always inside a convolutional design. Nobody had thrown the convolutions away and shown that the result works.

**Sentence 1** says that in language the question was settled. "De-facto standard" means "the standard in practice, even if nobody voted on it". By 2020 every strong language model was a Transformer. The second half of the sentence says vision had not followed.

**Sentence 2** names the two ways vision had used attention so far. Both keep the **convolutional network** in charge. To see why that matters, we need the two words.

> [!DEFINITION] Convolution
> A small table of numbers (the **filter**) is slid over the picture. At each position, the filter's numbers are multiplied with the pixels under it and added up, giving one output number. Sliding it over the whole picture gives a new, filtered picture. A 3×3 filter looks at 9 neighbouring pixels at a time.

> [!DEFINITION] Filter (kernel)
> The small table of learned numbers that a convolution slides over the picture, for example 3×3 = 9 numbers. One filter might light up on vertical edges, another on a patch of orange. "Kernel" means the same thing.

> [!DEFINITION] Convolutional neural network (CNN)
> A neural network built from layers of convolutions. Early layers find edges and colours, middle layers find textures and parts, late layers find whole objects. From 2012 to 2020, essentially every strong image model was a CNN.

> [!DEFINITION] Self-attention
> The core operation of a Transformer. Each token produces a score for every other token (how relevant is that one to me?), turns the scores into weights that add up to 1, and takes a weighted mix of the other tokens' vectors. "Self" because the tokens attend to each other, within one input.

{{FIG:p1_local_vs_global|The two operations side by side. Left: a convolution slides a 3×3 window over the picture; each output sees only its 9 neighbours and the same 9 weights are used at every position. Right: in a Transformer one patch attends to all 49 patches at once, in the very first layer, with weights that are learned rather than fixed by position.}}

The figure shows the real difference between the two operations. A convolution is **local** (each output looks at a small neighbourhood) and **shares weights** (the same filter is used everywhere). Self-attention is **global** (each token can look at every token) and its weights depend on the content, not on the position. Hold on to those two words, local and global; the introduction will come back to them under the name "inductive bias".

> [!PAPER] Dosovitskiy et al. (2020), ViT · Abstract · page 1
> [![The second half of the ViT abstract: we show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train](/img/papers/vit/p1-abstract-2.png)](/img/papers/vit/p1-abstract-2.png)
>
> **Context:** the last two sentences of the abstract: the claim and the result.
>
> **What it says:** "this reliance on CNNs is not necessary". A "pure transformer applied directly to sequences of image patches" works well for classification. The recipe: pre-train "on large amounts of data", then transfer to "mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.)". The result: "excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train".
>
> **Why it matters:** three claims in two sentences. (1) No convolutions are needed. (2) It works when pre-trained on a lot of data and then transferred. (3) It is cheaper to train than the best CNNs. Parts 4 and 5 test each one.

**Sentence 3** is the thesis. "Pure transformer" means no convolution layers inside the network (Part 2 will show that the very first step, the patch embedding, can be written as one convolution, but the authors mean no CNN *body*). "Sequences of image patches" is the title again.

**Sentence 4** is the method and the result, and it introduces the two-step recipe that the whole paper follows.

> [!DEFINITION] Pre-training
> Training a model first on a very large, general dataset, before it ever sees the task you care about. For ViT, pre-training means learning to classify the 14 million pictures of ImageNet-21k or the 303 million pictures of JFT-300M. It is the expensive step, done once.

> [!DEFINITION] Fine-tuning and transfer
> Taking the pre-trained model and training it a little more on a smaller target task, with a new output layer for that task's labels. The knowledge "transfers" from the big dataset to the small one. In this paper, every reported number is a pre-trained model that was then fine-tuned on the benchmark in question. Part 3 covers the details.

> [!DEFINITION] Benchmark
> A fixed, public dataset and scoring rule that everyone uses, so results can be compared fairly. The abstract names three: ImageNet, CIFAR-100 and VTAB.

> [!DEFINITION] ImageNet (ImageNet-1k)
> The standard image classification benchmark since 2012: about 1.3 million training photos sorted into 1,000 classes (120 dog breeds among them), with 50,000 test photos. "ImageNet" in this paper means this 1,000-class set, officially the ILSVRC-2012 dataset.

> [!DEFINITION] CIFAR-100
> A small benchmark of 60,000 tiny 32×32 colour pictures in 100 classes (apple, bicycle, whale, ...), 50,000 for training and 10,000 for testing. Models are often tested on it after pre-training elsewhere, because 50,000 tiny pictures are not enough to learn from scratch.

> [!DEFINITION] VTAB
> The Visual Task Adaptation Benchmark: 19 different classification tasks (natural photos, medical and satellite images, and tasks about counting or distance), each with only 1,000 training examples. It tests how well a pre-trained model adapts to new kinds of pictures with very little data. The score is the average over the 19 tasks.

> [!DEFINITION] State of the art (SOTA)
> The best result anyone has published so far on a benchmark. "Compared to state-of-the-art convolutional networks" means "compared to the best CNN results published".

{{FIG:p1_recipe|The recipe of the paper. Step 1: pre-train ViT once on a very large labelled picture set (ImageNet-21k or JFT-300M), which costs thousands of TPU-core-days. Step 2: for each small benchmark, start from a copy of the pre-trained model, add one new output layer, and fine-tune.}}

If you have read the [BERT breakdown](/papers/bert/), this recipe is familiar: BERT pre-trains on text once and fine-tunes a copy per task. ViT does the same with pictures, with one difference. BERT's pre-training needs no labels (it hides words and guesses them). ViT's pre-training in this paper is **supervised**: every pre-training picture comes with a class label.

> [!DEFINITION] Supervised learning
> Learning from examples that come with the right answer attached. For ViT, every training picture has a label such as "tabby cat", and the model is trained to output that label. The opposite is learning from unlabelled data, which the paper only tries in a small side experiment (Part 5).

The abstract also says what the paper does *not* do. It says "image classification tasks", not detection or segmentation.

{{FIG:p1_scope|What the paper does and does not do. It trains ViT for image classification: one label for the whole picture. Object detection (a box around every object) and segmentation (a class for every pixel) are left to later work, which Part 6 covers.}}

### Let us run it

Before reading the introduction, it helps to see the object the paper is talking about. We loaded the released ViT-B/16 model (`google/vit-base-patch16-224`: pre-trained on ImageNet-21k, fine-tuned on ImageNet) and gave it one picture, the test photo from the `huggingface/cats-image` dataset.

<figure><img src="/img/papers/vit/p1-sample.png" alt="The sample picture: two tabby cats lying on a pink sofa, with a remote control between them" loading="lazy" width="480" /><figcaption>The picture we use throughout this part: two tabby cats on a pink sofa, with two remote controls. It is 640×480 pixels; the model resizes it to 224×224.</figcaption></figure>

The code is in [`vit_part1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part1.py):

```python
proc = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').eval()
x = proc(images=image, return_tensors='pt').pixel_values            # resized to 224 x 224, scaled to [-1, 1]
C, H, W = x.shape[1:]
P = vit.config.patch_size
N = (H // P) * (W // P)
```

```text
ViT-B/16 (google/vit-base-patch16-224)
  image tensor: (1, 3, 224, 224)  = batch of 1, 3 channels, 224 x 224 pixels
  numbers in one image: 3 x 224 x 224 = 150,528
  patch size P = 16; patches per side = 224 // 16 = 14; number of patches N = 14 x 14 = 196
  numbers in one patch: 16 x 16 x 3 = 768
  sequence length with the [class] token: 196 + 1 = 197
  hidden size D = 768, layers = 12, heads = 12
  parameters: 86,567,656
  token matrix after the embedding step: (1, 197, 768)  (1 image, 197 tokens, 768 numbers each)
  output scores: (1, 1000)  (one score per ImageNet class)
```

Every number in that output is one line of arithmetic, and it is worth doing the arithmetic once by hand. The picture is a block of numbers of shape channels × height × width:

$$
\mathbf{x} \in \mathbb{R}^{C \times H \times W}, \qquad C \cdot H \cdot W = 3 \times 224 \times 224 = 150{,}528
$$

where:

- $$C = 3$$ is the number of colour channels (red, green, blue);
- $$H = W = 224$$ are the height and width in pixels, after the model's preprocessing resized the photo;
- $$150{,}528$$ is how many numbers the model receives for one picture.

Cutting it into patches of side $$P = 16$$ gives

$$
N = \frac{H}{P} \cdot \frac{W}{P} = \frac{224}{16} \cdot \frac{224}{16} = 14 \times 14 = 196 \ \text{patches},
\qquad
P^2 C = 16 \times 16 \times 3 = 768 \ \text{numbers per patch}
$$

and the sequence the Transformer reads has one extra token at the front, the `[class]` token (explained in Part 2), so its length is

$$
N + 1 = 197
$$

The 196 × 768 = 150,528 numbers of the patches are exactly the 150,528 numbers of the picture: nothing is lost or added by cutting, only rearranged. The name **ViT-B/16** encodes two of these numbers: "B" for Base (the BERT-base size: 12 layers, 768-wide vectors, 12 attention heads) and "/16" for the patch side. The model has 86.6 million parameters, which the paper rounds to 86M in its Table 1.

> [!DEFINITION] Parameter
> One learned number inside a model: a weight. ViT-B/16 has about 86 million of them; they are set by training and then fixed. "Parameters", "weights" and "model size" are used almost interchangeably.

> [!DEFINITION] Sequence
> An ordered list of tokens. A Transformer always reads a sequence. ViT's sequence has 197 tokens: 196 patches plus the class token.

{{FIG:p1_arithmetic|The arithmetic of ViT-B/16. A 224×224 picture is a 14×14 grid of 16×16 patches: 196 patches, each 16 × 16 × 3 = 768 numbers. With one class token the sequence has 197 tokens, each a vector of 768 numbers.}}

Now the answer. The model outputs 1,000 scores, one per ImageNet class; a softmax turns them into probabilities that add up to 1:

$$
p_k = \frac{e^{s_k}}{\sum_{j=1}^{1000} e^{s_j}}
$$

where $$s_k$$ is the raw score for class $$k$$ and $$p_k$$ is its probability. We printed the five largest, and then did the same with a classic convolutional network, ResNet-50 (`microsoft/resnet-50`), on the same picture:

```python
rproc = AutoImageProcessor.from_pretrained('microsoft/resnet-50')
resnet = ResNetForImageClassification.from_pretrained('microsoft/resnet-50').eval()
xr = rproc(images=image, return_tensors='pt').pixel_values
with torch.no_grad():
    ro = resnet(pixel_values=xr, output_hidden_states=True)
p_res = torch.softmax(ro.logits[0], -1)
```

```text
  top-5 classes:                       (ViT-B/16)
     0.9374  Egyptian cat
     0.0384  tabby
     0.0144  tiger cat
     0.0033  lynx
     0.0007  Siamese cat

ResNet-50 (microsoft/resnet-50), a convolutional network
  image tensor: (1, 3, 224, 224)
  parameters: 25,557,032
  feature map after stage 0: (64, 56, 56)  (channels, height, width)
  feature map after stage 1: (256, 56, 56)  (channels, height, width)
  feature map after stage 2: (512, 28, 28)  (channels, height, width)
  feature map after stage 3: (1024, 14, 14)  (channels, height, width)
  feature map after stage 4: (2048, 7, 7)  (channels, height, width)
  top-5 classes:
     0.9416  tiger cat
     0.0344  tabby
     0.0016  remote control
     0.0014  Egyptian cat
     0.0008  jinrikisha

both models agree on the top class: False  (ViT: Egyptian cat, ResNet-50: tiger cat)
```

{{FIG:p1_top5|The same picture through two models. ViT-B/16 says Egyptian cat with probability 0.937; ResNet-50 says tiger cat with 0.942. Both are sure it is a cat and both put "tabby" second. They disagree about the breed, and ResNet-50 also noticed the remote control.}}

Three things to notice:

- **Both models are right**, in the sense that matters: the picture shows cats, and both put about 94% on a cat class. ImageNet has several house-cat classes (Egyptian cat, tabby, tiger cat, Persian, Siamese), and the picture does not clearly belong to one, so the disagreement is about a fine label, not about the object.
- **ResNet-50 is a different shape of machine.** Its printed "feature maps" shrink from 56×56 to 7×7 as the picture passes through five stages, while the number of channels grows from 64 to 2,048. That is what a CNN does: it summarises larger and larger neighbourhoods. ViT keeps its 197 tokens of 768 numbers at every layer; nothing shrinks.
- **ResNet-50 has 25.6 million parameters, ViT-B/16 86.6 million.** The paper's fair comparison is against much bigger CNNs (Part 4). This run is only a first look, not a contest.

> [!DEFINITION] Accuracy (top-1)
> The share of test pictures for which the model's single most probable class is the correct label. If a model gets 885 of 1,000 pictures right, its top-1 accuracy is 88.5%. All the headline numbers of this paper are top-1 accuracies.

## Transformers took over language {§1}

Now the introduction. Its first paragraph is about language, not pictures.

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 1 · page 1
> [![The first paragraph of the ViT introduction: self-attention-based architectures, in particular Transformers, have become the model of choice in natural language processing. The dominant approach is to pre-train on a large text corpus and then fine-tune on a smaller task-specific dataset. Thanks to Transformers' computational efficiency and scalability, it has become possible to train models of unprecedented size, with over 100B parameters. With the models and datasets growing, there is still no sign of saturating performance](/img/papers/vit/p1-intro-nlp.png)](/img/papers/vit/p1-intro-nlp.png)
>
> **Context:** the opening paragraph of Section 1 (Introduction). It cites four language papers: the Transformer (Vaswani et al., 2017), BERT (Devlin et al., 2019), GPT-3 (Brown et al., 2020) and GShard (Lepikhin et al., 2020).
>
> **What it says:** Transformers are "the model of choice" in NLP. The standard recipe is to "pre-train on a large text corpus and then fine-tune on a smaller task-specific dataset". Because Transformers are efficient and scale well, models "with over 100B parameters" exist, and "there is still no sign of saturating performance".
>
> **Why it matters:** this paragraph is the motivation for everything that follows. In language, bigger Transformers trained on more data kept getting better, with no ceiling in sight. The authors want to know whether pictures behave the same way.

Each of the four citations stands for one idea, and the paper leans on all four. We screenshot the sentences it points to.

> [!PAPER] Vaswani et al. (2017), Attention Is All You Need · Abstract · page 1
> [![The first sentences of the Transformer paper's abstract: the dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely](/img/papers/vit/p1-vaswani-abstract.png)](/img/papers/vit/p1-vaswani-abstract.png)
>
> **Context:** the 2017 paper that introduced the Transformer, for translation between languages.
>
> **What it says:** earlier models used recurrent or convolutional networks. The Transformer is "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely".
>
> **Why it matters:** "dispensing with convolutions entirely" was the Transformer's pitch in 2017, for text. ViT makes the same move for pictures, where convolutions were far more entrenched.

> [!PAPER] Devlin et al. (2019), BERT · Abstract · page 1
> [![The first paragraph of the BERT abstract: BERT is designed to pre-train deep bidirectional representations from unlabeled text, and the pre-trained model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks](/img/papers/vit/p1-bert-abstract.png)](/img/papers/vit/p1-bert-abstract.png)
>
> **Context:** the 2018 paper that made pre-train-then-fine-tune the standard recipe in NLP. The [BERT breakdown](/papers/bert/) on this site reads it in full.
>
> **What it says:** BERT is designed to "pre-train deep bidirectional representations from unlabeled text" and then be "fine-tuned with just one additional output layer".
>
> **Why it matters:** this is "the dominant approach" the ViT paragraph describes. ViT copies the shape of the recipe (pre-train once, fine-tune with one new layer) and even copies BERT's model sizes and its `[class]` token.

> [!PAPER] Brown et al. (2020), GPT-3 · Abstract · page 1
> [![A sentence from the GPT-3 abstract: we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model](/img/papers/vit/p1-gpt3-abstract.png)](/img/papers/vit/p1-gpt3-abstract.png)
>
> **Context:** GPT-3, published five months before ViT.
>
> **What it says:** a language model "with 175 billion parameters, 10x more than any previous non-sparse language model".
>
> **Why it matters:** one of the two "over 100B parameters" examples. The point is not the number itself but that the model got better as it got bigger.

> [!PAPER] Lepikhin et al. (2020), GShard · Abstract · page 1
> [![A sentence from the GShard abstract: GShard enabled us to scale up a multilingual neural machine translation Transformer model with Sparsely-Gated Mixture-of-Experts beyond 600 billion parameters using automatic sharding](/img/papers/vit/p1-gshard-abstract.png)](/img/papers/vit/p1-gshard-abstract.png)
>
> **Context:** GShard, a Google paper on spreading one giant model across many chips.
>
> **What it says:** a translation Transformer scaled "beyond 600 billion parameters".
>
> **Why it matters:** the other "over 100B" example, and a Google-internal one: the authors had the tools to train very large models, and the question was whether vision would reward them.

{{FIG:p1_params|How big the language Transformers in that paragraph are, on a log scale: 213 million parameters for the 2017 Transformer's large version, 340 million for BERT-large, 175 billion for GPT-3 and 600 billion for GShard. The ViT models of this paper (86 million to 632 million) are BERT-sized, not GPT-3-sized.}}

The paragraph's last sentence is the one to remember: "there is still no sign of saturating performance". In language, the curve of accuracy against model size and data size had not flattened. The rest of the introduction asks whether the same curve exists for pictures, and answers yes, with a catch.

> [!DEFINITION] Embedding
> The vector of numbers that stands for one token inside the model. A word embedding is the vector for a word; a patch embedding is the vector for a patch. ViT-B uses embeddings of 768 numbers, the same width as BERT-base.

## In vision, convolutions still ruled {§1}

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 1 · page 1
> [![The second paragraph of the ViT introduction: in computer vision, convolutional architectures remain dominant. Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention, some replacing the convolutions entirely. The latter models, while theoretically efficient, have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns. Therefore, in large-scale image recognition, classic ResNet-like architectures are still state of the art](/img/papers/vit/p1-intro-cnn.png)](/img/papers/vit/p1-intro-cnn.png)
>
> **Context:** the second paragraph of the introduction: the state of computer vision in 2020, with nine citations.
>
> **What it says:** "convolutional architectures remain dominant" (LeCun 1989, AlexNet 2012, ResNet 2016). Several works tried "combining CNN-like architectures with self-attention" (Non-local networks, DETR), some "replacing the convolutions entirely" (stand-alone self-attention, Axial-DeepLab). The latter are "theoretically efficient" but "have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns". So "classic ResNet-like architectures are still state of the art" (Mahajan 2018, Noisy Student, Big Transfer).
>
> **Why it matters:** this paragraph explains *why nobody had done the obvious thing*. Attention over pixels is too expensive, so people invented clever local attention patterns, and those patterns did not run fast on the chips that matter. The paper's way out is to attend over patches, not pixels, so that plain attention becomes cheap enough.

Let us take the three sentences in turn.

**"Convolutional architectures remain dominant."** The three citations are the history of CNNs in three steps. LeCun et al. (1989) trained a convolutional network to read handwritten postcodes, the first practical CNN. Krizhevsky et al. (2012), the AlexNet paper, won the 2012 ImageNet contest by a large margin with a deep CNN trained on GPUs, which started the deep-learning era in vision. He et al. (2016), the ResNet paper, made CNNs of 100+ layers trainable.

> [!PAPER] He et al. (2015), Deep Residual Learning (ResNet) · Abstract · page 1
> [![A sentence from the ResNet abstract: on the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers, 8 times deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task](/img/papers/vit/p1-resnet-abstract.png)](/img/papers/vit/p1-resnet-abstract.png)
>
> **Context:** the ResNet paper, the CNN design that every vision baseline in the ViT paper is built on.
>
> **What it says:** residual networks "with a depth of up to 152 layers" reached "3.57% error on the ImageNet test set" and "won the 1st place on the ILSVRC 2015 classification task".
>
> **Why it matters:** "ResNet-like architectures" is what ViT has to beat. The baseline in this paper, Big Transfer (BiT), is a very large ResNet trained on the same giant datasets as ViT.

> [!DEFINITION] ResNet
> A family of CNNs from 2015 built from "residual blocks", in which each block adds its output to its input, so that very deep stacks (50, 101, 152 layers) can be trained. ResNet-50 is the everyday workhorse; the ViT paper's baselines are ResNets with 152 layers made several times wider.

**"Combining CNN-like architectures with self-attention."** The second sentence lists what people had tried. Non-local networks (Wang et al., 2018) added an attention layer inside a CNN for video. DETR (Carion et al., 2020) put a Transformer on top of a CNN's output to detect objects. Stand-alone self-attention (Ramachandran et al., 2019) and Axial-DeepLab (Wang et al., 2020a) went further and replaced every convolution with a *local* form of attention. The abstract's two categories are these: attention *with* a CNN, and attention *replacing parts of* a CNN while "keeping their overall structure in place".

{{FIG:p1_prior|The three earlier ways of using attention in vision, next to ViT. (1) A CNN does the work and attention is added on top (Non-local networks, DETR). (2) Some CNN layers are swapped for local attention, but the CNN skeleton stays (stand-alone self-attention, Axial-DeepLab). (3) Every layer is local attention with a special pattern, efficient in theory but hard to run fast. ViT is a standard Transformer with global attention over patch tokens: the "fewest possible modifications".}}

**"Specialized attention patterns."** The third sentence is the technical heart of the paragraph. Why did the convolution-free models need special patterns at all? Because plain attention over pixels is impossibly expensive. The paper spells this out in its related work (Part 2): "each pixel attends to every other pixel. With quadratic cost in the number of pixels, this does not scale to realistic input sizes." Let us compute the cost, because the whole design of ViT follows from it.

> [!DEFINITION] Hardware accelerator (GPU, TPU)
> A chip built to do many multiplications at once, which is what neural networks need. GPUs (graphics processing units) came from video games; TPUs (tensor processing units) are Google's own chips. Both are fastest on big, regular blocks of arithmetic, such as multiplying two large matrices. Irregular "patterns" of attention, where each pixel looks at a hand-picked subset of others, do not map well onto them.

### The quadratic cost of attention over pixels

Self-attention computes, for every token, a score against every token. With $$n$$ tokens that is an $$n \times n$$ table of scores per attention head per layer. The [attention series](/writings/attention-1-self-attention/) derives the formula; here it is with the shapes written in:

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

where:

- $$Q$$, $$K$$, $$V$$ are the queries, keys and values, three $$n \times d$$ tables made from the $$n$$ token vectors (for ViT-B, $$d = 64$$ per head);
- $$QK^\top$$ is the $$n \times n$$ table of scores: entry $$(i, j)$$ says how much token $$i$$ should look at token $$j$$;
- the softmax turns every row into weights that add up to 1;
- the result is $$n \times d$$: one new vector per token.

The expensive part is the $$n \times n$$ table. Its size is

$$
\text{attention weights per head per layer} = n^2, \qquad \text{memory in float32} = 4\,n^2 \ \text{bytes}
$$

> [!DEFINITION] Quadratic cost
> A cost that grows with the square of the input size. Doubling the number of tokens makes a quadratic cost four times larger; multiplying the tokens by 255 makes it about 65,000 times larger. Attention has a quadratic cost in the number of tokens.

We computed this table for several choices of what a token is ([`vit_part1_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part1_math.py), part (a)):

```python
for name, n in [('32 x 32 CIFAR pixels', 32 * 32), ('224 x 224 ImageNet pixels', 224 * 224),
                ('ViT-B/32 patches + [class] (7x7+1)', 7 * 7 + 1), ('ViT-B/16 patches + [class] (14x14+1)', 14 * 14 + 1),
                ('ViT-B/16 at 384 px (24x24+1)', 24 * 24 + 1)]:
    nn_ = n * n
    bytes_ = nn_ * 4
```

```text
    tokens are                                n            n x n  float32 memory
    32 x 32 CIFAR pixels                  1,024        1,048,576         4.19 MB
    224 x 224 ImageNet pixels            50,176    2,517,630,976        10.07 GB
    ViT-B/32 patches + [class] (7x7+1)       50            2,500         10.0 kB
    ViT-B/16 patches + [class] (14x14+1)      197           38,809        155.2 kB
    ViT-B/16 at 384 px (24x24+1)            577          332,929         1.33 MB
    pixels / patch tokens = 50,176 / 197 = 254.7 x more tokens, so 64,872 x more attention weights
    ViT-B/16 has 12 heads x 12 layers = 144 attention tables of 197 x 197 = 5,588,496 weights for one image
```

{{FIG:p1_quadratic|Attention weights per head per layer, on a log scale. Over the 50,176 pixels of a 224×224 picture, one attention table has 2.5 billion entries (10 GB in float32). Over ViT-B/16's 197 patch tokens it has 38,809 entries (155 kB). Even the 1,024 pixels of a tiny CIFAR picture need a million entries.}}

Read the second row. If every pixel of a 224×224 picture attended to every other pixel, one attention table of one head of one layer would hold $$50{,}176^2 \approx 2.5$$ billion numbers, about 10 GB. ViT-B has 144 such tables (12 heads × 12 layers), and training needs several copies of each. It cannot be done. Even for the tiny 32×32 pictures of CIFAR, one table has a million entries. That is why the earlier convolution-free models restricted each pixel to a local neighbourhood, and why those restrictions needed "specialized attention patterns" that ran poorly on accelerators.

Now read the fourth row. With 16×16 patches as tokens, $$n = 197$$ and one table has 38,809 entries. The ratio between the two is

$$
\frac{50{,}176^2}{197^2} = \left(\frac{50{,}176}{197}\right)^2 \approx 254.7^2 \approx 64{,}872
$$

Patches make attention about 65,000 times cheaper than pixels. That single fact is the engineering reason the title says "16×16 words" rather than "50,176 words". And 197 tokens is a very ordinary sequence length for a Transformer (BERT was trained with 512), so the standard, dense, accelerator-friendly attention can be used unchanged. The paper's related work (Part 2) credits Cordonnier et al. (2020) with the patch idea, using 2×2 patches on small pictures; the ViT authors use bigger patches, bigger pictures and far more data.

**"Classic ResNet-like architectures are still state of the art."** The paragraph closes with the three results that defined the top of the ImageNet leaderboard in 2020: Mahajan et al. (2018) pre-trained ResNeXt CNNs on 3.5 billion Instagram pictures; Xie et al. (2020), Noisy Student, used 300 million unlabelled pictures with a self-training loop; and Kolesnikov et al. (2020), Big Transfer (BiT), pre-trained very large ResNets on the same ImageNet-21k and JFT-300M datasets that ViT will use. All three are CNNs made better with *more data*, which is the same lever ViT pulls.

## The fewest possible modifications {§1}

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 1 · page 1
> [![The third paragraph of the ViT introduction: inspired by the Transformer scaling successes in NLP, we experiment with applying a standard Transformer directly to images, with the fewest possible modifications. To do so, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer. Image patches are treated the same way as tokens (words) in an NLP application. We train the model on image classification in supervised fashion](/img/papers/vit/p1-intro-ours.png)](/img/papers/vit/p1-intro-ours.png)
>
> **Context:** the third paragraph: what the authors actually do, in four sentences.
>
> **What it says:** apply "a standard Transformer directly to images, with the fewest possible modifications". Split the image into patches, feed "the sequence of linear embeddings of these patches" to the Transformer. "Image patches are treated the same way as tokens (words) in an NLP application." Train "on image classification in supervised fashion".
>
> **Why it matters:** "fewest possible modifications" is a deliberate scientific choice. By changing nothing but the input, the authors can attribute everything they find to the data and the scale, not to clever new architecture. It also means every tool built for language Transformers (training code, scaling tricks) works out of the box.

The four sentences are the whole method. We illustrate each.

**"Split an image into patches."** The picture is cut into a grid of equal squares and the squares are read off in order, row by row, like text.

{{FIG:p1_grid_flatten|An image cut into a 3×3 grid of patches and flattened, row by row, into a sequence of nine patch tokens. Patch 1 is the top-left corner and patch 9 the bottom-right. The order is fixed; in Part 2 a position embedding tells the model where each patch came from.}}

**"The sequence of linear embeddings of these patches."** Each patch is 16 × 16 × 3 = 768 raw pixel numbers. A "linear embedding" multiplies those 768 numbers by a learned matrix to give the patch's token vector. In the paper's notation (Equation 1, which Part 2 covers in full) a flattened patch $$\mathbf{x}_p^i$$ becomes

$$
\mathbf{x}_p^i \mathbf{E}, \qquad \mathbf{x}_p^i \in \mathbb{R}^{1 \times 768}, \quad \mathbf{E} \in \mathbb{R}^{768 \times D}
$$

where $$D$$ is the Transformer's width ($$D = 768$$ for ViT-B, so $$\mathbf{E}$$ happens to be square, 768 × 768). "Linear" means no non-linearity: it is one matrix multiplication, the simplest possible way to turn a patch into a vector. The output has the same shape as a word embedding in BERT, which is the point.

{{FIG:p1_patch_vector|One patch becomes one token. The 16×16×3 block of pixels is flattened into a row of 768 numbers, multiplied by a learned 768×768 matrix E (the "linear embedding"), and comes out as a vector of 768 numbers, the same shape as a word vector in BERT.}}

Our code found this layer in the released model. In the implementation it is stored as a convolution with a 16×16 filter and a stride of 16, which is exactly the same computation as "cut into 16×16 patches and multiply each by $$\mathbf{E}$$", because a stride-16 filter of size 16 touches each pixel exactly once:

```text
    patch embedding layer: Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    patch vectors: (196, 768) = 196 patches x 768 numbers, arranged on a 14 x 14 grid
```

**"Treated the same way as tokens (words)."** After the embedding, the Transformer sees 196 vectors of 768 numbers. It would see the same thing for a 196-word sentence in BERT. Nothing in the attention layers, the feed-forward layers or the normalisation knows about pixels, rows or columns.

**"In supervised fashion."** Pre-training is ordinary classification with labels, on a very large labelled set. BERT's label-free pre-training was its big trick; ViT does not need a trick, because Google had hundreds of millions of labelled pictures. (Part 5 shows a first attempt at label-free pre-training for ViT, and Part 6 shows how later papers made it work.)

## Mid-sized data, and the inductive-bias explanation {§1}

The fourth paragraph starts at the bottom of page 1 and ends at the top of page 2. It reports the first, disappointing result, and explains it.

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 1 · pages 1 to 2
> [![The fourth paragraph of the ViT introduction, first part: when trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNets of comparable size. This seemingly discouraging outcome may be expected: Transformers lack some of the inductive biases](/img/papers/vit/p1-intro-midsize.png)](/img/papers/vit/p1-intro-midsize.png)
> [![The paragraph continues on page 2: inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data](/img/papers/vit/p1-intro-midsize2.png)](/img/papers/vit/p1-intro-midsize2.png)
>
> **Context:** the fourth paragraph of the introduction. The first honest result of the paper.
>
> **What it says:** trained on "mid-sized datasets such as ImageNet without strong regularization", ViT models reach "modest accuracies of a few percentage points below ResNets of comparable size". The authors call this "seemingly discouraging" but expected: "Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data."
>
> **Why it matters:** this is the conceptual centre of the paper. A CNN comes with built-in assumptions about pictures; a Transformer does not. With little data, assumptions help. The next paragraph says what happens with a lot of data.

Four technical words appear here, and the rest of the series uses all of them, so we take them slowly.

> [!DEFINITION] Regularization
> Any technique that stops a model from memorising its training set instead of learning general rules: adding noise, randomly dropping parts of the network during training (dropout), shrinking the weights (weight decay), or distorting the training pictures. "Without strong regularization" means the authors did not lean hard on such tricks in this first comparison. Part 4 shows that heavy regularization helps ViT on small data.

> [!DEFINITION] Inductive bias
> The built-in assumptions of a model design, the things it "believes" about the data before seeing any. A CNN assumes that nearby pixels belong together and that an object looks the same wherever it appears. These assumptions let it learn from fewer examples, but they also limit what it can learn. A Transformer has far fewer such assumptions: it must learn from data even that neighbouring patches are related.

> [!DEFINITION] Locality
> The assumption that what matters for a pixel is mostly its neighbours. A convolution with a 3×3 filter builds this in: each output depends on 9 nearby pixels and nothing else. Far-away parts of the picture only meet after many layers. Attention has no such rule: in layer 1, any patch can look at any patch.

The ResNet-50 run above showed locality in action: its feature maps shrink from 56×56 to 7×7 stage by stage, which is how a CNN gradually lets far-away pixels meet. ViT's 197 tokens all meet in the first layer.

> [!DEFINITION] Translation equivariance
> "Translation" here means *shifting*: moving the picture (or an object in it) left, right, up or down. A function is **equivariant** to shifts if shifting the input shifts the output by the same amount, and changes nothing else. A convolution is equivariant: if the cat moves 2 pixels right, the "cat features" move 2 pixels right. This is different from **invariance**, which means the output does not change at all when the input shifts. A classifier should be *invariant* (a cat is a cat wherever it sits); the layers inside a CNN are *equivariant*, and a final pooling step turns equivariant features into an invariant answer.

Let us write these two properties down, because they are often confused. A picture is a function $$x[i, j]$$ of its row $$i$$ and column $$j$$. A two-dimensional convolution with a filter $$w$$ (here 3×3) computes

$$
(x * w)[i, j] = \sum_{u=0}^{2} \sum_{v=0}^{2} w[u, v] \, x[i + u, \, j + v]
$$

where:

- $$x[i + u, j + v]$$ are the 9 pixels under the filter when its top-left corner sits at $$(i, j)$$;
- $$w[u, v]$$ are the 9 filter weights, the same at every position (that is **weight sharing**);
- the sum is one output number; sliding over all $$(i, j)$$ gives the filtered picture.

Define the **shift** operator $$T_{t}$$ that moves a picture $$t$$ pixels to the right:

$$
(T_{t}\, x)[i, j] = x[i, \, j - t]
$$

Then equivariance and invariance of a function $$f$$ are

$$
\text{equivariance:}\quad f(T_{t}\, x) = T_{t}\, f(x)
\qquad\qquad
\text{invariance:}\quad f(T_{t}\, x) = f(x)
$$

In words: an equivariant $$f$$ lets the shift pass through (shift then filter equals filter then shift); an invariant $$f$$ swallows the shift. A convolution is equivariant because the same weights $$w$$ are used at every position: the sum at output position $$(i, j + t)$$ of the shifted picture sees exactly the pixels that the sum at $$(i, j)$$ saw in the original.

**Equivariance with real numbers.** We built a 12×12 toy picture with a bright 5×5 square in it, applied a 3×3 vertical-edge filter, then shifted the picture 2 pixels right and applied the same filter ([`vit_part1_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part1_math.py), part (b)):

```python
img = torch.zeros(1, 1, 12, 12)
img[0, 0, 3:8, 2:7] = 1.0                                           # a bright 5 x 5 square at rows 3..7, columns 2..6
k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)   # a 3 x 3 vertical-edge filter (Sobel)
shift = 2
shifted = torch.zeros_like(img)
shifted[..., shift:] = img[..., :-shift]                             # the same square, 2 pixels to the right
out = Fn.conv2d(img, k)                                              # 10 x 10 output (no padding)
out_s = Fn.conv2d(shifted, k)
diff = (out_s[..., shift:] - out[..., :-shift]).abs().max().item()   # move the second output back by 2 and compare
```

```text
    toy image (12, 12), bright square at rows 3-7, columns 2-6; filter (3, 3) (vertical edges)
    filter weights: [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    output (10, 10); strongest response in the original at column 0, in the shifted at column 2
    max |shifted output moved back by 2 - original output| = 0.0
    row 5 of the output, original:    4    4    0    0    0   -4   -4    0    0    0
    row 5 of the output, shifted:     0    0    4    4    0    0    0   -4   -4    0
    the same filter weights (9 numbers) are used at every one of the 100 output positions (weight sharing)
```

Read the two printed rows. The filter fires +4 where the square's left edge is (dark to bright) and -4 where its right edge is (bright to dark). In the shifted picture the very same values appear, two columns to the right. Moving the second output back by two columns and subtracting the first gives a maximum difference of exactly 0.0. That is equivariance: *the feature moved with the object, nothing else changed.* The filter never had to learn anything about "the square at column 4" versus "the square at column 6"; one set of 9 weights handles every position.

{{FIG:p1_equivariance|Translation equivariance made concrete. Left pair: the toy picture and the edge filter's output, which marks the square's left edge (positive) and right edge (negative). Right pair: the same picture moved 2 pixels to the right, and the output, which has moved by exactly 2 pixels and is otherwise identical. The maximum difference after shifting back is 0.0.}}

This is why a CNN can learn from fewer examples. A cat in the top-left corner teaches the same filters as a cat in the bottom-right. The network does not need to see cats everywhere; the design guarantees that what it learns in one place applies in every place.

### Patch tokens are not shift-equivariant

Now the other side. ViT's first step cuts the picture along a fixed 16-pixel grid. Shift the picture by less than 16 pixels and every patch now contains a different mix of pixels, so every patch vector changes. Shift by exactly 16 and each patch's content moves, whole, into the neighbouring slot. We measured this on the real cat picture and the real ViT-B/16 patch-embedding layer, using cosine similarity to compare vectors:

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert}
$$

where $$\mathbf{a}$$ and $$\mathbf{b}$$ are two 768-number patch vectors, the dot is their dot product and $$\lVert \cdot \rVert$$ is the length. The cosine is 1 for identical directions, 0 for unrelated ones, and negative for opposite ones. For each shift $$k$$ we compared patch $$(i, j)$$ of the original with patch $$(i, j + \lfloor k/16 \rfloor)$$ of the shifted picture, that is, we move the token grid back by the whole number of patches the picture moved (0 for shifts under 16, 1 for a shift of 16):

```python
embed = vit.vit.embeddings.patch_embeddings                          # one 16 x 16 convolution with stride 16 = the linear patch projection
with torch.no_grad():
    ek = embed(shift_right(x, k))[0].view(14, 14, -1)
back = k // P                                                    # whole patches the picture moved
a, b = grid[:, :14 - back], ek[:, back:]                        # compare patch (i, j) of the original with patch (i, j + back) of the shifted
cos = Fn.cosine_similarity(a.reshape(-1, 768), b.reshape(-1, 768), dim=-1)
```

```text
     shift k  grid moved back  mean cosine  min cosine   ViT-B/16 top-1             ResNet-50 top-1
           0                0        1.000       1.000   Egyptian cat 0.937         tiger cat 0.942
           1                0        0.935       0.643   Egyptian cat 0.888         tiger cat 0.939
           4                0        0.649      -0.219   Egyptian cat 0.861         tiger cat 0.847
           8                0        0.507      -0.353   Egyptian cat 0.939         tiger cat 0.826
          12                0        0.457      -0.446   Egyptian cat 0.847         tiger cat 0.607
          16                1        1.000       1.000   Egyptian cat 0.908         tiger cat 0.737
    for k = 16 compared at the SAME grid position (no moving back): mean cosine = 0.428  (each slot now holds its left neighbour)
```

{{FIG:p1_shift|What a shift does to ViT's patch tokens, and to both models' answers. Green: the mean cosine similarity between the original and shifted patch vectors falls from 1.000 to 0.457 as the shift grows to 12 pixels, then jumps back to 1.000 at 16 pixels, when whole patches move one slot. Blue and orange: the probability of each model's top class moves around but the class never changes.}}

Two lessons sit in this table.

- **The patch tokens are not equivariant.** A 1-pixel shift, invisible to a person, already changes the average patch vector's cosine to 0.935 and the worst one to 0.643. By 12 pixels the mean is 0.457 and some vectors point in the opposite direction. Only a shift of exactly one patch (16 pixels) restores a cosine of 1.000, and then only because we moved the token grid back by one slot. Compared at the same slot, the 16-pixel shift gives 0.428. A convolution has none of this: its output moved by exactly 2 pixels and changed by exactly 0.0.
- **ViT still gets the answer right.** At every shift its top class stays "Egyptian cat" with probability between 0.847 and 0.939. The model was never told that shifts do not matter; it learned it, from millions of pictures in which cats sat in different places. That is the paper's argument in one table: what the CNN has by design, the Transformer can learn from data, *if there is enough data*. ResNet-50 also keeps "tiger cat" throughout, as expected for a CNN.

An honesty note on the ResNet column: its confidence drops to 0.607 at a 12-pixel shift and 0.737 at 16. A real ResNet is not perfectly equivariant either, because its strided layers and pooling sample the picture on coarse grids, and our shift fills the strip that enters on the left by repeating the edge column, which adds a small artificial band. The toy filter above is exactly equivariant; a whole ResNet is only approximately so. The qualitative point stands: both networks keep their answer under shifts that scramble every ViT patch vector.

## Large scale training trumps inductive bias {§1}

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 1 · page 2
> [![The fifth paragraph of the ViT introduction: however, the picture changes if the models are trained on larger datasets (14M-300M images). We find that large scale training trumps inductive bias. Our Vision Transformer (ViT) attains excellent results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints. When pre-trained on the public ImageNet-21k dataset or the in-house JFT-300M dataset, ViT approaches or beats state of the art on multiple image recognition benchmarks. In particular, the best model reaches the accuracy of 88.55% on ImageNet, 90.72% on ImageNet-ReaL, 94.55% on CIFAR-100, and 77.63% on the VTAB suite of 19 tasks](/img/papers/vit/p1-intro-scale.png)](/img/papers/vit/p1-intro-scale.png)
>
> **Context:** the last paragraph of the introduction, at the top of page 2. The turn of the story, and the headline numbers.
>
> **What it says:** "the picture changes if the models are trained on larger datasets (14M-300M images). We find that large scale training trumps inductive bias." Pre-trained on "the public ImageNet-21k dataset or the in-house JFT-300M dataset", ViT "approaches or beats state of the art on multiple image recognition benchmarks". The best model reaches "88.55% on ImageNet, 90.72% on ImageNet-ReaL, 94.55% on CIFAR-100, and 77.63% on the VTAB suite of 19 tasks".
>
> **Why it matters:** "large scale training trumps inductive bias" is the sentence the paper is remembered for. With enough data, a model with fewer built-in assumptions wins, because it is free to learn better assumptions than the ones a human would build in.

The two datasets named here are the paper's fuel.

> [!DEFINITION] ImageNet-21k
> The full ImageNet collection, of which the usual 1,000-class benchmark is a subset: about 14 million pictures labelled with about 21,000 classes (the paper writes "21k classes and 14M images"). It is public, so anyone with enough compute can repeat the ImageNet-21k experiments, and the released ViT weights were pre-trained on it.

> [!DEFINITION] JFT-300M
> Google's internal dataset of about 303 million pictures with about 18,000 noisy labels, described in Sun et al. (2017). It is not public. The paper's very best numbers come from models pre-trained on it; its ImageNet-21k models are the ones the rest of us can download.

"14M-300M images" refers to these two: 14 million for ImageNet-21k, 303 million for JFT-300M. Against them, the 1.3 million pictures of ImageNet are "mid-sized", which is a startling thing to call the dataset that defined the field for a decade.

{{FIG:p1_data_sketch|A sketch of the paper's claim. With little pre-training data, a convolutional network's built-in assumptions help, and it beats ViT. With a lot of data, ViT learns those assumptions, and more, from the pictures themselves, and overtakes. This is a drawing of the idea, not measured data; the measured numbers follow.}}

The paper proves the sketch in Section 4.3, which Part 4 reads in full. Two sentences from it are worth seeing now, because they are the evidence behind "the picture changes".

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 4.3 · page 6
> [![A passage from Section 4.3: first, we pre-train ViT models on datasets of increasing size: ImageNet, ImageNet-21k, and JFT-300M. When pre-trained on the smallest dataset, ImageNet, ViT-Large models underperform compared to ViT-Base models](/img/papers/vit/p1-sec43-small.png)](/img/papers/vit/p1-sec43-small.png)
> [![A sentence from Section 4.3 on page 7: the BiT CNNs outperform ViT on ImageNet, but with the larger datasets, ViT overtakes](/img/papers/vit/p1-sec43-overtake.png)](/img/papers/vit/p1-sec43-overtake.png)
>
> **Context:** Section 4.3, "Pre-training Data Requirements", pages 6 to 7. The experiment that tests the introduction's claim.
>
> **What it says:** ViT models were pre-trained "on datasets of increasing size: ImageNet, ImageNet-21k, and JFT-300M". On the smallest, "ViT-Large models underperform compared to ViT-Base models". And: "The BiT CNNs outperform ViT on ImageNet, but with the larger datasets, ViT overtakes."
>
> **Why it matters:** two crossovers, both in the direction the inductive-bias story predicts. A bigger Transformer is worse than a smaller one on small data (more freedom, more ways to go wrong). A CNN beats the Transformer on small data and loses on large data.

The exact numbers are in the paper's Table 5 (appendix), which Part 4 redraws in full. Here is the table, and the ImageNet rows for the two 16-pixel-patch models as a chart.

> [!PAPER] Dosovitskiy et al. (2020), ViT · Table 5 · page 15
> [![Table 5 of the ViT paper: top-1 accuracy of ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32 and ViT-H/14 on CIFAR-10, CIFAR-100, ImageNet, ImageNet ReaL, Oxford Flowers-102 and Oxford-IIIT Pets, after pre-training on ImageNet, ImageNet-21k or JFT-300M. The ImageNet rows are highlighted: 77.91, 73.38, 76.53, 71.16 after ImageNet pre-training; 83.97, 81.28, 85.15, 80.99, 85.13 after ImageNet-21k; 84.15, 80.73, 87.12, 84.37, 88.04 after JFT-300M](/img/papers/vit/p1-table5.png)](/img/papers/vit/p1-table5.png)
>
> **Context:** Table 5 in Appendix D, the numbers behind Figure 3 of the main text. Models fine-tuned at 384-pixel resolution.
>
> **What it says:** every model gets better as the pre-training set grows. Read the highlighted ImageNet rows for ViT-B/16 and ViT-L/16: 77.91 and 76.53 after ImageNet, 83.97 and 85.15 after ImageNet-21k, 84.15 and 87.12 after JFT-300M.
>
> **Why it matters:** the larger model (L/16) is *worse* than the smaller (B/16) with 1.3 million pictures, and *better* with 14 million or 303 million. "Large scale training trumps inductive bias" in five numbers.

{{FIG:p1_table5|The ImageNet rows of Table 5 for ViT-B/16 and ViT-L/16, against the size of the pre-training set on a log scale. With 1.3 million pictures the smaller model wins (77.91 against 76.53). With 14 million the larger model is ahead (85.15 against 83.97), and with 303 million it is far ahead (87.12 against 84.15).}}

Now the four headline numbers. They all belong to the paper's biggest model, ViT-H/14 (632 million parameters, 14-pixel patches), pre-trained on JFT-300M.

{{FIG:p1_headline|The four headline results of the introduction: 88.55% top-1 on ImageNet, 90.72% on ImageNet-ReaL (ImageNet with cleaner labels), 94.55% on CIFAR-100 and 77.63% averaged over the 19 tasks of VTAB. All from ViT-H/14 pre-trained on JFT-300M.}}

| Benchmark | What it tests | ViT-H/14 (JFT) | Note |
|---|---|---|---|
| ImageNet | 1,000-class photo classification, 50,000 test pictures | 88.55% | the standard number everyone compares |
| ImageNet-ReaL | the same pictures with corrected labels (Beyer et al., 2020) | 90.72% | higher because fewer "wrong" labels in the test set |
| CIFAR-100 | 100 classes of tiny 32×32 pictures | 94.55% | shows transfer to a very different kind of picture |
| VTAB (19 tasks) | 19 varied tasks with 1,000 training examples each | 77.63% | shows transfer with very little data |

These numbers are compared with the best CNNs in Table 2 of the paper (Part 4). The short version: ViT-H/14 beat the Big Transfer ResNet and Noisy Student on every one of these benchmarks, and did so with less pre-training compute. Part 4 goes through every row and every error bar.

## Who's who: the papers the introduction is talking to

The introduction cites sixteen earlier works in five paragraphs. Here they are in one place, so the rest of the series can refer back to them. Each is listed in full in the references at the end of this part.

| Short name | Paper | Year | What it is, in one line | Role in the ViT paper |
|---|---|---|---|---|
| Transformer | Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) | 2017 | the attention-only network for translation | ViT is its encoder, unchanged |
| BERT | Devlin et al., [*BERT*](https://arxiv.org/abs/1810.04805) | 2018 | pre-train a Transformer encoder, fine-tune per task | the recipe, the model sizes and the class token |
| GPT-3 | Brown et al., [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165) | 2020 | a 175-billion-parameter language model | "over 100B parameters" |
| GShard | Lepikhin et al., [*GShard*](https://arxiv.org/abs/2006.16668) | 2020 | a 600-billion-parameter translation model | "over 100B parameters" |
| LeNet | LeCun et al., [*Backpropagation Applied to Handwritten Zip Code Recognition*](https://doi.org/10.1162/neco.1989.1.4.541) | 1989 | the first practical CNN | "convolutional architectures remain dominant" |
| AlexNet | Krizhevsky et al., [*ImageNet Classification with Deep CNNs*](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | 2012 | the deep CNN that won ImageNet 2012 | start of the deep-learning era in vision |
| ResNet | He et al., [*Deep Residual Learning*](https://arxiv.org/abs/1512.03385) | 2015 | CNNs of 100+ layers | the "ResNet-like architectures" ViT must beat |
| Non-local | Wang et al., [*Non-local Neural Networks*](https://arxiv.org/abs/1711.07971) | 2018 | attention blocks inside a CNN, for video | CNN + attention |
| DETR | Carion et al., [*End-to-End Object Detection with Transformers*](https://arxiv.org/abs/2005.12872) | 2020 | a Transformer on top of a CNN, for detection | CNN + attention |
| Stand-alone | Ramachandran et al., [*Stand-Alone Self-Attention in Vision Models*](https://arxiv.org/abs/1906.05909) | 2019 | local attention replacing every convolution | "replacing the convolutions entirely" |
| Axial-DeepLab | Wang et al. (2020a), [*Axial-DeepLab*](https://arxiv.org/abs/2003.07853) | 2020 | attention along rows, then along columns | "replacing the convolutions entirely" |
| Instagram pre-training | Mahajan et al., [*Exploring the Limits of Weakly Supervised Pretraining*](https://arxiv.org/abs/1805.00932) | 2018 | CNNs pre-trained on 3.5 billion Instagram pictures | ResNets are still state of the art |
| Noisy Student | Xie et al., [*Self-training with Noisy Student*](https://arxiv.org/abs/1911.04252) | 2020 | CNN self-training with 300M unlabelled pictures | ResNets are still state of the art |
| BiT | Kolesnikov et al., [*Big Transfer*](https://arxiv.org/abs/1912.11370) | 2020 | very large ResNets pre-trained on ImageNet-21k and JFT | the main baseline, by the same team |
| ImageNet | Deng et al., [*ImageNet: A Large-Scale Hierarchical Image Database*](https://doi.org/10.1109/CVPR.2009.5206848) | 2009 | the 14-million-picture collection behind ImageNet-1k and 21k | the public pre-training set |
| JFT | Sun et al., [*Revisiting Unreasonable Effectiveness of Data*](https://arxiv.org/abs/1707.02968) | 2017 | Google's 300-million-picture internal dataset | the private pre-training set |

> [!TAKEAWAYS] Key takeaways
> - **The title is the method.** A picture is cut into 16×16-pixel patches, each patch is treated as a word token, and a standard Transformer encoder reads the sequence. A 224×224 picture gives 14 × 14 = 196 patches of 16 × 16 × 3 = 768 numbers; with the class token the sequence has 197 tokens.
> - **"At scale" is the finding.** With only ImageNet's 1.3 million pictures ViT is a few points below ResNets; with 14 million (ImageNet-21k) or 303 million (JFT-300M) it matches or beats the best CNNs. In the paper's words, "large scale training trumps inductive bias".
> - **Why pixels would not work:** attention costs $$n^2$$ per head per layer. Over 50,176 pixels that is 2.5 billion weights (10 GB); over 197 patch tokens it is 38,809 (155 kB), about 65,000 times less. Patches, not clever attention patterns, make a plain Transformer affordable.
> - **Inductive bias** means the assumptions a design builds in. CNNs build in **locality** (look at neighbours) and **translation equivariance** (shift the input, the output shifts the same way). We checked equivariance on a real convolution: output shifted by exactly 2 pixels, maximum difference 0.0.
> - **ViT has neither built in.** Shifting the cat picture by 1 to 12 pixels changed every patch vector (mean cosine 0.935 down to 0.457); only a 16-pixel shift, one whole patch, restored them. Yet ViT kept answering "Egyptian cat" (probability 0.847 to 0.939): it had learned shift-robustness from data.
> - **Two steps:** pre-train once on a huge labelled set (supervised), then fine-tune a copy with one new output layer per benchmark. The same recipe as BERT, but with labels.
> - **The crossover in numbers (Table 5):** ViT-B/16 vs ViT-L/16 on ImageNet after pre-training on 1.3M / 14M / 303M pictures: 77.91 / 83.97 / 84.15 against 76.53 / 85.15 / 87.12. The bigger model loses on small data and wins on large data.
> - **Headline results** of ViT-H/14 pre-trained on JFT-300M: **88.55%** ImageNet, **90.72%** ImageNet-ReaL, **94.55%** CIFAR-100, **77.63%** VTAB.
> - On our sample picture, ViT-B/16 (86.6M parameters) and ResNet-50 (25.6M) both said "cat" with about 94% confidence and disagreed only on the breed (Egyptian cat 0.937 vs tiger cat 0.942).
> - The paper does classification only. Detection, segmentation and label-free pre-training came later (Part 6).

**Next, in [Part 2](part-2-the-model.md):** the related work the paper builds on (local attention, Sparse Transformers, iGPT, and the 2×2-patch model of Cordonnier et al.), and then the model itself: Figure 1 block by block, Equations 1 to 4 with the real numbers of ViT-B/16 (the patch embedding, the class token, the position embeddings, one encoder layer), the Appendix A formulas, and where the 86 million parameters live.

<details>
<summary>Run it yourself</summary>

The two scripts behind this part are [`code/papers/vit/vit_part1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part1.py) (the cat picture through ViT-B/16 and ResNet-50) and [`code/papers/vit/vit_part1_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part1_math.py) (the quadratic cost table, the equivariance check on a toy convolution, and the patch-shift experiment). Both run on a laptop CPU in under a minute each and download `google/vit-base-patch16-224` (about 350 MB) and `microsoft/resnet-50` (about 100 MB) the first time.

```bash
pip install torch transformers datasets pillow
python vit_part1.py         # shapes, patch arithmetic, top-5 of both models; writes results/part1.json
python vit_part1_math.py    # quadratic cost, equivariance, patch shifts; writes results/part1_math.json
```

<figure><img src="/img/papers/vit/part1-run.png" alt="Terminal output of vit_part1.py: the picture size, the ViT-B/16 tensor shapes and patch arithmetic, its top-5 classes led by Egyptian cat 0.9374, the ResNet-50 feature-map shapes and its top-5 classes led by tiger cat 0.9416" loading="lazy" /><figcaption>The real output of vit_part1.py.</figcaption></figure>

<figure><img src="/img/papers/vit/part1_math-run.png" alt="Terminal output of vit_part1_math.py: the table of attention weights for pixels versus patches, the toy convolution equivariance check with a maximum difference of 0.0, and the patch-shift table with mean cosine similarities and both models' top-1 classes for shifts of 0, 1, 4, 8, 12 and 16 pixels" loading="lazy" /><figcaption>The real output of vit_part1_math.py.</figcaption></figure>

</details>

## References

**The ViT paper**

1. A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby. [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929) (ViT). ICLR 2021 ([OpenReview](https://openreview.net/forum?id=YicbFdNTTy)). arXiv:2010.11929, version 2 (June 2021), which is the version shown in the screenshots.
2. Google Research. [Vision Transformer code and pre-trained models](https://github.com/google-research/vision_transformer). GitHub, 2020.

**Papers the ViT paper cites in this part**

3. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin. [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (the Transformer). NeurIPS 2017.
4. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805) (BERT). NAACL 2019.
5. T. B. Brown et al. [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165) (GPT-3). NeurIPS 2020.
6. D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, Z. Chen. [*GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*](https://arxiv.org/abs/2006.16668) (GShard). ICLR 2021.
7. Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, L. D. Jackel. [*Backpropagation Applied to Handwritten Zip Code Recognition*](https://doi.org/10.1162/neco.1989.1.4.541). Neural Computation 1(4), 1989.
8. A. Krizhevsky, I. Sutskever, G. E. Hinton. [*ImageNet Classification with Deep Convolutional Neural Networks*](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (AlexNet). NeurIPS 2012.
9. K. He, X. Zhang, S. Ren, J. Sun. [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) (ResNet). CVPR 2016.
10. X. Wang, R. Girshick, A. Gupta, K. He. [*Non-local Neural Networks*](https://arxiv.org/abs/1711.07971). CVPR 2018.
11. N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, S. Zagoruyko. [*End-to-End Object Detection with Transformers*](https://arxiv.org/abs/2005.12872) (DETR). ECCV 2020.
12. P. Ramachandran, N. Parmar, A. Vaswani, I. Bello, A. Levskaya, J. Shlens. [*Stand-Alone Self-Attention in Vision Models*](https://arxiv.org/abs/1906.05909). NeurIPS 2019.
13. H. Wang, Y. Zhu, B. Green, H. Adam, A. Yuille, L.-C. Chen. [*Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation*](https://arxiv.org/abs/2003.07853) ("Wang et al., 2020a" in the paper). ECCV 2020.
14. D. Mahajan, R. Girshick, V. Ramanathan, K. He, M. Paluri, Y. Li, A. Bharambe, L. van der Maaten. [*Exploring the Limits of Weakly Supervised Pretraining*](https://arxiv.org/abs/1805.00932). ECCV 2018.
15. Q. Xie, M.-T. Luong, E. Hovy, Q. V. Le. [*Self-training with Noisy Student improves ImageNet classification*](https://arxiv.org/abs/1911.04252) (Noisy Student). CVPR 2020.
16. A. Kolesnikov, L. Beyer, X. Zhai, J. Puigcerver, J. Yung, S. Gelly, N. Houlsby. [*Big Transfer (BiT): General Visual Representation Learning*](https://arxiv.org/abs/1912.11370) (BiT). ECCV 2020.
17. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei. [*ImageNet: A Large-Scale Hierarchical Image Database*](https://doi.org/10.1109/CVPR.2009.5206848). CVPR 2009.
18. C. Sun, A. Shrivastava, S. Singh, A. Gupta. [*Revisiting Unreasonable Effectiveness of Data in Deep Learning Era*](https://arxiv.org/abs/1707.02968) (JFT-300M). ICCV 2017.

**Other sources used in this part**

19. X. Zhai, J. Puigcerver, A. Kolesnikov, et al. [*A Large-scale Study of Representation Learning with the Visual Task Adaptation Benchmark*](https://arxiv.org/abs/1910.04867) (VTAB). arXiv 2019. The benchmark behind the 77.63% number.
20. L. Beyer, O. J. Hénaff, A. Kolesnikov, X. Zhai, A. van den Oord. [*Are we done with ImageNet?*](https://arxiv.org/abs/2006.07159) (ImageNet-ReaL). arXiv 2020. The cleaned-up labels behind the 90.72% number.
21. A. Krizhevsky. [*The CIFAR-10 and CIFAR-100 datasets*](https://www.cs.toronto.edu/~kriz/cifar.html). University of Toronto, 2009.
22. Hugging Face model cards: [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224) and [`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50); the sample picture is the [`huggingface/cats-image`](https://huggingface.co/datasets/huggingface/cats-image) dataset.
23. Code for this part: [`vit_part1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part1.py) and [`vit_part1_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part1_math.py).
