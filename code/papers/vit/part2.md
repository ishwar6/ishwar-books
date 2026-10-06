---
title: "Inside ViT: Patches, Embeddings and the Encoder"
description: "Section 2, Figure 1, Section 3.1 and Appendix A, line by line: the earlier attempts to put attention on images, then every step from a 224×224 picture to a class label, with Equations 1 to 8 recomputed by hand on the released ViT-B/16 and matched against the library at every stage, the 86,567,656 parameters counted exactly, and a four-token attention example small enough to check with a pencil."
part: 2
covers: "§2, §3.1, Figure 1, Eq. 1 to 4, Appendix A"
date: 2026-10-06
tags: [vit, vision-transformer, transformers, computer-vision]
---

In [Part 1](part-1-the-big-idea.md) we read the paper's claim: cut a picture into 16×16 patches, feed them to an unchanged Transformer, and with enough data it beats the best convolutional networks. This part opens the model. First we read Section 2, the earlier work the paper builds on and argues against. Then Figure 1 and Section 3.1, which describe the model in one picture, four equations and a few paragraphs. Last we read Appendix A, the attention equations the paper hands to its appendix.

Everything that can be checked, we check on a released model, `google/vit-base-patch16-224` (ViT-B/16, pre-trained on ImageNet-21k and fine-tuned on ImageNet at 224 pixels), with one real picture of two cats. Every step of the model is recomputed in plain PyTorch and compared with the library's own output.

## Related work: attention on images before ViT {§2}

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 2 · page 2
> [![The first paragraph of Section 2: Transformers were proposed by Vaswani et al. (2017) for machine translation and have since become the state of the art method in many NLP tasks. Large Transformer-based models are often pre-trained on large corpora and then fine-tuned: BERT uses a denoising self-supervised pre-training task, while the GPT line of work uses language modeling as its pre-training task](/img/papers/vit/p2-related-transformers.png)](/img/papers/vit/p2-related-transformers.png)
>
> **Context:** the opening of Related Work, right after the headline numbers of the introduction.
>
> **What it says:** Transformers "were proposed by Vaswani et al. (2017) for machine translation" and are now the standard in language. Big ones are pre-trained and then fine-tuned: BERT with "a denoising self-supervised pre-training task" (hide words, predict them), the GPT line with "language modeling" (predict the next word).
>
> **Why it matters:** this is the recipe ViT wants to copy for images: a plain Transformer, pre-trained on a lot of data, then fine-tuned. The whole paper is about whether the recipe survives the move from words to pixels.

> [!DEFINITION] Transformer encoder
> A stack of identical layers that take a sequence of vectors in and give a sequence of vectors of the same size out. Each layer lets every position look at every other position (self-attention) and then processes each position on its own (an MLP). It is the half of the 2017 Transformer that reads; BERT is one, and so is ViT. The [attention series](/writings/attention-1-self-attention/) builds it up from scratch.

> [!DEFINITION] Self-attention
> The step in which every token in a sequence compares itself with every other token, decides how much to look at each one, and takes a weighted mix of their information. The weights are computed from the data, not fixed in advance. Appendix A of the paper, covered at the end of this part, writes it out.

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 2 · page 2
> [![The second paragraph of Section 2: naive application of self-attention to images would require that each pixel attends to every other pixel, with quadratic cost in the number of pixels. Parmar et al. applied self-attention only in local neighborhoods for each query pixel; such local blocks can completely replace convolutions (Hu, Ramachandran, Zhao); Sparse Transformers (Child et al.) employ scalable approximations; attention in blocks of varying sizes (Weissenborn) or only along individual axes (Ho, Wang); these require complex engineering to be implemented efficiently on hardware accelerators](/img/papers/vit/p2-related-approx.png)](/img/papers/vit/p2-related-approx.png)
>
> **Context:** the longest paragraph of Section 2. It lists the ways people made attention cheap enough for pictures.
>
> **What it says:** attention over pixels means "each pixel attends to every other pixel", which has "quadratic cost in the number of pixels" and "does not scale to realistic input sizes". So people approximated: attention "only in local neighborhoods for each query pixel" (Parmar et al., 2018), local attention blocks that "can completely replace convolutions" (Hu et al., 2019; Ramachandran et al., 2019; Zhao et al., 2020), "Sparse Transformers" (Child et al., 2019), attention "in blocks of varying sizes" (Weissenborn et al., 2019) or "only along individual axes" (Ho et al., 2019; Wang et al., 2020a). All of them work, but "require complex engineering to be implemented efficiently on hardware accelerators".
>
> **Why it matters:** the last sentence is the argument. Every approximation is a custom attention pattern that needs custom code to run fast. ViT avoids the problem a different way: it does not make attention cheaper, it makes the sequence shorter (196 patches instead of 50,176 pixels), so plain attention is affordable.

> [!DEFINITION] Quadratic cost
> If a computation over $$n$$ items has to look at every pair of items, its cost grows with $$n^2$$. Double the items and the work is four times larger. Self-attention over $$n$$ tokens has this cost, because every token compares itself with every token. Part 1 works out the numbers for pixels.

> [!DEFINITION] Locality
> The assumption that what matters for a pixel is mostly the pixels near it. Convolutions build it in: each output only sees a small window of the input. Local attention borrows the same assumption to cut the cost.

> [!DEFINITION] Global versus local
> A **global** operation lets every position see every other position at once (plain self-attention). A **local** one lets each position see only a neighbourhood (a convolution, or local attention). Global is more flexible; local is cheaper and carries a built-in assumption.

> [!DEFINITION] Sparse attention
> Self-attention in which each token looks at only a chosen subset of the tokens rather than all of them, following a fixed pattern such as "my row, plus every k-th token". The cost drops from quadratic towards linear, at the price of a pattern that has to be designed and implemented by hand.

> [!DEFINITION] Axial attention
> Sparse attention for grids: a pixel attends first to the other pixels in its row, then (in another layer) to those in its column. Two cheap passes cover the whole image indirectly. Ho et al. (2019) and Wang et al. (2020a) use it.

Two of these ideas are worth seeing in their own papers. Here is Parmar et al. (2018), the Image Transformer, saying in its abstract why it went local:

> [!PAPER] Parmar et al. (2018), Image Transformer · Abstract
> [![The abstract of the Image Transformer: by restricting the self-attention mechanism to attend to local neighborhoods we significantly increase the size of images the model can process in practice, despite maintaining significantly larger receptive fields per layer than typical convolutional neural networks](/img/papers/vit/p2-ext-parmar.png)](/img/papers/vit/p2-ext-parmar.png)
>
> **Context:** the paper the ViT authors cite for "self-attention only in local neighborhoods". Its goal was generating images pixel by pixel, not classifying them.
>
> **What it says:** "By restricting the self-attention mechanism to attend to local neighborhoods we significantly increase the size of images the model can process in practice", while still seeing more than a convolution does.
>
> **Why it matters:** this is the trade every method in the paragraph makes: give up global attention to afford larger images. ViT keeps global attention and gives up pixels instead.

And here are the patterns of the Sparse Transformer, which the ViT paper names explicitly:

> [!PAPER] Child et al. (2019), Generating Long Sequences with Sparse Transformers · Figure 3
> [![Figure 3 of the Sparse Transformer paper: three connectivity patterns for a 6 by 6 image. (a) full attention of a standard Transformer, (b) strided sparse attention where a position attends to its row and to every k-th position, (c) fixed sparse attention where positions attend within blocks and to summary positions; the bottom row shows each pattern as a connectivity matrix](/img/papers/vit/p2-ext-child-fig3.png)](/img/papers/vit/p2-ext-child-fig3.png)
>
> **Context:** "two 2d factorized attention schemes" compared with full attention, for a tiny 6×6 image.
>
> **What it says:** in (a) every output can see every input (the full lower triangle of the matrix). In (b) and (c) only the coloured cells are computed: a row, a column stride, a block. The patterns are chosen so that two heads in sequence still connect everything.
>
> **Why it matters:** the pictures make "scalable approximations to global self-attention" concrete: it means deleting most cells of the attention matrix by hand. ViT's attention matrix is the full one, (a), but only 197×197.

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 2 · page 2
> [![The third paragraph of Section 2: most related to ours is the model of Cordonnier et al. (2020), which extracts patches of size 2 by 2 from the input image and applies full self-attention on top. This model is very similar to ViT, but our work goes further to demonstrate that large scale pre-training makes vanilla transformers competitive with state-of-the-art CNNs. Cordonnier et al. use a small patch size of 2 by 2 pixels, which makes the model applicable only to small-resolution images](/img/papers/vit/p2-related-cordonnier.png)](/img/papers/vit/p2-related-cordonnier.png)
>
> **Context:** the paragraph that names the closest earlier work.
>
> **What it says:** Cordonnier et al. (2020) "extracts patches of size 2 × 2 from the input image and applies full self-attention on top". "This model is very similar to ViT." The differences the authors claim: ViT shows that "large scale pre-training makes vanilla transformers competitive with (or even better than) state-of-the-art CNNs", and 2×2 patches make the earlier model "applicable only to small-resolution images", while 16×16 patches handle "medium-resolution images as well".
>
> **Why it matters:** this is the honest answer to "was the idea new?". No: patches plus full attention existed. What was new is the scale, both of the patches and of the data, and the result that followed.

> [!DEFINITION] Patch
> A small square cut out of the picture, here 16×16 pixels with 3 colour channels. ViT treats each patch as one token, the way BERT treats each word piece. The picture is cut into a grid of patches with no overlap.

The Cordonnier paper is about something else: it proves that attention *can* do what a convolution does, and shows that trained attention layers often learn to. Its 2×2 step appears in the experiments as a down-sampling detail:

> [!PAPER] Cordonnier, Loukas and Jaggi (2020), On the Relationship between Self-Attention and Convolutional Layers · Abstract
> [![The abstract of Cordonnier et al.: this work provides evidence that attention layers can perform convolution and indeed they often learn to do so in practice. Specifically, we prove that a multi-head self-attention layer with sufficient number of heads is at least as expressive as any convolutional layer](/img/papers/vit/p2-ext-cordonnier-abs.png)](/img/papers/vit/p2-ext-cordonnier-abs.png)
> [![The experimental setup of Cordonnier et al.: in all experiments, we use a 2 by 2 invertible down-sampling on the input to reduce the size of the image, since the attention coefficient tensors scale quadratically with the size of the input image and full attention cannot be applied to bigger images](/img/papers/vit/p2-ext-cordonnier-2x2.png)](/img/papers/vit/p2-ext-cordonnier-2x2.png)
>
> **Context:** the abstract, and the experiments section on CIFAR-10 (32×32 images).
>
> **What it says:** they "prove that a multi-head self-attention layer with sufficient number of heads is at least as expressive as any convolutional layer". In the experiments "we use a 2 × 2 invertible down-sampling on the input", because the attention tensors "scale quadratically with the size of the input image" and "full attention cannot be applied to bigger images".
>
> **Why it matters:** their 2×2 groups are the "patches" the ViT paper refers to. On a 32×32 image that gives 256 tokens; on a 224×224 image it would give 12,544, far too many. ViT's 16×16 patches give 196 at 224×224. Same idea, eight times larger patches, and a different question.

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 2 · page 2
> [![The fourth paragraph of Section 2: there has also been a lot of interest in combining convolutional neural networks with forms of self-attention, by augmenting feature maps for image classification (Bello et al.) or by further processing the output of a CNN using self-attention, for object detection (Hu et al., Carion et al.), video processing (Wang et al., Sun et al.), image classification (Wu et al.), unsupervised object discovery (Locatello et al.) or unified text-vision tasks (Chen et al., Lu et al., Li et al.)](/img/papers/vit/p2-related-cnn-attn.png)](/img/papers/vit/p2-related-cnn-attn.png)
>
> **Context:** the fourth paragraph: attention as an add-on to a CNN rather than a replacement.
>
> **What it says:** people have combined CNNs "with forms of self-attention", either "by augmenting feature maps" inside the CNN (Bello et al., 2019) or "by further processing the output of a CNN using self-attention". The second kind covers object detection (relation networks; DETR), video (non-local networks; VideoBERT), classification (Visual Transformers), object discovery (slot attention) and joint text-and-image models (UNITER, ViLBERT, VisualBERT).
>
> **Why it matters:** in all of these the CNN does the seeing and attention does the reasoning on top. ViT's "hybrid" model, later in Section 3.1, is exactly this kind of design, and the paper tests it against the pure version.

> [!DEFINITION] Feature map
> The output of a convolutional layer: a grid of positions, each holding a vector of numbers. A ResNet turns a 224×224×3 picture into, for example, a 14×14×1024 feature map: 196 positions, each described by 1,024 numbers. The hybrid ViT reads these positions as tokens.

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 2 · page 2
> [![The fifth paragraph of Section 2: another recent related model is image GPT (iGPT), which applies Transformers to image pixels after reducing image resolution and color space. The model is trained in an unsupervised fashion as a generative model, and the resulting representation can then be fine-tuned or probed linearly for classification performance, achieving a maximal accuracy of 72% on ImageNet](/img/papers/vit/p2-related-igpt.png)](/img/papers/vit/p2-related-igpt.png)
>
> **Context:** the one earlier model that used a Transformer on raw pixels at scale.
>
> **What it says:** iGPT "applies Transformers to image pixels after reducing image resolution and color space". It is trained "as a generative model" (predict the next pixel), and its representation "can then be fine-tuned or probed linearly", reaching "a maximal accuracy of 72% on ImageNet".
>
> **Why it matters:** iGPT shows the other way to shorten the sequence: shrink the picture (to 32×32 or 64×64) and squash the colours into 512 values, so a pixel is a token. The price is 72% on ImageNet, far below the 85% to 88% of good CNNs. ViT keeps the resolution and shortens the sequence by patching.

> [!DEFINITION] Generative model
> A model trained to produce the data itself, for example to predict the next pixel given the previous ones (iGPT) or the next word given the previous ones (GPT). Nothing is labelled; the data is its own target.

> [!DEFINITION] iGPT
> Image GPT (Chen et al., 2020a): a GPT-2-style Transformer run over the pixels of small images, one pixel per token, trained to predict the next pixel.

> [!DEFINITION] Linear probe
> A test of a representation: freeze the model, take its output vectors, and train only one linear layer on top to predict the classes. If the probe scores well, the frozen vectors already contain the information. Part 4 uses a related "few-shot linear" measure.

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 2 · page 2
> [![The last paragraph of Section 2: our work adds to the increasing collection of papers that explore image recognition at larger scales than the standard ImageNet dataset. Additional data sources allow state-of-the-art results (Mahajan et al., Touvron et al., Xie et al.). Sun et al. (2017) study how CNN performance scales with dataset size, and Kolesnikov et al. (2020) and Djolonga et al. (2020) explore CNN transfer learning from ImageNet-21k and JFT-300M. We focus on these two latter datasets as well, but train Transformers instead of ResNet-based models](/img/papers/vit/p2-related-scale.png)](/img/papers/vit/p2-related-scale.png)
>
> **Context:** the closing paragraph, which places ViT in the "more data" line of work rather than the "new attention" line.
>
> **What it says:** the paper "adds to the increasing collection of papers that explore image recognition at larger scales than the standard ImageNet dataset": Mahajan et al. (2018, Instagram hashtags), Touvron et al. (2019), Xie et al. (2020, Noisy Student). Sun et al. (2017) studied "how CNN performance scales with dataset size"; Kolesnikov et al. (2020, Big Transfer) and Djolonga et al. (2020) studied transfer from "ImageNet-21k and JFT-300M". ViT uses "these two latter datasets as well, but train[s] Transformers instead of ResNet-based models".
>
> **Why it matters:** this tells you how to read the results in Part 4. The comparison that matters to the authors is with Big Transfer, the same team's ResNets trained on the same data. Same data, same recipe, different architecture.

{{FIG:p2_related_map|The related work on one map. Five earlier ways of putting attention on images, each a different answer to the quadratic cost: local attention, sparse and axial patterns, 2×2 patches with full attention, attention bolted on to a CNN, and iGPT on shrunken pixels. ViT, below, keeps full attention and the standard encoder, uses 16×16 patches, and joins the large-data line of work.}}

## Figure 1: the whole model in one picture {§3}

Section 3 is short. It opens with one sentence of design philosophy and one picture, Figure 1. Here is the picture first, because everything else in this part explains one piece of it.

> [!PAPER] Dosovitskiy et al. (2020), ViT · Figure 1 · page 3
> [![Figure 1 of the ViT paper. Left: a picture is split into nine patches, each goes through a linear projection of flattened patches, position embeddings 1 to 9 are added along with an extra learnable class embedding at position 0, the sequence enters a Transformer encoder, and an MLP head on the class token outputs Bird, Ball, Car. Right: the Transformer encoder block, repeated L times: Norm, Multi-Head Attention, a residual addition, Norm, MLP, another residual addition. Caption: we split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder; to perform classification we use the standard approach of adding an extra learnable classification token](/img/papers/vit/p2-figure1.png)](/img/papers/vit/p2-figure1.png)
>
> **Context:** the top of page 3, the one figure of the method section.
>
> **What it says:** "We split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. In order to perform classification, we use the standard approach of adding an extra learnable 'classification token' to the sequence."
>
> **Why it matters:** the caption is the whole method in two sentences. Learn to read the picture and the four equations of Section 3.1 are just its labels.

How to read Figure 1, left to right and bottom to top:

- **Bottom left, the picture:** split into a grid of patches (nine in the drawing; 196 for a real 224×224 picture with 16×16 patches).
- **"Linear Projection of Flattened Patches" (pink bar):** each patch is flattened into one long row of numbers and multiplied by one matrix, $$\mathbf{E}$$. Every patch uses the same matrix.
- **"Patch + Position Embedding" (the circles 0 to 9):** to each projected patch a learned position vector is added, so the model can tell patch 3 from patch 7. The starred circle 0 is the "extra learnable [class] embedding": a token that belongs to no patch.
- **"Transformer Encoder" (grey box):** the standard encoder, $$L$$ layers of it. Every token can look at every token.
- **"MLP Head" and "Class":** only the output at position 0, the class token, is read. A small head turns it into a label.
- **Right half, the encoder block:** "Norm", "Multi-Head Attention", an addition; "Norm", "MLP", an addition. Note that Norm comes *before* each block, with the plus signs *after*. That is the "pre-norm" arrangement, and it is Equations 2 and 3.

{{FIG:p2_figure1_frames|Figure 1 redrawn as nine numbered frames with the real sizes. The 224×224×3 picture is cut into 196 patches of 16×16×3; each is flattened to 768 numbers and multiplied by E (768×768); the [class] token is prepended and 197 position rows are added; the 12-layer encoder keeps the shape 197×768; the [class] output is normalised and a single linear layer gives 1000 scores, the largest of which is "Egyptian cat" with probability 0.937 on our picture.}}

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 3 · page 3
> [![The opening of Section 3: in model design we follow the original Transformer (Vaswani et al., 2017) as closely as possible. An advantage of this intentionally simple setup is that scalable NLP Transformer architectures, and their efficient implementations, can be used almost out of the box](/img/papers/vit/p2-method-intro.png)](/img/papers/vit/p2-method-intro.png)
>
> **Context:** the two sentences between the Figure 1 caption and Section 3.1.
>
> **What it says:** "we follow the original Transformer (Vaswani et al., 2017) as closely as possible." The reason: "scalable NLP Transformer architectures, and their efficient implementations, can be used almost out of the box."
>
> **Why it matters:** this is a deliberate design choice, not laziness. The paragraph on related work complained that custom attention needs custom engineering. A standard Transformer needs none: the code, the hardware kernels and the scaling know-how from language transfer directly. Later parts show that this is what allowed the authors to train 632-million-parameter models.

The phrase "out of the box" is also why we can check this paper so thoroughly. ViT-B/16 has the same layer count, hidden size, head count and MLP size as BERT-base ([BERT Part 2](/papers/bert/part-2-architecture-and-input/)), and the released code is a few hundred lines on top of a standard Transformer implementation.

{{FIG:p2_bert_vs_vit|BERT and ViT side by side. BERT reads word pieces with a [CLS] token in front; ViT reads 16×16 patches with a [class] token in front. BERT embeds a token by table lookup plus segment plus position; ViT embeds a patch by a linear projection plus position. The encoder in between has the same dimensions (12 layers, width 768, 12 heads, MLP 3072), and in both the first output token feeds the task head.}}

## From pixels to a sequence: the reshape {§3.1}

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 3.1 · page 3
> [![The first paragraph of Section 3.1: the standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, we reshape the image x in R H by W by C into a sequence of flattened 2D patches x p in R N by P squared times C, where (H, W) is the resolution of the original image, C is the number of channels, (P, P) is the resolution of each image patch, and N equals HW over P squared is the resulting number of patches, which also serves as the effective input sequence length. The Transformer uses constant latent vector size D through all of its layers, so we flatten the patches and map to D dimensions with a trainable linear projection (Eq. 1). We refer to the output of this projection as the patch embeddings](/img/papers/vit/p2-vit-reshape.png)](/img/papers/vit/p2-vit-reshape.png)
>
> **Context:** the first paragraph of Section 3.1, which does the only real translation work in the paper: from a 2D picture to a 1D sequence.
>
> **What it says:** a Transformer wants "a 1D sequence of token embeddings". So the picture $$\mathbf{x} \in \mathbb{R}^{H \times W \times C}$$ is reshaped into "a sequence of flattened 2D patches" $$\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$$. $$(H, W)$$ is the picture's resolution, $$C$$ its number of channels, $$(P, P)$$ the patch size, and $$N = HW/P^2$$ the number of patches, "which also serves as the effective input sequence length". Because the Transformer "uses constant latent vector size $$D$$ through all of its layers", the flattened patches are mapped "to $$D$$ dimensions with a trainable linear projection". The result is called "the patch embeddings".
>
> **Why it matters:** five symbols ($$H, W, C, P, N$$) and one matrix do the whole job. There is no convolution stack, no feature pyramid, nothing learned about pictures before the Transformer sees them. The rest of the paper's "surprising" results rest on how little happens here.

> [!DEFINITION] Tensor shape
> The list of sizes of an array of numbers. A colour picture of height 224 and width 224 has shape $$224 \times 224 \times 3$$ (in PyTorch the channels come first: $$3 \times 224 \times 224$$). A sequence of 197 tokens with 768 numbers each has shape $$197 \times 768$$. Following the shapes is the easiest way to follow a model.

> [!DEFINITION] Reshape
> Rearranging the same numbers into a different shape without changing any of them. Cutting a picture into patches and stacking them as rows is a reshape: no arithmetic, only a new order.

> [!DEFINITION] Flattening
> Reshaping a block of numbers into one long row. A 16×16×3 patch flattened is a row of 768 numbers. The order is fixed (here: channel, then row, then column), so position 0 is always the red value of the top-left pixel.

> [!DEFINITION] Sequence length N
> How many tokens the Transformer reads at once. For text it is the number of word pieces; for ViT it is the number of patches, $$N = HW/P^2$$, plus one for the class token. Attention cost grows with $$N^2$$, so this number decides what the model can afford.

> [!DEFINITION] Hidden size D
> How many numbers describe each token inside the model. ViT-B uses $$D = 768$$, the same as BERT-base. It is the same from the first layer to the last, which is what "constant latent vector size" means.

> [!DEFINITION] Linear projection
> Multiplying a vector by a matrix (and usually adding a bias vector). It maps a row of $$a$$ numbers to a row of $$b$$ numbers with $$a \times b$$ learned weights. "Trainable" means the matrix is learned with the rest of the model.

The arithmetic for our picture, with the paper's symbols:

$$
N = \frac{HW}{P^2} = \frac{224 \times 224}{16^2} = \frac{50{,}176}{256} = 196, \qquad P^2 \cdot C = 16 \times 16 \times 3 = 768
$$

where $$H = W = 224$$ is the resolution the model was trained at, $$P = 16$$ is the patch size (the "/16" in "ViT-B/16"), and $$C = 3$$ for red, green and blue. So the picture becomes $$\mathbf{x}_p$$, a matrix of 196 rows and 768 columns. A happy coincidence of ViT-B/16 at 224: the row length $$P^2 C = 768$$ equals $$D = 768$$, so $$\mathbf{E}$$ happens to be square. For ViT-B/32 the rows are $$32 \cdot 32 \cdot 3 = 3{,}072$$ long and $$\mathbf{E}$$ is $$3{,}072 \times 768$$.

{{FIG:p2_reshape|The reshape of Section 3.1 drawn out. Left: the 224×224×3 picture as a 14×14 grid of 16-pixel patches (224/16 = 14 per side). Right: the matrix x_p with 196 rows and 768 columns; row 0 is the top-left patch, row 195 the bottom-right. N = 224·224/16² = 50,176/256 = 196.}}

Here is the picture we use, resized to 224×224 the way the released preprocessing does it, with the 14×14 patch grid drawn on top:

<figure><img src="/img/papers/vit/p2-sample-grid.png" alt="The sample picture: two cats lying on a red sofa with remote controls, resized to 224 by 224 pixels and shown with a 14 by 14 grid of white lines marking the 16 by 16 pixel patches" loading="lazy" width="448" /><figcaption>Our test picture (from the `huggingface/cats-image` dataset) at 224×224 with the 196 patches marked. Each cell is one token for the model.</figcaption></figure>

Cutting and flattening in PyTorch is three lines ([`vit_part2_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part2_math.py)). `unfold` slides a 16-wide window with step 16 along the height and then the width, which gives one 16×16 square per patch and per channel; a `permute` and a `reshape` put the patches as rows:

```python
patches = x.unfold(1, P, P).unfold(2, P, P)                      # (3, 14, 14, 16, 16)
patches = patches.permute(1, 2, 0, 3, 4).reshape(N, C * P * P)   # (196, 768): row n = patch n
```

```text
== 1. cut x (3, 224, 224) into 196 patches of 16x16x3 and flatten each to 768 numbers ==
  x.unfold -> (3, 14, 14, 16, 16)  (channels, 14 rows of patches, 14 columns, 16, 16)
  x_p = flattened patches: (196, 768)   (N = 196 patches, P^2*C = 768 numbers each)
  patch 0 (top-left corner), first 8 of its 768 numbers: [  0.114,   0.169,   0.184,   0.200,   0.208,   0.239,   0.231,   0.200]
  patch 0, number 0 is red channel, pixel (0,0): x[0,0,0] = 0.114; number 256 is green (0,0): -0.804; number 512 is blue (0,0): -0.545
  check: patch 17 (row 1, col 3) == x[:, 16:32, 48:64] flattened: True
```

The numbers are pixel values after the released preprocessing: each channel is scaled from 0 to 255 into $$[0, 1]$$ and then mapped to $$[-1, 1]$$ by $$(v - 0.5)/0.5$$. The top-left pixel of our picture is reddish (red 0.114, green −0.804, blue −0.545), which is the sofa. The check on patch 17 confirms the order: patch $$n$$ sits at grid row $$\lfloor n/14 \rfloor$$ and column $$n \bmod 14$$, so patch 17 is row 1, column 3, and it is exactly the pixels `x[:, 16:32, 48:64]`.

{{FIG:p2_patch_flatten|One 16×16×3 patch becomes one row of 768 numbers: the 256 red values first, then 256 green, then 256 blue. The strip shows the real first 40 numbers of patch 0 (the red channel of pixel row 0, then the start of row 1); the darker the square, the larger the value.}}

**The projection $$\mathbf{E}$$.** Each row of $$\mathbf{x}_p$$ is multiplied by the same matrix $$\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$$. One matrix product does all 196 patches at once:

$$
\underbrace{\mathbf{x}_p}_{196 \times 768} \cdot \underbrace{\mathbf{E}}_{768 \times 768} = \underbrace{\mathbf{x}_p \mathbf{E}}_{196 \times 768}
$$

{{FIG:p2_E_matrix|The patch embedding is one matrix multiplication. x_p (196×768) times E (768×768) gives 196 patch embeddings of 768 numbers. E holds 768 × 768 = 589,824 weights plus a bias of 768, and it is the only thing the model learns about pixels before the Transformer.}}

There is one small gap between the paper and the released code here, and it is worth closing. The paper says "linear projection". The released PyTorch model (and the authors' original code) implement it as a **convolution** with a 16×16 kernel and a stride of 16. These are the same operation: a convolution with kernel size equal to its stride touches each patch exactly once, and what it does to a patch is a dot product with each of its 768 filters, which is a linear layer. I checked by reshaping the convolution's weight into a $$768 \times 768$$ matrix and multiplying the flattened patches with it:

```python
conv = model.vit.embeddings.patch_embeddings.projection    # Conv2d(3, 768, kernel_size=16, stride=16)
E = conv.weight.reshape(768, 768).T                        # (P²·C) × D: column d is filter d, flattened
ours = patches @ E + conv.bias                             # Eq. 1's x_p E, for all 196 patches
theirs = conv(pixel_values).flatten(2).transpose(1, 2)[0]  # what the library computes
print((ours - theirs).abs().max())
```

```text
== 2. the patch embedding E: the library's Conv2d(3, 768, kernel 16, stride 16) equals Linear(768 -> 768) on x_p ==
  conv weight (768, 3, 16, 16) reshaped to E (768, 768) (rows = the 768 pixel numbers of a patch, columns = D = 768)
  x_p E: (196 x 768) . (768 x 768) = (196, 768)
  max |difference|, our x_p E + b vs the library conv: 5.13e-06
  patch 0 embedding, first 6 of 768: [  0.058,  -0.024,  -0.248,   3.689,   0.356,   0.098]
```

A difference of $$5 \times 10^{-6}$$ is rounding noise in 32-bit arithmetic. So "linear projection of flattened patches" and "16×16 convolution with stride 16" are two names for the same 589,824 weights. The convolution is simply the faster way to run it.

{{FIG:p2_conv_linear|Two views of the same patch embedding. Left: the library's convolution places a 16×16×3 filter on each patch with stride 16, so there is no overlap, and 768 filters give 768 numbers per patch. Right: Eq. 1's linear layer on the flattened rows. The weights are the same numbers reshaped; the outputs agree to 5e-6.}}

## The [class] token and the head {§3.1}

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 3.1 · page 3
> [![The second paragraph of Section 3.1: similar to BERT's class token, we prepend a learnable embedding to the sequence of embedded patches (z 0 0 equals x class), whose state at the output of the Transformer encoder (z L 0) serves as the image representation y (Eq. 4). Both during pre-training and fine-tuning, a classification head is attached to z L 0. The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time](/img/papers/vit/p2-vit-class.png)](/img/papers/vit/p2-vit-class.png)
>
> **Context:** the second paragraph of Section 3.1, about the one token that is not a patch.
>
> **What it says:** "Similar to BERT's [class] token, we prepend a learnable embedding to the sequence of embedded patches ($$\mathbf{z}_0^0 = \mathbf{x}_{\text{class}}$$)". Its state at the encoder output, $$\mathbf{z}_L^0$$, "serves as the image representation $$\mathbf{y}$$ (Eq. 4)". A classification head sits on $$\mathbf{z}_L^0$$ both in pre-training and fine-tuning; it is "a MLP with one hidden layer at pre-training time and ... a single linear layer at fine-tuning time".
>
> **Why it matters:** the model does not average the 196 patch outputs. It reads one token, which never saw a pixel directly and has to gather everything through attention. This is copied from BERT, and the paper's Appendix D.3 (Part 5) shows that averaging would have worked too, with a different learning rate.

> [!DEFINITION] [class] token
> An extra token at position 0 that belongs to no patch. Its input vector $$\mathbf{x}_{\text{class}}$$ is a learned list of 768 numbers, the same for every picture. Through 12 layers of attention it collects information from all patches, and its output vector is what the classifier reads. BERT calls the same thing `[CLS]`.

> [!DEFINITION] Learnable embedding
> A vector of numbers that is a parameter of the model: it starts random and is adjusted by training like any weight. The class token and the position embeddings are learnable embeddings. Nothing computes them from the input; they are simply looked up.

> [!DEFINITION] Image representation
> The one vector that stands for the whole picture, here $$\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$$, 768 numbers. Everything the classifier knows about the picture must be in it.

> [!DEFINITION] Classification head
> The last, small piece of the model that turns the image representation into one score per class. For 1,000 ImageNet classes and $$D = 768$$ a linear head is a $$768 \times 1000$$ matrix and 1,000 biases.

> [!DEFINITION] Logits
> The raw scores the head outputs, one per class, before they are turned into probabilities. Softmax (defined below) converts them: the largest logit becomes the largest probability.

Here is the BERT sentence the paper points to:

> [!PAPER] Devlin et al. (2018), BERT · Section 3, Input/Output Representations
> [![The BERT paper: the first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks](/img/papers/vit/p2-ext-bert-cls.png)](/img/papers/vit/p2-ext-bert-cls.png)
>
> **Context:** the paragraph of the BERT paper that introduces `[CLS]`.
>
> **What it says:** "The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks."
>
> **Why it matters:** ViT took this sentence as is and renamed the token. "Aggregate sequence representation" becomes "image representation".

The two heads can be seen in the released checkpoints. The fine-tuned model `google/vit-base-patch16-224` ends in exactly one linear layer, and the pre-trained-only model `google/vit-base-patch16-224-in21k` ends in a dense layer with a tanh (a "pooler", the hidden layer of the pre-training MLP head; the 21,843-class output layer on top of it was not released):

```text
  head of the fine-tuned checkpoint: Linear(in_features=768, out_features=1000, bias=True)   (one linear layer, as Section 3.1 says for fine-tuning)
  the pre-trained-only checkpoint google/vit-base-patch16-224-in21k ends in: ViTPooler(
  (dense): Linear(in_features=768, out_features=768, bias=True)
  (activation): Tanh()
)
  fine-tuned checkpoint has a pooler: False;  in21k checkpoint has a pooler: True
  (the paper: an MLP with one hidden layer at pre-training time, a single linear layer at fine-tuning time)
```

So the released files match the sentence: an MLP with one hidden layer for pre-training, one linear layer after fine-tuning. [Part 3](part-3-fine-tuning-and-setup.md) covers how the head is swapped and why it starts from zeros.

## Position embeddings {§3.1}

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 3.1 · page 3
> [![The third paragraph of Section 3.1: position embeddings are added to the patch embeddings to retain positional information. We use standard learnable 1D position embeddings, since we have not observed significant performance gains from using more advanced 2D-aware position embeddings (Appendix D.4). The resulting sequence of embedding vectors serves as input to the encoder](/img/papers/vit/p2-vit-pos.png)](/img/papers/vit/p2-vit-pos.png)
>
> **Context:** the third paragraph of Section 3.1.
>
> **What it says:** position embeddings "are added to the patch embeddings to retain positional information". They are "standard learnable 1D position embeddings", because "more advanced 2D-aware position embeddings (Appendix D.4)" did not help significantly. The sum "serves as input to the encoder".
>
> **Why it matters:** "1D" means the model is told "you are token 37", not "you are at row 2, column 9". It has to work out the grid from data. Appendix D.4, covered in [Part 5](part-5-scaling-and-inside-vit.md), shows that it does: the learned position vectors end up arranged in a grid.

> [!DEFINITION] Position embedding
> A learned vector for each position in the sequence, added to the token at that position. Attention on its own does not know the order of its inputs (we show this below), so without these vectors the model could not tell the top-left patch from the bottom-right one. ViT-B/16 at 224 learns 197 of them, one per patch plus one for the class token.

Why are they needed at all? Because self-attention treats its input as a *set*. Shuffle the input rows and the output rows shuffle in exactly the same way, with the same numbers. We check this on the tiny four-token example of the Appendix A section below ([`vit_part2_attn.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part2_attn.py)):

```text
== permutation equivariance: shuffle the input rows, the output rows shuffle the same way ==
  new order of the tokens: ['t2', 't0', 't3', 't1']
  MSA(z shuffled)  shape (4, 6)      (MSA(z) in the original order is printed in the Appendix A section)
       t2  1.784  2.885  0.000 -0.442 -2.007 -1.173
       t0  1.510  0.045  0.000  0.070 -1.023 -0.240
       t3  2.091  1.299  0.000 -0.228 -1.766 -0.560
       t1  1.679  0.231  0.000 -0.155 -1.116 -0.110
  MSA(z shuffled) == MSA(z) with its rows shuffled the same way: True   (max |difference| 4.8e-07)
  so without position embeddings the model cannot tell where a patch came from: only the set of patches matters.
  with a position row added before attention (z + E_pos), the shuffled run no longer matches: max |difference| 3.875
```

Compare with the original-order output in the Appendix A section: the rows for t2 and t0 swap places and nothing else changes. A model built only from attention and per-token MLPs would give the same class to a picture and to the same picture with its patches scrambled. Adding a different vector to each position breaks the tie: the shuffled run now differs by up to 3.875.

{{FIG:p2_permutation|Permutation equivariance in the tiny example. The four tokens go through multi-head self-attention in the original order (left) and in a shuffled order (right); the dashed lines join equal output rows. The numbers are identical, only the order moved. With a position row added to each token first, the two runs differ.}}

## The encoder: LayerNorm, attention, MLP {§3.1}

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 3.1 · pages 3 and 4
> [![The fourth paragraph of Section 3.1: the Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded self-attention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before every block, and residual connections after every block (Wang et al., 2019; Baevski and Auli, 2019)](/img/papers/vit/p2-vit-encoder.png)](/img/papers/vit/p2-vit-encoder.png)
> [![The paragraph continues at the top of page 4: the MLP contains two layers with a GELU non-linearity, followed by Equations 1 to 4](/img/papers/vit/p2-eq1-4.png)](/img/papers/vit/p2-eq1-4.png)
>
> **Context:** the description of the encoder, which runs from the bottom of page 3 to the top of page 4, where the four equations follow.
>
> **What it says:** the encoder "consists of alternating layers of multiheaded self-attention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3)". "Layernorm (LN) is applied before every block, and residual connections after every block", citing Wang et al. (2019) and Baevski and Auli (2019) for this pre-norm arrangement. "The MLP contains two layers with a GELU non-linearity."
>
> **Why it matters:** this is the standard encoder with the one modern change: normalising *before* each block rather than after, which the two cited papers found makes deep Transformers train more stably. BERT normalises after. Everything else is 2017.

> [!DEFINITION] Multi-head self-attention (MSA)
> Several self-attention computations ("heads") run side by side on the same input, each with its own small set of weights, and their outputs joined and mixed by one more matrix. ViT-B runs 12 heads of 64 numbers each. Appendix A, at the end of this part, gives the equations.

> [!DEFINITION] LayerNorm (and why pre-norm)
> A step that takes one token's vector, subtracts its mean and divides by its spread so the 768 numbers have mean 0 and spread 1, then multiplies by a learned vector $$\gamma$$ and adds a learned vector $$\beta$$. It keeps the numbers in a stable range. **Pre-norm** puts it at the entrance of each block, so what is added back on the residual path is never normalised and the main path stays a clean running sum; this trains more stably in deep stacks. **Post-norm** (BERT, the original Transformer) normalises after the addition.

> [!DEFINITION] Residual connection
> Adding a block's input to its output: $$\text{out} = \text{in} + \text{block}(\text{in})$$. The block only has to learn a correction, and information can flow straight through the stack. The two plus signs in Figure 1 are residual connections.

> [!DEFINITION] MLP / feed-forward block
> Two linear layers with a non-linearity between them, applied to each token on its own: $$768 \to 3{,}072 \to 768$$ in ViT-B. It is the part of the layer that does not look at other tokens. The Transformer paper calls it the feed-forward network.

> [!DEFINITION] GELU
> The Gaussian Error Linear Unit (Hendrycks and Gimpel, 2016), $$\text{GELU}(x) = x \cdot \Phi(x)$$ where $$\Phi$$ is the standard normal cumulative distribution. It is a smooth cousin of ReLU ($$\max(0, x)$$): large positive numbers pass through, large negative ones become almost 0, and small negative ones come out slightly negative instead of exactly 0.

{{FIG:p2_encoder_block|One encoder layer, left to right. The input from the previous layer goes through LayerNorm and multi-head self-attention and is added back to itself (Eq. 2, the first residual arc), then through LayerNorm and the MLP and is added back again (Eq. 3, the second arc). Every box keeps the shape 197×768. LayerNorm and the MLP work on one token at a time; only MSA mixes tokens.}}

## Equations 1 to 4, with the real numbers {§3.1}

The four equations are the whole model. We take them in order and, for each, run the real ViT-B/16 on the cat picture by hand.

### Equation 1: the input sequence

The idea in one sentence: put the class token in front of the 196 projected patches, then add a position vector to each of the 197.

$$
\mathbf{z}_0 = [\mathbf{x}_{\text{class}};\ \mathbf{x}_p^1 \mathbf{E};\ \mathbf{x}_p^2 \mathbf{E};\ \cdots;\ \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}, \qquad \mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D},\ \mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}
$$

where:

- $$\mathbf{x}_p^n$$ is row $$n$$ of the patch matrix, the 768 pixel numbers of patch $$n$$ ($$1 \times 768$$);
- $$\mathbf{E}$$ is the projection, $$768 \times 768$$ here (the released model also adds a bias of 768, which the equation leaves out);
- $$\mathbf{x}_p^n \mathbf{E}$$ is the patch embedding of patch $$n$$, $$1 \times 768$$;
- $$\mathbf{x}_{\text{class}}$$ is the learned class token, $$1 \times 768$$;
- the square brackets stack the 197 rows into a $$197 \times 768$$ matrix;
- $$\mathbf{E}_{pos}$$ is the table of 197 learned position vectors, $$197 \times 768$$, added row by row: row 0 to the class token, row $$n$$ to patch $$n$$;
- $$\mathbf{z}_0$$ is the result, $$197 \times 768$$, the input of layer 1.

In code, with the library's own tables and our own patch embeddings from above:

```python
emb = model.vit.embeddings
x_class = emb.cls_token.detach()[0]                 # (1, 768)
E_pos = emb.position_embeddings.detach()[0]         # (197, 768)
z0 = torch.cat([x_class, our_patch_emb], 0) + E_pos # (197, 768)   Eq. 1
print((z0 - emb(pixel_values)[0]).abs().max())      # against the library's embedding layer
```

```text
== 3. Eq. 1: z_0 = [x_class; x_p^1 E; ...; x_p^N E] + E_pos ==
  x_class (1, 768)   [x_class; patches] (197, 768)   E_pos (197, 768)   z_0 (197, 768)
  x_class, first 6:        [  0.010,   0.015,  -0.267,  -0.001,   0.405,   0.054]
  E_pos[0] (for [class]):  [  0.010,   0.015,  -0.267,  -0.000,   0.406,   0.054]
  z_0[0] = x_class + E_pos[0]: [  0.020,   0.030,  -0.534,  -0.001,   0.811,   0.108]
  E_pos[1] (for patch 1):  [  0.156,  -0.124,   0.447,   0.009,   0.428,  -0.462]
  z_0[1] = x_p^1 E + E_pos[1]: [  0.214,  -0.148,   0.198,   3.698,   0.783,  -0.365]
  max |difference|, our z_0 vs the library embedding output: 5.13e-06
```

Check the second row by hand: patch 1's embedding started $$0.058, -0.024, -0.248, 3.689, \dots$$ (step 2 above); add $$\mathbf{E}_{pos}[1] = 0.156, -0.124, 0.447, 0.009, \dots$$ and you get $$0.214, -0.148, 0.198, 3.698$$, which is what the script prints. The class token and its position vector are a curiosity: in this checkpoint they are almost identical lists of numbers ($$0.010, 0.015, -0.267, \dots$$ in both), so $$\mathbf{z}_0[0]$$ is close to $$2\,\mathbf{x}_{\text{class}}$$. Nothing in the model requires that; the two vectors are only ever used as a sum, so training was free to split the sum between them any way it liked.

{{FIG:p2_class_pos|Equation 1 drawn with the real shapes. Top row: the learned class embedding and the 196 patch embeddings, 197 rows of 768. Middle: the 197 learned position rows. Bottom: their sum z_0. Below, the real first six numbers of the class position; our z_0 matches the library's embedding output to 5e-6.}}

### Equation 2: LayerNorm, attention, residual

The idea in one sentence: normalise every token, let the tokens look at each other, and add the result back to what came in.

$$
\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, \qquad \ell = 1 \dots L
$$

where:

- $$\mathbf{z}_{\ell-1}$$ is the output of the previous layer ($$\mathbf{z}_0$$ for the first), $$197 \times 768$$;
- $$\text{LN}$$ normalises each of the 197 rows on its own (formula below), shape unchanged;
- $$\text{MSA}$$ is multi-head self-attention, Appendix A, $$197 \times 768$$ in and out;
- $$+\ \mathbf{z}_{\ell-1}$$ is the residual connection;
- $$\mathbf{z}'_\ell$$ is the intermediate result, $$197 \times 768$$, which Equation 3 continues; $$L = 12$$ for ViT-B.

**LayerNorm by hand.** For one token's vector $$x$$ of $$D$$ numbers:

$$
\mu = \frac{1}{D}\sum_{i=1}^{D} x_i, \qquad \sigma = \sqrt{\frac{1}{D}\sum_{i=1}^{D}(x_i - \mu)^2 + \epsilon}, \qquad \text{LN}(x)_i = \gamma_i\,\frac{x_i - \mu}{\sigma} + \beta_i
$$

where $$\mu$$ is the mean of the 768 numbers, $$\sigma$$ their standard deviation (with a tiny $$\epsilon = 10^{-12}$$ added so that it is never zero), and $$\gamma, \beta$$ are two learned vectors of 768 numbers, the same for every token. Here it is on the class token entering layer 1:

```python
v = z0[0]                                                  # the [class] token, 768 numbers
ln = model.vit.layers[0].layernorm_before
mu, var = v.mean(), ((v - v.mean()) ** 2).mean()
ours = ln.weight * (v - mu) / torch.sqrt(var + 1e-12) + ln.bias
```

```text
  LayerNorm of the [class] token (768 numbers), eps = 1e-12:
    input, first 6:            [  0.020,   0.030,  -0.534,  -0.001,   0.811,   0.108]
    mean mu = 0.0034   variance = 0.0349   std = sqrt(var + eps) = 0.1869
    (x - mu)/std, first 6:     [  0.086,   0.141,  -2.875,  -0.023,   4.320,   0.561]
    gamma, first 6:            [  0.156,   0.191,   0.033,   0.119,   0.065,   0.126]
    beta, first 6:             [  0.008,   0.016,   0.091,   0.067,   0.007,  -0.034]
    gamma*(..)+beta, first 6:  [  0.022,   0.043,  -0.005,   0.064,   0.287,   0.036]
    library LayerNorm:         [  0.022,   0.043,  -0.005,   0.064,   0.287,   0.036]   max |difference| 6.0e-08
```

Check the fifth number: $$(0.811 - 0.0034)/0.1869 = 4.321$$, then $$0.065 \times 4.321 + 0.007 = 0.288$$, against the printed 0.287 (the inputs shown are rounded). Notice how small $$\gamma$$ is here (0.03 to 0.19): the trained model scales the normalised numbers down a lot before attention sees them.

{{FIG:p2_layernorm|LayerNorm step by step on the first six of the 768 numbers of the class token: the input, minus the mean (0.0034), divided by the standard deviation (0.1869), times the learned gamma, plus the learned beta. Orange cells are negative. Our hand computation matches the library to 6e-8.}}

**Attention, head by head.** The normalised $$197 \times 768$$ matrix is multiplied by three matrices to give queries, keys and values, each $$197 \times 768$$, which are split into 12 heads of 64 columns. For each head, every token's query is compared with every token's key, the 197 scores are divided by $$\sqrt{64} = 8$$ and turned into weights by softmax, and the token's output is the weighted sum of the 197 value rows. The 12 outputs ($$197 \times 64$$ each) are placed side by side into $$197 \times 768$$ and multiplied by one more $$768 \times 768$$ matrix. Appendix A, at the end of this part, is the formal version; here is the real one, for the class token's query in head 1 of layer 1:

```python
lay = model.vit.layers[0]; at = lay.attention
q = ln1 @ at.q_proj.weight.T + at.q_proj.bias          # (197, 768); the same for k and v
qh, kh, vh = (t.view(197, 12, 64).transpose(0, 1) for t in (q, k, v))   # (12, 197, 64)
scores = qh @ kh.transpose(1, 2) / 8.0                 # (12, 197, 197)
A = torch.softmax(scores, -1)                          # every row sums to 1
SA = A @ vh                                            # (12, 197, 64)
concat = SA.transpose(0, 1).reshape(197, 768)          # the 12 heads side by side
msa = concat @ at.o_proj.weight.T + at.o_proj.bias     # U_msa
z_prime = msa + z0                                     # Eq. 2
```

```text
  q = LN(z_0) W_q + b_q: (197, 768)  (the same for k and v); split into 12 heads of 64: (12, 197, 64)
  head 1: q_class . k_j / sqrt(64) for all 197 tokens j: (197,); first 6 scores: [  5.125,  -0.698,  -0.880,  -0.946,  -1.040,  -1.146]
  softmax -> weights, first 6: [  0.761,   0.002,   0.002,   0.002,   0.002,   0.001]   sum of all 197 = 1.0000   max 0.7613   min 0.0005
  the 6 largest weights of the [class] query in head 1 of layer 1:
    token   0  weight 0.7613   [class] itself
    token  14  weight 0.0026   patch  13 = grid row  0, col 13
    token   1  weight 0.0023   patch   0 = grid row  0, col  0
    token 109  weight 0.0022   patch 108 = grid row  7, col 10
    token  13  weight 0.0021   patch  12 = grid row  0, col 12
    token 153  weight 0.0020   patch 152 = grid row 10, col 12
  weight on [class] itself: 0.7613;  on all 196 patches together: 0.2387
  weight of the [class] query on itself in each of the 12 heads of layer 1: 0.76 0.90 0.78 0.11 0.90 0.65 0.79 0.02 0.82 1.00 0.37 0.39
  head 8 spreads the [class] query most (self weight 0.022); its top-5 patches:
    token   0  weight 0.0219   [class] itself
    token   8  weight 0.0146   patch   7 = grid row  0, col  7
    token   3  weight 0.0122   patch   2 = grid row  0, col  2
    token   7  weight 0.0122   patch   6 = grid row  0, col  6
    token   4  weight 0.0120   patch   3 = grid row  0, col  3
```

> [!DEFINITION] Softmax
> A function that turns a list of scores into a list of positive weights that add up to 1: each score $$s_i$$ becomes $$e^{s_i} / \sum_j e^{s_j}$$. Larger scores get larger weights, and the gap grows quickly: a score 5 above the others takes almost all of the weight.

Two things are worth noticing. First, the score of the class token against itself is 5.125 while the others are around −1, and after softmax that one score takes 0.761 of the weight. Head 1 of layer 1 mostly leaves the class token alone. The same is true of most heads in this layer (0.65 to 1.00 on itself); only heads 4, 8, 11 and 12 look outward. Second, when a head does look outward, as head 8 does, no patch gets much: the largest weight is 0.0146, and the five largest all sit in the top row of the picture (the red sofa behind the cats). This is layer 1; there is no "cat detector" yet. [Part 5](part-5-scaling-and-inside-vit.md) measures how far the heads look in every layer and shows that later layers attend to the object.

{{FIG:p2_cls_attention|The class token's 196 patch weights in layer 1, drawn on the 14×14 grid, for two heads of the real model. Head 1 (left) puts 0.76 on the class token itself and at most 0.0026 on any patch. Head 8 (right) puts only 0.022 on itself and spreads its weight, mostly along the top row of the picture. Each heatmap is scaled to its own largest weight.}}

{{FIG:p2_attn_on_grid|Head 8 of layer 1 on top of the patch grid of the picture, with the five strongest patches outlined and numbered. All five are in the top row (patches 7, 2, 6 and 3 of row 0), and the largest weight is 0.0146: with 197 weights that add up to 1, no single patch can be large.}}

The weighted sum, the join of the heads, the output matrix and the residual, still for the class token:

```text
  SA_1 for [class] = sum_j A_0j v_j: (64,), first 6: [ -0.019,   0.029,  -0.053,   0.131,  -0.007,   0.001]
  concat of 12 heads: (197, 768);  times U_msa (768, 768) + bias: (197, 768)
  Eq. 2: z'_1 = MSA + z_0, [class] first 6: [ -0.178,  -0.015,  -0.848,  -0.393,   0.661,   0.162]
```

{{FIG:p2_msa_shapes|The shapes inside multi-head self-attention for ViT-B/16. The normalised input (197×768) goes through 12 heads; in each, q, k and v are 197×64, the attention matrix is 197×197 with rows that sum to 1, and the output is 197×64. The 12 outputs are concatenated to 197×768 and multiplied by U_msa (768×768). Four 768×768 matrices and four biases make 2,362,368 attention parameters per layer.}}

### Equation 3: LayerNorm, MLP, residual

The idea in one sentence: normalise again, push every token through the same small two-layer network, and add the result back.

$$
\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell, \qquad \ell = 1 \dots L
$$

where:

- $$\mathbf{z}'_\ell$$ is the output of Equation 2, $$197 \times 768$$;
- $$\text{LN}$$ is a second LayerNorm with its own $$\gamma, \beta$$;
- $$\text{MLP}(x) = \text{GELU}(x W_1 + b_1)\, W_2 + b_2$$ with $$W_1$$ of $$768 \times 3{,}072$$ and $$W_2$$ of $$3{,}072 \times 768$$, applied to each of the 197 rows separately;
- $$\mathbf{z}_\ell$$ is the output of layer $$\ell$$, $$197 \times 768$$, the input of the next layer.

And GELU itself, number by number:

$$
\text{GELU}(x) = x \cdot \Phi(x), \qquad \Phi(x) = \tfrac{1}{2}\left(1 + \operatorname{erf}\!\left(x/\sqrt{2}\right)\right)
$$

where $$\Phi(x)$$ is the probability that a standard normal random number is below $$x$$: close to 0 for very negative $$x$$, 0.5 at 0, close to 1 for large $$x$$. So GELU keeps a number roughly in proportion to how "positive" it is. The script computes it from `torch.erf` and compares with the library's `F.gelu`:

```text
  GELU(x) = x * Phi(x) at a few values, against torch.nn.functional.gelu:
    x:                 -3.00   -2.00   -1.00   -0.50    0.00    0.50    1.00    2.00    3.00
    Phi(x):           0.0013  0.0228  0.1587  0.3085  0.5000  0.6915  0.8413  0.9772  0.9987
    x*Phi(x):        -0.0040 -0.0455 -0.1587 -0.1543  0.0000  0.3457  0.8413  1.9545  2.9960
    F.gelu(x):       -0.0040 -0.0455 -0.1587 -0.1543  0.0000  0.3457  0.8413  1.9545  2.9960
    relu(x):          0.0000  0.0000  0.0000  0.0000  0.0000  0.5000  1.0000  2.0000  3.0000
```

$$\text{GELU}(-1) = -1 \times 0.1587 = -0.159$$, not 0 as ReLU would give; $$\text{GELU}(2) = 2 \times 0.9772 = 1.955$$, almost the 2 of ReLU. The two agree exactly with the library's function. Now the MLP on the class token in layer 1:

```python
ln2 = lay.layernorm_after(z_prime)                               # second LayerNorm, own gamma and beta
h = ln2 @ lay.mlp.fc1.weight.T + lay.mlp.fc1.bias                # (197, 3072)
mlp = gelu_exact(h) @ lay.mlp.fc2.weight.T + lay.mlp.fc2.bias    # (197, 768)
z1 = mlp + z_prime                                               # Eq. 3
```

```text
  LN(z'_1) [class] first 6:   [ -0.055,  -0.043,  -0.247,   0.003,   0.249,   0.020]
  MLP: LN(z') W_1 + b_1 -> (197, 3072); GELU; W_2 + b_2 -> (197, 768)
  [class] before GELU, first 6: [ -2.016,  -2.504,   0.053,  -1.484,  -0.649,   0.060]
  [class] after GELU, first 6:  [ -0.044,  -0.015,   0.028,  -0.102,  -0.168,   0.031]
  share of the 3072 numbers that are positive before GELU ([class]): 0.071
  Eq. 3: z_1 = MLP + z'_1, [class] first 6: [ -0.212,  -0.046,  -0.840,  -0.748,   0.774,   0.083]
  max |difference|, our z_1 vs the library hidden_states[1]: 6.68e-06
  max |difference|, our head-1 weights vs the library attentions[0] (all 12 heads): 2.68e-06
```

Only 7.1% of the 3,072 hidden numbers of the class token are positive before GELU in this layer; most of the wide hidden layer is "off" for this token, and the off numbers come out small but not zero ($$-2.016 \to -0.044$$). Our $$\mathbf{z}_1$$ matches the library's output of layer 1 to $$6.7 \times 10^{-6}$$, and our attention weights match its weights to $$2.7 \times 10^{-6}$$. So Equations 2 and 3, as written above, are the complete layer.

{{FIG:p2_mlp_gelu|Left: the MLP block of one layer, 768 numbers widened to 3,072 by W₁, passed through GELU and narrowed back to 768 by W₂; 4,722,432 parameters per layer, two thirds of the layer. Right: the GELU curve x·Φ(x) computed in code, with ReLU dashed; GELU(−1) = −0.159 and GELU(2) = 1.955.}}

**All twelve layers.** The same two equations run 12 times, each layer with its own weights. Our loop against the library, layer by layer:

```text
== 5. all 12 layers, our loop vs the library ==
  layer  1: z_1 (197, 768)   max |difference| vs hidden_states[1]: 6.68e-06   (largest value in z_1:    21.2, so relative 3.1e-07)
  layer  2: z_2 (197, 768)   max |difference| vs hidden_states[2]: 8.34e-06   (largest value in z_2:    23.7, so relative 3.5e-07)
  layer  3: z_3 (197, 768)   max |difference| vs hidden_states[3]: 9.54e-06   (largest value in z_3:    24.1, so relative 4.0e-07)
  layer  4: z_4 (197, 768)   max |difference| vs hidden_states[4]: 8.39e-05   (largest value in z_4:    56.0, so relative 1.5e-06)
  layer  5: z_5 (197, 768)   max |difference| vs hidden_states[5]: 9.16e-04   (largest value in z_5:   336.9, so relative 2.7e-06)
  layer  6: z_6 (197, 768)   max |difference| vs hidden_states[6]: 1.50e-03   (largest value in z_6:   989.2, so relative 1.5e-06)
  ... (layers 7 to 11: 1.65e-03 to 1.77e-03, relative 1.0e-06 to 1.1e-06)
  layer 12: z_12 (197, 768)   max |difference| vs hidden_states[12]: 1.83e-03   (largest value in z_12:  1699.6, so relative 1.1e-06)
  the final logits from our z_12 vs the library differ by at most 2.86e-06 (see step 6)
```

An honesty note on the numbers in the middle column: the absolute difference grows from $$7 \times 10^{-6}$$ in layer 1 to $$1.8 \times 10^{-3}$$ in layer 12. That is not a bug in the equations. The residual stream of a trained ViT contains a few enormous values (the largest number in $$\mathbf{z}_{12}$$ is about 1,700, against 21 in $$\mathbf{z}_1$$), and 32-bit floating point keeps about seven significant digits, so the rounding noise grows with the values. Relative to the largest value the difference stays at about $$10^{-6}$$ in every layer, and the final class scores agree to $$3 \times 10^{-6}$$.

### Equation 4: the image representation

The idea in one sentence: after the last layer, normalise the class token one more time; that vector is the picture.

$$
\mathbf{y} = \text{LN}(\mathbf{z}_L^0)
$$

where $$\mathbf{z}_L^0$$ is row 0 (the class token) of the last layer's output, 768 numbers, and $$\text{LN}$$ is a final LayerNorm with its own $$\gamma, \beta$$. $$\mathbf{y}$$, also 768 numbers, is the image representation. The 196 patch rows of $$\mathbf{z}_L$$ are computed and then ignored. The head is not in the equation; in the fine-tuned model it is $$\text{logits} = \mathbf{y}\,W_{\text{head}} + b_{\text{head}}$$ with $$W_{\text{head}}$$ of $$768 \times 1000$$:

```python
y = model.vit.layernorm(zL[0])                                 # Eq. 4, 768 numbers
logits = y @ model.classifier.weight.T + model.classifier.bias # (1000,)
probs = torch.softmax(logits, -1)
```

```text
== 6. Eq. 4: y = LN(z_L^0), then the classification head ==
  z_L^0 (the [class] token after 12 layers), first 6: [  2.313,   5.512,  11.788,   0.577,   6.548,  -2.913]   mean 0.073 var 34.490
  y = LN(z_L^0), first 6: [  0.294,   0.835,   1.904,   0.081,   1.039,  -0.518]   max |difference| vs the library: 3.81e-06
  head of the fine-tuned checkpoint: Linear(in_features=768, out_features=1000, bias=True)   (one linear layer, as Section 3.1 says for fine-tuning)
  logits = y W_head + b: (1000,)   max |difference| vs model.logits: 2.86e-06
  top-5 ImageNet classes:
     0.937  class 285  Egyptian cat
     0.038  class 281  tabby, tabby cat
     0.014  class 282  tiger cat
     0.003  class 287  lynx, catamount
     0.001  class 284  Siamese cat, Siamese
```

The model says "Egyptian cat" with probability 0.937, and the next four guesses are all cats. Everything from the 50,176 pixels to this answer was computed above with nothing but matrix multiplications, LayerNorm, softmax and GELU, and matched the library at every step.

## Inductive bias {§3.1}

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 3.1, Inductive bias · page 4
> [![The Inductive bias paragraph: we note that Vision Transformer has much less image-specific inductive bias than CNNs. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model. In ViT, only MLP layers are local and translationally equivariant, while the self-attention layers are global. The two-dimensional neighborhood structure is used very sparingly: in the beginning of the model by cutting the image into patches and at fine-tuning time for adjusting the position embeddings for images of different resolution. Other than that, the position embeddings at initialization time carry no information about the 2D positions of the patches and all spatial relations between the patches have to be learned from scratch](/img/papers/vit/p2-inductive.png)](/img/papers/vit/p2-inductive.png)
>
> **Context:** the first of two named paragraphs that close Section 3.1. It restates the thesis of the introduction in the vocabulary of the model.
>
> **What it says:** ViT "has much less image-specific inductive bias than CNNs". In a CNN, "locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer". In ViT "only MLP layers are local and translationally equivariant, while the self-attention layers are global". The 2D grid is used "very sparingly": when "cutting the image into patches" and at fine-tuning time when "adjusting the position embeddings for images of different resolution". The position embeddings "at initialization time carry no information about the 2D positions of the patches", so "all spatial relations between the patches have to be learned from scratch".
>
> **Why it matters:** this paragraph is the mechanism behind the paper's main finding. Less built-in knowledge means more must be learned from data, which is why ViT loses on small datasets and wins on huge ones (Part 4). It also tells you exactly where to look for the learned knowledge: in the position embeddings and the attention patterns (Part 5).

> [!DEFINITION] Inductive bias
> The assumptions a model has before it sees any data. A CNN assumes that nearby pixels matter most and that a cat is a cat wherever it appears. ViT assumes almost nothing about pictures. Part 1 has the long version.

> [!DEFINITION] Translation equivariance
> If the input shifts, the output shifts the same way and is otherwise unchanged. A convolution has it because the same filter is used at every position. In ViT the MLP has it in a weak sense (the same weights act on every token), but shifting a picture by less than a patch changes every patch's content, so the model as a whole does not.

The paragraph's two claims about ViT are easy to confirm from what we computed. The MLP is "local and translationally equivariant" because it acts on one token at a time with the same $$W_1, W_2$$ for every token; the attention is "global" because every row of the $$197 \times 197$$ matrix had a non-zero weight for every column (the smallest weight in the class token's row was 0.0005, not 0). And the position embeddings carry no 2D information at the start because they are 197 independent learnable rows; nothing in Equation 1 says that row 15 is below row 1.

{{FIG:p2_inductive_bias|The two places where the 2D structure of the picture enters ViT, highlighted: cutting into 16×16 patches at the start, and the 2D interpolation of position embeddings when fine-tuning at a new resolution (Part 3). The embedding and the encoder in between carry no built-in knowledge of the grid; the position rows start random and the global attention treats all patches alike.}}

## Hybrid architecture {§3.1}

> [!PAPER] Dosovitskiy et al. (2020), ViT · Section 3.1, Hybrid Architecture · page 4
> [![The Hybrid Architecture paragraph: as an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN (LeCun et al., 1989). In this hybrid model, the patch embedding projection E (Eq. 1) is applied to patches extracted from a CNN feature map. As a special case, the patches can have spatial size 1x1, which means that the input sequence is obtained by simply flattening the spatial dimensions of the feature map and projecting to the Transformer dimension. The classification input embedding and position embeddings are added as described above](/img/papers/vit/p2-hybrid.png)](/img/papers/vit/p2-hybrid.png)
>
> **Context:** the last paragraph of Section 3.1, which defines the third model family of the paper (ResNet, ViT, hybrid).
>
> **What it says:** "the input sequence can be formed from feature maps of a CNN". The projection $$\mathbf{E}$$ "is applied to patches extracted from a CNN feature map". In the special case of "spatial size 1x1" patches, the sequence is "obtained by simply flattening the spatial dimensions of the feature map and projecting to the Transformer dimension". The class embedding and position embeddings "are added as described above".
>
> **Why it matters:** this is the "CNN + attention" family of Section 2, built inside ViT's own framework so that the three families can be compared fairly. Part 4 shows the hybrid winning at small compute budgets and the pure ViT catching up and passing it at large ones.

> [!DEFINITION] CNN stage
> A group of convolutional layers that work at one resolution; between stages the grid is halved. A ResNet-50 has four stages after its first layer, with feature maps of 56×56, 28×28, 14×14 and 7×7 for a 224×224 input. The hybrids in the paper take either the 7×7 output of stage 4 or the 14×14 output of a stage 3 that is extended to replace stage 4 (Section 4.1, Part 3).

> [!DEFINITION] Hybrid
> A ViT whose tokens are positions of a CNN feature map instead of raw pixel patches. The CNN does some of the seeing; the Transformer does the rest. The paper writes it "R50+ViT-B/16".

In shapes: a ResNet turns the $$224 \times 224 \times 3$$ picture into a $$14 \times 14 \times 1{,}024$$ feature map (with the extended-stage-3 option the paper pairs with ViT-B/16). Reading each position as a $$1 \times 1$$ "patch" gives $$N = 196$$ tokens of $$1{,}024$$ numbers, so $$\mathbf{E}$$ becomes $$1{,}024 \times 768$$ and everything from Equation 1 on is unchanged. The sequence length is the same 196 as ViT-B/16, which is why the paper pairs them.

{{FIG:p2_hybrid|The hybrid architecture. The picture goes through ResNet stages; the 14×14×1024 feature map is read as 196 one-by-one patches of 1,024 numbers; E (now 1,024 × 768) projects them; the class token and position embeddings are added; and the same encoder follows. Everything after E is identical to the plain ViT.}}

## Table 1, as far as this part needs it {§4.1}

The next part reads the experimental setup in full. We need one table from it now, because every number above (768, 12, 3,072) came from it:

> [!PAPER] Dosovitskiy et al. (2020), ViT · Table 1 · page 5
> [![Table 1, details of Vision Transformer model variants: ViT-Base has 12 layers, hidden size D 768, MLP size 3072, 12 heads, 86M parameters; ViT-Large has 24 layers, 1024, 4096, 16 heads, 307M; ViT-Huge has 32 layers, 1280, 5120, 16 heads, 632M](/img/papers/vit/p2-table1.png)](/img/papers/vit/p2-table1.png)
>
> **Context:** Section 4.1, Model Variants. The paper says the Base and Large sizes "are directly adopted from BERT".
>
> **What it says:** **ViT-Base**: 12 layers, $$D = 768$$, MLP 3,072, 12 heads, 86M parameters. **ViT-Large**: 24 layers, $$D = 1{,}024$$, MLP 4,096, 16 heads, 307M. **ViT-Huge**: 32 layers, $$D = 1{,}280$$, MLP 5,120, 16 heads, 632M. The MLP size is always $$4D$$ and the head size always $$D/\text{heads} = 64$$ for Base and Large, 80 for Huge.
>
> **Why it matters:** with these numbers and the equations above we can count every parameter of every model, which we do next. [Part 3](part-3-fine-tuning-and-setup.md) reads the table together with the patch-size notation and the baselines.

{{FIG:p2_table1_blocks|Table 1 as three stacks of blocks, one block per layer, block width proportional to D. ViT-Base: 12 layers of 768. ViT-Large: 24 of 1,024. ViT-Huge: 32 of 1,280. Under each, the paper's parameter count and ours without a classification head.}}

### Where the 86 million parameters come from

> [!DEFINITION] Parameter count
> The number of learned values in a model: every entry of every weight matrix, bias vector, embedding table and LayerNorm scale. "86M" means about 86 million such numbers.

Every piece of the model above has a known shape, so the count is a formula. For hidden size $$D$$, $$L$$ layers, MLP size $$4D$$, patch size $$P$$, $$C$$ channels, $$N$$ patches and $$K$$ classes:

$$
\underbrace{P^2 C \cdot D + D}_{\text{patch embedding } \mathbf{E} \text{ + bias}} + \underbrace{D}_{\mathbf{x}_{\text{class}}} + \underbrace{(N+1)\,D}_{\mathbf{E}_{pos}} + L\,\big(\underbrace{4(D^2 + D)}_{\text{q, k, v, } U_{msa}} + \underbrace{8D^2 + 5D}_{\text{MLP}} + \underbrace{4D}_{\text{2 LayerNorms}}\big) + \underbrace{2D}_{\text{final LN}} + \underbrace{D K + K}_{\text{head}}
$$

The per-layer part simplifies to $$12D^2 + 13D$$. Each term, for ViT-B/16 at 224 with $$K = 1{,}000$$:

- patch embedding: $$16 \cdot 16 \cdot 3 \cdot 768 + 768 = 590{,}592$$;
- class token: $$768$$; positions: $$197 \times 768 = 151{,}296$$;
- attention per layer: $$4 \times (768^2 + 768) = 2{,}362{,}368$$ (four $$768 \times 768$$ matrices with biases);
- MLP per layer: $$768 \cdot 3{,}072 + 3{,}072 + 3{,}072 \cdot 768 + 768 = 4{,}722{,}432$$;
- two LayerNorms per layer: $$4 \times 768 = 3{,}072$$; so one layer is $$7{,}087{,}872$$ and twelve are $$85{,}054{,}464$$;
- final LayerNorm: $$1{,}536$$; head: $$768 \times 1{,}000 + 1{,}000 = 769{,}000$$.

The script evaluates the formula and counts the real model's parameters. For ViT-L/16 and ViT-H/14 it builds the model from its configuration on PyTorch's "meta" device, which creates every tensor's shape without allocating any memory, so no weights are needed:

```python
def formula(D, L, P, C, N, K, mlp=None):
    mlp = mlp or 4 * D
    return (P * P * C * D + D) + D + (N + 1) * D + L * (4 * (D * D + D) + D * mlp + mlp + mlp * D + D + 4 * D) + 2 * D + (D * K + K)

c = ViTConfig(hidden_size=1280, num_hidden_layers=32, num_attention_heads=16, intermediate_size=5120, patch_size=14, num_labels=1000)
with torch.device('meta'):                       # shapes only, no memory for weights
    huge = ViTForImageClassification(c)
print(sum(p.numel() for p in huge.parameters()))
```

```text
== 7. counting the parameters ==
  formula: (P^2 C D + D) + D + (N+1) D + L (12 D^2 + 13 D) + 2 D + (D K + K)
           patch emb       cls  position  L layers            LN   head
  ViT-B/16 at 224, K = 1000:
    patch embedding      590,592   [class]    768   positions   151,296
    one layer: attention 4(D^2 + D) = 2,362,368   MLP 8D^2 + 5D = 4,722,432   2 LayerNorms 4D = 3,072   -> 7,087,872   x 12 = 85,054,464
    final LayerNorm 1,536   head D K + K = 769,000
    formula total 86,567,656   counted in model.parameters() 86,567,656   same: True
    without the 1000-class head: 85,798,656   (paper, Table 1: 86M)
    where they sit: attention 28.35M (32.7%)   mlp 56.67M (65.5%)   embeddings 0.74M (0.9%)   layernorms 0.04M (0.0%)   head 0.77M (0.9%)
  ViT-L/16 at 224 (meta device, no weights): D=1024 L=24 heads=16 MLP=4096 N=196
    formula 304,326,632   counted 304,326,632   same: True
    with the 1000-class head 304.3M, without it 303.3M   (paper: 307M)
  ViT-H/14 at 224 (meta device, no weights): D=1280 L=32 heads=16 MLP=5120 N=256
    formula 632,045,800   counted 632,045,800   same: True
    with the 1000-class head 632.0M, without it 630.8M   (paper: 632M)
```

The formula matches the counted parameters exactly for all three sizes: **86,567,656** for ViT-B/16 with its 1,000-class head, **304,326,632** for ViT-L/16 and **632,045,800** for ViT-H/14. Compared with Table 1: ViT-Base rounds to 86M with or without the head (85.8M or 86.6M). ViT-Huge rounds to 632M *with* a 1,000-class head (632.0M; without it, 630.8M), so Table 1's figure for Huge appears to include a head of that size. ViT-Large is the odd one: we get 303.3M or 304.3M, and the paper says 307M. An honesty note: the 3M gap is real and we cannot explain it from the model's configuration (ViT-L/16 at 384 pixels has 577 position rows instead of 197, which adds only 0.4M; the 21,843-class ImageNet-21k head would add 22M, far too much). The released ViT-L/16 checkpoints have 304M parameters, so we trust the count and flag the table.

{{FIG:p2_params|Where the 86,567,656 parameters of ViT-B/16 sit. The MLP blocks hold 56.67 million (65.5%), attention 28.35 million (32.7%), the embeddings (E, the class token and the 197 position rows) only 0.74 million, the head 0.77 million and all 25 LayerNorms 0.04 million.}}

Two things stand out. The embeddings are tiny: 0.9% of the model, against about a fifth for BERT-base, because BERT carries a 30,522-row vocabulary table and ViT carries one $$768 \times 768$$ matrix. And two thirds of ViT is MLP, one third attention; the ratio $$8D^2 : 4D^2$$ is fixed by the $$4D$$ MLP width and is the same for every Transformer with that width.

## Appendix A: the attention equations {§A}

The paper keeps the attention formulas out of the main text and puts them in Appendix A. They are the inside of the "MSA" box above.

> [!PAPER] Dosovitskiy et al. (2020), ViT · Appendix A · page 13
> [![Appendix A, Multihead Self-Attention: standard qkv self-attention is a popular building block. For each element in an input sequence z in R N by D, we compute a weighted sum over all values v in the sequence. The attention weights A i j are based on the pairwise similarity between two elements of the sequence and their respective query q i and key k j representations. Equation 5: q, k, v equals z U qkv, with U qkv in R D by 3 D h. Equation 6: A equals softmax of q k transpose over square root of D h, with A in R N by N. Equation 7: SA of z equals A v](/img/papers/vit/p2-appendix-a.png)](/img/papers/vit/p2-appendix-a.png)
>
> **Context:** the first half of Appendix A, one attention head.
>
> **What it says:** for each element of the input $$\mathbf{z} \in \mathbb{R}^{N \times D}$$, "we compute a weighted sum over all values $$\mathbf{v}$$ in the sequence", with weights $$A_{ij}$$ "based on the pairwise similarity between two elements of the sequence and their respective query $$\mathbf{q}^i$$ and key $$\mathbf{k}^j$$ representations". Then Equations 5, 6 and 7.
>
> **Why it matters:** three lines define the one operation that lets tokens interact. Everything else in the model works on one token at a time.

> [!DEFINITION] Query, key and value
> Three vectors computed from each token by three linear projections. A token's **query** is what it is looking for; a token's **key** is what it offers to be matched against; its **value** is what it hands over when it is attended to. The weight from token $$i$$ to token $$j$$ is the similarity (dot product) of $$i$$'s query with $$j$$'s key.

> [!DEFINITION] Head
> One complete self-attention computation with its own $$U_{qkv}$$, working on a slice of $$D_h$$ numbers (64 in ViT-B). A layer runs $$k$$ heads in parallel so that different heads can learn different ways of relating tokens.

### Equations 5 to 7: one head

The idea: from the same input compute queries, keys and values; score every pair; turn scores into weights; average the values with those weights.

$$
[\mathbf{q}, \mathbf{k}, \mathbf{v}] = \mathbf{z}\,\mathbf{U}_{qkv}, \qquad \mathbf{U}_{qkv} \in \mathbb{R}^{D \times 3D_h}
$$

$$
A = \text{softmax}\!\left(\mathbf{q}\mathbf{k}^\top / \sqrt{D_h}\right), \qquad A \in \mathbb{R}^{N \times N}
$$

$$
\text{SA}(\mathbf{z}) = A\,\mathbf{v}
$$

where:

- $$\mathbf{z}$$ is the (normalised) input, $$N \times D$$: 197 × 768 in ViT-B;
- $$\mathbf{U}_{qkv}$$ is one matrix of $$D \times 3D_h$$ that produces all three at once; $$\mathbf{q}, \mathbf{k}, \mathbf{v}$$ are each $$N \times D_h$$ (197 × 64). The released model stores it as three separate $$768 \times 768$$ matrices covering all 12 heads, which is the same numbers arranged differently, plus biases the equation omits;
- $$\mathbf{q}\mathbf{k}^\top$$ is $$N \times N$$: entry $$(i, j)$$ is the dot product of query $$i$$ with key $$j$$;
- $$\sqrt{D_h}$$ is 8 for $$D_h = 64$$; dividing keeps the scores from growing with the head size, so softmax does not saturate ([BERT Part 2](/papers/bert/part-2-architecture-and-input/) measures why);
- softmax is applied to each row, so row $$i$$ holds the 197 weights of token $$i$$ and they add up to 1;
- $$A\,\mathbf{v}$$ is $$N \times D_h$$: row $$i$$ is the weighted average of all value rows, with token $$i$$'s weights.

This is Vaswani et al.'s Equation 1 with different letters:

> [!PAPER] Vaswani et al. (2017), Attention Is All You Need · Section 3.2.1, Equation (1)
> [![Section 3.2.1 of Attention Is All You Need: in practice we compute the attention function on a set of queries simultaneously, packed together into a matrix Q; the keys and values are also packed into matrices K and V; Attention of Q, K, V equals softmax of Q K transpose over square root of d k, times V](/img/papers/vit/p2-ext-vaswani-eq1.png)](/img/papers/vit/p2-ext-vaswani-eq1.png)
>
> **Context:** scaled dot-product attention in the original Transformer paper.
>
> **What it says:** queries are "packed together into a matrix $$Q$$", keys and values into $$K$$ and $$V$$, and $$\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d_k})\,V$$.
>
> **Why it matters:** ViT's $$\mathbf{q}, \mathbf{k}, \mathbf{v}, D_h$$ are $$Q, K, V, d_k$$. The paper changed nothing here, as Section 3 promised.

**A worked example small enough to check with a pencil.** Four tokens, $$D = 6$$, two heads, so $$D_h = 3$$, with small whole numbers chosen so the sums stay short ([`vit_part2_attn.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part2_attn.py)). Head 1:

```text
== Eq. 5 to 8 on a tiny example: N = 4 tokens, D = 6, k = 2 heads, D_h = D/k = 3 ==
  z (the input, one row per token)  shape (4, 6)
       t0      0      1      0      0      1      0
       t1      0      1      1      1      2      1
       t2      1      1      1      0      0      0
       t3      2      2      2      1      2      1
-- head 1 --
  U_qkv (head 1): D x 3 D_h  shape (6, 9)
               0      1      0      0      0      0     -1      1      0
               1     -1      1      0      1      0      0      0      1
               0      1      0     -1      1      1     -1      1      0
               1      1     -1     -1      0     -1     -1     -1      0
               1      0      1      1     -1      0      0     -1     -1
               0     -1     -1      1      0     -1      0      0     -1
  [q, k, v] = z U_qkv  (the first 3 columns are q, the next 3 k, the last 3 v)  shape (4, 9)
       t0      2     -1      2      1      0      0      0     -1      0
       t1      4      0      1      1      0     -1     -2     -2     -2
       t2      1      1      1     -1      2      1     -2      2      1
       t3      5      2      2      0      2      0     -5      1     -1
  q k^T / sqrt(D_h) = q k^T / 1.732  shape (4, 4)   (row i = query of token i, column j = key of token j)
       t0  1.155  0.000 -1.155 -1.155
       t1  2.309  1.732 -1.732  0.000
       t2  0.577  0.000  1.155  1.155
       t3  2.887  1.732  0.577  2.309
  A = softmax(row by row)  shape (4, 4)
       t0  0.661  0.208  0.066  0.066
       t1  0.596  0.335  0.010  0.059
       t2  0.195  0.110  0.348  0.348
       t3  0.506  0.160  0.050  0.284
    row sums: [1.0, 1.0, 1.0, 1.0]
  SA(z) = A v  (row i = weighted average of the value rows, weights = row i of A)  shape (4, 3)
       t0 -0.876 -0.880 -0.416
       t1 -0.986 -1.185 -0.718
       t2 -2.653  0.629 -0.219
       t3 -3.014  0.002 -0.995
```

Follow one number through. Row t0 of $$\mathbf{z}$$ is $$(0, 1, 0, 0, 1, 0)$$, so its query is the sum of rows 2 and 5 of $$\mathbf{U}_{qkv}$$'s first three columns: $$(1, -1, 1) + (1, 0, 1) = (2, -1, 2)$$, as printed. Its score against its own key $$(1, 0, 0)$$ is $$2 \cdot 1 + (-1) \cdot 0 + 2 \cdot 0 = 2$$; divided by $$\sqrt{3} = 1.732$$ that is 1.155. Softmax over the row $$(1.155, 0, -1.155, -1.155)$$:

$$
\frac{e^{1.155}}{e^{1.155} + e^{0} + e^{-1.155} + e^{-1.155}} = \frac{3.173}{3.173 + 1.000 + 0.315 + 0.315} = \frac{3.173}{4.803} = 0.661
$$

which the script checks the same way (`A[t0, t0] = 3.173 / 4.803 = 0.661`). And the first number of t0's output is $$0.661 \cdot 0 + 0.208 \cdot (-2) + 0.066 \cdot (-2) + 0.066 \cdot (-5) = -0.876$$: the weighted average of the first column of $$\mathbf{v}$$. Token t0 listens mostly to itself (0.661) and to t1 (0.208).

{{FIG:p2_tiny_attention|Head 1 of the tiny example as a picture. Left: the 4×4 scaled scores. Middle: softmax turns each row into weights that add up to 1. Right: the four value rows, and below them SA(z), each row a weighted average of the value rows. Orange cells are negative numbers.}}

### Equation 8: many heads

> [!PAPER] Dosovitskiy et al. (2020), ViT · Appendix A · page 13
> [![The second half of Appendix A: multihead self-attention (MSA) is an extension of SA in which we run k self-attention operations, called heads, in parallel, and project their concatenated outputs. To keep compute and number of parameters constant when changing k, D h (Eq. 5) is typically set to D over k. Equation 8: MSA of z equals the concatenation of SA 1 of z to SA k of z, times U msa, with U msa in R k D h by D](/img/papers/vit/p2-appendix-msa.png)](/img/papers/vit/p2-appendix-msa.png)
>
> **Context:** the second half of Appendix A.
>
> **What it says:** MSA runs "$$k$$ self-attention operations, called 'heads', in parallel", and projects "their concatenated outputs". "To keep compute and number of parameters constant when changing $$k$$, $$D_h$$ (Eq. 5) is typically set to $$D/k$$." Then Equation 8.
>
> **Why it matters:** the rule $$D_h = D/k$$ is why ViT-B has 12 heads of 64: $$768/12 = 64$$. It means that the choice of head count does not change the size of the model.

$$
\text{MSA}(\mathbf{z}) = [\text{SA}_1(\mathbf{z});\ \text{SA}_2(\mathbf{z});\ \cdots;\ \text{SA}_k(\mathbf{z})]\ \mathbf{U}_{msa}, \qquad \mathbf{U}_{msa} \in \mathbb{R}^{k \cdot D_h \times D}
$$

where:

- $$\text{SA}_h(\mathbf{z})$$ is head $$h$$'s output, $$N \times D_h$$, each head with its own $$\mathbf{U}_{qkv}$$;
- the brackets place the $$k$$ outputs side by side: $$N \times k D_h$$, which is $$N \times D$$ when $$D_h = D/k$$;
- $$\mathbf{U}_{msa}$$ mixes the heads, $$k D_h \times D$$ (768 × 768 in ViT-B; the released model adds a bias);
- the result is $$N \times D$$, the same shape as the input, ready for the residual addition of Equation 2.

Vaswani et al. draw it like this:

> [!PAPER] Vaswani et al. (2017), Attention Is All You Need · Figure 2
> [![Figure 2 of Attention Is All You Need. Left: scaled dot-product attention as a flow of MatMul, Scale, optional Mask, SoftMax, MatMul over Q, K, V. Right: multi-head attention, where V, K and Q each pass through h parallel linear layers into h scaled dot-product attention blocks whose outputs are concatenated and passed through a final linear layer](/img/papers/vit/p2-ext-vaswani-fig2.png)](/img/papers/vit/p2-ext-vaswani-fig2.png)
>
> **Context:** the original diagram of one head (left) and of multi-head attention (right).
>
> **What it says:** multi-head attention "consists of several attention layers running in parallel". Each head has its own linear projections of $$V, K, Q$$; the heads are concatenated and go through one more linear layer.
>
> **Why it matters:** the right half is Equation 8 as a picture. The "Mask (opt.)" box in the left half is the one piece ViT never uses: every patch may see every patch.

The tiny example's second head and the join:

```text
-- Eq. 8 --
  [SA_1(z); SA_2(z)]  (the two heads side by side: 4 x 6)  shape (4, 6)
       t0 -0.876 -0.880 -0.416  0.727  0.760  0.367
       t1 -0.986 -1.185 -0.718  0.673  0.910  0.288
       t2 -2.653  0.629 -0.219  0.950  0.210  0.615
       t3 -3.014  0.002 -0.995  0.884  0.769  0.653
  U_msa: k D_h x D = 6 x 6  shape (6, 6)
               0     -1      0      0      0      0
               0      1      0     -1     -1     -1
              -1     -1      0      1      1      0
               1      0      0      0     -1     -1
               0      0      0     -1     -1     -1
               1     -1      0      1      0      1
  MSA(z) = [SA_1(z); SA_2(z)] U_msa  shape (4, 6)
       t0  1.510  0.045  0.000  0.070 -1.023 -0.240
       t1  1.679  0.231  0.000 -0.155 -1.116 -0.110
       t2  1.784  2.885  0.000 -0.442 -2.007 -1.173
       t3  2.091  1.299  0.000 -0.228 -1.766 -0.560
```

The first three columns of the concatenation are head 1's output from above; the last three are head 2's. Multiplying by the $$6 \times 6$$ matrix $$\mathbf{U}_{msa}$$ gives four rows of six numbers, the same shape as the input $$\mathbf{z}$$. (The third column of our random $$\mathbf{U}_{msa}$$ happens to be all zeros, so the third output column is zero; a trained matrix would not do that.) In ViT-B the same picture has 197 rows, 12 heads of 64, and a $$768 \times 768$$ $$\mathbf{U}_{msa}$$; the shapes figure above shows it.

> [!TAKEAWAYS] Key takeaways
> - Section 2 lists five earlier ways of putting attention on pictures (local, sparse and axial attention, 2×2 patches, CNN + attention, iGPT on pixels). All fight the **quadratic cost** by changing attention. ViT changes the *sequence* instead: **196 patches** rather than 50,176 pixels, and keeps full attention and the standard encoder.
> - The closest prior work is **Cordonnier et al. (2020)**: 2×2 patches with full attention, on 32×32 images. ViT's claim to novelty is scale: 16×16 patches, medium-resolution images, and 14M to 300M training images.
> - A 224×224×3 picture becomes $$\mathbf{x}_p$$ of **196 × 768** ($$N = HW/P^2$$); one matrix $$\mathbf{E}$$ (768 × 768) projects every patch. The library's 16×16 stride-16 convolution is exactly this linear layer (difference $$5 \times 10^{-6}$$).
> - **Eq. 1**: prepend the learned **[class]** token and add 197 learned **position** rows; our $$\mathbf{z}_0$$ matched the library to $$5 \times 10^{-6}$$. Without positions, attention is permutation equivariant: shuffle the patches and the outputs shuffle identically.
> - **Eq. 2 and 3**: pre-norm LayerNorm, 12-head attention, residual; LayerNorm, MLP 768 → 3,072 → 768 with GELU, residual. Our hand-written layer 1 matched to $$6.7 \times 10^{-6}$$, all 12 layers to a relative $$10^{-6}$$, the final logits to $$2.9 \times 10^{-6}$$.
> - In layer 1 the class token mostly attends to itself (0.76 in head 1); the most outward-looking head puts at most 0.0146 on any patch. Attention to the object comes later (Part 5).
> - **Eq. 4**: $$\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$$, then one linear layer in the fine-tuned checkpoint (768 × 1000) versus a dense + tanh pooler in the pre-trained one. On our picture: **Egyptian cat, 0.937**.
> - Parameter count by formula $$= $$ real count, exactly: **86,567,656** (ViT-B/16 with head), **304,326,632** (ViT-L/16), **632,045,800** (ViT-H/14). Base and Huge match Table 1's 86M and 632M; Large gives 304M against the table's 307M. Two thirds of the parameters are MLP, one third attention, under 1% embeddings.
> - **Appendix A**: $$[\mathbf{q}, \mathbf{k}, \mathbf{v}] = \mathbf{z}\mathbf{U}_{qkv}$$, $$A = \text{softmax}(\mathbf{q}\mathbf{k}^\top/\sqrt{D_h})$$, $$\text{SA} = A\mathbf{v}$$, heads concatenated and multiplied by $$\mathbf{U}_{msa}$$, with $$D_h = D/k = 64$$. Unchanged from Vaswani et al. (2017).

**Next, in Part 3:** how a pre-trained ViT is adapted to a new task. Section 3.2 removes the head, attaches a zero-initialised $$D \times K$$ layer and fine-tunes at a *higher* resolution, which means more patches than the position table has rows and a 2D interpolation of the position embeddings, worked by hand. Then Section 4.1: the datasets (ImageNet, ImageNet-21k, JFT-300M), Table 1 read properly, the ResNet baselines and the hybrids, and the training recipe of Tables 3 and 4. Continue to [Part 3](part-3-fine-tuning-and-setup.md).

<details>
<summary>Run it yourself</summary>

Every number in this part comes from two scripts in [`code/papers/vit/`](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vit). `vit_part2_math.py` downloads `google/vit-base-patch16-224` (about 350 MB) and `google/vit-base-patch16-224-in21k`, loads one picture from the `huggingface/cats-image` dataset, and recomputes every step of Section 3.1 and Appendix A by hand; ViT-L/16 and ViT-H/14 are built on the meta device from their configurations, so no large weights are needed. It runs on a laptop CPU in about a minute. `vit_part2_attn.py` is the four-token example and runs in a second.

```bash
pip install torch transformers datasets pillow
python vit_part2_math.py      # prints everything below, writes results/part2_math.json
python vit_part2_attn.py      # the tiny Eq. 5 to 8 example, writes results/part2_attn.json
```

<figure><img src="/img/papers/vit/part2_math-run.png" alt="Terminal output of vit_part2_math.py: the picture and model configuration, the 196 patches and the first numbers of patch 0, the conv-equals-linear check, Equation 1 with the class token and position numbers, LayerNorm by hand, the attention of head 1 and head 8 of layer 1 for the class token, GELU values, the twelve layers compared with the library, Equation 4 and the top-5 classes, the head and pooler check and the parameter counts" loading="lazy" /><figcaption>The real output of vit_part2_math.py (first 60 lines).</figcaption></figure>

<figure><img src="/img/papers/vit/part2_attn-run.png" alt="Terminal output of vit_part2_attn.py: the input z, U qkv, q k v, the scores, the softmax weights and SA of z for two heads, the concatenation and U msa, the hand check of one softmax entry and the permutation equivariance test" loading="lazy" /><figcaption>The real output of vit_part2_attn.py (first 60 lines).</figcaption></figure>

</details>

## References

**The ViT paper**

1. A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby. [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929) (ViT). ICLR 2021.
2. Google Research. [vision_transformer](https://github.com/google-research/vision_transformer): the official code and pre-trained checkpoints.
3. Hugging Face model cards [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) (fine-tuned on ImageNet) and [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) (pre-trained only), the checkpoints used in this part.

**Papers the ViT paper cites in this part**

4. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin. [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (the Transformer). NeurIPS 2017.
5. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805). NAACL 2019.
6. A. Radford, K. Narasimhan, T. Salimans, I. Sutskever. [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT). OpenAI, 2018.
7. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2). OpenAI, 2019.
8. T. B. Brown et al. [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165) (GPT-3). NeurIPS 2020.
9. N. Parmar, A. Vaswani, J. Uszkoreit, Ł. Kaiser, N. Shazeer, A. Ku, D. Tran. [*Image Transformer*](https://arxiv.org/abs/1802.05751) (local self-attention). ICML 2018.
10. H. Hu, Z. Zhang, Z. Xie, S. Lin. [*Local Relation Networks for Image Recognition*](https://arxiv.org/abs/1904.11491). ICCV 2019.
11. P. Ramachandran, N. Parmar, A. Vaswani, I. Bello, A. Levskaya, J. Shlens. [*Stand-Alone Self-Attention in Vision Models*](https://arxiv.org/abs/1906.05909). NeurIPS 2019.
12. H. Zhao, J. Jia, V. Koltun. [*Exploring Self-attention for Image Recognition*](https://arxiv.org/abs/2004.13621). CVPR 2020.
13. R. Child, S. Gray, A. Radford, I. Sutskever. [*Generating Long Sequences with Sparse Transformers*](https://arxiv.org/abs/1904.10509). arXiv 2019.
14. D. Weissenborn, O. Täckström, J. Uszkoreit. [*Scaling Autoregressive Video Models*](https://arxiv.org/abs/1906.02634) (attention in blocks). ICLR 2020.
15. J. Ho, N. Kalchbrenner, D. Weissenborn, T. Salimans. [*Axial Attention in Multidimensional Transformers*](https://arxiv.org/abs/1912.12180). arXiv 2019.
16. H. Wang, Y. Zhu, B. Green, H. Adam, A. Yuille, L.-C. Chen. [*Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation*](https://arxiv.org/abs/2003.07853). ECCV 2020.
17. J.-B. Cordonnier, A. Loukas, M. Jaggi. [*On the Relationship between Self-Attention and Convolutional Layers*](https://arxiv.org/abs/1911.03584) (the 2×2 patch model). ICLR 2020.
18. I. Bello, B. Zoph, A. Vaswani, J. Shlens, Q. V. Le. [*Attention Augmented Convolutional Networks*](https://arxiv.org/abs/1904.09925). ICCV 2019.
19. H. Hu, J. Gu, Z. Zhang, J. Dai, Y. Wei. [*Relation Networks for Object Detection*](https://arxiv.org/abs/1711.11575). CVPR 2018.
20. N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, S. Zagoruyko. [*End-to-End Object Detection with Transformers*](https://arxiv.org/abs/2005.12872) (DETR). ECCV 2020.
21. X. Wang, R. Girshick, A. Gupta, K. He. [*Non-local Neural Networks*](https://arxiv.org/abs/1711.07971). CVPR 2018.
22. C. Sun, A. Myers, C. Vondrick, K. Murphy, C. Schmid. [*VideoBERT: A Joint Model for Video and Language Representation Learning*](https://arxiv.org/abs/1904.01766). ICCV 2019.
23. B. Wu et al. [*Visual Transformers: Token-based Image Representation and Processing for Computer Vision*](https://arxiv.org/abs/2006.03677). arXiv 2020.
24. F. Locatello et al. [*Object-Centric Learning with Slot Attention*](https://arxiv.org/abs/2006.15055). NeurIPS 2020.
25. Y.-C. Chen et al. [*UNITER: UNiversal Image-TExt Representation Learning*](https://arxiv.org/abs/1909.11740). ECCV 2020.
26. J. Lu, D. Batra, D. Parikh, S. Lee. [*ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks*](https://arxiv.org/abs/1908.02265). NeurIPS 2019.
27. L. H. Li, M. Yatskar, D. Yin, C.-J. Hsieh, K.-W. Chang. [*VisualBERT: A Simple and Performant Baseline for Vision and Language*](https://arxiv.org/abs/1908.03557). arXiv 2019.
28. M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, I. Sutskever. [*Generative Pretraining From Pixels*](https://proceedings.mlr.press/v119/chen20s.html) (iGPT). ICML 2020.
29. D. Mahajan et al. [*Exploring the Limits of Weakly Supervised Pretraining*](https://arxiv.org/abs/1805.00932). ECCV 2018.
30. H. Touvron, A. Vedaldi, M. Douze, H. Jégou. [*Fixing the train-test resolution discrepancy*](https://arxiv.org/abs/1906.06423) (FixRes). NeurIPS 2019.
31. Q. Xie, M.-T. Luong, E. Hovy, Q. V. Le. [*Self-training with Noisy Student improves ImageNet classification*](https://arxiv.org/abs/1911.04252). CVPR 2020.
32. C. Sun, A. Shrivastava, S. Singh, A. Gupta. [*Revisiting Unreasonable Effectiveness of Data in Deep Learning Era*](https://arxiv.org/abs/1707.02968) (JFT-300M). ICCV 2017.
33. A. Kolesnikov, L. Beyer, X. Zhai, J. Puigcerver, J. Yung, S. Gelly, N. Houlsby. [*Big Transfer (BiT): General Visual Representation Learning*](https://arxiv.org/abs/1912.11370). ECCV 2020.
34. J. Djolonga et al. [*On Robustness and Transferability of Convolutional Neural Networks*](https://arxiv.org/abs/2007.08558). CVPR 2021.
35. Q. Wang, B. Li, T. Xiao, J. Zhu, C. Li, D. F. Wong, L. S. Chao. [*Learning Deep Transformer Models for Machine Translation*](https://arxiv.org/abs/1906.01787) (pre-norm). ACL 2019.
36. A. Baevski, M. Auli. [*Adaptive Input Representations for Neural Language Modeling*](https://arxiv.org/abs/1809.10853) (pre-norm). ICLR 2019.
37. D. Hendrycks, K. Gimpel. [*Gaussian Error Linear Units (GELUs)*](https://arxiv.org/abs/1606.08415). arXiv 2016.
38. Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, L. D. Jackel. [*Backpropagation Applied to Handwritten Zip Code Recognition*](https://doi.org/10.1162/neco.1989.1.4.541) (the CNN cited for feature maps). Neural Computation 1(4), 1989.
39. K. He, X. Zhang, S. Ren, J. Sun. [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) (ResNet). CVPR 2016.

**Other sources used in this part**

40. J. L. Ba, J. R. Kiros, G. E. Hinton. [*Layer Normalization*](https://arxiv.org/abs/1607.06450). arXiv 2016.
41. Code for this part: [`vit_part2_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part2_math.py) and [`vit_part2_attn.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/vit_part2_attn.py); figures by [`figs_part2.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/figs_part2.py), paper excerpts by [`shots_part2.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vit/shots_part2.py).
