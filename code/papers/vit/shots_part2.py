"""Highlighted excerpts of the ViT paper for Part 2 (Section 2, Figure 1, Section 3 and 3.1, Eq. 1 to 4, Table 1, Appendix A)
and of the papers it cites there (Cordonnier et al., Image Transformer, Sparse Transformers, Vaswani et al., BERT)."""
from paper_shots import run

JOBS = [
    # ---- Section 2, Related Work (page 2 of the PDF = page index 1), paragraph by paragraph
    dict(name='p2-related-transformers', page=1, anchor='Transformers were proposed by Vaswani et al. (2017) for machine translation',
         highlight=['Transformers were proposed by Vaswani et al. (2017) for machine translation', 'denoising self-supervised pre-training task',
                    'language modeling as its pre-training task'], end='Brown et al., 2020).', above=30),
    dict(name='p2-related-approx', page=1, anchor='Naive application of self-attention to images would require',
         highlight=['each pixel attends to every other pixel', 'quadratic cost in the number of pixels', 'only in local neighborhoods for each query pixel',
                    'can completely replace convolutions', 'Sparse Transformers', 'blocks of varying sizes', 'only along individual axes',
                    'require complex engineering to be implemented efficiently on hardware accelerators'],
         end='efficiently on hardware accelerators.', above=4),
    dict(name='p2-related-cordonnier', page=1, anchor='Most related to ours is the model of Cordonnier et al. (2020)',
         highlight=['extracts patches of size 2 × 2', 'applies full self-attention on top', 'large scale pre-training makes vanilla transformers competitive',
                    'applicable only to small-resolution images'], end='while we handle medium-resolution images as well.', above=4),
    dict(name='p2-related-cnn-attn', page=1, anchor='There has also been a lot of interest in combining convolutional neural networks',
         highlight=['combining convolutional neural networks (CNNs) with forms of self-attention', 'augmenting feature maps for image classification',
                    'further processing the output of a CNN using self-attention'], end='Lu et al., 2019; Li et al., 2019).', above=4),
    dict(name='p2-related-igpt', page=1, anchor='Another recent related model is image GPT (iGPT)',
         highlight=['applies Transformers to image pixels after reducing image resolution and color space', 'generative model',
                    'probed linearly', 'maximal accuracy of 72% on ImageNet'], end='maximal accuracy of 72% on ImageNet.', above=4),
    dict(name='p2-related-scale', page=1, anchor='Our work adds to the increasing collection of papers that explore image recognition',
         highlight=['image recognition at larger scales than the standard ImageNet dataset', 'how CNN performance scales with dataset size',
                    'ImageNet-21k and JFT-300M', 'train Transformers instead of ResNet-based models'], end='ResNet-based models used in prior works.', above=4),
    # ---- Figure 1 and Section 3 (page index 2)
    dict(name='p2-figure1', page=2, anchor='Figure 1: Model overview.',
         highlight=['split an image into fixed-size patches, linearly embed each of them, add position embeddings',
                    'standard Transformer encoder', 'extra learnable “classification token”'], figure=True, column='full', below=0),
    dict(name='p2-method-intro', page=2, anchor='In model design we follow the original Transformer',
         highlight=['as closely as possible', 'intentionally simple setup', 'can be used almost out of the box'], end='can be used almost out of the box.', above=30),
    dict(name='p2-vit-reshape', page=2, anchor='An overview of the model is depicted in Figure 1',
         highlight=['1D sequence of token embeddings', 'sequence of flattened 2D patches', 'effective input sequence length',
                    'constant latent vector size D', 'trainable linear projection', 'patch embeddings'], end='We refer to the output of this projection as the patch embeddings.', above=30),
    dict(name='p2-vit-class', page=2, anchor='Similar to BERT’s [class] token, we prepend a learnable embedding',
         highlight=['prepend a learnable embedding to the sequence of embedded patches', 'image representation y',
                    'MLP with one hidden layer at pre-training time', 'single linear layer at fine-tuning time'], end='by a single linear layer at fine-tuning time.', above=4),
    dict(name='p2-vit-pos', page=2, anchor='Position embeddings are added to the patch embeddings to retain positional information',
         highlight=['retain positional information', 'standard learnable 1D position embeddings', 'more advanced 2D-aware position embeddings'],
         end='serves as input to the encoder.', above=4),
    dict(name='p2-vit-encoder', page=2, anchor='The Transformer encoder (Vaswani et al., 2017) consists of alternating layers',
         highlight=['alternating layers of multiheaded self-attention', 'MLP blocks', 'Layernorm (LN) is applied before every block',
                    'residual connections after every block'], end='Baevski & Auli, 2019).', above=4),
    # ---- page index 3: the GELU line and Eq. 1 to 4, Inductive bias, Hybrid
    dict(name='p2-eq1-4', page=3, anchor='The MLP contains two layers with a GELU non-linearity.',
         highlight=['two layers with a GELU non-linearity'], band=(5, 73), above=30, below=76),
    dict(name='p2-inductive', page=3, anchor='We note that Vision Transformer has much less image-specific inductive bias',
         highlight=['much less image-specific inductive bias than CNNs', 'locality, two-dimensional neighborhood structure, and translation equivariance',
                    'only MLP layers are local and translationally equivariant', 'self-attention layers are global', 'used very sparingly',
                    'cutting the image into patches', 'adjusting the position embeddings for images of different resolution',
                    'learned from scratch'], end='have to be learned from scratch.', above=14),
    dict(name='p2-hybrid', page=3, anchor='As an alternative to raw image patches, the input sequence can be formed',
         highlight=['feature maps of a CNN', 'applied to patches extracted from a CNN feature map', 'spatial size 1x1',
                    'simply flattening the spatial dimensions of the feature map'], end='position embeddings are added as described above.', above=14),
    # ---- Table 1 (page index 4)
    dict(name='p2-table1', page=4, anchor='Table 1: Details of Vision Transformer model variants.',
         highlight=['Details of Vision Transformer model variants'], figure=True, column='full', below=0),
    # ---- Appendix A (page index 12)
    dict(name='p2-appendix-a', page=12, anchor='and their respective query',
         highlight=['weighted sum over all values v in the sequence', 'pairwise similarity between two elements of the sequence'],
         band=(4, 54), above=64, below=58),
    dict(name='p2-appendix-msa', page=12, anchor='is typically set to D/k',
         highlight=['run k self-attention operations, called “heads”, in parallel, and project their concatenated outputs',
                    'is typically set to D/k'], band=(4, 32), above=30, below=38),
    # ---- the cited papers
    dict(name='p2-ext-cordonnier-abs', arxiv_id='1911.03584', page=0, anchor='Recent trends of incorporating attention mechanisms in vision',
         highlight=['attention layers can perform convolution', 'a multi-head self-attention layer with sufficient number of heads is at least as expressive as any convolutional layer'],
         end='corroborating our analysis.', above=4),
    dict(name='p2-ext-cordonnier-2x2', arxiv_id='1911.03584', page=5, anchor='we use a 2 × 2 invertible down-sampling',
         highlight=['2 × 2 invertible down-sampling'], above=30, below=30),
    dict(name='p2-ext-parmar', arxiv_id='1802.05751', page=0, anchor='By restricting the self-attention mechanism to attend to local neighborhoods',
         highlight=['restricting the self-attention mechanism to attend to local neighborhoods', 'significantly increase the size of images the model can process in practice'],
         end='than typical convolutional neural networks.', above=4),
    dict(name='p2-ext-child-fig3', arxiv_id='1904.10509', page=2, anchor='Figure 3. Two 2d factorized attention schemes',
         highlight=['Two 2d factorized attention schemes'], figure=True, column='full', below=46, fig_top=250),
    dict(name='p2-ext-vaswani-eq1', arxiv_id='1706.03762', page=3, anchor='In practice, we compute the attention function on a set',
         highlight=['packed together into a matrix Q', 'We compute the matrix of outputs as:'], end='Dot-product attention is identical to our algorithm', above=4),
    dict(name='p2-ext-vaswani-fig2', arxiv_id='1706.03762', page=3, anchor='Figure 2: (left) Scaled Dot-Product Attention.',
         highlight=['Multi-Head Attention consists of several attention layers running in parallel'], figure=True, column='full', below=0),
    dict(name='p2-ext-bert-cls', arxiv_id='1810.04805', page=3, anchor='The first token of every sequence is always a special',
         highlight=['always a special classification token', 'aggregate sequence representation for classification tasks'],
         end='representation for classification tasks.', above=4),
]

def trim_stamp():
    """The Cordonnier et al. title page carries the rotated arXiv stamp in its left margin; cut that strip off."""
    import os
    from PIL import Image
    from paper_shots import OUT
    path = os.path.join(OUT, 'p2-ext-cordonnier-abs.png')
    if os.path.exists(path):
        im = Image.open(path)
        if im.width > 1200:
            im.crop((300, 0, im.width, im.height)).save(path, optimize=True)


if __name__ == '__main__':
    run(JOBS)
    trim_stamp()
