# arXiv:2409.04410v3[cs.CV]9Feb2025

OPEN-MAGVIT2: AN OPEN-SOURCE PROJECT TOWARD DEMOCRATIZING AUTO-REGRESSIVE VISUAL GENERATION

Zhuoyan Luo1,2˚ Fengyuan Shi1,3˚ Yixiao Ge1: Yujiu Yang2 Limin Wang3 Ying Shan1

1ARC Lab, Tencent PCG 2Tsinghua University 3Nanjing University https://github.com/TencentARC/SEED-Voken

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Figure 1: Reconstruction and generation samples of Open-MAGVIT2. We show 1024 ˆ 1024 reconstructed samples (top) and 256 ˆ 256 generated samples (middle and bottom).

ABSTRACT

The Open-MAGVIT2 project produces an open-source replication of Google’s MAGVIT-v2 tokenizer, a tokenizer with a super-large codebook (i.e., 218 codes), and achieves the state-of-the-art reconstruction performance on ImageNet and UCF benchmarks. We also provide a tokenizer pre-trained on large-scale data, significantly outperforming Cosmos on zero-shot benchmarks (1.93 vs. 0.78 rFID on ImageNet original resolution). Furthermore, we explore its application in plain auto-regressive models to validate scalability properties, producing a family of auto-regressive image generation models ranging from 300M to 1.5B. To assist auto-regressive models in predicting with a super-large vocabulary, we factorize it into two sub-vocabulary of different sizes by asymmetric token factorization, and further introduce “next sub-token prediction” to enhance sub-token interaction for better generation quality. We release all models and codes to foster innovation and creativity in the field of auto-regressive visual generation.

˚Equal Contribution. Work done during an internship at ARC Lab, Tencent PCG. :Corresponding author and project lead.

- Table 1: Configurations of visual tokenizers in Open-MAGVIT2 series. We provide both image and video tokenizers, where the former has a pretrained version to facilitate text-conditional image generation.

Tokenizer

Training

Codebook Size Resolution

Temporal Spatial

Data Ratio pt Ratio ps Open-MAGVIT2-I-In1k ImageNet1k 262144 256 ˆ 256 ´ 16 ˆ 16

128 ˆ 128 ´ 8 ˆ 8 Open-MAGVIT2-I-PT Image-text data

16384 256 ˆ 256 ´ 16 ˆ 16 262144 256 ˆ 256 ´ 16 ˆ 16

Open-MAGVIT2-V-UCF UCF-101 262144 128 ˆ 128 4 8 ˆ 8

- Table 2: Configurations of auto-regressive image generation models in Open-MAGVIT2 series. We partially follow the scaling rule proposed in the previous works (Sun et al., 2024; Tian et al., 2024).

Model Parameters Inter-Blocks N Intra-Blocks L Widths w Heads h

Open-MAGVIT2-AR-B 343M 24 2 1024 16 Open-MAGVIT2-AR-L 804M 36 3 1280 20 Open-MAGVIT2-AR-XL 1.5B 48 4 1536 24

- 1 INTRODUCTION

Large Language Models (LLMs), built upon auto-regressive transformer (Vaswani et al., 2017; OpenAI, 2023; Chowdhery et al., 2022; Touvron et al., 2023), have demonstrated dominance in natural language generation due to the incredible context modeling and scalability. Inspired by this, emergent works introduce auto-regressive models into visual generation (Van Den Oord et al., 2017; Esser et al., 2021; Yu et al., 2022; Lee et al., 2022; Sun et al., 2024). These approaches first utilize a vector quantizer for visual tokenization and de-tokenization, then employ an auto-regressive transformer for discrete visual token sequence modeling.

Although great processes are achieved, the quality of visual generation still falls behind the diffusion-based methods. The main factor is limited tokenizer performance. Tokenizers are generally posited as the upper bound of the visual generation, and inferior off-the-shelf tokenizers (e.g., VQ-VAE (Van Den Oord et al., 2017)) will lead to poor generation quality. Although some improvements are done (Yu et al., 2022; Lee et al., 2022; Sun et al., 2024), current tokenizers are limited by the codebook size and utilization, and the reconstruction performance is still far worse than VAE(Kingma, 2013; Rombach et al., 2022b) used in diffusion models. To unlock the potential of tokenizers, MAGVIT-v2 (Yu et al., 2024a) proposes Lookup-Free Quantizer to enable a highly codeactivated and super-large codebook, and achieves better generation quality than diffusion models. However, such a powerful visual tokenizer is completely closed-source and we have no access to this so far, limiting the development of the academic community.

In this work, we push forward the auto-regressive visual generation in two folds:

- • Replication of the visual tokenizer: We re-implement the advanced Lookup-Free Quantizer proposed by MAGVIT-v2. To our best knowledge, our open-source replication achieves the closest reconstruction performance stated in MAGVIT-v2 (1.18 vs. 1.15 rFID on ImageNet 128ˆ128) and outperforms all other methods on the hallmark Imagenet and UCF benchmarks. Furthermore, we provide the visual tokenizer for text-conditional image generation with improved representational capacity by pretraining it on large-scale general-domain datasets. Our pretrained visual tokenizer achieves state-of-the-art zero-shot reconstruction performance on COCO and ImageNet, surpassing Cosmos (Agarwal et al., 2025) and Llamagen (Sun et al.,

2024) significantly (Table 6).

- • Integrating a super-large codebook with AR visual generation: Instead of simply following MAGVIT-v2 that leverages the vision-oriented design (i.e., mask generative methods (Chang

𝑇𝑇2 𝑇𝑇3 𝑇𝑇𝑡𝑡

𝑇𝑇1

[Figure 16]

[Figure 17]

5 6 ... 2t-1 2t

[Figure 18]

[Figure 19]

1 2

3 4

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Shared Param.

... Block

MAGVIT2 Encoder

[Figure 46]

IntraBlock

IntraBlock

Intra-

IntraBlock

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

×L ×L ×L ×L

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

...

[Figure 73]

Image

C 3 C 5 C 2t-1

C 1 𝑇𝑇1

𝑇𝑇𝑡𝑡

𝑇𝑇2 𝑇𝑇3

𝑇𝑇𝑡𝑡

[Figure 74]

LFQ

Llama Inter-Block

[Figure 75]

[Figure 76]

[Figure 77]

×N

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

S 3 4 ...

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

MAGVIT2 Decoder

[Figure 101]

1 + 2 + 2t-3 + 2t-2

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

Class Token

[Figure 122]

𝑇𝑇1 𝑇𝑇2 𝑇𝑇𝑡𝑡−1

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

...

𝐹𝐹 𝐹𝐹

[Figure 129]

+1 +1 -1 Subtoken

Intra-Token Modeling

C

[Figure 130]

𝑚𝑚𝐾𝐾

Reconstruction

𝒌𝒌

𝑚𝑚2

...

Context Token

[Figure 131]

Subtoken

Inter –Token Modeling

+1 -1 +1

𝑚𝑚0

𝑲𝑲 − 𝒌𝒌

- Figure 2: Overview of Open-MAGVIT2. There are two crucial stages in Open-MAGVIT2. In Stage I: the image is first encoded by MAGVIT-v2 Encoder and subsequently transformed into bits format by Lookup-Free Quantizer (LFQ). In Stage II: The quantized features are further mapped into discrete visual tokens and input into the Llama-based auto-regressive framework for intra- and inter-token relationship modeling.

et al., 2022) for visual synthesis), we seek to exploit the potential of such a large codebook in vanilla auto-regressive generation. To assist auto-regressive models in predicting with a superlarge vocabulary, we factorize it into two sub-vocabulary of different sizes by asymmetric token factorization, and further introduce “next sub-token prediction” to enhance sub-token interaction for better generation quality. Our experiments on the standard visual generation dataset ImageNet suggest that, with the powerful tokenizer, the plain auto-regressive model exhibits superiority and scalability.

## UPDATE LOG

- • V0.1 (17/06/2024): Release an initial version of Open-MAGVIT2 image tokenizer.
- • V1.0 (09/09/2024): Release a series of Open-MAGVIT2 image tokenizers trained on ImageNet (dubbed as Open-MAGVIT2-I-In1k in the paper) and a family of auto-regressive models ranging from 300M to 1.5B (dubbed as Open-MAGVIT2-AR-B/L/XL in the paper).
- • V1.1 (21/01/2025): Release a pretrained version of Open-MAGVIT2 image tokenizers (dubbed as Open-MAGVIT2-I-PT in the paper).
- • V1.2 (09/02/2025): Release an Open-MAGVIT2 video tokenizer with 4ˆ temporal downsampling rate (dubbed as Open-MAGVIT2-V-UCF in the paper).

- 2 METHOD

- 2.1 OVERVIEW

Open-MAGVIT2 is composed of two significant stages. One is a powerful visual tokenizer that maps the input visual signal into the discrete token representations. Subsequently, the vector-quantized sequence will be fed into the auto-regressive transformer for intra- and inter-token relationship modeling, eventually for visual synthesis.

- 2.2 VISUAL TOKENIZER

Preliminary. Visual tokenization is fundamentally deemed as the crucial component in multimodal large language models (MLLMs) to understand the visual signal input. The CNN-based

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

LPIPS↓= 0.315 0.256 0.236

Original

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Original

0.194

0.148 0.134

LPIPS↓=

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

0.082 0.059 Open-MAGVIT2

0.085

LPIPS↓=

Original

VQGAN LlamaGen

## Figure 3: Reconstruction comparison with different tokenizers. We compare VQGAN (Esser

- et al., 2021), LlamaGen (Sun et al., 2024) and our models trained on ImageNet. (Best viewed with zooming in. The original images are from Unsplash).

encoder-quantizer-decoder architecture first proposed in VQVAE (Van Den Oord et al., 2017) is well adopted as the visual tokenizer, which maps input pixels into discrete representations and reconstructs images from quantized features. Specifically, given an video V P RTˆ3ˆHˆW (When T “ 1, the input is an image), the encoder projects it into the feature map Z P RT

1ˆDˆH1ˆW1, where T1 “ T{pt,H1 “ H{ps,W1 “ W{ps, and pt, ps are the temporal and spatial downsample ratio respectively. The quantizer containing a learnable codebook E P R2

KˆD then selects the closest entry zˆ P RD from the codebook for each feature vector z P RD. And we can use discrete token indices X “ txiuT

1ˆH1ˆW1

i“1 to represent the continuous feature map Z. For decoding, each code index will be mapped back to the quantized feature vector and input into the decoder for pixel-level image reconstruction.

Review of Lookup-Free Quantization. Motivated by the relationship between the size of the codebook and the dimension of code embeddings, MAGVIT-v2 (Yu et al., 2024a) eliminates the need for embedding lookup by reducing the dimension of code embedding to zero. Specifically, the codebook is shrunk into an integer set where the latent space of each entry is decomposed as the Cartesian product of single-dimensional variables (i.e., Cˆ “ ŚK

i“1t´1,1u,|Cˆ| “ 2K). As shown in Fig. 2, the tokenization process can be simplified as:

zˆi “ signpziq “ ´ tzi ď 0u ` tzi ą 0u, (1)

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

LlamaGen (16384 VQ Pretrain)

Show-o (8192 LFQ Pretrain)

Original 1024 × 1024

Cosmos (64000 FSQ Pretrain)

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Open-MAGVIT2 (16384 LFQ In1k)

Open-MAGVIT2 (262144 LFQ In1k)

Open-MAGVIT2 (262144 LFQ Pretrain)

Open-MAGVIT2 (16384 LFQ Pretrain)

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

LlamaGen (16384 VQ Pretrain)

Cosmos (64000 FSQ Pretrain)

Show-o (8192 LFQ Pretrain)

Original 1024 × 1024

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Open-MAGVIT2 (16384 LFQ In1k)

Open-MAGVIT2 (262144 LFQ In1k)

Open-MAGVIT2 (262144 LFQ Pretrain)

Open-MAGVIT2 (16384 LFQ Pretrain)

- Figure 4: Reconstruction comparison with different tokenizers . We compare LlamaGen (Sun et al., 2024), Showo (Xie et al., 2024), Cosmos (Agarwal et al., 2025) and our models. The bottom of each case illustrates that the tokenizer pretrained on large-scale datasets gains more superiority in facial and textual scenarios than ones trained on imagenet. Moreover, our pretrained visual tokenizer achieves better reconstruction soundness compared with LlamaGen, Show-o and Cosmos. (Best viewed with zooming in.)

where zˆi denotes the quantized representation of the feature vector zi. And the token index for zi is given by:

ÿK

2k´1 tzˆik ą 0u. (2)

Indexpziq “

k“1

To encourage the confident assignment of each codebook entry and utilization of the whole codebook simultaneously, MAGVIT-v2 further introduces entropy loss:

ÿ pf pzqq ´ p

ÿf pzqq, (3)

1 BH1W1

1 BH1W1

Lentropy “

where p¨q denotes the entropy, B is batch sizes, and fp¨q is a mapping function from latent space to a categorical distribution specifying the probability of assignment to each entry. In our experiment, we observe that replacing traditional code assignment (i.e., pair-wise distance) with this lookup-free quantization enables training a super-large codebook (i.e., 218 codes) of high utilization (100%).

Review of Architecture improvements. Intuitively, since each continuous feature vector will be quantized into K bits, it poses a significant challenge to both the encoder and decoder. Therefore, we re-implement the architecture improvements technique illustrated in (Yu et al., 2024a). 1) Downsamplers in the encoder are strided convolutions with learned kernels while upsamplers in the decoder are the depth-to-space operator. 2) Following (Karras et al., 2019; Peebles & Xie, 2023; Huang & Belongie, 2017), we re-implement the Adaptive GroupNorm Layer, which integrates the quantized vector with the output of each residual block in the decoder.

- 2.3 AUTO-REGRESSIVE TRANSFORMER

Preliminary. Given a sequence of discrete tokens X “ txiuTi“1,T “ H1 ˆ W1 from the visual tokenizer, the auto-regressive transformer predicts the next token xt conditioned on the previous tokens tx1,x2,¨¨¨ ,xt´1u:

źT

ppxt|x1,x2,¨¨¨ ,xt´1q. (4)

ppx1,x2,¨¨¨ ,xTq “

t“1

Auto-regressive Architecture. Considering the different scales of auto-regressive transformer (i.e., from „300M to 1B) and the limited training academic data, directly optimizing such a large vocabulary (i.e., 218 codes) is impractical. Therefore, we propose the asymmetric token factorization technique to assist models in performing “next-token prediction” within concatenated codebooks. Specifically, the LFQ token’s latent space is factorized into M subspaces tx1iuTi“1, tx2iuTi“1, ¨¨¨, txMi uTi“1, each of which contains 2k

m tokens. As shown in Fig. 2, each subspace is embedded individually and their summation is used as the transformer inputs. Conventionally, an intuitive solution to perform auto-regressive within subspaces is leveraging M separate heads for independent categorical distribution modeling. However, since both sub-tokens are derived from the same latent spaces, such a simple operation may ignore their intra-correlation. Consequently, inspired by (Lee

- et al., 2022), we reformulate the autoregression paradigm into modeling both intra- and inter-token dependency, which is essentially “next sub-token prediction”. In this manner, the representational capacity of the super-large codebook can exhibit great potential in auto-regressive generation with better scalability.

- 1) Inter-token Relationship: Given a set of sub-tokens from the visual tokenizers, a stacked of Llama blocks with N layers and w width are leveraged to capture the in-context information between tokens. The process can be formulated as:

Ct “ LlamaBlockps,p

ÿM

i“1

xi1q,¨¨¨ ,p

ÿM

i“1

xit´1qq, (5)

where s denotes the conditional tokens, Ct P RTˆw

s is the t-th context token.

- 2) Intra-token Relationship: We further utilize a transformer with L intra-blocks to autoregressively

predict the each sub-token (x1t, x2t,¨¨¨ ,xMt ) at the position t. By associating the sub-token conditioned with contextual-enriched vector C, the intra-dependency within tokens can be well modeled.

Formally, at t position, the autoregression of predicting the conditional distribution of each subtoken is:

ptm “ LlamaBlockpCt,x1t ¨¨¨ ,xmt ´1q. (6) Therefore, the auto-regressive likelihood is formulated as:

źT

ppX1,X2,¨¨¨ ,XTq “

ppXt|X1,X2,¨¨¨ ,Xt´1q

t“1

(7)

źT

źM

ppxmt |pX1,X2,¨¨¨ ,Xt´1q,px1t,x2t,¨¨¨xtm´1qq,

“

t“1

m“1

where Xt specifies a set of sub-token tx1t,x2t,¨¨¨ ,xMt u at each position t.

- 3 EXPERIMENTS

- 3.1 DATASET AND METRICS

Class-Conditional Image Generation. The training of the visual tokenizer and auto-regressive transformer are both on ImageNet (Deng et al., 2009). Specifically, we train the tokenizer in 128 ˆ 128 and 256 ˆ 256 resolutions. For visual reconstruction, the reconstruction-FID, denoted as rFID (Heusel et al., 2017), codebook utilization, the use percentage of codes, and PSNR on ImageNet 50k validation set are adopted to measure the quality of reconstructed images. Simultaneously, we measure the quality of image generation by the prevalent metrics FID, IS (Salimans et al.,

- 2016) and Precision/Recall (Kynk¨a¨anniemi et al., 2019). Text-Conditional Image Generation. Aiming to serve text-conditional image generation, we pretrain the tokenizer on large-scale general-domain datasets, i.e., 1) General: LAION-COCO (LAION, 2022), CC12M (Changpinyo et al., 2021) and CC3M (Sharma et al., 2018). 2) High-quality: LAION-aesthetics-12M1, LAION-aesthetics (Schuhmann & Beaumont, 2022), JourneyDB (Pan

et al., 2023) and LAION-HD2. Following (Agarwal et al., 2025), we evaluate our visual tokenizer by the zero-shot performance on MS-COCO (Lin et al., 2014) and Imagenet with rFID, PSNR, SSIM.

Video Generation. Following (Yu et al., 2024a), we train our video tokenizer on UCF101 (Soomro, 2012). We use 17-frame video clips with a spatial resolution of 128 ˆ 128 for both training and evaluation. The reconstruction-FVD, denoted as rFVD (Unterthiner et al., 2018) is adopted as the main metrics for quantifying the soundness of reconstructed videos.

- 3.2 IMPLEMENTATIONS DETAILS

Visual Tokenizer Setup. Open-MAGVIT2 follows the same architecture of the visual tokenizer proposed in (Yu et al., 2024a). The visual tokenizer for different purposes, e.g., class-conditional and text-conditional image generation, is trained with similar settings. Specifically, for computational efficiency, we remove the gradient penalty loss, and adopt PatchGAN (Isola et al., 2017) as the discriminator instead of StyleGAN (Karras et al., 2019). All models corresponding to different resolutions are trained with similar settings: an initial 1e ´ 4 learning rate, an Adam Optimizer with β1 “ 0.5, β2 “ 0.9, a total 256 batch size from 270 to 350 epochs (1500k steps for textconditional image generation), a combination of reconstruction, GAN, perceptual (Zhang et al., 2018), entropy penalty (Yu et al., 2024a), commitment losses, LeCAM regularization (Tseng et al., 2021) for training stability, and 32 ˆ GPU / NPU with Pytorch.

Video Tokenizer Setup. We extend the image version of tokenizer following (Yu et al., 2024a) where the temporally casual 3D convolution serves as the main design. Simultaneously, we inflate the image tokenizer trained at 128 ˆ 128 imagenet for video modeling. Similarly, we adopt 3D PatchGAN as the discriminator. The model is trained with the following settings: an initial 1e ´ 4 learning rate with a cosine annealing scheduler, an Adam Optimizer with β1 “ 0.5, β2 “ 0.9, a total 128 batch size with 2500 epochs, the losses are the same as the image tokenizer and 64 ˆ GPU / NPU with Pytorch.

- 1https://huggingface.co/datasets/dclure/laion-aesthetics-12m-umap
- 2https://huggingface.co/datasets/yuvalkirstain/laion-hd-subset

- Table 3: Model designs and reconstruction performance comparison with the original MAGVIT-v2 on 128 ˆ 128 ImageNet 50k validation set (rFID) and UCF-101 (rFVD).

Method Tokens

Train

LFQ

Large Up/Down Deeper Adaptive

rFID rFVD Resolution Codebook Sampler Model GroupNorm

Open-MAGVIT2 16 ˆ 16 128 ˆ 128 ✓ ✓ ✓ ✓ ✓ 1.18 16.0 MAGVIT2 (Yu et al., 2024a) 16 ˆ 16 128 ˆ 128 ✓ ✓ ✓ ✓ ✓ 1.15 8.6

- Table 4: Reconstruction performance of different tokenizers on ImageNet 50k validation set and UCF-101 dataset. For image tokenizers, Open-MAGVIT2 achieves SOTA results on different downsampling rates. : specifies that the training is on OpenImages. ˚ denotes that the results are from the direct inference using the model trained with 128 ˆ 128 resolution without fine-tuning. For video tokenizers, Open-MAGVIT2 achieves the SOTA performance on the reconstruction results. :: denotes that training SweetTokenizer without token compression.

Token

Codebook Type Resolution Size UsageÒ

Train Codebook

Method

Tokens Ratio

rFI(V)DÓ PSNRÒ

Image Tokenizer

VQGAN (Esser et al., 2021) 2D 16 ˆ 16 16 256 ˆ 256 1024 7.94 19.4 ´ SD-VQGAN (Rombach et al., 2022a) 2D 16 ˆ 16 16 256 ˆ 256 16384 5.15 ´ ´ MaskGIT (Chang et al., 2022) 2D 16 ˆ 16 16 256 ˆ 256 1024 2.28 ´ ´ LlamaGen (Sun et al., 2024) 2D 16 ˆ 16 16 256 ˆ 256 16384 2.19 20.79 97%

#### Open-MAGVIT2-I-In1k 2D 16 ˆ 16 16 256 ˆ 256 262144 1.17 22.64 100%

ViT-VQGAN (Yu et al., 2022) 2D 32 ˆ 32 8 256 ˆ 256 8192 1.28 ´ ´ VQGAN: (Esser et al., 2021) 2D 32 ˆ 32 8 256 ˆ 256 16384 1.19 23.38 ´ SD-VQGAN: (Rombach et al., 2022a) 2D 32 ˆ 32 8 256 ˆ 256 16384 1.14 ´ ´ OmiTokenizer-VQ (Wang et al., 2024b) 2D 32 ˆ 32 8 256 ˆ 256 8192 1.11 ´ ´ LlamaGen (Sun et al., 2024) 2D 32 ˆ 32 8 256 ˆ 256 16384 0.59 24.45 ´ Open-MAGVIT2-I-In1k 2D 32 ˆ 32 8 128 ˆ 128 262144 0.34 27.02 100%

Titok-L (Yu et al., 2024b) 1D 32 ´ 256 ˆ 256 4096 2.21 ´ ´ Titok-B (Yu et al., 2024b) 1D 64 ´ 256 ˆ 256 4096 1.70 ´ ´ Titok-S (Yu et al., 2024b) 1D 128 ´ 256 ˆ 256 4096 1.71 ´ ´

Video Tokenizer

TATS (Ge et al., 2022) 2D 4 ˆ 16 ˆ 16 8 128 ˆ 128 16384 162 ´ ´ MAGVIT (Yu et al., 2023) 2D 4 ˆ 16 ˆ 16 8 128 ˆ 128 1024 25 ´ ´ SweetTokenizer (Tan et al., 2024) 1D 256 + 1024 ´ 256 ˆ 256 10481 + 11139 44 ´ ´ LARP-L (Wang et al., 2024a) 1D 1024 ´ 128 ˆ 128 8192 24 ´ ´ LARP-L-Long (Wang et al., 2024a) 1D 1024 ´ 128 ˆ 128 8192 20 ´ ´ SweetTokenizer:: (Tan et al., 2024) 1D 5120 ´ 256 ˆ 256 10481 + 11139 18 ´ ´ Open-MAGVIT2-V-UCF 2D 5 ˆ 16 ˆ 16 8 128 ˆ 128 262144 16 ´ 100

Auto-regressive Transformer Setup. As illustrated, we propose asymmetric token factorization to assist the auto-regressive transformer models in making the precise prediction with a large codebook. Note that, we empirically set M “ 2 and k1 “ 6, k2 “ 12. Since our main focus is on democratizing scalable auto-regressive visual generation, the plain auto-regressive transformer is utilized while the techniques that introduce inductive bias such as AdaLn (Karras et al., 2020) are excluded. Specifically, we adopt the Llama-based (Touvron et al., 2023) architecture (e.g., RoPE (Su

- et al., 2024), SwiGLU (Shazeer, 2020), RMSNorm (Zhang et al., 2022) technique, each of which has been proven effective in (Sun et al., 2024)). The class embedding which is indexed from a set of learnable embeddings serves as the start token. Open-MAGVIT2 follows the simple scaling principle proposed in (Sun et al., 2024), which is in Tab. 2. All models are trained with similar settings:

a base learning rate of 1e ´ 4 per 256 batch size, an AdamW optimizer with β1 “ 0.9, β2 “ 0.95, weight decay = 5e ´ 2, a total 768 batch size and 300 „ 350 training epochs, gradient clipping of 1.0, 0.1 dropout rate for input embedding, FFN module and conditional embedding, 32 „ 96 ˆ GPU / NPU for different scales of the model with Pytorch.

- 3.3 MAIN RESULTS

Visual Reconstruction. As shown in Tab. 3, by incorporating all useful designs proposed in (Yu et al., 2024a), Open-MAGVIT2 matches MAGVIT-v2 performances with merely 0.03 FID margin on ImageNet 128 ˆ 128. Further, we also compare our Open-MAGVIT2 with previous visual tokenizers on ImageNet 256ˆ256 in Tab. 4. Benefiting from the super-large codebook with lookup-free

- Table 5: Class-conditional generation on 256 ˆ 256 ImageNet. ˚ specifies the generated images are 384 ˆ 384 and are resized to 256×256 for evaluation. The evaluation protocol and implementation are the same with ADM.

Type Model #Para. FIDÓ ISÒ PrecisionÒ RecallÒ

Diffusion

ADM (Dhariwal & Nichol, 2021) 554M 10.94 101.0 0.69 0.63 CDM (Ho et al., 2022) ´ 4.88 158.7 ´ ´ LDM-4 (Rombach et al., 2022a) 400M 3.60 247.7 ´ ´ DiT-XL/2 (Peebles & Xie, 2023) 675M 2.27 278.2 0.83 0.57

AR

VQGAN (Esser et al., 2021) 227M 18.65 80.4 0.78 0.26 VQGAN (Esser et al., 2021) 1.4B 15.78 74.3 ´ ´ VQGAN-re (Esser et al., 2021) 1.4B 5.20 280.3 ´ ´ ViT-VQGAN (Yu et al., 2022) 1.7B 4.17 175.1 ´ ´ ViT-VQGAN-re (Yu et al., 2022) 1.7B 3.04 227.4 ´ ´ RQTran. (Lee et al., 2022) 3.8B 7.55 134.0 ´ ´ RQTran.-re (Lee et al., 2022) 3.8B 3.80 323.7 ´ ´

VAR

VAR-d16 (Tian et al., 2024) 310M 3.30 274.4 0.84 0.51 VAR-d20 (Tian et al., 2024) 600M 2.57 302.6 0.83 0.56 VAR-d24 (Tian et al., 2024) 1.0B 2.09 312.9 0.82 0.59 VAR-d30 (Tian et al., 2024) 2.0B 1.92 323.1 0.82 0.59

AR

LlamaGen-L˚ (Sun et al., 2024) 343M 3.07 256.06 0.83 0.52 LlamaGen-XL˚ (Sun et al., 2024) 775M 2.62 244.08 0.80 0.57 LlamaGen-XXL˚ (Sun et al., 2024) 1.4B 2.34 253.90 0.80 0.59 LlamaGen-L (Sun et al., 2024) 343M 3.80 248.28 0.83 0.51 LlamaGen-XL (Sun et al., 2024) 775M 3.39 227.08 0.81 0.54 LlamaGen-XXL (Sun et al., 2024) 1.4B 3.09 253.61 0.83 0.53 Open-MAGVIT2-AR-B 343M 3.08 258.26 0.85 0.51 Open-MAGVIT2-AR-L 804M 2.51 271.70 0.84 0.54 Open-MAGVIT2-AR-XL 1.5B 2.33 271.77 0.84 0.54

- Table 6: Zero-shot reconstruction performance on ImageNet 50k validation set and MS-COCO val2017. The tokenizers are trained with large-scale general-domain datasets and aim to serve textconditional image generation. The results are reported under the same setup for fair comparison (text in gray signifies the results directly from Cosmos (Agarwal et al., 2025) report). : indicates that LlamaGen loads the model initially trained on Imagenet while the others are training from scratch, i.e., MS-COCO and Imagenet-1k are excluded from training data.

Quantizer Training

Codebook MS-COCO 2017 Imagenet-1k Type Data Size rFIDÓ PSNRÒ SSIMÒ rFIDÓ PSNRÒ SSIMÒ

Method

Ratio

Resize 256 ˆ 256

LlamaGen: (Sun et al., 2024) VQ 70M 16 16384 8.40 20.28 0.55 2.47 20.65 0.54 Show-o (Xie et al., 2024) LFQ 35M 16 8192 9.26 20.90 0.59 3.50 21.34 0.59 Cosmos (Agarwal et al., 2025) FSQ - 16 64000 11.97 19.22 0.48 4.57 19.93 0.49 Open-MAGVIT2-I-PT LFQ 100M 16 16384 7.93 22.21 0.62 2.55 22.21 0.62 Open-MAGVIT2-I-PT LFQ 100M 16 262144 6.76 22.31 0.65 1.67 22.70 0.64

Original Resolution

Cosmos (Agarwal et al., 2025) FSQ - 16 64000 7.51 20.45 0.52 1.93 20.56 0.51 Cosmos (Agarwal et al., 2025) FSQ - 16 64000 7.23 20.45 0.53 2.52 20.49 0.52 Open-MAGVIT2-I-PT LFQ 100M 16 16384 6.65 21.61 0.57 1.39 21.74 0.56 Open-MAGVIT2-I-PT LFQ 100M 16 262144 5.10 22.18 0.60 0.78 22.24 0.59

quantization, Open-MAGVIT2 outperforms all previous image tokenizers under fair settings. Moreover, we provide an illustrative visual comparison in Fig. 3. As indicated, our visual tokenizer gains more superiority in detail perception as well as precise facial and text reconstruction.

To facilitate the development of autoregressive text-conditional image generation, we provide an improved version of our tokenizers by pretraining on large-scale image-text datasets. As illustrated in Fig. 4, our tokenizers achieve state-of-the-art performance compared to concurrent methods such as Cosmos (Agarwal et al., 2025) in terms of zero-shot reconstruction on both ImageNet and COCO

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

- (a)
- (b)

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

- (a)
- (b)

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

- Figure 5: Visualization of Open-MAGVIT2 video reconstruction. (a) indicates the original videos while (b) specifies the reconstruction videos.

datasets. It is worth noting that some recent efforts in residual tokenization (Qu et al., 2024; Han et al., 2024) can achieve better results, but are not listed here because residual techniques are orthogonal and compatible with quantization methods such as VQ, LFQ, and FSQ.

Furthermore, we extend our image tokenizer to the video version. As depicted in Tab. 3, The reconstruction result gap on UCF-101 between MAGVIT-v2 and us is narrow. It is worth noting that Open-MAGVIT2 achieves a competitive performance among previous video tokenizers (see in Tab. 4)

Visual Generation. MAGVIT-v2 leverages the non-autoregressive framework for image synthesis and achieves competitive performance. Considering the scalability of auto-regressive models and the remarkable success of the auto-regressive paradigm in MLLM (Team, 2024), we instead focus on exploring the potential of incorporating a super-large codebook for auto-regressive visual generation. As shown in Tab. 5, Open-MAGVIT2 outperforms all previous image generation models using a plain auto-regressive approach. This benefits from the increased representational capacity of the large scale of the codebook. However, we believe that the strength of such a large codebook is still underestimated because of the data bottleneck and the model size. We hope our effort in building such a powerful visual tokenizer helps merit future research in unified MLLM for image generation.

- 3.4 QUALITATIVE RESULTS

We present the qualitative results on Imagenet Benchmark in terms of visual reconstruction (see in Fig. 6) and visual generation (see in Fig. 7), respectively. The video reconstruction visualization (see in Fig. 5) are on UCF-101.

- 4 RELATED WORKS

- 4.1 VISUAL TOKENIZER

Visual tokenizer is to map an image into compact discrete tokens, which are subsequently fed into the generative models for sequence modeling. Early pioneer VQVAE (Van Den Oord et al.,

- 2017) first introduces learnable codebook mechanism for 2D tokens generation. Subsequently, ViTVQGAN (Yu et al., 2022) and RQ-VAE (Lee et al., 2022) improve VQVAE through normalized and multi-scale quantization respectively. Recently, LlamaGen (Sun et al., 2024) reexamines the design of vanilla tokenizer (Esser et al., 2021) and reveals the conflict between the fidelity of the synthesized image and the size of codebook. Therefore, following the simple intuition (Yu et al., 2022)

that reducing code dimension limits the representational capacity of individual tokens, MAGVIT2 (Yu et al., 2024a) proposes an advanced visual tokenizer which significantly enlarges the size of codebook to 218 with Lookup-Free Quantization.

- 4.2 VISUAL GENERATION

Given a set of compact discrete image tokens, there exist two prevalent frameworks for the subsequent image synthesis, including Non-autoregressive and Auto-regressive generation.

Non-autoregressive frameworks. MaskGIT (Chang et al., 2022) utilizes BERT-style transformer (Devlin et al., 2018) to parallelly generate all visual tokens via masked-prediction mechanism. MAGVIT (Yu et al., 2023; 2024a) adopts the same architecture but includes an additional embedding mask for better generation quality.

Auto-regressive frameworks. Autoregressive-based Multi-Modal Large Language Models (Liu et al., 2024; Li et al., 2024) has achieved remarkable success in versatile visual understanding. In contrast, the progress in counterpart visual generation still remains unsatisfactory. The simplest approach VQGAN (Esser et al., 2021) employs tiny GPT2 (Radford et al., 2019) („ 300M) for nexttoken prediction. VAR (Tian et al., 2024) reformulates the image generation approach into next-scale prediction and unveils the scaling principle simultaneously. Subsequently, LlamaGen (Sun et al., 2024) extends VQGAN with Llama (Touvron et al., 2023) architecture, showcasing significant improvement in fidelity. However, the limited codebook size (e.g., 214) in existing auto-regressive models may incur the representational bottleneck. Therefore, considering that the capacity of the visual tokenizer is highly correlated with the quality of visual synthesis (Yu et al., 2024a), we democratize the plain auto-regressive approach with a super-large codebook.

- 5 CONCLUSION

In this work, we re-implement the powerful visual tokenizer, which achieves state-of-the-art performance compared with previous methods, and make it available to the community. Instead of simply following (Yu et al., 2024a) that leverages masked-generative transformer for visual generation, we delve into a more promising manner (i.e., auto-regressive visual synthesis). To excavate the potential of the large vocabulary, we introduce the “next sub-token prediction” paradigm with the asymmetric token factorization technique. The experiment suggests that with the powerful tokenizer, the plain auto-regressive model exhibits superiority and scalability. We hope our contribution to the open-source community can facilitate more innovative and creative works in the field of autoregressive visual generation, eventually making a difference in building an omnipotent multi-modal framework.

Limitations and future work. We expect that the effectiveness of such a super-large codebook, (i.e., 218 codes), is still underestimated due to the limited data scale and the sacrifice of the representational capacity with the token factorization technique. We believe that by amplifying the task with more training data (e.g., text-conditional image generation, video generation, etc.), and enlarging the model size to 7B or even larger, the potential of AR generation with a super-large codebook can be dramatically exploited. Therefore, extending Open-MAGVIT2 into more broad multi-modal generation applications will be a high priority in our future exploration.

ACKNOWLEDGMENTS We sincerely thank Lijun Yu for his encouraging discussions and support. We also thank Tianheng Cheng and Yuxin Chen for their helpful suggestions on this project.

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

- (a)
- (b)

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

- (a)
- (b)

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

- (a)
- (b)

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

- Figure 6: Visualization of the Open-MAGVIT2 tokenizer. The upper part illustrates the model trained at 128 ˆ 128 resolution and tested at 512 ˆ 512 resolution. The second part showcases the tokenizer trained at 256ˆ256 resolution and tested at 256ˆ256 resolution. (a) indicates the original images while (b) specifies the reconstruction images.

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

## Figure 7: Visualization of Open-MAGVIT2 auto-regressive generations. Class-conditional generation on ImageNet 256 ˆ 256.

REFERENCES

Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 2, 5, 7, 9

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer. In CVPR, pp. 11305–11315, 2022. 2, 8, 11

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In CVPR, pp. 3558–3568,

2021. 7

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022. 2

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, pp. 248–255, 2009. 7

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 11

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 34:8780–8794, 2021. 9

Patrick Esser, Robin Rombach, and Bj¨orn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pp. 12873–12883, 2021. 2, 4, 8, 9, 10, 11

Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and time-sensitive transformer. In ECCV, pp. 102–118, 2022. 8

Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. arXiv preprint arXiv:2412.04431, 2024. 10

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, volume 30, 2017. 7

Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. JMLR, 23(1):2249–2281,

2022. 9 Xun Huang and Serge J. Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV, pp. 1510–1519, 2017. 6 Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. In CVPR, pp. 5967–5976, 2017. 7 Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, pp. 4401–4410, 2019. 6, 7 Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In CVPR, pp. 8110–8119, 2020. 8 Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 2

Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alch´e-Buc, Emily B. Fox, and Roman Garnett (eds.), NeurIPS, pp. 3929–3938, 2019. 7

LAION. Laion-coco 600m. https://laion.ai/blog/laion-coco, 2022. 7 Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image

generation using residual quantization. In CVPR, pp. 11513–11522, 2022. 2, 6, 9, 10

Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814, 2024. 11

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, pp. 740–755, 2014. 7

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances

in neural information processing systems, 36, 2024. 11 OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 2 Junting Pan, Keqiang Sun, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun

Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. arXiv preprint arXiv:2307.00716, 2023. 7

William Peebles and Saining Xie. Scalable diffusion models with transformers. In CVPR, pp. 4195–4205, 2023. 6, 9

Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. arXiv preprint arXiv:2412.03069, 2024. 10

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI Blog, 2019. 11

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, pp. 10684–10695, 2022a. 8, 9

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, pp. 10674–10685, 2022b. 2

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In NeurIPS, volume 29, 2016. 7

Christoph Schuhmann and Romain Beaumont. Laion-aesthetics. https://laion.ai/blog/ laion-aesthetics/, 2022. 7

Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL, pp. 2556–2565,

2018. 7 Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. 8 K Soomro. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint

arXiv:1212.0402, 2012. 7 Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2024. 8

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 2, 4, 5, 8, 9, 10, 11

Zhentao Tan, Ben Xue, Jian Jia, Junhao Wang, Wencai Ye, Shaoyun Shi, Mingjie Sun, Wenjin Wu, Quan Chen, and Peng Jiang. Sweettokenizer: Semantic-aware spatial-temporal tokenizer for compact visual discretization. arXiv preprint arXiv:2412.10443, 2024. 8

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 10

Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. arXiv preprint arXiv:2404.02905, 2024. 2, 9, 11

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aur´elien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 2, 8, 11

Hung-Yu Tseng, Lu Jiang, Ce Liu, Ming-Hsuan Yang, and Weilong Yang. Regularizing generative adversarial networks under limited data. In CVPR, pp. 7921–7931, 2021. 7

Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7

Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. volume 30, 2017. 2, 4, 10

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (eds.), NeurIPS, pp. 5998–6008, 2017. 2

Hanyu Wang, Saksham Suri, Yixuan Ren, Hao Chen, and Abhinav Shrivastava. Larp: Tokenizing videos with a learned autoregressive generative prior. arXiv preprint arXiv:2410.21264, 2024a. 8

Junke Wang, Yi Jiang, Zehuan Yuan, Binyue Peng, Zuxuan Wu, and Yu-Gang Jiang. Omnitokenizer: A joint image-video tokenizer for visual generation. arXiv preprint arXiv:2406.09399, 2024b. 8

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 5, 9

Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved VQGAN. In ICLR, 2022. 2, 8, 9, 10

Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, and Lu Jiang. MAGVIT: masked generative video transformer. In CVPR, pp. 10459–10469, 2023. 8, 11

Lijun Yu, Jose Lezama, Nitesh Bharadwaj Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A Ross, and Lu Jiang. Language model beats diffusion - tokenizer is key to visual generation. In ICLR, 2024a. 2, 4, 6, 7, 8, 11

Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. arXiv preprint arXiv:2406.07550, 2024b. 8

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pp. 586–595, 2018. 7

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona T. Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. OPT: open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022. 8

