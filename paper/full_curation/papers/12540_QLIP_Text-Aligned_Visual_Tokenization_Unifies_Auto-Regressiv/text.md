# arXiv:2502.05178v1[cs.CV]7Feb2025

## QLIP: Text-Aligned Visual Tokenization Unifies Auto-Regressive Multimodal Understanding and Generation

Yue Zhao1,* Fuzhao Xue2,† Scott Reed2 Linxi Fan2 Yuke Zhu1,2 Jan Kautz2 Zhiding Yu2 Philipp Kr¨ahenb¨uhl1 De-An Huang2

1UT Austin 2NVIDIA

https://nvlabs.github.io/QLIP/

### Abstract

We introduce Quantized Language-Image Pretraining (QLIP), a visual tokenization method that combines stateof-the-art reconstruction quality with state-of-the-art zeroshot image understanding. QLIP trains a binary-sphericalquantization-based autoencoder with reconstruction and language-image alignment objectives. We are the first to show that the two objectives do not need to be at odds. We balance the two loss terms dynamically during training and show that a two-stage training pipeline effectively mixes the large-batch requirements of image-language pretraining with the memory bottleneck imposed by the reconstruction objective. We validate the effectiveness of QLIP for multimodal understanding and text-conditioned image generation with a single model. Specifically, QLIP serves as a drop-in replacement for the visual encoder for LLaVA and the image tokenizer for LlamaGen with comparable or even better performance. Finally, we demonstrate that QLIP enables a unified mixed-modality auto-regressive model for understanding and generation.

### 1. Introduction

Auto-regressive sequence modeling and its variants have become the state-of-the-art paradigm for natural language modeling [1, 22], multi-modal understanding [51, 78], and arguably visual generation [80, 93]. Despite encouraging progress, a unified auto-regressive model that performs well from any to any modality [53, 77, 88] has proven difficult to train. One key issue lies in visual tokenization. Commonly, an auto-encoder learns to reconstruct the input image with a set of visual tokens and leaves the joint visual-language modeling to the auto-regressive model. This leads to tokenization that compresses the inputs visually, but not semantically, and consecutively leads to the two modalities

*Work done during an internship at NVIDIA Research. †Now at Google DeepMind.

[Figure 1]

Figure 1. State-of-the-art visual tokenizers excel at either understanding (high zero-shot accuracy, e.g. SigLIP [96]) or reconstruction (low reconstruction FID, e.g. MAGVIT2 [93]), but not both. QLIP can perform well on both understanding and reconstruction with a marginal performance drop, opening up an opportunity for unified multi-modal understanding and generation.

competing and slow training [77].

In this paper, we propose to perform multi-modal alignment as early as the visual tokenization phase. The result is a generic visual tokenizer for multi-modal language modeling that excels at capturing semantics and reconstructs highquality visuals at the same time. We train a Binary Spherical Quantization (BSQ)-based Auto-encoder with a textaligned visual-encoder through a contrastive objective. We term the framework Quantized Language-Image Pretraining, QLIP for short.

We identify two main challenges when training QLIP. First, contrastive alignment and regression objectives compete and are hard to balance. Second, contrastive learning relies on large-batch training, while reconstruction losses

incur a heavy memory cost [40, 97] and thus allow for only small batches. To handle the first challenge, we observe the stark difference in the gradient magnitude leads to different convergence rates between the contrastive image-text alignment and pixel reconstruction objectives. We introduce a simple and effective automated weighting scheme between the two losses. We weigh the loss terms by the inverse of their post-hoc loss values without needing any extra cost to compute the gradient. To handle the second challenge, we propose a two-stage training recipe. In the first stage, we train QLIP with a combination of alignment loss and MSE loss with memory-efficient Transformer architecture [10, 17, 61]. In the second stage, we drop the text encoder, freeze the visual encoder, and no longer optimize the contrastive loss. This allows for a smaller batch size and enables fine-tuning of just the bottleneck quantizer and the decoder using a weighted sum of MSE, perceptual loss, and generative adversarial (GAN) loss.

We empirically show that QLIP achieves competitive reconstruction results compared to cutting-edge visual tokenizers, including continuous tokenizer (SD-VAE) and discrete tokenizer (BSQViT) under a similar compression ratio. At the same time, QLIP yields visual-text alignment capability similar to a CLIP-only objective. Furthermore, we validate the effectiveness of our QLIP tokenizer on a wide spectrum of multimodal understanding and generation benchmarks. On LLaVA-based multimodal models, QLIP shows a marginal loss of performance compared to the CLIP-only baseline under a fair comparison (e.g. same input resolution and same instruction-tuning data). This is in contrast to the prior belief that vision tokenizers lead to substantial degradation when used in VLMs. On textconditioned image generation, QLIP shows improved generation FID and better text-image alignment qualitatively compared to the language-agnostic visual tokenizer (VQVAE and BSQViT). Finally, QLIP enables a unified mixedmodal auto-regressive model that can handle language-only, image-to-text, and text-to-image tasks in a single model.

### 2. Related Work

Visual Tokenzation. Analogous to LLM tokenizers [42, 68, 70] that losslessly transform a text string into discrete tokens, visual tokenization aims to map an image or video to tokens while keeping as much visual information as possible. VQ-VAE [83] introduced the concept of discrete tokenized bottlenecks in auto-encoder architectures. Later improvements include better training objectives [24, 62], increasing VQ codebook usage [91, 99], and advanced quantization techniques [44, 55, 93, 98]. All of these efforts aim for improved reconstruction quality using the same compression budget and benefit visual generation [8, 80, 93]. However, better reconstruction quality does not necessarily lead to better visual representation [33, 87]. On the

other hand, visual tokens serve as good intermediate supervision to learn visual encoders with strong representation [3, 46, 57, 102]. Our work shows that by properly adding textual supervision the visual tokenizer can be a strong visual encoder without introducing extra parameters. The concept of aligning visual tokenizer with language is also related to LQAE [50] and SPAE [92]. SPAE [92] aligns the raw pixels with the language token embeddings from a frozen LLM directly. However, SPAE needs more tokens to reconstruct comparably well with VQ-VAE, indicating that the frozen language codebook might not be optimal.

Unifying understanding and generation. Visual tokenization enables unifying multi-modality in the same token space [39, 52, 53, 77, 85, 88, 101]. Chameleon [77] interleaves discrete visual and text tokens with a single Transformer and reported training difficulties. Transfusion [101] combines text token prediction with diffusion for images. Show-o [90] unifies understanding and generation by masked language modeling but uses different tokenizers for different tasks. We use an auto-regressive objective to handle both modalities and QLIP enables quick visuallanguage adaptation from a pre-trained LLM. Another line of works is encoder-free [4, 21], which maps patches of raw pixels into embeddings for joint visual-language modeling. However, this approach is much less data-efficient [6] and unable to generate visual content. VILA-U [89] is closely relevant in that its tokenizer is initialized from SigLIP [96]. However, the understanding performance drops drastically after re-training (see Figure 1). Finally, our visual tokenizer takes advantage of textual supervision and pixel-level reconstruction, echoing recent studies that a mixture of expert vision encoders complement each other for vision-language understanding [71, 81].

### 3. Preliminaries

Visual Tokenization transforms an image to a set of discrete tokens, which are later used for compression, generation, multi-modal understanding [8, 77, 98] via autoregressive sequence modeling. It has three basic components: a visual encoder E, a quantization bottleneck Q, and a visual decoder G. Given an input image X ∈ RH×W×3, the visual encoder E produces a grid of d-dimensional latent embeddings Z = E(X) ∈ R(

H p ×Wp )×d downsampled by a factor p. The bottleneck Q transforms the real-valued latent embeddings into discrete tokens {c1 ...cK} in an elementwise fashion: Zˆ = Q(Z) ∈ {c1 ...cK}(

H p ×Wp ). Finally, the decoder G maps the discretized tokens back to the raw pixel space Xˆ = G(Zˆ) ∈ RH×W×3. The entire network (E,G,and Q) is end-to-end trainable by minimizing a weighted sum of MSE loss Lmse = ∥Xˆ − X∥2, quantization loss Lq(Q), and regularization terms, e.g. a commitment loss [83], or perceptual and adversarial losses [24].

[Figure 2]

[Figure 3]

[Figure 4]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 5]

Visual

|𝑋෠|
|---|

|𝑋෠|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Decoder

[Figure 6]

[Figure 7]

Visual

Visual

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Decoder

Decoder

Text Output Visual Output

|𝑍መ|
|---|

|𝑍መ|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 8]

Auto-regressive Multi-modal Model

[Figure 9]

[Figure 10]

Quantizer

Quantizer

Text Embed Visual Embed

|𝑍|
|---|

|𝑍|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|CLS|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|CLS|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

BPE tokenizer

Quantizer

Quantizer

Text Encoder

Visual Encoder

Text Encoder

Visual Encoder

[Figure 16]

[Figure 17]

Visual Encoder

Visual Encoder

|𝑋|
|---|

|𝑋|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|CLS|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|CLS|
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 18]

[Figure 19]

[Figure 20]

Male Border Collie standing next to road

Male Border Collie standing next to road

Male Border Collie

standing next to road

(a) Tokenizer training: Stage 1 (b) Tokenizer training: Stage 2 (c) Unified multimodal understanding and generation

Figure 2. Overview. (a-b) Two-stage training pipeline of QLIP. (a) In Stage 1, we train QLIP with a combination of alignment loss and MSE loss. (b) In Stage 2, we drop the text encoder, freeze the visual encoder, and no longer optimize the contrastive loss. Only the bottleneck quantizer and the decoder are fine-tuned. (c) With the text-aligned visual tokenizer, we transform the image into visual tokens, concatenate them with text tokens, and use an auto-regressive multi-modal model (Sec 4.1) to model jointly.

Vector Quantization (VQ) [83] QVQ maps latent inputs z ∈ Z to the closest entry in a learnable codebook C = [c1,··· ,cK] ∈ RK×d: QVQ(z) = arg minc

k∈C ∥z − ck∥2. It uses the straightthrough estimator (STE) [5] to propagate gradients through the quantization bottleneck. Empirically, VQ scales poorly with increasing vocabulary size K [93].

Binary Spherical Quantization (BSQ) [98] and Look-up Free Quantization (LFQ) [93] provide a more scalable alternative. They optimize an implicit codebook. For example BSQ projects a hypercube onto a unit sphere and uses the corners of the hypercube as code vectors CBSQ = {−√1L, √1L}L. Each corner ck ∈ CBSQ corresponds to a unique token k. BSQ linear-projects the d-dimensional latent embedding z to a L-dimensional unit hypersphere

- u ∈ SL−1, applies binary quantization per axis uˆ =

Lsign(u), and back-projects to a quantized vector in the original latent space zˆ. The code index at inference is obtained through binarization k = Li=1 1[u

√1

i>0]2i−1.

To optimize for an effective latent code and encourage usage of the implicit codebook, the quantization loss uses an entropy objective [38, 93]

LBSQ = E[H(Q(z))] − γH(E[Q(z)]), (1)

where both entropy terms rely on a soft quantization [2] and an efficient approximate computation exists [98].

The quantization-based auto-encoder enables compressing complex visual content and generating photorealistic images. However, the learned visual tokens yield inferior performance on understanding tasks [77, 90] because of lacking semantic training objectives.

Language-Image Pre-training learns visual representation from natural language supervision via a contrastive objective [32, 56, 59]. The training data is image-text pair (X,Y ), where Y is free-form alt-text or short captions encoded in enumerable text tokens. We employ a visual encoder Ev and a text encoder Et to obtain the visual and text embeddings v = E

∥Ev(X)∥2 , and w = E

v(X)

t(Y ) ∥Et(Y )∥2 .

Given a batch of samples B, the contrastive loss, such as InfoNCE [56], learns to associate embedding pairs for the same sample and separate pairs that are not.

  .

 log

|B|

etvi⊤wi

etvi⊤wi

Lalign(v, w) =

|B| j=1 etvi⊤wj + log

|B| j=1 etvj⊤wi

i=1

(2)

The contrastive-based alignment leads to strong visual representations, which can be integrated into state-of-theart LLMs through cheap and fast adaptation for visuallanguage understanding [49, 51]. However, it cannot generate visual content due to the encoder-only design.

### 4. Quantized Language-Image Pre-training

Our goal is a text-aligned visual tokenizer whose visual embeddings are projected in a shared space with the text embeddings. We start from BSQ-autoencoder and add a contrastive language-image alignment branch. See Figure 2 for an illustration. Specifically, we use a text encoder Et to obtain the language feature w of alt-text Y accompanying the input image X. In the visual encoder Ev, we append a learnable classification token xcls and obtain an extra latent embedding zcls through Ev. (Z,zcls) = E(X;xcls) ∈

| |
|---|

| |
|---|

| |
|---|

[Figure 21]

[Figure 22]

[Figure 23]

where αr′ = αq′ = 1, and αp′ = αg′ = 0.1. We drop the text encoder and freeze the visual encoder to prevent degradation when the batch-size restriction is relaxed. See Figure 4 for the reconstruction result after two stages.

Accelerated training with better initializations. Training a visual tokenizer with only a reconstruction objective is data efficient1. In contrast, CLIP-style training requires 30∼50 billion samples to maximize performance. To narrow the gap, we propose to initialize the visual encoder from either Masked Image Modeling (MIM) pre-training [25] or contrastive language-image pre-training (CLIP) and the text encoder from CLIP. Empirically, this significantly increases convergence and training can be finished using 4 billion samples, 10× faster than training from scratch.

[Figure 24]

[Figure 25]

[Figure 26]

(a) input (b) recon after 1st stage (c) recon after 2nd stage

Figure 4. Comparison of reconstruction results to the input image after the first and second stage. The second-stage model produces more high-frequency details. The figure is best viewed on a PDF viewer with zoom-in.

Balancing reconstruction and alignment objectives. It is important to balance the reconstruction and alignment objective, namely αr : αa. If we probe the gradient of each loss with respect to the last shared layer, i.e. the linear layer in the visual encoder’s last MLP, we see a difference of several orders of magnitude, leading to different convergence rates between the alignment and reconstruction objectives. The problem seems more distinct when the straight-through estimator [5] exists. We visualize this phenomenon in Figure 5 by comparing the gradient norm of two AEs, one of whose quantization bottleneck is replaced with an identity mapping without compression. To mitigate this, we propose a post-hoc way to weigh the two terms. Specifically, we first train the model with either reconstruction or alignment loss only and then choose the multi-task loss weight to be inversely proportional to the final loss values, i.e. αr/αa ≈ Lalign(∞)/Lmse(∞), where L(·)(∞) denotes the loss value after convergence.

AE

102

BSQ-VAE

100

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

R(

p ×Wp +1)×d. The normalized global visual feature for alignment is computed through a linear projection head hv:

H

10 2

10 4

- v = h

v(zcls)

∥hv(zcls)∥2 . Though it seems straightforward at first glance, we observe several challenges when training QLIP and elaborate on how we handle them as follows.

| wLalign| | wLmse| | wLalign| | wLmse|

Figure 5. Comparison of gradient magnitude. Here, w refers to the linear layer in the visual encoder’s last MLP.

Two-stage training. Training QLIP at once is infeasible. It is common practice to use a perceptual and adversarial loss for high-quality reconstruction. Both losses rely on an extra convolutional network [40, 72] and thus increase the memory footprint (See Figure 3). On the other hand, effective contrastive learning requires a large batch size (32k∼98k [96]). To reduce memory costs, we opt for a decoupled training recipe in two stages.

PeakGPUmemory(GB)

40

20

w/o LLPIPS and LGAN

w/. LLPIPS and LGAN

64256 1024 2048

Batch size per device

Figure 3. Memory usage of QLIP.

In the first stage, we optimize a weighted sum of reconstruction loss, quantization loss in Eq (1), and contrastive loss in Eq (2) without the perceptual and adversarial loss:

We opt out of adaptive weight methods [12, 24, 69] for the reason below. Adaptive weight tuning requires computing the gradient with respect to the last shared layer in the visual encoder. Therefore, we need an additional backward call of the decoder which introduces non-negligible (∼31) time and memory overhead. In our experiments, we find the ratio determined above is robust and works well for different settings of patch size and model parameters.

EX,Y [αrLmse + αqLBSQ + αaLalign(v,w)]. (3)

Here, we prioritize learning semantics-rich representation over better visual reconstruction, which is not always beneficial for representation learning. We elaborate on our choice of balancing losses in the following paragraph.

Improved bottleneck in BSQ-AE. In addition to the training recipe, we improve the tokenizer by replacing linear projection from the latent space z ∈ Rd to the codebook

In the second stage, we improve the reconstruction quality and restore higher-frequency details by fine-tuning the quantization bottleneck and the visual decoder:

1A common recipe is to train on ImageNet-1K for 100 epochs. In other words, the model sees 1.3 billion samples

EX αr′ Lmse + αq′ LBSQ + αp′ LLPIPS + αg′ LGAN , (4)

space u ∈ SL−1 with an MLP. So is the mapping from uˆ to zˆ symmetrically.

1 √

u = MLP⇓(z),uˆ =

sign(u),zˆ = MLP⇑(uˆ), (5)

L

where MLP⇓/⇑ denotes down/up projection respectively. Since now the quantization bottleneck is deeper, we option-

ally add an auxiliary term ∥sg(Zˆ) − Z∥2 during training similar to the commitment loss in VQ-VAE [83]. Though it was not necessary in the linear case [98], we see adding it improves reconstruction in our case.

#### 4.1. Unifying Understanding and Generation

Now that we have visual tokens aligned with language, we concatenate them with text tokens with appropriately padded special tokens. On top of this visual-textual token sequence, we apply a Transformer to predict the next token in an auto-autoregressive way without bells and whistles to see if it generates multiple modalities. See Figure 2 (c). We call our final model the Unified Multimodal Model (UM3). Architecture. We begin with the Llama 3 architecture [22]. To handle the issue of norm growth due to competition from multiple modalities reported by Chameleon [77], we apply query-key normalization (QK-Norm) [19] in the attention layer. We observe adding QK-Norm is compatible with a pre-trained Llama 3 without QK-Norm. Therefore, instead of training from scratch like Chameleon, we start from Llama 3 initialization which greatly accelerates training. We augment the token embedding and the output layers to fit the visual tokens. The augmented part is initialized with the mean of the existing text embeddings

ei = Vj=1t ej /Vt,∀i ∈ [Vt + 1,Vt + Vv], where Vt and Vv denotes the vocabulary size of textual and visual tokens.

To alleviate the logit shift problem, we apply the softmax to textual and visual tokens separately:

Vt+Vv

i=1

ex

i

+ 1[i>V

t] log

1[i≤V

t] log

Vt j=1 exj

ex

i

.

Vt+Vv j=Vt+1 exj

(6)

Data Mixing. Each mini-batch is a mixture of text-only, image-text, or text-image. Inspired by the warm-up schedule for learning rate [30], we propose a calm-down schedule for mixing data, i.e. the proportion of text-only data in a mini-batch linearly decays from r0 to rT with respect to training step t:

r(t) =

rT −r0

T (t − T) + rT, if t ≤ T rT, otherwise

, (7)

where r0,rT are pre-defined hyper-parameters and 0 < rT < r0. This prevents the language modeling ability from collapsing at the beginning of multi-modality training.

Dataset Images Text (# tok/src) Usage/Metrics DataComp-1B [27] 1B 20B/alt-text QLIP

LAION-COCO [67]2 4M/600M 40M/BLIP2 T2I (LlamaGen), UM3 SA-1B [41] 11M 400M/Qwen2VL-7B T2I (LlamaGen), UM3 CC-12M [9] 6M/12M 200M/Qwen2VL-7B UM3 DCLM [45] - 300B/raw+filtered UM3

LAION-CC-SBU [51] 558K -/BLIP2 VLM (LLaVA-1.5) LLaVA-Instruct [51] 665K -/convo. VLM (LLaVA-1.5)

ImageNet [20] 1.3M -/label Classi. (ZS), Recon. (RC) MS-COCO [11, 48] 160K 10M/MTurk Caption, generation

Table 1. Dataset summary. We list the statistics of datasets used throughout the paper, including the number of images, the number of text tokens with source, and the usage of the respective dataset.

### 5. Experiments

#### 5.1. Datasets

Table 1 summarizes our datasets. To train QLIP, we use DataComp-1B [27], the largest public image-text pair dataset with 1B samples. Training details are in Sec. A. We evaluate the understanding and reconstruction performance on the validation set of ImageNet-1k [20].

For vision-language understanding, we use the pretraining and instruct-tuning data from LLaVA 1.5 [51]. The evaluation benchmarks will be covered in Sec 5.2.

For text-to-image generation, we use images from Conceptual 12M (CC-12M) [9], SA-1B [41], and a 5M subset of LAION-COCO [67] filtered by aesthetic scores. We use Qwen2-VL-7B [84] to generate captions and use FLANT5 [15, 60] to obtain the text embeddings for conditioning.

To train the unified multi-modal model for understanding and generation, we use a mixture of text data from DCLMbaseline [45] (a 300B-tokens subset), image-text pairs from CC-12M+SA-1B (18M images, or 10B tokens in total).

#### 5.2. Evaluating QLIP

We validate the effectiveness of QLIP on a wide spectrum of visual and multi-modal benchmarks. We categorize them into three parts, i.e. vision-centric understanding, visionlanguage understanding, and text-conditioned visual generation. Finally, we showcase the performance of UM3 on a combination of text-only, I2T, and T2I tasks.

Vision-centric understanding includes (1) image classification, measured by zero-shot accuracy and linear-probing accuracy, and (2) reconstruction quality, measured by reconstruction FID (rFID) [35], PSNR, and SSIM [86].

Vision-language understanding takes as input one or more images X and a text sequence Yi, often known as a prompt or an instruction, and outputs another text sequence Yo that follows the prompt. Following LLaVA 1.5 [51], we employ QLIP’s visual encoder Ev on the image, adapt the visual embeddings through a learnable projection network Fproj,

2hf.co/datasets/guangyil/laion-coco-aesthetic

and feed the adapted feature into a pre-trained LLM.

Hv = Fproj(Ev(X)), Yo ∼ LLM(Hv; Yi). (8)

Instruction tuning undergoes two stages: (1) feature alignment, where we train the visual-to-text projector, and (2) end-to-end fine-tuning, where we train the projector and LLM using curated instruction-following data. We evaluate the instruction-tuned model on visual question-answering datasets including VQAv2 [31], GQA [37], TextVQA [73], plus more comprehensive VLM benchmarks including POPE [47], MME [26], and MM-Vet [94].

Text-conditioned Image Generation (T2I) takes as input a short caption Yi and outputs an image X that depicts the text description. We employ QLIP to transform the input image into a set of discrete visual token indices {k1,··· ,kN}, where N = HW/p2, and use a text encoder to convert the caption into text embeddings Et(Yi). A Llama-2 style Transformer [82] learns from scratch the visual token sequence auto-regressively with the adapted textual embedding as the prefix condition.

Ht = Gproj(Et(Yi)), kn ∼ p(kn′ |Ht, k<n). (9)

Unified Multimodal Models. We evaluate UM3 on a suite of language-only benchmarks, image-to-text captioning, and text-to-image generation. The language-only benchmarks include ARC-Challenge [16], HellaSwag [95], PIQA [7], Social IQA [65], and WinoGrande [64]. For captioning, we report BLEU@4, METEOR, and CIDEr on the MS-COCO Karpathy split. For T2I generation, we report generation FID and CLIPScore [34] on MS-COCO 30k.

#### 5.3. Experiment Results on QLIP

Main results of tokenizations. We compare QLIP with the state-of-the-art visual encoders or tokenizers in Table 2. QLIP-B achieves comparable zero-shot classification accuracy with CLIP-only counterparts. At the same time, it also enables compression with a similar ratio and decoding with a comparable reconstruction quality. Specifically, we compare with VILA-U [89]’s vision tower: QLIP-L with 300M parameters outperforms their shape-optimized ViT (SO) with 400M parameters. Also, we get a very close rFID while achieving 8× compression rate than VILA-U.

Next, we present ablation studies that manifest the advantages of the proposed training strategy. Note that for efficiency we use ViT-B/16 under a shorter schedule (2 billion seen samples). Though a shorter schedule may favor singleobjective baselines, the conclusions we draw generally hold for a full schedule and a bigger backbone.

Ablation: how to balance different objectives? From Table 3a, we see the effect of the loss weights between the alignment and the reconstruction objectives. At higher αa, the alignment loss takes control and the reconstruction result degrades drastically; At higher αr, the reconstruction objective dominates and the zero-shot accuracy improves

0-shot Comp. Reconstruction Seen Data Acc.↑ # bits Ratio rFID↓ PSNR↑ SSIM↑

(BASE BACKBONE) CLIP [59] WIT-400M 68.3 / / / / / EVA-CLIP [75] Merged-2B 74.7 / / / / / SigLIP-B [96] WL-10B 76.7 / / / / / VQGAN [24] IN-1k / 14 438.8 4.98 - MaskGIT [8] IN-1k / 10 614.4 1.98 18.63 0.4619 MoVQGAN [100] IN-1k / &40 153.6 1.12 22.42 0.6731 RQ-VAE/f32 [44] IN-1k / &112 219.4 2.69 - OpenCLIP-B [13] DC-1B 73.5 / - / / / BSQViT [98]† DC-1B / 28 219.4 3.81 24.12 0.6638

- QLIP-B (ours) DC-1B 74.3 28 219.4 3.21 23.16 0.6286

(BASE BACKBONE, SMALLER PATCH) SigLIP-B [96] WL-10B 79.2 / / / / / DALL-E dVAE [62] CC3M+YF / 13 118.2 32.63 27.31 0.7943 ViT-VQGAN [91] IN-1k / 13 118.2 1.55 - -

- SD-VAE 1.x [63] OI-2M / 14 109.7 1.40 23.65 0.6354
- SD-VAE 2.x [58] OI-2M+LAae / #64 24 0.70 26.90 0.7592 SDXL-VAE [58] OI-2M+LAae++ / #64 24 0.67 27.37 0.7814 SBER-MoVQGAN [66] LAHR-166M / 14 109.7 0.96 26.45 0.7250 BSQViT [98] IN-1k / 18 85.3 0.99 27.78 0.8171 EVA-CLIP [75]† DC-1B 77.2 / / / / /

- QLIP-B (ours) DC-1B 75.6 28 54.8 0.70 26.79 0.7905 (LARGE BACKBONE)

- CLIP/f14 [59] WIT-400M 75.5 / / / / / SigLIP-L [96] WL-10B 80.5 / / / / / OpenCLIP-L [13] DC-1B 79.2 / / / / /

- EVA-CLIP-L [75] Merged-2B 79.8 / / / / / Open-MAGVIT2 [54, 93] IN-1k / 18 85.3 1.17 21.90 VILA-U [89] WL-10B+CY-1B 73.3 &56 27.4 1.80 - -

(LARGE BACKBONE, HIGH RESOLUTION) CLIP/f14 [59] WIT-400M 76.6 / / / / / SigLIP-L [96] WL-10B 82.1 / / / / /

- EVA-CLIP-L [75] Merged-2B 80.4 / / / / / VILA-U [89] (SO400M) WL-10B+CY-1B 78.0 &224 21 1.25 - QLIP-L (ours) DC-1B 79.1 28 168 1.46 25.36 0.6903

Table 2. Comparison to state-of-the-art visual encoders or tokenizers. We highlight rows that are most comparable in each group. †: our reproduction. #: effective number of bits when latents are stored in bf16. &: quantizer uses residual quantization (RQ), where the total bits are multiplied by RQ depth.

slowly. With appropriate loss balancing, QLIP matches the reconstruction-only model and is close to the CLIP baseline by ∼1% accuracy drop.

Ablation: How to initialize the visual encoder. In Table 3b, we examine different ways of initializing the visual encoder, i.e. (1) random initialization, (2) EVA-02 [25] trained with Masked Image Modeling (MIM) objective on ImageNet-21k, and (3) EVA-CLIP [75] trained with CLIP objective on Merged-2B. We observe poor zero-shot accuracy using random initialization because 2B samples are insufficient for the visual encoder to learn from textual supervision. Both MIM and CLIP initializations do not suffer from this and achieve similarly high zero-shot accuracy. However, MIM works noticeably better at reconstruction than CLIP. We conjecture that outlier tokens with high norms in CLIP may harm reconstruction [18].

Ablation: Two-Stage training. In Table 3c, we study the two-stage training. We first show that fine-tuning the visual decoder greatly improves rFID from 35.3 to 3.21 with some loss of PSNR. Although fine-tuning on ImageNet yields an

ZS(%) RC(rFID) RC(PSNR)

αa : αr ZS(%) RC(rFID)↓

RC(PSNR)

- (1) Et, Ev, Q, G 35.3 24.49

- Recipe 1

(2) Finetune G

74.3

3.21 23.16 (2)∗ (on IN-1k) 2.90 23.33

- Recipe 2

- 1 : 0 75.7 367.8 11.7

- 1 : 1 75.1 162.6 17.8

Pretrain ZS(%) RC(rFID) RC(PSNR)

- 1 : 102 74.7 41.7 22.5

- 1 : 103 74.3 35.3 24.5

- 1 : 104 35.4 35.6 24.5 0 : 1 0.1 35.7 24.5

- (1) Et, Ev, G

75.0

17.2 26.72

- (2) Train Q 13.7 23.34

None 26.4 35.0 24.8 MIM [25] 74.3 35.3 24.5 CLIP [75] 74.7 41.7 23.9

+ Finetune G

(a) Balancing Loss.

(b) Initialization.

(c) Training Recipe.

Table 3. Ablation studies of training QLIP. ZS: zero-shot classification; RC: reconstruction. We highlight the default setting.

Method Vision Encoder Res LLM VQAv2 GQA TextVQA POPE MME MM-Vet

SEED-X [28] ViT-bigG-14 448 LLaMA-2-13B - 47.9 - 84.2 1435.7 LaVIT [39] ViT-G 224 LLaMA-2-7B 68.2 48.0 - - - EVE [21] - 1344 Vicuna-1.5-7B 78.6∗ 62.6∗ 56.8 85.0 1305.7 25.7 Fuyu - 1080 Persimmon-8B 74.2 - - 74.1 728.6 21.4 VILA-U [89] SigLIP-SO400M 384 LLaMA-2-7B 79.4∗ 60.8∗ 60.8 85.8 1401.8 33.5 Chameleon [77] VQ-VAE 512 LLaMA-2-34B+ 69.6 - - - - Show-o [90] MAGVIT-v2 256 Phi-1.5-1.3B 59.3∗ 48.7∗ - 73.8 948.4 Emu3 [85] MoVQGAN 512 LLaMA-2-8B+ 75.1∗ 60.3∗ 64.7 85.2 - 37.2

78.5∗ 62.0∗ 58.2 85.9 1510.7 30.5

CLIP-Large (orig.) 336

CLIP-Large (repro.) 392 79.1∗(+0.0) 62.3∗(+0.0) 55.4(+0.0) 87.5(+0.0) 1484.9(+0.0) 33.3(+0.0) QLIP-Large (ours) 392 78.3∗(-0.8) 61.8∗(-0.5) 55.2(-0.2) 86.1(-1.4) 1498.3(+13.4) 33.3(+0.0)

LLaVA-1.5 [51]

Vicuna-1.5-7B

- Table 4. Comparison to vision-language modeling on vision-language understanding benchmarks. QLIP’s encoder works on par with LLaVA-1.5 with our reproduced CLIP-Large under a controlled experiment.

even better metric, we stick to the original DC-1B images by default because text-to-image generation later needs a more general decoder. Next, we explore another stage-wise strategy, where we first train the text-aligned auto-encoder without quantization and only train the quantization while fine-tuning the visual decoder. We can see an improved zero-shot accuracy and a similar PSNR. However, the rFID score is much worse than the default recipe where the quantizer is included in the first stage. Recall that FID measures the distance of the high-level feature extracted from Inception-V3 [76], which are strongly correlated to highlevel semantics. This illustrates the importance of learning quantization with language supervision.

#### 5.4. Experiment Results on Multimodal Understanding and Generation

Main results of VLMs on vision-language understanding. We present the performance of VLMs using QLIP’s encoder on vision-language benchmarks in Table 4. Since VLM performance varied significantly due to instruction tuning data, model (vision encoder and LLM) size, and the number of visual patches [43], we tried our best to conduct a controlled experiment by strictly following the training data of LLaVA-1.5 and using Vicuna-1.5-7B [14] as the underlying LLM. As for the vision encoder, we train a CLIP-large with an image resolution of 392 and a patch size of 14 to match QLIP. We see our QLIP-equipped VLM works comparably well with our reproduced CLIP-Large baseline.

Ablation: How to use QLIP in VLMs? We continue the ablation studies on visual tokenization regarding its effect on vision-language understanding. Specifically, we replace the vision encoder in LLaVA 1.5 with QLIP at different layers. We can see that the performance drops severely using the last layer before quantizer Q and after Q, compared to the default second last layer. For the latter one, we ascribe it to the effect of quantization. For the first one, the reason could be that the last layer’s features focus more on the generative/reconstructing objective due to the skip connection design, leaving features with the highest semantic content to earlier layers [23]. We examine the same auto-encoder model with only the reconstruction objective and see a similar drop, indicating again that the reconstruction-only objective does not provide sufficient semantics.

Main results of text-conditioned image (T2I) generation. We present the zero-shot image generation result on MSCOCO using 30K captions in Table 7. We compare QLIP with BSQViT [98], an image tokenizer without semantic alignment, using the same LlamaGen [74] framework and show improved generation FID. Note that QLIP is better than the original LlamaGen with VQGAN with only 30% of the training images. We also provide results on more comprehensive T2I benchmarks including GenEval [29] and DPG-Bench [36]. The full comparison is left in Sec. C.

Qualitative results of T2I generation. In Figure 6, we present side-by-side generated images by LlamaGen with

Text-only I2T (COCO-Karpathy) T2I (COCO-30k) Method # Params ARC-C Hellaswag PIQA SIQA Winogrande BLEU@4 METEOR CIDEr gFID↓ CLIPScore↑

Llama-3.2 [22] 1.4B 34.90 48.70 75.95 43.50 61.48 - - - - ZeroCap [79] 0.5B - - - - - 2.6 11.5 14.6 - LlamaGen [74] 0.8B - - - - - - - - 33.4∗ 0.257∗ UM3 (Ours) 1.5B 34.30 45.35 74.65 43.09 54.22 8.6 20.2 17.3 44.1 0.250

- Table 5. Results of the Unified Multi-modal Language Model. The number with ∗ is obtained using the checkpoint trained with a similar number of seen image tokens (60M image samples, or 30B visual tokens) as ours.

[Figure 27]

[Figure 28]

a kitchen with a nasty dirty kitchen with a light beam through the window

[Figure 29]

[Figure 30]

A bathroom with a sink, counter, toilet and shower curtain.

[Figure 31]

[Figure 32]

An orange and white cat sitting underneath a white bush.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Two giraffes in a room with people looking at them A Wii controller facing a TV with a video game on the channel.

[Figure 37]

[Figure 38]

A burger sitting on a table, with another next to it. A parade on a street with several officers on motorcycles riding while confetti

[Figure 39]

[Figure 40]

falls from the sky on them.

Figure 6. Comparison of generated images with conditioning captions in the bottom. For each pair, the left is from LlamaGen+VQGAN and the right is from LlamaGen+QLIP-B/16 (ours). The caption is also provided at the bottom.

Contra. Recon. layer# use Q ZS(%) GQA TextVQA POPE MME ✓(CLIP-B) ✗ -2 ✗ 68.3 59.9 51.2 84.3 1397.9 ✓ ✗ -2 ✗ 75.7 62.1 51.7 85.9 1411.0

✓ ✓

- -2 ✗ 74.3

61.2 51.1 86.1 1398.7

- -1 ✗ 50.7 45.0 77.2 1077.7

- -1 ✓ 40.4 43.0 50.6 677.17

✗ ✓ -2 ✗ 0.1 50.8 43.8 78.3 1093.8

- Table 6. Ablations studies on vision-language understanding benchmarks. The first row denotes the original CLIP-B model while all other rows are from our models. “use Q” means that the feature is after the quantizer.

Tokenizer # Images

MS-COCO 30K GenEval DPG-Bench gFID↓ CLIPScore↑ Overall↑ Overall↑

VQGAN (used in [74]) 50M 15.68 0.309 0.32 43.22 BSQViT-B/16 15M 19.03 0.303 0.31 34.03 QLIP-B/16 15M 15.29 0.316 0.48 78.17

- Table 7. Zero-shot generation results on MS-COCO 30K, GenEval [29], and DPG-Bench [36]. All use LlamaGen-XL [74].

beam”, “sink, counter”, “white bush”, and “people looking at [the giraffes]”. See Sec. D for more results.

Main results of Unified Multimodal Models (UM3). Finally, we show the performance of the unified multimodal models that perform all text-only, image-to-text, and textto-image tasks in one single model in Table 5. For reference, we list specialized models with a similar model size. For text-only benchmarks, UM3 achieves comparable results to Llama-3.2 on 3 out of 5 benchmarks. In zeroshot COCO captioning, UM3 outperforms ZeroCap [79], a zero-shot captioning model using CLIP and GPT-2. In textconditioned image generation, UM3 achieves slightly worse gFID but comparable CLIP-Score.

### 6. Conclusion

We present Quantized Language-Image Pre-training, a visual tokenization method that performs well on both understanding and reconstruction. The visual tokenizer can be seamlessly plugged into state-of-the-art VLMs and imagegeneration models with comparable performance. Integrating text-aligned tokens with the pre-trained LLM, we show the feasibility of training a unified multi-modal model.

the original VQGAN and QLIP. We put the conditioning caption under each image pair. We can see images generated by QLIP follow captions better by depicting all aspects that might be missing from the VQGAN baseline, e.g. “light

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1
- [2] Eirikur Agustsson, Fabian Mentzer, Michael Tschannen, Lukas Cavigelli, Radu Timofte, Luca Benini, and Luc V Gool. Soft-to-hard vector quantization for end-to-end learning compressible representations. In NeurIPS, 2017. 3
- [3] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. In ICLR, 2022. 2
- [4] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. Fuyu-8b: A multimodal architecture for ai agents, 2023. 2
- [5] Yoshua Bengio, Nicholas L´eonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013. 3, 4
- [6] Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024. 2
- [7] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In AAAI, 2020. 6
- [8] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In CVPR, 2022. 2, 6, 3
- [9] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, 2021. 5
- [10] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016. 2
- [11] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 5
- [12] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In ICML, 2018. 4
- [13] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In CVPR, 2023. 6, 3
- [14] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, 2023. 7
- [15] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa

- Dehghani, Siddhartha Brahma, et al. Scaling instructionfinetuned language models. JMLR, 2024. 5
- [16] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457,

2018. 6

- [17] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness. In NeurIPS, 2022. 2
- [18] Timoth´ee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In ICLR,

2024. 6

- [19] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In ICML, 2023. 5
- [20] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 5
- [21] Haiwen Diao, Yufeng Cui, Xiaotong Li, Yueze Wang, Huchuan Lu, and Xinlong Wang. Unveiling encoder-free vision-language models. In NeurIPS, 2024. 2, 7
- [22] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 1, 5, 8

- [23] Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar, Joshua M Susskind, and Armand Joulin. Scalable pre-training of large autoregressive image models. In ICML, 2024. 7
- [24] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR,

2021. 2, 4, 6, 3

- [25] Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. Image and Vision Computing, 2024. 4, 6, 7
- [26] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 6
- [27] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. In NeurIPS, 2023. 5
- [28] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396,

2024. 7

- [29] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. NeurIPS, 2024. 7, 8, 1
- [30] P Goyal. Accurate, large minibatch sg d: training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017. 5
- [31] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017. 6
- [32] Michael Gutmann and Aapo Hyv¨arinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In AISTATS, 2010. 3
- [33] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 2
- [34] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021. 6
- [35] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 5
- [36] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. 7, 8, 1
- [37] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019. 6
- [38] Aren Jansen, Daniel PW Ellis, Shawn Hershey, R Channing Moore, Manoj Plakal, Ashok C Popat, and Rif A Saurous. Coincidence, categorization, and consolidation: Learning to recognize sounds with minimal supervision. In ICASSP,

2020. 3

- [39] Yang Jin, Kun Xu, Liwei Chen, Chao Liao, Jianchao Tan, Bin Chen, Chenyi Lei, An Liu, Chengru Song, Xiaoqiang Lei, et al. Unified language-vision pretraining with dynamic discrete visual tokenization. In ICLR, 2024. 2, 7
- [40] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In CVPR, 2020. 2, 4
- [41] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 5
- [42] Taku Kudo and John Richardson. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In EMNLP, 2018. 2
- [43] Hugo Laurenc¸on, L´eo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? arXiv preprint arXiv:2405.02246, 2024. 7
- [44] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In CVPR, 2022. 2, 6
- [45] Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick

- Keh, Kushal Arora, et al. Datacomp-lm: In search of the next generation of training sets for language models. arXiv preprint arXiv:2406.11794, 2024. 5
- [46] Tianhong Li, Huiwen Chang, Shlok Mishra, Han Zhang, Dina Katabi, and Dilip Krishnan. Mage: Masked generative encoder to unify representation learning and image synthesis. In CVPR, 2023. 2
- [47] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In EMNLP, 2023. 6
- [48] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 5
- [49] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 3
- [50] Hao Liu, Wilson Yan, and Pieter Abbeel. Language quantized autoencoders: Towards unsupervised text-image alignment. In NeurIPS, 2023. 2, 1
- [51] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024. 1, 3, 5, 7
- [52] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. In ICLR, 2022. 2
- [53] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In CVPR, 2024. 1, 2
- [54] Zhuoyan Luo, Fengyuan Shi, Yixiao Ge, Yujiu Yang, Limin Wang, and Ying Shan. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024. 6, 3
- [55] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple. In ICLR, 2023. 2
- [56] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 3
- [57] Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, and Furu Wei. Beit v2: Masked image modeling with vector-quantized visual tokenizers. arXiv preprint arXiv:2208.06366, 2022. 2
- [58] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 6, 3
- [59] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 3, 6
- [60] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li,

- and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. 5
- [61] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC, 2020. 2
- [62] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML,

2021. 2, 6, 3

- [63] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 6, 3
- [64] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM,

2021. 6

- [65] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. SocialIQA: Commonsense reasoning about social interactions. In EMNLP, 2019. 6
- [66] SberBank. SBER-MoVQGAN or a new effective image encoder for generative models. https://habr.com/ ru/companies/sberbank/articles/740624/,

2023. Accessed: 2024-10-23. 6, 3

- [67] Christoph Schuhmann, Andreas K¨opf, Richard Vencu, Theo Coombes, and Romain Beaumont. Laion coco: 600m synthetic captions from laion2b-en. https://laion. ai/blog/laion-coco/, 2022. Accessed: 2024-10-

23. 5

- [68] Mike Schuster and Kaisuke Nakajima. Japanese and Korean voice search. In ICASSP, 2012. 2
- [69] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In NeurIPS, 2018. 4
- [70] Rico Sennrich, Barry Haddow, and Birch Alexandra. Neural machine translation of rare words with subword units. In ACL, 2016. 2
- [71] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv preprint arXiv:2408.15998, 2024. 2
- [72] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015. 4
- [73] Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, 2019. 6
- [74] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 7, 8, 1
- [75] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023. 6, 7, 1, 3
- [76] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception ar-

chitecture for computer vision. In CVPR, 2016. 7

- [77] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818,

2024. 1, 2, 3, 5, 7

- [78] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1
- [79] Yoad Tewel, Yoav Shalev, Idan Schwartz, and Lior Wolf. Zerocap: Zero-shot image-to-text generation for visualsemantic arithmetic. In CVPR, 2022. 8
- [80] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In NeurIPS, 2024. 1, 2
- [81] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. In NeurIPS, 2024. 2
- [82] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 6
- [83] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. In NeurIPS, 2017. 2, 3, 5, 1
- [84] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 5, 1
- [85] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 2, 7
- [86] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 2004. 5
- [87] Chen Wei, Karttikeya Mangalam, Po-Yao Huang, Yanghao Li, Haoqi Fan, Hu Xu, Huiyu Wang, Cihang Xie, Alan Yuille, and Christoph Feichtenhofer. Diffusion models as masked autoencoders. In CVPR, 2023. 2
- [88] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and TatSeng Chua. Next-gpt: Any-to-any multimodal llm. In ICML, 2024. 1, 2
- [89] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 2, 6, 7, 3
- [90] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528,

2024. 2, 3, 7

- [91] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. In ICLR, 2022. 2, 6, 3
- [92] Lijun Yu, Yong Cheng, Zhiruo Wang, Vivek Kumar, Wolfgang Macherey, Yanping Huang, David Ross, Irfan Essa, Yonatan Bisk, Ming-Hsuan Yang, et al. Spae: Semantic pyramid autoencoder for multimodal generation with frozen llms. In NeurIPS, 2023. 2
- [93] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion–tokenizer is key to visual generation. In ICLR, 2024. 1, 2, 3, 6
- [94] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 6
- [95] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In ACL, 2019. 6
- [96] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023. 1, 2, 4, 6, 3
- [97] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 2
- [98] Yue Zhao, Yuanjun Xiong, and Philipp Kr¨ahenb¨uhl. Image and video tokenization with binary spherical quantization. arXiv preprint arXiv:2406.07548, 2024. 2, 3, 5, 6, 7, 1
- [99] Chuanxia Zheng and Andrea Vedaldi. Online clustered codebook. In ICCV, 2023. 2
- [100] Chuanxia Zheng, Tung-Long Vuong, Jianfei Cai, and Dinh Phung. Movq: Modulating quantized vectors for highfidelity image generation. In NeurIPS, 2022. 6, 3
- [101] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 2
- [102] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pretraining with online tokenizer. In ICLR, 2022. 2

config Stage 1 Stage 2 peak learning rate 5e-4 5e-4

Ev learning rate 2e-4 0 Et learning rate 2e-5 0 G learning rate 2e-3 1e-4

learning rate schedule cosine annealing cosine annealing optimizer LAMB AdamW

optimizer (β1,β2) (0.9, 0.95) (0.9, 0.95) weight decay 0.05 0.05 gradient clip 5 1 input resolution 256 256

patch size 8 8 warm-up iterations 2,000 2,000

total iterations 120,000 120,000 batch size per device 512 128

total batch size 65,536 16,384

D optimizer - AdamW D learning rate - 1e-4

reconstruction loss weight αr 1e3 1

contrastive loss weight αa 1 0 quantization loss weight αq 1 1

perceptual loss weight αp 0 0.1

GAN loss weight αg 0 0.1 commitment loss weight αz 1.0 0

- Table 8. Hyperparamters for training QLIP. Please refer to Sec.4 for the notions of loss weights.

### A. Implementation Details

Training QLIP. Table 8 lists the key hyper-parameters of training QLIP-B-8. The recipe for training other configurations, e.g.QLIP-B-16 and QLIP-L-14, is similar.

Training LLaVA. This strictly follows the training recipe of LLaVA 1.5 for the sake of a controlled experiment. For details, please refer to the original paper [51].

Training LlamaGen. We mostly follow the recipe provided in the original work [74]. Since the authors did not release the training data, we curated the training data by ourselves. We use a combination of two sources: (1) a 5M subset of LAION-COCO, filtered by aesthetic scores, and (2) the full set of SA-1B (with 11M images), whose caption is generated by Qwen2-VL-7B [84].

Training UM3. Table 9 lists the key hyper-parameters of training UM3-1.5B.

### B. More Results on QLIP

Full version of Table 2. We present a more detailed comparison to the state-of-the-art visual encoders or tokenizers in Table 14. Compared to Table 2, we add a column that computes the number of parameters. Though the convolution-based methods, e.g. VQGAN, have fewer parameters than ViT-based methods, e.g. BSQViT and QLIPB, the runtime is slower as is reported in [98]. Therefore, we subsume those under “base backbone”.

Linear Evaluation. In addition to the zero-shot image classification, we conduct a linear probing evaluation to compare all visual encoder methods. Table 11 gives the linear probing settings. For VQ-VAE [83] and LQAE [50], we di-

config Training UM3 peak learning rate 1e-4

learning rate schedule cosine annealing optimizer AdamW

optimizer (β1,β2) (0.9, 0.95) weight decay 0.1 gradient clip 1 warm-up iterations 2,000

total iterations 600,000 batch size per device 8

total batch size 512 sequence length 4,096 calm-down steps 10,000 mix ratio (rtext,0 : ri2t : rt2i) 60:1:3 mix ratio (rtext,T : ri2t : rt2i) 12:1:3

sampling temperature 1.0 sampling top-p 0.95

##### Table 9. Hyperparamters for training UM3.

Method Seen Data Probing Pos. IN-1k Acc.(%) (BASE BACKBONE) VQVAE [83] IN-1k / 18.4 LQAE [50] IN-1k / 39.7 EVA-CLIP-B [75] Merged-2B cls-token 82.7 BSQViT [98]† DC-1B cls-token 29.3 BSQViT [98]† DC-1B ft (avg.) 25.4 QLIP-B (ours) DC-1B cls-token 81.8 QLIP-B (ours) DC-1B ft (avg.) 77.7 QLIP-B (ours) DC-1B cls + ft 82.1 (LARGE BACKBONE, HIGH RESOLUTION) EVA-CLIP-L [75] Merged-2B cls-token 86.3 QLIP-L (ours) DC-1B cls-token 85.2

Table 10. Linear evaluation on image classification.

rectly copy the numbers from the paper due to the inaccessibility of models. We see significant improvement in linear classification accuracy over reconstruction-only tokenizers, such as VQ-VAE and BSQ-ViT, and language-quantized tokenizers, such as LQAE. We explore two probing positions, namely using the reserved [CLS] token (cls-token) or the averaged feature tokens (ft), and their concatenation. Using the averaged feature tokens yields a linear probing accuracy similar to the cls token, indicating that the encoder learns strong semantics. As a reference, we also run the linear evaluation on EVA-CLIP [75] and see QLIP is very close to this upper bound.

### C. More Results on Generation Benchmarks

We show the full results on comprehensive benchmarks such as GenEval [29] and DPG-Bench [36] in Tables 12 and 13 respectively. Under the same T2I framework, QLIPequipped LlamaGen significantly outperforms the opensourced VQGAN-LlamaGen and our reproduced baseline with BSQ-ViT. It also achieves competitive or better results than diffusion-based methods, e.g. SDv1.5 which is trained on much more data. We will add the results in the final version.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

A boat traveling down a canal with a blurred background. Two apples and a bowl and jar of applesauce on a cloth. A rainbow cake with white frosting and a slice cut out.

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Lots of people at the airport waiting to get their luggage A crowd of people walking down a street next to a traffic light. The living room has pictures of trees on the wall

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

A person riding down a trail in front of a person on skis. A bear is sitting in the grass in front of a rusty chain-link fence A cat sitting on top of a television.

Figure 7. Comparison of generated images with conditioning captions in the bottom. For each pair, the left is from LlamaGen+VQGAN and the right is from LlamaGen+QLIP-B/16 (ours). The caption is also provided at the bottom.

config ImageNet linear probing peak learning rate 0.2 / 1.0 (BSQViT)

Model Tokenizer Average Global Entity Attribute Relation Other

VQGAN 43.22 76.60 57.88 66.96 75.78 42.80 (0.8B)

LlamaGen

learning rate schedule cosine annealing optimizer AdamW optimizer (β1,β2) (0.9, 0.999) weight decay 0. input resolution 256 (QLIP-B) / 392 (QLIP-L)

BSQ-ViT 34.03 68.39 47.70 63.40 73.77 33.60 QLIP (Ours) 78.17 82.37 84.68 86.97 92.50 79.20

SDv1.5 (0.9B) 63.18 74.63 74.23 75.39 73.49 67.81

Table 13. Evaluation on DPG-Bench.

patch size 16 (QLIP-B) / 14 (QLIP-L) warm-up epochs 1

total epochs 10 / 20 (BSQViT) batch size per device 128

better following the captions. The visual quality can be improved by adding more training data, long training iterations, and larger backbones. However, this is beyond the scope of this paper.

total batch size 1,024

##### Table 11. Hyperparamters for ImageNet linear probing.

Model Tokenizer Overall Single Obj. Two Obj. Counting Colors Position Attribute

VQGAN 0.32 0.69 0.36 0.20 0.57 0.06 0.02 (0.8B)

LlamaGen

BSQ-ViT 0.31 0.77 0.26 0.13 0.56 0.05 0.06 QLIP (Ours) 0.48 0.91 0.59 0.22 0.80 0.17 0.24

SDv1.5 (0.9B) 0.43 0.97 0.38 0.35 0.76 0.04 0.06

Table 12. Evaluation on GenEval.

### D. More Generation Results

In Figure 7, we show more side-by-side generated images by LlamaGen with the original VQGAN and the proposed QLIP. We emphasize the advantage of QLIP in terms of

# Param. Understanding Reconstruction Seen Data (|E| + |G| + |Q|) # bits 0-shot Acc.↑ rFID↓ PSNR↑ SSIM↑

(BASE BACKBONE) CLIP [59] WIT-400M 87M+0+0 / 68.3 / / / EVA-CLIP [75] Merged-2B 87M+0+0 / 74.7 / / / SigLIP-B [96] WL-10B 87M+0+0 / 76.7 / / / VQGAN [24] IN-1k 29M+42M+4M 14 / 4.98 - MoVQGAN [100] IN-1k (82.7M) &40 / 1.12 22.42 0.6731 MaskGIT [8] IN-1k 24M+30M+6k 10 / 1.98 18.63 0.4619 Open-MAGVIT2 [54, 93] IN-1k 25M+40M+18k 18 / 1.53 21.53 OpenCLIP-B [13] DC-1B 87M+0+0 / 73.5 / / / BSQViT [98]† DC-1B 87M+87M+1M 28 / 3.81 24.12 0.6638

- QLIP-B (ours) DC-1B 87M+87M+1M 28 74.3 3.21 23.16 0.6286

(BASE BACKBONE, SMALLER PATCH) SigLIP-B [96] WL-10B 87M+0+0 / 79.2 / / / DALL-E dVAE [62] CC3M+YF 54M+44M+0 13 / 32.63 27.31 0.7943 ViT-VQGAN [91] IN-1k 91M+91M+0.5M 13 / 1.55 - -

- SD-VAE 1.x [63] OI-2M 34M+49M+0 14 / 1.40 23.65 0.6354

- SD-VAE 2.x [58] OI-2M+LA-ae 34M+49M+0 #64 / 0.70 26.90 0.7592 SDXL-VAE [58] OI-2M+LA-ae++ 34M+49M+0 #64 / 0.67 27.37 0.7814 SBER-MoVQGAN [66] LAHR-166M 29M+42M+4M 14 / 0.96 26.45 0.7250 BSQViT [98] IN-1k 87M+87M+28k 18 / 0.99 27.78 0.8171 EVA-CLIP [75]† DC-1B 87M+0+0 / 77.2 / / /

- QLIP-B (ours) DC-1B 87M+87M+1M 28 75.6 0.70 26.79 0.7905 (LARGE BACKBONE)

- CLIP/f14 [59] WIT-400M 304M+0+0 / 75.5 / / / SigLIP-L [96] WL-10B 304M+0+0 / 80.5 / / / OpenCLIP-L [13] DC-1B 304M+0+0 / 79.2 / / /

- EVA-CLIP-L [75] Merged-2B 304M+0+0 / 79.8 / / / Open-MAGVIT2 [54, 93] IN-1k 50M+65M+18k 18 / 1.17 21.90 VILA-U [89] WL-10B+CY-1B 316M+42M+134M &56 73.3 1.80 - -

(LARGE BACKBONE, HIGH RESOLUTION) CLIP/f14 [59] WIT-400M 304M+0+0 / 76.6 / / / SigLIP-L [96] WL-10B 304M+0+0 / 82.1 / / /

- EVA-CLIP-L [75] Merged-2B 304M+0+0 / 80.4 / / / VILA-U [89] WL-10B+CY-1B 428M+42M+537M &224 78.0 1.25 - QLIP-L (ours) DC-1B 304M+304M+2M 28 79.1 1.46 25.36 0.6903

Table 14. Comparison to state-of-the-art visual encoders/tokenizers. †:our reproduction. #: effective number of bits when latents are stored in bf16. &: quantizer uses residual quantization (RQ), where the total bits are multiplied by RQ depth.

