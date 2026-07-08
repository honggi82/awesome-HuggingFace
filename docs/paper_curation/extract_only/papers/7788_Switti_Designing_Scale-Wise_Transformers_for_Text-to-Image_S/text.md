## SWITTI: Designing Scale-Wise Transformers for Text-to-Image Synthesis

# arXiv:2412.01819v4[cs.CV]20Mar2025

Anton Voronov1,2,3 Denis Kuznedelev1,4 Mikhail Khoroshikh6 Valentin Khrulkov1,5 Dmitry Baranchuk1

1Yandex Research 2HSE University 3MIPT 4Skoltech 5AIRI 6ITMO

https://yandex-research.github.io/switti

[Figure 1]

Figure 1. SWITTI produces high quality and aesthetic 1024×1024 image samples in around 0.5 seconds.

### Abstract

### 1. Introduction

This work presents SWITTI, a scale-wise transformer for text-to-image generation. We start by adapting an existing next-scale prediction autoregressive (AR) architecture to T2I generation, investigating and mitigating training stability issues in the process. Next, we argue that scale-wise transformers do not require causality and propose a non-causal counterpart facilitating ∼21% faster sampling and lower memory usage while also achieving slightly better generation quality. Furthermore, we reveal that classifier-free guidance at high-resolution scales is often unnecessary and can even degrade performance. By disabling guidance at these scales, we achieve an additional sampling acceleration of ∼32% and improve the generation of fine-grained details. Extensive human preference studies and automated evaluations show that SWITTI outperforms existing T2I AR models and competes with state-of-the-art T2I diffusion models while being up to 7× faster.

Diffusion models (DMs) [25, 28, 29, 64–66] are a dominating paradigm in visual content generation and have achieved remarkable performance in text conditional image [4, 15, 38, 50], video [5, 51] and 3D modeling [17, 47]. Inspired by the unprecedented success of autoregressive (AR) models in natural language generation [13, 72, 74], numerous studies have focused on developing AR models specifically for visual content generation [14, 16, 37, 40, 68, 73, 88] to offer a more practical solution to the generative trilemma [81].

Traditional visual AR generative models perform nexttoken prediction [3, 14, 37, 43, 43, 68, 71, 79]. These models flatten a 2D image into a 1D token sequence, and a causal transformer then predicts each token sequentially, resembling the text generation pipeline [13, 52, 53, 74]. While this direction aims to unify vision and language modeling within a single AR framework, it still does not reach stateof-the-art diffusion models in terms of both speed and visual generation quality.

This discrepancy raises an important question: why do traditional AR models struggle in vision domains, whereas diffusion models excel? Tian et al. [73] and Chang et al. [6] argue that next-token prediction imposes an unsuitable inductive bias for visual content modeling. In contrast, diffusion models generate images in a coarse-to-fine manner [2, 11, 57], a process that closely resembles human perception and drawing — starting with a global structure and gradually adding details. Moreover, Rissanen et al. [57] and Dieleman [11] show that image diffusion models approximate spectral autoregression: progressively generating higher-frequency image components at each diffusion step.

Recently, scale-wise AR modeling has emerged as a natural and highly effective image generation solution via a next-scale prediction paradigm [46, 70, 73, 91]. Unlike nexttoken prediction or masked image modeling [6, 7, 16, 39, 40], scale-wise models start with a single pixel and progressively predict higher resolution versions of the image, while attending to the previously generated scales. Therefore, next-scale prediction models perform coarse-to-fine image generation and may also share the spectral inductive bias observed in diffusion models [11, 57], as upscaled images are generally produced by adding higher frequency details. This makes scale-wise AR models a highly promising direction in visual generative modeling. An important advantage of scale-wise models over DMs is that they perform most steps at lower resolutions, while diffusion models always operate at the highest resolution during the entire sampling process. Therefore, scale-wise models yield significantly faster sampling while having the potential to provide similar generation performance to DMs [73].

This work advances the family of scale-wise image generative models by introducing SWITTI, a next-scale prediction transformer for text-to-image generation, that rivals the performance of diffusion models, while being significantly faster. Drawing on recent developments [46, 73], we begin with implementing SWITTI (AR), exploring and mitigating stability issues in the training process.

Then, we investigate whether the scale-wise AR models require attending to all previous scales. We notice that an input image at the current resolution already contains information about preceding scales by design. Therefore, we hypothesize that the model may not need explicit attention to these levels within its architecture. To test this, we remove the causal component from next-scale prediction transformers, enabling faster inference and improved scalability. In addition to the efficiency gains, we find that non-causal models deliver slightly better generative performance.

Also, we explore the influence of text conditioning across different resolution levels and observe that higher scales show minimal reliance on textual information. Leveraging this insight, we disable classifier-free guidance (CFG) [24] at the last scales, thereby reducing inference time by skipping extra forward passes required for CFG calculation. Interest-

ingly, this not only accelerates sampling but also tends to mitigate generation artifacts in fine-grained details.

To sum up, the paper presents the following contributions:

- • We introduce SWITTI, a text-to-image next-scale prediction transformer that excludes explicit autoregression for more efficient sampling and better scalability. As evidenced by human preference studies and automated evaluation, SWITTI outperforms previous publicly available visual AR models. Compared to state-of-the-art text-toimage diffusion models, SWITTI is up to 7× faster while demonstrating competitive performance.
- • We demonstrate that using a non-causal transformer makes SWITTI ∼21% more efficient for 1024×1024 image generation due to cheaper attention operations. Additionally, SWITTI reduces memory consumption during inference, previously needed for storing key-value (KV) cache, by up to 2.3 GB for a single image, enabling better scaling to higher resolution image generation. Moreover, SWITTI slightly surpasses its AR counterpart in generation quality under the same training setups.
- • We find that SWITTI has weaker reliance on the text at high resolution scales. This observation allows us to disable classifier-free guidance on the last two steps, resulting in further ∼32% acceleration and better generation of finegrained details, as confirmed by human evaluation.

### 2. Related work

#### 2.1. Text-to-image diffusion models

Text-conditional diffusion models (DMs) [3, 4, 15, 50, 55, 59, 93] have become the de facto solution for text-to-image (T2I) generation. Despite their impressive performance, a well-known limitation of DMs is slow sequential inference, which hampers real-time or large-scale generation tasks. Most publicly available state-of-the-art T2I diffusion models [4, 15, 38, 50, 58, 93] operate in the VAE [32] latent space, allowing for more efficient sampling of highresolution images. However, these models still require 20−50 diffusion steps in the latent space.

Diffusion distillation methods [31, 45, 48, 60, 61, 67, 85, 86] are the most promising direction for reducing the number of diffusion steps to just 2−4. Current state-of-the-art approaches, such as DMD2 [85] and ADD [60], demonstrate strong generation performance in 4 steps and may even surpass the teacher performance in terms of image quality thanks to additional adversarial training on real images.

#### 2.2. Visual autoregressive modeling

Autoregressive (AR) models is a promising alternative paradigm for image generation that can be categorized into three main groups: next-token prediction [14, 37, 68, 88], next-scale prediction [46, 70, 73], and masked autoregressive models [16, 40].

Next-token prediction AR models are similar to GPTlike causal transformers [13, 52, 53, 74] and generate an image token by token using some scanning strategy, e.g., raster order (left to right, top to bottom). The tokens are typically obtained using VQ-VAE-based discrete image tokenizers [14, 37, 75, 87]. VQ-VAE maps an image to a lowresolution 2D latent space and assigns each latent "pixel" to an element in the learned vocabulary.

Masked autoregressive image modeling (MAR) [16, 40] extends masked image generative models [6, 7, 39] and predicts multiple masked tokens in random order at a single step. Notably, MAR operates with continuous tokens, using a diffusion loss for training and a lightweight token-wise diffusion model for token sampling. Fluid [16] applies this approach to T2I generation and explores its scaling behavior.

Next-scale prediction AR modeling, introduced by VAR [73], represents an image as a sequence of scales of different resolutions. Unlike next-token prediction and masked AR modeling, the scale-wise transformer predicts all tokens at a higher resolution in parallel, attending to previously generated lower-resolution scales.

To represent an image with a sequence of scales, VAR [73] uses a hierarchical VQ-VAE that maps an image to a pyramid of latent variables of different resolutions (scales), progressively constructed using residual quantization (RQ) [37]. In the following, we will refer to this VAE model as RQ-VAE. Each latent variable in RQ-VAE is associated with a set of discrete tokens from a shared vocabulary across all scales, similar to a single-layer VQ-VAE.

During sampling, scale-wise AR model θ iteratively predicts image tokens scale-by-scale, formulated as:

pθ(s1,...,sN|c) =

N

pθ(si|s1,...,si−1,c),

i=1

where si represents RQ-VAE tokens at the current scale, N is the total number of scales, and c is the conditioning information. The model is a transformer [53] with a blockwise causal attention mask, as shown in Figure 5 (Left). VAR [73] adopts a transformer architecture from DiT [49].

Recent works have applied next-scale prediction models to T2I generation [46, 70, 91]. STAR [46] uses the pretrained RQ-VAE model from VAR and modifies its generator to effectively handle text conditioning. Although STAR has not been released as of the writing of this paper, we consider STAR as our baseline architecture, from which we gradually progress towards the proposed model, SWITTI.

A concurrent work, HART [70], proposes a lightweight T2I scale-wise AR model with only 0.7B parameters. It mainly addresses the limitations of the discrete RQ-VAE in VAR by introducing an additional diffusion model to model continuous error residuals, resulting in a hybrid model: a scale-wise AR model combined with a diffusion model for refining the reconstructed latents. In contrast, we focus solely

on designing the scale-wise generative transformer using the pretrained RQ-VAE from VAR [73], slightly tuning it for 512×512 resolution. Combining our scale-wise generative model design with HART’s hybrid tokenization could be a promising direction for future work.

MAR [40] can also be considered as a hybrid model that combines both autoregressive and diffusion priors. Disco-diff [84] conditions a diffusion model on discrete tokens produced with a next-token prediction transformer. DART [19] introduces AR transformers as a backbone for non-markovian diffusion models. Opposed to this line of works, SWITTI does not use any diffusion prior.

### 3. Method

#### 3.1. Architecture

As a starting point, we design a basic text-conditional architecture closely following VAR [73] and STAR [46]. Scalewise AR text-to-image generation pipeline comprises three main components: RQ-VAE [37] as an image tokenizer, a pretrained text encoder [54], and a scale-wise block-wise causal transformer [73].

Our model adopts the pretrained RQ-VAE from VAR [73], which represents an image with N=10 scales for resolutions 256 and 512, and N=14 for 1024. We slightly tune RQ-VAE on 512×512 resolution, as discussed in Section 4.3.

To ensure strong image-text alignment of the resulting model, we follow the literature in T2I diffusion modeling [15, 50] and employ two text encoders: CLIP ViT-L [54] and OpenCLIP ViT-bigG [26]. The text embeddings extracted from each model are concatenated along the channel axis.

The transformer block architecture is adopted from VAR [73], where we incorporate cross-attention [76] layers between a self-attention layer and feed-forward network (FFN) in each transformer block. A pooled embedding from OpenCLIP ViT-bigG is propagated to the transformer blocks via Adaptive Layer Normalization (AdaLN) [82].

We also incorporate conditioning on the cropping parameters following SDXL [50] to mitigate unintended object cropping in generated images. Specifically, we transform center crop coordinates ctop,cleft into Fourier feature embeddings and concatenate them. Then, we map the obtained vector to the hidden size of OpenCLIP ViT-bigG via a linear layer and add it to the pooled embedding.

RMSNorm [90] layers are applied to the inputs of attention and FFN blocks. Layer-normalization (LN) layers [1] are used for query-key (QK) normalization. We also use normalized 2D rotary positional embeddings (RoPE) [21, 46], which allow faster model adaptation to higher resolutions. We use SwiGLU activation function [63] in FFN blocks.

#### 3.2. Training stability issues

Here, we analyze the training performance of the basic model with d=20 transformer blocks and introduce a simple modi-

###### d ×

RMS

- – RMSnorm
- – LayerNorm

RMS

SwiGLU FFN

LN

RMS

RMS

|Pro|j|
|---|---|
| | |

Multi-Head Cross-Attn

Q K V

LN Proj Proj Proj

LN

CLIP ViT-L/14

RMS

RMS

| | |
|---|---|
| | |

Proj

|RM|S|
|---|---|
|Pr|oj|

CLIP ViT-bigG/14

Multi-Head Self-Attn Q K V

Normalized RoPE

| | |
|---|---|
|LN| |
| | |
|Pr|oj|

LN Proj Proj

RMS

Figure 2. Transformer block in the SWITTI model.

fication to improve training stability.

We train the model in mixed-precision BF16/FP32 for 150K iterations on the 256×256 image-text dataset described in Section 4.1. The detailed training and evaluation setups are presented in Appendix A. During the training, we track activation norms and standard metrics, such as FID [23], CLIP Score [22] and PickScore [33].

First, we observe stability issues during training, leading to eventual divergence in our large scale experiments or suboptimal performance. Our investigation reveals that the root of this issue lies in the rapid growth of transformer activation norms, as illustrated in Figure 3 (Blue). Activation norms of the last transformer block grow throughout training iterations, reaching extremely large values of 1016.

Therefore, the first step towards stabilizing the training is to cast the model head to FP32 during training. We find this as a critical technical detail that significantly reduces activation norms, resulting in much better convergence, as we show in Figure 4 (Orange). However, this trick does not fully address the problem since activation norms still keep growing and reach high values of 104 by the end of training.

To further reduce the growth of activation norms during

| |Switti (AR) BF16 head Switti (AR) FP32 head Switti (AR) + “sandwich”<br><br>Switti| |
|---|---|---|
| | | |
| | | |
| | | |

16

Meanactivationnorm,log10

14

12

10

8

6

4

2

0

0 10K 20K 30K 40K 50K 60K 70K 80K 90K 100K 110K 120K 130K 140K

Train iteration

- Figure 3. Last transformer block activation norms over training. Casting the prediction head to full-precision reduces the norm growth. “Sandwich”-normalization further mitigates the issue.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 50K 100K 150K

Train iteration

0.200

0.225

0.250

0.275

0.300

CLIP Score

0 50K 100K 150K

Train iteration

10

20

30

40

FID

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 50K 100K 150K

Train iteration

0.190

0.195

0.200

0.205

PickScore

Switti (AR) BF16 head Switti (AR) Switti

- Figure 4. Evaluation of d=20 models on COCO 30K. Using the non-causal attention mask also slightly improves the performance.

training, we employ “sandwich”-like normalizations [12, 93], to keep the activation norms in a reasonable range. Specifically, we insert additional normalization layers right after each attention and feed-forward blocks. As we show in Figure 3, this modification further mitigates the growth of activation norms during training. However, we find that neither “sandwdich” normalizations nor the choice of normalization functions do not affect the metrics of the models at this scale, as evidenced in Appendix B.

We illustrate the transformer block of the described architecture in Figure 2 and denote the scale-wise AR model with the proposed architecture as SWITTI (AR).

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

Block-wise causal mask (VAR) Block-wise non-causal mask (Switti)

- Figure 5. Visualization of the block-wise self-attention masks in VAR (Left) and SWITTI (Right).

[Figure 2]

| | | | |
|---|---|---|---|
| |Switti Tr|ansformer (non-ca|usal)|
| | | | |

RQ-VAE

| | | | |
|---|---|---|---|
| |Map to embed|dings and upscale| |
| | | | |

Figure 6. Sampling with SWITTI. Model inputs at each scale already incorporate representations of previously generated scales, motivating us to employ a non-causal transformer.

#### 3.3. Employing a non-causal transformer

Next, we delve into the autoregressive next-scale prediction sampling of the original VAR [73]. At each generation step i, the VAR transformer predicts a sequence of tokens si

conditioned on the previously generated tokens s0,...,si−1. The embeddings of these tokens form a feature map of the spatial size hi×wi. This feature map is then upscaled to hi+1×wi+1 and combined with the previously predicted feature maps to serve as an input for the next step.

Therefore, we observe that the conditioning on the preceding scales occurs twice in VAR’s image generation: first, implicitly, when forming the model input, and second, explicitly, using the causal transformer. Based on this observation, we update the attention mask to allow self-attention layers to attend only to tokens at the current scale, as shown in Figure 5 (Right). This implies that the transformer is no longer causal, enabling more efficient sampling due to cheaper attention operations and eliminating the need for key-value (KV) cache. Figure 6 illustrates the next-scale prediction sampling using a non-causal transformer. Overall, we refer to our scale-wise model with the non-causal transformer architecture as SWITTI.

A concurrent work, HART [69], makes a similar observation that self-attention in VAR is mostly local, allowing them to discard up to 80% tokens during training, while the model remains auto-regressive. In contrast, we remove causality in transformer and observe that this modification slightly improves performance in terms of CLIP Score and PickScore, as we show in Figure 4 (Green). Appendix C provides additional training loss analysis to justify our choice.

#### 3.4. The role of text conditioning

Finally, we examine the effect of text conditioning at different model scales.

Cross-attention. First, we plot a cross-attention map between image tokens at different scales and text tokens (except the BOS-token), averaged across transformer blocks. We

[Figure 3]

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

[Figure 4]

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

[Figure 5]

0.25

0.20

###### iScale

0.15

0.10

0.05

arobot ,fullbodytransformer, style , 8 kEOS

a sadpuppywithlargeeyesEOS

- Figure 7. Cross-attention maps between image tokens at each scale (Y-axis) and text tokens (X-axis). Attention scores decrease for main semantic tokens and pooled embeddings (EOS), while increase for less informative tokens, e.g., articles.

[Figure 6]

Witcher, portrait, white background, 8k Robot, full head, transformer style

60% 40% 80% 20% 90% 10% 100%

100% 20% 80% 40% 60% 50% 50%

- Figure 8. Prompt switching within 512×512 generation. Changing the prompt at late steps has a minor effect on the resulting image.

find that attention scores to the BOS-token are high for all scales and omit it to better visualize the pattern for other tokens. Figure 7 shows a typical pattern for two random prompts. At higher scales, the scores decrease for the EOStoken (used as a pooled embedding) but increase for the less semantic tokens, such as articles and stylistic words. This suggests a weaker reliance on the prompt at later scales. Such behavior is consistent across different prompts.

Prompt switching. Next, we investigate the impact of textconditioning at various scales by switching the text prompt to a new one starting from a certain scale. The visualization of prompt switching is presented in Figure 8, with additional examples provided in Appendix D. Indeed, the prompt has minimal influence on the image semantics at the last two scales. Interestingly, switching a prompt at the middle scales results in a simple image blending approach.

Practical implication. Classifier-free guidance (CFG) [24] is an important sampling technique for high-quality textconditional generation that requires an extra model forward pass at each step. Specifically for the scale-wise models, calculations at the last scales take up most of the computing time of the entire sampling process. To save costly model forward passes in high resolution, we propose disabling CFG at the last scales, expecting little effect on generation performance, as also was recently noted in diffusion models [35].

### 4. Model Training

#### 4.1. Pretraining

Following the transformer scaling setup in VAR [73], we set the number of transformer layers d=30 for our main model, resulting in ∼2.5B trainable parameters. We pretrain the model in two stages: first on images of 256×256 resolution, followed by training at 512×512.

As a training dataset, we collect 100M image-text pairs that are filtered from a base set of 6B pairs from the web. Images are filtered to ensure high aesthetic quality and recaptioned using open-source VLMs. We provide more technical details about data filtering and training hyperparameters in Appendix A.

#### 4.2. Supervised fine-tuning

After the pretraining phase, we further fine-tune the model using ∼40,000 text-image pairs, inspired by practices in large-scale T2I diffusion models [30, 38, 42]. The pairs are manually selected by assessors instructed to capture exceptionally aesthetic, high quality images with highly relevant and detailed textual descriptions.

The model is fine-tuned for 40K iterations with a batch size of 64 and a learning rate 5e−7 on image central crops of resolution 1024×1024.

In addition, we slightly perturb the RQ-VAE latents prior to the quantization step with Gaussian noise (σ=0.01) as an augmentation to mitigate overfitting. We observe that the perturbed latents after the quantization and dequantization steps produce visually indistinguishable images while resulting in ∼80% different tokens in a sequence, indicating that tokens at latter scales have less impact on the image quality.

#### 4.3. RQ-VAE tuning

In this work, we use the released RQ-VAE from VAR [73] that was trained on 256×256 images and fine-tune its decoder to adapt it for 512×512 resolution. The encoder and codebook are kept frozen to preserve the same latent space, allowing to fine-tune the autoencoder independently from the generator. Following the standard practice in superresolution literature [36], we use a combination of L1 reconstruction, LPIPS perceptual [92] and adversarial [36] losses, resulting in the following fine-tuning objective:

+ LLPIPS + Ladv (1)

L = LL1

We adopt a UNetSN discriminator from Wang et al. [78] for adversarial loss. The generator and discriminator are trained for 100K steps with a batch size of 256 and a constant learning rate of 1e−5.

To compare the reconstruction quality of the original RQVAE and the one with a tuned decoder, we compute classic full-reference metrics PSNR, SSIM [80], LPIPS [92] and noreference CLIP-IQA [77] metric on a held-out dataset of 300

images. Results in Table 1 demonstrate that the fine-tuned RQ-VAE outperforms an original model with respect to all metrics. Interestingly, the gains from fine-tuning RQ-VAE decoder for 512×512 resolution automatically transferred to 1024×1024 image reconstruction. Additionally, we provide several representative comparisons in Appendix E.

Model PSNR ↑ SSIM ↑ LPIPS ↓ CLIP-IQA ↑ 512×512 Reconstruction

- Original 21.60 0.634 0.200 0.727

- Fine-tuned 22.27 0.653 0.188 0.772 1024×1024 Reconstruction

Original 22.53 0.703 0.207 0.683

- Fine-tuned 23.91 0.733 0.177 0.748 Table 1. Comparison of a fine-tuned and an original RQ-VAE.

We believe that more pronounced gains can be achieved via more thorough investigation of RQ-VAE and training the entire model. We leave this direction for future work.

### 5. Experiments

We compare our final models with several competitive textto-image baselines from various generative families:

- • Diffusion models: Stable Diffusion XL [50], Stable Diffusion 3 Medium [15], Lumina-Next [93].
- • Diffusion distillation: SDXL-Turbo [60], DMD2 [85]
- • Autoregressive models: Emu3 [79], Lumina-mGPT [43], LlamaGen-XL [68], HART [70].

#### 5.1. Automated evaluation

Evaluation setting. To comprehensively evaluate the performance of our models, we use a combination of established automated metrics and a human preference study. For automated metrics, we report CLIP Score [22], ImageReward [83], PickScore [33], FID [23] and GenEval [18]. For all evaluated models, we generate images in their native resolution and resize them to 512×512. More details about the evaluation pipeline are provided in Appendix F.

We calculate metrics on two validation datasets frequently used for text-to-image evaluation: MS-COCO [41] and MJHQ [38]. For both datasets, we generate one image for each of 30,000 validation prompts.

Results. We present the metric values in Table 2. SWITTI achieves comparable performance to the baselines, ranking top-3 for 7 out of 9 automated metrics while exhibiting higher efficiency than most competitors. We do not provide automated metrics for Emu3 and Lumina-mGPT due to their exceptionally long sampling times, that makes generating 60,000 images infeasible in a reasonable time.

#### 5.2. Human evaluation

While automated metrics are widely adopted for evaluating text-to-image models, we argue that they do not fully re-

COCO 30K eval prompts MJHQ 30K eval prompts Model

Latency, s/image

Parameters count, B

PickScore ↑ CLIP ↑ IR ↑ FID ↓ PickScore ↑ CLIP ↑ IR ↑ FID ↓ GenEval ↑ Distilled Diffusion Models

SDXL-Turbo [60] 0.4 2.6 0.229 0.355 0.83 17.6 0.216 0.365 0.84 15.7 0.55 DMD2 [85] 0.4 2.6 0.231 0.356 0.87 14.3 0.219 0.374 0.87 7.2 0.58

Diffusion Models

SDXL [50] 2.3 2.6 0.226 0.360 0.77 14.4 0.217 0.384 0.78 7.6 0.55 SD3-medium [15] 3.9 2.0 0.227 0.354 1.01 19.5 0.215 0.363 0.91 13.1 0.65 Lumina-Next [93] 5.8 2.0 0.224 0.329 0.55 18.4 0.216 0.353 0.75 5.9 0.47

Autoregressive Models

LlamaGen [68] 3.8 0.8 0.208 0.274 -0.25 44.8 0.194 0.288 -0.45 26.9 0.32 HART [70] 0.5 0.7 0.223 0.341 0.75 20.9 0.216 0.366 0.84 5.8 0.55

SWITTI 512 (ours) 0.1 2.5 0.227 0.356 0.95 17.6 0.217 0.381 0.91 9.5 0.62 SWITTI 1024 (ours) 0.5 2.5 0.229 0.355 0.96 18.2 0.219 0.378 0.81 8.1 0.62

Table 2. Quantitative comparison of SWITTI to other competing open-source models. The best model is highlighted in red, the second-best in blue, and the third-best in yellow according to the respective automated metric.

Relevance

Aesthetics

Switti SDXL-Turbo

SDXL-Turbo

Switti DMD2

DMD2

Switti SDXL

SDXL

Switti SD3

SD3

Switti Lumina-Next

Lumina-Next

Switti Emu3

Emu3

Switti Lumina-mGPT

Lumina-mGPT

Switti LlamaGen

LlamaGen

Switti HART

HART

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Image complexity

Defects

Switti SDXL-Turbo

SDXL-Turbo

Switti DMD2

DMD2

Switti SDXL

SDXL

Switti SD3

SD3

Switti Lumina-Next

Lumina-Next

Switti Emu3

Emu3

Switti Lumina-mGPT

Lumina-mGPT

Switti LlamaGen

LlamaGen

Switti HART

HART

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Figure 9. Human study comparing SWITTI with competing AR, diffusion-based models. Error bars correspond to a 95% confidence interval.

flect all aspects of image generation quality. Therefore, we conduct a rigorous human preference study across several aspects: the presence of defects, textual alignment, image complexity and aesthetics quality. Appendix G provides more details about the human evaluation setup. For human evaluation, we generate four images for each of 128 captions from Parti prompts collection [89], a set specifically curated for human preference study [15, 60, 85].

Sampling setup. For a side-by-side comparison, we generate images with classifier-free guidance set to 6 and deactivate it at the last two scales, as described in Section 3.4. We follow the original VAR inference implementation and apply Gumbel softmax sampling [27] with a decreasing temperature, starting from the fifth scale. At the first four scales, we use nucleus sampling with top-k=400 and top-p=0.95.

Results. We provide the results of a side-by-side comparison of SWITTI against the baselines for various image quality aspects in Figure 9. As follows from the human evaluation, SWITTI outperforms all AR baselines in most aspects, only lagging behind Lumina-mGPT, Emu3 and LlamaGen in terms of image complexity. As for diffusion models, DMD2 slightly outperforms SWITTI in terms of defects presence, which can be attributed to its adversarial training, SD3-medium is better at text alignment, likely due to an additional text encoder, whereas SDXL surpasses our model in aesthetics. In other comparisons, SWITTI is on par with the diffusion models and their distilled versions, with respect to statistical error. We provide qualitative comparisons in Figures 10 and 15.

[Figure 7]

Figure 10. Qualitative comparison of SWITTI for 1024×1024 generation against the baselines. More examples are in Appendix H.

#### 5.3. Inference performance evaluation

Next, we analyze the efficiency of SWITTI’s sampling and compare it to the baselines. We consider two settings: measurement of a single generator step without considering time for text encoding and VAE decoding; and inference time of a full text-to-image pipeline. All models are evaluated in half-precision with a batch size of 8. KV-cache is enabled for all AR models. All models are evaluated on a single NVIDIA A100 80GB GPU.

As follows from Table 3, SWITTI is among the most efficient image generation models of similar size, being more

than 4 times faster than SDXL. Notably, SWITTI takes the same time as HART to generate a batch of 1024×1024 images while being more than three times larger. This efficiency stems from the fact that we do not employ an additional diffusion model during de-tokenization in RQ-VAE, from the transitioning to a non-causal architecture, and from disabling classifier-free guidance at the latest scales. Table 8 presents the evaluation on 512×512 resolution.

#### 5.4. Ablation study

Finally, we evaluate the effect of our architectural choices on the image generation quality and inference time. The quanti-

Generator size, B

Full, s/image Diffusion Models

1 step, ms/image

Model

N steps

SDXL-Turbo 2.6 4 42.1 0.4 DMD2 2.6 4 42.1 0.4 SDXL 2.6 25 42.1 2.3 SD3-medium 2.0 28 61.3 3.9

Autoregressive Models

Lumina-mGPT 7.0 1024 — 527.2 LlamaGen 0.8 1024 — 3.8* HART 0.7 14 4.9** 0.5 SWITTI (AR) 2.5 14 33.2** 0.6

SWITTI 2.5 14 26.4** 0.5

- *Can only generate in 512x512.
- **Time averaged over 14 steps.

- Table 3. Comparison of 1024×1024 image generation times.

tative evaluation in Tables 4 and 5 demonstrates that SWITTI is not only more sample efficient than its AR alternative, but also exhibits slightly better visual quality with respect to all aspects of human evaluation and automated metrics. Moreover, disabling classifier-free guidance at the last two scales noticeably reduces defect presence in SWITTI’s images without affecting other aspects of evaluation, as illustrated in Figure 11. Nevertheless, it should be noted that enabling CFG at the last scales can still be beneficial. For example, in scenarios where prompts contain text to be rendered in a small font size, guidance can improve the spelling.

Setup 1 Setup 2 Relevance ↑ Aesthetics ↑ Complexity ↑ Defects ↑ SWITTI SWITTI AR 0.550.06 0.550.06 0.520.06 0.600.06 SWITTI SWITTI

+ late CFG

0.500.06 0.500.06 0.510.06 0.560.06

- Table 4. Human evaluation of SWITTI design choices. Scores are the mean of Setup 1 wins plus half-ties, with subscripts denoting half the 95% confidence interval.

As we show in Table 6, a non-autoregressive architecture improves sampling efficiency, reducing latency by around 21%. Disabling CFG on the last two scales further decreases latency by an additional 32%. Moreover, by removing the causality in SWITTI, we free up to 2.3 GB of GPU memory (in half precision), required to store the KV caches of the self-attention layers during the sampling for a single image.

COCO MJHQ Setup PS ↑ CLIP ↑ IR ↑ FID ↓ PS ↑ CLIP ↑ IR ↑ FID ↓ GenEval

SWITTI (AR) 0.227 0.351 0.91 18.7 0.217 0.375 0.76 8.3 0.62 SWITTI w/ CFG 0.228 0.353 0.90 17.8 0.218 0.375 0.75 8.1 0.62 SWITTI 0.229 0.355 0.96 18.2 0.219 0.378 0.81 8.1 0.62

- Table 5. Impact of the acceleration modifications at 1024×1024.

[Figure 8]

Figure 11. Illustrative examples when disabled CFG at the last scales (Bottom) mitigates the artifacts in fine-grained details (Top).

Full, ms/image SWITTI (AR)

KV cache, GB/image

Generator step, ms/image

Disable late CFG

Model

✗ 908 1.2 ✓ 640

2.3

465

0

✗ 735 0 ✓ 540

SWITTI

369

Table 6. Efficiency comparison of architecture and sampling modifications proposed in SWITTI.

### 6. Limitations and future directions

Hierarchical tokenizers. We believe that one of the major limitations of the existing scale-wise models is the inferior hierarchical discrete VAE performance compared to the recent continuous [4, 8, 50, 56] or discrete single-level [56, 68] counterparts.

Typical failure cases are distorted middle/long-shot faces, text rendering and checker-board artifacts on high-frequency textures such as distant foliage or rocky surfaces. We hope that future advances in hierarchical image tokenizers, either discrete or continuous, may significantly improve the performance of scale-wise generative models without using an additional diffusion prior [40, 70].

### 7. Conclusion

We introduce SWITTI, a novel scale-wise generative transformer for text-to-image generation. In contrast to previous next-scale prediction approaches, SWITTI eliminates explicit autoregressive prior and makes use of more effective sampling with guidance. These modifications result in SWITTI generating images as fast as HART, while being three times larger. Trained on a large-scale curated text-image dataset, SWITTI surpasses prior text-conditional visual autoregressive models and achieves up to 7× faster sampling than state-of-the-art text-to-image diffusion models while delivering comparable generation quality.

### References

- [1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization, 2016. 3
- [2] Dmitry Baranchuk, Andrey Voynov, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Label-efficient semantic segmentation with diffusion models. In International Conference on Learning Representations, 2022. 2
- [3] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 1, 2
- [4] Black Forest Labs. Flux.1. https://huggingface. co/black-forest-labs/FLUX.1-dev, 2024. 1, 2, 9
- [5] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. 1
- [6] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 3
- [7] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T. Freeman, Michael Rubinstein, Yuanzhen Li, and Dilip Krishnan. Muse: Text-to-image generation via masked generative transformers, 2023. 2, 3
- [8] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models, 2024. 9
- [9] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 1
- [10] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. 2022. 2
- [11] Sander Dieleman. Diffusion is spectral autoregression, 2024. 2
- [12] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, and Jie Tang. Cogview: Mastering text-to-image generation via transformers. In Advances in Neural Information Processing Systems, 2021. 4
- [13] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 1, 3

- [14] Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image synthesis, 2020. 1, 2, 3
- [15] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and

- Robin Rombach. Scaling rectified flow transformers for highresolution image synthesis. CoRR, abs/2403.03206, 2024. 1, 2, 3, 6, 7
- [16] Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens, 2024. 1, 2, 3
- [17] Ruiqi Gao*, Aleksander Holynski*, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul P. Srinivasan, Jonathan T. Barron, and Ben Poole*. Cat3d: Create anything in 3d with multi-view diffusion models. Advances in Neural Information Processing Systems, 2024. 1
- [18] Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-toimage alignment, 2023. 6
- [19] Jiatao Gu, Yuyang Wang, Yizhe Zhang, Qihang Zhang, Dinghuai Zhang, Navdeep Jaitly, Josh Susskind, and Shuangfei Zhai. Dart: Denoising autoregressive transformer for scalable text-to-image generation, 2024. 3
- [20] Shuai He, Yongchang Zhang, Rui Xie, Dongxiang Jiang, and Anlong Ming. Rethinking image aesthetics assessment: Models, datasets and benchmarks. In IJCAI, pages 942–948,

2022. 1

- [21] Byeongho Heo, Song Park, Dongyoon Han, and Sangdoo Yun. Rotary position embedding for vision transformer. In European Conference on Computer Vision (ECCV), 2024. 3
- [22] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning, 2022. 4, 6, 2
- [23] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 4, 6
- [24] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 2, 5
- [25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1
- [26] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. If you use this software, please cite it as below. 3
- [27] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax, 2017. 7
- [28] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022. 1
- [29] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proc. CVPR, 2024. 1
- [30] Sergey Kastryulin, Artem Konev, Alexander Shishenya, Eugene Lyapustin, Artem Khurshudov, Alexander Tselousov, Nikita Vinokurov, Denis Kuznedelev, Alexander Markovich,

- Grigoriy Livshits, Alexey Kirillov, Anastasiia Tabisheva, Liubov Chubarova, Marina Kaminskaia, Alexander Ustyuzhanin, Artemii Shvetsov, Daniil Shlenskii, Valerii Startsev, Dmitrii Kornilov, Mikhail Romanov, Artem Babenko, Sergei Ovcharenko, and Valentin Khrulkov. Yaart: Yet another art rendering technology, 2024. 6
- [31] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Yuhta Takida, Naoki Murata, Toshimitsu Uesaka, Yuki Mitsufuji, and Stefano Ermon. Pagoda: Progressive growing of a onestep generator from a low-resolution diffusion teacher, 2024. 2
- [32] Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014. 2
- [33] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. 2023. 4, 6
- [34] Shu Kong, Xiaohui Shen, Zhe Lin, Radomir Mech, and Charless Fowlkes. Photo aesthetics ranking network with attributes and content adaptation. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14, pages 662–679. Springer, 2016. 1
- [35] Tuomas Kynkäänniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 5
- [36] Christian Ledig, Lucas Theis, Ferenc Huszár, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, et al. Photorealistic single image super-resolution using a generative adversarial network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4681–4690,

2017. 6

- [37] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization, 2022. 1, 2, 3
- [38] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation,

2024. 1, 2, 6

- [39] Tianhong Li, Huiwen Chang, Shlok Kumar Mishra, Han Zhang, Dina Katabi, and Dilip Krishnan. Mage: Masked generative encoder to unify representation learning and image synthesis, 2023. 2, 3
- [40] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization, 2024. 1, 2, 3, 9
- [41] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. Microsoft coco: Common objects in context, 2015. 6, 1
- [42] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models, 2024. 6

- [43] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining, 2024. 1, 6
- [44] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 1
- [45] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference, 2023. 2
- [46] Xiaoxiao Ma, Mohan Zhou, Tao Liang, Yalong Bai, Tiejun Zhao, Huaian Chen, and Yi Jin. Star: Scale-wise text-toimage generation via auto-regressive representations, 2024. 2, 3
- [47] Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, Natalia Neverova, Andrea Vedaldi, Oran Gafni, and Filippos Kokkinos. Im-3d: Iterative multiview diffusion and reconstruction for high-quality 3d generation. arXiv preprint arXiv:2402.08682, 2024. 1
- [48] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14297–14306, 2023. 2
- [49] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022. 3
- [50] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024. 1, 2, 3, 6, 9
- [51] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dimitry Vengertsev, Edgar Schonfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie gen: A cast of media foundation models,

2024. 1

- [52] Alec Radford and Karthik Narasimhan. Improving language understanding by generative pre-training. 2018. 1, 3
- [53] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. 1, 3

- [54] Alec Radford, Jong Wook Kim, Chris Hallacy, A. Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 3
- [55] Recraft AI. Recraft - ai. https://www.recraft.ai/ about, Accessed 2024. 2
- [56] Fitsum Reda, Jinwei Gu, Xian Liu, Songwei Ge, TingChun Wang, Haoxiang Wang, and Ming-Yu Liu. Cosmos tokenizer: A suite of image and video neural tokenizers. http://https://research.nvidia.com/labs/ dir/cosmos-tokenizer/, 2024. 9
- [57] Severi Rissanen, Markus Heinonen, and Arno Solin. Generative modelling with inverse heat dissipation. In International Conference on Learning Representations (ICLR), 2023. 2
- [58] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 2
- [59] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo-Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems, 2022. 2
- [60] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation, 2023. 2, 6, 7
- [61] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial diffusion distillation, 2024. 2
- [62] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. 1, 2
- [63] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. 3
- [64] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502,

2020. 1

- [65] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.
- [66] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1
- [67] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023. 2
- [68] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 1, 2, 6, 9

- [69] Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and Song Han. Hart: Efficient visual generation with hybrid autoregressive transformer, 2024. 5
- [70] Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and Song Han. Hart: Efficient visual generation with hybrid autoregressive transformer. arXiv preprint, 2024. 2, 3, 6, 9
- [71] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models, 2024. 1
- [72] Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. 1
- [73] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. 2024. 1, 2, 3, 5, 6
- [74] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. 1, 3
- [75] Aaron van den Oord, Oriol Vinyals, and koray kavukcuoglu. Neural discrete representation learning. In Advances in Neural Information Processing Systems. Curran Associates, Inc.,

2017. 3

- [76] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 3
- [77] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In AAAI, 2023. 6
- [78] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data, 2021. 6
- [79] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1, 6
- [80] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600–612, 2004. 6
- [81] Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma with denoising diffusion GANs. In International Conference on Learning Representations,

2022. 1

- [82] Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, and Junyang Lin. Understanding and improving layer normalization, 2019. 3
- [83] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 6
- [84] Yilun Xu, Gabriele Corso, Tommi Jaakkola, Arash Vahdat, and Karsten Kreis. Disco-diff: Enhancing continuous diffusion models with discrete latents. In International Conference on Machine Learning, 2024. 3

- [85] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024. 2, 6, 7
- [86] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024. 2
- [87] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved VQGAN. In International Conference on Learning Representations, 2022. 3
- [88] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-toimage generation, 2022. 1, 2
- [89] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-toimage generation, 2022. 7
- [90] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019. 3
- [91] Qian Zhang, Xiangzi Dai, Ninghua Yang, Xiang An, Ziyong Feng, and Xingyu Ren. Var-clip: Text-to-image generator with visual auto-regressive modeling, 2024. 2, 3
- [92] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6
- [93] Le Zhuo, Ruoyi Du, Xiao Han, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. arXiv preprint arXiv:2406.18583, 2024. 2, 4, 6

## SWITTI: Designing Scale-Wise Transformers for Text-to-Image Synthesis Supplementary Material

### A. Training details

For all our experiments, we use FSDP with a hybrid strategy for effective multi-host training and mixed precision BF16/FP32. To additionally reduce memory usage and speed up the training steps, we use precomputed textual embeddings from the text encoders. We use AdamW optimizer with (β1=0.9, β2=0.95). In the normalized RoPE, we use θ=10,000 and max size 128.

- A.1. Pretraining

Data. Starting from ∼6B image-text pairs from the web, we filter out images of low aesthetic quality, based on the AADB [34] and TAD66k [20] aesthetic filters. We additionally consider sufficiently high-resolution images with at least 512px on each side. The resulting dataset contains central crops of the images with an aspect ratio in [0.75,1.33]. The images are recaptioned using the LLaVA-v1.4-13B, LLaVAv1.6-34B [44], and ShareGPT4V [9] models. The best caption is selected according to OpenCLIP ViT-G/14 [62].

Technical details. During the first stage of pretraining, we train d=30 models on 256×256 resolution for 400K iterations using a batch size of 2,560. This stage takes ∼25K NVIDIA A100 GPU hours. We start with a learning rate of 1e−4 with a linear decay to 1e−5.

Next, we train the models on 512×512 resolution for 200K iterations using a batch size of 768 and a learning rate of 1e−5, linearly decaying to 5e−7. This stage takes another ∼12K NVIDIA A100 GPU hours.

- A.2. Ablation experiments

For the experiments in Section 3.2, we train more lightweight models, with d=20 transformer blocks, resulting in approximately 0.7B parameters. All models are trained for 150,000 iterations with a learning rate of 1e−4, linearly decaying to 1e−5. Batch size is 768. Image resolution is 256×256. For these experiments, we disable the conditioning on the cropping parameters.

For evaluation, we use 30,000 prompts from the COCO2014 validation set [41].

### B. Normalization ablation

In this section, we investigate, how the choice of normalization affects the final model performance. In Table 7, we compare our d=20 version of SWITTI (AR), trained on 256×256 against the same version with switched normalization functions (i.e RMSNorms are used for QK-normalization in attention, and LN for input/output normalization) and against the version without “sandwich” normalization. We find the difference in models’ performance insignificant.

Model (d=20) PickScore ↑ CLIP Score ↑ FID ↓

Switti (AR) 0.206 0.312 10.8 LN ↔ RMSNorm 0.206 0.313 11.1 w/o “sandwich” 0.206 0.311 11.0

Table 7. Ablation of normalization choices for d=20 models trained on 256×256 images.

### C. Training loss analysis

Training loss at 1st scale

Training loss at 2nd scale

5.2

5.1

5.1

5.0

4.9

5.0

4.8

0 50K 100K 150K 200K

0 50K 100K 150K 200K

Training loss at 9th scale

Total training loss

5.55

Switti (AR)

Switti

5.50

5.3

5.45

5.2

5.40

5.1

5.35

5.30

5.0

0 50K 100K 150K 200K

0 50K 100K 150K 200K

Figure 12. Comparison of training losses at various scales on 512×512 resolution for d=30 models.

We find that both models converge to similar total losses at the end of the training, as illustrated in Figure 12. However, SWITTI (AR) has slightly lower values of loss at later scales, while the non-AR version exhibits marginally better performance at the earlier ones. This pattern holds for all configurations of our model during all stages of the training.

This phenomenon combined with our observation in Section 4.2 that around 80% of image tokens can be replaced without affecting the final image visual quality, and the fact that the final non-causal model slightly outperforms its AR counterpart, indicates that the accuracy at the last scales is less crucial.

### D. Additional prompt switching visualizations

In Figures 17 and 18, we provide additional examples of the prompt switching analysis, discussed in Section 3.4.

### E. Visual comparison of the finetuned RQ-VAE

To illustrate the difference between the original RQ-VAE checkpoint and fine-tuned version, we depict several representative examples in Figure 14. One can observe that

the fine-tuned VAE decoder is less prone to reconstruction artifacts and color shifts and produces more contrast images.

### F. Evaluation details

We compute the CLIP Score [22], using features from a pretrained CLIP-ViT-H-14-laion2B-s32B-b79K encoder [62], to assess image-text alignment.

FID is measured on 30K generated images reduced to 512×512 resolution using bicubic interpolation. Then, the resolution is further reduced to 256×256 using Lanczos interpolation following the practices in FID calculation on COCO2014. Real data statistics are collected for all images in the validation sets: ∼40K images in COCO2014 and 30,000 images in MJHQ.

For GenEval, we generate 4 images for each of the 533 evaluation prompts, followed by the original evaluation protocol using Mask2Former [10] with Swin-S backbone as an object detector.

### G. Human evaluation setup

The evaluation is performed using Side-by-Side (SbS) comparisons, i.e., the assessors are asked to make a decision between two images given a textual prompt. For each evaluated pair, three responses are collected and the final prediction is determined by majority voting.

The human evaluation is performed by professional assessors. They are officially hired, paid competitive salaries, and informed about potential risks. The assessors have received detailed and fine-grained instructions for each evaluation aspect and passed training and testing before accessing the main tasks.

In our human preference study, we compare the models in terms of four aspects: relevance to a textual prompt, presence of defects, image aesthetics, and complexity. Figures 19 to 22 present the interface for each of these criteria. Note that the selected answers on the images are random.

### H. Visual comparison against T2I models

We provide qualitative comparison of SWITTI against the baselines considered in this work in Figure 15 and Figure 10.

### I. Effect of disabling CFG at different scales

In Figure 16, we provide some examples of disabling CFG at various level ranges. One can observe that presence of CFG at first scales improves image quality and relevance. At the same time, it can be turned off at the last scales without noticeable quality degradation or loss of details.

### J. List of prompts used in our figures

##### Figure 1 prompts

1) “Cute winter dragon baby, kawaii, Pixar, ultra detailed, glacial background, extremely realistic.“

Full, s/image Distilled Diffusion Models

Generator size, B

1 step, ms/image

Model

N steps

SDXL-Turbo 2.6 4 12.4 0.25 DMD2 2.6 4 12.4 0.25

Diffusion Models

SDXL 2.6 25 12.4 0.87 SD3-medium 2.0 28 16.8 0.93

Autoregressive Models

Lumina-mGPT 7.0 1024 — 224.2 LlamaGen 0.8 1024 — 3.82 HART 0.7 10 4.7* 0.06 SWITTI (AR) 2.5 10 11.2* 0.14

SWITTI 2.5 10 9.5* 0.13

*Time averaged over 10 steps.

Table 8. Comparison of 512×512 image generation times.

- 2) “A lizard that looks very much like a man, with de-

veloped muscles, leather armor with metal elements, in the hands of a large trident decorated with ancient runes, against the background of a small lake, everything is well drawn in the style of fantasy”

- 3) “An ancient ruined archway on the moon, fantasy,

ruins of an alien civilization, concept art, blue sky, reflection in water pool, large white planet rising behind it”

- 4) “Cat as a wizard”
- 5) “The Mandalorian by masamune shirow, fighting

stance, in the snow, cinematic lighting, intricate detail, character design”

- 6) “a human face wearing a massive aztec headdress.

With hyper intricate war paint on it’s face, detailed texture, vibrant colors, solid background, hyper realistic, 8K“

- 7) “32 – bit pixelated future Hiphop producer in glowing

power street ware, noriyoshi ohrai, in the style of minecraft tomer hanuka.“

- 8) “Portrait of an alien family from the 1970’s, futur-

istic clothes, absurd alien helmet, straight line, surreal, strange, absurd, photorealistic, Hasselblad, Kodak, portra 800, 35mm lens, F 2.8, photo studio.”

##### Figure 11 prompts

- 1) “the girl is 36 years old. hairstyle square, blonde hair,

slightly curly. light blue jeans. the top is a classic mintcolored jacket. a brooch in the form of a bird is attached to the jacket. there are silver earrings in her ears. a small leather shoulder bag in dark brown color.”

- 2) “a squirrel and an acorn”
- 3) “Baby Yoda Walking in Manhattan.”

##### Figure 16 prompts

- 1) “a drawing of a house on a mountain”
- 2) “a wooden jewelry box and a fabric rug”

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Figure 13. Visual comparison of 512×512 image reconstruction between an original RQ-VAE (left) and the one with a fine-tuned decoder (right).

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Figure 14. Visual comparison of 1024×1024 image reconstruction between an original RQ-VAE (left) and the one with a fine-tuned decoder (right).

[Figure 25]

###### Figure 15. Qualitative comparison of SWITTI against the baselines.

No CFG CFG at scales 0 1 CFG at scales 0 2 CFG at scales 0 3 CFG at scales 0 5

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

CFG at scales 0 7 CFG at scales 0 9 CFG at scales 0 11 CFG at scales 0 12 CFG at all scales

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

No CFG CFG at scales 0 1 CFG at scales 0 2 CFG at scales 0 3 CFG at scales 0 5

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

CFG at scales 0 7 CFG at scales 0 9 CFG at scales 0 11 CFG at scales 0 12 CFG at all scales

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Figure 16. Impact of CFG at different resolutions. There are 14 scales: higher index indicates higher resolution. The guidance scale is 6.

100% 20% 80% 40% 60% 50% 50%

[Figure 46]

100%

60% 40% 80% 20% 90% 10%

An image of a cat, full-face, 8k An image of a crocodile, full-face, 8k

Figure 17. Impact of the prompt switching during image generation.

100% 20% 80% 40% 60% 50% 50%

[Figure 47]

60% 40% 80% 20% 90% 10% 100%

Forest, winter, snowfall, 8k Forest, summer, flowers, bright sky, 8k

Figure 18. Impact of the prompt switching during image generation.

[Figure 48]

[Figure 49]

- Figure 19. Human evaluation interface for aesthetics.

[Figure 50]

[Figure 51]

- Figure 20. Human evaluation interface for defects.

[Figure 52]

[Figure 53]

Figure 21. Human evaluation interface for relevance.

[Figure 54]

Figure 22. Human evaluation interface for image complexity.

