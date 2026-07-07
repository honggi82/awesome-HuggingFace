arXiv:2411.15098v6[cs.CV]7Jul2025

# OminiControl: Minimal and Universal Control for Diffusion Transformer

Zhenxiong Tan Songhua Liu Xingyi Yang Qiaochu Xue Xinchao Wang National University of Singapore

{zhenxiong, songhua.liu, xyang, e1352520}@u.nus.edu xinchao@nus.edu.sg

[Figure 1]

[Figure 2]

[Figure 3]

skiing

[Figure 4]

sunglasses; beach holding a sign saying "Omini Control!"

[Figure 5]

[Figure 6]

[Figure 7]

on the mirror mountain background in the snow

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Trump holding beer girl in a café guy on a street

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Canny

Deblurring Out-painting

[Figure 22]

[Figure 23]

[Figure 24]

Depth

Colorization

In-painting

Figure 1. Results of our OminiControl on both subject-driven generation (top) and spatially-aligned tasks (bottom). The small images in red boxes show the input conditions.

## Abstract

We present OminiControl, a novel approach that rethinks how image conditions are integrated into Diffusion Transformer (DiT) architectures. Current image conditioning methods either introduce substantial parameter overhead or handle only specific control tasks effectively, limiting their practical versatility. OminiControl addresses these limitations through three key innovations: (1) a minimal architectural design that leverages the DiT’s own VAE encoder and transformer blocks, requiring just 0.1% additional parameters; (2) a unified sequence processing strategy that combines condition tokens with image tokens for flexible token interactions; and (3) a dynamic position encoding mechanism that adapts to both spatially-aligned and non-aligned control tasks. Our extensive experiments show that this streamlined approach not only matches but surpasses the performance of specialized methods across multiple conditioning tasks. To overcome data limitations in

subject-driven generation, we also introduce Subjects200K, a large-scale dataset of identity-consistent image pairs synthesized using DiT models themselves. This work demonstrates that effective image control can be achieved without architectural complexity, opening new possibilities for efficient and versatile image generation systems.

## 1. Introduction

The ability to generate high-quality images with precise user control remains a central challenge in computer vision. While diffusion models [17, 44] have significantly advanced image generation, surpassing traditional GANbased approaches [11] in both quality and diversity, finegrained control remains problematic. Text-conditioned models [9, 24, 41, 44] serve as the primary controllable generation paradigm, yet fundamentally lack the capacity to specify exact spatial details and visual attributes that users often need. To address this limitation, some works have

explored image-based control methods [2, 14, 15, 18, 26– 30, 37, 42, 46, 50, 52, 54, 56, 57, 57, 60, 61, 63], enabling users to specify their intentions through reference images or visual hints, offering more precise guidance than text alone [44].

However, current image control methods face several key challenges: First, existing methods require substantial architectural modifications with dedicated control modules [38, 60, 61]. Second, these approaches show clear task bias – they typically work well for either spatially aligned controls (e.g., edge-guided or depth-guided generation [2, 15, 29, 30, 37, 42, 52, 61, 63]) or spatially unaligned ones (e.g., style transfer or subject-driven generation [18, 27, 28, 50, 60][14, 33]), but rarely both. Third, current approaches are built primarily for UNet architectures [45]. These approaches yield suboptimal results when applied to newer Diffusion Transformer (DiT) models [40] due to fundamental architectural differences, despite the latter’s superior generation capabilities [4, 9, 24].

These challenges, particularly within the emerging paradigm of Diffusion Transformer models, motivate us to rethink the fundamental approach to image control. We propose OminiControl , an omni-capable yet minimal framework that achieves effective and flexible control.

For minimal architecture, OminiControl employs a parameter reuse strategy that leverages DiT’s existing components—particularly its VAE encoder and transformer blocks—which already possess the necessary capabilities to process visual control signals. By reusing these components with minimal fine-tuning, our approach dramatically reduces parameter overhead while maintaining control effectiveness, requiring only 0.1% additional parameters compared to the base model.

To achieve omni-capability across diverse tasks, OminiControl introduces a unified sequence processing approach that fundamentally differs from previous methods’ rigid feature addition. By directly concatenating condition tokens with noisy image tokens in a unified sequence, we allow the multi-modal attention mechanism to discover appropriate relationships between tokens—whether spatial or semantic—without imposing artificial constraints. This flexibility is crucial for handling both spatially-aligned tasks like edge-guided generation and non-aligned tasks like subjectdriven generation within a single framework.

Building on this unified approach, OminiControl incorporates a dynamic positioning strategy that adaptively assigns position indices based on the task type, enabling true omni-capability without task-specific architectural modifications. Furthermore, for practical flexibility, we introduce an attention bias mechanism that allows users to dynamically adjust the influence of image conditions at inference time, providing crucial control over the generation process.

The effectiveness of our architecture depends on high-

quality training data, particularly for subject-driven generation. Recognizing that state-of-the-art DiT models can generate remarkably consistent image pairs of the same subject, we develop an automated data synthesis pipeline that creates Subjects200K—a dataset of over 200,000 diverse, identity-consistent images that provides the rich training signals needed to fully realize OminiControl ’s potential.

In summary, we highlight our contributions as follows:

- • We propose OminiControl , a minimal and universal control framework for DiT models that requires only 0.1% additional parameters while effectively handling both spatially-aligned and non-aligned tasks, demonstrating that extensive architectural modifications are unnecessary for effective image conditioning.
- • We identify two key technical innovations that enable omni-capability: (1) unified sequence processing that outperforms traditional feature addition, and (2) adaptive position encoding that strategically assigns position indices based on task requirements.
- • We design a flexible attention bias mechanism that allows precise adjustment of conditioning strength at inference time, enhancing practical control within the multi-modal framework without compromising performance.
- • We develop and release Subjects200K, a large-scale dataset containing over 200,000 identity-consistent images to advance future research.

## 2. Related works

### 2.1. Diffusion models

Diffusion models have emerged as a powerful framework for image generation[17, 44], demonstrating success across diverse tasks including text-to-image synthesis [4, 44, 49], image-to-image translation [48], and image editing [1, 35]. Recent advances have led to significant improvements in both quality and efficiency, notably through the introduction of latent diffusion models [44]. To further enhance generative capabilities, large-scale transformer architectures have been integrated into these frameworks, leading to advanced models like DiT[4, 5, 24, 40]. Building on these architectural innovations, FLUX[24] incorporates transformerbased design with flow matching objectives [31], achieving state-of-the-art generation performance.

### 2.2. Controllable generation

Controllable generation has been extensively studied in the context of diffusion models. Text-to-image models[41, 44] have established a foundation for conditional generation, while various approaches have been developed to incorporate additional control signals such as image. Notable methods include ControlNet [61], enabling spatially aligned control in diffusion models, and T2I-Adapter [37], which improves efficiency with lightweight adapters. UniCon-

trol [42] uses Mixture-of-Experts (MoE) to unify different spatial conditions, further reducing model size. However, these methods rely on spatially adding condition features to the denoising network’s hidden states, inherently limiting their effectiveness for spatially non-aligned tasks like subject-driven generation. IP-Adapter [60] addresses this by introducing cross-attention through an additional encoder, and SSR-Encoder [62] further enhances identity preservation in image-conditioned tasks. Despite these advances [10, 14, 18, 23, 27, 32, 33, 38, 47], a unified solution for both spatially aligned and non-aligned tasks remains elusive.

## 3. Methods

### 3.1. Preliminary

The Diffusion Transformer (DiT) model [40], employed in architectures like FLUX.1 [24], Stable Diffusion 3 [44], and PixArt [4], uses transformer as denoising network to refine noisy image tokens iteratively.

A DiT model processes two types of tokens: noisy image tokens X ∈ RN×d and text condition tokens CT ∈ RM×d, where d is the embedding dimension, N and M are the number of image and text tokens respectively (Figure 2a). Throughout the network, these tokens maintain consistent shapes as they pass through multiple transformer blocks.

In FLUX.1, each DiT block consists of layer normalization followed by Multi-Modal Attention (MMA) [39], which incorporates Rotary Position Embedding (RoPE) [51] to encode spatial information. For image tokens X, RoPE applies rotation matrices based on the token’s position (i,j) in the 2D grid:

Xi,j → Xi,j · R(i,j), (1)

where R(i,j) is the rotation matrix at position (i,j). Text tokens CT undergo the same transformation with their positions set to (0,0).

The multi-modal attention mechanism then projects the position-encoded tokens into query Q, key K, and value V representations. It enables the computation of attention between all tokens:

MMA([X;CT]) = softmax

QK⊤ √

d

V, (2)

where [X;CT] denotes the concatenation of image and text tokens. This formulation enables bidirectional attention.

### 3.2. OminiControl

Building upon the DiT architecture, with FLUX.1 [24] as our base model, we aim to develop a minimal, omnicapable and controllable generation framework that accepts flexible control signals. This vision leads to our OminiControl , which we describe in this section.

#### 3.2.1. Minimal architecture

To minimize the extra architectural and parameter overhead, OminiControl adopts a parameter reuse strategy.

As its name implies, we reuses the VAE [22, 44] encoder from the base DiT model encode the condition image. These images are projected into the same latent space as the noisy input tokens, ensuring compatibility without introducing new modules. Following this, OminiControl processes both noisy image tokens and condition tokens jointly through the original DiT blocks. To adapt the shared DiT blocks for handling condition tokens, only lightweight LoRA fine-tuning [8] is applied, avoiding costly full-parameter updates.

This approach contrasts with previous methods that rely on separate feature extractors like CLIP [43, 60] or additional control modules [62], significantly reducing architectural complexity. Meanwhile, LoRA fine-tuning ensures parameter efficiency compared to duplicating the entire network as done in ControlNet [61].

#### 3.2.2. Omni-capable token interaction

To achieve effective control across diverse tasks, OminiControl needs to enable flexible interactions between condition tokens and image tokens. We address this challenge through two key mechanisms: unified sequence processing and position-aware token interaction.

Unified sequence processing. Building upon the shared latent space, we now think how to integrate condition tokens into the model for flexible control across.

Previous methods [37, 61] incorporate condition images through direct feature adding:

, (3) where the condition features hC

##### hX ← hX + hC

I

are spatially aligned and added to the noisy image features hX.

I

We first implement this direct adding approach as illustrated in Figure 2a. While the bare effectiveness is shown in Figure 2b, this approach faces two limitations: (1) it lacks flexibility for non-aligned scenarios where spatial correspondence doesn’t exist, and (2) the rigid addition operation constrains potential interactions between condition and image tokens.

We then explored the unified sequence processing to integrate condition tokens into the model. Specifically, this approach directly concatenates condition tokens with noisy image tokens [X;CT;CI] for multi-modal attention processing. This formulation enables flexible token interactions through DiT’s multi-modal attention mechanism, allowing direct relationships to emerge between any pair of tokens without imposing rigid spatial constraints.

As shown in Figure 4, this approach effectively handles both spatially aligned and non-aligned tasks, with attention

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

|module input<br><br>module output<br><br>module training module|
|---|

,

,

, ,

|Poor match|
|---|

[Figure 29]

norm norm

norm

norm

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
|Te Embe<br><br>N|xt dder|

| | |
|---|---|
|Im<br><br>DiT B<br><br>Pro|age<br><br>locks<br><br>jectio|

MM Attention

MM Attention

MM Attention

MM Attention

MLP

MLP

MLP

MLP

Direct Adding

Embedder

|concat| |
|---|---|
| | |

|concat| |
|---|---|
| | |

|Perfect match|
|---|

[Figure 30]

concat

concat

D

Out Projection

Out Projection

Out Projection

Out Projection

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Out n

Unified sequence

|,|
|---|

|,|
|---|

|, ,<br><br>|
|---|

|Noise|
|---|

|[Figure 31]|
|---|

(b) Original No image condition

(c) Integrate image condition by Direct Adding

(d) Integrate image condition by Unified Sequence

"camel"

(a) Diffusion Transformer

(a) Overview of the Diffusion Transformer (DiT) architecture and integration methods for image conditioning.

(b) Comparison of results

Figure 2. Exploration of different methods for integrating image conditions.

Loss for different pos.

Integration methods loss.

[Figure 32]

[Figure 33]

1.0

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | |Sh|are|d|Po|s.| | |
| | | | | | | | | | |
| | | |Sh|ifti|ng|P|os|.| |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

Direct Addition

|[Figure 34]| |
|---|---|
| | |

1.5

MM Attention

0.8

Loss

Loss

1.0

- 0.6

Input Condition Output Image

[Figure 35]

[Figure 36]

0.4

0.5

0.2

100 400 1k Training Steps

0 2500 5000 7500 10000 Training Steps

(a) Training losses for different image condition integration methods.

(b) Training loss for shared vs. shifting position. (subject-driven)

(a) Canny to image (b) Subject driven generation

Figure 3. Training loss comparisons.

maps revealing clear cross-token relationships and interaction patterns. Empirically, this unified sequence approach consistently achieves lower training loss compared to direct feature adding (Figure 3a), demonstrating its superior conditioning capability across diverse generation scenarios.

Position-aware token interaction. While the unified sequence approach allows flexible token interactions, encoding position information for the newly appended conditional tokens is not straightforward.

FLUX.1 adopts the RoPE mechanism to encode spatial information for both image and text tokens. Specifically, for a 512×512 input image, the VAE encoder produces a 32×32 grid of latent tokens, with each token assigned a position index (i,j) where i,j ∈ [0,31].

We first assign the same position indices to condition tokens as noisy image tokens, it works well for spatially aligned tasks like edge-guided generation. However, for non-aligned tasks like subject-driven generation, this shared position indexing can lead to suboptimal performance due to spatial overlap between condition and noisy image tokens. But if we shift the position indices of condition tokens

- Figure 4. (a) Attention maps for the Canny-to-image task (with setting from Figure 2b), showing interactions between noisy im-

age tokens X and image condition tokens CI. Strong diagonal patterns indicate effective spatial alignment. (b) Subject-driven generation task, with input condition and output image. (Prompt: This item is placed on a table with Christmas decorations around it.) Attention maps for X → Ci and Ci → X illustrate accurate subject-focused attention.

[Figure 37]

- Figure 5. The results from models with shared and shifted position indices. Both models are fully trained with 15k iterations.

by a fixed offset ∆ (e.g., (0,32)), ensuring no spatial overlap with noisy image tokens, the training convergence accelerates significantly and final performance improves. (see Figure 3b and Figure 5).

This observation suggests that for spatially aligned tasks, a shared position indexing facilitates direct spatial correspondence between condition and image tokens, while

for non-aligned tasks, shared position indexing can constrain the model’s ability to establish semantic relationships. Hence, we propose a dynamic positioning strategy based on the control task:

##### (i,j)C

I

=

(i,j)X for aligned tasks (i,j) + ∆ for non-aligned tasks

(4)

which makes OminiControl truly omni-capable by adapting to both spatially aligned and non-aligned control tasks.

#### 3.2.3. Control with flexibility

Although our unified sequence processing and positionaware token interaction mechanisms effectively enable joint attention between conditions and images, they also present a new challenge. Unlike previous methods [60, 61] that could simply scale condition features (e.g., hX ← hX + α · hC

I

where α controls strength), the joint attention approach of OminiControl does not inherently support adjustable conditioning strength, which is crucial for practical applications where users need to balance text and image influences.

To address this limitation, we design a flexible control mechanism by introducing a bias term into the multi-modal attention computation. Specifically, for a given strength factor γ, we modify the attention operation in Equation 2 to:

MMA([X;CT;CI]) = softmax

QK⊤ √

+ B(γ) V, (5)

d

where B(γ) is a bias matrix modulating the attention between concatenated tokens [X;CT;CI]. Given CT ∈ RM×d and X,CI ∈ RN×d, the bias matrix has the structure:

 

 . (6)

- 0M×M 0M×N log(γ)1N×N
- 0N×M 0N×N 0N×N

B(γ) =

log(γ)1N×N 0M×N 0N×N

This formulation preserves the original attention patterns within each token type while scaling attention weights between X and CI by log(γ). At test time, setting γ = 0 removes the condition’s influence, while γ > 1 enhances it, which makes OminiControl more flexible.

#### 3.2.4. Comparison with concurrent works

Several recent works also explore controllable image generation with DiT models. Some approaches [12, 24, 34] employ channel concatenation to integrate condition tokens, which offers less flexibility than our unified sequence processing. Others [3, 6] focus exclusively on specific tasks such as style transfer or subject-driven generation, limiting their flexibility and generality. In contrast, our approach combines dynamic positioning with a flexible control mechanism to provide a more comprehensive and adaptable control framework compared to existing methods.

[Figure 38]

Figure 6. Examples from our Subjects200K dataset. Each pair of images shows the same object in varying contexts.

### 3.3. Subjects200K datasets

A critical challenge in developing universal control frameworks lies in obtaining high-quality training data for subject-driven generation. While OminiControl ’s architecture enables flexible control, effective training requires data that maintains subject consistency while incorporating natural variations in pose, lighting, and context.

Existing solutions often rely on identical image pairs [60] or limited-scale datasets [23, 47], which present several limitations. Using identical pairs can lead to overfitting in our framework, causing the model to simply reproduce the input. Meanwhile, existing datasets with natural variations often lack either the scale or diversity needed for robust training.

To address this data challenge, we leverage a key observation: state-of-the-art DiT models like FLUX.1 [25] can generate identity-consistent image pairs when provided with appropriate prompts [19, 20]. Building on this, we develop an pipeline to create Subjects200K, a large-scale dataset specifically designed for subject-driven generation1:

- • Prompt Generation. We use GPT-4o to generate over 30,000 diverse subject descriptions. Each description represents the same subject in multiple scenes.
- • Paired-image synthesis. We then reorganized the collected descriptions into structured prompts. Each prompt describes the same subject in two different scenes. The template of such prompt is shown in the Figure S3 of supplementary material.These prompts are then fed into FLUX to generate image pairs. Each pair is designed to maintain subject consistency while varying in context.
- • Quality Assesment. Finally, we use GPT-4o to evaluate the generated pairs. Misaligned pairs are removed to ensure identity consistency and high image quality.

The resulting dataset (Figure 6) contains over 200,000 high-quality images spanning diverse categories. Each subject appears in multiple contexts, providing rich training signals for learning robust subject-driven control. To facilitate future research, we release both the dataset and our complete generation pipeline2.

- 1Detailed prompt design and synthesization pipeline are provided in the

Section A of supplementary material.

- 2Dataset and code available at supplementary material.

## 4. Experiment 4.1. Setup

Tasks and base model. We evaluate our method on two categories of conditional generation tasks: spatially aligned tasks (including Canny-to-image, depth-to-image, maskedbased inpainting, and colorization) and subject-driven generation. We build our method upon FLUX.1 [24], a latent rectified flow transformer model for image generation. By default, we use FLUX.1-dev to generate images for spatially aligned tasks. In subject-driven generation tasks, we switch to FLUX.1-schnell as we observed it tend to produce better visual quality.

Implement details. Our method utilizes LoRA [8]for fine-tuning the base model with a default rank of 4. To preserve the model’s original capabilities and achieve flexibility, the LoRA scale is set to 0 by default when processing non-condition tokens.

Training. Our model is trained with a batch size of 1 and gradient accumulation over 8 steps (effective batch size of 8). We employ the Prodigy optimizer [36] with safeguard warmup and bias correction enabled, setting the weight decay to 0.01. For spatially aligned tasks, we use text-to-image-2M[13] dataset with the last 300,000 images. For subject-driven generation, we utilize our proposed Subjects200K dataset. The experiments are conducted on 2 NVIDIA H100 GPUs (80GB each). For spatially aligned tasks, models are trained for 50,000 iterations, while subject-driven generation models are trained for 15,000 iterations.

Baselines. For spatially aligned tasks, we compare our method with both the original ControlNet [61] and T2IAdapter [37] on Stable Diffusion 1.5, as well as ControlNetPro [25], the FLUX.1 implementation of ControlNet. For subject-driven generation, we compare with IPAdapter [60], evaluating its implementations FLUX.1 [58]. Additionally, we also with the official FLUX.1 Tools [24] implementation.

Evaluation of spatially aligned tasks. We evaluate our model on both spatially aligned tasks and subject-driven generation. For spatially aligned tasks, we assess two aspects: generation quality and controllability. Generation quality is measured using FID [16], SSIM, CLIP-IQA[53], MAN-IQA [59], MUSIQ [21], and PSNR[7] for visual fidelity, along with CLIP Text and CLIP Image scores [43] for consistency. For controllability, we compute F1 Score between extracted and input edge maps in edge-conditioned generation, and MSE between extracted and original condition maps for other tasks (using Depth Anything for depth, color channel separation for colorization, etc.). We use 5,000 images from COCO 2017 validation set, and resize them to 512×512, then generate task-specific conditions and associated captions as prompts with a fixed seed of 42.

Methods Base model Parameters Ratio ControlNet

361M ∼42% T2I-Adapter 77M ∼9.0% IP-Adapter 449M ∼52.2%

SD1.5 / 860M

ControlNet

3.3B ∼27.5%

FLUX.1 / 12B

IP-Adapter 918M ∼7.6% FLUX.1 Tools 612M ∼5.1%

14.5M / 48.7M w/ Encoder

∼0.1% / ∼0.4% w/ Encoder

Ours FLUX.1 / 12B

Table 1. Additional parameters introduced by different image conditioning methods. For IP-Adapter, the parameter count includes the CLIP Image encoder. For our method, we also report results when using the original VAE encoder from FLUX.1.

Evaluation of subject-driven generation. For subjectdriven generation, we propose a five-criteria framework evaluating both preservation of subject characteristics (identity preservation, material quality, color fidelity, natural appearance) and accuracy of requested modifications, with all assessments conducted through GPT-4o’s vision capabilities to ensure systematic evaluation. Details are presented in the Section B.2 of supplementary material. We test on 750 text-condition pairs (30 subjects × 25 prompts) from DreamBooth[47] dataset with 5 different seeds, using one selected image per subject as the condition.

### 4.2. Main result

Spatially aligned tasks As shown in Table 2, we comprehensively evaluate our method against existing approaches on five spatially aligned tasks. Our method achieves the highest F1-Score of 0.38 on depth-to-image generation, significantly outperforming both SD1.5-based methods ControlNet [61] and T2I-Adapter [37], as well as FLUX.1based ControlNetPro [25]. In terms of general quality metrics, our approach demonstrates consistent superiority across most tasks, showing notably better performance in SSIM [55], CLIP-IQA[53], MAN-IQA [59], MUSIQ [21] and PSNR[7] scores. For challenging tasks like deblurring and colorization, our method achieves substantial improvements: the MSE is reduced by 77% and 93% respectively compared to ControlNetPro, while the FID scores [16] improve from 30.38 to 11.49 for deblurring. The CLIPText and CLIP-Image metrics [43] indicate that our method maintains high consistency across all tasks, suggesting effective preservation of semantic alignment and image alignment while achieving better control and visual quality. As shown in Figure 7, our method produces sharper details and more faithful color reproduction in colorization tasks, while maintaining better structural fidelity in edge-guided generation and deblurring scenarios.

Subject driven generation Figure 8 presents a comprehensive comparison against existing baselines. Our method demonstrates superior performance, particularly in identity preservation and modification accuracy. Averaging over

[Figure 39]

- Figure 7. Qualitative comparison. Left: Spatially aligned tasks. Right: Subject-driven generation with beverage can and shoes. Our method demonstrates superior controllability and quality across all tasks. (More results are provided in supplementary materials.)

Task Methods / Setting Base Model

Controllability Image Quality Alignment

F1↑ / MSE↓ FID↓ SSIM ↑ CLIP-IQA↑ MAN-IQA↑ MUSICQ↑ PSNR ↑ CLIP Text↑ CLIP Image↑

Canny

ControlNet

SD1.5

0.35 18.74 0.36 0.65 0.45 67.81 10.27 0.305 0.752 T2I Adapter 0.22 20.06 0.35 0.57 0.39 67.89 9.53 0.305 0.748

ControlNet Pro

FLUX.1

0.21 98.69 0.25 0.48 0.37 56.91 9.22 0.192 0.537 Flux Tools 0.20 22.13 0.32 0.66 0.60 75.47 9.69 0.308 0.701 Ours 0.50 24.20 0.45 0.66 0.62 74.87 11.34 0.305 0.785

Depth

ControlNet

SD1.5

923 23.03 0.34 0.64 0.47 70.73 10.63 0.308 0.726 T2I Adapter 1560 24.72 0.28 0.61 0.39 69.99 9.50 0.309 0.721

ControlNet Pro

FLUX.1

2958 62.20 0.26 0.55 0.39 66.85 9.38 0.212 0.547 Flux Tools 767 24.56 0.32 0.68 0.59 75.30 10.15 0.308 0.715 Ours 537 31.04 0.39 0.68 0.60 74.04 10.53 0.305 0.749

Mask

ControlNet SD1.5 7588 13.14 0.68 0.58 0.42 67.22 18.96 0.300 0.848

Flux Tools

FLUX.1

6610 11.40 0.73 0.56 0.45 68.92 18.37 0.305 0.874 Ours 6351 10.20 0.78 0.59 0.49 70.78 19.59 0.305 0.892

Colorization

ControlNet Pro

FLUX.1

994 30.38 0.75 0.40 0.31 54.38 16.23 0.279 0.781

Ours 73 10.37 0.92 0.56 0.48 69.40 21.56 0.305 0.884 Deblur

ControlNet Pro

FLUX.1

338 16.27 0.64 0.55 0.43 70.95 20.53 0.294 0.853 Ours 62 18.89 0.58 0.59 0.54 70.98 21.82 0.301 0.856

- Table 2. Quantitative comparison with baseline methods on five spatially aligned tasks. We evaluate methods based on Controllability (F1-Score for Canny, MSE for others), Image Quality (SSIM, FID, CLIP-IQA, MAN-IQA, MUSIQ, PSNR), and Alignment (CLIP Text and CLIP Image). For F1-Score (used in Canny to Image task), higher is better; for MSE, lower is better. Best results are shown in bold.The second-best results are highlighted with underlines.

Identity preservation

Material quality Color fidelity

Natural appearance

Modification accuracy

0

25

50

75

100

Average over 5 random seeds

Identity preservation

Material quality Color fidelity

Natural appearance

Modification accuracy

0

25

50

75

100

Best score over 5 random seeds

IP-Adapter (SD 1.5) SSR-Encoder IP-Adapter (FLUX) Ours

- Figure 8. Radar charts visualization comparing our method (blue) with baselines across five evaluation metrics.

15.8 and 18.0 percentage points, demonstrating effective subject-fidelity editing. These quantitative results are further corroborated by user studies presented in supplementary material B.2.

Paramter efficiency As shown in Table 1, our approach achieves remarkable parameter efficiency compared to existing methods. For the 12B parameter FLUX.1 model, our method requires only 14.5M trainable parameters (approximately 0.1%), which is significantly lower than ControlNet (27.5%) and IP-Adapter (7.6%). Even when utilizing the original VAE encoder from FLUX.1, our method still maintains high efficiency with just 0.4% additional parameters, demonstrating the effectiveness of our parameter-efficient design.

Condition strength factor We evaluate our condition strength control mechanism (Section 3.2.3) through qualitative experiments. Figure 9 shows generated results with varying strength factor γ. Results show that γ effectively controls the generation process for both spatially aligned tasks like depth-to-image generation and non-aligned tasks

random seeds, we achieve 75.8% modification accuracy compared to IP-Adapter (FLUX)’s 57.7%, while maintaining 50.6% identity preservation against IP-Adapter (SD

- 1.5)’s 29.4%. The advantage amplifies in best-seed scenarios, achieving 90.7% modification accuracy and 82.3% identity preservation - surpassing the strongest baselines by

|[Figure 40]<br><br>Depth<br><br>A portrait of a woman.|
|---|

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

|[Figure 45]<br><br>Subject<br><br>A can of beer in the snow.|
|---|

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Condition

Outputs

Figure 9. Demonstration of the condition strength control.

Study Setting FID ↓ SSIM ↑ F1 Score ↑ CLIP Score ↑

- 1 21.09 0.412 0.385 0.765

- 2 21.28 0.411 0.377 0.751 4 20.63 0.407 0.380 0.761

LoRA Rank

8 21.40 0.404 0.3881 0.761

16 19.71 0.425 0.407 0.764 Condition Blocks

Early 25.66 0.369 0.23 0.72 Full 20.63 0.407 0.38 0.76

- Table 3. Ablation studies on (1) LoRA rank for the Canny-toimage task and (2) condition signal integration approaches. Results show that LoRA rank of 16 and full-depth integration achieve the best performance. Rows with blue background indicate our default settings (LoRA rank=4, Full condition integration). Best results are in bold.

like subject-driven generation, enabling flexible control over the condition’s influence.

- 4.3. Ablation Studies

To better understand the key factors that influence our model’s control capabilities, we conducted comprehensive ablation studies examining parameter efficiency, architectural decisions, and component contributions.

Impact of LoRA rank. We conducted extensive experiments with different LoRA ranks (1, 2, 4, 8, and 16) for the Canny-to-image task. As shown in Table 3, our experiments show that increasing the LoRA rank generally improves model performance, with rank 16 achieving the best results across multiple aspects. However, even with smaller ranks (e.g., rank 1), the model demonstrates competitive performance, especially in text-image alignment, showing the efficiency of our approach even with limited parameters.

Conditioning depth. FLUX.1’s transformer architecture features two distinct types of blocks: early blocks that employ separate normalization modules for different modalities tokens (text and image) and later blocks that share unified normalization across all tokens. As shown in Table 3, experiments reveal that restricting condition signal integration to only these early blocks results in insufficient controllability. This suggests that allowing the condition signals to influence the entire transformer stack is crucial for generate better results. Notably, this finding indicates that the preview approaches[25, 37, 58, 60, 61] of injecting condition signals primarily in early blocks, which were ef-

[Figure 50]

Figure 10. Ablation study of critical modules for conditional generation. Given the censored Mona Lisa image and text prompt (bottom left), we test removing LoRA from different components.

fective in UNet-based architectures, may not fully translate to DiT-based models like FLUX.1.

Critical module analysis. To identify essential components for effective control, we conducted fine-grained ablation studies as shown in Figure 10. Results demonstrate that normalization layers and attention projections—particularly query (WQ) and key (WK)—are critical for maintaining control quality. Removing LoRA from these components significantly degrades conditional rendering, while value projections (WV ) have less impact. These insights reveal that control signals might propagate through normalization pathways and attention routing mechanisms rather than feature transformation processes.

## 5. Conclusion

OminiControl offers parameter-efficient image-conditional control for DiT across diverse tasks using a unified token approach, requiring only 0.1% additional parameters. The Subjects200K dataset — featuring over 200,000 highquality, subject-consistent images—further supports advancements in subject-driven generation, with experimental results confirming OminiControl ’s effectiveness in both spatially-aligned and non-aligned tasks.

However, the unified sequence approach increases the total number of tokens processed through the network, potentially limiting computational efficiency during inference. Addressing this token efficiency challenge while maintaining our method’s control capabilities represents an important direction for future research in parameter-efficient conditional generation.

## 6. Acknowledgment

We would like to acknowledge that the computational work involved in this research work is partially supported by NUS IT’s Research Computing group using grant numbers NUSREC-HPC-00001.

## References

- [1] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18208–18218, 2022. 2
- [2] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. 2023. 2
- [3] Shengqu Cai, Eric Chan, Yunzhi Zhang, Leonidas Guibas, Jiajun Wu, and Gordon Wetzstein. Diffusion self-distillation for zero-shot customized image generation. arXiv preprint arXiv:2411.18616, 2024. 5
- [4] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 2, 3
- [5] Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, and Zhenguo Li. Pixart-{\delta}: Fast and controllable image generation with latent consistency models. arXiv preprint arXiv:2401.05252, 2024. 2
- [6] Jooyoung Choi, Chaehun Shin, Yeongtak Oh, Heeseung Kim, and Sungroh Yoon. Style-friendly snr sampler for styledriven generation. arXiv preprint arXiv:2411.14793, 2024. 5
- [7] Wikipedia contributors. Peak signal-to-noise ratio, 2024. Accessed: 2024-03-04. 6
- [8] Shilpa Devalal and A Karthikeyan. Lora technology-an overview. In 2018 second international conference on electronics, communication and aerospace technology (ICECA), pages 284–290. IEEE, 2018. 3, 6
- [9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 1, 2

- [10] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3
- [11] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 1
- [12] Zhen Han, Zeyinzi Jiang, Yulin Pan, Jingfeng Zhang, Chaojie Mao, Chenwei Xie, Yu Liu, and Jingren Zhou. Ace: Allround creator and editor following instructions via diffusion transformer. arXiv preprint arXiv:2410.00086, 2024. 5

- [13] Jacky Hate. Text-to-image-2m dataset. https:// huggingface.co/datasets/jackyhate/textto-image-2M, 2024. 6
- [14] Junjie He, Yuxiang Tuo, Binghui Chen, Chongyang Zhong, Yifeng Geng, and Liefeng Bo. Anystory: Towards unified single and multiple subject personalization in text-to-image generation, 2025. 2, 3
- [15] Qingdong He, Jinlong Peng, Pengcheng Xu, Boyuan Jiang, Xiaobin Hu, Donghao Luo, Yong Liu, Yabiao Wang, Chengjie Wang, Xiangtai Li, and Jiangning Zhang. Dynamiccontrol: Adaptive condition selection for improved text-toimage generation, 2024. 2
- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1, 2
- [18] Miao Hua, Jiawei Liu, Fei Ding, Wei Liu, Jie Wu, and Qian He. Dreamtuner: Single image is enough for subject-driven generation, 2023. 2, 3
- [19] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Huanzhang Dou, Yupeng Shi, Yutong Feng, Chen Liang, Yu Liu, and Jingren Zhou. Group diffusion transformers are unsupervised multitask learners. 2024. 5
- [20] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024. 5
- [21] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021. 6
- [22] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3
- [23] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 3, 5, 1
- [24] Black Forest Labs. Flux: Official inference repository for flux.1 models, 2024. Accessed: 2024-11-12. 1, 2, 3, 5, 6
- [25] Shakker Labs. Flux.1-dev-controlnet-union-pro. https: //huggingface.co/Shakker- Labs/FLUX.1dev-ControlNet-Union-Pro, 2024. 5, 6, 8
- [26] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems, 36:30146–30166, 2023. 2
- [27] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems, 36, 2024. 2, 3, 1
- [28] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, MingMing Cheng, and Ying Shan. Photomaker: Customizing re-

- alistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8640–8650, 2024. 2, 1
- [29] Zongming Li, Tianheng Cheng, Shoufa Chen, Peize Sun, Haocheng Shen, Longjin Ran, Xiaoxin Chen, Wenyu Liu, and Xinggang Wang. Controlar: Controllable image generation with autoregressive models, 2025. 2
- [30] Kuan Heng Lin, Sicheng Mo, Ben Klingher, Fangzhou Mu, and Bolei Zhou. Ctrl-x: Controlling structure and appearance for text-to-image generation without guidance, 2024. 2
- [31] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2
- [32] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subjectdiffusion: Open domain personalized text-to-image generation without test-time fine-tuning. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 3
- [33] Yuhang Ma, Wenting Xu, Jiji Tang, Qinfeng Jin, Rongsheng Zhang, Zeng Zhao, Changjie Fan, and Zhipeng Hu. Character-adapter: Prompt-guided region control for highfidelity character customization, 2024. 2, 3
- [34] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instructionbased image creation and editing via context-aware content filling. arXiv preprint arXiv:2501.02487, 2025. 5
- [35] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 2
- [36] Konstantin Mishchenko and Aaron Defazio. Prodigy: An expeditiously adaptive parameter-free learner. In Forty-first International Conference on Machine Learning, 2024. 6
- [37] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4296–4304, 2024. 2, 3, 6, 8
- [38] Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. Kosmos-g: Generating images in context with multimodal large language models. arXiv preprint arXiv:2310.02992, 2023. 2, 3
- [39] Zexu Pan, Zhaojie Luo, Jichen Yang, and Haizhou Li. Multimodal attention for speech emotion recognition. arXiv preprint arXiv:2009.04107, 2020. 3
- [40] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 2, 3

- [41] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 1, 2
- [42] Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, et al. Unicontrol: A unified diffusion model for controllable visual generation in the wild. arXiv preprint arXiv:2305.11147, 2023. 2, 3

- [43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3, 6
- [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 1, 2, 3
- [45] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 2
- [46] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. arXiv preprint arXiv:2410.10792, 2024. 2
- [47] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 3, 5, 6, 1
- [48] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10,

2022. 2

- [49] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 2
- [50] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8543–8552, 2024. 2
- [51] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 3

- [52] Yanan Sun, Yanchen Liu, Yinhao Tang, Wenjie Pei, and Kai Chen. Anycontrol: Create your artwork with versatile control on text-to-image generation, 2024. 2
- [53] Jianyi Wang, Kelvin C. K. Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images, 2022. 6
- [54] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 2
- [55] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to

- structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6
- [56] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943–15953, 2023. 2
- [57] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. International Journal of Computer Vision, pages 1–20, 2024. 2
- [58] XLabs-AI. Flux-ip-adapter. https://huggingface. co/XLabs-AI/flux-ip-adapter, 2024. 6, 8
- [59] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1191–1200, 2022. 6
- [60] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2, 3, 5, 6, 8

- [61] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023. 2, 3, 5, 6, 8
- [62] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8069–8078, 2024. 3
- [63] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 2

# OminiControl: Minimal and Universal Control for Diffusion Transformer Supplementary Material

## A. Details of Subjects200K datasets

We present a comprehensive synthetic dataset constructed to address the limitations in scale and image quality found in previous datasets [23, 27, 28, 47]. Our approach leverages FLUX.1-dev [24] to generate high-quality, consistent images of the same subject under various conditions.

Subjects200K dataset currently consists of two splits, both generated using similar pipelines. Split-1 contains paired images of objects in different scenes, while Split-2 pairs each object’s scene image with its corresponding studio photograph. Due to their methodological similarities, we primarily focus on describing the synthesis process and details of Split-2, although both splits are publicly available. Our complete Subjects200K dataset can be fully accessed via this link.

### A.1. Generation pipeline

Our dataset generation process consists of three main stages: description generation, image synthesis, and quality assessment.

Description Generation We employed ChatGPT-4o to create a hierarchical structure of descriptions: We first generated 42 diverse object categories, including furniture, vehicles, electronics, clothing, and others. For each category, we created multiple object instances, totaling 4,696 unique objects. Each object entry consists of: (1) A brief description, (2) Eight diverse scene descriptions, (3) One studio photo description. Figure S2 shows a representative example of our structured description format.

Image Synthesis We designed a prompt template to leverage FLUX’s capability of generating paired images containing the same subject. Our template synthesizes a comprehensive prompt by combining a brief object description with two distinct scene descriptions, ensuring subject consistency while introducing environmental variations.

The detailed prompt structure is illustrated in Figure S3. For each prompt, we set the image dimensions to 1056×528 pixels and generated five images using different random seeds to ensure diversity in our dataset. During the training process, we first split the paired images horizontally, then performed central cropping to obtain 512×512 pixel image pairs. This padding strategy was implemented to address cases where the generated images were not precisely bisected, preventing potential artifacts from appearing in the wrong half of the split images.

Quality assessment We leveraged ChatGPT-4o’s vision capabilities to rigorously evaluate the quality of images generated by FLUX.1-dev. The assessment focused on multiple

[Figure 51]

[Figure 52]

Not separate views

Chair

[Figure 53]

[Figure 54]

Inconsistent

Cola

[Figure 55]

[Figure 56]

Low quality

Table

Figure S1. Examples of successful and failed generation results from Subjects200K dataset. Green checks indicate successful cases where subject identity and characteristics are well preserved, while red crosses show failure cases.

critical aspects:

- • Image composition: Verifying that each image properly contains two side-by-side views.
- • Subject consistency: Ensuring the subject maintains identity across both views.
- • Image quality: Confirming high resolution and visual fidelity.

To maintain stringent quality standards, each image underwent five independent evaluations by ChatGPT-4o. Only images that passed all five evaluations were included in our training dataset. Figure S1 presents representative examples from our quality-controlled dataset.

### A.2. Dataset Statistics

In Split-2, we first generated 42 distinct object categories, from which we created and curated a set of 4,696 detailed object instances. Then we combine these descriptions to generate 211,320 subject-consistent image pairs. Through rigorous quality control using GPT-4o, we selected 111,767 high-quality image pairs for our final dataset. This extensive filtering process ensured the highest standards of image quality and subject consistency, resulting in a collection of 223,534 high-quality training images.

{

"brief_description":

"A finely-crafted wooden seating piece.", "scene_descriptions": [

"Set on a sandy shore at dusk, it faces the ocean with a gentle breeze rustling nearby palms, bathed in soft, warm twilight.",

"Positioned in a bustling urban cafe, it stands out against exposed brick walls, capturing the midday sun through a wide bay window." // Additional six scene descriptions omitted

], "studio_photo_description":

"In a professional studio against a plain white backdrop, it is captured in threequarter view under uniform high-key lighting, showcasing the delicate grain and smooth of its finely-crafted surfaces."

}

Figure S2. An example of our structured description format for dataset generation.

- prompt_1 = f"Two side-by-side images of the same object: {brief_description}"

- prompt_2 = f"Left: {scene_description1}"

- prompt_3 = f"Right: {scene_description2}" prompt_image = f"{prompt_1}; {prompt_2}; {prompt_3}"

- Figure S3. Our prompt template for paired image generation. The template combines a brief object description with two distinct scene descriptions to maintain subject consistency while varying environmental conditions.

|[Figure 57]|
|---|

[Figure 58]

[Figure 59]

"In a bright room, it is placed on a table near a window."

Condtion

Training with data augmentation

Training with Suject200K

[Figure 60]

|[Figure 61]|
|---|

[Figure 62]

Result Image

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

- Figure S4. Comparison of models trained with different data. The model trained by data augmentation tends to copy inputs directly, while model trained by our Subjects200K generates novel views while preserving identity.

description as input, aiming to generate novel images of the same subject following the text guidance while preserving its key characteristics.

To validate the effectiveness of our Subjects200K dataset described in Section 3.3, we compare two training strategies for this task. The first approach relies on traditional data augmentation, where we apply random cropping, rotation, scaling, and adjustments to contrast, saturation, and color to the original images. The second approach utilizes our Subjects200K dataset. As shown in Figure S4, the model trained with data augmentation only learns to replicate the input conditions with minimal changes. In the first row, it simply places the taco plush toy in a bright room setting while maintaining its exact appearance and pose. Similarly, in the second row, the yellow alarm clock is reproduced with nearly identical details despite the windowside placement instruction. In contrast, our Subjects200K -trained model demonstrates the ability to generate diverse yet consistent views of the subjects while faithfully following the text prompts.

## B. Additional experimental results

### B.1. Effect of training data

For subject-driven generation, our model takes a reference image of a subject (e.g., a plush toy or an object) and a text

### B.2. Evaluation for subject-driven generation

Framework and criteria. To systematically evaluate subject-driven generation quality, we establish a framework with five criteria assessing both preservation of subject characteristics and accuracy of requested modifications:

• Identity Preservation: Evaluates preservation of essen-

Score Comparison Across Methods

100

90

Rating(%)

80

70

60

50

IP-Adapter (SD) SSR-Encoder IP-Adapter (FLUX) Ours

Evaluation Methods

Metrics Identity Consistency Text-image alignment Visual coherence Overall Rating

| |
|---|

- Figure S5. User study results comparing different methods across three metrics: identity consistency, text-image alignment, and visual coherence.

tial identifying features (e.g., logos, brand marks, distinctive patterns)

- • Material Quality: Assesses if material properties and surface characteristics are accurately represented
- • Color Fidelity: Evaluates if colors remain consistent in regions not specified for modification
- • Natural Appearance: Assesses if the generated image appears realistic and coherent
- • Modification Accuracy: Verifies if the changes specified in the text prompt are properly executed

User studies. To further validate our approach, we conducted user studies collecting 375 valid responses. Participants evaluated the generated images across three key dimensions: identity consistency, text-image alignment, and visual coherence between subjects and backgrounds. The results shown in Figure S5 corroborate our quantitative findings, with our method achieving superior performance across all evaluation criteria.

### B.3. Additional generation results

We showcase more generation results from our method. Figure S7 presents additional results on the DreamBooth dataset, while Figure S8 demonstrates our method’s effectiveness on other subject-driven generation tasks.

Method Identity Material Color Natural Modification Average

preservation quality fidelity appearance accuracy score Average over 5 random seeds

IP-Adapter (SD 1.5) 29.4 86.1 45.3 97.9 17.0 55.1 SSR-Encoder 46.0 92.0 54.2 96.3 28.5 63.4 IP-Adapter (FLUX) 11.8 65.8 30.8 98.1 57.7 52.8 Ours 50.6 84.3 55.0 98.5 75.8 72.8

Best score over 5 random seeds

IP-Adapter (SD 1.5) 56.3 98.9 70.1 99.7 37.2 72.5 SSR-Encoder 64.3 99.2 74.4 99.1 53.6 78.1 IP-Adapter (FLUX) 27.5 86.1 53.6 99.9 74.9 68.4 Ours 82.3 98.0 88.4 100.0 90.7 91.9

Table S1. Quantitative evaluation results (in percentage) across different evaluation criteria. Higher values indicate better performance.

[Figure 67]

[Figure 68]

[Figure 69]

###### Figure S8. More results on other subject-driven generation tasks.

