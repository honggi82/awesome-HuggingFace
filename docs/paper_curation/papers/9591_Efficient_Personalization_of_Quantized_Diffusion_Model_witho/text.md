# arXiv:2503.14868v1[cs.CV]19Mar2025

## Efficient Personalization of Quantized Diffusion Model without Backpropagation

Hoigi Seo1∗ Wongi Jeong1∗ Kyungryeol Lee1 Se Young Chun1,2† 1Dept. of Electrical and Computer Engineering, 2INMC & IPAI Seoul National University, Republic of Korea

{seohoiki3215, wg7139, kr.lee, sychun}@snu.ac.kr

### Abstract

Diffusion models have shown remarkable performance in image synthesis, but they demand extensive computational and memory resources for training, fine-tuning and inference. Although advanced quantization techniques have successfully minimized memory usage for inference, training and fine-tuning these quantized models still require large memory possibly due to dequantization for accurate computation of gradients and/or backpropagation for gradient-based algorithms. However, memory-efficient finetuning is particularly desirable for applications such as personalization that often must be run on edge devices like mobile phones with private data. In this work, we address this challenge by quantizing a diffusion model with personalization via Textual Inversion and by leveraging a zeroth-order optimization on personalization tokens without dequantization so that it does not require gradient and activation storage for backpropagation that consumes considerable memory. Since a gradient estimation using zeroth-order optimization is quite noisy for a single or a few images in personalization, we propose to denoise the estimated gradient by projecting it onto a subspace that is constructed with the past history of the tokens, dubbed Subspace Gradient. In addition, we investigated the influence of text embedding in image generation, leading to our proposed time steps sampling, dubbed Partial Uniform Timestep Sampling for sampling with effective diffusion timesteps. Our method achieves comparable performance to prior methods in image and text alignment scores for personalizing Stable Diffusion with only forward passes while reducing training memory demand up to 8.2×. Project page: https: //ignoww.github.io/ZOODiP_project/

### 1. Introduction

Recent advances in diffusion models [21, 56, 58, 59, 64] have revolutionized generative AI, offering a powerful

* Authors contributed equally. † Corresponding author.

framework for high-quality, diverse data generation. Diffusion models have shown impressive capabilities across various applications, including image synthesis [21, 37, 45, 56– 58, 64], text-guided image synthesis [9, 15, 46, 50, 51], video generation [4, 22, 23, 25, 38], and 3D content generation [31, 35, 47, 55, 63], outperforming traditional generative approaches. However, modern diffusion models face significant memory and computational costs during training, fine-tuning, and inference due to their size.

These overheads pose major challenges in personalization tasks, which require fine-tuning diffusion models with only a few user-provided images. Typically, personalization is achieved by either fine-tuning the denoising network [20, 28, 52, 53, 61, 65] or introducing new text tokens [1, 17, 28, 42, 62, 68] to represent desired subjects. However, both approaches require gradient-based optimization, leading to significant memory and computational overhead, which is especially problematic in memoryconstrained on-device training for sensitive data. To tackle the overheads in personalization, efficient tuning techniques have emerged. Existing methods reduce trainable parameters [20, 28, 53, 61, 65], leverage quantized models [26, 54], or employ gradient-free optimization with evolution strategy [16]. However, these methods have limitations: 1) they mostly rely on backpropagation, unsuitable for most mobile processors, which are designed to accelerate inference [48]; 2) they still require significant memory for storing activations and gradients; 3) evolutionary algorithms can be unstable and inefficient in small-batch.

Here we propose Zeroth-Order Optimization for Diffusion model Personalization (ZOODiP) that can personalize Stable Diffusion with 2.37GB VRAM consumption using only forward passes without compromising the image quality. Our method relies on three key observations: First, Zeroth-Order (ZO) optimization effectively handles nondifferentiable objectives [39] (e.g., accuracy) during training. Second, tokens optimized via Textual Inversion [17] undergo significant changes in a low-dimensional subspace. Principal Component Analysis (PCA) on the token embeddings reveals that the initial and personalized tokens pri-

[Figure 1]

[Figure 2]

: Stable Diffusion (FP32)

TextAlignment

Dreambooth

: Quantized Stable Diffusion (INT8) : Gradient memory

- 4.5GB

3.3GB

6.6GB

- 5.0GB

Textual Inversion

PEQA TuneQDM ZOODiP (Ours)

: Optimizer state memory

: Others (activations, caches, etc.)

Total VRAM usage (GB)

4.4GB

- 0.29GB

- 1.8GB

[Figure 3]

[Figure 4]

ImageAlignment

Dreambooth

- 0.5GB

- 1.0GB

- 2.7GB

0.14GB

Textual Inversion

2.0GB

###### 2.37 GB 0.5GB

|0.26 MB|
|---|
|0.0 MB|

PEQA TuneQDM ZOODiP (Ours)

0.37GB

4.5GB

2.0GB

2.0GB 2.0GB

DreamBooth PEQA TuneQDM

Textual Inversion

ZOODiP (Ours)

Total VRAM usage (GB)

(a) GPU memory breakdown across various personalization methods.

(b) VRAM usage versus image and text alignment scores.

- Figure 1. Analysis of memory consumption and performance of Stable Diffusion personalization methods. (Left) GPU memory breakdown for each method on a Stable Diffusion personalization with a batch size of 1. ZOODiP (Ours) shows significantly higher memory efficiency compared to other methods. (Right) Comparison of memory usage versus performance across methods. Performance is measured with text (CLIP-T) and image (CLIP-I) alignment scores. ZOODiP achieves comparable performance to other methods while using significantly less memory (up to 8.2× less than DreamBooth). Memory usage was profiled using the PyTorch profiler and nvidia-smi command.

dient projection, and customized timestep sampling for efficient diffusion model personalization.

marily update within this subspace. Third, based on prior works [2, 7, 10, 12, 19, 30, 34, 44, 68], which argue that timesteps have distinct roles in diffusion models, we identified an effective timestep section for personalization.

• We demonstrate that ZOODiP achieves competitive performance with significantly lower memory requirements on the DreamBooth dataset (see Fig. 1).

Following the first observation, we trained token embeddings using ZO optimization in a quantized, nondifferentiable model [3, 39, 66]. This method reduces memory usage for activations, weights, and computational costs on edge devices. Inspired by the second finding, we introduced Subspace Gradient (SG) to accelerate training by mitigating noisy gradients. SG projects out dimensions with noisy gradients based on parameter trajectory, improving performance. Based on the third observation, we applied Partial Uniform Timestep Sampling (PUTS) within targeted timestep sections, skipping less influential timesteps to maximize impact within fixed training iterations.

### 2. Related Works

Diffusion model personalization. Diffusion model personalization aims to adapt a pre-trained model to generate images of new, user-defined concepts using a small set of provided images. Textual Inversion [17] tackles personalization by optimizing a single token embedding that represents the target concept. DreamBooth [52] tunes the denoising U-Net, while Custom Diffusion [28] personalizes multiple concepts by adopting new tokens and adjusting the key and value matrices in cross-attention. P+ [62] assigns unique text tokens to each U-Net stage, and TextBoost [42] introduces an augmentation token and employs SNR-based sampling for single-image personalization. However, these methods often require significant computational resources and memory, limiting their applicability on resource-constrained devices. Inspired by the efficiency of single-token optimization in Textual Inversion, ZOODiP introduces a new token to represent the target concept and optimizes it efficiently.

We comprehensively assessed ZOODiP using both qualitative and quantitative measures on DreamBooth [52] dataset. We measured performance across two key metrics: 1) text-image alignment and 2) reference-generated image alignment. Through GPU memory profiling during training, we demonstrate that ZOODiP utilizes up to 8.2× less memory than existing methods, achieving similar image quality. This substantial reduction in memory footprint underscores ZOODiP’s efficacy in enabling diffusion model personalization on memory-constrained devices.

Our contributions can be summarized as follows:

Efficient fine-tuning. Fine-tuning large models requires significant memory, motivating the memory-efficient training methods. Many approaches aim to reduce the number of trainable parameters. For example, LoRA [24] applies low-rank matrices to linear layers, effectively decreas-

- • We empirically identify that Textual Inversion tokens primarily optimize within a low-dimensional subspace, and that focusing on partial timesteps accelerates training.
- • We introduce ZOODiP, a novel method that combines ZO optimization with a quantized model, subspace gra-

[Figure 5]

###### “A photo ofV*”

[Figure 6]

ℒ 𝜖̂ ,𝜖 − ℒ 𝜖̂,𝜖 𝜇

[Figure 7]

low Σ

low Σ

𝑔 =

𝑒

high Σ

high Σ

[Figure 8]

[Figure 9]

[Figure 10]

Tokenizer

Reference images

PCA

𝜏

eigenvecs w/ low Σ

𝜏

[Figure 11]

Token embeddings

𝜏

VAE

𝜏 < 𝑑

Cross-attention

Cross-attention

Cross-attention

Cross-attention

𝑑

𝑑

𝑷𝝂

Proj. matrix

<dog>

Traj. buffer

𝜖̂

𝑧 = 𝛼 𝑧 + 1 − 𝛼 𝜖 𝑡 ~ 𝑈(𝑇 ,𝑇 )

iterations 𝜏

2𝜏

𝜖̂′

copy

…

[Figure 12]

Partial Uniform Timestep Sampling

Initial token

<V*>

Optimized token

𝒈 ′

perturb 𝜃 = 𝜃 + 𝜇𝑒

𝜏

𝑑

𝑔 𝒈 ′

𝑒~𝑁 0,𝐼 𝜇 ≪ 1

𝑔

𝑑

𝑑

Text transformer

###### . .

=

[Figure 13]

: frozen : trainable

No backprop. (ZO w/ SG)

[Figure 14]

8-bit Quantized Text Encoder 8-bit Quantized denoising U-Net

𝑷𝝂𝑻

𝒈

𝒈

𝒈 = 𝒈 − 𝒈

𝑷𝝂

(a) Overall illustration of ZOODiP training framework.

(b) Illustration of Subspace Gradient (SG) updates.

- Figure 2. (a) Illustration of overall ZOODiP framework. A target token is initialized and added to the prompt. Reference images are encoded, and Partial Uniform Timestep Sampling (PUTS)-sampled timestep noise is predicted. The loss is calculated with the original and perturbed token to estimate the gradient. (b) Illustration of Subspace Gradient (SG). Updated tokens from the previous τ iterations are stored. PCA identifies low-variance eigenvectors to project out noisy dimensions from the estimated gradient for the next τ iterations.

### 3. Method

ing the number of parameters to update. QLoRA [13] further reduces memory usage by applying LoRA to quantized models. Another direction explores fine-tuning quantized models by adjusting quantization parameters. PEQA [26] only tunes the quantization scales to reduce the optimizer state memory. TuneQDM [54] personalizes quantized diffusion models by tuning the scales specific to each diffusion timestep set. Beyond parameter reduction and quantization, Gradient-Free Textual Inversion [16] employs an evolution strategy to optimize tokens, bypassing backpropagation.

In this section, we propose ZOODiP, a memory-efficient approach to personalize diffusion models. It leverages zeroth-order (ZO) optimization with a quantized model, eliminating the need for backpropagation, and thereby significantly reducing memory usage. Based on the observation that trained tokens in Textual Inversion primarily change within a low-dimensional subspace, we introduce Subspace Gradient (SG) to optimize tokens within this reduced space. Furthermore, we propose Partial Uniform Timestep Sampling (PUTS) for efficient timestep selection, capitalizing on the observation that text embeddings predominantly influence image generation at specific diffusion timesteps. By integrating ZO optimization with quantization, SG, and PUTS, ZOODiP enables efficient personalization on resource-constrained devices. The overall framework is illustrated in Fig. 2 and Algorithm. 1.

However, these techniques can be limited by memoryintensive backpropagation or training instability. ZOODiP overcomes these limitations by utilizing zeroth-order optimization on a quantized model, eliminating backpropagation and its associated memory overhead.

Zeroth-order optimization. As model sizes increased, memory consumption during backpropagation became a major challenge for fine-tuning. Zeroth-order (ZO) optimization [5, 6, 14, 18, 36, 40, 60], which estimates gradients using only forward passes with random perturbations, has emerged as a promising solution. MeZO [39] demonstrated memory efficient tuning of large language models with forward-pass only, storing random seeds instead of perturbation vectors. Additionally, MeZO showed that ZO’s convergence rate depends on the model’s effective rank and that non-differentiable objectives can be optimized with ZO. Similarly, DeepZero [8] trained models from scratch using parallelized ZO, achieving performance comparable to first-order methods. Inspired by these advancements, we propose a personalization framework that employs ZO optimization on a quantized model, significantly reducing memory overhead for both activations and weights.

#### 3.1. Formulation

ZOODiP builds upon Textual Inversion, aiming to learn a pseudo-token v∗ that represents a desired concept from a set of reference images x1,...,xn ∈ X containing that concept. This pseudo-token is incorporated into the model’s text embedding. We denote the condition for denoising network y∗ as the text prompt that includes v∗. Following the previous studies [17, 51], we formulate the objective as follows:

LLDM = Ez∼E(x),y∗,ϵ∼N(0,I),t ||ϵ − ϵϕ(zt,t,c(y∗))||22 , zt = √α¯tz + √1 − α¯tϵ

(1) where x is the input reference image, E is a Variational Autoencoder (VAE) encoder, z is the latent encoded with the

- Algorithm 1 Fine-tuning algorithm of ZOODiP

Require: token embedding θ, reference image set X, text input y∗, text encoder c, VAE encoder E, denoising network ϵϕ, buffer size τ, threshold ν, number of estimation n, perturbation size µ, total iteration L, trajectory buffer B, learning rate η

- 1: Pν ← 0
- 2: for l = 1,...,L do
- 3: Sample image x ∈ X
- 4: t ∼ U(TL,TU) ▷ PUTS
- 5: gˆθ ← RGE(x,y∗,n,t,E,c,ϵϕ,µ) ▷ Eq. 1 and Eq. 3
- 6: gˆθ′ ← gˆθ(I − Pν⊤Pν) ▷ SG update
- 7: θ ← optimizer(θ,gˆθ′ ,η)
- 8: Bmod(l,τ) ← θ ▷ Fill the buffer with updated token
- 9: if mod(l,τ) = 0 then ▷ Update Pν for SG
- 10: Pν ←updatePν(B,ν) ▷ Algorithm. 2
- 11: end if
- 12: end for

- Algorithm 2 Subspace generation

- 1: function UPDATEPν(B,ν)
- 2: B¯ ←Normalize(B) ▷ Eq. 4
- 3: λ, V ← PCA(B¯)
- 4: i∗ ← arg mini i ∈ {1 : τ} |

- i
- j=1 λj τ

- k=1 λk > 1 − ν

- 5: Pν ← Vi⊤∗:τ
- 6: return Pν
- 7: end function

encoder, ϵ is the noise from the forward process, ϵϕ is a denoising network, t is the diffusion timestep, c is a condition encoder which is text encoder in ZOODiP, and α¯t is cumulative product of α from 0 to t as defined in DDPM [21].

#### 3.2. ZO Optimization with Quantized Model

To optimize the v∗ with memory efficiency, we quantize the weights in the Linear and Convolution layer of all components in the diffusion model: U-Net, VAE, and the text encoder. The quantization is formulated as follows [27]:

W = s · clamp

W s

+ o,QN,QP − o (2)

where W is a weight to be quantized, ⌊·⌉, s, o represent the rounding function, quantization scale, and zero-point, respectively. The minimum and maximum quantization bounds, QN = −2N−1 and QP = 2N−1 − 1, are configured to ensure symmetric N-bit quantization. After the quantization of the network, LLDM in Eq. 1 becomes nondifferentiable due to the existence of the rounding function ⌊·⌉, which makes training a quantized model difficult [3].

To estimate the gradient on a quantized model, we leverage Random Gradient Estimation (RGE) [6, 8, 40, 60].

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

𝒌 = 𝟕𝟔𝟖 𝒌 = 𝟔𝟒𝟎 𝒌 = 𝟓𝟏𝟐 𝒌 = 𝟑𝟖𝟒

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

𝒌 = 𝟐𝟓𝟔 𝒌 = 𝟏𝟐𝟖 𝒌 = 𝟔𝟒 𝒌 = 𝟑𝟐

Reference Images

Figure 3. Sparse effective dimension in the token trained with Textual Inversion. Notably, the concept was preserved even when retaining only one-third of the optimized dimensions (k = 256).

RGE perturbs θ ∈ Rd, the token embedding for v∗, with random variables and estimates the gradient by calculating the non-differentiable objective LLDM(θ) as follows:

n

LLDM(θ + µei) − LLDM(θ) µ

1 n

ei (3)

gˆθ =

i=1

where gˆθ denotes an estimation of First-Order (FO) gradient ∇θL(θ) with respect to θ, µ is a perturbation size, n is the number of random directions to estimate the gradient and ei a random vector sampled from N(0,I). With the ZO approach, we were able to estimate the gradient without backpropagation while effectively reducing memory usage.

#### 3.3. Subspace Gradient

We investigated the optimization of the pseudo-token by Textual Inversion [17](TI) using Principal Component Analysis (PCA) of the token embeddings. Both the initial and optimized tokens were projected onto the principal components. The components with the top-k largest changes in the optimized token were retained, while the rest were replaced with components from the initial token before back-projection. Fig. 3 shows that the personalized concept is preserved after back-projection, indicating that key changes from TI lie within this subspace.

ZO optimization suffers from slow training due to noisy gradient estimates. Inspired by the fact that token optimization mainly occurs in a lower-dimensional subspace, we propose Subspace Gradient (SG) to accelerate training. SG leverages token trajectories to eliminate noisy dimensions. Specifically, we store updated tokens for τ iterations in a trajectory buffer B ∈ Rτ×d, where d > τ. Once B is full, we normalize it as follows:

B¯ = (B − µB)/σB (4)

where µB and σB is the mean and standard deviation of B along each feature dimension, respectively. Then we perform PCA on B¯ with singular value decomposition to get eigenvalues and eigenvectors of the covariance matrix:

###### B¯ = UΣV ⊤, λi = Σ2ii (5)

The square of i-th diagonal elements of Σ, denoted as λi, represent the variance explained by the corresponding eigenvectors. We calculate the ratio of the cumulative sum of λi to the total sum:

i∗ = arg min

i

i ∈ {1,...,τ}

- i
- j=1 λj τ

- k=1 λk

> 1 − ν (6)

where ν is a hyperparameter that controls the amount of variance retained. This determines i∗, the smallest index i∗ for which the cumulative ratio exceeds 1 − ν. We then construct a projection matrix Pν using the eigenvectors corresponding to the remaining variance:

Pν := V(⊤i∗+1):τ (7)

This matrix Pν represents the dimensions to be removed. We project the estimated row vector gradient gˆ onto the subspace orthogonal to Pν and subtract from gˆ to get gˆ′:

gˆ′ = gˆ(I − Pν⊤Pν) (8)

This subtraction effectively eliminates noisy components from the gˆ′. After updating Pν, the buffer B is cleared. Over the next τ iterations, the estimated gradient is projected out through Pν, the token trajectory is simultaneously accumulated in B. Fig. 2b, Algorithm. 1, and Algorithm. 2 represents the SG process as described above.

#### 3.4. Partial Uniform Timestep Sampling

To facilitate more efficient training, we introduce Partial Uniform Timestep Sampling (PUTS). The diffusion model can be interpreted as a mixture-of-experts based on the timesteps [2, 19, 30, 34], with each timestep playing a distinct role. In text-to-image diffusion models, several studies [7, 10, 12, 44, 68] have shown that text conditioning impact varies across timesteps. However, most works focus on inference or training from scratch, while using this insight for personalization remains underexplored.

TextBoost [42] empirically observed that text influence increases as the timestep nears noise, proposing SNR-based timestep sampling. However, our experiments showed text influence is negligible at certain timesteps (see Fig. 4). Specifically, sampling timestep t from U(0,500) failed to effectively learn the reference image concept, whereas sampling from U(500,1000) led to successful learning.

Based on these observations, we propose PUTS, which focuses on uniformly sampling timesteps within a specific range of the diffusion process. This range is chosen to prioritize timesteps where the text embedding has the most significant influence. PUTS can be formulated as follows:

zt = √α¯tz + √1 − α¯tϵ, t ∼ U (TL,TU) (9)

Reference images 𝑡 ~	𝑈(0,500)

𝑡 ~	𝑈(500,1000)

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

“A photo ofV*in the snow”

Figure 4. Textual Inversion [17] with various timestep sampling. When the timestep t for training is sampled from U(0, 500), key conceptual features such as color and body shape of the reference image are not effectively trained. In contrast, sampling from U(500, 1000) results in successful learning of these features.

where ϵ represents the noise sampled from Gaussian distribution, α¯t is the cumulative product of α values to sampled timestep t, and TL and TU denote the lower and upper bound of the partial timestep range. Our dense ablation study in Tab. 5 and Fig. 7 further display the effectiveness of the proposed method and validate the observations.

### 4. Experiments

To evaluate ZOODiP, we conducted quantitative and qualitative comparisons with methods based on DreamBooth (DB) [13, 26, 52, 54] and Textual Inversion (TI) [16, 17]. Personalization used the DB dataset and Stable Diffusionv1.5 [51], with INT8 quantization applied to Linear and Conv2D layers. All experiments ran on a single Nvidia RTX 3090 GPU with batch size 1. ZOODiP was trained with n = 2, µ = 10−3, τ = 128, ν = 10−3, TL = 500, TU = 900, L = 30,000, η = 5 × 10−3 using the ZOAdam [39] optimizer. Further results and experimental details are provided in the supplementary materials.

#### 4.1. Quantitative Results

Image and text alignment score. To evaluate ZOODiP’s personalization performance, we assessed text and image alignment against baselines. Using the DB dataset, we personalized 30 subjects with 25 prompts, generating 5 images per prompt for a total of 3,750 images. Image alignment was measured via cosine similarity with CLIP [49] (CLIPI) and DINOv2 [41] (DINO) embeddings. Text alignment was measured using cosine similarity between CLIP text embeddings of prompts and generated image embeddings (CLIP-T). Tab. 1 shows ZOODiP achieves similar performance to prior methods, outperforming GF-TI by 43.0% on CLIP-I and 13.4% on CLIP-T with the same memory usage. Compared to TuneQDM, a state-of-the-art quantized diffusion personalization method, ZOODiP shows a 0.7% drop on CLIP-T and a 0.5% increase on DINO. Compared to our baseline, TI, performance differences were +0.7%, -0.8%,

Base. Method Quant. Grad. Free Mem.↓ (GB) Stor.↓ (MB) CLIP-T↑ CLIP-I↑ DINO↑

DB [52] ✗ ✗ 19.4 3438 0.281 0.782 0.592 QLoRA [13] ✓ ✗ 7.56 1.63 0.297 0.762 0.607 PEQA [26] ✓ ✗ 6.31 1.32 0.275 0.791 0.604 TuneQDM [54] ✓ ✗ 8.96 2.48 0.289 0.788 0.555

DB

TI [17] ✗ ✗ 6.75 0.003 0.285 0.778 0.559

GF-TI [16] ✓ ✓ 2.37 0.003 0.253 0.540 0.011 ZOODiP (Ours) ✓ ✓ 2.37 0.003 0.287 0.772 0.558

TI

Table 1. Quantitative comparisons of DreamBooth [52] (DB), QLoRA [13] (r = 2), PEQA [26], TuneQDM [54], Textual Inversion [17] (TI), Gradient-Free Textual Inversion [16] (GF-TI), and Ours. ↑ / ↓ indicates higher / lower values are better. Performance was evaluated with CLIP-I and DINO for image alignment, CLIP-T for text-image alignment, and memory requirements of training (Mem.) and storage (Stor.). The worst-performance is double-underlined, and the second worst is single-underlined. ZOODiP achieves performance comparable to that of gradient-based methods with significantly less memory.

and -0.2% on CLIP-T, CLIP-I, and DINO, respectively.

Memory efficiency. To assess the training memory efficiency of each method, we tracked peak memory usage using the nvidia-smi command after quantization. As shown in Tab. 1 , ZOODiP requires significantly less memory during training—87.8% reduction compared to DB and 64.9% reduction compared to TI, which serves as a baseline for ZOODiP. We also evaluated storage requirements for the optimized models in safetensors format. ZOODiP requires only 3KB of storage, similar to other TI-based methods, making it well-suited for edge devices.

Training speed. The training speed of personalization methods is shown in Tab. 2. Since ZOODiP relies solely on forward passes, avoiding costly backpropagation, it achieves significantly faster training speeds compared to other backpropagation-based methods. Specifically, ZOODiP is 2.2× and 1.7× faster than TI for n = 1 and n = 2, respectively, and 4.2× and 3.3× faster than TuneQDM. GF-TI is the slowest, with ZOODiP showing a 28× and 22× speed improvement due to GF-TI’s requirement for 30 forward passes per iteration. Additionally, ZOODiP with FP16 precision, fully supported by the hardware, further enhances training speed.

#### 4.2. Qualitative Results

Fig. 5 shows qualitative comparisons between ZOODiP and other methods. ZOODiP generates images highly faithful to both prompts and reference images, with minimal distinction from other advanced techniques. In contrast, GF-TI, under similar memory constraints, struggles to maintain fidelity, resulting in noticeably inaccurate depictions. Fig. 6 shows the results of ZOODiP’s style personalization. Using several photos with consistent styles as references, ZOODiP demonstrates accurate reflection of the learned style for the

Method Prec. Speed (iter/sec) ↑ TI FP32 9.42

- Ours (n = 1) FP16

22.3

- Ours (n = 2) 18.4 TuneQDM

4.94 GF-TI 0.74

INT8

- Ours (n = 1) 20.7
- Ours (n = 2) 16.1

Table 2. Training speed comparisons of prior works and ZOODiP. ZOODiP achieves the fastest training speed by estimating gradients with only forward passes, bypassing backpropagation.

given prompts, indicating that ZOODiP has successfully inherited the style personalization capabilities of TI.

#### 4.3. Ablation Study

To analyze the contribution of each component in ZOODiP and assess the impact of hyperparameter variations, we conducted an ablation study. Experiments in this section were performed with <dog6> and <shiny sneaker> data from the DreamBooth [52] dataset.

Effectiveness of SG and PUTS. To measure the impact of ZOODiP’s key elements—SG and PUTS—we evaluated its performance by incrementally adding each component, as shown in Tab. 3. The results demonstrate that both SG and PUTS significantly enhance performance, outperforming na¨ıve Textual Inversion with ZO optimization.

Hyperparameter study. We conducted exhaustive experiments to analyze the impact of two key hyperparameters used in SG: τ and ν. As shown in Tab. 4, ZOODiP shows robust performance unless τ or ν is too large or small. When τ is small, frequent Pν updates may occur, introduce com-

Ref. images (V*) GF-TI ZOODiP (Ours)

DB PEQA TuneQDM TI

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

“A photo ofV*in a chef outfit”

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

“A photo ofV*wearing pink glasses”

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

“A photo ofV*in the snow”

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

“A photo ofV*with a city in the background”

- Figure 5. Qualitative comparison of image and text alignment. This figure shows how well each method generates images that match the input text prompt while preserving the identity of the personalized subject. ZOODiP generates images that faithfully reflect the prompt while maintaining the concept of the reference image, demonstrating strong image-text alignment.

[Figure 72]

- Figure 6. Qualitative results on style personalization. This figure showcases the results of style personalization achieved through ZOODiP, using few reference images with a consistent style. The outcome highlights ability of ZOODiP to personalize not only the subject but also the style with a high degree of accuracy. This demonstrates the versatility and extensive personalization capabilities of ZOODiP, effectively adapting both stylistic elements and subject details to match the reference images.

SG PUTS CLIP-T↑ CLIP-I↑ DINO↑

- ✗ ✗ 0.273 0.736 0.505

✓ ✗ 0.265 0.747 0.527

- ✗ ✓ 0.277 0.744 0.562

##### ✓ ✓ 0.266 0.759 0.569

- Table 3. Ablation study on ZOODiP components with <shiny sneaker>. ✓ denotes the component is applied, while

✗ means it is not. Without PUTS, timesteps are sampled uniformly.

τ

ν

10−1 10−2 10−3 10−4

32 0.704 0.686 0.716 0.729 64 0.721 0.736 0.712 0.724

128 0.736 0.735 0.759 0.716 256 0.739 0.727 0.738 0.716 512 0.705 0.704 0.676 0.707

- Table 4. Ablation study on hyperparameters τ and ν incorporated with SG. We optimized the pseudo-token with various τ and ν and measured the performance with the CLIP-I score. Experiments were done with <shiny sneaker> dataset.

Method CLIP-T↑ CLIP-I↑ DINO↑ Uniform 0.265 0.747 0.527

SNR-based 0.271 0.719 0.545 PUTS (Ours) 0.266 0.759 0.569

- Table 5. Ablation study on various diffusion timestep sampling method with <shiny sneaker>. PUTS outperforms in image alignment score among all sample methods with minor degradation in text alignment score compared to SNR-based sampling.

putational overhead, so we choose values of τ = 128 and ν = 10−3 which show the best performance.

Diffusion timestep sampling. To assess the effectiveness of PUTS on performance, we conducted experiments with different diffusion timestep sampling strategies: uniform sampling, SNR-based sampling [42], and PUTS. The results in Tab. 5 confirm that PUTS outperforms the other sampling methods, validating its effectiveness.

Furthermore, we conducted exhaustive experiments on various combinations of TU and TL. As shown in Fig. 7, we divided the diffusion timesteps into 10 units and tested all possible combinations. The results indicate that training with timesteps closer to the noise improves both image and text fidelity, supporting our choice of TL and TU values.

[Figure 73]

[Figure 74]

[Figure 75]

(a) CLIP-T scores. (b) CLIP-I scores. (c) DINO scores.

- Figure 7. Heatmap of CLIP-T, CLIP-I and DINO scores across varying TL and TU on the <dog6> dataset. x-axis is the TU and y-axis is the TL applied to the sampling distribution.

[Figure 76]

Count

𝑖∗/𝜏

- Figure 8. Histogram of i∗/τ ratios for <dog6> and <shiny sneaker> dataset with hyperparameter τ = 128, ν = 10−3 during training with SG. Despite the small ν, a significant portion (> 80%) of dimensions are projected out.

### 5. Discussion

ZO optimization’s convergence rate is influenced by the effective dimension of the optimized parameters [33, 39, 67]. We conjecture that ZOODiP’s success arises from the low dimensionality of the optimized token and its even smaller effective dimension (Sec. 3.3). The concentrated variance of the token trajectory allows SG to effectively project out over two-thirds of the dimensions in the trajectory buffer, even with a small ν value. It aligns with Fig. 3 and the analysis in Sec. 3.3. We selected ν = 10−3 to concentrate variance ratios within one-third of the total (Fig. 8).

### 6. Conclusion

We propose ZOODiP, a framework that enables the personalization of diffusion models even in highly memoryconstrained environments. ZOODiP leverages quantized models during training to reduce memory footprint. We then propose zeroth-order optimization to personalize these models. To overcome the limitations of zeroth-order optimization, we introduce Subspace Gradient (SG), which utilizes the trajectory of the optimizing token to eliminate noisy gradient dimensions and enhance performance based on the empirical analysis of the effective dimension of personalized tokens. Additionally, we systematically analyzed the impact of diffusion timesteps on personalization and identified the most effective timesteps for efficient training. Based on this analysis, we proposed Partial Uniform

Timestep Sampling (PUTS), which samples only these relevant timesteps. Through extensive qualitative and quantitative experiments, we demonstrate that ZOODiP achieves performance comparable to prior arts while requiring significantly less memory of only 2.37GB during training.

### Acknowledgements

This work was supported in part by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [NO. RS-2021-II211343, Artificial Intelligence Graduate School Program (Seoul National University)], National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. NRF-2022M3C1A309202211) and Samsung Electronics MX Division. Also, the authors acknowledged the financial support from the BK21 FOUR program of the Education and Research Program for Future ICT Pioneers, Seoul National University.

Appendix

- S1. Additional Experimental Details

- S1.1. Dataset

All quantitative experiments were conducted using the DreamBooth [52] (DB) dataset, which includes 30 distinct subjects, which are denoted in Tab. S1, each paired with 25 unique prompts. Among these subjects, nine are living entities—specifically dogs and cats—while the remaining 21 represent non-living objects. Details of the subjects and their respective prompts are provided in the Tab. S2 (living) and Tab. S3 (non-living). Notably, the Textual Inversion [17]-based method did not incorporate class tokens in the prompts used for image generation, ensuring a fair comparison by excluding scenarios where class tokens are utilized during the unique token learning process. For Textual Inversion [17] (TI) and Gradient-Free Textual Inversion [16] (GF-TI), we employed the ImageNet-based template proposed in the original TI paper for training. ZOODiP followed the DCO [29], utilizing Comprehensive Captioning (CC) for the text prompt.

- S1.2. Metrics

To compute the CLIP-I score, we use the openai/clip-vit-base-patch32 model [49] from Huggingface to extract image features for both the reference and generated images. We then calculate the cosine similarity for all possible pairs and average these values. For the CLIP-T score, we leverage the same model’s text encoder to extract text features from the input prompt. We also calculate image features for the generated images and measure the pairwise cosine similarity between the text and image features, averaging the results to obtain

the final score. To compute the DINO score, we utilize the facebook/dinov2-base model from Huggingface to extract DINOv2 [41] embeddings for both the reference and generated images. The score is determined by averaging the pairwise cosine similarities between these embeddings.

#### S1.3. Quantization

We utilized optimum-quanto, the PyTorch [43] quantization backend provided by Huggingface, to enable accelerated matrix multiplications on CUDA devices. For the 8-bit weight quantization described in the main paper, we employed optimum-quanto to quantize the weights of all Linear and Conv2D layers in the VAE, U-Net, and text encoder modules to 8-bit integer. Other layers’ parameters including text embedding, layer normalization, and batch normalization were not quantized. With this process, 96.4% of the whole Stable Diffusion [51] pipeline (U-Net, VAE, and text encoder) parameters are quantized to INT8, while the other 3.6% parameters remain FP16.

#### S1.4. Memory usage

We primarily monitored memory usage using the nvidia-smi command to observe real-time memory consumption during training and employed the PyTorch profiler for detailed memory breakdowns. Notably, the memory utilized by the CUDA context can vary based on the CUDA and PyTorch versions, so the reported memory usage may differ depending on the experimental environment. Our measurements were conducted under the following settings: CUDA Toolkit 11.8, Torch 2.4.1, Torchvision 0.19.1, and Python 3.8.10. Since PyTorch loads all CUDA libraries by default, including unused ones, the actual memory usage could potentially be reduced by unloading unnecessary libraries. However, as our focus was not on such optimizations, we ensured a fair comparison by evaluating all training methods under the same experimental environment. We also observed that Variational Auto-Encoder (VAE) encoding introduces significant memory overhead. Given that all methods do not require backpropagation through the VAE, we standardized the process by blocking gradients in the VAE encoding step across all methods. This adjustment ensured that each personalization method used the minimal memory required, facilitating a fair evaluation of memory usage.

#### S1.5. Baseline training configuration

For our experiments, we utilized the unofficial implementations provided by Huggingface for Textual Inversion [17] and DreamBooth [52] without prior-preservation loss which is suitable to memory-constrained environment. For PEQA [26] and TuneQDM [54], we conducted experiments using reproduced code based on the Huggingface implementation of DreamBooth. QLoRA [13] was imple-

Subjects in DreamBooth [52] dataset backpack, backpack dog, bear plushie, berry bowl, can, candle, cat, cat2, clock, colorful sneaker, dog, dog2, dog3, dog5, dog6, dog7, dog8, duck toy, fancy boot, grey sloth plushie, monster toy, pink sunglasses, poop emoji, rc car, red cartoon, robot toy, shiny sneaker, teapot, vase, wolf plushie

- Table S1. Full subjects name of DreamBooth dataset. The dataset names in the main paper are based on the corresponding subject datasets.

Full prompt used in evaluation (living)

- 1 ‘a {0} {1} in the jungle’.format(unique token, class token)

- 2 ‘a {0} {1} in the snow’.format(unique token, class token)

- 3 ‘a {0} {1} on the beach’.format(unique token, class token)

- 4 ‘a {0} {1} on a cobblestone street’.format(unique token, class token)

- 5 ‘a {0} {1} on top of pink fabric’.format(unique token, class token)

- 6 ‘a {0} {1} on top of a wooden floor’.format(unique token, class token)

- 7 ‘a {0} {1} with a city in the background’.format(unique token, class token)

- 8 ‘a {0} {1} with a mountain in the background’.format(unique token, class token)

- 9 ‘a {0} {1} with a blue house in the background’.format(unique token, class token)

- 10 ‘a {0} {1} on top of a purple rug in a forest’.format(unique token, class token)

- 11 ‘a {0} {1} wearing a red hat’.format(unique token, class token)

- 12 ‘a {0} {1} wearing a santa hat’.format(unique token, class token)

- 13 ‘a {0} {1} wearing a rainbow scarf’.format(unique token, class token)

- 14 ‘a {0} {1} wearing a black top hat and a monocle’.format(unique token, class token)

- 15 ‘a {0} {1} in a chef outfit’.format(unique token, class token)

- 16 ‘a {0} {1} in a firefighter outfit’.format(unique token, class token)

- 17 ‘a {0} {1} in a police outfit’.format(unique token, class token)

- 18 ‘a {0} {1} wearing pink glasses’.format(unique token, class token)

- 19 ‘a {0} {1} wearing a yellow shirt’.format(unique token, class token)

- 20 ‘a {0} {1} in a purple wizard outfit’.format(unique token, class token)

- 21 ‘a red {0} {1}’.format(unique token, class token)

- 22 ‘a purple {0} {1}’.format(unique token, class token)

- 23 ‘a shiny {0} {1}’.format(unique token, class token)

- 24 ‘a wet {0} {1}’.format(unique token, class token)

- 25 ‘a cube shaped {0} {1}’.format(unique token, class token)

- Table S2. Full prompts used in evaluation of living category objects. unique token represents the special token corresponds to object which aims to personalize, and class token denotes the class that unique token is in.

Full prompt used in evaluation (non-living)

- 1 ‘a {0} {1} in the jungle’.format(unique token, class token)

- 2 ‘a {0} {1} in the snow’.format(unique token, class token)

- 3 ‘a {0} {1} on the beach’.format(unique token, class token)

- 4 ‘a {0} {1} on a cobblestone street’.format(unique token, class token)

- 5 ‘a {0} {1} on top of pink fabric’.format(unique token, class token)

- 6 ‘a {0} {1} on top of a wooden floor’.format(unique token, class token)

- 7 ‘a {0} {1} with a city in the background’.format(unique token, class token)

- 8 ‘a {0} {1} with a mountain in the background’.format(unique token, class token)

- 9 ‘a {0} {1} with a blue house in the background’.format(unique token, class token)

- 10 ‘a {0} {1} on top of a purple rug in a forest’.format(unique token, class token)

- 11 ‘a {0} {1} with a wheat field in the background’.format(unique token, class token)

- 12 ‘a {0} {1} with a tree and autumn leaves in the background’.format(unique token, class token)

- 13 ‘a {0} {1} with the Eiffel Tower in the background’.format(unique token, class token)

- 14 ‘a {0} {1} floating on top of water.format(unique token, class token)

- 15 ‘a {0} {1} floating in an ocean of milk’.format(unique token, class token)

- 16 ‘a {0} {1} on top of green grass with sunflowers around it’.format(unique token, class token)

- 17 ‘a {0} {1} on top of a mirror’.format(unique token, class token)

- 18 ‘a {0} {1} on top of the sidewalk in a crowded street’.format(unique token, class token)

- 19 ‘a {0} {1} on top of a dirt road’.format(unique token, class token)

- 20 ‘a {0} {1} on top of a white rug’.format(unique token, class token)

- 21 ‘a red {0} {1}’.format(unique token, class token)

- 22 ‘a purple {0} {1}’.format(unique token, class token)

- 23 ‘a shiny {0} {1}’.format(unique token, class token)

- 24 ‘a wet {0} {1}’.format(unique token, class token)

- 25 ‘a cube shaped {0} {1}’.format(unique token, class token)

- Table S3. Full prompts used in evaluation of non-living category objects. unique token represents the special token corresponds to object which aims to personalize, and class token denotes the class that unique token is in.

Method CLIP-T↑ CLIP-I↑ DINO↑

RGE 0.266 0.759 0.569 SPSA 0.277 0.732 0.506

One-point 0.296 0.703 0.393

- Table S4. Personalization performance across different gradient estimation methods with n = 2. The results show that RGE outperforms other two different gradient estimations methods. Notably, RGE is more efficient in terms of computational cost, requiring fewer forward passes than SPSA when n > 1. Due to this efficiency and performance advantages, we adopted RGE as the gradient estimation method in ZOODiP.

mented using the BitsandBytes library. For GradientFree Textual Inversion [16], we utilized the official implementation for our experiments. We configured the training settings for each method to be as consistent as possible, adjusting only the batch size to 1. For DreamBooth, we set η = 5 × 10−6, L = 400, for Textual Inversion, we set η = 5 × 10−3, L = 5,000, for QLoRA, we set η = 1 × 10−4, L = 500, for PEQA, we set η = 3 × 10−6, L = 400, for TuneQDM, we set η = 3×10−6, L = 400, for Gradient-Free Textual Inversion, η = 5×106, L = 13,000, intrinsic dimension di = 256, σ = 1, and α = 1.

### S2. Additional Ablation Study

We conducted an extensive ablation study to analyze the impact of each component of ZOODiP. Beyond evaluating RGE, we examined the effectiveness of alternative gradient estimation methods, the influence of varying perturbation scales on performance, the impact of changing the number of gradient estimations, and the resulting time overhead associated with these changes. Unless otherwise noted, all additional experiments were performed on two subjects: <dog6> as a representative of living objects and <shiny sneaker> as a representative of non-living objects. The results from these experiments were averaged to ensure a balanced and comprehensive evaluation.

#### S2.1. Gradient estimation method

There are various methods for estimating gradients, each with unique advantages and trade-offs. In ZOODiP, we opted for Random Gradient Estimation (RGE, Eq. S10) due to its efficiency in estimating gradients using Monte Carlo gradient estimation. While many existing works utilize Simultaneous Perturbation Stochastic Approximation (SPSA, Eq. S11), which perturbs the gradient in two directions, or a one-point estimation method (Eq. S12) that calculates the gradient directly at the perturbed point, we evaluated all three methods for gradient estimation as seen in Tab. S4.

µ CLIP-T↑ CLIP-I↑ DINO↑

- 1 × 10−2 0.281 0.778 0.599

- 1 × 10−3 0.277 0.797 0.613

- 1 × 10−4 0.296 0.724 0.470

- Table S5. Quantitative results for different µ values. Optimal performance is observed at µ = 10−3 with varying µ, and we have set the value of µ accordingly for ZOODiP.

n CLIP-T↑ CLIP-I↑ DINO↑ Speed↑ (iter/s)

- 1 0.298 0.736 0.495 20.7

- 2 0.277 0.796 0.613 16.1 4 0.282 0.784 0.584 9.78 8 0.282 0.798 0.627 6.20

- Table S6. Quantitative results for various n, the number of gradient estimations. n = 2 is the promising choice between performance and computation efficiency.

n

1 n

gˆθ =

i=1

LLDM(θ + µei) − LLDM(θ) µ

ei (S10)

n

1 n

gˆθ =

i=1

LLDM(θ + µei) − LLDM(θ − µei) 2µ

ei

(S11)

n

LLDM(θ + µei+1) − LLDM(θ + µei) µ

1 n

gˆθ =

ei (S12)

i=1

Our results indicate that SPSA and RGE exhibit similar performance, with RGE occasionally outperforming SPSA. In contrast, the one-point estimation method performed poorly. From a computational perspective, SPSA requires 2n forward passes for n gradient estimation steps, whereas RGE needs only n + 1 forward passes, making it more efficient. Given its favorable balance of computational efficiency and performance, we selected RGE as the gradient estimation method for ZOODiP.

#### S2.2. Perturbation scale

We analyzed the impact of varying µ, the scaling parameter (a.k.a smoothing parameter) for perturbations in Eq. S10, on personalization performance. In general, the optimal value of µ can vary depending on the configuration of the model to be tuned as seen in Sec. S3.2 and Sec. S3.4. The results in Tab. S5 demonstrate that the chosen value of 10−3 is the most suitable perturbation scale for our setup.

Method PUTS CLIP-T↑ CLIP-I↑ DINO↑ DB

✗ 0.266 0.853 0.751 ✓ 0.272 0.854 0.758

✗ 0.241 0.798 0.584 ✓ 0.247 0.825 0.661

TI

- Table S7. Quantitative results from applying PUTS to DreamBooth and Textual Inversion. The results confirm that PUTS enhances the performance of gradient-based personalization methods by up to 13.2%. The improvement was particularly pronounced in Textual Inversion, which is highly influenced by the text encoder, highlighting the significant impact of PUTS in this context.

U-Net Precision CLIP-T↑ CLIP-I↑ DINO↑

INT8 0.288 0.834 0.657 INT4 0.212 0.835 0.647

- Table S8. The performance comparison between INT8 and INT4 for U-Net precision on the <dog6> subset of the DB dataset.

#### S2.3. Number of gradient estimation

According to MeZO [39], increasing n when fine-tuning large language models provides only marginal performance gains compared to the proportional increase in the number of forward passes, making n = 1 the most efficient choice. Tab. S6 illustrates the personalization performance across different n values with RGE. While increasing n slightly improves performance due to more accurate gradient estimation, the associated increase in training time becomes a limiting factor. Thus, we set n = 2 as a compromise between performance and efficiency.

### S3. Additional Experiments

#### S3.1. PUTS on gradient-based methods

Partial Uniform Timestep Sampling (PUTS), one of the key components of ZOODiP, not only enhances ZOODiP’s performance but also proves effective in gradient-based approaches. To validate this, we conducted experiments using two representative gradient-based personalization methods on Stable Diffusion v1.5: Textual Inversion and DreamBooth. These experiments were performed with the same hyperparameter settings for TL and TU as those used in ZOODiP. The results in Tab. S7 demonstrate that applying Partial Uniform Timestep Sampling (PUTS) to gradient-based methods improves performance by up to 11.6%. PUTS enhances efficiency by prioritizing informative timesteps, showcasing its versatility in optimizing both zeroth-order and gradient-based approaches.

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

INT8

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

INT4

- Figure S1. Generated images with the prompt “a photo of a dog” with various weight precision. While INT8 precision produces results nearly equivalent to full-precision performance, INT4 precision exhibits noticeable degradation in image quality, highlighting the trade-off between lower precision and fidelity.

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Ref. images INT8

[Figure 93]

[Figure 94]

“A photo ofV* wearing a yellow shirt”

INT4

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

“A photo ofV* in the jungle”

INT8 INT4

- Figure S2. Qualitative results of U-Net precision at INT8 and INT4 in <dog6> dataset. ZOODiP works on INT4 and INT8, but performance diminishes due to degradation caused by INT4 quantization.

#### S3.2. 4-bit quantized models

We applied ZOODiP to a 4-bit quantized model of the UNet, the component with the highest VRAM usage. We observed that quantizing with INT4 precision led to a performance loss in the quantization library we utilized as seen in Fig. S1. Consequently, we also noted a degradation in the performance of the personalized results when using this quantization approach. Nevertheless, our results demonstrate that ZOODiP is fully capable of personalizing 4-bit quantized diffusion models using only 1.9 GB of memory during the entire training process. Fig.S2 and Tab.S8 present the qualitative and quantitative results, respectively, for personalization with the 4-bit quantized model. To utilize the qint4 data type, we employed BFloat16 (BF16) for activation in all units, including the VAE, U-Net, and text encoder. When using BF16, the default µ value of 10−3 was insufficient, so we adjusted it to 10−2 to achieve optimal performance.

#### S3.3. Diversity of generated images

TI-based methods tend to exhibit less overfitting compared to DreamBooth (DB)-based methods, resulting in higher diversity in the generated images. To evaluate this tendency,

Ref. images DB PEQA TuneQDM TI GF-TI ZOODiP (Ours)

[Figure 99]

[Figure 100]

[Figure 101]

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

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

“A photo ofV*on the beach”

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

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

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

“A photo ofa purpleV*”

- Figure S3. Qualitative comparison of the diversity of generated images This figure compares the diversity achieved by different personalization methods. ZOODiP demonstrates the ability to generate highly diverse images while utilizing minimal memory resources.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

“A photo ofV*ona dirt road”

“A photo ofV*in front of Acropolis”

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

“A photo ofV*ina shallow water”

“A photo ofV*in the jungle”

Ref. images (V*) SD2.1 SDXL SD2.1 SDXL

- Figure S4. Qualitative results for personalizing SD2.1 and SDXL with ZOODiP. The figure demonstrate that ZOODiP can be applied not only to SD1.5, as discussed in the main paper, but also to various other models. For SD2.1, inference were conducted with images at a resolution of 768 × 768, while for SDXL, image generation was performed with resolution of 1024 × 1024. However, for SDXL, it was observed that the model’s inherent color interpretation prevents the subject’s colors from being completely replicated. This indicates that the model’s color rendering can vary depending on the environmental context, leading to shifts in the perceived color scheme.

we introduce the DINOinter score. For each subject, we generated 50 images using the same prompt with a personalized model. The DINOv2 embedding cosine similarity was then calculated for all pairs of generated images and averaged. A higher DINOinter score indicates that the generated images are more similar to each other, reflecting lower diversity in the generated outputs. The quantitative results for DINOinter

are presented in the Tab. S9. Additionally, we provide qualitative diversity results in Fig. S3. Fig. S3 shows a comparison of diversity across different methods, clearly showing that TI-based methods achieve significantly higher diversity than other approaches.

Method DINOinter ↓ DB 0.825 QLoRA 0.731

PEQA 0.806 TuneQDM 0.778

TI 0.679 GF-TI 0.150

ZOODiP (Ours) 0.671

- Table S9. Comparison of DINOinter scores across various personalization methods. The results indicate that TI-based methods are capable of generating a diverse range of images, whereas DBbased methods exhibit relatively lower diversity. This observation was consistent across all subjects in the DreamBooth (DB) dataset, highlighting a fundamental difference in the ability of these methods to capture and reflect variations in the generated outputs.

#### S3.4. Generalizability to other models

In the main paper, our experiments were conducted using Stable Diffusion v1.5 (SD1.5). However, ZOODiP is not limited to the certain model and can be extended to other models trained on different datasets and architectures, such as SD2.1 and SDXL [46] as seen in Fig. S4. For SD2.1, training was performed on a larger text token embedding dimension (1024 in OpenCLIP-ViT-H) using the same hyperparameters as SD1.5 except for µ = 10−2 and η = 10−2. For SDXL, we maintained all hyperparameters except for µ = 10−2, TL = 700, L = 20,000 and trained both token embeddings of OpenCLIP-ViT-L (768 dimension) and OpenCLIP-ViT-G (1280 dimension). The adjustment of TL is supported by prior research which indicates that information loss vary with dimensionality [11, 32], and the value of TL was selected empirically. Notably, fullprecision (FP32) Textual Inversion on SDXL requires approximately 17 GB of memory, and DreamBooth cannot be trained on customer level GPUs, such as the RTX3090. In contrast, tuning SDXL with ZOODiP enables successful training of concepts with only 5 GB of memory, highlighting its efficiency and scalability across diverse model configurations.

#### S3.5. Training time

We measured the total training time for each personalization method and observed a general inverse relationship between memory usage and training time in Tab. S10. The table shows a trade-off between time complexity and space complexity. However, Gradient-Free Textual Inversion (GF-TI) deviates from this trend, requiring significantly more training time despite its low memory usage. This behavior indicates that GF-TI struggles with learning effectively under small batch training conditions, which impacts its overall efficiency and practicality in resource-constrained devices.

Base. Method Time (min)

DB 2 QLoRA 1.1

DB

PEQA 1.5 TuneQDM 1.4

TI 8.8

TI

GF-TI 293 ZOODiP (n = 2) 31

Table S10. Total training time with the configurations in Sec S1.5. DB-based methods consume more memory but train faster, while TI-based methods are more memory-efficient but require more iterations.

iter=0 2000 4000 6000 8000 30000

Ref. images

10000

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Naïve

•••

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

•••

ZOODiP

“A photo ofV*with a city in the background”

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Naïve

•••

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

•••

###### ZOODiP

“A photo ofV*on the beach”

Figure S5. Qualitative comparisons on na¨ıve ZO textual inversion without SG and PUTS to ZOODiP (Ours) over iterations. The na¨ıve approach exhibits slower training and tends to produce images that are less aligned with the reference image. In contrast, ZOODiP achieves faster training and generates images that are closely aligned with the reference subject.

### S4. Additional Results

#### S4.1. Qualitative results of SG and PUTS

Subspace Gradient (SG) and Partial Uniform Timestep Sampling (PUTS) play crucial roles in enabling efficient personalization by addressing two key challenges: SG removes noisy gradient components to focus on the most informative subspace, while PUTS suppresses sampling from ineffective timesteps to optimize training efficiency. To evaluate the impact of these components during the learning process, we generated images using the same seed with token embeddings learned at every 1,000 iterations. As illustrated in Fig. S5, using both SG and PUTS significantly accelerates the training process compared to na¨ıve zerothorder optimization that does not incorporate these enhancements. The results show that models using SG and PUTS converge to high-quality personalization much faster than the baseline, underscoring the effectiveness of these components in improving the efficiency and speed of training.

#### S4.2. Additional qualitative results

In this section, we present additional qualitative comparisons of personalization results that were not included in the main paper. These results are illustrated in Fig. S6, Fig. S7, Fig. S8, Fig. S9, and Fig. S10. The comparisons demonstrate that ZOODiP effectively captures the characteristics of the reference image and text prompts, achieving a level of fidelity comparable to other gradient-based personalization methods.

### S5. Limitation

In this section, we delve into the limitations of ZOODiP. While ZOODiP incorporates innovative techniques such as Subspace Gradient and Partial Uniform Timestep Sampling to mitigate the slow learning speed—a well-known challenge of zeroth-order optimization—it still requires a considerably larger number of iterations compared to gradientbased approaches. This increased iteration count stems from the fundamental nature of zeroth-order optimization, which relies on function evaluations rather than gradient backpropagation, inherently making it less sample-efficient. Although ZOODiP compensates with a faster training speed per iteration, the substantial number of iterations can introduce a significant time overhead, especially when proper hardware acceleration, such as high-performance Neural Processing Units (NPUs), is unavailable.

Furthermore, as ZOODiP is built upon the Textual Inversion framework, its performance is influenced by the strengths and weaknesses of Textual Inversion. This dependency implies that ZOODiP may face challenges in cases where Textual Inversion struggles, such as subjects with complex or highly variable visual characteristics, or when adapting to certain models that inherently perform poorly in personalization tasks. For example, if the base model lacks sufficient representational capacity or if the dataset used for training does not adequately capture the nuances of the subject, the effectiveness of ZOODiP can diminish.

“A photo ofV*wearing a yellow shirt”

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

“A photo ofV*with a blue house in the background”

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

“A photo ofV*with a mountain in the background”

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

“A photo ofV*with a city in the background”

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

“A photo ofV*in the snow”

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

“A photo ofV*in the jungle”

- Figure S6. Qualitative comparison of image and text alignment on the <cat> subset of DB dataset.

“A photo ofV*on the beach”

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

“A photo ofV*on top of pink fabric”

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

“A photo ofV*wearing asantahat”

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

“A photo ofV*wearing a rainbow scarf”

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

“A photo ofV*with a mountain in the background”

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

“A photo of a purpleV*”

- Figure S7. Qualitative comparison of image and text alignment on the <cat2> subset of DB dataset.

“A photo ofV*in the jungle”

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

“A photo ofV*on a cobblestone street”

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

“A photo ofV*with a mountain in the background”

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

“A photo ofV*in a chef outfit”

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

“A photo ofV*in a firefighter outfit”

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

“A photo of a wetV*”

- Figure S8. Qualitative comparison of image and text alignment on the <dog6> subset of DB dataset.

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

“A photo ofV*on top of pink fabric”

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

“A photo ofV*floating on top of water”

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

“A photo ofV*on top of a dirt road”

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

“A photo ofa purpleV*”

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

“A photo ofa cube shapedV*”

- Figure S9. Qualitative comparison of image and text alignment on the <pink sunglasses> subset of DB dataset.

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

“A photo ofV*onthebeach”

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

“A photo ofV*ontopofawoodenfloor”

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

“A photo ofV*withamountaininthebackground”

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

“A photoofV*floating on top of water”

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

“A photo ofV*on top of a dirt road”

- Figure S10. Qualitative comparison of image and text alignment on the <shiny sneaker> subset of DB dataset.

### References

- [1] Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. A neural space-time representation for text-toimage personalization. ACM TOG, 42(6):1–10, 2023. 1
- [2] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-toimage diffusion models with an ensemble of expert denoisers. arXiv:2211.01324, 2022. 2, 5
- [3] Yoshua Bengio, Nicholas L´eonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv:1308.3432, 2013. 2, 4
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv:2311.15127, 2023. 1
- [5] Raghu Bollapragada, Richard Byrd, and Jorge Nocedal. Adaptive sampling strategies for stochastic optimization. SIAM Journal on Optimization, 28(4):3312–3343, 2018. 3
- [6] HanQin Cai, Daniel McKenzie, Wotao Yin, and Zhenliang Zhang. Zeroth-order regularized optimization (zoro): Approximately sparse gradients and adaptive sampling. SIAM Journal on Optimization, 32(2):687–714, 2022. 3, 4
- [7] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM TOG, 42(4):1–10, 2023. 2, 5
- [8] Aochuan Chen, Yimeng Zhang, Jinghan Jia, James Diffenderfer, Jiancheng Liu, Konstantinos Parasyris, Yihua Zhang, Zheng Zhang, Bhavya Kailkhura, and Sijia Liu. Deepzero: Scaling up zeroth-order optimization for deep model training. arXiv:2310.02025, 2023. 3, 4
- [9] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-\sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv:2403.04692, 2024. 1
- [10] Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, and Sungroh Yoon. Perception prioritized training of diffusion models. In CVPR, pages 11472– 11481, 2022. 2, 5
- [11] CrossLabs. Diffusion with offset noise. Technical report, CrossLabs, 2023. 14
- [12] Giannis Daras and Alexandros G Dimakis. Multiresolution textual inversion. NeurIPSW, 2022. 2, 5
- [13] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. NeurIPS, 36, 2024. 3, 5, 6, 9
- [14] John C Duchi, Michael I Jordan, Martin J Wainwright, and Andre Wibisono. Optimal rates for zero-order convex optimization: The power of two function evaluations. IEEE Transactions on Information Theory, 61(5):2788– 2806, 2015. 3
- [15] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik

Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 1

- [16] Zhengcong Fei, Mingyuan Fan, and Junshi Huang. Gradientfree textual inversion. In ACM MM, pages 1364–1373, 2023.

- 1, 3, 5, 6, 9, 11

[17] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv:2208.01618, 2022. 1,

- 2, 3, 4, 5, 6, 9

- [18] Saeed Ghadimi and Guanghui Lan. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM journal on optimization, 23(4):2341–2368,

2013. 3

- [19] Hyojun Go, Yunsung Lee, Seunghyun Lee, Shinhyeok Oh, Hyeongdon Moon, and Seungtaek Choi. Addressing negative transfer in diffusion models. NeurIPS, 36, 2024. 2, 5
- [20] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. NeurIPS, 36, 2024. 1
- [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. 1, 4
- [22] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv:2210.02303, 2022. 1
- [23] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. NeurIPS, 35:8633–8646, 2022. 1
- [24] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv:2106.09685, 2021. 2
- [25] Gwanghyun Kim, Alonso Martinez, Yu-Chuan Su, Brendan Jou, Jos´e Lezama, Agrim Gupta, Lijun Yu, Lu Jiang, Aren Jansen, Jacob Walker, et al. A versatile diffusion transformer with mixture of noise levels for audiovisual generation. arXiv:2405.13762, 2024. 1
- [26] Jeonghoon Kim, Jung Hyun Lee, Sungdong Kim, Joonsuk Park, Kang Min Yoo, Se Jung Kwon, and Dongsoo Lee. Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization. NeurIPS, 36,

2024. 1, 3, 5, 6, 9

- [27] Raghuraman Krishnamoorthi. Quantizing deep convolutional networks for efficient inference: A whitepaper. arXiv:1806.08342, 2018. 4
- [28] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, pages 1931–1941,

2023. 1, 2

- [29] Kyungmin Lee, Sangkyung Kwak, Kihyuk Sohn, and Jinwoo Shin. Direct consistency optimization for compositional text-

to-image personalization. arXiv preprint arXiv:2402.12004,

2024. 9

- [30] Yunsung Lee, JinYoung Kim, Hyojun Go, Myeongho Jeong, Shinhyeok Oh, and Seungtaek Choi. Multi-architecture multi-expert diffusion models. In AAAI, pages 13427–13436,

2024. 2, 5

- [31] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In CVPR, pages 300–309, 2023. 1
- [32] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 5404–5411, 2024. 14
- [33] Zhenqing Ling, Daoyuan Chen, Liuyi Yao, Yaliang Li, and Ying Shen. On the convergence of zeroth-order federated tuning for large language models. In ACM SIGKDD, pages 1827–1838, 2024. 8
- [34] Enshu Liu, Xuefei Ning, Zinan Lin, Huazhong Yang, and Yu Wang. Oms-dpm: Optimizing the model schedule for diffusion probabilistic models. In ICML, pages 21915–21936. PMLR, 2023. 2, 5
- [35] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In ICCV, pages 9298– 9309, 2023. 1
- [36] Sijia Liu, Pin-Yu Chen, Bhavya Kailkhura, Gaoyuan Zhang, Alfred O Hero III, and Pramod K Varshney. A primer on zeroth-order optimization in signal processing and machine learning: Principals, recent advances, and applications. IEEE Signal Processing Magazine, 37(5):43–54, 2020. 3
- [37] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv:2209.03003, 2022. 1
- [38] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv:2402.17177, 2024. 1
- [39] Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D Lee, Danqi Chen, and Sanjeev Arora. Finetuning language models with just forward passes. NeurIPS, 36:53038–53075, 2023. 1, 2, 3, 5, 8, 12
- [40] Yurii Nesterov and Vladimir Spokoiny. Random gradientfree minimization of convex functions. Foundations of Computational Mathematics, 17(2):527–566, 2017. 3, 4
- [41] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv:2304.07193, 2023. 5, 9
- [42] NaHyeon Park, Kunhee Kim, and Hyunjung Shim. Textboost: Towards one-shot personalization of text-to-image models via fine-tuning text encoder. arXiv:2409.08248,

2024. 1, 2, 5, 8

- [43] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. NeurIPS, 32, 2019. 9
- [44] Or Patashnik, Daniel Garibi, Idan Azuri, Hadar AverbuchElor, and Daniel Cohen-Or. Localizing object-level shape variations with text-to-image diffusion models. In CVPR, pages 23051–23061, 2023. 2, 5
- [45] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195–4205, 2023. 1
- [46] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv:2307.01952,

2023. 1, 14

- [47] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv:2209.14988, 2022. 1
- [48] Qualcomm. Unlocking on-device generative ai with an npu and heterogeneous computing. Technical report, Qualcomm,

2024. 1

- [49] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763. PMLR, 2021. 5, 9
- [50] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv:2204.06125, 1(2):3, 2022. 1
- [51] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 1, 3, 5, 9
- [52] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pages 22500–22510, 2023. 1, 2, 5, 6, 9, 10
- [53] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. In CVPR, pages 6527–6536, 2024. 1
- [54] Hyogon Ryu, Seohyun Lim, and Hyunjung Shim. Memory-efficient fine-tuning for quantized diffusion model. arXiv:2401.04339, 2024. 1, 3, 5, 6, 9
- [55] Hoigi Seo, Hayeon Kim, Gwanghyun Kim, and Se Young Chun. Ditto-nerf: Diffusion-based iterative text to omnidirectional 3d model. arXiv:2304.02827, 2023. 1
- [56] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, pages 2256–

2265. PMLR, 2015. 1

- [57] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv:2010.02502, 2020.

- [58] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. NeurIPS, 32, 2019. 1
- [59] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv:2011.13456, 2020. 1
- [60] James C Spall. Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. IEEE transactions on automatic control, 37(3):332–341, 1992. 3, 4
- [61] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization. In ACM SIGGRAPH, pages 1–11, 2023. 1
- [62] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-toimage generation. arXiv:2303.09522, 2023. 1, 2
- [63] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In CVPR, pages 12619–12629, 2023. 1
- [64] Daniel Watson, William Chan, Jonathan Ho, and Mohammad Norouzi. Learning fast samplers for diffusion models by differentiating through sample quality. In ICLR, 2022. 1
- [65] Chendong Xiang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. A closer look at parameter-efficient tuning in diffusion models. arXiv:2303.18181, 2023. 1
- [66] Penghang Yin, Jiancheng Lyu, Shuai Zhang, Stanley Osher, Yingyong Qi, and Jack Xin. Understanding straightthrough estimator in training activation quantized neural nets. arXiv:1903.05662, 2019. 2
- [67] Pengyun Yue, Long Yang, Cong Fang, and Zhouchen Lin. Zeroth-order optimization with weak dimension dependency. In COLT, pages 4429–4472. PMLR, 2023. 8
- [68] Yuxin Zhang, Weiming Dong, Fan Tang, Nisha Huang, Haibin Huang, Chongyang Ma, Tong-Yee Lee, Oliver Deussen, and Changsheng Xu. Prospect: Prompt spectrum for attribute-aware personalization of diffusion models. ACM TOG, 42(6):1–14, 2023. 1, 2, 5

