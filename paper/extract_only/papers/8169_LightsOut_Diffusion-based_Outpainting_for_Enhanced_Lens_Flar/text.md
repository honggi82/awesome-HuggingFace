## LightsOut: Diffusion-based Outpainting for Enhanced Lens Flare Removal

Shr-Ruei Tsai Wei-Cheng Chang Jie-Ying Lee Chih-Hai Su Yu-Lun Liu National Yang Ming Chiao Tung University

# arXiv:2510.15868v1[cs.CV]17Oct2025

[Figure 1]

|[Figure 2]|
|---|

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Origin + Ours

Origin + Ours

[Figure 8]

Input image

Off-the-Shelf Flare Removal Models

[Figure 9]

|[Figure 10]<br><br>| |
|---|
|
|---|

[Figure 11]

[Figure 12]

[Figure 13]

| |
|---|

| |
|---|

| |
|---|

[Figure 14]

[Figure 15]

Predicted light source

Origin + Ours Origin + Ours

Image outpainted with full light source

Figure 1. Illustration of our diffusion-based outpainting method. Given an input image with incomplete or missing off-frame light sources, existing Single Image Flare Removal (SIFR) models struggle to effectively remove lens flare artifacts due to incomplete context. Our proposed approach accurately predicts and outpaints the off-frame light sources, allowing subsequent SIFR models to perform significantly better. As demonstrated, integrating our outpainting strategy as a plug-and-play preprocessing step substantially enhances flare removal quality and visual realism.

### Abstract

Lens flare significantly degrades image quality, impacting critical computer vision tasks like object detection and autonomous driving. Recent Single Image Flare Removal (SIFR) methods perform poorly when off-frame light sources are incomplete or absent. We propose LightsOut, a diffusionbased outpainting framework tailored to enhance SIFR by reconstructing off-frame light sources. Our method leverages a multitask regression module and LoRA fine-tuned diffusion model to ensure realistic and physically consistent outpainting results. Comprehensive experiments demonstrate LightsOut consistently boosts the performance of existing SIFR methods across challenging scenarios without additional retraining, serving as a universally applicable plug-and-play preprocessing solution. Project page: https://ray-1026.github.io/lightsout/

### 1. Introduction

Lens flare, categorized into reflective flares, scattering flares, and lens orbs (backscatter) [26, 34, 71], significantly degrades image quality and negatively impacts computer vi-

sion tasks such as object detection and autonomous driving. Traditional flare removal methods [1, 7, 62] relied on handcrafted cues like intensity thresholding or template matching but struggled with complex artifacts. Recent deep learning approaches, such as U-Net [71] and Uformer [11], have achieved substantial improvements due to dedicated datasets. Nevertheless, Single-Image Flare Removal (SIFR) remains challenging, particularly in nighttime scenarios, due to limited real-world paired training data.

Recent SIFR advances primarily focus on dataset construction and architectural improvements. Wu et al.[71] introduced a flare dataset with captured and simulated images. Flare7k[11] and its enhanced version, Flare7k++[13], provide extensive synthetic and real flare data. Recent methods like Difflare[86] and MFDNet [28] employ sophisticated architectures, such as diffusion models and multi-scale processing, achieving state-of-the-art performance.

Despite these advancements, current SIFR methods still struggle when images lack complete views of off-frame light sources. As illustrated in Fig. 2, existing methods significantly degrade in such scenarios, resulting in increased flare artifacts and reduced realism. Metrics like PSNR and LPIPS [83] confirm that complete light source information is crucial for effective flare removal.

|[Figure 16]<br><br>| |
|---|
<br><br>|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

| |
|---|

| |
|---|

| |
|---|

NolightsourceFulllightsource

|[Figure 20]|
|---|

|[Figure 21]|
|---|

|[Figure 22]|
|---|

|[Figure 23]|
|---|

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Input

Zhou, et al

Flare7K++

MFDNet

- Figure 2. Motivation for outpainting incomplete off-frame light sources. (a) With complete off-frame light sources, state-of-the-art SIFR methods effectively remove lens flare artifacts. (b) In scenarios lacking complete views of off-frame sources, these methods degrade significantly, leaving noticeable artifacts. This highlights the importance of complete light source context, motivating our proposed outpainting solution.

To address this, we propose LightsOut, a diffusion-based outpainting method designed for accurate completion of missing off-frame light sources. Our approach integrates a multitask regression module to predict light source parameters precisely, alongside LoRA [25] fine-tuning of a stable diffusion inpainting model explicitly conditioned on these predictions. This ensures that generated content aligns closely with real-world flare and illumination distributions.

As shown in Fig. 1, our method seamlessly integrates as a plug-and-play preprocessing step with existing SIFR frameworks, significantly enhancing performance in challenging scenarios. Extensive quantitative and qualitative evaluations confirm that LightsOut consistently improves the realism and effectiveness of flare removal across various state-of-the-art methods.

Our contributions are summarized as follows:

- • We identify and address a key limitation of SIFR methods dealing with incomplete off-frame light sources through a specialized decomposition strategy.
- • We propose a LoRA fine-tuned diffusion-based model that accurately reconstructs physically consistent off-frame light sources and flare artifacts.
- • We introduce a plug-and-play preprocessing framework that universally enhances existing SIFR models without additional retraining.

### 2. Related Work

Lens Flare Removal. Physical lens modifications [1] become ineffective with strong light sources [12]. Early approaches followed a flare detection-then-removal pipeline [1, 7, 62], focused on veiling glare removal [52, 60] and

reflective flare removal [1, 7, 47, 47, 62]. Deep learning approaches were limited by paired data scarcity. Wu et al. [71] developed a dataset with a U-Net-based [54] approach. Dai et al. [11] expanded this with physics-based simulation [26, 31], later enhanced as Flare7K++ [13]. Qiao et al. [50] used unpaired data with a CycleGAN-inspired [88] framework. Recent architectures like MFDNet [34] and Zhou et al.’s approach [85] further improved performance. Despite advances, current models degrade significantly when offframe light sources are incomplete or absent, which is the limitation our work addresses.

Adapting Diffusion Models. Diffusion models [20, 44, 59] have been adapted for image generation, editing, and restoration [8, 9, 24, 45, 53, 56, 57, 69, 78]. Early applications to inpainting by Sohl-Dickstein et al. [58] and Song et al. [59] showed promise. Guided synthesis approaches [10, 41] offered alternative conditioning strategies but have limitations for outpainting tasks. Several approaches focus on fine-tuning pre-trained diffusion models [55, 74] or text embeddings [14, 63] using a single or a few reference object images. Additionally, to improve fine-tuning efficiency, researchers have proposed methods that simplify the adaptation process of diffusion models [18, 25, 51]. Low-Rank Adaptation (LoRA) [25] has emerged as an efficient finetuning approach. However, existing diffusion models lack physics-based modeling of optical phenomena [22, 26] and struggle with contextual extrapolation beyond image boundaries.

Image Completion and Outpainting. Traditional image completion used low-level cues [2–4, 19]. GAN-based methods [15, 48] introduced encoder-decoder architectures with innovations including dilated convolutions [27, 79], partial/gated convolutions [37, 81], contextual attention [80], edge maps [17, 43, 72, 73], and semantic segmentation [23, 46]. Outpainting techniques include semantic regeneration networks [66], edge-guided models [36], spiral generation [16], and RCT blocks with LSTM [21, 75]. Recent transformer-based methods [68, 76] still struggle with physically consistent light sources and flare extrapolation. Diffusion-based approaches [38, 39, 41, 53, 56, 65, 70] demonstrate generative capabilities but lack explicit modeling of light sources and optical effects.

Image Conditioned Diffusion Models. Diffusion models excel in generative tasks but struggle with fine-grained controllability. ControlNet [82] and IP-Adapter [77] enhance reference-based generation by integrating input image features, while T2I-Adapter [42] conditions on external modalities like sketches and keypoints. InstructPix2Pix [5] enables explicit attribute control via user instructions, and methods [40, 41] refine generation using stochastic differential equations. Building on these advances, our study leverages the conditional techniques by introducing light source constraints as a novel conditioning factor.

Input Image 𝐼𝐼

[Figure 28]

[Figure 29]

|[Figure 30]|
|---|

: Trainable : Frozen

[Figure 31]

BLIP-2

| | |
|---|---|
|Canvas a prepar|nd mask ation|
| | |

###### "𝑎𝑎 𝑟𝑟𝑜𝑜𝑜𝑜𝑚𝑚 𝑤𝑤𝑖𝑖𝑡𝑡ℎ 𝑎𝑎 𝑝𝑝𝑎𝑎𝑖𝑖𝑛𝑛𝑡𝑡𝑖𝑖𝑛𝑛𝑔𝑔 𝑜𝑜𝑛𝑛 𝑡𝑡ℎ𝑒𝑒 𝑤𝑤𝑎𝑎𝑙𝑙𝑙𝑙” + ", 𝑓𝑓𝑢𝑢𝑙𝑙𝑙𝑙 𝑙𝑙𝑖𝑖𝑔𝑔ℎ𝑡𝑡 𝑠𝑠𝑜𝑜𝑢𝑢𝑟𝑟𝑐𝑐𝑒𝑒 𝑤𝑤𝑖𝑖𝑡𝑡ℎ 𝑓𝑓𝑙𝑙𝑎𝑎𝑟𝑟𝑒𝑒"

[Figure 32]

[Figure 33]

SIFR Models

1

|[Figure 34]|
|---|

SDInpainting

[Figure 35]

Mask 𝑀𝑀

LoRA

[Figure 36]

|[Figure 37]|
|---|

Crop

Outpainted Image 𝐼𝐼out

Masked image 𝐼𝐼M

Flare-free Image 𝐼𝐼free

Noise Reinjection

[Figure 38]

|[Figure 39]|
|---|

[Figure 40]

|[Figure 41]|
|---|

Lightsource

condition

module

|Rendering Fucntion|
|---|

Multitask Light Regression Module

Cropped flare-free Image 𝐼𝐼final

Light Source Mask 𝑀𝑀L

(a) Stage 1: Light source prediction and conditioning (b) Stage 2: Light source outpainting (c) Stage 3: SIFR boosting

- Figure 3. Overview of our proposed three-stage pipeline. (a) Light source prediction and conditioning: We introduce a multitask regression module to accurately predict off-frame or incomplete light source parameters (positions, radii, and confidences). These predicted parameters guide a rendering function to generate the corresponding light source mask. (b) Light source outpainting: Leveraging a LoRA fine-tuned diffusion-based inpainting model with light source conditioning, our approach accurately outpaints both missing off-frame light sources and associated flare artifacts, producing visually coherent and realistic results. (c) SIFR boosting: Our generated outpainted images serve as enhanced inputs to existing SIFR methods, significantly improving their performance on previously challenging scenarios with incomplete light source information. The proposed pipeline thus effectively operates as a plug-and-play module to boost existing flare removal models.

### 3. Method

Our objective is to resolve the issue of the degradation in existing SIFR models with limited light source information.

- As illustrated in Fig. 3, we propose a three stage method: (a)

Given a flare-corrupted input image Iin ∈ RH×W×3 with an incomplete light source, we first define an outpainted region, producing the masked image IM ∈ RH

′×W′×3, where H′ > H and W′ > W, and its corresponding binary mask M ∈ {0,1}H

′×W′×3. Then, we predict the light source mask ML for the guided condition (Sec. 3.3). (b) After preparing the images and corresponding guided conditions, our outpainting approach completes the scene and reconstructs the full light source, constrained by ML. The resulting outpainted image, denoted as Iout ∈ RH

′×W′×3, is generated as described in Sec. 3.2. (c) The generated image Iout is then processed by a SIFR model, yielding the flare-free image Ifree ∈ RH

′×W′×3. Finally, we extract the origin region in Iin from Ifree to get the final flare-free image Ifinal ∈ RH×W×3 (Sec. 3.4). This threestage pipeline effectively addresses the limitations of existing SIFR models by enhancing light source reconstruction before flare removal.

#### 3.1. Preliminaries

Diffusion models. transform a simple Gaussian noise distribution into the target data distribution. During the train-

ing phase, these models progressively add Gaussian noise to the original data x0. Formally, at each diffusion step

t√, 1the− αnoisytϵ, wheresampleϵ x∼t canN(0be,I)writtenis a Gaussianas xt = noise√αtxand0 + αt is a variance-scheduling parameter that governs how much noise is added. A neural network ϵθ then learns to predict noise ϵ from the noisy sample xt by minimizing:

L = Ex,t,ϵ ϵθ(xt,t,c) − ϵ 22 . where c denotes conditioning signal such as text or masked images. At the inference

stage, the model starts from random noise and iteratively denoises the sample until it converges to a data point in the target distribution.

LoRA fine-tuning. Instead of updating all parameters of the weight matrix W ∈ Rm×n in the denoising U-Net, LoRA introduces a low-rank decomposition by injecting a small trainable matrix ∆Wi = AiBi, where Ai ∈ Rm×d,Bi ∈ Rd×n, with d ≪ n. The final weight matrix can be written as Wi′ = Wi + ∆Wi, where only the added ∆W is optimized during training, while the original weights Wi remain frozen.

#### 3.2. Image Outpainting for Light Source

Existing diffusion-based outpainting models often lack the task-specific adaptation necessary for high-quality light source reconstruction. To address this, we fine-tune the pretrained Stable Diffusion v2 inpainting model, enabling real-

istic and structurally consistent light source generation.

Training. To generate the outpainted results with complete light source, we inject LoRA weights and fine-tune them on the given IM and M. The loss function is

L = Ex,t,ϵ,m ϵθ(xt,t,p,M,IM) − ϵ 22 , (1)

where x ∈ IM, p represents a text prompt derived from the input image using BLIP-2 [32], and ⊙ denotes the elementwise product.

Inference. After training, the diffusion-based outpainting model aims to predict missing pixels at the corners of the masked region while preserving the integrity of the existing regions in IM. One approach is ensuring that the masked regions are modified by incorporating the intermediate noisy state of the source data from the corresponding timestep in the forward diffusion process. This can be formulated as follows:

xmaskedt−1 = √α¯tx0 + √1 − α¯tϵ, ϵ ∼ N(0,I), xunmaskedt−1 = µθ(xt,t) + σθ(xt,t) · ϵ, ϵ ∼ N(0,I), xt−1 = M ⊙ xmaskedt−1 + (1 − M) ⊙ xunmaskedt−1 .

(2)

where µθ(xt,t) and σθ(xt,t) are the predicted mean and variance from the denoising model. Since our method builds on the Stable Diffusion inpainting model, the operations in Eq. (2) are performed in latent space. However, as observed in prior works [61, 89], this can still introduce distortions in preserved regions of IM, resulting in inconsistencies in the generated output Iout. To resolve this, instead of combining in latent space, we use the mask M to perform alpha composition between Iout and IM directly in the RGB space. It ensures that Iout can be with full recovery on the existing area and a smooth transition at the boundary of the generated region.

Noise reinjection. Although we composite in RGB space, noticeable discrepancies arise between masked and unmasked regions in the final output. As described in Eq. (2), the denoising steps treat the masked and unmasked regions as separate entities. This can cause inconsistencies and error accumulation across denoising steps. To mitigate this, we adopt noise reinjection, inspired by prior works [40, 64], as formalized in Algorithm 1. This technique reintroduces noise at intermediate steps, allowing the model to re-denoise and better align with the correct distribution.

#### 3.3. Light Source Prediction and Conditioning

Our outpainting approach (Sec. 3.2) leverages Stable Diffusion’s generative capabilities to reconstruct scenes and their light sources. However, it faces limitations, including spatial misalignment, incomplete synthesis, and inconsistent handling of multiple light sources. To address these, we propose multitask regression and light source conditioning modules to enhance outpainting accuracy and realism.

Algorithm 1 Noise reinjection

Require: Masked image IM, binary mask M, Pretrained SD inpainting model ϵθ, Timesteps sequence {t1, t2, . . . , tN}, and Repeat time R

Ensure: Outpainted image Iout

- 1: r ← R
- 2: xT ∼ N(0, I)
- 3: i ← N − 1
- 4: while i ≥ 0 do
- 5: t ← ti, tprev ← ti+1
- 6: zt ∼ N(0, I)
- 7: xt−1 = √α¯t−1

xt −

√1 − α¯tϵθ(xt, IM, M, t) √α¯t

+ 1 − α¯tprev − σt2 · ϵθ(xt, IM, M, t) + σtzt

- 8: if i > 0 and r > 0 then
- 9: xt ∼ N

√αtxt−1, √1 − αtI ▷ Noise reinjection

- 10: r ← r − 1
- 11: else
- 12: i ← i − 1
- 13: r ← R
- 14: end if
- 15: end while
- 16: Iout = x0
- 17: return Iout

Multitask regression module. We tackle the challenging task of predicting light sources within masked regions of IM, which is more complex than prediction from complete images. Unlike conventional U-Net [54] methods generating full maps, we adopt a parameterized regression approach, modeling sources as circular entities to reduce computational cost, stabilize training, and ensure physically meaningful results.

The multitask regression module, as shown in Fig. 4, predicts N sets of (x,y,r) parameters to environments with multiple light sources, where (x,y) are the planar coordinates, r is the radius, and N serves as a hyperparameter. Since real-world scenes rarely contain exactly N light sources, we introduce a confidence score to estimate the existence probability of each predicted source. The proposed architecture consists of a CNN-based feature extractor Fθ and two specialized MLPs: Gϕ for estimating physical parameters and Hψ for computing confidence scores. This architecture can be formally expressed as:

##### P c = Gϕ(Fθ(Itgt)) Hψ(Fθ(Itgt)) , (3)

 

 

⊤

- x1 x2 ... xN
- y1 y2 ... yN r1 r2 ... rN

where P =

∈ RN×3 represents the matrix of predicted light source parameters, and c = c1 c2 ... cN ⊤ ∈ [0,1]N×1 denotes the vector of confidence scores for each predicted light source. The optimiza-

Training

[Figure 42]

𝐏𝐏𝐠𝐠𝐠𝐠

𝓛𝓛𝐩𝐩𝐩𝐩𝐩𝐩

MLP 𝐏𝐏

| | |
|---|---|
| | |

|[Figure 43]|
|---|

[Figure 44]

𝓖𝓖𝝓𝝓

CNN

[Figure 45]

|MLP|
|---|

𝐜𝐜𝐠𝐠𝐠𝐠

𝐜𝐜

𝓛𝓛𝐜𝐜𝐩𝐩𝐜𝐜𝐜𝐜

Input image

𝓕𝓕𝜽𝜽

Multi-task Loss

𝓗𝓗𝝍𝝍

𝑁𝑁

Inference

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

|MLP|
|---|

[Figure 50]

𝐏𝐏

|[Figure 51]|
|---|

𝓖𝓖𝝓𝝓

[Figure 52]

Sigmoid

|[Figure 53]|
|---|

CNN

[Figure 54]

Thresholding ∑

𝐜𝐜

Input image 𝓕𝓕𝜽𝜽 MLP

Predicted light source mask 𝑀𝑀L

𝓗𝓗𝝍𝝍

- Figure 4. Overview of the multitask regression module. Our model performs multitask regression to simultaneously predict two essential components: the physical parameters P and the corresponding confidence probabilities c for potential light sources. During training, these are supervised by a designed multitask loss.

- At inference, predicted parameters are integrated to generate light source masks ML.

tion is designed with the position loss Lpos:

smoothL1(P − Pgt), (4)

Lpos(P,Pgt) =

i∈{x,y,r}

in which

0.5x2, if |x| < 1 |x| − 0.5 otherwise,

(5)

smoothL1(x) =

where Pgt represents the ground truth of the physical parameters. To ensure permutation invariance in the matching between predicted and ground-truth parameters, we adopt a bipartite matching strategy [6] to obtain the optimal assignment. Since our task involves predicting both the number and spatial distribution of light sources, we introduce an additional confidence loss Lconf to supervise the prediction of existence probabilities. The confidence loss is formulated as the binary cross-entropy between the predicted c and the ground truth cgt:

Lconf = −

i

(cgt,i,log ci + (1 − cgt,i)log(1 − ci)). (6)

To optimize only relevant predictions, we compute the position loss exclusively for pairs where cgt = 1, ignoring cases where cgt = 0 as these correspond to non-existent light sources. Furthermore, to address the inherent uncertainty in predicting both light source locations and their existence, we introduce an uncertainty-aware weighting mechanism[30, 35]. Specifically, we model two learnable parameters, σ1 and σ2, and define the total loss function as:

- 1

- 2σ12Lpos+

- 1

- 2σ22Lconf+log(1+σ12)+log(1+σ22) (7)

L =

This allows the network to adaptively balance between the losses, leading to more robust learning in either position or confidence predictions.

Rendering function. During inference in the multitask regression module, we perform a forward pass to obtain the predicted P and c. To generate a spatial representation of the light sources, we apply an activation function and a confidence threshold to suppress unreliable predictions, as illustrated in Fig. 4. The final predicted light source mask ML is computed as:

N

c˜i · σ ri − (x − xi)2 + (y − yi)2 ,

ML(x,y) =

i=1

(8) where c˜i denotes the c after thresholding, and σ(·) is the sigmoid function. The design of the function helps reconstruct the spatial representation by suppressing unreliable predictions.

Light source condition module is designed to guide our outpainting approach mentioned in Sec. 3.2 by leveraging ML, ensuring that light sources are generated in physically plausible locations. The module employs a learnable mechanism that conditions the generative process on the provided light source map, allowing explicit control over light placement. During optimization, we enforce an L2 loss:

Llight = M ˜L − ML 22 , (9)

where M˜L represents the generated light source map. After training, we integrate the model into our outpainting model as constraints to effectively guide the generated content.

#### 3.4. Flare Removal Methods

Our method reconstructs incomplete or off-frame light sources, providing accurate illumination context and enhancing flare removal effectiveness. It serves as a model-agnostic preprocessing step, seamlessly integrating into existing SIFR pipelines without architectural modifications.

### 4. Experiments 4.1. Experimental Setup

Training dataset. We train on Flare7K [11], following its established synthesis pipeline. Background images from Flickr24K [84] are composited with reflective and scattering flares. To simulate incomplete off-frame scenarios, we apply luminance masks to define realistic regions for outpainting. Evaluation dataset. We evaluate our approach on Flare7K’s test set, comprising 100 real and 100 synthetic images in two scenarios: (1) no light source and (2) incomplete light sources created by shifting the boundary outward by 15 pixels from scenario (1). This setup assesses our method’s effectiveness in handling incomplete illumination.

Table 1. Quantitative evaluation against state-of-the-art diffusion-based outpainting methods. We comprehensively compare our method with baseline and existing diffusion-based inpainting and outpainting approaches. Our solution demonstrates superior performance across diverse scenarios, validating the effectiveness of our diffusion-based strategy in enhancing subsequent flare removal tasks.

Flare7k Real Flare7k Synthetic PSNR ↑ SSIM ↑ LPIPS ↓ G-PSNR ↑ S-PSNR ↑ PSNR ↑ SSIM ↑ LPIPS ↓ G-PSNR ↑ S-PSNR ↑

Setting SIFR Model Method

Direct input 26.42 0.8770 0.0445 22.04 20.01 31.86 0.9499 0.0181 25.16 23.92 SD-Inpainting [53] 26.84 0.8823 0.0449 22.73 20.04 31.39 0.9518 0.0185 24.62 23.92 SDXL-Inpainting [49] 26.00 0.8774 0.0474 21.27 19.53 30.80 0.9506 0.0196 23.94 23.23 PowerPaint [90] 26.59 0.8736 0.0559 22.66 20.66 28.90 0.9262 0.0456 23.32 22.72

Zhou et al. [87]

- Ours 27.09 0.8856 0.0424 23.07 21.12 31.56 0.9534 0.0181 24.74 24.17

Flare7k++ [13]

Direct input 26.29 0.8337 0.0442 21.35 18.71 31.28 0.9685 0.0151 23.51 23.59 SD-Inpainting [53] 27.98 0.8938 0.0421 23.63 21.12 33.43 0.9704 0.0129 25.86 25.84 SDXL-Inpainting [49] 27.01 0.8893 0.0452 22.13 19.65 31.63 0.9675 0.0151 23.87 24.03 PowerPaint [90] 27.10 0.8814 0.0839 22.20 20.92 29.24 0.9289 0.0890 23.86 23.62

- Ours 28.41 0.8956 0.0397 24.15 22.83 33.91 0.9719 0.0120 26.24 26.59

Nolightsource

Direct input 27.04 0.8904 0.0463 22.37 19.94 33.42 0.9721 0.0122 26.43 26.24 SD-Inpainting [53] 26.82 0.8886 0.0483 22.21 18.54 32.42 0.9676 0.0145 25.73 24.93 SDXL-Inpainting [49] 25.55 0.8816 0.0535 21.23 16.66 29.95 0.9605 0.0204 23.21 20.91 PowerPaint [90] 25.28 0.8746 0.0509 20.49 17.11 27.57 0.9181 0.0952 22.06 20.61 Ours 27.43 0.8940 0.0451 22.97 20.49 33.54 0.9714 0.0119 26.53 25.89

MFDNet [28]

Direct input 26.05 0.8771 0.0480 21.88 19.92 30.03 0.9464 0.0210 24.24 23.14 SD-inpainting [53] 26.42 0.8817 0.0469 22.59 20.14 30.07 0.9483 0.0212 24.01 23.43 SDXL-Inpainting [49] 25.50 0.8773 0.0492 21.34 19.59 29.61 0.9466 0.0227 23.35 23.12 PowerPaint [90] 25.87 0.8716 0.0595 22.12 20.50 27.92 0.9227 0.0486 22.73 22.34 Ours 26.29 0.8842 0.0453 22.68 20.80 30.11 0.9504 0.0202 24.06 23.50

Zhou et al. [87]

Incompletelightsource

Direct input 26.07 0.8333 0.0463 21.58 18.34 30.23 0.9672 0.0160 23.88 23.70 SD-Inpainting [53] 28.02 0.8944 0.0431 23.81 21.71 31.02 0.9671 0.0153 24.70 25.06 SDXL-Inpainting [49] 26.99 0.8906 0.0453 22.41 20.12 30.33 0.9657 0.0164 23.73 23.98 PowerPaint [90] 26.85 0.8802 0.0869 22.94 20.76 28.00 0.9253 0.0923 23.27 23.12 Ours 28.15 0.8957 0.0409 24.20 22.24 31.38 0.9682 0.0144 25.01 25.57

Flare7k++ [13]

Direct input 26.53 0.8886 0.0457 22.07 20.08 31.52 0.9701 0.0137 25.90 25.45 SD-Inpainting [53] 26.48 0.8884 0.0496 22.07 19.09 31.32 0.9672 0.0155 25.49 24.79 SDXL-Inpainting [49] 24.90 0.8807 0.0565 20.78 16.27 29.63 0.9605 0.0208 23.57 22.27 PowerPaint [90] 24.88 0.8728 0.0535 20.10 16.66 26.89 0.9156 0.0982 21.87 20.48 Ours 26.94 0.8922 0.0457 22.75 20.43 31.60 0.9696 0.0136 26.17 25.47

MFDNet [28]

Baseline methods. We compare with state-of-the-art SIFR methods: Zhou et al. [87], Flare7K++ [13], and MFDNet [28] 1. Additionally, our diffusion-based outpainting approach compares with state-of-the-art diffusionbased inpainting and outpainting models, including SDInpainting [53], SDXL-Inpainting [49], and PowerPaint [90].

#### 4.2. Quantitative Evaluations

Comparisons with Existing Methods. Tab. 1 compares our approach with two categories of baselines: state-of-the-art SIFR models and diffusion-based outpainting methods. Additionally, we report G-PSNR and S-PSNR [13], which assess flare removal performance in the glare and streak regions, as also shown in Tab. 1. Our approach significantly boosts performance, notably increasing PSNR from 26.29 dB to 28.41 dB with Flare7K++ [13] on real images without light sources. It also outperforms existing diffusion-based methods on both real and synthetic datasets, effectively addressing cases with incomplete illumination.

Evaluation of multitask regression-based light source prediction. Tab. 2 compares our multitask regression-based module with a baseline UNet approach using mIoU. Our

1The recent work Difflare [86] does not provide publicly available implementations or pre-trained models, making its inclusion in our comprehensive evaluation infeasible.

Table 2. Quantitative evaluation of our proposed light source prediction method. We compare our multitask regression module against a baseline UNet-based approach using mIoU scores. These results demonstrate the effectiveness and reliability of our regression-based strategy.

Method Flare7K Real Flare7K Synthetic

UNet 0.6216 0.6563 Ours 0.6310 0.6619

method achieves superior mIoU scores of 0.6310 (real) and 0.6619 (synthetic), outperforming the baseline’s 0.6216 and 0.6563, respectively.

#### 4.3. Qualitative Evaluations

Qualitative impact of off-frame light source completion. Fig. 5 compares the qualitative results of existing SIFR models with and without our method. Without outpainting, these models produce noticeable residual flares and lower realism. Integrating our diffusion-based preprocessing significantly improves off-frame context, enabling more effective flare removal and results closely resembling ground truth.

Comparison with standard diffusion-based outpainting methods. We compare our outpainting results qualitatively against other diffusion-based methods in Fig. 6. Standard diffusion methods often generate unrealistic off-frame content

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

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

IncompletelightsourceNolightsource

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

| |
|---|

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

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

###### Ours +

###### Ours +

Ours + MFDNet

Zhou, et al Flare7K++

Flare7k++ MFDNet

Input Zhou, et al Ground truth

- Figure 5. Qualitative comparison of lens flare removal results. We compare state-of-the-art SIFR methods (Zhou et al.[87], Flare7K++[13], and MFDNet [28]) alone and combined with our proposed method in two challenging scenarios: (top) no visible light sources and (bottom) incomplete light sources. Integrating our outpainting method (“Ours +”) significantly improves flare removal quality, producing results closer to ground truth.

SD-Inpainting SDXL-Inpainting PowerPaint Ours

[Figure 88]

| |
|---|

[Figure 89]

| |
|---|

[Figure 90]

| |
|---|

[Figure 91]

| |
|---|

[Figure 92]

| |
|---|

[Figure 93]

| |
|---|

[Figure 94]

| |
|---|

[Figure 95]

| |
|---|

- Figure 6. Qualitative comparison of our outpainting results. We qualitatively compare our method with SD-Inpainting [53], SDXLInpainting [49], and PowerPaint [90]. Our method produces more realistic outpainting results, accurately capturing flare artifacts and aligning closely with real-world scenes.

- Table 3. Ablation study on the SIFR model trained for the offframe setting. Although the baseline is explicitly designed and trained for the off-frame scenario, it still underperforms compared to our approach. ∗ indicates that the Flare7k++ baseline was retrained using both incomplete and no-light-source data.

Method

Flare7k Real (no light source) Flare7k Real (incomplete light source) PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Flare7k++∗ 27.03 0.8679 0.0467 26.18 0.8650 0.0500 Ours 28.41 0.8956 0.0397 28.15 0.8957 0.0409

- Table 4. Ablation study on noise reinjection strategy. We quantitatively evaluate the impact of noise reinjection during diffusionbased outpainting compared to a baseline without it. Results confirm the effectiveness of noise reinjection in enhancing flare removal performance.

Flare7k Real Flare7k Synthetic PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Noise regret

✗ 28.28 0.8949 0.0412 33.55 0.9704 0.0130 ✓ 28.41 0.8956 0.0397 33.91 0.9719 0.0120

due to the lack of explicit conditioning on light sources and flare distributions. In contrast, LightsOut, with the multitask regression module and LoRA-fine-tuned diffusion model, ensures coherent outpainting with accurate flare artifacts and seamless illumination, closely matching the original scene and surpassing baseline methods.

hyperparameters identical to the original setup. As shown in Tab. 3, despite being specifically designed and trained for the off-frame scenario, it still lags behind our method in performance. This confirms the effectiveness of our method in handling challenging cases where the light source lies outside the image frame.

#### 4.4. Ablation Study

Effectiveness of the SIFR model trained for the off-frame setting. We retrained the original SIFR model using both incomplete and no-light-source data, while keeping all training

Effectiveness of noise reinjection. We quantitatively and qualitatively evaluate the impact of noise reinjection during

###### (a) (b) (c)

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

| |
|---|

| |
|---|

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

| |
|---|

| |
|---|

w/o noise reinjection

w/ noise reinjection

Input image Laten space RGB space Input image w/o condition w/ condition

Input image

- Figure 7. Ablation studies. We ablate three components on the light source outpainting results:(a) Incorporating noise reinjection significantly enhances the diffusion model’s capability to produce realistic, seamless, and visually coherent outpainted regions; (b) Latent space blending tends to produce inconsistent illumination and noticeable artifacts, while blending in RGB space yields results with smoother transitions and improved alignment with real-world intensity distributions. (c) Integrating the proposed conditioning module significantly improves the accuracy and realism of the generated off-frame light sources and flare patterns.

Table 5. Ablation study comparing latent space blending versus RGB space blending. We conduct a quantitative evaluation of performance disparities between two distinct blending methodologies: latent space blending and RGB space blending. The results confirms that RGB space blending is more effectively boosting the SIFR baselines.

Flare7k Real Flare7k Synthetic PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Blending

Latent space 26.91 0.8859 0.0434 24.13 0.7156 0.0926 RGB space 27.09 0.8856 0.0424 31.55 0.9526 0.0178

diffusion-based outpainting. Tab. 4 shows that excluding noise reinjection notably reduces performance, especially LPIPS scores. Qualitative comparisons (Fig. 7(a)) confirm that noise reinjection significantly enhances visual coherence and realism, justifying its necessity.

RGB vs. Latent space blending. We analyze two blending strategies: latent space and RGB space. As shown in Tab. 6, RGB blending consistently outperforms latent space blending. Qualitative comparisons in Fig. 7(b) confirm that RGB blending offers smoother transitions and enhanced visual coherence, making it our preferred choice.

Impact of light source condition module. We validate the contribution of our light source condition module by comparing results with and without conditioning (Fig. 7(c)). Without conditioning, the diffusion model struggles to localize offframe sources. Incorporating our module consistently yields more accurate and realistic predictions.

Multitask regression for light source prediction. We evaluate our multitask regression against alternatives, including differentiable rendering [33] and weighted-sum loss formulations. Tab. 6 shows that our method consistently achieves superior accuracy, validating its robustness and design effectiveness.

Performance gap between SD-Inpainting and proposed method. To analyze the performance gap, we ablate two key

- Table 6. Ablation on optimization strategies for light source prediction. We compare three strategies: differentiable rendering [33], standard regression, and our multitask regression. Our multitask regression approach achieves superior mIoU, validating its simplicity and effectiveness for predicting off-frame light sources.

Optimization Strategy Flare7K Real Flare7K Synthetic

Differentiable rendering [33] 0.5212 0.5077 Regression with weighted loss 0.6081 0.6577 Multitask regression (ours) 0.6310 0.6619

- Table 7. Ablation on the performance gap between SDInpainting and ours. To assess the effectiveness of each component, we compare LoRA fine-tuning and our light source conditioning module. The results demonstrate that the combination of all components yields the best performance, indicating that each component is essential.

Flare7k Real PSNR ↑ SSIM ↑ LPIPS ↓ G-PSNR ↑ S-PSNR ↑

LoRA fine-tuning Condition Module

- - - 26.82 0.8886 0.0483 24.59 26.33

✓ - 27.12 0.8926 0.0453 24.83 26.59

- - ✓ 27.06 0.8906 0.0456 24.84 26.36

###### ✓ ✓ 27.43 0.8940 0.0451 25.21 26.79

components: (1) LoRA fine-tuning and (2) our light source condition module. As shown in Tab. 7, each yields a PSNR gain of 0.30/0.24 dB and an LPIPS drop of 0.0030/0.0027. Combining both achieves the best overall performance, confirming their complementary contributions.

### 5. Conclusion

LightsOut addresses the limitation of SIFR models caused by incomplete light source context. Our diffusion-based outpainting and conditioning modules effectively reconstruct off-frame illumination, significantly enhancing the flare removal performance of existing methods.

Limitations. The added outpainting stages introduce computational overhead. Future work could explore end-to-end optimization strategies to reduce this overhead.

Acknowledgements. This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628-EA49-023-. The authors are grateful to Google, NVIDIA, and MediaTek Inc. for their generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

### References

- [1] CS Asha, Sooraj Kumar Bhat, Deepa Nayak, and Chaithra Bhat. Auto removal of bright spot from images captured against flashing light source. In 2019 IEEE International Conference on Distributed Computing, VLSI, Electrical Circuits and Robotics, 2019. 1, 2
- [2] Coloma Ballester, Marcelo Bertalmio, Vicent Caselles, Guillermo Sapiro, and Joan Verdera. Filling-in by joint interpolation of vector fields and gray levels. IEEE TIP, 2001. 2
- [3] Marcelo Bertalmio, Guillermo Sapiro, Vincent Caselles, and Coloma Ballester. Image inpainting. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, 2000.
- [4] Marcelo Bertalmio, Luminita Vese, Guillermo Sapiro, and Stanley Osher. Simultaneous structure and texture image inpainting. IEEE TIP, 2003. 2
- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 2
- [6] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. Endto-end object detection with transformers. In ECCV, 2020. 5
- [7] Floris Chabert. Automated lens flare removal, 2014. Technical Report, Department of Electrical Engineering, Stanford University. 1, 2
- [8] Chen-Hao Chao, Wei-Fang Sun, Bo-Wun Cheng, Yi-Chen Lo, Chia-Che Chang, Yu-Lun Liu, Yu-Lin Chang, Chia-Ping Chen, and Chun-Yi Lee. Denoising likelihood score matching for conditional score-based data generation. arXiv preprint arXiv:2203.14206, 2022. 2
- [9] Ting-Hsuan Chen, Jie Wen Chan, Hau-Shiang Shiu, Shih-Han Yen, Changhan Yeh, and Yu-Lun Liu. Narcan: Natural refined canonical image with integration of diffusion prior for video editing. NeurIPS, 2024. 2
- [10] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Conditioning method for denoising diffusion probabilistic models. arXiv preprint arXiv:2108.02938, 2021. 2
- [11] Yuekun Dai, Chongyi Li, Shangchen Zhou, Ruicheng Feng, and Chen Change Loy. Flare7k: A phenomenological nighttime flare removal dataset. NeurIPS, 2022. 1, 2, 5, 12
- [12] Yuekun Dai, Yihang Luo, Shangchen Zhou, Chongyi Li, and Chen Change Loy. Nighttime smartphone reflective flare removal using optical center symmetry prior. In CVPR, 2023. 2

- [13] Yuekun Dai, Chongyi Li, Shangchen Zhou, Ruicheng Feng, Yihang Luo, and Chen Change Loy. Flare7k++: Mixing synthetic and real datasets for nighttime flare removal and beyond. IEEE TPAMI, 2024. 1, 2, 6, 7, 13
- [14] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618,

2022. 2

- [15] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. NeurIPS, 2014. 2
- [16] Dongsheng Guo, Hongzhi Liu, Haoru Zhao, Yunhao Cheng, Qingwei Song, Zhaorui Gu, Haiyong Zheng, and Bing Zheng. Spiral generative network for image extrapolation. In ECCV,

- 2020. 2

[17] Xiefan Guo, Hongyu Yang, and Di Huang. Image inpainting via conditional texture and structure dual generation. In ICCV,

- 2021. 2

- [18] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In ICCV, 2023. 2
- [19] James Hays and Alexei A Efros. Scene completion using millions of photographs. ACM TOG, 2007. 2
- [20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 2
- [21] Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory. Neural Computation, 1997. 2
- [22] L. L. Holladay. The Fundamentals of Glare and Visibility. Journal of the Optical Society of America (1917-1983), 1926. 2
- [23] Seunghoon Hong, Xinchen Yan, Thomas Huang, and Honglak Lee. Learning hierarchical semantic image manipulation through structured representations. arXiv preprint arXiv:1808.07535, 2018. 2
- [24] Chi-Wei Hsiao, Yu-Lun Liu, Cheng-Kun Yang, Sheng-Po Kuo, Kevin Jou, and Chia-Ping Chen. Ref-ldm: A latent diffusion model for reference-based face image restoration. NeurIPS, 2024. 2
- [25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 2022. 2, 12
- [26] Matthias Hullin, Elmar Eisemann, Hans-Peter Seidel, and Sungkil Lee. Physically-based real-time lens flare rendering. ACM TOG, 2011. 1, 2
- [27] Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa. Globally and locally consistent image completion. ACM TOG, 2017. 2
- [28] Yiguo Jiang, Xuhang Chen, Chi-Man Pun, Shuqiang Wang, and Wei Feng. Mfdnet: Multi-frequency deflare network for efficient nighttime flare removal. The Visual Computer, 2024. 1, 6, 7, 13
- [29] Glenn Jocher and Jing Qiu. Ultralytics yolo11, 2024. 13
- [30] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In CVPR, 2018. 5

- [31] Sungkil Lee and Elmar Eisemann. Practical real-time lensflare rendering. Computer Graphics Forum, 2013. 2
- [32] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023. 4, 12
- [33] Tzu-Mao Li, Michal Luk´aˇc, Micha¨el Gharbi, and Jonathan Ragan-Kelley. Differentiable vector graphics rasterization for editing and learning. ACM TOG, 2020. 8
- [34] Xiaoyu Li, Bo Zhang, Jing Liao, and Pedro V Sander. Let’s see clearly: Contaminant artifact removal for moving cameras. In ICCV, 2021. 1, 2
- [35] Lukas Liebel and Marco K¨orner. Auxiliary tasks in multi-task learning. arXiv preprint arXiv:1805.06334, 2018. 5
- [36] Han Lin, Maurice Pagnucco, and Yang Song. Edge guided progressively generative image outpainting. In CVPR, 2021. 2
- [37] Guilin Liu, Fitsum A Reda, Kevin J Shih, Ting-Chun Wang, Andrew Tao, and Bryan Catanzaro. Image inpainting for irregular holes using partial convolutions. In ECCV, 2018. 2
- [38] Kuan-Hung Liu, Cheng-Kun Yang, Min-Hung Chen, Yu-Lun Liu, and Yen-Yu Lin. Corrfill: Enhancing faithfulness in reference-based inpainting with correspondence guidance in diffusion models. In WACV, 2025. 2
- [39] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In ECCV, 2022. 2
- [40] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In CVPR,

2022. 2, 4

- [41] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 2
- [42] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In AAAI, 2024. 2
- [43] Kamyar Nazeri, Eric Ng, Tony Joseph, Faisal Qureshi, and Mehran Ebrahimi. Edgeconnect: Structure guided image inpainting using edge prediction. In ICCV Workshops, 2019. 2
- [44] Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. arXiv preprint arXiv:2102.09672,

2021. 2

- [45] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2
- [46] Evangelos Ntavelis, Andr´es Romero, Iason Kastanis, Luc Van Gool, and Radu Timofte. Sesame: semantic editing of scenes by adding, manipulating or erasing objects. In ECCV,

2020. 2

- [47] Andreas Nussberger, Helmut Grabner, and Luc Van Gool. Robust aerial object tracking in images with lens flare. In ICRA, 2015. 2

- [48] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In CVPR, 2016. 2
- [49] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 6, 7
- [50] Xiaotian Qiao, Gerhard P. Hancke, and Rynson W. H. Lau. Light source guided single-image flare removal from unpaired data. In ICCV, 2021. 2
- [51] Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, and Bernhard Sch¨olkopf. Controlling text-to-image diffusion by orthogonal finetuning. NeurIPS, 2023. 2
- [52] Erik Reinhard, Greg Ward, Sumanta Pattanaik, and Paul Debevec. High Dynamic Range Imaging: Acquisition, Display, and Image-Based Lighting (The Morgan Kaufmann Series in Computer Graphics). Morgan Kaufmann Publishers Inc.,

2005. 2

- [53] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 6, 7, 12
- [54] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015. 2, 4, 13
- [55] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 2
- [56] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 conference proceedings, 2022. 2
- [57] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 2022. 2
- [58] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 2
- [59] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 2
- [60] Eino-Ville Talvala, Andrew Adams, Mark Horowitz, and Marc Levoy. Veiling glare in high dynamic range imaging. ACM TOG, 2007. 2
- [61] Luming Tang, Nataniel Ruiz, Qinghao Chu, Yuanzhen Li, Aleksander Holynski, David E Jacobs, Bharath Hariharan, Yael Pritch, Neal Wadhwa, Kfir Aberman, et al. Realfill: Reference-driven generation for authentic image completion. ACM TOG, 2024. 4
- [62] Patricia Vitoria and Coloma Ballester. Automatic flare spot artifact detection and removal in photographs. Journal of Mathematical Imaging and Vision, 2019. 1, 2

- [63] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-to-image generation. arXiv preprint arXiv:2303.09522, 2023. 2
- [64] Fu-Yun Wang, Xiaoshi Wu, Zhaoyang Huang, Xiaoyu Shi, Dazhong Shen, Guanglu Song, Yu Liu, and Hongsheng Li. Beyour-outpainter: Mastering video outpainting through inputspecific adaptation. In ECCV, 2024. 4
- [65] Weilun Wang, Jianmin Bao, Wengang Zhou, Dongdong Chen, Dong Chen, Lu Yuan, and Houqiang Li. Sindiffusion: Learning a diffusion model from a single natural image. arXiv preprint arXiv:2211.12445, 2022. 2
- [66] Yi Wang, Xin Tao, Xiaoyong Shen, and Jiaya Jia. Widecontext semantic image extrapolation. In CVPR, 2019. 2
- [67] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 13(4):600–612, 2004. 12
- [68] Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, and Houqiang Li. Uformer: A general u-shaped transformer for image restorationn. In CVPR, 2022. 2
- [69] Zhixiang Wang, Baiang Li, Jian Wang, Yu-Lun Liu, Jinwei Gu, Yung-Yu Chuang, and Shin’Ichi Satoh. Matting by generation. In ACM SIGGRAPH 2024 Conference Papers, 2024. 2
- [70] Chung-Ho Wu, Yang-Jung Chen, Ying-Huan Chen, Jie-Ying Lee, Bo-Hsu Ke, Chun-Wei Tuan Mu, Yi-Chuan Huang, Chin-Yang Lin, Min-Hung Chen, Yen-Yu Lin, et al. Aurafusion360: Augmented unseen region alignment for referencebased 360deg unbounded scene inpainting. In CVPR, 2025. 2
- [71] Yicheng Wu, Qiurui He, Tianfan Xue, Rahul Garg, Jiawen Chen, Ashok Veeraraghavan, and Jonathan T Barron. How to train neural networks for flare removal. In ICCV, 2021. 1, 2
- [72] Wei Xiong, Jiahui Yu, Zhe Lin, Jimei Yang, Xin Lu, Connelly Barnes, and Jiebo Luo. Foreground-aware image inpainting. In CVPR, 2019. 2
- [73] Shunxin Xu, Dong Liu, and Zhiwei Xiong. E2i: Generative inpainting from edge to image. IEEE Transactions on Circuits and Systems for Video Technology, 2020. 2
- [74] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In CVPR, 2023. 2
- [75] Zongxin Yang, Jian Dong, Ping Liu, Yi Yang, and Shuicheng Yan. Very long natural scenery image prediction by outpainting. In ICCV, 2019. 2
- [76] Kai Yao, Penglei Gao, Xi Yang, Jie Sun, Rui Zhang, and Kaizhu Huang. Outpainting by queries. In ECCV, 2022. 2
- [77] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2

- [78] Chang-Han Yeh, Chin-Yang Lin, Zhixiang Wang, ChiWei Hsiao, Ting-Hsuan Chen, Hau-Shiang Shiu, and YuLun Liu. Diffir2vr-zero: Zero-shot video restoration with diffusion-based image restoration models. arXiv preprint arXiv:2407.01519, 2024. 2

- [79] Fisher Yu and Vladlen Koltun. Multi-scale context aggregation by dilated convolutions. arXiv preprint arXiv:1511.07122, 2015. 2
- [80] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Generative image inpainting with contextual attention. In CVPR, 2018. 2
- [81] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Free-form image inpainting with gated convolution. In ICCV, 2019. 2
- [82] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2
- [83] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 1, 12
- [84] Xuaner Zhang, Ren Ng, and Qifeng Chen. Single image reflection separation with perceptual losses. In CVPR, 2018. 5
- [85] Shangchen Zhou, Chongyi Li, and Chen Change Loy. LEDNet: Joint low-light enhancement and deblurring in the dark. In ECCV, 2022. 2
- [86] Tianwen Zhou, Qihao Duan, and Zitong Yu. Difflare: Removing image lens flare with latent diffusion model. arXiv preprint arXiv:2407.14746, 2024. 1, 6
- [87] Yuyan Zhou, Dong Liang, Songcan Chen, Sheng-Jun Huang, Shuo Yang, and Chongyi Li. Improving lens flare removal with general-purpose pipeline and multiple light sources recovery. In ICCV, 2023. 6, 7, 13
- [88] Junyan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In ICCV, 2017. 2
- [89] Zixin Zhu, Xuelu Feng, Dongdong Chen, Jianmin Bao, Le Wang, Yinpeng Chen, Lu Yuan, and Gang Hua. Designing a better asymmetric vqgan for stablediffusion. arXiv preprint arXiv:2306.04632, 2023. 4
- [90] Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In ECCV,

2024. 6, 7

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Direct Input Ours

- Figure 8. Comparison of the downstream tasks. The visual results indicate that LightsOut enhances performance on object detection tasks as well. Our approach not only boosts detection confidence scores but also enables the identification of objects previously undetectable due to flare artifacts.

### A. Appendix Section

#### A.1. Implementation Details

Dataset and Preprocessing. We use the benchmark dataset Flare7k [11] for both training and testing. Since the dataset was not originally designed for our tasks, we preprocess it to better suit our requirements. Specifically, to handle off-frame or incomplete light source images and define outpainted regions, we first generate YCbCr luminance masks and then apply an algorithm, formalized in Algorithm 2, to identify the largest rectangular area in each image that excludes the light source. Once the bounding box is obtained, we crop the image on-the-fly during training and inference. The cropped region is then masked with a pixel value of 127, defining the area to be outpainted.

Training Details. Our framework comprises three independently trained modules, all implemented on an NVIDIA RTX4090 GPU. The components are optimized independently, allowing each module to specialize in a distinct subtask and enabling them to collectively improve the system’s overall performance when integrated. The multitask regression module was trained with a learning rate of 1 × 10−4, batch size of 32, for 100 epochs, and we set the number of predicted light sources N to 4. The light source condition module was optimized using a learning rate of 1×10−5 and a

Algorithm 2 Cropping Algorithm

- 1: function IMAGECROP(image)
- 2: function LARGESTRECTANGLE(heights)
- 3: heights.append(0)
- 4: stack ← [−1]
- 5: max area ← 0

- 6: max bbox ← (0, 0, 0, 0) ▷ (area, left, right, height)

- 7: for i ← 0 to len(heights) − 1 do
- 8: while heights[i] ¡ heights[stack[-1]] do
- 9: h ← heights[stack.pop()]
- 10: w ← i − stack[−1] − 1
- 11: area ← h × w
- 12: if area > max area then

- 13: max area ← area

- 14: max bbox ← (area, stack[−1] + 1, i − 1, h)

- 15: end if
- 16: end while
- 17: stack.append(i)
- 18: end for
- 19: return max bbox

- 20: end function
- 21: max area ← 0

- 22: max bbox ← [0, 0, 0, 0]

- 23: heights ← zeros like(image.shape[1])

- 24: for row ← 0 to image.shape[0] − 1 do
- 25: temp ← 1 − image[row]
- 26: heights ← (heights + temp) × temp
- 27: (area, left, right, height) ← LargestRectangle(heights)
- 28: if area > max area then

- 29: max area ← area

- 30: max bbox ← [left, right, (row−height+1), row]

- 31: end if
- 32: end for
- 33: return max bbox

- 34: end function

batch size of 8 for 20,000 steps. Finally, the Stable Diffusion inpainting network [53] was fine-tuned using LoRA [25] with a learning rate of 1 × 10−4 and a batch size of 8 for 25,000 steps to achieve optimal performance while maintaining computational efficiency.

Inference Settings. During outpainting process, we set the number of sampling steps to 50, the guidance scale to 7.0, and perform noise reinjection 4 times. Additionally, we utilize BLIP-2 [32] to automatically generate captions, thereby minimizing human bias.

Evaluation metrics. We evaluate flare removal quality using PSNR, SSIM [67], and LPIPS [83], and assess the accuracy of our light source prediction using mean Intersection over Union (mIoU).

[Figure 118]

[Figure 119]

[Figure 120]

Zhou, et al Ours + Zhou, et al Flare7K++ Ours + Flare7k++ MFDNet Ours + MFDNet

Figure 9. Failure Cases.

#### A.2. Downstream Tasks

Lens flare artifacts can negatively impact images in various computer vision tasks. To examine how flare removal affects object detection performance, we utilize the pre-trained YOLOv11 [29] detector to compare two scenarios: images directly processed by SIFR models, and images first enhanced by our proposed outpainting approach before being input to SIFR models. Fig. 8 demonstrates that our proposed approach yields improvements in detection accuracy, particularly for objects located in regions previously compromised by flare artifacts.

#### A.3. In-the-Wild Images.

We present additional outpainting results on self-collected in-the-wild scenes in Fig. 11, along with flare removal comparisons against baseline methods (Zhou et al.[87], Flare7K++[13], and MFDNet [28]) in Fig. 10. These results highlight our method’s effectiveness in outpainting off-frame regions and improving the performance of existing SIFR models, even on challenging in-the-wild images.

#### A.4. Failure Cases

The main failure cases exhibit two characteristic features. First, when the overall image brightness is high, the brightness differential between the flare and other parts of the image becomes less pronounced. Second, when the flare occupies a relatively large proportion of the entire image. Both scenarios make it difficult to delineate the flare region precisely, even with the integration of our proposed method.

#### A.5.QualitativeComparisonsofLightSourceMask Prediction.

Fig. 12 compares the light source predictions from our multitask regression module with those generated by U-Net [54]. The results demonstrate that our proposed module predicts the positions and radii of light sources more accurately, both in single and multiple light source scenarios.

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

| |
|---|

| |
|---|

| |
|---|

|v|
|---|

| |
|---|

| |
|---|

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

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

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Direct input Ours Direct input Ours

Figure 10. Flare removal results for in-the-wild scens. The red boxes indicate flare regions in the images. Our method effectively addresses off-frame light source scenes, which existing SIFR models fail to handle.

#### A.6. Additional Qualitative Comparisons

We present extensive supplementary visual evidence to demonstrate the efficacy of our approach. Figures Fig. 13, and Fig. 14 showcase additional flare removal results across diverse imaging conditions. Furthermore, we provide comparative analyses between our outpainting results and those produced by both baseline methods and state-of-the-art diffusion-based inpainting and outpainting techniques in Fig. 15, Fig. 16. These comprehensive visual comparisons substantiate the superior robustness and effectiveness of our proposed methodology across a wide spectrum of challenging scenarios.

[Figure 133]

[Figure 134]

|[Figure 135]|
|---|

|Outpainted image (full light source)<br><br>[Figure 136]|
|---|

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

|[Figure 141]|
|---|

|[Figure 142]|
|---|

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

|[Figure 147]|
|---|

|Outpainted image (full light source)<br><br>[Figure 148]|
|---|

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

|[Figure 153]|
|---|

[Figure 154]

|[Figure 155]|
|---|

[Figure 156]

###### Figure 11. Outpainting results for in-the-wild scens.

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

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Input U-Net Ours Ground truth

###### Figure 12. Qualitative comparisons of light source mask prediction. .

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

IncompletelightsourceNolightsource

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

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

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

###### Ours +

###### Ours +

Ours + MFDNet

Zhou, et al Flare7K++

Flare7k++ MFDNet

Input Zhou, et al Ground truth

###### Figure 13. Additional Qualitative Comparisons. .

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

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

IncompletelightsourceNolightsource

[Figure 253]

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

[Figure 290]

[Figure 291]

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

[Figure 292]

| |
|---|

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

###### Ours +

###### Ours +

Ours + MFDNet

Zhou, et al Flare7K++

Flare7k++ MFDNet

Input Zhou, et al Ground truth

###### Figure 14. Additional Qualitative Comparisons. .

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

SD-Inpainting SDXL-Inpainting PowerPaint Ours

###### Figure 15. Additional Qualitative Comparisons. .

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

SD-Inpainting SDXL-Inpainting PowerPaint Ours

###### Figure 16. Additional Qualitative Comparisons. .

