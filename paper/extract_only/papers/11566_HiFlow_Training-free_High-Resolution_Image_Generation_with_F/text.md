# arXiv:2504.06232v2[cs.CV]16May2025

[Figure 1]

## HiFlow: Training-free High-Resolution Image Generation with Flow-Aligned Guidance

Jiazi Bu1,5∗ Pengyang Ling2,5∗ Yujie Zhou1,5∗ Pan Zhang5† Xiaoyi Dong3,5 Yuhang Zang5 Yuhang Cao5 Tong Wu4 Dahua Lin3,5,7 Jiaqi Wang5,6† 1Shanghai Jiao Tong University 2University of Science and Technology of China 3The Chinese University of Hong Kong 4Stanford University 5Shanghai AI Laboratory 6Shanghai Innovation Institute 7CPII under InnoHK https://bujiazi.github.io/hiflow.github.io/

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

4096 × 4096 4096 × 4096

2048 × 4096

[Figure 6]

[Figure 7]

[Figure 8]

2048 × 4096

| |
|---|

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

4096 × 2048 4096 × 4096 4096 × 4096

2048 × 2048

[Figure 13]

2048 × 2048

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

3072 × 3072 3072 × 4096

3072 × 2048 3072 × 3072

Figure 1: Gallery of HiFlow. The proposed HiFlow enables pre-trained text-to-image flow models (Flux.1.0-dev integrated with various LoRA models) to synthesize high-resolution images with high fidelity and rich details in a training-free manner. All prompts are listed in the appendix.

∗Equal contribution. †Corresponding author.

Preprint. Under review.

### Abstract

Text-to-image (T2I) diffusion/flow models have drawn considerable attention recently due to their remarkable ability to deliver flexible visual creations. Still, high-resolution image synthesis presents formidable challenges due to the scarcity and complexity of high-resolution content. Recent approaches have investigated training-free strategies to enable high-resolution image synthesis with pre-trained models. However, these techniques often struggle with generating high-quality visuals and tend to exhibit artifacts or low-fidelity details, as they typically rely solely on the endpoint of the low-resolution sampling trajectory while neglecting intermediate states that are critical for preserving structure and synthesizing finer detail. To this end, we present HiFlow, a training-free and model-agnostic framework to unlock the resolution potential of pre-trained flow models. Specifically, HiFlow establishes a virtual reference flow within the high-resolution space that effectively captures the characteristics of low-resolution flow information, offering guidance for high-resolution generation through three key aspects: initialization alignment for low-frequency consistency, direction alignment for structure preservation, and acceleration alignment for detail fidelity. By leveraging such flow-aligned guidance, HiFlow substantially elevates the quality of high-resolution image synthesis of T2I models and demonstrates versatility across their personalized variants. Extensive experiments validate HiFlow’s capability in achieving superior high-resolution image quality over state-of-the-art methods.

### 1 Introduction

The text-to-image (T2I) diffusion/flow models [41, 37, 36, 2, 8, 34, 36, 42, 28, 46, 26], which allow for flexible content creation from textual prompts, have recently achieved a landmark advancement. Despite considerable improvements, existing T2I models are typically confined to a restricted resolution (e.g., 1024 × 1024) and experience notable quality decline and even structural breakdown when attempting to generate higher-resolution images, as illustrated in Fig. 2. Such shortcoming limits the utility and diminishes their appeal to contemporary artistic and commercial applications, in which detail and precision are paramount. As initial efforts, several methods [14, 19, 32, 40, 47, 57, 51, 50] suggest fine-tuning T2I models on higher-resolution samples to enhance the adaptability to large-scale images. Nevertheless, this straightforward approach entails significant costs, primarily the burden of high-resolution image collection and the necessity for model-specific fine-tuning. Therefore, recent studies have investigated training-free strategies [15, 10, 21, 1, 23, 54, 27, 24, 38, 11, 45] to harness the inherent potential of pre-trained T2I models in high-resolution image synthesis. However, the majority of these methods [15, 21, 54, 38, 23] involve manipulating the internal features within models, exhibiting restricted transferability across architectures, such as applying methods tailored for U-Net architecture in DiT-based models. Another line of research [10, 24, 45, 11] suggests fusing the upsampled low-resolution images into the denoising target during high-resolution synthesis for structure guidance. Despite simplicity, these methods merely align high-resolution images predicted at different time steps with the upsampled low-resolution image from the final step as single guidance anchor, risking the introduction of artifacts due to distribution discrepancy, as shown in Fig. 3 (b). Moreover, they primarily emphasize structural guidance, leaving the potential of detail-oriented cues underexplored, resulting in suboptimal detail fidelity in the outputs, as illustrated in Fig. 3 (d).

In this work, we introduce a novel training-free and model-agnostic generation framework, termed HiFlow, which is designed to advance high-resolution T2I synthesis of Rectified Flow models and can be seamlessly extended to diffusion models by modifying the denoising scheduler. Specifically, HiFlow involves a cascade generation paradigm: First, a virtual reference flow is constructed in the high-resolution space based on the step-wise estimated clean samples of the low-resolution sampling flow. Then, during high-resolution synthesizing, the reference flow offers guidance in sampling initialization, denoising direction, and moving acceleration, aiding in achieving consistent lowfrequency patterns, preserving structural features, and maintaining high-fidelity details, respectively. Such flow-aligned guidance from the sampling trajectory facilitates better merging of the structure synthesized at the low-resolution scale and the details synthesized at the high-resolution scale, facilitating superior visual quality. Furthermore, HiFlow exhibits broad generalizability across U-Net and DiT architectures, owing to its independence from internal model characteristics. Extensive

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

1024 × 1024 4096 × 4096 1024 × 1024 3072 × 3072 1024 × 1024 4096 × 4096

FLUX (DiT/Flow) SD3 Medium (DiT/Flow) SDXL (U-Net/Diffusion)

#### Figure 2: T2I models suffer significant quality degradation in high-resolution image generation.

experiments demonstrate that the proposed method surpasses state-of-the-art baselines on the latest Rectified Flow T2I model and even achieves better performance than leading training-based methods like UltraPixel [40] and Diffusion-4k [51]. The main contributions of this paper are summarized as follows: (i) We propose HiFlow, a novel training-free and model-agnostic framework to unlock the resolution potential of pre-trained Rectified Flow models, which constructs virtual reference flow derived from low-resolution sampling trajectories to enable high-resolution synthesis; (ii) The constructed virtual reference flow provides flow-aligned guidance in terms of initialization, direction, and acceleration, thus promoting low-frequency consistency, structural preservation, and detail fidelity, respectively; and (iii) Comprehensive experiments validate HiFlow’s superiority over stateof-the-art competitors and highlight its flexibility and versatility in various applications including LoRA, ControlNet, and Quantization.

### 2 Related work

#### 2.1 Text-to-image generation.

Text-to-image (T2I) synthesis [41, 37, 36, 2, 8, 34, 42, 28, 46, 26] has witnessed significant advancements with the advent of diffusion models, which have demonstrated remarkable capabilities in producing high-quality images based on given textual prompts. Denoising Diffusion Probabilistic Models (DDPM) [17] and Guided Diffusion [7] showcased the potential of diffusion processes to generate high-fidelity images. Subsequently, the introduction of latent space diffusion [41] marked a revolutionary advancement in the field, which significantly reduced computational demands and enabled more efficient training, giving rise to pioneering models such as Stable Diffusion and Stable Diffusion XL [37]. Recently, the integration of transformer-based architectures [3, 2, 12, 13, 58, 26] into diffusion models has also led to improvements in both image quality and computational efficiency. In this work, we primarily construct our method upon Flux.1.0-dev [26], an advanced Rectified Flow T2I backbone renowned for its superior generation quality.

#### 2.2 High-resolution image generation.

High-resolution image generation with diffusion models has gained increasing popularity in recent years. Several studies [14, 19, 32, 40, 47, 57, 51, 50] have proposed training or fine-tuning existing T2I diffusion models to enhance their capability for high-resolution image generation. However, it remains a challenging task due to the scarcity of high-resolution training data and the substantial computational resources required for modeling such data. Another line of researchs [52, 9, 30, 48, 4] employs super-resolution techniques to upscale the resolution of generated low-resolution images. Nevertheless, the quality of generated images via this approach heavily depends on the initial quality of the low-resolution images and the performance of the super-resolution model. Recent efforts have focused on training-free strategies [15, 10, 21, 1, 23, 54, 27, 24, 38, 11, 45, 49] that modify the inference strategies of diffusion models for low-resolution generation. For instance, HiDiffusion [54] suggests reshaping features in the outermost blocks of the U-Net architecture to match the training size in deeper blocks. DiffuseHigh [24] upscales the low-resolution generation result and re-denoises it under the structural guidance from the Discrete Wavelet Transform. I-Max [11] projects the highresolution flow into the low-resolution space, enabling training-free high-resolution image generation with flow models. Many existing training-free methods [15, 21, 54, 38, 23] are inherently entangled with model-specific internal features, limiting their generalizability across different architectures. Others [10, 24, 45, 11] lack effective guidance throughout the high-resolution generation process,

Single Anchor (Previous Works) Time-dependent Anchor (HiFlow) Time-dependent Anchor (HiFlow)

FID between 𝑋 ←  and 𝑋

Single Anchor (Previous Works)

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

(a) FID score (b) Comparison of different direction guidance strategies

w/o Acc Alignment (Previous Works) w Acc Alignment (HiFlow) w Acc Alignment (HiFlow)

w/o Acc Alignment (Previous Works)

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

t=0.1result

t=0.3t=0.6

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

(c) Acceleration visualization (d) Effects of acceleration alignment in detail synthesis

- Figure 3: Observations. (a) Distribution discrepancy between predicted clean sample X0←t and clean sample X0. (b) Comparison with constant and time-dependent direction guidance. The former exhibits artifacts, the latter demonstrates better structure preservation. (c) Visualization of acceleration. (d) Effect of acceleration alignment, validating its role in facilitating high-fidelity details generation.

including exhibiting distributional discrepancy in the selection of structure guidance anchors and the neglect of detail synthesis guidance, resulting in artifact emergence and decreased detail fidelity.

### 3 Method

#### 3.1 Preliminaries

Flow Matching [31] and Rectified Flow [33] aim at streamlining the formulation of Ordinary Differential Equation (ODE) models by establishing a linear transition between two distinct distributions. Consider clean image samples X0 ∼ π0 and Gaussian noise X1 ∼ π1. Rectified Flow delineates a linear trajectory from X1 to X0, with the intermediate state Xt defined as:

Xt = tX1 + (1 − t)X0, (1)

in which t ∈ [0,1] denotes a continuous time interval. By taking the derivative of t on both sides of Eq. 1, its linear progression gives the following equation:

dXt = (X1 − X0)dt. (2)

During the denoising phase, given Xt and time t, a neural network vθ is introduced to estimate the vector of flow, i.e., X1 − X0), which can be expressed as:

vθ(Xt,t,c) = X1←t − X0←t, (3)

in which c represents optional control signs such as textual or image prompts, and X1←t and X0←t denote the predicted noisy and clear component within Xt, respectively. Since Xt and t are known, from Eq. 1, vθ(Xt,t,c) is essentially determined by the predicted clean component X0←t, i.e.,

Xt − X0←t t

. (4) The progressive denoising of flow models can be formulated as follows:

vθ(Xt,t,c) =

,ti,c)(ti−1 − ti), (5) in which the movement of ti from 1 to 0 indicates the trajectory of Xt

Xt

= Xt

+ vθ(Xt

i−1

i

i

from gaussian noise X1 to clean sample X0. During denoising, the predicted vθ(Xt

i

,ti,c) determines the denoising direction at each time ti. In the following sections, vt is used to denote vθ(Xt,t,c) for simplicity.

i

#### 3.2 Virtual reference flow

Rectified flow models have shown great promise in advancing high-quality image generations. Yet, these models experience a critical quality drop when attempting high-resolution synthesizing, as illustrated in Fig. 2. To this end, previous works [10, 24, 45, 11] typically leverage the synthesized

low

low

low

low

low

low

 −1

0

0← −1

0← 

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

...

...

Low

Acceleration

Inter Inter

[Figure 60]

ref

ref

ref

ref

ref

 −1

0

0← 

0← −1

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

low

0← 

...

[Figure 66]

###### Ref

= 0.6

Add noise

[Figure 67]

ref

LPF

LPF

 ( 0← low)

 −1← 

LPF : Low pass filter

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Inter : Interpolation

Inter

: Next step

= 0.3

: Predict clean sample

Add noise

Initialization Detail Structure

Structure

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

... High

= 0.1

...

High

High

High

High

High

 −1

0← 

0← −1

0

Initialization Alignment Direction Alignment Acceleration Alignment

- Figure 4: Pipeline of HiFlow. HiFlow constructs reference flow from low-resolution sampling trajectory to offer guidance for high-resolution generation in initialization, direction, and acceleration.

low-resolution sample X0low for guidance by fusing its upsampled variant into predicted clean sample X0high←t to modify the denoising direction. However, for given flow models, it is observed that there are ignored distribution discrepancies between X0←t at different t and X0, and these discrepancies increase significantly as t approaches 0, as depicted in Fig. 3 (a). Therefore, the fusion between upsampled X0low (single guidance anchor) and X0high←t may cause artifacts and thus lead to sub-optimal results, as shown in Fig. 3 (b). To this end, in this work, a virtual reference flow is constructed in the high-resolution space that can fully characterize the information of the low-resolution sampling trajectory. Specifically, for reference flow, its predicted clean image X0ref←t at time t is defined as:

X0ref←t = ϕ(X0low←t) , t ∈ [0,1] (6)

in which ϕ(X0low←t) is the upsampled X0low←t by interpolation function ϕ(·). Meanwhile, the noisy image Xtref of the reference flow at time t is defined as a virtual noisy image in high-resolution space. For each time t, the vector of reference flow is (Xtref − X0ref←t)/t = (Xtref − ϕ(X0low←t))/t. Indeed, the reference flow employs virtual Xtref to construct an imaginary sampling trajectory in high-resolution space that can produce the upsampled low-resolution image ϕ(X0low). Such a virtual reference flow acts as a bridge connecting low-resolution and high-resolution sampling trajectories, facilitating guided high-resolution synthesis with enhanced structure and fidelity.

#### 3.3 Flow-aligned guidance

Given that T2I diffusion/flow models prioritize low-frequency structure components before synthesizing high-frequency details [46, 25], we follow the cascade generation pipeline [10, 24, 38], i.e., first synthesize a low-resolution image and mapping its sampling trajectory to high-resolution space to obtain reference flow. This reference flow is then used for flow-guided high-resolution generation, which is analyzed from the following three aspects. The pipeline of HiFlow is illustrated in Fig. 4.

Initialization alignment. For a given virtual reference flow, the sampling of high-resolution generation starts from the noisy variant of X0ref←τ, which can be expressed as:

Xτhigh = τX1high + (1 − τ)X0ref←τ, (7)

in which Xτhigh is the sampling initialization of high-resolution generation, X1high is gaussian noise, and τ ∈ (0,1) is noise addition ratio. Such initialization alignment allows skipping the early stage

in high-resolution generation, thereby maintaining the consistency of high-resolution results and low-resolution images in low-frequency components, while also facilitating higher inference speeds.

Direction alignment. While initialization alignment ensures low-frequency consistency at the beginning of high-resolution generation, such structural information may be destroyed at subsequent denoising steps due to the limited stability of models in synthesizing high-resolution content (as shown in Fig. 2), risking broken structures. Therefore, direction alignment is designed for structural preservation, which is achieved by modifying the denoising direction based on reference flow, i.e.,

Xˆ0high←t = X0high←t + αt[ F(F(X0ref←t) ⊙ L(D)) − F(F(X0high←t) ⊙ L(D))], (8) in which F and F denote 2D Fast Fourier Transform and its inverse operation, respectively, L(·) denotes a butterworth low-pass filter and D is the normalized cutoff frequency, and αt is the direction guidance weight. Essentially, Eq. 8 replaces the low-frequency component of X0high←t with that in X0ref←t, thus rejecting the updating on low-frequency structures when synthesizing high-frequency details. Unlike previous work [24, 10, 45] that aligns with a single guidance anchor ϕ(X0low), Eq. 8 manipulates X0high←t by using time-dependent X0ref←t, such fusion between samples from the same t helps avoid artifacts caused by distribution discrepancy (see Fig. 3 (a)), facilitating better quality.

Acceleration alignment. Although the above strategies allow for rich detail generation while maintaining structure, the fidelity of synthesized details risks dropping in some cases, in which unrealistic contents appear, such as repetitive patterns and abnormal textures, as shown in Fig. 3(d). This issue primarily arises from the constrained capability of generation models in synthesizing high-resolution content. To this end, acceleration alignment is proposed to enhance detail fidelity by aligning the synthesized content of the high-resolution flow with the reference flow at each time t. Specifically, the acceleration, which is defined as the second-order derivative of movement Xt, and also denotes the first-order derivation of vector vt, can be expressed as:

d2Xt dt2

i−1 − vt

vt

dvt dt

. (9) Furthermore, based on Eq. 4 and Eq. 5, the acceleration in Eq. 9 can be simplified as follows:

i

at

=

=

i−1←ti =

ti−1 − ti

##### ) − t1

1

i−1 − X0←t

i − X0←t

ti−1 (Xt

) ti−1 − ti

(Xt

i−1

i

i

at

i−1←ti =

+ (ti−1 − ti)(Xt

i − X0←t

) − tiX0←t

i−1 − ti−1(Xt

i − X0←t

tiXt

) ti−1ti(ti−1 − ti)

(10)

i

i

i

=

##### i−1 − X0←t

X0←t

1 ti−1

i

= −

.

ti−1 − ti

It can be concluded from Eq. 10 that the acceleration depicts the variation in the predicted clean sample X0←t between adjacent time t, with time-dependent term 1/t. As shown in Fig. 3 (c), the acceleration primarily captures texture and contour information while also indicating the sequence of content synthesis at each t, i.e., it showcases what content the model is responsible for adding at different t. Therefore, we propose aligning the acceleration of high-resolution generation with that of the reference flow to synchronize the model’s preference for content synthesis order, enabling guided detail synthesis both in content and timing. Mathematically, acceleration alignment is modeled as:

aˆhight

i−1←ti = ahight

i−1←ti − ahight

i−1←ti + βt(areft

i−1←ti), (11) in which βt is the acceleration guidance weight. Substituting Eq. 9 into Eq. 11, we have

vˆthigh

− vthigh

vthigh

− vthigh

vthigh

− vthigh

vtref

##### − vtref

). (12) Furthermore, Eq. 12 can be simplified into the following form:

i−1

i−1

i−1

i−1

i

i

i

i

ti−1 − ti −

=

+ βt(

ti−1 − ti

ti−1 − ti

ti−1 − ti

− vthigh

##### + vthigh

vˆthigh

##### = vthigh

##### − vtref

##### + βt(vtref

). (13)

i−1

i−1

i−1

i−1

i

i

Subsequently, the obtained vˆthigh based on Eq. 13 is plugged into the vθ in Eq. 5 to facilitate acceleration alignment. The detailed discussion and analysis are provided in the appendix.

### 4 Experiments

#### 4.1 Implementation details

Experimental settings. If not specified, the generated images are based on Flux.1.0-dev [26], an advanced open-sourced Rectified Flow model based on DiT architecture. The sampling steps are set as 30, the noise-adding ratio τ in initialization alignment is set as [0.6, 0.3, 0.3] for 1K → 2K → 3K → 4K cascade generation, and the normalized cutoff frequency is set as D = 0.4. The guidance weights in direction/acceleration alignment are set as αt = βt = t/τ for gradually weakening control. All experiments are conducted on a single NVIDIA A100 GPU.

Baselines. The compared methods encompass training-free approaches (DemoFusion [10], DiffuseHigh [24], I-Max [11]), training-based approaches (UltraPixel [40], Diffusion-4k [51]), and an image super-resolution method, BSRGAN [52]. For a fair comparison, the training-free methods are tested on the same Flux model according to their official implementations.

Evaluation. We collect 1K high-quality captions across various scenarios for diverse image generation. The CLIP [39] score is used to assess the prompt-following capability, Frechet Inception Distance [16] (FID) and Inception Score [43] (IS) are reported to measure image quality, in which FID is calculated between generated images and 10K real high-quality images (with at least 1024 × 1024 resolution) sourced from LAION-High-Resolution [44]. Furthermore, the patch-version FIDpatch and ISpatch are calculated based on local image patches to quantify the quality of synthesized details.

Table 1: Quantitative comparison with other baselines. The best result is highlighted in bold, while the second-best result is underlined. * indicates methods adapted from U-Net architecture.

Resolution (height × width) Method FID ↓ FIDpatch ↓ IS ↑ ISpatch ↑ CLIP Score ↑

DemoFusion* [10] 56.07 51.69 27.23 13.48 35.05 DiffuseHigh* [24] 61.62 50.25 26.76 13.10 34.83

I-Max [11] 57.57 54.56 28.84 12.07 34.96 Flux + BSRGAN [52] 60.25 52.06 25.85 13.39 35.34

2048 × 2048 (2K)

UltraPixel [40] 59.67 49.02 28.49 13.74 35.16 Diffusion-4k [51] 60.96 54.67 27.15 12.31 34.76

HiFlow (Ours) 55.39 47.70 28.67 13.86 35.32

DemoFusion* [10] 56.72 49.48 21.17 8.49 35.27 DiffuseHigh* [24] 62.01 50.98 20.60 8.09 34.98

I-Max [11] 53.27 52.93 22.21 7.65 35.05 Flux + BSRGAN [52] 59.53 54.12 19.32 8.87 35.37

4096 × 4096 (4K)

UltraPixel [40] 60.90 47.19 24.73 9.47 35.01 Diffusion-4k [51] 77.95 70.07 15.79 6.42 32.67

HiFlow (Ours) 52.55 45.01 24.62 9.73 35.40

#### 4.2 Comparison to state-of-the-art methods

Qualitative comparison. As shown in Fig. 5, while DemoFusion and DiffuseHigh are capable of preserving the overall image structure, they often synthesize low-fidelity details and suffer from reduced contrast, primarily due to insufficient guidance in high-resolution detail synthesis. I-Max sometimes produces blurred results because it uses upsampled low-resolution images as the ideal projected flow endpoints, leading to overly limited detail creation. Although BSRGAN, an image super-resolution method, enhances image clarity to some extent, it struggles to synthesize finer details. In comparison, the proposed HiFlow consistently yields aesthetically pleasing and semantically coherent outcomes. Furthermore, as depicted in Fig. 6, HiFlow delivers high-resolution images of exceptional quality, outperforming leading training-based models such as UltraPixel and Diffusion-4k, underscoring its outstanding generative capabilities. More results are provided in the appendix.

Quantitative comparison. The quantitative results are presented in Tab. 1. It is observed that the proposed HiFlow achieves competitive performance in terms of both image quality (FID, FIDpatch, IS and ISpatch) and image-text alignment (CLIP score) under different resolutions, validating its stable and superior performance in high-resolution image generation. Moreover, as presented in Tab. 2, HiFlow surpasses all its training-free competitors in inference speed, offering higher efficiency.

Flux 1024 × 1024

DemoFusion DiffuseHigh I-Max Flux + BSRGAN HiFlow (Ours)

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

2048 × 2048 2048 × 2048 2048 × 2048 2048 × 2048 2048 × 2048

[Figure 83]

1024 × 1024

A cute and adorable fluffy puppy wearing a witch hat in a halloween autumn evening forest, falling autumn leaves, ...

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

4Resolution×

20482048×

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

2048 × 2048 2048 × 2048 2048 × 2048 2048 × 2048 2048 × 2048

[Figure 99]

1024 × 1024

A majestic, metallic giraffe walking through a futuristic city park, its long neck towering above the plants and creatures.

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

4096 × 4096 4096 × 4096 4096 × 4096 4096 × 4096 4096 × 4096

[Figure 115]

1024 × 1024

A city at night, illuminated by neon lights, with cars zooming between towering skyscrapers, and a giant holographic owl in the sky.

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

|[Figure 125]|
|---|

16Resolution×

40964096×

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

4096 × 4096 4096 × 4096 4096 × 4096 4096 × 4096 4096 × 4096

[Figure 131]

1024 × 1024

A mystical elf with emerald green eyes and a crown of leaves, standing in a forest glade bathed in soft moonlight.

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

|[Figure 137]|
|---|

|[Figure 138]|
|---|

|[Figure 139]|
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

- Figure 5: Visual comparison of synthesized 2K and 4K images. HiFlow yields high-resolution images characterized by high-fidelity details and coherent structure. Best viewed zoomed in.

#### 4.3 Ablation study

HiFlow performs guided high-resolution generation by aligning its flow with the reference flow in three aspects: initialization (Ai), direction (Ad), and acceleration (Aa). As can be observed in Fig. 7, initialization alignment helps avoid semantic incorrectness by facilitating low-frequency consistency with low-resolution images. Furthermore, direction alignment enhances structure preservation in the generation process by suppressing the updating in the low-frequency component. Moreover, acceleration alignment contributes to high-fidelity detail synthesizing, eliminating the production of repetitive patterns in the garden, as shown in the realistic woman’s face and the appearance of her clothing. Quantitative results of the ablation study are shown in Tab. 3.

#### 4.4 Applications

Application on customization T2I models. Customization T2I models, which allow tailored generation to match users’ specific requirements, have recently garnered increasing attention. As illustrated in Fig. 8 (a)-(b), the proposed HiFlow is compatible with various customization modules (LoRA [20] and ControlNet [53]), thus advancing personalized high-resolution image generation.

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

UltraPixel 4096 × 4096

UltraPixel 4096 × 4096 Diffusion-4k 4096 × 4096

Diffusion-4k 4096 × 4096

Diffusion-4k 4096 × 4096

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

HiFlow 4096 × 4096

HiFlow 4096 × 4096 HiFlow 4096 × 4096

HiFlow 4096 × 4096

HiFlow 4096 × 4096

#### Figure 6: Qualitative comparison between HiFlow and training-based methods.

−𝐴 ,−𝐴 ,−𝐴

−𝐴 ,−𝐴

−𝐴

###### HiFlow (Ours)

|[Figure 152]|
|---|

[Figure 153]

|[Figure 154]|
|---|

|[Figure 155]|
|---|

|[Figure 156]|
|---|

[Figure 157]

[Figure 158]

[Figure 159]

2048 × 2048

2048 × 2048

2048 × 2048

2048 × 2048

|[Figure 160]|
|---|

|[Figure 161]|
|---|

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

|[Figure 167]|
|---|

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

4096 × 4096 4096 × 4096 4096 × 4096 4096 × 4096

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

#### Figure 7: Ablation experiments in 2K and 4K resolution. Best viewed zoomed in.

Application on quantized T2I models. Given the substantial rise in computational complexity introduced by advanced T2I models, model quantization techniques [6, 29, 55, 56] have been widely explored to decrease computing resource requirements. As shown in Fig. 8 (c), the training-free and model-agnostic attributes of HiFlow allow it to be directly employed with the 4-bit version of Flux (quantized by SVDQuant [29]), thereby substantially accelerating high-resolution image generation.

Application on U-Net based T2I models. U-Net-based T2I models, such as SD1.5 and SDXL [37], constitute pivotal parts of T2I diffusion models and have been extensively developed by the community. As depicted in Fig. 8 (d), the integration of HiFlow and SDXL [37] enables the synthesis of realistic high-resolution images, demonstrating its broad applications.

#### Table 3: Quantitative results of ablation study.

#### Table 2: Comparison in latency.

Resolution Method FID ↓ FIDpatch ↓ IS ↑ ISpatch ↑ CLIP ↑

Resolution Method Latency (sec.) ↓

−Aa,−Ad,−Ai 67.87 71.79 23.47 9.16 33.81 −Aa,−Ad 58.26 54.22 27.46 12.58 34.47 −Aa 58.40 50.38 28.92 13.14 34.90 HiFlow (Ours) 55.39 47.70 28.67 13.86 35.32

DemoFusion* [10] 106 DiffuseHigh* [24] 59

20482

20482

I-Max [11] 94 HiFlow (Ours) 56

DemoFusion* [10] 972 DiffuseHigh* [24] 533

−Aa,−Ad,−Ai 234.38 198.21 9.33 3.31 11.42 −Aa,−Ad 56.20 48.30 19.85 6.51 33.64 −Aa 55.36 51.79 22.78 8.77 35.44 HiFlow (Ours) 52.55 45.01 24.62 9.73 35.40

40962

40962

I-Max [11] 735 HiFlow (Ours) 379

### 5 Conclusion

We present HiFlow, a tuning-free and model-agnostic framework enabling pre-trained Rectified Flow models to generate high-resolution images with high fidelity and rich details. HiFlow involves a

(a) HiFlow + LoRA (b) HiFlow + ControlNet

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

Genshin Impact Furina 4096 × 4096 Black Myth Wukong 4096 × 4096 Depth 4096 × 4096

Canny 4096 × 4096

(c) HiFlow + T2I Quantization (SVDQuant) (d) HiFlow + U-Net-based T2I Models (SDXL)

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

2048 × 1024 2048 × 1024 2048 × 2048 4096 × 4096 2048 × 4096

[Figure 188]

[Figure 189]

2048 × 2048 2048 × 2048

#### Figure 8: Versatile applications of HiFlow.

novel cascade generation paradigm: (i) a virtual reference flow is constructed based on the step-wise predicted clean samples of the low-resolution sampling trajectory; (ii) the high-resolution flow is aligned with the reference flow via flow-aligned guidance, which encompasses three aspects: initialization alignment for low-frequency consistency, direction alignment for structure preservation, and acceleration alignment for detail fidelity. Extensive experiments demonstrate that HiFlow surpasses state-of-the-art methods and highlight its broad applicability across different model architectures.

### References

- [1] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. 2023.
- [2] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer, 2024.
- [3] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [4] Tao Dai, Jianrui Cai, Yongbing Zhang, Shu-Tao Xia, and Lei Zhang. Second-order attention network for single image super-resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11065–11074, 2019.
- [5] Yingying Deng, Xiangyu He, Changwang Mei, Peisong Wang, and Fan Tang. Fireflow: Fast inversion of rectified flow for image semantic editing. arXiv preprint arXiv:2412.07517, 2024.
- [6] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. Advances in neural information processing systems, 36:10088–10115, 2023.
- [7] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [8] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. Advances in neural information processing systems, 34:19822–19835, 2021.
- [9] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence, 38(2):295–307, 2015.
- [10] Ruoyi Du, Dongliang Chang, Timothy Hospedales, Yi-Zhe Song, and Zhanyu Ma. Demofusion: Democratising high-resolution image generation with no $$$. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6159–6168, 2024.
- [11] Ruoyi Du, Dongyang Liu, Le Zhuo, Qin Qi, Hongsheng Li, Zhanyu Ma, and Peng Gao. I-max: Maximize the resolution potential of pre-trained rectified flow transformers with projected flow. arXiv preprint arXiv:2410.07536, 2024.
- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.
- [13] Peng Gao, Le Zhuo, Ziyi Lin, Chris Liu, Junsong Chen, Ruoyi Du, Enze Xie, Xu Luo, Longtian Qiu, Yuhang Zhang, et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024.
- [14] Lanqing Guo, Yingqing He, Haoxin Chen, Menghan Xia, Xiaodong Cun, Yufei Wang, Siyu Huang, Yong Zhang, Xintao Wang, Qifeng Chen, et al. Make a cheap scaling: A self-cascade diffusion model for higher-resolution adaptation. In European Conference on Computer Vision, pages 39–55. Springer, 2024.
- [15] Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. Scalecrafter: Tuning-free higher-resolution visual generation with diffusion models. In The Twelfth International Conference on Learning Representations, 2023.
- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [18] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [19] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232. PMLR, 2023.
- [20] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.
- [21] Linjiang Huang, Rongyao Fang, Aiping Zhang, Guanglu Song, Si Liu, Yu Liu, and Hongsheng Li. Fouriscale: A frequency perspective on training-free high-resolution image synthesis. In European Conference on Computer Vision, pages 196–212. Springer, 2024.
- [22] Zhiyu Jin, Xuli Shen, Bin Li, and Xiangyang Xue. Training-free diffusion model adaptation for variablesized text-to-image synthesis. Advances in Neural Information Processing Systems, 36:70847–70860, 2023.
- [23] Zhiyu Jin, Xuli Shen, Bin Li, and Xiangyang Xue. Training-free diffusion model adaptation for variablesized text-to-image synthesis. Advances in Neural Information Processing Systems, 36, 2024.
- [24] Younghyun Kim, Geunmin Hwang, Junyu Zhang, and Eunbyung Park. Diffusehigh: Training-free progressive high-resolution image synthesis through structure guidance. arXiv preprint arXiv:2406.18459, 2024.
- [25] Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli. Flowedit: Inversionfree text-based editing using pre-trained flow models. arXiv preprint arXiv:2412.08629, 2024.
- [26] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

- [27] Yuseung Lee, Kunho Kim, Hyunjin Kim, and Minhyuk Sung. Syncdiffusion: Coherent montage via synchronized joint diffusions. Advances in Neural Information Processing Systems, 36:50648–50660, 2023.
- [28] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024.
- [29] Muyang Li*, Yujun Lin*, Zhekai Zhang*, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdquant: Absorbing outliers by low-rank components for 4-bit diffusion models. In The Thirteenth International Conference on Learning Representations, 2025.
- [30] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1833–1844, 2021.
- [31] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [32] Songhua Liu, Weihao Yu, Zhenxiong Tan, and Xinchao Wang. Linfusion: 1 gpu, 1 minute, 16k image. arXiv preprint arXiv:2409.02097, 2024.
- [33] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [34] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [35] Bowen Peng and Jeffrey Quesnelle. Ntk-aware scaled rope allows llama models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation, 2023.
- [36] Pablo Pernias, Dominic Rampas, Mats L Richter, Christopher J Pal, and Marc Aubreville. Würstchen: An efficient architecture for large-scale text-to-image diffusion models. arXiv preprint arXiv:2306.00637, 2023.
- [37] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [38] Haonan Qiu, Shiwei Zhang, Yujie Wei, Ruihang Chu, Hangjie Yuan, Xiang Wang, Yingya Zhang, and Ziwei Liu. Freescale: Unleashing the resolution of diffusion models via tuning-free scale fusion. arXiv preprint arXiv:2412.09626, 2024.
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [40] Jingjing Ren, Wenbo Li, Haoyu Chen, Renjing Pei, Bin Shao, Yong Guo, Long Peng, Fenglong Song, and Lei Zhu. Ultrapixel: Advancing ultra-high-resolution image synthesis to new peaks. arXiv preprint arXiv:2407.02158, 2024.
- [41] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [42] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.
- [43] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.
- [44] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.
- [45] Shuwei Shi, Wenbo Li, Yuechen Zhang, Jingwen He, Biao Gong, and Yinqiang Zheng. Resmaster: Mastering high-resolution image generation via structural and fine-grained guidance. arXiv preprint arXiv:2406.16476, 2024.
- [46] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. In CVPR, 2024.
- [47] Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, and Jie Tang. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350, 2023.
- [48] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind superresolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1905–1914, 2021.
- [49] Haoning Wu, Shaocheng Shen, Qiang Hu, Xiaoyun Zhang, Ya Zhang, and Yanfeng Wang. Megafusion: Extend diffusion models towards higher-resolution image generation without further tuning. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 3944–3953. IEEE, 2025.
- [50] Ruonan Yu, Songhua Liu, Zhenxiong Tan, and Xinchao Wang. Ultra-resolution adaptation with ease. 2025.

- [51] Jinjin Zhang, Qiuyu Huang, Junjie Liu, Xiefan Guo, and Di Huang. Diffusion-4k: Ultra-high-resolution image synthesis with latent diffusion models. arXiv preprint arXiv:2503.18352, 2025.
- [52] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4791–4800, 2021.
- [53] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.
- [54] Shen Zhang, Zhaowei Chen, Zhenyu Zhao, Zhenyuan Chen, Yao Tang, Yuhao Chen, Wengang Cao, and Jiajun Liang. Hidiffusion: Unlocking high-resolution creativity and efficiency in low-resolution trained diffusion models. arXiv preprint arXiv:2311.17528, 2023.
- [55] Tianchen Zhao, Tongcheng Fang, Haofeng Huang, Enshu Liu, Rui Wan, Widyadewi Soedarmadji, Shiyao Li, Zinan Lin, Guohao Dai, Shengen Yan, et al. Vidit-q: Efficient and accurate quantization of diffusion transformers for image and video generation. arXiv preprint arXiv:2406.02540, 2024.
- [56] Tianchen Zhao, Xuefei Ning, Tongcheng Fang, Enshu Liu, Guyue Huang, Zinan Lin, Shengen Yan, Guohao Dai, and Yu Wang. Mixdq: Memory-efficient few-step text-to-image diffusion models with metric-decoupled mixed precision quantization. In European Conference on Computer Vision, pages 285–302. Springer, 2024.
- [57] Qingping Zheng, Yuanfan Guo, Jiankang Deng, Jianhua Han, Ying Li, Songcen Xu, and Hang Xu. Anysize-diffusion: Toward efficient text-driven synthesis for any-size hd images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 7571–7578, 2024.
- [58] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Lirui Zhao, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. arXiv preprint arXiv:2406.18583, 2024.

### Appendix

In the appendix, we present additional implementation details (Section A), additional qualitative results (Section B), text prompts for image generation in both the main paper and appendix (Section C), more discussion on acceleration alignment (Section D), the limitations of our method (Section E) as well as its broader impacts (Section F), as a supplement to the main paper.

- A Additional implementation details

To enhance the Rectified Flow model’s stability when generalizing to extrapolated resolutions, we adopt inference techniques suggested by previous works [35, 22, 11] in the high-resolution generation process, including NTK-aware scaled RoPE [35], balancing the entropy shift of self-attention [22] and balancing the image/text sequence length ratio for MMDiT [11].

- B Additional qualitative results

We present more visual results of HiFlow integrated with various LoRA models at 4096 × 4096 resolution in Fig. F.3, Fig. F.4, Fig. F.5, Fig. F.6, Fig. F.7 and Fig. F.8, along with generated results using the same prompts and different random seeds (0/1/2) at 4096 × 4096 resolution in Fig. F.9. Moreover, additional results of HiFlow on SVDQuant [29] and SDXL [37] are shown in Fig. F.10.

- C Text prompts Text prompts used to generate images in this paper are provided in Tab. F.1, Tab. F.2 and Tab. F.3.
- D Discussion on acceleration alignment

In this section, we provide a deeper analysis on acceleration alignment. Given that Rectified Flow models are based on the premise that the noisy latent variable Xt transitions linearly between between the noise distribution π1 and the clean image distribution π0, it is logical to introduce an acceleration term drawn from physics to account for non-linear dynamics during the denoising process:

d2Xt dt2

dXt dt

dvt dt

, (14)

vt =

,at =

=

where vt is the denoising direction predicted by the model (the first derivative of Xt with respect to t), and at denotes the acceleration component (the first derivative of vt with respect to t, the second derivative of Xt with respect to t). The above definition is consistent with previous works [5] that introduced acceleration in image editing tasks. To explore the practical implication of at, Eq. 4 is substituted into the definition of acceleration in Eq. 14:

1

) − t1

i−1 − X0←t

i − X0←t

ti−1(Xt

) ti−1 − ti

(Xt

i−1 − vt

vt

dvt dt

. (15) Furthermore, based on Eq. 4 and Eq. 5, we are able to express Xt

i−1

i

i

i

at =

=

=

ti−1 − ti

and X0←t

in terms of Xt

, X0←t

i−1

i

i

through the following relation:

i−1

i − X0←t

Xt

ti−1 ti

i

(ti−1 − ti) = Xt

(ti−1 − ti) = X0←t

i − X0←t

Xt

= Xt

+ vt

+

+

(Xt

).

i−1

i

i

i

i

i

ti

(16) By plugging Eq. 16 into Eq. 15, Xt

is eliminated and the expression for at can be simplified as:

i−1

+ t

1

) − t1

i−1

i − X0←t

) − X0←t

i − X0←t

ti−1 (X0←t

ti (Xt

) ti−1 − ti

(Xt

i−1

i

i

i

i

at =

i−1 − X0←t

1 ti

X0←t

(17)

i

= −

,

ti−1 − ti

1 ti

dX0←t dt

= −

.

From Eq. 17, it is concluded that the acceleration term essentially depicts the first order derivative of the predicted clean sample X0←t with respect to t, multiplied by a time-dependent factor −1t. Therefore, the acceleration in the denoising process of Rectified Flow models primarily captures the variance in image details, including what details should be synthesized at each timestep and how these details should be synthesized for better fidelity, especially in the late denoising stage when the model focuses on the generation of fine image details. Fig. D.1 provides a visualization of the acceleration across different timesteps, and the results are consistent with our analysis.

t = 0.6 t = 0.3 t = 0.1 Result

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

#### Figure D.1: Visualization of acceleration at different timesteps.

The above discussion indicates that acceleration characterizes the model’s ordering, rhythm, and preferences in synthesizing image details, while the reference flow reflects how this information, learned at the training resolution, manifests in the high-resolution space. To this end, the acceleration alignment in Eq. 11 is introduced to guide the acceleration in high-resolution generation in the form of classifier-free guidance [18], for better detail fidelity.

### E Limitation and discussion

Despite the advancements of HiFlow in enhancing high-resolution image generation, it faces certain constraints. As a training-free method, the generation capability of HiFlow is highly dependent on the quality of the reference flow, which provides flow-aligned guidance in each sampling step. Therefore, the structural irregularities within the reference flow might compromise the quality of the final production. As shown in Fig. E.2, the low-resolution image of a tiger exhibits a duplicated tail, and this structural anomaly is preserved in the high-resolution generation result. Fortunately, given its model-agnostic nature, improvements can be anticipated when combining it with more advanced models, bypassing the need for fine-tuning or adjustment.

### F Broader impacts

The introduction of HiFlow, a novel training-free framework for high-resolution image generation, brings with it a range of societal impacts, offering significant benefits while also posing notable risks.

On the beneficial side, HiFlow’s ability to produce detailed, high-resolution images directly from textual prompts without requiring model retraining opens up exciting possibilities across multiple domains. In creative industries, such as design, marketing, and digital media, professionals can rapidly prototype visuals, enhance visual storytelling, and generate production-ready assets with minimal cost and effort. The high resolution and quality of the outputs make them suitable for both

[Figure 198]

[Figure 199]

Flux.1.0-dev 1024 × 1024 HiFlow 4096 × 4096

#### Figure E.2: Failure case of HiFlow.

conceptual and final-use purposes. In education and scientific communication, HiFlow can assist instructors and researchers in generating accurate visual aids tailored to specific topics, enriching the learning experience—especially in subjects that rely on precise imagery, like anatomy, architecture, or environmental science.

However, this powerful capability also introduces challenges. The ease and accessibility of generating photorealistic images without any fine-tuning may lead to the creation and spread of deceptive or harmful content, including synthetic news photos, manipulated evidence, or deepfake creation. Without robust safeguards, such misuse could contribute to misinformation, digital fraud, or the erosion of trust in visual media.

In conclusion, while HiFlow marks a significant milestone in the realm of high-resolution image generation, its responsible use requires the establishment of ethical standards, transparent practices, and public awareness. As the technology continues to advance, collaboration among developers, regulators, and society at large will be essential to ensure that its benefits are maximized while minimizing potential harms. By fostering accountability and encouraging the development of verification mechanisms, we are able to guide the use of HiFlow toward constructive and trustworthy applications.

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

#### Figure F.3: Visual results of HiFlow at 4096 × 4096 resolution.

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

#### Figure F.4: Visual results of HiFlow at 4096 × 4096 resolution.

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

#### Figure F.5: Visual results of HiFlow at 4096 × 4096 resolution.

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

#### Figure F.6: Visual results of HiFlow at 4096 × 4096 resolution.

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

#### Figure F.7: Visual results of HiFlow at 4096 × 4096 resolution.

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

#### Figure F.8: Visual results of HiFlow at 4096 × 4096 resolution.

[Figure 236]

[Figure 237]

Seed=2Seed=0Seed=1

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

#### Figure F.9: Generated results using same prompts and different seeds at 4096 × 4096 resolution.

[Figure 242]

[Figure 243]

SVD-Q 1024 × 2048 SVD-Q 2048 × 2048

[Figure 244]

SVD-Q 1024 × 2048

[Figure 245]

[Figure 246]

[Figure 247]

SVD-Q 2048 × 1024 SVD-Q 2048 × 1024 SVD-Q 2048 × 2048

[Figure 248]

[Figure 249]

SDXL 4096 × 4096 SDXL 4096 × 4096

#### Figure F.10: More visual results of HiFlow on SVDQuant and SDXL.

- Figure 1

Knight holding kitten in flower garden: A knight in full plate armor stands amidst blooming flowers, gently cradling a tabby kitten in their gauntleted hands. Sunlight filters through the foliage, creating a warm, dappled light. The kitten looks up at the knight with a curious expression. Focus on the contrast between the hard armor and the soft fur. Photorealistic style with a touch of fantasy. The image depicts a detailed and dynamic illustration of a Gundam, specifically Gundam Barbatos from the anime Mobile Suit Gundam: Iron-Blooded Orphans. This powerful mecha is portrayed in a close-up action shot, emphasizing its imposing and battle-worn appearance. Gundam Barbatos features a distinct design with a white and blue color scheme, accented by gold and red details. The mecha’s head is particularly striking, with a prominent yellow V-fin crest that is characteristic of many Gundam designs. Its eyes glow a vibrant green, adding to the intensity and life-like presence of the machine. The head unit also has a faceplate that suggests a fierce and determined expression, fitting for a battle-hardened warrior. The armor plating on B4RB4T0S appears worn and weathered, with visible scratches and damage, indicating that it has seen many battles. The chest area is reinforced with blue armor, and the overall structure of the Gundam is muscular and robust, reflecting its strength and combat capabilities. In the background, there are blurred elements that suggest a chaotic battlefield, with dark, smoky skies and hints of fire or explosions, adding to the dramatic atmosphere of the scene. The lighting and shading in the image are expertly done, creating a sense of depth and realism, while also highlighting the metallic surfaces of the Gundam. Overall, the image captures the essence of Gundam Barbatos as a powerful and relentless war machine, ready to engage in fierce combat. The detailed rendering and dynamic composition emphasize the Gundam’s role as a central figure in the struggle depicted in Iron-Blooded Orphans. A breathtaking mountain landscape featuring towering peaks with snow caps under a serene blue sky. The scene captures the early morning light, casting soft shadows and illuminating the majestic Tetons. A tranquil river runs through, reflecting the mountains and surrounding evergreen trees, creating a mirror-like effect. The composition emphasizes depth, framing the mountains with a cluster of pines on the left and gentle frost covering the ground. The cool color palette evokes a sense of calm and tranquility, enhanced by the crisp air and delicate mist lingering over the water. The scene embodies a peaceful wilderness atmosphere, reminiscent of fine art photography. A mouthwatering photograph of a well-plated gourmet burger and French fries, with a glass of cola with ice cubes in the background. Portrait of a bear as a roman general in a roman city-state, with a helmet, decorative, fantasy environment, detailed, sharp, clear, 8k. A whimsical village scene is nestled within an enormous teacup, where winding cobblestone streets and quaint cottages create a surreal microcosm reminiscent of dreamlike landscapes. From a high vantage point, the viewer looks down on the diminutive inhabitants going about their day-to-day activities, contrasting the juxtaposition of innocence and chaos in this fantastical setting. Soft, ethereal lighting envelops the scene, creating an atmosphere of tranquility as ordinary life intertwines with elements of fantasy and absurdity to weave a captivating visual narrative that invites the audience into an extraordinary world where the commonplace becomes surreal. Super detailed, half-body portrait of a beautiful young girl with flawless skin and golden hair, wearing an elegant ball gown. Posed in an ancient and opulent castle hall with ornate decorations and grand architecture in the background. Bright, soft light illuminates her face. Clear, perfect, and detailed face with brilliant blue eyes, full red lips, a friendly smile, pale skin with freckles, and vivid colors. A fluffy, round-faced British Shorthair cat with big expressive eyes, sitting calmly on a wooden desk in a cozy study room. The background features a bookshelf filled with vintage books and decorative objects. Soft, natural lighting enhances the cozy, nostalgic atmosphere. A blood-red Ferrari SF90 parked on a rain-soaked city street at night, reflecting neon lights from nearby buildings. The wet pavement glistens, and the car’s smooth curves are highlighted by the ambient glow of the urban environment. A professional photograph of a quirky squirrel with a fiery red mohawk, energetically playing a whimsical drum set in a vibrant autumn forest. The drum kit is entirely imagined, with bass drums made from hollowed, oversized chestnuts that have been polished to a gleaming finish. The snare drum is crafted from a woven basket of intertwined twigs and leaves, producing a crisp, earthy sound. Cymbals are formed from large, dry maple leaves, their edges curled and textured, hanging on thin branch stands. Smaller acorns serve as tom-toms, and the hi-hat is composed of overlapping oak leaves, delicately stacked on a twig frame. Every element of the drum set is organic, blending seamlessly into the forest floor, which is blanketed in a rich layer of autumn leaves, as the squirrel’s paws expertly tap the instruments, filling the scene with the imagined rhythm of nature. Wukong, wearing the armor of the Monkey King, holding his weapon with a serious expression, inside an abandoned traditional Chinese temple. A close-up of a blooming peony, with layers of soft, pink petals, a delicate fragrance, and dewdrops glistening in the early morning light. A futuristic robotic unicorn with a chrome-plated horn and neon-glowing hooves, shredding a halfpipe in a post-apocalyptic skatepark. Sparks fly from its hooves, and graffiti-covered ruins in the background in electric blue spray paint. Dynamic motion blur, ultra-realistic metallic textures, vibrant cyberpunk colors, Mad Max meets Lisa Frank aesthetic.

- Figure 2

A tiny astronaut hatching from an egg on the moon. A lighthouse standing tall against crashing waves. Teddy bear walking down 5th Avenue, beautiful sunset, close up, high definition, 4k.

A pair of navy blue Converse shoes displayed on the table. A highly detailed oil painting of tiger. A woman in pink T-shirt. Small cottage near the lake, summer.

- Figure 3

- Figure 5

A cute and adorable fluffy puppy wearing a witch hat in a halloween autumn evening forest, falling autumn leaves, brown acorns on the ground, halloween pumpkins spiderwebs, bats, a witch’s broom. A majestic, metallic giraffe walking through a futuristic city park, its long neck towering above the plants and creatures. A city at night, illuminated by neon lights, with cars zooming between towering skyscrapers, and a giant holographic owl in the sky. A mystical elf with emerald green eyes and a crown of leaves, standing in a forest glade bathed in soft moonlight.

- Figure 6

A majestic Bengal tiger strides confidently through a dense, misty rainforest. A poised young woman stands against a warm golden backdrop, exuding elegance and grace. She wears a richly textured red coat adorned with intricate floral patterns, layered over a crisp white collared shirt and a finely detailed red vest. Her look is elevated by ornate gold jewelry, including chandelier earrings and layered necklaces, which add a regal touch to her refined ensemble. Her long, softly wavy hair is adorned with a red floral accessory that complements her outfit, while her composed expression and direct gaze convey quiet confidence and sophistication. A cat wearing a tiny wizard’s hat, casting spells with a flick of its paw. A peaceful snowy village during the night. A detailed view of a blooming magnolia tree, with large, white flowers and dark green leaves, set against a clear blue sky.

- Figure 7

A gardener tending to a colorful, blooming garden, oil painting by Van Gogh. Portrait of a luxurious model in a plush coat.

- Figure 8

Furina from Genshin Impact sitting gracefully at a lavish dining table inside a grand palace hall. The setting is elegant and opulent, with towering arched windows letting in soft afternoon sunlight. The table is set for a classic English afternoon tea — delicate porcelain teacups, a silver teapot, and a multi-tiered tray filled with colorful macarons, scones with clotted cream and jam, finger sandwiches, and petits fours. Furina wears an ornate, Victorian-inspired outfit with lace gloves, looking both noble and contemplative. Crystal chandeliers hang overhead, casting a warm glow, and the palace decor is rich with gold accents and deep velvet drapes. Wukong sits cross-legged in deep meditation atop the blazing peaks of Flame Mountain. Surrounded by rivers of molten lava and pillars of rising smoke, his golden fur glows softly in the fiery light. His staff rests beside him, half-buried in scorched stone. Despite the searing heat and roaring flames around him, his expression is calm and focused, as if in perfect harmony with the chaos. The sky above is a storm of red and orange, casting a dramatic backdrop to his solitary training. A close-up portrait of a young woman with flawless skin, vibrant red lipstick, and wavy brown hair, wearing a vintage floral dress and standing in front of a blooming garden. Character of lion in style of saiyan, mafia, gangsta, citylights background, Hyper detailed, hyper realistic, unreal engine ue5, cgi 3d, cinematic shot, 8k. A jellyfish dances in the sea, against a backdrop of coral and seaweed. A product image of an iPhone standing upright in a forest. The iPhone’s screen shows a view of the forest with a dramatic lightning strike captured in the background. The surrounding forest is lush and dense with trees, and the sky above is stormy and dark, with the lightning illuminating the scene. The background is clear and detailed, highlighting the contrast between the natural elements and the advanced technology of the phone. This image emphasizes the iPhone’s high-quality display and camera capabilities, ideal for use in a tech blog or product design portfolio. A futuristic astronaut exploring an alien planet, with a detailed spacesuit and a landscape of strange plants and rock formations. A cat next to a stack of pancakes in a living room, best quality, extremely detailed, 8k. A frozen tundra with glowing ice spires and northern lights. Steampunk makeup, in the style of vray tracing, colorful impasto, dark cyan and amber makeup. Rich colourful plumes. Victorian style. Cinematic photo of delicious chocolate icecream.

- Figure E.2 A tiger is playing football.

- Figure F.3

Floating islands connected by waterfalls in a sunset sky. Summer landscape, vivid colors, a work of art, grotesque, Mysterious. A swirling night sky filled with bright stars and a small village below, inspired by Van Gogh’s Starry Night. A frozen tundra with glowing ice spires and northern lights. A dreamlike landscape emerges from a first-person viewpoint, immersing the observer in an alluring world where waterlilies of soft lavender and violet hues gracefully drift on the surface of an opalescent pond. Towering lotus blossoms stretch towards an indigo sky embellished with celestial bodies that gleam like stars, invoking both tranquility and awe. A regal swan presides over this fantastical garden, its iridescent feathers creating captivating ripples across the water that seem to distort time itself, crafting a harmonious melody of dreams and nature, encapsulating the spirit of beauty and whimsy in one stunning tableau. A serene lakeside during autumn, with trees displaying a palette of fiery colors.

- Figure F.4

A panoramic view of a city skyline at twilight, with skyscrapers and city lights. A long-exposure photograph of a vibrant city street at night, with light trails from moving cars. A deep forest clearing with a mirrored pond reflecting a galaxy-filled night sky. A desert oasis with palm trees and a shimmering pond. Primitive forest, towering trees, sunlight falling, vivid colors. A well-lit photograph of a modern, minimalist living room with large windows overlooking a city.

- Figure F.5

A classic still life composition featuring a teapot, teacup, flowers, and other decorative objects. Burning pile of money, epic composition, digital painting, emotionally profound, thought-provoking, intense and brooding tones, high quality, masterpiece. Create a hyper-realistic scene of a grand, classical hallway inside an opulent palace. The hallway is lined with towering columns and adorned with ornate, gilded paintings on the walls. Massive, powerful ocean waves surge through the corridor, crashing against the columns and splashing onto the walls. A jungle temple overgrown with vines and ancient carvings. A photograph of an abandoned building, showing decay and the interplay of light and shadow. A painting of brooklyn new york 1940 storefronts, by John Kay, highly textured, rich colour and detail, ballard, deep colour’s, style of raymond swanland, trio, oill painting, h 768, well worn, displayed, detailed 4k oil painting, glenn barr, textured oil on canvas, looking cute.

- Figure F.6

A photorealistic, cinematic still from a historical drama depicting a battle-worn medieval knight amidst the ruins of a burning village at night. The scene is illuminated by flickering flames and the faint glow of embers, casting long, dancing shadows across the landscape. The knight’s plate armor is heavily scarred and scratched, reflecting the firelight in fragmented patterns. A close-up portrait of a young woman with flawless skin, vibrant red lipstick, and wavy brown hair, wearing a vintage floral dress and standing in front of a blooming garden. Super detailed, selfie, college co-ed with pink hair in a pixie cut posed in front of her computer. Close up of face. brilliant blue eyes, full lips, pale skin and freckles. Portrait of man in a suit sitting in a gorgeous classical office. Cyberpunk hero with neon tattoos and futuristic armor. A charming, tech-savvy [girl with short, silver pixie-cut] hair and vibrant [blue] eyes, wearing a casual yet futuristic outfit. She’s focused on a holographic interface while working in a sleek, high-tech workshop.

- Figure F.7

A hamster piloting a tiny hot air balloon. Digital art of a beautiful tiger under an apple tree, cartoon style, matte painting, magic realism, bright colors, hyper quality, high detail, high resolution. A cinematic photo of a cat with flower. A Samoyed wearing a sunglasses, sticking out its tongue, dslr image, 8k. A cute corgi cooking in the kitchen. A sea turtle swimming through a coral reef.

- Figure F.8

Astronaut in a jungle, cold color palette, muted colors, detailed, 8k. A woman in a pink dress walking down a street, cyberpunk art, inspired by Victor Mosquera, conceptual art, style of raymond swanland, yume nikki, restrained, robot girl, ghost in the shell. A quirky miniature scene, potato chip soldiers parachuting onto a ceramic bowl filled with ridged potato chips, tiny plastic figurines suspended by yellow mushroom cloud-like parachutes, surreal food photography, soft lighting. A DSLR shot of fresh strawberries in a ceramic bowl, with tiny water droplets on the fruit, highly detailed, sharp focus, photo-realistic, 8K. By Tang Yau Hoong, ultra hd, realistic, vivid colors, highly detailed, UHD drawing, pen and ink, perfect composition, beautiful detailed intricate insanely detailed octane render trending on artstation, 8k artistic photography, photorealistic concept art, soft natural volumetric cinematic perfect light, ultra hd, realistic, vivid colors, highly detailed, UHD drawing, pen and ink, perfect composition, beautiful detailed intricate insanely detailed octane render trending on artstation, 8k artistic photography, photorealistic concept art, soft natural volumetric cinematic perfect light. Gentleman cyborg robot with a fish bowl head, with a small goldfish.

- Figure F.9

Brown wooden house in the middle of snow covered trees. A sleek, high-performance supercar cruises through the bustling city streets.

- Figure F.10

A magnificent Gothic castle perched on a cliff, with waves crashing against the cliff walls, sunset over sea. A bird is singing on the branch with tender leaf, beside it is its nest, and the morning sunlight is shining. Close-up Advertisement, a floating pool table with a chilled bottle of Coca Cola, around with lemons and ice-cubes, all set against the bright sun and clear water. Photo of a rhino dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography, Elke vogelsang. A beautiful woman, slightly bend and lower head, perfect face, pale red lips, Ultraviolet, Charlie Bowater style, Paper, The composition mode is waist shot style, Hopeful, Octane render, 4k HD, wearing a modest high-neck dress that elegantly covers the chest area, with intricate patterns reminiscent of traditional art. A heartwarming close-up scene featuring a playful kitten and a curious hedgehog sharing a cozy blanket. Einstein, a bronze statue, with a fresh red apple besides it, by Bruno Catalano. Astronaut on Mars During sunset.

