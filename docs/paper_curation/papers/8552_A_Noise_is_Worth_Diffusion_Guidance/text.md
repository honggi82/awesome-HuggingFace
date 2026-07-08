## A Noise is Worth Diffusion Guidance

Donghoon Ahn•1 Jiwon Kang•1 Sanghyun Lee◦2 Jaewon Min◦1 Minjae Kim1 Wooseok Jang1 Hyoungwon Cho1 Sayak Paul4 SeonHwa Kim1

Eunju Cha†3 Kyong Hwan Jin†1 Seungryong Kim†2

1Korea University 2KAIST 3Sookmyung Women’s University 4Hugging Face

# arXiv:2412.03895v1[cs.CV]5Dec2024

###### [Standard Gaussian Noise Space 𝑁(0,𝐼)] [Guidance-Free Noise Space]

[Figure 1]

|[Figure 2]|
|---|

|[Figure 3]|
|---|

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

|[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]|
|---|

|[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]|
|---|

|[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]|
|---|

[Figure 17]

(b) Add residual noise signals

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Low Quality High Quality

(c) Images generated from refined noise

(a) Images generated from random noise

without guidance

without guidance

Figure 1. Effectiveness of NoiseRefine. Diffusion models often fail to generate high-quality images without guidance, such as classifierfree guidance (CFG) [13]. We propose NoiseRefine, a novel approach to improve image quality without use of guidance by learning to map initial random noise to a guidance-free noise space. Results are demonstrated using Stable Diffusion 2.1[36].

### Abstract

fined noise can eliminate the need for guidance. See our project page: https://cvlab-kaist.github.io/NoiseRefine/.

Diffusion models excel in generating high-quality images. However, current diffusion models struggle to produce reliable images without guidance methods, such as classifierfree guidance (CFG). Are guidance methods truly necessary? Observing that noise obtained via diffusion inversion can reconstruct high-quality images without guidance, we focus on the initial noise of the denoising pipeline. By mapping Gaussian noise to ‘guidance-free noise’, we uncover that small low-frequency components significantly enhance the denoising process, removing the need for guidance and thus improving both inference throughput and memory. Expanding on this, we propose NoiseRefine, a novel method that replaces guidance methods with a single refinement of the initial noise. This refined noise enables high-quality image generation without guidance, within the same diffusion pipeline. Our noise-refining model leverages efficient noisespace learning, achieving rapid convergence and strong performance with just 50K text-image pairs. We validate its effectiveness across diverse metrics and analyze how re-

### 1. Introduction

In recent years, Text-to-Image (T2I) diffusion models [3, 6, 32, 36, 39], which generate images conditioned on text prompts, have achieved remarkable advancements. However, their ability to produce high-quality samples largely relies on guidance techniques, such as classifier-free guidance (CFG) [13] and its variants [1, 5, 15, 16]. These methods significantly enhance image quality during inference but double the computational cost. Despite drawbacks such as increased batch size, high guidance scale requirements, oversaturation, and reduced diversity, the dramatic performance gains make CFG the de facto standard. Recent works [4, 37, 38] aim to mitigate these limitations, but the impact of CFG on image quality makes it indispensable in most diffusion pipelines.

This raises a fundamental question: Can we replace the effects of guidance techniques with minimal changes to the diffusion pipeline? While some works have proposed

•,◦: Equal contribution, †: Co-corresponding author

[Figure 26]

Diffusion sampling w/o guidance

[Figure 27]

|𝑥0|
|---|

𝑥𝑇

Diffusion sampling w/ guidance

###### Possible to learn?

[Figure 28]

[Figure 29]

Inversion w/o guidance

Diffusion sampling

|𝑥0Guide|
|---|

###### w/o guidance

𝑥𝑇Guide

- Figure 2. Insight of NoiseRefine. We combine inversion methods [8, 29, 46] and guidance methods [1, 13, 15, 16, 21, 38] to establish a mapping between standard noise xT and guidance-free noise xGuideT .

distilling classifier-free guided scores into student models [19, 24, 30], these methods require extensive training data, significant computational resources, and large model capacities to replicate guided denoising trajectories. Instead, we explore the inputs of T2I diffusion pipelines: a prompt and an initial noise. A single prompt can yield diverse samples depending on the initial noise. In addition, recent studies [41, 50] highlighted the strong influence of initial noise on output quality, a correlation between the starting noise and the resulting image. In fact, we observe that, on rare occasions, certain random initial noises can produce high-quality images. This implies that if we could find such noise easily, we would successfully eliminate the need for guidance methods. Thus, we aim to find a noise space capable of generating high-quality images, which we term the ‘guidance-free noise space’.

Throughout this work, we explore how we can find this guidance-free noise space. Some works have proposed to select or optimize noise space to improve perceptual quality [7, 34] or prompt adherence [10]. However, those are not intended to replace the guidance techniques and then still rely on them. Furthermore, they typically require extensive iterations to optimize the input noise [7, 10, 34] and often only work in the few-step diffusion model [7].

Our key insight to find a guidance-free noise space lies in leveraging diffusion inversion methods [8, 29, 46] to obtain initial noises that are reconstructed to their corresponding high-quality images without guidance. Building on this, we generate multiple high-quality images using guidance techniques [1, 13] and apply inversion to these images. This process yields a collection of initial noises capable of reconstructing images without guidance, constructing the

guidance-free noise space. The overall concept is illustrated in Fig. 2.

We analyze the difference between inversion noise and standard Gaussian noise, and we find that the difference arises primarily from subtle variations in pixel values and is concentrated in the low-magnitude, low-frequency component. Our objective is for the noise refining model to learn the mapping from standard Gaussian noise to ‘guidancefree noise’.

Although we could directly learn the mapping between the initial noises and the inversion noises, error accumulation during the inversion process [8, 29] makes this approach suboptimal. Thus, we mitigate this inversion error by shifting the distance loss from noise space to its denoised image space. We refer to our method as NoiseRefine. Fig. 4 illustrates the overall training process of NoiseRefine.

Training noise refining model by backpropagating gradients through the full denoising steps can result in substantial memory consumption and unstable training due to multi-step gradient propagation. To address this challenge, we propose multistep score distillation (MSD), a simple yet effective technique enabling efficient full-step model optimization without incurring the high cost of backpropagation. Inspired by score distillation sampling (SDS) [33], MSD skips gradient computation within the denoising network during the denoising process. Notably, we found that skipping a computation of gradient not only reduces computational overhead but also accelerates model convergence.

With just prompts and the basic diffusion model, our approach easily accomplish self-distillation without the need for natural images. We demonstrate that sampling from the noise refined by our model produces high-quality images without guidance, as validated across various benchmarks. These results are comparable to images generated using CFG [13] and PAG [1] on the same diffusion model while being approximately twice as fast (compared to CFG alone) or three times as fast (compared to both CFG and PAG).

Our contributions are summarized as:

- • We identify the existence of a noise space that enables high-quality generation without guidance [1, 13], which we refer to as ‘guidance-free noise space’.
- • We show that the mapping between a standard normal distribution and guidance-free noise space can be efficiently learned by a neural network.
- • To reduce backpropagation costs in training, we propose Multistep Score Distillation that detaches gradients during denoising, accelerating convergence.
- • Our approach achieves a 2x speed-up compared to using guidance methods, maintaining comparable quality.

(a) Absolute Difference Histogram (b) Radial Frequency-Magnitude Plot

- Figure 3. Analysis on the relationship between the initial Gaussian noise xT and the guidance-free noise xGuideT . (a) shows the

histogram of the absolute difference between xT and xGuideT . Here, ‘Random’ denotes the setting where the both noises are replaced with independent gaussian white noise. (b) presents the magnitude difference between the 2D Fourier-transformed frequency components of F(xT) and F(xGuideT ). The difference between xT and xGuideT is significantly smaller than in the random case, which mainly corresponds to the low-frequency components.

### 2. Related Work

Diffusion guidance. Classifier Guidance (CG) [27] enhances fidelity by leveraging trained classifier gradients, albeit at the cost of diversity. CFG [13] models an implicit classifier to achieve similar effects. Ahn et al. [1] and Karras et al. [21] further generalize those guidance methods by intentionally generating lower-quality samples to guide the process toward improved outputs and other guidance techniques [15, 16, 38] generate ‘bad’ samples in various ways. While effective, these methods double computational and memory costs by requiring degraded sample generation at each step, which is essential to their operation.

Diffusion inversion. Denoising Diffusion Implicit Models (DDIM) [46] introduced deterministic sampling, enabling inversion from image to noise. This means that by starting the sampling process from the inverted noise, we can reconstruct the original images. Although DDIM Inversion [46] is the most commonly used inversion method for diffusion models, its reliance on linear approximation often leads to noticeable artifacts in reconstructed images. Several works [8, 29, 31] employ fixed-point iteration to reduce the approximation error. If guidance is used during inversion, then guidance must be applied during sampling to achieve the same generated image and the same holds in reverse.

Noise optimization. Optimizing or selecting noises with certain objectives has been a key research focus in diffusion models [7, 28, 41]. ReNO [7] optimizes noises based on reward models and Samuel et al. [41] proposed a bootstrapbased method to optimize initial noises for rare concept generation. However, these optimization methods require a substantial number of iterations, which poses a challenge

for real-world applications. Another approach [28] involves constructing a noise database to generate initial noise during inference but is not generalized to unseen prompts. Our approach overcomes both limitations by learning a direct mapping to the learned noise space, enabling efficient mapping with a single inference step and providing generalization power for new prompts.

### 3. Method

In this section, we first identify characteristics of mapping from the Gaussian noise space to guidance-free noise space, a space of initial noises that can be denoised into high-quality images without guidance (Sec. 3.1). Next, we introduce a method for learning this mapping from arbitrary Gaussian noise (Sec. 3.2). Finally, we demonstrate that, with a carefully designed dataset construction and filtering process, predicting guidance-free noise using a simple single-step neural network can effectively replace traditional guidance techniques [1, 13], enabling efficient and high-quality image generation without guidance (Sec. 3.3).

##### 3.1. Guidance-Free Noise Space

To obtain the guidance-free noise space, we emphasize the capability of inversion methods [8, 29, 46] to precisely reconstruct the original image without guidance. In theory, inverting an infinite set of natural images would fully capture this space, but this is infeasible. Instead, we leverage the powerful generation capabilities of text-to-image diffusion models [36], which produce high-quality images with guidance [1, 13].

Specifically, Gaussian noise xT is sampled from standard Gaussian distribution N(0,I) and denoised into a plausible image xGuide0 , using text prompt or condition c with CFG [13] or any other guidance method [1, 15]. Inverting the image with an inversion method [8, 29] gives us the noise xGuideT , defined as:

xGuideT := Inversion(DenoiseGuide(xT,c)), (1) where Inversion(·) and DenoiseGuide(·) denote inversion [8, 29] and denoising with condition c and guidance, respectively. A more detailed explanation of the notations can be found in supplementary material A.1. Note that the generated image xGuide0 and inversion noise xGuideT are conditioned on context c such as the text prompt, but we omit the notation c in the paper for simplicity. Now, we can map xT into xGuide0 . Ideally, if the mapping is consistent or generalizable, a neural network can learn to map initial noise to guidance-free noise. This concept is illustrated in Fig. 2.

We investigate the structure of this mapping by gen-

erating {xT,xGuideT } pairs via the aforementioned process with 10K randomly selected prompts from the MS-

COCO dataset [25]. We employ Stable Diffusion 1.5 [36],

|Initial noise<br><br>𝑥𝑇~𝑁(0,𝐼)|
|---|

|Generated image w/ Guidance 𝑥0Guide|
|---|

[Figure 30]

[Figure 31]

|× 𝑁′|
|---|

[Figure 32]

𝜃

Denoising process w/ guidance

| | |
|---|---|
|Distan|ce 𝒅|
| | |

[Figure 33]

Noise Refining LoRA

[Figure 34]

𝜃

residual connection

|Noise refining model 𝑔𝜙|
|---|

[Figure 35]

[Figure 36]

|× 𝑁|
|---|

[Figure 37]

Stop

gradient

𝜀0 𝑥𝑡

| | |
|---|---|
|?|?|
| | |

| |𝑥ො𝑡−1<br><br>|
|---|---|
| | |

|𝑥ො𝑡|
|---|

Backpropagation

|Guidance-free noise 𝑥ො𝑇|
|---|

|Generated image w/o Guidance 𝑥ො0|
|---|

Denoising process w/o guidance

- Figure 4. Training pipeline. We propose a training methodology to learn a mapping from initial noise to guidance-free noise. Given an initial Gaussian noise xT, the original diffusion model parameterized by θ generates an image xGuide0 using guidance [1, 13]. Noise refining model refines the initial noise xT to produce xˆT = gϕ(xT), which is then input to the original model to generate an image xˆ0 without guidance. By minimizing the distance between

two images d(xGuide0 , xˆ0), noise refining model effectively learns the desired mapping. Note that both noise refining model and original model also receive a prompt c as input, though this is omitted here for simplicity.

CFG [13], and ReNoise [8]. Comparing the absolute differences between xT and xGuideT to those between random noise instances, Fig. 3 (a) shows that the differences in {xT,xGuideT } pairs are significantly smaller than those of ‘Random’ pairs. These differences mainly correspond to low-frequency components in the frequency domain as shown in Fig. 3, which plots the magnitude differences between Fourier-transformed noises. This suggests that guiding initial noise with suitable small low-frequency components for given condition c can generate high-quality samples without additional guidance during the sampling stage.

##### 3.2. Learning to Map Guidance-Free Noise Space

Mitigating inversion error. A straightforward approach for learning a mapping to guidance-free noise space would be to learn the inversion noise directly. Although possible enough, inversion methods [8, 29, 46] have inherent limitations. They rely on approximations, which means true inversion noise xGuideT † is not guaranteed. Thus, attempting to learn this approximated inversion noise which includes inversion error may limit the performance. Hence, we try to sidestep this issue, by learning directly in the im-

(a) Optimizing in noise space

[Figure 38]

[Figure 39]

|[Figure 40]<br><br>[Figure 41]<br><br>Optimized noise (log freq)<br><br>Optimized noise (log freq)|
|---|

[Figure 42]

MSE loss

Inversion

Ground truth 𝑥

Inversion noise 𝑥𝑇Guide

[Figure 43]

(b) Optimizing in image space

[Figure 44]

MSE loss

[Figure 45]

[Figure 46]

Denoise

Ground truth 𝑥0Guide

[Figure 47]

Denoised image 𝑥 0

Figure 5. Comparison between noise optimization methods. We compare two methods to optimize a noise for target image generation. (a) illustrates direct optimization using inversion noise from the target image, while (b) shows optimization by minimizing the loss between denoised image and the target image. The rightmost column visualizes each optimized noise in a lowfrequency area, indicating the similarity between the two noises.

age space. To demonstrate the validity of our approach, we investigate whether reducing the image space distance d(x0,xGuide0 ) also effectively decreases the noise space distance d(xT,xGuideT ) as training progresses, where d is distance metric measuring the difference between two data points. This relationship is clarified in Proposition 1 and illustrated in Fig. 4. A detailed proof is provided in supplementary material A.2.

Proposition 1. Let xT be an initial noise, and suppose that x0 is the image obtained through denoising. Assuming Lipschitz continuity with distance metric d, for every xT, there exists a constant κ > 0 such that the following holds:

###### d(xT,xGuideT †) < κd(x0,xGuide0 ).

We support our proposition by conducting toy experiments. In Fig. 5, we compare two strategies: directly optimizing the noise xT with the empirical inversion noise xGuideT , which we treat as true, and optimizing the loss between denoised image xGuide0 and the target image x0. To visualize, we plot the low-frequency regions of two optimized noises, revealing their similarity and supporting our proposition.

Training pipeline. Our overall training framework is illustrated in Fig. 4. Given randomly sampled Gaussian noise xT and a prompt c, a diffusion pipeline takes the noise xT as an initial input and generates a guided image xGuide0 by executing N′ denoising steps with guidance [1, 13]. Our noise refining model gϕ(·) estimates the refined noise xˆT. Using this xˆT as initial input, the same diffusion pipeline generates an image xˆ0 through N denoising steps without guidance. To reduce the gap between xˆ0 and xGuide0 , we apply a distance loss d(ˆx0,xGuide0 ) = ∥xˆ0 − xGuide0 ∥22. In this

[Figure 48]

| |[Figure 49]|
|---|---|
| | |
| |[Figure 50]|
| | |

| |[Figure 51]|
|---|---|
| | |
| |[Figure 52]|
| | |

|[Figure 53]| |
|---|---|
| | |
|[Figure 54]| |
| | |

Figure 6. Comparison of optimization results using different loss functions. The orange line represents the optimization process using the full gradient of the MSE loss, while the blue line depicts optimization using MSD loss. The images are sample outputs generated every 1000 optimization steps. The result demonstrates faster convergence but higher image quality when using MSD Loss.

way, our model learns to guide the initial noise xT toward a guidance-free noise space.

For the architecture of the noise refining model gϕ(·), we found that by attaching a lightweight LoRA [17] to the pre-trained diffusion models, the noise refining model can effectively leverage the diffusion model’s rich knowledge of text and image information, allowing for faster convergence. Additionally, as shown in Fig. 3 (a), the difference between xT and xGuideT is slight; therefore, we incorporate residual connection [11] in the noise refining model gϕ(·) to enable the model to converge rapidly during training.

Multistep score distillation. Naively applying our method incurs high costs from backpropagating through the denoising network up to N times and requires significant memory usage. Such requirements are one of the main reasons that recent noise optimization work [7, 22] typically relies on one or few-step models. However, we aim to maximize the number of denoising steps N used in our method since the quality of xGuide0 affects the performance of our model prediction.

To circumvent the backpropagation costs of the full-step diffusion model, we propose a novel approach, multistep score distillation (MSD), inspired by score distillation sampling [33], where we detach gradients through a denoising network during backpropagation. Specifically, the typical denoising process LDenoise(gϕ(xT),θ,d) is defined as follows:

LDenoise(gϕ(xT),θ,d) := d D1 (...DT(gϕ(xT))),xGuide0

(2) where

Dt(x) = atxt + btϵ(θt)(x), (3) and at and bt are coefficients derived directly from the

(a) Absolute Difference Histogram (b) Radial Frequency-Magnitude Plot

Figure 7. Analysis of the relationship between the initial Gaussian noise xT and the refined noise xˆT. (a) shows a histogram of the absolute difference between xT and xˆT. (b) displays the magnitude difference between the 2D Fourier-transformed frequency components F(xT) and F(ˆxT). This demonstrates that the model refines the noise by appropriately adding small, low-frequency components similar to the results shown in Fig. 3.

DDIM sampler [46] and defined in supplementary material A.1.

In MSD, we perform the typical denoising process but detach the gradients on the denoising network ϵθ at each step. Specifically:

LMSD(gϕ(xT),θ,d) := d F1 (...FT(gϕ(xT))),xGuide0 ,

(4) where

Ft(x) = atxt + bt SG(ϵ(θt)(x)), (5) where SG(·) denotes the stop-gradient (detach) operation.

Our noise refining model gϕ(·) is trained to minimize LMSD(gϕ(xT),θ,d). Fig. 6 compares optimization results with and without detached gradients, showing that disabling gradients in the denoising network leads to faster convergence and sharper images at substantially lower computational costs. We validate our approach, showing that MSD serves as a close approximation to learning with fullgradient objective LDenoise(gϕ(xT),θ,d). This is clarified in the following proposition. We provide a detailed proof in supplementary material A.2.

Proposition 2. By approximating the gradients through Multistep Score Distillation (MSD) using detached gradients at each step, we approximate the full-gradient objective with a mild assumption. In conclusion, the two gradients can be approximated as follows:

∇ϕLDenoise(gϕ(xT),θ,d) ≈ k∇ϕLMSD(gϕ(xT),θ,d), (6) where k ∈ (0,1) is constant.

##### 3.3. Dataset Construction

We observe that some proportion of images generated with CFG [13] in Stable Diffusion 2.1 [36] exhibit low quality, often appearing blurry or displaying distorted facial

(ours) Refined noise w/o guidance

(ours) Refined noise w/o guidance

Random noise w/o guidance

Random noise w/o guidance

Random noise w/ guidance

Random noise w/ guidance

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

“A black colored car” “A steampunk airship”

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

“Three persons with military attire seated on a bench” “Portrait of young Jerry Lewis in comic style, colorized and

created digitally by four artists.”

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

“majestic cat mountain top” “a quaint cottage in the forest, mossy, along a creek, in the

rain”

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

“In late afternoon in January in New England, a man stands in the shadow of a maple tree.”

“Beefy cowboy, tucked in shirt”

Figure 8. Qualitative results. Samples starting from Gaussian noise and generated without guidance (left), samples starting from Gaussian noise and generated with sampling guidance (middle), and samples starting from refined noise and generated without guidance (right).

features, eyes, and noses. Fortunately, our framework is not constrained to CFG. It can incorporate any qualityenhancing techniques applicable at inference time [15, 16], including PAG [1], which is known to improve structural accuracy. To enhance the quality of samples, we apply PAG along with CFG, as PAG has been shown to reduce blurriness and improve anatomical structure effectively. Additionally, to mitigate the bias of a fixed guidance scale, we use both CFG and PAG with a randomly varied scale.

Furthermore, we filter out low-quality images prone to structural issues or artifacts using a human preference model [43], effectively eliminating poor samples. Detailed implementation details are provided in Sec. 4.2. As shown in Tab. 3, this filtering produces superior results compared to training without these enhancements. Note that gener-

ating guided image xGuide0 can be done either online or offline; that is, xGuide0 can be pre-generated to enhance computational efficiency.

### 4. Experiment

In this section, to show the effectiveness and efficiency of noise refining model, we present extensive qualitative and quantitative results on Stable Diffusion 2.1 [36]. We train noise refining model with text prompts of MS-COCO [25] and Pick-a-pic [23]. CFG [13] and PAG [1] are applied to generating images from those datasets with filtering by aesthetic score [43]. We evaluate our model using text prompts of Drawbench [39], HPDv2 [48], Pick-a-pic [23], and MSCOCO [25]. More implementation details and experimental

###### Dataset Initial Noise Guidance PickScore [23] HPSv2 [48] AES [43] ImageReward [49] CLIPScore [35]

Gaussian ✗ 19.67 0.1689 5.129 -1.399 25.16

Ours ✗ 20.90 0.2306 5.351 -0.291 29.40 Gaussian ✓ 21.29 0.2482 5.475 -0.020 30.36

DrawBench [39]

Gaussian ✗ 19.18 0.1778 5.319 -1.299 26.85

Ours ✗ 20.47 0.2386 5.608 -0.163 31.21 Gaussian ✓ 21.02 0.2469 5.788 0.159 32.27

HPD [48]

Gaussian ✗ 18.89 0.1919 5.226 -1.520 24.89 Ours ✗ 20.18 0.2347 5.497 -0.304 29.40

Pickapic [23]

- Gaussian ✓ 20.67 0.2559 5.651 0.018 30.53

MS-COCO [25]

Gaussian ✗ 19.56 0.1654 5.138 -1.134 26.45 Ours ✗ 21.01 0.2474 5.368 0.338 30.27

- Gaussian ✓ 21.59 0.2504 5.487 -0.058 31.29

Table 1. Quantitative comparison of difference metrics across datasets. Starting from a refined noise using noise refining model consistently yields higher human preference scores than starting with Gaussian noise, with scores comparable to the guidance case (CFG [13] + PAG [1]).

settings can be found in supplementary material D.

##### 4.1. Results

Qualitative Comparison. We present our qualitative results evaluated on the aforementioned T2I benchmark datasets on Fig. 8. We observe notably degraded image quality when the initial noise is Gaussian and guidance is not applied. In contrast, when using the noise refined by our model, we observe consistently superior image quality compared to images from Gaussian noise. Moreover, images from our refined noises without guidance show comparable quality to those from guidance. This result demonstrates the effectiveness of noise refining model. Additional qualitative results can be found in supplementary materials C.

Quantitative Comparison. To comprehensively evaluate image fidelity and diversity with FID [12] and IS [40], we generate samples from 30K randomly selected prompts on MS-COCO 2014 validation set [25] using three methods: sampling without guidance from Gaussian noise, sampling with guidance from Gaussian noise, and sampling without guidance from noise refined by noise refining model (ours).

Tab. 2 presents FID [12] and IS [40] evaluation results and Tab. 1 presents the results of human preference scores [23, 43, 48, 49] and prompt adherence evaluation [35]. Across all metrics, our model shows a consistent and substantially improved quality over that of images from Gaussian noise, achieving results comparable to those obtained with guidance. Notably, our model achieves a lower FID [12] score than even the guidance setting, addressing concerns about potential declines in image fidelity and diversity when using model-generated data as training data. This result highlights the advantage of our method in enabling efficient training without relying on large-scale image datasets. More details about those datasets for evaluation can be found in supplementary material D.

User Study. Tab. 4 shows the results of user study, confirming noise refining model’s comparable to results starting from Gaussian initial noise without guidance. 45 participants compared 30 image pairs generated with guidance and our method (refined noise without guidance), using generated images for evaluation in Tab. 1, and evaluated visual appealiing and prompt alignment.

Initial Noise Guidance FID [12] ↓ IS [40] ↑ Inference Time ↓

Gaussian ✗ 42.71 20.86 1.357s

Ours ✗ 11.39 35.73 1.504s Gaussian ✓ 13.38 37.64 2.589s

- Table 2. Quantitative comparison of image quality and computational cost.

Parameter FID [12] ↓ IS [40] ↑ # of steps

5 13.74 30.80

10 13.36 32.81 Filtering

✗ 15.27 29.44 ✓ 13.55 31.15

- Table 3. Ablation study on the number of denoising steps and dataset filtering during training.

Metric Gaussian Noise + CFG Refined Noise (Ours)

Image Quality 46.04% 53.96% Prompt Adherence 48.24% 51.76%

- Table 4. User study on the image quality and prompt adherence of generated images.

##### 4.2. Ablation Study

Number of denoising steps. We demonstrate that the number of denoising steps significantly impacts performance. Specifically, we compare cases with denoising steps

N = 5 and N = 10, reporting FID [12], IS [40] in Tab. 3. The results indicate improved performance with a bigger number of denoising steps. Considering that a high number of steps (e.g., N ≥ 10) incurs prohibitive backpropagation costs, this supports the necessity of MSD to circumvent backpropagation costs.

Dataset filtering. To evaluate the impact of dataset filtering, we generate images using 80K prompts from the MS-COCO [25] dataset and remove those with an aesthetic score (AES) [43] below 6.0, resulting in about 25% of the images remaining. As shown in Tab 3, metrics for generated images are significantly improved in the filtered dataset, demonstrating the substantial impact of dataset filtering on enhancing guidance-free image generation.

### 5. Analysis and Discussion

We analyze what noise refining model learns and identify components in refined noise that contribute to guidancefree generation, discussing the advantages of working in this space.

##### 5.1. What Does noise refining model Learn?

In Fig. 7, we show that our model refines the noise by adding mostly small, low-frequency components. The distribution of absolute norm and frequency of the added noise difference is similar to the noise difference between a Gaussian noise xT and the inversion noise xGuideT shown in Fig. 3, without explicit constraints to achieve this.

Low-frequency components aid denoising. Fig. 9 shows that the low-frequency components in the noise difference are dependent on the condition (e.g., text prompt) and act as an initial layout for the synthesized image. These low-frequency signals significantly help diffusion models in forming object shapes in the early steps of denoising.

- Fig. 10 (a) shows that starting from refined noise, the model quickly establishes plausible images at much earlier stages, allowing the model to focus on adding details within the given layout during denoising. In contrast, as shown in
- Fig. 10 (b), the diffusion model struggles to create a coherent layout in the early denoising steps, resulting in partial details filled in incorrect locations, often leaving ambiguous regions untouched throughout the denoising process.

Diversity and generalizability. While these initial layouts might appear to limit diversity, Fig. 9 shows that the generated layouts vary significantly depending on the initial noise. We confirm this with diversity metrics of IS [40], demonstrating greater diversity than guidance-based methods. In addition, our model generalizes beyond the training data, performing well with unseen noise and prompts (Fig. 9, Table. 1), suggesting a generalizable mapping between initial and refined noise.

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

- Figure 9. Visualization of noise difference between xT and xˆT. The top row shows the difference, while the bottom row displays the corresponding generated images. The added signal functions as a layout, guiding the structure of the image during generation. Here, prompt ‘a photo of corgi’ is used.

(b) Starting from Gaussian noise

(a) Starting from refined noise

[Figure 89]

[Figure 90]

Denoising Direction

𝑡 = 𝑇 𝑇 − 1 … 𝑡 = 0

- Figure 10. Visualization of denoising process. (a) Starting from refined noise aids the model in establishing the overall layout early in the generation process, facilitating the successful creation. (b) In contrast, when beginning with Gaussian noise, the model struggles to capture the overall layout, resulting in incomplete or disjointed details rather than producing a fully plausible image.

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

- Figure 11. Comparison of training noise refining model gϕ (top) and the denoising network ϵθ (bottom). This demonstrates that tuning the input of the diffusion pipeline converges faster than tuning the pipeline (denoising network) within the same number of training iterations.

##### 5.2. Why Learn Noise Mapping?

We consider the rise of prompt learning. Large-scale models like CLIP [35], trained on web-scale datasets often contain up to billions of parameters. Fine-tuning such models is impractical and risks disturbing well-learned representations [52]. Instead, tuning the input prompts of the model, a method known as prompt learning, has gained popularity as an effective approach [18, 45, 51, 52].

In this context, limiting training to the noise space can be seen as an efficient alternative to tuning the entire denoising pipeline. As seen in Fig. 7 and Fig. 9, certain low-frequency components in the noise space provide critical information, such as image layout, which allows learning with a smaller dataset without the need to tune all model parameters. Unlike typical text-to-image diffusion models [36] or their distilled versions [24, 32, 42], which require up to billions of images [44], our model achieves effective training with only 50K noise and self-generated image pairs.

We verify this by tuning the diffusion model. However, as shown in Fig. 11, this approach led to slower convergence and frequent loss explosions, making training unstable. Details and discussion of this comparison can be found in supplementary material E.

### 6. Conclusion

In this work, we propose NoiseRefine, an efficient and effective approach to replacing guidance in diffusion sampling with a noise refinement by a single neural network forward pass. Our noise refining model functions as a plug-and-play module based on the original diffusion model and significantly improves image fidelity. Furthermore, our method is highly efficient, which can be trained using lightweight lora, requiring only a small set of modelgenerated images for training and remaining feasible on consumer-grade GPUs, thanks to our proposed MSD loss. Beyond its practicality, we believe this work serves as a stepping stone toward a deeper understanding of the role of guidance and noise in diffusion models.

### References

- [1] Donghoon Ahn, Hyoungwon Cho, Jaewon Min, Wooseok Jang, Jungwoo Kim, SeonHwa Kim, Hyun Hee Park, Kyong Hwan Jin, and Seungryong Kim. Self-rectifying diffusion sampling with perturbed-attention guidance. arXiv preprint arXiv:2403.17377, 2024. 1, 2, 3, 4, 6, 7, 11, 12, 15
- [2] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–10, 2023. 16
- [3] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 1
- [4] Hyungjin Chung, Jeongsol Kim, Geon Yeong Park, Hyelin Nam, and Jong Chul Ye. Cfg++: Manifold-constrained classifier free guidance for diffusion models. arXiv preprint arXiv:2406.08070, 2024. 1
- [5] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 1

- [6] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 1
- [7] Luca Eyring, Shyamgopal Karthik, Karsten Roth, Alexey Dosovitskiy, and Zeynep Akata. Reno: Enhancing one-step text-to-image models through reward-based noise optimization. arXiv preprint arXiv:2406.04312, 2024. 2, 3, 5
- [8] Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, and Daniel Cohen-Or. Renoise: Real image inversion through iterative noising. arXiv preprint arXiv:2403.14602, 2024. 2, 3, 4
- [9] Daniel Geng, Inbum Park, and Andrew Owens. Factorized diffusion: Perceptual illusions by noise decomposition. In European Conference on Computer Vision, pages 366–384. Springer, 2025. 15
- [10] Xiefan Guo, Jinlin Liu, Miaomiao Cui, Jiankai Li, Hongyu Yang, and Di Huang. Initno: Boosting text-to-image diffusion models via initial noise optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9380–9389, 2024. 2, 16
- [11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 5
- [12] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 7, 8
- [13] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 1, 2, 3, 4, 5, 6, 7, 11, 12, 14, 15, 16, 18
- [14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1, 2
- [15] Susung Hong. Smoothed energy guidance: Guiding diffusion models with reduced energy curvature of attention. arXiv preprint arXiv:2408.00760, 2024. 1, 2, 3, 6, 11
- [16] Susung Hong, Gyuseong Lee, Wooseok Jang, and Seungryong Kim. Improving sample quality of diffusion models using self-attention guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7462– 7471, 2023. 1, 2, 3, 6, 11, 15
- [17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 5, 12, 13
- [18] Zhengbao Jiang, Frank F Xu, Jun Araki, and Graham Neubig. How can we know what language models know? Transactions of the Association for Computational Linguistics, 8: 423–438, 2020. 8
- [19] Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, and Taesung Park. Distilling diffusion models into condi-

- tional gans. In European Conference on Computer Vision, pages 428–447. Springer, 2025. 2
- [20] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 2, 19
- [21] Tero Karras, Miika Aittala, Tuomas Kynk¨a¨anniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. arXiv preprint arXiv:2406.02507, 2024. 2, 3, 11
- [22] Jeeyung Kim, Ze Wang, and Qiang Qiu. Model-agnostic human preference inversion in diffusion models. arXiv preprint arXiv:2404.00879, 2024. 5, 18, 19
- [23] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36: 36652–36663, 2023. 6, 7, 12
- [24] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxllightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024. 2, 9
- [25] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 3, 6, 7, 8, 12, 13, 19
- [26] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022. 19
- [27] Jiafeng Mao, Xueting Wang, and Kiyoharu Aizawa. Guided image synthesis via initial image editing in diffusion model. In Proceedings of the 31st ACM International Conference on Multimedia, pages 5321–5329, 2023. 3
- [28] Jiafeng Mao, Xueting Wang, and Kiyoharu Aizawa. Semantic-driven initial image construction for guided image synthesis in diffusion model. arXiv preprint arXiv:2312.08872, 2023. 3, 16
- [29] Barak Meiri, Dvir Samuel, Nir Darshan, Gal Chechik, Shai Avidan, and Rami Ben-Ari. Fixed-point inversion for text-toimage diffusion models. arXiv preprint arXiv:2312.12540,

2023. 2, 3, 4

- [30] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14297–14306, 2023. 2
- [31] Zhihong Pan, Riccardo Gherardi, Xiufeng Xie, and Stephen Huang. Effective real image editing with accelerated iterative diffusion inversion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15912– 15921, 2023. 3
- [32] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion mod-

- els for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 1, 9
- [33] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2, 5
- [34] Zipeng Qi, Lichen Bai, Haoyi Xiong, et al. Not all noises are created equally: Diffusion noise selection and optimization. arXiv preprint arXiv:2407.14041, 2024. 2
- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 7, 8, 13
- [36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 3, 5, 6, 9, 12
- [37] Seyedmorteza Sadat, Jakob Buhmann, Derek Bradley, Otmar Hilliges, and Romann M Weber. Cads: Unleashing the diversity of diffusion models through condition-annealed sampling. arXiv preprint arXiv:2310.17347, 2023. 1
- [38] Seyedmorteza Sadat, Manuel Kansy, Otmar Hilliges, and Romann M Weber. No training, no problem: Rethinking classifier-free guidance for diffusion models. arXiv preprint arXiv:2407.02687, 2024. 1, 2, 3, 11
- [39] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 1, 6, 7, 12
- [40] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 7, 8
- [41] Dvir Samuel, Rami Ben-Ari, Simon Raviv, Nir Darshan, and Gal Chechik. Generating images of rare concepts using pretrained diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4695–4703, 2024. 2, 3
- [42] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer,

2025. 9

- [43] Christoph Schuhmann. Improved aesthetic predictor. https://github.com/christophschuhmann/ improved-aesthetic-predictor, 2022. 6, 7, 8, 12
- [44] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 9
- [45] Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. Autoprompt: Eliciting knowl-

- edge from language models with automatically generated prompts. arXiv preprint arXiv:2010.15980, 2020. 8
- [46] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2, 3, 4, 5, 1, 11, 12, 14, 15, 18, 19
- [47] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 18
- [48] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341,

2023. 6, 7, 12

- [49] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation.(2023). ArXiv Prepr ArXiv230405977. 7
- [50] Katherine Xu, Lingzhi Zhang, and Jianbo Shi. Good seed makes a good crop: Discovering secret seeds in text-toimage diffusion models. arXiv preprint arXiv:2405.14828,

2024. 2

- [51] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16816–16825,

2022. 8, 13

- [52] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9):2337–2348,

2022. 8

## A Noise is Worth Diffusion Guidance

### Supplementary Material

In the supplementary material, we clarify the notations and formulations related to diffusion models used in the main paper and provide the proofs for our propositions (Section A), more ablation studies regarding noise refining model (Section B), additional results including qualitative results, comparison with other methods, user study (Section C), implementation details and experimental settings (Section D) and further discussions (Section E).

### A. Theoretical Background

##### A.1. Preliminaries

Denoising Diffusion Probabilistic Models (DDPM). DDPM [14] defines a forward process that derives xt by adding Gaussian noise to the image xt−1 according to the variance schedule, and a reverse process that samples xt−1 from xt, both as a Markovian chain. The forward process is defined as

αt αt−1

αt αt−1

I , (7) q(xt|x0) = N(xt;√αt x0,(1 − αt)I), (8)

xt−1, 1 −

q(xt|xt−1) = N xt;

with noise rate at timestep t as 1 − αt/αt−1, where αt denotes noise scaling factors up to time step t. The reverse process is defined below.

pθ(xt−1|xt) = N xt−1;µ(θt)(xt),σt2I . (9) To reparameterize the equation using

xt = √αtx0 + √1 − αt ϵ for ϵ ∼ N(0,I), (10) and ϵθ, which is a function approximator for predicting ϵ from xt, the inference process becomes

1 − α

t

1

√1 −αtα−1t

ϵ(θt)(xt) + σtz, (11)

xt −

xt−1 =

αt αt−1

Where z ∼ N(0,I) and σt2 denotes the variance of Gaussian trainsitions .The objective of DDPM is defined as

0,ϵ ∥ϵ − ϵ(θt)(xt)∥2 , (12)

Lsimple(θ) = Et,x

where the L2 loss between the actual noise ϵ added during training and the noise prediction ϵθ(xt,t) for uniformly sampled t ∈ {1,...,T}.

Denoising Diffusion Implicit Models (DDIM). DDIM [46] consider the following inference distributions:

T

qσ(xt−1|xt,x0). (13)

qσ(x1:T|x0) := qσ(xT|x0)

t=2

with a mean function as below.

√αtx0 √1 − αt

xt −

√αt−1x0 + 1 − αt−1 − σt2 ·

,σt2I . (14)

qσ(xt−1|xt,x0) = N

Distinctively from DDPM, the forward process is Non-Markovian since each xt could depend on both xt−1 and x0. Reparameterizing with ϵθ, we can sample xt−1 from xt through an equation:

√1 − αt ϵ(θt)(xt) √αt

xt −

xt−1 = √αt−1

+ 1 − αt−1 − σt2 · ϵ(θt)(xt) + σtϵt

predicted x0

= atxt + btϵ(θt)(x),

where ϵt ∼ N(0,I) and at = √αt−1/√αt, bt = √1 − αt−1 − at√1 − αt. The objective of DDIM is the same as that of DDPM:

0,ϵ ∥ϵ − ϵ(θt)(xt)∥2 . (15)

LDDIM(θ) = Et,x

Denoising and inversion process. We denote the denoising process as Denoise(xT). When using the DDIM sampler [46], the denoising process is defined as:

Denoise(xT) := D1 (...DT(gϕ(xT))), (16) where each step Dt is given by:

Dt(x) := atxt + btϵ(θt)(x). (17)

The guided denoising process, denoted as DenoiseGuide(xT,c), follows the same steps as Eq. 16, but replaces ϵ(θt)(x) with guided scores, such as the classifier-free guided score ϵCFGθ (xt,c) [13], the perturbed-attention guided score ϵPAGθ (xt) [1], or a combination of both (ϵCFG,PAGθ (xt)). These guided scores are defined in Eqs. 26, 27, and 28.

While we utilize the DDIM scheduler in this work, any other diffusion scheduler [14, 20, 46] can be used by appropriately modifying at and bt.

For the inversion process Inversion(x0,c), we follow the method in [8] to obtain the initial noise xT, which can be denoised back to the given image x0 without employing any guidance methods [1, 13] during inversion.

##### A.2. Derivations

- Proposition 1. Let xT be an initial noise, and suppose that x0 is the image obtained through denoising. Assuming Lipschitz continuity with distance metric d, for every xT, there exists a constant κ > 0 such that the following holds:

d(xT,xGuideT †) < κd(x0,xGuide0 ). proofs. The Lipschitz condition is expressed as follows:

d(ϵ(θt)(x),ϵ(θt)(y)) ≤ Ltd(x,y), (18)

where Lt is constant dependent on t, x and y are arbitrary inputs to ϵ(θt). DDIM step in terms of xt can be expressed as follows:

 ϵ(θt)(xt). (19)

  1 − αt−1 −

αt−1(1 − αt) αt

αt−1 αt

xt +

xt−1 =

Eq. (19) can be expressed in terms of xGuidet † which is denoised from xGuideT †. With those equations, we can get the following equation,

 (ϵ(θt)(xt) − ϵ(θt)(xGuidet †))

  1 − αt−1 −

αt−1(1 − αt) αt

αt−1 αt

xt−1 − xGuidet−1 † =

(xt − xGuidet †) +

αt−1 αt

(xt − xGuidet †) − γt(ϵ(θt)(xt) − ϵ(θt)(xGuidet †)),

=

√1 − αt−1 > 0. If the distance metric d have translation invariance, the equation can be expressed as follows with Eq. (18):

where γt = αt−1(1 − αt)/αt −

αt αt−1

d(xt−1,xtGuide−1 †) ≤

(1 + γtLt)d(xt,xGuidet †). (20) Recursively organizing Eq. (20) for t = T,T − 1,...,1, it can be expressed as follows:

T

αT α0

d(xT,xGuideT †) ≤

d(x0,xGuide0 †). (21)

(1 + γtLt)

t=1

Since αT is close to 0, using d(x0,xGuide0 †) is sufficient to directly learn xGuideT † if d(x0,xGuide0 †) is small enough.

- Proposition 2. By approximating the gradients through Multistep Score Distillation (MSD) using detached gradients at each step, we approximate the full-gradient objective with a mild assumption. In conclusion, the two gradients can be approximated as follows:

###### ∇ϕLDenoise(gϕ(xT),θ,d) ≈ k∇ϕLMSD(gϕ(xT),θ,d), (22) where k ∈ (0,1) is constant.

proofs. Since the only difference between the two losses is the stop gradient in the diffusion model and all other components are identical, it suffices, by the chain rule, to show that the gradient of F1(F2(...FT(gϕ(xT)) with respect to ϕ is proportional to the gradient of Denoise(gϕ(xT)) with respect to ϕ. The derivation proceeds as follows:

∇ϕDenoise(gϕ(xT)) = ∇ϕ

=

α0 αT

T

α0 αT

α0 αt−1

γtϵ(θt)(xk)

gϕ(xT) −

t=1

T

∂ϵ(θt)(xt) ∂xt

α0 αt−1

∂xt ∂gϕ(xT)

I −

γt

t=1

∂gϕ(xT) ∂ϕ

.

(23)

As detailed in B, the term ∂ϵ(θt)(xk)/∂xk can be approximated as being proportional to the identity matrix. Additionally, the term ∂xk/∂gϕ(xT) can be expressed in terms of ∂ϵ(θt)(xk)/∂xk. Then, each component of ∂ϵ(θt)(xk)/∂xk can be approximated by the identity matrix. Consequently, (∂ϵ(θt)(xk)/∂xk)(∂xk/∂gϕ(xT)) becomes proportional to the identity matrix. Denoting the proportionality constant as ηt, Eq. (23) is simplified as follows:

(23) =

α0 αT −

√αT

= 1 −

√αT

= 1 −

T

α0 αt−1

∂gϕ(xT) ∂ϕ

γtηt

t=1

T

1 √αt−1

α0 αT

∂gϕ(xT) ∂ϕ

γtηt

t=1

T

1 √αt−1

γtηt ∇ϕF1(F2(...FT(gϕ(xT))).

t=1

(24)

### B. More Ablation Studies

##### B.1. Diffusion Model Jacobian Approximation

In this section, we present experimental results demonstrating that the Jacobian of the diffusion model ϵtθ with respect to the input xt can be approximated as proportional to the identity matrix. Fig. 12 illustrates the Jacobian of ϵtθ. We observe that the Jacobian of diffusion model behaves like the identity regardless of the timestep, except when t is significantly small. In such cases, the deviation does not affect our primary analysis. According to the results of Proposition 2, the timestep-dependent constant multiplied to each Jacobian term ηt is expressed as follows:

1 − αt αt −

1 − αt−1 αt−1

1 √αt−1

. (25)

γt =

[Figure 101]

This value can be numerically determined based on the scheduling, and in the case of DDIM [46], it is presented in Fig. 13. The graph shows that the constant decreases as t approaches 0, becoming 0.

- Figure 12. Visualization of Jacobian of a denoising network. Starting from T = 1000, we performed denoising over 10 steps and plotted the Jacobian heatmap at each timestep.We extracted a 500 × 500 section from the full Jacobian matrix for visualization. Each plot demonstrates that the Jacobian is close to the identity matrix.

- Figure 13. Visualization of constant values √αγtt−1 over timesteps. Visualization of the time-dependent constant value √αγtt−1 corresponding to Equation (19) across different timesteps. The results numerically demonstrate that for small timesteps, where the Jacobian deviates from the identity matrix, the multiplied constant values are sufficiently close to zero.

##### B.2. Utilizing Pretrained Knowledge of Diffusion Models

To train noise refining model, we adopt attaching LoRA layers to the original model to effectively leverage its pretrained knowledge. To assess the impact of pretrained knowledge, we conduct an ablation study. The comparison involves noise refining model and the same UNet architecture of the original model, but trained from scratch. We used only filtered MS COCO dataset among both datasets and trained models for 25K steps using two RTX 3090 GPUs. All the other experimental settings are kept consistent.

Qualitative results are presented in Fig. 14, and quantitative results are detailed in Tab. 5. Both results indicate that leveraging pretrained knowledge results in superior performance compared to training from scratch.

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

- Figure 14. Qualitative comparison with noise refining model (top) and UNet trained from scratch (bottom).

###### Model FID

UNet trained from scratch 37.87 Pretrained + LoRA (noise refining model) 13.74

Table 5. Quantitative comparison with noise refining model and UNet trained from scratch.

### C. Additional Results

##### C.1. Additional Qualitative Results

We present our additional qualitative results on Fig. 15, 16, 17 and 18. Results show that the performance of using refined noise by noise refining model is comparable to that of using guidance on random Gaussian noise. All the results are selected from images used in Tab. 1 and 2.

##### C.2. User Study

We conducted a user study to evaluate prompt adherence and image quality by comparing images generated from random Gaussian noise and our refined noise. The results are presented in Tab. 6. The study demonstrates that our method outperformed the baseline in all human evaluation criteria. A total of 26 participants anonymously evaluated 20 pairs of images, each pair consisting of an image generated using initial Gaussian noise and our refined noise from noise refining model. The percentage was calculated by dividing the total number of selections for each option by the total number of responses, following the same methodology as in Tab. 4.

Participants were provided with the following instructions for each pair of images:

- 1. Which image has better overall quality? (left/right)
- 2. Which image more faithfully reflects the given prompt? (left/right)

#### Metric Gaussian Noise Refined Noise (Ours)

Image Quality 3.08% 96.92% Prompt Adherence 6.73% 93.27%

Table 6. User study on the image quality and prompt adherence of generated images.

(ours) Refined noise w/o guidance

(ours) Refined noise w/o guidance

Random noise w/ guidance

Random noise w/o guidance

Random noise w/o guidance

Random noise w/ guidance

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

“Two cars on the street.” “a bedroom with a night stand near the bed”

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

“A large motor vehicle carrying passengers by road, typically one serving the public on a fixed route and for a fare.”

“A woman in sunglasses and hat standing by plant.”

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

“A brightly painted temple with ornate structures and dramatic lighting inspired by Mayan and Islamic architecture.”

“A bathroom is shown with a toilet, sink and a mirror.”

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

“opal gun” “A close-up of grains and pastries on a table.”

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

“The stainless steel refrigerator is being moved into the newly constructed home.”

“A small vase with a few flowers is in the snow.”

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

“An image of Malta, covered in Palm trees, highly detailed and realistic”

“Photo of a guy having with a chubby young redhead, POV”

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

“Some luggage sets in the living room ready to go.” “beautiful summer landscape, an ultrafine detailed painting, intricate pasta waves, made of noodles, paper quilling, inspired by van Gogh”

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

“Traditional library with floor-to-ceiling bookcases” “A large hawk flying through a purple and orange sky.”

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

“Navy blue wall livingroom with dusty pink curtains” “a boy in a blue shirt is eating some food”

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

“far away camera shot of an abandoned pickup truck, overgrown”

“A blue plate topped with rice and stew.”

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

“A painting depicting a foothpath at Indian summer with an epic

evening sky at sunset and low thunder clouds.” “A bench sitting on top of a sandy beach next to the ocean.”

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

“A watercolor portrait of a woman by Luke Rueda Studios and David Downton.”

“a man that is on a surfboard on some water”

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

“A painting depicting a snowy winter scene featuring a river, a

small house on a hill, and a dreamy cloudy sky.” “An orange colored sandwich.”

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

“A lighted birthday cake with chunks of walnuts.” “A yellow and black bus cruising through the rainforest.”

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

“logo of a blue elephant, flat modern vector icon” “A beautiful ancient Chinese chivalrous woman”

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

“Alight blue haired anime girl ocean themed anime is opon antenna twintails”

“A sail boat entering a majestic fjord landscape in winter”

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

“<pixel art> gray French bulldog” “An oil painting of a bowl of fruit”

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

“A silhouette of a dog looking at the stars” “portrait of sir borzoi dog wearing royal uniform

and crown”

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

“A snowy Chicago street during Christmas art by

Ludwig Fahrenkrog” “a low light photo of a city at night”

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

“Gothic cathedral in a stormy night” “Face shaped out of old rusty technology”

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

“Photo of a black panther” “An stylized entrance to a rocky cave”

(ours) Refined noise

(ours) Refined noise w/o guidance

Random noise w/ guidance

Random noise w/ guidance

Random noise w/o guidance

Random noise w/o guidance

w/o guidance

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

“A black vase on display with lights in the background.” “A feast of meat, potatos, and veggies on a plate”

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

“A red fire hydrant on the side of a street.” “photograph of a man when he was much younger”

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

“A subway train next to the boarding platform.” “A white sandy beach has a chair and straw umbrella.”

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

“A woman holding a plate with a slice of cake.” “A cute kitten hiding in something on a chair.”

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

“A large boat sitting in the middle of a body of water.” “A grey motorcycle on dirt road next to a building.”

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

“A bird perched on a branch by some leaves.” “A mirror image of a bathroom and a scenic view

from a window.”

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

“A building has a gold clock inside of it.” “A white Volkswagen beetle parked on a lush grass field.”

### D. Implementation and Experimental Details

##### D.1. Implementation Details

|Initial noise 𝑥𝑇~𝑁(0,𝐼)|
|---|

|Generated image w/ Guidance 𝑥0Guide|
|---|

|Text prompt 𝑐|
|---|

[Figure 287]

[Figure 288]

|× 𝑁′|
|---|

[Figure 289]

𝜃

Guidance T2I Pipeline

| | |
|---|---|
|Distan|ce 𝒅|
| | |

|Text prompt 𝑐 (𝑃1)| | |
|---|---|---|
| |Noise Refining| |

[Figure 290]

LoRA

[Figure 291]

𝜃

residual connection

###### NoiseRefiner 𝑔𝜙

Text prompt 𝑐 (𝑃2)

[Figure 292]

[Figure 293]

× 𝑁

[Figure 294]

Stop gradient

|𝜀0 𝑥𝑡|
|---|

| | |
|---|---|
|?|?|
| | |

| | | |
|---|---|---|
| |𝑥ො𝑡−1| |
| | | |

|𝑥ො𝑡|
|---|

Backpropagation

|Guidance-free noise 𝑥ො𝑇|
|---|

|Generated image w/o Guidance 𝑥ො0|
|---|

Guidance-Free T2I Pipeline

- Figure 19. Training framework. We provide an annotated illustration of the training framework to clarify the notation in the following discussion.

More details of our framework. We can generalize our framework NoiseRefine from pixel-level diffusion models to latentlevel diffusion models, but in our experiments, we use MSE loss in latent space for d(xGuide0 ,xˆ0). We provide our training framework in Fig. 19. It consists of three parts: Guidance T2I Pipeline takes Gaussian noise xT ∼ N(0,I) and condition (text prompt) c as inputs and generates an image xGuide0 with guidance methods [1, 13, 15, 16, 21, 38]. noise refining model gϕ refines Gaussian noise xT. Guidance-Free T2I Pipeline takes refined noise xˆT = gϕ(xT) and condition (text prompt) c and generates an image xˆ0 without guidance. For Guidance T2I Pipeline, with the denoising network ϵθ, we can use the guided score ϵCFGθ (xt,c) for CFG [13] or ϵPAGθ (xt,c) for PAG [1] in denoising process as below:

ϵCFGθ (xt,c) = ϵθ(xt,c) + w(ϵθ(xt,c) − ϵθ(xt,∅)), (26) ϵPAGθ (xt,c) = ϵθ(xt,c) + s(ϵθ(xt,c) − ϵˆθ(xt,c)), (27)

ϵCFG,PAGθ (xt,c) = ϵθ(xt,c) + w(ϵθ(xt,c) − ϵθ(xt,∅)) + s(ϵθ(xt,c) − ϵˆθ(xt,c)), (28) where w and s denote the guidance scale of CFG [13] and PAG [1], c denotes the condition, and ∅ denotes the null condition (i.e., empty prompt). Note that the perturbed score ϵˆθ is from perturbing the forward process of the denoising network ϵθ [1]. With the denoising step N′ = 20, we can get the guided image xGuide0 . Our noise refining model refines Gaussian noise xT with gϕ at timestep t = T, which is from the reverse step of DDIM [46] in Eq. (15). The output of noise refining model gϕ is denoted as xˆT = gϕ(xT) and becomes the input of Guidance-Free T2I Pipeline. In this pipeline, xˆT is denoised into xˆ0 without guidance using N denoising steps.

Model details. For noise refining model gϕ, we use Stable Diffusion 2.1 [36] with LoRA [17] rank of 128, applied to all attention, convolutional, and feed-forward layers. We use DDIM [46] scheduler with the same settings as the pre-trained model. For noise refinement, we use an input timestep T = 999, and the default denoising step N is set to 10.

##### D.2. Experimental Details

Training details. The training dataset consists of two parts: 20K images generated with CFG [13] (guidance scale 7.5) using prompts from MS COCO [25], and 30K images generated with both CFG [1] (guidance scale 3.0) and PAG [1] (guidance scale 2.0) using prompts from Pick-a-pic [23]. Only images scoring above 6.0 on the LAION Aesthetics Predictor V2 [43] are selected for training. Furthermore, for training, we use 8 A100 GPUs of 40GB vRAM with a batch size of 4 and sample the images for evaluation using weights of 39k training steps.

Evaluation details. For sampling images with guidance in Tab. 2, 40% of 30k images are generated with CFG [13] (w = 7.5) and remaining 60% are generated with both CFG [13] (uniformly from w ∈ [3.0,5.0]) and PAG [1] (uniformly from s ∈ [2.0,3.0]). For Tab. 1, we use 200 prompts from Drawbench [39], 400 prompts from HPDv2 [48], and 500 prompts from test set of Pick-a-pic [23] for generating 5 images per prompt. Here, for MS-COCO [25], we use 5k generated images selected from those used in Tab. 2. The qualitative results for Fig. 8 are from the images used in Tab. 1 and 2. Additionally, Inference time in Tab 2 is computed by averaging time per image across 30K images generated with the inference step of 20 and a batch size of 1 on RTX 3090.

Ablation study settings. For the ablation study on the number of denoising steps, we use the training dataset which consists of MS COCO [25] and Pick-a-pic [23] used in training noise refining model. The models are trained in two V100 GPUs for 100K steps. In the case of training dataset filtering, we only use MS-COCO [25] dataset, and the models are trained in two RTX 3090 GPUs for 25K steps. For the unfiltered case, entire 80K images are used. For the filtered case, we use the same filtering criteria detailed in D.1, resulting in 20K images. All the other training details are kept consistent.

### E. Discussion

In this section, we compare the performance between training the noise refining model gϕ and the denoising network ϵθ in the denoising process without guidance (Sec. E.1). In addition, we present our hypothesis on why refined noise eliminates the need for guidance methods, explaining it step by step (Sec. E.2). We further analyze the impact of initial noise and prompt on the generated image (Sec. E.3).

##### E.1. Effectiveness of Prompt Learning

As shown in the training framework of our method (Fig. 19), the noise refining model can be trained using the loss d(xGuide0 ,xˆ0), but the denoising network ϵθ within the Guidance-Free T2I pipeline can also be trained. Instead of directly training the model itself (e.g., fine-tuning models like CLIP [35]), our method demonstrates the efficiency of learning the noise input to the Guidance-Free T2I pipeline, akin to prompt learning, which optimizes prompts instead of models. Specifically, similar to conditional prompt learning such as CoCoOp [51], noise prompts are conditionally generated based on different inputs (Gaussian noise xT and text prompt c).

By leveraging the knowledge of pretrained denoising networks, noise prompts can be generated efficiently, as verified in Sec. B.2. Here, we examine the case of training Guidance-Free T2I pipeline itself. Specifically, following the settings of ablation study on the number of denoising steps (Tab. 3) where we use filtered MS COCO [25] dataset, we compare the performance of models trained using our learning method (Fig. 19) with models where ϵθ is fine-tuned on xT as input without using the noise refining model. Instead of directly fine-tuning ϵθ, we train a LoRA [17] module with the same rank and layers

- as gϕ (used in the noise refining model). The results in Fig. 11 clearly show that training the noise prompt leads to significantly faster convergence and higher-

quality outputs at the same training steps. In the figure, the first row represents the outputs from models trained with the noise refining model ϵθ, while the second row shows outputs from models where ϵθ was fine-tuned using LoRA [17].

##### E.2. Why does refined noise help denoising?

To identify which refined noise components contribute to guidance-free generation, we first decompose the refined noise into multiple frequency components. In this study, we utilize a two-dimensional Fourier transform to break down both the refined noise and the initial noise into their respective frequency components. Each frequency component is represented by a frequency band, denoted as (a,b), which corresponds to the frequency range from a to b. Note that although we explored other decomposition methods, such as dividing the noise into patches, they did not yield interpretable results.

(a) denoised images according to the cutoff band

[Figure 295]

(0, 0.1) (0.1, 0.2) … (0.9, 1)

(b) denoised images according to the cutoff radius

[Figure 296]

R = 0 0.01 … 0.1

- Figure 20. Visualization of denoised images according to the cutoff band. Both refined and initial noise were transformed into the frequency domain using Fourier transforms. The frequency domain of the initial noise, normalized such that the maximum radius is 1. (a) The frequency divided into intervals of 0.1. For each interval, the corresponding frequency components were replaced with those from the refined noise, followed by denoising. The results show that only when the (0, 0.1) frequency band was replaced does an image generated by the refined noise emerge. (b) Visualization of denoised images by incrementally increasing the cutoff radius R from 0 in steps of 0.01 and replacing the corresponding components of the initial noise with refined noise. The results demonstrate that images denoised using refined noise are obtained starting at a cutoff radius of 0.03.

- Figure 21. Visualization of the norm based on the frequency-filtered radius R of refined noise. This visualization demonstrates the increase in the norm as the cutoff radius R in the frequency domain is expanded. The refined noise was transformed into the frequency domain using a Fourier transform, and the norm corresponding to each cutoff radius was calculated and plotted.

Low-frequency components matter. Using 2D Fourier transforms, we transform both refined and initial noise into the frequency domain. The initial and refined noise frequency domain is normalized into (0,1). We synthesize a new noise signal by replacing specific frequency bands of the initial noise with the corresponding bands from the refined noise. Fig. 20 (a) presents the generated images corresponding to different frequency bands, demonstrating that the low-frequency components of the refined noise predominantly influence the generation process. In Fig. 20 (b), images are generated by varying the band length within the low-frequency region. The results indicate that, despite the low magnitude of the low-frequency components, which can be confirmed through Fig. 21, they are sufficient to reconstruct the image effectively.

[Figure 297]

LowfreqHighfreq

(0, 1) (0.1, 1) (0.2, 1) (0.3, 1) (0.4, 1) (0.5, 1) (0.6, 1) (0.7, 1) (0.8, 1) (0.9, 1) (1, 1)

(0,0) (0, 0.1) (0, 0.2) (0, 0.3) (0, 0.4) (0, 0.5) (0, 0.6) (0, 0.7) (0, 0.8) (0, 0.9) (0, 1.0)

Frequency band to keep, otherwise zero

- Figure 22. Denoised images using only low(top) / high(bottom) frequency components. Diffusion models can generate the overall structure of the image using only the low-frequency bands of the refined noise. We use DDIM [46] with 20 steps for denoising without CFG, and the prompt was “a photo of a corgi”.

Diffusion models can generate images using only low-frequency components. In Fig. 22, we examine how well diffusion models can denoise when specific frequency bands of refined noise are retained, and the values of the remaining bands are set to zero (using ideal high/low pass filters). The top row shows the results of applying a 2D Fourier transform to the refined noise, normalizing the FFT frequency domain into (0,1), and sequentially retaining lower frequency bands, such as (0,0),(0,0.1),(0,0.2),...,(0,1), while setting the remaining bands to zero. These noise inputs are then denoised without CFG [13]. The figure demonstrates that the diffusion model begins forming a recognizable corgi shape even when only the lower 50% of frequency bands of the refined noise are present. In contrast, noise containing only high-frequency bands fails to generate coherent images.

Frequency band to keep, otherwise reinitialize

(0,0) (0, 0.1) (0, 0.2) (0, 0.3) (0, 0.4) (0, 0.5) (0, 0.6) (0, 0.7) (0, 0.8) (0, 0.9) (0, 1.0)

[Figure 298]

LowfreqHighfreq

(0, 1) (0.1, 1) (0.2, 1) (0.3, 1) (0.4, 1) (0.5, 1) (0.6, 1) (0.7, 1) (0.8, 1) (0.9, 1) (1, 1)

- Figure 23. Denoised images using only low (top) / high (bottom) frequency components with reinitialization. We use DDIM [46] with 20 steps for denoising without CFG, and the prompt was “a photo of a corgi”. High-frequency components contribute details. Here, we use the same noise decomposition process of refined noise as Fig. 22 but following [9], we reinitialize the frequency components that were set to zero with corresponding components from standard Gaussian noise, then denoise again. The results, shown in Fig. 23, indicate that when all frequency components are present, the diffusion model can generate clear and complete images. Randomly reinitialized high-frequency components appear to add details onto the structure formed by the low-frequency components. While refined noise retaining only the lower 10%–20% of frequencies can still reconstruct the original image when the rest is reinitialized, noise retaining only the high-frequency components fails to do so. This suggests that low-frequency components alone carry the significant information needed for image generation.

[Figure 299]

(0,0.1)(0,0.05)(0,0.2)(0,0.3)

Seed 0 Seed 1 Seed 2 …

(a) Low-frequency components of refined noise

[Figure 300]

(0.95,1)(0.9,1)(0.8,1)(0.7,1)

Seed 0 Seed 1 Seed 2 …

(b) High-frequency components of refined noise

- Figure 24. Different denoised images using only low(a) / high(b) frequency components for different seeds. Here we use 8 different seeds. From the top rows, it visualizes 8 images using only the lower (a) / higher (b) 5%, 10%, 20%, and 30% (from the top to the last rows) frequency components of the refined noise.

In Fig. 24, each row visualizes images generated with only the lower 5%, 10%, 20%, and 30% (from the top rows to last rows) frequency components of the refined noise, while the bottom row shows images generated with only the upper 5%, 10%, 20%, and 30% frequency components. These results confirm that low-frequency components encode the overall layout and structure, whereas high-frequency components lack meaningful information.

From these observations, we infer that the poor quality of unguided diffusion model outputs is due to their failure to form appropriate low-frequency components during denoising. High-frequency details added on poorly formed layouts result in artifacts that are perceived as unnatural.

How do guidance methods form plausible initial layouts? As highlighted in [1], classifier-free guidance (CFG) [13] enhances the difference between conditional and unconditional predictions at each step, amplifying “signals that can only be generated with the condition” (e.g., features like the eyes or nose of a corgi in “a photo of a corgi”). This effectively strengthens salient features corresponding to low-frequency components in the early denoising steps. From this, we deduce that guidance methods [1, 13, 16] add appropriate low-frequency components during inference, aiding the formation of high-quality layouts.

Denoising Direction

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

Gaussian w/oguidance

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

Gaussian w/guidance

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

Refined w/oguidance

Noise Delta t = 𝑇 t = 𝑇 − 1 … Denoised

t = 1

- Figure 25. Visualization of 11th layer cross attention map. Token corresponding to ‘cat’ is used for visualization among the prompt ‘a photo of a cat’. First and second row is the case starting from random Gaussian noise where guidance is not used for the first row and used for the second row. Third row is the starting from refined noise by the noise refining model. ‘Noise Delta’ means the difference between

initial Gaussian noise xT and refined noise xˆT. When guidance is not used, failure to create meaningful attention map across all timestep is notable, leading to completely broken generation. However when guidance or our refined noise is used, meaningful cross attention map is observed, leading to successful generation. Notably thanks to noise delta, a better aligned cross attention map is observed even in earlier step (t = T) when the refined noise is used.

How does the noise refining model form low-frequency layouts? Interestingly, the noise refining model naturally forms low-frequency layouts even though our training framework does not explicitly enforce learning them as can be seen in Fig. 7. To understand this, we analyze cross-attention and self-attention maps across denoising steps. Fig. 25 visualizes these maps at different timesteps. Gaussian noise fails to form meaningful cross-attention maps in early steps due to its near-zero signal-tonoise ratio (SNR), which is expected. However, this failure persists in later steps, indicating an inability to form well-aligned layouts (Fig. 25 first row).

Research [2, 10, 28] has shown that reducing noisy artifacts in cross-attention maps and aligning them with object regions during inference improves performance. This suggests that the failure of cross-attention maps to align is a key reason for the diffusion model’s inability to create coherent layouts. When using CFG [13] (second row) or refined noise (third row), the cross-attention maps align well with the prompt, resulting in better outputs. Notably, cross-attention maps for refined noise exhibit accurate object shapes from the very first step, implying that the diffusion model can form plausible layouts from the beginning of the denoising process. This is further supported by Fig. 10, which visualizes x0 predictions at each denoising step.

Implications for guidance-free generation. Without guidance methods or noise refiners aiding the formation of lowfrequency layouts, diffusion models fail to create plausible initial layouts. Random low-frequency components lead to artifacts that are perceived as unnatural. An interesting avenue for future research would be identifying why diffusion models struggle to form low-frequency components without guidance and developing training techniques to eliminate the need for guidance during the training stage.

##### E.3. Impact of Initial Noise and Prompt on Generated Image

Noise refining prompt 𝑃 = “a photo of a lion in the wild”

Denoising Direction

𝑡 = 𝑇 𝑇 − 1 … 𝑡 = 0

[Figure 363]

- (a) Gaussian noise, 𝑃 = “a photo of a lion in the wild”
- (b) Refined noise, 𝑃 = “a photo of a lion in the wild”

[Figure 364]

[Figure 365]

[Figure 366]

- (d) Refined noise, 𝑃 = “a photo of a tiger in the wild”

(c) Refined noise, 𝑃 = “”

- (e) Refined noise, 𝑃 = “a laptop computer on a desk”

[Figure 367]

[Figure 368]

(e) Refined noise, 𝑃 = “a laptop computer on a desk”, with CFG

Figure 26. Visualization of denoised image using different prompt for noise refinement ϵθ and denoising gϕ.

We previously demonstrated how refined noise affects initial layouts and how guidance and refined noise contribute to forming these layouts effectively. In this section, we investigate how the ‘layout’ and the prompt influence the final generated image during the denoising process. Specifically, we explore what happens when the prompt used to generate the initial layout (P1, one of the inputs to the noise refining model gϕ) differs from the prompt used during denoising (P2, one of the inputs to the denoising network ϵθ in the Guidance-Free T2I Pipeline shown in Fig. 19). Does the model prioritize one prompt over the other? Or does it attempt to harmonize both? We investigate this question through the results shown in Fig. 26.

- • Fig. 26 (a) visualizes the predicted x0 term in Eq. 15 during the denoising process when no layout is provided (starting

- from Gaussian noise). The leftmost image corresponds to the predicted x0 at t = T, and subsequent images are visualized every three steps. Due to the noisy and ambiguous nature of the initial layout of Gaussian noise, the diffusion model fails to form a coherent lion layout from the initial structure. Instead, it partially adds features such as fur, mane, nose, or mouth, resulting in poor perceptual quality.
- • In contrast, (b) shows that in the case of P1 = P2, refined noise effectively forms the lion layout from the beginning. The diffusion model accurately places the overall lion shape, including its mane, eyes, nose, and mouth, in appropriate positions during the denoising process.
- • (c) shows the results when the denoising prompt P2 is set to an empty prompt (null prompt). Despite this, the model successfully generates a feline animal based solely on unconditional generation, as the layout sufficiently captures the overall structure of the object. This can be interpreted as the information embedded in the refined noise.
- • (d) demonstrates the case where the denoising prompt P2 is set to a prompt similar to the initial layout prompt (“a photo of a tiger in the wild”). When a similar prompt is used, the image retains the layout provided by the refined noise while also adhering to the prompt.
- • In (e), P2 is set to an entirely independent prompt (“a laptop computer on a desk”). Here, the model fails to generate a coherent image corresponding to the layout or the prompt. The diffusion model attempts to form a laptop on the existing lion or feline layout but fails to align with the laptop prompt, leading to failure.
- • Finally, (f) shows that applying CFG [13] in the settings of (e) allows the diffusion model to disregard the initial layout and generate a laptop. This partially explains why CFG consistently produces high-quality images. Randomly generated initial noise is unlikely to align with the prompt (as shown in (a)), and CFG helps the model ignore such initial noise and generate images consistent with the given prompt.

𝑎 = 0.0 𝑎 = 0.2 𝑎 = 0.4 𝑎 = 0.6 𝑎 = 0.8 𝑎 = 1.0

[Figure 369]

(a) Interpolated Gaussian noise

[Figure 370]

(b) Refined noise

Figure 27. Images from interpolated refined Gaussian noise.

Interpolation between refined noise. To evaluate whether the noise refining model effectively learns noise mapping, we follow [46, 47] to perform spherical interpolation on initial noise samples, generating multiple interpolated noises. We then refine each interpolated noise using the noise refining model and verify that the refined noises effectively interpolate natural images. In Fig. 27, (a) shows the images denoised by the diffusion model without any guidance method, starting from spherical interpolations of two random Gaussian noises. Specifically, each interpolated noise is obtained by performing slerp(xT

, xT

, a) for various interpolation ratios a, where slerp performs spherical interpolation between two Gaussian noise

1

2

- at a ratio of a. Fig. 27 (b) shows the results of denoising the refined versions of these interpolated noises without guidance. The results

demonstrate that the refined noises effectively interpolate between the two images. This indicates that the noise refining model does not simply memorize specific low-frequency signals while ignoring the input noise. Instead, it effectively learns a mapping from a Gaussian noise space to a guidance-free noise space where semantic interpolation between guidance-free images is possible.

##### E.4. Comparison with a related work

A recent study [22] exists under the category of noise manipulation. To the best of our knowledge, this work is unique in its focus on learning the noise space itself, rather than optimizing or selecting. Therefore, we compare our proposed approach

with this methodology PAHI (Prompt Adaptive Human preference Inversion) [22] in this section.

There are several key differences between the two approaches. First, the tasks being addressed are distinct. While PAHI [22] aims at generating outputs aligned with human preferences, our objective is to replace conventional guidance mechanisms entirely. Second, our method offers much greater flexibility. PAHI [22] assumes that sampling from certain N(µ,Σ) instead of a standard normal Gaussian distribution is more beneficial and predict µ and Σ. However, this assumption lacks a strong theoretical foundation. In contrast, our approach aims to learn a gaussian-free noise space without imposing such constraints. Additionally, while PAHI [22] is limited to few-step models due to the computational overhead of backpropagation, our approach leverages MSD loss, enabling the use of full-step models without modification.

Although the official code for PAHI [22] is unavailable, we adhere to the guidelines presented in their paper as possible and compare with our method. Specifically, we compare the noise refining model with the setup that samples noise from N(µ,Σ) where µ and Σ is predicted by MLP for a given prompt. Both models are trained with filtered 20K MS COCO[25] dataset for 25K steps using two RTX 3090 GPUs. Example qualitative results of employing MLP are presented in Fig. 28, and quantitative comparisons are shown in Tab.7. Across both evaluations, the noise refining model outperforms the other setup by a significant margin, showing the effectiveness of our proposed method.

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

- Figure 28. Qualitative results when employing a shallow 2-layer MLP for estimating Gaussian parameters, as proposed by [22]. The results are significantly blurry, indicating that the simple approach of predicting µ and Σ under the assumption that the optimal noise lies within N(µ, Σ) performs poorly.

###### Method FID

MLP [22] estimating Gaussian parameters 217.30 Noise refining model (ours) 13.74

Table 7. Quantitative results when employing a shallow 2-layer MLP for estimating Gaussian parameters, as proposed by [22].

##### E.5. Robustness to the number of denoising steps and schedulers

Since the noise refining model is trained with a fixed scheduler (DDIM [46]) and denoising steps (10), concerns arise regarding its performance when using different schedulers or denoising steps. To examine the impact of varying schedulers and denoising steps, we conduct experiments comparing qualitative results across diverse configurations. For comparison, we select DPM++ SDE [26], DPM++ 2M [26], and EDM [20], using the prompt “a photo of a cat”. The results, presented in Fig. 29, show that our refined noise consistently produces reliable outputs regardless of the denoising timestep or scheduler. This demonstrates the robustness of the noise refining model across diverse schedulers and denoising step configurations.

[Figure 376]

DDIM

DPM++ SDE

DPM++ 2M

EDM

(a) 10 Steps

[Figure 377]

DDIM

DPM++ SDE

DPM++ 2M

EDM

(b) 20 Steps

[Figure 378]

DDIM

DPM++ SDE

DPM++ 2M

EDM

(c) 50 Steps

- Figure 29. Inference results on our refined noise in various denoising steps and scheduler settings. (a), (b), and (c) present inference results employing different schedulers at denoising steps of 10, 20, and 50, respectively. The consistency observed across these results highlights the robustness of our refined noise to variations in both denoising steps and schedulers.

