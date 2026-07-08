## Reflection Removal through Efficient Adaptation of Diffusion Transformers

# arXiv:2512.05000v1[cs.CV]4Dec2025

Daniyar Zakarin∗,1,2, Thiemo Wandel∗,2, Anton Obukhov†,2, Dengxin Dai2

1ETH Z¨urich, 2HUAWEI Bayer Lab, ∗Equal contributors, †Project lead

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

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Figure 1. We present WindowSeat, a model and fine-tuning protocol for one-step reflection removal. It repurposes a foundation image diffusion transformer (DiT) into a state-of-the-art computational photography tool, enabled by an efficient and scalable Physically Based Rendering (PBR) pipeline for data synthesis. WindowSeat demonstrates stronger scene understanding and source-separation capabilities than competing methods, yielding cleaner outputs with fewer artifacts. For the visualization above, the “Scenario”, “Ground truth” transmission, and reflection layers were generated; “Photos with reflection” were produced by our proposed PBR pipeline; results images are obtained from the respective methods. Best viewed zoomed in; arrows point at artifacts of methods; contrast enhanced for visualization.

### Abstract

We introduce a diffusion-transformer (DiT) framework for single-image reflection removal that leverages the generalization strengths of foundation diffusion models in the restoration setting. Rather than relying on task-specific architectures, we repurpose a pre-trained DiT-based foundation model by conditioning it on reflection-contaminated inputs and guiding it toward clean transmission layers. We systematically analyze existing reflection removal data sources for diversity, scalability, and photorealism. To address the shortage of suitable data, we construct a physically based rendering (PBR) pipeline in Blender, built around the Principled BSDF, to synthesize realistic glass materials and reflection effects. Efficient LoRA-based adaptation of the foundation model, combined with the proposed synthetic data, achieves state-of-the-art performance on indomain and zero-shot benchmarks. These results demonstrate that pretrained diffusion transformers, when paired with physically grounded data synthesis and efficient adaptation, offer a scalable and high-fidelity solution for reflec-

tion removal. Project page: https://hf.co/spaces/huaweibayerlab/windowseat-reflection-removal-web

### 1. Introduction

Reflections have long been persistent artifacts in photography, especially in casual or on-the-go capture. With most images now taken on mobile devices, and with glass and glossy materials increasingly common in modern architecture, it has become difficult to take a snapshot without including at least one unintended reflection. Therefore, effective singleimage reflection removal (SIRR) is essential for improving the quality of computational photography on mobile devices. At its core, reflection removal is a fundamentally ill-posed source separation problem. Addressing it requires a strong scene understanding prior together with principles rooted in the physics of image formation.

Prior approaches often depended on task-specific architectures with limited pretraining [49], simplified physics-based assumptions such as modeling reflections as low-frequency

components [28], and restricted data sources. These include screen-space simulation pipelines [43], small-scale datasets [27], collections with potential pixel-level misalignment [51], or pseudo-labels refined by manual postprocessing [44]. This often resulted in sparse coverage of the underlying data distribution, leading to poor in-the-wild and out-of-distribution performance.

A natural way forward is to leverage a foundation model that already encodes rich scene understanding and finetune it for the task using only small amounts of in-domain data. However, recent work on fine-tuning foundation models, both within their native domain [47] and far outside it [22, 45], shows that noisy or unrepresentative training data of any nature severely limits their ability to adapt while preserving the benefits of pretraining. In contrast, sufficiently diverse photorealistic synthetic data provides the coverage and controllability needed to fully exploit these pretrained priors and adapt them to the target task.

This is the direction we take in this paper. Accordingly, we outline a practical path from foundation DiTs to strong single-image reflection removal solutions and contribute:

- • a scalable PBR data pipeline that simulates light transport to produce realistic training data from large-scale sources;
- • an efficient protocol for adapting foundation DiTs, covering fine-tuning objectives, reuse of the base model’s inputs and outputs, and practical settings for LoRA and quantization, enabling one-day training on a single consumer GPU and allowing future work to benefit from improved base models through a simple model swap;
- • a comprehensive evaluation of our model (WindowSeat) derived from a recent DiT on SIRR benchmarks and inthe-wild images, demonstrating top performance (Fig. 1).

### 2. Related Work

Taxonomy of reflection removal. To compensate for the missing prior knowledge about the scene, prior work often leverages side-channel information available only in controlled capture setups, such as flash-no-flash pairs [1, 41], polarization [35], stereo baseline [36], or burst and video sequences [32]. In contrast, our work advances the singleimage (SIRR) setup with no special hardware support.

Classical methods. Traditional reflection removal methods rely on statistical priors and simplified mathematical models of the reflection process [10, 11, 25, 26]. Modern end-to-end deep learning approaches surpass these classical ones, but depend on high-quality training data.

Real data. Existing real-world datasets are typically captured by placing a glass plate between the camera and the scene, collecting either single images [27, 39, 48] or videos [15, 51]. However, these datasets are limited in the number of real scenes they can cover and often contain samples without pixel-perfect alignment due to refraction, object, or camera motion (e.g., wind-driven foliage or subtle tripod

shifts). Yang et al. [44] propose a more scalable approach by starting from mobile captures with reflections and generating pseudo-ground truth using an existing reflection removal algorithm followed by manual refinement. While easier to scale, this method is practical only for images requiring minimal manual cleanup and provides limited control over glass properties and the resulting data distribution.

Alpha blending. Due to the limited availability of realworld reflection-removal datasets, many SIRR approaches additionally rely on synthetic data generated by alpha blending, typically created by overlaying images from PASCAL VOC [9] [13, 15–17, 19, 40, 49] or COCO [29] [13, 15, 50]. These methods employ various screen-space mixing models, which lack the physical realism needed to capture true glass behavior.

PBR data. Several methods have explored physical simulation as a source of high-quality training data. Kim et al. [23] reconstruct 3D meshes from RGBD images and simulate light transport between the scene and glass to reproduce anisotropic reflection effects. Meanwhile, [20] uses a carefully curated rendered dataset to learn the removal of both specular highlights and reflections. Our approach extends this direction by providing a more scalable alternative in which the 3D scene remains fixed and only the input images vary, avoiding the need for per-scene geometric reconstruction while still preserving physically faithful light transport. Foundation models. Diffusion-based image generation [38] has gained widespread attention across vision and graphics. Early text-to-image models often relied on auxiliary modules such as ControlNets [46] to enable practical fine-tuning. For cross-modal prediction, Marigold [21, 22] popularized latent concatenation with full-model fine-tuning. LoRA [14], originally introduced as an efficient alternative to full finetuning in transformer-heavy NLP, became the default strategy for adapting diffusion models once they evolved into DiTs, where full fine-tuning is prohibitively expensive. As we show in this paper, modern flow matching [30] imageediting models [2] are sufficiently expressive to maintain high image quality with minimal modification, even under lightweight 4-bit QLoRA adaptation [31].

Foundation models and SIRR. To exploit potential synergies between the transmission and reflection layers, several works predict both modalities simultaneously [16– 19, 49, 50]. Recent diffusion-based approaches further extend SIRR with multi-step [13, 40, 50] or single-step [15] denoising, and some incorporate text encoders to provide semantic priors [13, 40, 50]. However, these methods rely on costly multi-step sampling, auxiliary ControlNets, crosslatent skip connections, or cross-attention for dual-stream prediction, all of which fall outside our goal of a simple and efficient fine-tuning protocol. Instead, we adapt an imageediting diffusion model [2] into a feed-forward reflectionremoval network. The procedure for obtaining WindowSeat

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

[Figure 46]

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

- Figure 2. Physically Based Rendering (PBR) pipeline for synthetic data generation. Left: The synthesis begins by sampling the foreground and background images, which can be in sRGB or HDR formats. The images are placed into a static 3D scene with a glass plate positioned in front of a virtual camera. The camera parameters and object distances are chosen to cover the view frustum of the virtual camera along transmission and reflection paths. Middle: At the heart of our pipeline is the Principled BSDF shading model [3, 4], implemented in Blender [6], which enables simulation of a wide range of photorealistic glass effects and light interactions. Right: Visualizations of three factors of variation. Index of Refraction (IoR) affects reflection strength. Thickness increases ghosting, which appears as larger gap between the multiple reflections (arrows). Roughness controls the degree of scatter and blur. Such a simulation cannot be faithfully reproduced by screen-space alpha blending models. Details in Sec. 3.1 and 3.2. Best viewed zoomed in.

remains straightforward, and the resulting model outperforms prior work across the board.

### 3. Method

#### 3.1. Recap: Alpha Blending

A central obstacle in developing robust reflection-removal systems is the acquisition of training data. Capturing realworld transmission and reflection-contaminated image pairs is difficult due to refractive pixel shifts, inconsistent illumination, and scene or camera motion. Consequently, most existing methods predominantly rely on synthetic data, typically generated using some form of a linear blending of two images to simulate glass reflections over natural scenes. Recent state of the art methods [5, 15, 18, 49] adopt the formulation introduced in [17]:

B = αT + βR − αβT ◦ R,

where B denotes the blended output, and T and R represent the transmission and reflection images, respectively. We refer to this approach as alpha blending.

Although simple and computationally efficient, this approach fails to capture the underlying optical phenomena responsible for realistic reflections, such as subsurface scattering, which produces characteristic blurring, and multiple internal reflections, which give rise to ghosting artifacts in real glass. In practice, Gaussian blurring is often used to approximate the scattering, which provides limited realism. Moreover, alpha blending operates in the standard RGB (sRGB) color space and cannot produce high-intensity specular highlights on glass surfaces.

#### 3.2. Physically Based Rendering Pipeline

To address these limitations of alpha blending, we propose a light-weight physically based rendering (PBR) pipeline that explicitly simulates the light-glass interaction (Fig. 2). We achieve this by capturing a transmission scene with simulated reflections from a glass surface, where the reflection source is either a panoramic high-dynamic-range (HDR) environment map or a planar RGB image. HDR panoramic images are widely used in 3D modeling, and in our case they help us generate light interaction in a wider light spectrum, generating strong effects like hazy scattering or strong highlights. The panoramic format enables modeling of light from multiple incident directions. In contrast, the wide availability of RGB images as planar reflection sources, such as COCO [29] and Pascal VOC [9], lets us generate a rich family of reflection patterns.

For each training sample, the virtual camera is positioned to observe a scene through a glass material. For efficiency, each scene is represented as a textured mesh plane, which greatly accelerates rendering while preserving spatial detail. Unlike sRGB images, HDR sources encode radiance rather than display intensities, enabling more physically accurate reflection synthesis and internal light interactions.

To model the glass material, we use Blender’s Principled BSDF shader [6] (Fig. 2), a physically based shading model derived from Disney’s BRDF [3, 4]. This formulation provides a compact parameterization of the optical properties relevant to glass. The index of refraction (IoR) and metallic parameters modulate the reflection strength, while the surface roughness controls the microfacet distribution that determines the blur of specular reflections. Light attenua-

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

- Figure 3. Model architecture. Foundation DiTs [37] operate in a compressed latent space in the bottleneck of a VAE [24]. Finetuning DiTs can be done efficiently with lightweight LoRA [14] adapters. Modern DiTs [2] with more than 10B parameters often employ quantized representations, such as QLoRA [8, 31]. The end-to-end fine-tuning procedure is elaborated in Sec. 3.3.

tion through the medium is simulated by assigning a base color to the material, and ghosting is produced by varying the glass thickness. Reflection-free ground-truth images are rendered by setting IoR to 1.0 and zeroing the metallic and roughness terms. By carefully controlling the glass thickness and the camera–glass spacing, we avoid refractive pixel shifts and maintain pixel alignment between the reflectioncontaminated and transmission images.

These controls enable systematic ablation studies (Sec. 4.3) and flexible synthesis of training data.

#### 3.3. One-Step Flow Matching

We perform reflection removal in the latent space of a frozen variational autoencoder (VAE). Given a reflectioncontaminated input image B and clean target T, we encode zB = E(B) with encoder E and decode with the frozen decoder D (see Fig. 3). Our DiT backbone was pre-trained with a flow matching objective and processes two latent token streams along with a text prompt. Concretely, it operates on two sequences of latent tokens of identical spatial layout: a primary stream containing the encoded input latent zB and an auxiliary second stream. In pre-training, the second stream receives a stochastically perturbed ground truth image latent, whereas in our setup we duplicate the latent, i.e. z˜B = zB (see ablation in Sec. 4.3). The text embedding is precomputed in advance.

Let vθ(·;p) be the DiT-predicted latent velocity under prompt conditioning p. Our one-step update produces a reflection-free latent

zˆedit = zB + vθ(zB ;p), Yˆ = D(zˆedit),

replacing multi-step diffusion sampling with a single forward pass while leveraging the DiT pre-training. Unlike prior diffusion pipelines, we do not use ControlNet modules, crosslatent skip connections, losses in latent space, or any VAE fine-tuning. We train the DiT LoRA adapters in pixel space with PSNR and SSIM losses between Yˆ and T:

L = λPSNR LPSNR(Yˆ ,T) + λSSIM LSSIM(Yˆ ,T).

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Figure 4. Ablations. Left: PBR vs Alpha Blending; Right: Latent vs Flow objectives (Sec. 4.3). As seen, PBR data and Flow objective produce more accurate results. Best viewed zoomed in.

### 4. Experiments

#### 4.1. Experimental Setup

Datasets. Following previous work [5, 13, 15, 18, 19, 49, 51], we include the train split of Nature [27] and Real [48] in our training data and evaluate on the corresponding indomain test splits. The SIR2 [39] benchmark is evaluated in a zero-shot manner and contains three sub-datasets: Objects (200), Postcard (199), and Wild (54). SIR2 (500) contains 56 additional wild-scene samples.

Evaluation details. We compare the performance of WindowSeat against the officially reported PSNR and SSIM metrics from other SotA methods in Tab. 1. Additionally, we re-evaluate available SotA methods [15, 17, 18, 49] with the best checkpoint available and compare MS-SSIM and LPIPS values in Tab. 2. For re-evaluation, all predictions are saved to disk and evaluated in 8-bit precision. SSIM and PSNR are computed with skimage, MS-SSIM with pytorch-msssim, and LPIPS with the lpips library.

PBR data. We resort to the procedure described in Sec. 3.2 and Fig. 2 to produce high-quality data. Specifically, we sourced 924 HDR panoramic images from the public domain [12], featuring diverse urban and nature settings. The pool of sRGB images is formed from the COCO dataset [9]. For each PBR sample, the foreground is taken from the sRGB pool, and the background is sampled uniformly from both pools. The dataset contains 25,000 rendered images in total, exhibiting varying range in reflection strengths and form. More specifically, we sample IoR values between 1.25 and 1.75, roughness between 0 and 0.05, and glass thickness between 0 and 5cm.

Implementation details. The LoRA adapter is trained for 11k steps on a single consumer GPU. AdamW [33] is initialized with a learning rate of 10−5, following 100 warm-up steps to 10−4 and a linear decay of 5 × 10−6 every thousand steps. We use batch size 2 and enable gradient checkpointing for the VAE and the DiT. We set λPSNR = 0.1 and λSSIM = 20 and apply global gradient-norm clipping before each optimizer step. We use a LoRA adapter of rank

- Table 1. Quantitative comparison on in-domain (Nature [27], Real [48]) and zero-shot (SIR2 [39]) datasets. Values are sourced from the respective papers, unpublished values are denoted with ‘–’. SIR2 (454) is the weighted average of Objects (200), Postcard (199), and Wild

(55). Best viewed on screen and zoomed in.

Method

Nature [27] (20) Real [48] (20) Objects [39] (200) Postcard [39] (199) Wild [39] (55) SIR2 [39] (454) SIR2 [39] (500) PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑

DSRNet (extra data) [17] (ICCV 2023) – – 23.91 0.818 26.74 0.920 24.83 0.911 26.11 0.906 25.83 0.914 – – DSRNet (w/o extra data) [17] (ICCV 2023) – – 24.23 0.820 26.28 0.914 24.56 0.908 25.68 0.896 25.45 0.909 – – RRW [51] (CVPR 2024) 25.96 0.843 23.82 0.817 – – – – – – 25.45 0.910 – – L-DiffER [13] (ECCV 2024) 23.95 0.831 23.77 0.821 – – – – – – – – 25.18 0.911

- DSIT (data setting I) [18] (NeurIPS 2024) – – 25.06 0.836 26.81 0.919 25.63 0.924 27.06 0.910 26.32 0.920 – –
- DSIT (data setting II) [18] (NeurIPS 2024) 26.77 0.847 25.22 0.836 27.27 0.932 25.58 0.922 27.40 0.918 26.54 0.926 – – RDNet w/o nature [49] (CVPR 2025) – – 24.43 0.835 25.76 0.905 25.95 0.920 27.20 0.910 26.02 0.912 – – RDNet w nature [49] (CVPR 2025) – – 25.58 0.846 26.78 0.921 26.33 0.922 27.70 0.915 26.69 0.921 – – F2T2-HiT [5] (ICIP 2025) 26.08 0.837 21.64 0.766 – – – – – – 25.72 0.903 – – Huang et al. [19] (arXiv 2025) 27.03 0.853 25.12 0.828 27.07 0.930 26.43 0.931 27.96 0.922 26.90 0.929 – – DAI [15] (arXiv 2025) 26.81 0.843 25.21 0.841 – – – – – – – – 27.19 0.930

WindowSeat (ours) 27.12 0.849 26.28 0.856 28.81 0.944 29.17 0.934 28.97 0.935 28.99 0.939 28.75 0.940 WindowSeat (ours, Apache 2.0) 27.57 0.855 26.60 0.864 28.85 0.938 28.70 0.933 29.44 0.936 28.84 0.936 28.60 0.937

- Table 2. Quantitative comparison with MS-SSIM↑ and LPIPS↓ evaluation metrics. Values are re-evaluated by saving all predictions as images on disk and computing metrics in 8-bit precision. Nature (20) and Real (20) are in-domain, while Objects (200), Postcard (199), and Wild (55) are evaluated in a zero-shot manner.

Method

Nature (20) Real (20) Objects (200) Postcard (199) Wild (55) MS-SSIM↑ LPIPS↓ MS-SSIM↑ LPIPS↓ MS-SSIM↑ LPIPS↓ MS-SSIM↑ LPIPS↓ MS-SSIM↑ LPIPS↓

DSRNet [17] (ICCV 2023) 0.9144 0.1478 0.8737 0.1831 0.9564 0.0847 0.9263 0.1260 0.9338 0.1096 DAI [15] (arXiv 2025) 0.9309 0.2161 0.9045 0.1790 0.9638 0.0689 0.9567 0.1029 0.9423 0.0941 RDNet [49] (CVPR 2025) 0.9231 0.1361 0.9081 0.1442 0.9609 0.0836 0.9361 0.1121 0.9406 0.0992 DSIT [18] (NeurIPS 2024) 0.9223 0.1598 0.8934 0.1618 0.9586 0.0939 0.9441 0.1242 0.9447 0.0967

WindowSeat (ours) 0.9435 0.1368 0.9296 0.1131 0.9759 0.0470 0.9693 0.0504 0.9625 0.0632 WindowSeat (ours, Apache 2.0) 0.9494 0.1355 0.9396 0.1074 0.9661 0.0550 0.9664 0.0549 0.9655 0.0682

- Table 3. Parameter count per component. We train 3.6% of all parameters and quantize 95.7% of them to 4-bit. The text encoder is not used at inference. The total model size is ∼12.5B parameters.

to 4-bit. We also quantize the first and second moments of AdamW [33] using the bitsandbytes [7] library and use PEFT [34] to attach bfloat16 LoRA [14] adapters, which we then train with the AdamW optimizer in a memory-efficient fine-tuning setup. The loss computations are performed in float32. Training with batch size 2 and training resolution of 608 has a peak GPU memory consumption of 21 GB.

Component Total Params (K) 16-bit (K) 4-bit (K) Trainable (K) VAE Encoder 34,274 34,274 0 0 DiT (w/o LoRA) 11,901,408 396 11,901,012 0 LoRA Adapter 450,249 450,249 0 450,249 VAE Decoder 49,545 49,545 0 0 Total 12,435,477 534,465 11,901,012 450,249

Resolution strategy. To preserve aspect ratio and match the training setup, we split the image into overlapping tiles and resize each tile to the training resolution (608 × 608) using Lanczos resampling. If the input resolution is lower than the training resolution, we first upsample the shorter side of the input image and then apply tiling. In regions where tiles overlap, the final prediction is computed as a linear combination. Multiple tiles of an image are stacked and then processed in parallel.

128 with Gaussian initialization. During training, we apply random cropping and color jitter, which includes brightness, contrast, saturation, and hue augmentations.

Model quantization. A detailed overview of the number of trainable, frozen, and quantized parameters is provided in Tab. 3. To fine-tune a 12.5B-parameter model on a consumer GPU within one day, we follow the quantization scheme of [31] and adapt it to our DiT-based architecture [2]. The VAE, as well as the first and last layers of the DiT, are kept in bfloat16, while 95.7% of the total parameters are quantized

Apache 2.0 model. The open-source model we provide (ours, Apache 2.0) in Tab. 1, 2 is based on Qwen Image-Edit2509 [42]. It was trained with a batch size 1 and a processing resolution of 768×768.

- Table 4. Ablation on DiT input choices. Mode indicates the output parameterization: FLOW predicts a latent-space velocity vθ that is added to the image latent zB = E(B), whereas LATENT predicts the edited latent zˆedit directly. z1 and z2 denote the first and second latent token sets fed to the DiT; N(0, I) is Gaussian noise and “—” indicates that no second latent is used.

Mode z1 z2

Nature (20) Real (20) Postcard (199) Objects (200) Wild (55) PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑

FLOW E(B) — 26.18 0.816 24.68 0.801 26.06 0.863 26.47 0.890 27.10 0.889 FLOW E(B) N(0,I) 27.43 0.849 26.19 0.861 28.58 0.929 28.46 0.937 28.98 0.931

LATENT E(B) E(B) 26.58 0.837 25.43 0.824 28.18 0.924 27.89 0.926 27.48 0.908 FLOW E(B) E(B) 27.12 0.849 26.28 0.856 28.81 0.944 29.17 0.934 28.97 0.935

- Table 5. Ablation on loss terms evaluated on Nature (20), Real (20), and SIR2 sub-datasets (Postcard (199), Objects (200), Wild (55)).

Nature (20) Real (20) Objects (200) Postcard (199) Wild (55) PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑

LSSIM LPSNR

✗ ✓ 27.65 0.847 26.40 0.850 28.75 0.928 28.87 0.940 28.89 0.934 ✓ ✗ 23.16 0.805 22.90 0.785 28.31 0.930 23.08 0.872 21.62 0.830

✓ ✓ 27.12 0.849 26.28 0.856 28.81 0.944 29.17 0.934 28.97 0.935

#### 4.2. Comparison with the State of the Art

We compare our method against prior state-of-the-art approaches in Tab. 1 and report PSNR and SSIM values as provided in the respective papers; missing values are marked with “—”. Our method achieves a 1.56 dB gain in PSNR and raises the SSIM score from 0.930 to 0.940 on the zero-shot SIR2 (500) dataset. On the in-domain Real dataset [48] we achieve a PSNR gain of 1.06 dB and an SSIM improvement from 0.846 to 0.856. On Nature [27], our method is on par with [19], which can be explained by slight pixel-shifts present between input and GT. Additionally, we compare MS-SSIM and LPIPS with the best open-source checkpoints from DAI [15], RDNet [49], DSIT [18], and DSRNet [17] in Tab. 2. Quantitative MSSSIM and LPIPS results further confirm that WindowSeat achieves perceptually better reflection removal. We qualitatively evaluate WindowSeat in Fig. 5. Our method identifies reflections more reliably and removes both simple and complex reflections with fewer artifacts than competing approaches. In regions with very strong reflections, WindowSeat further exhibits strong inpainting capabilities, inherited from the base DiT.

#### 4.3. Ablation Study

We ablate the network architecture, loss functions, and training data to quantify the contribution of each design choice. Architecture. The base DiT image editing model stacks two latent inputs along the token dimension, namely the encoded image latent zB = E(B) and noise. Since the GPU requirements grow quadratically with token length, we

experiment whether and which second token set is best for one-step reflection removal. In Tab. 4, we show the results of re-training three models with different latent inputs and modes, revealing the following insights.

Passing two latents to the network instead of only E(B) increases the peak GPU memory consumption from 19GB to 21GB. However, it significantly improves the performance compared to omitting the second token set, up to almost 3 dB in the Postcard test set. Using N(0,I) as the second latent slightly increases PSNR in Nature, but overall performs worse than simply duplicating the RGB latent E(B).

Third, we ablate the output parameterization of the DiT. In the flow variant, the model predicts a latent-space velocity (change in latent space) vθ. In the latent variant, the model directly predicts the reflection-free latent zˆedit. Predicting the latent-space velocity vθ rather than the absolute latent zˆedit consistently yields better inpainting, an example of which is shown in Fig. 4 (right).

Loss. We use a PSNR-based loss to reduce global pixel-wise reconstruction error and an SSIM loss to preserve local structure and perceptual quality. Tab. 5 summarizes the impact of the two pixel losses LPSNR and LSSIM. While omitting LSSIM still yields competitive results in Real (20), Nature (20), and Postcard (199), omitting LPSNR is significantly worse in PSNR (up to 7 dB on Wild) and SSIM metrics. Overall, using both losses is best.

PBR data. We systematically analyze the effect of the physically based rendered data qualitatively and quantitatively. Concretely, we train once with an alpha-blended augmentation, once with our PBR dataset, and once with a mixture

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

InputGTWindowSeatDSITDSRNetRDNetDAI

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

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

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

- Figure 5. Qualitative comparison. Each column shows one Input-GT pair with the corresponding predictions from WindowSeat and SotA methods. WindowSeat detects and removes the reflection in the first two examples, while other methods leave the reflections unaltered. Columns 3-5 visualize the improved reflection removal capabilities of WindowSeat, leaving fewer artifacts in the predictions. Best viewed on screen and zoomed in.

of both and then analyze the differences. To make a fair comparison, all experiments sample their transmission and blended layers from the same dataset, namely COCO [29]. Tab. 6 demonstrates that training without alpha-blended data achieves the best results for SIR2 with a 1.36 dB gain in Postcard (199). Fig. 4 (left) shows an example where training

on alpha-blended data fails, while the model trained on PBR data successfully removes the reflection.

To further analyze the effect of physically based rendering, we created a test set of 30 scenes, rendered in five settings with progressively increasing glass IoR values, resulting in progressively stronger reflections. Fig. 6 presents

Table 6. Ablation on training data evaluated on Nature (20), Real (20), and SIR2 sub-datasets (Postcard (199), Objects (200), Wild (55)).

Nature (20) Real (20) Objects (200) Postcard (199) Wild (55) PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑ PSNR ↑ SSIM ↑

Alpha-Blended PBR

✓ ✓ 26.96 0.845 26.45 0.860 28.62 0.939 27.81 0.930 28.89 0.932 ✓ ✗ 27.25 0.850 26.53 0.859 27.80 0.931 27.39 0.930 28.93 0.928

##### ✗ ✓ 27.12 0.849 26.28 0.856 28.81 0.944 29.17 0.934 28.97 0.935

[Figure 111]

- Figure 6. Ablation of PBR training data. Comparing PSNR, SSIM, MS-SSIM, and LPIPS metrics of two models on five test splits. One model was trained on PBR data and the other on alphablended data. The x-axis denotes the IoR range of the data split, indicating increasing reflection strength.

the behaviour of PSNR, SSIM, MS-SSIM, and LPIPS metrics across these test sets, demonstrating that training on PBR data provides improved robustness to stronger reflections.

#### 4.4. In-the-wild Evaluations and Limitations

We show qualitative in-the-wild predictions in Fig. 7, demonstrating the strong generalization capacity of WindowSeat. However, as shown in Fig. 8, it removes secondary and higher-order reflections, which might be misaligned with ground truth or the user’s intent. Future directions might include more fine-grained user control, e.g. specifying how many reflection layers should be removed.

### 5. Conclusion

We presented WindowSeat, a foundation DiT-based model fine-tuned with LoRA that achieves efficient single-image

[Figure 112]

- Figure 7. WindowSeat predictions in the wild. Our model demonstrates generalization across real-world reflection-removal scenarios. It effectively handles previously unseen conditions and remains robust even under challenging circumstances, including intense reflections and extreme cases such as wet or contaminated glass surfaces.

[Figure 113]

- Figure 8. WindowSeat failures. Ground truth of this SIR2 (WildScene 010) sample contains secondary reflections, removed by our method. Overshooting with removal may lead to loss of realism.

reflection removal on a single consumer GPU within one day of training. Unlike prior work, we place a strong emphasis on a high-fidelity PBR data generation pipeline, enabling a model that attains state-of-the-art performance on both established benchmarks and in-the-wild images. These results demonstrate that diffusion transformers, supported by physically grounded synthetic data and lightweight adaptation, provide an efficient and scalable framework for reflection removal and a strong basis for advancing other computational photography tasks. Looking ahead, the same principles can be extended to more challenging settings, including temporally consistent video and reflections involving subject self-shadows, paving the way towards more comprehensive through-glass imaging.

### References

- [1] Amit Agrawal, Ramesh Raskar, Shree K Nayar, and Yuanzhen Li. Removing photography artifacts using gradient projection and flash-exposure sampling. In ACM SIGGRAPH 2005 Papers, pages 828–835. Association for Computing Machinery,

2005. 2

- [2] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv–2506,

2025. 2, 4, 5

- [3] Brent Burley. Extending the disney brdf to a bsdf with integrated subsurface scattering. SIGGRAPH Course: Physically Based Shading in Theory and Practice. ACM, New York, NY, 19(7):9, 2015. 3
- [4] Brent Burley and Walt Disney Animation Studios. Physicallybased shading at disney. In Acm siggraph, pages 1–7. vol. 2012, 2012. 3
- [5] Jie Cai, Kangning Yang, Ling Ouyang, Lan Fu, Jiaming Ding, Huiming Sun, Chiu Ho, and Zibo Meng. F2t2-hit: A u-shaped fft transformer and hierarchical transformer for reflection removal. pages 809–814, 2025. 3, 4, 5
- [6] Blender Online Community. Blender - a 3D modelling and rendering package. Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018. 3
- [7] Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via block-wise quantization. 9th International Conference on Learning Representations, ICLR,

2022. 5

- [8] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. In Advances in Neural Information Processing Systems, pages 10088–10115. Curran Associates, Inc., 2023. 4
- [9] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88(2):303–338, 2010. 2, 3, 4
- [10] Kun Gai, Zhenwei Shi, and Changshui Zhang. Blind separation of superimposed moving images using image statistics. IEEE transactions on pattern analysis and machine intelligence, 34(1):19–32, 2011. 2
- [11] Xiaojie Guo, Xiaochun Cao, and Yi Ma. Robust separation of reflection from multiple images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2187–2194, 2014. 2
- [12] Poly Haven. Poly haven hdris. https://polyhaven.com/ hdris, 2025. Accessed: 2025-11-13. 4
- [13] Yuchen Hong, Haofeng Zhong, Shuchen Weng, Jinxiu Liang, and Boxin Shi. L-differ: Single image reflection removal with language-based diffusion model. In European Conference on Computer Vision, pages 58–76. Springer, 2024. 2, 4, 5
- [14] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3,

2022. 2, 4, 5

- [15] Jichen Hu, Chen Yang, Zanwei Zhou, Jiemin Fang, Xiaokang Yang, Qi Tian, and Wei Shen. Dereflection any image with diffusion priors and diversified data. arXiv preprint arXiv:2503.17347, 2025. 2, 3, 4, 5, 6
- [16] Qiming Hu and Xiaojie Guo. Trash or treasure? an interactive dual-stream strategy for single image reflection separation. Advances in Neural Information Processing Systems, 34:24683–24694, 2021. 2
- [17] Qiming Hu and Xiaojie Guo. Single image reflection separation via component synergy. In Proceedings of the IEEE/CVF international conference on computer vision, pages 13138– 13147, 2023. 2, 3, 4, 5, 6
- [18] Qiming Hu, Hainuo Wang, and Xiaojie Guo. Single image reflection separation via dual-stream interactive transformers. Advances in Neural Information Processing Systems, 37: 55228–55248, 2024. 3, 4, 5, 6
- [19] Yue Huang, Zi’ang Li, Tianle Hu, Jie Wen, Guanbin Li, Jinglin Zhang, Guoxu Zhou, and Xiaozhao Fang. Single image reflection removal via inter-layer complementarity. arXiv preprint arXiv:2505.12641, 2025. 2, 4, 5, 6
- [20] Sangho Jo, Ohtae Jang, Chaitali Bhattacharyya, Minjun Kim, Taeseok Lee, Yewon Jang, Haekang Song, Hyukmin Kwon, Saebyeol Do, and Sungho Kim. S-light: Synthetic dataset for the separation of diffuse and specular reflection images. Sensors, 24(7), 2024. 2
- [21] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
- [22] Bingxin Ke, Kevin Qu, Tianfu Wang, Nando Metzger, Shengyu Huang, Bo Li, Anton Obukhov, and Konrad Schindler. Marigold: Affordable adaptation of diffusionbased image generators for image analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–18, 2025. 2
- [23] Soomin Kim, Yuchi Huo, and Sung-Eui Yoon. Single image reflection removal with physically-based training images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5164–5173, 2020. 2
- [24] Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014. 4
- [25] Anat Levin, Assaf Zomet, and Yair Weiss. Learning to perceive transparency from the statistics of natural scenes. In Advances in Neural Information Processing Systems. MIT Press, 2002. 2
- [26] Anat Levin, Assaf Zomet, and Yair Weiss. Separating reflections from a single image using local features. In Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2004. CVPR 2004., pages I–I. IEEE, 2004. 2
- [27] Chao Li, Yixiao Yang, Kun He, Stephen Lin, and John E Hopcroft. Single image reflection removal through cascaded refinement. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3565–3574,

2020. 2, 4, 5, 6

- [28] Yu Li and Michael S Brown. Single image layer separation using relative smoothness. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2752–2759, 2014. 2
- [29] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 2, 3, 7
- [30] Yaron Lipman, {Ricky T.Q.} Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. 2023. 11th International Conference on Learning Representations, ICLR 2023 ; Conference date: 01-05-2023 Through 05-05-2023. 2
- [31] Derek Liu. (lora) fine-tuning flux.1-dev on consumer hardware, 2025. Accessed: 2025-11-10. 2, 4, 5
- [32] Yu-Lun Liu, Wei-Sheng Lai, Ming-Hsuan Yang, Yung-Yu Chuang, and Jia-Bin Huang. Learning to see through obstructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2
- [33] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2017. 4, 5
- [34] Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. PEFT: Stateof-the-art parameter-efficient fine-tuning methods. https: //github.com/huggingface/peft, 2022. 5
- [35] Shree K Nayar, Xi-Sheng Fang, and Terrance Boult. Separation of reflection components using color and polarization. International Journal of Computer Vision, 21(3):163–186,

1997. 2

- [36] Simon Niklaus, Xuaner Cecilia Zhang, Jonathan T Barron, Neal Wadhwa, Rahul Garg, Feng Liu, and Tianfan Xue. Learned dual-view reflection removal. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3713–3722, 2021. 2
- [37] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 4

- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [39] Renjie Wan, Boxin Shi, Ling-Yu Duan, Ah-Hwee Tan, and Alex C. Kot. Benchmarking single-image reflection removal algorithms. In International Conference on Computer Vision (ICCV), 2017. 2, 4, 5
- [40] Tao Wang, Wanglong Lu, Kaihao Zhang, Wenhan Luo, TaeKyun Kim, Tong Lu, Hongdong Li, and Ming-Hsuan Yang. Promptrr: Diffusion models as prompt generators for single image reflection removal. arXiv preprint arXiv:2402.02374,

2024. 2

- [41] Tianfu Wang, Mingyang Xie, Haoming Cai, Sachin Shah, and Christopher A Metzler. Flash-split: 2d reflection removal with flash cues and latent diffusion separation. In Proceedings

- of the Computer Vision and Pattern Recognition Conference, pages 5688–5698, 2025. 2
- [42] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. 5
- [43] Jie Yang, Dong Gong, Lingqiao Liu, and Qinfeng Shi. Seeing deeply and bidirectionally: A deep learning approach for single image reflection removal. In Proceedings of the European Conference on Computer Vision (ECCV), 2018. 2
- [44] Kangning Yang, Ling Ouyang, Huiming Sun, Jie Cai, Lan Fu, Jiaming Ding, Chiu Man Ho, and Zibo Meng. Openrr-1k: A scalable dataset for real-world reflection removal. In 2025 IEEE International Conference on Image Processing (ICIP), pages 839–844, 2025. 2
- [45] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In Advances in Neural Information Processing Systems, pages 21875–21911. Curran Associates, Inc., 2024. 2
- [46] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 2
- [47] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [48] Xuaner Zhang, Ren Ng, and Qifeng Chen. Single image reflection separation with perceptual losses. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4786–4794, 2018. 2, 4, 5, 6
- [49] Hao Zhao, Mingjia Li, Qiming Hu, and Xiaojie Guo. Reversible decoupling network for single image reflection removal. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26430–26439, 2025. 1, 2, 3, 4, 5, 6
- [50] Haofeng Zhong, Yuchen Hong, Shuchen Weng, Jinxiu Liang, and Boxin Shi. Language-guided image reflection separation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24913–24922, 2024. 2
- [51] Yurui Zhu, Xueyang Fu, Peng-Tao Jiang, Hao Zhang, Qibin Sun, Jinwei Chen, Zheng-Jun Zha, and Bo Li. Revisiting single image reflection removal in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25468–25478, 2024. 2, 4, 5

