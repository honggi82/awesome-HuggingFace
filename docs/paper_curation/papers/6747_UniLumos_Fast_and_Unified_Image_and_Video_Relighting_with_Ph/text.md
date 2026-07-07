# arXiv:2511.01678v1[cs.CV]3Nov2025

## UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback

Ropeway Liu1,2,∗, Hangjie Yuan†2,3,1,∗, Bo Dong2,3, Jiazheng Xing1,2,4, Jinwang Wang2,3,1, Rui Zhao4, Yan Xing2,3, Weihua Chen†2,3, Fan Wang2 1Zhejiang University, 2DAMO Academy, Alibaba Group, 3Hupan Lab, 4National University of Singapore ∗ Equal contributions. † Corresponding author.

{yuanhangjie.yhj, kugang.cwh}@alibaba-inc.com

Relighting is a crucial task with both practical demand and artistic value, and recent diffusion models have shown strong potential by enabling rich and controllable lighting effects. However, as they are typically optimized in semantic latent space, where proximity does not guarantee physical correctness in visual space, they often produce unrealistic results—such as overexposed highlights, misaligned shadows, and incorrect occlusions. We address this with UniLumos, a unified relighting framework for both images and videos that brings RGB-space geometry feedback into a flow-matching backbone. By supervising the model with depth and normal maps extracted from its outputs, we explicitly align lighting effects with the scene structure, enhancing physical plausibility. Nevertheless, this feedback requires high-quality outputs for supervision in visual space, making standard multi-step denoising computationally expensive. To mitigate this, we employ path consistency learning, allowing supervision to remain effective even under few-step training regimes. To enable fine-grained relighting control and supervision, we design a structured six-dimensional annotation protocol capturing core illumination attributes. Building upon this, we propose LumosBench, a disentangled attribute-level benchmark that evaluates lighting controllability via large vision-language models, enabling automatic and interpretable assessment of relighting precision across individual dimensions. Extensive experiments demonstrate that UniLumos achieves state-of-the-art relighting quality with significantly improved physical consistency, while delivering a 20x speedup for both image and video relighting. Code is available at https://github.com/alibaba-damo-academy/Lumos-Custom.

Date: November 4, 2025

1 Introudction

Relighting, altering illumination in images or videos while preserving intrinsic scene attributes such as geometry, reflectance, and content, is a longstanding problem in computer vision and graphics [33, 55]. It underpins a wide range of applications in film production, gaming, and augmented reality, where seamless lighting integration is critical to visual fidelity. Beyond realism, lighting conveys rich aesthetic and semantic cues—it defines atmosphere, evokes emotion, and reinforces narrative structure. Relighting is thus not only a technical challenge but also a creative tool that shapes how characters, objects, and environments are perceived. However, achieving physically consistent lighting remains a core challenge—requiring the alignment of illumination effects with different scene attributes across both space and time. To address this, traditional approaches often rely on inverse rendering pipelines [46, 50, 2] that estimate intrinsic scene properties, such as geometry, reflectance, and environmental lighting, from input images. While these methods provide physically grounded results, they typically require complex inputs—such as high dynamic range images or spherical harmonics coefficients—and are limited to constrained domains. This makes them impractical for real-world applications, where users often provide only a single image, a short video, or a high-level lighting prompt as input. These limitations underscore the need for a new paradigm—one that can deliver high-quality, physically plausible relighting while operating under minimal and naturalistic input conditions.

Recent diffusion-based relighting methods [49, 6, 12, 57] have shown promise by leveraging large-scale image and video datasets to produce diverse and controllable lighting effects under various user-defined conditions,

Input Relight Input Relight

Input Relight

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Text-ConditionedRelighting

Left, Artificial, Dim, Cool, Dynamic, None

Back, Natural Light, Warm, Static, Transmission

Left, Natural, Moderate, Neutral, Static, Transmission

Reference Background Video

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

Input

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Background-ConditionedRelighting

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Frame 1 Frame 16 Frame 32 Frame 48

- Figure 1 UniLumos performs physically plausible image and video relighting, conditioned on textual prompts and reference videos.

such as reference images or text prompts. However, this strength reveals a fundamental weakness: diffusion models typically operate in semantic latent space, where similarity does not guarantee physical correctness in the visual domain. As a result, they often fail to respect scene geometry, especially in complex scenes with dynamic lighting or temporal constraints. For example, IC-Light [49] and SynthLight [6], which are primarily designed for image relighting, lack both temporal modeling and explicit physical supervision in the visual domain. Instead, they rely on latent-space representations, such as MLP-based embeddings (IC-Light) or multi-stage training (SynthLight), which often lead to artifacts like misaligned shadows, overexposed highlights, or incorrect lighting directions—particularly under complex geometry or extreme illumination. Light-A-Video [12] and RelightVid [12] extend these methods to the video domain, aiming to improve temporal coherence while retaining the visual quality of diffusion-based relighting. Light-A-Video is a training-free framework that combines IC-Light with a pre-trained video diffusion model via iterative alignment. While this improves frame-wise consistency, it incurs high inference costs due to multiple model passes. RelightVid adopts a joint training strategy with a video diffusion backbone, which enhances temporal stability compared to training-free approaches. However, it still operates without explicit physical supervision, resulting in inaccurate light-scene interactions and limited generalization to complex or dynamic environments. In short, existing diffusion-based methods excel in synthesizing plausible appearances but fall short in enforcing the physical plausibility that is essential for high-quality relighting.

To bridge the gap between generative flexibility and physical correctness, we propose UniLumos, a unified relighting framework for both images and videos that brings RGB-space geometry feedback into a flow-matching

backbone. Unlike existing diffusion-based approaches that operate purely in latent space, UniLumos introduces a physics-plausible feedback mechanism that supervises generation with dense geometric signals—specifically, depth and surface normals—estimated from its outputs. These lighting-invariant cues serve as ideal supervision signals, enabling the model to explicitly align illumination with the scene structure, significantly improving shadow alignment, shading consistency, and spatial coherence. Nevertheless, this feedback requires high-quality outputs for supervision in visual space, making standard multi-step denoising computationally expensive. To mitigate this, we employ path consistency learning [13], allowing supervision to remain effective even under few-step training regimes.

Beyond model-level improvements, existing relighting methods lack structured illumination descriptions and dedicated evaluation metrics. Generic generation scores (e.g., FID, LPIPS) fail to capture lighting-specific errors such as shadow misalignment, intensity mismatch, or incorrect light direction. To address this, we introduce LumosData, a scalable data pipeline that extracts diverse relighting pairs from real-world videos. At its core is a structured six-dimensional annotation protocol covering direction, light source type, intensity, color temperature, temporal dynamics, and optical phenomena—enabling both fine-grained conditioning during training and physically grounded evaluation at test time. Building upon this, we propose LumosBench, a disentangled attribute-level benchmark that evaluates lighting controllability via large vision-language models, enabling automatic and interpretable assessment of relighting precision across individual dimensions.

Our contributions are summarized as follows:

- • Unified Relighting with Physics-Plausible Feedback: We propose UniLumos, a unified relighting framework for both images and videos that incorporates RGB-space geometry feedback into a flow-matching backbone, explicitly aligning lighting effects with the scene structure to enhance the physical plausibility of relighting.
- • Structured Illumination Annotation and Evaluation Benchmark: We design a structured six-dimensional annotation protocol that captures core illumination attributes, enabling fine-grained control and supervision. Building upon this, we introduce LumosBench, a disentangled attribute-level benchmark that leverages large vision-language models to automatically and interpretably evaluate relighting controllability across individual lighting dimensions.
- • Extensive Validation: Extensive experiments demonstrate that UniLumos achieves state-of-the-art relighting quality with significantly improved physical consistency, while delivering a 20x speedup for both image and video relighting.

- 2 Related Work

Video Diffusion Models. Recent advances in video diffusion models [3, 4, 7, 38] have enabled the generation of temporally coherent video sequences conditioned on various inputs such as text or images. In the field of text-to-video (T2V) generation [45], most methods extend existing text-to-image diffusion backbones with additional modules that capture temporal dynamics across frames. In contrast, a few approaches train video diffusion models from scratch to directly learn spatiotemporal priors [38]. For image-to-video (I2V) tasks, where static images are animated with plausible motion, several methods propose specialized architectures tailored for image animation [35, 40]. Other strategies offer lightweight, plug-and-play adapters that can be integrated into pre-trained models. Stable Video Diffusion [2], for example, fine-tunes T2V models for I2V tasks, achieving state-of-the-art performance. Beyond synthesis quality, a growing body of work emphasizes controllability, allowing users to guide generation with fine-grained constraints [52, 48].

Relighting Methods. Recent advances in deep neural networks have significantly improved lighting control for 2D and 3D visual content, especially in portrait relighting. Methods such as Relightful Harmonization [32], SwitchLight [21], ConceptSliders [14], Intrinsic Image Diffusion [22], Neural Gaffer [20], DI-Light [44], SynthLight [6], and IC-Light [49] demonstrate progress in realism and controllability. While numerous portrait relighting approaches exist [31, 47, 42, 5], most of them rely heavily on portrait-specific priors. In contrast, UniLumos is designed as a general-purpose relighting framework that is not constrained to any particular object category. With the rise of diffusion-based generative models, approaches like LumiSculpt [51] extend lighting control to text-to-video (T2V) generation. Moreover, RelightVid [12] and Light-A-Video [57] implemented

video relighting based on IC-Light. However, achieving both precise lighting control and high visual quality in video remains challenging due to the trade-off between spatial realism and temporal consistency.

Feedback Learning in Generative Models. Feedback learning has become a powerful tool to improve output alignment in generative models, from language systems trained with human preferences [36, 30] to visual diffusion models guided by aesthetic or attribute-based rewards [9, 43, 37, 25], e.g., InstructVideo [43] and DRaFT [10]. In visual domains, feedback can also be physical—for example, using geometric cues to guide generation toward realism. Recent advances in distillation and consistency training [15, 34, 29, 39, 53, 13] have accelerated diffusion inference by reducing the number of denoising steps, enabling models to recover high-quality RGB outputs with just a few iterations. However, most existing techniques focus on appearance synthesis and overlook geometry-aware feedback, which typically requires high-fidelity outputs and is incompatible with few-step inference. UniLumos bridges this gap by combining physically grounded supervision with path-consistency learning, enabling efficient and physically plausible relighting under fast sampling regimes.

- 3 Preliminaries

ProblemFormulation. Given an image or video S1 ∈ RT×H×W×C with intrinsic scene properties (e.g., geometry, reflectance, content) under initial illumination L1, the goal of relighting is to modify the illumination within a subject region specified by a binary mask M ∈ {0,1}T×H×W to match a target lighting condition C. The condition C may take the form of an image, video, or text description and implicitly defines a desired illumination field L2. The relit output S2 ∈ RT×H×W×C should exhibit lighting consistent with L2 in the masked region M while preserving the intrinsic attributes of S1. This can be formulated as a conditional generation problem:

S2 = fθ(S1,C,M), (1) where fθ is the relighting model parameterized by θ.

Flow Matching. To efficiently model complex illumination transformations in relighting, we build upon Wan2.1 [38], a foundation video generation model based on flow matching [28, 11]. Flow matching formulates generative modeling as learning velocity fields between noise x0 ∼ N(0,I) and data x1, using a linear interpolation:

xt = t · x1 + (1 − t) · x0, vt =

dxt dt

= x1 − x0, (2)

where t ∈ [0,1] is sampled from a logit-normal distribution. The model learns to predict vt from xt, conditioned on timestep t and context c (e.g., text embeddings), by minimizing the mean squared error:

L0 = Ex

0,x1,t,c ∥vθ(xt,t,c) − vt∥22 . (3)

Path Consistency Learning. To further accelerate inference, we adopt path consistency learning [13], which encourages consistent velocity predictions under larger integration steps. Given a velocity field vθ and step size d > 0, we recursively define:

xt+d = xt + d · vθ(xt,t,d), (4) Moreover, enforce two-step consistency using:

Lfast = Ex

t,t,d vθ(xt,t,2d) −

- 1

- 2

[vθ(xt,t,d) + vθ(xt+d,t + d,d)]

2

2

. (5)

This objective enables the model to learn shortcut-consistent velocity fields without separate teacher-student stages, allowing for fast and high-quality generation with arbitrary step budgets.

- 4 Methodology

We present UniLumos, a unified framework for physically plausible image and video relighting, as illustrated in Fig. 2. Built upon Wan 2.1 [38], a flow-matching diffusion model for video generation, UniLumos relights

###### LumosData UniLumos

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

𝑽 ∈ ℝ[   , , , ] 𝑽 ∈ ℝ[   , , , ] 𝑽 ∈ ℝ[   , , , ]

Noise 𝒙

###### Step1

###### ❄ ❄

[Figure 46]

[Figure 47]

###### ❄

[Figure 48]

Subject Mask

Wan-VAE Encoder

Wan-VAE Encoder

Wan-VAE Encoder

𝑽 ∈ ℝ[   , , , ]

𝑴 ∈ ℝ[   , , ]

𝒙 𝒙

𝒙

Concat

###### Add Noise

Gaussian Background

Lumos Augmentation

Step2 Step3

ℝ[ /   , / , / ,  ]

Time-step 𝑡

New Caption

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

###### 🔥

[Figure 53]

[Figure 54]

[Figure 55]

umT5

N × DiT Blocks

Concat

###### Desired-step 𝑑

###### ❄

[Figure 56]

###### Wan-VAE Decoder

𝑽 ∈ ℝ[   , , , ]

𝑽 ∈ ℝ[   , , , ]

❄ 🔥

[Figure 57]

Frozen Weights Trainable Weights

Feedback

Feedback

[Figure 58]

Original Caption

Step4 Caption Augmentation

A man in a plaid shirt … In the background, …

𝑽 ∈ ℝ[   , , , ]

“Front Light, Side Light, Back Light, Top Light, Bottom Light, Split Light, None”

|Direction of Light|
|---|

###### ❄

[Figure 59]

|[Figure 60]|
|---|

Physics-Plausible Feedback

“Natural Light, Artificial Light, Back Light, Rendering Light” “Glare (>1000 lumens), Moderate (200– 1000 lumens), Dim (<200 lumens)” “Cool, Neutral, Warm” “Static, Dynamic” “Transmission (Glass), Refraction/Reflection (Water Surface, Mirror), Refraction/Reflection

|Light Source Type|
|---|

Vision language models (VLM)

|Light Intensity|
|---|

A pre-trained Dense estimation model (e.g. Lotus)

Tasks: Analyze the input image/video

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

|Color Temperature|
|---|

[Figure 65]

[Figure 66]

|New Caption<br><br>A man in a plaid shirt … In the background, … Front Light, Ar ﬁcial Light, Moderate, Neutral, Sta c Light, None.<br><br>|
|---|

|Light Changes|
|---|

Depth Map 𝐷 ∈ ℝ[   , , ]

|Optical|
|---|

Normal Map 𝑽 ∈ ℝ[   , , , ] 𝑁 ∈ ℝ[   , , ]

(Mirror), Scattering (Fog Effect), and None.”

- Figure 2 The overall pipeline of UniLumos. The left is LumosData, our proposed data construction pipeline, which consists of four stages for generating diverse relighting pairs from real-world sources. The right shows the architecture of UniLumos, a unified framework for image and video relighting, designed to achieve physically plausible illumination control.

images and videos under user-specified lighting conditions—including reference images, video clips, or text prompts—while preserving scene content and temporal coherence.

To bridge the gap between semantic generation and physical correctness, UniLumos incorporates two key innovations: (1) a physics-plausible feedback that supervises the model with geometry signals from RGB space, and (2) a structured illumination annotation protocol that enables fine-grained control and evaluation. We jointly train the model with geometry-aware supervision and lighting-conditioned objectives, achieving high-quality and efficient few-step inference.

- 4.1 Physics-Plausible Feedback

While most relighting methods rely on photometric reconstruction or latent-space consistency, such signals offer limited geometric grounding—often resulting in misaligned shadows, implausible shading, and incorrect light directions. To address this, UniLumos enforces consistency between generated illumination and underlying scene geometry, promoting more realistic light–scene interactions.

As illustrated in Fig. 2 (right), we introduce a physics-plausible feedback that guides the generation process using geometry-aware supervision. This component complements the flow-matching architecture with explicit structural priors, enhancing physical plausibility without altering the model’s inference inputs. We adopt depth and surface normals as our supervisory targets due to their generality, accessibility, and strong disentanglement from illumination. Unlike shadow masks or material properties, which are often ambiguous, entangled with lighting, or costly to obtain, monocular depth and normal maps capture intrinsic scene structure. They can be reliably estimated by a pre-trained dense estimator (e.g., Lotus [17]).

Specifically, after decoding the predicted latent variable into RGB frames via the Wan-VAE Decoder [38], we extract estimated depth Dˆ ∈ RT×H×W and normals Nˆ ∈ RT×H×W using a frozen dense estimator. These are

compared against pseudo-ground-truth maps (D,N) from the reference input to compute the geometry-aware feedback loss:

+ ∥Nˆ − N∥2 ∥N∥2

∥Dˆ − D∥2 ∥D∥2

, (6)

Lphy = Ex

0,x1,t,c M ⊙

where M ∈ RT×H×W denotes the foreground subject mask. This feedback encourages the model to align its lighting predictions with consistent structural interpretation while keeping inference lightweight and geometry-free.

However, the proposed physics-plausible feedback requires supervision in the RGB domain, which relies on high-quality predictions that are typically only available after full-step denoising has been completed. This poses a major computational bottleneck for standard diffusion models. To mitigate this, we adopt path consistency learning [13], which reformulates denoising as a velocity regression task, thereby supporting practical training under a few-step regimes. Enforcing consistency between intermediate outputs and final predictions enables reliable geometric feedback without sacrificing inference efficiency.

- 4.2 Structured Illumination Annotation and Evaluation Benchmark

In the problem formulation, the relighting task involves conditioning on a target illumination descriptor C. However, most prior work treats C as unstructured prompts—such as text, images, or reference frames—offering limited control or interpretability. Moreover, conventional evaluation metrics, such as FID or LPIPS, focus on perceptual similarity but fail to capture lighting-specific discrepancies, such as shadow misalignment or intensity mismatches.

To address this, we construct LumosData, a scalable dataset pipeline that enriches C with structured lighting semantics. As shown in Fig. 2 (left), we extract relighting pairs from real-world videos. Given an input sequence Vreal ∈ R[T+1,H,W,3], we first obtain subject masks M ∈ {0,1}[T+1,H,W] using BiRefNet [54] to isolate the foreground. We then apply a pre-trained relighting model such as IC-Light [49] to generate synthetic relit versions Vdeg under diverse lighting conditions, guided by a curated prompt set. To avoid entanglement with background semantics, we inpaint the background using Gaussian noise, ensuring clean illumination signals without introducing artifacts.

Beyond this relighting pipeline, LumosData introduces a structured six-dimensional annotation protocol that covers direction, intensity, color temperature, light source type, temporal dynamics, and optical phenomena.

These attributes are automatically generated using vision-language models (e.g., Qwen2.5-VL [1]) with carefully designed prompts, and are integrated into C to provide an enriched semantic label. This protocol serves dual purposes: (1) Fine-grained conditioning: During training, the model is guided by explicit lighting attributes embedded in C, promoting more interpretable and controllable generation across scenarios. (2) Attributealigned Benchmark: Building upon the same attribute protocol, we construct LumosBench. This automatic benchmark uses vision-language models to assess whether generated outputs accurately reflect intended lighting conditions, enabling interpretable, attribute-level controllability evaluation beyond pixel-based metrics.

Each training tuple is structured as (Vdeg,Vbg,M,C) → Vreal, where the model learns to restore realistic lighting consistent with the semantic cue C and structural context. LumosData introduces rich diversity in content and illumination by leveraging a variety of real-world sources. Built on Panda70M [8], we curate ∼110K high-quality video pairs and augment training with 1.2M additional relit images using IC-Light. This combination supports robust learning of physically plausible relighting without relying on expensive hardware or manual annotations. See more details in Appendix B.

- 4.3 Joint Learning Objective and Training Strategy

Model Implementation. The inputs of aligned videos (Vdeg,Vbg,Vreal) are passed through a Wan-VAE Encoder [38] to obtain semantic latent representations (xdeg,xbg,x0). During training, we generate the noisy latent input xt via Eq. 2, and concatenate it with the strong conditional signals xdeg and xbg along the channel dimension. This combined tensor is injected into the DiT blocks of the Wan backbone. Additionally, to support path-consistency learning, the diffusion step t and the expected denoising step d are appended as

temporal condition vectors. All new projection and fusion layers are initialized with zero weights to preserve compatibility with the pre-trained Wan initialization and ensure stable optimization from the outset.

Joint Objective. Our training objective integrates three complementary losses to balance appearance fidelity, geometric consistency, and fast inference. The full loss is defined as:

L = λ0L0 + λ1Lfast + λ2Lphy, (7)

where L0 is the standard flow-matching loss that aligns the predicted velocity field with the ground-truth field, Lfast is the path consistency loss that improves model performance under few-step denoising regimes, and Lphy is a physics-guided loss that supervises the RGB outputs using estimated depth and normal maps. We adopt fixed weights of λ0 = 1.0 and λ1 = λ2 = 0.1 for all experiments. This unified objective encourages the model to generate relit results that are photorealistic, temporally smooth, and physically grounded while supporting efficient inference without sacrificing output quality.

Training Strategy. To balance physical supervision and training efficiency, we adopt a selective optimization strategy inspired by path consistency scheduling [13]. In each training iteration, we divide the batch based on supervision type, following an 80/20 split to avoid prohibitive costs from full supervision while still maintaining effective learning signals. As shown in Alg. 1, 20% of each batch is allocated to compute the path consistency loss Lfast, which involves three forward passes and one backward pass to enforce consistency across timesteps. The remaining 80% is used for the standard flowmatching loss L0, with 50% of these samples further supervised using RGB-space geometry feedback via Lphy (i.e., depth and normal alignment). This probabilistic scheduling ensures high training throughput while allowing the model to benefit from multi-level supervision. To further enhance illumination diversity during training, we apply randomized lighting augmentations on the degraded subject Vdeg, which introduces realistic lighting variability without the need for explicitly paired captures.

Algorithm 1 Loss Sampling Strategy per Iteration Require: Batch size B, total training samples

- 1: for each training iteration do
- 2: Randomly sample 20% of batch → Lfast
- 3: Compute path consistency loss:
- 4: 3× forward, 1× backward
- 5: Remaining 80% → L0
- 6: Among those, 50% → RGB reconstruction
- 7: Compute physics-guided loss Lphy
- 8: end for

### 5 Experiments

- 5.1 Experimental Details

Training Details. We adopt the Wan2.1-T2V-1.3B-480P [38] as the base model, and initialize all new trainable parameters with zeros to minimize its influence at the beginning of training. We use the AdamW optimizer with the learning rate of 1e-5 for training the entire framework. All the models are trained with a batch size of 8 for 5,000 iterations on 8 NVIDIA H20 GPUs (with 96GB RAM).

Baselines. We compare UniLumos against a range of representative image and video relighting methods. For image-based relighting, we include SwitchLight [21], DiLightNet [44], IC-Light [49], and SynthLight [6], which leverage various forms of latent modeling or light disentanglement to relight single images. For video relighting, we apply IC-Light via frame-by-frame and include Light-A-VideoCogVideoX-2B[41] and another using Wan 2.1 T2V-1.3B [38]. These baselines together represent state-of-the-art performance across both image and video relighting settings.

Dataset. For testing, we selected samples from the internal dataset, processed using the method described in Sec. B. These samples were evenly split: half for image generation at 768x512 resolution, and half for video generation at 480p resolution (832x480), with each video sample containing 49 frames. To further demonstrate the model’s generalization to non-human scenes, we conducted additional evaluations on two public object-centric relighting benchmarks: StanfordOrb [24] and Navi [19], which include objects and sculptures under a variety of lighting environments and are completely disjoint from our training data, and see more results in Appendix C.1.

##### Table 1 Quantitative comparison. Bold number indicate the best performance.

(a) Quality (b) Temporal Consistency (c) Lumos Consistency

Model

###### PSNR ↑ SSIM ↑ LPIPS ↓ R-Motion↓ Avg. Score ↑ Dense L2 Error ↓

Image Relighting

SwitchLight [21] 20.483 0.901 0.094 - 0.717 0.388 DiLightNet [44] 21.894 0.860 0.131 - 0.682 0.401 IC-Light [49] 24.316 0.884 0.108 - 0.703 0.447 SynthLight [6] 25.572 0.905 0.102 - 0.791 0.214

UniLumos 26.719 0.913 0.089 - 0.912 0.103

Video Relighting

###### IC-Light Per Frame [49] 20.132 0.851 0.133 2.437 0.672 0.432 Light-A-Video [57] + CogVideoX[41] 19.851 0.859 0.124 1.784 0.641 0.383 Light-A-Video [57] + Wan2.1[38] 20.784 0.876 0.129 1.582 0.682 0.371

UniLumos 25.031 0.891 0.109 1.436 0.871 0.147

Evaluation metrics. We evaluate relighting performance across three key dimensions: (1) visual fidelity: We assess image quality using Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS). For video relighting, we report the average metric across all frames. (2) temporal consistency: Following VBench [18], we adopt the R-Motion metric, which measures temporal smoothness using motion priors from a pre-trained video frame interpolation model [27]. This captures the coherence of lighting transitions across frames. (3) lumos consistency: (i) Lumos Score, computed by applying the same caption-based lighting annotation protocol as in our LumosData construction. Six lighting attributes are predicted and compared with targets, and each is weighted equally to yield an average consistency score. (ii) Dense L2 Error, which quantifies the relative L2 error between predicted and reference depth/normal maps, estimated via a pre-trained geometry model (e.g., Lotus [17]). This provides a physically grounded measure of illumination-geometry alignment.

- 5.2 Main Results

Quantitative Evaluation. As shown in Tab. 1, UniLumos delivers consistent improvements across three key dimensions: visual fidelity, temporal consistency, and physically grounded lighting alignment. (1) Visual Fidelity. UniLumos produces higher-quality relighting results across both images and videos. Benefiting from structured lighting supervision and geometry-guided feedback, our model generates outputs with clearer shading, sharper details, and more coherent illumination compared to prior works. (2) Temporal Consistency. For video relighting, UniLumos ensures smoother transitions and reduced flickering artifacts. Our use of flowmatching architecture and path consistency learning helps maintain stable lighting across frames, addressing a key limitation in frame-wise or training-free methods.

[Figure 67]

(3) Lumos Consistency. Going beyond appearancebased evaluation, UniLumos aligns well with intended lighting semantics. Through structured caption conditioning and physics-guided training, the model better preserves lighting direction, tone, and geometry—validated by both vision-language alignment and dense geometric error metrics.

277 s

IC-Light

917 s

Light-A-Video CogVideoX-2B

Light-A-Video Wan-1.3B

756 s

76x

12s

UniLumos-1.3B

Efficiency. To assess inference efficiency, we evaluate the video relighting task under a standardized setting, generating 49-frame videos at a 480p resolution. As shown in Fig. 5, UniLumos achieves a significant speedup compared to prior methods, benefiting from its geometry-free inference and few-step generation. While existing models, such as Light-A-Video or IC-Light, require either iterative frame-by-frame processing or complex sampling schedules, UniLumos completes generation over 20 times faster without sacrificing visual fidelity or physical plausibility. This efficiency advantage makes it well-suited for real-time relighting applications and scalable deployment scenarios.

Inference Time on 49x832x480

Figure 5 Comparison of inference time costs of different methods under the same settings.

Frame 1 Frame 24 Frame 48 Frame 1 Frame 24 Frame 48

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

InputIC-Light Light-A-Video

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

CogVideoX Light-A-Video

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Wan UniLumos

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Our（）

“A man in a black hoodie and cap sits at … The room‘s walls are black, and the floor is a light color … The ceiling lights are arranged in a grid pattern, … Front Light, Artificial Light, Moderate, Neutral, Static Light, None”

“A man with a beard, wearing a dark shirt and ... The background is a plain, light-colored wall … Front Light, Artificial Light, Moderate, Neutral, Static Light, None”

###### Caption

- Figure 3 Qualitative comparison of baseline methods. Each method takes a subject video and a textual illumination description as input, generating the related subject with the corresponding background under the specified lighting condition.

Qualitative Results We present qualitative comparisons in Fig. 3 and Fig. 4, highlighting the advantages of UniLumos in terms of lighting realism, temporal coherence, and controllability. (1) Lighting Quality and Controllability: In Fig. 3, UniLumos produces lighting effects that better match the target description, capturing nuanced directional shadows, color tone, and intensity. Competing methods either fail to reflect the intended lighting change or produce overly uniform results that lack realism. (2) Temporal Consistency: Compared to baseline methods such as frame-wise IC-Light and Light-A-Video, UniLumos achieves smoother frame transitions without flickering or structural distortion. This benefit arises from the joint modeling of space and time, which is further reinforced by physics-aware supervision and path consistency training. (3) Foreground Detail Preservation: UniLumos preserves fine subject details—such as facial structure and clothing texture—better than baselines. For instance, Light-A-Video occasionally introduces deformation or identity drift, while our model maintains high fidelity over long sequences. (4) Relighting with Reference Videos: Fig. 4 showcases UniLumos conditioned on different reference videos. The model successfully adapts both global lighting direction and subtle spatial variations across scenes, demonstrating strong generalization under diverse illumination cues.

LumosBench To evaluate the fine-grained controllability of lighting generation, we introduce LumosBench, a structured benchmark that targets six core illumination attributes defined in our annotation protocol. Unlike prior works that treat lighting holistically or implicitly, LumosBench provides a disentangled, attribute-level evaluation, enabling precise diagnosis of model behavior under controllable lighting conditions. See more results in Appendix C.2.

- 5.3 Ablation Study

As shown in Tab. 2 and Fig. 6, we conduct ablation studies to analyze the effectiveness of different components. Physics-Guided Feedback. Removing both depth and normal feedback (w/o All Feedback) leads to significant degradation in both image quality and physical consistency, confirming the necessity of our physics-guided loss. Notably, omitting only normal supervision causes a larger drop than removing depth, suggesting that surface orientation plays a more critical role than distance in shaping light–shadow interactions. Path Consistency Learning. Excluding this component (w/o Path Consistency) yields only minor drops in physical metrics while maintaining competitive SSIM and LPIPS scores. This shows that path consistency incurs little performance

Frame 1 Frame 24 Frame 48

Frame 1 Frame 24 Frame 48

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Input

Relight

Reference Background Video

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

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

- Figure 4 UniLumos performs physically plausible video relighting conditioned on different reference videos.

- Table 2 Quantitative comparison. Bold number indicate the best performance.

(a) Quality (b) Temporal Consistency (c) Lumos Consistency

Model

###### PSNR ↑ SSIM ↑ LPIPS ↓ R-Motion↓ Avg. Score ↑ Dense L2 Error ↓

Ablative Study

w/o Depth Feedback 23.472 0.883 0.118 1.443 0.870 0.265 w/o Normal Feedback 22.115 0.874 0.123 1.446 0.863 0.173

w/o All Feedback 21.433 0.862 0.139 1.473 0.859 0.297 w/o Path Consistency 25.317 0.902 0.113 1.438 0.875 0.153

Effect of Training Domain

###### Only Video 22.487 0.863 0.119 1.487 0.857 0.173 Only Image 24.471 0.872 0.123 2.429 0.841 0.182

UniLumos 25.031 0.891 0.109 1.436 0.871 0.147

cost but offers substantial efficiency benefits in few-step regimes, justifying its inclusion. Training Modality. To evaluate the effectiveness of our unified training paradigm, we compare domain-specific variants: training solely on videos leads to poor visual quality, while image-only training sacrifices temporal smoothness. In contrast, our unified approach strikes the best balance—achieving high-quality and temporally coherent relighting across both input types.

- 6 Conclusion

We introduce UniLumos, a unified framework for physically plausible image and video relighting. It aligns illumination with scene geometry via RGB-space depth and normal supervision, improving shadow accuracy and spatial consistency. To enhance controllability and evaluation, we propose a structured six-dimensional lighting annotation protocol, enabling fine-grained conditioning and physically grounded assessment through VLMs. Experiments show that UniLumos achieves superior relighting quality, physical consistency, and inference efficiency.

Frame 1 Frame 24 Frame 48 Frame 1 Frame 24 Frame 48

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Input

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Matching

Consistency Flow

###### DenoisingStep=1

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Path

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Feedback UniLumos

DenoisingStep=5

w/o

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

“A woman with long, wavy brown hair sings into a gold-colored microphone … The background is a blur of blue and white lights ... The woman's makeup includes dark eyeliner and eyeshadow … Front Light, Artificial Light, Moderate, Cool Tone, Static Light, None”

“A man with short, dark hair and a light beard, wearing a light gray buttonup shirt with a blue collar, sits at a table, looking contemplative … The background is blurred with lights and dark areas … Side Light, Artificial Light, Moderate, Neutral, Static Light, None”

Caption

- Figure 6 Ablation study. We compare the effects of different components under few-step denoising. For 1-step, we show the impact of flow-matching with and without path consistency. For 5-step, we visualize results before and after introducing physics-plausible feedback.

Acknowledgment

This work was supported by Damo Academy through Damo Academy Research Intern Program.

References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [4] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575, 2023.
- [5] Ziqi Cai, Kaiwen Jiang, Shu-Yu Chen, Yu-Kun Lai, Hongbo Fu, Boxin Shi, and Lin Gao. Real-time 3d-aware portrait video relighting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6221–6231, 2024.

- [6] Sumit Chaturvedi, Mengwei Ren, Yannick Hold-Geoffroy, Jingyuan Liu, Julie Dorsey, and Zhixin Shu. Synthlight: Portrait relighting with diffusion model by learning to re-render synthetic faces. arXiv preprint arXiv:2501.09756, 2025.
- [7] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023.
- [8] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024.
- [9] Weifeng Chen, Jiacheng Zhang, Jie Wu, Hefeng Wu, Xuefeng Xiao, and Liang Lin. Id-aligner: Enhancing identity-preserving text-to-image generation with reward feedback learning. arXiv preprint arXiv:2404.15449, 2024.
- [10] Kevin Clark, Paul Vicol, Kevin Swersky, and David J Fleet. Directly fine-tuning diffusion models on differentiable rewards. arXiv preprint arXiv:2309.17400, 2023.
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [12] Ye Fang, Zeyi Sun, Shangzhan Zhang, Tong Wu, Yinghao Xu, Pan Zhang, Jiaqi Wang, Gordon Wetzstein, and Dahua Lin. Relightvid: Temporal-consistent diffusion model for video relighting. arXiv preprint arXiv:2501.16330, 2025.
- [13] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557, 2024.
- [14] Rohit Gandikota, Joanna Materzyńska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept sliders: Lora adaptors for precise control in diffusion models. In European Conference on Computer Vision, pages 172–188. Springer, 2024.
- [15] Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. Knowledge distillation: A survey. International Journal of Computer Vision, 129(6):1789–1819, 2021.
- [16] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.
- [17] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Zhang, Bingbing Liu, and Ying-Cong Chen. Lotus: Diffusion-based visual foundation model for high-quality dense prediction. arXiv preprint arXiv:2409.18124, 2024.
- [18] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [19] Varun Jampani, Kevis-Kokitsi Maninis, Andreas Engelhardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan Popov, André Araujo, Ricardo Martin Brualla, Kaushal Patel, et al. Navi: Category-agnostic image collections with high-quality 3d shape and pose annotations. Advances in Neural Information Processing Systems, 36:76061–76084, 2023.
- [20] Haian Jin, Yuan Li, Fujun Luan, Yuanbo Xiangli, Sai Bi, Kai Zhang, Zexiang Xu, Jin Sun, and Noah Snavely. Neural gaffer: Relighting any object via diffusion. Advances in Neural Information Processing Systems, 37:141129–141152, 2024.
- [21] Hoon Kim, Minje Jang, Wonjun Yoon, Jisoo Lee, Donghyun Na, and Sanghyun Woo. Switchlight: Co-design of physics-driven architecture and pre-training framework for human portrait relighting.

- In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25096–25106, 2024.
- [22] Peter Kocsis, Vincent Sitzmann, and Matthias Nießner. Intrinsic image diffusion for indoor single-view material estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5198–5208, 2024.
- [23] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [24] Zhengfei Kuang, Yunzhi Zhang, Hong-Xing Yu, Samir Agarwala, Elliott Wu, Jiajun Wu, et al. Stanfordorb: a real-world 3d object inverse rendering benchmark. Advances in Neural Information Processing Systems, 36:46938–46957, 2023.
- [25] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023.
- [26] Xiaowen Li, Haolan Xue, Peiran Ren, and Liefeng Bo. Diffueraser: A diffusion model for video inpainting. arXiv preprint arXiv:2501.10018, 2025.
- [27] Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, Chun-Le Guo, and Ming-Ming Cheng. Amt: All-pairs multi-field transforms for efficient frame interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9801–9810, 2023.
- [28] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [29] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [30] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [31] Rohit Pandey, Sergio Orts-Escolano, Chloe Legendre, Christian Haene, Sofien Bouaziz, Christoph Rhemann, Paul E Debevec, and Sean Ryan Fanello. Total relighting: learning to relight portraits for background replacement. ACM Trans. Graph., 40(4):43–1, 2021.
- [32] Mengwei Ren, Wei Xiong, Jae Shin Yoon, Zhixin Shu, Jianming Zhang, HyunJoon Jung, Guido Gerig, and He Zhang. Relightful harmonization: Lighting-aware portrait background replacement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6452–6462, 2024.
- [33] Peiran Ren, Yue Dong, Stephen Lin, Xin Tong, and Baining Guo. Image based relighting using neural networks. ACM Transactions on Graphics (ToG), 34(4):1–12, 2015.
- [34] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.
- [35] Aliaksandr Siarohin, Stéphane Lathuilière, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. First order motion model for image animation. Advances in neural information processing systems, 32, 2019.
- [36] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008–3021, 2020.
- [37] Yuiga Wada, Kanta Kaneda, Daichi Saito, and Komei Sugiura. Polos: Multimodal metric learning from human feedback for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13559–13568, 2024.
- [38] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai

- Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [39] Xiang Wang, Shiwei Zhang, Han Zhang, Yu Liu, Yingya Zhang, Changxin Gao, and Nong Sang. Videolcm: Video latent consistency model. arXiv preprint arXiv:2312.09109, 2023.
- [40] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1481–1490, 2024.
- [41] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [42] Yu-Ying Yeh, Koki Nagano, Sameh Khamis, Jan Kautz, Ming-Yu Liu, and Ting-Chun Wang. Learning to relight portrait images via a virtual light stage and synthetic-to-real adaptation. ACM Transactions on Graphics (TOG), 41(6):1–21, 2022.
- [43] Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. Instructvideo: Instructing video diffusion models with human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6463–6474, 2024.
- [44] Chong Zeng, Yue Dong, Pieter Peers, Youkang Kong, Hongzhi Wu, and Xin Tong. Dilightnet: Finegrained lighting control for diffusion-based image generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024.
- [45] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. International Journal of Computer Vision, pages 1–15, 2024.
- [46] Kai Zhang, Fujun Luan, Qianqian Wang, Kavita Bala, and Noah Snavely. Physg: Inverse rendering with spherical gaussians for physics-based material editing and relighting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5453–5462, 2021.
- [47] Longwen Zhang, Qixuan Zhang, Minye Wu, Jingyi Yu, and Lan Xu. Neural video portrait relighting in real-time via consistency modeling. In Proceedings of the IEEE/CVF international conference on computer vision, pages 802–812, 2021.
- [48] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.
- [49] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport. In The Thirteenth International Conference on Learning Representations, 2025.
- [50] Yuanqing Zhang, Jiaming Sun, Xingyi He, Huan Fu, Rongfei Jia, and Xiaowei Zhou. Modeling indirect illumination for inverse rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18643–18652, 2022.
- [51] Yuxin Zhang, Dandan Zheng, Biao Gong, Jingdong Chen, Ming Yang, Weiming Dong, and Changsheng Xu. Lumisculpt: A consistency lighting control network for video generation. arXiv preprint arXiv:2410.22979, 2024.

- [52] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and KwanYee K Wong. Uni-controlnet: All-in-one control to text-to-image diffusion models. Advances in Neural Information Processing Systems, 36:11127–11150, 2023.
- [53] Jianbin Zheng, Minghui Hu, Zhongyi Fan, Chaoyue Wang, Changxing Ding, Dacheng Tao, and Tat-Jen Cham. Trajectory consistency distillation. arXiv e-prints, pages arXiv–2402, 2024.
- [54] Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe. Bilateral reference for high-resolution dichotomous image segmentation. CAAI Artificial Intelligence Research, 3:9150038, 2024.
- [55] Hao Zhou, Sunil Hadap, Kalyan Sunkavalli, and David W Jacobs. Deep single-image portrait relighting. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7194–7202, 2019.
- [56] Shangchen Zhou, Chongyi Li, Kelvin CK Chan, and Chen Change Loy. Propainter: Improving propagation and transformer for video inpainting. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10477–10486, 2023.
- [57] Yujie Zhou, Jiazi Bu, Pengyang Ling, Pan Zhang, Tong Wu, Qidong Huang, Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, et al. Light-a-video: Training-free video relighting via progressive light fusion. arXiv preprint arXiv:2502.08590, 2025.

Appendix of UniLumos

In this appendix, we provide additional details to complement the main paper. First, we explain the motivation behind introducing physics-plausible feedback in Sec.A. Next, we present the detailed pipeline of our proposed LumosData relighting data construction process in Sec.B. Then, Sec.C contains three additional experimental results on the public dataset (Sec. C.1) and the LumosBench (Sec. C.2). Sec.D provides additional qualitative results to further illustrate the effectiveness of UniLumos. Finally, Sec. E discusses the limitations of our method.

- A Physics-Plausible Feedback

To further clarify the motivation and design behind our physics-plausible feedback mechanism, we present a breakdown of key questions and considerations addressed during its development.

- Q1: What is the motivation for introducing physical constraints in relighting?

- A1: The primary goal of relighting is to generate visually plausible illumination under new lighting conditions. However, many diffusion-based methods lack explicit physical modeling, leading to artifacts such as overexposed highlights, misaligned shadows, or inconsistent light directions. Introducing physical constraints serves as a refinement mechanism that aligns generated light with the scene’s underlying geometry. This helps enforce realism and spatial consistency in illumination, which is especially critical under complex lighting or HDR scenarios.

Q2: Why are depth and normal maps chosen as the targets for physical supervision?

- A2: Depth and surface normals are among the most accessible and general-purpose dense scene attributes. By design, these estimations intentionally suppress fine-scale lighting effects to focus on intrinsic geometry. This makes them ideal for supervising relighting, where the goal is to decouple geometry from illumination and enforce spatial structure in lighting behavior. In the proposed UniLumos, we align the predicted lighting with reference geometry by minimizing the L2 norm error between generated and reference-aligned depth/normal maps via a pre-trained dense estimation model (e.g., Lotus [17]) with frozen parameters. This provides a simple yet effective metric to quantify physical plausibility.

Q3: Why are alternative physical signals—such as albedo, shadow, or material—not used instead?

- A3: While albedo, shadow masks, and material properties can provide rich supervision, they come with significant drawbacks. Albedo and shadow estimation often rely on inverse rendering and suffer from domain sensitivity or ambiguity. Material annotations are expensive and dataset-dependent. Moreover, many of these properties are entangled with illumination, making them less reliable as supervisory signals. In contrast, depth and normals can be predicted from monocular images with high availability and generalize well across scenes, offering a favorable balance between supervision quality and computational cost.

Q4: Why are depth and normal maps used as training-time constraints rather than as model inputs?

- A4: While it is possible to condition the model directly on estimated depth and normal maps, doing so increases the input dimensionality and model complexity. It would also introduce a dependency on external estimators during inference, complicating the pipeline and potentially propagating errors. Instead, we use them as supervision signals during training. This design keeps the inference pipeline simple—relying only on image and lighting condition inputs—while still allowing the model to learn geometry-aware behaviors. The supervision acts as a form of inductive bias, guiding the model toward physically plausible outputs without requiring additional input channels at the test phase.

- B Details of Datasets

- Step 1: Subject Mask. Given an input video Vreal ∈ R[T+1,H,W,3], we first extract per-frame subject masks M ∈ R[T+1,H,W] using BiRefNet [54]. These subject masks allow us to isolate the target subject foreground and the target background.

##### Table 3 Lighting-related textual prompts used in Lumos Augmentation from IC-Light [49]. Each prompt can be combined with different canonical light directions during training.

ID Lighting Prompt Example Light Direction

- 1 sunshine from window None

- 2 neon light, city Left Light

- 3 sunset over sea Right Light

- 4 golden time Top Light

- 5 sci-fi RGB glowing, cyberpunk Bottom Light

- 6 natural lighting

- 7 warm atmosphere, at home, bedroom

- 8 magic lit

- 9 evil, gothic, Yharnam

- 10 light and shadow

- 11 shadow from window

- 12 soft studio lighting

- 13 home atmosphere, cozy bedroom illumination

- 14 neon, Wong Kar-wai, warm

- Step 2: Lumos Augmentation. To simulate diverse lighting degradations for training, we relight each subject sequence under multiple lighting conditions using a pre-trained 2D relighting model, such as IC-Light [49]. This operation is applied independently to each frame of the subject region, resulting in a degenerated video

Vdeg ∈ R[T+1,H,W,3]. To generate rich illumination variations, we refer to the description of light and shadow given by IC-Light [49], as listed in Tab. 3, which serve as the semantic guidance for image-level relighting and the light source directions. For each input video, we randomly sample 5 prompts and 3 directions, forming 5 × 3 = 15 unique prompt-direction pairs. The relighting is applied only to the subject region, extracted using the subject masks M ∈ R[T+1,H,W] from Step 1. Notably, we randomly sample one degradation condition from the 15 prompt-direction combinations for each subject in each iteration. This strategy reduces training cost while exposing the model to diverse illumination patterns, thereby improving generalization.

- Step 3: Gaussian Background. To provide external lighting context during training, we generate a background

video Vbg ∈ R[T+1,H,W,3] to accompany the relit subject. Instead of relying on complex inpainting-based synthesis (e.g., ProPainter [56], DiffuEraser [26]), we adopt a simple yet effective strategy by filling the background with either pure color or Gaussian noise. This design avoids injecting semantic or structural priors, allowing the model to focus solely on illumination learning.

Specifically, for each frame t ∈ [1,T + 1] and channel c ∈ {R,G,B}, we first define the background region using the subject mask Mt ∈ RH×W obtained in Step 1. Let Ωtbg = {(i,j) | Mt(i,j) = 0} denote the set of background pixels. We compute the mean and standard deviation of background pixel intensities as:

 

1 |Ωtbg|

µtc =

Vt(i,j,c),

(i,j)∈Ωtbg

(8)

1 |Ωtbg|

(Vt(i,j,c) − µtc)2

σct =



(i,j)∈Ωtbg

We then fill the background with pixel-wise samples from a Gaussian distribution:

Vbgt (i,j,c) ∼ N(µtc,(σct)2), ∀(i,j) ∈ Ωtbg. (9)

This procedure ensures that the background region maintains a similar color distribution to the original video while avoiding structural detail that may bias learning. As shown in Fig. 7, for comparison, we also test a variant that uses pure-color background, where each background pixel is set to µtc (i.e., σct = 0). In practice, we observe that such statistically consistent placeholders—particularly Gaussian-filled ones—accelerate early-stage

Frame 1 Frame 24 Frame 48 Frame 1 Frame 24 Frame 48

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Inpainting Pure

Gaussian

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Inpainting Gaussian

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Inpainting Pure

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Inpainting

- Figure 7 Comparison of background inpainting strategies of four representative cases. Here, Gaussian Inpainting fills the background using random noise sampled with the same mean and variance as the subject region, ensuring statistical consistency. Pure Inpainting directly fills the background with a uniform color (i.e., gray), without modeling spatial or color variation. The Gaussian strategy provides more realistic signal distribution and accelerates early-stage convergence in training.

convergence during training. We attribute this to the reduced visual complexity and improved normalization behavior, which make the model less sensitive to background variation. The resulting Vbg serves as a clean, distribution-aligned conditioning signal for the relighting network.

- Step 4: Caption Augmentation. In addition to relighting augmentation, we generate lighting-aware captions to provide rich semantic supervision aligned with physical lighting behavior. Specifically, we leverage Qwen2.5VL [1], a vision-language model with fine-grained visual reasoning capabilities, to analyze each input video and generate structured captions describing its lighting attributes. The input to Qwen2.5-VL consists of the original video and its corresponding scene-level caption. We then apply a custom-designed prompt (see Listing 1) to steer the model toward predicting six categories of lighting-related labels as shown in Tab. 4, including all subcategories and their physical interpretations. The output of this process is a structured caption C for each video, formatted as a dictionary mapping the six categories to their predicted labels (see example in Listing 1). These structured captions serve as auxiliary supervision and evaluation labels in later stages, helping the model better align with interpretable physical lighting semantics. They also enhance downstream controllability and facilitate attribute-based retrieval or evaluation.

- Table 4 Classification criteria and definitions for light-related scene attributes.

Primary Category Subcategory Definition

Front Light The light source is positioned directly in front of the subject, illuminating it head-on. Side Light The light source is positioned at a 90-degree or 45-degree angle to the subject, coming from the side. Back Light The light source is located behind the subject, directed towards the camera.

Direction of Light

Top Light The light source is positioned directly above the subject, casting light downwards. Bottom Light The light source is positioned below the subject, casting light upwards.

Split Light The light source illuminates one side of the subject while leaving the other side in shadow. Ambient Light Without Clear Direction Ambient light is non-directional, uniformly illuminating the environment from multiple sources.

Natural Light Illumination from nature without human intervention, varying with time of day, weather, and location.

Light Source Type

Artificial Light Human-made light sources (e.g., bulbs, LEDs) used in indoor/outdoor spaces for functional or artistic effects. Rendering Light Digitally simulated light in CGI, games, or animations using techniques like ray tracing.

Glare Extremely bright light over 1000 lumens that can cause discomfort or obscure detail. Moderate Balanced lighting (200–1000 lumens), suitable for most activities and comfortable viewing.

Light Intensity

Dim Low lighting under 200 lumens, often cozy but may reduce visibility and detail recognition.

Cool Tone 5000K–10000K; bluish hues, common in daylight or overcast scenes. Neutral 4000K–5000K; balanced light with no strong blue or yellow tint.

Color Temperature

Warm Tone 2000K–4000K; reddish or yellowish hues, typical in sunrise/sunset or indoor lighting.

Static Light Illumination remains constant in both intensity and direction over time. Dynamic Light (Intensity Changing) Light intensity changes gradually over time (e.g., dawn to daylight).

Light Changes in Time

Dynamic Light (Moving Source) Direction of light changes due to movement of light source (e.g., headlights, stage lights).

Transmission (Glass) Light passes through transparent materials like glass, with possible scattering or absorption. Refraction/Reflection (Water Surface, Mirror) Light bends or reflects at water or mirror surfaces, altering its direction.

Optical Phenomena

Scattering (Fog Effect) Light diffuses through particles like fog or mist, reducing visibility. None No significant optical phenomena are observed in the scene.

SYSTEM_PROMPT = """ You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while

being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.

""" PROMPT = """ Role: You are an expert in image/video light and shadow analysis, good at analyzing light and

shadow from multiple angles.

... Tasks: Analyze the input image/video, provide corresponding classification results for the

following multiple categories, and return them in the specified output format.

- 1. Direction of Light: Task 1: Analyze the image and classify the light source direction as front light, side light,

back light, top light, bottom light, or split light. Identify the angle of the light source relative to the subject, and describe its effect on shadow formation.

- 2. Light Source Type: Task 2: Analyze the image and classify the light source type as either Natural Light, Artificial

Light, or Rendering Light.

- 3. Light Intensity: Task 3: Analyze the image/video to assess the light intensity present. Classify the light

intensity into three categories: Glare, Moderate, and Dim. Special attention should be given to situations where bright light sources may create a glaring effect even in otherwise dim environments.

- 4. Color Temperature: Task 4: Analyze the image/video to assess the color temperature present. Classify the color

temperature into three categories: Cool Tone, Neutral, and Warm Tone.

- 5. Light Changes in Time: Task 5: Analyze the video to assess light changes over time. Classify the light changes into two

main categories: Static Light and Dynamic Light. For Dynamic Light, further categorize it into two subtypes: Intensity Gradient and Moving Light Source.

- 6. Optical Phenomena: Task 6: Analyze the image/video with a focus on the specific scene to assess the optical

phenomena present. Pay close attention to scenarios involving glass, water surfaces, mirrors , and fog. Classify the phenomena into the following categories: Transmission (Glass), Refraction/Reflection (Water Surface, Mirror), Refraction/Reflection (Mirror), Scattering ( Fog Effect), and None.

Guidelines:

- 1. Accuracy: Assign each tag to the most appropriate category and subcategory.

- 2. Multiple Tags: If an action fits multiple categories, assign all relevant tags.

- 3. Comprehensiveness: Capture all detectable dynamic attributes without omissions.

- 4. JSON Validity: Ensure the output JSON is correctly formatted and adheres to the specified structure.

Example Output: {

"Direction of Light": "Front Light",

"Light Source Type": "Artificial Light", "Light Intensity": "Moderate", "Color Temperature": "Cool Tone", "Light Changes in Time": "Dynamic Light (Intensity Changing Light)", "Optical Phenomena": "Transmission (Glass)"

} """

Listing 1 Prompt Definition

### C Additional Experimental Results

- C.1 Additional Main Results

To further demonstrate the model’s generalization to non-human scenes, we conducted additional evaluations on two public object-centric relighting benchmarks: StanfordOrb [24] and Navi [19]. These datasets include objects and sculptures under a variety of lighting environments and are completely disjoint from our training data. StanfordOrb contains canonical 3D scanned objects such as the Stanford bunny and dragon, while Navi includes a wide range of everyday objects like containers, toys, and mugs. As shown in Tab. 5 (StanfordOrb) and Tab. 6 (Navi), UniLumos achieves state-of-the-art results across perceptual (LPIPS), structural (SSIM), and physical (R-Motion) metrics, despite the significant domain gap and without any test-time fine-tuning, outperforming all baselines.

Table 5 Quantitative comparison on the StanfordOrb dataset. Bold number indicate the best performance.

Model PSNR ↑ SSIM ↑ LPIPS ↓ R-Motion ↓

IC-Light Per Frame [49] 24.132 0.914 0.126 1.742 Light-A-Video [57] + CogVideoX [41] 25.617 0.923 0.108 1.279 Light-A-Video [57] + Wan2.1 [38] 25.784 0.926 0.104 1.241

UniLumos 26.512 0.934 0.097 1.103

Table 6 Quantitative comparison on the Navi dataset. Bold number indicate the best performance.

Model PSNR ↑ SSIM ↑ LPIPS ↓ R-Motion ↓

IC-Light Per Frame [49] 22.021 0.883 0.125 1.974 Light-A-Video [57] + CogVideoX [41] 23.912 0.891 0.121 1.378 Light-A-Video [57] + Wan2.1 [38] 23.474 0.903 0.116 1.341

UniLumos 24.977 0.911 0.120 1.203

- C.2 LumosBench: An Attribute-level Controllability Benchmark

To evaluate the fine-grained controllability of lighting generation, we introduce LumosBench, a structured benchmark that targets six core illumination attributes defined in our annotation protocol. Unlike prior works that treat lighting holistically or implicitly, LumosBench provides a disentangled, attribute-level evaluation, enabling precise diagnosis of model behavior under controllable lighting conditions.

Specifically, we construct a set of 2k test prompts, each consisting of a video and a structured caption designed to isolate one lighting attribute at a time, while holding other variables constant. These prompts span six categories—direction, light source type, intensity, color temperature, temporal dynamics, and optical phenomena—with multiple subtypes per category (e.g., front/side/back for direction). This design facilitates controlled and interpretable evaluation across lighting axes that are often conflated in prior datasets.

##### Table 7 Quantitative comparison of the attribute-level controllability. Bold number indicate the best performance.

Model #Params Direction Light Source Type Intensity Color Temperature Temporal Dynamics Optical Phenomena Avg. Score General Models

LTX-Video [16] 1.9B 0.794 0.644 0.487 0.708 0.487 0.403 0.587 CogVideoX [41] 5.6B 0.837 0.692 0.552 0.739 0.532 0.449 0.634 HunyuanVideo [23] 13B 0.863 0.741 0.599 0.802 0.655 0.481 0.690 Wan2.1[38] 1.3B 0.842 0.685 0.436 0.741 0.504 0.433 0.607 Wan2.1[38] 14B 0.871 0.794 0.674 0.829 0.737 0.505 0.735

Specialized Models

IC-Light Per-Frame [49] 0.9B 0.793 0.547 0.349 0.493 0.284 0.339 0.468 Light-A-Video [57] + CogVideoX[41] 2.9B 0.787 0.581 0.327 0.536 0.493 0.373 0.516 Light-A-Video [57] + Wan2.1[38] 2.2B 0.801 0.603 0.361 0.582 0.557 0.412 0.553

UniLumos w/o lumos captions 1.3B 0.868 0.774 0.529 0.798 0.543 0.457 0.662 UniLumos 1.3B 0.893 0.847 0.832 0.813 0.662 0.592 0.773

To assess alignment between intended and generated lighting attributes, we use the vision-language model Qwen2.5-VL [1] to analyze relit outputs and classify whether the target attribute is correctly expressed. Each dimension is scored independently, and the final controllability score is the average across all six dimensions.

General vs. Specialized Models. Tab. 7 presents results for both general-purpose and task-specific relighting models. This benchmark allows us not only to assess overall lighting quality but also to dissect a model’s ability to interpret and respond to individual lighting controls. Among general models, Wan 14B shows the highest raw capability, demonstrating the strength of large-scale pretraining for visual generation. Notably, our fine-tuned Wan 1.3B variant achieves substantial gains across all six lighting dimensions, surpassing even much larger models. This highlights the benefit of relighting-specific supervision: fine-tuning on LumosData with structured lighting annotations significantly enhances the model’s ability to reason about illumination in a controllable and disentangled manner.

By contrast, specialized relighting models consistently underperform despite being designed for lighting manipulation. This is primarily due to the limitations of their base architectures—typically smaller and trained from scratch or on narrow domains—resulting in weaker generalization to diverse lighting attributes. While they may encode some prior knowledge of lighting physics (e.g., through latent constraints), their restricted modeling capacity hinders semantic alignment with user-intended lighting conditions. These findings underscore the importance of starting from a strong pre-trained backbone and introducing structured, high-level lighting supervision to achieve controllable and physically plausible relighting.

Effectiveness of Structured Captions. Within the specialized group, we conduct an ablation to assess the importance of our proposed lumos captions. w/o lumos captions uses only vanilla scene-level captions during training, omitting structured lighting tags. The performance drop—particularly in controllable dimensions like intensity and optical phenomena—confirms that our semantic annotations play a key role in teaching the model fine-grained illumination control. Compared to strong baselines, UniLumos achieves superior scores across nearly all dimensions, demonstrating the impact of LumosBench in pushing model understanding and control of illumination.

- D Additional Result Visualization

We present additional image relighting results in Fig. 8. We present additional background-conditioned video relighting results in Fig. 9 and Fig. 10.

- E Limitation and Future Work

UniLumos is still limited by a broader challenge—achieving physically precise and controllable relighting. UniLumos enforces geometry-aware consistency (e.g., shadows aligned with depth and normals) but does not yet produce physically quantifiable lighting outputs such as radiance or illuminance. Future work may explore finer control over lighting, including editable key lights, intensity ramps, and environmental reflections.

Input Relight

Input Relight

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

Bottom Left

Bottom Right

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

None Bottom

Left Right

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

Left Top

Right Left

- Figure 8 UniLumos performs physically plausible image relighting, conditioned on textual prompts.

Frame 1 Frame 24 Frame 48

[Figure 197]

[Figure 198]

[Figure 199]

Input

Relight

Reference Background Video

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

- Figure 9 UniLumos performs physically plausible video relighting, conditioned on reference videos.

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

Frame 1 Frame 24 Frame 48

Input

[Figure 230]

Reference Background Video

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Relight

- Figure 10 UniLumos performs physically plausible video relighting, conditioned on reference videos.

