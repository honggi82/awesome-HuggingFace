# arXiv:2508.03694v1[cs.CV]5Aug2025

## LongVie: Multimodal-Guided Controllable Ultra-Long Video Generation

Jianxiong Gao2,5, Zhaoxi Chen3, Xian Liu4 Jianfeng Feng2, Chenyang Si1,†, Yanwei Fu2,†, Yu Qiao5, Ziwei Liu3,† 1Nanjing University, 2Fudan University, 3S-Lab, Nanyang Technological University 4NVIDIA, 5Shanghai AI Laboratory https://vchitect.github.io/LongVie-project/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Input Image

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Dense Sparse

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

…

…

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

###### Multi-modal Guidance

[Figure 33]

[Figure 34]

00:10 00:20 00:30 00:40 00:50 01:00

### LongVie

Time

[Figure 35]

Figure 1: LongVie is a controllable ultra-long video generation framework guided by both dense and sparse control signals, with a degradation-aware training strategy to balance the contribution of the modalities. It applies global normalization to the control signals and employs unified initialization noise to autoregressively generate videos lasting up to one minute.

#### Abstract

Controllable ultra-long video generation is a fundamental yet challenging task. Although existing methods are effective for short clips, they struggle to scale due to issues such as temporal inconsistency and visual degradation. In this paper, we initially investigate and identify three key factors: separate noise initialization, independent control signal normalization, and the limitations of single-modality guidance. To address these issues, we propose LongVie, an end-to-end autoregressive framework for controllable long video generation. LongVie introduces two core designs to ensure temporal consistency: 1) a unified noise initialization strategy that maintains consistent generation across clips, and 2) global control signal normalization that enforces alignment in the control space throughout the entire video. To mitigate visual degradation, LongVie employs 3) a multi-modal control framework that integrates both dense (e.g., depth maps) and sparse (e.g., keypoints) control signals, complemented by 4) a degradation-aware training strategy that adaptively balances modality contributions over time to preserve visual quality. We also introduce LongVGenBench, a comprehensive benchmark consisting of 100 high-resolution videos spanning diverse real-world and synthetic environments,

†: Co-corresponding authors.

Preprint. Under review.

each lasting over one minute. Extensive experiments show that LongVie achieves state-of-the-art performance in long-range controllability, consistency, and quality.

#### 1 Introduction

Recent advancements in video generation have been significantly driven by the availability of large-scale datasets and the development of powerful generative architectures, particularly diffusion models [15, 28, 42, 26, 9]. These innovations have greatly enhanced the video generation, enabling models such as CogVideoX [37], HunyuanVideo [32], Kling [33], Sora [25], and Wanx2.1 [35] to generate high-quality videos with text prompts.

Despite these successes, a major challenge in video generation remains the ability to achieve precise control, ensuring that the generated video aligns seamlessly with the user’s creative vision. Recent efforts have attempted to overcome this limitation by integrating control framework [40, 7, 12, 23, 16, 11] into the generation process in order to achieve more refined and customizable outputs. However, these methods primarily focus on the controllable generation of short video clips, with limited exploration into the challenges associated with generating longer, controllable videos, such as those lasting up to one minute. In practical applications, the ability to generate long, coherent, and visually consistent videos is critical, yet it remains a fundamentally complex and unresolved problem.

In this paper, we focus on the problem of controllable long video generation. Directly generating one-minute-long controllable videos typically requires considerable computational resources. Hence, we begin by analyzing the applicability of autoregressive generation to long video synthesis, wherein short video segments are generated sequentially, each initialized from the final frame of the preceding segment. While this approach offers a more tractable solution, it exhibits notable limitations when extended to continuous long-form generation. As shown in Fig. 2, as the video length increases with each successive iteration, two main aspects become evident: 1) temporal inconsistency between adjacent clips, resulting in noticeable scene transitions or flickering quality; 2) visual degradation, where each subsequent clip exhibits reduced overall visual fidelity compared to the previous one. These two aspects highlight the challenges in maintaining both visual continuity and high quality over extended video sequences when relying solely on autoregressive mechanism.

Building upon this analysis, we further investigate the underlying causes of these two issues. We find that temporal inconsistency primarily stems from two factors: the independent normalization of control signals across clips and the separate noise initialization employed in each generation step. Perclip normalization disrupts global coherence in the control space, leading to inconsistencies between adjacent frames—especially when the control signals (e.g., depth or motion maps) lack sufficient scene-level alignment. Additionally, varying noise initialization inputs across clips introduces unpredictable variations in content, further weakening temporal continuity. On the other hand, visual degradation is largely attributed to the limitations of single-modality control. Dense signals provide fine-grained structural guidance but may dominate generation process, while sparse signals offer high-level semantics but lack adequate spatial detail. Over time, reliance on a single modality causes accumulated errors and a progressive decline in visual output quality.

To address both challenges, we propose LongVie, a controllable long video generation framework built on an autoregressive paradigm. LongVie introduces two core components to enhance temporal consistency: 1) unified noise initialization, which ensures consistent generative dynamics across clips, and 2) global control signal normalization, which maintains alignment in the control space throughout the entire video. These two techniques work together to reduce abrupt transitions and flickering artifacts, resulting in a temporally coherent generation process. To mitigate visual degradation, LongVie employs a 3) multi-modal control framework that integrates both dense (e.g., depth maps) and sparse (e.g., keypoints) modalities for complementary guidance, along with a 4) degradation-aware training scheme that balances their contributions over time. Through these designs, LongVie enables the generation of minute-long videos that are not only temporally consistent but also maintain high visual fidelity under fine-grained controllability.

To evaluate the effectiveness of controllable long video generation, we construct a dataset of 100 high-quality videos, each with a duration of at least one minute. The videos cover a wide range of scenes, including both real-world environments and game-based scenarios, ensuring diversity in content and motion. We refer to this dataset as LongVGenBench. We evaluate LongVie on

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Reference Video

… …

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Generated Video

Color Drift

… …

Spatial Degradation

Input Image

Temporal Inconsistency Visual Degradation

Figure 2: Temporal Inconsistency and Quality Degradation. These are the primary limitations encountered when applying current controllable models to long video generation.

LongVGenBench in terms of long-term visual quality and temporal consistency. The results show that LongVie achieves leading performance in controllable long video generation.

246，250，253 449，453

Our main contributions are as follows:

- • We present a comprehensive analysis of the limitations in existing controllable video generation models for long videos, identifying two key challenges: long-term temporal inconsistency and visual degradation. To address these issues, we propose LongVie, the first autoregressive framework for controllable long video generation.
- • To improve visual quality, we propose a multi-modal control mechanism that integrates dense and sparse control signals to exploit their respective advantages, alongside a degradationaware training strategy that balances their contributions.
- • To enhance temporal consistency, we leverage unified noise initialization and global control signals normalization for a world-consistent generative dynamics over time steps.
- • We introduce LongVGenBench, an evaluation dataset for controllable long video generation, comprising 100 diverse, high-quality videos, each lasting at least one minute.

#### 2 Methodology

Preliminary: Diffusion Models. Diffusion models are a strong type of video generator that turn random noise into realistic videos by learning how to remove the noise step by step. So, the model ϵθ learns to undo a process that adds noise to data. During training, it is trained to predict the noise added at each step:

L = Ex,ϵ∼N(0,1),t ∥ϵ − ϵθ(xt,t)∥22 , (1)

where xt, ϵ, x0 denote the noisy data at time t, the true noise, and the original data, respectively. To save computing power, Latent Diffusion Models [28] work in a smaller, compressed space. An

autoencoder turns x into a latent code z = E(x), and the model learns to predict noise in zt.

Overview. We extend CogVideoX [37] with a ControlNet-style architecture [39] to incorporate the external control signals. A lightweight control branch, partially shared with the base model, processes the control signals. While effective for short video synthesis, most controllable diffusion-based models, including CogVideoX and its variants, are not designed to handle long-duration generation, such as one-minute sequences. Generating such long videos in a single forward pass is computationally prohibitive. As a result, a common practice is to generate videos in an autoregressive manner—producing short clips sequentially, with each initialized from the final frame of the previous one. In our implementation, we follow this approach using a depth-conditioned variant of CogVideoX [37]. However, as discussed in the following subsection and illustrated in Fig. 2, this strategy introduces two major challenges: (1) temporal inconsistency across consecutive clips, and (2) progressive quality degradation due to accumulated errors over time.

##### 2.1 Rethinking Controllable Generation of Long Videos

Temporal Inconsistencies. To investigate the source of temporal inconsistencies, we analyze the input signals used in controllable video generation models. In models that rely on external control,

such as depth maps (see Fig. 3 (a)), these signals are usually normalized independently for each clip. We find that this per-clip normalization introduces inconsistencies across clips. For example, the same scene may appear with different depth values in consecutive clips. As a result, the model receives mismatched guidance across clips, which distorts its perception of scene geometry and motion continuity. This leads to temporal artifacts such as unnatural zooming or abrupt viewpoint changes (Fig.3 (a)). These findings suggest that independent normalization breaks the alignment of control signals across clips, especially when the signals lack global context or a consistent reference scale, ultimately causing visible inconsistencies between clips.

Inspired by our analysis of control signal normalization, we further examine the impact of noise initialization on temporal consistency. In diffusion-based video generation, the initial noise plays a critical role in determining the overall structure and motion of the output. We observe that temporal inconsistencies frequently occur at the beginning of each generated clip, suggesting a strong correlation between noise initialization and temporal disruptions. In standard autoregressive generation, each clip is sampled from a different random noise input. This variation introduces inconsistencies in motion, appearance, or scene layout across clips, even when the control signals remain aligned. Our empirical study, illustrated in Fig. 3 (b), confirms this effect: Clips with larger differences in initialization noise—measured by the root mean squared error (RMSE) relative to that of the first clip—tend to exhibit more noticeable temporal inconsistencies, as shown by the Structural Similarity Index (SSIM) curves in Fig. 3 (b).

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

###### Clip Normalization

###### Global Normalization

[Figure 54]

[Figure 55]

[Figure 56]

Clip Normalization

Moving Away

[Figure 57]

[Figure 58]

[Figure 59]

Global Normalization

Temporal Consistency

Previous Clip Current Clip Average Depth Curve

(a) Impact of Normalization Strategies on Temporal Consistency

[Figure 60]

[Figure 61]

[Figure 62]

Unified Noise

RMSE:1.4162 RMSE:1.4137 RMSE:1.4127

[Figure 63]

[Figure 64]

[Figure 65]

Noise RMSE: 1.416

| |
|---|

| |
|---|

RMSE:1.4123

SSIM Curve in Inconsistent Regions

Previous Clip Current Clip

(b) Sensitivity to Noise Variance in Initialization

Figure 3: Analysis of temporal inconsistency. Temporal inconsistency arises from clip-wise normalization and random noise initialization, while global normalization and unified noise alleviate these issues.

In summary, temporal inconsistencies in long video generation are primarily caused by misaligned control signals across clips and variations in initialization noise, both of which disrupt the continuity of motion and appearance.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

PointDepthD&P

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

- Figure 4: Visual degradation caused by single-modality control. Depth-only and point-only settings show noticeable degradation, while D&P, combining both modalities, alleviates the issue.

Visual Quality Degradation. Building on our analysis of temporal inconsistency, we further investigate the issue of visual quality degradation in long video generation. Controlling long video generation using per-frame signals is a practical strategy to maintain stability and visual quality. However, different control modalities come with inherent trade-offs that limit their effectiveness over extended sequences. Taking depth as an example of a dense modality, it provides pixel-level structural information across frames. While effective for preserving local geometry, it offers limited control over nearby or distant regions and lacks the capacity to represent high-level semantics such as object motion or scene dynamics. As shown in Fig. 4, these limitations result in artifacts and degraded quality, especially in complex scenes. In contrast, point-based control is a sparse modality

###### Unified Initialization

…

Global Normalized MM-Control DiT (Trainable Copy)

[Figure 84]

[Figure 85]

"

Trainable

[Figure 86]

[Figure 87]

Noise"! !

Frozen

❄

[Figure 88]

[Figure 89]

%&'(&%$#

MM Block

[Figure 90]

!"#$!$#

[Figure 91]

!

Dense Block

[Figure 92]

[Figure 93]

❄

Depth Video /-(!

[Figure 94]

)$*+(&%$#

E

C

… …

[Figure 95]

!

Zero Linear

GlobalNormalization

-!$# ∈ ℝ)×+×,×-

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

!

Sparse Block

%&'(&%

,&

Generated video -! Point Video -.'!

!"#$!

Denoising DiT

Latent Noise

)$*+(&%

[Figure 100]

[Figure 101]

❄

-! ∈ ℝ)×+×,×-

[Figure 102]

❄

[Figure 103]

❄ ❄

[Figure 104]

Padding 0

… …

DiT Block

E C D

%&'(&%"#

Last frame in -!$#

!"#$!"#

Prompt #: The video depicts a tranquil autumn landscape on horseback …

[Figure 105]

❄

)$*+(&%"#

Generation of !!" clip

T5

…

- Figure 5: Framework of LongVie. We adopt dense and sparse control signals as scene guidance to generate controllable long videos in an autoregressive manner. We also apply global normalization and unified initialization noise to improve temporal consistency during the generation process.

that captures semantic cues by specifying a few keypoints. While it effectively guides motion and object structure, its sparse nature makes it sensitive to scene changes and less reliable in maintaining semantic alignment across frames. These limitations show that neither dense nor sparse control alone is sufficient for consistent long video generation. When control signals fail to align with the evolving scene, visual quality degrades progressively.

##### 2.2 LongVie Framework

To address the aforementioned challenges, we propose LongVie, a framework for controllable long video generation, as illustrated in Fig. 5. At the core of our model is a multi-modal framework that leverages both dense and sparse controls to guide the scene effectively. In the following paragraphs, we will first introduce the general framework, and then present the effective training and inference strategies to solve the issues mentioned in Sec. 2.1.

Multi-Modal Control Injection. Specifically, we adopt depth maps as dense control signals and point maps as sparse signals, leveraging the detailed structural information provided by depth and the high-level semantic cues captured by point trajectories. To construct the point map sequences, we follow the procedure in DAS [11], where a set of keypoints is tracked across frames and colorized according to their depth values.

Inspired by the ControlNet [39] architecture, we build our Multi-Modal Control DiT by duplicating the initial layers of the pre-trained CogVideoX DiT and incorporating multi-modal conditioning inputs while keeping the base model frozen. Specifically, we freeze the parameters θ of the original DiT blocks and construct two trainable branches, each corresponding to a distinct control modality: dense (depth) and sparse (point). These branches, denoted as FD(·;θD) and FP(·;θS) , are lightweight copies of the original DiT layers with parameters θD and θP , respectively. Each processes the corresponding encoded control signal cD or cP. To integrate the control branches into the base generation path, we adopt zero-initialized linear layers ϕl, as shown in Fig. 6, similar in spirit to the zero convolutions used in ControlNet [39]. We compare two variant structures and adopt (b) in this paper for its empirical stability. These layers allow the control signals to be injected additively into the main generation stream without affecting the model’s initial behavior, as they output zero at the start of training. The overall computation in the l-th controlled DiT block is defined as:

zl = Fl(zl−1) + ϕl(FDl (clD−1) + FPl(clP−1)), (2) where Fl is the frozen base DiT block, and FDl , FPl are the control-specific sub-blocks.

In practice, we duplicate the first 18 blocks of the original CogVideoX DiT to construct the MultiModal Control DiT. The remaining blocks remain untouched to preserve the generative capacity of the base model. During training, only the parameters in the control branches and the fusion layers ϕl are updated, while the backbone remains fixed. This setup allows us to introduce strong, flexible conditioning from both dense and sparse modalities, while maintaining the stability and pre-trained knowledge of the original model.

Full Copy

Half Copy Half Copy

Dense Block

Sparse Block

Dense Block

Sparse Block

DiT Block

Ctrl Block

DiT Block

DiT Block

Zero Linear Zero Linear

Zero Linear

Zero Linear

Dense Block

Sparse Block

Dense Block

Sparse Block

DiT Block

Ctrl Block

DiT Block

DiT Block

(a)

(b)

(c)

- Figure 6: Variants of Multi-Modal Control Integration. Compared to the standard ControlNet design (a), we present two variant structures (b) and (c) that integrate dense and sparse control signals.

Global Normalization. To reduce temporal inconsistencies caused by independently normalized control inputs, we adopt a global normalization strategy for the depth video. Specifically, we compute the 5th and 95th percentiles of all pixel values across the entire video sequence and use these as the global minimum and maximum normalization bounds. The depth values are then clipped to this range and linearly scaled to [0, 1]. This percentile-based normalization is robust to outliers and ensures that the depth values across all clips are on a consistent scale. As shown in Fig. 3 (top-right), this global normalization effectively reduces inter-clip variations and leads to more temporally aligned control signals. After normalization, the depth video is segmented into overlapping clips to match the autoregressive inference process and facilitate corresponding point map extraction.

Unified Noise Initialization. To further enhance temporal consistency, we use a shared noise initialization across all video segments during generation. Instead of sampling a different noise vector for each clip, we sample a single noise instance and apply it uniformly across the entire sequence. This unified noise serves as a consistent latent prior, reducing variations between adjacent clips that typically arise from independently sampled noise. As illustrated in Fig. 3 (bottom right), this approach significantly improves temporal coherence, mitigating flickering and promoting smooth transitions throughout the generated video.

Degradation Strategy for Modal Balance. While multi-modal control offers the potential for richer and more accurate video generation, simply combining dense and sparse control signals does not guarantee improved performance. In practice, we observe that dense signals such as depth tend to dominate the generation process, often suppressing the semantic and motion-level guidance provided by sparse signals like keypoints. This imbalance can lead to suboptimal visual quality, particularly in scenarios requiring high-level semantic alignment over time.

To address this issue, we propose a degradation-based training strategy designed to regulate the relative influence of dense control signals and encourage more balanced utilization of both modalities. This strategy weakens the dominance of the dense input through controlled perturbations at both the feature and data levels:

- 1) Feature-level degradation: During training, with probability α, we randomly scale the latent representation of the dense control input by a factor sampled uniformly from the range [0.05,1]. Accordingly, Equation 3 can be reformulated as:

zl = Fl(zl−1) + ϕl(α · FDl (clD−1) + FPl(clP−1)), (3)

This operation reduces the magnitude of the dense features, making the model more reliant on complementary information provided by the sparse modality. Over time, this encourages the network to learn a more balanced integration of both control sources.

- 2) Data-level degradation: Given a dense control tensor D ∈ RB×C×H×W, we apply degradation with probability β using two techniques: a) Random Scale Fusion: A set of spatial scales {1,1/2,...,1/2n} is predefined. One scale is randomly excluded, and the remaining scales are used to generate downsampled versions of the input, which are then upsampled to the original resolution. Each version is assigned a random weight normalized to sum to 1. Their weighted sum forms a fused depth map with multi-scale degradation and randomness, enhancing robustness to spatial variation. b) Adaptive Blur Augmentation: An average blur with a randomly chosen odd-sized kernel is applied to the dense input to reduce sharpness, limiting the model’s tendency to overfit to local depth details.

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

SourceEdited

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

"Dressed in a suit and wearing a wristwatch, a grieving woman sits in a wooded graveyard, anxiously wringing her hands."

"Captain America leans over his desk, writing with his right hand. After a moment, he raises it to adjust the desk lamp."

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

EditedSource

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

"Someone is slicing a carrot on a wooden cutting board in the kitchen, with one hand holding a small knife and the other pressing down on the carrot. The carrot is being cut into thin slices, one by one. Onions and zucchini are placed nearby."

- Figure 7: Video Editing. LongVie enables high-quality and temporally consistent video editing.

" A lifelike tiger lies quietly on a rock in a shadowy forest, its demeanor shifting from relaxed to alert as it slowly lifts its head and scans the surroundings, seemingly sensing movement nearby. Its muscles tense subtly beneath its fur, ready to react. "

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

SourceTransferred

[Figure 141]

"The camera slowly moves along a snowy stream, revealing a fallen tree, snow-covered rocks, shrubs, and a dense pine forest in the distance.”

SourceTransferred

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

"A young woman wearing a mask and a hooded jacket stands at the port as a massive container ship slowly passes by in the background.”

- Figure 8: Motion & Scene Transfer. LongVie effectively transfers motion and scene across frames.

Together, these degradations prevent over-reliance on dense signals and improve the model’s ability to integrate complementary information from sparse modalities, ultimately enhancing long-term video quality and consistency.

##### 2.3 Versatility to Downstream Video Generative Tasks

In this section, we describe how LongVie can be adapted to various long video generation tasks.

Video Editing. LongVie can be adapted for long-range video editing. We first edit the initial frame by selecting a target region and completing it using the fill model from FLUX [19]. The completed frames are then fed into LongVie, along with dense and sparse control signals, to generate temporally consistent edited videos.

Motion & Scene Transfer. LongVie supports motion and scene transfer across long videos. Given a source video with target motion or layout, the depth-to-image model from FLUX [19] is used to synthesize an initial frame reflecting the desired attributes. Depth and point maps are extracted as control signals for LongVie, which generates videos that preserve the transferred motion or scene while ensuring temporal and visual consistency.

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

MeshTransferred

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

"In a sunlit living room with soft white curtains and a mint-green sofa set against the wall, a robotic arm takes center stage on the wooden coffee table, slowly rising, pausing in mid-air, then gracefully lowering itself as if performing a quiet routine."

- Figure 9: Mesh-to-Video. LongVie generates realistic long videos from animated 3D meshes.

Controllable Mesh-to-Video. LongVie generates long videos from animated 3D meshes without textures. We render the mesh in a 3D engine (e.g., Blender) to produce an animation. A depth-toimage model [19] synthesizes an initial stylized frame, and depth maps and point trajectories are extracted from the animation. These signals guide LongVie to produce coherent, high-quality videos, enabling seamless integration of animated 3D assets into photorealistic domains.

#### 3 Experiments

Implementation Details. We implement LongVie by copying and fine-tuning 18 DiT blocks in each model. During training, we first extract depth maps using Video Depth Anything [6] as the dense control signal, and then apply SpatialTracker [36] to track 3D points based on the normalized depth. Following DAS [11], we uniformly sample 4,900 points per short video as sparse control signals. Each training video is divided into 49-frame clips at a resolution of 480×720 and 8 frames per second (fps). We then use Qwen2.5-VL-7B [1] to automatically generate captions for the training videos. In total, we use 130,000 videos to train LongVie. The training data consists of ACID [21], VchitectT2VDataVerse [9], and MovieNet [17], with detailed information provided in the supplementary material. LongVie is trained using the AdamW optimizer with a learning rate of 1e-4. Training is conducted on 8 A100 GPUs over approximately 3,000 iterations, with an effective batch size of 64, and takes about 5 days to complete. During inference, LongVie requires approximately 4.5 minutes to sample a 6-second video. Consequently, generating a one-minute controllable video takes about 45 minutes on a single A100 GPU.

##### 3.1 Qualitative and Quantitative Results

LongVGenBench. To address the lack of suitable benchmarks for controllable long video generation, we introduce LongVGenBench, a dataset of 100 one-shot videos, each lasting at least one minute at 1080p resolution. Existing datasets are inadequate, as they lack long, continuous, one-shot videos—crucial for evaluating temporal consistency and controllability. LongVGenBench spans diverse real-world and game-based scenarios, and includes challenging cases such as rapid scene transitions and complex motions (see supplementary material for details), making it a strong benchmark for this task. For evaluation, each video is divided into 6-second clips, and captions are automatically generated using Qwen-2.5-VL-7B [1] to serve as prompts. Each video is further segmented into ten 49-frame clips at 8 frames per second (fps), with a 1-frame overlap, following the autoregressive setup used in our experiments. Control signals are extracted from the split clips. During validation, no transformation is applied to the first frame of each video, ensuring fair comparison and enabling accurate assessment of generation quality, as ground-truth frames are available for reference.

Evaluation Metrics and Baselines. To evaluate the effectiveness of LongVie, we adapt several video generation models for long video generation, including the base model CogVideoX [37]; the controllable models VideoComposer [34], Motion-I2V [29], Go-With-The-Flow [2], and DAS [11]; as well as a depth-controlled variant of CogVideoX, termed Depth-LV. We also compare against StreamingT2V [14], a strong image-driven baseline for long video generation. For evaluation, we follow the widely used benchmark VBench [18] and employ seven metrics—Background Consistency, Subject Consistency, Overall Consistency, Temporal Style, Dynamic Degree, Temporal Flickering, and Imaging Quality—to assess temporal coherence and visual fidelity. We also report traditional

- Table 1: Quantitative results of LongVie and baselines on our LongVGenBench. DAS-LV and Depth-LV refer to the adapted versions of DAS and depth-controlled CogVideo, respectively, for long video generation. Go-With-Flow refers to the model Go-With-The-Flow. Bold indicates the best performance, and underline denotes the second-best.

METHODS S.C.↑ B.C.↑ O.C.↑ D.D↑ T.F.↑ A.Q.↑ I.Q.↑ SSIM↑ LPIPS↓

CogVideoX 85.38% 90.46% 20.72% 22.06% 97.80% 54.96% 64.89% 0.374 0.521 StreamingT2V 83.18% 90.56% 20.58% 21.15% 97.52% 52.51% 62.45% 0.360 0.572 VideoComposer [34] 80.33% 88.83% 19.83% 27.78% 96.36% 52.83% 59.33% 0.346 0.583 Motion-I2V [29] 84.25% 89.32% 19.99% 37.34% 97.16% 53.26% 61.57% 0.385 0.504 Go-With-Flow [2] 84.37% 90.62% 21.79% 46.15% 97.77% 53.59% 62.21% 0.453 0.394 DAS [11] 86.06% 90.78% 21.10% 36.76% 98.11% 53.28% 64.57% 0.401 0.482 Depth-LV 87.09% 91.37% 21.25% 46.06% 97.70% 54.80% 64.84% 0.508 0.347 LongVie 87.12% 91.76% 21.82% 46.59% 98.43% 55.31% 64.91% 0.557 0.290

- Table 2: User Study Comparison with Baselines. We define 5 dimensions and invite 60 participants to evaluate LongVie with the baselines. LongVie outperforms all baselines in every aspect.

METHODS Visual Prompt-Video Condition Color Temporal Quality Consistency Consistency Consistency Consistency

CogVideoX [37] 2.232 2.251 1.967 2.514 2.272 StreamingT2V [14] 2.054 2.017 2.232 1.942 1.959 DAS-LV [11] 3.072 3.138 3.253 3.035 3.183 Depth-LV 3.286 3.153 3.318 3.267 3.262 LongVie 4.387 4.471 4.282 4.298 4.365

similarity-based metrics, including SSIM and LPIPS, to quantify the reconstruction quality of generated videos with respect to their input references.

Experimental Results. The quantitative results in Tab. 1 demonstrate that LongVie achieves the best temporal consistency and controllability among all baselines, achieving state-of-the-art performance. To further illustrate the effectiveness of LongVie in controllable long video generation, we present video editing results in Fig. 7, where LongVie faithfully replaces target characters or objects as specified. The results of motion and scene transfer are shown in Fig. 8, indicating that LongVie can handle complex motions and scene transformations. Additionally, we showcase controllable mesh-to-video generation results in Fig. 9. We first place the desired animated 3D models in Blender and repaint them using FLUX. As shown, LongVie successfully synthesizes high-quality videos from the repainted meshes.

User Study. To comprehensively evaluate the models, we carefully design and conduct a subjective user study. To mitigate participant fatigue, we structure the evaluation accordingly. From the generated videos, we randomly select 80 samples, each paired with its corresponding prompt and control signals. The evaluation focuses on five key aspects: Visual Quality, Prompt-Video Consistency, Condition Consistency, Color Consistency, and Temporal Consistency. We compare five models: CogVideoX [37], StreamingT2V [14], DAS-LV [11], Depth-LV, and LongVie. A total of 60 participants are invited. For each evaluation aspect, participants rank the outputs of the five models, assigning 5 points to the best and 1 point to the worst. The average scores across all evaluations are summarized in Tab. 2. As shown, our proposed method, LongVie, achieves the highest scores across all criteria.

##### 3.2 Ablation Study

Unified Initial Noise & Global Normalization. We observe that both the unified initialization of noise and the normalization of control signals significantly affect the consistency and quality of the generated videos. To evaluate their impact, we generate videos under three settings: without Global Normalization, without Unified Initial Noise, and without both. The results, reported in Tab. 3 and evaluated using four corresponding metrics, demonstrate that both Global Normalization and Unified Initial Noise contribute positively to controllable long video generation.

- Table 3: Ablation study for our proposed components. The block denotes experiments targeting temporal consistency, while the block denotes those focusing on visual quality.

METHODS Consistency Quality

Subject Background Aesthetic Imaging Full model 87.12% 91.76% 54.91% 64.91%

w/o Global Normalization 86.39% (↓0.73) 91.41% (↓0.35) 54.88% (↓0.03) 64.81% (↓0.10) w/o Unified Initial Noise 86.63% (↓0.49) 91.59% (↓0.17) 54.80% (↓0.11) 64.84% (↓0.07) w/o Both 86.23% (↓0.89) 91.37% (↓0.39) 54.78% (↓0.13) 64.86% (↓0.05)

w/o Feature Degradation 87.01% (↓0.11) 91.68% (↓0.08) 54.32% (↓0.59) 64.01% (↓0.90) w/o Data Degradation 87.11% (↓0.01) 91.69% (↓0.07) 54.08% (↓0.83) 63.97% (↓0.94) w/o Both 86.99% (↓0.13) 91.50% (↓0.26) 53.95% (↓0.96) 63.67% (↓1.24)

Degradation Training Strategy. We conduct an ablation study on the degradation-aware training strategy to balance the contributions of multiple modalities. Results in Tab. 3 show that both feature and data level strategies improve the visual quality of long video generation.

- 4 Related Works

Video Diffusion Models. Recently, video diffusion models have advanced rapidly, enabling the generation of high-resolution videos with consistent appearance and motion across dozens of frames. Representative open-source models include CogVideoX [37], HunyuanVideo [32], and Wan2.1 [35]. In parallel, several closed-source models such as Sora [25] and Kling [33] have also demonstrated impressive results. While these models can generate high-quality videos from text or image prompts, they still exhibit limitations in fine-grained controllability and scalability to long-duration content. To bridge this gap, our work focuses on controllable long video generation. Specifically, we build upon CogVideoX [37] as our base model. Importantly, our proposed LongVie framework can be seamlessly extended to enable controllable long video generation using any existing video diffusion architecture.

Controllable Video Generation. Efforts in controllable video generation [2, 29] have already enabled the synthesis of high-quality videos. VideoComposer [34] leverages diverse conditions to enhance controllability. Following the design of ControlNet [39], SparseCtrl [12] introduces sparse control for video generation, while DAS [11] employs 3D point maps for more precise control. Cosmos-Transfer1 [24] uses multi-modal control to improve the quality of controllable video generation. However, these methods focus on short clips and do not address the challenges of long-form generation.

Long Video Generation. FreeNoise [27] and PYoCo [10] have explored different noise initialization strategies to enhance the temporal consistency of generated videos. These methods significantly improve overall video consistency. However, fine-grained temporal consistency issues still persist in long video generation. To address this, we propose the use of Unified Noise Initialization in this work. Additionally, to mitigate the high computational resource requirements, several methods [3, 22, 31, 20, 41] have adopted training-free strategies. These include analyzing attention mechanisms or identifying key components in video models to extend the duration of generated videos. Furthermore, some training-based approaches focus on enhancing video length. StreamingT2V [14] adopts an autoregressive approach for long video generation by leveraging a short-term memory mechanism to maintain temporal consistency. Diffusion Forcing [5, 30] introduces an autoregressive framework that enables diffusion models to generate long videos through causal learning. This effectively captures temporal dependencies. TTT [8] and LCT [13] explicitly incorporate long-range temporal context to enhance generation quality and coherence across extended sequences. FramePack [38] compresses conditional or previously generated frames into a compact representation as a new condition, emphasizing recent frames through reduced spatial resolution. This improves efficiency and consistency in long video generation. Although these methods improve long video generation from various perspectives, achieving fine-grained control and maintaining long-term coherence remain open challenges.

#### 5 Conclusion

In this work, we investigate the causes of temporal inconsistency and visual degradation in controllable long video generation. To address these issues, we propose LongVie, a multi-modal guidance framework that integrates dense and sparse control signals in an autoregressive manner, supported by a degradation-aware training strategy to enhance visual quality. It also applies global normalization to control signals and uses unified noise initialization to improve temporal consistency. To evaluate controllable long video generation, we curate LongVGenBench, a dataset of 100 high-quality videos, each lasting over one minute and spanning real-world and game-like scenarios. Experiments on LongVGenBench and ablation studies show that LongVie achieves state-of-the-art performance in long video generation. Moreover, downstream video generation tasks demonstrate that LongVie can generate high-quality controllable videos up to one minute long.

#### A Overview of Supplementary Material

In this section, we briefly introduce the contents of the supplementary material. In Sec.B, we provide a detailed description of the LongVGenBench dataset collected for evaluation. In Sec.C, we elaborate on the implementation details of LongVie, including training data, inference adaptation, and model configuration. In Sec.D, we present ablation studies on the structure and number of MM-Blocks, as well as additional explorations of LongVie under varying initial noise and inaccurate control signals. In Sec. E, we show additional qualitative results generated by our model in various styles.

#### B LongVGenBench

To better introduce the proposed dataset, LongVGenBench, for evaluating controllable long video generation, we present several examples in Fig. 10. The dataset includes both real-world and synthetic scenes, each lasting at least one minute and having a minimum resolution of 1080p. As shown in the figure, the dataset contains various types of camera movements, which pose significant challenges for video generation models. Importantly, LongVGenBench is designed to be model-agnostic and can be used to evaluate any long video generation method, not just LongVie. In this paper, we utilize LongVGenBench by splitting each video into 6-second clips with a one-frame overlap. Captions are then extracted from these short clips to prepare the training data. To further illustrate the distribution of the dataset, we provide detailed statistics in Tab 4.

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

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

Figure 10: Examples from LongVGenBench. We show several videos from both real-world and synthetic scenarios in LongVGenBench, covering a variety of indoor and outdoor environments to evaluate the controllable long video generation ability of our model.

#### C More Implementation Details

Training Data. As discussed in the Implementation Details section of the main paper, our training data consists of three parts: 1) ACID and ACID-Large [21]: These datasets contain thousands of aerial drone videos featuring various coastlines and natural scenes sourced from YouTube. They are released in the same format as RealEstate10K [43]. 2) Vchitect_T2V_DataVerse [9]: This dataset comprises 14 million high-quality videos collected from the Internet, each paired with detailed textual captions. 3) MovieNet [17]: MovieNet includes 1,100 movies spanning a diverse range of years, countries, and genres.

Table 4: Statistics of LongVGenBench video categories.

DAY NIGHT TOTAL

Real 55 10 65 Indoor 22 0 22 Outdoor 33 10 43

Game 25 10 35 Indoor 10 0 10 Outdoor 15 10 25

Total 80 20 100

Data Pre-processing. During our experiments, we observed that if the training data contains transitions (e.g., cuts or scene changes), the temporal consistency of the generated videos tends to degrade, often resulting in unintended transitions in the output. To address this, we use the PySceneDetect toolkit [4] to detect transitions in the videos and segment them accordingly. Each resulting segment is then sampled at 8 fps and truncated to 49 frames for training LongVie. From these 49-frame clips, we use Video Depth Anything [6] to extract depth maps, and compute point tracking using SpatialTracker [36], based on the extracted depth and RGB frames. We also generate captions for each clip using Qwen-2.5-VL-7B [1] to ensure accurate and context-relevant descriptions. In total, we prepare 150,000 video–control signal pairs to train our model.

Test Time Adaptation. Point tracking becomes less effective during the inference of long videos because it relies on content visible in the first frame. As the video progresses, or when the initially tracked content moves entirely out of view, the tracked points tend to fail. In our framework, point tracking serves as a sparse control signal that primarily guides motion rather than appearance. To ensure its effectiveness, we avoid extracting point tracks from the entire 1-minute video directly. Instead, we first extract the depth map for the full 1-minute sequence and apply global normalization. We then divide the video into overlapping 49-frame clips and compute colorful point tracking for each clip based on the globally normalized depth maps. Regarding captions, in transfer tasks, the original captions of the source videos are often misaligned with the visual content after transfer. To address this, we employ a large language model (LLM) to analyze the differences between the transferred and original frames, and revise the captions to better reflect the updated content, as illustrated in Fig. 11.

[Figure 202]

Please carefully compare the two images and adjust the original prompt based on the content and style of the new image. Once the prompt has been updated to better reflect the new image, please return the revised version to me.

[Figure 203]

[Figure 204]

The video depicts a 3D-rendered scene featuring a large, muscular creature with a scaly, textured body and sharp claws. The creature appears to be in motion, crouching low to the ground as if preparing to pounce or charge forward. Its posture is dynamic, with its arms bent and hands gripping the ground for balance. The background consists of a cityscape that has been devastated by some form of destruction. Buildings are scattered and broken, with many structures lying in ruins …

[Figure 205]

The video depicts a 3D-rendered scene featuring a large, muscular ice creature with a bright blue, crystalline body and sharp claws. The creature appears to be in motion, crouching low to the snow-covered ground as if preparing to pounce or charge forward. Its posture is dynamic, with its arms bent and hands positioned with razor-sharp claws extended. The background consists of a small town or village that has been completely blanketed in snow. Small houses with snow-covered rooftops are scattered throughout the scene, creating a wintry residential landscape. In the background, a mysterious glowing blue portal or light emanates from what appears to be a church or larger structure. The sky above has a bluish tint, enhancing the cold and otherworldly atmosphere of the scene. The overall color palette is dominated by shades of blue and white, emphasizing the frigid environment where this ice monster roams. The creature's bright blue skin contrasts sharply with the snowy surroundings, making it the focal point of the otherwise peaceful winter setting.

- Figure 11: Caption Refinement via LLM. We revise the original captions by comparing the transferred and source images using a LLM. This process ensures that the updated captions accurately reflect the visual content of the transferred video.

- Table 5: Ablation study on control block architecture. Comparison between unified and separate zero-linear designs. The unified approach yields better performance across all the metrics.

VARIANT Consistency Quality

SSIM LPIPS Subject Background Aesthetic Imaging

Separate Zero-Linear 87.01% 91.59% 54.80% 64.83% 0.524 0.321 Unified Zero-Linear 87.12% 91.76% 54.91% 64.91% 0.557 0.290

- Table 6: Ablation study on the number of control blocks. We compare different numbers of control blocks to evaluate the efficiency and effectiveness of our model.

BLOCKS Consistency Quality

SSIM LPIPS Subject Background Aesthetic Imaging

12 85.79% 90.96% 54.52% 64.14% 0.502 0.348 18 87.12% 91.76% 54.91% 64.91% 0.557 0.290

Model Configuration. We provide additional details about our model, LongVie. During training, we set the feature-level degradation probability α to 15% and the data-level degradation probability β to 10%. One or both degradation methods are randomly selected at each step. We set n = 5 for the Random Scale Fusion method. The model is trained without any degradation strategies for the first 2000 iterations, and these strategies are introduced in the final 1000 iterations. For weight initialization, the dense and sparse blocks within each MM Block copy only half of the original weights, and the feature dimension is also reduced to half compared to that in CogVideoX [37]. Specifically, we split the original weights into two interleaved sets based on their index positions (e.g., weights at positions 0, 2, 4 and 1, 3, 5), and assign them to the dense and sparse blocks respectively. The half copy significantly reduces the total number of parameters in the model.

#### D Additional Ablation Studies

To comprehensively evaluate the effectiveness of LongVie, we conduct extensive ablation experiments.

Control Block Architecture. As discussed in Sec. 2.1, we compare two designs for integrating control features with the denoising features. The results, summarized in Tab. 5, indicate that the unified zero-linear approach outperforms the separate design across all metrics.

Number of Control Blocks. We also investigate the impact of the number of control blocks in LongVie. Specifically, we train a variant using only 12 control blocks and compare it with our default 18-block setting. As shown in Tab. 6, increasing the number of blocks leads to consistent improvements across all metrics.

Exploration of Initial Noise. To investigate the impact of initial noise on the temporal consistency of generated videos, we evaluate our model under controlled perturbations of the initialization noise. Specifically, we add Gaussian noise sampled from N(0,1) and scaled by a factor α to the Global Initialization Noise. Temporal consistency is assessed using our proposed LongVGenBench dataset, and we report four key metrics: Subject Consistency, Background Consistency, Temporal Style, and Overall Consistency, as shown in Tab. 7. The results indicate that smaller variations in the initial noise lead to improved temporal consistency in the generated videos.

Table 7: Comparison of different initial noise strategies.

METHOD Subject Background Overall Temporal Consistency Consistency Consistency Style

Random Noise 86.39% 91.41% 21.17% 21.17% Global Initial Noise with α = 0.5 86.51% 91.48% 21.25% 21.35% Global Initial Noise with α = 1 86.83% 91.62% 21.46% 21.56% Global Initial Noise with α = 0.05 86.96% 91.66% 21.64% 21.74% Global Initial Noise 87.12% 91.76% 21.76% 21.82%

Table 8: Evaluation of different control modalities across multiple video generation metrics.

Method S.C. B.C. O.C. D.D. T.F. A.Q. I.Q. SSIM LPIPS Masked Keypoint Map 86.74% 91.28% 21.54% 46.18% 97.71% 54.63% 64.52% 0.531 0.312 Blurred Depth Map 86.91% 91.42% 21.68% 46.42% 97.94% 54.75% 64.75% 0.548 0.297 LongVie 87.12% 91.76% 21.82% 46.59% 98.43% 55.31% 64.91% 0.557 0.290

Robustness to Inaccurate Control Signals. To further assess the robustness of our method, we evaluate its performance under degraded control conditions, including depth maps blurred with a 5×5 kernel and point maps with 20% of keypoints randomly masked. As shown in Tab. 8, our method achieves performance comparable to the clean-control setting, demonstrating strong resilience to control signal degradation with only minimal performance drop.

#### E More Qualitative Results

To better demonstrate the robustness of our model, we present additional experimental results on various long video generation tasks. The results are shown in Fig. 12, Fig. 13 and Fig. 14.

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

MeshStyle1Style2Style3MeshStyle1Style2Style3

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

[Figure 237]

[Figure 238]

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

[Figure 253]

- Figure 12: More results of Mesh-to-Video. We input a monster located in a village scene, animate the mesh, and convert it into a video, where our model supports rendering in various styles.

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

Style1Style2Style3Style4SourceStyle1Style2Style3SourceStyle4

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

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

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

- Figure 13: More results of Motion & Scene Transfer. A man riding a horse in various scenes.

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

SourceStyle1Style2Style3SourceStyle1Style2Style3

[Figure 320]

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

- Figure 14: More results of Motion & Scene Transfer. We transfer a car driving through a mountain valley across various seasons.

#### F Social Impacts

LongVie facilitates more customized video generation, with potential applications across various fields such as short-form content creation and the film industry. By enhancing controllability and consistency in generated videos, it can significantly streamline video production processes and expand creative possibilities, though it is also crucial to consider and address potential risks, including misinformation or misuse.

#### G Limitations and Future Work

Although LongVie is capable of generating high-quality, controllable 1-minute videos, the inference process remains relatively time-consuming, requiring approximately 45 minutes per video at 480×720 resolution. Reducing inference latency for long video generation remains a non-trivial challenge and is an important direction for future research. Furthermore, the current output resolution, while sufficient for benchmarking, still falls short of cinematic standards. In future work, we aim to explore higher-resolution generation frameworks to produce visually richer and more realistic long videos.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [2] Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski, Pascal Clausen, Mingming He, Li Ma, Yitong Deng, Lingxiao Li, Mohsen Mousavi, Michael Ryoo, Paul Debevec, and Ning Yu. Go-with-the-flow: Motion-controllable video diffusion models using real-time warped noise. In CVPR, 2025. Licensed under Modified Apache 2.0 with special crediting requirement.
- [3] Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, and Xiangyu Yue. Ditctrl: Exploring attention control in multi-modal diffusion transformer for tuning-free multi-prompt longer video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7763–7772, 2025.
- [4] Brandon Castellano. Pyscenedetect. https://github.com/Breakthrough/PySceneDetect, 2024. Video Cut Detection and Analysis Tool. Available at https://www.scenedetect.com. Accessed: 202505-19.
- [5] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.
- [6] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. arXiv:2501.12375, 2025.
- [7] Weifeng Chen, Yatai Ji, Jie Wu, Hefeng Wu, Pan Xie, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models. arXiv e-prints, pages arXiv–2305, 2023.
- [8] Karan Dalal, Daniel Koceja, Gashon Hussein, Jiarui Xu, Yue Zhao, Youjin Song, Shihao Han, Ka Chun Cheung, Jan Kautz, Carlos Guestrin, Tatsunori Hashimoto, Sanmi Koyejo, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training, 2025.
- [9] Weichen Fan, Chenyang Si, Junhao Song, Zhenyu Yang, Yinan He, Long Zhuo, Ziqi Huang, Ziyue Dong, Jingwen He, Dongwei Pan, et al. Vchitect-2.0: Parallel transformer for scaling up video diffusion models. arXiv preprint arXiv:2501.08453, 2025.
- [10] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, JiaBin Huang, Ming-Yu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22930–22941, 2023.
- [11] Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou, Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei Liu, Wenping Wang, and Yuan Liu. Diffusion as shader: 3d-aware video diffusion for versatile video generation control. arXiv preprint arXiv:2501.03847, 2025.
- [12] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In European Conference on Computer Vision, pages 330–348. Springer, 2024.
- [13] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation, 2025.
- [14] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text, 2025.
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020.
- [16] Hsin-Ping Huang, Yu-Chuan Su, Deqing Sun, Lu Jiang, Xuhui Jia, Yukun Zhu, and Ming-Hsuan Yang. Fine-grained controllable video generation via object appearance and context. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 3698–3708. IEEE, 2025.
- [17] Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and Dahua Lin. Movienet: A holistic dataset for movie understanding. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16, pages 709–727. Springer, 2020.

- [18] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [19] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [20] Yumeng Li, William Beluch, Margret Keuper, Dan Zhang, and Anna Khoreva. Vstar: Generative temporal nursing for longer dynamic video synthesis. arXiv preprint arXiv:2403.13501, 2024.
- [21] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021.
- [22] Yu Lu, Yuanzhi Liang, Linchao Zhu, and Yi Yang. Freelong: Training-free long video generation with spectralblend temporal attention. Advances in Neural Information Processing Systems, 37:131434–131455, 2024.
- [23] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Ying Shan, Xiu Li, and Qifeng Chen. Follow your pose: Pose-guided text-to-video generation using pose-free videos. arXiv preprint arXiv:2304.01186, 2023.
- [24] NVIDIA. Cosmos-transfer1: Conditional world generation with adaptive multimodal control, 2025.
- [25] OpenAI. Sora. Accessed February 15, 2024 [Online] https://sora.com/library, 2024.
- [26] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dimitry Vengertsev, Edgar Schonfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie gen: A cast of media foundation models, 2025.
- [27] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling. arXiv preprint arXiv:2310.15169, 2023.
- [28] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2021.
- [29] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.
- [30] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. Historyguided video diffusion, 2025.
- [31] Jiangtong Tan, Hu Yu, Jie Huang, Jie Xiao, and Feng Zhao. Freepca: Integrating consistency information across long-short frames in training-free long video generation via principal component analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 27979–27988, 2025.
- [32] Hunyuan Foundation Model Team. Hunyuanvideo: A systematic framework for large video generative models, 2024.
- [33] Kuaishou Team. Kling. Accessed December 9, 2024 [Online] https://klingai.kuaishou.com/, 2024.
- [34] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36:7594–7611, 2023.

- [35] WanTeam. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [36] Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker: Tracking any 2d pixels in 3d space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [37] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [38] Lvmin Zhang and Maneesh Agrawala. Packing input frame contexts in next-frame prediction models for video generation. Arxiv, 2025.
- [39] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023.
- [40] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023.
- [41] Min Zhao, Guande He, Yixiao Chen, Hongzhou Zhu, Chongxuan Li, and Jun Zhu. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025.
- [42] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.
- [43] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. ACM Trans. Graph. (Proc. SIGGRAPH), 37, 2018.

