## LongVie 2: Multimodal Controllable Ultra-Long Video World Model

Jianxiong Gao1, Zhaoxi Chen3, Xian Liu4, Junhao Zhuang5, Chengming Xu1 Jianfeng Feng1, Yu Qiao6, Yanwei Fu1,†, Chenyang Si2,†, Ziwei Liu3,† 1FDU, 2NJU, 3NTU, 4NVIDIA, 5THU, 6Shanghai AI Laboratory https://vchitect.github.io/LongVie2-project/

# arXiv:2512.13604v1[cs.CV]15Dec2025

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

00:10 00:20 00:30 00:40 00:50 01:00

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Dense Sparse

[Figure 16]

[Figure 17]

00:10 00:20 00:30 00:40 00:50 01:00

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

…

…

00:30 01:00 01:30 02:00 02:30 03:00

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

###### World-Level Guidance

00:50 01:40 02:30 03:20 04:10 05:00

[Figure 34]

[Figure 35]

LongVie2

Time

[Figure 36]

[Figure 37]

[Figure 38]

Figure 1. LongVie 2 is a controllable ultra-long video world model that autoregressively generates videos lasting up to 3–5 minutes. It is driven by world-level guidance integrating both dense and sparse control signals, trained with a degradation-aware strategy to bridge the gap between training and long-term inference, and enhanced with history-context modeling to maintain long-term temporal consistency.

#### Abstract

sive benchmark comprising 100 high-resolution one-minute videos covering diverse real-world and synthetic environments. Extensive experiments demonstrate that LongVie 2 achieves state-of-the-art performance in long-range controllability, temporal coherence, and visual fidelity, and supports continuous video generation lasting up to five minutes, marking a significant step toward unified video world modeling.

Building video world models upon pretrained video generation systems represents an important yet challenging step toward general spatiotemporal intelligence. A world model should possess three essential properties: controllability, long-term visual quality, and temporal consistency. To this end, we take a progressive approach—first enhancing controllability and then extending toward long-term, high-quality generation. We present LongVie 2, an endto-end autoregressive framework trained in three stages: (1) Multi-modal guidance, which integrates dense and sparse control signals to provide implicit world-level supervision and improve controllability; (2) Degradation-aware training on the input frame, bridging the gap between training and long-term inference to maintain high visual quality; and (3) History-context guidance, which aligns contextual information across adjacent clips to ensure temporal consistency. We further introduce LongVGenBench, a comprehen-

#### 1. Introduction

Recent breakthroughs in video generation have been largely driven by the availability of large-scale datasets and the rapid development of diffusion-based architectures [11, 18, 32, 33, 55]. These advances have enabled powerful models such as CogVideoX [43], HunyuanVideo [36], Kling [37], Sora [31], and Wan2.1 [40] to synthesize photorealistic and semantically rich videos directly from text prompts, demonstrating

Input Image Temporal Inconsistency Visual Degradation

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Input Image t=5 t=10 t=15

that diffusion models can effectively capture both visual dynamics and high-level semantic structures.

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

00:00 00:05 00:10 00:15

Building upon this foundation, research has recently shifted from merely generating visually appealing clips to modeling the underlying dynamics of the physical world. This shift has led to the emergence of video-based world models that simulate realistic environments and interactions, exemplified by HunyuanGameCraft [24], Yume [29], and Matrix-Game [16]. These models incorporate controllable elements—such as camera motion or agent actions—to emulate how the world evolves over time. However, despite these advances, current world models face two major challenges. First, their controllability remains limited, often restricted to low-level or localized adjustments rather than global, semantic-level control over the environment. Second, their temporal scalability is fragile—when extended to long horizons (e.g. beyond one minute), these models often exhibit visual degradation and temporal drift, as shown in Fig. 2, where visual quality collapses over time.

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

00:00 00:05 00:10 00:15

Figure 2. Unstable long-term generation. As the generated duration increases, current video world models gradually lose controllability, visual fidelity, and temporal consistency.

implicit world-level guidance. 2) Degradation-aware training introduces controlled degradations to guided frames, effectively bridging the gap between training and long-horizon inference. 3) History context guidance leverages preceding frames as contextual inputs to maintain long-range consistency across adjacent clips. Together, these stages enable LongVie 2 to generate minute-long controllable videos with strong temporal coherence and high visual fidelity, marking a crucial step toward unified and scalable video world models.

Toward the next generation of video world models, we seek to unify two complementary objectives—fine-grained controllability and long-term coherence—within a single framework. An ideal world model should not only generate visually and temporally consistent videos but also respond coherently to structured inputs and interactions. In essence, it should achieve: (1) Comprehensive controllability, ensuring that the entire evolving scene remains semantically interpretable and manipulable across time; (2) Long-term fidelity, maintaining visual quality over extended sequences; and (3) Long-context consistency, preserving stable, realistic dynamics aligned with real-world behavior.

To evaluate the effectiveness of LongVie 2, we construct LongVGenBench, a benchmark consisting of 100 high-quality videos, each lasting at least one minute. The dataset covers a wide range of scenarios, including both realworld environments and game-based scenes, spanning day and night as well as indoor and outdoor settings, ensuring rich variations in content, structure, and motion dynamics. We compare LongVie 2 and baselines on LongVGenBench in terms of long-term visual quality and temporal consistency. Experimental results demonstrate that LongVie 2 achieves state-of-the-art performance in controllable long video generation, producing videos both temporally coherent and visually realistic.

In this paper, we investigate how to construct video world models by extending pretrained video diffusion backbones into controllable long-horizon generators. Introducing controllability into these pretrained architectures is relatively straightforward, as most short-clip diffusion models already support modular conditioning through components such as LoRA [20], ControlNet [52], or similar plug-in adapters. Leveraging this property, we adopt a control-first, then long strategy: we first enhance controllability within short video generation and then progressively extend the temporal horizon to achieve long-range coherence.

Our main contributions are summarized as follows:

- • We present LongVie 2, a controllable long video generation framework that extends pretrained diffusion backbones toward video world modeling.
- • We design a three-stage progressive training scheme that integrates multi-modal guidance, degradation-aware learning, and history context guidance to enhance controllability, temporal coherence, and visual fidelity, aligning with essential capabilities required for video world models.
- • We establish LongVGenBench, a benchmark of 100 oneminute videos for evaluating controllability, consistency, and perceptual quality in long video generation.

Accordingly, we extend controllable short-video generators into long-horizon, temporally coherent frameworks, marking an essential step toward general-purpose video world modeling. To this end, we propose LongVie 2, a controllable long video generation framework built upon an autoregressive paradigm. LongVie 2 is trained through three progressive stages that jointly enhance controllability, temporal consistency, and visual fidelity: 1) Multi-modal guidance integrates dense (e.g., depth maps) and sparse (e.g., keypoints) control signals under balanced training, providing complementary structural and semantic constraints that offer

#### 2. Methodology

Preliminary: Diffusion Models. Diffusion models are powerful video world learners that synthesize realistic videos by progressively removing noise. So, the model ϵθ learns to undo a process that adds noise to data. During training, it is

###### Unified Initialization

…

[Figure 51]

Global Normalized MM-Control DiT (Trainable Copy)

[Figure 52]

"

Trainable

[Figure 53]

[Figure 54]

Noise!! !

Frozen

❄

[Figure 55]

[Figure 56]

- *)+()%

%$&'()%"#

- *)+()%"#

- *)+()%$#

MM Block

[Figure 57]

!"#$!$#

[Figure 58]

!

Dense Block

[Figure 59]

Depth Video /-(!

[Figure 60]

❄

[Figure 61]

%$&'()%$#

E

C

… …

[Figure 62]

!

Zero Linear

GlobalNormalization

-!$# ∈ ℝ)×+×,×-

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

!

Sparse Block

,&

Generated video -! Point Video -.'!

!"#$!

Denoising DiT

Latent Noise

%$&'()%

[Figure 67]

[Figure 68]

❄

-! ∈ ℝ)×+×,×-

[Figure 69]

[Figure 70]

"

[Figure 71]

❄ ❄

[Figure 72]

Padding 0

[Figure 73]

… …

DiT Block

E C D

Tail frames in -!$#

!"#$!"#

Prompt #: The video depicts a tranquil autumn landscape on horseback …

[Figure 74]

❄

Generation of !!" clip

T5

…

- Figure 3. Framework of LongVie 2. LongVie 2 serves as a controllable video world model that integrates both dense and sparse control signals to provide world-level guidance for enhanced controllability. A degradation-aware training strategy improves long-term visual quality, while tail frames from preceding clips are incorporated as historical context to maintain temporal consistency over extended durations.

I Predict token

conditioning inputs while keeping the base model frozen. Specifically, we freeze the parameters θ of the original DiT blocks and construct two trainable branches, each corresponding to a distinct control modality: dense and sparse. These branches, denoted as FD(·;θD) and FP(·;θS) , are lightweight copies of the original DiT layers with parameters θD and θP , respectively. Each processes the corresponding encoded control signal cD or cP. To integrate the control branches into the base generation path, we adopt zero-initialized linear layers ϕl. These layers allow the control signals to be injected additively into the main generation stream without affecting the model’s initial behavior, as they output zero at the start of training. The overall computation in the l-th controlled DiT block is defined as:

trained to predict the noise added at each step:

L = Ex,ϵ∼N(0,1),t ∥ϵ − ϵθ(xt,t)∥22 , (1)

where xt, ϵ, x0 denote the noisy data at time t, the true noise, and the original data, respectively. To save computing power, Latent Diffusion Models [33] work in a smaller, compressed space. An autoencoder turns x into a latent code z = E(x), and the model learns to predict noise in zt.

Overview. To address the aforementioned challenges and advance toward a unified video world model, we propose LongVie 2, an end-to-end framework trained through three progressive stages, as illustrated in Fig. 3. In Stage I, we inject multi-modal control signals into the model following the standard ControlNet paradigm, enabling implicit worldlevel guidance that enhances controllability. In Stage II, we introduce a first-frame degradation strategy that simulates long-form deterioration during training, effectively mitigating visual degradation in long-horizon generation. In Stage III, we incorporate historical frames as RGB guidance, allowing the model to leverage temporal context and maintain coherence across adjacent clips.

zl = Fl(zl−1) + ϕl(FDl (clD−1) + FSl(clS−1)), (2)

where Fl is the frozen base DiT block, and FDl , FPl are the control-specific sub-blocks.

While multi-modal control offers the potential for richer and more accurate video generation, simply combining dense and sparse control signals does not guarantee improved performance. In practice, we observe that dense signals tend to dominate the generation process, often suppressing the semantic and motion-level guidance provided by sparse signals. This imbalance can lead to suboptimal visual quality, particularly in scenarios requiring high-level semantic alignment over time. To address this issue, we propose a degradation-based training strategy designed to regulate the relative influence of dense control signals and encourage more balanced utilization of both modalities. This strategy weakens the dominance of the dense input through controlled perturbations at both the feature and data levels:

##### 2.1. Multi-Modal Control Injection

Specifically, we adopt depth maps as dense control signals and point maps as sparse control signals, leveraging the detailed structural information provided by depth and the high-level semantic cues captured by point trajectories to form an implicit world representation. To construct the point map sequences, we follow the procedure in DAS [13], where a set of keypoints is tracked across frames and colorized according to their corresponding depth values.

Inspired by the ControlNet [52] architecture, we build our Multi-Modal Control DiT by duplicating the initial 12 layers of the pre-trained Wan DiT and incorporating multi-modal

1) Feature-level degradation: During training, with probability α, we randomly scale the latent representation of the dense control by a factor λ sampled uniformly from [0.05,1].

###### Stage Ⅰ: Clean Pretraining

###### Stage Ⅱ: Degradation Tuning Stage Ⅲ: History-Aware Refinement

### LongVie 2

### LongVie 2

### LongVie 2

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

!

!

! "#̃ "$̃ ""

"! ""

"!̃ ""

- Figure 4. Training pipeline of LongVie 2. We first train the model using a standard ControlNet-based pipeline. In the second stage, we introduce degradation to the first frame to bridge the domain gap between ground truth and generated frames. Finally, to ensure temporal consistency, we incorporate historical frame information.

Accordingly, Equation 2 can be reformulated as:

[Figure 81]

[Figure 82]

D … E

E D

zl = Fl(zl−1) + ϕl(λ · FDl (clD−1) + FPl(clP−1)), (3)

- (a) Encoding Degradation
- (b) Generation Degradation

Input Image

Reconstruction Image (k=10)

This operation reduces the magnitude of the dense features, making the model more reliant on complementary information provided by the sparse modality. Over time, this encourages the network to learn a more balanced integration of both control sources.

[Figure 83]

[Figure 84]

noise

denoise

E DiT

D

Input Image Regenerate Image (t=8)

Figure 5. Details of Frame Degradation. We apply two types of degradation—VAE encoding and denoising—to simulate the quality decay that occurs during long-term generation.

2) Data-level degradation: Given a dense control tensor D ∈ RB×C×H×W, we apply degradation with probability β using two techniques: a) Random Scale Fusion: A set of spatial scales {1,1/2,...,1/2n} is predefined. One scale is randomly excluded, and the remaining scales are used to generate downsampled versions of the input, which are then upsampled to the original resolution. Each version is assigned a random weight normalized to sum to 1. Their weighted sum forms a fused depth map with multi-scale degradation and randomness, enhancing robustness to spatial variation. b) Adaptive Blur Augmentation: An average blur with a randomly chosen odd-sized kernel is applied to the dense input to reduce sharpness, limiting the model’s tendency to overfit to local depth details.

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

noise

[Figure 89]

[Figure 90]

denoise

To mitigate the gap between training and inference, we introduce a simple yet effective strategy: during training, we intentionally degrade the first image to mimic the corrupted inputs encountered in long-term generation. Specifically, we design two complementary degradation mechanisms that together define the degradation operator T (·): (1) Encoding degradation: We simulate VAE-induced corrup-

E DiT

D

(c) clip regenerate

E 𝒛 DiT

D

noise

(c) video regenerate

… E D

tion by repeatedly encoding and decoding the first image K times through the VAE and storing the resulting degraded reconstructions. (2) Generation degradation: We simulate diffusion-based degradation by adding Gaussian noise to the latent representation of the first image at a random timestep t, followed by denoising through the diffusion model to obtain a restored version. Formally, the degradation process can be expressed as:

(a) repeat encode

Together, these degradations mitigate the model’s overreliance on dense signals and enhance its ability to integrate complementary cues from sparse modalities, leading to more stable and consistent long-term video generation. During this pretraining stage, only the parameters of the control branches and fusion layers ϕl are updated, while the backbone remains frozen. This design introduces strong yet flexible conditioning from both dense and sparse modalities, preserving the stability and prior knowledge of the pre-trained base model.

(D◦E)K(I), w.p. 0.2, D Φ0(√αtE(I)+√1 − αtϵ) , w.p. 0.8,

(4)

T (I) =

where I denotes the input image, t < 15, and ϵ∼N(0,I).

##### 2.2. Frame Degradation Training

During training, we randomly apply T (·) with a probability α, where the degradation severity is inversely proportional to its sampling probability—milder degradations occur more frequently, while stronger degradations are applied less often. This strategy effectively improves the visual quality of long video generation. However, despite the improved visual fidelity, this training scheme introduces a new issue of temporal inconsistency: the model tends to enhance

The first image used in long-form video generation is itself a predicted frame whose quality gradually deteriorates over time, making it notably inferior to the clean images seen during training. As a result, the model tends to accumulate visual degradation across frames. Moreover, due to the nonlossless encoding and decoding process of the VAE, repeated reconstruction further amplifies this decline in quality.

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

- StageⅠ

… …

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

- StageⅡ

… …

[Figure 111]

- StageⅢ

Inconsistency Inconsistency

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

… …

###### 00:00 00:20 00:40

- Figure 6. Visual comparison across training stages. The results indicate that the second stage alleviates visual degradation but introduces intra-clip inconsistency, which is subsequently resolved by the final stage of training.

the quality of the next clip independently, leading to noticeable discrepancies between adjacent clips. To mitigate this problem, we introduce the history-context guidance stage, which explicitly incorporates the preceding clip as contextual conditioning to enforce temporal coherence across clips.

##### 2.3. History Context Guidance

To enhance temporal consistency between adjacent clips and further improve long-term coherence, we propose History Context Guidance. During training, we introduce history frames to encourage the model to learn causal temporal dependencies. Specifically, we first use the VAE encoder E(·) to obtain the latent representations of the NH history frames VH, denoted as zH = E(VH). Following the masking mechanism in Wan, we apply an all-ones mask to the history latent zH, forcing the model to reference historical context for maintaining temporal consistency. To align the training and long-term inference domains, we apply the degradation operator T (·) to the history frames, producing degraded inputs V˜H = T (VH). The degraded frames are then encoded as: z˜H = E(V˜H). Our model can be formulated as:

zt = F(zt+1 | zI,zH,cD,cP), (5)

where zI denotes the latent representation of the initial frame, and cD and cP represent the dense and sparse control signals, respectively. Here, F denotes the overall generative model.

In clip-based long video generation, the first frame of each clip serves as the temporal anchor linking consecutive segments. However, its latent representation may deviate from both the historical context and the degraded input, potentially weakening cross-clip coherence. To alleviate discontinuities at this boundary frame, we assign exponentially increasing weights {0.05, 0.325, 0.757} to the latents of the first three generated frames, encouraging smoother earlyframe transitions within each clip. In addition, we regularize the first latent of the predicted frames from complementary frequency perspectives to further stabilize this boundary frame. To jointly reinforce reconstruction fidelity and main-

tain smooth temporal boundaries, we introduce three loss terms as described below:

(1) History context consistency. We encourage the first predicted latent to align with the last latent of history context:

Lcons = zH−1 − zˆ0 2 , (6)

where zH−1 is the last latent of history segment and zˆ0 is the first predicted latent.

(2) Degradation consistency. To maintain structural coherence with degraded inputs and stabilize clip boundaries, we impose low-frequency consistency:

Ldeg = Flp z ˜I0 − Flp z ˆ0 2 , (7)

where z˜I0 is the latent of the degraded first image and Flp(·) denotes a low-frequency extraction operator.

(3) Ground-truth high-frequency alignment. To preserve fine-grained appearance details, we align highfrequency components with the ground truth:

Lgt = Fhp zgt0 − Fhp z ˆ0 2 , (8) where Fhp(·) denotes a high-frequency operator.

The final temporal regularization objective is defined as:

Ltemp = λdegLdeg + λgtLgt + λconsLcons. (9)

In practice, we set λdeg = 0.2, λgt = 0.15 and λcons = 0.5. To further capture the causal dependencies of historical information, we additionally update all parameters in the self-attention layers of the base model. The number of history frames NH is uniformly sampled from [0,16], enabling LongVie 2 to perform unified long-video inference—whether or not historical references are available. This contextual conditioning substantially enhances temporal continuity and cross-clip consistency, resulting in smoother and more coherent long-video generation.

##### 2.4. Training-free Inter-clip Consistency Strategies

To further improve temporal coherence across adjacent clips, we introduce two complementary training-free strategies that enhance inter-clip consistency.

Unified Noise Initialization. To further enhance temporal stability and consistency during video generation, we adopt a unified noise initialization across all video clips. Rather than sampling a new noise latent for each clip, a single shared noise instance is maintained throughout the entire sequence, providing a coherent stochastic prior that strengthens temporal continuity and overall coherence.

Global Normalization. Independent normalization of control inputs across clips often causes temporal discontinuities, as each segment may adopt a different depth scale. To mitigate this, we employ a global normalization strategy applied to the entire depth sequence. Specifically, we compute the 5th and 95th percentiles of all pixel values across the full video and use them as the global minimum and maximum bounds. The depth values are then clipped to this range and linearly scaled to [0,1]. This percentile-based normalization is robust to outliers and ensures a consistent depth scale across clips. After normalization, the depth sequence is divided into overlapping clips to match the autoregressive inference process and enable point-map extraction.

#### 3. Experiments

Implementation Details. The video diffusion backbone of LongVie 2 is Wan2.1-I2V-14B [40]. We replicate the first 12 DiT blocks to construct a dedicated control branch and further split each copied block along the feature dimension to form dense and sparse control pathways, effectively reducing computational overhead. During training, we first extract depth maps using Video Depth Anything [7] as dense control signals, and then apply SpatialTracker [41] to estimate and track 3D points from the normalized depth. Following DAS [13], we uniformly sample 4,900 points per clip as sparse control inputs. In total, around 100000 videos are used for training. In the first and second stages, approximately 60000 videos from ACID [26], Vchitect-T2VDataVerse [11], and MovieNet [21] are employed. Each video is divided into 81-frame clips with a resolution of 352 × 640 at 16 fps to save memory and improve efficiency. LongVie 2 is optimized with AdamW at a learning rate of 1 × 10−5, updating only the control branch parameters. In the final stage, longer videos are incorporated to enable temporal reasoning over extended sequences. About 40000 videos from OmniWorld [57] and SpatialVID [38] are used, all downsampled to 352 × 640 at 16 fps. This stage employs AdamW with a learning rate of 5 × 10−6 and updates the parameters in both the self-attention layers and the control modules. All three stages are conducted on 16 A100 GPUs with each GPU using a batch size of 1, for approximately

5000, 1000, and 2000 iterations, respectively. The full training process takes about two days.

LongVGenBench. To address the absence of suitable benchmarks for controllable long video generation, we introduce LongVGenBench—a dataset of 100 one-shot videos, each exceeding one minute in duration at 1080p resolution. Existing datasets fall short in this regard, as they lack long, continuous, one-shot videos that are essential for evaluating temporal consistency and controllability. LongVGenBench covers diverse real-world and game-based scenarios and includes challenging cases such as rapid scene transitions and complex object motions (see the supplementary material for details), establishing it as a comprehensive and demanding benchmark for long video generation. For evaluation, each video is divided into 81-frame clips at 16 fps with a one-frame overlap, following the autoregressive setup used in our experiments. Captions are automatically generated by Qwen2.5-VL-7B [1] to serve as text prompts, and corresponding control signals are extracted from each clip. During validation, the first frame of each video is left unaltered to ensure fair comparison and to enable accurate assessment of generation quality against ground-truth references.

##### 3.1. Qualitative Comparison

To comprehensively and fairly evaluate the performance of LongVie 2, we employ both objective and subjective metrics for comparison with existing baselines.

Objective Evaluation. We first evaluate the effectiveness of LongVie 2 by comparing it against three categories of baselines: (1) Pretrained video models: Wan2.1 [43]; (2) Controllable models: VideoComposer [39], Go-with-theFlow [3], DAS [13], and Motion-I2V [34]; (3) World models: Hunyuan-GameCraft [24] and Matrix-Game [16]. Following the widely adopted VBench protocol [23], we evaluate performance using seven metrics—Background Consistency (B.C.), Subject Consistency (S.C.), Overall Consistency(Q.C.), Dynamic Degree (D.D.), Aesthetic Quality (A.Q.), and Imaging Quality (I.Q.)—to assess temporal coherence and visual fidelity. We also report similarity-based metrics, including SSIM and LPIPS, to measure the controllability of the generated videos.

As shown in Tab. 1, LongVie 2 achieves substantial improvements over controllable baselines in controllability, and surpasses existing other baselines in both temporal consistency and visual quality. These results demonstrate that LongVie 2 effectively maintains high controllability while producing long videos of superior visual coherence and quality, achieving state-of-the-art performance.

Human Evaluation. To further assess the perceptual quality of long video generation, we conduct a carefully designed human evaluation study. To mitigate participant fatigue, the study follows a balanced and randomized sampling protocol. From all generated videos, we randomly select 80 samples,

- Table 1. Quantitative comparison of LongVie 2 and baselines on LongVGenBench. We compare LongVie 2 with base diffusion models, controllable generation models, and world models on LongVGenBench, evaluating visual quality, controllability, and long-term consistency. Bold indicates the best performance, and underline denotes the second-best.

METHODS

Quality Controllability Temporal Consistency

A.Q.↑ I.Q.↑ SSIM↑ LPIPS↓ S.C.↑ B.C.↑ O.C.↑ D.D.↑

Wan2.1 [40] 49.72% 63.78% 0.406 0.488 83.56% 88.86% 20.30% 15.15% VideoComposer [39] 43.53% 59.33% 0.346 0.583 80.33% 88.30% 19.83% 27.78% Motion-I2V [34] 48.34% 61.57% 0.385 0.504 82.35% 89.25% 20.46% 44.13% Go-With-The-Flow [3] 53.59% 62.21% 0.453 0.394 84.37% 90.62% 21.79% 46.15% DiffusionAsShader [13] 53.28% 64.57% 0.401 0.482 86.06% 90.78% 21.10% 36.76% Matrix-Game-2.0 [16] 55.24% 59.23% 0.427 0.501 77.82% 76.43% 19.37% 74.45% HunyuanGameCraft [24] 56.18% 67.73% 0.483 0.386 79.12% 74.24% 20.94% 80.46% LongVie 2 (Ours) 58.47% 69.77% 0.529 0.295 91.05% 92.45% 23.37% 82.95%

SourceLongVieDASGWF

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

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Figure 7. Controllability comparison between LongVie 2 and baselines. LongVie 2 demonstrates markedly superior controllability, maintaining precise structural alignment and realistic appearance across frames compared with the baselines. GWF denotes Go-With-TheFlow, and DAS denotes Diffusion as Shader.

- Table 2. Human Evaluation Results. Sixty participants rated LongVie 2 and baseline models across five evaluation dimensions. LongVie 2 consistently achieved the highest scores in all aspects.

controllability.

##### 3.2. Quantitative Results

METHOD VQ PVC CC ColC TC

To comprehensively demonstrate the performance of LongVie 2, we present results from two key aspects: controllability and long-term quality & consistency.

Matrix-Game-2.0 [16] 2.12 2.08 2.27 2.23 2.37 Go-With-The-Flow [3] 2.34 2.32 2.18 2.26 2.02 DiffusionAsShader [13] 3.03 3.07 3.09 3.02 2.79 HunyuanGameCraft [24] 3.11 3.14 3.28 3.37 3.35 LongVie (Ours) 4.40 4.39 4.18 4.12 4.53

Controllability Comparison. To verify that LongVie 2 preserves controllability, we directly evaluate it on standard controllable video generation tasks and compare it with strong baselines, including DAS [13] and GWF [3]. As illustrated in Fig. 7, our model consistently outperforms the baselines across three representative sub-tasks: motion transfer, style transfer, and mesh-to-video generation. It can be observed that LongVie 2 achieves superior controllability and produces finer visual details under the same settings.

each paired with its corresponding prompt and control signals. Participants evaluate five key aspects: Visual Quality (VQ), Prompt–Video Consistency (PVC), Condition Consistency (CC), Color Consistency (ColC), and Temporal Consistency (TC). We compare five representative models—GoWith-The-Flow [3], DAS [13], HunyuanGameCraft [24], Matrix-Game [16], and LongVie 2. A total of 60 participants are recruited. For each evaluation aspect, they rank the models from best to worst, assigning scores from 5 to 1 accordingly. The averaged results, summarized in Tab. 2, show that LongVie 2 consistently attains the highest scores across all criteria, demonstrating its superior perceptual quality and

Long-Term Generation Results. We further present longduration generation results of LongVie 2 in Fig. 8. Each row depicts a video exceeding two minutes in length, demonstrating the model’s ability to maintain high visual fidelity and long-term temporal coherence. Notably, under worldlevel guidance, the model can simulate realistic physical phenomena such as the flow change after turning a faucet

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

00:20 00:40 01:00 01:20 01:40 02:00

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

00:20 00:40 01:00 01:20 01:40 02:00

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

00:20 00:40 01:00 01:20 01:40 02:00

- Figure 8. Long-term generation demos. We present several videos generated by LongVie 2, each lasting over two minutes, demonstrating sustained visual quality and long-term temporal consistency, further validating the effectiveness of our training strategy.

- Table 3. Ablation on staged training of LongVie 2. We progressively introduce three stages: Control Learning, Degradation Adaptation, and History Context. Each stage consistently enhances visual quality, controllability, and long-term consistency.

TRAINING STRATEGY

Quality Controllability Temporal Consistency

A.Q.↑ I.Q.↑ SSIM↑ LPIPS↓ S.C.↑ B.C.↑ O.C.↑ D.D.↑

Base Model 49.72% 63.78% 0.406 0.488 83.56% 88.86% 20.30% 15.15% + Control Learning 51.36% 64.32% 0.456 0.376 85.94% 90.42% 20.95% 73.62% + Degradation Adaptation 54.08% 66.21% 0.501 0.328 88.12% 91.05% 21.56% 76.12% + History Context 58.47% 69.77% 0.529 0.295 91.05% 92.45% 23.37% 82.59%

- Table 4. Ablation study for our proposed components. The block denotes experiments targeting temporal consistency, while the block denotes those focusing on visual quality.

stage further enhances visual quality through degradationaware learning, while the third stage notably improves global temporal consistency by incorporating history-context guidance. These results confirm that our stage-wise training strategy is both necessary and effective.

METHODS A.Q. I.Q. S.C. B.C. Full model 58.47% 69.77% 91.05% 92.45%

Unified Initial Noise & Global Normalization. We further analyze the influence of unified initial noise and global normalization of control signals on temporal stability and controllability. Videos are generated under three configurations: (1) without Global Normalization, (2) without Unified Initial Noise, and (3) without both. As shown in Tab. 4, removing either component leads to performance drops across all metrics, indicating that unified noise initialization and global normalization are both beneficial for consistent and high-quality controllable long video generation.

w/o Global Normalization 57.73% 69.56% 88.81% 91.41% w/o Unified Initial Noise 57.80% 69.62% 88.73% 91.59% w/o Both 57.03% 68.89% 88.56% 91.37%

w/o Feature Degradation 56.92% 67.23% 89.94% 92.15% w/o Data Degradation 57.11% 67.84% 89.71% 92.08% w/o Both 56.74% 67.01% 89.57% 91.99%

on or off, highlighting its capability for world-consistent video synthesis and further demonstrating the effectiveness of LongVie 2. Additional long-form results are provided in the supplementary material for a more comprehensive evaluation.

#### 4. Related Works

Controllable Video Generation. Recent advances in controllable video generation have enabled the synthesis of visually compelling and semantically aligned videos under diverse structural and motion constraints. VideoComposer [39] enhances controllability through multiple conditioning signals. Following the ControlNet [52] paradigm, SparseCtrl [14] introduces sparse structural guidance for fine-grained control, while Go-With-The-Flow [3] and MotionI2V [34] utilize optical flow to direct temporal motion. DAS [13] further employs 3D point maps for precise spatial and motion control. More recently, Cosmos-Transfer-1 [30] and LongVie [12] integrate multi-modal control signals to jointly

##### 3.3. Ablation Study

We conduct ablation studies to verify the effectiveness of each component and training strategy adopted in our method. Stage-Wise Training. LongVie 2 is trained in three progressive stages, each designed to enhance different aspects of video generation. To examine the contribution of each stage, we systematically remove them and report the results in Tab. 3. The first stage yields a clear improvement in SSIM and LPIPS, indicating stronger controllability. The second

improve visual fidelity and alignment in controllable video synthesis.

World Models. Recent advances in video world modeling are powered by large-scale transformer and diffusion architectures that generate high-fidelity, temporally coherent videos conditioned on actions or camera trajectories. A series of world models [2, 8, 16, 19, 24, 29, 44, 50, 53, 58] follow this paradigm, focusing on camera-control-based simulation of dynamic environments. Leveraging large-scale video or gameplay data, these systems learn controllable and physically plausible dynamics, marking an important step toward general-purpose video world models capable of synthesizing consistent and interactive scenes under explicit control.

Long Video Generation. Generating long, coherent videos remains a fundamental challenge due to accumulated temporal drift and quality degradation over extended horizons. StreamingT2V [17] adopts an autoregressive strategy to progressively extend video length. Diffusion Forcing [6, 35, 49] introduces causal constraints to improve temporal stability, while TTT [10], LCT [15], and MoC [4] explicitly incorporate long-range contextual dependencies to enhance visual and temporal consistency. More recently, SVI [25] proposes an error-recycling fine-tuning paradigm that exposes the model to its own autoregressive outputs, thereby mitigating the mismatch between short-clip training and long-form inference. In parallel, Self-Forcing [22] leverages video-level distribution-matching objectives [47, 48], together with related extensions [9, 27, 28, 42, 45, 46, 51, 54, 59], to achieve long-horizon stability through autoregressive rollout and selfconsistent training. Inspired by the growing emphasis on narrowing the train-test discrepancy, we introduce a simple yet effective approach that simulates long-form degradation directly on the input frames during training, enabling the model to better preserve visual fidelity and temporal coherence over extended durations.

#### 5. Conclusion

In this work, we explore how to extend pretrained video diffusion models toward building general-purpose video world models capable of generating coherent, controllable, and temporally consistent long videos. We empirically demonstrate that strengthening controllability before enhancing long-term visual quality is an effective and practical strategy, and we propose LongVie 2, a three-stage autoregressive framework that progressively improves controllability, visual fidelity, and long-term consistency. First, we integrate dense and sparse controls to inject world-level guidance, providing complementary structural and semantic cues. Second, a degradation-aware training strategy applied to first-frame inputs bridges the gap between training and long-horizon inference. Finally, history context guidance models longrange dependencies to maintain temporal coherence across

extended sequences. To enable comprehensive evaluation, we establish LongVGenBench, a benchmark containing 100 one-minute videos covering diverse real-world and synthetic environments. Extensive experiments demonstrate that LongVie 2 achieves state-of-the-art performance in controllable long video generation and can autoregressively synthesize high-quality videos lasting up to 3–5 minutes, marking a significant step toward video world modeling.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5vl technical report. arXiv preprint arXiv:2502.13923, 2025. 6, 1
- [2] Philip J. Ball et al. Genie 3: A new frontier for world models.

2025. 9

- [3] Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski, Pascal Clausen, Mingming He, Li Ma, Yitong Deng, Lingxiao Li, Mohsen Mousavi, Michael Ryoo, Paul Debevec, and Ning Yu. Go-with-the-flow: Motion-controllable video diffusion models using real-time warped noise. In CVPR, 2025. Licensed under Modified Apache 2.0 with special crediting requirement. 6, 7, 8
- [4] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, et al. Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058, 2025. 9
- [5] Brandon Castellano. Pyscenedetect. https://github. com/Breakthrough/PySceneDetect, 2024. Video Cut Detection and Analysis Tool. Available at https:// www.scenedetect.com. Accessed: 2025-05-19. 1
- [6] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024. 9
- [7] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. arXiv:2501.12375, 2025. 6, 1
- [8] Taiye Chen, Xun Hu, Zihan Ding, and Chi Jin. Learning world models for interactive video generation. arXiv preprint arXiv:2505.21996, 2025. 9
- [9] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Selfforcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025. 9
- [10] Karan Dalal, Daniel Koceja, Gashon Hussein, Jiarui Xu, Yue Zhao, Youjin Song, Shihao Han, Ka Chun Cheung, Jan Kautz, Carlos Guestrin, Tatsunori Hashimoto, Sanmi Koyejo, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training, 2025. 9

- [11] Weichen Fan, Chenyang Si, Junhao Song, Zhenyu Yang, Yinan He, Long Zhuo, Ziqi Huang, Ziyue Dong, Jingwen He, Dongwei Pan, et al. Vchitect-2.0: Parallel transformer for scaling up video diffusion models. arXiv preprint arXiv:2501.08453, 2025. 1, 6
- [12] Jianxiong Gao, Zhaoxi Chen, Xian Liu, Jianfeng Feng, Chenyang Si, Yanwei Fu, Yu Qiao, and Ziwei Liu. Longvie: Multimodal-guided controllable ultra-long video generation,

2025. 8

- [13] Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou, Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei Liu, Wenping Wang, and Yuan Liu. Diffusion as shader: 3daware video diffusion for versatile video generation control. arXiv preprint arXiv:2501.03847, 2025. 3, 6, 7, 8
- [14] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In European Conference on Computer Vision, pages 330–348. Springer, 2024. 8
- [15] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation, 2025. 9
- [16] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, Baixin Xu, Hao-Xiang Guo, Kaixiong Gong, Cyrus Wu, Wei Li, Xuchen Song, Yang Liu, Eric Li, and Yahui Zhou. Matrix-game 2.0: An open-source, real-time, and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025. 2, 6, 7, 9
- [17] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text, 2025. 9
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 1
- [19] Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, et al. Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025. 9
- [20] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. 2
- [21] Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and Dahua Lin. Movienet: A holistic dataset for movie understanding. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16, pages 709–727. Springer, 2020. 6, 1
- [22] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009,

2025. 9

- [23] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In

- Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 6
- [24] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition, 2025. 2, 6, 7, 9
- [25] Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi. Stable video infinity: Infinite-length video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025. 9
- [26] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 6, 1
- [27] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025. 9
- [28] Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678, 2025. 9
- [29] Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025. 2, 9
- [30] NVIDIA. Cosmos-transfer1: Conditional world generation with adaptive multimodal control, 2025. 8
- [31] OpenAI. Sora. Accessed February 15, 2024 [Online] https: //sora.com/library, 2024. 1
- [32] Adam Polyak, Amit Zohar, et al. Movie gen: A cast of media foundation models, 2025. 1
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 1, 3
- [34] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 6, 7, 8
- [35] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion, 2025. 9
- [36] Hunyuan Foundation Model Team. Hunyuanvideo: A systematic framework for large video generative models, 2024. 1
- [37] Kuaishou Team. Kling. Accessed December 9, 2024 [Online] https://klingai.kuaishou.com/, 2024. 1
- [38] Jiahao Wang, Yufeng Yuan, Rujie Zheng, Youtian Lin, Jian Gao, Lin-Zhuo Chen, Yajie Bao, Yi Zhang, Chang Zeng, Yanxi Zhou, et al. Spatialvid: A large-scale video dataset with spatial annotations. arXiv preprint arXiv:2509.09676,

2025. 6, 1

- [39] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis

- with motion controllability. Advances in Neural Information Processing Systems, 36:7594–7611, 2023. 6, 7, 8
- [40] WanTeam. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 6, 7, 2
- [41] Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker: Tracking any 2d pixels in 3d space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 6, 1
- [42] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, Song Han, and Yukang Chen. Longlive: Realtime interactive long video generation, 2025. 9
- [43] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 6
- [44] Deheng Ye, Fangyun Zhou, Jiacheng Lv, Jianqi Ma, Jun Zhang, Junyan Lv, Junyou Li, Minwen Deng, Mingyu Yang, Qiang Fu, Wei Yang, Wenkai Lv, Yangbin Yu, Yewen Wang, Yonghang Guan, Zhihao Hu, Zhongbin Fang, and Zhongqian Sun. Yan: Foundational interactive video generation, 2025. 9
- [45] Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-rope: Action-controllable infinite video generation emerges from autoregressive selfrollout. arXiv preprint arXiv:2511.20649, 2025. 9
- [46] Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep forcing: Training-free long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081, 2025. 9
- [47] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024. 9
- [48] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024. 9
- [49] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025. 9
- [50] Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Gamefactory: Creating new games with generative interactive videos, 2025. 9
- [51] Yifei Yu, Xiaoshan Wu, Xinting Hu, Tao Hu, Yangtian Sun, Xiaoyang Lyu, Bo Wang, Lin Ma, Yuewen Ma, Zhongrui Wang, et al. Videossm: Autoregressive long video generation with hybrid state-space memory. arXiv preprint arXiv:2512.04519, 2025. 9
- [52] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 2, 3, 8
- [53] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric

- Li, Yang Liu, et al. Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701, 2025. 9
- [54] Zeyu Zhang, Shuning Chang, Yuanyu He, Yizeng Han, Jiasheng Tang, Fan Wang, and Bohan Zhuang. Blockvid: Block diffusion for high-quality and consistent minute-long video generation, 2025. 9
- [55] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 1
- [56] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. ACM Trans. Graph. (Proc. SIGGRAPH), 37, 2018. 1
- [57] Yang Zhou, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Haoyu Guo, Zizun Li, Kaijing Ma, Xinyue Li, Yating Wang, Haoyi Zhu, et al. Omniworld: A multi-domain and multi-modal dataset for 4d world modeling. arXiv preprint arXiv:2509.12201, 2025. 6, 1
- [58] Yixuan Zhu, Jiaqi Feng, Wenzhao Zheng, Yuan Gao, Xin Tao, Pengfei Wan, Jie Zhou, and Jiwen Lu. Astra: General interactive world model with autoregressive denoising, 2025. 9
- [59] Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, and Tianfan Xue. Flashvsr: Towards real-time diffusion-based streaming video super-resolution. arXiv preprint arXiv:2510.12747, 2025. 9

## LongVie 2: Multimodal Controllable Ultra-Long Video World Model Supplementary Material

#### 6. Overview of Supplementary Material

This supplementary document provides additional technical details, datasets, and qualitative analyses to complement the main paper. In Sec. 7, we elaborate on the implementation details of LongVie, including the training data, inference adaptation strategy, and model configuration. In Sec. 8, we offer a comprehensive description of the LongVGenBench dataset used for evaluating long-term consistency and controllability. In Sec. 9, we conduct ablation studies on the key components of our frame degradation training strategy. In Sec. 10, we provide additional qualitative results under diverse styles and scenarios, including ultra-long video cases.

#### 7. More Implementation Details 7.1. Training Data

Data for Stage 1 & Stage 2. As described in the Implementation Details section of the main paper, Stages 1 and 2 are trained on approximately 60,000 videos sourced from three complementary datasets. (1) ACID and ACID-Large [26] provide thousands of aerial drone videos capturing diverse coastlines, natural landscapes, and outdoor environments. The videos follow the same format as RealEstate10K [56], ensuring consistent metadata and camera-trajectory structure.

- (2) Vchitect_T2V_DataVerse [11] is a large-scale corpus of more than 14 million high-quality Internet videos, each paired with descriptive and fine-grained textual annotations, offering rich appearance and motion diversity for training.
- (3) MovieNet [21] contains 1,100 full-length movies spanning multiple decades, regions, and genres, providing complex scenes, camera motions, and human-centric activities. To standardize temporal resolution and facilitate stable training, all videos from these sources are uniformly converted into 81-frame clips sampled at 16 fps. Data for Stage 3. Stage 3 focuses on long-horizon temporal modeling and therefore requires explicit history context inputs. To support this stage, we incorporate long-form videos from OmniWorld [57] and SpatialVID [38], which contain extended sequences with coherent scene dynamics. For each video, we extract 81-frame target segments beginning from the 20th frame; these segments serve as the prediction targets for the model. The corresponding control signals and textual prompts are generated following the same pipeline as Stages 1 and 2. During training, all frames preceding each 81-frame segment are loaded as the history context, enabling the model to learn long-range dependencies and temporal evolution. From the complete set of processed samples, we randomly select 40,000 segments to construct the Stage 3

training split.

Data Pre-processing. During our experiments, we observed that the presence of scene transitions (e.g. cuts or abrupt changes in viewpoint) in the training data can significantly undermine temporal consistency during long-horizon generation, often leading to undesired transitions in the synthesized videos. To alleviate this issue, we apply the PySceneDetect toolkit [5] to automatically identify scene boundaries and segment each raw video into transition-free clips. Each resulting segment is uniformly sampled at 16 fps and truncated to 81 consecutive frames to form a stable and temporally coherent training unit for LongVie. For each 81-frame clip, we compute a comprehensive set of control signals. Depth maps are extracted using Video Depth Anything [7], and point trajectories are estimated with SpatialTracker [41], leveraging both the depth information and RGB frames. In addition, we generate descriptive captions for every clip using Qwen-2.5-VL-7B [1] to provide semantically rich and contextually aligned textual guidance. After applying this unified pre-processing pipeline to all videos, we obtain a curated corpus of approximately 100,000 video–control signal pairs. This dataset serves as the complete training foundation for LongVie, enabling consistent learning across diverse visual domains and motion patterns.

##### 7.2. Inference Preparation.

Point tracking naturally degrades during long-horizon inference because the tracked points depend on content visible in the first frame. As the video evolves—particularly when the originally tracked objects move out of view—the sparse tracks become unreliable. In our framework, point tracking acts as a sparse motion-control signal rather than an appearance cue. To preserve its effectiveness, we do not compute point tracks directly over the entire minute-long video. Instead, we first extract depth maps for the full sequence and apply global normalization across the entire duration. The video is then partitioned into overlapping 81-frame clips, and colorful point tracks are computed independently for each clip using the globally normalized depth, ensuring stable and consistent motion guidance throughout the generation process. A similar challenge arises with captions. In transfer scenarios, the original captions of the source videos often become semantically misaligned with the visual content after transformation. To address this mismatch and remain consistent with our training pipeline, we apply Qwen-2.5VL-7B [1] to analyze the differences between the transferred frames and the original sequence. The model then revises the captions to more accurately reflect the updated visual semantics, as illustrated in Fig. 9.

introduced during the final 1000 iterations.

[Figure 177]

For weight initialization, the dense and sparse branches within each MM Block inherit only half of the original weights from Wan2.1 [40]. Specifically, we interleave the pretrained weights by index—assigning weights at positions (0, 2, 4, ...) to the dense branch and those at positions (1, 3, 5, ...) to the sparse branch—while simultaneously halving the feature dimensionality. This “half-copy” initialization substantially reduces the total parameter count and provides a stable starting point for joint dense–sparse control learning.

Please revise the following caption so that it accurately matches the visual content and semantic context of the given image. Ensure the caption maintains a consistent style and tone with other video captions, describing what is happening in the image in a natural and coherent way.

Caption: The video depicts a 3D animated scene featuring a dragon perched atop a large, irregularly shaped rock formation. The dragon is rendered in grayscale, giving it a monochromatic appearance that emphasizes its form and structure. Its wings are spread wide, suggesting readiness for flight or perhaps a moment of rest after taking off …

[Figure 178]

#### 8. LongVGenBench

To better contextualize our proposed evaluation dataset, LongVGenBench, which is designed to assess both controllability and long-term quality and coherence in video world models, we present several representative examples in Fig. 10. The dataset consists of diverse real-world and synthetic scenes, each lasting at least one minute and captured at a minimum resolution of 1080p. As illustrated in the figure, LongVGenBench encompasses a wide range of camera motions, presenting substantial challenges for existing video generation models. Notably, the dataset is model-agnostic and can be used to evaluate any long-horizon video world model, not only LongVie. In this paper, we utilize LongVGenBench by splitting each video into 81-frame clips with a one-frame overlap, followed by extracting captions and control signals for each clip. These short-clip captions are then used to construct the corresponding inference data.

[Figure 179]

The video showcases a 3D animated scene where a majestic dragon is perched atop a large, irregularly shaped rock formation. The dragon is depicted in vibrant orange hues, with its wings spread wide, suggesting either readiness for flight or a moment of rest after taking off. The rock formation is lush and green, with cascading waterfalls flowing down its sides, creating a serene and mystical atmosphere …

- Figure 9. Caption Refinement via MLLM. Given a new firstframe image and its original caption, we use Qwen-2.5-VL to refine the caption so that the prompt for video generation more accurately matches the visual content and remains stylistically consistent with the input image.

#### 9. Additional Ablation Studies

##### 7.3. Model Configuration.

Ablation for Degradation To further validate the effectiveness of our degradation strategy, we perform ablation studies on two degradation types: Encoding Degradation and Generation Degradation. The corresponding results are presented in Tab. 5. These experiments demonstrate the contribution of each component to our overall frame degradation training pipeline and highlight their complementary effects.

We provide additional implementation details of our LongVie model. During training, we apply input-frame degradation with a probability of 20%. Within this process, VAE-based degradation is selected 20% of the time, while generation degradation is used for the remaining 80%. For encoding degradation, the parameter k is sampled from [0, 10], with smaller values assigned higher sampling probability to simulate stronger degradation. For generation degradation, we select t ∈ {1,5,8,10,12}, again with a probability distribution favoring smaller values to emulate more severe degradation levels. In Stage 3, when history context is introduced, we apply the same degradation strategies to ensure consistent robustness across both current-frame and historyframe inputs.

#### 10. More Qualitative Results

To further demonstrate the robustness and versatility of our LongVie model, we provide additional qualitative results across multiple long video generation scenarios, including 1-minute, 3-minute, and 5-minute cases. These examples highlight the model’s ability to maintain high visual quality, preserve global consistency, and adapt to diverse styles over extended temporal horizons.

In addition to frame-level degradation, we set the featurelevel degradation probability α to 15% and the data-level degradation probability β to 10%. At each training step, one or both degradation types are randomly activated. We adopt a fusion count of n = 5 for the Random Scale Fusion module. To stabilize training, all degradation strategies are disabled during the first 2000 iterations and are gradually

1-minute Cases. For the 1-minute videos, we showcase two representative settings: (1) a subject-driven scenario (a person riding a horse), and (2) a subject-free scenario captured from a drone perspective. Both settings are evaluated under three seasonal style transfers. As illustrated in Fig. 11 and Fig. 12, LongVie consistently preserves scene layout, motion

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

00:10 00:20 00:30 00:40 00:50 01:00

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

00:10 00:20 00:30 00:40 00:50 01:00

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

00:20 00:40 01:00 01:20 01:40 02:00

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

00:20 00:40 01:00 01:20 01:40 02:00

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

00:30 01:00 01:30 02:00 02:30 03:00

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

00:30 01:00 01:30 02:00 02:30 03:00

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

00:50 01:40 02:30 03:20 04:10 05:00

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

00:50 01:40 02:30 03:20 04:10 05:00

- Figure 10. Examples from LongVGenBench. We show several videos from both real-world and synthetic scenarios in LongVGenBench, covering a variety of indoor and outdoor environments to evaluate the controllable long video generation ability of our model.

Table 5. Ablation on the Degradation training of LongVie. We progressively introduce three stages: Control Learning, Degradation Adaptation, and History Context. Each stage consistently enhances visual quality, controllability, and long-term consistency.

Quality Controllability Temporal Consistency

TRAINING STRATEGY

A.Q.↑ I.Q.↑ SSIM↑ LPIPS↓ S.C.↑ B.C.↑ O.C.↑ D.D.↑

Base Model 49.72% 63.78% 0.406 0.488 83.56% 88.86% 20.30% 15.15% + Encoding Degradation 52.18% 64.95% 0.451 0.402 85.72% 89.63% 20.66% 47.38% + Generation Degradation 53.96% 65.87% 0.478 0.356 87.64% 90.74% 21.14% 62.14% + Both 54.08% 66.21% 0.501 0.328 88.12% 91.05% 21.56% 76.12%

dynamics, and semantic integrity while accurately adapting global textures, lighting, and seasonal atmospheres across different styles.

3-minute Cases. For the 3-minute scenarios, we evaluate long-term consistency under seasonal style variations. As shown in Fig. 13, LongVie maintains coherent scene semantics and smooth transitions across all three seasonal styles.

5-minute Cases. To highlight LongVie’s capability in ultralong video generation, we present two 5-minute examples covering both a subject-driven and a subject-free scenario. As shown in Fig. 14, LongVie preserves global structure, maintains coherent motion behavior, and sustains stable visual appearance throughout the entire 5-minute duration.

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

SourceStyle1Style2Style3

00:05 00:10 00:15 00:20 00:25 00:30

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

00:05 00:10 00:15 00:20 00:25 00:30

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

00:05 00:10 00:15 00:20 00:25 00:30

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

00:05 00:10 00:15 00:20 00:25 00:30

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

SourceStyle1Style2Style3

00:35 00:40 00:45 00:50 00:55 01:00

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

00:35 00:40 00:45 00:50 00:55 01:00

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

00:35 00:40 00:45 00:50 00:55 01:00

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

00:35 00:40 00:45 00:50 00:55 01:00

- Figure 11. Subject-driven 1-minute scenario. A man riding a horse, transferred into multiple seasonal styles while preserving motion dynamics and scene structure.

100

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

SourceStyle1Style2Style3

00:05 00:10 00:15 00:20 00:25 00:30

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

00:05 00:10 00:15 00:20 00:25 00:30

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

00:05 00:10 00:15 00:20 00:25 00:30

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

00:05 00:10 00:15 00:20 00:25 00:30

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

SourceStyle1Style2Style3

00:35 00:40 00:45 00:50 00:55 01:00

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

00:35 00:40 00:45 00:50 00:55 01:00

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

00:35 00:40 00:45 00:50 00:55 01:00

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

00:35 00:40 00:45 00:50 00:55 01:00

- Figure 12. Subject-free scenario. A drone-like car-drive sequence through a mountain valley, transferred across different seasonal styles with consistent global appearance.

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

SourceStyle1Style2Style3

- 00:15 00:30 00:45 01:00 01:15 01:30

- 01:45 02:00 02:15 02:30 02:45 03:00

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

- 00:15 00:30 00:45 01:00 01:15 01:30

- 01:45 02:00 02:15 02:30 02:45 03:00

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

- 00:15 00:30 00:45 01:00 01:15 01:30

- 01:45 02:00 02:15 02:30 02:45 03:00

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

- 00:15 00:30 00:45 01:00 01:15 01:30

- 01:45 02:00 02:15 02:30 02:45 03:00

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

SourceStyle1Style2Style3

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

- Figure 13. 3-minute scenario. A car driving through a mountain valley is transferred into multiple seasonal styles, demonstrating LongVie’s ability to maintain long-range temporal consistency and coherent global appearance over extended durations.

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

00:15 00:30 00:45 01:00 01:15 01:30

- 00:15 00:30 00:45 01:00 01:15 01:30

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

03:15 03:30 03:45 04:00 04:15 04:30

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

- 01:45 02:00 02:15 02:30 02:45 03:00

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

03:15 03:30 03:45 04:00 04:15 04:30

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

01:45 02:00 02:15 02:30 02:45 03:00

- Figure 14. 5-minute scenarios. We present two ultra-long video cases: a subject-driven sequence and a subject-free sequence. These results illustrate LongVie’s robustness in preserving structural stability, motion coherence, and style consistency across 5-minute continuous generations.

#### 11. Limitations and Future Work

LongVie presents a practical framework for adapting a pretrained video model into a controllable world model capable of generating coherent visual dynamics over 3–5 minutes. To systematically evaluate our training strategies while keeping computational costs tractable, all experiments in this paper are conducted at a resolution of 352×640. Although this setting allows for comprehensive ablations and long-horizon testing, the relatively low resolution limits the model’s ability to express fine-grained details and high-frequency structures. In future work, we aim to scale LongVie to higher resolutions in order to further improve visual fidelity, capture richer scene dynamics, and more fully showcase the potential of LongVie as a video world model.

