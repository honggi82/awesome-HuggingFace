[Figure 1]

[Figure 2]

## UnityVideo: Unified Multi-Modal Multi-Task Learning for Enhancing World-Aware Video Generation

[Figure 3]

[Figure 4]

Jiehui Huang1,† Yuechen Zhang2 Xu He3 Yuan Gao4 Zhi Cen4 Bin Xia2 Yan Zhou4 Xin Tao4 Pengfei Wan4 Jiaya Jia1

1HKUST 2CUHK 3Tsinghua University 4Kling Team, Kuaishou Technology Projects: https://jackailab.github.io/Projects/UnityVideo

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Atheistic

RGB+Anything

Segmentation RGB

Text

Dynamic Degree

Depth 𝛿 < 1.25

Depth

Trainingon human data

[Figure 9]

# arXiv:2512.07831v1[cs.CV]8Dec2025

Skeleton

Skeleton

[Figure 10]

Boosting Performance

Segmentation

UnityVideo

[Figure 11]

[Figure 12]

*

Depth Flow

[Figure 13]

Depth Abs Rel

Consistency

Flow Pose

Generalization

[Figure 14]

Pose

Video2Any,Any2Video

Testingon generaldata

Seg. mIoU

Seg. mAP

Unified Multi-ModalJointTraining

[Figure 15]

[Figure 16]

*GeneralizedResults *GeneralizedResults

Joint Generation:TexttoRGB&Anything

Estimator:VideotoAnything

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

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

RGB & DensePose

RGB & Depth

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

* *

* * *

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

RGB & Skeleton

Ref Video Depth Flow DensePose Skeleton Segmentation

RGB & Segmentation

[Figure 57]

Controllable Generation:AnythingtoVideo

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

|[Figure 66]|
|---|

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

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

Ref Depth Generated Video Ref Skeleton Generated Video Ref Seg. Generated Video Ref Flow Generated Video

Figure 1. In this paper, we present UnityVideo, a unified generalist framework for multi-task multi-modal video understanding. It supports text-to-video generation, controllable generation, and modality estimation, with strong zero-shot generalization to novel objects and styles.

### Abstract

video generation that jointly learns across multiple modalities (segmentation masks, human skeletons, DensePose, optical flow, and depth maps) and training paradigms. Our approach features two core components: (1) dynamic noising to unify heterogeneous training paradigms, and (2) a modality switcher with an in-context learner that enables unified processing via modular parameters and contextual learning. We contribute a large-scale unified dataset with 1.3M samples. Through joint optimization, UnityVideo accelerates convergence and significantly enhances zero-shot generalization to unseen data. We demonstrate that Uni-

Recent video generation models demonstrate impressive synthesis capabilities but remain limited by single-modality conditioning, constraining their holistic world understanding. This stems from insufficient cross-modal interaction and limited modal diversity for comprehensive world knowledge representation. To address these limitations, we introduce UnityVideo, a unified framework for world-aware

† This work was conducted during the author’s internship at Kling Team, Kuaishou Technology.

Corresponding Author.

tyVideo achieves superior video quality, consistency, and improved alignment with physical world constraints. Code and data can be found at: https://github.com/ dvlab-research/UnityVideo

###### TrainingLoss for RGBVideos(log-scaled)

3e-1

RGBSFT Flow Depth DensePose Unified

### 1. Introduction

Large language models (LLMs) have achieved strong generalization and reasoning ability by unifying diverse text-based modalities, including natural language, code, and mathematical expressions, within a single training paradigm [10, 22, 26, 29, 35, 41, 56]. This integration of complementary text sub-modalities improves task performance and supports emergent reasoning. Similarly, recent video foundation models show promising world modeling as scale and parameters increase [3, 19, 23, 43, 49]. However, visual scaling has largely centered on RGB video alone, analogous to training language models only on plain text. This gap motivates the question of whether unifying visual sub-modalities–such as depth, optical flow, segmentation, skeleton, and DensePose–can strengthen a model’s understanding of the physical world, as unified text learning has done for LLMs.

2e-1

TrainingSteps

1000 2000

Figure 2. Training on unified modalities benefits video generation. Unified multi-modal and multi-task joint training achieves the lowest final loss on RGB video generation, outperforming single-modality joint training and RGB finetuning baseline.

and modalities. The framework introduces a light-weight modality-adaptive learner that maps heterogeneous modality signals into a shared feature space, enabling plug-andplay selection of inputs at inference. To further improve generalization, we design an in-context learner that leverages internal contextual cues to enable text-driven video object segmentation without external detectors [28]. We also devise a dynamic noise scheduling strategy that switches among different training objectives, including joint generation, video estimation, and controllable generation, within a single training cycle to encourage cross-task synergy.

Recent work indicates that video generation can benefit from single auxiliary input, such as depth maps, optical flow, skeletons, and segmentation masks [32, 48, 55, 59]. Many approaches use a one-way interaction: conditioning RGB generation on auxiliary modalities for controllable synthesis [4, 16, 18, 51], or predicting these modalities from RGB via inverse estimation [13, 17, 45]. A few recent frameworks [1, 5, 6, 40, 53] explore bidirectional interactions and report gains in motion and geometric understanding through shared representations across modalities.

We curate OpenUni, a large-scale dataset of 1.3M multimodal video samples to enable this unified training paradigm, and construct a high-quality benchmark, UniBench. UniBench contains 30K synthetic videos and a subset of the training data, with ground-truth depth and optical flow rendered in Unreal Engine. These resources provide a solid basis for fair and comprehensive evaluation. As shown in Fig 1, UnityVideo is a general-purpose model that performs both video generation and estimation, and it generalizes in a zero-shot manner to novel objects that not provided in training data. Extensive quantitative and qualitative results demonstrate that our model outperforms existing approaches across multiple downstream tasks. Our main contributions are summarized as follows:

Despite this progress, the effect of a unified training paradigm on cross-modal interaction and world awareness remains unclear. Can joint training on multiple modalities and tasks improve reasoning, accelerate convergence, and yield emergent perception? Single-modal learning limits a model’s ability to infer physical dynamics, encouraging distribution fitting rather than reasoning. In practice, different modalities provide complementary cues: instance segmentation separates categories [20, 42, 62], DensePose distinguishes body parts [12, 34], and skeletons encode finegrained motion [30, 36]. This is demonstrated in Fig. 2, jointly learning from complementary information across diverse modalities benefits the convergence in video generation, further offers a path toward more comprehensive world modeling and improved zero-shot generalization.

- • We propose UnityVideo, a novel unified framework for integrating multiple video tasks and modalities, enabling mutual knowledge transfer, better convergence, and improved performance over single-task baselines.
- • We introduce a modality-adaptive switcher, an in-context learner, and a dynamic noise scheduling strategy that together enable efficient joint training across diverse objectives and scalability to larger datasets.
- • We construct and release OpenUni, a 1.3M-pair multimodal video dataset, and UniBench, a 30K-sample benchmark derived from Unreal Engine for fair evaluation of unified video models.

UnityVideo is presented motivated by these observations. UnityVideo is a unified framework for multimodal video generation, estimation, and joint modeling. UnityVideo integrates multiple modalities and training paradigms to accelerate convergence, enhance zeroshot generalization, and promote mutual gains across tasks

### 2. Releated Work

##### 2.1. Video Generation

Large-scale video generation has advanced world modeling and physical reasoning [7, 27, 39, 44, 49, 52], improving a model’s ability to capture physical dynamics [2, 8, 15, 24, 49]. Recent work integrates additional visual signals such as depth, camera pose, and optical flow to jointly model video [1, 5, 40]. Two main directions have emerged: (i) encoding multiple modalities into a shared latent space and using flow-matching to jointly predict video and auxiliary modalities, enabling mutual gains [5, 53]; and (ii) conditioning generation on multi-modal inputs for controllable synthesis, allowing simultaneous compliance with multiple control signals and improved visual quality [16, 18]. Despite strong results, most studies isolate either a single architecture or a single modality, limiting cross-task synergy. In contrast, we unify multi-task learning in a single framework and analyze how such unification enhances world perception and generalization.

##### 2.2. Video Reconstruction

Videos contain rich world knowledge, and classical vision methods estimate depth, camera pose, and optical flow directly from RGB [21, 40]. Recent diffusion-based approaches learn bidirectional mappings between conditions and video without external modules, revealing intrinsic bidirectional capacity in flow matching frameworks [13, 17, 37]. Representative systems such as Aether [40], GeoVideo [1], and 4DNex [9] couple video with geometric modalities, and EgoTwin [53] links skeletons and video. Bidirectional interactions also appear between video and audio [44] and between video and text [7, 54], for example UniVerse-1 for audio and video [44] and UniVid or Omni-Video for text and video. However, prior work has not fully unified diverse modalities or systematically studied their synergy, and it rarely activates in-context abilities for strong zero-shot generalization. Our approach addresses these gaps through joint training across modalities and tasks, yielding a unified model with stronger zero-shot performance and clearer insights into cross-modal coupling.

### 3. Method

UnityVideo unifies video generation and multimodal understanding within a single diffusion transformer. As illustrated in Fig. 3, the model processes RGB video Vr, text condition C, and auxiliary modality Vm through a shared DiT backbone u(·). During training, we dynamically sample task types and apply corresponding noise schedules (Sec. 3.1). To handle multiple modalities within this unified architecture, we introduce an In-Context Learner and a Modality-Adaptive Switcher (Sec. 3.2). Through progres-

sive curriculum training (Sec. 3.3), the model achieves simultaneous convergence across all tasks and modalities.

##### 3.1. Unifying Multiple Tasks

Conventional video generation models are trained for specific tasks in isolation, limiting their ability to leverage cross-task knowledge. We extend the flow matching framework [25] to support three complementary training paradigms within a single architecture. UnityVideo simultaneously handles three objectives: generating RGB videos from auxiliary modalities (u(Vr|Vm,C)), estimating auxiliary modalities from RGB videos (u(Vm|Vr)), and jointly generating both from noise (u(Vr,Vm|C)). The Vr and Vm tokens are concatenated along the width dimension and interact through the self-attention module. Following [18, 38], we incorporate 3D RoPE within the DiT backbone’s self-attention to effectively distinguish cross-modal spatiotemporal positions.

Dynamic Task Routing. To enable concurrent optimization across these three paradigms, we introduce probabilistic task selection during training. At each iteration, we sample one task type with probabilities pcond, pest, and pjoint (where pcond +pest +pjoint = 1), which determines the noise schedule applied to RGB and modality tokens at timestep t. For conditional generation, as depicted in the right part of Fig. 3, RGB tokens are gradually denoised from noise (t ∼ [0,1]) while modality tokens remain clean (t = 0). For modality estimation, RGB tokens remain clean while modality tokens are noised. For joint generation, both token types are independently corrupted with noise. We assign task probabilities inversely proportional to their learning difficulty: pcond < pest < pjoint. This strategy prevents the catastrophic forgetting common in sequential stage-wise training, allowing the model to learn all three distributions concurrently.

##### 3.2. Unifying Multiple Modalities

Joint training across different modalities can significantly enhance the performance of individual tasks, as in Fig. 2. However, processing diverse modalities with shared parameters requires explicit mechanisms to distinguish them. We introduce two complementary designs: a context learner for semantic-level modality awareness and a modality-adaptive switcher for architecture-level modulation.

In-Context Learner. To leverage the model’s inherent contextual reasoning capability, we inject modality-specific textual prompts Cm that describe the modality type (e.g., “depth map,” “human skeleton”) rather than video content. This design fundamentally differs from contentdescribing captions Cr. Given concatenated RGB tokens Vr and modality tokens Vm, we perform dual-branch crossattention separately: Vr′ = CrossAttn(Vr,Cr) for RGB fea-

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Condition Modality

𝑉 𝑉

Noise

RGB Video

Noise

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Task Select &Route

###### Modulation Parameters

###### Modulation Parameters

DiT DiT

Self-Attention

Condition Modalities

N/A N/A

RGB Video

VAE

(A) Conditional Generation

(B) Video Estimation

Dynamic Noise

𝐶 𝐶

RGB Input Layer Modality ExpertInput Layer

Dynamic Condition

Dynamic RGB

Noise Noise

Prompt CrossAttention

In-Context CrossAttention

Modality-Aware AdaLN Table

###### *L layers

DiT DiT

DiTBlocks

| | | |
|---|---|---|
| | | |

Modality Condition or N/A

Modality Condition

RGB Video or N/A

RGB Video

Feed-Forward NN

RGB Out Layer Modality Expert Output Layer

(C) Joint Generation (D) Ours : Any2Any

𝐿 𝑉 𝑉

𝐿 * Assure one input noise and one output modality

PureNoiseor Clean Latenton Modalities

N/A

NoOutputToken

Modal Adaptive Learning

Figure 3. Overview of UnityVideo. UnityVideo achieves task unification through a dynamic noise injection strategy applied to input tokens (left), and modality unification via the proposed Modality-Aware AdaLN Table (center). Specifically, Lr and Lm denote the learnable parameter tables for the RGB modality and auxiliary video-related modalities (e.g., depth, optical flow, DensePose, skeleton), respectively. Cr and Cm represent the prompt condition for RGB video content and in-context modaliy learning prompt, while Vr and Vm correspond to the token sequences from the RGB and auxiliary modalities, respectively.

tures with content captions, and Vm′ = CrossAttn(Vm,Cm) for modality features with type descriptions, before recombining them for subsequent processing. This lightweight mechanism introduces negligible computational overhead while enabling compositional generalization. For instance, training with the phrase “two persons” allows the model to generalize to “two objects” during segmentation tasks, as the model learns to interpret modality-level semantics rather than memorizing content-specific patterns. Detailed analysis is provided in the experimental section.

[Figure 91]

[Figure 92]

#### Check List ✔

OCR Optical Flow (Motion filter) Aesthetics Duration Resolution

Collect from Internet & Human

ﬁltering

Data Collection

VideoFiltering

Zero-Shot to Unlabeled data

[Figure 93]

Pretrained Models

UnityVideo

MLLM

Depth Anything V2

Detailed Caption Optical Flow Depth Segmentation Mask Dense Pose Skeleton RGB … (Scalable)

Quality Filtering

RAFT

DensePose(Meta)

Modality-Adaptive Switcher. While text-based differentiation provides semantic awareness, it may become insufficient as the number of modalities scales. We therefore introduce a learnable embedding list Lm = {L1,L2,...,Lk} for k modalities to enable explicit architecture-level modulation. Within each DiT block, AdaLN-Zero [31] produces modulation parameters (scale γ, shift β, gate α) for RGB features based on timestep embeddings. We extend this mechanism by learning modality-specific parameters: γm,βm,αm = MLP(Lm + temb), where Lm ∈ Pm is the modality embedding and temb is the timestep embedding. This design enables plug-and-play modality selection during inference. To further reduce modality confusion and stabilize outputs, we initialize modality expert input-output layers as a dedicated encoding and prediction head for each modality. Further details are provided in the Appendix A.

SAM

DWPose (IDEA)

Uniﬁed Multi-Modal Data Multi-Modal Extraction

###### 1.3M Uniﬁed World-Awareness Dataset

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

Single Human (~400K) : Dense Pose, Depth, Flow, Skeleton

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Koala36M (~500K) : Depth, Optical Flow

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

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Multi Human (~100K) : Segmentation, Skeleton

[Figure 135]

[Figure 136]

OpenS2V (~300K) : Depth, Optical Flow

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Figure 4. OpenUni dataset. OpenUni contains 1.3M pairs of unified multimodal data, designed to enrich video modalities with more comprehensive world perception.

##### 3.3. Training Strategy

Curriculum Learning for Multiple Modalities. Naively training all modalities jointly from scratch leads to slow convergence and suboptimal performance. We categorize modalities into two groups based on their spatial alignment

properties. Pixel-aligned modalities (optical flow, depth, DensePose) allow direct pixel-to-pixel correspondence with RGB frames, while pixel-unaligned modalities (segmenta-

tion, skeleton) require additional visual rendering steps. We adopt a two-stage curriculum strategy: Stage 1 trains only pixel-aligned modalities on curated single-person data, establishing a strong foundation for spatial correspondence learning. Stage 2 incorporates all modalities and diverse scene datasets, covering both human-centric and general scenarios. This progressive strategy enables the model to understand all five modalities while supporting robust zeroshot inference on unseen modality combinations.

OpenUni Dataset. Our training data comprises 1.3 million video clips spanning five modalities: optical flow, depth, dense pose, skeleton, and segmentation. As illustrated in Figure 4, we collect real-world videos from multiple sources and extract modality annotations using pre-trained models. The dataset includes 370,358 singleperson clips, 97,468 two-person clips, 489,445 clips from Koala36M [47], and 343,558 clips from OpenS2V [60], totaling 1.3M samples for training. To prevent overfitting to specific datasets or modalities, we partition each batch into four balanced groups, ensuring uniform sampling across all modalities and sources. More details on training data are provided in Appendix C.

##### 3.4. Training Objective

Following Conditional Flow Matching [25], our framework employs a dynamic training strategy that adaptively switches between three modes by selectively noising different modalities. The mode-specific losses are:

Lcond(θ;t) = E ∥uθ(rt,[m0,ctxt],t) − vr∥2 , (1)

Lest(θ;t) = E ∥uθ(mt,r0,t) − vm∥2 , (2) Ljoint(θ;t) = E ∥uθ([rt,mt],ctxt,t) − [vr,vm]∥2 , (3)

where rt = (1 − t)r0 + tr1 and mt = (1 − t)m0 + tm1 denote the interpolated latents at timestep t ∈ [0,1], with r and m representing RGB video and auxiliary modality (e.g., optical flow, depth) respectively. The velocity fields are defined as vr = r1−r0 and vm = m1−m0, where r0,m0 are clean latents encoded from real data and r1,m1 ∼ N(0,I) are independent Gaussian noise. The text conditioning ctxt is obtained from a pre-trained text encoder. Eq. (1) enables conditional generation of RGB video from auxiliary modality, Eq. (2) performs modality estimation from RGB video, and Eq. (3) jointly generates both modalities from text.

During training, each sample in a batch is randomly assigned to one of the three modes, enabling all tasks to contribute gradients within a single optimization step. This unified formulation allows seamless multi-task learning within a single architecture.

### 4. Experiment

In this section, we first provide implementation details in Sec. 4.1, followed by the main results in Sec. 4.2. We

conduct extensive benchmarks on both modality estimation and video generation tasks, comparing UnityVideo against state-of-the-art methods. The results demonstrate that UnityVideo exhibits strong unified capabilities across all settings. Subsequently, Sec. 4.3 presents ablation studies that validate the effectiveness of our design choices. Finally, we analyze the convergence behavior and zero-shot generalization ability of UnityVideo, complemented by a user study. Additional analysis of UnityVideo’s zero-shot generalization and its reasoning abilities about video modalities are provided in the Appendix B.

##### 4.1. Experimental Setup

Training Details. We use an internal DiT backbone with 10B parameters as our core architecture. Training is conducted in two stages. In the first stage, the model is trained on a human-centric dataset containing 500K video clips for 16K steps. In the second stage, we scale up training to a larger dataset of 1.3M video clips for an additional 40K steps. The model is trained with a batch size of 32 and a learning rate of 5 × 10−5. During inference, we use 50 DDIM sampling steps with a CFG scale of 7.5.

Baselines. Since our framework introduces a novel unified paradigm for video generation and estimation, no directly comparable models exist. We therefore evaluate against leading models in three related categories: (1) Video Generation: We compare with text-to-video models, including the commercial model Keling-1.6, and open-source models OpenSora [33], Hunyuan-13B [19], and Wan-2.1-13B [43]. For controllable generation, we include VACE [16] and Full-DiT [18]. We also consider models capable of jointly generating video and depth, such as Aether [40]. (2) Video Estimation: We evaluate against diffusion-based depth estimation models, including DepthCrafter [13], Geo4D [17], and Aether [40]. Additional results are provided in the Appendix. (3) Video Segmentation: We compare with two recent segmentation models that support prompt-based object segmentation, SAMWISE [11] and SeC [61].

To ensure fair comparisons, all evaluations are conducted on the public VBench [14] dataset and our newly constructed UniBench dataset, specifically designed for unified video tasks. UniBench comprises 200 high-quality samples obtained from Unreal Engine (UE) for accurate video estimation evaluation [57], and 200 manually curated samples covering diverse modalities from real-world videos [18] for controllable generation and segmentation assessments. More details are provided in Appendix C.

Evaluation Metrics. To comprehensively assess the performance of our model, we evaluate it across three categories of metrics. (1) Video Quality. We measure visual and temporal quality using multiple perceptual and consistency-based metrics [14], including subject consis-

- Table 1. Quantitative comparison of UnityVideo on controllable generation , text-to-video generation , and video estimation tasks. Best results are in bold, and second-best results are underlined. Compared to state-of-the-art methods and commercial models, UnityVideo achieves superior or competitive performance across all metrics.

Video Generation - VBench & UniBench Dataset Video Estimation - UniBench Dataset VBench Segmentation Depth Tasks Models Background Aesthetic Overall Dynamic mIoU ↑ mAP ↑ Abs Rel ↓ δ < 1.25 ↑

Consistency Quality Consistency Degree

Kling1.6 95.33 60.48 21.76 47.05 - - - OpenSora2 96.51 61.51 19.87 34.48 - - - HunyuanVideo-13B 96.28 53.45 22.61 41.18 - - - Wan2.1-14B 96.78 63.66 21.53 34.31 - - - Aether 95.28 48.25 20.26 37.32 - - 0.025 97.95

Text2Video

full-dit 95.58 54.82 20.12 49.50 - - - VACE 93.61 51.24 17.52 61.32 - - - -

Controllable Generation

Depth Video Reconstruction

depth-crafter - - - - - - 0.065 96.94 Geo4D - - - - - - 0.053 97.94

Video Segmentation

SAMWISE - - - - 62.21 20.12 - SeC - - - - 65.52 22.23 - -

Unified ControGen, UnityVideo 96.04 54.63 21.86 64.42 T2V, and Estimation UnityVideo 97.44 64.12 23.57 47.76 68.82 23.25 0.022 98.98

tency, background consistency, aesthetic quality, imaging quality, temporal flickering, motion smoothness, and dynamic degree. These metrics collectively evaluate spatial fidelity, aesthetic appeal, and temporal coherence of the generated videos. (2) Depth Estimation. For quantitative evaluation of video-based depth prediction [13], we report the absolute relative error (AbsRel) and the percentage of predicted depths within a 1.25 factor of the ground truth (δ < 1.25). (3) Video Segmentation. To evaluate segmentation accuracy [61, 62], we adopt standard metrics for both instance and semantic segmentation tasks, namely the mean Average Precision (mAP) and mean Intersectionover-Union (mIoU).

##### 4.2. Main Results

This section validates the superior performance of UnityVideo compared to single-task approaches. We comprehensively evaluate UnityVideo on text-to-video generation, controllable generation, and modality estimation tasks, demonstrating both improved generation quality and enhanced world perception capabilities.

Quantitative Comparison. As shown in Tab. 1, UnityVideo achieves competitive results across all tasks, demonstrating strong overall performance. For text-tovideo generation, we report the result of the depth-RGB joint generation. Our model obtains the best results on all metrics. We attribute this to joint training across multiple modalities, which enables collaborative refinement and enhances the model’s world perception capabilities, leading to superior video quality.

Compared to previous controllable generation methods, UnityVideo excels in background consistency, overall consistency, and dynamic degree, while maintaining compet-

itive aesthetic quality. This indicates that our model better understands and leverages control conditions, benefiting from multi-task joint training that enables the model to go beyond simply following control signals. Furthermore, through joint training with multimodal data, UnityVideo outperforms single-modality models such as Geo4D, Aether, and SeC on both video segmentation and depth estimation tasks. These results confirm that the unified training framework enhances the model’s perception and reasoning capabilities for complex visual scenes.

Qualitative Comparison. As shown in Fig. 2 (A), compared to advanced text-to-video models, UnityVideo demonstrates superior world perception. Our model exhibits stronger adherence to physical principles, more accurately reflecting the physical phenomenon of refraction. Furthermore, as shown in Fig. 2 (B), compared to advanced controllable generation methods, UnityVideo not only follows depth guidance more faithfully but also maintains overall video quality. In contrast, other methods often exhibit noticeable background flickering, with subject regions sometimes distorted by surrounding context.

For modality estimation tasks, as shown in Fig. 2 (C) and (D), UnityVideo produces finer edge details,a wider field of view, and accurate 3D point clouds, benefiting from the complementary nature of multiple modalities. Similarly, in other modality estimation tasks (Fig. 2 (E)), our model demonstrates strong reasoning capabilities, achieving accurate estimation on unseen data and overcoming the overfitting issues observed in other specialized models [12, 58].

Overall, these qualitative results confirm that joint training across multiple tasks and modalities yields significant improvements over single-task or single-modality approaches. This unified framework proves effective in en-

(A) T2V (B) Controllable Gen

(E) Generalizability Trained exclusively on human data, generalizes to where specialist models fail

UnityVideo has a better understanding of physics: The video captures aserene sunset sceneby a calm body of water.

Better thematic consistency and controllability.

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

UnityVideoHunyuan 13B

OursVACERef

[Figure 152]

RGB

[Figure 153]

[Figure 154]

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

Dese Pose (Meta)

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

(C) Segmentation (D) 3D Point Representation

Fine-grained recognition capability.

[Figure 170]

[Figure 171]

UnityVideo

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

|[Figure 178]|
|---|

[Figure 179]

[Figure 180]

[Figure 181]

RGB

[Figure 182]

RGB

UnityVideo : Uniﬁed Depth & Flow

Geo 4D : Depth & Camera

[Figure 183]

SAMWISEUnityVideo

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

DWPose (IDEA)

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

UnityVideo

- Figure 5. Comparison with state-of-the-art methods across diverse tasks. UnityVideo exhibits superior physical reasoning, better adherence to control conditions, and a more detailed understanding of auxiliary modalities.

- Table 2. Ablation study comparing single-modality and multimodal training. Only: single modality; Ours: multiple modalities.

Table 3. Ablation study on single-task versus unified multi-task training. Only: single-task; Ours: unified multi-task.

Subject Background Temporal Motion Consistency Consistency Flickering Smoothness

Subject Background Imaging Overall Consistency Consistency Quality Consistency

Baseline 96.51 96.06 98.73 99.30 Only ControlGen 96.53 95.58 98.45 99.28 Only JointGen 98.01 97.24 99.10 99.44

Baseline 96.51 96.06 64.99 23.17 Only Flow 97.82 97.14 67.34 23.70 Only Depth 98.13 97.29 69.09 23.48

Ours-ControlGen 96.53 (+0.02) 96.08 (+0.02) 98.79 (+0.06) 99.38 (+0.08) Ours-JointGen 97.94 (+1.43) 97.18 (+0.63) 99.13 (+0.40) 99.48 (+0.18)

Ours-Flow 97.97 (+1.46) 97.19 (+1.13) 69.36 (+4.37) 23.74 (+0.57) Ours-Depth 98.01 (+1.50) 97.24 (+1.18) 69.18 (+4.19) 23.75 (+0.58)

hancing the model’s perception and reasoning capabilities for the physical world. Additional visual results can be found in Appendix D.

tual benefits between different training tasks within our unified framework, we train models separately on Joint Generation and Controllable Generation tasks, both guided by depth modality. Results are summarized in Tab. 3. We find that training only on the ControlGen task leads to performance degradation compared to the baseline. However, unified multi-task training recovers and even surpasses this performance, achieving improvements across all metrics. Similarly, compared to training only on Joint Generation, unified training shows only slight decreases in subject consistency and background consistency, while overall performance still outperforms the baseline, demonstrating the effectiveness of multi-task interaction.

##### 4.3. Abalation Study

Our ablation study addresses two core questions: (a) Does unified training across multiple modalities and tasks enable mutual benefits between modalities, and in what aspects are these benefits manifested? (b) Are our proposed architectural designs effective? What roles do the In-Context Learner and Modality Switcher play in the model? The following experiments address these questions.

Impact of Different Modalities. To quantitatively evaluate the impact of unified multimodal training on video generation, we compare two commonly used modalities, depth and optical flow, as shown in Tab. 2. The results show that joint training consistently improves performance across all metrics compared to the baseline. Furthermore, unified training with multiple modalities yields additional gains, particularly in image quality and overall consistency. This indicates that unifying diverse modalities not only provides complementary supervision during training but also enables mutual enhancement between modalities.

Impact of Architectural Design. We investigate the impact of two architectural strategies, In-Context Learner and Modality Switcher, on model performance. To ensure consistent evaluation, we perform text-to-video generation conditioned on depth guidance during inference. Results shown in Tab. 4 and Fig. 6 demonstrate that each strategy effectively improves performance through multimodal fusion. Furthermore, combining both strategies yields additional significant gains, confirming their complementary roles in facilitating unified multimodal learning.

Effect of Multi-Task Training. To further quantify the mu-

Segmentation

In-Context Learner better helps the model generalize to any two objects

Table 4. Ablation study on architectural designs.

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Modality

Switcher

Subject Background Temporal Motion Consistency Consistency Flickering Smoothness

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

In-Context

Baseline 96.51 96.06 98.73 99.30 w/ In-Context Learner 97.92 97.08 99.04 99.42 w/ Modality Switcher 97.94 97.18 99.13 99.48 Ours 98.31 97.54 99.35 99.54

Learner Origin

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

RGB Video Step 100Step 10kStep 1k

Table 5. World perception evaluation comparing UnityVideo with state-of-the-art models.

Depth UnityVideo is integrating all modal information to learn deep information

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

User Study Score (%) Automatic Score

Physical Semantic Overall Subject Motion

Quality Quality Preference Consistency Smoothness

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

Kling1.6 10.15 21.25 20.20 83.47 98.08 HunyuanVideo 24.15 26.10 20.35 97.53 98.35 Wan2.1 27.20 22.40 27.65 97.73 98.30

###### Ours 38.50 30.25 31.80 98.01 99.33

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

A woman is performing a series of push-ups on a red mat in a living room…

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

Joint RAFT

Only

Figure 7. The In-Context Learner generalizes segmentation to unseen objects, while unified training enhances depth and semantic understanding in RGB video.

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

Modalities

Unified

across both human evaluations and automatic metrics.

A close-up view of a person‘s arm with tattoos, working on a bicycle’s rear derailleur. The person is adjusting the cable tension on the derailleur, …

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

Joint Depth

Only

### 5. Limitation and Future Work

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

While UnityVideo significantly advances unified visual modeling, several directions remain for future work. The current VAE occasionally introduces reconstruction artifacts, which could be addressed through fine-tuning or improved autoencoder architectures. Additionally, scaling to larger backbones and incorporating more visual modalities may further enhance emergent world understanding capabilities. Despite these limitations, UnityVideo establishes a strong foundation for unified multi-modal video understanding and represents an important step toward comprehensive world models across diverse visual representations.

Modalities

Unified

- Figure 6. Unlike single-modality training, unified multimodal learning provides complementary supervision that strengthens both motion understanding and geometric perception.

##### 4.4. Model Analyze

As shown in Fig. 7, the proposed In-Context Learner effectively generalizes a fixed two-person segmentation task to unseen two-object scenarios. In contrast, using only the Modality Switcher fails to achieve such generalization. Moreover, during unified training, as the model gradually learns additional modalities (e.g., depth), we observe improved motion understanding and more accurate text responses in RGB videos, demonstrating the complementary roles of different modalities throughout training.

### 6. Conclusion

We present UnityVideo, a unified framework that models multiple visual modalities and tasks within a single diffusion transformer. By leveraging modal-adaptive learning, UnityVideo enables bidirectional learning between RGB video and auxiliary modalities (depth, optical flow, segmentation, skeleton, and DensePose), achieving mutual enhancement across both tasks. Our experiments demonstrate state-of-the-art performance across diverse benchmarks with strong zero-shot generalization to unseen modality combinations. To support this research, we contribute OpenUni, a large multimodal dataset with 1.3M synchronized samples, and UniBench, a high-quality evaluation benchmark with ground-truth annotations. UnityVideo paves the way toward unified multimodal modeling as a promising step toward next-generation world models.

User Study. We conduct a user study using a standard win-rate protocol to evaluate our model’s understanding of the physical world [50]. The questionnaire contains 12 randomly selected videos generated with WISA-80K prompts [46], presented in random order. Each sample is rated by at least three annotators on (i) physical quality, (ii) semantic quality (PF), and (iii) overall quality. For automatic evaluation, we adopt two VBench [14] metrics: dynamism and aesthetics. In total, we collect 70 completed responses, and the results are summarized in Tab. 5. The study shows that our method achieves the best performance

### References

- [1] Yunpeng Bai, Shaoheng Fang, Chaohui Yu, Fan Wang, and Qixing Huang. Geovideo: Introducing geometric regularization into video generation model. In NeurIPS, 2025. 2, 3
- [2] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025. 3
- [3] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024. 2
- [4] Yuanhao Cai, He Zhang, Xi Chen, Jinbo Xing, Yiwei Hu, Yuqian Zhou, Kai Zhang, Zhifei Zhang, Soo Ye Kim, Tianyu Wang, et al. Omnivcus: Feedforward subject-driven video customization with multimodal control conditions. arXiv preprint arXiv:2506.23361, 2025. 2
- [5] Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin. Videojam: Joint appearance-motion representations for enhanced motion generation in video models. arXiv preprint arXiv:2502.02492, 2025. 2, 3
- [6] Junyi Chen, Haoyi Zhu, Xianglong He, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Zhoujie Fu, Jiangmiao Pang, et al. Deepverse: 4d autoregressive video generation as a world model. arXiv preprint arXiv:2506.01103, 2025. 2
- [7] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Zhenyu Tang, Li Yuan, et al. Sharegpt4video: Improving video understanding and generation with better captions. NeurIPS, 37:19472– 19495, 2024. 3
- [8] Yunuo Chen, Junli Cao, Anil Kag, Vidit Goel, Sergei Korolev, Chenfanfu Jiang, Sergey Tulyakov, and Jian Ren. Towards physical understanding in video generation: A 3d point regularization approach. arXiv preprint arXiv:2502.03639, 2025. 3
- [9] Zhaoxi Chen, Tianqi Liu, Long Zhuo, Jiawei Ren, Zeng Tao, He Zhu, Fangzhou Hong, Liang Pan, and Ziwei Liu. 4dnex: Feed-forward 4d generative modeling made easy. arXiv preprint arXiv:2508.13154, 2025. 3
- [10] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 2
- [11] Claudia Cuttano, Gabriele Trivigno, Gabriele Rosi, Carlo Masone, and Giuseppe Averta. Samwise: Infusing wisdom in sam2 for text-driven video segmentation. In CVPR, pages 3395–3405, 2025. 5
- [12] Rıza Alp G¨uler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild. In CVPR, pages 7297–7306, 2018. 2, 6
- [13] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter:

- Generating consistent long depth sequences for open-world videos. In CVPR, pages 2005–2015, 2025. 2, 3, 5, 6
- [14] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, pages 21807–21818, 2024. 5, 8
- [15] Sihui Ji, Xi Chen, Xin Tao, Pengfei Wan, and Hengshuang Zhao. Physmaster: Mastering physical representation for video generation via reinforcement learning. arXiv preprint arXiv:2510.13809, 2025. 3
- [16] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. ICCV, 2025. 2, 3, 5
- [17] Zeren Jiang, Chuanxia Zheng, Iro Laina, Diane Larlus, and Andrea Vedaldi. Geo4d: Leveraging video generators for geometric 4d scene reconstruction. arXiv preprint arXiv:2504.07961, 2025. 2, 3, 5
- [18] Xuan Ju, Weicai Ye, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Qiang Xu. Fulldit: Multi-task video generative foundation model with full attention. ICCV, 2025. 2, 3, 5
- [19] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2, 5
- [20] Irena Koprinska and Sergio Carrato. Temporal video segmentation: A survey. Signal processing: Image communication, 16(5):477–500, 2001. 2
- [21] Akshay Krishnan, Xinchen Yan, Vincent Casser, and Abhijit Kundu. Orchid: Image latent diffusion for joint appearance and geometry generation. ICCV, 2025. 3
- [22] Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, et al. From generation to judgment: Opportunities and challenges of llm-as-ajudge. In EMNLP, pages 2757–2791, 2025. 2
- [23] Xuanyi Li, Daquan Zhou, Chenxu Zhang, Shaodong Wei, Qibin Hou, and Ming-Ming Cheng. Sora generates videos with stunning geometrical consistency. arXiv preprint arXiv:2402.17403, 2024. 2
- [24] Minghui Lin, Xiang Wang, Yishan Wang, Shu Wang, Fengqi Dai, Pengxiang Ding, Cunxiang Wang, Zhengrong Zuo, Nong Sang, Siteng Huang, et al. Exploring the evolution of physics cognition in video generation: A survey. arXiv preprint arXiv:2503.21765, 2025. 3
- [25] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. ICLR, 2023. 3, 5
- [26] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 2
- [27] Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Rongxin Jiang, Jiebo Luo, Hao Fei, et al.

- Javisdit: Joint audio-video diffusion transformer with hierarchical spatio-temporal prior synchronization. arXiv preprint arXiv:2503.23377, 2025. 3
- [28] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, pages 38–55. Springer, 2024. 2
- [29] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024. 2
- [30] Romero Morais, Vuong Le, Truyen Tran, Budhaditya Saha, Moussa Mansour, and Svetha Venkatesh. Learning regularity in skeleton trajectories for anomaly detection in videos. In CVPR, pages 11996–12004, 2019. 2
- [31] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195–4205, 2023. 4
- [32] Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, MingChang Yang, and Jiaya Jia. Controlnext: Powerful and efficient control for image and video generation. arXiv preprint arXiv:2408.06070, 2024. 2
- [33] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, et al. Open-sora 2.0: Training a commercial-level video generation model in 200 k. arXiv preprint arXiv:2503.09642, 2025. 5
- [34] Artsiom Sanakoyeu, Vasil Khalidov, Maureen S McCarthy, Andrea Vedaldi, and Natalia Neverova. Transferring dense pose to proximal animal classes. In CVPR, pages 5233– 5242, 2020. 2
- [35] Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Michael Moor, Zicheng Liu, and Emad Barsoum. Agent laboratory: Using llm agents as research assistants. arXiv preprint arXiv:2501.04227,

2025. 2

- [36] Yukun Su, Guosheng Lin, Jinhui Zhu, and Qingyao Wu. Human interaction learning on 3d skeleton point clouds for video violence recognition. In ECCV, pages 74–90. Springer,

2020. 2

- [37] Yang-Tian Sun, Xin Yu, Zehuan Huang, Yi-Hua Huang, Yuan-Chen Guo, Ziyi Yang, Yan-Pei Cao, and Xiaojuan Qi. Unigeo: Taming video diffusion for unified consistent geometry estimation. arXiv preprint arXiv:2505.24521, 2025. 3
- [38] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. In ICCV, pages 14940– 14950, 2025. 3
- [39] Zhiyu Tan, Hao Yang, Luozheng Qin, Jia Gong, Mengping Yang, and Hao Li. Omni-video: Democratizing unified video understanding and generation. arXiv preprint arXiv:2507.06119, 2025. 3
- [40] Aether Team, Haoyi Zhu, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Chunhua Shen, Jiangmiao Pang, et al. Aether: Geometric-aware unified world modeling. arXiv preprint arXiv:2503.18945,

2025. 2, 3, 5

- [41] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025. 2
- [42] Yi-Hsuan Tsai, Ming-Hsuan Yang, and Michael J Black. Video segmentation via object flow. In CVPR, pages 3899– 3908, 2016. 2
- [43] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 5
- [44] Duomin Wang, Wei Zuo, Aojie Li, Ling-Hao Chen, Xinyao Liao, Deyu Zhou, Zixin Yin, Xili Dai, Daxin Jiang, and Gang Yu. Universe-1: Unified audio-video generation via stitching of experts. arXiv preprint arXiv:2509.06155, 2025. 3
- [45] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, pages 5294– 5306, 2025. 2
- [46] Jing Wang, Ao Ma, Ke Cao, Jun Zheng, Zhanjie Zhang, Jiasong Feng, Shanyuan Liu, Yuhang Ma, Bo Cheng, Dawei Leng, et al. Wisa: World simulator assistant for physics-aware text-to-video generation. arXiv preprint arXiv:2503.08153, 2025. 8
- [47] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In CVPR, pages 8428–8437, 2025. 5
- [48] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In SIGGRAPH, pages 1–11, 2024. 2
- [49] Thadd¨aus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 2, 3
- [50] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 8
- [51] Shengqiong Wu, Weicai Ye, Jiahao Wang, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Shuicheng Yan, Hao Fei, et al. Any2caption: Interpreting any condition to caption for controllable video generation. arXiv preprint arXiv:2503.24379, 2025. 2
- [52] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. ICLR, 2025. 3
- [53] Jingqiao Xiu, Fangzhou Hong, Yicong Li, Mengze Li, Wentao Wang, Sirui Han, Liang Pan, and Ziwei Liu. Egotwin: Dreaming body and view in first person. arXiv preprint arXiv:2508.13013, 2025. 2, 3

- [54] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 3
- [55] Yichao Yan, Jingwei Xu, Bingbing Ni, Wendong Zhang, and Xiaokang Yang. Skeleton-aided articulated motion generation. In ACMMM, pages 199–207, 2017. 2
- [56] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 2
- [57] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10371–10381, 2024. 5
- [58] Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. Effective whole-body pose estimation with two-stages distillation. In ICCV, pages 4210–4220, 2023. 6
- [59] Guy Yariv, Yuval Kirstain, Amit Zohar, Shelly Sheynin, Yaniv Taigman, Yossi Adi, Sagie Benaim, and Adam Polyak. Through-the-mask: Mask-based motion trajectories for image-to-video generation. In CVPR, pages 18198– 18208, 2025. 2
- [60] Shenghai Yuan, Xianyi He, Yufan Deng, Yang Ye, Jinfa Huang, Bin Lin, Jiebo Luo, and Li Yuan. Opens2v-nexus: A detailed benchmark and million-scale dataset for subjectto-video generation. arXiv preprint arXiv:2505.20292, 2025. 5
- [61] Zhixiong Zhang, Shuangrui Ding, Xiaoyi Dong, Songxin He, Jianfan Lin, Junsong Tang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Sec: Advancing complex video object segmentation via progressive concept construction. arXiv preprint arXiv:2507.15852, 2025. 5, 6
- [62] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Wang, Lijuan Wang, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. NeurIPS, 36:19769–19782, 2023. 2, 6

[Figure 238]

## UnityVideo: Unified Multi-Modal Multi-Task Learning for Enhancing World-Aware Video Generation

### Appendix

The appendix contains the following sections:

- • More Analysis of Model Design
- • More Experiments and Analysis
- • Details of OpenUni and UniBench
- • More Visuals and Applications

### A. More Analysis of Model Design

##### A.1. Modal Interaction Analysis

To further investigate the cross-modal interactions within our unified framework, we visualize the evolution of selfattention maps throughout the training process. We partition the attention map into four distinct regions based on modality interactions: self-modality regions comprising (RGB, RGB) and (Flow, Flow), and cross-modality regions consisting of (RGB, Flow) and (Flow, RGB), where Flow represents various auxiliary modality features. As illustrated in Figure 1, our analysis reveals three key findings. First, as joint training progresses, the interaction between RGB and auxiliary modalities becomes progressively more pronounced (A), indicating deepening cross-modal feature exchange. Second, the visualization results demonstrate that the model learns increasingly rich geometric representations with improved text-following capabilities (B), validating the effectiveness of our unified training paradigm in enhancing both visual understanding and conditional generation quality. This empirical evidence confirms that our unified framework not only enables technical integration of multiple modalities but also facilitates meaningful featurelevel interactions that contribute to improved world modeling capabilities.

##### A.2. Modality-Specific Output Layers

While our modality switcher and in-context learner effectively differentiate between modalities, we observed occasional modality confusion as the number of modalities scales. For instance, when instructed to generate segmentation masks, the model infrequently produces skeleton outputs instead. This confusion stems from all modalities sharing a common output layer, which can conflate distinct modality-specific features at the final projection stage.

To address this limitation, we introduce modalityspecific output layers (adaptive layer) while maintaining a unified input layer (share layer) for cross-modal information sharing. Each modality receives its own dedicated output projection layer, initialized independently, while the

Table 1. Comparison of different layer strategies.

Subject Background Temporal Motion Averaged

Consistency Consistency Flickering Smoothness

Baseline 96.51 96.06 98.73 99.30 97.650 Share Layer 98.31 97.54 99.35 99.54 98.685 Adaptive Layer 98.26 97.49 99.44 99.61 98.700

Table 2. Comparison with standalone T2V. Joint generation achieves better performance, with unified modality showing further improvements.

Subject Background Imaging Overall Averaged

Consistency Consistency Quality Consistency

Baseline 96.51 96.06 64.99 23.17 70.1825 T2V 96.51 97.23 66.52 23.44 70.9250

Depth Modality

JointGen 98.13 97.29 69.09 23.48 71.998 (Depth) (+1.62) (+0.06) (+2.57) (+0.04) (+1.073)

JointGen 98.01 97.24 69.18 23.75 72.045 (Unified) (+1.50) (+0.01) (+2.66) (+0.31) (+1.120)

Optical Flow Modality

- JointGen 97.82 97.14 67.34 23.70 71.500 (Optical Flow) (+1.31) (-0.09) (+0.82) (+0.26) (+0.575)

- JointGen 97.97 97.19 69.36 23.74 72.065 (Unified) (+1.46) (-0.04) (+2.84) (+0.30) (+1.140)

Densepose Modality

- JointGen 98.08 97.38 67.05 23.49 71.500 (Densepose) (+1.57) (+0.15) (+0.53) (+0.05) (+0.575)

- JointGen 98.03 97.30 70.20 23.53 72.265 (Unified) (+1.52) (+0.07) (+3.68) (+0.09) (+1.340)

input processing remains shared to preserve inter-modal knowledge transfer. This architectural refinement ensures clear modality boundaries during generation without sacrificing the benefits of unified representation learning.

As shown in Table 1, this lightweight design effectively eliminates modality confusion during scaled training while maintaining comparable performance across metrics. The modality-specific output layers provide improved flexibility and achieve balanced performance across diverse evaluation criteria, validating this architectural choice for scalable multi-modal generation.

### B. More Experiments and Analysis

##### B.1. Compare with T2V

While results in main paper demonstrates promising gains from joint generation over the baseline, we further investigate whether joint generation provides advantages over standard supervised fine-tuning (SFT) for text-to-video generation. We conduct extensive ablation studies across different modalities, training models with identical data and steps to ensure fair comparison of their text-to-video capabilities.

Step 200 Step 2k Step 20k Step 200 Step 2k Step 20k

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

(C) Joint Depth Generation(A) Attention Map

A man is working in a cluttered workshop, using a portable table saw to cut wooden planks. The man's movements are deliberate and precise. He adjusts the wooden planks on the table saw, ensuring they are properly aligned before starting the saw. The saw's blade moves back and forth as it cuts through the planks.

The man in the purple t-shirt is gesturing and talking, while the man in the jacket listens and occasionally looks back. The man in the jacket is wearing a black and orange jacket with a distinctive triangle logo on the back. The scene is set in a room with a metal shelving unit filled with various motorcycle helmets.

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

- Figure 1. Evolution of attention patterns in UnityVideo. Analysis of attention maps shows that interactions between RGB and auxiliary modalities strengthen progressively across layers. Meanwhile, the model’s text-following behavior and spatial reasoning capabilities also improve, reflecting more coherent cross-modal integration.

As shown in Table 2, all modality configurations with joint generation achieve significant improvements over both the baseline and T2V-only training. Each auxiliary modality contributes distinct supervisory signals that enhance the model’s visual understanding, confirming the complementary nature of different modalities. Moreover, unified multimodal training outperforms single-modality joint training by achieving better balance across evaluation dimensions, with substantial gains in overall performance (Averaged column). These results validate that diverse modality supervision collectively strengthens video generation through mutual reinforcement rather than simply additive improvements.

##### B.2. Scalability with Increasing Modalities

To demonstrate UnityVideo’s ability to continuously improve with expanded modality training, we evaluate performance scaling on both joint generation and controllable generation tasks. As shown in Table 3, UnityVideo achieves consistent performance gains across all metrics as the number of modalities increases. Specifically, we compare models trained with three modalities (depth, optical flow, and DensePose) against those trained with five modalities (additionally incorporating skeleton and segmentation).

The results reveal monotonic improvements across all evaluation criteria, confirming that our framework effectively leverages additional modality supervision without suffering from negative interference. This strong scalability suggests that UnityVideo’s architecture can accommodate further expansion in both model parameters and modality diversity, potentially enabling emergent world perception capabilities as the framework scales. The consistent gains validate our unified training paradigm as a promis-

Table 3. Analysis of the benefits brought by extended modal training for joint generation and control generation.

Subject Background Temporal Motion Consistency Consistency Flickering Smoothness Baseline 96.51 96.06 98.73 99.30

Joint Generation

- Depth 96.53 95.58 98.45 99.28 Three Modalities 98.01 97.24 99.10 99.44 Five Modalities 98.31 97.54 99.35 99.54

Control Generation

- Depth 97.78 96.79 98.80 99.30 Three Modalities 97.83 96.86 98.87 99.33 Five Modalities 97.87 97.32 99.57 99.39

ing foundation for developing increasingly comprehensive video world models through continued modality integration.

##### B.3. The influence of different modalities

As shown in main paper, incorporating additional modalities yields further improvements for the JointGeneration task compared with training on a single modality. To examine whether this benefit also extends to ControlGeneration, we conduct the ablation study summarized in Table 4. Here, Only denotes models trained on ControlGeneration using a single modality, while Ours refers to models trained jointly with three modalities. All training data and iteration budgets are kept strictly identical to ensure a fair comparison.

The results show that unified multimodal training consistently outperforms single-modality training on the ControlGeneration task. These findings demonstrate that UnityVideo effectively strengthens positive cross-modal interactions across tasks, enabling each modality to benefit from the shared training paradigm.”

(A) Compare with HunyuanVideo The video captures a breathtaking view of a mountainous landscape shrouded in mist.

- (B) Compare with Wan2.1

The video captures a vibrant nighttime scene, the overall atmosphere is festive and celebratory, enhanced by the bright lights and the spectacular firework.

- (C)Compare with Aether The video captures a serene sunset scene by a calm body of water. The sphere‘s reflective surface creates an illusion, making it appear as though the sunset is encapsulated within its transparent form.

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

HunyuanVideoUnityVideoHunyuanVideoHunyuanVideoUnityVideoUnityVideo

AetherUnityVideoWan2.1UnityVideoWan2.1UnityVideoWan2.1UnityVideo

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

The video captures a serene natural scene featuring a picturesque waterfall cascading down a rocky cliff into a flowing river below. The waterfall is composed of multiple streams of water, creating a dynamic and soothing visual effect.

Water flowing from a faucet into a kitchen sink. The stream should show realistic behavior including splash patterns when hitting the sink bottom and proper water flow dynamics around the drain.

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

The video captures a mesmerizing display of the Northern Lights (Aurora Borealis) over a serene landscape. The sky is filled with countless stars, creating a breathtaking backdrop for the vibrant green auroras that sweep across it.

A pendulum swinging back and forth, gradually losing amplitude due to air resistance and friction at the pivot point. The motion should follow natural pendulum physics with decreasing swing angles.

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

A clear glass tumbler on a white table. Fresh white milk being poured from a ceramic pitcher into the glass, showing distinct light refraction and bending effects through the transparent glass walls. The milk stream should appear distorted when viewed through the glass. Camera angle from the side, studio lighting.

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

HunyuanVideoUnityVideo

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

- Figure 2. Comparison of physical understanding. UnityVideo demonstrates stronger physical reasoning and improved text alignment compared with current state-of-the-art video generation models.

Table 4. The gain of joint modal training compared with single modal on ControlGeneration tasks.

Subject Background Temporal Motion Averaged

Consistency Consistency Flickering Smoothness

Baseline 96.51 96.06 98.73 99.30 97.65 Depth Modality

ControlGen 97.78 96.79 98.80 99.30 98.1675 (Depth) (+1.27) (+0.73) (+0.07) (+0.00) (+0.5175) Unified 97.83 96.86 98.87 99.33 98.2225 (Depth) (+1.32) (+0.80) (+0.14) (+0.03) (+0.5725)

Optical Flow Modality

ControlGen 97.40 96.59 98.67 99.23 97.9725 (Optical Flow) (+0.89) (+0.53) (-0.06) (-0.07) (+0.3225) ControlGen 97.47 96.72 98.83 99.32 98.0850

- (Unified) (+0.96) (+0.66) (+0.10) (+0.02) (+0.4350) Densepose Modality

ControlGen 97.01 96.47 98.58 99.10 97.790 (Densepose) (+0.50) (+0.41) (-0.15) (+0.20) (+0.5050) ControlGen 97.58 96.79 98.90 99.35 98.1550

- (Unified) (+1.07) (+0.73) (+0.17) (+0.05) (+0.5050)

##### B.4. World perception comparison

To further assess our model’s world understanding capabilities, we conduct comprehensive evaluations using physicsfocused prompts that test fundamental physical principles. As shown in Figure 2, we evaluate models on scenarios in-

volving refraction, collision dynamics, and other physical phenomena that require accurate world modeling.

Our results demonstrate that UnityVideo exhibits superior understanding of physical laws compared to baseline methods. The model accurately captures light refraction through transparent media, realistic collision responses between objects, and physically plausible motion trajectories. These improvements stem from the complementary supervision provided by auxiliary modalities—depth enhances spatial reasoning, optical flow captures motion dynamics, and segmentation clarifies object boundaries—collectively enabling more accurate physical world modeling. This enhanced physical reasoning capability further validates the effectiveness of our unified multimodal training paradigm in developing world-aware video generation models.

### C. Details of OpenUni and UniBench

##### C.1. OpenUni

The OpenUni dataset leverages diverse data sources and comprehensive modality extraction to create a large-scale multimodal training corpus. We employ multiple pretrained

UE Data : Used for evaluating video estimation tasks

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

Depth / DenseposeRGB (Real Data)Optical FlowRGB (UE Data)Optical FlowDepth

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

Real data: Used for evaluating the Text2Video and Control Generation tasks

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

- Figure 3. UniBench consists of two complementary components: (i) high-fidelity Unreal Engine depth data for evaluating depth estimation, and (ii) diverse real-world videos with rich multimodal annotations for assessing video generation quality.

models to extract modality-specific features and implement rigorous filtering pipelines to ensure data quality and usability.

Our data curation process follows strict quality criteria. We first filter source videos based on temporal, aesthetic, and resolution constraints: minimum duration of 5 seconds, aesthetic score exceeding 80/100, and spatial resolution above 512 pixels. Videos containing embedded text or subtitles are removed using OCR-based detection to prevent contamination of visual modalities. For each retained video, we extract corresponding modality annotations using specialized models—depth from Depth Anything V2, optical flow from RAFT, segmentation from SAM, skeleton from DWPose, and DensePose from Meta’s implementation. Automated quality metrics further filter low-quality modality extractions, ensuring reliable ground-truth annotations across all modalities.

Through this systematic pipeline, we obtain approximately 1.3M high-quality multimodal video pairs, each containing synchronized annotations across five modalities. This comprehensive dataset enables effective unified training while maintaining consistency and quality across diverse visual representations.

##### C.2. UniBench

To address the absence of standardized evaluation benchmarks for unified multimodal video tasks, we construct UniBench with two distinct evaluation categories tailored to different task requirements. For video estimation tasks requiring ground-truth annotations, we generate synthetic data using Unreal Engine to obtain pixel-accurate depth maps and optical flow. As shown in Figure 3, for controllable generation and text-to-video tasks requiring diverse modality conditions, we curate high-quality samples from our test split.

Specifically, we create 200 synthetic video sequences with precise ground-truth depth and optical flow using Unreal Engine’s rendering pipeline. These sequences feature significant camera and object motion to comprehensively evaluate depth estimation capabilities under challenging conditions. For generation tasks, we select 200 high-quality samples from the test subset, each containing complete annotations across all five modalities. This dual-track evaluation strategy enables rigorous assessment of both reconstruction accuracy and generation quality within our unified framework.

### D. More Visuals and Applications

- Figure 4 and 5 showcases UnityVideo’s extensive generalization capabilities across three core tasks: controllable generation, video estimation, and joint generation. The model accepts arbitrary modality inputs for precise controllable generation while supporting flexible modality estimation for diverse subjects and scenarios.

Our framework demonstrates remarkable zero-shot generalization beyond its training distribution. While trained primarily on single-person data, UnityVideo successfully generalizes to multi-person scenarios for all modality estimations. Similarly, skeleton estimation capabilities trained on human subjects transfer effectively to animal motion capture without additional fine-tuning. The model also exhibits robust cross-domain transfer, accurately estimating depth and segmentation for out-of-distribution objects and scenes. These diverse examples collectively demonstrate that UnityVideo’s unified training paradigm not only achieves technical integration across modalities but also develops genuine world understanding that enables flexible generalization to novel contexts and subjects.

Video Estimation

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

AnimalHuman

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

[Figure 372]

Figure 4. Representative outputs of UnityVideo on Video Estimation. The model consistently produces coherent RGB videos and aligned modalities—including densepose, optical flow, skeleton, and depth—demonstrating reliable cross-modal generation and estimation across diverse scenarios from human activities to animal motion.

###### Text2Video

[Figure 373]

[Figure 374]

DenseposeOptical FlowSkeletonDepthDensepose

###### Control Generation

[Figure 375]

[Figure 376]

optical FlowDepthDepthDenseposeoptical Flow

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

Figure 5. Representative outputs of UnityVideo on Text2Video and Control Generation. The model consistently produces coherent RGB videos and aligned modalities—including segmentation, densepose, optical flow, skeleton, and depth—demonstrating reliable cross-modal generation and estimation across various indoor and outdoor scenes with multiple subjects.

