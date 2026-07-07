## DualCamCtrl: Dual-Branch Diffusion Model for Geometry-Aware Camera-Controlled Video Generation

Hongfei Zhang1* Kanghao Chen1,5* Zixin Zhang1,5 Harold H. Chen1,5 Yuanhuiyi Lyu1 Yuqi Zhang3 Shuai Yang1 Kun Zhou4 Yingcong Chen1,2†

1HKUST (GZ) 2HKUST 3Fudan University 4Shenzhen University 5 Knowin

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

# arXiv:2511.23127v2[cs.CV]1Dec2025

WanCameraCtrlDualCamCtrlGTInput

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

10th Frame 20th Frame 15th Frame 30th Frame 20th Frame 40th Frame

Figure 1. Comparison with state-of-the-art methods [24, 78] on camera-controlled video generation. Under an identical target camera trajectory and a single input image, our approach achieves the closest adherence to the camera motion and yields the best perceptual quality.

#### Abstract

forms RGB–depth fusion in a semantics-guided and mutually reinforced manner. These designs collectively enable DualCamCtrl to better disentangle appearance and geometry modeling, generating videos that more faithfully adhere to the specified camera trajectories. Additionally, we analyze and reveal the distinct influence of depth and camera poses across denoising stages and further demonstrate that early and late stages play complementary roles in forming global structure and refining local details. Extensive experiments demonstrate that DualCamCtrl achieves more consistent camera-controlled video generation with over 40% reduction on camera motion errors compared with prior methods. Our project page:

This paper presents DualCamCtrl, a novel end-to-end diffusion model for camera-controlled video generation. Recent works have advanced this field by representing camera poses as ray-based conditions, yet they often lack sufficient scene understanding and geometric awareness. DualCamCtrl specifically targets this limitation by introducing a dual-branch framework that mutually generates camera-consistent RGB and depth sequences. To harmonize these two modalities, we further propose the SemantIc Guided Mutual Alignment (SIGMA) mechanism, which per-

*These authors contributed equally. †Corresponding author, yingcongchen@ust.hk

https://soyouthinkyoucantell.github.io/dualcamctrl-page/

#### 1. Introduction

Advances in video diffusion models [1, 6, 18, 19, 21, 22, 27, 28, 35, 42, 47, 81, 83, 85, 96, 97] have significantly improved the quality of video generation from textual descriptions. These developments have extended beyond basic text-to-video (T2V) or image-to-video (I2V) generation, with growing efforts toward integrating control mechanisms into the video generation process, such as camera control [1, 24, 25, 29, 37, 43, 86, 99–101], motion transfer [10, 32, 54, 57, 62, 84, 99], object manipulation [15, 38, 61, 66, 75, 88], and multi-modality prediction [13, 31, 58, 68, 69, 82, 92–94]. Among them, camera control has emerged as a particularly promising direction, enabling natural camera movements and viewpoint transitions that bridge the gap between generative modeling and real-world cinematography, supporting applications from virtual directors to interactive 3D scene generation [6, 11, 20, 33, 83, 89, 101].

Building on this progress, camera-conditioned video diffusion models [1, 24, 25, 86, 101] have been explored to enable models to respond to camera trajectories during video synthesis. However, most approaches rely solely on camera poses or further, their ray-conditioned representation (a.k.a. pl¨ucker embedding) as control signals. Although ray-conditioned models [1, 24, 33, 86, 101] encode camera motion by projecting poses into per-pixel ray directions, they inherently lack explicit scene understanding because the model must implicitly infer 3D structure to produce videos coherent with the given camera motion. This limitation results in suboptimal camera motion consistency, as these models are constrained by the entanglement of appearance and structure modeling and fail to fully capture the underlying geometry of the scene. Given these challenges, it is natural to incorporate geometric cues that encode scene structure. With advances in modern depth estimation [9, 31, 93], we turn to depth as an additional geometric modality that complements camera-pose conditioning.

However, to the best of our knowledge, few prior works have explored incorporating depth within the context of end-to-end camera-conditioned video generation models to facilitate precise camera control. This raises the fundamental question of this paper: What is the inherent relationship between depth and camera control, and how can depth be effectively incorporated to further benefit cameraconditioned video generation? To fill this gap, we explore an effective way to incorporate depth as an additional geometric cue into camera-conditioned video diffusion models. Yet, achieving coherent interaction between the depth and RGB modalities remains non-trivial. Given a single frame and its corresponding depth map provided

by existing depth estimators [9], our empirical results indicate that naive injection of depth information into existing camera-conditioned models does not consistently improve generation quality. For example, model conditioned on single-frame depth lacks sufficient temporal context to provide stable geometric cues, often leading to unnatural scene structures (Fig. 16). Conversely, jointly modeling RGB and depth within a single branch introduces noticeable interference between the modalities, resulting in degraded synthesis quality (Fig. 17). These observations underscore the need for an effective integration strategy that leverages depth to improve the consistency of camera motion.

To this end, we propose a dual-branch video diffusion model that effectively integrates depth information for camera control. Conditioned on shared camera poses, the model generates camera-consistent RGB and depth sequences, where depth provides guidance throughout the video, and the two modalities interact in a way that minimizes mutual interference while maintaining complementary contributions. We further introduce the SematIc Guided Mutual Alignment (SIGMA) mechanism to harmonize semantic and geometric representation. Motivated by the principles of prioritizing semantic fidelity and enabling mutual feedback, SIGMA ensures that the RGB and depth branches evolve together, leading to more consistent and stable video generation. Additionally, a two-stage training strategy is adopted to stabilize the training process, which consists of a decoupled stage followed by a fusion stage. Moreover, we conduct analysis and reveal that depth and camera poses affect latent representations in distinct ways across stages and layers. We also find that early and late denoising stages play complementary roles in the generation process where the former establishes global structure and the latter refines local details, offering insight for the community.

To summarize, our main contributions are threefold:

- • We propose DualCamCtrl, a dual-branch video diffusion framework that effectively incorporates geometric cues via the SIGMA mechanism and two-stage training.
- • We investigate how depth and camera pose influence the RGB denoising process and the distinct roles of early and late denoising stages. This analysis sheds light on their influences and provides insights for future research.
- • Extensive experiments validate our state-of-the-art capabilities, achieving more than 40% reductions in rotation errors compared with previous methods.

#### 2. Related Work

Foundation model of video generation. Recent foundational diffusion-based video generation models, such as CogVideoX [95], Stable Video Diffusion (SVD) [5], Wan [78], Lumiere [3], VideoCrafter [7, 8], ModelScopeT2V [80], Latte [52], and Open-Sora [44, 60], have established powerful baselines for text-to-video and condi-

tional video synthesis. These models employ large-scale latent diffusion or diffusion transformer architectures to achieve high temporal consistency, realistic motion, and fine-grained scene dynamics. Building upon these foundational models, subsequent studies have explored their adaptation to various downstream video generation tasks, including camera control, [1, 24, 25, 29, 37, 43, 86, 99–101], motion transfer [10, 32, 54, 57, 62, 84, 99], object manipulation [15, 38, 61, 66, 75, 88], and multi-modality prediction [13, 31, 58, 68, 69, 82, 92–94]. In this work, we focus on the problem of camera-controllable video generation, aiming to enable more consistent camera control video generation built upon diffusion-based foundation models [78].

Camera control for video models. Early efforts toward camera-controllable video generation were pioneered by methods such as MotionCtrl [86], which explicitly encode camera poses to guide video diffusion models. Following this line, CameraCtrl [24] and VD3D [2] further introduce ray-based conditioning (i.e., Pl¨ucker embeddings [71]) to inject camera parameters into pretrained diffusion backbones. More recently, [25] and [78] extend this idea by incorporating camera pose information through pre-DiT conditioning. AC3D [1] provides a deeper analysis of Video DiT behavior and mitigates training data limitations by constructing dynamic-scene datasets with rigid camera motion. Despite these advances, existing approaches still struggle to preserve the intended camera trajectory.

Novel view synthesis (NVS) with video diffusion. Generating novel views from a set of posed images has seen significant progress [39, 56, 72]. With the recent success of image and video generation models, ReconFusion [90] and CAT3D [17] began leveraging the prior knowledge learned by these models to facilitate sparse-view NVS. However, because they require per-scene optimization, these methods remain inherently slow. Recent works such as ReconX [50], ViewCrafter [98], and GEN3C [64] incorporate depth and point clouds to facilitate NVS and enable camera control from a few image inputs, which is somewhat similar to our setting. Nonetheless, they rely on multi-stage procedures that project and unproject point clouds into explicit camera views, in contrast to our end-to-end camera-conditional video generation. We pioneer the incorporation of depth into end-to-end camera-controlled video generation models to facilitate precise camera control.

#### 3. Method

In this section, we introduce the DualCamCtrl framework, which enables more geometry-aware camera-conditioned video diffusion by integrating geometric cues (i.e. depth) in a dual-branch manner. We first provide necessary preliminaries in Sec. 3.1. In Sec. 3.2, we describe how depth is injected via our dual-branch architecture. We introduce the SematIc Guided Mutual Alignment and 3D fusion

strategy for interaction between RGB and depth branches in Sec. 3.3. Sec. 3.4 details our two-stage training pipeline.

##### 3.1. Preliminary

The video diffusion process typically involves a gradual denoising procedure, where a noisy latent representation of the video zt is progressively transformed into a high-quality video sequence. In camera-controlled video generation, the process is mainly conditioned on two inputs: the context condition c (i.e. images or textual descriptions) and the camera pose (R,t). Following recent works [1, 24, 78], we encode camera geometry through ray-based conditioning, where each pixel is associated with a Pl¨ucker ray representation derived from (R,t) (See Algorithm 1).

Formally, given a video V ∈ RT×3×H×W, where T is the number of frames, and H,W are the height and width of each frame. This video is first encoded into a latent space using a pretrained encoder ε, producing z0 = ε(V), where z0 ∈ RT

′×C′×h×w. Under our setting [78], the latent z has

temporal length T′ = T−4 1 +1, channel dimension C′ = 16 and spatial dimensions h = H/8, w = W/8. The diffusion

process then gradually denoises zt at each timestep t, conditioned on inputs c, which can be an image cimage or text ctext, as well as camera pose parameters (R,t). The training objective is to predict the noise ϵ added to the latent (Eq. (1)):

L = Eϵ∼N(0,I) ϵ − θ zt,t,c,R,t 2 , (1)

where θ denotes the denoising network. During the inference stage, the denoising process is performed iteratively over a sequence of timesteps t. After denoising in the latent space, the output zˆ is decoded by the decoder D to obtain the predicted video Vˆ : Vˆ = D(ˆz).

##### 3.2. Dual-Branch Camera-Controlled Video Generation

Depth has long been recognized as a crucial cue for scene understanding as it explicitly encodes scene geometry and spatial layout [12, 23, 31, 45, 46, 53, 65, 69, 73, 87]. However, despite the importance of geometric priors, the integration of depth cues in camera-conditioned video generation remains unexplored. In this section, we investigate how depth information can be leveraged to enhance cameraconditioned video synthesis. Specifically, under the I2V setting, the input consists of a single reference frame, and monocular depth estimator [9] allows us to predict a plausible depth map for that frame. However, relying solely on this single-frame depth is insufficient for guiding longrange video generation, as it provides only a static snapshot of the scene’s geometry without maintaining temporal consistency. Moreover, naively coupling the generation of RGB and depth modalities tends to cause interference between them (Figs. 16 and 17). This motivates us to design a framework that can propagate and effectively utilize

|[Figure 14]|
|---|

|[Figure 15]|
|---|

Camera Trajectory Noise

SIGMA Fusion

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

VAE Encoder

DiT Block

DiT Block

z z

⨁ C

ℒ

| | |
|---|---|
|[Figure 21]| |
| | |

| | |
|---|---|
|[Figure 22]| |
| | |

Embed Embed

SIGMA Fusion

SIGMA Fusion

Depth

[Figure 23]

3D Conv Block

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

VAE Encoder

DiT Block

DiT Block

⨁ C

ℒ

Zero-Initial Conv Layer

RGB

×N Times ×M Times

- Figure 2. Overall architecture of DualCamCtrl. DualCamCtrl adopts a dual-branch framework that simultaneously generates RGB and depth video latents from an input image and its corresponding depth map. The two latents are then element-wise added to the encoded Pl¨ucker embedding and concatenated with noise (Sec. 3.2). Subsequently, the two modalities interact through our proposed SIGMA mechanism and fusion block (Sec. 3.3). During training, both predictions are supervised by their respective loss functions (Sec. 3.4).

##### 3.3. Injecting Depth Cues for Mutual Alignment

depth information throughout the video sequence, thereby assisting RGB synthesis while minimizing interference. To this end, we make a trial to adopt a dual-branch architecture, where RGB and depth features are processed in parallel, each conditioned on the same camera poses to generate modality-specific representations (Fig. 2). This design allows the model to produce coherent information for both modalities under consistent camera motion.

Sematic Guided Mutual Alignment. Based on our dualbranch framework, one branch synthesizes the RGB sequence, while the other generates the corresponding depth sequence. Despite being conditioned on the same input (i.e. image or text), the two branches evolve independently, resulting in inconsistency between their outputs, which necessitates an effective alignment strategy (Fig. 3).

Formally, given an input image IRGB ∈ R3×H×W, we first employ a monocular depth estimator [9] to obtain its depth map ID ∈ R1×H×W. Following previous works [26, 31, 68, 93], the estimated depth is replicated along the channel dimension to match the channel number of the RGB input. Both the RGB image and the replicated depth are then encoded into the latent space using a pretrained encoder ε, producing their respective latent representations zI

However, simple fusion strategies remain insufficient. We observe that one-way alignment transferring features from one modality to another without feedback, fails to preserve semantic consistency, while geometry-guided alignment overemphasizes geometric cues and disrupts appearance coherence. To overcome this, we introduce SematIc Guided Mutual Alignment (SIGMA). SIGMA adopts a semantic-guided bidirectional design: early layers leverage RGB features to anchor semantic structure, while later layers incorporate depth feedback to refine geometry.

D. The two latent representations are both zero-padded in the frame channel to match the length of noise. Then they are element-wisely added with the same encoded pl¨ucker embedding which represents the camera pose (R,t). Afterwards, they are concatenated with the same Gaussian noise ϵ and processed by a denoising network θ (i.e. Diffusion Transformer [59]), which consists of two parallel branches corresponding to the RGB and depth modalities. The model θ jointly predicts the denoised latent sequences, yielding an RGB video latent zˆV

RGB and zI

Notably, this design is motivated by two key insights. ❶ Priority matters: Appearance should first dominate the initial stages to ensure semantic fidelity, while depth is introduced later as a complementary corrective signal that refines geometric consistency. ❷ Mutual feedback matters: Enabling both branches to inform each other avoids the imbalance of one-way alignment, leading to more consistent representation. Together, these principles enable SIGMA to more effectively harmonize semantic and geometric representations, ultimately yielding more visually coherent video generation (Tab. 8).

RGB and a depth video latent zˆV

D.

During training, the predicted latents are supervised by the ground-truth video latents zV

D, which are obtained by encoding the ground-truth RGB-D video VRGB and VD. During inference, a pretrained decoder D is employed to reconstruct the corresponding RGB from the predicted latents. The training strategy is further described in Sec. 3.4. Details of the T2V setting are provided in the supplementary material due to space limitations (Fig. 11).

RGB and zV

3D Fusion Strategy. Our experiments reveal that achieving temporal and spatial consistency between RGB and depth sequences cannot rely solely on alignment strategies (Fig. 4). While SIGMA effectively coordinates the two branches, conventional shallow projection layers (e.g., linear feature injection as in [20, 83, 100]) still ignore the

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

|[Figure 32]<br><br>[Figure 33]|
|---|

| |
|---|

| |
|---|

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

The video shows a well-

| |
|---|

| |
|---|

lit kitchen

with modern design ….

[Figure 38]

[Figure 39]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Text to Video

[Figure 40]

[Figure 41]

|[Figure 42]<br><br>[Figure 43]<br><br>|
|---|

| |
|---|

| |
|---|

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

| |
|---|

| |
|---|

[Figure 48]

[Figure 49]

| |
|---|

| |
|---|

Image to Video

One-Way Condition Geometry-Guided SIGMA(Ours)

###### SIGMA(Ours)

(a)

(b)

(c)

- Figure 3. (a) Illustration of modality misalignment. Independent RGB and depth latent evolution leads misalignment across frames. This motivates the design of our SIGMA strategy to establish coherent cross-modal alignment. (b) Comparison with one-way alignment. One-way alignment transfers information unidirectionally, leading to misalignment on local semantics. (c) Comparison with geometryguided alignment Under the geometry-guided setting, geometry cues evolved too quickly and become inconsistent with RGB motion.

(a) (b)

[Figure 50]

[Figure 51]

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

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

| |
|---|

| |
|---|

flicker discontinuity

1DFuse3DFuse

- Figure 4. Comparison of fusion strategies. (1.a) Previous shallow linear fusion operates in a pixel-wise manner, ignoring temporal and spatial context and causing inconsistency over time. (1.b) We introduce 3D Fusion strategy to extend the fusion formulation from 1D to 3D by incorporating 3D operations. (2) Traditional linear-layer fusion often leads to visual artifacts, while our methods produce smoother, more coherent results.

namics of both modalities. We first observe that the RGB branch, initialized from pretrained visual backbones, naturally possesses a strong prior for appearance generation, while the depth branch starts from scratch without generative capability of depth modality. Directly training both branches jointly leads to severe convergence issues, where the depth stream fails to learn meaningful geometry and, in turn, destabilizes the overall optimization (Fig. 8 and Tab. 3). Thus, to perform effective training, we adopt a two-stage strategy: first, we independently learn appearance and geometry in the decoupling stage, and then we enable cross-branch interaction through a fusion block in the fusion stage. The detailed process is described below.

Initialization and Decoupled Stage. To incorporate knowledge from pretrained diffusion models, we first initialize both the RGB and depth branches with the same model weights [78]. The primary goal of this stage is to enable the model to learn a robust depth synthesis module that can be used as a complementary signal for the RGB branch. However, no existing dataset provides simultaneously scene-level realistic video motion, corresponding camera parameters, and ground-truth depth [4, 14, 36, 40, 48, 55, 91]. To address this, we employ state-of-the-art monocular depth estimation methods to predict depth maps for all frames and use them as supervision for the depth branch [9]. To avoid interference between the two modalities, we do not perform any cross-branch fusion at this stage. This separation ensures that each branch can independently capture its respective cues, appearance for RGB and geometry for depth, without destabilizing the training process (Fig. 8 and Tab. 3). Interestingly, we observed that although the depth branch cannot generate the target sequence initially, it still interprets depth as a hazy image and uses this as a condition for generation (Fig. 12).

temporal ordering and spatial layout of video, treating all latent pixels uniformly across frames and often producing temporally inconsistent outputs. To effectively capture spatiotemporal cues, we introduce 3D convolutions into the fusion block, enabling the model to fuse semantic and geometry information over both spatial and temporal dimension (Fig. 2). To balance computational cost and parameter efficiency, we further adopt an efficient bottleneck design inspired by [30, 67, 74]. A frame-wise gating mechanism is involved to adaptively modulate the fusion strength according to temporal dynamics. We provide a more detailed description in the appendix (Fig. 9).

##### 3.4. Two-stage Training Pipeline

Recall that our training objective is twofold: enable each modality to develop generative competence and foster effective cross-modal interaction. To achieve this, we adopt a carefully staged schedule that balances the learning dy-

Fusion Stage. In the second stage, we enable fusion between the RGB and depth branches to exploit their complementary strengths: the RGB branch provides rich ap-

- .1
- .2
- .3
- .4

400

pearance cues, while the depth branch conveys geometric structure. The training objective remains a combination of depth and RGB losses (Eq. (4)). We zero-initialize the fusion block, allowing their influence to emerge gradually.

CKA Score

FVD

390

Loss Function. Let γ ∈ {0,1} indicate whether crossbranch fusion is enabled, where γ = 0 corresponds to the decoupled stage and γ = 1 corresponds to the fusion stage. We define the 3D-aware cross features hRGBt →D (from RGB to Depth) and hDt →RGB (from Depth to RGB), with the corresponding losses for each branch as follows (Eqs. (2) and (3)):

380

∆Step

Block Index 5 10 15 30

0 5 15 30 50

(a)

(b)

400 / 0 (FVD/∆Step)

380 / 30 (FVD/∆Step)

- .1

Block Index

- .2
- .3
- .4

Early Stage Mid Stage Late Stage vs Depth vs Plücker Embedding

CKA Score

LRGB = E vtRGB − θRGB(ztRGB,t,c,

(2)

5 10 15 30 5 10 15 30

R,t;γ hDt →RGB) 2 ,

(c)

Figure 5. CKA vs. pose; early-stage effects. (a) RGB branch shows strongest camera motion alignment in early–mid layers. (b) Extra early steps yield the largest quality gains. (c) Stronger earlystage weights lower CKA variance and improve FVD.

LD = E vtD − θD ztD,t,c,R,t;γ hRGBt →D 2 . (3) The overall loss is then defined as (Eq. (4)):

LOverall = LRGB + λLD. (4)

##### 4.2. Which stage is important for camera control?

#### 4. Analysis

Building on the above analysis, we ask which denoising stage contributes most to camera control video generation. We compare different timestep schedules (i.e. more steps on early, mid or late) and quantify their effects on video generation. Timesteps are sampled linearly within each predefined stage, as variations in the within-stage distribution showed negligible impact in our pilot experiments. We use 15 steps uniformly distributed across the three stages, while ∆step denotes the additional steps allocated.

We first conduct an analysis to gain a deeper understanding of how camera pose and depth representation influence the denoising process. Specifically, we employ Centered Kernel Alignment analysis [41] to quantify the relationship between RGB latent representations and various signals (i.e. depth supervision and Pl¨ucker-based ray representation) throughout the denoising process. Based on this, we further investigate the relative importance of each stage in camera-conditioned video generation by analyzing common inference-time scheduling strategies and their effects on temporal coherence and camera controllability. For simplicity, we define the early stage as t > 0.9T, the mid stage as 0.75T < t ≤ 0.9T, and the late stage as t ≤ 0.75T, where T denotes the total number of denoising timesteps.

- I. Role of Early Denoising Steps in Establishing Camera Geometry. We observe that the early denoising stage (t > 0.9T) plays a particularly critical role in defining the global camera geometry. During this period, the Pl¨ucker embeddings exhibit high variance within the RGB latent space, suggesting that the model is actively aligning its internal representation with the camera pose. Increasing the number of denoising steps in different diffusion stages improves generation quality overall (Fig. 5b), but the most substantial gains arise from the early stage. This suggests that additional early denoising allows the model to better consolidate the global camera structure before appearance refinement begins. In contrast, insufficient early denoising leads to unstable alignment and degraded FVD performance, indicating that the camera–scene relationship is primarily established at the start of the diffusion process (See Fig. 5c).
- II. Late-stage Refinement Also Contributes to Poseaware Detail Consistency. While the early denoising stage primarily establishes the global camera geometry, we empirically find that the late stage also plays a crucial role in maintaining pose-consistent local refinement. In particular, once early denoising has sufficiently stabilized the coarse spatial structure aligned with the camera trajectory (50 steps

##### 4.1. Influence Dynamics Across Denoising Stages.

CKA curves (Fig. 5a) indicate that the similarity between RGB latents and the Pl¨ucker ray embeddings peaks in the early-to-mid part of the schedule (t ∈ (0.75T,T]), showing that camera-pose conditioning exerts its strongest influence when global structure is being formed. The similarity then drops sharply in the late range (t ≤ 0.75T). Consistent with prior observations [1, 78, 95], the mid layers carry the most accurate information about camera pose; notably, this also holds for depth information. We further observe pronounced variability in the early-stage similarity between Pl¨ucker embeddings and RGB features, suggesting that seemingly small changes to the schedule or conditioning strength during this window can produce large effects on the final trajectory and structure. Motivated by these findings, the next subsection analyzes how the denoising stage impacts camera-controlled video generation.

Table 1. Quantitative comparisons on I2V setting. ↑ / ↓ denotes higher/lower is better. We highlight the best and second best .

RealEstate10K DL3DV

Method

FVD ↓ FID ↓ CLIPSIM ↑ FC ↑ MS ↑ RE ↓ TE ↓ FVD ↓ FID ↓ CLIPSIM ↑ FC ↑ MS ↑ RE ↓ TE ↓

|MotionCtrl [86] CameraCtrl [24] Seva [101] Wan [78]<br><br>|137.4 71.70 118.7 69.90 104.2 76.69 109.2 77.80<br><br>|0.2401 0.9621 0.9733 0.2358 0.9623 0.9860 0.3059 0.9570 0.9815 0.2859 0.9567 0.9863<br><br>|2.80 1.04 2.38 1.03 2.62 0.32 2.08 0.32<br><br>|158.5 98.51 144.4 85.37 206.0 130.9 127.9 70.77<br><br>|0.2009 0.9386 0.9360 0.2015 0.9569 0.9745 0.2044 0.9443 0.9471 0.2022 0.9512 0.9731<br><br>|1.10 1.73 1.04 1.71 1.06 1.67 1.01 1.64<br><br>|
|---|---|---|---|---|---|---|
|DualCamCtrl|80.38 49.85<br><br>|0.2998 0.9677 0.9886<br><br>|1.25 0.23<br><br>|92.2 53.89<br><br>|0.2047 0.9637 0.9763<br><br>|0.88 1.39<br><br>|

.55

We sample frames with a stride ranging from 0 to 8 to capture a wide range of camera motions. Our video diffusion models are developed by fine-tuning pre-trained video diffusion models [78]. We follow the original training scheme of the baseline model with 1000 denoising steps, using flow matching [49] as the training target.

.30

TranslationError

RotationError

.45

.29

Early Stage Mid Stage Late Stage

.35

.28

1.25

0.27

∆Step 5 10 25 5 10 25

∆Step

Evaluation Metrics. We assess our video generation performance using several widely adopted metrics, including Fr´echet Video Distance (FVD) [77], Fr´echet Inception Distance (FID) [76], and CLIP similarity (CLIPSim) [63]. We also employ large-scale, open-source evaluation frameworks to evaluate Frame Consistency and Motion Strength (denoted as FC and MS) [34]. In addition, we calculate the rotation and translation error (denoted as RE and TE) using [79] with fine-tracking. We further conduct a human study to obtain qualitative feedback on various subjective aspects (Tab. 5), offering stronger evidence of the effectiveness of our method. More training and evaluation details are provided in the appendix.

[Figure 70]

[Figure 71]

[Figure 72]

Frame 10

| |
|---|

[Figure 73]

InputRGB

[Figure 74]

[Figure 75]

[Figure 76]

Frame 20

|∆Step = 25|
|---|

∆Step = 0

GT

- Figure 6. Importance of late denoising stages. (1) Rotation/Translation error vs. number of denoising steps under different strengths: once early steps pass a threshold, adding more early steps yields diminishing returns in geometric consistency, whereas late steps are key for finer appearance fidelity. (2) Late denoising sharpens object boundaries and surface details, indirectly improving pose-aware consistency. Please zoom in for better view.

##### 5.1. Comparisons

under our setting), the benefit of additional camera-control guidance in the early steps diminishes. By contrast, allocating more steps to the late stage yields larger gains where subsequent denoising helps refine object boundaries and high-frequency textures (Fig. 6). Due to space limitation, we provide more analysis in the appendix (Sec. 8).

Quantitative results. As shown in Tables 1 and 2, our method achieves consistent improvements across both I2V and T2V settings, demonstrating superior performance on objective metrics. Notably, due to the significant reduction in rotation and translation errors (i.e. 40%), our method also achieves a substantial decrease in FVD, which is particularly noticeable under the I2V setting.

#### 5. Experiments

Qualitative results. As illustrated in Fig. 7, our approach produces videos that exhibit higher visual fidelity and temporal coherence. Moreover, our method obtained the highest overall preference rate in the human study (Tab. 5) confirming that the integration of depth-aware geometry yields results that are not only quantitatively superior but also subjectively more visually appealing to human observers. Please refer to the appendix for more results.

Baselines. We evaluate our method against several stateof-the-art camera-controlled video generation models. In the I2V setting, we compare our approach with MOTIONCTRL [86] and CAMERACTRL [25], as well as two recent strong baselines, WAN-V2.1 [78] and SEVA [101]. In the T2V setting, we compare with CAMERACTRL, MOTIONCTRL, and the most recent AC3D [1]. All baseline models are evaluated under the same split and evaluation metrics for a consistent and unbiased comparison. We do not compare with [51] due to different settings and application scenarios.

##### 5.2. Ablation Study

In this section, we conduct ablation study to evaluate the contributions of key components in our DualCamCtrl framework on the REALESTATE10K dataset. We put more ablation cases in the appendix (Sec. 10.2).

Implementation Details. Following [1, 25, 86, 101], we train our model on the REALESTATE10K [102]. It contains over 60 M video clips with per-frame camera parameters.

MotionCtrl CameraCtrl Seva Wan Ours GT

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

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

Input Image

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

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

Camera Pose

- Figure 7. Comparison between our method and other state-of-the-art approaches. Given the same camera pose and input image as generation conditions, our method achieves the best alignment between camera motion and scene dynamics, producing the most visually accurate video. The ’+’ signs marked in the figure serve as anchors to indicate specific reference points for better visual comparison.

Table 2. Quantitative comparisons on T2V setting across REALESTATE10K and DL3DV.

Method

RealEstate10K DL3DV

FVD ↓ CLIPSIM ↑ FC ↑ MS ↑ RE ↓ TE ↓ FVD ↓ CLIPSIM ↑ FC ↑ MS ↑ RE ↓ TE ↓

|MotionCtrl [86] CameraCtrl [24] AC3D [1]|506.9 426.8 415.6<br><br>|0.2744 0.9471 0.9734 0.2734 0.9526 0.9750 0.3044 0.9496 0.9811<br><br>|2.90 1.06 2.68 0.98 2.67 0.38<br><br>|746.7 675.5 452.8<br><br>|0.2033 0.9448 0.9462 0.2027 0.9518 0.9610 0.2078 0.9529 0.9869<br><br>|1.17 3.23 1.01 4.00 1.06 2.11<br><br>|
|---|---|---|---|---|---|---|
|DualCamCtrl|408.1|0.3154 0.9540 0.9918<br><br>|1.23 0.25<br><br>|427.4|0.2090 0.9544 0.9807<br><br>|0.83 1.53<br><br>|

Table 3. Ablation on two-stage training. Variant FVD ↓ FID ↓ MS ↑ FC ↑

Single-Stage 96.7 58.3 0.9859 0.9537 Two-Stage (Ours) 80.4 49.9 0.9886 0.9677

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

One Stage Two Stage GT

[Figure 94]

One Stage Two Stage

- Figure 8. Effect of Two-Stage Training: Top: The single-stage model fails to converge properly. Bottom: Consequently, the geometry alignment is suboptimal, highlighting the importance of disentangling appearance and geometry learning in the two-stage training approach.

Table 4. Ablation on depth conditioning. Variant FVD ↓ FID ↓ MS ↑ FC ↑

w/o Depth 96.3 66.1 0.9872 0.9610 w/ Depth (Ours) 80.4 49.9 0.9886 0.9677

firms that depth provides strong structural cues for camera controlled scene generation.

Effect of Two-Stage Training. We further evaluate the effectiveness of our two-stage training scheme. As seen in Tab. 3, direct joint training (only fusion stage) leads to unstable convergence and suboptimal temporal coherence (Fig. 8). The two-stage variant achieves superior results across all metrics, suggesting that decoupling appearance and geometry learning before fusion significantly stabilizes training and enhances cross-modal consistency.

#### 6. Conclusion

We presented DualCamCtrl, a dual-branch video diffusion model that integrates depth for more accurate cameracontrolled video generation. By introducing the SemantIc Guided Mutual Alignment (SIGMA) mechanism and a two-stage training process, DualCamCtrl effectively synchronizes RGB and depth sequences, improving geometry awareness. Our experiments show over 40% reduction in rotation errors compared to previous methods, offering new insights into camera-controlled video generation and hopefully benefiting other video generation tasks.

Effect of Incorporating Depth. To assess the benefit of depth guidance, we compare our model with and without depth guidance. As shown in Tab. 4, removing depth leads to a notable degradation in FVD and MS, indicating weaker temporal stability and geometry understanding. This con-

#### References

- [1] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22875–22889, 2025. 2, 3, 6, 7, 8
- [2] Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky, Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, et al. Vd3d: Taming large video diffusion transformers for 3d camera control. arXiv preprint arXiv:2407.12781, 2024. 3
- [3] Omer Bar-Tal, Hila Chefer, Omer Tov, Tomer Michaeli, Tali Dekel, et al. Lumiere: A space-time diffusion model for video generation. arXiv preprint arXiv:2401.12945,

2024. 2

- [4] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5855–5864,

2021. 5

- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Robin Rombach, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [6] Shengqu Cai, Duygu Ceylan, Matheus Gadelha, ChunHao Paul Huang, Tuanfeng Yang Wang, and Gordon Wetzstein. Generative rendering: Controllable 4d-guided video generation with 2d diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7611–7620, 2024. 2
- [7] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 2
- [8] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2024. 2

- [9] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. arXiv:2501.12375, 2025. 2, 3, 4, 5
- [10] Weifeng Chen, Yatai Ji, Jie Wu, Hefeng Wu, Pan Xie, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Controla-video: Controllable text-to-video diffusion models with motion prior and reward feedback learning. arXiv preprint arXiv:2305.13840, 2023. 2, 3
- [11] Yiran Chen, Anyi Rao, Xuekun Jiang, Shishi Xiao, Ruiqing Ma, Zeyu Wang, Hui Xiong, and Bo Dai. Cinepregen:

- Camera controllable video previsualization via enginepowered diffusion. arXiv preprint arXiv:2408.17424, 2024. 2
- [12] Jaesung Choe, Sunghoon Im, Francois Rameau, Minjun Kang, and In So Kweon. Volumefusion: Deep depth fusion for 3d scene reconstruction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16086–16095, 2021. 3
- [13] Yue-Jiang Dong, Wang Zhao, Jiale Xu, Ying Shan, and Song-Hai Zhang. Depthsync: Diffusion guidance-based depth synchronization for scale-and geometry-consistent video depth estimation. arXiv preprint arXiv:2507.01603,

2025. 2, 3

- [14] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3d scanned household items. In 2022 International Conference on Robotics and Automation (ICRA), pages 2553–2560. IEEE, 2022. 5
- [15] Ruoyu Feng, Wenming Weng, Yanhui Wang, Yuhui Yuan, Jianmin Bao, Chong Luo, Zhibo Chen, and Baining Guo. Ccedit: Creative and controllable video editing via diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6712– 6722, 2024. 2, 3
- [16] Wanquan Feng, Tianhao Qi, Jiawei Liu, Mingzhen Sun, Pengqi Tu, Tianxiang Ma, Fei Dai, Songtao Zhao, Siyu Zhou, and Qian He. I2vcontrol: Disentangled and unified video motion synthesis control. arXiv preprint arXiv:2411.17765, 2024. 5
- [17] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024. 3, 5
- [18] Daniel Geng, Charles Herrmann, Junhwa Hur, Forrester Cole, Serena Zhang, Tobias Pfaff, Tatiana Lopez-Guevara, Yusuf Aytar, Michael Rubinstein, Chen Sun, et al. Motion prompting: Controlling video generation with motion trajectories. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1–12, 2025. 2
- [19] Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou, Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei Liu, et al. Diffusion as shader: 3d-aware video diffusion for versatile video generation control. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–12, 2025. 2
- [20] Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou, Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei Liu, Wenping Wang, and Yuan Liu. Diffusion as shader: 3d-aware video diffusion for versatile video generation control. arXiv preprint arXiv:2501.03847, 2025. 2, 4, 5
- [21] Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Pengfei Wan, Di Zhang, Yufan Liu, Weiming Hu, Zhengjun Zha, et al. I2v-adapter: A general image-to-video adapter for diffusion models. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 2

- [22] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In European Conference on Computer Vision, pages 330–348. Springer, 2024. 2
- [23] Kamal Gupta, Susmija Jabbireddy, Ketul Shah, Abhinav Shrivastava, and Matthias Zwicker. Improved modeling of 3d shapes with multi-view depth maps. In International Conference on 3D Vision (3DV), 2020. 3
- [24] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 1, 2, 3, 7, 8, 5
- [25] Hao He, Ceyuan Yang, Shanchuan Lin, Yinghao Xu, Meng Wei, Liangke Gui, Qi Zhao, Gordon Wetzstein, Lu Jiang, and Hongsheng Li. Cameractrl ii: Dynamic scene exploration via camera-controlled video diffusion models. arXiv preprint arXiv:2503.10592, 2025. 2, 3, 7
- [26] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Liu, Bingbing Liu, and YingCong Chen. Lotus: Diffusion-based visual foundation model for high-quality dense prediction. arXiv preprint arXiv:2409.18124, 2024. 4
- [27] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2
- [28] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022. 2
- [29] Chen Hou and Zhibo Chen. Training-free camera control for video generation. arXiv preprint arXiv:2406.10126,

2024. 2, 3

- [30] Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017. 5, 1
- [31] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2005– 2015, 2025. 2, 3, 4
- [32] Chi-Pin Huang, Yen-Siang Wu, Hung-Kai Chung, Kai-Po Chang, Fu-En Yang, and Yu-Chiang Frank Wang. Videomage: Multi-subject and motion customization of text-tovideo diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17603– 17612, 2025. 2, 3
- [33] Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li, Rynson WH Lau, Wangmeng Zuo, et al. Voyager: Longrange and world-consistent video diffusion for explorable 3d scene generation. arXiv preprint arXiv:2506.04225,

2025. 2

- [34] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 7, 3
- [35] Yash Jain, Anshul Nasery, Vibhav Vineet, and Harkirat Behl. Peekaboo: Interactive video generation via maskeddiffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8079– 8088, 2024. 2
- [36] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik Aanæs. Large scale multi-view stereopsis evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 406–413, 2014. 5
- [37] Wonjoon Jin, Qi Dai, Chong Luo, Seung-Hwan Baek, and Sunghyun Cho. Flovd: Optical flow meets video diffusion model for enhanced camera-controlled video synthesis. arXiv preprint arXiv:2502.08244, 2025. 2, 3
- [38] Kumara Kahatapitiya, Adil Karjauv, Davide Abati, Fatih Porikli, Yuki M Asano, and Amirhossein Habibian. Objectcentric diffusion for efficient video editing. In European Conference on Computer Vision, pages 91–108. Springer,

2024. 2, 3

- [39] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 3

- [40] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36(4):1–13, 2017. 5
- [41] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International conference on machine learning, pages 3519–3529. PMlR, 2019. 6
- [42] Mathis Koroglu, Hugo Caselles-Dupr´e, Guillaume Jeanneret, and Matthieu Cord. Onlyflow: Optical flow based motion conditioning for video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6226–6236, 2025. 2
- [43] Zhengfei Kuang, Shengqu Cai, Hao He, Yinghao Xu, Hongsheng Li, Leonidas J Guibas, and Gordon Wetzstein. Collaborative video diffusion: Consistent multi-video generation with camera control. Advances in Neural Information Processing Systems, 37:16240–16271, 2024. 2, 3
- [44] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, April

2024. 2

- [45] Tristan Laidlow, Jan Czarnowski, and Stefan Leutenegger. Deepfusion: Real-time dense 3d reconstruction for monocular slam using single-view depth and gradient predictions. CoRR, abs/2207.12244, 2022. 3
- [46] Ariel Lapid, Idan Achituve, Lior Bracha, and Ethan Fetaya. Gd-vdm: Generated depth for better diffusion-based video generation. arXiv preprint arXiv:2306.11173, 2023. 3
- [47] Jingyun Liang, Yuchen Fan, Kai Zhang, Radu Timofte, Luc Van Gool, and Rakesh Ranjan. Movideo: Motion-aware

- video generation with diffusion model. In European Conference on Computer Vision, pages 56–74. Springer, 2024. 2
- [48] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024. 5, 3
- [49] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 7
- [50] Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Reconx: Reconstruct any scene from sparse views with video diffusion model. arXiv preprint arXiv:2408.16767, 2024. 3, 5
- [51] Yawen Luo, Jianhong Bai, Xiaoyu Shi, Menghan Xia, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Tianfan Xue. Camclonemaster: Enabling reference-based camera control for video generation, 2025. 7
- [52] Xin Ma, Wen Zhang, Pan Zhou, et al. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 2
- [53] Davide Menini, Suryansh Kumar, Martin R. Oswald, Erik Sandstr¨om, Cristian Sminchisescu, and Luc Van Gool. A real-time online learning framework for joint 3d reconstruction and semantic segmentation of indoor scenes. CoRR, abs/2108.05246, 2021. 3
- [54] Tuna Han Salih Meral, Hidir Yesiltepe, Connor Dunlop, and Pinar Yanardag. Motionflow: Attention-driven motion transfer in video diffusion models. arXiv preprint arXiv:2412.05275, 2024. 2, 3
- [55] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (ToG), 38(4):1–14, 2019. 5
- [56] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 3
- [57] Saman Motamed, Wouter Van Gansbeke, and Luc Van Gool. Investigating the effectiveness of cross-attention to unlock zero-shot editing of text-to-video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7406–7415,

2024. 2, 3

- [58] Enrico Pallotta, Sina Mokhtarzadeh Azar, Shuai Li, Olga Zatsarynna, and Juergen Gall. Syncvp: Joint diffusion for synchronous multi-modal video prediction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13787–13797, 2025. 2, 3
- [59] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195– 4205, 2023. 4

- [60] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, et al. Opensora 2.0: Training a commercial-level video generation model in $200k. arXiv preprint arXiv:2503.09642, 2025. 2
- [61] Elia Peruzzo, Vidit Goel, Dejia Xu, Xingqian Xu, Yifan Jiang, Zhangyang Wang, Humphrey Shi, and Nicu Sebe. Vase: Object-centric appearance and shape manipulation of real videos. arXiv preprint arXiv:2401.02473, 2024. 2, 3
- [62] Alexander Pondaven, Aliaksandr Siarohin, Sergey Tulyakov, Philip Torr, and Fabio Pizzati. Video motion transfer with diffusion transformers. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22911–22921, 2025. 2, 3
- [63] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 7
- [64] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas M¨uller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6121–6132, 2025. 3, 5
- [65] Antoni Rosinol, John J. Leonard, and Luca Carlone. Probabilistic volumetric fusion for dense monocular slam. CoRR, abs/2210.01276, 2022. 3
- [66] Nirat Saini, Navaneeth Bodla, Ashish Shrivastava, Avinash Ravichandran, Xiao Zhang, Abhinav Shrivastava, and Bharat Singh. Invi: Object insertion in videos using off-theshelf diffusion models. arXiv preprint arXiv:2407.10958,

2024. 2, 3

- [67] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4510–4520, 2018. 5, 1
- [68] Jiahao Shao, Yuanbo Yang, Hongyu Zhou, Youmin Zhang, Yujun Shen, Vitor Guizilini, Yue Wang, Matteo Poggi, and Yiyi Liao. Learning temporally consistent video depth from video diffusion priors, 2024. 2, 3, 4
- [69] Jiahao Shao, Yuanbo Yang, Hongyu Zhou, Youmin Zhang, Yujun Shen, Vitor Guizilini, Yue Wang, Matteo Poggi, and Yiyi Liao. Learning temporally consistent video depth from video diffusion priors. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22841– 22852, 2025. 2, 3
- [70] Wenzhe Shi, Jose Caballero, Ferenc Husz´ar, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1874–1883, 2016. 3
- [71] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation render-

- ing. Advances in Neural Information Processing Systems, 34:19313–19325, 2021. 3
- [72] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, et al. Nerfstudio: A modular framework for neural radiance field development. In ACM SIGGRAPH 2023 conference proceedings, pages 1–12, 2023. 3
- [73] Keisuke Tateno, Federico Tombari, Iro Laina, and Nassir Navab. Cnn-slam: Real-time dense monocular slam with learned depth prediction. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 624–632, 2017. 3
- [74] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer look at spatiotemporal convolutions for action recognition. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 6450–6459, 2018. 5, 1
- [75] Shuyuan Tu, Qi Dai, Zhi-Qi Cheng, Han Hu, Xintong Han, Zuxuan Wu, and Yu-Gang Jiang. Motioneditor: Editing video motion via content-aware diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7882–7891, 2024. 2, 3
- [76] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7
- [77] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. arXiv preprint,

2019. 7

- [78] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 2, 3, 5, 6, 7
- [79] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. Vggsfm: Visual geometry grounded deep structure from motion. 2023. 7
- [80] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 2
- [81] Luozhou Wang, Yijun Li, Zhifei Chen, Jui-Hsien Wang, Zhifei Zhang, He Zhang, Zhe Lin, and Ying-Cong Chen.

- Transpixeler: Advancing text-to-video generation with transparency. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18229–18239, 2025. 2, 4
- [82] Qian Wang, Abdelrahman Eldesokey, Mohit Mendiratta, Fangneng Zhan, Adam Kortylewski, Christian Theobalt, and Peter Wonka. Vidseg: Training-free video semantic segmentation based on diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22985–22994, 2025. 2, 3
- [83] Qinghe Wang, Yawen Luo, Xiaoyu Shi, Xu Jia, Huchuan Lu, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Cinemaster: A 3d-aware and controllable framework for cinematic text-to-video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–10, 2025. 2, 4
- [84] Xingrui Wang, Xin Li, Yaosi Hu, Hanxin Zhu, Chen Hou, Cuiling Lan, and Zhibo Chen. Tiv-diffusion: Towards object-centric movement for text-driven image to video generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 7988–7996, 2025. 2, 3
- [85] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. International Journal of Computer Vision, 133(5):3059–3078, 2025. 2
- [86] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 2, 3, 7, 8
- [87] Silvan Weder, Johannes L. Sch¨onberger, Marc Pollefeys, and Martin R. Oswald. Routedfusion: Learning real-time depth map fusion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4886–4896,

2020. 3

- [88] Bichen Wu, Ching-Yao Chuang, Xiaoyan Wang, Yichen Jia, Kapil Krishnakumar, Tong Xiao, Feng Liang, Licheng Yu, and Peter Vajda. Fairy: Fast parallelized instructionguided video-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8261–8270, 2024. 2, 3
- [89] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, and Huan Ling. Difix3d+: Improving 3d reconstructions with single-step diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26024–26035, 2025. 2
- [90] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor Verbin, Jonathan T Barron, Ben Poole, et al. Reconfusion: 3d reconstruction with diffusion priors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21551–21561, 2024. 3, 5
- [91] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian,

- et al. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 803–814, 2023. 5
- [92] Dianbing Xi, Jiepeng Wang, Yuanzhi Liang, Xi Qiu, Yuchi Huo, Rui Wang, Chi Zhang, and Xuelong Li. Omnivdiff: Omni controllable video diffusion for generation and understanding. arXiv preprint arXiv:2504.10825, 2025. 2, 3
- [93] Honghui Yang, Di Huang, Wei Yin, Chunhua Shen, Haifeng Liu, Xiaofei He, Binbin Lin, Wanli Ouyang, and Tong He. Depth any video with scalable synthetic data. arXiv preprint arXiv:2410.10815, 2024. 2, 4
- [94] Lehan Yang, Lu Qi, Xiangtai Li, Sheng Li, Varun Jampani, and Ming-Hsuan Yang. Unified dense prediction of video diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28963–28973, 2025. 2, 3
- [95] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2025. 2, 6
- [96] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-tovideo diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2
- [97] Yu-Ying Yeh, Jia-Bin Huang, Changil Kim, Lei Xiao, Thu Nguyen-Phuoc, Numair Khan, Cheng Zhang, Manmohan Chandraker, Carl S Marshall, Zhao Dong, et al. Texturedreamer: Image-guided texture synthesis through geometry-aware diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4304–4314, 2024. 2
- [98] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 3, 5
- [99] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jia-Wei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. Motiondirector: Motion customization of text-to-video diffusion models. In European Conference on Computer Vision, pages 273–290. Springer, 2024. 2, 3
- [100] Guangcong Zheng, Teng Li, Rui Jiang, Yehao Lu, Tao Wu, and Xi Li. Cami2v: Camera-controlled image-to-video diffusion model. arXiv preprint arXiv:2410.15957, 2024. 4
- [101] Jensen Jinghao Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. Stable virtual camera: Generative view synthesis with diffusion models. arXiv preprint arXiv:2503.14489, 2025. 2, 3, 7
- [102] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. ACM Trans. Graph. (Proc. SIGGRAPH), 37, 2018. 7, 3

## DualCamCtrl: Dual-Branch Diffusion Model for Geometry-Aware Camera-Controlled Video Generation

### Supplementary Material

#### 7. Further Details of the DualCamCtrl Framework

##### 7.1. Detail of 3D Fusion Block

To balance accuracy and efficiency, our 3D-aware fusion block adopts a bottleneck design. Embeddings are first projected into a lower-dimensional space, aggregated by stacked 3D convolutions with a depthwise, pointwise decomposition [30, 67, 74], and then mapped back through a zero-initialized output layer. This initialization makes the block behave like an identity at the start of training, while a frame-level gate modulates how much geometric information is injected per frame (Fig. 9).

Concretely, given a sequence embedding z∈RB×L×C

emb

emb×T′×h×w. A bottleneck convolution reduces the channel dimension to Cb=C

with L=T′×h×w, we reshape it to RB×C

4 . The features are then processed by a stack of 3DConv blocks, each consisting of a depthwise 3×3×3 convolution followed by a pointwise 1×1×1 convolution. A zero-initialized 1×1×1 convolution restores the channels to C. Finally, a frame-level gating mechanism produces gt ∈ [0,1] and modulates the output:

emb

Yt = gt ⊙ F(Xt),

###### zD

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

zRGB

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Elided Layers

Both stage Stage 2 Addition

Fusion Module RGB Branch Depth Branch

- Figure 10. Training scheme of DualCamCtrl. The overall training follows a two-stage training scheme: (I) In the decouple stage (denoted by only the black arrows), the RGB and depth branches are trained separately (with no interaction) to learn modalityspecific representations. (II) In the fusion stage (both red and black arrows). In this process, SIGMA mechanism is applied through a 3D-aware fusion module, enabling effective cross-modal integration and improved synthesis consistency.

strategy lets each branch first stabilize within its own domain and then benefit from complementary cues, leading to stronger cross-modal correspondence.

- 7.3. Detail of T2V Setting

| | |
|---|---|
|[Figure 107]| |
| | |

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

|DiT Block<br><br>DiT<br><br>Block|SIGMA Fusion<br><br>| |
|---|---|---|
|× M Times| | |

× N Times

DiT Block

DiT

Block

C

| | |
|---|---|
|Text Embed.| |

⨁

C

|Text Embed.| |
|---|---|
| | |

⨁

SIGMA Fusion

ℒ𝐷

ℒ𝑅𝐺𝐵

[Figure 114]

The video shows a

well-lit kitchen with...

T5 Encoder

|Text Embed.|
|---|

| |
|---|

| |
|---|

Figure 11. Architecture of DualCamCtrl under the T2V setting. The model operates without any visual conditioning. The textual input is first encoded by a T5 encoder and fed into two diffusion transformers to condition both RGB and depth generation.

Under the T2V setting, the model still operate in a dualbranch jointly generation manner but without any input image or initial frame as conditioning information (Fig. 11). Nevertheless, our strategy allows the RGB branch to effectively provide semantic guidance to the depth branch during generation. We observe that the overall performance consistently improves (Tabs. 1 and 2), indicating the robustness of our cross-modal guidance mechanism.

- 8. Findings and Analysis

zD LayerBE zD

concat

BE Layer

zRGB

zRGB

||3D-DW-Conv|
|---|
<br><br>|3D-PW-Conv|
|---|
<br><br>|Frame Gate|
|---|
<br><br>|SiLU|
|---|
|
|---|

|3DConvBlock| |
|---|---|
| | |

3DConvBlock

3DConvBlock

Zero-Init Conv Layer

c

(a) (b)

Figure 9. Architecture of 3D-Aware Fusion Block. (a) Design of the fusion block. The fusion block consists of a bottleneck embedding, a series of 3D convolutional layers, a zero-initialized convolutional output layer. (b) Detail of the 3D convolutional blocks. Here, BE stands for bottleneck embedding. DW-Conv stands for depthwise 3D convolution. PW-Conv stands for pointwise 3D convolution. Please zoom in for a better view.

##### 7.2. Detail of Training Pipeline

As illustrated in Fig. 10, DualCamCtrl is trained in two stages. In the decoupled stage, the RGB and depth branches are optimized independently, to acquire modality-specific representations for appearance and geometry. In fusion stage, we enable cross-modal learning by coupling the two branches through a 3D-aware fusion module. This staged

This section highlights several key findings observed throughout our experiments and provides further analysis

to better understand the behavior and characteristics of the proposed framework.

470 / -20 (FVD/∆Step)

- .1
- .2
- .3
- .4

- .1
- .2
- .3
- .4

CKA Score

CKA Score

##### 8.1. Initial Interpretation of Depth Information

[Figure 115]

[Figure 116]

[Figure 117]

Block Index 5 10 15 30 5 10 15 30

(a) (b)

Frame 0 Frame 10 Frame 20

- Figure 13. CKA analysis of depth evolution during denoising; importance of early stage denoising. (a) Depth branch exhibits a gradual and persistent increase in similarity, indicating continuous geometric refinement and improved spatial consistency as denoising progresses. (b) Insufficient early-stage denoising results in high variance of CKA similarity across steps, suggesting unstable feature alignment. This instability propagates to later stages, degrading overall spatial coherence and FVD performance.

Effect of Insufficient Early Stage Denoising. As discussed above, increasing the number of early denoising steps effectively improves alignment stability and overall video quality. We further experimented with the opposite setting and found that insufficient early-stage denoising leads to unstable alignment and degraded FVD performance, as reflected by the high variance in CKA similarity in Fig. 13. This suggests that the camera–scene relationship is primarily established at the beginning of the diffusion process, and insufficient early refinement disrupts laterstage feature consistency.

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Frame 10 Frame 20 Frame 30

|1𝑒−5|
|---|

|3𝑒−6|
|---|

- Figure 14. Comparison of learning rate. High learning rate (i.e. 1e−5) lead to unstable training and fail to converge properly. Please zoom in for better visualization.

[Figure 124]

[Figure 125]

[Figure 126]

Frame 0 Frame 20 Frame 40

[Figure 127]

[Figure 128]

[Figure 129]

Frame 0 Frame 30 Frame 60

Figure 12. Illustration of depth-conditioned generation. The depth-conditioned generation may initially be processed similarly to low-light or hazy RGB input. The resulting videos maintain structural coherence, reflecting cross-modal adaptability.

Under our dual-branch framework, depth information is initially provided through the RGB branch in the first stage, where the depth map serves as the conditioning input for the first frame. This raises an interesting question: how is the depth map initially interpreted during this process?

We observe that directly providing a depth map as the input condition can lead the model to process it similarly to low-light, hazy, or monochromatic visual cues, rather than as a fully distinct modality. Nevertheless, the generated videos show minimal distortion or artifacts, indicating that the current state-of-the-art VAE-DiT ([78]) backbone exhibits strong robustness and notable adaptability across input domains (Fig. 12).

##### 8.2. Extented Analysis

How Depth Latent Evolves? We further investigate how the depth branch becomes involved and evolves under the guidance of Pl¨ucker embeddings (Fig. 13). It’s shown that the depth branch exhibits a steadily increasing similarity that remains high throughout denoising, especially during the middle and late stages. This indicates that depth features continuously absorb and reinforce geometric cues to maintain spatial fidelity. Unlike RGB, which relies on early global alignment, depth evolves gradually and persistently contributes to local geometric correction. This also suggests that the conditioning for the depth branch should remain active across the entire process to sustain precise geometric reconstruction.

#### 9. Additional Implementation Details

Ray Conditioning. For each camera view, we generate rays by back-projecting pixel coordinates into 3D using the camera intrinsics and extrinsics. Specifically, given the camera-to-world transformation c2w and intrinsics K = (fx,fy,cx,cy), we compute normalized ray directions in camera space and transform them to world coordinates. Each ray is represented in the Pl¨ucker form [m,d], where d denotes the ray direction and m = o × d encodes the

[Figure 130]

- (b)
- (c)

|[Figure 131]<br><br>[Figure 132]|
|---|
|[Figure 133]<br><br>[Figure 134]|

|[Figure 135]<br><br>[Figure 136]|
|---|
|[Figure 137]<br><br>[Figure 138]|

|[Figure 139]<br><br>[Figure 140]|
|---|
|[Figure 141]<br><br>[Figure 142]|

GTPrediction

(a)

Camera Trajectory Condition (GT) Camera Trajectory of Generated Video

(a) Incorrect Translation Direction (b) Excessive Rotation (c) Insufficient Movement

- Figure 15. Solely Relying on Pl¨ucker Embedding Lacks Scene Understanding. Given the same camera motion, variations in camera movement are observed in the generated video by [78] across different scenes, exhibiting distinct behaviors shown in (a), (b), and (c).

Table 5. User Study Results. Normalized scores (0–1, higher is better) averaged across participants. Each group was evaluated independently to ensure fairness under different numbers of compared methods.

Method Consistency Smoothness Visual Quality Semantic Consistency Average Image-to-Video Setting

Ours 0.96 0.95 0.94 0.97 0.96 Wans [86] 0.88 0.87 0.86 0.88 0.87 SVC [101] 0.80 0.79 0.78 0.81 0.80 CameraCtrl [25] 0.77 0.76 0.75 0.78 0.77 MotionCtrl [86] 0.74 0.73 0.72 0.75 0.74

Text-to-Video Setting

Ours 0.94 0.93 0.92 0.95 0.94 AC3D [1] 0.86 0.84 0.83 0.87 0.85 CameraCtrl [25] 0.79 0.78 0.77 0.80 0.78 MotionCtrl [86] 0.76 0.75 0.74 0.77 0.75

corresponding moment. This process yields a dense field of Pl¨ucker coordinates for all pixels across all views (Algorithm 1). We then encode these Pl¨ucker rays using a pretrained WANv2.1 camera pose encoder [78], which produces compact latent features that capture view-dependent geometry. Specifically, it applys an 8× downsampling operation implemented via pixel shuffle [70] to align with the spatial resolution of the VAE encoder.

Hyperparameter. We used a batch size of 8. Compared to the original training setup [78], which adopted a learning rate of 1 × 10−5, we used a smaller learning rate of 3 × 10−6. The learning rate scheduler remained linear without any modification. We found that this configuration led to better convergence performance (Fig. 14).

Evaluation Datasets. We evaluate our methods on two datasets: RE10K [102] and DL3DV [48]. For RE10K, we randomly sample frames with a stride of 2–8. For DL3DV, which already exhibits large camera motion, we use a stride of 1. We use VBench [34] to exclude videos whose motionsmoothness and consistency score is low.

Prompt Generation. We generate prompts for every sampled video to provide richer supervisory signals. How-

ever, generating captions for all frames can be computationally expensive and resource-inefficient, due to substantial frame overlap within each video. To mitigate this, we generate captions for frames every chosen sample stride, ensuring that each video contains at least one frame with an associated caption while avoiding redundant computation.

#### 10. Additional Experiments

##### 10.1. More Experimental Analysis

Limitation of sole reliance on pl¨ucker embedding. We found that although recent camera-conditioned generative models can recover or specify camera trajectories, they still exhibit important limitations. Many approaches encode motion in Pl¨ucker embeddings without explicit reasoning about scene geometry or semantics, causing the model to conflate camera motion with content-dependent appearance. As a result, the same nominal trajectory yields scene-specific behaviors, like changes in unintended rotations, and inconsistent parallax as illustrated in Fig. 15.

User Study. In our user study, we asked participants to evaluate the following aspects of the generated videos (Tab. 5): Consistency of Camera Trajectories with

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Input Image Naive Propagate Plücker + D Ours GT

- Figure 16. Comparison of fusion strategy. Single-frame depth conditioning fails to maintain a coherent understanding across the entire video, resulting in artifacts and unnatural generation. In contrast, our multi-frame depth synthesis methods help produce more plausible outputs. Please zoom in for more details.

GT: How closely do the camera trajectories match the expected ground truth, maintaining the correct motion paths? Smoothness and Coherence: How smooth and continuous is the video, without any noticeable jerks, flickering, or temporal discontinuities? Visual Quality and Detail: How high is the overall visual quality and detail of the rendered frames? Semantic Consistency with GT: How well does the generated scene match the expected semantic content in terms of object recognition, spatial relationships, and scene structure? The user study results (Table Tab. 5) show that our method outperforms all baselines across all four evaluation aspects.

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Single Branch Dual Branch

- Figure 17. Illustration of coupling the RGB and depth modalities within a single branch leads to interference. This interference causes misalignment between the two modalities and adversely affecting the generation quality.

Algorithm 1 Ray Condition: Pl¨ucker Ray Generation Require: Camera intrinsics K ∈ RB×V ×4, camera-to-

world matrices c2w ∈ RB×V ×4×4, image size (H,W), Ensure: Pl¨ucker ray representation pl¨ucker ∈

RB×V ×H×W×6

- 1: Initialize pixel grids (i,j) ← meshgrid(0,...,W − 1;0,...,H − 1)
- 2: for each batch b = 1,...,B do
- 3: for each view v = 1,...,V do
- 4: Extract intrinsics (fx,fy,cx,cy) ← K[b,v]
- 5: Compute camera-space directions:

x =

i − cx fx

, y =

j − cy fy

, z = 1

- 6: Normalize direction vectors: dcam = normalize([x,y,z])
- 7: Transform to world coordinates: dworld = Rb,vdcam, oworld = tb,v
- 8: Expand oworld to match the resolution (H×W)
- 9: Compute Pl¨ucker coordinates: m = oworld × dworld, pl¨ucker[b,v] = [m,dworld]
- 10: end for
- 11: end for
- 12: return pl¨ucker

2. Pl¨ucker + D: incorporate depth as an additional dimension in the Pl¨ucker ray embedding.

Comparison of Inject Depth Strategies To demonstrate that achieving coherent interaction between depth and RGB under explicit camera control is non-trivial, we compare several integration strategies (Fig. 16):

Single-branch integration adds an auxiliary depth branch with a shared encoder, generating RGB and depth within a single branch, similar with [81].

As illustrated in (Figs. 16 and 17 and Tab. 6), singleframe depth conditioning imposes a static prior and frequently introduces visible artifacts across time, leading to unnatural generations. In contrast, single-branch integration couples RGB and depth within a single pathway and causes unacceptable cross-modal interference, further exacerbating misalignment. By comparison, our dual-branch

Single-frame depth conditioning treats depth as static or simply propagates it across frames. In detail, we compare two strategies:

1. Naive-Prop: propagate the depth of the first frame to all subsequent frames and use it as a conditioning signal (i.e. use a side branch but with static frame).

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

Frame 10 Frame 20 Frame 30 Frame 40

- Figure 18. Failure case under large motion. When the inter-frame motion is large, severe artifacts emerge. Please zoom in for details.

###### Table 6. Ablation on integration strategy.

Table 8. Analysis on different alignment mechanism.

RGB Depth FVD ↓ FID ↓ MS ↑ FC ↑ RE ↓ TE ↓

Integration Strategy FVD ↓ FID ↓ MS ↑ FC ↑

- - 1-5 91.4 59.1 0.984 0.966 1.76 0.36

- - 1-10 87.0 58.4 0.984 0.964 1.56 0.37

- - 1-15 83.2 59.7 0.985 0.964 1.55 0.35

Naive-Prop 98.6 62.7 0.9823 0.9589 Pl¨ucker + D 100.8 65.2 0.9891 0.9648

6-10 1-5 87.4 54.2 0.985 0.966 1.56 0.33 6-15 1-5 84.6 53.2 0.987 0.966 1.55 0.34

Single-Branch 125.1 66.0 0.9787 0.9515 Ours 80.4 49.9 0.9886 0.9677

11-15 1-10 84.2 53.7 0.987 0.965 1.55 0.34

1-5 6-10 80.9 49.5 0.987 0.965 1.27 0.25 1-5 6-15 80.4 49.9 0.989 0.968 1.25 0.25

###### Table 7. Ablation on 3D fusion block.

1-10 11-15 80.3 50.2 0.989 0.969 1.27 0.23

Variant FVD ↓ FID ↓ MS ↑ FC ↑

ControlNet-Style (1D) 112.3 61.8 0.9819 0.9641 Spatial (2D) Fusion 98.4 60.1 0.9815 0.9643 Spatial Temporal (3D) Fusion 85.4 52.7 0.9861 0.9648 3D Fusion + Frame Gating 80.4 49.9 0.9886 0.9677

#### 11. Discussion and Limitation

While our method performs strongly, we identify two intriguing aspects that present potential avenues for enhancement. We discuss them below with the aim of stimulating future research.

framework enables balanced interaction without entanglement, achieving consistent geometry and vivid appearance.

Large Motion. Our method also suffers from issues related to large motion, which is a common challenge in video generation tasks. Other methods use NVS to resolve ambiguities and mitigating flickering for video generation [17, 50, 64, 90, 98]. However, despite our advancements, large motion still presents a difficulty in stable and coherent end to end video generation (Fig. 18).

##### 10.2. More ablation study

Effect of 3D fusion strategy. We perform a progressive ablation to evaluate each component of the 3D fusion block. As shown in Tab. 7, adding spatial, 3D spatiotemporal, and frame-gating modules progressively improves performance.

Effect of Sematic Guided Mutual Alignment. We further investigate the impact of cross-branch communication and fusion depth. Three fusion strategies are compared: (1) One-way Alignment, (2) Geometry-Guided Alignment, (3) Sematic Guided Mutual Alignment. (Ours), In Tab. 8, the left two columns indicate the layer ranges where RGB and depth features are injected, allowing us to analyze the effect of fusion depth under each strategy. As shown in Tab. 8, SIGMA consistently achieves the best video quanlity metrics as well as camera motion consistency, indicating that semantic guided mutual alignment is more effective than one-way or geometry-guided conditioning.

Model Capacity and Parameter Efficiency. Finally, although our parameter count remains comparable to prior designs [16, 20, 24], the overhead in our framework mainly arises from duplicating the backbone to form separate RGB and depth branches. This duplication is acceptable at the 1.3B scale, where even two copies of the WAN V2.1 backbone still yield clear performance advantages, but becomes less favorable when scaling to larger video diffusion models. A promising future direction is to develop more parameter efficient depth guidance, for instance by distilling a compact depth branch that can effectively guide the RGB pathway while keeping the overall model lightweight.

##### 10.3. More Visual Comparison

We provide additional examples to further demonstrate the effectiveness of our method (Figs. 19 and 20).

[Figure 156]

Input Image/ Camera Pose

CameraCtrl Seva Wan Ours GT

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

Input Image/ Camera Pose CameraCtrl Seva

Wan Ours GT

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

