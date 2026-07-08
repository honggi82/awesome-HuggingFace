## ReCamDriving: LiDAR-Free Camera-Controlled Novel Trajectory Video Generation

Yaokun Li1 Shuaixian Wang1,3 Mantang Guo2 Jiehui Huang4 Taojun Ding2 Mu Hu4 Kaixuan Wang2 Shaojie Shen4 Guang Tan1*

1Sun Yat-sen University 2ZYT 3Shenzhen Polytechnic University 4The Hong Kong University of Science and Technology

###### Final

# arXiv:2512.03621v2[cs.CV]29Dec2025

- (a)
- (b)

[Figure 1]

[Figure 2]

[Figure 3]

3DGS Rendering

Recorded Video Ours Output

[Figure 4]

- (c) Camera

Recorded Trajectory Novel Trajectory

[Figure 5]

[Figure 6]

3DGS-based Repair

3DGS Rendering

Output

[Figure 7]

LiDAR Projection

[Figure 8]

[Figure 9]

[Figure 10]

LiDAR-based Camera Controllable Generation

Output

Recorded Video

Trainable in Stage-1

[Figure 11]

[Figure 12]

Trainable in Stage-2

＋

pose

[Figure 13]

[Figure 14]

[Figure 15]

Figure 1. Comparison of novel-trajectory generation. Repair-based methods (e.g., Difix3D+ [53]) suffer from severe artifacts under novel viewpoints, while LiDAR-based camera-controlled methods (e.g., StreetCrafter [57]) show geometric inconsistencies in occluded or distant regions due to incomplete cues. In contrast, ReCamDriving employs a coarse-to-fine two-stage training strategy that leverages dense scenestructure information from novel-trajectory 3DGS renderings for precise camera control and structurally consistent generation.

[Figure 16]

[Figure 17]

### Abstract

for coarse control, while the second stage incorporates 3DGS renderings for fine-grained viewpoint and geometric guidance. Furthermore, we present a 3DGS-based crosstrajectory data curation strategy to eliminate the train–test gap in camera transformation patterns, enabling scalable multi-trajectory supervision from monocular videos. Based on this strategy, we construct the ParaDrive dataset, containing over 110K parallel-trajectory video pairs. Extensive experiments demonstrate that ReCamDriving achieves state-of-the-art camera controllability and structural consistency. Project page: https://recamdriving.github.io/.

We propose ReCamDriving, a purely vision-based, cameracontrolled novel-trajectory video generation framework. While repair-based methods fail to restore complex artifacts and LiDAR-based approaches rely on sparse and incomplete cues, ReCamDriving leverages dense and scenecomplete 3DGS renderings for explicit geometric guidance, achieving precise camera-controllable generation. To mitigate overfitting to restoration behaviors when conditioned on 3DGS renderings, ReCamDriving adopts a twostage training paradigm: the first stage uses camera poses

### 1. Introduction

High-quality multi-trajectory video data is essential for autonomous driving, as it provides diverse viewpoints for tasks such as 3D reconstruction [56, 65] and world-model training [8, 64]. However, collecting real-world multitrajectory data is extremely costly, as it either requires multiple synchronized vehicles to capture videos from different viewpoints or involves repeated traversals with inevitable spatio-temporal inconsistencies. These challenges make high-fidelity novel-trajectory video synthesis from a single recorded trajectory an attractive and scalable alternative.

A common approach for novel-trajectory generation is the reconstruction-then-repair pipeline [11, 28, 53, 59]. It first reconstructs the scene using Neural Radiance Fields (NeRF) [33] or 3D Gaussian Splatting (3DGS) [21], renders videos from novel trajectories, and then trains diffusionbased repair models to restore artifacts in the rendered results. Although effective at artifact removal, this pipeline often fails under complex rendering degradations (Fig. 1a). The core limitation is that repair models learn a local degraded-to-clean mapping based on training-time degradation patterns, whereas highly varying 3D rendering artifacts across scenes and viewpoints often fall out of distribution, leading to failed restoration and 3D inconsistency.

Another paradigm is camera-controlled video generation [2, 63], which regenerates novel-trajectory videos conditioned on camera pose sequences and generally produces visually coherent results. However, conditioning simply on camera poses often leads to imprecise viewpoint control. To achieve more accurate camera guidance, FreeVS [46] and StreetCrafter [57] incorporate novel-trajectory LiDAR projections to provide explicit geometric cues, but LiDAR projections are sparse and incomplete in background or occluded regions, often resulting in 3D-inconsistent results (Fig. 1b). Moreover, training such camera-transformation models requires ground-truth novel-trajectory supervision, which is unavailable in autonomous driving datasets. As a compromise, prior works [46, 57] construct pseudo training pairs from different segments of a single recorded trajectory (Fig. 2a), which only captures longitudinal motion patterns and leads to degraded performance when inferring lateral motion trajectories.

To address these challenges, we propose ReCamDriving, a pure vision-based novel-trajectory video generation framework for autonomous driving. Our approach performs camera-controlled video generation and, crucially, replaces costly LiDAR projections with novel-trajectory 3DGS renderings as camera-control signals. The key insight is that, although 3DGS renderings have lower geometric fidelity than LiDAR projections, they preserve more complete and structurally informative scene cues across all regions, enabling more precise camera control and structurally consistent generation. Specifically, to avoid the model collaps-

ing into a trivial artifact-repair solution when introducing 3DGS renderings, we adopt a two-stage training strategy for progressive learning of viewpoint transformation. We first train the model with camera poses to establish basic camera-controlled generation capability. Then, we freeze the base network and introduce additional modules that incorporate 3DGS renderings for fine-grained viewpoint and structural guidance. Without this scheme, the model tends to overfit to artifact restoration rather than learning true camera-controlled generation, exhibiting limitations similar to repair-based methods.

Furthermore, to construct lateral trajectory supervision and bridge the mismatch between training and inference camera-transformation patterns, we introduce a 3DGSbased cross-trajectory data curation strategy (Fig. 2). During training, 3DGS renderings of laterally offset trajectories are used as source views, while the recorded trajectory provides supervision. At inference, the recorded trajectory serves as the source to generate laterally novel trajectories, ensuring consistency between training and inference camera-transformation patterns. This strategy enables multi-trajectory supervision from monocular videos, allowing us to construct the ParaDrive dataset containing about

- 1.6K 3DGS scenes from Waymo Open Dataset (WOD) [40] and NuScenes [7], containing over 110K parallel-trajectory pairs of 3DGS renderings and recorded videos. By relying solely on monocular data, this approach enables scalable multi-trajectory dataset construction and has the potential to extend camera-controlled models toward web-scale videos. Extensive experiments validate the effectiveness of our method. Our main contributions are:

- • We propose ReCamDriving, a novel vision-based framework for novel-trajectory video generation that leverages 3DGS renderings to achieve precise camera control and structurally consistent generation.
- • We introduce a novel 3DGS-based cross-trajectory data curation strategy for scalable multi-trajectory supervision, and construct the ParaDrive dataset with over 110K parallel-trajectory pairs.
- • Extensive experiments demonstrate that ReCamDriving achieves the state-of-the-art performance in both camera controllability and 3D consistency.

- 2. Related Work 2.1. Diffusion Priors for Repairing 3D Renderings.

Recent advances in NeRF [33] and 3DGS [21] have greatly advanced 3D reconstruction [3, 4, 20, 24, 32, 49–51, 62]. However, under limited viewpoints, these methods still produce noticeable artifacts during novel-view synthesis, especially in extrapolated regions [23, 43, 66]. To mitigate this issue, recent works explore learning diffusion priors to repair degraded 3D renderings. 3DGS-Enhancer [28]

fine-tunes a video diffusion model for rendering restoration, Difix3D+ [53] enables real-time neural enhancement, and GSFixer [59] conditions video restoration on both semantic and geometric cues. Freesim [11] introduces a framework for novel-trajectory restoration through a progressive reconstruction strategy. While these methods can improve the visual fidelity of 3D renderings, they essentially address a restoration problem, focusing on local artifact correction rather than consistent scene-level geometry modeling. Consequently, their performance heavily depends on the training data distribution, and since rendering artifacts vary significantly across scenes and viewpoints, they often fail when facing out-of-distribution degradations.

###### (a) Train (b) Inference

Recorded

Recorded Novel

[Figure 18]

Camera Trans.

Previous work

Camera Trans.

[Figure 19]

Recorded Novel

Camera Trans.

[Figure 20]

Ours

###### (c)

Camera Camera Conditions

[Figure 21]

[Figure 22]

pose

Loss

[Figure 23]

Recorded-Traj. Video

Novel-Traj. Rendering

Train

[Figure 24]

#### 2.2. Camera-Controlled Video Generation.

Camera Conditions

Camera-controlled generation has attracted increasing attention for synthesizing immersive scenes and novel viewpoints [2, 16, 31, 37, 47, 54, 55, 63]. The key challenge lies in how to represent camera motion and inject it into generative models for precise viewpoint control. Early works [1, 2, 13, 52, 58] condition diffusion models on camera extrinsics or Pl¨ucker embeddings, enabling controllable generation but often resulting in inconsistent targetview geometry. From a representation-learning perspective [5, 35, 41, 42], such pose-conditioned models primarily capture statistical correlations between camera motion and appearance variation rather than geometric causality, leading to spatial misalignment and geometric instability.

[Figure 25]

[Figure 26]

[Figure 27]

Novel-Traj. Generation

Recorded-Traj. Video

Inference

Figure 2. (a–b) Comparison of training and inference cameratransformation patterns. (c) Our training and inference data strategy. (Trans.: Transformation; Traj.: Trajectory)

vised learning of lateral camera transformations. Specifically, given an original trajectory video, we first reconstruct a 3DGS representation of the scene and laterally shift the camera trajectory by a fixed offset (e.g., +3 m) to render a novel trajectory. Since the rendered novel-trajectory video contains artifacts and cannot serve as ground truth, we instead use it as the source input during training, while the clean recorded-trajectory video provides the groundtruth supervision (Fig. 2c). At inference, we use the clean recorded video as the source input to generate laterally shifted novel-trajectory views . Although the network is trained with rendered videos as source inputs but tested with clean recorded videos, experiments show that this train–inference source mismatch does not degrade performance. On the contrary, using clean source videos during inference further improves visual quality. Please refer to Sec. 5.3 for more details.

[Figure 28]

To address these issues, recent approaches [60, 61] incorporate 3D point-map priors [17, 22, 48] to inject explicit geometric structure, yet their performance remains limited by the quality of reconstructed point maps in large-scale driving scenes. Alternatively, FreeVS [46] and StreetCrafter [57] leverage LiDAR point clouds for camera control, yet LiDAR data are costly and spatially incomplete. In contrast, our approach adopts a purely visual formulation for novel-trajectory generation, replacing LiDAR with 3DGS renderings as structural camera conditions to achieve fine-grained, geometrically consistent control.

### 3. Cross-trajectory Data Curation and ParaDriving Dataset

To construct the training pairs, we reconstruct approximately 1.6K 3DGS scenes from WOD [40] and NuScenes [7] using the DriveStudio framework [9]. During 3DGS training, each scene saves intermediate models at 100, 500, and 1,000 iterations, producing underfitted recorded-trajectory renderings that serve as training-time camera conditions with varying artifact levels, in order to align them with the artifact-prone novel-trajectory renderings used during inference. The 30K-iteration models are used to render eight laterally shifted trajectories (±1 m, ±2 m, ±3 m, and ±4 m), which serve as source videos during training. Each trajectory contains three 121-frame

Training a camera-controlled video regeneration model requires ground-truth videos from novel trajectories for supervision. While general datasets [26, 36] provide such supervision, autonomous driving datasets typically contain only single-pass trajectories. Prior works [46, 57] mitigate this limitation by splitting a single trajectory into source and target segments (Fig. 2a), but this setting models only longitudinal motion, resulting in degraded performance for lateralview generation during inference.

To address this challenge, we propose a 3DGS-based cross-trajectory data curation strategy that enables super-

clips (front, middle, and rear). In total, the 3DGS training process consumed 8,240 L20 GPU hours and produced over 110K dual-trajectory video pairs. We name this dataset ParaDrive, which will be released to facilitate research on camera-controlled video generation.

### 4. ReCamDriving

Given a source trajectory video Vs, our goal is to synthesize a novel trajectory video Vt with lateral offsets. To this end, we adopt a two-stage training strategy that progressively guides the model to perform viewpoint transformation, as illustrated in Fig. 3. In Stage 1, we aim for the network to understand and learn the physical process of viewpoint transformation by using the relative camera pose ∆T = Tt←s ∈ SE(3) between trajectories, enabling it to warp information from the source video to the target trajectory. In Stage 2, we introduce novel-trajectory 3DGS renderings as fine-grained guidance, freeze the first-stage parameters, and add two additional attention modules to incorporate 3DGS renderings for accurate viewpoint and structural generation.

#### 4.1. Preliminary: Latent Diffusion Models with Flow Matching

Latent Diffusion Models (LDMs) [38] perform diffusion in a compact latent space learned by a pretrained autoencoder, achieving high-fidelity generation with reduced computational cost. This balance between efficiency and quality has made LDMs the foundation of many state-of-theart image and video generators [6, 38, 45]. Given a video V ∈ RF×3×H×W, a 3D VAE encoder projects it into latents x ∈ Rf×c×h×w with compressed spatial and temporal dimensions. Diffusion and generation are then performed in this latent space, and the decoded outputs reconstruct the final video frames.

Traditional diffusion models [15, 19, 39] formulate generation as a stochastic differential equation (SDE), reversing a predefined noise process but suffering from training instability and slow inference. Flow Matching [10, 27] replaces this stochastic formulation with a deterministic ordinary differential equation (ODE) that learns a time-dependent velocity field to transport samples from noise to data, yielding more stable training and faster sampling. The forward process is typically defined as a straight path between noise x0 and data x1:

xt = tx1 + (1 − t)x0, t ∈ [0,1]. (1) Differentiating with respect to t gives the corresponding velocity field:

dxt dt

= x1 − x0. (2) In the context of camera-controlled video generation, given a camera condition ccam, the training objective of flow

vt =

matching can be formulated as:

0, x1, ccam, t∼U(0,1) ϵθ(xt;ccam,t) − vt 2, (3)

LFM = Ex

where ϵθ denotes the neural network predicting the velocity field.

#### 4.2. Training Stage 1: Relative Pose-Guided Camera Transformation

In this stage, the relative camera pose ∆T is used as a condition to guide the model in re-generating the sourcetrajectory video. Instead of using absolute poses, relative poses are easier to specify during inference and are more robust to calibration errors. ReCamMaster [2] adopts a similar idea but concatenates the relative pose with both the source latent xs and the noisy latent xt, which can introduce ambiguity regarding which latent corresponds to the source or the target view.

To resolve this, we separately encode the relative pose ∆T and the identity pose TI = I4 ∈ SE(3) (representing zero motion) using a camera encoder Ecam, producing cr = Ecam(∆T) ∈ Rf×d and cI = Ecam(TI) ∈ Rf×d, where d denotes the feature dimension. We further introduce a learnable frame embedding Ef ∈ Rf×d that provides frame-wise correspondence cues, enhancing temporal alignment in self-attention operations.

After encoding the source and noisy videos using the 3D VAE encoder, we obtain latent representations xs,xt ∈ Rf×c×h×w. A patchify operation unfolds each spatial feature map into a sequence of local tokens, resulting in xs,xt ∈ Rf×l×d, where l = h × w. The camera and frame embeddings are then added to each latent via broadcasting along the l-dimension:

xi = Cat(xt + cr + Ef, xs + cI + Ef), (4)

where Cat denotes concatenation along the frame dimension f. The resulting sequence xi is processed by N layers of Diffusion Transformer (DiT) [34] blocks to predict the velocity field. Each DiT block in this stage (see Fig. 3) consists of a self-attention layer, a feed-forward network (FFN), and normalization layers, with only the self-attention parameters unfrozen for adaptation.

We adopt the flow-matching framework to schedule noise levels and optimize the diffusion process. We adopt the flow-matching framework to schedule noise levels and optimize the diffusion process. The training objective follows Eq. 3, where ccam denotes the relative camera-pose embedding and the target data x1 corresponds to the clean recorded-trajectory videos.

#### 4.3. Training Stage 2: Fine-grained Camera Control via 3DGS Renderings

In this stage, to improve camera control accuracy and structural guidance, besides the relative camera pose, we utilize

Identity Pose

Novel Trajectory Generation

Source Trajetory Video

Cam. Encoder

[Figure 29]

[Figure 30]

[Figure 31]

N x DiT Blocks

[Figure 32]

[Figure 33]

3DVAE Encoder

[Figure 34]

[Figure 35]

＋

- Stage 1
- Stage 2

Frame Embedding

Add&Norm

Add&Norm

Self-AttnSelf-Attn

FFN

[Figure 36]

Relative Pose

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Cam. Encoder

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

3DVAE Decoder

[Figure 46]

…

＋

C

[Figure 47]

Add&Norm

Add&Norm

Cross-Attn

Frame Embedding

FFN

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Novel Trajectory 3DGS Rendering

Relative Pose

Rendering-Attn

[Figure 54]

Cam. Encoder

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

3DVAE Encoder

＋

C

Source Trajectory Frozen Trainable

Add Concatnate Share

[Figure 61]

＋

[Figure 62]

Novel Trajectory

Frame Embedding

Figure 3. Overview of our framework. We adopt a two-stage training scheme for precise and structurally consistent novel-trajectory video generation. In Stage 1, ReCamDriving trains DiT blocks conditioned on the source trajectory video and relative camera (cam.) pose. When switching to Stage 2, the original DiT parameters are frozen, and additional attention modules are introduced to integrate 3DGS renderings for fine-grained view control and structural guidance. Shared modules between stages are connected with blue dashed lines.

Novel Trajectory Video

Identity pose

Source Trajetory Video

Cam. encoder

[Figure 63]

[Figure 64]

[Figure 65]

N x DiT Blocks

[Figure 66]

3DVAE Encoder

[Figure 67]

[Figure 68]

＋

DriveStudio [9] to reconstruct 3DGS representations and render novel-trajectory 3DGS renderings as additional conditions. The renderings Vgs are encoded by the same 3D VAE encoder and patchified to obtain xgs ∈ Rf×l×d. We then augment xgs with the relative camera-pose embedding and frame embedding:

cases similar to repair-based models during inference.

- Stage 1
- Stage 2

Training in this stage also follows Eq. 3, but with the condition ccam extended to include both the relative pose embedding and the 3DGS rendering latent. Finally, during inference, the 3D VAE decoder reconstructs the noveltrajectory video Vt from the stage 2 latent representation, achieving precise camera control and structural consistency.

Frame Embedding

Add&Norm

Add&Norm

Self-AttnSelf-Attn

[Figure 69]

FFN

Relative pose

[Figure 70]

[Figure 71]

[Figure 72]

Cam. encoder

[Figure 73]

[Figure 74]

[Figure 75]

…

[Figure 76]

3DVAE Decoder

[Figure 77]

＋

[Figure 78]

C

##### x¯gs = xgs + cr + Ef. (5)

Add&Norm

Add&Norm

Cross-Attn

Frame Embedding

### 5. Experiments

FFN

To integrate 3DGS features effectively, each DiT block introduces two additional components: a Rendering Attention and a Cross Attention, while the self-attention parameters from stage 1 are kept frozen. The Rendering Attention serves as an auxiliary self-attention layer that refines spatio–temporal representations within the 3DGS rendering latent space. It shares the same structure as the self-attention used in stage 1 but is named differently to avoid confusion. The Cross Attention then fuses the rendering latent features x¯gs with the diffusion latent x¯i = SelfAttn(xi) to enhance structural and geometric consistency between the generated and target trajectories. The self-attention modules from stage 1 remain frozen to preserve their cameracontrolled generation capability, ensuring that x¯i already encodes coarse camera transformations when interacting with 3DGS rendering features. This design encourages the model to leverage the geometric cues embedded in 3DGS renderings for fine-grained view refinement. Without this constraint, the model tends to overfit to artifact-repair behaviors on 3DGS renderings rather than assisting the firststage module in camera relocalization, leading to failure

[Figure 79]

#### 5.1. Experimental setup

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Novel Trajectory 3DGS Renderings

Relative pose

Implementation. We train ReCamDriving on the proposed ParaDrive dataset in two stages. Each stage is conducted on 64 NVIDIA A100 GPUs for 6,000 steps, with a batch size of 1 and a learning rate of 1e-4. The total training takes approximately 3.5 days. The training resolution is set to 480 × 832, and each video consists of 121 frames. We initialize our model from the Wan2.1 text-to-video foundation model [45], with the text prompt set to an empty string by default. Specifically, the 3D VAE, rendering attention, cross attention, and FFN modules are initialized from Wan2.1, while the camera encoder and self-attention layers are initialized from ReCamMaster [2].

Rendering-Attn

Cam. encoder

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

3DVAE Encoder

＋

C

Source Trajectory Frozen Trainable

Add Concatnate Share

＋

[Figure 89]

Novel Trajectory

Frame Embedding

Evaluation Dataset and Metrics. We select 20 scenes each from WOD [40] and NuScenes [7] for validation. We evaluate our method from three perspectives: (1) Visual Quality: we use Imaging Quality (IQ) [18] to assess fidelity and the average CLIP similarity of adjacent frames (CLIPF) to measure temporal consistency. (2) Camera Accuracy:

Table 1. Quantitative comparison results on WOD [40], where bold indicates the best performance, and underline denotes the second best.

Lateral Offset ±1m Lateral Offset ±2m Method Visual Quality Cam. Accuracy View Consistency Visual Quality Cam. Accuracy View Consistency

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

IQ↑ CLIP-F↑ RErr.↓ TErr.↓ FID↓ FVD↓ CLIP-V↑ IQ↑ CLIP-F↑ RErr.↓ TErr.↓ FID↓ FVD↓ CLIP-V↑

DriveStudio 52.13 98.84 - - 83.32 25.37 94.78 47.32 98.49 - - 104.24 39.79 94.23 Difix3D+ 64.24 98.92 1.36 2.42 56.35 27.80 95.32 63.11 98.41 1.64 2.66 57.73 31.88 92.85 FreeVS 62.74 95.74 1.71 2.88 63.06 37.06 88.99 60.16 92.59 2.12 2.93 67.87 43.59 88.41 StreetCrafter 63.57 97.31 1.52 2.53 28.18 20.51 96.01 63.78 97.17 1.79 2.77 46.78 22.81 94.74

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Ours 65.18 99.31 1.32 2.37 13.76 13.27 97.96 65.34 99.03 1.45 2.43 25.01 14.08 97.18

Method Lateral Offset ±3m Lateral Offset ±4m

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

DriveStudio 43.83 98.27 - - 116.12 63.21 90.37 41.47 98.21 - - 144.05 72.50 88.76 Difix3D+ 60.88 97.38 2.01 2.97 66.39 45.23 91.76 58.81 97.14 2.68 3.12 78.08 65.37 90.12 FreeVS 57.71 92.01 3.17 3.78 84.87 55.76 86.33 56.15 91.26 3.02 3.39 107.04 58.39 85.17 StreetCrafter 59.34 96.89 1.91 3.13 50.75 30.26 92.13 59.89 96.63 2.87 3.03 68.73 36.67 91.17

Source View DriveStudio Difix3D+ Ours

Ours 62.68 98.79 1.63 2.65 28.38 22.59 96.50 61.32 98.45 1.57 2.73 32.36 26.76 94.91

Table 2. Average quantitative results of novel-trajectory generation on NuScenes [7].

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Visual Quality View Consistency

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Method

[Figure 117]

[Figure 118]

[Figure 119]

IQ↑ CLIP-F↑ FID↓ FVD↓ CLIP-V↑

DriveStudio 45.39 97.78 103.25 52.93 89.19 Difix3D+ 61.76 98.16 65.14 41.37 90.48

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Ours 62.38 98.75 25.68 18.98 96.14

[Figure 124]

[Figure 125]

[Figure 126]

Source View DriveStudio Difix3D+ Ours

#### 5.2. Comparison Results

Figure 4. Qualitative comparison results on NuScenes [7].

Qualitative Results. FreeVS [46] and StreetCrafter [57] provide pretrained weights only on WOD. For a fair comparison, we train a version of our model on the WOD subset of ParaDrive, with qualitative results shown in Fig. 5. We further compare our full model trained on the entire ParaDrive with Difix3D+ [53] dataset and DriveStudio [9] on NuScenes [7] (Fig. 4).

Following CamCloneMaster [30], we adopt the state-ofthe-art (SOTA) camera estimation method MegaSaM [25] to recover camera parameters from the generated videos, and evaluate accuracy using Rotation Error (RErr.) and Translation Error (TErr.). (3) View Consistency: We compute the Fr´echet Video Distance (FVD) [44] and Fr´echet Image Distance (FID) [14] between the source and generated video distributions, and further compute the framewise CLIP similarity (CLIP-V) to assess cross-view semantic consistency.

ReCamDriving consistently outperforms all baselines. Difix3D+ often produces view-inconsistent corrections on fine structures near the ego vehicle (e.g., lane markings, road text), whereas ReCamDriving maintains structural continuity. FreeVS and StreetCrafter exhibit severe inconsistencies under large viewpoint shifts due to sparse LiDAR conditioning. As shown in Fig. 5, StreetCrafter fails to reconstruct complete vehicle geometry in the first scene, while both baselines yield blurred or 3D-inconsistent distant backgrounds caused by occluded or sparsely sampled LiDAR regions. In contrast, ReCamDriving leverages dense structural cues from 3DGS renderings for robust view and geometric guidance, enabling more reliable generation of

Baselines. We compare ReCamDriving against state-ofthe-art novel trajectory synthesis methods [9, 46, 53, 57]. DriveStudio [9] reconstructs 3DGS using a dynamic neural scene graph for novel-trajectory rendering. Difix3D+ [53] follows the reconstruction–repair paradigm, using a singlestep diffusion model to restore artifacted 3DGS renderings. In addition, FreeVS [46] and StreetCrafter [57] adopt a camera-controlled video generation approach, where the target trajectory is conditioned on colored LiDAR point projections to synthesize novel-trajectory videos.

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

Source View

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

w. Pose

[Figure 150]

[Figure 151]

[Figure 152]

w. Pose + LiDAR

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

w. Pose + LiDAR + GS

[Figure 161]

[Figure 162]

Source View DriveStudio Difix3D+ FreeVS StreetCrafter Ours

w. Pose

+ GS

Figure 5. Qualitative comparison results on WOD [40]. Our method and Difix3D+ [53] use novel-trajectory renderings from DriveStudio [9] for camera control and restoration, respectively. Note that the officially released FreeVS [46] model is trained and tested on a cropped resolution that excludes sky regions to reduce computation and avoid LiDAR-sparse areas.

Table 3. Quantitative ablation of camera conditions on WOD [40].

[Figure 163]

[Figure 164]

Source View

Source View

Camera Condition IQ↑ FID↓ FVD↓ RErr.↓ TErr.↓

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Pose 60.13 34.86 32.31 3.01 4.23 Pose + LiDAR 61.32 31.23 27.78 1.53 2.69 Pose + LiDAR + GS 63.42 24.75 19.27 1.41 2.47

w. Pose w. Pose+LiDAR w. Pose w. Pose+LiDAR

Pose + GS (Ours) 63.63 24.88 19.18 1.49 2.55

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

ble 2, respectively. Since DriveStudio renders results using ground-truth poses, we do not evaluate its camera accuracy. As shown, ReCamDriving consistently outperforms all baselines across all metrics. Notably, as the lateral offset increases, the view consistency of FreeVS [46], StreetCrafter [57], and Difix3D+ [53] degrades sharply, whereas ReCamDriving remains stable, demonstrating superior robustness under large viewpoint shifts.

w. Pose+LiDAR+GS w. Pose+GS (Ours) w. Pose+LiDAR+GS w. Pose+GS (Ours)

- Figure 6. Qualitative ablation of camera conditions on WOD [40].

Novel Trajectory 3DGS Rendering

Repair Baseline Ours

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

Source Trajectory Video

- Figure 7. Qualitative ablation of training paradigms on WOD [40].

#### 5.3. Ablation Studies

Effectiveness of 3DGS Rendering Condition. In Stage 2, 3DGS renderings of the target trajectory are used as conditions for fine-grained camera control. We ablate it with three variants: (1) without Stage 2 (Pose only); (2) Stage 2 with LiDAR projection (Pose + LiDAR); (3) Stage 2 with both LiDAR projection and 3DGS renderings (Pose + LiDAR + GS). LiDAR projections are generated following StreetCrafter [57] and injected into the Rendering Attention branch. Qualitative and quantitative results are presented in Fig. 6 and Table 3. As shown, using only camera poses often results in inaccurate camera control, while incorporating LiDAR improves pose accuracy but still causes geometric

distant content and occluded objects.

Quantitative Results. Quantitative comparisons on WOD and NuScenes are reported in Table 1 and Ta-

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 211]

Blurred Source View

Clean Source View

Results

Results

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

Clean Source Inputs Outputs (Ours)

Blurred Source Inputs Outputs

Figure 9. Ablation on different source inputs at inference.

One-Stage Training Outputs

Two-Stage Training Outputs (Ours)

Source Inputs

Source View One-Stage Training Two-Stage Training

Table 6. Ablation of our data curation strategy on WOD [40].

Figure 8. Qualitative ablation on our two-stage training strategy.

Camera Transformation RErr.↓ TErr.↓ FID↓ FVD↓ Longitudinal 1.97 3.02 34.17 28.54 Lateral (Ours) 1.49 2.55 24.88 19.18

- Table 4. Qualitative ablation of training paradigms on WOD [40].

Method IQ↑ CLIP-F↑ FID↓ FVD↓ CLIP-V↑

Repair Baseline 62.16 98.18 40.87 31.44 91.32 Ours 63.63 98.90 24.88 19.18 96.64

- Table 5. Quantitative ablation on our two-stage training strategy on WOD [40] and NuScenes [7].

[Figure 232]

firming its effectiveness.

Ablation on Cross-Trajectory Data Curation. We propose a cross-trajectory data curation strategy to align camera-transformation modes between training and inference. To evaluate its effectiveness, we train a baseline using longitudinal transformations following FreeVS [46], where the first and last 121 frames of each recorded trajectory are used as source and supervision, respectively. Table 6 presents the comparison results, showing that our crosstrajectory strategy significantly improves both camera accuracy and view consistency for novel-trajectory prediction, confirming its effectiveness.

[Figure 235]

Training strategy IQ↑ FID↓ FVD↓ CLIP-V↑

One-stage 59.97 32.64 25.16 94.78 Two-stage (Ours) 63.42 25.13 18.32 96.32

Source View One-Stage Training Two-Stage Training

inconsistencies under large offsets due to LiDAR sparsity. Our 3DGS-conditioned model effectively mitigates these issues. Moreover, although combining LiDAR and 3DGS yields slightly better scores, the gain is marginal compared to the higher LiDAR cost, so we adopt 3DGS-only conditioning as our final design.

Impact of Train–Test Source-Video Mismatch. During training, we use blurred 3DGS renderings as source videos, while at test time the source is the clean recorded trajectory. To examine the impact of this mismatch, we compare inference using either clean recorded videos or their degraded 3DGS-rendered counterparts as sources. As shown in Fig. 9, degraded inputs still yield reasonable results, but clean sources produce noticeably sharper backgrounds, indicating that clean source videos at test time do not cause degradation but instead improve performance.

Repair Model vs Regeneration Model. We further compare the adopted video regeneration paradigm with a repair-based baseline. To train the repair baseline, we construct pairs on the recorded trajectory using blurred 3DGS renderings as inputs and clean videos as targets, with Rendering Attention removed. Both methods are evaluated on WOD [40], and qualitative and quantitative results are presented in Fig. 7 and Table 4. As shown, the repair-based model often fails to correct artifacts, as these degradations are not well covered by its training distribution.

### 6. Conclusion and Limitations

In this work, we introduced ReCamDriving, a pure visionbased framework for controllable novel-trajectory video generation. By leveraging 3DGS renderings as structural camera conditions and adopting a two-stage training paradigm, our model achieves precise camera control and consistent geometry. We also construct the large-scale ParaDrive dataset using the proposed cross-trajectory data curation strategy, which enables scalable multi-trajectory supervision from monocular videos. Extensive experiments show that ReCamDriving achieves higher camera accuracy and visual fidelity than existing baselines. Nevertheless, our

Ablation on the Training Strategies. We employ a two-stage strategy to ensure that 3DGS renderings enhance camera control rather than artifact repair. To evaluate its effectiveness, we compare it with a single-stage scheme where all modules are trained jointly without freezing on the full ParaDrive dataset. Quantitative and qualitative results are shown in Table 5 and Fig. 8. The single-stage model tends to produce artifacts and exhibits degraded visual quality similar to repair-based methods, whereas the two-stage strategy yields clearer and more 3D-consistent results, con-

method remains limited in handling small distant objects (e.g., faraway pedestrians), where structural cues become unreliable. Incorporating stronger structural priors for such regions is a promising direction for future work.

### References

- [1] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22875–22889, 2025. 3
- [2] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647, 2025. 2, 3, 4, 5
- [3] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5855–5864,

2021. 2

- [4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470–5479, 2022. 2
- [5] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8):1798–1828, 2013. 3
- [6] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 4, 12
- [7] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020. 2, 3, 5, 6, 8, 12
- [8] Yuntao Chen, Yuqi Wang, and Zhaoxiang Zhang. Drivinggpt: Unifying driving world modeling and planning with multi-modal autoregressive transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 26890–26900, 2025. 2
- [9] Ziyu Chen, Jiawei Yang, Jiahui Huang, Riccardo de Lutio, Janick Martinez Esturo, Boris Ivanovic, Or Litany, Zan Gojcic, Sanja Fidler, Marco Pavone, Li Song, and Yue Wang. Omnire: Omni urban scene reconstruction. In The Thirteenth International Conference on Learning Representations, 2025. 3, 5, 6, 7
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik

- Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 4
- [11] Lue Fan, Hao Zhang, Qitai Wang, Hongsheng Li, and Zhaoxiang Zhang. Freesim: Toward free-viewpoint camera simulation in driving scenes. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12004– 12014, 2025. 2, 3
- [12] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. Advances in Neural Information Processing Systems, 37:91560–91596, 2024. 12
- [13] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 3
- [14] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 4
- [16] Teng Hu, Jiangning Zhang, Ran Yi, Yating Wang, Hongrui Huang, Jieyu Weng, Yabiao Wang, and Lizhuang Ma. Motionmaster: Training-free camera motion transfer for video generation. arXiv preprint arXiv:2404.15789, 2024. 3
- [17] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2005–2015, 2025. 3
- [18] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 5
- [19] Aapo Hyv¨arinen and Peter Dayan. Estimation of nonnormalized statistical models by score matching. Journal of Machine Learning Research, 6(4), 2005. 4
- [20] Yifan Jiang, Peter Hedman, Ben Mildenhall, Dejia Xu, Jonathan T Barron, Zhangyang Wang, and Tianfan Xue. Alignerf: High-fidelity neural radiance fields via alignmentaware training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 46–55,

2023. 2

- [21] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 2

- [22] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision, pages 71–91. Springer, 2024. 3

- [23] Yaokun Li, Chao Gou, and Guang Tan. Taming uncertainty in sparse-view generalizable nerf via indirect diffusion guidance. arXiv preprint arXiv:2402.01217, 3, 2024. 2
- [24] Yaokun Li, Shuaixian Wang, and Guang Tan. Id-nerf: Indirect diffusion-guided neural radiance fields for generalizable view synthesis. Expert Systems with Applications, 266: 126068, 2025. 2
- [25] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. Megasam: Accurate, fast and robust structure and motion from casual dynamic videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10486–10496, 2025. 6
- [26] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024. 3
- [27] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 4
- [28] Xi Liu, Chaoyi Zhou, and Siyu Huang. 3dgs-enhancer: Enhancing unbounded 3d gaussian splatting with viewconsistent 2d diffusion priors. Advances in Neural Information Processing Systems, 37:133305–133327, 2024. 2
- [29] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 851–866. 2023. 12
- [30] Yawen Luo, Jianhong Bai, Xiaoyu Shi, Menghan Xia, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Tianfan Xue. Camclonemaster: Enabling reference-based camera control for video generation. arXiv preprint arXiv:2506.03140,

2025. 6

- [31] Yue Ma, Kunyu Feng, Zhongyuan Hu, Xinyu Wang, Yucheng Wang, Mingzhe Zheng, Xuanhua He, Chenyang Zhu, Hongyu Liu, Yingqing He, et al. Controllable video generation: A survey. arXiv preprint arXiv:2507.16869,

2025. 3

- [32] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi, Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo collections. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7210–7219, 2021. 2
- [33] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2
- [34] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 4

- [35] Rui Qian, Tianjian Meng, Boqing Gong, Ming-Hsuan Yang, Huisheng Wang, Serge Belongie, and Yin Cui. Spatiotemporal contrastive video representation learning. In Proceedings

- of the IEEE/CVF conference on computer vision and pattern recognition, pages 6964–6974, 2021. 3
- [36] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10901–10911, 2021. 3
- [37] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas M¨uller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6121–6132, 2025. 3
- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 4
- [39] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 4
- [40] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2446–2454, 2020. 2, 3, 5, 6, 7, 8, 12
- [41] Zhenchao Tang, Hualin Yang, and Calvin Yu-Chian Chen. Weakly supervised posture mining for fine-grained classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 23735–23744,

2023. 3

- [42] Zhenchao Tang, Guanxing Chen, Shouzhi Chen, Jianhua Yao, Linlin You, and Calvin Yu-Chian Chen. Modal-nexus auto-encoder for multi-modality cellular data integration and imputation. Nature Communications, 15(1):9021, 2024. 3
- [43] Prune Truong, Marie-Julie Rakotosaona, Fabian Manhardt, and Federico Tombari. Sparf: Neural radiance fields from sparse and noisy poses. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4190–4200, 2023. 2
- [44] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 6
- [45] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 4,

- 5, 12

[46] Qitai Wang, Lue Fan, Yuqi Wang, Yuntao Chen, and Zhaoxiang Zhang. Freevs: Generative view synthesis on free driving trajectory. arXiv preprint arXiv:2410.18079, 2024. 2, 3,

- 6, 7, 8

- [47] Qinghe Wang, Yawen Luo, Xiaoyu Shi, Xu Jia, Huchuan Lu, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Cinemaster: A 3d-aware and controllable framework for cinematic text-to-video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–10, 2025. 3
- [48] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024. 3
- [49] Shuaixian Wang, Haoran Xu, Yaokun Li, Jiwei Chen, and Guang Tan. Ie-nerf: Inpainting enhanced neural radiance fields in the wild. arXiv preprint arXiv:2407.10695, 2024. 2
- [50] Shuaixian Wang, Yaokun Li, Chenhui Guo, and Guang Tan. Learning hierarchical uncertainty from hybrid representations for neural active reconstruction. Pattern Recognition, page 112493, 2025.
- [51] Yuehao Wang, Chaoyi Wang, Bingchen Gong, and Tianfan Xue. Bilateral guided radiance field processing. ACM Transactions on Graphics (TOG), 43(4):1–13, 2024. 2
- [52] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 3
- [53] Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, and Huan Ling. Difix3d+: Improving 3d reconstructions with single-step diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26024–26035, 2025. 1, 2, 3, 6, 7
- [54] FU Xiao, Xian Liu, Xintao Wang, Sida Peng, Menghan Xia, Xiaoyu Shi, Ziyang Yuan, Pengfei Wan, Di Zhang, and Dahua Lin. 3dtrajmaster: Mastering 3d trajectory for multientity motion in video generation. In The Thirteenth International Conference on Learning Representations, 2024. 3
- [55] Zeqi Xiao, Wenqi Ouyang, Yifan Zhou, Shuai Yang, Lei Yang, Jianlou Si, and Xingang Pan. Trajectory attention for fine-grained video motion control. arXiv preprint arXiv:2411.19324, 2024. 3
- [56] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and Sida Peng. Street gaussians: Modeling dynamic urban scenes with gaussian splatting. In European Conference on Computer Vision, pages 156–173. Springer, 2024. 2
- [57] Yunzhi Yan, Zhen Xu, Haotong Lin, Haian Jin, Haoyu Guo, Yida Wang, Kun Zhan, Xianpeng Lang, Hujun Bao, Xiaowei Zhou, et al. Streetcrafter: Street view synthesis with controllable video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 822–832, 2025. 1, 2, 3, 6, 7
- [58] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with userdirected camera movement and object motion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 3

- [59] Xingyilang Yin, Qi Zhang, Jiahao Chang, Ying Feng, Qingnan Fan, Xi Yang, Chi-Man Pun, Huaqi Zhang, and Xiaodong Cun. Gsfixer: Improving 3d gaussian splatting with reference-guided video diffusion priors. arXiv preprint arXiv:2508.09667, 2025. 2, 3
- [60] Mark YU, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. arXiv preprint arXiv:2503.05638, 2025. 3
- [61] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 3
- [62] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447–19456,

2024. 2

- [63] David Junhao Zhang, Roni Paiss, Shiran Zada, Nikhil Karnad, David E Jacobs, Yael Pritch, Inbar Mosseri, Mike Zheng Shou, Neal Wadhwa, and Nataniel Ruiz. Recapture: Generative video camera controls for user-provided videos using masked video fine-tuning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2050– 2062, 2025. 2, 3
- [64] Guosheng Zhao, Chaojun Ni, Xiaofeng Wang, Zheng Zhu, Xueyang Zhang, Yida Wang, Guan Huang, Xinze Chen, Boyuan Wang, Youyi Zhang, et al. Drivedreamer4d: World models are effective data machines for 4d driving scene representation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12015–12026, 2025. 2
- [65] Jingqiu Zhou, Lue Fan, Linjiang Huang, Xiaoyu Shi, Si Liu, Zhaoxiang Zhang, and Hongsheng Li. Flexdrive: Toward trajectory flexibility in driving scene gaussian splatting reconstruction and rendering. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1549– 1558, 2025. 2
- [66] Zixin Zou, Weihao Cheng, Yan-Pei Cao, Shi-Sheng Huang, Ying Shan, and Song-Hai Zhang. Sparse3d: Distilling multiview-consistent diffusion for object reconstruction from sparse views. In Proceedings of the AAAI conference on artificial intelligence, pages 7900–7908, 2024. 2

### A. More Details of Data Construction

We train 3D Gaussian Splatting (3DGS) on approximately 1.6K scenes from the official Waymo Open Dataset (WOD) [40] v1.4.3-train and NuScenes [7] v1.0-train. During training, we employ DriveStudio as the training framework to reconstruct scenes over their full temporal duration. For simplicity, we do not incorporate the SMPL model [29] for additional human body pose modeling, and we adopt default settings for hyperparameters such as the learning rate. For each 3DGS scene, we save 3DGS models at four training stages: after 100, 500, 1,000, and 30,000 iterations. The first three iterations are used to render underfitted videos along the original trajectory, serving as camera structural conditions during training. The final iteration (at 30,000 iterations) is used to render videos along offset trajectories, which act as source inputs during training. Visualizations of some training pairs are shown in Fig. 10.

Dataset Sequence Indices

022 025 031 034 049 053 084 086 089 096 210 323 350 382 403 410 430 534 570 640

NuScenes

Table 7. Selected 20 evaluation sequences from NuScenes. Each row contains 4 sequence indices.

### C. More Comparisons

StreetCrafter is trained on Vista [12], while FreeVS is built upon Stable Video Diffusion (SVD) [6]. The frame lengths of their generated videos are summarized in Table 8. We use the officially released rollout inference code to generate full-length sequences on WOD. However, because both models generate only short temporal windows, the rollout procedure often introduces background inconsistencies and noticeable temporal jitter. For a clearer assessment, please refer to the video results on our project page.

### D. More Qualitative Results

More qualitative results are shown in Fig. 11 and Fig. 12, where our method maintains geometric consistency and visual fidelity while accurately generating novel trajectories. For clearer and more intuitive comparisons, we recommend viewing the video results on our project page.

### B. Evaluation Dataset

We evaluate our method on 20 scenes selected respectively from WOD and NuScenes, whose sequence names are listed in Table 9 and Table 7, respectively. For each scene, we use the recorded trajectory video as the source trajectory input and the novel-trajectory 3DGS renderings as the cameracontrol conditions. Specifically, following the same training setup as in DriveStudio, we train a 3DGS representation for each scene for 30,000 iterations, and then render eight laterally offset novel trajectories (±1 m, ±2 m, ±3 m, and ±4 m) to serve as the structural camera conditions.

Method Base Model Frame Length

FreeVS SVD [6] 8 StreetCrafter Vista [12] 25 Ours Wan2.1 [45] 121

Table 8. Frame length comparison of different methods.

###### Dataset File Names

segment-10444454289801298640 4360 000 4380 000 with camera labels.tfrecord segment-10498013744573185290 1240 000 1260 000 with camera labels.tfrecord segment-10588771936253546636 2300 000 2320 000 with camera labels.tfrecord segment-10625026498155904401 200 000 220 000 with camera labels.tfrecord segment-10963653239323173269 1924 000 1944 000 with camera labels.tfrecord segment-11017034898130016754 697 830 717 830 with camera labels.tfrecord segment-11846396154240966170 3540 000 3560 000 with camera labels.tfrecord segment-1191788760630624072 3880 000 3900 000 with camera labels.tfrecord segment-11928449532664718059 1200 000 1220 000 with camera labels.tfrecord segment-12161824480686739258 1813 380 1833 380 with camera labels.tfrecord segment-16801666784196221098 2480 000 2500 000 with camera labels.tfrecord segment-18111897798871103675 320 000 340 000 with camera labels.tfrecord segment-6242822583398487496 73 000 93 000 with camera labels.tfrecord segment-1921439581405198744 1354 000 1374 000 with camera labels.tfrecord segment-2323851946122476774 7240 000 7260 000 with camera labels.tfrecord segment-1999080374382764042 7094 100 7114 100 with camera labels.tfrecord segment-4898453812993984151 199 000 219 000 with camera labels.tfrecord segment-4266984864799709257 720 000 740 000 with camera labels.tfrecord segment-175830748773502782 1580 000 1600 000 with camera labels.tfrecord segment-14561791273891593514 2558 030 2578 030 with camera labels.tfrecord

WOD

Table 9. Selected 20 evaluation sequences with official file names from the WOD dataset.

100 Iterations 500 Iterations 1000 Iterations

30000 Iterations

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

Loss

Or Or

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

Source Inputs Structural Camera Conditions Supervisions

Figure 10. Visualization of training pairs. During training, we use the 3DGS renderings of novel trajectories at 30,000 iterations as the source inputs, the 3DGS renderings of the original trajectory at 100, 500, or 1,000 iterations as the camera conditions, and the clean recorded videos of the original trajectory as the supervision.

time

- Left @ 2m

- Right @ 2m

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

Source Trajectory

Left @ 3m

- Right @ 3m

[Figure 275]

Source Trajectory

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

Figure 11. Qualitative results of our method on multiple novel-trajectory generations.

time

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

###### Source Trajectory

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

Novel-Trajectory 3DGS Rendering

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

Novel-Trajectory Generation

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

###### Source Trajectory

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

Novel-Trajectory 3DGS Rendering

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

Novel-Trajectory Generation

Figure 12. Qualitative novel-trajectory results and 3DGS rendering conditions, with the two scenes shifted right by 3 m and 4 m.

