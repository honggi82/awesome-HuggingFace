## Stereo World Model: Camera-Guided Stereo Video Generation

Yang-Tian Sun1 Zehuan Huang2∗ Yifan Niu2 Lin Ma3 Yan-Pei Cao2 Yuewen Ma3 Xiaojuan Qi1†

1 The University of Hong Kong 2 VAST 3 ByteDance Pico

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

# arXiv:2603.17375v1[cs.CV]18Mar2026

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

| | |
|---|---|
|[Figure 14]|[Figure 15]|

[Figure 16]

| | |
|---|---|
|[Figure 17]|[Figure 18]|

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

View Consistency Disparity

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Action Planning

[Figure 29]

[Figure 30]

VR/AR

|[Figure 31]|[Figure 32]|
|---|---|

|[Figure 33]|[Figure 34]|
|---|---|
| | |

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

“Put the rope in the black long box”

[Figure 42]

[Figure 43]

[Figure 44]

Figure 1. We introduce StereoWorld, a stereo world model capable of performing exploration based on given binocular images, generating view-consistent stereo videos with intrinsic geometric understanding. StereoWorld can be applied to downstream tasks like VR/AR visualization as well as action planning in embodied intelligence. Project: https://sunyangtian.github.io/StereoWorld-web/.

#### Abstract

correspondences with substantially lower compute. Across benchmarks, StereoWorld improves stereo consistency, disparity accuracy, and camera-motion fidelity over strong monocular-then-convert pipelines, achieving more than 3× faster generation with an additional 5% gain in viewpoint consistency. Beyond benchmarks, StereoWorld enables endto-end binocular VR rendering without depth estimation or inpainting, enhances embodied policy learning through metric-scale depth grounding, and is compatible with longvideo distillation for extended interactive stereo synthesis.

We present StereoWorld, a camera-conditioned stereo world model that jointly learns appearance and binocular geometry for end-to-end stereo video generation.Unlike monocular RGB or RGBD approaches, StereoWorld operates exclusively within the RGB modality, while simultaneously grounding geometry directly from disparity. To efficiently achieve consistent stereo generation, our approach introduces two key designs: (1) a unified camera-frame RoPE that augments latent tokens with camera-aware rotary positional encoding, enabling relative, view- and timeconsistent conditioning while preserving pretrained video priors via a stable attention initialization; and (2) a stereoaware attention decomposition that factors full 4D attention into 3D intra-view attention plus horizontal row attention, leveraging the epipolar prior to capture disparity-aligned

#### 1. Introduction

Learning a generative world model– i.e., predicting future observations conditioned on actions and camera motion– has become increasingly important for interactive perception and embodied intelligence. Modern world models [41, 50, 52, 81] predominantly use monocular video representations and achieve strong results in controllable video syn-

* Project Lead, †Corresponding Author

thesis. Yet monocular observations impose fundamental geometric limits: depth is implicit, scale is ambiguous, and geometric consistency must be inferred rather than observed, which accumulates 3D errors under long-horizon camera trajectories and constrains applications where accurate geometry is critical (e.g., embodied intelligence and navigation). RGB-D world models [10, 26] introduce an auxiliary depth channel, but predicted depth is scene-dependent and still scale-ambiguous, often requiring ad-hoc normalization and remaining unstable across domains [16].

In contrast, stereo vision – the dominant perceptual mechanism in many biological systems [22, 42]– provides direct, robust geometric cues to 3D scene structure. This motivates us to study a stereo world model that grounds geometry in binocular observations rather than inferring depth from monocular motion or relying on imperfect depth predictors (see Fig. 6). Compared to monocular world models, a stereo-conditioned system jointly learns the coupled evolution of appearance and geometry under camera motion and actions; compared to RGB-D systems, it avoids producing and stabilizing explicit metric depth maps while retaining strong geometric signals. The result is consistent, metric-scale perception well suited to VR/AR rendering and embodied navigation, as illustrated in Fig. 2.

Building a stereo world model remains non-trivial. First, the predictions must remain consistent across both binocular views and time while generalizing over varying intrinsics, extrinsics, and baselines– calling for a unified, view- and time-aware camera embedding. Ray-map concatenation [15, 52] encodes absolute coordinates tied to a specific frame, which can entangle viewpoint and scene layout and make relative cross-view generalization (across changing baselines or poses) harder; a relative camera formulation is preferable. Second, naive stereo extensions of monocular transformers incur prohibitive compute: selfattention scales quadratically with tokens, and full 4D spatiotemporal cross-view attention quickly becomes infeasible. Third, pretrained video diffusion backbones are highly sensitive to positional-encoding changes, so injecting viewcontrol signals risks wiping out learned priors.

To address these challenges, we introduce StereoWorld, the first camera-conditioned stereo world model. Our approach is built around two key designs. First, we propose a unified camera-frame RoPE strategy that expands the latent token space and augments it with camera-aware rotary positional encoding, enabling joint reasoning across time and binocular views while minimally modifying the pretrained backbone’s RoPE space. This formulation effectively encodes relative camera relationships, naturally supports scenes with varying intrinsics and baselines, while preserves pretrained video priors, facilitating stable and efficient adaptation to stereo video modeling. Second, we design a stereo-aware attention mechanism that decomposes

|World Model<br><br>Modality|Monocular,<br><br>RGB<br><br>Monocular,<br><br>RGBD|Stereoscopy,<br><br>RGB|
|---|---|---|
|Geometry<br><br>|[Figure 45]<br><br>Relative, Limited Range|Metric|
|Same Domain with<br><br>Pre-Trained Model|[Figure 46]<br><br>[Figure 47]|[Figure 48]|
|VR Display|Rely on External Depth Estimation and Inpainting Model|End-to-End|

[Figure 49]

[Figure 50]

| |
|---|

| |
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

Depth Estimation + Inpaint

[Figure 53]

[Figure 54]

| |
|---|

| |
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

Ours

Figure 2. World Model Comparison. StereoWorld incorporates metric-scale geometry, producing output modalities that are more compatible with pretrained models. Moreover, it can be applied end-to-end for VR visualization, ensuring better consistency of fine-grained details between the left and right views.

full 4D attention into 3D intra-view attention plus horizontal row attention, leveraging the epipolar prior that stereo correspondences concentrate along scanlines. This achieves strong stereo consistency while dramatically reducing computation. Together, these components allow StereoWorld to learn appearance and geometry jointly, delivering end-toend binocular video generation with accurate camera control and disparity-aligned 3D structure.

Experiments demonstrate that StereoWorld delivers significant improvements in stereo consistency (Fig. 4), disparity accuracy (Fig. 6), and camera motion fidelity (Fig. 5) over monocular world models. For instance, compared with the SOTA method augmented by post-hoc stereo conversion, our approach achieves a 3× improvement in generation speed, while also delivering an approximately 5% gain in viewpoint consistency (see Tab. 2). Beyond benchmarks, StereoWorld unlocks practical applications: (i) direct binocular VR rendering without depth estimation or inpainting pipelines (see Sec. 4.4.1); (ii) improved spatial awareness for embodied agents through metric-scale geometry grounding (see Sec. 4.4.2); and (iii) compatibility with long-range monocular video generation methods [27, 74] via distillation to support extended interactive stereo scene synthesis (see Sec. 4.4.3). To our knowledge, this is the first system to realize end-to-end, cameraconditioned stereo world modeling, opening a path toward geometry-aware generative world representations. Our contributions are summarized as follows:

• We introduce the first camera-conditioned stereo world model that jointly learns appearance and binocular geometry, producing view-consistent stereo videos under ex-

plicit camera trajectories or action controls.

- • We expand latent tokens with a camera-aware rotary positional encoding (without altering the backbone’s original RoPE), enabling relative, unified conditioning across time and binocular views while preserving pretrained video priors via a stable attention initialization.
- • We decompose full 4D spatiotemporal attention into 3D intra-view attention plus horizontal row attention for cross-view fusion, leveraging the epipolar prior to cut computation substantially while maintaining disparityaligned correspondence.
- • Our approach delivers superior quantitative and qualitative results. It enables end-to-end VR rendering with improved viewpoint consistency, provides potential geometry-grounded benefits for embodied policy learning, and extends naturally to long-video generation.

#### 2. Related Work

Camera-Controlled Video Generation. With advances in text-to-video models [6, 9, 13, 39, 71], recent work increasingly explores adding conditional signals for controllable generation [14, 17, 69, 72]. Among these, cameracontrolled video generation [2, 70, 79, 80] aims to explicitly regulate viewpoints via camera parameters. Notable methods include AnimateDiff [18], which uses motion LoRAs [23] to model camera motion; MotionCtrl [63], which injects 6DoF extrinsics into diffusion models; and CameraCtrl [19], which designs a dedicated camera encoder for improved control. CVD [33] extends control to multi-sequence settings through cross-video synchronization, while AC3D [1] systematically studies camera motion representations for better visual fidelity. Several trainingfree methods have also emerged [21, 24, 36], , further broadening the landscape of camera-controllable video synthesis. These methods pave the way for world modeling.

Stereo Video Generation. Recently, a growing number of studies [11, 47, 48, 51, 60, 76, 78] have focused on converting monocular videos into stereo videos. Most of these approaches rely on pre-existing depth estimation results, followed by warping and inpainting operations in the latent space. Some methods, like StereoDiffusion [60] and SVG [11] adopt a training-free paradigm, performing inpainting through optimization based on pretrained image or video diffusion priors. While works like StereoCrafter [78], SpatialMe [76], StereoConversion [38], ImmersePro [47] construct large-scale stereo video datasets to train feed-forward networks capable of directly completing the warped videos.

However, such approaches cannot be directly applied to explorable stereo world model generation. A straightforward solution might involve extending the outputs of a monocular world model using the aforementioned tech-

niques. Nonetheless, these methods depend heavily on video depth estimation and warping, making them non–endto-end, computationally inefficient, and susceptible to error accumulation—particularly in fine-detail regions (such as the wire fence illustrated in Fig 2).

Multi-View Video Generation. Multi-view generation has also emerged as a rapidly evolving research direction. CAT3D [15] enables novel view synthesis from singleor multi-view images by combining multi-view diffusion with NeRFs. SV4D [68] extend Stable Video Diffusion (SVD) [5] into Stable Video 4D (SV4D), which reconstructs a 4D scene from a single input video; however, their method is limited to a foreground animated object and does not model the background. Similar approaches, such as Generative Camera Dolly [56], CAT4D [65] and SynCamMaster [3], also explore view synthesis across large camera baselines. Nevertheless, these methods primarily target novel view generation and are not directly applicable to stereo video generation.

#### 3. Stereo World Model

Given a rectified stereo pair (Ileft,Iright) ∈ R3×H×W with baseline b and and a scene prompt c, our goal is to synthesize a stereo video conditioned on an action specified as a camera trajectory {camt} := {(Kt ∈ R3×3,Tt ∈ R4×4),t ∈ (1,2,··· ,N)} where K and T are the intrinsic and extrinsic respectively, and N denotes the number of actions. The generated sequences should (i) remain temporally smooth while following the prescribed camera motion, and (ii) be left-right consistent at every timestep. To this end, building upon a pre-trained video diffusion model (Sec. 3.1), we propose StereoWorld with two key components (Fig. 3): (a) a unified camera-frame positional embedding strategy that expands the backbone’s latent token space and augments it with camera-aware RoPE, minimally perturbing pretrained priors (Sec. 3.2); and (b) a stereoaware attention mechanism (Sec. 3.3) that decomposes cross-view fusion into 3D intra-view attention plus horizontal row attention, balancing computational efficiency with accurate epipolar (disparity-aligned) correspondence.

##### 3.1. Pre-trained Video Diffusion Model

Our work builds on a pretrained video diffusion model and repurposes it for stereo world modeling, enabling us to leverage the strong spatiotemporal priors and visual fidelity provided by large-scale video pretraining. Specifically, we adopt a latent diffusion model [7] consisting of a 3D Variational Autoencoder (VAE) [32] and a Transformer-based diffusion model (DiT) [44]. The VAE encoder E compresses the video (V ∈ RF×H×W×3) into a compact spatiotemporal latent representation:

z = E(V) ∈ Rf×h×w×c. (1)

Condition Frame Noisy Target Frame

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

| | | |
|---|---|---|
|𝑥1|𝑦1|𝑡1|
| | | |
| | | |

|𝑐1𝑙|
|---|
| |

| |
|---|

Frame 1

3D RoPE

| |
|---|

| | | |
|---|---|---|
|𝑥2|𝑦2|𝑡1|
| | | |
| | | |

|𝑐1𝑟|
|---|
| |

### …

[Figure 63]

[Figure 64]

Camera RoPE

|𝑐𝑁𝑙|
|---|
| |

| | | |
|---|---|---|
|𝑥1|𝑦1|𝑡𝑁|
| | | |
| | | |

| |
|---|

| | |
|---|---|

Frame 𝑁

| |
|---|

| | | |
|---|---|---|
| | | |
|𝑥2|𝑦2|𝑡𝑁|

|𝑐𝑁𝑟|
|---|
| |

Copy Init

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

…

Stereo View Video

Token Expansion

| |𝑅𝑏×2𝑓×ℎ×𝑤×𝑐。| |
|---|---|---|
| | | |

Unified Camera-Frame RoPE

| | |
|---|---|
||𝑅(𝑏×𝑓×ℎ)×2𝑤×𝑐。|
|---|
<br><br>|𝑅(𝑏×2)×(𝑓×ℎ×𝑤)×𝑐。| |
|---|---|
| | |
<br><br>Attn3D Attnrow| |

Stereo Attention Block × N

|𝑅𝑏×(2𝑓×ℎ×𝑤)×𝑐|
|---|

Attn

4D

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

…

| |𝑅𝑏×2𝑓×ℎ×𝑤×𝑐。| |
|---|---|---|
| | | |

Add

- Figure 3. Illustration of StereoWorld. Given a pair of stereo images and a conditional camera trajectory, StereoWorld first encodes conditional and noisy video latents from different viewpoints and timesteps using a unified camera–frame RoPE representation. It then performs denoising through a DiT equipped with stereo attention, ultimately producing the final stereo video.

The DiT is then trained in this latent space, progressively denoising noisy latent variables into video latents following the rectified flow formulation [12]. Once trained, the model can generate samples from pure noise via iterative denoising. After denoising, the VAE decoder D reconstructs the latents back into the pixel domain. In our stereo setting, a stereo video {Vleft,Vright} ∈ RF×H×W×3 is encoded in a viewpoint-agnostic manner using Eq. (1), producing latent representations {zleft,zright}.

Rotary Positional Encoding and Attention Vanilla RoPE [49] encodes relative positions by rotating the query and key vectors before dot-product attention. For a 1D sequence, the attention matrix is defined as:

At1,t2 = (qt1Rt1(d))(kt2Rt2(d))⊤ = qt1R∆t(d)k(t2), (2)

are the query and key embeddings at positions t1 and t2 and R∆

where ∆t = t1 − t2, qt

###### , kt

1

2

(d) is the relative rotation matrix acting on each 2D subspace of the d-dimensional embedding. The relative rotation matrix R∆

t

(d) = exp(∆tθni) ∈ Rd×d, where i is the imaginary unit, and θn is the frequency of rotation applied to a specific n-th pair of d dimensions (n = 0,...,d/2 − 1), enables the model to capture relative positional relationships directly within attention.

t

For video, recent RoPE variants (e.g., M-RoPE in Qwen2-VL [61]) preserve the inherent 3D structure by factorizing rotations along time and space. Let positions be (t,x,y). The attention term becomes:

1,x1,y1)R∆t,∆x,∆y(d)k⊤(t

###### A(t

1,x1,y1),(t2,x2,y2) = q(t

2,x2,y2), (3)

where ∆t = t1 − t2,∆x = x1 − x2,∆y = y1 − y2, and R∆t,∆x,∆y = R∆tR∆xR∆y. The rotations R∆t, R∆x, and R∆y act on disjoint 2D subspaces of the d-dimensional feature, so they commute and compose multiplicatively. In practice (e.g., Wan [58]-style implementations), the feature dimension d is partitioned evenly across t, x, and y, with independent 1D RoPEs applied per axis and then composed as above.

##### 3.2. Unified Camera-Frame RoPE

Fine-tuning a pretrained DiT video diffusion model into a stereo world model requires injecting camera conditioning – including stereo cameras with varying baselines and dynamic camera motions – while minimizing disruption to the pretrained prior.

A common approach concatenates Pl¨ukcer Ray encodings [77] onto the input feature channels. However, similar to early positional encoding methods [57], this approach relies on absolute coordinates, making it sensitive to the choice of reference frame. To mitigate this limitation, recent methods such as GTA [40] and PRoPE [35] model relative camera positions, yielding improved generalization. Specifically, PRoPE replaces R∆t,∆x,∆y in Eq. (3) with R∆∆camt,∆x,∆y, where

R∆∆camt,∆x,∆y(d) =Rcamt

t1,x1,y1(d)(Rcamt

t2,x2,y2(d))⊤, (4) Rcamt tj

1

2

Id/8 ⊗ Pj 0 0 Rt

, (5)

j,xj,yj(d) =

j,xj,yj(d/2)

Kj 0 0 1

Tj, Kj,Tj = camt

Pj =

.

j

Here j ∈ {1,2}, ⊗ is the Kronecker product, and Id/8 ∈

Rd/8×d/8 is the identity matrix. However, when fine-tuning a pretrained model (e.g., Wan [58]), directly modifying the original positional encoding with Eq. (5) can significantly disrupt the model’s learned prior, because the DiT’s attention weights, normalization statistics, and token bases are co-adapted to the original RoPE frequencies and axis partitioning.

To address this, we propose injecting camera positional encodings by expanding the token dimension, rather than altering the original encoding scheme. Concretely, we extend the original self-attention layer by increasing its feature dimension, i.e.

q˜(t,x,y) =

q(t,x,y) qcam(t,x,y) ∈ Rd+d

, (6)

c

Here dc is the expanded dimension for camera RoPE. The same expansion is also applied to k. Hence the rotary matrix in Eq. (5) can be extended to R(d+d

c)×(d+dc):

R∆t,∆x,∆y(d) 0 0 Id

###### R˜cam

t,x,y(d + dc) =

t

c/4 ⊗ Pt

leading to our unified camera-frame RoPE:

, (7)

R˜∆∆camt,∆x,∆y(d′) = R˜camt

t1,x1,y1(d′)(R˜camt

t2,x2,y2(d′))⊤, (8)

1

2

where d′ = d + dc. In this setup, the first d × d block of the matrix remains identical to that in Eq. (3), which aligns with the pretrained prior. For the newly added dc×dcblock, we experiment with two different initialization strategies for the expanded layer corresponding to qcam and kcam.

We experiment with two initialization schemes for the new subspace (qcam and kcam):

- • Zero Init ensures that the model’s initial output remains identical to that of the pretrained model. However, this initialization makes training more challenging, as the camera conditioning signal is difficult to activate effectively.
- • Copy Init initializes the new subspace with temporal attention weights. Since camera and temporal embeddings operate at the frame level, this provides a strong starting point while minimally affecting pretrained behavior.

In contrast to PRoPE [35], our unified camera–frame RoPE expands the token dimension rather than reparameterizing RoPE, preserving the pretrained positional subspace and adding an orthogonal, camera-conditioned channel. Empirically (Fig. 7), this yields more stable training, faster convergence.

##### 3.3. Stereo-Aware Attention

With the unified camera-frame representation, camera positional encodings for each viewpoint are injected into the stereo video latents zleft,zright, modeling relationships between arbitrary token pairs as

1,x1,y1)R˜∆∆camt,∆x,∆y(d′)k˜(t

2,x2,y2). This unified formulation allows our method to seamlessly accommodate multi stereo video datasets with varying baselines and intrinsic parameters, as demonstrated in Tab. 1.

q˜(t

With this representation, a naive stereo generator concatenates left–right tokens along the sequence dimension and applies full joint attention over features fin ∈ Rb×2f×h×w×c, yielding a 4D attention (Attn4D) that couples spatial, temporal, and viewpoint dependencies. However, because attention cost grows quadratically with the number of tokens, this approach is computationally prohibitive for video synthesis.

Observing that in rectified stereo pairs the epipolar lines align horizontally, we exploit this geometry to design a more efficient stereo-aware attention. The 4D attention is decomposed into: (a) intra-view 3D attentions (Attn3D) capturing spatial–temporal dynamics, and (b) cross-view attentions computed only among horizontally aligned tokens at the same timestep (Attnrow). As illustrated in Fig. 3, the final output aggregates both components:

fout = Attn3D(fin) + Attnrow(fin). (9)

With this design, the overall computational complexity is reduced from O((2f · h · w)2) to O(2 · (f · h · w)2) + f · h · (2w)2). We report a comparison of the performance differences between these two attention mechanisms in Tab. 5, which demonstrates the efficiency and effectiveness of the proposed decoupled attention scheme.

#### 4. Experiment

Table 1. Training Data Information.

|Dataset|Sample Num. Baseline / m Motion Domain|
|---|---|
|Stereo4D [29] TartanAir [62] TartanAirGround [43] DynamicReplica [30] VKitti [8]<br><br>|11718 0.063 Dynamic Realistic<br><br>6433 0.25 Static Synthetic 58168 0.25 Static Synthetic<br><br>1686 Varying Dynamic Synthetic 230 Varying Dynamic Synthetic|

##### 4.1. Implementation Details

We implement StereoWorld based on the video generation model Wan2.2-TI2V-5B [58]. The model is trained on a mixed dataset list in Tab. 1. Each video clip contains 49 frames, and is cropped and resized to 480×640 before feeding to the network. We train StereoWorld using AdamW optimizer [37] for 20k steps, with batch size of 24, on 24 NVIDIA H20 GPUS. The learning rate is set to 1e-4.

##### 4.2. Benchmark Datasets and Metric

Evaluation Datasets. We construct the evaluation set with 435 stereo images sampled from FoundationStereo [64](Synthetic), UnrealStereo4K [54](Synthetic), TartanAir Testset(Synthetic) and Middlebury [45](Realistic), covering both indoor and outdoor scenes, and versatile textures and various baselines.

Left View Right View Left View Right View Left View Right View

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

DeepVerse[10]ViewCrafter[75]Aether[52]SEVA[81]Ours

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

- Figure 4. Stereo video generation comparison with SOTA methods augmented by post-hoc stereo conversion. Our method directly generates stereo video in an end-to-end manner, enabling better preservation of inter-view detail consistency and tonal coherence.

For each stereo image, we use Qwen2.5-VL [53] to caption the scene and sample a random camera trajectory.

Evaluation Metrics. StereoWorld is evaluated on camera accuracy, left-right view synchronization, visual quality and FPS. For camera accuracy, we extract camera poses from the generated videos, computing both rotation and translation errors (RotErr and TransErr). View synchronization is measured using image matching technique GIM [46] to count the number of matching pixels exceeding a confidence threshold (Mat. Pix.). We further measure crossdomain alignment using the FVD-V score from SV4D [67] and the average CLIP similarity between corresponding source and target frames at each timestep, denoted CLIP-V [33]. For visual quality, we evaluate fidelity, text coherence, and temporal consistency using Fr´echet Image Distance (FID) [20], Fr´echet Video Distance (FVD) [55], CLIP-T, and CLIP-F, respectively, following [4]. We also benchmark our method using the standard VBench metrics [28].

##### 4.3. Stereo Video Comparison

Baselines. StereoWorld is the first stereo video generation model. To demonstrate the advantages of simultaneous stereo-view generation, we first use a series of state-of-the-

art camera-controlled video generation methods to obtain a monocular video, and then extend them into stereo videos using StereoCrafter [78]. StereoCrafter is a warp-inpainting video generation model. Therefore, for RGBD generation models [10, 26, 52], we directly use the generated depth to warp the video into another view; for RGB generation models [75, 81], we first use DepthCrafter [25] for video depth estimation, and then perform the warping. Compared to these multi-stage pipelines, StereoWorld achieves more efficient generation as an end-to-end model, as shown in the“FPS” column of Tab 2.

In addition, since the training data used by different models are not well aligned, we also trained a monocular version (“Ours Monocular”) of our method under the same settings as the stereo version for comparison, in order to better demonstrate the advantages brought by stereo generation.

###### 4.3.1. Stereo View Consistency

Fig. 4 presents a visual comparison between our method and the baseline approaches on the stereo video generation task. The comparison methods, which rely on additional depth estimation and view inpainting models, often suffer from misaligned details between the left and right views (e.g., the plants in the third column) or exhibit slight color inconsis-

- Table 2. Comparison of stereo video with SOTA methods on visual quality, camera accuracy, view synchronization and FPS.

|Method|Modality<br><br>|Visual Quality FID↓ FVD↓ CLIP-T↑ CLIP-F↑|Camera Accuarcy RotErr↓ TransErr↓<br><br>|View Synchronization Mat. Pix.(K)↑ FVD-V↓ CLIP-V↑<br><br>|FPS↑|
|---|---|---|---|---|---|
|Voyager [26] DeepVerse [10] Aether [52] SEVA [81] ViewCrafter [75]<br><br>|RGBD RGBD RGBD RGB RGB<br><br>|226.97 170.37 24.85 97.03 191.32 176.72 24.59 97.31 185.72 152.97 24.93 97.14 195.70 170.92 24.77 98.11 211.89 185.76 25.02 96.15<br><br>|1.34 0.25 1.51 0.16 1.50 0.13 1.09 0.51 1.24 0.20<br><br>|4.26 55.45 91.41<br><br>4.48 33.50 93.86 4.35 42.07 93.71<br><br>4.49 31.10 94.73 4.49 42.10 93.51<br><br><br>|0.03 0.35 0.11 0.10 0.13<br><br>|
|Ours Monocular Ours Stereo|RGB RGB<br><br>|126.83 96.87 24.97 97.12 111.36 83.04 25.74 97.55<br><br>|1.36 0.14 1.01 0.11<br><br>|✗ ✗ ✗ 4.56 22.00 97.50<br><br>|✗ 0.49<br><br>|

- Table 3. Comparison of stereo video on Vbench metrics.

Voyager (Depth Warp) DeepVerse (Discrete Action) Ours

|Method|Aesthetic Quality ↑<br><br>Imaging Quality ↑<br><br>Temporal Flickering ↑<br><br>Background Consistency ↑<br><br>|
|---|---|
|Voyager [26] DeepVerse [10] Aether [52] SEVA [81] ViewCrafter [75] Ours|38.23 59.32 94.55 92.81<br><br>38.71 60.11 94.52 92.61<br><br>39.02 60.26 93.63 92.46<br><br>40.60 64.28 93.49 93.01 40.31 61.90 90.63 91.45 44.27 66.51 93.63 92.42<br><br><br>|

|[Figure 99]|
|---|

[Figure 100]

|[Figure 101]|
|---|

[Figure 102]

[Figure 103]

|[Figure 104]|
|---|

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

tencies between the two views (e.g., the sky in the second column). In contrast, our method generates stereo videos end-to-end, effectively avoiding these artifacts and ensuring better view consistency. The results in the ”View Synchronization“ column of Tab 2 further validate this observation.

[Figure 111]

[Figure 112]

Predicted Camera Condition Camera

Figure 5. Visualization of camera trajectory comparison from methods with different camera conditioning types.

###### 4.3.2. Camera Trajectory

rectly. It is also worth noting that, unlike these comparison methods, our model is trained without any depth supervision, relying solely on binocular image signals.

Our method also achieves superior alignment between the generated results and the conditioned camera parameters. In contrast, warp-based world models [26] often suffer from inaccurate depth estimation or insufficient geometric cues when the viewpoint change is large, leading to misaligned camera conformity. Meanwhile, discrete actionbased world models [10] lack fine-grained camera control. Benefiting from the unified camera–frame RoPE, our approach effectively models relative relationships between cameras, enabling more precise and continuous camera control. We estimate the camera poses of the generated videos using VGGT [59] and compare them with the conditioned camera inputs to quantify accuracy. As shown in the “Camera Accuracy” column of Tab. 2, our method achieves the highest precision. Furthermore, Fig 5 visualizes the camera trajectory comparisons, clearly illustrating that our model better preserves the intended camera motion.

##### 4.4. Application

###### 4.4.1. Virtual Reality Display

Our method can be directly applied to VR/AR with professional head-mounted display devices. We visualize several red–blue anaglyph images in Fig. 1 as examples of the generated stereo outputs. In addition, we conducted tests on a VR headset and performed a user study, the results of which are provided in the supplementary materials.

###### 4.4.2. Embodided Scenarios

To demonstrate the potential of our approach, we further evaluated it in embodied scenarios. Specifically, we finetuned our model on the binocular robotic arm dataset from DROID [31]. The trained model can generate corresponding stereo manipulation videos conditioned on a given text prompt, while also accurately recovering metric-scale depth from the generated results. We illustrate the results in Fig. 1 and supplementary materials.

###### 4.3.3. Disparity

Fig. 6 compares the disparity maps generated by our method with those produced by other RGB-D world models. As shown, existing RGB-D approaches often exhibit artifacts where texture patterns from the RGB outputs are inadvertently transferred into the predicted disparity — for instance, in the third column of Voyager [26] and the first column of Aether [52]. In contrast, our method effectively mitigates this issue by generating stereo image pairs first and then estimating disparity [64] from them, leading to cleaner and more geometrically consistent results. Moreover, disparity in our setting can be transferred to metric depth di-

###### 4.4.3. Long Video Distillation

Our method can also serve as a bidirectional attention base model for stereo video generation in an interactive causal manner, similar to Self-Forcing [27]. Specifically, we distill the diffusion sampling process into four steps and convert the model into a causal attention mechanism while maintaining a key–value (KV) cache. The distilled model is ca-

Left View Disparity Left View Disparity Left View Disparity

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Voyager[26]Aether[52]Ours

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

Figure 6. Stereo disparity comparison. Notably, our approach does not rely on any depth supervision during training. Table 4. Ablation on camera injection strategies.

PRoPE Ours Zero Init Ours Copy Init

[Figure 131]

[Figure 132]

[Figure 133]

Visual Quality Camera Accuracy

Method

FID↓ FVD↓ CLIP-T↑ CLIP-F↓ RotErr↓ TransErr↑

Step=10kStep=0

Pl¨ukcer Ray [77] 142.46 130.39 24.90 95.65 1.52 0.21 PRoPE [35] 144.45 128.32 25.33 96.83 1.33 0.18 Ours Zero Init 131.07 96.62 25.49 97.21 1.81 0.24 Ours Copy Init 122.41 93.17 25.54 97.26 1.16 0.15

[Figure 134]

[Figure 135]

[Figure 136]

Table 5. Ablation on attention scheme.

|Method<br><br>|Visual Quality CLIP-T↑ CLIP-V↑<br><br>|View Synchronization Mat. Pix.(K)↑ CLIP-V↑|Efficiency FLOPs(×1010)↓ FPS↑<br><br>|
|---|---|---|---|
|4D Attn Stereo Attn<br><br>|25.74 97.55 25.43 97.05<br><br>|4.51 97.50<br>4.52 96.63<br>|3.11 0.34 1.56 0.49<br><br>|

RotErr: 1.58, TransErr: 0.23 RotErr: 2.21, TransErr: 0.29 RotErr: 1.09, TransErr: 0.12

pable of generating 10-second stereo videos, with the generation speed improved from 0.49 FPS to 5.6 FPS. Additional technical details regarding long-video distillation are provided in the supplementary materials.

##### 4.5. Ablation

Camera Injection. We compare different camera conditioning strategies on TartanAir dataset [62], reported in

- Tab. 4 and Fig. 7. Among them, Ours (Zero Init) preserves the pretrained model’s prior and achieves relatively high visual quality. However, because the weights are initialized to zeros, the learning of camera conditioning becomes more difficult, leading to lower camera accuracy. The Pl¨ukcer Ray [77] approach, which relies on absolute coordinates, shows limited generalization capability and suffers a performance drop. Compared with PRoPE [35] , our method better preserves the pretrained model prior, achieving superior results in both visual fidelity and camera conformity.

Attention Scheme. We also compare the impact of different attention mechanisms on the results. As shown in

- Tab. 5, although 4D Attention achieves slightly better visual quality, the performance of Stereo Attention is largely comparable—and even surpasses it in terms of view consistency. Meanwhile, FLOPs and FPS are improved by approximately 50%, demonstrating the efficiency of our de-

Figure 7. Comparison of different camera-condition strategies.

sign. The detailed FLOPs calculations are provided in the supplementary materials.

#### 5. Conclusion and Discussion

This paper presents StereoWorld, a camera-conditioned stereo vision model that jointly modeling binocular visual appearance, while supporting explicit geometry grounding. By employing a unified camera-frame Rotary Position Embedding (RoPE), the model encodes relative camera parameters effectively, with minimal interference to pre-trained priors. Furthermore, we introduce a stereo-aware attention mechanism that exploits the inherent horizontal epipolar constraints in stereo videos to reduce computational complexity. Experimental results demonstrate that StereoWorld achieves more efficient and view-consistent outcomes in stereo video generation, with strong potential for downstream applications such as virtual reality, embodied AI, and long-horizon video synthesis.

Despite these advances, stereo video generation remains more computationally demanding than its monocular counterparts, and the scarcity of large-scale stereo datasets further limits model scalability. We provide a detailed discussion of these limitations and potential future research directions in the supplementary materials.

#### References

- [1] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. arXiv preprint arXiv:2411.18673, 2024. 3
- [2] Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky, Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, et al. Vd3d: Taming large video diffusion transformers for 3d camera control. arXiv preprint arXiv:2407.12781, 2024. 3
- [3] Jianhong Bai, Menghan Xia, Xintao Wang, Ziyang Yuan, Xiao Fu, Zuozhu Liu, Haoji Hu, Pengfei Wan, and Di Zhang. Syncammaster: Synchronizing multi-camera video generation from diverse viewpoints. arXiv preprint arXiv:2412.07760, 2024. 3
- [4] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, and Di Zhang. Recammaster: Cameracontrolled generative rendering from a single video. ArXiv, abs/2503.11647, 2025. 6
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [6] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [7] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575, 2023. 3
- [8] Yohann Cabon, Naila Murray, and M. Humenberger. Virtual kitti 2. ArXiv, abs/2001.10773, 2020. 5
- [9] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7310– 7320, 2024. 3
- [10] Junyi Chen, Haoyi Zhu, Xianglong He, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Zhoujie Fu, Jiangmiao Pang, et al. Deepverse: 4d autoregressive video generation as a world model. arXiv preprint arXiv:2506.01103, 2025. 2, 6, 7
- [11] Peng Dai, Feitong Tan, Qiangeng Xu, David Futschik, Ruofei Du, Sean Fanello, Xiaojuan Qi, and Yinda Zhang. Svg: 3d stereoscopic video generation via denoising frame matrix. arXiv preprint arXiv:2407.00367, 2024. 3
- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik

- Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 4
- [13] Weijie Kong et. al. Hunyuanvideo: A systematic framework for large video generative models, 2024. 3
- [14] Xiao Fu, Xian Liu, Xintao Wang, Sida Peng, Menghan Xia, Xiaoyu Shi, Ziyang Yuan, Pengfei Wan, Di Zhang, and Dahua Lin. 3dtrajmaster: Mastering 3d trajectory for multi-entity motion in video generation. arXiv preprint arXiv:2412.07759, 2024. 3
- [15] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024. 2, 3
- [16] Vitor Guizilini, Igor Vasiljevic, Dian Chen, Rares, Ambrus,, and Adrien Gaidon. Towards zero-shot scale-aware monocular depth estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9233–9243,

2023. 2

- [17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. arXiv preprint arXiv:2311.16933, 2023. 3
- [18] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 3
- [19] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. ArXiv, abs/2404.02101, 2024. 3
- [20] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6
- [21] Chen Hou, Guoqiang Wei, Yan Zeng, and Zhibo Chen. Training-free camera control for video generation. arXiv preprint arXiv:2406.10126, 2024. 3
- [22] Ian P Howard and Brian J Rogers. Perceiving in depth, volume 2: Stereoscopic vision. Number 29. OUP USA, 2012. 2
- [23] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3
- [24] Teng Hu, Jiangning Zhang, Ran Yi, Yating Wang, Hongrui Huang, Jieyu Weng, Yabiao Wang, and Lizhuang Ma. Motionmaster: Training-free camera motion transfer for video generation. arXiv preprint arXiv:2404.15789, 2024. 3
- [25] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2005–2015, 2024. 6

- [26] Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li, Rynson W. H. Lau, Wangmeng Zuo, and Chunchao Guo. Voyager: Long-range and world-consistent video diffusion for explorable 3d scene generation. ArXiv, abs/2506.04225,

2025. 2, 6, 7, 8

- [27] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 2, 7, 3, 4
- [28] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 6
- [29] Linyi Jin, Richard Tucker, Zhengqi Li, David Fouhey, Noah Snavely, and Aleksander Holynski. Stereo4d: Learning how things move in 3d from internet stereo videos. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10497–10509, 2024. 5, 1
- [30] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. CVPR, 2023. 5
- [31] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, Vitor Guizilini, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Muhammad Zubair Irshad, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Mart´ın-Mart´ın, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. Droid: A large-scale in-the-wild robot manipulation dataset.

2024. 7, 1

- [32] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3
- [33] Zhengfei Kuang, Shengqu Cai, Hao He, Yinghao Xu, Hong-

- sheng Li, Leonidas Guibas, and Gordon Wetzstein. Collaborative video diffusion: Consistent multi-video generation with camera control. arXiv preprint arXiv:2405.17414, 2024. 3, 6
- [34] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. arXiv preprint arXiv:2506.18903, 2025. 4
- [35] Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. Advances in Neural Information Processing Systems, 2025. 4, 5, 8
- [36] Pengyang Ling, Jiazi Bu, Pan Zhang, Xiaoyi Dong, Yuhang Zang, Tong Wu, Huaian Chen, Jiaqi Wang, and Yi Jin. Motionclone: Training-free motion cloning for controllable video generation. arXiv preprint arXiv:2406.05338, 2024. 3
- [37] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2017. 5
- [38] Lukas Mehl, Andr´es Bruhn, Markus Gross, and Christopher Schroers. Stereo conversion with disparity-aware warping, compositing and inpainting. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4260–4269, 2024. 3
- [39] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, Tsai-Shien Chen, Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. Snap video: Scaled spatiotemporal transformers for text-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7038– 7048, 2024. 3
- [40] Takeru Miyato, Bernhard Jaeger, Max Welling, and Andreas Geiger. Gta: A geometry-aware attention mechanism for multi-view transformers. arXiv preprint arXiv:2310.10375,

2023. 4

- [41] Thomas M¨uller, Alexander Keller, Sanja Fidler, Xuanchi Ren, Jiahui Huang, Merlin Nimier-David, Huan Ling, Yifan Lu, Jun Gao, and Tianchang Shen. Gen3c: 3d-informed world-consistent video generation with precise camera control, 2025. 1
- [42] Vivek Nityananda and Jenny CA Read. Stereopsis in animals: evolution, function and mechanisms. Journal of Experimental Biology, 220(14):2502–2512, 2017. 2
- [43] Manthan Patel, Fan Yang, Yuheng Qiu, Cesar Cadena, Sebastian Scherer, Marco Hutter, and Wenshan Wang. Tartanground: A large-scale dataset for ground robot perception and navigation. arXiv preprint arXiv:2505.10696, 2025. 5, 1
- [44] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 3

- [45] Daniel Scharstein, Heiko Hirschm¨uller, York Kitajima, Greg Krathwohl, Nera Nesic, Xi Wang, and Porter Westling. High-resolution stereo datasets with subpixel-accurate ground truth. In German Conference on Pattern Recognition,

2014. 5, 1

- [46] Xuelun Shen, Zhipeng Cai, Wei Yin, Matthias M¨uller, Zijun Li, Kaixuan Wang, Xiaozhi Chen, and Cheng Wang. Gim: Learning generalizable image matcher from internet videos. In The Twelfth International Conference on Learning Representations, 2024. 6
- [47] Jian Shi, Zhenyu Li, and Peter Wonka. Immersepro: Endto-end stereo video synthesis via implicit disparity learning. arXiv preprint arXiv:2410.00262, 2024. 3
- [48] Jian Shi, Qian Wang, Zhenyu Li, Ramzi Idoughi, and Peter Wonka. Stereocrafter-zero: Zero-shot stereo video generation with noisy restart. arXiv preprint arXiv:2411.14295,

- 2024. 3

[49] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

- 2024. 4

- [50] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. In International Conference on Computer Vision (ICCV), 2025. 1
- [51] Yang-Tian Sun, Yi-Hua Huang, Lin Ma, Xiaoyang Lyu, YanPei Cao, and Xiaojuan Qi. Splatter a video: Video gaussian representation for versatile processing. arXiv preprint arXiv:2406.13870, 2024. 3
- [52] Aether Team, Haoyi Zhu, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Chunhua Shen, Jiangmiao Pang, and Tong He. Aether: Geometricaware unified world modeling. ArXiv, abs/2503.18945,

2025. 1, 2, 6, 7, 8

- [53] Qwen Team. Qwen2.5-vl, 2025. 6
- [54] Fabio Tosi, Yiyi Liao, Carolin Schmitt, and Andreas Geiger. Smd-nets: Stereo mixture density networks. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8938–8948, 2021. 5, 1
- [55] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. Openreview, 2019. 6
- [56] Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, and Carl Vondrick. Generative camera dolly: Extreme monocular dynamic novel view synthesis. In European Conference on Computer Vision, pages 313–331. Springer, 2024. 3
- [57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 4
- [58] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang,

- Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 4, 5
- [59] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 7
- [60] Lezhong Wang, Jeppe Revall Frisvad, Mark Bo Jensen, and Siavash Arjomand Bigdeli. Stereodiffusion: Training-free stereo image generation using latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7416–7425, 2024. 3
- [61] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 4
- [62] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian A. Scherer. Tartanair: A dataset to push the limits of visual slam. 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909–4916,

2020. 5, 8

- [63] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 3
- [64] Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, and Stan Birchfield. Foundationstereo: Zeroshot stereo matching. CVPR, 2025. 5, 7
- [65] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26057–26068, 2025. 3
- [66] Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory, 2025. 4
- [67] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470, 2024. 6
- [68] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470, 2024. 3
- [69] Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Yong Zhang, Yingqing He, Hanyuan Liu, Haoxin Chen, Xiaodong Cun, Xintao Wang, et al. Make-your-video: Customized video generation using textual and structural guid-

- ance. IEEE Transactions on Visualization and Computer Graphics, 2024. 3
- [70] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with userdirected camera movement and object motion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 3
- [71] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 3
- [72] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023. 3
- [73] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024. 3
- [74] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025. 2, 3
- [75] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 6, 7
- [76] Jiale Zhang, Qianxi Jia, Yang Liu, Wei Zhang, Wei Wei, and Xin Tian. Spatialme: Stereo video conversion using depth-warping and blend-inpainting. arXiv preprint arXiv:2412.11512, 2024. 3
- [77] Jason Y Zhang, Amy Lin, Moneish Kumar, Tzu-Hsuan Yang, Deva Ramanan, and Shubham Tulsiani. Cameras as rays: Pose estimation via ray diffusion. arXiv preprint arXiv:2402.14817, 2024. 4, 8
- [78] Sijie Zhao, Wenbo Hu, Xiaodong Cun, Yong Zhang, Xiaoyu Li, Zhe Kong, Xiangjun Gao, Muyao Niu, and Ying Shan. Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos. arXiv preprint arXiv:2409.07447, 2024. 3, 6
- [79] Guangcong Zheng, Teng Li, Rui Jiang, Yehao Lu, Tao Wu, and Xi Li. Cami2v: Camera-controlled image-to-video diffusion model. arXiv preprint arXiv:2410.15957, 2024. 3
- [80] Sixiao Zheng, Zimian Peng, Yanpeng Zhou, Yi Zhu, Hang Xu, Xiangru Huang, and Yanwei Fu. Vidcraft3: Camera, object, and lighting control for image-to-video generation. arXiv preprint arXiv:2502.07531, 2025. 3
- [81] Jensen Zhou, Hang Gao, Vikram S. Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. Stable virtual camera: Generative view synthesis with diffusion models. ArXiv, abs/2503.14489, 2025. 1, 6, 7

## Stereo World Model: Camera-Guided Stereo Video Generation Supplementary Material

#### S1. Experiment

##### S1.1. Dataset Construction

The datasets used for training are summarized in Tab.1 of the main paper. For Stereo4D [29], we filtered out videos in which the camera remained static, exhibited minimal motion, or suffered from excessive jitter, as such samples are unsuitable for camera-conditioned training. Each video was divided into 49-frame clips, which were then cropped and resized to a uniform resolution of 480 × 640. For each clip, we used the left-eye video to generate caption annotations. All training data were accompanied by metric-scale camera parameters.

For the test set, we selected approximately 280 video clips from the processed TartanAirGround [43] video clips, sampled at intervals of 200. In addition, we used the UnrealStereo4K [54] and Middlebury [45] stereo image datasets, for which we generated a set of random camera trajectories to conduct out-of-domain evaluations (approximately 160 clips). Each camera trajectory was composed of both translation and rotation components. The translation sampling range along the z-axis was [−20m,−4m] ∪ [4m,20m], and the rotation sampling range around the y-axis was [−150◦,−50◦] ∪ [50◦,150◦].

##### S1.2. Stereo Attention FLOPs

For each attention head, let L be the sequence length or number of query tokens, d be the head dimension, a vanilla full attention head costs:

FLOPsfull = 4L2d. (10)

In our experiment, the input feature has the shape f ∈ Rb×2f×h×w×c. As for 4D Attention, L = 2f × h × w, we have

= 16bf2h2w2d. (11) While for the stereo attention, we have

FLOPsAttn

4D

FLOPsAttn

3D

= 8bf2h2w2d, (12)

FLOPsAttn

row

= 4bfhw2d. (13)

Supposing we use b = 1, f = 13, h = 15, w = 20, d = 128, we can calculate that FLOPsAttn

4D ≈

###### 3.115 × 1010, while in comparison, the stereo attention

costs FLOPsAttn

= 1.561 × 1010. Hence the stereo attention block reduces multiply-adds by a factor about 2×.

+ FLOPsAttn

3D

row

K

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

No Attention

| | |
|---|---|
| | |

Q

| | |
|---|---|
| | |

Clean KV

| | |
|---|---|
| | |

View1 View2

| | |
|---|---|
| | |

Causal DiT Attention Mask

Denoising

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |

…

Self Forcing / AR Inference

Figure 8. Attention mask configuration in distillation process.

#### S2. Application

##### S2.1. VR/AR Display

The binocular videos generated by StereoWorld can be directly utilized in VR/AR applications to deliver immersive experiences. In Fig. 9, we provide additional generated scene examples, together with anaglyph image to demonstrate the diversity and practicality of our approach. We also report the user study results in Fig. 10, in which we compare our results with baselines in terms of “Camera Conformity”, “Temporal Consistency”, “Image Quality” and “Overall Experience”.

##### S2.2. Embodided Scenarios

By fine-tuning our model on binocular robotic arm datasets [31], our approach can also be applied to embodied scenarios for stereo video generation, supporting downstream tasks such as action planning. As shown in Fig. 11, given an action command and the initial stereo frame, our model is able to generate the corresponding subsequent motion sequence. The results demonstrate that the generated videos not only follow the specified action instructions but also maintain high stereo consistency between the left and right views. We further performed disparity estimation on the generated outputs to verify their geometric plausibility and assess their feasibility for action planning.

##### S2.3. Long Video Distillation

Our trained model employs a bidirectional attention mechanism, which limits it to relatively short video sequences (49 frames in our setting). In contrast, autoregressive video gen-

Left View Right View Anaglyph Right View Left View Anaglyph

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

t3t1t2

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

Left View Right View Anaglyph Right View Left View Anaglyph

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

t3t1t2

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

Left View Right View Anaglyph Right View Left View Anaglyph

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

t3t1t2

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

Figure 9. More StereoWorld Results with Anaglyph.

Ours

(b)

Voyager Deepverse Aether Seva ViewCrafter

| |0.2<br><br>0.4<br><br>0.6<br><br>0.8<br><br>1.0<br><br>|
|---|---|
| | |

(c)

(a)

(d)

Figure 10. The summary of quantitative feedback in the user study. (a) Camera Conformity (b) Temporal Consistency (c) Image Quality (d) Overall.

eration models [27, 74] can effectively overcome this limitation and improve efficiency through a rolling KV-cache mechanism. Inspired by these advancements, we further distill StereoWorld into an autoregressive binocular video generation model, enabling long-horizon video synthesis and improving generation speed.

Following Self-Forcing [27], we adopt a two-stage paradigm. In the first stage (ODE distillation), we replace the bidirectional attention with a causal attention mechanism and distill the denoising process into four steps. The attention mask is illustrated in Fig 8, which generates two views at one step. In the second stage [27], we condition each pair of stereo frame’s (or chunks in practice) generation on previously self-generated outputs by performing autoregressive rollout with KV-cache. In this stage, a distribution matching distillation [73] (DMD loss) is applied to address exposure bias via distribution matching. Unlike monocular autoregressive video generation, our method simultaneously synthesizes binocular views and incorporates camera pose–aware positional encoding. As a result, the KV-cache must be updated with two separate sets of keys and values at each step – one for the left-view tokens and one for the right-view tokens, each containing our Unified Camera-Frame RoPE.

The distilled model achieves a significant improvement in binocular video generation speed, increasing from 0.49 FPS to 5 FPS, and is no longer limited to generating video clips of 49 frames. We present the results of long-video distillation in Fig 12, and in the supplementary video materials.

However, we observe that as the video length increases, the generated results still exhibit noticeable degradation. This issue is also present in prior works such as SelfForcing. Improving the stability of long-horizon video gen-

Left View Right View Disparity

- t3t1t2

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

“pick up the cup”

- t3t2t1

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

“put the lid on the teapot”

Figure 11. Stereo Video Generation on Embodided Scenarios.

eration therefore remains an open challenge shared by both monocular and stereo video synthesis.

#### S3. Monocular & Stereo Generation Comparison.

“Ours Monocular” and “Ours Stereo” in Tab 2 employ the exact same parameter count and compute budget. The superior FID for “Ours Stereo” is because binocular views provide a physical “anchor”. As demonstrated below (Fig 13) monocular pipelines relies on a single condition frame and often hallucinate unrealistic structures due to occlusion, whereas stereo settings incorporates additional view and better maintains alignment with real scene by stereo-aware attention.

Left View Right View Left View Right View

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

frame192frame40frame80frame160frame120frame1

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

Figure 12. Long Video Distillation Results.

Left View Input Right View Input Monocular Output Stereo Output (Left)

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 237]|
|---|

|[Figure 238]|
|---|

|[Figure 239]|
|---|

|[Figure 240]|
|---|

Mono/Stereo Comparison

Figure 13. Monocular and stereo generation comparison.

#### S4. Large & Varying Baselines.

To evaluate the model’s performance under varying baselines, we construct a camera trajectory by expanding the right camera baseline from 0.25m to 0.75m– well beyond the training distribution (0.063m-0.25m). As illustrated below (Fig 14), StereoWorld maintains geometric plausibility and achieves accurate metric-scale recovery up to 0.42m, outperforming SOTA like DepthAnything V2. This confirms our Unified Camera-Frame RoPE performs genuine geometric reasoning rather than simple image stretching, also demonstrating robust generalization to unseen camera trajectories and baseline configurations.

#### S5. Discussion

Our method currently does not incorporate any explicit constraints on scene-level consistency. Although it handles most cases well, certain examples may still exhibit spatial inconsistencies across video frames, as illustrated in Fig 15. This issue may be alleviated by introducing a spatial mem-

Left View Right View Disparity Disparity Align Metrics on OOD Baselines

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

|Depth AnythingV2|
|---|

[Figure 248]

[Figure 249]

[Figure 250]

Figure 14. Effect of different baselines on StereoWorld.

t1 t2 t3

[Figure 251]

[Figure 252]

[Figure 253]

RightViewLeftView

[Figure 254]

[Figure 255]

[Figure 256]

Figure 15. Failure Case. Note that the blue road sign does not exist at the beginning of the sequence; however, as the viewpoint advances, it gradually emerges and increases in size.

ory mechanism [34, 66]. Since stereo video generation inherently provides geometric information about the scene, our approach can be naturally integrated with methods such as VMem [34] or SPMem [66], replacing their additional reconstruction modules maintaining consistency through a dedicated spatial memory.

Failure Case: Text Rendering

Frame 0 Frame 10

[Figure 257]

[Figure 258]

| |
|---|

| |
|---|

|[Figure 259]|
|---|

|[Figure 260]<br><br>and|
|---|

We also note that our method predominantly generates static scenes. This is primarily due to the limited availability of binocular video data for training stereo models. Most of our training corpus consists of static, rendered scenes, which restricts the model’s ability to synthesize dynamic environments. Exploring strategies for collecting more dynamic stereo video data, or leveraging richer monocular dynamic video datasets, represents a highly promising direction for future work. Scaling the training to substantially larger datasets may also help mitigate the aforementioned consistency issues.

Moreover, since the stereo world model generates binocular videos simultaneously, it inherently models fewer frames compared to monocular methods. Although distillation into autoregressive frameworks enables the generation of longer videos, we still observe noticeable degradation in the later stages of video generation, similar as reported in self-forcing [27]. Developing approaches to robustly distill stereo video models into long-term video generators will therefore be a key focus for our future work.

