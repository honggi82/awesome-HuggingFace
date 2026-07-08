## MVTrack4Gen: Multi-View Point Tracking as Geometric Supervision for 4D Video Generation

#### JoungBin Lee1 Jaewoo Jung1 Jongmin Lee1 Tongmin Kim1 Hyunsung Kim1 Takuya Narihira2 Kazumi Fukuda2 Jahyeok Koo1 Jisang Han1 Yuki Mitsufuji2,3† Seungryong Kim1† 1KAIST AI 2Sony AI 3Sony Group Corporation Project Page: https://cvlab-kaist.github.io/MVTrack4Gen/

# arXiv:2606.26087v1[cs.CV]24Jun2026

Jointly Generating Novel-View and Multi-View Point Tracking

[Figure 1]

[Figure 2]

|[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]|
|---|

[Figure 6]

Reference

Video Generated

###### MVTrack4Gen

Reference Video with Query Point

|[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]|
|---|

Video

[Figure 10]

[Figure 11]

[Figure 12]

Lifting Both Reference and Generated Videos into 3D

Camera Trajectory

Figure 1: MVTrack4Gen jointly generates a novel-view video and multi-view point tracks, given a monocular reference video with query points and a user-specified camera trajectory. Lifting both the reference and generated frames into 3D space using Depth Anything 3 [30] shows that the target views and multi-view point tracks faithfully preserve the dynamic motion of the reference video while remaining geometrically consistent.

### Abstract

Synthesizing a novel-view video from a monocular reference video along a target camera trajectory requires both geometric consistency and motion fidelity with respect to the reference video. Existing methods based on explicit 3D representations are limited by the accuracy of off-the-shelf reconstruction modules, which often produce inaccurate geometry for dynamic objects in monocular videos. In contrast, camera-conditioning-only methods can achieve high visual quality but often struggle to preserve geometric and motion consistency. In this work, we introduce MVTrack4Gen (Multi-View point Tracking for Novel-View Generation), a motion-aware training framework that leverages multi-view point tracking as an

†Corresponding authors.

Preprint.

additional geometric and motion supervision signal for camera-conditioning-only novel-view video diffusion models. Our key finding is that specific attention layers encode strong correspondence cues, where query features attend to key features at geometrically corresponding locations across views and over time, and the misalignment of these correspondences causes motion inconsistency. Based on this observation, we route these features into an auxiliary multi-view tracking head and jointly train the diffusion model with a point-tracking objective. By explicitly strengthening these motion-aware correspondences, MVTrack4Gen improves existing models to better follow the motion in the reference view and maintain crossview geometric consistency. Across diverse benchmarks, our method achieves state-of-the-art geometric consistency and competitive camera accuracy.

### 1 Introduction

Novel-view video generation from a monocular video aims to synthesize a target-view video along a user-specified camera trajectory, with applications in virtual cinematography, robotics, and immersive AR/VR. To be useful in these real-world scenarios, the generated video should satisfy four requirements: (1) accurate camera control that faithfully follows the user-specified trajectory, (2) geometric consistency in which the scene structure of the reference video is preserved across the synthesized view, (3) motion consistency that maintains the dynamics of the reference video, and (4) a photorealistic visual appearance. The central challenge is achieving all four at once, since the model must respect the underlying 3D geometry and motion of the scene while still producing high-quality videos.

Recent works leverage generative priors from pretrained video diffusion models [1, 5, 27, 40, 44] to address this task, achieving notable progress in dynamic scenes [36, 29, 45, 10, 43, 3, 33, 19]. These approaches largely fall into two paradigms. The first paradigm follows a reconstruct-then-generate pipeline [36, 29, 45, 10, 43], which first reconstructs an explicit 3D representation from the reference video, projects this geometry along the target camera trajectory, and feeds the projected appearance as a spatial condition. By supplying explicit geometric guidance, this design relieves the model from having to jointly infer geometry and synthesize photorealistic frames. However, its quality hinges on the accuracy of the reconstructed geometry, which becomes challenging for dynamic scenes, where inaccurate dynamic-object geometry introduces distortions and flying-pixel artifacts near object boundaries. These artifacts corrupt the spatial condition and prevent the diffusion model from faithfully capturing the geometry and motion of dynamic objects.

To avoid this dependence on error-prone reconstruction, a second and more recent line of work performs camera-conditioning-only generation [3, 33, 19], without any explicit 3D representation. These methods feed the reference video to the diffusion model as additional input tokens and inject the target trajectory through camera embeddings such as Plücker coordinates [37], and can be trained from pairs of multi-view videos. Because the reference and target views are processed jointly within the same 3D attention modules, cross-view and intra-view information is exchanged implicitly at every layer. This implicit design yields markedly more photorealistic results, but the lack of explicit geometric grounding leaves the model with a weak understanding of scene structure. As a result, dynamic objects are often placed at incorrect locations or assigned inconsistent motions, producing cross-view geometric and motion inconsistencies.

In this work, we start from the observation that camera-conditioning-only methods already excel at photorealistic synthesis, and ask whether their geometric understanding can be improved without reintroducing explicit 3D conditioning. To this end, we analyze the 3D attention maps of these diffusion models [3, 33] across layers and denoising timesteps, where information is implicitly exchanged across views and frames. Inspired by [32], which reveals emergent correspondence in the attention maps of video diffusion models, our analysis yields three key observations. First, querykey matching within the 3D attention blocks provides clear correspondence cues, capturing both intra-video temporal correspondences within each view and inter-video cross-view correspondences between the generated and reference views. Second, these temporal and cross-view correspondences become simultaneously prominent at specific intermediate layers, revealing which regions the model attends to when synthesizing each part of the generated frame. Third, in regions where dynamic objects exhibit geometric or motion inconsistencies, the attention maps at these layers exhibit incorrect

cross-view correspondences—indicating that the quality of these correspondences directly governs the geometric consistency of the output.

Building on this insight, we explore whether geometric and motion consistency can be improved by directly supervising the correspondences in these dominant attention layers. Since the motion of dynamic objects can be described by the trajectories of physical points over time [13], we introduce MVTrack4Gen (Multi-View point Tracking for Novel-View Generation), a framework that leverages ground-truth multi-view point tracks as auxiliary supervision, where each track follows the same physical point within and across views.

Specifically, we build a multi-view tracking head on top of local 4D correlation volumes computed from the query and key features of the selected attention layer, and jointly train it with the diffusion model. This encourages the model to encode motion-aware correspondences in its attention features, so that the target view more faithfully reflects the motion of the reference video. In addition, we introduce a multi-view correspondence loss that applies a cross-entropy objective directly to the attention map, encouraging each query token in the target view to attend to its corresponding ground-truth location. Together, we observe that these objectives improve both cross-view geometric consistency and intra-view temporal consistency of the generated novel-view videos.

To validate its generality, we apply our method to two camera-conditioning-only backbones, ReCamMaster [3] and Redirector [33], and evaluate on the DAVIS [34] and iPhone [15] benchmarks. Our method consistently improves both backbones, achieving the best scores on most VBench [20] visual-quality metrics while reaching state-of-the-art geometric consistency and camera accuracy comparable to both reconstruction-based and camera-conditioning-only baselines. Notably, it markedly improves geometric consistency and visual quality for dynamic objects, without requiring any explicit 3D reconstruction at inference time.

### 2 Related Work

Novel-View Generation Using Explicit 3D Representation. Recent video diffusion models [7, 44, 40] have demonstrated strong generation capability, ensuring visually plausible results. One line reconstructs explicit 3D representations from the input video and conditions a video diffusion model to synthesize newly visible regions from the target camera viewpoint. Geometry-guided approaches [45, 36, 18, 41, 46, 22, 23, 10, 8] first estimate intermediate geometric cues from the input video, such as depth maps, point cloud, or 3D representations, and use them as spatial conditions to guide the synthesis of disoccluded target-view regions. TrajectoryCrafter [45] warps point clouds, GEN3C [36] operates in latent space to generate novel views, and ChronosObserver [41] synchronizes multi-view diffusion sampling through a hyperspace representation to produce time-synchronized, 3D-consistent multi-view videos. More recently, PostCam [10] fuses pose and visual signals through cross-attention shared by queries, while Infinite-Homography [26] conditions the generation of homography transformations as a lightweight geometric proxy that avoids explicit depth estimation. Richer representations have also been explored, including pseudo-4D Gaussian fields from dense point tracking [6] and feed-forward 4DGS reconstructors [43]. Despite their strong spatial grounding, these approaches remain bottlenecked by the accuracy and completeness of off-the-shelf reconstruction models.

Novel-View Generation Using Camera Conditioning Only. An orthogonal line of work bypasses explicit geometry entirely, relying on camera pose conditioning and the generative prior of video diffusion models to implicitly reason about 3D structure [39, 3, 4, 33, 14, 38]. GCD [39] is one of the earliest works to introduce camera-controlled dynamic novel-view generation by training a video generation model on Kubric [16], a multi-view synthetic video dataset. To further improve generation quality, ReCamMaster [3] is trained on a realistic synthetic dataset with more diverse camera trajectories. Redirector [33] designs an additional camera rotational embedding for more accurate camera-controllable video generation. A related line pursues joint space–time controllability: CAT4D [42] trains a multi-view video diffusion model to synthesize novel-views at arbitrary camera poses and timestamps, enabling 4D reconstruction via deformable 3D Gaussians, while SpaceTimePilot [19] jointly conditions on camera trajectory and time to achieve generative rendering of dynamic scenes across both space and time.

Point Tracking. Tracking Any Point (TAP) formulates long-term pixel-level correspondence as a generalization of optical flow that explicitly handles occlusion [17, 12]. Early works such as PIPs [17] and TAP-Net [12] suffered from occlusion fragility or per-frame independence, which TAPIR [13] addressed by combining global matching-based initialization with a temporal refinement stage. CoTracker [25] further demonstrated that tracking points jointly via a transformer with cross-track attention substantially improves robustness under occlusion and fast motion, and CoTracker3 [24] unifies these ideas under a simpler architecture trained with pseudo-labels.

In the multi-view setting, MV-TAP [28] aggregates spatio-temporal information across views via cross-view attention for robust 2D trajectory estimation, while MVTracker [35] targets feed-forward

- 3D point tracking by fusing multi-view features. These formulations yield geometrically consistent correspondences beyond what monocular trackers can recover, motivating their use as an auxiliary supervision signal to strengthen the geometric feature learning of video diffusion models.

### 3 Preliminaries: Video Diffusion Transformer for Novel-View Generation

Here, we explain the details of camera-conditioning-only frameworks for novel-view video generation, which aim to synthesize a video from a novel camera viewpoint given a monocular reference video. A typical video generation model for novel-view synthesis consists of two main components: VAE and Diffusion Transformer (DiT). Given a reference video Xref ∈ RF×H×W×3 where F, H, W, and 3 denote the number of frames, spatial height, width, and RGB channels, respectively, the VAE encodes it into a latent representation zref ∈ Rf×h×w×d

video where (f,h,w,dvideo) are the temporal, spatial, and channel dimensions in the latent space. The DiT then denoises a target-view latent ztgt ∈ Rf×h×w×d

video conditioned on zref to generate the novel-view video.

Specifically, the VAE downsamples the input by 4× temporally and 16× spatially. The DiT vθ takes the concatenation of zref and a noisy ztgt as input, and predict the velocity field:

vˆ = vθ([zref, ztgt ], t, c, camtgt), (1)

where [·, ·] denotes concatenation along the token dimension, t is the flow matching timestep, c is the text caption, and camtgt is the target camera trajectory. The model is trained via flow matching [31].

#### 3D Attention Across Reference and Target Latent Tokens.

Since zref and ztgt are concatenated as input to DiT, they jointly participate in 3D attention. At each transformer layer l and flow matching timestep t, the i-th latent frame (i ∈ {1,...,f}) of each view, either reference or target, is projected into query and key matrices:

Qrefi , Kiref, Qtgti , Kitgt ∈ Rhw×d

, (2)

head

where dhead is the per-head channel dimension and we omit the attention head for brevity. The attention weight matrix Cv

1,v2 i,j , where each entry measures the similarity between the i-th frame of view v1 and the j-th frame in view v2, is computed as:

⊤ √dhead

Qv

i Kv

1

2

j

Cv

1,v2

, v1,v2 ∈ {ref, tgt}.

i,j = Softmax

(3)

- Fig. 2 illustrates the resulting attention map structure.

Q

Q

Q

Q

Q

Q

K K K K K K … …

|𝐶 ,    ,   |𝐶 ,    ,   |…|𝐶 ,    ,   |𝐶 ,    ,   |𝐶 ,    ,   |…|𝐶 ,    ,   |
|---|---|---|---|---|---|---|---|
|𝐶 ,    ,   |𝐶 ,    ,   <br><br>Intra-|…<br><br>Video|𝐶 ,    ,   |𝐶 ,    ,   |𝐶 ,    ,   <br><br>Inter-V|…<br><br>ideo|𝐶 ,    ,   |
|…<br><br>Tempo|…<br><br>ral Cor|…<br><br>respond|…<br><br>ence|…<br><br>Cross-V|…<br><br>iew Co|…<br><br>rrespon|…<br><br>dence|
|𝐶 ,    ,   |𝐶 ,    ,   |…|𝐶 ,    ,   |𝐶 ,    ,   |𝐶 ,    ,   |…|𝐶 ,    ,   |
|𝐶 ,    ,   |𝐶 ,    ,   |…|𝐶 ,    ,   |𝐶 ,    ,   |𝐶 ,    ,   |…|𝐶 ,    ,   |
|𝐶 ,    ,   |𝐶 ,    ,   <br><br>Inter-V|…<br><br>ideo|𝐶 ,    ,   |𝐶 ,    ,   |𝐶 ,    ,   <br><br>Intra-|…<br><br>Video|𝐶 ,    ,   |
|…<br><br>Cross-V|…<br><br>iew Co|…<br><br>rrespon|…<br><br>dence|…<br><br>Tempo|…<br><br>ral Cor|…<br><br>respond|…<br><br>ence|
|𝐶 ,    ,   |𝐶 ,    ,   |…|𝐶 ,    ,   |𝐶 ,    ,   |𝐶 ,    ,   |…|𝐶 ,    ,   |

……

……

……

……

……

……

……

Figure 2: Illustration of full 3D attention in video DiT for novelview generation. The attention jointly captures intra-video temporal and inter-video cross-view correspondences.

### 4 Analysis

Since camera-conditioning-only models [3, 33] implicitly exchange information through 3D attention both across the reference and target views and over time within each view, we analyze their attention maps to examine how geometry and motion are encoded inside the models. Following [32], we conduct our main analysis on ReCamMaster [3], and show that its 3D attention maps contain emergent

token-level correspondences across layers and denoising timesteps. By extracting matches from attention weights and comparing them with pseudo ground-truth point tracks, we identify three types of emergent correspondences: intra-video temporal correspondences in the reference view, intra-video temporal correspondences in the target view, and inter-video cross-view correspondences. The intra-video correspondences capture motion-related temporal consistency within each view, whereas the inter-video correspondences are responsible for geometric alignment across views. Our analysis is also applicable to other camera-conditioning-only architectures, as demonstrated by our analysis of Redirector [33] in Appendix A.3.

#### 4.1 Analysis Setup

: Query Point

[Figure 13]

[Figure 14]

ReCamMasterMVTrack4Gen(Ours)

We conduct our analysis on the MultiCamVideo dataset [3], which provides time-synchronized multiview recordings of dynamic scenes with groundtruth camera trajectories, naturally yielding paired reference–target videos. We sample 40 scenes and randomly select two views per scene as the reference and target.

[Figure 15]

[Figure 16]

To obtain pseudo ground-truth correspondences that span both temporal and cross-view axes, we use MVTAP [28], an off-the-shelf multi-view point tracker that jointly tracks points across reference and target video frames. This produces multi-view pseudo

Generated Frame Overlaid Attention Map on Reference Frame

Figure 3: Cross-View Attention Visualization. For the same query point in the generated frame, ReCamMaster attends to incorrect regions in the reference frame, whereas MVTrack4Gen localizes attention on the corresponding object, enabling more consistent motion across views.

ground-truth point tracks T = {pv,i GT} with visibility O = {ov,i GT}, where v and i index the view and frame, respectively. More details are in Appendix A.1.

#### 4.2 Evaluating Attention-based Correspondences

Correspondence in 3D attention map. We investigate token-level correspondences from the 3D attention map defined in Eq. 3. For a pair of latent frames, the attention matrix Cv

1,v2

i,j measures how each query features in the i-th frame of view v1 attends to key features in the j-th frame of view v2, where v1,v2 ∈ {ref,tgt}. Given a query feature in the i-th frame of view v1, we take the key feature with the highest attention weight in the j-th frame of view v2 as its forward match.

To enable a more accurate analysis, we apply a cycle-consistency check to filter out unreliable matches. A forward match is accepted as a reliable correspondence only if matching backward from the destination latent frame to the source latent frame returns to the original query token. Formal definitions are provided in Appendix A.2.

Matching accuracy and harmonic mean. We measure matching accuracy using the Percentage of Correct Keypoints (PCK), which counts a query point as a positive only if (i) it is reliable under the cycle-consistency check and (ii) its forward match coincides with the corresponding co-visible point in the pseudo ground-truth T on the latent grid. PCK is reported separately for intra-video temporal correspondences in the reference and target views, and for inter-video cross-view correspondences, averaged over all query points across latent frames and scenes. Beyond matching accuracy alone, we further analyze two attention-weight-based quantities to characterize not only whether the selected match is correct, but also how the attention distribution supports information exchange across layers and denoising timesteps: an attention score, which measures how much attention weight is assigned to the correct match, and a confidence score, which measures how sharply the attention distribution is localized. We then compute the harmonic mean of three normalized metrics—matching accuracy, attention score, and confidence score—to identify layers in which all three metrics are simultaneously high, thereby assessing how the correspondences emerging inside the attention layers contribute to generation; further details are provided in Appendix A.2.

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

(a) Intra-Video Temporal Correspondence in Reference View

(b) Intra-Video Temporal Correspondence in Target View

(c) Inter-Video Cross-View Correspondence

- Figure 4: Matching Accuracy and Harmonic Mean in ReCamMaster [3]. We visualize matching accuracy (top) and the harmonic mean of matching accuracy, attention score, and confidence score (bottom) across diffusion layers and denoising timesteps. Results are shown for intra-video temporal correspondence in the reference and target views in (a) and (b), respectively, and for inter-video cross-view correspondence in (c). The rightmost column plots the top three layers with the highest cross-view correspondence scores. This figure shows that accurate cross-view and temporal matching emerge at specific intermediate layers.

#### 4.3 Results

Fig. 4 shows the matching accuracy and harmonic mean by comparing correspondences from the attention maps and the pseudo ground-truth matches T across denoising timesteps and layers. We observe correspondence cues both within each view over time and across the reference and target views, which we refer to as intra-video temporal correspondences and inter-video cross-view correspondences, respectively.

We observe that specific intermediate layers exhibit strong matching performance for both intra-video temporal correspondences and inter-video cross-view correspondences, indicating that the same layer range supports motion tracking within each view and geometric alignment across views, as visualized in Fig. 4 (c), with a dominant peak around the 18th diffusion layer. The harmonic mean follows the same layer–timestep trend as the matching accuracy, suggesting that cross-view matching capability is encapsulated within only specific layers. Moreover, in failure cases of generation as visualized in

- Fig. 3, the attention map at the 18th layer fails to attend to the corresponding regions in the reference video. Together, these results motivate our design choice of explicitly reusing features from this layer range as a unified correspondence signal for both temporal and cross-view consistency.

### 5 Methodology

Given a reference video Xref captured from a known camera trajectory camref and a target camera trajectory camtgt, our goal is to synthesize a geometrically consistent target video Xtgt that faithfully depicts the same dynamic scene from the new viewpoint. We achieve this by jointly training a camera-controlled video diffusion model and a multi-view tracking module that shares the query and key features from the diffusion model’s 3D attention layers. We further directly supervise the 3D attention maps with a multi-view correspondence loss, guiding each query feature to attend to the correct corresponding region across views and time. Together, these objectives provide geometric and motion supervision to the generation process. An overview of our framework is illustrated in Fig. 5.

#### 5.1 Improved Camera Encoding

We adopt ReCamMaster [3] and Redirector [33] as our backbones, but condition on both the extrinsics and intrinsics of the reference and target views, rather than the target extrinsics alone. This richer conditioning improves geometric consistency in novel-view generation. Each camera is encoded

X

Tracks Visibilities

P V

[Figure 25]

Tokenize

VAE Decoder

Feed-Forward NN

[Figure 26]

|[Figure 27]|
|---|

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

DiT Block

Text Cross-Attention

MultiView Tracking Module

4D Correlation Volume in Reference View

∆P, ∆V, ∆C

[Figure 32]

|[Figure 33]|
|---|

[Figure 34]

[Figure 35]

Cam Patchify Encoder

Cam Encoder

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

3D Self-Attention

Add Noise

Plücker Ray

Plücker Ray

Tokenize

z z

4D Correlation Volume in Target View

Add Add

VAE Encoder

VAE Encoder

Tracks Visibilities

P V

cam z z cam

cam X X cam

× 𝑀 Iterations

(a) Overall DiT Architecture (c) Cross-View Tracking Module

(b) DiT Block

- Figure 5: Main architecture. We jointly train a camera-controlled DiT and a multi-view tracking module that shares query and key features from the input of the DiT’s 3D self-attention layers. From these shared features, the tracking module constructs intra-video temporal correlation for temporal consistency and inter-video cross-view correlation for geometric correspondence.

- as a dense Plücker ray map [37], in which every pixel is represented by a 6D ray derived from the extrinsics and intrinsics. Plücker maps from both views are injected into each DiT layer; additional details are provided in Appendix B.1.

#### 5.2 Multi-View Point Tracking as Geometric Supervision

Multi-Scale 4D Correlation Volume in Each View. We construct a multi-scale local 4D correlation volume [11] in each view, leveraging the query–key similarity in the 3D attention map. This is motivated by our analysis in Sec. 4, which shows that the intra-video temporal correspondences in the 3D attention map inherently encode motion across time. Local query and key features qiv and kjv are bilinearly sampled within a ∆-sized neighborhood centered at the query points pvi = (xvi ,yiv) of the i-th frame in each view v ∈ {ref, tgt} and the estimated points pˆvj = (ˆxvj,yˆjv) of the j-th frame. The local 4D correlation volume is then computed via softmax:

qiv kjv ⊤ √dhead ∈ R(2∆+1)

4

Corrvi,j = Softmax

, v ∈ {ref, tgt}. (4)

Each temporal correlation volume within each view, Corrref and Corrtgt, encodes the motion within its corresponding view and serves as the input to the multi-view tracking module. Query points are sampled among all video frames, since any point should be trackable and its motion detectable wherever it newly appears in the video. This encourages the query feature to be more similar to its corresponding ground-truth key feature, which we empirically find makes the generated motion more faithful to the reference. More details are described in Appendix B.2.

Multi-View Point Tracking Head. Built on the correlation volumes described above, we adopt a transformer-based multi-view tracking head following [28]. At each iteration, the tracking head Ψ, equipped with factorized temporal and multi-view attention, predicts residuals (∆P,∆V,∆C) = Ψ(G), where input tokens G are formed by concatenating current visibility V , confidence C, and local 4D correlation volume Corrvi,j in each view. Visibility and confidence are initialized to zero for both views, i.e., V = C = 0.

#### 5.3 Training Objectives

We supervise the correspondences in the 3D attention maps with two additional objectives: a multiview point-tracking loss Ltrack and a multi-view correspondence loss Lcorr. The tracking objective makes the model follow the dynamic motion of the reference video by jointly training the diffusion model with the multi-view point tracking head. The multi-view correspondence loss enforces geometric consistency—not only between the reference view and the target view, but also within the target view itself—by strengthening matching information directly into the 3D attention map to guide where the model should attend.

Table 1: Quantitative comparison on the DAVIS dataset [34]. The best score for each metric is in bold, and the colored numbers denote the change relative to each backbone, with (green) indicating a gain and (red) a loss.

Visual Quality↑ Geo. Consist. Camera Accuracy

Method

Subject Consistency

Background Consistency

Aesthetic Quality

Imaging Quality

Temporal Flickering

Motion Smoothness

MEt3R↓ MEt3Rdynamic ↓ mRotErr (◦)↓ mTransErr↓ mCamMC↓ Explicit 3D Lifting

GEN3C [36] 0.856 0.894 0.461 0.582 0.950 0.980 0.290 0.328 2.538 0.127 0.163 TrajectoryCrafter [45] 0.847 0.880 0.438 0.550 0.942 0.970 0.291 0.306 10.126 0.190 0.355 CogNVS [9] 0.811 0.862 0.417 0.530 0.959 0.978 0.333 0.346 10.439 0.228 0.400 NeoVerse [43] 0.858 0.878 0.446 0.591 0.954 0.983 0.302 0.323 4.705 0.159 0.228

Camera Conditioning Only

ReCamMaster [3] 0.904 0.907 0.502 0.652 0.962 0.985 0.337 0.369 3.660 0.113 0.169 Redirector [33] 0.897 0.911 0.506 0.680 0.952 0.985 0.318 0.395 1.714 0.086 0.109

MVTrack4Gen ReCamMaster 0.892(-.012) 0.909(+.002) 0.507(+.005) 0.685(+.033) 0.953(-.009) 0.984(-.001) 0.274(-.063) 0.287(-0.082) 1.858(-1.802) 0.100(-.013) 0.125(-.044) MVTrack4Gen Redirector 0.905(+.008) 0.919(+.008) 0.508(+.002) 0.687(+.007) 0.956(+.004) 0.986(+.001) 0.267(-.051) 0.349(-0.036) 1.718(+.004) 0.073(-.013) 0.097(-.012)

The total loss is Ltotal = Ldiff + λtrack Ltrack + λcorr Lcorr, where λtrack = λcorr = 0.01. The diffusion loss follows the rectified-flow formulation, Ldiff = w(t) vθ − (ϵ − x0) 2tgt, where w(t) is a timestepdependent weighting and the squared error is evaluated only on the target-view tokens. The tracking loss jointly supervises the multi-view point-tracking head and diffusion model with ground-truth trajectories and decomposes into Ltrack = λseq Lseq + λconf Lconf + λvis Lvis, where Lseq is a visibilityweighted Huber loss on the predicted 2D point coordinates, Lconf is a probabilistic confidence loss that calibrates the predicted per-point confidence against the regression error, and Lvis is a binary cross-entropy on the predicted visibility logits.

The multi-view correspondence loss Lcorr directly supervises the attention weight matrix defined in Eq. 3 with a cross-entropy objective. For each query point, the ground-truth multi-view point tracks provide its corresponding locations and visibility across all frames and views. Since each frame contains a single ground-truth correspondence for a given query point when the target point is visible, we formulate this as a single-label classification problem. Specifically, we supervise the attention weight matrix to assign the highest probability to the matching token in each visible reference or target frame. More details are provided in Appendix B.5.

### 6 Experiments

- 6.1 Experimental Setup

Implementation Details. We build on ReCamMaster [3] and Redirector [33] as our backbones, fine-tuning the 3D attention layers and camera encoder while keeping all other parameters frozen. The multi-view point tracking head estimates point tracks using query-key features from the 18th layer of the DiT. We train on 4 NVIDIA H100 GPUs using mixed-precision training and the AdamW optimizer with a learning rate of 1×10−4 for 13,000 iterations with a batch size of 16. All videos are used

- at 480×832 resolution with 81 frames for training.

Table 2: Quantitative comparison on the iPhone [15] dataset.

Visual Quality Geo. Consist.

Method

PSNR↑ SSIM↑ LPIPS↓ MEt3R↓ Explicit 3D Lifting

GEN3C [36] 10.419 0.270 0.699 0.382 TrajectoryCrafter [45] 10.271 0.276 0.766 0.370 CogNVS [9] 10.105 0.280 0.774 0.406 NeoVerse [43] 11.886 0.394 0.579 0.493

Camera Conditioning Only

ReCamMaster [3] 11.005 0.338 0.705 0.461 Redirector [33] 11.447 0.329 0.720 0.580

MVTrack4Gen ReCamMaster 11.521(+.516) 0.270(-.068) 0.640(-.065) 0.381(-.080) MVTrack4Gen Redirector 11.830(+.383) 0.283(-.046) 0.638(-.082) 0.397(-.183)

Datasets. We train our model on the combined Kubric and MultiCamVideo described in Sec. B.4. For evaluation, we adopt two benchmarks covering complementary regimes: (i) the DAVIS dataset [34] for in-the-wild monocular videos with diverse object and camera motion, and (ii) the iPhone dataset [15] for casual hand-held captures of dynamic scenes.

Baselines. We compare against two families of camera-controlled video generation methods. Explicit 3D Lifting approaches first reconstruct geometry and re-render under the target camera; in this category we include GEN3C [36], TrajectoryCrafter [45], CogNVS [9], and NeoVerse [43]. Camera Conditioning Only approaches rely solely on the diffusion prior together with camera conditioning, without explicit 3D reconstruction; here we compare with ReCamMaster [3] and Redirector [33].

[Figure 42]

MVTrack4GenReCamMaster

ReCamMaster

|[Figure 43]|
|---|
|[Figure 44]|

|[Figure 45]|
|---|
|[Figure 46]|

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Reference

Reference

video Generated

video Generated

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

video

video

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

MVTrack4GenRedirector

Redirector

[Figure 63]

[Figure 64]

[Figure 65]

|[Figure 66]|
|---|
|[Figure 67]|

|[Figure 68]|
|---|
|[Figure 69]|

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Reference

Reference

video Generated

video Generated

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

video

video

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

- Figure 6: Qualitative results on the DAVIS dataset [34]. We attach our MVTrack4Gen to two backbones, ReCamMaster [3] and Redirector [33], and compare against each baseline. For every example, we show the reference and generated frames, together with their point-cloud renderings reconstructed by Depth Anything3 [30] for the reference view and target view. By strengthening cross-view correspondences, MVTrack4Gen generates target frames that remain faithfully aligned with the reference view, even for dynamic objects, while preserving scene structure, appearance, and coherent motion across views.

Evaluation Metrics. We evaluate along three complementary axes: generation quality, geometric consistency, and camera accuracy. For generation quality, we report PSNR, SSIM, and LPIPS against ground-truth target views on the iPhone dataset [15]; on DAVIS, where ground-truth views are unavailable, we follow VBench [20] and report Subject Consistency, Background Consistency, Aesthetic Quality, Imaging Quality, Temporal Flickering, and Motion Smoothness. For geometric consistency, we report MEt3R [2], which measures multi-view geometric consistency via feedforward 3D reconstruction, and additionally report MEt3Rdynamic, which restricts this measure to dynamic regions. For camera accuracy, evaluated on DAVIS, we estimate camera trajectories from the generated videos using Depth Anything3 [30] and report mean rotation error (mRotErr), mean translation error (mTransErr), and mean camera-motion consistency (mCamMC) against the target trajectory.

#### 6.2 Quantitative Results

We compare MVTrack4Gen against state-of-the-art baselines from both the Explicit 3D Lifting and Camera Conditioning Only paradigms on the DAVIS [34] and iPhone [15] datasets, as summarized

amMaster

MVTrack4GenRedirector

MVTrack4GenReC

Gen3C TrajectoryCrafterNeoVerse

Redirector Reference CogNVS

amMaster

Target

ReC

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

- Figure 7: Qualitative results on the iPhone dataset [15]. We visualize novel-views generated from the reference video by each method, alongside the ground-truth target view. Our MVTrack4Gen, built on both ReCamMaster [3] and Redirector [33], produces geometrically consistent novel-views that faithfully preserve the scene structure and appearance of the reference, while prior methods exhibit viewpoint inaccuracies, geometric distortions, or texture degradation.

in Tab. 1 and Tab. 2. On DAVIS, MVTrack4Gen achieves the best overall performance across all three evaluation axes. For visual quality, it ranks first on Subject Consistency, Background Consistency, Aesthetic Quality, Imaging Quality, and Motion Smoothness. For geometric consistency, it attains the lowest MEt3R and MEt3Rdynamic, outperforming all baselines. For camera accuracy, it achieves the lowest mTransErr and mCamMC while remaining competitive on mRotErr. On iPhone, MVTrack4Gen consistently outperforms all Camera Conditioning Only baselines on PSNR, LPIPS, and MEt3R, demonstrating that strengthening cross-view correspondence is highly effective even without any explicit 3D representation. Although Explicit 3D Lifting methods attain higher pixel-level fidelity on iPhone owing to their direct reliance on reconstructed geometry, MVTrack4Gen achieves comparable or even better geometric consistency without using any explicit 3D structure at inference. This confirms that strengthening cross-view correspondence in a video diffusion model effectively bridges the gap between the two paradigms.

#### 6.3 Qualitative Results

- Fig. 6 presents qualitative comparisons on the DAVIS dataset, where we attach MVTrack4Gen to both ReCamMaster [3] and Redirector [33] backbones and show, for each method, the generated frames together with their point-cloud renderings reconstructed by DepthAnything3 [30] for the reference and target views. The baselines produce plausible-looking frames but lack geometric grounding under viewpoint change, particularly on dynamic objects: their reference and generated point clouds fail to align on the moving foreground, indicating inconsistent scene geometry across views. In contrast, MVTrack4Gen produces point clouds that align faithfully between the reference and target views even for dynamic objects, preserving scene structure and keeping the motion of moving foreground objects coherent across viewpoints.
- Fig. 7 further presents comparisons on challenging real-world scenes from the iPhone dataset. Camera Conditioning Only baselines such as ReCamMaster and Redirector synthesize plausible frames but exhibit geometric inconsistencies under viewpoint change, with foreground objects appearing at incorrect depths or undergoing non-rigid warping. Explicit 3D Lifting methods such as CogNVS, GEN3C, and TrajectoryCrafter preserve coarse scene structure but introduce noticeable artifacts in regions where depth estimation is unreliable, including severe distortions of dynamic foreground objects. In contrast, MVTrack4Gen synthesizes geometrically consistent novel-views that faithfully preserve both scene structure and appearance from the reference, even under significant viewpoint changes and complex object motion. These observations are consistent with our quantitative gains in geometric consistency (MEt3R) and camera accuracy, confirming that strengthening cross-view

- Table 3: Architecture ablation on the iPhone dataset [15]. We progressively add components to the ReCamMaster baseline to validate each design choice.

Visual Quality Geo. Consist. Camera Accuracy

Condition Type

PSNR↑ SSIM↑ LPIPS↓ MEt3R↓ mRotErr (◦)↓ mTransErr↓ mCamMC↓

- (i) Plücker Ray Camera Encoding 11.085 0.251 0.672 0.630 29.236 0.755 1.070

- (ii) (i) + Correspondence Loss 11.186 0.282 0.666 0.435 13.585 0.725 0.837

- (iii) (i) + Tracking Module 11.296 0.272 0.651 0.467 12.845 0.592 0.693

- (iv) (i) + Tracking Module + Correspondence Loss 11.389 0.267 0.645 0.416 11.694 0.579 0.669

correspondence in a video diffusion model leads to both visually and geometrically faithful novel-view synthesis. More generation results are provided in Sec. C.

#### 6.4 Ablation Studies

We conduct ablation studies on the iPhone dataset [15], progressively adding components to the ReCamMaster baseline; results are summarized in Tab. 3. All variants are fine-tuned for 10k iterations from ReCamMaster, with “+” indicating additions to (i). Configuration (ii) applies our multi-view correspondence loss on the 3D attention map. This attention-level supervision substantially improves geometric consistency and camera accuracy over (i) while leaving image quality largely unchanged, showing that supervising where each query attends strengthens cross-view geometric alignment. Configuration (iii) instead adds our multi-view tracking module, where query points are tracked within each view. By making the model follow the reference motion, it improves motion fidelity and generation quality. Configuration (iv) combines both: the tracking module improves motion fidelity while the correspondence loss enforces geometric consistency and camera accuracy. These two components are complementary, attaining the best visual quality, geometric consistency, and camera accuracy across all axes.

### 7 Conclusion

We introduce MVTrack4Gen, which strengthens both motion and geometric correspondence in a video diffusion model for dynamic novel-view generation. Our key insight is that correspondencespecialized attention layers naturally emerge in novel-view diffusion models, jointly encoding intravideo temporal and inter-video cross-view correspondences — a property largely overlooked in prior works. We exploit this property with two complementary objectives that share the DiT’s attention features: a multi-view point tracking objective that makes the generated view follow the reference motion, and a multi-view correspondence loss that enforces geometric consistency across and within views. Without any explicit 3D reconstruction at inference, our method achieves state-of-the-art geometric consistency over both reconstruction-based and reconstruction-free baselines.

### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [2] Mohammad Asim, Christopher Wewer, Thomas Wimmer, Bernt Schiele, and Jan Eric Lenssen. Met3r: Measuring multi-view consistency in generated images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6034–6044, 2025.
- [3] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14834–14844, 2025.
- [4] Jianhong Bai, Menghan Xia, Xintao Wang, Ziyang Yuan, Zuozhu Liu, Haoji Hu, Pengfei Wan, and Di Zhang. Syncammaster: Synchronizing multi-camera video generation from diverse viewpoints. In International Conference on Learning Representations, volume 2025, pages 58038–58060, 2025.
- [5] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. Lumiere: A space-time diffusion model for video generation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024.
- [6] Weikang Bian, Zhaoyang Huang, Xiaoyu Shi, Yijin Li, Fu-Yun Wang, and Hongsheng Li. Gs-dit: Advancing video generation with dynamic 3d gaussian fields through efficient dense 3d point tracking. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21717–21727, 2025.
- [7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [8] Chenjie Cao, Jingkai Zhou, Shikai Li, Jingyun Liang, Chaohui Yu, Fan Wang, Xiangyang Xue, and Yanwei Fu. Uni3c: Unifying precisely 3d-enhanced camera and human motion controls for video generation. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–12, 2025.
- [9] Kaihua Chen, Tarasha Khurana, and Deva Ramanan. Reconstruct, inpaint, test-time finetune: Dynamic novel-view synthesis from monocular videos. arXiv preprint arXiv:2507.12646, 2025.
- [10] Yipeng Chen, Zhichao Ye, Zhenzhou Fang, Xinyu Chen, Xiaoyu Zhang, Jialing Liu, Nan Wang, Haomin Liu, and Guofeng Zhang. Postcam: Camera-controllable novel-view video generation with query-shared cross-attention. arXiv preprint arXiv:2511.17185, 2025.
- [11] Seokju Cho, Jiahui Huang, Jisu Nam, Honggyu An, Seungryong Kim, and Joon-Young Lee. Local all-pair correspondence for point tracking. In European conference on computer vision, pages 306–325. Springer, 2024.
- [12] Carl Doersch, Ankush Gupta, Larisa Markeeva, Adria Recasens, Lucas Smaira, Yusuf Aytar, Joao Carreira, Andrew Zisserman, and Yi Yang. Tap-vid: A benchmark for tracking any point in a video. Advances in Neural Information Processing Systems, 35:13610–13626, 2022.
- [13] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. Tapir: Tracking any point with per-frame initialization and temporal refinement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10061–10072, 2023.
- [14] Xiang Fan, Sharath Girish, Vivek Ramanujan, Chaoyang Wang, Ashkan Mirzaei, Petr Sushko, Aliaksandr Siarohin, Sergey Tulyakov, and Ranjay Krishna. Omniview: An all-seeing diffusion model for 3d and 4d view synthesis. arXiv preprint arXiv:2512.10940, 2025.
- [15] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. Monocular dynamic view synthesis: A reality check. Advances in Neural Information Processing Systems, 35:33768–33780, 2022.
- [16] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al. Kubric: A scalable dataset generator. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3749–3761, 2022.
- [17] Adam W Harley, Zhaoyuan Fang, and Katerina Fragkiadaki. Particle video revisited: Tracking through occlusions using point trajectories. In European Conference on Computer Vision, pages 59–75. Springer, 2022.

- [18] Jiaxin Huang, Sheng Miao, BangBang Yang, Yuewen Ma, and Yiyi Liao. Vivid4d: Improving 4d reconstruction from monocular video by video inpainting, 2025.
- [19] Zhening Huang, Hyeonho Jeong, Xuelin Chen, Yulia Gryaditskaya, Tuanfeng Y Wang, Joan Lasenby, and Chun-Hao Huang. Spacetimepilot: Generative rendering of dynamic scenes across space and time. arXiv preprint arXiv:2512.25075, 2025.
- [20] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.
- [21] Peter J Huber. Robust estimation of a location parameter. In Breakthroughs in statistics: Methodology and distribution, pages 492–518. Springer, 1992.
- [22] Hyeonho Jeong, Suhyeon Lee, and Jong Chul Ye. Reangle-a-video: 4d video generation as video-tovideo translation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11164–11175, 2025.
- [23] Taewoong Kang, Kinam Kim, Dohyeon Kim, Minho Park, Junha Hyung, and Jaegul Choo. Egox: Egocentric video generation from a single exocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11116–11126, 2026.
- [24] Nikita Karaev, Yuri Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker3: Simpler and better point tracking by pseudo-labelling real videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6013–6022, 2025.
- [25] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In European conference on computer vision, pages 18–35. Springer, 2024.
- [26] Min-Jung Kim, Jeongho Kim, Hoiyeong Jin, Junha Hyung, and Jaegul Choo. Infinite-homography as robust conditioning for camera-controlled video generation. arXiv preprint arXiv:2512.17040, 2025.
- [27] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [28] Jahyeok Koo, Inès Hyeonsu Kim, Mungyeom Kim, Junghyun Park, Seohyeon Park, Jaeyeong Kim, Jung Yi, Seokju Cho, and Seungryong Kim. Mv-tap: Tracking any point in multi-view videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20932–20941, 2026.
- [29] JoungBin Lee, Jaewoo Jung, Jisang Han, Takuya Narihira, Kazumi Fukuda, Junyoung Seo, Sunghwan Hong, Yuki Mitsufuji, and Seungryong Kim. 3d scene prompting for scene-consistent camera-controllable video generation. arXiv preprint arXiv:2510.14945, 2025.
- [30] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.
- [31] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023.
- [32] Jisu Nam, Soowon Son, Dahyun Chung, Jiyoung Kim, Siyoon Jin, Junhwa Hur, and Seungryong Kim. Emergent temporal correspondences from video diffusion transformers, 2025.
- [33] Byeongjun Park, Byung-Hoon Kim, Hyungjin Chung, and Jong Chul Ye. Redirector: Creating any-length video retakes with rotary camera encoding. arXiv preprint arXiv:2511.19827, 2025.
- [34] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 724–732, 2016.
- [35] Frano Rajiˇc, Haofei Xu, Marko Mihajlovic, Siyuan Li, Irem Demir, Emircan Gündo˘gdu, Lei Ke, Sergey Prokudin, Marc Pollefeys, and Siyu Tang. Multi-view 3d point tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 59–68, 2025.

- [36] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6121–6132, 2025.
- [37] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. Advances in Neural Information Processing Systems, 34:19313–19325, 2021.
- [38] Basile Van Hoorick, Dian Chen, Shun Iwase, Pavel Tokmakov, Muhammad Zubair Irshad, Igor Vasiljevic, Swati Gupta, Fangzhou Cheng, Sergey Zakharov, and Vitor Campagnolo Guizilini. Anyview: Synthesizing any novel view in dynamic scenes. arXiv preprint arXiv:2601.16982, 2026.
- [39] Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, and Carl Vondrick. Generative camera dolly: Extreme monocular dynamic novel view synthesis. European Conference on Computer Vision (ECCV), 2024.
- [40] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models, 2025.
- [41] Qisen Wang, Yifan Zhao, Peisen Shen, Jialu Li, and Jia Li. Chronosobserver: Taming 4d world with hyperspace diffusion sampling. arXiv preprint arXiv:2512.01481, 2025.
- [42] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26057–26068, 2025.
- [43] Yuxue Yang, Lue Fan, Ziqi Shi, Junran Peng, Feng Wang, and Zhaoxiang Zhang. Neoverse: Enhancing 4d world model with in-the-wild monocular videos. arXiv preprint arXiv:2601.00393, 2026.
- [44] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, pages 83048–83077, 2025.
- [45] Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 100–111, 2025.
- [46] Jinjing Zhao, Fangyun Wei, Zhening Liu, Hongyang Zhang, Chang Xu, and Yan Lu. Spatia: Video generation with updatable spatial memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4245–4257, 2026.

### A Correspondence in 3D Attention Map

- A.1 Dataset for Analysis and Pseudo Ground-Truth Generation

MultiCamVideo Dataset. We conduct our analysis (Sec. 4) on the MultiCamVideo dataset [3], a multi-camera synchronized video dataset rendered via Unreal Engine 5 that features large camera movements. The dataset provides time-synchronized multi-view recordings of dynamic scenes together with diverse camera trajectories. Since each scene is captured by multiple cameras at the same timestamps, the dataset naturally offers paired reference–target videos suitable for evaluating both temporal and multi-view correspondences. From this dataset, we sample 40 scenes, and for each scene we randomly select two camera views where one serves as the reference view and the other as the target view for novel-view generation.

Pseudo Ground-Truth Multi-View Point Tracking via MV-TAP. To obtain dense point correspondences that span both the temporal axis within each video and the cross-view axis between the reference and target, we leverage MV-TAP [28], a recent multi-view point tracker that jointly reasons over synchronized views via cross-view attention. Concretely, we set query points on the first frame in a regular grid manner, and run MV-TAP jointly over the 10 synchronized view videos to estimate their correspondences across both time and viewpoints. Running MV-TAP on the reference–target pairs

from the MultiCamVideo dataset yields pseudo multi-view point tracks T = {pv,i GT} and visibility states O = {ov,i GT}, where pvi ∈ RN×2 denotes the 2D locations of N tracked points in view v at the i-th frame, and ovi ∈ {0,1}N indicates whether each point is visible. These tracks serve as our pseudo ground-truth, as shown in Fig. 9.

- A.2 Details of Attention-based Correspondence Evaluation

Forward Match via Mapping Operator. For a query point pv

i in the i-th latent frame of view v1, its forward match in the j-th latent frame of view v2 is obtained via argmax over the attention weight matrix Cv

1

1,v2

i,j in Eq. 3, which selects the spatial location with the highest attention score within the spatial domain Ω of the j-th latent. We define a mapping operator F that returns this forward match under a given attention weight matrix:

j = F Cv

1,v2

Cv

1,v2

pˆv

i,j , pv

i,j (pv

i , p), v1,v2 ∈ {ref,tgt}. (5)

i = argmax

2

1

1

p∈Ω

j = F(Cv

1,v2

With this mapping, the forward match from view v1 to view v2 is pˆv

i,j ,pv

i ), and the backward match from pˆv

2

1

j back to view v1 is obtained by applying F once more with the reverse attention weight matrix Cv

2

2,v1 j,i .

Reliable Correspondence via Cycle Consistency. To identify reliable correspondences in the attention map, we adopt a cycle-consistency check: a query and its forward match form a valid correspondence only when matching backward returns the original query. Formally, pv

i is regarded as reliable if

1

i ; Cv

1,v2

i,j = F Cv

2,v1

j,i , F(Cv

1,v2

i,j , pv

i ) − pv

Reliable pv

i 2 ≤ δ (6)

1

1

1

where δ denotes the cycle-consistency threshold on the pixel grid and we set δ = 16. This enforces a mutual best-match relation: both directions of attention should agree on the same point pair. Since the attention is computed in the latent space, the pseudo ground-truth tracks T are accordingly rescaled to the latent resolution.

Matching Accuracy via PCK. We measure matching accuracy using the Percentage of Correct Keypoints (PCK). Given the pseudo ground-truth tracks T rescaled to the latent grid, a query point pv

i is counted as a positive when both conditions hold: (i) it is reliable under Eq. 6, and (ii) its forward match pˆv

1

2,GT j

j = F(Cv

1,v2

i ) lies within δ pixels of the co-visible ground-truth match pv

i,j ,pv

2

1

on the latent grid. Formally,

1 |Q|

PCK =

pvi1∈Q

Reliable pv

1

2,GT

j − pv

i ; Cv

1,v2

i,j = 1 ∧ p ˆv

##### j 2 ≤ δ , (7)

2

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

(a) Intra-Video Temporal Correspondence in Reference View

(b) Intra-Video Temporal Correspondence in Target View

(c) Inter-Video Cross-View Correspondence

- Figure 8: Attention Score and Confidence Score in ReCamMaster [3]. We visualize the attention score (top) and confidence score (bottom) of query points across diffusion layers l and timesteps t for the three correspondence types: (a) intra-video temporal in the reference view, (b) intra-video temporal in the target view, and (c) inter-video cross-view.

where Q is the set of all query points whose ground-truth match is co-visible across the relevant frames. Matches are obtained on the latent grid, while distances are measured in the original pixel space with δ set to 16 pixels. PCK is computed separately for each of the three correspondence types and averaged across latent frames and scenes.

Attention Score. A single query point attends to tokens from both the reference and target views through the attention mechanism, allowing us to examine how its attention weights concentrate across layers and denoising timesteps. The resulting attention map encodes three types of correspondence: (a) intra-video temporal correspondence within the reference view, (b) intra-video temporal correspondence within the target view, and (c) inter-video cross-view correspondence between the two views. For each type, we visualize the attention weights of individual query points. As shown in the top row of Fig. 8, the temporal correspondences within each view, shown in (a) and (b), are emphasized in the early and final diffusion layers, whereas in the intermediate layers the attention shifts toward the cross-view correspondence in (c).

Confidence Score. The confidence score captures how sharply the correspondence is localized within the 3D attention map. For each query point, we re-normalize the attention over only the keys of the ground-truth frame and take the maximum logit value of the attention map, which approaches one when the distribution collapses onto a single key and stays low when it remains diffuse. As shown in the bottom row of Fig. 8, the temporal correspondences within each view, shown in (a) and (b), tend to become sharply localized in the later diffusion layers, whereas the cross-view correspondence in (c) exhibits higher confidence in the intermediate layers.

#### A.3 Generalization to Another Backbone

To verify that the correspondence-specialized layer structure is not an artifact of a single model, we repeat the same analysis on Redirector [33], a camera-controlled video diffusion backbone. We compute matching accuracy, attention score, and confidence score, and average the three into a harmonic mean, using identical settings to those used for ReCamMaster [3], and report them over various diffusion layers l and denoising timesteps t for the three correspondence types.

T = 1 T = 21 T = 41 T = 61 T = 81

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

View 1View 2View 3

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

- Figure 9: Pseudo Ground-Truth Multi-View Tracks on the MultiCamVideo Dataset [3]. We run MV-TAP [28] on the synchronized multi-view videos of the MultiCamVideo dataset to extract pseudo ground-truth multi-view point tracks. The tracked points are propagated consistently across both frames and views, providing intra-video temporal and inter-video cross-view correspondences that serve as supervision for training.

Accuracy and Harmonic Mean. As shown in Fig. 10, the matching accuracy (top) and the harmonic mean (bottom) on Redirector follow the same layer–timestep trend observed on ReCamMaster: intra-video temporal correspondences in both the reference and target views, shown in (a) and (b), peak in a similar intermediate layer range, whereas the inter-video cross-view correspondence in (c) is more sharply localized around a specific middle 18th layer. The selected-layer curves in the right column further show that matching becomes accurate as denoising progresses. This consistency indicates that the emergence of a correspondence-specialized layer is a property shared across camera-controlled video diffusion backbones rather than one tied to a particular model.

Attention Score. As shown in the top row of Fig. 11, Redirector exhibits a similar attention-routing pattern to ReCamMaster. Intra-video temporal correspondences in the reference and target views, shown in (a) and (b), receive stronger attention in the early and later diffusion layers, while inter-video cross-view correspondence in (c) becomes more prominent in the intermediate layers. This consistent trend suggests that camera-controlled video diffusion backbones tend to allocate intermediate layers to cross-view information exchange.

Confidence Score. The confidence maps in the bottom row of Fig. 11 further support this observation. Temporal correspondences become more sharply localized toward the later layers, whereas cross-view correspondences show higher confidence around the intermediate layer range. Together with the attention-score analysis, these results indicate that Redirector also develops a correspondencespecialized layer structure, supporting the generality of our choice to supervise attention features from this layer range.

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

(a) Intra-Video Temporal Correspondence in Reference View

(b) Intra-Video Temporal Correspondence in Target View

(c) Inter-Video Cross-View Correspondence

- Figure 10: Accuracy and Harmonic Mean in Redirector [33]. We visualize matching accuracy (top) and the harmonic mean of accuracy, confidence, and attention score (bottom) over various diffusion layers and denoising timesteps, for (a) intra-video temporal correspondence in the reference view, (b) intra-video temporal correspondence in the target view, and (c) inter-video cross-view correspondence. The right column shows selected layers from (c) plotted against timestep t. Consistent with our analysis on ReCamMaster, cross-view matching and temporal matching within the reference and target views emerge at specific middle layers and become stronger as denoising progresses, indicating that the correspondence-specialized layer structure is not specific to a single backbone.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

(a) Intra-Video Temporal Correspondence in Reference View

(b) Intra-Video Temporal Correspondence in Target View

(c) Inter-Video Cross-View Correspondence

[Figure 160]

- Figure 11: Attention Score and Confidence Score in Redirector [33]. We visualize the attention score (top) and confidence score (bottom) of query points across diffusion layers l and timesteps t, for (a) intra-video temporal correspondence in the reference view, (b) intra-video temporal correspondence in the target view, and (c) inter-video cross-view correspondence. As on ReCamMaster, the temporal correspondences (a, b) are emphasized in the early and final layers while the cross-view correspondence (c) concentrates in the intermediate layers, confirming that the same layer-wise behavior holds across backbones.

### B Model Architecture

#### B.1 Plücker Ray Camera Encoding

Plücker Ray Construction. We provide the formal construction of the Plücker ray map. For each pixel, the corresponding ray r ∈ R6 is defined as

d m

, where m = o × d, (8)

r =

with the ray direction d ∈ R3 and origin o ∈ R3 computed as

d = R⊤K−1x, o = −R⊤t, (9)

where x = (u,v,1)⊤ is the homogeneous pixel coordinate, R and t are the camera rotation and translation, and K is the intrinsic matrix. The direction d is normalized to unit length for scale invariance. Aggregating the rays over all pixels and frames yields the dense Plücker maps rref,rtgt ∈ R6×F×H×W.

Incorporating Plücker Maps into DiT Layers. The Plücker maps are injected into each DiT layer as

h′ = h + Proj CamEnc([rref,rtgt]) , (10) where h is the input token to the 3D attention at each layer, CamEnc is a lightweight convolutional encoder that maps the 6-channel Plücker rays to the token dimension, and Proj is a linear projection. We follow the same injection location as ReCamMaster [3].

#### B.2 Multi-Scale Local 4D Correlation

We construct a multi-scale local 4D correlation with varying receptive fields to capture multiscale query–key relationships from the 3D attention. To build a feature pyramid with S scales, we interpolate the DiT query and key features Qvi ,Kiv at each transformer layer to resolution

r×2s−1 × r×W2s−1 , where r = 4 denotes the model stride and s ∈ {1,2,3,4} denotes the scale factor:

H

sQvi = Interpolates(Qvi ), (11) sKiv = Interpolates(Kiv), v ∈ {ref,tgt}. (12)

At each scale s, we extract local features around points of interest via bilinear sampling. Local query features sqiv are sampled within a ∆-sized neighborhood centered at the query point pvi = (xvi ,yiv) in frame i of view v:

xvi 2s−1 + δx,

yiv 2s−1 + δy : ∥δ∥∞ ≤ ∆ , (13)

sqiv = sQvi

and local key features skjv are sampled around the estimated point pˆvj = (ˆxvj,yˆjv) across all frame indices j in view v:

x ˆvj 2s−1 + δx,

yˆjv

skjv = sKjv

2s−1 + δy : ∥δ∥∞ ≤ ∆ , (14) where δ ∈ Z2 and sqiv,skjv ∈ Rd×(2∆+1)

2

, with v ∈ {ref,tgt}. We then construct a local 4D

correlation volume s Corrvi,j between local features of frame i and frame j in view v via the softmax operation:

sqiv skjv ⊤ √dhead ∈ R(2∆+1)

4

s Corrvi,j = Softmax

, v ∈ {ref,tgt}. (15)

We concatenate the local 4D correlation volumes from all S scales along the channel dimension to obtain the multi-scale correlation descriptor Corrvi,j ∈ R4(2∆+1)

4

.

4

Corrvi,j = Concat 1 Corrvi,j, 2 Corrvi,j, 3 Corrvi,j, 4 Corrvi,j ∈ R4(2∆+1)

. (16)

- Table 4: Layer ablation on the MultiCamVideo [3] dataset. We vary the matching layer l that feeds the multi-view tracking head and the best value per metric is shown in bold.

Visual Quality Geo. Consist. Camera Accuracy

Matching Layer

PSNR↑ SSIM↑ LPIPS↓ MEt3R↓ mRotErr (◦)↓ mTransErr↓ mCamMC↓

l = 15 12.78 0.505 0.573 0.451 15.58 67.94 72.98 l = 18 14.08 0.559 0.496 0.394 4.55 20.31 22.35 l = 22 12.38 0.484 0.558 0.465 9.61 46.05 50.85 l = 24 9.99 0.422 0.708 0.469 25.71 74.61 85.57

#### B.3 Layer Ablation

Our analysis in Sec. 4 identifies the 18th DiT layer as the correspondence-specialized layer whose 3D attention most reliably encodes inter-video cross-view correspondence. This observation translates into the best downstream generation, i.e., that the layer carrying the strongest correspondence signal is also the most effective source of features for our multi-view tracking head. We ablate the matching layer l — the DiT layer whose query and key features are routed into the tracking head — across four layers spanning the network, and report generation quality, geometric consistency, and camera accuracy on the MultiCamVideo [3] evaluation set in Table 4. Layer 18 attains the best generation quality on PSNR, SSIM, and LPIPS, the lowest MEt3R, and the most accurate camera control on all three pose metrics, consistent with our analysis where cross-view correspondence peaks at this layer.

#### B.4 Training Data

We train on a hybrid dataset combining three complementary sources to maximize diversity in scene types, camera motions, and visual domains:

Kubric. The Kubric dataset provides synthetic multi-object scenes rendered with Blender, precise camera intrinsics, extrinsics, and dense point tracks with ground-truth visibility labels. Videos are center-cropped from 832×832 to 480×832 at 81 frames.

MultiCamVideo. The MultiCamVideo dataset provides cinematic multi-camera sequences rendered via Unreal Engine 5, featuring 10 synchronized camera viewpoints per scene with known calibration. Dense point tracks are computed on a 30×52 spatial grid (1,560 points per frame).

Reverse MultiCamVideo Augmentation. To improve robustness, we augment training by reversing the temporal order of the video sequences, which increases the diversity of both object motion and camera trajectories seen during training.

#### B.5 Training Objective Details Diffusion Loss. We adopt the rectified flow formulation,

Ldiff = w(t) vθ([zref, ztgt ],t,c,camref,camtgt) − (ϵ − x0) 2tgt, (17)

where vθ(·) is the predicted velocity field conditioned on the concatenated reference and target latents [zref,ztgt], the denoising timestep t ∼ U(0,1), the conditioning signal c, and the target camera camtgt. The training target is the rectified flow velocity ϵ − x0, where ϵ ∼ N(0,I) is the sampled noise and x0 denotes the clean target latent, with the noisy input constructed as xt = (1 − t)x0 + tϵ. The weighting w(t) is a timestep-dependent factor that balances the contribution of different noise levels during training. The squared error is evaluated only on the target-view tokens, as indicated by the ∥ · ∥2tgt notation; reference-view tokens serve purely as conditioning context and do not contribute to the diffusion loss.

Tracking Loss. The multi-view point tracking head [28] is supervised with ground-truth point trajectories using

Ltrack = λseq Lseq + λconf Lconf + λvis Lvis, (18)

with λseq = 0.05 and λconf = λvis = 1.0. Given the ground-truth tracks T = {pv,n,iGT} and visibility O = {ov,n,iGT ∈ {0,1}} of the N query points indexed by n over the i latent frames of both views

- v ∈ {ref,tgt}, together with the head’s predictions {pˆvn,i,cˆvn,i,oˆvn,i}, the three terms are defined as follows. The sequence loss Lseq is a visibility-weighted Huber loss [21] on the predicted 2D point coordinates,

1 v,n,i ov,n,iGT v,n,i

ov,n,iGT ρ p ˆvn,i − pv,n,iGT , (19)

Lseq =

where ρ(·) is the Huber function; this restricts coordinate regression to frames where the ground-truth point is visible and prevents occluded targets from injecting noisy gradients. The confidence loss Lconf is a probabilistic loss that calibrates the predicted per-point confidence cˆvn,i against the realized regression error: confidences are supervised to be high when the predicted location lies within a small radius of the ground-truth and low otherwise, encouraging the head to produce reliability estimates that downstream consumers can use to filter unreliable tracks.

Multi-View Correspondence Loss. While the tracking head is supervised only through the sampled local correlation volumes in each view, the attention weight matrix Cv

1,v2

i,j in Eq. 3 itself encodes three types of correspondence within the attention map: intra-video temporal correspondences within the reference view and within the target view, and inter-video cross-view correspondence between the two views. These correspondences act as a bridge that exchanges information both across views and across time, and visually failed regions in the generated output tend to coincide with correspondences that are misaligned in the attention map. We therefore supervise it directly with a cross-entropy objective derived from the multi-view point tracks T , so that each query point attends to the location where it physically reappears in the other frames and the other view. The key property we exploit is that the multi-view tracks specify the correct token exactly: for a query point pv

i in the i-th latent frame of view v1, its match in any other latent frame j of view v2 is either a single ground-truth location pv

1

2,GT

2,GT

j or occluded, as indicated by the visibility flag ov

j ∈ {0,1}.

We randomly sample a set of query points Q, and for each query we supervise its correspondence against every co-visible target token across all latent frames j of both the reference and target views v2 ∈ {ref,tgt}. Since each row Cv

1,v2

i,j (pv

i ,·) is already a per-frame softmax distribution over the keys of the j-th latent frame, we apply a visibility-weighted cross-entropy loss for each query point pv

1

i :

1

2,GT

2,GT

ov

j CE Cv

1,v2

i ,·),pv

i,j (pv

j . (20)

Lcorr =

1

v2 j

2,GT

Here, CE(·,·) denotes the cross-entropy loss with the ground-truth matching point pv

j as the target label. By averaging over all co-visible target frames j in both views v2 ∈ {ref,tgt}, the loss jointly supervises intra-video temporal correspondences within each view and inter-video cross-view correspondences across the two views, directly shaping where each query token attends to enforce geometric consistency both across the reference and generated views and within the generated views themselves.

### C More Generation Results

To complement the iPhone results in Fig. 7, we provide additional qualitative comparisons on the DAVIS dataset [34], which contains in-the-wild monocular videos with diverse object and camera motion. Figs. 12, 13, and 14 show novel-view generations across representative scenes covering both dynamic foreground subjects and wide outdoor environments, each under a different target camera trajectory. Explicit 3D Lifting baselines such as CogNVS, GEN3C, and TrajectoryCrafter preserve the coarse scene layout but introduce noticeable artifacts in regions where depth estimation becomes unreliable, including water surfaces, thin structures, and fast-moving subjects. NeoVerse mitigates some of these artifacts but still produces residual distortions on dynamic objects under large camera motion. Camera Conditioning Only baselines such as ReCamMaster and Redirector synthesize plausible-looking frames but lack geometric grounding under viewpoint change, leading to inaccurate parallax, drifting backgrounds, and non-rigid warping of foreground objects. In contrast, our MVTrack4Gen generates novel-views that remain visually faithful to the reference video while accurately following the prescribed target camera trajectory, preserving both the structure of dynamic

subjects and the geometry of the surrounding scene. These qualitative observations are consistent with the quantitative trends reported in Tab. 1, further confirming that strengthening multi-view correspondence learning into a video diffusion model is effective for in-the-wild novel-view synthesis, even without any explicit 3D representation at inference.

### D Attention Visualization Results

We examine the relationship between generation fidelity and the model’s internal attention behavior As established there, the 18th layer acts as the correspondence-specialized layer, where each query point most strongly attends to its geometrically corresponding location across views; we therefore visualize the attention map at this layer. We find that synthesis errors are tightly coupled with a breakdown of this alignment: when a query point is placed on an incorrectly generated region, a model without multi-view correspondence supervision attends to spurious locations in both the reference view and the generated view, rather than to the physically corresponding point. As shown in Fig. 15, ReCamMaster [3] produces visible geometric inconsistency in the dynamic object, and its attention is dispersed onto irrelevant background structures instead of the queried object. Built upon the same backbone, MVTrack4Gen not only synthesizes markedly higher-fidelity novel-view frames but also keeps the attention sharply concentrated and consistently aligned with the correct correspondences. This visual evidence is consistent with our central finding that explicitly strengthening multi-view correspondence directly translates into geometrically faithful novel-view generation.

### E Limitations

While MVTrack4Gen achieves state-of-the-art geometric consistency and camera-pose accuracy on dynamic novel-view generation, several limitations remain. First, our method inherits the architectural constraints of the underlying ReCamMaster [3] backbone, including a fixed output resolution of 480×832 and a maximum sequence length of 81 frames, which restricts its applicability to longer or higher-resolution videos. Second, the auxiliary tracking head requires ground-truth multi-view point correspondences during training, which currently restricts training to datasets where such supervision is available (e.g., synthetic or multi-camera captures); extending to fully unconstrained in-the-wild videos would require pseudo-labeling or self-supervised correspondence learning. Finally, our method shares the common limitation of diffusion-based video generation in terms of inference cost, which currently prevents real-time applications. We leave addressing these limitations to future work.

[Figure 161]

###### Zoom Out

[Figure 162]

[Figure 163]

[Figure 164]

Reference Video

[Figure 165]

ReCamMaster

[Figure 166]

[Figure 167]

[Figure 168]

MVTrack4Gen

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Redirector

MVTrack4Gen

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

ReCamMaster

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

Redirector

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

CogNVS

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

Gen3C

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

TrajectoryCrafter

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

NeoVerse

- Figure 12: Additional qualitative comparisons on the DAVIS dataset [34]. The top Reference Video row shows reference-view frames at three timesteps, each row below renders the same target viewpoint, and the rightmost Zoom Out column visualizes the target camera trajectory in 3D. We

compare our two models, MVTrack4GenReCamMaster and MVTrack4GenRedirector, against ReCamMaster [3], Redirector [33] and CogNVS [9], GEN3C [36], TrajectoryCrafter [45], NeoVerse [43]. Baselines show geometric distortions on dynamic foreground objects and floating or blurry background artifacts under large viewpoint changes, while MVTrack4Gen produces sharp, geometrically consistent novel-views faithful to the reference video.

[Figure 197]

[Figure 198]

[Figure 199]

Arc Left

[Figure 200]

Reference Video

[Figure 201]

ReCamMaster

[Figure 202]

[Figure 203]

[Figure 204]

MVTrack4Gen

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Redirector

MVTrack4Gen

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

ReCamMaster

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Redirector

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

CogNVS

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Gen3C

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

TrajectoryCrafter

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

NeoVerse

- Figure 13: Additional qualitative comparisons on the DAVIS dataset [34]. The top Reference Video row shows reference-view frames at three timesteps, each row below renders the same target viewpoint, and the rightmost Arc Left column visualizes the target camera trajectory in 3D. We

compare our two models, MVTrack4GenReCamMaster and MVTrack4GenRedirector, against ReCamMaster [3], Redirector [33] and CogNVS [9], GEN3C [36], TrajectoryCrafter [45], NeoVerse [43]. Baselines show geometric distortions on dynamic foreground objects and floating or blurry background artifacts under large viewpoint changes, while MVTrack4Gen produces sharp, geometrically consistent novel-views faithful to the reference video.

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

Translate Up

Reference Video

[Figure 237]

ReCamMaster

[Figure 238]

[Figure 239]

[Figure 240]

MVTrack4Gen

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

Redirector

MVTrack4Gen

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

ReCamMaster

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

Redirector

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

CogNVS

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Gen3C

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

TrajectoryCrafter

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

NeoVerse

- Figure 14: Additional qualitative comparisons on the DAVIS dataset [34]. The top Reference Video row shows reference-view frames at three timesteps, each row below renders the same target viewpoint, and the rightmost Translate Up column visualizes the target camera trajectory in 3D.

We compare our two models, MVTrack4GenReCamMaster and MVTrack4GenRedirector, against ReCamMaster [3], Redirector [33] and CogNVS [9], GEN3C [36], TrajectoryCrafter [45], NeoVerse [43]. Baselines show geometric distortions on dynamic foreground objects and floating or blurry background artifacts under large viewpoint changes, while MVTrack4Gen produces sharp, geometrically consistent novel-views faithful to the reference video.

ReCamMaster

T = 0 T = 20 T = 40 T = 60

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

GeneratedVideoReferenceVideo

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

GeneratedViewReferenceView

Query Point

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

60th Frame of the Generated Video

MVTrack4GenReCamMaster

T = 0 T = 20 T = 40 T = 60

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

GeneratedVideoReferenceVideo

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

GeneratedViewReferenceView

Query Point

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

60th Frame of the Generated Video

- Figure 15: Correlation between generation quality and attention-map alignment. For both ReCamMaster [3] (top) and our MVTrack4Gen (bottom), we show, from top to bottom, the reference video, the generated novel-view video, and the corresponding attention maps overlaid on the reference view and the generated view across frames. Given a query point placed on the 60th frame of the generated video (left), we visualize the attention distribution extracted from the 18th layer, identified as the correspondence-specialized layer in Sec. 4. For ReCamMaster, regions that are incorrectly synthesized in the generated video coincide with attention that is scattered and mislocalized in both the reference and generated views, revealing a breakdown of cross-view correspondence. In contrast, our method produces substantially higher-fidelity generations while keeping the attention concentrated and well-aligned with the true corresponding region of the query point.

Redirector

T = 0 T = 20 T = 40 T = 60

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

ReferenceVideo

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

GeneratedViewReferenceViewGeneratedVideo

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

Query Point

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

30th Frame of the Generated Video

MVTrack4GenRedirector

T = 0 T = 20 T = 40 T = 60

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

ReferenceVideoGeneratedVideo

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

GeneratedViewReferenceView

Query Point

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

30th Frame of the Generated Video

- Figure 16: Correlation between generation quality and attention-map alignment. For both ReCamMaster [3] (top) and our MVTrack4Gen (bottom), we show, from top to bottom, the reference video, the generated novel-view video, and the corresponding attention maps overlaid on the reference view and the generated view across frames. Given a query point placed on the 30th frame of the generated video (left), we visualize the attention distribution extracted from the 18th layer, identified as the correspondence-specialized layer in Sec. 4. For Redirector, regions that are incorrectly synthesized in the generated video coincide with attention that is scattered and mislocalized in both the reference and generated views, revealing a breakdown of cross-view correspondence. In contrast, our method produces substantially higher-fidelity generations while keeping the attention concentrated and well-aligned with the true corresponding region of the query point.

