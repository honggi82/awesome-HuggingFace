# arXiv:2507.10065v2[cs.CV]22Feb2026

## MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second

Chenguo Lin1∗, Yuchen Lin1,3∗, Panwang Pan2†, Yifan Yu2, Tao Hu2, Honglei Yan2, Katerina Fragkiadaki3, Yadong Mu1‡ 1Peking University, 2ByteDance, 3Carnegie Mellon University https://chenguolin.github.io/projects/MoVieS

##### Abstract

We present MoVieS, a Motion-aware View Synthesis model that reconstructs 4D dynamic scenes from monocular videos in one second. It represents dynamic 3D scenes with pixel-aligned Gaussian primitives and explicitly supervises their time-varying motions. This allows, for the first time, the unified modeling of appearance, geometry and motion from monocular videos, and enables reconstruction, view synthesis and 3D point tracking within a single learningbased framework. By bridging view synthesis with geometry reconstruction, MoVieS enables large-scale training on diverse datasets with minimal dependence on task-specific supervision. As a result, it also naturally supports a wide range of zero-shot applications, such as scene flow estimation and moving object segmentation. Extensive experiments validate the effectiveness and efficiency of MoVieS across multiple tasks, achieving competitive performance while offering several orders of magnitude speedups.

##### 1. Introduction

Humans and animals perceive a continuous stream of observations from a dynamic 3D world, and effortlessly interpret its underlying geometry and motion. Replicating this capability is essential for any embodied agent that must understand and act in the physical world.

Recent advances have made great strides in individual 3D tasks, such as monocular depth estimation [2, 6, 23, 53, 84], 3D scene reconstruction [56, 65, 69, 74, 91], novel view synthesis [12, 24, 43, 45, 76] and point tracking [10, 15, 22, 71, 77]. However, besides treating each task in isolation, most existing view synthesis and reconstruction studies focus on static scenes and require costly per-scene optimization without learning prior knowledge. Real-world environments are inherently dynamic and diverse, and all the aforementioned 3D scene understanding tasks could share common underlying principles.

∗: Equal contribution; †: Project lead; ‡: Corresponding author.

Motivated by this, we introduce MoVieS, a Motionaware dynamic View Synthesis model for feed-forward 4D reconstruction of monocular videos, that jointly models scenes’ appearance, geometry and motion. MoVieS represents 3D dynamic scenes using renderable and deformable 3D particles, termed dynamic splatter pixels, and utilizes a differentiable 3D Gaussian rendering framework [24]. Specifically, following recent practices on feedforward view synthesis [5, 34, 63, 80, 92], each input pixel is mapped to a 3D Gaussian primitive, with its 3D location determined by predicted depth. To model dynamics, MoVieS regresses per-pixel motion displacements toward arbitrary query timestamps, enabling temporal tracking of each splatter pixel. This design facilitates coherent reconstruction of both 3D geometry and appearance across camera viewpoints and temporal frames.

As shown in Figure 1, MoVieS is built upon a largescale pretrained transformer backbone [47, 69], which encodes each video frame independently and aggregates their information via attentions [66]. The aggregated features are then processed by specialized prediction heads: (1) a depth head estimates depth for each input frame, (2) a splatter head predicts per-pixel 3D Gaussian [24] appearance attributes, such as color and opacity, for novel view rendering, (3) a motion head estimates the time-conditioned movements of Gaussian primitives towards a target timestamp, allowing us to track its temporal evolution.

Benefiting from the unified architecture, MoVieS can be trained on large-scale datasets featuring both static [29, 75, 97] and dynamic [3, 44] scenes, as well as point tracking datasets [20, 21, 96]. During inference, it takes a monocular video, whether depicting a static or dynamic scene, and reconstructs per-pixel 3D Gaussian primitives along with their motion attributes at any target timestamp, enabling view synthesis, depth estimation, and 3D point tracking in a single model.

Extensive experiments on diverse benchmarks [14, 25, 90, 97] reveal that MoVieS achieves competitive performance across a variety of 4D perception tasks, while being several orders of magnitude faster than the existing state of

Time Emb.

[Figure 1]

[Figure 2]

[Figure 3]

t1

... ...

Camera Pos.

[Figure 4]

Depth Head

Patchify & PE

Image Encoder

[Figure 5]

+

𝒂 = {𝒒,𝒔,𝛼,𝒄 }

Splat Head

Feature Backbone

Share Weights

...

...

tN

...

Camera Pos.

Query Time

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Patchify & PE

... ...

Motion Head

Image Encoder

+

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

- Figure 1. Overview. MoVieS consists of a shared image encoder, an attention-based feature backbone (Sec. 3.2.1), and three heads

- (Sec. 3.2.2) to jointly model appearance, geometry and motion. Motion head is time-conditioned to model dynamic content with respect to several query timestamps. Normalized XYZ values in the 3D space of motion maps are treated as RGB channels for visualization. Time-varying Gaussian attributes are omitted and point clouds with color and identity Gaussian attributes are visualized here for brevity.

However, all of these works are limited to static scenes. In contrast, in this work, we extend feed-forward 3DGS reconstruction to dynamic scenarios with moving objects.

the art. Furthermore, empowered by novel view synthesis as a proxy task, MoVieS enables dense motion understanding from sparse tracking supervision. This naturally gives rise to various applications, such as scene flow estimation and moving object segmentation, further broadening the potential of our approach.

###### 2.2. Dynamic Reconstruction and View Synthesis

Compared to static reconstruction, dynamic reconstruction remains underexplored. Extensions of DUSt3R [8, 20, 61] handle dynamic reconstruction in a plug-and-play manner [8] or leveraging foundation models [42, 87, 91] with monocular depth [2, 51], optical flow [64], and point tracking [22], but remain limited to two-frame inputs and output only sparse point clouds. CUT3R [73] extends to video inputs via recurrent updates, while diffusion-based methods [19, 82] treat reconstruction as conditional generation, though at the cost of multiple passes per sequence.

In summary, our main contributions include:

- • We introduce MoVieS, a novel feed-forward framework that jointly models appearance, geometry and motion for 4D scene perception from monocular videos.
- • Dynamic splatter pixels are proposed to represent dynamic 3D scenes as renderable deforming 3D particles, bridging novel view synthesis and dynamic geometry reconstruction.
- • MoVieS delivers strong performance and orders of magnitude speedups for 4D reconstruction, and naturally enables a wide range of applications in a zero-shot manner.

Instead of geometric structure, dynamic novel view synthesis focuses on rendering quality and view-dependent effects. NeRF-based methods [4, 12, 45] reconstruct dynamic scenes from multi-view [1, 28, 70] or monocular videos [30, 48, 52, 95], while 3DGS [24] extends this with 4D primitives [31, 86] or deformable fields [35, 37, 43, 76, 85]. Recent works further express Gaussian motion explicitly [26, 36, 60, 62, 72], but require scratch training, iterative optimization, and external supervision from point tracking or optical flow estimation [10, 64].

##### 2. Related Work

###### 2.1. Feed-forward 3D Reconstruction

Traditional 3D reconstruction relies on dense multi-view supervision and per-scene optimization [16, 45, 46, 56]. Recent learning-based methods leverage large-scale priors to enable feed-forward prediction of depth, pose, and other 3D properties [17, 50, 53, 68]. With differentiable rendering techniques such as 3D Gaussian Splatting (3DGS) [24], some works further regress pixel-wise Gaussian attributes for novel view synthesis [5, 9, 92]. DUSt3R [74] pioneered the direct regression of pixel-aligned pointmaps in a canonical space from image pairs. Its extensions [27, 38, 59, 67, 94] generalize this framework with multi-view, streaming, and 3DGS integration. VGGT [69] unifies these advances with a strong image encoder [47] and task-specific heads.

For feed-forward 4D view synthesis, STORM [83] is designed for outdoor driving scenes from multi-view videos. BTimer [33] and NutWorld [57] estimate 3DGS attributes from monocular inputs. However, BTimer predicts independent pixel-wise 3D Gaussians per timestamp without modeling their frame-wise relations and needs an extra enhancer module for smooth intermediate frames. NutWorld models Gaussian motion but lacks explicit supervision, relying heavily on pretrained depth [6] and flow [79] estima-

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

...

Query Time ∈ ℝ𝑴

∆𝒙 ∆𝒂

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

...

Gaussian Deformation

AdaLN

...

...

...

[Figure 41]

[Figure 42]

Motion Head

...

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Rasterization

...

{𝒙,𝒒,𝒔,𝜶,𝒄𝒓𝒈𝒃}

Input Video ∈ ℝ𝑵×𝟑×𝑯×𝑾

Rendered Video ∈ ℝ𝑴×𝟑×𝑯×𝑾

Motion Map ∈ ℝ𝑴×𝑵×𝟑×𝑯×𝑾

- Figure 2. Motion Head. Given M query timesteps, the proposed motion head is conditioned via adaptive layer normalization (AdaLN) and predicts 3D displacements for each input pixel. After rasterization using the M corresponding query-time cameras, output images in shape M × 3 × H × W are rendered for supervision. Gaussian attribute deformation ∆a is omitted for brevity.

tion models, and uses an orthographic camera, which further leads to projection distortions.

- 3. Method

###### 3.2. Unify Appearance, Geometry and Motion

###### 3.2.1. Feature Backbone

Given a posed video V = {Ii,Pi,Ki,ti}Ni=1, we first patchify each input image Ii and use a pretrained image encoder [47] to extract their features as shown in Figure 1. To effectively incorporate camera information, we adopt two complementary strategies to embed the camera parameters into image features: (1) Pl¨ucker embedding: camera pose Pi and intrinsics Ki are transformed into pixelaligned Pl¨ucker embeddings [58], which are then downsampled and fused with image features via spatial-wise addition; (2) Camera token: Pi and Ki are passed through a linear layer and encoded to a camera token, which is appended to the sequence of image tokens. Ablation study on these two camera injection manners is validated in Sec. 4.4.

Following previous studies on monocular 4D reconstruction [26, 33, 37, 72, 83, 95], given a posed video V := {Ii,Pi,Ki,ti}Ni=1 with frames Ii ∈ R3×H×W, camera poses Pi ∈ SE(3), intrinsics Ki ∈ R3×3 and timestamps ti ∈ [0,1], we aim to train a generic model that jointly reconstructs its appearance, geometry, and motion. We leave the handling of unposed videos to future work.

###### 3.1. Dynamic Splatter Pixel

To model 3D scenes with moving contents, we propose a novel representation, namely dynamic splatter pixel, which decomposes dynamic scenes into a set of static Gaussian primitives and their corresponding deformation fields. Given an input video V, each pixel in the i-th frame Ii is associated with a splatter pixel gi [5, 63, 92] in a shared canonical space of the first frame camera’s coordinate system. Each g := {x,a} is parameterized by its position x ∈ R3 in the canonical space and other rendering attributes a ∈ R11, including rotation quaternion q ∈ R4, scale

To inform the model that input images originate from a temporally ordered video, we additionally encode the timestamp ti ∈ [0,1] of input frames by sinusoidal positional encoding [45] to produce a timestamp token, which is then concatenated with the aforementioned image and camera tokens as shown in Figure 1. After tokenizing input images, camera parameters, and timestamps, we apply the geometrically pretrained attention blocks [66] from VGGT [69] to enable interactions among image tokens across video frames. This produces a set of shared feature tokens enriched with inter-frame context as well as camera and temporal information, which are then used for predicting various properties of dynamic scenes. The effect of VGGT initialization is discussed in Sec. 4.4.

- s ∈ R3, opacity α ∈ R, and color crgb ∈ R3 [24]. Considering splatter pixels were originally designed for static scenes, we decouple motion from geometric structure to adapt them for dynamic scenes. An additional time-dependent deformation field is introduced, in which each splatter pixel g is associated with m(t) := {∆x(t),∆a(t)}. ∆x(t) ∈ R3 is the motion vector of a splatter pixel at time t with respect to the canonical 3D space, and ∆a(t) is the change of the corresponding attributes of the splatter pixel at time
- t. Therefore, the splatter pixel g is deformed to time t as: x ← x + ∆x(t), a ← a + ∆a(t). (1)

###### 3.2.2. Prediction Heads

Tokens from the feature backbone are fed into three parallel prediction heads that estimate the appearance, geometry and motion of a dynamic scene respectively, as shown in Figure 1. Each head adopts a DPT-style architecture [54] to convert aggregated tokens into dense predictions matching the input resolution and product dynamic splatter pixels.

By combining static splatter pixels and their deformation fields, we establish the correspondence between each Gaussian primitive and its temporal dynamics, thereby enabling dynamic scene modeling and dense motion estimation.

- Table 1. Training Datasets. Eight datasets from diverse sources are utilized to train MoVieS at scale.

Dataset Dynamic? Depth? Tracking? Real? #Scenes RealEstate10K [97] ✗ ✗ ✗ ✔ 70K TartanAir [75] ✗ ✔ ✗ ✗ 0.4K MatrixCity [29] ✗ ✔ ✗ ✗ 4.5K PointOdyssey [96] ✔ ✔ ✔ ✗ 0.1K DynamicReplica [21] ✔ ✔ ✔ ✗ 0.5K Spring [44] ✔ ✔ ✗ ✗ 0.03K VKITTI2 [3] ✔ ✔ ✗ ✗ 0.1K Stereo4D [20] ✔ ✔ ✔ ✔ 98K

Depth and Splatter Head Different from previous feedforward 3DGS reconstruction methods [5, 9, 92] that use a single head to predict all splatter pixel attributes, we adopt a decoupled design to better leverage geometric priors from the pretrained VGGT [69]. Specifically, a dedicated depth head, initialized from VGGT, is used for geometry prediction to provide spatial grounding for splatter pixel construction, while another DPT as splatter head is trained from scratch for appearance rendering. We further incorporate a direct RGB shortcut [88] from the input image to the final convolution layer of the splatter head to preserve highfrequency details and enhance color fidelity.

Motion Head To capture scene dynamics, a novel motion head as shown in Figure 2 is introduced to predict dense deformation m(t) for each dynamic splatter pixel at any target moment. Temporal variation is enabled by injecting the sinusoidally encoded query time tq into the aggregated tokens via adaptive layer normalization [81] before applying DPT convolutions. For each input frame Ii at time ti, the motion head predicts its 3D movements ∆x and Gaussian attribute deformation ∆a toward tq in a shared world coordinate system. To visualize motion maps, the XYZ coordinates in 3D space are jointly normalized by their shared minimum and maximum values to the range [0,1], and then mapped to RGB channels for visualization.

###### 3.3. Training

###### 3.3.1. Dataset Curation

An ideal dataset for dynamic reconstruction would include synchronized multi-view videos with dense depth and point tracking annotations. However, such data is infeasible to capture and annotate at scale in practice. Instead, we leverage a diverse set of open-source datasets [3, 20, 21, 29, 44, 75, 96, 97], each providing complementary supervision, as shown in Table 1. With the flexible model design, MoVieS can be trained on these heterogeneous sources by aligning objectives to their respective annotations. For datasets without an official test split, we randomly sample 10% of the data as the test set for generalization evaluation. Details about data curation are in the supplementary material.

3.3.2. Objectives MoVieS is trained by a multi-task objective that combines depth, rendering and motion losses:

L := λdLdepth + λrLrendering + λmLmotion. (2)

Depth and Rendering Losses Depth loss is computed as the mean squared error (MSE) between the predicted depth maps Dˆi and ground truth depth maps Di, along with their spatial gradients, after filtering out invalid values.

N

∥Di − Dˆi∥2 + ∥∇Di − ∇Dˆi∥1. (3)

Ldepth :=

i=1

Rendering loss combines pixel-wise MSE and perceptual loss [93] between 3DGS-rendered images Iˆunder M cam-

era views and the video frames at corresponding target timestamps, which are randomly sampled during training.

M

∥Iv − Iˆv∥2 + λLPIPS · LPIPS(Iv,Iˆv). (4)

Lrendering :=

v=1

Weighting terms λd, λr and λLPIPS are set to 1, 1 and 0.5 by default respectively.

Motion Loss Given 3D point tracking datasets, groundtruth motion ∆x is defined as the 3D displacement of each tracked point between any two frames. Since all 3D points are defined in the world coordinate and most tracked points remain static, implying that their corresponding motion vectors tend to zero. We apply a point-wise L1 loss between the predicted and ground-truth motions to promote sparsity after filtering out points that are not visible in the input frames. Additionally, to complement direct point-to-point alignment, a distribution loss is introduced that encourages the predicted motion vectors to preserve the internal relative distance structure within each frame. The final motion loss is defined as a combination of point-wise and distributionlevel supervision:

Lmotion := λptLpt + λdistLdist (5)

1 P i∈Ω

λpt∥∆xˆi − ∆xi∥1 (6)

=

1 P2

λdist∥∆xˆi · ∆xˆ⊤j − ∆xi · ∆x⊤j ∥1, (7)

+

(i,j)∈Ω×Ω

where Ω denotes the set of valid P tracked points, and Ω×Ω is its Cartesian product. Weighting term λm, λpt, and λdist are set to 10, 1, and 10 respectively. Ablation study on the effectiveness of these two types of motion supervision is presented in Sec. 4.4.

Normalization Similar to VGGT [69], we normalize the 3D scene scale by the average Euclidean distance from each 3D point to the origin of the canonical world coordinate system. As a result, unlike some other reconstruction methods [27, 73, 74], we do not apply additional normalization in the depth or motion loss. We also omit confidence-aware weighting [69, 74] for simplicity and more stable training.

Splatter-a-Video Shape-of-Motion MoSca Ours GT

Input View

|[Figure 47]|
|---|

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

- Figure 3. Novel View Synthesis for Dynamic Scenes. Given a monocular video, we compare synthesized novel views of different methods. Invisible regions are rendered as black or white, depending on the implementation. More results are in the supplementary material.

##### 4. Experiments

###### 4.1. Experimental Settings

Implementation MoVieS is built on a geometrically pretrained transformer, VGGT [69], via full fine-tuning. The splatter head and camera/time embeddings are trained from scratch. These new embeddings are zero-initialized to provide a stable initialization without interfering with the pretrained model. AdamW [41] with cosine learning rate scheduling and linear warm-up is used for optimization. We observed that training MoVieS is particularly unstable, likely due to sparse annotations and the heterogeneous nature of training datasets. A curriculum strategy is used that gradually increases the training complexity, involving (1) pretraining on static scenes, (2) dynamic scenes with varying views, and (3) fine-tuning in high resolution. Several techniques, such as gsplat rendering backend [89], DeepSpeed [55], gradient checkpointing [7], gradient accumulation, and bf16 mixed precision, are adopted to improve memory and computation efficiency. Training completes in approximately 5 days using 32 H20 GPUs. More details are provided in the supplementary material.

Evaluation We evaluate MoVieS on two primary tasks: novel view synthesis (Sec. 4.2) and 3D point tracking

- (Sec. 4.3). Considering pose-aware scenarios, we provide the same camera parameters for all methods to ensure a fair comparison. Following prior works [5, 9, 26, 33, 62, 72, 80], RealEstate10K [97] is used to evaluate novel view synthesis performance on static scenes. DyCheck [14] and NVIDIA dynamic scene dataset [90] are adopted for dy-

namic scenes, in terms of PSNR, SSIM and LPIPS [93]. Since the DyCheck dataset contains invisible regions in the target novel views, we compute reconstruction metrics using the provided covisibility masks (denoted by a prefix “m”), ensuring a fair comparison across methods. For 3D point tracking, the TAPVid-3D [25] benchmark serves as the evaluation protocol, covering both indoor and outdoor scenes across three datasets. End-point Euclidean error in the 3D space (EPE3D) and the percentage of 3D points within 0.05 and 0.10 units of the ground-truth locations (δ30D.05 and δ30D.10) are utilized as metrics. A large pretrained video depth estimation model, Video Depth Anything [6], and the same camera information are combined to unproject the predicted 2D tracking points of baselines into the 3D space for fair comparison.

###### 4.2. Novel View Synthesis

###### 4.2.1. Static Scene View Synthesis

As a special case of dynamic scenes, we first evaluate MoVieS on a static dataset, RealEstate10K [97], comparing against several state-of-the-art feed-forward static scene reconstruction methods, including DepthSplat [80], our reimplementation of GS-LRM [92], and our method pretrained solely on the first-stage static reconstruction. As shown in Table 2, although MoVieS is primarily designed for dynamic scenes, it maintains competitive performance on static scenes. Notably, when processing static inputs, our predicted motion naturally converges to zero (less than 1e3), demonstrating the ability of MoVieS to implicitly differentiate between static and dynamic regions.

- Table 2. Evaluation on Novel View Synthesis. The best , second best and third best results are highlighted for clarity. † indicates our reimplemented version of GS-LRM [92]. “Ours (static)” refers to our method pretrained solely on static datasets without the motion head. The same camera parameters are provided for all methods for fair comparison.

Novel View Synthesis

Time Per Scene

RealEstate10K DyCheck NVIDIA

↑ PSNR ↑ SSIM ↓ LPIPS ↑ mPSNR ↑ mSSIM ↓ mLPIPS ↑ PSNR ↑ SSIM ↓ LPIPS Static Feed-forward

DepthSplat [80] 0.60s 26.57 80.06 0.124 13.83 43.64 0.3850 17.16 50.02 0.3023 GS-LRM† [92] 0.57s 26.94 79.13 0.139 14.60 45.35 0.3775 17.83 49.88 0.3265 Ours (static) 0.84s 27.60 81.25 0.113 15.24 47.84 0.3783 18.73 50.42 0.2959

Optimization-based

Splatter-a-Video [62] 37min - - - 13.61 31.31 0.5706 14.39 25.38 0.5983 Shape-of-Motion [72] 10min - - - 17.96 56.62 0.3463 15.30 31.69 0.5087 MoSca [26] 45min - - - 18.24 55.14 0.3698 21.45 71.23 0.2653

Ours 0.93s 26.98 81.75 0.111 18.46 58.87 0.3094 19.16 51.41 0.3152

- Table 3. Evaluation on 3D Point Tracking. The best , second best and third best results are highlighted for clarity. † denotes combining a depth estimation model [6]. All methods are given the same camera information for unprojecting 3D tracked points.

3D Point Tracking

Aria Digital Twin DriveTrack Panoptic Studio ↓ EPE3D ↑ δ30D.05 ↑ δ30D.10 ↓ EPE3D ↑ δ30D.05 ↑ δ30D.10 ↓ EPE3D ↑ δ30D.05 ↑ δ30D.10

BootsTAPIR [11]† 0.5539 17.73% 32.97% 0.0617 55.82% 75.66% 0.0650 69.28% 87.95% CoTracker3 [22]† 0.5614 19.88% 35.82% 0.0637 55.30% 77.55% 0.0617 69.27% 88.04% SpatialTracker [77] 0.5413 18.08% 38.23% 0.0648 56.58% 80.67% 0.0519 72.91% 89.86% Ours 0.2153 52.05% 71.63% 0.0472 60.63% 79.87% 0.0352 87.88% 94.61%

- 4.2.2. Dynamic Scene View Synthesis

Video [62], which rely heavily on explicit motion segmentation. The challenge is particularly evident on the NVIDIA dataset, where significant camera shake hinders disentangling scene dynamics, leading to notably degraded performance and even worse than static baselines. In contrast, MoVieS exhibits strong robustness by directly learning to model motion. To evaluate the robustness of our method, visualization results on the in-the-wild DAVIS dataset [49] are provided in the supplementary material, where camera poses are estimated by MegaSaM [32].

We compare against three state-of-the-art open-sourced methods [26, 62, 72] for dynamic 3D Gaussian reconstruction, and also report static feed-forward baselines for reference. For DyCheck, we sample 13 frames of size 518×518 from a random clip of 65 frames from iPhone-record videos, and include the other two cameras for evaluation. For NVIDIA, 12 input views at 379 × 672 resolution are sampled in a round-robin style [13, 33, 39], where the i-th frame comes from the i-th camera at timestep i.

###### 4.3. 3D Point Tracking

As shown in Table 2, MoVieS achieves competitive or superior performance compared to baselines, while requiring only 0.93s per scene, which is orders of magnitude faster than prior approaches that rely on heavy pretrained models and complex multi-stage pipelines. Qualitative visualizations in Figure 3 further highlight the strength of our approach. While MoSca [26] demonstrates impressive performance, it struggles with sparse inputs and often overfits to seen poses, producing spiky and over-saturated artifacts under novel views and timesteps. In contrast, MoVieS leverages large-scale learned priors to generalize more effectively, yielding smoother and more realistic results.

Trained on large-scale point tracking datasets [20, 21, 96], the proposed method can also densely track any 3D point corresponding to a pixel across video frames. We compare MoVieS against three strong baselines, including two state-of-the-art 2D point tracking methods, BootsTAP [11] and CoTracker3 [22], as well as a native 3D point tracking approach, SpatialTracker [77]. For 2D trackers, a recent video depth estimation model [6] and ground-truth camera intrinsics are used to unproject tracked points into 3D space. Considering scale differences across methods, we normalize all predicted 3D points by their median norm before evaluation.

To ensure fair comparison and better reflect real scenarios, no video masks for dynamic objects are used in our experiments. It poses a major challenge for optimizationbased methods like Shape-of-Motion [72] and Splatter-a-

Quantitative results are reported in Table 3. While 3Dbased SpatialTracker generally outperforms 2D-based approaches, all of them rely heavily on pretrained monocular

Video Frame N/A (w/o Motion) w/o NVS + L1 Loss + Distribution Loss Ours

[Figure 65]

[Figure 66]

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

- Figure 4. Motion Visualization for Ablation Studies. We investigate key factors affecting motion learning in MoVieS, such as loss design and synergy with view synthesis. XYZ values in motion maps are normalized as RGB for visualization. Red arrows on video frames indicate motion directions.

Table 4. Ablation study on camera conditioning.

Camera Conditioning

RealEstate10K ↑ PSNR ↑ SSIM ↓ LPIPS

N/A 25.56 74.13 0.150 + Pl¨ucker embedding 25.81 74.44 0.143 + Camera token 26.81 78.65 0.121 Ours 27.60 81.25 0.113

Table 5. Ablation study on motion supervision.

###### Aria Digital Twin ↓ EPE3D ↑ δ30D.05 ↑ δ30D.10

Motion Supervision

N/A 0.7938 19.58% 32.86% + Point-wise L1 0.2262 48.74% 69.93% + Distribution loss 0.2496 45.98% 66.87% Ours 0.2153 52.05% 71.63%

depth estimators for geometry reasoning, introducing significant noise and inconsistency in the 3D space. In contrast, MoVieS directly estimates 3D point positions in a shared world coordinate, enabling more accurate and robust 3D tracking, and achieves consistently superior or competitive performance across all datasets. Visualization results are provided in the supplementary material.

###### 4.4. Ablation and Analysis

Camera Conditioning We investigate different camera conditioning strategies in the static pretraining stage. As shown in Table 4, camera tokens are injected throughout the feature backbone, enabling effective camera-aware modeling. In contrast, Pl¨ucker embeddings provide limited conditioning on their own and are merely comparable to having no camera information at all. However, as a pixel-aligned representation, Pl¨ucker embeddings are complementary to camera tokens, and their combination yields the most effective camera conditioning.

Motion Supervision To learn 3D movements in dynamic scenes, we provide two kinds of motion supervision: (1) point-wise L1 loss and (2) distribution loss, as described in Equation 5. Their efficiency is evaluated via the 3D point tracking task in Table 5. Without any motion supervision, i.e., learning solely from novel view synthesis, training exhibits severe loss oscillations and frequent None gradients. The distribution loss captures only relative motion between

pixels, while the point-wise L1 loss produces more reasonable motion maps. Combining both leads to sharper boundaries. Qualitative results of estimated motions from different motion objectives are provided in Figure 4.

Synergy of Motion and View Synthesis Thanks to the unified design of MoVieS, it supports simultaneous novel view synthesis (NVS) and motion estimation. We study their mutual benefits in Table 6. “NVS w/o motion” disables explicit motion supervision during training, relying solely on NVS as a proxy to learn dynamics. As shown in Table 6 and Figure 4, this setting fails to learn meaningful motion and tends to model static scenes. “Motion w/o NVS” detaches the motion head from 3DGS rendering and instead conditions the depth head on time. Although explicit supervision enables some motion learning, the predictions are blurry and low-quality, as shown in Figure 4. Moreover, the depth head must now model both geometry and dynamics, increasing its burden and negatively affecting NVS. These results highlight the mutual reinforcement between NVS and motion estimation, where joint training leads to better performance on both.

VGGT Initialization Although MoVieS is initialized by a pretrained model, i.e., VGGT [69], it is not an essential component of the proposed method. We verified it in our preliminary experiments that using a feature backbone trained entirely from scratch still leads to compara-

Table 6. Ablation Study on the synergy of motion estimation and novel view synthesis (NVS).

Motion & View Synthesis

DyCheck NVIDIA Aria Digital Twin

↑ mPSNR ↑ mSSIM ↓ mLPIPS ↑ PSNR ↑ SSIM ↓ LPIPS ↓ EPE3D ↑ δ30D.05 ↑ δ30D.10

NVS w/o motion 15.82 47.96 0.3741 18.38 48.56 0.3010 0.7938 19.58% 32.86% Motion w/o NVS 16.26 45.56 0.3461 18.98 49.39 0.3207 0.3801 24.72% 42.92% Ours 18.46 58.87 0.3094 19.16 50.41 0.3152 0.2153 52.05% 71.63%

|[Figure 77]|[Figure 78]<br><br>|
|---|---|
|[Figure 79]| |

|[Figure 80]|[Figure 81]<br><br>|
|---|---|
|[Figure 82]| |

|[Figure 83]|[Figure 84]<br><br>|
|---|---|
|[Figure 85]| |

[Figure 86]

(a) Scene Flow Estimation

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

(b) Moving Object Segmentation

- Figure 5. Zero-shot Applications. Predicted motion maps from our model can be directly applied to downstream tasks, such as (a) scene flow estimation and (b) moving object segmentation, in a zero-shot manner, without any task-specific fine-tuning or supervision.

ble performance with about 3 times slower convergence. The pretrained VGGT backbone simply serves to accelerate training, but does not fundamentally change the outcome. Specifically, we conduct quantitative experiments on DyCheck at 224 × 224 resolution. MoVieS initialized from VGGT reaches 17.97 dB mPSNR within 2 days, whereas the counterpart trained from scratch achieves 16.64 dB after 6 days and has not yet fully converged, suggesting that continuous training could further improve its performance.

###### 4.5. Zero-shot Applications

Scene Flow Estimation Scene flow can be naturally derived by transforming the estimated per-pixel motion vectors from world coordinates to the target camera’s coordinates. Visualization results in Figure 5(a) present sharp edges and accurate motion directions. Different colors in the flow maps indicate different movement directions that are marked by block arrows. More visualization results are provided in the supplementary material.

Moving Object Segmentation By thresholding the norm of per-pixel motion vectors, the estimated motion maps can also be used to segment moving objects (Fig-

ure 5(b)), which is an essential task in computer vision and robotics [18, 78]. Remarkably, this is achieved without any explicit supervision during training, demonstrating the strong potential of our method. More visualization results are provided in the supplementary material.

##### 5. Conclusion

We introduce MoVieS, a feed-forward model for dynamic novel view synthesis from monocular videos. It’s trained on large-scale, diverse datasets and jointly models scene appearance, geometry, and motion in a unified and efficient framework. Dynamic splatter pixels are proposed to represent dynamic scenes, enabling accurate and temporally coherent 4D reconstruction. MoVieS also supports various applications, such as depth estimation, 3D point tracking, scene flow estimation, and moving object segmentation, showcasing its versatility for dynamic scene perception. We hope this work could serve as a step toward generalizable dynamic scene understanding, and broader applications requiring spatial intelligence. Limitations and future work are discussed in the supplementary material.

##### References

- [1] Aayush Bansal, Minh Vo, Yaser Sheikh, Deva Ramanan, and Srinivasa Narasimhan. 4d visualization of dynamic events from unconstrained multi-view videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2
- [2] Aleksei Bochkovskii, AmaA ¸Gl˜ Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. In International Conference on Learning Representations (ICLR), 2024. 1, 2, 13
- [3] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2. arXiv preprint arXiv:2001.10773, 2020. 1, 4, 14
- [4] Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
- [5] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2, 3, 4, 5
- [6] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. In Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 1, 2, 5, 6, 13
- [7] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016. 5, 13
- [8] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Easi3r: Estimating disentangled motion from dust3r without training. arXiv preprint arXiv:2503.24391,

2025. 2

- [9] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision (ECCV), 2024. 2, 4, 5
- [10] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. Tapir: Tracking any point with per-frame initialization and temporal refinement. In International Conference on Computer Vision (ICCV), 2023. 1, 2
- [11] Carl Doersch, Pauline Luc, Yi Yang, Dilara Gokay, Skanda Koppula, Ankush Gupta, Joseph Heyward, Ignacio Rocco, Ross Goroshin, Jo˜ao Carreira, et al. Bootstap: Bootstrapped training for tracking-any-point. In Proceedings of the Asian Conference on Computer Vision (ACCV), 2024. 6, 13, 14
- [12] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1, 2
- [13] Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. Dynamic view synthesis from dynamic monocular video. In

- Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 6
- [14] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. Monocular dynamic view synthesis: A reality check. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 1, 5
- [15] Adam W Harley, Zhaoyuan Fang, and Katerina Fragkiadaki. Particle video revisited: Tracking through occlusions using point trajectories. In European Conference on Computer Vision (ECCV), 2022. 1
- [16] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge University Press,

2003. 2

- [17] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2024. 2
- [18] Nan Huang, Wenzhao Zheng, Chenfeng Xu, Kurt Keutzer, Shanghang Zhang, Angjoo Kanazawa, and Qianqian Wang. Segment any motion in videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 8
- [19] Zeren Jiang, Chuanxia Zheng, Iro Laina, Diane Larlus, and Andrea Vedaldi. Geo4d: Leveraging video generators for geometric 4d scene reconstruction. arXiv preprint arXiv:2504.07961, 2025. 2
- [20] Linyi Jin, Richard Tucker, Zhengqi Li, David Fouhey, Noah Snavely, and Aleksander Holynski. Stereo4d: Learning how things move in 3d from internet stereo videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 1, 2, 4, 6, 13, 14
- [21] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1, 4, 6, 13, 14
- [22] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In European Conference on Computer Vision (ECCV), 2024. 1, 2, 6, 14
- [23] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1
- [24] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuehler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. Transactions on Graphics (TOG),

2023. 1, 2, 3

- [25] Skanda Koppula, Ignacio Rocco, Yi Yang, Joe Heyward, Jo˜ao Carreira, Andrew Zisserman, Gabriel Brostow, and Carl Doersch. Tapvid-3d: A benchmark for tracking any point in 3d. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 1, 5, 14

- [26] Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2, 3, 5, 6, 14
- [27] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision (ECCV), 2024. 2, 4
- [28] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2
- [29] Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhenzhi Wang, Dahua Lin, and Bo Dai. Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 1, 4, 14
- [30] Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, and Noah Snavely. Dynibar: Neural dynamic image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
- [31] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 14
- [32] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. Megasam: Accurate, fast, and robust structure and motion from casual dynamic videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 6, 13, 15
- [33] Hanxue Liang, Jiawei Ren, Ashkan Mirzaei, Antonio Torralba, Ziwei Liu, Igor Gilitschenski, Sanja Fidler, Cengiz Oztireli, Huan Ling, Zan Gojcic, and Jiahui Huang. Feedforward bullet-time reconstruction of dynamic scenes from monocular videos. arXiv preprint arXiv:2412.03526, 2024. 2, 3, 5, 6
- [34] Chenguo Lin, Panwang Pan, Bangbang Yang, Zeming Li, and Yadong Mu. Diffsplat: Repurposing image diffusion models for scalable gaussian splat generation. arXiv preprint arXiv:2501.16764, 2025. 1
- [35] Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao. Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
- [36] Yuchen Lin, Chenguo Lin, Jianjin Xu, and Yadong Mu. Omniphysgs: 3d constitutive gaussians for general physics-based dynamics generation. arXiv preprint arXiv:2501.18982, 2025. 2
- [37] Qingming Liu, Yuan Liu, Jiepeng Wang, Xianqiang Lyu, Peng Wang, Wenping Wang, and Junhui Hou. Modgs: Dynamic gaussian splatting from casually-captured monocular videos with depth priors. In International Conference on Learning Representations (ICLR), 2025. 2, 3
- [38] Yuzheng Liu, Siyan Dong, Shuzhe Wang, Yanchao Yang, Qingnan Fan, and Baoquan Chen. Slam3r: Real-time dense

- scene reconstruction from monocular rgb videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2
- [39] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf, and Jia-Bin Huang. Robust dynamic radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 6
- [40] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations (ICLR), 2016. 13
- [41] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2018. 5, 13
- [42] Jiahao Lu, Tianyu Huang, Peng Li, Zhiyang Dou, Cheng Lin, Zhiming Cui, Zhen Dong, Sai-Kit Yeung, Wenping Wang, and Yuan Liu. Align3r: Aligned monocular depth estimation for dynamic videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2
- [43] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In International Conference on 3D Vision (3DV), 2024. 1, 2
- [44] Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andr´es Bruhn. Spring: A high-resolution highdetail dataset and benchmark for scene flow, optical flow and stereo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4981– 4991, 2023. 1, 4, 14
- [45] B Mildenhall, PP Srinivasan, M Tancik, JT Barron, R Ramamoorthi, and R Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision (ECCV), 2020. 1, 2, 3
- [46] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. Orb-slam: A versatile and accurate monocular slam system. IEEE Transactions on Robotics (T-RO), 2015. 2
- [47] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research (TMLR), 2024. 1, 2, 3
- [48] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 2
- [49] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Computer Vision and Pattern Recognition (CVPR), 2016. 6, 14, 15
- [50] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2

- [51] Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, and Luc Van Gool. Unidepthv2: Universal monocular metric depth estimation made simpler. arXiv preprint arXiv:2502.20110, 2025. 2
- [52] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2021. 2

- [53] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2020. 1, 2
- [54] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 3
- [55] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, 2020. 5, 13
- [56] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Computer Vision and Pattern Recognition (CVPR), 2016. 1, 2, 13
- [57] Qiuhong Shen, Xuanyu Yi, Mingbao Lin, Hanwang Zhang, Shuicheng Yan, and Xinchao Wang. Seeing world dynamics in a nutshell. arXiv preprint arXiv:2502.03465, 2025. 2
- [58] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 3
- [59] Brandon Smart, Chuanxia Zheng, Iro Laina, and Victor Adrian Prisacariu. Splatt3r: Zero-shot gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912, 2024. 2
- [60] Colton Stearns, Adam Harley, Mikaela Uy, Florian Dubost, Federico Tombari, Gordon Wetzstein, and Leonidas Guibas. Dynamic gaussian marbles for novel view synthesis of casual monocular videos. In SIGGRAPH Asia Conference Papers,

2024. 2

- [61] Edgar Sucar, Zihang Lai, Eldar Insafutdinov, and Andrea Vedaldi. Dynamic point maps: A versatile representation for dynamic 3d reconstruction. arXiv preprint arXiv:2503.16318, 2025. 2
- [62] Yang-Tian Sun, Yihua Huang, Lin Ma, Xiaoyang Lyu, YanPei Cao, and Xiaojuan Qi. Splatter a video: Video gaussian representation for versatile processing. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 2, 5, 6, 14
- [63] Stanislaw Szymanowicz, Chrisitian Rupprecht, and Andrea Vedaldi. Splatter image: Ultra-fast single-view 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 3

- [64] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European Conference on Computer Vision (ECCV), 2020. 2, 13
- [65] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 1
- [66] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), 2017. 1, 3
- [67] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. In International Conference on 3D Vision (3DV), 2025. 2
- [68] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. Vggsfm: Visual geometry grounded deep structure from motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
- [69] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 1, 2, 3, 4, 5, 7, 13
- [70] Liao Wang, Jiakai Zhang, Xinhang Liu, Fuqiang Zhao, Yanshun Zhang, Yingliang Zhang, Minye Wu, Jingyi Yu, and Lan Xu. Fourier plenoctrees for dynamic radiance field rendering in real-time. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2022. 2

- [71] Qianqian Wang, Yen-Yu Chang, Ruojin Cai, Zhengqi Li, Bharath Hariharan, Aleksander Holynski, and Noah Snavely. Tracking everything everywhere all at once. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 1
- [72] Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. In International Conference on Computer Vision (ICCV), 2025. 2, 3, 5, 6, 14
- [73] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2, 4
- [74] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2, 4
- [75] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In International Conference on Intelligent Robots and Systems (IROS), 2020. 1, 4, 14
- [76] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 2

- [77] Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker: Tracking any 2d pixels in 3d space. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 6
- [78] Junyu Xie, Charig Yang, Weidi Xie, and Andrew Zisserman. Moving object segmentation: All you need is sam (and flow). In Proceedings of the Asian Conference on Computer Vision (ACCV), 2024. 8
- [79] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, and Andreas Geiger. Unifying flow, stereo and depth estimation. Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2023. 2
- [80] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 1, 5, 6, 14
- [81] Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, and Junyang Lin. Understanding and improving layer normalization. In Advances in Neural Information Processing Systems (NeurIPS), 2019. 4
- [82] Tian-Xing Xu, Xiangjun Gao, Wenbo Hu, Xiaoyu Li, SongHai Zhang, and Ying Shan. Geometrycrafter: Consistent geometry estimation for open-world videos with diffusion priors. arXiv preprint arXiv:2504.01016, 2025. 2
- [83] Jiawei Yang, Jiahui Huang, Yuxiao Chen, Yan Wang, Boyi Li, Yurong You, Maximilian Igl, Apoorva Sharma, Peter Karkus, Danfei Xu, Boris Ivanovic, Yue Wang, and Marco Pavone. Storm: Spatio-temporal reconstruction model for large-scale outdoor scenes. In International Conference on Learning Representations (ICLR), 2025. 2, 3
- [84] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems (NeurIPS), 2024. 1
- [85] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for highfidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
- [86] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Realtime photorealistic dynamic scene representation and rendering with 4d gaussian splatting. In International Conference on Learning Representations (ICLR), 2024. 2
- [87] David Yifan Yao, Albert J Zhai, and Shenlong Wang. Uni4d: Unifying visual foundation models for 4d modeling from a single video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2
- [88] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, and Songyou Peng. No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. In International Conference on Learning Representations (ICLR), 2025. 4
- [89] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, et al. gsplat: An open-source library for

- gaussian splatting. Journal of Machine Learning Research (JMLR), 2025. 5, 13
- [90] Jae Shin Yoon, Kihwan Kim, Orazio Gallo, Hyun Soo Park, and Jan Kautz. Novel view synthesis of dynamic scenes with globally coherent depths from a monocular camera. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 1, 5
- [91] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and MingHsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. In International Conference on Learning Representations (ICLR), 2025. 1, 2
- [92] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In European Conference on Computer Vision (ECCV), 2024. 1, 2, 3, 4, 5, 6
- [93] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 4, 5
- [94] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views, 2025. 2
- [95] Xiaoming Zhao, Alex Colburn, Fangchang Ma, Miguel Angel Bautista, Joshua M. Susskind, and Alexander G. Schwing. Pseudo-generalized dynamic view synthesis from a video. In International Conference on Learning Representations (ICLR), 2024. 2, 3
- [96] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 1, 4, 6, 13, 14
- [97] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: learning view synthesis using multiplane images. Transactions on Graphics (TOG), 2018. 1, 4, 5, 13, 14

## MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second Supplementary Material

##### A. Implementation Details

###### A.1. Dataset Details

Detailed information about the training datasets used in this work is provided in Table S.1. The majority of training datasets used in our experiments are synthetic, providing rich annotations such as pixel-aligned depth maps and accurate camera intrinsics and extrinsics. Ground-truth 3D point trajectories are also available in PointOdyssey [96] and DynamicReplica [21]. On the other hand, Stereo4D [20] is a large-scale dataset constructed from YouTube stereo videos and annotated using pretrained foundation models [11, 64]. Despite being partially noisy, its diversity and scale offer strong generalization benefits.

The pretrained backbone, VGGT [69], requires all 3D scenes to be normalized to a unit scale on average, which in turn necessitates depth maps and camera extrinsics to be defined in a consistent metric space. It is satisfied by all synthetic datasets and Stereo4D, whose camera poses and depths are metric-aligned by construction. However, RealEstate10K [97] only provides relative camera parameters estimated via COLMAP [56], resulting in an unknown global scale. To address this issue, we re-estimate both depth maps and camera extrinsics using recent foundation models, including Video Depth Anything [6], Depth Pro [2] and MegaSaM [32], to recover aligned geometry across frames.

###### A.2. Curriculum Training

We observed that training MoVieS is particularly unstable: the training loss often fluctuates abruptly, and gradients are prone to becoming None. It may arise from sparse annotations and the heterogeneous nature of training datasets, which mix datasets from various sources with differing domains (e.g., indoor vs. outdoor, real vs. synthetic), camera FoV, recording frame rates, etc.

Thanks to the versatile design of MoVieS, we employ a curriculum strategy that gradually increases the training complexity. It begins by pretraining the model on lowresolution (224 × 224) static datasets with only depth and photometric losses, then introduces dynamic datasets along with motion supervision. We found that static datasets play a crucial role in stabilizing training for dynamic scenes, without which the loss would be highly unstable. Since modeling dynamic scenes requires reconstructing a set of 3DGS for each query time, which results in high GPU memory usage, we start by training on 5 input views for dynamic scenes and then expand to 13 views for fine-tuning. Finally, the training resolution is increased to 518. Similar

to VGGT [69], in the last training stage, we randomly sample the frame number from 2 ∼ 13 and the aspect ratio from 0.5 ∼ 2, with the largest side fixed to 518.

###### A.3. Training Details

Image encoder, feature backbone, and depth head of MoVieS are initialized from VGGT [69]. Motion head is initialized from its pointmap head. Remaining components, such as the splatter head and camera/time embeddings, are trained from scratch. We use AdamW optimizer [41] with a weight decay of 0.05, and adopt a cosine learning rate scheduler [40] with linear warm-up for all curriculum training stages. For static pretraining and dynamic scenes with 5 and 13 input views at a resolution of 224 × 224, we use learning rates of 4e-4, 4e-4 and 4e-5, respectively, with a batch size of 256. Training rates for parameters initialized from VGGT are multiplied by 0.1. With 32 H20 GPUs, training takes about 2 days for static and then dynamic scenes with 5 views, and 2 days for 13 views. MoVieS is then finetuned on 518 × 518 videos with 13 frames using a learning rate of 1e-5, which takes around 1 day. To improve memory and computation efficiency, several techniques such as gsplat [89] rendering backend, DeepSpeed [55], gradient checkpointing [7], gradient accumulation, and bf16 mixed precision are also adopted.

For the multi-task training objective, we did not manually tune the relative weights between different loss terms. Instead, the weights were set to bring the numerical values of the losses into roughly similar ranges: λd = λr = λpt = 1 and λm = λdist = 10.

##### B. Limitations and Future Work

Although MoVieS achieves competitive performance with inference speeds that are orders of magnitude faster than optimization-based methods, there remains a noticeable gap in reconstruction quality. This gap arises partly because many optimization-based approaches benefit from multiple pretrained models that preprocess input videos to provide richer and more accurate cues. Incorporating such richer prior knowledge directly into MoVieS represents a promising direction for future work to further improve reconstruction fidelity and robustness.

Currently, MoVieS depends on off-the-shelf tools for camera parameter estimation, which adds an external dependency and may limit end-to-end optimization. Seamlessly integrating camera pose estimation within the MoVieS pipeline could simplify the overall workflow, reduce error accumulation, and enhance adaptability to di-

Table S.1. Training Datasets. Eight datasets from diverse sources are utilized to train MoVieS at scale. “#Repeat” denotes dataset duplication count during integration to balance their contributions.

Dataset Dynamic? Depth? Tracking? Real? #Scenes #Frames #Repeat RealEstate10K [97] ✗ ✗ ✗ ✔ 70K 6.36M 1× TartanAir [75] ✗ ✔ ✗ ✗ 0.4K 0.49M 100× MatrixCity [29] ✗ ✔ ✗ ✗ 4.5K 0.31M 10× PointOdyssey [96] ✔ ✔ ✔ ✗ 0.1K 0.18M 1000× DynamicReplica [21] ✔ ✔ ✔ ✗ 0.5K 0.26M 100× Spring [44] ✔ ✔ ✗ ✗ 0.03K 0.003M 2000× VKITTI2 [3] ✔ ✔ ✗ ✗ 0.1K 0.03M 500× Stereo4D [20] ✔ ✔ ✔ ✔ 98K 19.6M 1×

verse scenarios.

Moreover, scaling MoVieS to handle long videos and achieve high-resolution rendering remains a challenge. The computational cost and memory demand grow significantly with scene complexity and resolution, which constrains practical deployment. Developing more compact and efficient dynamic scene representations, possibly through novel model architectures or sparse encoding strategies, is essential to push 4D reconstruction towards real-world applications.

Addressing these challenges will not only bridge the performance gap but also unlock the full potential of fast and high-quality dynamic scene reconstruction.

##### C. License Information

We employ several open-source implementations in our experimental comparisons, including: (1) DepthSplat [80]1 (MIT License), (2) Splatter-a-Video [62]2 (Apache License), (3) Shape-of-Motion [72]3 (MIT License), (4) MoSca [26]4 (MIT License), (5) BootsTAPIR [11]5 (Apache License), (6) CoTracker3 [22]6 (Creative Commons Attribution-NonCommercial 4.0 International Public License), and (7) SpatialTracker [31]7 (AttributionNonCommercial 4.0 International).

Datasets from diverse sources are utilized to train MoVieS, including: (1) RealEstate10K [97]8 (Creative Commons Attribution 4.0 International License), (2) TartanAir [75]9 (Creative Commons Attribution 4.0 International License), (3) MatrixCity [29]10 (Apache Li-

- 1https://github.com/cvg/depthsplat/tree/main
- 2https : / / github . com / SunYangtian / Splatter _ A _

Video

- 3https://github.com/vye16/shape-of-motion/
- 4https://github.com/JiahuiLei/MoSca
- 5https://github.com/google-deepmind/tapnet
- 6https://github.com/facebookresearch/co-tracker
- 7https://github.com/henry123-boy/SpaTracker
- 8https://google.github.io/realestate10k/
- 9https://theairlab.org/tartanair-dataset/
- 10https://city-super.github.io/matrixcity/

cense), (4) PointOdyssey [96]11 (MIT License), (5) DynamicReplica [21]12 (Attribution-NonCommercial 4.0 International License), (6) Spring [44]13 (CC BY 4.0), (7) VKITTI2 [3]14 (Creative Commons AttributionNonCommercial-ShareAlike 3.0), and (8) Stereo4D [20]15 (CC0 1.0 Universal).

##### D. Broader Impact

MoVieS provides substantial advantages for fields including robotics simulation, AR/VR, autonomous driving, and digital twins by enabling fast and generalizable dynamic scene understanding. Its ability to efficiently reconstruct dynamic environments can accelerate innovation and improve system performance in these areas. However, such powerful technology also raises risks related to the unauthorized generation of content and potential privacy violations. To mitigate these risks, it is essential to establish clear ethical guidelines and implement appropriate regulatory measures to ensure responsible and safe usage.

##### E. More Visualization Results

We provide more visualization results on novel view synthesis, 3D point tracking, scene flow estimation, and dynamic object segmentation in Figure S.1, S.2, S.3 and S.4 respectively. Qualitative visualizations of novel view synthesis and 3D point tracking are conducted on the DAVIS dataset [49] and TAPVid-3D [25] respectively, which are not included in the training datasets. Other visualization results are conducted on the test set of the curated datasets to evaluate the generalizability of MoVieS. XYZ values in the 3D space of motion maps are normalized to [0,1] and treated as RGB channels for visualization.

- 11https://pointodyssey.com/
- 12https://github.com/facebookresearch/dynamic_

stereo

- 13https://spring-benchmark.org/
- 14https : / / europe . naverlabs . com / research /

computer-vision/proxy-virtual-worlds-vkitti-2/

- 15https://stereo4d.github.io/

Target Viewpoint Target Time Mo on* Novel View Synthesis

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

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

- Figure S.1. Qualitative results of novel view synthesis on the DAVIS dataset [49]. Camera poses are estimated by MegaSaM [32]. “Motion∗” denotes the 3D pixel displacements between the “Target Time” frame with respect to the “Target Viewpoint” frame. The “Novel View Synthesis” results show the reconstructed scene at the target time from the target viewpoint.

### Tracks Depth Motion*

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

- Figure S.2. Qualitative results for 3D point tracking. “Motion∗” means the 3D points’ movements of the input frame with respect to the first one. Points in the 3D space are projected to the image space for 2D point tracking visualization.

#### Input Frame Depth Flow*

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

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

- Figure S.3. Qualitative results for scene flow estimation. “Flow∗” means the optical flow of the input pixels with respect to the first frame, and is obtained by projecting the estimated motion maps in the 3D space to the camera space. Each color of optical flow means a specific 2D direction and arrows on pictures roughly mark the directions.

#### Frame 1 Frame 2 Frame 3

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

- Figure S.4. Qualitative results for moving object segmentation. Masks for moving parts, which are filtered from the values of estimated motion maps, are highlighted across frames in light red. Regions with high motion values are regarded as moving parts.

