## AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views

LIHAN JIANG∗, University of Science and Technology of China, China and Shanghai Artificial Intelligence Laboratory, China YUCHENG MAO∗, Shanghai Artificial Intelligence Laboratory, China LINNING XU, The Chinese University of Hong Kong, China TAO LU, Brown University, United States of America KERUI REN, Shanghai Jiao Tong University, China and Shanghai Artificial Intelligence Laboratory, China YICHEN JIN, Shanghai Artificial Intelligence Laboratory, China XUDONG XU, Shanghai Artificial Intelligence Laboratory, China MULIN YU, Shanghai Artificial Intelligence Laboratory, China JIANGMIAO PANG, Shanghai Artificial Intelligence Laboratory, China FENG ZHAO†, University of Science and Technology of China, China DAHUA LIN, The Chinese University of Hong Kong, China BO DAI†, The University of Hong Kong, China

[Figure 1]

# arXiv:2505.23716v2[cs.CV]15Sep2025

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

|32-view Scene|
|---|

|[Figure 11]<br><br>64-view Scene|
|---|

|16-view Scene|
|---|

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

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Ours: 20.31dB (PSNR)

[Figure 33]

[Figure 34]

[Figure 35]

(a) Ours

RGB Depth

AnySplat

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

###### 3DGS: 19.56dB (PSNR)

[Figure 41]

###### …

…

RGB Depth

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Ours: 0.245 (LPIPS)

Instantly lift uncalibrated 2D to ready-to-view 3D

[Figure 47]

[Figure 48]

RGB Depth

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

3DGS: 0.277 (LPIPS)

(b) 3D-GS

> 100-view Scene

RGB Depth

Dense Captures

Fig. 1. AnySplat lifts multi-view captures, from sparse to dense, into ready-to-view 3D scenes represented with 3D Gaussians [Kerbl et al. 2023]. Unlike previous multi-view reconstruction and neural rendering methods, which rely on precise camera calibration, tedious per-scene optimization, and are often sensitive to input noise, AnySplat robustly handles a wide variety of capture scenarios in just seconds.

∗Equal Contribution, Alphabetical Order. †Corresponding Author

Artificial Intelligence Laboratory, Shanghai, China, yumulin@pjlab.org.cn; Jiangmiao Pang, Shanghai Artificial Intelligence Laboratory, Shanghai, China, pangjiangmiao@ gmail.com; Feng Zhao, University of Science and Technology of China, Anhui, China, fzhao956@ustc.edu.cn; Dahua Lin, The Chinese University of Hong Kong, Hong Kong, China, dhlin@ie.cuhk.edu.hk; Bo Dai, The University of Hong Kong, Hong Kong, China, bdai@hku.hk.

Authors’ Contact Information: Lihan Jiang, University of Science and Technology of China, Anhui, China and Shanghai Artificial Intelligence Laboratory, Shanghai, China, mr.lhjiang@gmail.com; Yucheng Mao, Shanghai Artificial Intelligence Laboratory, Shanghai, China, yucheng.mao.cs@gmail.com; Linning Xu, The Chinese University of Hong Kong, Hong Kong, China, linningxu@link.cuhk.edu.hk; Tao Lu, Brown University, Rhode Island, United States of America, tao_lu@brown.edu; Kerui Ren, Shanghai Jiao Tong University, Shanghai, China and Shanghai Artificial Intelligence Laboratory, Shanghai, China, renkerui@sjtu.edu.cn; Yichen Jin, Shanghai Artificial Intelligence Laboratory, Shanghai, China, 13905152060@163.com; Xudong Xu, Shanghai Artificial Intelligence Laboratory, Shanghai, China, xuxudong@pjlab.org.cn; Mulin Yu, Shanghai

for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed

We introduce AnySplat, a feed-forward network for novel-view synthesis from uncalibrated image collections. In contrast to traditional neural-rendering pipelines that demand known camera poses and per-scene optimization, or recent feed-forward methods that buckle under the computational weight of dense views—our model predicts everything in one shot. A single forward pass yields a set of 3D Gaussian primitives encoding both scene geometry and appearance, and the corresponding camera intrinsics and extrinsics for each input image. This unified design scales effortlessly to casually captured, multi-view datasets without any pose annotations. In extensive zero-shot evaluations, AnySplat matches the quality of pose-aware baselines in both sparse- and dense-view scenarios while surpassing existing pose-free approaches. Moreover, it greatly reduces rendering latency compared to optimization-based neural fields, bringing real-time novel-view synthesis within reach for unconstrained capture settings. Project page: https://city-super.github.io/anysplat/.

CCS Concepts: • Computing methodologies → Rendering; Reconstruction; Neural networks.

Additional Key Words and Phrases: Multi-View Capture, 3D Gaussian Splatting, Novel-View Synthesis, Feed-Forward Models

#### ACM Reference Format:

Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, Dahua Lin, and Bo Dai. 2025. AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views. ACM Trans. Graph. 44, 6 (December 2025), 16 pages. https: //doi.org/10.1145/3763326

1 Introduction

Recent advances in 3D foundation models [Wang et al. 2025a, 2024c; Yang et al. 2025a] have reshaped how we view the problem of reconstructing 3D scenes from 2D images. By inferring dense point clouds from a single view to thousands within seconds, these methods streamline or even eliminate traditional multi-stage reconstruction pipelines, making 3D scene reconstruction more accessible across a wider range of applications.

Despite their powerful geometry priors, current foundation models often struggle to capture fine detail, photorealism, and geometric consistency—especially when processing highly overlapping inputs, which can yield misaligned or noisy reconstructions. By contrast, novel-view synthesis (NVS) methods such as NeRF [Mildenhall et al. 2021] and its recent extensions [Kerbl et al. 2023] deliver exceptional rendering fidelity, but only by offloading the hard work to a costly preprocessing stage. These pipelines first estimate camera poses via structure-from-motion and then perform per-scene neural field optimization. This delay between capture and usable output, along with computation costs that grow with the number of input frames, limits their practical applicability in many real-world scenarios.

Witnessing this paradigm shift brought by feed-forward architectures like ViT [Dosovitskiy et al. 2020] in 3D modeling, we ask: can novel-view synthesis (NVS) from multiview captures naturally benefit? To bridge the gap between geometry priors and “ready-to-see” output, as exemplified in Fig. 1, we augment the foundation model with a lightweight rendering head. During training, this head refines and synthesizes appearance via a pseudo-label distillation training

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM 1557-7368/2025/12-ART https://doi.org/10.1145/3763326

strategy, no ground-truth 3D annotations required, thereby injecting texture priors and enforcing geometric coherence in a single, end-to-end pass. This training strategy paves the way for extending the reach of 3D foundation models [Wang et al. 2025a, 2024c; Yang et al. 2025a] far beyond finite, annotated datasets—enabling seamless generalization to unbounded new scenes with minimal overhead.

Specifically, we propose AnySplat, a feed-forward network for novel view synthesis trained on unconstrained and unposed multiview images. AnySplat employs a geometry transformer to encode these images into high-dimensional features, which are then decoded into Gaussian parameters and camera poses. To improve efficiency, we introduce a differentiable voxelization module that merges pixel-wise Gaussian primitives into voxel-wise Gaussians, eliminating 30–70% of redundant primitives while maintaining comparable rendering quality. Since 3D annotations in real-world scenarios are often noisy, we design a novel pseudo-label knowledge distillation pipeline. In this framework, we distill camera and geometry priors from pretrained VGGT [Wang et al. 2025a] backbone as external supervision. As a result, AnySplat can be trained without any 3D SfM or MVS supervision, relying solely on uncalibrated images, making it promising to scale up to unconstrained capture with readily usable input. We train AnySplat on nine diverse and large-scale datasets, exposing the model to a wide range of geometric and appearance variations. As a result, our method demonstrates superior zero-shot generalization performance on unseen datasets. Experimental results show that AnySplat achieves excellent novel view synthesis quality, more consistent geometry, more accurate pose estimation, and faster inference times compared to both stateof-the-art feed-forward and optimization-based methods.

In summary, our key contributions are:

- • Feed-forward reconstruction and rendering. Our model takes uncalibrated multi-view inputs and simultaneously predicts 3D Gaussian primitives and their camera intrinsics/extrinsics, delivering higher-quality reconstructions than prior feedforwardmethods—andevenoutperforming optimization-based pipelines in challenging scenarios.
- • Efficient pseudo-label knowledge distillation. We distill geometry and texture priors from a pretrained VGGT model via a novel, end-to-end training pipeline—with only RGB images—unlocking high-fidelity rendering and enhanced multiview consistency in under one day on 8–16 GPUs.
- • Differentiable voxel-guided Gaussian pruning. Our custom voxelization strategy eliminates 30–70 % of Gaussian primitives while preserving rendering quality, yielding a unified, compute-efficient model that gracefully handles both sparse and dense capture setups.

2 Related Work 2.1 Optimization-based Novel View Synthesis Methods.

Neural Radiance Fields (NeRF) [Barron et al. 2022; Mildenhall et al. 2021; Müller et al. 2022] pioneered high-quality novel view synthesis by learning continuous volumetric density and radiance fields via coordinate-based networks, but its reliance on expensive volume

rendering precludes real-time performance. In contrast, 3D Gaussian Splatting (3DGS) [Feng et al. 2025; Jiang et al. 2024; Kerbl et al. 2023; Lu et al. 2024a,b; Ren et al. 2024; Yang et al. 2025b; Yu et al. 2024b,a] explicitly represents scenes with millions of anisotropic Gaussians and exploits differentiable rasterization to render photorealistic views at over 30 FPS (1080p). Its core advances—adaptive density control for geometry refinement and spherical harmonics for view-dependent shading—enable real-time playback. Despite these advances, most NeRF and 3DGS methods assume access to accurate camera poses, typically obtained via classical Structurefrom-Motion tools such as COLMAP [Schonberger and Frahm 2016] or other relevant methods [Brachmann et al. 2024; Pan et al. 2024; Wang et al. 2024b]. This requirement introduces an implicit preprocessing step that conceals the significant time and logistical costs of large-scale, multi-view data acquisition and registration. To address these limitations, recent approaches attempt to jointly optimize camera poses and scene representation. However, they either require incremental image sequences and intrinsics [Fu et al. 2024; Keetha et al. 2024; Matsuki et al. 2024; Meuleman et al. 2025; Yan et al. 2024] as input or are limited to scenarios with minimal motion [Meng et al. 2021; Wang et al. 2021b] or sparse view coverage [Fan et al. 2024]. Furthermore, these methods still involve redundant optimization processes. In contrast, AnySplat can directly predict 3D Gaussians and camera parameters within seconds, significantly accelerating the 3D reconstruction process.

- 2.2 Generalizable 3D Reconstruction Methods.

Most view synthesis methods require tens of minutes or even hours to optimize on densely captured data. Recently, several generalizable 3D reconstruction methods have been proposed, which can be broadly categorized into two types: pose-aware methods, which assume known camera parameters, and pose-free methods, which jointly infer both geometry and camera poses.

Pose-awaregeneralizablemethods. Pose-awaregeneralizablemeth-

ods rapidly reconstructed 3D models from calibrated images and their corresponding poses. These approaches can be broadly categorized into three methodological strands: (1) 3D Gaussian Splatting based techniques [Charatan et al. 2024; Chen et al. 2024a,c; Wang et al. 2024a; Xu et al. 2024a] which directly predict 3D Gaussian primitive as the scene representation, (2) neural network based frameworks [Chen et al. 2021; Flynn et al. 2024; Jiang et al. 2025; Jin et al. 2024; Wang et al. 2021a; Yu et al. 2021] employing neural network to infer the appearance of the novel view image without any 3D representation, and (3) the emerging LRM architecture family [Hong et al. 2023; Xu et al. 2024b; Zhang et al. 2024; Ziwen et al. 2024]. Despite these pose-aware reconstruction methods significantly reducing optimization time and improving performance under sparse-view conditions, their broader applicability remains limited due to the necessity for accurate image poses as input.

Pose-free generalizable methods. To achieve truly end-to-end 3D reconstruction, pose-free generalizable methods rely solely on images as input, and most of them simultaneously predict image poses alongside the reconstructed 3D model. Among them, Dust3R [Wang et al. 2024c] and extended by MASt3R [Leroy et al. 2024], replace

traditional multi-stage pipelines with a single large-scale model that jointly predicts depth and fuses it into a dense scene. More recent methods [Liu et al. 2025; Murai et al. 2025; Tang et al. 2025; Wang and Agapito 2024; Wang et al. 2025a,b; Yang et al. 2025a], cascade transformer blocks to jointly infer camera poses, point trajectories, and scene geometry in a single forward pass, achieving substantial improvements in both accuracy and runtime. While these methods highlight the potential for efficiently scaling up 3D asset reconstruction, they generally struggle in poor texture representation and multi-view misalignment problem, which significantly hinder their novel view synthesis performance. Another line of work [Chen et al. 2024b; Hong et al. 2024; Jiang et al. 2023; Smart et al. 2024; Wang et al. 2023; Ye et al. 2024; Zhang et al. 2025] targets novel view synthesis from unposed images, but these methods only work in sparse-view settings.

3 Method

We propose AnySplat, a transformer-based neural network designed for rapid 3D scene reconstruction tailored for novel-view synthesis. Given uncalibrated images, from a single view up to hundreds, AnySplat directly predicts a set of 3D Gaussian primitives that compactly represent the reconstructed scene.

In the following sections, we first formalize our problem setup in Sec. 3.1, detail the model’s architecture and pipeline in Sec. 3.2, and finally present our training and inference strategies in Sec. 3.3.

- 3.1 Problem Setup Consider 𝑁 uncalibrated views of a single 3D scene, given as images

{𝐼𝑖}𝑖𝑁=1, where 𝐼𝑖 ∈ R𝐻×𝑊 ×3, AnySplat aims to jointly reconstruct the scene geometry and appearance by predicting a) a collection of

𝐺 anisotropic 3D Gaussians

(𝝁𝑔,𝜎𝑔, 𝒓𝑔, 𝒔𝑔, 𝒄𝑔) 𝐺𝑔=1, (1) where each Gaussian is parameterized by a center position 𝝁 ∈ R3, a positive opacity 𝜎 ∈ R+, an orientation quaternion 𝒓 ∈ R4, an anisotropic scale 𝒔 ∈ R3, and a color embedding 𝒄 ∈ R3×(𝑘+1)2 represented via spherical-harmonic coefficients of degree 𝑘, following practice of [Kerbl et al. 2023]; and 2) the camera parameters for each view

{𝑝𝑖 ∈ R9}𝑖𝑁=1, (2)

with 𝑝𝑖 encoding the intrinsics and extrinsics of image 𝐼𝑖. Formally, our model implements the mapping:

𝑓𝜽 : {𝐼𝑖}𝑖𝑁=1 ↦−→ (𝝁𝑔,𝜎𝑔, 𝒓𝑔, 𝒔𝑔, 𝒄𝑔)

𝐺 𝑔=1

∪ {𝑝𝑖}𝑖𝑁=1. (3)

We evaluate our model on two core tasks: novel view synthesis and multi-view camera pose estimation. Notably, this pipeline also produces several useful by-products—such as a global point map, per-frame depth maps, and associated confidence scores—that can support a variety of downstream applications.

- 3.2 Pipeline

Fig. 2 illustrates the overall pipeline of framework. In a nutshell, our model begins by encoding a set of uncalibrated multi-view images into high-dimensional feature representations, which are

[Figure 54]

[Figure 55]

Gaussian params μ,σ, r, s, c

Gaussian Head FG Depth Head FD Camera Head FC

Geometry… Transformer

Decoder Layers

[Figure 56]

[Figure 57]

Depth maps D

[Figure 58]

[Figure 59]

[Figure 60]

Camera poses p

Geometry Loss Lg ,Lp ,Ld

[Figure 61]

[Figure 62]

Task Predictions

…

[Figure 63]

[Figure 64]

Voxelized 3D Gaussians

Pseudo Geometry priors

##### p,~ ~D

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 69]

[Figure 70]

[Figure 71]

❄Pretrained Model

…

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

voxel-wise GS pixel-wise GS

RGB Loss LRGB

[Figure 76]

Diﬀerentiable Voxelization

Renderings

N Input images

(RGB & Depth)

- Fig. 2. Overview of AnySplat. Starting from a set of uncalibrated images, a transformer-based geometry encoder is followed by three decoder heads: F𝐺, F𝐷, and F𝐶, which respectively predict the Gaussian parameters (𝝁,𝜎, 𝒓, 𝒔, 𝒄), the depth map 𝐷, and the camera poses 𝑝. These outputs are used to construct a set of pixel-wise 3D Gaussians, which is then voxelized into pre-voxel 3D Gaussians with the proposed Differentiable Voxelization module. From the voxelized 3D Gaussians, multi-view images and depth maps are subsequently rendered. The rendered images are supervised using an RGB loss against the ground truth image, while the rendered depth maps, along with the decoded depth 𝐷 and camera poses 𝑝, are used to compute geometry losses. The geometries are supervised by pseudo-geometry priors (𝐷,˜ 𝑝˜) obtained by the pretrained VGGT [Wang et al. 2025a].

then decoded into both 3D Gaussian parameters and their corresponding camera poses. To manage the linear growth in per-pixel Gaussians under dense views, we introduce a differentiable voxelization module that clusters primitives into voxels, significantly reducing computational cost and facilitating smoother gradient flow.

center {𝝁𝑔}𝐺𝑔=1. The Gaussian head 𝐹𝐺 combines DPT features via F𝑑(𝑡ˆ𝐼) with shallow CNN–extracted appearance features F𝑎(𝐼), and feeds their sum into a final regression CNN F𝑏 to predict opacity 𝜎𝑔, orientation 𝒓𝑔, scale 𝒔𝑔, SH color coefficients 𝒄𝑔, and per-Gaussian confidence 𝐶𝑔. Formally:

(𝐷𝑖, 𝐶𝑖𝐷) = F𝐷(𝑡ˆ𝑖𝐼), {𝝁𝑔} = proj {𝑝𝑖}, {𝐷𝑖} , {𝜎𝑔, 𝒓𝑔, 𝒔𝑔, 𝒄𝑔,𝐶𝑔} = F𝑏 F𝑑2 ({𝑡ˆ𝑖𝐼}) + F𝑎({𝐼𝑖}) .

Geometry Transformer. Following VGGT [Wang et al. 2025a], we

begin by patchifying each image 𝐼𝑖 into 𝑙𝐼 = 𝐻 𝑊𝑝2 tokens of dimension𝑑 using DINOv2 [Oquab et al. 2023], where 𝑝 = 14 and𝑑 = 1024.

(4)

To each image’s token sequence 𝑡𝑖𝐼 ∈ R𝑙𝐼×𝑑, we prepend a learnable camera token 𝑡𝑖𝑔 ∈ R1×𝑑 and four register tokens 𝑡𝑖𝑅 ∈ R4×𝑑; for the first view only, we omit positional encodings on these tokens. The combined tokens 𝑡𝑖𝐼;𝑡𝑖𝑔;𝑡𝑖𝑅 from all 𝑁 views are processed by an 𝐿-layer Alternating-Attention transformer: each layer applies a frame attention over tokens of shape R𝑁×(𝑙𝐼+5)×𝑑, then a global attention over all views jointly as R1×𝑁(𝑙𝐼+5)×𝑑.

Differentiable Voxelization. Existing feed-forward 3DGS methods typically assign one Gaussian per pixel, which works for sparse-view inputs (2–16 images) but struggles with scaled-up complexity once more than 32 views are used. To address this, building upon [Lu et al. 2024b], we introduce a differentiable voxelization module that clusters the 𝐺 Gaussian centers {𝝁𝑔} into 𝑆 voxels of size 𝜖:

Camera Pose Prediction. Camera pose estimation is essential for geometry reconstruction via novel-view rendering. The refined camera tokens 𝑡ˆ𝑖𝑔 are passed through the camera decoder 𝐹𝐶, which consists of four additional self-attention layers followed by a linear projection head, to predict each camera parameters 𝑝𝑖. As in prior work, we set the first camera pose to the identity transformation and express all remaining poses in that shared local coordinate frame.

{𝝁𝑔}𝐺𝑔=1 𝜖

, (5)

{𝑽𝑠}𝑠𝑆=1 =

where 𝑽𝑠 ∈ {1, . . .,𝑆} denotes the voxel index of Gaussian 𝑔.

To keep voxelization differentiable, each Gaussian also predicts a confidence 𝐶𝑔. We convert these scores into intra-voxel weights via softmax:

exp(𝐶𝑔) ℎ:𝑽ℎ=𝑠 exp(𝐶ℎ)

. (6)

Pixel-wise Gaussian Parameter Prediction. As shown in Fig. 2, we adopt a dual-head design based on the DPT decoder [Ranftl et al. 2021] to predict all Gaussian parameters. The depth head, F𝐷, ingests the image tokens 𝑡ˆ𝑖𝐼 and outputs per-pixel depth maps 𝐷𝑖 (with associated confidence 𝐶𝑖𝐷); these depths are then back-projected through the predicted camera poses 𝑝𝑖 to yield each Gaussian’s

𝑤𝑔→𝑠 =

Finally, any per-pixel Gaussian attribute 𝑎𝑔 (e.g., opacity or color) is aggregated into its voxel by

𝑎¯𝑠 = ∑︁

𝑤𝑔→𝑠 𝑎𝑔. (7)

𝑔:𝑽𝑔=𝑠

The output of our pipeline is parameterized by the Gaussian

attribute 𝝁𝑣,𝜎𝑣, 𝒓𝑣, 𝒔𝑣, 𝒄𝑣 𝑠 of each voxel 𝑽𝑠 ∈ {1, . . .,𝑆}. We can efficiently render the Gaussians predicted by our model using

differentiable Gaussian rasterization [Kerbl et al. 2023; Ye et al. 2025]. This strategy dramatically reduces the number of primitives to process and enables end-to-end learning.

- 3.3 Training and Inference

Geometry Consistency Enhancement. Predicting depth maps and camera poses simultaneously introduces subtle ambiguities that stem from multiview alignment and aggregation: when lifting perimage predictions to 3D, these inconsistencies manifest as layered sheets in the reconstructed point cloud, which may go unnoticed in raw point-cloud form but become glaringly obvious in rendered views. Such layering not only degrades visual fidelity but also prevents our outputs from meeting human-perceptual quality standards. To mitigate this, we introduce a geometry consistency loss that enforces agreement between rendered appearances and the underlying depth predictions, effectively smoothing out these layers and restoring coherent surface geometry.

Specifically, we enforce alignment between the depth maps 𝐷𝑖 obtained from the DPT head F𝐷 and the rendered depth maps 𝐷ˆ𝑖

from 3D Gaussians. Since𝐷𝑖 can be unreliable in challenging regions, e.g., the sky or reflective surfaces, we utilize the jointly learned

confidence map 𝐶𝑖𝐷 and apply supervision only to the top 𝑁% of pixels with the confidence, ensuring that supervision focuses on the

most trustworthy predictions. We align two depth maps as:

### ∑︁𝑛

1 𝑁

(𝐷𝑖[𝑀] − 𝐷ˆ𝑖[𝑀])2, (8)

L𝑔 =

𝑖=1

where 𝑀 is a geometry mask corresponding to the top 𝑁-quantile of the confidence map, we set 𝑁 = 30% in our experiments.

Furthermore, we observed that, in the absence of supervision from novel views, the model tends to overfit to context views in an attempt to avoid interference from varying viewpoints. This results in poor generalization and leads to failures in depth and camera prediction. To mitigate this, we leverage a powerful pre-trained transformer network [Wang et al. 2025a] to distill both camera parameters and scene geometry for stable training. Specifically, we regularize the camera parameters using the following loss function:

### ∑︁𝑁

1 𝑁

∥𝑝˜𝑖 − 𝑝𝑖∥𝜖 , (9)

L𝑝 =

𝑖=1

where 𝑝˜𝑖 represents the pseudo ground-truth pose encoding, and ∥·∥𝜖 denotes the Huber loss. We then distill geometric information using:

### ∑︁𝑛

1 𝑁

(𝐷˜𝑖[𝑀] − 𝐷ˆ𝑖[𝑀])2, (10)

L𝑑 =

𝑖=1

where 𝐷˜ is the pseudo depth map obtained from the pre-trained model [Wang et al. 2025a]. Experimental results show that this distillation loss significantly improves training stability and helps avoid convergence to poor local minima.

Training Objective. To avoid noises in the input data and better scale up data, AnySplat is trained without any 3D supervision, using a pseudo-label training approach. Specifically, given a set of unposed and uncalibrated multi-view images {𝐼𝑖}𝑖𝑁=1 as input, our method first predicts their camera intrinsics and extrinsics. These predicted parameters are first used to project the positions of Gaussian primitives, and then rendered to produce the final outputs {𝐼ˆ𝑖}𝑖𝑁=1. Note that, although our model trains with only context views without novel views, AnySplat presents great performance in novel view rendering due to the distill functions and great scene modeling capacity.

Finally, we optimize our model using a set of unposed images. We minimize the following loss function:

L = Lrgb + 𝜆2 · L𝑔 + 𝜆3 · L𝑝 + 𝜆4 · L𝑑 Lrgb = MSE(𝐼,𝐼ˆ) + 𝜆1 · Perceptual(𝐼,𝐼ˆ)

(11)

Test-Time Camera Pose Alignment (Only for calculating the rendering metrics.) During inference, both the context views I𝑐 and target views I𝑡 are provided as inputs, where I𝑐 ∩ I𝑡 = ∅. We assume the first frame of I𝑐 is identical to the first frame of I𝑐∪I𝑡. Consequently, the rotation of I𝑐 and the context portion of I𝑐 ∪ I𝑡 remains the same; the only distinction lies in their scale. To address this, we compute the average context scale factor 𝑠 from I𝑐 and the average scale factor 𝑠ˆ from I𝑐 ∪ I𝑡. The target scale is then normalized by multiplying it by the ratio 𝑠/𝑠ˆ.

Post Optimization (Optional). We also include an optional postoptimization stage to further refine reconstructions, especially when inputs are dense. After AnySplat predicts the initial set of Gaussians and camera parameters, we first prune Gaussians with low opacity value (less than 0.01), and then render images from the input camera views and compute the MSE loss and the SSIM loss between the rendered and input images. We back-propagate the gradients through the Gaussian and camera parameters. The learning rates are set as follows: 1.6e-4 for position, 5e-3 for scale, 1e-3 for rotation, 5e-2 for opacity, 2.5e-3 for color, and 5e-3 for camera pose.

4 Experiments 4.1 Experimental Setup

Datasets. Following the common practice of CUT3R [Wang et al. 2025b] and DUST3R [Wang et al. 2024c], we train our model using images from nine public datasets: Hypersim [Roberts et al. 2021], ARKitScenes [Baruch et al. 2021], BlendedMVS [Yao et al. 2020], ScanNet++ [Yeshwanth et al. 2023], CO3D-v2 [Reizenstein et al. 2021], Objaverse [Deitke et al. 2023], Unreal4K [Tosi et al. 2021], WildRGBD [Xia et al. 2024], and DL3DV [Ling et al. 2024]. These datasets collectively span synthetic and real-world content, indoor and outdoor scenes, and object- to city-scale settings. This diverse data composition exposes the model to wide-ranging geometric and appearance variations, enhancing its generalization to unseen scenarios.

Training View Sampling Strategy. View-sampling strategy is crucial for ensuring model robustness. We apply three different strategies depending on the dataset type. For object-centric datasets such as CO3D-v2 [Reizenstein et al. 2021], Objaverse [Deitke et al.

- Table 1. Quantitative Comparison on both sparse-view NVS setting (the number of input images is fewer than 16) and dense-view NVS setting (the number of input images is more than 32) on Mip-NeRF360 [Barron et al. 2022] and VR-NeRF [Xu et al. 2023] dataset. We report both 3D scene reconstruction time and rendering quality metrics. We omit reporting the times for VR-NeRF, as its timings are consistent with the input values.

3 Views 6 Views 16 Views

32 Views 48 Views 64 Views

Sparse

Dense

PSNR↑ SSIM↑ LPIPS↓ Time(s)↓ PSNR↑ SSIM↑ LPIPS↓ Time(s)↓ PSNR↑ SSIM↑ LPIPS↓ Time(s)↓ PSNR↑ SSIM↑ LPIPS↓ Time↓ PSNR↑ SSIM↑ LPIPS↓ Time↓ PSNR↑ SSIM↑ LPIPS↓ Time↓ Mip-NeRF360 [Barron et al. 2022] Dataset

NoPoSplat [Ye et al. 2024] 16.36 0.430 0.453 0.119 15.92 0.416 0.541 0.290 15.47 0.361 0.606 1.198 3D-GS [Kerbl et al. 2023] 22.19 0.640 0.248 10min 21.86 0.636 0.274 10min 21.71 0.626 0.300 10min Flare [Zhang et al. 2025] 13.52 0.350 0.601 0.271 15.35 0.407 0.573 0.415 13.21 0.348 0.695 1.201 Mip-Splatting [Yu et al. 2024a] 22.07 0.643 0.256 11min 21.79 0.625 0.275 11min 21.78 0.638 0.299 11min Ours 16.20 0.550 0.349 0.171 18.32 0.524 0.329 0.297 21.85 0.670 0.250 0.767 Ours 22.31 0.688 0.247 1.4s 21.90 0.652 0.273 2.7s 21.15 0.589 0.272 4.1s

VR-NeRF [Xu et al. 2023] Dataset NoPoSplat [Ye et al. 2024] 18.37 0.707 0.437 17.57 0.704 0.466 17.66 0.720 0.472 3D-GS [Kerbl et al. 2023] 22.37 0.774 0.302 22.86 0.780 0.306 22.10 0.770 0.315 Flare [Zhang et al. 2025] 18.58 0.717 0.470 – 18.26 0.717 0.477 – 17.02 0.709 0.510 – Mip-Splatting [Yu et al. 2024a] 22.41 0.768 0.316 – 22.55 0.772 0.314 – 21.75 0.760 0.326 – Ours 20.63 0.738 0.339 21.57 0.729 0.356 22.32 0.784 0.304 Ours 23.09 0.781 0.230 22.58 0.785 0.238 22.13 0.779 0.250

2023], and WildRGBD [Xia et al. 2024], we randomly sample views within a selected capture sequence. For sequential datasets like ARKitScenes [Baruch et al. 2021] and DL3DV [Ling et al. 2024], we first define minimum and maximum temporal gaps, then randomly select a value within this range to determine the interval between the first and last frames; views are then randomly sampled from within this interval. For unordered datasets like Hypersim [Roberts et al. 2021], BlendedMVS [Yao et al. 2020], ScanNet++ [Yeshwanth et al. 2023], and Unreal4K [Tosi et al. 2021], we sample views based on pose distances. Specifically, we randomly choose a reference frame, compute the pose distance from all other frames to this reference, and sample views based on a predefined distance threshold.

Implementation details. We set layer number 𝐿 = 24 for the Alternating-AttentionTransformerand initialize the geometry transformer and depth DPT head with weights from VGGT [Wang et al. 2025a], while the remaining layers are initialized randomly. During training, we freeze the patch embedding weights. The model has approximately 886 million parameters in total. For differentiable voxelization, we set the voxel size 𝜖 to 0.002.

We train the model using the AdamW optimizer for 15K iterations. A cosine learning rate scheduler is employed, with a peak learning rate of 2e-4 and a warmup phase of 1K iterations. For layers initialized from VGGT, the learning rate is scaled by a factor of 0.1. We train AnySplat on 16 NVIDIA A800 GPUs for approximately two days. To save GPU memory and accelerate training, we use FlashAttention, bfloat16 precision, and gradient checkpointing. For stable training, we also skip optimization steps where the total loss exceeds 0.2 after the first 1K iterations. In each iteration, we first select a training dataset at random, where each dataset is sampled according to a predefined weight (Fig. 6). From the chosen dataset, we randomly sample between 2 and 24 frames, while maintaining a constant total of 24 frames per GPU. The maximum input resolution is set to 448 pixels on the longer side. The aspect ratio is randomized between 0.5 and 1.0. Additionally, we apply intrinsic augmentation by randomly center-cropping each image to between 77% and 100% of its original size. Images are also augmented via random flipping. For the training objective, we set 𝜆1 = 0.05, 𝜆2=0.1, 𝜆3=10.0, and 𝜆4=1.0.

Baselines. We establish our sparse-view novel view synthesis baseline using previous state-of-the-art pose-free feed-forward methods, including Flare [Zhang et al. 2025] and NoPoSplat [Ye et al. 2024]. For each evaluation dataset, we select three sparse-view subsets

per scene; details of the view selection policy are provided in the appendix. Notably, prior methods require a post-optimization step during evaluation to align predicted camera poses with ground truth. However, we observe that this often fails—especially when there is limited overlap between training views—and can even degrade performance by overfitting to regions not visible during training. To ensure a fair comparison, we propose a more robust alignment strategy: we fix the first predicted camera as the identity and transform all other predicted rotations into this reference coordinate system. For translation alignment, we compute the median camera distance and estimate a relative scale factor to align the predicted and ground truth translations. For our dense-view novel view synthesis baseline, we compare against 3D Gaussian Splatting [Kerbl et al. 2023] and Mip-Splatting [Yu et al. 2024a], which both train on 30K iterations. We use 32, 48, and 64 views for training, and select 4, 6, and 8 views for evaluation, respectively. Training and testing views are jointly sampled based on camera distance. Since COLMAP [Schönberger and Frahm 2016] is often unreliable under sparse-view conditions, we use VGGT to calibrate the input images and generate a point cloud for initialization.

Metrics. To evaluate the quality of novel view synthesis, we compute PSNR, SSIM [Wang et al. 2004], and LPIPS [Zhang et al. 2018] between the predicted images and the ground truth. Additionally, to assess the accuracy of the predicted relative image poses, we use the AUC metric, which measures the area under the accuracy curve across various angular thresholds. In our evaluation, we set thresholds as 5, 10, 20, and 30. Furthermore, to evaluate multi-view geometric consistency, we report two widely used depth consistency metrics: the Absolute Mean Relative Error (AbsRel), defined as:

### ∑︁𝑀

|𝐷ˆ𝑖 − 𝐷𝑖| 𝐷𝑖

1 𝑀

AbsRel =

, (12)

𝑖=1

and the 𝛿1 accuracy, which measures the percentage of pixels where

𝐷 ˆ𝑖 𝐷𝑖

𝐷𝑖 𝐷ˆ𝑖

< 1.25. (13)

max

,

4.2 Novel-view Synthesis

Compared to prior pose-free feed-forward methods, which are typically limited to sparse-view inputs (e.g., 2–24 images), and optimization-based approaches that require up to 10 minutes per

NoPoSplat FLARE Ours GT 3D-GS Mip-Splatting Ours GT

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

- Fig. 3. Visual comparisons between our method, NoPoSplat [Ye et al. 2024] and Flare [Zhang et al. 2025] in the sparse view setting; 3D-GS [Kerbl et al. 2023] and Mip-Splatting [Yu et al. 2024a] in the dense view setting from two real-world datasets [Barron et al. 2022; Xu et al. 2023]. Our method shows excellent zero-shot performance, outperforming baselines in capturing sharp edges and intricate details.

scene for dense-view reconstruction, our model generalizes to hundreds of input views and reconstructs 3D Gaussian primitives within just a few seconds on unseen scenes. We quantitatively evaluate our method against previous approaches on two zero-shot novel view synthesis datasets: MipNeRF-360 [Barron et al. 2022] and VRNeRF [Xu et al. 2023], under both sparse-view and dense-view settings.

As shown in Tab. 1, Fig. 3 and Fig. 9, AnySplat achieves improved rendering performance on sparse-view zero-shot datasets compared to recent feed-forward methods such as NoPoSplat [Ye et al. 2024] and Flare [Zhang et al. 2025]. There are two main reasons for this performance: 1) AnySplat is trained on a diverse set of datasets and incorporates a random input view selection strategy, which contributes to its superior zero-shot generalization; 2) it achieves more accurate geometry and pose estimation, and since rendering quality strongly depends on pose accuracy, this leads to better visual results. Moreover, with an increasing number of input views, our approach demonstrates faster inference times, which is important for real-world application.

In the dense-view setting (more than 32 views), AnySplat continues to outperform optimization-based methods such as 3D-GS [Kerbl et al. 2023] and Mip-Splatting [Yu et al. 2024a] (with VGGT initialization), as shown in Tab. 1, Fig. 3 and Fig. 9. These optimization-based methods tend to overfit the training views, often resulting in artifacts in novel views. In contrast, our method reconstructs finer,

cleaner geometry and delivers more detailed rendering results. Furthermore, AnySplat achieves reconstruction times that are an order of magnitude faster than those of 3D-GS and Mip-Splatting.

Post Optimization. Although AnySplat can efficiently perform end-to-end reconstruction of high-quality Gaussian models, further improvementscanbeachievedthrough an optional post-optimization step. As shown in Fig. 7 and Tab. 2, we conduct a 200 input views experiment on the Matricity dataset [Li et al. 2023]. We demonstrate that even with 200 input, applying just 1000 steps of postoptimization (taking less than two minutes) yields improved results and 3000 steps can achieve much better results. Additionally, we conduct a 16-view experiment on the Mip-NeRF360 dataset [Barron et al. 2022], comparing our method with the InstantSplat-style [Fan et al. 2024] model, which is initialized using VGGT geometry predictions and optimized per scene with rendering losses over 1,000 iterations. As shown in Fig. 4 and Tab. 3, our feed-forward approach achieves results comparable to InstantSplat-VGGT, and that postoptimization significantly improves performance.

4.3 Pose Estimation and Multi-view Geometry Consistency

AnySplat can be applied to relative pose estimation task. We evaluate it in a feed-forward setting on the RealEstate10K [Zhou et al. 2018] and CO3Dv2 [Reizenstein et al. 2021] dataset with 10 randomly selected frames using a fixed seed for reproducibility, and compare its performance with VGGT, as shown in Tab. 4. Both two methods

- Table 2. Quantitative comparison with 200 views on Matrixcity dataset [Li

- et al. 2023]. We compare our method, as well as its variants with 1K and 3K iters of post-optimization (Ours_1000 and Ours_3000), against 3D-GS and Mip-Splatting.

Method PSNR↑ SSIM↑ LPIPS↓ Time ↓ 3D-GS [Kerbl et al. 2023] 19.10 0.614 0.450 10min Mip-Splatting [Yu et al. 2024a] 18.20 0.556 0.485 11min Ours 19.46 0.574 0.446 33s Ours_1000 20.81 0.635 0.519 2min Ours_3000 21.64 0.671 0.421 7min

Table 3. Quantitative comparison with 16 views on Mip-NeRF360 dataset [Barron et al. 2022]. We compare our method, as well as 1K postoptimization (Ours_1000), against InstantSplat-VGGT [Fan et al. 2024] style.

Method PSNR↑ SSIM↑ LPIPS↓ Time ↓ InstantSplat-VGGT [Fan et al. 2024] 23.38 0.677 0.268 3min Ours 21.85 0.670 0.250 0.767s Ours_1000 25.51 0.813 0.115 2min

[Figure 109]

[Figure 110]

GT

[Figure 111]

[Figure 112]

PSNR: 21.83 PSNR: 25.86

[Figure 113]

[Figure 114]

[Figure 115]

Ours

[Figure 116]

PSNR: 25.01

[Figure 117]

Ours +1000

[Figure 118]

PSNR: 28.39

[Figure 119]

InstantSplat-VGGT

[Figure 120]

PSNR: 24.58

[Figure 121]

[Figure 122]

PSNR: 24.71

Fig. 4. Qualitative comparison of our method with InstantSplat-VGGT [Fan

- et al. 2024] style on Mip-NeRF360 dataset [Barron et al. 2022] (room).

used Co3Dv2 samples in training, while RealEstate10K is excluded from the training set. These results highlight the benefits of our rendering-based supervision, which slightly outperforms VGGT by enforcing stronger multi-view consistency constraints.

In addition to pose estimation, we assess the multi-view geometric consistency of our approach. While VGGT [Wang et al.

- 2025a] demonstrates strong performance in monocular depth prediction, it often suffers from poor consistency across views due to the lack of explicit 3D geometry constraints and its sensitivity to low-confidence regions, particularly around object boundaries. In contrast, our method leverages 3D rendering supervision to significantly enhance multi-view consistency. To evaluate this effect, we compare the depth maps rendered from Gaussians (𝐷ˆ𝑖) with those predicted by the DPT head (𝐷𝑖) at both the beginning and

- Table 4. Camera Pose Estimation on RealEstate10K [Zhou et al. 2018] and CO3Dv2 [Reizenstein et al. 2021] with 10 random frames against VGGT [Wang et al. 2025a].

Method

RealEstate10K (unseen) Co3Dv2 AUC@30↑ AUC@20↑ AUC@10↑ AUC@5↑ AUC@30↑ AUC@20↑ AUC@10↑ AUC@5↑

VGGT 89.1 84.9 74.1 56.9 74.9 67.2 50.4 31.2 Ours 89.2 85.1 74.6 57.9 78.3 71.6 56.9 39.2

- Table 5. Ablation Study. We evaluate the ablated variants of AnySplat, discussed in Sec. 4.4, by recording their rendering quality, geometric accuracy, and the size of the resulting Gaussian models.

Method PSNR↑ SSIM↑ LPIPS↓ 𝛿1 ↑ AbsRel↓ #GS (M)

Ours w/o Distill Loss 7.28 0.217 0.832 75.5 14.7 4.80 Ours w/o Geo. Loss 18.20 0.635 0.285 94.7 7.6 3.52 Ours w/o Diff. Voxel 17.77 0.609 0.303 95.8 5.7 4.82 Ours frozen AA transformer layers 17.90 0.616 0.306 96.5 5.3 3.51 Ours frozen all transformer layers 17.84 0.621 0.330 95.3 6.6 3.40 Ours 18.25 0.648 0.279 96.3 5.9 3.45

end of training on the Hypersim dataset. As illustrated in Fig. 8, the alignment between the two depth sources improves notably over training iterations, highlighting the effectiveness of our training strategy.

4.4 Ablation Study

In this section, we ablate each individual module to validate their effectiveness. We conduct all the experiments based on the Hypersim Dataset. Quantitative and qualitative results can be found in Tab. 5.

Distill Loss. To evaluate the impact of the distillation losses defined in Eq. 9 and 10, we perform an ablation study by removing them from training. As shown in Table 5, this leads to a significant drop in both rendering quality and geometric consistency. The results suggest that, in the absence of external supervision and when trained solely on unposed images, the model tends to overfit the input views with plausible renderings without preserving accurate 3D geometry. In our experiments, the absence of a distillation loss results in incorrect depth and pose predictions, leading to degraded performance in novel view renderings. The distillation loss mitigates this by reinforcing geometric consistency during training.

Geometry Consistency Loss. We further demonstrate the effectiveness of our geometry consistency loss, defined in Eq. 8, by comparing it against a variant of our model trained with only the rendering and distillation losses. As shown in the second and last rows of Table 5, incorporating the consistency loss encourages the model to produce more coherent multi-view geometry, resulting in a 1.7% reduction in AbsRel and a 1.6% improvement in 𝛿1 accuracy.

Differentiable Voxelization. To evaluate the impact of the differentiable voxelization module introduced in Sec. 3.2, we conduct an experiment in which this component is removed. Interestingly, the model achieves slightly better performance despite using fewer Gaussian primitives. This improvement can be attributed to the voxelization module’s ability to reduce redundancy among Gaussians

[Figure 123]

- Fig. 5. Growth of Gaussian Primitives and GPU Memory Usage. As the number of input views increases, the count of Gaussian primitives grows sublinearly and eventually plateaus when using the differentiable voxelization module. In contrast, without this module, the number of Gaussians increases approximately linearly. The GPU memory consumption for rendering mirrors this saturation behavior.

and alleviate artifacts caused by overlapping primitives. Furthermore, as illustrated in Fig. 5, when differentiable voxelization is used, the number of Gaussians increases more slowly with the number of context views and eventually reaches saturation. This leads to lower GPU memory consumption during rendering compared to pixel-wise rendering approaches.

Training strategy. We investigate different training strategies by exploring the following three experimental configurations:

- 1) Frozen All Transformer: All transformer layers initialized from VGGT are frozen during training, while the remaining parameters are trainable.
- 2) Frozen AA Transformer: Only the Alternating-Attention layers are frozen, while the vision tokenizer is fine-tuned.
- 3) Frozen Vision Tokenizer: The vision tokenizer is frozen, and only the Alternating-Attention layers are fine-tuned.

Our empirical results show that the third configuration yields the best performance, achieving PSNR gains of 0.41 dB and 0.35 dB over configuration 1 and 2, respectively. These findings suggest that preserving pre-trained visual representations while adapting the attention mechanism provides an effective balance between stability and adaptability during training.

5 Conclusion and Future Works

In this work, we introduce AnySplat, a feed-forward 3D reconstruction model that integrates a lightweight rendering head with our geometry-consistency enhancement, augmented by a pseudolabel knowledge distillation training strategy. We view this as a novel way to fully unlock the potential of 3D foundation models and elevate their scalability to a broader scope. Our experiments demonstrate AnySplat’s robust and competitive results on both sparse and dense multiview reconstruction and rendering benchmarks using unconstrained, uncalibrated inputs. Additionally, the model training remains efficient, requiring minimal time and compute, enabling feed-forward 3D Gaussian Splatting reconstructions and high-fidelity renderings in just seconds at inference time. We expect this low-latency pipeline to open new possibilities for future interactive and real-time 3D applications.

Despite its improvements, AnySplat still observes artifacts in challenging regions, such as skies, specular highlights, and thin

structures; its reconstruction-based rendering loss may be less stable under dynamic scenes or varying illumination, and the compute–resolution trade-off (i.e., number of Gaussians scaling alongside input and voxel resolution) can slow performance when handling very high resolution or large numbers of views. We see enhancing patch size flexibility, improving robustness to repetitive texture patterns, and streamlining scaling to thousands of high-resolution inputs as promising directions for future work.

Acknowledgments

This work was funded in part by the National Key R&D Program of China (2022ZD0160201), Shanghai Artificial Intelligence Laboratory, the HKU Startup Fund, the HKU Shanghai Intelligent Computing Research Center, and the Anhui Provincial Natural Science Foundation under Grant 2108085UD12.

References

Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman.

2022. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 5470–5479. Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, et al. 2021. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897 (2021).

Eric Brachmann, Jamie Wynn, Shuai Chen, Tommaso Cavallari, Aron Monszpart, Daniyar Turmukhambetov, and Victor Adrian Prisacariu. 2024. Scene coordinate reconstruction: Posing of image collections via incremental learning of a relocalizer. In European Conference on Computer Vision. Springer, 421–440.

David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. 2024. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 19457–19467.

Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. 2021. Mvsnerf: Fast generalizable radiance field reconstruction from multiview stereo. In Proceedings of the IEEE/CVF international conference on computer vision. 14124–14133.

Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. 2024a. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision. Springer, 370–386.

Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang, Andrea Vedaldi, Tat-Jen Cham, and Jianfei Cai. 2024c. Mvsplat360: Feed-forward 360 scene synthesis from sparse views. arXiv preprint arXiv:2411.04924 (2024).

Zequn Chen, Jiezhi Yang, and Heng Yang. 2024b. PreF3R: Pose-Free Feed-Forward 3D Gaussian Splatting from Variable-length Image Sequence. arXiv preprint arXiv:2411.16877 (2024).

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. 2023. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 13142–13153.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).

Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, et al. 2024. Instantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds. arXiv preprint arXiv:2403.20309 2, 3 (2024), 4.

Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang, Tao Liu, Boni Hu, Linning Xu, Zhilin Pei, Hengjie Li, et al. 2025. Flashgs: Efficient 3d gaussian splatting for large-scale and high-resolution rendering. In Proceedings of the Computer Vision and Pattern Recognition Conference. 26652–26662.

John Flynn, Michael Broxton, Lukas Murmann, Lucy Chai, Matthew DuVall, Clément Godard, Kathryn Heal, Srinivas Kaza, Stephen Lombardi, Xuan Luo, et al. 2024. Quark: Real-time, High-resolution, and General Neural View Synthesis. ACM Transactions on Graphics (TOG) 43, 6 (2024), 1–20.

Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A Efros, and Xiaolong Wang. 2024. Colmap-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20796–20805.

Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jisang Han, Jiaolong Yang, Chong Luo, and Seungryong Kim. 2024. PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting. arXiv preprint arXiv:2410.22128 (2024).

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. 2023. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023).

Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik Aanæs. 2014. Large scale multi-view stereopsis evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition. 406–413.

Hanwen Jiang, Zhenyu Jiang, Yue Zhao, and Qixing Huang. 2023. Leap: Liberate sparse-view 3d modeling from camera poses. arXiv preprint arXiv:2310.01410 (2023).

Hanwen Jiang, Hao Tan, Peng Wang, Haian Jin, Yue Zhao, Sai Bi, Kai Zhang, Fujun Luan, Kalyan Sunkavalli, Qixing Huang, et al. 2025. RayZer: A Self-supervised Large View Synthesis Model. arXiv preprint arXiv:2505.00702 (2025).

Lihan Jiang, Kerui Ren, Mulin Yu, Linning Xu, Junting Dong, Tao Lu, Feng Zhao, Dahua Lin, and Bo Dai. 2024. Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes. arXiv preprint arXiv:2412.01745 (2024).

Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. 2024. Lvsm: A large view synthesis model with minimal 3d inductive bias. arXiv preprint arXiv:2410.17242 (2024).

Yuhe Jin, Dmytro Mishkin, Anastasiia Mishchuk, Jiri Matas, Pascal Fua, Kwang Moo Yi, and Eduard Trulls. 2021. Image matching across wide baselines: From paper to practice. International Journal of Computer Vision 129, 2 (2021), 517–547.

Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. 2024. Splatam: Splat track & map 3d gaussians for dense rgb-d slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21357–21366.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023.

- 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42,
- 4 (2023), 139–1.

Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik.

2023. Lerf: Language embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 19729–19739.

Vincent Leroy, Yohann Cabon, and Jérôme Revaud. 2024. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision. Springer, 71–91. Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhenzhi Wang, Dahua Lin, and Bo Dai. 2023. Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3205–3215.

Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. 2024. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22160–22169.

Yuzheng Liu, Siyan Dong, Shuzhe Wang, Yingda Yin, Yanchao Yang, Qingnan Fan, and Baoquan Chen. 2025. Slam3r: Real-time dense scene reconstruction from monocular rgb videos. In Proceedings of the Computer Vision and Pattern Recognition Conference. 16651–16662.

Tao Lu, Ankit Dhiman, R Srinath, Emre Arslan, Angela Xing, Yuanbo Xiangli, R Venkatesh Babu, and Srinath Sridhar. 2024a. Turbo-gs: Accelerating 3d gaussian fitting for high-quality radiance fields. arXiv preprint arXiv:2412.13547 (2024).

Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. 2024b. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20654–20664.

Hidenobu Matsuki, Riku Murai, Paul H. J. Kelly, and Andrew J. Davison. 2024. Gaussian Splatting SLAM. (2024).

Quan Meng, Anpei Chen, Haimin Luo, Minye Wu, Hao Su, Lan Xu, Xuming He, and Jingyi Yu. 2021. Gnerf: Gan-based neural radiance field without posed camera. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 6351–6361.

Andreas Meuleman, Ishaan Shah, Alexandre Lanvin, Bernhard Kerbl, and George Drettakis. 2025. On-the-fly Reconstruction for Large-Scale Novel View Synthesis from Unposed Images. ACM Transactions on Graphics 44, 4 (2025).

Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. 2019. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (ToG) 38, 4 (2019), 1–14.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. 2022. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG) 41, 4 (2022), 1–15.

Riku Murai, Eric Dexheimer, and Andrew J Davison. 2025. MASt3R-SLAM: Real-time dense SLAM with 3D reconstruction priors. In Proceedings of the Computer Vision and Pattern Recognition Conference. 16695–16705.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023).

Linfei Pan, Dániel Baráth, Marc Pollefeys, and Johannes L Schönberger. 2024. Global structure-from-motion revisited. In European Conference on Computer Vision. Springer, 58–77.

Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. 2016. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition. 724–732.

René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. 2021. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision. 12179–12188.

Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. 2021. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF international conference on computer vision. 10901–10911.

Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. 2024. Octree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians. arXiv preprint arXiv:2403.17898 (2024).

Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. 2021. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings of the IEEE/CVF international conference on computer vision. 10912–10922.

Johannes L Schonberger and Jan-Michael Frahm. 2016. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition. 4104– 4113.

Johannes Lutz Schönberger and Jan-Michael Frahm. 2016. Structure-from-Motion Revisited. In Conference on Computer Vision and Pattern Recognition (CVPR).

Brandon Smart, Chuanxia Zheng, Iro Laina, and Victor Adrian Prisacariu. 2024. Splatt3r: Zero-shot gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912 (2024).

Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. 2020. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2446–2454.

Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu, Rakesh Ranjan, Alexander Schwing, and Zhicheng Yan. 2025. Mv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds. In Proceedings of the Computer Vision and Pattern Recognition Conference. 5283–5293.

Fabio Tosi, Yiyi Liao, Carolin Schmitt, and Andreas Geiger. 2021. Smd-nets: Stereo mixture density networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 8942–8952.

Haithem Turki, Deva Ramanan, and Mahadev Satyanarayanan. 2022. Mega-nerf: Scalable construction of large-scale nerfs for virtual fly-throughs. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 12922–12931.

Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T Barron, and Pratul P Srinivasan. 2022. Ref-nerf: Structured view-dependent appearance for neural radiance fields. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 5481–5490.

Hengyi Wang and Lourdes Agapito. 2024. 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061 (2024).

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. 2025a. Vggt: Visual geometry grounded transformer. arXiv preprint arXiv:2503.11651 (2025).

Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. 2024b. Vggsfm: Visual geometry grounded deep structure from motion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 21686–21697.

Peng Wang, Hao Tan, Sai Bi, Yinghao Xu, Fujun Luan, Kalyan Sunkavalli, Wenping Wang, Zexiang Xu, and Kai Zhang. 2023. Pf-lrm: Pose-free large reconstruction model for joint pose and shape prediction. arXiv preprint arXiv:2311.12024 (2023).

Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser. 2021a. Ibrnet: Learning multi-view image-based rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 4690–4699.

Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. 2025b. Continuous 3D Perception Model with Persistent State. arXiv preprint arXiv:2501.12387 (2025).

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. 2024c. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20697–20709.

Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee Lee. 2024a. Freesplat: Generalizable 3d gaussian splatting towards free view synthesis of indoor scenes. Advances in Neural Information Processing Systems 37 (2024), 107326–107349.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13, 4 (2004), 600–612.

Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Victor Adrian Prisacariu. 2021b. NeRF–: Neural radiance fields without known camera parameters. (2021).

Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang. 2024. RGBD objects in the wild: scaling real-world 3D object learning from RGB-D videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22378–22389. Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. 2024a. Depthsplat: Connecting gaussian splatting and depth. arXiv preprint arXiv:2410.13862 (2024).

Linning Xu, Vasu Agrawal, William Laney, Tony Garcia, Aayush Bansal, Changil Kim, Samuel Rota Bulò, Lorenzo Porzi, Peter Kontschieder, Aljaž Božič, et al. 2023. VRNeRF: High-fidelity virtualized walkable spaces. In SIGGRAPH Asia 2023 Conference Papers. 1–12.

Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen, Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wetzstein. 2024b. Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation. In European Conference on Computer Vision. Springer, 1–20.

Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li.

2024. Gs-slam: Dense visual slam with 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19595–19604. Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. 2025a. Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass. arXiv preprint arXiv:2501.13928 (2025).

Xijie Yang, Linning Xu, Lihan Jiang, Dahua Lin, and Bo Dai. 2025b. Virtualized 3D Gaussians: Flexible Cluster-based Level-of-Detail System for Real-Time Rendering of Composed Scenes. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 1–11.

Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. 2020. Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 1790–1799.

Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, and Songyou Peng. 2024. No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv preprint arXiv:2410.24207 (2024).

Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa. 2025. gsplat: An open-source library for Gaussian splatting. Journal of Machine Learning Research 26, 34 (2025), 1–17.

Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. 2023. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 12–22.

Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. 2021. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 4578–4587.

Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xiangli, and Bo Dai. 2024b. Gsdf: 3dgs meets sdf for improved neural rendering and reconstruction. Advances in Neural Information Processing Systems 37 (2024), 129507–129530.

Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. 2024a. Mipsplatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 19447–19456.

Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. 2024. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In European Conference on Computer Vision. Springer, 1–19.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. 2025. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. arXiv preprint arXiv:2502.12138 (2025).

Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. 2018. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817 (2018).

Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. 2024. Long-lrm: Long-sequence large reconstruction model for widecoverage gaussian splats. arXiv preprint arXiv:2410.12781 (2024).

[Figure 124]

Input Frames

Predicted 3D Gaussians Predicted 3D Gaussians

Sparse Inputs

Reference GT

Reference GT

[Figure 125]

|[Figure 126]|
|---|
|[Figure 127]|

[Figure 128]

|[Figure 129]|
|---|

[Figure 130]

2Views

[Figure 131]

[Figure 132]

|[Figure 133]|
|---|

(LLFF) (DTU)

NVS

NVS

[Figure 134]

[Figure 135]

[Figure 136]

|[Figure 137]|
|---|

[Figure 138]

|[Figure 139]|
|---|

[Figure 140]

[Figure 141]

8Views

…

…

|[Figure 142]|
|---|

|[Figure 143]|
|---|

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

(MegaNeRF)

(LERF)

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

|[Figure 152]|
|---|

|[Figure 153]|
|---|

[Figure 154]

[Figure 155]

16Views

…

…

|[Figure 156]|
|---|

|[Figure 157]|
|---|

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

(HorizonGS) (HorizonGS)

Dense Inputs Reference GT

Predicted 3D Gaussians

Additional Novel View Renderings

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

|[Figure 168]|
|---|

[Figure 169]

…

32Views64Views

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

…

|[Figure 175]|
|---|

[Figure 176]

…

(VR-NERF)

NVS

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

|[Figure 184]|
|---|

…

…

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

|[Figure 189]|
|---|

[Figure 190]

[Figure 191]

…

(Waymo)

||
|---|

- Fig. 6. Example visualization of our AnySplat reconstruction and novel-view synthesis across a spectrum of scene complexities and input frames densities. From top to bottom, the number of input images increases—from extremely sparse to medium and dense captures, while the scene scale grows from object-centric setups (LLFF [Mildenhall et al. 2019], DTU [Jensen et al. 2014]) through mid-scale trajectories (MegaNeRF [Turki et al. 2022], LERF [Kerr et al. 2023], HorizonGS [Jiang et al. 2024]) to large-scale indoor and outdoor environments (VR-NeRF [Xu et al. 2023], Waymo [Sun et al. 2020]). For each setting, we display the input views, the reconstructed 3D Gaussians, the corresponding ground-truth renderings, and example novel-view renderings.

||
|---|

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

PSNR: 15.67 PSNR: 18.27 PSNR: 19.54 PSNR: 21.47 PSNR: 21.65

Mip-Splatting 3D-GS Ours Ours +1000 Ours +3000 GT

- Fig. 7. Improved Rendering with Post-Optimization. In our experiments using 200 input views, an optional post-optimization stage yields noticeably higher rendering fidelity, particularly in dense-view scenarios.

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

GT Consistency Diﬀ

GS Depth

Model Depth

AbsRel: 14.0 δ1: 77.1

AbsRel: 6.2 δ1: 95.9

GS Depth

Model Depth

Consistency Diﬀ

（a) Step 0 （b) Step 10K

- Fig. 8. Improvements of Multiview Consistency. From the initial iteration to 10k training steps, we observe a marked enhancement in multiview geometry consistency, clearly visible in the depth renderings, across both the model’s outputs and the 3D Gaussian Splatting renderings. This confirms the effectiveness of our geometry consistency enhancement design.

16 Views

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

FLARE NoPoSplat AnySplat

2 Views GT Novel View GT Novel View

…

…

32 Views GT Novel View 64 Views GT Novel View

FLARE NoPoSplat AnySplat

3DGS MipSplatting AnySplat

…

3DGS MipSplatting AnySplat

3DGaussiansRenderingInput3DGaussiansRenderingInput

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

- Fig. 9. Qualitative comparisons against baseline methods: for sparse-view inputs, we benchmark against the state-of-the-art FLARE [Zhang et al. 2025] and NoPoSplat [Ye et al. 2024]; for dense-view inputs, we include 3DGS [Kerbl et al. 2023] and MipSplatting [Yu et al. 2024a] as representative comparisons. The slight misalignment between the rendered novel-views and the ground-truth is likely caused by pose-free reconstruction method’s estimated pose not perfectly matching the annotated ground-truth camera poses.

The following appendices provide additional technical details and experimental results that support the main findings of this work.

A Experiment Details

In this section, we provide additional details of our training protocol, model initialization, and experiments.

Training Setting. We train on a heterogeneous mix of nine datasets spanning synthetic indoor scenes (Hypersim [Roberts et al. 2021], ARKitScenes [Baruch et al. 2021], BlendedMVS [Yao et al. 2020], ScanNet++ [Yeshwanth et al. 2023], CO3D-v2 [Reizenstein et al. 2021], Objaverse [Deitke et al. 2023], Unreal4K [Tosi et al. 2021], WildRGBD [Xia et al. 2024], and DL3DV [Ling et al. 2024]). During each iteration we randomly sample one dataset according to the distribution shown in Tab. 6, ensuring balanced exposure to both synthetic and real environments. This mixture stabilizes convergence and improves generalization across diverse scene types. In future work, we plan to incorporate additional high-fidelity datasets, particularly from open-world simulators such as game engines that provide accurate 3D geometry and scene consistency—to further empower our model’s scalability. We also intend to include a wider variety of camera trajectories.

Table 6. Training Datasets Statistics. We report the sampling distribution over our nine training datasets: at each iteration, we randomly select one dataset according to the probabilities listed in the Prob column, which reflects the relative frequency with which each dataset is drawn during training.

Dataset Scene Type Real/Synthetic # of Frames # of Scenes Training Prob. (%)

ARKitScenes [Baruch et al. 2021] Indoor Real 9.2M 4406 16.7 ScanNet++ [Yeshwanth et al. 2023] Indoor Real 1.0M 935 16.7 BlendedMVS [Yao et al. 2020] Mixed Real 114K 467 8.3 Unreal4K [Tosi et al. 2021] Mixed Synthetic 16K 18 8.3 CO3Dv2 [Reizenstein et al. 2021] Object Real 5.5M 27520 8.3 DL3DV [Ling et al. 2024] Mixed Real 3.4M 9894 16.7 WildRGBD [Xia et al. 2024] Object Real 3.9M 11050 8.3 Hypersim [Roberts et al. 2021] Indoor Synthetic 73K 744 8.3 Objaverse [Deitke et al. 2023] Object Synthetic 8M 199K 8.3

Model Initialization. To leverage prior geometric structure, we initialize our geometry-transformer backbone with weights pretrained on the VGGT dataset. All parameters in the Gaussian-prediction head are drawn from a zero-mean Gaussian distribution with standard deviation 0.02, while all biases are set to zero. This strategy allows the geometry branch to start from a strong prior, accelerating convergence, while the Gaussian head learns scene appearance and density from scratch.

Evaluation Setting. We evaluate our approach on two widely used benchmarks: the VR-NeRF dataset [Xu et al. 2023] and the MipNeRF360 dataset [Barron et al. 2022]. From VR-NeRF, which offers richly textured indoor environments with varied layouts, we randomly select four representative scenes—apartment, kitchen, raffurnishedroom, and workshop—ensuring a mix of both compact and spacious rooms. From Mip-NeRF360, a dataset known for its challenging viewpoint diversity and complex lighting, we include all available scenes: bonsai, counter, kitchen and room. Together, these seven scenes cover a broad spectrum of indoor settings, camera

densities, and appearance variations, allowing us to stress-test both sparse- and dense-view reconstruction scenarios.

In the dense-view setting, we select one out of every eight images as the test view. We first choose 72 images from the dataset, either randomly or based on spatial distribution. From these 72 images, we further sample subsets of 54 and 36 images. After excluding the test views, the numbers of input images for these three cases are 64, 48, and 32, respectively. In the sparse-view setting, we select one out of every two images as the test view. The view-selection procedure is the same as in the dense-view setting.

B More Results

Same Test Views. In Table 1, we use the view-selection strategy described in sec. A. However, those results do not isolate how rendering performance depends on the number of input views. To make this dependence explicit, we compare 3D-GS [Kerbl et al. 2023] and Mip-Splatting [Yu et al. 2024a] using the same fixed test views in the dense-view setting (Table 7). Specifically, we sample 72 images and hold out 8 as test views; keeping these test views fixed, we then construct training sets of 64, 48, and 32 input images by randomly selecting from the remaining 64 images. This setup not only reveals the relationship between input count and performance but also rigorously evaluates our model’s sensitivity to view sampling and its robustness across arbitrary camera configurations. These results lead to two conclusions: (1) in 3D scene reconstruction, more input views yield higher rendering quality for both feed-forward and perscene optimization methods; and (2) AnySplat’s rendering quality is consistently competitive with per-scene optimization methods, underscoring the promise of feed-forward approaches.

Table 7. Quantitative Comparison on dense-view NVS setting on MipNeRF360 [Barron et al. 2022] dataset with same test view images.

32 Views 48 Views 64 Views PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

Method

3D-GS [Kerbl et al. 2023] 17.25 0.416 0.439 18.62 0.461 0.415 19.05 0.490 0.417 Mip-Splatting [Yu et al. 2024a] 16.95 0.381 0.450 18.39 0.457 0.424 18.94 0.486 0.427 Ours 16.59 0.417 0.422 18.72 0.479 0.370 19.57 0.508 0.356

Failure Case. Although AnySplat performs well on most scenes, we still observe some failure cases (Fig. 10). For example, AnySplat can struggle with (a) variable illumination and transient occlusions, (b) specular highlights, (c) dynamic scenes, and (d) fine-grained geometry. The first three issues arise because these factors are not explicitly modeled; incorporating appropriate modeling strategies and richer training data could mitigate them. The last issue likely requires a more powerful geometry encoder. We leave these directions to future work.

MoreComparisons. We presentmorevisualizationresults in Fig. 11 and Fig. 12. For sparse-view inputs, AnySplat delivers higher visual quality, with reliable geometry and finer details, than NoPoSplat [Ye et al. 2024] and Flare [Zhang et al. 2025]. For dense-view inputs, 3D-GS [Kerbl et al. 2023] and Mip-Splatting [Yu et al. 2024a] tend to overfit in the training views, leading to unavoidable artifacts. In contrast, AnySplat consistently produces cleaner renderings with fewer artifacts.

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

GT Views

(a) (b) (c) (d)

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

Predictions

- Fig. 10. Example failure cases. AnySplat exhibits visible artifacts under (a) variable illumination or transient occluders for the Brandenburg Gate (Phototourism [Jin et al. 2021]); (b) specular highlights on the sedan (Ref-NeRF [Verbin et al. 2022]); (c) a dynamic bus scene (DAVIS [Perazzi et al. 2016]); and (d) the bicycle’s thin structures (Mip-NeRF360 [Barron et al. 2022]).

NoPoSplat FLARE Ours GT 3D-GS Mip-Splatting Ours GT

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

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

Fig. 11. Example visualization results on the Mip-NeRF360 [Barron et al. 2022] dataset (bonsai, kitchen, room).

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

ACM Trans. Graph., Vol. 44, No. 6, Article . Publication date: December 2025.

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

NoPoSplat FLARE Ours GT 3D-GS Mip-Splatting Ours GT

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

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

Fig. 12. Example visualization results on the VR-NeRF [Xu et al. 2023] dataset (apartment, raf_furnishedroom, kitchen).

[Figure 384]

[Figure 392]

