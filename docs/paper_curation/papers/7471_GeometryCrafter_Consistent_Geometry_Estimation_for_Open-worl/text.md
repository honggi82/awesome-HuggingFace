## GeometryCrafter: Consistent Geometry Estimation for Open-world Videos with Diffusion Priors

# arXiv:2504.01016v1[cs.GR]1Apr2025

Tian-Xing Xu1 Xiangjun Gao3 Wenbo Hu2† Xiaoyu Li2 Song-Hai Zhang1† Ying Shan2 1 Tsinghua University 2 ARC Lab, Tencent PCG 3 HKUST

Project Page: https://geometrycrafter.github.io

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Camera Pose Estimation and 3D Reconstruction

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

Camera Pose Estimation and 4D Reconstruction

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

Input Video Point Map Prediction Depth/Disparity Maps for Video Generation/Editing

Figure 1. We present GeometryCrafter, a novel approach that estimates temporally consistent, high-quality point maps from open-world videos, facilitating downstream applications such as 3D/4D reconstruction and depth-based video editing or generation.

#### Abstract

Variational Autoencoder (VAE) that learns a latent space agnostic to video latent distributions for effective point map encoding and decoding. Leveraging the VAE, we train a video diffusion model to model the distribution of point map sequences conditioned on the input videos. Extensive evaluations on diverse datasets demonstrate that GeometryCrafter achieves state-of-the-art 3D accuracy, temporal consistency, and generalization capability.

Despite remarkable advancements in video depth estimation, existing methods exhibit inherent limitations in achieving geometric fidelity through the affine-invariant predictions, limiting their applicability in reconstruction and other metrically grounded downstream tasks. We propose GeometryCrafter, a novel framework that recovers highfidelity point map sequences with temporal coherence from open-world videos, enabling accurate 3D/4D reconstruction, camera parameter estimation, and other depth-based applications. At the core of our approach lies a point map

#### 1. Introduction

Inferring 3D geometry from 2D observations remains a long-standing challenge in computer vision, serving as

†Corresponding authors.

a fundamental pillar for numerous applications, ranging from autonomous navigation [14, 52, 72] and virtual reality [27, 93] to 3D/4D reconstruction [30, 41, 68, 77] and generation [64, 88]. However, its inherently ill-posed nature poses persistent difficulties in achieving reliable and consistent geometry estimation from diverse open-world videos.

| |
|---|

[Figure 34]

[Figure 35]

[Figure 36]

Input Video GT Disparity

GT Depth

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Pioneered by Marigold [36], recent methods harness diffusion models [4, 26, 59, 63] to generate affine-invariant depth maps [18, 19, 23, 25] or sequences [31, 62, 80], which is achieved by recasting depth cues as pseudo-RGB frames that are suitable for Variational Autoencoder (VAE) [37] processing. Although these methods exhibit remarkable spatial and temporal fidelity, the compression of unbounded depth values into the fixed input range of the VAE inevitably leads to a non-trivial information loss, especially for distant scene elements, as shown in Fig. 2. Moreover, the absence of camera intrinsics and the presence of unknown shift values impede accurate 3D reconstruction, thereby limiting their utility in downstream applications. Another line of research [5, 56, 69, 70, 87] uses pretrained image foundation models to directly estimate metric depth or point maps. However, neglecting temporal context often induces flickering artifacts when applying these methods to videos.

DepthCrafter DAV

Figure 2. Diffusion-based depth estimation methods, e.g., DepthCrafter [31] and DAV [80], suffer from significant metric errors in distant regions due to the compression of unbounded depth values into the bounded input range of VAEs.

geometric fidelity, and a regularization term penalizing deviations from the original latent distribution. Furthermore, our GeometryCrafter integrates a diffusion U-net that generates point map latents from video latents, forming a robust framework for producing high-fidelity and temporally coherent point maps from open-world videos.

We comprehensively evaluate GeometryCrafter on diverse datasets, ranging from static to dynamic scenes, indoor to outdoor environments, and realistic to cartoonish styles. Our method significantly outperforms existing methods by a large margin, both qualitatively and quantitatively. Extensive ablation studies validate the effectiveness of our proposed components, and demonstrate the applicability of our method to 3D/4D point cloud reconstruction and camera pose estimation. Our contributions are summarized as follows:

In this paper, we propose a novel approach, named GeometryCrafter, to estimate high-fidelity and temporally coherent point maps from open-world videos. These point maps facilitate 3D/4D point cloud reconstruction, camera pose estimation, and the derivation of temporally consistent depth maps and camera intrinsics. Our method exhibits robust zero-shot generalization capabilities by exploiting the inherent video diffusion priors of natural videos. Central to our approach is a novel point map VAE, tailored to effectively encode and decode unbounded 3D coordinates without compressing depth values into a bounded range. It contains a dual-encoder architecture: an encoder inherited from the original video VAE to capture the primary point map information, and a newly designed residual encoder to embed the remaining information in a latent offset. Leveraging this design, we can preserve the latent space analogous to the video VAE by regulating the adjusted latent code with the original video decoder. This analogous latent distribution enables the utilization of pre-trained diffusion weights for robust zero-shot generalizations.

- • We present GeometryCrafter, a novel approach for estimating high-fidelity and temporally coherent geometry from diverse open-world videos.
- • We propose a point map VAE for effective encoding and decoding of point maps, which employs a dual-encoder architecture to maintain the latent space analogous to the inherited video VAE for generalization ability.
- • We introduce the disentangled point map representation and multi-scale depth loss to train the VAE, significantly improving the robustness and fidelity of our method.

For VAE training, we disentangle the point map into log-space depth and diagonal field of view, rather than directly encoding 3D coordinates in the camera coordinate system or adopting a cuboid-based representation as in prior works [69, 70]. This disentangled representation demonstrates enhanced suitability for the VAE to capture the intrinsic structure of the point map, largely attributed to its location invariance and resolution independence. For supervision, we augment the standard reconstruction objective with a normal loss, a multi-scale depth loss to enhance local

#### 2. Related Works

Monocular depth estimation (MDE). MDE methods [1, 3, 15, 17, 40, 43, 44, 53, 79] predict depth maps from single images or videos. To achieve zero-shot generalization, MiDaS [57] introduces affine-invariant supervision and trains on mixing datasets. Depth Anything [81] and its V2 [82] extend this framework to transformer-based architectures [51] and semi-supervised learning, using large-

|[Figure 41]<br><br>| |
|---|---|
| | |

|[Figure 42]<br><br>| |
|---|---|
| | |

Frozen Trainable

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Diffusion Model / Deterministic Model

Offset Latent

Decoded Point Maps

Point Maps

|[Figure 47]<br><br>| |
|---|---|
| | |

|[Figure 48]<br><br>| |
|---|---|
| | |

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Disparity Latent Point Map Latent

Decoded Disparity Maps

Disparity Maps

Figure 3. Architecture of our point map VAE. The point map VAE encodes and decodes point maps with unbounded values, alleviating the inaccurate prediction in distant regions. We adopt a dual-encoder design: the native encoder ESVD inherited from SVD captures normalized disparity maps, while a residual encoder Eϵ embeds remaining information as an offset. It preserves the original latent space by regulating the latents via the original decoder DSVD, enabling the utilization of pretrained diffusion priors. A point map decoder Dpmap recovers the final point maps from the latent codes.

phy visually hides secret information within existing features [2, 33, 95]. Previous works have demonstrated the capacity of neural networks to embed visual data within images or videos, such as invertible down-scaling [75], grayscaling [74], and mono-nizing [29]. The most relevant work to ours is LayerDiffuse [90], which conceals image transparency information within a small perturbation in the latent space of Stable Diffusion [59]. Adhering to the same insight, we encode point maps into the latent space of diffusion models while preserving the underlying distribution, facilitating the utilization of pretrained diffusion models for geometry estimation.

scale unlabeled images for improved generalization. Pioneered by Marigold [36], recent works [18, 19, 23, 25, 55] adapt pretrained image diffusion models [59] to MDE by converting depth maps to pseudo-RGB representations and exclusively finetuning the U-net on depth latent codes, achieving superior quality and robustness. To generalize to videos, previous methods [10, 39, 47, 73, 83, 92] employ test-time optimization, memory mechanisms, or stabilization networks for temporal coherence, whereas recent studies [31, 62, 80] finetune video diffusion models [4] to yield high-quality, temporally consistent depth sequences. However, these methods ignore the camera intrinsic estimation and provide only affine-invariant depth, which is scale and shift ambiguous, hindering 3D accuracy and downstream applications that require projection into 3D space.

#### 3. Method

Given an input RGB video v ∈ RT×H×W×3, we aim to predict a temporally consistent point map sequence p ∈ RT×H×W×3 alongside a valid mask m ∈ [0,1]T×H×W to exclude undefined regions (e.g., sky). Each point map contains the 3D coordinates p = (xp, yp, zp)T in the camera coordinate system for every pixel. To this end, we propose GeometryCrafter, a novel approach that leverages video diffusion models (VDMs) for robust point map estimation from open-world videos. We model the joint distribution P(p, m|v) in the latent space. While VDMs’ native VAE effectively encodes video frames and masks, accurate point map representation necessitates a dedicated VAE tailored for geometric encoding and decoding.

Monocular geometry estimation (MGE). To overcome these limitations, MGE methods jointly infer camera parameters and metric or up-to-scale depth maps. LeRes [85, 86] utilizes 3D point cloud encoders to recover missing shift and focal parameters during depth estimation. UniDepth [56] decouples camera parameter prediction from depth estimation via a pseudo-spherical 3D representation and a camera self-prompting mechanism. DepthPro [5] introduces a ViT-based design [51] for high-resolution depth estimation, coupled with a dedicated image encoder for focal length prediction. DUSt3R [70] projects two-view images or identical pairs to scale-invariant point maps in camera space, facilitating the derivation of camera intrinsic and depth maps. MoGe [69] employs affine-invariant point maps to mitigate focal-distance ambiguity, achieving state-of-the-art performance. Besides, Metric3D [87] and its V2 [28] rely on user-provided camera parameters to estimate metrically accurate depth maps. However, these approaches are restricted to static images and incur flickering artifacts when directly applied to video sequences.

##### 3.1. Architecture of Point Map VAE

Existing diffusion-based depth estimation methods [31, 80] simply employ the native VAE to encode and decode only partial information from the point maps, i.e. the normalized disparity maps xdisp:

xdisp − min(xdisp)

max(xdisp) − min(xdisp) − 1, xdisp = b · f /zp,

xdisp = 2 ×

(1)

Steganography and information hiding. Steganogra-

where b is the baseline, f is the focal length, and zp is the z-coordinate of the point map p. However, such normalization often misestimates depths in distant regions (Fig. 2), resulting in geometric distortions due to compressing unbounded depths into the VAE’s fixed input range.

[Figure 54]

Spatial Layer (Stage 1 only)

[Figure 55]

[Figure 56]

[Figure 57]

Temporal Layer (Stage 1+2)

[Figure 58]

C Concatenate

[Figure 59]

Per-Frame Priors

To this end, we propose a point map VAE that directly handles point maps over the unbounded range [0,+∞]. Crucially, its latent distribution should be tightly aligned with that of the native VAE to fully exploit pre-trained VDMs. Inspired by LayerDiffuse [90], we propose a dualencoder architecture: the inherited native encoder ESVD captures the primary point map features, while a newly designed residual encoder Eϵ encodes remaining information

[Figure 60]

[Figure 61]

C

Add Noise

Video Frames

[Figure 62]

|[Figure 63]<br><br>|
|---|

| | |
|---|---|
| | |

[Figure 64]

[Figure 65]

- as an offset (see Fig. 3). Given that the normalized disparity maps xdisp encapsulate significant relative depth cues, we employ ESVD on xdisp and harness Eϵ to embed the residual information into the offset. The final point map latent is obtained by their summation:

Frozen Trainable

Ground Truth Point Maps

Figure 4. Diffusion UNet. We jointly condition the diffusion model on video latents and per-frame geometry priors from an image MGE model Mimg. The geometry is encoded into latent space via our point map VAE, while the video latents are obtained from the native VAE.

zpmap = ESVD( xdisp) + Eϵ(p,m, xdisp), (2)

where θdiag = √W2 + H2 / 2f denotes the diagonal field of view, a constant map for all points in a frame. Since pdec is independent of spatial location, it is more suitable for our VAE to learn an effective latent distribution. Moreover, this formulation enables us to train our network only on fixedresolution videos, while generalizing to varying resolutions and aspect ratios, owing to the invariance of θdiag. The original point map p can be effortlessly recovered from pdec via the inverse perspective transformation.

This dual-encoder architecture allows us to explicitly regularize the latent space of zpmap to avoid disrupting the original latent distribution. Considering most VAEs in VDM are diagonal Gaussian models (i.e. mean and variance), we apply the offset solely to the mean, retaining the original variance for simplicity. For decoding, we design a dedicated decoder Dpmap to reconstruct both the point map p and the valid mask m:

p, m = Dpmap(zpmap). (3)

Loss functions. To train the point map VAE, we define reconstruction loss Lrecon as the L1 norm between the decoded depth and diagonal field of view and their ground truth counterparts. Besides, we also impose a mask loss Lmask to exclude undefined regions, e.g. sky, as the L2 norm between the predicted and ground truth valid masks. To promote surface quality, we introduce a normal loss Ln that supervises the normal maps derived from the reconstructed point maps and the ground truth, as well as a multi-scale depth loss Lms that measures the alignment between reconstructed and ground truth depth maps within local regions, inspired by MoGe [69]. Importantly, to regularize our latent space agnostic to the original SVD’s latent distribution, we employ a loss term Lidentity to penalize the latent deviation:

To ensure temporal consistency, we employ temmporal layers in the decoder to capture the temporal dependencies across frames.

##### 3.2. Training of Point Map VAE

Point map representation. The points p = (xp, yp, zp)T are scattered non-uniformly across the view frustum, resulting in a complex spatial distribution that poses a challenge to deep networks in capturing their inherent structure. To mitigate this, existing point map estimation methods [69, 70] assume a centered camera principal point and remap depth values into log-space, thereby projecting the points into a cuboid domain:

###### pcuboid = [xp/zp, yp/zp, log zp]. (4)

Lidentity = || xdisp − DSVD(zpmap)||22. (6) The final training objective LVAE is defined as:

However, this representation is suboptimal for our point map VAE. In particular, the first two channels of pcuboid encode ray directions from the camera center to each pixel, conveying location-specific information that diverges from the translation-invariant nature of RGB features. To address this discrepancy, we propose decoupling the point map into:

LVAE = Lrecon + Lms + λnLn

+Lidentity + λmaskLmask.

Lpmap

(7) Please refer to the supplementary material for more details on the loss functions.

pdec = [θdiag, log zp], (5)

- Table 1. Evaluation on point map estimation. Results are aligned with the ground truth by optimizing a shared scale factor across the entire video. Relp and δp are in percentage. The best and second-best results are highlighted in bold and underline, respectively. “G” denotes the diffusion version of our model and “D” denotes the deterministic variant.

GMU Kitchen [21] Monkaa [48] Sintel [7] ScanNet [12] DDAD [24] KITTI [20] DIODE [65] Avg.

Method

###### Relp↓ δp↑ Relp↓ δp↑ Relp↓ δp↑ Relp↓ δp↑ Relp↓ δp↑ Relp↓ δp↑ Relp↓ δp↑ Rank↓

DUSt3R [70]† 22.2 68.2 37.0 45.1 43.9 35.6 13.3∗ 87.7∗ 37.8 37.3 17.2 87.8 20.0 85.3 6.3 MonST3R [89]† 24.1 64.4 40.2 32.6 40.0 34.1 13.6∗ 87.2∗ 40.7 25.9 24.0 58.1 22.2 79.4 7.4 MonST3R [89]‡ 11.4 91.5 36.2 42.6 38.6 35.9 6.13∗ 97.6∗ 40.0 29.3 25.0 58.7 22.2 79.4 5.4 UniDepth [56] 9.96 94.1 23.0 65.6 30.9 50.6 6.54∗ 98.4∗ 23.0 64.9 4.24 99.3 16.1 88.0 2.9 DepthPro [5] 14.0 86.5 29.5 50.4 45.0 36.3 10.5∗ 93.6∗ 39.8 43.1 12.8 93.6 18.6 87.1 5.6 MoGe [69] 21.3 69.1 28.0 58.1 31.2 52.0 13.5 88.0 16.0 85.5 8.51 95.7 13.5 93.5 4.4

Ours(D) 8.88 94.3 18.8 79.7 25.9 62.8 8.92 96.4 15.6 89.0 6.73 98.4 13.0 92.8 2.0 Ours(G) 8.52 94.3 20.5 75.5 25.6 64.9 9.16 96.0 15.0 90.6 6.34 98.7 13.1 92.7 1.9

Reld↓ δd↑ Reld↓ δd↑ Reld↓ δd↑ Reld↓ δd↑ Reld↓ δd↑ Reld↓ δd↑ Reld↓ δd↑ Rank↓ DUSt3R [70]† 21.6 64.0 35.8 41.9 41.3 36.0 13.1∗ 84.5∗ 32.3 46.5 10.9 87.9 15.9 85.4 6.7 MonST3R [89]† 22.6 61.9 38.8 31.4 37.5 33.4 13.3∗ 84.4∗ 30.7 46.4 9.06 91.8 17.4 80.8 6.6 MonST3R [89]‡ 8.91 90.7 33.9 41.3 35.9 36.1 5.18∗ 97.1∗ 31.9 46.1 13.4 80.4 17.4 80.8 5.3 UniDepth [56] 8.11 93.7 20.8 60.0 28.4 48.5 5.32∗ 98.0∗ 22.9 63.4 3.45 99.1 11.5 89.9 2.6 DepthPro [5] 14.0 83.1 28.4 45.2 43.2 34.3 10.0∗ 90.7∗ 38.3 40.6 9.47 91.5 12.6 87.4 6 MoGe [69] 20.6 64.7 25.7 54.8 29.2 49.0 13.3 84.9 14.6 85.2 7.69 94.1 8.13 93.5 4.3 Ours(D) 8.51 93.4 16.1 77.1 22.2 65.7 7.88 95.5 12.3 88.4 5.88 97.6 10.2 92.1 2.3 Ours(G) 8.30 93.6 18.3 71.5 22.6 63.7 8.39 95.0 12.0 90.4 5.44 98.2 10.0 92.4 2.1

∗: Not strictly zero-shot (trained on ScanNet [12] or ScanNet++ [84]); †: Inference with duplicated frames; ‡: Post-optimization with external data.

##### 3.3. Diffusion UNet

Since our point map VAE is meticulously designed and regularized to align closely with the original SVD’s latent distribution, we can train a diffusion UNet to estimate point maps from videos with only synthetic data, using the pretrained generative prior of video diffusion models. Although it significantly alleviates the issue of lacking highquality point map annotations in real-world videos, the synthetic data still suffers from limited diversity in camera intrinsics, which may degrade the generalization ability of diagonal field-of-view prediction on real-world scenarios. To mitigate this, alongside video latents, we propose the integration of per-frame geometry priors as conditioning inputs within the diffusion UNet, as shown in Fig. 4. We employ our point map VAE to encode the per-frame point maps predicted by MoGe [69] into the latent space to act as geometry priors that provide strong camera-intrinsic clues, although they may suffer from inaccuracies and flickering.

Following DepthCrafter [31], we train the diffusion UNet with the EDM [35] pre-conditioning and noise schedule, and adopt a multi-stage training strategy to capture long temporal context under GPU memory constraints. After training, the UNet can process videos with varying lengths (e.g. 1 to 110 frames) at a time, and we adopt the stitching inference strategy [31] to handle videos with arbitrary lengths. Besides, inspired by recent advancements [19, 25] in reformulating the diffusion process into a deterministic single-step framework for depth estimation, we also train a deterministic variant by removing the noisy latent from input.

#### 4. Experiments

##### 4.1. Implementation Details

We build GeometryCrafter upon the SVD [4] framework. The residual encoder and point decoder in the point map VAE adopt the same architecture as in SVD’s VAE, supplemented by zero convolution [91] in the output layers. We collected 14 synthetic RGBD datasets [8, 16, 22, 32, 34, 42, 49, 50, 58, 60, 66, 67, 71, 94], comprising 1.85M frames, for training. Among these, 11 datasets can form 12K video clips with up to 150 frames each. For training stability, we normalize point clouds with a shared scale factor across frames, yielding up-to-scale point clouds akin to structurefrom-motion [61]. We first train the point map VAE from scratch on RGBD images with an AdamW [46] optimizer at a learning rate of 10−4 for 40K iterations, then finetune on video data for an additional 20K iterations. The diffusion UNet is finetuned with a learning rate of 10−5 for 40K and 30K iterations in two stages. All experiments are conducted on 8 GPUs and take about 3 days. Further details are in the supplementary material.

##### 4.2. Quantitative and Qualitative Evaluation

Evaluation protocol. For evaluation, we employ seven datasets unseen during training: GMU Kitchens [21] and ScanNet [12] are captured with Kinect for indoor scenes; DDAD [24] and KITTI [20] are collected via lidar sensors for outdoor driving; Monkaa [48] and Sintel [7] are synthetic datasets with precise depth annotations and challenging dynamics; and DIODE [65] is a high-resolution

| |
|---|
| |
| |

| |
|---|
| |
| |

| |
|---|
| |
| |

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

| |
|---|
| |
| |

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Disp. Max

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Disp. Min

Ours

Input Video UniDepth MoGe

[Figure 79]

[Figure 80]

[Figure 81]

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

Ours

UniDepth MoGe

- Figure 5. Qualitative comparison of point map estimation. Disparity maps are derived from estimated point maps via Eq. (1). The green boxes highlight temporal profiles of the disparity maps, sliced along the time axis at the green lines. Zoom in for better visualization.

- Table 2. Evaluation on depth map estimation. Results are aligned with the ground truth by optimizing a shared scale factor and shift across the entire video. Reld and δd are in percentage. The best and second-best results are highlighted in bold and underline, respectively. “G” denotes the diffusion version of our model and “D” denotes the deterministic variant.

GMU Kitchen [21] Monkaa [48] Sintel [7] Scannet [12] DDAD [24] KITTI [20] DIODE [65] Avg.

Method

###### Reld↓ δd↑ Reld↓ δd↑ Reld↓ δd↑ Reld↓ δd↑ Reld↓ δd↑ Reld↓ δd↑ Reld↓ δd↑ Rank↓

DA [81] 18.0 68.2 24.1 62.1 37.2 59.8 11.3 87.8 13.4 85.1 8.70 92.4 6.94 94.6 4.1 DA V2 [82] 19.3 65.4 24.0 61.5 40.6 55.0 12.3 85.1 13.9 84.7 11.1 87.0 6.94 95.0 5.1 ChronoDepth [62] 20.1 65.2 35.1 52.6 45.1 57.9 14.3 81.4 34.6 45.8 15.0 79.8 12.2 90.6 6.9 DepthCrafter [31] 13.8 80.6 23.4 73.8 30.5 67.0 11.3 87.3 15.6 80.7 9.96 89.6 12.6 86.2 4.9 DAV [80] 10.8 89.4 19.0 72.3 35.7 67.3 8.83 92.7 12.3 85.4 7.13 95.3 7.47 93.7 3.1

Ours(D) 8.28 93.2 12.0 83.5 16.3 74.3 7.27 96.1 13.4 86.2 5.60 97.7 7.00 96.2 1.9 Ours(G) 8.03 94.0 13.0 80.5 16.9 73.2 7.57 95.9 12.7 87.5 5.25 98.3 7.03 96.1 2.0

image dataset with far-range depth maps. Besides, we also qualitatively evaluate on DAVIS [54], DL3DV [45], Sora [6]-generated, and open-world videos. To assess upto-scale point map quality, we use the relative point error Relp and percentage of inliers δp (threshold 0.25), following MoGe [69]. We align predicted point maps with ground truth by optimizing a shared scale factor across the entire video for all methods. We also evaluate derived depth sequences using the absolute relative error Reld and the inlier percentage δd (threshold 1.25), following [31].

uated by inputting two identical frames. For MonST3R, we also evaluate with its post-processing, which requires external optical flows to refine global point clouds and poses. As shown in Tab. 1, our method outperforms others on most benchmarks, with substantial gains on the challenging Monkaa and Sintel datasets. Although UniDepth shows better performance on KITTI (likely due to training on DrivingStereo [78] with a shared LiDAR sensor), our approach attains a superior average rank. For the image benchmark DIODE, our method still achieves competitive performance compared to methods specialized for static images. Notably, some methods are trained on ScanNet [12] or ScanNet++ [84], violating the zero-shot evaluation, yet our method sustains comparable accuracy on ScanNet. Moreover, visual comparisons in Fig. 5 indicate that only our method can produce temporally consistent point maps with

Evaluation on point maps. We compare our method with representative point map estimation approaches, e.g., DUSt3R [70], MonST3R [89], UniDepth [56], DepthPro [5], and MoGe [69]. Among them, DUSt3R and MonST3R are designed for two-view scenarios, addressing static and dynamic scenes, respectively, and are eval-

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

[Figure 98]

Depth Min Depth Max

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

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

Depth Min Depth Max

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

Input Ground Truth DepthCrafter DAV Ours

- Figure 6. Qualitative comparison of depth map estimation. We transform point maps and disparity maps into metric depth maps for better visualization of distant regions. Zoom in for better visualization.

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Input Ground Truth Ours w/o point VAE

- Figure 7. Comparison on disparity (top) and depth (bottom) quality between our full model and the w/o point map VAE variant.

Table 3. Ablation study on the effectiveness of point map VAE.

GMU Kitchen Monkaa Sintel Scannet KITTI

###### Reld↓ δd↑ Reld↓ δd↑ Reld↓ δd↑ Reld↓ δd↑ Reld↓ δd↑

w/o 9.50 90.8 16.1 79.2 25.2 72.8 8.11 95.0 5.00 97.9 w/ 8.03 94.0 13.0 80.5 16.9 73.2 7.57 95.9 5.25 98.3

ground truth exceeds the LiDAR sensor’s range.

##### 4.3. Ablation Study

fine-grained details, while others (UniDepth and MoGe) exhibit issues like flickering or blurred details.

Effectiveness of point map VAE. We conduct ablation studies by removing the point map VAE and using SVD’s VAE to encode normalized disparity maps while keeping other components unchanged and retraining the model. As shown in Tab. 3, the estimated depth maps exhibit a significant performance drop across various datasets, except for KITTI, where the ground truth is constrained by the LiDAR sensor’s range. The visual comparison in Fig. 7 reveals that the performance decline is due to information loss from compressing unbounded depth values into a bounded range, leading to the neglect of distant objects.

Evaluation on depth maps. To compare our method with cutting-edge monocular depth estimation methods, e.g., ChronoDepth [62], DepthCrafter [31], DAV [80], and DepthAnything (DA) V1 [81] and V2 [82], we follow the evaluation protocol in [31], except for a center crop to meet the aspect ratio requirement (0.5 to 2). As shown in Tab. 2, our method achieves the best performance on almost all video datasets and remains competitive even on the image dataset DIODE. The qualitative comparison in Fig. 6 demonstrates that our method generates superior depth maps and point clouds, e.g., the potato chip bucket and the plant in the first two examples. In driving scenarios, such as the third example, DepthCrafter and DAV predict infinity values for distant buildings, resulting in missing structures, whereas our method consistently produces regular structures and plausible depth values, even when the

Components in point map VAE. We perform ablation studies to examine the effectiveness of the point map representation, multi-scale loss Lms, temporal layers in the decoder, and latent alignment. As shown in Tab. 4, the decoupled point map representation Eq. (5) markedly enhances reconstruction fidelity. Results in the second to fourth rows also highlight the importance of multi-scale supervision in

- Table 4. VAE reconstruction performance with different components. Light gray background highlights our final VAE configuration.

|Representation Lms Temporal layers Latent alignment<br><br>|Scannet Relp↓ δp↑ Reld↓ δd↑<br><br>|Sintel Relp↓ δp↑ Reld↓ δd↑|Monkaa Relp↓ δp↑ Reld↓ δd↑<br><br>|
|---|---|---|---|
|Eq. (4) ✓ ✓<br>Eq. (5) ✓ ✓ Eq. (5) ✓ Eq. (5) ✓ ✓ ✓ Eq. (5) ✓ ✓<br><br><br>|7.67 99.8 2.03 99.8<br><br>1.63 99.8 1.51 99.7<br><br>2.95 99.8 2.31 99.6 1.65 99.9 1.47 99.9<br><br><br>1.95 99.9 1.77 99.9|8.25 94.4 6.46 94.8<br><br>4.24 97.8 4.02 97.9<br><br>5.32 97.3 4.72 97.5<br><br><br>3.45 98.1 3.06 98.1<br><br>4.48 98.0 3.78 97.9<br><br><br>|6.05 99.5 3.93 99.4 2.19 99.6 2.07 99.5 2.83 99.5 2.68 99.4 2.01 99.5 1.84 99.4 2.94 99.1 2.64 98.9<br><br>|

- Table 5. UNet prediction performance with different components. Light gray background highlights our final UNet configuration.

|Latent alignment Per-frame geometry prior<br><br>|GMU Kitchen Relp↓ δp↑ Reld↓ δd↑|Sintel Relp↓ δp↑ Reld↓ δd↑<br><br>|DDAD Relp↓ δp↑ Reld↓ δd↑|
|---|---|---|---|
|✓<br><br>✓<br><br>✓ ✓<br><br>|9.51 93.6 9.26 92.6 12.9 88.7 12.0 84.7 8.52 94.3 8.30 93.6<br><br>|25.4 65.5 23.0 61.7 34.4 38.8 24.7 57.9 25.6 64.9 22.6 63.7<br><br>|14.3 90.1 12.5 89.6 25.5 58.7 15.7 78.3<br><br>15.0 90.6 12.0 90.4<br><br><br>|

DUSt3R 22.2 68.2 21.6 64.0 43.9 35.6 41.3 36.0 37.8 37.3 32.3 46.5 Ours(G) + DUSt3R 12.2 90.4 11.5 88.1 34.4 39.9 26.2 55.8 28.7 48.8 15.9 80.9

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Input w.o. Latent Alignment w. Latent Alignment

Figure 8. Effectiveness of latent alignment.

Figure 9. Application of depth-conditioned video generation. The prompt is “a car is drifting on roads, snowy day, artstation”.

the spatial domain and contextual information in the temporal domain. Eliminating the latent alignment component not only increases point map errors (see the last two rows of Tab. 4), but also hinders effectively leveraging video diffusion priors. As shown in Tab. 5 and Fig. 8, latent alignment substantially improves the quality and robustness of point map predictions.

and track them via SpaTracker [76]. Finally, we optimize the camera poses with the established correspondences by 3D geometric constraints, taking only a few minutes to converge. Examples of 3D/4D reconstruction are shown in Fig. 1 and supplementary materials.

UNet design. We investigate the impact and robustness of per-frame geometry priors derived from MoGe [69] by excluding them from the UNet input and replacing MoGe with DUSt3R [70]. As shown in Tab. 5, per-frame priors benefit the model across diverse scenarios by compensating for limited camera intrinsics in the training data. Moreover, replacing MoGe with DUSt3R also consistently improves performance, confirming the robustness of our method to different priors. Besides, we present two variants of the UNet: one with the diffusion framework (noted as Ours(G)) and the other with a deterministic scheme (noted as Ours(D)). As shown in Tab. 1 and Tab. 2, the deterministic approach exhibits slightly lower accuracy but achieves a 1.1× acceleration in inference speed, e.g. 4.1 v.s. 3.7 FPS at a 448×768 resolution on our experimental setup. Users may choose one of the two variants based on their requirements for speed or accuracy.

Depth-conditioned video generation. Depth sequences are pivotal to controllable video generation, capturing the inherent 3D structures of videos. Our consistent depth maps serve directly as conditioning inputs in existing depthdriven methods (e.g., Control-A-Video [9]), enabling creative outputs as shown in Fig. 9.

#### 5. Conclusion

We present GeometryCrafter, a novel method that estimates temporally consistent, high-quality point maps from openworld videos, facilitating downstream applications such as 3D/4D reconstruction and depth-based video editing or generation. Our core design is a point map VAE that learns a latent space agnostic to original video latent distribution, enabling effective encoding and decoding of unbounded point map values. We also introduce a decoupled point map representation to eliminate the location-dependent characteristics of point maps, enhancing the robustness to resolutions and aspect ratios. Furthermore, we integrate a perframe geometry prior conditioned diffusion model to model the distribution of point sequences conditioned on the input videos. Comprehensive evaluations confirm that our method outperforms prior methods in performance and generalization. Its main limitation is relatively high computa-

##### 4.4. Applications

3D/4D reconstruction. With our temporally consistent, high-quality point maps, we enable 3D/4D reconstruction, whose cornerstone is the camera pose estimation. To this end, if dynamic objects exist, we first obtain their masks using SegmentAnything [38] and XMem [11]. Then, we detect interest points in the static regions with SuperPoint [13]

tional and memory overhead due to the large model size.

#### Acknowledgement

Tian-Xing Xu completed this work during his internship

- at Tencent ARC Lab. The project was supported by the Tsinghua-Tencent Joint Laboratory for Internet Innovation Technology. References

- [1] Shubhra Aich, Jean Marie Uwabeza Vianney, Md Amirul Islam, and Mannat Kaur Bingbing Liu. Bidirectional attention network for monocular depth estimation. In ICRA, 2021. 2
- [2] Shumeet Baluja. Hiding images in plain sight: Deep steganography. Advances in neural information processing systems, 30, 2017. 3
- [3] Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. Adabins: Depth estimation using adaptive bins. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4009–4018, 2021. 2
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3, 5
- [5] Aleksei Bochkovskii, Ama¨el Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. arXiv preprint arXiv:2410.02073, 2024. 2, 3, 5, 6
- [6] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators,

2024. 6, 3

- [7] D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black. A naturalistic open source movie for optical flow evaluation. In European Conf. on Computer Vision (ECCV), pages 611–

625. Springer-Verlag, 2012. 5, 6, 1

- [8] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2. arXiv preprint arXiv:2001.10773, 2020. 5, 1
- [9] Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models,

2023. 8

- [10] Yuhua Chen, Cordelia Schmid, and Cristian Sminchisescu. Self-supervised learning with geometric constraints in monocular video: Connecting flow, depth, and camera. In ICCV, 2019. 3
- [11] Ho Kei Cheng and Alexander G Schwing. Xmem: Longterm video object segmentation with an atkinson-shiffrin memory model. In European Conference on Computer Vision, pages 640–658. Springer, 2022. 8, 2
- [12] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017. 5, 6, 1

- [13] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 224–236, 2018. 8, 2
- [14] Xingshuai Dong, Matthew A Garratt, Sreenatha G Anavatti, and Hussein A Abbass. Towards real-time monocular depth estimation for robotics: A survey. IEEE Transactions on Intelligent Transportation Systems, 23(10):16940–16961,

2022. 2

- [15] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. Advances in neural information processing systems, 27, 2014. 2
- [16] Michael Fonder and Marc Van Droogenbroeck. Mid-air: A multi-modal dataset for extremely low altitude drone flights. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 0–0, 2019. 5, 1
- [17] Huan Fu, Mingming Gong, Chaohui Wang, Kayhan Batmanghelich, and Dacheng Tao. Deep ordinal regression network for monocular depth estimation. In CVPR, 2018. 2
- [18] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. In ECCV, 2024. 2, 3
- [19] Gonzalo Martin Garcia, Karim Abou Zeid, Christian Schmidt, Daan de Geus, Alexander Hermans, and Bastian Leibe. Fine-tuning image-conditional diffusion models is easier than you think. arXiv preprint arXiv:2409.11355,

2024. 2, 3, 5

- [20] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. IJRR,

2013. 5, 6, 1

- [21] Georgios Georgakis, Md Alimoor Reza, Arsalan Mousavian, Phi-Hung Le, and Jana Koˇseck´a. Multiview rgb-d dataset for object instance detection. In 2016 Fourth international conference on 3D vision (3DV), pages 426–434. IEEE, 2016. 5, 6, 1
- [22] Jose L G´omez, Manuel Silva, Antonio Seoane, Agn`es Borr´as, Mario Noriega, Germ´an Ros, Jose A IglesiasGuitian, and Antonio M L´opez. All for one, and one for all: Urbansyn dataset, the third musketeer of synthetic driving scenes. arXiv preprint arXiv:2312.12176, 2023. 5, 1
- [23] Ming Gui, Johannes Schusterbauer, Ulrich Prestel, Pingchuan Ma, Dmytro Kotovenko, Olga Grebenkova, Stefan Andreas Baumann, Vincent Tao Hu, and Bj¨orn Ommer. Depthfm: Fast monocular depth estimation with flow matching. arXiv preprint arXiv:2403.13788, 2024. 2, 3
- [24] Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos, and Adrien Gaidon. 3d packing for self-supervised monocular depth estimation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 5, 6, 1
- [25] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Liu, Bingbing Liu, and YingCong Chen. Lotus: Diffusion-based visual foundation

- model for high-quality dense prediction. arXiv preprint arXiv:2409.18124, 2024. 2, 3, 5
- [26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. 2
- [27] Fa-Ting Hong, Longhao Zhang, Li Shen, and Dan Xu. Depth-aware generative adversarial network for talking head video generation. In CVPR, 2022. 2
- [28] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE TPAMI, 46(12):10579–10596, 2024. 3
- [29] Wenbo Hu, Menghan Xia, Chi-Wing Fu, and Tien-Tsin Wong. Mononizing binocular videos. ACM Transactions on Graphics (TOG), 39(6):1–16, 2020. 3
- [30] Wenbo Hu, Yuling Wang, Lin Ma, Bangbang Yang, Lin Gao, Xiao Liu, and Yuewen Ma. Tri-miprf: Tri-mip representation for efficient anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19774–19783, 2023. 2
- [31] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. In CVPR, 2025. 2, 3, 5, 6, 7, 1
- [32] Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra Ahuja, and Jia-Bin Huang. Deepmvs: Learning multiview stereopsis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2821–2830,

2018. 5, 1

- [33] Junpeng Jing, Xin Deng, Mai Xu, Jianyi Wang, and Zhenyu Guan. Hinet: Deep image hiding by invertible network. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4733–4742, 2021. 3
- [34] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13229–13239, 2023. 5, 1
- [35] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 5, 2
- [36] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In CVPR, 2024. 2, 3
- [37] DP Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 2
- [38] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 8, 2
- [39] Johannes Kopf, Xuejian Rong, and Jia-Bin Huang. Robust consistent video depth estimation. In CVPR, 2021. 3

- [40] Jin Han Lee, Myung-Kyu Han, Dong Wook Ko, and Il Hong Suh. From big to small: Multi-scale local planar guidance for monocular depth estimation. arXiv preprint arXiv:1907.10326, 2019. 2
- [41] Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. arXiv preprint arXiv:2405.17421, 2024. 2
- [42] Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhenzhi Wang, Dahua Lin, and Bo Dai. Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3205–3215, 2023. 5, 1
- [43] Zhenyu Li, Zehui Chen, Xianming Liu, and Junjun Jiang. Depthformer: Exploiting long-range correlation and local information for accurate monocular depth estimation. Machine Intelligence Research, 20(6):837–854, 2023. 2
- [44] Zhenyu Li, Xuyang Wang, Xianming Liu, and Junjun Jiang. Binsformer: Revisiting adaptive bins for monocular depth estimation. IEEE Transactions on Image Processing, 2024. 2
- [45] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024. 6, 5
- [46] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5, 2
- [47] Xuan Luo, Jia-Bin Huang, Richard Szeliski, Kevin Matzen, and Johannes Kopf. Consistent video depth estimation. TOG (Proceedings of ACM SIGGRAPH), 39(4), 2020. 3
- [48] Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4040–4048, 2016. 5, 6, 1
- [49] Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andr´es Bruhn. Spring: A high-resolution highdetail dataset and benchmark for scene flow, optical flow and stereo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4981– 4991, 2023. 5, 1
- [50] Simon Niklaus, Long Mai, Jimei Yang, and Feng Liu. 3d ken burns effect from a single image. ACM Transactions on Graphics (ToG), 38(6):1–15, 2019. 5, 1
- [51] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, ShangWen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. In TMLR, 2024. 2, 3
- [52] Dennis Park, Rares Ambrus, Vitor Guizilini, Jie Li, and Adrien Gaidon. Is pseudo-lidar needed for monocular 3d

- object detection? In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3142–3152, 2021. 2
- [53] Vaishakh Patil, Christos Sakaridis, Alexander Liniger, and Luc Van Gool. P3depth: Monocular depth estimation with a piecewise planarity prior. In CVPR, 2022. 2
- [54] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Computer Vision and Pattern Recognition, 2016. 6
- [55] Duc-Hai Pham, Tung Do, Phong Nguyen, Binh-Son Hua, Khoi Nguyen, and Rang Nguyen. Sharpdepth: Sharpening metric depth predictions using diffusion distillation. arXiv preprint arXiv:2411.18229, 2024. 3
- [56] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10106–10116, 2024. 2, 3, 5, 6, 1
- [57] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020. 2
- [58] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10912–10922, 2021. 5, 1
- [59] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3
- [60] German Ros, Laura Sellart, Joanna Materzynska, David Vazquez, and Antonio M Lopez. The synthia dataset: A large collection of synthetic images for semantic segmentation of urban scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3234–3243,

2016. 5, 1

- [61] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In CVPR, 2016. 5
- [62] Jiahao Shao, Yuanbo Yang, Hongyu Zhou, Youmin Zhang, Yujun Shen, Vitor Guizilini, Yue Wang, Matteo Poggi, and Yiyi Liao. Learning temporally consistent video depth from video diffusion priors. arXiv preprint arXiv:2406.01493,

2024. 2, 3, 6, 7

- [63] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 2
- [64] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. arXiv preprint arXiv:2411.04928, 2024. 2
- [65] Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z. Dai, Andrea F. Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R. Walter, and Gregory Shakhnarovich. DIODE: A Dense Indoor and Out-

- door DEpth Dataset. CoRR, abs/1908.00463, 2019. 5, 6, 1
- [66] Kaixuan Wang and Shaojie Shen. Flow-motion and depth network for monocular stereo and beyond. IEEE Robotics and Automation Letters, 5(2):3307–3314, 2020. 5, 1
- [67] Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng, Kaiyong Zhao, and Xiaowen Chu. Irs: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation. In 2021 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6. IEEE, 2021. 5, 1
- [68] Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. arXiv preprint arXiv:2407.13764,

2024. 2

- [69] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. arXiv preprint arXiv:2410.19115, 2024. 2, 3, 4, 5, 6, 8, 1
- [70] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024. 2, 3, 4, 5, 6, 8
- [71] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909–4916. IEEE, 2020. 5, 1
- [72] Yan Wang, Wei-Lun Chao, Divyansh Garg, Bharath Hariharan, Mark Campbell, and Kilian Q Weinberger. Pseudolidar from visual depth estimation: Bridging the gap in 3d object detection for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8445–8453, 2019. 2
- [73] Yiran Wang, Min Shi, Jiaqi Li, Zihao Huang, Zhiguo Cao, Jianming Zhang, Ke Xian, and Guosheng Lin. Neural video depth stabilizer. In ICCV, 2023. 3
- [74] Menghan Xia, Xueting Liu, and Tien-Tsin Wong. Invertible grayscale. ACM Transactions on Graphics (TOG), 37(6):1– 10, 2018. 3
- [75] Mingqing Xiao, Shuxin Zheng, Chang Liu, Yaolong Wang, Di He, Guolin Ke, Jiang Bian, Zhouchen Lin, and Tie-Yan Liu. Invertible image rescaling. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part I 16, pages 126–144. Springer,

2020. 3

- [76] Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker: Tracking any 2d pixels in 3d space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20406–20417, 2024. 8, 3
- [77] Tian-Xing Xu, Wenbo Hu, Yu-Kun Lai, Ying Shan, and Song-Hai Zhang. Texture-gs: Disentangling the geometry and texture for 3d gaussian splatting editing. In European

Conference on Computer Vision, pages 37–53. Springer,

2024. 2

- [78] Guorun Yang, Xiao Song, Chaoqin Huang, Zhidong Deng, Jianping Shi, and Bolei Zhou. Drivingstereo: A large-scale dataset for stereo matching in autonomous driving scenarios. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 899–908, 2019. 6
- [79] Guanglei Yang, Hao Tang, Mingli Ding, Nicu Sebe, and Elisa Ricci. Transformer-based attention networks for continuous pixel-wise prediction. In ICCV, 2021. 2
- [80] Honghui Yang, Di Huang, Wei Yin, Chunhua Shen, Haifeng Liu, Xiaofei He, Binbin Lin, Wanli Ouyang, and Tong He. Depth any video with scalable synthetic data. arXiv preprint arXiv:2410.10815, 2024. 2, 3, 6, 7
- [81] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR, 2024. 2, 6, 7
- [82] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. NeurIPS, 2024. 2, 6, 7
- [83] Rajeev Yasarla, Hong Cai, Jisoo Jeong, Yunxiao Shi, Risheek Garrepalli, and Fatih Porikli. Mamo: Leveraging memory and attention for monocular video depth estimation. In ICCV, 2023. 3
- [84] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023. 5, 6
- [85] Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, and Chunhua Shen. Learning to recover 3d scene shape from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 204–213, 2021. 3
- [86] Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Simon Chen, Yifan Liu, and Chunhua Shen. Towards accurate reconstruction of 3d scene shape from a single monocular image. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5):6480–6494, 2022. 3
- [87] Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3d: Towards zero-shot metric 3d prediction from a single image. In ICCV, 2023. 2, 3
- [88] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint

- arXiv:2409.02048, 2024. 2

[89] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and MingHsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint

- arXiv:2410.03825, 2024. 5, 6

- [90] Lvmin Zhang and Maneesh Agrawala. Transparent image layer diffusion using latent transparency. arXiv preprint arXiv:2402.17113, 2024. 3, 4, 2
- [91] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In

- Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 5, 2
- [92] Zhoutong Zhang, Forrester Cole, Richard Tucker, William T Freeman, and Tali Dekel. Consistent depth of moving objects in video. TOG (Proceedings of ACM SIGGRAPH), 40(4):1– 12, 2021. 3
- [93] Sijie Zhao, Wenbo Hu, Xiaodong Cun, Yong Zhang, Xiaoyu Li, Zhe Kong, Xiangjun Gao, Muyao Niu, and Ying Shan. Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos. arXiv preprint arXiv:2409.07447, 2024. 2
- [94] Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3d: A large photo-realistic dataset for structured 3d modeling. In Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IX 16, pages 519–535. Springer, 2020. 5, 1
- [95] Jiren Zhu, Russell Kaplan, Justin Johnson, and Li Fei-Fei. Hidden: Hiding data with deep networks. In Proceedings of the European conference on computer vision (ECCV), pages 657–672, 2018. 3

## GeometryCrafter: Consistent Geometry Estimation for Open-world Videos with Diffusion Priors

### Supplementary Material

Table 1. An overview of the training datasets.

Dataset Domain #Frames #Videos 3DKenBurns [50] In-the-wild 76K 526

DynamicReplica [34] Indoor/Outdoor 145K 1126 GTA-SfM [66] Outdoor/In-the-wild 19K 234 Hypersim [58] Indoor 75K ×

IRS [67] Indoor 103K 722 MatrixCity [42] Outdoor/Driving 452K 3029

MidAir [16] Outdoor/In-the-wild 357K 2433 MVS-Synth [32] Outdoor/Driving 12K 120

Spring [49] In-the-wild 5K 49 Structured3D [94] Indoor 71K ×

Synthia [60] Outdoor/Driving 178K 1276 TartanAir [71] In-the-wild 306K 2245 UrbanSyn [22] Outdoor/Driving 7K × VirtualKitti2 [8] Driving 43K 320

Total - 1.85M 12K

#### 1. Datasets

##### 1.1. Training Datasets

We collect 14 open-source synthetic RGBD datasets to facilitate the training of GeometryCrafter, among which 11 can be composited into video sequences. To construct the training video dataset, we extract non-overlapping segments with a sequence length not exceeding 150 frames. An overview of the training datasets is provided in Tab. 1, categorized into four distinct domains: indoor, outdoor, in-thewild and driving scenarios. It is noteworthy that the frame count may slightly differ from the original datasets, owing to the exclusion of invalid frames. To ensure computational efficiency and adhere to GPU memory constraints, we preprocess all images and videos to a standardized resolution of 320 × 640. Specifically, we apply cover resizing while preserving the original aspect ratio, followed by center cropping to achieve the desired resolution. Additionally, we implement random resizing as a technique for augmenting camera intrinsics.

##### 1.2. Evaluation Datasets

We exhaustively evaluate GeometryCrafter and previous state-of-the-art methods using seven datasets with ground truth labels that remain entirely unseen during the training phase. Notably, to ensure compatibility with most baselines, such as MoGe [69] and UniDepth [56], which necessitate an input image aspect ratio of less than 2, we preprocess the evaluation datasets in the following manner:

- • GMU Kitchens [21]: All scenarios are employed for evaluation. For each scenario, we extract 110 frames with a stride of 2 to ensure extensive spatial coverage while preserving temporal coherence. To reduce memory usage during evaluation, we downsample the generated 1920p videos and ground truth depth maps to a resolution of 960 × 540.
- • ScanNet [12]: Following DepthCrafter [31], we select 100 scenes from the test split for evaluation, wherein each video comprises 90 frames with a frame stride of 3. Due to the discrepancy in resolutions between the RGB images and depth maps, we first resize the RGB images to align with the depth maps, followed by center cropping to remove the black space around RGB images, yielding videos of resolution 624 × 464 .
- • DDAD [24]: All 50 sequences from the validation split of the DDAD dataset are utilized for evaluation, with sequence lengths of either 50 or 100 frames. Owing to the high memory demands of the raw resolution 1936×1216, we apply center cropping to reduce the resolution to 1920 × 1152, followed by downsampling to 640 × 384 for evaluation. The ground truth depth maps, acquired via LiDAR sensors, are inherently sparse; consequently, the preprocessing has negligible influence on the comparative analysis of various methods.
- • KITTI [20]: All sequence in the valid split of depth annotated dataset are used evaluation. For excessively long video sequences, we extract the initial 110 frames, resulting in 13 videos with sequence lengths ranging between 67 and 110 frames. Given that the original resolution of 1242 × 375 fails to conform to the aspect ratio requirements of most baseline methods, we apply center cropping to achieve a resolution of 736 × 368.
- • Monkaa [48]:We select 9 scenes from the original dataset for evaluation, truncating each video sequence to 110 frames while maintaining the original resolution of 960×

540. To derive valid masks, we manually annotate the sky regions within each sequence.

- • Sintel [7]: All sequences within the training split are employed for evaluation, with sequence lengths ranging between 21 and 50 frames. Given the original resolution of 1024×436 for each image, we apply cropping to achieve a standardized resolution of 872 × 436.
- • DIODE [65]: We utilize all 771 images from the validation split of DIODE for evaluation purposes. To address the noisy values along the edges of objects within the depth maps, we employ a Canny filter to detect edge

regions, subsequently refining the valid masks based on the filtering outcomes.

#### 2. Loss Functions of VAE and UNet

To train the point map VAE, we define the loss function Lpmap to measure the reconstruction errors of point maps. The reconstruction loss Lrecon for each valid pixel is defined as the L1 norm

Lrecon =

||θdiag − θdiag||1 (8)

||zp − zp||1 +

p∈M

p∈M

where M = {p|m(p) = 1} and zp, θdiag are the reconstructed values at pixel p. To enhance surface quality, we additionally supervise the normal maps derived from the reconstructed point maps and the ground truth:

(1 − np · np) (9)

Ln =

p∈M

To enhance supervision for local geometry, we draw inspiration from MoGe [69] and propose a multi-scale depth loss function that measures the alignment between reconstructed and ground truth depth maps within local regions Hα, parameterized by scale α

) − ( zp − zp,H

)||1

Lms =

||(zp − zp,H

α

α

Hα p∈Hα&p∈M

(10) Here, zp,H

and zp,H

are the mean value of predicted and ground truth depth map defined on local region Hα. In practice, we split video frames into non-overlapped patches of

α

α

size Wα × Hα to define the local regions. The reconstruction objective Lpmap is thus given by

Lpmap = Lrecon + Lms + λnLn (11)

Following LayerDiffuse [90], we apply the frozen decoder DSVD to measure the extent to which the latent offset disrupts the modified latent distribution during training, given by

Lidentity = || xdisp− xdisp||22 = || xdisp−DSVD(zpmap)||22 (12)

where || · ||22 denotes the mean square loss function. Additionally, we introduce a mask loss to regularize the reconstructed valid mask:

Lmask = || m − m||22 (13)

where m ∈ RT×H×W is the ground truth valid mask. The final training objective of VAE is defined as

LVAE = Lidentity + Lpmap + λmaskLmask (14)

To finetune the UNet Dθ parameters on the adjusted latent space obtained by our proposed point map VAE, we employ the objective LUNet for supervision, written as

Ezt∼p(z,σt),σt∼p(σ)[λσt||Dθ(zt; σt, zv, zprior)−zpmap||22] (15)

Here the noisy latent input zt is generated by adding Gaussian noise n to the latent code zpmap. zv is the conditional latent code of input video. zprior denotes the per-frame geometry priors provided by MoGe [69]. σt denotes noise level at time t, satisfying log σt ∼ N(Pmean,Pstd) with Pmean = 0.7 and Pstd = 1.6 adopted in the EDM [35] noise schedule and λσ

is a weight parameter at time t.

t

#### 3. More Implementation Details

For the point map VAE design, we reuse the architecture of SVD’s VAE with minor modification: we adopt zero convolution initialization [91] to the output convolution layer of encoder and apply a scale factor of 0.1 to ensure that latent offsets do not disrupt the latent distribution during the initial stage of training. Inspired by the training strategy of SVD, we first train the model from scratch with an AdamW [46] optimizer on RGBD images, with a fixed learning rate of

- 1e-4 for 40K iterations. Then, we finetune the temporal layers in the decoder for another 20K iterations on video data. The batch sizes are set to be 64 and 8 for the respective stages, with sequence lengths randomly sampled from [1,8] for video data in the second stage. For the UNet denoiser, we initialize UNet with the pretrained parameters provided by DepthCrafter [31], finetuning it with a learning rate of
- 1e-5 and a batch size of 8. We train our diffusion UNet in two stages, where we first train it on videos with sequence lengths sampled from [1, 25] frames to adapt the model to our generation task, and then solely finetune the temporal layers with the sequence length randomly sampled from [1, 110] frames due to the limitation of GPU memory. After training, the UNet can process videos with varying lengths (e.g., 1 to 110 frames) at a time. Both components are trained on 320 × 640 images or videos for efficiency, with random resizing and center cropping applied for data augmentation and resolution alignment. All trainings are conducted on 8 GPUs, with the entire process requiring about

- 3 days.
- 4. Camera Pose Estimation

To recover camera poses from point maps, we need to establish correspondences of the static background across frames. We first obtain the dynamic object masks by annotating the first frame using SegmentAnything [38], and then apply XMem [11], a robust method for video object segmentation, to generate the dynamic target masks for the subsequent frames. Given the dynamic masks, we adopt SuperPoint [13] to detect reliable points of interest in the first

Table 2. Inference time.of different components on 448 × 768 videos with 110 frames.

|Method|Per-frame Prior Encoder UNet Decoder<br><br>|Total|
|---|---|---|
|Ours(G) Ours(D)<br><br>|0.1 0.04 0.04 0.08 0.1 0.04 0.01 0.08|0.27s/frame 0.24s/frame<br><br>|

frame and filter out those points that belong to the dynamic objects. After that, we employ SpaTracker [76] to generate the 2D trajectory of each point, which is subsequently used to form the constraints for the camera pose optimization. Let pt denote the XY coordinate of a 2D trajectory at time step t, the 2D point pt can be lifted to the world coordinate pt using the following transformation

pt = Wt−1πK−1

(pt,D(pt)) (16)

t

Here Wt denotes the camera pose at time step t, D(·) denotes the scale-invariant depth value obtained from our pre-

dicted point maps and πK−1

refers to the back-projection of the 2D point to camera coordinate with camera intrinsic K, which can also be estimated from the point maps. For time step t′, the 2D projected coordinate should align with the trajectory position at timestep t′. Therefore, we formulate the camera pose estimation as the following problem

t

||πKjWjWi−1πK−1i[pi, D(pi)] − [pj, D(pj)]||22 (17)

min

W1...WT

i,j∈[1...T]

Due to the sequence length limitation of SpaTracker (12 for each segment), we apply a shifted window strategy with 6 overlapping frames to regularize the optimization of all camera poses. The optimization process for each scene takes from less than 1 minute to several minutes, relying on the number of frames.

#### 5. Limitations

The major limitation of our method is the expensive computation and memory cost, primarily attributing to the large model size inherent in both the VAE and U-Net architectures. As shown in Tab. 2, we provide a comparison of the inference times of different components in GeometryCrafter. Our experiments are conducted on a single GPU, revealing that the decoder of the point map VAE is the bottleneck during inference. How to design a lightweight decoder capable of producing temporally consistent outputs will be a focal point of our future works.

#### 6. More results

In the following pages, we provide more visual results of our method. We provide more results on Sora [6]-generated videos to demonstrate the temporal consistency and geometry quality of our method, as shown in Fig. 1. For comprehensive comparison with MGE methods, we provide a

visual analysis in Fig. 2. Our method achieves robust and sharp point map estimation compared to other methods. In contrast, UniDepth [56] fails to segment the sky region from the input frames, while MoGe [69] struggles to handle finegrained structure. Fig. 3 and Fig. 4 show the point maps aligned with the optimized camera poses, where the rows from left to right are 4 input frames uniformly sampled from the whole video and two views of aligned point maps in the world coordinates. We only provide the results of concatenating 8 point maps sampled from the predicted point sequences for better visualization.

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

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

###### Figure 1. Visual results on Sora-generated videos. The rows from left to right are the input videos, the disparity maps and the point cloud of the first frame.

[Figure 151]

[Figure 152]

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

Input Video UniDepth MoGe Ours

###### Figure 2. Visual comparison with monocular geometry estimation methods. All point maps are converted to disparity maps for better visualization the sharpness of depth prediction.

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

###### Figure 3. Visual results on DL3DV [45] with camera poses estimated from the output point maps. We concatenate 8 aligned point maps from the original point map sequence for visualization.

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

[Figure 216]

###### Figure 4. Visual results on DAVIS [54] with camera poses estimated from the output point maps. We concatenate 8 aligned point maps from the original point map sequence for visualization.

