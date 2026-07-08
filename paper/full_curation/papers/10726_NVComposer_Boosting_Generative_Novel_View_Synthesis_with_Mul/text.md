## NVComposer: Boosting Generative Novel View Synthesis with Multiple Sparse and Unposed Images

Lingen Li1,2 Zhaoyang Zhang2† Yaowei Li2,3 Jiale Xu2 Wenbo Hu2 Xiaoyu Li2 Weihao Cheng2 Jinwei Gu1 Tianfan Xue1 Ying Shan2

1The Chinese University of Hong Kong 2ARC Lab, Tencent PCG 3Peking University

# arXiv:2412.03517v2[cs.CV]6Dec2024

Input Views ViewCrafter Ours

Legend

###### PSNR

###### DISTS

23

0.28

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

|DUSt3R<br><br>ViewCrafter<br><br>NVComposer<br><br>(Ours)|
|---|

- #1

[Figure 6]

- #2

[Figure 7]

- #3

- 1-View

[Figure 8]

- 2-View

[Figure 9]

- 3-View

(Higher is Better)

(Lower is Better)

21

0.24

19

0.2

[Figure 10]

[Figure 11]

[Figure 12]

17

0.16

Real Reference

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

15

0.12

Num. of Input View(s) Num. of Input View(s)

13

0.08

1 2 3 4

1 2 3 4

Figure 1. As the number of unposed input views increases, NVComposer (blue circle) effectively uses the extra information to improve NVS quality. In contrast, ViewCrafter [43] (green triangle), which relies on external multi-view alignment (via pre-reconstruction from DUSt3R [34]), suffers performance degradation as the number of views grows due to instability of the external alignment. This result contradicts the common expectation that “more views lead to better performance.” Please refer to Sec. 4.2 for full results.

### Abstract

### 1. Introduction

Recent advancements in generative models have significantly improved novel view synthesis (NVS) from multi-view data. However, existing methods depend on external multiview alignment processes, such as explicit pose estimation or pre-reconstruction, which limits their flexibility and accessibility, especially when alignment is unstable due to insufficient overlap or occlusions between views. In this paper, we propose NVComposer, a novel approach that eliminates the need for explicit external alignment. NVComposer enables the generative model to implicitly infer spatial and geometric relationships between multiple conditional views by introducing two key components: 1) an image-pose dualstream diffusion model that simultaneously generates target novel views and condition camera poses, and 2) a geometryaware feature alignment module that distills geometric priors from dense stereo models during training. Extensive experiments demonstrate that NVComposer achieves stateof-the-art performance in generative multi-view NVS tasks, removing the reliance on external alignment and thus improving model accessibility. Our approach shows substantial improvements in synthesis quality as the number of unposed input views increases, highlighting its potential for more flexible and accessible generative NVS systems.

With recent advances in generative models, generative novel view synthesis (NVS) methods have drawn considerable attention [7, 20, 21, 27, 39, 43] due to its ability to synthesize novel views with only one or a few images. Unlike reconstruction-based NVS methods, where dense-view images with a full coverage of the scene are required [15, 16, 22, 34], generative NVS methods could take only one or a few views as inputs, completing unseen parts of a scene with plausible content [25, 42]. This capability is particularly useful in applications where capturing extensive views is impractical, offering greater flexibility and efficiency for virtual scene exploration and content creation.

In addition to generate novel views from a single input image, generative NVS methods [7, 19, 43] have demonstrated more flexible utility by reducing ambiguity through giving additional input images [41]. To leverage multiview images in the generative NVS tasks, existing methods [7, 39, 43] all rely on external multi-view alignment processes before generation. For example, assuming accurate poses of condition images are given (through explicit pose estimation) [7] or generate novel views conditioned on results extracted from reconstructive NVS methods (through pre-reconstruction) [19, 39, 43]. However, in the case of the overlap region being small and hard to do

† Project Lead. Project Page: https://lg-li.github.io/project/nvcomposer

#

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

t

###### CLIP

[Figure 23]

Input Image-Pose Bundles (IPBs)

Output IPBs

Dual-Stream Diffusion Model

Legend

…

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Anchor Frame

[Figure 28]

[Figure 29]

[Figure 30]

Cross Attention

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

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

#### ? ? ? ?

t

[Figure 51]

Anchor

[Figure 52]

CLIP CLIP Image Encoder

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

!

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Latent Image Stream Encoder

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

!

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Latent Decoder

[Figure 87]

Timestep Embedding

!

[Figure 88]

[Figure 89]

[Figure 90]

Target Segment

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

!

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

+

[Figure 103]

[Figure 104]

[Figure 105]

Extrinsics Embedding

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

!

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

!

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

C Concatenate

[Figure 132]

Pose Stream

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

C

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Res-Block with a Spatial and a Temporal Attention Layer

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

…

…

…

…

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

|?| |
|---|---|
|?| |
| | |

Anchor

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Spatial-Temporal Joint Self Attention Layer

[Figure 187]

[Figure 188]

Condition Segment

+

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

###### DUSt3R

[Figure 197]

[Figure 198]

[Figure 199]

Geometry-Aware Feature Alignment Adapter

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

|[Figure 207]|
|---|

[Figure 208]

[Figure 209]

[Figure 210]

(External) Image Channel

[Figure 211]

[Figure 212]

Convolutional Res-Block

Pose Channel

Image Channel

Pose Channel

Dense Views Input (Training Only)

Figure 2. Framework illustration of NVComposer. It contains an image-pose dual-stream diffusion model that generates novel views while implicitly estimating camera poses for conditional images, and a geometry-aware feature alignment adapter that uses geometric priors distilled from pretrained dense stereo models [34].

stereo matching, external multi-view alignment processes like camera pose estimation becomes unreliable [41]. As a result, multi-view generative NVS methods which heavily rely on the external alignment also tend to fail as shown in Fig. 1.

To overcome this limitation, we explore the possibility of removing the dependency on the external alignment process and propose Novel View Composer (NVComposer) which could generate novel views from spare unposed images without relying on any external alignment process. Our method is able to generate reasonable results in the sparse views with small overlap and large occlusion.

Firstly, to leverage the powerful generation ability of the video diffusion model, we use a pre-trained video diffusion model as the backbone of our NVComposer with unposed images as the condition in synthesis. Without external alignment of these images, we introduce a novel dualstream diffusion model in NVComposer to learn the relative poses of condition images during generation. The dualstream diffusion model not only generates novel views but also implicitly predicts the correct pose relationships between the condition images, ensuring that the model understands the relative positioning of the condition images in the scene and uses them to synthesize novel views correctly.

Moreover, to generate more view-consistent results, we employ features produced by a pretrained dense stereo model [34] to train our model with geometry awareness. Unlike previous methods [19, 43] that directly rely on the reconstruction results from dense stereo model as input, we propose a more flexible and accessible geometry-aware feature alignment adapter. This adapter aligns our model’s features with the predicted 3D features of the dense stereo model and requires no explicit reconstruction during inference. This strategy allows us to distill 3D knowledge implicitly from the dense stereo model [34]. Experiments (Sec. 4.2) demonstrate that this implicit geometry-aware

learning achieves competitive performance compared to explicit reconstruction-relied methods [34, 43]. It provides enhanced accessibility and flexibility, as it operates in an end-to-end manner and eliminates the need for an extra step of explicit pose estimation or pre-reconstruction during the inference.

To train our model, we construct a mixed dataset from different sources such as video [18, 24, 47] data and 3D [4] data with real indoor and outdoor scenes as well as synthetic 3D objects. NVComposer is trained on this diverse dataset using as few as one to four randomly sampled unposed condition images. Extensive experiments demonstrate that, when provided with multiple unposed input views, NVComposer outperforms state-of-the-art controllable video diffusion models and generative NVS models.

Our key contributions are summarized as follows:

- • We introduce the first pose-free multi-view generative NVS model for both scenes and objects, without the requirement for explicit multi-view alignment processes on input images.
- • Our proposed design which includes image-pose dualstream diffusion and geometry-aware feature alignment adapter, highlights a promising direction for creating more flexible and accessible generative NVS systems.
- • Our NVComposer achieves state-of-the-art performance on generative NVS tasks for both scenes and objects when given multiple unposed input views.

### 2. Related Work

Single-View Generative Novel View Synthesis. Early NVS methods using feed-forward networks map a single input image to new views [30, 33], but are limited to small rotations and translations due to the restricted information from one input image. Generative NVS method effectively hallucinate unseen views given limited input [3, 25, 37]. Recent advances in diffusion models [11, 29] leverage rich

❄

#

!

!

!

!

!

!

image priors for NVS to synthesize more reasonable multiview content [14, 20, 21, 32] by utilizing pretrained image diffusion models [26]. Zero-1-to-3[20] fine-tunes a latent diffusion model [26] with image pairs and their relative poses for novel view synthesis from a single image. Wonder3D[21] incorporates image-normal joint training and view-wise attention to enhance generative quality. SV3D [32] fine-tunes a video diffusion model [1] for NVS of synthetic objects.

However, single-view NVS models struggle to infer occluded or missing details due to the limited information from one viewpoint, making them less practical for realworld applications requiring complete scene understanding. Multi-View Generative Novel View Synthesis. To overcome single-view limitations, multi-view conditioned generative NVS utilizes images from multiple viewpoints [7, 19, 38, 43], enhancing the fidelity of generated views by capturing finer details and accurate spatial relationships. iFusion [38] employs a pretrained Zero-1-to-3 [20] as an inverse pose estimator and tunes a LoRA [13] adapter for each object to support multi-view NVS. CAT3D [7] uses Pl¨ucker ray embeddings [28] as pose representations and masks the target view for inpainting, allowing flexibility in the number of conditioning images. ViewCrafter [43] reconstructs an initial point cloud using a dense stereo model [34] and then employs a video diffusion model to inpaint missing regions in rendered novel views.

These methods, however, rely on accurate pre-computed poses of conditional images. Sparse views that lead to inaccurate poses can significantly degrade the quality of generated views, limiting their robustness in practical scenarios.

Video Diffusion Models. Advancements in diffusion models have extended their capabilities from static images to dynamic videos, enabling temporally coherent video generation conditioned on various inputs [1, 2, 6, 9, 12, 36, 40]. Ho et al. [12] first introduced diffusion models for video generation. Video LDM [2] operates in the latent space [26] to reduce computational demands. Subsequent works enhance controllability by incorporating additional conditions. AnimateDiff [8] extends text-to-image diffusion models to video by attaching motion modules while keeping the original model frozen. DynamiCrafter [40] introduces an image adapter for image-conditioned video generation. MotionCtrl [36] and CameraCtrl [9] incorporate camera trajectory control using pose matrices and Pl¨ucker embeddings, respectively. ReCapture [44] generates new camera trajectory views based on a given video.

Building upon the video diffusion model, our method leverages temporal coherence to synthesize unseen areas not included in the input images. Compared to previous controllable video diffusion models, our approach achieves better accuracy in camera controllability, offering robust performance for generative NVS tasks.

### 3. Methodology

The objective of NVComposer is to develop a model capable of generating novel views at specified target camera poses, using multiple unposed conditional images without requiring external multi-view alignment (e.g., explicit pose estimation). To achieve this, we propose to enable the model itself to infer the spatial relationships of the conditional views during generation. We introduce this capability through two key strategies: (1) instead of explicitly solving for camera poses, we model pose estimation as a generative task that jointly happens with the image generation, and (2) we distill effective geometric knowledge from expert models into our generative model.

This leads to two main components of our NVComposer, as shown in Fig. 2: an image-pose dual-stream diffusion model that generates novel target views while implicitly estimating camera poses for conditional images, and a geometry-aware feature alignment adapter that uses geometric priors distilled from pretrained dense stereo models [34]. The design and implementation of these components are detailed below.

##### 3.1. Image-Pose Dual-Stream Diffusion

Assume the model accepts T elements as input and produces T elements as output, where each element corresponds to an image captured within the current scene, accompanied by its pose annotation. We refer to these elements as image-pose bundles. We partition these bundles into two segments: the first N bundles constitute the target segment, and the remaining M bundles form the condition segment, as illustrated in Fig. 2.

Image-Pose Bundles. Specifically, let It ∈ R3×H×W denote the t-th RGB image, and Pt ∈ R6×H×W be the corresponding Pl¨ucker ray embedding [28] representing the camera pose. The conditional input and generated output are sequences of T image-pose bundles for a specific scene, denoted as B = {[It′,Pt′]}Tt=1, where the t-th image-pose bundle consists of the concatenation (along the channel dimension) of the latent image It′ = E(It) ∈ R4×H8 ×W8 and the resized Pl¨ucker ray Pt′ ∈ R6×H8 ×W8 . Here, [·] denotes concatenation and E represents the VAE encoder of latent diffusion.

The main difference between the target output B and the conditional input Bc is that B is complete, while Bc is partially masked. Specifically, for Bc, image latents in the target segment (i.e., the first N elements) and pose embeddings in the condition segment are set to zero. Additionally, we utilize relative camera poses in our model and designate the first elements of both the target and condition segments as the anchor view for this relative coordinate system. This implies that these two elements are expected to have the

same image content, and their camera extrinsic matrices are identity transformations. The image for the anchor view is always provided during training and inference, corresponding to the scenario where at least one conditional image is available. With this design, our image-pose dual-stream diffusion model accepts M unposed conditional images and N target poses as input, and output N novel view images at target poses along with M predicted poses for conditional images, where N,M ≥ 1 and N + M = T.

Video Prior. Video priors from video diffusion model has been validated to be useful for generative NVS tasks [7, 19, 43]. To fully leverage the generative priors in NVComposer, we initialize the dual-stream diffusion model using the pretrained weights of the video diffusion model DynamiCrafter [40]. We omit the frame rate and text conditions from the original video diffusion model, focusing solely on the relevant components for our task. We retain the image CLIP [23] feature conditioning with its Q-Formerlike [17] image adapter in the cross-attention layers, using the anchor view as the conditioning image. Additionally, we enhance the model’s capacity to capture cross-view correspondences at different spatial locations by adding extra spatio-temporal self-attention layers after each Res-Block of original video diffusion model. Although camera pose information is provided in Bc as Pl¨ucker ray embeddings, we further incorporate the sequence of target camera extrinsics embedding encoded by a learnable multilayer perceptron layer from the corresponding 3 × 4 camera-to-world matrices Rc ∈ RT×3×4 of all image-pose bundles, where the last M elements on the temporal dimension are masked with zeros like what we have done with Bc. These embeddings are added to the time-step embedding and serve as supplemental signals indicating the poses for each imagepose bundle in the target segment.

Pose Decoding Head Separation. We observed that training the model to generate image-pose bundles directly is hard to converge. This is caused by the difference of the two modality: images latents It′ contain complex latent features representing diverse content, while the pose embeddings Pt′ are dominated by low-frequency component. This disparity can lead to interference when jointly denoising [It′,Pt′] using tightly coupled network layers.

To address this issue, we design an additional decoding head specifically for denoising Pt′. As shown in Fig. 2, this pose decoding head operates in parallel with the original decoding part of the diffusion denoising U-Net and follows a fully convolutional architecture similar to the original decoder. It takes as input the bottleneck features, residual connections from the encoding part, and the denoising time-step embeddings. Since the Pl¨ucker ray embeddings of poses are predominantly low-frequency and relatively

###### Features from Dual-Stream Diffusion Dense View Input (Training Only)

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

×2 ×4 ×8

DUSt3R (External)

[Figure 221]

C

[Figure 222]

[Figure 223]

Lalign

…

Geometry-Aware Feature Alignment Adapter

[Figure 224]

[Figure 225]

Figure 3. Structure of the geometry-aware feature alignment adapter in NVComposer, which aligns the internal features of the dual-stream diffusion models with the 3D point maps produces by DUSt3R [34] during training. Block with notation “×2”, “×4”, and “×8” refer to bilinear upsampling on spatial dimensions. The four red bars refer to the channel-wise MLPs.

straightforward to denoise, we empirically reduce the base channel number of the pose decoding head to one-tenth of that of the original decoder for images and remove its attention layers. The outputs of the pose decoding head are concatenated along the channel dimension with those of the original decoding part of the diffusion U-Net to form the final output.

##### 3.2. Geometry-aware Feature Alignment

Since the dual-stream diffusion model is initialized from a video diffusion model that is not inherently trained with geometric constraints across views, we introduce a geometryaware feature alignment mechanism in NVComposer. This mechanism distills effective geometric knowledge from an external model with strong geometry priors during training. Specifically, we leverage the dense stereo model DUSt3R [34], which performs well with dense views (both target and condition images), to compute T pointmaps across all views (Tt, for t = 1,2,...,T) relative to the anchor view T1.

We then align the internal features of our diffusion model with these point maps through a geometry-aware feature alignment adapter during training. Specifically, as illustrated in Fig. 3, the alignment adapter (the red block on the left) takes features from the encoding part of the dualstream diffusion U-Net, immediately after each spatialtemporal self-attention layer. These features are resized to match the spatial dimensions of the image latent inputs,

8 ×W8 (the white squares in Fig. 3). The resized features retain their temporal dimension of length T, and all operations within the geometry-aware feature alignment adapter are temporally independent. The features are then processed by channel-wise MLPs (four red bars in Fig. 3) to reduce them to 320 channels, followed by a convolutional residual block that outputs a 4-D tensor F ∈ RT×6×H8 ×W8 .

H

For all item ft,t = 1,2,..,T in F along the temporal dimension, we minimize the mean squared error (MSE) with the concatenated point maps produced by DUSt3R [34] D

given the t-th view It and the anchor view I1:

1 T

Lalign =

T

∥ft − D(I1,It)∥22. (1)

t=1

##### 3.3. Training Objectives

We train the image-pose dual-stream diffusion model hθ in NVComposer to predict noises ϵ given the uniformly sampled denoising time step k, the noisy version of complete image-pose bundles B{k} at time step k, the conditional image-pose bundles Scond, and CLIP [23] image feature of the anchor view Φ(I1):

c,Φ(I0),Rc,k[∥ϵ−hθ(B{k},Bc,Φ(I1),Rc,k)∥],

Ldiff = EB,ϵ,B

(2) where θ is trainable parameters of the image-pose dualstream diffusion model, Φ is the CLIP [23] image feature encoder. Our total loss combines the diffusion loss Ldiff and the feature alignment loss Lalign:

Ltotal = Ldiff + λLalign, (3) where λ is a loss re-weighting factor.

### 4. Experiments

In this section, we evaluate the performance of NVComposer on generative NVS tasks for real-world scenes and synthetic 3D objects, followed by an analysis of the model’s sub-components. More results are in the supplementary.

##### 4.1. Training Details

We train our model on a large-scale mixed dataset built from Objaverse [4], RealEstate10K [47], CO3D [24], and DL3DV [18]. The sequence length of image-pose bundles T is set to 16. To get samples from video datasets (RealEstate10K [47], CO3D [24], and DL3DV [18]), we randomly select a frame interval between 1 and ⌊Tmax/N⌋ to sample T consecutive frames, where Tmax is the total number of frames in the scene. The value of Tmax ranges from 100 to 400 depending on the data sample. We randomly sample N condition views and shuffle them within the image-pose bundle. For samples from the 3D dataset (Objaverse [4]), we render each 3D object in two versions: one with 36 orbit views and another with 32 random views. Target views are sampled from the orbit renderings, and condition views are sampled from the random renderings, following the procedure described above.

We firstly train the model at a resolution of 512×512 for 10,000 steps across all datasets. Next, we tune the model on RealEstate10K and DL3DV at a resolution of 576×1024 for 20,000 steps for a higher resolution support. We use a learning rate of 1×10−5 and perform all training on a cluster with 64 NVIDIA V100 GPUs, with an effective batch

size of 128. For more details, please refer to the supplementary material.

##### 4.2. Results

###### 4.2.1. Generative NVS in Scenes

Benchmark Settings. We evaluate the performance of NVComposer on generative NVS tasks for scenes, comparing it with four state-of-the-art models: MotionCtrl [36], CameraCtrl [9], DUSt3R [34], and ViewCrafter [43]. MotionCtrl and CameraCtrl are controllable video generation models that work with a single input image, while DUSt3R is a dense stereo model for multi-view reconstructive NVS, and ViewCrafter is a multi-view generative NVS method that relies on explicit pose estimation and point cloud guidance.

For the RealEstate10K [47] dataset, we categorize scenes into three difficulty levels: easy, medium, and hard. The difficulty is based on the angular distances between views, specifically the rotation angle between the anchor view and the furthest target view (θtarget), and between the anchor and the furthest condition view (θcond), when more than one condition image is used. Samples are classified as follows: (1) Easy: θcond < 10 and θtarget < 10; (2) Medium: 10 ≤ θcond < 30 and 10 ≤ θtarget < 30; (3) Hard: 60 ≤ θcond < 120 and 30 ≤ θtarget < 60. We then randomly select 20 samples from the easy set, 60 from the medium set, and 20 from the hard set for evaluation. For the DL3DV [18] dataset, we randomly select 20 test scenes. Results. We measure performance by comparing generated novel views to reference images using several metrics: peak signal-to-noise ratio (PSNR), structural similarity index (SSIM)[35], and perceptual distance metrics including LPIPS[46] and DISTS [5]. Tab. 1 shows the numerical results on RealEstate10K and Tab. 2 shows the results on the DL3DV test set. As seen, NVComposer outperforms other methods across both datasets. Fig. 4 further demonstrates the visualized comparison among all these methods. For MotionCtrl [36] and CameraCtrl [9], pose controllability is limited. When the target camera poses involve large rotations or translations, these models generate sequences with minimal motion, failing to accurately follow the given instructions. These visual results align with the poor numerical performance observed in Tab. 1.

It it noteworthy that, when there are more given input views, the performance of our method consistently increases, as we also showed in Fig. 1 before. In contrast, ViewCrafter [43] suffers from a performance drop when the number of given views increases from one to two in the hard set. This is because the two conditional views in the hard set has large rotation difference, i.e., small overlapping region and possibly some occlusion are there between the two given views. This makes the external alignment process (explicit pose estimation and pre-reconstruction) tends to pro-

Easy Medium Hard

Model Views

(θcond < 10 and θtarget < 10) (10 ≤ θcond < 30 and 10 ≤ θtarget < 30) (60 ≤ θcond < 120 and 30 ≤ θtarget < 60)

PSNR↑ SSIM↑ LPIPS↓ DISTS↓ PSNR↑ SSIM↑ LPIPS↓ DISTS↓ PSNR↑ SSIM↑ LPIPS↓ DISTS↓

MotionCtrl [36] 1 15.0741 0.6071 0.3616 0.0999 12.0674 0.5667 0.5439 0.1584 11.6381 0.5276 0.5762 0.1633 CameraCtrl [9] 1 13.6082 0.5050 0.4234 0.1458 11.9639 0.4934 0.5217 0.1957 11.7599 0.4716 0.5478 0.2021

- 1 13.9443 0.5582 0.3914 0.1565 11.4854 0.4520 0.5570 0.2294 10.9003 0.4029 0.6089 0.2495

- 2 17.4837 0.6148 0.3582 0.1503 13.3077 0.4886 0.5434 0.2126 11.5381 0.4003 0.6407 0.2551

- 3 17.2341 0.6097 0.3585 0.1504 13.2212 0.4978 0.5287 0.2056 11.9211 0.4387 0.5942 0.2313

- 4 17.3545 0.6193 0.3541 0.1481 14.6845 0.5534 0.4892 0.1870 14.2381 0.5280 0.5295 0.1917

DUSt3R [34]

- 1 17.3750 0.6670 0.2849 0.1221 13.6015 0.6016 0.4315 0.1762 14.0781 0.5894 0.4293 0.1676

- 2 18.8906 0.6685 0.3079 0.1334 14.2891 0.5947 0.4478 0.1761 13.5859 0.5537 0.5100 0.1925

- 3 18.4531 0.6548 0.3024 0.1294 14.1172 0.5913 0.4401 0.1717 13.7031 0.5620 0.4867 0.1784

- 4 18.4844 0.6553 0.3068 0.1346 14.7421 0.6011 0.4230 0.1672 15.1875 0.5874 0.4327 0.1638

ViewCrafter [43]

- 1 18.7227 0.7215 0.2354 0.0996 15.3101 0.6056 0.3445 0.1516 15.2115 0.6408 0.4048 0.1462

- 2 20.7395 0.7681 0.1781 0.0793 16.9100 0.6445 0.2742 0.1198 15.3461 0.6638 0.3789 0.1384

- 3 21.5278 0.7981 0.1522 0.0716 17.7071 0.7418 0.2759 0.1097 15.3825 0.6822 0.3699 0.1324

- 4 22.5519 0.8226 0.1188 0.0537 19.5346 0.7847 0.2030 0.0851 17.8181 0.7359 0.2644 0.0988

NVComposer (Ours)

- Table 1. NVS evaluation with varying numbers of input views on RealEstate10K [47] for controllable video models MotionCtrl [36] and CameraCtrl [9], reconstructive model DUSt3R [34], and generative models ViewCrafter [43] and NVComposer. θtarget denotes the rotation angle between the anchor view and the furthest target view, while θcond indicates the angle between the anchor view and the furthest conditional view (when multiple conditions are used).

Model Views PSNR↑ SSIM↑ LPIPS↓ DISTS↓

MotionCtrl [36] 1 13.4003 0.5539 0.4004 0.1396 CameraCtrl [9] 1 12.2995 0.4692 0.4337 0.1829

DUSt3R [34]

- 1 11.7650 0.4652 0.4900 0.2295

- 2 14.6660 0.5158 0.4531 0.2104

- 3 13.9156 0.5010 0.4699 0.2127

- 4 14.8716 0.5193 0.4478 0.2072

ViewCrafter [43]

- 1 15.5625 0.4932 0.4122 0.2125

- 2 15.6875 0.4775 0.4417 0.2212

- 3 14.8593 0.4670 0.4617 0.2273

- 4 15.0625 0.4712 0.4549 0.2301

NVComposer (Ours)

- 1 15.3101 0.6056 0.3445 0.1516

- 2 16.9100 0.6445 0.2742 0.1198

- 3 17.3115 0.6687 0.2558 0.1122

- 4 17.9248 0.6958 0.2277 0.1023

- Table 2. NVS evaluation on DL3DV [18]. When more unposed input views are provided, our model consistently reports higher performance.

Method FID↓ FVD↓ KVD↓ MotionCtrl [36] 60.83 509.96 14.26

CameraCtrl [9] 52.33 561.97 24.38 ViewCrafter [43] 46.08 485.11 13.06

NVComposer (Ours) 46.19 425.44 8.04

Table 3. Distribution evaluation on generated views of MotionCtrl [36], CameraCtrl [9], ViewCrafter [43], and our NVComposer using FID [10], FVD [31], and KVD [31] metrics.

our method produces more accurate novel views when considering the entire multi-view sequence. Overall, our model generates results that are closer to the ground truth, both in terms of image and video generation.

###### 4.2.2. Generative NVS in Objects

In addition to scenes, another important scenario involves generating novel views for synthetic 3D objects. To evaluate the versatility of our proposed pipeline, we compare the generative NVS performance of our model with the objectbased generative model SV3D [32] on the Objaverse test set. The numerical and visual results of this comparison are presented in Tab. 4 and Fig. 5. Our model achieves better PSNR and comparable SSIM with SV3D when only a single conditional view is provided. Furthermore, as more unposed input views are added, our model effectively leverages the additional information, producing results that are closer to the real reference.

duce unstable results, thus leading to a poor generative NVS performance.

Distribution Evaluation. In addition to evaluating perview NVS performance, we assess the distribution of generated novel view sequences using several metrics: Fr´echet Inception Distance (FID) [10], Fr´echet Video Distance (FVD) [31], and Kernel Video Distance (KVD) [31]. For FID, we treat the novel views as individual images, while for FVD and KVD, we treat them as video clips. We compute these metrics for each model using 1,000 ground truth sequences. To ensure fairness, we report results based on single input view conditions. Results are shown in Tab. 3. Our method achieves a comparable FID to ViewCrafter, but outperforms it in both FVD and KVD. This suggests that

##### 4.3. Analysis

In this section, we perform several ablation studies and analyses to validate the effectiveness of our model.

###### Input Views MotionCtrl CameraCtrl ViewCrafter NVComposer (Ours) Real Reference

###### DUSt3R

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

- # 1

[Figure 241]

- # 2

[Figure 242]

- # 3

[Figure 243]

- # 4

|[Figure 244]|
|---|

|[Figure 245]|
|---|

|[Figure 246]|
|---|

|[Figure 247]|
|---|

|[Figure 248]|
|---|

|[Figure 249]|
|---|

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

- Input 1

[Figure 257]

- Input 2

[Figure 258]

|[Figure 259]|
|---|

|[Figure 260]|
|---|

|[Figure 261]|
|---|

|[Figure 262]|
|---|

|[Figure 263]|
|---|

|[Figure 264]|
|---|

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

|[Figure 286]|
|---|

|[Figure 287]|
|---|

|[Figure 288]|
|---|

|[Figure 289]|
|---|

|[Figure 290]|
|---|

|[Figure 291]|
|---|

[Figure 292]

- Input 1

[Figure 293]

- Input 2

[Figure 294]

|[Figure 295]|
|---|

|[Figure 296]|
|---|

|[Figure 297]|
|---|

|[Figure 298]|
|---|

|[Figure 299]|
|---|

|[Figure 300]|
|---|

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

|[Figure 318]|
|---|

|[Figure 319]|
|---|

|[Figure 320]|
|---|

|[Figure 321]|
|---|

|[Figure 322]|
|---|

|[Figure 323]|
|---|

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

- Input 1

[Figure 329]

- Input 2

[Figure 330]

|[Figure 331]|
|---|

|[Figure 332]|
|---|

|[Figure 333]|
|---|

|[Figure 334]|
|---|

|[Figure 335]|
|---|

|[Figure 336]|
|---|

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

|[Figure 358]|
|---|

|[Figure 359]|
|---|

|[Figure 360]|
|---|

|[Figure 361]|
|---|

|[Figure 362]|
|---|

|[Figure 363]|
|---|

[Figure 364]

- Input 1

[Figure 365]

- Input 2

[Figure 366]

|[Figure 367]|
|---|

|[Figure 368]|
|---|

|[Figure 369]|
|---|

|[Figure 370]|
|---|

|[Figure 371]|
|---|

|[Figure 372]|
|---|

[Figure 373]

Figure 4. Visual comparison of NVS results on the RealEstate10K [47] and DL3DV [18] test sets. MotionCtrl [36] and CameraCtrl [9] uses the first view as input while other methods use two views as input. MotionCtrl and CameraCtrl produce incorrect camera trajectories. DUSt3R and ViewCrafter exhibit better camera control but introduce artifacts due to occlusions or misaligned multi-view inputs. Our model generates views that are visually closer to the reference. We provide zoomed-in details of the first three scenes in white boxes for a closer look. Additional visual comparisons can be found in the supplementary material.

Model Views PSNR↑ SSIM↑ LPIPS↓ SV3D [32] 1 13.8861 0.8130 0.2731

- 1 16.3764 0.8218 0.2286

- 2 17.1507 0.8268 0.2067

NVComposer (Ours)

4 17.7234 0.8352 0.1889

- Table 4. Generative NVS results on the Objaverse [4] test set. When only a single conditional view is provided, NVComposer achieves performance comparable to SV3D [32]. As more random unposed condition views are added, NVComposer ’s performance improves significantly.

Ablation on Image-Pose Dual Stream Diffusion. To ensure both fairness and feasibility, we train two models with and without the dual-stream diffusion design on a subset of Objaverse [4] containing 5,000 objects for one epoch from the initial weight of the video diffusion model and evaluate them on a test set with 100 objects. The results shown in Tab. 5 demonstrate that the dual-stream diffusion significantly improves the model’s performance on genera-

###### SV3D

###### NVComposer

###### NVComposer

Input Views

Reference

(1 View)

###### (Ours, 1 View)

(Ours, 4 Views)

(Random, Unposed)

|[Figure 374]|[Figure 375]<br><br>[Figure 376]<br><br>[Figure 377]|
|---|---|
|[Figure 378]<br><br>[Figure 379]<br><br>[Figure 380]|[Figure 381]|

|[Figure 382]<br><br>[Figure 383]|
|---|

|[Figure 384]<br><br>[Figure 385]|
|---|

|[Figure 386]<br><br>[Figure 387]|
|---|

|[Figure 388]<br><br>[Figure 389]|
|---|

|[Figure 390]<br><br>[Figure 391]|
|---|

|[Figure 392]<br><br>[Figure 393]|
|---|

|[Figure 394]<br><br>[Figure 395]<br><br>[Figure 396]|[Figure 397]|
|---|---|
|[Figure 398]|[Figure 399]<br><br>[Figure 400]<br><br>[Figure 401]|

|[Figure 402]<br><br>[Figure 403]|
|---|

|[Figure 404]<br><br>[Figure 405]|
|---|

|[Figure 406]<br><br>[Figure 407]|
|---|

|[Figure 408]<br><br>[Figure 409]|
|---|

|[Figure 410]<br><br>[Figure 411]<br><br>[Figure 412]|[Figure 413]|
|---|---|
|[Figure 414]|[Figure 415]<br><br>[Figure 416]<br><br>[Figure 417]|

|[Figure 418]<br><br>[Figure 419]|
|---|

|[Figure 420]<br><br>[Figure 421]|
|---|

Figure 5. Visual comparison of novel view generation results on the Objaverse [4] test set. All input views are unposed and randomly rendered from the same 3D object.

tive NVS tasks with unposed multiple condition views.

Ablation on Geometry-Aware Feature Alignment. We further conduct an ablation study on the geometry-aware

Dual-Stream PSNR↑ SSIM↑ LPIPS↓

w/ 17.0510 0.7501 0.1353 w/o 14.6857 0.7458 0.2095

- Table 5. Ablation experiments on dual-stream diffusion on Objaverse [4]. We train the two models (initialized from the same checkpoint) for one epoch on a small subset of Objaverse. The model without dual-stream only generates images instead of the image-pose bundles.

Alignment PSNR↑ SSIM↑ LPIPS↓ DISTS↓

w/o 14.7218 0.6291 0.3799 0.1494 w/ 15.6568 0.6440 0.3284 0.1340

- Table 6. Ablation experiments on the geometry-aware feature alignment (Alignment in table). We initialize two models with and without the alignment mechanism from a same checkpoint, and train the two models for an epoch, then evaluate them on RealEstate10K [47].

feature alignment mechanism using the RealEstate10K dataset [47]. In this experiment, we train two models from the same initial checkpoint for one epoch, with and without geometry-aware feature alignment. Tab. 6 demonstrate the numerical results and Fig. 6 shows the visualized results of this ablation study. We can clearly tell that this feature alignment mechanism helps our model learn the generative NVS task with unposed multiple conditional views.

Sparse-View Pose Estimation Thanks to the unique design of the dual-stream diffusion, NVComposer can implicitly estimate the pose information. We follow the method [45] to solve the camera poses from the Pl¨ucker rays generated by the dual-stream diffusion of NVComposer. We perform the evaluation on the RealEstate10K dataset, asking the model to estimate the two sparse condition images given in the easy and hard subsets we discussed in Tab. 1. The accuracy of estimated poses is quantitatively evaluated in the average degrees of rotation angle differences and the average translation difference (with normalization according to the 2-norm of the translation of the furthest view). The results are given in Tab. 7, where we can see that our method is comparable to the performance of DUSt3R [34] in the easy case and outperforms DUSt3R in the hard case. This is because DUSt3R estimates poses using explicit deep feature correspondences, while NVComposer implicitly generates pose information during novel view generation. When given sparse condition views with minimal overlap (i.e., in ill-posed cases), our method’s implicit pose estimation proves more robust, delivering accurate pose estimates directly corresponding to the current scene in novel view generation.

Input Views w/o Alignment w/ Alignment

Real Reference

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

|[Figure 433]|
|---|

|[Figure 434]|
|---|

|[Figure 435]|
|---|

|[Figure 436]|
|---|

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

|[Figure 447]|
|---|

|[Figure 448]|
|---|

|[Figure 449]|
|---|

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

|[Figure 454]|
|---|

|[Figure 455]|
|---|

|[Figure 456]|
|---|

[Figure 457]

Figure 6. A visual sample in the ablation results of the geometryaware feature alignment with two input views given. Some patches are zoomed in for a better view. The feature alignment helps NVComposer to properly utilize contents from other views.

Subset Method ∆Rˆ ↓ ∆Tˆ ↓ Easy

DUSt3R [34] 9.6968 0.5757

NVComposer (Ours) 2.7225 0.0257 Hard

DUSt3R [34] 58.3987 0.7603 NVComposer (Ours) 5.8566 0.0263

Table 7. Comparison with pose estimation accuracy on two spare condition images in our RealEstate10K [47] test sets. Our NVComposer implicitly predicts camera poses by generating ray embeddings of condition views while generating target views.

### 5. Conclusion

We presented NVComposer, a novel multi-view generative NVS model that eliminates the need for external multi-view alignment, such as explicit camera pose estimation or prereconstruction of conditional images. By introducing an image-pose dual-stream diffusion model and a geometryaware feature alignment module, NVComposer is able to effectively synthesize novel views from sparse and unposed condition images. Our extensive experiments demonstrate that NVComposer outperforms state-of-the-art methods that rely on external alignment processes. Notably, we show that the model’s performance improves as the number of unposed conditional images increases, highlighting its ability to implicitly infer spatial relationships and leverage available information from unposed views. This paves the way for more flexible, scalable, and robust generative NVS systems that do not depend on external alignment processes.

### References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [2] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 3
- [3] Eric R Chan, Koki Nagano, Matthew A Chan, Alexander W Bergman, Jeong Joon Park, Axel Levy, Miika Aittala, Shalini De Mello, Tero Karras, and Gordon Wetzstein. Generative novel view synthesis with 3d-aware diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4217–4229, 2023. 2
- [4] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13142–13153, 2023. 2, 5, 7, 8
- [5] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence, 44(5):2567–2581, 2020. 5
- [6] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023. 3
- [7] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024. 1, 3, 4
- [8] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 3
- [9] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 3, 5, 6, 7
- [10] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [12] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video dif-

- fusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 3
- [13] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3
- [14] Yash Kant, Aliaksandr Siarohin, Ziyi Wu, Michael Vasilkovsky, Guocheng Qian, Jian Ren, Riza Alp Guler, Bernard Ghanem, Sergey Tulyakov, and Igor Gilitschenski. Spad: Spatially aware multi-view diffusers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10026–10038, 2024. 3
- [15] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 1

- [16] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding image matching in 3d with mast3r. arXiv preprint arXiv:2406.09756, 2024. 1
- [17] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 4

- [18] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024. 2, 5, 6, 7
- [19] Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Reconx: Reconstruct any scene from sparse views with video diffusion model. arXiv preprint arXiv:2408.16767, 2024. 1, 2, 3, 4
- [20] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298–9309, 2023. 1, 3
- [21] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9970–9980, 2024. 1, 3
- [22] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 1
- [23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 4, 5

- [24] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10901–10911, 2021. 2, 5
- [25] Chris Rockwell, David F Fouhey, and Justin Johnson. Pixelsynth: Generating a 3d-consistent experience from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14104–14113, 2021. 1, 2
- [26] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [27] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023. 1
- [28] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. Advances in Neural Information Processing Systems, 34: 19313–19325, 2021. 3
- [29] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2
- [30] Richard Tucker and Noah Snavely. Single-view view synthesis with multiplane images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 551–560, 2020. 2
- [31] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 6
- [32] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision, pages 439–457. Springer, 2025. 3, 6, 7
- [33] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser. Ibrnet: Learning multi-view image-based rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2021. 2
- [34] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024. 1, 2, 3, 4, 5, 6, 8
- [35] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 5

- [36] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 3, 5, 6, 7
- [37] Olivia Wiles, Georgia Gkioxari, Richard Szeliski, and Justin Johnson. Synsin: End-to-end view synthesis from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7467–7477,

2020. 2

- [38] Chin-Hsuan Wu, Yen-Chun Chen, Bolivar Solarte, Lu Yuan, and Min Sun. ifusion: Inverting diffusion for posefree reconstruction from sparse views. arXiv preprint arXiv:2312.17250, 2023. 3
- [39] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor Verbin, Jonathan T Barron, Ben Poole, et al. Reconfusion: 3d reconstruction with diffusion priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21551–21561, 2024. 1
- [40] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pages 399–417. Springer, 2025. 3, 4
- [41] Chao Xu, Ang Li, Linghao Chen, Yulin Liu, Ruoxi Shi, Hao Su, and Minghua Liu. Sparp: Fast 3d object reconstruction and pose estimation from sparse views. In European Conference on Computer Vision, pages 143–163. Springer, 2025. 1, 2
- [42] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4578–4587, 2021. 1
- [43] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 1, 2, 3, 4, 5, 6
- [44] David Junhao Zhang, Roni Paiss, Shiran Zada, Nikhil Karnad, David E. Jacobs, Yael Pritch, Inbar Mosseri, Mike Zheng Shou, Neal Wadhwa, and Nataniel Ruiz. Recapture: Generative video camera controls for user-provided videos using masked video fine-tuning. arXiv preprint arXiv:2411.05003, 2024. 3
- [45] Jason Y Zhang, Amy Lin, Moneish Kumar, Tzu-Hsuan Yang, Deva Ramanan, and Shubham Tulsiani. Cameras as rays: Pose estimation via ray diffusion. arXiv preprint arXiv:2402.14817, 2024. 8
- [46] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 5
- [47] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view

synthesis using multiplane images. ACM Trans. Graph, 37,

2018. 2, 5, 6, 7, 8

