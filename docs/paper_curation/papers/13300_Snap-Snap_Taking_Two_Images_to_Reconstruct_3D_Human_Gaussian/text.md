## Snap-Snap: Taking Two Images to Reconstruct 3D Human Gaussians in Milliseconds

Jia Lu1*, Taoran Yi1*, Jiemin Fang2†, Chen Yang3, Chuiyun Wu1, Wei Shen3, Wenyu Liu1, Qi Tian2, Xinggang Wang1† 1Huazhong University of Science and Technology 2Huawei Inc. 3Shanghai Jiaotong University

# arXiv:2508.14892v1[cs.GR]20Aug2025

{jialu2023, taoranyi, wuchuiyun, liuwy, xgwang}@hust.edu.cn jaminfong@gmail.com {ycyangchen, wei.shen}@sjtu.edu.cn tian.qi1@huawei.com

[Figure 1]

### Abstract

[Figure 2]

###### single view dense views two views

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

###### ···

Reconstructing 3D human bodies from sparse views has been an appealing topic, which is crucial to broader the related applications. In this paper, we propose a quite challenging but valuable task to reconstruct the human body from only two images, i.e., the front and back view, which can largely lower the barrier for users to create their own 3D digital humans. The main challenges lie in the difficulty of building 3D consistency and recovering missing information from the highly sparse input. We redesign a geometry reconstruction model based on foundation reconstruction models to predict consistent point clouds even input images have scarce overlaps with extensive human data training. Furthermore, an enhancement algorithm is applied to supplement the missing color information, and then the complete human point clouds with colors can be obtained, which are directly transformed into 3D Gaussians for better rendering quality. Experiments show that our method can reconstruct the entire human in 190 ms on a single NVIDIA RTX 4090, with two images at a resolution of 1024×1024, demonstrating state-of-the-art performance on the THuman2.0 and cross-domain datasets. Additionally, our method can complete human reconstruction even with images captured by low-cost mobile devices, reducing the requirements for data collection. Demos and code are available at https://hustvl.github.io/SnapSnap/.

[Figure 8]

[Figure 9]

misaligned human pose

uncontrollable textures

expensive data collection

Snap-Snap

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

SMPL(X) input ❌ fast inference ✅ easy data capture ✅

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

Figure 1. Under the setting of two input images, we propose a feed-forward framework named Snap-Snap which can directly predict 3D human Gaussians in milliseconds.

cal application prospects, including virtual/augmented reality, games, and Metaverse. Professional/expensive facilities to capture human data are usually required in previous reconstruction methods [20, 24, 33, 53, 59], e.g., using synchronized cameras to capture the target human from many views. Loosening the capturing requirements allows more users to complete their own reconstruction and makes it applicable to more scenarios.

### 1. Introduction

Reconstructing human bodies from images has always been a challenging topic and has been studied in many previous works [3, 4, 18–20, 40, 52]. In single-view human reconstruction [8, 51, 58], human prior estimations (SMPL-X [32]) are frequently misaligned with real-world coordinates (e.g., body inclination), and generative methods [35, 41] introduced to infer occluded regions often pro-

Human reconstruction has always been an important topic in the 3D field, which can be considered a crucial bridge between the real and digital worlds. It has broad practi-

*Equal contribution (during internship at Huawei Inc.) †Corresponding authors

duce results with limited controllability. Meanwhile, human prior estimations based on limited viewpoints still suffer from inaccuracies in details, such as the hands. Although human reconstruction methods based on dense input views [59] achieve high-quality results, they often require expensive data acquisition setups, making them less accessible to users. Fig. 1 provides a visual comparison of human reconstruction methods under different settings. Under the premise of preserving a controllable and consistent human appearance, we propose to explore an extremely challenging task - reconstructing human bodies from just two images, i.e., the front and back view, in milliseconds. This task makes it quite easy for any user to reconstruct the desired 3D digital human, without needing professional knowledge to capture redundant images or a long wait to get the result. One main challenge lies in the lack of overlap between the front and back views, making it hard to establish geometry consistency for reconstruction. Additionally, the information from the two views is too limited to cover all the details of the human body.

To tackle the above challenges, we propose to construct a fast point cloud prediction model which can reconstruct complete human point clouds from four viewpoints (front, back, left and right views) even with two input images. With training upon the human datasets, the reconstruction model based on geometry reconstruction model [44] adapts the generalizable geometric prior to the human domain. As the geometry reconstruction model only predicts point clouds without color information, we further design an enhancement algorithm to enhance the side-view color by wrapping. With the above processes, a complete set of colored point clouds for the target human is obtained. To achieve better rendering quality, we transform these points into 3D Gaussians [15] by directly inferring corresponding Gaussian attributes. Overall, the framework is efficient enough to complete one human reconstruction at the millisecond level. All you need to do is take two human images from the front and back views, so we name our method Snap-Snap to represent the shutter sound when capturing the human. Our approach demonstrates superior reconstruction performance on several datasets.

Our contributions can be summarized as follows:

- • We design a feed-forward reconstruction framework, which can directly predict 3D human Gaussians from just two images in milliseconds without human prior.
- • We redesign a geometry reconstruction model which can build human point clouds even with highly sparse input, adapting the generalizable geometric prior to the human domain. In addition, a side view enhancement algorithm is proposed to supplement the unseen information.
- • With two-view images at a resolution of 1024×1024, our method can obtain complete human reconstruction results in 190 ms and demonstrates state-of-the-art perfor-

mance on the THuman2.0 [55] and cross-domain [6, 45] datasets. The method is also shown to perform well on data acquired from low-cost mobile devices.

- 2. Related Works
- 3D Gaussian Splatting in Human. 3D Gaussian Splatting [15] explicitly reconstructs the scene with Gaussian points, achieving good reconstruction results in an efficient way. Many methods have introduced 3DGS into the field of human reconstruction, achieving state-of-the-art reconstruction results, and even high-quality animated results. Most of the work uses SMPL [27] or SMPL-X [32] as human prior for point clouds initialization and accepts videos as input for scene reconstruction. Most methods [10, 11, 13, 16, 21, 23] accept monocular video as input for the human reconstruction, while a few methods based on multi-view video reconstruction [14, 25, 30] have expanded in other aspects. However, most of these works lack generalizability, requesting to be trained separately for each scene. In contrast, we propose a feed-forward generalizable human reconstruction method, achieving good reconstruction quality even under extremely sparse viewpoints in milliseconds.

Generalizable Human Reconstruction. Generalizable Human Reconstruction aims to reconstruct the 3D human body solely through inference from input images. Many works have achieved generalizable reconstruction of the 3D human body under the framework of implicit representation, such as those based on pixel-aligned features [38, 39], those based on generalizable neural voxels [53] and those based on sparse 3D keypoints [29]. Meanwhile, the series of works [12, 49, 50] have generalized the extraction of human geometry solely from a single input image. HumanSplat [31] infers explicit human representation from a single image with semantic cues. Single-view human reconstruction often involves generative models, making it difficult for the reconstructed results to fully match the target subject. Under the sparse camera setting, NHP [18] learns generalizable neural radiance representations with body motion prior. GHG [20] learns generalizable human Gaussians on the 2D UV space of a human template. GPSGaussian [59] builds upon a depth estimation model, leveraging neural networks to extract Gaussian properties, and demonstrates strong results in novel view synthesis tasks.

Points Prediction. With the development of neural networks [5], image-based scene perception capabilities have also advanced, where point clouds are extracted solely based on RGB images. Several methods [2, 34, 42] achieve scene perception based on monocular depth estimation. Furthermore, [54] utilizes point cloud neural networks to improve the estimation of point clouds. Disparity is another representation of point cloud information in stereo vision. [26] obtains point cloud information by converting the disparity map to a depth map. Recently, with the progress

in general 3D models [47], methods for directly predicting point clouds [22, 44] have emerged and exhibited excellent perception capabilities.

### 3. Method

##### 3.1. Preliminary

3D Gaussian Splatting 3D Gaussians [15] exhibit high quality during rendering while enjoying the speed of realtime rendering. 3D Gaussians represent space as ellipses of different sizes and orientations. Specifically, the 3D Gaussian is defined with its center position µ ∈ R3, color c ∈ R3, opacity o ∈ R1, and covariance Σ. For optimization purposes, the covariance is decomposed into the scale s ∈ R3 and a quaternion q ∈ R4 representing rotation. Therefore, 3D Gaussians can be represented as:

θ = (µ,c,o,Σ = (s,q)) (1)

During the projection of 3D Gaussians onto the 2D pixel plane, 3D Gaussians accumulate along the corresponding rays. The entire rendering process is differentiable, which lays the foundation for optimization.

##### 3.2. Overview

Given only the front and back RGB images of human body, we aim to obtain high-quality human Gaussian in a feedforward manner without camera parameters. Our generalizable human reconstruction algorithm can be divided into three stages as follows, as shown in Fig. 2:

Point Cloud Prediction. We redesign a human point cloud prediction model Rp to reconstruct complete human point clouds from four viewpoints (front, back, left and right viewpoints). We additionally introduce two heads to predict point clouds from side views under the condition of missing side geometry information, while maintaining alignment with the real-world coordinate system. By training Rp on human datasets, we adapt the geometric reconstruction prior to better fit the human domain. Finally, we concatenate the point clouds from all four views to form the complete human point clouds.

Side-view Enhancement. Since the point cloud prediction model only predicts geometry for side views without color information, we construct an enhancement module to improve the left and right side views Il,Ir with the the front and back images If,Ib in absence of camera parameters, thereby enhancing the completeness of the final representation.

Gaussian Attribute Regression. Similar to point clouds prediction, we regress Gaussian attributes from four viewpoints directly. We input the point clouds of the four perspectives, the front and back images of the input, and the enhanced pseudo-color information into the Gaussian Re-

gression network to obtain the final complete human Gaussian.

##### 3.3. Point Cloud Prediction

We design the geometric reconstruction model Rp to perform point cloud prediction from four viewpoints: front, back, and two sides, ensuring the completeness of the predicted human point clouds. Specifically, we take the front and back images If,Ib ∈ RW×W×3 as input, and process them through an encoder Ep and a decoder Dp with B blocks to obtain intermediate image representations Gf and Gb.

To fully leverage the priors from the foundation geometric reconstruction model, we adopt a similar architecture to DUSt3R [44]. In order to generate point clouds from the front and back views, two prediction heads Hf,Hb are used to process Gf and Gb respectively. Benefiting from the information exchange between front and back tokens within the decoder, we simply aggregate Gf and Gb to form the input tokens Gv for the side-view heads Hl and Hr. By training the foundation geometric reconstruction model Rp on human data, the model learns to infer plausible geometry for two sides, even in the absence of explicit side-view observations.

{Gv}Bi=1 = (Gf + Gb)/2 Bi=1 , Pl,f = Hl {Gv}Bi=1 ,Pr,f = Hr {Gv}Bi=1 ,

(2)

We concatenate the point clouds of the front, back, left, and right in the perspective of the front viewpoint to obtain the final complete point clouds of the human body. Furthermore, to align the predicted point cloud with the human in the real world, we introduce a learnable parameter to estimate the actual human scale, thus obtaining a scaling factor δ and generate the final human point cloud with the correct proportions in the real-world coordinate.

Ph = δ ∗ (Pf,f ⊕ Pb,f ⊕ Pl,f ⊕ Pr,f), (3)

Based on the priors of the foundation geometry reconstruction model and training on human datasets, we obtain the complete human point clouds from four views predicted in the perspective of the front viewpoint, even with almost no overlapping input. Rp implicitly learns the mapping relationships between viewpoints, directly learning the positional relationships of point clouds between different viewpoints, enabling the acquisition of the complete human point clouds even without camera parameters.

##### 3.4. Side-view Enhancement

Since we only use the images from the front and back views, the predicted side point clouds Pl,f,Pr,f do not contain any color information, leading to a severe lack of color information on the side views of the human body. To address this,

Point Cloud Prediction with Geometry Reconstruction Prior

###### Gaussian Attribute Regression

[Figure 34]

[Figure 35]

[Figure 36]

Input Images

𝑃 , 

𝓗𝒇

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

𝓗𝒃

𝑃 , 

𝓔𝒑 𝓓𝒑

Original Head

Concatenate

[Figure 44]

𝑃 ,  𝑃 , 

𝐼 𝐼

𝓗𝒍

𝑃 , 

Encoder & Decoder

𝐼

[Figure 45]

𝑃 , 

𝓗𝒓

Point Clouds 𝑃

𝐼

Input Images

Additional Head

𝓕𝓰

Gaussian attribute regression

Back View

[Figure 46]

Pseudo View

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Project

[Figure 51]

Side Images

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Project

[Figure 58]

𝐼

Human Gaussian Splatting

Pseudo View

NNS

Point Clouds 𝑃 𝐼

Point Clouds 𝑃

Point Clouds 𝑃

𝑃 ,  𝑃 , 

𝐼 𝐼

Side-view Enhancement

Side Images

Front View

- Figure 2. The framework of Snap-Snap. With the input front and back view images If and Ib, the point cloud prediction model Rp generate the human point clouds from the front Pf,f, back Pb,f, left Pl,f, and right Pr,f views. Side-view color information is supplied by the side-view enhancement module. With the enhanced images Il, Ir and the input images If, Ib, we obtain fianl human Gaussians through Gaussian attribute regression Fg.

Input Images

ℋ

ℋ

ℰ

|𝐺 𝐺| | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|𝐺| | | | | | | |

Average

ℋ

ℋ

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

𝒟

𝒟

[Figure 66]

: information sharing

𝛿

- Figure 3. The framework of point cloud prediction network.

algorithm is further used to transfer colors in front and back point clouds to side point clouds, which are then unprojected to side pseudo-views to obtain the side color information. We use NNS to find the nearest neighbors of the side point clouds Pl,f,Pr,f in the known colored point clouds Pf,f,Pb,f, and then assign their color information to the side point clouds, thereby obtaining the color information Cl,f and Cr,f for the side point clouds. Taking the left view for example. Assume Pl,f = {pl,f1 ,pl,f2 ,...,pl,fn },Cl,f = {cl,f1 ,cl,f2 ,...,cl,fn }, this process can be formulated as

∀i ∈ {1,2,...n} :

we propose a simple nearest neighbor search (NNS) algorithm to warp the color information from the front and back views to the side views.

j = Fnns({Pf,f,Pb,f};pl,fi ), cl,fi ← Index({Cf,f,Cb,f};j),

(4)

In Sec. 3.3, the human front and back point clouds Pf,f,Pb,f along with side point clouds Pl,f,Pr,f are obtained, but color information is missing. Since the point clouds from the geometric reconstruction model Rp is pixel-wise, we can easily obtain the color information Cf,f,Cb,f of front and back point clouds Pf,f,Pb,f. Specifically, we project the front and back image pixels to their corresponding point clouds.

where Fnns denotes the process of searching the nearest neighbor index j in the point cloud set {Pf,f,Pb,f} for

each element pl,fi ∈ Pl,f. Index denotes an indexing process which looks up the color set {Cf,f,Cb,f} using index

j and then assigns the fetched value to cl,fi . Similarly, we can obtain the color information for Pr,f. Since Pl,f and

Pr,f are also predicted in a pixel-wise manner, we can easily establish correspondence between the point cloud colors Cl,f,Cr,f and the corresponding side pseudo-view pixels

Due to the missing color information on the sides, the final reconstruction results are adversely affected. The NNS

Il,Ir without known camera parameters.

##### 3.5. Gaussian Attribute Regression

In a feed-forward manner, we predict 3D Gaussians θ = (µ,c,o,Σ = (s,q)) based on the obtained human point clouds. The point cloud prior predicted by Sec. 3.3 represents points upon the human body surface. Considering the differences between point clouds and 3D Gaussians, the prediction of absolute 3D coordinates µ is reformulated as predicting the offset ∆µ relative to the point cloud prior. In addition, we need to predict the scale, opacity, color, and rotation of the Gaussians. To reconstruct the entire human body, we regress the Gaussian attributes from the input front and back viewpoints and the left and right pseudo-viewpoints. With the point clouds predicted from the four viewpoints Pf,f,Pb,f,Pl,f,Pr,f ∈ RW×W×3 and the color maps If,Ib,Il,Ir ∈ RW×W×3, we use a UNetlike [36] network Fg to predict the Gaussian attributes θ = {∆µ,c,o,s,q}:

θf,θb = Fg({Pf,f,If},{Pb,f,Ib}), θl,θr = Fg({Pl,f,Il},{Pr,f,Ir}),

(5)

θ = θf ⊕ θb ⊕ θl ⊕ θr,

where θf,θb,θl,θr denote the Gaussians predicted from front, back, left, and right viewpoints, respectively. By concatenating them together, we obtain the final Gaussian representation θ for the entire human body.

##### 3.6. Training and Inference

The entire framework’s learnable components can be divided into two modules: (1) The point cloud prediction network, and (2) The Gaussian regression network. To ensure the performance of the algorithm, we train the two modules separately.

- Training Stage 1: For the point cloud prediction network, we use 3D point clouds Pgt and 2D image mask Mgt as supervision. The L2 loss and cross-entropy loss are denoted as the regression loss Lreg and the confidence loss Lconf.

- Lstage1 = Lreg Ph,Pgt + Lconf(Mconf,Mgt). (6)

Training Stage 2: For the Gaussian regression network, we render image Irender from novel views with differentiable splatting, and use the ground-truth image Igt as training supervision. L1 loss and SSIM loss, denoted as Lrgb and Lssim, are employed to train the Gaussian regression.

- Lstage2 = βLrgb Irender,Igt + (1 − β)Lssim Irender,Igt . (7)

Inference: During inference, we first use the input front and back view images If,Ib to obtain the point clouds of the human body Pf,f,Pb,f,Pl,f,Pt,f through the point cloud prediction model Rp. Then, we obtain the enhanced images Il,Ir through the side-view enhancement algorithm. Based on the enhanced images Il,Ir and the input images If,Ib and the point clouds of the human body Pf,f,Pb,f,Pl,f,Pr,f, the human Gaussian are obtained through Gaussian attribute regression Fg.

### 4. Experiments

##### 4.1. Implementation Details

We train our model on a single RTX 4090 with 24G memory and set the batch size as 1. For the training of each module, we apply the AdamW [28] optimizer with a learning rate of 1 × 10−4, and a weight decay of 5 × 10−2. The learning rate is cosine annealed to 1×10−7 during the training. The training iterations for stage 1 are 100k, while the training iterations for stages 2 are only 50k. The training times for the two stages are approximately 13 and 6 hours respectively. Besides, the β in Eq. 7 for the stage 2 is 0.8.

Datasets. We train and evaluate on the Thuman2.0 dataset [55] while evaluating on the 2K2K [6] and 4DDress [45] for cross-domain evaluation. The THuman2.0 dataset consists of 526 high-quality human assets. Following [20], we select the same 100 subjects to evaluate the algorithm. Due to the requirement of ground-truth SMPLX parameters by GHG [20], we select the first 100 subjects from the 2K2K training set for evaluation. Furthermore, to evaluate the reconstruction quality of loose clothing, we select 50 loose clothing examples from Thuman2.1 [55] (exclude human scans of Thuman2.0) by calculating the chamfer distance between the SMPL-X [32] mesh and the ground-truth human meshes. We sort the Chamfer Distance from largest to smallest and select the top human bodies as Loose Clothes val set.

##### 4.2. Comparisons

In Tab. 1, we present the comparison results with GPSGaussian [59] and GHG [20] with three metrics: Peak Signal-to-Noise Ratio (PSNR), Structure Similarity Index Measure (SSIM) [46], and Learned Perceptual Image Patch Similarity (LPIPS) [56] to assess the quality of the rendered images. LPIPS is calculated with AlexNet [17] model. The resolution of the rendered images is 1024 × 1024, and we only use the human regions given by the bounding box of humans when calculating the metrics following the GHG. For better comparison, we provide visual results in Fig. 4, where the resolution of the rendered images for all methods is 2024 × 2048. GHG under 2 views is trained by the publicly released code, while we use the released inpainting

###### Thuman2.0 [55] 2K2K [6] 4D-Dress [45]

Method Views SMPL-X [32] Infer. (ms) PSNR↑ SSIM ↑ LPIPS ↓ PSNR↑ SSIM↑ LPIPS ↓ PSNR↑ SSIM↑ LPIPS ↓

GPS-Gaussian [59] 5 N/A 144 20.70 88.11 19.15 21.68 88.45 17.17 20.47 84.78 22.43 GHG [20] 3 GT 2858 21.99 88.45 13.79 21.73 87.50 13.88 21.20 86.03 17.24 GHG 2 GT 2853 21.79 87.66 17.54 21.26 86.38 17.54 20.69 85.13 20.58

GHG 2 Estimated 10696 16.71 81.92 26.22 16.72 80.48 26.53 18.34 80.99 26.55 Snap-Snap (Ours) 2 N/A 190 22.44 88.78 13.24 22.38 88.01 13.08 21.62 85.55 17.03

Table 1. We present the comparison results with the GPS-Gaussian [59] and GHG [20]. We find that GHG uses the ground-truth SMPLX [32] parameters during inference. For a fair comparison, we estimate the SMPL-X parameters only using two viewpoints though EasyMocap [1]. Infer. denotes the total inference time from receiving images and masks to the final Gaussians. The detailed timeconsuming analysis is presented in the supplementary materials.

#### Input 5 Views Input 2 Views

Snap-Snap (Ours)

GHG (w/ GT SMPL-X)

Input GPS-Gaussian Input GT

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

#### NormalClothingLooseClothingCrossDomain

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

[Figure 109]

[Figure 110]

[Figure 111]

Figure 4. Visual comparisons with GPS-Gaussian [59] and GHG [20].

model weight. GPS-Gaussian under 5 views is completely trained following the instructions.

The reconstruction quality of GPS-Gaussian is lower than ours even with five viewpoints. In Fig. 4 we can see

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

Method Views SMPL-X [32] PSNR↑ SSIM ↑ LPIPS ↓

GPS-Gaussian [59] 5 N/A 19.49 84.45 22.02 GHG [20] 3 GT 20.02 84.36 17.67 GHG 2 GT 19.74 83.44 21.60

GHG 2 Estimated 15.80 78.77 29.08 Snap-Snap (Ours) 2 N/A 20.98 84.42 17.08

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Table 2. The comparison on loose clothes val set.

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Method Views SMPL-X PSNR↑ SSIM↑ LPIPS ↓

SiTH [8] 2 GT 15.10 76.74 31.68 Snap-Snap (Ours) 2 N/A 22.44 88.78 13.24

- Table 3. The comparison between Snap-Snap and mesh-based human reconstruction SiTH [8].

Side Head NNS PSNR ↑ SSIM ↑ LPIPS↓

× × 22.15 88.20 14.11 ✓ × 22.34 88.39 13.75 ✓ ✓ 22.44 88.78 12.47

- Table 4. We perform ablation studies on the additional side-view heads in the point cloud prediction network and the side-view enhancement module.

Input images SiTH* Snap-Snap (Ours) GT SiTH*: use the input two images and ground-truth SMPL-X parameters

Figure 5. Qualitative comparisons with SiTH [8].

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

front SIFU LGM GTA Human3Diffusion ECON Ours GT

Figure 6. Visual comparisons of Snap-Snap with single-view reconstruction methods.

that GPS-Gaussian’s reconstruction shows missing body parts due to the limitations of the depth estimation module in GPS-Gaussian, which cannot provide reasonable results due to the sparsity of the viewpoints. Unlike GPS-Gaussian, our method completes the side views both in terms of point clouds and colors, which can directly infer the complete human point clouds from the front and back views.

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

GHG completes human body from sparse viewpoints based on the human prior SMPL-X. We find that GHG uses the ground-truth SMPL-X parameters during the inference, which are calculated from far more than two viewpoints. To fairly evaluate GHG’s reconstruction quality, we use EasyMocap [1] to estimate the SMPL-X parameters from two viewpoints. Moreover, our method enables human reconstruction at the millisecond level shown in Tab. 1, in contrast to GHG which requires time computing SMPL-X parameters and preprocessing data.

[Figure 159]

[Figure 160]

[Figure 161]

Input images SiTH (2views) Snap-Snap Input images SiTH (2views) Snap-Snap

Figure 7. Reconstruction results from in-the-wild data.

##### 4.3. Comparison with Single-view Reconstruction

To further evaluate the effectiveness of our method, we compare it with the mesh-based single-view reconstruction approaches. To minimize the influence of generative components [8] on the human reconstruction, we select SiTH [8] as the method for quantitative comparison. SiTH allows replacing the pseudo back-view with the ground-truth backview image, and we use the ground-truth SMPL-X parameters as the input human prior.

In Tab. 2, we evaluate our method on human bodies with loose clothing, demonstrating the robustness of our method over different human bodies. In the visual results, we can see that due to GHG being based on SMPL-X, it cannot reconstruct loose clothing well. Our method directly infers complete geometric information through the point cloud prediction model, achieving good modeling even for loose clothing.

Since mesh-based human reconstruction methods are often not aligned with the world coordinate system, we ap-

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

Point clouds Side-view Human Gaussians

Training Scans PSNR ↑ SSIM ↑ LPIPS ↓

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

426 22.44 88.78 13.47 2992 22.77 88.98 13.56

Table 5. Evaluation results with more datasets.

##### 4.5. Ablation Studies

We conduct ablation experiments on the heads of the point cloud prediction model Rp and the side-view enhancement algorithm to verify the impact of these two modules on the human reconstruction, as shown in Tab. 4. More ablation studies are given in the supplementary material.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

+ Side Heads + NNS

+ Side Heads + NNS

[Figure 183]

[Figure 184]

+ Side Heads

+ Side Heads

Figure 8. We present the visual results of ablation studies on the additional side-view head in the point cloud prediction network and the side-view enhancement module.

Heads of Point Cloud Prediction Model. We perform ablation experiments on the side heads in the point cloud prediction model Rp, as shown in the Tab. 4. We find that the point clouds of the human exhibit obvious gaps without the side heads, adversely affecting the final human reconstruction performance. To compensate for these gaps, we use two heads to predict the side point clouds of the human body, which greatly improves the completion of the point clouds and the final reconstruction quality shown in Fig. 8.

ply ICP registration [37] using the ground-truth mesh and render the registered meshes for evaluation. We compute PSNR on the rendered images, as shown in Tab. 3, with more qualitative results provided in Fig. 5. The results demonstrate that our method achieves better reconstruction quality and aligns more accurately with the human pose in the input images. As shown in Fig. 5, despite using accurate human prior (SMPL-X parameters), SiTH still produces distorted poses. In contrast, our method achieves high consistency with the target human in both pose and texture. Furthermore, we observe that due to the misalignment between mesh-based reconstruction results and the real-world coordinate system, evaluating such methods using PSNR can be unreliable.

Side-view Enhancement. We conduct ablation experiments on the side-view enhancement algorithm shown in Tab. 4. Instead of using NNS to obtain side-view images, we directly feed the front and back images into the Gaussian attribute regression under the condition that camera parameters are not available. By leveraging the color warping of NNS which based on the spatial relationships within the point clouds, our side-enhancement algorithm achieves superior reconstruction results. Fig. 8 illustrates the human point clouds with colors after wrapping. It can be observed that the side-view enhancement leads to more consistent textures, particularly on the side views.

Considering the alignment issue mentioned, we further provide a qualitative comparison with existing single-view reconstruction methods [43, 50, 51, 57, 58] in Fig. 11. These single-image methods exhibit significantly worse reconstruction consistency than ours, even when manually aligned. Moreover, the results of single-view reconstruction are often difficult to align with the human body scale in the world coordinate system, whereas our reconstructions maintain a reasonable scale due to the point clouds prediction network.

##### 4.6. Scalability

To further explore the relationship between the current method and the amount of training data, we expand the dataset by adding Thuman2.1 [55] (1919 scans) and the CustomHuman dataset [9] (647 scans) to Thuman2.0 [55] (426 scans), resulting in a total of 2992 human scans for training. Experimental results in Tab. 5 show that as the training data increases, the reconstruction performance of the current method further improves, demonstrating its strong scalability.

##### 4.4. Reconstruction from In-the-Wild Data

To validate our method on capture from low-cost mobile phone, we use two mobile phones to build a capture setup and reconstruct human bodies from the collected data. The reconstruction results are shown in Fig. 7. Details about the low-cost data collection device are provided in the supplementary material. In the absence of camera parameters under the simple capture setup, GPS-Gaussian and GHG are unable to successfully reconstruct human. Consequently, we provide a qualitative comparison with the two-view SiTH. Our method achieves superior human reconstruction in a more robust and end-to-end manner.

### 5. Conclusion

In this paper, we propose a feed-forward framework capable of directly predicting 3D human Gaussians from only

two images in 190 ms on one GPU. We redesign a geometric reconstruction model for human point clouds prediction from four viewpoints, adapting the robust geometric prior from recent foundational reconstruction models to the specific human domain via training on human data. To complete the omitted information from the input two images, we propose a simple nearest neighbor search algorithm. 3D human Gaussians can be directly obtained from the completed point clouds. We expect our method could widen the applications of human body reconstruction.

### References

- [1] Easymocap - make human motion capture easier. Github,

2021. 6, 7, 14, 16

- [2] Jia-Wang Bian, Huangying Zhan, Naiyan Wang, Tat-Jun Chin, Chunhua Shen, and Ian Reid. Auto-rectify network for unsupervised indoor depth estimation. IEEE transactions on pattern analysis and machine intelligence, 44(12):9802– 9813, 2021. 2
- [3] Mingfei Chen, Jianfeng Zhang, Xiangyu Xu, Lijuan Liu, Yujun Cai, Jiashi Feng, and Shuicheng Yan. Geometry-guided progressive nerf for generalizable and efficient neural human rendering. In European Conference on Computer Vision, pages 222–239. Springer, 2022. 1
- [4] Wei Cheng, Su Xu, Jingtan Piao, Chen Qian, Wayne Wu, Kwan-Yee Lin, and Hongsheng Li. Generalizable neural performer: Learning robust radiance fields for human novel view synthesis. arXiv preprint arXiv:2204.11798, 2022. 1
- [5] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2
- [6] Sang-Hun Han, Min-Gyu Park, Ju Hong Yoon, Ju-Mi Kang, Young-Jae Park, and Hae-Gon Jeon. High-fidelity 3d human digitization from single 2k resolution images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12869–12879, 2023. 2, 5, 6
- [7] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 14
- [8] I Ho, Jie Song, Otmar Hilliges, et al. Sith: Single-view textured human reconstruction with image-conditioned diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 538–549, 2024. 1, 7, 12
- [9] Jie Song Hsuan-I Ho, Lixin Xue and Otmar Hilliges. Learning locally editable virtual humans. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 8
- [10] Liangxiao Hu, Hongwen Zhang, Yuxiang Zhang, Boyao Zhou, Boning Liu, Shengping Zhang, and Liqiang Nie. Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 634–644, 2024. 2

- [11] Shoukang Hu, Tao Hu, and Ziwei Liu. Gauhuman: Articulated gaussian splatting from monocular human videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20418–20431, 2024. 2
- [12] Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Jiaxiang Tang, Deng Cai, and Justus Thies. Tech: Text-guided reconstruction of lifelike clothed humans. In 2024 International Conference on 3D Vision (3DV), pages 1531–1542. IEEE, 2024. 2
- [13] Rohit Jena, Ganesh Subramanian Iyer, Siddharth Choudhary, Brandon Smith, Pratik Chaudhari, and James Gee. Splatarmor: Articulated gaussian splatting for animatable humans from monocular rgb videos. arXiv preprint arXiv:2311.10812, 2023. 2
- [14] Yuheng Jiang, Zhehao Shen, Penghao Wang, Zhuo Su, Yu Hong, Yingliang Zhang, Jingyi Yu, and Lan Xu. Hifi4g: High-fidelity human performance rendering via compact gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19734–19745, 2024. 2
- [15] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 2, 3

- [16] Muhammed Kocabas, Jen-Hao Rick Chang, James Gabriel, Oncel Tuzel, and Anurag Ranjan. Hugs: Human gaussian splats. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 505–515, 2024. 2
- [17] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012. 5
- [18] Youngjoong Kwon, Dahun Kim, Duygu Ceylan, and Henry Fuchs. Neural human performer: Learning generalizable radiance fields for human performance rendering. Advances in Neural Information Processing Systems, 34:24741–24752,

2021. 1, 2

- [19] Youngjoong Kwon, Dahun Kim, Duygu Ceylan, and Henry Fuchs. Neural image-based avatars: Generalizable radiance fields for human avatar modeling. arXiv preprint arXiv:2304.04897, 2023.
- [20] Youngjoong Kwon, Baole Fang, Yixing Lu, Haoye Dong, Cheng Zhang, Francisco Vicente Carrasco, Albert MosellaMontoro, Jianjin Xu, Shingo Takagi, Daeil Kim, et al. Generalizable human gaussians for sparse view synthesis. arXiv preprint arXiv:2407.12777, 2024. 1, 2, 5, 6, 7, 12, 14, 16
- [21] Jiahui Lei, Yufu Wang, Georgios Pavlakos, Lingjie Liu, and Kostas Daniilidis. Gart: Gaussian articulated template models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19876–19887,

2024. 2

- [22] Vincent Leroy, Yohann Cabon, and Jerome Revaud. Grounding image matching in 3d with mast3r, 2024. 3
- [23] Mengtian Li, Shengxiang Yao, Zhifeng Xie, Keyu Chen, and Yu-Gang Jiang. Gaussianbody: Clothed human reconstruction via 3d gaussian splatting. arXiv preprint arXiv:2401.09720, 2024. 2

- [24] Ruilong Li, Julian Tanke, Minh Vo, Michael Zollh¨ofer, J¨urgen Gall, Angjoo Kanazawa, and Christoph Lassner. Tava: Template-free animatable volumetric actors. In European Conference on Computer Vision, pages 419–436. Springer, 2022. 1
- [25] Zhe Li, Zerong Zheng, Lizhen Wang, and Yebin Liu. Animatable gaussians: Learning pose-dependent gaussian maps for high-fidelity human avatar modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19711–19722, 2024. 2
- [26] Lahav Lipson, Zachary Teed, and Jia Deng. Raft-stereo: Multilevel recurrent field transforms for stereo matching. In 2021 International Conference on 3D Vision (3DV), pages 218–227. IEEE, 2021. 2
- [27] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 851–866. 2023. 2
- [28] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5
- [29] Marko Mihajlovic, Aayush Bansal, Michael Zollhoefer, Siyu Tang, and Shunsuke Saito. Keypointnerf: Generalizing image-based volumetric avatars using relative spatial encoding of keypoints. In European conference on computer vision, pages 179–197. Springer, 2022. 2
- [30] Arthur Moreau, Jifei Song, Helisa Dhamo, Richard Shaw, Yiren Zhou, and Eduardo P´erez-Pellitero. Human gaussian splatting: Real-time rendering of animatable avatars. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 788–798, 2024. 2
- [31] Panwang Pan, Zhuo Su, Chenguo Lin, Zhen Fan, Yongjie Zhang, Zeming Li, Tingting Shen, Yadong Mu, and Yebin Liu. Humansplat: Generalizable single-image human gaussian splatting with structure priors. arXiv preprint arXiv:2406.12459, 2024. 2
- [32] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3d hands, face, and body from a single image. In Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2019. 1, 2, 5, 6, 7, 14, 16
- [33] Sida Peng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang, Qing Shuai, Hujun Bao, and Xiaowei Zhou. Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9054–9063, 2021. 1
- [34] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021. 2
- [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 12
- [36] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmen-

- tation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 5, 14
- [37] Szymon Rusinkiewicz and Marc Levoy. Efficient variants of the icp algorithm. In Proceedings third international conference on 3-D digital imaging and modeling, pages 145–152. IEEE, 2001. 8
- [38] Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, and Hao Li. Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2304–2314, 2019. 2
- [39] Shunsuke Saito, Tomas Simon, Jason Saragih, and Hanbyul Joo. Pifuhd: Multi-level pixel-aligned implicit function for high-resolution 3d human digitization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 84–93, 2020. 2
- [40] Ruizhi Shao, Liliang Chen, Zerong Zheng, Hongwen Zhang, Yuxiang Zhang, Han Huang, Yandong Guo, and Yebin Liu. Floren: Real-time high-quality human performance rendering via appearance flow using sparse rgb cameras. In SIGGRAPH Asia 2022 Conference Papers, pages 1–10, 2022. 1
- [41] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv:2308.16512, 2023. 1
- [42] Libo Sun, Jia-Wang Bian, Huangying Zhan, Wei Yin, Ian Reid, and Chunhua Shen. Sc-depthv3: Robust selfsupervised monocular depth estimation for dynamic scenes. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 2
- [43] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision, pages 1–18. Springer, 2025. 8
- [44] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024. 2, 3, 12, 14
- [45] Wenbo Wang, Hsuan-I Ho, Chen Guo, Boxiang Rong, Artur Grigorev, Jie Song, Juan Jose Zarate, and Otmar Hilliges. 4d-dress: A 4d dataset of real-world human clothing with semantic annotations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 550–560, 2024. 2, 5, 6
- [46] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 5
- [47] Philippe Weinzaepfel, Thomas Lucas, Vincent Leroy, Yohann Cabon, Vaibhav Arora, Romain Br´egier, Gabriela Csurka, Leonid Antsfeld, Boris Chidlovskii, and J´erˆome Revaud. Croco v2: Improved cross-view completion pretraining for stereo matching and optical flow. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17969–17980, 2023. 3
- [48] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong

[Figure 185]

- Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21469–21480, 2025. 11
- [49] Yuliang Xiu, Jinlong Yang, Dimitrios Tzionas, and Michael J Black. Icon: Implicit clothed humans obtained from normals. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13286–13296. IEEE, 2022. 2
- [50] Yuliang Xiu, Jinlong Yang, Xu Cao, Dimitrios Tzionas, and Michael J Black. Econ: Explicit clothed humans optimized via normal integration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 512–523, 2023. 2, 8
- [51] Yuxuan Xue, Xianghui Xie, Riccardo Marin, and Gerard Pons-Moll. Human-3diffusion: Realistic avatar creation via explicit 3d consistent diffusion models. Advances in Neural Information Processing Systems, 37:99601–99645, 2024. 1, 8
- [52] Chen Yang, Sikuang Li, Jiemin Fang, Ruofan Liang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Gaussianobject: Just taking four images to get a high-quality 3d object with gaussian splatting. arXiv preprint arXiv:2402.10259,

2024. 1

- [53] Taoran Yi, Jiemin Fang, Xinggang Wang, and Wenyu Liu. Generalizable neural voxels for fast human radiance fields. arXiv preprint arXiv:2303.15387, 2023. 1, 2
- [54] Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, and Chunhua Shen. Learning to recover 3d scene shape from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 204–213, 2021. 2
- [55] Tao Yu, Zerong Zheng, Kaiwen Guo, Pengpeng Liu, Qionghai Dai, and Yebin Liu. Function4d: Real-time human volumetric capture from very sparse consumer rgbd sensors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5746–5756, 2021. 2, 5, 6, 8
- [56] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 5
- [57] Zechuan Zhang, Li Sun, Zongxin Yang, Ling Chen, and Yi Yang. Global-correlated 3d-decoupling transformer for clothed avatar reconstruction. Advances in Neural Information Processing Systems, 36:7818–7830, 2023. 8
- [58] Zechuan Zhang, Zongxin Yang, and Yi Yang. Sifu: Side-view conditioned implicit function for real-world usable clothed human reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9936–9947, 2024. 1, 8
- [59] Shunyuan Zheng, Boyao Zhou, Ruizhi Shao, Boning Liu, Shengping Zhang, Liqiang Nie, and Yebin Liu. Gpsgaussian: Generalizable pixel-wise 3d gaussian splatting for real-time human novel view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19680–19690, 2024. 1, 2, 5, 6, 7, 12, 13, 15

Figure 9. Data collection setup visualization.

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

Input Images TRELLIS (2 view) Snap-Snap (Ours) GT

Figure 10. Visual comparisons of Snap-Snap with TRELLIS [48].

### A. Appendix

##### A.1. Data Collection Setup Visualization

We present the low-cost data collection setup in Fig. 9. We place the two phones on two stands and synchronize the images captured by the phones with a computer. Directly reconstructing a human body from images taken in random poses is quite challenging, so to make the camera poses of the two mobile phones as similar as possible to those in the training set, we manually adjust the poses of the two phones to reduce the reconstruction difficulty.

##### A.2. Visual Comparison with Two-view 3D Generation

We provide a visual comparison between our method and TRELLIS [48], which can reconstruct 3D assets based on two images (e.g., human bodies). As shown in Fig. 10, our method demonstrates superior clarity and texture consistency in a faster inference speed, especially in terms of skin tone.

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

SMPL-X Estimator

Human Reconstruction

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Generative Model

[Figure 219]

SiTH (single-view image)

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Human Reconstrcution

SMPL-X Estimator

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

GHG (3 view images)

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

···

[Figure 238]

···

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Depth Estimator

Gaussian Regression

[Figure 246]

GPS-Gaussian (16 view images)

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

front-view back-view hallucination GT back-view

Point Clods Predictor

Gaussian Regression

Figure 12. Visualization of different texture introduced by generative methods.

###### Snap-Snap

Ours (2 view images)

Figure 11. Comparison of pipelines with different methods.

##### A.4. Uncontrollability of Generative Models

##### A.3. Comparison with Different Methods

As shown in Fig. 12, we additionally visualize the texture uncertainty introduced by generative model [35] in singleview reconstruction method SiTH [8]. Even with the same front view is provided, the model may predict inconsistent back-view textures that significantly deviate from the expected appearance. This may lead to uncontrollable reconstruction results and a noticeable misalignment with the target subject.

To further highlight the advantages of our approach, we provide a comparative illustration of different representative frameworks: single-view reconstruction methods (e.g., SiTH [8]), sparse-view methods based on SMPL-X estimations (e.g., GHG [20]), and dense-view depth-based methods (e.g., GPS-Gaussian [59]). Unlike existing approaches, our method achieves high-quality human reconstruction in milliseconds, using only a minimal number of input views, without relying on generative models [35] which often lead to uncontrollable textures and human prior estimations which often exist misalignment due to the lacking of views.

##### A.5. Ablation Studies on Gaussian Regression

Regarding Gaussian attribute regression, considering the potential inconsistency between the input front/back views and the side views generated by wrapping based on nearest neighbor search (NNS), we conduct an ablation study on the number of regression network, as shown in Tab. 6. The results show that adding the regression network for the side views specially brings no significant improvement to the human reconstruction, which also verifies the effectiveness of ours proposed side-views enhancement module.

Regression Networks PSNR ↑ SSIM ↑ LPIPS ↓

2 22.41 88.74 13.36 1 (Ours) 22.44 88.78 13.47

- Table 6. We perform ablation studies on the number of Gaussian regressions networks.

Pretraining PSNR ↑ SSIM ↑ LPIPS ↓

× 21.11 87.49 16.22 ✓ 22.44 88.78 13.47

- Table 7. We perform ablation studies on the impact of DUSt3R [44] pretraining weight.

##### A.6. Foundation Geometry Prior

To validate the importance of DUSt3R’s [44] geometry priors, we present ablation experiments on loading foundation geometry reconstruction prior in Tab. 7. The geometry priors of DUSt3R improve the reconstruction results of our method, which demonstrates the importance of introducing general foundation priors into the specific human domain.

fusion strategy PSNR ↑ SSIM ↑ LPIPS ↓

Concat 22.40 88.76 13.29 Average 22.44 88.78 13.47

- Table 8. Comparison on fusion strategy of side-view tokens.

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

Input images Ours GT Input images Ours GT

Figure 13. Reconstruction results based on arbitrary image inputs.

##### A.7. Token Fusion Strategy

In Sec. 3.3 of the main paper, we obtain global side-view tokens by averaging the front and back image tokens. In Tab. 8 we present ablation experiments comparing the averaging and concatenation strategies for obtaining side-view tokens. We observe that both methods yield good reconstruction quality, which demonstrates the effectiveness of the point cloud prediction model.

##### A.8. Reconstruction results from arbitrary inputs.

We provide visualization results of human reconstruction using two input images from two views apart from front and back views in Fig. 13. As shown in the figure, our method is able to produce high-quality reconstructions even from arbitrary image input, demonstrating strong generalization capability.

##### A.9. Multi-view Scenario

To further evaluate our reconstruction performance, we conduct experiments in a scenario with five input views and compare it with GPS-Gaussian [59], as shown in Tab. 9 and Fig. 18. Considering that it is different from the two-view scenario, there is no information lacking in the scenario with 5 input views. We only predict the point clouds in the two input views, removing the part that predicts the side view point clouds. Our algorithm achieves high-definition reconstruction under multiple views. We conduct quantitative and qualitative comparisons with GPS-Gaussian. It can be observed that our reconstruction method not only achieves complete human body reconstruction in multiview input scenarios but also attains higher reconstruction quality.

Method PSNR ↑ SSIM ↑ LPIPS ↓ GPS-Gaussian [59] 20.76 88.04 14.83

Snap-Snap (Ours) 24.73 91.59 10.67

Table 9. We compare the reconstruction results of our method with GPS-Gaussian [59] in a scenario with five input views.

[Figure 262]

[Figure 263]

[Figure 264]

w/oSideHeadw/SideHead

[Figure 265]

[Figure 266]

[Figure 267]

- Figure 14. We present the visual results of the impact of the additional side-view heads on the human construction.

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

Input Images Rendered Images

- Figure 15. Reconstruction results of the same human in different poses.

Inference Time Distribution

| |91ms 87ms 12ms| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |

Inference Time

0 50 75 100 125 175 200 Time (ms)

25

150

Gaussian Regression

Side Enhencement

DUSt3R Inference

Figure 16. The visual results of the time consumption for different modules in the inference process.

##### A.10. Impact of Overlapping Point Clouds.

In the main paper, we adapt additional side-view prediction heads to complete the geometry of the side views. However, it is unavoidable that there is some overlap between the point clouds predicted for the side views and those for the front view, which may affect the quality of front view reconstruction. Therefore, in Fig. 14, we present visualization results of front view reconstruction before and after incorporating side-view prediction heads. We can observe that after adding side-view prediction heads, although there are some additional overlapping point clouds in the front view, this has almost no impact on the front view reconstruction quality.

##### A.11. Consistency of Reconstruction

To validate the consistency of our method in reconstructing human bodies with different poses, we present reconstruction results of the same human in various poses in Fig. 15. We can observe that our method achieves good reconstruction quality across different poses while maintaining excellent consistency.

##### A.12. Time-consuming Analysis

In Fig. 16 we show the inference time spent by each module of our model, with the total inference time around 190 ms.

##### A.13. Network Architecture

We use a UNet-like [36] network architecture in the Gaussian attribute regression. The specific architecture is shown in the Fig. 17. For the Gaussian attribute regression network, the inputs are the images from four viewpoints, which are concatenated together and fed into the network. The output is the corresponding Gaussian attributes for the corresponding view. Considering the resolution of input images, we only use ResNet blocks [7] to form the UNet-like network. The downsampling layer number and upsampling layer number are 3. For Gaussian attribute regression network, we use convolution dimensions of (16, 16, 16).

##### A.14. Discussion on Snap-Snap

We observe that the reconstruction results of Snap-Snap still contain some holes, particularly around the armpits or regions occluded by the arms. These missing parts in the hu-

man Gaussians are due to the limitations of point cloud supervision, which is derived from depth maps [44], wherein certain points are filtered out because of occlusions. The hollows on the human bodies could potentially be reduced with the use of geometric generative priors. As the occluded areas are relatively small, they have little effect on the consistency of the final reconstruction.

##### A.15. Discussion on GHG [20]

We further visualize the SMPL-X [32] parameters predicted based on EasyMocap [1] in Fig. 19, and it can be observed that estimating human body parameters using only two views is very challenging. The visualization results show that the predicted SMPL-X parameters have a significant impact on reconstruction methods based on the SMPLX model, such as GHG [20]. In contrast, our method reconstructs the human body based on directly predicted point clouds, effectively avoiding inaccuracies in SMPL-X parameter estimation.

Downsampling layers Upsampling layers

Block

Block

Block

Block

Block

Block

Block

1

2

3

4

5

6

7

Resnet

Resnet

Resnet

Resnet

Resnet

Resnet

Block

Block

Block

Block

Block

Block

Conv SiLU GroupNorm Resnet Block

Figure 17. The network architecture of the Gaussian attribute regression network.

Input 5 Views GPS-Gaussian Snap-Snap (Ours) GT

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

Figure 18. We visualize the comparison of reconstruction results between our method and GPS-Gaussian [59] with five input views.

[Figure 306]

Keypoints SMPL-X GHG Snap-Snap (Ours) GT

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

| |
|---|

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

| |
|---|

[Figure 322]

[Figure 323]

[Figure 324]

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

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

Figure 19. We visualize the reconstruction results of GHG [20] based on SMPL-X [32] parameters predicted by EasyMocap [1], as well as our reconstruction results in the same scenario. The first column shows the input views, the second column shows the detected keypoints, and the third column shows the corresponding predicted SMPL-X mesh visualization results. Columns 4-9 show the reconstruction results of GHG, our reconstruction results, and the ground truth, respectively.

