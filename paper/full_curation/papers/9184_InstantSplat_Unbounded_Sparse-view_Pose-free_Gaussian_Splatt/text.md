# InstantSplat: Sparse-view SfM-free Gaussian Splatting in Seconds

arXiv:2403.20309v6[cs.CV]3Jul2025

Zhiwen Fan1,2†∗, Wenyan Cong1∗, Kairun Wen3∗, Kevin Wang1, Jian Zhang3, Xinghao Ding3, Danfei Xu2,4, Boris Ivanovic2, Marco Pavone2,5, Georgios Pavlakos1, Zhangyang Wang1, Yue Wang2,6

*Authors contributed equally; † Z. Fan is the Project Lead.

1University of Texas at Austin 2Nvidia Research 3Xiamen University 4Georgia Institute of Technology 5Stanford University 6University of Southern California Project Website: https://instantsplat.github.io

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

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

⇧ ⇩

⇧ ⇩

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

[Figure 33]

⇩

⇩

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

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

⇧ ⇩

⇧ ⇩

⇧ ⇩

⇧ ⇩

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

- Figure 1. InstantSplat processes sparse-view, unposed images to reconstruct a radiance field, capturing detailed scenes rapidly without relying on Structure-from-Motion. Optimization occurs under self-supervision with the support of the large pretrained model, MASt3R [23].

## Abstract

ing gradient-based photometric error. Overall, InstantSplat achieves large-scale 3D reconstruction in mere seconds by reducing the required number of input views. It achieves an acceleration of over 20 times in reconstruction, improves visual quality (SSIM) from 0.3755 to 0.7624 than COLMAP with 3D-GS, and is compatible with multiple 3D representations (3D-GS, 2D-GS, and Mip-Splatting).

While neural 3D reconstruction has advanced substantially, it typically requires densely captured multi-view data with carefully initialized poses (e.g., using COLMAP). However, this requirement limits its broader applicability, as Structure-from-Motion (SfM) is often unreliable in sparseview scenarios where feature matches are limited, resulting in cumulative errors. In this paper, we introduce InstantSplat, a novel and lightning-fast neural reconstruction system that builds accurate 3D representations from as few as 2-3 images. InstantSplat adopts a self-supervised framework that bridges the gap between 2D images and 3D representations using Gaussian Bundle Adjustment (GauBA) and can be optimized in an end-to-end manner. InstantSplat integrates dense stereo priors and co-visibility relationships between frames to initialize pixel-aligned geometry by progressively expanding the scene avoiding redundancy. Gaussian Bundle Adjustment is used to adapt both the scene representation and camera parameters quickly by minimiz-

## 1. Introduction

3D reconstruction from a limited number of images has been a long-standing goal in computer vision. Conventional methods for 3D reconstruction utilize a complex and modular pipeline by decomposing this task into multiple subtasks, involving several stages of complex mappings between different data representations. For example, dense reconstruction using 3D Gaussian Splatting (3D-GS) [21] requires preprocessing from Structure-from-Motion (SfM), which first transforms images into key points, matches them, builds a scene graph, and incrementally reconstructs

the scene representation while optimizing camera parameters and structures. Then, the subsequent 3D-GS assumes accurate camera parameters and performs a lengthy optimization process driven by a multi-view photometric loss, complemented by Adaptive Density Control (ADC), a heuristic that governs the creation or deletion of 3D primitives. The processes of SfM and 3D-GS require densely captured multi-view images with large overlapping regions for feature matching [33] and Gaussian point densification [21], whereas the error derived from SfM can significantly degrade the 3D-GS optimization, poses challenges to the ADC process. As shown in Tab. 1, merely lowering the densification threshold or slightly perturb the initial camera parameters markedly affect the reconstruction quality.

Significant efforts have been made to alleviate these strict requirements by either reducing the required image number or estimating camera poses in during optimization. Recent advancements in sparse-view reconstruction [17, 31, 39, 48, 57] have shown notable progress by reducing the number of views from hundreds to just a few, though they still assume accurate camera poses obtained by leveraging dense views for pre-computation—an assumption that is rarely feasible. Another line of research explores pose-free settings in NeRFmm [42], Nope-NeRF [5], and CF-3DGS [13]. These approaches also assume dense data coverage, often sourced from video sequences. Such dense data requires an extensive optimization process not only assumes the known camera intrinsics [5, 13], but also usually taking hours to reconstruct a single 3D scene.

In this paper, we present InstantSplat, a system that builds large-scale 3D representations from sparse-view data, offering two distinct advantages: (1) it generalizes well to various scene types with minimal multi-view coverage (as few as 2 or 3 images) for high-quality reconstruction, and (2) it is lightning-fast thanks to its self-supervised paradigm with gradient-based photometric error.

To achieve this, InstantSplat switches from previous system incorporating sparse SfM with complex ADC [21] to a end-to-end optimization framework aided by geometric priors. Specifically, it introduces Gaussian Bundle Adjustment to link 2D and 3D, rendering 2D images from 3D representations and computing gradients based on the residual between rendered images and ground-truth images, jointly optimizing both the scene representation and camera parameters. To facilitate convergence, InstantSplat incorporates generalizable geometry priors from MASt3R [23] to obtain densely covered, pixel-aligned stereo points and subsequently adopts a co-visibility cross-view check to initialize the scene representation by expanding the points. The end-to-end framework requires only a few optimization steps with an adaptive confidence-aware learning rate, constructing an explicit 3D representation in just seconds. We evaluate InstantSplat on the Tanks and Temples [22] and

| |SSIM ↑ LPIPS ↓<br><br>|
|---|---|
|COLMAP+3DGS [21]<br><br>|0.7633 0.2122|
|+ Adjust ADC Threhold. + Mask 30% SfM Points ± 1◦ Noise on Rotation|0.7817(+0.0184) 0.1666(-0.0456) 0.7406(-0.0227) 0.2427(+0.0305) 0.3529(-0.4104) 0.5265(+0.3143)<br><br>|

Table 1. Sensitivity Analysis in Adaptive Density Control (ADC). The 3D-GS [21] utilizes camera parameters and sparse points from Structure-from-Motion (SfM). Adjusting the densification gradient threshold, an ADC parameter [21], to half significantly enhances rendering quality (LPIPS: 0.2122  → 0.1666). Conversely, non-uniform point distribution through 30% downsampling poses challenges (LPIPS: 0.2122  → 0.2427), and slight rotational perturbations in SfM poses lead to substantial performance drops (LPIPS: 0.2122  → 0.5265). Experiments conducted on the “Bicycle” scene from the MipNeRF360 [2] dataset.

MVImgNet [52] datasets. InstantSplat achieves a dramatic reduction in optimization time to just 7.5 seconds while significantly improving SSIM from 0.3755 to 0.7624 in a 3view setting, and it further enhances pose accuracy. Moreover, InstantSplat is compatible with other neural 3D representations, such as 2D-GS [15] for surface reconstruction and Mip-Splatting [53] for multi-resolution reconstruction, highlighting its robust generalization capability.

In summary, our main contributions are:

- • We propose a novel system for neural 3D reconstruction that transitions from traditional SfM, which rely on lengthy 3D-GS optimizations, to a rapid and selfsupervised framework. It leverages geometric priors and joint optimization to significantly reduce the required number of views, while effectively generalizing to diverse point-based representations.
- • We introduce Gaussian Bundle Adjustment to connect

- 2D and 3D, establishing an end-to-end paradigm to adjust scene representation and camera parameters based on gradient-based photometric error.

• We propose initializing the framework with a learnable dense model, equipped with co-visibility expansion to initialize dense surface points.

2. Related Works

- 3D Representations Novel view synthesis aims to generate unseen views of an object or scene from a given set of images [1, 29]. Neural Radiance Fields (NeRF)[30], enables photo-realistic rendering quality, employs MLPs to represent 3D scenes, taking the directions and positions of 3D points as input and employing volume rendering for image synthesis. Despite its popularity, NeRF faces challenges in terms of speed during both training and inference phases. Subsequent enhancements primarily focus on either enhancing quality[2–4] or improving efficiency [32, 35– 37], with few achieving both. Recent advancements in unstructured radiance fields[8, 21, 47] introduce a set of prim-

itives to represent scenes. Notably, 3D Gaussian Splatting (3D-GS) [21] uses anisotropic 3D Gaussians [59] to depict radiance fields, coupled with differentiable splatting for rendering. This method has shown considerable success in rapidly reconstructing complex real-world scenes with high quality. And many works extend 3D-GS for surface reconstruction [16, 54, 55], neural gaussians [27], multi-resolution modeling [12, 53] and feed-forward reconstruction [6, 7, 11, 14, 20, 24, 46, 49, 58]. However, these point-based representations involves a large number of hyperparameters for different datasets, especially for the adaptive density control, which servers as the core function to densify from sparse SfM point cloud to densely covered 3D Gaussian points. However, the optimization of SfM is independent with the followed 3D Gaussian optimization, and the accumulated error cannot be mitigated with current pipelines. This motivate us to design a holistic, efficient and robust framework to represent the 3D world.

Unconstraint Novel View Synthesis NeRFs and 3DGS require carefully captured hundreds of images to ensure sufficient scene coverage as input and utilize preprocessing Structure-from-Motion (SfM) software, such as COLMAP [33], to compute camera parameters, and sparse SfM point cloud as additional inputs. However, the densely captured images with the strong reliance on COLMAP significantly limits practical applications, it requires the users with expertise in the field of photography and requires significant computing resources (hours for each individual scene). The accumulated error from SfM will propogate to the following differential 3D representation and SfM may fail with captured images without sufficient overlappings and rich textures. To address the challenge of the requisite number of views, various studies have introduced different regularization techniques to optimize the radiance fields. For instance, Depth-NeRF [10] employs additional depth supervision to enhance rendering quality. RegNeRF [31] and SparseNeRF [39] introduces a depth prior loss for geometric regularization. DietNeRF [17], SinNeRF [45], and ReconFusion [43] leverages supervision in the CLIP/DINOViT/Diffusion Model to constrain the rendering of unseen views. PixelNeRF [51] and FreeNeRF [48] utilize pre-training and frequency annealing for few-shot NeRF. FSGS [57], and SparseGS [44] employ monocular depth estimators or diffusion models on Gaussian Splatting for sparse-view conditions. However, these methods require known ground-truth camera poses computed and sampled from using dense views in the preprocessing, and Structurefrom-Motion (SfM) algorithms often fail to predict camera poses and point cloud with sparse inputs due to insufficient image correspondences. Therefore, another line of research focuses on pose-free 3D optimization with uncalibrated images as direct input. NeRFmm [42] simultaneously optimizes camera intrinsics, extrinsics, and NeRF training.

BARF [25] introduces a coarse-to-fine strategy for encoding camera poses and joint NeRF optimization. SCNeRF [18] adds camera distortion parameterization and employs geometric loss for ray regularization. Similarly, GARF [18] demonstrates that Gaussian-MLPs facilitate more straightforward and accurate joint optimization of pose and scene. SPARF [38] adds Gaussian noise to simulate noisy camera poses. Recent works, such as Nope-NeRF [5], LuNeRF [9], LocalRF [28] and CF-3DGS [13], leverage depth information to constrain NeRF or 3DGS optimization. The work in [19] utilizes monocular depth estimator and additional image matching network try to reduce the view number, with roughly hours of optimization for a single scene. These pose-free works generally presume the input are dense video sequences [5, 13] with known viewing order and camera intrinsics [5, 13, 19], and the optimization for each scene (usually several hours) is even longer than COLMAP with NeRF or 3D-GS variants.

In contrast, InstantSplat proposes reconstructing a 3D neural representation using only 2D images, jointly optimizing camera poses and the 3D model aided by geometric priors from the large-scale trained model, MASt3R [23]. We categorize InstantSplat as a self-supervised framework following INeRF’s definition [50], where it employs a trained NeRF model to estimate camera poses.

## 3. Method

Overview Our optimization framework is a selfsupervised structure that leverages photometric error to guide the training of the 3D representation and camera poses, as illustrated in Fig. 2. In Sec. 3.1, we provide background on 3D-GS and the 3D prior model. Following this, in Sec. 3.2, we describe our self-supervised training scheme and the development of the end-to-end optimizable framework. For the initialization of the 3D representation, we utilize a prior model to obtain regressed point maps and camera parameters, enhanced by a co-visibility-based geometric initialization (Sec. 3.3). The overall gradientbased joint optimization framework utilizes photometric error to align 3D Gaussians with 2D images, employing a confidence-aware optimizer with few optimization steps (Sec. 3.4).

##### 3.1. Preliminary

3D Gaussian Splatting (3D-GS) [21] is an explicit 3D scene representation utilizing a set of 3D Gaussians to model the scene. A 3D Gaussian is parameterized by a mean vector x ∈ R3, an opacity α and a covariance matrix Σ ∈ R3×3:

- 1

- 2

G(p,α,Σ) = α exp −

(p − x)TΣ−1 (p − µ) (1)

|𝑦|
|---|

|3D Representation<br><br>[Figure 77]| |
|---|---|
| |𝑥|

- 2D to 3D Operator (unprojection)

[Figure 78]

|[Figure 79]|
|---|

[Figure 80]

Images

Correspondence Search Feature Extraction

Matching

Geometric Verification

Initialization

Incremental SfM Reconstruction

Image Registration

Triangulation

Outlier Filtering

Bundle Adjustment

Dense 3D Reconstruction

Initializing 3D Gaussian

Rasterizer

Adaptive Density Control

Representative 3D Gaussian Splatting System

Multi-view Captures

(unposed, uncalibrated) InstantSplat System

Encoder Decoder

Prior Model

[Figure 81]

|(𝑻𝟎, 𝑮𝟎)|
|---|

Loss (Rasterizer(𝑇𝑣, 𝐺𝑛),RGB(𝑇𝑣, 𝐺𝑛))

3D Gaussian

Splatting

|[Figure 82]|
|---|

|[Figure 83]|
|---|

(𝑻𝒗, 𝑮𝒏)

Figure 2. Overall Framework of InstantSplat. Unlike the modular COLMAP pipeline with Gaussian Splatting, which relies on timeconsuming and accuracy-sensitive ADC processes within 3D-GS, and accurate camera poses and sparse point clouds from SfM, InstantSplat employs a deep model to initialize dense surface points. It adopts an end-to-end framework that iteratively optimizes both

- 3D representation and camera poses, enhancing efficiency and robustness.

[Figure 84]

[Figure 85]

|[Figure 86]|
|---|

|𝑧|
|---|

[Figure 87]

[Figure 88]

3D to 2D Operator

(Rasterization)

To represent the view-direction-dependent, spherical harmonic (SH) coefficients are attached to each Gaussian, and the color is computed via c(d) = ni=1 ciBi (d), where Bi is the ordi SH basis. And the color is rendered via c = ni=1 ciαi ij−=11(1 − αj), where ci is the color computed from the SH coefficients of the ordi Gaussian. αi is given by evaluating a 2D Gaussian with covariance multiplied by a learned per-Gaussian opacity. The 2D covariance matrix is calculated by projecting the 3D covariance to the camera coordinate system. The 3D covariance matrix is decomposed into a scaling matrix and a rotation matrix. In summary, 3D-GS uses a set of 3D Gaussians {Gi|i = 1,2,...,n} to represent a scene, where each 3D Gaussian Gi is characterized by: position x ∈ R3, a series of SH coefficients ci ∈ R3|i = 1,2,...,n , opacity α ∈ R, rotation q ∈ R4 and scaling s ∈ R3.

Distinguished from 3D-GS, 2D-GS [15] proposes that reducing one scaling dimension to zero and approximating the primitive as a surface can improve view consistency. To implement this, 2D-GS replaces the volumetric representations used in 3D-GS with planar disks, which are defined within the local tangent plane and mathematically expressed as follows:

u(p)2 + v(p)2 2

G(p) = exp −

(2)

where u(p) and v(p) are local UV space coordinates.

During the rendering process, 2D-GS substitutes direct affine transformations with three non-parallel planes to define ray-splat intersections. Following this, rasterization is applied after a low-pass filter to produce smoother results.

MASt3R is a feed-forward Multi-View Stereo model where it regresses the pixel-aligned point maps directly from raw images. It utilize a Transformer architecture with cross-view information flow to predict the Pˆ1,1 and Pˆ2,1, obtained from two corresponding views {1,2} where the camera origin as view 1, on which the ground-truth is de-

fined. The regression loss for training MASt3R is as:

Lreg =

1 zi · Pv,1 −

1 zi · Pˆv,1 (3)

The view v ∈ {1,2}, P and Pˆ are the predicted and groundtruth point maps, separately. zi = norm(P1,1,P2,1) is the normalization factor. It also regresses the pixel-aligned confidence score Oiv,1 and optimizes as:

Oiv,1 · Lreg(v,i) − α · log Oiv,1, (4)

Lconf =

v∈{1,2} i∈Dv

where α is a hyper-parameter controlling the regularization term, encouraging the network to extrapolate in harder areas.

##### 3.2. Optimizating 3D with Self-supervision

Gaussian Bundle Adjustment Given multiple views and a 3D representation with initialized poses, represented by a set of Gaussians G, and T, we employ the pre-built rasterizer from Gaussian Splatting as a differentiable operator to render the images corresponding to the target poses T. Gradient descent with photometric error minimization is used to reduce the discrepancy between the optimizable 3D representation and 2D observations, iteratively refining G, and T toward representative orientations:

HW

G∗,T∗ = arg min

###### G,T

v∈N

i=1

###### C ˜iv(G,T) − Civ(G,T)

The C denotes the rasterization function applied to G at T, while C˜ represents the observed 2D images.

We initialize G and T using a post-processing step following the pairwise predictions inferred by MASt3R, which we will elaborate on later.

Aligning Camera Poses on Test Views Unlike benchmark datasets where camera pose estimation and calibration are pre-computed with access to all training and testing views, a pose-free setting requires additional steps to regress the poses for the test views. Similar to INeRF [42], we freeze the 3D model learned by training images, while optimizing the camera poses for the test views. This optimization process focuses on minimizing the photometric discrepancies between the synthesized images and the actual images on test views, aiming to achieve alignment between the 3D model and test images for evaluating visual quality and pose estimation metrics.

##### 3.3. Co-visible Global Geometry Initialization

Scaling Pairwise to Global Alignment With the off-theshell stereo model, MASt3R, we regress a set of stereo point maps along with calculated camera intrinsics and extrinsics following Sec. 3.1. To align these pair wise predictions into a globally aligned results, we follow MASt3R to build a complete connectivity graph G(V,E) of all the N input views, where the vertices V and each edge e = (n,m) ∈ E indicate a shared visual content between images In and Im. To convert the initially predicted point map {(Pi ∈ RH×W×3)}Ni=1 to be a globally aligned one {(P˜i ∈ RH×W×3)}Ni=1, we follow the MASt3R to update these point maps, transformation matrix, and a scale factor into globally corrdinate system. Please refer to the supplementary material for detailed description. However, the pairwise predicted focal lengths remain inconsistent post-optimization. Assuming a single-camera setup akin to COLMAP for capturing a scene, we stabilize the estimated focal lengths by averaging them across all training views: f¯ = N1 Ni=1 fi∗ which is simple but significantly improve optimized visual quality.

Co-visible Redundancy Elimination Utilizing point maps from all training images typically results in redundancy, manifested through overlapping regions that produce duplicated, pixel-aligned point clouds. This not only increases the free parameters in scene representation but also slows down the optimization process. To combat this, we exploit co-visibility across training images to eliminate these duplicate pixels. View ranking is conducted by computing the average confidence levels from the confidence maps of each image, with a higher average indicating a more accurate initial geometry from the 3D prior model. We prune points from views with lower confidence scores to enhance point quality, measured by the view confidence score,

1 |Oi| o∈O

o (5)

vi =

i

where vi denotes the average confidence score of view i.

Cross-view checking starts with the view (j) having the lowest average confidence, where we combine point maps from views with higher average confidence score than view

j, {(P˜i ∈ RH×W×3)}Ni=1,i̸=j,vi>vj, and project them onto view j. Depth ambiguity is addressed through a depth ver-

ification process, pruning points only if the difference between the projected and original depth values is higher than a predefined threshold: ∆d = |dproj −dorig|. This procedure is iterated in ascending order across all images, ultimately generating a binary mask {(M˜ i ∈ RH×W×1)}Ni=1 to filter out redundant points. More formally, the cross-view covisibility check for removing redundancy in view j can be written as:

Projj(P˜i) (6)

Dproj,j =

{i|vi>vj}

1 if |Dproj,j − Dorig,j| < θ 0 otherwise

(7)

Mj =

Pj = (1 − Mj) · P˜j (8) where Proj is the function applying the projection operator across all views except view i (vi ≥ vj). Mj is the visibility mask, determined by comparing the projected depth dproj,j to the original depth dorig,j. Points with depth differences below a threshold θ are considered visible and consistent, thus redundant. We use inverse of Mj to mask out redundant points.

##### 3.4. Optimization

Confidence-aware Optimizer In our pursuit to refine the accuracy of scene reconstruction, we introduce a confidence-aware per-point optimizer designed to facilitate convergence during self-supervised optimization. This approach is tailored to dynamically adjust learning rates based on the per-point confidence as detailed in Sec. 3.3. By prioritizing points with lower confidence, which are more susceptible to errors, we ensure a targeted and efficient optimization process. The calibration of confidence scores involves normalizing them into a proper range, thereby addressing points in need of significant corrections. The normalization and adjustment process is mathematically outlined as follows: Onorm = (1 − sigmoid(Oinit)) · β where β is a hyperparameter used to fine-tune the scaling effect, optimizing the learning rates to enhance the convergence.

Training Objective To optimize our neural 3D representation efficiently, we employ purely visual signals, using photometric loss to minimize discrepancies between the rasterized images from our 3D model, denoted as C˜(G,T), and the observed images C. Overall, the process maps observed unposed RGB images, to a 3D representation G∗ with accurate pose estimations T∗, utilizing end-to-end iterative optimization and a self-supervised training objective. InstantSplat aligns scene representations and camera

poses toward visual observation, improving rendering accuracy and optimization efficiency without additional heuristic regularization.

mechanism efficiently adapts parameters through the use of photometric loss, applicable to both camera and Gaussian parameters.

## 4. Experiments 4.1. Experimental Setup

Datasets. Following methodologies from previous posefree research [5, 13], we evaluated our InstantSplat on eight scenes from the Tanks and Temples dataset [22] with view counts ranging from 3 to 12. Additionally, we tested on seven outdoor scenes from the MVImgNet dataset [52], covering diverse scene types including Car, Suv, Bicycle, Chair, Ladder, Bench, and Table. Our experiments also extend to in-the-wild data, incorporating frames from the Sora video (link), and navigation stereo camera data from the Perseverance Rover (NASA), along with three randomly sampled training views from the DL3DV-10K dataset [26].

Train/Test Datasets Split. For both training and evaluation, 12 images from each dataset were sampled, spanning the entire set. The test set includes 12 images uniformly chosen to exclude the first and last frames, while the training set comprises an equal number of images selected from the remainder, ensuring sparse-view training setting. This uniform sampling approach was applied consistently across both MVImgNet and Tanks and Temples datasets.

Metrics. We evaluate our approach on benchmark datasets [22, 52] across three tasks: novel view synthesis, camera pose estimation, and depth estimation. For novel view synthesis, we use Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM) [41], and Learned Perceptual Image Patch Similarity (LPIPS) [56]. For camera pose estimation, we report the Absolute Trajectory Error (ATE) as defined in [5], using COLMAP poses from all dense views as ground-truth references. For depth estimation, we evaluate using the Absolute Relative Error (Rel) and the Inlier Ratio (τ) with a threshold of 1.03, following DUSt3R [40].

Baselines. Our comparisons on pose-free methods include Nope-NeRF [5] and CF-3DGS [13], both supported by monocular depth maps and ground-truth camera intrinsics. We also consider NeRFmm [42], which involves joint optimization of NeRF and all camera parameters. Additionally, we compare our method with 3D-GS [21] and FSGS [57], which utilize COLMAP for pre-computing the camera parameters.

Implementation Details. Our implementation is built using the PyTorch framework. Optimization iterations are set to 200 for Ours-S and 1,000 for Ours-XL, striking a balance between quality and training efficiency. During evaluation, 500 iterations are used for test view optimization. The parameter β is fixed at 100. For multi-view stereo (MVS) depth map prediction, MASt3R is configured with a resolution of 512 and the cupy library is used to accelerate the initialization. All experiments are conducted on a single Nvidia A100 GPU to ensure fair comparisons.

##### 4.2. Experimental Results

Quantitative and Qualitative Results on Novel-view Synthesis We evaluate novel-view synthesis and pose estimation on the Tanks and Temples dataset, with results summarized in Tab.2 and Fig.4.

Nope-NeRF [5], which leverages Multilayer Perceptrons (MLPs), delivers promising pose estimation accuracy with dense video sequences. However, it often generates overly blurred images (see the third column in Fig. 4), primarily due to the heavy constraints imposed by its geometric field. Additionally, it suffers from prolonged training times for a single scene and inference delays (∼3 seconds per frame for rendering one image). CF-3DGS [13], which employs Gaussian Splatting with local and global optimization stages along with adaptive density control and opacity reset policies, encounters artifacts when rendering from novel viewpoints. These artifacts result from its complex optimization pipeline and erroneous pose estimations, as shown in the second column of Fig. 4. Moreover, both NopeNeRF and CF-3DGS assume accurate focal lengths, limiting their robustness in scenarios with focal length uncertainties. NeRFmm, designed for simultaneous optimization of camera parameters and the radiance field, often struggles with suboptimal results due to the inherent challenges of naive joint optimization.

Pose metrics, as reflected in Tab. 2, reveal significant artifacts caused by sparse observations and inaccurately estimated poses. These limitations are particularly pronounced in CF-3DGS and Nope-NeRF, which depend on dense video sequences, akin to SLAM, and encounter difficulties when these dense frames are downsampled to sparse multi-view images. In contrast, our method initializes with the MVS scene structure and employs a gradient-based joint optimization framework, providing enhanced robustness and superior performance.

We demonstrate the geometric accuracy of InstantSplat by integrating it with 2D Gaussian Splatting, visualizing rasterized depth and normal maps. As shown in Fig. 3 and Tab. 6, under the 3-view setting, InstantSplat significantly improves geometry quality, benefiting from dense initialization and the joint optimization framework. Additionally, we validate InstantSplat within the multi-resolution Mip-

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

2D-GS SSIM⇧: 0.6611 LPIPS⇩: 0.2468 Time⇩ : 9min35s

IS-2DGS2DGS

[Figure 93]

[Figure 94]

[Figure 95]

IS-2DGS SSIM⇧: 0.8250 LPIPS⇩: 0.1237 Time⇩ : 42s

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

2D-GS SSIM⇧: 0.5672 LPIPS⇩: 0.3570 Time⇩ : 9min30s

IS-2DGS2DGS

[Figure 100]

[Figure 101]

[Figure 102]

IS-2DGS SSIM⇧: 0. 8154 LPIPS⇩: 0.1419 Time⇩ : 47s

Color Depth Normal Ground-Truth

- Figure 3. Visual Comparisons between 2D-GS and InstantSplat-2D-GS under 3-view setting. We present the rendering results of RGB images, depth maps, and normal maps from the Tanks and Temples dataset. The corresponding SSIM, LPIPS, and training time metrics are displayed on the right. Here, IS is used as an abbreviation for InstantSplat.

- Table 2. Quantitative Evaluations on Tanks and Temples Datasets. Our method achieves significantly clearer details (lower LPIPS) compared to other pose-free methods, even with an optimization time of only ∼7.5 seconds (Ours-S), and avoids artifacts caused by noisy pose estimation, as quantified by Absolute Trajectory Error (ATE). Extending training time further improves rendering quality. InstantSplat also demonstrates superior camera pose estimation accuracy in ATE with ground truth scale. In 3D-GS, ADC refers to Adaptive Density Control [21]. We further leverage cupy to accelerate MASt3R inference.

| |Training Time 3-view 6-view 12-view<br><br>|SSIM 3-view 6-view 12-view|LPIPS 3-view 6-view 12-view<br><br>|ATE 3-view 6-view 12-view|
|---|---|---|---|---|
|COLMAP + 3DGS [21] COLMAP + FSGS [57]|4min28s 6min44s 8min11s 2min37s 3min16s 3min49s<br><br>|0.3755 0.5917 0.7163 0.5701 0.7752 0.8479<br><br>|0.5130 0.3433 0.2505 0.3465 0.1927 0.1477<br><br>|- - -<br>- - -<br>|
|NoPe-NeRF [5] CF-3DGS [13]<br><br>|33min 47min 84min 1min6s 2min14s 3min30s<br><br>|0.4570 0.5067 0.6096 0.4066 0.4690 0.5077|0.6168 0.5780 0.5067 0.4520 0.4219 0.4189<br><br>|0.2828 0.1431 0.1029 0.1937 0.1572 0.1031|
|NeRF-mm [42]|7min42s 14min40s 29min42s<br><br>|0.4019 0.4308 0.4677|0.6421 0.6252 0.6020<br><br>|0.2721 0.2329 0.1529|
|Our-S Our-XL|7.5s 13.0s 32.6s 25.4s 29.0s 44.3s<br><br>|0.7624 0.8300 0.8413 0.7615 0.8453 0.8785<br><br>|0.1844 0.1579 0.1654 0.1634 0.1173 0.1068|0.0191 0.0172 0.0110 0.0189 0.0164 0.0101<br><br>|

##### 4.3. Ablations and Analysis

Splatting framework. Tab. 7 shows evaluations on the Tanks and Temples dataset using the original resolution for testing while varying the training resolution. InstantSplat consistently enhances sparse-view rendering metrics.

We conducted ablation studies to validate our design choices, transitioning from the use of non-differentiable Structure-from-Motion (SfM) complemented by Gaussian Splatting with adaptive density control, to adopting MultiView Stereo (MVS) with Co-visible Global Geometry Initialization, Confidence-aware Optimizer and Gaussian Bundle Adjustment for efficient and robust sparse-view 3D modeling. Experiments were conducted using 12 training views on all scenes in the Tanks and Temples dataset, with COLMAP poses derived from full dense views serving as ground truth, unless otherwise specified.

Further experiments on the MVImgNet dataset with 3, 6, and 12 training images are presented in Tab. 3. This section demonstrates that InstantSplat consistently outperforms all baselines across all evaluated datasets and metrics. Please refer to our supplementary video for additional visualizations, including in-the-wild test cases such as the Sora video, DL3DV-10K dataset, and stereo image pairs from the Mars rover.

• Question 1: How does the quality of multi-view stereo predictions impact performance?

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

| |
|---|

MVImgNetTankandTemples

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

| |
|---|

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Ours CF-3DGS Nope-NeRF Ground-Truth

NeRF-mm

- Figure 4. Visual Comparisons between InstantSplat and various pose-free baseline methods. InstantSplat achieves faithful 3D reconstruction and renders novel views with as few as three training images on the Tanks and Temples datasets, and the MVImgNet datasets.

- Table 3. Quantitative Evaluations on MVImgNet Datasets. Our method renders significantly clearer details (by LPIPS) compared to other pose-free methods, devoid of artifacts typically associated with noisy pose estimation (e.g., CF-3DGS [13], NeRFmm [42]) and NoPe-NeRF [5]. COLMAP on full views as the ground-truth reference. Our method produces more accurate camera pose estimation in the Absolute Trajectory Error (ATE) than previous COLMAP-free methods, quantified using the ground truth scale. ADC is denoted as Adaptive Density Control [21]. Cupy library is used to accelerate the initialization.

| |Training Time 3-view 6-view 12-view<br><br>|SSIM 3-view 6-view 12-view<br><br>|LPIPS 3-view 6-view 12-view|ATE 3-view 6-view 12-view<br><br>|
|---|---|---|---|---|
|NoPe-NeRF [5] CF-3DGS [13]<br><br>|37min42s 53min 95min 3min47s 7min29s 10min36s<br><br>|0.4326 0.4329 0.4686 0.3414 0.3544 0.3655|0.6674 0.6614 0.6257 0.5523 0.4326 0.4492<br><br>|0.2780 0.1740 0.1493 0.1593 0.1981 0.1243|
|NeRF-mm [42]<br><br>|8min35s 16min30s 33min14s<br><br>|0.3752 0.3685 0.3718|0.7001 0.6252 0.6020<br><br>|0.2721 0.2376 0.1529|
|Our-S Our-XL<br><br>|10.4s 15.4s 34.1s 35.7s 48.1s 76.7s|0.5489 0.6835 0.7050 0.5628 0.6933 0.7321<br><br>|0.3941 0.2980 0.3033 0.3688 0.2611 0.2421|0.0184 0.0259 0.0165 0.0184 0.0259 0.0164<br><br>|

- Table 4. Ablation Study for validating design choices. We opt for the XL setting, where 1,000 iterations are adopted to ensure reconstruction quality. Results are averaged on all scenes of Tank and Temple datasets while Cupy library is adopted.

accurate rendering?

• Question 5: Can InstantSplat achieve rendering quality comparable to methods using dense-view images, but with sparse-view inputs?

- For Q1, the evaluation of camera pose accuracy from MASt3R reveals significant discrepancies in rendering quality and pose accuracy, as shown in Tab. 4’s first and last rows.
- For Q2, aggregating focal lengths from all images stabilizes the 3D representation optimization, evidenced by results in the second row.
- For Q3, co-visible geometry initialization markedly improves FPS by reducing redundancy. Confidence-based view ranking further enhances rendering accuracy, as detailed in the third and fourth rows.
- For Q4, Gaussian Bundle Adjustment (GauBA), through

Model Train↓ FPS↑ SSIM↑ LPIPS↓ ATE↓

|Baseline by Stitching 51s 150|0.8014 0.1654<br><br>|0.0142|
|---|---|---|
|+ Focal Averaging 51s 150 + Co-visibility Geometry Init. 45s 181 + Confidence-aware Optimizer 45s 181 + Gaussian Bundle Adjustment 45s 181|0.8406 0.1288 0.8411 0.1313 0.8553 0.1277 0.8822 0.1050<br><br>|0.0115 0.0115 0.0115 0.0102|

- • Question 2: What is the impact of focal averaging on the results?
- • Question 3: Does co-visible geometric initialization effectively reduce redundancy and preserve reconstruction accuracy?
- • Question 4: Is Gaussian Bundle Adjustment essential for

- Table 5. Performance comparison between InstantSplat (ours) with 12 training views, CF-3DGS and 3D-GS with dense (100-

400) training views on the Tanks and Temples dataset. We report the LPIPS, reconstruction time, and pose accuracy.

| |Views ↓ Time ↓ LPIPS ↓ ATE ↓|
|---|---|
|CF-3DGS [13] COLMAP+3DGS [21] InstantSplat (Ours)|100-400 ∼30min 0.09 0.004<br><br>100-400 ∼30min 0.10 -<br><br><br>12 ∼45s 0.10 0.010|

- Table 6. InstantSplat with 2D-GS. InstantSplat demonstrates strong compatibility with 2D-GS, notably improving sparse-view rendering quality and depth accuracy while also accelerating training by eliminating the need for an additional SfM step. Cupy library is used to accelerate the initialization.

| |View Num.<br><br>|Train Time<br><br>|Novel View Synthesis|Depth Estimation|
|---|---|---|---|---|
| | | |PSNR↑ SSIM↑ LPIPS↓<br><br>|rel↓ τ↑|
|2D-GS +InstantSplat<br><br>|3 6 12|45s 58s 98s<br><br>|21.90 0.7447 0.1920 24.99 0.8281 0.1450 26.07 0.8557 0.1407|28.7557 26.1918<br><br>29.9461 25.7376<br><br><br>29.4234 25.3569|
|2D-GS|3 6 12<br><br>|10min3s<br>11min22s 11min54s<br>|15.09 0.5495 0.3719 19.88 0.7251 0.2388 23.70 0.8184 0.1723<br><br>|59.8629 5.2962 45.4608 17.8145 40.1948 20.1998|

- Table 7. InstantSplat with Mip-Splatting. Both methods were trained in sparse view settings, with variations in training and evaluation resolutions consistent with Mip-Splatting protocols. InstantSplat demonstrates compatibility with Mip-Splatting, significantly enhancing sparse-view rendering quality.

| |Training Resolution|3-view<br><br>|12-view|
|---|---|---|---|
| | |SSIM LPIPS|SSIM LPIPS|
|Mip-Splatting + Ours-XL<br><br>|1 1/2 Res. 1/4 Res.|0.7647 0.1618 0.7456 0.2343 0.6630 0.3774<br><br>|0.8415 0.1305 0.7945 0.2027 0.7204 0.3348<br><br>|
|Mip-Splatting|1 1/2 Res. 1/4 Res.<br><br>|0.5422 0.3780 0.5777 0.3939 0.5477 0.4840|0.7947 0.1901 0.7703 0.2418 0.6664 0.3770<br><br>|

self-supervised optimization, substantially improves visual and pose metrics by aligning the 3D and 2D representations. For Q5, InstantSplat using 12 training images approaches the rendering quality of denser-view methods as measured by the Learned Perceptual Image Patch Similarity (LPIPS), indicative of near-human perceptual sharpness. However, it exhibits discrepancies in pose estimation accuracy (ATE), inevitably leading to lower PSNR metrics (Tab. 5).

## 5. Conclusion

InstantSplat is a system tailored to reconstruct large-scale scenes from sparse-view, unposed images. Operating within a self-supervised framework, InstantSplat optimizes camera poses and 3D representations using photometric signals and avoids the time-consuming Adaptive Density Control. The system leverages multi-view stereo priors for surface point initialization and employs co-visible cross-view checking to minimize model redundancy. Compared with previous bestperforming pose-free methods [5, 13], InstantSplat dramatically reduces the required number of views from hundreds

to just a few and demonstrates robustness across in-the-wild datasets, as well as adaptability to various point-based representations.

However, InstantSplat faces limitations when applied to scenes with more than hundreds of images due to its reliance on MVS for globally aligned point clouds. Addressing this through progressive alignment techniques will be a focus of future research.

## A6. Details of Global Geometry Initialization

We provide additional details on the scaling process for pairwise-to-global alignment as discussed in Co-visible Global Geometry Initialization (Section 3.3 of the main draft).

Specifically, the geometric prior model, MASt3R, originally uses image pairs as input, requiring post-processing to align scales when more than two views are captured from the scene. This necessity arises because the predicted point maps are generated at varying scales. Such scale misalignment across independently computed relative poses introduces variance, leading to inaccuracies in camera pose estimation.

To integrate these pairwise pointmaps and camera information into a globally aligned geometry, we follow MASt3R [23] to construct a complete connectivity graph G(V,E) of all the N input views, where the vertices V and each edge e = (n,m) ∈ E indicate a shared visual content between images In and Im. To convert the initially predicted point map {(Pi ∈ RH×W×3)}Ni=1 to be a globally aligned one {(P˜i ∈ RH×W×3)}Ni=1, we update the point maps, transformation matrix, and a scale factor: For the complete graph, any image pair e = (n,m) ∈ E, the point maps Pn,n,Pm,n and confidence maps On,n,Om,n. For clarity, let us define Pn,e := Pn,n, and Pm,e := Pm,n, On,e := On,n, and Om,e := Om,n. The optimization for the transformation matrix of edge Te, scaling factor σe and point map P˜ are given by:

HW

P˜∗ = arg min

Ov,ei P ˜vi − σeTePv,ei . (9)

P˜,T ,σ e∈E v∈e

i=1

Here, we slightly abuse notation and use v ∈ e for v ∈ {n,m} if e = (n,m). The idea is that, for a given pair e, the same rigid transformation Te should align both pointmaps Pn,e and Pm,e with the world-coordinate pointmaps P˜n,e and P˜m,e, since Pn,e and Pm,e are by definition both expressed in the same coordinate frame. To avoid the trivial optimum where σe = 0, ∀e ∈ E, we enforce e σe = 1. Having aligned the point clouds, we commence the integration process by initializing 3D Gaussians [21], treating each point as a primitive. This post-processing step yields a globally aligned 3D Gaussians within only seconds, where infer-

ring per-view point and confidence maps can be achieved real-time on a modern GPU.

##### A6.1. Additional Implementation Details.

We implement InstantSplat-2D-GS using the original 2DGS CUDA kernel. All core components of our system are integrated into 2D-GS, including Focal Averaging, covisible Global Geometry Initialization, Confidence-aware Optimization, and Gaussian Bundle Adjustment. The optimization iterations are set to 1,500 for Gaussian training and 500 for aligning test view camera poses. Regularization for depth distortion and normal consistency begins at iterations 500 and 700, respectively.

##### A6.2. Evaluation on 2D-GS

In Table 6, we present the performance of InstantSplat2D-GS on the tasks of novel view synthesis and multiview stereo depth estimation using the Tanks and Temples dataset [22]. Following the train/test dataset split scheme described in the main draft, we sample 12 images for training and 12 for evaluation, covering the entire dataset. The test set contains 12 uniformly selected images (excluding first/last frames), with an equal number of training images sampled from remaining frames for sparse-view training. For novel view synthesis, we evaluate performance using PSNR, SSIM, and LPIPS. For depth estimation, we use the Absolute Relative Error (Rel) and Inlier Ratio (τ) with a threshold of 1.03 to assess each scene, following the approach of DUSt3R [40]. We align the scene scale between the predictions and the ground truth, as our system does not depend on camera parameters for prediction. We normalize the predicted depth maps using the median of the predicted depths and similarly normalize the ground truth depths, which follows the procedures established in previous studies [34, 40] to ensure alignment between the predicted and ground truth depth maps. We use COLMAP [33] to perform dense stereo reconstruction on the entire video sequence, generating a high-quality multi-view stereo point cloud. The 3D dense points are then projected onto 2D image planes to create pseudo ground-truth depth maps for evaluation.

## A7. More Experimental Results

##### A7.1. Additional Qualitative Results on the Benchmarks

We provide more visualized results and pose estimations on realistic outdoor scenes from the Tanks and Temples dataset [22] and MVImgNet [52]. As shown in Fig.A9 and Fig.A7, InstantSplat consistently produces fewer artifacts than Gaussian-based methods (e.g., CF-3DGS [13]) and exhibits sharper details than NeRF-based methods (e.g., Nope-NeRF [5] and NeRFmm [42]).

##### A7.2. Video Demonstration

We provide free-viewpoint rendering on various in-the-wild datasets, including the Sora video, by uniformly sampling three frames from the entire video sequence. Additionally, we use stereo camera data from the Perseverance Rover, available on the NASA website, and randomly sample three training views from the DL3DV-10K dataset [26]. As shown in Fig. A8 and Fig. A9, InstantSplat achieves accurate 3D reconstruction and high-quality novel view synthesis with as few as three training images, highlighting the robustness and effectiveness of our framework. For additional visualizations, please refer to our instantsplat-webpage.zip with opening the “index.html”.

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

2D-GS SSIM⇧: 0.5068 LPIPS⇩: 0.3908 Time⇩ : 9min12s

IS-2DGS2DGS

[Figure 127]

[Figure 128]

[Figure 129]

IS-2DGS SSIM⇧: 0.6526 LPIPS⇩: 0.2514 Time⇩ : 46s

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

2D-GS SSIM⇧: 0.5415 LPIPS⇩: 0.3124 Time⇩ : 11min25s

IS-2DGS2DGS

[Figure 134]

[Figure 135]

[Figure 136]

IS-2DGS SSIM⇧: 0.6884 LPIPS⇩: 0.1812 Time⇩ : 40s

Color Depth Normal Ground-Truth

- Figure A5. Additional Visual Comparisons between 2D-GS and InstantSplat-2D-GS. We present the rendering results of RGB images, depth maps, and normal maps from the Tanks and Temples dataset. The corresponding SSIM, LPIPS, and training time metrics are displayed on the right. Here, IS is used as an abbreviation for InstantSplat.

Ours-XL CF-3DGS Nope-NeRF NeRF-mm Ground-Truth

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

- Figure A6. Additional Visual Comparisons on the Tanks and Temples Dataset. We present the rendering results from NeRFmm, Nope-NeRF, CF-3DGS, and Ours-XL.

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

Ours-XL CF-3DGS Nope-NeRF NeRF-mm GT

- Figure A7. Additional Visual Comparisons on the MVimgNet Dataset. We present the rendering results from NeRFmm, Nope-NeRF, CF-3DGS, and Ours-XL.

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

TanksandTemples

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

### Free-view rendering

Figure A8. Free-view Rendering on the Tanks and Temples Dataset.

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

DL3DVMarsSora

[Figure 211]

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

#### Free-view rendering

Figure A9. Free-view Rendering on in-the-wild data. We present the rendering results of three different in-the-wild data: Mars Rover Navigation, Sora Video and DL3DV-10K dataset

## References

- [1] Shai Avidan and Amnon Shashua. Novel view synthesis in tensor space. In Proceedings of IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 1034–1040. IEEE, 1997. 2
- [2] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5460–5469, 2022. 2
- [3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5470–5479, 2022.
- [4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-based neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19697–19705, 2023. 2
- [5] Wenjing Bian, Zirui Wang, Kejie Li, Jia-Wang Bian, and Victor Adrian Prisacariu. Nope-nerf: Optimising neural radiance field with no pose prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4160–4169, 2023. 2, 3, 6, 7, 8, A9, A10
- [6] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19457–19467, 2024. 3
- [7] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision, pages 370–386. Springer, 2025. 3
- [8] Zhang Chen, Zhong Li, Liangchen Song, Lele Chen, Jingyi Yu, Junsong Yuan, and Yi Xu. Neurbf: A neural fields representation with adaptive radial basis functions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4182–4194, 2023. 2
- [9] Zezhou Cheng, Carlos Esteves, Varun Jampani, Abhishek Kar, Subhransu Maji, and Ameesh Makadia. Lu-nerf: Scene and pose estimation by synchronizing local unposed nerfs. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18312–18321, 2023. 3
- [10] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised nerf: Fewer views and faster training for free. arXiv preprint arXiv:2107.02791, 2021. 3
- [11] Zhiwen Fan, Jian Zhang, Wenyan Cong, Peihao Wang, Renjie Li, Kairun Wen, Shijie Zhou, Achuta Kadambi, Zhangyang Wang, Danfei Xu, et al. Large spatial model: End-to-end unposed images to semantic 3d. arXiv preprint arXiv:2410.18956, 2024. 3
- [12] Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang, Tao Liu, Zhilin Pei, Hengjie Li, Xingcheng Zhang, and Bo Dai. Flashgs: Efficient 3d gaussian splatting for

- large-scale and high-resolution rendering. arXiv preprint arXiv:2408.07967, 2024. 3
- [13] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A Efros, and Xiaolong Wang. Colmap-free 3d gaussian splatting. arXiv preprint arXiv:2312.07504, 2023. 2, 3, 6, 7, 8, A9, A10
- [14] Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jisang Han, Jiaolong Yang, Chong Luo, and Seungryong Kim. Pf3plat: Pose-free feed-forward 3d gaussian splatting. arXiv preprint arXiv:2410.22128, 2024. 3
- [15] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In SIGGRAPH 2024 Conference Papers. Association for Computing Machinery, 2024. 2, 4
- [16] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 3
- [17] Ajay Jain, Matthew Tancik, and Pieter Abbeel. Putting nerf on a diet: Semantically consistent few-shot view synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5885–5894, 2021. 2, 3
- [18] Yoonwoo Jeong, Seokjun Ahn, Christopher Choy, Anima Anandkumar, Minsu Cho, and Jaesik Park. Self-calibrating neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5846– 5854, 2021. 3
- [19] Kaiwen Jiang, Yang Fu, Yash Belhe, Xiaolong Wang, Hao Su, Ravi Ramamoorthi, et al. A construct-optimize approach to sparse view synthesis without camera pose. arXiv preprint arXiv:2405.03659, 2024. 3
- [20] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. Lvsm: A large view synthesis model with minimal 3d inductive bias. arXiv preprint arXiv:2410.17242, 2024. 3
- [21] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (ToG), 42(4):1–14, 2023. 1, 2, 3, 6, 7, 8, A9
- [22] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36

(4):1–13, 2017. 2, 6, A10

- [23] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding image matching in 3d with mast3r. arXiv preprint arXiv:2406.09756, 2024. 1, 2, 3, A9
- [24] Hao Li, Yuanyuan Gao, Chenming Wu, Dingwen Zhang, Yalun Dai, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, and Junwei Han. Ggrt: Towards pose-free generalizable 3d gaussian splatting in real-time. CoRR, 2024. 3
- [25] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey. Barf: Bundle-adjusting neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5741–5751, 2021. 3
- [26] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu,

- et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024. 6, A10
- [27] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654–20664, 2024. 3
- [28] Andreas Meuleman, Yu-Lun Liu, Chen Gao, Jia-Bin Huang, Changil Kim, Min H Kim, and Johannes Kopf. Progressively optimized local radiance fields for robust view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16539–16548, 2023. 3
- [29] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (TOG), 38(4):1–14, 2019. 2
- [30] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing Scenes As Neural Radiance Fields for View Synthesis. Communications of the ACM, 65(1):99–106,

2021. 2

- [31] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. arXiv preprint arXiv:2112.00724, 2021. 2, 3
- [32] Eric Penner and Li Zhang. Soft 3D Reconstruction for View Synthesis. ACM Transactions on Graphics (TOG), 36(6):1– 11, 2017. 2
- [33] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016. 2, 3, A10
- [34] Philipp Schr¨oppel, Jan Bechtold, Artemij Amiranashvili, and Thomas Brox. A benchmark and a baseline for robust multiview depth estimation. In 2022 International Conference on 3D Vision (3DV), pages 637–645. IEEE, 2022. A10
- [35] Steven M Seitz and Charles R Dyer. Photorealistic Scene Reconstruction by Voxel Coloring, 2002. US Patent 6,363,170. 2
- [36] Pratul P Srinivasan, Richard Tucker, Jonathan T Barron, Ravi Ramamoorthi, Ren Ng, and Noah Snavely. Pushing the Boundaries of View Extrapolation With Multiplane Images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 175–184, 2019.
- [37] Pratul P Srinivasan, Ben Mildenhall, Matthew Tancik, Jonathan T Barron, Richard Tucker, and Noah Snavely. Lighthouse: Predicting Lighting Volumes for SpatiallyCoherent Illumination. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8080–8089, 2020. 2
- [38] Prune Truong, Marie-Julie Rakotosaona, Fabian Manhardt, and Federico Tombari. Sparf: Neural radiance fields from sparse and noisy poses. In Proceedings of the IEEE/CVF

- Conference on Computer Vision and Pattern Recognition, pages 4190–4200, 2023. 3
- [39] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. Sparsenerf: Distilling depth ranking for fewshot novel view synthesis. arXiv preprint arXiv:2303.16196,

2023. 2, 3

- [40] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. arXiv preprint arXiv:2312.14132, 2023. 6, A10
- [41] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6
- [42] Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Victor Adrian Prisacariu. Nerf–: Neural radiance fields without known camera parameters. arXiv preprint arXiv:2102.07064, 2021. 2, 3, 5, 6, 7, 8, A10
- [43] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor Verbin, Jonathan T Barron, Ben Poole, et al. Reconfusion: 3d reconstruction with diffusion priors. arXiv preprint arXiv:2312.02981, 2023. 3
- [44] Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay, Pradyumna Chari, and Achuta Kadambi. Sparsegs: Realtime 360 {\deg} sparse view synthesis using gaussian splatting. arXiv preprint arXiv:2312.00206, 2023. 3
- [45] Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Humphrey Shi, and Zhangyang Wang. Sinnerf: Training neural radiance fields on complex scenes from a single image. In European Conference on Computer Vision, pages 736–753. Springer,

2022. 3

- [46] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. arXiv preprint arXiv:2410.13862, 2024. 3
- [47] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Pointnerf: Point-based neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5438–5448, 2022. 2
- [48] Jiawei Yang, Marco Pavone, and Yue Wang. Freenerf: Improving few-shot neural rendering with free frequency regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8254– 8263, 2023. 2, 3
- [49] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, and Songyou Peng. No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv preprint arXiv:2410.24207, 2024. 3
- [50] Lin Yen-Chen, Pete Florence, Jonathan T Barron, Alberto Rodriguez, Phillip Isola, and Tsung-Yi Lin. inerf: Inverting neural radiance fields for pose estimation. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1323–1330. IEEE, 2021. 3
- [51] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images.

- In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4578–4587, 2021. 3
- [52] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Chenming Zhu, Zhangyang Xiong, Tianyou Liang, et al. Mvimgnet: A large-scale dataset of multi-view images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9150–9161, 2023. 2, 6, A10
- [53] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19447–19456,

- 2024. 2, 3

[54] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Efficient and compact surface reconstruction in unbounded scenes. arXiv preprint arXiv:2404.10772,

- 2024. 3

- [55] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, and Ping Tan. Rade-gs: Rasterizing depth in gaussian splatting. arXiv preprint arXiv:2406.01467, 2024. 3
- [56] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6
- [57] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. arXiv preprint arXiv:2312.00451, 2023. 2, 3, 6, 7
- [58] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Longsequence large reconstruction model for wide-coverage gaussian splats. arXiv preprint arXiv:2410.12781, 2024. 3
- [59] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. Ewa volume splatting. In Proceedings Visualization, 2001. VIS’01., pages 29–538. IEEE, 2001. 3

