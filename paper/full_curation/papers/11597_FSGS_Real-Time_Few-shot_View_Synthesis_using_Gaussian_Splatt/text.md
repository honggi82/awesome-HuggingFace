# arXiv:2312.00451v2[cs.CV]16Jun2024

## FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting

##### Zehao Zhu1†, Zhiwen Fan1†, Yifan Jiang1, Zhangyang Wang1

†Equal Contribution, The University of Texas at Austin

Abstract. Novel view synthesis from limited observations remains a crucial and ongoing challenge. In the realm of NeRF-based few-shot view synthesis, there is often a trade-off between the accuracy of the synthesized view and the efficiency of the 3D representation. To tackle this dilemma, we introduce a Few-Shot view synthesis framework based on 3D Gaussian Splatting, which facilitates real-time, photo-realistic synthesis from a minimal number of training views. FSGS employs an innovative Proximity-guided Gaussian Unpooling, specifically designed for sparse-view settings, to bridge the gap presented by the extremely sparse initial point sets. This method involves the strategic placement of new Gaussians between existing ones, guided by a Gaussian proximity score, enhancing the adaptive density control. We have identified that Gaussian optimization can sometimes result in overly smooth textures and a propensity for overfitting when training views are limited. To mitigate these issues, FSGS introduces the synthesis of virtual views to replicate the parallax effect experienced during training, coupled with geometric regularization applied across both actual training and synthesized viewpoints. This strategy ensures that new Gaussians are placed in the most representative locations, fostering more accurate and detailed scene reconstruction. Our comprehensive evaluation across various datasets—including NeRF-Synthetic, LLFF, Shiny, and Mip-NeRF360 datasets—illustrates that FSGS not only delivers exceptional rendering quality but also achieves an inference speed more than 2000 times faster than existing state-of-the-art methods for sparse-view synthesis. Project webpage https://zehaozhu.github.io/FSGS/.

Keywords: Neural Rendering · Gaussian Splatting · Sparse View Synthesis

### 1 Introduction

Novel view synthesis (NVS) from a set of view collections, as demonstrated by recent works [16,35,46], has played a critical role in the domain of 3D vision and is pivotal in many applications, e.g., VR/AR and autonomous driving. Despite its effectiveness in photo-realistic rendering, the requirement of dense support views has hindered its practical usages [33]. Previous studies have focused on reducing the view requirements by leveraging Neural Radiance Field (NeRF) [30], a powerful implicit 3D representation that captures scene details, combined with

[Figure 1]

[Figure 2]

[Figure 3]

…

Sparse input views

Few SfM points

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

3D Gaussians

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

203FPS

FPS vs. SSIM

- Fig. 1: Real-Time Few-shot Novel View Synthesis. We present a point-based framework that is initialized from extremely sparse SfM points, achieving a significantly faster rendering speed (2900×) while enhancing the visual quality (from 0.684 to 0.745, in SSIM) compared to the previous SparseNeRF [50].

volume rendering techniques [13]. Depth regularization [11,33,47,56] within the density field, additional supervision from 2D pre-trained models [23,50], largescale pre-training [8,61], and frequency annealings [57] have been proposed and adopted to address the challenge of few-shot view synthesis. While these NeRFbased approaches are promising, they often lead to substantial computational demands, which can affect real-time performance adversely. Subsequent research has managed to reduce training time in real-world scenarios from days to mere hours [40,43,51,60], and even minutes in some cases [31]. However, a noticeable gap persists between attaining real-time rendering speeds and the desired photorealistic, high-resolution output quality.

In our research, we delve into the advancements in efficient 3D Gaussian Splatting (3D-GS) [26] and examine the challenges associated with deploying 3D-GS on sparse inputs. A crucial aspect for the effectiveness of 3D-GS is the densification process, which transforms the sparse initial point cloud into a more detailed representation of the 3D environment. However, the placement of new Gaussians, dictated by the spatial gradient, tends to be noisy and unrepresentative, especially in sparse-view scenarios. Additionally, the reliance on photometric loss with limited view counts often results in overly smooth textures when adhering to the conventional densification approach.

To address these issues, we introduce Proximity-guided Gaussian Unpooling, a novel strategy designed specifically for sparse inputs. This method enhances

the Gaussian representation by inserting new Gaussians between existing ones, based on the proximity to their neighbors. This strategic placement, combined with the initialization using observations from existing Gaussians, significantly improves scene representation by increasing the Gaussian density. Furthermore, we advocate for view augmentation through the generation of virtual camera, not present during training, to apply additional constraints in sparse setups. Incorporating monocular depth priors helps regularize the new Gaussians, steering them towards a plausible solution while enhancing texture detail, essential for accurate relative positioning in both actual training and synthetic camera views. We have conducted thorough evaluations of our Few-Shot Gaussian Splatting (FSGS) framework across a variety of few-shot Novel View Synthesis benchmarks. These include the object-centric NeRF-Synthetic datasets, the forward-facing LLFF datasets, the Shiny datasets with intricate lighting conditions, and the unbounded Mip-NeRF360 datasets. Our experiments demonstrate that FSGS sets a new benchmark in rendering quality and operates at a realtime speed (203 FPS), making it suitable for real-world applications. The efficacy of our method allows FSGS to outperform 3D-GS even with fewer Gaussians, enhancing both the efficiency and quality on the rendered scenes.

Our key contributions are as follows:

- – We propose a novel point-based framework, FSGS, for few-shot view synthesis that densifies new Gaussians via Proximity-guided Gaussian Unpooling. This method effectively increases the density of Gaussians, ensuring detailed and comprehensive scene representation.
- – FSGS addresses the overfitting challenge inherent in sparse-view Gaussian splatting. It achieves this by generating unseen viewpoints during training and incorporating distance correspondences on both training and synthesized pseudo views. This strategy directs the Gaussian optimization process toward solutions that are both highly accurate and visually compelling.
- – FSGS significantly enhances the visual quality, and also facilitates real-time rendering speeds (over 200 FPS) leading to a viable option for practical implementation in various real-world applications.

### 2 Related Works

#### 2.1 Neural Representations for 3D Reconstruction

The recent advancement of neural rendering techniques, such as Neural Radiance Fields (NeRFs) [30], has shown encouraging progress for novel view synthesis. NeRF learns an implicit neural scene representation that utilizes a MLP to map 3D coordinates (x,y,z) and view dependency (θ,ϕ) to color and density through a volume rendering function. Tremendous works focus on improving its efficiency [7,15,17,26,31,38,43], quality [1,2,4,9,20,42,48,52], generalizing to unseen scenes [8,10,24,44,53,61], applying artistic effects [14,22,49,62] and 3D generation [5, 6, 18, 21, 25, 27, 28, 34, 41, 45]. In particular, Reiser et al. [38] accelerate NeRF’s training by splitting a big MLP into thousands of tiny MLPs.

MVSNeRF [8] constructs a 3D cost volume [19, 59] and renders high-quality images from novel viewpoints. Moreover, Mip-NeRF [1] adopts conical frustum rather than a single ray in order to mitigate aliasing. Mip-NeRF 360 [3] further extends it to the unbounded scenes. While these NeRF-like models present strong performance on various benchmarks, they generally require several hours of training time. Muller et al. [31] adopt a multiresolution hash encoding technique that reduces the training time significantly. Kerbl et al. [26] propose to use a 3D Gaussian Splatting pipeline that achieves real-time rendering for either objects or unbounded scenes. The proposed FSGS approach is based on the 3D Gaussian Splatting framework but largely reduces the required training views.

#### 2.2 Novel-View Synthesis Using Sparse Views

The original neural radiance field takes more than one hundred images as input, largely prohibiting its practical usage. To tackle this issue, several works have attempted to reduce the number of training views. Specifically, DepthNeRF [12] applies additional depth supervision to improve the rendering quality. RegNeRF [32] proposes a depth smoothness loss as geometry regularization to stabilize training. DietNeRF [23] adds supervision on the CLIP embedding space [36], to constraint the rendered unseen views. PixelNeRF [61] trains a convolution encoder to capture context information and learns to predict 3D representation from sparse inputs. More recently, FreeNeRF [57] proposes a dynamic frequency controlling module for few-shot NeRF. SparseNeRF [50] proposes a new spatial continuity loss to distill spatial coherence from monocular depth estimators. Concurrent work ReconFusion [55] employs diffusion models to synthesize additional views, which may not always adhere to view consistency and time consuming. ReconFusion jointly train a Zip-NeRF with synthesized views under a sparse-view setting. In contrast, our method improves the optimization process of Gaussian Splatting, and facilitates both the real-time rendering speed and rendering quality.

### 3 Method

Overview. This section provides an overview of the FSGS framework, as illustrated in Fig. 2. FSGS processes a limited set of images captured from a static scene. The camera poses and point clouds are derived using the Structure-fromMotion (SfM) software, COLMAP [39]. The initialization of 3D Gaussians is based on a sparse point cloud, incorporating attributes such as color, position, and a predefined conversion rule for shape and opacity. The issue of extremely sparse points is tackled through the implementation of Proximity-guided Gaussian Unpooling. This method densifies Gaussians and populates the empty spaces by assessing the proximity between existing Gaussians and positioning new ones in the most representative areas, thereby enhancing scene details. To mitigate overfitting in standard 3D-GS with sparse-view data, we introduce the generation of pseudo camera viewpoints around the training cameras. This approach,

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Training view Synthesized view

[Figure 25]

[Figure 26]

Loss

[Figure 27]

RGB

[Figure 28]

GT

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Projection + Rasterization

Gaussian Graph

Existing Gaussian

Unpooled Gaussian

Proximity-guided Gaussian Unpooling

Depth (Pred)

Depth(DPT)

Real-time Rendering

[Figure 37]

SfM

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

,

[Figure 46]

[Figure 47]

L1 + SSIM losses Depth correlation

[Figure 48]

Rendered Paths

Loss

Sparse input

Initialization

Optimized Gaussians

- Fig. 2: FSGS Pipeline. 3D Gaussians are initialized from COLMAP, with a few images (black cameras). For the sparsely placed Gaussians, we propose densifying new Gaussians to enhance scene coverage by unpooling existing Gaussians into new ones, with properly initialized Gaussian attributes. Monocular depth priors, enhanced by sampling unobserved views (red cameras), guide the optimization of grown Gaussians towards a reasonable geometry. The final loss consists of a photometric loss term, and a geometric regularization term calculated as depth relative correspondence.

coupled with the geometry regularization, steers the model towards accurately reconstructing the scene’s geometry.

#### 3.1 Preliminary and Problem Formulation

- 3D Gaussian Splatting (3D-GS), as delineated in Kerbl et al. [26], represents an 3D scene explicitly through a collection of 3D Gaussians, with attributes: a position vector µ ∈ R3 and a covariance matrix Σ ∈ R3×3. Each Gaussian influences a point x in 3D space following the 3D Gaussian distribution:

1 (2π)3/2 |Σ|1/2

- 1

- 2(x−µ)TΣ−1(x−µ) (1)

e−

G(x) =

To ensure that Σ is positive semi-definite and holds practical physical significance, Σ is decomposed into two learnable components by Σ = RSSTRT, where R is a quaternion matrix representing rotation and S is a scaling matrix.

Each Gaussians also store an opacity logit o ∈ R and the appearance feature represented by n spherical harmonic (SH) coefficients ci ∈ R3|i = 1,2,...,n where n = D2 is the number of coefficients of SH with degree D. To render the 2D image, 3D-GS orders all the Gaussians that contributes to a pixel and blends the ordered Gaussians overlapping the pixels using the following function:

- i−1
- j=1

n

(1 − αj) (2)

c =

ciαi

i=1

where ci is the color computed from the SH coefficients of the ith Gaussian. αi is given by evaluating a 2D Gaussian with covariance Σ′ ∈ R2×2 multiplied by the opacity. The 2D covariance matrix Σ′ is calculated by Σ′ = JWΣWTJT, projecting the 3D covariance Σ to the camera coordinates. Here, J denotes the Jacobian of the affine approximation of the projective transformation, W

[Figure 49]

| | |[Figure 50]<br><br>| | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

[Figure 51]

Full-view SfM points

Three-view SfM points

- Fig. 3: Points Sparsity vs. Synthesized Quality. The SfM points from COLMAP using 3-views (Bottom Left) is significantly sparse than full-view(Top Left). 3D-GS with sparse SfM points will decrease its quality when the training view number decreases.

is the view transformation matrix. A heuristic Gaussian densification scheme is introduced in 3D-GS [26], where Gaussians are densified based on an average magnitude of view-space position gradients which exceed a threshold. Although this method is effective when initialized with comprehensive SfM points, it is insufficient for fully covering the entire scene with an extremely sparse point cloud, from sparse-view input images. Additionally, some Gaussians tend to grow towards extremely large volumes, leading to results that overfit the training views and generalize badly to novel viewpoints (See Fig. 3).

However, 3D-GS is initialized from SfM points, and its performance strongly relies on both the quantity and accuracy of the initialized points. Although the subsequent Gaussian densification [26] can increase the number of Gaussians in both under-reconstructed and over-reconstructed regions, this straightforward strategy falls short in few-shot settings: it suffers from inadequate initialization, leading to oversmoothed outcomes and a tendency to overfit on training views.

- 3.2 Proximity-guided Gaussian Unpooling The granularity of the modeled scene depends heavily on the quality of the
- 3D Gaussians representing the scene; therefore, addressing the limited 3D scene coverage is crucial for effective sparse-view modeling.

Proximity Score and Graph Construction. During Gaussian optimization, we construct a directed graph, referred to as the proximity graph, to connect each existing Gaussian with its nearest K neighbors by computing the proximity (a.k.a. Euclidean distance). Specifically, we denote the originating Gaussian at the head as the “source” Gaussian, while the one at the tail as the “destination” Gaussian, which is one of the source’s K neighbors. These “destination” Gaussians are determined via the rule:

DiK = K-min(dij), ∀j ̸= i (3) Here, dij is calculated via dij = ∥µi−µj∥, representing the Euclidean distance

among the centers of Gaussian Gi and Gaussian Gj. The assigned proximity

score Pi to Gaussian Gi is calculated as the average distance to its K nearest neighbors:

K

1 K

DiK (4)

Pi =

j=1

The proximity graph is updated following the densification or pruning process during optimization. We set K to 3 in practice.

Gaussian Unpooling. Inspired by the vertex-adding strategy of the mesh subdivision algorithm [63] which is widely used in computer graphics, we propose unpooling Gaussians based on the proximity graph and the proximity score of each Gaussian. Specifically, if the proximity score of a Gaussian exceeds the threshold tprox, our method will grow a new Gaussian at the center of each edge, connecting the “source” and “destination” Gaussians, as shown in Fig. 4. The attributes of scale and opacity in the newly created Gaussians are set to match those of the “destination” Gaussians. Meanwhile, other attributes such as rotation and SH coefficients are initialized to zero. The Gaussian unpooling strategy encourages the newly densified Gaussians to be distributed around the representative locations and progressively fill observation gaps during optimization.

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

Gaussian Unpool.

3D-GS Gaussian Unpool.

+ Depth Guidance

New Gaussians

GT Shape Existing Gaussians

Viewing Camera

[Figure 62]

[Figure 63]

Positional Gradient

Edge with Proximity Score

Monocular Depth Guidance

1D Depth Example

Fig. 4: Gaussian Unpooling Illustration. We show a 2D toy case for visualizing Gaussian Unpooling with depth guidance, where the example 1D depth provides priors on the relative distance of the Gaussians from the viewing direction, guide the Gaussian deformation toward a better solution.

#### 3.3 Geometry Guidance for Gaussian Optimization

Having achieved dense coverage by unpooling Gaussians, a photometric loss with sparse-view clues is applied for optimizing Gaussians. However, the insufficient parallax in the sparse-view setting limit the 3D Gaussians to be optimized toward a globally consistent direction where it tend to overfit on training views, and poor generalization to novel views. To inject more regularization to the optimization, we propose to create some virtual cameras that unseen in training, and apply the pixel wise geometric correspondences as additional regularization.

Synthesize Pseudo Views. To address the inherent issue of overfitting to sparse training views, we employ unobserved (pseudo) view augmentation to incorporate more prior knowledge within the scene derived from a 2D prior model. The synthesized view is sampled from the two closest training views in Euclidean space, calculating the averaged camera orientation and interpolating a virtual one between them. A random noise is applied to the 3 degrees-of-freedom (3DoF) camera location as shown in Eq. 5, and then images are rendered.

P′ = (t + ε,q), ε ∼ N(0,δ) (5)

Here, t ∈ P denotes camera location, while q is a quaternion representing the rotation averaged from the two cameras. This approach of synthesizing online pseudo-views enables dynamic geometry updates, as the 3D Gaussians will update progressively, reducing the risk of overfitting.

Inject Geometry Coherence from Monocular Depth. We generate the monocular Dest depth maps at both training and pseudo views by using the pre-trained Dense Prediction Transformer (DPT) [37], trained with 1.4 million image-depth pairs as a handy yet effective choice. To mitigate the scale ambiguity between the true scene scale and the estimated depth, we introduce a relaxed relative loss, Pearson correlation, on the estimated and rendered depth maps. It measures the distribution difference between 2D depth maps and follows the below function:

Cov(Dˆ ras,Dˆ est) Var(Dˆ ras)Var(Dˆ est)

Corr(Dˆ ras,Dˆ est) =

(6)

This soften constraint allows for the alignment of depth structure without being hindered by the inconsistencies in absolute depth values.

Differentiable Depth Rasterization. To enable the backpropogation from depth prior to guide Gaussian training, we implement a differentiable depth rasterizor, allowing for receiving the error signal between the rendered depth Dras and the estimated depth Dest. Specifically, we utilize the alpha-blending rendering in 3D-GS for depth rasterization, where the z-buffer from the ordered Gaussians contributing to a pixel is accumulated for producing the depth value:

- i−1
- j=1

n

(1 − αj) (7)

d =

diαi

i=1

Here di represents the z-buffer of the ith Gaussians and α is identical to that in Eq. 2. This differentiable implementation enables the depth correlation loss.

#### 3.4 Optimization

Combining all together, we can summarize the training loss:

+λ2 D-SSIM(C,Cˆ )

L(G,C) = λ1 C − Cˆ

(8)

+λ3 ∥Corr(Dras,Dest)∥1

1 L1

Lssim

Lregularization

where L1, and Lssim stands for the photometric loss term between predicted image Cˆ and ground-truth image C. Lregularization represents the geometric regularization term on both the training views and synthesized pseudo views. We set λ1, λ2, λ3 as 0.8, 0.2, 0.05 respectively by grid search. The pseudo views sampling is enabled after 2,000 iterations to ensure the Gaussians can roughly represent the scene.

### 4 Experiments

#### 4.1 Experimental Settings

LLFF Datasets [29] consist of eight forward-facing real-world scenes. Following RegNeRF [33], we select every eighth image as the test set, and evenly sample sparse views from the remaining images for training. We utilize 3 views to train all the methods, and downsample their resolutions to 4× and 8×, which are 504 × 378 and 1008 × 756 respectively.

Mip-NeRF360 Datasets [3] consist of nine scenes, each featuring a complex central object or area against a detailed background. We utilize 24 training views for comparison, with images downsampled to 4× and 8×. Test images are selected the same with LLFF Datasets. We aim to establish this challenge benchmark for testing few-shot view synthesis in complex outdoor scenarios.

NeRF-Synthetic Datasets (Blender) [30] have eight objects with realistic images synthesized by Blender. We align with DietNeRF [23], where we use 8 images for training and 25 for testing, at resolution of 400×400.

Shiny Datasets [54] contain more challenging view-dependent effects, like the rainbow reflections on a CD and refraction through a liquid bottle. We evenly select 3 views from the Shiny datasets for training at resolutions of 504×378.

Baselines. We compare FSGS with several few-shot NVS methods on these three dataset, including DietNeRF [23], RegNeRF [33], FreeNeRF [57], and SparseNeRF [50]. Additionally, we include comparisons with the high-performing Mip-NeRF [2], primarily designed for dense-view training, and point-based 3DGS, following its original dense-view training recipe. Following [26,33,50], we report the average PSNR, SSIM, LPIPS scores and FPS for all the methods.

Implementation Details. We implemented FSGS using the PyTorch framework, with initial point cloud computed from SfM, using only the training views. During optimization, we densify the Gaussians every 100 iterations and start densification after 500 iterations. The total optimization steps are set to 10,000, requiring approximately 9.5 minutes on LLFF datasets, and ∼24 minutes on MipNeRF360 datasets. We set proximity threshold tprox to 10, and the pseudo views are sampled after 2,000 iterations. We utilize the pre-trained DPT model [37] for depth estimation. All results are obtained using a NVIDIA A6000 GPU.

#### 4.2 Comparisons to other Few-shot Methods

Comparisons on LLFF Datasets. As shown in Tab. 1, our method FSGS, despite trained from sparse SfM point clouds, provides the best quantitative results and effectively addresses the insufficient scene coverage in the initialization. Our method surpasses SparseNeRF by 0.45dB and 0.81dB in PSNR at both test resolutions, while inferencing 2180 times faster, which makes FSGS a viable choice for practical usages. FSGS also outperforms 3D-GS by 2.88dB in PSNR and boost the FPS from 385 to 458, demonstrating that our refined Gaussians are more compact for scene representation from sparse views.

- Table 1: Quantitative Comparison in LLFF Datasets, with 3 Training Views. FSGS achieves the best performance in terms of rendering accuracy and inference speed across all resolutions. Significantly, FSGS runs 2,180× faster than the previous best, SparseNeRF, while improving the SSIM from 0.624 to 0.652, at the resolution of 504 × 378. We color each cell as best , second best , and third best .

|Methods<br><br>|1/8 Resolution<br><br>|1/4 Resolution|
|---|---|---|
| |FPS↑ PSNR↑ SSIM↑ LPIPS↓|FPS↑ PSNR↑ SSIM↑ LPIPS↓<br><br>|
|Mip-NeRF 3D-GS<br><br>|0.21 16.11 0.401 0.460 385 17.43 0.522 0.321<br><br>|0.14 15.22 0.351 0.540 312 16.94 0.488 0.402<br><br>|
|DietNeRF RegNeRF FreeNeRF SparseNeRF Ours<br><br>|0.14 14.94 0.370 0.496 0.21 19.08 0.587 0.336 0.21 19.63 0.612 0.308 0.21 19.86 0.624 0.328 458 20.31 0.652 0.288<br><br>|0.08 13.86 0.305 0.578 0.14 18.06 0.535 0.411<br><br>0.14 18.73 0.562 0.384<br><br>0.14 19.07 0.564 0.401 351 19.88 0.612 0.340<br><br><br>|

The qualitative analysis, as presented in Fig. 5, demonstrates that Mip-NeRF and 3D-GS struggle with the extreme sparse view problem; Mip-NeRF [1] leads to degraded geometric modeling, and 3D-GS produces blurred results in areas with complex geometry. The geometry regularization in RegNeRF [32], SparseNeRF [50] and frequency annealing in FreeNeRF [57] do improve the quality to some extent, but still exhibit insufficient visual quality. In contrast, our proposed Proximity-Guided Gaussian Unpooling, and the relative geometric regularizations on both training and synthesized virtual views, pulls more Gaussians to the unobserved regions, and thus recovers more textural and structural details.

GT 3D-GS Mip-NeRF360 SparseNeRF Ours

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

FlowerLeavesFern

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

- Fig. 5: Qualitative Results on LLFF Datasets. We demonstrate novel view results produced by 3D-GS [26], Mip-NeRF360 [3], SparseNeRF [50] and our approach for comparison. We can observe that NeRF-based methods generate floaters (Scene: Flower) and show aliasing results (Scene: Leaves) due to limited observation. 3D-GS produces oversmoothed results, caused by overfitting on training views. Our method produces pleasing appearances while demonstrating detailed thin structures.

- Table 2: Quantitative Comparison in Mip-NeRF360 Datasets, with 24 Training Views. Our FSGS shows obvious advantages over NeRF-based methods, with an improvement of more than 0.05 in SSIM and running 4,142× faster. Additionally, our method not only performs better than 3D-GS in rendering metrics but also shows improvement in FPS (from 223 to 290), thanks to the Gaussian unpooling which motivates Gaussians to expand to unseen regions more accurately.

|Methods<br><br>|1/8 Resolution|1/4 Resolution|
|---|---|---|
| |FPS↑ PSNR↑ SSIM↑ LPIPS↓<br><br>|FPS↑ PSNR↑ SSIM↑ LPIPS↓|
|Mip-NeRF360 3D-GS|0.12 21.23 0.613 0.351 223 20.89 0.633 0.317<br><br>|0.07 19.78 0.530 0.431 145 19.93 0.588 0.401<br><br>|
|DietNeRF RegNeRF FreeNeRF SparseNeRF Ours<br><br>|0.05 20.21 0.557 0.387 0.07 22.19 0.643 0.335 0.07 22.78 0.689 0.323 0.07 22.85 0.693 0.315 290 23.70 0.745 0.220<br><br>|0.03 19.11 0.482 0.452<br>0.04 20.55 0.546 0.398 0.04 21.04 0.587 0.377 0.04 21.13 0.600 0.389 203 22.82 0.693 0.293<br><br><br>|

It is worth to note that, the rendering speed of the 3D Gaussian representation is contingent upon the number of optimized Gaussians, where a reduced Gaussian count leads to faster rendering speed. FSGS outperforms 3D-GS while using less Gaussian counts, by effectively places and optimizes the Gaussians to the most representative positions than 3D-GS. With the same initialization method as 3D-GS, the average optimized Gaussian count from FSGS is 57,513 on LLFF dataset, considerably lower than 63,219 from 3D-GS, which results in a faster rendering speed of FSGS than 3D-GS.

GT 3D-GS Mip-NeRF360 SparseNeRF Ours

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

GardenRoomStumpBonsai

|[Figure 99]| |
|---|---|
| | |

|[Figure 100]| |
|---|---|
| | |

|[Figure 101]| |
|---|---|
| | |

|[Figure 102]| |
|---|---|
| | |

|[Figure 103]|
|---|

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

|[Figure 109]| |
|---|---|
| |[Figure 110]|

|[Figure 111]| |
|---|---|
| |[Figure 112]|

|[Figure 113]| |
|---|---|
| |[Figure 114]|

|[Figure 115]| |
|---|---|
| |[Figure 116]|

|[Figure 117]| |
|---|---|
| |[Figure 118]|

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

|[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

###### Fig. 6: Qualitative Results on Mip-NeRF360 Datasets. Comparisons were conducted with 3D-GS [26], Mip-NeRF360 [3], and SparseNeRF [50]. Our method continues to produce visually pleasing results with sharper details than other methods in large-scale scenes.

- Table 3: Quantitative Comparison in Blender and Shiny Datasets. FSGS outperforms existing few-shot methods and 3D-GS across all metrics, validating the generalization of the proposed techniques to handheld object-level 3D modeling and datasets with challenging reflective effects.

|Methods|Blender Dataset [30]<br><br>|Shiny Dataset [54]|
|---|---|---|
| |FPS↑ PSNR↑ SSIM↑ LPIPS↓|FPS↑ PSNR↑ SSIM↑ LPIPS↓|
|Mip-NeRF 3D-GS|0.22 20.89 0.830 0.168 332 21.56 0.847 0.130<br><br>|0.19 17.37 0.525 0.432 316 17.83 0.547 0.385<br><br>|
|DietNeRF RegNeRF FreeNeRF SparseNeRF Ours|0.14 22.50 0.823 0.124<br><br>0.22 23.86 0.852 0.105<br><br>0.22 24.26 0.883 0.098 0.22 24.04 0.876 0.113 467 24.64 0.895 0.095<br><br><br>|0.16 17.67 0.546 0.403 0.19 18.10 0.574 0.378 0.19 18.65 0.586 0.360 0.19 18.81 0.591 0.354 341 19.63 0.612 0.327<br><br>|

Comparisons on Mip-NeRF360 Datasets. As shown in Tab. 2, methods requiring dense view coverage (Mip-NeRF360, 3D-GS) are outperformed by ours in terms of rendering speed and metrics, across the two resolutions. Methods employing regularizations from their respective geometry and appearance fields (DietNeRF, RegNeRF, FreeNeRF, SparseNeRF) still fall short in rendering quality, while remains far from achieving real-time speed. Our FSGS significantly outperforms NeRF-based approaches, boosting PNSR by 0.85dB and improving FPS from 0.07 to 290 at 1/8 resolution. We provide a qualitative comparison in Fig. 6, where we observe that Mip-NeRF360 and SparseNeRF fail to capture the intricate details of scenes and tend to overfit on sparse training views, most notably in areas far away from cameras. In comparison, FSGS recovers the finegrained details such as the leaves on the ground (Scene: Stump) and the piano keys (Scene: Bonsai), aligning well with the ground truth.

GT 3D-GS DietNeRF FreeNeRF Ours

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

HotdogFicus

|[Figure 139]|
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|[Figure 142]|
|---|

|[Figure 143]|
|---|

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

|[Figure 149]|
|---|

|[Figure 150]|
|---|

|[Figure 151]|
|---|

|[Figure 152]|
|---|

|[Figure 153]|
|---|

###### Fig. 7: Qualitative Results on Blender Datasets. Our method consistently outperforms other baselines in the task of novel view synthesis for object-centric datasets.

Comparisons on Blender Datasets. The left column of Tab. 3 presents the quantitative results on the Blender datasets. Here, our method significantly outperforms the baselines on object-level datasets, with an improvement of 0.40 in PSNR compared to FreeNeRF, although primarily designed for scene-level scenarios with complex geometry. Fig. 7 visualizes the rendered image. We find that DietNeRF hallucinates geometric details, and FreeNeRF exhibits noticeable aliasing effects. 3D-GS falls short into the excessive blurriness and distorts the edges of the objects. In contrast, our model not only captures the precise geometry of objects but also accurately simulates their shading effects.

Comparisons on Shiny Datasets. We report the quantitative results of the Shiny datasets on the right column of Tab. 3, which feature complex viewdependent effects. Our method performs better than other baselines and improves the PSNR by 1.80dB compared to 3D-GS. The superior performance validates the robustness of our method to handle unusual and challenging materials, such as CDs or glass.

[Figure 154]

[Figure 155]

[Figure 156]

RenderedRGB

| |
|---|

| |
|---|

[Figure 157]

[Figure 158]

[Figure 159]

RenderedDepth

| |
|---|

| |
|---|

Baseline (3D-GS)

+ Gaussian Unpool. + Pseudo-views + Depth Reg.

- Fig. 8: Ablation Study by Visualization. 3D-GS [26] (1st column) shows that the baseline method is significantly degraded when the view coverage is insufficient. Gaussian Unpooling provides extra capacity to 3D Gaussians to model the scene, but the learned geometry may not be accurate (2nd column). Adding Relative Depth Correspondence regularization (3rd column) can further improve the modeled details.

#### 4.3 Ablation Studies

We ablate our design choices on the LLFF dataset under the 3-view setting.

Effectiveness of Promity-guided Gaussian Unpooling. As shown in the second row of Tab. 4, our Promity-guided Gaussian Unpooling expands the scene geometry caused by limited training views, resulting in a PSNR improvement of 1.21dB compared to 3D-GS. We also visualize its visual effects in Fig. 8. The

- Table 4: Ablation Study on proposed components. Starting from 3D-GS [26] (1st row), we find that our proposed Gaussian Unpooling (2nd row) is more effective than the densification scheme in 3D-GS for few-shot view synthesis. Applying additional supervision from a monocular depth estimator further regularizes the Gaussian optimization towards a better solution (3rd row). Introducing pseudo-view augmentation to apply additional regularization when optimizing Gaussians further enhances the results in a few-shot scenario.

Gaussian Geometry Pseudo Unpooling Guidance Views PSNR↑ SSIM↑ LPIPS↓

✗ ✗ ✗ 17.43 0.522 0.321 ✓ ✗ ✗ 18.64 0.580 0.311 ✓ ✓ ✗ 19.83 0.634 0.297 ✓ ✓ ✓ 20.31 0.652 0.288

- Table 5: Ablation Study on different depth estimators. We utilize various monocular depth estimators on FSGS, and find that FSGS demonstrates strong robustness over different pretrained depth estimators.

|Method<br><br>|FPS↓ PSNR↑ SSIM↑ LPIPS↓|
|---|---|
|SparseNeRF 3D-GS FSGS (MiDaS small) FSGS (DPT Hybrid) FSGS (DPT Large) FSGS (DepthAnything)<br><br>|0.21 19.86 0.624 0.328 385 17.43 0.522 0.321 446 20.17 0.647 0.294 460 20.29 0.652 0.290 458 20.31 0.652 0.288 468 20.37 0.654 0.289|

heuristic Gaussian densification leads to blurring results, particularly noticeable in areas like bush and grass, our approach enriches structural and visual details.

Impact of Relative Depth Regularization. The third row of Tab. 4 demonstrates the improvement by introducing depth priors, guiding the Gaussian unpooling towards more plausible geometry. In Fig. 8, we observe that the depth regularization effectively eliminates the artifacts in grassy regions, and enforces more consistent and solid surfaces with geometric coherence. We also display the rendered depth map, where depth regularization leads to depths aligning better with the actual geometric structures.

Pseudo-view Matters in Few-shot Modeling. Tab. 4 validates the impact of synthesizing more unseen views during training, which anchors the Gaussians to a plausible geometry and further enhances the modeling quality when the geometry in densification is not accurate.

Robustness on different pretrained depth estimators. We have shown that FSGS can generalize across various datasets: Blender datasets (object-level), LLFF datasets (indoor), MipNeRF-360 datasets (indoor and outdoor), and selfcollected scenes using a mobile phone (Fig.B in the supplementary). To further substantiate its robustness, we employ different monocular depth estimators on LLFF dataset. As shown in Tab. 5, all of the depth estimators outperform the

baselines, and the Depth-Anything [58] method achieves the most comparable results. Collectively, FSGS consistently exhibits strong robustness across different pretrained depth estimators.

### 5 Conclusion and Limitation

In this work, we present a real-time few-shot framework, FSGS, for novel views synthesis within an insufficiently view overlapping. Starting from extremely sparse point clouds, FSGS adopts the point-based representation and proposes an effective Proximity-guided Gaussian Unpooling by measuring the proximity of each Gaussian to its neighbors. The overfitting issue in few-view 3D-GS can be alleviated by the adoption of pseudo-view generation and monocular relative depth correspondences to guide the expanded scene geometry toward a better solution. FSGS is capable of generating photo-realistic images with as few as three images, and perform inference at more than 200FPS, offering new avenues for real-time rendering and more cost-effective capture methods.

Although FSGS notably enhances the quality and efficiency of real-time fewshot neural rendering, it cannot generalize to occluded views that are unobserved during training. We hope that our proposed approach drives new research towards few-shot novel view synthesis in arbitrary 3D scenes.

### Acknowledgement

The work is supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/Interior Business Center (DOI/IBC) contract number 140D0423C0074. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government.

### References

- 1. Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srinivasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5855–5864 (2021)
- 2. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mipnerf 360: Unbounded anti-aliased neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5470– 5479 (2022)
- 3. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 5460–5469 (2022)

- 4. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Zip-nerf: Anti-aliased grid-based neural radiance fields. ICCV (2023)
- 5. Cao, Y., Cao, Y.P., Han, K., Shan, Y., Wong, K.Y.K.: Dreamavatar: Text-andshape guided 3d human avatar generation via diffusion models. arXiv preprint arXiv:2304.00916 (2023)
- 6. Chan, E., Monteiro, M., Kellnhofer, P., Wu, J., Wetzstein, G.: pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. In: arXiv

(2020)

- 7. Chen, A., Xu, Z., Geiger, A., Yu, J., Su, H.: Tensorf: Tensorial radiance fields. In: European Conference on Computer Vision. pp. 333–350. Springer (2022)
- 8. Chen, A., Xu, Z., Zhao, F., Zhang, X., Xiang, F., Yu, J., Su, H.: Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14124–14133

(2021)

- 9. Chen, T., Wang, P., Fan, Z., Wang, Z.: Aug-nerf: Training stronger neural radiance fields with triple-level physically-grounded augmentations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15191– 15202 (2022)
- 10. Chibane, J., Bansal, A., Lazova, V., Pons-Moll, G.: Stereo radiance fields (srf): Learning view synthesis from sparse views of novel scenes. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE (jun 2021)
- 11. Deng, C., Jiang, C., Qi, C.R., Yan, X., Zhou, Y., Guibas, L., Anguelov, D., et al.: Nerdi: Single-view nerf synthesis with language-guided diffusion as general image priors. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20637–20647 (2023)
- 12. Deng, K., Liu, A., Zhu, J.Y., Ramanan, D.: Depth-supervised nerf: Fewer views and faster training for free. arXiv preprint arXiv:2107.02791 (2021)
- 13. Drebin, R.A., Carpenter, L., Hanrahan, P.: Volume rendering. ACM Siggraph Computer Graphics 22(4), 65–74 (1988)
- 14. Fan, Z., Jiang, Y., Wang, P., Gong, X., Xu, D., Wang, Z.: Unified implicit neural stylization. In: European Conference on Computer Vision. pp. 636–654. Springer

(2022)

- 15. Fridovich-Keil, S., Yu, A., Tancik, M., Chen, Q., Recht, B., Kanazawa, A.: Plenoxels: Radiance fields without neural networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5501–5510 (2022)
- 16. Gao, K., Gao, Y., He, H., Lu, D., Xu, L., Li, J.: Nerf: Neural radiance field in 3d vision, a comprehensive review (2023)
- 17. Garbin, S.J., Kowalski, M., Johnson, M., Shotton, J., Valentin, J.: Fastnerf: Highfidelity neural rendering at 200fps. arXiv preprint arXiv:2103.10380 (2021)
- 18. Gu, J., Liu, L., Wang, P., Theobalt, C.: Stylenerf: A style-based 3d aware generator for high-resolution image synthesis. In: International Conference on Learning Representations (2022)
- 19. Gu, X., Fan, Z., Zhu, S., Dai, Z., Tan, F., Tan, P.: Cascade cost volume for high-resolution multi-view stereo and stereo matching. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2495– 2504 (2020)
- 20. Guo, Y.C., Kang, D., Bao, L., He, Y., Zhang, S.H.: Nerfren: Neural radiance fields with reflections. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18409–18418 (2022)

- 21. Höllein, L., Cao, A., Owens, A., Johnson, J., Nießner, M.: Text2room: Extracting textured 3d meshes from 2d text-to-image models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 7909–7920 (October 2023)
- 22. Jain, A., Mildenhall, B., Barron, J.T., Abbeel, P., Poole, B.: Zero-shot text-guided object generation with dream fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 867–876 (2022)
- 23. Jain, A., Tancik, M., Abbeel, P.: Putting nerf on a diet: Semantically consistent few-shot view synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5885–5894 (2021)
- 24. Johari, M.M., Lepoittevin, Y., Fleuret, F.: Geonerf: Generalizing nerf with geometry priors. Proceedings of the IEEE international conference on Computer Vision and Pattern Recognition (CVPR) (2022)
- 25. Karnewar, A., Vedaldi, A., Novotny, D., Mitra, N.: Holodiffusion: Training a 3D diffusion model using 2D images. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2023)
- 26. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (ToG) 42(4), 1–14 (2023)
- 27. Lin, C.H., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M.Y., Lin, T.Y.: Magic3d: High-resolution text-to-3d content creation. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- 28. Liu, R., Wu, R., Hoorick, B.V., Tokmakov, P., Zakharov, S., Vondrick, C.: Zero-1to-3: Zero-shot one image to 3d object (2023)
- 29. Mildenhall, B., Srinivasan, P.P., Ortiz-Cayon, R., Kalantari, N.K., Ramamoorthi, R., Ng, R., Kar, A.: Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (TOG) 38(4), 1–14

(2019)

- 30. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing Scenes As Neural Radiance Fields for View Synthesis. Communications of the ACM 65(1), 99–106 (2021). https://doi.org/10.1145/ 3503250
- 31. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG) 41(4), 1– 15 (2022)
- 32. Niemeyer, M., Barron, J.T., Mildenhall, B., Sajjadi, M.S., Geiger, A., Radwan, N.: Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. arXiv preprint arXiv:2112.00724 (2021)
- 33. Niemeyer, M., Barron, J.T., Mildenhall, B., Sajjadi, M.S., Geiger, A., Radwan, N.: Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5480–5490 (2022)
- 34. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022)
- 35. Rabby, A.S.A., Zhang, C.: Beyondpixels: A comprehensive review of the evolution of neural radiance fields (2023)
- 36. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning. pp. 8748–8763. PMLR (2021)

- 37. Ranftl, R., Bochkovskiy, A., Koltun, V.: Vision transformers for dense prediction. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 12179–12188 (2021)
- 38. Reiser, C., Peng, S., Liao, Y., Geiger, A.: Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14335–14345 (2021)
- 39. Schonberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4104–4113

(2016)

- 40. Schwarz, K., Sauer, A., Niemeyer, M., Liao, Y., Geiger, A.: Voxgraf: Fast 3dAware Image Synthesis With Sparse Voxel Grids. ArXiv Preprint ArXiv:2206.07695

(2022)

- 41. Seo, J., Jang, W., Kwak, M.S., Ko, J., Kim, H., Kim, J., Kim, J.H., Lee, J., Kim, S.: Let 2d diffusion model know 3d-consistency for robust text-to-3d generation. arXiv preprint arXiv:2303.07937 (2023)
- 42. Suhail, M., Esteves, C., Sigal, L., Makadia, A.: Light field neural rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8269–8279 (2022)
- 43. Sun, C., Sun, M., Chen, H.T.: Direct Voxel Grid Optimization: Super-Fast Convergence for Radiance Fields Reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5459–5469 (2022)
- 44. T, M.V., Wang, P., Chen, X., Chen, T., Venugopalan, S., Wang, Z.: Is attention all that neRF needs? In: The Eleventh International Conference on Learning Representations (2023), https://openreview.net/forum?id=xE-LtsE-xx
- 45. Tang, J., Wang, T., Zhang, B., Zhang, T., Yi, R., Ma, L., Chen, D.: Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior (2023)
- 46. Tewari, A., Thies, J., Mildenhall, B., Srinivasan, P., Tretschk, E., Wang, Y., Lassner, C., Sitzmann, V., Martin-Brualla, R., Lombardi, S., Simon, T., Theobalt, C., Niessner, M., Barron, J.T., Wetzstein, G., Zollhoefer, M., Golyanik, V.: Advances in neural rendering (2022)
- 47. Truong, P., Rakotosaona, M.J., Manhardt, F., Tombari, F.: Sparf: Neural radiance fields from sparse and noisy poses. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4190–4200 (2023)
- 48. Verbin, D., Hedman, P., Mildenhall, B., Zickler, T., Barron, J.T., Srinivasan, P.P.: Ref-nerf: Structured view-dependent appearance for neural radiance fields. arXiv preprint arXiv:2112.03907 (2021)
- 49. Wang, C., Chai, M., He, M., Chen, D., Liao, J.: Clip-nerf: Text-and-image driven manipulation of neural radiance fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3835–3844 (2022)
- 50. Wang, G., Chen, Z., Loy, C.C., Liu, Z.: Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. arXiv preprint arXiv:2303.16196 (2023)
- 51. Wang, L., Zhang, J., Liu, X., Zhao, F., Zhang, Y., Zhang, Y., Wu, M., Yu, J., Xu, L.: Fourier PlenOctrees for Dynamic Radiance Field Rendering in Real-Time. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13524–13534 (2022)
- 52. Wang, P., Liu, Y., Chen, Z., Liu, L., Liu, Z., Komura, T., Theobalt, C., Wang, W.: F2-nerf: Fast neural radiance field training with free camera trajectories. CVPR

(2023)

- 53. Wang, Q., Wang, Z., Genova, K., Srinivasan, P.P., Zhou, H., Barron, J.T., MartinBrualla, R., Snavely, N., Funkhouser, T.: Ibrnet: Learning multi-view image-based

- rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4690–4699 (2021)
- 54. Wizadwongsa, S., Phongthawee, P., Yenphraphai, J., Suwajanakorn, S.: Nex: Realtime view synthesis with neural basis expansion. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2021)
- 55. Wu, R., Mildenhall, B., Henzler, P., Park, K., Gao, R., Watson, D., Srinivasan, P.P., Verbin, D., Barron, J.T., Poole, B., et al.: Reconfusion: 3d reconstruction with diffusion priors. arXiv preprint arXiv:2312.02981 (2023)
- 56. Xu, D., Jiang, Y., Wang, P., Fan, Z., Shi, H., Wang, Z.: Sinnerf: Training neural radiance fields on complex scenes from a single image. In: European Conference on Computer Vision. pp. 736–753. Springer (2022)
- 57. Yang, J., Pavone, M., Wang, Y.: Freenerf: Improving few-shot neural rendering with free frequency regularization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8254–8263 (2023)
- 58. Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., Zhao, H.: Depth anything: Unleashing the power of large-scale unlabeled data. In: CVPR (2024)
- 59. Yao, Y., Luo, Z., Li, S., Fang, T., Quan, L.: Mvsnet: Depth inference for unstructured multi-view stereo. In: Proceedings of the European conference on computer vision (ECCV). pp. 767–783 (2018)
- 60. Yu, A., Fridovich-Keil, S., Tancik, M., Chen, Q., Recht, B., Kanazawa, A.: Plenoxels: Radiance Fields Without Neural Networks. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 5491–5500 (2022)
- 61. Yu, A., Ye, V., Tancik, M., Kanazawa, A.: pixelnerf: Neural radiance fields from one or few images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4578–4587 (2021)
- 62. Zhang, K., Kolkin, N., Bi, S., Luan, F., Xu, Z., Shechtman, E., Snavely, N.: Arf: Artistic radiance fields (2022)
- 63. Zorin, D., Schröder, P., Sweldens, W.: Interpolating subdivision for meshes with arbitrary topology. In: Proceedings of the 23rd annual conference on Computer graphics and interactive techniques. pp. 189–192 (1996)

### 6 More Technical Details

#### 6.1 Initialization

Similar to 3D Gaussian Splatting [26], we start our pipeline from unstructured multi-view images, and calibrate the images using Structure-from-Motion [39]. Next, we will continue the dense stereo matching under COLMAP with the function “patch_match_stereo ” and utilize the fused stereo point cloud from “stereo_fusion ”. We then initialize the SH coefficients at degree 0 and the positions of the 3D Gaussians based on the fused point cloud. Additionally, we remain the rest coefficients and rotation to 0. We also initialize the opacity to 0.1 and set the scale to match the average distance between points.

#### 6.2 Training

During training, we start with a SH degree of 0 for a basic lighting representation, incrementing by 1 every 500 iterations up to a degree of 4 to increase complexity over time. We set the learning rate of position, SH coefficients, opacity, scaling, and rotation to 0.00016, 0.0025, 0.05, 0.005, and 0.001 respectively. At iterations 2000, 5000, and 7000, the opacity for all Gaussians is reset to 0.05 to eliminate the low-opacity floaters. In the Blender dataset [30], the Pearson correlation is only computed in pixels where the depth values are greater than 0. Additionally, we utilize an open-source code1 to estimate the inverse depth map for both the input images and the rendered images from pseudo views. We detail the procedures of FSGS in Algorithm 1.

- Table 6: Quantitative Analysis on the Effects of Training Views. We conduct experiments by using different training views (from 3 to 9) to test the adopted baseline methods. Our FSGS consistently outperforms other methods across all metrics.

|Methods|PSNR|SSIM|LPIPS|
|---|---|---|---|
| |3-view 6-view 9-view<br><br>|3-view 6-view 9-view|3-view 6-view 9-view|
|Mip-NeRF 3D-GS<br><br>|16.10 22.91 24.88<br>17.43 22.87 24.65<br>|0.401 0.756 0.826 0.522 0.732 0.813<br><br>|0.460 0.213 0.160 0.321 0.204 0.169<br><br>|
|DietNeRF RegNeRF FreeNeRF SparseNeRF Ours<br><br>|14.94 21.75 24.28 19.08 23.10 24.86 19.63 23.73 25.13<br><br>19.86 23.64 24.97<br><br>20.31 24.55 25.89<br><br><br>|0.370 0.717 0.801 0.587 0.760 0.820 0.612 0.779 0.827 0.624 0.784 0.834 0.652 0.795 0.845<br><br>|0.496 0.248 0.183 0.336 0.206 0.161 0.308 0.195 0.160 0.328 0.202 0.158 0.288 0.177 0.143<br><br>|

1 https://pytorch.org/hub/intelisl_midas_v2/

Algorithm 1 The training pipeline of FSGS

- 1: Training view images I = {Ii ∈ RH×W×3}Ni=1 and their associated camera poses P = {ϕi ∈ R3×4}Ni=1.
- 2: Run SfM with the input images and camera poses and obtain an initial point cloud P, used to define 3D Gaussians function G = {Gi(µi, σi, ci, αi)}Ki=1.
- 3: Leverage pretrained depth estimator E to predict the depth map Di = E(Ii).
- 4: Synthesize pseudo views P† = {ϕ†i ∈ R3×4}Mi=1 from input camera poses P.
- 5: while until convergence do
- 6: Randomly sample an image Ii ∈ I and the corresponding camera pose ϕi
- 7: Rasterize the rgb image Iˆi and the depth map Dˆi with camera pose ϕi
- 8: L = λ1∥Ii − Iˆi∥1 + λ2 D-SSIM(Ii, Iˆi) + λ3 Pearson(Di, Dˆi)
- 9: if iteration > titer then
- 10: Sample a pseudo camera pose ϕ†j ∈ P†.
- 11: Rasterize the rgb image Iˆ†j and the depth Dˆ†i
- 12: Compute the estimated depth as D†j = E(Iˆ†j).
- 13: L = L + λ4 Pearson(D†j, Dˆ†i)
- 14: end if
- 15: if IsRefinement(iteration) then
- 16: for Gi(µi, σi, ci, αi) ∈ G do
- 17: if αi > ε or IsTooLarge(µi, σi) then
- 18: RemoveGaussian()
- 19: end if
- 20: if ∇pL > tpos then
- 21: GaussianDensify()
- 22: end if
- 23: if NoProximity(G) then
- 24: GaussianUnpooling()
- 25: end if
- 26: end for
- 27: end if
- 28: Update Gaussians parameter G via ∇GL.
- 29: end while

### 7 More Experiment Results

#### 7.1 Effects of Training Views

We demonstrate the quantitative results of FSGS on LLFF datasets under 3, 6,

- 9 views in Tab. 6. We can observe that training with more views often leads to better photo-realistic performance on sparse training data. More views provide a more comprehensive coverage of the scene, capturing more details of the scenes. This richness in data boosts supervision signals during optimization, leading to more detailed and structural texture. Across all the settings, FSGS delivers superior performance compared to all the baselines, affirming the effectiveness of our proposed method.

###### Fig. 9: Qualitative Results on Datasets Collected by Mobile Phones. We test the generalization capacity of all methods on self-captured iPhone images, with a calibration process from COLMAP. Our method reveals the majority of scene details despite only adopting three views in training.

GT 3D-GS FreeNeRF SparseNeRF Ours

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

Ourdoor-1Indoor-1Indoor-2

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

Table 7: Quantitative Comparison in Mobile Phone Datasets, with 3 Training Views. Our method continues to achieve the best performance in the challenging mobile phone dataset.

|Methods|1/8 Resolution|1/4 Resolution|
|---|---|---|
| |FPS↑ PSNR↑ SSIM↑ LPIPS↓|FPS↑ PSNR↑ SSIM↑ LPIPS↓<br><br>|
|Mip-NeRF 3D-GS|0.07 14.74 0.337 0.602 225 16.29 0.408 0.439<br><br>|0.14 13.84 0.284 0.631 127 15.83 0.413 0.530<br><br>|
|DietNeRF RegNeRF FreeNeRF SparseNeRF Ours<br><br>|0.05 13.62 0.263 0.598<br><br>0.07 17.41 0.517 0.440<br><br>0.07 18.07 0.497 0.426 0.07 18.79 0.539 0.441<br><br><br>263 19.54 0.539 0.403<br><br>|0.08 12.57 0.228 0.722<br><br>0.14 16.44 0.443 0.504<br><br>0.14 17.14 0.462 0.509 0.04 17.82 0.472 0.524<br><br><br>190 18.41 0.493 0.471<br><br>|

#### 7.2 FSGS on Mobile Phones Data

To validate the generalization capability of FSGS in various real-world settings, we created a new dataset using only a consumer smartphone, the iPhone 15 Pro. This dataset contains three scenes, comprising two indoor scenes and one outdoor scene. Each scene consists of a collection of RGB images under 5712×4284 resolution, with the viewpoint number ranging from 20 to 40. Our data calibration pipeline follows the same process procedures as the LLFF datasets [29], and we also select every 8-th image as the novel views for evaluation. For training, we evenly sample 3 images from the remaining views. These images are

RGB (GT) Depth (3D-GS) Depth (FreeNeRF) Depth (SparseNeRF) Depth (Ours)

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

- Fig. 10: Visual Comparisons of Predicted Depth. We visualize the estimated scene depth from all baselines. Noticeable NeRF alias artifacts are found in SparseNeRF and FreeNeRF. 3D-GS produces an oversmoothed scene geometry, while our method demonstrates visually pleasing geometric details.

then downsampled to 4× and 8× for both training and evaluation. We utilize the pretrained monocular depth estimator to predict the depth for input images, and the poses are computed via COLMAP. We compare our method with FreeNeRF [57], SparseNeRF [50], and 3D-GS [26].

Tab. 7 presents the quantitative results, where FSGS outperforms SparseNeRF with over 0.75 higher PSNR and runs 3,757× faster, a significant leap that underscores its potential for practical, real-world applications where speed is crucial and the environment is intricate. We also visualize the qualitative results in Fig. 9, where FSGS significantly improves the visual quality of the scenes over 3D-GS, particularly in the realm of geometry reconstruction. FreeNeRF and SparseNeRF are constrained by geometric continuity from unseen perspectives and do not fully capitalize on the available depth information.

#### 7.3 Visual Comparisons of the Rendered Depth

We demonstrate the qualitative results of the predicted depth for each methods, as shown in Fig. 10. We compare our method with 3DGS [26], FreeNeRF [57] and SparseNeRF [50]. In the left we visualize the ground truth of the images. FSGS significantly outperforms the three baselines in terms of depth quality and details. The depth maps produced by FSGS are not only more accurate but also exhibit a higher level of detail, showcasing its robustness in reconstructing complex scenes. In contrast, both FreeNeRF and SparseNeRF exhibit limitations in geometric modeling and struggle to accurately learn complex geometries, leading to a distorted scene representation. Although SparseNeRF leverage depth prior, it still does not fully capture the fine-grained structures in real-world structures, resulting in a noticeable drop in quality compared to FSGS. 3D-GS, on the other hand, tends to lose fine details in the areas from away the camera, leading to a diminished overall quality in the depth and texture of distant objects.

