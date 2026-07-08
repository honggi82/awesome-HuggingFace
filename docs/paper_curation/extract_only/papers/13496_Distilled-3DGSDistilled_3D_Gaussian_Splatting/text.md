## Distilled-3DGS: Distilled 3D Gaussian Splatting

Lintao Xiang1,2∗ Xinkai Chen2∗ Jianhuang Lai3 Guangcong Wang2†

1The University of Manchester, 2Vision, Graphics, and X Group, Great Bay University, 3Sun Yat-Sen University

[Figure 1]

[Figure 2]

[Figure 3]

Ground-Truth

Ours

PSNR: 31.54 LPIPS: 0.192 Nums: 0.46 M

| |
|---|

| |
|---|

||Ours Mini-Splatting EAAGLES LP-3DGS Taming 3D-GS 3D-GS|
|---|
<br><br>|
|---|

28.0 27.8 27.6 27.4 27.2 27.0

|[Figure 4]|
|---|

|[Figure 5]|
|---|

# arXiv:2508.14037v1[cs.CV]19Aug2025

PSNR(dB)

[Figure 6]

[Figure 7]

Mini-Splatting 3D-GS

PSNR: 31.36 LPIPS: 0.225 Nums: 0.58 M

PSNR: 31.43 LPIPS: 0.221 Nums: 1.24 M

| |
|---|

| |
|---|

0.50 1.00 1.50 2.00 2.50 3.00 3.50

|[Figure 8]|
|---|

|[Figure 9]|
|---|

Number of Gaussians(106)

Figure 1. Compared with state-of-the-art 3DGS-based methods on Mip360 dataset, our Distilled-3DGS introduces a novel lightweight framework for high-quality view synthesis, achieving better detail preservation with lower storage.

### Abstract

### 1. Introduction

Novel view synthesis (NVS) is a fundamental task [40] in computer vision and computer graphics, serving as a cornerstone in many applications, e.g., VR/AR and autonomous driving. The goal of NVS is to generate photorealistic images from novel, previously unseen viewpoints. This process typically begins by constructing a 3D representation from a set of existing 2D observations. 3D Gaussian Splatting (3DGS) [13] has recently demonstrated remarkable effectiveness in novel view synthesis. This approach employs a point-based representation augmented with 3D Gaussian attributes and utilizes a rasterization-based rendering pipeline to synthesize images. However, 3DGS necessitates a large number of 3D Gaussians to ensure high-fidelity image rendering, particularly in the presence of complex scenes. This limits their applicability on platforms and devices with constrained computational resources and limited memory capacity.

3D Gaussian Splatting (3DGS) has exhibited remarkable efficacy in novel view synthesis (NVS). However, it suffers from a significant drawback: achieving high-fidelity rendering typically necessitates a large number of 3D Gaussians, resulting in substantial memory consumption and storage requirements. To address this challenge, we propose the first knowledge distillation framework for 3DGS, featuring various teacher models, including vanilla 3DGS, noiseaugmented variants, and dropout-regularized versions. The outputs of these teachers are aggregated to guide the optimization of a lightweight student model. To distill the hidden geometric structure, we propose a structural similarity loss to boost the consistency of spatial geometric distributions between the student and teacher model. Through comprehensive quantitative and qualitative evaluations across diverse datasets, the proposed Distilled-3DGS—a simple yet effective framework without bells and whistles—achieves promising rendering results in both rendering quality and storage efficiency compared to state-of-theart methods. Project page: https://distilled3dgs.github.io. Code: https://github.com/lt-xiang/Distilled-3DGS.

On the other hand, knowledge distillation [3] has proven effective in compressing neural networks across various vision tasks. However, applying it to 3D Gaussian Splatting (3DGS) introduces unique challenges. First, 3DGS is an explicit and unstructured representation composed of variable 3D Gaussians, lacking the consistent latent feature spaces typically leveraged in conventional KD. Second, the Gaussian primitives are scene-dependent and unordered, preventing straightforward correspondence between teacher

∗Equal contribution. †Corresponding author.

and student components. Third, since rendering outputs are view-dependent and non-differentiable w.r.t. individual Gaussians, designing stable and informative distillation losses becomes nontrivial. These challenges necessitate careful design of both teacher model ensembles and geometry-aware distillation strategies, as introduced in our Distilled-3DGS framework.

To address this challenge, we propose a lightweight 3D Gaussian representation framework based on knowledge distillation, termed Distilled-3DGS. This approach enhances the performance of a compact student model by distilling knowledge from multiple complex teacher models. The overall pipeline of Distilled-3DGS comprises two main stages: multi-teacher training and student training via distillation. In the multi-teacher training stage, we begin by training a standard 3DGS model. Subsequently, we introduce random perturbations and dropout strategies separately to obtain two additional diverse teacher models. During the distillation-based student training stage, we first aggregate predictions from the teacher ensemble to synthesize a pseudo image. The student model is then supervised by enforcing similarity between its rendered output and this pseudo image. This strategy effectively transfers rich knowledge priors from the teacher models, providing a more comprehensive and robust supervisory signal for optimizing the student model.

In the context of Distilled-3DGS, the teacher model typically contains dense and high-quality point clouds, while the student model is compressed to obtain much sparser points due to efficiency or deployment considerations. Despite its sparsity, the student model is trained to reconstruct the same underlying 3D scene as the teacher. So it is expected to preserve the essential spatial layout and local geometric patterns present in the teacher model. To facilitate this, we propose a spatial distribution distillation strategy that guides the student to align with the teacher’s point distribution in space. Rather than enforcing exact point-wise correspondence, this structure-aware supervision encourages the student to learn how the teacher organizes points, focusing on global and local geometric consistency. In this way, structural knowledge from the teacher can be effectively and comprehensively distilled into the student model.

In summary, our main contributions are as follows: 1) We propose a novel distillation-based 3DGS framework, termed as Distilled-3DGS, which is the first method to leverage multi-teacher knowledge priors to optimize 3DGS and boost rendering quality and storage efficiency. 2) We propose a spatial distribution consistency distillation to enable the student model to learn similar geometric structure distributions from the teacher model. 3) Extensive experiments on several real-world datasets—including Mip-NeRF 360, Tanks&Temples, and Deep Blending—demonstrate that the proposed Distilled-3DGS achieves promising per-

formance in both rendering quality and efficiency compared to existing methods.

- 2. Related Work
- 3D Representation. Radiance fields have been extensively employed for 3D scene reconstruction, particularly in the context of novel view synthesis. Neural Radiance Fields (NeRFs) have achieved remarkable progress by learning neural volumetric representations of 3D scenes, enabling high-fidelity image synthesis via volumetric rendering techniques. After that, many works have focused on improving the rendering quality [1, 2] and accelerating the efficiency [6, 11, 31, 39] of NeRFs. Nevertheless, NeRF-based approaches continue to rely on numerous MLP queries during rendering, thereby limiting their applicability in scenarios with real-time constraints. To enhance the training and rendering efficiency, Plenoxels [6] improve NeRF efficiency by optimizing a sparse voxel grid and removing the need for MLPs, while Instant NGP [39] uses hash-grid encodings to boost expressivity. However, despite these improvements, grid-based methods still struggle to achieve real-time rendering. Recently, 3D Gaussian Splatting (3DGS) [13] has gained significant attention as an efficient and effective approach for 3D scene representation. 3DGS represents 3D scenes explicitly with millions of anisotropic Gaussians and utilizes differentiable rasterization, enabling real-time, photorealistic view synthesis. When 3DGS overfits the scene by optimizing Gaussian properties, it typically produces many redundant Gaussians, thereby reducing rendering efficiency and substantially increasing memory usage.

To tackle these issues, several subsequent approaches have aimed to prune redundant Gaussians based on handcrafted importance criteria. Mini-Splatting [5] addresses overlapping and reconstruction artifacts by employing blur splitting, depth reinitialization, and stochastic sampling. Radsplatting [24] enhances robustness by applying a max operator to derive importance scores from ray contributions. Taming-3DGS [21] leverages pixel saliency and gradient information for selective densification, while LP-3DGS [38] utilizes a learned binary mask for efficient Gaussian pruning. Additionally, Scaffold-GS [20] proposes a structured dual-layer hierarchical scene representation to better regulate the distribution of 3D Gaussian primitives.

Overall, the aforementioned methods that prioritize efficiency generally achieve faster performance, but this often comes at the expense of rendering quality compared to the standard 3DGS. Conversely, approaches that focus on enhancing rendering quality tend to demand substantially higher computational resources. To address this trade-off, we propose a knowledge distillation-based 3DGS framework that simultaneously improves storage efficiency and rendering fidelity.

Knowledge Distillation.Knowledge distillation (KD)

transfers knowledge from a large teacher model to a compact student model. Initially proposed for model compression [3, 10], KD began with matching teacher outputs and was later extended to mimic intermediate representations [27, 32]. KD has since been applied to various tasks, including detection [18, 35],segmentation [12, 19], and generation [17, 36]. To overcome the limitations of singleteacher KD, multi-teacher distillation (MKD) is proposed to aggregate diverse knowledge from multiple teachers. While early approaches assign equal weights [7, 30], recent methods adopt adaptive strategies, such as entropy-based weighting (EB-KD [15]) and confidence-aware distillation (CA-MKD [33]). MMKD [34] further introduces metalearning to jointly distill features and logits. These methods often rely on CNNs for structured feature spaces, facilitating effective alignment via soft labels or intermediate supervision.

However, extending KD to 3D Gaussian Splatting (3DGS) poses new challenges. 3DGS uses an explicit and unstructured representation composed of a variable set of discrete Gaussian primitives. These primitives are unordered, scene-dependent, and lack a shared latent space, making it infeasible to directly align elements between teacher and student. As a result, existing KD strategies must be fundamentally rethought to accommodate the unique properties of 3DGS. Based on the above analysis, we propose to utilize multiple pre-trained 3DGS teacher models to render high-quality images as supervision targets and optimize the student model. Besides, we propose a spatial distribution distillation strategy that guides the student to align with the teacher’s point distribution in space.

### 3. Method

In this section, we present Distilled-3DGS, an efficient 3D Gaussian Splatting framework that distills knowledge from powerful teacher models to a small student model.

#### 3.1. Preliminaries

3DGS [13] is a cutting-edge method for novel view synthesis, which fundamentally depends on an explicit pointbased representation to achieve high-fidelity rendering from arbitrary viewpoints. Specifically, 3DGS models a scene as a set of Gaussian distributions. The ith Gaussian primitive is denoted as Gi = (µi,Σi,oi,fi), where µi is the 3D position, Σi is the covariance matrix, fi represents spherical harmonics (SH) coefficients associated with the Gaussian, and oi indicates opacity. The effect of the ith Gaussian primitive at position x is represented by Gi(x) = e−21(x−µi)TΣ−i 1(x−µi), where Σi can be factorized as Σi = RSSTRT, with R as a rotation matrix and S as a scaling matrix, both being learnable. Subsequently, the Gaussians are mapped to the 2D image plane via the projection matrix W, resulting in the projected 2D covariance matrix

′

i = JWΣiWTJT, where J represents the Jacobian of the affine projection. The pixel color is computed through alpha blending as follows:

Σ

- i−1
- j=1

N

(1 − αj), (1)

c =

ciαi

i=1

where N is the number of Gaussians covering the pixel, the color ci is derived from the spherical harmonics (SH) coefficients of each Gaussian, while αi is determined by the projected 2D covariance matrices Σ

′

and the associated opacity oi. The Gaussian parameters are optimized using a photometric loss [13] function, with the posed training images providing the ground-truth supervision.

#### 3.2. Distilled 3D Gaussian Splatting

3DGS has enabled highly detailed and accurate 3D scene reconstruction, yet such state-of-the-art models are often extremely large and computationally expensive, limiting their practicality in real-time and resource-constrained scenarios. Knowledge distillation (KD) has emerged as a highly effective and popular approach for model compression in image classification, semantic segmentation [19], and object detection [4]. Inspired by these observations, one could ask if knowledge distillation works for 3DGS. Different from conventional KD in neural networks, it is an explicit 3D representation with variable unstructured 3D Gaussians, which remains unexplored. To address this problem, we first provide an overview of the proposed Distilled-3DGS, and then detail the design of diverse teacher models and the distillation method, as discussed in the following.

#### 3.3. Overview of Distill-3DGS

As shown in Fig.2, we firstly train three independent 3DGS models with diverse strategies to obtain cumbersome yet high-quality teacher models with millions of Gaussian primitives, optimized by the standard photometric loss. Then we leverage the optimized teacher 3DGS representation to generate pseudo labels by fusing multiple teachers’ outputs. In the training process of the student model, pseudo labels are leveraged to transfer prior knowledge from multiple teachers to a single student. To obtain a lightweight student model, we prune the number of Gaussians based on the importance score proposed in Mini-Splatting [5]. To distill knowledge hidden in geometric structure of 3D Gaussians, we propose structural knowledge distillation for unstructured 3D Gaussians to encourage the similar spatial distribution between teacher and student models.

#### 3.4. Training Diverse Teacher models

To provide the student model with richer supervision signals and facilitate a better understanding of 3D scene structures and details, we train the base 3DGS model multiple

Perturbation

𝐺𝐺𝑝𝑝𝑝𝑝𝑝𝑝𝑝𝑝

|[Figure 10]<br><br>⋮<br><br>[Figure 11]<br><br>𝐼𝐼𝑝𝑝𝑝𝑝𝑝𝑝𝑝𝑝<br><br>[Figure 12]<br><br>𝐼𝐼𝑠𝑠𝑝𝑝𝑑𝑑𝑝𝑝<br><br>𝐼𝐼𝑠𝑠𝑠𝑠𝑠𝑠| |
|---|---|
| | |

Densification

| |
|---|

| |
|---|

Render

Init.

Dropout

𝐺𝐺𝑠𝑠𝑝𝑝𝑑𝑑𝑝𝑝

|[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>SfM<br><br>[Figure 16]|
|---|

Densification

| |
|---|

| |
|---|

Render

Init.

Standard

𝐺𝐺𝑠𝑠𝑠𝑠𝑠𝑠 Init.

Densification

| |
|---|

| | |
|---|---|
| | |

Render

Color Loss

Structure Similarity

𝐺𝐺𝑠𝑠𝑠𝑠𝑠𝑠 Init.

[Figure 17]

Simplication

| |
|---|

| |
|---|

Render

𝐼𝐼𝑠𝑠𝑠𝑠𝑠𝑠

Figure 2. The architecture of multi-teacher knowledge distillation framework for 3DGS. It consists of two stages. First, a standard teacher model Gstd is trained, along with two variants: Gperb with random perturbation and Gdrop with random dropout. Then, a pruned student model Gstd is supervised by the outputs of these teachers. Additionally, a spatial distribution distillation strategy is introduced to help the student learn structural patterns from the teachers.

introducing parameter perturbations during training, the model is compelled to learn scene structures that are less dependent on the precise positions and shapes of Gaussian primitives, thereby enhancing its generalization capability.

times using diverse strategies to enhance the robustness and generalization ability of the teacher models.

Regular training. First, we train a vanilla 3DGS model Gstd with the same settings in [13]. The training loss is defined as:

Random Dropout. Dropout [8, 26] is recognized as one of the most effective techniques for improving model robustness by randomly deactivating a subset of neurons during training. Inspired by the remarkable success of dropout, we introduce a Random Dropout Strategy to further enhance both the robustness and redundancy of the representation of model Gdrop. Specifically, during training, each Gaussian primitive is randomly deactivated with probability p, while the remaining primitives are optimized to fit the observed views. During inference, all Gaussian primitives are activated to facilitate novel view synthesis. By randomly dropping a subset of Gaussian primitives during training, our approach encourages the model to learn a collaborative and distributed scene representation, rather than relying on a limited set of critical Gaussian primitives. Inspired by [25], the dropping rate rt is updated based on the current iteration index t as follows:

##### Lcolor = (1 − λ)L1(I,Iˆ gt) + λLD-SSIM(I,Iˆ gt) (2)

Feature Perturbation. Then, we train a 3DGS model Gperb with random perturbations on Gaussian parameters, following the same optimization and density control strategies introduced in 3DGS. At each training iteration t, each Gaussian is perturbed as:

##### Gtperb = Gtstd + δt, (3)

where random noises are added to corresponding Gaussian parameters including 3D positions µp, 3D rotations Rp, scales Sp, and opacities op. Since the representation of rotation as a 3D matrix is discontinuous, we instead perturb its continuous 6D representation as :

##### Rˆ tp = f−1 f(Rtp) + δtR , δpt ∈ R6 (4)

rt = rinit · (t − t0)/(t1 − t0), (5) where t0 and t1 are the starting and end iterations of in-

where f and f−1 are the forward and inverse mappings between the rotation matrix and its 6D representation. By

troducing random dropout strategy. rinit is the initial drop rate.

#### 3.5. Training Efficient Student Model

Knowledge distillation is a technique that transfers knowledge from a larger teacher model to a smaller, faster student model. This approach is particularly useful when deploying deep neural networks in resource-constrained environments. The student model, trained under the guidance of the teacher, can achieve comparable performance with significantly fewer parameters. This process mainly consists of pseudo label generation and student training.



Student Teacher

ℎ𝑠𝑠𝑠𝑠𝑠𝑠 ℎ𝑠𝑠𝑡𝑡𝑡𝑡

Pseudo labeling with teacher model. By evaluating the optimized diverse teachers Gstd, Gperb and Gdrop, we can render per-view image denoted as Istd, Iperb and Idrop, these rendered images as prior knowledge are further aggregated by average strategy to generate pseudo label Itea and guide the learning of the student model.

Figure 3. Overview of Spatial Distribution Distillation.

determine a common 3D bounding box that encompasses both sets of points.The bounding box is partitioned into a regular voxel grid with a resolution of 128 . Each point from both clouds is assigned to a corresponding voxel cell based on its spatial coordinates. For point cloud Ptea and Pstu, we count the number of points falling into every voxel separately, resulting in two high-dimensional voxel occupancy histograms htea and hstu. These histograms are then normalized to form probability distributions that capture the spatial structure of each cloud, independent of point count or density. Finally, we compute the cosine similarity between their normalized voxel occupancy histograms. The cosine similarity loss is given by:

Conventional Knowledge Distillation. In the training process of the student network, we utilize the ground-truth labels and the pseudo label of multiple teachers as additional knowledge to jointly guide the optimization of student model Gstu. Following the conventional knowledge distillation loss, we formulate our objective by incorporating fused knowledge from multiple teachers, as follows:

Lkd = Lcolor(Istu,Igt) + λkdLcolor(Istu,Itea) (6)

Spatial Distribution Distillation. In the context of 3DGS, these optimized teacher models provide a structure-rich and dense 3D point distribution. In contrast, the student model operates under sparse or limited sampling conditions and aims to reconstruct the similar scene representation. Therefore, we hope to design a structural similarity loss to encourage the student model to capture spatial geometric distributions similar to those of the teacher. However, challenges arise due to varying point densities, sampling noise, and non-uniform point distributions between student and teacher models. Direct coordinate-based distance measures are often insufficiently robust to these variations.

htea · hstu ∥htea∥2 ∥hstu∥2

(7)

Lhist = 1 −

This loss quantitatively reflects how closely the Student point cloud matches the Teacher’s structural distribution. The final loss function during the student training phase is defined as:

L = Lkd + Lhist (8)

To address this problem, we leverage the voxel histogram representation shown in Fig. 3, which discretizes the 3D space into regular voxels and counts the number of points within each voxel. This approach encodes the spatial distribution of points as a high-dimensional structural feature, inherently robust to point count and density variations. Comparing voxel histograms thus enables efficient and structureaware similarity evaluation between different point clouds. To this end, we propose a voxel histogram-based structural loss to enhance the structural learning capability of the student model.

### 4. Experiments

Datasets. We conducted experiments on three widely used datasets: LLFF [22], Mip360 [2] and two scenes from the Tanks and Temples(T&T) [14]. LLFF contains eight scenes with forward-facing camera. Mip-NeRF360 comprises nine distinct scenes that encompass both expansive outdoor scenes and intricate indoor settings. These scenes exhibit a wide range of capture styles and encompass both bounded indoor environments as well as large, unbounded outdoor settings. To partition the dataset into training and test sets, we follow the protocol of 3DGS by allocating every eight image to the test set. The resolution of all images is kept consistent with that used in 3DGS.

During the training phase of the student model, we firstly obtain point cloud Ptea and Pstu from the optimized standard teacher model Gstd and student model Gstu. Then, we

GT Ours Taming 3DGS Mini-splatting 3DGS

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

| |
|---|

| |
|---|

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

| |
|---|

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

| |
|---|

| |
|---|

| |
|---|

[Figure 37]

[Figure 38]

[Figure 39]

- Figure 4. Visualized comparison on the Bicycle, Garden, and Kitchen scenes. As shown in the rendered images and corresponding local regions, the proposed method can better capture fine details.

|Type<br><br>|Method<br><br>|Mip-NeRF 360<br><br>|Tanks & Temples|Deep Blending|
|---|---|---|---|---|
| | |PSNR↑ SSIM↑ LPIPS↓ #G(106)↓<br><br>|PSNR↑ SSIM↑ LPIPS↓ #G(106)↓|PSNR↑ SSIM↑ LPIPS↓ #G(106)↓|
|Quality|Plenoxels (CVPR’22)<br><br>INGP-Big (SIGGRAPH’22)<br><br>Mip-NeRF360 (CVPR’22)<br><br>3D-GS (TOG’23)<br><br>3D-GS∗<br><br>ScaffoldGS (CVPR’24)<br><br>|23.08 0.626 0.463 25.59 0.699 0.331 27.69 0.792 0.237 27.26 0.815 0.214 3.5 27.39 0.819 0.219 3.43 27.60 0.812 0.222 0.6|21.08 0.719 0.379 -<br><br>21.92 0.745 0.305 -<br><br>22.22 0.759 0.257 -<br><br>23.14 0.841 0.183 2.0<br><br><br>23.61 0.849 0.180 1.84<br><br>24.08 0.854 0.165 0.6<br>|23.06 0.795 0.510 -<br><br>24.96 0.817 0.390 -<br><br><br>29.40 0.901 0.245 -<br><br>29.41 0.903 0.243 3.2<br><br><br>29.55 0.912 0.241 3.24<br><br>30.25 0.907 0.245 0.40<br><br><br>|
|Efficiency|CompactGaussian (CVPR’24)<br><br>LP-3DGS (NIPS’24)<br><br>MiniSplatting (CVPR’24)<br><br>EAGLES(ECCV’24)<br><br>Taming 3DGS(SIGGRAPH Asia’24)<br><br>CompGS (ECCV’24)<br><br>Ours<br><br>|27.08 0.798 0.247 1.388 27.47 0.812 0.227 1.959 27.25 0.820 0.217 0.5 27.20 0.809 0.232 1.3 27.71 0.820 0.207 0.63 27.12 0.806 0.240 0.84 27.81 0.827 0.202 0.49<br><br>|23.32 0.831 0.201 0.836 23.60 0.842 0.188 1.244 23.21 0.836 0.203 0.32<br><br>23.26 0.837 0.201 0.7<br><br>23.95 0.837 0.201 0.29 23.44 0.838 0.198 0.52 23.76 0.845 0.179 0.25<br><br>|29.79 0.901 0.258 1.06 - - - 29.98 0.908 0.253 0.40<br><br>29.86 0.910 0.246 1.20 29.82 0.904 0.237 0.27 29.90 0.907 0.251 0.55<br>29.87 0.916 0.251 0.33<br>|

Table 1. Quantitative evaluations across the Mip-NeRF 360, Tanks&Temples, and Deep Blending datasets. Best and second-best results are highlighted for each. * denotes our re-runs of the existing codebase to ensure a fair evaluation.

Evaluation Metrics. For the evaluation of comparative view synthesis quality, we adopt several widely used quantitative metrics, including Peak Signal-to-Noise Ratio (PSNR), Learned Perceptual Image Patch Similarity (LPIPS)[37], and Structural Similarity Index Measure (SSIM)[28]. PSNR and SSIM primarily assess pixellevel fidelity and structural consistency, respectively, while LPIPS reflects a more human-aligned assessment of visual quality. In addition, we present the average memory usage associated with storing the optimized parameters.

ter the 15000th iteration as in 3DGS. For the teacher training with random perturbation, random noise δt is applied to those Gaussian primitives exhibiting large view-space positional gradients, as these typically correspond to regions that have not yet been well reconstructed, starting from 500th to 15000th iteration with interval 500. Introducing appropriate perturbations in this manner can enhance the robustness of the model. For the teacher training with random dropout, rinit, t0 and t1 are respectively set 0.2, 500 and 15000. Each Gaussian primitive is randomly deactivated with probability p, and the remaining primitives are optimized to fit the observed views. For inference, all Gaussian primitives are activated to enable novel view synthesis.

Implementations. Our implementations are based on the official 3DGS codebase. All models were trained on a single NVIDIA RTX 3090 GPU. Training process of Distilled3DGS contains two stages: In the training phase of teacher models, these teacher models are trained for 30k iterations by following 3DGS. Gaussian densification is stopped af-

In the training phase of student model, the total number of optimization steps is set to 30K. Densification is applied up to the 15000th iteration, after which simplification is car-

ried out at both the 15000th iterations. Subsequently, the structural similarity loss is computed and applied every 500 iterations throughout the optimization process.

#### 4.1. Comparisons with State-of-the-Arts

We evaluate model performance across several real-world datasets, including Mip-NeRF 360, Tanks&Temples, and Deep Blending. For NeRF-based methods, we compare with the state-of-the-art Mip-NeRF 360 [2] and two efficient NeRF variants, INGP [23] and Plenoxels [6]. For 3DGSbased methods, our comparisons include the vanilla 3DGS as well as leading Gaussian simplification techniques such as CompactGaussian [16], LP-3DGS [38], EAGLES [9], MiniSplatting [5], and Taming 3DGS [21]. For vanilla 3D-GS, we include both the metrics reported in [13] and those obtained through our own experimental runs. The quantitative results for all datasets are presented in Table 1. Our method surpasses both the voxel grid-based approach, Plenoxels, and the fast NeRF-based method, INGP, across all datasets and evaluation metrics. Compared to the Mip-NeRF360 baseline, Distilled-3DGS yields PSNR improvements of 0.12 dB, 1.54 dB, and 0.47 dB on the Mip-NeRF360, Tanks&Temples, and Deep Blending datasets, respectively, verifying its effectiveness across diverse datasets.

Compared to other 3DGS-based methods, our proposed Distilled-3DGS consistently outperforms the baseline 3DGS across all three metrics while using significantly fewer Gaussians on Mip360 dataset. We attribute this improvement to the comprehensive knowledge supervision provided by the diverse teacher models. Specifically, compared to the vanilla 3DGS, our Distilled-3DGS achieves PSNR improvements of 0.55 dB on Mip-NeRF360, 0.62 dB on Tanks&Temples, and 0.46 dB on Deep Blending. The number of Gaussians is also reduced by 86.0%, 87.5%, and 89.6% on these three datasets, respectively. Compared with these 3DGS simplification methods, such as Taming 3DGS, our method also can improve rendering quality while maintaining a comparable number of Gaussians. Besides, visual comparison is illustrated in Fig. 4, compared with existing simplification approaches such as Taming-3DGS, Mini-Splatting, and the vanilla 3DGS, our Distilled-3DGS achieves rendering results that preserve fine details most faithfully to the ground truth while utilizing a significantly reduced number of Gaussian primitives.

#### 4.2. Ablation Studies and Further Analyses

To study the contribution of each component in the proposed framework, we conducted a series of ablation experiments on the Deep Blending and Tanks&Temples datasets. Effect of the number of teachers. To further understand the contribution of each teacher model in our distillation framework, we conducted ablation studies by gradually

removing the Perturbation-based, Dropout-based 3DGS teachers, respectively. The quantitative results are shown in Table 2. Specifically, the student model distilled from all three teachers consistently achieves the best performance, indicating that each teacher provides complementary knowledge. Teacher Gperb enhances the student’s robustness to input variations, while teacher Gdrop prevents overfitting and encourages generalization. The regular 3DGS teacher Gstd serves as a strong baseline, providing high-fidelity supervision. The progressive decrease in performance with the removal of these specialized teachers underscores their critical roles in enriching the distilled knowledge, validating the effectiveness of leveraging diverse teacher models for optimal student performance.

Deep Blending Tanks&Temples PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

Method

Ours 29.87 0.916 0.251 23.76 0.845 0.179 Without Gdrop 29.71 0.899 0.257 23.58 0.840 0.186 Without Gperb 29.63 0.878 0.262 23.43 0.838 0.195 Without Lhist 29.47 0.871 0.263 23.32 0.836 0.197

Table 2. Ablation study on two datasets.

Grid size PSNR↑ SSIM↑ LPIPS↓ Train Mem.(MB)

32 27.51 0.819 0.198 9564 64 27.62 0.821 0.199 10232 128 27.81 0.827 0.202 12235 256 27.92 0.829 0.203 15456

Table 3. The impact of different grid size in spatial distribution distillation.

Room (Mip360) PSNR↑ SSIM↑ LPIPS↓ #G(106)↓

Method

Teachers (Gstd+Gdrop+Gperb) 32.15 0.935 0.185 1.56 3DGS 31.59 0.920 0.200 1.50 Student(Base) 31.54 0.927 0.193 0.46 Student(Big) 31.89 0.934 0.189 1.13 Student(Small) 31.39 0.923 0.194 0.21

Table 4. The impact of the number of Gaussians.

|Teacher|Student<br><br>|Room (Mip360)| | | |
|---|---|---|---|---|---|
| | |PSNR↑<br><br>|SSIM↑<br><br>|LPIPS↓|#G(106)↓|
|Gstd/Gperb/Gdrop Gstd/Gstd/Gstd Gstd Gperb Gdrop|Gstu<br><br>|31.54 31.36 31.19 31.31 31.23<br><br>|0.927 0.923 0.918 0.921 0.919<br><br>|0.193 0.191 0.187 0.186 0.189<br><br>|0.460 0.469 0.465 0.453 0.459|

Table 5. The impact of different teachers

Effect of Spatial Distribution Distillation. The results presented in Table 2 verify that spatial distribution distillation plays a crucial role in enhancing rendering quality. Without this, performance in PSNR is decreased by 0.16 dB. In addition, we investigate the impact of varying grid

sizes on the student model training using the Mip360 dataset shown in Table 3. Generally, a larger grid size produces smaller voxel dimensions and a greater number of voxels, leading to a more detailed scene representation. While increasing the grid size can improve PSNR performance, it also incurs a substantial increase in GPU memory.

Impact of the number of Gaussians. We conduct experiments on the Room scene from Mip360 to evaluate the impact of Gaussian count. Table 4 reports the reconstruction quality and the number of Gaussians for different model variants. The ensemble of three diverse teacher models achieves the highest PSNR of 32.15 dB. Compared to vanilla 3DGS, the student (Base) model—trained via multiteacher distillation—preserves comparable reconstruction quality while significantly reducing the number of Gaussians. Although the student (Big) model achieves higher PSNR, it uses nearly as many Gaussians as the teacher models. In contrast, the student (Small) model applies further pruning, resulting in only a slight PSNR drop of 0.15 dB.

Impact of different teachers. We analyze the effects of various teacher models on the performance of the student model. As shown in Table 5, employing multiple diverse teachers (Gstd,Gperb,Gdrop) to distill the student yields the best overall performance. In contrast, using three standard teachers(Gstd) results in a lower PSNR (31.36), and single-teacher configurations perform even worse compared to these teacher ensembles. These results highlight that diversity among teachers provides richer and more complementary supervisory signals, thereby enhancing student model performance.

### 5. Conclusion

In this paper, we proposed a multi-teacher distillation framework for 3DGS, aiming to preserve reconstruction quality under significantly reduced Gaussian counts. By leveraging knowledge from multiple teacher models, our approach effectively transfers both scene geometry and appearance priors to a more compact student representation. Besides, we leverage a spatial distribution distillation strategy to encourage the student to learn spatial geometric distributions consistent with those of a standard teacher model. Extensive experiments across different scenes demonstrate that our distilled student model-Distilled-3DGS achieves promising performance with substantially fewer Gaussians, highlighting the potential of our method for deployment in memory-constrained or real-time scenes.

Limitation. Distilled-3DGS also has some drawbacks: first, it requires pre-training multiple high-performance teacher models, increasing training time and computational resources by at least N-fold compared to a single model; second, generating distillation soft labels via multi-model inference significantly increases GPU memory usage. Future work could explore end-to-end distillation pipelines or

adaptive pruning strategies for Gaussian parameters to further improve efficiency and generalization.

### Acknowledgement

The computational resources are supported by SongShan Lake HPC Center (SSL-HPC) in Great Bay University. This work was also supported by Guangdong Research Team for Communication and Sensing Integrated with Intelligent Computing (Project No. 2024KCXTD047).

### References

- [1] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, and Ricardo Martin-Brualla. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5855–5864, 2021. 2
- [2] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470–5479, 2022. 2, 5, 7
- [3] Cristian Buciluˇa, Rich Caruana, and Alexandru NiculescuMizil. Model compression. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 535–541, 2006. 1, 3
- [4] Guobin Chen, Wongun Choi, Xiang Yu, Tony Han, and Manmohan Chandraker. Learning efficient object detection models with knowledge distillation. Advances in neural information processing systems, 30, 2017. 3
- [5] Guangchi Fang and Bing Wang. Mini-splatting: Representing scenes with a constrained number of gaussians. In European Conference on Computer Vision, pages 165–181. Springer, 2024. 2, 3, 7
- [6] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5501–5510, 2022. 2, 7
- [7] Takashi Fukuda, Masayuki Suzuki, Gakuto Kurata, Samuel Thomas, Jia Cui, and Bhuvana Ramabhadran. Efficient knowledge distillation from an ensemble of teachers. In Interspeech, pages 3697–3701, 2017. 3
- [8] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pages 1050–1059. PMLR, 2016. 4
- [9] Sharath Girish, Kamal Gupta, and Abhinav Shrivastava. Eagles: Efficient accelerated 3d gaussians with lightweight encodings. In European Conference on Computer Vision, pages 54–71. Springer, 2024. 7
- [10] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 3
- [11] Wenbo Hu, Yuling Wang, Lin Ma, Bangbang Yang, Lin Gao, Xiao Liu, and Yuewen Ma. Tri-miprf: Tri-mip representation for efficient anti-aliasing neural radiance fields. In

- Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19774–19783, 2023. 2
- [12] Deyi Ji, Haoran Wang, Mingyuan Tao, Jianqiang Huang, Xian-Sheng Hua, and Hongtao Lu. Structural and statistical texture knowledge distillation for semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16876–16885, 2022. 3
- [13] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 1, 2, 3, 4, 7

- [14] Arno Knapitsch, Jaesik Park, and Qian-Yi Zhou. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36(4):1–13,

2017. 5

- [15] Kisoo Kwon, Hwidong Na, Hoshik Lee, and Nam Soo Kim. Adaptive knowledge distillation based on entropy. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7409–7413. IEEE, 2020. 3
- [16] Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation for radiance field. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21719– 21728, 2024. 7
- [17] Muyang Li, Ji Lin, Yaoyao Ding, Zhijian Liu, Jun-Yan Zhu, and Song Han. Gan compression: Efficient architectures for interactive conditional gans. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5284–5294, 2020. 3
- [18] Quanquan Li, Shengying Jin, and Junjie Yan. Mimicking very efficient network for object detection. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 6356–6364, 2017. 3
- [19] Yifan Liu, Ke Chen, Chris Liu, Zengchang Qin, Zhenbo Luo, and Jingdong Wang. Structured knowledge distillation for semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2604–2613, 2019. 3
- [20] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654–20664, 2024. 2
- [21] Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Francisco Vicente Carrasco, and Fernando De La Torre. Taming 3dgs: High-quality radiance fields with limited resources. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. 2, 7
- [22] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (ToG), 38(4):1–14, 2019. 5
- [23] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1–15, 2022. 7

- [24] Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakotosaona, Michael Oechsle, Daniel Duckworth, Rama Gosula, Keisuke Tateno, John Bates, Dominik Kaeser, and Federico Tombari. Radsplat: Radiance field-informed gaussian splatting for robust real-time rendering with 900+ fps. arXiv preprint arXiv:2403.13806, 2024. 2
- [25] Hyunwoo Park, Gun Ryu, and Wonjun Kim. Dropgaussian: Structural regularization for sparse-view gaussian splatting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21600–21609, 2025. 4
- [26] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929–1958, 2014. 4
- [27] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive representation distillation. arXiv preprint arXiv:1910.10699, 2019. 3
- [28] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6
- [29] Xiaoyang Wu, Daniel DeTone, Duncan Frost, Tianwei Shen, Chris Xie, Nan Yang, Jakob Engel, Richard Newcombe, Hengshuang Zhao, and Julian Straub. Sonata: Selfsupervised learning of reliable point representations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22193–22204, 2025. 11
- [30] Shan You, Chang Xu, Chao Xu, and Dacheng Tao. Learning from multiple teacher networks. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, pages 1285–1294, 2017. 3
- [31] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for real-time rendering of neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5752– 5761, 2021. 2
- [32] Sergey Zagoruyko and Nikos Komodakis. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928, 2016. 3
- [33] Hailin Zhang, Defang Chen, and Can Wang. Confidenceaware multi-teacher knowledge distillation. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4498–4502. IEEE, 2022. 3
- [34] Hailin Zhang, Defang Chen, and Can Wang. Adaptive multiteacher knowledge distillation with meta-learning. In 2023 IEEE International Conference on Multimedia and Expo (ICME), pages 1943–1948. IEEE, 2023. 3
- [35] Linfeng Zhang and Kaisheng Ma. Improve object detection with feature-based knowledge distillation: Towards accurate and efficient detectors. In International conference on learning representations, 2020. 3
- [36] Linfeng Zhang, Xin Chen, Xiaobing Tu, Pengfei Wan, Ning Xu, and Kaisheng Ma. Wavelet knowledge distillation: Towards efficient image-to-image translation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12464–12474, 2022. 3

- [37] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6
- [38] Zhaoliang Zhang, Tianchen Song, Yongjae Lee, Li Yang, Cheng Peng, and Rama Chellappa. Lp-3dgs: Learning to prune 3d gaussian splatting. Advances in Neural Information Processing Systems, 37:122434–122457, 2024. 2, 7
- [39] Yang Katie Zhao, Shang Wu, Jingqun Zhang, Sixu Li, Chaojian Li, and Yingyan Celine Lin. Instant-nerf: Instant on-device neural radiance field training via algorithmaccelerator co-designed near-memory processing. In 2023 60th ACM/IEEE Design Automation Conference (DAC), pages 1–6. IEEE, 2023. 2
- [40] Tinghui Zhou, Shubham Tulsiani, Weilun Sun, Jitendra Malik, and Alexei A Efros. View synthesis by appearance flow. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 286–301. Springer, 2016. 1

### A. Additional ablation experiments

#### A.1. Point cloud structural similarity

Methods for evaluating the similarity of spatial geometric distribution between two point clouds can be broadly categorized into three types: distance-based, distribution matching-based, and learning-based approaches. Our proposed spatial distribution distillation strategy, which utilizes voxel histograms, falls under the distribution matching category. For the distance-based approach, we employ Chamfer Distance (CD) to directly measure the spatial distance between the two point sets. For the learning-based approach, we adopt Sonata [29], a state-of-the-art point cloud representation learning method, to extract point features and compute the similarity loss between the two point clouds.

Tanks&Temples PSNR↑ SSIM↑ LPIPS↓ Mem.(GB) Student Time(min)

Method

Distance-based 23.69 0.832 0.187 23.82 50 Feature-based 23.78 0.847 0.178 40.00 60

Ours 23.76 0.845 0.179 13.83 30

Table 6. Impact of different structural similarity strategy.

As shown in Table 6, both the distance-based and feature-based methods consume considerable memory and time(student model training) yet fail to deliver substantial performance gains. In contrast, our proposed voxel histogram-based method outperforms these two approaches while requiring significantly less memory and computation time.

### B. Per-scene breakdown results

To provide a more detailed evaluation of our model, we present the per-scene breakdown results of Mip-NeRF360, Tanks&Temples and Deep Blending datasets.

Scene Method PSNR↑ SSIM↑ LPIPS↓ #G(106)↑

EAGLES 30.38 0.910 0.250 0.80 Mini-Splatting 30.62 0.915 0.249 0.41 3D-GS∗ 28.79 0.911 0.241 3.39 Ours 30.45 0.926 0.243 0.26

Playroom

EAGLES 29.35 0.900 0.240 1.57 Mini-Splatting 29.36 0.903 0.260 0.38 3D-GS∗ 30.31 0.913 0.241 3.08 Ours 29.29 0.906 0.259 0.39

Johnson

EAGLES 29.86 0.910 0.250 1.19 Mini-Splatting 29.99 0.909 0.255 0.40 3D-GS∗ 29.55 0.912 0.241 3.24 Ours 29.87 0.916 0.251 0.33

Average

Figure 6. Quantitative per-scene breakdown results on Deep Blending dataset.

Scene Method PSNR SSIM LPIPS Num.(M)

EAGLES 25.04 0.750 0.240 2.26 Mini-Splatting 25.21 0.760 0.246 0.60 3D-GS∗ 25.03 0.740 0.241 5.67 Ours 24.97 0.777 0.233 0.59

Bicycle

EAGLES 31.32 0.940 0.190 0.64 Mini-Splatting 31.41 0.940 0.182 0.33 3D-GS∗ 31.99 0.960 0.170 1.64 Ours 32.79 0.946 0.179 0.31

Bonsai

EAGLES 28.40 0.900 0.200 0.56 Mini-Splatting 28.32 0.913 0.181 0.36 3D-GS∗ 28.89 0.920 0.190 1.58 Ours 29.55 0.914 0.181 0.35

Counter

EAGLES 21.29 0.58 0.370 1.33 Mini-Splatting 21.31 0.614 0.334 0.62 3D-GS∗ 21.30 0.600 0.359 3.67 Ours 21.45 0.617 0.313 0.62

Flowers

EAGLES 26.91 0.840 0.150 1.65 Mini-Splatting 26.67 0.844 0.153 0.69 3D-GS∗ 27.32 0.870 0.125 5.92 Ours 27.58 0.871 0.108 0.68

Garden

EAGLES 30.77 0.930 0.130 1.00 Mini-Splatting 31.24 0.924 0.123 0.38 3D-GS∗ 31.43 0.930 0.120 2.01 Ours 31.65 0.932 0.117 0.37

Kitchen

EAGLES 31.47 0.920 0.200 0.67 Mini-Splatting 31.21 0.920 0.191 0.32 3D-GS∗ 31.59 0.920 0.200 1.99 Ours 31.54 0.927 0.193 0.31

Room

EAGLES 26.78 0.770 0.240 2.22 Mini-Splatting 27.32 0.804 0.215 0.61 3D-GS∗ 26.53 0.770 0.240 4.68 Ours 27.73 0.811 0.193 0.60

Stump

EAGLES 22.69 0.640 0.340 1.60 Mini-Splatting 22.58 0.656 0.331 0.63 3D-GS∗ 22.43 0.660 0.325 3.67 Ours 22.98 0.645 0.314 0.62

Treehill

EAGLES 27.23 0.810 0.240 1.33 Mini-Splatting 27.25 0.820 0.217 0.50 3D-GS∗ 27.39 0.819 0.219 3.43 Ours 27.81 0.827 0.202 0.49

Average

- Table 7. Quantitative per-scene breakdown results on MiPNeRF360 dataset.

Scene Method PSNR↑ SSIM↑ LPIPS↓ #G(106)↑

Train

EAGLES 21.65 0.800 0.240 0.46 Mini-Splatting 21.28 0.801 0.238 0.29 3D-GS∗ 21.94 0.810 0.200 1.11 Ours 22.14 0.812 0.207 0.23

Truck

EAGLES 25.09 0.870 0.160 0.83 Mini-Splatting 25.13 0.871 0.166 0.35 3D-GS∗ 25.31 0.880 0.150 2.54 Ours 25.37 0.878 0.150 0.29

Average

EAGLES 23.37 0.835 0.200 0.64 Mini-Splatting 23.21 0.836 0.203 0.32 3D-GS∗ 23.62 0.845 0.175 1.83 Ours 23.76 0.845 0.179 0.25

- Table 8. Quantitative per-scene breakdown results on Tanks&Temples dataset.

[Figure 40]

[Figure 41]

GT

Ours

| |
|---|

| |
|---|

[Figure 42]

[Figure 43]

w/o 𝐺𝐺𝑑𝑑

w/o 𝐺𝐺𝑝𝑝

| |
|---|

| |
|---|

- Figure 5. Visual comparison with different teacher models. Without the guidance of diverse teacher models, the rendering quality of the student 3DGS model gradually deteriorates.

