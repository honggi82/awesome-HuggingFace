## Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction

# arXiv:2412.04887v2[cs.CV]2Aug2025

Jixuan Fan1,2,∗, Wanhua Li3,∗, Yifei Han1,2, Tianru Dai1,2, Yansong Tang1,2,

1Tsinghua Shenzhen International Graduate School 2Tsinghua University 3Harvard University

fjx23@mails.tsinghua.edu.cn, wanhua@seas.harvard.edu, hyf23@mails.tsinghua.edu.cn dtr24@mails.tsinghua.edu.cn, tang.yansong@sz.tsinghua.edu.cn

(a) Mega-NeRF (b) 3DGS

(c) CityGaussian (e) Momentum-GS (Ours) (f) Ground Truth

(d) DOGS

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

|[Figure 6]| |
|---|---|
| | |
|[Figure 7]<br><br>|[Figure 8]|
|---|
<br><br>| |
|---|
| |

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Figure 1. Comparison of reconstruction results on the Rubble [55] dataset. Our Momentum-GS reconstructs finer details, such as the clear structure of the vehicle in the zoomed-in view. Additionally, our method produces smoother transitions across blocks, demonstrating better consistency and avoiding the noticeable lighting discrepancies observed in other Gaussian-based methods.

#### Abstract

3D Gaussian Splatting has demonstrated notable success in large-scale scene reconstruction, but challenges persist due to high training memory consumption and storage overhead. Hybrid representations that integrate implicit and explicit features offer a way to mitigate these limitations. However, when applied in parallelized block-wise training, two critical issues arise since reconstruction accuracy deteriorates due to reduced data diversity when training each block independently, and parallel training restricts the number of divided blocks to the available number of GPUs. To address these issues, we propose MomentumGS, a novel approach that leverages momentum-based selfdistillation to promote consistency and accuracy across the blocks while decoupling the number of blocks from the physical GPU count. Our method maintains a teacher Gaussian decoder updated with momentum, ensuring a stable reference during training. This teacher provides each block with global guidance in a self-distillation manner, promot-

∗ Equal contribution. Corresponding authors.

ing spatial consistency in reconstruction. To further ensure consistency across the blocks, we incorporate block weighting, dynamically adjusting each block’s weight according to its reconstruction accuracy. Extensive experiments on large-scale scenes show that our method consistently outperforms existing techniques, achieving a 18.7% improvement in LPIPS over CityGaussian with much fewer divided blocks and establishing a new state of the art. Project page: https://jixuan-fan.github.io/Momentum-GS Page/

#### 1. Introduction

Large-scale 3D scene reconstruction is essential for a wide range of applications, including autonomous driving [25, 42, 53, 62], virtual reality [16, 18], environmental monitoring [32, 61], and aerial surveying [5, 6, 51]. The ability to accurately reconstruct large, complex scenes from collections of images is critical for creating realistic, navigable 3D models and supporting high-quality visualization, analysis, and simulation [8, 19, 22, 34].

3D Gaussian Splatting (3D-GS) [20] has recently gained

- #1 Individual Gauss. Dec.

- #2 Individual Gauss. Dec.

Model

Gaussians

Voxels

Voxels

Model

Gaussians

Shared

- (a) Independent training

Voxels

Voxels

Gaussians

Gaussians

Merged

Model

- (b) Parallel training

Voxels

Voxels

Gaussians

Gaussians

Merged Model

- (c) Momentum self-distillation training (Ours)

Gauss. Dec.

Shared Student Gauss. Dec.

Momentum Teacher Gauss. Dec.

Momentum update

Consistency supervise

Gaussians

Gaussians

- Figure 2. Comparison of three approaches for using hybrid representations to reconstruct large-scale scenes in a divideand-conquer manner. Examples with two blocks: (a) Independent training of each block, resulting in separate models that cannot be merged due to independent Gaussian Decoders, complicating rendering; (b) Parallel training with a shared Gaussian decoder, allowing merged output but limited by GPU count; (c) Our approach with a Momentum Gaussian Decoder, providing global guidance to each block and improving consistency across blocks.

attention for its high reconstruction quality and fast rendering speed, outperforming NeRF-based methods [2, 4, 38]. Building on this foundation, recent methods [9, 21, 29, 33] have further enhanced its performance on large-scale scenes. To handle large environments more efficiently, these approaches often employ a divide-and-conquer strategy that partitions a large scene into multiple independent blocks, allowing for multi-GPU training across these blocks. This method facilitates scalable training for complex, expansive reconstructions. However, representing millions of Gaussians explicitly creates substantial memory and storage demands [33], limiting the scalability of 3D-GS for extensive scenes. Additionally, due to unavoidable factors in large scene capture, such as lighting variations, auto-exposure adjustments, or inaccuracies in camera poses [24], independently training each block often disregards inter-block relationships, leading to inconsistencies across block boundaries. This issue can result in visible transitions, as seen in Figure 1 with methods like CityGaussian [33], where abrupt lighting variations are incorrectly rendered. Addressing these concerns has become a core focus in advancing the field of 3D scene reconstruction.

Hybrid representations [28, 35, 46] have emerged as a

promising approach to address memory and storage limitations by combining implicit and explicit features. To manage the complexity of large scenes, these representations integrate dense voxel grids or anchor-based structures with sparse 3D Gaussian fields. These methods typically use MLP as the Gaussian decoder, enabling the generation of neural Gaussians that achieve high reconstruction accuracy while ensuring efficient inference. The decoded Gaussians adapt dynamically to different viewing angles, distances, and scene details. For instance, in Scaffold-GS [35], during inference, the prediction of neural Gaussians is restricted to anchors within the visible frustum, and trivial Gaussians are filtered out based on opacity using a learned selection process. This approach enables rendering speeds comparable to the original 3D-GS. Additionally, neural Gaussians are generated on-the-fly within the view frustum, allowing each anchor to adaptively predict Gaussians for diverse viewing directions and distances in real time. This adaptive mechanism enhances the robustness of novel view synthesis, delivering high-quality renderings across various perspectives while keeping acceptable computational overhead.

However, applying hybrid representations in parallelized reconstruction for large 3D scenes presents two main challenges. First, training each block independently limits data diversity within each block’s Gaussian decoder, reducing reconstruction quality and producing separate models that cannot be merged due to their independent Gaussian decoders, as illustrated in Figure 2 (a). In contrast, parallel training with a shared Gaussian decoder, as in Figure 2 (b), allows for merging the trained models but constrains scalability, as the number of blocks is limited by the available GPUs. These limitations underscore the need for an approach balancing inter-block consistency and scalability.

To overcome these limitations, we propose MomentumGS, a novel approach that combines the benefits of hybrid representations with a strategy tailored to meet the unique demands of large-scale scene reconstruction. Our method decouples the number of blocks from GPU constraints, allowing flexible scaling of reconstruction tasks. This is achieved by periodically sampling k blocks from a set of n blocks and distributing them across k GPUs. To enhance consistency between blocks, we introduce scene momentum self-distillation, where a teacher Gaussian decoder, updated with momentum, provides consistent global guidance to each block, as depicted in Figure 2 (c). This framework encourages collaborative learning across blocks, ensuring that each block benefits from the broader context of the entire scene. Additionally, we introduce reconstructionguided block weighting, a dynamic mechanism that adjusts the emphasis on each block based on its reconstruction quality. This adaptive weighting enables the shared decoder to prioritize underperforming blocks, enhancing global consistency and preventing convergence to local minima.

To thoroughly evaluate the effectiveness of the proposed method, we conduct extensive experiments on five challenging large-scale scenes [27, 30, 55], including Building, Rubble, Residence, Sci-Art, and MatrixCity. Our MomentumGS achieves substantial improvements, demonstrating a 18.7% gain in LPIPS over CityGaussian [33] while utilizing much fewer divided blocks.

In summary, our contributions are:

- 1. We introduce scene momentum self-distillation to enhances Gaussian decoder performance and decouples the number of divided blocks from the number of GPUs, enabling scalable parallel training.
- 2. Our approach incorporates reconstruction-guided block weighting, dynamically adjusting block emphasis based on reconstruction quality to ensure focused improvement on weaker blocks, enhancing overall consistency.
- 3. Our approach, Momentum-GS, achieves better reconstruction quality than state-of-the-art methods, highlighting the strong potential of hybrid representations for large-scale scene reconstruction.

#### 2. Related work

Neural Rendering. Neural Radiance Fields (NeRF) [38] have pioneered a breakthrough in novel view synthesis by representing a 3D scene as a continuous volumetric function, where each point along an emitted ray is sampled to produce color and density values. Numerous extensions [2– 4, 36, 39, 41, 43, 45, 50, 54, 59] have been developed to improve various aspects of NeRF, including its efficiency and scalability. However, NeRFs require intensive sampling along rays for accurate results, leading to high computational costs and prolonged training and inference times. 3D Gaussian Splatting [20] has emerged as a promising alternative, leveraging Gaussian splats for efficient scene representation. Compared to NeRFs, 3DGS significantly reduces sampling requirements while maintaining high fidelity. It has been widely used for many applications [10, 26, 44] due to the speed advantage. Another approach, hybrid representation, combines explicit and implicit elements to benefit from the strengths of both [28, 40, 46, 49, 56]. Often constructed on dense, uniform voxel grids, hybrid representations leverage a mix of methods to improve scene reconstruction. For instance, K-Planes [15] uses planar factorization to represent multi-dimensional scenes, supporting efficient memory use and applying priors like temporal smoothness. Plenoxels [14] adopts a sparse 3D grid with spherical harmonics, bypassing neural networks to directly optimize photorealistic view synthesis from images, achieving significant speedups over traditional radiance fields. Scaffold-GS [35] builds on 3D Gaussian Splatting by using anchor points to distribute local 3D Gaussians and predict their attributes dynamically based on viewing direction and distance. These hybrid approaches showcase the advantages

of combining explicit and implicit elements for scalable, efficient scene reconstruction.

Large Scene Reconstruction. Large-scale scene reconstruction has a long history, with traditional methods often relying on Structure-from-Motion (SfM) [1, 52] to estimate camera poses and create a sparse point cloud from image collections. Subsequent methods, such as MultiView Stereo (MVS), expanded on this foundation to produce denser reconstructions, advancing the capability of photogrammetry systems to handle large scenes. With the advent of Neural Radiance Fields (NeRF) [38], a shift toward neural representations for photo-realistic view synthesis has enabled more detailed scene reconstructions. Many NeRF-based approaches [24, 37, 53, 55, 60, 66], use a similar divide-and-conquer approach, representing each block independently to facilitate scalable reconstruction. However, these methods still face challenges in rendering speed and consistency across scene segments. Recently, 3D Gaussian Splatting [20] has emerged as a promising alternative, offering real-time rendering with high visual fidelity. Numerous methods extend 3DGS to large-scale scenes by enhancing its scalability and efficiency [7, 11– 13, 17, 23, 31, 47, 57, 63]. Some methods [9, 21, 29, 33, 64] partition these large scenes into independent blocks for parallel training, allowing for efficient processing and reconstruction . VastGaussian and CityGaussian, by employing a divide-and-conquer approach to reconstruct large-scale scenes, effectively ensure training convergence, though they lack cross-block interaction, which may limit consistency. DOGS introduces a distributed training method that accelerates 3DGS through scene decomposition and ADMM, while not focusing on optimizing the Gaussian representation for large-scale scenes. These recent 3DGS-based methods demonstrate the potential of 3D Gaussian representations for scalable, high-quality large scene reconstruction, though challenges remain in achieving seamless transitions and efficient memory usage.

#### 3. Methods

Overview. Hybrid representations have demonstrated success in small, object-centric scenes. However, when applied to parallel training in a divide-and-conquer manner for larger environments, they encounter a fundamental dilemma. In this paper, we leverage hybrid representations for large-scale scene reconstruction, harnessing their high reconstruction capability while effectively decoupling the number of blocks from the physical GPU count. Section 3.1 introduces the essential foundations of 3DGS. Section 3.2 then explores how Scene Momentum Self-Distillation effectively addresses the challenges of scaling hybrid representations to large scenes. Lastly, Sec. 3.3 presents the

Grid-based Blocks

###### Shared Online Gaussian Decoder

###### Shared Online Gaussian Decoder

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Sparse Voxels

ImagesGT

- GPU0

GPU3

GPU2

- GPU1

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

(Divided into 8 blocks)

###### ℒ𝐫𝐞𝐜𝐨𝐧𝐬 … … …

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

Color Decoder

Color Decoder

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

c

Reconstruction-guided

ℒℒ𝐜𝐨𝐧𝐬𝐢𝐬𝐭𝐞𝐧𝐜𝐲𝐫𝐞𝐜𝐨𝐧𝐬

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

BlockWeighting

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

Splat

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

…

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

Covariance Decoder

Covariance Decoder

###### Sample 4 blocks

GPU0 GPU1 GPU2 GPU3

Gaussians

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Online

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

[Figure 193]

[Figure 194]

[Figure 195]

Opacity Decoder

Opacity Decoder

ℒ𝐜𝐨𝐧𝐬𝐢𝐬𝐭𝐞𝐧𝐜𝐲 … … …

Momentum

Gaussians

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

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

⊗(1−𝑚)

⊗(1−𝑚)

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

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

…

…

Momentum Gaussian Decoder

⊗𝑚

⊗𝑚

Iteration: t Iteration: t+1

- Figure 3. Overview of the proposed Momentum-GS. Our method begins by dividing the scene into multiple blocks (left), periodically sampling a subset of blocks (e.g., 4 blocks) and assigning them to available GPUs for parallel processing. The momentum Gaussian decoder provides stable global guidance to each block, ensuring consistency across blocks. To align the online Gaussians with the momentum Gaussian decoder, a consistency loss is applied. During splatting, predicted images are compared with ground truth images, and the resulting reconstruction loss is used to update the shared online Gaussian decoder. Additionally, reconstruction-guided block weighting dynamically adjusts the emphasis on each block, prioritizing underperforming blocks to enhance overall scene consistency.

Reconstruction-guided Block Weighting strategy, which enhances global scene consistency by dynamically adjusting each block’s weight based on its reconstruction quality.

##### 3.1. Preliminaries

3DGS offers an efficient solution for accurate scene reconstruction by leveraging the differentiable properties of Gaussian representations along with tile-based rendering. It models each 3D scene point as an anisotropic Gaussian, allowing for streamlined rendering through projection and blending without the computational overhead of dense ray marching typical in traditional volumetric methods.

Each 3D point is represented as a Gaussian function centered at µ ∈ R3, where x is the spatial position, µ is the center, and Σ defines the Gaussian’s shape and orientation:

- 1

- 2(x−µ)⊤Σ−1(x−µ). (1)

G(x) = e−

Rendering projects each 3D Gaussian onto the 2D image plane, resulting in a 2D Gaussian G′(x′), where x′ represents a pixel. The projected Gaussian contributes to pixel color via alpha blending:

C(x′) =

i∈N

- i−1
- j=1

(1 − σj), (2)

ciσi

where N is the set of Gaussians affecting x′, ci is the color in view-dependent spherical harmonics form, and σi = αiG′i(x′) is the opacity with αi as a learnable parameter.

The training of Gaussians uses differentiable rendering to refine Gaussian parameters, starting from an initial point cloud. Gaussians are optimized based on image reconstruction error, with operations like cloning, densifying, and pruning to improve coverage and accuracy. For large scenes, the high Gaussian count presents memory and computational challenges, managed by controlling the active Gaussians during rendering.

##### 3.2. Scene-Aware Momentum Self-Distillation

Hybrid representations face a fundamental challenge when applied to parallel training in a divide-and-conquer approach. Specifically, the limitation of GPU availability restricts the number of blocks that can be processed simultaneously, reducing scalability, while the need for data diversity to maintain the Gaussian decoder’s predictive accuracy remains critical. To address these challenges, we propose Scene Momentum Self-Distillation, a method that both decouples the block count from GPU limitations and enhances the Gaussian decoder’s robustness through improved data diversity. Our method ensures that the Gaussian decoder benefits from a broader range of data, enabling more accurate and consistent predictions across large scenes.

In our approach, we train each block simultaneously in parallel, with all blocks sharing a single Gaussian decoder. During each forward pass, each block randomly selects a viewpoint from its assigned data and uses the shared Gaussian decoder to predict the Gaussian parameters accurately. These predicted parameters are then used to render the cor-

responding image, which is compared to the ground truth to calculate the reconstruction loss. We optimize the learnable parameters using a loss function that combines the L1 loss on rendered pixel colors with an SSIM [58] term LSSIM, aiming to improve structural similarity:

Lrecons = L1 + λSSIMLSSIM, (3)

where λSSIM is a weighting factor that balances the contributions of the L1 and SSIM terms. The gradients from each block are accumulated into the shared Gaussian decoder, allowing it to learn from the full range of scene information.

By adopting a sequential training strategy, our method circumvents the GPU-bound constraint on block count. Each GPU handles one block at a time, with periodic switching to ensure coverage of all blocks. This design decouples block quantity from hardware limitations, thereby supporting scalability as scene complexity increases.

To maintain coherence across staggered training blocks and enhance global consistency, we introduce a momentumbased teacher Gaussian decoder Dt alongside a shared student Gaussian decoder Ds. The Gaussian decoder D dynamically predicts Gaussian attributes based on viewing positions. Specifically, given the anchor feature F ∈ RN×32, viewing distance δ ∈ RN×3, and viewing direction d ∈ RN×1, the decoder outputs Gaussian parameters including color, opacity, rotation, and scale. To ensure computational efficiency, the decoder is implemented as a twolayer MLP. Let B denote the index of each parallel training block, and let θt and θs represent the parameters of the teacher and student Gaussian decoders, respectively. We employ a self-supervised approach to stabilize the teacher Gaussian decoder Dt through momentum-based parameter updates, thus mitigating inconsistencies arising from staggered training. Consequently, the teacher decoder provides a stable global reference, guiding the student decoder via a consistency loss computed between their outputs.

More formally, the parameters θt of the teacher Gaussian decoder are updated using a momentum-based formula that ensures temporal stability:

θt ← m · θt + (1 − m) · θs, (4)

where m is the momentum coefficient, set to 0.9 to balance stability and update speed. If m is too close to 1, the decoder updates too slowly, hindering reconstruction efficiency, while a smaller m may lead to instability due to excessive fluctuations in the teacher decoder. This momentum-based update ensures that the teacher Gaussian decoder evolves smoothly, providing stable and consistent guidance to the student decoder across all blocks. For each block, Gaussian parameters are predicted by both the teacher and student decoders, with a consistency loss applied to align the student decoder with the global guidance

from the teacher. This approach leverages increased data diversity while decoupling the number of blocks from the GPU count, allowing scalability to arbitrarily large scenes.

The consistency loss is computed as the mean squared error between the predictions of the teacher and student Gaussian decoders for each block B:

Lconsistency = ∥Dm(fb,vb;θt) − Do(fb,vb;θs)∥2, (5)

where fb represents the anchor feature and vb the relative viewing direction for each sample within block B. This loss encourages the student decoder Do to progressively align with the stable global guidance provided by the teacher decoder Dm, promoting spatial consistency across different blocks throughout the reconstruction process.

Thus, the total loss function is defined as:

L = L1 + λSSIMLSSIM + λconsistencyLconsistency, (6)

where λconsistency is a weighting factor that balances the impact of the consistency loss relative to the reconstruction loss. This combined loss ensures that the model not only reconstructs the scene accurately but also maintains global spatial coherence across blocks.

##### 3.3. Reconstruction-guided Block Weighting

In order to balance training progress across blocks and mitigate issues arising from uneven initial scene partitioning, we introduce Reconstruction-guided Block Weighting. This method dynamically adjusts weights based on each block’s reconstruction quality, enhancing consistency by giving priority to blocks with lower reconstruction accuracy.

To monitor and adjust the reconstruction performance of each block, we maintain a table that tracks key reconstruction metrics, specifically PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index). These metrics provide quantitative measures of reconstruction quality, with higher values indicating better visual fidelity. Block PSNR is defined as the average PSNR of every image within a block. Block SSIM is calculated similarly. To ensure that these metrics reflect stable performance across training iterations, we update them using a momentum-based approach, which smooths fluctuations and provides a more reliable indication of each block’s progress.

Using these momentum-smoothed metrics, we identify the block with the highest reconstruction performance, labeling its PSNR and SSIM values as PSNRmax and SSIMmax, respectively. These reference values serve as benchmarks for evaluating the relative accuracy of each block. For every block in the scene, we calculate deviations δp and δs to quantify how closely its reconstruction aligns with the highest-performing block. Specifically, the PSNR deviation δp is obtained by subtracting the current block’s PSNR from PSNRmax. δs is derived similarly.

With these deviations calculated, we assign each block a weight wi that reflects its relative reconstruction performance. The weight wi is constructed to resemble a Gaussian distribution, placing greater emphasis on blocks with larger deviations from the best-performing block. By prioritizing blocks with lower reconstruction accuracy, this approach directs the model’s attention to underperforming blocks, helping to improve overall consistency across the scene. Additionally, wi is capped within a range slightly above one, which prevents excessively high adjustments, ensuring stable training dynamics and avoiding overpenalization of blocks with moderate deviations.

δp2 + λ · δs2 −2σ2

, (7)

wi = 2 − exp

This design guiding the Gaussian decoder to focus on the global scene rather than converging on blocks with locally high-quality reconstructions. Consequently improves consistency across all blocks, ultimately enhancing the overall scene reconstruction quality.

#### 4. Experiments

##### 4.1. Experimental Setup

Dataset and Metrics. We conducted experiments on six large-scale scenes across three aerial drone-captured datasets: Building and Rubble from the Mill19 dataset [55], Campus, Residence, and Sci-Art from the UrbanScene3D dataset [30], and Small City from the MatrixCity dataset [27]. Each dataset contains thousands of highresolution images, with the MatrixCity scene notably covering an extensive area of 2.7 square kilometers. Following previous methods [9, 29, 33, 55], we downsampled both training and test images by a factor of 4 for all scenes except MatrixCity, for which we resized the image width to 1,600 pixels. We evaluated reconstruction accuracy using PSNR, SSIM [58], and LPIPS [65], and additionally reported allocated memory and rendering framerate during evaluation to compare rendering performance.

Implementations and Compared methods We use the sparse point cloud from COLMAP [48] as our initial input. Each sparse voxel is initialized using a corresponding point from the point cloud. Following previous methods [29, 33], each block was optimized for 60,000 iterations. To ensure fair comparisons, we use the same initial point cloud and scene partitioning strategy as CityGaussian, but with significantly fewer blocks. Specifically, we divided all scenes into 8 blocks. Additionally, we applied the same color correction method as DOGS [9] when computing metrics. For CityGaussian, we used the checkpoints released by the authors and applied the same color correction method for evaluation. We compared our method against Mega-NeRF [55],

Switch-NeRF [37], 3D-GS [20], VastGaussian [29], CityGaussian [33], and DOGS [9]. All experiments are conducted on the Nvidia RTX 3090 GPUs with 24 GB memory.

##### 4.2. Results Analysis

Quantitative Results. In Table 1 and Table 2, we present the quantitative evaluation results across six large-scale scenes. Our proposed Momentum-GS consistently achieves the best overall performance, substantially outperforming other approaches. These results highlight the capability of our Momentum-GS in preserving fine details and delivering high-quality renderings. Notably, NeRF-based methods yield higher PSNR scores on the Sci-Art dataset. This phenomenon likely arises due to the inherent blurriness present in the Sci-Art source images, perhaps due to out-offocus capture conditions. Since NeRF-based methods typically generate smoother and blurred reconstructions, their outputs naturally align better with these blurred groundtruth images, resulting in elevated PSNR scores. However, when considering SSIM and LPIPS metrics, Gaussianbased methods, typically our Momentum-GS, significantly outperform NeRF-based approaches in perceptual quality.

Visualization Results. In Figure 4 and Figure 5, we provide visual comparisons of reconstruction results across six scenes. Our proposed Momentum-GS consistently produces sharp and realistic images, demonstrating superior detail preservation and excellent visual clarity across all scenes. In contrast, other methods often suffer from noticeable blurring and structural degradation, especially in complex regions. These qualitative results further highlight the effectiveness of Momentum-GS in capturing fine-grained details and maintaining overall rendering quality.

##### 4.3. Ablation Studies

Parallel training vs. Independent training. In Table 5, we demonstrate that parallel training (b) achieves better reconstruction quality compared to independent training (c) when the scene is divided into the same number of blocks. This improvement arises from the increased data diversity available to the shared Gaussian decoder. However, direct parallel training is constrained by the requirement that the number of blocks must match the number of available GPUs. Consequently, independent training can further enhance accuracy by utilizing a larger number of blocks, whereas direct parallel training (b) remains limited by GPU availability. As shown in the Table 5, independent training with eight blocks (d) yields better performance and surpass (b). To overcome this limitation, we introduce scene momentum self-distillation (e), enabling the Gaussian decoder to benefit from increased data diversity while decoupling the number of blocks from the GPU count. Our approach achieves significant accuracy improvements compared to

Scene Building Rubble Campus Residence Sci-Art Metrics PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Mega-NeRF [55] 20.93 0.547 0.504 24.06 0.553 0.516 23.42 0.537 0.636 22.08 0.628 0.489 25.60 0.770 0.390 Switch-NeRF [37] 21.54 0.579 0.474 24.31 0.562 0.496 23.62 0.541 0.616 22.57 0.654 0.457 26.52 0.795 0.360 3D-GS [20] 22.53 0.738 0.214 25.51 0.725 0.316 23.67 0.688 0.347 22.36 0.745 0.247 24.13 0.791 0.262 VastGaussian [29] 21.80 0.728 0.225 25.20 0.742 0.264 23.82 0.695 0.329 21.01 0.699 0.261 22.64 0.761 0.261 CityGaussian [33] 22.70 0.774 0.246 26.45 0.809 0.232 22.80 0.662 0.437 23.35 0.822 0.211 24.49 0.843 0.232 DOGS [9] 22.73 0.759 0.204 25.78 0.765 0.257 24.01 0.681 0.377 21.94 0.740 0.244 24.42 0.804 0.219

Momentum-GS (Ours) 23.65 0.813 0.194 26.66 0.826 0.200 24.34 0.760 0.290 23.37 0.828 0.196 25.06 0.860 0.204

- Table 1. Quantitative comparison of our Momentum-GS against prior methods across four large-scale scenes. We present metrics for PSNR↑, SSIM↑, and LPIPS↓ on test views. The best and second best scores are highlighted.

[Figure 260]

(a) Mega-NeRF (b) 3DGS (c) CityGaussian

[Figure 261]

[Figure 262]

| |
|---|

RubbleBuilding

|[Figure 263]|
|---|

|[Figure 264]|
|---|

|[Figure 265]|
|---|

| |
|---|

| |
|---|

[Figure 266]

[Figure 267]

[Figure 268]

|[Figure 269]|
|---|

|[Figure 270]|
|---|

|[Figure 271]|
|---|

[Figure 272]

(d) DOGS

|[Figure 273]|
|---|

| |
|---|

[Figure 274]

|[Figure 275]|
|---|

| |
|---|

[Figure 276]

(f) Ground Truth

|[Figure 277]|
|---|

| |
|---|

[Figure 278]

|[Figure 279]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 280]

(e) Momentum-GS (Ours)

|[Figure 281]|
|---|

| |
|---|

[Figure 282]

Sci-Art

[Figure 283]

[Figure 284]

[Figure 285]

|[Figure 286]|
|---|

|[Figure 287]|
|---|

|[Figure 288]|
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 289]

|[Figure 290]|
|---|

| |
|---|

[Figure 291]

|[Figure 292]|
|---|

| |
|---|

[Figure 293]

|[Figure 294]|
|---|

| |
|---|

|[Figure 295]|
|---|

| |
|---|

[Figure 296]

[Figure 297]

[Figure 298]

|[Figure 299]|
|---|

|[Figure 300]|
|---|

|[Figure 301]|
|---|

[Figure 302]

|[Figure 303]|
|---|

[Figure 304]

|[Figure 305]|
|---|

| |
|---|

[Figure 306]

|[Figure 307]|
|---|

Campus

[Figure 308]

[Figure 309]

[Figure 310]

| |
|---|

|[Figure 311]|
|---|

|[Figure 312]|
|---|

|[Figure 313]|
|---|

| |
|---|

| |
|---|

[Figure 314]

|[Figure 315]|
|---|

| |
|---|

[Figure 316]

|[Figure 317]|
|---|

| |
|---|

[Figure 318]

|[Figure 319]|
|---|

| |
|---|

Residence

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 4. Qualitative comparisons of our Momentum-GS with previous methods on five large-scale scenes. Red insets highlight areas with notable visual differences. Our approach (e) captures finer details and more accurately represents textures, producing visual reconstructions closer to the ground truth (f). In contrast, previous methods exhibit noticeable artifacts, blur, or inconsistencies in these regions.

Method PSNR ↑ SSIM ↑ LPIPS ↓

3D-GS [20] 27.36 0.818 0.237 VastGaussian [9] 28.33 0.835 0.220 CityGaussian [33] 28.61 0.868 0.205 DOGS [9] 28.58 0.847 0.219

Momentum-GS (Ours) 29.11 0.881 0.180

- Table 2. Quantitative comparison on the extremely large-scale urban scene, MatrixCity. We report PSNR↑, SSIM↑, and LPIPS↓ on test views, with the best results highlighted.

Method 3D-GS VastGaussian CityGaussian DOGS Momentum-GS (Ours) FPS ↑ 45.57 40.04 26.10 48.34 59.91 Mem ↓ 6.31 6.99 14.68 5.82 4.62

Table 3. We report the allocated memory (in GB) and rendering framerate (in FPS) measured during evaluation on the extremely large scene MatrixCity, with the best results highlighted.

Block weighting. In Table 4, we evaluate different methods for measuring the reconstruction quality of each block. Results show that combining PSNR and SSIM yields higher accuracy than using either alone.

independently training eight blocks. Moreover, incorporating reconstruction-guided block weighting (denoted as ”(f) Full”) further enhances the overall reconstruction quality.

Scalability. We evaluated our method on various numbers of divided blocks, keeping the GPU count constant at four.

(a) 3DGS (b) CityGaussian (c) Momentum-GS (Ours) (d) Ground Truth

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

|[Figure 324]|
|---|

|[Figure 325]|
|---|

|[Figure 326]|
|---|

|[Figure 327]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

|[Figure 332]|
|---|

|[Figure 333]|
|---|

|[Figure 334]|
|---|

|[Figure 335]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 340]|
|---|

|[Figure 341]|
|---|

|[Figure 342]|
|---|

|[Figure 343]|
|---|

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

|[Figure 348]|
|---|

|[Figure 349]|
|---|

|[Figure 350]|
|---|

|[Figure 351]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 5. Qualitative comparisons between our Momentum-GS and other methods on the extremely large-scale urban scene, MatrixCity. Our method (c) demonstrates superior detail preservation in challenging regions, such as building facades and edges, closely resembling the ground truth (d). In contrast, other approaches exhibit noticeable artifacts, blurring, and loss of structural details in these areas.

Models PSNR ↑ SSIM ↑ LPIPS ↓

w/ PSNR 23.49 0.809 0.197 w/ SSIM 23.53 0.806 0.203

Full (PSNR + SSIM) 23.65 0.813 0.194

Table 4. Ablation study on different strategy of measuring the reconstruction quality in block weighting.

Training strategy #Block PSNR ↑ SSIM ↑ LPIPS ↓

- (a) baseline 1 22.25 0.742 0.272

- (b) w/ Parallel training 4 23.10 0.790 0.221

- (c) w/ Independent training 4 22.85 0.781 0.229

- (d) w/ Independent training 8 23.23 0.796 0.211

- (e) w/ momentum self-distill. 8 23.56 0.806 0.205

- (f) Full 8 23.65 0.813 0.194 Table 5. Ablation study on different training strategies.

As shown in Table 6, reconstruction quality consistently improves with more blocks, demonstrating our method’s scalability under limited GPU resources.

#### 5. Conclusion

In this paper, we have introduced Momentum-GS, a novel momentum-based self-distillation framework that notably

Method #Block PSNR ↑ SSIM ↑ LPIPS ↓

CityGaussian 32 28.61 0.868 0.205 Momentum-GS (Ours) 4 28.93 0.870 0.203 Momentum-GS (Ours) 8 29.11 0.881 0.180 Momentum-GS (Ours) 16 29.15 0.884 0.172

Table 6. Ablation study on the different number of divided blocks.

enhances 3D Gaussian Splatting for large-scale scene reconstruction. The core of Momentum-GS is a momentumupdated teacher Gaussian decoder, which serves as a stable global reference to guide parallel training blocks, effectively promoting spatial consistency and coherence across the reconstructed scene. We further introduce a reconstruction-guided block weighting mechanism, which dynamically adjusts the emphasis on each block based on reconstruction quality, further improving overall consistency. Our approach leverages hybrid representations, integrating both implicit and explicit features, to enable flexible scaling that decouples the number of blocks from GPU constraints. Experimental results demonstrate the strong capability of hybrid representations and momentum-based selfdistillation for robust, large-scale 3D scene reconstruction.

#### Acknowledgements

This work was supported by Guangdong Natural Science Funds for Distinguished Young Scholar (No. 2025B1515020012) and Shenzhen Science and Technology Program (JCYJ20240813111903006).

#### References

- [1] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski. Building rome in a day. Communications of the ACM, 54

(10):105–112, 2011. 3

- [2] Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In ICCV, pages 5855–5864, 2021. 2, 3
- [3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In CVPR, pages 5470– 5479, 2022.
- [4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased gridbased neural radiance fields. In ICCV, pages 19697–19705,

2023. 2, 3

- [5] Ilker Bozcan and Erdal Kayacan. Au-air: A multi-modal unmanned aerial vehicle dataset for low altitude traffic surveillance. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 8504–8510. IEEE, 2020. 1
- [6] Guikun Chen and Wenguan Wang. A survey on 3d gaussian splatting. arXiv preprint arXiv:2401.03890, 2024. 1
- [7] Junyi Chen, Weicai Ye, Yifan Wang, Danpeng Chen, Di Huang, Wanli Ouyang, Guofeng Zhang, Yu Qiao, and Tong He. Gigags: Scaling up planar-based 3d gaussians for large scene surface reconstruction. arXiv preprint arXiv:2409.06685, 2024. 3
- [8] Timothy Chen, Ola Shorinwa, Joseph Bruno, Javier Yu, Weijia Zeng, Keiko Nagami, Philip Dames, and Mac Schwager. Splat-nav: Safe real-time robot navigation in gaussian splatting maps. arXiv preprint arXiv:2403.02751, 2024. 1
- [9] Yu Chen and Gim Hee Lee. Dogs: Distributed-oriented gaussian splatting for large-scale 3d reconstruction via gaussian consensus. In NeurIPS, 2024. 2, 3, 6, 7
- [10] Zilong Chen, Feng Wang, Yikai Wang, and Huaping Liu. Text-to-3d using gaussian splatting. In CVPR, pages 21401– 21412, 2024. 3
- [11] Jiadi Cui, Junming Cao, Yuhui Zhong, Liao Wang, Fuqiang Zhao, Penghao Wang, Yifan Chen, Zhipeng He, Lan Xu, Yujiao Shi, et al. Letsgo: Large-scale garage modeling and rendering via lidar-assisted gaussian primitives. arXiv preprint arXiv:2404.09748, 2024. 3
- [12] Xiao Cui, Weicai Ye, Yifan Wang, Guofeng Zhang, Wengang Zhou, Tong He, and Houqiang Li. Streetsurfgs: Scalable urban street surface reconstruction with planar-based gaussian splatting. arXiv preprint arXiv:2410.04354, 2024.
- [13] Guofeng Feng, Siyan Chen, Rong Fu, Zimu Liao, Yi Wang, Tao Liu, Zhilin Pei, Hengjie Li, Xingcheng Zhang,

- and Bo Dai. Flashgs: Efficient 3d gaussian splatting for large-scale and high-resolution rendering. arXiv preprint arXiv:2408.07967, 2024. 3
- [14] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In CVPR, pages 5501–5510, 2022. 3
- [15] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In CVPR, pages 12479–12488, 2023. 3
- [16] Jiaming Gu, Minchao Jiang, Hongsheng Li, Xiaoyuan Lu, Guangming Zhu, Syed Afaq Ali Shah, Liang Zhang, and Mohammed Bennamoun. Ue4-nerf: Neural radiance field for real-time rendering of large-scale scene. NeurIPS, 36,

2024. 1

- [17] Changjian Jiang, Ruilan Gao, Kele Shao, Yue Wang, Rong Xiong, and Yu Zhang. Li-gs: Gaussian splatting with lidar incorporated for accurate large-scale reconstruction. arXiv preprint arXiv:2409.12899, 2024. 3
- [18] Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li, Henry Lau, Feng Gao, Yin Yang, et al. Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality. In ACM SIGGRAPH 2024 Conference Papers, pages 1–1, 2024. 1
- [19] Rui Jin, Yuman Gao, Yingjian Wang, Haojian Lu, and Fei Gao. Gs-planner: A gaussian-splatting-based planning framework for active high-fidelity reconstruction. arXiv preprint arXiv:2405.10142, 2024. 1
- [20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. TOG, 42(4), 2023. 1, 3, 6, 7
- [21] Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas, Michael Wimmer, Alexandre Lanvin, and George Drettakis. A hierarchical 3d gaussian representation for real-time rendering of very large datasets. TOG, 43(4):1–15, 2024. 2, 3
- [22] Xiaohan Lei, Min Wang, Wengang Zhou, and Houqiang Li. Gaussnav: Gaussian splatting for visual navigation. arXiv preprint arXiv:2403.11625, 2024. 1
- [23] Bingling Li, Shengyi Chen, Luchao Wang, Kaimin Liao, Sijie Yan, and Yuanjun Xiong. Retinags: Scalable training for dense scene rendering with billion-scale 3d gaussians. arXiv preprint arXiv:2406.11836, 2024. 3
- [24] Ruilong Li, Sanja Fidler, Angjoo Kanazawa, and Francis Williams. NeRF-XL: Scaling nerfs with multiple GPUs. In ECCV, 2024. 2, 3
- [25] Wei Li, CW Pan, Rong Zhang, JP Ren, YX Ma, Jin Fang, FL Yan, QC Geng, XY Huang, HJ Gong, et al. Aads: Augmented autonomous driving simulation using data-driven algorithms. Science robotics, 4(28):eaaw0863, 2019. 1
- [26] Wanhua Li, Renping Zhou, Jiawei Zhou, Yingwei Song, Johannes Herter, Minghan Qin, Gao Huang, and Hanspeter Pfister. 4d langsplat: 4d language gaussian splatting via multimodal large language models. In CVPR, pages 22001– 22011, 2025. 3
- [27] Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhenzhi Wang, Dahua Lin, and Bo Dai. Matrixcity: A large-scale

- city dataset for city-scale neural rendering and beyond. In ICCV, pages 3205–3215, 2023. 3, 6
- [28] Zhuopeng Li, Yilin Zhang, Chenming Wu, Jianke Zhu, and Liangjun Zhang. Ho-gaussian: Hybrid optimization of 3d gaussian splatting for urban scenes. arXiv preprint arXiv:2403.20032, 2024. 2, 3
- [29] Jiaqi Lin, Zhihao Li, Xiao Tang, Jianzhuang Liu, Shiyong Liu, Jiayue Liu, Yangdi Lu, Xiaofei Wu, Songcen Xu, Youliang Yan, and Wenming Yang. Vastgaussian: Vast 3d gaussians for large scene reconstruction. In CVPR, 2024. 2, 3, 6, 7
- [30] Liqiang Lin, Yilin Liu, Yue Hu, Xingguang Yan, Ke Xie, and Hui Huang. Capturing, reconstructing, and simulating: the urbanscene3d dataset. In ECCV, pages 93–109, 2022. 3, 6
- [31] Jinpeng Liu, Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Ying Shan, and Yansong Tang. Novelgs: Consistent novel-view denoising via large gaussian reconstruction model. arXiv preprint arXiv:2411.16779, 2024. 3
- [32] Shuhong Liu, Xiang Chen, Hongming Chen, Quanfeng Xu, and Mingrui Li. Deraings: Gaussian splatting for enhanced scene reconstruction in rainy. arXiv preprint arXiv:2408.11540, 2024. 1
- [33] Yang Liu, He Guan, Chuanchen Luo, Lue Fan, Naiyan Wang, Junran Peng, and Zhaoxiang Zhang. Citygaussian: Real-time high-quality large-scale scene rendering with gaussians. In ECCV, 2024. 2, 3, 6, 7
- [34] Guanxing Lu, Shiyi Zhang, Ziwei Wang, Changliu Liu, Jiwen Lu, and Yansong Tang. Manigaussian: Dynamic gaussian splatting for multi-task robotic manipulation. In ECCV, pages 349–366, 2025. 1
- [35] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In CVPR, pages 20654–20664, 2024. 2, 3
- [36] Ricardo Martin-Brualla, Noha Radwan, Mehdi SM Sajjadi, Jonathan T Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo collections. In CVPR, pages 7210–7219,

2021. 3

- [37] Zhenxing Mi and Dan Xu. Switch-nerf: Learning scene decomposition with mixture of experts for large-scale neural radiance fields. In ICLR, 2023. 3, 6, 7
- [38] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2, 3
- [39] Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul P Srinivasan, and Jonathan T Barron. Nerf in the dark: High dynamic range view synthesis from noisy raw images. In CVPR, pages 16190–16199, 2022. 3
- [40] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. TOG, 41(4):1–15, 2022. 3
- [41] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In CVPR, pages 5480–5490, 2022. 3

- [42] Julian Ost, Fahim Mannan, Nils Thuerey, Julian Knodt, and Felix Heide. Neural scene graphs for dynamic scenes. In CVPR, pages 2856–2865, 2021. 1
- [43] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In CVPR, pages 10318–10327, 2021. 3
- [44] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In CVPR, pages 20051–20060, 2024. 3
- [45] Christian Reiser, Rick Szeliski, Dor Verbin, Pratul Srinivasan, Ben Mildenhall, Andreas Geiger, Jon Barron, and Peter Hedman. Merf: Memory-efficient radiance fields for realtime view synthesis in unbounded scenes. TOG, 42(4):1–12,

2023. 3

- [46] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians. arXiv preprint arXiv:2403.17898, 2024. 2, 3
- [47] Xuanchi Ren, Yifan Lu, Hanxue Liang, Jay Zhangjie Wu, Huan Ling, Mike Chen, Francis Fidler, Sanja annd Williams, and Jiahui Huang. Scube: Instant large-scale scene reconstruction using voxsplats. In NeurIPS, 2024. 3
- [48] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016. 6
- [49] Jiansong Sha, Haoyu Zhang, Yuchen Pan, Guang Kou, and Xiaodong Yi. Nerf-is: Explicit neural radiance fields in semantic space. In Proceedings of the 5th ACM International Conference on Multimedia in Asia, pages 1–7, 2023. 3
- [50] Shuai Shen, Wanhua Li, Xiaoke Huang, Zheng Zhu, Jie Zhou, and Jiwen Lu. Sd-nerf: Towards lifelike talking head animation via spatially-adaptive dual-driven nerfs. IEEE Transactions on Multimedia, 26:3221–3234, 2023. 3
- [51] Surendra Pal Singh, Kamal Jain, and V Ravibabu Mandla. 3d scene reconstruction from video camera for virtual 3d city modeling. American Journal of Engineering Research, 3(1): 140–148, 2014. 1
- [52] Noah Snavely, Steven M. Seitz, and Richard Szeliski. Photo Tourism: Exploring Photo Collections in 3D. Association for Computing Machinery, 2023. 3
- [53] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron, and Henrik Kretzschmar. Block-nerf: Scalable large scene neural view synthesis. In CVPR, pages 8248–8258, 2022. 1, 3
- [54] Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, et al. Nerfstudio: A modular framework for neural radiance field development. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–12, 2023. 3
- [55] Haithem Turki, Deva Ramanan, and Mahadev Satyanarayanan. Mega-nerf: Scalable construction of large-scale nerfs for virtual fly-throughs. In CVPR, pages 12922–12931,

2022. 1, 3, 6, 7

- [56] Haithem Turki, Vasu Agrawal, Samuel Rota Bul`o, Lorenzo Porzi, Peter Kontschieder, Deva Ramanan, Michael

- Zollh¨ofer, and Christian Richardt. Hybridnerf: Efficient neural rendering via adaptive volumetric surfaces. In CVPR, pages 19647–19656, 2024. 3
- [57] Zipeng Wang and Dan Xu. Pygs: Large-scale scene representation with pyramidal 3d gaussian splatting. arXiv preprint arXiv:2405.16829, 2024. 3
- [58] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. TIP, 13(4):600–612, 2004. 5, 6
- [59] Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Humphrey Shi, and Zhangyang Wang. Sinnerf: Training neural radiance fields on complex scenes from a single image. In ECCV, pages 736–753, 2022. 3
- [60] Linning Xu, Yuanbo Xiangli, Sida Peng, Xingang Pan, Nanxuan Zhao, Christian Theobalt, Bo Dai, and Dahua Lin. Grid-guided neural radiance fields for large urban scenes. In CVPR, pages 8296–8306, 2023. 3
- [61] Daniel Yang, John J. Leonard, and Yogesh Girdhar. Seasplat: Representing underwater scenes with 3d gaussian splatting and a physically grounded image formation model. arxiv,

2024. 1

- [62] Zhenpei Yang, Yuning Chai, Dragomir Anguelov, Yin Zhou, Pei Sun, Dumitru Erhan, Sean Rafferty, and Henrik Kretzschmar. Surfelgan: Synthesizing realistic sensor data for autonomous driving. In CVPR, pages 11118–11127, 2020. 1
- [63] Chubin Zhang, Hongliang Song, Yi Wei, Yu Chen, Jiwen Lu, and Yansong Tang. Geolrm: Geometry-aware large reconstruction model for high-quality 3d gaussian generation. arXiv preprint arXiv:2406.15333, 2024. 3
- [64] Hanyue Zhang, Zhiliu Yang, Xinhe Zuo, Yuxin Tong, Ying Long, and Chen Liu. Garfield++: Reinforced gaussian radiance fields for large-scale 3d scene reconstruction. arXiv preprint arXiv:2409.12774, 2024. 3
- [65] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586–595,

2018. 6

- [66] Yuqi Zhang, Guanying Chen, and Shuguang Cui. Efficient large-scale scene representation with a hybrid of highresolution grid and plane features. Pattern Recognition, 158: 111001, 2025. 3

## Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction

### Supplementary Material

#### A. More Details

Scene partition. (1) Criteria: The scene is first equally divided along the x-axis and then along the z-axis, with each block having the same area. Corresponding views are selected based on visibility. (2) Initialization: Each block is initialized from the same point cloud generated by COLMAP, but only the assigned part and overlapping boundary are reconstructed. (3) Views selection: Views at the boundaries are selected based on visibility, and each block reconstructs an extended region to ensure better reconstruction quality at the boundary area.

Motivation of momentum updates. The momentumbased update provides stable, global guidance, allowing each block’s Gaussian decoder to effectively leverage the broader scene context, thereby significantly enhancing reconstruction consistency. As demonstrated in Table 9, using a momentum value of 0.9 outperforms a setting without momentum updates.

#### B. More Ablation Study

Effectiveness of self-distillation. As shown in Table 7, we performed additional experiments to validate the effectiveness of our self-distillation approach: (1) As shown in setting (b), extending parallel training to 8 blocks with 8 GPUs improved the reconstruction quality. (2) Alternating training across blocks every 500 iterations, using 4 GPUs to train 8 blocks in parallel (setting (c)), slightly decreased the reconstruction quality compared with setting (b). (3) Incorporating our momentum-based self-distillation into setting (c) enhanced the reconstruction quality (setting (d)), clearly demonstrating the effectiveness of our proposed method.

Training strategy #Block #GPU PSNR ↑ SSIM ↑ LPIPS ↓

- (a) w/ Parallel training 4 4 23.10 0.790 0.221

- (b) w/ Parallel training 8 8 23.34 0.800 0.210

- (c) w/ Parallel training (alternating) 8 4 23.17 0.797 0.211

- (d) w/ momentum self-distill. 8 4 23.56 0.806 0.205

- (e) Full 8 4 23.65 0.813 0.194 Table 7. Ablation study on different training strategies.

The weight of consistency loss. An ablation study is performed to evaluate the impact of the consistency loss weight λconsistency. As reported in Table 8, the results indicate that model performance remains stable across a wide range of λconsistency values.

Momentum value. We ablated momentum value m and

Scene Building Rubble λconsistency PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

1 23.53 0.808 0.201 26.51 0.816 0.210 10 23.63 0.810 0.200 26.62 0.821 0.204 50 23.65 0.813 0.194 26.66 0.826 0.200 100 23.63 0.810 0.197 26.69 0.829 0.198

Table 8. Ablation study on λconsistency.

Momentum values PSNR ↑ SSIM ↑ LPIPS ↓

0.0 23.44 0.806 0.203 0.5 23.59 0.808 0.201 0.7 23.62 0.810 0.198 0.9 (default) 23.65 0.813 0.194 0.95 23.50 0.806 0.201 0.99 22.06 0.741 0.254

Table 9. Comparison between different momentum values.

Table 9 shows that our model is robust to variations. The reconstruction quality show minimal differences, with the best performance achieved at m=0.9 (our default setting).

#### C. Quantitative Evaluation

VRAM. We report the peak VRAM usage during inference across five large-scale scenes, as shown in Table 10. Despite achieving superior reconstruction quality, our method requires less VRAM compared to the purely 3DGS-based approach. The VRAM usage, measured in MB, highlights the efficiency of our method. Notably, as scene complexity increases (e.g., in MatrixCity), the advantages of our method become even more pronounced.

Scene Building Rubble Residence Sci-Art MatrixCity CityGaussian 8977 5527 6494 2726 14677 Momentum-GS (Ours) 5830 4106 6419 6647 4616

Table 10. Peak VRAM usage (in MB) during inference.

Storage. We report the storage usage across five large-scale scenes, as shown in Table 11. Leveraging our hybrid representation, our method significantly reduces the number of parameters required for storage compared to purely 3DGSbased methods. This reduction is especially notable in larger and more complex scenes, such as MatrixCity, where the storage savings are most substantial. Notably, as scene complexity increases (e.g., in MatrixCity), the advantages

(a) Mega-NeRF (b) 3DGS (c) CityGaussian

(d) DOGS

(e) Momentum-GS (Ours)

(f) Ground Truth

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

RubbleBuilding

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

|[Figure 370]|
|---|

|[Figure 371]|
|---|

|[Figure 372]|
|---|

|[Figure 373]|
|---|

|[Figure 374]|
|---|

|[Figure 375]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

|[Figure 382]|
|---|

|[Figure 383]|
|---|

|[Figure 384]|
|---|

|[Figure 385]|
|---|

|[Figure 386]|
|---|

|[Figure 387]|
|---|

Campus

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

Residence

|[Figure 394]|
|---|

|[Figure 395]|
|---|

|[Figure 396]|
|---|

|[Figure 397]|
|---|

|[Figure 398]|
|---|

|[Figure 399]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

Sci-Art

|[Figure 406]|
|---|

|[Figure 407]|
|---|

|[Figure 408]|
|---|

|[Figure 409]|
|---|

|[Figure 410]|
|---|

|[Figure 411]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 6. Qualitative comparisons of our Momentum-GS and prior methods across four large-scale scenes.

of our method become even more pronounced, demonstrating its effectiveness in handling challenging scenarios. For clarity and consistency, storage usage is reported in GB.

Scene Building Rubble Residence Sci-Art MatrixCity CityGaussian 3.07 2.22 2.49 0.88 5.40 Momentum-GS (Ours) 2.45 (20.2%↓) 1.50 (32.7%↓) 2.00 (19.7%↓) 0.97 2.08 (61.5%↓)

Table 11. Storage usage (in GB).

Number of primitives. We report the number of primitives across five large-scale scenes, as shown in Table 12.

Scene Building Rubble Residence Sci-Art MatrixCity Primitives 8.33M 5.09M 6.79M 3.30M 7.08M

Table 12. Primitives counts for each scene.

Comparison of different implementations of VastGaussian. We further compare our method with the unofficial implementation of VastGaussian in Table 13, which demonstrates improved performance over the results reported in DOGS.

Scene Building Rubble Metrics PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

VastGaussian (DOGS version) 21.80 0.728 0.225 25.20 0.742 0.264 VastGaussian (Unofficial) 22.49 0.742 0.208 25.64 0.760 0.202

Momentum-GS (Ours) 23.65 0.813 0.194 26.66 0.826 0.200

Table 13. Comparison of different implementations of VastGaussian.

#### D. More Visual Comparisons

We provide additional visual comparisons for the Building, Rubble, Residence, and Sci-Art scenes in Figure 6. Our method consistently reconstructs finer details across these scenes. Notably, our approach demonstrates a superior ability to reconstruct luminance, as illustrated by the Sci-Art example shown in Figure 6. While NeRF-based methods are capable of capturing luminance by leveraging neural networks to learn global features such as lighting, they tend to produce blurrier results compared to 3DGS-based methods. This underscores the effectiveness of our hybrid representation, which combines the strengths of both NeRF-based and 3DGS-based approaches.

