## Compact 3D Gaussian Splatting for Static and Dynamic Radiance Fields

Joo Chan Lee, Graduate Student Member, IEEE, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, Member, IEEE, and Eunbyung Park, Member, IEEE

### arXiv:2408.03822v1[cs.CV]7Aug2024

Abstract—Neural Radiance Fields (NeRFs) have demonstrated remarkable potential in capturing complex 3D scenes with high fidelity. However, one persistent challenge that hinders the widespread adoption of NeRFs is the computational bottleneck due to the ray-wise volumetric rendering. On the other hand, 3D Gaussian splatting (3DGS) has recently emerged as an alternative representation that leverages a 3D Gaussian-based representation and introduces an approximated volumetric rendering, achieving very fast rendering speed and promising image quality. Furthermore, subsequent studies have successfully extended 3DGS to dynamic 3D scenes, demonstrating its wide range of applications. However, a significant drawback arises as 3DGS and its following methods entail a substantial number of Gaussians to maintain the high fidelity of the rendered images, which requires a large amount of memory and storage. To address this critical issue, we place a specific emphasis on two key objectives: reducing the number of Gaussian points without sacrificing performance and compressing the Gaussian attributes, such as view-dependent color and covariance. To this end, we propose a learnable mask strategy that significantly reduces the number of Gaussians while preserving high performance. In addition, we propose a compact but effective representation of view-dependent color by employing a grid-based neural field rather than relying on spherical harmonics. Finally, we learn codebooks to compactly represent the geometric and temporal attributes by residual vector quantization. With model compression techniques such as quantization and entropy coding, we consistently show over 25× reduced storage and enhanced rendering speed compared to 3DGS for static scenes, while maintaining the quality of the scene representation. For dynamic scenes, our approach achieves more than 12× storage efficiency and retains a high-quality reconstruction compared to the existing state-of-the-art methods. Our work provides a comprehensive framework for both static and dynamic 3D scene representation, achieving high performance,

Manuscript received August, 2024; This work was supported in part by the Institute of Information and Communications Technology Planning and Evaluation (IITP) funded by the Korea government (MSIT) under Grant RS-2021-II212068 (Artificial Intelligence Innovation Hub), Grant RS-2019II190421 (AI Graduate School Support Program (Sungkyunkwan University)), and Grant No. 2018-0-00207, RS-2018-II180207 (Immersive Media Research Laboratory); and in part by the Culture, Sports, and Tourism R&D Program through the Korea Creative Content Agency funded by the Ministry of Culture, Sports and Tourism in 2024 under Grant RS-2024-00348469 (Research on neural watermark technology for copyright protection of generative AI 3D content). (Corresponding author: Eunbyung Park; Jong Hwan Ko.)

Joo Chan Lee is with the Department of Artificial Intelligence, Sungkyunkwan University, Suwon 16419, South Korea (e-mail: maincold2@skku.edu).

Daniel Rho is with the Department of Computer Science, University of North Carolina at Chapel Hill, NC 27599, USA and KT, Seoul 06763, South Korea (e-mail: dnl03c1@cs.unc.edu).

Xiangyu Sun is with the Department of Electrical and Computer Engineering, Sungkyunkwan University, Suwon 16419, South Korea (e-mail: xiangyusun@g.skku.edu).

Jong Hwan Ko and Eunbyung Park are with the Department of Electronic and Electrical Engineering, Sungkyunkwan University, Suwon 16419, South Korea (e-mail: jhko@skku.edu; epark@skku.edu).

fast training, compactness, and real-time rendering. Our project page is available at https://maincold2.github.io/c3dgs/.

Index Terms—3D Gaussian splatting, neural rendering, novel view synthesis, compact scene representation

I. INTRODUCTION

# T

HE field of neural rendering has witnessed substantial advancements in recent years, driven by the pursuit

of rendering photorealistic 3D scenes from limited input data. Among the pioneering approaches, Neural Radiance Field (NeRF) [2] has gained considerable attention for its remarkable ability to generate high-fidelity images and 3D reconstructions of scenes from only a collection of 2D images. Follow-up research efforts have been dedicated to improving image quality [3], [4], accelerating training and rendering speed [5], [6], [7], [8], [4], [9], reducing memory and storage footprints [10], [11], [12], and expanding its use to dynamic 3D scenes [13], [14], [15], [8], [16].

Despite the massive efforts, one persistent challenge that hinders the widespread adoption of NeRFs is the computational bottleneck due to pixel-wise volumetric rendering. Since it demands dense point sampling along the ray to render a pixel, which requires significant computational resources, NeRFs often fail to achieve real-time rendering on hand-held devices or low-end GPUs. This challenge limits their use in practical scenarios where fast rendering speed is essential, such as various interactive 3D applications.

3D Gaussian splatting (3DGS) [1] has emerged as an alternative representation that can achieve both real-time rendering and high rendering quality. This approach leverages a pointbased representation associated with 3D Gaussian attributes and adopts the rasterization pipeline to render the images. Highly optimized customized cuda kernels to maximize the parallelism and clever algorithmic tricks enable unprecedented rendering speed without compromising the image quality. Not confined to static 3D scenes, several works [17], [18], [19] have demonstrated the applicability of 3DGS also in dynamic 3D scenes. Despite increased temporal dimension, they also achieved fast rendering with high representation quality as well as in static 3D scenes.

However, these 3D Gaussian-based rendering methods share a significant drawback: they require a large amount of memory and storage (e.g., 3DGS often needs over 1GB to represent a static real-world scene). This is primarily due to the necessity of a substantial number of Gaussians to ensure high-quality images (Fig. 1). Moreover, each Gaussian has many attributes,

Gaussian Center Points Rendered Ellipsoids Rendered Result

3DGS

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]<br><br>3DGS (110 fps) 29.06 PSNR, 774 MB|
|---|

|[Figure 5]<br><br>GT|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]<br><br>Ours (155 fps) 29.16 PSNR, 28 MB|
|---|

Ours

- Fig. 1. Our method achieves reduced storage and faster rendering speed while maintaining high-quality renderings of 3DGS [1]. The core idea is to effectively remove the redundant Gaussians that do not significantly contribute to the overall performance (the sparser distribution of Gaussian points and reduced ellipsoid redundancy shown in the figure). We also introduce a more compact representation of Gaussian attributes, resulting in markedly improved storage efficiency and rendering speed.

such as position, scale, rotation, color, and opacity, requiring numerous parameters. For dynamic scenes, the requirement to model temporal movements introduces extra attributes, further increasing the storage overheads.

In this work, we propose a compact 3D Gaussian representation framework that can enhance memory and storage efficiency while attaining high reconstruction quality, fast training speed, and real-time rendering (Fig. 1). Our proposed method is an end-to-end framework, broadly applicable to 3DGS-based methods and primarily improving efficiency in two key areas: the number of Gaussians and the average size of each Gaussian. First, we reduce the total number of Gaussians using learnable masks without sacrificing representation performance. The densification process of 3DGS or 3DGS-based methods, consisting of cloning and splitting Gaussians, increases the number of Gaussians, and this is a crucial component in achieving a high level of detail. However, we observed that the current densification algorithm produces myriads of redundant and insignificant Gaussians, resulting in high memory and storage requirements. We propose a novel learnable masking strategy for Gaussian Splatting in both static and dynamic scenes, designed to identify and eliminate non-essential Gaussians that contribute minimally to the overall rendering quality. With the proposed masking method, we can reduce the number of Gaussians based on their volume and transparency during training, while achieving high performance. In addition to the efficient memory and storage usage, we can achieve faster rendering speed since the computational complexity of the rendering process in 3DGS is highly correlated to the number of Gaussians.

Second, we decrease the average size of each Gaussian by compressing the Gaussian attributes, such as view-dependent color, covariance, and temporal attributes. In 3DGS, each Gaussian has its own attributes, and it does not exploit spatial redundancy, which has been widely utilized for various types of signal compression. For example, neighboring Gaussians may share similar color attributes, and we can reuse similar colors from neighboring Gaussians. Given this motivation, we incorporate a grid-based neural field to efficiently represent

view-dependent colors rather than using per-Gaussian color attributes. When provided with the query Gaussian points, we extract the color attribute from the compact grid representation, avoiding the need to store it for each Gaussian separately. For our initial approach, we opt for a hash-based grid representation (Instant NGP [4]) from among several candidates due to its compactness and fast processing speed. This choice has led to a significant reduction in the spatial complexity of the previous methods.

In addition, 3DGS-based approaches represent a scene with numerous small Gaussians collectively, and each Gaussian primitive is not expected to show high diversity. We found that the majority of Gaussians exhibit similar geometry in both static and dynamic scenes, with limited variation in scale and rotation. Based on this observation, we propose a codebook-based approach for modeling the geometry of Gaussians. It learns to find similar patterns or geometry shared across each scene and only stores the codebook index for each Gaussian, resulting in a very compact representation. Moreover, we found that a compact codebook suffices to represent a highly detailed scene, therefore, the spatial and computational overhead can be insignificant. Similarly, in a dynamic scene, Gaussians may exhibit similar motion trajectories. For instance, groups of moving parts typically follow similar motion patterns, while static areas remain motionless. Therefore, we also learn codebooks to represent temporal attributes of dominant motions.

We have extensively tested our proposed compact Gaussian representation on various datasets. In end-to-end training, our approach consistently showed reduced storage (15 × less than 3DGS for static scenes and 9 × less than STG [19] for dynamic scenes) and enhanced rendering speed, all while maintaining the quality of the scene representation. Furthermore, our method can benefit from simple post-processing techniques such as quantization and entropy coding, consequently achieving over 25× and 12× compression for static and dynamic scenes, respectively.

The earlier version of this research [20] was published at CVPR 2024 as a highlight presentation, focusing on a compact

Gaussian representation for static scenes. This updated version broadens the scope to include dynamic scenes with significant enhancements: 1) We successfully extend the learnable masking approach for Gaussians moving over time, demonstrating its wide applicability. For static scenes, several methods [21], [22] have attempted to estimate and remove non-essential Gaussians after training, yielding promising results. However, removing non-essential Gaussians after training has been more challenging in dynamic scenes, as it requires measuring the importance of each Gaussian over the entire duration. In contrast, our proposed masking strategy simplifies the process by eliminating such complexities, learning the actual rendering impact of each Gaussian across all timestamps during training iterations through gradient descent. 2) To compactly represent the motions of Gaussians, we propose learning representative temporal trajectories by applying the codebook-based approach to temporal attributes. We successfully represent temporal attributes parameter-efficiently and validate that other compact representations for geometry and color are applicable for dynamic scenes as well as for static scenes. 3) Extensive experiments and analysis demonstrate the effectiveness of our approach in dynamic settings. We achieve more than a tenfold increase in parametric efficiency compared to STG, the stateof-the-art method for dynamic scene representation, while maintaining comparable performance.

II. RELATED WORK

- A. Neural Rendering for 3D Scenes

1) Neural Radiance Fields: Neural radiance fields (NeRFs) have significantly expanded the horizons of 3D scene reconstruction and novel view synthesis. NeRF [2] introduced a novel approach to synthesizing novel views of 3D scenes, representing volume features by utilizing Multilayer Perceptrons (MLPs) and introducing volumetric rendering. Since its inception, various works have been proposed to enhance performance in diverse scenarios, such as different resolutions of reconstruction [3], [23], the reduced number of training samples [24], [25], [26], [27], [28], and reconstruction of large realistic scenes [29], [30] and dynamic scenes [13], [14], [15], [31]. However, NeRF’s reliance on MLP has been a bottleneck, particularly causing slow training and inference.

In an effort to address the limitations, grid-based methods emerged as a promising alternative. These approaches using explicit voxel grid structures [6], [32], [33], [34], [35],

- [36] have demonstrated a significant improvement in training speed compared to traditional MLP-based NeRF methods. Nevertheless, despite this advancement, grid-based methods still suffer from relatively slow inference speeds and, more importantly, require large amounts of memory. This has been a substantial hurdle in advancing towards more practical and widely applicable solutions.

Subsequent research efforts have been directed toward the reduction of the memory footprint while maintaining or even enhancing the performance quality by grid factorization [7],

- [37], [8], [38], [39], [40], hash grids [4], [41], grid quantization [11], [42] or pruning [6], [10]. These methods have also been instrumental in the fast training of 3D scene representation, thereby making more efficient use of computational

resources. However, a persistent challenge that remains is the ability to achieve real-time rendering of large-scale scenes. The volumetric sampling inherent in these methods, despite their advancements, still poses a limitation.

2) Point-based Rendering and Radiance Field: To achieve high computational efficiency, Point-NeRF [43] proposed rendering with discrete points rather than continuous fields. NeRF-style volumetric rendering and point-based α-blending fundamentally share the same model for rendering images but differ significantly in their rendering algorithms [1]. NeRFs offer a continuous feature representation of the entire volume as empty or occupied spaces, which necessitate costly volumetric sampling to render a pixel, leading to high computational demands. In contrast, points provide an unstructured, discrete representation of a volume geometry by the creation, destruction, and movement of points, and a pixel is rendered by blending several ordered points overlapping the pixel. By optimizing opacity and positions [44], point-based approaches can achieve fast rendering while avoiding the drawbacks of sampling in a continuous space.

Point-based methods have been widely used in rendering 3D scenes, where the simplest form is point clouds. However, point clouds can lead to visual artifacts such as holes and aliasing. To mitigate this, point-based neural rendering methods have been proposed, processing the points through rasterization-based point splatting and differentiable rasterization [45], [46], [47]. The points were represented by neural features and rendered with CNNs [48], [44], [49]. However, these methods heavily rely on Multi-View Stereo (MVS) for initial geometry, inheriting its limitations, especially in challenging scenarios like areas lacking features, shiny surfaces, or fine structures.

Neural Point Catacaustics [50] addressed the issue of viewdependent effect through the use of an MLP, yet it still depends on MVS geometry for its input. Without the need for MVS, Zhang et al.[51] incorporated Spherical Harmonics (SH) for directional control. However, this method is constrained to managing scenes with only a single object and requires the use of masks during its initialization phase. Recently, 3D Gaussian Splatting (3DGS) [1] proposed using 3D Gaussians as primitives for real-time neural rendering, opening up a new paradigm for 3D scene rendering [52], [53]. 3DGS utilizes highly optimized custom CUDA kernels and ingenious algorithmic approaches to achieve unparalleled rendering speed without sacrificing image quality.

While 3DGS does not require dense sampling for each ray, it does require a substantial number of 3D Gaussians to maintain a high level of quality in the resulting rendered images. Additionally, since each Gaussian consists of several rendering-related attributes like covariance matrices and SH with high degrees, 3DGS demands significant memory and storage resources, e.g., exceeding 1GB for a realistic scene. Our work aims to alleviate this parameter-intensive requirement while preserving high rendering quality, fast training, and real-time rendering.

3) Concurrent works: Several recent works have pursued storage-efficient 3DGS, similar to our objective. These approaches used conventional compression techniques such as

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

N Positions

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

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

|Hash Grids| |
|---|---|
| |View|

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

[Figure 65]

[Figure 66]

[Figure 67]

Direction

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

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

|Tiny MLP| |
|---|---|
| | |

[Figure 86]

Project & Rasterize

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

[Figure 112]

[Figure 113]

Color

[Figure 114]

[Figure 115]

[Figure 116]

Slight distortion

[Figure 117]

[Figure 118]

[Figure 119]

N Gaussians

Masked Gaussians R-VQ for Scale and Rotation View-dependent Color

- Fig. 2. The detailed architecture of our proposed compact 3D Gaussian.

scalar or vector quantization [21], [54], [22], [55], [56] and entropy coding [22], [55], while some of them involve pruning 3D Gaussians based on the significance assessed posttraining [21], [22]. Despite the incorporation of a line of compression techniques, only EAGLES [56] and our method support end-to-end training. Whereas EAGLES controlled the number of Gaussians by just adjusting the densification schedule, resulting in a sub-optimal reduction, our approach is the only work that successfully masks out ineffective Gaussians during training.

constructs a sequence of 3D Gaussians, representing the positions and rotations discretely at each timestamp. To improve efficiency in modeling temporal movements, a line of works used additional architectures such as MLP [72] or grids [17], [73], following the NeRF paradigm. Another line of research learned additional parameters as coefficients for various bases, such as linear [18], polynomial [74], [19], Fourier [75], radial basis [19], or even learned basis [76]. Thanks to the rasterization-based renderings used in 3DGS, most of the aforementioned approaches achieve real-time rendering with promising performance. However, despite the effort at efficient representation, these methods still require substantial memory and storage. Among these, we have selected STG [19] as our baseline method due to its superior performance, to validate the effectiveness of our universally applicable compact representation.

- B. Neural Rendering for Dynamic Scenes

Neural rendering literature has evolved to capture the temporal changes of dynamic scenes, building on the pioneering methods proposed for static environments.

1) NeRF-based Methods: Several extensions of NeRF initially attempted to represent sparser dynamic scenes from a monocular video [13], [14], [34], [31], [57], [58], [59], [60], [61], [62], where a single camera captures the scene from one perspective per timestamp. These methods generally utilize scene flow or depth information to overcome the limited supervision from single viewpoints. Although these approaches have shown promising advancements in non-rigid scenarios, reconstructing large-scale dynamic scenes remains challenging when relying on monocular videos.

III. COMPACT 3D GAUSSIAN SPLATTING

1) Background: In our approach, we build upon the foundation of 3D Gaussian Splatting (3DGS) [1], a point-based representation associated with 3D Gaussian attributes for representing 3D scenes. N Gaussian are parmeterized by center position p ∈ RN×3, opacity o ∈ [0,1]N, 3D scale s ∈ RN+×3, 3D rotation represented as a quaternion r ∈ RN×4, and spherical harmonics (SH) coefficients h ∈ RN×48 (max 3degrees) for view-dependent color. The covariance of each Gaussian Σn ∈ R3×3 is positive semi-definite, calculated as follows,

To reconstruct complex dynamic scenes in practical applications, recent works have utilized synchronized multi-view videos, which offer detailed supervision from various viewpoints and timestamps. DyNeRF [15] integrates NeRF with time-conditioned latent codes, successfully representing more complicated real-world scenes. However, DyNeRF demands extensive training and rendering times, yielding sub-optimal performance. Subsequent studies have explored resolving its inefficiency by focusing on motion between frames [63], efficient sampling [64], or decomposing scene components [65], [16], [66]. In addition, several methods have adopted efficient grid structures, which have succeeded for static scenes, such

##### Σn = R(rn)S(sn)S(sn)TR(rn)T, (1)

where n is the index of the Gaussian, and S(·) : R3+ → R3+×3,R(·) : R4 → R3×3 stand for diagonal scale matrix from 3D scale and rotation matrix from quaternion, respectively.

To render an image, 3D Gaussians are projected into 2D space by viewing transformation W and Jacobian of the affine approximation of the projective transformation J:

- as factorized [67], [8], [68], [69] or hash [70] grids. However, reliance on the dense sampling of those NeRF-based methods still poses challenges in achieving real-time rendering for large real-world scenes.

##### Σ′n = JWΣnWTJT, (2)

2) 3DGS-based Methods: More recently, several works have extended 3DGS to dynamic scenes. Dynamic3D [71]

where Σ′n is the projected 2D covariance. Each pixel color in the image C(·) is then rendered through the alpha composition

A. Gaussian Volume Mask

Densification Interval

3DGS originally densifies Gaussians with large gradients by cloning or splitting. To regulate the increase in the number of Gaussians, opacities are set to a small number at every specific interval, and after some iterations, those with still minimal opacities are removed. Although this opacity-based control effectively eliminates some floaters, we empirically found that a significant number of redundant Gaussians still exist (×2.42 Gaussians show similar performance in Fig. 3). Among them, small-sized Gaussians, due to their minimal volume, have a negligible contribution to the overall rendering quality, often to the point where their effect is essentially imperceptible. In such cases, it becomes highly beneficial to identify and remove such unessential Gaussians.

×2.42 Compact

#Gaussians(M)

3DGS: PSNR 29.87 / Train 19m 22s / 154 fps Ours : PSNR 29.91 / Train 17m 51s / 254 fps

- 0

- 0.5
- 1

1.5

2

- 2.5

0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30

Iteration (K)

- Fig. 3. The number of Gaussians during the training (Bonsai scene). ‘# Gaussians’ denotes the number of Gaussians.

As such, we propose a learnable masking of Gaussians based on their volume as well as transparency. We apply binary masks not only on the opacities but also on the scale attributes that determine the volume geometry of Gaussians. We introduce an additional mask parameter mn ∈ R, based on which we generate a binary mask Mn ∈ {0,1}. As it is not feasible to calculate gradients from binarized masks, we employ the straight-through estimator [78], [10], formulated as:

using colors cn, determined by spherical harmonics under the given view direction, and the final opacity in 2D space αn(·),

N(x)

k−1

(1 − αj(x)), (3)

C(x) =

ckαk(x)

j=1

k=1

- 1

- 2

(x − p′n)TΣ′−n1(x − p′n) , (4)

αn(x) = on exp −

##### Mn = sg(1[σ(mn) > ϵ] − σ(mn)) + σ(mn), (5)

where x is a coordinate of the pixel to be rendered, N(x) is the number of Gaussians around x, the Gaussians are depth-based sorted given the viewing direction, and p′n is the projected Gaussian center position, respectively. 3DGS approximates and accelerates this process by introducing the tile-based parallel rasterization pipeline, achieving real-time rendering. For more details, please refer to the original 3DGS paper [1].

where ϵ is the masking threshold, sg(·) is the stop gradient operator, and 1[·] and σ(·) are indicator and sigmoid function, respectively. By applying the binary mask for the scale and opacity, Eq. 1,4 can be reformulated as follows,

Σˆn = R(rn)S(Mnsn)S(Mnsn)TR(rn)T, (6) αˆn(x) = Mnon exp −

- 1

- 2

−1 n (x − p′n) , (7)

(x − p′n)TΣˆ′

3DGS constructs initial 3D Gaussians derived from the sparse data points obtained by Structure-from-Motion (SfM), such as COLMAP [77]. These Gaussians are cloned, split, pruned, and refined towards enhancing the anisotropic covariance for a precise depiction of the scene. This training process is based on the gradients from the differentiable rendering without unnecessary computation in empty areas, which accelerates training and rendering. However, 3DGS’s high-quality reconstruction comes at the cost of memory and storage requirements, particularly with numerous Gaussians increased during training and their associated attributes.

−1

where Σˆ′

n denotes the projected 2D covariance (Eq. 2) after masking. This method allows for the incorporation of masking effects based on Gaussian volume and transparency in rendering. Considering both aspects together leads to more effective masking compared to considering either aspect alone.

We balance the accurate rendering and the number of Gaussians eliminated during training by adding masking loss Lm as follows,

2) Overall architecture: Our primary objectives are to 1) reduce the number of Gaussians and 2) represent attributes compactly while retaining the original performance. To this end, along with the optimization process, we mask out Gaussians that minimally impact performance, as shown in Fig. 2. For geometry attributes, such as scale and rotation, we propose using a codebook-based method that can fully exploit the limited variations of these attributes. We represent the color attributes using a grid-based neural field rather than storing their large parameters directly per each Gaussian. Finally, a small number of Gaussians with compact attributes are then used for the subsequent rendering steps, including projection and rasterization to render images.

N

1 N

σ(mn). (8)

Lm =

n=1

At every densification step, we eliminate Gaussians based on the binary mask. Furthermore, unlike the original 3DGS, which stops densifying in the middle of the training and retains the number of Gaussians to the end, we consistently mask out throughout the entire training process, reducing unessential Gaussians effectively and ensuring efficient computation with low GPU memory throughout the training phase (Fig. 3). Once training is completed, we remove the masked Gaussians, so the mask parameter or binary mask does not need to be stored.

[Figure 120]

[Figure 121]

[Figure 122]

Residuals 𝑠̂

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

[Figure 123]

[Figure 124]

[Figure 125]

Scale 𝑠

𝑠 − 𝑠̂

𝑠̂

𝑠̂

[Figure 126]

[Figure 127]

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

| | | |
|---|---|---|

…

[Figure 128]

+

[Figure 129]

[Figure 130]

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

[Figure 131]

𝑟 − 𝑟̂

Rotation 𝑟

𝑟̂

𝑟̂

𝑟̂

[Figure 132]

𝑠 𝑟 𝑠̂ 𝑟̂ 𝑠̂ 𝑟̂

[Figure 133]

Stage 𝐿

Input

Stage 1

Stage 2

- Fig. 4. The detailed process of R-VQ to represent the scale and rotation of Gaussians. In the first stage, the scale and rotation vectors are compared to codes in each codebook, with the closest code identified as the result. In the next stage, the residual between the original vector and the first stage’s result is compared with another codebook. This process is repeated up to the final stage, as a result, the selected indices and the codebook from each stage collectively represent the original vector.

- B. Geometry Codebook

- A number of Gaussians collectively construct a single scene,

where similar geometric components can be shared throughout the entire volume. We have observed that the geometrical shapes of most Gaussians are very similar, showing only minor differences in scale and rotation. In addition, a scene is composed of many small Gaussians, and each Gaussian primitive is not expected to exhibit a wide range of diversity. Given this motivation, we propose a codebook learned to represent representative geometric attributes, including scale and rotation, by employing vector quantization (VQ) [79]. As naively applying vector quantization requires a high level of computational complexity and GPU memory [80], we adopt residual vector quantization (R-VQ) [80] that cascades L stages of VQ with codebook size C (Fig. 4), formulated as follows,

rˆnl =

l

j=1

Zj[ijn], l ∈ {1,...,L}, (9)

iln = argmin

k

||Zl[k] − (rn − rˆnl−1)||22, rˆn0 = ⃗0 (10)

where rn ∈ R4 is the input rotation vector, rˆnl ∈ R4 is the output rotation vector after l quantization stages, and n is the index of the Gaussian. Zl ∈ RC×4 is the codebook at the stage l, il ∈ {0,...,C − 1}N is the selected indices of the codebook at the stage l, and Z[i] ∈ R4 represents the vector at index i of the codebook Z.

The objective function for training the codebooks is as follows,

Lr =

1 NC

L

k=1

N

n=1

||sg[rn − rˆnk−1] − Zk[ikn]||22, (11)

where sg[·] is the stop-gradient operator. We use the output from the final stage rˆL (we will omit the superscript L for brevity from now onwards), and the R-VQ process is similarly applied to scale s before masking (we also similarly use the objective function for scale Ls).

- C. Compact View-dependent Color

different colors according to the viewing direction. Instead of using the naive and parameter-inefficient approach, we propose representing the view-dependent color of each Gaussian by exploiting a grid-based neural field. To this end, we contract the unbounded positions p ∈ RN×3 to the bounded range, motivated by Mip-NeRF 360 [23], and compute the 3D view direction d ∈ R3 for each Gaussian based on the camera center point. We exploit hash grids [4] followed by a tiny MLP to represent color. Here, we input positions into the hash grids, and then the resulting feature and the view direction are fed into the MLP. More formally, view-dependent color cn(·) of Gaussian at position pn ∈ R3 can be expressed as,

cn(d) = f(contract(pn),d;θ), (12)

pn ||pn|| ≤ 1 2 − ||p1

(13)

contract(pn) =

pn

||pn|| ||pn|| > 1,

n||

where f(·;θ),contract(·) : R3 → R3 stand for the neural field with parameters θ, and the contraction function, respectively. We represent the 0-degree components of SH (the same number of channels as RGB, but not view-dependent) and then convert them into RGB colors due to the slightly increased performance compared to representing the RGB color directly.

D. Training

Here, we have N Gaussians and their attributes, position pn, opacity on, rotation rˆn, scale sˆn, and view-dependent color cn(·), which are used to render images (Eq. 3). The entire model is trained end-to-end based on the rendering loss Lren, the weighted sum of the L1 and SSIM loss between the GT and rendered images. By adding the loss for masking Lm and geometry codebooks Lr,Ls, the overall loss L is as follows,

##### L = Lren + λmLm + Lr + Ls, (14)

where λm is a hyper-parameter to regularize the number of Gaussians. To avoid heavy computational costs and ensure fast and optimal training, we initialize the learnable codebooks with K-means and apply R-VQ only during the last 1K training iterations. Except for that period, we set Lr,Ls to zero.

Each Gaussian in 3DGS requires 48 of the total 59 parameters to represent SH (max 3 degrees) to model the

Input time coordinate

Time

|HashGrid&MLP|
|---|

|FeatureSplatting|
|---|

[Figure 134]

[Figure 135]

HashGrid&MLP

FeatureSplatting

| |
|---|

|MLP|
|---|

ImpactonRendering

[Figure 136]

[Figure 137]

Position

MLP

[Figure 138]

[Figure 139]

[Figure 140]

Rendered Image

View Time coefficient Direction Time feature

[Figure 141]

Geometry

[Figure 142]

Masking Gaussians with minimal impact over time

[Figure 143]

[Figure 144]

Space-Time Masking

R-VQ View-dependent Color

- Fig. 5. The detailed architecture of our proposed compact Gaussian representation for dynamic scenes.

IV. COMPACT 3D GAUSSIAN SPLATTING FOR DYNAMIC SCENES

where son ∈ [0,1] denotes the time-independent spatial opacity, ξn ∈ R represents a temporal scale that indicates the effective duration for each Gaussian (i.e., the duration in which its temporal opacity is high), and p′n(·),Σ′n(·) are the projected center position and covariance at each timestamp.

Our compact Gaussian representation can be extended to dynamic scenarios. We use STG [19], the state-of-the-art method for dynamic scenes, as our baseline model. STG is designed to learn space-time Gaussian attributes, including its nop-th and nor-th order polynomial coefficients for moving position {un,k ∈ R3}nok=1p and rotation {vn,k ∈ R4}no

STG optimizes a 9-dimensional feature for each Gaussian scn ∈ R9 to represent spatial, view-directional, and temporal colors, with 3 dimensions for each, and construct the timevariant color feature of each Gaussian cn(t) ∈ R9 as follows,

k=1, respectively, and the temporal center and scale µn,ξn ∈ R as well as the static (time-independent) attributes such as position spn, rotation srn, scale sn, opacity son, and color feature scn.

r

##### cn(t) = stack(scn,1:6,(t − µn)scn,7:9), (20)

where scn,1:6 is the extracted column vector from the first to 6-th element of the color feature vector scn and stack(·,·) operator stacks input vectors into a single vector.

STG models the time-conditioned attributes of each Gaussian by introducing a temporal center µn ∈ R, the time step when each Gaussian is most prominent. The motions of each Gaussian are represented by learning polynomial coefficients associated with the position {un,k ∈ R3}nok=1p and rotation {vn,k ∈ R4}no

This feature is splatted into the image space through the projection and rasterization process in Eq. 2-4, and the splatted feature F(·,·) can be formulated as follows,

k=1. At any time t, the position pn(·) ∈ R3 and rotation rn(·) ∈ R4 are defined as follows:

r

N(x,t)

k−1

(1 − αj(x,t)), (21)

F(x,t) =

ck(t)αk(x,t)

nop

un,k(t − µn)k, (15)

pn(t) = spn +

j=1

k=1

where N(x,t) is the number of Gaussians around x at time t, while the Gaussians are depth-based sorted given the viewing direction. Then the splatted feature F(x,t) is split into F(x,t)1:3, F(x,t)4:6, and F(x,t)7:9, which represent spatial, view-directional, and temporal color features, respectively, and the final RGB color C(·,·) at pixel and time coordinates (x,t) can be obtained as follows:

k=1

nor

vn,k(t − µn)k, (16)

rn(t) = srn +

k=1

where spn ∈ R3, srn ∈ R4 are the canonical position and rotation when t = µn, and nop,nor are the maximum polynomial orders for position and rotation (STG sets nop = 3,nor = 1), respectively. STG remains the scale attribute for each Gaussian sn ∈ R3+ constant over time. Therefore, the covariance matrix at time t can be written as,

##### C(x,t) = F(x,t)1:3 + ϕ(F(x,t)4:6,F(x,t)7:9,d), (22)

where d ∈ R3 is the viewing direction and ϕ(·,·,·) is an MLP for the view- and time-dependent color. For more details, please refer to the original paper [19].

##### Σn(t) = R(rn(t))S(sn)S(sn)TR(rn(t))T. (17)

For time-conditioned visibility, STG uses a temporal radial basis function, such that the final projected opacity of each Gaussian αn(·,·) at the pixel and time coordinates (x,t) is formulated as:

A. Space-Time Mask

To eliminate the redundant Gaussians in dynamic scenes, we consider not only the spatial but also the temporal influence of each Gaussian. We estimate both significances simultaneously by extending the masking strategy (Eq. 6,7), where the perGaussian masks are optimized to reflect their impact on rendering quality over time, as shown in Fig. 5. Specifically, we

- 1

- 2

(x − p′n(t))TΣ′−n 1(t)(x − p′n(t)) ,

αn(x,t) = on(t)exp −

(18) on(t) = son exp −ξn|t − µn|2 , (19)

apply the binary mask in Eq. 5 to the time-varying covariance (Eq. 17) and opacity (Eq. 19), reformulated as follows:

Σˆn(t) = R(rn(t))S(Mnsn)S(Mnsn)TR(rn(t))T, (23) oˆn(t) = Mnon exp −ξn|t − µn|2 . (24)

In the context of static scenes, several methods [21], [22] have been suggested to estimate and remove non-essential Gaussians as a post-processing after training, showing promising results. However, applying these techniques to dynamic scenes presents greater challenges, as it requires assessing the effectiveness of each Gaussian over the entire time duration. Our proposed masking strategy, on the other hand, avoids these complexities thus simplifying the process. It learns the actual rendering impact across all timestamps during training iterations through gradient descent.

- B. Compact Attributes

In Sec. III, we present the efficient representation for Gaussian attributes using R-VQ and neural fields, depending on the redundancy and continuity of the attributes. As for static scenes (Eq. 9), we apply R-VQ to time-invariant geometric attributes (from sn,srn to sˆn,srˆ n), exploiting the redundancy of them. In addition, as temporal features exhibit redundancy over time, we also use R-VQ for rotation coefficients vˆn,k and temporal features scˆ n,7:9. However, since the positions require high precision in 3D space and are already compactly represented using polynomial bases, we bypass additional compression for coefficients un,k.

For static color attributes, we similarly exploit the continuity of colors. We use the neural field f(·;θ) : R3 → R6 to represent the spatial and view directional color features scn,1:6 ∈ R6 at each canonical position spn, thus the final color feature cn(t) can be reformulated as,

##### cn(t) = stack(f(contract(spn); θ), (t − µn) ˆscn), (25)

where scˆ n ∈ R3 is the R-VQ-applied temporal color feature (we omitted the subscript 7:9 from scˆ n,7:9 for brevity). Following STG, we splat this feature cn(t) into the image space, and obtain the final color by using the MLP (Eq. 22), as shown in

- Fig. 5. The entire training process resembles that for static scenes

described in Sec. III-D, optimizing rendering loss in conjunction with masking loss and adding further introduced R-VQ losses for temporal coefficients and color. Also, we use the same strategy for efficient R-VQ applications.

V. EXPERIMENT A. Implementation Details

1) Static scenes: We tested our approach on three realworld datasets (Mip-NeRF 360 [23], Tanks&Temples [81], and Deep Blending [82]) and a synthetic dataset (NeRFSynthetic [2]). Following 3DGS, we chose two scenes from Tanks&Temples and Deep Blending. We retained all hyperparameters of 3DGS and trained models for 30K iterations, and we set the codebook size C and the number of stages L of R-VQ for geometry to 64 and 6, respectively. The

neural field for view-dependent color uses hash grids with 2channel features across 16 different resolutions (16 to 4096) and a following 2-layer 64-channel MLP. Due to the different characteristics between the real and synthetic scenes, we adjusted the maximum hash map size and the hyper-parameters for learning the neural field and the mask. For the real scenes, we set the max size of hash maps to 219, the control factor for the number of Gaussians λm to 5e−4, and the learning rate of the mask parameter and the neural fields to 1e−2. The learning rate of the neural fields is decreased at 5K, 15K, and 25K iterations by multiplying a factor of 0.33. For the synthetic scenes, the maximum hash map size and the control factor λm were set to 216 and 4e−3, respectively. The learning rate of the mask parameter and the neural fields were set to 1e−3, where the learning rate of the neural fields was reduced at 25K iterations with a factor of 0.33.

- 2) Dynamic scenes: We trained models for 25K iterations

using two real-world multi-view video datasets (DyNeRF [15] and Technicolor [83]), retaining all other hyper-parameters of STG. We set the codebook size C to 256 and the number of stages L (geometry, temporal attributes) to (4,3) and (5,4) for DyNeRF and Technicolor datasets, respectively. The max hash map sizes were set to 214, 216 for DyNeRF and Technicolor datasets, respectively, and the learning rate of the neural fields is decreased at 3, 6, 9, 12, 18, and 21K iterations by multiplying a factor of 0.33. The other settings for the neural field structure and learning rates remained the same as those we used for real-world static scenes.

- 3) Post-processing: For the end-to-end trained models with

the proposed method (denoted as Ours), we stored the position (including coefficients for position) and scalar attributes (opacity of 3DGS; opacity, temporal center, and temporal scale of STG) with 16-bit precision using half-tensors. Additionally, we implemented straightforward post-processing techniques on the model attributes, a variant we denote as Ours+PP. These post-processing steps include:

- • Applying 8-bit min-max quantization to hash grid parameters and scalar attributes.
- • Pruning hash grid parameters with values below 0.1.
- • Sorting Gaussians in Morton order [22].
- • Applying Huffman encoding [84] on the 8-bit quantized values (hash parameters and scalar attributes) and RVQ indices, and compressing the results using DEFLATE [85].

B. Static Scene Representation

1) Real-world scenes: Table I and Table II show the qualitative results evaluated on real-world scenes. Across datasets, our approach achieves high reconstruction performance comparable to 3DGS while drastically reducing the overall size and accelerating rendering. Especially for the Deep Blending dataset (Table II), our method even outperforms the original 3DGS in terms of visual quality (measured in PSNR and SSIM), achieving state-of-the-art performance with the fastest rendering speed as well as compactness (almost 40% faster rendering and over 15× compactness compared to 3DGS). Comparing our method to 3DGS, the qualitative results in

GT 3DGS Ours

#### Ours+PP

|[Figure 145]|
|---|

|[Figure 146]<br><br>21.83 / 1.4 GB|
|---|

|[Figure 147]<br><br>21.79 / 63 MB|
|---|

|[Figure 148]<br><br>22.04 / 38 MB|
|---|

GardenRoomBicycle

|[Figure 149]|
|---|

|[Figure 150]<br><br>27.09 / 1.3 GB|
|---|

|[Figure 151]<br><br>28.31 / 63 MB|
|---|

|[Figure 152]<br><br>27.98 / 38 MB|
|---|

|[Figure 153]|
|---|

|[Figure 154]<br><br>33.87 / 350 MB|
|---|

|[Figure 155]<br><br>33.44 / 34 MB|
|---|

|[Figure 156]<br><br>33.25 / 14 MB|
|---|

- Fig. 6. Qualitative results for static scenes compared to 3DGS. We present the rendering PSNR and storage on the results.

TABLE I QUANTITATIVE RESULTS OF THE PROPOSED METHOD EVALUATED ON MIP-NERF 360 AND TANKS&TEMPLES DATASETS. WE REPORTED THE NUMBERS OF BASELINES FROM THE ORIGINAL PAPER (DENOTED AS 3DGS), WHICH WERE RUN ON AN NVIDIA A6000 GPU. FOR A FAIR COMPARISON, WE RE-EVALUATE 3DGS WITH THE SAME TRAINING CONFIGURATIONS AS OUR METHOD USING AN NVIDIA A100 GPU (DENOTED AS 3DGS*).

Dataset Mip-NeRF 360 Tanks&Temples Method PSNR SSIM LPIPS Train FPS Storage PSNR SSIM LPIPS Train FPS Storage Plenoxels 23.08 0.626 0.463 25m 49s 6.79 2.1 GB 21.08 0.719 0.379 25m 05s 13.0 2.3 GB INGP-base 25.30 0.671 0.371 05m 37s 11.7 13 MB 21.72 0.723 0.330 05m 26s 17.1 13 MB INGP-big 25.59 0.699 0.331 07m 30s 9.43 48 MB 21.92 0.745 0.305 06m 59s 14.4 48 MB Mip-NeRF 360 27.69 0.792 0.237 48h 0.06 8.6 MB 22.22 0.759 0.257 48h 0.14 8.6 MB 3DGS 27.21 0.815 0.214 41m 33s 134 734 MB 23.14 0.841 0.183 26m 54s 154 411 MB 3DGS* 27.46 0.812 0.222 24m 07s 120 746 MB 23.71 0.845 0.178 13m 51s 160 432 MB Ours 27.08 0.798 0.247 33m 06s 128 48.8 MB 23.32 0.831 0.201 18m 20s 185 39.4 MB Ours+PP 27.03 0.797 0.247 - - 26.2 MB 23.32 0.831 0.202 - - 18.9 MB

TABLE II QUANTITATIVE RESULTS OF THE PROPOSED METHOD EVALUATED ON DEEP BLENDING DATASET. WE RE-EVALUATE 3DGS* UNDER THE SAME CONFIGURATIONS WITH OUR METHOD.

Dataset Deep Blending Method PSNR SSIM LPIPS Train FPS Storage Plenoxels 23.06 0.795 0.510 27m 49s 11.2 2.7 GB INGP-base 23.62 0.797 0.423 06m 31s 3.26 13 MB INGP-big 24.96 0.817 0.390 08m 00s 2.79 48 MB Mip-NeRF 360 29.40 0.901 0.245 48h 0.09 8.6 MB 3DGS 29.41 0.903 0.243 36m 02s 137 676 MB 3DGS* 29.46 0.900 0.247 21m 52s 132 663 MB Ours 29.79 0.901 0.258 27m 33s 181 43.2 MB Ours+PP 29.73 0.900 0.258 - - 21.6 MB

- Fig. 6 further demonstrate our method’s high-quality rendering

TABLE III QUANTITATIVE RESULTS OF THE PROPOSED METHOD EVALUATED ON NERF-SYNTHETIC DATASET. * DENOTES THE REPORTED VALUE IN THE ORIGINAL PAPER.

Dataset NeRF-Synthetic Method PSNR Storage Train FPS 3DGS 33.32* 68.1 MB 6m 14s 359 Ours 33.33 5.55 MB (×0.08) 8m 04s (×1.29) 545 (×1.52) Ours+PP 32.88 2.47 MB (×0.04) - -

with substantially less size.

2) Synthetic scenes: We also evaluate our method on synthetic scenes. As 3DGS has proven its effectiveness in improving visual quality, rendering speed, and training time compared to other baselines, we focus on comparing our method with

GT STG Ours

Ours+PP

|[Figure 157]|
|---|

|[Figure 158]<br><br>29.38 / 297 MB|
|---|

|[Figure 159]<br><br>29.03 / 26.8 MB|
|---|

|[Figure 160]<br><br>29.02 / 19.6 MB|
|---|

FlameSalmonSearSteakTheater

|[Figure 161]|
|---|

|[Figure 162]<br><br>33.81 / 159 MB|
|---|

|[Figure 163]<br><br>33.71 / 19.0 MB|
|---|

|[Figure 164]<br><br>33.57 / 13.0 MB|
|---|

|[Figure 165]|
|---|

|[Figure 166]<br><br>31.07 / 57.0 MB|
|---|

|[Figure 167]<br><br>31.34 / 11.9 MB|
|---|

|[Figure 168]<br><br>31.30 / 8.0 MB|
|---|

- Fig. 7. Qualitative results for dynamic scenes compared to STG. We present the rendering PSNR and storage on the results.

TABLE IV QUANTITATIVE RESULTS OF THE PROPOSED METHOD EVALUATED ON DYNERF DATASET. WE RE-EVALUATE STG* UNDER THE SAME CONFIGURATIONS WITH OUR METHOD.

Dataset DyNeRF Method PSNR SSIM SSIM’ LPIPS FPS Storage NeRFPlayer 30.69 0.932 - 0.111 0.05 5.1 GB K-Planes 31.63 - 0.964 - 0.3 311 MB MixVoxels-L 31.34 - 0.966 0.096 37.7 500 MB Dynamic 3DGS 30.67 0.930 0.962 0.099 460 2.7 GB 4DGS 31.15 - 0.968 0.049 30 90 MB STG-Lite 31.59 0.944 0.968 0.047 310 103 MB STG 32.05 0.946 0.970 0.044 140 200 MB

STG* 31.94 0.948 0.971 0.046 181 197 MB Ours 31.73 0.945 0.969 0.053 186 21.8 MB Ours+PP 31.69 0.945 0.969 0.054 - 15.4 MB

3DGS, highlighting the improvements from our method. As shown in Table III, although our approach requires slightly more training duration, we achieve over 10× compression and 50% faster rendering compared to 3DGS, maintaining highquality reconstruction.

3) Post-processings: With post-processings, our model can be further downsized by over 40 % regardless of the dataset. Consequently, we achieve more than 28× compression from 3DGS (Mip-NeRF 360), while maintaining high performance.

- C. Dynamic Scene Representation

Table IV and Table V show the qualitative results evaluated on DyNeRF and Technicolor datasets. For both datasets, our approach achieves high-performance representation comparable to STG while significantly reducing the storage. Especially for the DyNeRF dataset, our method achieves over 9× compactness compared to STG, even though STG has already been

TABLE V QUANTITATIVE RESULTS OF THE PROPOSED METHOD EVALUATED ON TECHNICOLOR DATASET. WE RE-EVALUATE STG* UNDER THE SAME CONFIGURATIONS WITH OUR METHOD.

Dataset Technicolor Method PSNR SSIM LPIPS FPS Storage/Fr

DyNeRF 31.8 - 0.140 0.02 0.6 MB HyperReel 32.7 0.906 0.109 4.0 1.2 MB STG 33.6 0.920 0.084 86.7 1.1 MB

STG* 33.5 0.920 0.083 105 1.3 MB Ours+PP 33.1 0.910 0.098 116 0.16 MB

designed for compact representation. With post-processings, our model can be further downsized by almost 30 %, consequently, we achieve more than 12× compression from STG, while maintaining high performance. Fig. 7 illustrates the qualitative results compared to STG, highlighting our method’s high-quality reconstruction with significantly reduced size.

D. Ablation Study

1) Learnable masking: As shown in Table VI, the proposed volume-based masking significantly reduces the number of Gaussians while retaining (even slightly increasing) the visual quality, demonstrating its effectiveness in removing redundant Gaussians. The reduced Gaussians show several additional advantages: reducing training time, storage, and testing time. Specifically on the Playroom scene, our proposed masking method shows a 140% increase in storage efficiency and a 65% increase in rendering speed. Furthermore, our method can remove redundant Gaussians effectively regarding spacetime redundancy in dynamic scenarios (Table VII). Especially for the Painter scene, we reduce almost 75% of Gaussians and

TABLE VI ABLATION STUDY ON THE PROPOSED CONTRIBUTIONS, MASKING, COLOR REPRESENTATION, GEOMETRY CODEBOOK, AND HALF TENSOR FOR POSITIONS AND OPACITIES. ‘#GAUSS’ MEANS THE NUMBER OF GAUSSIANS.

Method \Dataset Playroom Bonsai Mask Col Geo Half Post PSNR Train time #Gauss Storage FPS PSNR Train time #Gauss Storage FPS

3DGS 29.87 19m 22s 2.34 M 553 MB 154 32.16 19m 18s 1.25 M 295 MB 200 ✓ 29.91 17m 51s 967 K 228 MB 254 32.22 18m 50s 643 K 152 MB 247 ✓ ✓ 30.33 23m 56s 770 K 59 MB 210 32.08 23m 09s 592 K 51 MB 196 ✓ ✓ ✓ 30.33 24m 58s 761 K 44 MB 204 32.08 24m 06s 598 K 40 MB 198 ✓ ✓ ✓ ✓ 30.32 24m 35s 778 K 38 MB 206 32.08 24m 16s 601 K 35 MB 196 ✓ ✓ ✓ ✓ ✓ 30.30 - - 17 MB - 31.98 - - 15 MB -

TABLE VII ABLATION STUDY ON THE PROPOSED CONTRIBUTIONS, MASKING, COLOR REPRESENTATION, TEMPORAL CODEBOOK, GEOMETRY CODEBOOK, AND HALF TENSOR FOR POSITIONS AND SCALAR ATTRIBUTES, FOR DYNAMIC SCENES. ‘#GAUSS’ MEANS THE NUMBER OF GAUSSIANS.

Method \Dataset Painter Cut Roasted Beef Mask Color Time Geo Half Post PSNR SSIM #Gauss Storage FPS PSNR SSIM #Gauss Storage FPS

STG 36.21 0.929 553 K 84.1 MB 110 33.43 0.959 1.00 M 152 MB 181 ✓ 36.29 0.927 145 K 22.0 MB 132 33.32 0.958 342 K 51.9 MB 220 ✓ ✓ 36.45 0.925 121 K 19.2 MB 127 33.11 0.956 287 K 42.7 MB 208 ✓ ✓ ✓ 36.28 0.923 132 K 16.4 MB 122 33.06 0.955 286 K 33.0 MB 210 ✓ ✓ ✓ ✓ 36.22 0.923 132 K 14.0 MB 124 33.05 0.955 286 K 28.6 MB 212 ✓ ✓ ✓ ✓ ✓ 36.35 0.923 132 K 10.2 MB 115 33.09 0.955 286 K 19.4 MB 208 ✓ ✓ ✓ ✓ ✓ ✓ 36.35 0.923 - 6.56 MB - 33.03 0.955 - 13.3 MB -

|[Figure 169]|
|---|

|[Figure 170]|
|---|

|[Figure 171]|
|---|

|[Figure 172]|
|---|

3DGSw/masking

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

Gaussian Points Ellipsoids Rendered Result GT

- Fig. 8. Effect of the proposed learnable volume masking, compared to the original 3DGS. We visualize Gaussian center points, ellipsoids, and rendered results using the Playroom scene.

increase rendering speed by 20% without sacrificing rendering quality.

Fig. 8 further illustrates the effect of the masking method on Gaussians and the resulting images. Despite the noticeable reduction in the number of Gaussians, as evidenced by the sparser points in the visualization, the quality of the rendered results remains high, with no visible differences. These results demonstrate the effectiveness and efficiency of the proposed method, both quantitatively and qualitatively.

2) Compact color representation: For static scenes, the proposed color representation based on the neural field offers more than a threefold improvement in storage efficiency with a slightly reduced number of Gaussians compared to directly storing high-degree SH, despite necessitating slightly more time for training and rendering. Nonetheless, when compared to 3DGS, the proposed color representation with the masking

strategy demonstrates either a faster or comparable rendering speed.

When applying our compact color representation to STG, we achieve an additional 14% compression from the model with masked Gaussians. This is significant, considering that STG is already focused on compactness for color representation.

3) Codebook approach: Our proposed geometry codebook approach achieves a reduction in storage requirements by approximately 30% while maintaining the reconstruction quality, training time, and rendering speed, as shown in Table VI. Furthermore, we have validated that this codebook approach effectively reduces the storage requirements for temporal attributes in dynamic scenes as well as geometry (Table VII).

To analyze the actual effectiveness of the codebook-based approach, we showcase the geometry of the Gaussians, as

In [26]:

###### import numpy as np import seaborn as sns import matplotlib.pyplot as plt

# Assuming 'a' is defined and valid data = a[2].squeeze().cpu()

# Set up a 3x2 subplot fig, axs = plt.subplots(2, 3, figsize=(20, 15))

# Flatten the array of axes for easy iterating axs = axs.flatten()

for i in range(6): # Select the column col_data = data[:, i]

# Plot the histogram on the appropriate subplot sns.histplot(col_data, bins=64, kde=False, ax=axs[i]) axs[i].set_title(f'Stage{i+1} (Norm {norm[i]})', fontsize=40) axs[i].set_xlabel('Index', fontsize=30) axs[i].set_ylabel('Count', fontsize=30) axs[i].tick_params(axis='both', which='major', labelsize=20)

[Figure 176]

[Figure 177]

# Adjust layout for a clean look plt.tight_layout()

PREPRINT 12

# Show the plot plt.show()

[Figure 178]

|[Figure 179]<br><br>w/o R-VQ|
|---|

|[Figure 180]<br><br>w/ R-VQ|
|---|

|[Figure 181]<br><br>| |
|---|
|
|---|

|[Figure 182]<br><br>| |
|---|
|
|---|

(a) Rendered results (b) Learned codebook indices for rotation

- Fig. 9. Effect of the proposed geometry codebook. We visualize (a) ellipsoids and rendered results, and (b) learned codebook indices for rotation using the Stump scene. ‘Norm’ denotes the average norm of all code vectors in each codebook (representing magnitude).

TABLE VIII AVARGE STORAGE (MB) FOR EACH GAUSSIAN ATTRIBUTE, EVALUATED ON MIP-NERF 360 DATASET. F, 8B, H, AND P MEAN FLOATING-POINT, MIN-MAX QUANTIZATION TO 8-BIT, HUFFMAN ENCODING, AND PRUNING PARAMETERS BELOW 0.1, RESPECTIVELY. THE VALUE IN PARENTHESES INDICATES THE RESULT AFTER DEFLATE COMPRESSION.

Pos. Opa. Sca. Rot. Col. Tot. 3DGS

32f

746 37.9 12.6 37.9 50.6 606.9

16f R-VQ Hash(16f) MLP(16f)

Ours

48.8 8.3 2.8 6.3 6.3 25.2 0.016

Ours

16f 8b+H +H +8b+P+H MLP(16f) 29.1

+PP

(26.2) 8.3 1.2 5.9 6.2 7.4 0.016

of Gaussians without sacrificing visual quality through a novel learnable masking. Furthermore, this work proposed combining the neural field and exploiting the learnable codebooks to represent Gaussian attributes compactly. Through extensive experiments, our approach demonstrated more than 25× and 12× reduction in storage compared to 3DGS and STG, respectively, and an increase in rendering speed while retaining high-quality reconstruction. These results set a new benchmark with high visual quality, compactness, fast training, and realtime rendering for both static and dynamic radiance fields. Our framework thus stands as a comprehensive solution, paving the way for broader adoption and application in various fields requiring efficient and high-quality 3D scene representation.

shown in Fig. 9-(a). We can observe that the majority of Gaussians maintain their scales and rotations regardless of R-VQ, with only minor differences in a few that are hardly noticeable. We also explore the patterns of learned indices across each stage of R-VQ, shown in Fig. 9-(b). The lower stages exhibit relatively even distributions with large magnitudes of codes. As the stages progress, the distribution gets uneven and the magnitude of codes decreases, indicating the residuals of each stage have been reduced and trained to represent geometry.

4) Effect of post-processing : As shown in Table VI,VII, our post-processing techniques significantly reduce the storage requirement without performance drop, both for static and dynamic scenes. Additionally, Table VIII describes the size of each attribute with and without the application of postprocessing techniques for static scenes. While our end-to-end trainable framework demonstrates significant effectiveness, it requires relatively large storage for color representation. Nonetheless, as indicated in the table, this can be effectively reduced through simple post-processing.

VI. CONCLUSION

We have proposed a compact 3D Gaussian representation for both static and dynamic 3D scenes, reducing the number

REFERENCES

- [1] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3d gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics (ToG), vol. 42, no. 4, pp. 1–14, 2023.
- [2] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in European Conference on Computer Vision, 2020, p. 405–421.
- [3] J. T. Barron, B. Mildenhall, M. Tancik, P. Hedman, R. Martin-Brualla, and P. P. Srinivasan, “Mip-nerf: A multiscale representation for antialiasing neural radiance fields,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5855–5864.
- [4] T. M¨uller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics primitives with a multiresolution hash encoding,” ACM Trans. Graph., vol. 41, no. 4, 2022.
- [5] C. Reiser, S. Peng, Y. Liao, and A. Geiger, “Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 14335–14345.
- [6] S. Fridovich-Keil, A. Yu, M. Tancik, Q. Chen, B. Recht, and A. Kanazawa, “Plenoxels: Radiance fields without neural networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5501–5510.
- [7] A. Chen, Z. Xu, A. Geiger, J. Yu, and H. Su, “Tensorf: Tensorial radiance fields,” in European Conference on Computer Vision, 2022.
- [8] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, and A. Kanazawa, “K-planes: Explicit radiance fields in space, time, and appearance,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12479–12488.

- [9] K. Deng, A. Liu, J.-Y. Zhu, and D. Ramanan, “Depth-supervised nerf: Fewer views and faster training for free,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12882–12891.
- [10] D. Rho, B. Lee, S. Nam, J. C. Lee, J. H. Ko, and E. Park, “Masked wavelet representation for compact neural radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 20680–20690.
- [11] T. Takikawa, A. Evans, J. Tremblay, T. M¨uller, M. McGuire, A. Jacobson, and S. Fidler, “Variable bitrate neural fields,” in ACM SIGGRAPH 2022 Conference Proceedings, 2022.
- [12] J. C. Lee, D. Rho, S. Nam, J. H. Ko, and E. Park, “Coordinate-aware modulation for neural fields,” in International Conference on Learning Representations, 2024.
- [13] A. Pumarola, E. Corona, G. Pons-Moll, and F. Moreno-Noguer, “Dnerf: Neural radiance fields for dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 10318–10327.
- [14] Z. Li, S. Niklaus, N. Snavely, and O. Wang, “Neural scene flow fields for space-time view synthesis of dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

- 2021, pp. 6498–6508.

[15] T. Li, M. Slavcheva, M. Zollh¨ofer, S. Green, C. Lassner, C. Kim, T. Schmidt, S. Lovegrove, M. Goesele, R. Newcombe, and Z. Lv, “Neural 3d video synthesis from multi-view video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

- 2022, pp. 5521–5531.

- [16] F. Wang, S. Tan, X. Li, Z. Tian, Y. Song, and H. Liu, “Mixed neural voxels for fast multi-view video synthesis,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 19706–19716.
- [17] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20310–20320.
- [18] Z. Yang, H. Yang, Z. Pan, and L. Zhang, “Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting,” in The Twelfth International Conference on Learning Representations, 2024.
- [19] Z. Li, Z. Chen, Z. Li, and Y. Xu, “Spacetime gaussian feature splatting for real-time dynamic view synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 8508–8520.
- [20] J. C. Lee, D. Rho, X. Sun, J. H. Ko, and E. Park, “Compact 3d gaussian representation for radiance field,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21719–21728.
- [21] Z. Fan, K. Wang, K. Wen, Z. Zhu, D. Xu, and Z. Wang, “Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps,” arXiv preprint arXiv:2311.17245, 2023.
- [22] S. Niedermayr, J. Stumpfegger, and R. Westermann, “Compressed 3d gaussian splatting for accelerated novel view synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 10349–10358.
- [23] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5470–5479.
- [24] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, “pixelnerf: Neural radiance fields from one or few images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 4578–4587.
- [25] M. Niemeyer, J. T. Barron, B. Mildenhall, M. S. Sajjadi, A. Geiger, and N. Radwan, “Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5480–5490.
- [26] D. Xu, Y. Jiang, P. Wang, Z. Fan, H. Shi, and Z. Wang, “Sinnerf: Training neural radiance fields on complex scenes from a single image,” in European Conference on Computer Vision, 2022, pp. 736–753.
- [27] Q. Wang, Z. Wang, K. Genova, P. P. Srinivasan, H. Zhou, J. T. Barron, R. Martin-Brualla, N. Snavely, and T. Funkhouser, “Ibrnet: Learning multi-view image-based rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 4690–4699.
- [28] B. Roessle, J. T. Barron, B. Mildenhall, P. P. Srinivasan, and M. Nießner, “Dense depth priors for neural radiance fields from sparse input views,”

- in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 12892–12901.
- [29] B. Mildenhall, P. Hedman, R. Martin-Brualla, P. P. Srinivasan, and J. T. Barron, “Nerf in the dark: High dynamic range view synthesis from noisy raw images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16190–16199.
- [30] M. Tancik, V. Casser, X. Yan, S. Pradhan, B. Mildenhall, P. P. Srinivasan, J. T. Barron, and H. Kretzschmar, “Block-nerf: Scalable large scene neural view synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 8248–8258.
- [31] C. Gao, A. Saraf, J. Kopf, and J.-B. Huang, “Dynamic view synthesis from dynamic monocular video,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5712–5721.
- [32] C. Sun, M. Sun, and H.-T. Chen, “Direct voxel grid optimization: Superfast convergence for radiance fields reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5459–5469.
- [33] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Zip-nerf: Anti-aliased grid-based neural radiance fields,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 19697–19705.
- [34] J. Fang, T. Yi, X. Wang, L. Xie, X. Zhang, W. Liu, M. Nießner, and Q. Tian, “Fast dynamic radiance fields with time-aware neural voxels,” in SIGGRAPH Asia 2022 Conference Papers, 2022.
- [35] L. Liu, J. Gu, K. Zaw Lin, T.-S. Chua, and C. Theobalt, “Neural sparse voxel fields,” Advances in Neural Information Processing Systems, pp. 15651–15663, 2020.
- [36] J.-W. Liu, Y.-P. Cao, W. Mao, W. Zhang, D. J. Zhang, J. Keppo, Y. Shan, X. Qie, and M. Z. Shou, “Devrf: Fast deformable voxel radiance fields for dynamic scenes,” Advances in Neural Information Processing Systems, vol. 35, pp. 36762–36775, 2022.
- [37] E. R. Chan, C. Z. Lin, M. A. Chan, K. Nagano, B. Pan, S. De Mello, O. Gallo, L. J. Guibas, J. Tremblay, S. Khamis, T. Karras, and G. Wetzstein, “Efficient geometry-aware 3d generative adversarial networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16123–16133.
- [38] K. Han and W. Xiang, “Multiscale tensor decomposition and rendering equation encoding for view synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 4232–4241.
- [39] Q. Gao, Q. Xu, H. Su, U. Neumann, and Z. Xu, “Strivec: Sparse trivector radiance fields,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 17569–17579.
- [40] S. Nam, D. Rho, J. H. Ko, and E. Park, “Mip-grid: Anti-aliased grid representations for neural radiance fields,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 2837–2849.
- [41] Y. Chen, Q. Wu, M. Harandi, and J. Cai, “How far can we compress instant-ngp-based nerf?” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20321– 20330.
- [42] S. Shin and J. Park, “Binary radiance fields,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 55919–55931.
- [43] Q. Xu, Z. Xu, J. Philip, S. Bi, Z. Shu, K. Sunkavalli, and U. Neumann, “Point-nerf: Point-based neural radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5438–5448.
- [44] G. Kopanas, J. Philip, T. Leimk¨uhler, and G. Drettakis, “Point-based neural rendering with per-view optimization,” in Computer Graphics Forum, vol. 40, no. 4, 2021, pp. 29–43.
- [45] W. Yifan, F. Serena, S. Wu, C. Oztireli,¨ and O. Sorkine-Hornung, “Differentiable surface splatting for point-based geometry processing,” ACM Transactions on Graphics (TOG), vol. 38, no. 6, pp. 1–14, 2019.
- [46] O. Wiles, G. Gkioxari, R. Szeliski, and J. Johnson, “Synsin: End-to-end view synthesis from a single image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.
- [47] C. Lassner and M. Zollhofer, “Pulsar: Efficient sphere-based neural rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 1440–1449.
- [48] K.-A. Aliev, A. Sevastopolsky, M. Kolos, D. Ulyanov, and V. Lempitsky, “Neural point-based graphics,” in European Conference on Computer Vision, 2020, pp. 696–712.
- [49] M. Meshry, D. B. Goldman, S. Khamis, H. Hoppe, R. Pandey, N. Snavely, and R. Martin-Brualla, “Neural rerendering in the wild,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

- [50] G. Kopanas, T. Leimk¨uhler, G. Rainer, C. Jambon, and G. Drettakis, “Neural point catacaustics for novel-view synthesis of reflections,” ACM Transactions on Graphics (TOG), vol. 41, no. 6, pp. 1–15, 2022.
- [51] Q. Zhang, S.-H. Baek, S. Rusinkiewicz, and F. Heide, “Differentiable point-based radiance fields for efficient view synthesis,” in SIGGRAPH Asia 2022 Conference Papers, 2022.
- [52] G. Chen and W. Wang, “A survey on 3d gaussian splatting,” arXiv preprint arXiv:2401.03890, 2024.
- [53] B. Fei, J. Xu, R. Zhang, Q. Zhou, W. Yang, and Y. He, “3d gaussian splatting as new era: A survey,” IEEE Transactions on Visualization and Computer Graphics, 2024.
- [54] K. Navaneet, K. P. Meibodi, S. A. Koohpayegani, and H. Pirsiavash, “Compact3d: Compressing gaussian splat radiance field models with vector quantization,” arXiv preprint arXiv:2311.18159, 2023.
- [55] W. Morgenstern, F. Barthel, A. Hilsmann, and P. Eisert, “Compact 3d scene representation via self-organizing gaussian grids,” arXiv preprint arXiv:2312.13299, 2023.
- [56] S. Girish, K. Gupta, and A. Shrivastava, “Eagles: Efficient accelerated 3d gaussians with lightweight encodings,” arXiv preprint arXiv:2312.04564, 2023.
- [57] Y. Du, Y. Zhang, H.-X. Yu, J. B. Tenenbaum, and J. Wu, “Neural radiance flow for 4d view synthesis and video processing,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 14324–14334.
- [58] Z. Li, Q. Wang, F. Cole, R. Tucker, and N. Snavely, “Dynibar: Neural dynamic image-based rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 4273–4284.
- [59] Y.-L. Liu, C. Gao, A. Meuleman, H.-Y. Tseng, A. Saraf, C. Kim, Y.-Y. Chuang, J. Kopf, and J.-B. Huang, “Robust dynamic radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 13–23.
- [60] K. Park, U. Sinha, J. T. Barron, S. Bouaziz, D. B. Goldman, S. M. Seitz, and R. Martin-Brualla, “Nerfies: Deformable neural radiance fields,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 5865–5874.
- [61] K. Park, U. Sinha, P. Hedman, J. T. Barron, S. Bouaziz, D. B. Goldman, R. Martin-Brualla, and S. M. Seitz, “Hypernerf: a higher-dimensional representation for topologically varying neural radiance fields,” ACM Transactions on Graphics (TOG), vol. 40, no. 6, 2021.
- [62] E. Tretschk, A. Tewari, V. Golyanik, M. Zollh¨ofer, C. Lassner, and C. Theobalt, “Non-rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 12959–12970.
- [63] L. Li, Z. Shen, Z. Wang, L. Shen, and P. Tan, “Streaming radiance fields for 3d video synthesis,” Advances in Neural Information Processing Systems, vol. 35, pp. 13485–13498, 2022.
- [64] B. Attal, J.-B. Huang, C. Richardt, M. Zollh¨ofer, J. Kopf, M. O’Toole, and C. Kim, “Hyperreel: High-fidelity 6-dof video with ray-conditioned sampling,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 16610–16620.
- [65] L. Song, A. Chen, Z. Li, Z. Chen, L. Chen, J. Yuan, Y. Xu, and A. Geiger, “Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields,” IEEE Transactions on Visualization and Computer Graphics, vol. 29, no. 5, pp. 2732–2742, 2023.
- [66] L. Wang, Q. Hu, Q. He, Z. Wang, J. Yu, T. Tuytelaars, L. Xu, and M. Wu, “Neural residual radiance fields for streamably free-viewpoint videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 76–87.
- [67] A. Cao and J. Johnson, “Hexplane: A fast representation for dynamic scenes,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 130–141.
- [68] R. Shao, Z. Zheng, H. Tu, B. Liu, H. Zhang, and Y. Liu, “Tensor4d: Efficient neural 4d decomposition for high-fidelity dynamic reconstruction and rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 16632– 16642.
- [69] M. I¸sık, M. R¨unz, M. Georgopoulos, T. Khakhulin, J. Starck, L. Agapito, and M. Nießner, “Humanrf: High-fidelity neural radiance fields for humans in motion,” ACM Transactions on Graphics (TOG), vol. 42, no. 4, 2023.
- [70] F. Wang, Z. Chen, G. Wang, Y. Song, and H. Liu, “Masked space-time hash encoding for efficient dynamic scene reconstruction,” Advances in Neural Information Processing Systems, vol. 36, 2024.

- [71] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, “Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis,” in International Conference on 3D Vision (3DV), 2024, pp. 800–809.
- [72] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20331–20341.
- [73] Z. Xu, S. Peng, H. Lin, G. He, J. Sun, Y. Shen, H. Bao, and X. Zhou, “4k4d: Real-time 4d view synthesis at 4k resolution,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20029–20040.
- [74] Y. Lin, Z. Dai, S. Zhu, and Y. Yao, “Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 21136–21145.
- [75] K. Katsumata, D. M. Vo, and H. Nakayama, “An efficient 3d gaussian representation for monocular/multi-view dynamic scenes,” arXiv preprint arXiv:2311.12897, 2023.
- [76] A. Kratimenos, J. Lei, and K. Daniilidis, “Dynmf: Neural motion factorization for real-time dynamic view synthesis with 3d gaussian splatting,” arXiv preprint arXiv:2312.00112, 2023.
- [77] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.
- [78] Y. Bengio, N. L´eonard, and A. Courville, “Estimating or propagating gradients through stochastic neurons for conditional computation,” arXiv preprint arXiv:1308.3432, 2013.
- [79] R. Gray, “Vector quantization,” IEEE Assp Magazine, vol. 1, no. 2, pp. 4–29, 1984.
- [80] N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “Soundstream: An end-to-end neural audio codec,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 495–507, 2021.
- [81] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples: Benchmarking large-scale scene reconstruction,” ACM Transactions on Graphics (ToG), vol. 36, no. 4, pp. 1–13, 2017.
- [82] P. Hedman, J. Philip, T. Price, J.-M. Frahm, G. Drettakis, and G. Brostow, “Deep blending for free-viewpoint image-based rendering,” ACM Transactions on Graphics (ToG), vol. 37, no. 6, pp. 1–15, 2018.
- [83] N. Sabater, G. Boisson, B. Vandame, P. Kerbiriou, F. Babon, M. Hog, R. Gendrot, T. Langlois, O. Bureller, A. Schubert, and V. Allie, “Dataset and pipeline for multi-view light-field video,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2017.
- [84] D. A. Huffman, “A method for the construction of minimum-redundancy codes,” Proceedings of the IRE, vol. 40, no. 9, pp. 1098–1101, 1952.
- [85] P. Deutsch, “Deflate compressed data format specification version 1.3,” Tech. Rep., 1996.

[Figure 183]

Joo Chan Lee (Graduate Student Member, IEEE) received the B.S. degree in information and communication engineering from Inha University in 2020. He is currently pursuing the Ph.D. degree in artificial intelligence from Sungkyunkwan University. His current research interests include the areas of computer vision, graphics, and machine learning.

Daniel Rho is currently a Ph.D. student at the University of North Carolina at Chapel Hill. He received an M.S. degree in Artificial Intelligence from Sungkyunkwan University, South Korea, in 2022, and a double major in Economics and Computer Science from the same university in 2020. His research interests include neural rendering, computer graphics, and machine learning.

[Figure 184]

[Figure 185]

Xiangyu Sun received the bachelor’s and master’s degrees from Huazhong University of Science and Technology, China, in 2018 and 2021. He is currently working toward the Ph.D. degree with the VSC Lab, Sungkyunkwan University. His research interests include 3D reconstruction, neural radiance field and 3D generative model.

[Figure 186]

Jong Hwan Ko (Member, IEEE) received the dual B.S. degrees in computer science and engineering and mechanical and aerospace engineering and the M.S. degree in electrical engineering and computer science from Seoul National University, and the Ph.D. degree from the School of Electrical and Computer Engineering, Georgia Tech, in 2018. During his seven years of research experience at the Agency for Defense Development (ADD) in South Korea, he conducted advanced research on the design and performance analysis of military wireless sensor

networks. He joined Sungkyunkwan University (SKKU), South Korea, as an Assistant Professor. His research interests include design of low-power image sensor systems and deep-learning accelerators for efficient image/audio processing. He has received the Best Paper Award from the International Symposium on Low Power Electronics and Design (ISLPED), in 2016.

Eunbyung Park (Member, IEEE) received a B.S. degree in computer science from Kyung Hee University in 2009, an M.S. degree in computer science from Seoul National University in 2011, and a Ph.D. degree in computer science from the University of North Carolina at Chapel Hill in 2019. He is currently an assistant professor in the Department of Electronic and Electrical Engineering at Sungkyunkwan University (SKKU), South Korea. Before joining SKKU, he was a research scientist at Nuro and an applied scientist at Microsoft. During

[Figure 187]

his doctoral study, he has worked at various research institutes, including Google DeepMind, Microsoft Research, Adobe Research, and HP Labs. His current research interests include computer vision and machine learning and its applications to visual and scientific computing.

