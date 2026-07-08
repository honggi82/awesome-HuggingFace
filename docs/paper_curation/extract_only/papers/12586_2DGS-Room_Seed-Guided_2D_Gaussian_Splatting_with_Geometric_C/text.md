## 2DGS-Room: Seed-Guided 2D Gaussian Splatting with Geometric Constrains for High-Fidelity Indoor Scene Reconstruction

Wanting Zhang Haodong Xiang Zhichao Liao Xiansong Lai Xinghui Li† Long Zeng† Tsinghua University https://valentina-zhang.github.io/2DGS-Room/

# arXiv:2412.03428v1[cs.CV]4Dec2024

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

#### 2DGSOurs

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

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Figure 1. 2DGS-Room achieves high-fidelity geometric reconstructions for indoor scenes. We introduce seed points to guide the distribution of 2D Gaussians coupled with geometric constraints, leading to clearer structures and more accurate geometry.

##### Abstract

further enhance reconstruction quality. Extensive experiments on ScanNet and ScanNet++ datasets demonstrate that our method achieves state-of-the-art performance in indoor scene reconstruction.

The reconstruction of indoor scenes remains challenging due to the inherent complexity of spatial structures and the prevalence of textureless regions. Recent advancements in 3D Gaussian Splatting have improved novel view synthesis with accelerated processing but have yet to deliver comparable performance in surface reconstruction. In this paper, we introduce 2DGS-Room, a novel method leveraging 2D Gaussian Splatting for high-fidelity indoor scene reconstruction. Specifically, we employ a seed-guided mechanism to control the distribution of 2D Gaussians, with the density of seed points dynamically optimized through adaptive growth and pruning mechanisms. To further improve geometric accuracy, we incorporate monocular depth and normal priors to provide constraints for details and textureless regions respectively. Additionally, multi-view consistency constraints are employed to mitigate artifacts and

##### 1. Introduction

3D reconstruction from multi-view RGB images is a fundamental task in the fields of computer vision and computer graphics. The reconstructed models can be utilized in a wide range of applications, including virtual reality, video games, autonomous driving, and robotics. Reconstructing indoor scenes is a challenging task in the field of 3D reconstruction, as indoor environments often contain large textureless regions. MVS-based methods [1–3] often yield incomplete or geometrically flawed reconstructions, primarily due to the geometric ambiguities arising from the presence of textureless regions.

Recent advancements in neural-radiance-field-based methods [4–8] that utilize signed distance fields (SDF) for scene modeling have enabled accurate and complete mesh reconstruction in indoor environments. This progress is attributed to the continuity of neural SDFs and the integration of monocular geometric priors [6]. Although neuralradiance-field-based methods achieve high-quality reconstruction, they are computationally expensive due to the need for dense ray sampling, resulting in long optimization times. Fortunately, 3D Gaussian Splatting (3DGS) [9] enhances the optimization and rendering efficiency of neural rendering through its differentiable rasterization technique, offering new possibilities for 3D scene reconstruction. 2DGS [10] build upon 3DGS by using 2D-oriented planar Gaussians as primitives, significantly improving surface reconstruction quality. Despite these advances, Gaussian splatting-based methods still often produce floating artifacts and incomplete reconstructions in indoor scenes, due to the lack of structured geometric constraints.

In this work, we present a novel approach named 2DGSRoom, aiming to achieve high-fidelity geometric reconstruction for indoor scenes based on 2D Gaussian Splatting. Considering the scene’s underlying structure, we propose a seed-guided mechanism to control the distribution and density of 2D Gaussians. Specifically, we introduce a seed-guided initialization to generate 2D Gaussians, ensuring their alignment with scene surfaces to improve geometric accuracy. To further refine the reconstruction, we propose a seed-guided optimization strategy that dynamically adjusts seed point density through gradient-guided growth and contribution-based pruning, enabling efficient representation of fine details. Additionally, we incorporate monocular depth and normal priors to provide crucial geometric constraints. The depth prior addresses distortions in detailed areas, while the normal prior ensures accurate surface estimation in textureless regions. Furthermore, we introduce multi-view consistency constraints to address residual artifacts, which enforces both geometric and photometric consistency across multiple views.

Extensive qualitative and quantitative experiments show that compared with Gaussian-based methods, 2DGS-Room achieves start-of-the-art performance in indoor scenarios. In summary, our contributions are as follows:

- • We propose 2DGS-Room, a novel method for indoor scene reconstruction based on 2DGS, which leverages the seed points maintaining the scene structure to guide the distribution and density of 2D Gaussians.
- • We introduce monocular depth and normal priors to provide geometric cues, improving the reconstruction of detailed areas and textureless regions respectively.
- • We employ multi-view constraints incorporating geometric and photometric consistency to further enhance the reconstruction quality.

• Our method achieves high-quality surface reconstruction for indoor scenes. Extensive experiments on indoor scene datasets show that our method achieves state-of-the-art in multiple evaluation metrics.

##### 2. Related work

###### 2.1. Multi-View Stereo

Multi-view stereo (MVS) methods [1, 11–13] estimate the 3D coordinates of pixels and explicitly reconstruct objects and scenes by matching features across a collection of posed images. The surface is then obtained through the application of Poisson surface reconstruction [14]. In indoor scenes, particularly in large texture-less regions, these methods frequently encounter difficulties due to the scarcity of features. Voxel-based approaches [15–18] optimize spatial occupancy and color within a voxel grid, thus avoiding the challenges of feature matching. However, highresolution memory constraints degrade reconstruction quality. Learning-based multi-view stereo methods [2, 3, 19–25] implicitly match corresponding multi-view features through neural networks, enabling end-to-end 3D reconstruction. Nonetheless, even with extensive training data, errors may still occur in the results when handling occlusions, complex lighting, or regions with subtle textures.

###### 2.2. Neural Radiance Field

Neural Radiance Fields (NeRF) [26] employs a multi-layer perceptron (MLP) to model a continuous volumetric function of density and color, enabling novel view synthesis through volume rendering. Methods such as Mip-NeRF [27–29] enhance rendering quality by improving the ray sampling strategy. Other works [30–34] accelerate training and rendering through techniques such as multi-resolution hash encoding or resizing MLPs. Some studies aim to enhance rendering quality by incorporating regularization terms. For example, depth regularization [35, 36] explicitly supervises ray termination to minimize unnecessary sampling time. Other approaches focus on enforcing smoothness constraints on rendered depth maps [37] or utilizing multi-view consistency regularization in sparse-view scenarios [38, 39]. Some research explores the use of alternative implicit functions to enhance the geometric reconstruction capabilities of NeRF, such as occupancy grids [40, 41] and signed distance functions (SDFs) [4, 5, 34, 42, 43], replacing NeRF’s volumetric density field. To further enhance reconstruction quality, [44, 45] suggest regularizing optimization with SfM points, while [6, 46] incorporate priors like the Manhattan world assumption and pseudo depth supervision. However, these approaches often lead to incomplete reconstructions and require extensive optimization time.

###### (a) Seed Points Guidance

[Figure 37]

| | | |
|---|---|---|
| | | |
| | | |

Multi-view Images

[Figure 38]

[Figure 39]

Voxelization from SfM Points

[Figure 40]

[Figure 41]

[Figure 42]

|[Figure 43]<br><br>| | | |
|---|---|---|
| | | |
| | | |
<br><br>|
|---|

𝜵𝒗 > 𝜽𝒈

[Figure 44]

Rendering

[Figure 45]

[Figure 46]

[Figure 47]

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| |
|---|

[Figure 48]

[Figure 49]

|[Figure 50]<br><br>[Figure 51]|
|---|

Seed Point 2D Gaussian

[Figure 52]

α𝒗 < 𝜽α

[Figure 53]

Normal

Offset

Depth Map

[Figure 54]

###### TSDF 𝓛pho

[Figure 55]

|pn|
|---|

|[Figure 56]| |
|---|---|
| | |

[Figure 57]

[Figure 58]

Depth Model

###### D

𝓛d

|pr|
|---|

[Figure 59]

|[Figure 60]| |
|---|---|
| | |

[Figure 61]

(x, y, z)

|prn|
|---|

Normal Model

𝓛n N

𝓛geo

[Figure 62]

|pr|
|---|

Normal Map

(b) Monocular Cues Supervision

(c) Multi-View Consistency

Reconstruction

- Figure 2. Overview of 2DGS-Room. Given multi-view posed images, we improve 2DGS to achieve high-fidelity geometric reconstruction for indoor scenes. (a) Starting from an SfM-derived point cloud, we generate a set of seed points through voxelization, establishing a stable foundation for guiding the distribution and density of 2D Gaussians. We further introduce an adaptive growth and pruning strategy to optimize seed points. (b) We incorporate depth and normal priors, addressing the challenges of detailed areas and textureless regions. (c) We introduce multi-view consistency constraints to further enhance the quality of the indoor scene reconstruction.

###### 2.3. Gaussian Splatting

- 3D Gaussian Splatting [9] explicitly represents 3D scenes using learnable Gaussian primitives, enabling high-quality novel view synthesis with short training times and high rendering frame rates. The 3DGS method is solely responsible for the image loss, and after initializing with sparse point clouds generated by SfM [47], no further constraints are applied to the Gaussian primitives. This leads to a disorganized distribution of the optimized Gaussian primitives, resulting in poor geometric properties. Works such as DNSplatter [48], GaussianRoom [49] and GSDF [50] introduce geometric priors or leverage the accurate geometric information from SDFs to supervise the optimization of Gaussians. SuGaR [51], PGSR [52] and RaDe-GS [53] use Flatten Gaussians to represent scenes, enhancing surface reconstruction capabilities. In contrast, 2DGS [10] directly applies 2D oriented planar Gaussians instead of 3D Gaussian primitives to represent 3D scenes, achieving better surface reconstruction results. However, it still encounters poor reconstruction in indoor scenes due to Gaussian primitives lacking geometric constraints.

##### 3. Preliminary

The key innovation of 2DGS [10] lies in its transformation of 3D volumetric Gaussians into flat 2D Gaussians, or surfels, for scene representation. It directly models scenes with

- 2D elliptical disks, simplifying the representation process and yielding more accurate geometry without extra mesh refinement.

Each 2D Gaussian disk, defined in a local tangent plane, is parameterized by a central point pk, two orthogonal tangential vectors tu and tv, and a scaling vector (su,sv) that controls the variances along each direction. The normal tw of each Gaussian disk is computed as tw = tu × tv and this orientation can be arranged into a rotation matrix R = [tu,tv,tw]. The scaling factors can be arranged into a

- 3 × 3 diagonal matrix S = [su,sv,0]. Then a 2D Gaussian can be parameterized:

###### P(u,v) = pk + sutuu + svtvv = H(u,v,1,1), (1)

where H ∈ 4 × 4 is a homogeneous transformation matrix representing the geometry of the 2D Gaussian:

H =

sutu svtv 0 pk 0 0 0 1

=

RS pk 0 1

. (2)

In the Gaussian’s tangent frame (u,v), the 2D Gaussian value G(u) at point u = (u,v) is evaluated as:

u2 + v2 2

. (3)

G(u) = exp −

For efficient rendering, each 2D Gaussian is projected onto the image plane by a general 2D-to-2D mapping in ho-

mogeneous coordinates. Given a world-to-screen transformation matrix W, the screen space points can be derived from:

###### x = (xy,yz,z,z)⊤ = WH(u,v,1,1)⊤. (4)

where x represents a homogeneous ray emitted from the camera and passing through pixel (x,y) and intersecting the splat at depth z.

To avoid numerical instability, a ray-splat intersection is calculated explicitly by finding the intersection of three non-parallel planes in the 3D scene. Given an image coordinate x = (x,y), the ray of a pixel can be defined by the intersection of two homogeneous planes: the x-plane hx = (−1,0,0,x) and the y-plane hy = (0,−1,0,y). To compute the intersection with the Gaussian splat, both planes are transformed to uv-space:

###### hu = (WH)⊤hx, hv = (WH)⊤hy. (5)

By homography, the two planes are used to find the intersection point (u(x),v(x)) with the 2D Gaussian splats, given by:

h2uh4v − h4uh2v h1uh2v − h2uh1v

h4uh1v − h1uh4v h1uh2v − h2uh1v

, (6)

u(x) =

, v(x) =

where hiu and hiv are components of the transformed planes in the Gaussian’s tangent frame.

##### 4. Methods

Given multi-view posed images, our goal is to optimize 2DGS [10] to accurately reconstruct the geometry of indoor scenes. To this end, we first propose a seed-guided mechanism, which leverages seed points to control the distribution and density of 2D Gaussians, thereby improving the accuracy and efficiency of scene representation in indoor scenes (Sec. 4.1). To further improve geometric accuracy, we incorporate depth and normal priors, which enhance the representation of detailed areas and textureless regions, respectively (Sec. 4.2). Finally, to mitigate floating artifacts caused by lighting variations in indoor scenes, we introduce multi-view consistency constraints, further enhancing the quality of the indoor scene reconstruction (Sec. 4.3). An overview of our framework is provided in Fig. 2.

###### 4.1. Seed Points Guidance

Existing methods [9, 10] tend to optimize Gaussians relying on each training view, ignoring the underlying structure of the scene. As illustrated in Fig. 3 (a) and (b), the Gaussian primitives fail to align with the surfaces. To overcome this limitation, we propose a seed-guided mechanism to control the distribution of 2D Gaussians. Specifically, we utilize a set of seed points to provide a stable foundation for generating 2D Gaussians, ensuring that the reconstruction reflects

the underlying scene structure more accurately. Additionally, we introduce an adaptive growth and pruning strategy to dynamically adjust the density of seed points.

Seed-Guided Initialization. Starting from an SfM-derived point cloud P ∈ RM×3, we first filter some unreliable outliers. We define a confidence measure Op

for each individual point pi in the point cloud. This measure is expressed as follows:

i

1 if m ≥ ϵ 0 if m < ϵ

, (7)

Op

=

i

where m represents the number of image feature matches associated with pi, and ϵ is a predefined threshold. Points with a number of matched features below ϵ are deemed unreliable and removed from the point cloud to ensure a more accurate reconstruction.

Following the filtering process, we apply voxelization to generate a set of seed points V ∈ RN×3 by selecting the center points of each voxel grid to represent the seed points:

V =

P δ · δ , (8)

where δ denotes the voxel grid size. Each seed point v ∈ V serves as the basis for deriving several 2D Gaussians, which are positioned based on learnable offsets from the seed point. This initialization ensures that the distribution of Gaussians is closely aligned with the underlying geometry of the scene, thereby improving the overall robustness of the reconstruction quality.

For each seed point v ∈ V, we initialize a set of k 2D Gaussians {Gi,j}, where Gi,j denotes the j-th Gaussian associated with the i-th seed. The position of each Gaussian is determined by a learnable offset Oi,j from the seed point location:

pi,j = vi + Oi,j, (9)

where pi,j ∈ R3 represents the global position of the Gaussian, and Oi,j ∈ R3 is a learnable offset which is optimized during training to adjust each Gaussian’s local position for better alignment with the scene.

Expect for the center position, each 2D Gaussian is parameterized by the scaling s ∈ R2, rotation t ∈ R2, appearance c ∈ R3 and opacity α ∈ R. At initialization, the scaling and rotation are aligned with the local geometry derived from the point cloud, which provides a starting approximation that reflects the scene’s spatial distribution. During training, these parameters are iteratively optimized to refine the representation.

Seed-Guided Optimization. In order to capture different levels of detail in complex indoor scenes, we develop an adaptive approach to dynamically adjust seed point density by combining gradient-guided growth and contributionbased pruning.

[Figure 63]

[Figure 64]

[Figure 65]

(a) 3DGS (b) 2DGS (c) Ours

- Figure 3. Ground truth scene surface and Gaussian primitives distribution. Compared with 3DGS and 2DGS, our method significantly reduces scattered floaters in the non-surface areas, benefitting from our designed structured geometric constraints.

We utilize a gradient-guided growth strategy to increase seed point density adaptively, especially in areas with high structural complexity or fine details. For each voxel, we compute the average gradient ∇v of the included 2D Gaussians across Ng training iterations, using it as an indicator of structural complexity. When ∇v exceeds a threshold θg, additional seed points are introduced to enhance representation. This growth occurs within a multi-resolution voxel structure, with thresholds that adapt according to the resolution level, ensuring a higher seed density in regions requiring more detail.

Moreover, we implement a contribution-based pruning strategy that selectively removes low-impact seed points. For each seed, we calculate the cumulative opacity αv of the connected 2D Gaussians over Nα iterations. If αv is below a predefined threshold θα, the seed point is pruned, as its minimal contribution to scene opacity suggests the limited impact on the overall representation. This strategy allows us to allocate Gaussians to regions of higher structural significance, enhancing both computational efficiency and reconstruction quality.

- 4.2. Monocular Cues Supervision

While the control of seed points enhances the structural consistency of the scene, it remains insufficient for achieving highly accurate geometry, particularly in detailed or textureless regions which are common in indoor environments. Therefore, we incorporate depth and surface normal priors, providing geometric constraints to further improve the scene reconstruction.

Monocular Depth Supervision. The depth prior is leveraged to mainly refine the spatial alignment of objects in the scene by aligning the rendered depths with reference depths predicted from a pre-trained model [54]. We incorporate depth supervision by aligning the rendered depths with reference depths through a scale-and-shift-invariant loss [55], compensating for relative scaling discrepancies that may arise in the representation of complex indoor geometries.

Given the rendered depths Dˆ, we first compute optimal scale s and shift t values to minimize discrepancies in scale and translation between our rendered depths and the

reference depths to address potential inconsistencies that may arise due to relative scaling differences in complex scenes. Then we adjust the predicted depth map to obtain the aligned prediction: Dˆaligned = s · Dˆ + t.

The depth loss Ld consists of two terms: a data term that minimizes the mean squared error (MSE) between the aligned rendered depths Dˆaligned and the reference depths D, and a regularization term for gradient consistency that encourages local smoothness in the depth rendering. Formally, the depth loss is defined as:

1 |Vd|

∥Dˆaligned − D∥2 + λgrad · Lgrad, (10)

Ld =

where |Vd| represents the number of pixels with valid depths, and Lgrad is a spatial regularization term that penalizes abrupt depth variations across neighboring pixels.

Monocular Normal Supervision. Additionally, the normal prior plays a crucial role in addressing the reconstruction challenges of textureless or planar regions like walls and floors. So we also incorporate normal supervision to enforce a smooth and realistic surface orientation throughout the scene.

Let Nˆ denote the reference normals derived from a pretrained model [56], and N represents the rendered normals. We first use the L1 norm loss to quantify the absolute difference in magnitude between the rendered and reference normals, promoting consistency in the length of the vectors:

1 |Vn|

N − Nˆ , (11)

L1 =

where |Vn| is the number of pixels with valid reference normals.

To further encourage the alignment of with Nˆ, we use a cosine similarity loss that penalizes angular differences between the two normal vectors:

N · Nˆ ∥N∥ · ∥N∥ˆ

1 |Vn|

. (12)

Lcos =

1 −

The final normal supervision loss Ln is defined as:

Ln = λ1 · L1 + λcos · Lcos. (13)

###### 4.3. Multi-View Consistency Constraints

The strategies outlined above significantly improve the accuracy of indoor scene reconstruction, but we observe that some small floaters may still persist in certain scenarios. These cases are likely caused by the complex lighting variations and subtle spatial structures typical in indoor environments. Therefore, we introduce multi-view consistency constraints to further refine the reconstruction by reducing the inconsistencies that occasionally manifest across different views. Specifically, as shown in Figure 2, given a

reference view Vr, we select a neighboring view Vn and enforce geometric consistency and photometric consistency between the two views.

Geometric Consistency Constraint. To ensure consistent geometry across views, we define a pixel-wise geometric consistency loss that penalizes discrepancies in the forward and backward projections for each individual pixel.

We compute a transformation Hrn to represent the homography matrix mapping a pixel pr from Vr to the corresponding pixel pn in Vn:

TrnNr⊤ Dr

Kr−1, (14)

Hrn = Kn Rrn −

where K denotes the camera’s intrinsic matrix. Rrn and Trn are the relative rotation and translation from the reference frame to the neighboring frame.

For each pixel pr , we project it forward from Vr to Vn using Hrn, and then back-project from Vn to Vr using Hnr. The resulting multi-view geometric consistency loss Lgeo is formulated as:

1 |Ve| p

∥pr − HnrHrnpr∥, (15)

Lgeo =

r∈Ve

where Ve is a set of valid pixels excluding those with high forward and backward projection errors.

Photometric Consistency Constraint. To account for local variations in texture and illumination, we also enforce photometric consistency which is measured using the normalized cross-correlation (NCC) [59], penalizing differences in pixel intensity distributions between the views.

Focusing on geometric details, we convert color images into grayscale and the photometric consistency loss Lpho is defined as:

1 |Ve| p

Lpho =

r∈Ve

(1 − NCC(Gr(pr),Gn(Hrnpr))),

(16)

where Gr and Gn denote the grayscale intensities of the patches in Vr and Vn, respectively.

Finally, the total multi-view consistency loss Lmv is given by:

Lmv = λgeoLgeo + λphoLpho. (17)

###### 4.4. Optimization

In summary, with Lrgb representing the photometric supervision that minimizes the difference between rendered and input images proposed in the original 2DGS, our final training loss L is given by:

L = Lrgb + λd · Ld + λn · Ln + Lmv, (18)

where λd and λn control the relative contributions of depth and normal supervision, respectively.

##### 5. Experiments 5.1. Experimental Setup

Dataset. We evaluate the performance of our approach on reconstruction quality across 12 real-world indoor scenes from publicly available datasets: 8 scenes from ScanNet(V2) [57] and 4 scenes from ScanNet++ [58].

Implementation Details. Our training strategy and hyperparameters are consistent with the baseline 2DGS method to ensure comparability. We set k = 10, λ1 = 0.01, λcos = 0.01, λgrad = 0.5, λgeo = 0.05, λpho = 0.2, λd = 1.0, λn = 1.0, in all our experiments. We render depth maps for all training views and then adopt TSDF fusion [60] for mesh extraction. We train all models for 30k iterations. All experiments are conducted on an NVIDIA RTX 4090 GPU to ensure consistent processing.

Metrics. Consistent with existing methods [5, 6], five standard metrics are employed to evaluate the quality of reconstructed meshes: Accuracy, Completion, Precision, Recall, and F-score.

ScanNet [57] ScanNet++ [58] Method

Acc. ↓ Comp. ↓ Prec. ↑ Recall ↑ F-score ↑ Acc. ↓ Comp. ↓ Prec. ↑ Recall ↑ F-score ↑

NeuS [4] 0.105 0.124 0.448 0.378 0.409 0.160 0.224 0.294 0.221 0.251 Neuralangelo [34] 0.185 0.223 0.252 0.260 0.255 0.363 0.264 0.172 0.120 0.141 3DGS [9] 0.338 0.406 0.129 0.067 0.085 0.144 0.990 0.322 0.066 0.104 SuGaR [51] 0.167 0.148 0.361 0.373 0.366 0.158 0.178 0.383 0.349 0.361 2DGS [10] 0.157 0.151 0.336 0.347 0.341 0.359 0.228 0.230 0.160 0.183 PGSR [52] 0.125 0.117 0.420 0.433 0.426 0.204 0.202 0.353 0.217 0.249 RaDe-GS [53] 0.167 0.205 0.309 0.307 0.306 0.284 0.252 0.171 0.179 0.166 2DGS-Room (Ours) 0.055 0.092 0.648 0.518 0.575 0.262 0.112 0.450 0.498 0.464

- Table 1. Quantitative reconstruction comparison on ScanNet and ScanNet++ dataset. Averaged results are reported over 8 scenes and

- 4 scenes, respectively. 2DGS-Room achieves the best F-score.

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

NeuS SuGaR RaDe-GS PGSR 2DGS Ours GT

- Figure 4. Qualitative reconstruction comparisons. For each indoor scene, the first row is the top view of the whole room, and the second row is the details of the masked region.

###### 5.2. Results Analysis

Baselines. We compare our approach with several stateof-the-art methods, covering both neural volume rendering and Gaussian splatting techniques. The baselines include (1) Neural volume rendering methods: NeuS [4] and NeuralAngelo [34]; (2) Gaussian splatting methods: 3DGS [9], SuGaR [51], RaDe-GS [53], PGSR [52], and 2DGS [10].

Qualitative Results. To show the visualized reconstruction results of our method, we compare our 2DGS-Room with different reconstruction methods, including NeuS [4], SuGaR [51], RaDe-GS [53], PGSR [52], 2DGS [10], and the ground truth. As illustrated in Figure 4, our method ex-

Seed Points Guidance. Figure 5 shows that without seed points guidance, the scene lacks clear structural organization, leading to a significantly inflated and disorganized reconstruction. Adding this module enables our method to better capture the underlying geometric framework of indoor scenes, improving the F-score by 87.3% in Table 2.

hibits significantly clearer scene structures, which is largely attributed to the seed-guided strategy. Additionally, thanks to the incorporation of depth and normal priors, the overall quality of our reconstructions is noticeably higher. In comparison with Gaussian-based methods, our method obtains a more visually coherent and accurate representation of the indoor scenes, with well-defined surfaces and consistent details across different views.

Monocular Depth Supervision. As shown in Figure 5, removing depth supervision leads to spatial misalignments and unrealistic arrangements. Incorporating depth supervision significantly enhances geometric accuracy, achieving a 31.3% F-score increase as reported in Table 2.

Quantitative Results. Quantitative results are presented in Table 1, showing a comprehensive comparison in geometry metrics on indoor scene datasets. On the ScanNet dataset, our method achieves the best results in all metrics. Compared to NeRF-based methods [4, 34] which typically require over 20 hours to train a scene, our method significantly reduces training time, being approximately 30 times faster.

Monocular Normal Supervision. Removing normal supervision results in surface inconsistencies as shown in Figure 5, with certain planar areas like walls, floors, and doors misaligned. Adding this module improves surface alignment, increasing the F-score by 10.6% in Table 2.

Multi-View Consistency Constraints. Figure 5 reveals some Gaussians fail to align with the correct areas with the absence of multi-view constraints. Introducing this component reduces view-dependent inconsistencies to a certain degree, further enhancing the reconstruction quality.

Since our method directly uses 2D Gaussians to represent scene surfaces, allowing the Gaussian splat to better adhere to the surface geometry, it outperforms 3DGS-based methods [9, 51]. Furthermore, while 2DGS [10] and some other methods [52, 53] that employ depth strategies do improve geometric reconstruction quality, they still struggle in indoor scenes due to the complexity of spatial structures and the prevalence of textureless regions. By integrating seedguided strategies and geometric constraints, our method enhances the accuracy of scene structure capture and achieves higher reconstruction quality, resulting in superior metrics.

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

As shown in Fig. 4, some methods [4, 51] produce noisy reconstructions with scattered floaters, and fail to represent the actual surfaces accurately due to the lack of geometric constraints. However, they may cover more ground truth data and thus achieve higher Accuracy than 2DGS on the ScanNet++ dataset in Table 1. Our method improves the structural coherence of the reconstruction, leading to a more accurate representation of the scene and a significant improvement in the Accuracy metric compared to 2DGS.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

###### 5.3. Ablation Studies

w/o Seed w/o Depth w/o Normal w/o MV Full Model

To assess the individual contributions of each component in our model, we perform ablation studies on the ScanNet dataset. The quantitative results are reported in Table 2 and

Figure 5. Qualitative results of ablation study.

##### 6. Conclusion

- Figure 5 shows the qualitative results. These allow us to isolate the impact of key elements on the overall reconstruction quality.

We propose 2DGS-Room, a novel method for indoor scene reconstruction based on 2D Gaussian splatting by incorporating structural information from the scene to generate seed points, which guide the local Gaussian distributions. By leveraging geometric priors, we enhance the reconstruction quality of textureless regions and fine details in complex indoor environments. We also utilize multi-view consistency to reduce view-dependent inconsistencies to a certain degree. Extensive experiments show our method achieves superior performance compared with existing methods on multiple metrics and various indoor scenes.

Method Acc.↓ Comp.↓ Prec.↑ Recall↑ F-score↑

w/o Seed 0.128 0.152 0.336 0.284 0.307 w/o Depth 0.084 0.139 0.510 0.386 0.438 w/o Normal 0.066 0.102 0.596 0.463 0.520 w/o MV 0.055 0.092 0.644 0.508 0.566 Full model 0.055 0.092 0.648 0.518 0.575

- Table 2. Results of the ablation study on ScanNet dataset. The best results are marked in bold.

##### References

- [1] Johannes L Sch¨onberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. Pixelwise view selection for unstructured multi-view stereo. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14, pages 501–518. Springer, 2016. 1, 2
- [2] Wenjie Luo, Alexander G Schwing, and Raquel Urtasun. Efficient deep learning for stereo matching. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5695–5703, 2016. 2
- [3] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured multi-view stereo. In Proceedings of the European conference on computer vision (ECCV), pages 767–783, 2018. 1, 2
- [4] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. NeurIPS, 2021. 2, 6, 7, 8
- [5] Jiepeng Wang, Peng Wang, Xiaoxiao Long, Christian Theobalt, Taku Komura, Lingjie Liu, and Wenping Wang. Neuris: Neural reconstruction of indoor scenes using normal priors. In European Conference on Computer Vision, pages 139–155. Springer, 2022. 2, 6
- [6] Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger. Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction. Advances in neural information processing systems, 35:25018–25032, 2022. 2, 6
- [7] Xinghui Li, Yikang Ding, Jia Guo, Xiansong Lai, Shihao Ren, Wensen Feng, and Long Zeng. Edge-aware neural implicit surface reconstruction. In 2023 IEEE International Conference on Multimedia and Expo (ICME), pages 1643–

1648. IEEE, 2023.

- [8] Xinghui Li, Yuchen Ji, Xiansong Lai, Wanting Zhang, and Long Zeng. Fine-detailed neural indoor scene reconstruction using multi-level importance sampling and multi-view consistency. In 2024 IEEE International Conference on Image Processing (ICIP), pages 3477–3483. IEEE, 2024. 2
- [9] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42

(4):1–14, 2023. 2, 3, 4, 6, 7, 8

- [10] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. arXiv preprint arXiv:2403.17888, 2024. 2, 3, 4, 6, 7, 8
- [11] Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B Goldman. Patchmatch: A randomized correspondence algorithm for structural image editing. ACM Trans. Graph., 28(3):24, 2009. 2
- [12] Silvano Galliani, Katrin Lasinger, and Konrad Schindler. Gipuma: Massively parallel multi-view stereo reconstruction. Publikationen der Deutschen Gesellschaft f¨ur Photogrammetrie, Fernerkundung und Geoinformation e. V, 25 (361-369):2, 2016.

- [13] Robust Multiview Stereopsis. Accurate, dense, and robust multiview stereopsis. IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, 32(8), 2010. 2
- [14] Michael Kazhdan and Hugues Hoppe. Screened poisson surface reconstruction. ACM Transactions on Graphics (ToG), 32(3):1–13, 2013. 2
- [15] Adrian Broadhurst, Tom W Drummond, and Roberto Cipolla. A probabilistic framework for space carving. In Proceedings eighth IEEE international conference on computer vision. ICCV 2001, pages 388–393. IEEE, 2001. 2
- [16] Jeremy S De Bonet and Paul Viola. Poxels: Probabilistic voxelized volume reconstruction. In Proceedings of International Conference on Computer Vision (ICCV), page 2. Citeseer, 1999.
- [17] Steven M Seitz and Charles R Dyer. Photorealistic scene reconstruction by voxel coloring. International journal of computer vision, 35:151–173, 1999.
- [18] Shaohui Liu, Yinda Zhang, Songyou Peng, Boxin Shi, Marc Pollefeys, and Zhaopeng Cui. Dist: Rendering deep implicit signed distance function with differentiable sphere tracing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2019–2028, 2020. 2
- [19] Benjamin Ummenhofer, Huizhong Zhou, Jonas Uhrig, Nikolaus Mayer, Eddy Ilg, Alexey Dosovitskiy, and Thomas Brox. Demon: Depth and motion network for learning monocular stereo. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5038–5047,

2017. 2

- [20] Sergey Zagoruyko and Nikos Komodakis. Learning to compare image patches via convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4353–4361, 2015.
- [21] Gernot Riegler, Ali Osman Ulusoy, Horst Bischof, and Andreas Geiger. Octnetfusion: Learning depth fusion from data. In 2017 International Conference on 3D Vision (3DV), pages 57–66. IEEE, 2017.
- [22] Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra Ahuja, and Jia-Bin Huang. Deepmvs: Learning multiview stereopsis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2821–2830, 2018.
- [23] Yao Yao, Zixin Luo, Shiwei Li, Tianwei Shen, Tian Fang, and Long Quan. Recurrent mvsnet for high-resolution multi-view stereo depth inference. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5525–5534, 2019.
- [24] Zehao Yu and Shenghua Gao. Fast-mvsnet: Sparse-todense multi-view stereo with learned propagation and gaussnewton refinement. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1949–1958, 2020.
- [25] Jingyang Zhang, Yao Yao, Shiwei Li, Zixin Luo, and Tian Fang. Visibility-aware multi-view stereo network. arXiv preprint arXiv:2008.07928, 2020. 2
- [26] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:

- Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2
- [27] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5855–5864,

2021. 2

- [28] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Pointnerf: Point-based neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5438–5448, 2022.
- [29] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-based neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19697–19705, 2023. 2
- [30] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1–15, 2022. 2
- [31] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. Advances in Neural Information Processing Systems, 33:15651–15663, 2020.
- [32] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5501–5510, 2022.
- [33] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In European Conference on Computer Vision, pages 333–350. Springer, 2022.
- [34] Zhaoshuo Li, Thomas M¨uller, Alex Evans, Russell H Taylor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin. Neuralangelo: High-fidelity neural surface reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8456–8465, 2023. 2, 6, 7, 8
- [35] Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised nerf: Fewer views and faster training for free. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12882– 12891, 2022. 2
- [36] Yi Wei, Shaohui Liu, Yongming Rao, Wang Zhao, Jiwen Lu, and Jie Zhou. Nerfingmvs: Guided optimization of neural radiance fields for indoor multi-view stereo. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5610–5619, 2021. 2
- [37] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5480–5490, 2022. 2

- [38] Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9065–9076,

2023. 2

- [39] Yixing Lao, Xiaogang Xu, Xihui Liu, Hengshuang Zhao, et al. Corresnerf: Image correspondence priors for neural radiance fields. Advances in Neural Information Processing Systems, 36, 2024. 2
- [40] Michael Niemeyer, Lars Mescheder, Michael Oechsle, and Andreas Geiger. Differentiable volumetric rendering: Learning implicit 3d representations without 3d supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3504–3515, 2020. 2
- [41] Michael Oechsle, Songyou Peng, and Andreas Geiger. Unisurf: Unifying neural implicit surfaces and radiance fields for multi-view reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5589–5599, 2021. 2
- [42] Lior Yariv, Yoni Kasten, Dror Moran, Meirav Galun, Matan Atzmon, Basri Ronen, and Yaron Lipman. Multiview neural surface reconstruction by disentangling geometry and appearance. Advances in Neural Information Processing Systems, 33:2492–2502, 2020. 2
- [43] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Volume rendering of neural implicit surfaces. Advances in Neural Information Processing Systems, 34:4805–4815, 2021. 2
- [44] Qiancheng Fu, Qingshan Xu, Yew Soon Ong, and Wenbing Tao. Geo-neus: Geometry-consistent neural implicit surfaces learning for multi-view reconstruction. Advances in Neural Information Processing Systems, 35:3403–3416, 2022. 2
- [45] Jingyang Zhang, Yao Yao, Shiwei Li, Tian Fang, David McKinnon, Yanghai Tsin, and Long Quan. Critical regularizations for neural surface reconstruction in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6270–6279, 2022. 2
- [46] Haoyu Guo, Sida Peng, Haotong Lin, Qianqian Wang, Guofeng Zhang, Hujun Bao, and Xiaowei Zhou. Neural 3d scene reconstruction with the manhattan-world assumption. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5511–5520, 2022. 2
- [47] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In CVPR, pages 4104–4113, 2016. 3
- [48] Matias Turkulainen, Xuqian Ren, Iaroslav Melekhov, Otto Seiskari, Esa Rahtu, and Juho Kannala. Dn-splatter: Depth and normal priors for gaussian splatting and meshing. arXiv preprint arXiv:2403.17822, 2024. 3
- [49] Haodong Xiang, Xinghui Li, Xiansong Lai, Wanting Zhang, Zhichao Liao, Kai Cheng, and Xueping Liu. Gaussianroom: Improving 3d gaussian splatting with sdf guidance and monocular cues for indoor scene reconstruction. arXiv preprint arXiv:2405.19671, 2024. 3
- [50] Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xiangli, and Bo Dai. Gsdf: 3dgs meets sdf for improved rendering and reconstruction. arXiv preprint arXiv:2403.16964, 2024. 3

- [51] Antoine Gu´edon and Vincent Lepetit. Sugar: Surfacealigned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. arXiv preprint arXiv:2311.12775, 2023. 3, 6, 7, 8, 2
- [52] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction. arXiv preprint arXiv:2406.06521, 2024. 3, 6, 7, 8
- [53] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, and Ping Tan. Rade-gs: Rasterizing depth in gaussian splatting. arXiv preprint arXiv:2406.01467, 2024. 3, 6, 7, 8, 2
- [54] Aleksei Bochkovskii, Ama¨el Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. arXiv preprint arXiv:2410.02073, 2024. 5
- [55] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020. 5
- [56] Gwangbin Bae, Ignas Budvytis, and Roberto Cipolla. Estimating and exploiting the aleatoric uncertainty in surface normal estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13137–13146,

2021. 5

- [57] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017. 6, 1
- [58] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023. 6, 1
- [59] Jae-Chern Yoo and Tae Hee Han. Fast normalized crosscorrelation. Circuits, systems and signal processing, 28:819– 843, 2009. 6
- [60] Brian Curless and Marc Levoy. A volumetric method for building complex models from range images. In Proceedings of the 23rd annual conference on Computer graphics and interactive techniques, pages 303–312, 1996. 6

## 2DGS-Room: Seed-Guided 2D Gaussian Splatting with Geometric Constrains for High-Fidelity Indoor Scene Reconstruction

### Supplementary Material

In this supplementary material, we provide the following components:

- • Definitions of the 3D geometry metrics used to evaluate reconstruction quality in Sec. A.
- • Additional details of the datasets, training configuration, and the iteration schedule for key modules in Sec. B.
- • Additional qualitative results, including mesh comparison, ablation results, and rendering comparison in Sec. C.

##### A. Definitions of Eevaluation Metrics

Training details. For all scenes, our seed-guided optimization is performed between 1,500 and 15,000 iterations. We set Ng = 100 for the gradient-guided growth and Nα = 100 for the pruning strategy. Depth supervision and normal supervision are applied consistently from the first iteration through to the end of training, providing continuous geometric constraints. The multi-view consistency constraint is introduced after 7,000 iterations, once the foundational structure has been established, to further improve view alignment.

We evaluate our method using five widely-used 3D geometry metrics: Accuracy, Completion, Precision, Recall, and F-score, defined in Table 3. These metrics collectively assess the geometric fidelity of the reconstructed point clouds by measuring the alignment between the predicted and ground truth point clouds.

##### C. Additional Qualitative Results

###### C.1. Additional Ablation Results

To complement the local detail comparisons in the main paper, we provide additional ablation results focusing on the overall scene structure in Figure 6. These visualizations highlight the contributions of key components, including the seed points guidance, monocular depth supervision, and monocular normal supervision. The multi-view consistency constraints are primarily designed to further mitigate floating artifacts in certain scenarios, which have a limited impact on the overall structure. Therefore, they are not included in these structural comparisons. Their effectiveness is instead reflected in the qualitative results shown in Figure 5 and the quantitative metrics presented in Table 2 of the main paper.

Accuracy measures the average distance between reconstructed points and the ground truth, with smaller values indicating better alignment. Completion assesses how well the reconstruction covers the ground truth, where lower values are better. Precision and Recall evaluate the proportion of points within a set threshold, with higher values indicating better performance. F-score, the harmonic mean of Precision and Recall, provides a balanced measure of reconstruction quality, where higher values reflect superior results.

When the seed points guidance strategy is removed, the reconstructed objects appear fused together, with unclear boundaries, compromising the scene’s structural clarity.

Metric Definition

Acc. meanc∈C(minc∗∈C∗ ||c − c∗||) Comp. meanc∗∈C∗(minc∈C ||c − c∗||) Prec. meanc∈C(minc∗∈C∗ ||c − c∗|| < .05) Recall meanc∗∈C∗(minc∈C ||c − c∗|| < .05) zoF-score 2×Prec×Recall

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Prec+Recall

Table 3. Definitions of 3D metrics. c and c∗ are the predicted and ground truth point clouds.

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

##### B. Additional Implementation Details

Datasets. As described in the main paper, the quantitative evaluation metrics are derived from results tested two datasets. Specifically, we select 8 scenes from the ScanNet dataset [57]: scene0050 00, scene0085 00, scene0114 02, scene0580 00, scene0603 00, scene0616 00, scene0617 00, scene0721 00, and 4 scenes from the ScanNet++ dataset [58]: 8b5caf3398, 8d563fc2cc, 41b00feddb, b20a261fdf.

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

w/o Seed w/o Depth w/o Normal Full Model

###### Figure 6. Additional qualitative results of ablation study.

###### C.3. Rendering Comparison

Without depth supervision, objects exhibit depth misalignments, leading to unrealistic spatial arrangements. Similarly, excluding normal supervision results in uneven surfaces, especially on planar regions like walls, where visible curvature or misalignment artifacts occur.

We also provide extensive rendering results comparing our 2DGS-Room with 2DGS across various scenes and viewpoints from the ScanNet and ScanNet++ datasets in Figures 8, 9, and 10. Rendered RGB, depth, and normal maps are shown for visual comparison. Our method achieves significant improvements in the rendering quality of depth and normal maps, showcasing smoother transitions and more accurate surface details. Furthermore, the quality of the RGB images rendered by our method remains robust and shows clear advantages over 2DGS in challenging scenarios, such as handling fine details and varying lighting conditions. This demonstrates the effectiveness of our method in achieving superior geometric reconstructions while maintaining photometric accuracy.

###### C.2. Additional Qualitative Comparison

In addition to the four indoor scenes shown in the main paper, we further include qualitative reconstruction comparison results of the different methods [4, 10, 51–53] on additional scenes from ScanNet and ScanNet++. As demonstrated in Figure 7, our method significantly outperforms other approaches in capturing global structures, preserving fine-grained details as well as reducing artifacts in textureless regions.

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

[Figure 196]

[Figure 197]

###### NeuS SuGaR RaDe-GS PGSR 2DGS Ours GT

Figure 7. Additional qualitative reconstruction comparison. For each indoor scene, the first row is the top view of the whole room and the second row is the details of the masked region.

2DGS Ours

2DGS Ours

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

NormalRGB

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

DepthRGBNormalDepth

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

###### Figure 8. Rendering comparison on the ScanNet dataset (scene0580 and scene0050).

2DGS Ours

2DGS Ours

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

NormalRGB

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

DepthRGBNormalDepth

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

###### Figure 9. Rendering comparison on the ScanNet dataset (scene0085 and scene0617).

2DGS Ours

2DGS Ours

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

NormalRGB

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

DepthRGBNormalDepth

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

###### Figure 10. Rendering comparison on the ScanNet++ dataset (8d563fc2cc and 41b00feddb).

