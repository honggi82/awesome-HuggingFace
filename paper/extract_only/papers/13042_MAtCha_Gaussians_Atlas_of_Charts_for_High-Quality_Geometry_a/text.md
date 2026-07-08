# arXiv:2412.06767v1[cs.CV]9Dec2024

## MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views

Antoine Gu´edon† Tomoki Ichikawa‡ Kohei Yamashita‡ Ko Nishino‡ † LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, France ‡Graduate School of Informatics, Kyoto University, Japan

antoine.guedon@enpc.fr {tichikawa,kyamashita}@vision.ist.i.kyoto-u.ac.jp

kon@i.kyoto-u.ac.jp https://anttwo.github.io/matcha/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

MAtCha Gaussians from 3 views MAtCha Gaussians from 10 views MAtCha Gaussians from 10 views

Figure 1. We propose MAtCha Gaussians, a novel surface representation for reconstructing high-quality 3D meshes with photorealistic rendering from sparse-view images. Our key idea is to model the underlying scene geometry as an Atlas of Charts in 2D image planes, which we render with 2D Gaussian surfels. We initialize the charts with a monocular depth estimation model and refine them using differentiable Gaussian rendering and a lightweight neural chart deformation model. Combined with a sparse-view SfM model like MASt3R-SfM [15], MAtCha can recover sharp and accurate surface meshes of both foreground and background objects in unbounded scenes within minutes, from a few unposed RGB images. We used 3 views for training for the left most, and 10 views for the rest.

#### Abstract

We present a novel appearance model that simultaneously realizes explicit high-quality 3D surface mesh recovery and photorealistic novel view synthesis from sparse view samples. Our key idea is to model the underlying scene geometry Mesh as an Atlas of Charts which we render with 2D Gaussian surfels (MAtCha Gaussians). MAtCha distills high-frequency scene surface details from an off-the-shelf monocular depth estimator and refines it through Gaussian surfel rendering. The Gaussian surfels are attached to the charts on the fly, satisfying photorealism of neural volumetric rendering and crisp geometry of a mesh model, i.e., two

seemingly contradicting goals in a single model. At the core of MAtCha lies a novel neural deformation model and a structure loss that preserve the fine surface details distilled from learned monocular depths while addressing their fundamental scale ambiguities. Results of extensive experimental validation demonstrate MAtCha’s state-of-theart quality of surface reconstruction and photorealism onpar with top contenders but with dramatic reduction in the number of input views and computational time. We believe MAtCha will serve as a foundational tool for any visual application in vision, graphics, and robotics that require explicit geometry in addition to photorealism.

#### 1. Introduction

The neural revitalization of volume rendering has revolutionalized novel view synthesis of real-world scenes. Neural Radiance Fields (NeRF) [35], 3D Gaussian Splatting [26] and their many variants provide reliable means to extract appearance representations of intricate scenes that used to be out of our reach, e.g., a flower bouquet. The volume rendering formulation is essential for this as it offers reliable gradients for end-to-end neural function fitting.

The learned NeRF or Gaussians are, however, fundamentally trained for photorealism in the end results, i.e., 2D images. The physical scene representation that one can tease out from these neural representations are only crude approximations of reality. Most notable, the geometry is blurry as it is accurate only up to what is necessary for volume rendering. For instance, it is easy for the networks to distribute the view-dependent reflectance of a surface with multiple 3D Gaussians of unique colors scattered around the surface. The same applies to NeRF, too.

Explicit extraction of the scene geometry would be of natural importance for many applications in vision, robotics, and graphics, as interaction with the scene including editing is often the end goal. Extraction of explicit geometry from learned neural representations has typically been achieved by applying algorithms such as TSDF [13], marching cubes [33] or screened Poisson surface reconstruction [25] to the learned volumetric representation. This, however, fundamentally forces a sequential process, in which a volumetric appearance model first needs to be learned. This is suboptimal in two fundamental ways.

The first is that it likely requires view samples (i.e., input images) more than necessary. The dense view sampling for NeRF and Gaussian Splatting are essential to learn volumetric representations for photorealistic view synthesis, but for geometry reconstruction our rich literature in computer vision suggests that much less should suffice. The second is that, even after this sequential process, we cannot recover accurate scene geometry, especially pertaining to its highfrequency changes, e.g., sharp corners and edges, as it is extracted from an already low-pass filtered representation due to the volume rendering formulation.

In this paper, we ask, can we learn a neural scene representation that realizes photorealistic rendering, on par with leading volumetric representations, but at the same time enables conversion to accurate and sharp scene geometry? Better yet, can we achieve this within minutes and from a much smaller number of images? We answer these fundamental questions in the affirmative with a novel neural appearance model which we refer to as MAtCha Gaussians or MAtCha for short. MAtCha stands for Mesh as an Atlas of Charts. It models the surface as a 2D manifold in 3D space, i.e., an atlas of n charts.

MAtCha offer three key benefits as a learnable scene rep-

resentation. First, it can be initialized with a monocular depth estimator. This enables distillation of high-frequency surface details from a pre-trained model. Second, it lets us perform surface refinement in 2D rather than 3D, which can be achieved efficiently with a lightweight neural deformation model. This enables fast and stable scene optimization from just a few images. Finally, it serves as a basis for photorealistic rendering with 2D Gaussian surfels aligned with the charts on the fly. Gaussian surfel rendering provides better gradients compared with mesh rendering for geometry refinement as well as novel view synthesis even with a limited number of input views.

MAtCha is initialized with depth estimates from learned monocular models. Monocular depth estimators, however, suffer from scale ambiguities. Scale variation across views lead to erroneous 3D scene structure, although its derivatives are unaffected. We address this depth ambiguity by introducing a neural deformation model that deforms viewdependent depth estimates to match and align while preserving the high-frequency details. We achieve this with a tiny MLP that takes in a feature vector sampled from a sparse 2D grid in the image space (charts encodings) and a depth dependent feature (depth encodings). The sparsity of the 2D grid ensures that the MLP deforms only lowfrequency scene structure for regions with similar depths.

Given a sparse set of RGB images and charts initialized by a monodepth model, we first optimize the neural deformation model for each chart using surface points recovered with structure-from-motion (SfM). These charts are then refined by differentiable Gaussian rendering and a photometric loss. To further preserve the high-frequency information in the monodepth estimates, we impose a structure loss that encourages the deformed charts to maintain the normals and curvatures computed from the derivatives of the initial depth. After this refinement, a unified surface mesh of both foreground and background of the scene can be recovered from our geometrically accurate charts using our two custom methods, multi-resolution TSDF fusion and adaptive tetrahedralization.

We validate the effectiveness of MAtCha with an extensive set of experiments. We demonstrate that it can recover accurate scene geometry from sparse RGB images within minutes. This is in contrast to current state-of-the-art methods that require dense view sampling and long training time. We also show that MAtCha Gaussians can render high-quality images from novel viewpoints in the sparseview scenarios where existing sparse-view Gaussian Splatting methods suffer from little overlap of views. Ablation studies show that our proposed deformation model is crucial for accurate surface reconstruction.

We believe MAtCha Gaussians seamlessly integrates our rich history of 3D geometry reconstruction research into cutting-edge neural representations for appearance model-

ing and can serve as a foundational tool for a wide range of downstream application domains in vision, graphics, robotics, and beyond.

#### 2. Related Work

Tab. 1 summarizes the key contributions of our method in comparison with previous novel view synthesis and surface reconstruction methods.

Structure from Motion Structure-from-Motion (SfM) [32, 41, 42] estimates extrinsic and intrinsic camera parameters and reconstructs a sparse 3D point cloud from uncalibrated multi-view images. Recent differentiable SfM methods [5, 15, 29, 45, 46] demonstrate impressive results on complex real-world scenes. SfMreconstructed point clouds are, however, sparse and do not have sufficient high-frequency details as the underlying formulation fundamentally relies on multi-view correspondences. This is fatal for photorealistic rendering and accurate surface reconstruction. SfM can instead provide strong initializations for these tasks.

Novel View Synthesis NeRF [35], 3D Gaussian Splatting [26], and their derivatives [2, 4, 7, 18, 36, 55] achieve impressive photorealism in novel view synthesis. They optimize a representation based on a neural implicit function or a set of Gaussian primitives with differentiable volume rendering. These methods require very dense view samples to learn an accurate scene representation. Recent works have introduced various approaches for learning these representations from sparser views, for instance by applying regularizations on 3D Gaussians [17, 24, 30, 37, 47, 53, 58, 61] or by training a feed-forward deep network to directly estimate 3D Gaussian parameters [6, 11, 49]. These methods, however, are focused on novel view synthesis and surfaces extracted from them remain inaccurate and noisy.

Surface Reconstruction from RGB images Recent image-based surface reconstruction methods also leverage differentiable volume rendering [8–10, 14, 19, 20, 22, 34, 38, 43, 52, 54, 56, 57, 59]. For instance, Gaussian Surfels [14] and 2D Gaussian Splatting (2DGS) [22] use flat 2D Gaussians instead of 3D Gaussians to represent the surface accurately. These methods, however, require dense view sampling to constrain the millions of tiny Gaussians.

A few methods handle sparse inputs. SparseNeus [31], VolRecon [40], and S-VolSDF [48] exploit pretrained feed forward networks for multi-view inputs. NeuSurf [23] first reconstructs global structures from an SfM point cloud, then refines local geometry by fitting a signed distance function (SDF) with local feature consistency. Spurfies [39] leverages local priors from a pretrained geometry decoder for optimizing an SDF.

Multi-view feed-forward networks and geometry decoders, however, struggle to generalize to unseen, unbounded scenes, as they are trained with a limited number of object-centric or synthetic datasets. Additionally, the optimization of a signed distance field might cause loss of geometry details due to ineffective constraints on the 3D volumetric representation. To the best of our knowledge, no existing method can reconstruct sharp meshes of unbounded scenes from sparse input views.

MAtCha Gaussians fills this hole. Its explicit surface representation exploits and preserves local geometry details obtained from a monocular depth estimation model for fast and sharp surface reconstruction from sparse-view images.

#### 3. Preliminaries

Let us first recall some preliminaries for surface representations.

Gaussian Splatting Gaussian Splatting [26] and its derivatives model 3D scenes with large collections of tiny, smooth 3D ellipsoids. They realize efficient rasterizationbased volume rendering which enables robust learning from RGB images. More recently, their 2D variants, namely 2D Gaussian surfel representations have been proposed [14, 22]. All these representations obviously have an extremely large degree of freedom. Each of the numerous Gaussians is free to slightly move away from the true surface to approximate view-dependent appearance. This fundamental redundancy causes even the most recent methods to struggle to recover accurate surfaces from sparse-view images.

Surfaces as 2D Manifolds Let M be a subset of R3, typically a surface. A chart (U,ϕ) on M consists of an open subset U ⊂ M equipped with a homeomorphism ϕ : U → R2. In other words, a chart is a continuous bijection between U and an open subset of R2, implying that the subset U of M can be represented by continuously deforming a flat surface patch.

The set M is called a 2-dimensional manifold in R3, if there exists a collection of n charts (Ui,ϕi)0≤i<n covering M, i.e., such that i Ui = M. The collection of charts (Ui,ϕi)i is called an atlas on M. Intuitively, a 2dimensional manifold is a subset of R3 that can be represented by deforming and patching together a collection of planar pieces. In this paper, we will denote charts by ϕi : Ui → Vi where Vi ⊂ R2, and their inverse mapping by ψi := ϕ−i 1 : Vi → Ui.

A UV map is an example of a chart which is widely-used for representing the texture of a surface. Given a texture image, a UV map maps every vertex of the mesh to a single point in the image, allowing for texturing the 3D surface by looking up the 2D texture.

Fast training (<15min) Gaussian Surfel [14], 2DGS [22], GOF [56] ✓ ✓

Method Sparse view Surface reconstruction

Explicit surface optimization

Reconstruction of unbounded scenes

SuGaR [20], Gaussian Frosting [19] ✓ ✓

InstantSplat [17] ✓ ✓ ✓ SparseNeus [31], VolRecon [40] ✓ ✓ ✓

S-VolSDF [48], NeuSurf [23], Spurfies [39] ✓ ✓

Ours (MAtCha Gaussians) ✓ ✓ ✓ ✓ ✓

- Table 1. Comparisons between our method and existing methods for image-based 3D reconstruction. Our method achieves fast reconstruction of unbounded scene mesh from sparse-view images by directly optimizing explicit surface manifolds.

#### 4. MAtCha Gaussians

Let us derive MAtCha Gaussians, an appearance model learnable from a sparse set of N RGB images, from which a detailed surface mesh can be extracted. MAtCha is a 2D manifold equipped with Gaussians. Specifically, we model the surface of the scene as a collection of n ≤ N charts, each chart corresponding to one of the input views. There are three key benefits of using this representation, particularly with sparse views.

First, we can directly initialize the charts by using detailed depth maps computed with a pre-trained monocular depth estimation model (monodepth model) [50, 51] and explicitly distill the high-frequency geometry captured by the depth maps into our representation. Second, it allows us to optimize the surface with 2D deformation maps instead of dense 3D grids, resulting in a significantly more efficient optimization. It is also more robust, as we can leverage a lightweight neural deformation model on the 2D map to efficiently constrain the geometry for sparse-view surface reconstruction. Finally, we can refine our explicit surface representation with differentiable volumetric Gaussian rendering by instantiating 2D Gaussian surfels aligned with the charts on the fly, which enables efficient refinement of the manifold and photorealistic rendering. Reciprocally, our charts explicitly constrain Gaussians and prevent them from diverging even with very sparse view samples.

Fig. 2 depicts the overall pipeline of MAtCha Gaussians. Given a sparse set of N RGB images Ii and their corresponding cameras ci with extrinsics (Ri,ti) ∈ SO(3)×R3 and intrinsics Ki ∈ R3×3, we optimize a set of charts so that it approximates the true geometry of the scene. Note that, in practice, we can obtain the camera parameters even for sparse unposed images using a sparse structure-frommotion (SfM) method [15]. We first initialize the charts using a monodepth model. Then, we align the charts with surface points recovered by the SfM method using our novel deformation model. Finally, we refine the charts with differentiable rendering based on Gaussian surfels.

##### 4.1.ChartInitializationwithMonoDepthEstimates

For a given number of charts n ≤ N, we initialize the charts by using depth maps (Di)0≤i<n estimated with a pre-trained monodepth model [51] from the input views Ii.

A depth map is indeed a mapping from a 2D plane to 3D scene points, making it a natural candidate for representing a chart. In practice, in sparse view scenarios, we set n = N and backproject all depth maps to 3D space.

We denote the initial chart constructed by backprojecting the depth map Di to 3D space with ψi(0) : Vi → Ui(0). The mapping ψi(0) maps the 2D UV coordinates in Vi ⊂ [0,1]2 to some 3D points in Ui(0) ⊂ R3, where Ui(0) is supposed to be a subset of the true surface of the scene.

The relative scales of these initial charts computed from monodepth results are, however, generally inaccurate. Simply backprojecting depth maps into 3D space results in a chaotic, unaligned manifold and does not provide an accurate enough initial estimate of the true surface. We can try to estimate the relative scales between the initialized charts and the true surface, by exploiting surface points recovered by the SfM method. Previous works [27, 43] model the discrepancies with simple models such as a global affine rescaling of the depth maps. Although easy to compute, such an approach inevitably results in poor accuracy due to differences in the relative scales between objects. On the other hand, pixel-wise scaling does not work in sparse view scenarios as its over-parameterization will cause loss in high-frequency of the geometry.

##### 4.2. Lightweight Chart Deformation Model

To refine and align the initialized charts by resolving the complex object-dependent scaling while preserving the high-frequency information from the monodepth estimates, we introduce a novel deformation model based on what we refer to as chart encodings.

Chart Encoding For deforming each chart i, we maintain 1) a sparse 2D grid of learnable features Ei ∈ Rrh×rw×d in UV space, where r is a size ratio, h and w are the height and width of the depth map, and d is the feature dimension; and 2) a tiny MLP fθ

: Rd → R3 that maps these features to 3D deformation vectors.

i

The deformation field for chart i at UV coordinate u is

∆i(u) = fθ

i

[Ei(u)] , (1)

where Ei(u) bilinearly interpolates features from the sparse grid at coordinate u. The deformed inverse map ψi which

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Monodepth SfM data Images

Depth Maps and Initial Charts Aligned Charts Refined Charts equipped with Gaussians

Sparse RGB Images

###### Optimizing Charts with a Robust Deformation Model

###### Rendering Charts with 2D Gaussian Splatting

[Figure 17]

[Figure 18]

[Figure 19]

MLP

[Figure 20]

Charts encodings

+

###### +

[Figure 21]

[Figure 22]

Deformed Charts

We combine our deformation model with Gaussians instantiated on the surface to refine the geometry.

Depth encodings

Initial Charts

- Figure 2. Overview of MAtCha Gaussians. Given a few RGB images and their camera poses obtained using a sparse-view SfM method such as MASt3R-SfM [15], we first initialize charts using a pretrained monocular depth estimation model. Each chart is represented as a mesh equipped with a UV map, mapping a 2D plane to the 3D surface. We then optimize our charts and enforce their alignment with input SfM data using two key components: (1) 1D depth encodings for quickly aligning the initial depth maps together, and (2) charts encodings for efficiently deforming the geometry while preserving surface details. Our aligned charts provide a sharp, dense and accurate estimate of the 3D scene, which can be further refined using input images and a Gaussian Splatting-based rendering pipeline. Our representation allows for reconstructing high-quality surface meshes within minutes, even in sparse-view scenarios.

maps the UV coordinates in Vi to 3D points on the updated surface Ui becomes

###### ψi(u) = ψi(0)(u) + ∆i(u). (2)

The sparsity of the 2D feature grid encourages the 2D deformation field to contain only low-frequency deformation, i.e., the high-frequency structures in the initial charts are preserved during the optimization.

Depth Discontinuities The deformation field, however, needs to be discontinuous at contours of the objects, due to inconsistent scales between objects in the initial charts.

To model such discontinuities, we augment our chart encodings with an additional depth-dependent feature which we refer to as depth encodings. For each UV coordinate u, we compute an encoding zi(d(u)) that depends only on the initial depth value of the pixel d(u) = (Pi ◦ ψi(0)(u))z (i.e., the z-component of the backprojected point), where Pi is the function transforming 3D points to the coordinate frame of the depth map Di. These features are stored along the depth axis and interpolated depending on the depth of the point. The complete feature vector used for deformation becomes

∆i(u) = fθ

i

[Ei(u) + zi(d(u))] , (3)

for any 2D point u in the UV space. The feature zi(d(u)) acts as positional encoding that allows points at different depths to be deformed independently, even if they are close

in the UV space. The depth encoding helps disambiguate spatial relationships that are not captured by the 2D chart encoding alone, leading to more accurate surface reconstruction. It also acts as a useful prior by enforcing points with similar depths to be deformed similarly, which is important particularly for sparse view samples. Combining features stored in sparse 2D grids and along the depth axis has two main advantages. It requires less memory than storing a full 3D grid of features as it has quadratic space complexity, and simultaneously makes the deformation model more robust to sparsity in the input data.

##### 4.3. Aligning the Manifold with SfM Points

With our neural deformation model, we first optimize its weights to make our charts fit as much as possible with the SfM surface points, while maintaining the detailed structure as originally captured by the depth maps. We also encourage the charts to align together to form a coherent and unified manifold. We achieve this with the following losses.

Fitting loss Lfit We encourage the charts to fit the SfM points by minimizing the distance between the SfM points and the deformed charts. Specifically, for each chart, we project the SfM points visible in image i to the UV space of the chart i, and we minimize the distance between the SfM points and the corresponding points on the chart:

Lfit =

mi−1

n−1

∥ψi(uik) − pik∥1 , (4)

i=0

k=0

where uik is the UV coordinate of the k-th SfM point visible in image i, and pik is the 3D position of the k-th SfM point visible in image i.

In practice, we cannot just rely on a simple fitting loss, as the SfM points may contain outliers. To address this issue, we introduce for each chart i a learnable confidence map Ci ∈ [0,+∞)h×w that indicates the regions of the chart that are likely to be located on the true surface of the scene. If the optimization struggles to fit the chart to the SfM points, the confidence map will automatically adjust to downweight the loss in regions where the optimization struggles. Such regions are likely to contain the outlier SfM points. Specifically, we take inspiration from DUSt3R [46] and use a revised fitting loss

mi−1

n−1

n−1

1 n

Ci(uik)∥ψi(uik)−pik∥1−α

Lfit =

log(Ci),

i=0

i=0

k=0

(5) where the second term is a regularization term [44] and α is a hyperparameter. We compute the confidence map as Ci = 1 + exp(Cˆi), where Cˆi are optimizable parameters.

Structure loss Lstruct We also explicitly encourage the charts to maintain the same sharp structure as the initial depth maps by minimizing the distance between the first and second order derivatives of both the depth maps and the charts. Rather than using all explicit derivatives, we rely on normal and mean curvature regularization, acting respectively on first and second order derivatives

n−1

n−1

1 4

1 − Ni · Ni(0) +

∥Mi − Mi(0)∥1 ,

Lstruct =

i=0

i=0

(6) where Ni and Mi are the normal and mean curvatures of chart i computed following [14, 22], and Ni(0) and Mi(0) are those of the initial depth map of chart i, respectively.

Mutual alignment loss Lalign We encourage the charts to align together to form a coherent manifold by minimizing the distance between neighboring points located on different charts. Specifically, for each chart i, we project the points located on the charts into the screen space of other charts j. If the corresponding point in j is close enough to the chart i, these points are likely to be on the same surface so we minimize the distance between them. Overall, we compute the mutual alignment loss

n−1

Lalign =

min(∥ψi(u) − ψj ◦ Pj ◦ ψi(u)∥1,τ) ,

i,j=0 u∈Vi

(7) where τ is an attraction hyperparameter controlling the maximum distance between points on different charts for

considering them to be on the same surface. In contrast to the fitting loss, which may rely only on a sparse set of points depending on the SfM back-end used, the alignment loss acts as a dense regularization on the full surface, helping to form a coherent manifold.

Our complete optimization loss for aligning the charts is L = Lfit + λstructLstruct + λalignLalign , (8)

with λstruct = 4 and λalign = 5. This alignment step is very fast and generally takes less than a few minutes to converge. The charts, however, may still struggle to perfectly align with fine structures. We resolve this by further refining the charts with differentiable rendering.

##### 4.4. Refining the Manifold with Gaussian Surfels

We refine our manifold representation with a photometric rendering loss. We instantiate 2D Gaussian surfels on the fly to texture our charts and render them with a Gaussian surfel rasterizer [22]. Specifically, we learn color and opacity textures for each chart. Since we know the UV coordinates of any Gaussian we instantiate on the charts, we can use our textures to compute color and opacity values for all Gaussians. All other parameters of the Gaussians, such as positions and covariances, are not learnable and computed on the fly depending on the position of the vertices.

We use Gaussian surfels, instead of triangle rasterization, because once splat in the screen space, the support of a rasterized Gaussian is larger than the visible ellipsoid and covers neighboring pixels. Rendering charts with Gaussians realizes better propagation of gradients across the different pixels, in contrast to triangle rasterization which would require blurring on the rendering to help propagate gradients. In this regard, Gaussian surfels could be considered as local kernels performing adaptive Gaussian blurring dependent on the size of the triangles.

We exploit a conventional photometric loss [26], a regularization term from 2DGS [22], and the structure loss for this refinement. We weight the structure term using our confidence maps Ci for depth regularization robust to outliers. Please refer to the appendix for more details. After refinement, we can extract a single-piece mesh from our manifold.

##### 4.5. Extracting Meshes from Gaussian Surfels

Most existing surface reconstruction methods relying on 3D Gaussians or Gaussian Surfels [14, 22] apply TSDF fusion on rendered depth maps to extract a mesh from the volumetric representation. However, as observed in [56], TSDF fusion is limited to bounded scenes and does not allow for extracting high-quality meshes including both foreground and background objects of the scene. Moreover, applying TSDF fusion on Gaussian Surfels can over-smooth the geometry, erode fine details, and produce artifacts, such as

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

3 views 3 views 5 views 5 views 10 views 10 views

- Figure 3. Reconstruction with different numbers of input views. Our method can produce high-quality renderings (top) and surfaces (bottom) even with very sparse input views (3-10 views). The quality of our meshes is visually pleasing even in extreme sparse scenarios.

[Figure 35]

[Figure 36]

Multi-resolution TSDF Adaptive tetrahedralization

[Figure 37]

[Figure 38]

- Figure 4. Comparison between our two different mesh extraction methods: Multi-resolution TSDF fusion (left), and Adaptive tetrahedralization (right). We optimized MAtCha Gaussians representations with only 10 training images. Contrary to vanilla TSDF fusion, our multi-resolution TSDF can reconstruct both foreground and background objects with a decent number of vertices. However, similarly to vanilla TSDF fusion, it produces eroded meshes with holes in the surface, as well as “disk-aliasing” artifacts. On the contrary, our adaptive tetrahedralization inspired by GOF [56] is able to reconstruct accurate and complete surfaces meshes (see top right image), with sharp and fine details (see bottom right image).

resolution TSDF fusion including foreground and background objects in our implementation, we also propose to adapt the tetrahedralization from GOF [56] to make it compatible with any Gaussian-based method capable of rendering perspective-accurate depth maps.

First, we propose to change the definition of the opacity field, using depth maps instead of 3D Gaussians as in GOF: For any set of input depth maps, we define a binary opacity field from the depth maps as well as an adaptive dilation operation to avoid eroding geometry during mesh extraction. Second, because the tetrahedralization introduced in GOF generally produces very large meshes with more than 10M vertices, we propose a new sampling strategy to build the initial tetrahedron grid to easily adjust or lower the resolution of the output mesh. Please refer to our implementation for more details. A qualitative comparison of mesh extraction methods is available in Fig. 4.

We believe that our adaptation of GOF tetrahedralization provides a high-quality alternative to TSDF fusion and can generalize to most Gaussian-based surface reconstruction methods.

#### 5. Experiments

We focus on two different tasks to thoroughly evaluate the effectiveness of MAtCha Gaussians: surface mesh reconstruction from sparse RGB images, and novel view synthesis from sparse RGB images. We evaluate our method in both bounded and unbounded environments.

“disk-aliasing” patterns on the surface. To solve this issue and generate meshes with better quality, Gaussian Opacity Fields [56] proposes to use an adaptive tetrahedralization to extract a surface mesh with optimal resolution, where the local density of the SDF grid depends on the positions and the size of the closest Gaussians. The SDF values used during the tetrahedralization are derived from an opacity field, which is defined using 3D Gaussians and GOF [56] rasterization. Because the opacity field relies on GOF’s specificities to be computed, the tetrahedralization cannot be directly applied on different Gaussian-based representations, such as Gaussian surfels.

Implementation Details We implement our method in PyTorch and conduct experiments on a single NVIDIA RTX A6000 GPU. For all experiments, we initialize our manifold charts using DepthAnythingV2 [51] as the monodepth model, and use MASt3R-SfM [15] for camera pose estimation in sparse-view scenarios. For reconstructing highquality surfaces from 3 to 10 input views, our pipeline takes less than 3 minutes for aligning the charts, and between 5 and 10 minutes for refinement.

In this regard, while we propose a custom multi-

###### Scan ID 21 24 34 37 38 40 82 106 110 114 118 Mean CD

Points2Surf [16] 3.73 2.85 2.55 5.13 3.85 2.41 2.30 3.95 3.33 2.37 2.84 3.21 CAP-UDF [60] 2.72 1.53 1.45 4.05 2.78 1.81 4.22 3.51 3.83 2.24 3.65 2.89

NeuS [38] 4.52 3.33 3.03 4.77 1.87 4.35 1.89 4.18 5.46 1.09 2.40 3.36 VolSDF [48] 4.54 2.61 1.51 4.05 1.27 3.58 3.48 2.62 2.79 0.52 1.10 2.56

SuGaR [20] 2.71 2.04 2.14 4.01 2.90 2.45 4.68 3.82 3.28 2.44 2.66 3.01 2D Gaussian Splatting [22] + MASt3R-SfM [15] 1.43 1.29 2.02 2.79 2.05 1.71 2.24 1.23 2.26 0.85 1.72 1.79 Gaussian Opacity Fields [56] + MASt3R-SfM [15] 1.71 1.37 1.41 2.38 1.59 2.05 2.20 1.62 1.99 1.21 1.81 1.76

SparseNeus [31] 3.73 4.48 3.28 5.21 3.29 4.21 3.30 2.73 3.39 1.40 2.46 3.41 VolRecon [40] 3.05 3.30 2.27 4.36 2.51 3.24 3.30 3.10 3.58 1.86 3.68 3.11 S-VolSDF [48] 3.18 2.95 2.19 3.40 2.30 2.69 2.69 1.60 1.48 1.21 1.16 2.26 NeuSurf [23] 3.22 2.42 1.38 2.61 1.72 3.46 2.68 1.44 2.42 0.61 0.87 2.08 Spurfies [39] 2.36 1.12 0.83 2.39 1.14 1.55 1.67 1.26 1.14 0.61 0.94 1.36

MAtCha Gaussians (Ours) MAtCha Gaussians (Ours) 1.27 0.88 0.85 1.89 1.08 1.06 1.15 0.89 0.87 0.58 0.89 1.04

- Table 2. Quantitative evaluation of surface reconstruction in a sparse-view scenario (3 images only), based on Chamfer Distance (mm) (↓) on the DTU dataset [1]. We evaluate the quality of meshes reconstructed with various methods, using only 3 input RGB images. We outperform the previous best method Spurfies [39] by 24% on average (1.04 vs 1.36 CD). Moreover, our method partly relies on 2D Gaussian rasterization [22], which requires cameras to have a centered principal point. As a consequence, we had to crop input images for extracting our meshes, resulting in incomplete reconstruction for some scenes such as scans 34 and 38, for instance. In this regard, even though we outperform previous works, the performance of our method is underestimated in this experimental setup.

5 training images 10 training images

2DGS [22]+MASt3R-SfM [15] 0.052 0.121 GOF [56]+MASt3R-SfM [15] 0.054 0.144 MAtCha Gaussians (Ours) 0.072 0.156

- Table 3. Quantitative evaluation for surface reconstruction in a sparse-view scenario for unbounded scenes of the Tanks&Temples dataset [28]. We evaluate our approach and two baselines based on F-Score (↑). The baselines combine recent surface-reconstruction methods [22, 56] augmented with MASt3R-SfM [15] for greater robustness to sparse-view inputs.

DTU dataset [1] and Tanks & Temples (T&T) dataset [28]. For a fair comparison with prior work, in the experiments on the DTU dataset, we follow the evaluation protocol from previous sparse-view reconstruction methods Spurfies [39] and S-VolSDF [48]: We take three input views 22, 25, and 28 from each scan, use calibrated camera parameters for the global alignment of MASt3R-SfM [15], and filter the recovered meshes with masks before comparison. For T&T, we optimize the models using 5 and 10 images sparsely sampled in the scenes. Since the implementation of Spurfies [39] is not publicly available yet and none of other previous works evaluates sparse-view reconstruction in large or unbounded scenes, we propose two strong baselines for fair comparison on T&T: We use MASt3R-SfM to initialize both 2DGS [22] and GOF [56] and optimize the representations with depth-normal regularization. Indeed, augmenting recent state-of-the-art methods with MASt3R-SfM [15] provides much better robustness in sparse-view scenarios.

[Figure 39]

[Figure 40]

MVSplat [11]

Spurfies [39]

Tabs. 2 and 3 show quantitative comparisons using Chamfer Distance (CD) and F-Score. Our method achieves state-of-the-art performance across both datasets, outperforming both traditional surface reconstruction approaches (Points2Surf [16], CAP-UDF [60] ) and recent neural methods (NeuS [38], VolSDF [48], Spurfies [39]). Notably, we achieve these results with significantly faster reconstruction times—minutes versus hours for most baselines.

[Figure 41]

[Figure 42]

Ours

Ours

Fig. 3 shows qualitative results on different datasets [1, 3, 28] with different numbers of views. The results show the effectiveness and robustness of our method in extreme sparse scenarios.

- Figure 5. Comparisons with Spurfies [39] and MVSplat [11] on an unbounded scene. Our method outperforms state-of-the-art approaches for surface reconstruction and feed-forward Gaussian splatting regression in sparse view scenarios.

Surface Reconstruction We evaluate the accuracy of our reconstructed meshes on two standard benchmarks: the

Novel View Synthesis We provide results of novel view synthesis in sparse-view settings across three challenging real-world datasets of unbounded scenes: Mip-NeRF

Mip-NeRF 360 [3] Tanks&Temples [28] DeepBlending [21] 10%Q PSNR ↑ Avg PSNR ↑ 10%Q PSNR ↑ Avg PSNR ↑ 10%Q PSNR ↑ Avg PSNR ↑

###### 5 training views

2DGS [22]+MASt3R-SfM [15] 15.37 20.84 14.23 16.42 15.84 19.86 GOF [56]+MASt3R-SfM [15] 15.78 21.24 13.69 16.50 15.58 19.87 MAtCha Gaussians (Ours) 18.18 21.90 15.33 17.30 17.22 20.60

###### 10 training views

2DGS [22]+MASt3R-SfM [15] 19.94 24.31 16.63 19.59 14.06 21.14 GOF [56]+MASt3R-SfM [15] 20.99 24.50 16.81 19.59 12.61 21.12 MAtCha Gaussians (Ours) 21.55 25.10 17.96 20.38 17.41 22.98

- Table 4. Quantitative evaluation of Novel View Synthesis in sparse-view scenarios across multiple real-world datasets. We evaluate our method against baselines on three challenging datasets: Mip-NeRF 360 [3], Tanks&Temples [28], and DeepBlending [21]. Baselines consist of recent state-of-the-art approaches augmented with MASt3R-SfM [15] for more robustness to sparse-view scenarios. For each dataset and method, we report both the average PSNR and the 10% quantile PSNR (10%Q PSNR) which better reflects performance on challenging views and better capture the ability of a method to generalize to novel viewpoints. Results are shown for both 5-view and 10-view scenarios, demonstrating our method’s superior performance across different sparsity levels.

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

2DGS [22] +MASt3R-SfM [15]

GOF [56] +MASt3R-SfM [15]

MAtCha Gaussians (Ours)

- Figure 6. Qualitative evaluation for surface reconstruction in a sparse-view scenario for unbounded scenes from Tanks&Temples [28], with 5 training images (top row) and 10 training images (bottom row). Contrary to the baselines, our approach is able to reconstruct accurate and complete surfaces meshes: It not only includes both foreground and background objects, but also recover sharper and finer details.

360 [3], Tanks&Temples [28], and DeepBlending [21]. Similarly to the previous section, we trained our models with either 5 or 10 input images, and evaluate them with 10 test images. More details about the experimental setting are available in the appendix. We report in Tab. 4 both average PSNR and 10% quantile PSNR (10%Q PSNR) metrics. The 10%Q PSNR is the PSNR value below which 10% of the test views fall. Average PSNR provides an overall measure of reconstruction quality, whereas the 10%Q PSNR specifically captures accuracy on the most challenging views as well as the ability of a method to generalize to novel viewpoints. This metric is particularly relevant to sparse-view settings where some novel viewpoints may have very limited overlap with input views.

As shown in Tab. 4 and Fig. 6, our method consistently outperforms the baselines across all datasets and metrics

###### CD ↓ PSNR ↑ SSIM ↑

No Charts Encodings 2.693 16.37 0.369 No Depth Encodings 1.601 17.38 0.424

No Lfit confidence 1.703 17.39 0.428 No Lstruct 1.716 17.00 0.410 No Lalign 1.565 17.33 0.424

###### Full Model 1.04 17.59 0.443

Table 5. Ablation studies on the deformation model and loss components. For evaluating in bounded scenes, we compute the Chamfer Distance on the DTU dataset [1]. For evaluating in unbounded scenes, we optimize our representation on 5 images of each scene of the Mip-NeRF 360 [3] dataset, and we compute rendering metrics on 10 challenging views with little overlap. We see that the charts encodings (CE), the 1D depth encodings (1D-DE), and all of the loss components are required for reaching optimal performance.

even though its main focus is high-quality surface reconstruction, not novel view synthesis. In the 5-view scenario, we achieve significant improvements over the baselines. The performance gap remains substantial even when increasing to 10 input views, where our method maintains superior reconstruction quality across datasets. This consistent performance advantage demonstrates the effectiveness of our chart-based representation and refinement approach in handling sparse-view scenarios. Note that test images include images with very little overlap with training images, which should explain the low values as well as the low variance in the results between the methods.

Qualitative Comparisons with Sparse-View Methods Fig. 5 shows qualitative comparisons with the state-of-

the-art sparse-view surface reconstruction [39] and novel view synthesis [11] methods. Existing methods struggle to generalize to the unbounded scene. In contrast, our method achieves high-quality surface reconstruction even for a scene with a complex background.

Ablation Studies We conduct extensive ablation studies on the deformation architecture and loss components to validate our design choices on the DTU dataset [1] and MipNeRF 360 dataset [3]. As shown in Tab. 5, removing either the charts encodings or 1D depth encodings leads to decreased performance, confirming the importance of both components. The results also show the effectiveness of the weighting of the fitting loss Lfit with learnable confidence maps, the structure loss Lstruct, and the mutual alignment loss Lalign.

#### 6. Conclusion

We presented a novel approach for reconstructing highquality 3D surface meshes from sparse-view RGB images within minutes. Our method leverages a novel representation that models surfaces as 2D manifolds through a collection of charts, initialized using pretrained monocular depth estimation. By combining geometric priors from a depth estimation model, efficient chart-based deformation, and differentiable rendering with 2D Gaussian surfels, our method is able to extract a sharp and accurate estimate of the 3D scene from a few RGB images only. Our representation also allows for high-quality rendering and significantly faster optimization than other state-of-the-art methods. Our experiments demonstrate that our method outperforms existing approaches in sparse-view scenarios while being significantly faster, typically requiring only a few minutes for complete reconstruction.

While our method shows promising results, there are several directions for future work. Extending our framework to handle dynamic scenes and deformable objects would broaden its applications in computer vision and graphics. We believe our work opens new possibilities for fast, high-quality 3D reconstruction from sparse views, with potential applications in virtual reality, digital content creation, and robotics.

#### Acknowledgements

This work was in part supported by JSPS 20H05951 and 21H04893, and JST JPMJCR20G7 and JPMJAP2305. This work was also in part supported by the ERC grant “explorer ” (No. 101097259). This work was granted access to the HPC resources of IDRIS under the allocation 2024AD011013387R2 made by GENCI.

#### References

- [1] Henrik Aanæs, Rasmus Ramsbøl Jensen, George Vogiatzis, Engin Tola, and Anders Bjorholm Dahl. Large-Scale Data for Multiple-View Stereopsis. IJCV, pages 1–16, 2016. 8, 9, 10, 1, 2, 3
- [2] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin Brualla, and Pratul P Srinivasan. Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields. In ICCV, pages 5855–5864, 2021. 3
- [3] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In CVPR, pages 5460–

5469. IEEE, 2022. 8, 9, 10, 2, 3, 4

- [4] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields. In ICCV, pages 19697– 19705, 2023. 3
- [5] Eric Brachmann, Jamie Wynn, Shuai Chen, Tommaso Cavallari, Aron´ Monszpart, Daniyar Turmukhambetov, and Victor Adrian Prisacariu. Scene Coordinate Reconstruction: Posing of Image Collections via Incremental Learning of a Relocalizer. In ECCV, pages 421–440. Springer, 2024. 3
- [6] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction. In CVPR, pages 19457–19467, 2024. 3
- [7] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. TensoRF: Tensorial Radiance Fields. In ECCV, pages 333–350, 2022. 3
- [8] Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction. arXiv preprint arXiv:2406.06521, 2024. 3
- [9] Hanlin Chen, Chen Li, and Gim Hee Lee. NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance. arXiv preprint arXiv:2312.00846, 2023.
- [10] Hanlin Chen, Fangyin Wei, Chen Li, Tianxin Huang, Yunsong Wang, and Gim Hee Lee. VCR-GauS: View Consistent Depth-Normal Regularizer for Gaussian Surface Reconstruction. In NeurIPS, 2024. 3
- [11] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images. In ECCV, pages 370–386. Springer, 2024. 3, 8, 10
- [12] Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang, Andrea Vedaldi, Tat-Jen Cham, and Jianfei Cai. MVSplat360: Feed-Forward 360 Scene Synthesis from Sparse Views. In NeurIPS, 2024. 4
- [13] Brian Curless and Marc Levoy. A Volumetric Method for Building Complex Models from Range Images. In ACM SIGGRAPH Conference Papers, pages 303–312, 1996. 2
- [14] Pinxuan Dai, Jiamin Xu, Wenxiang Xie, Xinguo Liu, Huamin Wang, and Weiwei Xu. High-quality Surface Re-

- construction using Gaussian Surfels. In ACM SIGGRAPH Conference Papers, pages 1–11, 2024. 3, 4, 6
- [15] Bardienus Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, and Jerome Revaud. MASt3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion. arXiv preprint arXiv:2409.19152, 2024. 1, 3, 4, 5, 7, 8, 9
- [16] Philipp Erler, Paul Guerrero, Stefan Ohrhallinger, Niloy J. Mitra, and Michael Wimmer. Points2Surf Learning Implicit Surfaces from Point Clouds. In ECCV, pages 108–124. Springer, 2020. 8
- [17] Zhiwen Fan, Wenyan Cong, Kairun Wen, Kevin Wang, Jian Zhang, Xinghao Ding, Danfei Xu, Boris Ivanovic, Marco Pavone, Georgios Pavlakos, Zhangyang Wang, and Yue Wang. InstantSplat: Unbounded Sparse-view Posefree Gaussian Splatting in 40 Seconds. arXiv preprint arXiv:2403.20309, 2024. 3, 4
- [18] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance Fields without Neural Networks. In CVPR, pages 5501–5510, 2022. 3
- [19] Antoine Gu´edon and Vincent Lepetit. Gaussian Frosting: Editable Complex Radiance Fields with Real-Time Rendering. In ECCV, 2024. 3, 4
- [20] Antoine Gu´edon and Vincent Lepetit. SuGaR: SurfaceAligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering. In CVPR, pages 5354–5363, 2024. 3, 4, 8
- [21] Peter Hedman, Julien Philip, True Price, Jan-Michael Frahm, George Drettakis, and Gabriel J. Brostow. Deep blending for free-viewpoint image-based rendering. ACM TOG, 37(6): 257, 2018. 9, 4
- [22] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. In ACM SIGGRAPH Conference Papers, pages 1–11, 2024. 3, 4, 6, 8, 9, 1
- [23] Han Huang, Yulun Wu, Junsheng Zhou, Ge Gao, Ming Gu, and Yu-Shen Liu. NeuSurf: On-Surface Priors for Neural Surface Reconstruction from Sparse Input Views. In AAAI, pages 2312–2320. AAAI Press, 2024. 3, 4, 8
- [24] Kaiwen Jiang, Yang Fu, Mukund Varma T, Yash Belhe, Xiaolong Wang, Hao Su, and Ravi Ramamoorthi. A ConstructOptimize Approach to Sparse View Synthesis without Camera Pose. In ACM SIGGRAPH Conference Papers, pages

- 1–11, 2024. 3

[25] Michael M. Kazhdan and Hugues Hoppe. Screened poisson surface reconstruction. ACM TOG, 32(3):29:1–29:13, 2013.

- 2

- [26] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM TOG, 42(4), 2023. 2, 3, 6, 1, 4
- [27] Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas, Michael Wimmer, Alexandre Lanvin, and George Drettakis. A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets. ACM TOG, 43(4), 2024.

4, 1

- [28] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction. ACM TOG, 36(4), 2017. 8, 9, 2, 3, 4
- [29] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding Image Matching in 3D with MASt3R. In ECCV, pages 71–91, 2024. 3
- [30] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. DNgaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization. In CVPR, pages 20775–20785, 2024. 3
- [31] Xiaoxiao Long, Cheng Lin, Peng Wang, Taku Komura, and Wenping Wang. SparseNeuS: Fast Generalizable Neural Surface Reconstruction from Sparse Views. In ECCV, pages 210–227. Springer, 2022. 3, 4, 8
- [32] H.C. Longuet-Higgins. A computer algorithm for reconstructing a scene from two projections. In Readings in Computer Vision, pages 61–62. Morgan Kaufmann, San Francisco (CA), 1987. 3
- [33] William E. Lorensen and Harvey E. Cline. Marching Cubes: A High Resolution 3D Surface Construction Algorithm. In ACM SIGGRAPH Conference Papers, page 163–169, 1987.

- 2

[34] Xiaoyang Lyu, Yang-Tian Sun, Yi-Hua Huang, Xiuzhe Wu, Ziyi Yang, Yilun Chen, Jiangmiao Pang, and Xiaojuan Qi.

- 3DGSR: Implicit Surface Reconstruction with 3D Gaussian Splatting. arXiv preprint arXiv:2404.00409, 2024. 3

- [35] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In ECCV, pages 99–106, 2020. 2, 3
- [36] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant Neural Graphics Primitives with a Multiresolution Hash Encoding. ACM TOG, 41(4):102:1– 102:15, 2022. 3
- [37] Avinash Paliwal, Wei Ye, Jinhui Xiong, Dmytro Kotovenko, Rakesh Ranjan, Vikas Chandra, and Nima Khademi Kalantari. CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians. In ECCV, pages 19–37. Springer, 2024. 3
- [38] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction. In NeurIPS, pages 27171–27183, 2021. 3, 8
- [39] Kevin Raj, Christopher Wewer, Raza Yunus, Eddy Ilg, and Jan Eric Lenssen. Spurfies: Sparse Surface Reconstruction using Local Geometry Priors. arXiv preprint arXiv:2408.16544, 2024. 3, 4, 8, 10
- [40] Yufan Ren, Fangjinhua Wang, Tong Zhang, Marc Pollefeys, and Sabine S¨usstrunk. VolRecon: Volume Rendering of Signed Ray Distance Functions for Generalizable MultiView Reconstruction. In CVPR, pages 16685–16695. IEEE,

2023. 3, 4, 8

- [41] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-Motion Revisited. In CVPR, pages 4104–4113, 2016. 3
- [42] Noah Snavely, Steven M. Seitz, and Richard Szeliski. Photo tourism: exploring photo collections in 3D. ACM TOG, 25

(3):835–846, 2006. 3

- [43] Matias Turkulainen, Xuqian Ren, Iaroslav Melekhov, Otto Seiskari, Esa Rahtu, and Juho Kannala. DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing. In IEEE/CVF Winter Conference on Applications of Computer Vision, 2025. 3, 4, 1
- [44] Sheng Wan, Tung-Yu Wu, Wing Hung Wong, and Chen-Yi Lee. Confnet: Predict with Confidence. In ICASSP, pages 2921–2925. IEEE, 2018. 6
- [45] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. VGGSfM: Visual Geometry Grounded Deep Structure From Motion. In CVPR, pages 21686–21697,

2024. 3

- [46] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3D Vision Made Easy. In CVPR, pages 20697–20709, 2024. 3, 6
- [47] Wangze Xu, Huachen Gao, Shihe Shen, Rui Peng, Jianbo Jiao, and Ronggang Wang. MVPGS: Excavating Multi-view Priors for Gaussian Splatting from Sparse Input Views. In ECCV, 2024. 3
- [48] Haoyu Wu, Alexandros Graikos, and Dimitris Samaras. Svolsdf: Sparse multi-view stereo regularization of neural implicit surfaces. In ICCV, pages 3533–3545. IEEE, 2023. 3, 4, 8
- [49] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. DepthSplat: Connecting Gaussian Splatting and Depth. arXiv preprint arXiv:2410.13862, 2024. 3
- [50] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. In CVPR, pages 10371–10381. IEEE, 2024. 4
- [51] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth Anything V2. In NeurIPS, 2024. 4, 7
- [52] Yaniv Wolf, Amit Bracha, and Ron Kimmel. GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views. In ECCV, 2024. 3
- [53] Hanyang Yu, Xiaoxiao Long, and Ping Tan. LM-Gaussian: Boost Sparse-view 3D Gaussian Splatting with Large Model Priors. arXiv preprint arXiv:2409.03456, 2024. 3
- [54] Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xiangli, and Bo Dai. GSDF: 3DGS Meets SDF for Improved Rendering and Reconstruction. In NeurIPS, 2024. 3
- [55] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-Splatting: Alias-free 3D Gaussian Splatting. In CVPR, pages 19447–19456, 2024. 3
- [56] Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes. ACM TOG, 2024. 3, 4, 6, 7, 8, 9, 2
- [57] Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, and Ping Tan. RaDe-GS: Rasterizing Depth in Gaussian Splatting. arXiv preprint arXiv:2406.01467,

2024. 3

- [58] Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu, Jin Zheng, and Xiao Bai. CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization. In ECCV, pages 335–352. Springer, 2024. 3

- [59] Wenyuan Zhang, Yu-Shen Liu, and Zhizhong Han. Neural Signed Distance Function Inference through Splatting 3D Gaussians Pulled on Zero-Level Set. In NeurIPS, 2024. 3
- [60] Junsheng Zhou, Baorui Ma, Shujuan Li, Yu-Shen Liu, Yi Fang, and Zhizhong Han. CAP-UDF: Learning Unsigned Distance Functions Progressively from Raw Point Clouds with Consistency-Aware Field Optimization. IEEE TPAMI,

2024. 8

- [61] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting. In ECCV, pages 145–163. Springer, 2024. 3

## MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views

### Appendix

In this appendix, we describe

- • additional implementation details,
- • details about our mesh extraction method,
- • and additional qualitative results. We also provide a video that offers an overview of the approach and showcases additional qualitative results.

#### 7. Implementation Details

##### 7.1. Initializing Charts

For initializing the charts using a monocular depth estimation model, we not only backproject the depth maps into 3D but also roughly adjust the scale of the depth estimates using a global affine rescaling model [27, 43]. Note that we can compute an explicit closed-form solution for this affine rescaling which executes in less than a second.

In our experiments, the size of our charts is proportional to the input views, and the longest sides of the charts have length max(h,w) = 512. We rely on MASt3R-SfM [15] to obtain an SfM point cloud for aligning the charts.

##### 7.2. Chart Deformation Model

We can adjust the resolution of the learnable charts encodings (i.e., r) according to the density of the SfM points or the number of views. The sparser the SfM point cloud or the training images, the lower the resolution of the charts encodings. In other words, we can explicitly adjust the strength of the inductive bias in our chart deformation model according to the different scenarios. For small scenes with only 3 input views like the objects from the DTU [1] dataset, we use a small resolution parameter r = 0.1 for the charts encodings. In larger and unbounded scenes with

- 5 or 10 input views, we use a larger resolution parameter r = 0.4 for our charts encodings.

The other hyperparameters are constants and independent of the inputs. In practice, we set d = 32 and use an MLP with only 1 hidden layer. The number of channels in the hidden layer is 64. For aligning our charts with the initial SfM points, we optimize our model for 1000 iterations. For refining the charts, we optimize our model for 3000 iterations.

During the alignment with the SfM points, we deform the charts along the camera rays, as we empirically found it to be more robust. Moreover, deforming the charts along the camera rays enables very efficient computation of the mutual alignment loss, as in this case, the 3D to 2D mapping of our charts is equivalent to the camera screen projection

transform. To deform charts along the rays, we use a onedimensional output layer for the MLP, and we compute the 3D deformation by multiplying the MLP output by the ray direction.

During the refinement with Gaussian surfel rendering,

we first update the initial charts ψi(0) and replace them with the deformed charts ψi; Then, we reinitialize the weights of the MLP and replace the output layer with a 3-dimensional layer in order to learn a full 3D deformation for the charts.

##### 7.3. Refining the Manifold with Gaussian Surfels

During the second optimization stage, we rely on a photometric loss to refine the manifold. At each iteration, we render the manifold by first instantiating 2D Gaussian surfels on the surface, then rasterizing the Gaussians with a surfel rasterizer [22].

Photometric loss The photometric loss consists of an L1 loss L1 and a D-SSIM term LD-SSIM:

Lphoto = (1 − λ)L1 + λLD-SSIM , (9) where we set λ = 0.2, following past 3DGS works [26].

Structure loss To preserve the fine geometry of our aligned charts, we maintain the structure loss but replace the depth estimates with the depth of our aligned charts. We also weight the structure loss using our confidence maps Ci estimated during the manifold alignment to the SfM points, which makes it a scale-accurate depth regularization robust to outliers.

To further regularize the geometry, we also use a depthnormal consistency loss and a depth distortion loss, as introduced in 2DGS [22].

Distortion loss The distortion loss prevents Gaussians from spreading around the surface. Since we instantiate Gaussian surfels on the manifold represented as a collection of charts, the distortion term enforces the surfaces of the different charts to align together and form a coherent manifold. For each pixel p, the distortion loss is given by

Ld =

ωiωj|zi − zj|, (10)

i,j

where i and j represent the i-th and j-th Gaussian surfels intersected along the ray, zi is the depth of the intersection point between the ray and the i-th Gaussian surfel, and ωi is the blending weight of the i-th intersection.

Depth-Normal consistency loss The depth-normal consistency loss aims to align the normals of the closest Gaussian surfels along the ray with the gradient of the depth map. In our case, this term encourages the surfaces of the different charts to have the same orientation. For a pixel p, the normal consistency loss at p is given by

Ln =

i

ωi(1 − nTi Np), (11)

where ni is the normal of the i-th Gaussian surfel along the ray and Np is the normal at pixel p computed from the gradient of the depth map.

Optimization loss The complete loss for refining the charts is

Lrefine = Lphoto + λstructLstruct + λdLd + λnLn , (12)

where we set λstruct = 1, λd = 500, and λn = 0.25. We refine the representation for 3000 iterations, and introduce Ld and Ln only after 600 iterations.

##### 7.4. Extracting a Surface Mesh from the Manifold

We propose two different approaches for extracting a surface mesh from our manifold representation, depending on the scene complexity and the desired level of detail.

Direct Mesh Extraction For scenes with moderate complexity or extreme sparse-view setups (e.g., 3 views on DTU), we can directly extract a surface mesh from our manifold using a custom multi-resolution TSDF fusion approach, or a custom implementation of the adaptive tetrahedralization from Gaussian Opacity Fields [56]. Since we describe our tetrahedralization in the main paper, we focus on providing additional details about the multi-resolution TSDF below.

We render depth maps from our manifold and fuse them into several TSDF volumes with different resolutions. The lower the resolution, the larger the bounding box used for applying the TSDF algorithm. Then, we merge the TSDF volumes and remove the overlapping regions. Note that, in a sparse-view scenario, the number of depth maps is very low, so that integrating depth maps for computing the TSDF volumes is very fast and takes less than a minute.

Our multi-resolution approach allows us to accurately reconstruct both foreground objects and background regions with a decent number of vertices, which is crucial for unbounded scenes. However, even though our multiresolution TSDF is very fast, it generally erodes the geometry and creates holes in the extracted surface. In this regard, we recommend using the tetrahedralization for extracting meshes.

Free Gaussians Refinement For scenes requiring finer geometric details, particularly in large unbounded environments, we propose an additional refinement step that leverages our manifold as a strong geometric prior. Instead of directly extracting the mesh, we first let Gaussian surfels get freely optimized in 3D space for a few iterations while strongly constraining them with our manifold representation.

For this, we freeze the manifold but unfreeze the Gaussians’ parameters (position, scale, and rotation) and regularize them using depth maps rendered from the manifold through a combination of our refinement loss and an L1 depth loss with a weighting factor λdepth = 0.75. We also use our confidence maps to weigh the depth regularization, but not the structure loss that relies on the derivatives of the depth. Indeed, applying the structure loss everywhere in the scene enables regularization of Gaussians located even in low-confidence areas, where normal maps and curvature maps still provide a reliable supervision signal despite of inaccurate depth values.

This refinement step is particularly effective because our manifold provides scale-accurate regularization, unlike traditional depth-based regularization methods that often struggle with scale ambiguity. The manifold acts as a reliable geometric prior that prevents Gaussians from diverging while letting them recover fine surface details that might not be fully captured by the manifold representation alone.

After this Gaussian refinement stage, we extract the final mesh using the same multi-resolution TSDF fusion or tetrahedralization approaches described above, but now applied to the refined Free Gaussians representation. This two-stage approach allows us to recover very fine geometric details while maintaining the overall accuracy and robustness of our manifold representation.

#### 8. Additional Results and Details

Surface Reconstruction Fig. 7 shows qualitative results. Our method can reconstruct high-quality surfaces across different scenarios, from bounded objects (DTU [1] dataset) to unbounded scenes (Tanks&Temples [28] and Mip-NeRF 360 [3] datasets), using varying numbers of, but sparse, input views (3, 5, and 10 views). For each example, we show a rendered view, the estimated depth map, surface normals, and the extracted mesh, which collectively show the consistency of our reconstruction across different representations. For the objects from the DTU dataset, we directly extract the mesh from the manifold representation using our multi-resolution TSDF fusion approach. For the unbounded scenes from the T&T and Mip-NeRF 360 datasets, we first refined free Gaussians around the manifold as explained in the previous section, then extracted the mesh using the same multi-resolution TSDF fusion approach.

In the 3-view scenarios (first two rows), our method suc-

Mesh

Rendering Depth Normal

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

3views3views5views5views10views10views10views

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

[Figure 76]

- Figure 7. Qualitative reconstruction results across different scenarios and numbers of input views. We show results on both bounded objects from DTU [1] (first two rows, 3 views) and unbounded scenes from Tanks&Temples [28] and Mip-NeRF 360 [3] (middle and bottom rows). For each example, we show (from left to right): the rendered novel view, estimated depth map, surface normals, and the extracted mesh. For bounded objects (DTU), meshes are extracted directly from our manifold representation, while for unbounded scenes, we first refine free Gaussians around the manifold before mesh extraction. Note how our method maintains consistent quality across different scenarios, from small objects to large-scale scenes with complex backgrounds.

MVSplat360 [12] Ours

[Figure 77]

[Figure 78]

- Figure 8. Qualitative comparisons of novel view synthesis with MVSplat360 [12] on an unbounded scene with 5 training images. Our method can render more photorealistic images than the concurrent feed-forward novel view synthesis method in sparse view scenarios.

cessfully recovers detailed geometry despite the extreme sparsity of input views. The 5-view and 10-view examples (middle and bottom rows) demonstrate how our approach scales to larger unbounded scenes while maintaining reconstruction quality throughout the scene, including distant background regions.

Novel View Synthesis Tab. 4 of the main paper provides results of novel view synthesis in sparse-view settings across three challenging real-world datasets of unbounded scenes: Mip-NeRF 360 [3], Tanks&Temples [28], and DeepBlending [21]. Specifically, we follow 3DGS [26] and use the scenes Playroom and Dr. Johnson for evaluation on the DeepBlending dataset. For the T&T dataset, we use the standard split of 6 scenes as used in 2DGS [22] and GOF [56] but removed Courthouse and Meetingroom, as these very large scenes are not suitable for sparse-view scenarios with only 5 or 10 input views.

We consider two scenarios with 5 and 10 training views, respectively. For each dataset, we built training sets of 5 and 10 input views and evaluated on a set of 10 test views, including both easy views with high overlap with the training views and much more challenging views with very limited overlap. For fair comparison, we augment recent state-ofthe-art methods (2DGS [22] and GOF [56]) with MASt3RSfM [15], as it provides better robustness in sparse-view scenarios.

We report both average PSNR and 10% quantile PSNR (10%Q PSNR) metrics. The 10%Q PSNR is the PSNR value below which 10% of the test views fall. Average PSNR provides an overall measure of reconstruction quality. In contrast, the 10%Q PSNR specifically captures accuracy on the most challenging views as well as the ability of a method to generalize to novel viewpoints. This metric is particularly relevant to sparse-view settings where some novel viewpoints may have very limited overlap with input views.

As shown in Tab. 4, our method consistently outperforms the baselines across all datasets and metrics. In the

5-view scenario, we achieve significant improvements over the baselines. The performance gap remains substantial even when increasing to 10 input views, where our method maintains superior reconstruction quality across datasets. This consistent performance advantage demonstrates the effectiveness of our chart-based representation and refinement approach in handling sparse-view scenarios.

Notably, our method shows particular strength in maintaining quality for challenging views, as evident in the larger improvements in 10%Q PSNR compared to average PSNR. This suggests that our chart-based representation, combined with the robust deformation model and multistage refinement process, helps maintain consistency even in regions with limited overlap in views.

We also qualitatively compare our method with MVSplat360 [12], a concurrent method for feed-forward novel view synthesis in sparse-view settings. Fig. 8 shows rendering results of MVSplat360 and our method from novel viewpoints by using 5 views. MVSplat360 suffers from a domain gap from training data and limited image resolution due to training of its feed-forward networks, leading to unrealistic rendering results.

