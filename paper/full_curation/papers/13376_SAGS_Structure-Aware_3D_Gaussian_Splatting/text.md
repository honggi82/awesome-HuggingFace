# arXiv:2404.19149v1[cs.CV]29Apr2024

## SAGS: Structure-Aware 3D Gaussian Splatting

Evangelos Ververas1,2∗, Rolandos Alexandros Potamias1,2∗, Jifei Song2, Jiankang Deng1,2, and Stefanos Zafeiriou1

1 Imperial College London {e.ververas16, r.potamias, j.deng16, s.zafeiriou}@imperial.ac.uk 2 Huawei Noah’s Ark Lab {jifeisong}@huawei.com ∗ Equal contribution

Abstract. Following the advent of NeRFs, 3D Gaussian Splatting (3DGS) has paved the way to real-time neural rendering overcoming the computational burden of volumetric methods. Following the pioneering work of 3D-GS, several methods have attempted to achieve compressible and high-fidelity performance alternatives. However, by employing a geometry-agnostic optimization scheme, these methods neglect the inherent 3D structure of the scene, thereby restricting the expressivity and the quality of the representation, resulting in various floating points and artifacts. In this work, we propose a structure-aware Gaussian Splatting method (SAGS) that implicitly encodes the geometry of the scene, which reflects to state-of-the-art rendering performance and reduced storage requirements on benchmark novel-view synthesis datasets. SAGS is founded on a local-global graph representation that facilitates the learning of complex scenes and enforces meaningful point displacements that preserve the scene’s geometry. Additionally, we introduce a lightweight version of SAGS, using a simple yet effective mid-point interpolation scheme, which showcases a compact representation of the scene with up to 24× size reduction without the reliance on any compression strategies. Extensive experiments across multiple benchmark datasets demonstrate the superiority of SAGS compared to state-of-the-art 3D-GS methods under both rendering quality and model size. Besides, we demonstrate that our structure-aware method can effectively mitigate floating artifacts and irregular distortions of previous methods while obtaining precise depth maps. Project page https://eververas.github.io/SAGS/.

### 1 Introduction

Novel View Synthesis (NVS) is a long-studied problem that aims to generate images of a scene from a specific point of view, using only a sparse set of images from different viewpoints with known camera parameters. Due to its diverse applications spanning from Virtual Reality (VR) [7] to content creation [4,31], novel view synthesis has garnered significant attention. With the advent of Neural Radiance Field (NeRF) [22], an enormous amount of methods have been proposed to utilize volumetric rendering and learn implicit fields of the scene, achieving remarkable rendering results. However, albeit achieving highly detailed

3D-GS Proposed Proposed-Lite

[Figure 1]

[Figure 2]

[Figure 3]

PSNR 19.33 dB LPIPS 0.225 Mem 414 Mb

PSNR 31.02 dB LPIPS 0.135 Mem 99 Mb

PSNR 30.61 dB LPIPS 0.147 Mem 43 Mb

[Figure 4]

[Figure 5]

[Figure 6]

PSNR 27.02 dB LPIPS 0.178 Mem 64Mb

PSNR 22.96 dB LPIPS 0.349 Mem 240 Mb PSNR 25.91 dB LPIPS 0.187 Mem 39 Mb

- Fig. 1: Structure-Aware GS (SAGS) leverages the intrinsic structure of the scene and enforces point interaction using graph neural networks outperforming the structure agnostic optimization scheme of 3D-GS [15]. The 3D-GS method optimizes each Gaussian independently which results in 3D floaters and large point displacements from their original position (left). This can be also validated in the histogram of displacements (right) between the initial and the final position (mean) of a 3D Gaussian. Optimization-based methods neglect the scene structure and displace points far from their initial position to minimize rendering loss, in contrast to SAGS that predicts displacements that preserve the initial structure. The 3D-GS figures are taken directly from the original 3D-GS website.

results, volumetric rendering methods fail to produce real-time renderings which hinders their real-world applications.

Recently, Kerbl et al. [15] introduced 3D Gaussian Splatting (3D-GS) to tackle this limitation using a set of differentiable 3D Gaussians that can achieve state-of-the-art rendering quality and real-time speed on a single GPU, outperforming previous NeRF-based methods [1–3, 22]. In 3D-GS, the scene is parametrised using a set of 3D Gaussians with learnable shape and appearance attributes, optimized using differentiable rendering. To initialize the 3D Gaussians, Kerbl et al. [15] relied on the point clouds derived from COLMAP [29], neglecting any additional scene structure and geometry during optimization.

Undoubtedly, one of the primary drawbacks of the 3D-GS method is the excessive number of points needed to produce high-quality scene renderings. Following the success of 3D-GS, numerous methods have been proposed to reduce the storage requirements using compression and quantization schemes [9,17,24] while retaining the rendering performance. However, similar to 3D-GS, each Gaussian is optimized independently to fit the ground truth views, without the use of any additional structural inductive biases to guide the optimization. As can be seen in Fig. 1 (right), such a naive setting results in points being displaced far away from their initialization, thus neglecting their initial point structure and introducing floating points and artifacts [36] (highlighted in Fig. 1 (left) with red arrows). Apart from a significant degradation in the rendering quality, neglecting the scene’s geometry directly influences the scene’s properties, including depth, which thereby limits its VR/AR applications.

In this study, we propose a structure-aware Gaussian splatting method that aims to implicitly encode the scene’s geometry and learn inductive biases that

lead to point displacements that maintain the topology of the scene. Intuitively, points within the same local region often share common attributes and features, such as normals and color, that are neglected by current 3D-GS methods. Inspired by the success of Point Cloud analysis [26], we found our method on a graph constructed from the input scene and learn to model 3D Gaussians, as displacements from their original positions. The constructed graph serves as an inductive bias that aims to encode and preserve the scene structure while learning robust Gaussian attributes. Unlike the 3D-GS, the proposed method leverages the inter- and intra-connectivity between 3D point positions and learns to predict the Gaussian attributes using graph neural networks. Using both local and global structural information the network can not only reduce artifact floaters, but it can also increase the expressivity of the network leading to accurate scene representations compared to structure agnostic 3D-GS methods.

Under a series of experiments on different datasets and scenes, we evaluate the proposed method in terms of rendering quality, structure preservation, storage, and rendering performance, demonstrating the importance of structure in 3D Gaussian splatting. The proposed method can outperform the rendering quality of 3D-GS [15] while reducing the memory requirements without sacrificing rendering speed. To sum up, our contributions can be summarized as follows:

- – We introduce the first structure-aware 3D Gaussian Splatting method that leverages both local and global structure of the scene.
- – We bridge the two worlds between point cloud analysis and 3D Gaussian splatting, leveraging graph neural networks on the 3D point space.
- – We showcase that our method can produce state-of-the-art rendering quality, while reducing the memory requirements by up to 11.7× and 24× with our full and lightweight models, respectively.

### 2 Related Work

Traditional scene reconstruction. Traditional 3D scene reconstruction techniques [28] utilize the structure-from-motion (SfM) pipeline [29] for sparse point cloud estimation and camera calibration, and further apply multi-view stereo (MVS) [11] to obtain mesh-based 3D scene reconstructions. Specifically, the traditional pipeline starts with a feature extraction step that obtains robust feature descriptors like [19] and Superpoint [8], followed by a feature matching module, e.g., SuperGlue [27], that matches the 2D image descriptors. After that, pose estimation and bundle adjustment steps are conducted to obtain all the reconstructed parameters, according to Incremental SfM [6], Global SfM [39], or Hybrid-SfM [5]. Finally, MVS methods [14] are employed to reconstruct depth and normals of the target 3D object and subsequently fuse them to produce the final reconstruction.

NeRF based scene reconstruction. Neural radiance fields (NeRF) [22] introduced an implicit neural representation of 3D scenes, that revolutionized neural

rendering based novel-view synthesis achieving remarkable photo-realistic renders. Several methods have extended NeRF model by including a set of appearance embeddings [21] and improved training strategies [30, 35] to tackle complex and large-scale scenes. MipNeRF360 [2] achieved state-of-the-art rendering quality by addressing the aliasing artifacts observed in previous methods, suffering however from exceptionally slow inference times. To improve the training and rendering efficiency of NeRFs, numerous methods have utilized grid-based structures to store compact feature representations. Interestingly, Plenoxels [10] optimized a sparse voxel grid and achieved high-quality rendering performance without resorting to MLPs. Muller et al. [23] highlighted the importance of positional encodings for high-fidelity neural rendering and introduced a set of hash-grid encodings that significantly improved the expressivity of the model. Despite improving the efficiency compared to the global MLP representations, grid-based methods still struggle to achieve real-time rendering performance.

3D-GS based scene reconstruction. Recently, 3D Gaussian splatting [15] has been proposed to construct anisotropic 3D Gaussians as primitives, enabling high-quality and real-time free-view rendering. Similar to NeRFs, 3D-GS attempts to overfit the scene by optimizing the Gaussian properties. However, this usually results in an enormous amount of redundant Gaussians that hinder the rendering efficiency and significantly increase the memory requirements of the model. Several methods [17,24] have attempted to reduce memory consumption by compressing the 3D Gaussians with codebooks, yet with the structure neglected, which could be vital in both synthesizing high-quality rendering and reducing the model size. Most relevant to our work, Scaffold-GS [20] introduced a structured dual-layered hierarchical scene representation to constrain the distribution of the 3D Gaussian primitives. However, Scaffold-GS still relies on structure-agnostic optimization which neglects the scene’s global and local geometry, resulting in locally incoherent 3D Gaussians. This not only degrades the rendering quality but also significantly impacts the structural properties of the scene, including its depth. To tackle this, we devise a structure-aware Gaussian splatting pipeline that implicitly encodes the scene structure leveraging the inter-connectivity of the Gaussians. The proposed method introduces an inductive bias that not only facilitates high-fidelity renderings using more compact scene representations but also preserves the scene topology.

### 3 Method

#### 3.1 Preliminaries: 3D Gaussian Splatting

3D Gaussian Splatting [15] is a state-of-the-art novel-view synthesis method that relies on explicit point-based representation. In contrast to the implicit representation of NeRFs that require computationally intensive volumetric rendering, 3D-GS renders images using the efficient point-based splatting rendering [34,37]. Each of the 3D Gaussians is parametrized by its position µ, covariance matrix

[Figure 7]

[Figure 8]

SfM Point Cloud

Novel View

Position

[Figure 9]

[Figure 10]

| | | |
|---|---|---|
| |[Figure 11]<br><br>[Figure 12]<br><br>MLP| |
| | | |

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

###### k-NN Graph

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Color

Densiﬁcation

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

HashEncoding

[Figure 27]

[Figure 28]

| | | |
|---|---|---|
| |[Figure 29]<br><br>[Figure 30]<br><br>MLP| |
| | | |

[Figure 31]

Rendering

[Figure 32]

[Figure 33]

| | |
|---|---|
| | |
| | |

| | | | | |
|---|---|---|---|---|
| |[Figure 34]<br><br>[Figure 35]|GNN| |[Figure 36]<br><br>[Figure 37]|
| | | | | |

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Opacity

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

| | | |
|---|---|---|
| |[Figure 50]<br><br>[Figure 51]<br><br>MLP| |
| | | |

[Figure 52]

[Figure 53]

[Figure 54]

: Initial Positions : Interpolated Positions

[Figure 55]

Covariance

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

| | | |
|---|---|---|
| |[Figure 60]<br><br>[Figure 61]<br><br>MLP| |
| | | |

[Figure 62]

[Figure 63]

[Figure 64]

- Fig. 2: Overview of the proposed method. Given a point cloud obtained from COLMAP [29], we initially apply a curvature-based densification step to populate under-represented areas. We then apply k-NN search to link points p within local regions and create a point set graph. Leveraging the inductive biases of graph neural networks, we learn a local-global structural feature for each point Φ(pi, fi). Using a set of small MLPs we decode the structural features to 3D Gaussian attributes, i.e., color c, opacity α, covariance Σ and point displacements ∆p for the initial point position. Finally, we render the 3D Gaussians following the 3D-GS Gaussian rasterizer [15].

Σ, opacity α and color c and has a density function defined as:

- 1

- 2(x−µ)TΣ−1(x−µ) (1)

f(x|µ,Σ) = e−

where x is an arbitrary 3D point. To prevent the occurrence of non-positive semi-definite covariance matrices the authors proposed to decompose covariance matrix using rotation and scaling Σ = RSSTRT where the rotation is represented using quaternions. Finally, the 3D Gaussians are splatted on the 2D image, and their corresponding 2D pixel color is calculated by blending the N ordered Gaussians at the queried pixel as:

- i−1
- j=1

N

(1 − αj)ci (2)

C =

αi

i=1

where ci and αi are the color and the opacity of Gaussian i.

#### 3.2 Structure-Aware 3D Gaussian Splatting

In this work, we propose a structure-aware 3D Gaussian Splatting method, that takes as input a sparse point cloud P ∈ RM×3 from COLMAP [29] along with a set of sparse views with known camera parameters and learns a set of 3D Gaussians that fit the input views while maintaining the initial structure. The proposed method can be divided into three main components: a) the curvatureaware densification, b) the structure-aware encoder, and c) the refinement layer.

- Fig. 2 illustrates the pipeline of the proposed method.

Curvature-Aware Densification. Undoubtedly, the performance of 3D-GS methods is significantly impacted by the sparse initialization of the Gaussians, which relies on COLMAP. In scenarios with challenging environments featuring

texture-less surfaces, conventional Structure-from-Motion (SfM) techniques frequently fall short of accurately capturing the scene details, and thus are unable to establish a solid 3D-GS initialization. To tackle such cases, we introduce a densification step that aims to populate areas with zero or few points.

In essence, 3D-GS methods attempt to reconstruct a scene from a sparse point cloud by employing a progressive growing scheme. This approach closely aligns with the well-explored field of point cloud upsampling. Drawing inspiration from the recent Grad-PU method [12], we incorporate an upsampling step to augment the point cloud’s density, particularly in regions characterized by low curvature, which are typically under-represented in the initial COLMAP point cloud. We estimate the Gaussian mean curvature of the point cloud P following the localPCA approach [25]. To generate a set of midpoints pm, we define a k-nearest neighbour graph for each of the low curvature points p ∈ PL ⊂ P and calculate its midpoints as:

- 1

- 2

(p + pj), j ∈ k-NN(p) (3)

pm =

where PL is the set of points p with curvature lower than a threshold and pj is a neighbour of p. An illustration of our densification approach is shown in Fig. 3.

Leveraging the mid-point densification step, we can train a lightweight model solely on the initial point set P, while the remaining points, along with their attributes, can be defined on-the-fly.This approach allows us to achieve both good performance and an extremely compact size, without the need for any compression scheme. We will refer to this model as SAGS-Lite.

[Figure 65]

- Fig. 3: Overview of the densification. Given an initial SfM [29] point cloud (left) we estimate the curvature following [25]. Curvature values are presented color-coded on the input COLMAP point cloud (middle) where colors with minimum curvature are closer to the purple color. The curvature-aware densification results in more points populating the low-curvature areas (right).

##### Structure-Aware Encoder. Intuitively, points that belong to adjacent regions will share meaningful structural features that could improve scene understanding and reduce the floating artifacts. To enable point interactions within local regions

and learn structural-aware features, we founded our method on a graph neural network encoder that aggregates local and global information within the scene.

In particular, the first step of the proposed structure-aware network involves creating a k-Nearest Neighbour (NN) graph that links points within a local region. Using such k-NN graph we can enable point interaction and aggregate local features using graph neural networks. To further enhance our encoder with global structural information of the scene, we included a global feature that is shared across the points. In particular, the structure-aware GNN learns a feature encoding fi for each point pi ∈ P, using the following aggregation function:

 

  (4)

wijhΘ(γ(pj),fj − fi,g)

Φ(pi,fi) = ϕ

j∈N(i)

where γ(·) denotes a positional encoding function that maps point p to a higher dimensional space D, g represents a global feature of the scene calculated as the maximum of the feature encoding max(f), hΘ is a learnable MLP, ϕ represents a non-linear activation function and wij is an inverse-distance weight defined from the softmax of the inverse distances between the point pi and its neighbors N(i):

dist−1(pi,pj) j∈N(i) dist−1(pi,pj)

(5)

wij =

Following [18,32], we opted to utilize relative features fj −fi since it is more efficient than aggregating raw neighborhood features and it enriches the expressivity of the network. To encode Gaussian positions we selected the high-performing multi-resolution hash encoding [23] given its lightweight nature and its ability to expressively encode complex scenes.

Refinement Network. In the final state of the proposed model, the structureaware point encodings are decoded to the 3D Gaussian attributes using four distinct networks, one for each of the attributes; namely position µ ∈ R3, color c ∈ R3, opacity α ∈ R1 and covariance Σ ∈ R3×3. Aligned with 3DGS, we parametrize covariance matrix Σ, with a scale vector S ∈ R3 and a rotation matrix R ∈ R3×3 represented with quaternions. To enforce high rendering speed, we defined each decoder as a small MLP that takes as input the structure-aware encoding and the view-dependent point positions pi and outputs the Gaussian attributes for each point. For example, the color attribute c can be defined as:

ci = MLPc(Φ(pi,fi),pci ) (6)

where MLPc(·) represents the color attribute MLP layer and pci are the viewdepended point positions, normalized with the camera coordinates xc as:

pi − xc ||pi − xc||2

pci =

(7)

Similarly, we predict opacity α, scale S and rotation R attributes using viewdepended point positions. In contrast to the aforementioned view-dependent Gaussian attributes, we opted to learn the 3D scene, represented from the Gaussian mean positions µ, in a camera-agnostic manner. This way we can enforce the model to learn the underlying 3D geometry solely using the world-space point position and shift the bulk of the view-depended attributes to the rest of the MLPs. Additionally, to enforce stable training, we model the 3D Gaussian positions µ as displacement vectors from the initial COLMAP positions:

µi = pi + ∆pi, ∆pi = MLP(Φ(pi,fi)) (8) where pi denotes the initial point derived from structure-from-motion.

SAGS-Lite. In addition to our best-performing model, we present a simple but effective strategy to reduce the storage requirements of the model while retaining the high quality of the renderings and its structural properties. Considering that the predominant storage burden in 3D Gaussian Splatting methods stems from the abundance of stored Gaussians, our objective was to devise a pipeline that would yield a much more compact set of Gaussians, without relying on vector quantization or compression techniques. In particular, leveraging the initial densification step, we can define the midpoints using the initial key points of the COLMAP and predict their Gaussian attributes using the interpolated key-point features.

Under this setting, the mid-points can be generated on the fly and their corresponding features will be interpolated from the structure-aware feature encodings f′ as:

- 1

- 2

fm′ =

(fi′ + fj′), (i,j) ∈ P (9)

where fi′,fj′ are the feature encodings of two keypoints i,j ∈ P and fm′ defines the interpolated feature of their midpoint. Aligned with our full model, the interpolated features along with their corresponding view-depended interpolated positions are fed to the refinement networks to predict their Gaussian attributes. Training. To train our model we utilized a L1 loss and a structural-similarity loss LSSIM on the rendered images, following [15]:

L = (1 − λ)L1 + λLSSIM (10) where λ is set to 0.2.

Implementation. We build our model on top of the original 3D-GS [15] PyTorch implementation. Similar to 3D-GS, we train our method for 30,000 iterations across all scenes and apply a growing and pruning step until 15,000 iterations, every 100 iterations, starting from the 1500 iterations. Throughout our implementation, we utilize small MLPs with a hidden size of 32.

### 4 Experiments

Datasets. To evaluate the proposed method, on par with the 3D-GS [15], we utilized 13 scenes including nine scenes from Mip-NeRF360 [2], two scenes from Tanks&Temples [16] and two scenes from Deep Blending [13] datasets.

Baselines. We compared the proposed method with NeRF- and 3D-GS-based state-of-the-art works in novel-view synthesis, including the Mip-NeRF360 [2], Plenoxels [10], iNGP [23], 3D-GS [15] along with the recent Scaffold-GS [20].

Metrics. We evaluate the proposed SAGS model in terms of rendering quality, structure preservation, and rendering performance. To measure the rendering quality, we utilized the commonly used PSNR, SSIM [33], and LPIPS [38] metrics. In addition, we report model storage requirements in megabytes (MB) and rendering speed in frames per second (FPS).

#### 4.1 Novel-View Synthesis

Rendering Quality. In Tab. 1, we report the average evaluation performance of the proposed and the baseline methods over the three datasets. As can be easily seen, SAGS outperforms 3D-GS and the recently introduced Scaffold-GS method under all datasets and metrics. Leveraging the inter-connectivity between the 3D Gaussians, the SAGS model can facilitate high-quality reconstruction in challenging cases that the independent and unstructured optimization scheme of 3D-GS and Scaffold-GS methods struggle. As can be qualitatively validated in Fig. 4, the proposed SAGS model can better capture high-frequency details, such as the letters on the train wagon, the door handle, and the desk chair mechanism.

- Table 1: Quantitative comparison between the proposed and the baseline methods on Mip-NeRF360 [2], Tanks&Temples [16] and Deep Blending [13] datasets.

Dataset Mip-NeRF360 Tanks&Temples Deep Blending Method Metrics PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

|3D-GS [15] Mip-NeRF360 [2] iNPG [23] Plenoxels [10] Scaffold-GS [20]<br><br>|28.69 0.870 0.182<br><br>29.23 0.844 0.207 26.43 0.725 0.339 23.62 0.670 0.443 28.84 0.848 0.220<br><br><br>|23.14 0.841 0.183<br><br>22.22 0.759 0.257<br><br>21.72 0.723 0.330 21.08 0.719 0.379<br><br>23.96 0.853 0.177<br><br><br>|29.41 0.903 0.243<br><br>29.40 0.901 0.245 23.62 0.797 0.423 23.06 0.795 0.510<br><br>30.21 0.906 0.254<br><br><br>|
|---|---|---|---|
|SAGS-Lite SAGS<br><br>|28.54 0.841 0.225<br><br>29.65 0.874 0.179<br><br><br>|24.16 0.846 0.181 24.88 0.866 0.166<br><br>|29.07 0.889 0.292<br><br>30.47 0.913 0.241<br><br><br>|

Structure Preservation. Apart from the visual quality of the renderings, preserving the 3D geometry of the scene is extremely crucial for downstream VR/AR applications of NVS. Using the proposed structure-aware encoder, we manage to tackle the structure preservation limitations of previous 3D-GS methods and constrain the point displacements close to their initial positions. As pointed out with the red arrows in Fig. 4, this substantially diminishes floater artifacts, which

Ground Truth Scaffold-GS 3D-GS Ours

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

|[Figure 70]| |
|---|---|
| | |

|[Figure 71]| |
|---|---|
| | |

|[Figure 72]| |
|---|---|
| | |

|[Figure 73]| |
|---|---|
| | |

PSNR 21.41 dB SSIM 0.791 LPIPS 0.243 PSNR 21.16 dB SSIM 0.769 LPIPS 0.207 PSNR 22.05 dB SSIM 0.800 LPIPS 0.201

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

PSNR 32.84 dB SSIM 0.933 LPIPS 0.269 PSNR 31.26 dB SSIM 0.929 LPIPS 0.260 PSNR 33.80 dB SSIM 0.937 LPIPS 0.250

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

| |
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

PSNR 23.32 dB SSIM 0.862 LPIPS 0.195 PSNR 23.17 dB SSIM 0.867 LPIPS 0.182 PSNR 23.66 dB SSIM 0.883 LPIPS 0.171

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

PSNR 27.95 dB SSIM 0.804 LPIPS 0.316 PSNR 27.27 dB SSIM 0.807 LPIPS 0.255 PSNR 28.71 dB SSIM 0.838 LPIPS 0.242

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

PSNR 32.36 dB SSIM 0.931 LPIPS 0.162 PSNR 31.10 dB SSIM 0.931 LPIPS 0.171 PSNR 32.90 dB SSIM 0.945 LPIPS 0.145

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

- PSNR 27.81 dB SSIM 0.803 LPIPS 0.293 PSNR 28.32 dB SSIM 0.837 LPIPS 0.157 PSNR 28.87 dB SSIM 0.824 LPIPS 0.203
- PSNR 28.02 dB SSIM 0.924 LPIPS 0.225 PSNR 26.52 dB SSIM 0.900 LPIPS 0.250 PSNR 29.91 dB SSIM 0.939 LPIPS 0.200

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

- Fig. 4: Qualitative comparison. We qualitatively evaluate the proposed and the baseline methods (3D-GS [15] and Scaffold-GS [20]) across six scenes from different datasets. We highlight some detailed differences between the three methods using a magnified crop in yellow. We also emphasize additional visual artifacts using red arrows. The proposed method consistently captures more structural and high-frequency details while minimizing floaters and artifacts compared to the baseline methods.

- were prevalent in the 3D-GS method. To quantitatively validate the structural preservation of our method, in Fig. 5, we illustrate the displacements of points, in a color-coded format, on top of their original positions. In particular, we depict the color-coded displacements for the train scene from the Tanks&Temples dataset, where points with color closer to purple indicate small displacements and colors closer to yellow indicate large displacements. Aligned with the findings of Fig. 1 (right), SAGS better constrains the Gaussians to lie in the original geometry of the scene compared to 3D-GS and Scaffold-GS methods that rely on structure-agnostic optimization to fit the scene.
- Fig. 5: Color Coded Gaussian Displacements. We measured the Gaussians’ displacements from their original positions, on the “train” scene from Tanks&Temples [16] dataset, and encoded them in a colormap scale. Colors closer to purple color indicate small displacements. Both the 3D-GS and Scaffold-GS methodologies depend on a rudimentary point optimization approach, that neglects the local topology and fails to guide the Gaussians in a structured manner.

[Figure 122]

In addition, preserving the geometry of the scene ensures the preservation of spatial relationships and distances between objects, which is significantly important in 3D modeling. The depth information provides crucial cues that verify the spatial distances within a scene and can easily validate the suppression of floater artifacts that are not visible in the rendered image. Therefore, in Fig. 6, we qualitatively evaluate the depth maps that correspond to various scene renderings, generated by the SAGS and the Scaffold-GS methods. The proposed method can not only achieve sharp edges and corners (e.g., on the table and the door) but also accurate high-frequency details (e.g., the tire track of the bulldozer and the staircase banister). On the contrary, Scaffold-GS method provides noisy depth maps that fail to capture the scene’s geometry, e.g., the chair back and the chandelier are modeled with an enormous set of points that do not follow any 3D representation. This is caused by the unstructured nature of the Gaussian optimization that attempts to minimize only the rendering constraints without any structural guidance. Furthermore, Scaffold-GS method falls short in accurately representing flat surfaces, as can be seen in the walls and the table, compared to SAGS which accurately captured flat surfaces.

Performance. Apart from the rendering quality we evaluated the performance of the proposed and the baseline methods in terms of rendering speed (FPS) and

[Figure 123]

[Figure 124]

###### 12 E. Ververas et al.

Scaﬀold-GS Scaﬀold-GS Depth SAGS SAGS Depth

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 131]

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

[Figure 142]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Fig. 6: Depth Structural Preservation. Comparison between the proposed and the Scaffold-GS method on the scene’s structure preservation. The proposed method can accurately capture sharp edges and suppress “floater” artifacts that are visible on the Scaffold-GS depth maps.

- Table 2: Performance comparison between the proposed and the 3D-GS based methods. We also report the storage reduction of each model compared to original 3D-GS method [15].

Dataset Mip-NeRF360 Tanks&Temples Deep Blending Methods FPS Mem (MB) FPS Mem (MB) FPS Mem (MB)

3D-GS [15] 97 693 123 411 109 676 Scaffold-GS [20] 102 252 (2.8× ↓) 110 87 (4.7× ↓) 139 66 (10.2× ↓) Ours 110 135 (5.1× ↓) 108 75 (5.5× ↓) 138 58 (11.7× ↓) Ours-Lite 101 76 (9.1× ↓) 112 35 (12× ↓) 142 28 (24× ↓)

storage size (MB) under all datasets. In Tab. 2, we report the FPS and memory sizes for each method averaged per dataset. Both SAGS and SAGS-Lite models achieve a real-time rendering speed, with over 100 FPS under all scenes, on par with the Scaffold-GS and 3D-GS methods. Importantly, SAGS reduces the storage requirements of 3D-GS by 5× on the challenging MipNeRF360 dataset [2] achieving state-of-the-art performance with an average model of 135MB. The storage requirements are reduced even more with our lightweight model that can achieve up to 24×-storage reduction compared to 3D-GS without relying on any compression scheme.

[Figure 155]

Comparison with SAGS-Lite. In Fig. 7, we qualitatively evaluate the lightweight version of the proposed SAGS model. As shown in Tab. 2, the SAGS-Lite model can drastically reduce the storage requirements of 3D-GS by up to 28 times, while achieving similar rendering performance with 3D-GS [15]. Despite lacking some sharp details when compared to our full SAGS model, SAGS-Lite

|[Figure 156]<br><br>Ground Truth|SAGS: Struc<br><br>[Figure 157]<br><br>3D-GS<br><br>PSNR: 25.25dB LPIPS: 0.205 Mem: 1.3 Gb|ture-Aware 3D Gauss<br><br>[Figure 158]<br><br>SAGS-Lite<br><br>PSNR: 24.41dB LPIPS: 0.287 Mem: 113 Mb|ian Splatting 13<br><br>[Figure 159]<br><br>SAGS<br><br>PSNR: 25.30dB LPIPS: 0.241 Mem: 195 Mb|
|---|---|---|---|
|[Figure 160]|[Figure 161]<br><br>PSNR: 27.41dB LPIPS: 0.103 Mem: 1.3 Gb|[Figure 162]<br><br>PSNR: 25.62dB LPIPS: 0.230 Mem: 76 Mb|[Figure 163]<br><br>PSNR: 27.25dB LPIPS: 0.161 Mem: 128 Mb|
|[Figure 164]|[Figure 165]<br><br>PSNR: 30.32dB LPIPS: 0.129 Mem: 414 Mb|[Figure 166]<br><br>PSNR: 30.70dB LPIPS: 0.137 Mem: 43 Mb|[Figure 167]<br><br>PSNR: 31.96dB LPIPS: 0.115 Mem: 99 Mb|
|[Figure 168]<br><br>Fig. 7: Comparison SAGS-Lite model aga quality renderings wit|[Figure 169]<br><br>with SAGS-Lite. inst SAGS and 3DGS h up to 24× storage|[Figure 170]<br><br>We qualitatively com<br><br>. SAGS-Lite can achi reduction compared to|[Figure 171]<br><br>pared the proposed eve to maintain high<br><br>3DGS.|

PSNR: 28.77 LPIPS: 0.244 Mem: 715 Mb PSNR: 28.61 LPIPS: 0.285 Mem: 32 Mb PSNR: 29.95 LPIPS: 0.235 Mem: 63 Mb

can accurately represent the scene while at the same time avoid the floater artifacts caused by 3D-GS optimization.

#### 4.2 Ablation Study

To evaluate the impact of individual components within the proposed method, we conducted a series of ablation studies using the Deep Blending and Tanks&Temples datasets. The quantitative and qualitative results of these studies are presented in Tab. 3 and Fig. 8, respectively.

Effect of the Densification. First, removing the curvature-aware densification step leads to parts of the scene being under-represented in the resulting point cloud, as the gradient-based growing struggles to sufficiently fill them. This is particularly evident in areas of COLMAP which are initially undersampled, such as the floor in the drjohnson scene and the gravel road in the train scene. In contrast, our curvature-aware densification adequately fills those areas supporting further growing during training.

Effect of the Structure-Aware Encoder. Replacing the aggregation layer in Eq. (4) with an MLP diminishes the inductive biases and the expressivity of the model to encode the scene’s structure, leading to Gaussians with locally inconsistent attributes. As can be seen in Fig. 8, the absence of structure results in renderings with more floaters and artifacts, which are evident in both the drjohnson and train scenes.

Feature Analysis. Using the points’ positions directly in Eq. (4) instead of their positional encodings γ(p), results in lower resolution representations of the scene which implies less high frequency details in renderings. A similar effect is

###### Table 3: Ablation study on the components of SAGS. The ablation was performed on the Deep Blending and the Tanks&Temples datasets.

###### Scene Deep Blending Tanks&Temples Ablation Metrics PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

|w/o Curvature-Aware Densification w/o GNN w/o Positional-Encoding γ(p) w/o Global Feature g w/o View Dependent Positions pci<br><br>|29.87 0.901 0.259<br><br>29.94 0.905 0.254<br>30.21 0.909 0.252<br><br><br>30.17 0.911 0.250 30.07 0.903 0.256<br>|23.97 0.851 0.175<br><br>24.19 0.844 0.181<br><br><br>24.31 0.852 0.169 24.42 0.861 0.174 24.37 0.849 0.173<br><br>|
|---|---|---|
|SAGS|30.47 0.913 0.241<br><br>|24.88 0.866 0.166|

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

PSNR: 27.81 dB SSIM: 0.796 LPIPS: 0.277

PSNR: 28.06 dB SSIM: 0.814 LPIPS: 0.251

PSNR: 28.41 dB SSIM: 0.821 LPIPS: 0.254

a) w/o upsampling b) w/o GNN c) w/o positional enc.

| |
|---|

[Figure 176]

[Figure 177]

[Figure 178]

Ground Truth

PSNR: 28.34 dB SSIM: 0.827 LPIPS: 0.249

PSNR: 28.12 dB SSIM: 0.811 LPIPS: 0.260

PSNR: 28.71 dB SSIM: 0.838 LPIPS: 0.242

d) w/o global feature e) w/o view dep. positions f) SAGS

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

- PSNR: 21.54 dB SSIM: 0.764 LPIPS: 0.178

PSNR: 21.89 dB SSIM: 0.769 LPIPS: 0.174

PSNR: 22.31 dB SSIM: 0.772 LPIPS: 0.171

- PSNR: 22.39 dB SSIM: 0.773 LPIPS: 0.169

a) w/o upsampling b) w/o GNN c) w/o positional enc.

[Figure 183]

[Figure 184]

[Figure 185]

Ground Truth

PSNR: 22.47 dB SSIM: 0.774 LPIPS: 0.176

PSNR: 22.78 dB SSIM: 0.780 LPIPS: 0.164

d) w/o global feature e) w/o view dep. positions f) SAGS

Fig. 8: Ablation study on the components of SAGS. We perform a series of ablation experiments on the Deep Blending and the Tanks&Temples datasets and demonstrate qualitative results from the drjohnson and the train scenes. We emphasise differences across model configurations over the same crop of the resulting images and highlight additional visual artifacts using red arrows.

caused by removing the global structure information offered by g, which leads to less expressive feature encodings Φ(pi,fi) limiting the capacity of the refinement network and the quality of its predictions. For both previous configurations, the floor in the drjohnson scene and areas of the train scene demonstrate parts with flat textures. Last, by ablating the view dependent positions pci from the appearance related attributes resulted in missing reflections, for example on the floor in drjohnson, and floating points as on the black cable in the train scene.

### 5 Conclusion

In this paper, we present Structure-Aware Gaussian Splatting (SAGS), a novel Gaussian Splatting approach that leverages the intrinsic scene structure for highfidelity neural rendering. Motivated by the shortcomings of current 3D Gaussian Splatting methods to naively optimize Gaussian attributes independently neglecting the underlying scene structure, we propose a graph neural network based approach that predicts Gaussian’s attributes in a structured manner. Using the proposed graph representation, neighboring Gaussians can share and aggregate information facilitating scene rendering and the preservation of its geometry. We showcase that the proposed method can outperform current state-of-the-art methods in novel view synthesis while retaining the real-time rendering of 3D-GS. We further introduce a simple yet effective mid-point interpolation scheme that attains up to 24×-storage reduction compared to 3D-GS method while retaining high-quality rendering, without the use of any compression and quantization algorithm. Overall, our findings demonstrate the benefits of introducing structure in 3D-GS.

### References

- 1. Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srinivasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. 2021 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 5835–5844 (2021), https://api.semanticscholar.org/CorpusID:232352655
- 2. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR (2022)
- 3. Chen, Z., Funkhouser, T., Hedman, P., Tagliasacchi, A.: Mobilenerf: Exploiting the polygon rasterization pipeline for efficient neural field rendering on mobile architectures. In: The Conference on Computer Vision and Pattern Recognition (CVPR) (2023)
- 4. Chen, Z., Wang, F., Liu, H.: Text-to-3d using gaussian splatting. arXiv preprint arXiv:2309.16585 (2023)
- 5. Cui, H., Gao, X., Shen, S., Hu, Z.: Hsfm: Hybrid structure-from-motion. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1212–1221 (2017)
- 6. Cui, H., Shen, S., Gao, X., Hu, Z.: Batched incremental structure-from-motion. In: 2017 International Conference on 3D Vision (3DV). pp. 205–214. IEEE (2017)
- 7. Deng, N., He, Z., Ye, J., Duinkharjav, B., Chakravarthula, P., Yang, X., Sun, Q.: Fov-nerf: Foveated neural radiance fields for virtual reality. IEEE Transactions on Visualization and Computer Graphics 28(11), 3854–3864 (2022)
- 8. DeTone, D., Malisiewicz, T., Rabinovich, A.: Superpoint: Self-supervised interest point detection and description. In: Proceedings of the IEEE conference on computer vision and pattern recognition workshops. pp. 224–236 (2018)
- 9. Fan, Z., Wang, K., Wen, K., Zhu, Z., Xu, D., Wang, Z.: Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps (2023)
- 10. Fridovich-Keil, S., Yu, A., Tancik, M., Chen, Q., Recht, B., Kanazawa, A.: Plenoxels: Radiance fields without neural networks. In: CVPR (2022)

- 11. Goesele, M., Snavely, N., Curless, B., Hoppe, H., Seitz, S.M.: Multi-view stereo for community photo collections. In: 2007 IEEE 11th International Conference on Computer Vision. pp. 1–8. IEEE (2007)
- 12. He, Y., Tang, D., Zhang, Y., Xue, X., Fu, Y.: Grad-pu: Arbitrary-scale point cloud upsampling via gradient descent with learned distance functions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

(2023)

- 13. Hedman, P., Philip, J., Price, T., Frahm, J.M., Drettakis, G., Brostow, G.: Deep blending for free-viewpoint image-based rendering 37(6), 257:1–257:15 (2018)
- 14. Kaya, B., Kumar, S., Oliveira, C., Ferrari, V., Van Gool, L.: Uncertainty-aware deep multi-view photometric stereo. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12601–12611 (2022)
- 15. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4) (July 2023), https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- 16. Knapitsch, A., Park, J., Zhou, Q.Y., Koltun, V.: Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics 36(4) (2017)
- 17. Lee, J.C., Rho, D., Sun, X., Ko, J.H., Park, E.: Compact 3d gaussian representation for radiance field. arXiv preprint arXiv:2311.13681 (2023)
- 18. Li, G., Muller, M., Thabet, A., Ghanem, B.: Deepgcns: Can gcns go as deep as cnns? In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9267–9276 (2019)
- 19. Lowe, D.G.: Distinctive image features from scale-invariant keypoints. International journal of computer vision 60, 91–110 (2004)
- 20. Lu, T., Yu, M., Xu, L., Xiangli, Y., Wang, L., Lin, D., Dai, B.: Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2024)
- 21. Martin-Brualla, R., Radwan, N., Sajjadi, M.S.M., Barron, J.T., Dosovitskiy, A., Duckworth, D.: NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2021)
- 22. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021)
- 23. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG) 41(4), 1– 15 (2022)
- 24. Niedermayr, S., Stumpfegger, J., Westermann, R.: Compressed 3d gaussian splatting for accelerated novel view synthesis (2023)
- 25. Pauly, M., Gross, M., Kobbelt, L.P.: Efficient simplification of point-sampled surfaces. In: IEEE Visualization, 2002. VIS 2002. pp. 163–170. IEEE (2002)
- 26. Qi, C.R., Yi, L., Su, H., Guibas, L.J.: Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems 30 (2017)
- 27. Sarlin, P.E., DeTone, D., Malisiewicz, T., Rabinovich, A.: Superglue: Learning feature matching with graph neural networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4938–4947 (2020)
- 28. Schonberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4104–4113

(2016)

- 29. Snavely, N., Seitz, S.M., Szeliski, R.: Photo Tourism: Exploring Photo Collections in 3D. Association for Computing Machinery, New York, NY, USA, 1 edn. (2023), https://doi.org/10.1145/3596711.3596766
- 30. Tancik, M., Casser, V., Yan, X., Pradhan, S., Mildenhall, B., Srinivasan, P.P., Barron, J.T., Kretzschmar, H.: Block-nerf: Scalable large scene neural view synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8248–8258 (2022)
- 31. Tang, J., Ren, J., Zhou, H., Liu, Z., Zeng, G.: Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653 (2023)
- 32. Wang, Y., Sun, Y., Liu, Z., Sarma, S.E., Bronstein, M.M., Solomon, J.M.: Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog) 38(5), 1–12 (2019)
- 33. Wang, Z., Bovik, A., Sheikh, H., Simoncelli, E.: Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600–612 (2004). https://doi.org/10.1109/TIP.2003.819861
- 34. Wiles, O., Gkioxari, G., Szeliski, R., Johnson, J.: Synsin: End-to-end view synthesis from a single image. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 7465–7475 (2019), https://api.semanticscholar.org/ CorpusID:209405397
- 35. Xiangli, Y., Xu, L., Pan, X., Zhao, N., Rao, A., Theobalt, C., Dai, B., Lin, D.: Bungeenerf: Progressive neural radiance field for extreme multi-scale scene rendering. In: Avidan, S., Brostow, G.J., Cissé, M., Farinella, G.M., Hassner, T. (eds.) Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XXXII. Lecture Notes in Computer Science, vol. 13692, pp. 106–122. Springer (2022). https://doi.org/10.1007/978-3-03119824-3_7, https://doi.org/10.1007/978-3-031-19824-3_7
- 36. Xiong, H., Muttukuru, S., Upadhyay, R., Chari, P., Kadambi, A.: Sparsegs: Realtime 360 {\deg} sparse view synthesis using gaussian splatting. arXiv preprint arXiv:2312.00206 (2023)
- 37. Yifan, W., Serena, F., Wu, S., Öztireli, C., Sorkine-Hornung, O.: Differentiable surface splatting for point-based geometry processing. ACM Transactions on Graphics (TOG) 38(6), 1–14 (2019)
- 38. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2018)
- 39. Zhuang, B., Cheong, L.F., Lee, G.H.: Baseline desensitizing in translation averaging. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 4539–4547 (2018)

