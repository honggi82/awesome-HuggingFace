# arXiv:2409.11211v1[cs.CV]17Sep2024

## SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction

Marko Mihajlovic1∗ , Sergey Prokudin1,3 , Siyu Tang1 , Robert Maier2 , Federica Bogo2 , Tony Tung2 , and Edmond Boyer2 ETH Zürich1 Meta Reality Labs2 Balgrist University Hospital3

markomih.github.io/SplatFields

Abstract. Digitizing 3D static scenes and 4D dynamic events from multi-view images has long been a challenge in computer vision and graphics. Recently, 3D Gaussian Splatting (3DGS) has emerged as a practical and scalable reconstruction method, gaining popularity due to its impressive reconstruction quality, real-time rendering capabilities, and compatibility with widely used visualization tools. However, the method requires a substantial number of input views to achieve high-quality scene reconstruction, introducing a significant practical bottleneck. This challenge is especially severe in capturing dynamic scenes, where deploying an extensive camera array can be prohibitively costly. In this work, we identify the lack of spatial autocorrelation of splat features as one of the factors contributing to the suboptimal performance of the 3DGS technique in sparse reconstruction settings. To address the issue, we propose an optimization strategy that effectively regularizes splat features by modeling them as the outputs of a corresponding implicit neural field. This results in a consistent enhancement of reconstruction quality across various scenarios. Our approach effectively handles static and dynamic cases, as demonstrated by extensive testing across different setups and scene complexities.

Keywords: Novel view synthesis · Gaussian splatting · Implicit models

### 1 Introduction

Building a realistic replica of static and dynamic environments can revolutionize the world by transforming the way we interact, work, and engage online [69]. This ambitious vision has motivated a surge in recent research to develop new representations and rendering techniques that allow for comprehensive and photorealistic capture and reconstruction of scenes from multi-view imagery.

Recent advancements, notably the introduction of Neural Radiance Fields (NeRF) [47], have shown exceptional quality in photorealistic 3D reconstruction from casually captured images. This success comes from modeling a 3D scene as a neural field [79] and optimizing it through volume rendering techniques. The parameterization of the rendering volume using a continuous differentiable field presents several benefits. It enables a compact representation of the scene’s geometry and appearance through neural network weights, offering a more practical

∗Partly done during an internship at Meta.

3D Gaussian Splatting (3DGS) SplatFields 10 input views

[Figure 1]

[Figure 2]

|Novel Views:<br><br>𝑆𝑆𝐼𝑀 ↑:𝟗𝟏.𝟑𝟒 𝑃𝑆𝑁𝑅 ↑:𝟐𝟔.𝟓𝟏|
|---|

|Novel Views:<br><br>𝑆𝑆𝐼𝑀 ↑:86.30 𝑃𝑆𝑁𝑅 ↑:23.79|
|---|

[Figure 3]

[Figure 4]

|Spatial Autocorrelation<br><br>(Moran’s I) ↑ Covariance: Opacity: Color:<br><br>0.40 0.74 0.54|
|---|

|Spatial Autocorrelation<br><br>(Moran’s I) ↑ Covariance: Opacity: Color:<br><br>0.68 0.83 0.78|
|---|

- Fig. 1: SplatFields regularizes 3D Gaussian Splatting (3DGS) [29] by predicting the splat features and locations via neural fields to improve the reconstruction under unconstrained sparse views. We measure spatial autocorrelation (Moran’s I [48]) of splat features in the local neighborhoods to assess their similarity and observe that better reconstruction quality achieved by our method corresponds to higher Moran’s I. The figure presents the results of a static reconstruction from ten calibrated images from Blender dataset [47]. Metrics are reported on the full test set; the rendered view is a novel view.

alternative to explicit volume modeling, which is often unfeasible. Crucially, for the focus of this work, the continuous nature and the spectral bias [60] of MultiLayer Perceptrons (MLPa) introduce a spatial bias—nearby primitives are likely to exhibit similar features as predicted by the neural field MLP. This concept of implicitly modeling spatiotemporal signals has captured the research community’s attention in recent years [20], marking a significant shift in methods for 3D scene reconstruction and novel view synthesis. A substantial portion of research has also focused on adapting these methods for sparse view setups [50,89] and enhancing training and rendering efficiency [8,49,54].

3D Gaussian Splatting (3DGS) [29] offers an alternative 3D reconstruction framework using point-based rasterization rather than computationally demanding volume rendering. The method quickly gained traction within both the computer vision and graphics communities due to its real-time rendering capabilities, potential compatibility with the standard rasterization pipelines, and the intuitive way of editing and combining the reconstructed scenes. This makes 3DGS a practical and scalable solution that is currently being rapidly adopted and supported by many 3D development platforms and visualization tools [9,18,72].

3D Gaussian Splatting represents the 3D scene as a set of unordered 3D Gaussian primitives, rendered from arbitrary views via rasterization, akin to traditional point splatting techniques [4,94,95]. Each rendering primitive comprises trainable parameters such as position, orientation, scale, color, and opacity, which are optimized by rendering the representation with respect to multiview input images. The flexible parameterization, coupled with the efficient rasterization framework, is key to high-quality novel view synthesis results at scale. However, the flexibility of rendering primitives comes at the cost of requiring a

large number of input views to fully constrain the optimization process, making Gaussian splatting unsuitable for more practical captures from sparse views.

We analyze the performance of 3DGS and its 4D variants [84,85] in sparse input view scenarios. We first show that splat-based techniques, with their independently modeled set of rendering primitives, are particularly vulnerable to the training view overfitting in such cases (see Fig. 1). In contrast, volumetric rendering techniques [71] which imply shared feature representations appear to be more robust in such scenarios as demonstrated in [45], at the expense of a significantly increased training time and suboptimal rendering efficiency. This key observation provides the basis for the method introduced in this work.

Our key idea is to regularize the behavior of independent Gaussian primitives by utilizing neural networks that regress splat features at different levels. First, inspired by [68], we aim to enforce the spatial bias through a hierarchical convolutional decoder [55] that outputs a tri-plane representation [7] of deep features associated with each splat. Please note that the tri-plane representation and the associated network are utilized only during the optimization phase to constrain the attributes of the Gaussian primitives; both are discarded thereafter for accelerated rendering and compatibility with established splat rasterization pipelines. The produced deep splat features are then utilized to condition neural fields [79] that model the geometric and appearance properties of Gaussian splats at various locations and time steps. This design is equipped with positional encoding [47] to represent high-frequency details while retaining the good spatial properties required to regularize Gaussian splatting.

We thoroughly analyze our representation (dubbed SplatFields) and demonstrate its superior reconstruction quality under sparse input views compared to alternative 3D Gaussian splatting techniques [16,22,29,91]. We further present an effective extension of our optimization framework to model dynamic 4D scenes and propose a new forward-flow field formulation to model the dynamics of Gaussian splats, warping rendering primitives into the observation space. We observe that existing techniques that model 3D splat deformations either lack the modeling capacity due to simplified assumptions on scene motion [85] or have an insufficient spatial bias in the model [78], leading to suboptimal performance in sparse setups. Therefore, we introduce a forward-flow neural network for 3D Gaussians based on the recent ResFields MLP architecture [45]. Our method outperforms recent baselines [78, 84, 85] while retaining the key properties of Gaussian splatting, such as rendering efficiency and compatibility with existing frameworks. In summary, our key contributions are:

- – We propose a novel optimization strategy, named SplatFields, which introduces spatial bias into the 3D Gaussian Splatting technique to stabilize the optimization process under sparse views.
- – We extend our formulation to dynamic scenes, demonstrating superior reconstruction quality compared to recent state-of-the-art methods [78,84,85].
- – We provide a detailed analysis of various modeling strategies, confirming the optimality of our framework for the tasks of sparse multi-view reconstruction.

The code is publicly available: markomih.github.io/SplatFields.

### 2 Related Work

Implicit volumetric rendering [14]. Novel View Synthesis (NVS) enables the generation of new images from arbitrary viewpoints using a given set of input images [47,71]. Over the past few years, the predominant method of choice for NVS has been the Neural Radiance Field (NeRF) [47], which represents a 3D scene as a continuous neural field [79]. This field takes as input a location and view direction and predicts color and density. Then, the color of a pixel from an arbitrary viewpoint is rendered by casting a ray and employing volume rendering, which requires sampling multiple points along the ray and converting them into color and density values by querying the neural field. Numerous extensions have been proposed to handle various scenarios and setups such as sparse view reconstruction [10,12,13,27,44,50,89,92], dynamic [5,37,52,53,57,74] and large scale unbounded scenes [3,70]. However, the implicit volume rendering process is inherently expensive due to the large number of sampled points whose predictions need to be integrated. Despite recent efforts to accelerate NeRFs [7,8,36,49,61,66], achieving interactive rendering capabilities for regular scenes remains challenging without additional post-processing or compression [15,62,86,88].

Point-based rendering. The limitations of volumetric rendering methods have led to a resurgence in point-based techniques [1]. The seminal work by [21] introduced the rasterization of fixed-size, unstructured point samples for NVS. However, this naive rendering approach often results in aliasing artifacts and images with holes. These issues have been partially addressed by employing splatting techniques, where points are rendered with extended sizes to cover multiple pixels, using shapes like circular ellipsoids or surfels [4,94,95]. The era of deep learning led to a new wave of point-based neural rendering methods which allowed differentiable point rendering [34,64,77,87] and combined point-based rasterizers with 2D convolutional networks [2,33,46].

3D Gaussian Splatting (3DGS [29]) utilizes the volumetric composition of ordered splats to merge the advantages of volumetric representations with exceptional real-time rendering capabilities. Gaining rapid popularity due to its efficiency, 3DGS has been incorporated into a wide range of downstream tasks [9,18]. Modifications to the original framework have enhanced its robustness to novel views [91], improved geometry reconstruction [22], and reduced model size [16]. Nevertheless, 3DGS’s dependence on numerous independent splatting primitives necessitates a large number of views for effective optimization, impacting its performance in sparse view reconstructions. Our work introduces neural networks to regularize splat behavior by regressing splat features based on their 3D location, introducing the spatial autocorrelation bias which substantially enhances reconstruction in sparse scenarios, as demonstrated in our experiments.

Dynamic Gaussian splatting. Several recent modifications of 3DGS have been proposed to extend its capabilities to dynamic sequences. The dynamic 3DGS [43] extends the basic pipeline by optimizing the motion of each splatting primitive and the change in its corresponding features. Although regularizations such as the enforcement of local rigidity and isometry losses [30,56] help stabilize the learning process to a certain extent, the overall pipeline still requires a large

number of input views for each step and struggles to reconstruct scenes faithfully from sparse observations. The works closest to ours are [84, 85], which utilize MLPs to model time-dependent deformations of Gaussian splats. However, the single MLP used for modeling deformation in [85] lacks the capacity to represent non-trivial scene dynamics, while the parameterizations used in [78, 84] lack a substantial spatial and temporal autocorrelation bias, leading to suboptimal reconstructions in sparse view scenarios. We thoroughly analyze the behavior of the aforementioned methods and compare them with our approach on several dynamic scenes of varying complexity and view sparsity. We demonstrate that our model, which combines a triplane-based CNN generator [7] for splat features with ResFields-based [45] dynamics modeling, offers the optimal combination of expressivity and robustness in sparse capture scenarios.

A growing body of work also addresses dynamic scenes [11,26,38,40,42,90] and template-based approaches [35] for modeling full-body [23, 24, 32, 51, 59, 81] and head avatars [58, 65]. In contrast, our model is capable of handling unbounded, topologically varying generic dynamic scenes.

### 3 Preliminaries: 3D Gaussian Splatting

In the following, we provide a brief overview of the Gaussian splatting rendering technique, which is a fundamental building block of our model.

Scene representation. 3DGS [29] parametrizes the 3D scene via static

- 3D Gaussian primitives {Gk}Kk=1 that contain the geometric and appearance information. These rendering primitives are utilized for efficient differentiable rasterization-based volume splatting.

The geometry of each Gaussian splat Gk is defined by the mean location pk ∈ R3×1, the opacity value αk ∈ [0,1], and the covariance matrix Σk ∈ R3×3 defined in the world space. Each splatting primitive Gk then induces the following Gaussian distribution in 3D space:

Gk(x) ∝ exp −

- 1

- 2

(x − pk)TΣ−k 1(x − pk) , (1)

where the covariance matrix is modeled by the scaling vector sk ∈ R3 and the rotation matrix Ok ∈ R3×3 (parameterized via quaternions) to ensure positive semi-definiteness:

Σk = OksksTk OTk . (2) The appearance of splats is view-dependent and described by C coefficients

that are converted to color ck ∈ R3 via spherical harmonics, similar to [88].

Rendering. Given an arbitrary camera viewpoint described by the rotation R ∈ R3×3 and translation t ∈ R3×1, we can obtain the 2D coordinates of the splat center p′k ∈ R2 on the image plane:

p′k = (Rpk + t)1:2/(Rpk + t)3. (3)

Further, we can obtain the 2D projection Σ2Dk of the covariance matrix on the image plane:

Σ2Dk = JkΣ′kJTk 1:2,1:2 ∈ R2×2, where Σ′k = RΣkRT, (4)

and the Jacobian Jk ∈ R3×3 is an affine approximation to the projective transformation (see [94] for details). The subscript 1:2 denotes row and column selection.

Using the image-space splat center and 2D covariance matrix, we obtain the

- 2D image-space Gaussian distribution induced by the corresponding splat:

- 1

- 2

Gk2D(x′) ∝ exp −

(x′ − p′k)T(Σ2Dk )−1(x′ − p′k) , (5)

Finally, we can predict the color c(x′) ∈ R3 at each pixel location x′ ∈ R2 by blending the splats, sorted according to their projection depth:

k−1 j=1

K k=1

1 − αjGj2D(x′) , (6) where αk is the learned opacity of the splat.

ckαkGk2D(x′)

c(x′) =

Training. The collection of splats {Gk}Kk=1 is optimized by minimizing the following rendering loss w.r.t the input images via the Adam [31] optimizer:

L = (1 − λ)L1 + λLD-SSIM, (7)

where the first term is a standard L1 loss between the target and rendered images, and LD-SSIM is a differentiable version of structural similarity index [76]. As this optimization is highly sensitive to a local minima, 3DGS additionally employs periodic adaptive densification and pruning of splats through randomized sampling. We refer to [29] for further details.

### 4 SplatFields: Neural Gaussian Splats

Limitations of 3DGS. Modeling 3D scenes with irregularly spaced point primitives offers significant flexibility and facilitates rapid and efficient optimization when an extensive number of training views is provided. However, with limited views, these independent point primitives are prone to overfitting. Therefore, we advocate for integrating a spatial autocorrelation bias within the splats. This can be accomplished by deriving splat features through implicit neural models, presenting a viable method to constrain and regularize the otherwise ill-posed optimization in sparse-view environments.

Key insight. We analyze the optimization procedure of 3DGS under sparse view inputs and observe (Fig. 1) that the splats do not exhibit any local structure and display incoherent patterns. To quantify the local spatial autocorrelation of each splat, we select the five nearest neighbors and measure Moran’s I [48] of the splat’s attributes (color, opacity, covariance). We note that a low level of spatial autocorrelation is associated with overfitting to training views, which impedes

|Adaptive Density Control|
|---|

𝛜 ∼ 𝒩 0,𝕀

|CNN Generator 𝑔|
|---|

||Color Field 𝑓 𝐜|
|---|
<br><br>|Scale Field 𝑓 𝐬|
|---|
<br><br>|Opacity Field 𝑓|
|---|
<br><br>|Rotation Field 𝑓 𝐎|
|---|
<br><br>View Direction<br><br>|Flow Field 𝑓 𝐩|
|---|
|
|---|

𝐜 𝐬 𝛼 𝐎

[Figure 5]

[Figure 6]

[Figure 7]

| | | |
|---|---|---|
| | | |
| | | |

𝐩 𝐟 𝐩 ,𝐟 𝐩 ,𝐟

[Figure 8]

[Figure 9]

Image

|Deform MLP 𝑓|
|---|

|3DGS Rasterizer|
|---|

[Figure 10]

[Figure 11]

[Figure 12]

Gaussian Centers 𝐩 ∈ ℝ

Feature Set

Time 𝑡

[Figure 13]

𝐩 Camera

𝒑 ,𝒇 Triplane 𝑭

3D Splats 𝒢

Time 𝑡

- Fig. 2: Overview. SplatFields takes as input a point cloud (e.g., initialized from SfM [67]), for which it models the geometric (position pk, scale sk, rotation Ok) and appearance attributes (color ck, opacity αk). These attributes represent the point set as 3D splats that are then rendered with the 3DGS rasterizer [29]. First, the point

location set {pk ∈ R3}Kk=1 is encoded into features {fk}Kk=1 by sampling the tri-plane representation generated by a CNN generator gθ to provide a deep structural prior [73] on the feature values. These values are then propagated through a deformation MLP fΘ to refine the point locations pˆk. The new point set, along with the features, is then propagated through a series of compact neural fields to predict the properties of rendering primitives {Gk}Kk=1 that are rendered with respect to arbitrary viewpoints. During the optimization, we adopt the adaptive density control [29] to periodically prune and densify the point set. SplatFields seamlessly adapts to 4D reconstruction by conditioning neural fields on the time step t and introducing an extra time-conditioned flow field. Gray blocks indicate learnable modules.

the learning of a structured 3D representation. This correlation is further showcased in our discussion and empirical evidence presented in Tab. 1.

The core idea of our method is to introduce a spatial bias during the optimization phase, which encourages nearby primitives to share similar features, thereby emulating the more continuous behavior characteristic of widely used implicit representations for volumetric rendering [45, 47, 71]. However, directly enforcing this constraint—ensuring local neighborhoods exhibit common patterns—yields sub-optimal performance (Tab 2) for volumetric point representations. To overcome this, we propose a novel neural framework, termed SplatFields, designed to adaptively regularize the optimization of 3DGS. Importantly, SplatFields straightforwardly extends to 4D, facilitating the reconstruction of dynamic scenes.

SplatFields (Fig. 2) builds on the core property of neural networks to discover local patterns and fit low frequencies of a signal first [60,73]. To that end, we implement SplatFields as a neural generator that infers the attributes of Gaussian splats. The neural generator combines key properties of convolutional neural networks, which model local structured patterns, with multi-layer perceptrons that serve as global approximators. This approach straightforwardly extends to

- 4D reconstruction by conditioning the MLP networks on time t. Deep structural prior. First, we follow the idea of a deep image prior [68,

73] and aim to utilize CNNs to model locally structured patterns of splat fea-

- Table 1: Impact of the spatial autocorrelation on static scene reconstruction. Results on Owlii [80] dataset. See Section 5.1 for discussion

Train Test Spatial Autocorrelation

View Synthesis Novel View Synthesis (Moran’s I) ↑

SSIM↑ PSNR↑ LPIPS↓ SSIM↑ PSNR↑ LPIPS↓ Color Opacity Covariance

- 3DGS [29] 99.85 44.07 0.493 91.68 27.50 8.881 0.547 0.670 0.232 SplatFields3D 98.87 37.58 3.274 96.13 30.33 5.973 0.935 0.874 0.431

tures. In the original work [73], the CNN takes as input low-dimension Gaussian noise ϵ ∼ N(0,I) and gradually upsamples it into the desired image resolution; the weights of the network are then optimized to fit the observed noisy image. In our case, we aim to generate a 3D field of splat features; as 3D CNNs are computationally prohibitive, we use 2D CNNs that generate axis-aligned tri-plane representations [7,55]. Overall, the step is a splat-based variation of the approach utilized in [68] for a fully volumetric NeRF-based sparse rendering.

Specifically, given a randomly initialized noise ϵ, the convolutional network gθ regresses the three H × W-resolution planes F:

F = gθ(ϵ) ∈ R3×H×W×l , (8)

where l denotes the feature dimension and θ indicates learnable network weights. The overall CNN structure resembles the one originally proposed in [68].

Neural splat fields. Next, the splat center pk is projected onto each of the three feature planes to obtain feature vectors via bilinear interpolation. These features are then concatenated along the feature dimension and denoted as fk ∈ R3l. The feature and the initial point are propagated through a deformation MLP fΘ which refines the position of the input point:

pˆk = fΘ(pk,fk,t), (9) where t indicates an optional time step input provided in the case of dynamic

- 4D reconstruction. Finally, the updated point location, along with the inferred feature vector, is provided as input to a set of compact (5-6 layers, 64-128 neu-

rons) neural fields {fΘ

O} to obtain properties of Gaussian splats. Rather than using spherical harmonics to model color, we directly predict viewdependent color. The obtained splats are then rendered w.r.t. the input views to optimize the learnable modules by minimizing the photometric loss (Eq. 7).

, fΘ

, fΘ

, fΘ

c

s

α

Splat norm regularization. For static reconstruction, we add additional norm regularization ||pˆk||2 to the loss function to bias the resulting splats to not deviate significantly from the origin, similar to the floor loss considered in [43].

4D reconstruction. SplatFields is a flexible representation that straightforwardly extends to 4D reconstruction of dynamic scenes. It models temporal variations in the splat features by conditioning the corresponding neural field MLPs {fΘ,fΘ

O} on the time step t. Additionally, we add a timeconditioned flow field fp to warp the center of Gaussians pk to the desired time step t. To enhance the expressivity of the neural fields and allow for complex geometry changes, we utilize the recently proposed ResField MLP architecture [45].

, fΘ

, fΘ

, fΘ

c

s

α

##### Table 3: Ablation study of SplatFields. Blender dataset [47], setup from Tab. 2

12 Views 10 Views 8 Views 6 Views 4 Views

SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑

basic (MLP-only) 89.63 24.82 88.69 23.93 87.79 23.32 85.70 21.58 81.98 19.07 +L2-norm reg. 89.66 24.98 88.81 24.21 87.98 23.64 85.93 21.79 82.44 19.46 +tri-CNN 90.83 25.23 90.04 24.66 88.01 23.19 85.91 21.46 81.28 18.72 full model 91.18 25.80 90.32 24.94 88.94 23.98 86.62 22.26 82.27 19.16

### 5 Experiments

#### 5.1 Static Scene Reconstruction

Impact of the spatial autocorrelation. First, we conduct a toy experiment to verify our intuition that the absence of the spatial bias hampers the reconstruction quality from sparse views. We utilize four sequences from the Owlii dynamic dataset [80]1 and select the first frame from each. Each scene comprises nine training views and one validation view, on which we report training and test metrics. We compare 3DGS [29] and SplatFields (both initialized from visual hulls) and observe (Tab. 1) that 3DGS demonstrates extremely high fitting quality on the training views while poorly generalizing to the novel views. In contrast, SplatFields demonstrates a slightly lower training quality while achieving higher reconstruction quality on novel views. This observation is followed by computing the Moran’s I metric [48] which shows the amount of spatial correlation between nearby splat features; as hypothesized, the lower test-time error is in line with the increased level of spatial consistency of all groups of splat features. Fig. 1 presents both qualitative and quantitative results from the same experiment conducted on a scene from the Blender dataset [47].

Static reconstruction from sparse views. We benchmark SplatFields on Blender [47] under 6 and 12 input views (see Sup. Mat. for more extensive benchmarking). The main goal of this section is to showcase the efficiency of the utilized spatial regularization for 3DGS methods. We, therefore, focus on comparison against the recent splat-based techniques [16, 22, 29, 91] and SparseNeRF [75], leaving the comprehensive comparison against a broader range of NeRF-based methods for more challenging dynamic scenarios considered in Sec. 5.2.

Extensive quantitative results presented in Tab. 2 demonstrate that SplatFields consistently outperforms the respective baselines across varying numbers of input views. The achieved improvement is also verified by visually sharper reconstructions outlined in Fig. 3. More importantly, we observe that the relative gap in performance between our method and the baselines is increasing as the input views become more scarce, which confirms our intuition of the spatial bias as a powerful regularizer in such scenarios. We further validate our improvement on the real-world DTU dataset [28] for the challenging task of 3-view recon-

1All experiments presented in this publication were performed by ETH Zürich. ETH Zürich obtained the licenses for the data used in such experiments.

###### Fig. 3: Static reconstruction of Blender [47] scenes for the setup from Tab. 2

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

SuGaR [22] Mip3DGS [91] 3DGS [29] Light3DGS [16] SplatFields3D GT

12Views

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

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

6Views

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

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

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

12Views

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

6Views

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

12 Input Views

mean Toy Ficus Hotdog Chair Mic Ship Drums Materials

SparseNeRF [75] - 23.02 18.19 - 26.20 23.26 20.81 19.21 20.80 SparseNeRF wo. depth 22.92 24.00 18.84 27.52 27.11 23.35 21.84 19.17 21.50 SuGaR [22] 21.78 23.77 23.08 22.36 25.72 18.72 21.09 19.55 19.94 ScaffoldGS [41] 23.82 23.65 22.78 26.34 25.80 28.28 21.17 20.47 22.06 Mip3DGS [91] 24.86 24.65 25.62 26.53 26.25 28.40 22.52 21.98 22.94 3DGS [29] 25.29 25.14 25.92 27.51 27.10 29.02 22.79 22.10 22.71 Light3DGS [16] 25.39 25.08 27.53 27.10 27.40 28.04 23.02 22.07 22.90

- 2DGS [25] 25.62 25.50 25.62 29.24 28.52 28.07 23.08 22.19 22.75

- 3DGS w. LMoran 25.44 25.26 26.55 28.96 27.91 27.87 22.33 21.98 22.65 SplatFields3D 25.80 26.98 26.27 29.45 27.42 27.60 23.78 22.55 22.32

6 Input Views

SparseNeRF [75] - 20.86 18.03 - 22.75 22.40 19.33 16.24 19.54 SparseNeRF wo. depth 20.86 22.62 17.63 25.84 22.65 20.72 19.85 17.25 20.30

- SuGaR [22] 19.07 19.89 20.61 20.80 21.92 18.26 17.72 16.86 16.53 ScaffoldGS [41] 19.65 18.21 20.72 19.48 22.20 24.31 16.47 17.21 18.62 Mip3DGS [91] 20.04 19.39 21.81 19.70 21.72 24.44 17.02 17.72 18.52 3DGS [29] 20.62 19.80 22.25 21.16 22.75 25.21 17.58 17.77 18.48 Light3DGS [16] 20.76 20.25 23.12 20.66 22.69 24.89 17.83 18.02 18.63

- 2DGS [25] 20.74 19.38 21.93 23.85 23.26 24.48 16.92 17.91 18.17

- 3DGS w. LMoran 21.03 20.34 23.05 23.92 22.50 24.64 17.20 18.14 18.48

- SplatFields3D 22.26 22.41 22.26 26.19 25.03 24.84 19.33 18.97 19.05

- Table 2: Sparse static scene reconstruction of Blender [47] scenes. Reported numbers indicate PSNR metric on the novel views (“-” denotes failed runs). Colors denote the 1st , 2nd , and 3rd best-performing model. See Sec. 5.1 for discussion

- Table 4: Monocular reconstruction of dynamic sequences from the NeRF-DS dataset [82] with recent state-of-the-art methods. The forward slash in FPS indicates the rendering speed without the neural network inference when the rendering primitives are extracted and stored for each frame vs. with the neural network inference

Resources mean↑ LPIPS↓ (×102)

FPS ↑ t ↓ PSNR SSIM mean Sieve Plate Bell Press Cup As Basin

- 3D-GS [29] 120+ 15 m. 20.29 78.16 29.20 22.47 40.93 25.03 29.04 25.48 29.94 31.53 TiNeuVox [17] < 1 30 m. 21.61 82.34 27.66 31.76 33.17 25.68 30.01 36.43 39.67 26.90

- 4DGaussians [78] 120+/50 30 m. 23.68 83.22 21.06 16.39 23.80 21.84 21.68 19.06 22.06 22.57 HyperNeRF [53] < 1 1 d. 23.45 84.88 19.90 16.45 29.40 20.52 19.59 16.50 17.77 19.11 Deformable3DGS [85] 120+/30 1 h. 23.54 84.05 19.79 15.30 25.04 15.93 29.89 15.38 17.88 19.10 NeRF-DS [82] < 1 1 d. 23.60 84.94 18.16 14.72 19.96 18.67 20.47 17.37 17.41 18.55

- SplatFields4D 120+/30 1 h. 23.84 85.17 17.86 14.72 22.43 16.10 19.26 15.67 17.71 19.11

4DGaussians [78] Deformable3DGS [85] SplatFields4D Ground Truth

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Press

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Plate

- Fig. 4: Monocular reconstruction of sequences from [82] on the setup from Tab. 4.

struction (Fig. 5) and demonstrate consistent improvement over NeRF- [68] and splatting-based [25,29] baselines. See Sup. Mat. for more extensive evaluation.

Please note that all the methods, including ours, demonstrate real-time rendering performance with high interactive rates (120+FPS) during test time since the generator gθ is discarded after the training completion. We refer the reader to the supplementary for further details.

Ablation of the triplane CNN generator. We validate the impact of the proposed triplane CNN generator on the performance of the SplatFields model in Tab. 3. Here, the basic pipeline implies using only the set of MLPs to directly predict the splat rendering features (opacity, scale, etc.) and point displacements from the initial splat locations, without conditioning on the deep features produced by the triplane CNN. Results indicate that utilizing the deep features regressed by a CNN improves the quality, with the splat L2-norm regularization term further benefiting the reconstruction. Note that the regularization has a marginal improvement on the results of our pipeline that does not utilize the CNN feature generator, demonstrating the synergy of both modeling strategies.

#### 5.2 Dynamic Scene Reconstruction

Monocular dynamic reconstruction. We further evaluate our method on seven sequences of varying lengths (ranging from 424 to 881 frames) from the

GT SplatFields3D 2DGS [25] 3DGS [29] ZeroRF [68]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

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

Mean PSNR↑ 21.07 20.70 19.40 19.10

- Fig. 5: Three-view reconstruction on DTU [28]; PSNR are averaged across all 15 scenes. See Tab. C.3 for the individualized scores.

NeRF-DS dataset [82] and compare our method against both NeRF- [17,53,82] and 3DGS-based [78,85] dynamic reconstruction baselines.

The results are reported in Tab. 4, and the qualitative comparison of the

- 3DGS-based methods is presented in Fig. 4. The results demonstrate that our method achieves competitive or superior reconstruction quality across all sequences, while also maintaining real-time rendering capabilities and facilitating accelerated training processes. However, we observed that the sequences in the dataset involve relatively small motion and large static parts. Therefore, we further analyze SplatFields on multi-view sequences with more challenging dynamics.

Multi-view dynamic reconstruction. Following [45], we use 4 sequences from Owlii [80]. We opt for the dataset as it has realistic and more complex motion compared to the commonly utilized synthetic sequences [57]. Each sequence is 100 frames long and comprises multi-view video streams, where we

- Table 5: Multi-view reconstruction of dynamic sequences from the Owlii dataset [47] under varying number of input views. The reported metric is PSNR averaged across novel views. The forward slash in FPS indicates the rendering speed without the neural network inference when the rendering primitives are extracted and stored for each frame vs. with the neural network inference.

Resources 10 Input Views

FPS↑ t↓ mean Dancer Exercise Model Basketball

4DNeRFs

DyNeRF [37,45]

<1

1 day 29.70 28.22 30.64 29.95 30.00 TNeRF [37,45] 1 day 30.39 29.12 31.00 30.71 30.71 DNeRF [45,57] 1.5 day 30.25 29.39 30.63 30.63 30.35 Nerfies [45,52] 1.5 day 30.70 29.57 31.08 30.53 31.60 HyperNeRF [45,53] 2 days 30.36 30.09 30.39 30.88 30.08

Splatting

4D-GS [84] 120+ 10h 28.05 28.11 29.09 29.06 25.94 Deformable3DGS [85] 120+/30 8h 27.76 27.86 28.78 26.47 27.95 4DGaussians [78] 120+/50 2h 29.80 28.46 30.21 30.69 29.82

SplatFields4D (30k it) 120+/30 2h 30.88 30.46 30.78 31.14 31.15 SplatFields4D (40k it) 120+/30 3h 30.96 30.57 30.85 31.20 31.24 SplatFields4D (100k it) 120+/30 7h 31.12 30.79 30.99 31.33 31.39 SplatFields4D (200k it) 120+/30 14h 31.32 31.05 31.16 31.50 31.58

8 Input Views

Splatting

4D-GS [84] 120+ 10h 26.20 26.99 26.34 27.41 24.07 Deformable3DGS [85] 120+/30 8h 26.06 26.77 26.24 25.61 25.62 4DGaussians [78] 120+/50 2h 28.16 27.34 28.10 29.60 27.62

SplatFields4D (30k it) 120+/30 2h 29.46 29.38 28.84 29.80 29.83 SplatFields4D (40k it) 120+/30 3h 29.53 29.46 28.90 29.85 29.90 SplatFields4D (100k it) 120+/30 7h 29.66 29.66 29.01 29.96 30.02 SplatFields4D (200k it) 120+/30 14h 29.84 29.92 29.16 30.10 30.18

6 Input Views

Splatting

4D-GS [84] 120+ 10h 21.42 22.89 20.80 21.60 20.40 Deformable3DGS [85] 120+/30 8h 24.46 25.37 24.31 24.12 24.02 4DGaussians [78] 120+/50 2h 26.52 26.13 26.27 27.34 26.36

SplatFields4D (30k it) 120+/30 2h 28.04 28.36 27.31 28.44 28.07 SplatFields4D (40k it) 120+/30 3h 28.10 28.43 27.35 28.48 28.13 SplatFields4D (100k it) 120+/30 7h 28.22 28.61 27.44 28.56 28.25 SplatFields4D (200k it) 120+/30 14h 28.36 28.84 27.54 28.67 28.39

4 Input Views

Splatting

4D-GS [84] 120+ 10h 17.40 17.70 16.86 18.35 16.71 Deformable3DGS [85] 120+/30 8h 20.04 21.42 19.56 19.71 19.45 4DGaussians [78] 120+/50 2h 21.31 21.49 21.05 21.90 20.80

SplatFields4D (30k it) 120+/30 2h 21.88 22.60 20.73 21.83 22.34 SplatFields4D (40k it) 120+/30 3h 21.89 22.63 20.74 21.83 22.35 SplatFields4D (100k it) 120+/30 7h 21.92 22.73 20.75 21.82 22.36 SplatFields4D (200k it) 120+/30 14h 21.95 22.83 20.76 21.83 22.39

- Table 6: Flow model ablation study. Multi-view reconstruction task from Tab. 5. See Sec. 5.2 for discussion. Symbol “-” denotes failed runs

10 Views 8 Views 6 Views 4 Views

SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑ SSIM↑ PSNR↑

DCT [74] 95.10 28.72 94.53 27.34 94.00 26.42 90.94 21.68 DCT+ResFields [45] 96.56 30.99 96.05 29.66 95.46 28.22 91.47 22.12 offset [78] 95.20 28.91 94.74 27.83 94.15 26.73 90.85 21.63 offset+ResFields [45] 96.75 31.24 96.28 29.83 95.68 28.38 91.41 21.62 SE3 [85] 95.33 29.05 94.78 27.95 94.39 27.09 91.13 22.09 SE3+ResFields [45] 96.81 31.32 96.28 29.84 95.69 28.36 91.60 21.95 scaled SE3 [93] 95.05 28.78 92.42 23.88 - - - scaled SE3+ResFields [45] 96.74 31.30 96.08 29.83 95.63 28.64 92.54 23.55

vary the number of input views from 4 to 10 to study the robustness of our and baseline models. For evaluation, we use 100 images of a rotating camera around the performer, where each image comes from a different time step.

We provide a comprehensive comparison of the SplatFields4D against the respective baselines in Tab. 5. Dynamic NeRF methods [37,52,53,57] improved with ResFields [45] generally require significantly longer training times, while recent dynamic 3DGS-based methods [78, 84, 85] showcase suboptimal modeling capabilities and performance in the considered sparse scenario. Our method demonstrates a clear metric improvement over the baselines while retaining the key properties of 3DGS, such as interactive rendering speed and compatibility with the existing visualization pipelines. We specifically emphasize the difference in performance between our and the recent closely related method [84], where our combination of the triplane features and MLP-based dynamics modeling proves to be more robust compared to the HexPlane-based [6] approach in the case of rapid motion and sparse camera setups.

Splat flow model ablation. We compare the flow field fp modeling deformations via DCT basis [39], translation vectors [57], SE(3) transformation [52], and via scaled SE(3) transformation [93]. We also ablate the impact of implementing neural fields via ResFields [45] to further increase the modeling capacity of our pipeline without affecting its training speed. The results (Tab. 6) suggest that modeling flow as SE(3) achieves slightly better quality when the number of views is large. We further observe that implementing neural fields via ResFields further boosts the reconstruction quality across all setups.

### 6 Conclusion

In this work, we proposed an effective optimization strategy that introduces spatial and connectivity biases into the 3D Gaussian splats during optimization process by modeling them through a continuous neural field. We demonstrated that our optimization strategy considerably enhances reconstruction quality in the sparse setups, without the need for any external, data-driven priors. Furthermore, we introduced an effective extension of our method for reconstructing dynamic sequences and demonstrated state-of-the-art results under sparse views.

Limitations and future work. The performance of our method noticeably diminishes in extremely sparse and highly dynamic scenarios, such as those involving rapid motion with as few as four views, exemplified by the Owlii dataset [80]. This performance is inferior when compared to the best-performing NeRF-based methods in similar sparse configurations [45]. Therefore, further exploration is required to narrow the performance gap between 3DGS- and NeRFbased methods in sparse settings. Future work should also consider incorporating learning-based priors [10,27,83,89] as promising directions for advancement.

## SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction –Supplementary Material–

Overview. We provide additional details related to training (Sec. A), implementation (Sec. B), and reported results (Sec. C). We refer the reader to the supplementary video for more qualitative results.

### A Training

Optimization. Given a set of calibrated multi-view input images and an initial collection of K Gaussian splats G = {Gk}Kk=1, differential rasterizer R based on Gaussian Splatting (Sec. 3) propagates image changes to the scene parameters G. This feedback loop is used to optimize the scene parameters by imposing the photometric loss between the rendered I and the input image I∗:

L(R(G),I∗), (A.1)

arg min

G

where the initial collection of splats is initialized either randomly [34], by SfM [67], or by the visual hull [19].

Loss definition. We follow the training scheme used by 3DGS (Sec. 3) and optimize the rendering objective (Eq. A.1) via the Adam optimizer [31]. We also employ the mask loss between rendered and ground-truth masks for the object-level scenes:

L = (1 − λ1)L1 + λ1LD-SSIM + λ2LMASK + λ3Lnorm, (A.2)

where LMASK is the L1 loss between the rendered opacity and the ground truth mask akin to [59] and Lnorm is the splat norm regularization term from Sec. 4.

Hyperparameter λ1 is empirically set to 0.2, λ2 is set to 0.1 for object-level scenes [47,80] and to 0 for unbounded scenes, while λ3 is set to 0.01 for static reconstruction and to 0 for all of the other experiments. We further employ the exponential learning rate decay that starts from 8 × 10−4 until it reaches 1.6 × 10−6 at 40k iterations.

### B Implementation Details

#### B.1 Spatial Autocorrelation

To quantify the spatial similarity of nearby features, we measure local spatial autocorrelation via Moran’s I [48] between the features of splats in their local

neighborhoods. Specifically, for each splat Gk and its attribute (color, opacity, and covariance) we query N nearest neighbors [Xi]Ni=1 with associated locations loc(Xi) ∈ R3 and measure Moran’s I (Eq. B.3) of its attributes attr(Xi) ∈ R:

[I(X)], (B.3) where

I = E

X∈G

N i=1

N j=1 Wij(X)attr(Xi)attr(Xj)

N

, (B.4)

I(X) =

N i=1

N j=1 Wij(X)

N i=1 attr(Xi)2

Wij(X) = ∥loc(Xi) − loc(Xj)∥−2 1 if i ̸= j , 0 otherwise.

(B.5)

For attributes with more than one feature dimension (e.g. color and covariance matrix), we average Moran’s I across all feature dimensions. In all of the experiments, we set N = 5.

Moran’s Loss LMoran (in Tab. 2, C.1, C.2) enhances 3DGS as a straightforward baseline that incorporates the spatial bias by enforcing a higher Moran’s I score and is implemented as the negative autocorrelation score:

LMoran = λMoran 1 − E

X∈G

where λMoran is empirically set to 0.01.

[I(X)] , (B.6)

#### B.2 SplatFields

In the following, we describe the network architectures.

CNN Generator gθ consists of three CNN decoders to produce three axisaligned feature planes F. Each decoder, takes as input a 20 × 20-resolution noise ϵ ∈ R20×20×8 with 8 channels to produce the 160 × 160-resolution feature plane with 16 channels (R160×160×16) through up-sampling blocks with residual connections. First, the noise is expanded to 32 channels via an up-sampling CNN layer, which is then processed by a single attention layer and propagated through a ResNet block with two CNN layers to output an intermediate feature (20 × 20 × 32). This feature is then propagated through four up-sampling blocks until the feature resolution of 160 × 160 × 32 which is then down-scaled to 16 channels via a single CNN layer to form the final tri-plane representation F ∈ R3×160×160×16. Each up-sampling block consists of two CNN ResNet blocks (each with two CNN layers) and one up-sampling CNN layer. Then the splat center pk is projected onto each axis-aligned feature plane to obtain feature vectors via bi-linear interpolation. These features are then concatenated along the feature dimension and propagated through a tiny 2-layer MLP with 48 neurons to produce the point feature fk ∈ R48.

Deform MLP fΘ takes as input the splat center pk and the feature fk. The splat location is first positionally encoded [47] with 4 levels and propagated

through an 8-layer MLP with 128 neurons that deforms the splat center pˆk by predicting its residual akin to [56,57].

takes as input the deformed query point (positionally encoded with 4 levels) along with fk and propagates them through a 6-layer MLP with 128 neurons, where the last layer takes as input the viewing direction akin to NeRF [47].

Color Field fΘ

c

Fields take the same input as the color MLP and are implemented as 5-layer 64-neuron MLPs. The output of the opacity MLP is activated by the sigmoid function.

Scale fΘ

and Opacity fΘ

s

α

is implemented as a 4-layer MLP that takes the same input as the color MLP and predicts a four-dimensional vector that is normalized to produce the quaternion representation.

Rotation Field fΘ

O

is utilized only for the 4D reconstruction. It takes as input the deformed splat center pˆk (positionally encoded with 4 levels) and the feature vector fk and propagates them through an architecture similar to Deform MLP to model the forward flow. In the paper, we consider different types of modeling the flow: DCT [39], SE(3) [52], scaled SE(3) [93], and offsets [56,57]. See Sec. 5.2 for further details.

Flow Field fΘ

p

All of the MLP fields take time (positionally encoded with 4 levels) as an additional input and are implemented as ResField MLPs [45]. We empirically set the ResFields’ rank to 40 for the multi-view dynamic reconstruction on Owlii [80] and to 0 for the monocular reconstruction [82] as the scenes are semi-static.

### C Experiment Details

Static reconstruction on Blender (Sec. 5.1). We compare SplatFields with SparseNeRF [75] and with recent 3DGS methods: SuGaR [22], Mip3DGS [91], 3DGS [29], 2DGS [25], and Light3DGS [16] on Blender [47]. Mip3DGS [91], 3DGS [29], and 2DGS [25] are run for 40k iterations like SplatFields, while Light3DGS [16], SuGaR [22], and SparseNeRF [75] are run with their default configurations as they have a particular training scheme. All of the methods are initialized from the randomly sampled points inside the visual hull of the objects and are further supervised with the mask loss implemented as the L1 distance between the ground truth and the rendered opacity akin to [59].

We further provide extended comparisons of Tab. 2 in Tab. C.1-C.2 for varying number of views ranging from 4 to 12. Consistently with the main paper, SplatFields demonstrates superior metric reconstruction quality over the baseline methods across varying number of input views.

Static reconstruction on DTU (Sec. 5.1). We compare with NeRF (VolRecon [63], ZeroRF [68]) and splatting (3DGS [29], 2DGS [25]) methods on DTU [28] on the task of 3-view reconstruction (Tab. C.3). All of the baselines are run with the default configurations, with the difference that the splattingbased baselines adopt the mask loss for fair comparisons.

Monocular dynamic reconstruction (Sec. 5.2). Our method adopts annealing smooth training [85] and is trained for 30k iterations after being initial-

- Table C.1: Sparse static scene reconstruction of Blender [47] scenes. Reported numbers indicate PSNR metric on the novel views (“-” denotes failed runs). Colors denote the 1st , 2nd , and 3rd best-performing model. See Sec. 5.1 for discussion

12 Input Views

mean Toy Ficus Hotdog Chair Mic Ship Drums Materials

SparseNeRF [75] - 23.02 18.19 - 26.20 23.26 20.81 19.21 20.80 SparseNeRF wo. depth 22.92 24.00 18.84 27.52 27.11 23.35 21.84 19.17 21.50 SuGaR [22] 21.78 23.77 23.08 22.36 25.72 18.72 21.09 19.55 19.94 ScaffoldGS [41] 23.82 23.65 22.78 26.34 25.80 28.28 21.17 20.47 22.06 Mip3DGS [91] 24.86 24.65 25.62 26.53 26.25 28.40 22.52 21.98 22.94 3DGS [29] 25.29 25.14 25.92 27.51 27.10 29.02 22.79 22.10 22.71 Light3DGS [16] 25.39 25.08 27.53 27.10 27.40 28.04 23.02 22.07 22.90

- 2DGS [25] 25.62 25.50 25.62 29.24 28.52 28.07 23.08 22.19 22.75

- 3DGS w. LMoran 25.44 25.26 26.55 28.96 27.91 27.87 22.33 21.98 22.65 SplatFields3D 25.80 26.98 26.27 29.45 27.42 27.60 23.78 22.55 22.32

10 Input Views

SparseNeRF [75] - 22.64 18.27 - 25.30 23.27 20.29 18.61 19.72 SparseNeRF wo. depth 22.58 23.89 18.75 27.56 26.42 23.23 21.68 18.20 20.87 SuGaR [22] 21.10 22.78 22.42 23.60 24.25 17.93 20.35 19.11 18.40 ScaffoldGS [41] 22.63 21.98 22.68 24.37 24.15 27.76 20.39 19.64 20.08 Mip3DGS [91] 23.65 23.49 24.97 25.27 24.49 27.69 21.38 21.23 20.66 3DGS [29] 24.11 23.79 25.54 26.16 25.28 28.39 21.87 21.34 20.51 Light3DGS [16] 24.21 23.94 26.95 25.62 25.91 27.45 21.82 21.38 20.60

- 2DGS [25] 24.42 24.06 25.17 27.92 26.96 27.53 21.83 21.58 20.27

- 3DGS w.LMoran 24.21 23.91 26.09 27.65 25.86 27.07 21.38 21.26 20.46 SplatFields3D 24.94 26.51 25.59 28.29 25.92 27.36 23.12 21.86 20.88

8 Input Views

SparseNeRF [75] - 22.33 17.97 - 23.81 23.01 19.85 17.85 20.02 SparseNeRF wo. depth 22.20 24.06 18.42 27.09 25.12 23.04 21.23 17.94 20.74

- SuGaR [22] 20.62 21.91 22.33 23.01 23.30 18.60 19.59 18.66 17.55 ScaffoldGS [41] 21.53 20.95 21.35 23.77 22.77 26.40 18.88 18.96 19.17 Mip3DGS [91] 22.37 22.05 23.23 24.24 23.57 26.32 19.91 20.10 19.55 3DGS [29] 22.93 22.55 23.69 25.57 24.43 27.37 19.98 20.33 19.49 Light3DGS [16] 22.98 22.67 24.98 24.79 24.40 26.59 20.60 20.41 19.41

- 2DGS [25] 23.04 22.19 23.63 26.76 25.46 26.24 20.16 20.60 19.25

- 3DGS w. LMoran 23.19 22.79 24.56 26.57 25.14 26.97 19.79 20.41 19.32 SplatFields3D 23.98 24.71 23.97 27.87 25.64 26.49 22.15 21.12 19.85

6 Input Views

SparseNeRF [75] - 20.86 18.03 - 22.75 22.40 19.33 16.24 19.54 SparseNeRF wo. depth 20.86 22.62 17.63 25.84 22.65 20.72 19.85 17.25 20.30 SuGaR [22] 19.07 19.89 20.61 20.80 21.92 18.26 17.72 16.86 16.53 ScaffoldGS [41] 19.65 18.21 20.72 19.48 22.20 24.31 16.47 17.21 18.62 Mip3DGS [91] 20.04 19.39 21.81 19.70 21.72 24.44 17.02 17.72 18.52 3DGS [29] 20.62 19.80 22.25 21.16 22.75 25.21 17.58 17.77 18.48 Light3DGS [16] 20.76 20.25 23.12 20.66 22.69 24.89 17.83 18.02 18.63

- 2DGS [25] 20.74 19.38 21.93 23.85 23.26 24.48 16.92 17.91 18.17

- 3DGS w. LMoran 21.03 20.34 23.05 23.92 22.50 24.64 17.20 18.14 18.48 SplatFields3D 22.26 22.41 22.26 26.19 25.03 24.84 19.33 18.97 19.05

4 Input Views

SparseNeRF [75] - 20.94 17.48 23.81 21.41 21.52 - 15.37 17.03 SparseNeRF wo. depth 17.87 19.31 17.05 23.54 20.26 11.56 17.86 13.64 19.77 SuGaR [22] 16.94 16.96 19.30 19.36 19.07 17.47 15.22 14.73 13.38 ScaffoldGS [41] 16.86 15.40 19.58 17.31 18.40 20.54 14.70 15.27 13.69 Mip3DGS [91] 16.94 16.23 19.60 16.98 18.38 20.56 14.64 14.92 14.21 3DGS [29] 17.37 16.44 19.72 18.65 18.72 20.75 15.43 15.08 14.15 Light3DGS [16] 17.70 16.94 20.35 18.56 18.96 21.53 15.67 15.44 14.19

- 2DGS [25] 17.58 16.32 19.69 20.67 19.39 21.17 14.45 14.84 14.14

- 3DGS w. LMoran 18.13 17.06 20.53 22.10 18.25 22.06 15.18 15.32 14.53 SplatFields3D 19.16 18.89 20.19 24.31 19.31 21.73 16.83 16.35 15.69

###### Table C.2: Sparse static scene reconstruction. Synthetic Blender [47] dataset, reported numbers indicate SSIM metric on the novel views. See Sec. 5.1 for discussion

12 Input Views

mean Toy Ficus Hotdog Chair Mic Ship Drums Materials

SparseNeRF [75] - 86.07 84.57 - 90.45 92.23 76.28 83.65 85.40 SparseNeRF wo. depth 87.54 88.64 85.10 93.96 91.77 92.55 77.69 83.69 86.94 SuGaR [22] 85.60 86.24 89.09 90.29 90.94 85.86 76.81 82.77 82.76 ScaffoldGS [41] 87.47 86.22 90.65 92.06 90.75 96.21 73.02 85.75 85.10 Mip3DGS [91] 89.78 87.81 93.87 93.15 92.46 96.90 76.42 89.67 87.96 3DGS [29] 90.01 88.57 94.11 93.45 93.48 96.98 76.11 89.81 87.55 Light3DGS [16] 90.30 88.51 95.31 93.45 93.52 96.64 76.57 89.98 88.42

- 2DGS [25] 91.09 90.18 94.20 94.85 94.93 96.77 79.09 90.41 88.27

- 3DGS w. LMoran 90.45 89.31 94.62 94.40 94.07 96.52 76.71 89.97 88.01 SplatFields3D 91.18 91.06 94.36 95.55 92.42 96.17 81.03 90.90 87.98

10 Input Views

SparseNeRF [75] - 85.65 84.30 - 89.57 92.27 75.51 82.52 83.82 SparseNeRF wo. depth 86.95 88.45 84.65 94.06 91.26 92.39 77.25 82.12 85.42 SuGaR [22] 83.83 84.27 88.55 90.52 88.77 84.39 74.27 80.78 79.10 ScaffoldGS [41] 85.45 83.27 90.14 89.66 88.56 95.77 70.66 83.46 82.10 Mip3DGS [91] 88.00 85.68 93.15 91.76 90.34 96.44 73.98 88.22 84.41 3DGS [29] 88.27 86.30 93.65 92.19 91.23 96.49 73.89 88.48 83.93 Light3DGS [16] 88.76 86.66 94.82 92.41 91.70 96.27 74.37 88.75 85.12

- 2DGS [25] 89.51 88.42 93.61 93.79 93.26 96.37 76.52 89.38 84.72

- 3DGS w. LMoran 88.94 87.53 94.24 93.32 91.88 95.98 75.10 88.74 84.69 SplatFields3D 90.32 91.34 93.70 95.09 91.14 95.95 80.25 89.85 85.26

8 Input Views

SparseNeRF [75] - 84.85 83.98 - 88.39 92.06 74.32 81.44 83.66 SparseNeRF wo. depth 86.30 88.27 83.97 93.65 90.23 92.15 76.01 81.38 84.78 SuGaR [22] 82.74 82.04 87.95 89.41 87.60 85.08 72.80 79.84 77.23 ScaffoldGS [41] 83.62 80.54 88.25 89.04 86.23 95.01 68.49 81.49 79.91 Mip3DGS [91] 86.24 82.93 91.03 90.84 89.21 95.77 71.86 86.15 82.14 3DGS [29] 86.63 84.05 91.56 91.63 90.49 95.99 70.96 86.62 81.76 Light3DGS [16] 87.11 84.44 93.01 91.47 90.17 95.76 72.16 86.97 82.92

- 2DGS [25] 87.72 84.80 91.74 93.13 91.79 95.72 74.06 87.89 82.65

- 3DGS w. LMoran 87.35 85.15 92.57 92.50 91.04 95.88 72.03 87.26 82.36

- SplatFields3D 88.94 88.04 91.69 94.68 90.91 95.48 78.76 88.50 83.46

6 Input Views

SparseNeRF [75] - 83.25 83.70 - 87.76 91.53 72.96 78.58 83.47 SparseNeRF wo. depth 84.42 85.68 82.33 92.51 87.47 90.72 72.50 79.58 84.59 SuGaR [22] 79.85 77.67 85.51 86.44 84.98 84.24 68.93 76.35 74.68 ScaffoldGS [41] 80.34 74.13 87.01 82.79 84.99 93.10 62.49 78.61 79.59 Mip3DGS [91] 83.09 77.68 89.34 86.81 86.46 94.58 66.45 82.08 81.34 3DGS [29] 83.56 78.58 89.79 87.81 87.35 94.81 66.71 82.45 80.94 Light3DGS [16] 84.34 79.64 91.08 88.67 87.49 94.76 67.63 83.09 82.33

- 2DGS [25] 84.43 78.54 89.71 90.36 88.16 94.59 68.63 83.87 81.61

- 3DGS w. LMoran 84.54 80.20 90.84 90.15 87.76 94.50 67.49 83.47 81.89

- SplatFields3D 86.62 84.05 89.56 93.62 89.53 94.50 74.14 85.03 82.55

4 Input Views

SparseNeRF [75] - 83.38 82.98 90.95 85.14 90.49 - 76.46 79.48 SparseNeRF wo. depth 78.66 78.80 80.86 90.57 82.26 72.23 69.29 71.97 83.28 SuGaR [22] 75.61 72.07 83.08 83.68 80.66 82.48 63.46 70.73 68.69 ScaffoldGS [41] 74.99 68.04 84.68 76.92 77.58 90.24 59.60 71.20 71.68 Mip3DGS [91] 77.67 71.47 85.73 82.68 81.00 91.42 61.17 74.27 73.62 3DGS [29] 78.12 72.17 85.97 83.78 81.49 91.55 62.05 74.80 73.18 Light3DGS [16] 79.38 73.58 86.93 85.98 81.91 92.25 63.27 75.88 75.22

- 2DGS [25] 79.26 72.84 86.04 86.62 82.54 92.04 63.57 76.25 74.14

- 3DGS w. LMoran 79.65 73.79 87.05 87.79 82.01 92.39 63.31 75.95 74.88

- SplatFields3D 82.26 78.23 86.17 92.10 83.85 91.92 70.40 78.67 76.77

- Table C.3: Static three-view reconstruction on the DTU dataset [28]. SplatFields demonstrates more accurate reconstructions compared to the NeRF- (VolRecon [63], ZeroRF [68]) and splatting-based (3DGS [29], 2DGS [25]) baselines; the displayed metric is PSNR↑

| |mean PSNR↑<br><br>|Scene ID Number (PSNR↑) 105 106 110 114 118 122 24 37 40 55 63 65 69 83 97|
|---|---|---|
|VolRecon ZeroRF 3DGS 2DGS<br><br>|11.42 19.10 19.40 20.70<br><br>|9.03 15.19 16.45 11.66 17.96 18.29 7.86 6.30 7.86 12.90 6.54 10.06 14.84 7.77 8.64 21.36 14.30 20.96 18.86 19.24 23.45 15.78 15.23 16.06 21.02 23.62 18.31 15.05 24.17 19.13<br><br>20.07 17.06 17.04 20.56 18.25 20.23 18.76 19.82 18.29 21.03 22.63 20.02 15.86 21.64 19.76<br><br>21.25 19.23 19.17 19.90 19.75 22.04 19.71 20.22 19.56 21.95 23.16 22.37 17.64 23.13 21.47<br><br><br>|

SplatFields 21.07 21.93 19.11 19.77 22.03 21.35 24.49 18.43 19.82 19.67 21.45 23.79 22.64 17.54 23.52 20.52

ized from static 3DGS ran for 3k iterations akin to [85]. We run recent dynamic

- 3DGS methods [78, 85] with their default configurations, while results for the NeRF-based methods are adopted from the previous work [82,85].

We further provide additional metrics SSIM and PSNR in Tab. C.4. Note that SSIM and PSNR are less reliable metrics due to noisy camera calibrations.

Multi-view dynamic reconstruction (Sec. 5.2). Dynamic NeRF-based methods (DyNeRF [37], TNeRF [37], DNeRF [57], HyperNeRF [53]) are trained with SDF parametrization as they are better suited for sparse view reconstruction. We use the implementations with ResField MLPs [45] (256 neurons) and train them for 400k iterations, following the training scheme from [45].

For dynamic Gaussian splatting methods (4D-GS [84], Deformable3DGS [85],

- 4DGaussians [78]), we use their default implementations and adopt additional mask loss with the weight of 0.1. All of these methods, including ours, are trained with a batch size of 5. 4D-GS is trained with default 30k iterations. Deformable3DGS is trained until the full convergence of 200k iterations. 4DGaussians is trained for 30k iterations, we noticed that longer training leads to overfitting and the loss becomes an invalid number.

Additional SSIM metric is reported in Tab. C.5. Akin to the main paper, SplatFields demonstrates consistently better reconstruction quality across all scenes and varying number of input views.

Compute, memory overhead, and inference time. Compared to the original 3DGS, our method requires longer training to converge (∼10 min. for 3DGS vs. ∼70 min. for ours on the Toy scene) and consumes a greater amount of GPU memory (∼5GB for 3DGS vs. about ∼8GB for ours). However, after training, the neural components can be discarded, leaving the inference speed and memory usage equivalent to that of 3DGS. In dynamic setups, the training times of our method are comparable to other dynamic 3DGS methods that also employ neural networks. Our neural network architecture comprises ∼1M parameters for the static case. All the run-times reported in the paper are calculated on an NVIDIA RTX 3090.

CNNs vs. MLPs on extremely sparse view setups. CNN module enhances the capacity of the SplatFields, which may lead to slight overfitting in extremely sparse scenarios, such as a 4-view setup. However, as additional views are incorporated and the model receives more diverse inputs, the ability of CNNs to

- Table C.4: Monocular reconstruction of dynamic sequences from the NeRF-DS dataset [82] with recent state-of-the-art methods. The forward slash in FPS indicates the rendering speed with the inference of neural network vs. without when the rendering primitives are extracted and stored for each frame

Resources LPIPS↓ (×102)

FPS ↑ t ↓ mean Sieve Plate Bell Press Cup As Basin

- 3D-GS [29] 120+ 15 min 29.20 22.47 40.93 25.03 29.04 25.48 29.94 31.53 TiNeuVox [17] < 1 30 min 27.66 31.76 33.17 25.68 30.01 36.43 39.67 26.90

- 4DGaussians [78] 120+/50 30 min 21.06 16.39 23.80 21.84 21.68 19.06 22.06 22.57 HyperNeRF [53] < 1 1 day 19.90 16.45 29.40 20.52 19.59 16.50 17.77 19.11 Deformable3DGS [85] 120+/30 1 h 19.79 15.30 25.04 15.93 29.89 15.38 17.88 19.10 NeRF-DS [82] < 1 1 day 18.16 14.72 19.96 18.67 20.47 17.37 17.41 18.55

- SplatFields4D 120+/30 1 h 17.86 14.72 22.43 16.10 19.26 15.67 17.71 19.11

PSNR↑

FPS ↑ t ↓ mean Sieve Plate Bell Press Cup As Basin

- 3D-GS [29] 120+ 15 min 20.29 23.16 16.14 21.01 22.89 21.71 22.69 18.42 TiNeuVox [17] < 1 30 min 21.61 21.49 20.58 23.08 24.47 19.71 21.26 20.66

- 4DGaussians 120+/30 30 min 23.68 26.77 20.51 24.25 25.55 23.69 25.50 19.47 HyperNeRF [53] < 1 1 day 23.45 25.43 18.93 23.06 26.15 24.59 25.58 20.41 Deformable3DGS [85] 120+/30 1 h 23.54 25.16 19.97 25.02 24.18 24.64 26.26 19.57 NeRF-DS [82] < 1 1 day 23.60 25.78 20.54 23.19 25.72 24.91 25.13 19.96

- SplatFields4D 120+/30 1 h 23.84 25.35 20.36 25.51 25.43 24.29 26.21 19.71

SSIM↑

FPS ↑ t ↓ mean Sieve Plate Bell Press Cup As Basin

- 3D-GS [29] 120+ 15 min 78.16 82.03 69.70 78.85 81.63 83.04 80.17 71.70 TiNeuVox [17] < 1 30 min 82.34 82.65 80.27 82.42 86.13 81.09 82.89 81.45

- 4DGaussians 120+/30 30 min 83.22 87.18 79.70 81.14 85.73 86.46 85.73 76.62 HyperNeRF [53] < 1 1 day 84.88 87.98 77.09 80.97 88.97 87.70 89.49 81.99 Deformable3DGS [85] 120+/30 1 h 84.05 87.58 79.14 84.52 81.22 88.71 88.49 78.69 NeRF-DS [82] < 1 1 day 84.94 89.00 80.42 82.12 86.18 87.41 87.78 81.66

- SplatFields4D 120+/30 1 h 85.17 87.78 80.26 84.74 86.64 88.73 88.59 79.44

capture structural patterns become increasingly beneficial; this is demonstrated by the improved performance in denser view setups. Please also note that the CNN-based SplatFields model is still better than the vanilla 3DGS method in the 4-view setup (Tab. C.1-C.2).

Spatial autocorrelation: sparse vs. dense view setup. A simple experiment under different view setups on Toy [47] demonstrates (Tab. C.6) a tendency that overfitting (high ∆PSNR) corresponds to lower autocorrelation, especially for RGB.

- Table C.5: Multi-view reconstruction of dynamic sequences from the Owlii dataset [47] under varying number of input views. The reported metric is SSIM↑ averaged across novel views. See Sec. 5.2 for discussion

10 Input Views

mean Dancer Exercise Model Basketball

4D-GS [84] 95.34 95.31 95.96 94.92 95.16 Deformable3DGS [85] 93.80 94.10 95.09 91.58 94.43 4DGaussians [78] 95.91 95.19 96.47 95.71 96.28

SplatFields4D (30k it) 96.52 96.41 96.72 95.99 96.98 SplatFields4D (40k it) 96.57 96.47 96.76 96.04 97.02 SplatFields4D (100k it) 96.67 96.59 96.83 96.16 97.11 SplatFields4D (200k it) 96.81 96.76 96.92 96.32 97.23

8 Input Views

mean Dancer Exercise Model Basketball

4D-GS [84] 93.71 94.19 93.94 93.29 93.40 Deformable3DGS [85] 92.37 93.24 93.29 90.33 92.62 4DGaussians [78] 95.00 94.39 95.45 94.92 95.26

SplatFields4D (30k it) 95.99 95.97 96.05 95.44 96.52 SplatFields4D (40k it) 96.04 96.02 96.08 95.49 96.56 SplatFields4D (100k it) 96.15 96.15 96.16 95.62 96.65 SplatFields4D (200k it) 96.28 96.31 96.26 95.78 96.77

6 Input Views

mean Dancer Exercise Model Basketball

4D-GS [84] 87.23 89.52 87.05 85.16 87.20 Deformable3DGS [85] 90.95 91.73 91.48 89.11 91.48 4DGaussians [78] 93.87 93.58 94.45 93.05 94.40

SplatFields4D (30k it) 95.40 95.62 95.51 94.53 95.95 SplatFields4D (40k it) 95.45 95.67 95.54 94.59 95.99 SplatFields4D (100k it) 95.56 95.81 95.62 94.70 96.09 SplatFields4D (200k it) 95.69 95.99 95.71 94.85 96.21

4 Input Views

mean Dancer Exercise Model Basketball

4D-GS [84] 78.94 80.81 78.72 78.09 78.15 Deformable3DGS [85] 87.10 89.04 87.67 85.05 86.63 4DGaussians [78] 89.50 90.34 90.67 87.47 89.51

SplatFields4D (30k it) 91.46 92.61 91.99 88.98 92.28 SplatFields4D (40k it) 91.49 92.65 92.01 88.99 92.31 SplatFields4D (100k it) 91.54 92.76 92.04 89.01 92.36 SplatFields4D (200k it) 91.60 92.89 92.07 89.02 92.41

Table C.6: Moran’s I on Toy [47] for a varying number of views

#Views 5 25 50 75 100 Train PSNR 51.85 45.10 41.11 40.23 40.03 Test PSNR ↑ 18.07 30.12 33.75 35.02 35.34 ∆PSNR ↓ 33.78 14.98 7.36 5.21 4.69

Moran RGB ↑ 0.467 0.588 0.634 0.655 0.661 Moran Opacity ↑ 0.710 0.746 0.744 0.742 0.736 Moran Covariance ↑ 0.426 0.414 0.452 0.465 0.476

### References

- 1. Alexa, M., Gross, M., Pauly, M., Pfister, H., Stamminger, M., Zwicker, M.: Pointbased computer graphics. In: SIGGRAPH notes (2004) 4
- 2. Aliev, K.A., Sevastopolsky, A., Kolos, M., Ulyanov, D., Lempitsky, V.: Neural point-based graphics. In: ECCV (2020) 4
- 3. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In: CVPR (2022) 4
- 4. Botsch, M., Hornung, A., Zwicker, M., Kobbelt, L.: High-quality surface splatting on today’s gpus. In: Proceedings Eurographics/IEEE VGTC Symposium PointBased Graphics, 2005. pp. 17–141. IEEE (2005) 2, 4
- 5. Cai, H., Feng, W., Feng, X., Wang, Y., Zhang, J.: Neural surface reconstruction of dynamic scenes with monocular rgb-d camera. In: NeurIPS (2022) 4
- 6. Cao, A., Johnson, J.: Hexplane: A fast representation for dynamic scenes. CVPR

(2023) 14

- 7. Chan, E.R., Lin, C.Z., Chan, M.A., Nagano, K., Pan, B., De Mello, S., Gallo, O., Guibas, L.J., Tremblay, J., Khamis, S., et al.: Efficient geometry-aware 3d generative adversarial networks. In: CVPR (2022) 3, 4, 5, 8
- 8. Chen, A., Xu, Z., Geiger, A., Yu, J., Su, H.: Tensorf: Tensorial radiance fields. In: ECCV (2022) 2, 4
- 9. Chen, G., Wang, W.: A survey on 3d gaussian splatting. arXiv preprint arXiv:2401.03890 (2024) 2, 4
- 10. Chen, H., Gu, J., Chen, A., Tian, W., Tu, Z., Liu, L., Su, H.: Single-stage diffusion nerf: A unified approach to 3d generation and reconstruction. In: ICCV (2023) 4, 14
- 11. Das, D., Wewer, C., Yunus, R., Ilg, E., Lenssen, J.E.: Neural parametric gaussians for monocular non-rigid object reconstruction. In: CVPR (2024) 5
- 12. Deng, C., Jiang, C., Qi, C.R., Yan, X., Zhou, Y., Guibas, L., Anguelov, D., et al.: Nerdi: Single-view nerf synthesis with language-guided diffusion as general image priors. In: CVPR (2023) 4
- 13. Deng, K., Liu, A., Zhu, J.Y., Ramanan, D.: Depth-supervised nerf: Fewer views and faster training for free. In: CVPR (2022) 4
- 14. Drebin, R.A., Carpenter, L., Hanrahan, P.: Volume rendering. SIGGRAPH (1988) 4
- 15. Duckworth, D., Hedman, P., Reiser, C., Zhizhin, P., Thibert, J.F., Lučić, M., Szeliski, R., Barron, J.T.: Smerf: Streamable memory efficient radiance fields for real-time large-scene exploration. arXiv preprint arXiv:2312.07541 (2023) 4
- 16. Fan, Z., Wang, K., Wen, K., Zhu, Z., Xu, D., Wang, Z.: Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. arXiv preprint arXiv:2311.17245 (2023) 3, 4, 9, 10, 5
- 17. Fang, J., Yi, T., Wang, X., Xie, L., Zhang, X., Liu, W., Nießner, M., Tian, Q.: Fast dynamic radiance fields with time-aware neural voxels. In: SIGGRAPH Asia

(2022) 11, 12, 7

- 18. Fei, B., Xu, J., Zhang, R., Zhou, Q., Yang, W., He, Y.: 3d gaussian as a new vision era: A survey. arXiv preprint arXiv:2402.07181 (2024) 2, 4
- 19. Franco, J.S., Boyer, E.: Exact polyhedral visual hulls. In: BMVC (2003) 1
- 20. Gao, K., Gao, Y., He, H., Lu, D., Xu, L., Li, J.: Nerf: Neural radiance field in 3d vision, a comprehensive review. arXiv preprint arXiv:2210.00379 (2022) 2
- 21. Grossman, J.P., Dally, W.J.: Point sample rendering. In: Eurographics Workshop

(1998) 4

- 22. Guédon, A., Lepetit, V.: Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. In: CVPR (2024) 3, 4, 9, 10, 5
- 23. Hu, L., Zhang, H., Zhang, Y., Zhou, B., Liu, B., Zhang, S., Nie, L.: Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians. In: CVPR (2024) 5
- 24. Hu, S., Hu, T., Liu, Z.: Gauhuman: Articulated gaussian splatting from monocular human videos. In: CVPR (2024) 5
- 25. Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S.: 2d gaussian splatting for geometrically accurate radiance fields. In: SIGGRAPH (2024) 10, 11, 12, 3, 4, 5, 6
- 26. Huang, Y.H., Sun, Y.T., Yang, Z., Lyu, X., Cao, Y.P., Qi, X.: Sc-gs: Sparsecontrolled gaussian splatting for editable dynamic scenes. In: CVPR (2024) 5
- 27. Jain, A., Tancik, M., Abbeel, P.: Putting nerf on a diet: Semantically consistent few-shot view synthesis. In: ICCV (2021) 4, 14
- 28. Jensen, R., Dahl, A., Vogiatzis, G., Tola, E., Aanæs, H.: Large scale multi-view stereopsis evaluation. In: CVPR (2014) 9, 12, 3, 6
- 29. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ToG (2023) 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
- 30. Kilian, M., Mitra, N.J., Pottmann, H.: Geometric modeling in shape space. In: SIGGRAPH (2007) 4
- 31. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: ICLR

(2015) 6, 1

- 32. Kocabas, M., Chang, R., Gabriel, J., Tuzel, O., Ranjan, A.: Hugs: Human gaussian splats. In: CVPR (2024) 5
- 33. Kopanas, G., Philip, J., Leimkühler, T., Drettakis, G.: Point-based neural rendering with per-view optimization. Computer Graphics Forum (2021) 4
- 34. Lassner, C., Zollhofer, M.: Pulsar: Efficient sphere-based neural rendering. In: CVPR (2021) 4, 1
- 35. Lei, J., Wang, Y., Pavlakos, G., Liu, L., Daniilidis, K.: Gart: Gaussian articulated template models. In: CVPR (2024) 5
- 36. Li, R., Gao, H., Tancik, M., Kanazawa, A.: Nerfacc: Efficient sampling accelerates nerfs. In: ICCV (2023) 4
- 37. Li, T., Slavcheva, M., Zollhoefer, M., Green, S., Lassner, C., Kim, C., Schmidt, T., Lovegrove, S., Goesele, M., Newcombe, R., et al.: Neural 3d video synthesis from multi-view video. In: CVPR (2022) 4, 13, 14, 6
- 38. Li, Z., Chen, Z., Li, Z., Xu, Y.: Spacetime gaussian feature splatting for real-time dynamic view synthesis. In: CVPR (2024) 5
- 39. Li, Z., Wang, Q., Cole, F., Tucker, R., Snavely, N.: Dynibar: Neural dynamic imagebased rendering. In: CVPR (2023) 14, 3
- 40. Lin, Y., Dai, Z., Zhu, S., Yao, Y.: Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In: CVPR (2024) 5
- 41. Lu, T., Yu, M., Xu, L., Xiangli, Y., Wang, L., Lin, D., Dai, B.: Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In: CVPR (2024) 10, 4, 5
- 42. Lu, Z., Guo, X., Hui, L., Chen, T., Yang, M., Tang, X., Zhu, F., Dai, Y.: 3d geometry-aware deformable gaussian splatting for dynamic view synthesis. In: CVPR (2024) 5
- 43. Luiten, J., Kopanas, G., Leibe, B., Ramanan, D.: Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In: 3DV (2024) 4, 8

- 44. Mihajlovic, M., Bansal, A., Zollhoefer, M., Tang, S., Saito, S.: Keypointnerf: Generalizing image-based volumetric avatars using relative spatial encoding of keypoints. In: ECCV (2022) 4
- 45. Mihajlovic, M., Prokudin, S., Pollefeys, M., Tang, S.: ResFields: Residual neural fields for spatiotemporal signals. In: ICLR (2024) 3, 5, 7, 8, 12, 13, 14, 6
- 46. Mihajlovic, M., Weder, S., Pollefeys, M., Oswald, M.R.: Deepsurfels: Learning online appearance fusion. In: CVPR (2021) 4
- 47. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: ECCV

(2020) 1, 2, 3, 4, 7, 9, 10, 13, 5, 8

- 48. Moran, P.A.: Notes on continuous stochastic phenomena. Biometrika (1950) 2, 6, 9, 1
- 49. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ToG (2022) 2, 4
- 50. Niemeyer, M., Barron, J.T., Mildenhall, B., Sajjadi, M.S., Geiger, A., Radwan, N.: Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In: CVPR (2022) 2, 4
- 51. Pang, H., Zhu, H., Kortylewski, A., Theobalt, C., Habermann, M.: Ash: Animatable gaussian splats for efficient and photoreal human rendering. In: CVPR (2024) 5
- 52. Park, K., Sinha, U., Barron, J.T., Bouaziz, S., Goldman, D.B., Seitz, S.M., MartinBrualla, R.: Nerfies: Deformable neural radiance fields. In: ICCV (2021) 4, 13, 14, 3
- 53. Park, K., Sinha, U., Hedman, P., Barron, J.T., Bouaziz, S., Goldman, D.B., MartinBrualla, R., Seitz, S.M.: Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields. ToG (2021) 4, 11, 12, 13, 14, 6, 7
- 54. Peng, S., Yan, Y., Shuai, Q., Bao, H., Zhou, X.: Representing volumetric videos as dynamic mlp maps. In: CVPR (2023) 2
- 55. Peng, S., Niemeyer, M., Mescheder, L., Pollefeys, M., Geiger, A.: Convolutional occupancy networks. In: ECCV (2020) 3, 8
- 56. Prokudin, S., Ma, Q., Raafat, M., Valentin, J., Tang, S.: Dynamic point fields. In: ICCV (2023) 4, 3
- 57. Pumarola, A., Corona, E., Pons-Moll, G., Moreno-Noguer, F.: D-nerf: Neural radiance fields for dynamic scenes. In: CVPR (2021) 4, 12, 13, 14, 3, 6
- 58. Qian, S., Kirschstein, T., Schoneveld, L., Davoli, D., Giebenhain, S., Nießner, M.: Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians. In: CVPR

(2024) 5

- 59. Qian, Z., Wang, S., Mihajlovic, M., Geiger, A., Tang, S.: 3dgs-avatar: Animatable avatars via deformable 3d gaussian splatting. In: CVPR (2024) 5, 1, 3
- 60. Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y., Courville, A.: On the spectral bias of neural networks. In: ICML (2019) 2, 7
- 61. Reiser, C., Peng, S., Liao, Y., Geiger, A.: Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps. In: ICCV (2021) 4
- 62. Reiser, C., Szeliski, R., Verbin, D., Srinivasan, P., Mildenhall, B., Geiger, A., Barron, J., Hedman, P.: Merf: Memory-efficient radiance fields for real-time view synthesis in unbounded scenes. ToG (2023) 4
- 63. Ren, Y., Wang, F., Zhang, T., Pollefeys, M., Süsstrunk, S.: Volrecon: Volume rendering of signed ray distance functions for generalizable multi-view reconstruction. In: CVPR (2023) 3, 6
- 64. Rückert, D., Franke, L., Stamminger, M.: Adop: Approximate differentiable onepixel point rendering. ToG (2022) 4

- 65. Saito, S., Schwartz, G., Simon, T., Li, J., Nam, G.: Relightable gaussian codec avatars. In: CVPR (2024) 5
- 66. Sara Fridovich-Keil and Alex Yu, Tancik, M., Chen, Q., Recht, B., Kanazawa, A.: Plenoxels: Radiance fields without neural networks. In: CVPR (2022) 4
- 67. Schönberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: CVPR (2016) 7, 1
- 68. Shi, R., Wei, X., Wang, C., Su, H.: Zerorf: Fast sparse view 360deg reconstruction with zero pretraining. In: CVPR (2024) 3, 7, 8, 11, 12, 6
- 69. Singh, M., Fuenmayor, E., Hinchy, E.P., Qiao, Y., Murray, N., Devine, D.: Digital twin: Origin to future. Applied System Innovation (2021) 1
- 70. Tancik, M., Casser, V., Yan, X., Pradhan, S., Mildenhall, B., Srinivasan, P.P., Barron, J.T., Kretzschmar, H.: Block-nerf: Scalable large scene neural view synthesis. In: CVPR (2022) 4
- 71. Tewari, A., Thies, J., Mildenhall, B., Srinivasan, P., Tretschk, E., Yifan, W., Lassner, C., Sitzmann, V., Martin-Brualla, R., Lombardi, S., et al.: Advances in neural rendering. In: Computer Graphics Forum (2022) 3, 4, 7
- 72. Tosi, F., Zhang, Y., Gong, Z., Sandström, E., Mattoccia, S., Oswald, M.R., Poggi, M.: How nerfs and 3d gaussian splatting are reshaping slam: a survey. arXiv preprint arXiv:2402.13255 (2024) 2
- 73. Ulyanov, D., Vedaldi, A., Lempitsky, V.: Deep image prior. In: CVPR (2018) 7, 8
- 74. Wang, C., Eckart, B., Lucey, S., Gallo, O.: Neural trajectory fields for dynamic novel view synthesis. arXiv preprint arXiv:2105.05994 (2021) 4, 13
- 75. Wang, G., Chen, Z., Loy, C.C., Liu, Z.: Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. In: ICCV (2023) 9, 10, 3, 4, 5
- 76. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing

(2004) 6

- 77. Wiles, O., Gkioxari, G., Szeliski, R., Johnson, J.: Synsin: End-to-end view synthesis from a single image. In: CVPR (2020) 4
- 78. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Xinggang, W.: 4d gaussian splatting for real-time dynamic scene rendering. In: CVPR (2024) 3, 5, 11, 12, 13, 14, 6, 7, 8
- 79. Xie, Y., Takikawa, T., Saito, S., Litany, O., Yan, S., Khan, N., Tombari, F., Tompkin, J., Sitzmann, V., Sridhar, S.: Neural fields in visual computing and beyond. In: Computer Graphics Forum. Wiley Online Library (2022) 1, 3, 4
- 80. Xu, Y., Lu, Y., Wen, Z.: Owlii dynamic human mesh sequence dataset. In: ISO/IEC JTC1/SC29/WG11 m41658, 120th MPEG Meeting (2017) 8, 9, 12, 14, 1, 3
- 81. Xu, Y., Chen, B., Li, Z., Zhang, H., Wang, L., Zheng, Z., Liu, Y.: Gaussian head avatar: Ultra high-fidelity head avatar via dynamic gaussians. In: CVPR (2024) 5
- 82. Yan, Z., Li, C., Lee, G.H.: Nerf-ds: Neural radiance fields for dynamic specular objects. In: CVPR (2023) 11, 12, 3, 6, 7
- 83. Yang, C., Li, S., Fang, J., Liang, R., Xie, L., Zhang, X., Shen, W., Tian, Q.: Gaussianobject: Just taking four images to get a high-quality 3d object with gaussian splatting. arXiv preprint arXiv:2402.10259 (2024) 14
- 84. Yang, Z., Yang, H., Pan, Z., Zhu, X., Zhang, L.: Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. In: ICLR (2024) 3, 5, 13, 14, 6, 8
- 85. Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang, Y., Jin, X.: Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In: CVPR (2024) 3, 5, 11, 12, 13, 14, 6, 7, 8

- 86. Yariv, L., Hedman, P., Reiser, C., Verbin, D., Srinivasan, P.P., Szeliski, R., Barron, J.T., Mildenhall, B.: Bakedsdf: Meshing neural sdfs for real-time view synthesis. In: SIGGRAPH (2023) 4
- 87. Yifan, W., Serena, F., Wu, S., Öztireli, C., Sorkine-Hornung, O.: Differentiable surface splatting for point-based geometry processing. ToG (2019) 4
- 88. Yu, A., Li, R., Tancik, M., Li, H., Ng, R., Kanazawa, A.: PlenOctrees for real-time rendering of neural radiance fields. In: ICCV (2021) 4, 5
- 89. Yu, A., Ye, V., Tancik, M., Kanazawa, A.: pixelnerf: Neural radiance fields from one or few images. In: CVPR (2021) 2, 4, 14
- 90. Yu, H., Julin, J., Milacski, Z.A., Niinuma, K., Jeni, L.A.: Cogs: Controllable gaussian splatting. In: CVPR (2024) 5
- 91. Yu, Z., Chen, A., Huang, B., Sattler, T., Geiger, A.: Mip-splatting: Alias-free 3d gaussian splatting. In: CVPR (2024) 3, 4, 9, 10, 5
- 92. Zhang, J., Yang, G., Tulsiani, S., Ramanan, D.: Ners: Neural reflectance surfaces for sparse-view 3d reconstruction in the wild. NeurIPS (2021) 4
- 93. Zhang, Y., Prokudin, S., Mihajlovic, M., Ma, Q., Tang, S.: Degrees of freedom matter: Inferring dynamics from point trajectories. In: CVPR (2024) 13, 14, 3
- 94. Zwicker, M., Pfister, H., Van Baar, J., Gross, M.: Ewa volume splatting. In: VIS

(2001) 2, 4, 6

- 95. Zwicker, M., Pfister, H., Van Baar, J., Gross, M.: Surface splatting. In: PACMCGIT

(2001) 2, 4

