### Gaussian Splatting with Discretized SDF for Relightable Assets

Zuo-Liang Zhu1 Jian Yang1† Beibei Wang2† 1Nankai University 2Nanjing University

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

# arXiv:2507.15629v1[cs.GR]21Jul2025

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Tbell Horse Helmet Car Qilin

Figure 1. We present a relightable Gaussian splatting framework that introduces a discretized SDF to promote decomposition quality. We show five relighting results (left) and their normal estimation (right), including ‘Tbell’ and ‘Horse’ from NeRO [17], ‘Helmet’ and ‘Car’ from Ref-NeRF [26], and ‘Qilin’ from NeILF++ [37], where ‘Qilin’ is a real scene. Our method demonstrates a robust decomposition of geometry and material for various objects, leading to photo-realistic object relighting.

##### Abstract

Gaussians onto the zero-level set of SDF and enforce alignment with the surface from splatting, namely a projectionbased consistency loss. Thanks to the discretized SDF, our method achieves higher relighting quality, while requiring no extra memory beyond GS and avoiding complex manually designed optimization. The experiments reveal that our method outperforms existing Gaussian-based inverse rendering methods. Our code is available at https: //github.com/NK-CS-ZZL/DiscretizedSDF.

3D Gaussian splatting (3DGS) has shown its detailed expressive ability and highly efficient rendering speed in the novel view synthesis (NVS) task. The application to inverse rendering still faces several challenges, as the discrete nature of Gaussian primitives makes it difficult to apply geometry constraints. Recent works introduce the signed distance field (SDF) as an extra continuous representation to regularize the geometry defined by Gaussian primitives. It improves the decomposition quality, at the cost of increasing memory usage and complicating training. Unlike these works, we introduce a discretized SDF to represent the continuous SDF in a discrete manner by encoding it within each Gaussian using a sampled value. This approach allows us to link the SDF with the Gaussian opacity through an SDF-to-opacity transformation, enabling rendering the SDF via splatting and avoiding the computational cost of ray marching. The key challenge is to regularize the discrete samples to be consistent with the underlying SDF, as the discrete representation can hardly apply the gradientbased constraints (e.g., Eikonal loss). For this, we project

##### 1. Introduction

While creating relightable 3D assets from multi-view images is valuable for many downstream tasks (e.g., augmented reality, virtual reality), it is challenging due to the ambiguity from decomposing geometry, material, and light.

A key point for decomposing these factors is the regularization of geometry, as high-quality geometry is the prerequisite to estimate reasonable material and light. The signed distance field (SDF) has been introduced into neural radiance field (NeRF) [21] as an effective geometric prior, as NeuS [28]. Due to SDF constraints, several methods [14, 25, 42] have shown a robust decomposition of geometry and material. Unfortunately, these methods require long training and rendering time, due to the expensive ray-

†Corresponding author. 1PCA Lab, VCIP, College of Computer Science, Nankai University 2School of Intelligence Science and Technology, Nanjing University,

Suzhou, China.

marching operation. More recently, 3D Gaussian splatting (3DGS) [12] achieves more detailed appearance modeling while enabling real-time rendering and fast training. It has also been introduced into inverse rendering [6, 9, 16, 44]. However, the decomposition in the GS framework becomes more problematic, due to insufficient geometric constraints. Several works [32, 35, 41, 44] optimize an extra SDF network with Gaussians jointly for geometry regularization. While these designs improve the geometry quality, they increase the memory usage and require complex optimization strategies (e.g., warm-up [35], or multi-stage training [44]) to balance two representations.

In this paper, we introduce a discretized SDF to represent the continuous SDF in a discrete manner by encoding it within each Gaussian using a sampled value. This design enables effective geometry regularization in Gaussianbased relighting frameworks while maintaining simplicity and memory efficiency. We link the SDF with the Gaussian opacity through an SDF-to-opacity transformation, allowing rendering the SDF via splatting and avoiding the computational cost of ray marching. The key problem is to regularize SDF with the discrete samples, as the gradientbased losses used to regularize SDF (e.g., Eikonal loss [7]) are unfeasible for the discrete samples. For this, we project Gaussians onto the zero-level set of SDF and enforce alignment with the surface from splatting, namely projectionbased consistency loss, proven to be an approximation of the Eikonal loss. This way, we unify the representation between SDF and Gaussian, benefiting from the flexibility of the Gaussian primitive and the robustness of SDF while maintaining simplicity, without introducing extra representation. Besides, inspired by previous SDF works [28, 33], we propose a spherical initialization tailored for Gaussian splatting, which avoids falling into local minima and improves the robustness of the model.

As a result, our method provides high-quality and robust decomposition for objects across various materials, enables realistic relighting with only 20% of memory usage compared to previous methods [44], and simplifies the optimization process, thanks to the unified representation. In summary, our main contributions include

- • a novel regularization in the GS framework with an discretized SDF to improve decomposition quality,
- • a projection-based loss to regularize discretized SDF, proven as an approximation of the Eikonal loss, and
- • extensive experiments show our method outperforms existing Gaussian-based methods with diverse metrics.

##### 2. Related works

###### 2.1. Inverse rendering

Inverse rendering aims to decompose geometry, material, and light from multi-view RGB images. One group of meth-

ods adopts the geometry formulation from NeRF [21] in a density [3, 4, 10, 24] or SDF manner [2, 25, 27, 38, 39, 42]. Due to the well-constrained geometry of the SDF, SDFbased methods can decompose the material and geometry robustly. Especially, NeRO [17] can handle challenging reflective objects by explicitly integrating the rendering equation. TensoSDF [14] uses roughness-aware integration of radiance and reflectance fields to manage diverse materials. However, these methods rely on expensive ray marching and suffer from slow training speed with high memory usage in training. Unlike prior works, our discretized SDF ensures robust decomposition across various materials, while maintaining a fast training speed and low memory costs.

More recently, Gaussian-based inverse rendering methods have emerged. These methods regularize geometry from Gaussians variously. GShader [9] and GIR [23] link the shortest axis of Gaussian with the normal. GShader, R3DG [6], and GS-IR [16] introduce a loss to align the Gaussian normal with the depth-derived one. Later, SDF is introduced into this field to regularize geometry. DeferredGS [32] distills the normal from the pretrained SDF to refine the normal from Gaussians. GS-ROR [44] utilizes the normal and depth as a bridge to optimize the Gaussian and SDF jointly. However, introducing an extra SDF network makes the model larger and harder to optimize.

Instead of using additional continuous representation for SDF, our work aims to directly encode discrete values of SDF samples on Gaussian primitives and regularize the geometry via this discretized SDF. In this way, our framework benefits from both GS and SDF within a unified representation, leading to a more lightweight and efficient model.

###### 2.2. Learning SDF from multi-view images

SDF is a fundamental surface representation [19] adopted by various tasks (e.g., surface reconstruction, inverse rendering). Particularly, NeuS [28] that links the SDF with the density in NeRF [21] became mainstream for learning SDF from multi-view images. In their framework, the SDF is stored by multi-layer perceptrons [33, 36] or grid-like representations [14, 15, 22, 29], which is continuous and convenient to obtain the gradient to regularize the SDF [7, 31, 36].

Recently, 3DGS [12] became a new framework for the multi-view stereo, providing impressive real-time novel view synthesis (NVS) results. However, due to its discrete nature, reconstructing an accurate and continuous surface is challenging. To address this, some works optimize the SDF with 3DGS jointly. NeuSG [5] align the SDF gradient with the Gaussian normal. 3DGSR [18] builds an SDF-toopacity transformation and regularizes the depth along with normal from Gaussians by the ones from SDF. GSDF [35] utilizes the depth from Gaussians to accelerate the sampling and manipulate densifying/pruning via SDF. GS-Pull [41] introduces the pulling operation to align the zero-level set

Rendered Image

[Figure 11]

Splatting with Discretized SDF Prior (Sec. 3.2)

Underlying SDF

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Gaussian Attributes

[Figure 16]

𝒎𝑖,𝒓𝑖,𝒂𝑖 𝑠𝑖,𝜇𝑖,Σ𝑖

[Figure 17]

Render

| |
|---|

Alpha Blending

[Figure 18]

[Figure 19]

PBR Attributes

[Figure 20]

Discretized SDF Samples on Gaussian

[Figure 21]

𝑜𝑖

ℒn

ℒc Normal Depth

Opacity

Sec. 3.4

Sec. 3.3

[Figure 22]

[Figure 23]

Median Loss ℒ𝛾

Ground Truth

Projection-based Consistency Loss Differentiable Env. Map

Spherical Initialization

SDF-to-opacity Transformation

- Figure 2. Overview of our framework. Our model discretizes the underlying continuous SDF into discrete samples and encodes the SDF values of samples on Gaussian primitives. We propose the median loss to facilitate convergence and projection-based consistency loss to regularize the discrete values of SDF samples. Besides, we design a spherical initialization for foreground objects to avoid local minima.

###### 3.2. Relightable GS with a discretized SDF

and the Gaussian position for an accurate surface. All previous works utilize an extra representation while our work employs a single representation that uses Gaussian primitives as discrete samples from SDF to regularize geometry.

Previous works [34, 35, 41] introduced an extra SDF representation (e.g., a network) into the GS framework as geometry constraints to improve decomposition quality, leading to higher memory costs and more complex optimization strategies. Hence, the main question is how to unify the Gaussian primitive and the SDF to avoid these drawbacks. To this end, we propose the discretized SDF into the Gaussian splatting as a geometry prior, as shown in Fig. 2.

##### 3. Method

In this section, we briefly review Gaussian splatting (Sec. 3.1) and propose a discretized SDF within the Gaussian splatting framework for inverse rendering (Sec. 3.2). Then, we present a projection-based loss to constrain the geometry when common gradient-based regularization becomes unfeasible for the discrete representation (Sec. 3.3). Finally, we propose a spherical initialization to avoid the geometry trapping into local minima (Sec. 3.4).

We first describe our baseline solution and introduce our discretized SDF. Our framework is built on 2DGS [8] with the deferred shading, as the 2D Gaussian is robust for geometry [8] and deferred shading is more stable across diverse materials [32, 44]. Besides, we choose the Principled Bidirectional Reflectance Distribution Function (BRDF) [20] with split-sum approximation [11] as our shading model.

###### 3.1. Preliminaries

Next, we propose the discretized SDF into our framework. Instead of using an extra representation for SDF, we encode an SDF value on the Gaussian as an extra attribute, besides PBR attributes, scaling, and quaternion. Then, the opacity in each Gaussian is derived from the SDF value via an SDF-to-opacity transformation:

2DGS. 2DGS [8] flattens the Gaussian primitive into a 2D disk defined by two tangent vectors (tu,tv) and their corresponding scaling (su,sv), whose normal is n = tu × tv. They compute the ray-Gaussian response αi by a ray-splat intersection and follow the rendering pipeline in 3DGS as:

n

C =

i=0

- i−1
- j=1

(1 − αj), (1)

ciαi

where ci,αi is the color and the ray-Gaussian response.

Relightable Gaussian splatting. To enable relighting, the color of Gaussian is replaced by physically based rendering (PBR) attributes (e.g., albedo, roughness). Two existing pipelines differ in the order of blending and evaluating the rendering equation, namely forward and deferred shading. The forward one renders the color for each Gaussian and blends them, while the deferred one blends the PBR attributes first and renders each pixel.

4 · e−γs

i

, (2)

oi = Tγ(si) =

(1 + e−γsi)2

where oi is the opacity, si is the SDF value of i-th Gaussian, and γ is a global learnable parameter to control the variance of the transformation. Note that the opacity of each Gaussian is computed from the SDF, rather than directly learned. With the transformation, Gaussians can be splatted the same as in previous works [8, 18], and the Gaussian attributes together with the parameter of the transformation can be optimized. However, we observe a slow convergence, as the joint optimization of the transformation parameter γ and the SDF value si spans a large search space.

###### SDF-to-opacity Transformation

Opacity 𝑜

- 0

- 1

∇𝑓 ∇𝑓

4𝑒−𝛾𝑠 1 + 𝑒−𝛾𝑠 2

−𝑠

ℒ𝛾 Median Loss

𝜀

𝒯𝛾(𝑠) =

𝒯𝛾(𝑠) 𝒯𝛾m(𝑠)

###### On the Zero-level Set

ℒ𝛾 ℒ𝛾

ℒp

0.5

Project

Signed Distance 𝑠 𝑠 𝑚

Figure 5. The projection-based consistency loss minimizes the difference between the zero-level set of SDF and the alpha-blended surface defined by Gaussians. The exceptional Gaussians whose difference is larger than ε are discarded.

- Figure 3. The SDF-to-opacity transformation Tγ for splatting is a bell-shaped function ranging from zero to one. The proposed median loss Lγ drags Tγ (red line) to narrower Tγm (blue line).

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

𝜇

𝜇proj

𝑓(𝜇) 𝑓(𝜇 proj)

∇𝑓(𝜇) ∇𝑓(𝜇proj)

𝜇

𝜇proj

Continuous Rep. Discrete Samples

[Figure 29]

[Figure 30]

Available Unavailable Only Dir.

𝑓(𝜇) 𝑓(𝜇proj)

∇𝑓(𝜇) ∇𝑓(𝜇proj)

[Figure 31]

[Figure 32]

[Figure 33]

Zero-level Set

- Figure 4. The difference between continuous and discretized SDF in regularization. For discrete samples, we can only obtain the gradient direction of SDF,1so the Eikonal loss is infeasible. Besides, since discretized SDF does not support the query of SDF value at the projected point, we cannot constrain projected points on the surface directly and thus use depth as a bridge to achieve that.

it to be sharper. In this way, the Gaussians can quickly approach the surface and converge into a sharp distribution.

###### 3.3. Projection-based consistency loss

A signed distance function f meets the Eikonal condition |∇f| = 1, indicating the gradient magnitude to position should be constantly one, which is usually used to constrain an implicit SDF, as the gradient ∇f can be computed easily from a continuous SDF. However, it is unclear how to obtain it for SDF samples encoded on Gaussians. As shown in Fig. 4, only the gradient direction on Gaussian is available, which is the normal direction n = |∇∇ff| = tu × tv. Inspired by Neural-Pull [1], we design a new constraint for the SDF samples tailored with Gaussians by projecting Gaussians to the surface defined by the zero-level set, enforcing the consistency between the projected point and alpha-blended surface defined by Gaussians, namely projection-based consistency loss, proven to be an approximation of the Eikonal condition. Note that although our design shares a similar theory with Neural-Pull, we do not use an extra network for SDF and have no ground-truth point cloud as supervision. Specifically, we project a Gaussian gi onto the zero-level set, leading to a projected point µproj:

To mitigate the challenges in the joint learning of γ and si, we found the median of unsigned distance |s|m across all Gaussian primitives is an effective indicator for convergence. Specifically, a high opacity om = Tγm

(|s|m) at the median distance indicates the transformation should be narrowed, as shown in Fig. 3. Hence, we link the transformation parameter to the median |s|m and empirically encourage the opacity om below 0.5. Accordingly, we derive γm from Tγm

µproj = µi − si ∇fi |∇fi|

(|s|m) = 0.5 to meet this condition:

, (5)

log(3 − 2√2) |s|m

, (3)

where µi, si, ∇fi are the position, SDF value and SDF gradient of gi, respectively. The projected point should align with the alpha-blended surface from Gaussians. For this, we ensure this property by minimizing their difference. Here, we use depth rather than position, due to simplicity, leading to the following difference ϵproj = |Dagg − Dproj| where Dagg is the alpha-blended or aggregated depth, and Dproj is the depth of projected point µproj under the same view.

γm = −

which is used to guide the transformation parameter γ with our proposed SDF-to-opacity transformation median loss:

Lγ = max(γm − γ,0). (4)

Note that, γm is used as a guide of γ to quickly narrow the transformation. After it reaches Tγm

defined by γm, we do not force the transformation Tγ identical to Tγm

This difference can not be used as the loss directly, as some exceptional Gaussians should be discarded. One case is that some Gaussians belong to occluded surfaces or objects in the current view. The other case is that nondifferentiable regions (such as self-intersection [19]) might

and allow

1An ideal SDF has a gradient magnitude of one, which does not hold during optimization, and discrete-sample magnitudes cannot be directly measured. Thus, only the direction on Gaussian is accessible.

GShader GS-IR R3DG GS-ROR Ours

PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS Angel 17.49 / 0.8336 / 0.1901 15.64 / 0.6126 / 0.1428 16.65 / 0.8013 / 0.1181 20.81 / 0.8775 / 0.0858 22.03 / 0.8919 / 0.0819 Bell 19.01 / 0.8804 / 0.5203 12.61 / 0.2807 / 0.1431 16.15 / 0.8391 / 0.1329 24.49 / 0.9267 / 0.0795 24.67 / 0.9280 / 0.0842 Cat 16.00 / 0.8642 / 0.1591 18.04 / 0.7907 / 0.1171 17.49 / 0.8503 / 0.1114 26.28 / 0.9421 / 0.0596 26.48 / 0.9374 / 0.0661 Horse 22.49 / 0.9262 / 0.1606 17.40 / 0.7270 / 0.0866 20.63 / 0.8832 / 0.0498 23.31 / 0.9376 / 0.0356 24.01 / 0.9481 / 0.0351 Luyu 15.62 / 0.8254 / 0.1391 19.00 / 0.7727 / 0.1094 17.47 / 0.8168 / 0.1057 22.61 / 0.8995 / 0.0672 23.80 / 0.9017 / 0.0699 Potion 12.33 / 0.7575 / 0.2142 16.37 / 0.7051 / 0.1614 14.99 / 0.7799 / 0.1832 25.67 / 0.9175 / 0.0937 27.31 / 0.9280 / 0.0982 Tbell 14.42 / 0.8007 / 0.3368 14.35 / 0.5419 / 0.1970 15.99 / 0.7965 / 0.1877 22.80 / 0.9180 / 0.0953 23.66 / 0.9191 / 0.0981 Teapot 18.21 / 0.8560 / 0.1716 16.63 / 0.7646 / 0.1312 17.36 / 0.8389 / 0.1194 21.17 / 0.8932 / 0.0982 24.19 / 0.9293 / 0.0760 Mean 16.95 / 0.8430 / 0.2365 16.26 / 0.6494 / 0.1361 17.09 / 0.8258 / 0.1260 23.39 / 0.9140 / 0.0769 24.52 / 0.9229 / 0.0762 Training 0.5h 0.5h 1h 1.5h 1h Ren. FPS 50 214 50 208 143 Memory 4G 8G 20G 22G 4G

- Table 1. The quantitative comparison with Gaussian-based methods on the Glossy Blender dataset in terms of PSNR↑, SSIM↑, and LPIPS↓. Numbers in red indicate the best performance, and numbers in orange indicate the second best.

GShader GS-IR R3DG GS-ROR Ours

PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS Armad. 22.86 / 0.9280 / 0.0821 28.31 / 0.9121 / 0.0591 30.76 / 0.9526 / 0.0649 31.33 / 0.9593 / 0.0482 31.05 / 0.9621 / 0.0536 Ficus 24.61 / 0.9491 / 0.0956 24.39 / 0.8762 / 0.0360 27.23 / 0.9637 / 0.0437 26.28 / 0.9542 / 0.0439 27.85 / 0.9639 / 0.0390 Hotdog 17.45 / 0.8838 / 0.1408 22.36 / 0.8931 / 0.0875 24.59 / 0.9162 / 0.1275 25.21 / 0.9307 / 0.0771 26.23 / 0.9360 / 0.0746 Lego 13.41 / 0.7904 / 0.1193 23.79 / 0.8750 / 0.0879 22.49 / 0.8682 / 0.1423 25.46 / 0.9083 / 0.0724 25.81 / 0.9087 / 0.0791 Mean 19.58 / 0.8878 / 0.1095 24.71 / 0.8891 / 0.0676 26.27 / 0.9252 / 0.0946 27.07 / 0.9381 / 0.0604 27.78 / 0.9427 / 0.0626

- Table 2. The quantitative comparison with Gaussian-based methods on the TensoIR Synthetic dataset in terms of PSNR↑, SSIM↑, and LPIPS↓. Numbers in red indicate the best performance and numbers in orange indicate the second best.

lead to a huge difference ϵproj. These two cases interrupt consistency significantly. Unfortunately, it is difficult to recognize them. Therefore, we use a simple and practical solution, by assuming the difference between the two surfaces should be already close and thresholding the difference with a hyper-parameter ε, as shown in Fig. 5.

Lp =

1 N i∈N

ϵiproj, ϵiproj ≤ ε, 0, ϵiproj > ε.

(6)

Note that we apply this loss after 1K iterations when the coarse geometry is stable so that the assumption holds.

The theory behind the loss is an approximation of the Eikonal condition, proven in the supplementary. This loss encourages smooth reconstructed surfaces while maintaining the flexibility from Gaussian for geometry details.

- 3.4. Spherical initialization

minima and to encourage optimal geometry. Specifically, we employ the points uniformly sampled on a unit sphere to initialize Gaussians. The design mitigates broken surfaces and overfitting under the training appearance, further improving the geometry robustness.

##### 4. Experiments

In this section, we present our training details (Sec. 4.1) and evaluation setup (Sec. 4.2). Then, we compare our method with the representative methods (Sec. 4.3) and conduct ablation studies on our proposed designs (Sec. 4.4).

###### 4.1. Implementation details

We optimized our model using Adam optimizer [13] for 30,000 iterations with the loss:

L = Lc+λnLn+λdLd+λγLγ+λpLp+λsmLsm+λmLm, (7)

We found that some erroneous geometry emerges at the early stage of training and can hardly be recovered. This indicates that the initialization makes the geometry prone to trapping into local minima. Since sphere-like initializations revealed their effect in volumetric rendering of SDF [28, 33], we design a spherical initialization of foreground objects tailored for Gaussian splatting to avoid local

where λ[·] is the corresponding weight of loss L[·]. Lc is the rendering loss between the rendered image and the ground

truth. Ln and Ld are the normal consistency and the distortion loss, inheriting from 2DGS. Lsm is a smoothness regularization for PBR attributes, and Lm is the optional mask loss widely adopted by existing relighting methods [6, 38].

GS-IR

R3DG

GShader

GS-ROR

Ours

Ground Truth

|[Figure 34]|
|---|

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

- 19.39 / 0.8198 18.63 / 0.8388 20.09 / 0.8521 22.19 / 0.8819 25.44 / 0.9284

18.71 / 0.8622 17.94 / 0.8470 19.30 / 0.9054 21.85 / 0.9179 22.87 / 0.9361

- 20.72 / 0.8495 21.21 / 0.8658 15.53 / 0.8388 20.57 / 0.9014 21.89 / 0.9017

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

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

[Figure 79]

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

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

|[Figure 98]|
|---|

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

- Figure 6. The comparison of relighting results and normal on the Glossy Blender (the 1st-4th rows) and the TensoIR synthetic dataset (the 5th, 6th rows). Our method provides reasonable normal robustly across materials, thus rendering realistic relighting results that include specular highlights. The PSNR/SSIM of relighting results under the current view are below images. More results are in the supplementary.

Lγ and Lp are our median and projection-based loss. More details are in the supplementary.

the relighting images for quantitative evaluation. We compare the mean angular error (MAE) and the Chamber distance (CD) for geometry quality. We record the mean training time, the frame per second (FPS), and the memory usage during training on RTX 4090.

###### 4.2. Evaluation setup

Dataset. We evaluate our method on three synthetic datasets, including TensoIR [10] synthetic dataset, Glossy Blender dataset [17] and Shiny Blender dataset. For real scenes, we select some objects from NeILF++ [37].

###### 4.3. Comparison with previous methods

Relighting of synthetic objects. We evaluate the relighting performance of synthetic objects with various materials on the Glossy Blender and TensoIR synthetic datasets. The quantitative measurements are shown in Tab. 1 and Tab. 2. The relighting results with normal visualizations are in Fig. 6. Due to the robustness of decomposition, our method provides high-quality relighting results and smooth normal. In contrast, GS-IR, R3DG, and GShader overfit the training appearance, leading to erroneous normal. Regarding GS-ROR, it shows higher geometry quality, as it introduces a continuous SDF for regularization. However, the

Methods for comparison. We select 4 representative Gaussian-based inverse rendering methods for comparison, including GShader [9], GS-IR [16], R3DG [6] and GSROR [44]. We trained these models based on their public codes and configurations with an extra mask loss for stabilizing training and fair comparison.

Metrics. We use the peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM) [30], the learned perceptual image patch similarity (LPIPS) [40] to measure the relighting quality. Following NeRO and TensoIR, we rescale

Ref. Image & Env. Map GShader GS-ROR Ours

[Figure 105]

[Figure 106]

[Figure 107]

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

[Figure 111]

[Figure 112]

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]|
|---|

[Figure 116]

[Figure 117]

[Figure 118]

- Figure 7. The qualitative comparison of relighting results and normal on the NeILF++ dataset. Our method can provide realistic relighting results and detailed normal for real data. More results are in the supplementary.

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

[Figure 129]

[Figure 130]

[Figure 131]

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

[Figure 136]

[Figure 137]

[Figure 138]

|[Figure 139]|
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|[Figure 142]|
|---|

R3DG GS-IR GShader GS-ROR Ours Ground Truth

27.34

19.77

15.26 10.97 7.62

25.91 8.39 3.70 3.65

25.34

- Figure 8. The qualitative comparison of normal on the Shiny Blender dataset. Our method provides globally smooth normal while preserving the sharp details. The MAE under the current view is below the visualization. More results are in the supplementary.

GS-IR R3DG GShader GS-ROR Ours

GS-IR R3DG GShader GS-ROR Ours Ball 25.79 22.44 7.03 0.92 1.20 Car 28.31 26.02 14.05 11.98 6.82 Coffee 15.38 13.39 14.93 12.24 11.42 Helmet 25.58 19.63 9.33 4.10 3.90 Teapot 15.35 9.21 7.17 5.88 5.02 Toaster 33.51 28.17 13.08 8.24 10.51 Mean 23.99 19.81 10.93 7.23 6.48

Angel 0.0177 0.0098 0.0085 0.0063 0.0053 Bell 0.1153 0.0418 0.0110 0.0096 0.0094 Cat 0.0588 0.0339 0.0256 0.0222 0.0251 Horse 0.0196 0.0135 0.0073 0.0061 0.0052 Luyu 0.0225 0.0168 0.0107 0.0098 0.0106 Potion 0.0623 0.0380 0.0474 0.0146 0.0081 Tbell 0.1021 0.0500 0.0574 0.0270 0.0101 Teapot 0.0719 0.0479 0.0340 0.0166 0.0115 Mean 0.0588 0.0315 0.0253 0.0140 0.0107

Table 3. Normal quality with Gaussian-based methods on Shiny Blender dataset in terms of MAE↓. Numbers in red indicate the best performance and numbers in orange indicate the second best.

Table 4. Surface quality with Gaussian-based methods on Glossy Blender dataset regarding CD↓. Numbers in red indicate the best performance, and numbers in orange indicate the second best.

geometry details are lost in some regions, due to the difficulty in balancing two representations during training.

methods on most scenes and average metrics. However, our method fails in scenes with complex inter-reflection (‘Toaster’, ‘Luyu’, ‘Cat’) but still achieves sub-SOTA performance. Only GS-ROR succeeds in the scenes, at the cost of 5× of GPU memory usage than ours, due to their extra SDF network. Visually, our model provides globally smooth normal (teapot in Fig. 6) and preserves local details (horse tail in Fig. 6), while GS-ROR discards sharp details and others fail to reconstruct smooth surfaces.

Relighting of real objects. The results of real scenes from NeILF++ are shown in Fig. 7. As ground truth is unavailable, we show reference training views and the results under novel light conditions.2 Our method produces reasonable relighting results with sharp geometry details preserved.

Geometry quality. We compare reconstructed geometry quality in terms of normal MAE in Tab. 3 with the visualization in Fig. 8 and CD in Tab. 4.3Numerically, our model outperforms all Gaussian-based inverse rendering

3The choice of quantitative metrics depends on the available ground truth, and we use TSDF fusion [43] for surface extraction to evaluate CD.

2R3DG, GS-IR do not converge on the scenes, thus no results provided.

Baseline SDF Proj. Sph. SDF Proj. Sph. SDF Proj. Sph. SDF Proj. Sph. SDF Proj. Sph.

Ground Truth

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

N/A

- Figure 9. Ablation of the key components in our method on the Glossy Blender dataset, including the discretized SDF framework with the median loss, the projection-based consistency loss (Proj.), and spherical initialization (Sph.). (The red arrow points out the artifacts.)

|Components<br><br>|Scene|
|---|---|
|SDF Sph. Proj.<br><br>|Angel Horse Teapot PSNR/SSIM PSNR/SSIM PSNR/SSIM|
|✔<br><br>✔<br><br>✔ ✔<br><br>✔ ✔<br>✔ ✔ ✔<br>|20.23/0.8533 20.72/0.8998 19.22/0.8674<br>21.09/0.8815 22.13/0.9120 23.26/0.9223 21.43/0.8881 21.96/0.9102 22.97/0.9176 21.59/0.8910 22.87/0.9284 23.45/0.9222<br><br><br>21.65/0.8917 23.21/0.9395 24.05/0.9289<br><br>22.03/0.8919 24.01/0.9481 24.19/0.9293<br><br><br>|

Spherical initialization. The spherical initialization leads to performance increase with or without SDF designs (the 3rd,4th,6th columns of Fig. 9). However, applying it individually cannot ensure artifact-free results.

###### 4.5. Discussion and limitations

In this work, we focus on improving the relighting quality by introducing SDF prior and leave mesh extraction with BRDF parameters for future work. Besides, we consider the direct lighting only for efficiency, so our method may fail for objects with complex occlusion. An extra indirect illumination term [6, 17] will further benefit our approach.

- Table 5. Ablation study of three key components on the Glossy Blender dataset. “SDF” means incorporating the discretized SDF with the median loss, “Proj.” means the projection-based consistency loss, and “Sph.” means the spherical initialization.

###### 4.4. Ablation study

We conduct ablation studies of our main components on the Glossy Blender dataset, and the ablations of detailed choices (e.g., losses, threshold) are in the supplementary. The quantitative results are in the Tab. 5. The performance of the model improves consistently when employing new components. We start from the baseline described in Sec. 3.2. As shown in the 1st column of Fig. 9, this baseline overfits the training appearance and fails to decompose geometry and material for relighting.

Discretized SDF with the median loss. Then, we incorporate the SDF-to-opacity transformation and the median loss that promotes the Gaussians converging to the opaque surface. As shown in the 2nd column of Fig. 9. The variant can decompose the material and geometry more robustly, thus improving the relighting quality. However, due to insufficient constraints, we still observe erroneous normal.

Projection-based consistency loss. Due to this loss is coupled with SDF, it cannot be ablated individually. Applying the projection-based consistency loss leads to more reasonable normal. However, erroneous geometry caused by misleading initialization still exists, as in the 5th row of Fig. 9.

##### 5. Conclusion

In this paper, we presented our Gaussian splatting framework for real-time object relighting, which regularizes the geometry from Gaussian primitives with a discretized SDF. In this framework, we proposed the median loss to facilitate the convergence and the projection-based consistency loss to constrain the discrete samples of SDF, along with the spherical initialization to avoid local minima. Therefore, our method benefits from the flexibility of GS and the robustness of SDF, thus outperforming the existing Gaussianbased inverse rendering methods in terms of relighting and decomposition quality, while maintaining simplicity and efficiency. In future work, extending our framework to unbounded scenes is a promising direction.

Acknowledgment: We thank the reviewers for the valuable comments. This work has been partially supported by the National Science and Technology Major Project under grant No. 2022ZD0116305, National Natural Science Foundation of China (NSFC) under grant No. 62172220, and the NSFC under Grant Nos. 62361166670 and U24A20330. Computation is supported by the Supercomputing Center of Nankai University.

##### References

- [1] Ma Baorui, Han Zhizhong, Liu Yu-Shen, and Zwicker Matthias. Neural-Pull: Learning signed distance functions from point clouds by learning to pull space onto surfaces. In ICML, 2021. 4
- [2] Sai Bi, Zexiang Xu, Pratul Srinivasan, Ben Mildenhall, Kalyan Sunkavalli, Miloˇs Haˇsan, Yannick Hold-Geoffroy, David Kriegman, and Ravi Ramamoorthi. Neural reflectance fields for appearance acquisition. arXiv preprint arXiv:2008.03824, 2020. 2
- [3] Mark Boss, Raphael Braun, Varun Jampani, Jonathan T. Barron, Ce Liu, and Hendrik P.A. Lensch. NeRD: Neural reflectance decomposition from image collections. In ICCV, pages 12684–12694, New York, NY, USA, 2021. IEEE. 2
- [4] Mark Boss, Varun Jampani, Raphael Braun, Ce Liu, Jonathan Barron, and Hendrik PA Lensch. Neural-PIL: Neural pre-integrated lighting for reflectance decomposition. In NeurIPS, pages 10691–10704, Red Hook, NY, USA, 2021. Curran Associates, Inc. 2
- [5] Hanlin Chen, Chen Li, and Gim Hee Lee. NeuSG: Neural implicit surface reconstruction with 3D Gaussian splatting guidance. arXiv preprint arXiv:2312.00846, 2023. 2
- [6] Jian Gao, Chun Gu, Youtian Lin, Zhihao Li, Hao Zhu, Xun Cao, Li Zhang, and Yao Yao. Relightable 3D Gaussians: Realistic point cloud relighting with BRDF decomposition and ray tracing. In ECCV, 2024. 2, 5, 6, 8
- [7] Amos Gropp, Lior Yariv, Niv Haim, Matan Atzmon, and Yaron Lipman. Implicit geometric regularization for learning shapes. In ICML. JMLR.org, 2020. 2
- [8] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2D Gaussian splatting for geometrically accurate radiance fields. In SIGGRAPH, New York, NY, USA,

2024. Association for Computing Machinery. 3

- [9] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and Yuexin Ma. GaussianShader: 3D Gaussian splatting with shading functions for reflective surfaces. In CVPR, pages 5322–5332, New York, NY, USA,

2024. IEEE. 2, 6

- [10] Haian Jin, Isabella Liu, Peijia Xu, Xiaoshuai Zhang, Songfang Han, Sai Bi, Xiaowei Zhou, Zexiang Xu, and Hao Su. TensoIR: Tensorial inverse rendering. In CVPR, pages 165– 174, New York, NY, USA, 2023. IEEE. 2, 6
- [11] Brian Karis. Real shading in unreal engine 4. Proc. Physically Based Shading Theory Practice, 2013. 3
- [12] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3D Gaussian splatting for real-time radiance field rendering. ACM TOG, 42(4):139–1, 2023. 2
- [13] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2017. 5

- [14] Jia Li, Lu Wang, Lei Zhang, and Beibei Wang. TensoSDF: Roughness-aware tensorial representation for robust geometry and material reconstruction. ACM TOG, 43(4), 2024. 1, 2
- [15] Zhaoshuo Li, Thomas M¨uller, Alex Evans, Russell H. Taylor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin. Neuralangelo: High-fidelity neural surface reconstruction. In

- CVPR, pages 8456–8465, New York, NY, USA, 2023. IEEE. 2
- [16] Zhihao Liang, Qi Zhang, Ying Feng, Ying Shan, and Kui Jia. GS-IR: 3D Gaussian splatting for inverse rendering. In CVPR, pages 21644–21653, New York, NY, USA, 2024. IEEE. 2, 6
- [17] Yuan Liu, Peng Wang, Cheng Lin, Xiaoxiao Long, Jiepeng Wang, Lingjie Liu, Taku Komura, and Wenping Wang. NeRO: Neural geometry and BRDF reconstruction of reflective objects from multiview images. ACM TOG, 42(4), 2023. 1, 2, 6, 8
- [18] Xiaoyang Lyu, Yang-Tian Sun, Yi-Hua Huang, Xiuzhe Wu, Ziyi Yang, Yilun Chen, Jiangmiao Pang, and Xiaojuan Qi. 3DGSR: Implicit surface reconstruction with 3D Gaussian splatting. ACM TOG, 43(6), 2024. 2, 3
- [19] Daniel Mayost. Applications of the signed distance function to surface geometry. University of Toronto, 2014. 2, 4
- [20] Stephen McAuley, Stephen Hill, Adam Martinez, Ryusuke Villemin, Matt Pettineo, Dimitar Lazarov, David Neubelt, Brian Karis, Christophe Hery, Naty Hoffman, and Hakan Zap Andersson. Physically based shading in theory and practice. In ACM SIGGRAPH 2013 Courses, New York, NY, USA, 2013. Association for Computing Machinery. 3
- [21] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In ECCV, pages 405–421, Berlin, Heidelberg, 2020. Springer. 1, 2
- [22] Radu Alexandru Rosu and Sven Behnke. PermutoSDF: Fast multi-view reconstruction with implicit surfaces using permutohedral lattices. In CVPR, pages 8466–8475, New York, NY, USA, 2023. IEEE. 2
- [23] Yahao Shi, Yanmin Wu, Chenming Wu, Xing Liu, Chen Zhao, Haocheng Feng, Jingtuo Liu, Liangjun Zhang, Jian Zhang, Bin Zhou, Errui Ding, and Jingdong Wang. GIR: 3D Gaussian inverse rendering for relightable scene factorization. arXiv preprint arXiv:2312.05133, 2023. 2
- [24] Pratul P. Srinivasan, Boyang Deng, Xiuming Zhang, Matthew Tancik, Ben Mildenhall, and Jonathan T. Barron. NeRV: Neural reflectance and visibility fields for relighting and view synthesis. In CVPR, pages 7495–7504, New York, NY, USA, 2021. IEEE. 2
- [25] Cheng Sun, Guangyan Cai, Zhengqin Li, Kai Yan, Cheng Zhang, Carl Marshall, Jia-Bin Huang, Shuang Zhao, and Zhao Dong. Neural-PBIR reconstruction of shape, material, and illumination. In ICCV, pages 18046–18056, New York, NY, USA, 2023. IEEE. 1, 2
- [26] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T. Barron, and Pratul P. Srinivasan. Ref-NeRF: Structured view-dependent appearance for neural radiance fields. In CVPR, pages 5491–5500, New York, NY, USA,

2022. IEEE. 1

- [27] Haoyuan Wang, Wenbo Hu, Lei Zhu, and Rynson W.H. Lau. Inverse rendering of glossy objects via the neural plenoptic function and radiance fields. In CVPR, pages 19999–20008, New York, NY, USA, 2024. IEEE. 2

- [28] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. NeuS: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. In NeurIPS, pages 27171–27183, Red Hook, NY, USA, 2021. Curran Associates, Inc. 1, 2, 5
- [29] Yiming Wang, Qin Han, Marc Habermann, Kostas Daniilidis, Christian Theobalt, and Lingjie Liu. NeuS2: Fast learning of neural implicit surfaces for multi-view reconstruction. In ICCV, pages 3295–3306, New York, NY, USA,

2023. IEEE. 2

- [30] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 13(4):600–612, 2004. 6
- [31] Zixiong Wang, Yunxiao Zhang, Rui Xu, Fan Zhang, PengShuai Wang, Shuangmin Chen, Shiqing Xin, Wenping Wang, and Changhe Tu. Neural-singular-hessian: Implicit neural representation of unoriented point clouds by enforcing singular hessian. ACM TOG, 42(6), 2023. 2
- [32] Tong Wu, Jia-Mu Sun, Yu-Kun Lai, Yuewen Ma, Leif Kobbelt, and Lin Gao. DeferredGS: Decoupled and editable Gaussian splatting with deferred shading. arXiv preprint arXiv:2404.09412, 2024. 2, 3
- [33] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Volume rendering of neural implicit surfaces. In NeurIPS, pages 4805–4815, Red Hook, NY, USA, 2021. Curran Associates, Inc. 2, 5
- [34] Keyang Ye, Qiming Hou, and Kun Zhou. 3D Gaussian splatting with deferred reflection. In SIGGRAPH, New York, NY, USA, 2024. Association for Computing Machinery. 3
- [35] Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xiangli, and Bo Dai. GSDF: 3DGS meets SDF for improved rendering and reconstruction. In NeurIPS, 2024. 2, 3
- [36] Jingyang Zhang, Yao Yao, Shiwei Li, Tian Fang, David McKinnon, Yanghai Tsin, and Long Quan. Critical regularizations for neural surface reconstruction in the wild. In CVPR, pages 6270–6279, New York, NY, USA, 2022. IEEE. 2
- [37] Jingyang Zhang, Yao Yao, Shiwei Li, Jingbo Liu, Tian Fang, David McKinnon, Yanghai Tsin, and Long Quan. NeILF++: Inter-reflectable light fields for geometry and material estimation. In ICCV, pages 3601–3610, New York, NY, USA,

2023. IEEE. 1, 6

- [38] Kai Zhang, Fujun Luan, Qianqian Wang, Kavita Bala, and Noah Snavely. PhySG: Inverse rendering with spherical Gaussians for physics-based material editing and relighting. In CVPR, pages 5453–5462, New York, NY, USA, 2021. IEEE. 2, 5
- [39] Kai Zhang, Fujun Luan, Zhengqi Li, and Noah Snavely. IRON: Inverse rendering by optimizing neural sdfs and materials from photometric images. In CVPR, pages 5565–5574, New York, NY, USA, 2022. IEEE. 2
- [40] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, New York, NY, USA, 2018. IEEE. 6
- [41] Wenyuan Zhang, Yu-Shen Liu, and Zhizhong Han. Neural signed distance function inference through splatting 3D Gaussians pulled on zero-level set. In NeurIPS, 2024. 2, 3

- [42] Yuanqing Zhang, Jiaming Sun, Xingyi He, Huan Fu, Rongfei Jia, and Xiaowei Zhou. Modeling indirect illumination for inverse rendering. In CVPR, pages 18643–18652, New York, NY, USA, 2022. IEEE. 1, 2
- [43] Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Open3D: A modern library for 3d data processing. arXiv preprint arXiv:1801.09847, 2018. 7
- [44] Zuo-Liang Zhu, Beibei Wang, and Jian Yang. GS-ROR: 3D Gaussian splatting for reflective object relighting via sdf priors. arXiv preprint arXiv:2406.18544, 2024. 2, 3, 6

## Gaussian Splatting with Discretized SDF for Relightable Assets Supplementary Material

#### 1. Model details

In this section, we complete the details of our model, including the losses, hyperparameters, shading model, and lighting model.

###### 1.1. Loss details

We introduce the definition and weights of losses we used in the main paper. The color loss is defined as in 3DGS [4]

Lc = λ||Cgs−Cgt||1+(1−λ)(1−SSIM(Cgs,Cgt)), (1)

where Cgs is the rendered color from Gaussians and Cgt is the ground truth color. The coefficient λ is set to 0.8. The supervision Ln from normal is defined as

Ln = ||nˆ − n||2, (2)

where n,nˆ are the normal from depth and the normal from Gaussians, respectively. We follow the 2DGS and apply the distortion loss as

Ld =

wiwj|Di − Dj|, (3)

i,j

where wi is the blending weight of i-th Gaussian and Di is the intersection depth for the current pixel. The smoothness loss for BRDF parameters (i.e., albedo a, roughness r, metallicity m) is defined as

Lsm = ||∇p||exp−||∇C

gt||,(p ∈ {a,r,m}) (4)

where ∇ is the gradient operator and Cgt is the color of ground-truth image. The mask loss is the binary cross entropy between predicted and ground-truth masks. The corresponding weights for the above losses are shown in Tab. 1.

The learning rate of Gaussian attributes follows Gaussian Shader [1], except for the rate of SDF value (set to 0.05). The details of losses are present in the supplementary. The entire training takes about 1 hour on an RTX 4090.

###### 1.2. Hyperparameters

As the median loss is used to promote convergence, we remove this loss when the SDF median sm < 0.2, which indicates the transformation has been converged. Besides, the relaxing threshold ε for the projection-based consistency loss is set to 0.05. The number N for the spherical initialization is 106.

|Loss<br><br>|Ln Ld Lγ Lp Lsm Lm|
|---|---|
|Weight<br><br>|0.2 2000 1 10 0.05 0.2|

Table 1. The weights of losses present in the paper.

###### 1.3. Split-sum approximation

A typical rendering equation needs to compute an integral on the upper hemisphere involving incident light, view direction, normal, and Bidirectional Reflectance Distribution Function (BRDF):

c(ωo) =

Li(ωi)f(ωi,ωo)(ωi · n)dωi, (5)

Ω

where Li(ωi) denotes the light from incident direction ωi, f(ωi,ωo) is the BRDF with respect to the incident direction ωi and outgoing direction ωo, and n denotes the surface normal. In practice, the Disney Principled BRDF [7] is the most widely used BRDF, consisting of a diffuse lobe and a specular lobe

a π

f(ωi,ωo) = (1 − m)

diffuse

DFG 4(ωi · n)(ωo · n) specular

, (6)

+

where D is the normal distribution function, F is the Fresnel term, G is the geometry term, a is the albedo, and m is the metallicity. However, the integral is expensive to evaluate, and the split-sum technique [3] is an alternative widely used in real-time rendering. The specular term is approximated as

cspecular = Lspecular ·((1−m)×0.04+m×a)×F1 +F2,

(7) where F1,F2 are two scalars from a pre-computed table. The diffuse term is computed as

a(1 − m)

π · Ldiffuse. (8)

cdiffuse =

Ldiffuse and Lspecular are directed queried for a pre-filtered environment map or neural light representations. The rendered color c(ωo) is computed as

c(ωo) = cdiffuse + cspecular. (9)

###### 1.4. Lighting model

Considering the efficiency of the light query, we use a simple differential environment map and do not model the indirect illumination and occlusion. The environment light is in the cube map format, whose resolution is 6 × 512 × 512.

MII TensoIR TensoSDF NeRO Ours PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS Angel 16.24 / 0.8236 / 0.1404 10.24 / 0.2238 / 0.2739 20.40 / 0.8969 / 0.0871 16.21/ 0.7819 / 0.1923 22.03 / 0.8919 / 0.0819 Bell 17.41 / 0.8594 / 0.1534 10.11 / 0.1018 / 0.2806 29.91 / 0.9767 / 0.0263 31.19 / 0.9794 / 0.0189 24.67 / 0.9280 / 0.0842 Cat 17.68 / 0.8521 / 0.1429 9.10 / 0.1644 / 0.2146 26.12 / 0.9354 / 0.0675 28.42 / 0.9579 / 0.0455 26.48 / 0.9374 / 0.0661 Horse 20.98 / 0.8997 / 0.0713 10.42 / 0.1931 / 0.2913 27.18 / 0.9567 / 0.0318 25.56 / 0.9437 / 0.0410 24.01 / 0.9481 / 0.0351 Luyu 17.89 / 0.8050 / 0.1393 8.27 / 0.2375 / 0.2463 19.91 / 0.8825 / 0.0807 26.22 / 0.9092 / 0.0696 23.80 / 0.9017 / 0.0699 Potion 17.13 / 0.8094 / 0.1747 6.21 / 0.0846 / 0.2954 27.71 / 0.9422 / 0.0759 30.14 / 0.9561 / 0.0623 27.31 / 0.9280 / 0.0982 Tbell 16.54 / 0.8262 / 0.1938 7.47 / 0.1609 / 0.2786 23.33 / 0.9404 / 0.0543 25.45 / 0.9607 / 0.0407 23.66 / 0.9191 / 0.0981 Teapot 16.71 / 0.8546 / 0.1426 9.96 / 0.2093 / 0.2030 25.16 / 0.9482 / 0.0485 29.87 / 0.9755 / 0.0193 24.19 / 0.9293 / 0.0760 Mean 17.57 / 0.8413 / 0.1448 8.97 / 0.1719 / 0.2605 24.97 / 0.9349 / 0.0590 26.63 / 0.9331 / 0.0612 24.52 / 0.9229 / 0.0762 Training 4h 5h 6h 12h 1h Ren. FPS 1/30 1/60 1/4 1/4 143 Memory 12G 23G 20G 8G 4G

- Table 2. The quantitative comparison with NeRF-based methods on the Glossy Blender dataset in terms of PSNR↑, SSIM↑, and LPIPS↓. Numbers in red indicate the best performance, numbers in orange indicate the second best, and numbers in yellow indicate the third best. Although our method has a performance gap compared to NeRO and TensoSDF, our method is more efficient (at most 50% memory usage and 17% training time).

MII NeRO TensoSDF TensoIR Ours PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS PSNR / SSIM / LPIPS Armad. 26.85 / 0.9441 / 0.0692 23.02 / 0.9335 / 0.0644 23.02 / 0.9355 / 0.0578 34.51 / 0.9754 / 0.0368 31.05 / 0.9621 / 0.0536 Ficus 20.65 / 0.9068 / 0.0728 27.43 / 0.9404 / 0.0677 28.53 / 0.9499 / 0.0533 24.32 / 0.9465 / 0.0543 27.85 / 0.9639 / 0.0390 Hotdog 22.65 / 0.9011 / 0.0893 20.45 / 0.9262 / 0.0888 20.47 / 0.9241 / 0.0906 27.92 / 0.9324 / 0.0833 26.23 / 0.9360 / 0.0746 Lego 23.20 / 0.8643 / 0.1715 17.76 / 0.8577 / 0.1228 17.92 / 0.8670 / 0.1088 27.61 / 0.9253 / 0.0702 25.81 / 0.9087 / 0.0791 Mean 23.34 / 0.9041 / 0.1007 22.17 / 0.9145 / 0.0859 22.48 / 0.9191 / 0.0776 28.59 / 0.9449 / 0.0612 27.78 / 0.9427 / 0.0626

- Table 3. The quantitative comparison with NeRF-based methods on the TensoIR Synthetic dataset in terms of PSNR↑, SSIM↑, and LPIPS↓. Numbers in red indicate the best performance, numbers in orange indicate the second best, and numbers in yellow indicate the third best.

#### 2. Proof of the projection-based loss

This part reveals how our projection-based consistency loss approximates the Eikonal loss. Proposition 1. Consider a differentiable function f in the region Ω, for ∀x1 ∈ Ω,f(x1) ̸= 0 and its projection point x0 = x1−∇f(x1)/|∇f(x1)|·f(x1), if we guarantee x0 on the surface (f(x0) = 0), the Eikonal condition |∇f(x)| = 1 can be approximated at least in a narrow thin-shell space surrounding the surface. Proof. The Taylor expansion of f at position x1 is

f(x) = f(x1) + ∇f(x1)(x − x1) + O((x − x1)2) (10)

where ∇f(x1) means the gradient of f at x1 and O((x − x1)2) means the higher-order term of f. Then substitute x0 into f(x):

x0 − x1 = −f(x1) · ∇f(x1)/|∇f(x1)|, f(x0) = f(x1) − f(x1) ∇f(x1)2

(11)

+ O((x0 − x1)2) f(x1)(1 − |∇f(x1)|) + O((x0 − x1)2) = 0

|∇f(x1)|

GS-ROR Ours w/o mask Ours w/ mask

PSNR / SSIM PSNR / SSIM PSNR / SSIM Angel 20.81 / 0.8775 21.03 / 0.8842 22.03 / 0.8919 Horse 23.31 / 0.9376 23.35 / 0.9406 24.01 / 0.9481 Teapot 21.17 / 0.8932 22.57 / 0.9087 24.19 / 0.9293

Table 4. The impact of the mask loss on our method.

which means |∇f(x)| = 1 is approximated for each x1 where x0,x1 is close enough to make the high-order term O((x0 − x1)2) negligible. Therefore, we can guarantee the Eikonal condition near the surface where Gaussians exist.

#### 3. More results

In this section, we present the ablation of detailed choices in our model and more results of our method. Additionally, we select some representative NeRF-based methods for further validation, including MII [10], TensoIR [2], TensoSDF [5], and NeRO [6].

ε 0.01 0.05 0.1 0.2

PSNR / SSIM PSNR / SSIM PSNR / SSIM PSNR / SSIM Angel 21.78 / 0.8860 21.91 / 0.8880 22.03 / 0.8919 21.54 / 0.8769 Horse 22.10 / 0.9214 22.51 / 0.9317 24.01 / 0.9481 20.12 / 0.9220 Teapot 23.28 / 0.9190 23.84 / 0.9232 24.19 / 0.9293 23.37 / 0.9200

Table 5. Ablation study on the threshold ε in our method.

Iterations 7K 15K 30K

PSNR / SSIM PSNR / SSIM PSNR / SSIM No reg. 17.96 / 0.8881 22.13 / 0.9175 22.82 / 0.9330 γ = γm 18.85 / 0.9032 22.46 / 0.9239 22.91 / 0.9332

Lγ 22.20 / 0.9221 23.64 / 0.9319 24.19 / 0.9481

- Table 6. Ablation study on the median loss Lγ in our method. “No reg.” means imposing no regularization on the transformation.

while others variants fail, as shown in Tab. 6. Besides, the setting of Tγm

(|s|m) = om = 0.5 is from our observation, as larger γm values cause early over-pruning of Gaussians, while smaller values fail to narrow the transformation. Both cases reduce performance.

The impact of the mask loss. While removing the mask loss leads to a performance decrease of 1dB, it still outperforms GS-ROR trained with the mask loss, as shown in Tab. 4.

###### 3.2. Extra benefits from the projection-based loss

Our projection loss also promotes the Gaussian outliers moving toward the surface and diminishes the artifacts, as shown in Fig. 1.

###### 3.3. Glossy Blender dataset

GS-IR GShader R3DG GS-ROR Ours PSNR / CD PSNR / CD PSNR / CD PSNR / CD PSNR / CD

We present the relighting results with the decomposition of material and geometry on the Glossy Blender dataset [6] in Fig. 2. As there is no material ground truth, we only provide the visualization of our method. More qualitative comparisons are in Fig. 3. Our method offers smooth geometry with details preserved and thus outperforms other Gaussian-based methods regarding relighting quality. Besides, we compare our method with four NeRF-based inverse rendering methods in Tab. 2, and our method is still competitive. MII and TensoIR do not design for reflective surfaces and thus perform poorly on the Glossy dataset. Although our method has a performance gap compared to NeRO and TensoSDF, our method is far more efficient than theirs with 50% memory usage and 17% training time at most. Our method also supports real-time rendering while other NeRF-based methods fail1. Besides, as shown in Fig. 4, the results from our methods include sharp details that other NeRF-based methods fail to reconstruct and thus are more visually realistic.

Ball 18.30/—– 30.40/—– 21.39/—– 35.50/—– 34.64/—– Car 25.30/0.61 28.39/0.26 26.59/0.23 30.52/0.17 30.55/0.14

Coffee 30.72/1.22 30.79/1.05 32.57/1.14 30.79/1.51 32.91/0.92 Helmet 25.08/1.22 26.95/1.09 28.78/1.31 32.62/0.23 30.06/0.15 Toaster 18.66/0.72 23.95/0.75 20.07/0.80 25.89/0.53 25.02/0.68 Teapot 38.21/—– 43.35/—– 43.86/—– 43.88/—– 43.39/—–

Table 7. Comparison in terms of PSNR and CD↓ (×100).

|[Figure 156]|
|---|

|[Figure 157]|
|---|

[Figure 158]

[Figure 159]

- Figure 1. Without the projection-based consistency loss, we observe the artifacts caused by outliers (left). The loss encourages the outlier towards the surface, thus diminishing the artifacts.

###### 3.4. Shiny Blender dataset

###### 3.1. Extra ablation studies

We present the decomposition of material and geometry with the relighting results on the Shiny Blender dataset [8] in Fig. 5. Due to no relighting ground truth, we show the environment map for shading as the reference. Besides, we present more qualitative comparisons of normal in Fig. 6. Due to the robustness of our method, our method provides high-quality relighting results for reflective surfaces. Additionally, We present the evaluation of surface quality in terms of Chamber distance (CD) along with NVS results in Tab. 7. Our method outperforms other methods in terms of CD, while GS-ROR achieves a higher quality than ours in some cases. The main reason is that our backbone is 2DGS

We ablate our hyperparameter choice and loss usage in this part, including the threshold in the projection-based loss, the impact of the median and the mask loss.

The choice of the threshold ε. A large threshold forces Gaussians onto wrong surfaces, while a small one causes most Gaussians to be excluded, resulting in decreased performance. Therefore, we choose to set ε = 0.1 for all scenes, which reveals its generalization ability and yields the best performance in Tab. 5.

The impact of the median loss. We use the median loss to encourage γ towards γm, instead of directly setting γ = γm, because γm is used to give the direction of optimization and not the target. Besides, the median loss enables the further optimization of the transformation. Our experiment also proves that the median loss promotes the convergence

1NeRO and TensoSDF use the Cycles Render Engine in Blender to provide relighting results, so the FPS depends on samples per pixel, which is 1024 following the NeRO setting.

while theirs is 3DGS. Although 2DGS demonstrates superior reconstruction quality, its NVS quality is lower compared to that of 3DGS.

###### 3.5. TensoIR synthetic dataset

We present the decomposition of material and geometry on the TensoIR synthetic dataset [2] in Fig. 7. Besides, we present more qualitative comparisons of normal in Fig. 8. Our method provides a reasonable decomposition for diffuse objects and realistic relighting results. The quantitative comparison between our method and NeRF-based methods is shown in Tab. 3, and the qualitative comparison between our method and NeRF-based methods is Fig. 9. Our method is numerically and visually competitive for diffuse object relighting while maintaining efficiency.

###### 3.6. NeILF++ dataset

We present more comparisons between our method and other Gaussian-based inverse rendering methods on the NeILF++ [9] dataset, revealing the robustness of our method for real-world objects. As shown in Fig. 10, our method provides realistic relighting results and detailed normal for diverse real objects.

#### References

- [1] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and Yuexin Ma. GaussianShader: 3D Gaussian splatting with shading functions for reflective surfaces. In CVPR, pages 5322–5332, New York, NY, USA,

2024. IEEE. 1

- [2] Haian Jin, Isabella Liu, Peijia Xu, Xiaoshuai Zhang, Songfang Han, Sai Bi, Xiaowei Zhou, Zexiang Xu, and Hao Su. TensoIR: Tensorial inverse rendering. In CVPR, pages 165– 174, New York, NY, USA, 2023. IEEE. 2, 4
- [3] Brian Karis. Real shading in unreal engine 4. Proc. Physically Based Shading Theory Practice, 2013. 1
- [4] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3D Gaussian splatting for real-time radiance field rendering. ACM TOG, 42(4):139–1, 2023. 1
- [5] Jia Li, Lu Wang, Lei Zhang, and Beibei Wang. TensoSDF: Roughness-aware tensorial representation for robust geometry and material reconstruction. ACM TOG, 43(4), 2024. 2
- [6] Yuan Liu, Peng Wang, Cheng Lin, Xiaoxiao Long, Jiepeng Wang, Lingjie Liu, Taku Komura, and Wenping Wang. NeRO: Neural geometry and BRDF reconstruction of reflective objects from multiview images. ACM TOG, 42(4), 2023. 2, 3
- [7] Stephen McAuley, Stephen Hill, Adam Martinez, Ryusuke Villemin, Matt Pettineo, Dimitar Lazarov, David Neubelt, Brian Karis, Christophe Hery, Naty Hoffman, and Hakan Zap Andersson. Physically based shading in theory and practice. In ACM SIGGRAPH 2013 Courses, New York, NY, USA, 2013. Association for Computing Machinery. 1

- [8] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T. Barron, and Pratul P. Srinivasan. Ref-NeRF: Structured view-dependent appearance for neural radiance fields. In CVPR, pages 5491–5500, New York, NY, USA,

- 2022. IEEE. 3

[9] Jingyang Zhang, Yao Yao, Shiwei Li, Jingbo Liu, Tian Fang, David McKinnon, Yanghai Tsin, and Long Quan. NeILF++: Inter-reflectable light fields for geometry and material estimation. In ICCV, pages 3601–3610, New York, NY, USA,

- 2023. IEEE. 4

- [10] Yuanqing Zhang, Jiaming Sun, Xingyi He, Huan Fu, Rongfei Jia, and Xiaowei Zhou. Modeling indirect illumination for inverse rendering. In CVPR, pages 18643–18652, New York, NY, USA, 2022. IEEE. 2

Rendering Ground Truth Normal Albedo Roughness Metallicity

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

- Figure 2. Decomposed maps of our method on the Glossy Blender dataset. Our method can provide a reasonable decomposition for reflective surfaces.

GS-IR R3DG GShader GS-ROR Ours Ground Truth

|[Figure 184]|
|---|

|[Figure 185]|
|---|

|[Figure 186]|
|---|

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

- 17.53 / 0.7840 15.22 / 0.7895 11.91 / 0.7659 28.18 / 0.9368 29.31 / 0.9457
- 18.10 / 0.7919 18.37 / 0.8016 19.34 / 0.8309 20.26 / 0.8455 21.54 / 0.8706

|[Figure 196]|
|---|

|[Figure 197]|
|---|

|[Figure 198]|
|---|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

|[Figure 207]|
|---|

|[Figure 208]|
|---|

|[Figure 209]|
|---|

|[Figure 210]|
|---|

|[Figure 211]|
|---|

|[Figure 212]|
|---|

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

|[Figure 219]|
|---|

|[Figure 220]|
|---|

|[Figure 221]|
|---|

|[Figure 222]|
|---|

|[Figure 223]|
|---|

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

- Figure 3. The qualitative comparison with Gaussian-based methods in terms of relighting results and normal on the Glossy Blender dataset. Our method provides high-quality relighting results for reflective surfaces. The PSNR/SSIM of relighting results under the current view are below the images.

TensoIR MII NeRO TensoSDF Ours Ground Truth

|[Figure 230]|
|---|

|[Figure 231]|
|---|

|[Figure 232]|
|---|

|[Figure 233]|
|---|

|[Figure 234]|
|---|

|[Figure 235]|
|---|

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

9.28 / 0.2265 19.44 / 0.8186 16.32 / 0.7531 19.96 / 0.8585 21.54 / 0.8706

|[Figure 242]|
|---|

|[Figure 243]|
|---|

|[Figure 244]|
|---|

|[Figure 245]|
|---|

|[Figure 246]|
|---|

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

|[Figure 253]|
|---|

|[Figure 254]|
|---|

|[Figure 255]<br><br>[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

|[Figure 259]|
|---|

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

5.54 / 0.1050 18.69 / 0.8778 25.08 / 0.9421 27.50 / 0.9525 22.87 / 0.9361

|[Figure 266]|
|---|

|[Figure 267]|
|---|

|[Figure 268]|
|---|

|[Figure 269]|
|---|

|[Figure 270]|
|---|

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

|[Figure 277]|
|---|

|[Figure 278]|
|---|

|[Figure 279]|
|---|

|[Figure 280]|
|---|

|[Figure 281]|
|---|

|[Figure 282]|
|---|

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

9.30 / 0.2184 14.08 / 0.8420 27.98 / 0.9574 23.60 / 0.9399 27.00 / 0.9437

|[Figure 289]|
|---|

|[Figure 290]|
|---|

|[Figure 291]|
|---|

|[Figure 292]|
|---|

|[Figure 293]|
|---|

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

- Figure 4. The qualitative comparison with NeRF-based methods in terms of relighting results and normal on the Glossy Blender dataset. Our method is competitive with these methods and outperforms in the detailed regions. The PSNR/SSIM of relighting results under the current view are below the images.

Env. Map Rendering Normal Albedo Roughness Metallicity

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

- Figure 5. Decomposed maps of our method on the Shiny Blender dataset. The relighting ground truth is unavailable on the dataset, so we provide the environment map for shading as a reference.

[Figure 318]

[Figure 319]

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

|[Figure 328]|
|---|

|[Figure 329]|
|---|

R3DG GS-IR GShader GS-ROR Ours Ground Truth

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

|[Figure 336]|
|---|

|[Figure 337]|
|---|

|[Figure 338]|
|---|

|[Figure 339]|
|---|

|[Figure 340]|
|---|

|[Figure 341]|
|---|

[Figure 342]

[Figure 343]

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

|[Figure 352]|
|---|

|[Figure 353]|
|---|

7.46 12.61 6.48 4.56 4.44

14.22 13.19 14.78 18.28 9.57

55.96 50.22 9.38 0.95 1.26

- Figure 6. The qualitative comparison with Gaussian-based methods in terms of normal on the Shiny Blender dataset. The MAE under the current view is above the visualization. Our method can provide reasonable normal for diverse objects.

Rendering Ground Truth Normal Normal GT Albedo Roughness

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

- Figure 7. Decomposed maps of our method on the TensoIR synthetic dataset. Our method can provide a reasonable decomposition for diffuse objects.

GS-IR R3DG GShader GS-ROR Ours Ground Truth

|[Figure 372]|
|---|

|[Figure 373]|
|---|

|[Figure 374]|
|---|

|[Figure 375]|
|---|

|[Figure 376]|
|---|

|[Figure 377]|
|---|

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

31.18 / 0.9341 30.08 / 0.9481 25.37 / 0.9265 33.49 / 0.9702 33.99 / 0.9706

|[Figure 384]|
|---|

|[Figure 385]|
|---|

|[Figure 386]|
|---|

|[Figure 387]|
|---|

|[Figure 388]|
|---|

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

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

|[Figure 400]|
|---|

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

22.49 / 0.8940 21.69 / 0.8533 11.70 / 0.7507 24.55 / 0.8937 25.21 / 0.9002

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

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

- Figure 8. The qualitative comparison with Gaussian-based methods in terms of relighting results and normal on the TensoIR synthetic dataset. Our method can provide robust normal for diffuse objects. The PSNR/SSIM of relighting results under the current view are below the images.

TensoIR MII NeRO TensoSDF Ours Ground Truth

|[Figure 418]|
|---|

|[Figure 419]|
|---|

|[Figure 420]|
|---|

|[Figure 421]|
|---|

|[Figure 422]|
|---|

|[Figure 423]|
|---|

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

35.43 / 0.9723 21.71 / 0.8987 23.43 / 0.9157 23.49 / 0.9171 33.99 / 0.9706

|[Figure 430]|
|---|

|[Figure 431]|
|---|

|[Figure 432]|
|---|

|[Figure 433]|
|---|

|[Figure 434]|
|---|

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

|[Figure 441]|
|---|

|[Figure 442]|
|---|

|[Figure 443]|
|---|

|[Figure 444]|
|---|

|[Figure 445]|
|---|

|[Figure 446]|
|---|

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

25.09 / 0.9287 19.05 / 0.9189 20.78 / 0.9303 33.49 / 0.9702 27.19 / 0.9390

|[Figure 453]|
|---|

|[Figure 454]|
|---|

|[Figure 455]|
|---|

|[Figure 456]|
|---|

|[Figure 457]|
|---|

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

- Figure 9. The qualitative comparison with NeRF-based methods in terms of relighting results and normal on the TensoIR synthetic dataset. Our method is competitive while maintaining high efficiency. The PSNR/SSIM of relighting results under the current view are below the images.

Ref. Image & Env. Map GShader GS-ROR Ours

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

|[Figure 468]|
|---|

|[Figure 469]|
|---|

|[Figure 470]|
|---|

[Figure 471]

[Figure 472]

[Figure 473]

|[Figure 474]|
|---|

|[Figure 475]|
|---|

|[Figure 476]|
|---|

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

|[Figure 481]|
|---|

|[Figure 482]|
|---|

|[Figure 483]|
|---|

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

|[Figure 488]|
|---|

|[Figure 489]|
|---|

|[Figure 490]|
|---|

[Figure 491]

- Figure 10. The qualitative comparison in terms of relighting results and normal on the NeILF++ dataset. Our method ensures the smoothness of normal while preserving details for real data, thus providing more realistic relighting results.

