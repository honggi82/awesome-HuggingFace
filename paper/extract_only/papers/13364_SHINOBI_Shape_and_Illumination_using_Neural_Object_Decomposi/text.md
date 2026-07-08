## SHINOBI: Shape and Illumination using Neural Object Decomposition via BRDF Optimization In-the-wild

Andreas Engelhardt† University of T¨ubingen

Amit Raj Google Research

Mark Boss∗ Unity

Yunzhi Zhang† Stanford University

# arXiv:2401.10171v2[cs.CV]29Mar2024

Abhishek Kar Google Research

Yuanzhen Li Google Research

Deqing Sun Google Research

Ricardo Martin Brualla Google Research

Jonathan T. Barron Google Research

Hendrik P. A. Lensch University of T¨ubingen

Varun Jampani∗ Google Research

Normals Illumination

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

…

[Figure 12]

[Figure 13]

Neural Volume & Camera Parameters

Basecolor Metallic Roughness Decomposition

In-the-wild images

Figure 1. Object reconstruction using SHINOBI. SHINOBI decomposes challenging in-the-wild image collections into shape, material and illumination using a neural field representation while also optimizing camera parameters. Visit the project page at Project page: https://shinobi.aengelhardt.com

### Abstract

(i.e. material) we jointly optimize BRDF and illumination together with the object’s shape. Our method is class-agnostic and works on in-the-wild image collections of objects to produce relightable 3D assets for several use cases such as AR/VR, movies, games, etc.

We present SHINOBI, an end-to-end framework for the reconstruction of shape, material, and illumination from object images captured with varying lighting, pose, and background. Inverse rendering of an object based on unconstrained image collections is a long-standing challenge in computer vision and graphics and requires a joint optimization over shape, radiance, and pose. We show that an implicit shape representation based on a multi-resolution hash encoding enables faster and robust shape reconstruction with joint camera alignment optimization that outperforms prior work. Further, to enable the editing of illumination and object reflectance

### 1. Introduction

We present a category-agnostic technique to jointly reconstruct 3D shape and material properties of objects from unconstrained in-the-wild image collections. This data regime poses multiple challenges as images are captured in different environments using a variety of devices resulting in varying backgrounds, illumination, camera poses, and intrinsics. In addition, camera baselines tend to be large. Fig. 1 (left) shows examples from an input image set. Many graphics

*Current affiliation is Stability AI. †Work done during a Student Researcher position at Google.

applications in AR/VR, games, and movies depend on highquality 3D assets of real-world objects. Physically based materials are essential to integrate objects into new environments. The conventional acquisition involves laborious tasks like 3D modeling, texture painting, and light calibration or use controlled setups [6, 53] that are hard to scale. It is easier to obtain casually captured images from smartphones or image collections from the internet for a large number of objects.

Conventional structure-from-motion techniques like COLMAP [65, 66] fail to reconstruct image collections under these challenging circumstances [14, 33]. Despite constraining the correspondences to lie within object bounds, specifically in the context of the NAVI [33] in-the-wild scenes, less than half of the views are registered on average with half the scenes failing completely. Consequently, we observe that camera pose optimization has the largest impact on the reconstruction quality in this setting. Many existing works on shape and material estimation [6, 13, 69, 80, 82, 87, 90] assume constant camera intrinsics and initialization of camera poses close to the true poses. We support 360° multiview data with a rough quadrant-based pose initialization with poses potentially far from the ground truth, as in SAMURAI [14] and NeRS [84]. For challenging data this can be annotated in only a few minutes per image collection. Even though in SAMURAI [14], camera poses can be initialized from very coarse directions slight offsets often lead to overly smooth textures and shapes in the final reconstructions. Further, existing methods for material decomposition with camera pose optimization are slow, often running more than 12 hours on a single object [14, 37]. In contrast, we propose a pipeline based on multiresolution hash grids [51] which allows us to process more rays in a shorter time during optimization. Using this advantage we are able to improve reconstruction quality compared to SAMURAI while still keeping a competitive run-time (Tab. 1).

Naive integration of multi-resolution hash grids is not well suited to camera pose estimation due to discontinuities in the gradients with respect to the input positions. We propose several components that work together to stabilize the camera pose optimization and encourage sharp features. The key distinguishing features of SHINOBI include:

- • Hybrid Multiresolution Hash Encoding with level annealing. We combine the multiresolution hash-based encoding [51] with regular Fourier feature transformation of the input coordinates to regularize the low-frequency gradient propagation. This makes the optimization significantly more robust while only adding a small overhead. A similar approach has been recently proposed by Zhu et al. [92] for a different task. We show that it is also beneficial for camera pose optimization.
- • Camera multiplex constraint. We modify the camera parameterization of SAMURAI to avoid over-

- parameterization of the camera rotations. Furthermore, we constrain the camera optimization with a projectionbased loss to enforce consistency over the camera proposals inside a multiplex which further helps to smooth the optimization in the initial phase.
- • Per-view importance weighting. We propose a per-view importance weighting to leverage the important observation that some views are more useful for optimization than others. Specifically, we use well-working cameras to anchor the reconstruction during the optimization.
- • Patch-based alignment losses. SHINOBI proposes a novel patch level loss to aid in camera alignment and additionally introduces a silhouette loss inspired by Lensch et al. [38] for better image to 3D alignment.

Experiments on NAVI [33] in-the-wild datasets demonstrate better view synthesis and relighting results with SHINOBI compared to existing works with a reduced run-time. Compared to SAMURAI the results look sharper and the average runtime is cut in half. Fig. 1 (right) shows some sample application results with 3D assets generated by SHINOBI . Our representation enables editing of appearance parameters, illumination and based on the mesh extraction also shape, facilitating various tasks in a downstream graphics pipeline.

### 2. Related works

Neural fields have emerged as a popular technique of late to encode spatial information in the network weights of e.g. an MLP, which can be retrieved by simply querying the coordinates [16, 49, 57, 70]. Works like NeRF [50] leverage this neural volume rendering to achieve photo-realistic view synthesis results with view-dependent appearance variations. Rapid research in neural fields followed, which alternated the surface representations [56, 72, 75, 76, 78, 81], allowed reconstruction from sparse data [8, 32, 46, 55, 61, 71, 79], enabled extraction of 3D geometry and materials [12, 37, 52, 84], or enabled relighting of scenes [5, 12–14, 43, 47, 82]. However, most prior works rely on pose information extracted from COLMAP [65, 66], which can be inaccurate or completely fail in complex settings or sparse data regimes. SHINOBI is independent of any pose reconstruction that relies on feature matching and robust to very coarse initialization.

Instant Neural Graphics Primitives (I-NGP) [51] is a popular geometric representation that enables fast optimization with improved memory utilization by using an encoding scheme based on multi-resolution hash tables. Despite the improvement in speed, I-NGP suffers from discontinuous and oscillating gradient flow through the hash-based encoding, which complicates camera pose optimization [31, 92, 92]. To enable reconstruction with camera pose fine-tuning using hash grids, Heo et al. [31] propose a modification to the interpolation weighting, BAA-NGP [45] dynamically replicates low-resolution features and CAMP [59] pairs a robust sampling scheme [4] with camera preconditioning. These

|[Figure 14]|
|---|

Patch-based Losses

|[Figure 15]|
|---|

Annealed Hybrid Encoding

[Figure 16]

: Optimizable Prameters

|Illumination zj| |
|---|---|
| | |
| | |

###### H(x)

Annealing

Pjs

Network

|Intrinsics fˆj|
|---|

|PIL-Renderer| |
|---|---|
| | |

PIL-Renderer

||MLP|
|---|
<br><br>MLP<br><br>|MLP|
|---|
<br><br>MLP| | |
|---|---|---|
| | | |
| | | |

Color cˆj

BRDFs bt

| | |
|---|---|
| | |

MLP

MLP

|Extrinsics pjeye, djϕθ, djup|
|---|

Cjs

MLP

Densities σt

γ(x)

Positions xt

Direction

d

- Figure 2. The SHINOBI pipeline. Two resolution annealed encoding branches, the multiresolution hash grid H(x) and the Fourier embedding γ(x) are used to learn a neural volume conditioned on the input coordinates. This enables robust optimization of camera parameters jointly with the shape, material and illumination.

methods however are sensitive to camera initialization and lighting conditions. In contrast to these works, SHINOBI is able to reconstruct consistent objects from images captured under varying illuminations and backgrounds besides supporting coarser poses.

Joint camera and shape estimation is a highly ambiguous task, traditionally relying on accurate poses for precise shape reconstruction and vice versa. Often techniques rely on correspondences across images to estimate camera poses [65, 66]. Recent approaches integrate camera calibration with neural volume training; SCNeRF [34] and NopeNeRF [8] use correspondences and monocular depth images, respectively. Other recent methods rely on rough initialization of the camera, global alignment, or a template shape for joint optimization [15, 44, 77, 84]. Other methods use transformer-based models[23] to predict the initial pose from image collection [68, 85]. In comparison, SHINOBI works on unconstrained image collections, including various camera parameters and object environments, where existing methods struggle to generalize or require additional input data like depth.

BRDF and illumination estimation is a challenging and ambiguous problem. Casual BRDF estimation enables on-site material acquisition with simple cameras and a co-located camera flash. These techniques often constrain the problem to planar surfaces with either a single shot [2, 9, 20, 30, 40, 62], few-shot [2] or multi-shot [3, 10, 21, 22, 26] captures. Casual capture can also be extended to a joint BRDF and shape reconstruction [5–7, 11, 35, 53, 62, 83], even on entire scenes [41, 67]. Most of these methods, however, require a known active illumination. Recovering a BRDF under unknown passive illumination is significantly more challenging as it requires disentangling the BRDF from the illumination. Recently, neural field-based decomposition achieved decomposition of scenes under varying illumination [12, 13] or

fixed illumination [43, 86, 87, 89, 90]. IntrinsicNeRF [82] extends decomposition to larger scenes at the cost of a simplified reflectance model. However, all these approaches require known, near-perfect camera poses, whereas SHINOBI can work with unposed image collection to recover per-image illumination.

### 3. Method

The aim of SHINOBI is to convert 2D image collections into a 3D representation with minimal manual work. The representation includes shape, material parameters and per-view illumination, allowing for view synthesis with relighting.

Problem setup. We define in-the-wild data as a collection of q images Cj ∈ Rs

j×3;j ∈ {1,...,q} that show the same object captured with different backgrounds, illuminations and cameras with potentially varying resolutions sj. In addition, we assume a rough camera initialization. For our experiments we annotate camera pose quadrants as in SAMURAI [14]. Foreground masks can be added if available or automatically generated and might be imperfect at this point. At each point x ∈ R3 in the neural volume V, we estimate the BRDF parameters for the Cook-Torrance model [19] b ∈ R5 (basecolor bc ∈ R3, metallic bm ∈ R, roughness br ∈ R), unit-length surface normal n ∈ R3 and volume density σ ∈ R (Fig. 1). To enable the decomposition we also estimate the latent per-image illumination vectors zjl ∈ R128;j ∈ {1,...,q} [13]. Furthermore, we estimate per-image camera poses and intrinsics. Next, we provide a brief overview of prerequisites: NeRF [50], InstantNGP [51] and SAMURAI [14].

Coordinate-based MLPs and NeRF [50] uses a dense neural network to model a continuous function that takes 3D location x ∈ R3 and view direction d ∈ R3 and outputs a view-dependent output color c ∈ R3 and volume density σ ∈ R. Mildenhall et al. [50] overcome the spectral bias of

the MLPs by transforming the input coordinates by a second function; A frequency encoding γ that maps from R to R2L [50, 70]:

γ(x) = (sin(20πx),cos(20πx),

...,sin(2L−1πx),cos(2L−1πx))

(1)

InstantNGP [51] speed up the NeRF optimization drastically by replacing the MLP-based volume representation by a multiresolution voxel hash grid that is tailored to current GPU hardware. For a hash-size T, grid vertices are indexed

d

by a spatial hash function h(x) =

xiπi mod T using large unique prime numbers πi [51]. At each voxel vertex a d-dimensional embedding is optimized. Instead of the Fourier embedding, the 3D coordinates x are directly used to tri-linearly interpolate between neighboring vertices at each level. The results are concatenated and fed to a MLP to decode the representation. We denote the full encoding function including interpolation and concatenation as H(x). Brief overview of SAMURAI. SAMURAI is a method for joint optimization of 3D shape, BRDF, per-image camera parameters, and illuminations for a given in-the-wild image collection. SAMURAI [14] follows the NeRF idea outlined above but uses the Neural-PIL [13] method for physicallybased differentiable rendering. It takes 3D locations as input and outputs volume density and BRDF parameters. An additional GLO (generative latent optimization) embedding models the changes in appearances (due to different illuminations) across images. Neural-PIL [13] introduced the use of per-image latent illumination embedding zjl and a specialized illumination pre-integration (PIL) network for fast rendering, which we refer to as ‘PIL rendering’. Neural-PIL optimizes a per-image embedding to model image-specific illumination. The rendered output color cˆ is equivalent to NeRF’s output c, but due to the explicit BRDF decomposition and illumination modeling, it enables relighting and material editing. To address the unavailability of accurate camera parameters for in-the-wild images, SAMURAI jointly optimizes camera extrinsics and per-view intrinsics from a very coarse initialization. In addition to a coarse-to-fine annealing [44], this is achieved with a multiplexed optimization scheme where multiple camera proposals per view are kept and weighted according to their performance on the loss over time.

i=1

#### 3.1. SHINOBI Optimization with Hash Encoding

We identify misaligned and inconsistent camera poses as the main limiting factor for in-the-wild reconstructions. Joint shape and camera optimization is a severely underdetermined problem. Reconstruction is typically slow and often lacks high-frequency detail in textures and shape. Multiresolution hash grids from Instant-NGP [51] have the potential to speed up the reconstruction while simultaneously allowing

for larger ray counts to be processed and thereby improving visual quality and alignment (see Tab. 1). However, the naive replacement of the point encoding with Hash grids reduces the reconstruction quality and robustness of the joint camera and shape optimization.

Hash grids adapt to individual views faster resulting in a noisy shape in the presence of misaligned cameras. As reported previously [31, 42, 45, 92] multi-resolution hash grids with the default linear interpolation backpropagate noisy and discontinuous gradients with respect to the input position. Additionally, the coarse-to-fine scheme from BARF [44] often used for camera fine-tuning cannot be directly transferred to hash grids. Therefore, we propose an approach that makes use of a camera multiplex, adds additional geometrical constraints, and a new encoding scheme to be able to improve both reconstruction speed and quality. Next, we explain each of the components in detail.

Architecture overview. A high-level overview of the SHINOBI architecture is shown in Fig. 2, which follows the skeleton of SAMURAI [14] with the ‘PIL renderer’ [13]. However, we map the input coordinates x using a new hybrid encoding. The combined embedding is processed by a small MLP like in I-NGP [51] to predict the density σ, and the view and appearance conditioned radiance for a given image patch. We also predict a regular direction-dependent radiance c˜to stabilize the early training stages as in [12, 14]. The BRDF decoder operates as in SAMURAI [14], expanding the feature representation to the BRDF (base color, metallic, roughness). Per sample, we estimate normal direction from the first order derivative of the density w.r.t. the input position

∂σ ∂x. From there the volumetric rendering from NeRF [50] is performed and the shading for the given pixel coordinate

is determined using BRDF, normals and the pre-integrated illumination estimated by the NeuralPIL network. See supplementary material for further details on the architecture.

Camera pose initialization and parameterization. Camera pose optimization is a highly non-convex problem and tends to quickly get stuck in local minima. Our initial camera poses are much noisier and feature larger distances between initial and true poses compared to many related works [37, 77]. To combat this, we assume a rough initialization in the form of camera pose quadrants in line with SAMURAI [14] and NeRS [84]. We use a ‘lookat + direction’ representation for the camera parameters, storing initial values and offsets for an eye position peye ∈ R3, lookat direction ∆dϕθ ∈ R2. and up rotation angle dup ∈ R as well as the focal length f ∈ R per camera. We notice that this removes the overparameterization regarding the rotation component encoded in eye and center position of the regular ‘lookat’ parameterization. This formulation performs best in our setting also compared to other recently proposed representations [59, 91].

Hybrid positional encoding. We use a hash grid hybrid as coordinate encoding to improve the gradient flow w.r.t. the in-

[Figure 17]

the multiplex to the ones originally rendered at Θ1...m−1.

m−1

Lmultiplex =

Limage(ci,FcˆV (Pi,0(Xi,Di,Θi,Θ0)))

i=1

+ Lmask(αi,FαV (Pi,0(Xi,Di,Θi,Θ0))) (2)

where Pi,0 is the perspective warp from image coordinates in camera i to the reference camera. FV is the rendering function connected to the neural field outputting color cˆand mask value α, respectively. This regularization comes roughly at the cost of adding a camera to the multiplex. Subsampling of Xi can decrease the memory footprint if needed. Limage and Lmask are the optimization losses active at the time as outlined in Sec. 3.2. Naturally, this component is only active while there are multiple cameras rendered during the first part of the overall schedule. Used as an additional loss it turns out to be surprisingly effective in constraining the camera optimization and therefore increasing the robustness of the overall optimization. Essentially, we are enforcing a consistent surface to be generated and smooth the optimization landscape around an initial camera pose.

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

- Figure 3. Constrained camera multiplex. We optimize multiple camera proposals per image and weight the contribution to the reconstruction according to a camera’s performance on the loss. Between cameras of a multiplex we add a projection based regularization: Points from all members are projected into the currently best camera and then compared against a new render to enforce a consistent geometry.

View importance scaling of input images. Not every input might contribute to the reconstruction in the same way and individual views that are not aligned with the current 3D shape might have a negative impact on the overall optimization progress. To improve high-frequency detail in the reconstruction we reduce the impact of potentially misaligned cameras while anchoring the optimization using cameras that work well given the loss. We keep a circular buffer of around 1000 elements with the recent per-image losses. Like in SAMURAI, this is used to re-weigh images in the given collection

put coordinates x. A Fourier-based coordinate mapping γ(x) followed by a small MLP generates a base embedding that is concatenated with the output of the multiresolution hash grid H(x) resulting in the following formulation of the neural volume F⊕ ((H(x),γ(x))). On γ, we apply BARF’s [44] Fourier annealing. Similarly, we progressively add resolution levels to the hash grid encoding. Starting with only the features from a low resolution dense grid we increase the weights of the higher resolution levels gradually over time (cf. [42, 45].

according to: L(networkj) = spj L(networkj) , where

µl − (Lmask(j) + L(imagej) ) σl

+ 1, 1 , (3)

spj = max tanh

with the mean µl and standard deviation σl of the loss buffer. This limits the influence of badly aligned camera poses on the shape reconstruction. In addition, we also apply an importance weighting on Lcamera that reduces the gradient magnitude for views that are performing well given the loss history.

Camera multiplexes. An effective way to reduce the chance of camera pose optimization to be stuck in local minima is the camera multiplex [14, 27]. For each image, m cameras are jittered around the initial camera and simultaneously optimized. Over time the worst performing camera is repeatedly faded out until m = 1. This process is visualized in Fig. 3. Since we render multiple proposals for a given image anyway, we see an opportunity to further constrain the optimization using projective geometry. Specifically, we project the 2D point sets Xi rendered by the m−1 members into the currently highest ranking camera Θ0 of the multiplex using the estimated depth Di from the volumetric rendering. Then we render the projected coordinates using Θ0 and compare the rendered color ci and alpha values αi of all cameras in

Specifically, at step t we compute: L(cameraj) = sqj,t L(cameraj) , with

µl − (Lmask(j) + L(imagej) ) σl

+1, 1

sqj,t =sqj,t−1λp max tanh

+(1 − λp)sqj,t−1 (4)

In practice, we set the hyperparameter λp to 0.05.

#### 3.2. Losses and Optimization

Multiscale patch loss. After a short initial phase of random ray sampling, we render randomly sampled patches of size 16x16 to 32x32. The goal is to constrain the updates and

[Figure 25]

[Figure 26]

[Figure 27]

(a) Reference silhouette (b) Rendered silhouette (c) Loss map

- Figure 4. Our silhouette based alignment loss penalizes the unaligned pixels given a reference and the rendered gray scale masks.

especially the alignment to be consistent on local neighborhoods. Therefore, we add a multi-scale patch loss on the rendered color cˆwhich computes a Charbonnier loss at four different resolution levels, by simple bilinear resampling. We weigh each level to compensate for the different pixel counts and enforce the low-resolution version to align first.

Mask losses. We add a silhouette loss LSilhouette whenever patch-based sampling is active. Here, we penalize the area between the two silhouettes which can be interpreted as the result of an xor operation on the rendered and input mask [38]. Both masks are filtered using a Gaussian blur where the radius is heuristically chosen based on the patch size. Fig. 4 visualizes how the loss helps with the alignment task. We combine this loss with a regular binary-cross-entropy loss on the mask value as well as a loss enforcing a transparent background.

Regularization losses. To regularize the hash grid encoding we apply a normalized weight decay as proposed in [4] to put a higher penalty on coarser grid levels compared to naive weight decay. Additionally, we apply regularization to the camera poses and normal output. Refer to the supplements for details and the hyperparameters used.

Optimization. In total, we use three optimizers: One ADAM [36] optimizer for the networks, hash grid embeddings and cameras, respectively. The learning rate is decayed exponentially on all optimizers. In addition to the camera representation and constraints mentioned above we use ADAM with the β1 value reduced to 0.2 to smooth out the noise in the camera updates. The learning rate is tuned between 1e-3 to 2e-3 depending on scene size. Render resolution is continuously increased over the first half of the optimization while the number of active multiplex cameras is reduced. The direct color optimization is faded to the BRDF optimization and the encoding annealing is performed over the first third of the optimization. Focal length updates and the view importance weighting are delayed until an initial shape has been formed. See the supplementary material for a detailed description and visualization of the optimization scheduling.

Implementation. We implement the multi-resolution hash grid encoding as a custom CUDA extension for Tensorflow [1]. The implementation roughly follows the official CUDA implementation [51]. We enable first- and secondorder gradients for the encoding to allow for computing

Method PSNR↑ SSIM↑ LPIPS↓ Runtime

SC ∼ SC SC ∼ SC SC ∼ SC NeROIC [37] 22.75 21.31 0.91 0.90 0.0984 0.0845 18 hours (4 GPUs) NeRS [84] 17.92 18.02 0.92 0.93 0.114 0.1098 3 hours (1 GPU) SAMURAI [14] 25.34 24.61 0.92 0.91 0.0958 0.1054 12 hours (1 GPU) SHINOBI 27.69 27.79 0.94 0.94 0.0607 0.0578 4 hours (1 GPU)

Table 1. Metrics for view synthesis on NAVI. View synthesis metrics are computed over two subsets from all wild-sets depending on the success of COLMAP (SC / ∼ SC). Rendering quality is evaluated on a holdout set of test views. We initialize with the GT poses provided by NAVI [33].

analytical surface normals. The remaining components are implemented in Tensorflow.

Method PSNR↑ SSIM↑ Transl.↓ Rot. °↓

w/o Multiplex Consistency Loss 25.80 0.93 0.29 23.12 w/o Per View Importance 22.43 0.90 0.36 35.10 w/o Coarse-to-fine (annealing) 21.47 0.90 0.37 30.44 w/o Hybrid Encoding 25.31 0.93 0.30 23.33 w/o Patch-based Training 20.60 0.89 0.45 41.30 Full 25.87 0.93 0.30 22.90

Table 3. Ablation study. Ablating components of our framework results in worse view synthesis and relighting results (averaged over ”Keywest” and ”School Bus” scenes from NAVI) demonstrating their importance.

### 4. Experiments

Dataset For evaluations, we use the in-the-wild collections from the NAVI dataset [33] which feature objects captured in diverse environments using multiple mobile devices. Highquality annotated camera poses allow us to ablate and perform quantitative evaluation of our pose estimation.

Baselines. The closest prior work that can tackle our task outline in Sec. 3 is SAMURAI [14] on which our method is based. We compare against SAMURAI as a baseline and also conduct experiments using NeROIC [37], GNeRF [48], and a modified version of NeRS [84] (details in the supplement). For experiments on joint shape and pose estimation, we use the same quadrant-based pose initialization for NeRS, SAMURAI and SHINOBI (ours); and we use the the methods’ default pose initializations for NeROIC (COLMAP) and GNeRF (Random).

Evaluation. We use two strategies for evaluation. First, the standard novel view synthesis metrics using the learned volumes that measure PSNR, SSIM, and LPIPS [88] scores on held-out test images. Second, to evaluate camera poses w.r.t GT poses, we use Procrustes analysis [28] to align the cameras and then compute the mean absolute rotation and translation differences in camera pose estimations for all available views. For evaluation purposes, we optimize the cameras and illuminations on the test images but do not al-

Method Pose Init PSNR↑ SSIM↑ LPIPS↓ Translation↓ Rotation ◦ ↓

SC ∼ SC SC ∼ SC SC ∼ SC SC ∼ SC SC ∼ SC GNeRF [48] Random 8.30 6.25 0.64 0.63 0.52 0.57 1.02± 0.16 1.04± 0.09 93.15± 26.54 80.22± 27.64 NeROIC [37] COLMAP 19.77 - 0.88 - 0.150 - 0.09± 0.12 - 42.11± 17.19 NeRS [84] Directions 18.67 18.66 0.92 0.93 0.108 0.107 0.49± 0.21 0.52± 0.19 122.41± 10.61 123.63± 8.80 SAMURAI [14] Directions 25.34 24.61 0.92 0.91 0.096 0.105 0.24± 0.17 0.35± 0.24 26.16± 22.72 36.59± 29.98

SHINOBI Directions 25.15 24.77 0.92 0.92 0.090 0.095 0.250± 0.085 0.28± 0.09 22.84± 16.19 33.00± 19.97

Table 2. Metrics for 3D shape and pose on NAVI. View synthesis and pose metrics over two subsets from all wild-sets depending on the success of COLMAP (SC / ∼ SC). Rendering quality is evaluated on a holdout set of test views that are aligned as part of the optimization without contributing to the shape recovery. We include GNeRF as a separate baseline although this method is not designed for multi-illumination data. We report metrics with the methods’ default camera initialization and evaluate against the annotation provided in NAVI [33].

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Input Basecolor Metallic Roughness Normal Illumination Re-render

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

SAMURAIOurs

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

- Figure 5. Comparison with SAMURAI decomposition for joint pose and object reconstruction. Due to the improved alignment and representation higher frequency details are reconstructed in shape and the BRDF components compared to SAMURAI. Notice the improved texture detail and silhouettes of ours. Both methods optimize camera poses jointly initialized from rough quadrants.

low the test images to affect the other network parts or hash grid embedding. For a fair comparison, we use the ground truth masks as input to all methods although our method also includes functionality to automatically generate segmentation masks. We run experiments on a single Nvidia A100 or V100 GPU per scene.

in SHINOBI obtaining better LPIPS perceptual metrics compared to SAMURAI. The on-par mean PSNR compared to SAMURAI mostly stems from individual test cameras not being aligned properly. This also happens for other methods but seems to be emphasized by the faster optimization scheduling in SHINOBI. NeROIC can also achieve good results if camera poses are close to the ground truth but fails for many scenes where a COLMAP-based initialization is not possible. NeRS also succeeds in reconstructing all scenes. However, it achieves lower-quality camera alignments. Fig. 6 visually compares view synthesis results from different methods, which visually confirms that SHINOBI can produce sharper results that are more faithful to the input images. Further results on the NAVI dataset are shown in Fig. 7, where we show novel views predicted by SHINOBI initialized with either GT poses or rough quadrants. Visual results clearly show that SHINOBI can recover the pose and provide a consistent illumination w.r.t the groundtruth target views in both settings.

Results. Tab. 1 shows the performance of different methods for in-the-wild reconstruction when using GT poses from NAVI. Following NAVI [33], we divide the scenes into two subsets based on whether the COLMAP works (SC) or not (∼ SC) as some techniques like NeROIC need COLMAP poses to work on unposed image collections. Using the provided annotated poses SHINOBI clearly performs best on the view synthesis task (Tab. 1). This shows the advantage of our hybrid encoding scheme and the patch-based losses over previous methods for in-the-wild scenes. Optimization runtimes of different techniques show that we are 3 times faster than the next-best SAMURAI approach.

Tab. 2 shows results of joint shape and pose optimization from in-the-wild image collections when the GT camera poses are not given as input. SHINOBI outperforms both NeROIC and NeRS by a healthy margin while being on-par with SAMURAI. While PSNR of SHINOBI is similar to SAMURAI, our method is able to reconstruct scenes consistently with lower translation and rotation pose errors (with also lower standard deviation in pose metrics). This results

Decomposition results. Fig 5 compares the BRDF and illumination decomposition of SHINOBI to SAMURAI where the same output modalities are available. Visual results show significantly more high-frequency detail and plausible material parameters with SHINOBI compared to SAMURAI.

Ablation study. We ablate different aspects of SHINOBI in

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

| |
|---|

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

NeRS NeROIC

| |
|---|

GT Test View SAMURAI SHINOBI

[Figure 76]

[Figure 77]

[Figure 78]

- Figure 6. Novel view synthesis compared to existing methods. Compared to other methods on an example view from the NAVI [33] in-the-wild test set, SHINOBI preserves fine detail and recreates the lighting realistically.

(a) GT Novel View (b) GT pose init. (c) Direction pose init.

- Figure 7. View synthesis on NAVI. Renderings from SHINOBI using models initialized with camera pose quadrants only or the GT provided by NAVI [33] compared to the input image.

[Figure 79]

(a) Kitchen sink (rgb, normals) (b) Water gun (rgb, normals)

[Figure 80]

[Figure 81]

- Figure 8. Failure cases. Unconstrained image collections featuring highly symmetric objects or homogenous surfaces still pose a challenge and potentially require additional assistance.

terms of reconstruction metrics using the “Keywest” and “School Bus”, two in-the-wild sets from NAVI [33] of medium complexity. Metrics in Tab. 3 show that the resolution annealing coarse-to-fine scheme and the patch-based losses contribute most significantly to the final quality. The latter improves local details and registration accuracy compared to a simple pixel-wise loss. The view importance weighting is another important factor for improved sharpness. It helps to stabilize the optimization after the initial resolution annealing schedule has ended. While the hybrid encoding and camera multiplex consistency do not seem to have a large impact quantitatively, they play a critical role in stabilizing the optimization over different scene types and scales. Without them, the optimization might take longer or diverge depending on the initialization. Visual examples of the specific ablations are compared in the supplementary material.

limited capabilities of the illumination representation [13]. Furthermore, our BRDF and illumination decomposition is not capable of modeling shadowing and inter-reflections. As we are mainly concerned with single-object decomposition, these are not crucial. Extending this method to more complex light transport modeling forms an important future work.

Applications. In addition to novel view synthesis using the NeRF [50] representation, the parametric material model allows for controlled editing of the object’s appearance. Also the illumination can be adjusted, e.g. for realistic composites. A mesh extraction allows further editing and integration in the standard graphics pipeline including real-time rendering. SHINOBI can help in obtaining relightable 3D assets for e-commerce applications as well as 3D AR and VR for entertainment and education. Refer to the supplementary material for sample visual results on relighting, material editing etc.

### 5. Conclusion

We present SHINOBI, a framework for shape, pose, and illumination estimation of objects from unposed in-the-wild image collections. Using a hybrid hash grid encoding scheme we enable easier camera pose optimization using a multiresolution hash grid. Additionally, our choice of camera parameterization along with per-view importance weighting and patch-based alignment loss allows for a better image-to3D alignment resulting in better reconstruction with highfrequency details. Although SHINOBI is able to recover the geometry of objects from any category, its performance is limited on thin/transparent structures and fails to recover high-frequency details under extreme illumination changes, which we leave as exploration for future work.

Limitations. Joint pose and shape reconstruction is an inherently ill-posed problem. While SHINOBI improves over previous work, especially symmetric objects and highly specular materials can lead to failure cases as shown in Fig. 8. The coarse-to-fine scheme is not able to resolve the disambiguities and the camera poses are stuck in a local minimum. All existing methods show these limitations to some extent. In some regions, high-frequency detail is still not reconstructed properly due to misaligned views and the band

### Acknowledgements

This work has been partially funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC number

2064/1 – Project number 390727645 and SFB 1233, TP 02 Project number 276693517.

### References

- [1] Mart´ın Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, and Andy Davis et al. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org. 6
- [2] Miika Aittala, Timo Aila, and Jaakko Lehtinen. Reflectance modeling by neural texture synthesis. ACM TOG, 2018. 3
- [3] Rachel Albert, Dorian Yao Chan, Dan B. Goldman, and James F. O’Brian. Approximate svBRDF estimation from mobile phone video. Eurographics Symposium on Rendering,

2018. 3

- [4] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Zip-NeRF: Anti-Aliased GridBased Neural Radiance Fields. ICCV, 2023. 2, 6, 13
- [5] Sai Bi, Zexiang Xu, Pratul Srinivasan, Ben Mildenhall, Kalyan Sunkavalli, Miloˇs Haˇsan, Yannick Hold-Geoffroy, David Kriegman, and Ravi Ramamoorthi. Neural reflectance fields for appearance acquisition. arXiv, 2020. 2, 3
- [6] Sai Bi, Zexiang Xu, Kalyan Sunkavalli, Miloˇs Haˇsan, Yannick Hold-Geoffroy, David Kriegman, and Ravi Ramamoorthi. Deep reflectance volumes: Relightable reconstructions from multi-view photometric images. ECCV, 2020. 2
- [7] Sai Bi, Zexiang Xu, Kalyan Sunkavalli, David Kriegman, and Ravi Ramamoorthi. Deep 3d capture: Geometry and reflectance from sparse multi-view images. CVPR, 2020. 3
- [8] Wenjing Bian, Zirui Wang, Kejie Li, Jiawang Bian, and Victor Adrian Prisacariu. Nope-nerf: Optimising neural radiance field with no pose prior. CVPR, 2023. 2, 3
- [9] Mark Boss and Hendrik P.A. Lensch. Single image brdf parameter estimation with a conditional adversarial network. arXiv, 2019. 3
- [10] Mark Boss, Fabian Groh, Sebastian Herholz, and Hendrik P. A. Lensch. Deep Dual Loss BRDF Parameter Estimation. Workshop on Material Appearance Modeling, 2018. 3
- [11] Mark Boss, Varun Jampani, Kihwan Kim, Hendrik P.A. Lensch, and Jan Kautz. Two-shot spatially-varying BRDF and shape estimation. CVPR, 2020. 3
- [12] Mark Boss, Raphael Braun, Varun Jampani, Jonathan T. Barron, Ce Liu, and Hendrik P.A. Lensch. NeRD: Neural reflectance decomposition from image collections. ICCV, 2021. 2, 3, 4, 13
- [13] Mark Boss, Varun Jampani, Raphael Braun, Ce Liu, Jonathan T. Barron, and Hendrik P.A. Lensch. Neural-pil: Neural pre-integrated lighting for reflectance decomposition. NeurIPS, 2021. 2, 3, 4, 8, 12
- [14] Mark Boss, Andreas Engelhardt, Abhishek Kar, Yuanzhen Li, Deqing Sun, Jonathan T. Barron, Hendrik P.A. Lensch, and Varun Jampani. SAMURAI: Shape And Material from Unconstrained Real-world Arbitrary Image collections. NeurIPS,

2022. 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16

- [15] Yue Chen, Xingyu Chen, Xuan Wang, Qi Zhang, Yu Guo, Ying Shan, and Fei Wang. Local-to-global registration for

- bundle-adjusting neural radiance fields. CVPR, pages 8264– 8273, 2023. 3
- [16] Zhiqin Chen and Hao Zhang. Learning implicit fields for generative shape modeling. CVPR, 2019. 2
- [17] Weihao Cheng, Yan-Pei Cao, and Ying Shan. Id-pose: Sparseview camera pose estimation by inverting diffusion models. arXiv preprint arXiv:2306.17140, 2023. 16
- [18] Blender Online Community. Blender - a 3D modelling and rendering package. Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018. 16
- [19] Robert L. Cook and Kenneth E. Torrance. A reflectance model for computer graphics. ACM TOG, 1982. 3
- [20] Valentin Deschaintre, Miika Aitalla, Fredo Durand, George Drettakis, and Adrien Bousseau. Single-image SVBRDF capture with a rendering-aware deep network. ACM TOG,

2018. 3

- [21] Valentin Deschaintre, Miika Aitalla, Fredo Durand, George Drettakis, and Adrien Bousseau. Flexible SVBRDF capture with a multi-image deep network. Eurographics Symposium on Rendering, 2019. 3
- [22] Valentin Deschaintre, George Drettakis, and Adrien Bousseau. Guided fine-tuning for large-scale material transfer. Eurographics Symposium on Rendering, 2020. 3
- [23] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021. 3
- [24] Charles Dugas, Yoshua Bengio, Franc¸ois B´elisle, Claude Nadeau, and Ren´e Garcia. Incorporating second-order functional knowledge for better option pricing. In NeurIPS. MIT Press, 2000. 12
- [25] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoidweighted linear units for neural network function approximation in reinforcement learning. Neural networks : the official journal of the International Neural Network Society, 107:3–11, 2017. 12
- [26] Duan Gao, Xiao Li, Yue Dong, Pieter Peers, and Xin Tong. Deep inverse rendering for high-resolution SVBRDF estimation from an arbitrary number of images. ACM Transactions on Graphics (SIGGRAPH), 2019. 3
- [27] Shubham Goel, Angjoo Kanazawa, and Jitendra Malik. Shape and viewpoint without keypoints. ECCV, 2020. 5
- [28] John C Gower and Garmt B Dijksterhuis. Procrustes problems. OUP Oxford, 2004. 6
- [29] Richard Hahnloser, Rahul Sarpeshkar, Misha Mahowald, Rodney Douglas, and H. Seung. Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. Nature, 405:947–51, 2000. 12
- [30] Philipp Henzler, Valentin Deschaintre, Niloy J Mitra, and Tobias Ritschel. Generative modelling of BRDF textures from flash images. ACM Transactions on Graphics (SIGGRAPH ASIA), 2021. 3
- [31] Hwan Heo, Taekyung Kim, Jiyoung Lee, Jaewon Lee, Soohyun Kim, Hyunwoo J. Kim, and Jin-Hwa Kim. Robust camera pose refinement for multi-resolution hash encoding. ICML, 2023. 2, 4, 13

- [32] Ajay Jain, Matthew Tancik, and Pieter Abbeel. Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis. ICCV, 2021. 2
- [33] Varun Jampani, Kevis-Kokitsi Maninis, Andreas Engelhardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan Popov, Andre Araujo, Ricardo Martin-Brualla, Kaushal Patel, Daniel Vlasic, Vittorio Ferrari, Ameesh Makadia, Ce Liu, Yuanzhen Li, and Howard Zhou. Navi: Category-agnostic image collections with high-quality 3d shape and pose annotations. NeurIPS, 2023. 2, 6, 7, 8, 12, 14, 15, 16
- [34] Yoonwoo Jeong, Seokjun Ahn, Christopher Choy, Animashree Anandkumar, Minsu Cho, and Jaesik Park. Selfcalibrating neural radiance fields. ICCV, 2021. 3
- [35] Berk Kaya, Suryansh Kumar, Carlos Oliveira, Vittorio Ferrari, and Luc Van Gool. Uncalibrated neural inverse rendering for photometric stereo of general surfaces. ICCV, 2021. 3
- [36] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv, 2014. 6, 14
- [37] Zhengfei Kuang, Kyle Olszewski, Menglei Chai, Zeng Huang, Panos Achlioptas, and Sergey Tulyakov. NeROIC: Neural object capture and rendering from online image collections. arXiv, 2022. 2, 4, 6, 7, 12
- [38] Hendrik P. A. Lensch, Wolfgang Heidrich, and Hans-Peter Seidel. Automated texture registration and stitching for real world models. Pacific Graphics, 2000. 2, 6
- [39] Axel Levy, Mark Matthews, Matan Sela, Gordon Wetzstein, and Dmitry Lagun. MELON: NeRF with Unposed Images Using Equivalence Class Estimation. 13
- [40] Zhengqin Li, Kalyan Sunkavalli, and Manmohan Chandraker. Materials for masses: SVBRDF acquisition with a single mobile phone image. ECCV, 2018. 3
- [41] Zhengqin Li, Mohammad Shafiei, Ravi Ramamoorthi, Kalyan Sunkavalli, and Manmohan Chandraker. Inverse rendering for complex indoor scenes: Shape, spatially-varying lighting and SVBRDF from a single image. CVPR, 2020. 3
- [42] Zhaoshuo Li, Thomas M¨uller, Alex Evans, Russell H Taylor, Mathias Unberath, Ming-Yu Liu, and Chen-Hsuan Lin. Neuralangelo: High-fidelity neural surface reconstruction. CVPR,

2023. 4, 5, 12

- [43] Ruofan Liang, Huiting Chen, Chunlin Li, Fan Chen, Selvakumar Panneer, and Nandita Vijaykumar. ENVIDR: Implicit Differentiable Renderer with Neural Environment Lighting. arXiv, 2023. 2, 3
- [44] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey. BARF: Bundle-Adjusting Neural Radiance Fields. ICCV, 2021. 3, 4, 5, 12, 15
- [45] Sainan Liu, Shan Lin, Jingpei Lu, Shreya Saha, Alexey Supikov, and Michael Yip. BAA-NGP: Bundle-Adjusting Accelerated Neural Graphics Primitives. arXiv, 2023. 2, 4, 5, 12, 13
- [46] Xiaoxiao Long, Cheng Lin, Peng Wang, Taku Komura, and Wenping Wang. Sparseneus: Fast generalizable neural surface reconstruction from sparse views. ECCV, 2022. 2
- [47] Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi, Jonathan T. Barron, Alexey Dosovitskiy, and Daniel Duckworth. NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections. CVPR, 2021. 2

- [48] Quan Meng, Anpei Chen, Haimin Luo, Minye Wu, Hao Su, Lan Xu, Xuming He, and Jingyi Yu. GNeRF: GAN-based Neural Radiance Field without Posed Camera. ICCV, 2021. 6, 7
- [49] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. CVPR, 2019. 2
- [50] Ben Mildenhall, Pratul Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. ECCV, 2020. 2, 3, 4, 8, 12
- [51] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM TOG, 2022. 2, 3, 4, 6
- [52] Jacob Munkberg, Jon Hasselgren, Tianchang Shen, Jun Gao, Wenzheng Chen, Alex Evans, Thomas Mueller, and Sanja Fidler. Extracting Triangular 3D Models, Materials, and Lighting From Images. CVPR, 2022. 2
- [53] Giljoo Nam, Diego Gutierrez, and Min H. Kim. Practical SVBRDF acquisition of 3d objects with unstructured flash photography. ACM Transactions on Graphics (SIGGRAPH ASIA), 2018. 2, 3
- [54] Niranjan D. Narvekar and Lina J. Karam. A no-reference image blur metric based on the cumulative probability of blur detection (cpbd). TIP, 20(9):2678–2683, 2011. 15
- [55] Michael Niemeyer, Jonathan T. Barron, Ben Mildenhall, Mehdi S. M. Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. CVPR, 2022. 2
- [56] Michael Oechsle, Songyou Peng, and Andreas Geiger. Unisurf: Unifying neural implicit surfaces and radiance fields for multi-view reconstruction. ICCV, 2021. 2, 13
- [57] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. CVPR,

2019. 2

- [58] Keunhong Park, Utkarsh Sinha, Jonathan T. Barron, Sofien Bouaziz, Dan B Goldman, Steven M. Seitz, and Ricardo Martin-Brualla. Deformable neural radiance fields. ICCV,

2021. 12

- [59] Keunhong Park, Philipp Henzler, Ben Mildenhall, Jonathan T. Barron, and Ricardo Martin-Brualla. Camp: Camera preconditioning for neural radiance fields. ACM Trans. Graph., 2023. 2, 4, 13
- [60] G. Ponimatkin, Y. Labbe, B. Russell, M. Aubry, and J. Sivic. Focal length and object pose estimation via render and compare. In CVPR, 2022. 13
- [61] Daniel Rebain, Mark Matthews, Kwang Moo Yi, Dmitry Lagun, and Andrea Tagliasacchi. LOLNeRF: Learn from One Look. CVPR, 2022. 2
- [62] Shen Sang and Manmohan Chandraker. Single-shot neural relighting and SVBRDF estimation. ECCV, 2020. 3
- [63] Paul-Edouard Sarlin, Cesar Cadena, Roland Siegwart, and Marcin Dymczyk. From coarse to fine: Robust hierarchical localization at large scale. In CVPR, 2019. 15
- [64] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. SuperGlue: Learning feature matching with graph neural networks. In CVPR, 2020. 15

- [65] Johannes Lutz Sch¨onberger and Jan-Michael Frahm. Structure-from-motion revisited. CVPR, 2016. 2, 3, 16
- [66] Johannes Lutz Sch¨onberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm. Pixelwise view selection for unstructured multi-view stereo. ECCV, 2016. 2, 3, 16
- [67] Soumyadip Sengupta, Jinwei Gu, Kihwan Kim, Guilin Liu, David W. Jacobs, and Jan Kautz. Neural inverse rendering of an indoor scene from a single image. ICCV, 2019. 3
- [68] S. Sinha, J. Y. Zhang, A. Tagliasacchi, I. Gilitschenski, and D. B. Lindell. Sparsepose: Sparse-view camera pose regression and refinement. CVPR, 2023. 3
- [69] Pratul P. Srinivasan, Boyang Deng, Xiuming Zhang, Matthew Tancik, Ben Mildenhall, and Jonathan T. Barron. NeRV: Neural reflectance and visibility fields for relighting and view synthesis. CVPR, 2021. 2
- [70] Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. NeurIPS, 2020. 2, 4, 12
- [71] Prune Truong, Marie-Julie Rakotosaona, Fabian Manhardt, and Federico Tombari. Sparf: Neural radiance fields from sparse and noisy poses. CVPR, 2023. 2
- [72] Itsuki Ueda, Yoshihiro Fukuhara, Hirokatsu Kataoka, Hiroaki Aizawa, Hidehiko Shishido, and Itaru Kitahara. Neural density-distance fields. ECCV, 2022. 2
- [73] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T. Barron, and Pratul P. Srinivasan. Ref-neRF: Structured view-dependent appearance for neural radiance fields. CVPR, 2022. 13
- [74] Jianyuan Wang, Christian Rupprecht, and David Novotny. PoseDiffusion: Solving pose estimation via diffusion-aided bundle adjustment. ICCV, 2023. 15, 16
- [75] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. NeuS: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. NeurIPS, 2021. 2
- [76] Yiming Wang, Qin Han, Marc Habermann, Kostas Daniilidis, Christian Theobalt, and Lingjie Liu. Neus2: Fast learning of neural implicit surfaces for multi-view reconstruction. ICCV,

2023. 2

- [77] Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Victor Adrian Prisacariu. NeRF−−: Neural radiance fields without known camera parameters. arXiv, 2021. 3, 4
- [78] Jiamin Xu, Zihan Zhu, Hujun Bao, and Weiwei Xu. A Hybrid Mesh-neural Representation for 3D Transparent Object Reconstruction. cvmj, 2022. 2
- [79] Jiawei Yang, Marco Pavone, and Yue Wang. FreeNeRF: Improving Few-shot Neural Rendering with Free Frequency Regularization. CVPR, 2023. 2
- [80] Yao Yao, Jingyang Zhang, Jingbo Liu, Yihang Qu, Tian Fang, David McKinnon, Yanghai Tsin, and Long Quan. NeILF: Neural Incident Light Field for Physically-based Material Estimation. ECCV, 2022. 2
- [81] Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Volume rendering of neural implicit surfaces. NeurIPS, 2021. 2

- [82] Weicai Ye, Shuo Chen, Chong Bao, Hujun Bao, Marc Pollefeys, Zhaopeng Cui, and Guofeng Zhang. IntrinsicNeRF: Learning Intrinsic Neural Radiance Fields for Editable Novel View Synthesis. ICCV, 2023. 2, 3
- [83] Jianzhao Zhang, Guojun Chen, Yue Dong, Jian Shi, Bob Zhang, and Enhua Wu. Deep inverse rendering for practical object appearance scan with uncalibrated illumination. ACG,

2020. 3

- [84] Jason Zhang, Gengshan Yang, Shubham Tulsiani, and Deva Ramanan. NeRS: Neural reflectance surfaces for sparse-view 3d reconstruction in the wild. NeurIPS, 2021. 2, 3, 4, 6, 7, 14
- [85] Jiahui Zhang, Fangneng Zhan, Rongliang Wu, Yingchen Yu, Wenqing Zhang, Bai Song, Xiaoqin Zhang, and Shijian Lu. VMRF: View Matching Neural Radiance Fields. ACM MM,

2022. 3

- [86] Jingyang Zhang, Yao Yao, Shiwei Li, Jingbo Liu, Tian Fang, David McKinnon, Yanghai Tsin, and Long Quan. Neilf++: Inter-reflectable light fields for geometry and material estimation. ICCV, 2023. 3
- [87] Kai Zhang, Fujun Luan, Qianqian Wang, Kavita Bala, and Noah Snavely. PhySG: Inverse rendering with spherical Gaussians for physics-based material editing and relighting. CVPR,

2021. 2, 3

- [88] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. CVPR, 2018. 6
- [89] Xiuming Zhang, Pratul P. Srinivasan, Boyang Deng, Paul Debevec, William T. Freeman, and Jonathan T. Barron. Nerfactor: Neural factorization of shape and reflectance under an unknown illumination. ACM Trans. Graph., 40(6), 2021. 3
- [90] Yuanqing Zhang, Jiaming Sun, Xingyi He, Huan Fu, Rongfei Jia, and Xiaowei Zhou. Modeling indirect illumination for inverse rendering. CVPR, 2022. 2, 3
- [91] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. CVPR, 2019. 4, 13
- [92] Hao Zhu, Fengyi Liu, Qi Zhang, Xun Cao, and Zhan Ma. Rhino: Regularizing the hash-based implicit neural representation. arXiv, 2023. 2, 4, 12

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

| |
|---|

| |
|---|

N/A

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

NeRS NeROIC

NeRS NeROIC

[Figure 105]

NeROIC NeRS

[Figure 106]

| |
|---|

| |
|---|

| |
|---|

GT Novel View SAMURAI SHINOBI

GT Novel View SAMURAI SHINOBI

GT Novel View SAMURAI SHINOBI

- Figure 9. Novel view synthesis compared to existing methods. Additional example objects from NAVI [33] in-the-wild image collections. SHINOBI robustly reconstructs even when initialized with exteremely coarse poses while e.g. NeROIC [37] does not succeed on some scenes.

### Overview

In the supplement to SHINOBI, a method for 3D joint reconstruction of shape, illumination and materials from inthe-wild image sequences, we first present additional details on the method’s architecture (Sec.A.2) and the optimization (Sec. A.5). In Sec. B we introduce additional qualitative results from object reconstructions of the NAVI dataset [33] and add visual examples to our ablation study. Finally, applications of our reconstructed data are shown in Sec. B.5. Please also visit our project page for an overview of this work and further visual results video.

### A. Additional Method Details

#### A.1. NeRF Raymarching

As introduced in Sec. 3 the neural networks in NeRF [50] output a vector for view-dependent output color c ∈ R3 and volume density σ ∈ R given a 3D location x ∈ R3 and view direction d ∈ R3. A camera ray r(t) = o+td is cast into the volume, with ray origin o ∈ R3 and view direction d. The final color is then approximated via numerical quadrature

of the integral: cˆ(r) = t tf

T(t)σ(t)c(t)dt with T(t) = exp(− t t

n

σ(t)dt), using the near and far bounds of the ray

n

- tn and tf respectively [50]. Originally, the first MLP learns a coarse representation by sampling the volume in a fixed uniform sampling pattern along each ray. The second MLP is then evaluated sampled according to the coarse density distribution, placing more samples in high-density areas. In SHINOBI we only use one sampling stage with uniformly samples along rays. Using a raymarching scheme that skips empty space based iteratively updated occupancy data could bring additional performance gain during optimization.

#### A.2. Architecture

Hybrid hash encoding configuration. The hybrid encoding features two branches. For the base encoding we use 10 random offset annealed Fourier frequencies for the positional encoding followed by a small MLP featuring a single hidden layer with 64 dimensions and silu activation [25]. The output equals the input dimension (3), again as it is done by Zhu et al. [92]. We apply BARF’s [44] Fourier annealing and add

random frequencies as offsets to the logarithmically spaced frequencies [14, 70] to prevent artifacts from axis-aligned frequencies. The multiresolution hash grid is configured with 16 levels with a base resolution of 8 and a maximum target resolution of 2048. The embedding dimensions are 2 or 4. The experiments reported in Sec. 4 of our paper are generated using 2 dimensions. A slightly better decomposition quality can be achieved by increasing the dimensionality at the cost of increased memory consumption and runtime. Hence, the final feature dimension after encoding and concatenation is 35 or 67. See Sec. A.2 for an explanation of the annealing strategy applied to the hash grids.

Networks. The main network taking in the encoded features consists of 3 ReLU [29] activated layers with 64 channels. An additional linear layer generates the output for the σ density parameter from the 64 channel activation. Softplus softplus(x) = ln(1 + ex) [24] is applied to the raw σ. The directions are encoded using 4 non-annealed regular Fourier components as in Mildenhall et al. [50] and then, concatenated with the main network output, fed to a secondary MLP to predict the view direction-dependent radiance c˜used in the beginning of the optimization. The secondary conditional network has a hidden dimension of 32 in our case. For the BRDF prediction a single linear layer compresses the main network output to 16 channels. From there the BRDF decoder is applied which consists of another two layers with 64 channels and ReLU activation each. Each BRDF output; basecolor, metallic and roughness has its own output layer followed by a sigmoid activation [13]. An additional diffuse embedding is added as conditioning to the basecolor branch before output. The illumination network decoding the per view latent vector is conditioned by the same configuration of mapping layers as outlined in Neural-PIL [13].

Multiresolution hash grid level annealing. Inspired by BARF [44] and Nerfies [58] we apply a coarse-to-fine annealing to the hash grid encoding by weighting the different grid levels. Starting with only the features from the low resolution dense grid and all other features set to zero we increase the weights of the higher resolution levels gradually over time (cf. [42, 45]). Similar to the implementation by

Lin et al. we formulate it as a truncated Hann window:

Γk(x;α) = wk(α) sin(2kx),cos(2kx) (5) wk(α) =

1 − cos(π clamp(α − k,0,1)) 2

(6)

where α ∈ [0,L] with L being the number of resolution levels of the hash grid encoding. We also tested the idea of BAA-NGP [45] replicating embeddings from low-resolution levels but observed reduced performance in our optimization setting. Similarly, we had no success with adding a straight-through operator to the interpolation on the hash grid as proposed in [31].

#### A.3. Camera Parameterization.

We label initial poses based on 3 simple binary questions: Left vs. Right, Above vs. Below, and Front vs. Back. This only takes about 4-5 minutes for a typical 80 image collection. Alternatively, our framework allows to extend the initialization to a camera multiplex spanning more than one quadrant. This can enable fully random initialization for front-facing scenes and image sets featuring rotating cameras with a fixed object distance as shown by Levy et al. [39]. As these constrained settings are uncommon for in-the-wild collections we discard it here. We use a perspective pinhole camera model and an initial field of view of 53.13 degrees. We optimize offsets to the original camera parameters of our ‘lookat + direction’ parameterization as outlined in the main paper. Here, we encode the trainable lookat parameter ∆d directly as two direction components, ϕ, θ, which are used

- to offset the viewing direction d to obtain the updated dˆas follows:

d = (peye + ∆peye) − pcenter (7) θ = arcsin(dy) + ∆dθ (8) ϕ = arctan2(dx,dz) + ∆dϕ (9) dˆ= ⟨cosϕsinθ,sinϕ,cosϕcosθ⟩ (10)

We limit ∆d to the range [−0.5π,0.5π]. We also tried other camera parameterizations like the popular 6D rotation representation by Zhou et al. [91] or FocalPose [60] that has recently been applied to NeRF with camera fine-tuning [59]. Interestingly, our lookat + direction parameterization performs the best in our setting as it seems to work well with the regularizations on camera poses.

#### A.4. Regularization and Losses.

Multiresolution hash grid regularization. To regularize the hash grid encoding we use the following normalized weight decay as proposed by Barron et al. [4]: LGrid =

l mean(Vl) with Vl referring to the grid embeddings at resolution level l. Computing the sum of the mean per-level puts a higher penalty on coarser grid levels compared to

naive weight decay over all parameters at once. We find a weighting of 0.02 to 0.05 work well in our setting and settle for 0.02 as the final value. We apply gradient scaling to the gradients for the network by the norm of 0.1. Furthermore, gradient norm clipping with a clip value of 2.5 is applied to the camera gradients before the parameter update.

Surface normals regularization. We use the normal direction loss Lndir from [73] to constrain the normals to face the camera until the ray reaches the surface. This helps in providing sharper surfaces without floater artifacts. Additionally, we observe that the explicit rendering step helps to constrain the surface normals as noise is reduced compared to optimization using only the predicted radiance.

Camera regularization. The camera regularization losses from SAMURAI are kept, particularly one to force the lookat-direction to point towards the origin (LLookat) and one to prevent the cameras from moving too far away from the bounding volume (LBounds) [14]. An additional term on the magnitude of the camera offset parameters helps to keep cameras from moving too far too fast with respect to the initial position due to strong updates in the beginning of the optimization.

BRDF losses. Joint estimation of BRDF and illumination is a delicate endeavor. For example, the illumination can easily fall into a local minimum. The object is then tinted in a bluish color, and the illumination is an orange color to express a more neutral color tone, for example. As our image collections have multiple illuminations, we can force the base color bc to replicate the pixel color from the input images. This way, a mean color over the dataset is learned and it becomes less likely to be trapped in local minima. We evaluate the Mean Squared Error (MSE) for this: LInit = LMSE(Cs,bc). Additionally, we add a smoothness loss LSmooth for the normal, roughness, and metallic parameters similar to the one used in UNISURF [56] to further regularize BRDF estimation [14].

Image reconstruction loss is a Charbonnier loss: LImage(g,p) = (g − p)2 + 0.0012 between the input color from C for pixel s and the corresponding predicted color of the networks c˜. We also calculate the loss with the rendered color cˆ which becomes the main loss over time. This loss is computed over multiple resolution levels as outlined in Sec. 3 of the main paper whenever patches are rendered.

Mask losses. In total we use three mask loss terms. The Lsilhouette as described in Sec 3.2 as well as the binary crossentropy loss LBCE between the volume-rendered mask and estimated foreground object mask and the background loss LBackground from NeRD [12]. The latter enforces all rays cast to the background to return 0. Consequently, the total mask loss is defined as: LMask = λxorLsilhouette + LBCE + LBackground where λxor is set to 50 and Lsilhouette is normalized by the number of elements in the reference mask.

Final loss ensemble. Overall we compute two loss terms LNetwork and LCamera which consist of differently weighted versions of the photometric rendering loss and alignment losses plus the respective regularizations. The loss to optimize the decomposition network can be written as LNetwork = λbLImage(Cs,c˜) + (1 − λb)LImage(Cs,cˆ) + LMask + λaLInit + λndirLndir + λSmoothLSmooth +

λDecSmoothLDecSmooth+λDecSparsityLDecSparsity. Here, λb and λa are the optimization scheduling weights described below in more detail. As long as the camera multiplex has size m > 1 the camera multiplex consistency loss is added as follows: LNetwork = LNetwork+0.1(Lmultiplex). To these losses the camera posterior scaling is applied as in SAMURAI [14]. The camera loss is weighted according to our view importance scaling instead. Badly initialized camera poses can still recover over the training duration as they get potentially large updates while cameras that perform well in terms of the losses are gradually faded out of the optimization. Additionally, the regularizations from above, LBounds and LLookat are added.

View Importance Start

Focal Length Start

Annealing End

- 0.8
- 1

image resolution factor

ScalerValue

0.6

0.4

0.2

0

0 1 0k 20k 30k 40k 50k 60k

Optimization Steps

- Figure 10. Optimization schedule. We use three λ parameter to scale losses to enable a smooth flow of the optimization parameters. Additionally, we indicate at which points in time the view importance weighting is introduced, the focal length parameters start to get updated and the encoding annealing ends.

#### A.5. Optimization

Optimization scheduling. We use three fading λ variables to steer the optimization schedule smoothly as visualized in Fig. 10. Render resolution is continuously increased over the first half of the optimization while the number of active multiplex cameras is reduced. This is controlled by λc. Input image resolution is increased from 100 pixels to a resolution of 400 pixels on the longer image side over the first half of the training. For higher final output resolutions an even larger downsample factor (> 4) might be needed. This strategy allows the image patches to include even larger structures

of the objects and improves camera alignment. The direct color optimization is faded to the BRDF optimization and the encoding annealing is performed over the first third of the optimization. λb is used for the BRDF transition and an independent α value is kept for the annealing. Finally, λa is used to scale some losses in a non-linear way. Focal length updates are delayed until a quarter of the optimization time. We start with the view importance weighting at the half-way point of the annealing schedule. SHINOBI renders image patches for most of the training time which adds more context to each update step, allowing us to add new losses tailored to camera alignment. The first 1000 steps are trained using regular random ray sampling, though, to help initialize a global shape quickly while both the render resolution as well as the hash grid resolution are low.

Optimizer settings. The ADAM [36] optimizer updates the network weights based on LNetwork with a learning rate of 1e-3 that is exponentially decayed by an order of magnitude over the training time. The same decay rate is applied to the optimizer concerned with the hash grid embeddings. The gradient are computed based on LNetwork with the hash grid specific regularization LGrid added. The learning rate of the camera optimizer is exponentially decayed by an order of magnitude every 40k steps. As mentioned before the β1 parameter is set to 0.2 for the camera optimizer to stabilize the training in the presence of noisy gradients. It uses the gradients computed based on LCamera. The framework is trained using float16 mixed precision. The coordinate input to the encoding is 32 bit as is the rendering and illumination evaluation. The other MLPs and specifically the interpolation on the hash grids run at 16 bit precision, though.

### B. Additional Experiments

#### B.1. Details on Compared Methods.

In addition to SAMURAI which has been introduced in Sec. 3 of the main paper we compare against two more recent methods for in-the-wild object reconstruction.

NeRS stands for Neural Reflectance Surfaces [84] that constrain reconstructions using a mesh-based representation. Starting from manually annotated rough initial poses and a template mesh the objects are decomposed into a surface mesh, illumination and surface reflectivity parameterized as albedo and shininess. We define the dimensions of an initial cuboid that approximates the object’s bounding box for each scene in line with [33, 84].

NeROIC presents a multi-stage approach to reconstruct geometry and material properties of objects from online image collections. Camera poses are initialized with a COLMAPbased pipeline and fine-tuned during the first reconstruction stage. Following high-quality surface normals are estimated during the second stage. Finally, material properties and illumination are optimized to enable relighting in addition to

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

novel view synthesis.

| |
|---|

| |
|---|

#### B.2. Additional Visual Results

[Figure 111]

[Figure 112]

Fig. 9 shows additional qualitative results on objects from the NAVI dataset compared to the baseline methods. Note, that the methods work at different image resolutions and that we show the original output. NeROIC is able to reconstruct high-frequency detail for scenes that have good initial poses but shows artifacts or fails on others. NeRS suffers from its low resolution mesh representation and often inaccurate camera alignment. SAMURAI and SHINOBI both reproduce appearance that is closer to the original illumination setting due to superior decomposition capabilities while SHINOBI recovers more high-frequency details. Consequently, on the reference free sharpness metric CPBD [54] our method clearly improves upon SAMURAI and NeRS, with CPBD scores of 0.82 (Ours) vs. 0.77 for SAMURAI vs. 0.65 for NeRS on the most challenging subset of the NAVI scenes where COLMAP reconstruction fails.

(a) Only Fourier

(b) Fourier faded out

[Figure 113]

[Figure 114]

| |
|---|

| |
|---|

(c) Eval hash grid only

(d) Hybrid encoding

Figure 12. Hybrid Encoding Ablation. The Fourier encoding on its own is band limited, e.g. text is not reconstructed. Fading out the Fourier encoding during training destabilizes the optimization showing the importance of both encoding schemes. Evaluation using only the hash grid encoding results in noisy density but sharp texture. Our hybrid encoding yields sharp results with a consistent density at the object’s surface.

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

#### B.3. Qualitative Results of Ablations

- Fig. 11 shows qualitative results corresponding to the numerical results from the ablation study reported in the main paper. It can be observed that a robust reconstruction is only possible using the full configuration of our method. While the multiplex consistency loss has only minimal impact on this example the result still shows some visible artifacts and overall increased noise level. It is apparent that a plain integration of multi-resolution hash grid encoding does not perform well on the task of joint camera pose and shape reconstruction
- Fig. 12 visualizes how the encoding schemes complement each other. It has been shown that Fourier encoding together with a large MLP and a coarse-to-fine scheme does perform reasonable well for camera optimization [14, 44]. Hence, we merge the two encoders while keeping the MLP small and therefore band-limited. The main advantage is the continuity of the gradients for the coarse geometry that help to pull the optimization targets closer to the optimum in the beginning of the optimization. Updates based on higher-frequency details later in the optimization can propagate through the hash grids as they are constrained by our additional losses.

[Figure 119]

[Figure 120]

| |
|---|

[Figure 121]

[Figure 122]

Full

w/o Multiplex Consistency Loss

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

w/o Per View Importance w/o Coarse-to-fine

Method Translation↓ Rotation °↓

| |
|---|

SC ∼ SC SC ∼ SC

PoseDiffusion [74] 0.51± 0.09 0.43± 0.11 41.33± 15.15 43.50± 13.67 HLoc [63, 64] 0.07± 0.13 0.06± 0.10 9.10± 18.75 9.72± 20.08 SHINOBI 0.250± 0.0850.28± 0.09 22.84± 16.19 33.00± 19.97

w/o Hybrid Encoding w/o Patch-based Training

Table 4. Pose estimation on in-the-wild data.. Evaluation of absolute rotation and translation errors after alignment on the NAVI [33] in-the-wild scenes. We compare SHINOBI against specialized camera pose estimation solutions. Note, that HLoc [63] fails completely on 5 scenes and is only able to recover 55% of views on average.

- Figure 11. Qualitative ablation study. We show view synthesis results from novel view synthesis on the ‘School Bus’ scene from NAVI where we ablate components of our method. The visual results underline the importance of each part.

[Figure 127]

(a) Reconstructed assets under novel illumination

[Figure 128]

(b) Edited materials

Figure 13. Integration and editing. Although objects are initially captured under diverse illumination settings we can integrate multiple objects consistently into a scene in the end. BRDF parameters can be modified independently from the illumination.

#### B.4. Comparison to other Camera Pose Estimation Methods

Tab. 4 compares methods for camera pose estimation on the NAVI in-the-wild scenes [33]. Traditional SfM methods like COLMAP [65, 66] paired with a neural feature detection and matching can recover poses with great accuracy but only succeed on a subset of scenes and images. PoseDiffusion [74] and ID-Pose [17], both fully neural models trained on large datasets, struggle on these out-of-distribution examples. We only report a full evaluation on PoseDiffusion as an example here. We observe that these models take important pose cues also from the background of object-centric image sets. This leads to poor results on in-the-wild image collections. A simple fine-tuning on masked images did not improve performance. In our experiments, camera pose estimation usually regresses to a front-facing camera layout for in-thewild examples featuring different illumination and object scales. Consequently, our approach appears to be a good trade-off in-terms of camera pose quality.

[Figure 129]

[Figure 130]

[Figure 131]

Figure 14. Relighting application. View synthesis under three different illumination settings using the estimated decomposition for a sample view from the “Tractor” scene.

#### B.5. Downstream Applications

The object decomposition into BRDF, illumination and shape enables us to edit illumination and material independently of the shape representation to re-light the object, for example. Furthermore, we can convert our neural representation into a parametric model like a mesh and physically based material suitable for easy integration into standard graphics pipelines. Mesh extraction and asset generation. We use a modified version of the mesh extraction component from SAMURAI [14] to extract triangle meshes from the learnt volume and the corresponding material parameters. Marching cubes is used to create an initial mesh. We postprocess the mesh and perform automatic UV unwrapping using Blender [18]. Finally, textures are extracted by querying our pipeline for the BRDF around the baked surface locations. The extraction of a mesh takes around 3 minutes. Relighting and material editing. Our reconstructed assets can then be easily integrated into existing graphics pipelines. In Fig. 13 we show a SHINOBI themed scene featuring objects from the NAVI dataset in a new consistent illumination environment as it would be required for AR and VR applications. We can also modify the BRDF parameters independently of the lighting. Fig. 14 compares renderings of the same camera view but lit with different environment lights. Please also consider watching the supplementary video including more examples for the given applications.

