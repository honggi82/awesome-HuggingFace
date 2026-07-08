## DreamWaltz-G: Expressive 3D Gaussian Avatars from Skeleton-Guided 2D Diffusion

Yukun Huang, Jianan Wang, Ailing Zeng, Member, IEEE, Zheng-Jun Zha, Member, IEEE, Lei Zhang, Fellow, IEEE, Xihui Liu Member, IEEE

### arXiv:2409.17145v1[cs.CV]25Sep2024

Abstract—Leveraging pretrained 2D diffusion models and score distillation sampling (SDS), recent methods have shown promising results for text-to-3D avatar generation. However, generating high-quality 3D avatars capable of expressive animation remains challenging. In this work, we present DreamWaltz-G, a novel learning framework for animatable 3D avatar generation from text. The core of this framework lies in Skeleton-guided Score Distillation and Hybrid 3D Gaussian Avatar representation. Specifically, the proposed skeleton-guided score distillation integrates skeleton controls from 3D human templates into 2D diffusion models, enhancing the consistency of SDS supervision in terms of view and human pose. This facilitates the generation of high-quality avatars, mitigating issues such as multiple faces, extra limbs, and blurring. The proposed hybrid 3D Gaussian avatar representation builds on the efficient 3D Gaussians, combining neural implicit fields and parameterized 3D meshes to enable real-time rendering, stable SDS optimization, and expressive animation. Extensive experiments demonstrate that DreamWaltz-G is highly effective in generating and animating 3D avatars, outperforming existing methods in both visual quality and animation expressiveness. Our framework further supports diverse applications, including human video reenactment and multi-subject scene composition. For more vivid 3D avatar and animation results, please visit https://yukun-huang.github.io/DreamWaltz-G/.

Index Terms—3D avatar generation, 3D human, expressive animation, diffusion model, score distillation, 3D Gaussians.

✦

1 INTRODUCTION

# A

NIMATABLE 3D avatar generation is essential for a wide range of applications, such as film and car-

toon production, video game design, and immersive media such as virtual/augmented reality. Traditional techniques for creating such intricate 3D avatars are costly and time-consuming, requiring thousands of hours from skilled artists with extensive aesthetics and 3D modeling knowledge. Meanwhile, the advancement of 3D reconstruction [1], [2], [3], [4] has enabled promising methods which can reconstruct 3D human models from monocular images [5], [6], [7], [8], [9], monocular videos [10], [11], [12], [13], or 3D scans [14], [15], [16], [17]. Nonetheless, these methods rely heavily on the collection of image/video data captured with a monocular camera or a synchronized camera array. This makes them unsuitable for generating 3D avatars from imaginative but abstract prompts like texts.

Recently, integrating pretrained text-to-image diffusion models [18], [19] into 3D modeling with score distillation sampling (SDS) [20], [21] has gained significant attention to make 3D digitization more accessible, alleviating the need

- • Y. Huang and X. Liu are with The University of Hong Kong (HKU), Hong Kong SAR 999077, China. E-mail: yukun@hku.hk, xihuiliu@eee.hku.hk
- • J. Wang is with Astribot, Shenzhen 518063, China. E-mail: jiananwang@astribot.com
- • A. Zeng is with Tencent, Shenzhen 518054, China. E-mail: ailingzengzzz@gmail.com
- • Z. Zha is with University of Science and Technology of China (USTC), Hefei 230026, China. E-mail: zhazj@ustc.edu.cn
- • L. Zhang is with International Digital Economy Academy (IDEA), Shenzhen 518045, China. E-mail: leizhang@idea.edu.cn

: Corresponding author.

for data collection. However, creating 3D avatars using a

- 2D diffusion model remains challenging. First, static avatars require articulated structures with intricate parts (e.g., hands and faces) and detailed textures, which pretrained diffusion models and score distillation struggle to generate. Secondly, dynamic avatars assume various poses in a coordinated and constrained manner, where changes in shape and appearance should be realistic without artifacts caused by inaccurate skeleton rigging. Although previous methods [22], [23], [24], [25], [26], [27], [28] have demonstrated impressive results on text-driven 3D avatar creation, they still struggle with producing intricate geometric structures and detailed appearances, let alone for realistic animation.

In this paper, we present DreamWaltz-G, a zero-shot learning framework for text-driven 3D avatar generation. At the core of this framework are Skeleton-guided Score Distillation (SkelSD) and Hybrid 3D Gaussian [4] Avatars (H3GA) for stable optimization and expressive animation.

For SkelSD, different from previous methods [24], [25], [26] that only apply human priors to 3D avatar representations (e.g., 3D mesh [24]), we additionally inject human priors into diffusion model through skeleton control [29], [30], leading to a more stable SDS that conforms to the 3D human body structure. This design brings three benefits: (1) skeleton guidance from 3D human templates [31], [32] enhances the 3D consistency of SDS and prevents the Janus (multi-face) problem; (2) it eliminates pose uncertainty of SDS and avoids defects such as extra limbs and ghosting; (3) randomly posed skeleton guidance enables pose-dependent shape and appearance learning from 2D diffusion model.

H3GA is a hybrid 3D representation for animatable

- 3D avatars, specifically designed to adapt SDS optimization and enable expressive animation. Specifically, H3GA

Expressive Animation Shape Editing

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

Human Video Reenactment

Multi-subject Scene Composition

|[Figure 11]|[Figure 12]|
|---|---|

|[Figure 13]|[Figure 14]|
|---|---|

Fig. 1. We present DreamWaltz-G, a text-driven animatable 3D avatar generation framework, which can create high-quality 3D avatars from imaginative text prompts and animate them given motion sequences without manual rigging and retraining. Our method enables various downstream applications, such as expressive animation production, shape editing, human video reenactment, and multi-subject scene composition.

combines the efficiency of 3D Gaussian Splatting [4], the local continuity of neural implicit fields [1], [2], and the geometric accuracy of parameterized meshes [31], [32]. As a result, H3GA supports real-time rendering, is robust to SDS optimization, and enables expressive animation with finger movements and facial expressions. Furthermore, considering the dynamic characteristics of different body parts, we designed a dual-branch deformation strategy to drive canonical 3D Gaussians for realistic animation.

Based on the proposed SkelSD and H3GA, DreamWaltzG generates animatable 3D avatars in two training stages:

- (I) Canonical Avatar Generation. For Stage I, we aim to create a canonical 3D avatar given text descriptions. Specifically, we employ Instant-NGP [33] as the canonical avatar representation and optimize it with SkelSD for shape and appearance learning, where the skeleton guidance is extracted from SMPL-X [32] in the canonical pose.
- (II) Animatable Avatar Learning. For Stage II, we aim to make the canonical avatar from Stage I rigged to SMPLX and accurately animated. We employ H3GA as the animatable avatar representation for efficient deformation and stable optimization. Similar to Stage I, we use SkelSD for pose-dependent shape and appearance learning, except the skeleton guidance is extracted from SMPL-X in randomly sampled plausible poses.

In summary, our framework learns a hybrid 3D Gaussian avatar representation using skeleton-guided score distillation, ready for expressive animation and a wide range of applications, as illustrated in Figure 1. The key contributions of this work lie in four main aspects:

- • We introduce a text-driven animatable 3D avatar generation framework, i.e., DreamWaltz-G, ready for expressive animation and various applications.
- • We propose SkelSD, a novel skeleton-guided score distillation strategy to reduce the view and pose inconsistencies between the 3D avatar’s rendering and the 2D diffusion model’s supervision.

- • We propose H3GA, a hybrid 3D Gaussian avatar representation that enables stable SDS optimization, real-time rendering, and expressive animation with finger movements and facial expressions.
- • Experiments demonstrate that DreamWaltz-G can effectively create animatable 3D avatars, achieving superior generation and animation quality compared to existing text-to-3D avatar methods.

Compared with the preliminary conference version [28], this work introduces several non-trivial improvements. The most significant enhancement is the redesign of 3D avatar representation. Specifically, DreamWaltz [28] uses InstantNGP [33] for modeling 3D avatars. However, when applied to dynamic avatars with deformation, high-resolution sampling combined with inverse LBS [31] becomes computationally expensive and impractical for training. To address this, DreamWaltz-G adopts a novel hybrid 3D Gaussian representation, benefiting from efficient deformation and rendering of 3DGS [4] while remaining compatible with SDS optimization and SMPL-X parameters. Additionally, we replace the used 3D human parametric model SMPL [31] with SMPL-X [32], introduce local geometric constraints for NeRF training, and explore more potential applications.

###### 2 RELATED WORK

We first review the previous methods for 2D diffusion models and then discuss recent advances in text-driven 3D object and 3D avatar generation.

###### 2.1 Text-driven Image Generation

Recently, there have been significant advancements in textto-image models such as GLIDE [34], unCLIP [18], Imagen [35], and Stable Diffusion [19], which enable the generation of highly realistic and imaginative images based on text prompts. These generative capabilities have been made

TABLE 1 Comparisons of different text-driven 3D avatar generation methods. To clarify, Shape Control refers to specifying the avatar’s shape during generation instead of the shape initialization†, while Shape Editing involves adjusting the avatar’s shape after generation.

|Methods<br><br>|3D Model<br><br>|Body Animation|Hand Animation<br><br>|Face Animation<br><br>|Shape Control|Shape Editing<br><br>|
|---|---|---|---|---|---|---|
|DreamHuman [25] DreamWaltz [28] TADA† [24] HumanGaussian [27] GAvatar [26]<br><br>|NeRF NeRF Mesh 3DGS 3DGS<br><br>|✓ ✓ ✓ ✓ ✓<br><br>|✕ ✕ ✓ ✕ ✕|✕ ✕ ✓ ✕ ✕<br><br>|✕ ✓ ✕ ✓ ✕<br><br>|✕ ✕ ✓ ✓ ✓<br><br>|
|DreamWaltz-G (Ours)|3DGS<br><br>|✓<br><br>|✓|✓<br><br>|✓|✓|

possible by advancements in modeling, such as diffusion models [36], [37], [38], and the availability of large-scale web data containing billions of image-text pairs [39], [40], [41]. These datasets encompass a wide range of general objects, with significant variations in color, texture, and camera viewpoints, providing pre-trained models with a comprehensive understanding of general objects and enabling the synthesis of high-quality and diverse objects. Furthermore, recent works [29], [30], [42], [43] have explored incorporating additional conditioning, such as depth maps and human skeleton poses, to generate images with more precise control. With more advanced network architectures [44], [45], [46] and larger, higher-quality datasets [47], [48], the capabilities of text-to-image generation models continue to improve.

###### 2.2 Text-driven 3D Object Generation

Dream Fields [49] and CLIPmesh [50] were groundbreaking in their utilization of CLIP [51] to optimize an underlying 3D representation, aligning its 2D renderings with userspecified text prompts without necessitating costly 3D training data. However, this approach tends to result in less realistic 3D models since CLIP only provides discriminative supervision for high-level semantics. In contrast, recent works have demonstrated remarkable text-to-3D generation results by employing powerful text-to-image diffusion models as a robust 2D prior for optimizing a differentiable 3D representation with Score Distillation Sampling (SDS) [20], [21], [52], [53], [54]. Nonetheless, the high variation in SDS leads to blurriness, over-saturated colors, and 3D inconsistencies. Although a series of subsequent works [55], [56], [57], [58], [59], [60] have introduced fundamental improvements to SDS optimization, the results remain unsatisfactory when applied to generating animatable 3D avatars with intricate details.

###### 2.3 Text-driven 3D Avatar Generation

Different from everyday objects, 3D avatars have detailed textures and intricate geometric structures that can be driven for realistic animation. Avatar-CLIP [22] employs CLIP [51] for shape sculpting and texture generation but tends to produce less realistic and oversimplified 3D avatars. Unlike CLIP-based methods, both AvatarCraft [23] and DreamAvatar [61] leverage powerful text-toimage diffusion models to provide 2D image guidance, effectively improving the visual quality of generated avatars.

DreamWaltz [28] and AvatarVerse [62] further utilizes ControlNet [29] and SMPL [31] to provide view/pose-consistent

- 2D human guidance such as skeleton and DensePose [63]. Considering the limited 3D awareness of 2D diffusion models, HumanNorm [64] proposes the normal-adapted and depth-adapted diffusion models for accurate geometry generation. In addition, to enable animatable avatar learning, DreamHuman [25] employs implicit 3D human model imGHUM [65] as 3D avatar representation, which improves the dynamic visual quality of generated avatars. Recently,
- 3D Gaussian Splatting (3DGS) [4] has emerged as an explicit 3D representation enabling real-time deformation [66] and rendering. Some works [14], [16], [26], [27], [67], [68] have explored using 3DGS to represent 3D avatars. HumanGaussian [27] proposes a Structure-Aware SDS, which guides the adaptive density control of 3DGS with intrinsic human structures. GAvatar [26] introduces a primitive-based 3DGS representation where 3D Gaussians are defined inside posedriven primitives to facilitate animation.

To highlight our contributions, we summarize the key differences between our work and related works in Table 1.

###### 3 METHOD

We first review some preliminary knowledge in Sec. 3.1, then present the proposed Skeleton-guided Score Distillation

- in Sec. 3.2 and Hybrid 3D Gaussian Avatar Representation
- in Sec. 3.3. Finally, we introduce the text-driven 3D avatar generation framework DreamWaltz-G in Sec. 3.4.

3.1 Preliminary Before delving into our proposed method, we first introduce some concepts that form the basis of our framework.

3D Gaussian Splatting (3DGS) [4] represents a 3D scene through a set of 3D Gaussians G = {Gi | i = 1,...,N}. The geometry of each 3D Gaussian Gi is parameterized by a position (mean) pi ∈ R3×1 and covariance matrix Σi ∈ R3×3 defined in world space:

- 1

- 2(x−pi)TΣ−i 1(x−pi),

Gi(x) = e−

where x is a 3D point in world coordinates. To maintain the position semi-definite property of Σi, a decomposition is used: Σi = RiSiSTi RTi , where the scaling matrix S and the rotation matrix R are parameterized by a 3D vector s and a quaternion q for gradient descent.

To render an image, the 3D Gaussians can be projected to 2D using: Σ′ = JWΣWTJT, where W is a viewing

transformation from world to camera coordinates, and J denotes the Jacobian of the affine approximation of the projective transformation. We use G′i parameterized by Σ′ to represent the 2D Gaussian projected from Gi. Finally, the color c of each pixel x is rendered by alpha blending according to the 3D Gaussians’ depth order 1,...,N:

c(x) =

- i−1
- j=1

N

ciαiG′i(x)

(1 − αjG′j(x)),

i=1

where αi ∈ [0,1] is the opacity of Gi.

Neural Radiance Field (NeRF) [1], [33] is commonly used as the differentiable 3D representation for text-driven 3D generation [20], [52], parameterized by a trainable MLP. For rendering, a batch of rays r(k) = o + kd are sampled based on the camera position o and direction d on a perpixel basis. The MLP takes r(k) as input and predicts density τ and color c. The volume rendering integral is then approximated using numerical quadrature to yield the final color of the rendered pixel:

Nc

Cˆc(r) =

Ωi · (1 − exp(−τiδi))ci,

i=1

where Nc is the number of sampled points on a ray, Ωi = exp(− ij−=11 τjδj) is the accumulated transmittance, and δi is the distance between adjacent sample points.

Diffusion models [38], [69] which have been pre-trained on extensive image-text datasets [18], [35], [70] provide a robust image prior for supervising text-to-3D generation. Diffusion models learn to estimate the denoising score ∇x log pdata(x) by adding noise to clean data x ∼ p(x) (forward process) and learning to reverse the added noise (backward process). Noising the data distribution to isotropic Gaussian is performed in T timesteps, with a pre-defined noising schedule αt ∈ (0,1) and α¯t := ts=1 αs, according to:

###### xt = √α¯tx + √1 − α¯tϵ, where ϵ ∼ N(0,I).

In the training process, the diffusion models learn to estimate the noise by

###### Lt = Ex,ϵ∼N(0,I) ∥ϵϕ (xt,t) − ϵ∥22 .

Once trained, one can estimate x from noisy input and the corresponding noise prediction.

Score Distillation (SDS) [20], [52], [71] is a technique introduced by DreamFusion [20] and extensively employed to distill knowledge from a pre-trained diffusion model ϵϕ into a differentiable 3D representation. For a NeRF model parameterized by θ, its rendering x can be obtained by x = g(θ) where g is a differentiable renderer. SDS calculates the gradients of NeRF parameters θ by,

∂xt ∂x

∂x ∂θ

, (1)

∇θLSDS(ϕ,x) = Et,ϵ w(t)(ϵϕ(xt;y,t) − ϵ)

where w(t) is a weighting function that depends on the timestep t and y denotes the given text prompt.

SMPL-X [32] is a unified parametric 3D human model that extends SMPL [31] with fully articulated hands and an expressive face, containing Nv = 10,475 vertices and Nj = 54 joints. Benefiting from its efficient and expressive

human motion representation ability, SMPL-X has been widely used in human motion-driven tasks [22], [72], [73]. The input parameters for SMPL-X include a 3D body joint and global rotation ξ ∈ R(N

j+1)×3, a body shape β ∈ R300, and a 3D global translation t ∈ R3.

v×3 in canonical pose is constructed by combining the template shape T¯, the shape-dependent deformations BS(β), and the pose-dependent deformations BP(ξ) as,

Formally, a triangulated mesh Tcnl(β,ξ) ∈ RN

###### Tcnl(β,ξ) = T¯ + BS(β) + BP(ξ), (2)

where BP(ξ) is used to relieve artifacts in Linear Blend Skinning (LBS) [74]. Then, the LBS function is employed to transform the canonical mesh Tcnl(β,ξ) into a triangulated mesh Tobs(β,ξ) in the observed pose as,

###### Tobs(β,ξ) = LBS(Tcnl(β,ξ),J (β),ξ,Wlbs), (3) where J (β) ∈ RN

j×3 denotes the corresponding joint positions, and Wlbs ∈ RN

v×Nj is a set of blend weights.

###### 3.2 SkelSD: Skeleton-Guided Score Distillation

Vanilla score distillation methods [20], [21] utilize viewdependent prompt augmentations such as “front view of ...” for diffusion model to provide crucial 3D view-consistent supervision. However, this prompting strategy cannot guarantee precise view consistency, leaving the disparity between the viewpoint of the diffusion model’s supervision image and the 3D avatar’s rendering image unresolved. Such inconsistency causes quality issues for 3D generation, such as blurriness and the Janus (multi-face) problem.

Occlusion-Aware Skeleton Extraction

[Figure 15]

[Figure 16]

[Figure 17]

| |
|---|

| |
|---|

|[Figure 18]|
|---|

|[Figure 19]| |
|---|---|
| | |

| | |
|---|---|
| | |
| | |

[Figure 20]

Visible

Occluded

|SMPL-X| | |
|---|---|---|
| |Condition 𝑐| |

[Figure 21]

|∇𝐿cSDS(𝑥; 𝑦,𝑐)|
|---|

|ControlNet<br><br>[Figure 22]|
|---|

[Figure 23]

###### Naruto Uzumaki

Rendered

|Text Prompt 𝑦|
|---|

Image 𝑥

Noise

Fig. 2. The proposed skeleton-guided score distillation utilizes 2D skeleton images c extracted from SMPL-X [32] to condition controllable 2D diffusion model (where we adopt ControlNet [29]), which enhances the view and pose consistencies between the rendered image x and the SDS supervision ∆LcSDS. In addition, we introduce occlusion culling to eliminate keypoints that are invisible from the current viewpoint, preventing ambiguity for the diffusion model.

Skeleton-guided Score Distillation (SkelSD). Inspired by recent works in controllable image generation [29], [30], we propose SkelSD, which utilizes additional 3D-aware

|Instant-NGP<br><br>[Figure 24]|
|---|

||[Figure 25]|
|---|
<br><br>Trainable Modules|
|---|

##### Hybrid 3D Gaussian

colors, opacities

positions

###### Avatar (H3GA)

[Figure 26]

[Figure 27]

[Figure 28]

|Non-Rigid<br><br>Deformation| |
|---|---|
| | |

|LBS| |
|---|---|
| | |

[Figure 29]

|Splat| |
|---|---|
| | |

[Figure 30]

[Figure 31]

Body Animation

[Figure 32]

Mesh-binding Deformation

Parameterized

[Figure 33]

3D Meshes

Hands & Face

Canonical 3D Gaussians

Observed 3D

Rendered

Animation

(w/o colors and opacities)

Gaussians

Image

- Fig. 3. The proposed hybrid 3D Gaussian avatar representation integrates efficient 3D Gaussian Splatting [4] with neural implicit field (where we adopt Instant-NGP [33]) and parameterized 3D meshes of SMPL-X [32] body parts (e.g., hands and face). Specifically, the canonical 3D Gaussian avatar is jointly represented by unconstrained 3D Gaussians Gu and mesh-binding 3D Gaussians Gm bound to parameterized 3D meshes. The colors and opacities of both Gu and Gm are predicted by the neural implicit field. For animation, Gu and Gm are deformed separately and merged to form observed 3D Gaussians, then splatted to obtain the rendered avatar image.

skeleton images from 3D human template [32] to condition SDS for view/pose-consistent score distillation, as shown in Figure 2. Specifically, the skeleton conditioning image c is injected to Equation 1 for SDS gradients, yielding:

∂x ∂θ

∂xt ∂x

∇θLcSDS(ϕ,x) = Et,ϵ w(t)(ϵϕ(xt;y,t,c) − ϵ)

,

where the conditioning image c can be one or a combination of skeletons, depth maps, normal maps, etc. In practice, we opt for skeletons as the conditioning type because they offer minimal human shape priors, thereby facilitating the generation of complex geometries, as illustrated in Figure 8. In order to acquire 3D-aware skeleton images, we use the parametric 3D human model SMPL-X [32] for skeleton rendering, where the skeleton image’s viewpoint is strictly aligned with the avatar’s rendering viewpoint.

Occlusion Culling. The introduction of 3D-aware conditioning images can enhance the 3D consistency in the SDS optimization process. However, the effectiveness is constrained by the adopted diffusion model [29] on its interpretation of the conditioning images. As shown in Fig. 9 (a), we provide a back-view skeleton map as the conditioning image to ControlNet [29] and perform textto-image generation. However, a frontal face still appears in the generated image. Such defects bring problems such as multiple faces (the Janus problem) and unclear facial features to 3D avatar generation. To this end, we propose to use occlusion culling algorithms [75] in computational graphics to detect whether facial keypoints are visible from the given viewpoint and subsequently remove them from the skeleton map if considered invisible. Body keypoints remain unaltered because they reside in the SMPL-X mesh, and it is difficult to determine whether they are occluded without introducing new priors.

###### 3.3 H3GA: Hybrid 3D Gaussian Avatars

The previous method DreamWaltz [28] utilizes NeRF [1] to represent 3D avatars, which is computationally expensive

and results in extremely slow rendering and animation at high image resolutions (e.g., 1024×1024). To achieve higher training and inference efficiency, we adopt 3D Gaussian Splatting [4] as the representation for 3D avatars.

Specifically for diffusion-guided 3D avatar creation, we review existing 3D Gaussian avatar representations [26], [27] and propose several effective improvements for better generation and animation quality:

- 1) The high variance of score distillation gradients makes optimizing millions of 3D Gaussians challenging, as illustrated in Figure 10. Thus, we use pre-trained Instant-NGP [33] to initialize the 3D Gaussians and to predict the 3D Gaussian properties for stable SDS optimization.
- 2) Considering that existing pre-trained 2D diffusion models struggle to generate intricate hands or control facial expressions, we embed the learnable 3D meshes of SMPL-X body parts (i.e., hands and face) into 3D Gaussians to ensure accurate geometry and animation for these body parts.
- 3) To articulate 3D Gaussians for animation, we bind each 3D Gaussian to the SMPL-X joints by assigning LBS weights and propose a geometry-aware smoothing algorithm based on K-Nearest Neighbors (KNN) for adaptive adjustments.
- 4) We introduce a deformation network conditioned on human pose to predict the pose-dependent variations of 3D Gaussian properties.

These improvements constitute the proposed hybrid 3D Gaussian avatar representation, an overview of which is illustrated in Figure 3.

Formulation. The proposed hybrid 3D Gaussian avatar representation consists of two types of 3D Gaussians: Gavatar = Gu ∪ Gm, where Gu denotes unconstrained 3D Gaussians, and Gm denotes mesh-binding 3D Gaussians.

For unconstrained 3D Gaussians Gu, the initial positions are extracted from a pre-trained NeRF. Specifically, we query

NeRF to obtain the density distribution of a high-resolution 3D grid, and positions where the density exceeds a constant threshold are used as the initial positions pu for Gu. Then, the colors cu and opacities αu of Gu are predicted by:

c,α = NeRF(p). (4)

The scales su and rotations qu of Gu are explicitly initialized following 3DGS [4] rather than being predicted by NeRF.

For mesh-binding 3D Gaussians Gm, we utilize the predefined 3D meshes of the hands and face from SMPL-X and construct mesh-binding 3D Gaussians following SuGaR [76] and GaMeS [77]. Exceptionally, the colors cm and opacities αm of Gm are predicted by NeRF following Equation 4. Besides, we parameterize the pre-defined 3D meshes using the shape parameters β of SMPL-X, which are learnable.

Articulation and Pose Transformation. SMPL-X utilizes linear blend skinning (LBS) [74] for the pose transformation of an articulated human body. This technique transforms the vertices of 3D meshes by blending multiple joint transformations based on LBS weights. Therefore, for mesh-binding 3D Gaussians Gm bound to SMPL-X body parts, we can animate them by transforming the mesh vertices, following Equation 3. For unconstrained 3D Gaussians Gu, the pose transformation involves translating the position p and rotating the quaternion q. We extend the LBS transformation of SMPL-X vertices to unconstrained 3D Gaussians as follows:

###### Gu(ξ) = LBS(Gucnl,J ,ξ,Wlbs), (5)

where Gucnl denotes unconstrained 3D Gaussians in the canonical pose, J represents SMPL-X joint positions, ξ is

the SMPL-X pose, and Wlbs is a set of LBS weights for Gu. The acquisition of LBS weights Wlbs is given in Section 3.4.2.

Non-rigid Deformation. Pose-dependent deformations (i.e., BP(ξ) in Equation 2) allow the SMPL-X model to finely adjust and deform the body surface during pose changes. Still, it struggles to generalize to clothed avatars generated from texts. Thus we introduce a MLP-based deformation network [66] to model pose-dependent deformations for unconstrained 3D Gaussians Gu:

###### (δp,δs,δq) = NRDeform(ξ), (6)

where (δp,δs,δq) represents the offsets of positions, scales, and quaternions of the unconstrained 3D Gaussians Gucnl in the canonical pose. Note that the deformation network is subject-specific and trained from the diffusion guidance.

In addition, for mesh-binding 3D Gaussians Gm, we model pose-dependent deformations following the mesh transformations of SMPL-X as described in Equation 2.

- 3.4 DreamWaltz-G: Learning 3D Gaussian Avatars via Skeleton-guided Score Distillation

Based on the proposed Skeleton-guided Score Distillation and Hybrid 3D Gaussian Avatar Representation, We further introduce a text-driven avatar generation framework: DreamWaltz-G. The framework comprises two training stages: (I) Static NeRF-based Canonical Avatar Learning (Sec. 3.4.1), (II) Deformable 3DGS-based Animatable Avatar Learning (Sec. 3.4.2), as illustrated in Figure 4.

- 3.4.1 Canonical Avatar Learning

In this stage, we employ a static NeRF (implemented with Instant-NGP [33]) as the canonical avatar representation and train it using the skeleton-conditioned ControlNet [29] and the canonical-posed SMPL-X model [32]. In particular, it leverages the SMPL-X model in three ways: (1) pre-training NeRF, (2) providing geometry constraints, and (3) rendering skeleton images to condition ControlNet for 3D-consistent and pose-aligned score distillation.

Pre-training with SMPL-X. To speed up the NeRF optimization and to provide reasonable initial renderings for the diffusion model, we pre-train NeRF based on an SMPLX mesh template. Specifically, we render the silhouette and depth images of NeRF and SMPL-X given a randomly sampled viewpoint, and minimize the MSE loss between the NeRF renderings and the SMPL-X renderings. The NeRF initialization from the human template significantly improves the geometry and the convergence efficiency for subsequent text-specific avatar generation.

Score Distillation in Canonical Pose. Given the target text prompt, we optimize the pre-trained NeRF through skeleton-guided score distillation loss LcnlcSDS in the canonical pose space. We adopt the A-pose as the canonical pose because it best aligns with the diffusion prior and avoids leg overlap. Unlike DreamWaltz [28] using SMPL [31] skeletons as condition images, we employ the more advanced SMPLX [32] skeletons with hand joints and facial landmarks.

Local Geometric Constraints of Body Parts. During NeRF training, we introduce a local geometry loss based on pre-defined meshes of body parts, such as hands and faces. This ensures the trained NeRF is geometrically compatible with mesh-binding 3D Gaussians when serving as 3DGS initialization in subsequent stages. Specifically, we align the NeRF densities τ of local regions with the pre-defined meshes using a margin ranking loss:

Lgeo =

(max(0,τmax − τ(p)))2 if p on mesh (max(0,τ(p) − τmin))2 if p not on mesh,

where p represents 3D points sampled on and near the predefined meshes, τ(p) denotes the densities of 3D points p predicted by NeRF, τmin and τmax are constant hyperparameters. Notably, Latent-NeRF [71] also introduces shape guidance to constrain NeRF geometry given a mesh sketch. Although both methods use pre-defined meshes as geometry guidance for NeRF optimization, the difference lies in their aim to provide a coarse geometry alignment, whereas we enforce strictly consistent geometries.

Overall Objective. To learn a canonical 3D avatar given text prompts, we optimize the NeRF-based static avatar representation using:

Lcnltotal = LcnlcSDS + λgeoLgeo,

where LcnlcSDS denotes the conditional SDS loss with canonical skeleton images as conditions, and λgeo = 1.0 is a balanced weight of the local geometry constraint.

- 3.4.2 Animatable Avatar Learning In this stage, we initialize the proposed hybrid 3D Gaussians Gavatar as the animatable avatar representation and optimize

||[Figure 34]|
|---|
<br><br>|[Figure 35]|
|---|
<br><br>|[Figure 36]|
|---|
<br><br>|[Figure 37]|
|---|
<br><br>Stage I: Canonical Avatar Learning<br><br>|𝐿geo|
|---|
<br><br>|Instant-NGP<br><br>[Figure 38]<br><br>|render|
|---|---|
| | |
<br><br>|ControlNet<br><br>[Figure 39]|
|---|
<br><br>condition<br><br>|Canonical<br><br>SMPL-X<br><br>[Figure 40]|render|
|---|---|
| | |
<br><br>|𝐿cSDS|
|---|
<br><br>Rendered Image<br><br>Skeleton Image<br><br>add noise<br><br>Stage II: Animatable Avatar Learning<br><br>|Random Pose & Expr.<br><br>{𝜃body,𝜃hand,𝛽expr}<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]|
|---|
<br><br>[Figure 48]<br><br>[Figure 49]<br><br>|H3GA<br><br>[Figure 50]|render|
|---|---|
| | |
<br><br>|ControlNet<br><br>[Figure 51]|
|---|
<br><br>condition<br><br>|SMPL-X<br><br>[Figure 52]|
|---|
<br><br>render<br><br>|𝐿cSDS|
|---|
<br><br>3DGS-based Animatable Avatar Rendered Image<br><br>Human Template Skeleton Image<br><br>add noise<br><br>|Trained Instant-NGP|
|---|
<br><br>NeRF-based<br><br>Canonical Avatar|
|---|

- Fig. 4. The proposed animatable 3D avatar generation framework DreamWaltz-G consists of two training stages: (I) Canonical Avatar Learning and (II) Animatable Avatar Learning. In Stage I, We adopt the static Instant-NGP [33] as canonical avatar representation. For each iteration, we

extract a skeleton image from canonical SMPL-X [32] to condition ControlNet [29]. Skeleton-conditioned score distillation loss LcSDS is used as a training objective to learn the canonical avatar. In Stage II, the proposed animatable avatar representation H3GA is first initialized with the trained

Instant-NGP from Stage I and then optimized by LcSDS. Unlike Stage I, which uses a fixed canonical pose, in Stage II, we randomly sample plausible human poses and expressions in each iteration to drive H3GA and SMPL-X, encouraging avatar learning across different motions.

it in random pose space using score distillation conditioned on SMPL-X skeletons.

LBS Weight Initialization with SMPL-X. Assigning LBS weights from SMPL-X vertices to each unconstrained 3D Gaussian G ∈ Gu is necessary for articulation and pose transformation. A naive implementation is mapping LBS weights based on nearest vertex criteria; however, this method cannot handle the geometric mismatches between SMPL-X and the generated avatars, leading to erroneous skeletal binding and distortions, as demonstrated in Figure 14. To address this, we propose using a geometryaware KNN smoothing algorithm to adjust the assigned LBS weights of the 3D Gaussians adaptively. Specifically, for a 3D Gaussian G ∈ Gu, its initial LBS weights Wlbs(0) can be derived from the nearest vertex in SMPL-X. Next, we update Wlbs iteratively by weighted aggregation of the LBS weights Wlbs,k of the Klbs nearest 3D Gaussians:

Klbs

Zlbs dng,k · dnv,k

Wlbs(i),k, (7)

Wlbs(i+1) =

k=1

where i ∈ {0,1,...,Nlbs} denotes the current iteration step, Zlbs represents the normalization constant ensuring Zlbs K

k=1 (dng,k · dnv,k)−1 = 1, dng,k is the squared distance from the k-th nearest 3D Gaussian Gk to the current 3D Gaussian G, and dnv,k is the squared distance from Gk to its nearest vertex in SMPL-X. For clarity, d−ng1,k reflects the contribution of Gk to G, while d−nv1,k indicates the confidence of the initial LBS weights of Gk.

lbs

###### Score Distillation in Arbitrary Poses and Expressions.

Skeleton-guided score distillation LarbcSDS in arbitrary poses helps to enhance visual quality and mitigate motion artifacts

in novel poses. The previous work DreamWaltz [28] samples random poses using the off-the-shelf VPoser [32], which is a variational autoencoder that learns a latent representation of human pose. However, optimizing directly in arbitrary pose spaces may be challenging to converge, leading to quality issues such as blurring. Therefore, we adopt a curriculum learning strategy from simple to difficult tasks, starting with sampling various canonical poses (such as A-pose, Tpose, and Y-pose), followed by sampling random poses from VPoser. Note that VPoser does not encompass hand poses

and facial expressions. To obtain random hand poses and facial expressions, we randomly sample PCA coefficients from a Gaussian distribution and use the SMPL-X prior to compute corresponding pose and shape parameters.

Overall Objective. To learn an animatable 3D avatar given text prompts, we optimize the hybrid 3DGS-based dynamic avatar representation using LarbcSDS only.

###### 4 EXPERIMENTS

4.1 Implementation Details DreamWaltz-G is implemented in PyTorch and can be trained and evaluated on a single NVIDIA L40S GPU.

For the Canonical Avatar Learning stage, we employ Instant-NGP [33] as the static 3D avatar representation. We optimize it for 15,000 iterations, which takes about one hour. We adopt a progressive resolution sampling strategy for efficient optimization, where the rendering resolution increases from 64×64 to 512×512 as iterations progress. More details on NeRF optimization, such as the optimizer and learning rate, are consistent with DreamWaltz [28].

For the Animatable Avatar Learning stage, we use the proposed H3GA as the dynamic 3D avatar representation, which is trained for 15,000 iterations, and the rendering resolution is maintained at 512×512. To optimize 3D Gaussian attributes, we adhere to the original implementation of 3DGS [4]. However, we do not use the densification strategy for two reasons: (i) The high variance of SDS gradients makes gradient-based densification unstable; (ii) The initialization based on a trained NeRF can provide accurate and quantitative 3D Gaussians.

Diffusion Guidance. We use Stable-Diffusion-v1.5 [19] and ControlNet-v1.1-openpose [29] to provide SDS guidance for both training stages. We randomly sample the timestep from a uniform distribution of [0.02,0.98], and the classifierfree guidance scale is set to 50.0. The weight term w(t) for SDS loss is set to 1.0. The conditioning scale for ControlNet is set to 1.0 by default. To further improve 3D consistency and visual quality, both view-dependent text augmentation [20] and negative prompts are used.

Camera Sampling. For each iteration, the camera view is randomly sampled in spherical coordinates, where the

TABLE 2 User preference studies. We report the preference percentages (%) of our method over existing state-of-the-art methods in terms of geometric quality, appearance quality, and consistency with the text prompts.

|Methods<br><br>|Geometry Quality|Appearance Quality<br><br>|Text Consistency|
|---|---|---|---|
|Ours vs. DreamWaltz [28] Ours vs. DreamHuman [25] Ours vs. TADA [24] Ours vs. GAvatar [26] Ours vs. HumanGaussian [27]|84.93 82.61 70.27 82.05 70.31<br><br>|86.30 86.96 77.03 76.92 75.00|78.08 84.78 66.22 79.49 76.56<br><br>|

radius, azimuth, elevation, and FoV are uniformly sampled from [1.0,2.0], [0,360], [60,120], and [40,70], respectively. The camera focus strategy is also employed, with a 0.2 probability of focusing on the face of the 3D avatar to enhance facial details. Additionally, we empirically find that horizontal camera jitter during training helps improve the visual quality of the foot region.

Motion Sequences. To create animation demonstrations, we utilize SMPL-X motion sequences from 3DPW [78], AIST++ [79], Motion-X [80], and TalkSHOW [81] datasets to animate avatars. SMPL-X motion sequences extracted from in-the-wild videos are also used.

###### 4.2 Comparisons

We provide both qualitative and quantitative results of our DreamWaltz-G compared to existing text-driven 3D avatar generation methods, including DreamWaltz [28], DreamHuman [25], TADA [24], HumanGaussian [27], and GAvatar [26].

Qualitative Results of Canonical Avatars. We present the results of canonical avatars, as shown in Figure 5. Compared to existing methods, our approach achieves highdefinition and realistic appearances, alleviating blurriness and over-saturation issues. Additionally, our approach can generate accurate hand and facial shapes by leveraging the geometric priors of predefined meshes, addressing the diffusion model’s difficulty in generating detailed human body parts. We provide more examples of canonical 3D avatars generated by our method in Figure 6.

Qualitative Results of Animatable Avatars. We demonstrate the animation results of our method compared to HumanGaussian [27] and TADA [24], as shown in Figure 7. The SMPL-X motion sequences from the AIST++ dance dataset [79] are used to animate the generated avatars. Compared to existing competing methods, our approach achieves clearer hand motions and higher-fidelity animation quality. In comparison to HumanGaussian, which is also based on 3DGS [4], we effectively avoid sharp artifacts caused by the incorrect driving of 3D Gaussians. More examples of avatar animations can be seen in Figure 6 and Figure 16.

User Studies. To quantitatively evaluate the quality of the generated 3D avatars compared to existing methods, we conducted a A/B user preference study based on 24 text prompts released by GAvatar [26]. Twenty participants are asked to view 3D avatars generated by our method and one of the competing methods and then choose the better method based on (1) geometric quality, (2) appearance quality, and (3) consistency with the text prompts. As reported

in Table 2, the participants favor 3D avatars generated by our method across all evaluation criteria.

4.3 Ablation and Analysis We perform a comprehensive ablation analysis to demonstrate the effectiveness of the proposed improvements.

Effectiveness of Skeleton Guidance. We visualize the SDS gradients and generated images in Figure 8 to illustrate the advantages of skeleton guidance compared to text-only guidance and depth guidance. It is evident that depth and skeleton images from human templates offer more informative guidance than text alone. However, the strong contour priors in depth images cause the SDS gradients to conform tightly to the avatar’s skin, leading to a lack of complex appearances (e.g., the disappearance of Superman’s cape in the second row of Figure 8). On the other hand, skeleton images, as adopted by DreamWaltz-G, provide both informative and flexible supervision, accurately capturing the avatars’ poses and intricate shapes.

Ablation Studies on Occlusion Culling. Occlusion culling is crucial for resolving view ambiguity both for skeleton-conditioned 2D and 3D generation, as shown in

- Figure 9. Limited by the view-aware capability, ControlNet [29] fails to generate the back-view image of a character even with view-dependent text and skeleton prompts, as shown in Figure 9(a). The introduction of occlusion culling eliminates the ambiguity of skeleton conditions and helps ControlNet to generate correct views. Similar effects can be observed in text-to-3D avatar generation. As shown in Figure 9(b), The Janus (multi-face) problem is solved by introducing occlusion culling to the rendering process from 3D SMPL-X to the 2D skeleton images.

Ablation Studies on Hybrid 3D Gaussian Avatars. The proposed 3D avatar representation, H3GA, incorporates several improvements to accommodate SDS optimization and enable expressive avatar animation. We analyze the effects of these improvements individually, as shown in

- Figure 10. Specifically, “NeRF Initialization” provides a well-structured point cloud to initialize the 3D Gaussians, facilitating the capture of complex geometries that differ from SMPL-X templates. “NeRF Encoding” utilizes multiresolution hash grids [33] and MLPs to predict 3D Gaussian attributes, resulting in more stable SDS optimization and avoiding high-frequency noise in textures.

For body parts that are challenging to generate and animate (e.g., hands and face), we adopt a “Mesh Binding” strategy. This strategy binds the corresponding 3D Gaussians to the meshes of SMPL-X body parts, achieving sharp and joint-aligned geometries. Note that these meshbinding body parts are parameterized by SMPL-X shape

[Figure 53]

9

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

| |
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

| |
|---|

|[Figure 65]|
|---|

###### Amanwearingawhite

###### tanktopandshorts.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

DreamHuman DreamWaltz

TADA

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

|[Figure 75]|
|---|

|[Figure 76]|
|---|

| |
|---|

|[Figure 77]|
|---|

| |
|---|

|[Figure 78]|
|---|

[Figure 79]

HumanGaussian GAvatar DreamWaltz-G (Ours)

[Figure 80]

[Figure 81]

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

###### Anelderlymanwearingabeigesuit.

| |
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

[Figure 90]

[Figure 91]

[Figure 92]

DreamHuman

TADA

DreamWaltz

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

|[Figure 97]|
|---|

| |
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

| |
|---|

|[Figure 100]|
|---|

HumanGaussian GAvatar DreamWaltz-G (Ours)

- Fig. 5. Qualitative results of canonical avatars compared to existing text-driven 3D avatar generation methods: DreamWaltz [28], DreamHuman [25], TADA [24], GAvatar [26], HumanGaussian [27]. The text prompts used are listed on the left.

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

An elderly woman with a sunhat. A medieval European king. Mulan.

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Jane Goodall. A professional boxer. A Bedouin dressed in white. A farmer.

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

An elderly man wearing

A gardener in overalls

An Asian woman in a

A black female surgeon.

a beige suit.

and a wide-brimmed hat.

leather jacket.

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

A football player.

A clown.

A Viking.

A karate master.

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Harley Quinn.

Rapunzel in Tangled. Spiderman.

Goku.

- Fig. 6. More examples of 3D avatars and their animations produced by our approach. The text prompts used are listed below.

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

AViking AIST++Dance

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

RapunzelinTangled

HumanGaussian TADA

DreamWaltz-G (Ours)

- Fig. 7. Qualitative results of animatable avatars compared to existing 3d avatar generation and animation methods: HumanGaussian [27] and TADA [24]. Compared to competing methods, our approach achieves clearer hand motions and higher-fidelity animation quality. In comparison to HumanGaussian, which is also based on 3DGS [4], we effectively avoid sharp artifacts caused by the incorrect driving of 3D Gaussians.

|[Figure 157]|
|---|

|[Figure 158]|
|---|

|[Figure 159]|
|---|

|[Figure 160]|[Figure 161]|
|---|---|
| | |

|[Figure 162]|[Figure 163]|
|---|---|
| | |

|[Figure 164]<br><br>[Figure 165]| |
|---|---|
| | |

|[Figure 166]|[Figure 167]|
|---|---|
| | |

|[Figure 168]<br><br>[Figure 169]| |
|---|---|
| | |

|[Figure 170]|[Figure 171]|
|---|---|
| | |

One-step Denoised Images

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|[Figure 176]|
|---|---|
| | |

|[Figure 177]<br><br>[Figure 178]| |
|---|---|
| | |

|[Figure 179]<br><br>[Figure 180]| |
|---|---|
| | |

|[Figure 181]<br><br>[Figure 182]| |
|---|---|
| | |

|[Figure 183]<br><br>[Figure 184]| |
|---|---|
| | |

|[Figure 185]|[Figure 186]|
|---|---|
| | |

SDS Gradients

SkeletonDepth ControlNet

Stable

Diffusion

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]|[Figure 191]|
|---|---|
| | |

|[Figure 192]|[Figure 193]|
|---|---|
| | |

|[Figure 194]<br><br>[Figure 195]| |
|---|---|
| | |

|[Figure 196]|[Figure 197]|
|---|---|
| | |

|[Figure 198]<br><br>[Figure 199]| |
|---|---|
| | |

|[Figure 200]|[Figure 201]|
|---|---|
| | |

Generated Images

TextOnly

- Fig. 8. Visualization of SDS gradients and generated images under different guidance conditions. The results in the first row are conditioned only on text. In contrast, the second and third rows are conditioned on additional depth and skeleton images, respectively, as indicated in the upper left corner of each visualization. These results are based on the text prompt “superman”. It is evident that skeleton conditions, as adopted by our DreamWaltz-G, provide more informative supervision than text-only conditions. Skeleton conditions are also less restrictive than depth conditions, successfully avoiding the loss of complex appearances, such as the disappearance of Superman’s cape.

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

(a) Text-to-2D results, produced by ControlNet. (b) Text-to-3D results, rendered from side and back views.

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

w/o Occlusion Culling w/ Occlusion Culling

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

w/o Occlusion Culling w/ Occlusion Culling

Text Prompt: backview of Mulan Text Prompt: Hatsune Miku

| |
|---|

| |
|---|

| |
|---|

|[Figure 210]|
|---|

|[Figure 211]|
|---|

|[Figure 212]|
|---|

|[Figure 213]|
|---|

- Fig. 9. Ablation studies on occlusion culling. We employ occlusion culling to refine skeleton condition images by removing invisible human keypoints, such as the eyes and nose in the back view. It helps (a) ControlNet [29] to generate the character’s back view correctly, and (b) text-to-3D avatar generation to resolve the multi-face problem, as highlighted by the bounding boxes.

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 221]|
|---|

|[Figure 222]|
|---|

|[Figure 223]|
|---|

| |
|---|

| |
|---|

Baseline + NeRF Initialization + NeRF Encoding

+ Mesh Binding (Ours)

- Fig. 10. Ablation studies on the proposed Hybrid 3D Gaussian Avatar representation, which incorporates several improvements to accommodate SDS optimization and enable expressive avatar animation. Specifically, “NeRF Initialization” provides a well-structured point cloud to initialize the 3D Gaussians, facilitating the capture of complex geometries. “NeRF Encoding” utilizes Instant-NGP [33] to predict 3D Gaussian attributes, resulting in more stable SDS optimization and avoiding high-frequency noise in textures. For intricate body parts like hands, we adopt a “Mesh Binding” strategy, which binds the corresponding 3D Gaussians to the SMPL-X body parts, achieving sharp and joint-aligned geometries.

w/o Learnable Hand Shapes w/ Learnable Hand Shapes

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

| |
|---|

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

| |
|---|

- Fig. 11. Ablation studies on learnable shape parameters (e.g., βhand of SMPL-X [32]) for mesh-binding 3D Gaussian body parts. We use the hands of “Princess Elsa in Frozen” as an example to demonstrate. By optimizing the hand shape parameters of mesh-binding 3D Gaussians, slimmer hands that match Elsa’s characteristics can be generated.

[Figure 242]

[Figure 243]

w/o Local Geometric Loss w/ Local Geometric Loss

[Figure 244]

[Figure 245]

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

- Fig. 12. Ablation studies on local geometric constraints. Without

[Figure 246]

[Figure 247]

| |
|---|

| |
|---|

|[Figure 248]|
|---|

|[Figure 249]|
|---|

w/o AAL

w/ AAL

Fig. 13. Ablation studies on Animatable Avatar Learning (AAL), which is the Stage II of DreamWaltz-G. For “w/o AAL”, we train for the same iterations as “w/ AAL” but use a fixed canonical pose to ensure a fair comparison. It can be observed that the introduction of AAL fixes texture information for areas not visible in the canonical pose. Besides, it reduces animation artifacts caused by incorrect skeleton binding.

parameters and are trainable. As shown in Figure 11, hands that conform to the character’s features can be obtained by optimizing the SMPL-X hand shape parameters.

Ablation Studies on Local Geometric Constraints. The local geometric constraints Lgeo are introduced during canonical NeRF training to maintain the geometric structures of intricate body parts, such as hands and faces. As shown in Figure 12, without the local geometric loss, the generated avatar’s hands appear in a clenched fist state, exhibiting unclear geometric structures and difficulties with rigging and animation. Introducing the local geometric loss ensures that the hand structure is accurately aligned with canonical SMPL-X, avoiding erroneous geometries and facilitating subsequent hand animation.

Ablation Studies on DreamWaltz-G. The proposed avatar generation framework, DreamWaltz-G, consists of two training stages: Canonical Avatar Learning (CAL), and Animatable Avatar Learning (AAL). The CAL stage aims to provide a good NeRF initialization for H3GA, the effectiveness of which is validated as shown in Figure 10. The AAL stage aims to learn the appearance and geometry of the 3D

the local geometric loss Lgeo, the generated avatar’s hands appear in a clenched fist state (highlighted by dashed boxes), exhibiting unclear geometric structures. The introduction of Lgeo ensures that the hand structure is accurately aligned with canonical SMPL-X (highlighted by dashed boxes), avoiding erroneous geometries and facilitating subsequent rigging and hand animation.

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

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

w/o KNN Smooth

w/ KNN Smooth Canonical Avatar

w/o KNN Smooth

w/ KNN Smooth

(Baseline)

(Ours)

(Baseline)

(Ours)

Canonical Avatar

(a) Continuous deformation of complex clothing. (b) Accurate skeleton binding.

- Fig. 14. Ablation studies on KNN smoothing for LBS weight initialization. The proposed geometry-aware KNN Smoothing algorithm refines the 3D Gaussians’ initial LBS weights (representing the association of each 3D Gaussian to body joints). Compared to the baseline that assigns LBS weights based solely on the nearest neighbor criterion, the proposed algorithm enables (a) continuous deformation of complex clothing, e.g., the stretching of the chef’s apron; (b) accurate skeleton binding, for example, the hat hanging from Woody’s waist is not affected by arm movements.

[Figure 256]

[Figure 257]

Fat “Joker” Thin SMPL-X Thin “Joker”

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

Template

Fat SMPL-X

Template

(a) Shape Control (b) Shape Editing

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

- Fig. 15. Application: Shape Control and Editing. Our method enables (a) training-time shape control by modifying the SMPL-X template and (b) inference-time shape editing during inference by explicitly adjusting the 3D Gaussians. Both shape control and editing are compatible with the SMPL-X shape parameters β, allowing users to simply adjust β to achieve the desired 3D shape.

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

“Kobe Bryant” + TalkShow (Conan) “A chef dressed in white” + TalkShow (Chemistry)

[Figure 274]

[Figure 275]

- Fig. 16. Application: Talking 3D Avatars. Benefiting from the proposed expressive H3GA representation, our method can learn animatable 3D avatars from 2D diffusion priors while preserving the fine details of hands and faces. This allows us to create more expressive 3D avatar animations like talking 3D avatars.

avatar in a random pose space. As shown in Figure 13, the introduction of AAL fixes texture information for areas not visible in the canonical pose and reduces animation artifacts caused by incorrect skeleton binding.

Ablation Studies on KNN Smoothing for LBS Weight Initialization. We propose a geometry-aware KNN Smoothing algorithm to refine the initial LBS weights (representing the association of each 3D Gaussian to body joints), bringing various improvements in avatar rigging and animation. As shown in Figure 14, the proposed KNN smoothing algorithm enables: (a) continuous deformation of complex clothing, e.g., the stretching of a dress; (b) accurate skeleton binding, which should be geometry-aware rather than based

solely on the nearest neighbor criterion.

###### 4.4 Applications

We explore practical applications of our method, including: shape control and editing, talking 3D avatars, human video reenactment, and multi-subject 3D scene composition.

Shape Control and Editing. Our method utilizes the SMPL-X template to provide skeleton guidance for 3D avatar creation. By adjusting the shape parameters of the SMPL-X template, the shape of the generated 3D avatar can be controlled, as shown in Figure 15(a). However, this shape control requires re-training, which leads to inefficiency and

|[Figure 276]|[Figure 277]|[Figure 278]|
|---|---|---|

|[Figure 279]|[Figure 280]|[Figure 281]|
|---|---|---|

Original Video Frame SMPL-X Motion Reenacted Video Frame

- Fig. 17. Application: Human Video Reenactment. Combined with 3D human pose estimation and video inpainting techniques, the 3D avatars generated by our method can be projected onto 2D human videos. This integration allows for seamless blending of animated 3D avatars with real-world footage, enhancing the realism and interactivity of the reenacted scenes.

[Figure 282]

[Figure 283]

- Fig. 18. Application: Multi-subject Scene Composition. The generated 3D avatars can be seamlessly integrated with existing 3D assets. The presented 3D environments are from the Mip-NeRF 360 dataset [82] and reconstructed by vanilla 3D Gaussian Splatting [4].

appearance randomness. Thanks to the explicit 3D avatar representation, our method can also achieve shape editing by adjusting the 3D Gaussians. Compared to shape control, shape editing is real-time, interactive, and able to maintain a consistent appearance, as shown in Figure 15(b).

Talking 3D Avatars. The proposed H3GA representation enables the modeling of animatable 3D avatars from 2D diffusion priors while preserving the fine details of hands and faces. This allows us to create more expressive 3D avatar animations, for example, talking 3D avatars. As shown in Figure 16, the results exhibit realistic appearances, intricate geometries, and accurate hand and face animations.

Human Video Reenactment. Combined with 3D human pose estimation [80] and video inpainting techniques, the 3D avatars generated by our method can be projected onto 2D human videos, as shown in Figure 17. This integration

allows for seamless blending of animated 3D avatars with real-world footage, enhancing the realism and interactivity of the reenacted scenes.

Multi-subject Scene Composition. The generated 3D avatars can be integrated with existing 3D assets into the same scene. As shown in Figure 18, we place the animated 3D avatars “Kobe Bryant” and “a chef dressed in white” into 3D scenes, seamlessly integrating the avatars into the environment.

###### 5 CONCLUSIONS

We introduce DreamWaltz-G, a novel learning framework for animatable 3D avatar generation from texts. At the core of this framework are skeleton-guided score distillation and hybrid 3D Gaussian avatar representation. Specifically, we

leverage the skeleton priors from the human parametric model [32] to guide the score distillation process, providing 3D-consistent and pose-aligned supervision for highquality avatar generation. The hybrid 3D Gaussian representation builds on the efficiency of 3D Gaussian splatting [4], combining NeRF [1] and 3D meshes [76] to accommodate SDS optimization and enable expressive animations. Extensive experiments demonstrate that DreamWaltz-G is effective and outperforms existing text-to-3D avatar generation methods in both visual quality and animation. Benefiting from DreamWaltz-G, we could unleash our imagination and enable a wide range of avatar applications.

Similar to previous 3D generation methods [20], [21], [28], DreamWaltz-G generates 3D avatars through score distillation [20]. Leveraging more powerful foundational models [45], [46] and advanced score distillation techniques [55], [56] can further enhance the generation quality and efficiency. Additionally, the generated 3D avatars still lack hierarchical semantic structures and physical properties, which will be a direction worth exploring in future work.

###### REFERENCES

- [1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis,” Communications of the ACM, vol. 65, no. 1, pp. 99–106, 2021. 1, 2, 4, 5, 15
- [2] P. Wang, L. Liu, Y. Liu, C. Theobalt, T. Komura, and W. Wang, “NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction,” Advances in Neural Information Processing Systems, vol. 34, pp. 27171–27183, 2021. 1, 2
- [3] T. Shen, J. Gao, K. Yin, M.-Y. Liu, and S. Fidler, “Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis,” in Advances in Neural Information Processing Systems,

2021. 1

- [4] B. Kerbl, G. Kopanas, T. Leimkuhler,¨ and G. Drettakis, “3D Gaussian Splatting for Real-Time Radiance Field Rendering,” ACM Transactions on Graphics, vol. 42, no. 4, July 2023. 1, 2, 3, 5, 6, 7, 8, 11, 14, 15
- [5] S. Saito, Z. Huang, R. Natsume, S. Morishima, A. Kanazawa, and H. Li, “Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 2304–2314. 1
- [6] Y. Xiu, J. Yang, D. Tzionas, and M. J. Black, “Icon: Implicit clothed humans obtained from normals,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2022, pp. 13286–13296. 1
- [7] Y. Xiu, J. Yang, X. Cao, D. Tzionas, and M. J. Black, “Econ: Explicit clothed humans optimized via normal integration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 512–523. 1
- [8] C.-Y. Weng, P. P. Srinivasan, B. Curless, and I. KemelmacherShlizerman, “Personnerf: Personalized reconstruction from photo collections,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 524–533. 1
- [9] J. Wang, J. S. Yoon, T. Y. Wang, K. K. Singh, and U. Neumann, “Complete 3d human reconstruction from a single incomplete image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8748–8758. 1
- [10] C.-Y. Weng, B. Curless, P. P. Srinivasan, J. T. Barron, and

I. Kemelmacher-Shlizerman, “Humannerf: Free-viewpoint rendering of moving people from monocular video,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16210–16220. 1

- [11] W. Jiang, K. M. Yi, G. Samei, O. Tuzel, and A. Ranjan, “Neuman: Neural human radiance field from a single video,” in Proceedings of the European conference on computer vision (ECCV). Springer, 2022, pp. 402–418. 1
- [12] Z. Yu, W. Cheng, X. Liu, W. Wu, and K.-Y. Lin, “MonoHuman: Animatable Human Neural Field from Monocular Video,” arXiv preprint arXiv:2304.02001, 2023. 1

- [13] Z. Qian, S. Wang, M. Mihajlovic, A. Geiger, and S. Tang, “3DGSAvatar: Animatable Avatars via Deformable 3D Gaussian Splatting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 1
- [14] W. Zielonka, T. Bagautdinov, S. Saito, M. Zollh¨ofer, J. Thies, and J. Romero, “Drivable 3D Gaussian Avatars,” arXiv preprint arXiv:2311.08581, 2023. 1, 3
- [15] F. Zhao, Y. Jiang, K. Yao, J. Zhang, L. Wang, H. Dai, Y. Zhong, Y. Zhang, M. Wu, L. Xu et al., “Human Performance Modeling and Rendering via Neural Animated Mesh,” ACM Transactions on Graphics (TOG), vol. 41, no. 6, pp. 1–17, 2022. 1
- [16] Y. Jiang, Q. Liao, X. Li, L. Ma, Q. Zhang, C. Zhang, Z. Lu, and Y. Shan, “UV Gaussians: Joint Learning of Mesh Deformation and Gaussian Textures for Human Avatar Modeling,” arXiv preprint arXiv:2403.11589, 2024. 1, 3
- [17] Y. Zheng, Q. Zhao, G. Yang, W. Yifan, D. Xiang, F. Dubost, D. Lagun, T. Beeler, F. Tombari, L. Guibas et al., “PhysAvatar: Learning the Physics of Dressed 3D Avatars from Visual Observations,” arXiv preprint arXiv:2404.04421, 2024. 1
- [18] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical Text-Conditional Image Generation with CLIP Latents,” arXiv preprint arXiv:2204.06125, 2022. 1, 2, 4
- [19] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-Resolution Image Synthesis with Latent Diffusion Models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10684–10695. 1, 2, 7
- [20] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “DreamFusion: Text-to-3D using 2D Diffusion,” arXiv preprint arXiv:2209.14988,

2022. 1, 3, 4, 7, 15

- [21] H. Wang, X. Du, J. Li, R. A. Yeh, and G. Shakhnarovich, “Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation,” arXiv preprint arXiv:2212.00774, 2022. 1, 3, 4, 15
- [22] F. Hong, M. Zhang, L. Pan, Z. Cai, L. Yang, and Z. Liu, “AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars,” ACM Transactions on Graphics (TOG), vol. 41, no. 4, pp. 1–19, 2022. 1, 3, 4
- [23] R. Jiang, C. Wang, J. Zhang, M. Chai, M. He, D. Chen, and J. Liao, “AvatarCraft: Transforming Text into Neural Human Avatars with Parameterized Shape and Pose Control,” arXiv preprint arXiv:2303.17606, 2023. 1, 3
- [24] T. Liao, H. Yi, Y. Xiu, J. Tang, Y. Huang, J. Thies, and M. J. Black, “TADA! Text to Animatable Digital Avatars,” in International Conference on 3D Vision (3DV), 2024. 1, 3, 8, 9, 11
- [25] N. Kolotouros, T. Alldieck, A. Zanfir, E. Bazavan, M. Fieraru, and C. Sminchisescu, “DreamHuman: Animatable 3D Avatars from Text,” Advances in Neural Information Processing Systems, vol. 36,

2024. 1, 3, 8, 9

- [26] Y. Yuan, X. Li, Y. Huang, S. De Mello, K. Nagano, J. Kautz, and U. Iqbal, “GAvatar: Animatable 3D Gaussian Avatars with Implicit Mesh Learning,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 1, 3, 5, 8, 9
- [27] X. Liu, X. Zhan, J. Tang, Y. Shan, G. Zeng, D. Lin, X. Liu, and Z. Liu, “HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6646–6657. 1, 3, 5, 8, 9, 11
- [28] Y. Huang, J. Wang, A. Zeng, H. Cao, X. Qi, Y. Shi, Z.-J. Zha, and L. Zhang, “DreamWaltz: Make a Scene with Complex 3D Animatable Avatars,” in Advances in Neural Information Processing Systems, 2023. 1, 2, 3, 5, 6, 7, 8, 9, 15
- [29] L. Zhang and M. Agrawala, “Adding Conditional Control to Text-to-Image Diffusion Models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 1, 3, 4, 5, 6, 7, 8, 11
- [30] X. Ju, A. Zeng, C. Zhao, J. Wang, L. Zhang, and Q. Xu, “HumanSD: A Native Skeleton-Guided Diffusion Model for Human Image Generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 1, 3, 4
- [31] M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, and M. J. Black, “SMPL: a skinned multi-person linear mode,” ACM transactions on graphics (TOG), vol. 34, no. 6, pp. 1–16, 2015. 1, 2, 3, 4, 6
- [32] G. Pavlakos, V. Choutas, N. Ghorbani, T. Bolkart, A. A. Osman, D. Tzionas, and M. J. Black, “Expressive body capture: 3d hands, face, and body from a single image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 10975–10985. 1, 2, 4, 5, 6, 7, 12, 15

- [33] T. Muller,¨ A. Evans, C. Schied, and A. Keller, “Instant Neural Graphics Primitives with a Multiresolution Hash Encoding,” ACM Transactions on Graphics (ToG), vol. 41, no. 4, pp. 1–15, 2022. 2, 4, 5, 6, 7, 8, 12
- [34] A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen, “GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models,” arXiv preprint arXiv:2112.10741, 2021. 2
- [35] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. Denton, S. K. S. Ghasemipour, B. K. Ayan, S. S. Mahdavi, R. G. Lopes et al., “Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding,” arXiv preprint arXiv:2205.11487, 2022. 2, 4
- [36] P. Dhariwal and A. Nichol, “Diffusion Models Beat GANs on Image Synthesis,” Advances in Neural Information Processing Systems, vol. 34, pp. 8780–8794, 2021. 3
- [37] J. Song, C. Meng, and S. Ermon, “Denoising Diffusion Implicit Models,” in International Conference on Learning Representations,

2021. 3

- [38] A. Q. Nichol and P. Dhariwal, “Improved Denoising Diffusion Probabilistic Models,” in International Conference on Machine Learning. PMLR, 2021, pp. 8162–8171. 3, 4
- [39] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman et al., “LAION-5B: An open large-scale dataset for training next generation image-text models,” arXiv preprint arXiv:2210.08402, 2022. 3
- [40] P. Sharma, N. Ding, S. Goodman, and R. Soricut, “Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2018, pp. 2556–2565. 3
- [41] S. Changpinyo, P. Sharma, N. Ding, and R. Soricut, “Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 3558–

3568. 3

- [42] L. Huang, D. Chen, Y. Liu, Y. Shen, D. Zhao, and J. Zhou, “Composer: Creative and controllable image synthesis with composable conditions,” in International Conference on Machine Learning, 2023. 3
- [43] J. Xiao, K. Zhu, H. Zhang, Z. Liu, Y. Shen, Z. Yang, R. Feng, Y. Liu, X. Fu, and Z.-J. Zha, “CCM: Real-Time Controllable Visual Content Creation Using Text-to-Image Consistency Models,” in International Conference on Machine Learning, 2024. 3
- [44] W. Peebles and S. Xie, “Scalable Diffusion Models with Transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4195–4205. 3
- [45] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Muller,¨ J. Penna, and R. Rombach, “SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis,” arXiv preprint arXiv:2307.01952, 2023. 3, 15
- [46] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller,¨ H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling Rectified Flow Transformers for High-Resolution Image Synthesis,” in International Conference on Machine Learning, 2024. 3, 15
- [47] X. Liu, J. Ren, A. Siarohin, I. Skorokhodov, Y. Li, D. Lin, X. Liu, Z. Liu, and S. Tulyakov, “HyperHuman: Hyper-Realistic Human Generation with Latent Structural Diffusion,” in International Conference on Learning Representations, 2024. 3
- [48] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani, A. Kembhavi, and A. Farhadi, “Objaverse: A Universe of Annotated 3D Objects,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 13142–13153. 3
- [49] A. Jain, B. Mildenhall, J. T. Barron, P. Abbeel, and B. Poole, “Zero-Shot Text-Guided Object Generation With Dream Fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 867–876. 3
- [50] N. Mohammad Khalid, T. Xie, E. Belilovsky, and T. Popa, “CLIPMesh: Generating textured meshes from text using pretrained image-text models,” in SIGGRAPH Asia 2022 Conference Papers, 2022, pp. 1–8. 3
- [51] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning Transferable Visual Models From Natural Language Supervision,” in International Conference on Machine Learning. PMLR, 2021, pp. 8748–8763. 3

- [52] C.-H. Lin, J. Gao, L. Tang, T. Takikawa, X. Zeng, X. Huang, K. Kreis, S. Fidler, M.-Y. Liu, and T.-Y. Lin, “Magic3D: High-Resolution Text-to-3D Content Creation,” arXiv preprint arXiv:2211.10440, 2022. 3, 4
- [53] R. Chen, Y. Chen, N. Jiao, and K. Jia, “Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation,” arXiv preprint arXiv:2303.13873, 2023. 3
- [54] J. Tang, J. Ren, H. Zhou, Z. Liu, and G. Zeng, “DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation,” in International Conference on Learning Representations, 2024. 3
- [55] Y. Huang, J. Wang, Y. Shi, B. Tang, X. Qi, and L. Zhang, “DreamTime: An Improved Optimization Strategy for Diffusion-Guided 3D Generation,” in International Conference on Learning Representations, 2024. 3, 15
- [56] O. Katzir, O. Patashnik, D. Cohen-Or, and D. Lischinski, “Noisefree Score Distillation,” in International Conference on Learning Representations, 2024. 3, 15
- [57] X. Yu, Y.-C. Guo, Y. Li, D. Liang, S.-H. Zhang, and X. QI, “Text-to3d with classifier score distillation,” in International Conference on Learning Representations, 2024. 3
- [58] Y. Liang, X. Yang, J. Lin, H. Li, X. Xu, and Y. Chen, “Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching,” arXiv preprint arXiv:2311.11284, 2023. 3
- [59] J. Zhu, P. Zhuang, and S. Koyejo, “HiFA: High-fidelity Text-to-3D Generation with Advanced Diffusion Guidance,” in International Conference on Learning Representations, 2024. 3
- [60] Z. Wang, C. Lu, Y. Wang, F. Bao, C. Li, H. Su, and J. Zhu, “ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation,” in Advances in Neural Information Processing Systems, 2023. 3
- [61] Y. Cao, Y.-P. Cao, K. Han, Y. Shan, and K.-Y. K. Wong, “DreamAvatar: Text-and-Shape Guided 3D Human Avatar Generation via Diffusion Models,” arXiv preprint arXiv:2304.00916, 2023. 3
- [62] H. Zhang, B. Chen, H. Yang, L. Qu, X. Wang, L. Chen, C. Long, F. Zhu, D. Du, and M. Zheng, “AvatarVerse: High-quality & Stable 3D Avatar Creation from Text and Pose,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 7, 2024, pp. 7124–

7132. 3

- [63] R. A. Guler,¨ N. Neverova, and I. Kokkinos, “DensePose: Dense Human Pose Estimation in the Wild,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 7297–7306. 3
- [64] X. Huang, R. Shao, Q. Zhang, H. Zhang, Y. Feng, Y. Liu, and Q. Wang, “HumanNorm: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,

2024. 3

- [65] T. Alldieck, H. Xu, and C. Sminchisescu, “imGHUM: Implicit Generative Models of 3D Human Shape and Articulated Pose,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5461–5470. 3
- [66] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20331–20341. 3, 6
- [67] L. Hu, H. Zhang, Y. Zhang, B. Zhou, B. Liu, S. Zhang, and L. Nie, “GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition,

2024. 3

- [68] G. Moon, T. Shiratori, and S. Saito, “Expressive whole-body 3d gaussian avatar,” arXiv preprint arXiv:2407.21686, 2024. 3
- [69] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” Advances in Neural Information Processing Systems, vol. 33, pp. 6840–6851, 2020. 4
- [70] J. Tang, “Stable-dreamfusion: Text-to-3d with stable-diffusion,” 2022, https://github.com/ashawkey/stable-dreamfusion. 4
- [71] G. Metzer, E. Richardson, O. Patashnik, R. Giryes, and D. CohenOr, “Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures,” arXiv preprint arXiv:2211.07600, 2022. 4, 6
- [72] A. Zeng, X. Ju, L. Yang, R. Gao, X. Zhu, B. Dai, and Q. Xu, “DeciWatch: A Simple Baseline for 10× Efficient 2D and 3D Pose Estimation,” in Proceedings of the European conference on computer vision (ECCV). Springer, 2022, pp. 607–624. 4
- [73] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black, “AMASS: Archive of motion capture as surface shapes,”

- in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 5442–5451. 4
- [74] A. Mohr and M. Gleicher, “Building efficient, accurate character skins from examples,” ACM Transactions on Graphics (TOG), vol. 22, no. 3, pp. 562–568, 2003. 4, 6
- [75] I. Pantazopoulos and S. Tzafestas, “Occlusion Culling Algorithms: A Comprehensive Survey,” Journal of Intelligent and Robotic Systems, vol. 35, pp. 123–156, 2002. 5
- [76] A. Gu´edon and V. Lepetit, “SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 5354–5363. 6, 15
- [77] J. Waczynska,´ P. Borycki, S. Tadeja, J. Tabor, and P. Spurek, “GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting,” arXiv preprint arXiv:2402.01459, 2024. 6
- [78] T. Von Marcard, R. Henschel, M. J. Black, B. Rosenhahn, and G. Pons-Moll, “Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 601–617. 8
- [79] R. Li, S. Yang, D. A. Ross, and A. Kanazawa, “Learn to Dance with AIST++: Music Conditioned 3D Dance Generation,” 2021. 8
- [80] J. Lin, A. Zeng, S. Lu, Y. Cai, R. Zhang, H. Wang, and L. Zhang, “Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset,” in Advances in Neural Information Processing Systems,

2023. 8, 14

- [81] H. Yi, H. Liang, Y. Liu, Q. Cao, Y. Wen, T. Bolkart, D. Tao, and M. J. Black, “Generating Holistic 3D Human Motion from Speech,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 8
- [82] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5470–5479. 14

[Figure 284]

Ailing Zeng (Member, IEEE) is a senior researcher at Tencent AI Lab. Previously, she obtained her PhD degree from the Department of Computer Science and Engineering, the Chinese University of Hong Kong. Her research targets to build multi-modal human-like intelligent agents on scalable big data, especially for Large Motion Models to capture, understand, interact, and generate the motion of humans, animals, and the world. She has published over thirty toptier conference papers at CVPR, NeurIPS, etc.

Zheng-Jun Zha (Member, IEEE) received the BE and PhD degrees from the University of Science and Technology of China, Hefei, China, in 2004 and 2009, respectively. He is currently a full professor with the School of Information Science and Technology, University of Science and Technology of China, and the executive director with the National Engineering Laboratory for Brain-Inspired Intelligence Technology and Application (NEL-BITA). He has authored or coauthored more than 200 papers in his research

[Figure 285]

field with a series of publications on top journals and conferences, which include multimedia analysis and understanding, computer vision, pattern recognition, and brain-inspired intelligence. He was a recipient of multiple paper awards from prestigious conferences, including the Best Paper/Student Paper Award in Association for Computing Machinery (ACM) Multimedia and AAAI Distinguished Paper. He serves/served as an associated editor for IEEE Transactions on Multimedia, IEEE Transactions on Circuits and Systems for Video Technology, etc.

[Figure 286]

Yukun Huang is a Post-doctoral Research Fellow at the HKU Musketeers Foundation Institute of Data Science (HKU IDS). Previously, he obtained his PhD degree from the University of Science and Technology of China (USTC) and did his undergraduate studies at the South China University of Technology. His research interests broadly lie in the computer vision and machine learning. In particular, he is interested in 3D synthesis, virtual human, generative model, and person re-identification.

Lei Zhang (Fellow, IEEE) received the PhD degree in computer science from Tsinghua University, Beijing, China, in 2001. He is currently the chief scientist of computer vision and robotics with International Digital Economy Academy (IDEA) and an adjunct professor with the Hong Kong University of Science and Technology, Guangzhou, China. Prior to his current post, he was a principal researcher and research manager with Microsoft. He has authored or coauthored more than 150 techinical papers, and

[Figure 287]

holds more than 60 U.S. patents in his research field, which include computer vision and machine learning, with particular focus on generic visual recognition at large scale. He was a editorial board member for IEEE Transactions on Multimedia, IEEE Transactions on Circuits and Systems for Video Technology, and Multimedia System Journal and as the area chair of many top conferences.

[Figure 288]

Jianan Wang received the MSc degree from the University of Oxford and currently serves as the chief researcher in AI cognition at Astribot. She has previously worked with DeepMind and the International Digital Economy Academy (IDEA). Her research interests and publications span computer vision and machine learning theory, with a recent focus on generative AI and robotics.

Xihui Liu (Member, IEEE) is an assistant professor at Department of Electrical and Electronic Engineering and Institute of Data Science, The University of Hong Kong. Before joining HKU, she was a postdoctoral researcher at University of California, Berkeley. She received the Bachelor’s degree from Tsinghua University and PhD degree from The Chinese University of Hong Kong. Her research interests include computer vision, deep learning, generative models, and multimodal AI. She was awarded Adobe Re-

[Figure 289]

search Fellowship 2020, EECS Rising Stars 2021, and WAIC Rising Star Award 2022. She serves as area chairs for CVPR 2024, ACM MM 2024, and ICLR 2025.

