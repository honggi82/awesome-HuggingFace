# arXiv:2405.16822v1[cs.CV]27May2024

## Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels

#### Yikai Wang∗1, Xinzhou Wang∗1,2,3, Zilong Chen1,2, Zhengyi Wang1,2, Fuchun Sun1, Jun Zhu†1,2

1Department of Computer Science and Technology, BNRist Center, Tsinghua University 2ShengShu 3College of Electronic and Information Engineering, Tongji University

yikaiw@outlook.com, wangxinzhou@tongji.edu.cn, dcszj@tsinghua.edu.cn

### Abstract

Video generative models are receiving particular attention given their ability to generate realistic and imaginative frames. Besides, these models are also observed to exhibit strong 3D consistency, significantly enhancing their potential to act as world simulators. In this work, we present Vidu4D, a novel reconstruction model that excels in accurately reconstructing 4D (i.e., sequential 3D) representations from single generated videos, addressing challenges associated with non-rigidity and frame distortion. This capability is pivotal for creating high-fidelity virtual contents that maintain both spatial and temporal coherence. At the core of Vidu4D is our proposed Dynamic Gaussian Surfels (DGS) technique. DGS optimizes time-varying warping functions to transform Gaussian surfels (surface elements) from a static state to a dynamically warped state. This transformation enables a precise depiction of motion and deformation over time. To preserve the structural integrity of surfacealigned Gaussian surfels, we design the warped-state geometric regularization based on continuous warping fields for estimating normals. Additionally, we learn refinements on rotation and scaling parameters of Gaussian surfels, which greatly alleviates texture flickering during the warping process and enhances the capture of fine-grained appearance details. Vidu4D also contains a novel initialization state that provides a proper start for the warping fields in DGS. Equipping Vidu4D with an existing video generative model, the overall framework demonstrates highfidelity text-to-4D generation in both appearance and geometry. Project page: https://vidu4d-dgs.github.io.

### 1 Introduction

The field of multimodal generation exhibits significant advancements and holds great promise for various applications. Recently, video generative models have garnered attention for their remarkable capability to craft immersive and lifelike frames [4, 8]. These models produce visually stunning content while also exhibiting strong 3D consistency [15, 80], largely increasing their potential to simulate realistic environments.

Parallel to these developments, high-quality 4D reconstruction has made great strides [19, 57, 62, 93, 99]. This technique involves capturing and rendering detailed spatial and temporal information. When integrated with generative video technologies, 4D reconstruction potentially enables the creation of models that capture static scenes and dynamic sequences over time. This synthesis provides a more holistic representation of reality, which is crucial for applications such as virtual reality, scientific visualization, and embodied artificial intelligence.

∗Equal contribution. †The corresponding author.

Preprint. Under review.

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

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

- (a) Prompt: A portrait captures the dignified presence of an orange cat with striking blue eyes. The cat wears a single pearl earring. Her head tilts in contemplation, reminiscent of a Dutch cap.

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

- (b) Prompt: A dragon with its hair blown by a strong wind. Devil enters the soul with ethereal landscapes.

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

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

(c) Prompt: Light painting photo of a cheetah, cinematic.

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

(d) Prompt: A goldfish seemingly swimming through the air.

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

(e) Prompt: A small, fluffy creature with an appearance reminiscent of a mythical being. The creature’s fur texture is rendered in high detail. The monster’s large eyes and open mouth express wonder and curiosity.

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

(f) Prompt: An isolated coloured abstract sculpture with a dali shape.

- Figure 1: Text-(to-video)-to-4D samples generated by equipping Vidu4D with a pretrained video diffusion model [4]. For each sample, we exhibit per-frame 3D rendering for novel-view color, normal, and surfel feature. We observe that Vidu4D can reconstruct precisely detailed and photo-realistic 4D representation. See our accompanying videos in our project page for better visual quality.

However, achieving high-fidelity 4D reconstruction from generated videos poses great challenges. Non-rigidity and frame distortion are prevalent issues that can undermine the temporal and spatial coherence of the reconstructed content, thus complicating the creation of a seamless and coherent depiction of dynamic subjects.

In this work, we introduce Vidu4D, a novel reconstruction pipeline designed to accurately reconstruct 4D representations from single generated videos, facilitating the creation of 4D content with high precision in spatial and temporal coherence. Vidu4D contains two novel stages, namely, the initialization of non-rigid warping fields and Dynamic Gaussian Surfels (DGS), together enabling the reconstruction of high-fidelity 4D content with high-fidelity appearance and accurate geometry.

Specifically, the proposed DGS optimizes non-rigid warping functions that transform Gaussian surfels from static to dynamically warped states. This dynamic transformation accurately represents motion and deformation over time, crucial for capturing realistic 4D representations. Besides, DGS demonstrates superior 4D reconstruction performance due to two other key aspects. Firstly, in terms of geometry, DGS adheres to Gaussian surfels principles [16, 28] to achieve precise geometric representation. Unlike existing methods, DGS incorporates warped-state normal consistency regularization to align surfels with actual surfaces with learnable continuous fields (w.r.t. spatial coordinate and time) to ensure smooth warping when estimating normals. Secondly, for appearance, DGS learns additional refinements on the rotation and scaling parameters of Gaussian surfels by a dual branch structure. This refinement reduces the flickering artifacts during warping and allows for the precise rendering of appearance details, resulting in high-quality reconstructed 4D representations.

By integrating Vidu4D with an existing powerful video generative model named Vidu [4], the overall framework demonstrates exceptional capabilities in text-to-4D generation. We provide 4D visualization results in Fig. 1. Extensive experiments based on the generated videos verify the effectiveness of our method compared to current state-of-the-art methods.

- 2 Related works
- 3D representation. Transforming 2D images into 3D representations has long been a central challenge in the field. Initially, triangle meshes were favored for their compactness and compatibility with rendering pipelines [9, 17, 66, 77, 81, 92]. However, the transition to more sophisticated volumetric methods was inevitable due to the limitations of surface-based approaches. Early volumetric representations included voxel grids [35, 47, 60, 71] and multi-plane images [20, 53, 73, 74, 79, 104], which, despite their straightforwardness, demanded intricate optimization strategies. The introduction of neural radiance fields (NeRF) [54] marked a significant advancement, offering an implicit volumetric neural representation that could store and query the density and color of each point, leading to highly realistic reconstructions. The NeRF paradigm has since been improved upon in terms of reconstruction quality [5, 6, 33, 52, 91] and rendering [12, 23, 25, 27, 41, 44, 48, 63–65, 84, 101]. To address the limitations of NeRF, such as rendering speed and memory usage, recent work dubbed 3D Gaussian splatting (3DGS) [33] has proposed anisotropic Gaussian representations with GPUoptimized tile-based rasterization. This has opened up new avenues for surface extraction [24, 28], generation [14, 76, 95], and large-scale scene reconstruction [34, 45, 69], with 3DGS emerging as a universal representation for 3D scenes and objects. Gaussian surfels methods [16, 28] further exhibit advantages in modeling accurate geometry. While these methods have significantly advanced the field of static 3D representation, capturing the dynamic aspects of real-world scenes with non-rigid motion and deformation introduces a distinct set of challenges that demand innovative solutions.

Dynamic reconstruction and generation. The dynamic reconstruction of scenes from video captures presents a more complex challenge than static reconstruction, necessitating the capture of non-rigid motion and deformation over time [30, 37, 59, 75, 87]. Traditional methods have explored dynamic reconstruction using synchronized multi-view videos [1, 3, 11, 36, 47, 58, 72, 82, 83, 85, 88] or have focused on specific dynamic elements like humans or animals. More recently, there has been a shift towards reconstructing non-rigid objects from monocular videos, which is a more practical yet challenging scenario. One approach involves incorporating time as an additional input to the neural radiance field [11, 38, 67, 97], allowing for explicit querying of spatiotemporal information. Another line of research decomposes the spatiotemporal radiance field into a canonical space and a deformation field, representing spatial attributes and their temporal variations [18, 19, 19, 21, 22, 31, 39, 43, 46, 56, 57, 62, 68, 78, 94, 103]. With advancements in 3DGS, deformable-GS [99] and 4DGS

[93] have been developed, utilizing neural deformation fields with multi-layer perception (MLP) and triplane, respectively. SCGS [29] and dynamic 3D Gaussians [51] also advance the field by modeling time-varying scenes. Building on these advances, our work introduces dynamic Gaussian surfels, a novel extension of Gaussian representations that enhances the quality of both appearance and surface reconstruction under dynamic scenarios. In the realm of 3D or 4D generation, our approach diverges from recent progress in optimization-based [2, 13, 14, 40, 42, 61, 70, 87, 89], feed-forward [26, 90, 105], and multi-view reconstruction methods [15, 49, 50] by leveraging a video generative model to achieve generation capabilities. Our primary focus is on preserving high-quality appearance and geometrical integrity from generated videos. This results in a generation process that not only captures the nuances of motion and deformation but also maintains the high standards of realism and detail that are essential for creating immersive and lifelike virtual 3D representations.

### 3 Method

In this section, we first introduce the basic problem definition of 4D reconstruction (see Sec. 3.1). We then present our method dubbed Dynamic Gaussian Surfels (DGS) for accurately modeling both the appearance and geometry during the 4D reconstruction with large non-rigidity (see Sec. 3.2). Finally, we introduce Vidu4D as a reconstruction pipeline and the overall framework for performing a generation task (see Sec. 3.3).

- 3.1 Problem Definition

When given a single sequence of RGB video with T frames, the goal of 4D reconstruction is to determine a sequential 3D representation that could be rendered to fit each video frame as much

- as possible. Specifically, suppose the 3D representation for the t-th frame (termed as time t) is

parameterized by θt, where t = 1,··· ,T. Given a differentiable rendering mapping g, we could obtain the rendered color at the frame pixel x¯t ∈ R2. We choose volume rendering as commonly adopted in NeRF [54], Gaussian Splatting [33], and Gaussian Surfels [16, 28]. The optimization of 4D reconstruction can be implemented by minimizing the empirical loss as

min

θ

1 T

T

t=1 x ¯t

L c(x¯t) = g θt,{xti}i=1,···,N ,cˆ(x¯t) , (1)

where xti ∈ R3 is the i-th 3D point sampled or intersected with Gaussian primitives along the ray that emanates from the frame pixel x¯t; N is the number of sampled or intersected points per ray; c(x¯t)

and cˆ(x¯t) are the rendered color and the observed color at x¯t, respectively. 3.2 Dynamic Gaussian Surfels

By optimizing Eq. (1), essentially our goal is to build a sequential 3D representation that could deform to be consistent with each 2D frame. We first start by considering an ideal video exhibiting different views of the same static object without object deformation, movement, or video distortion. To model the 3D representation with high appearance fidelity and geometry accuracy, we follow the method of using differentiable 2D Gaussian primitives as proposed by recent Gaussian Surfels advances [16, 28]. Specifically, the k-th Gaussian surfel (of the total K) is characterized by a central point p∗k ∈ R3 and a local coordinate system centered at p∗k with two principal tangential vectors t∗u ∈ R3×1, t∗v ∈ R3×1 and scaling factors s∗u ∈ R, s∗v ∈ R. Here, we use the notation “∗” to represent parameters in the static state. A Gaussian surfel is computed as a 2D Gaussian defined in a local tangent plane in the world space. Following [28], for any point u = (u,v) located on the uv coordinate system centered

- at p∗k, its coordinate in the world space, denoted as Pk∗(u) ∈ R3×1, is computed by Pk∗(u) = p∗k + s∗ut∗uu + s∗vt∗vv = [R∗kS∗k p∗k](u,v,1,1)⊤, (2)

where R∗k = [t∗u,t∗v,t∗u × t∗v] ∈ SO(3) denotes the rotation matrix, and the diagonal matrix S∗k = diag(s∗u,s∗v,0) ∈ R3×3 denotes the scaling matrix.

In this work, our focus is on 4D reconstruction from a single generated video, which may exhibit significant non-rigidity, distortion, or illumination changes. We introduce Dynamic Gaussian Surfels (DGS), a method designed to achieve precise 4D reconstruction while accommodating non-rigidity and other time-varying effects.

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Text prompt

[Figure 96]

|Video diffusion model|
|---|

[Figure 97]

Warped-state normal regularization

[Figure 98]

Field init.

s⇤vR˜ tt⇤v

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

DGS

[Figure 104]

[Figure 105]

“an orange cat with striking blue eyes…”

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Pkt(u)

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

R˜ tp⇤k + T˜t

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

s⇤uR˜ tt⇤u

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

###### s⇤vt⇤v

Method details of DGS

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

Pk⇤(u)

[Figure 148]

[Figure 149]

s⇤ut⇤u

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Joint training with shared centers and warping

[Figure 157]

[Figure 158]

p⇤k

[Figure 159]

[Figure 160]

[Figure 161]

Warping field cond. on 𝑡 and 𝐮

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

Warped state at time 𝑡

[Figure 175]

Jt = [R˜ t,T˜t]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

###### (s⇤v +  s⇤v)R˜ t R⇤kt⇤v

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

Refinement

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

Pk0t(u)

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

 R⇤k  S⇤k

[Figure 204]

[Figure 205]

[Figure 206]

R˜ tp⇤k + T˜t Rasterization

Static state

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

(s⇤u +  s⇤u)R˜ t R⇤kt⇤u

[Figure 214]

[Figure 215]

[Figure 216]

- Figure 2: Illustration of the overall framework and our DGS in detail. For DGS, Gaussian surfels in the static state are transformed to the warped state by learning non-rigid warping functions conditioned on time t and coordinate u. We incorporate warped-state normal regularization for accurate geometry, and refined rotation and scaling matrices of Gaussian surfels for detailed appearance. Both branches in the warped state, including with and without refinement, share the same centers of Gaussian surfels and the same warping functions. “Field init.” stands for field initialization as introduced in Sec. 3.3.

Motivated by recent advancements in non-rigid reconstruction methods [56, 87, 97], we aim to ensure that the target object maintains a consistent static state across different frames, thereby mitigating non-rigidity and distortion effects. To achieve this, we employ warping techniques on each Gaussian surfel represented by Pk∗(u), transforming them into a corresponding Gaussian surfel Pkt(u) at time t, which is centered at ptk ∈ R3 with a rotation matrix Rtk ∈ SO(3) and a scaling matrix Stk ∈ R3×3. Non-rigid warping for Gaussian surfels. We now build the warping process from the static state to the warped state. We define a time-varying non-rigid warping function by leveraging B bones as key points to ease the training of deformation. In the static state, the b-th bone is represented by

- 3D Gaussian ellipsoids [96] with the center c∗b ∈ R3×1, rotation matrix Vb∗ ∈ R3×3, and diagonal scaling matrix Λ∗b ∈ R3×3. We let Jtb ∈ SE(3) represent a rigid transformation that moves the b-th bone from its static state to the warped state at time t. For a 3D point Pk∗(u), the skinning weight vectors wt ∈ RB×1 at time t is calculated by the normalized Mahalanobis distance following [97]

mtb = Pk∗(u) − ctb ⊤Qtb (Pk∗(u) − ctb , wt = σsoftmax mt1,mt2,··· ,mtB ⊤, (3) where mtb denotes the squared distance between Pk∗(u) and the b-th bone; ctb ∈ R3×1 is the center of the b-th bone at time t, and Qtb = Vbt⊤Λ∗bVbt is the precision matrix composed by the bone orientation matrix Vbt ∈ R3×3 at time t and Λ∗b. Specifically, there is (Vbt|ct) = Jtb(Vb∗|c∗) with c∗b, Vb∗, and Λ∗b being learnable parameters. σsoftmax is the softmax function.

In effect, Jtb is achieved by non-linear mappings using a multi-layer perception (MLP) with SE(3) guaranteed, as will be given later in Eq. (6). The non-rigid warping function can be written as the

weighted combination of Jtb ∈ SE(3), where we apply dual quaternion blend skinning (DQB) [32] to ensure valid SE(3) after combination,

B

Jt = R

wbtQ(Jtb) , (4)

b=1

where wbt is the b-th element of wt; Q and R denote the quaternion process and the inverse quaternion process, respectively. In this case, Jt ∈ SE(3).

We therefore rewrite the warping as Jt = [R˜ t,T˜t] with the rotation R˜ t ∈ SO(3) and translation T˜t ∈ R3, and apply the corresponding transformation to Eq. (2) by

##### Pkt(u) = JtPk∗(u) = R ˜ tR∗kS∗k R˜ tp∗k + T˜t (u,v,1,1)⊤. (5)

Note that Eq. (5) holds for any given point Pk∗(u) including the center point of the k-th Gaussian surfel (i.e., p∗k) when u = (0,0). By deriving Eq. (5), we enable connection of the warping function w.r.t. to any point u = (u,v) on the local coordinate system centered at p∗k, which is needed later in Eq. (9) where u is an intersection with Gaussian surfels and a ray that emanates from the frame pixel.

Warped-state normal regularization. To accurately capture the geometric representation, we follow similar methods in Gaussian Surfels [16, 28] to add normal consistency regularization which encourages all Gaussian surfels to be locally aligned with the actual surfaces. Differently, unlike 3D reconstruction for static scenes, 4D reconstruction commonly faces non-rigidity and distortion. Thus simply performing regularization to promote surface-aligned Gaussian surfels like previous methods harms the structural integrity due to the non-rigid warping.

We therefore design a warped-state normal regularization. As mentioned, each point Pkt(u) in the warped state at time t is transformed from its corresponding static point Pk∗(u) based on the warping function in Eq. (5), namely, Pkt(u) = JtPk∗(u) with Jt composed by Jtb. To maintain the structural integrity to a large extent when regularizing normal, we design Jtb as a continuous field that takes both the point Pk∗(u) (or equivalently, u in the local coordinate system) and the time t as conditions. By this setting, Jtb is expected to change continuously with the change of u or t. We implement the continuous field by using a NeRF-style MLP which directly outputs a 6-dimensional dual quaternion, and rely on the inverse quaternion process R to guarantee SE(3), i.e.,

##### Jtb = R MLP(γbt;u,t) , (6)

where γbt is a learnable latent code for encoding the b-th bone at time t; both u and t are sent to the MLP as conditions to obtain Jtb. Thus Jt is also expected to be continuous w.r.t. u and t. Based on the above design, the normal consistency loss at time t is obtained similar to [28],

K

ωk(1 − n⊤k Nt), Nt(x,y) = ∇xpt × ∇ypt |∇xpt × ∇ypt|

, (7)

Ln =

k=1

where k indexes over intersected surfels along the ray that emanates from the frame pixel x¯; ωk = αk Gk(u(x¯)) kj=1−1(1 − αj Gj(u(x¯))) denotes the blending weight of the intersection point; nk represents the normal of the surfel that is oriented towards the camera; Nt, computed with finite differences, is the surface normal estimated by the nearby depth point pt at warped state time t.

In summary, by learning a continuous warping field and aligning the surfel normal with the estimated surface normal in the warped state, we ensure that all Gaussian surfels locally approximate the actual object surface without being noticeably impaired by the non-rigid warping.

Dual branch structure with refinement. To further achieve fine-grained appearance and reduce the texture flickering during warping, we propose to learn refinement terms for adjusting the rotation matrices R∗k and scaling matrices S∗k (defined in Eq. (2)) in the static state. We suppose the refinement terms are ∆R∗k ∈ SO(3) and ∆S∗k ∈ R3×3, respectively. Note that the third-axis of ∆S∗k is no longer necessarily 0. During refinement, we remain the center points p∗k and the warping Jt (i.e., including both R˜ t and T˜t) to be unchanged. The new warped process is formulated as,

##### Pk′t(u) = R ˜ t(∆R∗kR∗k)(S∗k + ∆S∗k) R˜ tp∗k + T˜t (u,v,1,1)⊤. (8)

During the training of DGS, we maintain two branches including one with refinement and one without. In the warped state, both branches are jointly trained with shared warping functions and centers of Gaussian primitives2. Due to the involvement of ∆R∗k and ∆S∗k, both branches have different rotation and scaling matrices of Gaussian primitives.

Rasterization. Given a frame pixel x¯ and a camera ray that emanates from x¯, following the staticstate methods to calculate intersection coordinates with Gaussian primitives along the ray [28, 33], we could obtain warped-state intersection coordinates based on Eq. (5) and Eq. (8). We then perform the volume rendering process [28] that integrates alpha-weighted appearance along the ray by

c(x¯) =

k

ck αk Gk u(x¯)

k−1

j=1

1 − αj Gj u(x¯) , (9)

where k indexes over intersected Gaussian primitives along the ray that emanates from the frame pixel x¯; αk and ck denote the opacity and view-dependent appearance parameterized with spherical

2Here, since the third-axis of the refined scaling matrix is not necessarily 0, we adopt “Gaussian primitive” for commonly referring to both Gaussian surfel and the refined Gaussian.

Field initialization stage DGS stage

Initialization

Forward warping (for sampled points)

Forward warping (for Gaussian surfels)

|Static state|
|---|

|Warped state|
|---|

|[Figure 217]<br><br>[Figure 218]<br><br>[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]<br><br>[Figure 222]<br><br>[Figure 223]<br><br>[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]<br><br>[Figure 228]<br><br>Static state|
|---|

|[Figure 229]<br><br>[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]<br><br>[Figure 237]<br><br>[Figure 238]<br><br>[Figure 239]<br><br>[Figure 240]<br><br>[Figure 241]<br><br>[Figure 242]<br><br>[Figure 243]<br><br>Warped state|
|---|

[Figure 244]

[Figure 245]

[Figure 246]

Inversion

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

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

Backward warping (for sampled points)

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Sampled point Gaussian surfel

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

- Figure 3: Illustration of the pipeline of Vidu4D, including the initialization stage and the DGS stage.

harmonics of the k-th Gaussian surfel, respectively; Gk(u(x¯)) = exp −u

2+v2

2 corresponds to the

k-th intersection point u(x¯) which could be directly calculated when given Pkt(u) or Pk′t(u) and the corresponding local coordinate system. During implementation, Gk(u(x¯))) is further applied a low-pass filter following [7, 28].

A detailed architecture of DGS is depicted in Fig. 2. Important symbols are summarized in Table 2.

- 3.3 Vidu4D

Given that the camera trajectory of generated videos is unknown, SfM methods like COLMAP struggle to converge due to rigidity violations. Additionally, since the background of generated videos appears to exhibit soft deformation or flickering colors, proper estimation of camera/body poses through background SfM is hindered. These challenges often result in very few successful registrations, as demonstrated in previous monocular 4D reconstruction tasks [97].

In this part, we arrive at Vidu4D, a reconstruction pipeline comprising two key stages as illustrated in Fig. 3, including a field initialization stage and the DGS stage. Specifically, we propose the field initialization as another key component of our pipeline to initialize the field in Eq. (6) of DGS for fast and stable convergence. We first train a neural SDF [86] using the same bone-based warping structure as utilized in our DGS. Unlike DGS, which warps Gaussian surfels from the static state to the warped state for rasterization, the neural SDF warps sampled points on camera rays from the warped state back to the static state. For the neural SDF part, we optimize the backward warping and learn a forward warping as the inversion of the backward warping by employing a cycle loss, inspired by [10, 97]. We then initialize the MLP to obtain warping functions Jtb by the MLP learned by the neural SDF part. We provide more details in our Appendix.

With our field initialization before DGS, our Vidu4D is capable of performing a text-(to-video)-to-4D generation task with the integration of existing video diffusion models.

4 Experiment

In this section, we provide an extensive evaluation of our method DGS with the initialization in Sec. 3.3, comparing both appearance and geometry against previous state-of-the-art methods. Additionally, we analyze the contributions of each proposed component in detail.

- 4.1 Implementation

For all qualitative and quantitative experiments, we follow the standard pipeline for dynamic reconstruction [57], to construct our evaluation setup by selecting every fourth frame as a training frame and designating the middle frame between each pair of training frames as a validation frame.

Our model configuration involves several key parameters to balance reconstruction and regularization losses. For the field initialization stage, we use a similar architecture with 8 layers for volume rendering as in NeRF [54], and initialize MLP for predicting SDF as an approximate unit sphere [100]. We obtain a neural SDF, a warping field, and camera poses after this stage. For the DGS stage, we initialize centers of the Gaussian surfels with the sampled surface points extracted from the neural SDF, and initialize the warping field by the forward field from the first stage. The dimension of the latent code embedding γbt is set as 128. Following BANMo [97], we adopt 25 bones to optimize skinning weights. For each reconstruction, the overall training takes over 1 hour on an A800 GPU.

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

BANMo

D-NeRF

Deformable-GS

SCGS

Deformable-GS w. Poses

SCGS w. Poses

###### DGS (Ours)

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

BANMo

D-NeRF

Deformable-GS

SCGS

Deformable-GS w. Poses

SCGS w. Poses

###### DGS (Ours)

Novel view 1 Normal Novel view 2 Novel view 1 Normal Novel view 2 Novel view 1 Normal Novel view 2 Novel view 1 Normal Novel view 2

- Figure 4: Novel-view qualitative evaluation compared with SOTA methods including NeRF-based methods (BANMo [97] and D-NeRF [62]) and Gaussian splatting-based methods (DeformableGS [99] and SCGS [29]). We also provide our learned camera poses to baseline approaches for a fair comparison. These variants are denoted as “w. Poses”. Best view in color and zoom in.

Table 1: Novel-view quantitative results on generated videos. Evaluation metrics are PSNR, SSIM, and LPIPS. We report results on three single videos and the averaged results over 30 single videos.

Cat Cheetah Dragon Average over 30 videos

PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

BANMo [97] 15.10 0.6514 0.2575 13.15 0.5921 0.3241 18.48 0.6423 0.3500 13.62 ± 2.99 0.6153 ± 0.0714 0.3738 ± 0.0665 D-NeRF [62] 15.15 0.6537 0.2657 13.21 0.5930 0.3344 18.53 0.6489 0.3527 21.01 ± 2.86 0.8519 ± 0.0717 0.1522 ± 0.0754

Deformable-GS [99] 19.09 0.7815 0.2434 20.35 0.8039 0.1982 24.19 0.9100 0.0992 13.22 ± 3.42 0.5934 ± 0.0535 0.3749 ± 0.0763 SCGS [29] 19.46 0.7867 0.2405 20.87 0.8123 0.1919 24.03 0.9083 0.1009 21.17 ± 2.69 0.8547 ± 0.0691 0.1504 ± 0.0737 Deformable-GS w. Poses 21.94 0.8123 0.1816 22.41 0.8200 0.1687 26.05 0.9218 0.0894 22.63 ± 2.14 0.8469 ± 0.0438 0.1452 ± 0.0354 SCGS w. Poses 23.25 0.8268 0.1574 23.70 0.8338 0.1497 28.40 0.9375 0.0686 24.75 ± 2.11 0.8680 ± 0.0440 0.1201 ± 0.0359 DGS (Ours) 24.63 0.8432 0.1559 25.68 0.8843 0.1117 28.58 0.9392 0.0618 27.30 ± 2.66 0.9152 ± 0.0602 0.0877± 0.0564

#### 4.2 Qualitative Evaluation

In the qualitative evaluation, we visually compare the novel-view reconstructions produced by our DGS against those generated by other state-of-the-art models, as illustrated in Fig. 4. Our evaluation focuses on several key aspects including detail preservation, texture quality, and geometric accuracy. Compared to methods based on implicit fields, the integration of Gaussian in our approach facilitates the rendering of highly detailed textures. Additionally, benefiting from a more geometry-aware representation, our method produces superior normal maps compared to those purely Gaussian-based methods. This also enhances the robustness of our method against artifacts of the generated videos like occlusions. For instance, in the third clip of the series, which features a dragon shrouded in fog, both SCGS and Deformable-GS methods tend to overfit and subsequently show a decline in performance. In contrast, our method consistently delivers superior results.

[Figure 299]

[Figure 300]

[Figure 301]

|[Figure 302]|
|---|

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

| |
|---|

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

|[Figure 311]|
|---|

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

| |
|---|

[Figure 316]

Rendered color Rendered normal Surface normal

Rendered color

Rendered normal

(d) w/o. warped-state normal regularization

(a) Ground truth

(b) Our full model

(c) w/o. refinement branch

- Figure 5: Ablation studies on the geometric regularization and refinement strategy. For our full model shown in (b), we provide our rendered color, rendered normal, and surface normal (estimated from the depth points for regularization). Additionally, for comparison, we visualize the rendered color for the case without refinements in (c) and the rendered normal for the case without warped-state normal regularization in (d), respectively. We showcase our model’s fidelity with close-ups.

#### 4.3 Quantitative Evaluation

We provide the quantitative evaluation comparing our method with state-of-the-art works in Table 1. Metrics include Peak Signal-to-Noise Ratio (PSNR) to evaluate the fidelity of the reconstructed textures, Structural Similarity Index (SSIM) for the quality evaluation, and LPIPS [102] as a perceptual metric. Our method exhibits superiority over all baseline methods, even with our learned poses, e.g., ∼2.5 PSNR increase over SCGS with poses for the averaged results.

#### 4.4 Ablations

To understand the contributions of each component in Vidu4D, especially DGS, we conduct ablation studies in this section. We remove or alter specific elements of our model and observe the resulting performance changes in both appearance and geometry reconstruction.

Geometric regularization. We evaluate the impact of warped-state normal regularization by disabling it during training. From Fig. 5(b)(d), we observe that when removing the regularization, there is a significant degradation in the structural integrity of surface-aligned Gaussian surfels, leading to noticeable inconsistency in the reconstructed 4D models.

Refinement strategy. We examine the effect of omitting refinements by keeping one branch (the concept of branches could be better visualized in Fig. 2) during training, shown in Fig. 5(b)(c). The performance indicates that removing refinements increases the loss of fine-grained appearance details. Additionally, we also find that refinements are crucial for mitigating the texture flickering issue.

Additional ablations. Please refer to the Appendix for additional ablation studies that detail the effectiveness of our refinement strategy and field initialization.

### 5 Conclusion

We introduce Vidu4D as a novel reconstruction model to achieve high-fidelity 4D representations from single generated videos. Vidu4D is powerful with our proposed DGS which builds the non-rigid warping field to transform Gaussian surfels, ensuring precise capture of motion and deformation over time. DGS also introduces key innovations that significantly enhance the accuracy and fidelity of 4D reconstruction, including dual branch refinement and warped-state geometric regularization. Our experiments demonstrate that Vidu4D outperforms existing methods in both quantitative and qualitative evaluations, highlighting its superiority in generating realistic and immersive 4D content.

Limitations and broader impact. While Vidu4D with DGS presents a significant performance in

- 4D reconstruction, currently there are still limitations such as the reliance on video quality, scalability challenges for large scenes, and computational difficulties in real-time applications. Additionally, when equipping Vidu4D with generative models, as with any generative technology, there is a risk of producing deceptive content which needs more caution.

### References

- [1] Attal, B., Huang, J.B., Richardt, C., Zollhoefer, M., Kopf, J., O’Toole, M., Kim, C.: Hyperreel: Highfidelity 6-dof video with ray-conditioned sampling. arXiv preprint arXiv:2301.02238 (2023)
- [2] Bahmani, S., Skorokhodov, I., Rong, V., Wetzstein, G., Guibas, L., Wonka, P., Tulyakov, S., Park, J.J., Tagliasacchi, A., Lindell, D.B.: 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In: CVPR) (2024)
- [3] Bansal, A., Vo, M., Sheikh, Y., Ramanan, D., Narasimhan, S.: 4d visualization of dynamic events from unconstrained multi-view videos. In: CVPR (2020)
- [4] Bao, F., Xiang, C., Yue, G., He, G., Zhu, H., Zheng, K., Zhao, M., Liu, S., Wang, Y., Zhu, J.: Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233 (2024)
- [5] Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srinivasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In: ICCV (2021)
- [6] Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Zip-nerf: Anti-aliased grid-based neural radiance fields. arXiv preprint arXiv:2304.06706 (2023)
- [7] Botsch, M., Hornung, A., Zwicker, M., Kobbelt, L.: High-quality surface splatting on today’s gpus. In: Proceedings Eurographics/IEEE VGTC Symposium Point-Based Graphics (2005)
- [8] Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., Ng, C., Wang, R., Ramesh, A.: Video generation models as world simulators (2024), https: //openai.com/research/video-generation-models-as-world-simulators
- [9] Buehler, C., Bosse, M., McMillan, L., Gortler, S., Cohen, M.: Unstructured lumigraph rendering. In: Proceedings of the 28th annual conference on Computer graphics and interactive techniques (2001)
- [10] Cai, H., Feng, W., Feng, X., Wang, Y., Zhang, J.: Neural surface reconstruction of dynamic scenes with monocular RGB-D camera. In: NeurIPS (2022)
- [11] Cao, A., Johnson, J.: Hexplane: a fast representation for dynamic scenes. arXiv preprint arXiv:2301.09632

(2023)

- [12] Cao, J., Wang, H., Chemerys, P., Shakhrai, V., Hu, J., Fu, Y., Makoviichuk, D., Tulyakov, S., Ren, J.: Real-time neural light field on mobile devices. arXiv preprint arXiv:2212.08057 (2022)
- [13] Chen, R., Chen, Y., Jiao, N., Jia, K.: Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. In: ICCV (2023)
- [14] Chen, Z., Wang, F., Wang, Y., Liu, H.: Text-to-3d using gaussian splatting. In: CVPR (2024)
- [15] Chen, Z., Wang, Y., Wang, F., Wang, Z., Liu, H.: V3d: Video diffusion models are effective 3d generators. arXiv preprint arXiv:2403.06738 (2024)
- [16] Dai, P., Xu, J., Xie, W., Liu, X., Wang, H., Xu, W.: High-quality surface reconstruction using gaussian surfels. In: SIGGRAPH (2024)
- [17] Debevec, P.E., Taylor, C.J., Malik, J.: Modeling and rendering architecture from photographs: A hybrid geometry-and image-based approach. In: Proceedings of the 23rd annual conference on Computer graphics and interactive techniques (1996)
- [18] Du, Y., Zhang, Y., Yu, H.X., Tenenbaum, J.B., Wu, J.: Neural radiance flow for 4d view synthesis and video processing. In: ICCV (2021)
- [19] Fang, J., Yi, T., Wang, X., Xie, L., Zhang, X., Liu, W., Nießner, M., Tian, Q.: Fast dynamic radiance fields with time-aware neural voxels. In: SIGGRAPH Asia (2022)
- [20] Flynn, J., Broxton, M., Debevec, P., DuVall, M., Fyffe, G., Overbeck, R., Snavely, N., Tucker, R.: Deepview: View synthesis with learned gradient descent. In: CVPR (2019)
- [21] Gao, C., Saraf, A., Kopf, J., Huang, J.B.: Dynamic view synthesis from dynamic monocular video. In: ICCV (2021)
- [22] Gao, H., Li, R., Tulsiani, S., Russell, B., Kanazawa, A.: Dynamic novel-view synthesis: A reality check. In: NeurIPS (2022)
- [23] Garbin, S.J., Kowalski, M., Johnson, M., Shotton, J., Valentin, J.: Fastnerf: High-fidelity neural rendering at 200fps. In: ICCV (2021)
- [24] Guédon, A., Lepetit, V.: Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. arXiv preprint arXiv:2311.12775 (2023)
- [25] Hedman, P., Srinivasan, P.P., Mildenhall, B., Barron, J.T., Debevec, P.: Baking neural radiance fields for real-time view synthesis. ICCV (2021)

- [26] Hong, Y., Zhang, K., Gu, J., Bi, S., Zhou, Y., Liu, D., Liu, F., Sunkavalli, K., Bui, T., Tan, H.: Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400 (2023)
- [27] Hu, T., Liu, S., Chen, Y., Shen, T., Jia, J.: Efficientnerf efficient neural radiance fields. In: CVPR (2022)
- [28] Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S.: 2d gaussian splatting for geometrically accurate radiance fields. In: SIGGRAPH. Association for Computing Machinery (2024)
- [29] Huang, Y.H., Sun, Y.T., Yang, Z., Lyu, X., Cao, Y.P., Qi, X.: Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. arXiv preprint arXiv:2312.14937 (2023)
- [30] Jiakai, Z., Xinhang, L., Xinyi, Y., Fuqiang, Z., Yanshun, Z., Minye, W., Yingliang, Z., Lan, X., Jingyi, Y.: Editable free-viewpoint video using a layered neural representation. In: SIGGRAPH (2021)
- [31] Jiang, Y., Hedman, P., Mildenhall, B., Xu, D., Barron, J.T., Wang, Z., Xue, T.: Alignerf: High-fidelity neural radiance fields via alignment-aware training. arXiv preprint arXiv:2211.09682 (2022)
- [32] Kavan, L., Collins, S., Zára, J., O’Sullivan, C.: Skinning with dual quaternions. In: SI3D (2007)
- [33] Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. (2023)
- [34] Kerbl, B., Meuleman, A., Kopanas, G., Wimmer, M., Lanvin, A., Drettakis, G.: A hierarchical 3d gaussian representation for real-time rendering of very large datasets. ACM Trans. Graph. (2024)
- [35] Kutulakos, K.N., Seitz, S.M.: A theory of shape by space carving. IJCV (2000)
- [36] Li, L., Shen, Z., Wang, Z., Shen, L., Tan, P.: Streaming radiance fields for 3d video synthesis. arXiv preprint arXiv:2210.14831 (2022)
- [37] Li, R., Tanke, J., Vo, M., Zollhofer, M., Gall, J., Kanazawa, A., Lassner, C.: Tava: Template-free animatable volumetric actors (2022)
- [38] Li, T., Slavcheva, M., Zollhoefer, M., Green, S., Lassner, C., Kim, C., Schmidt, T., Lovegrove, S., Goesele, M., Newcombe, R., et al.: Neural 3d video synthesis from multi-view video. In: CVPR (2022)
- [39] Li, Z., Niklaus, S., Snavely, N., Wang, O.: Neural scene flow fields for space-time view synthesis of dynamic scenes. In: CVPR (2021)
- [40] Lin, C.H., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M.Y., Lin, T.Y.: Magic3d: High-resolution text-to-3d content creation. In: CVPR (2023)
- [41] Lindell, D.B., Martel, J.N., Wetzstein, G.: Autoint: Automatic integration for fast neural volume rendering. In: CVPR (2021)
- [42] Ling, H., Kim, S.W., Torralba, A., Fidler, S., Kreis, K.: Align your gaussians: Text-to-4d with dynamic 3d gaussians and composed diffusion models. arXiv preprint arXiv:2312.13763 (2023)
- [43] Liu, J.W., Cao, Y.P., Mao, W., Zhang, W., Zhang, D.J., Keppo, J., Shan, Y., Qie, X., Shou, M.Z.: Devrf: Fast deformable voxel radiance fields for dynamic scenes. arXiv preprint arXiv:2205.15723 (2022)
- [44] Liu, L., Gu, J., Zaw Lin, K., Chua, T.S., Theobalt, C.: Neural sparse voxel fields. In: NeurIPS (2020)
- [45] Liu, Y., Guan, H., Luo, C., Fan, L., Peng, J., Zhang, Z.: Citygaussian: Real-time high-quality large-scale scene rendering with gaussians. arXiv preprint arXiv: 2404.01133 (2024)
- [46] Liu, Y.L., Gao, C., Meuleman, A., Tseng, H.Y., Saraf, A., Kim, C., Chuang, Y.Y., Kopf, J., Huang, J.B.: Robust dynamic radiance fields. In: CVPR (2023)
- [47] Lombardi, S., Simon, T., Saragih, J., Schwartz, G., Lehrmann, A., Sheikh, Y.: Neural volumes: Learning dynamic renderable volumes from images. arXiv preprint arXiv:1906.07751 (2019)
- [48] Lombardi, S., Simon, T., Schwartz, G., Zollhoefer, M., Sheikh, Y., Saragih, J.: Mixture of volumetric primitives for efficient neural rendering. arXiv preprint arXiv:2103.01954 (2021)
- [49] Long, X., Guo, Y.C., Lin, C., Liu, Y., Dou, Z., Liu, L., Ma, Y., Zhang, S.H., Habermann, M., Theobalt, C., et al.: Wonder3d: Single image to 3d using cross-domain diffusion. arXiv preprint arXiv:2310.15008

(2023)

- [50] Lu, Y., Zhang, J., Li, S., Fang, T., McKinnon, D., Tsin, Y., Quan, L., Cao, X., Yao, Y.: Direct2. 5: Diverse text-to-3d generation via multi-view 2.5 d diffusion. arXiv preprint arXiv:2311.15980 (2023)
- [51] Luiten, J., Kopanas, G., Leibe, B., Ramanan, D.: Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In: 3DV (2024)
- [52] Ma, L., Li, X., Liao, J., Zhang, Q., Wang, X., Wang, J., Sander, P.V.: Deblur-nerf: Neural radiance fields from blurry images. arXiv preprint arXiv:2111.14292 (2021)
- [53] Mildenhall, B., Srinivasan, P.P., Ortiz-Cayon, R., Kalantari, N.K., Ramamoorthi, R., Ng, R., Kar, A.: Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Trans. Graph. (2019)

- [54] Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: ECCV (2020)
- [55] Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)
- [56] Park, K., Sinha, U., Barron, J.T., Bouaziz, S., Goldman, D.B., Seitz, S.M., Martin-Brualla, R.: Nerfies: Deformable neural radiance fields. In: ICCV (2021)
- [57] Park, K., Sinha, U., Hedman, P., Barron, J.T., Bouaziz, S., Goldman, D.B., Martin-Brualla, R., Seitz, S.M.: Hypernerf: a higher-dimensional representation for topologically varying neural radiance fields. ACM Trans. Graph. (2021)
- [58] Peng, S., Yan, Y., Shuai, Q., Bao, H., Zhou, X.: Representing volumetric videos as dynamic mlp maps. In: CVPR (2023)
- [59] Peng, S., Zhang, Y., Xu, Y., Wang, Q., Shuai, Q., Bao, H., Zhou, X.: Neural body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans. In: CVPR

(2021)

- [60] Penner, E., Zhang, L.: Soft 3d reconstruction for view synthesis. ACM Trans. Graph. (2017)
- [61] Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022)
- [62] Pumarola, A., Corona, E., Pons-Moll, G., Moreno-Noguer, F.: D-nerf: Neural radiance fields for dynamic scenes. In: CVPR (2021)
- [63] Rebain, D., Jiang, W., Yazdani, S., Li, K., Yi, K.M., Tagliasacchi, A.: Derf: Decomposed radiance fields. In: CVPR (2021)
- [64] Reiser, C., Peng, S., Liao, Y., Geiger, A.: Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps. In: CVPR (2021)
- [65] Reiser, C., Szeliski, R., Verbin, D., Srinivasan, P.P., Mildenhall, B., Geiger, A., Barron, J.T., Hedman, P.: Merf: Memory-efficient radiance fields for real-time view synthesis in unbounded scenes. arXiv preprint arXiv:2302.12249 (2023)
- [66] Riegler, G., Koltun, V.: Free view synthesis. In: ECCV (2020)
- [67] Sara Fridovich-Keil and Giacomo Meanti, Warburg, F.R., Recht, B., Kanazawa, A.: K-planes: Explicit radiance fields in space, time, and appearance. In: CVPR (2023)
- [68] Shao, R., Zheng, Z., Tu, H., Liu, B., Zhang, H., Liu, Y.: Tensor4d: Efficient neural 4d decomposition for high-fidelity dynamic reconstruction and rendering. In: CVPR (2023)
- [69] Shuai, Q., Guo, H., Xu, Z., Lin, H., Peng, S., Bao, H., Zhou, X.: Real-time view synthesis for large scenes with millions of square meters (2024)
- [70] Singer, U., Sheynin, S., Polyak, A., Ashual, O., Makarov, I., Kokkinos, F., Goyal, N., Vedaldi, A., Parikh, D., Johnson, J., et al.: Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280 (2023)
- [71] Sitzmann, V., Thies, J., Heide, F., Nießner, M., Wetzstein, G., Zollhofer, M.: Deepvoxels: Learning persistent 3d feature embeddings. In: CVPR (2019)
- [72] Song, L., Chen, A., Li, Z., Chen, Z., Chen, L., Yuan, J., Xu, Y., Geiger, A.: Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields. arXiv preprint arXiv:2210.15947

(2022)

- [73] Srinivasan, P.P., Mildenhall, B., Tancik, M., Barron, J.T., Tucker, R., Snavely, N.: Lighthouse: Predicting lighting volumes for spatially-coherent illumination. In: CVPR (2020)
- [74] Srinivasan, P.P., Tucker, R., Barron, J.T., Ramamoorthi, R., Ng, R., Snavely, N.: Pushing the boundaries of view extrapolation with multiplane images. In: CVPR (2019)
- [75] Su, S.Y., Yu, F., Zollhöfer, M., Rhodin, H.: A-nerf: Articulated neural radiance fields for learning human shape, appearance, and pose. In: NeurIPS (2021)
- [76] Tang, J., Chen, Z., Chen, X., Wang, T., Zeng, G., Liu, Z.: Lgm: Large multi-view gaussian model for high-resolution 3d content creation. arXiv preprint arXiv:2402.05054 (2024)
- [77] Thies, J., Zollhöfer, M., Nießner, M.: Deferred neural rendering: Image synthesis using neural textures. ACM Trans. Graph. (2019)
- [78] Tretschk, E., Tewari, A., Golyanik, V., Zollhöfer, M., Lassner, C., Theobalt, C.: Non-rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video. In: ICCV

(2021)

- [79] Tucker, R., Snavely, N.: Single-view view synthesis with multiplane images. In: CVPR (2020)

- [80] Voleti, V., Yao, C.H., Boss, M., Letts, A., Pankratz, D., Tochilkin, D., Laforte, C., Rombach, R., Jampani, V.: Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. arXiv preprint arXiv: 2403.12008 (2024)
- [81] Waechter, M., Moehrle, N., Goesele, M.: Let there be color! large-scale texturing of 3d reconstructions. In: ECCV (2014)
- [82] Wang, F., Chen, Z., Wang, G., Song, Y., Liu, H.: Masked space-time hash encoding for efficient dynamic scene reconstruction. In: NeurIPS (2023)
- [83] Wang, F., Tan, S., Li, X., Tian, Z., Liu, H.: Mixed neural voxels for fast multi-view video synthesis. arXiv preprint arXiv:2212.00190 (2022)
- [84] Wang, H., Ren, J., Huang, Z., Olszewski, K., Chai, M., Fu, Y., Tulyakov, S.: R2l: Distilling neural radiance field to neural light field for efficient novel view synthesis. In: ECCV (2022)
- [85] Wang, L., Zhang, J., Liu, X., Zhao, F., Zhang, Y., Zhang, Y., Wu, M., Yu, J., Xu, L.: Fourier plenoctrees for dynamic radiance field rendering in real-time. In: CVPR (2022)
- [86] Wang, P., Liu, L., Liu, Y., Theobalt, C., Komura, T., Wang, W.: Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. In: NeurIPS (2021)
- [87] Wang, X., Wang, Y., Ye, J., Wang, Z., Sun, F., Liu, P., Wang, L., Sun, K., Wang, X., He, B.: Animatabledreamer: Text-guided non-rigid 3d model generation and reconstruction with canonical score distillation. arXiv preprint arXiv:2312.03795 (2023)
- [88] Wang, Y., Dong, Y., Sun, F., Yang, X.: Root pose decomposition towards generic non-rigid 3d reconstruction with monocular videos. In: ICCV (2023)
- [89] Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., Zhu, J.: Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In: NeurIPS (2023)
- [90] Wang, Z., Wang, Y., Chen, Y., Xiang, C., Chen, S., Yu, D., Li, C., Su, H., Zhu, J.: Crm: Single image to 3d textured mesh with convolutional reconstruction model. arXiv preprint arXiv:2403.05034 (2024)
- [91] Wang, Z., Li, L., Shen, Z., Shen, L., Bo, L.: 4k-nerf: High fidelity neural radiance fields at ultra high resolutions. arXiv preprint arXiv:2212.04701 (2022)
- [92] Wood, D.N., Azuma, D.I., Aldinger, K., Curless, B., Duchamp, T., Salesin, D.H., Stuetzle, W.: Surface light fields for 3d photography. In: Proceedings of the 27th annual conference on Computer graphics and interactive techniques (2000)
- [93] Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Xinggang, W.: 4d gaussian splatting for real-time dynamic scene rendering. arXiv preprint arXiv:2310.08528 (2023)
- [94] Xian, W., Huang, J.B., Kopf, J., Kim, C.: Space-time neural irradiance fields for free-viewpoint video. In: CVPR (2021)
- [95] Xu, Y., Shi, Z., Yifan, W., Peng, S., Yang, C., Shen, Y., Gordon, W.: Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation. arXiv preprint arXiv: 2403.14621 (2024)
- [96] Yang, G., Sun, D., Jampani, V., Vlasic, D., Cole, F., Chang, H., Ramanan, D., Freeman, W.T., Liu, C.: LASR: learning articulated shape reconstruction from a monocular video. In: ICCV (2021)
- [97] Yang, G., Vo, M., Neverova, N., Ramanan, D., Vedaldi, A., Joo, H.: Banmo: Building animatable 3d neural models from many casual videos. In: CVPR (2022)
- [98] Yang, G., Wang, C., Reddy, N.D., Ramanan, D.: Reconstructing animatable categories from videos. In: CVPR (2023)
- [99] Yang, Z., Gao, X., Zhou, W., Jiao, S., Zhang, Y., Jin, X.: Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. arXiv preprint arXiv:2309.13101 (2023)
- [100] Yariv, L., Kasten, Y., Moran, D., Galun, M., Atzmon, M., Basri, R., Lipman, Y.: Multiview neural surface reconstruction by disentangling geometry and appearance. In: NeurIPS (2020)
- [101] Yu, A., Li, R., Tancik, M., Li, H., Ng, R., Kanazawa, A.: Plenoctrees for real-time rendering of neural radiance fields. In: CVPR (2021)
- [102] Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: CVPR (2018)
- [103] Zhao, F., Yang, W., Zhang, J., Lin, P., Zhang, Y., Yu, J., Xu, L.: Humannerf: Efficiently generated human radiance field from sparse inputs. In: CVPR (2022)
- [104] Zhou, T., Tucker, R., Flynn, J., Fyffe, G., Snavely, N.: Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817 (2018)
- [105] Zou, Z.X., Yu, Z., Guo, Y.C., Li, Y., Liang, D., Cao, Y.P., Zhang, S.H.: Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers. arXiv preprint arXiv:2312.09147

(2023)

### A Appendix / supplemental material

Table 2: A summary of important symbols in DGS.

Symbol Definition and Usage t∗u ∈ R3×1, t∗v ∈ R3×1 Principal tangential vectors in the static state. s∗u ∈ R, s∗v ∈ R Scaling factors in the static state. p∗k ∈ R3×1 Center point coordinate (world space) of the k-th Gaussian surfel in the static state. Pk∗(u) ∈ R3×1 Coordinate (world space) in the static state, given u = (u, v) on the local uv coordinate system centered at p∗k.

- R∗k = [t∗u, t∗v, t∗u × t∗v] ∈ SO(3) Rotation matrix of the k-th Gaussian surfel in the static state.
- S∗k = diag(s∗u, s∗v, 0) ∈ R3×3 Scaling matrix of the k-th Gaussian surfel in the static state, a diagonal matrix.

ptk ∈ R3×1 Center point coordinate (world space) of the k-th Gaussian surfel in the warped state. Pkt(u) ∈ R3×1 Coordinate (world space) in the warped state, given u = (u, v) on the local uv coordinate system centered at ptk.

c∗b ∈ R3×1, Vb∗ ∈ R3×3, Λ∗b ∈ R3×3 Center, rotation matrix, and diagonal scaling matrix of the b-th Gaussian ellipsoid bone. wt ∈ RB×1 Skinning weight vectors.

Jtb ∈ SE(3) A rigid transformation that moves the b-th bone from its static state to the warped state at time t. Jt = [R˜ t, T˜t] ∈ SE(3) The warping function, a weighted combination of Jtb. Q, R The quaternion process and the inverse quaternion process.

ωbt ∈ R128 A learnable latent code for representing the body pose at time t. nk ∈ R3×1 The normal of the k-intersected Gaussian surfel that is oriented towards the camera. Nt ∈ R3×1 The surface normal estimated by the nearby depth point pt at warped state time t.

- ∆R∗k ∈ SO(3) Learnable refinement term for adjusting R∗k.
- ∆S∗k ∈ SO(3) Learnable refinement term for adjusting S∗k.

#### A.1 Ablation: Field Initialization and Refinement

In dynamic videos captured in the wild, one of the primary challenges is the initialization of camera poses. In synthetic videos, preserving temporal consistency in texture and geometry is problematic, which significantly complicates the task of camera registration. To address this, we utilize an implicit field to both initialize the camera poses and establish the warping field. Initially, we estimate the transformation for each frame, followed by the computation of coarse camera poses through an iterative process. Subsequently, we adopt the approach outlined in NeuS [86] for scene representation. Feature extraction is performed using DinoV2 [55], facilitating unsupervised registration. To enhance this process, we train an additional channel in NeuS specifically for rendering features, which are then employed for registration purposes as described in RAC [98]. The camera poses without initialization and refined camera poses are depicted in Fig. 6. Without field initialization, the performance of DGS will degrade, as shown in Table 3. Also, please refer to the quantitative ablation of refinement in Table 3.

Table 3: Quantitative ablation studies of the initialization and refinement.

Cat Cheetah Dragon

PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Ours w.o. init. 20.15 0.7961 0.2393 20.96 0.8194 0.1940 25.33 0.9146 0.0938 Ours w.o. refinement 24.19 0.8196 0.1797 24.10 0.8582 0.1242 27.71 0.9128 0.0687 Ours full 24.63 0.8432 0.1559 25.68 0.8843 0.1117 28.58 0.9392 0.0618

#### A.2 Additional Qualitative Comparison

In this section, we present a detailed comparison of our results with previous works, as illustrated in Fig. 7-10. Our method consistently achieves high-quality texture details while maintaining smooth and realistic geometry.

###### A.3 Interpolation on Time and Views We present results for interpolation on time and views, as illustrated in Fig. 11 and Fig. 12.

[Figure 317]

[Figure 318]

(a) Camera poses without field initialization (b) Refined camera poses

Figure 6: Coarse camera poses and refined camera poses.

[Figure 319]

BANMo

D-NeRF

Deformable-GS

SCGS

Deformable-GS w. Poses

SCGS w. Poses

###### DGS (Ours)

- Figure 7: Additional qualitative comparison with more novel views.

[Figure 320]

BANMo

D-NeRF

Deformable-GS

SCGS

Deformable-GS w. Poses

SCGS w. Poses

###### DGS (Ours)

- Figure 8: Additional qualitative comparison with more novel views.

[Figure 321]

BANMo

D-NeRF

Deformable-GS

SCGS

Deformable-GS w. Poses

SCGS w. Poses

###### DGS (Ours)

- Figure 9: Additional qualitative comparison with more novel views.

[Figure 322]

BANMo

D-NeRF

Deformable-GS

SCGS

Deformable-GS w. Poses

SCGS w. Poses

###### DGS (Ours)

- Figure 10: Additional qualitative comparison with more novel views.

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

###### Figure 11: Interpolation on time and views.

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

###### Figure 12: Interpolation on time and views.

