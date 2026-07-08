## Shape-for-Motion: Precise and Consistent Video Editing With 3D Proxy

Yuhao Liu

Tengfei Wang†

Fang Liu

yuhliu9-c@my.cityu.edu.hk City University of Hong Kong Hong Kong SAR, China

tfwang@connect.ust.hk Tencent China

fliu66-c@cityu.edu.hk City University of Hong Kong Hong Kong SAR, China

Zhenwei Wang

Rynson W.H. Lau†

zhenwwang2-c@my.cityu.edu.hk City University of Hong Kong Hong Kong SAR, China

Rynson.Lau@cityu.edu.hk City University of Hong Kong Hong Kong SAR, China

# arXiv:2506.22432v2[cs.CV]26Sep2025

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Pose Editing

InputVideoEditedVideoInputVideoEditedVideo

[Figure 6]

Reconstruction

[Figure 7]

[Figure 8]

∆𝑧 = 50o ∆𝑧 = -10o

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Rendering

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Object Composition

[Figure 20]

Reconstruction

[Figure 21]

Composition

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Rendering

Animation

Input Image Animated Video

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

∆𝑥 = -30o

∆𝑥=20o

[Figure 35]

[Figure 36]

Reconstruction Rendering

Figure 1: The proposed 3D-aware framework, Shape-for-Motion, supports precise and consistent video editing by reconstructing an editable 3D mesh to serve as control signals for video generation. The first two examples demonstrate pose editing (by rotating the back to the right by 50 degrees and the head to the left by 10 degrees) and object composition (by composing a tree from the reference image onto the top of the car). In each example, the first row shows the input video frames, followed by the editing in 3D space at the right end; the bottom row of images shows the corresponding edited frames. In addition, our approach also supports diverse applications, such as Image-to-Video Animation, as shown in the third example.

### Abstract

Recent advances in deep generative modeling have unlocked unprecedented opportunities for video synthesis. In real-world applications, however, users often seek tools to faithfully realize their creative editing intentions with precise and consistent control. Despite the progress achieved by existing methods, ensuring fine-grained

†Co-corresponding authors.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2137-3/2025/12 https://doi.org/10.1145/3757377.3763816

SA Conference Papers ’25, Hong Kong, Hong Kong

alignment with user intentions remains an open and challenging problem. In this work, we present Shape-for-Motion, a novel framework that incorporates a 3D proxy for precise and consistent video editing. Shape-for-Motion achieves this by converting the target object in the input video to a time-consistent mesh, i.e., a 3D proxy, allowing edits to be performed directly on the proxy and then inferred back to the video frames. To simplify the editing process, we design a novel Dual-Propagation Strategy that allows users to perform edits on the 3D mesh of a single frame, and the edits are then automatically propagated to the 3D meshes of the other frames. The 3D meshes for different frames are further projected onto the

- 2D space to produce the edited geometry and texture renderings, which serve as inputs to a decoupled video diffusion model for generating edited results. Our framework supports various precise and physically-consistent manipulations across the video frames, including pose editing, rotation, scaling, translation, texture modification, and object composition. Our approach marks a key step toward high-quality, controllable video editing workflows. Extensive experiments demonstrate the superiority and effectiveness of our approach. Project Page: https://shapeformotion.github.io.

CCS Concepts

• Computing methodologies → Computer vision.

Keywords

- 3D-Aware Video Editing, Generative Model

ACM Reference Format:

Yuhao Liu, Tengfei Wang, Fang Liu, Zhenwei Wang, and Rynson W.H. Lau. 2025. Shape-for-Motion: Precise and Consistent Video Editing With 3D Proxy. In SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers ’25), December 15–18, 2025, Hong Kong, Hong Kong. ACM, New York, NY, USA, 19 pages. https://doi.org/10.1145/3757377.3763816

### 1 Introduction

The rapid proliferation of video content on online platforms, combined with the explosive emergence of video generation models, has fueled the surge in video content generation. Recently, controllable video editing, which focuses on modifying the source video content to align with user intentions, has gained increasing attention.

Early approaches [Bar-Tal et al. 2022; Wu et al. 2023] use text as the interaction signal, but text often lacks precision and flexibility for editing spatial attributes. Besides text, image-based methods [Ku et al. 2024; Ouyang et al. 2024] introduce an edited image as guidance and propagate edits across frames, while drag-based methods [Deng et al. 2024; Teng et al. 2023] propose to manually drag anchor points for localized adjustments. However, they struggle in handling complex editing and ensuring frame-to-frame consistency.

Unlike these existing works, our objective in this work is to develop a video editing system with two key features. 1) Precision. Accurate controllability gives users precise control over various aspects of the video elements, including object pose, shape, location, and spatial layout. Such fine-grained control often extends to object attributes with quantifiable precision. For example, rotating the panda in the third example of Fig. 1 by 20 degrees to the left requires precise manipulation. 2) Consistency. Consistent alignment demands edits to remain coherent across frames. For

example, placing a tree on the moving car in the second example of Fig. 1 requires alignment that simultaneously accounts for the car’s motion, rotation, and changing perspectives. While these two key features are essential for video editing, it is non-trivial to achieve this with a 2D framework, due to the absence of underlying 3D representations.

To address the above challenges, in this paper, we introduce Shape-for-Motion, a novel video editing framework that incorporates a 3D proxy (i.e., mesh) to enable precise editing while maintaining temporal and spatial consistency. Our framework follows a 3D-aware workflow by first reconstructing a 3D proxy of the target object from the video, followed by interactive manipulation in the 3D space, and finally, producing a video with the help of the edited 3D proxy. We leverage three key designs to ensure (1) temporalconsistent 3D proxy reconstruction, (2) consistent 3D editing across frames, and (3) generative rendering from edited 3D to video.

The first is a consistent mesh reconstruction of the target object. We note that reconstructing a separate mesh for each frame individually [Tang et al. 2025] leads to poor consistency due to the lack of inter-frame correspondences. To address this problem, we propose to reconstruct a consistent mesh representation for the object across all frames using a canonical mesh with a time-varying deformation field. However, the limited viewpoint information in a monocular input video often results in unsatisfactory reconstruction. To mitigate this, we leverage novel views generated from an existing multi-view generator [Voleti et al. 2025] to enhance mesh reconstruction and propose a balanced view sampling strategy to alleviate inconsistencies caused by the generated novel views.

The second design is automatic editing propagation on 3D mesh. While editing the mesh in each frame individually suffices for simple global operations, e.g., global rotation or scaling, it becomes impractical for more complex tasks, like localized pose editing as shown in the first example of Fig. 1. To enable a user-friendly editing process (in which users only need to edit the 3D mesh of a single frame, and the editing is then automatically propagated to other frames), we propose a novel Dual-Propagation Strategy, which utilizes learned correspondences between canonical and deformed meshes to propagate geometry and texture edits across frames, enabling the generation of consistent editing.

The final key design is generative rendering from the edited 3D proxy. Once we have edited the object in 3D space, we then need to convert the edited 3D meshes to a high-quality edited video. However, it is difficult to train such a generation model, due to the absence of paired training data (3D mesh and corresponding video). To address this challenge, we adopt a decoupled control strategy in our video diffusion model, treating geometry and texture information from the edited proxy as two separate conditioning signals. A self-supervised mixed training strategy is proposed to alternate between geometry and texture controls, enabling the model to preserve its generative capabilities while achieving temporal and spatial consistency across frames.

In summary, Shape-for-Motion allows users to perform finegrained geometry control (e.g., pose editing, rotation, scaling, translation, and object composition) and texture modification in 3D space. It also supports appearance editing in 2D space by incorporating 2D editing tools. Extensive experiments and user studies on the new

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

∆𝑥 = 20

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Geometry Controller

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Step I Step II Step III

[Figure 56]

Canonical Mesh

Edited Canonical Mesh

###### Propagate

Sec. 3.1 Sec. 3.2 Sec. 3.3

[Figure 57]

…

…

[Figure 58]

[Figure 59]

[Figure 60]

###### …

…

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

…

[Figure 67]

Rendering

[Figure 68]

[Figure 69]

|Time 𝑡|
|---|

𝜃

VideoFrame

[Figure 70]

[Figure 71]

Denoising U-Net

Frame 1

Deformation Field

Frame 0

Texture Geometry

[Figure 72]

Consistent 3D Proxy Reconstruction Editing Propagation

Generative Rendering from the 3D Proxy Edited Video

Input Video

[Figure 73]

Figure 2: Overview of Shape-for-Motion. Our approach is an interactive video editing framework that utilizes an editable 3D proxy (e.g., mesh) to enable users to perform precise and consistent video editing. Given an input video, our approach first converts the target object into a 3D mesh with frame-by-frame correspondences. Users can then perform editing on this 3D proxy once only (i.e., on a single frame), and the edits will be automatically propagated to the 3D meshes of all other frames. The edited 3D meshes are then converted back to 2D geometry and texture maps, which are used as control signals in a decoupled video diffusion model to generate the final edited result. Note that, for better visualization, the colors of the propagated meshes are disabled in Step-II.

V3DBench dataset demonstrate that our approach generates temporally consistent video edits, and outperforms existing methods qualitatively and quantitatively. Our main contributions are:

generation [Bahmani et al. 2024; Jiang et al. 2024] has also attracted huge research interests.

### 2.2 Controllable Video Editing

- • We propose Shape-for-Motion, a novel video editing framework that incorporates a 3D proxy to allow diverse and precise video object manipulation while ensuring temporal consistency across frames.
- • We propose a video object reconstruction method, which produces meshes with correspondences across frames, and a dual propagation strategy. Together, these components enable a user-friendly editing process: users can perform edits directly on the canonical 3D mesh only once, with the edits are automatically propagated to subsequent frames.
- • We propose a self-supervised mixed training strategy to train a decoupled video diffusion model that leverages geometry and texture information from the edited 3D proxy as control signals, achieving more consistent editing results.

To bridge image and video modalities, early methods like Layered Neural Atlases [Kasten et al. 2021] decompose frames into foreground and background atlases, each with a dedicated map for appearance and geometry, enabling independent object editing and seamless compositing. In recent years, Control-based methods [Ma et al. 2023; Wang et al. 2022; Zhang et al. 2023b] are proposed to utilize additional reference maps (e.g., depth or edge maps) to modify object motion and shape. However, creating references for each frame by the user is impractical. VideoSwap [Gu et al. 2024] leverages sparse semantic point correspondences to enable shape change in swapped results while aligning with the source motion trajectory. First-frame-based methods further reduce this burden by editing on the first frame only and propagating [Kagaya et al. 2010] the changes to subsequent frames, either by attention manipulation [Ceylan et al. 2023; Fan et al. 2024; Ku et al. 2024] or a fine-tuned model manner [Liu et al. 2025; Mou et al. 2024]. Recently, drag-based methods [Deng et al. 2024; Pan et al. 2023; Teng et al. 2023] allow users to modify the shape of an object by dragging its local region from a start point to an endpoint. Concurrent works, GS-Dit [Bian et al. 2025] and Diffusion-as-Shader [Gu et al. 2025] represent subjects using point clouds with explicit tracking, offering faster speed without requiring explicit object reconstruction. The most closely related work, VideoHandles [Koo et al. 2025], edits 3D object compositions in videos of static scenes with camera motion. In contrast, our method handles dynamic objects with independent motions, enabling precise and consistent video editing.

- 2 Related Works

- 2.1 Controllable Video Generation

Recent success in diffusion models [Ho et al. 2020; Song et al. 2021] have fueled significant progress in image [Liang et al. 2025; Rombach et al. 2022] and video generation [Brooks et al. 2024]. Earlier Text-to-Video (T2V) models [Blattmann et al. 2023; Ho et al. 2022] are mostly evolved from Text-to-Image (T2I) models by incorporating additional temporal layers. Later, Image-to-Video (I2V) methods [Guo et al. 2024b,a] are proposed to animate an image to produce videos. Methods like [Chen et al. 2023; Esser et al. 2023; Hu and Xu 2023; Huang et al. 2025; Yang et al. 2024b] are also proposed to generate videos by conditioning on a sequence of control frames. Several works [Cai et al. 2024; Lv et al. 2024; Shi et al. 2024] also explore the application of 3D priors for video generation, either as direct inputs or as intermediate outputs to enhance the generation quality. Recently, incorporating video priors for controllable 4D

### 2.3 Image Editing Using 3D Proxy

Recently, a growing body of methods has sought to integrate 3D priors into image editing. 3DIT [Michel et al. 2023] introduces an object-centric image editing framework that edits objects based on

language instructions. Diffusion Handles [Pandey et al. 2024] edits selected objects by leveraging depth to lift diffusion activations into

[Figure 74]

[Figure 75]

𝑡 𝜃

[Figure 76]

[Figure 77]

[Figure 78]

DPSR Diff.MC

[Figure 79]

- 3D space. The most related works to ours are Image Sculpting [Yenphraphai et al. 2024] and 3D-Fixup [Cheng et al. 2025], which use 3D mesh to support precise image editing. In contrast, our approach enables precise and consistent video editing with 3d proxy.

[Figure 80]

[Figure 81]

Canonical Gaussians Deformed Mesh Mesh Image

Deformed Gaussians

[Figure 82]

||[Figure 83]|
|---|
<br><br>|[Figure 84]|
|---|
<br><br>[Figure 85]<br><br>[Figure 86]<br><br>… …<br><br>Multi-Views Depth of Multi-Views| |
|---|---|
| | |
| |Loss|

[Figure 87]

### 2.4 Dynamic 3D Reconstruction

Compared to NeRF [Mildenhall et al. 2021; Pumarola et al. 2021], 3D Gaussian Splatting (3DGS) [Kerbl et al. 2023] supports both high-quality rendering and real-time efficiency. When extended to dynamic scenes, deformable 3DGS incorporates explicit temporal changes of the canonical points [Luiten et al. 2023; Wu et al. 2024b; Yang et al. 2024a]. However, the point-based representation is inherently less suited for precise editing. Most recently, DG-Mesh [Liu et al. 2024b] integrates deformable 3DGS for mesh reconstruction, and can be used to handle dynamic scenes. However, it requires 360◦ capturing of the target object, which is not applicable for general videos. In contrast, our method aims at reconstructing a consistent 3D mesh from a general monocular video, which typically has limited views of the target object. Inspired by multi-view generation [Shi et al. 2023b; Wang et al. 2025], several methods [Ren et al. 2024; Wu et al. 2024a; Xie et al. 2024] leverage diffusion models to produce novel views for a video, followed by 4D reconstruction. However, these methods lack explicit mesh correspondences across frames, limiting their applicability for video editing.

Depth

Frame t

Figure 3: Pipeline of our consistent object reconstruction. We use deformable-3DGS to reconstruct the 3D mesh of the target object in a video by maintaining canonical Gaussians and a time-varying deformation field 𝜃. To achieve complete reconstruction, in addition to the standard Gaussian Splatting (GS) loss, we incorporate multi-view augmentation as additional constraints.

training a backward deformation field 𝜃−1(·), we map the deformed mesh back to the canonical space at each time𝑡, and query the vertex colors via 𝑀𝐿𝑃𝑐(·). Finally, we perform differentiable rasterization [Laine et al. 2020] to render the depth map, mesh image, and mask of the deformed mesh. Meanwhile, Gaussian image is rendered from the deformed Gaussians.

Dynamic Object Reconstruction with Novel-View Augmentation. Although deformable 3DGS can provide frame-to-frame correspondences, it relies heavily on accurate multi-view observations of a dynamic scene for reconstruction. However, most video clips are typically captured from a certain view, lacking contents from other viewpoints across frames, which in turn leads to unsatisfactory reconstruction [Liu et al. 2024b; Yang et al. 2024a]. To address this, we propose to leverage multi-view diffusion models [Voleti et al. 2025] to assist in dynamic object reconstruction. Given a video 𝑋𝑠𝑟𝑐 = {𝑋𝑖𝑠𝑟𝑐|𝑖 = 1,2, . . . ,𝑇}, where T is the number of frames, we utilize a multi-view generator Mdiff to generate 𝑁 (imperfect) novel views V = {𝑉𝑖,𝑗 | 𝑖 = 1, 2, . . .,𝑇; 𝑗 = 1, 2, . . ., 𝑁} with associated camera pose. We can then combine these novel views with input frames to optimize the reconstruction of the target object.

3 OUR APPROACH

Given an input video, we aim to enable precise manipulation of the target object. To this end, we propose a novel video editing pipeline (see Fig. 2) consisting of three key steps. First, we convert the target object into its 3D mesh to provide a consistent geometry structure for editing (Sec. 3.1). Second, we propose a new Dual-Propagation Strategy to enable precise and controllable edits in 3D space, which are then consistently propagated across frames (Sec. 3.2). Third, we introduce a self-supervised mixed training strategy for a decoupled video diffusion model, which uses geometry and texture renderings as control signals to produce more consistent results. (Sec. 3.3).

### 3.1 Consistent 3D Proxy Reconstruction

Balanced View Sampling. Since 3D serves merely as a proxy during the video editing process, users primarily focus on editing the observed view 𝑋src, while the generated novel views V are employed mainly to enhance the completeness of the reconstruction. However, trivially combining them inevitably amplifies inter-frame and intra-frame inconsistencies arising from the target object variations and generated novel views, leading to incomplete or irregular geometry (See row-2 in Fig. 7). To this end, we introduce a Balanced View Sampling to alleviate the inconsistency between observed and generated views. We define two sampling probabilities: 𝛽𝑖 for each observed view 𝑋𝑖𝑠𝑟𝑐, and 𝜁𝑖,𝑗 for each novel view𝑉𝑖,𝑗 in V, representing their respective selection probabilities during sampling. Since the number of novel views is 𝑁 times larger than that of the observed view at each frame 𝑖, an equal sampling strategy would disproportionately favor the novel views, amplifying inconsistencies due to their synthetic nature. Thus, we enforce the constraint that the total sampling probability of all views in V𝑖 equals to that of

One naive way to obtain the 3D proxy is to reconstruct each frame individually. However, this separate modeling lacks frame-to-frame correspondences, resulting in poor temporal consistency. To address this, we adopt the deformable Gaussian Splatting [Yang et al. 2024a] to reconstruct the target object in the input video. Furthermore, manipulating Gaussian points is not user-friendly and only supports limited editing types. To enable 3D-guided video editing, we aim to obtain consistent 3D meshes as the editing proxy.

Overall Pipeline. The workflow is illustrated in Fig. 3. At time 𝑡, we first obtain the deformation of the canonical Gaussians from the deformation field encoded with a learnable MLP 𝜃(·) to form the deformed Gaussians. We then utilize DPSR [Peng et al. 2021] combined with Marching Cubes (MC) to convert the deformed Gaussians into a Gaussian-propagated deformed mesh. To obtain the vertex colors of the deformed mesh, we follow [Liu et al. 2024b] to store the texture in a canonical color MLP, denoted as𝑀𝐿𝑃𝑐(·). By

the observed view 𝑋𝑖𝑠𝑟𝑐, i.e., 𝛽𝑖 = 𝑁𝑗=1 𝜁𝑖,𝑗. This inherently assigns a lower sampling frequency to each novel view in 𝑉𝑖 compared to

[Figure 88]

[Figure 89]

[Figure 90]

Texture Propagation

𝑋𝑖𝑠𝑟𝑐, reducing inconsistencies caused by the novel views. Loss function. The overall training objective is:

L = L𝑔𝑠 + L𝑚𝑎𝑠𝑘 + L𝑟𝑔𝑏 + L𝑑𝑒𝑝𝑡ℎ. (1)

Mesh-propagated Deformed Mesh 𝑀

Mesh-propagated Edited Mesh 𝑀

Canonical Mesh 𝑀

We employ a combination (i.e., L𝑔𝑠) of L1 loss and SSIM loss on the Gaussian images, using the input video frames and generated novel views as supervision. To constrain the shape of the reconstructed mesh from different views, we apply an L1 loss (i.e., L𝑚𝑎𝑠𝑘). Additionally, we adopt the same rendering loss as L𝑔𝑠 to supervise the mesh image (i.e., L𝑟𝑔𝑏). To mitigate sunken surfaces and floating artifacts in reconstructed meshes, we further enforce a scale-invariant depth constraint L𝑑𝑒𝑝𝑡ℎ between mesh depth and GT image depth.

###### Mesh Editing

𝜃 t

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Geometry Propagation

Edited Canonical Mesh 𝑀 ′

Gaussian-propagated Edited Mesh 𝑀

Propagated Mesh 𝑀

### 3.2 Editing 3D Proxy with Automatic Propagation

To enable a user-friendly editing process, we aim to transfer crossframe correspondences in deformable Gaussian splatting to the reconstructed mesh, such that users can edit the canonical mesh once, with the edits propagated to all frames in 3D space automatically. Formally, given a canonical Gaussian 𝐺𝑐, we first convert it to the canonical mesh 𝑀𝑐 via Marching Cube (MC). The user then performs edits directly on 𝑀𝑐, yielding the edited canonical mesh 𝑀𝑐′ and thus the editing offset in the canonical space is Δ𝑚 = 𝑀𝑐′ − 𝑀𝑐.

Figure 4: Workflow of our consistent editing propagation. Solid orange arrows denote texture propagation from the canonical mesh, and solid blue arrows indicate geometry propagation from canonical Gaussians.

Instead, we note that although the mesh-propagated edited mesh 𝑀˜𝑡𝑒, may contain geometric errors due to outliers in vertices and faces, the majority of its color remains correct. Thus, we establish an additional mapping 𝜉(·) (similar to Eq. 2) between the two edited meshes, 𝑀˜𝑡𝑒 and 𝑀ˆ𝑡𝑒, which are propagated from the canonical mesh and canonical Gaussians, respectively. We can then retrieve the color information from 𝑀˜𝑡𝑒 and map it to 𝑀ˆ𝑡𝑒. Specifically, we first obtain the color 𝐶˜𝑡𝑑 of deformed mesh 𝑀˜𝑡𝑑 before editing. As the number of vertex is the same for 𝑀˜𝑡𝑑 and 𝑀˜𝑡𝑒 before and after geometry editing, the color of the mesh-propagated edited mesh can be queried by:

A naive approach to propagate this editing offset is to directly apply the deformation field 1 𝜃(·) to the 𝑀𝑐 to obtain the meshpropagated deformed mesh 𝑀˜𝑡𝑑 = 𝑀𝑐 + 𝜃(𝑀𝑐) at each frame 𝑡, and then integrate the 𝑀˜𝑡𝑑 with the editing offset to produce the meshpropagated edited mesh 𝑀˜𝑡𝑒 = 𝑀˜𝑡𝑑 + Δ𝑚 (see flow indicated by the orange arrow in Fig. 4). However, since the deformation field is optimized for Gaussian points rather than mesh vertices directly, certain vertices may incur positional shifts, leading to inaccurate geometry (see Fig. 8, Variant-1). To this end, we develop a Dual Propagation Strategy that propagates geometry and texture edits using the canonical Gaussians and the canonical mesh, respectively. Geometry Propagation. As the user’s editing is performed in the canonical mesh, we first build a vertex-point mapping to transfer the offset Δ𝑚 for the canonical mesh to Δ𝑔 for the canonical Gaussian.

𝐶˜𝑡𝑒 = 𝐶˜𝑡𝑑 = 𝑀𝐿𝑃𝑐 𝑀 ˜𝑡𝑑 − 𝜃−1(𝑀˜𝑡𝑑) , (3) where𝜃−1(·) is the backward deformation field. Then, we can obtain the propagated texture color 𝐶ˆ𝑡𝑒 = 𝐶˜𝑡𝑒 𝜉(𝑀ˆ𝑡𝑒) for the Gaussianpropagated edited mesh 𝑀ˆ𝑡𝑒, and form the propagated mesh 𝑀𝑡 = {𝑀ˆ𝑡𝑒;𝐶ˆ𝑡𝑒}. Finally, we render it into its 2D representations: normal map for geometry and mesh image for texture.

We introduce a nearest-neighbor mapping from 𝐺𝑐 to 𝑀𝑐. For any gaussian point x in 𝐺𝑐, we seek the closest vertex in 𝑀𝑐 by:

𝜙(x) = arg min

∥x − v∥2. (2)

v ∈ 𝑀𝑐

### 3.3 Generative Rendering from the 3D Proxy

Thus, the editing offset for canonical Gaussian is Δ𝑔 = Δ𝑚(𝜙(𝐺𝑐)). Then, we leverage the correspondence in the deformation field 𝜃(·), to obtain the deformed Gaussian 𝐺𝑡 = 𝐺𝑐 + 𝜃(𝐺𝑐). The Gaussianpropagated edited mesh at frame 𝑡 can thus be produced by integrating 𝐺𝑡 with the editing offset Δ𝑔 by 𝑀ˆ𝑡𝑒 = MC(𝐺𝑡 + Δ𝑔).

With the edited geometry and texture in place, the next step is to synthesize the final refined video renderings by leveraging these signals along with the input frames. To this end, we train a decoupled video diffusion model using a self-supervised mixed training strategy. The overall training pipeline is illustrated in Fig. 5.

Texture Propagation. With 𝑀ˆ𝑡𝑒, one naive method of obtaining the edited texture is to query the color directly by using it as input

Data Construction. Since no such 3D-video paired video editing datasets are available, we generate training data by augmenting the input video to simulate pre- and post-editing states. (1) Pre-editing: The reference object is constructed by the reference generator, which first segments the target object from the original video and then applies random augmentations, including scaling, shifting, and rotation. (2) Post-editing: The original video frames serve as Ground

to the color network 𝑀𝐿𝑃𝑐(·). However, due to vertex misalignment before and after the editing, this method often results in spatially shifted colors (see Fig. 8, Variant-2).

- 1For brevity, we omit the 𝑡 symbol in time-dependent MLPs, e.g., deformation field 𝜃 (·) and color MLP network 𝑀𝐿𝑃𝑐 (·).

#### Table 1: Quantitative comparison of video editing methods. The user study reports the average rank in editing quality (EQ), semantic consistency (SC), and visual plausibility (VP).

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

Training

Frozen

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Noise Normal Target Mask Reference

Reference Simulator

Metrics User Study

[Figure 109]

Methods

[Figure 110]

[Figure 111]

[Figure 112]

Fram-Acc ↑ Tem-Con ↑ CLAP Score ↑ EQ ↓ SC ↓ AP ↓

Geometry Controller

Geometry Controller

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Tune-A-Video 0.510 0.979 0.423 5.35 5.68 5.85 Pix2Video 0.825 0.988 0.647 4.81 5.03 4.65 I2V-Edit 0.856 0.989 0.792 2.23 2.33 2.23 DragVideo 0.875 0.983 0.774 3.27 2.72 2.68 VideoShop 0.812 0.985 0.726 4.21 3.88 3.78 Ours 0.945 0.990 0.878 1.13 1.36 1.81

Texture Enhancer

###### Texture Enhancer

Ground Truth

Output

(Denoising U-Net)

(Denoising U-Net)

[Figure 119]

[Figure 120]

Stage 1 Stage 2

Texture Simulator

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

[Figure 132]

Background Inpainting Mask Background Coarse Texture

In the second stage, the model focuses on texture enhancement by simultaneously refining details within the edited object and inpainting missing regions (e.g., areas filled with white in Fig. 5). Unlike the first stage, this phase introduces an additional set of stage-specific inputs, i.e., an inpainting mask and a coarse texture rendered from the edited 3D proxy. To preserve the model’s ability to follow geometric guidance, the texture control is provided as input with a probability of 20%, while it is replaced by a background image with a probability of 80%. This mixed-training strategy encourages the model to balance appearance preservation when texture information is available, while retaining generative flexibility when it is not.

- Figure 5: Training pipeline of the generative rendering from the 3D proxy. Using rendered geometry (i.e., normal map) and texture as separate control signals, we adopt a self-supervised mixed training strategy in which the geometry controller and the texture enhancer (i.e., denoising U-Net) are alternately trained in two stages. The contents enclosed by the purple dashed box indicates inputs shared by both stages, while those in the orange and blue dashed boxes are specific to the first and second stages, respectively. Reference objects and coarse textures are constructed via the reference and texture simulators to facilitate self-supervised training.

Truth (GT), with masked normal maps extracted as geometry control. Then, the texture simulator generates coarse texture by randomly degrading the GT with SLIC-based segmentation [Achanta et al. 2012], median blurring, and down-up sampling.

### 4 Experiments 4.1 Comparisons with State of the Art Methods

4.1.1 Qualitative Results. To the best of our knowledge, ours is the first work to leverage a 3D proxy for video editing. As existing methods lack such precise editing capabilities, we compare our approach against four baseline methods (i.e., Tune-A-Video [Wu et al. 2023], Pix2Video [Guo et al. 2024b], Image-Sculting+I2V-Edit [Ouyang et al. 2024]), DragVideo [Deng et al. 2024], and Videoshop [Fan et al. 2024]+Zero-1-2-3 [Shi et al. 2023a].

Overall Pipeline. We use the I2V release of the stable video diffusion (SVD) [Blattmann et al. 2023] as the base model, and follow the ControlNet [Zhang et al. 2023a] paradigm. To handle the domain gap between texture and geometry, we employ the base model to refine coarse texture, and utilize the control branch to guide the structure via geometry control. Refer to Supp. C.2 for more details. Mixed Training. A straightforward approach is to train the entire model jointly using two control signals. Nonetheless, the coarse texture is pixel-aligned to the GT, while the distribution of the geometry is very different. It is easy to discard the geometry branch and downgrade the generative model to an enhancement model again. Motivated by the animation production that first draws structure and then coloring, we propose to train the geometry controller and texture enhancer (i.e., base model) in a two-stage manner.

Fig. 6 shows the visual comparison. It clearly demonstrates that when dealing with fine-grained video editing scenarios, such as locally rotating the dog in case-1 and stretching the car’s roof in case-2, all compared methods exhibit varying degrees of failure. Among the compared methods, I2V-Edit produces acceptable geometry editing results, due to reliable 3D editing on the first frame 3. However, as it heavily relies on the first frame for propagation, it often fails when there is a constant geometry/shape variation across frames. For example, the dog’s leg hangs across all frames in case-1. Moreover, since DragVideo only supports point-to-point editing in the 2D space, its limited point coverage is insufficient for fine-grained local editing. As a result, it can extend the car’s roof but fails to account for the entire upper part of the car in case-2, and generates multiple legs for the dog in case-1. As for the prompt-based methods, neither the training-free Pix2Video nor the optimization-based Tune-a-Video can handle fine-grained editing with text prompts. In contrast, our approach achieves high-quality results due to precise and consistent editing in the 3D space and reliable enhancement in the 2D space.

Specifically, in the first stage, we fix the base model and train only the geometry controller. In the second stage, we freeze the trained geometry controller and fine-tune the base model. In both stages, the model receives the same set of shared inputs: random noise, a normal map for geometry control, a target mask, and a reference object 2. The primary difference between the two stages lies in the handling of texture control. In the first stage, texture control is replaced by a background image, where random noise is inserted into the object region. In this way, although the generated appearance may appear coarse (see Fig. 9, Case-2) due to the spatial misalignment between the reference object and the target geometry, the model is encouraged to learn geometry-following generation.

3For a fair comparison, we use the 3D editing-capable Image-Sculpting to edit the first frame.

- 2For simplicity, we omit the process of transforming pixels to latents through the VAE.

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Input

[Figure 139]

[Figure 140]

Change the SUV's sleek, rounded roof to a taller, boxy design.

∆𝑧 = 30𝑜

[Figure 141]

Rotate the back half of the Shiba Inu and tail inward 30o.

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Ours

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

DragVideo

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

Image Sculpting + I2VEdit

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Pix2Video

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Tune-A -Video

- Figure 6: Qualitative comparison with four video editing baselines (DragVideo [Deng et al. 2024], Image Sculpting [Yenphraphai et al. 2024] + I2V-Edit [Ouyang et al. 2024], Pix2Video [Guo et al. 2024b], and Tune-A-Video [Wu et al. 2023]) on our V3DBench dataset samples. Due to their lack of 3D awareness, the baseline methods only achieve limited modifications to the object geometry. Instead, our approach enables direct editing in the 3D space only once, ensuring precise and consistent results for all frames. The first row shows inputs, and the second highlights text instructions (provided solely for illustrating the performed edits) and 3D space edits.

Table 2: Ablation study on consistent 3D proxy reconstruction. All ablations are trained and tested on 21 novel views. BVS: balanced view sampling.

- 4.1.2 Quantitative Results. We collect V3DBench to evaluate our method. V3DBench consists of 70 videos, covering six categories: pose editing, rotation, scaling, translation, texture modification, and object composition. For evaluation, we employ widely used CLIP [Radford et al. 2021]-based metrics, Fram-Acc [Ma et al. 2023] and Tem-Con [Esser et al. 2023]. However, for fine-grained edit types, CLIP cannot measure the appearance consistency of objects before and after editing. To this end, we also introduce CLAP (CLIPAPpearance) Score, to measure the cumulative error of the FramAcc and the DINO similarity accuracy: CLAP Score = Fram-Acc × DINO(𝑖𝑛𝑝𝑢𝑡,𝑜𝑢𝑡𝑝𝑢𝑡), where 𝑖𝑛𝑝𝑢𝑡 and 𝑜𝑢𝑡𝑝𝑢𝑡 represent the input and edited frames. Tab. 1 shows that our method achieves the highest performance, demonstrating the superior editing quality and temporal consistency compared to the baselines. We also conduct a user study to evaluate the perceptual quality in editing quality (EQ), semantic consistency (SC), and visual plausibility (VP). The results indicate that our method consistently outperforms others and is the most preferred.

Novel Views BVS 𝐿𝑑 PSNR ↑ SSIM ↑ LPIPS ↓ DINO Score ↑

20.35 0.903 0.060 0.375 ✓ 22.87 0.922 0.057 0.396 ✓ ✓ 22.91 0.923 0.055 0.396 ✓ ✓ ✓ 23.00 0.923 0.055 0.397

#### Table 3: Ablation study of decoupled video diffusion.

Stage-1 Stage-2 Fram-Acc ↑ Temp-Con ↑ CLAP Score ↑

✓ 0.758 0.987 0.723 ✓ 0.873 0.988 0.769 ✓ ✓ 0.945 0.990 0.878

Input Video

Reconstructed Results

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Observed view Unseen view

Observed view Unseen view

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

Ours

Baseline: w/o novel views

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

Ours

Baseline: w/o balanced view sampling

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

Ours

Baseline: w/o depth constraint

- Figure 7: Comparison of the key components in consistent 3D proxy reconstruction. Each row starts with an input video on the left, followed by results of our full model in the middle and ablated baselines on the right, each consisting of a rendered image and its corresponding normal map.

[Figure 218]

[Figure 219]

𝑠𝑐𝑎𝑙𝑒 0.9

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

Input Ours Variant-1 ({𝑀 ; 𝐶 }) Variant-2 ({𝑀 ; 𝑀𝐿𝑃 (𝑀 )})

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

∆𝑥 = 20𝑜

∆z = −20𝑜

Mesh Editing

- Figure 8: Comparison of editing propagation strategies. The edited 3D mesh is rendered into the texture (i.e., mesh image) and geometry (i.e., normal map). The first column presents the input image at frame 𝑡, followed by the editing in 3D space at this frame in the second column, with results from ours and two variants shown in subsequent columns. Variant-1: mesh-propagated edited mesh with its color. Variant-2: Gaussian-propagated edited mesh with its color.

### 4.2 Ablation Study

highlighting the critical role of novel views as constraints; (2) Further incorporating balanced view sampling and depth constraints leads to additional performance gains (row 3 & 4).

4.2.1 Consistent 3D Proxy Reconstruction. We validate each component by individually removing them, with visual results shown in Fig. 7. It shows that the absence of novel views as constraints leads to wrong reconstructions (row 1). Disabling the balanced sampling strategy increases the sampling frequencies for novel views, which, in turn, results in greater inconsistencies and incomplete reconstructions (row 2). When the depth loss is removed, the mesh images remain relatively unaffected due to their depth-invariance, but the normal map exhibits numerous sunken holes (row 3).

4.2.2 Editing 3D Proxy with Automatic Propagation. We compare two variants for propagating edits. (1) We directly utilize the meshpropagated edited mesh 𝑀˜𝑡𝑒 combined with its vertex color 𝐶˜𝑡𝑒 as the propagated 3D mesh {𝑀˜𝑡𝑒;𝐶˜𝑡𝑒} at frame𝑡. (2) After obtaining the Gaussian-propagated edited mesh 𝑀ˆ𝑡𝑒, we send it to the color network 𝑀𝐿𝑃𝑐(·) to query its vertex colors and get {𝑀ˆ𝑡𝑒;𝑀𝐿𝑃𝑐(𝑀ˆ𝑡𝑒)}.

The results are presented in Tab. 2 in terms of PSNR, SSIM, LPIPS, and DINO score. We can conclude that (1) Using only the monocular video results in poor reconstruction quality, while introducing novel views significantly improves all metrics (row 1 vs. row 2),

We present the comparisons of geometry and texture renderings in Fig. 8. Although the deformation field originally designed for GS points can be applied to mesh vertices, discrepancies in quantity and position between GS points and mesh vertices lead to several

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

Rotation

[Figure 242]

[Figure 243]

∆𝑧 = −20𝑜

Scaling

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Input Video Ours w/o Mixed Training

Mesh Editing

#### Figure 9: Comparison of mixed training in generative rendering from the 3D proxy. For each case, the input video is shown on the left, followed by 3D space editing in the second column, with results from our method and the baseline in the subsequent columns. Two frames are displayed for each case.

[Figure 252]

[Figure 253]

[Figure 254]

| | |
|---|---|
| | |

[Figure 255]

Mesh Editing

[Figure 256]

Reconstruction

[Figure 257]

##### …

𝑇𝑟𝑎𝑛𝑠𝑙𝑎𝑡𝑖𝑜𝑛

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

Input Video

Input Video Reconstructed Mesh Edited Video

[Figure 262]

[Figure 263]

[Figure 264]

… ∆𝑧=-30o ∆𝑧=-20o

[Figure 265]

[Figure 266]

[Figure 267]

Rendering

[Figure 268]

[Figure 269]

Edited Video

[Figure 270]

#### Figure 10: Visual illustration of multi-objects editing.

Input Image Animated Mesh Animated Video

#### Figure 11: The top row illustrates failure cases where our framework struggles to reconstruct and edit objects with highly complex motion and invisible parts. The bottom row shows limitations in handling shadows and reflections.

inaccuracies in the reconstructed geometry and texture. In contrast, the second variant naively queries colors, leading to noticeable color shifts. Our method, instead, establishes a dual propagation for both geometry and texture, thus achieving higher editing consistency.

- 4.2.3 Generative Rendering from 3D Proxy. We evaluate the effectiveness of the mixed training strategy by disabling Stage 2 and present the visual comparisons in Fig. 9. Training exclusively on augmented data causes the model to struggle with view-changing edits (e.g., rotating the penguin 20 degrees to the left) when relying solely on geometry as control. This is because such paired data cannot be accurately simulated during the augmentation process. Additionally, spatial misalignment between the target geometry and the reference object leads to failures in maintaining the correct appearance (e.g., Patrick Star’s face and pants are distorted or lost). Quantitative comparisons in Tab. 3 on the V3DBench dataset also demonstrate that our mixed training strategy significantly improves the Frame-Acc and CLAP Score metrics. 4.3 Discussion
- 4.3.1 Efficiency. We report the runtime and inference cost. The reconstruction stage takes approximately 90 minutes on a single A100 GPU, while generative rendering requires 7 days of training on 8 A100 GPUs. At inference, for 14 frames of 512×512 resolution video, our mesh editing takes between 30 seconds and 10 minutes, depending on the task complexity, and generative rendering takes around 43 seconds on a single A100 GPU. In terms of GPU memory

usage, reconstruction requires 12GB, while diffusion training and inference consume 70GB and 22GB, respectively.

- 4.3.2 Multi-Object Editing. Our method focuses on addressing single-object editing but is adaptable to multi-objects. The reconstruction phase leverages a multi-view generator [Voleti et al. 2025], optimized for single-object contexts, necessitating individual reconstruction for each object. Objects are then manually edited, with edits propagated via offsets, and finally processed sequentially by our decoupled video diffusion model. Fig. 10 shows an example.
- 5 APPLICATIONS

Image-to-Video Animation. Given an input image, we first reconstruct a 3D model of the target object. The reconstructed 3D model can then be rigged and animated to create various motions. Finally, the edited 3D meshes are further employed by our decoupled video diffusion model to generate the animated video (see Fig. 12).

Appearance Editing. As the geometry and texture controls are decoupled in our stage 3, we can easily integrate various image editing tools into our framework to support flexible object appearance editing. Visual results are presented in Fig. 13.

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

Input Image Driven Skeleton Animated Mesh Animated Video

- Figure 12: Our approach enables Image-to-Video animation: from a single input image (col. 1), we reconstruct a 3D mesh, rig it with a skeleton (col. 2), and animate it using motion sequences to produce edited meshes (col. 3). Final videos are rendered from the animated geometry and textures.

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

Input Video Appearance Editing Edited Video

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

Normal & Text

[Figure 317]

“Pink bird”

Edited 1st Frame

[Figure 318]

“White duck”

[Figure 319]

[Figure 320]

[Figure 321]

“white rhino”

[Figure 322]

Mesh Editing

Flux

Flux

Flux

- Figure 13: Our method integrates with existing 2D editing tools (e.g., Flux [Labs 2024]) for flexible appearance editing. Left: input frames; center: 3D geometry + 2D appearance edits; right: final results.

### 6 CONCLUSION

proxy, the proposed method demonstrates compelling performance in various video editing applications.

In this work, we introduce Shape-for-Motion, a 3D-aware video editing method that enables users to perform precise and consistent edits on target objects in a video. Our method leverages generated novel views as constraints to reconstruct the target object into a time-consistent mesh, allowing users to perform edits directly in 3D space with a dual-propagation strategy. With the help of 3D

Limitation. Our approach does have limitations. First, as illustrated in Fig. 11, it encounters difficulties during the reconstruction stage when dealing with objects exhibiting complex motions. Second, it struggles to handle object associations. In the future, we aim to introduce the Visual Association Model [Liu et al. 2024a] and the Large Reconstruction Model to solve these problems.

This supplemental material provides additional details on our methods and experiments. Sec. A covers 3D proxy reconstruction, including data preparation and implementation. Sec. B focuses on editing 3D proxies with automatic propagation. Sec. C discusses generative rendering from 3D proxies, including inference workflow and experiments. Sec. D includes a discussion on pipeline design, efficiency, mesh topology, reconstruction performance, and 3D proxy quality. Finally, Sec. E presents additional visual results.

A Consistent 3D Proxy Reconstruction

- A.1 Preliminaries

3D Gaussian Splatting (GS) [Kerbl et al. 2023] adopts a novel approach based on explicit point clouds to efficiently model 3D scenes. Each 3D Gaussian is parameterized by its mean position 𝜇, a covariance matrix Σ and other attributes. Formally, a 3D Gaussian 𝐺 is defined as:

𝐺(𝑥) = 𝑒−12 (𝑥)𝑇 Σ−1(𝑥) (4) Each Gaussian is multiplied by opacity 𝛼 during the blending process. When projecting 3D Gaussian to 2D space, the covariance matrix is updated to Σ′ using a Jacobian matrix 𝐽 and a viewing transformation𝑊 via Σ′ = 𝐽𝑊 Σ𝑊𝑇 𝐽𝑇. To handle the differentiable optimization, the covariance matrix Σ is divided into two learnable elements𝑟 and𝑠 to represent the rotation and scaling, which is then transformed into the corresponding matrices 𝑆 and 𝑅 to form the Σ via Σ = 𝑅𝑆𝑆𝑇𝑅𝑇. Each 3D Gaussian is represented as 𝐺(𝑥;𝜇,𝑟,𝑠,𝛼).

Deformable 3DGS [Yang et al. 2024a] extend the 3D GS to dynamic scene by learning a set of canonical Gaussians along with a time-varying deformation field 𝜃 parametrized as an MLP. At each time 𝑡, the position 𝛾(𝑥) of 3D Gaussians and time 𝛾(𝑡) with positional encoding are used as input to the deformation MPL to obtain the offset 𝜃(𝑥), 𝜃(𝑟) and 𝜃(𝑠) of the dynamic 3D Gaussians in canonical space. 𝛾(·) denotes the positional embedding function. The new 3D Gaussian at the deformed space can then be represented as 𝐺′(𝑥 + 𝜃(𝑥);𝜇,𝑟 + 𝜃(𝑟),𝑠 + 𝜃(𝑠),𝛼).

Directly performing editing on the Gaussian Splatting has been explored [Shin et al. 2024; Sun et al. 2024] recently. However, manipulating irregular 3D points is not user-friendly and only supports limited editing types.

- A.2 Data Preparation

The workflow of data preparation during the target object reconstruction is depicted in Fig. A14. Given an input video, we first crop and segment the target object using SAM2 [Ravi et al. 2024]. Each frame of the cropped video is processed by a multi-view generator [Voleti et al. 2025] to produce novel views. During the novel views generation, we assume that the camera in the input video is fixed and set the same camera pose for all input frames, where the field of view (FOV) = 33.8, elevation angle = 0, and azimuth angle = 0. For the newly generated views, we keep the field-of-view and elevation angles the same as in the input frame and sample different azimuths from the 360-degree sphere. The six azimuths are: {51.43, 102.86, 154.29, 205.71, 257.14, 308.57}. Then, we send both the original input video frames and the generated novel views to a depth estimation method [Yang et al. 2024c] to produce depth maps, which will be used as the Ground-Truth depth supervision. We also employ SAM2 to obtain the Ground-Truth mask from the input frames and novel

views. By default, we process 21 frames. If the motion between consecutive frames in the input video is minimal, the number of frames is increased to 42.

### A.3 Implementation Details

Formally, given the input video frames Xsrc and generated novel views V, during the reconstruction process, we render a Gaussiansplatted image 𝐼𝑔𝑠 ∈ R𝐻×𝑊 ×3 from the deformed gaussian, and render three mesh outputs from the Gaussian-propagated deformed mesh, i.e., mesh mask 𝑀𝑝𝑟𝑒𝑑 ∈ R𝐻×𝑊 ×1, mesh depth 𝐷𝑝𝑟𝑒𝑑 ∈ R𝐻×𝑊 ×1 and mesh image 𝐼𝑚𝑒𝑠ℎ ∈ R𝐻×𝑊 ×3. The loss functions are illustrated as follows.

GS Loss. We follow [Kerbl et al. 2023] to adopt a combination of L1 loss and SSIM loss to supervise the gaussian-splatted image:

L𝑔𝑠 = (1 − 𝜆𝑠𝑠𝑖𝑚) · ||𝐼𝑔𝑠 − 𝐼𝑔𝑡 || + 𝜆𝑠𝑠𝑖𝑚 · L𝑠𝑠𝑖𝑚(𝐼𝑔𝑠,𝐼𝑔𝑡), (5)

where 𝜆𝑠𝑠𝑖𝑚 = 0.2. We use the input frames and generated views as the 𝐼𝑔𝑡 for the observed (input) and generated (novel) views, respectively.

Mesh Mask Loss. We apply an L1 loss to the rendered mesh mask to help constrain the shape of the mesh:

L𝑚𝑎𝑠𝑘 = ||𝑀𝑝𝑟𝑒𝑑 − 𝑀𝑔𝑡 || (6)

Mesh Image Loss. We employ the same loss as L𝑔𝑠 to supervise the rendered mesh image 𝐼𝑚𝑒𝑠ℎ:

L𝑟𝑔𝑏 = (1 − 𝜆𝑠𝑠𝑖𝑚) · ||𝐼𝑚𝑒𝑠ℎ − 𝐼𝑔𝑡 || + 𝜆𝑠𝑠𝑖𝑚 · L𝑠𝑠𝑖𝑚(𝐼𝑚𝑒𝑠ℎ,𝐼𝑔𝑡). (7) Mesh Depth Loss. Although the balanced view sampling strategy can alleviate inconsistencies and improve the overall quality of the 3D mesh, the surface geometry still suffers from depth ambiguities, resulting in issues such as sunken surfaces or floating artifacts. To address this, we further introduce a scale-invariant depth constraint:

∑︁𝑀

1 𝑀𝑔𝑡

||𝐷𝑖pred − 𝐷𝑖gt||, (8)

Ldepth =

𝑖=1

where 𝑀𝑔𝑡 denotes the Ground-Truth mask. D𝑖gt and D𝑖pred are the disparity maps predicted by Depth-Anything [Yang et al. 2024c]

and rendered from the reconstructed mesh (i.e., 𝐷𝑝𝑟𝑒𝑑), respectively.

During optimization, gradients can propagate back to the Gaussians and the deformation MLP through both the Gaussian splatted image and the mesh outputs, enabling updates to these components jointly. In this way, the correspondences across frames in the deformable GS are transferred to the mesh, resulting in a correspondence-enabled mesh for the target object.

Implementation. We use the implementation from 3D Gaussian Splatting [Kerbl et al. 2023] for differentiable Gaussian rasterization. Instead of using SfM for initialization, we initialize the 3DGS using points uniformly sampled from a sphere of radius 1. The model is trained for 25k iterations, with the first 3k optimizing only the 3D Gaussians for the first frame. Joint training of 3D Gaussians and the deformation field follows, and from iteration 12k, DPSR with Marching Cubes is introduced to optimize the mesh geometry. By default, 𝑁 = 6 novel views are uniformly sampled along a circular trajectory using SV3D [Voleti et al. 2025]. All reconstructions were conducted against a white background at a resolution of 576 × 576 on an NVIDIA A100 GPU. For the optimized networks, such as

View

View

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

…

…

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

…

…

Multi-View Generator

Depth Anything

SAM

…

…

…

…

…

…

…

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

…

…

Input Video

Input Object Novel Views Depth of Input Views

#### Figure A14: Workflow of data pre-processing in consistent 3D proxy reconstruction.

#### Table A4: Ablation study on the use of novel views. [f, m, l] represent the first, middle, and last frames, respectively. Note that all the metrics in this experiment are evaluated on six views.

the deformation field 𝜃 and the color network 𝑀𝐿𝑃𝑐(·), we follow the same implementations as in [Yang et al. 2024a]. Additionally, the backward deformation field 𝜃−1 adopts the same network as 𝜃. Considering the inconsistency caused by the novel views, for all losses, we apply a smaller weight (1/5) to novel views than the observed view.

Time slots uses novel views Number of novel views

Metrics

Querying the color during training. We adopt the approach of [Liu et al. 2024b], storing vertex colors in a canonical color MLP, denoted as 𝑀𝐿𝑃𝑐(·). The vertex color query process for the deformed mesh proceeds as follows:

[f] [f,m] [f,m,l] All 12 6 3

PSNR ↑ 22.53 22.40 22.60 23.45 23.59 24.04 23.30 SSIM ↑ 0.919 0.919 0.921 0.925 0.926 0.928 0.925 LPIPS ↓ 0.054 0.054 0.054 0.054 0.053 0.054 0.054 DINO Sim. ↑ 0.428 0.445 0.410 0.403 0.405 0.419 0.389

- • Deformation Query: Given the canonical Gaussians𝐺𝑐, we query their deformation of 𝐺𝑐 at time t via the deformation field 𝜃(𝐺𝑐).
- • Gaussian Wraping: The deformed Gaussians 𝐺𝑡 are computed as 𝐺𝑡 = 𝐺𝑐 + 𝜃(𝐺𝑐).
- • Mesh Extraction: Using DPSR combined with Marching

Cubes (MC), we convert 𝐺𝑡 into its corresponding deformed mesh.

- • Canonical Mapping: The deformed mesh is projected back to the canonical space via the inverse deformation field 𝜃−1(·).
- • Color Assignment: Vertex colors are queried from 𝑀𝐿𝑃𝑐(·) in the canonical space and transferred to the deformed mesh, leveraging their one-to-one vertex correspondence.

### B Editing 3D proxy with automatic propagation B.1 Implementation Details

Our approach supports diverse video editing tasks, including but not limited to pose editing, rotation, scaling, translation, texture modification, and object composition. While the key idea and implementation rely on our dual-propagation strategy, specific task implementation may involve task-dependent settings and variations.

General Editing. For four types of editing, pose editing, rotation, translation, and scaling, users only need to apply their desired modifications to the canonical mesh once, after which the edits are automatically propagated across all frames.

Object Composition. To composite a static object into a dynamic object, we first generate the 3D mesh of the new object. The generated mesh is then imported into Blender, where it is manually positioned to align with the intended location, forming a new edited mesh. Next, an anchor vertex is selected from the canonical mesh of the original dynamic object by finding the closest vertex to the new object. The motion of this anchor vertex is then transferred to all vertex of the new object, ensuring consistent movement within the scene. Additionally, users can achieve more diverse motion effects by applying complex binding strategies to the new object.

### A.4 Additional Experiments

We also conduct experiments to evaluate the use of novel views: (1) determining which time slots (i.e., frames) utilize novel views, with the number of novel views 𝑁 fixed; and (2) evaluating the impact of the number of novel views 𝑁, where novel views are applied to all frames. The results in Tab. A4 demonstrate that (1) providing novel views for only key frames is insufficient and (2) due to the inconsistencies introduced by novel views, using more novel views does not necessarily yield better results. Thus, by default, we utilize six generated novel views for each frame to balance the reconstruction completeness and novel-views inconsistency.

Texture modification. Texture editing in this context refers to directly modifying the vertex color of the mesh (e.g., editing the flower in Fig. E17). Since the canonical mesh does not store color information, we replace it with the mesh from the first frame (i.e., 𝑡 = 0). Users can modify the texture color using Blender’s ‘vertex

paint’ function or employ advanced UV mapping techniques for more precise adjustments.

C Generative rendering from 3D proxy

- C.1 Data Preparation

As ourdecoupledvideodiffusionmodel operates in a self-supervised manner, we provide more details on the data construction process. Reference Object. For a target object in a video, we first extract the object from the background using the ground-truth mask provided in the VOS dataset. Next, we apply random scaling, shifting, and rotation augmentations to all frames simultaneously, ensuring that the augmentation parameters remain consistent across all frames (including both the image and the corresponding mask). After augmentation, the background of the target object is filled with a uniform gray color. In addition, during the training process, we also randomly re-order the input video frames to simulate the reference object. In this way, we simulate the generation of preediting video data.

Texture Control. Since our dual-propagation strategy relies on nearest-neighbor mapping for texture propagation, the texture colors exhibit a locally similar pattern. To simulate this characteristic, we first apply Simple Linear Iterative Clustering (SLIC) to segment each frame into multiple small patches, with the number of patches randomly selected from the range [800, 1200]. Next, we use median blur filtering to smooth details within the patches and reduce boundary artifacts, with the kernel size randomly chosen from the range [3, 11]. Finally, consecutive downsampling and upsampling operations are applied, with the scaling factor randomly selected from the range [0.25, 0.5]. In this way, we simulate the coarse texture after editing.

Geometry control.We utilize Depth-Anything [Yang et al. 2024c], to generate depth maps of the Ground-Truth object. These depth maps are then converted into normal maps, serving as the geometry control input for the geometry controller.

- C.2 Inference Workflow The inference pipeline is shown in Fig. C15. Formally, we denote the input video as V𝑖𝑛𝑝. The object mask for editing, obtained from SAM, is represented as M𝑖𝑛𝑝. The edited color is denoted by V𝑟𝑔𝑏𝑒𝑑𝑖𝑡𝑒𝑑,

and the edited normal map is represented as V𝑛𝑜𝑟𝑚𝑎𝑙𝑒𝑑𝑖𝑡𝑒𝑑 . The target object mask is extracted by SAM and denoted by M𝑡𝑔𝑡. We then obtain the inpainting mask M𝑖𝑛𝑝𝑎𝑖𝑛𝑡 by subtracting the overlapping regions of M𝑖𝑛𝑝 and M𝑡𝑔𝑡 from M𝑖𝑛𝑝:

M𝑖𝑛𝑝𝑎𝑖𝑛𝑡 = M𝑖𝑛𝑝 − (M𝑖𝑛𝑝 ∩ M𝑡𝑔𝑡) (9) To reduce the distraction from the backgrounds of the input video V𝑖𝑛𝑝, we mask its background and only retain its foreground object, to obtain the reference object video V𝑟𝑒𝑓 :

V𝑟𝑒𝑓 ,V𝑏𝑔 = Split(V𝑖𝑛𝑝,M𝑖𝑛𝑝) (10)

where Split(·) denotes the binary separating function and V𝑏𝑔 represents the background information, in which the foreground regions

in V𝑏𝑔 are filled with white color, and the background regions in V𝑟𝑒𝑓 are filled with gray color, respectively. Then, the texture control map can be obtained by:

V𝑐𝑜𝑛𝑡𝑟𝑜𝑙𝑡𝑒𝑥 = V𝑟𝑔𝑏𝑒𝑑𝑖𝑡𝑒𝑑 × M𝑡𝑔𝑡 + V𝑏𝑔 × (1 − M𝑡𝑔𝑡) (11)

We then obtain the geometry control map V𝑔𝑒𝑜𝑐𝑜𝑛𝑡𝑟𝑜𝑙 by replacing the background region in V𝑛𝑜𝑟𝑚𝑎𝑙𝑒𝑑𝑖𝑡𝑒𝑑 with a gray color.

With all inputs ready, we first convert the reference object V𝑟𝑒𝑓

and the texture control V𝑐𝑜𝑛𝑡𝑟𝑜𝑙𝑡𝑒𝑥 from pixel-space to latent space using VAE’s encoder. We then apply nearest-neighbor down-sampling

to the target mask M𝑡𝑔𝑡 and the inpainting mask M𝑖𝑛𝑝𝑎𝑖𝑛𝑡 to resize them. For geometry control, we first extract its features using several consecutive convolutional blocks with down-sampling [Zhang et al. 2023a]. These extracted features are then concatenated with the noisy latents obtained from the base model and passed into the control branch. The CLIP image embeddings extracted from the reference object V𝑟𝑒𝑓 are also used as Key and Value in the cross-attention of the base model and the control branch. Finally, the denoised latents are sent to the VAE decoder to generate the output editing results.

### C.3 Implementation Details

We use theI2Vrelease ofthestablevideo diffusion (SVD) [Blattmann et al. 2023] as the base model. Our model is trained on the VOS [Xu et al. 2018] dataset 4, filtering out extremely small-sized objects from the original 7800 unique objects to retain 5968 samples. As we do not have real paired data (source/target videos), we generate a training dataset via on-the-fly augmentation (Supp. C.1). We use a two-stage mixed-training method, first training the geometry controller and then fixing it to train the denoising UNet. In both stages, we adopt the same denoising loss as in SVD. The two training stages are optimized for 120K and 60K iterations, respectively, using the Adam [Loshchilov 2017] optimizer on 8 NVIDIA A100 GPUs for 7 days. Each GPU processes a batch size of 1 with a resolution of 384×256. During testing, the default resolution is 768 × 512, but the model supports higher resolutions. During training, classifier-free guidance is applied to the reference object with a probability of 0.1. During the inference, we use the Euler [Karras et al. 2022] sampler with 25 sampling steps and a classifier-free guidance scale of 3.

### C.4 Experimental Details

Metrics. We adopt the widely-used video editing evaluation metrics: Fram-Acc and Tem-Con. Fram-Acc measures the frame-wise editing accuracy, defined as the percentage of frames where the edited image achieves a higher CLIP similarity to the target prompt than to the source prompt. Tem-Con evaluates temporal consistency by calculating the cosine similarity between consecutive frame pairs. However, for fine-grained edit types, CLIP cannot measure the consistency of objects before and after editing. To this end, we introduce a new metric, CLAPScore (CLIP-APpearance Score), to jointly consider textual and semantic consistency. CLAPScore measures the accumulative error of the Fram-Acc and the DINO similarity score:

CLAP Score = Fram-Acc × DINO(𝑖𝑛𝑝𝑢𝑡,𝑜𝑢𝑡𝑝𝑢𝑡), where 𝑖𝑛𝑝𝑢𝑡 and 𝑜𝑢𝑡𝑝𝑢𝑡 represent the input and edited frames.

V3DBench . To evaluate the fine-grained video editing capabilities, we collect a new benchmark dataset. The videos in this dataset are sourced from the Internet [Pexels [n.d.]], the DAVIS dataset [Perazzi et al. 2016], and generated videos [Brooks et al. 2024; Chai

4Note that all examples used during testing are entirely unseen during training.

[Figure 344]

[Figure 345]

𝑽𝒏𝒐𝒓𝒎𝒂𝒍𝒆𝒅𝒊𝒕𝒆𝒅 𝑽𝒈𝒆𝒐𝒄𝒐𝒏𝒕𝒓𝒐𝒍

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

Conv. Blocks

Geometry Controller

Edited Normal Geometry Control

[Figure 350]

[Figure 351]

𝑽𝒓𝒆𝒇

[Figure 352]

CLIP

Zero Convs

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

𝑽𝒊𝒏𝒑 𝑴𝒊𝒏𝒑

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

Reference Object

[Figure 362]

[Figure 363]

𝑽𝒃𝒈

[Figure 364]

VAE

Input Video Object Mask

Background Video

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

𝑽𝒐𝒖𝒕

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

𝑽𝒓𝒈𝒃𝒆𝒅𝒊𝒕𝒆𝒅 𝑴𝒕𝒈𝒕 𝑽𝒕𝒆𝒙𝒄𝒐𝒏𝒕𝒓𝒐𝒍

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

| | |
|---|---|
| | |

VAE

VAE

Edited RGB Target Mask Texture Control Output Video

[Figure 383]

[Figure 384]

[Figure 385]

Denoising U-Net

Downsample

[Figure 386]

𝑴𝒊𝒏𝒑𝒂𝒊𝒏𝒕 𝑴𝒊𝒏𝒑𝒂𝒊𝒏𝒕

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

Downsample

Concatenation

Gaussian Noise

Inpainting Mask

#### Figure C15: Inference pipeline of our decoupled video diffusion.

et al. 2023]. This dataset includes diverse video content across categories such as animals, humans, vehicles, and etc., covering six types of video editing tasks: pose editing, rotation, scaling, translation, texture modification, and object composition. As many video editing methods rely on text prompts, we leverage GPT-4o to generate source, target, and instruction-based prompts by providing the input keyframe along with editing details. Specifically, we first manually edit a keyframe, typically the first frame, using ImageSculpturing [Yenphraphai et al. 2024], which serves as the edited reference. The original and edited frames are then input to GPT-4o, accompanied by specific editing instructions to generate appropriate prompts. The prompt to GPT-4o is: “Describe the difference between the two images before and after the object is edited. Note that you should describe the object and its state. You do not need to mention phrases like “object xx does not exist” or “no operation was done” in the source prompt; simply describe the state of the edited object in the target prompt. Avoid using vague or redundant phrases, such as “in a simple and unmodified state.” The editing for this case is [xxx]”. We will release this benchmark dataset.

participants are asked to evaluate the editing realism by considering the overall visual realism, lighting consistency, and geometric coherence. The results in Table 1 of the main paper indicate that our method consistently outperforms others and is the most preferred.

### D Discussion D.1 Pipeline Design

While one might initially perceive our framework as an intricate integration of multiple components, it is designed explicitly to address several crucial challenges in 3D-aware video editing through a modular structure. Specifically, our method introduces: (1) a novel-view augmentation strategy combined with balanced-view sampling to significantly improve 3D proxy reconstruction consistency; (2) a dual-propagation strategy utilizing canonical Gaussian and mesh representations to efficiently propagate geometry and texture edits without the need for frame-by-frame manual adjustments; and (3) a decoupled video diffusion model trained with a self-supervised mixed-training strategy to ensure appearance consistency, conditioned directly on the geometry and texture from the manipulated 3D proxies. This clearly structured pipeline—consisting of distinct stages for reconstruction, interactive manipulation, and generative rendering—facilitates ease of use, maintainability, and future extensibility. Therefore, the perceived complexity is actually strategic, enhancing the robustness and flexibility of our approach (e.g., swapping in improved reconstruction or diffusion modules), and the integration of these innovative elements is what enables our novel and effective solution.

User Study. In addition to measuring high-level input-output similarity, we conduct a user study to evaluate the perceptual quality of our approach in two aspects: editing quality (EQ), semantic consistency (SC), and visual plausibility (VP). We collect results from four methods compared and from our own on the V3DBench benchmark. The videos were randomized and presented to 45 participants. For editing quality, participants ranked the results based on their alignment with the edited 3D mesh, with higher alignment receiving a higher rank. For semantic consistency, rankings were based on the degree of variation in the object before and after editing, where greater variation resulted in a lower rank. For visual plausibility,

Rotation

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

InputVideoEditedVideo

[Figure 401]

Reconstruction

∆𝑥 = −20

… …

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

… …

Rendering

[Figure 411]

[Figure 412]

[Figure 413]

Frames 0, 20 (0-0.7s) Frames 122, 136 (4.1-4.5s) Frames 256, 278 (8.5-9.3s)

Figure D16: A 9.3-second edited video example demonstrating our method’s ability to handle long video sequences.

Table D5: Inference time comparison between our method and state-of-the-art (SOTA) methods. To ensure a fair comparison with I2V-Edit, we first apply Image-Sculpting to edit the initial frame and then use the resulting edited frame as input for the subsequent process.

Reconciling Topological Differences. Meshes extracted at different temporal instances may differ in topology. Given that the number of Gaussian points remains fixed throughout propagation, we utilize Gaussian points as intermediaries to reconcile these topological discrepancies. Specifically, we impose mask, RGB, and depth constraints during reconstruction to align mesh vertices closely with Gaussian points, thereby ensuring consistent geometry representation.

Methods Tune-a-video Pix2Video I2V-Edit Drag-Video Ours Time (mins) 10.7 2.8 44.3 12.2 91.4

### D.2 Efficiency Comparisons

### D.4 Baseline Reconstruction Performance

We conducted the inference speed comparisons at a fixed resolution (512×512) for a 14-frame video using a single A100 GPU. The result is shown in Tab. D5. Note that we exclude manual editing time, which naturally varies based on the complexity and user interaction. Although our framework requires relatively more computational time, with approximately 91 minutes for reconstruction and 43 seconds for the video diffusion model stage, this trade-off is justified by significantly improved geometric fidelity and temporally coherent appearances, as demonstrated in our experiments (Tab. 1 & Fig. 6 of the main paper). Moreover, the design of our pipeline emphasizes practical usability: once the initial 3D proxy reconstruction is completed, multiple subsequent edits can be efficiently performed without re-optimization (e.g., rotate the object or adjust its pose), unlike most alternatives, which require a full recomputation for each new edit. Thus, despite its higher initial reconstruction time, our approach provides substantial long-term efficiency benefits and enhanced editing quality, aligning closely with practical usage scenarios where precision and consistency are paramount. We also believe that future developments in faster 4D reconstruction methods for video will further reduce the computational overhead, further improving the practicality of our approach.

In Fig.7 (row 1) of the main paper, the baseline configuration (without novel views) utilizes only the input frames directly for dynamic 3D reconstruction, optimizing solely the observed view across time steps. Consequently, this configuration yields inferior reconstruction quality for unseen viewpoints due to its inherent lack of multi-view constraints. While state-of-the-art image-to3D methods achieve impressive single-image reconstructions, their performance primarily focuses on static objects. These methods typically lack explicit inter-frame correspondence modeling, making them unsuitable for maintaining the temporal consistency necessary for coherent dynamic video editing tasks.

### D.5 Influence of the 3D Proxy Quality

The reconstructed geometry and texture from the rendered 3D proxy serve as critical control signals guiding the video generation process. The quality of this proxy can impact the effectiveness and coherence of subsequent editing tasks. For instance, as illustrated in Fig.9 and quantified in Tab.3 of the main paper, removing texture controls (i.e., relying only on stage-1 geometry control) results in notably diminished visual quality. This degradation is primarily due to limited data augmentation during model training, as well as spatial misalignment between the target object before and after editing. Thus, inaccuracies or lower fidelity in the proxy’s geometry and texture inherently propagate through the pipeline, leading to compromised appearance and coherence in the final generated video. Incorporating more advanced or robust video reconstruction methods capable of providing higher-quality proxies could potentially enhance the overall coherence and controllability of the generated video, opening avenues for further improvements in dynamic editing tasks.

### D.3 Topology of Extracted Mesh

Topology and Quality of Extracted Meshes. A potential concern in the propagation of geometry is that extracted meshes might exhibit topological inconsistencies, such as disconnected components or internal holes. To address this, we enhance mesh quality by reducing geometric inconsistencies through balanced-view-sampling (BVS) and a scale-invariant depth constraint. These strategies help mitigate artifacts such as sunken surfaces (holes) and floating outliers. The effectiveness of these measures is validated through ablation studies presented in Fig.7 of the main paper.

### D.6 User APIs

Our framework uses Blender’s intuitive and widely-used API along with automatic rigging tools (e.g., Mixamo5 and Anything World6) to streamline the mesh manipulation. For instance, once the canonical mesh is reconstructed, we can first apply auto-rigging to generate a skeletal structure. Subsequently, the geometry of the object can be easily controlled by manipulating skeletal keypoints.

### D.7 Long Video Processing

In the video diffusion stage, to handle sequences exceeding the default 14-frame limitation of the SVD model, we divide the video into several overlapping windows. The overlapping regions from previous windows are utilized to initialize the noise of subsequent windows via an SDEdit [Meng et al. 2021]-based initialization strategy. Then, overlapping segments across windows are seamlessly merged using a progressive alpha-blending mechanism, where the weight of the previous window gradually decreases from 1 to 0, while the weight of the next window correspondingly increases from 0 to 1 across overlapping frames, ensuring smooth temporal transitions. In Fig. D16, we present an example of editing results on a 9.3-second video. The result exhibits consistent geometry and appearance throughout, highlighting temporal coherence over extended durations.

### E Additional Visual Results

We present additional visual results of our approach in Fig. E17, showcasing various editing types, including pose editing, rotation, translation, texture modification, object composition, and mixed edits (i.e., combinations of different editing types).

- 5https://www.mixamo.com/
- 6https://anything.world/

### References

Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk. 2012. SLIC superpixels compared to state-of-the-art superpixel methods. IEEE transactions on pattern analysis and machine intelligence 34, 11 (2012), 2274–2282.

Sherwin Bahmani, Xian Liu, Wang Yifan, Ivan Skorokhodov, Victor Rong, Ziwei Liu, Xihui Liu, Jeong Joon Park, Sergey Tulyakov, Gordon Wetzstein, et al. 2024. Tc4d: Trajectory-conditioned text-to-4d generation. In European Conference on Computer Vision. Springer, 53–72.

Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. 2022. Text2live: Text-driven layered image and video editing. In European conference on computer vision. Springer, 707–723.

Weikang Bian, Zhaoyang Huang, Xiaoyu Shi, Yijin Li, Fu-Yun Wang, and Hongsheng Li.

2025. Gs-dit: Advancing video generation with pseudo 4d gaussian fields through efficient dense 3d point tracking. arXiv preprint arXiv:2501.02690 (2025).

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. 2024. Video generation models as world simulators. (2024). https: //openai.com/research/video-generation-models-as-world-simulators

Shengqu Cai, Duygu Ceylan, Matheus Gadelha, Chun-Hao Paul Huang, Tuanfeng Yang Wang, and Gordon Wetzstein. 2024. Generative rendering: Controllable 4d-guided video generation with 2d diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7611–7620.

Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. 2023. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23206–23217.

Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. 2023. Stablevideo: Text-driven consistency-aware diffusion video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23040–23050.

Weifeng Chen, Yatai Ji, Jie Wu, Hefeng Wu, Pan Xie, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. 2023. Control-a-video: Controllable text-to-video generation with diffusion models. arXiv preprint arXiv:2305.13840 (2023).

Yen-Chi Cheng, Krishna Kumar Singh, Jae Shin Yoon, Alexander Schwing, Liangyan Gui, Matheus Gadelha, Paul Guerrero, and Nanxuan Zhao. 2025. 3D-Fixup: Advancing Photo Editing with 3D Priors. In Proceedings of the SIGGRAPH Conference Papers. ACM. https://doi.org/10.1145/3721238.3730695

Yufan Deng, Ruida Wang, Yuhao Zhang, Yu-Wing Tai, and Chi-Keung Tang. 2024. Dragvideo: Interactive drag-style video editing. In European Conference on Computer Vision. Springer, 183–199.

Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. 2023. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Xiang Fan, Anand Bhattad, and Ranjay Krishna. 2024. Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion. arXiv preprint arXiv:2403.14617 (2024).

Yuchao Gu, Yipin Zhou, Bichen Wu, Licheng Yu, Jia-Wei Liu, Rui Zhao, Jay Zhangjie Wu, David Junhao Zhang, Mike Zheng Shou, and Kevin Tang. 2024. Videoswap: Customized video subject swapping with interactive semantic point correspondence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7621–7630.

Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou, Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei Liu, et al. 2025. Diffusion as Shader: 3D-aware Video Diffusion for Versatile Video Generation Control. arXiv preprint arXiv:2501.03847 (2025). Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Pengfei Wan, Di Zhang, Yufan Liu, Weiming Hu, Zhengjun Zha, et al. 2024b. I2v-adapter: A general imageto-video adapter for diffusion models. In ACM SIGGRAPH 2024 Conference Papers.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2024a. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. International Conference on Learning Representations (2024).

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. 2022. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022).

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851.

Zhihao Hu and Dong Xu. 2023. Videocontrolnet: A motion-guided video-to-video translation framework by using diffusion model with controlnet. arXiv preprint arXiv:2307.14073 (2023).

Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li, Rynson WH Lau, Wangmeng Zuo, et al. 2025. Voyager: LongRange and World-Consistent Video Diffusion for Explorable 3D Scene Generation. arXiv preprint arXiv:2506.04225 (2025).

Input Video Edited Video

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

Composition TranslationPoseEditing

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

Rotation(pose+translation) Texture

|[Figure 446]|
|---|

|[Figure 447]|
|---|

|[Figure 448]|
|---|

|[Figure 449]|
|---|

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

Modification

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

Object

MixedEditing

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

#### Figure E17: Additional visual results of our method. For each case, three or four input frames are shown on the left, and the edited results are displayed on the right. For each case, we highlight the editing using the red arrow in the first frame, except the texture modification.

Yanqin Jiang, Chaohui Yu, Chenjie Cao, Fan Wang, Weiming Hu, and Jin Gao. 2024. Animate3d: Animating any 3d model with multi-view video diffusion. arXiv preprint arXiv:2407.11398 (2024).

Mizuki Kagaya, William Brendel, Qingqing Deng, Todd Kesterson, Sinisa Todorovic, Patrick J Neill, and Eugene Zhang. 2010. Video painting with space-time-varying style parameters. IEEE transactions on visualization and computer graphics 17, 1 (2010), 74–87.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. 2022. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems 35 (2022), 26565–26577.

Yoni Kasten, Dolev Ofri, Oliver Wang, and Tali Dekel. 2021. Layered neural atlases for consistent video editing. ACM Transactions on Graphics (TOG) 40, 6 (2021), 1–12. Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023.

- 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42,
- 4 (2023), 139–1.

Juil Koo, Paul Guerrero, Chun-Hao P Huang, Duygu Ceylan, and Minhyuk Sung. 2025. Videohandles: Editing 3d object compositions in videos using video generative priors. In Proceedings of the Computer Vision and Pattern Recognition Conference. 17692–17701.

Max Ku, Cong Wei, Weiming Ren, Huan Yang, and Wenhu Chen. 2024. AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks. Transactions on Machine Learning Research (2024).

Black Forest Labs. 2024. FLUX. https://github.com/black-forest-labs/flux. Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, and Timo

Aila. 2020. Modular primitives for high-performance differentiable rendering. ACM Transactions on Graphics (ToG) 39, 6 (2020), 1–14.

Dong Liang, Jinyuan Jia, Yuhao Liu, Zhanghan Ke, Hongbo Fu, and Rynson WH Lau. 2025. VODiff: Controlling Object Visibility Order in Text-to-Image Generation. In Proceedings of the Computer Vision and Pattern Recognition Conference. 18379–18389.

Isabella Liu, Hao Su, and Xiaolong Wang. 2024b. Dynamic Gaussians Mesh: Consistent Mesh Reconstruction from Monocular Videos. arXiv preprint arXiv:2404.12379

(2024).

Shaoteng Liu, Tianyu Wang, Jui-Hsien Wang, Qing Liu, Zhifei Zhang, Joon-Young Lee, Yijun Li, Bei Yu, Zhe Lin, Soo Ye Kim, et al. 2025. Generative video propagation. In Proceedings of the Computer Vision and Pattern Recognition Conference. 17712–17722.

Yuhao Liu, Zhanghan Ke, Fang Liu, Nanxuan Zhao, and Rynson WH Lau. 2024a. Diffplugin: Revitalizing details for diffusion-based low-level tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4197–4208.

I Loshchilov. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).

Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. 2023. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. arXiv preprint arXiv:2308.09713 (2023).

Jiaxi Lv, Yi Huang, Mingfu Yan, Jiancheng Huang, Jianzhuang Liu, Yifan Liu, Yafei Wen, Xiaoxin Chen, and Shifeng Chen. 2024. GPT4Motion: Scripting Physical Motions in Text-to-Video Generation via Blender-Oriented GPT Planning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1430–1440.

Yue Ma, Xiaodong Cun, Yingqing He, Chenyang Qi, Xintao Wang, Ying Shan, Xiu Li, and Qifeng Chen. 2023. Magicstick: Controllable video editing via control handle transformations. arXiv preprint arXiv:2312.03047 (2023).

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2021. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073 (2021).

Oscar Michel, Anand Bhattad, Eli VanderBilt, Ranjay Krishna, Aniruddha Kembhavi, and Tanmay Gupta. 2023. Object 3dit: Language-guided 3d-aware image editing. Advances in Neural Information Processing Systems 36 (2023), 3497–3516.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang. 2024. ReVideo: Remake a Video with Motion and Content Control. arXiv preprint arXiv:2405.13865 (2024).

Wenqi Ouyang, Yi Dong, Lei Yang, Jianlou Si, and Xingang Pan. 2024. I2VEdit: FirstFrame-Guided Video Editing via Image-to-Video Diffusion Models. In SIGGRAPH Asia 2024 Conference Papers.

Xingang Pan, Ayush Tewari, Thomas Leimkühler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. 2023. Drag your gan: Interactive point-based manipulation on the generative image manifold. In ACM SIGGRAPH 2023 Conference Proceedings.

Karran Pandey, Paul Guerrero, Matheus Gadelha, Yannick Hold-Geoffroy, Karan Singh, and Niloy J Mitra. 2024. Diffusion handles enabling 3d edits for diffusion models by lifting activations to 3d. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7695–7704.

Songyou Peng, Chiyu Jiang, Yiyi Liao, Michael Niemeyer, Marc Pollefeys, and Andreas Geiger. 2021. Shape as points: A differentiable poisson solver. Advances in Neural Information Processing Systems 34 (2021), 13032–13044.

Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. 2016. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference

on computer vision and pattern recognition. 724–732. Pexels. [n.d.]. PEXELS. https://www.pexels.com. Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. 2021.

D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10318–10327.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In ICML.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. 2024. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714 (2024).

Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling, et al. 2024. L4gm: Large 4d gaussian reconstruction model. Advances in Neural Information Processing Systems 37 (2024), 56828–56858.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-resolution image synthesis with latent diffusion models. In CVPR.

Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. 2023a. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110 (2023).

Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. 2024. Motioni2v: Consistent and controllable image-to-video generation with explicit motion modeling. In ACM SIGGRAPH 2024 Conference Papers. 1–11.

Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. 2023b. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512

(2023).

Inkyu Shin, Qihang Yu, Xiaohui Shen, In So Kweon, Kuk-Jin Yoon, and Liang-Chieh Chen. 2024. Enhancing Temporal Consistency in Video Editing by Reconstructing Videos with 3D Gaussian Splatting. arXiv preprint arXiv:2406.02541 (2024).

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2021. Denoising Diffusion Implicit Models. In ICLR. Yang-Tian Sun, Yi-Hua Huang, Lin Ma, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi.

2024. Splatter a Video: Video Gaussian Representation for Versatile Processing. arXiv preprint arXiv:2406.13870 (2024).

Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. 2025. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision. Springer, 1–18.

Yao Teng, Enze Xie, Yue Wu, Haoyu Han, Zhenguo Li, and Xihui Liu. 2023. Drag-avideo: Non-rigid video editing with point-based interaction. arXiv (2023).

Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. 2025. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In European Conference on Computer Vision. Springer, 439–457. Tengfei Wang, Ting Zhang, Bo Zhang, Hao Ouyang, Dong Chen, Qifeng Chen, and Fang Wen. 2022. Pretraining is All You Need for Image-to-Image Translation. In arXiv.

Zhenwei Wang, Tengfei Wang, Zexin He, Gerhard Hancke, Ziwei Liu, and Rynson WH Lau. 2025. Phidias: A generative model for creating 3d content from text, image, and 3d conditions with reference-augmented diffusion. ICLR (2025).

Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 2024b. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20310–20320.

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. 2023. Tune-a-video: Oneshot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7623–7633.

Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. 2024a. Cat4d: Create anything in 4d with multi-view video diffusion models. arXiv preprint arXiv:2411.18613 (2024).

Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. 2024. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470 (2024).

Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas Huang. 2018. Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327 (2018).

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. 2024c. Depth Anything V2. arXiv preprint (2024).

Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. 2024b. Direct-a-video: Customized video generation with user-directed camera movement and object motion. In ACM SIGGRAPH 2024 Conference Papers. 1–12.

Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. 2024a. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Jiraphon Yenphraphai, Xichen Pan, Sainan Liu, Daniele Panozzo, and Saining Xie. 2024. Image sculpting: Precise object editing with 3d geometry control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4241–4251.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023a. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International

Conference on Computer Vision. 3836–3847.

Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. 2023b. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077 (2023).

