## HHAvatar: Gaussian Head Avatar with Dynamic Hairs

Zhanfeng Liao*, Yuelang Xu*, Zhe Li, Qijing Li, Boyao Zhou, Ruifeng Bai, Di Xu, Hongwen Zhang, Yebin Liu

### arXiv:2312.03029v3[cs.CV]20Nov2024

Abstract—Creating high-fidelity 3D head avatars has always been a research hotspot, but it remains a great challenge under lightweight sparse view setups. In this paper, we propose HHAvatar represented by controllable 3D Gaussians for high-fidelity head avatar with dynamic hair modeling. We first use 3D Gaussians to represent the appearance of the head, and then jointly optimize neutral 3D Gaussians and a fully learned MLP-based deformation field to capture complex expressions. The two parts benefit each other, thereby our method can model fine-grained dynamic details while ensuring expression accuracy. Furthermore, we devise a well-designed geometry-guided initialization strategy based on implicit SDF and Deep Marching Tetrahedra for the stability and convergence of the training procedure. To address the problem of dynamic hair modeling, we introduce a hybrid head model into our avatar representation based Gaussian Head Avatar and a training method that considers timing information and an occlusion perception module to model the non-rigid motion of hair. Experiments show that our approach outperforms other state-of-the-art sparse-view methods, achieving ultra high-fidelity rendering quality at 2K resolution even under exaggerated expressions and driving hairs reasonably with the motion of the head. Project page: https://liaozhanfeng.github.io/HHAvatar.

Index Terms—Head Avatar, Gaussian Splatting, Novel View Synthesis

✦

1 INTRODUCTION

# H

IGH-FIDELITY 3D human head avatar modeling is of great significance in many fields, such as VR/AR,

telepresence, digital human and film production. Although some traditional head avatars [1], [2], [3], [4] realize highfidelity animation, they typically require accurate geometries reconstructed and tracked from dense multi-view videos, thus limiting their applications in lightweight settings. On the other hand, recent works [5], [6], [7] have verified that Neural Radiance Fields (NeRF) [8] can skip the geometry reconstruction and the tracking steps but directly learn high-quality NeRF-based head avatars in dense or sparse views, greatly lowering the threshold for head avatar reconstruction. However, it still remains challenging for these NeRF-based approaches to synthesize high-fidelity images at 2K resolutions with pixel-level details, including hairs, wrinkles, and eyes.

Recently, 3D Gaussian Splatting (3DGS) [9], an explicit and efficient point-based representation, has been proposed for both high-fidelity rendering quality and real-time rendering speed. Compared to NeRF, the reconstruction quality of static and dynamic scenes [10], [11], [12] is much better while rendering time cost has been significantly reduced. Some works [13], [14], [15], [16], [17], [18], [19] have verified 3DGS can also create photorealistic head avatars that are controllable in terms of expression and pose. Nevertheless, these methods continue to face challenges in generating

* indicates equal contribution. Zhanfeng Liao, Yuelang Xu, Zhe Li, Qijing Li, Boyao Zhou, and

Yebin Liu are with the Department of Automation, Tsinghua University, Beijing 100084, P.R.China.

Ruifeng Bai and Di Xu are with Huawei Technologies Co Ltd, Huawei Cloud, Shenzhen, Guangdong, P.R.China.

Hongwen Zhang is with the School of Artificial Intelligence, Beijing Normal University, Beijing, P.R.China. Corresponding author: Yebin Liu.

high-fidelity images at 2K resolutions with precise pixellevel details, faithfully representing highly complex and exaggerated facial expressions, and capturing dynamic hairs. To overcome this bottleneck and further improve the avatar quality, we propose Gaussian Head Avatar with Dynamic Hairs (HHAvatar), a novel representation that utilizes 3DGS for ultra high-fidelity head avatar with dynamic hair modeling.

Previous explicit [20] and implicit [21], [22], [23] head avatars usually formulate the facial deformation via linear blend skinning (LBS) using the skinning weights and blendshapes like the FLAME model [24]. However, such a LBSbased formulation fails to represent exaggerated and finegrained expressions by simple linear operations, limiting the representation ability of the head avatars. Inspired by NeRSemble [25], we propose a fully learnable expressionconditioned deformation field for the 3D head Gaussians, avoiding the limited capability of the LBS-based formulation. Specifically, we input the positions of the 3D Gaussians with expression coefficients into a MLP to directly predict the displacements from the neutral expression to the target one. Similarly, we control the motion of non-face areas, such as the neck, using the head pose as the condition. 3D Gaussian-based representation has the powerful ability to reconstruct high-frequency details, enabling our method to learn accurate deformation fields. In turn, the learned accurate deformation field facilitates the dynamic Gaussian head model to fit more dynamic details. As a result, our method is able to reconstruct finer-grained dynamic details of expressive human heads.

Unfortunately, even though using MLPs can fit more dynamic details, it is not easy to use the same method to model the physical motion of hairs. The reason is that the face can be determined by facial expressions and pose,

[Figure 1]

[Figure 2]

|[Figure 3]|
|---|

|[Figure 4]|
|---|

[Figure 5]

[Figure 6]

[Figure 7]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

###### Source Actor

- Fig. 1: HHAvatar achieves ultra high-fidelity image synthesis with controllable expressions at 2K resolution. The above shows different identities animated by the same expression. The bottom shows that variations in hair positions can arise for identical poses, stemming from diverse hair status (i.e., position and speed) at the previous moment.

but hair also needs to consider the speed and previous position due to inertia. Meanwhile, some works [26], [27], [28], [29], [30] mainly focus on hair reconstruction, but can not animate the head and the hair by facial expressions and head pose (i.e., not avatars). To this end, we propose to use more suitable representations for different head components to develop head avatars capable of simulating the physical motion of hair. Specifically, for hair modeling, we first obtain the 3DGS representation of canonical space, and then use MLP which takes hair status (i.e., position and speed) at the previous moment and the motion of the head as additional condition inputs, similar to deformable 3DGS [31], to deform and capture hair dynamics. Meanwhile, by employing mask supervision that introduces facial and hair occlusion, we segment the hair from the head and maintain the completeness of the head. Furthermore, we introduce a temporal module for fusing time series information as the motion of hair is influenced by the speed and position of the previous moment. As a result, our model can produce realistic animation with physical motion of the hairs.

As a discretized representation, the gradients backpropagated to the 3D Gaussians cannot spread through the whole space. Thus the convergence of training heavily relies on a plausible initialization for both the geometry and the deformation field. However, simply initializing the 3D head Gaussians with a morphable template like FLAME [24] fails to model the long hairstyle and the shoulders. Hence, we further propose an efficient and well-designed geometryguided initialization strategy. This strategy not only initializes the parts included in the morphable template, but also provides a better initialization for dynamic hairs. Specifically, instead of starting from stochastic Gaussians or a FLAME model, we initially optimize an implicit signed

distance function (SDF) field along with a color field and a deformation MLP for modeling the basic geometry, color, and the expression-conditioned deformations of the head avatar respectively. The SDF field is converted to a mesh through Deep Marching Tetrahedra (DMTet) [32], with the color and deformation of the vertices predicted by the MLPs. Then we render the mesh and optimize them jointly under the supervision of multi-view RGB images. Finally, we use the mesh with per-vertex features from the SDF field to initialize the 3D Gaussians to lie on the basic head surface while the color and deformation MLPs are carried over to the next stage, ensuring stable training for convergence. The entire initialization process takes only around 10 minutes.

A preliminary version of this work has been published in CVPR 2024 [33], in which we propose a novel head avatar representation for modeling realistic animatable head avatars under lightweight sparse-view setups. In this paper, we extend it to handle avatar modeling of dynamic hairs. The contributions of our method can be summarized as:

- • We propose HHAvatar, a new head avatar representation that employs controllable dynamic 3D Gaussians to model expressive human head avatars, producing ultra high-fidelity synthesized images at 2K resolutions. For modeling high-frequency dynamic details, we employ a fully learned deformation field upon the 3D head Gaussians, which accurately models extremely complex and exaggerated facial expressions and dynamic hair motion.
- • As far as we know, we propose the first head avatar that can model the physical motion of the hairs. By modeling hair separately from the head and introducing the kinematic features of the hair, the hair movement can be realistically simulated based

on the motion of the head. By designing a training method that considers timing information and an occlusion perception module, dynamic hairs can be reconstructed under lightweight sparse-view setups.

• We design an efficient initialization strategy that leverages implicit representations to initialize the geometry and deformation, leading to efficient and robust convergence when training the HHAvatar. This strategy not only initializes the parts included in the morphable template, but also provides a better initialization for dynamic hairs.

Benefiting from these contributions, our method surpasses recent state-of-the-art methods under lightweight sparseview setups on the avatar quality by a large margin and model the physical motion of the hairs as shown in Fig. 1.

#### 2 RELATED WORK

In this section, we will briefly present the research related to head avatars and data-driven hair animation.

###### 2.1 3D Head Avatar Modeling

Due to the wide application value in the film and digital human industry, 3D head avatar reconstruction from multiview images has always been a research hotspot. However, most of the works only focus on reconstruction without hair motion, and even only facial reconstruction. Traditional works [34], [35], [36], [37] reconstruct the scan geometry through multi-view stereo and then register a face mesh template to it. However, such methods usually require heavy computation. With the utilization of deep neural networks, current methods [38], [39], [40], [41] achieve very fast reconstruction, producing even more accurate geometry. Lombardi et al. [1], Bi et al. [42] and Ma et al. [3] represent the full head mesh through a deep neural network and train it with multi-view videos as supervision. However, due to the errors in geometric estimation, mesh-based head avatars typically suffer from texture blur. Therefore, some recent methods [43], [44] utilize NeRF representation [8] to synthesize novel view images without geometry reconstruction, or build NeRF on the head mesh template [2]. Furthermore, the NeRF-based methods are extended to sparse view reconstruction tasks [5], [6], [7], [25] and achieve impressive performance.

Methods which focus on generative model [24], [45], [46], [47], [48], [49], [50] are dedicated to learning general mesh face templates from large-scale multi-view face images or 3D scans. Recently, implicit SDF-based [51] or NeRFbased [52], [53], [54], [55], [56] methods can learn fullhead templates without the limitations of fixed topology, thereby better modeling complex hairstyles and glasses. Cao et al. [55] adopts a hybrid representation of local NeRF built on the mesh surface, which enables high-fidelity rendering and flexible expression control.

3D head avatars reconstruction from monocular videos is also a popular yet challenging research topic. Early methods [57], [58], [59], [60], [61], [62] optimize a morphable mesh to fit the training video. Recent methods [63], [64] leverage neural networks to learn non-rigid deformation upon 3DMM face templates [24], [65], thus can recover more

dynamic details. Such methods are not flexible enough to handle complex topologies. Therefore, the latest methods explore to construct head avatar models based on implicit SDF [22], point clouds [20] or NeRF [21], [23], [66], [67], [68], [69], [70], [71], [72], [73].

###### 2.2 Head Avatar by Gaussian Splatting

Point elements as a discrete and unstructured representation can fit geometry with arbitrary topology [74] efficiently. Recent methods [75], [76], [77] open up a differentiable rasterization pipeline, such that the point-based representation is widely used in multi-view reconstruction tasks. Aliev et al. [78] and Ruckert et al. [79] propose to first render the feature map, which is transferred to the images through a convolutional renderer. Xu et al. [80] use neural point cloud associated with neural features to model a NeRF.

Recently, 3D Gaussian splatting [9] shows its superior performance, beating NeRF in both novel view synthesis quality and rendering speed. Some approaches [10], [11], [12], [81], [82], [83], [83], [84] extend Gaussian representation to dynamic scene reconstruction. However, these methods can not be migrated to the head avatar reconstruction tasks. GaussianAvatars [13] create photorealistic head avatars that are fully controllable in terms of expression and pose from multi-view videos. While other approaches like SplattingAvatar [14], PSAvatar [15], MonoGaussianAvatar [16], Rig3DGS [17], FlashAvatar [18] and GaussianHead [19] employ the coefficients of 3DMM to control the head and reconstruct a 3DGS-based animatable head avatar from monocular videos. However, these approaches need the hair maintain as still as possible and model the hair as a part rigidly attached to the head.

###### 2.3 Data-driven Hair Animation

In academia and the film and gaming industries, using physics based simulations to create hair animations is a common practice [85]. However, using physics based simulations to generate hair animations may be computationally expensive. To address this issue, a simplified data-driven approach [86], [87], [88] simulates only a small portion of guided hair bundles and interpolates the remaining parts using skin weights learned from the complete simulation.

With the latest advances in deep learning, the use of neural networks has improved the efficiency of dynamic generation [89] and rendering [90], [91] of hair. Some methods [89] use deep neural networks for adaptive binding between normal hair and guided hair. Some methods [91] treat hair rendering as an image translation problem and generate realistic rendering of hair based on 2D hair masks and strokes. Some methods [90] achieve faster rendering and realistic results by using screen space neural rendering technology instead of the rendering part in the animation pipeline. However, these methods are still based on traditional hair simulation pipelines and use synthetic wigs, which require artists to manually set and are not easy to measure and evaluate. Some methods [92] propose to use a secondary motion graph for hair animation at runtime, without relying on traditional hair simulation pipelines. However, this method is limited by the artist’s design of

wigs and control over hair simulation parameters, and cannot be used to animate real hair motions.

Additionally, certain techniques require only multi-view videos and camera parameters as input without necessitating the manual design of wigs by artists [28], [30], [93]. Some methods [28], [30] do not need the artist’s design of wigs. NeuWigs [28] uses the mixture of volumetric primitives (MVP) to model hair and does not rely on a manual hair design. However, this method needs a lot of cameras to capture a multi-view video for reconstruction, and only considers the driving of the hair and cannot complete the driving of the entire head. GaussianHair [30] and Gaussian Haircut [93] use specially designed Gaussian representation to model hair, which can achieve high-quality hair reconstruction. However, this method requires first relying on multi-view images to complete static reconstruction, and then driving based on a conventional CG rendering engine, which is not data-driven. Moreover, this method cannot animate the head by facial expressions and head poses. Compared with previous methods, our approach can achieve data-driven head modeling, including the face and the hair, under lightweight sparse-view setups.

#### 3 METHOD

The pipeline of the reconstruction of HHAvatar is illustrated in Fig. 2, including the initialization stage and the training stage of HHAvatar. The detail of the temporal module is illustrated in Fig. 3. Before the beginning of the pipeline, we remove the background [94] of each image and jointly estimate the 3DMM model [65], 3D facial landmarks and the expression coefficients for each frame, and we obtain the mask of the hair and the head by face-parsing [95] 1. The whole model is optimized under the supervision of multiview RGB videos and hair masks.

###### 3.1 Avatar Representation

Generally, the static 3D Gaussians [9] with N points are represented by their positions X, the multi-channel color C, the rotation Q, scale S and opacity A. The rotation Q is represented in the form of quaternion. Subsequently, the Gaussians can be rasterized and rendered to the multichannel image I given the camera parameters µ. This process can be formulated as:

##### I = R(X,C,Q,S,A;µ). (1)

Our task is to reconstruct a dynamic head avatar controlled by facial expression, head pose, and head speed. For the region outside the hair on the head, the head avatar only depends on the facial expression and the head pose. For the hairs, a point Pt in hairs at time step t depend on the point Pt−1 at time step t−1, the speed of the point, the head pose at time step t, and the head speed. Therefore, we formulate the head avatar as dynamic 3D Gaussians conditioned on the facial expression, the head pose, head speed, and the points in hairs at the previous 2 time steps. To handle the dynamic changes, we input the above conditions to the head avatar model and output the position and other attributes of the Gaussians as above.

1. https://github.com/zllrunning/face-parsing.PyTorch

Specifically, we first extract the neutral mesh for the head and the hair through DMTet [32] to initialize the neutral Gaussians (Sec. 3.3) and construct a canonical neutral Gaussian model with expression-independent attributes including X0 ∈ RN×3 denotes the positions of the Gaussians with a neutral expression in the canonical space, F0 ∈ RN×128 denotes the point-wise feature vectors as their intrinsic properties for generating high-resolution images, Q0 ∈ RN×4, S0 ∈ RN×3, and A0 ∈ RN×1 denote the neutral rotation, scale and opacity respectively. X0, F0, Q0, S0, and A0 are fully optimizable. Note that we do not define the neutral color C0, but directly predict expression-dependent dynamic color from the point-wise feature vectors F0. Then, we construct a MLP-based expression conditioned dynamic generator Φ to generate all the extra dynamic changes to the neutral model. Because the deformation of hair at time step t is related to the hair position Xt−1 at time step t − 1, the hair speed which can get from Xt−1 − Xt−2, the head pose βt at time step t, and the head movement speed which can get from βt and βt−1, the dynamic generator Φ needs Xt−1, Xt−2, and βt−1 as extra inputs. Overall, the whole HHAvatar can be formulated as:

##### {Xt,Ct,Qt,St,At} = Φ(X0,F0,Q0,S0,A0; Xt−1,Xt−2,θt,βt,βt−1),

(2)

with {Xt,Ct,Qt,St,At} denoting {X,C,Q,S,A} at the time step t and θt denoting expression coefficients at the time step t. During the training, we optimize all the parameters of the dynamic generator Φ and the neutral Gaussian model {X0,F0,Q0,S0,A0}, which are highlighted in bold in the following.

3.1.1 Head Avatar

Next, we explain the process of adding expression-related changes to the neutral Gaussian model through the dynamic generator Φ as described in Eqn. 2 in detail. We use superscript (’) to denote the parameters of Gaussians which are changed by MLPs in the canonical space but not transformed to the world space by head pose βt.

Positions Xt′ of the Gaussians at time step t. Expressions bring about the geometric deformation of the neutral model, which is modeled as the displacements of the Gaussian points. Specifically, we predict the displacements respectively controlled by the expression and the head pose in the canonical space through two different MLPs: fexpdef ∈ Φ and fposedef ∈ Φ. Then, we add them to the neutral positions.

##### Xt′ = X0 + λexp(X0)fexpdef(X0,θt) +λpose(X0)fposedef (X0,βt).

(3)

λexp(·) and λpose(·) represent the extent to which the points are affected by the expression or the head pose respectively. Without decoupling but just using the expression coefficients as the global condition [68] which also controls the shoulders and upper body, will produce jittering results during the animation. Here, we assume that the Gaussian points closer to 3D landmarks are more affected by the expression coefficients and less affected by the head pose, while the opposite is true for the Gaussian points far away. Specifically, The 3D landmarks P0 of the canonical model

Expression+Pose

|[Figure 10]|
|---|

|[Figure 11]<br><br>[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

……

Resolution

[Figure 20]

Super

[Figure 21]

Loss

Expression+Pose

[Figure 22]

[Figure 23]

[Figure 24]

Feature Map

Head Dynamic MLPs

2K RGB

GT Image

[Figure 25]

|[Figure 26]|
|---|

[Figure 27]

| | | |
|---|---|---|
|lor orm| |MLP MLP|
| | | |

[Figure 28]

Resolution

Super

C De

Loss

Head Neutral Gaussians Head Expressive Gaussians

Concat

Feature Map

Head Expressive Mesh Head Neutral Mesh

[Figure 29]

[Figure 30]

[Figure 31]

Temporal Module

[Figure 32]

[Figure 33]

ExtractExtract

[Figure 34]

[Figure 35]

Loss

Full Expressive Gaussians

###### Hair Dynamic MLPs

Segmentation Head Mask

GT Mask

Color MLP Deform MLP

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Loss

Temporal Inputs

Pose Pose

Hair Expressive Mesh Hair Neutral Mesh Hair Neutral Gaussians Hair Expressive Gaussians

Segmentation Hair Mask

Geometry-guided Initialization Gaussian Head Avatar with Dynamic Hair

- Fig. 2: The pipeline of the HHAvatar rendering and reconstruction. We first optimize the guidance model including a neutral mesh, a deformation MLP and a color MLP in the Initialization stage. Then we use them to initialize the neutral Gaussians and the dynamic generator. Finally, 2K RGB images are synthesized through differentiable rendering and the super-resolution network, and the segmentation maps of the hair and the head are also synthesized through differentiable rendering. The HHAvatar are trained under the supervision of multi-view RGB videos and multi-view masks from faceparsing.

| | | |
|---|---|---|
|Head Pose<br><br>time step t-1| |Hair Expressive Gaussians|

|time step t+1|Hair Dyn MLP<br><br>|amic s| |
|---|---|---|---|
|time step t<br><br>Head Pose<br><br>[Figure 41]| | | |

Temporal Module

Hair Neutral Gaussians

Head Pose Hair Expressive Gaussians

|Hair Dynamic MLPs|
|---|

t

time step t+2

t time step t-2

[Figure 42]

[Figure 43]

t

t-1

[Figure 44]

[Figure 45]

Hair Expressive Gaussians

t+1

[Figure 46]

[Figure 47]

[Figure 48]

Hair Dynamic MLPs

- Fig. 3: The detail of the temporal module. The input for the Hair Dynamic MLPs at time step t is X0 (the position of the

minimum distance from the point x to the 3D landmarks P0. t1 = 0.15 and t2 = 0.25 are predefined hyperparameters when the length of the head is set to approximately 1. For the hairs, we set λexp(x) = 0 since the hairs are hardly affected by the expression coefficients.

Color Ct′ of the Gaussians at time step t. Modeling the dynamic details typically requires dynamic color that changes with expressions. As we do not pre-define the neutral value in Eqn. 2, the color are directly predict by two color MLPs: fexpcol ∈ Φ and fposecol ∈ Φ:

##### Ct′ = λexp(X0)fexpcol (F0,θt) +λpose(X0)fposecol (F0,βt).

(4)

Rotation, Scale and Opacity {Q′t,St′,A′t} of the Gaussians at time step t. These three attributes also dynamic, thereby modeling some detailed expressions-related appearance changes. Here, we just use another two attribute MLPs fexpatt ∈ Φ and fposeatt ∈ Φ to predict their shift from the neutral value.

neutral Gaussian point), {Xt′−1,Xt′−2} (the position of the expressive Gaussian point at time step t − 1 and time step

##### {Q′t,St′,A′t} = {Q0,S0,A0} +λexp(X0)fexpatt (F0,θt) +λpose(X0)fposeatt (F0,βt).

t−2), and {βt,βt−1,βt−2} (the pose of the head at time step t, t − 1, and t − 2). The specific details of the Hair Dynamic MLPs are detailed in Sec. 3.1.2.

(5)

3.1.2 Dynamic Hair

Next, we explain the process of the changes in hair with head motion in detail. We assume the head and the hair maintain still at time step 0, which means that hairs do not have extra deformation due to the inertia, the hair positions X0 = X−1 = X−2, and the head poses β0 = β−1 = β−2.

are first estimated through the 3DMM model in the data preprocessing and then optimized in the initialization stage 3.3. Then for each Gaussian point, we calculate the above weight λexp(·) and λpose(·) as follows:

 

Hair deformation. The deformation of hair is related to the previous hair position Xt−1 which can get from Xt′−1

1, dist(x,P0) < t1 t2−dist(x,P 0)

λexp(x) =

t2−t1 , dist(x,P0) ∈ [t1,t2] 0, dist(x,P0) > t2,



- and βt−1, hair speed which can get from Xt′−1, Xt′−2, βt−1,
- and βt−2, head pose βt, and head movement speed which

can get from βt and βt−1. We use a MLP fhairdef ∈ Φ to predict the extra hair deformation.

with λpose(x) = 1 − λexp(x). And x ∈ X0 denotes the position of one neutral Gaussian. dist(x,P0) denotes the

Xt′ = X0 + λhair(X0)fhairdef (X0,Xt′−1,t−2,βt,t−1,t−2). (6)

Here, Xt′−1,t−2 and βt,t−1,t−2 represent respectively the X′ at time step t − 1,t − 2 and the β at time step t,t − 1,t − 2.

λhair(·) represents the extent to which the points of the hair are affected by the head pose. The strength of the hair deformation is related to the distance between the hairs and the scalp. We assume that the Gaussian points closer to the scalp are more affected by the head pose, while the opposite is true for the Gaussian points far away. Specifically, The scalp P1 of the canonical model are first estimated through the FLAME model [24] in the data preprocessing. Then for each Gaussian point, we calculate the above weight λexp(·) and λpose(·) as follows:

 

1, dist(x,P1) < t3 t4−dist(x,P 1)

λhair(x) =

t4−t3 , dist(x,P1) ∈ [t3,t4] 0, dist(x,P1) > t4,



with x ∈ X0 denotes the position of one neutral Gaussian. dist(x,P1) denotes the minimum distance from the point x to the scalp P1. t3 = 0.05 and t4 = 0.15 are predefined hyperparameters when the length of the head is set to approximately 1.

Hair attribution. The Gaussian properties of hair should also undergo corresponding changes after deformation. Here we use two MLPs fhaircol and fhairatt ∈ Φ to predict the extra changes in the Gaussian properties of hair.

Ct′ = Ct′ + fhaircol (X0,Xt′−1,t−2,βt,t−1,t−2), (7)

{Q′t,St′} = {Q0,S0} + fhairatt (X0,Xt′−1,t−2,βt,t−1,t−2). (8) Finally, we apply rigid rotations and translations T(·) to the Gaussians, transforming them from the canonical space to the world space. Note, the transformation is only implemented for directional variables: {Xt′,Q′t}, while the multichannel color, the scale and the opacity {Ct′,St′,A′t} are not directional thus remain unchanged.

{Xt,Qt} = T({Xt′,Q′t},βt), (9) {Ct,St,At} = {Ct′,St′,A′t}. (10)

- 3.2 Training

In this part, we explain the training pipeline of the HHAvatar 3.1 and the loss function. In each iteration, we first generate the expression conditioned 3D Gaussians as Eqn. 2. Then, given a camera view, we render a 32-channel image with 512 resolution IC ∈ R512×512×32 referring to Eqn. 1. After that we feed the image to a super resolution network Ψ to generate a 2048 resolution RGB image Ihr ∈ R2048×2048×3, such that more details are recovered and noise caused by uneven ambient light or camera chromatic aberration in the training data will be filtered out [72], [79]. Meanwhile, the expressive Gaussians are also rendered to a segmentation map for separating the hair from the head.

During training, we jointly optimize all the learnable parameters mentioned above in bold, including the neutral

Gaussians: {X0,F0,Q0,S0,A0}, the dynamic generator: {fexpcol ,fposecol ,fexpdef,fposedef ,fexpatt ,fposeatt fhairdef ,fhaircol ,fhairatt },

and the super resolution network Ψ. Our training loss consists of two parts: the reconstruction loss Lrecon and the mask loss Lmask:

L = Lrecon + λmaskLmask. (11)

Reconstruction loss. For the reconstruction loss function, we only use the foreground RGB images Igt as supervision to construct an L1 loss and a VGG perceptual loss [96] between the generated images Ihr and the ground truth Igt. Besides, we encourage the first three channels of the 32-channel feature image IC to be RGB channels, which is ensured by a L1 loss term. The total loss is:

##### Lrecon = ||Ihr − Igt||1 + λvggV GG(Ihr,Igt) +λlr(||Ilr − Igt||1 + λvggV GG(Ilr,Igt)),

(12)

with Ilr denoting the first three channels of the 32-channel image IC. We set the weights λvgg = 0.1 and λlr = 1.

Mask loss. To separate the points of the head and the hair, we introduce the occlusion perception module to use the head mask and the hair mask to supervise the head part and the hair part. We first use face-parsing [95] to obtain the mask of the hair and the head. Then, we set the Gaussian color of the head and the hair to (1.0,0.0,0.0) and (0.0,1.0,0.0) respectively. Finally, we can get the predicted color by rendering to obtain the head mask and the hair mask, which can avoid supervision of the parts covered by hair.

##### Lmask = ||Mhead′ − Mhead||1 + ||Mhair′ − Mhair||1, (13)

with Mhead′ and Mhair′ denoting the predicted head mask and the predicted hair mask respectively, Mhead and Mhair denoting the head mask and the hair mask from faceparsing [95]. We set the weight λmask = 0.1.

###### 3.3 Geometry-guided Initialization

Unlike neural networks, the Gaussians act as an unordered and unstructured representation. Random initialization leads to failure to converge while naively using the FLAME model to initialize will significantly reduce the reconstruction quality. In this section, we describe in detail how to optimize a mesh guidance model to provide reliable initialization for the Gaussians in Sec. 3.1.

Mesh Guidance Model. Specifically, we first construct two MLPs fsdf head and fsdf hair to represent two signed distance fields for the head and the hair respectively. In addition, this network will also output the corresponding feature vector of each point, which is used for predicting the point color. It can be formulated as:

shead,ηhead = fsdf head(x), (14) shair,ηhair = fsdf hair(x), (15)

with shead and shair denote the SDF value of the head and the hair respectively, ηhead and ηhair denote the feature vector of the head and the hair respectively and x denotes the point position. Then through DMTet [32], we can differentially extract the mesh with vertices X, pervertex feature vectors F, and its faces for the head and the

hair respectively, and we merge the mesh of the head and the hair to obtain the complete mesh of the whole head. We also predict the per-vertex 32-channel color as Eqn. 4 by the two color MLPs fexpcol and fposecol for the head and the hair. In parallel, we construct the two deformation MLPs: fexpdef and fposedef as described in Sec. 3.1 to predict the displacements and add them to the vertex positions. This process is similar to Eqn 3 above, with the Gaussian positions X0 replaced by the vertex positions X. Finally, we also apply rigid rotations and translations to the deformed mesh, transforming it to the world space and rendering the deformed mesh into an image I, a mask M, a head mask Mhead′ , and a hair mask Mhair′ through differentiable rasterization [97] according to the camera parameters µ. Note that during the geometryguided initialization, we temporarily ignore the real physical motion of the hair (i.e., we do not consider the additional motion of hair due to inertia).

Loss Function and Training. Next, we can first construct the RGB loss and the silhouette loss to train the guidance model:

##### LRGB = ||Ir,g,b − Igt||1, (16) Lsil = IOU(M,Mgt), (17)

with Igt and Mgt denote the ground truth RGB image and mask, respectively. IOU(·) denotes Intersection over Union metrics. Note that only the first three channels R,G,B of the 32-channel image I are supervised by the ground truth RGB images.

We also use the estimated 3D facial landmarks Pgt to provide rough guidance for the expression deformation MLP. Specifically, we input the neutral 3D landmarks P0 into the expression deformation MLP to predict the expression conditioned landmarks P:

P = P0 + fexpdef(P0,θ). (18) Then we construct the loss function with 3D facial landmarks Pgt as the supervision:

##### Ldef = ||P − Pgt||2. (19)

Besides, we introduce three constraints: (1) a regularization term Loffset to punish all non-zero displacements to prevent the two deformation MLPs from learning a global constant offset [71], (2) a regularization term Llmk to limit the SDF value at the 3D landmarks to be close to zero, such that the landmarks are located on the surface of the mesh, (3) a Laplacian term Llap for maintaining the extracted mesh smooth to a certain extent. To obtain the signed distance field of the head and the hair respectively, we also use the head mask and the hair mask to supervise the head part and the hair part:

##### Lmask = IOU((1 − Mo) · Mhead′ ,Mhead) +IOU(Mo · Mhair′ ,Mhair),

(20)

##### Mo = Dhair < Dhead, (21)

with Mhead′ and Mhair′ denoting the predicted head mask and the predicted hair mask respectively, Mhead and Mhair denoting the head mask and the hair mask from faceparsing [95] respectively, and Dhead and Dhair denoting the predicted head depth and the predicted hair depth respectively.

Overall, the total loss function in the initialization stage is formulated as:

L = LRGB + λsilLsil + λdefLdef +λoffsetLoffset + λlmkLlmk +λlapLlap + λmaskLmask,

(22)

with λ denoting the weights of each term, which are set as follows: λsil = 0.1, λdef = 1, λoffset = 0.01, λlmk = 0.1, λlap = 100, and λmask = 0.1. We jointly optimize the MLPs mentioned above with the neutral 3D landmarks P0 jointly until all MLPs are converged.

Parameters Transfer. Finally, we use the roughly trained mesh guidance model to initialize the Gaussian model. Specifically, we extract the neutral mesh with vertices X and per-vertex features F through DMTet [32] for the head and the hair respectively, and directly assign their values to the neutral positions X0 = X and the per-vertex feature vectors F0 = F of the neutral Gaussians respectively. Then, we retain all the four optimized MLPs: {fexpcol ,fposecol ,fexpdef,fposedef } for the Gaussian model. For the other neutral attributes: rotation, scale and opacity, we adopt the original initialization strategy in Gaussian Splatting [9]. And the parameters of the two attribute MLPs: {fexpatt ,fposeatt ,fhairdef ,fhaircol ,fhairatt } and the super resolution network Ψ are just randomly initialized.

4 EXPERIMENTS

###### 4.1 Implementation Details

In the experiment, we use 15 sets of data, with 10 from NeRSemble [25], 2 from multi-view video data from HAvatar [5], and 3 extra self-captured and constructed data containing hair motion as there is no complete non-rigid dynamic hair trajectory (e.g., nodding and swinging at different speeds) in NeRSemble [25]. For the 10 identities from NeRSemble, each set contains 2500 to 3000 frames, 16 cameras are distributed about 120 degrees in front, and simultaneously capture 2K resolution video. For each identity, We use the sequences marked with ”FREE” as the evaluation data, and the rest as the training data. For the 2 identities from HAvatar, each set contains 3000 frames, 8 cameras are distributed about 120 degrees in front, and 4K resolution videos are collected simultaneously. Later, we crop the face area and resize all images to 2K resolution. For the 3 identities from self-captured data, each set contains about 1000 frames, 4 cameras are distributed about 90 degrees in front, and 4K resolution videos are collected simultaneously. Later, we crop the face area and resize all images to 2K resolution.

For data preprocessing, we first remove the background [94], segment the hair from the head by the faceparsing [95], and extract 68 2D facial landmarks [99] for all the images. Then, for each frame, we use multi-view images to estimate the corresponding 3D landmarks, the expression coefficients, and the head pose by fitting the Basel Face Model (BFM) [65] to the extracted 2D landmarks. Note that we define the 3D landmarks as the usual 68 landmarks with vertices indexed as multiples of 100 in the BFM vertices.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

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

|[Figure 64]|
|---|

NeRFBlenderShape NeRFace HAvatar Ours GT

- Fig. 4: Qualitative comparisons of different methods on self reenactment task in NeRSemble dataset [5]. From left to right: NeRFBlendShape [70], NeRFace [68], HAvatar [5] and Ours. Our method can reconstruct details like beards, teeth, eyes, etc. with high quality.

###### 4.2 Training Details

until fully convergence.

During the geometry-guided initialization stage, we use an Adam optimizer, and set the learning rate to 1 × 10−3 for all the networks and 1 × 10−4 for the neutral 3D landmarks P0. Then, we train the model for 10000 iterations with a batch size of 4. During the Gaussian model training stage, we also use an Adam optimizer, and set the learning rate to 1 × 10−4 for the three color MLPs {fexpcol ,fposecol ,fhaircol }, the three deformation MLPs {fexpdef,fposedef ,fhairdef }, and the three attribute MLPs {fexpatt ,fposeatt ,fhairatt }, 1 × 10−5 for the neutral positions X0 and the point-wise feature vectors F0, 1×10−4 for the neutral rotation Q0, 3×10−4 for the neutral scale S0, 1 × 10−3 for the neutral opacity Q0 and 1 × 10−4 for the super resolution network Ψ. Finally, we train the Gaussian model for 600000 iterations with a batch size of 1

###### 4.3 Results and Comparisons

Self Reenactment. In this section, we first compare our method with existing SOTA methods in qualitative experiments on self reenactment task. Specifically, NeRFace [68] uses a deep MLP to fit an expression condtioned dynamic NeRF. The current SOTA method HAvatar [5] introduces 3DMM template prior and uses a deep convolutional network to generate a human head NeRF represented by three planes from a mesh template with expression. Note, HAvatar leverages the GAN framework using the adversarial loss function to force the network to generate details that are not view-consistent. For a fair comparison, we remove this part and use VGG perceptual loss as Sec. 3.2 instead.

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

HAvatar GaussianAvatars MeGA Ours GT

- Fig. 5: Qualitative comparisons of different methods on self reenactment task with dynamic hairs in the self-captured dataset. From left to right: HAvatar [5], GaussianAvatars [13], MeGA [98] and Ours. Our method can reconstruct details with high quality.

Method PSNR ↑ SSIM ↑ LPIPS (512) ↓ LPIPS (2K) ↓ FID (2K) ↓

NeRFBlendShape 25.91 0.836 0.123 0.229 54.80 NeRFace 27.14 0.849 0.147 0.234 65.11 HAvatar 27.19 0.883 0.064 0.209 31.06

Ours (w/o SR) 27.82 0.887 0.080 0.202 45.50 Ours 27.70 0.883 0.056 0.098 18.50

- TABLE 1: Quantitative evaluation results of NeRFBlendShape [70], NeRFace [68], HAvatar [5], our method without super resolution and our full method on self reenactment task in NeRSemble dataset [5] and HAvatar dataset [5].

[Figure 76]

- Fig. 6: Qualitative comparisons of different methods on cross-identity reenactment task. From left to right: NeRFBlendShape [70], NeRFace [68], HAvatar [5] and Ours. Our method synthesizes high-fidelity images while ensuring the accuracy of expression transfer.

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

HAvatar GaussianAvatars MeGA Ours Actor

- Fig. 7: Qualitative comparisons of different methods on cross-identity reenactment task with dynamic hairs in the selfcaptured dataset. From left to right: HAvatar [5], GaussianAvatars [13], MeGA [98] and Ours. Our method synthesizes high-fidelity images while ensuring the accuracy of hair movement.

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Novel View Synthesis Results Reference

- Fig. 8: Novel view synthesis results of our method. Top: we use 8-view synchronized videos for training the avatar. Bottom: we use 4-view synchronized videos for training the avatar with dynamic hais.

Method PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ HAvatar 26.25 0.879 0.063 36.34

GaussianAvatars 24.21 0.823 0.181 65.27 MeGA 25.13 0.880 0.165 55.34 Ours 27.05 0.883 0.060 30.53

- TABLE 2: Quantitative evaluation results of the other SOTA methods and our method on self reenactment task with the motion of the hair in our self-captured video.

Qualitative results on self reenactment task are shown in the Fig. 4. Our method can accurately reconstruct pixellevel high-frequency details such as beards, teeth, and hair. Besides, our method can achieve expression transfer more accurately, such as eye movements in the figure.

Next, we conduct a quantitative evaluation for the four

Method PSNR ↑ SSIM ↑ LPIPS ↓

NeRFBlendShape 25.43 0.812 0.148 NeRFace 26.65 0.825 0.151 HAvatar 27.13 0.880 0.65

Ours 27.58 0.882 0.059

TABLE 3: Quantitative evaluation results of the other SOTA methods and our method on 3D consistency.

methods on 5 identities and 6 cameras using the evaluation split. The evaluation metrics include: Peak Signal-to-Noise Ratio (PSNR), Structure Similarity Index (SSIM), Learned Perceptual Image Patch Similarity (LPIPS) [96] and Fr´echet Inception Distance (FID) [100]. Note, we calculate FID by comparing the distribution of all the training images and all the rendered images. As the task mainly focuses on the re-

construction of the head, we use face-parsing [95] to remove the body parts in the image to eliminate their impact in the experiment. As shown in Tab. 1, our method demonstrates a slight improvement in PSNR and SSIM compared with previous methods, and a significant improvement in LPIPS and FID, which means that our method can generate more high-frequency details. Note that HHAvatar only made additional changes to the position and attributes of the Gaussians in the hair part, and the final rendering method remains the same as the previous version [33]. Therefore, when the dataset is organized chronologically and the hair maintains relatively static compared to the head, the reconstructed results are consistent with the previous version [33]. When the dataset is not in chronological order, simply canceling the change in hair inertia will result in the same reconstruction as the previous version [33].

We also compare our method with existing SOTA methods in qualitative experiments on self reenactment task with dynamic hairs. Qualitative results on self reenactment task are shown in the Fig. 5. It can be seen that all other methods fail to capture the accurate physical motion of the hair component and the hair part is blurry. Furthermore, as a result of the occlusion created by hair during motion, the face itself appears blurry. Because all existing methods neglect the kinematic attributes of hair, none of them can successfully complete self reenactment task with the motion of the hair. By modeling hair separately from the human head and introducing the kinematic features of the hair, our method can accurately reconstruct high-frequency details of the hair. Besides, our method can capture the realistic physical motion of hair.

Quantitative results on self reenactment task with dynamic hairs are shown in Tab. 2. As shown in Tab. 2, our method demonstrates a slight improvement in PSNR and SSIM compared with previous methods, and a significant improvement in LPIPS, which means that our method can generate more high-frequency details.

Cross-Identity Reenactment. We qualitatively compare our method with the above SOTA methods on cross-identity reenactment task. As shown in Fig. 6, our method is able to synthesize higher-fidelity images with more accurate expression transfer and richer emotions.

We also qualitatively compare our method with the above SOTA methods on cross-identity reenactment task with dynamic hairs. As shown in Fig. 7, other methods did not consider the status of the hair at the previous moment, resulting in no reasonable physical motion of the hair during driving. Therefore, those methods fail to capture the accurate physical motion of the hair component and the hair part is blurry. Our method is able to synthesize higherfidelity images with more accurate hair motion.

Novel View Synthesis. In this section, we show the results of novel view synthesis as shown in the top of Fig. 8 shown. In this case, we use video data from 8 views for training and render the image at a new viewpoint. Next, we quantitatively evaluate the 3D consistency of our method and compare it with other SOTA methods mentioned above. Specifically, we select the 5 identities as above and use the video data from 8 cameras for training while the other 8 hold-out cameras for evaluation. The evaluation metrics include PSNR, SSIM and LPIPS (512 resolution). The results

[Figure 94]

Fig. 9: Ablation study on the initialization strategies: FLAME-initialization and our geometry-guided initialization. Our strategy ensures the hair strands away from the head are well reconstructed.

are shown in Tab. 3. Our method outperforms other methods in 3D consistency.

We also show the results of novel view synthesis with dynamic hairs, and the results are shown in the bottom of Fig. 8. In this case, we use our self-captured video data from 4 views for training and render the image at five new viewpoints. As shown in the bottom of Fig. 8, our method can maintain 3D consistency and synthesize high-fidelity images with accurate hair motion.

###### 4.4 Ablation Study

Method PSNR ↑ SSIM ↑ LPIPS ↓ FLAME-Init 28.73 0.875 0.123

Mesh-Deform 28.83 0.874 0.116 Ours 28.94 0.876 0.108

TABLE 4: Quantitative evaluation results of the other two ablation baselines and ours on self reenactment task.

Ablation on Initialization Strategies. In order to verify the effectiveness of our geometry-guided initialization strategy 3.3, we compare it with the strategy to use the FLAME model for initialization (FLAME-Init). Specifically, after fitting a FLAME model through multi-view data, we first subdivide the FLAME mesh 4 times and use the neutral vertices as the positions of the neutral Gaussians. Then, the expression deformation MLP is optimized to learn the displacement of FLAME vertices. We set the per-vertex feature to zeros, while randomly initialize the parameters of the expression color MLP. The initialization of other variables is the same as our strategy. Qualitative results are shown in the Fig. 9. Due to the lack of initialization for the hair and shoulders in FLAME-Init, the points to model these parts are offset from nearby vertices, which leads to sparseness of the Gaussians, resulting in blurring.

Ablation on Deformation Modeling Approaches. We compare our fully learned deformation field with the previous

Method PSNR ↑ SSIM ↑ LPIPS ↓ W/o Dynamic Hair 26.23 0.825 0.123

[Figure 95]

W/o Occlusion Perception 26.83 0.874 0.116 Full Model 27.02 0.876 0.108

TABLE 5: Quantitative evaluation results of the baseline and our full model on self reenactment task.

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

W/o Occlusion Perception Ours Actor

- Fig. 12: Ablation study on the dynamic hair module: our approach can capture the motion of the hair and reconstruct more accurate hair.

[Figure 101]

- Fig. 13: Failure case: our method produce relatively less exaggerated results.

- Fig. 10: Ablation study on the deformation modeling: mesh LBS-based deformation and our fully learned deformation. Our approach can learn complex and exaggerated expressions.

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

W/o Dynamic Hair Module Ours GT

[Figure 108]

- Fig. 11: Ablation study on the dynamic hair module: our approach can capture the motion of the hair and reconstruct more accurate hair.

separately from the head, not only will the hair not have realistic physical motion, but it will also produce blurring. Therefore, it is necessary to model hair separately. Quantitative results are shown in the Tab. 5. It can be seen that our method outperforms both the two ablation baselines on PSNR, SSIM and LPIPS metrics.

Ablation on Occlusion Perception Module. To assess the effectiveness of the mask that considers occlusion, we designed an experiment in which we used alpha rendering 2 for separating the hair and the head. Qualitative results are shown in the Fig. 12. It can be seen that due to the lack of consideration for occlusion, the hole may sometimes occur in the face during driving because of the hair motion. Therefore, it is necessary to consider the occlusion. Quantitative results are shown in the Tab. 5. It can be seen that our method outperforms both the two ablation baselines on PSNR, SSIM and LPIPS metrics.

mesh-based deformation (Mesh-Deform). Specifically, we migrate the method in INSTA [23] for controlling the NeRF deformation to our Gaussians. First we fit a 3DMM mesh template. Then, for each Gaussian point, find the closest face on the mesh, and calculate the deformation gradient to estimate the displacement. Qualitative results are shown in the Fig. 10. For some expressions that cannot be captured well by the 3DMM mesh template, our method can learn accurate deformation, thereby achieving the modeling of complex expressions. Quantitative results are shown in the Fig. 4. Our method outperforms both the two ablation baselines on PSNR, SSIM and LPIPS metrics.

#### 5 DISCUSSION AND CONCLUSION

Ablation on Hair Dynamic Module. To assess the effectiveness of independently modeling hair, we designed an experiment in which we kept modeling the head and the hair in the same manner. Qualitative results are shown in the Fig. 11. It can be seen that if the hair is not modeled

Ethical Considerations. Our method is capable of creating artificial portrait videos, which have the potential to dis-

2. https://github.com/ashawkey/diff-gaussian-rasterization

seminate misinformation, influence public perceptions, and erode confidence in media sources.

Limitation. For the tongue and teeth inside the mouth, blurring is sometimes produced in our method due to the lack of tracking methods. On the other hand, the reconstructed head avatar cannot make expressions other than those in the training set. Therefore, when the actor’s expression is too exaggerated, our method will output relatively less exaggerated results as shown in Fig. 13. When the head movement speed in the training set is excessively high, leading to complete hair coverage over facial features, the fitting of 3DMM expression coefficients will fail. Hence, it is essential to avoid moving the head too rapidly.

Conclusion. In this paper, we propose HHAvatar, a novel representation for head avatar reconstruction, which leverages dynamic 3D Gaussians controlled by a fully learned expression deformation and models the physical motion of the hairs. Experiments demonstrate our method can synthesize ultra high-fidelity images while modeling exaggerated expressions and capturing the realistic physical motion of the hair. In addition, we propose a well-designed minute-level initialization strategy to ensure the training convergence and a temporal module to fuse time series information. We believe our HHAvatar will become the mainstream direction for head avatar reconstruction in the future.

#### REFERENCES

- [1] S. Lombardi, J. Saragih, T. Simon, and Y. Sheikh, “Deep appearance models for face rendering,” ACM Trans. Graph., vol. 37, no. 4, pp. 68:1–68:13, Jul. 2018. 1, 3
- [2] S. Lombardi, T. Simon, G. Schwartz, M. Zollhoefer, Y. Sheikh, and J. Saragih, “Mixture of volumetric primitives for efficient neural rendering,” ACM Trans. Graph., vol. 40, no. 4, jul 2021. 1, 3
- [3] S. Ma, T. Simon, J. Saragih, D. Wang, Y. Li, F. D. La Torre, and Y. Sheikh, “Pixel codec avatars,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 64–73. 1, 3
- [4] C. Wang, D. Kang, Y. Cao, L. Bao, Y. Shan, and S.-H. Zhang, “Neural point-based volumetric avatar: Surface-guided neural points for efficient and photorealistic volumetric head avatar,” in ACM SIGGRAPH Asia 2023 Conference Proceedings, 2023. 1
- [5] X. Zhao, L. Wang, J. Sun, H. Zhang, J. Suo, and Y. Liu, “Havatar: High-fidelity head avatar via facial model conditioned neural radiance field,” ACM Trans. Graph., 2023. 1, 3, 7, 8, 9, 10
- [6] A. Raj, M. Zollhoefer, T. Simon, J. Saragih, S. Saito, J. Hays, and S. Lombardi, “Pva: Pixel-aligned volumetric avatars,” in arXiv:2101.02697, 2020. 1, 3
- [7] M. Mihajlovic, A. Bansal, M. Zollhoefer, S. Tang, and S. Saito, “Keypointnerf: Generalizing image-based volumetric avatars using relative spatial encoding of keypoints,” in European conference on computer vision, 2022. 1, 3
- [8] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in Proceedings of the European Conference on Computer Vision (ECCV), 2020. 1, 3
- [9] B. Kerbl, G. Kopanas, T. Leimkuhler,¨ and G. Drettakis, “3d gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics, 2023. 1, 3, 4, 7
- [10] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, and X. Wang, “4d gaussian splatting for real-time dynamic scene rendering,” 2023. 1, 3
- [11] J. Luiten, G. Kopanas, B. Leibe, and D. Ramanan, “Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis,” 2023. 1, 3
- [12] Z. Yang, H. Yang, Z. Pan, X. Zhu, and L. Zhang, “Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting,” 2023. 1, 3

- [13] S. Qian, T. Kirschstein, L. Schoneveld, D. Davoli, S. Giebenhain, and M. Nießner, “Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 20299–

20309. 1, 3, 9, 10

- [14] Z. Shao, Z. Wang, Z. Li, D. Wang, X. Lin, Y. Zhang, M. Fan, and Z. Wang, “Splattingavatar: Realistic real-time human avatars with mesh-embedded gaussian splatting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1606–1616. 1, 3
- [15] Z. Zhao, Z. Bao, Q. Li, G. Qiu, and K. Liu, “Psavatar: A pointbased morphable shape model for real-time head avatar creation with 3d gaussian splatting,” arXiv preprint arXiv:2401.12900, 2024. 1, 3
- [16] Y. Chen, L. Wang, Q. Li, H. Xiao, S. Zhang, H. Yao, and Y. Liu, “Monogaussianavatar: Monocular gaussian point-based head avatar,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–9. 1, 3
- [17] A. Rivero, S. Athar, Z. Shu, and D. Samaras, “Rig3dgs: Creating controllable portraits from casual monocular videos,” arXiv preprint arXiv:2402.03723, 2024. 1, 3
- [18] J. Xiang, X. Gao, Y. Guo, and J. Zhang, “Flashavatar: High-fidelity head avatar with efficient gaussian embedding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1802–1812. 1, 3
- [19] J. Wang, J.-C. Xie, X. Li, F. Xu, C.-M. Pun, and H. Gao, “Gaussianhead: Impressive head avatars with learnable gaussian diffusion,” arXiv preprint arXiv:2312.01632, 2023. 1, 3
- [20] Y. Zheng, W. Yifan, G. Wetzstein, M. J. Black, and O. Hilliges, “Pointavatar: Deformable point-based head avatars from videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 1, 3
- [21] S. Athar, Z. Xu, K. Sunkavalli, E. Shechtman, and Z. Shu, “Rignerf: Fully controllable neural 3d portraits,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 1, 3
- [22] Y. Zheng, V. F. Abrevaya, M. C. Buhler,¨ X. Chen, M. J. Black, and O. Hilliges, “I m avatar: Implicit morphable head avatars from videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 13535–

13545. 1, 3

- [23] W. Zielonka, T. Bolkart, and J. Thies, “Instant volumetric head avatars,” 2022. 1, 3, 12
- [24] T. Li, T. Bolkart, M. J. Black, H. Li, and J. Romero, “Learning a model of facial shape and expression from 4d scans,” ACM Trans. Graph., vol. 36, no. 6, nov 2017. 1, 2, 3, 6
- [25] T. Kirschstein, S. Qian, S. Giebenhain, T. Walter, and M. Nießner, “Nersemble: Multi-view radiance field reconstruction of human heads,” ACM Trans. Graph., 2023. 1, 3, 7
- [26] V. Sklyarova, J. Chelishev, A. Dogaru, I. Medvedev, V. Lempitsky, and E. Zakharov, “Neural haircut: Prior-guided strand-based hair reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 19762–19773. 2
- [27] R. A. Rosu, S. Saito, Z. Wang, C. Wu, S. Behnke, and G. Nam, “Neural strands: Learning hair geometry and appearance from multi-view images,” in European Conference on Computer Vision. Springer, 2022, pp. 73–89. 2
- [28] Z. Wang, G. Nam, T. Stuyck, S. Lombardi, C. Cao, J. Saragih, M. Zollh¨ofer, J. Hodgins, and C. Lassner, “Neuwigs: A neural dynamic model for volumetric hair capture and animation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 2, 4
- [29] Z. Wang, G. Nam, T. Stuyck, S. Lombardi, M. Zollh¨ofer, J. Hodgins, and C. Lassner, “Hvh: Learning a hybrid neural volumetric representation for dynamic hair performance capture,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 6143–6154. 2
- [30] H. Luo, M. Ouyang, Z. Zhao, S. Jiang, L. Zhang, Q. Zhang, W. Yang, L. Xu, and J. Yu, “Gaussianhair: Hair modeling and rendering with light-aware gaussians,” arXiv preprint arXiv:2402.10483, 2024. 2, 4
- [31] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2
- [32] T. Shen, J. Gao, K. Yin, M.-Y. Liu, and S. Fidler, “Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape

- synthesis,” in Advances in Neural Information Processing Systems (NeurIPS), 2021. 2, 4, 6, 7
- [33] Y. Xu, B. Chen, Z. Li, H. Zhang, L. Wang, Z. Zheng, and Y. Liu, “Gaussian head avatar: Ultra high-fidelity head avatar via dynamic gaussians,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 11
- [34] M. Levoy, K. Pulli, B. Curless, S. Rusinkiewicz, D. Koller, L. Pereira, M. Ginzton, S. Anderson, J. Davis, J. Ginsberg, J. Shade, and D. Fulk, “The digital michelangelo project: 3d scanning of large statues,” in Proceedings of the 27th Annual Conference on Computer Graphics and Interactive Techniques. USA: ACM Press/Addison-Wesley Publishing Co., 2000. 3
- [35] T. Beeler, B. Bickel, P. Beardsley, B. Sumner, and M. Gross, “Highquality single-shot capture of facial geometry,” ACM Trans. Graph., 2010. 3
- [36] A. Ghosh, G. Fyffe, B. Tunwattanapong, J. Busch, X. Yu, and P. Debevec, “Multiview face capture using polarized spherical gradient illumination,” ACM Trans. Graph., 2011. 3
- [37] D. Bradley, W. Heidrich, T. Popa, and A. Sheffer, “High resolution passive facial performance capture,” in ACM SIGGRAPH 2010 papers, 2010, pp. 1–10. 3
- [38] T. Li, S. Liu, T. Bolkart, J. Liu, H. Li, and Y. Zhao, “Topologically consistent multi-view face inference using volumetric sampling,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 3824–3834. 3
- [39] T. Bolkart, T. Li, and M. J. Black, “Instant multi-view head capture through learnable registration,” in Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 768–779. 3
- [40] Y. Xiao, H. Zhu, H. Yang, Z. Diao, X. Lu, and X. Cao, “Detailed facial geometry recovery from multi-view images by learning an implicit function,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2022. 3
- [41] K. Yang, H. Shang, T. Shi, X. Chen, J. Zhou, Z. Sun, and W. Yang, “Asm: Adaptive skinning model for high-quality 3d face modeling,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 20708–20717. 3
- [42] S. Bi, S. Lombardi, S. Saito, T. Simon, S.-E. Wei, K. Mcphail, R. Ramamoorthi, Y. Sheikh, and J. Saragih, “Deep relightable appearance models for animatable faces,” ACM Trans. Graph., vol. 40, no. 4, jul 2021. 3
- [43] Z. Wang, T. Bagautdinov, S. Lombardi, T. Simon, J. Saragih, J. Hodgins, and M. Zollhofer, “Learning compositional radiance fields of dynamic human heads,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 5704–5713. 3
- [44] S. Lombardi, T. Simon, J. Saragih, G. Schwartz, A. Lehrmann, and Y. Sheikh, “Neural volumes: Learning dynamic renderable volumes from images,” ACM Trans. Graph., vol. 38, no. 4, pp. 65:1–65:14, Jul. 2019. 3
- [45] P. Paysan, R. Knothe, B. Amberg, S. Romdhani, and T. Vetter, “A 3d face model for pose and illumination invariant face recognition,” in 2009 Sixth IEEE International Conference on Advanced Video and Signal Based Surveillance, 2009, pp. 296–301. 3
- [46] C. Cao, Y. Weng, S. Zhou, Y. Tong, and K. Zhou, “Facewarehouse: A 3d facial expression database for visual computing,” in IEEE Transactions on Visualization and Computer Graphics, vol. 20, 2014, pp. 413–425. 3
- [47] A. Brunton, T. Bolkart, and S. Wuhrer, “Multilinear wavelets: A statistical shape space for human faces,” in Proceedings of the Proceedings of the European Conference on Computer Vision (ECCV),

2014. 3

- [48] V. Blanz and T. Vetter, “A morphable model for the synthesis of 3d faces,” in 26th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH 1999). ACM Press, 1999, pp. 187–194. 3
- [49] S. Wu, Y. Yan, Y. Li, Y. Cheng, W. Zhu, K. Gao, X. Li, and G. Zhai, “Ganhead: Towards generative animatable neural head avatars,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 437–447. 3
- [50] L. Wang, Z. Chen, T. Yu, C. Ma, L. Li, and Y. Liu, “Faceverse: a fine-grained and detail-controllable 3d face morphable model from a hybrid dataset,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2022. 3
- [51] T. Yenamandra, A. Tewari, F. Bernard, H. Seidel, M. Elgharib, D. Cremers, and C. Theobalt, “i3dmm: Deep implicit 3d morphable model of human heads,” in Proceedings of the IEEE/CVF

- Conference on Computer Vision and Pattern Recognition (CVPR), June 2021. 3
- [52] Y. Zhuang, H. Zhu, X. Sun, and X. Cao, “Mofanerf: Morphable facial neural radiance field,” in Proceedings of the European Conference on Computer Vision (ECCV), 2022. 3
- [53] Y. Hong, B. Peng, H. Xiao, L. Liu, and J. Zhang, “Headnerf: A real-time nerf-based parametric head model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 20374–20384. 3
- [54] D. Wang, P. Chandran, G. Zoss, D. Bradley, and P. Gotardo, “Morf: Morphable radiance fields for multiview neural head modeling,” in ACM SIGGRAPH 2022 Conference Proceedings, ser. SIGGRAPH ’22. New York, NY, USA: Association for Computing Machinery, 2022. 3
- [55] C. Cao, T. Simon, J. K. Kim, G. Schwartz, M. Zollhoefer, S.-S. Saito, S. Lombardi, S.-E. Wei, D. Belko, S.-I. Yu, Y. Sheikh, and J. Saragih, “Authentic volumetric avatars from a phone scan,” ACM Trans. Graph., vol. 41, no. 4, jul 2022. 3
- [56] J. Sun, X. Wang, L. Wang, X. Li, Y. Zhang, H. Zhang, and Y. Liu, “Next3d: Generative neural texture rasterization for 3daware head avatars,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3
- [57] C. Cao, D. Bradley, K. Zhou, and T. Beeler, “Real-time highfidelity facial performance capture,” ACM Trans. Graph., vol. 34, no. 4, jul 2015. 3
- [58] C. Cao, H. Wu, Y. Weng, T. Shao, and K. Zhou, “Real-time facial animation with image-based dynamic avatars,” ACM Trans. Graph., vol. 35, no. 4, jul 2016. 3
- [59] A. E. Ichim, S. Bouaziz, and M. Pauly, “Dynamic 3d avatar creation from hand-held video input,” ACM Trans. Graph., vol. 34, no. 4, jul 2015. 3
- [60] L. Hu, S. Saito, L. Wei, K. Nagano, J. Seo, J. Fursund, I. Sadeghi, C. Sun, Y.-C. Chen, and H. Li, “Avatar digitization from a single image for real-time rendering,” ACM Trans. Graph., vol. 36, no. 6, nov 2017. 3
- [61] Y. Deng, J. Yang, S. Xu, D. Chen, Y. Jia, and X. Tong, “Accurate 3d face reconstruction with weakly-supervised learning: From single image to image set,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June

2019. 3

- [62] K. Nagano, J. Seo, J. Xing, L. Wei, Z. Li, S. Saito, A. Agarwal, J. Fursund, and H. Li, “Pagan: Real-time avatars using dynamic textures,” ACM Trans. Graph., vol. 37, no. 6, dec 2018. 3
- [63] P.-W. Grassal, M. Prinzler, T. Leistner, C. Rother, M. Nießner, and J. Thies, “Neural head avatars from monocular rgb videos,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 18632–18643. 3
- [64] T. Khakhulin, V. Sklyarova, V. Lempitsky, and E. Zakharov, “Realistic one-shot mesh-based head avatars,” in Proceedings of the European Conference on Computer Vision (ECCV), 2022. 3
- [65] T. Gerig, A. Morel-Forster, C. Blumer, B. Egger, M. Luthi, S. Sch¨onborn, and T. Vetter, “Morphable face models-an open framework,” in 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018). IEEE, 2018, pp. 75–82. 3, 4, 7
- [66] Y. Guo, K. Chen, S. Liang, Y.-J. Liu, H. Bao, and J. Zhang, “Ad-nerf: Audio driven neural radiance fields for talking head synthesis,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021, pp. 5764–5774. 3
- [67] X. Liu, Y. Xu, Q. Wu, H. Zhou, W. Wu, and B. Zhou, “Semanticaware implicit neural audio-driven video portrait generation,” in Proceedings of the European Conference on Computer Vision (ECCV),

2022. 3

- [68] G. Gafni, J. Thies, M. Zollhofer, and M. Niessner, “Dynamic neural radiance fields for monocular 4d facial avatar reconstruction,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 8645–8654. 3, 4, 8, 9
- [69] S. Athar, Z. Shu, and D. Samaras, “Flame-in-nerf: Neural control of radiance fields for free view face animation,” in IEEE 17th International Conference on Automatic Face and Gesture Recognition (FG), 2023, pp. 1–8. 3
- [70] X. Gao, C. Zhong, J. Xiang, Y. Hong, Y. Guo, and J. Zhang, “Reconstructing personalized semantic facial nerf models from monocular video,” ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia), vol. 41, no. 6, 2022. 3, 8, 9
- [71] Y. Xu, L. Wang, X. Zhao, H. Zhang, and Y. Liu, “Avatarmav: Fast 3d head avatar reconstruction using motion-aware neural

- voxels,” in ACM SIGGRAPH 2023 Conference Proceedings, 2023. 3, 7
- [72] Y. Xu, H. Zhang, L. Wang, X. Zhao, H. Han, Q. Guojun, and Y. Liu, “Latentavatar: Learning latent expression code for expressive neural head avatar,” in ACM SIGGRAPH 2023 Conference Proceedings, 2023. 3, 6
- [73] M. Qin, Y. Liu, Y. Xu, X. Zhao, Y. Liu, and H. Wang, “Highfidelity 3d head avatars reconstruction through spatially-varying expression conditioned neural radiance field,” in AAAI Conference on Artificial Intelligence, 2023. 3
- [74] W. Yifan, F. Serena, S. Wu, C. Oztireli,¨ and O. Sorkine-Hornung, “Differentiable surface splatting for point-based geometry processing,” ACM Transactions on Graphics (proceedings of ACM SIGGRAPH ASIA), vol. 38, no. 6, 2019. 3
- [75] O. Wiles, G. Gkioxari, R. Szeliski, and J. Johnson, “SynSin: Endto-end view synthesis from a single image,” in CVPR, 2020. 3
- [76] C. Lassner and M. Zollhofer, “Pulsar: Efficient sphere-based neural rendering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 1440–

1449. 3

- [77] G. Kopanas, T. Leimkuhler,¨ G. Rainer, C. Jambon, and G. Drettakis, “Neural point catacaustics for novel-view synthesis of reflections,” ACM Transactions on Graphics (TOG), vol. 41, no. 6, pp. 1–15, 2022. 3
- [78] K.-A. Aliev, A. Sevastopolsky, M. Kolos, D. Ulyanov, and V. Lempitsky, “Neural point-based graphics,” in European Conference on Computer Vision, 2020, pp. 696–712. 3
- [79] D. Ruckert,¨ L. Franke, and M. Stamminger, “Adop: Approximate differentiable one-pixel point rendering,” ACM Trans. Graph.,

2022. 3, 6

- [80] Q. Xu, Z. Xu, J. Philip, S. Bi, Z. Shu, K. Sunkavalli, and U. Neumann, “Point-nerf: Point-based neural radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5438–5448. 3
- [81] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, and X. Jin, “Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction,” 2023. 3
- [82] Z. Li, Z. Zheng, L. Wang, and Y. Liu, “Animatable gaussians: Learning pose-dependent gaussian maps for high-fidelity human avatar modeling,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [83] S. Zheng, B. Zhou, R. Shao, B. Liu, S. Zhang, L. Nie, and Y. Liu, “Gps-gaussian: Generalizable pixel-wise 3d gaussian splatting for real-time human novel view synthesis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [84] R. Shao, J. Sun, C. Peng, Z. Zheng, B. Zhou, H. Zhang, and Y. Liu, “Control4d: Efficient 4d portrait editing with text,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,

2024. 3

- [85] F. Bertails, S. Hadap, M.-P. Cani, M. Lin, T.-Y. Kim, S. Marschner, K. Ward, and Z. Kaˇci´c-Alesi´c, “Realistic hair simulation: animation and rendering,” in ACM SIGGRAPH 2008 classes, 2008. 3
- [86] M. Chai, C. Zheng, and K. Zhou, “A reduced model for interactive hairs,” ACM Transactions on Graphics (TOG), 2014. 3
- [87] M. Chai, C. Zheng, and K. Zhou, “Adaptive skinning for interactive hair-solid simulation,” IEEE transactions on visualization and computer graphics, 2016. 3
- [88] P. Guan, L. Sigal, V. Reznitskaya, and J. K. Hodgins, “Multilinear data-driven dynamic hair model with efficient hairbody collision handling,” in Proceedings of the 11th ACM SIGGRAPH/Eurographics conference on Computer Animation, 2012. 3
- [89] Q. Lyu, M. Chai, X. Chen, and K. Zhou, “Real-time hair simulation with neural interpolation,” IEEE Transactions on Visualization and Computer Graphics, 2020. 3
- [90] M. Chai, J. Ren, and S. Tulyakov, “Neural hair rendering,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVIII 16. Springer,

2020. 3

- [91] K. Olszewski, D. Ceylan, J. Xing, J. Echevarria, Z. Chen, W. Chen, and H. Li, “Intuitive, interactive beard and hair synthesis with generative models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 3
- [92] C. Wu and T. Kanai, “Data-driven detailed hair animation for game characters,” Computer Animation and Virtual Worlds, 2016. 3

- [93] E. Zakharov, V. Sklyarova, M. Black, G. Nam, J. Thies, and O. Hilliges, “Human hair reconstruction with strand-aligned 3d gaussians,” arXiv preprint arXiv:2409.14778, 2024. 4
- [94] S. Lin, A. Ryabtsev, S. Sengupta, B. Curless, S. Seitz, and

I. Kemelmacher-Shlizerman, “Real-time high-resolution background matting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2021. 4, 7

- [95] C. Yu, C. Gao, J. Wang, G. Yu, C. Shen, and N. Sang, “Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation,” International journal of computer vision, 2021. 4, 6, 7, 11
- [96] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2018, pp. 586–595. 6, 10
- [97] J. Munkberg, J. Hasselgren, T. Shen, J. Gao, W. Chen, A. Evans, T. Muller,¨ and S. Fidler, “Extracting Triangular 3D Models, Materials, and Lighting From Images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 8280–8290. 7
- [98] C. Wang, D. Kang, H.-Y. Sun, S.-H. Qian, Z.-X. Wang, L. Bao, and S.-H. Zhang, “Mega: Hybrid mesh-gaussian head avatar for high-fidelity rendering and head editing,” arXiv preprint arXiv:2404.19026, 2024. 9, 10
- [99] A. Bulat and G. Tzimiropoulos, “How far are we from solving the 2d & 3d face alignment problem? (and a dataset of 230,000 3d facial landmarks),” in International Conference on Computer Vision,

2017. 7

- [100] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” in Neural Information Processing Systems, 2017. 10

