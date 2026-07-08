# arXiv:2407.21686v1[cs.CV]31Jul2024

## Expressive Whole-Body 3D Gaussian Avatar

##### Gyeongsik Moon1,2 , Takaaki Shiratori2 , and Shunsuke Saito2

1DGIST 2Codec Avatars Lab, Meta mks0601@dgist.ac.kr {tshiratori,shunsukesaito}@meta.com https://mks0601.github.io/ExAvatar

Abstract. Facial expression and hand motions are necessary to express our emotions and interact with the world. Nevertheless, most of the 3D human avatars modeled from a casually captured video only support body motions without facial expressions and hand motions. In this work, we present ExAvatar, an expressive whole-body 3D human avatar learned from a short monocular video. We design ExAvatar as a combination of the whole-body parametric mesh model (SMPL-X) and 3D Gaussian Splatting (3DGS). The main challenges are 1) a limited diversity of facial expressions and poses in the video and 2) the absence of 3D observations, such as 3D scans and RGBD images. The limited diversity in the video makes animations with novel facial expressions and poses non-trivial. In addition, the absence of 3D observations could cause significant ambiguity in human parts that are not observed in the video, which can result in noticeable artifacts under novel motions. To address them, we introduce our hybrid representation of the mesh and 3D Gaussians. Our hybrid representation treats each 3D Gaussian as a vertex on the surface with pre-defined connectivity information (i.e., triangle faces) between them following the mesh topology of SMPL-X. It makes our ExAvatar animatable with novel facial expressions by driven by the facial expression space of SMPL-X. In addition, by using connectivity-based regularizers, we significantly reduce artifacts in novel facial expressions and poses.

### 1 Introduction

Humans use all facial expressions, body motions, and hand motions to express our emotions and intentions, and interact with other people and objects. In particular, facial expressions and hand gestures are one of the most powerful channels for non-verbal communication, and hand motions are necessary to interact with diverse types of objects. Modeling the facial expression, body motion, and hand motion altogether is extremely challenging. Several whole-body 3D human geometry models have been introduced [2,21,37,50]. Among them, SMPL-X [37] is the most widely used one, which motivated a number of 3D whole-body pose estimation methods [4,9,11,26,28,32,45,52] and benchmarks [36].

To represent 3D humans beyond the minimally clothed parametric models, personalized 3D human avatars have been recently studied. The 3D human avatar is a representation that combines 3D geometry and the appearance of a

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

(a) Monocular video from a single person

(b) Animatable expressive whole-body 3D avatar

- Fig. 1: From (a) a monocular video from a single person, we create our (b) ExAvatar, an expressive whole-body 3D avatar, animatable with novel facial expression code, hand poses, and body poses of SMPL-X.

certain person, which can be animated and rendered with novel poses. However, most of existing 3D human avatars [6, 8, 15, 18–20, 24, 25, 38, 39] modeled from a casually captured video only support body motions without facial expressions and hand motions. Their avatars bake facial expressions and hand poses, and animating them is not possible. A recent work [47] introduced a whole-body avatar that supports animation with facial expressions, and body and hand poses; however, it requires 3D observations, such as 3D scans or RGBD images with highly accurate SMPL-X registrations, with diverse poses and facial expressions. Such an assumption does not hold for the majority of casually captured videos in daily life.

We present ExAvatar, an expressive whole-body 3D human avatar that can be made from a short monocular video. ExAvatar is designed as a combination of the whole-body 3D parametric model (SMPL-X) [37] and 3D Gaussian Splatting (3DGS) [22]. It utilizes the whole-body drivability of SMPL-X and the photorealistic and efficient rendering capability of 3DGS. After the training, it is animatable with novel facial expression code and 3D pose of SMPL-X, as shown in Fig. 1. Despite its desired properties, modeling ExAvatar is an non-trivial task with the following two challenges: 1) a limited diversity of facial expressions and poses in the video and 2) the absence of 3D observations, such as 3D scans and RGBD videos. The limited diversity in the video makes a drivability with novel facial expressions and poses non-trivial. In addition, the absence of 3D observations creates ambiguity in the occluded human parts, exhibiting noticeable artifacts in novel facial expressions and poses.

To address them, we propose a novel hybrid representation of the surface mesh and 3D Gaussians in ExAvatar. Our hybrid representation treats each 3D Gaussian as a vertex on the surface, where the vertices have pre-defined connectivity (i.e., triangle faces) between them following the mesh topology of

Expressive Whole-Body 3D Gaussian Avatar 3

SMPL-X. Existing volumetric avatars [6,8,15,19,20,25,38,39,47] do not have the connectivity by the definition. Also, previous 3DGS-based [18,24] works consider a set of 3D Gaussian points as a point cloud without considering the connectivity between them.

Using our hybrid representation, our ExAvatar becomes fully compatible with the facial expression space of SMPL-X. Therefore, it can be driven with any facial expression code of SMPL-X even from a short monocular video without diverse facial expressions. As our 3D Gaussians share the exactly same mesh topology with SMPL-X, we simply add the vertex offsets to our 3D Gaussian points to move them according to the facial expression as in FLAME [27] and SMPLX [37]. Hence, unlike previous works [47], our drivability of the facial expression is not strictly limited by the number of training frames (e.g., 30 seconds of a short video).

Another benefit is that we can significantly reduce artifacts in novel facial expressions and poses using connectivity-based regularizers. As the pose diversity in the training set is very limited, there could be human parts that are not observed at all in the video. Without 3D observations, the ambiguous human parts could suffer from artifacts in novel poses. While several point-based regularizers (e.g., L2 regularization of the underlying SMPL/SMPL-X template mesh) have been proposed [18], they do not consider connectivity between vertices. Such a lack of connectivity could introduce floating 3D Gaussians.By considering the connectivity, we can naturally enforce local similarity, significantly reducing artifacts.

Throughout our experiments, our method substantially outperforms all previous 3D human avatars in various benchmarks. Our contributions can be summarized as follows.

- – We present ExAvatar, an expressive whole-body 3D human avatar that can be made from a short monocular video without requiring 3D observations.
- – We propose a hybrid representation of the surface mesh and 3D Gaussians. It allows ExAvatar to be animated with any novel facial expression code of SMPL-X even from a short monocular video without diverse facial expressions.
- – Using connectivity information between 3D Gaussians, we significantly reduce artifacts, especially in novel facial expressions and poses.

- 2 Related works
- 3D human avatars. Various 3D representations are used for modeling 3D human avatars. Alldieck et al. [1] extended SMPL mesh with per-vertex offsets. Bagautdinov et al. [3] achieved high-fidelity results using conditional variational autoencoder, which can be animated with incomplete driving signals. Remelli et al. [43] propose to use texel-aligned features, a localized representation. Motivated by neural radiance fields [31], many volumetric and implicit representation-based avatars have been introduced. Peng et al. [38,39] created a 3D human avatar from a capture studio, which provides accurate 3D pose and

multi-view images for supervision. Kwon et al. [25] improved the previous works by utilizing vertex-aligned features. Shen et al. [47] created a whole-body 3D avatar, which supports whole-body animation with facial expression, from their capture studio dataset. In contrast to the above works that make 3D avatars from capture studios, recent works focus on making 3D avatars from a short monocular video without requiring 3D observations, such as 3D scans, RGBD images, or multi-view images. Jiang et al. [20] introduced a dataset and method for making a 3D human avatar from a short monocular video taken from inthe-wild environments. Guo et al. [15] proposed a system that can decompose a scene and human with self-supervised learning. Jiang et al. [19] introduced a system that can make a 3D human within several minutes. Recently introduced 3DGS [22], which achieves both powerful and efficient rendering capability, motivated several 3DGS-based avatars [18,24,30]. Kocabas et al. [24] use the triplane for creating 3D avatar. Hu et al. [18] introduced a robust system that takes a positional map of a posed SMPL mesh. Moon et al. [34] presented universal hand model (UHM) to create authentic hand avatars from a phone scan. Chen et al. [7] extended UHM of Moon et al. [34] for the relightability.

Except for a few works [3, 30, 47], most of the above works only support body motions without hand motions and facial expressions. X-Avatar [47] supports whole-body animation including facial expressions; however, it has two limitations. First, it requires diverse facial expressions with accurate 3D geometry registration in videos to create avatars. This is because they cannot directly utilize the facial expression space of FLAME [27]/SMPL-X [37]. They need to transform the mesh-based facial expression space of FLAME [27]/SMPL-X [37] to their implicit representation using learnable modules. To train the transformation module, they need training data with sufficiently diverse facial expressions and accurate 3D geometry registrations. Second, it requires 3D observations, such as 3D scans or RGBD images with accurate SMPL-X registrations, for the training, hard to obtain from in-the-wild environments. Due to the above two reasons, X-Avatar [47] is hard to apply to practical settings, such as short monocular videos. Liu et al. [30] proposed another whole-body 3D avatar; however, their avatar is not animated with novel facial expressions.

Whole-body 3D human modeling and perception. Modeling face, body, and hands at the same time is an extremely challenging problem as each human part has its own different characteristics. Several whole-body 3D human models have been introduced [2,21,37,50], which model 3D geometry of minimally clothed humans. They are parametric models, parameterized by 3D poses, facial expression code, and shape parameter. Among them, SMPL-X [37] is the most widely used one due to its completeness. Motivated by the optimization baseline [37] and benchmarks [36], a number of 3D whole-body pose estimation methods [4,9,11,26,28,32,45,52] have been introduced.

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

###### Expressive Whole-Body 3D Gaussian Avatar 5

| | | |
|---|---|---|

[Figure 24]

[Figure 25]

[Figure 26]

(b) With joint offset (Ours)

(c) Without joint offset

(b) With face offset (Ours)

(c) Without face offset

(a) Image

(a) Image

- Fig. 2: The effectiveness of our joint offset ∆J and face offset ∆Vface. They are necessary for the accurate registration of hands and face, which results in accurate coregistration of the whole body.
- 3 ExAvatar

#### 3.1 Accurate co-registration of SMPL-X

We assume videos, which usually consist of 30 seconds of frames, are taken from an in-the-wild environment. The video is from a single person with a natural backgrounds. Before training our ExAvatar, we first preprocess the video. Following previous works [15,20], we first run an off-the-shelf SMPL-X regressor [26]

- and 2D pose estimator [10] to all frames. Then, we additionally fit the regressed SMPL-X parameters (i.e., 3D poses θ ∈ R55×3, shape parameter β ∈ R100, and facial expression code ψ ∈ R50) and 3D translation t to the estimated 2D pose of each frame. The shape parameter is shared across all frames as all frames are from the same person.

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

One challenge during registering SMPLX parameters to a video is accurate coregistration of body, hands, and face, unique challenges of the whole-body avatar. The registration of hands and face can be negatively affected by a limited expressiveness of SMPL-X and registration accuracy of the body, which can limit the co-registration accuracy. To achieve the accurate co-registration of body, hands, and face, we introduce two optimizable offsets initialized with zero and shared across all frames. Both offsets are identity (ID)-dependent offsets and are not dependent on the poses and facial expressions. Hence, they are added to the T-pose template mesh of SMPL-X before performing the linear blend skinning (LBS).

[t]

###### (a) With the face offset (Ours) (b) Without the face offset

Fig. 3: Without the face offset ∆Vface, the final 3D geometry of the avatar becomes totally inauthentic and inaccurate. For each setting, normals of 3D Gaussian points and colors are used for the rendering.

First, we introduce joint offset ∆J, added to the joints in the T-pose space of SMPL-X. The joint offset ∆J, which affect both 3D skeleton and surface, are

especially helpful to fit hands more perfectly as the shape parameter of SMPL-X has limited coverage of 3D hand skeleton, as shown in Fig. 2 left. Second, we introduce face offset ∆Vface, per-vertex offset of the face region of SMPL-X, added to the face vertex in the T-pose space of SMPL-X. To optimize the face offset ∆Vface, we first fit the 3D face-only model (i.e., FLAME [27]) to 2D poses and images of the face by running DECA [12] and further optimizing it to 2D poses. Then, we optimize the face offset by making a summation of the face offset and 3D face vertices of SMPL-X close to fitted FLAME vertices. The optimization is straightforward as the face region of SMPL-X has exactly the same topology as that of FLAME. The rationale is that 1) the face-only model has higher expressiveness in its shape space than the whole-body model and 2) the registration of the face-only model is not affected by the body registration. Fig. 2 right and 3 show the effectiveness of our face offsets. Such a special treatment in the registration stage is greatly helpful for the final 3D avatar, not considered in previous whole-body avatars [30,47]. Please refer to the supplementary material about the details of the fitting.

#### 3.2 Architecture

- Fig. 4 shows the architecture of ExAvatar. We model ExAvatar on top of a

canonical 3D human mesh, denoted by V¯ ∈ RN×3, where it has N = 167K upsampled vertices and 335K upsampled triangle faces. To obtain it, we first pass the optimized SMPL-X shape parameter β, joint offsets ∆J, and face offsets ∆Vface from Sec. 3.1, and a pre-defined neural pose (i.e., 大-pose) to the SMPLX layer. Then, we upsample it with the subdivision function of PyTorch3D [42], which can upsample other 3D assets, such as facial expression blend shapes, in a consistent way.

Per-vertex Gaussian assets regression. We initialize a learnable triplane [5] T ∈ R3×C×H×W with zero, where C = 32, H = 128, and W = 128 represent channel dimension, height, and width of the triplane, respectively. Then, we prepare a positional encoding mesh P¯ ∈ RN×3 with a pre-defined neutral pose (i.e., 大-pose) and zero shape parameter. We upsample the positional encoding mesh with the above subdivision function, which produces the same mesh topology as the canonical mesh V¯ . We extract the per-vertex feature from the triplane by orthogonally projecting P¯ to each plane and performing the bilinear interpolation. The triplane is useful as it naturally enforces similarity between close vertices. In practice, we construct another triplane dedicated to the face, as the face requires detailed geometry and appearance modeling with a small physical size. The reason for not using the canonical mesh V¯ for the feature extraction is that it keeps changing during the training as we further optimize the shape parameter β and the joint offset ∆J during the training. If the position of a certain vertex changes, the extracted triplane feature of that vertex could be one that was from other vertices, which can make the training unstable.

The interpolated features from the triplane are concatenated, denoted by F ∈ RN×96. We pass F to two multi-layer perceptrons (MLPs), which regress 1)

##### 3D offset ∆Vtri ∈ RN×3 and scale Stri ∈ RN×1 and 2) RGB values Ctri ∈ RN×3

###### Expressive Whole-Body 3D Gaussian Avatar 7

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Canonical mesh ( )

[Figure 36]

[Figure 37]

Pos. enc. mesh ( )

[Figure 38]

[Figure 39]

Per-vertex feature ( )

[Figure 40]

###### MLPs

LBS + 3DGS

[Figure 41]

Triplane ( )

[Figure 42]

Per-vertex normal

[Figure 43]

MLPs

3D animatable avatar

[Figure 44]

[Figure 45]

Animated avatar with driving signals

[Figure 46]

3D pose ( ) Facial expr. ( )

Driving signals

- Fig. 4: The architecture of our ExAvatar. From the canonical mesh V¯ , triplane T, pervertex normal, and 3D pose θ, we build a 3D animatable avatar. Then, with driving signals, 3D pose θ and facial expression code ψ of SMPL-X [37], we animate the avatar and render it to the screen space with 3DGS [22]. For the normal rendering, we calculate the normal vectors using the positions of 3D Gaussian points and mesh topology of SMPL-X.

for the 3DGS, respectively. The MLPs are shared across all vertices. Motivated by Hu et al. [18], for better generalization to novel viewpoints, we limit all Gaussian assets to be isotropic by limiting a degree of freedom of the scale to 1 and setting the rotation and opacity to identity and one, respectively. Please refer to the supplementary material for the detailed architecture of the MLPs. The regressed Gaussian assets (i.e., 3D offset, scale, and RGB values) are solely from the triplane, shared across all frames. Hence, they represent identity (ID)-dependent and environment (e.g., lighting)-dependent assets without pose dependency as ID and environment are fixed in the input video, while pose changes for each frame.

To additionally model pose-dependent deformations, we employ two additional MLPs. The first MLP takes F and 3D poses θ without the root pose and outputs 3D vertex offset ∆Vpose ∈ RN×3 and scale offset ∆Spose ∈ RN×1. The second MLP takes F, 3D poses θ without the root pose, and the normal vector of each vertex and outputs RGB offset ∆Cpose ∈ RN×3. The additional normal vector can 1) provide the view-dependent shading information to the network and 2) be useful to disentangle geometry and appearances [15, 46]. Thanks to our hybrid representation, we can easily obtain the per-vertex normal vector by averaging normals of triangle faces that include the vertex. Instead of directly predicting pose-dependent Gaussian assets, ours output pose-dependent offsets. This is helpful for the generalization to novel poses as Gaussian assets solely from the triplane already have reasonable expressiveness, which makes the role of the pose-dependent Gaussian assets small. Such a design is especially important when making 3D avatars from a short video like ours as limited pose diversity makes generalization to novel poses challenging.

#### 3.3 Animation and rendering

- Fig. 5 shows examples of our animated and rendered avatars, made from short monocular videos. Animation. We need to animate Gaussian points from the canonical space with given facial expression code ψ and 3D poses θ of SMPL-X. To this end, we first

replace the pose-dependent vertex offset ∆Vpose of hand and face vertices to those of SMPL-X. This is because the hand and face are often naked; hence, we can directly utilize vertex offsets of SMPL-X. Then, we add vertex offsets from facial expression code ψ of SMPL-X to face vertices. By directly using the facial expression offsets of SMPL-X, we do not have to learn a new facial expression space. Such a direct utilizing is from our hybrid representation of the mesh and 3D Gaussians. The below equations describe the above deformations in the canonical space.

V¯ tri = V¯ + ∆Vtri + ∆Vexpr, (1)

V¯ pose = V¯ + ∆Vtri + ∆Vpose + ∆Vexpr, (2)

where ∆Vexpr represents the facial expression offset of SMPL-X, obtained from the facial expression code ψ. Then, for the body vertices, we take the skinning weight of the nearest vertices from downsampled V¯ , while for the hand and face vertices, we use the original skinning weight of the vertices. This is because, for the body vertices, their semantic meaning could change due to the cloth geometry. The final animated geometry, Vtri and Vpose, are represented with below equations.

Vtri = LBS(V¯ tri,θ,Wtri) and Vpose = LBS(V¯ pose,θ,Wpose), (3)

where Wtri and Wpose represent the skinning weight of V¯ tri and V¯ pose, respectively. Rendering. To render animated 3D geometry, we use 3DGS rendering pipeline [22] like the below equations.

Itri = f(Vtri,exp(Stri),Ctri,K,E), (4)

Ipose = f(Vpose,exp(Stri + ∆Spose),Ctri + ∆Cpose,K,E), (5)

where f, K, and E represent rendering function of 3DGS, camera intrinsic, and extrinsic matrices, respectively. As described above, following Hu et al. [18], we restrict all Gaussian assets to isotropic for better generalization; hence, rotation and opacity of all Gaussian points are set to identity and one, respectively, not described in the equations.

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Expressive Whole-Body 3D Gaussian Avatar 9

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

- Fig. 5: Our animated expressive whole-body avatars, made from monocular videos of NeuMan dataset [20]. Avatars of each row are animated with the same facial expression code ψ and 3D pose θ of SMPL-X.

#### 3.4 Loss functions

During the training of our ExAvatar, we optimize the triplane T, MLPs for the regression of Gaussian assets in Sec. 3.2, 3D pose θ of each frame, facial expression code ψ of each frame, 3D translation t of each frame, the shape parameter β, and the joint offset ∆J. Also, we simultaneously optimize a 3DGS for the background following the original implementation [22] by segmenting out human regions using human masks from an off-the-shelf human segmentation model [16]. Modeling background simultaneously produces better foreground mask [15], as estimated masks often have errors, especially on hand parts. We denote rendered images from a combination of Eq. 4 and the 3DGS for the background by I∗tri. Likewise, we denote rendered images from a combination of

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

###### 10 Moon et al.

mouth open

mouth open

(a) With the face loss (Ours) (b) Without the face loss

- Fig. 6: The effectiveness of our face loss. Without the face loss, the geometry and texture of the face are not consistent, which makes significant artifacts when driving. The right one (b) shows that without the face loss, when the mouth is opened, the upper lip remains at the same position, while only below lip is opened.

(a) With Lap. reg. (Ours) (b) Without Lap. reg. (c) Without Lap. reg. + strong L2 reg.

[Figure 71]

| |
|---|

[Figure 72]

[Figure 73]

| |
|---|

[Figure 74]

[Figure 75]

| |
|---|

[Figure 76]

[Figure 77]

[Figure 78]

- Fig. 7: The effectiveness of the Laplacian regularizer, which makes 3D avatars in novel facial expressions and poses greatly stable. On the other hand, the widely used L2 regularizer to the distance from SMPL-X surface to 3D Gaussian points suffers from severe artifacts. We successfully incorporated the Laplacian regularizer using our hybrid representation of the surface mesh and 3D Gaussians.

Eq. 5 and the 3DGS for the background by I∗pose. To train ExAvatar, we minimize the below loss functions.

Image loss. Following 3DGS [22], we minimize L1 distance, 1 - SSIM, and LPIPS [53] between rendered images (i.e., I∗tri and I∗pose) and the captured image. We found that the additional LPIPS is helpful for sharper textures. To save the computation, we compute the image loss after cropping the human region.

Face loss. Unlike other human parts, the face has its unique characteristics as there should be a strong consistency between geometry and texture. For example, lip geometry usually has reddish textures. If other face geometry has lip textures, in novel facial expressions or jaw poses, the lip would not properly change, which can lead to significant artifacts, as shown in Fig. 6 (b). Simply minimizing the above image loss does not guarantee the consistency between the geometry and texture of the face region. To enforce the consistency, we minimize the L1 distance between the rendered face image with a standard differentiable mesh renderer and the captured image, where the texture for the mesh renderer is prepared by averaging the unwrapped UV texture of the face-only model [27] registrations from Sec. 3.1. The UV texture is fixed, and the positions of 3D Gaussian points of the face region are adjusted to minimize the loss function.

Expressive Whole-Body 3D Gaussian Avatar 11

- Fig. 6 (a) shows the effectiveness of our face loss functions. Thanks to our hybrid representation of the mesh and 3D Gaussians, such a mesh-based loss function can be easily incorporated into our system. Regularizers. Due to the limited pose diversity in the training set, there can be human parts that are not observed in the video. Such human parts suffer from occlusion ambiguity, which could result in artifacts in novel facial expressions and poses. In addition, to utilize the facial expression offsets of SMPL-X, we need to make the face geometry similar to that of SMPL-X. To address them, we utilize connectivity-based regularizers (i.e., Laplacian regularizer), motivated by the mesh modeling works [29,33]. Fig. 7 shows that our connectivity-based regularizer significantly reduces artifacts in novel facial expressions and poses. We minimize the difference of the 1) Laplacian of deformed 3D Gaussian points in

the canonical space (i.e., V¯ tri and V¯ pose) and 2) Laplacian of the canonical mesh V¯ . In this way, we can easily encourage the local similarity between 3D Gaussian points, which can prevent floating 3D Gaussians.In particular, our connectivitybased regularizer is much more effective than the widely used L2 regularizer [18] that simply penalizes distance between 3D Gaussian points and underlying template mesh without considering the connectivity information. Due to our hybrid representation of the mesh and 3D Gaussians, the Laplacian regularizer, widely used in mesh modeling works, can be easily included in our system. In addition to regularizing the 3D positions of 3D Gaussian points, we compute the same Laplacian regularizer for the scales and RGBs of 3D Gaussian points. For other regularizers, please refer to the supplementary material.

### 4 Experiments

#### 4.1 Datasets

NeuMan. NeuMan [20] provides several short monocular videos taken from inthe-wild environments. Each video contains a single person walking around for about 15 seconds. Following previous works [18] we use bike, citron, jogging, and seattle videos that exhibit most human body regions and contain minimal blurry images. We follow their official training and testing splits.

X-Humans. X-Humans [47] provides 3D scans and RGBD videos of multiple subjects, captured from a studio. Compared to NeuMan, X-Humans has more diverse facial expressions and hand poses. There are two experimental protocols: 1) using 3D scans and 2) using RGBD images for creating avatars. We create our avatar only with monocular RGB videos without depthmaps and compare ours against previous works [47] that use RGBD videos. We use 0028, 0034, and 0087 subjects as their pre-trained weights of the RGBD protocol are publicly available. We follow their official training and testing splits.

#### 4.2 Comparison to state-of-the-art methods

Tab. 1 and 2 show that our ExAvatar achieves the best results on NeuMan [20] dataset regardless of whether the rendered background pixels are included or

12 Moon et al.

Table 1: Comparisons of 3D human avatars on the test set of NeuMan [20] Rendered backgrounds are considered in the evaluation. Only our ExAvatar supports face and hand animations.

|Methods|PSNR↑ SSIM↑ LPIPS↓|
|---|---|
|NeuMan [20] Vid2Avatar [15] HUGS [24] ExAvatar (Ours)|24.22 0.77 0.27<br><br>15.41 0.53 0.66<br><br>25.17 0.83 0.16<br><br><br>27.47 0.90 0.10|

Table 2: Comparisons of 3D human avatars on the test set of NeuMan [20]. Rendered backgrounds are not considered in the evaluation. Only our ExAvatar supports face and hand animations.

|Methods<br><br>|PSNR↑ SSIM↑ LPIPS↓|
|---|---|
|HumanNeRF [48] InstantAvatar [19] NeuMan [20] Vid2Avatar [15] GaussianAvatar [18] 3DGS-Avatar [41] ExAvatar (Ours)|27.06 0.967 0.019<br><br>28.47 0.972 0.028<br><br>29.32 0.972 0.014<br><br>30.70 0.980 0.014<br><br><br>29.94 0.980 0.012 28.99 0.974 0.016 34.80 0.984 0.009<br><br>|

- Table 3: Comparisons of 3D human avatars on the test set of X-Humans [47]. Methods with * use additional depth maps for the training.

|Methods|00028 PSNR↑ SSIM↑ LPIPS↓<br><br>|00034 PSNR↑ SSIM↑ LPIPS↓<br><br>|00087 PSNR↑ SSIM↑ LPIPS↓|
|---|---|---|---|
|X-Avatar [47]* ExAvatar (Ours)<br><br>|28.57 0.976 0.026 30.58 0.981 0.018|28.05 0.965 0.035 28.75 0.966 0.029<br><br>|30.89 0.970 0.030 32.01 0.972 0.025|

not. All numbers are from papers [18,24] except NeuMan [20], Vid2Avatar [15],

- and 3DGS-Avatar [41] of Tab. 2, measured with their officially released code. To exclude background pixels, we used an off-the-shelf segmentation network [16] following GaussianAvatar [18]. Following previous works [6, 18, 19, 41], for the evaluation on NeuMan dataset, we fit SMPL-X parameters of testing frames while freezing all other parameters with the image loss of Sec. 3.4.

Fig. 8 shows that ours produces photorealistic renderings in novel views and poses. For example, prints on the shirts (the first and third rows) are significantly sharper and clearer than those of previous works. Most importantly, ours produces faces and hands in novel views and poses substantially better than previous avatars. As previous avatars do not have controllability on faces and hands, the averaged blurry textures are baked in (faces in the first row and hands in the second row and fourth row). On the other hand, ours has sharp textures benefiting from the whole-body modeling.

Tab. 3 and Fig. 9 show that our ExAvatar outperforms previous whole-body avatar [47] on X-Humans [47] dataset even without using depth maps, while the previous work relies on it. Our hybrid representation of the surface mesh and 3D Gaussians leads to stable training and shaper textures of faces and hands. For X-Avatar’s results, we used their officially released pre-trained weights and code. Following Shen et al. [47], we used given SMPL-X parameters without further fitting them to testing frames.

#### 4.3 Ablation study

In this section, we ablate the effectiveness of our hybrid representation of the surface mesh and 3D Gaussians, which enables us to incorporate Laplacian regularizer and face loss into our system. Fig. 7 and Tab. 4 show that incorporating

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

###### Expressive Whole-Body 3D Gaussian Avatar 13

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

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

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

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

(a) GT (b) ExAvatar (Ours) (c) Vid2Avatar (d) NeuMan (e) 3DGS-Avatar

- Fig. 8: Qualitative comparison of our ExAvatar, Vid2Avatar [15], NeuMan [20], and 3DGS-Avatar [41] on the test set of Neuman [20].

- Table 4: Ablation study for the effectiveness of incorporating Laplacian regularizer to our 3D Gaussian-based system on the test set of NeuMan [20].

|Settings|PSNR↑ SSIM↑ LPIPS↓<br><br>|
|---|---|
|Without Lap. reg. With Lap. reg. (Ours)|28.21 0.968 0.199 34.80 0.984 0.009<br><br>|

Table 5: Ablation study for the effectiveness of our face loss on the cropped face images of a test set of X-Humans [47].

|Settings|PSNR↑ SSIM↑ LPIPS↓<br><br>|
|---|---|
|Without face loss With face loss (Ours)|20.02 0.671 0.06 22.07 0.693 0.06<br><br>|

the Laplacian regularizer into our system brings significant performance boost and stability. Fig. 6 and Tab. 5 show the benefit of the proposed face loss. The numbers in Tab. 5 are measured only for cropped face images when the face is visible to evaluate the effectiveness of the face loss. The face visibility is decided by rasterizing SMPL-X meshes and checking the number of rasterized triangle faces of the face region.

- 5 Conclusion

Summary. We present ExAvatar, an expressive whole-body 3D avatar that can be made from a short monocular video. We propose a hybrid representation of the surface mesh and 3D Gaussians to address 1) the limited diversity of facial expressions and poses in the video and 2) the absence of 3D observations, such as 3D scans and RGBD images. Our hybrid representation makes ExAvatar fully

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

###### 14 Moon et al.

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

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

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

###### (a) GT (b) ExAvatar (Ours) (c) X-Avatar

- Fig. 9: Qualitative comparison between our ExAvatar and X-Avatar [47] on the test set of X-Humans [47].

compatible with the facial expression space of SMPL-X and significantly reduces artifacts in novel facial expressions and novel poses.

Limitations. First, as the inside of the mouth including the cavity and palm of the hands are often not observed in the video, our model hallucinates plausible geometry and textures. Second, like previous avatars [8,15,18–20,24,25,30,38,39, 47], ours struggles in modeling dynamic clothes. Material of clothes with motion information, such as velocity and acceleration, should be considered to properly model such dynamic clothes, out of our scope.

Future works. To hallucinate unobserved human parts better, such as inside of the mouth, score distillation sampling [40] can be used to generate images and use them for supervision. In addition, adding relightability to our ExAvatar is a promising and interesting future direction.

### Supplementary Material for “Expressive Whole-Body 3D Gaussian Avatar"

In this supplementary material, we provide more experiments, discussions, and other details that could not be included in the main text due to the lack of pages. The contents are summarized below:

- – Sec. A: Details of co-registration of SMPL-X
- – Sec. B: Detailed architecture of MLPs
- – Sec. C: Regularizers
- – Sec. D: Running time comparison
- – Sec. E: 3D geometry comparison
- – Sec. F: Comparison to generative AIs
- – Sec. G: Implementation details
- – Sec. H: Failure cases

### A Details of co-registration of SMPL-X

This section describes details of the co-registration of SMPL-X of Sec. 3.1 of the main manuscript.

FLAME registration. Given DECA [12]’s output FLAME parameters (i.e., face shape parameter, facial expression code, and jaw pose), we further fit them to 2D keypoints of face, where the 2D keypoints are from an off-the-shelf regressor [10]. For the fitting, we minimize the below loss functions.

LFLAME = Lkpt + 0.1Linit, (6)

where Lkpt and Linit represent 1) L1 distance between projected and target 2D keypoints and 2) L1 distance between optimizing FLAME parameters and the initial regressed ones, respectively. In this way, we can make FLAME’s meshes pixel-aligned with the image while preventing them from being too far from DEAC’s regressed ones.

SMPLX registration. Given Hybrid-X [26]’s output SMPL-X parameters, we further fit them to 2D keypoints of the whole body, where the 2D keypoints are from an off-the-shelf regressor [10]. We replace the facial expression code of Hybrid-X with that of DECA as DECA’s output is more accurate. As FLAME and SMPL-X share the same facial expression space, we share the facial expression code of FLAME and SMPL-X during the registration. Before performing the LBS inside of the SMPL-X layer, we 1) add the joint offset ∆J to the T-pose joint locations of SMPL-X and 2) add the face offset ∆Vface to the face region of the T-pose template mesh. The joint offset allows us to obtain a more personalized 3D skeleton and surface than only using the SMPL-X shape parameter. Please note that the joint offset affects both skeleton and surface as the added joint locations are used for following forward kinematics and LBS.

We minimize the below loss functions. LSMPLX = Lkpt + 0.1Linit + Lface + Lreg, (7)

16 Moon et al.

Lface = 10Lvertex + 10000Llap + Ledge, (8)

Lreg = 0.01Lshape + 100Ljo + Lsym, (9)

where Lkpt and Linit represent 1) L1 distance between projected and target 2D keypoints and 2) L1 distance between optimizing SMPL-X parameters and the initial regressed ones, respectively. We compute the Lkpt with and without the face offset ∆Vface to prevent the face offset from representing any posedependent and expression-dependent things. Lface is to optimize the face offset ∆Vface. Lvertex, Llap, and Ledge represent 1) L1 distance between SMPL-X face vertices and FLAME vertices, 2) L2 distance between Laplacian of SMPL-X face mesh and FLAME mesh, and 3) L1 distance between edge length of SMPL-X face mesh and FLAME mesh, respectively. We compute Lvertex with and without the pose and facial expression to prevent the face offset from representing any posedependent and expression-dependent things. Llap and Ledge are computed only for meshes without pose and facial expressions. We compute Lface only when the face is visible. To check the face visibility, we compute two vectors. First, a vector from the face center to the middle of the eyes in the xz space of the cameracentered coordinate system. Second, a vector from the camera position (origin) to the face center in xz space of the camera-centered coordinate system. We decide the face is visible if the dot product between the two vectors is smaller than cos(135°). Finally, Lreg is to regularize the SMPL-X shape parameter, joint offset, and face offset. Lshape is a squared L2 norm of the SMPL-X shape parameter. Ljo is a squared L2 norm of the joint offset, which prevents an extreme joint offset, and Lsym is for symmetricity of the joint offset and face offset, similar to Feng et al. [12].

### B Architecture of MLPs

This section describes the detailed architecture of MLPs of Sec. 3.2 of the main manuscript. The MLPs are briefly described in Fig. 4 of the main manuscript.

#### B.1 MLPs without pose conditioning

The MLPs without the pose conditioning (the red MLPs of Fig. 4 of the main manuscript) consist of two types of MLPs. The first MLP takes the triplane feature F and outputs the geometry of 3D Gaussian points (i.e., ∆Vtri and Stri). It consists of four fully connected layers with a hidden size of 128. We use the group normalization [49] and ReLU activation function between each fully connected layer. The second MLP takes the triplane feature F and outputs RGB colors of 3D Gaussian points Ctri. It has the same architecture as that of the first MLP except for the output dimension.

Table A: Frames per second comparisons of 3D human avatars.

NeuMan [20] Vid2Avatar [15] X-Avatar [47] ExAvatar (Ours) ExAvatar (Ours wo. pose cond.) 0.02 0.04 0.12 26 47

Table B: 3D geometry comparison between our ExAvatar and X-Avatar [47] on the subset of the test set of X-Humans [47].

|Methods|Mask IoU ↑ Depthmap (mm) ↓ Normal consistency ↑|
|---|---|
|X-Avatar ExAvatar (Ours)<br><br>|94.13 11.24 0.823 96.11 10.97 0.868|

#### B.2 MLPs with pose conditioning

The MLPs with the pose conditioning (the blue MLPs of Fig. 4 of the main manuscript) consist of two types of MLPs. The MLPs have the same architecture as that of the MLPs of Sec. B.1 except for the input and output dimensions. The first MLP takes the triplane feature F and 3D pose θ of SMPL-X without the root pose and outputs geometry offsets of 3D Gaussian points (i.e., ∆Vpose and ∆Spose). The second MLP takes the 1) triplane feature F, 2) 3D pose θ of SMPL-X without the root, and 3) normal of each 3D Gaussian point and outputs RGB offset ∆Cpose.

### C Regularizers

This section describes regularizers, not introduced in Sec. 3.4 of the main manuscript. First, to prevent 3D Gaussian points from being too far from underlying canonical mesh V¯ , we penalize squared L2 norm of 3D Gaussian offsets ∆Vtri and ∆Vpose. Second, to prevent 3D Gaussian from being too big, we penalize the squared L2 norm of 3D Gaussian scales. Third, as hands often suffer from noisy colors due to their small scales and complicated articulations, we minimize the squared L2 distance between RGB colors of hand 3D Gaussian points and the average colors of hand 3D Gaussian points. Finally, we use the same Ljo of Sec. A.

### D Running time comparison

Tab. A shows that ours achieves real-time animation and rendering speed while existing works [15, 20] fail to. Without the pose conditioning (the blue MLP of Fig. 4 of the main manuscript), ours achieves even faster speed. For all methods, we measure the running time only for the human animation and rendering without the background rendering. A single NVIDIA V100 GPU is used, and the rendering resolution is 1024 × 1024.

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

###### 18 Moon et al.

| |
|---|

| |
|---|

| |
|---|

(a) 3D scan (b) ExAvatar (Ours) (c) X-Avatar

- Fig. A: Qualitative comparison of rendered normals from (a) our ExAvatar and (b) X-Avatar [47] on the test set of X-Humans [47].

[Figure 184]

[Figure 185]

[Figure 186]

(a) Capture (b) ExAvatar (Ours) (c) MagicAnimate

- Fig. B: Qualitative comparison of (b) our ExAvatar and (c) state-of-the-art generative method [51] for the human animation.

### E Geometry comparison

Fig. A and Tab. B show that our ExAvatar produces better 3D geometry than previous state-of-the-art 3D whole-body avatar [47], especially around the face and neck. It is not straightforward to extract actual 3D geometry from

- 3D Gaussians. Instead, we render masks/depth maps/normal maps of Gaussians to multiple viewpoints and comparing them with ground truth. However, when rendering depth/normals, even recent SOTA methods [13] simply take the depth/normal from the center of Gaussians and alpha-blend it in the screen space. This may result in incorrect depth/normal maps as elongated Gaussians can deviate from the depth/normal values of the Gaussian centers. Nevertheless, we report 3D geometry comparisons in the figure and table for reference. We leave actual 3D geometry extraction from ExAvatar as a future work.

[Figure 187]

[Figure 188]

[Figure 189]

(a) Image (b) SMPL-X mesh (c) ExAvatar (Ours)

- Fig. C: Failure case of our ExAvatar due to (b) the wrong regressed facial expression code ψ from the off-the-shelf regressor [12].

[Figure 190]

(a) Image (b) SMPL-X mesh (c) ExAvatar (Ours)

[Figure 191]

[Figure 192]

- Fig. D: Failure case of our ExAvatar due to the baked facial appearance of a subject.

### F Comparison to generative AIs

Motivated by the powerful modeling capability of the diffusion models [17,44], several human animation methods [51] have been introduced. They can animate humans only from a single image using DensePose [14] or 2D pose as the driving signals. Despite their photorealistic and plausible outputs, they lack authenticity, as shown in Fig. B. The person of the (b) generated image from MagicAnimate [51] has definitely a different face identity and clothes compared to the (a) captured image, although they look similar. The result of MagicAnimate [51] is obtained using their official released code. This is because of the hallucination, one of the biggest challenges for modeling generative AIs. On the other hand, ours might lack plausibility, especially for unobserved human parts such as inside of mouth and cloth dynamics; however, ours has more authenticity. We think combining the plausibility of generative approaches and the authenticity of our approach should be an interesting and future direction, as discussed in Sec. 5 of the main manuscript.

### G Implementation details

PyTorch [35] is used for implementation. For the training, we use Adam optimizer [23]. The initial learning rate is set to 10−3 and reduced by a factor of 10 at the 75 % and 90 % of total number of iterations. We use a single V100 NVIDIA GPU for experiments, and the training takes 1.5 hours to 5 hours depending on the duration of the videos. All other details will be available in our code.

### H Failure cases

Fig. C shows that our rendered avatar does not have a smiling facial expression despite the person in the image is smiling. This is because of the wrong regressed facial expression code ψ from the off-the-shelf regressor [12] as (b) rendered SMPL-X mesh also has the wrong facial expression. As our ExAvatar is animated with the regressed facial expression code, ours also has wrong facial expression. We think advanced facial models and high-fidelity regressors should be investigated to address this failure case.

Fig. D additionally shows our failure case. Although the person in (a) the image is frowning, our rendered avatar is slightly smiling. This is because the subject of our avatar in the NeuMan dataset [20] is always smiling during the short video capture. Hence, the smiling facial appearance is baked into our avatar. Canonicalizing facial appearance from a short monocular is greatly challenging. In particular, when the face takes a few pixels in the video like our case, 2D keypoints are often noisy, which makes the lip geometry of SMPL-X/FLAME are not fit perfectly in the co-registration stage (Sec. 3.1 of the main manuscript). Such a misalignment error is propagated to our ExAvatar learning stage, which causes the failure case. We think using priors of the canonical facial appearances using generative methods can be one way to address this failure case.

### References

- 1. Alldieck, T., Magnor, M., Xu, W., Theobalt, C., Pons-Moll, G.: Video based reconstruction of 3d people models. In: CVPR (2018)
- 2. Alldieck, T., Xu, H., Sminchisescu, C.: imGHUM: Implicit generative models of 3D human shape and articulated pose. In: ICCV (2021)
- 3. Bagautdinov, T., Wu, C., Simon, T., Prada, F., Shiratori, T., Wei, S.E., Xu, W., Sheikh, Y., Saragih, J.: Driving-signal aware full-body avatars. ACM TOG (2021)
- 4. Cai, Z., Yin, W., Zeng, A., Wei, C., Sun, Q., Yanjun, W., Pang, H.E., Mei, H., Zhang, M., Zhang, L., et al.: SMPLer-X: Scaling up expressive human pose and shape estimation. NeurIPS (2023)
- 5. Chan, E.R., Lin, C.Z., Chan, M.A., Nagano, K., Pan, B., De Mello, S., Gallo, O., Guibas, L.J., Tremblay, J., Khamis, S., et al.: Efficient geometry-aware 3D generative adversarial networks. In: CVPR (2022)
- 6. Chen, J., Zhang, Y., Kang, D., Zhe, X., Bao, L., Jia, X., Lu, H.: Animatable neural radiance fields from monocular RGB videos. arXiv preprint arXiv:2106.13629

(2021)

- 7. Chen, Z., Moon, G., Guo, K., Cao, C., Pidhorskyi, S., Simon, T., Joshi, R., Dong, Y., Xu, Y., Pires, B., Wen, H., Evans, L., Peng, B., Buffalini, J., Trimble, A., McPhail, K., Schoeller, M., Yu, S.I., Romero, J., Zollhöfer, M., Sheikh, Y., Liu, Z., Saito, S.: URhand: Universal relightable hands. In: CVPR (2024)
- 8. Choi, H., Moon, G., Armando, M., Leroy, V., Lee, K.M., Rogez, G.: MonoNHR: Monocular neural human renderer. In: 3DV (2022)
- 9. Choutas, V., Pavlakos, G., Bolkart, T., Tzionas, D., Black, M.J.: Monocular expressive body regression through body-driven attention. In: ECCV (2020)
- 10. Contributors, M.: Openmmlab pose estimation toolbox and benchmark. https: //github.com/open-mmlab/mmpose (2020)
- 11. Feng, Y., Choutas, V., Bolkart, T., Tzionas, D., Black, M.J.: Collaborative regression of expressive bodies using moderation. In: 3DV (2021)
- 12. Feng, Y., Feng, H., Black, M.J., Bolkart, T.: Learning an animatable detailed 3D face model from in-the-wild images. ACM TOG (2021)
- 13. Guédon, A., Lepetit, V.: SuGaR: Surface-aligned gaussian splatting for efficient 3D mesh reconstruction and high-quality mesh rendering. In: CVPR (2024)
- 14. Güler, R.A., Neverova, N., Kokkinos, I.: DensePose: Dense human pose estimation in the wild. In: CVPR (2018)
- 15. Guo, C., Jiang, T., Chen, X., Song, J., Hilliges, O.: Vid2Avatar: 3D avatar reconstruction from videos in the wild via self-supervised scene decomposition. In: CVPR (2023)
- 16. He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask R-CNN. In: ICCV (2017)
- 17. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. In: NeurIPS

(2020)

- 18. Hu, L., Zhang, H., Zhang, Y., Zhou, B., Liu, B., Zhang, S., Nie, L.: GaussianAvatar: Towards realistic human avatar modeling from a single video via animatable 3D gaussians. arXiv preprint arXiv:2312.02134 (2023)
- 19. Jiang, T., Chen, X., Song, J., Hilliges, O.: InstantAvatar: Learning avatars from monocular video in 60 seconds. In: CVPR (2023)
- 20. Jiang, W., Yi, K.M., Samei, G., Tuzel, O., Ranjan, A.: NeuMan: Neural human radiance field from a single video. In: ECCV (2022)
- 21. Joo, H., Simon, T., Sheikh, Y.: Total Capture: A 3D deformation model for tracking faces, hands, and bodies. In: CVPR (2018)

- 22. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3D gaussian splatting for real-time radiance field rendering. ACM TOG (2023)
- 23. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: ICLR

(2014)

- 24. Kocabas, M., Chang, J.H.R., Gabriel, J., Tuzel, O., Ranjan, A.: HUGS: Human gaussian splats. arXiv preprint arXiv:2311.17910 (2023)
- 25. Kwon, Y., Kim, D., Ceylan, D., Fuchs, H.: Neural Human Performer: Learning generalizable radiance fields for human performance rendering. NeurIPS (2021)
- 26. Li, J., Bian, S., Xu, C., Chen, Z., Yang, L., Lu, C.: HybrIK-X: Hybrid analytical-neural inverse kinematics for whole-body mesh recovery. arXiv preprint arXiv:2304.05690 (2023)
- 27. Li, T., Bolkart, T., Black, M.J., Li, H., Romero, J.: Learning a model of facial shape and expression from 4D scans. ACM TOG (2017)
- 28. Lin, J., Zeng, A., Wang, H., Zhang, L., Li, Y.: One-stage 3D whole-body mesh recovery with component aware transformer. In: CVPR (2023)
- 29. Liu, S., Li, T., Chen, W., Li, H.: Soft Rasterizer: A differentiable renderer for image-based 3D reasoning. In: ICCV (2019)
- 30. Liu, X., Wu, C., Liu, X., Liu, J., Wu, J., Zhao, C., Feng, H., Ding, E., Wang, J.: GEA: Reconstructing expressive 3D gaussian avatar from monocular video. arXiv preprint arXiv:2402.16607 (2024)
- 31. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: NeRF: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM (2021)
- 32. Moon, G., Choi, H., Lee, K.M.: Accurate 3D hand pose estimation for whole-body 3D human mesh estimation. In: CVPRW (2022)
- 33. Moon, G., Shiratori, T., Lee, K.M.: DeepHandMesh: A weakly-supervised deep encoder-decoder framework for high-fidelity hand mesh modeling. In: ECCV (2020)
- 34. Moon, G., Xu, W., Joshi, R., Wu, C., Shiratori, T.: Authentic hand avatar from a phone scan via universal hand model. In: CVPR (2024)
- 35. Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., Lerer, A.: Automatic differentiation in pytorch (2017)
- 36. Patel, P., Huang, C.H.P., Tesch, J., Hoffmann, D.T., Tripathi, S., Black, M.J.: AGORA: Avatars in geography optimized for regression analysis. In: CVPR (2021)
- 37. Pavlakos, G., Choutas, V., Ghorbani, N., Bolkart, T., Osman, A.A., Tzionas, D., Black, M.J.: Expressive body capture: 3D hands, face, and body from a single image. In: CVPR (2019)
- 38. Peng, S., Dong, J., Wang, Q., Zhang, S., Shuai, Q., Zhou, X., Bao, H.: Animatable neural radiance fields for modeling dynamic human bodies. In: ICCV (2021)
- 39. Peng, S., Zhang, Y., Xu, Y., Wang, Q., Shuai, Q., Bao, H., Zhou, X.: Neural Body: Implicit neural representations with structured latent codes for novel view synthesis of dynamic humans. In: CVPR (2021)
- 40. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: DreamFusion: Text-to-3D using 2D diffusion. In: ICLR (2023)
- 41. Qian, Z., Wang, S., Mihajlovic, M., Geiger, A., Tang, S.: 3DGS-Avatar: Animatable avatars via deformable 3D gaussian splatting. In: CVPR (2024)
- 42. Ravi, N., Reizenstein, J., Novotny, D., Gordon, T., Lo, W.Y., Johnson, J., Gkioxari, G.: Accelerating 3D deep learning with pytorch3D. arXiv preprint arXiv:2007.08501 (2020)
- 43. Remelli, E., Bagautdinov, T., Saito, S., Wu, C., Simon, T., Wei, S.E., Guo, K., Cao, Z., Prada, F., Saragih, J., et al.: Drivable volumetric avatars using texel-aligned features. In: ACM SIGGRAPH Conference Proceedings (2022)

- 44. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR (2022)
- 45. Rong, Y., Shiratori, T., Joo, H.: Frankmocap: A monocular 3D whole-body pose estimation system via regression and integration. In: ICCVW (2021)
- 46. Saito, S., Huang, Z., Natsume, R., Morishima, S., Kanazawa, A., Li, H.: PIFu: Pixel-aligned implicit function for high-resolution clothed human digitization. In: ICCV (2019)
- 47. Shen, K., Guo, C., Kaufmann, M., Zarate, J.J., Valentin, J., Song, J., Hilliges, O.: X-Avatar: Expressive human avatars. In: CVPR (2023)
- 48. Weng, C.Y., Curless, B., Srinivasan, P.P., Barron, J.T., Kemelmacher-Shlizerman,

I.: HumanNeRF: Free-viewpoint rendering of moving people from monocular video. In: CVPR (2022)

- 49. Wu, Y., He, K.: Group normalization. In: ECCV (2018)
- 50. Xu, H., Bazavan, E.G., Zanfir, A., Freeman, W.T., Sukthankar, R., Sminchisescu, C.: GHUM & GHUML: Generative 3D human shape and articulated pose models. In: CVPR (2020)
- 51. Xu, Z., Zhang, J., Liew, J.H., Yan, H., Liu, J.W., Zhang, C., Feng, J., Shou, M.Z.: MagicAnimate: Temporally consistent human image animation using diffusion model. arXiv preprint arXiv:2311.16498 (2023)
- 52. Zhang, H., Tian, Y., Zhang, Y., Li, M., An, L., Sun, Z., Liu, Y.: PyMAF-X: Towards well-aligned full-body model regression from monocular images. TPAMI (2023)
- 53. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: CVPR (2018)

