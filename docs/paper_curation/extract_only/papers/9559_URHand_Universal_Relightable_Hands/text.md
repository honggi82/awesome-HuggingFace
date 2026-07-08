## URHand: Universal Relightable Hands

Zhaoxi Chen1,2∗ Gyeongsik Moon1 Kaiwen Guo1 Chen Cao1 Stanislav Pidhorskyi1 Tomas Simon1 Rohan Joshi1 Yuan Dong1 Yichen Xu1 Bernardo Pires1 He Wen1 Lucas Evans1 Bo Peng1 Julia Buffalini1 Autumn Trimble1 Kevyn McPhail1 Melissa Schoeller1 Shoou-I Yu1 Javier Romero1 Michael Zollh¨ofer1 Yaser Sheikh1 Ziwei Liu2 Shunsuke Saito1

[Figure 1]

[Figure 2]

1Codec Avatars Lab, Meta 2Nanyang Technological University https://frozenburning.github.io/projects/urhand

# arXiv:2401.05334v1[cs.CV]10Jan2024

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

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

High-fidelity Universal Prior

Quick Personalization from a Phone Scan

Light-stage Data

Generalize to Novel Views, Poses, Identities, and Illuminations

Figure 1. URHand (a.k.a. Your Hand). Our model is a high-fidelity Universal prior for Relightable Hands built upon light-stage data. It generalizes to novel viewpoints, poses, identities, and illuminations, which enables quick personalization from a phone scan.

### Abstract

data while generalizing to real-time rendering under arbitrary continuous illuminations across diverse identities. In addition, we introduce the joint learning of a physically based model and our neural relighting model, which further improves fidelity and generalization. Extensive experiments show that our approach achieves superior performance over existing methods in terms of both quality and generalizability. We also demonstrate quick personalization of URHand from a short phone scan of an unseen identity.

Existing photorealistic relightable hand models require extensive identity-specific observations in different views, poses, and illuminations, and face challenges in generalizing to natural illuminations and novel identities. To bridge this gap, we present URHand, the first universal relightable hand model that generalizes across viewpoints, poses, illuminations, and identities. Our model allows few-shot personalization using images captured with a mobile phone, and is ready to be photorealistically rendered under novel illuminations. To simplify the personalization process while retaining photorealism, we build a powerful universal relightable prior based on neural relighting from multi-view images of hands captured in a light stage with hundreds of identities. The key challenge is scaling the cross-identity training while maintaining personalized fidelity and sharp details without compromising generalization under natural illuminations. To this end, we propose a spatially varying linear lighting model as the neural renderer that takes physics-inspired shading as input feature. By removing non-linear activations and bias, our specifically designed lighting model explicitly keeps the linearity of light transport. This enables single-stage training from light-stage

### 1. Introduction

We engage our hands for various tasks throughout the day, and they consistently remain within our field of view. This constant visibility of our hands makes them one of the most frequently seen parts of our body, playing a central role in self-embodiment. To seamlessly reproduce this experience for games or social telepresence, an ideal hand representation in a digital medium is photorealistic, personalized, and importantly, relightable for coherent appearance in any environment. Our objective is to enable the quick creation of such a hand model for any individual given lightweight input such as a phone scan, without going through an expensive capture process in a production studio (Figure 1).

∗This work was done during an internship at Meta

Approaches to build a photorealistic relightable hand

Corresponding authors

model can be broadly categorized into one of two philosophies. On the one hand, physically based rendering models [25, 59] can generalize to various illuminations through costly offline path-tracing, but typically lack photorealism under a real-time constraint. Additionally, accurately estimating material parameters remains a non-trivial challenge from unconstrained inputs and the quality is often bounded by the expressiveness of the physical models. On the other hand, neural relighting [2, 14] recently achieves impressive photorealism in real-time by directly inferring the outgoing radiance from illumination conditions. However, for generalization to natural illuminations to be possible these approaches require expensive data augmentation with teacherstudent distillation, where the student model learns to match with offline renderings produced by the teacher model under natural illuminations. Most importantly, cross-identity generalization remains an open problem in both camps.

In this work, we propose URHand, the first Universal Relightable Hand model that generalizes across viewpoints, motions, illuminations, and identities. To achieve the best trade-off between generalization and fidelity, our work exploits both physically based rendering and datadriven appearance modeling from neural relighting. More specifically, we incorporate known physics, such as the linearity of light transport [8] and surface reflections in the inductive bias of the neural relighting framework. We modulate non-linear layers conditioned by pose and identity with linear layers conditioned by spatially varying physically based shading [3]. This explicitly ensures linearity between input lighting features and output radiance. Thus, it enables environment map relighting without an expensive two-stage teacher-student distillation process commonly used in existing models [2, 14]. Our single-stage training enabled by linearity preservation makes cross-identity training more scalable with better generalization to novel illuminations.

Furthermore, we observe that the quality of the input shading features directly influences both generalization and fidelity of the final neural relighting outputs. Inspired by recent inverse rendering techniques [5, 67, 70], we introduce an additional physical branch that estimates the material parameters and high-resolution geometry via inverse rendering, from which we produce the input lighting features to the neural branch. The physical branch prevents the neural branch from overfitting by reducing hallucinations, and the neural branch compensates for the complex global light transport effects, such as subsurface scattering, that cannot be well captured by the physical branch. In addition, the proposed physics-based refinement improves the accuracy of the tracking geometry with fine details such as wrinkles. Combining it with our novel lighting-aware adversarial loss, our method achieves highly detailed relighting with various illuminations under any pose for novel identities.

We run an extensive ablation study as well as compar-

isons with baseline methods. The experiments demonstrate the efficacy of our hybrid neural-physical relighting method by outperforming other methods quantitatively and qualitatively. We also demonstrate the quick personalization of URHand from a phone scan, and relighting with arbitrary natural illuminations. In summary, our contributions are:

- • The first method to learn a universal relightable hand model that generalizes to novel views, poses, illuminations, and identities.
- • A spatially varying linear lighting model that generalizes to continuous illuminations without expensive distillation, enabling high-fidelity neural rendering and scalable training with multiple identities.
- • A hybrid neural-physical relighting framework that leverages the best of both approaches to achieve high fidelity and generalization at the same time.
- • The quick personalization of our universal prior to create a photorealistic and relightable hand from a phone scan.

- 2. Related Work
- 3D Hand Modeling. Human hand modeling is an active research field within vision and graphics. Early works focus more on 3D geometry and representation including mixture of 3D Gaussians [49, 50], sphere meshes [54], and triangular meshes [1, 7, 44]. These parametric hand models facilitate 3D hand pose estimation from 2D observations [30, 31, 34–36]. Recent works also incorporate physical priors [25, 32, 48, 55, 71] to model more accurate non-rigid deformation and articulation of the hand geometry. The recent advances of neural fields [53, 60] also enable the learning of personalized articulated models [19]. Beyond geometry modeling, achieving a lifelike appearance for hands [41] is paramount for realistic rendering and animation. Handy [40] learns a texture space by using generative adversarial networks for better photorealism and generalization. More recently, some methods [4, 6, 37] showcase the modeling of animatable hands from monocular/multiview captures. However, the appearance models of these approaches simply bake the captured illumination and cannot be rendered under novel illuminations. NIMBLE [25] builds PCA reflectance maps including diffuse, normal, and specular maps from light-stage data. DART [9] also supports accessories. However, physically based materials are expensive to render with global illumination which often limits their rendering fidelity with a real-time constraint. RelightableHands [14] enables the photorealistic relighting of hands in real-time using a neural appearance model. However, the method only supports person-specific modeling and generalization to unseen identity is not possible. In contrast, our approach generalizes across poses, illuminations, and identities, supporting relightable hand modeling of unseen identities from a phone scan.

Image-based Relighting. Given the linearity of light trans-

port, Debevec et al. [8] propose to render human faces under novel illuminations by linear combinations of sampled reflectance fields from one-light-at-a-time (OLAT) captures. A series of follow-up work [29, 51, 58, 62] enable dynamic relighting and capture through learning-based approaches. Meanwhile, another line of work aims at intrinsic decomposition, enabling physically based rendering with disentangled geometry and reflectance in the image space [11, 12, 17, 22, 38, 47]. Recent advances in neural rendering [15, 39, 65] learn to match with ground truth using non-linear neural networks given the lighting features from a simple physical shading model. Yet, image-based relighting suffers from 3D inconsistency and flickering due to the lack of an underlying 3D representation.

Model-based Relighting. To address the lack of 3D consistency in image-based relighting, one can leverage a shared 2D parameterization [63, 69] or template models [2, 5, 14, 20] for model-based relighting. Most modelbased relighting approaches [5, 13, 16, 42, 56] rely on the intrinsic decomposition of geometry and reflectance followed by a physically based appearance model. While they achieve generalization to novel conditions, the fidelity is typically limited due to the lack of expressiveness in the underlying parametric BRDFs. On the other hand, methods with neural renderers support complex global illumination effects learned from captured data under point lights. However, generalization to continuous environments requires the expensive teacher-student training framework [2, 14], which is difficult to scale to cross-identity training. On the contrary, our spatially varying linear network achieves generalization to any type of illumination without additional training. Concurrently, Yang et al. [64] propose a linear lighting model for face relighting that eliminates the need of teacher-student distillation. While the motivation is similar, we observe that their holistic light representation does not generalize well for hands due to drastic visibility change by articulation (see Sec. 5.3 for analysis).

### 3. Preliminary

Data Acquisition. We use a multiview capture system consisting of 150 cameras and 350 LED lights to capture dynamic hands with time-multiplexed illuminations by interleaving fully lit (all lights on) and partially lit every other frame. Instead of OLAT, our partially lit frames use L = 5 grouped lights to increase brightness and reduce motion blur as in [2, 14, 23]. Images are captured in the resolution of 4096×2668 at 90 fps. We first reconstruct per-frame 3D meshes using [10] and detect 3D hand keypoints using [24] followed by triangulation from fully lit frames.

For partially lit frames, we leverage spherical linear interpolation over the pose parameter of adjacent fully lit frames to obtain the hand pose of partially lit frames. Our dataset contains 93 different identities in diverse hand mo-

tions with an average of 42000 frames for each identity.

Linearity of Light Transport. To render subjects in arbitrary illuminations, natural illumination is treated as a linear combination of distant point lights given the linearity of light transport [8]. Specifically, given the appearance value Ci based on the i-th point light, the final color C is computed as a linear combination of all light sources, i.e. C = Li=1 biCi, where L denotes the number of lights, and bi is the intensity of each light. Given a light transport function f(b) = C, we define it as linear w.r.t. b such that:

L

L

L

bi. (1)

f(b) = f(

bi) =

f(bi), b =

i=1

i=1

i

Hand Geometry Modeling. Similar to [32], our 3D hand representation is based on a mesh template with vertex offsets predicted by a neural network. The 3D hand can be driven by linear blend skinning (LBS) and represent identity- and pose-specific deformations.

Specifically, we design an autoencoder to obtain accurate hand tracking and geometry. The encoder learns to predict identity-dependent latent codes and 3D hand poses from the input fully lit frames. Given an articulated generic mesh template M¯ = {V,F,U,θ} with vertices V ∈ RnV×3, faces F ∈ RnF×3, texture coordinates U ∈ RnV×2, and 3D pose θ ∈ R60×3 in Euler angles and latent codes z, the decoder learns to predict the 3D offset of all vertices as δV ∈ RnV×3. We use nV = 15930 and nF = 32340. The tracked hand mesh will be represented as M = {V + δV,F,U,θ}. Please refer to the supplementary material for more details.

### 4. Universal Relightable Hands

Our goal is to build a universal relightable shape and appearance model for human hands that can be rendered for any identity under arbitrary illumination in real-time. To this end, we learn a relightable appearance model from crossidentity light-stage captures based on grouped point lights.

In this section, we will introduce our learning framework, URHand, which learns to relight target hands in different poses and views. The core of our model is a spatially varying linear lighting model that preserves the linearity of light transport, which enables the generalization to arbitrary illuminations by training with monochrome group lights only. Our model consists of two parallel rendering branches, physical and neural. The physical branch (Sec. 4.1) focuses on refining geometry and providing accurate shading features as an illumination proxy for the neural branch. The neural branch (Sec. 4.2) learns the final appearance of hands with global illumination. These two branches are trained jointly in an end-to-end manner with our tailored loss functions (Sec. 4.3). Finally, we use this universal prior to quickly personalize a relightable hand model from fewshot observations (Sec. 5.4).

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Coarse Mesh ℳ

Refined Normal 𝑛 Diffuse 𝐂 Specular 𝐂

Normal 𝑛

Light View

Eq. (3)

Physical BRDF ℱ

[Figure 30]

ℒ

𝐂

[Figure 31]

𝑠𝑔(⋅)

Hand Pose 𝜃

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Encoder ℱ

Displacement 𝛿𝑑 Roughness 𝛽

Mean Texture 𝒯

Linear Net ℱ

ℱ

U-Net

###### GT

Decoder ℱ

[Figure 36]

[Figure 37]

[Figure 38]

Gain 𝑔 Bias 𝑏

ℒ

Physical Branch Neural Branch

Eq. (5)

Eq. (6)

Non-Linear Net ℱ

𝐂

- Figure 2. Overview of URHand. Our model takes as input a mean texture T , hand pose θ, and a coarse mesh M for each identity. The physical branch (Sec. 4.1) focuses on geometry refinement and providing accurate shading features for the neural branch (Sec. 4.2). The core of the neural branch is the linear lighting model which takes as input the physics-inspired shading features from the physical branch. The neural branch learns to predict the gain and bias map over the mean texture. We leverage a differentiable rasterizer for rendering and minimize the loss of both branches against ground truth images (Sec. 4.3). The sg(·) denotes the stop-gradient operation.

#### 4.1. Physically based Geometry Refinement

The physical branch employs online physically based rendering to render images using a parametric BRDF. We optimize the material parameters of the BRDF via inverse rendering. The goal of the physical branch is two-fold: 1) further refine the initial hand geometry for better alignment, and 2) provide generalizable lighting features that best approximate the specular reflection and diffuse shading to prevent overfitting in the neural relighting.

Cpb(xˆ,d,L) = Li(L,xˆ,ωi)Fpb(xˆ,ωi,d,β)(ωi·nˆ)dωi,

(3) where Li(xˆ,ωi) is the incident light from direction ωi. The physically rendered texture map Cpb can be decomposed into physically based shading feature, i.e. Fpb(L) = {Cdpb,Cspb}, according to the equation Cpb = Cdpb ⊙ T + Cspb, where ⊙ is the element-wise multiplication, and T

is the mean texture approximating albedo. Note that, this computation is directly performed in the UV space.

We illustrate the physical branch in Fig. 2. The physical branch estimates a parametric Disney BRDF [3] Fpb. We use a 2D U-Net FG to infer a displacement map δd ∈ R1024×1024×3 and a roughness map β ∈ R1024×1024 in UV space. Instead of predicting a normal map directly, we use the predicted displacement map to update the normal map on top of the unwrapped coarse mesh in the UV space, which makes it easier to infer high-frequency geometric details on a smooth base surface [66]. To support cross-identity modeling with pose-dependent geometry and appearance change, FG takes as input the hand pose θ and the unwrapped mean texture T ∈ R1024×1024×3 for each identity, where the hand pose is concatenated to the bottleneck representation of the U-Net.

#### 4.2. Linear Lighting Model

The rendering based on a parametric BRDF generalizes to novel illuminations, but they lack correct global light transport effects such as subsurface scattering. To enable relighting with global illumination in real-time, we introduce a neural renderer FR. Given a target illumination, the outgoing radiance is computed as: C = FR(M,T ,Fpb(L),θ).

Our key insight is that removing the non-linear activation layers and bias in a convolutional neural network preserves the linearity with respect to the input features. Since our input feature is the physically based shading Fpb(L), our network satisfies the following:

Specifically, the refined surface xˆ is obtained by adding the displacement δd to the base surface x derived from the coarse mesh M along the direction of the normal n:

L

L

Fpb(Li), θ). (4)

FR(M, T , Fpb(Li), θ) = FR(M, T ,

i=1

i=1

xˆ = x + δd · n, (2)

Since the physically based rendering in Eq. 3 is energy preserving, the network holds the linearity w.r.t. the input illumination. Therefore, our network can produce accurate relighting with continuous environment maps by training only with discrete point lights without additional distillation.

where positional map xˆ is then used to obtain the refined normal nˆ in the UV space for physically based rendering. In our implementation, we apply a sigmoid activation followed by a scaling factor of 3 in order to constrain the range of displacement to ±3mm.

We illustrate the architecture of FR = {Fl,Fnl} in Fig. 2. It consists of a linear (Fl) and a non-linear (Fnl) branch, where the linear branch consists of an encoder Fl−enc and a decoder Fl−dec with physically based shading

The refined normal nˆ and roughness β are fed into Fpb that considers only the first bounce. Given an illumination L, the final color Cpb from camera view d is computed as:

features Fpb(L) as input. The pose- and identity-dependent features derived from the mean texture T and pose θ are fed into a non-linear branch. We fuse the linear and non-linear feature maps in the decoder of the linear branch as follows:

1 √2 · ConvT(Flj−enc + Flj−dec) ⊙ Fnlj , (5)

Flj−+1dec =

where j is the index of the layer, ConvT is the transposed convolutional layer without bias. This fusion mechanism keeps the linearity of the output w.r.t. the input lighting features while incorporating non-linearity w.r.t. identity and pose conditions. Instead of predicting the final texture, our neural renderer predicts the texel-aligned gain map g and bias map b, which contribute to the final texture as follows:

C = g ⊙ T + b · σT , (6) where σT = 64 is the standard deviation of textures.

Important to note that since our input feature Fpb is spatially varying in a texel-aligned manner similar to [14, 39], we can accurately incorporate shadow information for better generalization with diverse poses. While the concurrent work [64] also proposes a linear lighting model using a holistic illumination representation by simply reshaping environment maps, we observe that the holistic illumination representation cannot generalize for hands due to the infinite shadow variations caused by articulation. The same observation is also reported in person-specific relightable hand modeling [14]. Please refer to Sec. 5 for the analysis.

#### 4.3. Training Objectives

Our model is trained on multiview partially lit images in different identities and poses. The training objective Ltotal consists of three parts: reconstruction loss Limg, lightingaware adversarial loss LGAN, and L1 regularization Lreg:

Ltotal = λimgLimg + λGANLGAN + λregLreg, (7) where λ∗ are corresponding loss weights.

Reconstruction Loss. We leverage a sum of L1 loss LMAE and perceptual loss Leff based on the EfficientNet [52] backbone, i.e. Limg = LMAE + Leff. Both the renderings from the neural and the physical branch are supervised by the reconstruction loss against the ground truth images.

Lighting-aware Adversarial Loss. To improve the visual quality, we propose to use an adversarial loss on top of the reconstruction loss. We found that a naive imageconditioned discriminator performs poorly due to the significant appearance change in partially lit frames. To address this, we leverage a lighting-aware discriminator on multiple scales of renderings. Specifically, the discriminator FD is conditioned on the diffuse and specular feature, {A,S}, which prompts the network to discriminate real and fake images given the illumination information. These lighting features are based on simple Phong reflectance [14] to ensure

a consistent lighting prompt during training. We choose a hinge loss [26] operated on multi-resolution discriminated patches as the adversarial target:

LGAN = log FD(I|A,S) + log[1 − FD(Iˆ|A,S)], (8) where I is the ground truth, and Iˆis the rendered image.

L1 Regularization on Linear Model. We also discovered that without regularization, a linear lighting model often produces noticeable flickering. As a linear convolutional network is capacity-limited, it tends to have high variance in intermediate features, resulting in poor generalization to novel illuminations. Thus, we penalize the L1 norm of the intermediate features from all layers in the linear branch:

Lreg =

N

||Flj−enc||1. (9)

j=1

Implementation Detail. The optimizable modules are {FG,FR,FD}. We use the Adam [21] optimizer and set the loss weights as λimg = 1.0, λGAN = 0.01, and λreg = 0.01, respectively. We train the model for 2M iterations distributed on 8 A100 GPUs with a batch size of 24 in total. The initial learning rate is 1 × 10−4 which decays to 3 × 10−5 with a multistep learning rate scheduler. We describe the detailed network architecture and more training details in the supplementary.

Runtime Analysis. Our proposed linear lighting model is not only scalable and generalizable but also efficient for real-time rendering. Our model achieves 38 FPS (25.7 ms) given grouped lights as input and 31 FPS (31.9 ms) given environmental maps as input on an NVIDIA A100 GPU.

### 5. Experiments

#### 5.1. Evaluation Protocols

To quantitatively evaluate the fidelity of each method, we use Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure [57] (SSIM), and Learned Perceptual Image Patch Similarity [68] (LPIPS) as metrics. To solely evaluate the quality of rendered hands, we only take the foreground according to the mask obtained from the refined hand geometry. We exclude several segments from training data to evaluate the generalization of our model to novel poses. All metrics are evaluated on 1,000 images randomly sampled from those held-out segments.

For fair comparisons with existing works, we train and evaluate all methods in the single-identity setting except mentioned. For ablation studies, we train with ten identities and test on 1) unseen segments from a train subject, 2) unseen segments from an unseen subject, and 3) an unseen illumination from a train subject, which orthogonally evaluate the generalization of our model to novel poses, identities, and illuminations.

- Table 1. Quantitative comparisons on sequences with grouped lights. We evaluate our method for both per-subject optimization and novel identity generalization against the state-of-the-art methods in model-based hand relighting. †Methods are evaluated on the training identity with unseen segments. ∗Methods are evaluated on unseen identity during training. The top three techniques are highlighted in red, orange, and yellow, respectively.

Subject 1 Subject 2 Subject 3 PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ †RelightableHands [14] 25.97 0.9301 0.1425 25.92 0.9372 0.1426 27.16 0.9419 0.1280

Method

†Ours (Physical only) 23.44 0.9062 0.1708 23.25 0.9154 0.1715 24.90 0.9216 0.1510 †Ours (Full model) 27.77 0.9400 0.1204 26.36 0.9384 0.1344 27.75 0.9445 0.1226

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

∗Handy [40] + Phong 16.65 0.8235 0.3026 16.39 0.8328 0.3019 17.70 0.8311 0.2872 ∗Handy [40] + GGX 16.60 0.8212 0.3125 16.33 0.8300 0.3107 17.58 0.8288 0.2978

∗Ours (Full model) 26.94 0.9271 0.1335 26.24 0.9368 0.1341 27.58 0.9436 0.1197

a) Single Identity Evaluation b) Novel Identity Generalization

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

RelightableHands Ours (Physical Only) Ours Handy + Phong Handy + GGX Ours Ground Truth

- Figure 3. Qualitative comparisons on sequences with grouped lights. We evaluate our method for both per-subject optimization and novel identity generalization against comparison methods. a) All methods are evaluated on the training identity with unseen segments. b) Methods are evaluated on unseen identity during training.

#### 5.2. Comparisons

We compare our approach with the state-of-the-art 3D hand relighting and reconstruction methods. RelightableHands [14] reconstructs relightable appearance via peridentity optimization. Handy [40] predicts the UV texture of the hand from in-the-wild images. For fair comparisons, we apply physically based renderers on top of the predicted texture to enable relightable appearances. We also compare with our physical branch using the same loss functions. This provides a fair comparison with physically based relighting. For this reason, we omit the comparison with other existing physically based relightable hand methods such as HARP [20] and NIMBLE [25]. Please refer to the supplementary for our detailed implementations of these methods.

We present quantitative results in Table 1. For peridentity training, our method significantly outperforms baseline methods on all metrics, which highlights the effectiveness of our key designs. As shown in Figure 3, our method is able to reproduce detailed geometry, e.g. wrinkles and nails, together with high-fidelity details like specularities and shadows. Handy [40] fails to reproduce the base texture of the target hand due to their data-driven latent texture space. Its combinations of simple physical renderers

lead to artifacts of the relit results. RelightableHands [14] can reproduce correct shading and appearance. However, the quality of geometric details and specularity is not on par with the proposed method, which demonstrates the effectiveness of our hybrid neural-physical approach. Furthermore, the generalizability of our method is showcased in the right side of Figure 3, where the shown test subjects are withheld from the training set. Although the quality slightly degrades compared with per-identity optimization, it still outperforms all other baselines by a large margin.

#### 5.3. Ablation Studies

Ablation on Linear Lighting Model. We investigate different designs of the linear lighting model. We compare our spatially varying linear lighting model with three alternatives: 1) Non-linear model, where all non-linear activations are turned on; 2) Non-linear model with a linearity consistency loss to constrain the linearity (Eq. 1) between output and input of the network; 3) MLP-based linear model proposed in [64], where the lighting is represented as environment map and flattened to a 1-D vector.

Table 4 shows that the proposed spatially varying linear lighting model achieves the best performance on both

- Table 2. Ablation studies of the linear lighting model on sequences with grouped lights. The “Non-linear” denotes the model that does not satisfy the linearity of light transport while the “Linearity Consistency” denotes the non-linear model with regularizations to constrain the linearity between output and input. Moreover, in contrast to MLP-based linear model [64], our model is a spatially varying linear model. The top three techniques are highlighted in red, orange, and yellow, respectively.

Method

Trained Subject Unseen Subject Fully lit

PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ Non-linear 25.79 0.9283 0.1364 23.89 0.9145 0.1523 17.50 0.9070 0.1717

Linearity Consistency 24.77 0.9122 0.1560 23.11 0.9124 0.1537 19.75 0.9013 0.1619 MLP-based Linear [64] 22.19 0.8727 0.1885 21.56 0.8787 0.1823 11.56 0.8644 0.2259 Ours (Linear Model) 26.01 0.9270 0.1336 24.70 0.9121 0.1520 20.61 0.9153 0.1576

[Figure 53]

Non-linear Linear Consistency MLP-based Linear Ours

[Figure 54]

[Figure 55]

| |
|---|

[Figure 56]

| |
|---|

[Figure 57]

[Figure 58]

| |
|---|

[Figure 59]

| |
|---|

[Figure 60]

[Figure 61]

| |
|---|

[Figure 62]

| |
|---|

[Figure 63]

| |
|---|

[Figure 64]

| |
|---|

Figure 4. Ablation study on the design of linear lighting model. Our spatially varying linear lighting model produces realistic renderings, while the baseline methods fail to correctly model shadows or tend to be over smooth.

- Table 3. Ablation studies of the lighting features on sequences with grouped lights. The top three techniques are highlighted in red, orange, and yellow, respectively.

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

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

###### Ours

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

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Phongw/oVisibilityw/oRefinerw/oSpecular

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

| |
|---|

| |
|---|

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

N/A

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

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

Trained Subject Unseen Subject PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Method

Phong 25.53 0.9257 0.1382 23.87 0.9148 0.1523 w/o Specular 24.97 0.9160 0.1498 23.20 0.9075 0.1607 w/o Visibility 25.44 0.9228 0.1426 23.75 0.9133 0.1533

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

w/o Refiner 24.82 0.9144 0.1437 22.80 0.8993 0.1620 Full Model 26.01 0.9270 0.1336 24.70 0.9121 0.1520

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

trained and unseen identities. Moreover, we also quantitatively evaluate on fully lit frames to validate the generalization to novel illumination. The non-linear baseline fails to generalize, which shows the importance of linearity in our network. The MLP-based linear model fails to correctly model the light transport due to the lack of pose-aware visibility change. Figure 4 shows that the proposed lighting model produces the most realistic renderings, while the baseline methods struggle with correctly modeling shadows and detailed shading effects.

GT Render Normal Diffuse Specular

Figure 5. Ablation studies on the impact of lighting features and geometry refinement. Notably, our full model can produce fine-grained geometry like wrinkles and nails as well as specular highlights (e.g. little finger).

relighting. In fact, the neural relighting based on Phong specular features [14, 39] fails to produce accurate specular highlights. The visibility is also important for novel pose generalization. Compared with the model trained without visibility, our full model reproduces better soft shadows as well as detailed geometry. These observations are also supported by our quantitative evaluation presented in Table 3.

Ablation on Different Lighting Features. We also evaluate the effectiveness of our lighting feature representation, i.e. the diffuse Cdpb and specular feature Cspb. The contribution in the lighting features can be factored into 1) the underlying BRDF, 2) the visibility, and 3) the geometry. As shown in Figure 5, our model reproduces specular highlights due to the more accurate specular feature from the optimized Disney BRDF. This indicates that the quality of the lighting feature determines the quality of the neural

Effectiveness of Geometry Refinement. Our proposed neural-physical rendering offers the ability to refine geometry. As shown in Figure 5, our full model can produce more

- Table 4. Ablation studies of the proposed training objectives on sequences with grouped lights. The top three techniques are highlighted in red, orange, and yellow, respectively.

Method

Trained Subject Unseen Subject PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓ w/o LGAN 25.95 0.9271 0.1392 23.70 0.9113 0.1542

w/o Light-aware LGAN 25.38 0.9215 0.1391 23.88 0.9142 0.1521 w/o L1 Reg Lreg 25.52 0.9242 0.1351 23.90 0.9149 0.1491 Full Model 26.01 0.9270 0.1336 24.70 0.9121 0.1520

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Ground Truth

w/o GAN Loss w/o Light Awareness w/o L1 Reg Ours

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

| |
|---|

[Figure 147]

| |
|---|

[Figure 148]

| |
|---|

[Figure 149]

| |
|---|

[Figure 150]

| |
|---|

[Figure 151]

| |
|---|

[Figure 152]

| |
|---|

[Figure 153]

| |
|---|

[Figure 154]

| |
|---|

[Figure 155]

[Figure 156]

| |
|---|

[Figure 157]

| |
|---|

[Figure 158]

| |
|---|

[Figure 159]

| |
|---|

[Figure 160]

| |
|---|

[Figure 161]

| |
|---|

[Figure 162]

| |
|---|

[Figure 163]

| |
|---|

[Figure 164]

Light Probe

Figure 6. Ablation studies on the proposed training objectives. The adversarial loss improves the overall quality while the light awareness of the discriminator is critical for correct shadows.

fine-grained details such as wrinkles and nails. The geometry refinement evidently prevents the neural renderer from hallucinating, leading to better generalization to novel identities and illuminations. The quantitative results in Table 3 also support this observation.

Effectiveness of Adversarial Loss. We evaluate the effectiveness of the proposed lighting-aware adversarial loss. Quantitative results in Table 4 suggest that this adversarial loss improves the overall quality. The lighting-aware discriminator further improves the fidelity. Figure 6 validates that the adversarial loss is critical for reproducing shadow and detailed geometry. It also enhances specular highlights. Effectiveness of L1 Regularization. We validate the effectiveness of the L1 regularization on the intermediate features of the linear lighting model. Table 4 and Figure 6 illustrate that, compared with the model without L1 regularization, our model achieves better modeling of the soft shadows and appearance. Its effectiveness is more evident in the temporal sequence with less flickering artifacts. Please refer to our supplemental video.

- 5.4. Quick Adaptation on Unseen Subjects

As shown in Figure 7, we demonstrate the quick personalization of URHand to get a relightable and animatable hand from a monocular iPhone video. To achieve this, we fit a template hand model (Sec. 3) to input images by optimizing pose parameters and the identity latent code based on foreground segmentation and 2D keypoints following a similar

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Monocular video from iPhone

…

Figure 7. Quick Personalization of URHand from iPhone captures. Given a casual phone scan, we fit hand geometry (Sec. 3), unwrap RGB images to get the mean texture, and feed into URHand. Our model instantly enables photorealistic relighting in any poses and illuminations without finetuning.

pipeline to [20]. Then we unwrap input RGB images based on the fitted geometry to obtain the mean texture T , and refine it to remove shadows [20]. Finally, we directly feed the mean texture and the coarse mesh into URHand to render it with any poses and environment maps. Note that while URHand can instantly create a personalized relightable hand model without any finetuning, the aforementioned preprocessing stage takes several hours.

### 6. Conclusion

We have introduced URHand, the first universal relightable hand model that generalizes across viewpoints, poses, illuminations, and identities. We show that scalable crossidentity training for high-fidelity relightable hands now is possible with our physics-inspired spatially varying linear lighting model and hybrid neural-physical learning framework. Our experiments indicate that URHand even generalizes beyond studio data by showing quick personalization from a phone scan.

Limitation and Future Works. Since we learn global light transport with far-field lighting, it does not guarantee correct light transport with near-field lighting. Nevertheless, our work achieves plausible near-field relighting similarly to [2, 14, 46, 61]. Currently the quick personalization requires the complete mean texture of a target hand. Thus, it does not work with a single image. One future work would be inpainting the texture from a single image to enable single-view relightable hand reconstruction. As our hand model is only driven by hand poses, it cannot capture appearance variations due to blood pressure or temperature changes. As recently demonstrated in [33], photorealistic relightable hands can be used to augment training data for image-based pose regression tasks. Using URHand to synthesize large-scale two-hand or hand-to-object interaction images with diverse identities is also fruitful.

### References

- [1] Luca Ballan, Aparna Taneja, J¨urgen Gall, Luc Van Gool, and Marc Pollefeys. Motion capture of hands in action using discriminative salient points. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VI 12, pages 640–653. Springer, 2012. 2
- [2] Sai Bi, Stephen Lombardi, Shunsuke Saito, Tomas Simon, Shih-En Wei, Kevyn Mcphail, Ravi Ramamoorthi, Yaser Sheikh, and Jason Saragih. Deep relightable appearance models for animatable faces. ACM Transactions on Graphics (TOG), 40(4):1–15, 2021. 2, 3, 8, 4
- [3] Brent Burley and Walt Disney Animation Studios. Physically-based shading at disney. In Acm Siggraph, pages 1–7. vol. 2012, 2012. 2, 4, 1
- [4] Xingyu Chen, Baoyuan Wang, and Heung-Yeung Shum. Hand avatar: Free-pose hand animation and rendering from monocular video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8683–8693, 2023. 2
- [5] Zhaoxi Chen and Ziwei Liu. Relighting4d: Neural relightable human from videos. In European Conference on Computer Vision, pages 606–623. Springer, 2022. 2, 3
- [6] Enric Corona, Tomas Hodan, Minh Vo, Francesc MorenoNoguer, Chris Sweeney, Richard Newcombe, and Lingni Ma. Lisa: Learning implicit shape and appearance of hands. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20533–20543, 2022. 2
- [7] Martin de La Gorce, David J Fleet, and Nikos Paragios. Model-based 3d hand pose estimation from monocular video. IEEE transactions on pattern analysis and machine intelligence, 33(9):1793–1805, 2011. 2
- [8] Paul Debevec, Tim Hawkins, Chris Tchou, Haarm-Pieter Duiker, Westley Sarokin, and Mark Sagar. Acquiring the reflectance field of a human face. In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pages 145–156, 2000. 2, 3
- [9] Daiheng Gao, Yuliang Xiu, Kailin Li, Lixin Yang, Feng Wang, Peng Zhang, Bang Zhang, Cewu Lu, and Ping Tan. Dart: Articulated hand model with diverse accessories and rich textures. Advances in Neural Information Processing Systems, 35:37055–37067, 2022. 2
- [10] Kaiwen Guo, Peter Lincoln, Philip Davidson, Jay Busch, Xueming Yu, Matt Whalen, Geoff Harvey, Sergio OrtsEscolano, Rohit Pandey, Jason Dourgarian, et al. The relightables: Volumetric performance capture of humans with realistic relighting. ACM Transactions on Graphics (ToG), 38(6):1–19, 2019. 3
- [11] Andrew Hou, Ze Zhang, Michel Sarkis, Ning Bi, Yiying Tong, and Xiaoming Liu. Towards high fidelity face relighting with realistic shadows. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14719–14728, 2021. 3
- [12] Andrew Hou, Michel Sarkis, Ning Bi, Yiying Tong, and Xiaoming Liu. Face relighting with geometrically consistent shadows. In Proceedings of the IEEE/CVF Conference

- on Computer Vision and Pattern Recognition, pages 4217– 4226, 2022. 3
- [13] Umar Iqbal, Akin Caliskan, Koki Nagano, Sameh Khamis, Pavlo Molchanov, and Jan Kautz. Rana: Relightable articulated neural avatars. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23142– 23153, 2023. 3
- [14] Shun Iwase, Shunsuke Saito, Tomas Simon, Stephen Lombardi, Bagautdinov Timur, Rohan Joshi, Fabian Prada, Takaaki Shiratori, Yaser Sheikh, and Jason Saragih. Relightablehands: Efficient neural relighting of articulated hand models. In CVPR, 2023. 2, 3, 5, 6, 7, 8, 4
- [15] Chaonan Ji, Tao Yu, Kaiwen Guo, Jingxin Liu, and Yebin Liu. Geometry-aware single-image full-body human relighting. In European Conference on Computer Vision, pages 388–405. Springer, 2022. 3
- [16] Haian Jin, Isabella Liu, Peijia Xu, Xiaoshuai Zhang, Songfang Han, Sai Bi, Xiaowei Zhou, Zexiang Xu, and Hao Su. Tensoir: Tensorial inverse rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 165–174, 2023. 3
- [17] Yoshihiro Kanamori and Yuki Endo. Relighting humans: occlusion-aware inverse rendering for full-body human images. arXiv preprint arXiv:1908.02714, 2019. 3
- [18] Tero Karras, Miika Aittala, Samuli Laine, Erik H¨ark¨onen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. In Proc. NeurIPS, 2021. 3
- [19] Korrawe Karunratanakul, Adrian Spurr, Zicong Fan, Otmar Hilliges, and Siyu Tang. A skeleton-driven neural occupancy representation for articulated hands. In 2021 International Conference on 3D Vision (3DV), pages 11–21. IEEE, 2021. 2
- [20] Korrawe Karunratanakul, Sergey Prokudin, Otmar Hilliges, and Siyu Tang. Harp: Personalized hand reconstruction from a monocular rgb video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12802–12813, 2023. 3, 6, 8
- [21] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014. 5, 3

- [22] Manuel Lagunas, Xin Sun, Jimei Yang, Ruben Villegas, Jianming Zhang, Zhixin Shu, Belen Masia, and Diego Gutierrez. Single-image full-body human relighting. arXiv preprint arXiv:2107.07259, 2021. 3
- [23] Junxuan Li, Shunsuke Saito, Tomas Simon, Stephen Lombardi, Hongdong Li, and Jason Saragih. Megane: Morphable eyeglass and avatar network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12769–12779, 2023. 3
- [24] Wenbo Li, Zhicheng Wang, Binyi Yin, Qixiang Peng, Yuming Du, Tianzi Xiao, Gang Yu, Hongtao Lu, Yichen Wei, and Jian Sun. Rethinking on multi-stage networks for human pose estimation. arXiv preprint arXiv:1901.00148, 2019. 3
- [25] Yuwei Li, Longwen Zhang, Zesong Qiu, Yingwenqi Jiang, Nianyi Li, Yuexin Ma, Yuyao Zhang, Lan Xu, and Jingyi Yu. Nimble: a non-rigid hand model with bones and muscles. ACM Transactions on Graphics (TOG), 41(4):1–16, 2022. 2, 6

- [26] Jae Hyun Lim and Jong Chul Ye. Geometric gan. arXiv preprint arXiv:1705.02894, 2017. 5
- [27] Shanchuan Lin, Linjie Yang, Imran Saleemi, and Soumyadip Sengupta. Robust high-resolution video matting with temporal guidance. In WACV, 2022. 2
- [28] Stephen Lombardi, Tomas Simon, Gabriel Schwartz, Michael Zollhoefer, Yaser Sheikh, and Jason Saragih. Mixture of volumetric primitives for efficient neural rendering. ACM Transactions on Graphics (ToG), 40(4):1–13, 2021. 3
- [29] Abhimitra Meka, Christian Haene, Rohit Pandey, Michael Zollh¨ofer, Sean Fanello, Graham Fyffe, Adarsh Kowdle, Xueming Yu, Jay Busch, Jason Dourgarian, et al. Deep reflectance fields: high-quality facial reflectance field inference from color gradient illumination. ACM Transactions on Graphics (TOG), 38(4):1–12, 2019. 3
- [30] Gyeongsik Moon and Kyoung Mu Lee. I2l-meshnet: Imageto-lixel prediction network for accurate 3d human pose and mesh estimation from a single rgb image. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VII 16, pages 752–768. Springer, 2020. 2
- [31] Gyeongsik Moon, Ju Yong Chang, and Kyoung Mu Lee. V2v-posenet: Voxel-to-voxel prediction network for accurate 3d hand and human pose estimation from a single depth map. In Proceedings of the IEEE conference on computer vision and pattern Recognition, pages 5079–5088, 2018. 2
- [32] Gyeongsik Moon, Takaaki Shiratori, and Kyoung Mu Lee. Deephandmesh: A weakly-supervised deep encoder-decoder framework for high-fidelity hand mesh modeling. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 440–455. Springer, 2020. 2, 3, 1
- [33] Gyeongsik Moon, Shunsuke Saito, Weipeng Xu, Rohan Joshi, Julia Buffalini, Harley Bellan, Nicholas Rosen, Jesse Richardson, Mallorie Mize, Philippe De Bree, et al. A dataset of relighted 3d interacting hands. arXiv preprint arXiv:2310.17768, 2023. 8
- [34] Franziska Mueller, Dushyant Mehta, Oleksandr Sotnychenko, Srinath Sridhar, Dan Casas, and Christian Theobalt. Real-time hand tracking under occlusion from an egocentric rgb-d sensor. In Proceedings of the IEEE International Conference on Computer Vision, pages 1154–1163, 2017. 2
- [35] Franziska Mueller, Florian Bernard, Oleksandr Sotnychenko, Dushyant Mehta, Srinath Sridhar, Dan Casas, and Christian Theobalt. Ganerated hands for real-time 3d hand tracking from monocular rgb. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 49–59, 2018.
- [36] Franziska Mueller, Micah Davis, Florian Bernard, Oleksandr Sotnychenko, Mickeal Verschoor, Miguel A Otaduy, Dan Casas, and Christian Theobalt. Real-time pose and shape reconstruction of two interacting hands with a single depth camera. ACM Transactions on Graphics (ToG), 38(4):1–13,

2019. 2

- [37] Akshay Mundra, Mallikarjun B R, Jiayi Wang, Marc Habermann, Christian Theobalt, and Mohamed Elgharib. Livehand: Real-time and photorealistic neural hand rendering,

2023. 2

- [38] Thomas Nestmeyer, Jean-Fran¸cois Lalonde, Iain Matthews, and Andreas Lehrmann. Learning physics-guided face relighting under directional light. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5124–5133, 2020. 3
- [39] Rohit Pandey, Sergio Orts Escolano, Chloe Legendre, Christian Haene, Sofien Bouaziz, Christoph Rhemann, Paul Debevec, and Sean Fanello. Total relighting: learning to relight portraits for background replacement. ACM Transactions on Graphics (TOG), 40(4):1–21, 2021. 3, 5, 7
- [40] Rolandos Alexandros Potamias, Stylianos Ploumpis, Stylianos Moschoglou, Vasileios Triantafyllou, and Stefanos Zafeiriou. Handy: Towards a high fidelity 3d hand shape and appearance model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4670–4680, 2023. 2, 6, 3
- [41] Neng Qian, Jiayi Wang, Franziska Mueller, Florian Bernard, Vladislav Golyanik, and Christian Theobalt. Html: A parametric hand texture model for 3d hand reconstruction and personalization. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16, pages 54–71. Springer, 2020. 2
- [42] Haonan Qiu, Zhaoxi Chen, Yuming Jiang, Hang Zhou, Xiangyu Fan, Lei Yang, Wayne Wu, and Ziwei Liu. Relitalk: Relightable talking portrait generation from a single video. arXiv preprint arXiv:2309.02434, 2023. 3
- [43] Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. Pivotal tuning for latent-based editing of real images. ACM Trans. Graph., 2021. 3
- [44] Javier Romero, Dimitrios Tzionas, and Michael J Black. Embodied hands: modeling and capturing hands and bodies together. ACM Transactions on Graphics (TOG), 36(6):1–17,

2017. 2, 3

- [45] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015. 1
- [46] Kripasindhu Sarkar, Marcel C. Buehler, Gengyan Li, Daoye Wang, Delio Vicini, J´er´emy Riviere, Yinda Zhang, Sergio Orts-Escolano, Paulo Gotardo, Thabo Beeler, and Abhimitra Meka. Litnerf: Intrinsic radiance decomposition for highquality view synthesis and relighting of faces. In ACM SIGGRAPH Asia 2023, 2023. 8
- [47] Soumyadip Sengupta, Angjoo Kanazawa, Carlos D Castillo, and David W Jacobs. Sfsnet: Learning shape, reflectance and illuminance of facesin the wild’. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6296–6305, 2018. 3
- [48] Breannan Smith, Chenglei Wu, He Wen, Patrick Peluse, Yaser Sheikh, Jessica K Hodgins, and Takaaki Shiratori. Constraining dense hand surface tracking with elasticity. ACM Transactions on Graphics (TOG), 39(6):1–14, 2020. 2
- [49] Srinath Sridhar, Antti Oulasvirta, and Christian Theobalt. Interactive markerless articulated hand motion tracking using

- rgb and depth data. In Proceedings of the IEEE international conference on computer vision, pages 2456–2463, 2013. 2
- [50] Srinath Sridhar, Franziska Mueller, Antti Oulasvirta, and Christian Theobalt. Fast and robust hand tracking using detection-guided optimization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3213–3221, 2015. 2
- [51] Tiancheng Sun, Zexiang Xu, Xiuming Zhang, Sean Fanello, Christoph Rhemann, Paul Debevec, Yun-Ta Tsai, Jonathan T Barron, and Ravi Ramamoorthi. Light stage superresolution: continuous high-frequency relighting. ACM Transactions on Graphics (TOG), 39(6):1–12, 2020. 3
- [52] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR,

2019. 5

- [53] Ayush Tewari, Justus Thies, Ben Mildenhall, Pratul Srinivasan, Edgar Tretschk, Wang Yifan, Christoph Lassner, Vincent Sitzmann, Ricardo Martin-Brualla, Stephen Lombardi, et al. Advances in neural rendering. In Computer Graphics Forum, pages 703–735. Wiley Online Library, 2022. 2
- [54] Anastasia Tkach, Mark Pauly, and Andrea Tagliasacchi. Sphere-meshes for real-time hand modeling and tracking. ACM Transactions on Graphics (ToG), 35(6):1–11, 2016. 2
- [55] Bohan Wang, George Matcuk, and Jernej Barbiˇc. Hand modeling and simulation using stabilized magnetic resonance imaging. ACM Transactions on Graphics (TOG), 38(4):1– 14, 2019. 2
- [56] Yifan Wang, Aleksander Holynski, Xiuming Zhang, and Xuaner Zhang. Sunstage: Portrait reconstruction and relighting using the sun as a light stage. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20792–20802, 2023. 3
- [57] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600–612, 2004. 5
- [58] Andreas Wenger, Andrew Gardner, Chris Tchou, Jonas Unger, Tim Hawkins, and Paul Debevec. Performance relighting and reflectance transformation with timemultiplexed illumination. ACM Transactions on Graphics (TOG), 24(3):756–764, 2005. 3
- [59] Tim Weyrich, Wojciech Matusik, Hanspeter Pfister, Bernd Bickel, Craig Donner, Chien Tu, Janet McAndless, Jinho Lee, Addy Ngan, Henrik Wann Jensen, et al. Analysis of human faces using a measurement-based skin reflectance model. ACM Transactions on Graphics (ToG), 25(3):1013– 1024, 2006. 2
- [60] Yiheng Xie, Towaki Takikawa, Shunsuke Saito, Or Litany, Shiqin Yan, Numair Khan, Federico Tombari, James Tompkin, Vincent Sitzmann, and Srinath Sridhar. Neural fields in visual computing and beyond. In Computer Graphics Forum, pages 641–676. Wiley Online Library, 2022. 2
- [61] Yingyan Xu, Gaspard Zoss, Prashanth Chandran, Markus Gross, Derek Bradley, and Paulo Gotardo. Renerf: Relightable neural radiance fields with nearfield lighting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22581–22591, 2023. 8

- [62] Zexiang Xu, Kalyan Sunkavalli, Sunil Hadap, and Ravi Ramamoorthi. Deep image-based relighting from optimal sparse samples. ACM Transactions on Graphics (ToG), 37

(4):1–13, 2018. 3

- [63] Shugo Yamaguchi, Shunsuke Saito, Koki Nagano, Yajie Zhao, Weikai Chen, Kyle Olszewski, Shigeo Morishima, and Hao Li. High-fidelity facial reflectance and geometry inference from an unconstrained image. ACM Transactions on Graphics (TOG), 37(4):1–14, 2018. 3
- [64] Haotian Yang, Mingwu Zheng, Wanquan Feng, Haibin Huang, Yu-Kun Lai, Pengfei Wan, Zhongyuan Wang, and Chongyang Ma. Towards practical capture of high-fidelity relightable avatars. In SIGGRAPH Asia 2023 Conference Proceedings, 2023. 3, 5, 6, 7, 4
- [65] Yu-Ying Yeh, Koki Nagano, Sameh Khamis, Jan Kautz, Ming-Yu Liu, and Ting-Chun Wang. Learning to relight portrait images via a virtual light stage and synthetic-to-real adaptation. ACM Transactions on Graphics (TOG), 41(6): 1–21, 2022. 3
- [66] Wang Yifan, Lukas Rahmann, and Olga Sorkine-hornung. Geometry-consistent neural shape representation with implicit displacement fields. In International Conference on Learning Representations, 2022. 4
- [67] Kai Zhang, Fujun Luan, Qianqian Wang, Kavita Bala, and Noah Snavely. Physg: Inverse rendering with spherical gaussians for physics-based material editing and relighting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5453–5462, 2021. 2
- [68] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 5
- [69] Xiuming Zhang, Sean Fanello, Yun-Ta Tsai, Tiancheng Sun, Tianfan Xue, Rohit Pandey, Sergio Orts-Escolano, Philip Davidson, Christoph Rhemann, Paul Debevec, et al. Neural light transport for relighting and view synthesis. ACM Transactions on Graphics (TOG), 40(1):1–17, 2021. 3
- [70] Xiuming Zhang, Pratul P Srinivasan, Boyang Deng, Paul Debevec, William T Freeman, and Jonathan T Barron. Nerfactor: Neural factorization of shape and reflectance under an unknown illumination. ACM Transactions on Graphics (ToG), 40(6):1–18, 2021. 2
- [71] Mianlun Zheng, Bohan Wang, Jingtao Huang, and Jernej Barbiˇc. Simulation of hand anatomy using medical imaging. ACM Transactions on Graphics (TOG), 41(6):1–20, 2022. 2

## URHand: Universal Relightable Hands Supplementary Material

### 7. Demo Video and Additional Results

We provide the supplementary video on our project page (https : / / frozenburning . github . io / projects/urhand), which includes more visual results and additional discussions of our work. Specifically, it contains:

- • Motivation and key features of URHand.
- • An animated overview and illustration of the proposed framework.
- • Video comparisons with baseline methods.
- • Additional qualitative results with diverse identities, including 1) relighting with monochrome directional light,

2) relighting with arbitrary environment map, and 3) quick personalization from a phone scan with corresponding relighting results with environment maps.

### 8. Network Architecture

In this section, we provide the details of our network architecture and hyperparameters for our hand geometry model (Sec. 8.1), physical branch (Sec. 8.2), and neural branch (Sec. 8.3), respectively.

#### 8.1. Hand Geometry Autoencoder

We design an autoencoder to obtain accurate hand tracking and geometry from input fully lit frames similar to [32]. The architecture of this autoencoder {Eid,Did,Eθ,Dθ} is illustrated in Figure 8. Specifically, it consists of an identity encoder Eid, an identity decoder Did, a pose encoder Eθ, and a pose decoder Dθ.

The identity encoder Eid takes as input the depth map in the neutral pose and the coordinates of joints in the neutral pose, which predicts the mean and variance of the distribution of the identity code (i.e. ID code). The inputs of the identity encoder have normalized viewpoints by rigidly aligning them to a reference coordinate system; hence both pose and viewpoints are normalized and only identity information is included in them. The identity decoder Did learns to decode the identity-dependent offset of joints and vertices from the ID code z. The identity-dependent offset of joints is responsible for adjusting 3D joint coordinates in the template space for each identity, and the offset of vertices are for adjusting 3D vertices in the template space for each identity. The pose encoder Eθ directly regresses hand pose θ from the input image and the coordinates of joints. The pose decoder Dθ learns to predict the pose-dependent offset of vertices given the pose θ and ID code z. To get posed 3D meshes, we apply the three types of correctives to

the template mesh and perform linear blend skinning with the estimated 3D pose from the pose encoder.

We train this autoencoder {Eθ,Dθ,Eid,Did} on fully lit frames with all identities to obtain a general hand tracker. The autoencoder is trained by minimizing 1) L1 distance between joint coordinates, 2) point-to-point L1 distance from 3D scans with weight 10, 3) KL-divergence of the ID code with weight 0.001, and 4) various regularizers like Moon et al. [32]. We freeze it during the training of URHand as well as the quick personalization from phone scans.

#### 8.2. Physical Branch

The physical branch of URHand consists of a 2D UNet [45] FG and a parametric BRDF [3] Fpb, where only the U-Net FG contains optimizable parameters. The U-Net encoder is a 6-layer convolutional neural network (CNN) with channel sizes (3,64,64,64,64,64,64), which takes as input the mean texture T ∈ R1024×1024×3. Here, the hand pose θ is tiled into a UV-aligned 2D feature map θ′, concatenated with the output feature from the U-Net encoder as a joint feature Fθ,id, and passed to the U-Net decoder. The U-Net decoder is a 6-layer CNN with skip connections from the U-Net encoder, with channel sizes (64,64,64,64,64,64,2). We use a transposed convolution layer followed by bilinear interpolation as the upsampling layer in the U-Net decoder. The U-Net decoder predicts the displacement map δd ∈ R1024×1024 and the roughness map β ∈ R1024×1024. We unwrap the coarse mesh M from our hand geometry autoencoder into the UV space to obtain the corresponding coarse normal map n ∈ R1024×1024×3. The predicted displacement map is applied on top of this coarse normal map to obtain the refined normal map nˆ according to Eq. 2 in the main paper.

The parametric BRDF Fpb takes as input the refined normal map nˆ, the roughness map β, light L = {Li(ωi)}i, and view direction d. The physics-inspired shading features

Fpb = {Cdpb,Cspb} are computed accordingly. Specifically, the diffuse feature Cdpb is computed as:

###### Cdpb = Li(ωi) · Vi · (ωi · nˆ)dωi, (10)

where Li(ωi) is the light intensity from the incident direction ωi, Vi is the visibility given the light Li. Furthermore,

ID-dependent Vertex Offset

Mean

Depth in

…

ID

[Figure 172]

FC

FC

FC

ID Code 𝒛

ResNet-18

Neutral Pose

Feature

ID-dependent

Var

Skeleton Offset

concatenate

Joints in Neutral Pose

ID Encoder ℰid

ID Decoder 𝒟id

… VertexOffset

Pose Feature

…

Pose-dependent

FC

FC

FC

FC

Pose 𝜽

##### Image ResNet-50

grid sample

Joints in

Pose Encoder ℰ𝜃

Pose Decoder 𝒟𝜃

Poses

- Figure 8. The architecture of our hand geometry autoencoder. The identity encoder Eid takes as input the depth map in the neutral pose and the coordinates of joints in the neutral pose, which predicts the mean and variance of the distribution of the identity code (i.e. ID code). The identity decoder Did learns to decode the identity-dependent offset of vertices and skeletons from the ID code z. The pose encoder Eθ directly regresses hand pose θ from the input image and the coordinates of joints. The pose decoder Dθ learns to predict the pose-dependent offset of vertices given the pose θ and ID code z.

the specular feature Cspb is computed as:

Cspb = D · F · G · Li(ωi) · Vi · (ωi · nˆ)dωi, (11)

β4 π[(h · nˆ)2(β4 − 1) + 1]2

, (12)

D =

- F = F0 + (1 − F0) · 2[λ

F1(d·h)+λF2](d·h), (13)

- G =

1 4[(nˆ · d)(1 − K) + K][(nˆ · ωi)(1 − K) + K]

, (14) h =

(β + 1)2 8

ωi + d ||ωi + d||

, (15)

, K =

where we set Fresnel coefficient F0 = 0.04, λF1 = −5.55473, and λF2 = −6.98316, respectively.

#### 8.3. Neural Branch

The neural branch of URHand consists of a non-linear network Fnl and a linear network Fl (i.e. linear lighting model). We illustrate the detailed architecture of the neural branch in Figure 9. Specifically, the nonlinear network Fnl is a 7-layer CNN with channel sizes (128,256,128,128,64,32,16,4), which takes as input the pose- and identity-dependent joint feature Fθ,id. The linear network Fl, namely the linear lighting model, consists of an encoder Fl−enc and a decoder Fl−dec. The linear encoder Fenc consists of unbiased convolutional layers which takes as input the concatenated physics-inspired shading features

{Cdpb,Cspb}. The linear decoder is a 7-layer unbiased CNN with channel sizes (128,256,128,128,64,32,16,4). We

fuse the linear features from Fl−enc and the non-linear features from Fnl as layer-wise modulation at each layer of the linear decoder Fl−dec according to Eq. 5 in the main paper. The predicted gain map g ∈ R1024×1024×3 and bias map b ∈ R1024×1024 contributes to the final rendering according to Eq. 6 in the main paper.

### 9. Adaptation to a Phone Scan

In this section, we present the details of how to quickly adapt URHand to a personalized use case from a phone scan. We use a single iPhone 12 to scan a hand, which incorporates a depth sensor that can be used to extract better geometry of the user’s hand. Our phone scans include hands with neutral finger poses and static 3D global translations with varying 3D global rotations to expose most of the hand surfaces.

We pre-process the phone scan with 1) our in-house 2D hand keypoint detector to obtain 2D hand joint coordinates and 2) RVM [27] to obtain the foreground mask of the phone scan. After the preprocessing, we optimize 3D global rotation, 3D pose, 3D global translation, and ID code of the phone scan. The 3D global rotation, 3D pose, and 3D global translation are optimized for each frame, and a single ID code is shared across all frames as all frames are from a single identity. We optimize them by minimizing 1) L1 distance between projected 2D joint coordinates and targets

|ℱl−enc|𝐂𝑑 ,𝐂𝑠|
|---|---|
|Unbiased Conv<br><br>Unbiased Conv<br><br>Light Feature<br><br>Unbiased Conv<br><br>ℱl−enc𝑗<br><br>Unbiased Conv<br><br>…<br><br>……<br><br>pb pb<br><br>| |

……

…

ℱl−dec

Unbiased ConvT

ℱl−dec𝑗

|ℱnl| |
|---|---|
|ID&Pose Feature 𝐅𝜃,id<br><br>ConvT<br><br>Leaky ReLU<br><br>ConvT<br><br>Leaky ReLU<br><br>ℱnl𝑗<br><br>…<br><br>| |

| | |
|---|---|
| | |

Unbiased ConvT

⊙

ℱl−dec𝑗+1

…

…

- Figure 9. The architecture of the neural branch of URHand. The neural branch of our model consists of a non-linear network

Fnl and a linear network Fl (i.e. linear lighting model). Notably, we remove all non-linear activation and use the convolutional layer without bias in the linear network so that the linearity of the output is explicitly kept w.r.t. the input physics-inspired shading feature {Cdpb, Cspb}. This figure also illustrates how Eq. 5 in the main paper is implemented within our network.

2) L1 distance between differentiably rendered masks and targets with weight 50, and 3) L1 distance between differentiably rendered depth maps and targets with weight 100. After the optimization, we unwrap per-frame images to UV space and average intensity values at each texel considering the visibility to get the unwrapped texture map.

To remove shadows from the unwrapped textures, we first obtain the average color of the foreground pixels of captured images. Then, we optimize shadow as a 1-channel difference (i.e., darkness difference) between the averaged color and the captured image in the UV space. To prevent the shadow from dominating local sharp textures (i.e., hairs and tattoos), we apply a total variation regularizer to the shadow. The unwrapped texture without the shadow is simply obtained by dividing the unwrapped texture by the

shadow. We empirically observed that such a statistical approach produces better shadow than the physics-based approach of HARP [20], which assumes a single point light, as there is often more than one light source in the scan environment. Then, we take this texture map after shadow removal as the input to URHand for relighting without any finetuning.

### 10. Implementation of Baselines

In this section, we present implementation details of our baseline methods for comparisons in the main paper. Specifically, we introduce our modifications to RelightableHands [14] and Handy [40] in Sec. 10.1. Moreover, we introduce our implementations of all baselines for ablation studies in Sec. 10.2.

#### 10.1. Methods for Main Comparisons

RelightableHands [14] is originally proposed for peridentity relightable appearance reconstruction tailored with volumetric representation [28]. For a fair comparison, we re-implement it with our mesh-based representation. Specifically, we leverage a U-Net as the texture decoder Atex that takes as input the UV-aligned view direction, light direction, and visibility. The pose parameter is tiled into a UV-aligned feature map and concatenated with the bottleneck representation of the U-Net. This texture decoder Atex predicts texture map T ∈ R1024×1024×3 and shadow map S ∈ R1024×1024 in the UV space. The final texture C for rendering is obtained as:

###### C = σ(S)(ReLU(λsT) + λb), (16)

where σ(·) is the sigmoid function, ReLU(x) = max(0,x), λs is a scale factor, and λb is a bias parameter. In our experiments, we set λs = 25, and λs = 100, respectively.

Handy [40] leverages the parametric hand model [44] as the shape representation and StyleGAN3 [18] as the texture model. As the shape and latent code regressor are not publicly available, we cannot infer the latent code w or shape parameters from input images as in the original paper. Instead, we fit hand shape parameters using our multiview fully lit frames on the training segments. Then, we do the StyleGAN inversion following [43]. Specifically, we randomly initialize the latent code w and optimize it given the reconstruction loss between the ground truth fully lit images and the images rendered with the current predicted texture map. In our experiments, we optimize 50,000 iterations for each identity. We use the Adam [21] optimizer with the initial learning rate as 1×10−3. Once the inversion is done, we take the latent code w and feed it into the pretrained texture model of Handy to get the unwrapped texture map. We treat the unwrapped texture map as the albedo map for physically based relighting evaluation.

#### 10.2. Baselines for Ablation Studies

Non-linear model is based on our full model but we add LeakyReLU function to all layers in the original linear network Fl which breaks the linearity.

Linear consistency model is based on the aforementioned non-linear model. We additionally constrain the linearity of this non-linear network by applying linearity consistency loss during training. Specifically, for every n iterations, we augment two physics-inspired shading features with two random scalars, i.e. a1F1pb + a2F2pb, where a1,a2 ∈ (0,1). The linearity consistency loss is defined as:

Llc = ||a1Fl(F1pb)+a2Fl(F2pb)−Fl(a1F1pb+a2F2pb)||2 (17)

MLP-based linear model [64] is a variant of the linear lighting model with no spatially varying lighting feature. We replace the encoder of linear network Fl−enc as a onelayer MLP without bias. It takes as input the environment map with a resolution of 3 × 16 × 32, and predicts the lighting feature. Then we reshape the prediction into a UValigned feature map with a resolution of 128 × 16 × 16 and feed into the decoder of linear network to predict the final gain and bias map for neural rendering.

Phong based model is implemented by replacing our physics-inspired shading feature Fpb = {Cdpb,Cspb} with simple diffuse and specular feature from the Phong reflectance model. This neural relighting model is similar to [2, 14] with no learnable material parameter.

w/o Specular is the baseline where we dropout the specular feature Cspb during training.

w/o Visibility is the baseline where we do not incorporate visibility Vi when compute the physics-inspired shading feature in Eq. 10 and Eq. 11.

w/o Refiner is the baseline where we only use the normal map n from the coarse geometry without further refinement during training.

w/o LGAN is the baseline trained with the reconstruction loss Limg and L1 regularization Lreg only.

w/o Light-aware LGAN is the baseline trained with the vanilla adversarial loss without conditional discriminator. Specifically, the adversarial loss of Eq. 8 in the main paper degrades to LGAN = log FD(I)+log[1−FD(Iˆ)], where I is the ground truth and Iˆis the rendered image.

w/o L1 Reg is the baseline trained with the reconstruction loss Limg and lighting-aware adversarial loss LGAN only.

