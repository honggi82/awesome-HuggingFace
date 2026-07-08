## ReconX: Reconstruct Any Scene from Sparse Views with Video Diffusion Model

Fangfu Liu∗, Wenqiang Sun∗, Hanyang Wang∗, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, Fellow, IEEE and Yueqi Duan, Member, IEEE,

arXiv:2408.16767v4[cs.CV]25Jun2025

Abstract—Advancements in 3D scene reconstruction have transformed 2D images from the real world into 3D models, producing realistic 3D results from hundreds of input photos. Despite great success in dense-view reconstruction scenarios, rendering a detailed scene from sparse views is still an ill-posed optimization problem, often resulting in artifacts and distortions in unseen areas. In this paper, we propose ReconX, a novel 3D scene reconstruction paradigm that reframes the ambiguous reconstruction problem as a temporal generation task. The key insight is to unleash the strong generative prior of large pretrained video diffusion models for sparse-view reconstruction. Nevertheless, it is challenging to preserve 3D view consistency when directly generating video frames from pre-trained models. To address this issue, given limited input views, the proposed ReconX first constructs a global point cloud and encodes it into a contextual space as the 3D structure condition. Guided by the condition, the video diffusion model then synthesizes video frames that are detail-preserved and exhibit a high degree of 3D consistency, ensuring the coherence of the scene from various perspectives. Finally, we recover the 3D scene from the generated video through a confidence-aware 3D Gaussian Splatting optimization scheme. Extensive experiments on various real-world datasets show the superiority of ReconX over stateof-the-art methods in terms of quality and generalizability.

Index Terms—Sparse-view Reconstruction, Video Diffusion, Gaussian Splatting

I. INTRODUCTION

With the rapid development of photogrammetry techniques such as NeRF [1] and 3D Gaussian Splatting (3DGS) [2], 3D reconstruction has become a popular research topic in recent years, finding various applications from virtual reality [3] to autonomous navigation [4] and beyond [5], [6], [7], [8], [9]. However, sparse-view reconstruction is an ill-posed problem [10], [11] since it involves recovering a complex 3D structure from limited viewpoint information (i.e., even as few

- as two images) that may correspond to multiple solutions.

Fangfu Liu and Yueqi Duan is with the Department of Electronic Engineering, Tsinghua University, Beijing 100084, China (e-mail: liuff23@mails.tsinghua.edu.cn, duanyueqi@tsinghua.edu.cn).

Wenqiang Sun and Jun Zhang is with the Department of Electronic and Computer Engineering, Hong Kong University of Science and Technology, Hong Kong 999077, China (e-mail: wsunap@connect.ust.hk, eejzhang@ust.hk)

Hanyang Wang, Junliang Ye is with the Department of Computer Science, Tsinghua University, Beijing 100084, China (e-mail: hanyang21@mails.tsinghua.edu.cn, yejl23@mails.tsinghua.edu.cn).

Yikai Wang is with the School of Artificial Intelligence, Beijing Normal University, Beijing 100875, China (e-mail: yikaiw@bnu.edu.cn)

Haowen Sun is with the Department of Automation, Tsinghua University, Beijing 100084, China (e-mail: sunhw24@mails.tsinghua.edu.cn).

Corresponding authors: Yueqi Duan, Yikai Wang. The ∗ denotes equal contributions.

This uncertain process requires additional assumptions and constraints to yield a viable solution.

Recently, powered by the efficient and expressive 3DGS [2] with fast rendering speed and high quality, several feedforward Gaussian Splatting methods [12], [13], [14] have been proposed to explore 3D scene reconstruction from sparse view images. Although they can achieve promising interpolation results by learning scene-prior knowledge from feature extraction modules (e.g., epipolar transformer [12]), insufficient captures of the scene still lead to an ill-posed optimization problem [15]. As a result, they often suffer from severe artifact and implausible imagery issues when rendering the 3D scene from novel viewpoints, especially in unseen areas.

To address the limitations, we propose ReconX, a novel 3D scene reconstruction paradigm that reformulates the inherently ambiguous reconstruction problem as a generation problem. Our key insight is to unleash the strong generative prior of pretrained large video diffusion models [16], [17], [18] to create more observations for the downstream reconstruction task. Despite the capability to synthesize video clips featuring plausible 3D structures [10], recovering a high-quality 3D scene from current video diffusion models is still challenging, due to the poor 3D view consistency across generated 2D frames. Grounded by theoretical analysis, we explore the potential of incorporating 3D structure condition into the video generative process, which bridges the gap between the under-determined 3D creation problem and the fully-observed 3D reconstruction setting. Specifically, given sparse images, we first build a global point cloud through a pose-free stereo reconstruction method. Then we encode it into a rich context representation space as the 3D condition in cross-attention layers, which guides the video diffusion model to synthesize detail-preserved frames with 3D consistent novel observations of the scene. Finally, we reconstruct the 3D scene from the generated video through Gaussian Splatting with a 3D confidence-aware and robust scene optimization scheme, which further deblurs the uncertainty in video frames effectively. Extensive experiments verify the efficacy of our framework and show that ReconX outperforms existing methods for high quality and generalizability, revealing the great potential to craft intricate 3D worlds from video diffusion models. The overview and examples of reconstructions are shown in Fig. 1.

In summary, our main contributions are as follows:

- • We introduce ReconX, a novel sparse-view 3D scene reconstruction framework that reframes the ambiguous reconstruction challenge as a temporal generation task.
- • We incorporate the 3D structure condition into the con-

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

ReconX PSNR:24.57

pixelSplat

Sparse Views

- PSNR:17.62

MVSplat

- PSNR:18.83

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

MultiView Optim.

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Video Gen.

[Figure 17]

[Figure 18]

# Recon

[Figure 19]

[Figure 20]

- Fig. 1. An overview of our ReconX framework for sparse-view reconstruction. Unleashing the strong generative prior of video diffusion models, we can create more observations for 3D reconstruction and achieve impressive performance.

MVsplat [14] introduces the cost volume and depth refinements to produce a clean and high-quality 3D Gaussians in a faster way. LatentSplat [31] encodes the variational 3D Gaussians and utilizes a discriminator to synthesize more realistic images. To reconstruct a complete scene from a single image, Flash3D [32] adopts a hierarchical 3DGS learning policy and depth constraint to achieve high-quality interpolation and extrapolation view synthesis. Although these methods leverage the 3D data priors, they are limited by the scarcity and diversity of 3D data. Consequently, these methods struggle to achieve high-quality renderings in unseen areas, especially when out-of-distribution (OOD) data is used as input.

ditional space of the video diffusion model to generate 3D consistent frames and propose a 3D confidence-aware optimization scheme in 3DGS to reconstruct the scene given the generated video.

• Extensive experiments demonstrate that our ReconX outperforms existing methods for high-fidelity and generalizability on a variety of real-world datasets.

II. RELATED WORK

Sparse-view reconstruction. NeRF [1] and 3DGS [2] typically demand hundreds of input images and rely on the multi-view stereo reconstruction (MVS) approach (e.g., COLMAP [19]) to estimate the camera parameters. To address the issue of low-quality 3D reconstruction caused by sparse views, PixelNeRF [11] proposes using convolutional neural networks to extract features from the input context. Moreover, FreeNeRF [20] adopts the frequency and density regularized strategies to alleviate the artifacts caused by insufficient inputs without any additional cost. To mitigate the overfitting to input sparse views in 3DGS, FSGS [21] and SparseGS [22] employ a depth estimator to regularize the optimization process. However, these methods all require known camera intrinsics and extrinsics, which is not practical in real-world scenario. Benefiting from the existing powerful 3D reconstruction model (i.e., DUSt3R [23]), InstantSplat [24] is able to acquire accurate camera parameters and initial 3D representations from unposed sparse-view inputs, leading to efficient and highquality 3D scene reconstruction.

Generative models for 3D reconstruction. Constructing comprehensive 3D scenes from limited observations demands generating 3D content, particularly for unseen areas. Earlier studies distill the knowledge in the pre-trained text-to-image diffusion models [33], [34], [35], [36] into a coherent 3D model. Specifically, the Score Distillation Sampling (SDS) technique [15], [37], [38], [39] is adopted to synthesize a 3D object from the text prompt. To enhance the 3D consistency, several approaches [8], [40], [41] inject the camera information into diffusion models, providing strong multi-view priors. Furthermore, ZeroNVS [42] and CAT3D [10] extend the multiview diffusion to the scene level generation. GeNVS [43] embeds a 3D feature field into the diffusion model to enhance the novel view synthesis ability. More recently, video diffusion models [16], [18] have shown an impressive ability to produce realistic videos and are believed to implicitly understand 3D structures [44]. SV3D [45] and V3D [46] explore fine-tuning the pre-trained video diffusion model for 3D object generation. Meanwhile, MotionCtrl [47] and CameraCtrl [48] achieve scene-level controllable video generation from a single image by explicitly injecting the camera pose into video diffusion models. However, they suffer from performance degradation in the unconstrained sparse-view 3D scene reconstruction, which requires strong 3D consistency.

Regression model for generalizable view synthesis. While 3D reconstruction methods like NeRF and 3DGS are optimized per-scene [25], [26], [27], [28], [29], [30], a line of research aims to train feed-forward models that output a 3D representation directly from a few input images, bypassing the need for time-consuming optimization. Splatter image [13] performs an efficient feed-forward manner for monocular 3D object reconstruction by predicting a 3D Gaussian for each image pixel. Meanwhile, pixelSplat [12] proposes predicting the scene-level 3DGS from the image pairs, using the epipolar transformer to better extract scene features. Following that,

III. PRELIMINARIES Video Diffusion Models. Diffusion models [49], [50] have emerged as the cutting-edge paradigm to generate high-quality

##### 3D Structure Condition

##### Confidence-aware 3DGS Optimization

[Figure 21]

Normalization

Image Cross-Attn

Attention

PointCloud

Self-Attn

Embedder

[Figure 22]

Cross

[Figure 23]

FFN

[Figure 24]

⊕

FFN

⊕

⊕

3D Structure Cross-Attn

frame confidence map

[Figure 25]

[Figure 26]

Point Cloud extraction

downsample

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Consistent Frames

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

CLIP

[Figure 38]

[Figure 39]

[Figure 40]

...?...

Dec

[Figure 41]

Diffusion

[Figure 42]

[Figure 43]

[Figure 44]

###### ... ...

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Enc

[Figure 49]

[Figure 50]

Input Sparse Views

Video Diffusion U-Net

- Fig. 2. Pipeline of ReconX. Given sparse-view images as input, we first build a global point cloud and project it into 3D context representation space as 3D structure condition. Then we inject the 3D structure condition into the video diffusion process and guide it to generate 3D consistent video frames. Finally, we reconstruct the 3D scene from the generated video through Gaussian Splatting with a 3D confidence-aware and robust scene optimization scheme. In this way, we unleash the strong power of the video diffusion model to reconstruct intricate 3D scenes from very sparse views.

videos. These models learn the underlying data distribution by adding and removing noise on the clean data. The forward process aims to transform a clean data sample x0 ∼ p(x) to a pure Gaussian noise xT ∼ N(0,I), following the process:

For every pixel, the Gaussians are traversed in depth order from the image plane, and their view-dependent colors ci are combined through alpha compositing, leading to the pixel color C:

- i−1
- j=1

xt = √α¯tx0 + √1 − α¯tϵ, ϵ ∼ N(0,1), (1) where xt and α¯t denotes the noisy data and noise strength

(1 − αi). (5)

#### C =

ciαi

i∈N

End-to-end Dense Unconstrained Stereo. DUSt3R [23] is a new model to predict a dense and accurate 3D scene representation solely from image pairs without any prior information about the scene. Given two unposed images {I1,I2}, this endto-end model is trained to estimate the point maps {P1,1,P2,1} and confidence maps {C1,1,C2,1}, which can be utilized to recover the camera parameters and dense point cloud. The training procedure for view v ∈ {1,2} is formulated as a regression loss:

- at the timestep t. The denoising neural network ϵθ is trained to predict the noises added in the forward process, which is achieved by the MSE loss:

L = Ex∼p,ϵ∼N(0,I),c,t ∥ϵ − ϵθ (xt,t,c)∥22 , (2)

where c represents the embeddings of conditions like text or image prompt. For the video diffusion models, Latent Diffusion Models (LDMs) [33], which compress images into the latent space, are commonly employed to mitigate the computation complexity while maintaining competitive performance.

1 zi · Pv,1 −

1 zˆi · Pˆv,1 , (6)

L =

3D Gaussian Splatting. 3DGS [2] represents a scene explicitly by utilizing a set of 3D Gaussian spheres, achieving a fast and high-quality rendering. A 3D Gaussian is modeled by a position vector µ ∈ R3, a covariance matrix Σ ∈ R3×3, an opacity α ∈ R, and spherical harmonics (SH) coefficient c ∈ Rk [51]. Moreover, the Gaussian distribution is formulated as the following:

where P and Pˆ denote the ground-truth and prediction point maps, respectively. The scaling factors zi = norm(P1,1,P2,1) and zˆi=norm(Pˆ1,1,Pˆ2,1) are adopted to normalize the point maps, which merely indicate the mean distance D of all valid points from the origin:

1 |D1| + |D2|

Pvi . (7)

norm(P1,1,P2,1) =

- 1

- 2(x−µ)TΣ−1(x−µ), (3)

G(x) = e−

v∈{1,2} i∈Dv

where Σ = RSSTRT, S denotes the scaling matrix and R is the rotation matrix.

IV. METHOD A. Motivation for ReconX

In the rendering stage, the 3D Gaussian spheres are transformed into 2D camera planes through rasterization [52]. Specifically, given the perspective transformation matrix W and Jacobin of the projection matrix J, the 2D covariance matrix in the camera space is computed as

In this paper, we focus on the fundamental problem of 3D scene reconstruction and novel view synthesis (NVS) from very sparse view (e.g., as few as two) images. Most existing works [14], [11], [12], [32] utilize 3D prior and geometric constraints (e.g., depth, normal, cost volume) to

′

= JWΣWTJT. (4)

Σ

fill the gap between observed and novel regions in sparseview 3D reconstruction. Although capable of producing highly realistic images from the given viewpoints, these methods often struggle to generate high-quality images in areas not visible from the input perspectives due to the inherent problem of insufficient viewpoints and the resulting instability in the reconstruction process. To address this issue, a natural idea is to create more observations to convert the under-determined 3D creation problem into a fully constrained 3D reconstruction setting. Recently, video generative models have shown promise for synthesizing video clips featuring 3D structures [45], [16], [18]. This inspires us to unleash the strong generative prior of large pre-trained video diffusion models to create temporal consistent video frames for sparse-view reconstruction. Nevertheless, it is non-trivial as the main challenge lies in poor 3D view consistency among video frames, which significantly limits the downstream 3DGS training process. To achieve 3D consistency within video generation, we first analyze the video diffusion modeling from a 3D distributional view. Let x be the set of rendering 2D images from any 3D scene in the world, q(x) be the distribution of the rendering data x, and our goal is to minimize the divergence D:

D (q(x)∥pθ,ψ(x)), (8)

min

θ∈Θ,ψ∈Ψ

where pθ,ψ is a diffusion model parameterized by θ ∈ Θ (the parameters in the backbone) and ψ ∈ Ψ (any embedding function shared by all data). The vanilla video diffusion model [18] chooses a CLIP [53] model g to add an imagebased condition (i.e., ψ = g). However, in sparse-view 3D reconstruction, only conditioning on 2D images cannot provide sufficient condition for approximating q(x) [12], [14], [15]. Motivated by this, we explore the potential of incorporating the native 3D prior (denoted by F) to find an optimal solution in Equation 8 and derive a theoretical formulation for our analysis in Proposition 1.

Proposition 1: Let θ∗,ψ∗ = g∗ be the optimal solution of the solely image-based conditional diffusion scheme and θ˜∗,ψ˜∗ = {g∗,F∗} be the optimal solution of the diffusion scheme with a native 3D prior. Suppose the divergence D is convex and the embedding function space Ψ includes all measurable functions, then we have D(q (x)∥pθ˜∗,ψ˜∗(x)) < D (q (x)∥pθ∗,ψ∗(x)).

Towards this end, we reformulate the inherently ambiguous reconstruction problem as a generation problem by incorporating a 3D native structure condition into the diffusion process. Detailed proof can be found in the appendix.

B. Overview of ReconX Given K sparse-view (i.e., as few as two) images I =

Ii Ki=1 , Ii ∈ RH×W×3 , our goal is to reconstruct the underlying 3D scene, where we can synthesize novel views of unseen viewpoints. In our framework ReconX, we first build a global point cloud P = {pi,1 ≤ i ≤ N} ∈ RN×3 from I and project P into the 3D context representation space F as the structure condition F(P) (Sec. IV-C). Then we inject F(P) into the video diffusion process to generate 3D consistent video frames I′ = Ii K

′

i=1 ,(K′ > K),

thus creating more observations (Sec. IV-D). To alleviate the negative artifacts caused by the inconsistency among generated videos, we utilize the confidence maps C = {Ci}K

′

i=1 from the DUSt3R model and LPIPS loss [54] to achieve a robust 3D reconstruction (Sec. IV-E). In this way, we can unleash the full power of the video diffusion model to reconstruct intricate 3D scenes from very sparse views. Our pipeline is depicted in Fig. 2.

C. Building the 3D Structure Condition

Grounded by the theoretical analysis in Sec. IV-A, we leverage an unconstrained stereo 3D reconstruction method DUSt3R [23] with point-based representations to build the 3D structure condition F. Given a set of sparse images

I = Ii Ki=1, we first construct a connectivity graph G(V,E) of K input views similar to DUSt3R, where vertices V and

each edge e = (n,m) ∈ E indicates that the images In and Im shares visual contents. Then we use G to recover a globally aligned point cloud P. For each image pair e = (n,m), we predict pairwise pointmaps Pn,n,Pm,n and their corresponding confidence maps Cn,n,Cm,n ∈ RH×W×3. For clarity, we denote Pn,e := Pn,n and Pm,e := Pm,n. Since we aim to rotate all pairwise predictions into a shared coordinate frame, we introduce transformation matrix Te and scaling factor σe associated with each pair e ∈ E to optimize global point cloud P as:

HW

Civ,e ∥Piv − σeTePiv,e∥. (9)

P∗ = arg min

P,T,σ

e∈E v∈e

i=1

More details of the point cloud extraction can be found in [23]. Having aligned the point clouds P, we now project it into a 3D context representation space F through a transformer-based encoder for better interaction with latent features of the video diffusion model. Specifically, we embed the input point cloud P into a latent code using a learnable embedding function and a cross-attention encoding module:

F(P) = FFN CrossAttn(PosEmb(P˜),PosEmb(P)) , (10)

where P˜ is a down-sampled version of P at 1/8 scale to efficiently distill input points to a compact 3D context space. Finally, we get the 3D structure guidance F(P) which contains sparse structural information of the 3D scene that can be interpreted by the denoising U-Net. The PosEmb is a column-wise positional embedding function: R3 → RC, where C is the dimension of embedding. More specifically, the PosEmb function is implemented as follows: (1) Fixed Sinusoidal Basis: The basis e is a 3D sinusoidal encoding: e = [sin(20πp),sin(21πp),...], where p ∈ R3 is the position. (2) Embedding Calculation: The input x is projected onto e and its sine and cosine are concatenated: embeddings = concat(sin(proj),cos(proj)). (3) Learnable Transformation: The positional encoding is passed through an MLP along with the input x: y = MLP(concat(embeddings,x)). In short, PosEmb combines a fixed sinusoidal encoding with a learnable MLP transformation.

For the transformer-based encoder, we encode the DUSt3R point cloud data to a fixed-length sparse representation of the

point cloud. Specifically, we first employ a subsampling based on farthest point sampling (FPS) to reduce the point cloud to a smaller set of key points while retaining its overall structural characteristics. Then, we apply cross-attention between the embeddings of the original point cloud and downsampled point cloud. This mechanism can be interpreted as a form of partial self attention, where the downsampled points act as query anchors that aggregate information from the original point cloud. The encoder is not initialized from any pretrained models. Instead, it is trained jointly with the video diffusion model in an end-to-end manner. This design choice ensures that the encoder is specifically adapted to the characteristics of DUSt3R point clouds in our experiment datasets.

- D. 3D Consistent Video Frames Generation

In this subsection, we incorporate the 3D structure condition F(P) into the video diffusion process to obtain 3D consistent frames. To achieve consistency between generated frames and high-fidelity rendering views of the scene, we utilize the video interpolation capability to recover more unseen observations, where the first frame and the last frame of input to the video diffusion model are two reference views. Specifically, given

sparse-view images I = Iiref Ki=1 as input, we aim to render consistent frames f(Iiref−1,Iiref) = {Iiref−1,I2,...,IT,Iiref} ∈ R(T+2)×3×H×W where T is the number of generated novel frames. To unify the notation, we denote the embedding of image condition in the pretrained video diffusion model as Fg = g(Iref) and the embedding of 3D structure condition

- as FF = F(P). Subsequently, we inject the 3D condition into the video diffusion process by interacting with the U-Net

intermediate feature Fin through the cross-attention of spatial layers:

Fout = Softmax(

QKgT √

d

)Vg + λF · Softmax(

QKFT

√

d

)VF, (11)

where Q = FinWQ,Kg = FgWK,Vg = FgWV , KF = FFWK′ ,VF = FFWV′ are the query, key, and value of

- 2D and 3D embeddings respectively. WQ,WK,WK′ ,WV ,WV′ are the projection matrices and λF denotes the coefficient that balances image-conditioned and 3D structure-conditioned features. Given the first and last two views condition cview from Fg and 3D structure condition cstruc from FF, we apply the classifier-free guidance [55] strategy to incorporate the condition and our training objective is:

Ldiffusion = Ex∼p,ϵ∼N(0,I),t ∥ϵ − ϵθ (xt,t,cview,cstruc)∥22 ,

(12) where xt is the noise latent from the ground-truth views of the training data.

- E. Confidence-Aware 3DGS Optimization

Built upon the well-designed 3D structure condition, our video diffusion model generates highly consistent video frames, which can be used to reconstruct the 3D scene. As conventional 3D reconstruction methods are originally designed to handle real-captured photographs with calibrated camera metrics, directly applying these approaches to the

generated videos is not effective to recover the coherent scene due to the uncertainty of unconstrained images [23], [24]. To alleviate the uncertainty issue, we adopt a confidence-aware 3DGS mechanism to reconstruct the intricate scene. Different from recent approaches [56], [57] which model the uncertainty in per-image, we instead focus on a global alignment among a series of frames. For the generated frames Ii K

′

i=1, we denote Cˆi and Ci as the per-pixel color value for predicted and generated view i. Then, we model the pixel values as a Gaussian distribution in our 3DGS, where the mean and variance of Ii are Ci and σi. The variance σi measures the discrepancy between the predicted and generated images. The uncertainty metric σi for each image is estimated by minimizing the following negative log-likelihood among all frames:

′

∥Cˆi − C

i∥22 2σ2

- 1

- 2πσi2

, (13)

exp −

LIi

= −log

′

′

i=1 \ Ci) and A is a tailored global align function to establish connections between each frame and the other frames, enabling a more robust global uncertainty estimation. Specifically, the training objective of DUSt3R is to map image pairs to 3D space, while the confidence map C represents the model’s confidence in the pixel matches of image pairs within the 3D scene. Through its training process, DUSt3R inherently assigns low confidence to mismatched regions in image pairs, achieving the goal of Eq. 13. The

where C

i = A(Ci,{Ci}K

′

′

i=1 for each generated frames Ii K

confidence maps {Ci}K

i=1 are equivalent to the uncertainty σi. Meanwhile, the pairwise matching between all frames accomplishes the global alignment operation A. Moreover, we introduce the LPIPS [58] loss to remove the artifacts and further enhance the visual quality. Towards this end, we formulate the confidence-aware 3DGS loss between the Gaussian rendered image Iˆi and generated frame Ii as:

K′

Ci λrgbL1(Iˆi,Ii) + λssimLssim(Iˆi,Ii)

Lconf =

i=1

+λlpipsLlpips(Iˆi,Ii) , (14)

where L1, Lssim, and Llpips denote the L1, SSIM, and LPIPS loss, respectively, with λrgb, λssim, and λlpips being their corresponding coefficient parameters. In comparison to the photometric loss (e.g., L1 and Lssim), the LPIPS loss mainly focuses on the high-level semantic information.

V. EXPERIMENTS

In this section, we conduct extensive experiments to evaluate our ReconX. We first present the setup of the experiment (Sec V-A). Then we report our qualitative and quantitative results compared to 3D scene reconstruction method from two-views (Sec V-B) and multi-views (Sec. V-C) in various settings. We also show more generalizable experiment results to evaluate of extrapolation ability (Sec. V-D). Finally, we conduct ablation studies to further verify the efficacy of our framework design (Sec. V-E).

Input Ground Truth ReconX (Ours) MVSplat pixelSplat Input Ground Truth ReconX (Ours) MVSplat pixelSplat

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

HardSetCrossSet

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

EasySet

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

- Fig. 3. Qualitative comparison with two-view 3D scene reconstruction. We provide the comparison with other baselines in Easy Set, Hard Set, and Cross Set. In comparison to these two-views novel view synthesis methods, ReconX achieves better visual quality and generalization.

TABLE I QUANTITATIVE COMPARISONS WITH FEED-FORWARD BASED METHODS FOR SMALL ANGLE VARIANCE (EASY SET) IN INPUT VIEWS. FOR EACH SCENE, THE MODEL TAKES TWO VIEWS AS INPUT AND RENDERS THREE NOVEL VIEWS FOR EVALUATION.

TABLE II QUANTITATIVE COMPARISON WITH FEED-FORWARD BASED METHODS FOR LARGE ANGLE VARIANCE (HARD SET) IN INPUT VIEWS AND CROSS-DATASET (CROSS SET) COMPARISONS TO EVALUATE GENERALIZATION ABILITY.

Easy Set RealEstate10K ACID Method PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

Hard Set ACID RealEstate10K Method PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

pixelNeRF 20.43 0.589 0.550 20.97 0.547 0.533 GPNR 24.11 0.793 0.255 25.28 0.764 0.332 AttnRend 24.78 0.820 0.213 26.88 0.799 0.218 MuRF 26.10 0.858 0.143 28.09 0.841 0.155

pixelSplat 16.83 0.476 0.494 19.62 0.730 0.270 MVSplat 16.49 0.466 0.486 19.97 0.732 0.245 ReconX 24.53 0.847 0.083 23.70 0.867 0.143 Cross Set LLFF DTU

pixelSplat 11.42 0.312 0.611 12.89 0.382 0.560 MVSplat 11.60 0.353 0.425 13.94 0.473 0.385 ReconX 21.05 0.768 0.178 19.78 0.476 0.378

pixelSplat 25.89 0.858 0.142 28.14 0.839 0.150 MVSplat 26.39 0.839 0.128 28.25 0.843 0.144 ReconX 28.31 0.912 0.088 28.84 0.891 0.101

- A. Experiment Setup

implementation follows the pipeline of the original 3DGS [2], but unlike this method, we omit the adaptive control process and attain high-quality renderings in just 1000 steps. The coefficients λrgb, λssim, and λlpips are set to 0.8, 0.2, and 0.5, respectively.

Implementation Details. In our framework, we choose DUSt3R [23] as our unconstrained stereo 3D reconstruction backbone and the I2V model DynamiCrafter [18] (@ 512×512 resolution) as the video diffusion backbone. We first finetune the image cross-attention layers with 2000 steps on the learning rate 1 × 10−4 for warm-up. Then we incorporate the

Datasets. The video diffusion model of ReconX is trained on three datasets: RealEstate-10K [62], ACID [63], and DL3DV10K [61] based on the pretrained model. RealEstate-10K is a dataset downloaded from YouTube, which is split into 67,477 training scenes and 7,289 test scenes. The ACID dataset consists of natural landscape scenes, with 11,075 training scenes and 1,972 testing scenes. DL3DV-10K is a largescale outdoor dataset containing 10,510 videos with consistent capture standards. For each scene video, we randomly sample 32 contiguous frames with random skips and serve the first and last frames as the input for our video diffusion model. In two-views novel view synthesis experiment, we follow MVSplat [14] and pixelSplat [12] to choose test views in Easy Set. For Hard Set, we choose the frame intervals much larger (i.e., > 200 frames) than Easy Set. To further

- 3D structure condition cstruc into the video diffusion model and further finetune the spatial layers with 30K steps on the learning rate of 1 × 10−5. Our video diffusion was trained on 3D scene datasets by sampling 32 frames with dynamic FPS

- at the resolution of 512 × 512 in a batch. The AdamW [59] optimizer is employed for optimization. At the inference of our video diffusion, we adopt the DDIM sampler [60] using multi-condition classifier free guidance [55]. Similar to [18],

we adopt tanh gating to learn λF adaptively. The training is conducted on 8 NVIDIA A800 (80G) GPUs in two days. In the 3DGS optimization stage, we choose the point maps of the first and end frames as the initial global point cloud and all 32 generated frames are used to reconstruct the scene. Our

Input Views Ground Truth ReconX (Ours) DNGaussian 3DGS SparseNeRF

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

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

- Fig. 4. Qualitative comparison with sparse-view reconstruction methods on Mip-Nerf 360 and Tank and Temples. With sparse views as input, our ReconX achieves much better reconstruction quality compared with baselines.

TABLE III QUANTITATIVE COMPARISONS WITH MULTI-VIEWS RECONSTRUCTION METHODS ON MIPNERF 360 AND TANK AND TEMPLES, AND DL3DV. WE EVALUATE THE RECONSTRUCTION PERFORMANCE WITH DIFFERENT INPUT VIEWS FOR EACH SCENE.

Method

2-view 3-view 6-view 9-view PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ Mip-NeRF 360

3DGS 10.36 0.108 0.776 10.86 0.126 0.695 12.48 0.180 0.654 13.10 0.191 0.622 SparseNeRF 11.47 0.190 0.716 11.67 0.197 0.718 14.79 0.150 0.662 14.90 0.156 0.656 DNGaussian 10.81 0.133 0.727 11.13 0.153 0.711 12.20 0.218 0.688 13.01 0.246 0.678

- ReconX (Ours) 13.37 0.283 0.550 16.66 0.408 0.427 18.72 0.451 0.390 18.17 0.446 0.382

Tank and Temples

3DGS 9.57 0.108 0.779 10.15 0.118 0.763 11.48 0.204 0.685 12.50 0.202 0.669 SparseNeRF 9.23 0.191 0.632 9.55 0.216 0.633 12.24 0.274 0.615 12.74 0.294 0.608 DNGaussian 10.23 0.156 0.643 11.25 0.204 0.584 12.92 0.231 0.535 13.01 0.256 0.520

- ReconX (Ours) 14.28 0.394 0.564 15.38 0.437 0.483 16.27 0.497 0.420 18.38 0.556 0.355

DL3DV

3DGS 9.46 0.125 0.732 10.97 0.248 0.567 13.34 0.332 0.498 14.99 0.403 0.446 SparseNeRF 9.14 0.137 0.793 10.89 0.214 0.593 12.15 0.234 0.577 12.89 0.242 0.576 DNGaussian 10.10 0.149 0.523 11.10 0.274 0.577 12.65 0.330 0.548 13.46 0.367 0.541 ReconX (Ours) 13.60 0.307 0.554 14.97 0.419 0.444 17.45 0.476 0.426 18.59 0.584 0.386

ReconX(Ours)3DGSDNGaussian

[Figure 129]

[Figure 130]

[Figure 137]

[Figure 138]

[Figure 139]

Input Views Novel Views

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

3D Gaussians Rendered Views

- Fig. 5. Rendering comparison of sparse-view 3D scene reconstruction with Gaussian-based methods frame by frame.

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

[Figure 198]

[Figure 199]

[Figure 200]

validate our strong generalizability, we also directly evaluate our method on the DTU [64], NeRF-LLFF [65], and more challenging outdoor datasets Mip-NeRF 360 [66] and Tank-

and-Temples dataset [67]. For DTU, NeRF-LLFF and Tankand-Templates datasets, we select the training views evenly from all the frames and use every 8th of the remaining frames

TABLE IV QUANTITATIVE COMPARISONS WITH MORE MULTI-VIEWS RECONSTRUCTION METHODS ON MIPNERF 360. WE EVALUATE THE RECONSTRUCTION PERFORMANCE WITH DIFFERENT INPUT VIEWS FOR EACH SCENE.

3-view 6-view 9-view PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

Method

Zip-NeRF 12.77 0.271 0.705 13.61 0.284 0.663 14.30 0.312 0.633 ZeroNVS 14.44 0.316 0.680 15.51 0.337 0.663 15.99 0.350 0.655 ReconFusion 15.50 0.358 0.585 16.93 0.401 0.544 18.19 0.432 0.511 CAT3D 16.62 0.377 0.515 17.72 0.425 0.482 18.67 0.460 0.460 ReconX (Ours) 17.16 0.435 0.407 19.20 0.473 0.378 20.13 0.482 0.356

Input Views Novel Views

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

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

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

- Fig. 6. Qualitative results of our ReconX on outdoor scenes from DL3DV [61].

for evaluation. For nine scenes in Mip-NeRF 360 dataset, we manually choose a training 9-view split of views that are uniformly distributed around the hemisphere and pointed toward the central object of interest. Then we further choose the 6- and 3-view splits to be subsets of the 9-view split.

view reconstruction. Specifically, we compare with NeRFbased pixelNeRF [11] and MuRF [68]; Light Field based GPNR [69] and AttnRend [70]; and the recent state-of-the-art 3DGS-based pixelSplat [12] and MVSplat [14] in feed-forward based comparisons. On the other hand, we compare with SparseNeRF [71], original 3DGS [2], and DNGaussian [72] for per-scene optimization comparisons. Furthermore, we qualitatively compare our method with more recent works CAT3D [10] and ReconFusion [15] that incorporate generative power. For quantitative results, we report the standard metrics

Baselines and Metrics. To comprehensively demonstrate our strong capability in sparse-view reconstruction, we compare our ReconX with (a) feed-forward based methods trained from 3D scenes to learn 3D prior and (b) per-scene optimization based methods with specific priors (e.g., , depth) for sparse-

###### ReconX (Ours)

[Figure 261]

Input image1 Interpolation Input image2 Extrapolation

DNGaussian

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

- Iteration 1
- Iteration 2

3DGS

9

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

Input Views Novel Views

View Interpolation View Extrapolation

Input image 1 Input image 2

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Select image 1 View Interpolation Select image 2 View Extrapolation

### ···

···

Input image1

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

Select image 1 View Interpolation Select image 2 View Extrapolation

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

### ···

Iteration N

Select image 1 View Interpolation Input image1

###### ···

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

Output Video

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

···

Fig. 8. The incremental strategy to generate full 360-degree scenes using only two initial images.

Outputs Sequences

- Fig. 7. Evaluation of extrapolation ability of ReconX. We highlight the extrapolated regions in the red boxes in the novel rendered views.

###### Input Images Generated Views

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

in NVS, including PSNR, SSIM [73], LPIPS [58].

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

- B. 3D Scene Reconstruction from Two-Views

Comparison for small angle variance in input views. For fair comparison with two-views novel view synthesis baseline methods like MuNeRF [68], pixelSplat [12], and MVSplat [14], we first compare our reconX with baseline method from sparse views with small angle variance (see Easy Set from Table I and Fig. 3). We observe that our ReconX surpasses all previous state-of-the-art models in terms of all metrics on visual quality and qualitative perception.

Comparison for large angle variance in input views. As MVSplat and pixelSplat are much better than previous baselines, we conduct thorough comparisons with them in more difficult settings. In more challenging settings (i.e., given sparse views with large angle variance), our proposed ReconX demonstrate more significant improvement than baselines, especially in unseen and generalized viewpoints (see Hard Set from Table II and Fig. 3). This clearly shows the effectiveness of ReconX in creating more consistent observations from video diffusion to mitigate the inherent ill-posed sparse-view reconstruction problem.

Cross-dataset generalization. Unleashing the strong generative power of the video diffusion model through 3D structure condition, our ReconX is inherently superior in generalizing to out-of-distribution novel scenes. To demonstrate the strong generalizability of ReconX, we conduct two cross-dataset evaluations. For a fair comparison, we train the models solely on the RealEstate10K and directly test them on two popular NVS datasets (i.e., NeRF-LLFF [65] and DTU [64]). As shown in Cross Set from Table II and Fig. 3, the competitive baseline methods MVSplat [14] and pixelSplat [12] fail to render such OOD datasets which contain different camera distributions and image appearance, leading to dramatic performance degradation. In contrast, our ReconX shows impressive generalizability and the gain is larger when the domain gap from training and test data becomes larger.

- C. 3D Scene Reconstruction from Multi-Views

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

Fig. 9. Qualitative results of full 360-degree scenes. This incremental approach demonstrates the effectiveness of our ReconX in reconstructing expansive scenes with only two input views.

III and more visual comparisons in Fig. 4. We observe that our method outperforms all the other per-scene optimization baselines in PSNR, SSIM, and LPIPS scores. As shown in Fig. 4, we find that the baselines produce extremely blurry results in only two view settings with noisy camera estimations. In contrast, by unleashing the generative power of the video diffusion model, our ReconX can create more observations from only two sparse views and ensures high-quality novel view rendering, avoiding local minima issues.

To further demonstrate our superiority, we compare in even with recent works like CAT3D [10] and ReconFusion [15] that incorporate generative prior to mitigate ill-posed sparse view reconstruction. As the data is open-sourced in ReconFusion [15], we conduct an additional quantitative experiment in comparison with ZipNeRF [74], ZeroNVS [42], CAT3D [10], and ReconFusion [15]. It is worth noting that the data split used in CAT3D [10] follows a heuristic loss [10] to encourage reasonable camera spacing and coverage of the central object. We observe that our ReconX is better than all baselines in Table V-A.

To verify the capability of ReconX in sparse-view (more than two views) reconstruction in more challenging outdoor settings, we compare with multi-views reconstruction methods in different input views (i.e., , 2, 3, 6, and 9 views) in Table

Regarding the DL3DV dataset, we trained our model on this to demonstrate its performance on outdoor scenes. Due

Input Views Ground Truth ReconX (Ours) base w/o 3D structure condition w/o confidence-aware opt. w/o LPIPS loss

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

- Fig. 10. Visualization results of ablation study. We ablate the design choices of 3D structure guidance, confidence-aware optimization, and the LPIPS loss.

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

Reconx

Rendering Dust3Rinit+GS

Renderss

- Fig. 11. Visualization results on the impact of video diffusion. We ablate the impact of video diffusion in improving the reconstruction result of DUSt3R.

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

TABLE V QUANTITATIVE RESULTS OF ABLATION STUDY. WE REPORT THE QUANTITATIVE METRICS IN ABLATIONS OF OUR FRAMEWORK IN REAL-WORLD DATA [62].

Video diffusion Structure cond. DUSt3R init. Conf-aware opt. LPIPS loss PSNR↑ SSIM↑ LPIPS↓

- - ✓ - - 17.34 0.527 0.259 ✓ - ✓ - - 19.70 0.789 0.229 ✓ - ✓ ✓ ✓ 25.13 0.901 0.131 ✓ ✓ - ✓ ✓ 27.11 0.908 0.113 ✓ ✓ ✓ - ✓ 27.83 0.897 0.097 ✓ ✓ ✓ ✓ - 27.47 0.906 0.111 ✓ ✓ ✓ ✓ ✓ 28.31 0.912 0.088

to the limitations of feed-forward methods on this dataset, we did not present quantitative results in the main paper, as these methods fail on it. However, to highlight our model’s strengths in outdoor environments, we have included visual results in the supplementary video and have added comparisons with per-scene optimization methods in Table III. We have also provided more visual results on DL3DV in Fig. 6. We also compare our ReconX in 3D Gaussians with frame-by-frame results in Fig. 5.

- D. Evaluation of Extrapolation Ability

As we use a pair of input views in our method, it is worthy to note that if the angular difference between the two views is too large, it is hard to ensure that the entire interpolated region falls within the visible perspective of the input views, which requires the extrapolation ability. We have evaluated it in our generalizable experiments with DTU dataset. For instance, in the case of DTU in Fig. 3, we cannot see the

roof area from the input views, while our ReconX is able to extrapolate and generate the red and yellow roof with 3D structure-guided generative prior. To further demonstrate the extrapolation capability of our method, we conduct a specific experiment in Fig. 7. This experiment selects two views with large angular spans and highlights the extrapolated regions in the red boxes in the novel-rendered views. This emphasizes our model’s generative power to extrapolate unseen regions and extend beyond the visible input views.

As the position of our conditional images in ReconX is inherently flexible, allowing us to unleash more extrapolation capability by adjusting the placement of the conditional images. To further investigate the generative capabilities of our framework and demonstrate its extrapolation potential, we conduct experiments by conditioning on the first and an intermediate frame of the target video with a new tuning version of the video diffusion model in ReconX by only moving the last frame to the intermediate position. In this setup, frames

between the first and intermediate images correspond to view interpolation, while frames beyond the intermediate image correspond to extrapolation.

- • View interpolation can not only synthesize visible areas between the input images but also generate previously unseen regions caused by occlusions.
- • View extrapolation continues along the camera’s motion trajectory, generating entirely new content not present in the input images, such as unseen objects and expanded scene regions.

Such extrapolation ability allows us to even recover a 360degree scene from only two sparse views. Specifically, we adopt an incremental generation approach shown in Fig. 8. Given two initial input images (i.e., input image 1 and input image 2 in Fig. 8), we first generate a video sequence divided into four segments: input image 1, view interpolation, input image 2, and view extrapolation. From this generated frame sequence, we select two images—one from the interpolation part and another one from the extrapolation part. These two images function as a sliding window, and repeat the generation process, progressively advancing with each iteration. This approach allows our framework to autoregressively generate a much longer 360-degree panoramic sequence while maintaining a limited-length video frame window. In the final iteration, we select one image from the video generated in the previous step (i.e., select image 1 in Fig. 8) and pair it with the original first input image (i.e., input image 1 in Fig. 8) as the input pair. This ensures a seamless connection back to the starting view, completing a full 360-degree scene reconstruction shown in Fig. 9. This incremental approach demonstrates the strong generative extrapolation potential of our method.

- E. Ablation Study and Analysis

We carry out ablation studies on RealEstate10K to analyze the design of our ReconX framework in Fig. 10 and extends the ablation to contain all different combinations of leaving out individual components in Table V. A naive combination of pretrained video diffusion model and Gaussian Splatting is regarded as the “base”. Specifically, we ablate on the following aspects of our method: 3D structure condition, DUSt3R initialization, confidence-aware optimization, and LPIPS loss. The results indicate that the omission of any of these elements leads to a degradation in terms of quality and consistency. Notably, the basic combination of original video diffusion model and 3DGS leads to significant distortion of the scene. The absence of 3D structure condition causes inconsistent generated frames especially in distant input views, resulting in blur and artifact issues. The lack of confidence-aware optimization leads to suboptimal results in some local detail areas. Adding LPIPS loss in confidence-aware optimization would provide clearer rendering views. This illustrates the effectiveness of our overall framework, which drives generalizable and high-fidelity 3D reconstruction given only sparse views as input.

Moreover, we ablate the impact of DUSt3R and video diffusion priors in Fig. 11. Although the point cloud may not include enough high-quality information, such coarse 3D structure is sufficient to guide the video diffusion in our ReconX to fill in the distortions, occlusions, or missing regions.

This demonstrates that our ReconX has learned a comprehensive understanding of the 3D scene and can generate highquality novel views from imperfect conditional information and exhibit robustness to the point cloud conditions.

VI. CONCLUSION

In this paper, we introduce ReconX, a novel sparse-view 3D reconstruction framework that reformulates the inherently ambiguous reconstruction problem as a generation problem. The key to our success is that we unleash the strong prior of video diffusion models to create more plausible observations frames for sparse-view reconstruction. Grounded by the empirical study and theoretical analysis, we propose to incorporate 3D structure guidance into the video diffusion process for better 3D consistent video frames generation. What’s more, we propose a 3D confidence-aware scheme to optimize the final 3DGS from generated frames, which effectively addresses the uncertainty issue. Extensive experiments demonstrate the superiority of our ReconX over the latest state-of-the-art methods in terms of high quality and strong generalizability in unseen data. We believe that ReconX provides a promising research direction to craft intricate 3D worlds from video diffusion models and hope it will inspire more works in the future.

VII. APPENDIX A. Theoretical Proof

Proposition 1. Let θ∗,ψ∗ = g∗ be the optimal solution of the solely image-based conditional diffusion scheme and θ˜∗,ψ˜∗ = {g∗,F∗} be the optimal solution of diffusion scheme with native 3D prior. Suppose the divergence D is convex and the embedding function space Ψ includes all measurable functions, we have D(q (x)∥pθ˜∗,ψ˜∗(x)) < D (q (x)∥pθ∗,ψ∗(x)).

Proof. According to the convexity of D and Jensen’s inequality D(E[X]) ≤ E[D(X)], where X is a random variable, we have:

D q(x)∥pθ˜∗,ψ˜∗(x) = D Eq(s)q(x|s)∥Eq(s)pθ˜∗,ψ˜∗(x|s) ≤ Eq(s)D q(x|s)∥pθ˜∗,ψ˜∗(x|s)

= Eq(s)D q(x|s)∥pθ˜∗,g∗,F∗(x|s) ,

(15) where we incorporate an intermediate variable s, which represents a specific scene. q(x|s) indicates the conditional distribution of rendering data x given the specific scene s. According to the definition of θ˜∗,g∗,F∗, we have:

Eq(s) = Eq(s)D q(x|s)∥pθ˜∗,g∗,F∗(x|s)

Eq(s)D (q(x|s)∥pθ,g,F(x|s))

= min

θ,g,F

(16)

Eq(s) min

D q(x|s)∥pθ,g(s),F(s)(x)

= min

θ

g(s),F(s)

Eq(s) min

D (q(x|s)∥pθ,g,E(x)),

= min

g,E

θ

where E is the general 3D encoder in 3D structure conditional scheme while it is a redundant embedding in solely image-

based conditional scheme, i.e., ψ = {g,E(∅)}. Combining Equation 15 and 16, we have:

- [12] D. Charatan, S. L. Li, A. Tagliasacchi, and V. Sitzmann, “pixelsplat: 3D gaussian splats from image pairs for scalable generalizable 3D reconstruction,” in CVPR, 2024, pp. 19457–19467. 1, 2, 3, 4, 6, 8, 9
- [13] S. Szymanowicz, C. Rupprecht, and A. Vedaldi, “Splatter image: Ultrafast single-view 3D reconstruction,” in CVPR, 2024, pp. 10208–10217. 1, 2
- [14] Y. Chen, H. Xu, C. Zheng, B. Zhuang, M. Pollefeys, A. Geiger, T.-J. Cham, and J. Cai, “Mvsplat: Efficient 3D gaussian splatting from sparse multi-view images,” arXiv preprint arXiv:2403.14627, 2024. 1, 2, 3, 4, 6, 8, 9
- [15] R. Wu, B. Mildenhall, P. Henzler, K. Park, R. Gao, D. Watson, P. P. Srinivasan, D. Verbin, J. T. Barron, B. Poole et al., “Reconfusion: 3D reconstruction with diffusion priors,” in CVPR, 2024, pp. 21551–21561. 1, 2, 4, 8, 9
- [16] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts et al., “Stable video diffusion: Scaling latent video diffusion models to large datasets,” arXiv preprint arXiv:2311.15127, 2023. 1, 2, 4
- [17] A. Blattmann, R. Rombach, H. Ling, T. Dockhorn, S. W. Kim, S. Fidler, and K. Kreis, “Align your latents: High-resolution video synthesis with latent diffusion models,” in CVPR, 2023, pp. 22563–22575. 1
- [18] J. Xing, M. Xia, Y. Zhang, H. Chen, X. Wang, T.-T. Wong, and Y. Shan, “Dynamicrafter: Animating open-domain images with video diffusion priors,” arXiv preprint arXiv:2310.12190, 2023. 1, 2, 4, 6
- [19] J. L. Sch¨onberger and J.-M. Frahm, “Structure-from-Motion Revisited,” in CVPR, 2016. 2
- [20] J. Yang, M. Pavone, and Y. Wang, “Freenerf: Improving few-shot neural rendering with free frequency regularization,” in CVPR, 2023, pp. 8254–

8263. 2

- [21] Z. Zhu, Z. Fan, Y. Jiang, and Z. Wang, “Fsgs: Real-time few-shot view synthesis using gaussian splatting,” arXiv preprint arXiv:2312.00451,

2023. 2

- [22] H. Xiong, S. Muttukuru, R. Upadhyay, P. Chari, and A. Kadambi, “Sparsegs: Real-time 360 {\deg} sparse view synthesis using gaussian splatting,” arXiv preprint arXiv:2312.00206, 2023. 2
- [23] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “Dust3r: Geometric 3D vision made easy,” in CVPR, 2024, pp. 20697–20709. 2, 3, 4, 5, 6
- [24] Z. Fan, W. Cong, K. Wen, K. Wang, J. Zhang, X. Ding, D. Xu, B. Ivanovic, M. Pavone, G. Pavlakos et al., “Instantsplat: Unbounded sparse-view pose-free gaussian splatting in 40 seconds,” arXiv preprint arXiv:2403.20309, 2024. 2, 5
- [25] S. Shen, “Accurate multiple view 3D reconstruction using patch-based stereo for large-scale scenes,” TIP, vol. 22, no. 5, pp. 1901–1914, 2013. 2
- [26] K. Wang, G. Zhang, and H. Bao, “Robust 3D reconstruction with an RGB-D camera,” TIP, vol. 23, no. 11, pp. 4893–4906, 2014. 2
- [27] L. Jiang, J. Zhang, B. Deng, H. Li, and L. Liu, “3D face reconstruction with geometry details from a single image,” TIP, vol. 27, no. 10, pp. 4756–4770, 2018. 2
- [28] M. Chen, L. Wang, Y. Lei, Z. Dong, and Y. Guo, “Learning spherical radiance field for efficient 360 unbounded novel view synthesis,” TIP,

2024. 2

- [29] C. Huang, Y. Hou, W. Ye, D. Huang, X. Huang, B. Lin, and D. Cai, “Nerf-det++: Incorporating semantic cues and perspective-aware depth supervision for indoor multi-view 3D detection,” TIP, 2025. 2
- [30] Y. Wang, X. Wei, M. Lu, and G. Kang, “Plgs: Robust panoptic lifting with 3D gaussian splatting,” TIP, 2025. 2
- [31] C. Wewer, K. Raj, E. Ilg, B. Schiele, and J. E. Lenssen, “latentsplat: Autoencoding variational gaussians for fast generalizable 3D reconstruction,” arXiv preprint arXiv:2403.16292, 2024. 2
- [32] S. Szymanowicz, E. Insafutdinov, C. Zheng, D. Campbell, J. F. Henriques, C. Rupprecht, and A. Vedaldi, “Flash3D: Feed-forward generalisable 3D scene reconstruction from a single image,” arXiv preprint arXiv:2406.04343, 2024. 2, 3
- [33] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in CVPR, 2022, pp. 10684–10695. 2, 3
- [34] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans et al., “Photorealistic text-to-image diffusion models with deep language understanding,” NeurIPS, vol. 35, pp. 36479–36494, 2022. 2
- [35] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, vol. 1, no. 2, p. 3, 2022. 2

Eq(s) min

D q(x)∥pθ˜∗,ψ˜∗(x) ≤ min

D (q(x|s)∥pθ,g,E(x)) < min

g,E

θ

D (q(x)∥pθ,g,E(x))

θ,g,E

D q(x)∥pθ,g,E(∅)(x)

= min

θ,g,E(∅)

D (q(x)∥pθ,ψ(x))

= min

θ,ψ

= D (q (x)∥pθ∗,ψ∗(x)).

(17) The second inequality holds because given general real-world scene s in any parameter θ ∈ Θ, approximating q(x|s) is simpler than q(x) by only tuning the encoder E of pθ,g,E1, i.e., minE D (q(x|s)∥pθ,g,E(x)) < minE D (q(x)∥pθ,g,E(x)) holds almost everywhere (a.e.), representing Pq(s) {minE D (q(x|s)∥pθ,g,E(x)) < minE D (q(x)∥pθ,g,E(x))}

is equal to 1. Consequently, the proof of Proposition 1 has been done.

ACKNOWLEDGMENT

This work was supported in part by the National Natural Science Foundation of China under Grant 62206147.

REFERENCES

- [1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” in ECCV. Springer, 2020, pp. 405–421. 1, 2
- [2] B. Kerbl, G. Kopanas, T. Leimk¨uhler, and G. Drettakis, “3D gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics, vol. 42, no. 4, July 2023. [Online]. Available: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ 1, 2, 3, 6, 8
- [3] A. Dalal, D. Hagen, K. G. Robbersmyr, and K. M. Knausg˚ard, “Gaussian splatting: 3D reconstruction and novel view synthesis, a review,” IEEE Access, 2024. 1
- [4] M. Adamkiewicz, T. Chen, A. Caccavale, R. Gardner, P. Culbertson, J. Bohg, and M. Schwager, “Vision-only robot navigation in a neural radiance world,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 4606–4613, 2022. 1
- [5] R. Martin-Brualla, N. Radwan, M. S. Sajjadi, J. T. Barron, A. Dosovitskiy, and D. Duckworth, “Nerf in the wild: Neural radiance fields for unconstrained photo collections,” in CVPR, 2021, pp. 7210–7219. 1
- [6] X. Yang, G. Lin, and L. Zhou, “Single-view 3d mesh reconstruction for seen and unseen categories,” TIP, vol. 32, pp. 3746–3758, 2023. 1
- [7] F. Liu, H. Wang, W. Chen, H. Sun, and Y. Duan, “Make-your-3d: Fast and consistent subject-driven 3d content generation,” in ECCV. Springer, 2024, pp. 389–406. 1
- [8] K. Wu, F. Liu, Z. Cai, R. Yan, H. Wang, Y. Hu, Y. Duan, and K. Ma, “Unique3d: High-quality and efficient 3d mesh generation from a single image,” in NeurIPS, 2024. 1, 2
- [9] C. Zhang, J. Yan, Y. Wei, J. Li, L. Liu, Y. Tang, Y. Duan, and J. Lu, “Occnerf: Advancing 3d occupancy prediction in lidar-free environments,” TIP, vol. 34, pp. 3096–3107, 2025. 1
- [10] R. Gao, A. Holynski, P. Henzler, A. Brussee, R. Martin-Brualla, P. Srinivasan, J. T. Barron, and B. Poole, “Cat3D: Create anything in 3D with multi-view diffusion models,” arXiv preprint arXiv:2405.10314, 2024. 1, 2, 8, 9
- [11] A. Yu, V. Ye, M. Tancik, and A. Kanazawa, “pixelnerf: Neural radiance fields from one or few images,” in CVPR, 2021, pp. 4578–4587. 1, 2, 3, 8

1A simple verifiable case is to optimize the parameters of 3DGS by only 2D images (solely image-based conditional learning) or using a SFM initialization from collected images (native 3D conditional learning) before optimization. The latter provides a more constrained and optimal solution space.

- [36] Y. Jing, W. Wang, L. Wang, and T. Tan, “Learning aligned image-text representations using graph attentive relational network,” TIP, vol. 30, pp. 1840–1852, 2021. 2
- [37] C.-H. Lin, J. Gao, L. Tang, T. Takikawa, X. Zeng, X. Huang, K. Kreis, S. Fidler, M.-Y. Liu, and T.-Y. Lin, “Magic3D: High-resolution text-to3D content creation,” in CVPR, 2023, pp. 300–309. 2
- [38] F. Liu, D. Wu, Y. Wei, Y. Rao, and Y. Duan, “Sherpa3D: Boosting highfidelity text-to-3D generation via coarse 3D prior,” in CVPR, 2024, pp. 20763–20774. 2
- [39] Z. Wang, C. Lu, Y. Wang, F. Bao, C. Li, H. Su, and J. Zhu, “Prolificdreamer: High-fidelity and diverse text-to-3D generation with variational score distillation,” NeurIPS, vol. 36, 2024. 2
- [40] Y. Shi, P. Wang, J. Ye, M. Long, K. Li, and X. Yang, “Mvdream: Multiview diffusion for 3D generation,” arXiv preprint arXiv:2308.16512,

2023. 2

- [41] Y. Liu, C. Lin, Z. Zeng, X. Long, L. Liu, T. Komura, and W. Wang, “Syncdreamer: Generating multiview-consistent images from a singleview image,” arXiv preprint arXiv:2309.03453, 2023. 2
- [42] K. Sargent, Z. Li, T. Shah, C. Herrmann, H.-X. Yu, Y. Zhang, E. R. Chan, D. Lagun, L. Fei-Fei, D. Sun et al., “Zeronvs: Zero-shot 360-degree view synthesis from a single real image,” arXiv preprint arXiv:2310.17994,

2023. 2, 9

- [43] E. R. Chan, K. Nagano, M. A. Chan, A. W. Bergman, J. J. Park, A. Levy, M. Aittala, S. De Mello, T. Karras, and G. Wetzstein, “Generative novel view synthesis with 3D-aware diffusion models,” in ICCV, 2023, pp. 4217–4229. 2
- [44] F. Liu, H. Wang, S. Yao, S. Zhang, J. Zhou, and Y. Duan, “Physics3D: Learning physical properties of 3D gaussians via video diffusion,” arXiv preprint arXiv:2406.04338, 2024. 2
- [45] V. Voleti, C.-H. Yao, M. Boss, A. Letts, D. Pankratz, D. Tochilkin, C. Laforte, R. Rombach, and V. Jampani, “Sv3d: Novel multi-view synthesis and 3D generation from a single image using latent video diffusion,” arXiv preprint arXiv:2403.12008, 2024. 2, 4
- [46] Z. Chen, Y. Wang, F. Wang, Z. Wang, and H. Liu, “V3d: Video diffusion models are effective 3D generators,” arXiv preprint arXiv:2403.06738,

2024. 2

- [47] Z. Wang, Z. Yuan, X. Wang, Y. Li, T. Chen, M. Xia, P. Luo, and Y. Shan, “Motionctrl: A unified and flexible motion controller for video generation,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–11. 2
- [48] H. He, Y. Xu, Y. Guo, G. Wetzstein, B. Dai, H. Li, and C. Yang, “Cameractrl: Enabling camera control for text-to-video generation,” arXiv preprint arXiv:2404.02101, 2024. 2
- [49] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” NeurIPS, vol. 33, pp. 6840–6851, 2020. 2
- [50] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” arXiv preprint arXiv:2011.13456, 2020. 2
- [51] R. Ramamoorthi and P. Hanrahan, “An efficient representation for irradiance environment maps,” in Proceedings of the 28th annual conference on Computer graphics and interactive techniques, 2001, pp. 497–500. 3
- [52] M. Zwicker, H. Pfister, J. Van Baar, and M. Gross, “Surface splatting,” in Proceedings of the 28th annual conference on Computer graphics and interactive techniques, 2001, pp. 371–378. 3
- [53] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML. PMLR, 2021, pp. 8748–8763. 4
- [54] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in CVPR, 2018, pp. 586–595. 4
- [55] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” arXiv preprint arXiv:2207.12598, 2022. 5, 6
- [56] R. Martin-Brualla, N. Radwan, M. S. Sajjadi, J. T. Barron, A. Dosovitskiy, and D. Duckworth, “Nerf in the wild: Neural radiance fields for unconstrained photo collections,” in CVPR, 2021, pp. 7210–7219. 5
- [57] W. Ren, Z. Zhu, B. Sun, J. Chen, M. Pollefeys, and S. Peng, “Nerf on-the-go: Exploiting uncertainty for distractor-free nerfs in the wild,” in CVPR, 2024, pp. 8931–8940. 5
- [58] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in CVPR, 2018, pp. 586–595. 5, 9
- [59] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2017. 6
- [60] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” 2022. [Online]. Available: https://arxiv.org/abs/2010.02502 6

- [61] L. Ling, Y. Sheng, Z. Tu, W. Zhao, C. Xin, K. Wan, L. Yu, Q. Guo, Z. Yu, Y. Lu et al., “Dl3dv-10k: A large-scale scene dataset for deep learning-based 3D vision,” in CVPR, 2024, pp. 22160–22169. 6, 8
- [62] T. Zhou, R. Tucker, J. Flynn, G. Fyffe, and N. Snavely, “Stereo magnification: Learning view synthesis using multiplane images,” ACM Trans. Graph. (Proc. SIGGRAPH), vol. 37, 2018. [Online]. Available: https://arxiv.org/abs/1805.09817 6, 10
- [63] A. Liu, R. Tucker, V. Jampani, A. Makadia, N. Snavely, and A. Kanazawa, “Infinite nature: Perpetual view generation of natural scenes from a single image,” in ICCV, 2021. 6
- [64] R. Jensen, A. Dahl, G. Vogiatzis, E. Tola, and H. Aanæs, “Large scale multi-view stereopsis evaluation,” in CVPR, 2014, pp. 406–413. 7, 9
- [65] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari, R. Ramamoorthi, R. Ng, and A. Kar, “Local light field fusion: Practical view synthesis with prescriptive sampling guidelines,” 2019. [Online]. Available: https://arxiv.org/abs/1905.00889 7, 9
- [66] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in CVPR, 2022, pp. 5470–5479. 7
- [67] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples: Benchmarking large-scale scene reconstruction,” ACM Transactions on Graphics (ToG), vol. 36, no. 4, pp. 1–13, 2017. 7
- [68] H. Xu, A. Chen, Y. Chen, C. Sakaridis, Y. Zhang, M. Pollefeys, A. Geiger, and F. Yu, “Murf: Multi-baseline radiance fields,” in CVPR,

2024. 8, 9

- [69] M. Suhail, C. Esteves, L. Sigal, and A. Makadia, “Generalizable patchbased neural rendering,” in ECCV, 2022. 8
- [70] Y. Du, C. Smith, A. Tewari, and V. Sitzmann, “Learning to render novel views from wide-baseline stereo pairs,” in CVPR, 2023. 8
- [71] G. Wang, Z. Chen, C. C. Loy, and Z. Liu, “Sparsenerf: Distilling depth ranking for few-shot novel view synthesis,” in ICCV, 2023, pp. 9065–

9076. 8

- [72] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, and L. Gu, “Dngaussian: Optimizing sparse-view 3D gaussian radiance fields with global-local depth normalization,” in CVPR, 2024, pp. 20775–20785. 8
- [73] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” TIP, vol. 13, no. 4, pp. 600–612, 2004. 9
- [74] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Zip-nerf: Anti-aliased grid-based neural radiance fields,” in ICCV, 2023, pp. 19697–19705. 9

