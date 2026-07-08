# arXiv:2411.08033v2[cs.CV]10Apr2025

## GAUSSIANANYTHING: INTERACTIVE POINT CLOUD FLOW MATCHING FOR 3D OBJECT GENERATION

Yushi Lan†, Shangchen Zhou†, Zhaoyang Lyuα, Fangzhou Hong†, Shuai Yangβ, Bo Daiγ, Xingang Pan†, Chen Change Loy† †S-Lab, Nanyang Technological University, Singapore

αShanghai Artificial Intelligence Laboratory, βWICT, Peking University γThe University of Hong Kong https://nirvanalan.github.io/projects/GA/

ABSTRACT

While 3D content generation has advanced significantly, existing methods still face challenges with input formats, latent space design, and output representations. This paper introduces a novel 3D generation framework that addresses these challenges, offering scalable, high-quality 3D generation with an interactive Point Cloud-structured Latent space. Our framework employs a Variational Autoencoder (VAE) with multi-view posed RGB-D(epth)-N(ormal) renderings as input, using a unique latent space design that preserves 3D shape information, and incorporates a cascaded latent flow-based model for improved shape-texture disentanglement. The proposed method, GAUSSIANANYTHING, supports multi-modal conditional 3D generation, allowing for point cloud, caption, and single image inputs. Notably, the newly proposed latent space naturally enables geometrytexture disentanglement, thus allowing 3D-aware editing. Experimental results demonstrate the effectiveness of our approach on multiple datasets, outperforming existing native 3D methods in both text- and image-conditioned 3D generation.

1 INTRODUCTION

3D content generation holds great potential for transforming the virtual reality, film, and gaming industries. Current approaches typically follow one of two paths: either a 2D-lifting method or the design of native 3D diffusion models. While the 2D-lifting approach (Shi et al., 2023b; Liu

- et al., 2023b) benefits from leveraging 2D diffusion model priors, it is often hindered by expensive optimization, the Janus problem, and inconsistencies between views. In contrast, native 3D diffusion models (Jun & Nichol, 2023; Lan et al., 2024a; Zhang et al., 2024b) are trained from scratch for 3D generation, offering improved generality, efficiency, and control.

Despite the progress in native 3D diffusion models, several design challenges still persist: (1) Input format to the 3D VAE. Most methods (Zhang et al., 2024b; Li et al., 2024) directly adopt point cloud as input. However, it fails to encode the high-frequency details from textures. Besides, this limits the available training dataset to artist-created 3D assets, which are challenging to collect on a large scale. LN3Diff (Lan et al., 2024a) adopt multi-view images as input. Though straightforward, it lacks direct 3D information input and cannot comprehensively encode the given object. (2) 3D latent space structure. Since 3D objects are diverse in geometry, color, and size, most 3D VAE models adopt the permutation-invariant set latent (Zhang et al., 2023; Sajjadi et al., 2022; Zhang

- et al., 2024b) to encode incoming 3D objects. Though flexible, this design lacks the image-latent correspondence as in Stable Diffusion VAE (Rombach et al., 2022), where the VAE latent code can directly serve as the proxy for editing input image (Mou et al., 2023b;a). Other methods adopt latent tri-plane (Wu et al., 2024; Lan et al., 2024a) as the 3D latent representation. However, the latent triplane is still unsuitable for interactive 3D editing as changes in one plane may not map to the exact part of the objects that need editing. (3) Choice of 3D output representations. Existing solutions either output texture-less SDF (Wu et al., 2024; Zhang et al., 2024b), which requires additional shading model for post-processing; or volumetric tri-plane (Lan et al., 2024a), which struggles with high-resolution rendering due to extensive memory required by volumetric rendering (Mildenhall

- et al., 2020).

In this study, we propose a novel 3D generation framework that resolves the problems above and enables scalable, high-quality 3D generation with an interactive Point Cloud-structured Latent space. The resulting method, dubbed GAUSSIANANYTHING, supports multi-modal conditional 3D generation, including point cloud, caption, and image. Specifically, we propose a 3D VAE that adopts multi-view posed RGB-D(epth)-N(ormal) renderings as the input, which are easy to render and contain comprehensive 3D attributes corresponding to the input 3D object. The information of each input view is channel-wise concatenated and efficiently encoded with the scene representation transformer (Sajjadi et al., 2022), yielding a set latent (Zhang et al., 2023) that compactly encodes the given 3D input. Instead of directly applying it for diffusion learning (Zhang et al., 2024b; Li et al., 2024), our novel design concretizes the unordered tokens into the shape of the 3D input. Specifically, this is achieved by cross-attending (Huang et al., 2024b) the set latent via a sparse point cloud sampled from the input 3D shape, as visualized in Fig. 1. The resulting point-cloud structured latent space significantly facilitate shape-texture disentanglement and 3D editing. Afterward, a DiTbased 3D decoder (Peebles & Xie, 2023; Lan et al., 2024a) gradually decodes and upsamples the latent point cloud into a set of dense surfel Gaussians (Huang et al., 2024a), which are rasterized to high-resolution renderings to supervise 3D VAE training.

After the 3D VAE is trained, we conduct cascaded latent diffusion modeling on the latent space through flow matching (Albergo et al., 2023; Lipman et al., 2023; Liu et al., 2023c) using the DiT (Peebles & Xie, 2023) framework. To encourage better shape-texture disentanglement, a point cloud diffusion model is first trained to carve the overall layout of the input shape. Then, a pointcloud feature diffusion model is cascaded to output the corresponding feature conditioned on the generated point cloud. The generated featured point cloud is then decoded into surfel Gaussians via pre-trained VAE for downstream applications.

In summary, we contribute a comprehensive 3D generation framework with a point cloud-structured 3D latent space. The redesigned 3D VAE efficiently encodes the 3D input into an interactive latent space, which is further decoded into high-quality surfel Gaussians. The diffusion models trained on the compressed latent space have shown superior performance in text-conditioned 3D generation and editing, as well as impressive image-conditioned 3D generation on general real world data.

- 2 RELATED WORK
- 3D Generation via 2D Diffusion Models. The success of 2D diffusion models (Song et al., 2021; Ho et al., 2020) has inspired their application to 3D generation. Score distillation sampling (Poole et al., 2022; Wang et al., 2023) distills 3D from a 2D diffusion model, but faces challenges like expensive optimization, mode collapse, and the Janus problem. More recent methods propose learning the 3D via a two-stage pipeline: multi-view images generation (Shi et al., 2023b; Long et al., 2024; Shi et al., 2023a) and feed-forward 3D reconstruction and editing (Hong et al., 2024b; Xu et al., 2024a; Tang et al., 2024; Xu et al., 2024b; Chen et al., 2024b). However, their performance is bounded by the multi-view generation results, which usually violate view consistency (Liu et al.,

- 2023b) and fail to scale up to higher resolution (Shi et al., 2023a). Moreover, this two-stage pipeline limits the 3D editing capability due to the lack of a 3D-aware latent space.

Native 3D Diffusion Models. Native 3D diffusion models (Zhang et al., 2023; Zeng et al., 2022; Zhang et al., 2024b; Lan et al., 2024a; Li et al., 2024) are recently proposed to achieve high-quality, efficient and scalable 3D generation. A native 3D diffusion pipeline involves a two-stage training process: encoding 3D objects into the VAE latent space (Kingma & Welling, 2013; Kosiorek et al., 2021), and latent diffusion model on the corresponding latent codes. Though straightforward, existing methods differ in VAE input formats, latent space structure and output 3D representations. While most methods adopt point alone as the VAE input (Zhang et al., 2023; 2024b; Li et al., 2024), our proposed method encodes a hybrid 3D information through convolutional encoder. Moreover, comparing to the latent set (Zhang et al., 2023; Sajjadi et al., 2022) representation, our proposed method adopts a point cloud-structured latent space, which can be directly used for interactive 3D editing. Besides, rather than producing textureless SDF, our method directly decodes the 3D latent codes into high-quality surfel Gaussians (Huang et al., 2024a), which can be directly used for efficient rendering.

### Point-based Shape Representation and Rendering. The proliferation of 3D scanners and RGB-

- D cameras makes the capture and processing of 3D point clouds commonplace (Gross & Pfister,

2011). In the era of deep learning, learning-based methods are emerging for point set processing (Qi et al., 2016; Zhao et al., 2021), up-sampling (Yu et al., 2018), shape representation (Genova

- et al., 2020; Lan et al., 2024b), and rendering (Pfister et al., 2000; Yifan et al., 2019; Lassner & Zollh¨ofer, 2021; Xu et al., 2022; Kerbl et al., 2023). Moreover, given its affinity for modern network architectures (Huang et al., 2024b; Zhao et al., 2021), more explicit nature against other 3D representations (Chan et al., 2022; Mildenhall et al., 2020; M¨uller et al., 2022), efficient rendering (Kerbl et al., 2023), and even high-quality surface modeling (Huang et al., 2024a), point-based 3D representations are rapidly developing towards the canonical 3D representation for learning 3D shapes. Thus, we choose (featured) point cloud as the representation for the 3D VAE latent space, and 2D Gaussians (Huang et al., 2024a) as the output 3D representations.

Feed-forward 3D Reconstruction and View Synthesis. To bypass the per-scene optimization of NeRF, researchers have proposed learning a prior model through image-based rendering (Wang

- et al., 2021; Yu et al., 2021). However, these methods are primarily designed for view synthesis and lack explicit 3D representations. Sajjadi et al. (2022; 2023) propose Scene representation transformer (SRT) to process RGB images with Vision Transformer (Dosovitskiy et al., 2021) and infers a set-latent scene representation. Though benefiting from the flexible design, its geometryfree paradigm also fails to generate explicit 3D outputs. Recently, LRM-line of work (Hong et al.,

2024b; Tang et al., 2024; Wang et al., 2024) have proposed a feed-forward framework for generalized monocular reconstruction. However, they are still regression-based models and lack the latent space designed for generative modeling and 3D editing. Besides, they are limited to 3D reconstruction only and fail to support other modalities.

3 GAUSSIANANYTHING

This section introduces our native 3D flow-based model, which learns 3D-aware flow-based prior over the novel point-cloud structured latent space through a dedicated 3D VAE. The goal of training is to learn

- 1. An encoder Eϕ that maps a set of posed RGB-D-N images R = {Ri,...,RV }, corresponding to the given 3D object to a point-cloud structured latent z = [zx ⊕ zh];
- 2. A conditional cascaded transformer denoiser ϵhΘ(zh,t,zx,0,t,c) ◦ ϵxΘ(zx,t,t,c) to denoise the noisy latent code zt given the time step t and condition prompt c;
- 3. A decoder Dψ (including a Transformer DT and a cascaded attention-base Upsampler

DU) to map z0 to the surfel Gaussian G corresponding to the input object. Moreover, our attention-based decoding of dense surfel Gaussian also provides a novel way for efficient Gaussian prediction

Beyond the advantages shared by existing 3D LDM (Zhang et al., 2024b; Lan et al., 2024a), our design offers a flexible point-cloud structured latent space and enables interactive 3D editing.

In the following subsections, we first discuss the proposed 3D VAE with a detailed framework design in Sec 3.1. Based on that, we introduce the cascaded conditional 3D flow-based model stage in Sec. 3.2. The method overview is shown in Fig. 1.

- 3.1 POINT-CLOUD STRUCTURED 3D VAE

Unlike image and video, the 3D domain is un-uniform and represented differently for different purposes. Thus, how to encode 3D objects into the latent space for flow-based model learning plays a crucial role in the 3D generation performance. This challenge is two-fold: what 3D representations to encode, and what network architecture to process the input.

Versatile 3D Input. Instead of using dense point cloud (Zhang et al., 2024b; Li et al., 2024), we adopt multi-view posed RGB-D(epth)-N(ormal) images as input, which encode the 3D input more comprehensively and can be efficiently processed by well-established network architectures (Sajjadi

- et al., 2022; Wu et al., 2023a) in a flexible manner. Specifically, the input is a set of multi-view renderings R of a 3D object, where each rendering within the set R = (I,∆,N,π) contains thorough 3D attributes that depict the underlying 3D object from the given viewpoint: the rendered RGB

[Figure 1]

##### Published as a conference paper at ICLR 2025

[Figure 2]

FPS Sampling

Gaussians z: 𝐵𝐿 1𝐶

|𝐳| |
|---|---|
|𝐄| |
| | |

|𝐳|1|
|---|---|
|𝐄| |
| |𝑓|

Transformer Blocks

|𝐳|
|---|

|𝐳|
|---|

[Figure 3]

⊕

Gaussians z:𝐵𝐿𝐶

[Figure 4]

Learnable PE 𝐄: 𝐵𝐿 𝑓𝐶

[Figure 5]

[Figure 6]

RGB(*V)XYZ(*V)Plücker(*V)Normal(*V)

Sparse Point Cloud 𝐳

[Figure 7]

×𝑓 Gaussians

[Figure 8]

residual

×𝑓 upsampler z :𝐵(𝐿𝑓)𝐶

[Figure 9]

[Figure 10]

[Figure 11]

|CrossAttention| |
|---|---|
| | |

[Figure 12]

[Figure 13]

Multi-view Transformer

CrossAttention

[Figure 14]

[Figure 15]

[Figure 16]

CNN Encoder

Upsampler

Upsampler

Upsampler

3DDiT

𝒟

ℛ

𝒟 𝒟

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Surfel

𝒟

ℰ ℰ

[Figure 23]

Gaussians

Upsampling:

Point Cloud Latent

Set Latent

Sparse→Dense

𝐳

z ≔ 𝐳 𝐳

⊕

3D VAE Encoder ℰ 3D VAE Decoder 𝒟

### Figure 1: Pipeline of the 3D VAE of GAUSSIANANYTHING. In the 3D latent space learning stage,

our proposed 3D VAE Eϕ encodes V −views of posed RGB-D(epth)-N(ormal) renderings R into a point-cloud structured latent space. This is achieved by first processing the multi-view inputs into the un-structured set latent, which is further projected onto the 3D manifold through a cross attention block, yielding the point-cloud structured latent code z. The structured 3D latent is further decoded by a 3D-aware DiT transformer, giving the coarse Gaussian prediction. For high-quality rendering, the base Gaussian is further up-sampled by a series of cascaded upsampler DUk towards a dense Gaussian for high-resolution rasterization. The 3D VAE training objective is detailed in Eq. (5).

image I ∈ RH×W×3, depth map ∆ ∈ RH×W, normal map N ∈ RH×W×3, and the corresponding camera pose π.

To unify these 3D attributes in the same format, we further process the camera π into Pl¨ucker coordinates (Sitzmann et al., 2021) pi = (o × du,v,du,v) ∈ R6, where oi ∈ R3 is the camera origin, du,v ∈ R3 is the normalized ray direction, and × denotes the cross product. Thus, the Pl¨ucker embedding of a given camera π can be expressed as P ∈ RH×W×6. Besides, following MCC (Wu et al., 2023a), we use π to unproject the depth map into their 3D positions X ∈ RH×W×3. The resulting information is channel-wise concatenated, giving R˜ = [I ⊕ X ⊕ N ⊕ P] ∈ RH×W×(3+3+3+6=15).

Transformer-based 3D Encoding. Given the 3D renderings R, encoding them into a 3D latent space remains a significant challenge. Independently processing each input rendering R˜ with existing network architecture (Wu et al., 2023a; Dosovitskiy et al., 2021) overlooks the information from other views, leading to 3D inconsistency and content drift across views (Liu et al., 2023b).

Existing multi-view generation alleviates this issue by injecting 3D attention (Shi et al., 2023b; Tang et al., 2024; Shi et al., 2023a) into the U-Net architecture. Inspired by its effectiveness, here we directly adopt Scene Representation Transformer (SRT)-like encoder (Sajjadi et al., 2022; 2023) to process the multi-view inputs, which fully adopts 3D attention transformer block for the 3D representation learning. Specifically, the encoder first down-samples the multi-view inputs via a shared CNN backbone, and then processes the aggregated multi-view tokens through the transformer encoder (Dosovitskiy et al., 2021):

zz = EϕTX(EϕCNN({R˜)})), (1) where zz is the set latent corresponding to the 3D input. This can be seen as the full-attention version of the existing 3D attention-augmented architecture. The resulting latent codes zz fully capture the intact 3D information corresponding to the input. Compared to existing work that adopts point clouds only as input (Zhang et al., 2024b; Li et al., 2024), our proposed solution supports more 3D properties as input in a flexible way. In addition, attention operations can be well optimized in modern GPU architecture (Dao et al., 2022; Dao, 2024).

Point Cloud-structured Latent Space. Though zz fully captures the given 3D input, it is not ideal to serve as a latent space for our task due to the following limitations: 1) The latent space is

cumbersome to perform flow matching. Specifically, zz has a shape of V × (H/f) × (W/f) × C, where V is the number of input views, H,W is the input resolution and f is the down-sampling factor of the CNN backbone. Given V = 8, f = 8, and H = W = 512, the resulting latent codes will have the shape of 32768 × C. This latent space incurs a high computation cost for multi-view attention (Shi et al., 2023b). 2) The multi-view features zz are not native 3D representations and naturally suffer from view inconsistency (Liu et al., 2023b) even with enough compute available (Shi et al., 2023a). 3) Since zz is an un-structured set (Lee et al., 2019) 3D latent space (Zhang et al.,

- 2023; 2024b), it also sacrifices an explicit, editable latent space (Mou et al., 2023a) for flexibility.

Here, we resolve these issues by proposing a point cloud-structured latent space. Specifically, we project the un-structured features zz onto the sparse 3D point cloud of the input 3D shape through the cross attention layer:

zh := CrossAttn(PE(zx),zz,zz), (2)

where CrossAttn(Q,K,V ) denotes a cross attention block with query Q, key K, and value V . zx ∈ R3×N is a sparse point cloud sampled from the surface of the 3D input with Farthest Point Sampling (FPS) (Qi et al., 2017), and PE denotes positional embedding (Tancik et al., 2020). Intuitively, we define a read cross attention block (Huang et al., 2024b) that cross attends information from unstructured representation zz into the point-cloud structured feature zh ∈ RC

h×N, with Ch ≪ C. In this way, we obtain the point-cloud structured latent code z = [zx ⊕ zh] ∈ R(3+C

h)×N for flow matching.

High-quality 3D Gaussian Decoding. Given the point cloud-structured latent codes, how to decode them into high-quality 3D representation for supervision remains challenging. Though dense point cloud (Huang et al., 2024b) is a straightforward solution, it fails to depict high-quality 3D structure with limited point quantity. Here, we resort to surfel Gaussian (Huang et al., 2024a), an augmented point-based 3D representation that supports high-fidelity 3D surface modeling and efficient rendering. Specifically, our decoder first decodes the input through the 3D-DiT blocks (Peebles & Xie, 2023; Lan et al., 2024a), which has shown superior performance against traditional transformer layer:

#### z˜ := DT(MLP(z)), (3)

where an MLP layer first projects the input latent to the corresponding dimension, and DT is the DiT transformer. Since dense Gaussians are preferred for high-quality splatting (Kerbl et al., 2023), we gradually upsample the latent features through transformer blocks. Specifically, given a learnable embedding zu ∈ Rf

u×C where fu is the up-sampling ratio, we prepend it to each token in the latent sequence. Then, H layers of transformer blocks are used to model the upsampling process:

#### z(ik+1) := DUk ([zu ⊕ z˜i]), (4)

where DUk is a transformer block for predicting the k−th levels of details (LoD) Gaussian as shown in Fig. 1, and z(ik+1) ∈ Rf

u×C are the upsampled set of tokens. The overall tokens z(k+1) ∈ R(f

u×N)×C after up-sampling are used to predict the 13-dim attributes of surfel Gaussians.

To achieve denser Gaussians prediction, we cascade the upsampling transformer defined in Eq. (4) for K times, giving the final Upsampler DU for high-quality Gaussian output. Note that our solution outputs a set of Gaussians that are uniformly distributed on the 3D object surface with near 100% Gaussian utilization ratio. Existing pixel-aligned Gaussian prediction models (Tang et al., 2024; Yinghao et al., 2024; Szymanowicz et al., 2023), however, usually waste 50% Gaussians due to view overlaps and empty background color. Besides, our intermediate Gaussians output naturally serves as K LoDs (Takikawa et al., 2021), which can be used in different scenarios to balance the rendering speed and quality.

Training. Our 3D VAE model is end-to-end optimized across both input views and randomly chosen views, minimizing image reconstruction objectives between the splatting renderings and groundtruth renderings. Besides image reconstruction loss, we also impose loss over geometry regularizations, KL constraints, and adversarial loss:

L(ϕ, ψ) = Lrender + Lgeo + λklLKL + λGANLGAN, (5)

where Lrender is a mixture of the L1 and VGG loss (Zhang et al., 2018), Lgeo improves geometry reconstruction (Huang et al., 2024a), LKL is the KL-reg loss (Kingma & Welling, 2013; Rombach et al., 2022) to regularize a structured latent space, and LGAN improves perceptual fidelity. All loss terms except LKL are applied over a randomly chosen LoD in each iteration, and the Lrender is applied to both input-view and randomly sampled novel-view images. For details of geometry loss Lgeo, please refer to Sec. A.1.

- 3.2 CASCADED 3D GENERATION WITH FLOW MATCHING

After training the point-cloud structured 3D VAE, we get a dataset of D shapes paired with condition vectors (e.g., caption or images), {(zi,ci)}i∈[D], where the shape is represented by latent code z

###### ×N

###### ×N

⊕

⊕

|#!,>|
|---|

<$

<$

###### Text

Conditions

Scale

Scale

FFN

FFN

Text

conditi on

StageI

Point Cloud Diffusion

9$ , ;$ Scale, Shift

9$ , ;$ Scale, Shift

#$%

or

CLIP RMSNorm

[Figure 24]

RMS Norm

Image

⊕

⊕

Text Encoder

<#

!!,#

Scale

Multi-Head Cross Attention

Multi-Head Self Attention

(w/ QK-Norm)

Image

|#=,>| |
|---|---|
| | |

⊕

(w/ QK-Norm)

9# , ;#

Scale, Shift

<#

Scale

⊕

RMS Norm

DINO

Multi-Head Self Attention

⊕

StageII

Texture Diffusion

(w/ QK-Norm)

Image Encoder

9# , ;#

#$&

Multi-Head Cross Attention

Scale, Shift

[Figure 25]

MLP

RMS Norm

MLP

(w/ QK-Norm)

(shared)

(shared)

!#: !!,#, !$,#

Time t

Time t

(a) DiT Block (Text condition)

(b) DiT Block (Image condition)

(c) Two-stage Cascaded Diffusion

- Figure 2: Diffusion training of GAUSSIANANYTHING. Based on the point-cloud structure 3D VAE, we perform cascaded 3D diffusion learning given text (a) and image (b) conditions. We adopt DiT architecture with AdaLN-single (Chen et al., 2023) and QK-Norm (Dehghani et al., 2023; Esser

- et al., 2021). For both condition modality, we send in the conditional feature with cross attention block, but at different positions. The 3D generation is achieved in two stages (c), where a point cloud diffusion model first generates the 3D layout zx,0, and a texture diffusion model further generates the corresponding point-cloud features zh,0. The generated latent code z0 is decoded into the final

- 3D object with the pre-trained VAE decoder.

through the 3D VAE aforementioned. Our goal is to train a flow-matching generative model to learn a flow-based prior on top of it. Below we present how we adapt flow-based models to our case.

Cascaded Flow Matching over Symmetric Data. As detailed in Sec. A.3, flow matching involves training a neural network ϵΘ to predict the velocity v of the noisy input zt with the straight-line trajectory. After training, ϵΘ can sample from a standard Normal prior N(0,I) by solving the reverse ODE/SDE (Karras et al., 2022). In our case, the training data point is the point-cloud structured latent code z = [zx ⊕ zh] ∈ R(3+C

h)×N, which is symmetric and permutation invariant (Zeng et al., 2022; Nichol et al., 2022). Based on this property, we opt for diffusion transformer (Peebles & Xie, 2023) without positional encoding as the ϵΘ parameterization.

Here, rather than modeling zx and zh jointly, we empirically found that a cascaded framework (Ho et al., 2021; Lyu et al., 2024; 2023; Schr¨oppel et al., 2024) leads to better performance. Specifically,

a conditioned sparse point cloud generative model ϵxΘ is first trained to generate the overall structure of the given object:

- 1

- 2

Et∼U(t),ϵ∼N(0,I) wtFMλ′t∥ϵxΘ(zx,t,t,c) − ϵ∥2 , (6)

Lxw(x0) = −

and a point cloud feature generative model ϵhΘ is cascaded to learn the corresponding KLregularized feature conditioned on the sparse point cloud:

- 1

- 2

Et∼U(t),ϵ∼N(0,I) wtFMλ′t∥ϵhΘ(zh,t,zx,t,c) − ϵ∥2 . (7)

Lhw(x0) = −

The detailed cascading process is detailed Fig. 2 (c). Our proposed design enables better geometrytexture disentanglement and facilitates 3D editing over specific shape properties. For derivations of the flow matching training objective, please refer to Sec. A.3 for more details.

Conditioning Mechanism. Compared to LRM (Hong et al., 2024b; Tang et al., 2024) line of work which intrinsically relies on image(s) as the input, our native flow-based method enables more flexible 3D generation from diverse conditions. As shown in Fig. 2 (a-b), for the text-conditioned model, we adopt CLIP (Radford et al., 2021) to extract penultimate tokens as the condition embeddings; and

Single-view Image to 3D Multi-view Image to 3D Native 3D Diffusion Model

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Input Open-LRM Splatter Image One-2-3-45 CRM Lara LGM Shape-E LN3Diff Ours

- Figure 3: Qualitative Comparison of Image-to-3D. We showcase the novel view 3D reconstruction of all methods given a single image from unseen GSO dataset. Our proposed method achieves consistently stable performance across all cases. Note that though feed-forward 3D reconstruction methods achieve sharper texture reconstruction, these method fail to yield intact 3D predictions under challenging cases (e.g., the rhino in row 2). In contrast, our proposed native 3D diffusion model achieve consistently better performance. Better zoom in.

for the image conditioned model, we use DINOv2 (Oquab et al., 2023) to extract global and patch features. All conditions are injected into the DiT architecture through a pre-norm (Xiong et al., 2020) cross-attention block following the common practice of Zhang et al. (2024b). All the models are trained with Classifier-free Guidance (CFG) (Ho, 2021) by randomly dropping the conditions with a probability of 10%.

To cascade two flow-based models, we encode the output of stage-1 model ϵxΘ with PE(zx) as in Eq. (2), and add it to the first-layer features of ϵhΘ. This guarantees that generated features are paired with the input sparse point cloud structure.

- 4 EXPERIMENTS

Datasets. To train our 3D VAE, we use the renderings provided by G-Objaverse (Qiu et al., 2023; Deitke et al., 2023b) and choose a high-quality subset with around 176K 3D instances, where each consists of 40 random views with RGB, normal, depth map and camera pose. For text-conditioned diffusion training, we use the caption provided by Cap3D (Luo et al., 2023; 2024) and 3DTopia Hong et al. (2024a). For image-conditioned training, we randomly select an image in the dataset of the corresponding 3D instance as the condition.

Implementation Details. For 3D VAE, we choose V = 8 views of RGB-D-N renderings as input to guarantee a thorough coverage of the 3D object. The CNN Encoder is implemented with a similar architecture as LDM VAE (Rombach et al., 2022) with a down-sampling factor of f = 8, and the multi-view transformer has five layers as in RUST (Sajjadi et al., 2023). The sparse point cloud zx has a size of N ×3 where N = 768, and the corresponding featured point cloud zh has a dimension of N × 10. For upsampling blocks, we employ K = 3 blocks with fu1 = 8, fu1 = 4, and fu1 = 3, giving 73,768 Gaussians in total. All transformer blocks follow a pre-norm design (Xiong et al.,

- 2020). During 3D VAE training, the model is supervised by randomly chosen LoD renderings, with

λkl = 2e − 6, λd = 1000, λn = 0.2, and λGAN = 0.1. We adopt batch size 64 with both input and random novel views for training. During the conditional flow-based model training stage, we adopt batch size 256. All models are efficiently and stably trained with lr = 1e − 4 on 8×A100 GPUs for 1M iterations with BF16 and FlashAttention (Dao, 2024) enabled. We use CFG=4 and 250 ODE steps for all sampling results.

### Table 2: Quantitative evaluation of image-conditioned 3D generation. Here, quality of both

- 2D rendering and 3D shapes is evaluated. As shown below, the proposed method demonstrates strong performance across all metrics. Although multi-view images-to-3D approaches like LGM achieves better performance on the FID/KID metrics, they fall short on more advanced image quality assessment metrics such as CLIP-I and MUSIQ. Besides, they perform significantly worse in 3D shape quality. For multi-view to 3D baselines, we also include the number of input views (V=#).

Method CLIP-I↑ FID↓ KID(%)↓ MUSIQ↑ P-FID↓ P-KID(%)↓ COV(%)↑ MMD(‰)↓

OpenLRM 86.37 38.41 1.87 45.46 35.74 12.60 39.33 29.08 Splatter-Image 84.10 48.80 3.65 30.33 19.72 7.03 37.66 30.69

One-2-3-45 (V=12) 80.72 88.39 6.34 59.02 72.40 30.83 33.33 35.09 CRM (V=6) 85.76 45.53 1.93 64.10 35.21 13.19 38.83 28.91 Lara (V=4) 84.64 43.74 1.95 39.37 32.37 12.44 39.33 28.84 LGM (V=4) 87.99 19.93 0.55 54.78 40.17 19.45 50.83 22.06

Shape-E 77.05 138.53 11.95 31.51 20.98 7.41 61.33 19.17 LN3Diff 87.24 29.08 0.89 50.39 27.17 10.02 55.17 19.94 Ours 89.06 24.21 0.76 65.17 8.72 3.22 59.50 15.48

- 4.1 METRICS AND BASELINES

Evaluating Image-to-3D Generation. We evaluate GAUSSIANANYTHING on both image and text conditioned generation. Regarding image-conditioned 3D generation methods, we compare the proposed method with three lines of methods: single-image to 3D methods: OpenLRM (He & Wang, 2023; Hong et al., 2024b), Splatter Image (Szymanowicz et al., 2023), multi-view images to 3D methods: One-2-3-45 Liu et al. (2023a), CRM (Wang et al., 2024), Lara (Chen et al., 2024a), LGM (Tang et al., 2024), and native 3D diffusion models: LN3Diff-image (Lan et al., 2024a).

Quantitatively, we benchmark rendering metrics with CLIP-I Radford et al. (2021), FID (Heusel et al., 2017), KID (Bi´nkowski et al., 2018), and MUSIQ-koniq (Ke et al., 2021; Zhou et al., 2022). For 3D quality metrics, we benchmark Point cloud FID (P-FID), Point cloud KID (P-KID), Coverage Score (COV), and Minimum Matching Distance (MMD). Following previous works Nichol et al. (2022); Zhang et al. (2023); Yariv et al. (2024), we adopt the pre-trained PointNet++ provided by Point-E (Nichol et al., 2022) for calculating P-FID and K-FID. Qualitatively, GSO (Downs et al.,

- 2022; Zheng & Vedaldi, 2023) dataset is used for visually inspecting image-conditioned generation.

Table 1: Quantitative Evaluation on Text-to-3D. The proposed method outperforms competitive alternatives on both CLIP scores and aesthetic scores.

Method ViT-B/32↑ ViT-L/14↑ MUSIQ-AVA ↑ Q-Align ↑

Point-E 26.35 21.40 4.08 1.21 Shape-E 27.84 25.84 3.69 1.56 LN3Diff 29.12 27.80 4.16 2.22 3DTopia 30.10 28.11 3.31 1.42

Ours 31.80 29.38 4.99 3.13

Evaluating Text-to-3D Generation. Regarding text-conditioned 3D generation methods, we compare against PointE (Nichol et al., 2022), Shape-E (Jun & Nichol, 2023), 3DTopia (Hong et al., 2024a), and LN3Diff-text (Lan et al., 2024a). CLIP score (Radford et al., 2021) is reported following the previous works (Lan et al., 2024a; Hong et al., 2024a), with aesthetic scores MUSIQAVA (Ke et al., 2021) and Q-Align (Wu et al., 2023b) also included.

- 4.2 EVALUATION

In this section, we evaluate our proposed method over image-to-3D generation, text-to-3D generation, and 3D-aware editing. Please check the appendix for more visual results and point cloudconditioned 3D generation. Image-to-3D Generation. Our proposed framework enables 3D generation given single-view image conditions, leveraging the architecture detailed in Fig. 2 (b). Following current method (Chen et al., 2024a; Tang et al., 2024), we qualitatively benchmark our method in Fig. 3 over the single-view 3D reconstruction task on the unseen images from the GSO dataset. Our proposed framework is robust to inputs with complicated structures (row 1,3,4) and self-occlusion (row 2,5), yielding consistently intact 3D reconstruction. Besides, our generative-based method shows a more natural back-view reconstruction, as opposed to regression-based methods that are commonly blurry on uncertain areas.

Quantitatively, we showcase the evaluation in Tab. 2. As can be seen, our proposed method achieves state-of-the-art performance over CLIP-I and all 3D metrics, with competitive results over conventional 2D rendering metrics FID/KID. Note that LGM leverages pre-trained MVDream (Shi et al., 2023b) as the first-stage generation, and then maps the generated 4 views to pixel-aligned 3D Gaussians. This cascaded pipeline achieves better visual quality, but prone to yield distorted 3D geometry, as visualized in Fig. 3.

Text-to-3D Generation. We demonstrate the text-to-3D generation performance in Fig. 4 and Tab. 1. The flow-based model trained on GAUSSIANANYTHING’s latent space has demonstrated high-quality text-to-3D generation of generic 3D objects, yielding superior performance in terms of object structure, textures, and surface normals. Quantitatively, our proposed method achieves better text-3D alignment against competitive baselines.

3D-aware Editing. Compared to existing methods that use unstructured tokens for 3D diffusion learning (Jun & Nichol, 2023), our proposed point-cloud structured latent space naturally facilitates geometry-texture disentanglement and allows for interactive 3D editing. As visualized in Fig. 5, given the text-conditioned generated point cloud z0 by ϵxΘ, we sample the final 3D objects with ϵhΘ with a different random seed. As can be seen, the generated 3D objects maintain a consistent structure layout while yielding diverse textures. Besides, by directly manipulating the conditioned point cloud zx,0, our proposed method enables interactive 3D editing, as in 2D models (Pan et al.,

- 2023; Mou et al., 2023b). This functionality greatly facilitates the 3D content creation process for artists and opens up new possibilities for 3D editing with diffusion models.

- 4.3 ABLATION STUDY AND ANALYSIS

- Table 3: Ablation of 3D VAE Design. We ablate the design of our 3D VAE. Input-side, leveraging multi-view RGB-D-N renderings shows superior performance against dense point cloud. Besides, adding Gaussian up-sampling modules leads to consistent performance gain.

Table 4: Gaussian Utilization Ratio. We compare the effective Gaussians (opacity > 0.005) used during splatting here. Pixel-aligned Gaussian prediction methods waste a large portion of Gaussians when representing 3D object due to white background and multi-view overlap, while our proposed Gaussian predictions yields more compact reconstruction results.

[Figure 31]

Design LPIPS@100K Dense PCD as Input 0.174 Multi-view RGB-D as Input 0.163

Method High-opacity Gaussians (%)

+ Normal Map 0.157 + Gaussian SR Module 0.095 + 3 × Gaussian SR Module 0.067

Splatter Image 17.14 LGM 52.63 Ours 96.84

[Figure 32]

[Figure 33]

3D VAE Design. In Tab. 3, we benchmark each component of our 3D VAE architecture over a subset of Objaverse with 50K instances and record the LPIPS at 100K iterations. As shown in Tab. 3, our

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

A voxelized dog.

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

An 18th century cannon.

Ours 3D	Topia LN3Diff Shap-E Point-E

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

A fancy mech suit.

A MacDonald Hamburger.

- Figure 4: Qualitative Comparison of Text-to-3D. We present text-conditioned 3D objects generated by GAUSSIANANYTHING, displaying two views of each sample. The top section compares our results with baseline methods, while the bottom shows additional samples from our method along with their geometry maps. Our approach consistently yields better quality in terms of geometry, texture, and text-3D alignment.

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

A hydrant.

[Figure 60]

[Figure 61]

[Figure 62]

𝜖 𝜖 𝐳 ,  𝜖

[Figure 63]

[Figure 64]

An astral beacon.

Stage-2 𝐳 ,  generation (Texture Geometry Disentanglement)

Stage-1 𝐳 ,  generation

Interactive Editing

- Figure 5: 3D editing. Given two text prompts, we generate the corresponding point cloud z0,x with stage-1 diffusion model with ϵxΘ, and the corresponding point cloud features z0,h can be further

generated with ϵhΘ. As can be seen, the samples from stage-2 are consistent in overall 3D structures but with diverse textures. Thanks to the proposed Point Cloud-structured Latent space, our method

supports interactive 3D structure editing. This is achieved by first modifying the stage-1 point cloud z0,x → z′0,x, and then regenerate the 3D object with the same Gaussian noise.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

A voxelized dog.

(a) No cascaded text-to-3D generation

[Figure 71]

Editing on Gaussians Latent-based Editing (Ours)

| |
|---|

[Figure 72]

[Figure 73]

Input

[Figure 74]

(b) Editing on the point cloud latent space

- Figure 6: Qualitative ablation of Cascaded diffusion and latent space editing. We first show the effectiveness of our two-stage cascaded diffusion framework in (a). Compared to Fig. 4, the singlestage 3D diffusion yields worse texture details and 3D structure intactness. In (b), we disjoint the hydrant cover to demonstrate that our latent point cloud editing yields smoother and more reasonable results, while direct editing on 3D Gaussians shows tearing artifacts.

input design performs better against dense (16,384) colored point cloud (Zhang et al., 2024b), and the reconstruction quality consistently improves by including normal map as input and cascading more Gaussian upsampling blocks.

Gaussian Utilization Ratio. Besides, we showcase a high Gaussian utilization ratio of our proposed method. Specifically, we calculate the ratio of Gaussians with an opacity greater than 0.005 as effective Gaussians, as they contribute well to the final rendering. We calculate the statistics over 50K 3D instances. As shown in Tab. 4, our proposed Gaussian prediction framework achieves a much higher utilization ratio. On the contrary, pixel-aligned Gaussian prediction models waste a noticeable portion of Gaussians on the overlapping views and white backgrounds.

Effectiveness of Cascaded 3D Diffusion. We qualitatively ablate the cascaded model design in Fig. 6 (a), where a single text-conditioned DiT is trained to synthesize the 3D point cloud and features jointly. Clearly, the jointly trained model has a worse texture with 3D shape artifacts to our cascaded design. Besides bringing better editing capability as shown in Fig. 5, our cascaded design enables more flexible training, where the models of two stages can be trained in parallel.

3D Editing on the 3D Latent Space. Finally, we ablate the 3D editing performance in Fig. 6 (b). As can be seen, direct editing on the final Gaussians leads to 3D artifacts, while editing on our 3D latent space yields more holistic and cleaner results since suitable features are re-generated after editing. Besides, our method enables easy editing on the sparse point cloud, compared to directly manipulating dense 3D Gaussians (Dong et al., 2024).

- 5 CONCLUSION AND DISCUSSIONS

In this work, we present a new paradigm of 3D generative model by learning the diffusion model over a interactive 3D latent space. A dedicated 3D variational autoencoder encodes multi-view 3D attributes renderings into a point-cloud structured latent space, where multi-modal diffusion learning can be efficiently performed. Our framework achieves superior performance over both text- and image-conditioned 3D generation, and potentially facilitates numerous downstream applications in 3D vision and graphics tasks. Please check the appendix for the discussion of limitations.

Acknowledgement. This study is supported under the RIE2020 Industry Alignment Fund Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contributions from the industry partner(s). It is also supported by Singapore MOE AcRF Tier 2 (MOET2EP20221-0011) and National Research Foundation, Singapore, under its NRF Fellowship Award NRF-NRFF16-2024-0003.

REFERENCES

Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023.

Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P. Srinivasan. Mip-NeRF: A multiscale representation for anti-aliasing neural radiance fields. In ICCV, 2021.

Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR, 2022.

Mikołaj Bi´nkowski, Dougal J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying MMD GANs. In ICLR, 2018.

Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, and Gordon Wetzstein. Efficient geometry-aware 3D generative adversarial networks. In CVPR, 2022.

Anpei Chen, Haofei Xu, Stefano Esposito, Siyu Tang, and Andreas Geiger. Lara: Efficient largebaseline radiance fields. In ECCV, 2024a.

Honghua Chen, Yushi Lan, Yongwei Chen, Yifan Zhou, and Xingang Pan. Mvdrag3d: Dragbased creative 3d editing via multi-view generation-reconstruction priors. arXiv preprint arXiv:2410.16272, 2024b.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis, 2023.

Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Xin Chen, Zhongang Cai, Lei Yang, Gang Yu, Guosheng Lin, and Chi Zhang. Meshanything: Artist-created mesh generation with autoregressive transformers, 2024c.

Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3D reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5828–5839, 2017.

Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In ICLR, 2024.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In NeurIPS, 2022.

Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In ICML, 2023.

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. Objaverse-xl: A universe of 10m+ 3d objects. arXiv preprint arXiv:2307.05663, 2023a.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, 2023b.

Shaocong Dong, Lihe Ding, Zhanpeng Huang, Zibin Wang, Tianfan Xue, and Dan Xu. Interactive3d: Create what you want by interactive 3d generation. arXiv preprint arXiv:2404.16510, 2024.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3d scanned household items. In ICRA, 2022.

Patrick Esser, Robin Rombach, and Bj¨orn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, 2021.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

Kyle Genova, Forrester Cole, Avneesh Sud, Aaron Sarna, and Thomas Funkhouser. Local deep

implicit functions for 3D shape. In CVPR, 2020. Markus Gross and Hanspeter Pfister. Point-based graphics. Elsevier, 2011. Zexin He and Tengfei Wang. OpenLRM: Open-source large reconstruction models. https://

github.com/3DTopia/OpenLRM, 2023.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017.

Jonathan Ho. Classifier-free diffusion guidance. In NeurIPS, 2021. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS,

2020.

Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. arXiv preprint arXiv:2106.15282, 2021.

Fangzhou Hong, Jiaxiang Tang, Ziang Cao, Min Shi, Tong Wu, Zhaoxi Chen, Tengfei Wang, Liang Pan, Dahua Lin, and Ziwei Liu. 3dtopia: Large text-to-3d generation model with hybrid diffusion priors. arXiv preprint arXiv:2403.02234, 2024a.

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. In ICLR, 2024b.

Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In SIGGRAPH 2024 Conference Papers. Association for Computing Machinery, 2024a.

Zixuan Huang, Justin Johnson, Shoubhik Debnath, James M Rehg, and Chao-Yuan Wu. Pointinfinity: Resolution-invariant point diffusion models. In CVPR, 2024b.

Heewoo Jun and Alex Nichol. Shap-E: Generating conditional 3D implicit functions. arXiv preprint arXiv:2305.02463, 2023.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In NeurIPS, 2022.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In ICCV, pp. 5148–5157, 2021.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, 2023.

Diederik P Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data

augmentation. In Neurips, 2023. Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. arXiv, 2013. Adam R. Kosiorek, Heiko Strathmann, Daniel Zoran, Pol Moreno, Rosalia Schneider, Sovna

Mokr’a, and Danilo Jimenez Rezende. NeRF-VAE: A geometry aware 3D scene generative model. ICML, 2021.

Yushi Lan, Fangzhou Hong, Shuai Yang, Shangchen Zhou, Xuyi Meng, Bo Dai, Xingang Pan, and Chen Change Loy. LN3Diff: Scalable latent neural fields diffusion for speedy 3D generation. In ECCV, 2024a.

Yushi Lan, Feitong Tan, Di Qiu, Qiangeng Xu, Kyle Genova, Zeng Huang, Sean Fanello, Rohit Pandey, Thomas Funkhouser, Chen Change Loy, and Yinda Zhang. Gaussian3diff: 3d gaussian diffusion for 3d full head synthesis and editing. In ECCV, 2024b.

Christoph Lassner and Michael Zollh¨ofer. Pulsar: Efficient sphere-based neural rendering. In CVPR, 2021.

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention-based permutation-invariant neural networks. In ICML, 2019.

Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. CraftsMan: High-fidelity mesh generation with 3D native generation and interactive geometry refiner, 2024.

Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In WACV, 2023.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR, 2023.

Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Zexiang Xu, Hao Su, et al. One-2-3-45: Any single image to 3D mesh in 45 seconds without per-shape optimization. arXiv preprint arXiv:2306.16928, 2023a.

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3D object, 2023b.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023c.

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3D: Single image to 3D using cross-domain diffusion. In CVPR, 2024.

Tiange Luo, Chris Rockwell, Honglak Lee, and Justin Johnson. Scalable 3D captioning with pretrained models. arXiv preprint arXiv:2306.07279, 2023.

Tiange Luo, Justin Johnson, and Honglak Lee. View selection for 3D captioning via diffusion ranking. arXiv preprint arXiv:2404.07984, 2024.

Zhaoyang Lyu, Jinyi Wang, Yuwei An, Ya Zhang, Dahua Lin, and Bo Dai. Controllable mesh generation through sparse latent point diffusion models. In CVPR, 2023.

Zhaoyang Lyu, Ben Fei, Jinyi Wang, Xudong Xu, Ya Zhang, Weidong Yang, and Bo Dai. Getmesh: A controllable model for high-quality mesh generation and manipulation, 2024.

Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv, 2024.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020.

Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, and Zan Gojcic. 3d gaussian ray tracing: Fast tracing of particle scenes. ACM Transactions on Graphics and SIGGRAPH Asia, 2024.

Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Diffeditor: Boosting accuracy and flexibility on diffusion-based image editing. arXiv preprint arXiv:2402.02583, 2023a.

Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Dragondiffusion: Enabling drag-style manipulation on diffusion models. arXiv preprint arXiv:2307.02421, 2023b.

Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. TOG, 41(4):102:1–102:15, July 2022.

Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-E: A system for generating 3D point clouds from complex prompts, 2022.

Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision, 2023.

Xingang Pan, Ayush Tewari, Thomas Leimk¨uhler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag Your GAN: Interactive point-based manipulation on the generative image manifold. In SIGGRAPH, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. Hanspeter Pfister, Matthias Zwicker, Jeroen Van Baar, and Markus Gross. Surfels: Surface elements

as rendering primitives. In PACMCGIT, 2000. Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. DreamFusion: Text-to-3D using 2D diffusion. ICLR, 2022. Charles Qi, Hao Su, Kaichun Mo, and Leonidas Guibas. PointNet: Deep learning on point sets for 3D classification and segmentation. arXiv, 2016. Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. PointNet++: Deep hierarchical feature learning on point sets in a metric space. In NeurIPS, 2017.

Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, and Xiaoguang Han. Richdreamer: A generalizable normal-depth diffusion model for detail richness in text-to-3d. arXiv preprint arXiv:2311.16918, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, 2022.

Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, and Hao Li. PIFu: Pixel-aligned implicit function for high-resolution clothed human digitization. In ICCV, October 2019.

Mehdi S. M. Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Radwan, Suhani Vora, Mario Lucic, Daniel Duckworth, Alexey Dosovitskiy, Jakob Uszkoreit, Thomas Funkhouser, and Andrea Tagliasacchi. Scene Representation Transformer: Geometry-free novel view synthesis through set-latent scene representations. CVPR, 2022.

Mehdi S. M. Sajjadi, Aravindh Mahendran, Thomas Kipf, Etienne Pot, Daniel Duckworth, Mario Luˇci´c, and Klaus Greff. RUST: Latent Neural Scene Representations from Unposed Imagery. CVPR, 2023.

Philipp Schr¨oppel, Christopher Wewer, Jan Eric Lenssen, Eddy Ilg, and Thomas Brox. Neural point cloud diffusion for disentangled 3d shape and appearance generation. In CVPR, 2024.

Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. In arXiv, 2023a.

Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3D generation. arXiv:2308.16512, 2023b.

Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias Nießner. Meshgpt: Generating triangle meshes with decoderonly transformers. arXiv preprint arXiv:2311.15475, 2023.

Vincent Sitzmann, Semon Rezchikov, William T. Freeman, Joshua B. Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. In NeurIPS, 2021.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021.

Stanislaw Szymanowicz, Christian Rupprecht, and Andrea Vedaldi. Splatter image: Ultra-fast single-view 3D reconstruction. In arXiv, 2023.

Towaki Takikawa, Joey Litalien, Kangxue Yin, Karsten Kreis, Charles Loop, Derek Nowrouzezahrai, Alec Jacobson, Morgan McGuire, and Sanja Fidler. Neural geometric level of detail: Real-time rendering with implicit 3D shapes. In CVPR, 2021.

Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In NeurIPS, 2020.

Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In ECCV, 2024.

Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P. Srinivasan, Howard Zhou, Jonathan T. Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas A. Funkhouser. IBRNet: Learning Multi-View Image-Based Rendering. In CVPR, 2021.

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3D generation with variational score distillation. In NeurIPS, 2023.

Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu, Chongxuan Li, Hang Su, and Jun Zhu. CRM: Single image to 3D textured mesh with convolutional reconstruction model. In ECCV, 2024.

Chao-Yuan Wu, Justin Johnson, Jitendra Malik, Christoph Feichtenhofer, and Georgia Gkioxari. Multiview compressive coding for 3D reconstruction. arXiv preprint arXiv:2301.08247, 2023a.

Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Chunyi Li, Liang Liao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtai Zhai, and Weisi Lin. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023b. Equal Contribution by Wu, Haoning and Zhang, Zicheng. Project Lead by Wu, Haoning. Corresponding Authors: Zhai, Guangtai and Lin, Weisi.

Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3D: Scalable image-to-3d generation via 3D latent diffusion transformer, 2024.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506, 2024.

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On layer normalization in the transformer architecture, 2020.

Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191, 2024a.

Jiale Xu, Shenghua Gao, and Ying Shan. Freesplatter: Pose-free gaussian splatting for sparse-view 3d reconstruction. arXiv preprint arXiv:2412.09573, 2024b.

Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Point-NeRF: Point-based neural radiance fields. In CVPR, 2022.

Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Jiahao Li, Zifan Shi, Kalyan Sunkavalli, Gordon Wetzstein, Zexiang Xu, and Kai Zhang. DMV3D: Denoising multi-view diffusion using 3D large reconstruction model. In ICLR, 2024c.

Haitao Yang, Yuan Dong, Hanwen Jiang, Dejia Xu, Georgios Pavlakos, and Qixing Huang. Atlas gaussians diffusion for 3d generation. In ICLR, 2025.

Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. CVPR, 2020.

Lior Yariv, Omri Puny, Natalia Neverova, Oran Gafni, and Yaron Lipman. Mosaic-sdf for 3d generative models. In CVPR, 2024.

Wang Yifan, Felice Serena, Shihao Wu, Cengiz Oztireli,¨ and Olga Sorkine-Hornung. Differentiable surface splatting for point-based geometry processing. ACM Transactions on Graphics (proceedings of ACM SIGGRAPH ASIA), 38(6), 2019.

Xu Yinghao, Shi Zifan, Yifan Wang, Chen Hansheng, Yang Ceyuan, Peng Sida, Shen Yujun, and Wetzstein Gordon. GRM: Large gaussian reconstruction model for efficient 3d reconstruction and generation, 2024.

Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. PixelNeRF: Neural radiance fields from one or few images. In CVPR, 2021.

Lequan Yu, Xianzhi Li, Chi-Wing Fu, Daniel Cohen-Or, and Pheng-Ann Heng. Pu-net: Point cloud upsampling network. In CVPR, 2018.

Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Tianyou Liang, Guanying Chen, Shuguang Cui, and Xiaoguang Han. MVImgNet: A large-scale dataset of multi-view images. In CVPR, 2023.

Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and Karsten Kreis. Lion: Latent point diffusion models for 3D shape generation. In NeurIPS, 2022.

Biao Zhang, Jiapeng Tang, Matthias Nießner, and Peter Wonka. 3DShape2VecSet: A 3d shape representation for neural fields and generative diffusion models. ACM Trans. Graph., 42(4), jul

- 2023. ISSN 0730-0301. doi: 10.1145/3592442.

Biao Zhang, Jing Ren, and Peter Wonka. Geometry distributions. arXiv preprint arXiv:2411.16076,

- 2024a.

Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. CLAY: A controllable large-scale generative model for creating high-quality 3D assets. ACM Transactions on Graphics, 2024b.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In CVPR, 2021.

Chuanxia Zheng and Andrea Vedaldi. Free3d: Consistent novel view synthesis without 3d representation. arXiv, 2023.

Shangchen Zhou, Kelvin C.K. Chan, Chongyi Li, and Chen Change Loy. Towards robust blind face restoration with codebook lookup transformer. In NeurIPS, 2022.

APPENDIX

- A IMPLEMENTATION DETAILS

- A.1 TRAINING DETAILS

VAE Architecture. For the convolutional encoder Eϕ, we adopt a lighter version of LDM Rombach et al. (2022) encoder with channel 64 and 1 residual blocks for efficiency. When training on Objaverse with V = 8, we incorporate 3D-aware attention Shi et al. (2023b) in the middle layer of the convolutional encoder. The multi-view transformer architecture is similar to RUST (Sajjadi et al., 2023; 2022). For each upsampler DUk , we have 2 transformer blocks in the middle. All hyperparameters remain at their default settings. Regarding the transformer decoder DT, we employ the DiT-B/2 architecture due to VRAM constraints. Compared to LN3Diff (Lan et al., 2024a), we do not adopt cross-plane attention in the transformer decoder.

Diffusion Model. We mainly adopt the diffusion training pipeline implementation from SiT Ma et al. (2024), with pred-v objective, GVP schedule, and uniform t sampling. ODE solver with 250 steps is used for all the results shown in the paper. For the DiT architecture with cross attention and single-adaLN-zero design, we mainly refer to PixArt Chen et al. (2023). The diffusion transformer is built with 24 layers with 16 heads and 1024 hidden dimension, which result in 458M parameters. For all the diffusion models, we further add the global token to t features as part of the condition input.

Details of Geometry Loss in VAE Training. The geometry loss Lgeo is composed of two regularization terms, including the depth distortion loss to concentrate the weight distribution along rays, inspired by Mip-NeRF (Barron et al., 2021; 2022). Given a ray of pixel, the distortion loss is defined as

Ld =

i,j

ωiωj|di − dj|, (8)

where ωi = αi Gˆi(u(x)) ij−=11(1−αj Gˆj(u(x))) is the blending weight of the i−th intersection and di is the depth of the intersection points. Besides, as surfel Gaussians explicitly model the primitive normals, we encourage the splats’ normal to locally approximate the actual object surface:

Ln =

i

ωi(1 − NˆiTN), (9)

where Nˆ is the predicted normal maps. The final geometry loss is given by Lgeo = λdLd + λnLn.

- A.2 DATA AND BASELINE COMPARISON

Training Data. For Objaverse, we use a high-quality subset from the pre-processed rendering from G-buffer Objaverse Qiu et al. (2023) for experiments. Since G-buffer Objaverse splits the subset into 10 general categories, we use all the 3D instances except from “Poor-quality”: Human-Shape, Animals, Daily-Used, Furniture, Buildings&Outdoor, Transportations, Plants, Food and Electronics. The ground truth camera pose, rendered multi-view images, normal, depth maps, and camera poses are used for stage-1 VAE training.

Details about Baselines. We use the official released code and checkpoint for all the comparisons shown in the paper. For the evaluation on the GSO dataset, we use the rendering provided by Free3D (Zheng & Vedaldi, 2023).

Evaluation details. For quantitative benchmark in Tab. 2, we use 600 instances from Objaverse with ground truth 3D mesh for evaluation. To calculate the visual metrics (FID/KID/MUSIQ), we use the first rendered instance as the image condition and render 24 images with fixed elevation (+15 degrees) with uniform azimuths trajectory (24 × 15 degrees) with radius= 1.8. For 3D metrics, we export the extracted 3D mesh and sample 4096 points using FPS sampling on the mesh surface. The ground truth surface point cloud is processed in the same way. The pre-trained PointNet++ model from Point-E is used for P-FID and P-KID evaluation. All generated 3D models are aligned into the same canonical space before 3D metrics calculation. All intermediate results of the baselines for evaluation will be released.

- A.3 MORE PRELIMINARIES

- 2D Gaussian Splatting (2DGS). Since 3DGS (Kerbl et al., 2023) models the entire angular radiance in a blob, it fails to reconstruct high-quality object surfaces. To resolve this issue, Huang et al. (2024a) proposed 2DGS (surfel-based GS) that simplifies the 3-dimensional modeling by adopting “flat” 2D Gaussians embedded in 3D space, which enables better alignment with thin surfaces.

Notation-wise, the 2D splat is characterized by its central point pk, two principal tangential vectors tu and tv, and a scaling vector S = (su,sv) that controls the variances of the 2D Gaussian. Notice that the primitive normal is defined by two orthogonal tangential vectors tw = tu × tv. Thus, the

- 2D Gaussian is parameterized with P(u,v) = pk + sutuu + svtvv = H(u,v,1,1)T (10)

whereH =

sutu svtv 0 pk 0 0 0 1

=

RS pk 0 1

(11)

Where H parameterizes the local 2D Gaussian geometry. For the point u = (u,v) in uv space, its 2D Gaussian value can then be evaluated by standard Gaussian G(u) = exp −u

2+v2

2 , and the center

pk, scaling (su,sv), and the rotation (tu,tv) are all learnable parameters. Following 3DGS Kerbl et al. (2023), each 2D Gaussian primitive has opacity α and view-dependent appearance c, and can be rasterized via volumetric alpha blending:

ci αi Gˆi(u(x))

c(x) =

i=1

- i−1
- j=1

(1 − αj Gˆj(u(x))), (12)

where the integration process is terminated when the accumulated opacity reaches saturation. During optimization, pruning and densification operations are iteratively applied.

Flow Matching and Diffusion Model. Diffusion models create data from noise (Song et al., 2021) and are trained to invert forward paths of data towards random noise. The forward path is constructed

- as zt = atx0 + btϵ, where ϵ ∼ N(0,I) , at and bt are hyper parameters. The choice of forward process has proven to have important implications for the backward process of data sampling (Lin et al., 2023).

Recently, flow matching (Liu et al., 2023c; Albergo et al., 2023; Lipman et al., 2023) has introduced a particular choice for the forward path, which has better theoretical properties and has been verified on the large-scale study (Esser et al., 2024). Given a unified diffusion objective (Karras et al., 2022):

- 1

- 2

Et∼U(t),ϵ∼N(0,I) wtλ′t∥ϵΘ(zt,t) − ϵ∥2 , (13) where λt := log a

Lw(x0) = −

2 t

b2t denotes signal-to-noise ratio, and λ′t denotes its derivative. By setting wt = 1−t t with zt = (1 − t)x0 + tϵ, flow matching defines the forward process as a straight path between the data distribution and the Normal distribution. The network ϵΘ directly predicts the velocity vΘ, and please check the following section for more detailed derivation.

Derivation of the Training Objective of Flow Matching. Since three works (Liu et al., 2023c; Albergo et al., 2023; Lipman et al., 2023) proposed the flow matching idea simultaneously, we adopt the unified formulation defined in Esser et al. (2024) in Eq. 6 and Eq. 7. Here we brief the background of conditional flow matching, and please read the Sec.2 of Esser et al. (2024) for indepth analysis.

Specifically, consider the forward diffusion process (Ho et al., 2020)

zt = atx0 + btϵ, where ϵ ∼ N(0,I). (14) To express the relationship between zt, x0, and ϵ, we define the mappings ψt and ut as:

ψt(· | ϵ) : x0  → atx0 + btϵ, (15) ut(z | ϵ) := ψt′ ψt−1(z | ϵ) | ϵ , (16)

where ψt−1 and ψt′ are the inverse and derivative of ψt, respectively.

Since zt can be viewed as a solution to the ODE

zt′ = ut(zt | ϵ), with initial condition z0 = x0, (17) the conditional vector field ut(· | ϵ) generates the conditional probability path pt(· | ϵ). Remarkably, one can construct a marginal vector field ut that generates the marginal probability paths pt (Lipman et al., 2023), using the conditional vector fields ut(· | ϵ):

pt(z | ϵ) pt(z)

. (18)

ut(z) = Eϵ∼N(0,I) ut(z | ϵ)

The marginal vector field ut can be learned by minimizing the Conditional Flow Matching objective:

t(z|ϵ),p(ϵ) vΘ(z,t) − ut(z | ϵ) 22. (19) To make this objective explicit, we substitute:

LCFM = Et,p

ψt′(x0 | ϵ) = a′tx0 + b′tϵ, (20) ψt−1(z | ϵ) =

z − btϵ at

, (21) into the expression for ut(z | ϵ):

a′t at

b′t bt

a′t at −

zt′ = ut(zt | ϵ) =

. (22)

zt − ϵbt

′ t

′ t

2 t

b2t . With λ′t = 2 a

at − b

Next, consider the signal-to-noise ratio λt := log a

bt , the expression for ut(zt | ϵ) can be rewritten as:

a′t at

bt 2

λ′tϵ. (23)

ut(zt | ϵ) =

zt −

Using this reparameterization, the LCFM objective can be reformulated as a noise-prediction objective:

2

a′t at

bt 2

λ′tϵ

(24)

LCFM = Et,p

t(z|ϵ),p(ϵ) vΘ(z,t) −

z +

2

2

bt 2

ϵΘ(z,t) − ϵ 22, (25) where we define:

λ′t

#### = Et,p

t(z|ϵ),p(ϵ) −

a′t at

ϵΘ := −2 λ′

z . (26)

vΘ −

tbt

Since the optimal solution remains invariant to time-dependent weighting, one can derive various weighted loss functions that guide optimization towards the desired solution. For a unified analysis of different approaches, including classic diffusion formulations, we express the objective as Kingma & Gao (2023):

- 1

- 2

Et∼U(t),ϵ∼N(0,I) wtλ′t ϵΘ(zt,t) − ϵ 2 , (27) where wt = −12λ′tb2t corresponds to wtFM used in Eq. 6 and Eq. 7.

Lw(x0) = −

- B DISCUSSIONS OF LIMITATIONS

We acknowledge that the texture quality of our proposed method is still inferior to the state-ofthe-art multi-view based 3D generative models, i.e., LGM. Besides, the visual quality of the textconditioned 3D generation of native 3D generation methods is worse compared to SDS-based alternatives, despite being much faster and shows better diversity. We believe our method has made a step forward towards bridging the gap. To further improve the performance, we list some potential directions in the following:

- 1. Enhancing the 3D VAE Quality.. The performance of the 3D VAE could be improved by increasing the number of latent points and incorporating a pixel-aligned 3D reconstruction paradigm, such as PiFU (Saito et al., 2019), to achieve finer-grained geometry and texture alignment.
- 2. Incorporating Additional Losses in Diffusion Training. Currently, the diffusion training relies solely on latent-space flow matching. Prior work, such as DMV3D (Xu et al., 2024c), demonstrates that incorporating a rendering loss can significantly enhance the synthesis of high-quality 3D textures. Adding reconstruction supervision during diffusion training is another promising avenue to improve output fidelity.
- 3. Leveraging 2D Pre-training Priors. At present, the models are trained exclusively on 3D datasets and do not utilize 2D pre-training priors as effectively as multi-view (MV)based 3D generative models. A potential improvement is to incorporate 2D priors more effectively, for instance, by using multi-view synthesized images as conditioning during training instead of single-view images.
- 4. Expanding Dataset Diversity. Utilizing more diverse and extensive 3D datasets, such as Objaverse-XL (Deitke et al., 2023a) and MVImageNet (Yu et al., 2023), could further enhance the quality and generalizability of 3D generation.
- 5. Support of Physics-based Rendering. Currently, our proposed method leverages 2DGS as the underlying 3D representation since it balances the 2D rendering and 3D surface quality. However, incorporating more advanced 3D representations such as 3DGRT (MoenneLoccoz et al., 2024) can further support physics-based ray tracing over 3D Gaussians. Besides, this can be achieved by directly fine-tuning our pre-trained 3D VAE decoder only with the PBR rendering pipeline.

By addressing these aspects, the proposed method could achieve significant advancements in both the quality and versatility of its 3D generation capabilities.

- C DISCUSSIONS OF CONCURRENT WORK

After the submission of this paper, several related works on 3D generation have been released. We discuss them here.

FreeSplatter (Xu et al., 2024b) extends the multi-view generation and 3D reconstruction pipeline for

- 3D object generation, eliminating the need for input poses. It also supports 3D scene generation by training on the scene-level datasets (Yao et al., 2020; Dai et al., 2017). While following a different paradigm, our method could benefit from its pose-free design to enable image-conditioned 3D generation from multiple input views.

Geometric Distribution (Zhang et al., 2024a) introduces a novel 3D representation by learning a point cloud diffusion model for single-instance objects. This approach closely resembles our stage1 diffusion model but focuses on encoding geometric information. While it shows strong potential for downstream tasks, its efficiency remains a challenge for broader applications.

AtlasGaussians (Yang et al., 2025), like our method, proposes a native 3D diffusion model over 3D Gaussians. It employs point clouds as input for 3D VAE compression and utilizes local patch features for 3D Gaussian decoding. However, it lacks an explicit 3D latent space, making it unsuitable for interactive 3D editing.

Trellis (Xiang et al., 2024) adopts a similar cascaded 3D flow matching framework and achieves remarkable results in object generation. Unlike our method, which employs a point cloud latent space, Trellis leverages a sparse voxel structure with efficient operators for high-quality and fast 3D generation. It also supports interactive 3D editing via its explicit latent space. We hope this design, along with our proposed method, can establish a canonical paradigm for future 3D native generative models.

- D MORE VISUAL RESULTS AND VIDEOS Please also check our supplementary video demo and attached folders for video results.

##### Overview of the Qualitative Performance. In Fig. 7, we include an overview of the qualitative

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

Astronaut Suit and Helmet.

[Figure 86]

[Figure 87]

[Figure 88]

A pixelized dog.

[Figure 89]

Point Cloud (Stage 1)

Surfel Gaussians (Stage 2)

Surfel Gaussians Centers

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Latent Point Editing

Latent Point Editing

Surfel Gaussians Generation

Surfel Gaussians Generation

A yellow rubber duck with red mouth.

Input

Initial Results Interactive 3D Editing (1) Interactive 3D Editing (2)

- Figure 7: Our method generates high-quality and editable surfel Gaussians through a cascaded 3D diffusion pipeline, given single-view images or texts as the conditions.

performance of the proposed method, GAUSSIANANYTHING. Here, we show the single-view conditioned 3D generation, text-conditioned 3D generation, and 3D-aware editing capabilities.

- 3D VAE Reconstruction. In Fig. 8, we include the 3D VAE reconstruction results of our model

- at 3 level of details (LoD). Thanks to the versatile multi-view 3D attributes input and transformer design, our 3D VAE enables high-quality 3D reconstruction with visually attractive textures and smooth surface. The encoded point cloud-structured latent codes, z, serves as a compact proxy for efficient 3D diffusion training. Besides, the 2D Gaussians Upsampler naturally facilitates LoD and facilitates speed / quality trade off in practice.

More Text-to-3D results. In Fig. 9, we present additional qualitative comparisons of text-to-3D generation with GAUSSIANANYTHING. For this evaluation, we use relatively complex captions as input conditions and display two random samples generated by our model. As shown, GAUSSIANANYTHING produces visually appealing results that characterized rich textures, smooth surface, and notable diversity. To further demonstrate the generality of our proposed method, in Fig. 10 we include the uncurated text-to-3D results over DF-415 (Poole et al., 2022) prompts with captions and more detailed descriptions.

Point-to-3D Generation with Cascaded Point-E. Moreover, the cascaded design of our stage-2 diffusion model, ϵhΘ, enables flexible 3D generation given point clouds from diverse sources. To demonstrate this capability, we integrate the output of a state-of-the-art 3D point cloud generative model, such as Point-E (Nichol et al., 2022), into the GAUSSIANANYTHING generation pipeline. Specifically, we first generate the point cloud using Point-E based on a caption condition c. This generated point cloud is then used as input zx to our stage-2 point cloud/text-conditioned diffusion model ϵhΘ. As shown in Fig. 11, the generated surfel Gaussians exhibit significantly improved texture quality and geometry fidelity compared to the Point-E point cloud outputs. This capability broadens the applicability of our method, enabling it to benefit from recent advances in point cloud

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

###### LoD = 1 LoD = 2 LoD = 3

- Figure 8: 3D VAE Reconstruction. Here, we visualize the 3D VAE reconstruction performance across different level of details (LoD). As shown, higher LoD results in sharper textures and smoother surface. Better zoom in.

generation (Huang et al., 2024b) and mesh generation (Siddiqui et al., 2023; Chen et al., 2024c) for producing high-quality object-level surfel Gaussians.

Broader Social Impact. In this paper, we introduce a new latent 3D diffusion model designed to produce high-quality surfel Gaussians using a single model. As a result, our approach has the potential to be applied to generate DeepFakes or deceptive 3D assets, facilitating the creation of falsified images or videos. This raises concerns, as individuals could exploit such technology with malicious intent, aiming to spread misinformation or tarnish reputations.

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

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

A sword with a red handle. A cute and friendly pink teddy bear with sitting pose.

[Figure 124]

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

Low poly blue chess piece model.

[Figure 148]

[Figure 149]

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

Low poly tree model with green leaves.

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

Low poly model of a green pine tree, also resembling-a Christmas tree.

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

White Giraffe

###### Caption

Point-E Shape-E 3DTopia Ours (seed=1) Ours (seed=2)

- Figure 9: More Qualitative Comparison of Text-to-3D. We present more text-conditioned 3D objects generated by GAUSSIANANYTHING, alongside comparisons with competitive alternatives, including Point-E, Shape-E, and 3DTopia. As demonstrated, our approach consistently achieves superior quality in geometry, texture, and alignment between text and 3D content.

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

A plate of delicious taco.

A red eyed tree frog, low poly.

A zoomed out DSLR photo of a pile of dice on a green tabletop.

A zoomed out DSLR photo of a pita bread full of hummus and falafel and vegetables.

An amigurumi bulldozer.

An orchid flower planted in a clay pot.

- Figure 10: More Qualitative Results of Text-to-3D over DF-415 Captions. Our proposed method generalizes to long captions with detailed descriptions. All results are uncurated.

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

A cute and friendly pink teddy bear with sitting pose.

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Low-poly tree model with green leaves.

Caption Point-E

###### Point-E + GA (seed=1) Point-E + GA (seed=2)

- Figure 11: Cascaded Text-to-3D with Point-E. Thanks to our cascaded 3D generation design, the stage-2 diffusion model in GAUSSIANANYTHING seamlessly integrates with other point cloud generative models. To illustrate this capability, we leverage the state-of-the-art Point-E model. As demonstrated, our stage-2 diffusion model effectively transforms the point clouds generated by Point-E into diverse surfel Gaussians with more visually appealing features and enhanced geometric details.

