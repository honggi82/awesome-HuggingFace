# arXiv:2507.23785v1[cs.CV]31Jul2025

## Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis

Bowen Zhang1∗ Sicheng Xu2 Chuxin Wang1 Jiaolong Yang2 Feng Zhao1† Dong Chen2† Baining Guo2 1University of Science and Technology of China 2Microsoft Research Asia

[Figure 1]

[Figure 2]

|[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>… … …<br><br>[Figure 7]<br><br>In-the-wild Input Video<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]|
|---|

|[Figure 12]<br><br>[Figure 13]<br><br>… … …<br><br>[Figure 14]<br><br>In-the-wild Input Video<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]|
|---|

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

Figure 1. Our model is capable of creating high-fidelity 4D objects from in-the-wild video inputs. Best viewed with zoom-in.

### Abstract

### 1. Introduction

Recent advances in generative models have demonstrated remarkable capabilities across various modalities, including image [21, 23, 30, 31, 45, 55, 90], video [2, 3, 19], and 3D content [24, 66, 68, 81, 93, 95]. While these achievements mark important milestones, they naturally lead to the next frontier: 4D generation, which aims to create dynamic 3D content. The challenge of generating such content—a fundamental aspect of representing our inherently four-dimensional world—remains largely unexplored. This gap is particularly significant given that real-world phenomena inherently combine spatial and temporal dynamics, from the subtle object movements to complex character articulations.

In this paper, we present a novel framework for video-to-4D generation that creates high-quality dynamic 3D content from single video inputs. Direct 4D diffusion modeling is extremely challenging due to costly data construction and the high-dimensional nature of jointly representing 3D shape, appearance, and motion. We address these challenges by introducing a Direct 4DMesh-to-GS Variation Field VAE that directly encodes canonical Gaussian Splats (GS) and their temporal variations from 3D animation data without per-instance fitting, and compresses high-dimensional animations into a compact latent space. Building upon this efficient representation, we train a Gaussian Variation Field diffusion model with temporal-aware Diffusion Transformer conditioned on input videos and canonical GS. Trained on carefully-curated animatable 3D objects from the Objaverse dataset, our model demonstrates superior generation quality compared to existing methods. It also exhibits remarkable generalization to in-the-wild video inputs despite being trained exclusively on synthetic data, paving the way for generating high-quality animated 3D content. Project page: GVFDiffusion.github.io.

Despite diffusion models [15, 45, 59] having demonstrated strong modeling capabilities in both 2D and 3D domains, training a robust 4D diffusion model for dynamic 3D content generation presents two main technical challenges. First, obtaining a large-scale 4D dataset is time-consuming. A straightforward approach involves fitting individual dynamic Gaussian Splatting (4DGS) representations [76] for each 3D animation sequence, but this solution typically requires tens of minutes per instance, making it computationally expensive and less scalable as the number of instances increases. Second, the higher-dimensional nature of the

*Intern at Microsoft Research Asia. †Corresponding authors.

problem necessitates a large number of parameters (usually exceeding 100K tokens) to represent 3D shape, appearance, and motion simultaneously, making direct modeling with diffusion approaches extremely challenging. These limitations have significantly hindered the development of efficient and high-quality 4D generative models.

Motivated by the effectiveness of diffusion models applied to compact latent spaces in recent 2D and 3D generation works [3, 58, 59, 81, 95], we present a novel framework for 4D generative modeling that comprises a Direct 4DMeshto-GS Variation Field VAE and a Gaussian Variation Field diffusion model. Our VAE framework encodes the canonical 3D Gaussian Splatting (3DGS) of objects and compresses each Gaussian’s attribute variations (i.e., Gaussian Variation Fields) into a compact latent space from 4D mesh data, thereby bypassing costly per-instance reconstructions. Inspired by previous works [5, 91], we employ a perceiverstyle transformer network [26, 27, 72] with displacements of mesh points to effectively encode motion information. To bridge the gap between Gaussian Splatting representation and mesh-based ground truth motion, we introduce a meshguided loss that aligns the motion of Gaussian points with the corresponding mesh vertices. Our VAE is trained end-to-end with this mesh-guided loss and an image-level loss, enabling faithful compression of complex Gaussian Variation Fields. This approach reduces high-dimensional motion sequences to a compact 512-dimensional latent space, thus facilitating efficient diffusion modeling for 4D content generation.

Following the construction of our VAE, the 4D generative modeling naturally decomposes into canonical 3DGS generation and Gaussian Variation Field modeling. We leverage state-of-the-art 3D generative models [81] for the canonical component while focusing on modeling the Gaussian Variation Fields. To achieve this, we train a diffusion model to learn the latent space distribution of variation fields conditioned on the input video and canonical 3DGS, enabling controlled 4D content generation. Leveraging the compact nature of our latent space, we employ the Diffusion Transformer (DiT) architecture [52], augmented with temporal self-attention layers to capture smooth temporal dynamics across animations. The video frame features [48] and the canonical 3DGS are taken as conditions for the diffusion model via cross-attention layers. Additionally, we incorporate positional priors into the diffusion model, enhancing its awareness of correspondences between canonical GS and their variation fields during the denoising process, thereby improving generation quality.

We train our model on a carefully curated diverse collection of animatable 3D objects from the Objaverse [13] and Objaverse-XL [12]. Extensive evaluations demonstrate the superior video-to-4D generation quality of our method compared to existing approaches. Despite being trained on synthetic data, our model exhibits remarkable generaliza-

tion capabilities when applied to in-the-wild video inputs, effectively creating impressive animations from in-the-wild animation sequences. We believe that our approach represents a notable step toward narrowing the gap between static 3D generation and 4D content creation, paving the way for generating high-quality 4D content.

### 2. Related Work

- 3D generation. Early GAN-based 3D generation approaches [7, 14, 17, 64, 77, 80, 98, 100] laid the foundation for 3D content synthesis, while diffusion-based methods [6, 9, 22, 29, 44, 62, 69, 74, 86, 92, 93] advanced generation quality. Recent approaches have focused on latent space generation, either separating geometric modeling and appearance synthesis [37, 58, 71, 78, 91, 95, 97, 99] or jointly modeling both [10, 20, 29, 35, 46, 81, 83, 84]. Alternative methods [8, 40, 53, 65, 67, 75] leverage pretrained 2D models [59] through optimization techniques. Recent works [81, 95] have achieved high-quality 3D asset generation with detailed geometry and appearance, establishing a foundation for 4D content creation.
- 4D reconstruction. Early 4D reconstruction methods [4, 16, 50, 51, 54] extended neural volumetric techniques for dynamic scenes, while recent Gaussian Splatting-based approaches [11, 25, 32, 36, 42, 76] offer improved efficiency. Typical 4D reconstruction methods often require significant optimization time per instance (e.g., 6 minutes for 4DGaussians [76] and over 30 minutes for K-planes [16]), making them impractical to use as a preliminary step in fitting 4D representations for generation. In this paper, we explore an efficient approach to directly encode 4D mesh data for generative modeling in a single pass.

Video-to-4D generation. Early attempts [1, 28, 56, 63] at video-to-4D generation predominantly relied on optimization-based approaches, utilizing pre-trained generative priors [40, 60, 61, 73] as guidance. These methods typically employ score distillation [53] techniques to optimize either neural volumetric representations [43] or 3D Gaussian Splatting [32]. These methods suffer from lengthy optimization times and SDS-related issues [39, 75] such as spatial-temporal inconsistency or poor input alignments. While some works [87, 88] use pseudo-labels for better consistency, recent approaches [38, 49, 57, 82, 85, 94] directly reconstruct 4D content from multiview images or videos. Notably, the introduction of large-scale 4D reconstruction models [57] has significantly reduced the generation time from hours to seconds. However, most of these approaches often struggle to maintain consistent quality across temporal sequences due to inherent multiview inconsistency in 2D generation results.

###### SLAT 𝑺𝟏 GS 𝑮𝟏 GS Variation Field 𝑽𝒕 Pred. GS 𝑮𝒕

GS Disp. ∆𝒑𝒕𝒊𝒏𝒕𝒆𝒓𝒑 ∆𝒑𝒕𝒇𝒑𝒔 ∆𝒑𝒕𝒊𝒏𝒕𝒆𝒓𝒑

[Figure 45]

[Figure 46]

[Figure 47]

CanonicalGSEncoder

CanonicalGSDecoder

Mesh-guidedInterp.

|Apply to 𝑮𝟏<br><br>|
|---|

|𝓛𝒎𝒈|
|---|

FPS

|𝑮Pos.Emb.𝟏|
|---|

𝑮Pos.Emb.𝟏

###### Render

[Figure 48]

𝑮𝟏 Pos. Emb.

[Figure 49]

[Figure 50]

Query

𝑮𝟏

Cross Attention

[Figure 51]

Query

KV

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

CrossAttention

SelfAttention

|𝑷𝟏 Pos. Emb.|
|---|

Flatten

Surface

|𝓛𝒊𝒎𝒈|
|---|

… KV

[Figure 56]

Image 𝑰𝒕𝒓𝒆𝒏𝒅𝒆𝒓

Point Cloud 𝑷𝟏

3D Asset 𝑴𝟏

[Figure 57]

Sampling

[Figure 58]

[Figure 59]

Image 𝑰𝒕𝒈𝒕

Latent 𝔃

3D Asset 𝑴𝒕

Point Cloud 𝑷𝒕

Point Disp. ∆𝑷𝒕

[Figure 60]

[Figure 61]

: Subtraction : Position Encoding : Trainable : Frozen

Figure 2. Framework of 4DMesh-to-GS Variation Field VAE. Our VAE directly encodes 3D animation data into Gaussian Variation Fields within a compact latent space, optimized through image-level reconstruction loss and the proposed mesh-guided loss.

### 3. Method

mesh animations M = {Mt}Tt=1, we first convert them to point clouds P = {Pt|Pt ∈ RN×3}Tt=1 through uniform surface sampling, where each point cloud contains N points. The displacement fields {∆Pt|∆Pt ∈ RN×3}Tt=1 are computed as temporal differences of corresponding points between frames:

Given an input video sequence I = {It}Tt=1 of an object, our goal is to generate a sequence of 3DGS models

G = {Gt}Tt=1 that captures both the shape, appearance, and motion of the object. We decompose this task into canonical

GS G1 creation (using the first frame as canonical) and Gaus-

∆Pt = Pt − P1, (1)

sian Variation Fields V = {∆Gt}Tt=1 generation, where V describes each Gaussian’s attribute variations relative to G1 over time. Our framework comprises two main components: (1) a direct 4DMesh-to-GS Variation Field VAE that efficiently encodes 3D animation sequences into a compact latent space, and (2) a Gaussian variation field diffusion model that learns the latent distribution of variation fields conditioned on the input video and canonical GS. The following sections detail each component.

where P1 is the canonical frame’s point cloud. We then leverage a pretrained mesh-to-GS autoencoder EGS and DGS in [81] to obtain the canonical GS representation from canon-

ical mesh M1:

###### S1 = EGS(M1), G1 = DGS(S1),

(2)

where G1 ∈ RN

G×14 denotes the Gaussian parameters including positions p1, scales s1, rotation q1, colors c1, and opacity α1, with NG being the total number of canonical Gaussians. S1 is the structured latent (SLAT) representation for canonical GS (more details are included in the supplementary). We finetune DGS to ensure coherent canonical GS reconstruction with their variation fields, while keeping EGS frozen to leverage pretrained canonical GS diffusion models.

#### 3.1. Direct 4DMesh-to-GS Variation Field VAE

Extending 3DGS to generative modeling of dynamic content presents significant challenges. Fitting individual dynamic

- 3DGS representations for each animation instance is computationally expensive and scales poorly. Additionally, directly modeling temporal deformation of GS sequences with diffusion models is challenging due to the high dimensionality of both Gaussian quantities (e.g., typically over 100K in [32]) and the time dimension. Therefore, we propose an efficient autoencoding framework that directly encodes 3D animation data into Gaussian Variation Fields with a compact latent space, facilitating subsequent diffusion modeling. Gaussian Variation Field encoding. Given a sequence of

Inspired by 3DShape2VecSet [91], we employ a crossattention layer to aggregate motion information from 3D animation sequences into a fixed-length latent representation. While directly using G1 as query vectors is a straightforward approach, we find it leads to poor motion awareness.

To enhance the network’s sensitivity to mesh deformation, we introduce a mesh-guided interpolation mechanism that generates motion-aware query vectors based on the spatial correspondence between G1 and P1.

Specifically, for each canonical Gaussian position pi1, we identify its K nearest neighbors in the canonical point cloud P1 and compute their distances di,k. To handle varying point densities across the mesh-sampled point cloud, we introduce an adaptive radius ri that adjusts the influence region based on the local point distribution. The interpolation weight wi,k and adaptive radius ri are formulated as:

βdi,k ri2

wi,k = exp(−

), ri =

K

1 K

di,k, (3)

k=1

where β is a hyperparameter controlling the decay rate of interpolation weights with distance, with larger values producing more localized influence regions. We set β = 7.0 in this paper.

We then interpolate the displacement fields ∆Pt for the i-th Gaussian at time t:

K

∆pinterpt,i =

k=1

wi,k k wi,k

###### ∆Pt,n(i,k) (4)

where n(i,k) denotes the k-th nearest neighbor index. We perform farthest point sampling to ∆pinterpt based on their canonical positions to formulate our motion-aware query ∆pfpst ∈ RL×3 with reduced sequence length. The point cloud displacement fields ∆Pt serve as keys and values in the cross attention encoder. To preserve spatial relationships, we incorporate positional embedding PE(·) based on the canonical positions:

Qe = fdisp(∆pfpst ) + PE(G1), Ke = Ve = fdisp(∆Pt) + PE(P1),

z = CrossAttn(Qe,Ke,Ve),

(5)

where fdisp is the displacement embedding layer. This process yields a latent representation z ∈ RT×L×C, where T is the number of temporal frames, L is the latent size, and C is the feature dimension. Notably, our encoding procedure compresses the sequence length from N = 8192 to L = 512, significantly reducing the subsequent diffusion modeling space.

Gaussian Variation Field decoding. The decoding procedure first transforms the latent representation through n layers of self-attention blocks to enable thorough motion information exchange. The decoder then maps this processed latent to a Gaussian Variation Field V, defined by the variations of Gaussian attributes ∆Gt = {∆pt,∆st,∆qt,∆ct,∆αt}Tt=1. To ensure the decoder is

[Figure 62]

Figure 3. Architecture of Gaussian Variation Field diffusion model. Our model is built upon diffusion transformer, which takes noised latent as input and gradually denoises it conditioned on the video sequence and canonical GS.

aware of all canonical Gaussian attributes, we use all parameters of G1 to query the latent output through a cross attention layer:

Qd = fgs(G1) + PE(G1), Kd = Vd = zn, ∆Gt = CrossAttn(Qd,Kd,Vd),

(6)

where fgs is the embedding layer for the canonical Gaussians and zn is the final self attention layer output. The final 3DGS sequence is obtained by:

G = {Gt}Tt=1 = {G1 + ∆Gt}Tt=1. (7) Training objective. Our training objective consists of three main components. First, we employ image-level reconstruction loss between the rendered images Itrender from final predicted Gaussians and ground-truth images Itgt:

Limg = L1 + λlpipsLlpips + λssimLssim, (8)

where λlpips,λssim are loss weights for perception loss [96] and SSIM loss, respectively. To ensure faithful motion reconsturction, we introduce a mesh-guided loss that aligns the predicted Gaussian displacements with pseudo ground-truth ∆pinterpt obtained through mesh-guided interpolation:

Lmg =

T

∥∆pt − ∆pinterpt ∥22, (9)

t=1

which we find is crucial for motion reconstruction quality. Finally, to facilitate subsequent diffusion training, we also

Table 1. Quantitative comparison of video-to-4D generation results. Our method demonstrates consistent performance improvements across all metrics while maintaining efficient generation speed. The generation time is measured on a single A100 GPU.

###### Method PSNR↑ LPIPS↓ SSIM↑ CLIP↑ FVD↓ Time↓

Consistent4D [28] 16.20 0.146 0.880 0.910 935.19 ∼1.5 hr

SC4D [79] 15.93 0.164 0.872 0.870 833.15 ∼20 min STAG4D [88] 16.85 0.144 0.887 0.893 1008.40 ∼1 hr

DreamGaussian4D [56] 15.24 0.162 0.868 0.904 799.56 ∼15 min

L4GM [57] 17.03 0.128 0.891 0.930 529.10 3.5 s Ours 18.47 0.114 0.901 0.935 476.83 4.5s

regularize the latent distribution with a KL divergence loss Lkl. The total loss is: Ltotal = Limg + λmgLmg + λklLkl, where λmg,λkl are respective loss weights.

#### 3.2. Gaussian Variation Field Diffusion

The diffusion process can be formalized as the inversion of a discrete-time Markov forward process. Let z0 ∈ RT×L×C denote our initial latent of Gaussian Variation Field from the distribution p(z). During the forward phase, we progressively corrupt this latent sequence by adding Gaussian noise over diffusion steps s ∈ [0,S], following zs := αsz0 +σsϵ, where ϵ ∼ N(0,I), and αs,σs define the noise schedule. After sufficient diffusion steps, zS approaches pure Gaussian noise. Generation is achieved by reversing this process, starting from random Gaussian noise zS ∼ N(0,I) and progressively denoising it to recover z0.

The compact latent space enables us to build our diffusion model upon the powerful Diffusion Transformer (DiT) architecture [52]. As illustrated in Figure 3, the model takes noise-corrupted latent as input, and processes them through a series of transformer blocks for denoising. Each transformer block incorporates diffusion timestep information through adaptive layer normalization (adaLN) and a gating mechanism. Beyond the standard spatial self attention layers, we introduce dedicated temporal self-attention layers to ensure coherent motion generation across the sequence.

To condition the generation process, we inject two types of features through cross-attention layers: (1) visual features Cv = {Ctv}Tt=1 extracted from input video frames using DINOv2 [48], and (2) geometric features CGS = Gfps1 fartherest sampled from the static GS. We further incorporate positional embeddings based on canonical GS positions pfps1 in our diffusion transformer, which strengthens the model’s awareness of correspondences between canonical GS and their variation fields during the denoising process, thereby effectively improving the generation quality.

We parameterize our diffusion model vˆθ to predict the velocity vs := αsϵ − σsz0 at each diffusion step s. The diffusion model is trained using:

Lsimple = Es,z0,ϵ v ˆθ αsz0 + σsϵ,s,C − vs 22 , (10)

where C = {Cv,CGS} represents the conditional features of both Cv and CGS.

#### 3.3. Inference Pipeline

During inference, our framework operates in a sequential pipeline. First, we obtain the canonical GS G1 for the first frame using a pretrained 3D diffusion model [81]. Given an input video sequence {It}Tt=1, we extract visual features and combine them with the farthest sampled canonical Gaussians

- as conditioning signals for our diffusion model. The diffusion model generates latent codes z, which are subsequently decoded to obtain the Gaussian Variation Field V. The fi-

nal animated Gaussian representation Gt for each frame is obtained by applying these variations to the canonical Gaussians, effectively creating high-fidelity temporally coherent 4D animations.

4. Experiments

- 4.1. Dataset and Metrics

We conduct our experiments on Objaverse-V1 and ObjaverseXL [13], following previous work in 4D content generation. After filtering for objects with high-quality animations, we utilize 34K objects for training. To evaluate the video-to-4D generation quality, we construct a comprehensive test set of 100 objects by combining 7 instances from the widely-used Consistent4D [28] testset with 93 additional test instances from Objaverse-XL, ensuring a thorough evaluation with previous works. We render 4 novel views of each timestep for each instance. We assess the generation quality using multiple metrics: PSNR, LPIPS [96], and SSIM for frame-wise quality, and FVD [70] for temporal consistency of the generated sequences. All evaluations are performed on renderings at 512 × 512 resolution.

- 4.2. Implementation Details

For our VAE implementation, the canonical Mesh-to-GS autoencoder is builds upon TRELLIS [81], with training con-

ducted in two stages: finetuning the sparse GS decoder DGS on canonical 3D only for 150K iterations, followed by joint training with other modules for 200K iterations on 4D animation data. The VAE architecture employs point cloud

[Figure 63]

Figure 4. Qualitative comparison with previous state-of-the-art methods. Our model directly learns the distribution of Gaussian Variation Fields, enabling high-fidelity 4D generation with coherent temporal dynamics.

size N = 8192, latent size L = 512, and feature dimension C = 16. We optimize the VAE using AdamW with a learning rate 5e − 6 and 5e − 5 for DGS and other modules respectively, using batch size 32. The diffusion model is trained on 24-frame sequences using AdamW optimizer with identical learning rate and batch size over 1300K iterations. We apply the cosine noise schedule [45] with 1000 timesteps for training the diffusion model. We set T = 24 for training, and T = 32 during inference to compare with prior works. For more implementation details, please refer to the supplementary materials.

- 4.3. Main Results Quantitative comparisons. We compare the video-to-

- 4D generation results of our model with previous stateof-the-art methods including both optimization-based approaches [28, 56, 79, 88] and feedforward approach [57]. As shown in Table 1, our method consistently outperforms existing approaches across all quality metrics, demonstrating both superior reconstruction fidelity and better temporal coherence. Unlike some prior works [28, 56, 79, 88] require minutes to hours of optimization, our approach is also more

efficient, taking 4.5 seconds to generate a 4D animation sequence (3.0 seconds for canonical GS creation and 1.5 seconds for Gaussian Variation Field diffusion), only slightly slower than feedforward reconstruction method L4GM [57]. These quantitative results collectively validate both the effectiveness and efficiency of our proposed method.

Qualitative comparisons. We also provide qualitative comparisons with previous state-of-the-art methods in Figure 4. The SDS-based approaches [28, 88] turn to generate results with blurry textures and poor geometry. The feedforward method L4GM leverages multiview images generated from 2D generative prior [61] to reconstruct the 4DGS sequences. However, the results of L4GM suffer from 3D inconsistency of the generated multiview images. In contrast, our model directly generates the canonical GS and the Gaussian Variation Fields, capable of creating high-fidelity 3D consistent animations with coherent temporal dynamics.

More visualization of generated results. Figure 5 presents additional generation results from our method, including examples conditioned on both in-the-wild videos (left two cases) and test set videos (right two cases). Our model demonstrates high-quality generation capability with faithful

[Figure 64]

Figure 5. More generation results of our model including in-the-wild videos (left) and videos from test set (right).

#### 4.4. Ablation Study

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Ablation of our VAE. In Table 2 and Figure 6, we analyze key components of our VAE. Our baseline (Config. A) starts with using positions of farthest sampled canonical GS pfpst as query for the encoder’s cross-attention layer, with variation attributes limited to positions ∆pt, scaling ∆st, and rotation ∆qt following previous 4DGS works [76]. Since we do not have ground-truth GS motion for explicit supervision, the VAE initially struggles with motion learning. After equipped with our mesh-guided loss, the motion reconstruction capability is effectively improved through pseudo displacement supervision (Config. B). We then replace the encoding query with motion-aware ∆pfpst using mesh-guided interpolation, which successfully handles most of the motion sequences (Config. C). Finally, to give the model more flexibility to handle complex motion sequence, we incorporate color ∆ct and opacity ∆α of Gaussian attributes to the variation fields, which further enhance the

Config A. Config B. Config C. Config D. (Ours) Ground-truth

Figure 6. Visual ablation of VAE.

motion reproduction. Despite being trained on synthetic data, the model exhibits strong generalization capability by effectively capturing motion patterns from in-the-wild video inputs. Furthermore, the model successfully handles challenging multi-object scenarios, highlighting the robustness of our approach.

Table 2. Ablation study of key factors in our VAE. Config. Encoder Query Type Mesh-guided Loss Variation Attrs. PSNR↑ LPIPS↓ SSIM↑

- A. pfpst ✗ ∆pt, ∆st, ∆qt 23.25 0.0678 0.936
- B. pfpst ✓ ∆pt, ∆st, ∆qt 26.17 0.0544 0.950
- C. ∆pfpst ✓ ∆pt, ∆st, ∆qt 28.58 0.0478 0.958

D. (Ours) ∆pfpst ✓ ∆pt, ∆st, ∆qt, ∆ct, ∆αt 29.28 0.0439 0.964

Time 1 Time 2 Time 3

Time 1 Time 2 Time 3

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Input Ours

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

InputView

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

NovelView

Ours

Figure 7. Our model is also capable of creating animations for existing 3D assets with conditional videos.

Table 3. Ablation study of our diffusion model.

real-world applications, the users can first generate 2D animations from the rendered images of their 3D models using off-the-shelf video diffusion models [18, 33, 34, 47], then employ our model to create corresponding 4D animations.

Method PSNR↑ LPIPS↓ SSIM↑ CLIP↑ FVD↓ Ours w/o Pos. Emb. 17.86 0.121 0.897 0.931 547.20 Ours 18.47 0.114 0.901 0.935 476.83

### 5. Conclusion

reconstruction capability of VAE.

Ablation of our diffusion model. We examine the importance of positional embeddings in our diffusion model training in Table 3. By incorporating positional prior based on canonical GS positions pfps1 , the diffusion transformer better captures the correspondence between spatial positions and their variations. Removing these positional embeddings leads to a significant performance drop, demonstrating their crucial role in achieving high-quality results.

#### 4.5. Application

Despite being trained on single video inputs, our model is effective at animating existing 3D models according to the motions depicted in conditioned videos. More details are included in the supplementary materials. As demonstrated in Figure 7, this approach produces high-quality animations that faithfully reproduce the target motions. Therefore, for

In this paper, we introduce a novel framework to address the challenging task of 4D generative modeling. To efficiently construct the large-scale training dataset and reduce the modeling difficulty for diffusion, we first introduce a Direct 4DMesh-to-GS Variation Field VAE, which is able to efficiently compress complex motion information into a compact latent space without requiring costly per-instance fitting. Then, a Gaussian Variation Field diffusion model that generates high-quality dynamic variation fields conditioned on input videos and canonical 3DGS. By decomposing 4D generation into canonical 3DGS generation and Gaussian Variation Field modeling, our method significantly reduces computational complexity while maintaining high fidelity. Quantitative and qualitative evaluations demonstrate that our approach consistently outperforms existing methods. Furthermore, our model exhibits remarkable generalization capability with in-the-wild video inputs, advancing the state of high-quality animated 3D content generation.

Acknowledgments. We extend our gratitude to all the reviewers for their constructive feedback. We also appreciate Jiaqi Lou for the assistance with chart refinement and the production of the supplementary video.

### References

- [1] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7996–8006,

2024. 2

- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1
- [3] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 1, 2
- [4] Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 130–141, 2023. 2
- [5] Wei Cao, Chang Luo, Biao Zhang, Matthias Nießner, and Jiapeng Tang. Motion2vecsets: 4d latent vector set diffusion for non-rigid shape reconstruction and tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20496–20506, 2024. 2
- [6] Ziang Cao, Fangzhou Hong, Tong Wu, Liang Pan, and Ziwei Liu. Large-vocabulary 3d diffusion model with transformer. arXiv preprint arXiv:2309.07920, 2023. 2
- [7] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16123–16133, 2022. 2
- [8] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. ArXiv preprint, abs/2303.13873, 2023. 2
- [9] Zhaoxi Chen, Fangzhou Hong, Haiyi Mei, Guangcong Wang, Lei Yang, and Ziwei Liu. Primdiffusion: Volumetric primitives diffusion for 3d human generation. Advances in Neural Information Processing Systems, 36:13664–13677,

2023. 2

- [10] Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao, Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie, Tong Wu, Shunsuke Saito, et al. 3dtopia-xl: Scaling highquality 3d asset generation via primitive diffusion. arXiv preprint arXiv:2409.12957, 2024. 2

- [11] R James Cotton and Colleen Peyton. Dynamic gaussian splatting from markerless motion capture reconstruct infants movements. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 60–68,

2024. 2

- [12] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36:35799–35813, 2023. 2, 15
- [13] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13142–13153, 2023. 2, 5, 15
- [14] Yu Deng, Jiaolong Yang, Jianfeng Xiang, and Xin Tong. Gram: Generative radiance manifolds for 3d-aware image generation. In IEEE/CVF International Conference on Computer Vision, 2022. 2
- [15] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021. 1
- [16] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12479–12488, 2023.

- 2

[17] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high quality

- 3d textured shapes learned from images. arXiv preprint arXiv:2209.11163, 2022. 2

- [18] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Factorizing text-to-video generation by explicit image conditioning. In European Conference on Computer Vision, pages 205–224. Springer,

2024. 8, 15

- [19] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 1
- [20] Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas O˘guz. 3dgen: Triplane latent diffusion for textured mesh generation. arXiv preprint arXiv:2303.05371, 2023. 2
- [21] Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. Efficient diffusion training via min-snr weighting strategy. arXiv preprint arXiv:2303.09556, 2023. 1
- [22] Xianglong He, Junyi Chen, Sida Peng, Di Huang, Yangguang Li, Xiaoshui Huang, Chun Yuan, Wanli Ouyang, and Tong He. Gvgen: Text-to-3d generation with volumetric representation. arXiv preprint arXiv:2403.12957, 2024. 2

- [23] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 1
- [24] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023. 1
- [25] Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4220–4230, 2024. 2
- [26] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, et al. Perceiver io: A general architecture for structured inputs & outputs. arXiv preprint arXiv:2107.14795, 2021. 2
- [27] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pages 4651–4664. PMLR, 2021. 2
- [28] Yanqin Jiang, Li Zhang, Jin Gao, Weimin Hu, and Yao Yao. Consistent4d: Consistent 360 {\deg} dynamic object generation from monocular video. arXiv preprint arXiv:2311.02848, 2023. 2, 5, 6
- [29] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463, 2023. 2
- [30] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4401–4410, 2019. 1
- [31] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020. 1
- [32] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), 2023. 2, 3
- [33] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. Hunyuanvideo: A systematic framework for large video generative models, 2025. 8, 15

- [34] Kuaishou. Kling. https://klingai.kuaishou.com, 2024. 8, 14, 15
- [35] Yushi Lan, Fangzhou Hong, Shuai Yang, Shangchen Zhou, Xuyi Meng, Bo Dai, Xingang Pan, and Chen Change Loy. Ln3diff: Scalable latent neural fields diffusion for speedy 3d generation. In ECCV, 2024. 2
- [36] Mengtian Li, Shengxiang Yao, Zhifeng Xie, Keyu Chen, and Yu-Gang Jiang. Gaussianbody: Clothed human reconstruction via 3d gaussian splatting. arXiv preprint arXiv:2401.09720, 2024. 2
- [37] Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979, 2024. 2
- [38] Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N Plataniotis, Yao Zhao, and Yunchao Wei. Diffusion4d: Fast spatial-temporal consistent 4d generation via video diffusion models. arXiv preprint arXiv:2405.16645, 2024. 2, 15
- [39] Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. Luciddreamer: Towards highfidelity text-to-3d generation via interval score matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6517–6526, 2024. 2
- [40] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298–9309, 2023. 2
- [41] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787,

2022. 14

- [42] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. arXiv preprint arXiv:2308.09713, 2023. 2
- [43] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2
- [44] Norman M¨uller, Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulo, Peter Kontschieder, and Matthias Nießner. Diffrf: Rendering-guided 3d radiance field diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4328–4338, 2023. 2
- [45] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pages 8162–8171. PMLR,

2021. 1, 6

- [46] Evangelos Ntavelis, Aliaksandr Siarohin, Kyle Olszewski, Chaoyang Wang, Luc Van Gool, and Sergey Tulyakov. Autodecoding latent 3d diffusion models. arXiv preprint arXiv:2307.05445, 2023. 2

- [47] OpenAI. Video generation models as world simulators. https://openai.com/index/video-generation-models-asworld-simulators, 2024. 8, 15
- [48] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2, 5, 14
- [49] Zijie Pan, Zeyu Yang, Xiatian Zhu, and Li Zhang. Efficient4d: Fast dynamic 3d object generation from a singleview video. arXiv preprint arXiv:2401.08742, 2024. 2
- [50] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5865–5874, 2021. 2
- [51] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo MartinBrualla, and Steven M Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228, 2021. 2
- [52] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 2, 5, 14

- [53] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2
- [54] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10318–10327, 2021. 2
- [55] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 1

- [56] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142,

2023. 2, 5, 6

- [57] Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling, et al. L4gm: Large 4d gaussian reconstruction model. Advances in Neural Information Processing Systems, 37:56828–56858, 2025. 2, 5, 6, 14
- [58] Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams. Xcube: Large-scale 3d generative modeling using sparse voxel hierarchies. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4209–4219, 2024. 2
- [59] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 1, 2
- [60] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and

- Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023. 2
- [61] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 2, 6
- [62] J Ryan Shue, Eric Ryan Chan, Ryan Po, Zachary Ankner, Jiajun Wu, and Gordon Wetzstein. 3d neural field generation using triplane diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20875–20886, 2023. 2
- [63] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280,

2023. 2

- [64] Ivan Skorokhodov, Aliaksandr Siarohin, Yinghao Xu, Jian Ren, Hsin-Ying Lee, Peter Wonka, and Sergey Tulyakov. 3d generation on imagenet. arXiv preprint arXiv:2303.01416,

2023. 2

- [65] Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, and Yebin Liu. Dreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior. arXiv preprint arXiv:2310.16818, 2023. 2
- [66] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653,

2023. 1

- [67] Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. Make-it-3d: High-fidelity 3d creation from a single image with diffusion prior. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22819–22829, 2023. 2
- [68] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. arXiv preprint arXiv:2402.05054, 2024. 1
- [69] Zhicong Tang, Shuyang Gu, Chunyu Wang, Ting Zhang, Jianmin Bao, Dong Chen, and Baining Guo. Volumediffusion: Flexible text-to-3d generation with efficient volumetric encoder. arXiv preprint arXiv:2312.11459, 2023. 2
- [70] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. 2019. 5
- [71] Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, Karsten Kreis, et al. Lion: Latent point diffusion models for 3d shape generation. Advances in Neural Information Processing Systems, 35:10021–10039, 2022. 2
- [72] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017. 2, 14
- [73] Peng Wang and Yichun Shi. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201, 2023. 2
- [74] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang

- Wen, Qifeng Chen, et al. Rodin: A generative model for sculpting 3d digital avatars using diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4563–4573, 2023. 2
- [75] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. ArXiv preprint, abs/2305.16213, 2023. 2
- [76] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20310–20320, 2024. 1, 2, 7
- [77] Jiajun Wu, Chengkai Zhang, Tianfan Xue, Bill Freeman, and Josh Tenenbaum. Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. Advances in Neural Information Processing Systems, 29,

2016. 2

- [78] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. arXiv preprint arXiv:2405.14832, 2024. 2
- [79] Zijie Wu, Chaohui Yu, Yanqin Jiang, Chenjie Cao, Fan Wang, and Xiang Bai. Sc4d: Sparse-controlled video-to-4d generation and motion transfer. In European Conference on Computer Vision, pages 361–379. Springer, 2024. 5, 6
- [80] Jianfeng Xiang, Jiaolong Yang, Yu Deng, and Xin Tong. Gram-hd: 3d-consistent image generation at high resolution with generative radiance manifolds. arXiv preprint arXiv:2206.07255, 2022. 2
- [81] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506, 2024. 1, 2, 3, 5, 14, 15, 16
- [82] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470, 2024. 2
- [83] Bojun Xiong, Si-Tong Wei, Xin-Yang Zheng, Yan-Pei Cao, Zhouhui Lian, and Peng-Shuai Wang. Octfusion: Octreebased diffusion models for 3d shape generation. arXiv preprint arXiv:2408.14732, 2024. 2
- [84] Haitao Yang, Yuan Dong, Hanwen Jiang, Dejia Xu, Georgios Pavlakos, and Qixing Huang. Atlas gaussians diffusion for 3d generation with infinite number of points. arXiv preprint arXiv:2408.13055, 2024. 2
- [85] Zeyu Yang, Zijie Pan, Chun Gu, and Li Zhang. Diffusion 2: Dynamic 3d content generation via score composition of video and multi-view diffusion models. arXiv preprint arXiv:2404.02148, 2024. 2
- [86] Lior Yariv, Omri Puny, Oran Gafni, and Yaron Lipman. Mosaic-sdf for 3d generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4630–4639, 2024. 2

- [87] Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 4dgen: Grounded 4d content generation with spatial-temporal consistency. arXiv preprint arXiv:2312.17225, 2023. 2
- [88] Yifei Zeng, Yanqin Jiang, Siyu Zhu, Yuanxun Lu, Youtian Lin, Hao Zhu, Weiming Hu, Xun Cao, and Yao Yao. Stag4d: Spatial-temporal anchored generative 4d gaussians. In European Conference on Computer Vision, pages 163–179. Springer, 2024. 2, 5, 6
- [89] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019. 14
- [90] Bowen Zhang, Shuyang Gu, Bo Zhang, Jianmin Bao, Dong Chen, Fang Wen, Yong Wang, and Baining Guo. Styleswin: Transformer-based gan for high-resolution image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11304–11314, 2022. 1
- [91] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–16, 2023. 2, 3
- [92] Bowen Zhang, Yiji Cheng, Chunyu Wang, Ting Zhang, Jiaolong Yang, Yansong Tang, Feng Zhao, Dong Chen, and Baining Guo. Rodinhd: High-fidelity 3d avatar generation with diffusion models. arXiv preprint arXiv:2407.06938,

2024. 2

- [93] Bowen Zhang, Yiji Cheng, Jiaolong Yang, Chunyu Wang, Feng Zhao, Yansong Tang, Dong Chen, and Baining Guo. Gaussiancube: A structured and explicit radiance representation for 3d generative modeling. arXiv preprint arXiv:2403.19655, 2024. 1, 2
- [94] Haiyu Zhang, Xinyuan Chen, Yaohui Wang, Xihui Liu, Yunhong Wang, and Yu Qiao. 4diffusion: Multi-view video diffusion model for 4d generation. Advances in Neural Information Processing Systems, 37:15272–15295, 2025. 2
- [95] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 43(4):1–20, 2024. 1, 2
- [96] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018. 4, 5
- [97] Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, and Shenghua Gao. Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation. Advances in Neural Information Processing Systems, 36, 2024. 2
- [98] Xinyang Zheng, Yang Liu, Pengshuai Wang, and Xin Tong. Sdf-stylegan: Implicit sdf-based stylegan for 3d shape generation. In Computer Graphics Forum, pages 52–63, 2022. 2
- [99] Xin-Yang Zheng, Hao Pan, Peng-Shuai Wang, Xin Tong, Yang Liu, and Heung-Yeung Shum. Locally attentional

sdf diffusion for controllable 3d shape generation. ACM Transactions on Graphics (ToG), 42(4):1–13, 2023. 2

[100] Jun-Yan Zhu, Zhoutong Zhang, Chengkai Zhang, Jiajun Wu, Antonio Torralba, Josh Tenenbaum, and Bill Freeman. Visual object networks: Image generation with disentangled 3d representations. Advances in neural information processing systems, 31, 2018. 2

### A. Additional Implementation Details

#### A.1. Model Architecture

We will detail the architecture of each model below, with the summary demonstrated in Table 4.

- A.1.1. Gaussian Variation Field Encoder Our encoder mainly comprises two parts: the canonical GS

autoencoder EGS and DGS and a cross attention layer to create latent space for Gaussian Variation Fields.

For the canonical GS autoencoder, we adopt the model architecture from [81], which introduces a Structured Latent (SLAT) representation for static 3D assets. This representation defines a set of local latents on a 3D grid, where each latent is associated with an active voxel intersecting with the surface of the 3D asset. The SLAT representation effectively captures both the overall structure through active voxels and fine details through local latent codes. The canonical GS autoencoder is built using a transformer-based architecture. It first aggregates visual features from multiview images using a pre-trained DINOv2 [48] encoder to create voxelized features. These features are then processed through a sparse transformer encoder that handles variable-length tokens corresponding to active voxels. The transformer incorporates shifted window attention in 3D space to enhance local information interaction while maintaining computational efficiency. The encoder outputs structured latents that follow a regularized distribution through KL-divergence penalties, which are then decoded to various representations. For this work, we only leverage its Gaussian representation decoder for our canonical GS autoencoding. DGS is set to resolution 64, and decode to 8 Gaussians per voxel. We finetune the decoder DGS while keeping the encoder EGS frozen.

For the cross attention layer, we adopt the vanilla full attention [72] implementation. we set the motion-aware ∆pfpst ∈ R512×3 using proposed mesh-guided interpolation mechanism as query and point displacement fields ∆Pt ∈ R8192×3 from mesh as keys and values. Then the latent representation z ∈ R512×16 is obtained after the cross attention layer.

- A.1.2. Gaussian Variation Field Decoder

For the Gaussian Variation Field decoder, we first adopt 12 layers of vanilla self attention for thorough information exchange. For the last cross attention layer to decode Gaussian Variation Fields ∆Gt ∈ RN

G×14 The output feature of last self attention layer is set to keys and values, and we adopt all parameters of G1 ∈ RN

G×14 as query, where NG is the total number of canonical GS.

- A.1.3. Canonical GS Generation Model

We adopt the model architecture from [81] to generate structure latent representation for further decoding to canonical GS, which follows a two-stage process. First, a structure

generator creates the sparse structure by denoising a lowresolution feature grid using transformer blocks with adaptive layer normalization and cross-attention for condition injection. Then, a latent generator GL generates local latents for the given structure using a sparse transformer architecture with downsampling and upsampling blocks. These two generators both adopt RMSNorm [89] to the queries and keys (QK Norm.) in diffusion training. They are conditioned on image conditions through CLIP and DINOv2 features respectively, and are trained separately using a continuous flow matching objective. Since we freeze the EGS, we can directly leverage the pretrained image-to-3D model [81] to create canonical GS.

##### A.1.4. Gaussian Variation Field Diffusion Model

Our Gaussian Variation Field diffusion model builds upon the diffusion transformer architecture [52]. To enable temporal coherence in generation, we introduce a temporal selfattention layer that complements the existing cross-attention, spatial self-attention, and feedforward layers. For video sequence conditioning, we extract frame-wise features using DINOv2 [48] and incorporate the farthest-sampled canonical Gaussian Splatting to maintain awareness of the canonical 3D model. To enhance spatial consistency, we incorporate positional priors into the generation process. During training, we encode the Gaussian Variation Field latent along with their corresponding canonical GS positions to formulate positional embeddings. During inference, we directly utilize the positions from farthest-sampled Gaussian Splatting for positional embedding computation.

#### A.2. Additional Training and Inference Details

In this paper, we designate the first frame of each video as the canonical frame. For our Direct 4DMesh-to-GS Variation Field VAE training, we set the loss weights as follows: λlpips = 0.2, λssim = 0.2, λmg = 1.0, and λkl = 1e − 6. Computationally, the VAE training requires one day on 32 Nvidia Tesla V100 GPUs (32GB) for the first stage and two days on 8 Nvidia Tesla A100 GPUs (40GB) for joint training, while the diffusion model training spans approximately one week on 8 Nvidia Tesla A100 GPUs (80GB). During inference, we adopt the adaptive mode of DPM-Solver [41] with order 2, requiring approximately 18 steps per instance.

During inference, we address potential orientation misalignment between the generated canonical GS and input images through an azimuth alignment process similar to [57]. Specifically, we render the canonical GS from multiple azimuth angles and compute image-level losses between these renders and the first video frame. We then transform the canonical GS according to the azimuth angle that yields the minimal loss, ensuring better alignment with the input video.

The in-the-wild conditional videos shown in the teaser (Figure 1 in main paper) are created by Kling [34]. The walking astronaut and boxing rat video frames in Figure 5

Table 4. Detailed configuration of model architecture. SW and FFN denotes “Shifted Window” and “FeedForward Net”. MSA, MSSA, MTSA, MCA stand for “Multihead Self-Attention”, “Multihead Spatial Self-Attention”, “Multihead Temporal Self-Attention” and “Multihead Cross-Attention”, respectively.

###### Network #Layer #Dim. #Head Block Arch. Special Modules #Param.

EGS 12 768 12 3D-SW-MSA + FFN 3D Swin Attn. 85.8M DGS 12 768 12 3D-SW-MSA + FFN 3D Swin Attn. 85.1M

VAE Transformer 12 768 12 MSA / MCA + FFN - 125.21M Diffusion 12 512 16 MSSA + MTSA + MCA + FFN QK Norm. 105.51M

Table 5. Additional ablation of our proposed VAE.

of the main paper are sourced from consistent4D and Emu video [18], respectively.

Model PSNR↑ LPIPS↓ SSIM↑ Ours w/o DGS Finetuning 28.80 0.0460 0.962

#### A.3. Additional Details of Creating Animation for Existing 3D Model

###### Ours 29.28 0.0439 0.964

To animate existing 3D models using our approach, users follow a simple pipeline: First, their 3D assets are rendered as multiview images. These images are then processed to extract and aggregate DINOv2 features. Using these features, we construct a canonical Gaussian Splatting representation through our EGS encoder and DGS decoder. Finally, animations are generated by our diffusion model, which takes both the canonical GS and a conditional video as input. Users can create these conditional videos using state-of-the-art video diffusion models [18, 33, 34, 47] to specify their desired motion for the 3D model.

Table 6. Additional ablation of hyper-parameters in our meshguided interpolation.

###### K β PSNR↑ LPIPS↓ SSIM↑ K β PSNR↑ LPIPS↓ SSIM↑

16 7.0 28.38 0.0464 0.960 8 10.0 28.55 0.0462 0.961 8 7.0 29.28 0.0439 0.964 8 7.0 29.28 0.0439 0.964 4 7.0 28.94 0.0451 0.963 8 4.0 29.04 0.0446 0.963 1 7.0 28.22 0.0465 0.960 8 1.0 28.64 0.0457 0.962

t = 10 t = 40 t = 70 t = 120

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

### B. Data Preparation Details

Input Ours

Our training dataset consists of 34K 3D mesh animations sourced from Objaverse-V1 [13] and Objaverse-XL [12]. For Objaverse-V1, we utilize the curated set of 9K highquality 3D animations from [38]. For Objaverse-XL, we apply two filtering criteria: first, following [81], we filter out samples whose average aesthetic score 1 across 4 rendered views of the first frame falls below 5.5; second, we remove sequences with minimal motion. This filtering process yields 25K additional animations from Objaverse-XL.

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

NovelView

Figure 8. Sample of our autoregressive generation result.

neighbors K, and distance decay rate β of interpolation in Table 6. Our setting (K = 8,β = 7.0) yields optimal results. Performance is relatively stable for other values, showing reasonable robustness.

### C. Additional Ablation

Ablation of DGS Joint Finetuning. We investigate the importance of jointly finetuning the canonical GS decoder during our Direct 4DMesh-to-GS Variation Field VAE training. Starting from a pretrained canonical 3D DGS checkpoint, we compare two settings: freezing DGS while training other modules, and jointly training all modules (our approach). As shown in Table 5, joint training allows DGS to receive feedback from animation reconstruction rather than being limited to static data only. This ensures the canonical GS reconstruction coherent with its corresponding variation fields. Ablation of hyper-parameters in mesh-guided interpolation. We ablate the hyper-parameters including nearest

### D. More Results

#### D.1. Autoregressive Generation Results for Temporal Generalization

Temporal generalization is a known challenge in 2D/3D video generation. In our case, we can employ an autoregressive approach during inference for videos exceeding our training length: the GS from the last frame of a generated segment serves as the canonical GS for inferring the next segment’s variation fields, which allows for coherent long animations. We show a 120-frame generated sample using

1https://github.com/christophschuhmann/improved-aesthetic-predictor

Time 1 Time 2 Time 3 Time 1 Time 2 Time 3

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Input Ours

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

InputView Ours

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

NovelView

Video condition frame 0

Video condition frame 1

[Figure 116]

[Figure 117]

Figure 9. More generation results of real-world input videos.

such an approach in Figure 8.

#### D.2. VAE Reconstruction Results

- As illustrated in Figure 11, we demonstrate the reconstruction capabilities of our proposed Direct 4DMesh-to-GS Variation Field VAE. Our method efficiently encodes both canonical GS and their temporal variations from 4D meshes in a single pass, eliminating the need for time-consuming perinstance fitting procedures. The results demonstrate our model’s effectiveness in preserving both geometric fidelity and motion dynamics.

- D.3. More Visual Comparison with SOTA Methods

As illustrated in Figure 12, we provide extensive visual comparisons with state-of-the-art methods. Our approach demonstrates consistent superiority across diverse test cases, achieving better results in terms of both visual fidelity and temporal motion coherence.

- D.4. More Reults of Animating Existing 3D Models

As shown in Figure 13, we demonstrate additional results showcasing our method’s capability to animate existing 3D models using conditional videos. Our approach successfully extracts and transfers motion patterns from the input videos, generating high-fidelity animations that faithfully preserve both geometric and temporal characteristics.

- D.5.AdditionalResultsonReal-WorldVideoInputs

Canonical GS frame 0

Our Animation frame 1

Figure 10. Failure case. When the pretrained static 3D generative model produces canonical GS that are not well-aligned with the conditional video frames, our Gaussian Variation Field diffusion model struggles to bridge this inconsistency, resulting in suboptimal animations.

Although our model is trained on synthetic data, it effectively generalizes to real-world video inputs. Figure 9 presents additional results, demonstrating the model’s robust generalization capabilities.

### E. Borader Impact

Like all generative models, our video-to-4D generation framework requires careful consideration of societal implications. While we mitigate certain ethical concerns by training exclusively on synthetic 3D animations from the Objaverse dataset, thus avoiding privacy and copyright issues

associated with real-world data, we acknowledge potential risks. The ability to generate animated 3D content from videos could be misused for creating misleading content. We therefore emphasize the importance of establishing clear guidelines for the responsible deployment of video-to-4D generation technology.

### F. Limitation Discussion and Future Work

While our model demonstrates impressive results in videoto-4D generation, it has certain limitations. Our two-stage generation process first employs a pretrained static 3D generative model to create canonical Gaussian Splatting representations, which then serve as conditions for our diffusion model to generate Gaussian Variation Fields. A notable limitation arises when the static 3D generative model [81] produces canonical GS that exhibits discrepancies with the conditional video, such as mismatched head pose, incorrect eyes or light effects in Figure 10, potentially creating inconsistencies in the final animation. To address this limitation, future work could explore either fine-tuning the static 3D model to ensure better image alignment or developing an end-to-end 4D diffusion framework that jointly generates both the canonical representation and its temporal variations.

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Ground-truth

View1 Ours

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

- View1

Ground-truth

- View2

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

View2 Ground-truth

Ours

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

View1 Ours

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

- View1

Ground-truth

- View2

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

[Figure 165]

View2

Ours

Figure 11. Additional visual results of VAE reconstruction.

Time 1 Time 2 Time 1 Time 2 Time 1 Time 2 Time 1 Time 2

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

InputViewNovelViewNovelViewNovelViewNovelViewInputViewInputViewInputViewInputView

InputConsistent4DSTAG4DL4GMOurs

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

Figure 12. More visual comparison with SOTA Methods.

[Figure 238]

###### Figure 13. More results of animating existing 3D model input with conditional videos.

