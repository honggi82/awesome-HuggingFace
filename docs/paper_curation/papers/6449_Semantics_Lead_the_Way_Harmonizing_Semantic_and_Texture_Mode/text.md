# arXiv:2512.04926v2[cs.CV]5Dec2025

## Semantics Lead the Way: Harmonizing Semantic and Texture Modeling with Asynchronous Latent Diffusion

Yueming Pan1,2*‡, Ruoyu Feng3‡, Qi Dai2, Yuqi Wang3, Wenfeng Lin3, Mingyu Guo3, Chong Luo2†, Nanning Zheng1† 1IAIR, Xi’an Jiaotong University 2Microsoft Research Asia 3ByteDance

[Figure 1]

###### StageⅠ

###### Stage Ⅱ Stage Ⅲ

| | | |
|---|---|---|
| |𝐱1| |

| | |
|---|---|
|𝐬1| |
| | |

𝐬0

[Figure 2]

|100 × faster|
|---|

Δ𝑡 1.0 𝑡

| | |
|---|---|
|𝐳1| |
| | |

0.0

𝐳0

|33.3 × faster|
|---|

|𝒟𝑧|
|---|

0.0 1 − Δ𝑡 1.0 𝑡

(a) Semantic-First Diffusion

(b) Training convergence curve

Figure 1. (a) Overview of Semantic-First Diffusion (SFD). Semantics (dashed curve) and textures (solid curve) follow asynchronous denoising trajectories. SFD operates in three phases: Stage I – Semantic initialization, where semantic latents denoise first; Stage II – Asynchronous generation, where semantics and textures denoise jointly but asynchronously, with semantics ahead of textures; Stage III – Texture completion, where only textures continue refining. After denoising, the generated semantic latent s1 is discarded, and the final image is decoded solely from the texture latent z1. (b) Training convergence on ImageNet 256×256 without guidance. SFD achieves substantially faster convergence than DiT-XL/2 and LightningDiT-XL/1 by approximately 100× and 33.3×, respectively.

#### Abstract

noise schedules: semantics precede textures by a temporal offset, providing clearer high-level guidance for texture refinement and enabling natural coarse-to-fine generation. On ImageNet 256×256 with guidance, SFD achieves FID 1.06 (LightningDiT-XL) and FID 1.04 (1.0B LightningDiTXXL), while achieving up to 100× faster convergence than original DiT without guidance. SFD also improves existing methods like ReDi and VA-VAE, demonstrating the effectiveness of asynchronous, semantics-led modeling. Project page and code: https://yuemingpan.github.io/ SFD.github.io/.

Latent Diffusion Models (LDMs) inherently follow a coarse-to-fine generation process, where high-level semantic structure is generated slightly earlier than fine-grained texture. This indicates the preceding semantics potentially benefit the texture generation by providing a semantic anchor. Recent advances have integrated semantic priors from pretrained visual encoders to further enhance LDMs, yet they still denoise semantic and VAE-encoded texture synchronously, neglecting such ordering. Observing these, we propose Semantic-First Diffusion (SFD), a latent diffusion paradigm that explicitly prioritizes semantic formation. SFD first constructs composite latents by combining the compact semantic latent, which is extracted from pretrained visual encoder via a dedicated Semantic VAE, with the texture latent. The core of SFD is to denoise the semantic and texture latents asynchronously using separate

#### 1. Introduction

Latent Diffusion Models (LDMs) [37] have emerged as the leading approach for modeling visual signals, demonstrating remarkable performance in high-quality image synthesis [28, 32, 37]. LDMs comprise two key components: a Variational Autoencoder (VAE) [21] that compresses highdimensional visual signals into a compact latent space, and a diffusion model that learns the distribution of this latent space. However, this design presents an inherent challenge.

*This work was performed during Yueming Pan’s internship at MSRA. ‡Equal contribution. †Corresponding author.

The VAE, optimized for pixel-level reconstruction, predominantly captures low-level texture features in its latent representation. Consequently, the diffusion model faces a conflicting objective: it must simultaneously capture high-level semantic understanding while preserving low-level textural details, which leads to slow convergence and suboptimal generation quality.

To overcome these challenges, recent studies enhance LDMs with discriminative semantic priors from pretrained visual encoders, enabling faster convergence and improved generation quality. These approaches typically achieve this by explicitly aligning semantic representations with the VAE latent space [49] or diffusion intermediate features [24, 50], or by jointly modeling concatenated semantic and texture representations within the diffusion process [22, 47]. All these methods share a similar paradigm, i.e., all latent information, including high-level semantics and low-level textures, is denoised synchronously at the same noise level throughout the diffusion process. However, such design deviates from the inherent nature of diffusion model, which follows a coarse-to-fine mechanism that progressively generates low-frequency structure before high-frequency textures [30, 36, 41, 46]. Inspired by this natural property, we argue that discriminative semantics, which capture structural and high-level information, should not only be embedded in the latent space as part of the denoising target, but should also actively lead the generation process by evolving earlier than textures. This philosophy is akin to the principle that one should first draw a blueprint before engaging in fine decoration, rather than attempting to simultaneously define structure and detail from chaos.

In this paper, we propose to explicitly intervene in the order of information formation during generation, where discriminative semantics are synthesized first and serve as priors to guide texture generation. However, a hard sequential generation scheme would exhibit training-inference mismatch similar to exposure bias in teacher forcing [34]: the model is trained with ground-truth semantic conditions but must generate based on its own imperfect predictions at inference, leading to performance degradation. To address this, we introduce asynchronous diffusion to harmonize the joint modeling of semantics and textures: semantics evolve ahead to guide texture synthesis, while both denoise simultaneously at different noise levels.

Motivated by these insights, we propose Semantics-First Diffusion (SFD), a new paradigm for LDMs that consists of two key components. First, an explicitly constructed composite latent space that combines discriminative semantics and low-level textures. Second, asynchronous diffusion guided by cleaner semantic information. Specifically, for the first component, building upon a texture VAE (e.g., SDVAE), we introduce a dedicated Semantic VAE (SemVAE) that compresses high-dimensional semantic representations

from pretrained visual encoders into a compact latent space, which is then concatenated with the texture latents. For the second component, as illustrated in Figure 1 (a), a threestage asynchronous denoising process is proposed. In the first stage, only the semantic latents are denoised, allowing the model to establish a coarse global layout initially. In the second stage, semantics and textures are denoised jointly but at different noise levels. Since semantic features evolve ahead, they provide stronger global guidance for texture refinement. In the third stage, with semantics fully denoised, only textures continue refining details. Finally, the output image is decoded solely from the texture latent.

Our contributions are summarized as follows:

- • We design a composite latent space composed of semantic latents from a dedicated Semantic VAE and texture latents from SD-VAE, where the Semantic VAE compresses high-level features from pretrained visual encoders into compact representations while largely preserving semantic integrity and spatial layout.
- • We propose the semantic-first asynchronous diffusion mechanism, which employs a three-stage denoising schedule where semantics evolve earlier and subsequently guide texture generation.
- • SFD achieves state-of-the-art FID score of 1.04 on ImageNet 256×256, while demonstrating 100× and 33.3× faster training convergence compared to DiT and LightningDiT, respectively.
- • We validate the effectiveness and generalizability of SFD by integrating it into existing synchronous diffusion models like ReDi and VA-VAE, thus improving their performance.

#### 2. Related Work

##### 2.1. Diffusion Models for Image Generation

Probabilistic diffusion models synthesize images by iteratively denoising from Gaussian noise. Early models [14, 42] operate in pixel space, suffering from high computational cost and slow convergence. Latent Diffusion Models (LDMs) [37] mitigate this by performing diffusion in a VAE-compressed latent space, greatly improving efficiency and visual fidelity. Building upon this foundation, DiT [32] and SiT [28] replaced the U-Net backbone [38] with Vision Transformers, demonstrating superior scalability and generative capacity. More recent efforts have sought to accelerate convergence by optimizing the diffusability of latent representations. For instance, [41] regularizes the frequency spectrum of the latent space to make it more compatible with diffusion dynamics. Despite these advances, standard LDMs still treat all latent components uniformly during denoising, leaving the coarse-to-fine nature [30] of the diffusion synthesis process implicit. In contrast, SFD explicitly models this hierarchical evolution via asynchronous de-

noising: high-level semantic components are denoised earlier and progressively guide the refinement of low-level textures at cleaner noise levels, thereby accelerating convergence and improving generation quality.

##### 2.2. Semantic Representation Enhanced Diffusion

Motivated by the discriminative gap between generative models and pretrained visual encoders, a parallel line of research seeks to enhance diffusion models with external semantic representations. REPA [50] performs featurespace alignment between diffusion features and pretrained visual encoders. REPA-E [24] extends this alignment by enabling end-to-end joint optimization of the VAE and diffusion model. REG [47] and ReDi [22] jointly learn the distribution of low-level VAE features and high-level semantic features from DINOv2 [31], where REG employs the class token as the semantic descriptor while ReDi adopts PCA-compressed patch embeddings. Another line of work focuses on improving or changing the VAE latent space. VA-VAE [49] aligns latent space with pretrained vision foundation models to to enrich its semantic representations. RAE [51] and SVG [40] replace the conventional VAE with pretrained visual encoder representations, where RAE adapts the diffusion transformer with a wide DDT head, while SVG employs a residual branch to capture finegrained details. Collectively, these approaches reveal that integrating semantic signals into the diffusion process benefits generation. We extend this paradigm further with SFD, which introduces asynchronous denoising schedules that allow semantics to be established earlier and guide the refinement of low-level textures throughout generation.

##### 2.3. Asynchronous Denoising Methods

Asynchronous denoising allows different components (e.g., tokens, spatial regions, pixels) to evolve under distinct noise schedules rather than enforcing strict synchronicity. Diffusion Forcing [2] assigns each token an independent noise level and enables arbitrary per-token denoising schedule, combining the strengths of next-token prediction models and full-sequence diffusion models. AsynDM [16] dynamically modulates per-pixel timestep schedules, allowing prompt-related regions to denoise more gradually and thereby improving text-to-image alignment. In this paper, SFD applies asynchronous denoising to semantic and texture subspaces within the latent representation, enabling early semantic guidance during denoising while preserving a simple and unified latent diffusion architecture.

#### 3. Method

We propose Semantic-First Diffusion (SFD), which employs asynchronous denoising to harmonize semantic and texture modeling, achieving faster convergence and superior performance without sacrificing reconstruction fidelity.

𝐟መ𝑠

Linear Proj

Transformer Decoder

Layer Norm

MSE Loss + Cos Sim Loss

|𝐬1<br><br>|
|---|

Transformer blocks

Transformer Encoder

Linear Proj

𝐟𝑠

|× 𝑁|
|---|

Figure 2. Architecture of the Semantic VAE (SemVAE). A Transformer-based VAE compresses high-dimensional vision foundation model (VFM) features into compact semantic latents.

Section 3.1 introduces the preliminaries. Section 3.2 describes our Semantic VAE and composite latent construction, and Section 3.3 presents the complete SFD framework.

##### 3.1. Preliminaries

Flow matching-based [1, 8, 27, 28] diffusion models learn to reverse a continuous noising process that transforms clean data into Gaussian noise. Following the flow matching formulation, the forward process is modeled as a linear interpolation:

xt = tx1 + (1 − t)x0, (1)

where x1 ∼ p(x) denotes clean data sampled from the data distribution, x0 ∼ N(0,I) denotes sampled Gaussian noise, and t ∈ [0,1] denotes time. Generation starts at t = 0 with random noise and follows the learned velocity field toward clean data at t = 1.

The evolution of xt is governed by the velocity field:

v(xt,t) = E[x˙t | xt] = E[x1 − x0 | xt]. (2)

This velocity field uniquely defines the associated ordinary differential equation (ODE) or stochastic differential equation (SDE) for sampling. It is also closely related to the score function s(xt,t) through a simple proportional relationship, so estimating either one is sufficient.

During training, a neural network vθ(xt,t) is optimized to approximate the velocity field by minimizing:

1

1,x0 ∥vθ(xt,t) − (x1 − x0)∥22 dt. (3)

Ex

Lvel(θ) =

0

During inference, a numerical solver integrates this learned velocity field from noise to data.

##### 3.2. Composite Latent Construction 3.2.1. Semantic VAE

To comprehensively leverage high-level semantics from pretrained vision foundation models, we introduce a dedicated Semantic VAE (SemVAE) specifically designed to

compress rich semantic features into compact latent representations while maintaining their spatial layout and minimizing information loss.

Figure 2 illustrates the architecture of SemVAE. Given an input image x1, we extract its patch-level semantic features fs = f(x1) ∈ RL×C

in via a frozen vision foundation model (VFM) denoted as f(·), where L denotes the number of flattened patches and Cin is the VFM feature dimension. The SemVAE encoder Es(·), consisting of a linear projection layer, four Transformer blocks, a LayerNorm layer, and an output linear layer, maps these features to a lower-dimensional latent space:

###### hs = Es(fs), (4) where hs ∈ RL×2C

s and Cs denotes the latent dimension. The factor of 2 accounts for the mean and variance parameters.

The encoder outputs the parameters of a Gaussian distribution. Specifically, hs is split into mean and variance:

###### µ,σ2 = hs[:,: Cs], hs[:,Cs :], (5) and the latent variable s1 ∈ RL×C

s is sampled via the reparameterization trick [20]:

###### s1 = µ + σ ⊙ ϵ, ϵ ∼ N(0,I). (6)

The SemVAE decoder Ds(·) mirrors the encoder architecture and reconstructs the original VFM features from the latent variables:

###### ˆfs = Ds(s1), ˆfs ∈ RL×C

. (7)

in

Training Objective. The SemVAE is trained with a combination of reconstruction and regularization losses. The reconstruction quality is ensured by MSE loss and cosine similarity loss:

ˆfs · fs ∥ˆfs∥∥fs∥

LMSE = ∥ˆfs − fs∥2, Lcos = 1 −

, (8)

where LMSE enforces reconstruction fidelity, while Lcos ensures directional alignment of the feature vectors. The KL divergence regularizes the latent space:

LKL = DKL(q(s1|fs)∥N(0,I))

- 1

- 2 i

µ2i + σi2 − log σi2 − 1 .

=

The total training loss is:

(9)

###### LSemVAE = LMSE + Lcos + λklLKL. (10)

λkl is set as 10−7 by default. Once trained, SemVAE is frozen during diffusion model training.

|ℰ𝑠|
|---|

𝐱1

𝐟𝑠 𝐬1

VFM

|ℰ𝑧|
|---|

𝐳1

Figure 3. Composite Latent Construction. An input image is encoded into semantic and texture latents via distinct VAE encoders, which are then concatenated to form a composite latent for asynchronous diffusion modeling.

- 3.2.2. Latent Construction As illustrated in Figure 3, the composite latent is constructed by combining compressed high-level semantics s1 and low-level textures z1, which are encoded via SemVAE encoder Es and texture VAE encoder Ez, respectively. Here we implement SD-VAE [37] as the texture VAE. The two latents are concatenated along the channel dimension:

c = [s1,z1] ∈ RL×(C

s+Cz), (11)

where [·,·] denotes channel-wise concatenation, and Cs, Cz are the semantic and texture channel dimensions.

- 3.3. Semantic-First Diffusion

The core motivation of Semantic-First Diffusion (SFD) is to enable semantic latents to be denoised ahead of texture features, thereby providing clearer structural guidance throughout the asynchronous generation process. We describe its key components below.

Distinct timesteps for semantics and textures. To model semantics and textures asynchronously with a fixed temporal offset ∆t while ensuring both timesteps remain within [0,1], distinct timesteps ts and tz are assigned to the semantic and texture latents during training. Specifically, for each image, we first sample the semantic timestep ts from an extended interval, then derive the texture timestep tz by subtracting the offset ∆t, and finally clamp both to [0,1]:

ts ∼ U(0, 1 + ∆t), (12) tz = max(0, ts − ∆t), (13) ts = min(ts, 1), (14)

which ensures ts,tz ∈ [0,1] and ts ≥ tz. This guarantees the semantic latent experiences less noise corruption than the texture latent at each denoising step, thereby providing clearer structural guidance for texture denoising.

Diffusion transformer with dual timesteps. As shown in Figure 4, the diffusion model adopts a Transformer backbone vθ(·) that takes as input the noisy composite latent

|[𝐬𝑡𝑠,𝐳𝑡𝑧]| |
|---|---|
| | |

𝐯ො𝑠

DiT

[𝑡𝑠,𝑡𝑧] 𝑦

𝐯ො𝑧

- Figure 4. Input and output of Diffusion Transformer. A DiT backbone takes as input a composite latent that combines noisy

semantic and texture features [sts, ztz], along with their respective timestep [ts, tz] and class label y. It jointly predicts the velocities of both semantics and textures.

] at different noise levels, two separate timesteps [ts, tz], and the class label y:

###### [st

###### ,zt

s

z

###### [vˆs,vˆz] = vθ [st

###### ,zt

s

###### ], [ts, tz], y , (15)

z

where vˆs and vˆz denote the predicted velocities of the semantic and texture components, respectively.

Training objective. The training objective combines velocity prediction losses for both semantic and texture latents:

0,s1,z0,z1,ts,tz v ˆz − (z1 − z0) 2

Lpred = Es

+ β v ˆs − (s1 − s0) 2 ,

(16)

where s0 ∼ N(0,I), z0 ∼ N(0,I) are sampled from the prior, and β is a weighting hyperparameter.

Additionally, the representation alignment loss from REPA [50] is employed, which aligns the diffusion hidden states with pretrained vision encoder representations. Formally, it is defined as:

LREPA(ψ,ϕ) := −Es

ts,ztz,ts,tz Lsim y∗, hϕ(ht) ,

(17) where y∗ = f(x1) denotes the pretrained visual encoder output, ht = fψ([st

],[ts,tz]) is the diffusion transformer encoder output, hϕ(ht) projects ht through a trainable projection head, and Lsim(·,·) is the alignment function. Notably, y∗ corresponds to the representation input for SemVAE (Section 3.2.1). Under this formulation, LREPA can be regarded as striving to reconstruct the noisy semantic latents st

###### ,zt

s

z

back to their clean representations y∗. Compared to the original REPA, which distills the VFM’s analytical capabilities, this explicit reconstruction from semantic latents offers a more tractable learning objective, thereby better preserving the integrity of semantic information and enabling more effective utilization of semantic knowledge.

s

The final objective becomes the following:

Ltotal = Lvel + λLREPA. (18)

Three-phase denoising schedule. During inference, SFD employs a three-phase asynchronous denoising schedule, as illustrated in Figure 1(a):

- 1. Semantic initialization, where ts ∈ [0,∆t),tz = 0: Only semantic latents are denoised to establish global structural guidance.
- 2. Asynchronous generation, where ts ∈ [∆t,1],tz ∈ [0,1 − ∆t): Both semantic and texture latents are denoised jointly yet asynchronously, with semantics advancing slightly ahead to provide clearer structural guidance for texture generation.
- 3. Texture completion, where ts = 1,tz ∈ [1 − ∆t,1]: With semantic latents fully denoised, noisy texture latents continue to refine fine-grained details.

Formally, two binary masks Ms ∈ {0,1}B×C

s×H×W and Mz ∈ {0,1}B×C

z×H×W are introduced to control the denoising updates of semantic and texture latents, respectively. According to the three-phase asynchronous denoising schedule, the masks (Ms,Mz) are defined as:

[Ms,Mz] =

 



- [1, 0], ts ∈ [0, ∆t), tz = 0,
- [1, 1], ts ∈ [∆t, 1], tz ∈ [0, 1 − ∆t),

[0, 1], ts = 1, tz ∈ [1 − ∆t, 1],

(19) where 1 and 0 denote all-one and all-zero tensors with shapes matching Ms and Mz, respectively. The masked velocity for updating is then computed as:

vˆ = Ms ⊙ vˆs,Mz ⊙ vˆz , (20)

where ⊙ denotes element-wise multiplication. This mechanism explicitly controls which latents denoise at each phase, ensuring semantic latents denoise earlier to guide texture refinement continuously. By enabling asynchronous yet coordinated updates between semantic and texture latents, SFD achieves more stable optimization and naturally aligns with the coarse-to-fine generation paradigm of diffusion models.

Notably, while SFD extends the denoising timestep range by ∆t, we proportionally increase the interval between successive steps, keeping the total number of diffusion steps fixed. Therefore, no additional denoising steps are required for inference. Upon completion, only the fully denoised texture latent z1 is decoded to the final image.

- 4. Experiments 4.1. Experimental Setup

Implementation details. We employ SD-VAE [37] f16d32 from LightningDiT [49] to encode texture into 32channel latents with 16× spatial downsampling, and the SemVAE encoder (29M parameters) to encode semantic features extracted by DINOv2-B with registers [4, 31] into 16-channel latents. The concatenated 48-channel representation forms a unified 256-token latent for each 256 × 256

[Figure 3]

- Figure 5. Effect of the temporal offset ∆t in asynchronous denoising. A moderate offset (∆t = 0.3) yields the lowest FID, indicating the best semantic–texture cooperation.

image. We adopt LightningDiT [49] as the diffusion backbone and train on ImageNet-1K [5] with a batch size of 256, a learning rate of 1 × 10−4, and the AdamW optimizer. We set β = 2.0 and ∆t = 0.3. For REPA, we set λ = 1.0, the alignment depth to 2, and use cosine similarity as the similarity function. For sampling, the dopri5 solver [7] with adaptive sampling steps is employed, following the implementation in LightningDiT1 [49]. The absolute and relative tolerances are set to 10−6 and 10−3, respectively. AutoGuidance [18] is used as the guidance method with a DiT-B degradation model. Implementation details of SemVAE and complete configurations of diffusion models are provided in Appendix A.

Evaluation protocol. We adopt comprehensive quantitative metrics to assess generation quality: Fr´echet Inception Distance (FID) [12] for visual realism, structural FID (sFID) [29] for spatial coherence, Inception Score (IS) [39] for class-conditional diversity, Precision (Prec.) for sample fidelity, and Recall (Rec.) for distribution coverage [23]. All metrics are computed on 50K generated samples following the standardized ADM [6] evaluation pipeline.

##### 4.2. Main Results

###### Effect of Temporal Offset in Asynchronous Denoising.

We analyze how the temporal offset ∆t between semantics and textures influences SFD performance. In this experiment, we set the learning rate to 2 × 10−4 and train for 400K iterations to accelerate convergence. As shown in Figure 5, when ∆t = 0, SFD degenerates to conventional joint denoising of semantic and texture representations, similar to ReDi [22] and REG [47]. As ∆t increases, FID gradually decreases and reaches its optimal value of 3.03 at ∆t = 0.3, corresponding to our proposed semantics-leadtexture scheme. In this setting, semantics evolve slightly ahead of textures, providing clearer global guidance while

1https://github.com/hustvl/LightningDiT

maintaining cooperative optimization. However, increasing ∆t beyond 0.3 progressively degrades performance. When ∆t = 1.0, the model reduces to a teacher-forcing sequential generation scheme, where semantics are fully synthesized before texture generation begins, leading to traininginference mismatch and suboptimal results. Overall, these results demonstrate that a moderate offset (∆t = 0.3) achieves the optimal trade-off between early semantic stabilization and texture collaboration, better harmonizing the joint modeling of semantics and textures and thereby yielding the best generation quality.

Accelerating training convergence. Table 1 presents a comprehensive comparison between DiT [32], LightningDiT [49], REPA [50], and our SFD on ImageNet 256×256 without guidance. Results for LightningDiT and its REPA variant are our reproductions. SFD consistently achieves superior FID performance while significantly accelerating convergence across all evaluated model scales. For smaller models trained for 400K iterations, SFD reduces FID from 21.45 to 10.40 for LightningDiTB/1 + REPA, and from 7.48 to 3.89 for LightningDiT-L/1 + REPA. In the large-scale setting, LightningDiT-XL/1 with SFD achieves FID 3.53 at only 400K iterations, outperforming LightningDiT-XL/1 with REPA at 4M iterations by 2.31 points (from 5.84 to 3.53) and vanilla DiT-XL/2 at 7M iterations by 6.09 points (from 9.62 to 3.53), with only 10% and 5.7% of the training cost, respectively. Notably, SFD achieves comparable performance to DiT-XL trained for 7M iterations and LightningDiT-XL/1 trained for 4M iterations in just 70K and 120K iterations, achieving 100× and 33.3× faster convergence (see Figure 1(b)).

Comparison with SOTA methods. Table 2 presents a system-level comparison with recent state-of-the-art methods with guidance. Our proposed SFD achieves both significantly faster convergence and superior generation performance. Remarkably, SFD surpasses DiT-XL trained for 1400 epochs within only 80 epochs, demonstrating exceptional training efficiency. At this early stage (80 epochs), SFD already achieves impressive FID scores of 1.30 with LightningDiT-XL and 1.19 with the 1.0Bparameter LightningDiT-XXL, both surpassing many existing methods. With extended training to 800 epochs, SFD achieves new state-of-the-art results of FID 1.06 with LightningDiT-XL and FID 1.04 with LightningDiT-XXL on ImageNet 256×256. Complete comparisons with and without guidance are provided in Appendix B.

##### 4.3. Ablation Studies

All ablation experiments are conducted using the LightningDiT-XL model trained for 400K iterations on ImageNet 256×256. Models are optimized with the

Table 1. FID comparison on ImageNet 256×256 without guidance across various model sizes for DiT with REPA and SFD (ours).

Model #Params Iter. FID↓ DiT-B/2 130M 400K 43.47 LightningDiT-B/1 130M 400K 22.86 + REPA 130M 400K 21.45 + SFD (Ours) 130M 400K 10.40 DiT-L/2 458M 400K 23.33 LightningDiT-L/1 458M 400K 10.08

+ REPA 458M 400K 7.48 + SFD (Ours) 458M 400K 3.89

DiT-XL/2 675M 400K 19.47 DiT-XL/2 675M 7M 9.62 LightningDiT-XL/1 675M 400K 9.29

- LightningDiT-XL/1 675M 1M 7.48
- LightningDiT-XL/1 675M 2M 6.88 LightningDiT-XL/1 675M 4M 6.50

+ REPA 675M 400K 6.94

- + REPA 675M 1M 6.17
- + REPA 675M 2M 5.87

+ REPA 675M 4M 5.84 + SFD (Ours) 675M 70K 8.79 + SFD (Ours) 675M 120K 6.22 + SFD (Ours) 675M 400K 3.53

- + SFD (Ours) 675M 1M 2.82

- + SFD (Ours) 675M 2M 2.74

+ SFD (Ours) 675M 4M 2.54

###### Table 2. System-level comparison of class-conditional generation on ImageNet 256×256 with guidance. Performance metrics are annotated with ↑ (higher is better) and ↓ (lower is better).

Model Epochs #Params FID↓ sFID↓ IS↑ Pre.↑ Rec.↑ Autoregressive Models

VAR [43] 350 2.0B 1.80 - 365.4 0.83 0.57 MAR [26] 800 943M 1.55 - 303.7 0.81 0.62 xAR [35] 800 1.1B 1.24 - 301.6 0.83 0.64

Latent Diffusion Models

DiT-XL [32] 1400 675M 2.27 4.60 278.2 0.83 0.57 MaskDiT [52] 1600 675M 2.28 5.67 276.6 0.80 0.61 SiT-XL [28] 1400 675M 2.06 4.50 270.3 0.82 0.59 FasterDiT [48] 400 675M 2.03 4.63 264.0 0.81 0.60 MDT [9] 1300 675M 1.79 4.57 283.0 0.81 0.61 MDTv2 [10] 1080 675M 1.58 4.52 314.7 0.79 0.65 DDT [46] 400 675M 1.26 - 310.6 0.79 0.65

Leveraging Visual Representations

VA-VAE [49] 800 675M 1.35 4.15 295.3 0.79 0.65 REPA [50] 800 675M 1.42 4.70 305.7 0.80 0.65 REPA-E [24] 800 675M 1.12 4.09 302.9 0.79 0.66 ReDi [22] 800 675M 1.61 4.66 295.1 0.78 0.64 REG [47] 800 677M 1.36 4.25 299.4 0.77 0.66 RAE [51] (DiT-XL) 800 676M 1.41 - 309.4 0.80 0.63 RAE [51] (DiTDH-XL) 800 839M 1.13 - 262.6 0.78 0.67 SFD (XL) 80 675M 1.30 3.87 233.4 0.78 0.64 SFD (XL) 800 675M 1.06 3.89 267.0 0.78 0.67 SFD (XXL) 80 1.0B 1.19 4.00 240.4 0.78 0.65 SFD (XXL) 800 1.0B 1.04 3.75 264.2 0.78 0.66

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

Figure 6. Qualitative samples from our model trained at 256×256 resolution.

SemVAE substantially improves performance to FID 5.24, confirming the effectiveness of explicit semantic representations. Finally, integrating the semantic-first mechanism further reduces FID to 3.03, demonstrating the effectiveness of our proposed asynchronous denoising strategy.

AdamW optimizer using a learning rate of 2 × 10−4 and β2 = 0.95. FID-50K is reported as the evaluation metric.

Effect of different components. Table 3 summarizes the contribution of each component in SFD. Starting from the baseline with FID 8.17, adding REPA yields moderate improvement to FID 7.08. Introducing semantic latents from

Table 3. Ablation study on different components of SFD.

REPA SemVAE Semantic-First FID↓

✗ ✗ ✗ 8.17 ✓ ✗ ✗ 7.08 ✓ ✓ ✗ 5.24 ✓ ✓ ✓ 3.03

Table 4. Ablation on semantic latent compression methods.

###### Table 5. Semantic-First helps with ReDi [22].

Method FID↓ PCA 4.06 SemVAE 3.03

Semantic-First FID↓

✗ 5.33 ✓ 4.41

Ablation on semantic latent compression methods. Table 4 compares PCA dimensionality reduction employed in ReDi with our SemVAE as different compression methods. SemVAE achieves a significantly superior FID of 3.03 compared to PCA’s 4.06, demonstrating the necessity of preserving semantic information completeness.

Other Ablations. Due to space limitations, additional ablation studies on vision foundation model selection and scaling, SemVAE bottleneck dimension, semantic loss weight β, and REPA parameters are deferred to Appendix C.

##### 4.4. Generalization of Semantic-First Mechanism

To validate the generalization of our semantic-first mechanism to other methods, we conduct experiments on ReDi [22] and VA-VAE [49]. ReDi employs PCA-reduced DINOv2-B features as semantic latents and concatenates them with texture latents encoded by SD-VAE for simultaneous denoising. As shown in Table 5, incorporating our semantic-first mechanism into ReDi improves FID from 5.33 to 4.41, demonstrating the effectiveness of the semantic-first approach for methods that leverage semantictexture composition. Due to space constraints, results on VA-VAE are provided in Appendix B.

##### 4.5. Reconstruction Performance Analysis

While recent work has highlighted the dilemma between reconstruction and generation in VAE [49], our SFD framework achieves state-of-the-art generation performance, without sacrificing the reconstruction fidelity. Here we compare the reconstruction quality of latent space variants (VA-VAE [49] and RAE [51]) to SD-VAE, which is adopted as the texture VAE in our method. As shown in Table 6, SD-VAE achieves the best performance with the lowest rFID of 0.26 and LPIPS of 0.089, along with the highest PSNR of 28.59 and SSIM of 0.80. VA-VAE, which aligns

Table 6. Comparison of reconstruction performance.

Method rFID↓ PSNR↑ LPIPS↓ SSIM↑

VA-VAE 0.28 27.96 0.096 0.79 RAE 0.57 18.86 0.256 0.42 SD-VAE 0.26 28.59 0.089 0.80

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 7. Qualitative reconstruction comparison among different latent space variants. From left to right: Original image, VA-VAE, RAE and SD-VAE reconstructions. The top-left insets show zoomed-in regions focusing on text details (“40 g”). SDVAE achieves the best reconstruction fidelity.

its latent space with visual foundation model features, enhances semantic representation but slightly compromises pixel-level reconstruction fidelity. In contrast, RAE builds its latent space purely on pretrained visual encoders, leading to texture-deficient representations and significantly degraded reconstruction quality: rFID 0.57, PSNR 18.86, LPIPS 0.256, and SSIM 0.42. As illustrated in Figure 7, RAE severely loses fine-grained details such as the “40g” text region and exhibits noticeable color distortion, while VA-VAE shows slightly reduced fidelity and SD-VAE faithfully maintains superior reconstruction quality. By adopting SD-VAE for texture modeling while introducing a separate semantic pathway, our composite latent design enables SFD to achieve significant convergence acceleration and superior generation performance without sacrificing reconstruction fidelity. This preservation of reconstruction quality makes SFD inherently more suitable for complex image synthesis tasks, such as text-to-image generation and consistencydemanding image editing.

#### 5. Conclusion

We propose Semantic-First Diffusion (SFD), a novel paradigm that performs asynchronous denoising on semantic and texture latents in latent diffusion models. By prioritizing semantic denoising to guide texture refinement, SFD achieves faster convergence and superior generation quality. Extensive experiments on ImageNet class-conditional generation demonstrate that SFD consistently outperforms competing methods. Our findings suggest that controlling the relative denoising pace between semantics and textures is crucial for efficient generative modeling, establishing representation-level asynchronous denoising as a promising direction for future diffusion research.

#### Acknowledgments

Yueming Pan and Nanning Zheng were supported in part by the NSFC under Grant No. 62088102.

#### References

- [1] Michael S Albergo, Nicholas M Boffi, and Eric VandenEijnden. Stochastic interpolants: A unifying framework for flows and diffusions, 2023. URL https://arxiv. org/abs/2303.08797, 3, 2023. 3
- [2] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024. 3
- [3] Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, and Ping Luo. Pixelflow: Pixel-space generative models with flow. arXiv preprint arXiv:2504.07963, 2025. 3
- [4] Timoth´ee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. arXiv preprint arXiv:2309.16588, 2023. 5, 1
- [5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 6, 1
- [6] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 6, 1, 3
- [7] John R Dormand and Peter J Prince. A family of embedded runge-kutta formulae. Journal of computational and applied mathematics, 6(1):19–26, 1980. 6, 1
- [8] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 3

- [9] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image synthesizer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 23164–23173,

2023. 7, 3

- [10] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Mdtv2: Masked diffusion transformer is a strong image synthesizer. arXiv preprint arXiv:2303.14389,

2023. 7, 3

- [11] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 5
- [12] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6

- [13] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 1
- [14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [15] Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans. Simpler diffusion (sid2): 1.5 fid on imagenet512 with pixel-space diffusion. arXiv preprint arXiv:2410.19324, 2024. 3
- [16] Zijing Hu, Yunze Tong, Fengda Zhang, Junkun Yuan, Jun Xiao, and Kun Kuang. Asynchronous denoising diffusion models for aligning text-to-image generation. arXiv preprint arXiv:2510.04504, 2025. 3
- [17] Allan Jabri, David Fleet, and Ting Chen. Scalable adaptive computation for iterative generation. arXiv preprint arXiv:2212.11972, 2022. 3
- [18] Tero Karras, Miika Aittala, Tuomas Kynk¨a¨anniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. Advances in Neural Information Processing Systems, 37:52996–53021, 2024. 6, 1, 4
- [19] Tero Karras, Miika Aittala, Tuomas Kynk¨a¨anniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. Advances in Neural Information Processing Systems, 37:52996–53021, 2024. 4
- [20] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 4
- [21] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 1
- [22] Theodoros Kouzelis, Efstathios Karypidis, Ioannis Kakogeorgiou, Spyros Gidaris, and Nikos Komodakis. Boosting generative image modeling via joint image-feature synthesis. arXiv preprint arXiv:2504.16064, 2025. 2, 3, 6, 7, 8, 4
- [23] Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019. 6
- [24] Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483, 2025. 2, 3, 7, 4
- [25] Tianhong Li, Dina Katabi, and Kaiming He. Return of unconditional generation: A self-supervised representation generation method. Advances in Neural Information Processing Systems, 37:125441–125468, 2024. 4
- [26] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024. 7, 2, 3
- [27] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3
- [28] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024. 1, 2, 3, 7, 4

- [29] Charlie Nash, Jacob Menick, Sander Dieleman, and Peter W Battaglia. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021. 6
- [30] Mang Ning, Mingxiao Li, Jianlin Su, Haozhe Jia, Lanmiao Liu, Martin Beneˇs, Wenshuo Chen, Albert Ali Salah, and Itir Onal Ertugrul. Dctdiff: Intriguing properties of image generative modeling in the dct space. arXiv preprint arXiv:2412.15032, 2024. 2
- [31] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 3, 5, 1
- [32] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 1, 2, 6, 7, 3, 4

- [33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 5
- [34] Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence level training with recurrent neural networks. ICLR, 2016. 2
- [35] Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, and Liang-Chieh Chen. Beyond next-token: Next-x prediction for autoregressive visual generation. arXiv preprint arXiv:2502.20388, 2025. 7, 2, 3
- [36] Severi Rissanen, Markus Heinonen, and Arno Solin. Generative modelling with inverse heat dissipation. In International Conference on Learning Representations, 2023. 2
- [37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2, 4, 5
- [38] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015. 2
- [39] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 6
- [40] Minglei Shi, Haolin Wang, Wenzhao Zheng, Ziyang Yuan, Xiaoshi Wu, Xintao Wang, Pengfei Wan, Jie Zhou, and Jiwen Lu. Latent diffusion model without variational autoencoder. arXiv preprint arXiv:2510.15301, 2025. 3, 4
- [41] Ivan Skorokhodov, Sharath Girish, Benran Hu, Willi Menapace, Yanyu Li, Rameen Abdal, Sergey Tulyakov, and Aliaksandr Siarohin. Improving the diffusability of autoencoders. In Forty-second International Conference on Machine Learning, 2025. 2

- [42] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2
- [43] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024. 7, 2, 3
- [44] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 5
- [45] Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. Pixnerd: Pixel neural field diffusion. arXiv preprint arXiv:2507.23268, 2025. 3
- [46] Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. Ddt: Decoupled diffusion transformer. arXiv preprint arXiv:2504.05741, 2025. 2, 7, 3, 4
- [47] Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Jian Yang, et al. Representation entanglement for generation: Training diffusion transformers is much easier than you think. arXiv preprint arXiv:2507.01467, 2025. 2, 3, 6, 7, 4, 5
- [48] Jingfeng Yao, Cheng Wang, Wenyu Liu, and Xinggang Wang. Fasterdit: Towards faster diffusion transformers training without architecture modification. Advances in Neural Information Processing Systems, 37:56166–56189, 2024. 7, 3
- [49] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025. 2, 3, 5, 6, 7, 8, 1, 4
- [50] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In The Thirteenth International Conference on Learning Representations, 2025. 2, 3, 5, 6, 7, 4
- [51] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025. 3, 7, 8, 2, 4, 5
- [52] Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305, 2023. 7, 3

## Semantics Lead the Way: Harmonizing Semantic and Texture Modeling with Asynchronous Latent Diffusion

### Supplementary Material

#### A. Additional Implementation Details

##### A.1. SemVAE Configuration

For semantic representation extraction, we employ DINOv2-B with registers [4, 31] on 256 × 256 images. The SemVAE architecture consists of 4 transformer blocks for both the encoder and decoder, with 29M parameters each (58M total). We train on ImageNet-1K [5] for 1M iterations with random cropping as data augmentation. Detailed hyperparameters are shown in Table 7.

##### A.2. Diffusion Model Configuration

We adopt LightningDiT [49] as our diffusion backbone, which is available in multiple scales (B/L/XL/XXL). For latent construction, SD-VAE [37, 49] (implemented in LightningDiT2 [49]) encodes textures into 32 channels with 16× spatial compression, while SemVAE extracts 16-channel semantic representations. Their concatenation forms a unified 48-channel, 256-token latent for each 256×256 image.

Following the ADM [6] preprocessing pipeline, all images are center cropped and resized to 256×256 resolution. Training is conducted on ImageNet-1K [5] for 800 epochs with a batch size of 256, using AdamW optimizer with a learning rate of 1 × 10−4 and β values of (0.9, 0.999). We employ logit-normal timestep sampling following LightningDiT [49]. For sampling, the dopri5 solver [7] with adaptive step size is employed, with absolute and relative tolerances set to 10−6 and 10−3 respectively. Detailed hyperparameters across different model scales are shown in Table 8.

##### A.3. Dual Timestep Embedding

To support asynchronous denoising, SFD employs two independent timestep embedders corresponding to the semantic and texture timesteps ts and tz. Unlike LightningDiT [49], which uses a single MLP-based embedder of hidden dimension H, SFD constructs two smaller embedders whose hidden dimensions are reduced to H/2. Each embedder independently processes its respective timestep, the two embeddings are then concatenated along the channel dimension and injected into the backbone. This design allows the model to supply distinct timestep signals to the semantic and texture latents:

###### e = [τs(ts), τz(tz)], (21)

2https://github.com/hustvl/LightningDiT

###### Table 7. SemVAE training configuration.

Component Setting Feature Extraction Feature Extractor DINOv2-B-reg Input Patch Size 256×256 Architecture Total Parameters 58M Encoder Parameters 29M Decoder Parameters 29M Encoder Blocks 4 Decoder Blocks 4 Hidden Dimension 768 Attention Heads 6 Bottleneck Channels 16 KL Weight λkl 10−7 Training Dataset ImageNet-1K Total Iterations 1,000,000 Batch Size 64 Data Augmentation Random cropping Optimization Optimizer AdamW Learning Rate 5 × 10−5 (β1, β2) (0.9, 0.999) LR Schedule Warmup Steps 500 Constant Steps 800,000 Annealing Cosine to 5 × 10−6

where [·,·] denotes channel-wise concatenation, and τs(·) and τz(·) are the semantic and texture timestep embedders.

##### A.4. Evaluation Details

AutoGuidance. We employ AutoGuidance [18] as our primary guidance method. Unlike Classifier-Free Guidance (CFG) [13], which relies on an unconditional model, AutoGuidance guides the main diffusion model using a weaker version of itself—typically a model with smaller capacity or an earlier training snapshot. This self-guidance mechanism effectively suppresses out-of-manifold samples by aligning the denoising trajectory toward regions of higher data density, thereby improving image quality without sacrificing sample diversity. In practice, we use the degraded LightningDiT-B model as the guiding network. After searching, configurations of degraded models are illustrated in Tab. 9.

###### Table 8. Hyperparameter settings across different model scales.

Backbone LightningDiT-B LightningDiT-L LightningDiT-XL LightningDiT-XXL Architecture #Params 130M 458M 675M 1.0B Input 16 × 16 × 48 16 × 16 × 48 16 × 16 × 48 16 × 16 × 48 Layers 12 24 28 32 Hidden dim. 768 1024 1152 1280 Num. heads 12 16 16 16 SFD settings β 2.0 2.0 2.0 2.0 ∆t 0.3 0.3 0.3 0.3 REPA visual encoder DINOv2-B-reg DINOv2-B-reg DINOv2-B-reg DINOv2-B-reg REPA weight λ 1.0 1.0 1.0 1.0 REPA alignment depth 2 2 2 2 REPA similarity function cosine cosine cosine cosine Optimization Batch size 256 256 256 256 Optimizer AdamW AdamW AdamW AdamW lr 1 × 10−4 1 × 10−4 1 × 10−4 1 × 10−4 (β1, β2) (0.9, 0.999) (0.9, 0.999) (0.9, 0.999) (0.9, 0.999) Sampling Sampler dopri5 dopri5 dopri5 dopri5 Absolute tolerance 10−6 10−6 10−6 10−6 Relative tolerance 10−3 10−3 10−3 10−3

###### Table 9. Configurations of degraded models used for guidance.

Model Epochs Params Degraded Model Iterations Guidance Scale

LightningDiT-XL 80 675M LightningDiT-B 70K 1.6 LightningDiT-XL 800 675M LightningDiT-B 70K 1.5 LightningDiT-XXL 80 1.0B LightningDiT-B 60K 1.5 LightningDiT-XXL 800 1.0B LightningDiT-B 120K 1.5

Class-balanced Sampling. RAE [51] shows that classbalanced sampling yields more reliable and lower FID estimates. To ensure fair comparison with prior work [26, 35, 43, 46, 51], we follow this protocol and adopt classbalanced sampling for FID-50K evaluation. Specifically, we generate 50 images per class (50,000 in total).

#### B. Additional Experimental Results B.1. Complete Comparisons

- Table 10 presents a system-level comparison of classconditional generation on ImageNet 256 × 256. In the guidance setting, our SFD achieves state-of-the-art performance, surpassing existing methods in both FID and sFID. Notably, our SFD-XL (675M) outperforms the previous best model, RAE DiTDH (839M), with a lower FID (1.06 vs. 1.13), demonstrating superior generation quality with fewer parameters. Scaling up to SFD-XXL (1.0B) further pushes the performance boundary to a FID of 1.04. Notably, SFD achieves a superior sFID of 3.75, outperform-

ing previous methods by a substantial margin. Since sFID serves as a metric for structural coherence and spatial alignment, this improvement validates the advantage of our explicit compression of semantic representations with spatial layouts, which ensures robust global structure before texture refinement.

Regarding the unguided setting, SFD remains competitive but exhibits limitations in texture convergence. This is primarily attributed to the high complexity of the texture latents. Unlike methods such as ReDi [22] or REG [47] that utilize a standard f8d4 VAE, we employ the f16d32 variant (following LightningDiT), which results in a latent space with double the dimensionality. Consequently, modeling these high-dimensional texture latents is inherently more challenging and harder to converge.

##### B.2. Inference Strategies

Inference Steps. Table 11 illustrates FID scores without guidance of various sampling steps, showing that SFD

Table 10. System-level comparison of class-conditional generation on ImageNet 256×256.

Generation@256 w/o guidance Generation@256 w/ guidance

Method Epochs #Params

FID↓ sFID↓ IS↑ Prec.↑ Rec.↑ FID↓ sFID↓ IS↑ Prec.↑ Rec.↑ Autoregressive

VAR [43] 350 2.0B - - - - - 1.80 - 365.4 0.83 0.57 MAR [26] 800 943M 2.35 - 227.8 0.79 0.62 1.55 - 303.7 0.81 0.62 xAR [35] 800 1.1B - - - - - 1.24 - 301.6 0.83 0.64

###### Pixel Diffusion

ADM [6] 400 554M 10.94 6.02 101.0 0.69 0.63 3.94 6.14 215.8 0.83 0.53 RIN [17] 480 410M 3.42 - 182.0 - - - - - - PixelFlow [3] 320 677M - - - - - 1.98 5.83 282.1 0.81 0.60 PixNerd [45] 160 700M - - - - - 2.15 4.55 297.0 0.79 0.59 SiD2 [15] 1280 - - - - - - 1.38 - - - -

###### Latent Diffusion

DiT [32] 1400 675M 9.62 6.85 121.5 0.67 0.67 2.27 4.60 278.2 0.83 0.57 MaskDiT [52] 1600 675M 5.69 10.34 177.9 0.74 0.60 2.28 5.67 276.6 0.80 0.61 SiT [28] 1400 675M 8.61 6.32 131.7 0.68 0.67 2.06 4.50 270.3 0.82 0.59 FasterDiT [48] 400 675M 7.91 5.45 131.3 0.67 0.69 2.03 4.63 264.0 0.81 0.60 MDT [9] 1300 675M 6.23 5.23 143.0 0.71 0.65 1.79 4.57 283.0 0.81 0.61 MDTv2 [10] 1080 675M - - - - 1.58 4.52 314.7 0.79 0.65 DDT [46] 400 675M 6.27 - 154.7 0.68 0.69 1.26 - 310.6 0.79 0.65

###### Leveraging Visual Representations

VA-VAE [49] 800 675M 2.17 4.36 205.6 0.77 0.65 1.35 4.15 295.3 0.79 0.65 REPA [50] 800 675M 5.90 - - - - 1.42 4.70 305.7 0.80 0.65 REPA-E [24] 800 675M 1.69 4.17 219.3 0.77 0.67 1.12 4.09 302.9 0.79 0.66 ReDi [22] 800 675M 3.30 4.80 188.9 0.74 0.68 1.61 4.66 295.1 0.78 0.64 REG [47] 800 677M 1.80 4.59 230.8 0.77 0.66 1.36 4.25 299.4 0.77 0.66 RAE [51] (DiT-XL) 800 676M 1.87 - 209.7 0.80 0.63 1.41 - 309.4 0.80 0.63 RAE [51] (DiTDH-XL) 800 839M 1.51 - 242.9 0.79 0.63 1.13 - 262.6 0.78 0.67 SFD (XL) 80 675M 3.43 4.34 162.0 0.75 0.65 1.30 3.87 233.4 0.78 0.64 SFD (XL) 800 675M 2.54 4.38 191.7 0.75 0.67 1.06 3.89 267.0 0.78 0.67 SFD (XXL) 80 1.0B 2.84 4.25 172.6 0.75 0.65 1.19 4.00 240.4 0.78 0.65 SFD (XXL) 800 1.0B 2.38 4.37 197.9 0.75 0.67 1.04 3.75 264.2 0.78 0.66

- Table 11. FID (↓) comparison across inference steps. All models are trained for 400K iterations and evaluated using the Euler sampler without guidance. Reported values are FID-10K scores computed at different inference step counts.

Inference steps 250 200 150 100 80 60

Method

LightningDiT 12.50 12.58 12.67 12.91 13.03 13.40 LightningDiT+REPA 10.00 10.10 10.23 10.50 10.67 10.94 LightningDiT+VA-VAE 7.66 7.66 7.68 7.70 7.76 7.83 LightningDiT+ReDi 8.58 8.63 8.72 8.86 9.02 9.32 LightningDiT+SFD (Ours) 6.32 6.26 6.41 6.35 6.81 6.77

maintains strong performance even with significantly fewer inference steps. While other baselines require 200–250 steps to approach their optimal FID, SFD already achieves a competitive score of 6.35 at only 100 steps, and further increasing the steps to 250 yields only marginal improvement

(6.32). This observation suggests that the semantic-first design may facilitate more efficient sampling: by stabilizing global semantics early, the model requires fewer refinement steps to reach high-quality solutions.

Table 12 presents the FID scores with guidance across

- Table 12. FID (↓) comparison across inference steps for SFD (XL) and SFD (XXL) models at 4M training iterations with guidance. Reported values are FID-50K scores.

Method dopri5 250 200 150 100 80 60 50 40 30 25

SVG [40] - - - - - - - - - - 1.920 SFD (XL) 1.064 1.051 1.050 1.048 1.045 1.086 1.102 1.206 1.510 1.447 1.865 SFD (XXL) 1.040 1.035 1.040 1.041 1.058 1.080 1.106 1.190 1.429 1.456 1.844

Table 13. Comparison of class-random sampling and class-balanced sampling.

Random sampling Balanced sampling FID↓ sFID↓ IS↑ Prec.↑ Rec.↑ FID↓ sFID↓ IS↑ Prec.↑ Rec.↑

Method

SiT [28] 2.06 4.50 270.3 0.82 0.59 1.95 - 259.5 - REPA [50] 1.42 4.70 305.7 0.80 0.65 1.29 - 306.3 0.79 0.64 REPA-E [24] 1.26 4.11 314.9 0.79 0.66 1.12 4.09 302.9 0.79 0.66 DDT [46] 1.40 - 303.6 - - 1.26 - 310.6 0.79 0.65 VA-VAE [49] 1.35 4.15 295.3 0.79 0.65 1.23 4.20 296.0 0.79 0.65 ReDi [22] 1.61 4.66 295.1 0.78 0.64 1.60 5.99 294.7 0.78 0.64 REG [47] 1.36 4.25 299.4 0.77 0.66 1.19 4.44 305.4 0.78 0.66 RAE [51] (DiTDH-XL) 1.28 - 262.9 - - 1.13 - 262.6 0.78 0.67 SFD (XL) 1.18 3.89 266.8 0.78 0.67 1.06 3.89 267.0 0.78 0.67

varying sampling steps. Notably, SFD (XL) achieves a superior FID of 1.045 at only 100 steps using the Euler sampler, surpassing the result yielded by the dopri5 sampler (1.064). Furthermore, in the few-step regime (25 steps), SFD (XL) maintains its advantage over SVG [40], recording an FID of 1.865 compared to 1.920 by SVG.

Class-balanced Sampling. To ensure a rigorous comparison, we re-evaluate prior state-of-the-art methods employing the same class-balanced protocol as discussed in RAE [51]. Specifically, results for SiT [28], REPA [50], and DDT [46] are adopted from RAE [51], while REPA-E [24] figures are sourced from its original publication. Additionally, we conduct independent evaluations for VA-VAE [49], ReDi [22], and REG [47]. The quantitative comparison results are presented in Table 13. As observed, our proposed SFD (XL) demonstrates consistent superiority across both protocols. Remarkably, whether using class-balanced or class-random sampling, SFD achieves the best performance in terms of FID and sFID metrics, surpassing all competing state-of-the-art methods.

##### B.3. Unconditional Generation

We further evaluate the proposed SFD on unconditional image generation on ImageNet 256×256. During both training and sampling, we set the class label to 1000 (the null label). As shown in Table 14, SFD demonstrates remarkable performance with high training efficiency. Even without AutoGuidance (AG) [18], SFD significantly surpasses ReDi (FID 25.10 → 10.24) after only 80 epochs and further improves to an FID of 8.46 after 200 epochs. With AG

Table 14. Comparison of unconditional generation on ImageNet 256×256. RG and AG are short of Representation Guidance [22] and AutoGuidance [19].

Method Epochs Params FID↓ IS↑ DiT-XL [32] 400 675M 30.68 32.7 ReDi [22] 80 675M 25.10 – ReDi [22] (w/ RG) 80 675M 22.60 – RAE [51] (w/ AG) 200 839M 4.96 123.1 RCG [25] (DiT-XL/2) 400 675M 4.89 143.2 RCG [25] (MAGE-L) 800 502M 3.44 186.9 RCG-G [25] (MAGE-L) 800 502M 2.15 253.4 SFD (w/o AG) 80 675M 10.24 78.5 SFD (w/ AG) 80 675M 3.77 127.9 SFD (w/o AG) 200 675M 8.46 89.9 SFD (w/ AG) 200 675M 2.90 148.5

enabled, SFD achieves substantial gains, reaching FIDs of 3.77 and 2.90 at 80 and 200 epochs, respectively. We attribute these improvements to the asynchronous denoising mechanism of SFD, which becomes especially crucial in the unconditional setting. These results suggest that, without class labels as conditional guidance, smoother semantic representations are more easily modeled, thus providing accurate global structural cues for superior generation performance.

##### B.4. SFD for VA-VAE

Tab. 15 analyzes the impact of applying Semantic-First Diffusion (SFD) to VA-VAE [49]. For both VA-VAE and ReDi settings, the SFD implementations used for comparison are

###### Table 15. Effect of SFD for VA-VAE.

TexEnc SFD FID↓ VA-VAE ✗ 4.52 VA-VAE ✓ 4.14 SD-VAE (ours) ✓ 3.03

- Table 16. Computational cost and performance comparison between LightningDiT and LightningDiT+SFD at 400K iterations on ImageNet 256×256. SFD adds negligible computational overhead while delivering substantially improved generation quality.

Method #Params (M)↓ GFLOPs↓ FID↓ LightningDiT-XL 683.39 116.479 9.29 LightningDiT-XL + SFD 682.77 116.487 3.53

#### C. Additional Ablation Studies

- C.1. Semantic VAE Design

Our Semantic VAE (SemVAE) compresses pretrained vision foundation model features into compact semantic representations. To investigate its design choices, we conduct a series of ablation studies on three key aspects: the choice of pretrained vision encoder, model scaling within the encoder family, and the number of output channels representing semantic capacity.

Different target representation and model scaling.

- Tab. 17 (a) compares several pretrained vision encoders used as target representations. Among all candidates, DINOv2-B achieves the lowest FID of 3.03, outperforming MAE [11], CLIP [33], and SigLip [44], indicating that DINOv2 provides the most effective supervision for compact semantic latent learning. Tab. 17 (b) studies different model scales within the DINOv2 family. Larger encoders yield better semantic guidance, with DINOv2-L achieving the best FID of 2.97. Notably, this finding stands in contrast to recent works like REG [47] and RAE [51], which identified DINOv2-B as the optimal choice and observed performance degradation when scaling to larger VFMs due to their increased dimensionality. Our results demonstrate the superiority of our explicit semantic compression strategy, which effectively handles high-dimensional features and unlocks the potential for further scaling with more powerful VFMs. Considering the trade-off between performance and efficiency, we adopt DINOv2-B as the default pretrained visual encoder.

Channel capacity. DINOv2-B outputs 768-dimensional features, which are compressed by the Semantic VAE into a lower-dimensional semantic latent. Tab. 17 (c) investigates the impact of varying the latent channel capacity. We observe a consistent performance improvement as the number of channels increases from 2 to 16. This trend indicates that a higher channel capacity is essential for preserving the rich semantic information embedded in the original high-dimensional features. The 16-channel configuration achieves the best FID of 3.03, confirming that retaining more semantic details directly contributes to superior generation quality.

C.2. Effect of semantic loss weight

- Tab. 18 analyzes the impact of the semantic loss weight β in the velocity prediction objective. As the weight increases from 0.25 to 2.0, the FID score consistently decreases, indicating that stronger semantic supervision enhances training stability and generation performance. However, when β becomes excessively large (e.g., 4.0 or 8.0), the performance

our reproduced versions. When equipped with SFD, VAVAE improves performance from an FID of 4.52 to 4.14, indicating that SFD is also compatible with joint semantictexture latent space. However, its overall performance still lags behind our SD-VAE-based SFD (FID 3.03). This is likely because VA-VAE’s latent space inherently entangles semantics and textures, leaving limited flexibility for the asynchronous denoising mechanism to operate effectively. In contrast, disentangling semantic and texture representations (as done in SD-VAE) allows the semantic latents to stabilize early and provide clearer global guidance for texture refinement, ultimately yielding higher generative quality.

##### B.5. Computational Cost

We evaluate the computational overhead introduced by integrating SFD into LightningDiT-XL. SFD modifies the backbone in two ways. First, it augments the latent representation with a 16-channel semantic latent, which introduces a marginal increase in backbone FLOPs. Second, SFD replaces the single timestep embedder in LightningDiT with two independent embedders that operate on the semantic and texture timesteps ts and tz. As shown in equation 21, although two embedders are used, the total parameter count is actually smaller, since MLP parameters grow quadratically with hidden dimension. Consequently, two (H/2)width MLPs contain only 0.5× the parameters and FLOPs of a single H-width MLP.

Table 16 reports the computational cost comparison. SFD incurs only a negligible increase in FLOPs (less than 0.01%) while delivering a dramatic improvement in FID at 400K iterations. This indicates that SFD achieves an extremely favorable cost–performance tradeoff with virtually no additional computational burden.

- Table 17. Ablation on Semantic VAE design. (a) compares different target representation models; (b) studies model scaling within DINOv2 family; (c) analyzes semantic channel capacity.

Target Repr. FID↓ DINOv2-B 3.03 MAE-B 6.29 CLIP-B 4.89 SigLip-B 4.15

Target Repr. FID↓ DINOv2-S 4.14 DINOv2-B 3.03 DINOv2-L 2.97

#Channels FID↓

2 3.90 4 3.67 8 3.16

###### 16 3.03

(a) Model comparison.

(b) Scaling comparison.

(c) Channel capacity.

###### Table 18. Effect of semantic loss weight.

Weight β 0.25 0.5 1.0 2.0 4.0 8.0 FID↓ 3.46 3.26 3.08 3.03 3.28 3.96

Table 19. Ablation on REPA configurations. Depth of conducting REPA loss, loss weight λ, and loss type are included.

Depth Weight λ Type FID↓ – – – 4.15 2 0.5 cosine+MSE 3.03 4 0.5 cosine+MSE 3.07 6 0.5 cosine+MSE 3.24 8 0.5 cosine+MSE 3.16

10 0.5 cosine+MSE 3.19 12 0.5 cosine+MSE 3.28 2 0.25 cosine+MSE 3.30

- 2 0.5 cosine+MSE 3.03

- 2 1.0 cosine+MSE 3.18

- 2 2.0 cosine+MSE 3.25 2 4.0 cosine+MSE 3.20 2 0.5 cosine 3.16 2 0.5 MSE 3.13 2 0.5 cosine+MSE 3.03

degrades, suggesting that overemphasizing semantics suppresses texture learning and leads to loss of fine details. Overall, β = 2.0 achieves the best balance between semantic guidance and texture refinement, yielding the lowest FID of 3.03.

##### C.3. Effect of REPA configurations

Alignment depth. Tab. 19 presents a systematic study of REPA configurations. In our experiments, applying the REPA loss at shallow layers (specifically at depth 2) yields the best performance with an FID of 3.03, whereas the original REPA [50] reports optimal performance at depth 8. We attribute this discrepancy to the distinct role of the alignment loss in our framework. While the original REPA operates as a distillation process that forces the diffusion model to gradually analyze and understand the input la-

tents, our approach utilizes the REPA loss to drive the model to decode and reconstruct high-level semantic representations from the noisy compressed latents. Since decoding from a semantic latent is inherently a more straightforward task than analyzing semantics from scratch, our model can achieve effective alignment at much shallower layers. Consequently, early-layer alignment suffices to recover the semantic guidance, avoiding the need for deeper intervention.

REPA loss weight λ. For the REPA loss weight λ, the model achieves the lowest FID at λ = 0.5. This indicates that a moderate alignment strength provides a good balance between semantic consistency and generative fidelity.

REPA similarity function. We also compare results of different REPA similarity functions. While conventional REPA employs cosine similarity for feature alignment, we additionally explore combining cosine and MSE losses inspired by our SemVAE training. The combined objective (cosine+MSE) achieves the best performance of 3.03 FID score, outperforming single-loss variants. This suggests that employing a similarity function consistent with the SemVAE training metric yields optimal results. Furthermore, it demonstrates the complementary nature of the two terms: MSE ensures distribution-level precision, whereas cosine similarity enhances directional alignment, leading to better semantic matching and visual realism.

It is worth noting that the optimal settings identified in this ablation study differ slightly from the final hyperparameters presented in Table 8. This discrepancy arises because the ablation experiments were evaluated at 400K iterations; however, over the full training duration (4M iterations), the configuration detailed in Table 8 yielded superior performance. Consequently, our final model adopts the settings from Table 8 rather than strictly following the ablation outcomes.

#### D. Limitation and Future Work

Currently, SFD employs a fixed temporal offset ∆t to manage the asynchronous denoising process. However, a static

Training Iteration

160K 320K 480K 160K 320K 480K 160K 320K 480K

[Figure 22]

[Figure 23]

[Figure 24]

LightningDiTLightningDiT

+REPA LightningDiT

+VA-VAE LightningDiT

+SFD

Figure 8. Visualization of training results across different iterations (160K, 320K, and 480K). Under a fixed random seed and identical initial noise, SFD produces clearer structures and more realistic details at early stages, demonstrating faster convergence compared with other variants.

offset may not be optimal across all noisy levels. Future work could explore dynamic or adaptive schedules for ∆t to further enhance the synergy between semantic and texture generation. Furthermore, our framework presently relies on the REPA loss as an auxiliary objective to enforce feature alignment. A promising direction for future research is to investigate methods that eliminate the need for such auxiliary supervision, aiming for a cleaner and more streamlined optimization structure.

Beyond algorithmic refinements, extending and scaling SFD to more complex application scenarios represents a highly valuable research direction. Specifically, adapting SFD to text-to-image and text-to-video generation tasks could further validate its potential in handling intricate multimodal guidance and temporal consistency.

#### E. More Visualization Results

We qualitatively compare the training progression in Figure 8, where all models are evaluated using the same initial noise. The baseline LightningDiT, REPA, and VA-VAE variants exhibit weaker structural consistency and struggle to form coherent details in the early training stages. In contrast, SFD produces clearer structures and more realistic details at much earlier iterations, demonstrating noticeably faster convergence.

We also present more visualization results of SFD in Figures 9 - 17.

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

Figure 9. Visualization results of LightningDiT-XL + SFD for the ImageNet class “Bald eagle” (22).

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

Figure 10. Visualization results of LightningDiT-XL + SFD for the ImageNet class “Sulphur-crested cockatoo” (89).

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

- Figure 11. Visualization results of LightningDiT-XL + SFD for the ImageNet class “Giant panda” (388).

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

###### Figure 12. Visualization results of LightningDiT-XL + SFD for the ImageNet class “Teapot” (848).

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

###### Figure 13. Visualization results of LightningDiT-XL + SFD for the ImageNet class “Hamburger” (933).

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

###### Figure 14. Visualization results of LightningDiT-XL + SFD for the ImageNet class “Strawberry” (949).

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

- Figure 15. Visualization results of LightningDiT-XL + SFD for the ImageNet class “Castle” (483).

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

[Figure 120]

- Figure 16. Visualization results of LightningDiT-XL + SFD for the ImageNet class “Lakeside” (975).

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

Figure 17. Visualization results of LightningDiT-XL + SFD for the ImageNet class “Hot-air balloon” (417).

