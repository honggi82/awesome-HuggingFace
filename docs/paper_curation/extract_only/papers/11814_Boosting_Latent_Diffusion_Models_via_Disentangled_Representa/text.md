# arXiv:2601.05823v2[cs.CV]16Mar2026

## Boosting Latent Diffusion Models via Semantic-Disentangled VAE

##### John Page∗, Xuesong Niu∗, Kai Wu‡, and Kun Gai

Kolors Team, Kuaishou Technology

Abstract. Latent Diffusion Models (LDMs) rely heavily on the compressed latent space provided by Variational Autoencoders (VAEs) for high-quality image generation. Recent studies have attempted to obtain generation-friendly VAEs by directly adopting alignment strategies from LDM training, leveraging Vision Foundation Models (VFMs) as representation alignment targets. However, such alignment paradigms overlook the fundamental differences in representational requirements between LDMs and VAEs. Simple feature mapping from local patches to highdimensional semantics can induce semantic collapse, leading to the loss of fine-grained attributes. In this paper, we reveal a key insight: unlike LDMs that benefit from high-level global semantics, a generation-friendly VAE must possess strong semantic disentanglement capabilities to preserve fine-grained, attribute-level information in a structured manner. To address this discrepancy, we propose the Semantic-Disentangled VAE (Send-VAE). Deviating from previous shallow alignment approaches, SendVAE introduces a non-linear mapping architecture to effectively bridge the local structures of VAEs and the dense semantics of VFMs, thereby encouraging emergent disentangled properties in the latent space without explicit regularization. Extensive experiments establish a new paradigm for evaluating VAE latent spaces via low-level attribute separability and demonstrate that Send-VAE achieves state-of-the-art generation quality (FID of 1.21) on ImageNet 256 × 256. Code is available at https://github.com/Kwai-Kolors/Send-VAE.

Keywords: Image tokenizer · Latent diffusion model · Image synthesis

### 1 Introduction

Latent diffusion models (LDMs) [2,27,31,33] have recently achieved remarkable success in high-resolution image synthesis, establishing new benchmarks in visual fidelity and detail. A critical component of these models is the image tokenizer, typically implemented via a variational autoencoder (VAE) [17]. By compressing images into a structured latent space, VAEs significantly reduce the computational demands associated with high-resolution generation. Consequently, the

- 1 * These authors contributed equally to this work.
- 2 ‡Corresponding author

[Figure 1]

###### Previous Methods (Direct Alignment)

VAE Latent Space

[Figure 2]

[Figure 3]

|VFM<br><br>High-level<br><br>Semantics|
|---|

Texture

Color Shape

Shallow MLP mapper (Loss of fine-grained attributes)

VAE Latent Space

Non-linear Mapper (ViT+MLP)

[Figure 4]

|VFM High-level<br><br>Semantics|
|---|

[Figure 5]

Texture Color Shape

Alleviates information bottleneck and preserves structured semantics

Semantic-disentangled VAE

- Fig. 1: (Left) We reveal a strong correlation (Pearson corr. -0.9572) between the linear separability of low-level attributes (measured by Attribute Prediction Recall) in VAE latent spaces and the downstream generation quality (measured by FID). (Right) Comparison of alignment paradigms. Send-VAE introduces a non-linear mapper (ViT+MLP), which seamlessly absorbing dense VFM semantics while effectively preserving the VAE’s native structured and disentangled semantics.

quality of a VAE directly dictates both the training efficiency and the generative fidelity of downstream diffusion models. Despite its foundational importance, the defining characteristics of a generation-friendly VAE remain underexplored.

To bridge the gap between traditional pixel-level reconstruction and generative objectives, recent pioneering efforts [6, 42, 45] attempt to explicitly align VAE latents with the representations of large-scale, pre-trained Vision Foundation Models (VFMs) such as CLIP [32] or DINOv2 [30]. Motivated by the success of representation alignment in LDM training (e.g., REPA [44]), these approaches directly adopt similar alignment paradigms for VAEs. However, while yielding notable empirical gains, this direct inheritance harbors a critical conceptual flaw: it erroneously assumes that VAEs and LDMs share identical alignment targets, overlooking their fundamentally distinct representational demands. LDMs undoubtedly benefit from highly abstract, high-level semantics crucial for generative modeling. However, VAEs serve as the fundamental tokenizers, they must encode fine-grained, low-level visual elements (e.g., textures, colors, and local structures) in a structured way. We argue that forcing direct alignment with VFMs severely couples these low-level details, leading to feature entanglement that ultimately bottlenecks generation quality.

Drawing inspiration from recent tokenizer analyses [4], we hypothesize that the true catalyst for a generation-friendly VAE lies in its semantic disentanglement capabilities, which naturally enable the model to robustly encode attributelevel visual information. To validate this hypothesis, we conduct linear probing experiments on attribute prediction benchmarks to empirically quantify the disentanglement of various VAE latent spaces. As depicted in Fig. 1 (left), our analysis reveals a striking positive correlation between the linear separability of low-level attributes and the final generation quality of the downstream diffusion

model. This compelling evidence dictates that the richness and accessibility of structured, fine-grained semantics is a more fundamental prerequisite for VAE latents than high-level semantic alignment. Consequently, we advocate for linear probing on attribute prediction tasks as a novel, intrinsic metric for evaluating VAE latent quality.

Motivated by these insights, we introduce the Semantic-disentangled VAE (Send-VAE) to explicitly cultivate this property. Rather than enforcing the rigid, shallow alignment employed by prior works, Send-VAE leverages the rich representations of VFMs through a carefully designed non-linear mapping architecture. As illustrated in Fig. 1 (right), this mapper effectively bridges the representational gap between the VAE’s local structures and the VFM’s dense semantics. This design facilitates the seamless injection of contextual knowledge while actively preserving the VAE’s native structured semantics, thereby encouraging emergent disentanglement without requiring explicit regularization constraints. When integrated with flow-based transformers SiTs [27], Send-VAE can significantly accelerate the SiT training compared with REPA and achieves a new state-of-the-art FID score of 1.21 and 1.75 with and without classifier-free guidance on ImageNet 256 × 256 generation.

In summary, this paper makes the following key contributions:

- – We identify semantic disentanglement as a core property of generationfriendly VAEs, verified by the strong correlation between low-level attribute prediction accuracy and downstream generative performance.
- – We propose Send-VAE, a simple yet effective method for enhancing semantic disentanglement via alignment with vision foundation models through a sophisticated non-linear mapper network.
- – We demonstrate that Send-VAE substantially accelerates diffusion model training and establishes new state-of-the-art results on ImageNet 256×256 generation.

### 2 Related work

Tokenizers for Image Generation. Image tokenizers serve as the crucial bridge between high-dimensional pixel space and the compact latent space required by downstream generative models. These tokenizers are broadly categorized into continuous and discrete types. Continuous tokenizers, exemplified by Variational Autoencoders (VAEs) [17], are widely adopted as the foundational representation in diffusion-based models [27, 31, 33]. Conversely, discrete tokenizers, represented by VQGAN [11], dominate autoregressive (AR) generation paradigms. However, because traditional tokenizers are typically optimized with pure pixel-level reconstruction objectives, their latent spaces inevitably suffer from a semantic gap, making them not well aligned with the requirements of generation tasks. To tackle this semantic gap, recent studies draw inspiration from advances in diffusion transformer training [44] and attempt to explicitly align the VAE latent representations with the dense feature spaces of pre-trained VFMs.

For instance, VA-VAE [42] enforces a direct alignment between VAE latents and VFM semantics, achieving significant generative performance gains while preserving reconstruction capabilities. Similarly, inspired by MAE [14], MAETok [6] incorporates masked image modeling into tokenizer training, leveraging multiple target features to construct a semantically rich latent space. Parallel alignment strategies have also emerged in discrete tokenizers [23,40]. While these explicit alignment strategies deliver appreciable empirical gains, they inherently adopt the identical semantic targets to both VAEs and LDMs. As we established, this direct, shallow semantic forcing severely neglects the distinct representational role of VAEs, inevitably leading to feature entanglement and the suppression of fine-grained, low-level attributes. In contrast to explicit alignment, REPA-E [21] adopts an end-to-end joint training framework, directly backpropagating the representation alignment loss from diffusion transformers to the VAE. Although REPA-E avoids explicit feature mapping and achieves notable improvements, its black-box joint training paradigm sidesteps a more fundamental question: what intrinsic characteristics actually constitute a generation-friendly VAE? Diverging from both rigid explicit alignment and opaque joint training, we argue that the representational needs of LDMs and VAEs differ fundamentally, and the true catalyst for a generation-friendly VAE is its semantic disentanglement capability. Rather than naively forcing VFM semantics onto VAE latents, our Send-VAE introduces a non-linear mapping mechanism. This design effectively absorbs the rich contextual guidance from VFMs while actively safeguarding the structured, fine-grained details of the VAE, thereby ensuring high-quality generation.

Diffusion models for image generation. Diffusion models have emerged as a dominant paradigm in generative modeling, formulating high-fidelity image synthesis as a progressive denoising process. Early approaches, such as DDPM [16] and DDIM [36], operate directly in the high-dimensional pixel space, which heavily burdens computational resources and restricts scalability. To alleviate this, Latent Diffusion Models (LDMs) [33] map images into a lower-dimensional, compressed latent space via pre-trained VAEs, enabling dramatically faster training and inference. Alongside this spatial compression, the field has witnessed a significant architectural paradigm shift: transitioning from the traditional U-Net backbones [29,33] to transformer-based designs (e.g., DiTs) [27,31], which are inherently more adept at capturing complex, long-range dependencies. Beyond architectural upgrades, recent research heavily emphasizes enhancing the representation learning capabilities of the denoising network itself. Inspired by selfsupervised learning, methods like MaskDiT [47] and SD-DiT [49] adapt masked modeling paradigms [14,48] to accelerate feature learning within the DiT framework. More notably, a burgeoning line of work explicitly aligns the diffusion model’s intermediate features with the dense semantics of pre-trained Vision Foundation Models (VFMs). For instance, REPA [44] leverages external semantic priors from a frozen, high-capacity encoder to regularize the generative process. Building upon this, SARA [7] introduces additional structural and adversarial alignment objectives, while SoftREPA [20] extends this semantic alignment to multimodal text embeddings. Alternatively, Dispersive Loss [39] demonstrates

that even internal representation regularization, without relying on external VFMs, can significantly boost generative modeling. However, these works treat the underlying VAE latent space as a static, flawless canvas and overlook the representation bottleneck within the VAE itself. Deviating from these denoisingcentric alignments, our work shifts the focus back to the VAEs, proving that a semantically disentangled VAE is a crucial prerequisite for unleashing the full potential of modern diffusion models.

### 3 Method

In this section, we provide a comprehensive introduction to the design of SendVAE. We begin by rethinking existing VAE evaluation metrics and empirically establishing that semantic disentanglement is the fundamental prerequisite for a generation-friendly VAE. To explicitly cultivate this property, we introduce Send-VAE, which incorporates a non-linear mapping architecture to safely distill semantic knowledge from VFMs without compromising the VAE’s inherent structural integrity.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | |[Figure 6]<br><br>|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

[Figure 7]

[Figure 8]

[Figure 9]

- Fig. 2: We conduct experiments with three recently proposed evaluation methods for VAE latent space, and show their correlation with down stream generation performance (gFID). Experimental results on six VAEs with identical specifications indicate that these metrics do not accurately reflect the impact of VAEs on downstream generative performance. Conversely, we find that the ability of VAEs regarding low-level attributes is the key factor.

#### 3.1 Rethinking VAE Evaluation: The Disentanglement Hypothesis

To answer the question of what intrinsic characteristics a generation-friendly VAE should possess, we first investigate the behavior of VAE latent spaces using three recently proposed evaluation methods: semantic gap [44], latent space uniformity [42], and latent space discrimination [6]. For the semantic gap, linear probing on ImageNet classification is adopted following REPA [44]. Next, for latent space uniformity, we calculate the Gini coefficients of the data point distribution using kernel density estimation (KDE), as done in VA-VAE [42]. As for latent space discrimination, we fit a Gaussian mixture model (GMM) into the latent space following MAETok [6]. We benchmark four publicly available VAEs: SD-VAE [33], VA-VAE (f16d32) [42], E2E-VAE [21], and IN-VAE [21], alongside two different types of Send-VAE, with the final results presented in Fig. 2.

Uniformity and discrimination are insufficient indicators. As shown in Fig. 2, while VA-VAE exhibits improved uniformity and enhanced downstream generation performance compared with IN-VAE, this positive correlation does not hold true for E2E-VAE. A similarly inconsistent pattern emerges in the evaluation of latent space discrimination. We argue that these metrics only partially reflect the impact of VAEs on generative modeling and fail to accurately characterize a truly generation-friendly VAE.

Semantic disentanglement is the key catalyst. Aligning the hidden states of a diffusion model with pre-trained VFMs was first proposed in REPA [44] to reduce the semantic gap, which accelerates convergence. However, for VAEs, we observe that while directly injecting semantic information yields partial generation improvements (e.g., VA-VAE over IN-VAE), it is not the ultimate requirement, considering the further gains achieved by E2E-VAE without explicit semantic forcing. Drawing inspiration from recent tokenizer analyses [4], we hypothesize that the semantic disentanglement ability of a VAE is the underlying key factor. To verify this, we conduct linear probing on attribute prediction tasks. As shown in Fig. 2 (right), the Top5 Recall on DeepFashion [24] dataset shows a striking positive correlation with the generation performance, strongly validating our hypothesis. Notably, our Send-VAE achieves superior semantic disentanglement, directly translating to state-of-the-art generation performance.

#### 3.2 The Semantic Entanglement Dilemma in Direct Alignment

While recent explicit alignment strategies attempt to inject VFM semantics into VAEs, they predominantly rely on shallow mapping networks, such as a simple multilayer perceptron (MLP) used in VA-VAE [42]. However, this direct alignment paradigm harbors a critical conceptual flaw. Pre-trained VFMs (e.g., DINOv2) inherently encode highly condensed, globalized, and deeply entangled high-level semantics. Conversely, a foundational VAE must preserve fine-grained, disentangled local visual elements (e.g., textures, colors, and structures) to ensure high-fidelity tokenization. Forcing a rigid, shallow alignment between these two fundamentally mismatched representational spaces inevitably compels the VAE to discard its delicate structural details to accommodate the VFM’s dense representations. This indiscriminate semantic forcing overwhelmingly couples low-level features, leading to severe semantic collapse and feature entanglement, which ultimately bottlenecks the downstream generative modeling.

#### 3.3 Semantic Disentangled VAE

To overcome the aforementioned dilemma and actively enhance the disentanglement capability, we propose the Semantic-disentangled VAE (Send-VAE). Deviating from the direct alignment paradigm, Send-VAE introduces a sophisticated non-linear mapper network to bridge the substantial representational gap between the VAE and VFMs.

Specifically, rather than utilizing a simple MLP mapper, our mapper network consists of a patch embedding layer, a stack of Vision Transformer (ViT) [10] layers, and a final MLP projector. This deep architecture acts as an essential semantic buffer. It possesses sufficient capacity to digest and translate the dense, entangled semantics from the VFM into a structured format that the VAE can safely absorb. Consequently, this non-linear buffering facilitates the seamless injection of valuable contextual knowledge while actively safeguarding the VAE’s native structured semantics, naturally encouraging emergent disentanglement without explicit rigid regularization. The overall framework is illustrated in Fig. 3.

Formally, given a clean image x, let z be the latent representation of x output by the VAE Vθ. Let f denote a frozen Vision Foundation Model, where y = f(x) ∈ RN×D is the encoded representation of x, with N and D being the number of patches and the embedding dimension of f, respectively. To ensure the learned latent space is strictly robust and aligned with the actual requirements of downstream generative modeling, we propose a diffusion-aware alignment strategy. Instead of aligning clean latent codes, Send-VAE simulates the input distribution of downstream diffusion models by injecting random Gaussian noise into z to obtain the noisy latent zt:

1. VAE Training 2. Diffusion Model Training

Reconstruction Objective Denoising Objective

VAE

SiT Block

Decoder

SiT Block

Representation Alignment

MLP

MLP

ViT Block

SiT Block

Pre-trained Vision Foundation Model

ViT Block

SiT Block

VAE Encoder

[Figure 10]

[Figure 11]

VAE

Fig. 3: The overall structure of Send-VAE.

zt = (1 − αt)ϵ + αtz;ϵ ∈ N(0,I), (1)

where αt is randomly sampled from a uniform distribution between 0 and 1 and t is the time step (following the formulation in SiT [27]). Subsequently, the nonlinear mapper network hϕ transforms zt into hϕ(zt). The alignment loss is then calculated using the patch-wise cosine similarity between the transformed latent and the VFM features:

N

hϕ(zt)[n] · f(x)[n] ∥hϕ(zt)[n]∥∥f(x)[n]∥

1 N

, (2)

1 −

Lalign =

n=1

where n is the patch index.

In practice, we utilize Lalign to fine-tune a pre-trained VAE for fast convergence. The original VAE training objective LVAE [1] is simultaneously maintained to preserve foundational reconstruction capabilities, which comprises reconstruction losses (LMSE, LLPIPS), adversarial GAN loss (LGAN), and KL divergence loss (LKL). Thus, the overall training objective is formulated as:

L(θ,ϕ) = λalignLalign + LVAE, (3)

where θ and ϕ refer to the trainable parameters of the VAE and the non-linear mapper network, respectively.

### 4 Experiments

In this section, we conduct comprehensive experiments on the ImageNet dataset [8] at 256×256 resolution to validate the design choices of Send-VAE, and benchmark its generation performance to demonstrate its superiority over existing approaches.

#### 4.1 Implementation Details

We follow the same set up as in REPA-E [21] unless otherwise specified. All training is conducted on the training split of ImageNet [8]. The data preprocessing protocol is same as in ADM [9] including center-crop and resizing to 256x256 resolution. The mapper network consists of one layer of Transformer encoder with 12 heads for VFMs with 768 hidden dimension and 16 heads for VFMs with 1024 hidden dimension.

For VAE training, we train 80 epoch with a global batch size of 1024, AdamW [26] optimizer is adopted and the learning rate is set to 3.0 × 10−4. As for the initialization, we experiment with publicly available VAEs, including SDVAE (f8d4) [33], VA-VAE (f16d32) [42], and IN-VAE (f16d32), which is trained on ImageNet following [33]. Experimentally, we choose VA-VAE as the default setting. As for alignment loss Lalign, we use DINOv2 [30] as the vision foundation model, and λalign is set to 1.0.

For diffusion models, we choose SiT-XL/1 and SiT-XL/2 for VAEs with 4× and 16× downsampling rates, respectively, where 1 and 2 denote the patch sizes in the transformer embedding layer. We train either 80 epoch or 800 epoch with a global batch size of 256, and gradient clipping and exponential moving average (EMA) are applied stable optimization. The learning rate is set to 1.0× 10−4 and AdamW optimizer is used. REPA loss is also included following the setting in [44].

For sampling, the SDE Euler-Maruyama sampler is used, the number of function evaluations (NFE) is set to 250 by default and the cfg scale is set to 2.5

#### 4.2 Evaluation Metrics

For image generation evaluation, we strictly follow the ADM setup [9]. Generation quality is assessed using Fréchet Inception Distance (gFID) [15], Structural FID (sFID) [28], Inception Score (IS) [34], Precision, and Recall [18], computed on 50K generated samples. For sampling, we adopt the SDE Euler–Maruyama solver with 250 steps, following the protocols of REPA [44] and REPA-E [21]. For VAE evaluation, we report reconstruction FID (rFID) on 50K validation images from ImageNet at 256×256 resolution.

#### 4.3 System-level comparison on ImageNet 256x256 generation

To verify the effectiveness of Send-VAE, we conduct system-level comparison on ImageNet 256x256 generation with and without classifier-free guidance (CFG),

- Table 1: System-level comparison on ImageNet 256x256 generation with and without classifier-free guidance (CFG). Our Send-VAE can significant accelerate the convergence of diffusion models, which achieves a gFID socre of 2.88/1.41 wo/w CFG for only 80 epoch of training. Although the performance gap between Send-VAE and E2EVAE is narrowing when training longer, Send-VAE still achieves further improvements.

|Tokenizer<br><br>|Method<br><br>|Training Epoch<br><br>|#params|rFID|Generation w/o CFG<br><br>|Generation w/ CFG|
|---|---|---|---|---|---|---|
| | | | | |gFID sFID IS Prec. Rec.<br><br>|gFID sFID IS Prec. Rec.|

AutoRegressive (AR) MaskGiT MaskGIT [5] 555 227M 2.28 6.18 - 182.1 0.80 0.51 - - - - VQGAN LlamaGen [37] 300 3.1B 0.59 9.38 8.24 112.9 0.69 0.67 2.18 5.97 263.3 0.81 0.58 VQVAE VAR [38] 350 2.0B - - - - - - 1.80 - 365.4 0.83 0.57

LFQ tokenizers MagViT-v2 [43] 1080 307M 1.50 3.65 - 200.5 - - 1.78 - 319.4 - LDM MAR [22] 800 945M 0.53 2.35 - 227.8 0.79 0.62 1.55 - 303.7 0.81 0.62 Latent Diffusion Models (LDM)

|SD-VAE [33]<br><br>|MaskDiT [47] DiT [31] SiT [27] FastDiT [41] MDT [12] MDTv2 [13] REPA [44]|1600 1400 1400 400 1300 1080 800<br><br>|675M 675M 675M 675M 675M 675M 675M|0.61<br><br>|5.69 10.34 177.9 0.74 0.60 9.62 6.85 121.5 0.67 0.67 8.61 6.32 131.7 0.68 0.67 7.91 5.45 131.3 0.67 0.69<br><br>6.23 5.23 143.0 0.71 0.65<br><br><br>- - - - 5.90 5.73 157.8 0.70 0.69<br><br>|2.28 5.67 276.6 0.80 0.61 2.27 4.60 278.2 0.83 0.57 2.06 4.50 270.3 0.82 0.59 2.03 4.63 264.0 0.81 0.60 1.79 4.57 283.0 0.81 0.61 1.58 4.52 314.7 0.79 0.65 1.42 4.70 305.7 0.80 0.65|
|---|---|---|---|---|---|---|
|VA-VAE [42]<br><br>|LightingDiT [42]<br><br>|80 800|675M 675M<br><br>|0.28 0.28<br><br>|4.29 - - - 2.17 4.36 205.6 0.77 0.65<br><br>|- - - - 1.35 4.15 295.3 0.79 0.65|
|MAETok [6]|LightingDiT [42]<br><br>|800|675M<br><br>|0.48<br><br>|2.21 - 208.3 - -|1.73 - 308.4 - -|
|E2E-VAE [21]|REPA [44]<br><br>|80 800|675M 675M<br><br>|0.28<br><br>|3.46 4.17 159.8 0.77 0.63 1.83 4.22 217.3 0.77 0.66<br><br>|1.67 4.12 266.3 0.80 0.63 1.26 4.11 314.9 0.79 0.66|
|Send-VAE<br><br>|REPA [44]<br><br>|80 800|675M 675M<br><br>|0.31<br><br>|2.88 4.67 175.3 0.78 0.62 1.75 4.41 218.57 0.79 0.64<br><br>|1.41 4.41 301.7 0.79 0.65 1.21 4.10 315.1 0.79 0.66|

and present the results in Tab. 1. As we can see, using the same vision foundation model DINOV2, Send-VAE can achieve notable performance gains compared with E2E-VAE and set a new state-of-the-art generation FID score of 1.21 and

- 1.75 with and without classifier-free guidance on ImageNet 256x256 generation. These results highly demonstrate the effectiveness of enhancing the semantic

10 20 40 80 800 Training epoch

2

4

8

20

50

90

gFID-50K

REPA

REPA-E

Send-VAE (Ours)

Fig. 4: Send-VAE can significantly accelerate the SiT training.

disentanglement ability of VAE. Meanwhile, we can notice that Send-VAE can significantly speed up the convergence of diffusion models compared with REPA (shown in Fig. 4). When compared with REPA-E, Send-VAE can narrow the gFID score from 3.46 to

- 2.88 for generation without CFG when training with only 80 epoch, which demonstrates that Send-VAE is a generation-friendly VAE and can facilitate the learning of diffusion models. As for reconstruction performance, we observe that the reconstruction performance of Send-VAE is slightly inferior to that of VAVAE. We attribute this to the semantic disentangled

Training Iteration 10 Epoch

20 Epoch 80 Epoch 10 Epoch 20 Epoch 80 Epoch 10 Epoch 20 Epoch 80 Epoch

[Figure 12]

[Figure 13]

[Figure 14]

Send-VAEE2E-VAEVA-VAE

10 Epoch 20 Epoch 80 Epoch 10 Epoch 20 Epoch 80 Epoch 10 Epoch 20 Epoch 80 Epoch

[Figure 15]

[Figure 16]

[Figure 17]

E2E-VAESend-VAEVA-VAE

Fig. 5: Qualitative comparisons among VA-VAE, E2E-VAE, and Send-VAE. Results for both methods are sampled using the same seed, noise and class label. The classifierfree guidance scale is set to 4.0.

latent space of Send-VAE, which prevents it from capturing excessive fine-grained low-level details.

Besides, we also provide qualitative comparisons among VA-VAE, E2E-VAE and Send-VAE in Fig. 5 We generates images from the same label and initial noise using checkpoints trained by 10 epoch, 20 epoch, and 80 epoch, respectively. As we can see, training diffusion models using Send-VAE demonstrates superior image generation quality compared to VA-VAE and E2E-VAE. Meanwhile, Send-VAE can significantly speed up the training process of diffusion models, evidenced by the more structurally meaningful images during early stages of training process. Some visualization results are presented in Fig. 6 using SendVAE and SiT-XL/1 to show that training diffusion models with Send-VAE can generate high-quality images.

#### 4.4 Ablation Studies

In this section, we provide detailed ablation studies to demonstrate the effectiveness of each design in Send-VAE. Unless otherwise specified, we train a SiT-B/1 with REPA loss for 80 epoch, and report the downstream generation performance without classifier-free guidance.

Ablation on Depth of Mapper Network. We ablate the depth of our proposed mapper network to analyze its impact on downstream generation performance. As shown in Tab. 2, a mapper with one layer of ViT achieves the best

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

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Fig. 6: Qualitative Results on ImageNet 256 × 256 using Send-VAE and SiT-XL.

- Table 2: Ablation on the depth of mapper network.

Depth gFID↓ sFID↓ IS↑ Prec.↑ Rec.↑

- 0 9.20 7.06 104.2 0.73 0.57
- 1 8.42 5.05 108.3 0.74 0.60
- 2 9.47 5.33 100.4 0.73 0.60

###### Table 3: Ablation on noise injection.

Noise Injection

gFID↓ sFID↓ IS↑ Prec.↑ Rec.↑

✗ 8.42 5.05 108.3 0.74 0.60 ✓ 7.57 5.37 115.3 0.74 0.60

performance (gFID=8.42), outperforming both shallower (0 layer) and deeper (2 layer) configurations. We argue that the insufficient capacity of shallow mapper fails to bridge the representation gap between VAE and visual foundation models, resulting in a decrease in the semantic disentanglement ability of VAE. While for the deeper one, it weaken the foundational model’s impacts on VAE due to the stronger fitting capability. Such experimental results demonstrate the necessity of employing a mapper network to bridge representation gap, which can facilitate effective semantic injection.

#### Ablation on Injecting Noise to Latent Representations. Tab. 3 presents

the ablation results of injecting noise to latent representations. As we can see, injecting noise during the alignment process can bring significant performance gains. We attribute its effectiveness to a form of data augmentation, which ensures that even with noise injected, the latent representation extracted by the VAE retains rich disentangled semantic information, making it better suited for the denoising process of the downstream diffusion model.

Ablation on Vision Foundation Models. We also investigate the influence of vision foundation models and present the ablation results in Tab. 4. Specifically, we include six types of vision foundation models, including MAE [14], CLIP [32], I-JEPA [3], SigLIP [46], DINOv2 [30], and DINOv3 [35]. As we can see, regardless of the type of vision foundation models, adding Lalign consistently improve the generation performance of diffusion models. Among them, the DINO

Table 4: Ablation on different vision foundation models (VFMs)

VFMs gFID↓ sFID↓ IS↑ Prec.↑ Rec.↑

None (Baseline) 11.40 6.58 93.5 0.71 0.59 MAE 10.01 5.62 99.2 0.71 0.60 CLIP 9.85 5.59 100.8 0.71 0.62

I-JEPA 9.70 5.40 102.9 0.72 0.60 SigLIP 9.10 5.21 108.1 0.72 0.61s

DINOv2 7.57 5.37 115.3 0.74 0.60 DINOv3 7.16 5.57 125.3 0.75 0.58

family (DINOv2 and DINOv3) achieves the best performance, which is consistent with the findings of REPA and REPA-E. We argue that the object-centric features of DINO can more effectively facilitate the VAE in learning a semantic disentangled latent space, thus resulting in superior generation performance.

Table 5: Ablation on the VAE specification. VAE Specification rFID↓ gFID↓ sFID↓ IS↑ Prec.↑ Rec.↑

f16d16 0.48 13.85 5.30 65.0 0.68 0.62 +Lalign 0.47 7.02 4.64 120.41 0.75 0.60 f16d32 0.26 17.43 5.93 72.7 0.64 0.63 +Lalign 0.28 8.25 4.68 105.2 0.74 0.60 f16d64 0.17 26.27 8.02 54.69 0.58 0.64 +Lalign 0.19 10.99 5.17 94.11 0.69 0.62

Ablation on the VAE specification. We implement our Send-VAE cross different VAE specification and show the results in Tab. 5. It can be observed that under varying VAE latent dimensions, Send-VAE consistently achieves significant performance improvements, demonstrating the effectiveness of our method. Furthermore, while the reconstruction quality improves with increased VAE model capacity, the generation quality reduces. And the integration of SendVAE stably enhances the downstream generation performance.

Ablation on the Resolution and Initialization of VAE. To demonstrate the generalization of our method to various VAE initialization, we conduct experiments on three commonly used VAEs, including SD-VAE [1], IN-VAE [21] and VA-VAE [42]. The results are shown in Tab. 6. As we can see, across all variations, our Lalign can consistently improve final generation performance, which demonstrates that insensitiveness of our method to the VAE initialization. Moreover, to verify the effectiveness of Send-VAE on larger resolution generation,

we also finetune SD-VAE [1] using the proposed LAlign at 512x512 resolution on ImageNet-1k. Then, a Sit-B is trained following the setup in ablation studies. As shown in Tab. 6, at the 512x512 resolution, the proposed method still significantly improves downstream generative performance, demonstrating the effectiveness of our approach.

Table 6: Ablation on the resolution and initialization of VAE. VAE Initialization gFID↓ sFID↓ IS↑ Prec.↑ Rec.↑ SD-VAE (256 × 256) 21.41 5.30 65.0 0.62 0.63

+Lalign 11.86 5.25 95.2 0.73 0.58 SD-VAE (512 × 512) 23.59 6.74 65.67 0.71 0.59 +Lalign 13.32 4.75 93.15 0.78 0.60 IN-VAE (256 × 256) 17.43 5.93 72.7 0.64 0.63

+Lalign 8.25 4.68 105.2 0.74 0.60 VA-VAE (256 × 256) 11.40 6.58 93.5 0.71 0.59

+Lalign 7.57 5.37 115.3 0.74 0.60

Analysis of Reconstruction Performance. To explicitly analyze the trade-off between pixel fidelity and generation quality, we evaluate PSNR, SSIM, and LPIPS on ImageNet-1k validation set. As shown in Tab. 7, both E2E-VAE and Send-VAE exhibit similar reconstruction profiles (PSNR≈27.6, SSIM≈0.77), which are slightly lower than the Naive VAE. This confirms a community consensus: generation-friendly VAEs prioritize semantic structure over high-frequency pixel noise. Despite similar pixel-level metrics, Send-VAE achieves a better LPIPS (0.101 vs. 0.110) than E2E-VAE, indicating superior perceptual preservation. This marginal drop in PSNR is a necessary cost for semantic disentanglement. Crucially, with comparable reconstruction costs to E2E-VAE, Send-VAE yields superior generation performance and faster convergence.

Table 7: Measurement of the reconstruction performance. VAE types rFID↓ PSNR↑ LPIPS ↓ SSIM↑ gFID↓ Naive VAE 0.26 28.59 0.089 0.80 17.43

VA-VAE 0.28 27.96 0.096 0.79 11.40

E2E-VAE 0.28 27.63 0.110 0.77 8.96 Send-VAE 0.31 27.62 0.101 0.77 7.57

#### 4.5 Measurement of semantic disentanglement ability

To give a system-level measurement of semantic disentanglement capability, we adopt linear probing on attribute prediction benchmarks across distinct domains

Table 8: System-level measurement of semantic disentanglement ability of various VAEs. F1 score is adopted for all benchmarks.

Benchmarks IN-VAE VA-VAE E2E-VAE Send-VAE CelebA 0.6222 0.6347 0.6439 0.6647

DeepFasion 0.0786 0.1094 0.1177 0.1385 AwA 0.5567 0.5948 0.6441 0.6623 gFID 17.43 11.40 8.96 7.57

to measure the semantic disentanglement ability of various VAEs. Specifically, three attribute prediction benchmarks are used to ensure a comprehensive evaluation, including CelebA [25], DeepFashion [24] and AwA [19]. We conduct linear probing on the flattened latent representation from VAE encoder and show the results in Tab. 8. As we can see, among all benchmarks, the performance of attribute prediction is positively correlated with the down-stream generation performance. These results strongly support our hypothesis, and making the linear probing on attribute prediction task a suitable metric to evaluate the goodness of a VAE for diffusion. Meanwhile, we observe that Send-VAE can significantly enhance the semantic disentanglement ability of VAE and achieve superior generation performance.

### 5 Conclusion

In this paper, we investigate the fundamental characteristics that constitute a truly generation-friendly VAE. By revisiting recent representation alignment paradigms, we identify a critical conceptual flaw in prior works: the erroneous assumption that VAEs and LDMs share identical semantic alignment targets. Diverging from this, we empirically demonstrate that while LDMs thrive on abstract semantics, VAEs fundamentally require strong semantic disentanglement to faithfully preserve fine-grained, structured visual elements. This insight is robustly validated by a striking correlation we established between the linear separability of low-level attributes in the latent space and downstream generation quality. To explicitly cultivate this property, we introduce the Semanticdisentangled VAE (Send-VAE). Rather than forcing a rigid, shallow alignment, Send-VAE leverages a non-linear mapping architecture, which effectively bridges the representation gap by safely absorbing rich contextual guidance from VFMs while actively safeguarding the VAE’s native structured semantics. Extensive experiments on ImageNet 256×256 confirm the empirical superiority of our approach: Send-VAE not only significantly accelerates the training convergence of flow-based transformers such as SiTs, but also sets a new state-of-the-art generation benchmark, achieving remarkable FID scores of 1.21 and 1.75 with and without classifier-free guidance, respectively.

### References

- 1. AI, S.: Improved autoencoders ... https://huggingface.co/stabilityai/sdvae-ft-mse (nd), accessed: April 11, 2025.
- 2. Albergo, M.S., Vanden-Eijnden, E.: Building normalizing flows with stochastic interpolants. In: The Eleventh International Conference on Learning Representations

(2023), https://openreview.net/forum?id=li7qeBbCR1t

- 3. Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., Ballas, N.: Self-supervised learning from images with a joint-embedding predictive architecture. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15619–15629 (2023)
- 4. Beyer, L.L., Li, T., Chen, X., Karaman, S., He, K.: Highly compressed tokenizer can generate without training. arXiv preprint arXiv:2506.08257 (2025)
- 5. Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W.T.: Maskgit: Masked generative image transformer. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 11315–11325 (2022)
- 6. Chen, H., Han, Y., Chen, F., Li, X., Wang, Y., Wang, J., Wang, Z., Liu, Z., Zou, D., Raj, B.: Masked autoencoders are effective tokenizers for diffusion models. In: Forty-second International Conference on Machine Learning (2025)
- 7. Chen, H., Wang, J., Tan, Z., Li, H.: Sara: Structural and adversarial representation alignment for training-efficient diffusion models. arXiv preprint arXiv:2503.08253

(2025)

- 8. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A largescale hierarchical image database. In: 2009 IEEE conference on computer vision and pattern recognition. pp. 248–255. Ieee (2009)
- 9. Dhariwal, P., Nichol, A.: Diffusion models beat gans on image synthesis. Advances in neural information processing systems 34, 8780–8794 (2021)
- 10. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021), https://openreview. net/forum?id=YicbFdNTTy
- 11. Esser, P., Rombach, R., Ommer, B.: Taming transformers for high-resolution image synthesis. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12873–12883 (2021)
- 12. Gao, S., Zhou, P., Cheng, M.M., Yan, S.: Masked diffusion transformer is a strong image synthesizer. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 23164–23173 (2023)
- 13. Gao, S., Zhou, P., Cheng, M.M., Yan, S.: Mdtv2: Masked diffusion transformer is a strong image synthesizer. arXiv preprint arXiv:2303.14389 (2023)
- 14. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16000–16009 (2022)
- 15. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)
- 16. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)
- 17. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)

- 18. Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., Aila, T.: Improved precision and recall metric for assessing generative models. Advances in neural information processing systems 32 (2019)
- 19. Lampert, C.H., Nickisch, H., Harmeling, S.: Attribute-based classification for zeroshot visual object categorization. IEEE transactions on pattern analysis and machine intelligence 36(3), 453–465 (2013)
- 20. Lee, J.Y., Cha, B., Kim, J., Ye, J.C.: Aligning text to image in diffusion models is easier than you think. arXiv preprint arXiv:2503.08250 (2025)
- 21. Leng, X., Singh, J., Hou, Y., Xing, Z., Xie, S., Zheng, L.: Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483 (2025)
- 22. Li, T., Tian, Y., Li, H., Deng, M., He, K.: Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems 37, 56424–56445 (2024)
- 23. Li, X., Qiu, K., Chen, H., Kuen, J., Gu, J., Raj, B., Lin, Z.: Imagefolder: Autoregressive image generation with folded tokens. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum? id=QE1LFzXQPL
- 24. Liu, Z., Luo, P., Qiu, S., Wang, X., Tang, X.: Deepfashion: Powering robust clothes recognition and retrieval with rich annotations. In: Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2016)
- 25. Liu, Z., Luo, P., Wang, X., Tang, X.: Deep learning face attributes in the wild. In: Proceedings of International Conference on Computer Vision (ICCV) (December 2015)
- 26. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2019), https://openreview.net/forum? id=Bkg6RiCqY7
- 27. Ma, N., Goldstein, M., Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E., Xie, S.: Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In: European Conference on Computer Vision. pp. 23–40. Springer

(2024)

- 28. Nash, C., Menick, J., Dieleman, S., Battaglia, P.: Generating images with sparse representations. In: International Conference on Machine Learning. pp. 7958–7968. PMLR (2021)
- 29. Nichol, A.Q., Dhariwal, P.: Improved denoising diffusion probabilistic models. In: International conference on machine learning. pp. 8162–8171. PMLR (2021)
- 30. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., HAZIZA, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y., Li, S.W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (2024), https://openreview.net/forum?id=a68SUt6zFt, featured Certification
- 31. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205 (2023)
- 32. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

- 33. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 34. Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., Chen, X.: Improved techniques for training gans. Advances in neural information processing systems 29 (2016)
- 35. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al.: Dinov3. arXiv preprint arXiv:2508.10104 (2025)
- 36. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. In: International Conference on Learning Representations (2021), https://openreview.net/forum? id=St1giarCHLP
- 37. Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng, B., Luo, P., Yuan, Z.: Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525 (2024)
- 38. Tian, K., Jiang, Y., Yuan, Z., Peng, B., Wang, L.: Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems 37, 84839–84865 (2024)
- 39. Wang, R., He, K.: Diffuse and disperse: Image generation with representation regularization. arXiv preprint arXiv:2506.09027 (2025)
- 40. Xiong, T., Liew, J.H., Huang, Z., Feng, J., Liu, X.: Gigatok: Scaling visual tokenizers to 3 billion parameters for autoregressive image generation. arXiv preprint arXiv:2504.08736 (2025)
- 41. Yao, J., Wang, C., Liu, W., Wang, X.: Fasterdit: Towards faster diffusion transformers training without architecture modification. Advances in Neural Information Processing Systems 37, 56166–56189 (2024)
- 42. Yao, J., Yang, B., Wang, X.: Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15703–15712 (2025)
- 43. Yu, L., Lezama, J., Gundavarapu, N.B., Versari, L., Sohn, K., Minnen, D., Cheng, Y., Gupta, A., Gu, X., Hauptmann, A.G., Gong, B., Yang, M.H., Essa, I., Ross, D.A., Jiang, L.: Language model beats diffusion - tokenizer is key to visual generation. In: The Twelfth International Conference on Learning Representations

(2024), https://openreview.net/forum?id=gzqrANCF4g

- 44. Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., Xie, S.: Representation alignment for generation: Training diffusion transformers is easier than you think. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum?id=DJSZGGZYVi
- 45. Zha, K., Yu, L., Fathi, A., Ross, D.A., Schmid, C., Katabi, D., Gu, X.: Languageguided image tokenization for generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15713–15722 (2025)
- 46. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11975–11986 (2023)
- 47. Zheng, H., Nie, W., Vahdat, A., Anandkumar, A.: Fast training of diffusion models with masked transformers. Transactions on Machine Learning Research (2024), https://openreview.net/forum?id=vTBjBtGioE
- 48. Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., Kong, T.: Image bert pre-training with online tokenizer. In: International Conference on Learning Representations

###### 49. Zhu, R., Pan, Y., Li, Y., Yao, T., Sun, Z., Mei, T., Chen, C.W.: Sd-dit: Unleashing the power of self-supervised discrimination in diffusion transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8435–8445 (2024)

