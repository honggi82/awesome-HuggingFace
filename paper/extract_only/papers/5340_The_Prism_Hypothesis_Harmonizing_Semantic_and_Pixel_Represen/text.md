## The Prism Hypothesis: Harmonizing Semantic and Pixel Representations via Unified Autoencoding

Weichen Fan1,2 Haiwen Diao1 Quan Wang2 Dahua Lin2 Ziwei Liu1, 1S-Lab, Nanyang Technological University 2SenseTime Research

weichen002@e.ntu.edu.sg, haiwen.diao@ntu.edu.sg, {wangquan,dhlin}@sensetime.com, ziwei.liu@ntu.edu.sg

[Figure 1]

Github: https://github.com/WeichenFan/UAE. Hugging Face: https://huggingface.co/weepiess2383/UAE.

[Figure 2]

# arXiv:2512.19693v5[cs.CV]1Apr2026

[Figure 3]

###### Low-Level

(Image)

Low-Frequency

Appearance Geometry …

InternViT…

SigLIP,

Visual Abstraction

DINO,

###### High-Level

Encoders

Semantic

Category Attribute Relation …

Frequency-Band Modulator

###### UAE

(Text)

Pixel

Encoders

###### High-Level

###### Category Attribute Relation …

SD3VAE,FluxVAE…

Visual Fidelity

Starting/Evolving

SemanticEncoders

from

High-Frequency

Figure 1. The Prism Hypothesis. Our conceptual “prism” decomposes various natural inputs into spectral components along frequency. Low frequency bands capture global semantics and abstract meaning, while high frequency bands encode local detail and fine visual texture. This motivates our Unified Autoencoding (UAE), which harmonizes semantic and pixel representations within a single latent space.

### Abstract

that UAE effectively unifies semantic abstraction and pixellevel fidelity within a single latent space, achieving state-ofthe-art performance. Moreover, we show that UAE can be directly applied to pixel-space modeling, significantly improving both FID and IS over the vanilla JIT baseline.

Deep representations across modalities are inherently intertwined. In this paper, we systematically analyze the spectral characteristics of various semantic and pixel encoders. Interestingly, our study uncovers a highly inspiring and rarely explored correspondence between an encoder’s feature spectrum and its functional role: semantic encoders primarily capture low-frequency components that encode abstract meaning, whereas pixel encoders additionally retain high-frequency information that conveys fine-grained detail. This heuristic finding offers a unifying perspective that ties encoder behavior to its underlying spectral structure. We define it as the Prism Hypothesis, where each data modality can be viewed as a projection of the natural world onto a shared feature spectrum, just like the prism. Building on this insight, we propose Unified Autoencoding (UAE), a model that harmonizes semantic structure and pixel details via an innovative frequency-band modulator, enabling their seamless coexistence. Extensive experiments demonstrate

### 1. Introduction

Trained on massive corpora, recent foundation models have profoundly reshaped perception and generation systems, generalizing well across diverse downstream tasks [3, 24, 26, 30]. Yet, early advances in perception and generation evolve along largely separate trajectories. Their objectives are typically distributed across distinct network structures, e.g., employing pretrained semantic encoders [3, 26, 30], to capture high-level meaning, or pixel encoders [17, 32] to compress fine-grained visual detail. While each module excels within its own domain, this fragmentation compels subsequent unification efforts [10, 29, 38] to depend simultaneously on semantic and pixel encoders, forcing networks to reconcile fundamentally heterogeneous representations.

The sharp mismatch lowers training efficiency and induces representational conflicts, with these incompatible features often interfering rather than complementing one another.

This fragmentation uncovers a deep-seated tension between abstraction and fidelity, which is a driving force in shaping subsequent foundation models. To alleviate it, recent studies [34, 49, 50] attempt to transfer semantic encoders into visual generation domains alongside strong pixel decoders. This strategy substantially accelerates convergence and improves semantic correspondence, yet remains limited in recovering fine-grained visual details. In parallel, another research direction seeks to endow pixel encoders with semantic awareness through text supervision [44, 45, 47], semantic encoder distillation [4, 11], and hierarchical feature integration [44]. While these efforts move toward unifying understanding and generation representations in a single module, they often achieve coexistence through trade-offs rather than genuine integration.

At the core of this development lies a fundamental question: How is information about the world represented such that multimodal inputs share a common semantic meaning while preserving their native granularity of detail?

Impressively, we empirically observe that pre-trained semantic features, whether derived from text or vision, tend to reside at the coarse end of the decoupled feature spectrum, primarily capturing low-frequency structures such as categories, attributes, and relations. In contrast, pre-trained pixel-level features extend toward the finer end of the spectrum, representing higher-frequency components that convey intricate appearance and geometric detail. Strikingly, these complementary representations can be harmoniously integrated within a unified encoder, extremely aligning well with the spectral arrangement and fostering a progressive synergy from semantic perception to detailed reconstruction. We posit this as the Prism Hypothesis: as shown in Figure 1, the real-world inputs project onto a continuous shared feature spectrum, and what we perceive as different modalities are distinct slices of this underlying continuum.

From this perspective, we introduce Unified Autoencoding (UAE), a tokenizer that learns a shared latent space and harmonizes semantic structure and pixel-level detail. Specifically, UAE features a frequency decomposition that factorizes real-world content into a fundamental semantic band and residual pixel bands with controllable fine granularity. This design is not only grounded in comprehensive empirical evidence but also validated across diverse reconstruction and perception tasks. With its compact yet semantically expressive representations, UAE outperforms concurrent RAE [50], SVG [34], and UniFlow [49] across rFID, PSNR, gFID and Accuracy metrics on ImageNet, demonstrating that its learned latent space is both semantically representative and pixel-faithful.

Our work makes the following contributions.

- 1. We introduce the Prism Hypothesis, a spectral perspective that explains multimodal data through a shared fundamental band and modality-specific bands, supported by extensive empirical observations.
- 2. We propose Unified Autoencoding (UAE), a simple yet effective tokenizer via frequency transformation that integrates seamlessly with diffusion transformers and enables flexible generative modeling in the latent space.
- 3. UAE is effective in both pixel and latent spaces. In the latent space, UAE achieves competitive performance in both generative modeling and pixel-level reconstruction. In the pixel space, integrating UAE into JIT [19] leads to faster convergence and improved FID and IS.

### 2. Related Work

Unified Tokenizers and Unified Representations. Unifying the representations between pixel and semantic embeddings has become a central objective for existing foundation models. Joint embedding approaches align images and text in a shared representation and enable strong zeroshot transfer [14, 30], and have been extended to many modalities, e.g., audio, depth, thermal, and inertial signals [8]. In parallel, modality-agnostic backbones aim to build a unified architecture that can process diverse input modalities and generate task-specific outputs through learned queries or a shared token representation [2, 13, 21, 22, 42].

On the tokenizer side, discrete codebook methods have demonstrated that the design and granularity of visual tokens play a crucial role in determining how effectively a single sequence model can adapt to vision tasks [6, 48]. Recent work goes further and seeks tokenizers that support both understanding and generation at the same time. OmniTokenizer [41] learns a joint image video tokenizer with a spatial-temporal decoupled transformer and reports strong reconstruction and synthesis across both domains. Very recent studies deepen this trend. UniFlow [49] proposes a unified pixel flow tokenizer that adapts a pretrained encoder through layer-wise self-distillation and employs a lightweight patch-wise flow decoder, explicitly targeting the long-standing tension between semantic abstraction and pixel-faithful reconstruction. Two concurrent works remove the traditional variational bottleneck entirely and build unified representation latents for diffusion transformers. Diffusion Transformers with Representation Autoencoders (RAE) [50] replace the usual reconstruction-only encoder with a pretrained representation encoder, e.g., DINO or SigLIP, and a trained decoder, arguing that semantically rich latents accelerate convergence and improve generative fidelity. SVG [34] trains a diffusion model on DINO features with a small residual branch for details, reporting faster training, few-step sampling, and improved quality.

Our UAE aligns with this shift. It serves as a unified tokenizer that decouples continuous latent features explicitly

DINOv2 JEPA CLIP SigLIP2 SD-VAE InternViT

0.6

0.07

0.5

0.06

0.4

0.05

Energy()ek

0.04

0.3

6 7 8 9

0.2

0.1

| |
|---|

| |
|---|

0.0

0 2 4 6 8

Frequency band k

Figure 2. Frequency energy distribution. Normalized energy e(k) across frequency bands for diverse tokenizers. DINOv2 and CLIP focus on low-frequency (semantic) content, while SD-VAE retains more high-frequency energy, capturing finer details.

CLIP SigLIP2

| | | | | | | |
|---|---|---|---|---|---|---|
| | || |
|---|
<br><br>| | | | |
| || |
|---|
| | | | | |
| || |
|---|
<br><br>| | | | | |
| | | | | | | |
| | | | | | | |
| || |
|---|
<br><br>| | | | | |
| | | | | | | |

0.8

0.7

0.6

Recall@1

0.5

0.4

0.3

0.2

0.0 0.2 0.4 0.6 0.8 1.0

Low-pass cutoff (fraction of Nyquist)

Figure 3. Retrieval results via frequency filtering. Text–Image retrieval remains stable under low-pass filtering but degrades sharply under high-pass filtering, confirming that semantic alignment primarily resides in low-frequency components.

corresponding to the underlying spectral structure by factorizing real-world contents into a low-frequency base and residual high-frequency bands. It anchors semantics in the core representation while relegating fine-grained details to residuals for progressive reconstruction.

Frequency and Multi-Resolution Modeling. Classical image synthesis adopted pyramids and wavelets to separate structure by scale, enabling coarse-to-fine generation and targeted refinement of detail. A typical example is the Laplacian pyramid of adversarial networks [5], which trains a generator per level and synthesizes images by successively adding higher-frequency residuals. Subsequent analyses of neural networks from a spectral perspective showed that standard architectures prioritize low frequencies and learn higher frequencies later, a phenomenon known as spectral bias [31]. Two lines of work respond to this bias. The first uses input or architecture design to improve access to high frequencies, e.g., Fourier feature mappings and periodic activations that help multi-layer perceptrons represent fine detail [35, 37]. The second introduces frequency-aware objectives and signal processing choices, such as focal frequency loss to emphasize hard frequencies and alias-free synthesis to avoid spurious high-frequency artifacts [15, 16].

Modern generative models retain this multi-resolution view. Cascaded diffusion trains models at increasing resolutions, allowing each stage to learn the appropriate frequency band and its own error distribution [9]. Variants construct explicit feature pyramids or hierarchical patch schedules, enabling efficient diffusion on large images and video while preserving high-frequency detail [7, 36]. Recent latent diffusion designs introduce cross-magnification spaces or zoomable pyramids that share information across scales and enable large-image reconstruction without retraining [46]. In autoregressive models, VAR [39] casts generation as pre-

dicting the next scale or resolution, showing strong ImageNet results and clean scaling trends. This explicit progression from global layout to fine detail emerges as a viable alternative to diffusion. Building on this idea, Next Visual Granularity (NVG) generation [43] produces sequences at a fixed resolution but with progressively finer token granularity, surpassing prior VAR baselines while maintaining structured control over detail. Similarly, NFIG [12] performs discrete next-frequency prediction, demonstrating strong generative performance on the ImageNet benchmark. In pixel space, DCTdiff [25] explicitly trun image into DCT space and perform generative modeling. Our approach aligns with these trends but focuses on unified representation rather than the generator’s training schedule.

### 3. Methodology

#### 3.1. Preliminary Findings

The Prism Hypothesis. Natural inputs are regarded as projections of a common real-world signal onto a shared frequency spectrum. Semantic encoders emphasize a compact low-band that carries categories, attributes, and relations, while pixel encoders observe the same fundamental base together with higher bands that encode edges, textures, and fine appearance. The hypothesis indicates that cross-modal alignment depends primarily on the shared low band.

Formalization. Here, F and F† denote two-dimensional discrete Fourier transform and its inverse. For an image I ∈ [0,1]3×H×W and a smooth radial mask MLPρ that passes frequencies within normalized radius ρ ∈ (0,1],

IρLP = F† MLPρ ⊙ F(I) , (1) IρHP = F† MHPρ ⊙ F(I) , (2)

where MHPρ is complementary to MLPρ , and both masks use cosine transitions to limit ringing artifacts. All fil-

GAN Loss

[Figure 4]

🔥

[Figure 5]

[Figure 6]

[Figure 7]

Decoder

[Figure 8]

🧊

Decoder VIT

Semantic Encoder

…

Stage 1

DCTTransform

DCTTransform

InputImage

[Figure 9]

Inverse

Spatial Feature

Frequency Reorder

Semanticwise Loss

Frequency Masked Prediction

INIT.

INIT.

[Figure 10]

[Figure 11]

🔥

[Figure 12]

🔥

VIT

Semantic Stage 2 Encoder

…

Higher Frequency

Pixel-wise Reconstruction Loss

- Figure 4. Overall architecture of our proposed Unified Autoencoding (UAE). The input image is separately encoded by both a pretrained Semantic Encoder (e.g., DINOv2) and the trainable Unified Encoder. The unified encoder is initialized from the semantic encoder and optimized under two complementary objectives: a semantic-wise loss that aligns low-frequency components decomposed from the semantic encoder’s representations, and a pixel-wise reconstruction loss that enforces visual fidelity via the Pixel Decoder by adaptively dilating the high-frequency components. The decoder employs spectral transform blocks to refine residual-frequency content and produce the reconstructed image. This joint optimization harmonizes semantic structure and pixel detail within a single latent space.

frequency localization of cross-modal semantics, we evaluate text–image retrieval under progressive removal of highfrequency image components. Concretely, we apply a radial low-pass mask in the frequency domain, with a cutoff specified as a fraction of the Nyquist frequency, reconstruct the filtered images, and measure Recall@1 using frozen vision–language models (CLIP and SigLIP2) with fixed text prompts (an image of ...). Fig. 3 (right) shows that Recall@1 increases rapidly when only a small low-frequency portion is retained (e.g., ∼0.23 at 0.05 Nyquist vs. ∼0.58 at 0.10), and saturates near full-image performance by moderate cutoffs (around 0.3–0.4 Nyquist). This behavior indicates that a significant fraction of retrieval-relevant semantics is encoded in coarse, low-frequency image structure, whereas higher-frequency components are largely unrelated to semantic information.

tering is performed in linear space prior to any modelspecific normalization. With a frozen vision-language encoder E, we compute cosine similarities between text embeddings and image embeddings from I, IρLP, and IρHP. If semantic alignment is carried by the shared base, then the retrieval score R satisfies RLP(ρ) is nondecreasing in ρ, RHP(ρ) is nonincreasing in ρ, and RHP(ρ) approaches chance once the shared base is removed.

Empirical Verification. We empirically examine the Prism Hypothesis using two complementary analyses: (i) tokenspace frequency energy decomposition and (ii) robustness of text–image alignment under controlled spectral filtering.

- Exp1 (Token-spectrum energy). For each encoder, we

extract patch tokens from the input images, reshape them into a 2D token grid, and apply a channel-wise 2D DCT to obtain token-frequency coefficients. We then reorder the coefficients via zig-zag traversal (progressing from low to high frequencies), partition them into coarse bands, and compute the normalized band energy e(k) as the average of squared magnitudes across channels and images. As shown in Fig. 2 (left), the resulting spectra are dominated by the lowest band, revealing a strong low-frequency bias across models (which is consistent with the natural images). Importantly, different encoders allocate different relative mass to higher bands: Semantic representation encoders concentrate more energy in low-frequency components, while the pixel-centric SD-VAE preserves relatively more energy in the mid- and high-frequency bands (inset), consistent with its stronger retention of fine-grained appearance details.

- Exp2 (Low-pass retrieval). To directly investigate the

Taken together, Exp1 and Exp2 support the central intuition of the Prism Hypothesis: cross-modal semantic alignment is primarily associated with a shared low-frequency base, whereas higher-frequency bands increasingly encode modality-specific, fine-grained visual detail.

#### 3.2. Unified AutoEncoder

Using DINOv2 as an example, we initialize UAE from the pretrained encoder and discard the register tokens, retaining only the patch tokens with channel dimension C. Input images are center-cropped and resized to the encoder’s native input resolution. The resulting patch-token sequence is then reshaped into a 2D latent grid z ∈ RB×C×H×W, which enables explicit frequency-domain processing.

###### Frequency transform and tokenization. Given z, we ap-

##### Frequency-wise Modeling Spatial-wise Modeling

… …

… …

[Figure 13]

Frequency Progressive

Inverse DCT Transform

Diffusion Transformer

Diffusion Transformer

Spatial Feature

…

Zero-padding

… …

… …

Diffusion Transformer

[Figure 14]

[Figure 15]

[Figure 16]

VIT Decoder

VIT Decoder

VIT Decoder

- Figure 5. Frequency-wise vs. spatial-wise modeling. Left: The diffusion transformer operates directly in the DCT domain and progressively models frequency tokens from low to high bands before decoding. Right: Frequency tokens are first transformed back to spatial features via inverse DCT, and diffusion is performed in the spatial domain prior to decoding.

Semantic Regularization. To preserve the semantic priors of the pretrained teacher while expanding the representational bandwidth, we apply a semantic-wise alignment loss only on the lowest-frequency bands. Let fs denote features from the frozen pretrained semantic encoder (teacher), and fu denote features from the trainable unified encoder (student). After frequency decomposition, we obtain band-wise representations {fsk}Kk=0−1 and {fuk}Kk=0−1. We enforce alignment only for the first Kbase low-frequency bands, which primarily encode global structure and category-level semantics, while leaving higher frequency less constrained:

ply a channel-wise 2D discrete cosine transform (DCT) f(·) to obtain a frequency-domain representation

h = f(z), h ∈ RB×C×H×W. (3)

We then linearize the 2D DCT grid into a 1D token sequence via a frequency-preserving reordering scheme.

Token Reorder. In the 2D DCT grid, each coefficient C(u,v) (not to be confused with the channel dimension C) corresponds to a spatial frequency indexed by (u,v). A natural measure of frequency magnitude is the radial norm

###### r(u,v) = u2 + v2. (4)

Kbase−1 k=0 ∥fuk − fsk∥22

. (6)

Lsem =

As (u,v) moves away from the top-left DC component (0,0), r(u,v) increases in discrete levels, yielding progressively higher spatial frequencies. This allows us to interpret the DCT lattice as a set of concentric “radial shells” centered at (0,0).

Kbase

This restricted supervision transfers the teacher’s semantic organization to the low-frequency subspace, while leaving higher-frequency bands unconstrained so they can specialize in modality-specific, pixel-level detail. Empirically, this selective regularization stabilizes training and mitigates collapse toward purely pixel-driven features.

To convert the 2D frequency grid into a 1D token sequence while approximately preserving low-to-high frequency progression, we adopt zig-zag reordering. Zig-zag traversal roughly sorts coefficients by increasing u + v, which serves as a practical proxy for radial growth (since u2+v2 typically increases with u+v for small-to-moderate indices). This produces an ordered sequence

Frequency Masked Prediction. During training, we randomly mask a subset of high-frequency tokens to encourage robust inference of missing fine details. Let L and H denote the index sets of low- and high-frequency components, respectively. We construct a masked frequency representation F˜ as

{Ck}Nk=0−1, N = H × W, (5) where the frequency magnitude increases approximately monotonically with k.

Fi, i ∈ L, 0, i ∈ Hmask,

F˜i =

(7)

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

UAE(CLIP)Source(DINOv2) UAE

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

FLUX-VAE RAE

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

(DINOv2) UAE

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

[Figure 51]

[Figure 52]

(SIGLIP2)

- Figure 6. Qualitative comparison of reconstruction fidelity across autoencoding paradigms. We visualize reconstructed samples from representative methods, including Flux-VAE [18], RAE [50], and our proposed UAE. Each row corresponds to reconstructions from a fixed source set spanning text, human, object, and artistic domains. UAE produces the most consistent and semantically faithful reconstructions, preserving both high-frequency details (e.g., texture and edge sharpness) and global structure (e.g., layout and color harmony), while reducing the blurring and semantic drift observed in RAE.

where Hmask ⊆ H is sampled independently at each iteration. That is, selected high-frequency coefficients are set to zero, while low-frequency components remain intact.

The masked frequency tokens are mapped back to the spatial domain via inverse DCT and fed to the decoder. The decoder is trained to reconstruct the full high-fidelity image, including the masked high-frequency details, conditioned only on the preserved low-frequency structure and the remaining visible coefficients. This objective encourages the

model to infer fine-grained textures and high-frequency signals from global semantic cues, improving robustness and generalization.

#### 3.3. Generative Modeling

As illustrated in Figure 5, UAE latents admit two complementary parameterizations for generative modeling: spatial-domain features and frequency-domain coefficients. In spatial-wise modeling, we first apply the inverse DCT to

- Table 1. Reconstruction performance on ImageNet-1K and MS-COCO 2017. UAE consistently achieves strong PSNR/SSIM and competitive rFID compared with both continuous autoencoders and unified tokenizers. We additionally report linear probing accuracy to evaluate representation quality.

Method Type Ratio Linear prob (Acc)↑ ImageNet-1K MS-COCO 2017 PSNR↑ SSIM↑ rFID↓ PSNR↑ SSIM↑ rFID↓

- SD-VAE-1.x [33] Continuous 8 – 23.54 0.68 1.22 23.21 0.69 5.94 SD-VAE [33] Continuous 8 – 25.68 0.72 0.75 25.43 0.73 0.76
- SD-VAE-2.x [33] Continuous 8 – 23.54 0.68 1.22 26.62 0.77 4.26 SD-VAE-XL [28] Continuous 8 – 27.37 0.78 0.67 27.08 0.80 3.93 SD-VAE-3 [20] Continuous 8 – 31.29 0.87 0.20 31.18 0.89 1.67 FLUX-VAE [18] Continuous 8 – 32.74 0.92 0.18 32.32 0.93 1.35 VA-VAE Continuous 16 – 27.96 0.79 0.28 27.50 0.81 2.71

SVG (DINOv3) [34] Continuous 16 – 24.25 0.67 0.78 - - RAE (DINOv2-B) [50] Continuous 14 – 18.05 0.5 2.04 18.36 0.47 6.01 UniFlow (DINOv2-L) [49] Continuous 14 – 32.32 0.91 0.17 30.66 0.94 2.81 UAE (Siglip2) Continuous 16 80.1 (81.2) 31.00 0.91 0.43 30.20 0.89 2.91 UAE (DINOv2-B) Continuous 14 83.0 (83.0) 32.17 0.92 0.35 31.19 0.91 2.01 UAE (CLIP-L) Continuous 14 77.2 (79.4) 36.58 0.96 0.04 36.25 0.97 0.41

map the frequency tokens back to a spatial latent grid, and then train a standard latent diffusion transformer on these spatial features, following prior representation-based tokenizers [50]. In frequency-wise modeling, we operate directly on the reordered DCT coefficients. Since our decoder can reconstruct high-quality images even when conditioned only on low-frequency tokens, we can adopt a coarse-tofine generation strategy: the diffusion transformer is trained to model low-frequency components first and is then progressively extended to incorporate higher-frequency bands, yielding increasingly fine-grained details.

### 4. Experiments

#### 4.1. Implementation Details

UAE Training. We train UAE with DINOv2-B, DINOv2L, SigLIP2, and InternViT at 256 × 256 resolution. Training is performed in two stages. In Stage 1, we freeze the pretrained semantic encoder and train the decoder using a pixel-wise reconstruction loss and an adversarial loss, following the original RAE setting [50]. We use AdamW with an initial learning rate of 2 × 10−4, linearly decayed to 2×10−5, and train for 16 epochs. The discriminator is optimized with AdamW using the same learning-rate schedule; discriminator training starts at epoch 6, and the GAN loss term on the generator is enabled from epoch 8. In Stage 2, we unfreeze the encoder and fine-tune the full model using the semantic-wise loss together with the reconstruction loss. We additionally apply frequency-masked prediction by randomly masking high-frequency components and training the model to reconstruct the full image under the GAN objective. The masking ratio is set to 75% for DINOv2 and 50% for the other encoders. We set Kbase to 25%

of the total tokens for DINOv2 and 50% for the remaining models.Unless otherwise specified, Stage 2 uses the same optimizer and learning-rate schedule as Stage 1, with highfrequency masking enabled from the beginning of Stage 2.

Generative Modeling. We train SiT on two UAE latent parameterizations: spatial latents (768 × 16 × 16) and frequency latents (768 × 256). In both cases, training is performed progressively over frequency tokens: we start from 64 tokens for the first 80 epochs, then linearly increase the token count to 256, updating the token budget every 20 epochs. All models are trained for a total of 1400 epochs to ensure stable convergence. We adopt the AdamW optimizer with an initial learning rate of 2 × 10−4, which is linearly decayed to 2 × 10−5 over the course of training.

#### 4.2. Visual Reconstruction

Quantitative Evaluation. In Table 1, We quantify reconstruction quality at 256 × 256 on ImageNet-1K and MSCOCO 2017. The compared baselines include widely-used generative tokenizers and variational decoders. For a fair comparison, we report the evaluation results for our UAE and the corresponding baseline that follows the same configurations of DINOv2-base. Here, we report PSNR and SSIM for fidelity, and rFID for perceptual quality.

Notably, our UAE delivers state-of-the-art reconstruction quality among unified tokenizers. On ImageNet-1K, UAE improves over the RAE baseline from 18.05 to 31.00 in PSNR and from 0.50 to 0.92 in SSIM, while reducing FID from 2.04 to 0.35. On MS-COCO, UAE raises PSNR from 18.36 to 31.19 and SSIM from 0.47 to 0.91, with FID decreasing from 6.01 to 2.01. These notable gains validate that moving masking out of the decoder and factorizing latents into a low-frequency base and residual bands preserve

- Table 2. Class-conditional generation performance on ImageNet (256x256). We compare our proposed UAE with recent diffusion and autoregressive models using standard metrics. Note that the UAE performs generation in a causal manner, progressing from low- to high-frequency bands in the latent space.

Methods gFID↓ IS↑ Prec↑ Rec↑ DiT [27] 9.62 121.5 0.67 0.67 SiT [23] 8.61 131.7 0.68 0.67 SVG [34] 3.36 181.2 - UniFlow [49] 2.45 228.0 - RAE [50] 1.51 242.9 0.79 0.63 UAE 1.52 234.5 0.81 0.62

fine detail while maintaining semantic structure.

Beyond unified tokenizers, UAE is competitive with the best generative-only tokenizers. On ImageNet-1K, UAE(CLIP-L) attains an FID of 0.04, surpassing SD3 VAE and approaching Flux VAE, while maintaining high PSNR and SSIM. A similar pattern holds on MS-COCO, where UAE(CLIP-L) surpasses the strongest baselines in perceptual quality.

Qualitative Evaluation. Figure 6 compares source images with the resulting reconstructions from FLUX-VAE, RAE, and UAE. Note that our UAE can well preserve straight edges, fine textures, and small text, e.g., street signs and printed documents. The improvement is consistent across natural photos and illustrations. Moreover, we demonstrate that UAE is effective across different semantic encoders, including DINOv2 [26], CLIP [30], and SigLip2 [40].

#### 4.3. Generative Modeling

To further assess the effectiveness of the proposed UAE, we conduct class-conditional image generation experiments on ImageNet-1K at a resolution of 256 × 256.

We conduct all generative experiments in the multi-band latent space. All experimental settings follow those used in RAE [50] to ensure a fair comparison. As shown in Table 2, our UAE attains a gFID of 1.52 and an IS of 234.5, achieving performance on par with existing state-of-the-art generative models. This suggests that the unified frequencybased representation enables the generative model to progressively capture both global structure and fine-grained details in a coherent manner, keeping highly generative while preserving strong semantic quality. Overall, the UAE latent space, constructed through explicit low- and high-frequency decomposition, provides an effective and diffusion-friendly foundation for large-scale visual generation.

SiT RAE

UAE [spatial] UAE [freq-64tokens]

| |
|---|

4.3

40

4.2

4.1

4.0

80

FID

20

8

4

20 40 60 80 Training Epochs

Figure 7. UAE (spatial and frequency) converges significantly faster and achieves lower FID than SiT and RAE. The zoom-in highlights the final performance gap at 80 epochs.

Table 3. Efficiency comparison across different representations. UAE maintains comparable computational cost to RAE in both spatial and frequency variants, while enabling significant efficiency gains when integrated with SiT. In particular, SiT (UAEfreq-64tokens) achieves a 4× reduction in GFLOPs and over 2× lower latency compared to the RAE-based baseline.

###### Model GFLOPs Latency

RAE 266.24 8.40 UAE-spatial 266.24 8.40 UAE-freq 266.27 8.76

SiT (RAE) 313.84 8.82 SiT (UAE-freq-64tokens) 78.80 4.05

### 5. Discussion

#### 5.1. The Impact of Generative Modeling

We further study the effect of generative modeling using UAE representations. As shown in Fig. 7, UAE significantly accelerates diffusion model convergence compared with SiT and RAE. Both spatial and frequency variants consistently achieve lower FID throughout training and reach high-quality generation within substantially fewer epochs. In particular, UAE attains strong performance as early as 40 epochs and continues to improve steadily, while baseline methods converge much more slowly. The zoom-in view highlights the final performance gap at 80 epochs, where UAE achieves the best FID. These results suggest that the frequency-aware representation of UAE provides a more generation-friendly latent space, enabling more efficient generative modeling.

#### 5.2. Inference Efficiency

We compare the computational efficiency of different tokenizer and latent modeling configurations in Table 3. The

Table 4. Ablations on semantic loss and mask prediction.

(b) Ratio gFID@20 epoch↓ 85% 9.7 15% 12.2 50% 27.2 25% 33.1

###### (a)

Kbase PSNR↑ ACC↑

0% 34.3 60.1 25% 32.2 83.0 50% 28.1 83.0 75% 20.1 83.0

proposed UAE tokenizer introduces negligible overhead in stage-1 reconstruction, exhibiting nearly identical GFLOPs and latency to the RAE baseline. In contrast, the efficiency gain becomes significant in stage-2 generative modeling. When the SiT backbone operates on the compact UAE 64 low-freq tokens, the computational cost is reduced from 313.84 GFLOPs to 78.80 GFLOPs per image, resulting in a ∼4× reduction, while inference latency decreases from 8.82 ms to 4.05 ms per image.

#### 5.3. The Impact of Semantic Loss

We ablate the strength of semantic regularization by varying Kbase as shown in the Table 4 (a), the fraction of lowfrequency tokens constrained by the semantic alignment loss using DINOv2 as the teacher. When semantic regularization is disabled (Kbase = 0), the model achieves strong reconstruction but exhibits weak semantic alignment, as reconstruction gradients favor local appearance cues. Increasing Kbase improves semantic performance by anchoring the low-frequency latent space to the pretrained semantic geometry, but excessive constraints reduce reconstruction fidelity. We find that aligning only 25% of frequency tokens already preserves most semantic capability while maintaining high reconstruction quality, supporting the design that semantics mainly reside in the low-frequency base while higher frequencies capture fine details.

#### 5.4. The Impact of Mask Predictions

We further study the effect of frequency-masked prediction during UAE training by varying the masking ratio applied to high-frequency tokens. This strategy serves as a regularizer by forcing the decoder to reconstruct missing highfrequency details from the preserved low-frequency structure and remaining visible components. As shown in Table 4 (b), aggresive masking yields the best generative performance: masking 85% of high-frequency tokens achieves the lowest gFID@20 epochs (9.7). When the masking ratio is too small, the regularization effect is limited and the model tends to rely on deterministic high-frequency cues. In contrast, larger masking ratios remove excessive highfrequency information and significantly improve the converge speed. These results suggest that strong masking encourages stronger structural priors in the latent space and

DINOv2

CLIP-L

UAE[DINOv2]

UAE[CLIP-L]

0.6

0.07

0.5

0.06

0.4

Energy()ek

0.05

0.3

6 7 8 9

0.2

0.1

0.0

0 2 4 6 8

Frequency band k

Figure 8. Energy distribution across frequency bands before and after unified training. Unified training consistently shifts energy from low-frequency components to mid- and high-frequency bands. The zoom-in (right) highlights that UAE preserves or enhances high-frequency energy, indicating improved fine-grained detail representation, while only slightly reducing low-frequency dominance associated with global semantics.

Table 5. Comparison of generative performance in pixel space. We evaluate JIT and its unified-trained variant UAE (JIT) under the same architecture, model size, and patch size. UAE consistently improves both FID (↓) and IS (↑) across training stages.

Model Epoch Size Patch Size FID↓ IS↑

JIT [19] 40 700M 32 31.81 60.13 JIT [19] 80 700M 32 16.06 123.30

UAE (JIT) 40 700M 32 25.80 72.82 UAE (JIT) 80 700M 32 14.55 132.57

leads to more effective generative modeling.

#### 5.5. The Impact of Unified Training

As shown in Fig. 8, we analyze the frequency energy distribution of vanilla encoders and their unified-trained counterparts under the same setting described in Sec. 3.1. Unified training induces a consistent redistribution of energy across frequency bands. Specifically, the UAE models exhibit reduced dominance in low-frequency components, accompanied by a noticeable increase in mid- and high-frequency energy. This shift suggests that unified training enhances the representation of fine-grained details while largely preserving global semantic structure. The zoom-in view further confirms that high-frequency components are better retained or even amplified after unification.

#### 5.6. UAE in Pixel Space

In latent space, we observe that a pretrained semantic encoder concentrates most of the useful information in lowfrequency components, while fine-grained image details are

Table 6. (a). Latent space generative modeling converge speed with different low-freq tokens. (b). Linear-probing accuracy with models trained from scratch in different low-freq tokens.

(a) Tokens Epochs FID↓ 64 20 9.7 81 20 12.2 100 20 20.6 121 20 40.5

###### (b)

Tokens Accuracy↑

64 60.1 81 33.2 100 28.1 121 20.7

primarily encoded in high-frequency tokens. As shown in Table 6(a), we conduct generative modeling while retaining different numbers of low-frequency tokens. The results demonstrate that generative performance strongly favors low-frequency tokens (i.e., semantic components), which is consistent with recent findings [1].

In Table 6(b), we further train an encoder–decoder model from scratch while preserving only N low-frequency tokens, and perform linear probing on the learned representations. The results indicate that semantic information naturally emerges during the compression and reconstruction process.

These observations suggest that reconstruction and generation are intrinsically connected through the extent to which low-frequency tokens are preserved. From this perspective, pixel-space modeling and latent-space modeling are fundamentally equivalent, differing only in compression ratio and the specific compression architecture (e.g., SDVAE with 8× compression is analogous to JIT with patch size 8). This insight allows us to directly extend UAE to pixel-space modeling.

Therefore, We further evaluate the effectiveness of unified training in the pixel-space generative setting by applying UAE to the JIT framework. As shown in Table 5, we compare the original JIT model with its unified-trained counterpart under identical configurations. UAE consistently outperforms the vanilla JIT across all metrics and training stages. UAE achieves a notable improvement, consistently achieve lower fid score and high IS score. These results indicate that unified training extends beyond latent space and remains effective when applied directly in pixel space.

### 6. Conclusion

We propose the Prism Hypothesis, which views diverse natural inputs as projections of a shared spectrum composed of a compact low-frequency semantic component and residual higher-frequency detail. Preliminary experiments about frequency-band distributions of various visual and textual encoders strongly enlighten this connection between feature spectra and representational function. Hence, we launch

Unified AutoEncoding (UAE), which harmonizes semantic and pixel information within a single latent space via a hierarchical frequency-band modulator. Extensive experiments confirm that it delivers greater generative capability over existing unified tokenizers, e.g., RAE, SVG, and UniFlow, meanwhile preserving reconstruction quality on par with top-tier models like Flux-VAE. We further validate that the idea of UAE not only benefit in the latent space but also remains effective in pixel space. We position the UAE as a promising and practical route towards unified tokenizers for understanding and generation. .

### References

- [1] Alan Baade, Eric Ryan Chan, Kyle Sargent, Changan Chen, Justin Johnson, Ehsan Adeli, and Li Fei-Fei. Latent forcing: Reordering the diffusion trajectory for pixel-space image generation. arXiv preprint arXiv:2602.11401, 2026. 10
- [2] Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. Data2vec: A general framework for self-supervised learning in speech, vision and language. In International conference on machine learning, pages 1298–1312. PMLR, 2022. 2
- [3] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 1
- [4] Jun Chen, Deyao Zhu, Guocheng Qian, Bernard Ghanem, Zhicheng Yan, Chenchen Zhu, Fanyi Xiao, Sean Chang Culatana, and Mohamed Elhoseiny. Exploring open-vocabulary semantic segmentation from clip vision encoder distillation only. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 699–710, 2023. 2
- [5] Emily L Denton, Soumith Chintala, Rob Fergus, et al. Deep generative image models using a laplacian pyramid of adversarial networks. Advances in neural information processing systems, 28, 2015. 3
- [6] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 2
- [7] Wan-Cyuan Fan, Yen-Chun Chen, DongDong Chen, Yu Cheng, Lu Yuan, and Yu-Chiang Frank Wang. Frido: Feature pyramid diffusion for complex scene image synthesis. In Proceedings of the AAAI conference on artificial intelligence, pages 579–587, 2023. 3
- [8] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15180–15190, 2023. 2
- [9] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022. 3

- [10] Runhui Huang, Chunwei Wang, Junwei Yang, Guansong Lu, Yunlong Yuan, Jianhua Han, Lu Hou, Wei Zhang, Lanqing Hong, Hengshuang Zhao, et al. Illume+: Illuminating unified mllm with dual visual tokenization and diffusion refinement. arXiv preprint arXiv:2504.01934, 2025. 1
- [11] Wei Huang, Zhiliang Peng, Li Dong, Furu Wei, Jianbin Jiao, and Qixiang Ye. Generic-to-specific distillation of masked autoencoders. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15996– 16005, 2023. 2
- [12] Zhihao Huang, Xi Qiu, Yukuo Ma, Yifu Zhou, Junjie Chen, Hongyuan Zhang, Chi Zhang, and Xuelong Li. Nfig: Multiscale autoregressive image generation via frequency ordering. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 3
- [13] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, et al. Perceiver io: A general architecture for structured inputs & outputs. arXiv preprint arXiv:2107.14795, 2021. 2
- [14] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR,

2021. 2

- [15] Liming Jiang, Bo Dai, Wayne Wu, and Chen Change Loy. Focal frequency loss for image reconstruction and synthesis. In Proceedings of the IEEE/CVF international conference on computer vision, pages 13919–13929, 2021. 3
- [16] Tero Karras, Miika Aittala, Samuli Laine, Erik H¨ark¨onen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. Advances in neural information processing systems, 34:852–863, 2021. 3
- [17] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 1
- [18] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742,

2025. 6, 7

- [19] Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720, 2025. 2, 9
- [20] Joshua Lopez. Stable diffusion 3: Research paper. https: //stability.ai/news/stable-diffusion-3research-paper, 2025. Stability AI. 7
- [21] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. arXiv preprint arXiv:2206.08916, 2022. 2
- [22] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26439–26455, 2024. 2

- [23] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024. 8
- [24] Duy-Kien Nguyen, Mahmoud Assran, Unnat Jain, Martin R Oswald, Cees GM Snoek, and Xinlei Chen. An image is worth more than 16x16 patches: Exploring transformers on individual pixels. arXiv preprint arXiv:2406.09415, 2024. 1
- [25] Mang Ning, Mingxiao Li, Jianlin Su, Haozhe Jia, Lanmiao Liu, Martin Beneˇs, Albert Ali Salah, and Itir Onal Ertugrul. Dctdiff: Intriguing properties of image generative modeling in the dct space. arXiv preprint arXiv:2412.15032, 2024. 3
- [26] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 1, 8
- [27] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 8

- [28] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 7
- [29] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2545–2555, 2025. 1
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 1, 2, 8
- [31] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International conference on machine learning, pages 5301–

5310. PMLR, 2019. 3

- [32] Ali Razavi, Aaron Van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019. 1
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 7
- [34] Minglei Shi, Haolin Wang, Wenzhao Zheng, Ziyang Yuan, Xiaoshi Wu, Xintao Wang, Pengfei Wan, Jie Zhou, and Jiwen Lu. Latent diffusion model without variational autoencoder. arXiv preprint arXiv:2510.15301, 2025. 2, 7, 8
- [35] Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representa-

- tions with periodic activation functions. Advances in neural information processing systems, 33:7462–7473, 2020. 3
- [36] Ivan Skorokhodov, Willi Menapace, Aliaksandr Siarohin, and Sergey Tulyakov. Hierarchical patch diffusion models for high-resolution video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7569–7579, 2024. 3
- [37] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural information processing systems, 33:7537–7547, 2020. 3
- [38] Hao Tang, Chenwei Xie, Xiaoyi Bao, Tingyu Weng, Pandeng Li, Yun Zheng, and Liwei Wang. Unilip: Adapting clip for unified multimodal understanding, generation and editing. arXiv preprint arXiv:2507.23278, 2025. 1
- [39] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024. 3
- [40] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 8
- [41] Junke Wang, Yi Jiang, Zehuan Yuan, Bingyue Peng, Zuxuan Wu, and Yu-Gang Jiang. Omnitokenizer: A joint imagevideo tokenizer for visual generation. Advances in Neural Information Processing Systems, 37:28281–28295, 2024. 2
- [42] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for vision and visionlanguage tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19175– 19186, 2023. 2
- [43] Yikai Wang, Zhouxia Wang, Zhonghua Wu, Qingyi Tao, Kang Liao, and Chen Change Loy. Next visual granularity generation. arXiv preprint arXiv:2508.12811, 2025. 3
- [44] Ji-Jia Wu, Andy Chia-Hao Chang, Chieh-Yu Chuang, Chun-Pei Chen, Yu-Lun Liu, Min-Hung Chen, Hou-Ning Hu, Yung-Yu Chuang, and Yen-Yu Lin. Image-text codecomposition for text-supervised semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26794–26803, 2024. 2
- [45] Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, and Xiaolong Wang. Groupvit: Semantic segmentation emerges from text supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18134–18144, 2022. 2
- [46] Srikar Yellapragada, Alexandros Graikos, Kostas Triaridis, Prateek Prasanna, Rajarsi Gupta, Joel Saltz, and Dimitris Samaras. Zoomldm: Latent diffusion model for multi-scale image generation. In Proceedings of the Computer Vision

and Pattern Recognition Conference, pages 23453–23463,

2025. 3

- [47] Muyang Yi, Quan Cui, Hao Wu, Cheng Yang, Osamu Yoshie, and Hongtao Lu. A simple framework for textsupervised semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7071–7080, 2023. 2
- [48] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023. 2
- [49] Zhengrong Yue, Haiyu Zhang, Xiangyu Zeng, Boyu Chen, Chenting Wang, Shaobin Zhuang, Lu Dong, KunPeng Du, Yi Wang, Limin Wang, et al. Uniflow: A unified pixel flow tokenizer for visual understanding and generation. arXiv preprint arXiv:2510.10575, 2025. 2, 7, 8
- [50] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025. 2, 6, 7, 8

