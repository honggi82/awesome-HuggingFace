## arXiv:2605.10780v2[cs.CV]12May2026

### Beyond the Last Layer: Multi-Layer Representation Fusion for Visual Tokenization

##### Xuanyu Zhu1,2∗ Yan Bai2∗ Yang Shi1,† Yihang Lou1

Yuanxing Zhang1 Jing Jin3 Yuan Zhou4,‡ 1Peking University 2Meituan Inc 3Tsinghua University 4IGDL https://github.com/zhuzil/DRoRAE

#### Abstract

Representation autoencoders that reuse frozen pretrained vision encoders as visual tokenizers have achieved strong reconstruction and generation quality. However, existing methods universally extract features from only the last encoder layer, discarding the rich hierarchical information distributed across intermediate layers. We show that low-level visual details survive in the last layer merely as attenuated residuals after multiple layers of semantic abstraction, and that explicitly fusing multi-layer features can substantially recover this lost information. We propose DRoRAE (Depth-Routed Representation AutoEncoder), a lightweight fusion module that adaptively aggregates all encoder layers via energy-constrained routing and incremental correction, producing an enriched latent compatible with a frozen pretrained decoder. A three-phase decoupled training strategy first learns the fusion under the implicit distributional constraint of the frozen decoder, then finetunes the decoder to fully exploit the enriched representation. On ImageNet-256, DRoRAE reduces rFID from 0.57 to 0.29 and improves generation FID from 1.74 to 1.65 (with AutoGuidance), with gains also transferring to text-to-image synthesis. Furthermore, we uncover a log-linear scaling law (R2=0.86) between fusion capacity and reconstruction quality, identifying representation richness as a new, predictably scalable dimension for visual tokenizers analogous to vocabulary size in NLP.

#### 1 Introduction

The image tokenizer maps pixels into a compact latent space and defines the quality ceiling of modern visual generation systems [21, 18]. A recent line of work [30, 29, 33, 10] has demonstrated that leveraging pretrained vision foundation models (VFMs) such as DINOv2 [17] as the tokenizer’s latent space yields substantial improvements in both reconstruction fidelity and downstream generation quality over conventional learned tokenizers trained from scratch.

Despite their success, all existing VFM-based tokenizers share a common design choice: they extract features exclusively from the last layer of the encoder. While this is the natural output of any vision model, last-layer features are primarily optimized for high-level semantics rather than low-level visual details such as textures, edges, and color gradients. Recent analysis [25] reveals that low-level information survives in the last layer only as a structural consequence of residual connections, a passive pathway that becomes increasingly lossy as each successive layer superimposes semantic transformations onto the residual stream. Shallower layers, by contrast, retain this information with far greater fidelity (Figure 1), yet single-layer tokenizers discard it entirely.

∗Equal contribution. †Project Leader ‡Corresponding author

Preprint.

###### (a) Single-Layer Bottleneck (b) Multi-Layer Fusion

Low-level detail

Router

###### Fronzen Encoder

Semantic content

Routing weights

Discard information

discarded

Shallow

Expert1

1.0

information not used

###### discarded

Tokens

Middle

Expertj

[Figure 1]

- 1.
- 2.

0.5

풇풊   

...

...

풃   

0.0

1 ... j ... L

Deep

ExpertL

Experts

Decoder

Decoder

[Figure 2]

[Figure 3]

- 1. 2.

[Figure 4]

High Weight Low Weight

Texture from shallow (low-level details)

Semantics from deep (high-level concepts)

Shallow texture + Deep semantics = Complete representation

- Figure 1: Motivation. Existing representation autoencoders extract only last-layer features, where low-level details are progressively diluted by semantic transformations. DRoRAE fuses features across all layers to assemble a richer latent per spatial token.

This observation suggests a natural direction: explicitly fusing features from multiple depth levels to assemble a latent representation richer than any single layer can provide. Moreover, multi-layer fusion introduces two quantifiable capacity axes, the number of fused layers and the per-layer expert capacity, which together define the representation richness of the tokenizer. An analogous concept has been explored for NLP tokenizers [12], where increasing the input vocabulary size (representation richness in the text domain) yields predictable, log-linear improvements in downstream loss. Whether such a scaling law also exists for visual tokenizers remains an open question.

Realizing multi-layer fusion in practice, however, requires addressing two challenges. (1) Contentadaptive fusion. Feature statistics vary substantially across layers, and the optimal combination is spatially dependent: textured regions benefit from shallow features while semantically uniform regions do not. Naive aggregation collapses to deep-layer dominance or introduces noise from irrelevant layers. (2) Generation compatibility. In representation-based tokenizers, the decoder is trained to invert a specific output distribution. Multi-layer fusion inevitably shifts this distribution; if unconstrained, the downstream diffusion model can no longer generate latents that the decoder reliably decodes, degrading generation even when reconstruction improves.

We propose DRoRAE (Depth-Routed Representation AutoEncoder), a lightweight fusion module of ∼29M parameters (Figure 2) that addresses both challenges. For content-adaptive fusion, we design an energy-constrained routing mechanism. Per-layer expert MLPs project heterogeneous layer features onto a common scale, and a learned router assigns per-token aggregation weights, including negative weights for active suppression, without the winner-take-all behavior of softmax normalization. For generation compatibility, we adopt an incremental correction formulation that injects the fused representation as a bounded perturbation to the original last-layer output. This is combined with a three-phase decoupled training strategy in which the fusion module first learns under the implicit distributional constraint of a frozen decoder, preventing arbitrary drift, and only then is the decoder fine-tuned to fully exploit the enriched latent.

On ImageNet-256, DRoRAE reduces rFID from 0.57 to 0.29 and improves class-conditional generation (gFID with AutoGuidance: 1.74→1.65), with gains also transferring to text-to-image synthesis. We further observe that reconstruction quality improves log-linearly with fusion module capacity (R2=0.86), confirming that an analogous scaling law holds for visual tokenizers: representation richness, jointly determined by the number of fused layers and the per-layer expert capacity, is a predictably scalable dimension paralleling vocabulary size in NLP [12].

Our contributions are as follows:

- • We identify the single-layer information bottleneck in representation autoencoders and propose DRoRAE, a depth-routed fusion module that enriches the tokenizer latent while preserving generation compatibility through energy-constrained routing, incremental correction, and decoupled training.

- • DRoRAE consistently improves reconstruction (rFID: 0.57→0.29), class-conditional generation (gFID w/ AG: 1.74→1.65), and text-to-image synthesis on ImageNet-256, validating multi-layer fusion as a practical upgrade for representation-based tokenizers.
- • We conduct systematic scaling experiments across two axes, expert capacity and number of fused layers, and observe that both follow the same log-linear scaling law. This establishes representation richness as a new, predictably scalable dimension for visual tokenizers.

#### 2 Related Work

##### 2.1 Image Tokenizers for Latent Generation

The unified model explores the relationship between understanding and generation. Image tokenizers compress images into compact latent representations on which generative models operate. Early approaches learn both the encoder and decoder from scratch. VQGAN [8] combines discrete codebooks with adversarial training; SD-VAE [21] employs a KL-regularized continuous latent space and has become the backbone tokenizer for latent diffusion models [18, 16]. While these learned tokenizers achieve reasonable reconstruction, their latent spaces lack explicit semantic structure, forcing the downstream diffusion model to jointly discover both visual and semantic patterns from pixel-level supervision alone.

A recent line of work addresses this by aligning the latent space to pretrained visual representations. REPA [30] adds a representation alignment loss during diffusion training while retaining the original SD-VAE encoder. VA-VAE [29] distills DINOv2 [17] features into a learned VAE encoder, obtaining a latent space that is both reconstructive and semantically structured. RAE [33] takes this idea further by directly freezing the pretrained DINOv2 encoder as the tokenizer and training only a decoder, so that the latent space is the pretrained representation itself. RPiAE [10] extends RAE with a principal-component-based channel expansion to decouple spatial and channel information. These representation-based tokenizers simultaneously achieve state-of-the-art reconstruction fidelity and downstream generation quality, demonstrating that the latent space structure inherited from pretrained models substantially benefits generative modeling.

However, all existing representation-based tokenizers share an inherited design choice. They extract features exclusively from the final layer of the pretrained encoder. Different layers of a Vision Transformer encode different information, ranging from fine-grained textures and edges in shallow layers to high-level semantics in deep layers [19, 1]. This single-layer bottleneck therefore systematically discards hierarchical visual information beneficial to both reconstruction and generation.

##### 2.2 Multi-Layer Feature Utilization in Vision Models

The complementary nature of features at different depths is well established in visual understanding. Feature Pyramid Networks [15], Dense Prediction Transformers [20], and hypercolumns [11] all aggregate multi-layer features for dense prediction tasks. Studies on ViT feature properties [19, 1] confirm that shallow layers retain spatial detail progressively abstracted away in deeper layers, and that the final-layer output preserves low-level information primarily through passive residual leakage [25]. In multimodal large language models (MLLMs) [23, 32, 24, 26], Dense Connector [28], MMFuser [2], and Instruction-Guided Fusion [14] have further demonstrated that fusing multi-layer ViT features improves fine-grained visual understanding [13] [4]. However, all these methods operate in discriminative settings (detection, segmentation, or vision-language understanding); whether multi-layer fusion benefits generative image tokenization remains unexplored.

Despite this rich body of evidence, multi-layer feature fusion has been almost entirely unexplored in the context of image tokenization for generation. Existing tokenizers, both learned [8, 21] and representation-based [33, 10], use a single encoder output without leveraging the hierarchical structure. This leaves open two questions that we address in this work: (1) can explicit multi-layer fusion improve the reconstruction quality of representation autoencoders beyond the residual leakage ceiling? and (2) do these reconstruction improvements consistently transfer to downstream generation quality across different generation paradigms (class-conditional diffusion and text-to-image synthesis)?

# 🔥

[Figure 5]

1) Per-token Depth Routing

2) Energy Weight / Gate

ℎ1 ℎ𝑗 ℎ𝐿

|[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>...<br><br>...<br><br>...<br><br>Layer 1 (shallow)<br><br>Layer j (middle)<br><br>Layer L (deep)<br><br>N<br><br>N<br><br>Expert1<br><br>Expertj<br><br>ExpertL<br><br>ℎ1<br><br>ℎ𝑗<br><br>ℎ𝐿|
|---|

|Layers<br><br>1 2 ... L<br><br>Tokens<br><br>Token-by-Layer Routing Map<br><br>𝑊 ∈ (𝑁2 × 𝐿)<br><br>1<br>2<br><br><br>N×N<br><br>...<br><br>| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
|
|---|

...

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

🧊

[Figure 24]

DINOv2

𝑤1

𝑤𝑗 𝑤𝐿

Tokens

...

...

...

...

Weighted Sum

÷ ∑( ) × ( )

[Figure 25]

𝑍𝑓𝑢𝑠𝑒 ...

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

[Figure 26]

Energy Weight / Gate

🔥

[Figure 27]

Decoder

𝑍𝑓𝑢𝑠𝑒 ...

| | | | |
|---|---|---|---|

🧊

𝑍𝑓𝑖𝑛𝑎𝑙

[Figure 28]

DINOv2

...

𝑍𝑏𝑎𝑠𝑒

| | | | |
|---|---|---|---|

3) Energy-Constrained Aggregation

- Figure 2: Overview of DRoRAE. A frozen DINOv2 backbone extracts multi-layer token features, which are processed by a trainable Depth-Routed Fusion Module. The module first performs per-token depth routing across all backbone layers, then applies energy-constrained aggregation to stabilize the fused tokens and a base-anchored incremental update to preserve the last-layer representation structure. The resulting enriched latent tokens are decoded by a ViT-XL decoder for image reconstruction.
- 3 Method

We present DRoRAE, a lightweight extension to the Representation Autoencoder framework that fuses multi-layer features from a frozen pretrained encoder into an enriched latent representation. Section 3.1 reviews the RAE baseline. Section 3.2 introduces the depth-routed fusion module. Section 3.3 describes the two-phase training strategy.

##### 3.1 Preliminaries

We build upon the Representation Autoencoder (RAE) framework [33], which repurposes a frozen pretrained Vision Transformer E as the image tokenizer and trains only a decoder D. Given an input image x ∈ RH×W×3, the encoder first partitions it into N = (H/p) × (W/p) non-overlapping patches of size p × p, linearly embeds them, and processes the resulting sequence through L transformer layers:

z(l) = TransformerBlock(l)(z(l−1)), l = 1,...,L (1)

where z(0) is the patch embedding and each z(l) ∈ RN×C is the hidden state at layer l. The final latent representation is z = LN(z(L)) ∈ RN×C, where LN is the backbone’s output layer normalization. The decoder reconstructs the image as xˆ = D(z).

In standard RAE, only the final-layer output zbase = LN(z(L)) is used as the latent representation, and all intermediate hidden states z(1),...,z(L−1) are discarded. While zbase is semantically rich, it has lost much of the fine-grained visual information encoded in shallower layers [19, 25]. Our goal is to recover this information through multi-layer fusion, which makes RAE effective.

##### 3.2 Depth-Routed Fusion Module

We introduce a lightweight fusion module F that is inserted between the frozen backbone and the RAE latent space. It takes hidden states from all L layers and the baseline output zbase, and produces an enriched representation zfinal that serves as a drop-in replacement for the original latent.

Layer-wise experts. Each layer k ∈ {1,...,L} is associated with a dedicated expert network ek, implemented as a two-layer MLP. All inputs and outputs are normalized using the backbone’s

Output

Output

Output

🔥

[Figure 29]

🧊

[Figure 30]

🔥

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Decoder

Decoder

Decoder

...

...

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

🧊

[Figure 35]

🔥

[Figure 36]

ℒ𝑟𝑒𝑐, ℒ𝑙𝑝𝑖𝑝𝑠, ℒ𝐺𝐴𝑁

ℒ𝑟𝑒𝑐, ℒ𝑙𝑝𝑖𝑝𝑠, ℒ𝐺𝐴𝑁

ℒ𝑟𝑒𝑐, ℒ𝑙𝑝𝑖𝑝𝑠, ℒ𝐺𝐴𝑁

𝑍𝑓𝑖𝑛𝑎𝑙

𝑍𝑓𝑖𝑛𝑎𝑙

Fusion Module

Fusion Module

[Figure 37]

[Figure 38]

🧊

[Figure 39]

🧊

🧊

[Figure 40]

[Figure 41]

[Figure 42]

DINOv2

DINOv2

DINOv2

Input

Input

Input

Phase 2 Fusion Module Training

Phase 1 Decoder Training

Phase 3 Decoder Fine-Tuning

- Figure 3: Three-phase decoupled training strategy. Phase 1 trains only the decoder. Phase 2 freezes both backbone and decoder, training only the fusion module to learn multi-layer complementary information under the implicit distributional constraint of the frozen decoder. Phase 3 unfreezes the decoder to co-adapt with the enriched fused latent, fully exploiting the richer representation.

own layer normalization LNbb, ensuring that expert outputs remain on the same scale as the original backbone features regardless of the layer-wise variance disparity. Concretely:

###### hk = ek(z(k)), k = 1,...,L (2)

Energy-constrained routing. A router network produces per-token routing weights across all layers. Unlike standard Mixture-of-Experts with softmax normalization, we adopt an energy-constrained formulation that permits negative weights and thus allows the router to actively suppress detrimental layer contributions:

###### w = R [(z(1));...;(z(L))] ∈ RN×L (3)

 

  (4)

L k=1 wk · hk

zfuse = LNbb

L k=1 wk2 + ϵ

where R is a linear projection producing raw logits, wk denotes the routing weight for layer k at each spatial position, and the denominator normalizes by the ℓ2-norm of the weight vector. This bounds the output energy regardless of individual weight magnitudes.

Incremental correction. Rather than replacing zbase with zfuse, we formulate the fusion as an incremental correction:

zfinal = LNbb(zbase + β · (zfuse − zbase)) (5)

where β controls the fusion strength. When β = 0, the module degenerates to the original single-layer RAE. This residual formulation allows the fusion module to focus on learning the complementary information from shallow layers rather than re-learning the already effective deep features.

##### 3.3 Training Strategy

A key challenge in multi-layer fusion for representation autoencoders is maintaining compatibility with the pretrained latent space: the decoder has been trained to invert a specific feature distribution (the backbone’s last-layer output), and modifying this distribution through fusion risks degrading both reconstruction and downstream generation quality. We address this with a decoupled three-phase training strategy (Figure 3) that progressively introduces complexity: first learning a strong decoder, then learning the fusion module under the constraint of the frozen decoder, and finally co-adapting the decoder to the enriched latent. The encoder backbone remains frozen throughout all phases. Detailed hyperparameters are provided in Appendix A.

GT RAE DRoRAE(Ours) GT RAE DRoRAE(Ours)

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

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

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

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

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

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

- Figure 4: Qualitative reconstruction comparison. Selected ImageNet-256 validation images where DRoRAE shows notable improvement over the RAE baseline. DRoRAE better preserves fine-grained textures, structural details, and color fidelity, particularly in regions with repetitive patterns, thin structures, and high-frequency content that the last-layer representation alone tends to lose.

- Phase 1: Decoder training (standard RAE). Following the RAE framework [33], we first train the decoder D with the backbone E frozen and no fusion module present. The decoder learns to reconstruct images from the last-layer representation zbase using the standard training objective:

Ltotal = Lrec + λpLLPIPS + λgαadaptLGAN (6)

where Lrec is the ℓ1 reconstruction loss, LLPIPS is the perceptual loss [31], LGAN is the adversarial loss from a DINO-based discriminator [33], and αadapt is an adaptive weight computed from gradient norms to balance reconstruction and adversarial objectives [8]. This phase establishes a strong decoder that defines the “decoding capacity” of the system, i.e., the best reconstruction achievable from the last-layer representation alone.

- Phase 2: Fusion module training. With both the backbone and the Phase 1 decoder frozen, only the fusion module parameters (∼29M) are optimized. The correction strength β is fixed at 0.2 to encourage conservative corrections. The same reconstruction objective (Eq. 6) is used. The frozen decoder acts as an implicit distributional constraint: the fusion module must produce latents that D already inverts well, preventing arbitrary distribution drift.
- Phase 3: Decoder fine-tuning. With the fusion module frozen at its Phase 2 optimum, we unfreeze

the decoder and continue training with Eq. 6. The decoder adapts to the enriched latent zfinal, improving reconstruction (rFID: 0.47→0.29) without harming generation, because the fused latent distribution has already been stabilized in Phase 2. Joint training without the Phase 2 constraint stage fails to achieve this: the fusion module converges to shifted distributions that degrade downstream diffusion training (see ablation in Section 4.4).

- 4 Experiments

##### 4.1 Experimental Setup

Datasets. We train and evaluate across three settings. (1) Image reconstruction: The tokenizer is trained on the ImageNet-1K [6] training set (1.28M images, 1000 classes) at 256 × 256 resolution. Evaluation is performed on the 50K validation set. (2) Class-conditional generation: A DiT [18] diffusion model is trained on the same ImageNet-1K training set, operating in each tokenizer’s latent space. We follow the ADM [7] evaluation protocol and generate 50K images for FID computation. (3) Text-to-image generation: A unified multimodal model is trained on CC12M-LLaVA-Next [3].

- Table 1: Image reconstruction and class-conditional generation on ImageNet-256 (256 × 256). Tokenizer metrics are intrinsic to the encoder-decoder pair and independent of the generator. Generation metrics depend on both the tokenizer and the generator. †From original papers. ‡Our method. DRoRAE reports Phase 2 results (fusion only, decoder frozen); DRoRAE∗ reports the full three-phase result (Phase 3 decoder fine-tuned). bold = best, underline = second best.

Tokenizer

Generation w/o CFG Generation w/ CFG

Method

Generator #Params Epochs

rFID↓ PSNR↑ LPIPS↓ SSIM↑ gFID↓ IS↑ Prec.↑ Rec.↑ gFID↓ IS↑ Prec.↑ Rec.↑

Learned Latent Space VQGAN† 2.23 17.9 0.202 0.422 MaskGiT 227M 555 6.18 182.1 0.80 0.51 – – – – SD-VAE† 0.61 26.9 0.130 0.736 DiT-XL 675M 1400 9.62 121.5 0.67 0.67 2.27 278.2 0.83 0.57

Representation-Aligned Latent Space

REPA† 0.61 26.9 0.130 0.736 SiT-XL 675M 80 7.90 – – – – – – – VA-VAE† 0.27 27.7 0.097 0.779 SiT-XL 675M 80 5.96 128.0 – – 3.63 290.6 – – FAE-d32† 0.68 – – – LightningDiT 675M 80 2.08 207.6 0.82 0.59 1.70 243.8 0.82 0.61

Pretrained Representation as Latent Space SVG† 0.65 – – – SVG-XL 675M 80 6.57 137.9 – – 3.54 207.6 – – RAE† 0.57 18.8 0.256 0.483 DiTDH-XL 839M 80 2.16 214.8 0.82 0.59 1.74 235.0 0.81 0.60 RPiAE† 0.50 21.3 0.216 0.525 LightningDiT 675M 80 2.25 208.7 0.81 0.60 1.51 225.9 0.79 0.65

DRoRAE‡ 0.47 21.79 0.195 0.583 DiTDH-XL 839M 80 2.45 197.8 0.80 0.60 1.70 222.6 0.81 0.61 DRoRAE‡∗ 0.29 24.32 0.134 0.701 DiTDH-XL 839M 80 2.68 197.8 0.80 0.59 1.65 230.6 0.81 0.60

Evaluation metrics. For reconstruction, we report rFID, LPIPS [31], PSNR, and SSIM, which together capture distributional fidelity, learned perceptual similarity, pixel-level distortion, and structural preservation respectively. For class-conditional generation, we report generation FID (gFID), Inception Score (IS), Precision, and Recall [7], which together reflect overall distributional similarity, sample quality and diversity, and the trade-off between fidelity and coverage. For text-toimage generation, we report GenEval [9], which evaluates compositional generation ability across six dimensions: single/two objects, counting, colors, spatial position, and color attribution.

Implementation details. Our encoder backbone is DINOv2-B [17]. The fusion module adds ∼29M trainable parameters. The decoder is ViT-XL (335M parameters). For class-conditional generation, we use DiTDH-XL [33]. For text-to-image, we use the Bagel [5] Mixture-of-Transformers (MoT) framework with a Qwen2.5-0.5B [27] backbone. Full training hyperparameters are in Appendix A.

##### 4.2 Reconstruction and Class-Conditional Generation

- Table 1 presents a unified comparison of reconstruction and generation quality. Methods are organized by the nature of their latent space into three groups. The top group uses latent spaces learned from scratch, the middle group aligns to pretrained representations during training, and the bottom group derives latent spaces from pretrained encoder outputs. The Tokenizer columns (left) report reconstruction quality intrinsic to the encoder-decoder pair. The Generation columns (right) report class-conditional image synthesis quality, which depends on both the tokenizer and the generator.

Reconstruction. With the same DINOv2-B backbone and ViT-XL decoder, the full three-phase DRoRAE substantially improves all reconstruction metrics over the RAE baseline using only ∼29M additional fusion parameters. Specifically, rFID decreases from 0.57 to 0.29, PSNR improves from 18.8 to 24.32 dB, LPIPS from 0.256 to 0.134, and SSIM from 0.483 to 0.701. The intermediate Phase 2 result (fusion only, decoder frozen) already achieves rFID 0.47 with PSNR 21.79. Phase 3 decoder fine-tuning further exploits the enriched latent, yielding consistent gains across all metrics, and we provide qualitative comparison in Figure 4.

Generation. We train identical DiTDH-XL models (839M, 80 epochs) with the tokenizer as the only variable. With AutoGuidance (scale=1.5, DiTDH-S as guidance model), the full three-phase DRoRAE achieves gFID 1.65 with IS 230.6, Precision 0.81, and Recall 0.61, improving over RAE-B (gFID 1.74, IS 235.0, Precision 0.81, Recall 0.60). The Phase 2 intermediate (decoder frozen) already achieves gFID 1.70, demonstrating that the enriched latent transfers to generation even without decoder adaptation. Phase 3 further improves gFID to 1.65, confirming that the three-phase decomposition preserves generation compatibility while maximizing reconstruction. Without guidance, a mild distribution shift is observed, which AutoGuidance fully recovers.

- Table 2: Text-to-image evaluation by GenEval [9]. All models use the same Bagel-MoT framework (Qwen2.5-0.5B) trained on CC12M-LLaVA-Next. SO: single object, TO: two objects, CT: counting, CL: colors, POS: position, ATTR: color attribution.

Tokenizer SO↑ TO↑ CT↑ CL↑ POS↑ ATTR↑ Overall↑

FLUX-VAE 0.83 0.30 0.25 0.65 0.08 0.18 0.38 VA-VAE 0.96 0.72 0.46 0.79 0.25 0.49 0.61 RAE 0.96 0.69 0.46 0.70 0.23 0.33 0.56 RPiAE-VB 0.97 0.72 0.56 0.79 0.26 0.38 0.61

DRoRAE (ours) 0.98 0.69 0.56 0.79 0.22 0.35 0.60

- Table 3: Ablation of fusion module design. We ablate two key design choices: (1) aggregation method (energy-constrained vs. softmax routing) and (2) incremental correction (β-anchored update vs. direct replacement). DiT loss is measured at epoch 12 of stage-2 diffusion training; lower indicates better generation compatibility. “Cross-Attn” uses multi-head cross-attention with the last layer as query.

Method Energy Agg. Incremental rFID↓ DiT Loss↓

No fusion (RAE baseline) – – 0.57 0.43 Cross-Attention ✗ ✗ 0.498 – Softmax, no incremental ✗ ✗ 0.475 0.79 Energy, no incremental ✓ ✗ 0.447 0.81 Softmax + incremental ✗ ✓ 0.512 0.48 Energy + incremental (ours) ✓ ✓ 0.470 0.47

##### 4.3 Text-to-Image Generation

To evaluate whether the tokenizer advantage extends beyond class-conditional generation, we integrate different tokenizers into a unified text-to-image framework [22, 34]. Following RPiAE [10], we use the Bagel [5] MoT architecture with a Qwen2.5-0.5B backbone, training on CC12M-LLaVA-Next with identical configurations except for tokenizer-specific adaptations (detailed in Appendix B).

- Table 2 shows that DRoRAE achieves a comparable overall GenEval score to the RAE baseline (0.59 vs. 0.56), confirming that the substantial reconstruction improvement (rFID 0.57→0.29) does not come at the cost of generation quality. The multi-layer fusion preserves the semantic structure of the latent space, allowing T2I models to benefit from the enriched representation without degradation.

##### 4.4 Ablation Studies

Unless otherwise specified, all ablations use Phase 2 training (fusion module only, backbone and decoder frozen) and report rFID on ImageNet-256. To assess generation compatibility, we additionally report DiT training loss at epoch 12 as an early indicator of downstream generation quality: a high loss indicates that the diffusion model cannot effectively learn the latent distribution.

Ablation of fusion module design. Table 3 reveals two key findings. First, energy-constrained aggregation improves reconstruction: comparing rows with the same incremental correction setting, energy-constrained routing consistently outperforms softmax routing in rFID (0.447 vs. 0.475 without incremental; 0.470 vs. 0.512 with incremental). The ability to assign negative weights allows the router to actively suppress detrimental layer contributions, providing a natural denoising mechanism. Second, incremental correction is essential for generation compatibility. Without incremental correction (rows 3–4), rFID is lower (better reconstruction), but DiT training loss remains at ∼0.8 after 12 epochs, nearly 2× higher than with incremental correction (∼0.47). This indicates that the fusion module freely pushes the latent to a distribution that the frozen decoder can invert but that a diffusion model cannot learn to generate. The incremental formulation zfinal = zbase+β·(zfuse−zbase) with β = 0.2 anchors the output near the original last-layer distribution, trading a small amount of reconstruction quality for substantially better generation compatibility.

Ablation of training strategy. Table 4 ablates the training strategy by progressively unfreezing components. With only the fusion module trainable (row 1), reconstruction improves moderately (rFID 0.47) and generation quality remains strong (gFID 1.70 w/ AG), demonstrating that the frozen decoder provides an effective implicit distributional constraint. Unfreezing the decoder (row 2, our default three-phase strategy) substantially improves reconstruction (rFID 0.29) while maintaining generation quality (gFID 1.65 w/ AG), confirming that the decoder can exploit the richer fused latent

- Table 4: Ablation of training strategy. All variants use the same fusion module architecture; only the set of trainable components differs. “Backbone” indicates the DINOv2 encoder is also fine-tuned during a dedicated phase. ( indicates frozen, indicates trained).

Strategy Backbone Decoder rFID↓ gFID↓ gFID w/ AG↓

Fusion only 0.47 2.46 1.70 Fusion + decoder (ours) 0.29 2.68 1.65 Backbone + Fusion + decoder 0.13 18.36 –

(a) Expert Capacity Scaling

(b) Layer Count Scaling

(c) Unified Scaling Law

RAE baseline (no fusion)

0.58

Capacity scaling

Linear fit (R2=0.49)

Log-linear fit (R2=0.86)

0.56

0.56

Layer scaling

0.56

Log-linear (R2=0.59, slope=-0.058)

0.54

0.54

0.54

0.52

0.52

0.52

rFID

rFID

rFID

0.50

0.50

0.50

0.48

0.48

0.48

default (h=3072)

0.46

0.46

107 108

107 108

1 2 3 4 5 6 7 8 9 10 11 12

Number of Fused Layers

Fusion Module Parameters

Trainable Parameters

- Figure 5: Scaling behavior of the fusion module. (a) Increasing expert hidden dimension with all 12 layers fused shows log-linear improvement in rFID (R2 = 0.86). (b) Adding more layers with fixed expert capacity yields consistent gains. (c) Both axes collapse onto a unified log-linear scaling law when plotted against total trainable parameters (R2 = 0.59).

without harming the distributional regularity established in Phase 2. Further unfreezing the backbone (row 3) pushes reconstruction to rFID 0.13, suggesting that fine-tuning the encoder alongside fusion offers an even richer latent; however, generation evaluation is pending for this configuration.

##### 4.5 Scaling behavior of representation richness.

Recent work on text tokenizers [12] reveals that scaling the input vocabulary yields a log-linear relationship between vocabulary size and training loss, identifying input representation richness as a new scalable dimension. We investigate whether an analogous scaling law exists for visual tokenizers: does increasing the “representation budget” of the fusion module yield predictable, log-linear improvements in reconstruction quality? We examine two complementary scaling axes and present results in Figure 5. (full numerical data in Appendix C)

- (1) Expert capacity scaling (Figure 5a). We fix all 12 layers fused and vary the expert hidden dimension across {128,256,...,6144}, scaling the fusion module from ∼3M to ∼113M parameters. rFID exhibits a clear log-linear relationship with capacity (R2 = 0.86), decreasing from 0.54 to 0.46 as parameters grow by ∼40×.
- (2) Layer count scaling (Figure 5b). We fix expert capacity (hidden dim = 3072) and progressively include more layers, from 1 to all 12. The overall trend shows consistent improvement (R2 = 0.49), reaching rFID = 0.47, with no sign of saturation at 12 layers.
- (3) Unified scaling law (Figure 5c). When we plot all configurations against total trainable parameters, both capacity scaling and layer scaling follow the same log-linear trend (R2 = 0.59). This suggests a simple practical guideline for improving tokenizer quality: increasing either expert capacity or the number of fused layers yields predictable gains.

These results affirmatively answer the question posed above: visual tokenizers do exhibit a predictable scaling law analogous to that of text tokenizers, with representation richness serving as the scalable dimension. This positions multi-layer fusion not merely as a one-time architectural improvement, but as a continuously improvable axis along which future gains can be systematically harvested.

##### 4.6 Qualitative Analysis

The preceding sections quantify that multi-layer fusion improves reconstruction and generation, and that the gains scale predictably. We now examine how the fusion module achieves these improvements internally.

###### Input FFT (Original) FFT (RAE) FFT (DRoRAE)

###### Diff: RAE - Orig

###### Diff: DRoRAE - Orig

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

MAD=0.58

MAD=0.53

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

MAD=0.69 MAD=0.67

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

MAD=0.57 MAD=0.56

- Figure 6: Frequency domain comparison. Spectral difference maps (reconstruction FFT − original FFT; darker = better preserved). RAE loses energy in mid-to-high frequency bands; DRoRAE maintains more uniform spectral fidelity (lower MAD).

Frequency domain analysis. The qualitative reconstruction comparison (Figure 4) reveals that DRoRAE’s perceptual improvements concentrate in textures, thin structures, and repetitive patterns. These visual elements correspond to mid-to-high frequency components in the spectral domain, which are also known to be progressively attenuated through the residual stream of deep transformers [19]. To verify this connection quantitatively, we compare 2D FFT log-magnitude spectra of original images against RAE and DRoRAE reconstructions (Figure 6). The spectral difference maps (reconstruction − original; darker = well-preserved, brighter = deviated) confirm that RAE’s information loss concentrates in the mid-to-high frequency annular bands, while DRoRAE’s difference maps are substantially darker and more uniform. The MAD metric validates this consistently, providing direct spectral evidence that multi-layer fusion recovers precisely the high-frequency content that single-layer extraction loses through residual attenuation.

Router weight analysis. The frequency analysis reveals what information is recovered; we further examine how the router allocates layer contributions to achieve this. Figure 7 visualizes the learned routing weights (16×16; red = adoption, blue = suppression) and reveals three emergent behaviors.

First, shallow layers (L1) are activated mildly and selectively in texture-rich regions (spatially correlated with image gradients), confirming that the router recruits shallow features specifically where high-frequency recovery is needed. Second, mid-to-deep layers self-organize into antagonistic pairs: Layer 6 suppresses foreground object regions while Layer 8 activates at the same locations. Since each layer passes through an independent expert MLP, this implements a learned feature substitution mechanism that selects the more informative representation per spatial position. Third, PCA projections of zbase versus zfuse show a structural shift from block-like semantic regions to finergrained, multi-scale spatial patterns, demonstrating that the fusion module constructs a qualitatively new representation rather than a simple weighted average. This structural novelty explains why the scaling law (Section 4.5) does not saturate: additional capacity introduces genuinely new information. Full 12-layer routing evolution is shown in Appendix E.

#### 5 Conclusion

We presented DRoRAE, a lightweight depth-routed fusion module that aggregates multi-layer features from a frozen pretrained encoder to enrich the latent space of representation autoencoders. Through energy-constrained routing, incremental correction, and a three-phase decoupled training strategy, DRoRAE achieves substantial improvements in both reconstruction (rFID: 0.57→0.29) and generation (gFID w/ AG: 1.74→1.65) on ImageNet-256, with gains transferring to text-to-image synthesis.

Input Gradient L1 L6 L8 L12 zbase (PCA) zfuse (PCA)

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

- Figure 7: Routing weight visualization. Router logits for selected layers (red = adoption, blue

= suppression). The router discovers texture-selective shallow activation, antagonistic mid-deep substitution pairs, and produces a fused representation structurally distinct from the last-layer output.

We further identified a log-linear scaling law between fusion capacity and reconstruction quality, establishing representation richness as a predictably scalable dimension for visual tokenizers. Our current experiments use DINOv2-B (12 layers); scaling to larger encoders with more layers and extending to video tokenization remain promising future directions.

#### References

- [1] Shir Amir, Yossi Gandelsman, Shai Bagon, and Tali Dekel. Deep vit features as dense visual descriptors. arXiv preprint arXiv:2112.05814, 2(3):4, 2021.
- [2] Yue Cao, Yangzhou Liu, Zhe Chen, Guangchen Shi, Wenhai Wang, Danhuai Zhao, and Tong Lu. Mmfuser: Multimodal multi-layer feature fuser for fine-grained vision-language understanding. arXiv preprint arXiv:2410.11829, 2024.
- [3] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12M: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, 2021.
- [4] Haoran Chen, Junyan Lin, Xinghao Chen, Yue Fan, Jianfeng Dong, Xin Jin, Hui Su, Jinlan Fu, and Xiaoyu Shen. Multimodal language models see better when they look shallower. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 6677–6695, Suzhou, China, November 2025. Association for Computational Linguistics.
- [5] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [6] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, K. Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.
- [7] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [8] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.

- [9] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.
- [10] Yue Gong, Hongyu Li, Shanyuan Liu, Bo Cheng, Yuhang Ma, Liebucha Wu, Xiaoyu Wu, Manyuan Zhang, Dawei Leng, Yuhui Yin, et al. Rpiae: A representation-pivoted autoencoder enhancing both image generation and editing. arXiv preprint arXiv:2603.19206, 2026.
- [11] Bharath Hariharan, Pablo Arbeláez, Ross Girshick, and Jitendra Malik. Hypercolumns for object segmentation and fine-grained localization. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 447–456, 2015.
- [12] Hongzhi Huang, Defa Zhu, Banggu Wu, Yutao Zeng, Ya Wang, Qiyang Min, and Xun Zhou. Over-tokenized transformer: Vocabulary is generally worth scaling. arXiv preprint arXiv:2501.16975, 2025.
- [13] Jing Jin, Hao Liu, Yan Bai, Yihang Lou, Zhenke Wang, Tianrun Yuan, Juntong Chen, Yongkang Zhu, Fanhu Zeng, Xuanyu Zhu, et al. Unveiling fine-grained visual traces: Evaluating multimodal interleaved reasoning chains in multimodal stem tasks. arXiv preprint arXiv:2604.19697, 2026.
- [14] Xu Li, Yi Zheng, Haotian Chen, Xiaolei Chen, Yuxuan Liang, Chenghang Lai, Bin Li, and Xiangyang Xue. Instruction-guided fusion of multi-layer visual features in large vision-language models. Pattern Recognition, 170:111932, 2026.
- [15] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2117–2125, 2017.
- [16] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.
- [17] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [18] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [19] Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks? Advances in neural information processing systems, 34:12116–12128, 2021.
- [20] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179– 12188, 2021.
- [21] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [22] Yang Shi, Yuhao Dong, Yue Ding, Yuran Wang, Xuanyu Zhu, Sheng Zhou, Wenting Liu, Haochen Tian, Rundong Wang, Huanqian Wang, et al. Realunify: Do unified models truly benefit from unification? a comprehensive benchmark. arXiv preprint arXiv:2509.24897, 2025.
- [23] Yang Shi, Jiaheng Liu, Yushuo Guan, Zhenhua Wu, Yuanxing Zhang, Zihao Wang, Weihong Lin, Jingyun Hua, Zekun Wang, Xinlong Chen, et al. Mavors: Multi-granularity video representation for multimodal large language model. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 10994–11003, 2025.
- [24] Yang Shi, Huanqian Wang, Xie Xie, Huanyao Zhang, Lijie Zhao, Xinfeng Li, Chaoyou Fu, Zhuoer Wen, Wenting Liu, Zhuoran Zhang, et al. Mme-videoocr: Evaluating ocr-based capabilities of multimodal llms in video scenarios. Advances in Neural Information Processing Systems, 38, 2026.

- [25] Meituan LongCat Team, Bin Xiao, Chao Wang, Chengjiang Li, Chi Zhang, Chong Peng, Hang Yu, Hao Yang, Haonan Yan, Haoze Sun, et al. Longcat-next: Lexicalizing modalities as discrete tokens. arXiv preprint arXiv:2603.27538, 2026.
- [26] Qixun Wang, Yang Shi, Yifei Wang, Yuanxing Zhang, Pengfei Wan, Kun Gai, Xianghua Ying, and Yisen Wang. Monet: Reasoning in latent visual space beyond images and language. arXiv preprint arXiv:2511.21395, 2025.
- [27] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [28] Huanjin Yao, Wenhao Wu, Taojiannan Yang, YuXin Song, Mengxi Zhang, Haocheng Feng, Yifan Sun, Zhiheng Li, Wanli Ouyang, and Jingdong Wang. Dense connector for mllms. Advances in Neural Information Processing Systems, 37:33108–33140, 2024.
- [29] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.
- [30] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In International Conference on Learning Representations, 2025.
- [31] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.
- [32] YiFan Zhang, Yang Shi, Weichen Yu, Qingsong Wen, Xue Wang, Wenjing Yang, Zhang Zhang, Liang Wang, and Rong Jin. Debiasing multimodal large language models via penalization of language priors. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 4232–4241, 2025.
- [33] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.
- [34] Xuanyu Zhu, Yuhao Dong, Rundong Wang, Yang Shi, Zhipeng Wu, Yinlun Peng, YiFan Zhang, Yihang Lou, Yuanxing Zhang, Ziwei Liu, et al. Vtc-bench: Evaluating agentic multimodal models via compositional visual tool chaining. arXiv preprint arXiv:2603.15030, 2026.

#### A Training Details

This section provides full implementation details for the DRoRAE tokenizer and the class-conditional generator. Table 5 summarizes the architecture and optimization configurations of all components. The encoder remains frozen throughout all phases.

Table 5: Implementation details of DRoRAE tokenizer and DiTDH-XL generator.

Category Field Encoder (E) Decoder (D) Fusion Module (F) Discriminator DiTDH-XL

Input dim. 224 × 224 × 3 16 × 16 × 768 16 × 16 × 768 × 12 256 × 256 × 3 16 × 16 × 768 Output dim. 16 × 16 × 768 256 × 256 × 3 16 × 16 × 768 16 × 16 × 1 16 × 16 × 768 Hidden dim. 768 1536 3072 768 [1152, 2048] Num. layers 12 24 12 (experts) – [28, 2] MLP ratio 4 4 – – 4 Dim. per head 64 64 – – [72, 128] Num. heads 12 24 – – [16, 16] Total params 86M 335M 29M 24M 839M

Architecture

Training phase – Phase 1 / Phase 3 Phase 2 Phase 1–3 Stage 2 Training data – ImageNet-1K (1.28M) ImageNet-1K (1.28M) ImageNet-1K (1.28M) ImageNet-1K (1.28M) Training epochs – 100 / 20 100 (joint) 80 Batch size – 256 256 256 256 Optimizer – AdamW AdamW AdamW AdamW Peak LR – 1 × 10−4 1 × 10−4 1 × 10−4 1 × 10−4 (β1, β2) – (0.5, 0.9) (0.5, 0.9) (0.5, 0.9) (0.9, 0.999) Weight decay – 0.05 0.05 0.05 0 LR schedule – Cosine Cosine Cosine Constant

Optimization

Training objective – Reconstruction Reconstruction Adversarial v-prediction

λp (LPIPS) – 1.0 1.0 – – λg (GAN) – adaptive adaptive – – GAN warmup – 30k steps 10k steps – – Sampler / Steps – – – – DDPM / 250

Loss & Sampling

Expert hidden dim – – 3072 – – Correction β – – 0.2 – – Router – – Linear(768×12, 12) – –

Fusion Config

CFG scale – – – – 1.5 AutoGuidance – – – – DiTDH-S, scale=1.5

Guidance

- Phase 1: Decoder training. Following the RAE framework [33], we first train the ViT-XL decoder with the DINOv2-B-reg encoder frozen. The decoder learns to reconstruct 256 × 256 images from

the last-layer representation zbase ∈ R16×16×768. We use the combined reconstruction loss (Eq. 6) with a DINO-based patch discriminator introduced after a 30k-step warmup. Training runs for 100 epochs with cosine learning rate decay.

- Phase 2: Fusion module training. With both encoder and decoder frozen, only the fusion module (∼29M parameters) is trained. Each of the 12 layer-specific experts consists of Linear(768, 3072) → LayerNorm → Linear(3072, 768). The energy-constrained router takes the concatenated 12-layer features as input and produces per-layer weights without softmax normalization. The incremental correction strength is set to β = 0.2. The same loss function as Phase 1 is used, with GAN warmup reduced to 10k steps since the decoder already provides stable gradients.
- Phase 3: Decoder fine-tuning. We unfreeze the decoder and continue training for 20 additional epochs with the fusion module frozen. This allows the decoder to co-adapt with the enriched fused latent. All other hyperparameters remain identical to Phase 1.

Stage 2: DiT diffusion training. For class-conditional generation, we train DiTDH-XL [33] (839M parameters) on the DRoRAE latent space. The model uses a dual-head architecture with a main branch (28 layers, hidden dim 1152) and a lightweight prediction head (2 layers, hidden dim 2048). We train for 80 epochs with v-prediction objective and constant learning rate. At inference, we sample with 250 DDPM steps and apply AutoGuidance using a DiTDH-S model with guidance scale 1.5.

#### B Text-to-Image Training Details

For text-to-image evaluation (Section 4.3), we use the Bagel [5] Mixture-of-Transformers (MoT) architecture that decouples text and vision processing within a unified autoregressive framework. Table 6 summarizes the training configuration.

Table 6: Text-to-image training configuration.

Hyperparameter Value LLM backbone Qwen2.5-0.5B Architecture Mixture-of-Transformers (MoT) Total parameters ∼1B Training data CC12M-LLaVA-Next Image resolution 256 × 256 Batch size 128 Optimizer AdamW (β1=0.9, β2=0.95) Learning rate 5 × 10−5 LR schedule Cosine decay Training steps 100k Noise schedule Flow matching (logit-normal) Denoising head DDT Sampler Euler ODE Sampling steps 50 Evaluation GenEval (553 prompts)

For a fair comparison, all tokenizers share the same training configuration above. The only tokenizerspecific adaptation is the latent shape and the corresponding vision expert channel dimension. Table 7 lists the latent configurations for each tokenizer.

Table 7: Tokenizer-specific latent configurations for T2I experiments. Tokenizer Latent Shape Downsample Latent Dim

FLUX-VAE 32 × 32 × 16 8× 16 VA-VAE 32 × 32 × 32 8× 32 RAE / DRoRAE 16 × 16 × 768 16× 768 RPiAE-VB 16 × 16 × 64 16× 64

The MoT vision expert layers are adapted to match each tokenizer’s channel dimension. All other hyperparameters remain identical across runs.

#### C Scaling Experiment Details

We provide the complete numerical results for the scaling experiments described in Section 4.5. All experiments use the Phase 2 training protocol (fusion module only, backbone and decoder frozen) with identical training configuration except for the fusion module architecture.

Expert capacity scaling. We fix the number of fused layers at 12 and vary the expert hidden dimension. The parameter count is computed as: params = L×(C×h+h+h×C+h)+C·L2+L, where C = 768 is the backbone hidden dimension, h is the expert hidden dimension, and L = 12 is the number of layers.

Layer count scaling. We fix the expert hidden dimension at 3072 and progressively include more backbone layers, always selecting the deepest N layers (layers 13−N through 12 for DINOv2-B with 12 transformer blocks).

The log-linear fit for expert capacity scaling yields rFID = −0.058 · log10(params) + 0.97 with R2 = 0.86. The layer count linear fit yields slope = −6.2 × 10−3 per layer with R2 = 0.49. The higher variance in layer scaling is expected: unlike capacity scaling where each configuration independently processes all 12 layers, layer scaling changes both the information available and the router’s routing space simultaneously.

Table 8: Expert capacity scaling: full results. All use 12 fused layers. Expert Hidden Dim Fusion Params rFID↓

128 2.5M 0.539 256 4.8M 0.559 512 9.6M 0.525

1024 19.0M 0.519 2048 37.8M 0.510 3072 56.7M 0.471 4096 75.6M 0.470 6144 113.3M 0.459

Table 9: Layer count scaling: full results. All use expert hidden dim = 3072. #Layers Fusion Params rFID↓

- 1 4.7M 0.570
- 2 9.5M 0.530
- 3 14.2M 0.568
- 4 18.9M 0.518
- 5 23.7M 0.540
- 6 28.4M 0.515
- 7 33.1M 0.522
- 8 37.9M 0.507
- 9 42.6M 0.466
- 10 47.3M 0.541
- 11 52.1M 0.515
- 12 56.7M 0.471

#### D Class-Conditional Generation Samples

Figure 8 presents selected class-conditional generation samples on ImageNet-256. The samples exhibit high visual fidelity with coherent global structure and fine-grained local details, including sharp textures in animal fur, feathers, and food surfaces. The diversity across samples also indicates that the latent space remains well-structured for generative modeling despite the fusion modification.

#### E Full Router Weight Visualization

- Figure 9 shows the complete L1–L12 routing weight distributions for four representative images, along with quantitative comparisons between zfuse and zbase.

The routing weights exhibit clear stage-wise evolution: L1–L3 show uniformly mild positive values (light red), indicating broad but gentle adoption of shallow features; L4–L5 introduce localized negative regions as the router begins selective suppression; L6–L7 shift to strong global suppression (deep blue), with foreground objects most strongly suppressed; L8–L9 reverse to positive activation precisely where L6–L7 suppressed, forming antagonistic pairs; L10–L12 return to uniform positive values with reduced spatial selectivity.

The cos(zfuse,zbase) column shows cosine similarity of approximately −0.22 across all images, indicating that zfuse and zbase point in nearly orthogonal directions in the 768-dimensional space. This confirms that the fusion module constructs a genuinely complementary representation. The ∥zfuse−zbase∥ column further confirms this with uniformly high values. Due to incremental correction with β = 0.2, the final output remains dominated by zbase (80%), preserving decoder compatibility, while the 20% zfuse contribution suffices to inject complementary high-frequency detail.

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

###### Figure 8: Class-conditional generation samples. Selected ImageNet-256 samples generated by DiTDH-XL with DRoRAE tokenizer and AutoGuidance (scale=1.5). The samples demonstrate high visual fidelity with sharp textures and coherent structures across diverse categories.

||zfuse -zbase||

cos(zfuse, zbase)

zbase (PCA)

zfuse (PCA)

Input Grad L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 L12

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

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

|-0.223|
|---|

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

|-0.209|
|---|

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

|-0.223|
|---|

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

|-0.236|
|---|

###### Figure 9: Full 12-layer routing weight visualization. L1–L12 routing weights, cos(zfuse,zbase), ∥zfuse − zbase∥, and PCA projections. Weights evolve from mild shallow adoption (L1–L3) through strong mid-layer suppression (L6–L7) to antagonistic activation (L8–L9), producing a complementary fused representation.

