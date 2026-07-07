## Personalize Anything for Free with Diffusion Transformer

Haoran Feng1* Zehuan Huang2*† Lin Li3 Hairong Lv1 Lu Sheng2✉

1Tsinghua University 2Beihang University 3Renmin University of China Project page: https://fenghora.github.io/Personalize-Anything-Page/

# arXiv:2503.12590v1[cs.CV]16Mar2025

Single-subject Personalization

Layout-guided Subject Personalization

|[Figure 1]|
|---|

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

|[Figure 6]|
|---|

[Figure 7]

[Figure 8]

|[Figure 9]|
|---|

a can stuck on the beach

a dragon flying in the sky

|[Figure 10]|
|---|

|[Figure 11]|
|---|

a dog playing with a ball

a robot walking on the road

Multi-Subject Personalization Subject-Scene Composition

Inpainting and Outpainting

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

|[Figure 20]|
|---|

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

|[Figure 25]|
|---|

Visual Storytelling

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Figure 1. Personalize Anything is a training-free framework based on Diffusion Transformers (DiT) for personalized image generation. The framework demonstrates advanced versatility, excelling in single-subject personalization (top), multi-subject or subject-scene composition, inpainting and outpainting (middle), as well as applications like visual storytelling (bottom), all without any training or fine-tuning.

### Abstract

vation, we propose Personalize Anything, a training-free framework that achieves personalized image generation in DiT through: 1) timestep-adaptive token replacement that enforces subject consistency via early-stage injection and enhances flexibility through late-stage regularization, and 2) patch perturbation strategies to boost structural diversity. Our method seamlessly supports layout-guided generation, multi-subject personalization, and mask-controlled editing. Evaluations demonstrate state-of-the-art performance in identity preservation and versatility. Our work establishes new insights into DiTs while delivering a practical paradigm for efficient personalization.

Personalized image generation aims to produce images of user-specified concepts while enabling flexible editing. Recent training-free approaches, while exhibit higher computational efficiency than training-based methods, struggle with identity preservation, applicability, and compatibility with diffusion transformers (DiTs). In this paper, we uncover the untapped potential of DiT, where simply replacing denoising tokens with those of a reference subject achieves zero-shot subject reconstruction. This simple yet effective feature injection technique unlocks diverse scenarios, from personalization to image editing. Building upon this obser-

∗ Equal contribution † Project lead ✉ Corresponding author

### 1. Introduction

[Figure 32]

[Figure 33]

Personalized image generation aims to synthesize images of user-specified concepts while enabling flexible editing. The advent of text-to-image diffusion models [4, 27, 34, 39, 40, 42, 44, 48, 55] has revolutionized this field, enabling applications in areas like advertising production.

[Figure 34]

|Token Replacing|
|---|

[Figure 35]

[Figure 36]

Subject

Previous research on subject image personalization relies on test-time optimization or large-scale fine-tuning. Optimization-based approach [12, 21, 23, 24, 26, 32, 47, 49, 71] enables the pre-trained models to learn the specific concept through fine-tuning on a few images of a subject. While achieving identity preservation, these methods demand substantial computational resources and time due to per-subject optimization requiring hundreds of iterative steps. Large-scale fine-tuning alternatives [2, 3, 7, 11, 13, 15, 18–20, 22, 28–30, 33, 36–38, 43, 52, 54, 61– 65, 67, 68, 72] seek to circumvent this limitation by training auxiliary networks on large-scale datasets to encode reference images. However, these approaches both demand heavy training requirements and risk overfitting to narrow data distributions, degrading their generalizability.

U-Net DiT

Figure 2. Simple token replacement in DiT (right) achieves highfidelity subject reconstruction through its position-disentangled representation, while U-Net’s convolutional entanglement (left) induces blurred edges and artifacts.

tion in DiT, unlocking various scenarios ranging from personalization to inpainting and outpainting, without necessitating complicated attention engineering.

Building on this foundation, we propose “Personalize Anything”, a training-free framework enabling personalized image generation in DiT through timestep-adaptive token replacement and patch perturbation strategies. Specifically, we inject reference subject tokens (excluding positional information) in the earlier steps of the denoising process to enforce subject consistency, while enhancing flexibility in the later steps through multi-modal attention. Furthermore, we introduce patch perturbation to the reference tokens before token replacement, locally shuffling them and applying morphological operations to the subject mask. It encourages the model to introduce more global appearance information and enhances structural and textural diversity. Additionally, our framework seamlessly supports 1) layoutguided generation through translations on replacing regions, 2) multi-subject personalization and subject-scene composition via sequential injection of reference subjects or scene, and 3) extended applications (e.g. inpainting and outpainting) via incorporating user-specified mask conditions.

Recent training-free solutions [1, 5, 25, 50, 57, 69, 70] exhibit higher computational efficiency than the trainingbased approach. These methods typically leverage an attention sharing mechanism to inject reference features, processing denoising and reference subject tokens jointly in pre-trained self-attention layers. However, these attentionbased methods lack constraints on subject consistency and often fail to preserve identity. Moreover, their application to advanced text-to-image diffusion transformers (DiTs) [8, 27, 39] proves challenging stemming from DiT’s positional encoding mechanism. As analyzed in Sec. 3.2, we attribute this limitation to the strong influence of the explicitly encoded positional information on DiT’s attention mechanism. This makes it difficult for generated images to correctly attend to the reference subject’s tokens within traditional attention sharing.

In this paper, we delve into the diffusion transformers (DiTs) [8, 27, 39], and observe that simply replacing the denoising tokens with those of a reference subject allows for high-fidelity subject reconstruction. As illustrated in Fig. 2, DiT exhibits exceptional reconstruction fidelity under this manipulation, while U-Net [45] often induces blurred edges and artifacts. We attribute this to the separate embedding of positional information in DiT, achieved via its explicit positional encoding mechanism. This decoupling of semantic features and position enables the substitution of purely semantic tokens, avoiding positional interference. Conversely, U-Net’s convolutional mechanism binds texture and spatial position together, causing positional conflicts when replacing tokens and leading to low-quality image generation. This discovery establishes token replacement as a viable pathway for zero-shot subject personaliza-

Comprehensive evaluations on multiple personalization tasks demonstrate that our training-free method exhibits superior identity preservation, fidelity and versatility, outperforming existing approaches including those fine-tuned on DiTs. Our contributions are summarized as follows:

- • We uncover DiT’s potential for high-fidelity subject reconstruction via simple token replacement, and characterize its position-disentangled properties.
- • We introduce a simple yet effective framework, denoted as “Personalize Anything”, which starts with subject reconstruction and enhances the flexibility via timestepadaptive replacement and patch perturbation.
- • Experiments demonstrates that the proposed framework

exhibits high consistency, fidelity, and versatility across multiple personalization tasks and applications.

### 2. Related Work

#### 2.1. Text-to-Image Diffusion Models

Text-to-image generation has been revolutionized by diffusion models [17, 51] that progressively denoise Gaussian distributions into images. Among a series of effective works [4, 27, 34, 39, 40, 42, 44, 48, 55], Latent Diffusion Model [44] employs a U-Net backbone [45] for efficient denoising within compressed latent space, becoming the foundation for subsequent improvements in resolution [40]. A recent architectural shift replaces convolutional U-Nets with vision transformers [6, 39], exploiting their global attention mechanisms and geometrically-aware positional encodings. These diffusion transformers (DiTs) demonstrate superior scalability [8, 27]—performance improvements consistently correlate with increased model capacity and training compute, establishing them as the new state-of-the-art paradigm.

#### 2.2. Personalized Image Generation

Training-Based Approaches. Previous subject personalization methods primarily adopt two strategies: i) Test-time optimization techniques [10, 12, 14, 21, 23, 24, 26, 32, 47, 49, 56, 58, 71] that fine-tune foundation models on target concepts at inference time, often requiring 30 GPU-minute optimization per subject; and ii) large-scale training-based methods [2, 3, 7, 11, 13, 15, 18–20, 22, 28–30, 33, 36– 38, 43, 52, 54, 61–63, 63–65, 67, 68, 72] that learn concept embeddings through auxiliary networks pre-trained on large datasets. While achieving notable fidelity, both paradigms suffer from computational overheads and distribution shifts that limit real-world application.

Training-Free Alternatives. Emerging training-free methods [1, 5, 25, 50, 57, 69, 70] exhibit higher computational efficiency than the training-based approach. These methods typically leverage an attention sharing mechanism to inject reference features, processing denoising and reference subject tokens jointly in pre-trained self-attention layers. However, these attention-based methods lack constraints on subject consistency and fail to preserve identity. Moreover, their application to advanced diffusion transformers (DiTs) [8, 27, 39] proves challenging due to DiT’s explicit positional encoding, thereby limiting their scalability to larger-scale text-to-image generation models [8, 27].

### 3. Methodology

This paper introduces a training-free paradigm for personalized generation using diffusion transformers (DiTs) [39], synthesizing high-fidelity depictions of user-specified con-

cepts while preserving textual controllability. In the following sections, we start with an overview of standard architectures in text-to-image diffusion models. Sec. 3.2 systematically reveals architectural distinctions that impede the application of existing attention sharing mechanisms to DiTs. Sec. 3.3 uncovers DiT’s potential for subject reconstruction via simple token replacement, culminating the presentation of our Personalize Anything framework in Sec. 3.4.

#### 3.1. Preliminaries

Diffusion models progressively denoise a latent variable zT through a network ϵθ, with architectural choices being of paramount importance. We analyze two main paradigms:

U-Net Architectures. The convolutional U-Net [45] in Stable Diffusion [44] comprises pairs of down-sampling and up-sampling blocks connected by a middle block. Each block interleaves residual blocks for feature extraction with spatial attention layers for capturing spatial relationships.

Diffusion Transformer (DiT). Modern DiTs [39] in advanced models [8, 27] leverage transformers [6] to process discretized latent representations, including image tokens X ∈ RN×d and text tokens C ∈ RM×d, where d is the embedding dimension, N and M are the length of sequences. These models typically encode positional information of X through RoPE [53], which applies rotation matrices based on the token’s coordinate (i,j) in the 2D grid:

X˙ i,j = Xi,j · R(i,j) (1)

where R(i,j) denotes the rotation matrix at position (i,j) with 0 ≤ i < w and 0 ≤ j < h. Text tokens C receive fixed positional anchors (i = 0,j = 0) to maintain modality distinction. The multi-modal attention mechanism (MMA) is then applied to all position-encoded tokens [X˙ ;C˙T] ∈ R(N+M)×d, enabling full bidirectional attention across both modalities.

#### 3.2. Attention Sharing Fails in DiT

We systematically investigate why established U-Net-based personalization techniques [5, 57] fail when naively applied to DiT architectures [27], identifying positional encoding conflicts as the core challenge.

Positional Encoding Collision. Implementing attention sharing of existing methods [5, 57] in DiT, we concatenate position-encoded denoising tokens X˙ and reference tokens X˙ref (obtained via flow inversion [60]) into a unified sequence [X˙ ;X˙ref]. Both tokens keep the original positions (i,j) ∈ [0,w) × [0,h), causing destructive interference to attention computation. As visualized in Fig. 3a, this forces denoising tokens to over-attend to reference tokens with the same positions, resulting in ghosting artifacts of the reference subject in the generated image. Quantitative analysis in supplementary materials reveals that the attention score

Output Image Attention map of red blocks

the reference image, obtaining the reference tokens Xref without encoded positional information, as well as the reference subject’s mask Mref. We then inject Xref into specific region M of the denoising tokens X via token replacement:

|Denoising Tokens|Reference Tokens|
|---|---|

[Figure 37]

[Figure 38]

[Figure 39]

###### Attention Sharing

0 w

0

[Figure 40]

Xˆ = X ⊙ (1 − M) + Xref ⊙ M (2)

- (a) Keep original positions: over-attend to the same positions

(c) Shift to (𝑖 + 𝑤,𝑗): attention miss in reference tokens

- (b) Remove positions: attention miss in reference tokens

[Figure 41]

[Figure 42]

[Figure 43]

where M can be obtained by translating Mref. As shown in Fig. 2, token replacement in DiT reconstructs highfidelity images with consistent subjects in specified positions, while U-Net’s convolutional entanglement manifests as blurred edges and artifacts.

h

Denoising Tokens

C

(𝑖,𝑗)

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

We attribute this to the separate embedding of positional information in DiT, achieved via its explicit positional encoding mechanism (Sec. 3.1). This decoupling of semantic features and position enables the substitution of purely semantic tokens, avoiding positional interference. Conversely, U-Net’s convolutional mechanism binds texture and spatial position together, causing positional conflicts when replacing tokens and leading to low-quality image generation. This discovery establishes token replacement as a viable pathway for zero-shot subject personalization in DiT. It unlocks various scenarios ranging from personalization to inpainting and outpainting, without necessitating complicated attention engineering, and establishes the foundation for our personalization framework in Sec. 3.4.

Reference Tokens

C Concatenation

Positional Encoding

- Figure 3. Attention sharing [5, 57] fails in DiT due to the explicit positional encoding mechanism. When keeping the original positions (i, j) ∈ [0, w) × [0, h) in reference tokens, denoising tokens over-attend to reference ones with the same positions (shown in attention maps of (a)), resulting in ghosting artifacts in the generated image. Modified strategies, (b) removing positions and (c) shifting to non-overlapping regions, avoid collisions but loses identity alignment, as attention is almost absent on reference tokens.

#### 3.4. Personalize Anything

between denoising and reference tokens at the same position in DiT is 723% higher than in U-Net, confirming DiT’s position sensitivity.

Building upon these discoveries, we propose Personalize Anything, a novel training-free personalization for diffusion transformers (DiTs). This framework draws inspiration from zero-shot subject reconstruction in DiTs, and effectively enhances flexibility by timestep-adaptive token replacement and patch perturbation strategies (Fig. 4).

Modified Encoding Strategies. Motivated by DiT’s position-disentangled encoding (Sec. 3.1), we engineer two positional adjustments on Xref to obtain non-conflicting X˙ref: i) remove positions and fix all reference positions to (0,0) akin to text tokens, and ii) shift reference tokens to (i′,j′) = (i + w,j), creating non-overlapping regions. As shown in Fig. 3 (b) and (c), while eliminating collisions, both methods struggle to preserve identity, as attention is almost absent on reference tokens.

Timestep-adaptive Token Replacement. Our method begins by inverting a reference image containing the desired subject [60]. This process yields reference tokens Xref (excluding positional encodings) and a corresponding subject mask Mref [59]. Instead of continuous injection throughout the denoising process as employed in subject reconstruction, we introduce a timestep-dependent strategy:

In summary, the explicitly encoded positional information exhibits strong influence on the attention mechanism in DiT—a fundamental divergence from U-Net’s implicit position handling. This makes it difficult for generated images to correctly attend to the reference subject’s tokens within traditional attention sharing.

❶ Early-stage subject anchoring via token replacement (t > τ). During the initial denoising steps (t > τ, where τ is an empirically determined threshold set at 80% of the total denoising steps T), we anchor the subject’s identity by replacing the denoising tokens X within the subject region M with the reference tokens Xref (Eq. (2)). The region M can be obtained by translating Mref to the user-specified location. We preserve the positional encodings associated with the denoising tokens X to maintain spatial coherence.

#### 3.3. Token Replacement in DiT

Building on the foundational observation on DiT’s architectural distinctions, we extend our investigation to the latent representation in DiT. We uncover that simply replacing the denoising tokens with those of a reference subject allows for high-fidelity subject reconstruction.

❷ Later-stage semantic fusion via multi-modal attention (t ≤ τ). In later denoising steps t ≤ τ, we transition to semantic fusion. Here we concatenate zero-positioned reference tokens X˙ref with denoising tokens X˙ and text

Specifically, we apply inversion techniques [46, 60] on

[Figure 50]

[Figure 51]

[Figure 52]

Token Replacement

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Replace

[Figure 58]

[Figure 59]

… …

…

[Figure 60]

DiT Block

DiT Block

[Figure 61]

Perturbation

Reference Image 𝑋

𝑋

[Figure 62]

[Figure 63]

𝑋

[Figure 64]

[Figure 65]

Multi-Modal Attention

Token Replacement

Multi-Modal Attention

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

|[Figure 74]|
|---|

[Figure 75]

[Figure 76]

[Figure 77]

BlockDiT … BlockDiT …

…

…

“a dog on the beach” 𝑀𝑀𝐴( 𝑋̇;𝑋̇ ;𝐶̇ )

“a dog on the beach”

Timestep=T 𝜏 Timestep=0

- Figure 4. Method overview. Our framework anchors subject identity in early denoising through mask-guided token replacement with preserved positional encoding, and transitions to multi-modal attention for semantic fusion with text in later steps. During token replacement, we inject variations via patch perturbations. This timestep-adaptive strategy balances identity preservation and generative flexibility.

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

(a) Layout-guided generation

Token Replacing ①

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

- ②

- ③

[Figure 87]

(b) Multi-subject or subject-scene composition (c) Inpainting and outpainting

[Figure 88]

[Figure 89]

[Figure 90]

|[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>| |
|---|
|
|---|

[Figure 94]

[Figure 95]

[Figure 96]

|[Figure 97]|
|---|

[Figure 98]

[Figure 99]

[Figure 100]

- Figure 5. Seamless extensions. Our framework enables: (a) layout-guided generation by translating token-injected regions, (b) multisubject composition through sequential token injection, and (c) inpainting and outpainting via specifying masks and increased replacement.

embeddings C˙ . The unified sequence [X˙ ;X˙ref;C˙T] undergoes Multi-Modal Attention (MMA) to harmonize subject guidance with textual conditions. This adaptive threshold τ balances the preservation of subject identity with the flexibility afforded by the text prompt.

eration, while sequential injection of multiple {Xrefk } into disjoint {Mk} regions and unified Multi-Modal Attention MMA([X˙ ;{X˙refk };C˙T]) facilitate multi-subject or subjectscene composition. For image editing tasks, we incorporate user-specified masks in the inversion process to obtain reference Xref and Mref that should be preserved. Meanwhile, we disable perturbations and set τ to 10% total steps, preserving the original image content as much as possible and achieving coherent inpainting or outpainting.

Patch Perturbation for Variation. To prevent identity overfitting while preserving identity consistency, we introduce two complementary perturbations: 1) Random Local token shuffling within 3x3 windows disrupts rigid texture alignment, and 2) Mask augmentation of Mref, including simulating natural shape variations using morphological dilation/erosion with a 5px kernel, or manual selection of regions emphasizing identity. The idea behind this local interference technique is to encourage the model to introduce more global textural features while enhancing structural and local diversity.

### 4. Experiments

#### 4.1. Experimental Setup

Implementation Details. Our framework builds upon the open-source HunyuanDiT [31] and FLUX.1-dev [27]. We adopt 50-step sampling with classifier-free guidance (w = 3.5), generating 1024 × 1024 resolution images. Token replacement threshold τ is set to 80% total steps.

Seamless Extensions. As illustrated in Fig. 5, our framework naturally extends to complex scenarios through geometric programming: Translating M enables the spatial arrangement of subjects thereby achieving layout-guided gen-

Benchmark Protocols. We establish three evaluation tiers: 1) Single-subject personalization, compared against 10 approaches spanning training-based [7, 28, 29, 38, 47, 54, 63,

Input Image Prompt Training-Based Training-Free

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

a dog on the beach

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

a pair of sunglasses on the wooden floor

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

a clock in the forest

DreamBooth IP-Adapter MS-Diffusion OminiControl ConsiStory Ours (Hunyuan) Ours (FLUX)

Figure 6. Qualitative comparisons on single-subject personalization. More results can be found in the supplementary materials.

Table 1. Quantitative results on single-subject personalization.

65, 68] and training-free [57] paradigms, 2) Multi-subject personalization, evaluated against 6 representative methods [5, 19, 24, 32, 38, 63], and 3) Subject-scene composition, benchmarked using AnyDoor [2] as reference for contextual adaptation.

Method CLIP-T↑ CLIP-I↑ DINO↑ DreamSim↓

DreamBooth [47] 0.271 0.819 0.550 0.290 BLIP-Diffusion [29] 0.251 0.835 0.641 0.283 IP-Adapter [65] 0.249 0.861 0.652 0.256 λ-ECLIPSE [38] 0.235 0.866 0.682 0.224 SSR-Encoder [68] 0.244 0.860 0.701 0.220 EZIGen [7] 0.263 0.825 0.662 0.247 MS-Diffusion [63] 0.283 0.824 0.539 0.261 OneDiffusion [28] 0.255 0.817 0.603 0.298 OminiControl [54] 0.275 0.820 0.516 0.301

Evaluation Metrics. We evaluate our Personalize Anything on DreamBench [47] which comprises 30 base objects each accompanied by 25 textual prompts. We extend this dataset to 750, 1000, and 100 test cases for single-subject, multi-subject, and subject-scene personalization using combinatorial rules. Quantitative assessment leverages multidimensional metrics: FID [16] for quality analysis, CLIPT [41] for image-text alignment, and DINO [35], CLIPI [41], DreamSim [9] for identity preservation in singlesubject evaluation while SegCLIP-I [71] in multi-subject evaluation. DreamSim [9] is a new metric for perceptual image similarity that bridges the gap between “low-level” metrics (e.g., PSNR, SSIM, LPIPS [66]) and “high-level” measures (e.g. CLIP [41]). SegCLIP-I is similar to CLIP-I, but all the subjects in source images are segmented.

ConsiStory [57] 0.284 0.753 0.472 0.434 Ours (HunyuanDiT [31]) 0.291 0.869 0.679 0.206 Ours (FLUX [27]) 0.307 0.876 0.683 0.179

ified subjects, without necessitating training or fine-tuning. Quantitative results in Tab. 1 confirms our excellent performance on identity preservation and image-text alignment.

Multi-Subject Personalization. From a qualitative perspective in Fig. 7, existing approaches [19, 24, 32, 63, 68] may suffer from conceptual fusion when generating multiple subjects, struggling to maintain their individual identities, or produce fragmented results due to incorrect modeling of inter-subject relationships. In contrast, our approach manages to maintain natural interactions among subjects via layout-guided generation, while ensuring each subject retains its identical characteristics and distinctiveness. Quantitatively, results in Tab. 2 demonstrate the strength of Personalize Anything in SegCLIP-I and CLIP-T, demonstrating that our approach not only effectively captures identity of multiple subjects but also excellently preserves the text control capabilities.

#### 4.2. Comparison to State-of-the-Arts

Single-Subject Personalization. Fig. 6 shows qualitative comparison with representative baseline methods. Existing test-time fine-tuning methods [47] require 30 GPUminute optimization for each concept and sometimes exhibit concept confusion for single-image inputs, manifesting as treating the background color as a characteristic of the subject. Training-based but test-time tuning-free methods [28, 54, 63, 65], despite trained on large datasets, struggle to preserve identity in detail for real image inputs. Training-free methods [57] generate inconsistent subjects with single-image input. In contrast, our method produces high-fidelity images that are highly consistent with the spec-

Subject-Scene Composition. We further evaluate our Personalize Anything on subject-scene composition, conduct-

Subject 1 Subject 2 Training-Based Training-Free

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

Cones2 SSR-Encoder Custom-Diffusion MS-Diffusion MIP-Adapter FreeCustom Ours (FLUX)

Figure 7. Qualitative comparisons on multi-subject personalization.

Subject Scene AnyDoor Ours (FLUX)

- Table 2. Quantitative results on multi-subject personalization.

Method CLIP-T↑ CLIP-I↑ SegCLIP-I↑

λ-ECLIPSE [38] 0.258 0.738 0.757 SSR-Encoder [68] 0.234 0.720 0.761 Cones2 [32] 0.255 0.747 0.702 Custom Diffusion [24] 0.228 0.727 0.781 MIP-Adapter [19] 0.276 0.765 0.751 MS-Diffusion [63] 0.278 0.780 0.748

FreeCustom [5] 0.248 0.749 0.734 Ours (HunyuanDiT) [31] 0.284 0.817 0.832 Ours (FLUX) 0.302 0.843 0.891

- Table 3. Ablation studies. We evaluate the effects of token replacement threshold τ and patch perturbation.

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

Figure 8. Qualitative results on subject-scene composition.

τ Pertur. CLIP-T↑ CLIP-I↑ DINO↑ DreamSim↓

T 0.317 0.764 0.625 0.305 0.95 T 0.313 0.773 0.632 0.294 0.90 T 0.306 0.849 0.680 0.199 0.80 T 0.302 0.882 0.741 0.163 0.70 T 0.282 0.920 0.769 0.140

ing comparison with Anydoor [2]. We show the visualization results in Fig. 8, where AnyDoor produces incoherent results, manifested as inconsistencies between the subject and environmental factors such as lighting in the generated images. In contrast, our method successfully generates natural images while effectively preserving the details of the subjects. It demonstrates the huge potentials and generalization capabilities of Personalize Anything in generating high-fidelity personalized images.

0.80 T 0.307 0.876 0.683 0.179

Effects of Threshold τ. Our systematic investigation of the timestep threshold τ reveals its critical role in balancing reference subject consistency and flexibility. As visualized in Fig. 9, early-to-mid replacement phases (τ > 0.8 T) progressively incorporate geometric and appearance priors from reference tokens, initially capturing coarse layouts (0.9 T) then refining color patterns and textures (0.8 T). Beyond τ = 0.8 T, late-stage color fusion dominates, pro-

#### 4.3. Ablation Study

We conduct ablation studies on single-subject personalization, examining the effects of token replacement timestep threshold τ and the patch perturbation strategy.

Input Image 𝜏 = 𝑇 𝜏 = 0.95 𝑇 𝜏 = 0.90 𝑇 𝜏 = 0.80 𝑇 𝜏 = 0.70 𝑇 𝝉 = 𝟎. 𝟖𝟎 𝑻 + Pertur.

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

[Figure 177]

Figure 9. Qualitative ablation studies on token replacement threshold τ and patch perturbation.

|[Figure 178]|
|---|

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

a boy eating bread a European-style castle a yacht sails

eating with Luffy

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

a western knight a mask on the ground a girl in a maple forest

a girl in a futuristic city

Figure 10. Visualization results of Personalize Anything in layout-guided generation (top), inpainting (middle), and outpainting (bottom).

ducing subjects almost completely identical to the reference subject (0.7 T). Quantitative results are shown in Tab. 3, where the balanced τ = 0.8 T achieves 0.882 reference similarity preservation (CLIP-I) while maintaining 0.302 image-text alignment (CLIP-T).

demonstrate capabilities of our framework on layout-guided personalization, and precise editing with mask conditions, all without architectural modification or fine-tuning.

### 5. Conclusion

Effects of Patch Perturbation. By combining local token shuffling and mask morphing, our perturbation strategy reduces both texture and structure overfitting. With τ = 0.8 T, the generated subject and the reference one are structurally similar without perturbation (Fig. 9). Conversely, applying perturbation makes the structure and texture more flexible while maintaining identically consistency.

This paper reveals that simple token replacement achieves high-fidelity subject reconstruction in diffusion transformers (DiTs), due to the position-disentangled representation in DiTs. The decoupling of semantic features and position enables the substitution of purely semantic tokens, avoiding positional interference. Based on this discovery, we propose Personalize Anything, a training-free framework that achieves high-fidelity personalized image generation through timestep-adaptive token injection and strategic patch perturbation. Our method eliminates per-subject optimization or large-scale training while delivering superior identity preservation and unprecedented scalability to layout-guided generation, multi-subject personalization and

#### 4.4. Applications

As illustrated in Fig. 5, our Personalize Anything naturally extends to diverse real-world applications, including subject-driven image generation with layout guidance, inpainting and outpainting. Visualization results in Fig. 10

mask-controlled editing. DiTs’ geometric programming establishes new paradigms for controllable synthesis, with spatial manipulation principles extensible to video/3D generation, redefining scalable customization in generative AI.

### References

- [1] Omri Avrahami, Amir Hertz, Yael Vinker, Moab Arar, Shlomi Fruchter, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski. The chosen one: Consistent characters in textto-image diffusion models. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers ’24, page 1–12. ACM, 2024. 2, 3
- [2] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6593–6602, 2024. 2, 3, 6, 7
- [3] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, et al. Unireal: Universal image generation and editing via learning real-world dynamics. arXiv preprint arXiv:2412.07774, 2024. 2, 3
- [4] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2, 3
- [5] Ganggui Ding, Canyu Zhao, Wen Wang, Zhen Yang, Zide Liu, Hao Chen, and Chunhua Shen. Freecustom: Tuningfree customized image generation for multi-concept composition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9089–9098,

2024. 2, 3, 4, 6, 7, 1

- [6] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3
- [7] Zicheng Duan, Yuxuan Ding, Chenhui Gou, Ziqin Zhou, Ethan Smith, and Lingqiao Liu. Ezigen: Enhancing zero-shot subject-driven image generation with precise subject encoding and decoupled guidance. arXiv preprint arXiv:2409.08091, 2024. 2, 3, 5, 6
- [8] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. 2, 3
- [9] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data, 2023. 6
- [10] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion, 2022. 3
- [11] Rinon Gal, Or Lichter, Elad Richardson, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Lcmlookahead for encoder-based text-to-image personalization.

In European Conference on Computer Vision, pages 322–

340. Springer, 2024. 2, 3

- [12] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 2, 3
- [13] Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, Peng Zhang, and Qian He. Pulid: Pure and lightning id customization via contrastive alignment. arXiv preprint arXiv:2404.16022, 2024. 2, 3
- [14] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning, 2023. 3
- [15] Junjie He, Yuxiang Tuo, Binghui Chen, Chongyang Zhong, Yifeng Geng, and Liefeng Bo. Anystory: Towards unified single and multiple subject personalization in text-to-image generation. arXiv preprint arXiv:2501.09503, 2025. 2, 3
- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium, 2018. 6
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 3
- [18] Jiehui Huang, Xiao Dong, Wenhui Song, Hanhui Li, Jun Zhou, Yuhao Cheng, Shutao Liao, Long Chen, Yiqiang Yan, Shengcai Liao, et al. Consistentid: Portrait generation with multimodal fine-grained identity preserving. arXiv preprint arXiv:2404.16771, 2024. 2, 3
- [19] Qihan Huang, Siming Fu, Jinlong Liu, Hao Jiang, Yipeng Yu, and Jie Song. Resolving multi-condition confusion for finetuning-free personalized image generation. arXiv preprint arXiv:2409.17920, 2024. 6, 7
- [20] Zehuan Huang, Hongxing Fan, Lipeng Wang, and Lu Sheng. From parts to whole: A unified reference framework for controllable human image generation, 2024. 2, 3
- [21] Junha Hyung, Jaeyo Shin, and Jaegul Choo. Magicapture: High-resolution multi-concept portrait customization. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2445–2453, 2024. 2, 3
- [22] Jiaxiu Jiang, Yabo Zhang, Kailai Feng, Xiaohe Wu, and Wangmeng Zuo. Mc2: Multi-concept guidance for customized multi-concept generation. arXiv preprint arXiv:2404.05268, 2024. 2, 3
- [23] Zhe Kong, Yong Zhang, Tianyu Yang, Tao Wang, Kaihao Zhang, Bizhu Wu, Guanying Chen, Wei Liu, and Wenhan Luo. Omg: Occlusion-friendly personalized multi-concept generation in diffusion models. In European Conference on Computer Vision, pages 253–270. Springer, 2024. 2, 3
- [24] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 2, 3, 6, 7
- [25] Gihyun Kwon and Jong Chul Ye. Tweediemix: Improving multi-concept fusion for diffusion-based image/video generation, 2025. 2, 3

- [26] Gihyun Kwon, Simon Jenni, Dingzeyu Li, Joon-Young Lee, Jong Chul Ye, and Fabian Caba Heilbron. Concept weaver: Enabling multi-concept fusion in text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8880–8889, 2024. 2, 3
- [27] Black Forest Labs. Flux. [Online], 2024. https:// github.com/black-forest-labs/flux. 2, 3, 5, 6
- [28] Duong H. Le, Tuan Pham, Sangho Lee, Christopher Clark, Aniruddha Kembhavi, Stephan Mandt, Ranjay Krishna, and Jiasen Lu. One diffusion to generate them all, 2024. 2, 3, 5, 6
- [29] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems, 36, 2024. 5, 6
- [30] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, MingMing Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8640–8650, 2024. 2, 3
- [31] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, Dayou Chen, Jiajun He, Jiahao Li, Wenyue Li, Chen Zhang, Rongwei Quan, Jianxiang Lu, Jiabin Huang, Xiaoyan Yuan, Xiaoxiao Zheng, Yixuan Li, Jihong Zhang, Chao Zhang, Meng Chen, Jie Liu, Zheng Fang, Weiyan Wang, Jinbao Xue, Yangyu Tao, Jianchen Zhu, Kai Liu, Sihuan Lin, Yifu Sun, Yun Li, Dongdong Wang, Mingtao Chen, Zhichao Hu, Xiao Xiao, Yan Chen, Yuhong Liu, Wei Liu, Di Wang, Yong Yang, Jie Jiang, and Qinglin Lu. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding, 2024. 5, 6, 7
- [32] Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones 2: Customizable image synthesis with multiple subjects. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 57500–57519, 2023. 2, 3, 6, 7
- [33] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subjectdiffusion: Open domain personalized text-to-image generation without test-time fine-tuning. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 2, 3
- [34] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models, 2022. 2, 3
- [35] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herv´e Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024. 6

- [36] Gaurav Parmar, Or Patashnik, Kuan-Chieh Wang, Daniil Ostashev, Srinivasa Narasimhan, Jun-Yan Zhu, Daniel CohenOr, and Kfir Aberman. Object-level visual prompts for compositional image generation. arXiv preprint arXiv:2501.01424, 2025. 2, 3
- [37] Or Patashnik, Rinon Gal, Daniil Ostashev, Sergey Tulyakov, Kfir Aberman, and Daniel Cohen-Or. Nested attention: Semantic-aware attention values for concept personalization. arXiv preprint arXiv:2501.01407, 2025.
- [38] Maitreya Patel, Sangmin Jung, Chitta Baral, and Yezhou Yang. λ-eclipse: Multi-concept personalized text-to-image diffusion models by leveraging clip latent space, 2024. 2, 3, 5, 6, 7
- [39] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 2, 3, 1

- [40] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 3
- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 6
- [42] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2, 3

- [43] Elad Richardson, Yuval Alaluf, Ali Mahdavi-Amiri, and Daniel Cohen-Or. pops: Photo-inspired diffusion operators. arXiv preprint arXiv:2406.01300, 2024. 2, 3
- [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 2, 3
- [45] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation,

2015. 2, 3

- [46] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. arXiv preprint arXiv:2410.10792, 2024. 4
- [47] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 2, 3, 5, 6
- [48] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 2, 3
- [49] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora:

- Any subject in any style by effectively merging loras. In European Conference on Computer Vision, pages 422–438. Springer, 2024. 2, 3
- [50] Chaehun Shin, Jooyoung Choi, Heeseung Kim, and Sungroh Yoon. Large-scale text-to-image model with inpainting is a zero-shot subject-driven image generator. arXiv preprint arXiv:2411.15466, 2024. 2, 3
- [51] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022. 3
- [52] Kunpeng Song, Yizhe Zhu, Bingchen Liu, Qing Yan, Ahmed Elgammal, and Xiao Yang. Moma: Multimodal llm adapter for fast personalized image generation. In European Conference on Computer Vision, pages 117–132. Springer, 2024. 2, 3
- [53] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 3

- [54] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 3, 2024. 2, 3, 5, 6
- [55] Kolors Team. Kolors: Effective training of diffusion model for photorealistic text-to-image synthesis. arXiv preprint,

2024. 2, 3

- [56] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization, 2024. 3
- [57] Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. Training-free consistent text-to-image generation. ACM Transactions on Graphics (TOG), 43(4):1–18, 2024. 2, 3, 4, 6, 1
- [58] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. P+: Extended textual conditioning in text-toimage generation, 2023. 3
- [59] Jinglong Wang, Xiawei Li, Jing Zhang, Qingyuan Xu, Qin Zhou, Qian Yu, Lu Sheng, and Dong Xu. Diffusion model is secretly a training-free open vocabulary semantic segmenter. arXiv preprint arXiv:2309.02773, 2023. 4
- [60] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024. 3, 4
- [61] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 2, 3
- [62] Qinghe Wang, Baolu Li, Xiaomin Li, Bing Cao, Liqian Ma, Huchuan Lu, and Xu Jia. Characterfactory: Sampling consistent characters with gans for diffusion models. arXiv preprint arXiv:2404.15677, 2024.
- [63] X. Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance, 2025. 3, 5, 6, 7
- [64] Peng Xing, Haofan Wang, Yanpeng Sun, Qixun Wang, Xu Bai, Hao Ai, Renyuan Huang, and Zechao Li. Csgo: Content-style composition in text-to-image generation, 2024.

- [65] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2, 3, 6

- [66] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric, 2018. 6
- [67] Shilong Zhang, Lianghua Huang, Xi Chen, Yifei Zhang, Zhi-Fan Wu, Yutong Feng, Wei Wang, Yujun Shen, Yu Liu, and Ping Luo. Flashface: Human image personalization with high-fidelity identity preservation. arXiv preprint arXiv:2403.17008, 2024. 2, 3
- [68] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8069–8078, 2024. 2, 3, 6, 7
- [69] Yuxin Zhang, Minyan Luo, Weiming Dong, Xiao Yang, Haibin Huang, Chongyang Ma, Oliver Deussen, Tong-Yee Lee, and Changsheng Xu. Bringing characters to new stories: Training-free theme-specific image generation via dynamic visual prompting. arXiv preprint arXiv:2501.15641,

2025. 2, 3

- [70] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent selfattention for long-range image and video generation. arXiv preprint arXiv:2405.01434, 2024. 2, 3
- [71] Chenyang Zhu, Kai Li, Yue Ma, Chunming He, and Li Xiu. Multibooth: Towards generating all your concepts in an image from text. arXiv preprint arXiv:2404.14239, 2024. 2, 3, 6
- [72] Zhuofan Zong, Dongzhi Jiang, Bingqi Ma, Guanglu Song, Hao Shao, Dazhong Shen, Yu Liu, and Hongsheng Li. Easyref: Omni-generalized group image reference for diffusion models via multimodal llm. arXiv preprint arXiv:2412.09618, 2024. 2, 3

## Personalize Anything for Free with Diffusion Transformer Supplementary Material

### 6. Analysis of Position Sensitivity in DiT

We employ the attention sharing mechanism in existing UNet-based subject personalization methods [5, 57] to diffusion transformers (DiTs) [39]. Specifically, we concatenate position-encoded denoising tokens X˙ and reference tokens X˙ref into a single sequence [X˙ ;Xref˙ ] and apply pretrained multi-modal attention on it. In this process, both denoising and reference tokens retain their original positions (i,j) ∈ [0,w) × [0,h), causing destructive interference to attention computation and producing ghosting artifacts of the reference subject on the generated image. For quantitative analysis, we calculate the denoising tokens’ attention scores on the reference tokens at the same positions. By averaging these scores across 100 samples, we obtain a mean value of 0.4294. Subsequently, we apply the same procedure to U-Net, which encodes positional information by convolution layers instead of explicit positional encoding, yielding an average attention score of 0.0522. Comparatively, the average attention score in DiT is 723% higher than that in U-Net. The above quantitative analysis demonstrates the position sensitivity in DiT, where explicit positional encoding significantly influences the attention mechanism, contrasting with U-Net’s implicit position handling.

### 7. User Study

To further evaluate model performance, we conducted a user study involving 48 participants, with ages evenly distributed between 15 and 60 years. Each participant was asked to answer 15 questions, resulting in a total of 720 valid responses. For single-subject and multi-subject personalization tasks, users are required to select the optimal model based on three dimensions: textual alignment, identity preservation, and image quality. In subject-scene composition tasks, we substituted textual alignment with scene consistency to assess subject-scene coordination. The results are presented in Figs. 11 to 13, which further corroborates our qualitative findings, as our method outperforms other state-of-the-art methods across all metrics.

### 8. More Results

We present more qualitative comparison results in Fig. 14. As demonstrated, our method consistently achieves exceptional textual alignment and subject consistency. Specifically, our approach excels in preserving fine-grained details of subjects (e.g. the number “3” on the clock) while maintaining high fidelity to the textual description.

100

OneDiffusion OminiControl Ours

70

63

PreferenceRate(%)

44

50

35

21 19 18

18

12

0

Textual Alignment Identity Preservation Image Quality

Figure 11. User study results on single-subject personalization.

100

MIP-Adapter MS-Diffusion Ours

75

Textual Alignment Identity Preservation Image Quality PreferenceRate(%)

47

42

50

39

33

19 20

14

11

0

Figure 12. User study results on multi-subject personalization.

100

AnyDoor Ours

73

PreferenceRate(%)

66

56

44

50

34

27

0

Scene Consistency Identity Preservation Image Quality

Figure 13. User study results on subject-scene composition.

Input Image Prompt EZIGen MS-Diffusion OneDiffusion OminiControl ConsiStory

###### Ours (HunyuanDiT)

Ours (FLUX)

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

a dog on the beach

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

a pair of sunglasses on the wooden floor

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

a clock in the forest

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

a boot on the grass

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

a monster toy on the floor

Ours (HunyuanDiT) Ours (FLUX)

Input Image Prompt DreamBooth IP-Adapter SSR-Encoder BLIP-Diffusion λ-ECLIPSE

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

a dog on the beach

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

a pair of sunglasses on the wooden floor

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

a clock in the forest

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

a boot on the grass

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

a monster toy on the floor

Figure 14. Full qualitative comparisons on single-subject personalization.

