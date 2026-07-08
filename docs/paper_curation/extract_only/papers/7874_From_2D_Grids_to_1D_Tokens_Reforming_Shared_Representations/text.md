## From 2D Grids to 1D Tokens: Reforming Shared Representations for Multimodal Image Fusion

Yuchen Xian12 Yunqiu Xu3 Yang He45 Yi Yang12

### Abstract

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Base (Global appearance): distributed / broadcast across spatial positions

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Multimodal image fusion aims to integrate complementary information from different modalities into a fused image that preserves rich local details while maintaining globally consistent appearance. Existing approaches build shared representations on 2D feature grids, which excel at modeling local structures but offer limited leverage over image-level global appearance factors. To balance these objectives, we introduce a compact 1D token interface based on a frozen pretrained image tokenizer for modeling non-local appearance/base factors. Rather than using the tokenizer as a reconstruction backbone, our design uses the 1D token space as a global carrier while retaining the 2D spatial pathway for local structure restoration. Specifically, we introduce Selective Token Editing (STE), which sparsely updates/replaces a small set of critical tokens, providing a lightweight mechanism to steer global appearance coherence while keeping the fusion backbone unchanged and avoiding extra losses. Experiments on four commonly used benchmarks show that our method achieves the best overall performance, with consistent, multi-metric improvements in both global coherence and local fidelity. Project page: https://zju-xyc.github. io/1D-Fusion-Project-Page/

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

# arXiv:2606.12303v1[cs.CV]10Jun2026

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

𝑰𝒇 = 𝑫 𝑭 𝐹 = 𝐸 (𝐼 ,𝐼 )

Hard to regulate global appearance: base is spatially entangled

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

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Detail (Local Structures):

Shared 2D Feature Map 𝑭 ∈ ℝ × ×  localized High-frequency cues

- (a) Conventional 2D Shared Grid Representation

Token-to-Map interface 𝜋(⋅) Base Map

𝑩 = 𝜋(𝒁) ∈ ℝ𝑯×𝑾×𝑪

+

Detail Map Local Structures

- (b) Proposed 1D Token Shared Representation

Shared 1D Tokens 𝒁 = {𝒛𝒊}𝒊 𝟏𝑵 ∈ ℝ ×𝒅

Base is centralized in a few tokens → easier to steer.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Detail remains spatial for edge/texture fidelity.

Base Tokens Aux / Residual Tokens Global Appearance Carrier

Decoupled control: edit base without disrupting detail

[Figure 63]

[Figure 64]

[Figure 65]

𝑰𝒇 = 𝑫𝒆𝒄𝒐𝒅𝒆𝒓( 𝑭)

[Figure 66]

𝑭 = 𝑩 + 𝑫

Figure 1. 2D grids vs. 1D tokens for base/detail decoupling in multimodal image fusion. (a) A 2D shared feature map entangles global base appearance with local details. (b) We represent base in a compact 1D token set Z, map it to a base map via π(·), and combine it with a spatial detail map D for decoding.

serving structural backgrounds, target saliency, fine-grained textures, and coherent global appearance. For instance, infrared–visible image fusion (Li & Wu, 2019; Zhao et al., 2023a; Liu et al., 2025b; Guan et al., 2026) seeks to highlight thermally salient targets from infrared imagery while retaining the rich textures and semantic readability of visible images, thereby alleviating practical imaging challenges such as low illumination, noise contamination, and limited resolution. Moreover, fused images often serve as intermediate results for downstream perception tasks, and their quality directly impacts the stability and robustness of subsequent systems such as object detection and semantic segmentation (Bai et al., 2025; Zhang et al., 2024b; Zhao et al., 2023a; Li et al., 2023b).

### 1. Introduction

Multimodal image fusion (MMIF) (Zhao et al., 2024; Li et al., 2024; Liu et al., 2024a; Li et al., 2026) aims to integrate complementary information from different sensors or imaging mechanisms into a single output image, while pre-

1ReLER, The State Key Lab of Brain Machine Intelligence, Zhejiang University 2College of Artificial Intelligence, Zhejiang University 3National University of Singapore 4CFAR, Agency for Science, Technology and Research, Singapore 5IHPC, Agency for Science, Technology and Research, Singapore. Correspondence to: Yi Yang <yangyics@zju.edu.cn>.

Despite significant advances, most existing methods (Zhao et al., 2023b; Li & Wu, 2019; Ma et al., 2022) encode input pixels/patches into a shared dense 2D feature grid. We argue that the main bottleneck lies in this 2D-centric shared representation: image-level appearance factors (e.g., illumi-

Preprint. June 11, 2026.

nation, contrast, perceptual tone) are not naturally indexed by spatial coordinates, and thus are only captured implicitly through spatially broadcast patterns across many locations. This inevitably preserves spatial redundancy and entangles global base attributes with local textures, modality-specific cues, and residual noise, making global appearance alignment difficult to regulate and often resulting in brightness inconsistency, blurred details, or amplified artifacts.

From a separable perspective, a dense 2D shared grid is structurally mismatched with decoupling global base (appearance) from local detail, as illustrated Figure 1. Global factors must be spatially replicated and are therefore easily entangled with high-frequency edges, modality-specific cues, and residual noise. As a result, regulating image-level appearance in a 2D grid requires coordinated changes across many spatial locations, which makes appearance control statistically inefficient and optimization-sensitive, especially under distribution shifts.

Motivated by this observation, we explicitly decouple global appearance control from local detail restoration. Inspired by compact image tokenization (Yu et al., 2024; Zheng et al., 2025), we use a 1D token space as a non-spatial interface for appearance/base factors. Our claim is not that a 1D tokenizer is a universally superior reconstruction backbone or the strongest semantic encoder; rather, we exploit its compact and controllable organization to regulate non-local factors (e.g., illumination, contrast and perceptual tone), while leaving spatial details to the 2D fusion pathway. To make this interface actionable, we further introduce Selective Token Editing (STE), which identifies a small set of appearance-sensitive token entries and applies learnable edits to them. These edited slots are configuration-specific positions discovered through probing/selection in the 1D token space, rather than hand-crafted semantic indices.

We construct a lightweight hybrid multimodal image fusion framework that couples a frozen pretrained 1D tokenizer with a conventional 2D fusion backbone. The 1D tokenizer is used to construct a compact appearance/base carrier, while the 2D pathway remains responsible for spatially localized textures, edges, and structural details. The 1D tokens are mapped back to the spatial domain through a lightweight token-to-map interface and injected into the 2D fusion process as global guidance. This division of labor allows the model to regulate illumination and perceptual tone without forcing the compact token branch to reconstruct all high-frequency details, thereby improving the balance between global coherence and local fidelity.

We perform experiments on infrared-visible and medical image fusion tasks as well as two downstream applications (i.e., object detection and semantic segmentation). Extensive experimental results demonstrate that the proposed 1D token interface benefits multimodal image fusion by providing

a compact handle for regulating global appearance while preserving local structural fidelity. In summary, the main contributions of this paper are as follows:

- (i) We present a new perspective on multimodal image fusion by revisiting the carrier of shared appearance information. We introduce 1D tokenizer as a compact appearance/base interface that complements 2D spatial fusion and improves the regulation of illumination, contrast and perceptual tone.
- (ii) We propose a lightweight hybrid fusion paradigm that uses a frozen pretrained 1D tokenizer as a controllable global interface and preserves the 2D fusion backbone for local detail modeling. Fusion quality is improved by selectively editing only a small subset of appearance-sensitive token dimensions, without introducing complex loss designs.
- (iii) We provide empirical evidence that simple tokenlevel intervention in a one-dimensional representation space leads to consistent improvements in fusion quality, demonstrating enhanced illumination consistency, sharper details, and reduced visual artifacts across multiple fusion benchmarks.

### 2. Related Work

Compressed Tokenizers. Image tokenizers can be broadly categorized by whether they rely on a spatially dense 2D grid or a compact 1D sequence. 2D-grid tokenizers typically quantize VAE-style latents and decode images in a pixelwise manner (e.g., VQ-VAE/VQGAN) (Van Den Oord et al., 2017; Esser et al., 2021; Rombach et al., 2022). While effective for reconstruction, dense grids inherit strong locality bias (Wang & Wu, 2023) and require many tokens, making global appearance editing expensive (Yu et al., 2024). Recent 1D-sequence tokenizers instead represent an image with a small set of tokens without a dense spatial grid (e.g., TiTok, FlexTok), enabling extreme compression and more localized global-factor control in the token space (Yu et al., 2024; Bachmann et al., 2025; Beyer et al., 2025). Different from existing works, we treat the tokenizer as a fixed interface and study how to selectively manipulate a few tokens/channels to improve downstream fusion quality.

Multimodal Image Fusion. Early MMIF methods (Li & Wu, 2019; Zhao et al., 2023b; Li et al., 2023a; Wang et al.,

- 2025) fuse modalities on dense 2D feature maps, using CNN-style encoders and decoders to preserve local textures. To better capture long-range dependencies, several subsequent works (Qu et al., 2022; Yi et al., 2024; Liu et al.,
- 2026) introduce transformer-style interactions or auxiliary guidance. However, even with stronger context modeling, fusion is still performed on dense 2D maps where global appearance cues remain entangled with local details and are difficult to localize and regulate explicitly. In contrast,

fusion in a compact 1D shared space is less explored, yet it offers a natural handle for global appearance control with only a small number of tokens (Yu et al., 2024). Here, our method is established by coupling a 1D token-based shared representation with selective token editing, enabling controllable regulation of global appearance while remaining compatible with standard 2D fusion.

### 3. Rethinking 2D Shared Representations for Multimodal Image Fusion

##### 3.1. Locality-Biased Nature of 2D Representations

Most existing multimodal image fusion methods (Li & Wu, 2019; Zhao et al., 2020; Xu et al., 2020a) follow an encoder– fusion–decoder paradigm. Let IV and II denote the spatially aligned inputs from the visible modality V and the infrared modality I, respectively. An encoder Eθ(·) maps each input to a shared feature representation:

###### F(m) = Eθ(I(m)), (1)

where F(m) ∈ Rh×w×d indicates dense 2D feature grids for modality m ∈ {V,I}. Specifically, F(ijm) ∈ Rd denotes the feature vector at spatial location (i,j), with i ∈ {1,...,h} and j ∈ {1,...,w}.

A fusion module F(·) aggregates the modality-specific features in the grid space,

Ff = F FV,FI , (2)

and a decoder Dϕ(·) reconstructs the fused image If = Dϕ(Ff). A key characteristic of this pipeline is that the shared representation is explicitly parameterized as a dense 2D grid, and information interaction is primarily performed in a location-wise or neighborhood-wise manner.

In image fusion (Li & Wu, 2019; Zhao et al., 2020), base refers to image-level attributes such as overall brightness, contrast, and global perceptual tone, whereas detail corresponds to spatially varying high-frequency structures. These two factors differ fundamentally in semantic scale: detail is inherently tied to specific spatial locations, while base acts as a low-dimensional factor shared across the entire image and is not naturally associated with any coordinate.

However, in a 2D grid representation, all information is distributed across spatial locations in the form of Fij. Even when base-related information is encoded, it can only exist implicitly through a distributed encoding over multiple spatial positions, rather than as a compact and independent variable. Consequently, when fusion requires not only preserving detail but also aligning or adjusting base, the representation itself already exhibits a structural mismatch.

##### 3.2. Unstable Base Modeling in 2D Feature Space

To analyze the structural properties of base modeling in 2D shared representations R, we introduce a latent-factor abstraction and express an input image from modality m ∈ {V,I} as

I(m) = R base(m),detail(m) , (3)

where base(m) denotes the image-level base factor and detail(m) denotes the spatially varying detail factor.

In a 2D shared representation, the encoded feature at spatial location (i,j) can be abstractly expressed as

F(ijm) = ϕ detail(ijm) + Abase(m) + ϵij, (4)

where ϕ(·) denotes a local encoding function for detail, A is a shared linear operator that maps the base factor into the feature space and broadcasts it to spatial locations, and ϵij represents location-dependent residual terms. This formulation highlights a property: in 2D grids, base does not exist as an independent variable, but is inevitably entangled with detail and residual variations through spatial broadcasting.

Thus, estimating or aligning base(m) requires aggregating information from a high-dimensional and spatially varying feature field. Estimating a low-dimensional global factor from a high-dimensional spatial field is inherently unstable.

This distributed parameterization leads to two major consequences. From a statistical perspective, base estimation depends on aggregating features across many spatial locations and is therefore sensitive to location-dependent residuals, resulting in statistical inefficiency. From an optimization perspective, global base adjustment must be realized through coordinated changes across a high-dimensional feature grid, which induces a typical many-to-one inverse problem and yields ill-conditioned optimization behavior during decoding. We provide theoretical discussion on the control geometry of 2D grids and 1D tokens in Appendix §A.

### 4. Methodology

The analysis in §3 reveals that conventional MMIF methods, which rely on a shared 2D feature grid, are inherently biased toward modeling detail, while lacking a stable and compact carrier for image-level base. Based on this observation, we redesign the shared representation of MMIF at the methodological level.

As illustrated in Figure 2, the proposed framework takes aligned multimodal inputs, extracts compact 1D token representations through a frozen tokenizer, maps them back to token-induced 2D feature maps, and performs factorized base/detail fusion followed by residual decoding to generate

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

###### A. Stage I: Reconstruction Warm-up

###### B. Stage II (Fusion Training)

[Figure 71]

[Figure 72]

Residual Reference

Residual Reference

[Figure 73]

Encoder

[Figure 74]

|| |
|---|
<br><br>...| |
|---|---|
| | |

[Figure 75]

[Figure 76]

Base Encoder

[Figure 77]

[Figure 78]

| |
|---|

| |
|---|

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

(Transformer)

......

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

|[Figure 93]|
|---|

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Decoder

Fusion Layer Detail Fusion Layer 𝓕detail ⋅

[Figure 98]

[Figure 99]

Token-to-MapMapping

[Figure 100]

C

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

𝑫𝝓 ⋅

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Detail Encoder

DiscreteTokens

[Figure 109]

| |
|---|

| |
|---|

SharedEncoder

(Reconstruction)

[Figure 110]

1DTokenizer

[Figure 111]

(INN)

###### Decoder

|[Figure 112]|
|---|

Encoder

C

[Figure 113]

[Figure 114]

[Figure 115]

|...| |
|---|---|
| | |

Base Fusion Layer

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

𝑰𝒇

Detail Encoder

(Fuse)

[Figure 121]

[Figure 122]

(Reconstruction)

| |
|---|

| |
|---|

[Figure 123]

[Figure 124]

𝓕base ⋅

[Figure 125]

[Figure 126]

(INN)

[Figure 127]

[Figure 128]

|[Figure 129]|
|---|

[Figure 130]

[Figure 131]

[Figure 132]

Decoder C

[Figure 133]

𝝉 ⋅

[Figure 134]

[Figure 135]

[Figure 136]

𝝅 ⋅

𝑫𝝓 ⋅

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Base Encoder

| |
|---|

| |
|---|

(Transformer)

Private Encoder 𝑬pri ⋅

Residual Reference

Residual Reference

[Figure 141]

[Figure 142]

Frozen Trainable C Channel Concatenation 𝝆 ⋅ Channel Projection Element-Wise Summation

- Figure 2. Two-stage training scheme of our multimodal image fusion framework. The key design is to introduce a compact 1D token interface for global appearance/base modeling, while preserving the 2D pathway for local detail reconstruction. (A) Stage I (Reconstruction Warm-Up): the model learns stable modality-specific base/detail representations through per-modality reconstruction. (B) Stage II (Fusion Training): the base and detail fusion modules are activated to perform cross-modal fusion and generate the final fused output.

the final fused image. Specifically, we replace the conventional 2D feature grid with a compact 1D token sequence as the shared representation carrier. Crucially, this change is confined to the representation layer and does not disrupt the advantages of 2D fusion backbones in modeling detail.

2025). This motivates Selective Token Editing (STE), which edits only a sparse subset of token entries to improve sharpness and appearance consistency without disturbing the core semantics.

Let the shared token sequence be Z ∈ RK×C with K=32 and C=12. Instead of manually assigning editing positions, we identify appearance-sensitive slots through an offline Gumbel-Softmax probing step. For each editing slot s, we maintain selector logits as ∈ RK and compute

- 4.1. From 2D Grids to 1D Tokens: Redesigning the Shared Representation

As illustrated in §3, base is a low-dimensional, non-spatially indexed image-level semantic factor, whereas a 2D grid is a high-dimensional representation indexed by spatial locations. The mismatch in scale and parameterization between the two is the fundamental reason for unstable base modeling. To address this issue, we introduce a 1D token sequence as a new form of shared representation. For an input image I(m) from modality m ∈ {V,I}, we employ a 1D image tokenizer τ(·) to map it to

as + gs τg

, gs ∼ Gumbel(0,1), (6)

ys = softmax

where as ∈ RK denotes the selector logits for slot s, τg is the Gumbel temperature, and gs is sampled from the standard Gumbel distribution. The selected token position is obtained by

ys,k. (7)

ps = arg max

k

Z(m) = τ I(m) , (5) where Z(m) ∈ RN×d

With the tokenizer and subsequent modules frozen, we perturb the selected positions and evaluate the fusion outputs using edge intensity (EI) (Rajalingam & Priya, 2018), average gradient (AG) (Cui et al., 2015), spatial frequency (SF) (Eskicioglu & Fisher, 1995), and structural similarity (SSIM) (Wang et al., 2004), which jointly reflect edge clarity, texture variation, and structural preservation. Under the TiTok-32 configuration, this probing process consistently identifies positions 12 and 18 as the most effective appearance-sensitive slots, and channels {6,7,8} as the most stable editing group.

t denotes a token sequence of length N, and each token is a dt-dimensional vector.

Here, Z(m) it serves as a compact appearance/base interface that complements the spatial fusion pathway. Unlike a dense 2D grid, where base-related factors are implicitly broadcast across many locations, the 1D token space provides a nonspatial carrier through which global factors can be accessed and regulated with fewer degrees of freedom. In this work, we instantiate this interface with a frozen pretrained TiTok tokenizer, while relying on the subsequent token-to-map interface and 2D fusion modules to recover spatial details.

After identifying these positions and channels, STE replaces manual perturbations with a compact learnable offset:

Selective Token Editing. In a highly-compressed 1D representation with K=32 tokens, not every token contributes equally to the final image. The main content is typically robust to local token perturbations, while global appearance attributes concentrate in a few token positions (Beyer et al.,

###### Z = Z + M ⊙ ∆, (8)

where the binary mask M activates only channels {6,7,8} at positions {12,18} and is zero elsewhere, and ∆ ∈ RK×C

is a learnable bias. Since STE modifies only a few tokenchannel entries, it provides a lightweight plug-in for appearance regulation while remaining compatible with the subsequent token-to-map interface and arbitrary 2D fusion backbones. The positions 12 and 18 are configuration-specific slots discovered under TiTok-32, rather than universal semantic labels of the tokenizer.

- 4.2. Token-to-Map Interface: Bridging 1D Tokens and 2D Fusion Backbones

Although 1D tokens are better suited for carrying imagelevel global semantics (base), most mature MMIF fusion modules are still built upon 2D feature maps to model spatially localized structures (detail). To stay compatible with this 2D ecosystem, we introduce a token-to-map interface π(·) that adapts the token sequence of modality m ∈ {V,I}, Z(m), into a token-induced 2D feature map:

###### Fˆ(m) = π Z(m) . (9)

Here Fˆ(m) ∈ Rh×w×d is distinguished from the conventional 2D feature grid produced directly by a standard encoder. The role of π(·) is representation adaptation: global semantics remain primarily concentrated in the token space, while the 2D map serves only as an operational substrate for local processing. More module-level architecture and implementation details are provided in Appendix §B.

In practice, π(·) follows a hierarchical mapping design. We first lift tokens from dimension 12 to 64, then use a linear mapping to obtain a 32×32 coarse feature map, augmented with a residual local aggregation branch implemented by a 3×3 convolution with a learnable scaling coefficient. We then apply three-stage upsampling to recover a 256×256 map, and at each upsampling stage, we merge scale-aligned detail features extracted from the original image, using convolution kernels of 7×7, 5×5, and 3×3, respectively. This design suppresses early structured artifacts while avoiding over-smoothed outputs caused by pure upsampling.

Built upon the token-induced feature maps, we explicitly implement factorized modeling of base and detail. For each modality, a private encoder Epri(·) decomposes Fˆ(m) into

B(m),D(m) = Epri F ˆ(m) , (10)

where B(m) captures low-frequency, globally consistent appearance (base), and D(m) preserves high-frequency, spatially localized structures (detail). Fusion is then performed separately in the two subspaces:

Bf = Fbase BV,BI , Df = Fdetail DV,DI . (11)

This factorized fusion strategy decouples base alignment from detail preservation, alleviating the instability caused by appearance–detail entanglement in conventional 2D grids.

Algorithm 1 Two-Stage Training (Frozen Tokenizer)

- 1: Input: training set Dtrain, switch epoch Egap, total epochs Emax
- 2: Output: trained parameters Θ (tokenizer frozen)
- 3: Freeze pretrained tokenizer τ.
- 4: for e = 0 to Emax − 1 do
- 5: for I(m) ∼ Dtrain do (m ∈ {1,2})
- 6: if e < Egap then ▷ Stage I
- 7: (B(m),D(m)) := E I(m)
- 8: Iˆ(m) := Dϕ B(m),D(m); I(m)
- 9: Compute LI; backpropagate and update Θ.
- 10: else ▷ Stage II
- 11: (B(m),D(m)) := E I(m)
- 12: Bf := Fbase B(1),B(2)
- 13: Df := Fdetail D(1),D(2)
- 14: If := Dϕ Bf,Df; I(1) + I(2)
- 15: Compute LII; backpropagate and update Θ.
- 16: end if
- 17: end for
- 18: end for

Finally, we adopt residual reconstruction to stabilize the output. We define the reference input as the element-wise sum of the aligned inputs:

###### Iref = IV + II, (12)

and predict a residual ∆I to obtain the final fused image:

∆I = Dϕ ρ [Bf,Df] ; Iref , If = Iref + ∆I, (13)

where ρ(·) is a channel projection operator and Dϕ(·) denotes the decoder.

##### 4.3. Training Strategy and Optimization Details

To ensure that the shared representation based on 1D tokens can stably support subsequent factorized fusion and decoding, we adopt a two-stage training strategy, as summarized in Algorithm 1, and keep the tokenizer τ(·) frozen throughout the entire training process. This design prevents the shared representation from drifting during optimization, thereby ensuring that base is consistently carried by the compact token space.

Stage I: Intra-Modality Reconstruction and Factorization Stabilization. In the first stage, cross-modality fusion is disabled, and the model learns intra-modality reconstruction with stable base/detail factorization. For each modality m ∈ {V,I}, the input I(m) is mapped to token representation Z(m), converted into a feature map Fˆ(m), decomposed into B(m),D(m) , and reconstructed as Iˆ(m).

|[Figure 143]<br><br>[Figure 144]<br><br>Ours<br><br>|
|---|

|[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>|
|---|

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

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

#### 3FDM

Input CDDFuse SAGE EMMA DCEvo Text-DiFuse Ours

|[Figure 164]<br><br>[Figure 165]<br><br>Ours<br><br>|
|---|

|[Figure 166]<br><br>[Figure 167]<br><br>|
|---|

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

#### RoadScene

[Figure 182]

[Figure 183]

Input CDDFuse SAGE EMMA DCEvo Text-DiFuse Ours

|[Figure 184]<br><br>[Figure 185]<br><br>Ours<br><br>|
|---|

|[Figure 186]<br><br>[Figure 187]<br><br>|
|---|

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

#### TNO

Input CDDFuse SAGE EMMA DCEvo Text-DiFuse Ours

|[Figure 204]<br><br>[Figure 205]<br><br>Ours<br><br>|
|---|

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

#### Harvard

Input CDDFuse SAGE EMMA DCEvo Text-DiFuse Ours

- Figure 3. Qualitative comparisons on M3FD, RoadScene, TNO and Harvard datasets. Benefiting from the concentrated semantic information encoded in our 1D token representation, our method produces fused images with enhanced global coherence and sharper local structures compared to existing methods.

The reconstruction loss is defined as

2 2

L(recm) = αssim LSSIM I(m),Iˆ(m) +αmse I(m) − Iˆ(m)

,

(14) where αssim and αmse are balancing weights for the SSIM and pixel-wise ℓ2 terms; the former preserves structural consistency, while the latter stabilizes reconstruction.

To encourage complementarity between modalities in the base and detail dimensions, we introduce a decomposition regularization term:

cc DV,DI 2 δ + cc(BV,BI)

, (15)

Ldecomp =

where cc(·,·) denotes the correlation coefficient and δ is a small constant for numerical stability. This term encourages diversity in detail while enforcing consistency in base. The

- Stage I objective is

LI =

m∈{V,I}

λm L(recm) + λd Ldecomp, (16)

where λm denotes the reconstruction weight for modality m, and λd controls the decomposition regularization term.

- Stage II: Cross-Modality Fusion Training. In the second stage, the base and detail fusion modules are activated. The decomposed representations BV,DV and BI,DI are

fused separately to obtain Bf,Df , which are decoded into the final fused image If. The decomposition regularizer is retained to maintain factorization consistency.

The fusion loss is defined as

Lfusion = αin Imax − If 1 + αg Gmax − ∇If 1 ,

(17) where Imax and Gmax are obtained by taking the elementwise maximum of the input intensity maps and gradient maps, respectively, ∇(·) denotes the gradient operator, and αint and αgrad control the intensity and gradient terms. The Stage II objective is

LII = λf Lfusion + λd Ldecomp, (18)

where λf and λd respectively balance the fusion loss and the decomposition regularizer.

### 5. Experiments

##### 5.1. Experimental Setting

Setup and Metrics. We evaluate our method on two multimodal image fusion tasks: infrared-visible image fusion (IVIF) and medical image fusion (MIF). For IVIF tasks, we trained on MSRS (Tang et al., 2022) with 1083 aligned infrared-visible image pairs and 50 pairs in RoadScene (Xu et al., 2020b) for validation. For fusion quality evaluation, we test on three benchmarks: M3FD (Liu et al., 2022)

- Table 1. Quantitative comparisons of fusion metrics on M3FD, RoadScene, TNO, and Harvard datasets. The best results are highlighted in bold, and the second-best results are underlined.

Methods

M3FD Dataset RoadScene Dataset TNO Dataset Harvard Dataset EN SD SCD SSIM EN SD SCD SSIM EN SD SCD SSIM EN SD SCD SSIM

CDDFuse 6.80 35.22 1.62 1.02 7.52 57.62 1.70 0.99 7.17 48.49 1.73 1.08 4.68 59.14 1.26 1.21 DDFM 6.76 30.95 1.64 0.93 7.31 45.75 0.90 0.09 7.04 41.12 1.43 0.36 3.55 56.34 1.53 1.42 LRRNet 6.36 25.40 1.38 0.78 7.14 42.65 1.47 0.68 7.11 44.67 1.42 0.86 4.73 40.62 0.48 0.23 Text-IF 6.77 33.29 1.47 0.97 7.40 50.42 1.52 0.97 7.25 48.58 1.67 1.00 4.73 53.46 1.01 0.40 TC-MoA 6.70 32.30 1.34 0.98 7.36 47.22 1.35 0.99 7.10 42.41 1.46 1.05 4.90 60.33 1.51 0.99 EMMA 6.78 35.12 1.45 0.92 7.52 56.26 1.65 0.94 7.27 48.92 1.71 1.01 4.17 62.99 1.67 0.67 SAGE 6.78 35.69 1.65 1.00 7.11 46.29 1.44 0.94 7.11 46.32 1.61 1.03 4.59 51.73 0.79 0.37 DCEvo 6.72 33.48 1.44 1.02 7.23 45.41 1.38 1.05 7.02 41.26 1.48 1.12 4.29 55.74 1.10 0.41 Text-DiFuse 6.93 39.87 1.16 1.22 7.16 47.37 0.94 1.00 7.28 50.04 1.41 0.35 5.42 71.32 1.52 0.25 Ours 7.19 47.35 1.85 1.49 7.56 56.26 1.82 1.45 7.34 50.97 1.82 1.42 4.76 70.86 1.76 1.45

- Table 2. Quantitative comparisons of downstream tasks with other state-of-the-art fusion methods for object detection on M3FD and semantic segmentation on FMB dataset. The best results are highlighted in bold, and the second-best results are underlined.

M3FD Dataset (Object Detection) FMB Dataset (Semantic Segmentation)

Methods

People Car Bus Light Moto Trunk mAP Building Light Sign People Bus Pole mIoU

CDDFuse 0.284 0.495 0.613 0.107 0.157 0.419 0.346 0.860 0.346 0.660 0.649 0.829 0.425 0.684 DDFM 0.286 0.506 0.621 0.113 0.148 0.424 0.350 0.868 0.377 0.686 0.659 0.807 0.434 0.691 LRRNet 0.276 0.501 0.629 0.111 0.154 0.422 0.349 0.863 0.350 0.677 0.643 0.833 0.423 0.688 Text-IF 0.287 0.501 0.624 0.121 0.170 0.416 0.353 0.862 0.318 0.680 0.658 0.796 0.424 0.684 TC-MoA 0.288 0.501 0.624 0.122 0.155 0.405 0.349 0.866 0.366 0.703 0.646 0.817 0.415 0.687 EMMA 0.276 0.496 0.624 0.118 0.147 0.398 0.343 0.862 0.343 0.668 0.652 0.827 0.417 0.691 SAGE 0.286 0.500 0.622 0.109 0.136 0.403 0.343 0.865 0.353 0.689 0.658 0.794 0.421 0.692 DCEvo 0.289 0.498 0.618 0.117 0.147 0.398 0.344 0.866 0.349 0.654 0.650 0.800 0.421 0.687 Text-DiFuse 0.263 0.480 0.602 0.082 0.127 0.423 0.329 0.859 0.316 0.687 0.661 0.825 0.413 0.684 Ours 0.301 0.524 0.640 0.124 0.145 0.428 0.360 0.868 0.377 0.708 0.662 0.831 0.428 0.692

(202 pairs), RoadScene (152 pairs), and TNO (Toet, 2017) (30 pairs). To assess the preservation of global semantics, we further evaluate on downstream tasks: object detection on M3FD and semantic segmentation on FMB (Liu et al.,

- 2023). For MIF tasks, we trained on the Harvard Medical Image Dataset (Johnson & Becker) with 200 images pairs for training, 50 pairs for validation, and 55 pairs for testing.

For quantitative evaluation, we adopt metrics from IVIFZOO (Liu et al., 2024b): Entropy (EN), Standard Deviation (SD), Sum of Difference Correlation (SCD), Edge Intensity (EI), Spatial Frequency (SF), Average Gradient (AG) and Structural Similarity (SSIM). For downstream tasks, we use YOLOv8s (Varghese & Sambath, 2024) with mAP50:95 for object detection and SegFormer-B1 (Xie et al., 2021) with mIoU for semantic segmentation. In terms of downstream IVIF application, input images are resized to 256 × 256.

Baselines. We compare our method against 9 state-of-theart image fusion approaches, including CDDFuse (Zhao et al., 2023b), DDFM (Zhao et al., 2023c), LRRNet (Li et al., 2023a), Text-IF (Yi et al., 2024), TC-MoA (Zhu et al.,

- 2024), EMMA (Zhao et al., 2024), SAGE (Wu et al., 2025), DCEvo (Liu et al., 2025a), and Text-DiFuse (Zhang et al., 2024a). All baselines are evaluated using their official implementations with default settings.

Hyperparameters. We train the model in two stages.

- Stage I (reconstruction warm-up) runs for 40 epochs, and
- Stage II (fusion training) runs for 80 epochs, totaling 120 epochs. We use the Adam optimizer with an initial learning rate of 1 × 10−4 and step decay (factor 0.5 every 20 epochs, minimum 1 × 10−6). Loss weights are set as: λ1 = λ2 = 1, λd = 2, λg = 5 for Stage I; λf = 1, λd = 2, λr = 0.5 for Stage II. We use mixed-precision training and gradient clipping (max norm 0.01). The pretrained TiTok tokenizer uses 32 latent tokens with dimension 12.

##### 5.2. Main Results

Qualitative Comparison. As shown in Figure 3, our method consistently produces visually superior results across all three benchmarks. On M3FD, our method reveals clear vehicle structure in dark urban night scenes. On RoadScene, our approach effectively prevents overexposure while faithfully restoring the thermal radiation signatures. On TNO, our results successfully recover fine-grained human and environmental details textures in low-light military scenarios. These visual improvements validate that our 1D token representation effectively captures and coordinates global semantics, enabling holistic appearance harmonization rather than fragmented local fusion.

[Figure 224]

[Figure 225]

- (a)
- (b)

Error Detection

Error Detection

Error Detection

Error Detection

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

| |
|---|

Visible

###### CDDFuse

SAGE

EMMA

Low Confidence

Low Confidence

Error Detection

Detected

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

|[Figure 235]<br><br>Visible|
|---|

Infrared

Text-DiFuse

Ours

DCEvo

Background Car Road Tree Building TrafficLight TrafficSign Person Sky Pole Sidewalk

|[Figure 236]<br><br>GT|
|---|

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

Visible

CDDFuse SAGE EMMA

| |
|---|

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

DCEvo Text-DiFuse Ours

Infrared

- Figure 4. (a)Qualitative comparisons of object detection performance in smoke scene. (b)Qualitative comparisons of semantic segmentation performance in nighttime scene.

Quantitative Comparison. Table 1 reports quantitative results on three IVIF benchmarks. Our method achieves the best or second-best performance across all datasets, showing consistent advantages over existing approaches. The gains in EN and SD indicate stronger information preservation and global appearance coordination, while the improvements in SCD and SSIM suggest better cross-modal integration and structural fidelity. Overall, these results demonstrate that our method effectively balances global appearance consistency and local detail preservation. More qualitative comparisons are provided in Appendix §E.

Downstream Applications. To further validate the practical utility of our fused images, we evaluate them on two downstream tasks. As shown in Table 2, our method achieves the highest mAP50:95 on M3FD and the best mIoU on FMB among all fusion methods. Notably, our significant advantage in mAP50:95 demonstrates that our fused images allow for more precise object localization. This result indicates that our method preserves superior boundary details and spatial structures, enabling detectors to generate tighter bounding boxes. The consistent gains in semantic segmentation confirm that our token-based fusion effectively maintains both global context and fine-grained structural information.

- 5.3. Ablation Studies

- Table 3. Ablation study on token manipulation positions. Modifying position 12 or 18 alone causes ghosting effects, while jointly manipulating positions 12 and 18 achieves the best performance.

Setting EI SF AG SSIM

Base (w/o manipulation) 34.39 7.90 2.94 1.37 Pos. 12 only (sharpening) 36.01 8.19 3.10 1.40 Pos. 18 only (blurring) 36.87 8.44 3.17 1.40 Pos. 12 + Pos. 18 (ours) 37.42 8.63 3.21 1.42

- Table 4. Ablation study on the number of latent tokens. With 32 tokens, global semantics are more concentrated.

###### #Tokens Sharp Pos. Blur Pos. EI SF AG SSIM

128 69 20 32.13 7.32 2.79 1.36

64 18 60 34.57 7.73 2.94 1.38 32 (ours) 12 18 37.42 8.63 3.21 1.42

havior observed in highly compressed 1D tokenizers (Beyer et al., 2025). Jointly editing the two positions achieves the best overall balance, indicating that they provide complementary effects for local detail enhancement and global appearance regulation. This result supports the use of a sparse STE design rather than dense token modification. More results are provided in Appendix §C.

Ablation on Token Numbers. We further evaluate the impact of the latent token sequence length on fusion performance by comparing TiTok variants with 32, 64, and 128 tokens. As shown in Table 4, the model with 32 tokens achieves the best performance. This is because its global semantics are more concentrated, allowing the adjustment of specific tokens to more effectively enhance image quality.

Ablation on Token Position. To evaluate the effect of sparse token manipulation, we compare different editing choices in the 1D token space. As shown in Table 3, editing position 12 mainly improves edge-related quality, while editing position 18 contributes more to appearance smoothing, which is consistent with the token-level manipulation be-

- Table 5. Efficiency comparison with representative baselines. Here, #Params denotes trainable parameters.

Method #Params FLOPs Latency Memory SSIM

CDDFuse 1.2M 116.9G 46.8ms 0.45GB 1.02 EMMA 1.5M 8.9G 23.0ms 0.17GB 0.92 DCEvo 2.0M 194.7G 50.2ms 0.47GB 1.02 Text-DiFuse 119.5M 47709.9G 2736.2ms 3.05GB 1.22 Ours 1.3M 304.5G 124.3ms 2.78GB 1.49

- Table 6. Comparison with recognition-oriented semantic representation interfaces.

###### Dataset Interface EN SD SCD SSIM

TiTok 7.10 44.54 1.83 0.70 DINOv3 6.36 24.35 1.41 0.63 CLIP 6.58 27.29 1.51 0.59

M3FD

TiTok 7.42 50.79 1.84 0.71 DINOv3 6.73 30.62 1.25 0.66 CLIP 6.84 32.24 1.36 0.60

RoadScene

TiTok 7.17 43.94 1.82 0.70 DINOv3 6.40 25.87 1.40 0.65 CLIP 6.58 28.29 1.49 0.59

TNO

TiTok 4.27 74.35 1.68 0.74 DINOv3 4.44 58.88 1.66 0.66 CLIP 6.19 38.49 0.32 0.45

MIF

##### 5.4. Additional Analysis and Discussion

Efficiency and Overhead. Since our method introduces a frozen 1D tokenizer, Table 5 reports its efficiency profile. Although the total parameter count is larger due to the frozen tokenizer, only 1.325M parameters are trainable, remaining comparable to lightweight fusion baselines. Compared with Text-DiFuse, our method greatly reduces FLOPs and latency while achieving higher SSIM. These results show that the proposed design is lightweight in optimization cost, with the overhead coming from the frozen representation interface.

Choice of 1D Representation Technique. We further examine whether recognition-oriented encoders can serve as better image-level interfaces than compact 1D tokenizers. To this end, we replace TiTok with DINOv3 (Siméoni et al., 2026) and CLIP (Radford et al., 2021) under the same fusion setting. As shown in Table 6, TiTok achieves stronger overall performance across M3FD, RoadScene, TNO, and MIF. This suggests that semantic abstraction alone does not necessarily benefit reconstruction-oriented fusion: DINOv3 and CLIP are relatively invariant to appearance changes, whereas fusion requires sensitivity to illumination, contrast, structural saliency, and modality-specific cues. TiTok is better aligned with our framework because it provides an appearance-sensitive and reconstruction-compatible interface, while the 2D pathway preserves local structures. Further discussion is provided in Appendix §D.

Token Position Selection with Gumbel-Softmax. To

Table 7. Slot-budget ablation of Gumbel-Softmax token-position selection on M3FD. The 2-slot setting achieves the best overall performance.

Method #Slots EN SD SCD SSIM CDDFuse 0 6.80 35.22 1.62 1.02

- Slot 1 1 7.08 44.60 1.79 1.41

- Slot 2 (ours) 2 7.19 47.35 1.85 1.49

- Slot 3 3 7.16 47.02 1.84 1.47
- Slot 4 4 7.11 46.55 1.82 1.45

[Figure 247]

Figure 5. Hard selection frequency of the learned token-position selector under the 2-slot setting, selecting position 12, while Slot 1 consistently selects position 18, for both visible (VI) and infrared (IR) inputs.

verify that the STE positions are not manually assigned, we use a lightweight Gumbel-Softmax (Jang et al., 2017; Maddison et al., 2017) selector to learn token-position choices under different slot budgets. The selector treats each manipulation slot as a differentiable discrete choice, allowing appearance-sensitive positions to be identified from data. As shown in Figure 5, the selector concentrates on position 12 with one slot and consistently selects positions 12 and 18 with two slots. Table 7 further shows that the 2-slot setting achieves the best overall performance on M3FD, while additional slots bring no further improvement. This suggests that positions 12 and 18 capture the main sparse manipulation structure, so we use them in STE.

### 6. Conclusion

We propose a lightweight multimodal image fusion framework that replaces dense 2D shared feature grids with a compact 1D token space via a pretrained 1D tokenizer, which enhances global semantic coherence while preserving local structural details. Building on this token space, targeted token modification further improves fused image quality. Extensive experiments on IVIF benchmarks and downstream applications demonstrate consistent gains over prior methods, achieving state-of-the-art performance.

### Acknowledgments

This research is partially supported by the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China. This research is partially supported by A*STAR Career Development Fund (CDF) under Grant C243512011, the National Research Foundation, Singapore under its National Large Language Models Funding Initiative (AISG Award No: AISG-NMLP-2024-003). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.

### Impact Statement

This work introduces a 1D token-based shared representation and selective token editing for multimodal image fusion, improving global appearance consistency (e.g., illumination and sharpness) while preserving local structures. Better fusion can benefit downstream perception in applications such as nighttime driving, robotics, remote sensing, and medical imaging, where complementary sensors are used to increase robustness under adverse conditions. However, improved fusion quality may also enhance capabilities in sensitive settings such as surveillance or military reconnaissance, potentially enabling privacy-invasive monitoring or harmful targeting.

### References

Bachmann, R., Allardice, J., Mizrahi, D., Fini, E., Kar, O. F., Amirloo, E., El-Nouby, A., Zamir, A., and Dehghan, A. Flextok: Resampling images into 1d token sequences of flexible length. In ICML, 2025.

Bai, H., Zhang, J., Zhao, Z., Wu, Y., Deng, L., Cui, Y., Feng, T., and Xu, S. Task-driven image fusion with learnable fusion loss. In CVPR, 2025.

Beyer, L. L., Li, T., Chen, X., Karaman, S., and He, K. Highly compressed tokenizer can generate without training. In ICML, 2025.

Cui, G., Feng, H., Xu, Z., Li, Q., and Chen, Y. Detail preserved fusion of visible and infrared images using regional saliency extraction and multi-scale image decomposition. Optics Communications, 2015.

Eskicioglu, A. M. and Fisher, P. S. Image quality measures and their performance. IEEE TCOM, 1995.

Esser, P., Rombach, R., and Ommer, B. Taming transformers for high-resolution image synthesis. In CVPR, 2021.

Guan, T., Wei, H., Zhou, Y., Ma, J., Xu, Z., Jiang, Z., Liu, J.,

and Li, X. Domain adaptation guided infrared and visible image fusion. In AAAI, 2026.

Jang, E., Gu, S., and Poole, B. Categorical reparameterization with gumbel-softmax. In ICLR, 2017.

Johnson, K. A. and Becker, J. A. The whole brain atlas. http://www.med.harvard.edu/AANLIB/home. html. Harvard Medical School.

Li, H. and Wu, X.-J. Densefuse: A fusion approach to infrared and visible images. IEEE TIP, 2019.

Li, H., Xu, T., Wu, X.-J., Lu, J., and Kittler, J. Lrrnet: A novel representation learning guided fusion network for infrared and visible images. IEEE TPAMI, 2023a.

Li, X., Zou, Y., Liu, J., Jiang, Z., Ma, L., Fan, X., and Liu, R. From text to pixels: A context-aware semantic synergy solution for infrared and visible image fusion. arXiv preprint arXiv:2401.00421, 2023b.

Li, X., Du, S., Zou, Y., Xu, H., Jiang, Z., and Liu, J. Unifusion: A unified image fusion framework with robust representation and source-aware preservation. arXiv preprint arXiv:2603.14214, 2026.

Li, Z., Pan, H., Zhang, K., Wang, Y., and Yu, F. Mambadfuse: A mamba-based dual-phase model for multimodality image fusion. arXiv preprint arXiv:2404.08406, 2024.

Liu, J., Fan, X., Huang, Z., Wu, G., Liu, R., Zhong, W., and Luo, Z. Target-aware dual adversarial learning and a multi-scenario multi-modality benchmark to fuse infrared and visible for object detection. In CVPR, 2022.

Liu, J., Liu, Z., Wu, G., Ma, L., Liu, R., Zhong, W., Luo, Z., and Fan, X. Multi-interactive feature learning and a full-time multi-modality benchmark for image fusion and segmentation. In ICCV, 2023.

Liu, J., Li, X., Wang, Z., Jiang, Z., Zhong, W., Fan, W., and Xu, B. Promptfusion: Harmonized semantic prompt learning for infrared and visible image fusion. IEEE/CAA Journal of Automatica Sinica, 2024a.

Liu, J., Wu, G., Liu, Z., Wang, D., Jiang, Z., Ma, L., Zhong, W., Fan, X., and Liu, R. Infrared and visible image fusion: From data compatibility to task adaption. IEEE TPAMI, 2024b.

Liu, J., Zhang, B., Mei, Q., Li, X., Zou, Y., Jiang, Z., Ma, L., Liu, R., and Fan, X. Dcevo: Discriminative crossdimensional evolutionary learning for infrared and visible image fusion. In CVPR, 2025a.

Liu, J., Li, X., Mei, Q., Xu, H., Jiang, Z., Ma, L., Liu, R., and Fan, X. Bridging human evaluation to infrared and visible image fusion. arXiv preprint arXiv:2603.03871, 2026.

Liu, Y., Zou, Y., Li, X., Zhu, X., Han, K., Jiang, Z., Ma, L., and Liu, J. Toward a training-free plug-and-play refinement framework for infrared and visible image registration and fusion. In ACM MM, 2025b.

Ma, J., Tang, L., Fan, F., Huang, J., Mei, X., and Ma, Y. Swinfusion: Cross-domain long-range learning for general image fusion via swin transformer. IEEE/CAA Journal of Automatica Sinica, 2022.

Maddison, C. J., Mnih, A., and Teh, Y. W. The concrete distribution: A continuous relaxation of discrete random variables. In ICLR, 2017.

Qu, L., Liu, S., Wang, M., Li, S., Yin, S., Qiao, Q., and Song, Z. Transfuse: A unified transformer-based image fusion framework using self-supervised learning. arXiv preprint arXiv:2201.07451, 2022.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

Rajalingam, B. and Priya, R. Hybrid multimodality medical image fusion technique for feature enhancement in medical diagnosis. International Journal of Engineering Science Invention, 2018.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.

Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S. E., Ramamonjisoa, M., Massa, F., HAZIZA, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jegou, H., Labatut, P., and Bojanowski, P. DINOv3. TMLR, 2026.

Tang, L., Yuan, J., Zhang, H., Jiang, X., and Ma, J. Piafusion: A progressive infrared and visible image fusion network based on illumination aware. Information Fusion, 2022.

Toet, A. The tno multiband image data collection. Data in brief, 2017.

Van Den Oord, A., Vinyals, O., et al. Neural discrete representation learning. In NeurIPS, 2017.

Varghese, R. and Sambath, M. Yolov8: A novel object detection algorithm with enhanced performance and robustness. In ADICS, 2024.

Wang, Z. and Wu, L. Theoretical analysis of the inductive biases in deep convolutional networks. In NeurIPS, 2023.

Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 2004.

Wang, Z., Zhang, J., Guan, T., Zhou, Y., Li, X., Dong, M., and Liu, J. Efficient rectified flow for image fusion. In NeurIPS, 2025.

Wu, G., Liu, H., Fu, H., Peng, Y., Liu, J., Fan, X., and Liu, R. Every sam drop counts: Embracing semantic priors for multi-modality image fusion and beyond. In CVPR, 2025.

Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., and Luo, P. Segformer: Simple and efficient design for semantic segmentation with transformers. In NeurIPS, 2021.

Xu, H., Ma, J., Jiang, J., Guo, X., and Ling, H. U2fusion: A unified unsupervised image fusion network. IEEE TPAMI, 2020a.

Xu, H., Ma, J., Le, Z., Jiang, J., and Guo, X. Fusiondn: A unified densely connected network for image fusion. In AAAI, 2020b.

Yi, X., Xu, H., Zhang, H., Tang, L., and Ma, J. Text-if: Leveraging semantic text guidance for degradation-aware and interactive image fusion. In CVPR, 2024.

Yu, Q., Weber, M., Deng, X., Shen, X., Cremers, D., and Chen, L.-C. An image is worth 32 tokens for reconstruction and generation. In NeurIPS, 2024.

Zhang, H., Cao, L., and Ma, J. Text-difuse: An interactive multi-modal image fusion framework based on textmodulated diffusion model. In NeurIPS, 2024a.

Zhang, H., Zuo, X., Jiang, J., Guo, C., and Ma, J. Mrfs: Mutually reinforcing image fusion and segmentation. In CVPR, 2024b.

Zhang, X., Da, C., Yang, H., Gai, K., Lu, M., and Ma, Z. Restok: Learning hierarchical residuals in 1d visual tokenizers for autoregressive image generation. arXiv preprint arXiv:2601.03955, 2026.

Zhao, W., Xie, S., Zhao, F., He, Y., and Lu, H. Metafusion: Infrared and visible image fusion via meta-feature embedding from object detection. In CVPR, 2023a.

Zhao, Z., Xu, S., Zhang, C., Liu, J., Zhang, J., and Li, P. Didfuse: Deep image decomposition for infrared and visible image fusion. In IJCAI, 2020.

Zhao, Z., Bai, H., Zhang, J., Zhang, Y., Xu, S., Lin, Z., Timofte, R., and Van Gool, L. Cddfuse: Correlation-driven dual-branch feature decomposition for multi-modality image fusion. In CVPR, 2023b.

Zhao, Z., Bai, H., Zhu, Y., Zhang, J., Xu, S., Zhang, Y., Zhang, K., Meng, D., Timofte, R., and Van Gool, L. Ddfm: denoising diffusion model for multi-modality image fusion. In ICCV, 2023c.

Zhao, Z., Bai, H., Zhang, J., Zhang, Y., Zhang, K., Xu, S., Chen, D., Timofte, R., and Van Gool, L. Equivariant multi-modality image fusion. In CVPR, 2024.

Zheng, A., Wang, H., Zhao, Y., Deng, W., Wang, T., Zhang, X., and Qi, X. Hita: Holistic tokenizer for autoregressive image generation. In ICCV, 2025.

Zhu, P., Sun, Y., Cao, B., and Hu, Q. Task-customized mixture of adapters for general image fusion. In CVPR, 2024.

### A. Additional Theoretical Analysis

This section provides additional discussion on why a compact 1D token space can serve as a useful appearance/base carrier for multimodal image fusion. The purpose is not to claim that 1D tokenizers are universally superior reconstruction backbones, but to clarify why they provide a more controllable interface for non-local appearance factors than dense 2D shared grids.

##### A.1. Broadcasted Base Factors in 2D Grids

In conventional 2D shared representations, global appearance factors such as illumination, contrast, and perceptual tone are not represented as independent variables. Instead, they are implicitly broadcast across spatial positions and mixed with local structures. Following the abstraction in the main paper, a 2D feature at spatial location (i,j) can be written as

Fij(m) = ϕ detail(ijm) + Abase(m) + ϵij, (19)

where ϕ(·) encodes local detail, Abase(m) denotes the broadcasted base component, and ϵij is the location-dependent residual. Recovering or regulating the base factor therefore requires aggregating a low-dimensional global factor from a high-dimensional spatial field. When local residuals are structured or correlated, the aggregation process becomes sensitive to noise, high-frequency edges, and modality-specific artifacts.

This explains why 2D fusion backbones are effective for local detail preservation but less direct for global appearance control. The optimization needs to coordinate many spatial positions to adjust a single image-level factor, which can make appearance regulation unstable.

##### A.2. 1D Tokens as a Compact Control Interface

- A compact 1D token representation changes the control geometry of the problem. Instead of spreading appearance factors across a dense h × w grid, it places global information into a smaller set of token variables:

Z(m) = τ(I(m)), Z(m) ∈ RK×C, (20)

where K is the number of tokens and C is the token dimension. In our implementation, K = 32 and C = 12. This compactness allows sparse token-level editing to influence global appearance without directly modifying all spatial locations.

Importantly, the 1D token branch is not used to reconstruct all high-frequency details alone. The spatial pathway remains responsible for local structures, while the token branch provides global appearance guidance. Table 8 summarizes the functional difference between dense 2D grids, recognition-oriented semantic encoders, and the compact tokenizer interface used in our framework.

Table 8. Functional roles of different representation forms for multimodal image fusion. Representation Main Strength Limitation Dense 2D grid Strong local detail modeling and spatial

Global base factors are spatially broadcast and entangled with detail/noise

reconstruction

Recognition-oriented encoder Strong semantic abstraction and invariance

May suppress appearance variations needed by fusion

Compact 1D tokenizer Low-dimensional and controllable appearance/base carrier

Requires token-to-map adaptation for spatial reconstruction

##### A.3. Appearance Sensitivity vs. Semantic Invariance

Recognition-oriented encoders are often optimized to produce invariant features. Such invariance is beneficial for recognition, retrieval, and classification, but multimodal image fusion requires sensitivity to low-level appearance changes. In fusion, illumination, contrast, thermal saliency, and modality-specific structures are not nuisance factors; they are part of the information that should be preserved or regulated. Therefore, the most suitable representation for MMIF is not necessarily the most semantic one, but one that remains compact, appearance-sensitive, and compatible with reconstruction.

### B. Detailed Architecture and Implementation

This section provides module-level implementation details that are omitted from the main paper for space reasons. The frozen tokenizer provides compact 1D tokens, while the remaining trainable modules adapt these tokens to the 2D fusion pipeline.

##### B.1. Module-Level Design

Table 9 summarizes the main modules of the proposed framework. The tokenizer is frozen throughout training. The token-to-map interface adapts compact tokens into spatial feature maps, and the base/detail branches perform factorized fusion before residual reconstruction.

Table 9. Module-level implementation details of the proposed framework. Module Operation Output / Role Trainable

- 1D tokenizer Pretrained TiTok tokenizer, kept frozen Z ∈ R32×12 No Token lifting Linear projection from token dimension 12 to 64 Lifted token features Yes Token-to-map mapping Linear mapping to a coarse spatial feature map Coarse 32 × 32 feature map Yes

Local refinement Residual local aggregation with 3 × 3 convolution and learnable scaling

Refined coarse feature map Yes

Yes

Multi-stage upsampling Progressive upsampling to recover 256 × 256 spatial resolution

Token-induced 2D feature map

Detail injection Scale-aligned detail features from original images using

Detail-aware spatial features Yes

7 × 7, 5 × 5, and 3 × 3 convolutions

Private encoder Base/detail decomposition B(m), D(m) Yes Base fusion layer Fusion in base space Bf Yes Detail fusion layer Fusion in detail space Df Yes Residual decoder Decode fused base/detail features and add residual refer-

If = Iref + ∆I Yes

ence

##### B.2. Residual Reconstruction

We use residual reconstruction to stabilize the output image. Instead of directly generating the fused image from scratch, the decoder predicts a residual term ∆I, which is added to the reference input:

Iref = IV + II, If = Iref + ∆I. (21)

This design reduces the burden on the decoder and encourages the network to focus on complementary corrections rather than reconstructing all image content independently.

##### B.3. Implementation Notes

All input images are resized to 256 × 256 during training and evaluation. The tokenizer remains frozen, and only the token-to-map interface, factorization modules, fusion layers, and decoder are optimized. This design avoids drifting the pretrained token space and keeps the appearance-control interface stable during training.

### C. Extended Token Editing Evidence

The main paper reports the slot-budget ablation and the hard selection frequency under the 2-slot setting. This section provides additional visual evidence for understanding how the learned slots behave and how they affect fusion outputs.

##### C.1. Full Selector Distribution

- Figure 6 shows the full Gumbel-Softmax selector distributions under different slot budgets. With one slot, the selector concentrates on a single dominant position. With two slots, which is our final setting, the selector identifies two complementary positions. When the number of slots is further increased, additional slots are assigned to weaker residual positions, suggesting diminishing returns beyond the 2-slot configuration.

[Figure 248]

- Figure 6. Full Gumbel-Softmax selector distributions under different slot budgets. The first two slots capture the dominant sparse manipulation structure, while additional slots mainly absorb weaker residual effects.

- C.2. Slot-Wise Visual Interpretation

- Figure 7 compares the visual effects of editing Slot 0 only, Slot 1 only, and Slot 0 & Slot 1 jointly. Slot 0 mainly strengthens local edges and contours, producing a sharpening-oriented effect. Slot 1 mainly suppresses distracting background residue and improves global appearance smoothness. Jointly editing both slots combines these complementary effects and provides a better balance between local detail preservation and global coherence.

[Figure 249]

|VI IR|
|---|

|RAW|
|---|

|Slot 0 only Slot 1 only Slot 0 & Slot 1 (ours)|
|---|

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

- Figure 7. Slot-wise qualitative comparison of sparse token manipulation. Slot 0 mainly produces a sharpening-oriented effect, while Slot 1 mainly produces a background-smoothing-oriented effect. Jointly editing Slot 0 and Slot 1 yields a better balance between detail enhancement and global appearance consistency.

##### C.3. Interpretation of the Selected Slots

The selected slots should not be interpreted as universal semantic token labels. They are empirical effect-based slots discovered under the current TiTok-32 configuration. Table 10 summarizes their observed roles.

Table 10. Interpretation of selector-discovered STE slots under the TiTok-32 configuration. Slot Observed Effect Interpretation

- Slot 0 Edge and contour sharpening Enhances local structural clarity and fine boundary visibility

- Slot 1 Background smoothing Suppresses high-frequency background residue and improves appearance consistency Slot 0 + Slot 1 Complementary editing Balances local detail enhancement with global appearance coherence

##### C.4. Generalization of STE

The transferable part of STE is the selection-and-editing mechanism, not the fixed slot indices or token positions. When the tokenizer, token count, or token dimension changes, the sensitive slots may also correspond to different positions. Therefore, applying STE to another tokenizer configuration requires re-running the lightweight selection or probing step before sparse editing.

### D. Tokenizer Domain Gap and Robustness

Since the tokenizer is pretrained on natural images and kept frozen, it is necessary to examine whether it remains usable for infrared and medical images. This section reports reconstruction quality and alternative tokenizer results to assess the domain gap.

##### D.1. Reconstruction Quality on Infrared and Medical Images

- Table 11 reports tokenizer reconstruction quality on infrared and medical images. The frozen tokenizer reconstructs infrared images with high SSIM, suggesting that it remains robust despite the modality gap. Medical images show a larger structural mismatch, reflected by lower SSIM. Nevertheless, the main framework does not rely on the tokenizer as a standalone high-fidelity decoder; it uses the tokenizer as a compact appearance/base carrier and relies on the 2D pathway for local reconstruction.

Table 11. Reconstruction quality of the frozen tokenizer on infrared and medical domains. Domain MSE ↓ PSNR ↑ SSIM ↑

Infrared 68.3117 30.2214 0.9647 Medical 126.3955 29.3419 0.7032

- D.2. Alternative Tokenizers

Table 12 reports reconstruction quality for alternative compact tokenizers. These results indicate that different tokenizer families can provide usable compact representations, but their reconstruction behavior varies. This supports our view that the STE positions should be re-identified when the tokenizer configuration changes.

Table 12. Reconstruction comparison of alternative compact tokenizers. Tokenizer MSE ↓ PSNR ↑ SSIM ↑

ResTok (Zhang et al., 2026) 150.070 26.677 0.878 FlexTok (Bachmann et al., 2025) 86.434 29.150 0.962

- D.3. Implication for Fusion

The reconstruction results should not be interpreted as the final criterion for fusion quality. A tokenizer with high reconstruction fidelity may still require task-specific adaptation to support fusion, while a compact tokenizer with sufficient appearance sensitivity can be useful as a control interface. Our framework therefore separates the role of the tokenizer from the role of the fusion decoder: the tokenizer provides compact global guidance, and the 2D branch preserves local structure.

- D.4. Limitations and Future Directions

The above analysis also indicates several limitations of the current framework. First, the effectiveness of STE depends on whether the frozen tokenizer exposes appearance-sensitive factors that are useful for fusion. Although our results show that the TiTok-based interface remains effective on infrared-visible and medical fusion tasks, different tokenizer families may organize appearance information differently, and therefore may require task-specific probing before being used for selective token editing. Second, the selected STE positions are configuration-specific rather than universal semantic indices. In our TiTok-32 setting, positions 12 and 18 are consistently identified as effective manipulation slots, but applying the same mechanism to another tokenizer, token length, or token dimension requires re-identifying the editable positions. Third, while the number of trainable parameters remains small, the full inference pipeline still includes both a frozen tokenizer branch and a 2D reconstruction pathway, which introduces additional latency and memory cost compared with very compact

- 2D-only baselines. Future work may therefore explore lighter token-to-map interfaces, adaptive tokenizer selection, and more efficient token editing strategies for deployment-oriented fusion systems.

### E. Additional Qualitative Results

This section summarizes additional qualitative observations beyond the examples shown in the main paper. Since the main paper already provides representative visual comparisons, we use a compact summary table to describe the recurring visual patterns observed across different fusion scenarios.

- Figure 8 provides representative qualitative comparisons across several challenging fusion scenarios. In urban night scenes, where visible textures are weak and thermal targets dominate, our method enhances salient infrared responses while preserving the surrounding visible structures. In road and low-light surveillance scenes, existing methods tend to suffer from over-smoothed textures, unstable brightness, or distracting residual artifacts, whereas our result maintains a more coherent global tone and clearer object boundaries. For distant background regions with weak structural cues, our method better preserves building contours and spatial layout, indicating that the 1D token interface improves appearance regulation without sacrificing local detail. In the medical fusion example, our method also preserves complementary cross-modality structures while producing a more consistent fused appearance. These observations suggest that the proposed 1D token representation and STE mechanism jointly improve the balance between global appearance consistency and local structure preservation.

|[Figure 261]<br><br>[Figure 262]<br><br>Ours<br><br>|
|---|

|[Figure 263]<br><br>[Figure 264]<br><br>[Figure 265]<br><br>|
|---|

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Input CDDFuse SAGE EMMA DCEvo Text-DiFuse Ours

|[Figure 281]<br><br>[Figure 282]<br><br>Ours<br><br>|
|---|

|[Figure 283]<br><br>[Figure 284]<br><br>|
|---|

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

Input CDDFuse SAGE EMMA DCEvo Text-DiFuse Ours

|[Figure 300]<br><br>[Figure 301]<br><br>Ours<br><br>|
|---|

|[Figure 302]<br><br>[Figure 303]<br><br>|
|---|

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

Input CDDFuse SAGE EMMA DCEvo Text-DiFuse Ours

|[Figure 319]<br><br>[Figure 320]<br><br>Ours|
|---|

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

[Figure 335]

[Figure 336]

[Figure 337]

Input CDDFuse SAGE EMMA DCEvo Text-DiFuse Ours

- Figure 8. Qualitative comparisons across representative fusion scenarios. Our method produces more coherent global appearance and clearer local structures under low illumination, background clutter, weak boundaries, and cross-modality medical fusion.

As shown in Figures 9–11, the additional M3FD examples cover several difficult infrared-visible fusion cases. Existing methods can preserve salient infrared responses, but they often introduce unstable brightness, over-smoothed textures, or local artifacts around vehicles, pedestrians, and road boundaries. In contrast, our method maintains clearer thermal targets while better preserving visible structural cues, such as lane regions, building contours, vehicle boundaries, and background layout. These results further support the role of the 1D token interface in regulating global appearance without weakening local structural fidelity.

Figure 12 further reports additional medical image fusion examples on the Harvard dataset. Across MRI-CT, MRI-PET, and MRI-SPECT fusion settings, our method preserves complementary anatomical and functional information while producing more coherent fused appearances. Compared with competing methods, the proposed approach better balances structural clarity from MRI and modality-specific intensity information from CT, PET, or SPECT. This suggests that the proposed representation design is not limited to infrared-visible fusion, but can also generalize to cross-modality medical fusion scenarios.

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

Ir CDDFuseVis DCEvo Text-DiFuse Ours

Ir Vis DCEvo CDDFuse Text-DiFuse Ours

[Figure 350]

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

Ir CDDFuseVis DCEvo Text-DiFuse Ours

Ir Vis DCEvo CDDFuse Text-DiFuse Ours

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

Ir CDDFuseVis DCEvo Text-DiFuse Ours

Ir Vis DCEvo CDDFuse Text-DiFuse Ours

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

Ir CDDFuseVis DCEvo Text-DiFuse Ours

Ir Vis DCEvo CDDFuse Text-DiFuse Ours

- Figure 9. Additional qualitative comparisons on the M3FD dataset. Our method better preserves salient infrared targets while maintaining visible structural details under nighttime and low-illumination conditions.

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

Ir CDDFuseVis DCEvo Text-DiFuse Ours

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

Ir CDDFuseVis DCEvo Text-DiFuse Ours

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

Ir CDDFuseVis DCEvo Text-DiFuse Ours

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

Ir CDDFuseVis DCEvo Text-DiFuse Ours

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

Ir Vis DCEvo CDDFuse Text-DiFuse Ours

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

Ir Vis DCEvo CDDFuse Text-DiFuse Ours

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

Ir Vis DCEvo CDDFuse Text-DiFuse Ours

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

Ir Vis DCEvo CDDFuse Text-DiFuse Ours

- Figure 10. Additional qualitative comparisons on the M3FD dataset under challenging road scenes. Compared with existing methods, our method produces more coherent global brightness and clearer object boundaries.

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

Ir CDDFuseVis DCEvo Text-DiFuse Ours

Ir Vis DCEvo CDDFuse Text-DiFuse Ours

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

Ir CDDFuseVis DCEvo Text-DiFuse Ours

Ir Vis DCEvo CDDFuse Text-DiFuse Ours

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

Ir CDDFuseVis DCEvo Text-DiFuse Ours

Ir Vis DCEvo CDDFuse Text-DiFuse Ours

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

Ir CDDFuseVis DCEvo Text-DiFuse Ours

Ir Vis DCEvo CDDFuse Text-DiFuse Ours

- Figure 11. Additional qualitative comparisons on the M3FD dataset. Our method reduces visual artifacts and improves the balance between thermal saliency and visible-scene structure.

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

MRI CT DCEvo CDDFuse Text-DiFuse Ours

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

MRI CT DCEvo CDDFuse Text-DiFuse Ours

MRI PET DCEvo CDDFuse Text-DiFuse Ours

MRI SPECT DCEvo CDDFuse Text-DiFuse Ours

- Figure 12. Additional qualitative comparisons on the Harvard medical image fusion dataset. Our method preserves anatomical structures while maintaining complementary modality-specific information from CT, PET, and SPECT images.

