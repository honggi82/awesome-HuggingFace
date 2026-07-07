## VideoCanvas: Unified Video Completion from Arbitrary Spatiotemporal Patches via In-Context Conditioning

Minghong Cai1† Qiulin Wang2✉ Zongli Ye1 Wenze Liu1 Quande Liu2 Weicai Ye2 Xintao Wang2 Pengfei Wan2 Kun Gai2 Xiangyu Yue1✉ 1MMLab, The Chinese University of Hong Kong 2Kling Team, Kuaishou Technology

# arXiv:2510.08555v2[cs.CV]27May2026

Any-timestamp Patches to Video (with arbitrary spatial layout)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

###### Legend

Condition Patches

[Figure 6]

(1) (2) (3) (4) 🕒 Frame index

🕒 20 🕒 100 🕒 120

|[Figure 7]<br><br>[Figure 8]<br><br>🕒 80<br><br>(3)|
|---|

|[Figure 9]<br><br>[Figure 10]<br><br>🕒 0<br><br>(1)|
|---|

|[Figure 11]<br><br>[Figure 12]<br><br>🕒 40<br><br>(2)|
|---|

|[Figure 13]<br><br>[Figure 14]<br><br>🕒 156<br><br>(4)|
|---|

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

🕒 60

[Figure 20]

[Figure 21]

[Figure 22]

🕒 0 🕒 20 🕒 100

|[Figure 23]<br><br>[Figure 24]<br><br>🕒 80<br><br>(2)|
|---|
|[Figure 25]<br><br>[Figure 26]<br><br>🕒 80<br><br>(1)|

|[Figure 27]<br><br>[Figure 28]<br><br>🕒 40<br><br>(1)|
|---|

|[Figure 29]<br><br>[Figure 30]<br><br>🕒 156<br><br>(4)|
|---|
|[Figure 31]<br><br>[Figure 32]<br><br>🕒 156<br><br>(4)|

|[Figure 33]<br><br>[Figure 34]<br><br>🕒 120<br><br>(3)|
|---|
|[Figure 35]<br><br>[Figure 36]<br><br>🕒 120<br><br>(3)|

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

🕒 60

[Figure 43]

[Figure 44]

AnyV2VAnyP2VAnyI2V

🕒 0 🕒 20 🕒 40

|[Figure 45]<br><br>[Figure 46]<br><br>🕒 100<br><br>(2)|
|---|

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

🕒 60

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Any-timestamp Images to Video (with full images)

🕒 12 🕒 60 🕒 100 🕒 120

|[Figure 55]<br><br>[Figure 56]<br><br>🕒 80|
|---|

|[Figure 57]<br><br>[Figure 58]<br><br>🕒 0|
|---|

|[Figure 59]<br><br>[Figure 60]<br><br>🕒 156|
|---|

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

🕒 40

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

🕒 11 🕒 33 🕒 44 🕒 55 🕒 66 🕒 76

|[Figure 71]<br><br>[Figure 72]<br><br>🕒 0|
|---|

[Figure 73]

🕒 22

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

[Figure 85]

[Figure 86]

Outpainting Camera Control

Inpainting

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

SourceOutput

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Video Extension & Loop ( >1K frames)

| |[Figure 105]|[Figure 106]<br><br>[Figure 107]| | |
|---|---|---|---|---|

||[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]|
|---|
|
|---|

| | |[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]| | |
|---|---|---|---|---|

| |[Figure 114]|[Figure 115]|[Figure 116]| |
|---|---|---|---|---|

| | |[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]| | |
|---|---|---|---|---|

| | |[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]| | |
|---|---|---|---|---|

[Figure 123]

[Figure 124]

[Figure 125]

Initial shot along a coastal highway past a beautiful village past a golden field past an ice canyon past a classic building past a Hawaii beach into a mountain trail

Video Transition

🕒 48 🕒 72 🕒100 🕒 120 🕒 140

|[Figure 126]<br><br>[Figure 127]<br><br>🕒 0|
|---|

|[Figure 128]<br><br>[Figure 129]<br><br>🕒 156|
|---|

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

🕒 24

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Figure 1. VideoCanvas: Arbitrary Spatio-Temporal Video Completion. Given any conditions (frames or patches, outlined in red), the model fills in the remaining gray regions to generate coherent, high-quality videos. This unified formulation subsumes various tasks such as Any-Timestep-Patch/Image-to-Video, In/Outpainting, Camera Control, and Cross-scene Video Transitions, all in a zero-shot manner. More results are available on our Project Page: https://onevfall.github.io/project_page/videocanvas/. Best viewed zoomed in.

#### Abstract

temporal control as a set of isolated problems. We formalize a unified task, arbitrary spatio-temporal video completion, where a model generates a coherent video from userspecified patches placed at any spatial location and timestamp. However, realizing such a unified framework within modern latent video diffusion models is non-trivial: causal video VAEs compress multiple frames into a single latent slot,

Existing controllable video generation methods are typically designed for rigid, task-specific settings—such as first-frame image-to-video, inpainting, or interpolation—treating spatio-

†Work done at Kuaishou Technology. ✉Corresponding authors.

making frame-level conditioning fundamentally ill-posed, and directly feeding sparsely populated, zero-padded video inputs into the VAE leads to severe out-of-distribution artifacts. To address these challenges, we propose VideoCanvas, a simple yet effective framework that adapts the In-Context Conditioning paradigm to arbitrary spatio-temporal completion without modifying or retraining the VAE. Our key idea is a hybrid conditioning strategy that decouples spatial and temporal control: spatially, we encode zero-padded full-frame canvases in image mode to keep VAE inputs indistribution, and temporally we use Temporal RoPE Interpolation to assign each condition a continuous fractional index in the latent sequence for precise frame-level alignment. To evaluate this capability, we develop VideoCanvasBench, the first benchmark for arbitrary spatio-temporal video completion, covering both intra-scene fidelity and inter-scene creativity. Extensive experiments demonstrate that VideoCanvas achieves state-of-the-art performance across a diverse range of video generation tasks under a single, unified framework.

#### 1. Introduction

Video generation has made significant strides with the advent of Diffusion Transformers (DiTs) [5, 39, 50, 57], marking a turning point in the field’s ability to synthesize highquality videos. However, generating videos that truly align with user intent remains a significant challenge. Existing controllable approaches are typically constrained by rigid, task-specific formats—for example, conditioning only on a first frame [16, 28], using an initial clip with limited temporal horizon [1, 55], or performing structural inpainting and outpainting [51, 56, 62]. These methods treat spatiotemporal control as a set of isolated problems, lacking a unified approach. We propose a unified approach to bridge these fragmented tasks: treating video synthesis as painting on a spatio-temporal canvas. In this framework, users can place arbitrary content patches at any location and timestamp, and the model will synthesize a complete, temporally consistent video around them, as illustrated in Fig. 1. This fine-grained control enables a wide range of applications, from creative content generation to practical use cases, such as reconstructing videos from partially transmitted or corrupted data packets [11, 31], or generating videos with specific spatial and temporal conditions for diverse domains.

Realizing this vision presents fundamental challenges across both spatial and temporal dimensions. Temporally, causal video VAEs compress multiple pixel frames into a single latent slot, creating indexing ambiguity—precisely the source of the difficulty in achieving frame-accurate control—as illustrated in Fig. 2(a). Spatially, conditions may take arbitrary forms—from full frames to small, irregular patches—requiring a mechanism that can seamlessly unify inpainting and outpainting within one formulation. The core

difficulty lies in designing a conditioning paradigm that can resolve both temporal ambiguity and spatial irregularity simultaneously.

Viewed through this lens, the limitations of existing paradigms become clear. Latent Replacement [18, 28] was designed mainly for first-frame I2V but fails to generalize, as it overwrites entire latent slots and disrupts temporal consistency once applied to arbitrary timestamps. Channel Concatenation and Adapter-style injection methods [37, 50, 57, 59] fuse conditional features either by concatenating at the input or injecting via lightweight encoders. Despite architectural differences, these approaches remain coarse-grained: pixel-frame-aware control ultimately requires feeding zero-padded frames to the VAE, but pretrained VAEs are not robust to such inputs. Making them work would require expensive VAE fine-tuning and re-training of the DiT backbone. More recent In-Context Conditioning (ICC) methods [17, 19, 25, 43, 58] inherit the same difficulty when naively combined with zero-padding: they still demand VAE/DiT re-training to handle the distribution shift, and further double the sequence length by encoding padded frames, resulting in severe inefficiency during both training and inference.

In this paper, we introduce VideoCanvas, the first framework to apply In-Context Conditioning to the challenging task of arbitrary spatio-temporal video completion. We also propose a hybrid conditioning strategy that decouples space and time: spatial alignment is achieved by zero-padded VAE encoding of arbitrary patches, while temporal ambiguity is resolved by our novel RoPE Interpolation, which assigns continuous fractional indices to conditional frame tokens. This design removes the need for costly re-training of the VAE or architectural modifications of the DiT backbone, while allowing efficient fine-tuning to enable fine-grained pixel-frame-aware control within a simple, parameter-free ICC architecture.

To evaluate this unified task and framework, we present VideoCanvasBench, a comprehensive benchmark tailored for arbitrary spatio-temporal video completion. To the best of our knowledge, it is the first to systematically incorporate multi-frame, non-homologous image and patch conditions to test both intra-scene fidelity and inter-scene creativity. Our contributions are as follows:

- • We formalize the task of arbitrary spatio-temporal video completion, under which existing tasks such as imageto-video, interpolation, and inpainting emerge as special cases, and provide an in-depth analysis revealing the structural limitations of existing conditioning paradigms under this setting.
- • We propose VideoCanvas, the first framework to adapt InContext Conditioning to this task, with a hybrid strategy: Spatial Zero-Padding for in-distribution VAE encoding and Temporal RoPE Interpolation for precise frame-level

|0 Zero Noisy Latent Condition|
|---|

Latent Token

###### Latent

Frame 0 1 2 3 Index

4

Latent Replace

|[Figure 142]<br><br>[Figure 143]|
|---|

𝜀

Naive Solution Our Solution ❌

[Figure 144]

❌

Causal encoding

[Figure 145]

(E.g. HunyuanVideo)

}

Naive: Spatio-Temporal Zero-Pad

}

[Figure 146]

Not seen in Causal VAE training

0 0 0

Frame 0 1 2 3 Index

Channel Concate

4

Naive Solution ✅ Our Solution ❌

[Figure 147]

Linear

[Figure 148]

Latent Token

Position Index

|[Figure 149]|
|---|

|[Figure 150]|
|---|

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Videoframe

𝜀 𝜀

(E.g. Wan/CogVideoX)

1/4 3/4

[Figure 155]

RoPE Alignment

How to identify specific frames when they share the same latent representation? Like Frame Patch 1 and 3 as condition here.

Naive Solution ✅ Our Solution ✅

[Figure 156]

In-Context Condition

Ours: Hybrid (Pad + RoPE Interpolation)

[Figure 157]

[Figure 158]

Utilizing VAE to encode single frame

(Ours)

(b) Paradigms for Video Conditioning

(a) The Core Challenge and Our Proposed Solution

Figure 2. Core challenge and solution for pixel-frame-aware conditioning. (a) Causal VAEs create temporal ambiguity by mapping frames to a single latent. We propose a hybrid solution combining Spatial Padding with Temporal RoPE Interpolation. (b) We show how competing paradigms are ill-suited for fine-grained control, while our ICC approach provides an effective solution.

alignment—enabling pixel-frame-aware control on frozen VAEs without retraining or new parameters.

driven by the token’s coordinates in 3D space (time, height, and width), allowing the model to capture both temporal continuity and spatial relationships. This mechanism is central to our ability to perform precise temporal alignment in video generation, as RoPE naturally supports interpolation to non-integer positions.

• We introduce VideoCanvasBench, the first benchmark for arbitrary spatio-temporal completion, and demonstrate state-of-the-art performance across diverse settings.

#### 2. Methodology

Hybrid Video VAE. Modern video foundation models employ a Hybrid Video VAE [53, 57, 60] that supports both image and video modes. In video mode, the encoder performs causal temporal compression with a fixed stride N (e.g., N = 4): the first frame maps to latent index 0, and every subsequent N frames collapse into a single latent slot. Formally, for a pixel-frame index i, the latent index is ⌈i/N⌉. This stride-based compression is efficient but introduces pixel-frame ambiguity: multiple frames (e.g., frames 1, 2, 3) may share the same latent representation, making precise frame-level conditioning non-trivial.

##### 2.1. Task Definition

To enable flexible and unified video generation, we formalize the task of arbitrary spatio-temporal video completion.

Let a video be denoted as X = {x0,x1,...,xT−1} with T frames. A user provides a set of spatio-temporal conditions

P = {(pi,mi,ti)}Mi=1, where pi is an image, mi is a spatial mask specifying its placement within a frame, ti ∈ [0,T−1] is the temporal index, and M is the number of conditions. The goal is to generate a coherent video Xˆ such that

##### 2.3. Zero Padding Observations and Analysis

Xˆ [ti] ⊙ mi ≈ pi, ∀i ∈ {1,...,M},

Existing conditioning approaches [2, 23, 24] commonly represent missing content by zero-filling unobserved regions or frames. However, the impact of such zero-padded inputs on the causal VAEs of modern video models has not been thoroughly investigated. We analyze this in Fig. 3 (more details in Appendix E.1) and present two key findings:

while completing all unconditioned regions with plausible content. This formulation naturally unifies many prior settings as special cases: image-to-video (P contains one full frame), interpolation (P specifies first and last frames), and inpainting/outpainting (P contains masked regions). By allowing arbitrary spatial masks at arbitrary timestamps, it goes strictly beyond these rigid formats.

Zero-Padded Input Hybrid VAE

Result

|[Figure 159]|
|---|

##### 2.2. Preliminaries

|[Figure 160]|
|---|

0-padded

Video DiT with 3D RoPE. Our work builds upon a latent video diffusion model that uses a Diffusion Transformer (DiT) backbone [39] and is trained with a flow matching objective [32]. To handle the spatio-temporal nature of video data, the model’s self-attention mechanism is equipped with 3D Rotary Positional Embeddings (RoPE) [42]. The 3D RoPE encodes each token’s position by applying a rotation to the query and key vectors in the attention mechanism,

Single Image

(Image mode)

PSNR = 34.7

|[Figure 161]|
|---|

|[Figure 162]|
|---|

0-padded 0-padded

Zero-padded Video

(Video mode)

PSNR = 23.1

###### Figure 3. Analysis of Zero-Padding VAE Reconstruction.

Spatial padding is robust in image mode. Image-based pipelines (e.g., BrushNet [24] and VideoPainter [2]) routinely process frames with large masked/blank regions, and image VAEs are typically robust to such spatial sparsity. In our setting, we place patches on a zero-padded full-frame canvas and encode each frame with the video VAE in image mode. As shown in Fig. 3 (top), reconstruction fidelity drops only slightly, indicating that spatial zero-padding does not substantially shift the input distribution for the encoder.

Temporal padding is harmful in video mode. Causal video VAEs compress N consecutive frames into one latent slot (e.g., N = 4). If a conditioning sequence is formed by inserting zero-filled frames, then blank and non-blank frames are mixed within the same N-frame window, producing temporal dynamics that the VAE never observes during training. Consequently, temporal zero-padding causes severe out-ofdistribution degradation: Fig. 3 (bottom) shows large PSNR drops and visible corruption, consistently across strong backbones such as HunyuanVideo [28] and CogVideoX [57].

##### 2.4. Architecture Design Space Analysis

Given that spatial padding is VAE-friendly while temporal padding is destructive (Sec. 2.3), a natural question arises: which conditioning paradigm can best exploit this asymmetry? We revisit three common approaches under two solution strategies, as illustrated in Fig. 2(b).

The naive solution constructs a full-length video with zero-filled unobserved frames and processes it through the VAE in video mode. Latent Replacement [28] directly overwrites latent slots, but since it operates after VAE encoding, it cannot control which pixel frames contribute to each slotmaking arbitrary-timestamp conditioning infeasible even under this strategy. Channel Concatenation [50, 57] can adopt this approach by concatenating the zero-padded conditional sequence along the channel dimension, but doing so inevitably triggers the temporal zero-padding degradation identified in Sec. 2.3. ICC also supports this strategy, but inherits the same VAE quality loss.

Our solution instead encodes each conditional frame independently in image mode, bypassing temporal zeropadding entirely. This requires the paradigm to accept standalone tokens with flexible temporal positions. Latent Replacement cannot accommodate independently encoded tokens, as it relies on overwriting fixed slots in the noisy latent tensor. Channel Concatenation is similarly constrained to the integer latent grid, offering no mechanism for fractional temporal alignment. Only ICC, which concatenates conditions along the token dimension, naturally supports both independent encoding and continuous-time positional assignments via RoPE interpolation. We therefore build VideoCanvas upon the In-Context Conditioning paradigm, as detailed in Sec. 2.5.

##### 2.5. VideoCanvas Pipeline

To address the challenge of arbitrary spatio-temporal completion, we propose VideoCanvas, a unified framework built upon the In-Context Conditioning (ICC) paradigm. As established in Sec. 2.4, ICC uniquely enables sub-latent temporal precision through flexible positional assignments. Combined with the VAE’s tolerance for spatial padding (Sec. 2.3), we introduce a hybrid conditioning strategy that decouples spatial and temporal alignment, enabling fine-grained, pixel-frameaware control on a frozen VAE and a fine-tuned DiT with zero new parameters. The entire pipeline is illustrated in Fig. 4.

Spatial Conditioning via Zero-Padding. As shown on the left of Fig. 4, our process begins at the pixel level. For each conditional patch (pi,mi,ti) ∈ P, we construct a fullframe canvas, place the patch pi in its correct spatial location according to mask mi, and fill the remaining pixels with zeros. Formally, the prepared frame is xprep,i = mi ⊙ pi, where ⊙ denotes element-wise multiplication and unconditioned regions are zero-filled. This preserves the absolute positional information required for spatial control, and crucially, does not cause out-of-distribution issues as the VAE tolerates spatial padding well (Sec. 2.3).

Temporal Decoupling via Independent VAE Encoding. Next, each of these prepared frames is encoded independently by the frozen VAE in its image mode. This is a critical step for temporal decoupling: by encoding each frame individually, we bypass the VAE’s causal temporal compression mechanism. The result is a set of conditional latent tokens zcond,i = E(xprep,i), where each token purely represents its corresponding single pixel frame, free from the temporal ambiguity discussed in Sec. 2.2.

Temporal Alignment via RoPE Interpolation. Having obtained temporally decoupled conditional latents, we now address the core challenge: precisely aligning them within the DiT’s 3D spatio-temporal grid. Following the ICC paradigm, we construct a unified sequence by concatenating the conditional tokens with the target video latent. Let zsource denote the latent representation of the ground-truth target video, obtained by encoding the full video with the VAE in video mode. The complete input sequence is then:

z = Concat({zcond,i}Mi=1,zsource).

We leverage the continuous nature of the 3D RoPE used by our DiT backbone. In the standard implementation, for a token at integer latent index k, RoPE applies a rotation to the query vector q ∈ RD

t. For the j-th pair of elements, the

Conditional Patches Generated Video

Latent Tokens

Position Index

[Figure 163]

[Figure 164]

Frame 0

Frame 0

[Figure 165]

[Figure 166]

𝜀 𝜀 𝜀

0

<Zero-Pad>

[Figure 167]

𝒟

[Figure 168]

[Figure 169]

[Figure 170]

Frame 41

Frame 41

10.25

LatentDiffusionModel

<Zero-Pad>

[Figure 171]

…

[Figure 172]

[Figure 173]

Y / 4

Frame Y

Frame Y

Image mode

[Figure 174]

<Zero-Pad>

[Figure 175]

- (Y/4,0,0) (Y/4,0,1) (Y/4,0,2)
- (Y/4,1,0) (Y/4,1,1) (Y/4,1,2)

[Figure 176]

4 x noisy frames

Noisy Token Conditional Token Frozen Weight Trainable Weight

- 0
- 1

- (10.25,0,0) (10.25,0,1) (10.25,0,2)
- (10.25,1,0) (10.25,1,1) (10.25,1,2)

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

- (0,0,0) (0,0,1) (0,0,2)
- (0,1,0) (0,1,1) (0,1,2)

[Figure 182]

[Figure 183]

Conditional Patches

…

…

VAE Enc./Dec.

In-Context Conditioning Temporal RoPE Interpolation

Figure 4. The VideoCanvas pipeline. We fine-tune a base T2V model to enable arbitrary spatio-temporal control, without introducing any additional parameters. Our framework leverages the In-Context Conditioning (ICC) paradigm. After preparing conditional patches with zero-padding for spatial placement, we use independent VAE encoding for temporal decoupling. Our RoPE Interpolation then aligns each discrete token by mapping its source pixel-frame index Y to a fractional position Y/N, where N is the VAE temporal stride (here, N = 4). As illustrated, this maps Frame 41 to position 10.25. This strategy enables fine-grained control without architectural changes.

transformation is:

latent: zt = (1 − t)zsource + tϵ. The model is trained to predict the velocity field, and the loss only supervises the non-conditional regions:

q2′j q2′j+1

cos(kθj) −sin(kθj) sin(kθj) cos(kθj)

q2j q2j+1

, (1)

=

LFM(θ) = E fθ(zt,t,ctext) − (−zsource + ϵ) 2 .

where θj = b−2j/D

t are the base frequencies. This integer indexing k ∈ {0,1,...} suffices for the source latent zsource, which aligns with the VAE’s compressed grid.

This objective trains the DiT to treat the conditional tokens as fixed context while generating a coherent completion for the target video.

However, our conditional tokens originate from specific pixel frames that may fall between integer latent indices. For example, with VAE stride N = 4, a condition from Frame 41 should ideally be positioned between latent indices 10 and 11. To achieve this sub-latent precision, we introduce Temporal RoPE Interpolation: given a condition from pixel-frame τi, we assign it a fractional temporal position k′ = τi/N and substitute into Eq. (1):

#### 3. VideoCanvasBench

Existing benchmarks focus on rigid tasks such as I2V or outpainting, and cannot assess the flexible spatio-temporal control central to our formulation. We therefore introduce VideoCanvasBench, the first benchmark systematically designed for arbitrary spatio-temporal video completion.

N θj) −sin(τ

cos(τ

q ˜2j q˜2j+1

N θj) sin(τ

q2j q2j+1

i

i

. (2)

=

N θj) cos(τ

The benchmark probes two complementary capabilities: high-fidelity completion within a single scene (homologous) and creative synthesis across different sources (non-homologous). It consists of three categories: (1) AnyP2V, using partial patches at fixed anchor timestamps (Start, Middle, End). We construct all seven possible combinations—single-frame (S, M, E), two-frame (S+M, S+E, M+E), and three-frame (S+M+E)—to evaluate interpolation fidelity under varying temporal sparsity. (2) AnyI2V, using full-frame conditions at the same timestamps, designed to test the completion of full-frame content. (3) AnyV2V, covering video-level completion scenarios such as inpainting, outpainting, and transitions between non-homologous clips. In total, VideoCanvasBench comprises over 2,000 test cases. Further construction details are provided in Appendix D.

N θj)

i

i

As illustrated in Fig. 4, this maps Frame 41 to the fractional position 10.25. For N = 4, pixel frames τ ∈ {1,2,3} within the first latent window map to fractional positions {0.25,0.50,0.75}, allowing the attention mechanism to perceive conditions at precise sub-latent timestamps. This strategy enables pixel-frame-aware temporal control that is structurally inaccessible to other paradigms.

Training Objective. The DiT model, fθ, is fine-tuned with this unified sequence under the flow matching objective [32, 35]. The noising process is applied to the source latent of the sequence. The model input at time t is thus a combination between the clean condition and the noisy

Table 1. Quantitative evaluations on our benchmark against state-of-the-art models. Our unified framework demonstrates strong and consistent performance across a wide range of tasks, excelling in challenging few-frame-to-video interpolation and maintaining competitiveness in specialized domains like inpainting.

Video Quality & Video Consistency FVD↓ Aesthetic Background Dynamic Imaging Motion Overall Subject Temporal Normalized

Type Method

Quality Consistency Degree Quality Smoothness Consistency Consistency Flickering Average

I2V CogVideoX-1.5 [57] 14.980 57.10% 95.17% 21.00% 74.30% 99.03% 25.22% 95.32% 97.85% 70.62% HunyuanVideo [28] 16.756 57.21% 95.19% 27.00% 73.31% 99.13% 25.01% 93.60% 97.47% 70.99% VideoCanvas (Ours) 14.476 56.05% 95.96% 30.00% 72.54% 99.04% 25.21% 95.74% 96.98% 71.44%

FLF2V CogVideoX-FT [13] 13.545 56.88% 95.04% 18.00% 72.65% 98.89% 25.00% 95.55% 96.89% 69.86% Sci-Fi [6] 13.181 57.00% 95.28% 9.00% 72.97% 99.11% 25.11% 95.83% 97.35% 68.96% VideoCanvas (Ours) 9.053 55.91% 96.01% 42.00% 72.51% 98.97% 25.15% 95.44% 96.87% 72.86%

TF2V CogVideoX-FT [13] 7.522 55.24% 94.82% 35.00% 71.84% 99.05% 24.82% 95.66% 97.36% 71.72% Sci-Fi [6] 9.523 57.87% 94.59% 23.00% 72.37% 99.15% 24.98% 95.85% 97.72% 70.69% VideoCanvas (Ours) 7.393 55.76% 96.13% 45.00% 72.01% 98.97% 25.13% 95.69% 96.93% 73.20%

FLP2V Flux+Wan [29, 50] 18.689 57.24% 93.78% 67.00% 73.69% 96.84% 25.25% 91.50% 94.49% 74.97% VideoCanvas (Ours) 16.799 55.81% 96.08% 55.00% 72.27% 99.07% 24.85% 95.91% 97.14% 74.52%

Inpaint ProPainter [61] 4.236 52.97% 96.26% 49.00% 70.22% 98.62% 24.81% 95.39% 96.83% 73.01% VACE [23] 2.218 56.72% 95.55% 49.00% 74.81% 98.22% 20.41% 95.74% 96.24% 73.34% VideoCanvas (Ours) 5.521 54.82% 96.06% 49.00% 72.60% 98.87% 25.22% 95.88% 97.05% 73.69%

Outpaint M3DDM [12] 75.266 39.35% 95.62% 35.00% 52.13% 98.85% 11.24% 94.77% 98.10% 65.63% VACE [23] 5.129 56.82% 95.63% 47.00% 75.03% 98.29% 25.30% 95.82% 96.43% 73.79% VideoCanvas (Ours) 9.119 55.88% 95.93% 49.00% 72.78% 98.92% 25.19% 95.81% 97.07% 73.82%

#### 4. Experiments

##### 4.1. Setup

Backbone. We fine-tune a latent video diffusion model with a causal VAE (see Appendix C for training details). Inference uses 50 DDIM steps with a CFG scale of 7.5.

Compared Methods. We compare our approach with stateof-the-art video generation methods, where the tasks include i) image-to-video (I2V) generation, such as CogVideoX1.5 [57] and HunyuanVideo [28]; ii) first-last-frame-to-video (FLF2V) generation, such as CogVideoX-FT [13] and SciFi [6]; iii) first-middle-last-frame-to-video (TF2V), where we adapt CogVideoX-FT [13] and Sci-Fi [6] by decomposing the task into two FLF2V subproblems; iv) first-last-patchto-video (FLP2V) generation, which uses FLUX-Inpainting model [29] to generate two frames at first, and then use video interpolation model [50]; v) video inpainting and outpainting tasks such as VACE [23]. We provide an evaluation of overall video performance, including video fidelity (FVD [47]) and a comprehensive suite of metrics from VBench [22] for video quality and consistency.

##### 4.2. Main Results

SOTA Comparison on VideoCanvasBench. We evaluate our full model against leading specialized models on our benchmark. The results are summarized in Tab. 1. For standard I2V, our model achieves the lowest FVD and the highest consistency scores (Background and Subject Consistency), demonstrating strong distributional fidelity and

Table 2. Conditioning paradigm comparison. All methods are trained on the same backbone for fair comparison.

Dynamic Degree↑

Imaging Quality↑

Method FVD↓

Replacement 15.937 25.42 69.02 Channel Concat 14.187 41.77 68.58 ICC (Ours) 13.734 43.81 69.10

identity preservation within a single unified framework. The advantage of our approach becomes more pronounced in multi-frame interpolation tasks. In both FLF2V and TF2V settings, our method significantly outperforms competitors, achieving the lowest FVD and the highest Dynamic Degree. This highlights its superior capability in reasoning over sparse temporal conditions to generate coherent and dynamic motion. In specialized tasks like Inpainting and Outpainting, dedicated models such as VACE show very strong performance, particularly in FVD. However, our unified framework remains highly competitive, often achieving the best normalized average score. This demonstrates the versatility of our approach, which can handle a wide range of completion tasks effectively without being specifically designed for any single one.

Paradigm Comparison. We isolate the conditioning mechanism by comparing Latent Replacement, Channel Concatenation, and our In-Context Conditioning (ICC) on the same backbone. As shown in Tab. 2, ICC achieves the best

[Figure 184]

A woman in a white dress and a widebrimmed hat walks barefoot along a wooden suspension bridge surrounded by lush greenery. …

[Figure 185]

[Figure 186]

a breathtaking aerial view of a vast, green landscape with a prominent rocky cliff…

Input

Input

First Frame

First Patch Last Patch

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

HunyuanVideo

Flux + CogVideoX

| |
|---|

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Wan-Video

Discontinuous Transition

[Figure 198]

[Figure 199]

Flux + Wan

| |
|---|

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

CogVideoX

[Figure 204]

[Figure 205]

VideoCanvas (Ours)

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

VideoCanvas (Ours)

TASK: First-Last-Patch-to-Video (FLP2V)

TASK: First-Frame-to-Video (I2V)

One monkey is grooming the other...

[Figure 210]

[Figure 211]

a bustling urban scene featuring a prominent, tall, beige building with numerous...

[Figure 212]

Input

Input

First Frame Last Frame

[Figure 213]

|[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>Frozen Motion|
|---|

[Figure 218]

[Figure 219]

| |
|---|

[Figure 220]

CogVideoX

M3DDM

[Figure 221]

[Figure 222]

[Figure 223]

VACE

Sci-Fi

[Figure 224]

Artifacts

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

VideoCanvas (Ours)

VideoCanvas (Ours)

TASK: First-Last-Frame-to-Video (FLF2V)

TASK: Video Outpainting

###### Figure 5. Qualitative comparison with state-of-the-art methods on four representative VideoCanvasBench tasks.

Frame 0 Frame 40

Qualitative Comparison. Fig. 5 provides per-task qualitative evidence. For FLF2V, the test case involves highdynamic, long-range interpolation between two distant keyframes; both CogVideoX-FT and Sci-Fi produce frozen motion near the conditioning boundaries, failing to synthesize plausible intermediate dynamics, while our model generates smooth and coherent transitions. For FLP2V, although Flux+Wan achieves competitive metrics, its twostage pipeline (image generation followed by video interpolation) introduces discontinuous transitions between the independently generated patches; our end-to-end framework produces seamless completions. For Outpainting, M3DDM exhibits severe artifacts and VACE shows visible distortions in the expanded regions, whereas our model maintains high fidelity throughout. We further compare conditioning paradigms in Fig. 6: Latent Replacement suffers from frozen motion, while Channel Concatenation introduces unexpected artifacts. Our ICC-based method generates smooth motion with preserved identity. More visualizations are in Appendix F.

Condition

[Figure 230]

[Figure 231]

[Figure 232]

Frame 0 Frame 37 Frame 40 Frame 76

| |
|---|

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

. Replace.

[Figure 237]

Frozen Motion

| |
|---|

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

ChannelICC(Ours)

[Figure 242]

Unexpected Artifacts

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

- Figure 6. Qualitative comparison of conditioning paradigms on a two-frame I2V task.

(lowest) FVD, indicating the generated videos have the highest distributional similarity to real videos. Furthermore, it scores highest in both Imaging Quality and Dynamic Degree, demonstrating its superior ability to generate high-quality, dynamic content. Latent Replacement, while simple, struggles to synthesize motion, and Channel Concatenation lags behind ICC across all key metrics. This confirms that ICC is intrinsically a more effective paradigm for this task.

Overall, both quantitative and qualitative evidence show that our proposed framework, combining Temporal RoPE Interpolation with the In-Context Conditioning paradigm, establishes a new state of the art in the versatile and challenging task of arbitrary spatio-temporal video completion.

[Figure 247]

- Figure 7. Ablation Study on Temporal Alignment Strategies. Per-frame PSNR for single-frame I2V with frame index targets 2/3/4. Our method (red) peaks exactly at the target frame. “w/o RoPE Interpolation” (blue) misaligns, “Latent-space Condition” (orange) collapses motion, and “Pixel-space Padding” (green) is precise but degraded. Zoom in for best view.

Table 3. Ablation Study on Temporal Alignment Strategies.

[Figure 248]

[Figure 249]

[Figure 250]

Dynamic Degree↑

Imaging Quality↑

Method PSNR↑

[Figure 251]

[Figure 252]

[Figure 253]

Ours (RoPE Interp.) 23.86 39.75 71.61 w/o RoPE Interp. 22.95 23.00 70.85 Pixel-space Pad. 22.02 30.25 71.50 Latent-space Cond. 25.13 5.00 71.17

Input Frame Pixelspace-Padding RoPE Interpolation(Ours)

Figure 8. Pixelspace-Padding vs. RoPE Interpolation.

##### 4.3. Ablation Study

As discussed in Fig. 2 (a), causal video VAEs map several pixel frames into one latent, creating ambiguity when conditioning on a specific frame. To address this, we compare four alignment strategies: (i) Latent-space Conditioning: encode the entire video with the VAE to obtain a latent sequence, then use the corresponding latent slice as condition. (ii) Pixel-space Padding: construct a video where non-target frames are zeroed, then encode it with the VAE. (iii) w/o RoPE Interpolation: encode each conditional frame independently (image mode) and assign it to the nearest discrete temporal slot. (iv) Our full method with Temporal RoPE Interpolation.

Qualitative evidence. Fig. 8 illustrates that while pixelspace padding is temporally precise, it introduces visible artifacts like color shifts and texture degradation because the VAE was not trained on zero-filled inputs. In contrast, our RoPE-based alignment preserves the conditional frame with high fidelity.

Quantitative analysis. We further evaluate single-frame I2V generation targeting frame indices 2,3,4. As shown in Fig. 7 and Tab. 3, Latent-space Conditioning achieves high PSNR but collapses motion (very low Dynamic Degree). w/o RoPE Interpolation recovers dynamics but its PSNR peaks are misaligned. Pixel-space Padding peaks correctly but suffers from lower fidelity. Our method with RoPE Interpolation not only aligns perfectly with the target frames but also achieves the best balance of fidelity and motion generation. These results confirm that our proposed strategy uniquely provides both fine-grained control and high-quality generation.

##### 4.4. Applications and Emerging Capabilities

By framing video synthesis as completion on a spatiotemporal canvas, our framework naturally unlocks a diverse set of capabilities beyond the standard tasks evaluated above. As illustrated in Fig. 1, VideoCanvas supports flexible temporal control at arbitrary timestamps (AnyI2V), arbitrary spatiotemporal patch conditioning (AnyP2V), creative video transitions between non-homologous scenes, long-duration video extension via autoregressive generation, and unified video inpainting, outpainting, and camera control. Detailed results and analysis for each application are provided in Appendix F and our project page.

#### 5. Conclusion

We formalized the task of arbitrary spatio-temporal video completion and provided a systematic analysis showing that existing conditioning paradigms are structurally limited under causal VAEs. Based on this analysis, we proposed VideoCanvas, a framework that adapts In-Context Conditioning with a hybrid strategy—Spatial Zero-Padding and Temporal RoPE Interpolation—enabling pixel-frame-aware control on frozen VAEs without retraining or new parameters. Experiments on VideoCanvasBench demonstrate state-of-the-art performance across diverse tasks, from multi-frame interpolation to creative video transitions. While our approach scales inference cost with the number of conditioning frames, future work on token pruning and efficient attention could further improve scalability for dense conditioning scenarios.

#### Acknowledgements

This work is partially supported by the National Natural Science Foundation of China (Grant No. 62306261), and The Shun Hing Institute of Advanced Engineering (SHIAE) Grant (No. 8115074). This study was supported in part by the Centre for Perceptual and Interactive Intelligence, a CUHK-led InnoCentre under the InnoHK initiative of the Innovation and Technology Commission of the Hong Kong Special Administrative Region Government. This work is also partially supported by Hong Kong RGC Strategic Topics Grant STG1/E-403/24-N, and CUHK-CUHK(SZ)-GDST Joint Collaboration Fund YSP26-4760949.

#### References

- [1] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15791–15801, 2025. 2, 12
- [2] Yuxuan Bian, Zhaoyang Zhang, Xuan Ju, Mingdeng Cao, Liangbin Xie, Ying Shan, and Qiang Xu. Videopainter: Anylength video inpainting and editing with plug-and-play context control. arXiv preprint arXiv:2503.05639, 2025. 3, 4, 12
- [3] G. Bradski. The OpenCV Library. Dr. Dobb’s Journal of Software Tools, 2000. 15
- [4] Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, and Xiangyu Yue. Ditctrl: Exploring attention control in multi-modal diffusion transformer for tuning-free multi-prompt longer video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7763–7772, 2025. 14
- [5] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 2
- [6] Liuhan Chen, Xiaodong Cun, Xiaoyu Li, Xianyi He, Shenghai Yuan, Jie Chen, Ying Shan, and Li Yuan. Sci-fi: Symmetric constraint for frame inbetweening. arXiv preprint arXiv:2505.21205, 2025. 6, 13
- [7] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In The Twelfth International Conference on Learning Representations, 2023. 12
- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 15, 16
- [9] Duolikun Danier, Fan Zhang, and David Bull. Ldmvfi: Video frame interpolation with latent diffusion models. In Proceed-

- ings of the AAAI Conference on Artificial Intelligence, pages 1472–1480, 2024. 12
- [10] Kaixin Ding, Xi Chen, Sihui Ji, Yuan Gao, Liang Hou, Xin Tao, and Hengshuang Zhao. Surf: Signature-retained fast video generation. arXiv preprint arXiv:2603.21002, 2026. 20
- [11] Kuntai Du, Ahsan Pervaiz, Xin Yuan, Aakanksha Chowdhery, Qizheng Zhang, Henry Hoffmann, and Junchen Jiang. Server-driven video streaming for deep learning inference. In Proceedings of the Annual conference of the ACM Special Interest Group on Data Communication on the applications, technologies, architectures, and protocols for computer communication, pages 557–570, 2020. 2
- [12] Fanda Fan, Chaoxu Guo, Litong Gong, Biao Wang, Tiezheng Ge, Yuning Jiang, Chunjie Luo, and Jianfeng Zhan. Hierarchical masked 3d diffusion model for video outpainting. In Proceedings of the 31st ACM International Conference on Multimedia, pages 7890–7900, 2023. 6, 13
- [13] Zhengcong Fei. Cogvideox-interpolation. https:// github.com/feizc/CogvideX-Interpolation,

2025. 6, 13

- [14] Xiao Fu, Shitao Tang, Min Shi, Xian Liu, Jinwei Gu, MingYu Liu, Dahua Lin, and Chen-Hsuan Lin. Plenoptic video generation. arXiv preprint arXiv:2601.05239, 2026. 12
- [15] Chenjian Gao, Lihe Ding, Xin Cai, Zhanpeng Huang, Zibin Wang, and Tianfan Xue. Lora-edit: Controllable first-frameguided video editing via mask-aware lora fine-tuning. arXiv preprint arXiv:2506.10082, 2025. 12
- [16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2, 12
- [17] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv preprint arXiv:2503.10589,

2025. 2, 12

- [18] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024. 2, 12, 13
- [19] Xuanhua He, Quande Liu, Zixuan Ye, Wecai Ye, Qiulin Wang, Xintao Wang, Qifeng Chen, Pengfei Wan, Di Zhang, and Kun Gai. Fulldit2: Efficient in-context conditioning for video diffusion transformers. arXiv preprint arXiv:2506.04213,

2025. 2, 12, 20

- [20] Xu He, Haoxian Zhang, Hejia Chen, Changyuan Zheng, Liyang Chen, Songlin Tang, Jiehui Huang, Xiaoqiang Liu, Pengfei Wan, and Zhiyong Wu. From inpainting to editing: A self-bootstrapping framework for context-rich visual dubbing. arXiv preprint arXiv:2512.25066, 2025. 12
- [21] Jiehui Huang, Yuechen Zhang, Xu He, Yuan Gao, Zhi Cen, Bin Xia, Yan Zhou, Xin Tao, Pengfei Wan, and Jiaya Jia. Unityvideo: Unified multi-modal multi-task learning for enhancing world-aware video generation. arXiv preprint arXiv:2512.07831, 2025. 12
- [22] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang

- Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 6, 14
- [23] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025. 3, 6, 12, 13
- [24] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. In European Conference on Computer Vision, pages 150–168. Springer,

2024. 3, 4

- [25] Xuan Ju, Weicai Ye, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Qiang Xu. Fulldit: Multi-task video generative foundation model with full attention. arXiv preprint arXiv:2503.19907, 2025. 2, 12, 13
- [26] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. MUSIQ: multi-scale image quality transformer. CoRR, abs/2108.05997, 2021. 14
- [27] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 15
- [28] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2, 4, 6, 12, 13
- [29] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 6, 13
- [30] LAION-AI. aesthetic-predictor. https://github.com/ LAION-AI/aesthetic-predictor, 2022. 14
- [31] Tianhong Li, Vibhaalakshmi Sivaraman, Pantea Karimi, Lijie Fan, Mohammad Alizadeh, and Dina Katabi. Reparo: Loss-resilient generative codec for video conferencing. arXiv preprint arXiv:2305.14135, 2023. 2
- [32] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3, 5
- [33] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 15
- [34] Wenze Liu, Weicai Ye, Minghong Cai, Quande Liu, Xintao Wang, and Xiangyu Yue. In-context audio control of video diffusion transformers. arXiv preprint arXiv:2512.18772,

2025. 12

- [35] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 5
- [36] Haoyu Lu, Guoxing Yang, Nanyi Fei, Yuqi Huo, Zhiwu Lu, Ping Luo, and Mingyu Ding. Vdt: General-purpose video diffusion transformers via mask modeling. arXiv preprint arXiv:2305.13311, 2023. 12
- [37] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image

- diffusion models. In Proceedings of the AAAI conference on artificial intelligence, pages 4296–4304, 2024. 2, 12
- [38] OpenAI. Video generation models as world simulators. https://openai.com/index/videogeneration- models- as- world- simulators/,

2023. Accessed: 2024-2. 15

- [39] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205, 2023. 2, 3, 13
- [40] Pexels. Pexels license. https://www.pexels.com/ license/, 2025. 15, 16
- [41] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 12
- [42] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 3

- [43] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024. 2, 12, 13
- [44] Zachary Teed and Jia Deng. Raft: Recurrent all pairs field transforms for optical flow. In ECCV, 2020. 14, 15
- [45] Ultralytics. Yolov8. https : / / github . com / ultralytics/ultralytics, 2023. 15
- [46] Unsplash. Unsplash license. https://unsplash.com/ license, 2025. 15, 16
- [47] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. ArXiv, abs/1812.01717, 2018. 6, 14
- [48] Vikram Voleti, Alexia Jolicoeur-Martineau, and Chris Pal. Mcvd-masked conditional video diffusion for prediction, generation, and interpolation. Advances in neural information processing systems, 35:23371–23385, 2022. 12
- [49] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 12
- [50] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun

- Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 4, 6, 12, 13
- [51] Fu-Yun Wang, Xiaoshi Wu, Zhaoyang Huang, Xiaoyu Shi, Dazhong Shen, Guanglu Song, Yu Liu, and Hongsheng Li. Beyour-outpainter: Mastering video outpainting through inputspecific adaptation. In European Conference on Computer Vision, pages 153–168. Springer, 2024. 2, 12
- [52] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8428–8437, 2025. 15
- [53] Pingyu Wu, Kai Zhu, Yu Liu, Liming Zhao, Wei Zhai, Yang Cao, and Zheng-Jun Zha. Improved video vae for latent video diffusion model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18124–18133, 2025. 3
- [54] Wenhao Yan, Sheng Ye, Zhuoyi Yang, Jiayan Teng, ZhenHui Dong, Kairui Wen, Xiaotao Gu, Yong-Jin Liu, and Jie Tang. Scail: Towards studio-grade character animation via incontext learning of 3d-consistent pose representations. arXiv preprint arXiv:2512.05905, 2025. 12
- [55] Jiazhi Yang, Kashyap Chitta, Shenyuan Gao, Long Chen, Yuqian Shao, Xiaosong Jia, Hongyang Li, Andreas Geiger, Xiangyu Yue, and Li Chen. Resim: Reliable world simulation for autonomous driving. arXiv preprint arXiv:2506.09981,

2025. 2, 12

- [56] Shuzhou Yang, Xiaoyu Li, Xiaodong Cun, Guangzhi Wang, Lingen Li, Ying Shan, and Jian Zhang. Gencompositor: Generative video compositing with diffusion transformer. arXiv preprint arXiv:2509.02460, 2025. 2, 12
- [57] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2, 3, 4, 6, 12, 13
- [58] Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, and Wenhan Luo. Unic: Unified in-context video editing. arXiv preprint arXiv:2506.04216, 2025. 2, 12
- [59] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 2, 12
- [60] Sijie Zhao, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Muyao Niu, Xiaoyu Li, Wenbo Hu, and Ying Shan. Cvvae: A compatible video vae for latent generative video models. Advances in Neural Information Processing Systems, 37: 12847–12871, 2024. 3
- [61] Shangchen Zhou, Chongyi Li, Kelvin C.K. Chan, and Chen Change Loy. Propainter: Improving propagation and transformer for video inpainting. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 10477–10486, 2023. 6

[62] Shangchen Zhou, Chongyi Li, Kelvin CK Chan, and Chen Change Loy. Propainter: Improving propagation and transformer for video inpainting. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10477–10486, 2023. 2, 12, 13

#### Supplementary Material

Overview. This supplementary material provides additional technical details, extended analyses, and qualitative results for VideoCanvas. We also include broader comparisons and ablations that complement the main paper. More results are available on our Project Page: https://onevfall.

github.io/project_page/videocanvas/. The content is organized as follows:

- • Section A (Related Work) provides a comprehensive review of related literature.
- • Section B (Introduction of the Base Text-to-Video Generation Model) reviews the base text-to-video diffusion backbone and key design choices.
- • Section C (Implementation Details) describes training strategy, compared methods, and evaluation protocols for reproducibility.
- • Section D (VideoCanvasBench Construction Details) details dataset curation, task definitions, and annotation/licensing notes.
- • Section E (More Analysis and Results) provides deeper analyses (e.g., padding robustness) and additional quantitative/qualitative results.
- • Section F (Applications and Qualitative Results) showcases diverse applications and side-by-side visual comparisons across tasks.

#### A. Related Work

##### A.1. Arbitrary Spatio-Temporal Video Completion

Controllable video generation aims to synthesize content that adheres to user inputs beyond a simple text prompt. Existing approaches are often constrained by rigid, taskspecific formats, such as conditioning only on a first frame [15, 16, 28, 41, 49], on a short initial sequence [1, 55], or on structural inpainting and outpainting [2, 20, 51, 56, 62]. Conceptually, these represent special cases of the broader challenge of video completion, yet prior work has treated them as separate sub-tasks, each requiring specialized solutions. Several recent works [7, 9, 48] explore aspects of multi-frame conditioning or temporally flexible video generation. SEINE [7] focuses on short-to-long video transitions and temporal prediction, using a temporal mask modeling objective on continuous frame sequences. Although it enables multi-frame temporal reasoning, SEINE does not support spatially irregular conditioning. VDT [36] takes a complementary approach through masked video modeling on dense spatio-temporal token grids. It relies on a spatial-only VAE (i.e., no temporal compression), producing per-frame token maps over which bi-directional prediction is performed. MCVD [48] employs masked conditional video diffusion for prediction, generation, and interpolation tasks, but operates on pixel space without leveraging modern causal VAEs. LDMVFI [9] applies latent diffusion models specifically to video frame interpolation between given keyframes, but is

constrained to dense temporal sequences and does not address arbitrary spatial-temporal positioning. In contrast, we focus on the task of arbitrary spatio-temporal video completion under modern latent video diffusion models with causal video VAEs, addressing the pixel-frame ambiguity challenge that these prior works do not encounter.

##### A.2. Paradigms for Video Conditioning

Achieving arbitrary spatio-temporal control requires a robust conditioning mechanism. Existing approaches can be broadly categorized into three paradigms, each with distinct limitations when applied to our task. Latent Replacement [18, 28] directly overwrites latent slots with conditional content, but suffers from train-inference mismatch when applied beyond first-frame conditioning, often causing motion collapse. Channel Concatenation [50, 57] and adapter-based methods [23, 37, 59] fuse conditions via concatenation or lightweight encoders, yet require costly VAE and DiT retraining to handle the zero-padding needed for pixel-frame-aware control. In-Context Conditioning (ICC) [14, 17, 19, 21, 34, 54], pioneered by OminiControl [43] for images and extended to video by FullDiT [25] and UNIC [58], offers a parameter-free alternative by treating conditions as tokens in a unified sequence. While promising, prior ICC methods struggle with the pixel-frame ambiguity introduced by causal VAEs, limiting precise temporal alignment.

Building on ICC, we are the first to enable pixel-framelevel arbitrary spatio-temporal video completion under frozen causal VAEs. Our key innovation is Temporal RoPE Interpolation, which assigns fractional temporal positions to conditional tokens, achieving sub-latent precision without VAE retraining. Combined with a hybrid conditioning strategy, our approach unlocks ICC’s full potential for finegrained spatio-temporal control.

[Figure 254]

[Figure 255]

[Figure 256]

Skip

Addition

Transformer Block

3D VAE Encoder

| |RMSNorm<br><br>3DSelf-Attention|
|---|---|
|&Scale<br><br>| |

| |RMSNorm<br><br>Feed-Forward<br><br>|
|---|---|
|&Scale<br><br>NN<br><br>| |

| |RMSNorm<br><br>Cross-Attention<br><br>|
|---|---|
|&Scale<br><br>| |

2DSelf-Attention

3DSelf-Attention

Feed-ForwardNN

Cross-Attention

RMSNorm&Scale

RMSNorm&Scale

RMSNorm&Scale

RMSNorm&Scale

[Figure 257]

[Figure 258]

[Figure 259]

𝑧𝑧𝑡𝑡

𝑧𝑧0 Noise

3D VAE Decoder

T5 Encoder

“Text prompt”

[Figure 260]

TimeStep

[Figure 261]

[Figure 262]

###### Figure S9. Overview of the base text-to-video generation model.

#### B. Introduction of the Base Text-to-Video Generation Model

We use a transformer-based latent diffusion model [39] as the base T2V generation model, as illustrated in Fig. S9. We employ a 3D-VAE to transform videos from the pixel space to a latent space, upon which we construct a transformerbased video diffusion model. Unlike previous models that rely on UNets or transformers, which typically incorporate an additional 1D temporal attention module for video generation, such spatially-temporally separated designs do not yield optimal results. We replace the 1D temporal attention with 3D self-attention, enabling the model to effectively perceive and process spatiotemporal tokens, thereby achieving a high-quality and coherent video generation model. Specifically, before each attention or feed-forward network (FFN) module, we map the timestep to a scale, thereby applying RMSNorm to the spatiotemporal tokens.

#### C. Implementation Details

##### C.1. Training strategy

The model is fine-tuned for 20k steps on a curated highquality video dataset comprising diverse scenes and motion patterns (384 × 672 resolution, 5 seconds per clip), using the Adam optimizer with a learning rate of 5 × 10−5 and a batch size of 32 on 32 GPUs.

At each iteration, 20 frames are randomly sampled from a source video to serve as temporal anchors. From each anchor frame, we extract a spatial region by cropping a patch covering between 20%–100% of the original frame size. This unified training strategy ensures that the model encounters a diverse spectrum of conditioning scenarios, ranging from sparse local patches to nearly complete frames, and from early anchors to late anchors. Such exposure allows the model to learn arbitrary spatio-temporal conditioning in a single framework.

##### C.2. Details of Compared Methods

This section provides additional details on the methods compared against our approach, including (i) existing state-ofthe-art models across different tasks, (ii) the conditioning paradigms used for fair evaluation, and (iii) the alignment strategies included in our ablation studies.

###### C.2.1. Compared SOTA Methods

We evaluate our method against state-of-the-art video generation systems spanning multiple task settings:

- • Image-to-Video (I2V). We compare with CogVideoX1.5 [57] and HunyuanVideo [28], which represent strong image-to-video generators.
- • First–Last-Frame-to-Video (FLF2V). For FLF2V generation, we include CogVideoX-FT [13] and Sci-Fi [6], both

- of which are designed to synthesize temporally coherent sequences conditioned on the first and last frames.
- • First–Middle–Last-Frame-to-Video (TF2V). To evaluate tri-frame conditioning, we decompose the task into two FLF2V subproblems—first→middle and middle→last—and stitch the two generated segments into a complete sequence. Both subproblems are solved using CogVideoX-FT [13] and Sci-Fi [6].
- • First–Last-Patch-to-Video (FLP2V). Since this task requires completing missing regions from spatial patches, we first convert the patch inputs into full-frame images using FLUX [29], and then employ Wan-FLF2V-14B [50] to generate the full video sequence conditioned on the completed first and last frames.
- • Video Inpainting. For temporal inpainting, we compare with VACE [23] and ProPainter [62] for reconstructing missing or corrupted intervals.
- • Video Outpainting. For spatial and temporal outpainting, we benchmark against VACE [23] and M3DDM [12], which are designed to extrapolate beyond the original field of view or temporal extent.

###### C.2.2. Compared Conditioning paradigms Setting

For fair comparison of different conditioning paradigms, we re-implement other two representative paradigms on the same base model (Fig. 2b), following the references used in the main text:

- • Latent Replacement [18, 28]. For a given conditional frame, the corresponding latent tokens are overwritten with VAE-encoded ground-truth latents. Training applies a masked loss only to non-conditional regions, while conditional regions are assigned timestep 0.
- • Channel Concatenation [50, 57]. Condition frames are encoded into latents, assembled into a zero-padded latent sequence, and concatenated with the noisy latent sequence along the channel dimension. A learnable projection layer then restores the embedding dimension. In our implementation, concatenation is applied after patchification, as this setting empirically yields the best results; applying it before patchification leads to degraded visual quality. The tradeoff is that after-patchify concatenation substantially increases the channel dimensionality, resulting in a projection layer with ∼16.6M trainable parameters. Thus, while this design enriches the conditioning signal and improves learning, it comes at the cost of significantly more parameters compared to the other paradigms.
- • In-Context Conditioning (Ours) [25, 43]. Our method encodes condition frames into clean latent tokens and concatenates them with the noisy sequence along the token dimension. Temporal alignment is achieved with our RoPE Interpolation strategy (Sec. 2.5). The loss is applied only to noisy tokens, while conditional tokens are assigned timestep 0. This design requires no additional trainable

parameters.

All paradigms are trained under identical settings and restricted to the same set of conditionable frames defined by the VAE stride, ensuring a rigorous and controlled comparison.

###### C.2.3. Ablation Alignment Strategies

Modern causal video VAEs do not encode frames independently; instead, every latent token represents a temporal window (typically 4 frames). This design improves compression but introduces a fundamental limitation: a single latent slice does not correspond to a single pixel frame. All compared conditioning strategies differ primarily in how they attempt to recover frame-level alignment from these window-based latents.

- 1. Latent-Space Conditioning. This method encodes the entire groundtruth video with the causal VAE and takes the latent slice whose receptive field overlaps the target frame. However, because each latent mixes information from its surrounding N frames, the “condition latent” inevitably contains content from neighboring frames as well. Thus it does not represent the target frame alone. This explains why in Fig. 7 the PSNR peak does not occur at the correct frame index: the latent window is misaligned with the temporal position of the condition. Furthermore, conditioning the diffusion model on a temporally mixed latent suppresses motion, producing the collapse observed in Dynamic Degree.
- 2. Pixel-Space Padding. This strategy constructs a clip where only the target frame is present and all other frames are zero-padded, then encodes it using the video VAE. Zero-padded frames fall inside the VAE’s temporal window, causing the encoder to fuse blank and valid content—an out-of-distribution scenario that leads to color shifts and texture distortion (as shown in Fig. 8). Thus, although this method is temporally precise, its reconstructions are of low fidelity.
- 3. Nearest-slot Assignment (w/o RoPE Interpolation). To avoid temporal mixing, we instead encode each conditional frame independently in image mode, which is spatially robust. However, image-mode latents are continuous in time, whereas the diffusion transformer expects discrete latent slots determined by the VAE stride. Assigning each frame to its nearest temporal slot yields coarse alignment and explains the misaligned PSNR peaks in Fig. 7.
- 4. Our Method: Independent Encoding + Temporal RoPE Interpolation. Our approach resolves all issues above. By encoding each conditional frame independently, we avoid the temporal entanglement intrinsic to video VAEs. Our Temporal RoPE Interpolation then maps each independently encoded latent to an arbitrary continuous temporal coordinate while preserving ordering

and positional geometry. This provides precise pixelframe alignment without relying on video-VAE temporal windows or zero-padded inputs.

##### C.3. Evaluation Metrics

For completeness, we provide the full set of evaluation metrics used in our quantitative comparisons. In addition to video fidelity measured by FVD [47], we adopt the comprehensive VBench [22] suite, which evaluates both video quality and temporal consistency across multiple dimensions. Specifically, the following metrics are reported:

- • FVD [47] (↓): Measures overall video fidelity and temporal coherence with respect to real data.
- • Aesthetic Quality [30] (↑): Assesses the perceptual attractiveness of the generated video frames.
- • Background Consistency (↑): Evaluates how well the background remains stable across time. This differs from CSCV [4] (better for videos with significant camera motion or large scene transition), which measures adjacentframe CLIP similarity, whereas VBench computes consistency relative to the first frame.
- • Dynamic Degree [44](↑): Quantifies the amount of motion present in the generated video, reflecting whether the dynamics are neither overly static nor excessive.
- • Imaging Quality [26] (↑): Measures sharpness, clarity, and reconstruction of fine-grained details.
- • Motion Smoothness (↑): Captures the temporal stability of motion across adjacent frames.
- • Overall Consistency (↑): Evaluates global temporal coherence across the entire clip.
- • Subject Consistency (↑): Measures identity and appearance stability of the primary subject across time.
- • Temporal Flickering (↑): Detects flickering artifacts or frame-to-frame instability.
- • Normalized Average (↑): The mean of all above normalized VBench metrics, providing an aggregated measure of overall video quality.

These metrics jointly offer a detailed and reliable characterization of video quality and temporal stability, ensuring a fair and comprehensive comparison across all evaluated methods.

#### D. VideoCanvasBench Construction Details

This section provides a comprehensive overview of the data curation and task generation pipeline for VideoCanvasBench, the first systematic evaluation suite for arbitrary spatio-temporal video completion.

##### D.1. Data Curation

We curate two complementary types of sources: (1) homologous videos for testing fidelity within a single coherent scene, and (2) non-homologous images and videos for evaluating creativity across distinct content.

Homologous Video Set (100 Videos). We began with an initial pool of ∼2,000 videos from Pexels [40]. A multistage filtering pipeline was applied to ensure quality and diversity:

- • Blur filtering: blurry videos were removed by calculating the CV2.Laplacian [3] score for each frame and excluding those below a threshold of 200.
- • Motion filtering: static or nearly-static clips were excluded using RAFT-based motion magnitude thresholds exceeding 5 [44].
- • Length filtering: only videos longer than 5 seconds were retained.

From this pool, we selected 100 diverse, high-quality clips covering a wide range of scenes (e.g., human activities, animals, landscapes). All were standardized to 77 frames at 15 FPS to provide a consistent evaluation length. Each video is paired with captions generated by a captioning model finetuned on Koala36M [52] following the LLaVA-based [33] annotation pipeline. All captions are further verified by human annotators to ensure accuracy in both content and motion descriptions.

Non-Homologous Image and Video Sets. To test the ability to synthesize across unrelated contexts, we manually curated visually distinct sources from Pexels [40] and Unsplash [46], ensuring large appearance and semantic gaps. The set includes:

- • 50 pairs of non-homologous images, selected to maximize dissimilarity (e.g., indoor vs. outdoor, object vs. scene).
- • 50 triplets of non-homologous images, further increasing combinatorial diversity.
- • 30 pairs of non-homologous video clips, curated for challenging video transitions, similar to the blending function of Sora [38].

These non-homologous cases explicitly test the model’s capacity for creative interpolation and cross-scene reasoning. Each non-homologous source is annotated with captions automatically generated by Gemini 2.5 Pro [8] and manually corrected to ensure faithful descriptions of both appearance and motion.

##### D.2. Benchmark Task Definitions

Task 1: AnyI2V (Any-Timestamp Image-to-Video). This task uses full frames as conditions to test temporal reasoning and interpolation fidelity. We explicitly construct nine sub-tasks by combining conditions from fixed temporal anchors: start (frame 1), middle (frame 41), and end (frame 77).

• Homologous cases. From each source video we sample three anchor frames (start, middle, end), and construct:

– Single-frame I2V: start → video, middle → video, end

→ video.

- – Two-frame I2V: start+end → video, start+middle → video, middle+end → video.
- – Three-frame I2V: start+middle+end → video.

• Non-homologous cases. For curated pairs of images, we construct the three two-frame tasks (start+end, start+middle, middle+end). For curated triplets of images, we construct the three-frame task (start+middle+end). Each non-homologous source is annotated with captions automatically generated by Gemini 2.5 Pro [8] and manually checked for accuracy.

- Task 2: AnyP2V (Any-Timestamp Patch-to-Video). This variant follows the same nine sub-task definitions as AnyI2V setting, but replaces each full-frame condition with a cropped patch.

- • Patch extraction. For each conditional frame, patches are obtained via a semi-automated process: 50% object-aware masks using SAM [27] or YOLO [45], and 50% random crops.
- • Temporal anchors. The same start, middle, and end frame positions are used to construct single-, two-, and threeframe variants, for both homologous and non-homologous cases.
- • Difficulty. The subset explicitly includes challenging cases with very small subjects, requiring the model to extrapolate from minimal context.

- Task 3: AnyV2V (Transition, Inpainting and Outpainting). This task evaluates more general video-level completion scenarios beyond frame- or patch-level control. It consists of three sub-categories:

- • Video Transition. For 30 curated pairs of non-homologous video clips, the first clip provides the start segment and the second the end segment, while the model synthesizes the intermediate transition. This setup parallels the blending function explored in Sora [38]. Each case is annotated with captions generated by Gemini 2.5 Pro [8] and manually corrected to ensure faithful descriptions of both content and motion.
- • Inpainting. For homologous videos, interior rectangular masks are applied to each frame, covering 20%–50% of the width/height. The model must fill the missing regions with temporally consistent content.
- • Outpainting. Boundary masks are applied to crop the central region, masking out 60%–90% of the width/height. The model is required to extrapolate plausible outer regions beyond the visible content.

##### D.3. Scale

In total, VideoCanvasBench includes over 2,000 test cases: 900 for AnyP2V, 900 for AnyI2V, and 230 for AnyV2V. Each case is designed to probe a specific aspect of fidelity, creativity, or temporal reasoning in the proposed unified task.

- D.4. Licensing and Annotations.

All videos in our benchmark are sourced from Pexels [40], and images are sourced from both Pexels and Unsplash [46]. Content on Pexels is provided under the Pexels License, which permits free use for commercial and non-commercial purposes without requiring attribution, with restrictions against reselling unaltered copies, use in trademarks, or misuse of identifiable people or brands. A subset of Pexels content is explicitly marked as Creative Commons Zero (CC0), which places the work in the public domain. Unsplash photos are provided under the Unsplash License, which similarly allows free commercial and non-commercial use without attribution, while prohibiting resale of unaltered content, creation of competing stock services, or misleading association with brands or people. In both cases, all curated data is legally licensed for academic research use.

Captions generated by Gemini 2.5 Pro [8] were manually verified by the authors to ensure accuracy and consistency across all benchmark cases.

- E. More Analysis and Results

- E.1. Analysis of Zero-Padded Inputs

In Section 2, we describe using zero-padding to indicate unconditioned regions when preparing conditional frames. This approach is crucial for our spatial conditioning strategy, as it allows us to precisely specify the location of a condition patch within a frame without modifying the pre-trained VAE backbone. However, a critical question arises: can a standard hybrid video VAE, trained on natural images and videos, effectively handle inputs that contain large areas of zero-valued pixels (i.e., spatial padding)? As illustrated in Figure S10 and Figure S11, this distinction between spatial and temporal padding is fundamental to understanding our method.

To address this, we conducted an empirical study using two popular pre-trained VAE models: Hunyuan I2V and CogVideo. We evaluated their robustness to both spatial and temporal padding under realistic conditions.

Setup. We collected 20 diverse full-resolution images and 20 short video clips from YouTube, representing a wide range of content (e.g., landscapes, cityscapes, indoor scenes, moving vehicles). For each image, we applied random spatial zero-padding masks, covering approximately 40-60% of the pixels. For each video clip, we created three types of padded inputs: 1. A video with conditional frames containing the original content, while all other frames are filled with zeros (pure temporal padding). 2. A video where conditional frames contains cropped region of the original content, with all other frames being zero (temporal & spatial padding).

Each input was then encoded and decoded using the two hybird VAE model. We measured the reconstruction fidelity

using PSNR and qualitatively inspected the outputs.

Reconstruction Results. The results provide clear evidence of the differential impact of padding modes:

Spatial Padding Robustness: As shown in Figure S10, both VAE models demonstrate remarkable tolerance to spatial zero-padding. The average PSNR of reconstructed images with spatial padding is only marginally lower than that of the baseline (full image), with an average drop of 0.89 dB(Hunyuan I2V) and 1.13 dB(CogVideo).

Temporal Padding Vulnerability: In stark contrast, Figure S11 reveals the limitations of traditional approaches. When applying temporal zero-padding (encoding a single frame into a sequence where most frames are zero), both VAE models exhibit a dramatic degradation in reconstruction quality. The average PSNR drops by over 6.12 dB(Hunyuan I2V) and 7.01 dB(CogVideo) compared to the baseline.

Conclusion. These findings confirm that the key to achieving pixel–frame-aware control lies in decoupling spatial and temporal handling. Our method leverages the inherent robustness of the VAE to spatial padding while bypassing the ineffectiveness of temporal padding through our proposed Temporal RoPE Interpolation. This separation enables flexible, high-fidelity video completion using a frozen VAE without requiring retraining or architectural modification.

In addition to the controlled analyses presented above, our qualitative results under arbitrary spatiotemporal conditioning (Fig. 1 and Fig. S14) further demonstrate that spatial zero-padding remains stable even under large variations in placement, content, and temporal context. These observations provide complementary evidence supporting the effectiveness and general applicability of our approach across diverse zero-padded settings. Together with the quantitative results reported in the main text, these findings consistently validate the necessity and effectiveness of our design.

##### E.2. Advantages of Temporal RoPE Interpolation

Figure 7 in the main paper has shown that our Temporal RoPE Interpolation achieves precise one-to-one alignment between condition frames and their target temporal positions. Here we further demonstrate not only that our model can leverage this precision for dense conditioning, but also why this capability represents a crucial advantage over competing paradigms.

To this end, we conduct an additional experiment on the homologous video set from VideoCanvasBench. Each 77frame video is conditioned on the first five frames (0-4) in two different ways:

• Sparse Condition: Only the boundary frames (0 and 4) are provided. The model must interpolate the three missing frames (1, 2, 3) in between.

###### Input full image

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

Input image(spatial padding)

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

### (a) Input image

###### Rec. full image

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

PSNR=36.13 PSNR=37.86 PSNR=37.42 PSNR=36.79

[Figure 275]

###### Rec. image(spatial padding) spatial padding does not significantly reduce PSNR

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

PSNR=35.72 PSNR=37.34 PSNR=36.79 PSNR=35.04

### (b) Reconstruction image by Hunyuan I2V VAE model

Rec. full image

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

PSNR=35.12 PSNR=37.03 PSNR=36.89 PSNR=35.71

[Figure 284]

Rec. image(spatial padding) spatial padding does not significantly reduce PSNR

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

PSNR=34.88 PSNR=36.17 PSNR=36.14 PSNR=33.87

### (c) Reconstruction image by CogVideo VAE model

- Figure S10. Robustness of Hybrid Video VAEs to Spatial Padding. This figure demonstrates that both the Hunyuan I2V and CogVideo VAE models can tolerate spatial zero-padding well. When reconstructing images with large zero-padded regions (middle row), the PSNR values are only slightly lower than those of the full, unpadded images (top row). Crucially, the original content within the non-zero regions is faithfully preserved, while the padded areas remain visually neutral. This empirical evidence confirms that our spatial conditioning strategy, which relies on zero-padding before VAE encoding, is stable and practical, enabling precise spatial control without degrading the quality of the conditioned content.

• Dense Condition: All five frames (0, 1, 2, 3, 4) are explicitly provided as conditions, testing frame-wise alignment at every step.

evaluate the fidelity by computing PSNR on the first 5 frames against the ground truth.

The quantitative results in Table R4 confirm that explicitly conditioning on consecutive frames yields higher reconstruc-

Both settings are used to generate the full video, and we

###### Input full video

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

Input image(temporal padding)

[Figure 294]

[Figure 295]

Input image(temporal&spatial padding)

[Figure 296]

[Figure 297]

0 1 2 3 4

(a) Input video

Rec. full frame Rec. full frame

Rec. full frame

(temporal padding)

(temporal&spatial padding)

[Figure 298]

[Figure 299]

[Figure 300]

PSNR=34.58 PSNR=28.91 PSNR=27.63

[Figure 301]

tempotal padding does significantly reduce PSNR

- (b) Reconstruction frame by Hunyuan I2V VAE model
- (c) Reconstruction frame by CogVideo VAE model

[Figure 302]

[Figure 303]

[Figure 304]

PSNR=33.69 PSNR=26.54 PSNR=26.13

[Figure 305]

tempotal padding does significantly reduce PSNR

- Figure S11. Vulnerability of Hybrid Video VAEs to Temporal Padding. This figure contrasts the robustness observed in spatial padding. When applying temporal zero-padding (where only specific frames contain content), both VAE models suffer a relatively great drop in reconstruction quality. The PSNR values for the padded reconstructions (bottom rows) are much lower than those of the full video (top row), demonstrating a degradation in fidelity. The reconstructed frames exhibit noticeable color shifts, and loss of detail, highlighting that the VAE cannot handle such distributionally mismatched inputs. This mode underscores why direct temporal zero-padding is ineffective and validates the necessity of our Temporal RoPE Interpolation strategy, which avoids this problem by operating at the latent token level with fractional positions.

Table R4. Average PSNR (dB) across 100 videos under sparse vs. dense conditioning.

tion, but with minor, expected drift in the unconditioned intermediate frames. The dense case, in contrast, achieves a near-perfect reconstruction.

Condition Type Conditioned Frames PSNR (↑)

This comparison highlights a fundamental limitation of paradigms like Channel Concatenation. Due to their coarse, slot-based nature and the constraint of a frozen VAE, they can only condition on one frame per latent slot (e.g., one frame for every N = 4 pixel frames). They are therefore structurally incapable of providing dense guidance for the intermediate frames (e.g., frames 1, 2, 3) and are locked into a ”sparse” conditioning mode, inevitably suffering from the

Sparse (two frames) 0, 4 24.789 Dense (five frames) 0, 1, 2, 3, 4 25.033

tion fidelity. Figure S12 provides a visual illustration. In the sparse case, our model generates a plausible interpola-

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

Input

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

Sparse Condition (Frame 0/4)

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

Dense Condition (Frame 0/1/2/3/4)

Frame 0 Frame 1 Frame 2 Frame 3 Frame 4

- Figure S12. Visual comparison of Sparse vs. Dense conditioning. The top row shows the ground-truth frames. The middle row (Sparse) is generated using only frames 0 and 4 as conditions; note the plausible but slightly drifted interpolation for the intermediate frames. The bottom row (Dense) is generated using all five frames as conditions, resulting in a near-perfect reconstruction. This highlights the benefit of dense, frame-by-frame control—a capability unique to our method.

Table R5. Quantitative ablation of RoPE strategies. We report the FVD (↓) scores (lower is better) on the validation set. Our fractional interpolation strategy outperforms both integer extrapolation and the baseline without RoPE.

kind of interpolation drift shown in our sparse example. In contrast, our Temporal RoPE Interpolation uniquely enables true dense conditioning, allowing VideoCanvas to maintain high fidelity frame-by-frame—a capability that is structurally inaccessible to these competing methods.

Method Index Mapping Strategy FVD (↓)

##### E.3. More Ablation and User Studies

w/o RoPE N/A 12.568 Integer Extrapolation 0, 1, . . . , 76 11.079 Ours (Fractional) 0, 0.25, . . . , 19 10.943

###### E.3.1. Ablation of RoPE Strategy

First Frame Last Frame

[Figure 321]

[Figure 322]

better performance compared to the Integer method, which shifts the distribution to an unfamiliar range (0 ∼ 76).

Condition

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

Furthermore, removing RoPE proves disastrous. Without explicit positional indicators, the condition tokens injected via concatenation lack the temporal cues necessary to “lock” onto specific frame positions. Consequently, the mechanism degenerates from precise conditioning into a loose “image reference” mode, where the generated frames fail to strictly adhere to the condition frames. This degradation is visually evident in Fig. S13, where the model fails to maintain temporal position consistency.

w/o RoPE

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

w/ RoPE

Figure S13. Visualization of without RoPE.

We further investigate the impact of different Rotary Positional Embedding (RoPE) variant strategies on temporal alignment. Specifically, we compare our proposed Fractional Interpolation (mapping frames to 0,0.25,...,19) against an Integer Extrapolation strategy (scaling indices to 0,1,...,76 for a 4× expansion) and a baseline w/o RoPE.

###### E.3.2. User Study on Conditioning Paradigms

To complement the quantitative results in Table 2, where our method achieves the best FVD and Dynamic Degree, we further conducted a human preference study using the diverse tasks defined in the VideoCanvas benchmark. We compare our In-Context Conditioning (ICC) against the two representative paradigms: Latent Replacement and Channel Concatenation.

As reported in Table R5, our method achieves the best performance with the lowest FVD score. We attribute the superiority of our fractional strategy to the preservation of pre-trained priors. Since the base model was trained on a specific temporal range (indices 0 ∼ 19), our fractional approach ensures the input indices remain within this learned distribution. This allows the model to effectively inherit the original temporal priors, leading to faster convergence and

Setup. We sampled generated videos across the broad range of tasks in VideoCanvas. Evaluators performed pairwise comparisons based on visual quality, motion smoothness, and condition fidelity.

Results. As shown in Table R6, our method consistently outperforms both baselines:

- • Vs. Latent Replacement: Ours achieves a 65.4% win rate. Users penalized the replacement method for its lower dynamic degree (consistent with Table 2) and disjointed motion transitions.
- • Vs. Channel Concatenation: Ours wins by 48.2% (with a significant tie rate). This confirms that our parameterfree ICC strategy is more effective than increasing channel dimensionality (16.6M˜ extra params), yielding better visual harmony.

These results align with the quantitative metrics in Table 2, verifying that ICC provides the most effective guidance for the diffusion model.

- Table R6. Human Preference Evaluation. Comparison of our InContext Conditioning (Ours) against two representative paradigms: Latent Replacement and Channel Concatenation. We report the percentage of user votes favoring our model versus the variants, averaged across all evaluated tasks on the VideoCanvas benchmark.

Comparison

Preference Rate (Ours vs. Variant) Win (↑) Tie Loss (↓)

Ours vs. Latent Replacement 65.4% 22.1% 12.5% Ours vs. Channel Concatenation 48.2% 28.3% 23.5%

E.4. Training and Inference Cost

- Table R7. Training and inference cost comparison across paradigms. Training time is measured over 20k steps. Inference time is per 77-frame video at 384 × 672 with different numbers of conditional frames.

Method New Params Train Inference 1 frame 2 frame 3 frame

Replacement 0 21.47h 159s 159s 159s Channel Concat 16.6M 22.47h 164s 164s 164s ICC (Ours) 0 24.54h 168s 175s 184s

Tab. R7 compares the computational cost of different conditioning paradigms. Unlike channel concatenation, which relies on a 16.6M projection layer, our ICC design introduces no additional parameters, and the training cost remains comparable (24.5h vs. 21–22h) since ICC only adds lightweight spatio-temporal tokens.

During inference, ICC exhibits a content-aware and controllable scaling: the compute grows with the number of conditioning frames, because a richer conditioning context requires processing longer input sequences within the transformer. For a 77-frame video at 384 × 672, inference takes 168 s with a single conditioning frame, and gradually increases to 333 s / 520 s / 910 s when conditioning on 20 / 40 / 77 frames, respectively.

In sparse conditioning scenarios, the additional cost is almost negligible, since only a few frames expand the sequence length. When more frames are provided, ICC allows users to intentionally trade additional computation for higher fidelity on the corresponding conditioned timestamps, offering stronger control instead of incurring unavoidable overhead. While this scaling makes ICC marginally slower than baselines with fixed-cost inference, the trade-off is justified, as ICC consistently yields higher fidelity and better spatio-temporal alignment (see Sec. 4.2, Tab. 1 and Tab. 2).

This cost profile is orthogonal to recent accelerationoriented designs. For example, FullDiT2 [19] improves ICC efficiency through token selection and context caching, while SURF [10] targets fast high-resolution video generation by preserving the signature of a pretrained generator. These directions could be integrated with our pixel-frame-aware conditioning to further reduce latency without changing the core alignment mechanism.

#### F. Applications and Qualitative Results

##### F.1. Applications

The teaser figure (Fig. 1) has shown some cases, and in this section, we provide extensive qualitative results to demonstrate the versatility and effectiveness of our VideoCanvas framework across a wide range of applications.

- • Any-Timestamp Patch-to-Video (AnyP2V). In Figure S14, we demonstrate our core capability of generating a complete video from a varying number of sparse patches. We showcase challenging scenarios using one, two, three, and even four conditional patches, placed at arbitrary timestamps to rigorously test the model’s spatiotemporal reasoning beyond simple first-frame conditioning.
- • Any-Timestamp Image-to-Video (AnyI2V). Figure S15 illustrates the flexibility of our framework on full-frame conditions. The examples include standard cases like firstframe I2V and first-last-frame interpolation, as well as more challenging scenarios where conditions are placed at arbitrary middle timestamps, a capability not well supported by prior methods.
- • Video-Level Completion and Creation (AnyV2V). Our framework naturally unifies a variety of video editing and creation tasks within a single model. We provide examples of:

- – Video Transition: Creative transitions between nonhomologous clips are demonstrated in Figure S16.
- – Video Painting: Inpainting and outpainting results are shown in Figure S17, where the red dashed contours indicate the generated regions.
- – Video Extension and Looping: As demonstrated in Figure S18, we showcase long-duration synthesis by extending short clips to over a minute in length while

maintaining temporal consistency. This capability can be guided by interactive text prompts to evolve the narrative. Furthermore, we can create perfectly seamless loops by generating a smooth transition from the video’s end back to its beginning. Our approach leverages motion context from the last segment’s frames to effectively avoid the stuttering artifacts that are common in naive first-last frame-looping methods.

– Video Camera Control: As demonstrated in Figure S19, our framework can emulate camera cinematography by progressively translating or scaling content on the spatio-temporal canvas. This enables a variety of standard camera effects, such as zooms and pans. We showcase this capability by applying dynamic camera movements to classic movie shots, demonstrating its potential for creative post-production.

##### F.2. Qualitative Comparison

The following figures showcase side-by-side comparisons with baseline paradigms, illustrating our method’s superior performance in motion smoothness, detail sharpness, and temporal consistency.

Fig. S20 and Fig. S21 present qualitative results across the six varied tasks (I2V, FLF2V, TF2V, FLP2V, Inpainting, and Outpainting), visually confirming our framework’s strong and consistent performance across these diverse domains.

Finally, Figure S22 provides additional direct comparisons against different paradigms across a diverse set of challenging cases, further highlighting the robustness and superiority of our approach.

[Figure 331]

Condition

[Figure 332]

Frame 0 Frame 40 Frame 80 Frame 156

[Figure 333]

Generated Video

[Figure 334]

(a) case 1 “Blue airplane races ocean, glaciers, canyon, into bustling city…”

[Figure 335]

Condition

[Figure 336]

Frame 0 Frame 80 Frame 156

[Figure 337]

Generated Video

[Figure 338]

(b) case 2 “Yellow race car speeds forest, snow, then coastal highway...”

[Figure 339]

Condition

[Figure 340]

Frame 0 Frame 76

[Figure 341]

Generated Video

[Figure 342]

(c) case 3 “Elderly man, old house morphs into modern café...”

[Figure 343]

Condition

[Figure 344]

Frame 40

[Figure 345]

Generated Video

[Figure 346]

(d) case 4 “Young couple with dog walking serene autumn path…”

###### Figure S14. Results on Any-timestamp Patches to Videos.

[Figure 347]

Condition

[Figure 348]

Frame 0

[Figure 349]

Generated Video

| |
|---|

[Figure 350]

(a) case 1 “Young woman holds purple flowers by sunlit creek...”

[Figure 351]

Condition

[Figure 352]

Frame 0 Frame 40 Frame 76

[Figure 353]

Generated Video

[Figure 354]

(b) case 2 “Tree morphs into glowing object, erupts as volcano...”

[Figure 355]

Condition

[Figure 356]

Frame 0 Frame 40

[Figure 357]

Generated Video

[Figure 358]

(c) case 3 “Zebra leaps, transforms into colorful kite in sky...”

[Figure 359]

Condition

[Figure 360]

Frame 0 Frame 76

[Figure 361]

Generated Video

[Figure 362]

(d) case 4 “Child joyfully splashes in sunlit beach shallows...”

###### Figure S15. Results on Any-timestamp Images to Videos.

[Figure 363]

Condition

[Figure 364]

[Figure 365]

First Clip Last Clip

[Figure 366]

Generated Video

[Figure 367]

(a) case 1 (Without Text Prompt)

[Figure 368]

Condition

[Figure 369]

[Figure 370]

First Clip Last Clip

[Figure 371]

Generated Video

[Figure 372]

(b) case 2 “A continuous shot from a drone flying to a butterfly”

[Figure 373]

Condition

[Figure 374]

[Figure 375]

First Clip Last Clip

[Figure 376]

Generated Video

[Figure 377]

(c) case 3 “Off-road vehicle morphs into leopard sprinting jungle...”

[Figure 378]

Condition

[Figure 379]

[Figure 380]

First Clip Last Clip

[Figure 381]

Generated Video

[Figure 382]

(d) case 4 “Escalator ascends into arctic sky, woman runs snowy landscape...”

###### Figure S16. Results on Video Transition.

[Figure 383]

Source Video

[Figure 384]

[Figure 385]

Generated Video

[Figure 386]

(a) Video Inpainting case 1 “Cartoon bear surrenders humorously to rooftop gunman…”

[Figure 387]

Source Video

[Figure 388]

[Figure 389]

Generated Video

- (b) Video Inpainting case 2 “Léon and cartoon bear walk, carrying plants together…”

[Figure 390]

[Figure 391]

Generated Video

[Figure 392]

Source Video

- (c) Video Outpainting case 1 “Silver-haired sorceress stands in snowy fantasy landscape.…”

[Figure 393]

[Figure 394]

Generated Video

[Figure 395]

Source Video

- (d) Video Outpainting case 2 “Volcanic plain with dormant volcano under orange sky..…”

[Figure 396]

[Figure 397]

[Figure 398]

- Figure S17. Results on Video Inpainting and Outpainting. The red dashed contours indicate the regions that are subject to inpainting or outpainting.

|[Figure 399]| | |[Figure 400]<br><br>[Figure 401]| |
|---|---|---|---|---|

|[Figure 402]| | |[Figure 403]<br><br>[Figure 404]| |
|---|---|---|---|---|

|[Figure 405]| |[Figure 406]| |[Figure 407]|
|---|---|---|---|---|

|[Figure 408]|[Figure 409]| | |[Figure 410]|
|---|---|---|---|---|

| |[Figure 411]<br><br>[Figure 412]| | |[Figure 413]|
|---|---|---|---|---|

| |[Figure 414]| |[Figure 415]<br><br>[Figure 416]| |
|---|---|---|---|---|

| |[Figure 417]| |[Figure 418]<br><br>[Figure 419]| |
|---|---|---|---|---|

Initial shot drives along a coastal highway past a beautiful village past a golden field past an ice canyon past a classic-style building past a Hawaii beach into a mountain trail

[Figure 420]

- Figure S18. Results on Video Extension and Seamless Looping. The example showcases a video extended to over 1,000 frames by first applying our video extension capability and then generating a seamless transition back to the initial state. This highlights our model’s ability to maintain temporal consistency and visual quality over a long generation horizon without suffering from quality degradation or motion collapse.

[Figure 421]

Source Video

[Figure 422]

Translate & Scale

[Figure 423]

Condition

[Figure 424]

[Figure 425]

Generated Video

[Figure 426]

[Figure 427]

###### Condition

[Figure 428]

[Figure 429]

Generated Video

[Figure 430]

(a) Case 1: Video Clip From The Great Gatsby

[Figure 431]

Source Video

[Figure 432]

Translate & Scale

[Figure 433]

Condition

[Figure 434]

[Figure 435]

Generated Video

[Figure 436]

[Figure 437]

Condition

[Figure 438]

[Figure 439]

Generated Video

[Figure 440]

(b) Case 2: Video Clip From Titanic

- Figure S19. Results on Video Camera Control. The examples showcase emulated camera effects such as zoom and pan, achieved by progressively translating and scaling content on the spatio-temporal canvas.

[Figure 441]

A woman in a white dress and a wide-brimmed hat walks barefoot along a wooden suspension bridge surrounded by lush greenery. She holds her hat with one hand…

Input

First Frame

[Figure 442]

Hunyuan-13B

[Figure 443]

Wan-14B

[Figure 444]

CogVideoX

[Figure 445]

VideoCanvas (Ours)

TASK: First-Frame-to-Video (I2V)

[Figure 446]

[Figure 447]

a bustling urban scene featuring a prominent, tall, beige building with numerous windows and a classic architectural design. The building is situated in the center of a busy city...

Input

First Frame Last Frame

[Figure 448]

CogVideoX

[Figure 449]

Sci-Fi

[Figure 450]

VideoCanvas (Ours)

TASK: First-Last-Frame-to-Video (FLF2V)

a first-person perspective of a motorcycle ride along a winding road surrounded by lush greenery. The rider navigates through a series of curves...

[Figure 451]

[Figure 452]

[Figure 453]

Input

First Frame Middle Frame

Last Frame

[Figure 454]

CogVideoX

[Figure 455]

Sci-Fi

[Figure 456]

VideoCanvas (Ours)

TASK: First-Middle-Last-Frame-to-Video (TF2V)

###### Figure S20. Comparisons with baseline models (1/2).

[Figure 457]

[Figure 458]

a breathtaking aerial view of a vast, green landscape with a prominent rocky cliff. The camera captures the expansive terrain, featuring rolling hills and a winding road that stretches…

Input

First Patch Last Patch

[Figure 459]

Flux + CogVideoX

[Figure 460]

Flux + Wan

[Figure 461]

VideoCanvas (Ours)

TASK: First-Last-Patch-to-Video (FLP2V)

A white bus is driving down a dirt road in a rural area, surrounded by buildings...

[Figure 462]

Input

[Figure 463]

Propainter

[Figure 464]

VACE

[Figure 465]

VideoCanvas (Ours)

TASK: Video Inpainting

a tender moment between two monkeys in a natural setting. One monkey is grooming the other...

[Figure 466]

Input

[Figure 467]

M3DDM

[Figure 468]

VACE

[Figure 469]

VideoCanvas (Ours)

TASK: Video Outpainting

###### Figure S21. Comparisons with baseline models (2/2).

[Figure 470]

Condition

[Figure 471]

[Figure 472]

Frame 0 Frame 76

[Figure 473]

Replace.

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

Unnatural Transition Too Static

Channel.

[Figure 478]

[Figure 479]

[Figure 480]

Too Static

ICC(Ours)

[Figure 481]

(a) case 1 “Cheetah morphs into racing car on stadium track…”

[Figure 482]

Condition

[Figure 483]

[Figure 484]

[Figure 485]

Frame 0 Frame 76

Frame 40

[Figure 486]

Replace.

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

Deformed

Too Static Unnatural Transition

[Figure 491]

Channel.

[Figure 492]

[Figure 493]

[Figure 494]

Deformed

ICC(Ours)

[Figure 495]

- Condition (b) case 2 “Two pigeons calmly interact in peaceful natural setting…”

[Figure 496]

[Figure 497]

ICC(Ours)

[Figure 498]

Channel.

[Figure 499]

Replace.

[Figure 500]

Too Static

[Figure 501]

Unnatural Transition

Frame 0 Frame 76

- Condition (c) case 3 “Fawn forest run transitions to snowy sledding scene…”

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

First Clip Last Clip

[Figure 510]

Replace.

[Figure 511]

[Figure 512]

Too Static

[Figure 513]

Channel.

[Figure 514]

[Figure 515]

[Figure 516]

Unnatural Transition

ICC(Ours)

[Figure 517]

(c) case 4 “Escalator ascends into arctic sky, woman runs snowy landscape…”

###### Figure S22. Comparisons with baseline paradigms.

