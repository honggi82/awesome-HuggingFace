# arXiv:2512.07802v1[cs.CV]8Dec2025

## OneStory: Coherent Multi-Shot Video Generation with Adaptive Memory

Zhaochong An1,2, Menglin Jia1, Haonan Qiu1, Zijian Zhou1, Xiaoke Huang1, Zhiheng Liu1, Weiming Ren1, Kumara Kahatapitiya1, Ding Liu1, Sen He1, Chenyang Zhang1, Tao Xiang1, Fanny Yang1, Serge Belongie2, Tian Xie1,∗

1Meta AI, 2University of Copenhagen

∗Project lead

Storytelling in real-world videos often unfolds through multiple shots—discontinuous yet semantically connected clips that together convey a coherent narrative. However, existing multi-shot video generation (MSV) methods struggle to effectively model long-range cross-shot context, as they rely on limited temporal windows or single keyframe conditioning, leading to degraded performance under complex narratives. In this work, we propose OneStory, enabling global yet compact cross-shot context modeling for consistent and scalable narrative generation. OneStory reformulates MSV as a next-shot generation task, enabling autoregressive shot synthesis while leveraging pretrained image-to-video (I2V) models for strong visual conditioning. We introduce two key modules: a Frame Selection module that constructs a semantically-relevant global memory based on informative frames from prior shots, and an Adaptive Conditioner that performs importance-guided patchification to generate compact context for direct conditioning. We further curate a high-quality multi-shot dataset with referential captions to mirror real-world storytelling patterns, and design effective training strategies under the next-shot paradigm. Finetuned from a pretrained I2V model on our curated 60K dataset, OneStory achieves state-of-the-art narrative coherence across diverse and complex scenes in both text- and image-conditioned settings, enabling controllable and immersive long-form video storytelling.

Correspondence: zhan@di.ku.dk Project page: https://zhaochongan.github.io/projects/OneStory

1 Introduction

Recent advances in diffusion transformers (Peebles and Xie, 2023) have greatly advanced video generation (Yu et al., 2025; Chu et al., 2025; Song et al., 2025a; Wang et al., 2025c), achieving impressive visual fidelity. Despite this success, current models remain largely confined to producing a single continuous scene, ignoring long-range narrative modeling. In contrast, real-world applications (Xing et al., 2025; Gao et al., 2025; Qiu et al., 2025) demand multi-shot videos: a sequence of shots that together convey a coherent storyline. Consequently, multi-shot video generation (MSV) is emerging as a critical research direction (Wang et al., 2024c; Lu et al., 2025; Jiang et al., 2025; Atzmon et al., 2024).

Compared to single-shot generation, MSV is inherently more challenging as it requires both narrative consistency and spatio-temporal reasoning across discontinuous scenes (Liu et al., 2025; Wang et al., 2025d). First, consistent narrative entities, such as characters and environments, must persist even when intermittently off-screen (Guo et al., 2025b; Song et al., 2025b). Second, as consecutive shots may vary in terms of time, location, and viewpoint, the model must discern which aspects should remain invariant (e.g., identity, scene layout) and which should evolve (e.g., camera angles, actions) (Lin et al., 2025a; Liao et al., 2025; Chen et al., 2025). In essence, the core difficulty of MSV lies in effectively exploiting and maintaining the long-term cross-shot context.

Based on how the cross-shot context is modeled, existing approaches can be categorized into two paradigms. (1) Fixed-window attention paradigm (Wu et al., 2025a; Kara et al., 2025; Qi et al., 2025; Guo et al., 2025b) computes attention over several shots within a fixed temporal window, by applying caption-to-shot attention masks (Kara et al., 2025; Qi et al., 2025) or direct long context tuning (Guo et al., 2025b). However, due to

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Image-to-Multi-Shot

| | |[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]| | |
|---|---|---|---|---|

|[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]|[Figure 18]| |[Figure 19]| |
|---|---|---|---|---|

| | |[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]| | |
|---|---|---|---|---|

| | |[Figure 26]|[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]| |
|---|---|---|---|---|

| | |[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]| | |
|---|---|---|---|---|

[Shot 1] a cat floating on a board [Shot 2] the cat sleeps on the board [Shot 3] the cat runs cross the water [Shot 4] the cat under a sun umbrella [Shot 5] the cat watching the sea

| |[Figure 38]<br><br>[Figure 39]| | |[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]|
|---|---|---|---|---|

| |[Figure 44]| |[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]| |
|---|---|---|---|---|

| |[Figure 50]<br><br>[Figure 51]| | |[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]|
|---|---|---|---|---|

| |[Figure 56]| |[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]| |
|---|---|---|---|---|

| | |[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]| | |
|---|---|---|---|---|

[Shot 6] shot of the shoreline [Shot 7] the cat finds a piece of wood [Shot 8] the cat walks on the wood [Shot 9] day turns to night [Shot 10] the cat sleeps on the shore

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Text-to-Multi-Shot

| | |[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]| | |
|---|---|---|---|---|

|[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]| | |[Figure 86]<br><br>[Figure 87]| |
|---|---|---|---|---|

| | |[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]| | |
|---|---|---|---|---|

| | |[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]| | |
|---|---|---|---|---|

| |[Figure 100]<br><br>[Figure 101]| | |[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]|
|---|---|---|---|---|

[Shot 1] a woman talks about a plant [Shot 2] close-up of the green plant [Shot 3] close-up of the woman [Shot 4] a man starts speaking [Shot 5] the woman continues taking

| |[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]| |[Figure 111]| |
|---|---|---|---|---|

|[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]|[Figure 116]| | |[Figure 117]|
|---|---|---|---|---|

|[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]| | |[Figure 122]|[Figure 123]|
|---|---|---|---|---|

| |[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]| | |[Figure 129]|
|---|---|---|---|---|

|[Figure 130]| | |[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]| |
|---|---|---|---|---|

[Shot 6] close-up of the same plant [Shot 7] wide shot of both in the garden [Shot 8] scene shifts to a lake [Shot 9] the man speaks by the lake [Shot 10] back of both by the lake

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Out-of-Domain

| | |[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]| | |
|---|---|---|---|---|

| |[Figure 148]<br><br>[Figure 149]| | |[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]|
|---|---|---|---|---|

| |[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]|[Figure 159]| | |
|---|---|---|---|---|

| | |[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]| | |
|---|---|---|---|---|

| |[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]| | |[Figure 171]|
|---|---|---|---|---|

[Shot 1] a citizen meets a police [Shot 2] they discuss photos on the desk [Shot 3] close-up of the photo [Shot 4] both rise and argue [Shot 5] over-shoulder view by window

|[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]| |[Figure 176]<br><br>[Figure 177]| | |
|---|---|---|---|---|

| | |[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]| | |
|---|---|---|---|---|

|[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]| | |[Figure 188]<br><br>[Figure 189]| |
|---|---|---|---|---|

| |[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]| | |[Figure 195]|
|---|---|---|---|---|

|[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]| | |[Figure 200]<br><br>[Figure 201]| |
|---|---|---|---|---|

[Shot 6] the citizen’s uneasy face [Shot 7] other officers join argument [Shot 8] close-up of the police [Shot 9] the police making a call [Shot 10] a new police in suit arrives

Figure 1 Coherent multi-shot generations with OneStory. Each example shows 10-shots of a minute-long video. OneStory handles both image-to-multi-shot (top) and text-to-multi-shot (middle) generation within the same model, and generalizes well to out-of-domain scenes (bottom). It maintains consistent characters and environments while faithfully following complex and evolving prompts to produce coherent long-form narratives. A representative segment of each prompt is given with the corresponding shot. We recommend referring to our Project Page for better visualization.

the fixed window size, older shots are discarded as the window slides forward, leading to inevitable memory loss and inconsistency beyond the window. (2) Keyframe conditioning paradigm (He et al., 2025; Zhao et al.,

- 2024; Long et al., 2024; Xiao et al., 2025) generates a keyframe for each shot and expands it into a full clip using image-to-video (I2V) models before concatenation. However, such multi-stage pipelines restrict the cross-shot context to a single image, limiting the propagation of complex narrative cues and resulting in weak storyline adherence.

To this end, we propose OneStory, an effective framework that overcomes the context modeling limitations in prior work. First, we reformulate MSV as a next-shot generation task, to enable autoregressive shot synthesis and leverage the strong visual conditioning capability of pretrained I2V models. Second, inspired by varying correlations across shots, we introduce a Frame Selection module that identifies a sparse set of semantically-relevant frames across all prior shots, effectively mitigating memory loss and recovering long-range

context. Third, we design an Adaptive Conditioner that patchifies the selected context dynamically and injects directly into the generator, providing efficient and expressive conditioning. Unlike prior works (Gu et al.,

- 2025; Zhang and Agrawala, 2025) that rely on a fixed temporal ordering, our conditioner compresses a set of disconnected frames adaptively based on their importance. Together, our model enables an global yet compact cross-shot context, supporting consistent and scalable story generation.

Beyond model design, we carefully curate a high-quality dataset of ∼60K multi-shot videos through a three-step pipeline comprising shot detection, two-stage captioning, and quality filtering (see Section 3). This dataset reflects realistic storytelling where captions are provided shot by shot in a referential narrative flow, providing a better flexibility to evolve, without the need of having a separate global script (Guo et al., 2025b; Wu

- et al., 2025a). We also propose effective training strategies under the new MSV formulation, including unified three-shot training and a progressive coupling scheme, to facilitate end-to-end optimization and narrative coherence.

Built upon a pretrained I2V model and finetuned on our curated dataset, OneStory achieves superior performance across diverse and complex narratives. As shown in Figure 1, it generates minute-scale, ten-shot videos with strong visual consistency and narrative adherence. It supports both image-to-multi-shot (top) and text-to-multi-shot (middle), while also generalizing to out-of-domain scenes (bottom) despite trained only on human-centric data. The coherence, flexibility, and scalability of OneStory make it well-suited for real-world creative applications, paving the way for immersive multi-shot storytelling.

- 2 Related Works

Single-shot Video Generation. Single-shot video generation primarily includes text-to-video (T2V) and imageto-video (I2V) models. Early T2V models (Zhang et al., 2024; Chen et al., 2024a; Wang et al., 2024a; Luo

- et al., 2023; Zeng et al., 2024; Bar-Tal et al., 2024) extended text-to-image diffusion architectures (Rombach

- et al., 2022) with temporal modules, while the emergence of Diffusion Transformers (DiT) (Peebles and Xie,

2023) enabled unified spatial–temporal modeling (Brooks et al., 2024; Polyak et al., 2024; Kuaishou, 2024; HaCohen et al., 2024) through transformer-based designs. Large-scale T2V models such as Wan (Wan et al., 2025), HunyuanVideo (Kong et al., 2024), CogVideoX (Yang et al., 2024), and Mochi (Team, 2024) further boost fidelity by leveraging billion-scale datasets. In parallel, I2V models (Chen et al., 2023, 2024b; Shi et al., 2025; Hu, 2024; Lin et al., 2025b; Huang et al., 2025; Jiang et al., 2024) animate static images with textual conditioning, offering stronger visual realism and controllability (Guo et al., 2025a; Wang et al., 2025a; Li et al., 2025). Despite these advances, these methods remain confined to single-shot generation and thus fall short for real-world storytelling, motivating the exploration toward multi-shot video generation.

Multi-shot Video Generation. Recent efforts in MSV primarily follow two paradigms for modeling cross-shot context: (i) Fixed-window attention. These methods (Wu et al., 2025a; Bansal et al., 2024; Qi et al., 2025; Kara et al., 2025; Wei et al., 2025; Guo et al., 2025b; Cai et al., 2025; Jia et al., 2025; Meng et al., 2025; Wang et al., 2025b; Wu et al., 2025b) compute attention across multiple shots within a bounded temporal window. Mask2DiT (Qi et al., 2025) modifies attention masks to enforce caption–shot alignment, while LCT (Guo et al., 2025b) augments MMDiT (Esser et al., 2024) to encode multi-shot structure. However, their finite window inevitably discards earlier shots as the window slides forward, leading to memory loss and narrative inconsistency. (ii) Keyframe conditioning. Another line of work (Xie et al., 2024; Zhao et al., 2024; Zheng

- et al., 2024; Hu et al., 2024; Xiao et al., 2025; He et al., 2025; Zhang et al., 2025) decomposes multi-shot generation into subproblems by synthesizing a keyframe (or reference image (Long et al., 2024)) for each shot and expanding it into a full clip with an I2V model (Xing et al., 2024; Seawead et al., 2025). Captain Cinema (Xiao et al., 2025), for example, fine-tunes a text-to-image model (Black Forest Labs, 2024) for identity persistence. However, relying on only one keyframe per shot limits cross-shot context, hindering the propagation of complex narrative information and weakening storyline consistency. These limitations motivate our proposed OneStory, which models a global yet compact cross-shot context for consistent and scalable narrative generation.

### 3 High-quality Multi-shot Video Dataset

We define a high-quality multi-shot video as one maintaining a consistent theme across shots with coherent narrative progression, rather than a concatenation of unrelated clips. Regarding captions, existing MSV methods (Guo et al., 2025b; Wu et al., 2025a; Cai et al., 2025) typically rely on structured global prompts describing the overall storyline, characters, and environment, supplemented by per-shot prompts for local details. While such global scripts provide the model with overarching guidance, they restrict how subsequent shots can evolve beyond the predefined storyline. In contrast, we do not rely on a global script and rather construct shot-level captions with referential narrative flow derived from preceding shots, offering greater narrative flexibility and reflecting real-world storytelling, where shots evolve naturally from prior context.

As illustrated in Figure 2, our dataset is built from videos under research copyright, focusing on human-centric activities. We first apply TransNetV2 (Soucek and Lokoc, 2024) to detect shot boundaries and retain videos containing at least two shots. Next, we use a vision-language model (Meta, 2025; Bai et al., 2025; Yuan et al., 2025) for shot-level captioning in two stages: (i) captioning each shot independently, then (ii) rewriting subsequent captions based on the frames and caption of the previous shot, to introduce referential expressions (e.g., “the same man”) and describe scene/object variations. This approach ensures contextual linkage and smooth narrative flow across shots. After captioning, we perform multi-stage filtering to ensure quality. We apply keyword filters to remove videos with undesirable content. Then, we use feature-based filters, i.e., CLIP (Radford et al., 2021) and SigLIP2 (Tschannen et al., 2025), to eliminate videos with completely irrelevant transitions, and DINOv2 (Oquab et al., 2023) to discard overly similar shots. The resulting dataset contains approximately 60K high-quality multi-shot videos (50K two-shot and 10K three-shot), each exhibiting narrative continuity, forming a solid foundation for training multi-shot generation models.

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

Shot Detection

Two-stage Captioning

Quality Filtering

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

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

Raw Videos

High-quality Muti-shot Videos

- Shot 2 Shot-level caption: A man in a white shirt and a woman in a blue jacket are walking on the street. Rewritten caption: The same man, now wearing a white shirt, is walking on the street with a woman in a blue jacket.

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

Shot 1 Shot-level caption: A man in a black shirt is sitting in a kitchen, eating breakfast. Rewritten caption: (unchanged)

- Shot 3 Shot-level caption: A man in a white shirt is working on a laptop in an office. Rewritten caption: The same man in a white shirt is working on a laptop in an office.

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

Novel-Base Mix

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

Figure 2 Multi-shot video data curation pipeline. From raw videos, we obtain high-quality multi-shot sequences via three steps: (i) Shot detection, (ii) Two-stage captioning, and (iii) Quality filtering. In the second stage, each shot is first captioned independently and then rewritten into referential form based on preceding shots. Unlike prior datasets, no global captions are used, and only shot-level captions with progressive narrative flow are retained to ensure flexibility, while reflecting real-world storytelling.

### 4 Method

- 4.1 Task Formulation

Let an N-shot video be denoted as V = {S1,...,SN}, where each shot Si contains K frames Si = {fi1,...,fiK} with spatial resolution H × W. Each shot is paired with a caption Ci that explicitly references prior shots, as detailed in Section 3. Given captions {Ci}Ni=1 and an optional starting image I as the first-frame condition, a multi-shot video generation (MSV) model G aims to synthesize V that faithfully follows the narrative while preserving visual consistency.

To enable autoregressive shot generation and leverage the strong visual conditioning capabilities of pretrained image-to-video (I2V) models, we reformulate MSV as a next-shot generation task:

Si = G E, {Sj}ij−=11, T , Ci , (1)

where E is a 3D VAE encoder (Polyak et al., 2024; Wan et al., 2025) that maps each shot Si into latent features zi ∈ R

ft×fHs ×fWs ×Dv, with ft and fs denoting temporal and spatial compression factors, and Dv the

K

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

(b) Inference Phase

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

Historical Synthetic Memory

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

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

[Figure 300]

[Figure 301]

[Figure 302]

Real Shot 2

or

Shot 1 Shot 1 Shot N

Shot 1

Shot 2

Shot N+1 Caption

Shot 3 Caption

Shot 2 Frame Selection Caption Adaptive Cond. DiT

[Figure 303]

[Figure 304]

[Figure 305]

Frame Selection Adaptive Cond. DiT

Frame Selection Adaptive Cond. DiT

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

N steps

Noise Input

Noise Input Noise Input

A Average

(a)Training Phase

Shot 3 Shot N+1

Shot 2

C Concat Product

(c) Details of Frame Selection (d) Details of Adaptive Conditioner

A

| |
|---|
| |
| |
| |

[Figure 333]

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

[Figure 334]

[Figure 335]

VAEEnc.

Proj.

MatMul Average

[Figure 336]

| |
|---|

[Figure 337]

[Figure 338]

[Figure 339]

Conv.

Conv.

[Figure 340]

[Figure 341]

Novel-Base Mix

Adaptive Patchification

[Figure 342]

[Figure 343]

[Figure 344]

A

CrossAttention

| |
|---|

C Concat

Historical Memory

Context Tokens

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

Context Memory

DiT

C

Preceding Shots ,

VAEDec.

Projector

Standard Patchification

CrossAttention

Shot 3 Caption

Text Enc.

Learnable

Queries

Noisy Latents Noisy Tokens

- Figure 3 Overview of the proposed OneStory. Our model reframes multi-shot video generation (MSV) as a next-shot generation task. (a) During training, the model learns to generate the final shot conditioned on the preceding two; when only two shots are available, we inflate with a synthetic shot to enable unified three-shot training. (b) At inference, it maintains a memory bank of past shots and generates multi-shot videos autoregressively. The model is comprised of two key components: (c) a Frame Selection module that selects semantically-relevant frames from preceding shots to construct a global context, and (d) an Adaptive Conditioner that dynamically compresses the selected context and injects it directly into the generator for efficient conditioning. Together, OneStory realizes adaptive memory modeling, enabling global yet compact cross-shot context for coherent narrative generation.

latent dimension. The text encoder T encodes caption Ci into T tokens ti = T (Ci) ∈ RT×D

t of dimension Dt. Under this formulation, our model OneStory, initialized from a pretrained I2V model, achieves strong multi-shot generation after lightweight fine-tuning on our 60K dataset. As demonstrated in Figure 1, this unified formulation naturally supports both text- and image-conditioned generation: the first shot can be initiated from text or text+image conditions, while subsequent shots are generated autoregressively as new captions are introduced. The overall architecture is illustrated in Figure 3, while Section 4.2 details the Frame Selection module, Section 4.3 introduces the Adaptive Conditioner, and Section 4.4 describes our effective training strategy.

- 4.2 Frame Selection

A unique property of multi-shot videos is their unbounded spatio-temporal variance across shots: adjacent shots are not necessarily contiguous in time or space (Chasanis et al., 2008; Kara et al., 2025; Guo et al., 2025b). For example, Shot 1 may depict the protagonist, Shot 2 a secondary character, and Shot 3 the protagonist again. When generating Shot 3, the model should primarily reference Shot 1 to ensure subject consistency, while Shot 2 is less relevant. Motivated by such variable cross-shot relevance, we introduce a Frame Selection module that selects semantically relevant frames from prior shots, ensuring appropriate visual context for consistent generation.

Historical memory. When generating the i-th shot, we first encode all preceding shots {Sj}ij−=11 by the 3D VAE encoder E to obtain latent features. For simplicity, we refer to these latent frames directly as “frames” in the

following. These encoded frames are concatenated into a global memory:

K ft )

M = Fconcat z(1)j ,...,z(

j

i−1 j=1 ∈ RF×N

s×Dv, (2)

where z(jτ) is the τ-th frame feature of shot Sj, F = (i − 1)fK

#### is the total number of frames across preceding i − 1 shots, and Ns = fH

t

#### is the number of tokens per frame. Fconcat concatenates features along the

× Wf

s

s

|F4|F3|F2|F1|
|---|---|---|---|

|F6|F4|F3|F1|
|---|---|---|---|

Condition Tokens

Condition Tokens

F6 F5 F4 F3 F2 F1 F6 F5 F4 F3 F2 F1

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Frames to Present Finer Patchifier

| | |
|---|---|
| | |

| | |
|---|---|
| | |

- Figure 4 Patchification Comparison. Left: Prior fixed temporal schemes typically consider the most recent block of contiguous frames and assign patchifiers by temporal order (e.g., the finest patchifier for the latest frame). Right: Our adaptive scheme selects non-contiguous frames and allocates patchifiers based on content importance (i.e., finest patchifier for the most-important frame).

temporal axis, forming a unified visual memory of historical context.

Querying with caption and memory. To identify relevant historical frames, we introduce m learnable query tokens Q ∈ Rm×D, where D is the channel dimension of the model. These queries first attend to the current caption tokens to capture the semantic intent of the current shot:

Q′ = Fattn(Q, ϕT(ti), ϕT(ti)), (3) where ϕT : RD

→RD projects text features into the latent space of the model, and Fattn denotes the attention operation (Vaswani et al., 2017) (first argument: queries; remaining: keys/values). The updated queries Q′ then attend to the visual memory to extract corresponding visual cues:

t

M1 = ϕV (M), Q′′ = Fattn(Q′, M1, M1), (4) where ϕV : RD

→RD is a convolutional projector that reduces the number of spatial tokens Ns for efficiency.

v

Scoring and selection. With the semantically and visually enriched queries, we compute frame-wise relevance scores to the current shot via query–memory interactions:

s = ϕP(M1)Q′′⊤ ∈ RF×m, S = Fmean(s) ∈ RF, (5)

where ϕP projects each frame into a global embedding, and Fmean averages scores across queries. To assist the learning of S, we construct pseudo-labels indicating frame relevance to the current shot using DINOv2 (Oquab

- et al., 2023) and CLIP (Radford et al., 2021) (more details in Appendix). Next, the top-Ksel frames are selected from M based on S via the operation FTopK:

sel×D. (6)

M = FTopK(M, S, Ksel) ∈ RK

The resulting memory M forms a semantically-relevant historical context, which is then passed to the Adaptive Conditioner (Section 4.3) for effective conditioning.

- 4.3 Adaptive Conditioner

Although the context memory M contains rich semantic cues, directly using all its tokens as conditioning signals incurs substantial computational overhead. To address this, we design an Adaptive Conditioner which produces a compact set of context tokens with effective conditioning injection, balancing efficiency and informativeness.

Adaptive patchification. We define a set of patchifiers {Pℓ}Lℓ=1p , each with a distinct kernel size. Based on the relevance scores S from Equation (5), we divide the indices of M into Lp disjoint subsets {Iℓ}Lℓ=1p , assigning highly relevant frames to finer patchifiers with lower compression. As in Figure 4, unlike prior fixed, temporal-based assignments (Gu et al., 2025; Zhang and Agrawala, 2025), our scheme adaptively allocates patchifiers to semantically relevant, non-adjacent frames, enabling content-driven rather than temporal-driven conditioning. Each patchifier then transforms its assigned frames into context tokens:

Cℓ = Pℓ MI

ℓ

#### , C = Fconcat {Cℓ}Lℓ=1p , (7)

c×D, with Nℓ denoting the number of tokens produced by Pℓ and Nc = Lℓ=1p Nℓ the total number of context tokens. Condition injection. Let N ∈ RN

where Cℓ ∈ RN

ℓ×D and C ∈ RN

n×D denote the noise tokens of the current shot in the diffusion process (Ho et al., 2020). We concatenate the context tokens C with N along the token dimension to form the DiT (Peebles and Xie, 2023) input:

n+Nc)×D. (8) This simple yet effective injection scheme enables joint attention between noisy and context tokens, facilitating rich interactions. By adjusting the patchifiers {Pℓ}Lℓ=1p to balance compression and retention, the additional computation remains minimal. Overall, the Adaptive Conditioner provides efficient, relevance-aware conditioning that achieves compactness without sacrificing context expressiveness.

X = Fconcat([N, C]) ∈ R(N

- 4.4 Training Strategy

We train the model jointly and end-to-end by predicting the final shot in each sequence conditioned on its preceding shots. To ensure effective optimization and enhance narrative consistency, we introduce the following training strategies. More details are in Appendix.

Shot inflation. The dataset introduced in Section 3 contains videos with varying numbers of shots, primarily two-shot sequences with fewer three-shot ones, which destabilizes optimization when trained jointly. Therefore, we inflate two-shot sequences into three-shot ones by either (i) inserting a shot sampled from another video or (ii) augmenting the first shot (e.g., spatial or color transformations). This process yields mixed real and synthetic triplets, enabling uniform three-shot training.

Decoupled conditioning. Early in training, the Frame Selection module is randomly initialized and may select suboptimal frames, complicating optimization. We introduce a two-stage curriculum. During warm-up, we train on synthetic three-shot sequences and uniformly sample conditioning frames from the real shot, effectively decoupling conditioning from the selector’s outputs. Afterward, we switch to full selector-driven conditioning, where selected frames directly guide generation. This progressive coupling stabilizes convergence and enhances narrative coherence.

- 5 Experiments

- 5.1 Experimental Setup

Implementation Details. Our model builds upon the pretrained I2V model Wan2.1 (Wan et al., 2025). We optimize using AdamW with a learning rate of 0.0005 and weight decay of 0.01. The entire model is fine-tuned for one epoch on 128 NVIDIA A100 GPUs using our curated multi-shot dataset. All videos are centercropped to 480×832 while preserving aspect ratio. To comprehensively evaluate MSV, we construct dedicated benchmarks for both T2MSV and I2MSV settings, covering diverse human-centric narratives with complex cross-shot dynamics such as reappearance and composition. Further details are in Appendix.

Baselines. We compare OneStory with three strong MSV paradigms, using a vision-language model (Meta,

- 2025; Bai et al., 2025; Yuan et al., 2025) to convert shot-level captions into method-specific prompts:

- 1. Fixed-window attention extends attention to multiple shots within a fixed temporal window. We employ the public Mask2DiT (Qi et al., 2025) as a representative baseline.
- 2. Keyframe conditioning first generates a keyframe per shot, which is then expanded into a shot by an I2V model. We use StoryDiffusion (Zhou et al., 2024) for keyframes and Wan 2.1 (Wan et al.,

2025) / LTX-Video (HaCohen et al., 2024) for I2V synthesis.

- 3. Edit-and-extend treats MSV as next-shot generation, transferring the last frame of the previous shot via an I2I model before I2V synthesis. We use FLUX (Black Forest Labs, 2024) as the I2I model and Wan2.1 / LTX-Video for I2V generation.

Inter-shot Coherence Semantic

Intra-shot Coherence Aesthetic

Dynamic Character Degree↑

Method

Align.↑

Quality↑

↑ Env.↑ Avg.↑ Subject↑ BG.↑ Avg.↑ Text-to-multi-shot (T2MSV)

Flux + LTX-Video 0.5316 0.5456 0.5386 0.1837 0.8841 0.8957 0.8899 0.5070 0.3746 Flux + Wan2.1 0.5454 0.5598 0.5526 0.1915 0.9225 0.9353 0.9289 0.5572 0.4492 Mask2DiT 0.5472 0.5419 0.5446 0.2253 0.9024 0.9150 0.9087 0.5235 0.4247 StoryDiff. + LTX-Video 0.5468 0.5397 0.5433 0.2165 0.9036 0.9125 0.9081 0.5418 0.3694 StoryDiff. + Wan2.1 0.5633 0.5681 0.5657 0.2217 0.9286 0.9357 0.9322 0.5703 0.4231 OneStory (Ours) 0.5874 0.5752 0.5813 0.2389 0.9364 0.9410 0.9387 0.5731 0.4698 Image-to-multi-shot (I2MSV)

Mask2DiT 0.5452 0.5446 0.5449 0.2270 0.9056 0.9073 0.9065 0.5218 0.4256 Flux + LTX-Video 0.5336 0.5469 0.5403 0.1846 0.8803 0.8930 0.8867 0.5114 0.3790 Flux + Wan2.1 0.5419 0.5547 0.5483 0.1897 0.9186 0.9310 0.9248 0.5531 0.4544 OneStory (Ours) 0.5851 0.5716 0.5784 0.2354 0.9327 0.9389 0.9358 0.5704 0.4673

- Table 1 Quantitative results under text-to-multi-shot (T2MSV) and image-to-multi-shot (I2MSV) settings. The best and runner-up results are in bold and underlined, respectively. In both text- and image-conditioned settings, our model consistently outperforms all baselines on shot-level quality and narrative consistency, demonstrating superior multi-shot generation capabilities. “Env.” denotes environment consistency, “BG.” denotes background consistency, and “Avg.” indicates the average of the corresponding metrics.

[Shot1] Medium shot of a man speaking

[Shot 2] The man leans back in his chair

[Shot 3] Medium shot of a new woman speaking

[Shot 4] Close-up side view of the man

[Shot 5] Returns to front medium shot of the man

[Shot 6] Medium front shot of both seated together

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

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

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

Wan2.1 Storydiff.+

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

Wan2.1 Ours

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

Flux+

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

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

[Figure 446]

2MaskDIT

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

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

- Figure 5 Qualitative results. For a fair comparison, the given multi-shot generations share the same first shot (generated by Wan2.2) as the initial condition, except for StoryDiff.+Wan2.1, which does not rely on visual conditioning. The baseline methods fail to maintain narrative consistency across shots, struggling with prompt adherence, reappearance, and compositional scenes, whereas OneStory (ours) faithfully follows shot-level captions and produces coherent shots. A representative segment of each prompt is given with the corresponding shot.

- 5.2 Main Results

Quantitative Evaluation. We evaluate from two perspectives: shot-level quality and narrative consistency.

Shot-level quality follows single-shot metrics (Huang et al., 2024), including subject consistency, background consistency, aesthetic quality, and dynamic degree. For narrative consistency, we design metrics following prior studies (Kara et al., 2025):

AC FS C-Cons ↑ E-Cons ↑ S-Align ↑

SI DC C-Cons ↑ E-Cons ↑ S-Align ↑

Ctx len C-Cons ↑ E-Cons ↑

0.5153 0.5112 0.1814 ✓ 0.5465 0.5597 0.2172 ✓ 0.5526 0.5710 0.2238 ✓ ✓ 0.5874 0.5752 0.2389

0.5514 0.5615 0.2207 ✓ 0.5649 0.5790 0.2263 ✓ ✓ 0.5874 0.5752 0.2389

- 1-frame 0.5874 0.5752
- 2-frame 0.5890 0.5795

- 3-frame 0.5926 0.5863

(b) Impact of training strategies.

(c) Impact of #context tokens.

(a) Impact of model design.

- Table 2 Ablation study. (a) Model design: both Adaptive Conditioner (AC) and Frame Selection (FS) are crucial for multi-shot generation quality. (b) Training strategies: Shot Inflation (SI) and Decoupled Conditioning (DC) improve narrative learning. (c) Context token length (Ctx len): with one latent frame as the unit of context token budget, a single-frame equivalent performs strongly, and more tokens further improve. C-Cons and E-Cons denote character/environment consistency, with S-Align as semantic alignment. Best results are in bold.

[Figure 489]

[Figure 490]

[Figure 491]

- a. Shot 1 b. Shot 1 c. Shot 1
- a. Shot 2 b. Shot 2 c. Shot 2

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

| |
|---|

| |
|---|

Shot 6 Shot 6 Shot 6

[Figure 496]

[Figure 497]

| |
|---|

- (a)
- (b)

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

- Shot 1 Shot 1 Shot 1
- Shot 2

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

- d. Shot 1 e. Shot 1 f. Shot 1
- d. Shot 2 e. Shot 2 f. Shot 2

Shot 2 Shot 2

[Figure 512]

[Figure 513]

| |
|---|

| |
|---|

| |
|---|

Ours Uniform Most-recent

[Figure 514]

[Figure 515]

[Figure 516]

- Figure 6 Qualitative comparison of frame selection strategies. (a) The sixth-shot generation using the five preceding shots as context (shown in the last row of Figure 5). (b) Fine-grained cases where the first shot involves dynamic motion. In each, the first gray row shows a few sampled frames (first, middle, and last) depicting the motion range in the first shot. The subsequent row shows the generated next shot from each strategy. Both baselines fail to maintain visual coherence, whereas our method identifies semantically relevant frames and produces consistent shots.

Figure 7 Advanced narrative modeling in OneStory. (ab) Appearance changes: the character’s identity remains consistent under appearance variations. (c–d) Zoom-in effects: the model accurately localizes intended regions and preserves fine details when zooming in. (e–f) Human–object interactions: the model correctly continues event progression, maintaining coherent relationships between humans and surrounding objects across shots.

- 1. Character Consistency computes DINOv2 (Oquab et al., 2023) similarity between YOLO (Ultralytics,

2021) segmented persons across shots annotated as containing the same character.

- 2. Environment Consistency measures DINOv2 similarity between segmented background regions across shots annotated with matched environments.
- 3. Semantic Alignment quantifies the alignment between each generated shot and its caption using ViCLIP (Wang et al., 2024b).

We group subject/background consistency as intra-shot coherence and character/environment consistency as inter-shot coherence. As shown in Table 1, for T2MSV, our model significantly surpasses all baselines across inter-shot coherence and semantic alignment metrics, demonstrating superior narrative consistency. It also achieves better shot-level performance with improved motion control and intra-shot fidelity. In I2MSV, we omit keyframe baselines since they cannot directly accept image inputs. Our model again attains state-of-the-art results across both shot-level and narrative metrics, further confirming its effectiveness.

Qualitative Results. Figure 5 shows qualitative comparisons under complex narratives. All baselines struggle to follow shot-level prompts and maintain cross-shot coherence. For instance, in Shot 4, both StoryDiff. (Zhou

- et al., 2024) + Wan2.1 (Wan et al., 2025) and Flux (Black Forest Labs, 2024) + Wan2.1 fail to adjust the viewpoint accordingly, while Mask2DiT (Qi et al., 2025) and Flux+Wan2.1 generate the wrong character in Shot 5. When Shot 3 introduces a new character (the woman), the baselines further lose coherence when the protagonist (the man) reappears in Shots 4–6. Moreover, in the final compositional shot requiring both characters to appear together, they all collapse in maintaining character identity, revealing limited memory adaptation ability. In contrast, OneStory maintains subject and environment consistency across reappearances and compositions, while faithfully adhering to evolving narratives, highlighting the superiority of our adaptive memory for coherent narrative generation.

- 5.3 Ablation Study

Impact of model design. Table 2a analyzes each design under the same context-token budget, equal to the number of tokens in one latent frame. The baseline relies only on the last frame, showing the weakest performance due to missing historical context. Adding the Adaptive Conditioner extends contextual range, while the Frame Selection module enhances adaptation by conditioning on the most relevant frame. Combining both yields the best results, confirming their complementary roles in cross-shot context modeling.

Frame selection. Figure 6 compares our automatic frame selection with uniform and most-recent sampling strategies. Each variant is trained with its own sampling scheme under the same context budget. In Figure 6(a), the first five shots from the last row of Figure 5 serve as history for predicting the sixth shot. Figure 6(b) shows a more challenging fine-grained case where the first shot contains large camera motion (see the 2nd gray row), requiring precise frame selection to preserve continuity. Across both cases, our method maintains contextual consistency, while both baselines fail, highlighting the effectiveness of our frame selection module.

Impact of training strategies. Table 2b evaluates proposed training strategies. Training on mixed two- and three-shot videos (baseline) hinders narrative learning due to imbalanced context sequences, whereas shot inflation enables unified three-shot training with richer temporal context and improves generation ability. Adding the decoupled conditioning curriculum further stabilizes early optimization and strengthens narrative coherence.

Context tokens. We vary the number of context tokens in the Adaptive Conditioner by adjusting its patchifiers. We define one latent frame as the unit of context token amount to allow direct comparison with the number of original noise tokens (21-frame). As shown in Table 2c, even a single latent-frame equivalent of context tokens yields strong inter-shot coherence, and larger budgets further enhance performance. This confirms the efficiency of our compact adaptive memory in modeling cross-shot dynamics. By default, we use one latent-frame equivalent of context tokens.

- 5.4 Advanced Narrative Modeling

Real-world narratives exhibit complex cross-shot dependencies. OneStory captures these dynamics through adaptive memory, showing advanced generation ability with a global narrative understanding beyond surfacelevel visual continuity. We analyze its advanced narrative modeling from three perspectives.

Appearance changes. Beyond typical cross-shot variations (e.g., viewpoint), maintaining character consistency under appearance changes is especially challenging. As shown in Figure 7(a–b), OneStory preserves consistent facial features while adapting clothing and environments as prompted, showcasing robust narrative coherence under complex visual changes. We refer readers to Project Page for better visualizations.

Zoom-in effects. Transitions from wide shots to non-salient close-ups demand spatial reasoning to locate fine-grained targets while preserving fidelity. In Figure 7(c–d), OneStory accurately identifies local regions and maintains detail for both static (i.e., mirror) and dynamic (i.e., hand) targets. In Figure 1 (3rd example), small objects on the table, barely visible in Shot 2, are faithfully rendered in the zoomed-in Shot 3, yielding smooth narrative progression.

Human-object interactions. Shot transitions often hinge on human–object dynamics and implicit intent. In Figure 7(e–f), scenes of a man engaging with a car or unfolding a tent lead to coherent next shots that depict the expected subsequent state. These examples highlight OneStory’s ability to interpret human–object relations, enabling realistic and semantically grounded story development.

- 6 Conclusion

We presented OneStory, a novel framework for coherent multi-shot video generation via adaptive memory modeling. By reformulating MSV as a next-shot generation problem, OneStory leverages the strong visual conditioning capacity of pretrained I2V models, enabling scalable and autoregressive story synthesis. The proposed Frame Selection module identifies semantically relevant frames across prior shots, while the Adaptive Conditioner performs importance-guided patchification with direct condition injection, jointly enabling global yet compact cross-shot context. OneStory effectively handles complex narratives and achieves superior narrative coherence, offering valuable insights into adaptive memory modeling for immersive, story-driven video generation.

Acknowledgements

We would like to thank Junlin Han (University of Oxford), Mingqiao Ye (EPFL), and Feng Qiao (WashU) for their constructive feedback on this project. Zhaochong An and Serge Belongie are supported by funding from the Pioneer Centre for AI, DNRF grant number P1.

References

Yuval Atzmon, Rinon Gal, Yoad Tewel, Yoni Kasten, and Gal Chechik. Multi-shot character consistency for text-to-video generation. arXiv preprint, 2024.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint, 2025.

Hritik Bansal, Yonatan Bitton, Michal Yarom, Idan Szpektor, Aditya Grover, and Kai-Wei Chang. Talc: Time-aligned captions for multi-scene text-to-video generation. arXiv preprint, 2024.

Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur,

Guanghui Liu, Amit Raj, et al. Lumiere: A space-time diffusion model for video generation. arXiv preprint, 2024. Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024. Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman,

Eric Luhman, et al. Video generation models as world simulators. 2024. Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, et al. Mixture of contexts for long video generation. arXiv preprint, 2025. Vasileios T Chasanis, Aristidis C Likas, and Nikolaos P Galatsanos. Scene detection in videos using shot clustering and sequence alignment. IEEE transactions on multimedia, 11(1):89–100, 2008. Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, pages 7310–7320, 2024a. Jiaben Chen, Zixin Wang, Ailing Zeng, Yang Fu, Xueyang Yu, Siyuan Cen, Julian Tanke, Yihang Chen, Koichi Saito,

Yuki Mitsufuji, et al. Talkcuts: A large-scale dataset for multi-shot human speech video generation. NeurIPS, 2025. Xi Chen, Zhiheng Liu, Mengting Chen, Yutong Feng, Yu Liu, Yujun Shen, and Hengshuang Zhao. Livephoto: Real

image animation with text-guided motion control. In ECCV, pages 475–491. Springer, 2024b.

Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In ICLR, 2023.

Ruihang Chu, Yefei He, Zhekai Chen, Shiwei Zhang, Xiaogang Xu, Bin Xia, Dingdong Wang, Hongwei Yi, Xihui Liu, Hengshuang Zhao, et al. Wan-move: Motion-controllable video generation via latent trajectory guidance. In NeurIPS, 2025.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint, 2025.

Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint, 2025.

Yuwei Guo, Ceyuan Yang, Anyi Rao, Chenlin Meng, Omer Bar-Tal, Shuangrui Ding, Maneesh Agrawala, Dahua Lin, and Bo Dai. Keyframe-guided creative video inpainting. In CVPR, pages 13009–13020, 2025a.

Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. ICCV, 2025b.

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint, 2024.

Jingwen He, Hongbo Liu, Jiajun Li, Ziqi Huang, Yu Qiao, Wanli Ouyang, and Ziwei Liu. Cut2next: Generating next

shot via in-context tuning. arXiv preprint, 2025. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. Li Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In CVPR,

pages 8153–8163, 2024. Panwen Hu, Jin Jiang, Jianqi Chen, Mingfei Han, Shengcai Liao, Xiaojun Chang, and Xiaodan Liang. Storyagent: Customized storytelling video generation via multi-agent collaboration. arXiv preprint, 2024.

Yuzhou Huang, Ziyang Yuan, Quande Liu, Qiulin Wang, Xintao Wang, Ruimao Zhang, Pengfei Wan, Di Zhang, and Kun Gai. Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning. arXiv preprint, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, pages 21807–21818, 2024.

Weinan Jia, Yuning Lu, Mengqi Huang, Hualiang Wang, Binyuan Huang, Nan Chen, Mu Liu, Jidong Jiang, and Zhendong Mao. Moga: Mixture-of-groups attention for end-to-end long video generation. arXiv preprint, 2025.

Jiaxiu Jiang, Wenbo Li, Jingjing Ren, Yuping Qiu, Yong Guo, Xiaogang Xu, Han Wu, and Wangmeng Zuo. Lovic: Efficient long video generation with context compression. arXiv preprint, 2025.

Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. Videobooth: Diffusion-based video generation with image prompts. In CVPR, pages 6689–6700, 2024.

Ozgur Kara, Krishna Kumar Singh, Feng Liu, Duygu Ceylan, James M Rehg, and Tobias Hinz. Shotadapter: Text-to-multi-shot video generation with diffusion models. In CVPR, pages 28405–28415, 2025.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang,

et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint, 2024. Kuaishou. Kling video model. https://kling.kuaishou.com/en, 2024. Teng Li, Guangcong Zheng, Rui Jiang, Shuigen Zhan, Tao Wu, Yehao Lu, Yining Lin, Chuanyun Deng, Yepan Xiong,

Min Chen, et al. Realcam-i2v: Real-world image-to-video generation with interactive complex camera control. In ICCV, pages 28785–28796, 2025.

Kang Liao, Size Wu, Zhonghua Wu, Linyi Jin, Chao Wang, Yikai Wang, Fei Wang, Wei Li, and Chen Change Loy. Thinking with camera: A unified multimodal model for camera-centric understanding and generation. arXiv preprint, 2025.

Zhiqiu Lin, Siyuan Cen, Daniel Jiang, Jay Karhade, Hewei Wang, Chancharik Mitra, Tiffany Ling, Yuhan Huang, Sifan Liu, Mingyu Chen, et al. Towards understanding camera motions in any video. arXiv preprint, 2025a.

Zongyu Lin, Wei Liu, Chen Chen, Jiasen Lu, Wenze Hu, Tsu-Jui Fu, Jesse Allardice, Zhengfeng Lai, Liangchen Song, Bowen Zhang, et al. Stiv: Scalable text and image conditioned video generation. In ICCV, pages 16249–16259, 2025b.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint, 2022.

Hongbo Liu, Jingwen He, Yi Jin, Dian Zheng, Yuhao Dong, Fan Zhang, Ziqi Huang, Yinan He, Yangguang Li, Weichao Chen, et al. Shotbench: Expert-level cinematic understanding in vision-language models. arXiv preprint, 2025.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint, 2022.

Fuchen Long, Zhaofan Qiu, Ting Yao, and Tao Mei. Videostudio: Generating consistent-content and multi-scene videos. In ECCV, pages 468–485. Springer, 2024.

Chen-Yi Lu, Md Mehrab Tanjim, Ishita Dasgupta, Somdeb Sarkhel, Gang Wu, Saayan Mitra, and Somali Chaterji. Skald: Learning-based shot assembly for coherent multi-shot video creation. In ICCV, pages 17859–17868, 2025.

Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. arXiv preprint, 2023.

Yihao Meng, Hao Ouyang, Yue Yu, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Hanlin Wang, Yixuan Li, Cheng Chen, Yanhong Zeng, et al. Holocine: Holistic generation of cinematic multi-shot long video narratives. arXiv preprint, 2025.

Team Meta. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. https://ai.meta. com/blog/llama-4-multimodal-intelligence/, April 2025.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195–4205, 2023. Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi,

Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint, 2024.

Tianhao Qi, Jianlong Yuan, Wanquan Feng, Shancheng Fang, Jiawei Liu, SiYu Zhou, Qian He, Hongtao Xie, and Yongdong Zhang. Maskˆ 2dit: Dual mask-based diffusion transformer for multi-scene long video generation. In CVPR, pages 18837–18846, 2025.

Lu Qiu, Yizhuo Li, Yuying Ge, Yixiao Ge, Ying Shan, and Xihui Liu. Animeshooter: A multi-shot animation dataset for reference-guided video generation. arXiv preprint, 2025.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684–10695, 2022.

Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, et al. Seaweed-7b: Cost-effective training of video generation foundation model. arXiv preprint, 2025.

Shuwei Shi, Biao Gong, Xi Chen, Dandan Zheng, Shuai Tan, Zizheng Yang, Yuyuan Li, Jingwen He, Kecheng Zheng, Jingdong Chen, et al. Motionstone: Decoupled motion intensity modulation with diffusion transformer for image-to-video generation. In CVPR, pages 22864–22874, 2025.

Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion. ICML, 2025a.

Wenhui Song, Hanhui Li, Jiehui Huang, Panwen Hu, Yuhao Cheng, Long Chen, Yiqiang Yan, and Xiaodan Liang. Lavieid: Local autoregressive diffusion transformers for identity-preserving video creation. arXiv preprint, 2025b.

Tomás Soucek and Jakub Lokoc. Transnet v2: An effective deep network architecture for fast shot transition detection.

In ACM MM, pages 11218–11221, 2024. Genmo Team. Mochi 1. https://github.com/genmoai/models, 2024. Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil

Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint, 2025.

Ultralytics. YOLOv5: A state-of-the-art real-time object detection system. https://docs.ultralytics.com, 2021. Accessed: 2024-11-04.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 30, 2017.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint, 2025.

Boyang Wang, Xuweiyi Chen, Matheus Gadelha, and Zezhou Cheng. Frame in-n-out: Unbounded controllable image-to-video generation. NeurIPS, 2025a.

Jiahao Wang, Hualian Sheng, Sijia Cai, Weizhan Zhang, Caixia Yan, Yachuang Feng, Bing Deng, and Jieping Ye. Echoshot: Multi-shot portrait video generation. NeurIPS, 2025b.

Qinghe Wang, Yawen Luo, Xiaoyu Shi, Xu Jia, Huchuan Lu, Tianfan Xue, Xintao Wang, Pengfei Wan, Di Zhang, and Kun Gai. Cinemaster: A 3D-aware and controllable framework for cinematic text-to-video generation. In SIGGRAPH, pages 1–10, 2025c.

Xinran Wang, Songyu Xu, Xiangxuan Shan, Yuxuan Zhang, Muxi Diao, Xueyan Duan, Yanhua Huang, Kongming Liang, and Zhanyu Ma. Cinetechbench: A benchmark for cinematographic technique understanding and generation. arXiv preprint, 2025d.

Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. International Journal of Computer Vision, pages 1–20, 2024a.

Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In ICLR, 2024b.

Zun Wang, Jialu Li, Han Lin, Jaehong Yoon, and Mohit Bansal. Dreamrunner: Fine-grained storytelling video generation with retrieval-augmented motion adaptation. arXiv preprint, 2024c.

Cong Wei, Bo Sun, Haoyu Ma, Ji Hou, Felix Juefei-Xu, Zecheng He, Xiaoliang Dai, Luxin Zhang, Kunpeng Li, Tingbo Hou, et al. Mocha: Towards movie-grade talking character synthesis. arXiv preprint, 2025.

Xiaoxue Wu, Bingjie Gao, Yu Qiao, Yaohui Wang, and Xinyuan Chen. Cinetrans: Learning to generate videos with cinematic transitions via masked diffusion models. arXiv preprint, 2025a.

Ziyi Wu, Aliaksandr Siarohin, Willi Menapace, Ivan Skorokhodov, Yuwei Fang, Varnith Chordia, Igor Gilitschenski, and Sergey Tulyakov. Mind the time: Temporally-controlled multi-event video generation. In CVPR, pages 23989–24000, 2025b.

Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan Yuille, and Lu Jiang. Captain cinema: Towards short movie generation. arXiv preprint, 2025.

Zhifei Xie, Daniel Tang, Dingwei Tan, Jacques Klein, Tegawend F Bissyand, and Saad Ezzini. Dreamfactory: Pioneering multi-scene long video generation with a multi-agent framework. arXiv preprint, 2024.

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In ECCV, pages 399–417. Springer, 2024.

Jinbo Xing, Long Mai, Cusuh Ham, Jiahui Huang, Aniruddha Mahapatra, Chi-Wing Fu, Tien-Tsin Wong, and Feng Liu. Motioncanvas: Cinematic shot design with controllable image-to-video generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11, 2025.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint, 2024.

Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint, 2025.

Liping Yuan, Jiawei Wang, Haomiao Sun, Yuchen Zhang, and Yuan Lin. Tarsier2: Advancing large vision-language models from detailed video description to comprehensive video understanding. arXiv preprint, 2025.

Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, and Hang Li. Make pixels dance: High-dynamic video generation. In CVPR, pages 8850–8860, 2024.

David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. IJCV, pages 1–15, 2024.

Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint, 2025.

Yuang Zhang, Junqi Cheng, Haoyu Zhao, Jiaxi Gu, Fangyuan Zou, Zenghui Lu, and Peng Shu. Shouldershot: Generating over-the-shoulder dialogue videos. arXiv preprint, 2025.

Canyu Zhao, Mingyu Liu, Wen Wang, Weihua Chen, Fan Wang, Hao Chen, Bo Zhang, and Chunhua Shen. Moviedreamer: Hierarchical generation for coherent long visual sequence. arXiv preprint, 2024.

Mingzhe Zheng, Yongqi Xu, Haojian Huang, Xuran Ma, Yexin Liu, Wenjie Shu, Yatian Pang, Feilong Tang, Qifeng Chen, Harry Yang, et al. Videogen-of-thought: A collaborative framework for multi-shot video generation. arXiv preprint, 2024.

Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. NeurIPS 2024, 2024.

## Appendix

- A Additional Training Details

This section provides expanded details on the training formulation used in our model, including the unified threeshot training setup and the construction of frame-level pseudo-labels. These details complement Section 4.2 and Section 4.4 of the main paper.

- A.1 Unified Three-Shot Training

As discussed in Section 3 of the main paper, the dataset contains videos with varying numbers of shots, with two-shot sequences being the most common and three-shot sequences relatively fewer. Training directly on sequences of non-uniform length leads to unstable optimization. To mitigate this, we unify all training samples into a three-shot format by synthesizing an additional shot for two-shot videos.

Synthetic shot construction. Given a two-shot sequence (Sfirst,Slast), we create a synthetic shot Ssyn using one of:

- (i) Cross-video insertion: inserting a shot randomly sampled from another video.
- (ii) Augmented-first-shot variant: applying spatial or color transformations to Sfirst. This results in synthetic triplets that, for each sample, take one of the two forms:

Sfirst,Ssyn,Slast or Ssyn,Sfirst,Slast , (9)

while the real triplets are represented in the structure (Sfirst,Ssecond,Slast). In all cases, Slast serves as the prediction target.

Training objective. The model is trained to generate the final shot Slast conditioned on the first two shots and its caption Clast:

Lshot = E Ldiff G(Sfirst, Ssyn/second, Clast), Slast , (10)

where Ldiff denotes a rectified-flow diffusion loss (Lipman et al., 2022; Liu et al., 2022; Esser et al., 2024). This unified formulation standardizes all training samples to a consistent three-shot structure and enables unified three-shot training, improving optimization stability.

- A.2 Frame Relevance Pseudo-Labels

To assist the learning of the frame relevance scores S, we construct frame-level pseudo-labels y = {yr}Fr=1 that approximate the relevance of each historical frame in M to the target shot. The pseudo-labels incorporate both real and synthetic frames introduced in Section A.1.

Real historical frames. For frames originating from real historical shots, we compute cosine similarity between each historical frame and the target shot using DINOv2 (Oquab et al., 2023) and CLIP (Radford et al., 2021) embeddings, producing a scalar relevance score. These pseudo-labels help the Frame Selection module prioritize visually and semantically aligned frames while down-weighting irrelevant ones.

Synthetic frames. Frames from synthetic shots introduced in Section A.1 are assigned coarse relevance labels: yr = −1 for randomly inserted shots to indicate clear irrelevance, and yr = 0 for augmented-first-shot variants to reflect partial relevance. These labels explicitly guide the selector to down-weight non-informative or misleading frames.

Supervision loss. The predicted relevance scores S are supervised using a regression loss:

F

1 F

(sr − yr)2, (11)

Lsel =

r=1

where sr = S[r] denotes the predicted relevance score for the rth historical frame. The full training objective is given by:

Ltrain = Lshot + λLsel, (12)

with λ controlling the weight of the selector supervision loss. This joint optimization encourages the model to identify informative context frames while maintaining high-fidelity generation.

- B Additional Details on Evaluation Benchmark

We construct a human-centric benchmark for both T2MSV and I2MSV to evaluate multi-shot video generation under realistic narrative conditions. As introduced in Section 3 of the main paper, each shot is paired with a referential caption following a progressive narrative flow, reflecting real-world storytelling. To comprehensively evaluate MSV performance, the benchmark spans three canonical multi-shot storytelling patterns:

- 1. Main-subject consistency. Multiple shots focus on the same character(s), who may appear in different environments or perform different actions. This pattern evaluates the model’s ability to preserve identity under various cross-shot changes.
- 2. Insert-and-recall with an intervening shot. A shot introducing a new scene, such as an environment-only view or a new character, is inserted mid-sequence, after which the narrative returns to the primary subject(s) and later revisits the intervening shot. This pattern stresses the model’s ability to maintain long-range memory and remain robust to temporal distractors.
- 3. Composable generation. Characters introduced separately in earlier shots are composed together in later shots. This tests whether the model can correctly integrate multiple narrative threads into a coherent shared scene.

In total, we curate 64 six-shot test cases for T2MSV and 64 six-shot test cases for I2MSV, covering a diverse range of subjects, environments, and complex cross-shot relationships, thereby ensuring comprehensive MSV performance evaluation. More examples are provided in our Project Page.

- C Additional Qualitative Results

Generating coherent multi-shot videos that faithfully follow narrative captions is essential for real-world storytelling. Here, we analyze our model from three perspectives, using examples from the main paper to illustrate its ability to maintain continuity across shots. Additional video qualitative results are available our Project Page.

Identity consistency. Our model preserves character identity across long-range shots under diverse variations. In the 1st example of Figure 1 in the main paper, the same subject remains consistent across changes in viewpoint (Shots 4, 5) and actions (Shots 1, 3, 8). This illustrates the effectiveness of our adaptive memory in maintaining stable long-range identity cues.

Background details. Beyond character fidelity, our model maintains consistent background details across shots, enabling spatially coherent story progression. In the 2nd example of Figure 1 in the main paper, fine-grained elements such as plants and fences remain aligned from Shot 1 to Shot 7 despite large cross-shot dynamics. Similarly, in the 3rd example, the red flowers reappear consistently across Shots 1, 4, 5, 6, 7, and 9, demonstrating the model’s ability to preserve scene layout and spatial structure.

Reappearance and composition. Realistic narratives often involve disappear–reappear patterns and the merging of multiple narrative threads through composable generation. Our model effectively recalls characters or environments that reemerge after several intervening shots, e.g., Shots 4 and 9, or Shots 2 and 6 in the 2nd example of Figure 1 in the main paper. Furthermore, in Shot 7 of the same example, the woman from Shot 1 and the man from Shot 4 appear together, demonstrating the model’s capacity to unify distinct visual narratives into a coherent multi-subject scene.

