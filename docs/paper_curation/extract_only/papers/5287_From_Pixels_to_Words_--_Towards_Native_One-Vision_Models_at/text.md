# arXiv:2605.28820v1[cs.CV]27May2026

###### From Pixels to Words – Towards Native One-Vision Models at Scale

###### Haiwen Diao1,2*, Jiahao Wang2, Penghao Wu1,2, Yuhao Dong1 Yuwei Niu2, Yue Zhu2, Zhongang Cai2, Weichen Fan1,2, Linjun Dai2 Silei Wu2, Xuanyu Zheng2, Mingxuan Li2, Yuanhan Zhang1, Bo Li1, Hanming Deng2 Huchuan Lu3, Quan Wang2, Lei Yang2, Lewei Lu2, Dahua Lin2, Ziwei Liu1† 1S-Lab, NTU 2SenseTime Research 3DLUT

Website: https://github.com/EvolvingLMMs-Lab/NEO

[Figure 1]

###### Abstract

Current vision–language models (VLMs) typically stitch together separate image encoders and language decoders via multi-stage alignment, a modular framework that inevitably fragments pixel-level signals across frames and scatters early pixel–word interactions. In parallel, native VLMs, despite impressive performance on single images, remain largely unexplored in multi-image, video understanding, and spatial intelligence. Hence, we introduce NEO-ov, a native foundation model that learns cross-frame and pixel-word correspondence end-to-end, without any external encoders, auxiliary adapters, or post-hoc fusion. By eliminating module boundaries entirely, NEO-ov enables fine-grained and unified spatiotemporal modeling to emerge natively inside the model. Notably, NEO-ov largely narrows the gap to modular counterparts while excelling at finegrained visual perception, validating that native “one-vision” architectures are not only feasible but competitive at scale. Beyond empirical performance, we unveil systematic architectural analyses and detailed training recipes to facilitate subsequent native multimodal modeling.

###### 1 Introduction

Recently, vision–language models (VLMs) have evolved from basic image perception towards advanced understanding of multi-image analysis, video understanding, and spatial intelligence. Existing models typically adopt an encoder–decoder architecture, where pretrained image (Radford et al., 2021; Zhai et al., 2023) or video (Li et al., 2025d; Zhang et al., 2025b) encoders produce visual representations that are subsequently processed by a projector (Liu et al., 2024a; Meng et al., 2024; Dai et al., 2023; Liao et al., 2025) and a large language model (LLM) (Touvron et al., 2023; Yang et al., 2025a) for visual understanding and reasoning.

*Work was done during Haiwen’s remote collaboration with SenseTime Research. †Corresponding author.

Despite strong performance, this modular design imposes inherent constraints on 1) Flexibility: vision encoders are expected to process heterogeneous inputs, from single images to image sets or videos. Yet existing designs force a false dichotomy: image encoders favor static, framelevel representations and lack spatiotemporal reasoning, while video encoders overemphasize temporal dynamics and generalize poorly to singleimage or interleaved inputs. Besides, both struggle in early pixel–word interaction and unified visual understanding scenarios. 2) Efficiency: decoupling vision and language modules fragments training and incurs substantial post-alignment overhead. Furthermore, extending visual encoders to long-duration or high-resolution inputs remains prohibitively expensive for streaming and proactive video understanding, as KV caching is not applicable. 3) Scalability: modularity entangles scaling, optimization, and deployment by requiring delicate capacity balancing between VEs and LLMs. These frictions fundamentally preclude structural simplicity and deep vision–language integration, motivating a unified, monolithic backbone.

To address them, native VLMs have recently emerged as a compelling alternative. Early exemplars, e.g., Fuyu (Bavishi et al., 2023) and EVE (Diao et al., 2024) demonstrate that visual and textual inputs can be jointly modeled within one single and monolithic framework without explicit vision encoders. Building on this paradigm, subsequent efforts learn visual representations from scratch while mitigating vision–linguistic interference through visual feature distillation (Diao

- et al., 2024; Li et al., 2025e; Wang et al., 2025b), modality-agnostic embeddings (Diao et al., 2025a; Tao et al., 2025; Yan et al., 2025) and modalityspecific decomposition (Diao et al., 2025b; Luo et al., 2024, 2025). Notably, recent studies (Yi
- et al., 2025; Li et al., 2025c) extend native VLMs to video domains, enabling end-to-end modeling of

fine-grained video–language interactions and temporal dependencies. However, these approaches remain constrained by distillation from static visual encoders, inheriting strong inductive biases rooted in pretrained image semantics. More importantly, unifying single-image, multiple-image, video understanding, and spatial intelligence simultaneously remains an open frontier for native VLMs toward truly unified one-vision foundation models across diverse multimodal applications.

Hence, we introduce NEO-ov, a native visionlanguage foundation model that eliminates pretrained encoders and unifies spatial and temporal modeling within a single monolithic backbone. Built on multiple native primitives, NEO-ov jointly learns visual perception, temporal dynamics, and cross-modal alignment directly from raw inputs through end-to-end training. Despite being fully encoder-free, NEO-ov surpasses existing native VLMs and approaches encoder-based competitors of the same LLMs across diverse benchmarks. Notably, it exhibits strong spatial intelligence across both low-level geometric perception and high-level spatiotemporal reasoning, enabling robust understanding of structure, motion, and long-range visual dependencies in a unified representation space. Together, these results suggest that multimodal intelligence may emerge not only from specialized components, but from architectures that are native, unified, and intrinsically multimodal.

###### 2 Related Work

###### 2.1 Modular Vision-Language Models

Existing vision-language models (VLMs) largely follow a modular design that connects external visual encoders to large language models (LLMs) through lightweight adapters (Alayrac et al., 2022; Dai et al., 2023). Notably, LLaVA (Liu et al.,

- 2023a; Li et al., 2024a) standardizes this paradigm via the simple Encoder-MLP-LLM pipeline and visual instruction tuning, which is subsequently adopted by models such as InternVL series (Chen

- et al., 2024b; Zhu et al., 2025; Wang et al., 2025e), Qwen-VL series (Wang et al., 2024a; Bai
- et al., 2025b,a), and etc. They further extend this paradigm to unified visual understanding across single-image, multi-image, and video tasks.

Despite empirical success, they remain fundamentally constrained by the encode-then-project paradigm, where visual signals are compressed before reasoning begins. Pretrained vision encoders

such as CLIP (Radford et al., 2021) or SigLIP (Zhai

- et al., 2023; Tschannen et al., 2025) are optimized primarily for image–text alignment, emphasizing high-level semantics while discarding texture, local geometry, and fine spatial structure. Consequently, language models reason over semantically filtered representations rather than native visual signals, limiting fine-grained perception and precise geometric reasoning. This limitation becomes particularly pronounced in spatial intelligence settings, where cross-view and cross-frame interactions are mediated through compressed semantic features instead of native spatial correspondences, hindering the modeling of positional relations, local motion, and pixel-level consistency across space and time.

2.2 Native Vision-Language Models

Native multimodal models move beyond modular pipelines by learning directly from pixels and words within a unified backbone. Early works such as Fuyu (Bavishi et al., 2023) and EVE (Diao et al., 2024, 2025b) demonstrate that image patches can be integrated directly into decoder-only Transformers without separate visual encoders, establishing the feasibility of fully native multimodal modeling. Subsequent efforts further improve this paradigm through visual encoder distillation (Diao

- et al., 2024; Li et al., 2025e; Wang et al., 2025b), modality-specific parameterization (Diao et al.,
- 2025b; Luo et al., 2024, 2025), and shared multimodal representations (Diao et al., 2025a; Tao et al., 2025; Yan et al., 2025). Notably, NEO (Diao et al., 2025a) further formalizes native multimodal learning and substantially narrows the gap to strong modular VLMs through shared pixel–word representations and unified cross-modal reasoning.

Building on this direction, recent studies (Yi et al., 2025; Li et al., 2025c) extend native VLMs to the video domain, enabling end-to-end modeling of fine-grained video–language interactions and temporal dynamics. However, these efforts remain primarily focused on video understanding, without addressing broader multimodal settings involving single-image understanding, multi-image reasoning, spatial intelligence, and other unified perception tasks. In contrast, NEO-ov further advances this direction by extending native modeling from predominantly single-image settings to a unified framework spanning single-image, multi-image, and video inputs, moving native VLMs closer to a general one-vision foundation architecture.

the red pill , NEO ov !

[Figure 2]

#### Native Vision - Language Foundation Model

Native One-Vision Model with Encoder-free Monolithic Architecture

###### Word Embedding Layer

###### Patch Embedding Layer

[Figure 3]

[Figure 4]

Take the red

pill , NEO ov

Image Inputs with Original Resolutions Video Input with Timestamp in Text Format

- Figure 1: Overview of the NEO-ov model. Image or video inputs and text are encoded into token sequences via lightweight patch and word embeddings, then processed within a single decoder-only backbone composed of stacked native primitives, enabling efficient pixel–word and pixel–pixel alignment as well as spatial-temporal reasoning.

###### 3 NEO-ov: Native One-Vision Modeling

Pre-Buffer and Post-LLM layers from NEO (Diao et al., 2025a) and Qwen3 (Yang et al., 2025a).

NEO-ov is a native vision-language model that extends unified autoregressive modeling from singleimage understanding to multi-image understanding, video understanding, and spatial intelligence. By organizing images, frames, regions, and text into a unified sequence, NEO-ov naturally supports crossimage reasoning, temporal understanding, and spatial localization. To scale from single-image inputs to ordered visual sequences, we introduce a unified serialization scheme together with spatiotemporal attention mechanisms, enabling both high-level semantic reasoning and fine-grained spatial-temporal representation within one native backbone.

For attention heads, NEO-ov still adopts an explicit THW-decoupled design that preserves the original LLM’s head dimension as the temporal component T, while introducing extra head dimensions for the spatial components H and W. This retains the temporal modeling capability inherited from the LLM while augmenting it with dedicated spatial modeling capacity. For tokens i and j, the Query (Q) and Key (K) features are defined as:

qi = [qTi ;qHi ;qWi ], kj = [kTj ;kHj ;kWj ]. (2) Their correlation is then defined as:

###### 3.1 Revisiting Native Modeling

sij = ⟨qTi ,kTj ⟩ + ⟨qHi ,kHj ⟩ + ⟨qWi ,kWj ⟩. (3)

Following NEO (Diao et al., 2025a), NEO-ov adopts a unified native vision-language backbone. In Figure 1, we encode the image I into visual tokens by a lightweight embedding layer using two convolutional layers with a GELU activation:

The T branch models textual order, cross-image relations, and cross-frame dependencies, while the H and W branches capture 2D spatial structure.

For rotary positional embedding (RoPE), we continue to implement Native-RoPE with separate temporal and spatial index modeling in Figure 2 (1):

xv = Conv2(GELU(Conv1(I)) + PE), xt = Tokenizer(T),

(1)

where xv ∈ Rnv×d, xt ∈ Rnt×d, and PE denote visual, textual, and 2D RoPE embeddings (Su et al.,

###### idxi = [ti,hi,wi], (4)

- 2024), respectively. The text input T is tokenized using original LLM tokenizer. Besides, Conv1 extracts patches with stride 16, while Conv2 aggregates local features with stride 2, producing one visual token for each 32 × 32 image region. The visual tokens are wrapped with <img> and </img>, concatenated with the text tokens, and jointly processed by one unified backbone. We initialize the

where ti denotes the temporal or sequential positions, and hi,wi denote the spatial coordinates. Text tokens retain only the temporal index, with hi

= wi = 0, whereas image tokens share the same temporal index within each image and use hi and wi to encode spatial positions. Temporal indices remain continuous across modalities, while spatial indices are independently defined within each image.

###### (1) Native Rotary Position Embedding (Native RoPE)

###### (2) Native Attention

| |
|---|

| |
|---|

| |
|---|

|<img>|
|---|

|</img>|
|---|

[Figure 5]

[Figure 6]

| |
|---|

| |
|---|

|<img>|
|---|

|</img>|
|---|

[Figure 7]

|red|
|---|

|pills|
|---|

[Figure 8]

[Figure 9]

- Img1Text

Img1 Img2 Text

- Img2

T :	0	- D <

<

<

<

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[0,0,0] [1,0,0] [1,0,1] [2,0,0] [3,0,0] [4,0,0] [4,0,1] [5,0,0] [6,0,0] [7,0,0]

- [4,0,2]
- [4,1,2]

…

𝑫 𝟐

H :	0	-

| |
|---|

| |
|---|

| |
|---|

| |
|---|

…

QK

[Figure 15]

[Figure 16]

[Figure 17]

Dimensionhead

[1,1,0] [1,1,1] [4,1,0] [4,1,1]

…

| |
|---|

| |
|---|

| |
|---|

- - T rope	index	:	0 ~	𝒏 ∗ 𝟏𝟎𝟒～𝟔
- - H / W	rope	index :	0 ~	𝒏 ∗ 𝟏𝟎𝟐

- - T base rope	θ :	1M
- - H / W	rope	θ :	10K

𝑫 𝟐

W	:	0	-

[4,2,0] [4,2,1] [4,2,2]

- Figure 2: Overview of native rotary position embeddings and spatial-temporal attention. It unifies bidirectional spatial interactions within images with causal dependencies across text and video frames via THW-aware frequency, channel, and index allocation, enabling unified modeling across single-image, multi-image, and video understanding.

###### 3.2 Unified Visual Serialization

###### 3.3 Unified Spatial-Temporal Attention

Compared with single-image modeling, the central challenge in multi-image and video understanding lies not merely in handling longer sequences, but in enabling coherent interactions across multiple visual units within a unified backbone. To address this, we extend native mixed attention from a single visual unit to multiple images and temporally ordered video frames, allowing spatial and temporal dependencies to emerge jointly within the same end-to-end autoregressive framework.

For one single image, the model inserts one visual segment at the corresponding <img> position. For multi-image inputs, each <img> token in the prompt is replaced by an independent visual segment, following the textual order in which it appears. As a result, multiple images are represented as distinct visual units in the same sequence:

Xmulti = [ xt1, <img> xv1 </img>, . . . , xtm, <img> xvm </img>, q ].

(5)

In Figure 2 (2), we treat each image or sampled frame as an independent visual unit. Tokens within the same visual unit attend bidirectionally, while interactions across different visual units remain autoregressive. Let ui denote the visual unit index of token i, where ui = 0 indicates a text token and ui > 0 denotes a visual token from an image or video frame. The attention mask is defined as

Here, xvk denotes the visual segment of the k-th image. Each image is independently encoded at arbitrary resolution, so that the number of visual tokens adapts to its spatial size rather than being constrained to a fixed token budget. This allows different images to preserve visual details at different granularities, which is beneficial for fine-grained comparison and spatially sensitive tasks.

Mij = 1 ⇐⇒ j ≤ i ∨ ui = uj > 0 . (7)

For video inputs, NEO-ov represents the video as a temporally ordered sequence of sampled frames rather than a single global embedding. Specifically, we sample f frames from the raw video and serialize each frame as an image unit associated with a timestamp. Here we further prepend temporal cues to facilitate temporal localization and cross-frame reasoning. Given sampled frames with timestamps τ1,...,τf, the video input is written as

This design yields two important properties. First, tokens within the same visual unit attend bidirectionally, enabling dense spatial interactions inside each image or frame and allowing rich intraimage structure to be modeled directly. Second, interactions across different visual units remain causal, such that each unit can attend to all preceding text and visual tokens. Unlike modular VLMs, where cross-image or cross-frame reasoning operates on representations already compressed by an external visual encoder, our design allows interactions to emerge directly from patch-level tokens at the earliest layers of the backbone and evolve progressively throughout the network. Consequently, cross-image comparison and temporal reasoning are refined jointly from shallow to deep layers, enabling more precise modeling of fine-grained visual differences and subtle temporal dynamics.

Xvideo = [ pglobal, [τ1] : <img> xv1 </img>, . . . ,

(6)

[τf] : <img> xvf </img>, q ].

Here, pglobal denotes a global prefix encoding the video duration, the number of sampled frames, and the sampling rate when available. Temporal information is conveyed jointly with explicit timestamps and frame order within the unified sequence, allowing video understanding to emerge naturally within the same framework as multi-image understanding.

Stage 1: Pre-Training Stage 3: Supervised Fine-Tuning

Stage 2: Mid-Training

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

## 🔥 🧊

### NEO-ov 🔥 🔥 PEL WEL 🧊 🔥 PEL WEL 🔥

### 🔥

NEO-ov

##### NEO-ov

[Figure 24]

Pretrained Pre-Buffer, New QK

[Figure 25]

[Figure 26]

[Figure 27]

🔥 WEL

🔥

[Figure 28]

[Figure 29]

[Figure 30]

PEL

60M Single-Image Multi-Image, Video

256 - 4096 Any Resolution

6M High-Quality Image / Video Data

256 - 4096 Any Resolution

20M Image-Text Caption Data Pairs

256 - 1024 Any Resolution

- Figure 3: Overview of three-stage training recipe. NEO-ov first aligns the Pre-Buffer with the post-LLM using large-scale image-text data while preserving the language abilities of the pretrained LLM. After that, it is optimized with diverse image and video training data to improve spatial-temporal reasoning. Finally, high-quality instruction tuning data further enhances general multimodal understanding, fine-grained perception, and temporal dynamics.

###### 3.4 Training Procedure

Supervised Fine-Tuning Stage. In this stage, the model is refined using high-quality instructiontuning data, including approximately 4M singleimage, 1M multi-image, and 1M video samples, to enhance multimodal understanding and crossframe reasoning. The training corpus covers visual question answering, OCR understanding, finegrained perception, temporal reasoning, mathematical analysis, and complex dialogue. The entire model is optimized end-to-end under next-token prediction objectives, further strengthening finegrained perception, long-context reasoning, and temporal dynamics modeling. Combined with multi-resolution training up to 40962 and videos of up to 128 frames, this stage equips the model with strong generalization across a wide range of real-world multimodal visual understanding tasks.

Our training covers three progressive stages: pretraining, mid-training, and supervised fine-tuning.

Pre-Training Stage. At this stage, the model develops foundational visual perception while progressively aligning visual representations with the semantic space of the pretrained language backbone. Training is conducted on approximately 20M large-scale image–text pairs collected from diverse web sources, spanning both descriptive captions and OCR-intensive content. To preserve the linguistic priors of the pretrained LLM and ensure stable multimodal adaptation, optimization is restricted to the patch embedding layers, pre-buffer layers, and newly introduced QK-related parameters. An autoregressive next-token objective aligns visual tokens with the LLM representation space, while pretrained buffer initialization and expanded QK capacity allow visual specialization to emerge without compromising language performance.

###### 4 Experiment

###### 4.1 Implementation Details

The NEO-ov model is trained on sixteen 8-GPU nodes, each equipped with 80 GB GPUs. Here we use the AdamW optimizer (Loshchilov and Hutter, 2019) with cosine learning-rate decay and a warmup ratio of 0.01. The peak learning rates for the three training stages are set to 2 × 10−4, 5 × 10−5, and 5 × 10−5, respectively. We use Qwen3-1.7B and Qwen3-8B (Yang et al., 2025a) as the language backbones. The pre-buffer module consists of 12 layers for NEO-ov (2B) and 6 layers for NEO-ov (9B). The native RoPE base frequencies, θT, θH, and θW, are fixed at 1 × 106, 1 × 104, and 1 × 104.

Mid-Training Stage. This stage focuses on scaling spatial-temporal reasoning and enhancing perception over high-resolution visual content. Training continues on nearly 60M multimodal samples, covering resolutions from 2562 to 40962 and videos of up to 128 frames. At this stage, all model layers are jointly optimized to strengthen crossmodal interaction and contextual coherence across both pixel-world and pixel-pixel relations. The context length is progressively extended from 16K to 36K tokens, enabling more effective modeling of high-resolution inputs and long video sequences. To support diverse application scenarios, we adopt a unified mixture of text-only, image-text, multiimage, and video-text data with an approximate ratio of 2:4:1:1, improving optimization stability and generalization across heterogeneous tasks.

###### 4.2 Main Results

We evaluate NEO-ov using VLMEvalKit (Duan et al., 2024) on three domains: image understanding, video understanding, and spatial intelligence.

General VQA Understanding OCR Recognization MMMU MMB RWQA MMStar SEED-I HallB AI2D DocVQA ChartQA TextVQA OCRBench

Model

▼ Modular Vision-Language Models (Instruct-2B)

- Qwen2-VL 41.1 74.9 62.6 48.0 – 41.7 74.7 90.1 73.5 79.7 80.9 InternVL3 48.6 81.1 64.3 60.7 – 42.5 78.7 88.3 80.2 77.0 83.5 InternVL3.5 53.0 78.2 62.0 62.7 75.3 48.6 78.8 89.4 80.7 76.5 83.6
- Qwen3-VL 53.4 78.4 63.9 58.3 – 51.4 76.9 93.3 79.1 – 85.8

▼ Native Vision-Language Models (Instruct-2B) Mono-VL 33.7 65.5 – – 67.4 34.8 68.6 80.0 73.7 72.6 76.7 Mono-VL1.5 39.1 64.0 – – 66.9 32.5 67.4 81.7 72.2 73.7 80.1 HoVLE 32.2 73.3 – – 70.9 38.4 73.0 86.1 78.6 70.9 74.0 OneCAT 39.0 72.4 – – 70.9 – 72.4 87.1 76.2 67.0 – NEO 48.6 76.0 63.1 54.2 74.2 43.1 80.1 89.9 81.2 74.0 77.1 NEO-ov 54.7 80.0 64.4 58.6 76.2 54.5 81.4 91.2 83.1 77.3 81.2

▼ Modular Vision-Language Models (Instruct-8B)

Qwen2.5-VL 55.0 83.5 68.5 63.9 – 52.9 83.9 95.7 87.3 84.9 86.4 InternVL3 62.7 83.4 70.8 68.2 – 49.9 85.2 92.7 86.6 80.2 88.0 InternVL3.5 68.1 82.7 67.5 69.3 77.1 54.5 84.0 92.3 86.7 78.2 84.0 Qwen3-VL 69.6 84.5 71.5 70.9 – 61.1 85.7 96.1 89.6 – 89.6

▼ Native Vision-Language Models (Instruct-8B) Fuyu 27.9 10.7 43.7 – 59.3 – 64.5 – – – 36.6 EVE 32.6 52.3 – – 64.6 26.4 61.0 53.0 59.1 56.8 39.8 SOLO – 67.7 44.7 – 64.4 – 61.4 – – – 12.6 EVEv2 39.3 66.3 62.4 – 71.4 – 74.8 – 73.9 71.1 70.2 BREEN 42.7 71.4 – 51.2 – 37.0 76.4 – – 65.7 – VoRA 32.0 61.3 60.1 – 68.9 – 61.1 – – 58.7 – SAIL – 70.1 63.9 53.1 72.9 54.2 76.7 – – 77.1 78.3 NEO 54.6 82.1 67.3 62.4 76.3 46.4 83.1 88.6 82.1 75.0 77.7 NEO-ov 68.1 85.1 67.8 67.3 76.6 59.8 85.4 91.9 86.2 78.5 81.6

Table 1: Comparison with existing popular VLMs on general VQA and OCR benchmarks.

Image Understanding. We test NEO-ov on general visual perception and reasoning benchmarks such as MMMU (Yue et al., 2024), MMBenchEN (MMB) (Liu et al., 2024b), RealWorldQA (RWQA) (xAI, 2024), MMStar (Chen et al.,

- 2024a), and SEEDBench-IMG (SEED-I) (Li et al., 2023); document, diagram, chart, and text understanding benchmarks including AI2D (Kembhavi et al., 2016), DocVQA (Clark and Gardner, 2018), ChartQA (Masry et al., 2022), InfoVQA (Mathew et al., 2022), TextVQA (Singh et al., 2019), and OCRBench (Liu et al., 2023b); hallucination task on HallusionBench (HallB) (Guan et al., 2024).

Comparison with Native VLMs. As shown in Table 1, NEO-ov establishes a new performance frontier for native VLMs at both 2B and 8B scales, consistently surpassing prior native architectures including NEO (Diao et al., 2025a), EVE series (Diao

- et al., 2024, 2025b), Mono-InternVL series (Luo

- et al., 2024, 2025), OneCAT (Li et al., 2025b), Emu3 (Wang et al., 2024b), and SAIL (Lei et al.,

- 2025). The gains are particularly pronounced

on reasoning-intensive and hallucination-sensitive benchmarks such as MMMU, HallB, and InfoVQA, demonstrating that native end-to-end modeling can unlock strong visual reasoning and representation learning even without external visual encoders. It further underscores the scalability and emerging competitiveness of the native one-vision paradigm.

Comparison with Modular VLMs. Beyond native models, NEO-ov also demonstrates strong competitiveness against leading modular VLMs such as InternVL3.5 (Wang et al., 2025e) and Qwen3VL (Bai et al., 2025a). Despite operating without pretrained visual encoders, NEO-ov matches or surpasses its modular counterpart (Wang et al., 2025e) on several reasoning and perception benchmarks, particularly in complex reasoning and hallucination suppression. While OCR-intensive tasks remain challenging, native architectures are rapidly closing the gap with modular systems across diverse image understanding benchmarks. Overall, these findings further validate the competitiveness and scalability of fully native multimodal modeling.

Multi-Image Video Understanding BLINK MUIRBENCH VideoMME MVBench LVBench MLVU LongVideoBench VideoMMMU

Model

▼ Modular Vision-Language Models (Instruct-2B) VideoLLaMA3 44.2 – 59.6 65.5 41.6 65.4 57.1 – InternVL3.5 51.3 44.0 58.4 65.9 37.6 64.4 57.4 42.7

- Qwen3-VL 53.8 47.4 61.9 61.7 47.4 68.3 55.6 41.9

▼ Native Vision-Language Models (Instruct-2B) ELVA – – 41.8 43.5 – 47.6 – – NEO-ov 53.9 56.8 60.4 65.7 43.3 64.8 56.8 42.3

▼ Modular Vision-Language Models (Instruct-8B) LLaVA-Video – – 63.3 58.6 44.2 70.8 58.2 – VideoLLaMA3 56.7 – 66.2 69.7 45.3 73.0 59.8 – InternVL3.5 59.5 55.8 66.0 72.1 45.9 70.2 62.1 54.9 Qwen3-VL 69.1 64.4 71.4 68.7 58.0 78.1 63.6 65.3

▼ Native Vision-Language Models (Instruct-8B)

Fuyu – – 28.7 31.6 – 31.1 – – EVE – – 29.3 34.9 – 36.8 – – ELVA – – 47.1 51.2 – 51.8 – – NEO-ov 62.8 58.2 67.4 70.7 46.4 69.3 63.5 51.6

- Table 2: Comparison with existing popular VLMs on multi-image and video benchmarks.

Model VSI-Bench MMSI Mindcube ViewSpatial SITE 3DSR EmbSpatial SPAR Omni-Spatial

▼ Spatial-specialist Models (Instruct-2B)

Cambrian-S (3B) 56.1 27.0 38.4 41.0 31.0 41.4 63.5 33.0 41.9 Sensenova-SI 63.7 34.2 41.8 52.7 36.8 50.5 62.8 38.0 26.4

▼ General-purpose Models (Instruct-2B)

InternVL3.5 53.8 25.6 42.1 37.9 34.8 31.4 61.5 32.4 44.4 Qwen3-VL 53.9 27.8 34.2 36.7 35.8 47.6 69.2 34.1 36.3 NEO-ov 58.4 33.6 77.2 52.8 38.4 52.9 63.8 41.2 43.1

▼ Spatial-specialist Models (Instruct-8B)

Cambrian-S 67.5 25.8 39.6 40.9 33.0 45.0 72.8 37.9 41.9 Sensenova-SI 68.8 43.3 85.7 54.7 47.7 55.5 72.0 45.8 33.0 GeoThinker 72.6 30.9 83.0 45.9 55.9 51.9 78.8 68.2 40.1

▼ General-purpose Models (Instruct-8B)

InternVL3.5 56.3 29.1 40.4 40.0 54.4 35.3 75.7 38.2 47.8 Qwen3-VL 59.4 31.2 29.6 41.9 45.4 52.9 77.8 40.3 47.0 NEO-ov 64.8 41.3 90.0 55.2 54.3 61.7 78.8 48.8 45.0

- Table 3: Comparison with existing popular VLMs on spatial intelligence benchmarks.

Multi-Image and Video Understanding. Compared with prior native VLMs such as Fuyu (Bavishi et al., 2023), EVE (Diao et al., 2024), and ELVA (Li et al., 2025c) in Table 2, NEO-ov achieves substantial gains on VideoMME (Fu

- et al., 2025), MVBench (Li et al., 2024b), and MLVU (Zhou et al., 2025), highlighting its strong temporal reasoning and long-context visual understanding capabilities at both 2B and 8B scales. It also remains highly competitive with several modular VLMs, including VideoLLaMA3 (Zhang et al.,

- 2025a) and InternVL3.5 (Wang et al., 2025e) on

BLINK (Fu et al., 2024), MUIRBENCH (Wang et al., 2025a), LVBench (Wang et al., 2025d), LongVideoBench (Wu et al., 2024), and VideoMMMU (Hu et al., 2025). These results indicate that a unified native backbone can naturally support cross-image reasoning and temporal association within a single autoregressive framework.

Spatial Intelligence. In Table 3, NEO-ov displays strong spatial intelligence across geometric reasoning, spatial perception, and embodied understanding benchmarks. Compared with spatialspecialist models such as Cambrian-S (Yang et al.,

###### Avg. Accuracy (%)

###### Avg. Accuracy (%)

###### Avg. Accuracy (%)

60

70

80

Image Encoder Video Encoder Pre-Buffer

Baseline Tuned by SI Data

Stage 1 Stage 2

50

60

70

40

50

60

30

40

50

20

30

40

VQA OCR Video SI

InternVL3.5 Qwen3-VL NEO

NEO-ov (2B) NEO-ov (9B)

Figure 4: Pre-Buffer vs. VEs on diverse tasks.

Figure 5: Finetuned on SI data.

Figure 6: Three stages.

2025c), Sensenova-SI (Cai et al., 2025), and GeoThinker (Li et al., 2026), NEO-ov, as a generalpurpose native VLM, achieves comparable or even better performance at both 2B and 8B scales. In particular, NEO-ov shows clear advantages over other general VLMs on VSI-Bench (Yang et al., 2025b), MMSI (Yang et al., 2025d), Mindcube-tiny (Mindcube) (Wang et al., 2025c), ViewSpatial (Li et al., 2025a), SITE (Wang et al., 2025f), 3DSR (Ma et al., 2025), EmbSpatial (Du et al., 2024), SPAR (Zhang

- et al., 2026), and Omni-Spatial (manual CoT) (Jia

- et al., 2025), highlighting its ability to capture finegrained spatial and geometric representations.

- 4.3 Ablation Studies

Native Attention vs. Encoder-based Attention.

- Figure 4 compares the Pre-Buffer mechanism with conventional visual encoders across diverse tasks, including general VQA, OCR, video understanding (Video), and spatial intelligence (SI). Both architectures are randomly initialized for fair comparison. In image encoders, attention is restricted to bidirectional interactions among visual tokens within the same image, while video encoders further extend such interactions across frames. We can observe that Pre-Buffer consistently achieves competitive or superior performance across all benchmarks, especially on OCR and SI tasks, where fine-grained visual structure and long-range spatial dependencies are especially critical. These gains suggest that preserving richer intermediate visual context through native pixel-pixel and pixel-word interactions is more effective than relying solely on compressed image- or video-level representations. Moreover, the consistent performance across VQA, OCR, Video, and SI benchmarks highlights the strong generalization capability of native architectures under diverse multimodal scenarios.

Deep Interactions Benefit Spatial Intelligence.

- Figure 5 highlights a clear advantage of native ar-

chitectures on spatial intelligence tasks. Although all models benefit from additional SI supervision, NEO shows substantially larger gains than encoderbased models such as InternVL3.5 and Qwen3-VL. We attribute this to the native interaction pattern of NEO, where pixel-pixel and pixel-word interactions emerge directly in shallow layers of the unified backbone, enabling richer spatial and crossmodal representations from the early fusion.

Performance Improvements across Stages. Figure 6 illustrates performance evolution across all single-image, multi-image, video, and spatial intelligence benchmarks. Performance improves consistently from Stage 1 to Stage 2 for both the 2B and 9B variants of NEO-ov, with especially pronounced gains at smaller scales. These results suggest that progressive training effectively strengthens general visual understanding and leads to more robust multimodal capabilities across diverse tasks.

###### 5 Conclusion

In this paper, we launch NEO-ov, a fully native vision–language foundation model that unifies singleimage understanding, multi-image reasoning, video comprehension, and spatial intelligence within a single monolithic backbone. Unlike conventional modular VLMs, NEO-ov learns visual perception, temporal dynamics, and cross-modal correspondence directly from raw inputs through end-toend training, without relying on external visual encoders. Extensive experiments demonstrate that NEO-ov achieves competitive performance against strong encoder-based counterparts while showing clear advantages in fine-grained perception and spatial reasoning. Beyond empirical results, our findings suggest that unified native architectures provide a promising path toward scalable and generalpurpose one-vision foundation models.

###### 6 Limitations

Despite the strong empirical performance of NEOov, several challenges remain open for future exploration. First, although NEO-ov substantially advances native vision-language modeling, a gap still exists between NEO-ov and top-tier modular systems such as Qwen3-VL on certain single-image and video understanding benchmarks. We believe this gap is largely attributable to the current scale and quality of multimodal training data, particularly for complex reasoning, temporal perception, and fine-grained visual-text alignment.

Second, OCR-intensive and document-centric tasks remain relatively underexplored for native architectures. Unlike modular VLMs that benefit from specialized visual encoders and extensive OCR-oriented pretraining, NEO-ov currently lacks sufficiently diverse and high-quality supervision for documents, charts, and dense text perception. We expect that improving OCR-related data scales and quality will further strengthen them.

Finally, while NEO-ov already shows promising capabilities in multi-image reasoning, video understanding, and spatial intelligence, the broader potential of native multimodal modeling remains far from fully explored. Further scaling in model capacity, multimodal data diversity, and long-context training may unlock substantially stronger multimodal reasoning and perception capabilities.

###### 7 Ethical Considerations

All resources are drawn from open-access datasets with explicitly defined usage policies. Our work seeks to advance multimodal learning capabilities without introducing ethical or safety concerns beyond those already associated with existing models. Nevertheless, risks such as dataset biases and potential misuse cannot be entirely ruled out. We emphasize the importance of careful data curation, responsible deployment, and transparent reporting as essential practices to mitigate these challenges.

During manuscript preparation, large language models were used solely as writing assistants. They helped to check grammar, refine sentence structure, and provide style alternatives. All content related to methodology, experiments, and conclusions was developed entirely by the authors. LLM outputs were reviewed critically, and only human-verified edits were incorporated into the final text.

###### References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, and 8 others. 2022. Flamingo: a visual language model for few-shot learning. In Advances of Neural Information Processing Systems, New Orleans, LA, USA.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025a. Qwen3-vl technical report. CoRR, abs/2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, MingHsuan Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025b. Qwen2.5-vl technical report. CoRR, abs/2502.13923.

Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. 2023. Introducing our multimodal models.

Zhongang Cai, Ruisi Wang, Chenyang Gu, Fanyi Pu, Junxiang Xu, Yubo Wang, Wanqi Yin, Zhitao Yang, Chen Wei, Qingping Sun, and 1 others. 2025. Scaling spatial intelligence with multimodal foundation models. arXiv preprint arXiv:2511.13719.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. 2024a. Are we on the right way for evaluating large vision-language models? In Advances of Neural Information Processing Systems, Vancouver, BC, Canada.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, and 21 others. 2024b. Expanding performance boundaries of opensource multimodal models with model, data, and test-time scaling. CoRR, abs/2412.05271.

Christopher Clark and Matt Gardner. 2018. Simple and effective multi-paragraph reading comprehension. In Annual Meeting of the Association for Computational Linguistics, pages 845–855, Melbourne, Australia.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi.

2023. Instructblip: towards general-purpose visionlanguage models with instruction tuning. In Advances of Neural Information Processing Systems, New Orleans, LA, USA.

Haiwen Diao, Yufeng Cui, Xiaotong Li, Yueze Wang, Huchuan Lu, and Xinlong Wang. 2024. Unveiling encoder-free vision-language models. CoRR, abs/2406.11832.

Haiwen Diao, Mingxuan Li, Silei Wu, Linjun Dai, Xiaohua Wang, Hanming Deng, Lewei Lu, Dahua Lin, and Ziwei Liu. 2025a. From pixels to words–towards native vision-language primitives at scale. CoRR, abs/2510.14979.

Haiwen Diao, Xiaotong Li, Yufeng Cui, Yueze Wang, Haoge Deng, Ting Pan, Wenxuan Wang, Huchuan Lu, and Xinlong Wang. 2025b. Evev2: Improved baselines for encoder-free vision-language models. CoRR, abs/2502.06788.

Mengfei Du, Binhao Wu, Zejun Li, Xuan-Jing Huang, and Zhongyu Wei. 2024. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 346–355.

Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. 2024. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In ACM International Conference on Multimedia, pages 11198– 11201, Melbourne, VIC, Australia.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, and 1 others. 2025. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24108–24118.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, WeiChiu Ma, and Ranjay Krishna. 2024. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. 2024. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In IEEE Conference on Computer Vision and Pattern Recognition, pages 14375–14385, Seattle, WA, USA.

Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. 2025. Video-mmmu: Evaluating knowledge acquisition

from multi-discipline professional videos. arXiv preprint arXiv:2501.13826.

Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. 2025. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Min Joon Seo, Hannaneh Hajishirzi, and Ali Farhadi. 2016. A diagram is worth a dozen images. In European Conference on Computer Vision, volume 9908, pages 235–251, Amsterdam, The Netherlands.

Weixian Lei, Jiacong Wang, Haochen Wang, Xiangtai Li, Jun Hao Liew, Jiashi Feng, and Zilong Huang. 2025. The scalability of simplicity: Empirical analysis of vision-language learning with a single transformer. CoRR, abs/2504.10462.

Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. 2024a. Llava-next: stronger llms supercharge multimodal capabilities in the wild.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023. Seed-bench: benchmarking multimodal llms with generative comprehension. CoRR, abs/2307.16125.

Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, and 1 others. 2025a. Viewspatial-bench: Evaluating multiperspective spatial localization in vision-language models. arXiv preprint arXiv:2505.21500.

Han Li, Xinyu Peng, Yaoming Wang, Zelin Peng, Xin Chen, Rongxiang Weng, Jingang Wang, Xunliang Cai, Wenrui Dai, and Hongkai Xiong. 2025b. Onecat: Decoder-only auto-regressive model for unified understanding and generation. CoRR, abs/2509.03498.

Handong Li, Yiyuan Zhang, Longteng Guo, Xiangyu Yue, and Jing Liu. 2025c. Breaking the encoder barrier for seamless video-language understanding. arXiv preprint arXiv:2503.18422.

Haoyuan Li, Qihang Cao, Tao Tang, Kun Xiang, Zihan Guo, Jianhua Han, Hang Xu, JiaWang Bian, and Xiaodan Liang. 2026. Thinking with geometry: Active geometry integration for spatial reasoning. arXiv preprint arXiv:2602.06037.

KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. 2025d. Videochat: Chat-centric video understanding. Science China Information Sciences, 68(10):200102.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, and 1 others. 2024b. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference

on Computer Vision and Pattern Recognition, pages 22195–22206.

Tianle Li, Yongming Rao, Winston Hu, and Yu Cheng. 2025e. BREEN: bridge data-efficient encoder-free multimodal learning with learnable queries. CoRR, abs/2503.12446.

Jiaqi Liao, Yuwei Niu, Fanqing Meng, Hao Li, Changyao Tian, Yinuo Du, Yuwen Xiong, Dianqi Li, Xizhou Zhu, Li Yuan, and 1 others. 2025. Langbridge: Interpreting image as a combination of language embeddings. arXiv preprint arXiv:2503.19404.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024a. Improved baselines with visual instruction tuning. In IEEE Conference on Computer Vision and Pattern Recognition, pages 26286–26296, Seattle, WA, USA.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023a. Visual instruction tuning. In Advances of Neural Information Processing Systems, New Orleans, LA, USA.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. 2024b. Mmbench: is your multi-modal model an all-around player? In European Conference on Computer Vision, volume 15064, pages 216–233, Milan, Italy.

Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, and Xiang Bai. 2023b. On the hidden mystery of ocr in large multimodal models. CoRR, abs/2305.07895.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations, New Orleans, LA, USA.

Gen Luo, Wenhan Dou, Wenhao Li, Zhaokai Wang, Xue Yang, Changyao Tian, Hao Li, Weiyun Wang, Wenhai Wang, Xizhou Zhu, Yu Qiao, and Jifeng Dai. 2025. Mono-internvl-1.5: Towards cheaper and faster monolithic multimodal large language models. CoRR, abs/2507.12566.

Gen Luo, Xue Yang, Wenhan Dou, Zhaokai Wang, Jifeng Dai, Yu Qiao, and Xizhou Zhu. 2024. Monointernvl: pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training. CoRR, abs/2410.08202.

Wufei Ma, Haoyu Chen, Guofeng Zhang, Yu-Cheng Chou, Jieneng Chen, Celso de Melo, and Alan Yuille. 2025. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6924–6934.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, and Enamul Hoque. 2022. Chartqa: a benchmark for question answering about charts with visual and logical reasoning. In Annual Meeting of the Association for Computational Linguistics, pages 2263–2279, Dublin, Ireland.

Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. 2022. Infographicvqa. In IEEE Winter Conference on Applications of Computer Vision, pages 2582–2591, Waikoloa, HI, USA.

Lingchen Meng, Jianwei Yang, Rui Tian, Xiyang Dai, Zuxuan Wu, Jianfeng Gao, and Yu-Gang Jiang. 2024. Deepstack: Deeply stacking visual tokens is surprisingly simple and effective for lmms. Advances of Neural Information Processing Systems, 37:23464– 23487.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, volume 139, pages 8748–8763, virtual.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In IEEE Conference on Computer Vision and Pattern Recognition, pages 8317–8326, Long Beach, CA, USA.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063.

Chenxin Tao, Shiqian Su, Xizhou Zhu, Chenyu Zhang, Zhe Chen, Jiawen Liu, Wenhai Wang, Lewei Lu, Gao Huang, Yu Qiao, and Jifeng Dai. 2025. Hovle: Unleashing the power of monolithic vision-language models with holistic vision-language embedding. In IEEE Conference on Computer Vision and Pattern Recognition, pages 14559–14569, Nashville, TN, USA.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian CantonFerrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, and 49 others. 2023. Llama 2: open foundation and fine-tuned chat models. CoRR, abs/2307.09288.

Michael Tschannen, Alexey A. Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier J. Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. 2025. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. CoRR, abs/2502.14786.

Fei Wang, Xingyu Fu, James Y Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, and 1 others. 2025a. Muirbench: A comprehensive benchmark for robust multi-image understanding. In International Conference on Learning Representations, volume 2025, pages 62624–62650.

Han Wang, Yongjie Ye, Bingru Li, Yuxiang Nie, Jinghui Lu, Jingqun Tang, Yanjie Wang, and Can Huang. 2025b. Vision as lora. CoRR, abs/2503.20680.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. 2024a. Qwen2-vl: enhancing vision-language model’s perception of the world at any resolution. CoRR, abs/2409.12191.

Qineng Wang, Baiqiao Yin, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, and 1 others. 2025c. Mindcube: Spatial mental modeling from limited views. arXiv e-prints, pages arXiv–2506.

Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, and 1 others. 2025d. Lvbench: An extreme long video understanding benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22958–22967.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, and 1 others. 2025e. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. CoRR, abs/2508.18265.

Wenqi Wang, Reuben Tan, Pengyue Zhu, Jianwei Yang, Zhengyuan Yang, Lijuan Wang, Andrey Kolobov, Jianfeng Gao, and Boqing Gong. 2025f. Site: towards spatial intelligence thorough evaluation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9058–9069.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, and 6 others. 2024b. Emu3: next-token prediction is all you need. CoRR, abs/2409.18869.

Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. 2024. Longvideobench: A benchmark for longcontext interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857.

xAI. 2024. Grok-1.5 vision preview.

Rui Yan, Lin Song, Yicheng Xiao, Runhui Huang, Yixiao Ge, Ying Shan, and Hengshuang Zhao. 2025. Haplovl: A single-transformer baseline for multimodal understanding. CoRR, abs/2503.14694.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 40 others. 2025a. Qwen3 technical report. CoRR, abs/2505.09388.

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. 2025b. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643.

Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis L Brown II, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, and 1 others. 2025c. Cambrian-s: Towards spatial supersensing in video. In The Fourteenth International Conference on Learning Representations.

Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, and 1 others. 2025d. Mmsibench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764.

Jinhui Yi, Syed Talal Wasim, Yanan Luo, Muzammal Naseer, and Juergen Gall. 2025. Video-panda: Parameter-efficient alignment for encoder-free videolanguage models. In IEEE Conference on Computer Vision and Pattern Recognition, pages 24119–24128.

Xiang Yue, Yuansheng Ni, Tianyu Zheng, Kai Zhang, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, and 3 others. 2024. MMMU: a massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In IEEE Conference on Computer Vision and Pattern Recognition, pages 9556–9567, Seattle, WA, USA.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. In IEEE International Conference on Computer Vision, pages 11941–11952, Paris, France.

Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, Peng Jin, Wenqi Zhang, Fan Wang, Lidong Bing, and Deli Zhao. 2025a. Videollama 3: Frontier multimodal foundation models for image and video understanding. CoRR, abs/2501.13106.

Jiahui Zhang, Yurui Chen, Yueming Xu, Ze Huang, Jilin Mei, Chunhui Chen, Yanpeng Zhou, Yu-Jie Yuan,

Xinyue Cai, Guowei Huang, and 1 others. 2026. From flatland to space: Teaching vision-language models to perceive and reason in 3d. Advances in Neural Information Processing Systems, 38.

Yiyuan Zhang, Handong Li, Jing Liu, and Xiangyu Yue. 2025b. Learning beyond still frames: Scaling vision-language models with video. In IEEE International Conference on Computer Vision, pages 22425–22435.

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, and 1 others. 2025. Mlvu: Benchmarking multi-task long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13691– 13701.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, and 32 others. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. CoRR, abs/2504.10479.

