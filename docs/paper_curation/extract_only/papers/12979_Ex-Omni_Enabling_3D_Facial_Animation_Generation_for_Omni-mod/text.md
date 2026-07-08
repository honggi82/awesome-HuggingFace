# arXiv:2602.07106v2[cs.CV]11Jun2026

## Ex-Omni: Enabling 3D Facial Animation Generation for Omni-modal Large Language Models

Haoyu Zhang∗ 1 Zhipeng Li2 Yiwen Guo† ‡ 3 Tianshu Yu† 1 1The Chinese University of Hong Kong, Shenzhen 2LIGHTSPEED 3Independent Researcher 1{haoyuzhang3@link.cuhk.edu.cn, yutianshu@cuhk.edu.cn} 2zhipengxli@tencent.com 3guoyiwen89@gmail.com

### Abstract

Omni-modal large language models (OLLMs) aim to unify multimodal understanding and generation, yet extending them to jointly produce speech and 3D facial animation remains largely unexplored despite its importance for natural human-computer interaction. A key challenge is the mismatch between the discrete semantic reasoning of LLMs and the dense temporal dynamics required for 3D facial motion. We propose Expressive Omni (Ex-Omni), an open-source model that augments OLLMs with native speech-accompanied 3D facial animation. Ex-Omni decouples semantic reasoning from temporal generation through a blendshape-aware speech unit generator and a blendshape decoder, where speech units provide temporal scaffolding and hidden speech representations carry facially relevant cues. We further introduce a unified token-as-query gated fusion (TQGF) mechanism for controlled semantic injection, as well as InstructS2SF1200K, a dataset consisting of 1200K samples for pre-training. Extensive experiments show that Ex-Omni maintains competitive speech understanding and generation ability while achieving better audio-visual synchronization and lower face-generation latency than cascaded pipelines. a

### 1 Introduction

Large language models (LLMs) have achieved remarkable progress across a wide range of tasks (Minaee et al., 2024), demonstrating impressive generalization and reasoning capabilities. As research continues to expand from unimodal understanding toward multimodal understanding and generation, growing attention has been devoted to unifying these tasks within a single framework, commonly referred to as omni-modal large language models

∗This work was done during internship at LIGHTSPEED. †Corresponding Author. ‡ Project Leader.

aThe code is available at https://github.com/ Tencent/Ex-Omni

(OLLMs). With the continued development of such omni models (Fu et al., 2025; Xie and Wu, 2024b; Team, 2025; Li et al., 2025; Xie and Wu, 2024a; Luo et al., 2025; Xu et al., 2025; AI et al., 2025), further breakthroughs are expected in areas such as human-computer interaction and embodied intelligence.

In the area of human-computer interaction, there is a growing demand for OLLMs that can engage in natural and expressive interactions with humans. Human communication is inherently multimodal and further extends beyond verbal content alone. In face-to-face interaction, temporally coherent 3D facial animation synchronized with speech plays a crucial role in conveying non-verbal cues and enhancing interaction naturalness, particularly in applications such as virtual characters, digital avatars, and embodied agents. However, existing opensource OLLMs primarily focus on linguistic, acoustic, or pixel-level visual outputs, while expressive non-verbal modalities such as 3D facial animation remain largely underexplored. Motivated by this gap, we investigate integrating 3D facial animation generation into OLLMs. A natural idea is to directly attach a blendshape decoder to an LLM and predict animation from its hidden representations. In practice, we found this design exposes a challenge: LLM hidden states are optimized for sparse, token-level semantics with weakly constrained temporal structure, whereas 3D facial animation requires dense and temporally smooth motion at a much finer time scale. Bridging these representations forces the decoder to infer fine-grained dynamics from coarse semantic features, resulting in an ill-conditioned mapping that typically demands substantially larger model capacity and more paired speech–face supervision for stable generation.

In this paper, we propose Expressive Omni (ExOmni), an open-source omni-modal framework that augments OLLMs with speech-accompanied 3D facial animation, where facial motion is represented

using ARKit-52 blendshape coefficients (Lewis et al., 2014) and generated in a non-autoregressive manner. Ex-Omni follows text/speech instructions to generate synchronized speech paired with facial animation in an end-to-end manner. To facilitate stable and temporally coherent facial animation learning from LLM semantics, Ex-Omni decouples semantic reasoning from temporal generation through a two-stage design. Rather than directly predicting facial motion from LLM hidden states, Ex-Omni first employs a blendshape-aware speech unit generator, where discrete speech units provide explicit temporal scaffolding while the generator hidden states are encouraged to encode facially relevant cues. A blendshape decoder then predicts blendshape sequences conditioned on both signals. To better bridge high-level semantics and temporal motion generation, we further introduce a unified token-as-query gated fusion (TQGF) mechanism to selectively regulate how and when semantic information is injected into the speech and facial generation processes, simplifying optimization and improving temporal alignment.

In addition, to the best of our knowledge, we construct InstructS2SF-1200K, the first largescale dataset for augmenting OLLMs with speechaccompanied 3D facial animation. InstructS2SF1200K consists of two subsets tailored to different training stages: 1000K Text-to-SpeechFace (TTSF) samples for speech-blendshape copretraining, and 200K Speech-to-Speech-Face (S2SF) QA samples for dialogue-based speechblendshape co-pretraining. By combining large-scale supervision with speech-face data, InstructS2SF-1200K helps bridge the gap between limited real-world recordings and open-domain generalization. Experimental results show that Ex-Omni remains competitive with existing opensource OLLMs on speech tasks while offering native speech-aligned 3D facial animation generation with better synchronization and lower facegeneration latency than cascaded alternatives.

Overall, the key contributions are:

• We propose Expressive Omni (Ex-Omni), enabling unified instruction following and generation across text, speech, and speechaccompanied 3D facial animation. To the best of our knowledge, Ex-Omni is among the first open-source OLLMs to natively support speech-aligned 3D facial animation generation.

- • To reduce the difficulty of learning temporally coherent 3D facial animation from LLM semantics, Ex-Omni adopts a two-stage design consisting of a blendshape-aware speech unit generator and a Blendshape Decoder. It leverages discrete speech units as temporal scaffolding and employs a unified token-asquery gated fusion (TQGF) mechanism to regulate how and when semantic information is injected into temporal generation.
- • To the best of our knowledge, we construct InstructS2SF-1200K, the first largescale dataset for augmenting OLLMs with speech-accompanied 3D facial animation. It consists of 1000K TTSF synthesis samples and 200K dialogue-based S2SF samples, designed for speech-blendshape co-pretraining and dialogue-based speech-blendshape copretraining, respectively. Experiments show that Ex-Omni preserves competitive speechtask performance while improving synchronization and face-generation efficiency over cascaded pipelines.

### 2 Related Work

Omni-modal Large Language Models. OLLMs (Fu et al., 2025; Xie and Wu, 2024b; Team, 2025; Li et al., 2025) represent a significant advancement in multimodal large language models, as they integrate the capabilities of understanding and generating across multiple modalities, such as text, speech, and vision, within a unified framework. For example, Mini-Omni (Xie and Wu, 2024a) utilizes text-instructed speech generation and batchparallel strategies to achieve seamless speech synthesis while preserving the model’s text capabilities. OpenOmni (Luo et al., 2025) introduces a two-stage training framework to achieve zeroshot cross-modal alignment from vision-language tasks to speech-language tasks. Qwen2.5-Omni (Xu et al., 2025) introduce the Thinker-Talker architecture to integrate text, speech, and vision modalities into a unified end-to-end model. Ming-Omni (AI et al., 2025) is the first to integrate visual generation capabilities into a unified omni-modal model, utilizing modality-specific routers to achieve understanding and generation across multiple modalities.

Facial Animation Generation. Facial animation generation is an important research area for improving system interactivity. Earlier methods (Chen

et al., 2019; Mittal and Wang, 2020; Zhang et al., 2021; Hong et al., 2022) mainly focus on 2D facial animation generation, a field that has become mature after years of research. In recent years, 3D facial animation generation (Richard et al., 2021; Xing et al., 2023; Peng et al., 2023b,a; Fan et al.,

- 2024; Peng et al., 2025) has gradually received more attention. These methods have generally focused on predictions based on mesh representations or parameterized models to enhance realism. For mesh-based methods, FaceFormer (Fan

- et al., 2022) introduces a transformer-based approach for generating 3D facial animations, using autoregressive modeling to capture long-term audio context. CodeTalker (Xing et al., 2023) utilizes discrete motion priors learned from real facial movements, applying a vector quantized autoencoder to reduce uncertainty in the audio-tomotion mapping process. For parameterized methods, ARKit-like blendshape models (Lewis et al.,

2014) are commonly used. For example, (Peng

- et al., 2023b) propose EmoTalk which disentangles emotion and content from speech to generate expressive facial movements. Their approach uses an emotion-disentangling encoder and an emotionguided feature fusion decoder to improve emotional expression in 3D facial animations. (Peng et al., 2025) propose DuelTalker which supports multi-round, dual-speaker interactions in 3D talking head generation, aiming to capture dynamic interactions between speakers. Additionally, (Fan
- et al., 2024) introduce Unitalker, which combines both mesh-based and parameterized annotation styles, enabling scalable generation by leveraging the strengths of both approaches to produce more realistic and expressive 3D facial animations.

Speech Language Models. Recent advancements in speech language models (Fang et al., 2025a,b; Chen et al., 2025a; Zhang et al., 2023; Hassid et al., 2023; Chu et al., 2024; Chen et al.,

- 2025b; Xie and Wu, 2024a) have significantly improved speech understanding and generation in an end-to-end manner, eliminating the need for cascaded ASR and TTS models. For example, SpeechT5 (Ao et al., 2022) aligns text and speech representations into a shared semantic space using a unified encoder-decoder structure and crossmodal vector quantization methods. Moshi (Défossez et al., 2024) addresses the issues of latency and information bottlenecks in traditional speech dialogue systems through its full-duplex speech-to-

speech generation framework and the Inner Monologue design. SpeechGPT-Gen (Zhang et al., 2024) introduces the Chain-of-Information Generation to decouples the modeling of semantic and perceptual information,thus making the speech generation process more efficient and precise. GLM-4Voice (Zeng et al., 2024) addresses the delay and error accumulation problems by adopting a 12.5Hz speech segmenter, streaming reasoning and largescale speech-to-text pre-training.

### 3 Method

[Figure 1]

Figure 1: Model architecture of Ex-Omni.

#### 3.1 Overview

Figure 1 shows the overall pipeline of Ex-Omni. Given a text input x and a speech waveform a, Ex-Omni performs instruction understanding and multimodal generation within an LLM-centered unified framework, where the LLM focuses on semantic reasoning rather than direct temporal generation. The model produces a discrete speech unit sequence u, which is decoded into waveform speech, as well as a sequence of 3D facial animation parameters y (i.e., blendshape coefficients). At a high level, Ex-Omni adopts a structured decomposition that reduces the difficulty of learning temporally coherent generation by decoupling semantic reasoning from modality-specific temporal synthesis. Specifically, the LLM is responsible for instruction understanding and semantic reasoning, while speech units are used as an explicit temporal scaffolding to guide downstream speech and facial animation generation. We formulate the overall model as

(u, y) = F(x, a; θ), θ = {θE, θP, θL, θU, θF}, (1)

where θE, θP, and θL correspond to the speech encoder, speech projector and LLM, respectively. θU denotes the blendshape-aware speech unit generator while θF represents the Blendshape Decoder.

#### 3.2 Unified Speech-Text Representation

Given a speech waveform a and a text input x, we map both inputs into a shared LLM token embedding space. Specifically, speech is first encoded into high-level representations and then projected as

Xs = PθP (EθE(a)) ∈ RTs×d, (2)

while text tokens are embedded as

Xl = EmbθL(x) ∈ RTl×d, (3)

where d denotes the LLM embedding dimension.

The unified input is constructed by concatenation:

X = [Xl; Xs] ∈ R(Tl+Ts)×d, (4)

with positional encodings omitted for clarity.

#### 3.3 LLM-Centered Reasoning

The LLM serves as a semantic reasoner that focuses exclusively on instruction understanding and high-level reasoning. Given the unified input representation X, the LLM (Qwen3-8B) performs autoregressive generation to produce the text response t1:Tlr. During this process, we extract the sequence of last hidden states corresponding to the generated response tokens:

H = [h1, h2, . . . , hTlr] ∈ RTlr×d, (5)

where hi represents the features at step i containing high-level semantic reasoning information. The probability of the next token is predicted based on these states:

pθL(ti+1 | t1:i, X) = Softmax(Wohi). (6)

#### 3.4 Joint Speech and 3D Facial Animation Generation

Ex-Omni jointly generates speech and 3D facial animation, aiming to maintain semantic consistency and temporal alignment across modalities. We adopt a token-as-query gated fusion (TQGF) mechanism, which applies an asymmetric fusion rule where the token sequence always serves as the query, while upstream semantic representations act as contextual key/value. This design explicitly assigns temporal responsibility to the target token sequence, and selectively injects semantic

cues from the LLM via gated cross-attention. As a result, TQGF decouples high-level semantic reasoning from modality-specific temporal modeling, thereby simplifying temporal learning under limited multimodal supervision.

Formally, let Q ∈ RM×d denote query tokens and C ∈ RN×d denote context tokens. The gated fusion operation is

Fuse(Q, C) = Q + σ G(Q) ⊙ Attn(Q, C), (7)

where Attn(Q,C) is cross-attention from Q to C, G(·) is head-specific element-wise gating factors, and σ(·) is the sigmoid function.

For speech generation, we model speech synthesis as an autoregressive prediction of discrete speech units using a blendshape-aware speech unit generator (Qwen3-0.6B). Given the generated text tokens t1:Tlr, the blendshape-aware speech unit generator enriches the explicit token embeddings with the semantic hidden states H through:

H˜ = FuseθU (EmbθU (t1:Tlr), H) , (8)

where H˜ ∈ RTlr×d serves as the conditioning signal. Then, the blendshape-aware speech unit generator predicts a sequence of speech units u1:Tu:

pθU (u1:Tu | H˜) =

Tu

pθU (uj | u<j, H˜). (9)

j=1

For 3D facial animation generation, we parameterize facial motion using ARKit-52 blendshape coefficients and formulate S2F generation as a nonautoregressive sequence prediction problem, where the model outputs the full blendshape coefficients yˆ1:Ty ∈ RTy×52 in parallel. Given discrete speech units u1:Tu, we first obtain unit embeddings and align them to the target video frame rate by temporally resampling features to length Ty (e.g., via linear interpolation), yielding frame-level query representations Qy ∈ RTy×d. In parallel, we project the hidden states of the blendshape-aware speech unit generator into the same space to obtain contextual key/value representations S ∈ RTu×d, which carry semantically rich information. We then inject speech semantics into the frame-level queries using the TQGF module:

Hf = FuseθF (Qy, S) ∈ RTy×d. (10)

We then apply periodic rotary positional embeddings to Hf and refine it via a Transformer encoder, thus obtaining the predicted sequence of 3D facial parameters:

yˆ1:Ty = EθF (Hf), yˆt ∈ R52. (11)

#### 3.5 Training Strategy

- Stage I (Speech-to-Text Pretraining). We train the speech projector on ASR data while freezing all other components. This stage aligns speech representations with the semantic space of the base LLM.
- Stage II (Speech-Blendshape Co-pretraining). We pre-train the Unit Generator and the blendshape decoder on 1000K TTSF samples while freezing the base LLM and unrelated modules. This stage introduces joint speech and blendshape supervision to establish speech-blendshape alignment.
- Stage III (Dialogue-based Speech-Blendshape Co-pretraining). The LLM, speech projector, blendshape-aware speech unit generator, and blendshape decoder are jointly optimized on a mixture of ASR, TTSF, and S2SF data, while the speech encoder and speech decoder remain frozen. This stage adapts the model to dialogue-oriented speechblendshape generation.

#### 3.6 Training Objectives

Autoregressive Objectives for Text and Speech. For text tokens and discrete speech units, we adopt standard autoregressive modeling. Given a token sequence z = (z1,...,zT), the objective is

Lar = −

T

log p(zt | z<t), (12)

t=1

where zt denotes either a text token or a speech unit, depending on the supervision available for a given sample. In practice, this objective is used for text supervision on ASR and T2T data, and for discrete speech-unit prediction on TTSF and S2SF data.

3D Facial Animation Generation. For 3D facial animation, the blendshape decoder is trained with a frame-wise regression loss. Let yˆt ∈ RK and yt ∈ RK denote the predicted and ground-truth blendshape annotations at frame t, respectively. We define the facial loss as

1 B

Lbs =

B

1 |Ti| t∈T

i=1

i

y ˆt(i) − yt(i)

2 2

, (13)

where Ti denotes the valid temporal range determined by the target sequence length.

Table 1: Statistics of the three-stage training corpus.

Duration (Hour / Second) Total Mean Median Min Max Std

Stage Type

- I ASR 2113.38 10.47 10.46 0.83 30.00 4.81

- II TTSF 2814.57 10.13 8.68 0.4 161.8 5.71

- III

S2SF (Prompt) 782.73 14.09 12.68 1.16 75.36 6.61 S2SF (Response) 1434.67 25.82 23.04 0.4 296.4 14.31

ASR 25.91 9.33 8.07 3.00 30.0 5.01 TTSF 28.38 10.22 8.72 0.64 52.2 5.76

### 4 Data Construction

The data statistics of the full training corpus used by Ex-Omni are summarized in Table 1. Overall, the training data consist of an external ASR corpus together with our proposed InstructS2SF-1200K.

- In Stage I, about 720K ASR samples are primar-

ily sampled from the Emilia corpus. We further incorporate the train-clean-100, train-clean-360, and train-other-500 subsets of LibriSpeech (Panayotov et al., 2015), together with WenetSpeech (Zhang et al., 2022) training data whose confidence scores exceed 0.95.

- In Stage II, we construct 1000K TTSF samples.

Specifically, text prompts sampled from Emilia (He

- et al., 2024, 2025) are converted into speech using Qwen3-TTS (Hu et al., 2026) with a single, unified speaker identity. High-quality motion-capture facial animation data are scarce, and existing public datasets typically contain only a few thousand samples with limited coverage of speech content. We therefore adopt NVIDIA Audio2Face-3D (Chung
- et al., 2025), an open-weights model trained on large-scale professionally captured facial motion, as a teacher to generate blendshape pseudo-labels from the synthesized speech. Concretely, we first run an Audio2Emotion model on the synthesized audio to estimate a 5D compound emotion vector over anger, disgust, sadness, joy, and fear, where an all-zero vector is treated as neutral. This emotion vector is then used as the conditional input to Audio2Face-3D to generate the corresponding blendshape sequence. These teacher-generated blendshape sequences are used as weak supervision signals for speech-blendshape co-pretraining, rather than being treated as deterministic facial ground truth.

- In Stage III, we construct 200K dialogue-based

S2SF samples based on InstructS2S-200K (Fang et al., 2025a). To improve speech quality and consistency, we reconstruct the target speech using Qwen3-TTS, and then use Audio2Face-3D (Chung et al., 2025) to generate corresponding blendshape

pseudo-labels. As in Stage II, we first estimate a 5D compound emotion vector from the audio with Audio2Emotion and use it as the conditional input to Audio2Face-3D. In practice, we observe that most speech responses in QA-style dialogue are close to neutral under this pipeline, so the resulting pseudo-labels mainly emphasize accurate mouth articulation and lip-speech synchronization rather than exaggerated facial expressions. As in Stage II, these teacher-generated blendshape sequences serve as weak supervision signals, which is a more appropriate formulation given the one-to-many nature of speech-to-blendshape mapping. To preserve capabilities acquired in earlier stages, we additionally include 10K ASR replay data and 10K TTSF replay data during this stage.

### 5 Experiments

#### 5.1 Implementation Details.

All experiments are conducted on a machine equipped with 8 NVIDIA H20 GPUs, each with 96 GB of memory. We use CUDA 12.6, PyTorch 2.7.0 and Python 3.10 for model training and evaluation. The detailed hyperparameters of the three-stage training schedule are shown in Table 2.

- Table 2: The detailed training setup for Ex-Omni across the three training stages.

Hyperparameter I II III

epoch 1 2 2 effective batch size 128 128 32 optimizer AdamW AdamW AdamW warmup ratio 0.3 0.1 0.1 Gradient Accumulation 1 2 4 lr of Speech Encoder 0 0 0

lr of Speech Projector 1 × 10−3 0 2 × 10−5 lr of LLM 0 0 2 × 10−6 lr of Blendshape-Aware

0 1 × 10−4 2 × 10−5

Speech Unit Generator

lr of blendshape decoder 0 1 × 10−4 2 × 10−5 lr of Speech Decoder 0 0 0 freeze Speech Encoder ✓ ✓ ✓ freeze Speech Projector ✗ ✓ ✗ freeze LLM ✓ ✓ ✗ freeze Blendshape-Aware Speech Unit Generator

✓ ✗ ✗

freeze Blendshape Decoder ✓ ✗ ✗ freeze Speech Decoder ✓ ✓ ✓

#### 5.2 Evaluation

Speech-to-Face Evaluation. For S2F, evaluations are conducted on AlpacaEval (Fan et al., 2024) and CommonEval. We use Sync-C and SyncD from SyncNet (Chung and Zisserman, 2016) (higher Sync-C and lower Sync-D indicate better alignment) for lip-speech synchronization. SyncNet is a widely used third-party reference model for audio-visual synchronization evaluation, and is independent of the Audio2Face-3D teacher used for

pseudo-label generation. We focus on lip-speech synchronization because everyday dialogue is often neutral, and neutral responses dominate our dialogue data; in such scenarios, accurate articulation and temporal alignment are important.

Text-to-Face Evaluation. T2F evaluation follows the same protocol as S2F evaluation, except that the input is text rather than speech.

Speech QA Evaluation. We evaluate speech QA on VoiceBench (Chen et al., 2024), which covers a diverse set of speech-based tasks, including openended question answering, reference-based QA, multiple-choice QA, reasoning, instruction following and safety. Open-ended QA is evaluated using GPT-based scores (scores from 1-5), while other tasks are evaluated using accuracy-based metrics. All the evaluations were conducted using the opensource code of VoiceBench to ensure consistency.

Speech Generation Evaluation. For speech generation evaluation, we assess both response-level speech quality and speech-text consistency on the AlpacaEval and CommonEval speech QA sets. Specifically, we compute MOS on the generated speech, and transcribe the generated waveform using Whisper-Large-V3 (Radford et al., 2023). The resulting ASR transcript is then compared with the model’s textual response using Word Error Rate (WER), which measures consistency between spoken and textual outputs.

Baselines For S2F evaluation, we compare Ex-Omni with two recent S2F methods, EmoTalk (Peng et al., 2023b) and UniTalker (Fan et al., 2024), both of which support direct prediction of facial blendshape coefficients. For speech QA evaluation, we compare Ex-Omni with several representative OLLMs and speech large language models. Specifically, the baselines include Qwen2.5-Omni (Xu et al., 2025), VITA1.0 (Fu et al., 2024), VITA-1.5 (Fu et al., 2025), Mini-Omni (Chen et al., 2025b), Mini-Omni2 (Xie and Wu, 2024b), Moshi (Défossez et al., 2024), SLAM-omni (Chen et al., 2025c), and LLaMA-Omni (Fang et al., 2025a). For speech generation evaluation, we compare Ex-Omni with Qwen2.5-Omni under the same responsegeneration setting, and report both MOS and speech-text consistency measured by WER.

- Table 3: Performance comparison of 3D facial animation generation in dialogue scenes. ↓ indicates lower is better. Note: Ex-Omni+Task-specific S2F model adopt a two-stage pipeline, where Ex-Omni generates speech responses and the output audio is subsequently used as input to a S2F model. In contrast, Native Ex-Omni directly generates facial animation within a unified framework.

Method

Speech-to-Face (Sync-D ↓ / Sync-C ↑) Text-to-Face (Sync-D ↓ / Sync-C ↑) AlpacaEval CommonEval AlpacaEval CommonEval

Cascaded Qwen2.5-Omni-3B + EmoTalk 10.538 / 3.305 10.750 / 3.203 10.643 / 3.166 10.830 / 3.106 Qwen2.5-Omni-3B + Unitalker-B-D3 10.449 / 3.222 10.632 / 3.156 10.544 / 3.153 10.494 / 3.419 Qwen2.5-Omni-3B + Unitalker-B-D6 9.873 / 3.738 10.088 / 3.600 9.945 / 3.629 10.302 / 3.504 Qwen2.5-Omni-7B + EmoTalk 10.554 / 3.273 11.215 / 3.628 10.513 / 3.381 11.180 / 3.641 Qwen2.5-Omni-7B + Unitalker-B-D3 10.494 / 3.200 10.854 / 3.767 10.520 / 3.319 10.883 / 3.675 Qwen2.5-Omni-7B + Unitalker-B-D6 10.176 / 3.405 10.856 / 4.012 10.115 / 3.641 10.833 / 3.980 Ex-Omni+EmoTalk 10.758 / 4.321 10.416 / 3.710 10.534 / 4.319 10.418 / 3.490 Ex-Omni+Unitalker-B-D3 10.946 / 3.882 11.117 / 2.900 10.893 / 3.708 10.792 / 3.130 Ex-Omni+Unitalker-B-D6 10.701 / 4.343 10.668 / 3.516 10.786 / 4.049 10.334 / 3.771

Native Ex-Omni 9.233 / 5.385 9.212 / 5.363 9.313 / 5.236 9.239 / 5.403

- Table 4: Speech QA comparison on VoiceBench. ↑ means higher is better. ∗ means the results is reproduced by the authors using the open-source code.

Model AlpacaEval ↑ CommonEval ↑ WildVoice ↑ SD-QA ↑ MMSU ↑ OBQA ↑ BBH ↑ IFEval ↑ AdvBench ↑ Overall ↑ Qwen2.5-Omni-7B 4.49 3.93 2.71∗ 55.71 61.32 81.10 60.80∗ 52.87 99.42 70.42∗ Moshi 2.01 1.60 1.30 15.64 24.04 25.93 47.40 10.12 44.23 29.51 VITA-1.0 3.38 2.15 1.87 27.94 25.70 29.01 47.70 22.82 26.73 56.48 VITA-1.5 4.21 3.66 3.48 38.88 52.15 71.65 55.30 38.14 97.69 64.53 LLaMA-Omni 3.70 3.46 2.92 39.69 25.93 27.47 49.20 14.87 11.35 41.12 Mini-Omni 1.95 2.02 1.61 13.92 24.69 26.59 46.30 13.58 37.12 30.42 Mini-Omni2 2.32 2.18 1.79 9.31 24.27 26.59 46.40 11.56 57.50 33.49 SLAM-Omni 1.90 1.79 1.60 4.16 26.06 25.27 48.80 13.38 94.23 35.30

Ex-Omni 4.31 3.82 3.49 50.71 46.03 56.70 61.10 55.72 87.12 65.53

- Table 5: Speech response quality and speech-text consistency (S-T Consis.) comparison on AlpacaEval and CommonEval. MOS is computed on generated speech, while S-T Consis. is measured by WER, obtained by transcribing the generated audio with Whisper-LargeV3 and comparing it against the corresponding textual response. ↑ means higher is better while ↓ means lower is better.

bones, including Qwen2.5-Omni-3B, Qwen2.5Omni-7B, and Ex-Omni, still exhibit relatively similar performance. This suggests that in cascaded schemes the overall S2F quality is primarily determined by the downstream task-specific model rather than by the upstream OLLM backbone. In contrast, Ex-Omni benefits from native S2F generation, where facial animation and speech are generated jointly within a single framework. This design avoids potential information loss introduced by intermediate speech generation and leads to more synchronized and stable facial animation generation. We further evaluate T2F generation using the same evaluation protocol, with textual input as the only difference, and observe consistent trends across all benchmarks.

AlpacaEval CommonEval MOS ↑ S-T Consis. ↓ MOS ↑ S-T Consis. ↓

Method

Qwen2.5-Omni-3B 4.174 33.53 4.495 20.10 Qwen2.5-Omni-7B 4.502 55.11 4.525 21.72

Ex-Omni 4.523 34.22 4.491 3.54

#### 5.3 Experiments Results and Analysis

3D Facial Animation Generation Results. As shown in Table 3, compared with cascaded baselines that combine omni backbones with external blendshape decoders (e.g., EmoTalk and UniTalker), Ex-Omni achieves better audio-visual synchronization according to the independently computed SyncNet metrics, demonstrating the effectiveness of directly generating facial animation within a unified framework. After adding Qwen2.5Omni-3B to the comparison, we observe the same overall pattern more clearly: under cascaded settings with identical task-specific blendshape decoders, pipelines built upon different OLLM back-

Speech QA Results. As shown in Table 4, ExOmni obtains the second-best overall score (65.53) on VoiceBench, behind Qwen2.5-Omni-7B. Notably, Ex-Omni surpasses Qwen2.5-Omni-7B on several subsets, including WildVoice, BBH, and IFEval, and achieves competitive results on AlpacaEval, CommonEval, and SD-QA. This is achieved with only 200K dialogue-based S2SF samples in Stage III, corresponding to 782.73 hours of prompt speech and 1434.67 hours of target response speech, which is substantially smaller than

the billion-scale training data used by Qwen2.5Omni. Performance on MMSU and OBQA remains behind the strongest baseline, suggesting that speech-based multiple-choice reasoning still requires broader supervision. Overall, these results indicate that Ex-Omni preserves strong speech QA capability while introducing native speech-face generation, demonstrating a favorable performancedata efficiency trade-off.

Speech Generation Results. Table 5 reports speech response quality and speech-text consistency on the AlpacaEval and CommonEval speech QA sets. Unlike conventional TTS evaluation, this setting measures whether an omni-modal model can produce high-quality spoken responses that remain faithful to its own textual outputs under openended QA generation. We therefore report MOS for perceptual speech quality, and use WER as the S-T Consis. metric by transcribing the generated audio with Whisper-Large-V3 and comparing it with the corresponding text response. We observe that WER can become very high for a subset of long-form responses, especially when the generated speech exceeds roughly 60 seconds. This effect is particularly evident on CommonEval, where Qwen2.5-Omni often produces substantially longer textual responses; in some extreme cases, the corresponding speech responses last up to about 90 seconds. Although these long textual responses can be semantically accurate, their excessive length places a much heavier burden on the speech generation module, making the generated speech more likely to be truncated or to become inaccurate in later segments. This leads to poor speech-text consistency despite correct text answers, explaining why Ex-Omni achieves a much lower S-T Consis. WER on CommonEval. We believe this behavior mainly stems from the limited speech modeling capacity of the small speech generation backbones, namely the Qwen3-0.6B blendshape-aware speech unit generator used in Ex-Omni and the 0.5B talker used in Qwen2.5-Omni.

- Table 6: Effect of each component on facial animation generation on AlpacaEval. ↓ means lower is better.

Method Sync-D↓ / Sync-C↑ QA Score↑ MOS↑

Ex-Omni 9.233 / 5.385 4.31 4.523 w/o TQGF (only speech tokens) 9.591 / 4.826 4.29 4.161 w/o TQGF (only hidden features) 9.475 / 4.768 4.23 3.968 w/o TQGF (concatenation) 10.209 / 4.043 4.27 4.136

Ablation Study on Facial Animation Generation. Table 6 presents the impact of TQGF on facial animation generation and response quality. The full Ex-Omni model achieves the best results across all metrics, with the lowest Sync-D (9.233), the highest Sync-C (5.385), and the best QA score and MOS. When TQGF is removed and the decoder relies only on speech tokens, lip synchronization degrades to 9.591 / 4.826, although the QA score remains close to the full model. This indicates that speech units provide useful temporal scaffolding, but are insufficient to recover all fine-grained semantic-temporal cues needed for facial animation. Here, hidden features refer to the hidden representations produced by the blendshape-aware speech unit generator, rather than the LLM hidden states. Using only these speech-generator hidden features also hurts both synchronization and response quality, suggesting that contextual representations alone do not provide a stable temporal structure for frame-level blendshape prediction. The concatenation variant performs worst on synchronization (10.209 / 4.043), showing that simply merging speech-token and hidden representations is less effective than the proposed token-as-query gated fusion. These results demonstrate that TQGF provides a more reliable interface between discrete speech units and contextual speech representations by selectively injecting semantic information while preserving the temporal scaffold required for robust speech-to-face generation.

Latency Analysis. Table 7 reports latency on 100 randomly sampled CommonEval instances using three metrics: Overall RTF, Avg Speech TTFT, and Avg Face Latency. Since all compared systems share the same speech generation backbone, they have identical Overall RTF (2.158) and Avg Speech TTFT (0.029s). The main difference lies in face generation latency: native Ex-Omni requires only 0.012s, while cascaded pipelines with task-specific S2F models require 0.105–0.117s. This indicates that directly predicting facial animation inside the unified framework is more efficient than invoking a separate downstream S2F model. The overall RTF is still above real time under the tested NVIDIA H20 GPU, suggesting that the main bottleneck remains the Qwen3-8B semantic reasoning backbone rather than the facial animation branch.

Case Study of 3D Facial Animation Generation. We provide qualitative examples in Appendix Figure 2, where three representative cases are ren-

- Table 7: Latency comparison between native Ex-Omni and cascaded speech-to-face pipelines. ↓ means lower is better.

Avg Face Latency (s) ↓ Ex-Omni

Avg Speech TTFT (s) ↓

Model Overall RTF ↓

0.012 Ex-Omni + EmoTalk 0.110 Ex-Omni + Unitalker-B-D3 0.105 Ex-Omni + Unitalker-B-D6 0.117

2.158 0.029

dered with two different templates: the NVIDIAprovided Claire template and a commercially purchased template with realistic skin and hair materials. We compare native Ex-Omni with three cascaded variants, namely Ex-Omni+EmoTalk, ExOmni+UniTalker-B-D3, and Ex-Omni+UniTalkerB-D6. To isolate the effect of the facial animation module, all cascaded variants use the same text and speech responses generated by Ex-Omni as input, ensuring that the linguistic content, speech duration, and acoustic realization are controlled across methods. Therefore, the visual differences mainly reflect the quality of the predicted facial animation rather than differences in upstream response generation. As shown in the figure, Ex-Omni captures finer articulatory details more consistently than the cascaded alternatives. In Case 1, when pronouncing “quote”, the mouth generated by Ex-Omni moves toward a rounded shape, while the other methods fail to capture this phoneme-related transition. In Case 2, after the speech ends, Ex-Omni closes the mouth naturally, whereas the cascaded methods tend to leave the mouth partially open. In Case 3, both Ex-Omni and Ex-Omni+UniTalker-BD6 produce a large mouth opening for the pronunciation of “hard”, while the other cascaded variants miss this articulation cue. These differences are more evident in the supplementary videos. Overall, Ex-Omni produces more detailed and stable mouth dynamics while maintaining lip-speech alignment, suggesting that jointly modeling speech units and blendshape generation helps preserve fine-grained temporal cues that may be weakened when facial animation is generated by a separate downstream model.

### 6 Conclusion

In this paper, we introduced Ex-Omni, an opensource framework that extends OLLMs with native speech-accompanied 3D facial animation generation. To address the mismatch between token-level semantic reasoning and fine-grained facial motion,

Ex-Omni decouples high-level understanding from modality-specific temporal synthesis through discrete speech-unit scaffolding and a unified tokenas-query gated fusion mechanism. We further constructed InstructS2SF-1200K to provide large-scale speech-face supervision for both synthesis and dialogue-oriented scenarios. Experiments show that Ex-Omni preserves competitive speech understanding and generation ability while achieving better SyncNet-based audio-visual synchronization and lower face-generation latency than cascaded pipelines.

### References

Inclusion AI, Biao Gong, Cheng Zou, Chuanyang Zheng, Chunluan Zhou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng, Fudong Wang, Furong Xu, Guangming Yao, Jun Zhou, Jingdong Chen, Jianxin Sun, Jiajia Liu, Jianjiang Zhu, Jun Peng, Kaixiang Ji, and 39 others. 2025. Ming-omni: A unified multimodal model for perception and generation. CoRR, abs/2506.09344.

Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, and Furu Wei. 2022. Speecht5: Unified-modal encoder-decoder pretraining for spoken language processing. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, pages 5723–5738.

Lele Chen, Ross K. Maddox, Zhiyao Duan, and Chenliang Xu. 2019. Hierarchical cross-modal talking face generation with dynamic pixel-wise loss. In IEEE Conference on Computer Vision and Pattern Recognition, pages 7832–7841.

Qian Chen, Yafeng Chen, Yanni Chen, Mengzhe Chen, Yingda Chen, Chong Deng, Zhihao Du, Ruize Gao, Changfeng Gao, Zhifu Gao, Yabin Li, Xiang Lv, Jiaqing Liu, Haoneng Luo, Bin Ma, Chongjia Ni, Xian Shi, Jialong Tang, Hui Wang, and 17 others. 2025a. Minmo: A multimodal large language model for seamless voice interaction. CoRR, abs/2501.06282.

Wenxi Chen, Ziyang Ma, Ruiqi Yan, Yuzhe Liang, Xiquan Li, Ruiyang Xu, Zhikang Niu, Yanqiao Zhu, Yifan Yang, Zhanxun Liu, Kai Yu, Yuxuan Hu, Jinyu Li, Yan Lu, Shujie Liu, and Xie Chen. 2025b. Slamomni: Timbre-controllable voice interaction system with single-stage training. In Findings of the Association for Computational Linguistics, pages 2262– 2282.

Wenxi Chen, Ziyang Ma, Ruiqi Yan, Yuzhe Liang, Xiquan Li, Ruiyang Xu, Zhikang Niu, Yanqiao Zhu, Yifan Yang, Zhanxun Liu, and 1 others. 2025c. Slamomni: Timbre-controllable voice interaction system with single-stage training. In Findings of the Association for Computational Linguistics: ACL 2025, pages 2262–2282.

Yiming Chen, Xianghu Yue, Chen Zhang, Xiaoxue Gao, Robby T. Tan, and Haizhou Li. 2024. Voicebench: Benchmarking llm-based voice assistants. CoRR, abs/2410.17196.

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, and Jingren Zhou. 2024. Qwen2-audio technical report. CoRR, abs/2407.10759.

Chaeyeon Chung, Ilya Fedorov, Michael Huang, Aleksey Karmanov, Dmitry Korobchenko, Roger Blanco i Ribera, and Yeongho Seol. 2025. Audio2face-3d: Audio-driven realistic facial animation for digital avatars. CoRR, abs/2508.16401.

J. S. Chung and A. Zisserman. 2016. Out of time: automated lip sync in the wild. In Workshop on Multiview Lip-reading, ACCV.

Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. 2024. Moshi: a speechtext foundation model for real-time dialogue. CoRR, abs/2410.00037.

Xiangyu Fan, Jiaqi Li, Zhiqian Lin, Weiye Xiao, and Lei Yang. 2024. Unitalker: Scaling up audio-driven 3d facial animation through A unified model. In Computer Vision - ECCV 2024 - 18th European Conference, volume 15099, pages 204–221.

Yingruo Fan, Zhaojiang Lin, Jun Saito, Wenping Wang, and Taku Komura. 2022. Faceformer: Speech-driven 3d facial animation with transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18749–18758.

Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, and Yang Feng. 2025a. Llama-omni: Seamless speech interaction with large language models. In The Thirteenth International Conference on Learning Representations. OpenReview.net.

Qingkai Fang, Yan Zhou, Shoutao Guo, Shaolei Zhang, and Yang Feng. 2025b. Llama-omni 2: Llm-based real-time spoken chatbot with autoregressive streaming speech synthesis. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, pages 18617–18629. Association for Computational Linguistics.

Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, Ran He, Rongrong Ji, Yunsheng Wu, Caifeng Shan, and Xing Sun. 2024. VITA: towards open-source interactive omni multimodal LLM. CoRR, abs/2408.05211.

Chaoyou Fu, Haojia Lin, Xiong Wang, Yifan Zhang, Yunhang Shen, Xiaoyu Liu, Haoyu Cao, Zuwei Long, Heting Gao, Ke Li, Long Ma, Xiawu Zheng, Rongrong Ji, Xing Sun, Caifeng Shan, and Ran He. 2025. VITA-1.5: towards gpt-4o level real-time vision and speech interaction. CoRR, abs/2501.01957.

Michael Hassid, Tal Remez, Tu Anh Nguyen, Itai Gat, Alexis Conneau, Felix Kreuk, Jade Copet, Alexandre Défossez, Gabriel Synnaeve, Emmanuel Dupoux, Roy Schwartz, and Yossi Adi. 2023. Textually pretrained speech language models. In The ThirtySeventh Annual Conference on Neural Information Processing Systems.

Haorui He, Zengqiang Shang, Chaoren Wang, Xuyuan Li, Yicheng Gu, Hua Hua, Liwei Liu, Chen Yang, Jiaqi Li, Peiyang Shi, Yuancheng Wang, Kai Chen, Pengyuan Zhang, and Zhizheng Wu. 2024. Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation. In Proc. of SLT.

Haorui He, Zengqiang Shang, Chaoren Wang, Xuyuan Li, Yicheng Gu, Hua Hua, Liwei Liu, Chen Yang, Jiaqi Li, Peiyang Shi, Yuancheng Wang, Kai Chen, Pengyuan Zhang, and Zhizheng Wu. 2025. Emilia: A large-scale, extensive, multilingual, and diverse dataset for speech generation. In arXiv:2501.15907.

Fa-Ting Hong, Longhao Zhang, Li Shen, and Dan Xu. 2022. Depth-aware generative adversarial network for talking head video generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3387–3396.

Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, Xiong Wang, Zhifang Guo, Ziyue Jiang, Hongkun Hao, Zishan Guo, and 1 others. 2026. Qwen3-tts technical report. arXiv preprint arXiv:2601.15621.

John P. Lewis, Ken Anjyo, Taehyun Rhee, Mengjie Zhang, Frédéric H. Pighin, and Zhigang Deng. 2014. Practice and theory of blendshape facial models. In 35th Annual Conference of the European Association for Computer Graphics, pages 199–218.

Yunxin Li, Xinyu Chen, Shenyuan Jiang, Haoyuan Shi, Zhenyu Liu, Xuanyu Zhang, Nanhao Deng, Zhenran Xu, Yicheng Ma, Meishan Zhang, and 1 others. 2025. Uni-moe-2.0-omni: Scaling language-centric omnimodal large model with advanced moe, training and data. arXiv preprint arXiv:2511.12609.

Run Luo, Ting-En Lin, Haonan Zhang, Yuchuan Wu, Xiong Liu, Yongbin Li, Longze Chen, Jiaming Li, Lei Zhang, Xiaobo Xia, Hamid Alinejad-Rokny, Fei Huang, and Min Yang. 2025. Openomni: Advancing open-source omnimodal large language models with progressive multimodal alignment and real-time emotional speech synthesis. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. 2024. Large language models: A survey. arXiv preprint arXiv:2402.06196.

Gaurav Mittal and Baoyuan Wang. 2020. Animating face using disentangled audio representations. In IEEE Winter Conference on Applications of Computer Vision, pages 3279–3287.

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. 2015. Librispeech: An ASR corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2015, pages 5206– 5210.

Ziqiao Peng, Yanbo Fan, Haoyu Wu, Xuan Wang, Hongyan Liu, Jun He, and Zhaoxin Fan. 2025. Dualtalk: Dual-speaker interaction for 3d talking head conversations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21055– 21064.

Ziqiao Peng, Yihao Luo, Yue Shi, Hao Xu, Xiangyu Zhu, Hongyan Liu, Jun He, and Zhaoxin Fan. 2023a. Selftalk: A self-supervised commutative training diagram to comprehend 3d talking faces. In Proceedings of the 31st ACM International Conference on Multimedia, pages 5292–5301.

Ziqiao Peng, Haoyu Wu, Zhenbo Song, Hao Xu, Xiangyu Zhu, Jun He, Hongyan Liu, and Zhaoxin Fan. 2023b. Emotalk: Speech-driven emotional disentanglement for 3d face animation. In IEEE/CVF International Conference on Computer Vision, pages 20630–20640.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 28492–28518.

Alexander Richard, Michael Zollhöfer, Yandong Wen, Fernando De la Torre, and Yaser Sheikh. 2021. Meshtalk: 3d face animation from speech using crossmodality disentanglement. In 2021 IEEE/CVF International Conference on Computer Vision, pages 1153–1162.

Meituan LongCat Team. 2025. Longcat-flash-omni technical report. CoRR, abs/2511.00279.

- Zhifei Xie and Changqiao Wu. 2024a. Mini-omni: Language models can hear, talk while thinking in streaming. CoRR, abs/2408.16725.
- Zhifei Xie and Changqiao Wu. 2024b. Mini-omni2: Towards open-source gpt-4o with vision, speech and duplex capabilities. CoRR, abs/2410.11190.

Jinbo Xing, Menghan Xia, Yuechen Zhang, Xiaodong Cun, Jue Wang, and Tien-Tsin Wong. 2023. Codetalker: Speech-driven 3d facial animation with discrete motion prior. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12780–12790.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. 2025. Qwen2.5-omni technical report. CoRR, abs/2503.20215.

Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, and Jie Tang. 2024. Glm-4-voice: Towards intelligent and human-like end-to-end spoken chatbot. CoRR, abs/2412.02612.

Binbin Zhang, Hang Lv, Pengcheng Guo, Qijie Shao, Chao Yang, Lei Xie, Xin Xu, Hui Bu, Xiaoyu Chen, Chenchen Zeng, Di Wu, and Zhendong Peng. 2022. WENETSPEECH: A 10000+ hours multi-domain mandarin corpus for speech recognition. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2022, pages 6182–6186.

Chenxu Zhang, Yifan Zhao, Yifei Huang, Ming Zeng, Saifeng Ni, Madhukar Budagavi, and Xiaohu Guo. 2021. FACIAL: synthesizing dynamic talking face with implicit attribute learning. In 2021 IEEE/CVF International Conference on Computer Vision, pages 3847–3856.

Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, and Xipeng Qiu. 2023. Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities. In Findings of the Association for Computational Linguistics, pages 15757–15773.

Dong Zhang, Xin Zhang, Jun Zhan, Shimin Li, Yaqian Zhou, and Xipeng Qiu. 2024. Speechgpt-gen: Scaling chain-of-information speech generation. CoRR, abs/2401.13527.

[Figure 2]

Figure 2: Case study on 3D facial animation generation. Three representative cases are rendered using both the NVIDIA-provided Claire template and a commercially purchased template with realistic skin and hair materials. All cascaded baselines use the same Ex-Omni-generated text and speech responses as input, so the comparison focuses on facial animation quality.

