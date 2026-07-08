# arXiv:2602.09823v2[cs.SD]16Mar2026

## Covo-Audio Technical Report

#### Tencent Abstract

End-to-end large audio language models (LALMs) have emerged as a promising paradigm for unifying speech processing and language intelligence, yet achieving strong semantic reasoning alongside high-quality audio interaction remains challenging. In this work, we present Covo-Audio, a 7B-parameter end-to-end LALM that directly processes continuous audio inputs and generates audio outputs within a single unified architecture. Through large-scale curated pretraining and targeted post-training, Covo-Audio achieves state-of-the-art or competitive performance among models of comparable scale across a broad spectrum of tasks, including speech-text modeling, spoken dialogue, speech understanding, audio understanding, and full-duplex voice interaction. Extensive evaluations demonstrate that the pretrained foundation model exhibits strong speech–text comprehension and semantic reasoning capabilities on multiple benchmarks, outperforming representative open-source models of comparable scale. Furthermore, Covo-Audio-Chat, the dialogue-oriented variant, demonstrates strong spoken conversational abilities, including understanding, contextual reasoning, instruction following, and generating contextually appropriate and empathetic responses, validating its applicability to real-world conversational assistant scenarios. Covo-Audio-Chat-FD, the evolved full-duplex model, achieves substantially superior performance on both spoken dialogue capabilities and full-duplex interaction behaviors, demonstrating its competence in practical robustness. To mitigate the high cost of deploying end-to-end LALMs for natural conversational systems, we propose an intelligence–speaker decoupling strategy that separates dialogue intelligence from voice rendering, enabling flexible voice customization with minimal text-to-speech (TTS) data while preserving dialogue performance. Overall, our results highlight the strong potential of 7B-scale models to integrate sophisticated audio intelligence with high-level semantic reasoning, and suggest a scalable path toward more capable and versatile LALMs. We open-source Covo-Audio-Chat model and the inference pipeline.

GitHub: https://github.com/Tencent/Covo-Audio

HuggingFace: https://huggingface.co/tencent/Covo-Audio-Chat

#### 1 Introduction

Speech interaction, as the most natural and efficient form of human communication, is driving artificial intelligence toward more human-like conversational systems. In this context, the ideal conversational agent must harmonize intelligence (possessing deep linguistic understanding and reasoning), naturalness (preserving paralinguistic and emotional cues), and efficiency (enabling low-latency, fluent full-duplex voice interaction). However, current approaches often force compromises among these goals.

Traditional speech interaction systems predominantly rely on cascaded architectures, which combine independent modules for Automatic Speech Recognition (ASR), Large Language Model (LLM), and Text-to-Speech (TTS). While this modular paradigm offers interpretability and controllability, it suffers from the inherent issues such as information loss and error propagation, hindering real conversational

###### Pre-training

###### Speech-to-Speech Spoken Dialogue

Covo-Audio AlignSLM

SIMS Moshi GLM-4-Voice-Base Step-Audio-2-mini-Base Qwen2.5-7B-Base

Covo-Audio GPT-4o Audio

Step-Audio 2 mini Fun-Audio-Chat Qwen3-Omni GPT-4o Mini Audio Doubao

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

99.40

95.50 95.70 98.60 98.00 98.50

94.00 93.60

86.80 88.30

###### 83.51

83.00 83.30

80.98

78.11 79.59 78.69 77.26

76.36

4.75 4.81 4.82 4.89

68.83

A2A-tSC A2T-tSC T2T-tSC URO-Bench VCB-Bench VStyle

###### Audio Understanding

###### Speech Understanding

###### Full-Duplex

Covo-Audio Audio Flamingo 3

MiMo-Audio Step-Audio 2

Kimi-Audio Qwen2.5-Omni

Covo-Audio MinMo Step-Audio 2 mini

LongCat-Flash-Omni Qwen2.5-Omni

Qwen3-Omni Gemini 2.5 pro

Covo-Audio Moshi Freeze-Omni

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

###### 99.10 99.70

97.60

96.80

| |
|---|

| | | | |
|---|---|---|---|
| | | | |
| | | | |

80.86

79.03

###### 77.58 75.30

73.30 74.70

|27.23 27.64| | | |
|---|---|---|---|
| | | | |
| | | | |

71.98 71.15

69.80

66.64

61.40

59.28 60.57

53.20

51.20

39.30

37.28 38.31

36.35

| | | | |
|---|---|---|---|
| | | | |

MMAU MMSU CoVoST2 AIR-bench URO-Bench (en) Turn Taking Pause Handling

Figure 1: An Overview of Comprehensive Performance Comparison.

experience. Recent large audio language models (LALMs) such as Qwen2.5-Omni (Xu et al., 2025a) and Qwen3-Omni (Xu et al., 2025b) adopt the Thinker-Talker architecture, where an intermediate textual reasoning step (the Thinker) precedes acoustic token prediction (the Talker). While this improves textual intelligence preservation, it sacrifices end-to-end speech instruction following abilities and direct conversational controllability. Additionally, handling full-duplex dynamics in such a sequential generation paradigm is more challenging.

Human voice interaction derives its richness from a dynamic blend of flexibility and diversity. End-toend LALM, aiming to map raw audio input directly to audio output within a single, unified model, represents a promising frontier. Pioneering works like GPT-4o (Hurst et al., 2024), GLM-4-Voice (Zeng et al., 2024) and Step-Audio 2 (Wu et al., 2025a) have demonstrated the feasibility of this approach by augmenting LLMs with discrete audio tokens, showcasing the profound potential that enables lowlatency, contextually consistent, and emotionally empathetic interaction. Architecturally in terms of decoding text-speech tokens, end-to-end models can be categorized into generating either interleaved streams (Wu et al., 2025a; Xiaomi, 2025) or parallel streams (Ding et al., 2025; Chen et al., 2025b; Team, 2025). We adopt the former as our backbone text-speech decoding design since it is more adherent to the oracle LLM philosophy. However, a common pain point accompanied with this unified approach is the deep intelligence-speaker coupling problem when developing a production-level application, which brings challenges in data preparation and deteriorates flexible voice customization. Building a chat bot combining virtues of fascinating voice and high-intelligence is rather laborious, since it is required to gather much high-quality dialogue data for the desired speaker1. In this work, to mitigate this issue, we propose a novel intelligence-speaker decoupling technique, allowing us to build conversational capabilities directly from genuine TTS data, thereby avoiding the process of constructing lots of elaborate dialogue data.

1Commonly, we utilize advanced TTS engines to synthesize dialogue data, which inevitably lowers the upper bound of voice expressiveness.

Another emerging research direction is end-to-end full-duplex LALMs, which facilitate human-like interaction with low latency and complex behaviors inherent to natural conversations. Representative works are Moshi (Défossez et al., 2024) and Voila (Shi et al., 2025), which typically employ a synchronous dual-stream scheme, where both input and output streams are in discrete token sequences, to achieve full-duplex. However, they still require word-level text-speech alignment to generate coherent speech. OmniFlatten (Zhang et al., 2025c) adapts a text LLM backbone into a robust dialogue model through a curated multi-stage post-training. In this work, we put full-duplex interaction, without requirements of fine-grained text-speech alignment, directly into the pre-training phase to acquire robust full-duplex conversational capabilities via large-scale pre-training. Moreover, differing from other full-duplex models, we adopt a hybrid dual-stream scheme (i.e., continuous input stream and discrete output stream) in accordance with our half-duplex paradigm, which provides a more efficient and lossless way to perceive user expression and intention.

To facilitate these capabilities in an end-to-end paradigm, we present Covo-Audio, a compact LALM that achieves the fundamental comprehensive advantages of intelligence, naturalness, and efficiency that a voice conversational assistant should be empowered. Rather than focusing on a single task or setting, we demonstrate how pre-training and post-training strategies can endow an end-to-end LALM with robust audio perception, reasoning, emotional empathy, while economical flexible voice interaction capabilities. We conduct comprehensive evaluations of Covo-Audio across a broad range of tasks, including speech–text cross-modal alignments, speech understanding, audio question answering, and both half-duplex (Covo-Audio-Chat) and full-duplex (Covo-Audio-Chat-FD) speech-to-speech spoken dialogue. The results show that Covo-Audio achieves state-of-the-art (SOTA) or competitive performance among models of comparable scale. Our primary contributions are summarized as follows:

Hierarchical Tri-modal Speech-Text Interleaving: A framework designed to achieve deep alignment and fusion across modalities and scales. The Tri-modal aspect integrates continuous acoustic features, discrete speech tokens, and natural language text within a unified sequence, effectively bridging the gap between high-fidelity prosodic nuances and robust semantic structures. Hierarchical strategy aims to ensure structural coherence: phrase-level interleaving facilitates fine-grained alignment between acoustic fragments and lexical units, while sentence-level interleaving preserves the global semantic integrity and prosodic flow of long-form utterances. This dual-scale approach ensures the model captures precise acoustic nuances while maintaining macro-level coherence, allowing it to conform naturally to the LLM’s generative patterns.

Mitigating Intelligence-Speaker Coupling: We propose a intelligence-speaker decoupling technique that decouples speaker from dialogue intelligence via multi-speaker training, then develop a contextual adaptation method to transfer and share high-quality TTS voices. By reformatting TTS recordings into pseudo-conversations with masked text loss, we preserve reasoning abilities while achieving highfidelity voice naturalness, enabling versatile and personalized interaction in a more economical way.

Native Full-Duplex Voice Interaction: We evolve Covo-Audio-Chat into Covo-Audio-Chat-FD, a variant with low-latency, fluent full-duplex capability. In a more native way, we put full-duplex interaction directly into the pre-training phase to acquire full-duplex conversation capabilities via largescale pre-training. Through large-scale pre-training and elaborate lightweight supervised fine-tuning, it supports robust real-time conversational dynamics, including smooth turn-taking, pause handling, user interruption (barge-in), and backchanneling, while maintaining competitive performance with the half-duplex model.

Comprehensive State-of-the-Art Performance: Covo-Audio delivers consistently strong and competitive results across core speech and audio tasks, including speech understanding, audio question answering, and both half-duplex and full-duplex speech-to-speech dialogue, despite its compact 7B-

[Figure 5]

[Figure 6]

Prompt Token

Continuous Audio Feature

Speaker Token

Discrete Audio Token

Speech Decoder

Regular Text Token

Prompt Audio Feature

Start Token

Large Language Model

Adapter

Speech Encoder

[Figure 7]

[Figure 8]

Figure 2: An Overview of Covo-Audio.

parameter footprint. It substantially outperforms existing models of comparable size and, notably, even matches the performance of significantly larger systems on multiple key metrics. This demonstrates that careful co-design of architecture, multi-stage pre-training, and alignment-focused post-training enables high-fidelity and robust speech interaction without parameter bloat, providing a practical solution for efficient deployment.

#### 2 Methodology

##### 2.1 Architecture

Covo-Audio employs a well-designed end-to-end architecture designed for seamless cross-modal interaction, as illustrated in Figure 2. The model consists of four primary components: an audio encoder, a robust LLM backbone, a specialized speech tokenizer, and a high-fidelity speech decoder.

We utilize the pre-trained Whisper-large-v3 (Radford et al., 2023) as our primary audio encoder. Whisper stands out for its exceptional robustness to background noise and varied accents, having been trained on a massive, weakly supervised multilingual dataset, and also guaranteeing a versatile, generalpurpose ability of performing speech understanding. The output frame rate of the audio encoder is 50 Hz. An audio adapter is employed to bridge the audio encoder and the LLM. This adapter consists of three downsampling modules, each of which integrates two linear layers and a convolution layer with a downsampling rate of 2. Through this architecture, the adapter effectively reduces the frame rate from 50 Hz to 6.25 Hz.

To facilitate audio generation via the LLM, we developed a specialized speech tokenizer based on the WavLM-large (Chen et al., 2022), augmented with a single Vector Quantization (VQ) layer (Van Den Oord et al., 2017). The tokenizer utilizes a codebook size of 16,384, producing discrete audio tokens at a rate of 25 Hz to ensure a high-compression yet information-rich representation. To achieve acoustic-semantic alignment, the tokenizer is pre-trained on a massive multilingual dataset. The training objective is guided by a multi-task loss function, which incorporates ASR loss for semantic ground-

Table 1: List of pre-training tasks. 𝑥→𝑦 denotes 𝑥 is query and 𝑦 response, 𝑥|𝑦 denotes paired queries or responses. 𝑎𝑐, 𝑎𝑑, 𝑡 denote continuous audio, discrete audio, and text, respectively. The trained tokens are rounded for ease of presentation.

Training Stage Training Module Task Formula Trained Tokens

- Stage 1 Adapter ASR 𝑎𝑐→𝑡 30B

- Stage 2 Adapter+LLM

ASR 𝑎𝑐→𝑡 80B TTS 𝑡→𝑎𝑑 160B Audio-only 𝑎𝑐→𝑎𝑑, 𝑎𝑑→𝑎𝑑 240B Speech Continuation 𝑎𝑐→𝑡 160B

Speech-Text Interleave 𝑎𝑐→𝑡|𝑎𝑑, 𝑎𝑐→𝑡→𝑎𝑐, 360B

𝑎𝑑→𝑡→𝑎𝑑, 𝑎𝑐→𝑡→𝑎𝑑 180B Text-only 𝑡→𝑡 800B Full-Duplex 𝑎𝑐→𝑡|𝑎𝑑 5B

ing, TTS reconstruction loss for acoustic fidelity, and a specialized pitch loss to preserve the essential prosodic and intonational nuances of human speech.

The speech decoder employs a two-stage hierarchical decoding framework to reconstruct high-fidelity audio from discrete tokens (Zhang et al., 2025a). A Flow-Matching (FM) (Du et al., 2024) based decoder first transforms the discrete speech tokens into continuous acoustic-rich latent representations. Subsequently, these latents are fed into a BigVGAN (Lee et al., 2022) based vocoder to reconstruct 24K waveforms.

Covo-Audio is built upon the Qwen2.5-7B-Base (Yang et al., 2024a), which we adapt for end-to-end speech interaction by extending audio encoder for perception, and by expanding its original text vocabulary with discrete audio tokens for generation. The architecture is designed to natively process interleaved inputs, consisting of continuous acoustic features (processed via an audio encoder and adapter) and textual tokens, while generating a unified sequence of text and audio tokens as output. For multi-turn conversational scenarios, we implement a recursive context-filling strategy where the continuous audio features from the user’s input, along with the model’s generated text and audio tokens from previous turns, are prefixed to the current input as historical context.

##### 2.2 Pre-training

Covo-Audio is initialized from Qwen2.5-7B-Base and undergoes a rigorous two-stage pre-training pipeline, processing a cumulative total of 2T tokens. This process is specifically engineered to fuse speech and text modalities within the model’s latent space, evolving it from a text-centric LLM into a unified multi-modal engine capable of seamless cross-modal understanding and generation

- Stage 1: Modality Bridging via ASR Alignment. This stage aims to establish a functional bridge between the audio encoder and the LLM, achieving initial semantic alignment between speech and text modalities. In this phase, we keep both the audio encoder and the LLM backbone frozen, focusing exclusively on optimizing the audio adapter. This alignment is supervised using a massive 200,000hour multilingual ASR dataset, which forces the adapter to map continuous acoustic features into the LLM’s latent space effectively. The training is conducted over 50k steps, and employs a cosine learning rate scheduler with a peak learning rate of 1 × 10−4 and 40k steps for warm-up, which ensures stable convergence. By the end of this stage, the model develops a robust foundational ability to “understand” spoken language, providing a well-aligned multimodal representation for subsequent pre-training.
- Stage 2: Speech-Text Modality Fusion. This stage is specifically designed to achieve superior

Table 2: List of post-training tasks.

Task Ratio Formula General Intelligence 0.4 𝑡→𝑡, 𝑡→𝑎𝑑, 𝑎𝑐→𝑡, 𝑎𝑐→𝑡|𝑎𝑑 Spoken Dialogue 0.3 𝑎𝑐→𝑡|𝑎𝑑 Speech Understanding 0.1 𝑎𝑐→𝑡, 𝑎𝑐|𝑡→𝑡, 𝑎𝑐|𝑎𝑐→𝑡, 𝑎𝑐|𝑎𝑐→𝑡|𝑎𝑑 Speech Generation 0.1 𝑡→𝑎𝑑 Audio Understanding 0.1 𝑡|𝑎𝑐→𝑡

cross-modal alignment between the speech and text domains, moving beyond basic feature mapping to deep semantic and acoustic synchronization. During this phase, both the audio adapter and the LLM backbone are jointly unfrozen and optimized to refine the model’s unified representation space. We implement a diverse multi-task training objective that includes ASR, TTS, Audio-only modeling, Speech Continuation, and Speech-Text Interleave, alongside Text-only to preserve foundational linguistic knowledge. Additionally, to acquire robust full-duplex interaction capabilities, we also include Full-Duplex pre-training in this stage. The entire pre-training tasks are listed in Table 1.

A cornerstone of our pre-training phase is the Hierarchical Tri-modal Speech-text Interleaving strategy. Our framework acknowledges the dual nature of speech modeling: continuous acoustic representations 𝑎𝑐 derived from the Whisper encoder and discrete acoustic tokens 𝑎𝑑 generated by the speech tokenizer. Consequently, our pre-training objective is not merely to align speech with text, but to achieve a three-way fusion between continuous features, discrete tokens, and linguistic text within a unified latent space. In detail, we designed several tri-modal sequence structures. Specifically, we employ two primary patterns: (1) Sequential Interleaving (𝑎𝑐→𝑡→𝑎𝑑): Arranging continuous features (𝑎𝑐), text (𝑡), and discrete tokens (𝑎𝑑) in a progressive chain; (2) Parallel Integration (𝑎𝑐→𝑡|𝑎𝑑): Aligning continuous features with a coupled text-discrete unit. These paradigms ensure the model learns to map high-fidelity acoustic nuances (continuous) to robust semantic units (discrete and text) simultaneously. For Hierarchical Interleaving, previous interleaving methods, such as those utilized in GLM-4-Voice, primarily operate at the word or character level. While effective for local modality alignment, such granular interleaving often sacrifices the semantic integrity of long-form utterances and is highly sensitive to tokenization or forced-alignment errors. To overcome these challenges, our approach introduces a multi-scale interleaving mechanism that incorporates utterance-level paradigms alongside phrase-level segments. By interleaving longer, semantically complete speech and text fragments, we preserve the global context and natural flow of language.

The training is conducted using a sequence length of 8192 and a cosine learning rate scheduler with a peak learning rate of 3 × 10−5, spanning a total of 500k steps and 20k steps for warm-up. Our pretraining data consists of approximately 8 million hours of diverse audio and speech, and a text corpus encompassing multiple domains, totaling 3T tokens. This comprehensive approach ensures that the model achieves high-fidelity modal consistency, enabling seamless and accurate transitions between spoken and written language.

##### 2.3 Speech-to-Speech Spoken Dialogue

To incentivize strong intelligence, natural spoken expression, and empathetic interaction capabilities, we design a collaborative data construction and training strategy that integrates text instruction tuning, audio adaptation, and spoken dialogue alignment, ultimately obtaining the spoken chat model, Covo-Audio-Chat. Given the fundamental importance of the core intelligence and logical reasoning capabilities of Covo-Audio-Chat, it is essential to incorporate high-quality instruction data used in text LLM domain. Specifically, to transfer intelligence to the audio modality, we design multiple cross-

modal tasks including text-to-text (T2T), text-to-audio (T2A), audio-to-text (A2T) and audio-to-audio (A2A) tasks. Note that in the A2A task, the query typically consists of audio input only, while the answer includes both text response and the corresponding audio in an interleaved format. The involved post-training tasks are listed in Table 2.

Intelligence We start with a 10M internal text instruction data for T2T training. This corpus is balanced across diverse domains, including commonsense reasoning, mathematics, coding, and specialized knowledge. A randomly sampled subset is then converted into parallel speech data via a highfidelity neural TTS system, enabling T2A, A2T and A2A training. Speaker diversity is ensured by randomly sampling voice styles from thousands of distinct speech prompts, followed by WER-based filtering to maintain speech quality. This parallel text–audio construction paradigm facilitates effective knowledgetransferfromT2Ttoaudio-relatedtasks, andfinallymitigatesintelligencedegradation when extending the model to the speech modality. Additionally, we apply Kullback-Leibler (KL) based onpolicy model distillation during T2T training, where a stronger text-based LLM provides top-20 logits as soft targets to prevent intelligence degradation.

Colloquialism Beyond intelligence, spoken dialogue requires colloquialism and conciseness. To enhance colloquial expressiveness, we select a moderate-scale set of dialogue samples suitable for spoken interaction and rewrite assistant responses into a conversational style using a text-based LLM before TTS synthesis. Distinct system prompts are employed during training to explicitly differentiate colloquial and formal response styles, enabling flexible spoken expression without sacrificing general capabilities.

EmpathyTomodelempatheticspeechinteraction, wefurtherconstructemotion-awaredialoguedatasets

spanning seven emotion categories: joy, anger, sadness, fear, disgust, depression, and surprise. Each dialogue pairs explicit user emotional cues with assistant responses exhibiting active listening and context-sensitive support. These dialogues are synthesized with emotionally aligned speech, allowing Covo-Audio-Chat to jointly learn semantic and acoustic nuances for empathetic response generation.

The training is conducted using a sequence length of 8192 and a cosine learning rate scheduler with a peak learning rate of 1×10−5, spanning a total of 50k steps and 1,000 steps for warm-up. The involved tasks with respective training ratio are listed in Table 2. This recipe endows Covo-Audio-Chat with strong logical reasoning and natural spoken dialogue capabilities.

##### 2.4 Intelligence-Speaker Decoupling Technique

Building a versatile spoken dialogue model is rather challenging. There are two critical issues in endto-end scenarios. (1) Due to the scarcity of authentic high-quality2 spoken dialogue data, advanced TTS systems are often used to synthesize such data based on text dialogue datasets, not only inducing a prohibitive cost for flexible voice customization but also restricting the overall naturalness to the limitation of the TTS system. (2) On the other hand, high-quality TTS training data is relatively easy to collect but hard to exploit as it is originally not designed for dialogue-style interaction. A common practice is to construct conversational context manually based on such data, but often resulting in poor quality and degradation of overall dialogue performance. To address these two issues, during dialogue data preparation, we first randomly generated thousands of speakers to decouple speaker characteristics from intelligence, ensuring consistent intellectual levels across different speakers. However, when applying speakers trained solely for TTS tasks to dialogue tasks, we observed a significant decline in performance and reduced naturalness compared to TTS tasks. This drop is essentially due to the substantial format difference between TTS and dialogue tasks, making it difficult for models to di-

2That means of high intelligence and human-like.

|<br><br>对话大模型|
|---|

[Figure 13]

Covo-Audio Technical Report

###### Training

unmasked loss

- prompt | speaker A | query | text response |

|speech response|
|---|

high-intelligence dialogue data

- prompt | speaker B | query | text response |

masked loss

|speech response|
|---|

| |
|---|

constructed

| |
|---|

synthesized

…

constructed dialogue data from TTS data

| |
|---|

authentic

|speech response|
|---|

|| query|
|---|

prompt | speaker TTS-A | text response |

###### Inference

prompt | speaker TTS-A | query | generated text response | generated speech response

Figure 3: Data Utilization in Intelligence-Speaker Decoupling Technique.

rectly transfer speakers across tasks. To further address this, we transformed high-quality TTS data into dialogue format via constructing appropriate pseudo-context, and incorporated it into training. To preserve dialogue intelligence effectively, we excluded the text response portion from the loss calculation. Experiments show that this approach successfully transfers the naturalness of TTS speakers while maintaining intellectual levels comparable to those of dialogue speaker. A data utilization schematic diagram is shown in Figure 3. This technique enables versatile and personalized interaction while preserving high-fidelity voice naturalness with only lightweight TTS data needed.

1

##### 2.5 Full-Duplex Voice Interaction

Covo-Audio-Chat still relies on turn-based conversation mechanism, where the model is in either listening or speaking state. In contrast, our full-duplex model, Covo-Audio-Chat-FD, is endowed with simultaneous dual-stream communication that mirrors natural human interaction. To facilitate these merits, we made some modifications to the half-duplex model. First, the audio encoder is reformatted into a chunk streaming manner to enable real-time user stream input. Then the user stream and the model stream are chunk-interleaved in an 1:4 ratio3 to allow the model to listen and speak simultaneously. A comprehensive comparison between Covo-Audio-Chat and Covo-Audio-Chat-FD is illustrated in Figure 4. Additionally, we transformed the spoken dialogue datasets used in half-duplex training into dual-channel format and added conversational features such as barge-ins and backchanneling in proper positions according to semantic dynamics. To acquire robust full-duplex conversational capabilities, we performed large-scale pre-training, following the approach described in Section 2.2. Then we built Covo-Audio-Chat-FD directly from the pretrained foundation using both half-duplex and full-duplex datasets. The training is accomplished in one single step rather than using a multi-stage progressive training scheme as done in (Zhang et al., 2025c). We found this simple-yet-effective approach can yield more competitive results. In addition to intelligence preservation, full-duplex training also allows the model to master smooth turn-taking, to wisely handle user barge-ins and backchanneling behaviors.

Covo-Audio-Chat-FD works as follows. At start, during the user’s turn, the model continuously outputs THINK tokens, indicating a listening-only state while it waits to respond appropriately. Upon the end of the user’s turn, the model immediately outputs a SHIFT token to enter its speaking turn. Then four speech tokens are delivered sequentially preceded by a text token to guide semantic stability within each chunk, indicated as the listening-while-speaking state shown in Figure 4(b). During this period, the

3Since input audio features and output audio tokens are 6.25 Hz and 25 Hz respectively. Each chunk stands for 0.16s.

- (a) Half-Duplex Mode
- (b) Full-Duplex Mode

Discrete Audio Token

Start Token

Continuous Audio Feature

Speech Decoder

Text Token

Large Language Model

Adapter

(Non-) Streaming Encoder

| | | | |
|---|---|---|---|
| | | | |

User:

| | | |
|---|---|---|

Model:

Break Token

| |
|---|

Shift Token

Speech Decoder

Think Token

Large Language Model

Adapter

Chunk Streaming Encoder

| | |Backchannel| | | |
|---|---|---|---|---|---|
| | | | | | |

User:

| | | | |
|---|---|---|---|

Model:

Figure 4: A Comparative Overview of Covo-Audio-Chat and Covo-Audio-Chat-FD.

model also receives the user stream simultaneously to facilitate flexible barge-ins or backchanneling. Once an interruption signal is detected, or otherwise it would end normally, the model immediately outputs a BREAK token and terminates speaking, switching to listening.

##### 2.6 Speech Understanding

Speech conveys diverse information, including both linguistic content and paralinguistic attributes such as speaker identity, emotional state, and demographic characteristics. To enable fluent speech interaction, we define the scope of speech understanding along two dimensions: (1) content understanding, which focuses on what is said, and (2) attribute recognition, which captures how and by whom it is spoken. Content understanding encompasses tasks including automatic speech recognition (ASR), speech-to-text translation (S2TT), and spoken language identification. Attribute recognition covers gender recognition, speaker counting, and emotion recognition.

To equip the model with these capabilities, we construct multi-task question-answering data from largescale speech corpora, including ASR datasets (LibriSpeech (Panayotov et al., 2015) etc.), speaker-rich datasets (VoxCeleb (Nagrani et al., 2017), CN-Celeb (Fan et al., 2020)), emotion-rich datasets (MELD (Poria et al., 2019)), multi-attribute datasets (Common Voice (Ardila et al., 2020)), and speech rea-

soning subsets from AudioSkills (Goel et al., 2025a). During data construction, we address two key challenges: (1) cross-modal alignment, which enables consistent speech understanding across input modalities, and (2) data bias mitigation, which ensures equitable model performance across demographic groups.

Cross-Modal Data Construction We conduct cross-modal alignment to ensure the model can perceive paralinguistic attributes regardless of the input modality—whether the query is presented as text or speech. Specifically, we design a unified data construction framework centered on speech recordings with specific attributes. Let 𝑎𝑆 denote the source speech containing target attributes, 𝑡𝑄 and 𝑎𝑄 the text query and speech query, and 𝑡𝑅 and 𝑎𝑅 the text response and speech response. We construct three complementary modality configurations:

- • Text-to-Text QA: (𝑎𝑆,𝑡𝑄) → 𝑡𝑅
- • Speech-to-Text QA: (𝑎𝑆, 𝑎𝑄) → 𝑡𝑅
- • Speech-to-Speech QA: (𝑎𝑆, 𝑎𝑄) → (𝑡𝑅, 𝑎𝑅)

The construction pipeline proceeds as follows. First, we extract paralinguistic labels from open-source datasets and generate text-form QA pairs using predefined templates. Second, we synthesize corresponding speech queries and responses using high-quality TTS systems, ensuring natural prosody and diverse speaker characteristics. Finally, we mix all three modality configurations during training, enabling the model to learn modality-invariant representations for attribute recognition.

Bias Mitigation Model bias has been a challenge in dialog systems, where models trained on imbalanced data often exhibit disparate performance across demographic groups, resulting in varying accuracy for different age, gender, or language groups. Such bias is particularly detrimental for productionlevel systems. To address this challenge, we employ stratified sampling to maximize demographic balance across paralinguistic attributes. Given the varying difficulty of balancing different attributes in available corpora, we adopt a hierarchical strategy with priority: age → language → gender, ensuring that the hardest-to-balance attributes receive primary consideration.

##### 2.7 Audio Understanding

To cultivate robust audio understanding and complex reasoning capabilities, we employ a systematic training strategy centered on two distinct categories of data: Foundational Audio Perception Data and Cognitive Reasoning Data. These categories provide complementary signals that concurrently refine the model’s acoustic grounding and analytical reasoning.

FoundationalAudioSkillAcquisitionThefirstcategoryfocusesonestablishingcoreacoustic-semantic

grounding using the AudioSkill dataset (Goel et al., 2025b). This dataset encompasses a diverse array of tasks such as event recognition and cross-modal alignment, which are essential for mapping raw acoustic signals into the model’s latent space. This process ensures that the model possesses the requisite perceptual “vocabulary” to interpret complex auditory environments.

The training runs for 32k steps on the AudioSkill dataset containing 8M audio-text pairs. The model is optimized using AdamW (Loshchilov & Hutter, 2017) with differentiated learning rates: 3 × 10−6 for the LLM backbone and 1 × 10−5 for the audio adapter, with a weight decay of 5 × 10−4. We employ a linear warmup for 100 steps followed by cosine annealing.

Chain-of-Thought (CoT) Reasoning Enhancement The second category focuses on enhancing the model’s reasoning capabilities for audio understanding tasks. Traditional audio language models often

struggle with complex analytical tasks that require multi-step reasoning, such as emotion recognition involvingsubtle vocalcues, intent inference from prosodic patterns, or causal relationship identification in acoustic events. Chain-of-Thought (CoT) reasoning addresses this limitation by explicitly modeling the intermediate cognitive steps between audio perception and final conclusion, enabling the model to decompose complex audio comprehension into interpretable logical pathways. To incorporate CoT capabilities, we construct a reasoning-enhanced dataset by merging and shuffling two complementary sources: AudioMCQ (He et al., 2025) and AF-Think (Goel et al., 2025b), totaling 1M training instances. All samples follow a unified format that explicitly separates the reasoning process from the final answer.

Since AF-Think originally provides only the reasoning process without explicit answers, we employ DeepSeek V3.1 (DeepSeek-AI, 2024) to extract and format the final answers, ensuring consistency across the entire dataset. We perform full-parameter training for two complete epochs using the same hyperparameters as the foundational stage, except with 1,000 warmup steps to accommodate the longer trainingduration. ThisCoT-enhancedtrainingenablesthemodeltonotonlyrecognizeacousticpatterns but also articulate the reasoning behind its audio understanding decisions.

Following supervised fine-tuning, we employ Group Relative Policy Optimization (GRPO) (Shao et al.,

- 2024) to refine the model’s audio reasoning capabilities using data from the Audio-Visual Question Answering (AVQA) benchmark (Yang et al., 2022). Inspired by Audio-Thinker (Wu et al., 2025c), which demonstrates the effectiveness of explicit reasoning for audio understanding, we design a verifiable composite reward function comprising four complementary components:

𝑅total = 𝑅accuracy + 𝑅format + 𝑅consistency + 𝑅thinking (1)

Accuracy Reward (𝑅accuracy ∈ {0,1}): Measures whether the model’s final answer matches the ground truth, providing direct supervision for correctness.

Format Reward (𝑅format ∈ {0,1}): Verifies adherence to the structured output requirement, checking for the presence of both reasoning and answer sections:

𝑅format = 1 [match(output,<think>.*</think>.*<answer>.*</answer>)] (2)

Consistency Reward (𝑅consistency ∈ {0,1}): Evaluates logical coherence between the reasoning process and final answer using an auxiliary LLM judge. This ensures that the model’s explicit reasoning aligns with its conclusions.

Thinking Quality Reward (𝑅thinking ∈ [0,1]): Assesses the reasoning depth across five dimensions: logical soundness, factual correctness, error identification, language coherence, and conciseness. The score is obtained via LLM-based evaluation in 0.2 increments.

This multi-faceted reward structure enables the model to simultaneously optimize for correctness, structural compliance, internal consistency, and reasoning quality, addressing limitations of supervised learning alone in capturing nuanced audio understanding behaviors.

#### 3 Experiments

##### 3.1 Evaluation on Pre-training

To rigorously assess the multi-modal capabilities of our model, we established a comprehensive evaluation suite encompassing six distinct task categories. This framework covers A2A, A2T, and T2T scenarios. Specifically, for A2A tasks, we evaluated the model’s ability to generate speech responses directly

- Table 3: Performance comparison of pre-trained model and baselines on speech-text modeling capabilities.

Models Params A2A-tSC(↑) A2T-tSC(↑) T2T-tSC(↑) sBLIMP(↑) sWUGGY(↑) Qwen2.5-7B-Base 7B - - 98.6 - -

TWIST 7B 74.1 - - 59.0 73.90 SpiritLM 7B 82.9 88.6 98.0 58.3 69.00 Slam 358M 82.0 - - 58.8 68.38 AlignSLM 7B 86.8 - - 62.3 SIMS 7B 88.3 94.0 98.0 59.8 75.36

Moshi 7B 83.0 - - 58.8 MinMo 7B 75.4 - - 55.4 68.59 LLaMA-Omni2 7B 78.2 - - 53.6 73.21

GLM-4-Voice-Base 9B 82.9 93.6 - - Step-Audio-2-mini-Base 7B - 95.5 98.5 52.4 52.70 Covo-Audio-Base 7B 83.3 95.7 99.4 61.6 74.90

- Table 4: Performance comparison of pre-trained model and baselines on ASR and TTS tasks. LS clean and LS other represent Librispeech test clean and test other set, respectively.

ASR (↓) Seed-TTS (↓) Aishell-1 LS clean LS other Test-en Test-zh

Models Params

Whisper-large-v3 1.5B 9.31 2.50 4.53 - CosyVoice2 0.5B - - - 2.57 1.45

SpiritLM 7B - 6.00 11.0 - GLM-4-Voice-Base 9B 2.46 2.82 7.66 2.91 2.10 Covo-Audio-Base 7B 1.96 1.96 4.55 2.44 1.73

from speech input, focusing on dimensions such as creative story continuation (topic StoryCloze (Hassid et al., 2023)), grammatical integrity (sBLIMP (Nguyen et al., 2020)), and structural consistency (sWUGGY (Nguyen et al., 2020)). Based on the tSC data, we also evaluated the model’s performance on both A2T and T2T tasks. Furthermore, standard benchmarks for ASR and TTS are integrated to provide a holistic measure of the model’s foundational speech processing and synthesis performance.

To evaluate the performance of the pre-trained model, we compare it against a representative suite of speech language model baselines, categorized into three distinct groups. The first group consists of specialized A2A architectures, including TWIST (Hassid et al., 2023), SpiritLM (Nguyen et al., 2025), Slam (Maimon et al., 2025a), AlignSLM (Lin et al., 2025), and SIMS (Maimon et al., 2025b). These models are specifically engineered and trained for direct A2A tasks. Secondly, we include Moshi, Minimo Chen et al. (2025a) and Llama-Omni2 Fang et al. (2025), the prominent unified speech dialogue modelrecognizedforitscomprehensivemodeling. Furthermore, weevaluatedleadingpre-trainedfoundations for speech-to-speech dialogue, specifically GLM-4-Voice-Base and Step-Audio2-mini-Base. Beyond cross-modal alignment performance, we extended our evaluation to include domain-specific models: Whisper is utilized for ASR benchmarks, while CosyVoice2 Du et al. (2024) served as the baseline for TTS. By benchmarking against this diverse array of models—ranging from specialized S2S, ASR, TTS frameworks to generalized pre-trained speech systems—we aim to demonstrate the capabilities in both high-level semantic reasoning and foundational acoustic generation.

Our analysis primarily focused on the efficacy of speech-text modality alignment. The results are

shown in Table 3. In the A2A story continuation task, our model achieved performance parity with sota pre-trained and specialized A2A models. It is noteworthy that many specialized A2A baselines, such as SpiritLM and Slam, exhibit signs of overfitting on continuation tasks, as they are often biased toward speech-to-speech mapping at the expense of broader linguistic flexibility. Furthermore, while models like AlignSLM and SIMS benefit from extra reinforcement learning, our results were achieved through pre-training alone. Crucially, our model significantly outperformed all existing baselines in logical reasoning, grammatical accuracy in A2A metrics, and cross-modality consistency across A2T and T2T benchmarks. These results demonstrate that our pre-training objective successfully fused the speech modality with the core cognitive and generative powers of the LLM. This suggests that the model does not merely transcribe audio but deeply extracts semantic information from speech signals to inform its high-level reasoning.

Beyond high-level reasoning, we evaluated the model’s performance on fundamental ASR and TTS tasks. Despite being a unified multi-modal system, our model demonstrated competitive proficiency compared to the GLM-4-Voice-Base model, as shown in Table 4. The end-to-end pre-training paradigm we propose facilitates a robust bi-directional semantic mapping between speech and text. This synergy ensuresthatthe modelpreservesfine-grained acousticdetails while maintaining the sophisticated linguistic understanding inherent in large-scale language modeling.

##### 3.2 Evaluation on Speech-to-Speech Spoken Dialogue

Evaluation on Spoken Dialogue We evaluate the end-to-end spoken dialogue capability of CovoAudio-Chat on URO-Bench (Yan et al., 2025) and VCB Bench (Hu et al., 2025) in an A2A mode. Specifically, we adopt the basic Chinese and English track of URO-Bench to assess the model’s performance in speech comprehension, speech reasoning, and spoken dialogue. For VCB Bench, our evaluation focuses on the model’s text-side and speech-side instruction-following ability, internal knowledge retention, and anti-interference capability. The models used for comparison include: GLM-4Voice (Zeng et al., 2024), Freeze-Omni (Wang et al., 2024b), Kimi-Audio (Ding et al., 2025), Qwen2.5Omni (Xu et al., 2025a), Baichuan-Audio-Chat (Li et al., 2025), Step-Audio 2 series (Wu et al., 2025a), MiMo-Audio (Xiaomi, 2025), GPT-4o Audio, Qwen3-Omni (Xu et al., 2025b), and Fun-Audio-Chat (Chen et al., 2025b).

The results on the URO-Bench are shown in Table 5. As illustrated, Covo-Audio-Chat demonstrates comprehensive superiority in core tasks, particularly excelling in speech reasoning and spoken dialogue which are critical for practical speech interaction. Specifically, on the Chinese track, Covo-Audio-Chat achieves the highest scores in four reasoning tasks (SQuAD: 77.34, OpenbookQA: 83.60, APE: 68.42, MLC: 80.69) and two spoken dialogue tasks (AlpacaEval: 90.02, Wildchat: 90.41), outperforming all comparative models including the advanced ones like Qwen3-Omni. In speech comprehension, its Repeat (98.17) and HSK5 (88.00) scores are among the top tiers, only slightly lower than Step-Audio 2 mini and MiMo-Audio respectively. On the English track, Covo-Audio-Chat maintains its strength in reasoning, achieving the best performance in Gsm8kEval (85.68). Though not leading in speech comprehension and partial spoken dialogue tasks, its overall performance is consistently competitive. In summary, Covo-Audio-Chat demonstrates robust and balanced performance across both Chinese and English tasks despite at a compact 7B-parameter scale, highlighting its strong bilingual spoken dialogue capabilities. This gives it a key advantage over many models that lack comprehensive bilingual coverage, and solidifies its overall dominance in core speech-to-speech interaction scenarios.

The results on the VCB Bench are shown in Table 6, evaluating speech-to-speech models across three core dimensions: Instruction Following, Knowledge, and Robustness. As illustrated, our proposed Covo-Audio achieves excellent performance in key metrics, demonstrating its superiority in practi-

- Table 5: Performance Comparison on the Basic Track of URO Bench. Results marked with (∗) are sourced from (Yan et al., 2025), and those marked with (†) are from (Wu et al., 2025a). Covo-AudioChat-TTS denotes dialogue voice transferred from a TTS speaker.

(a) Results on Chinese Basic Track.

Understanding(↑) Reasoning(↑) Oral Conversation(↑) Repeat LCSTS HSK5 SQuAD OpenbookQA APE MLC AlpacaEval Claude Wildchat

Model

GLM-4-Voice∗ 92.64 77.08 69.00 28.75 56.96 15.78 78.85 83.35 82.12 84.48 Freeze-Omni∗ 4.97 71.82 7.66 9.58 16.40 11.75 47.35 67.98 64.89 71.28 GPT-4o Audio∗ 93.50 81.60 88.00 42.67 76.00 25.33 81.33 86.40 82.93 80.00 Kimi-Audio† 73.32 78.40 - - 66.93 - 52.45 69.90 81.81 Qwen2.5-Omni† 19.66 73.22 - - 67.70 - 61.76 70.40 79.71 -

- Step-Audio 2† 96.16 79.16 - - 74.06 - 62.99 76.23 84.54 Step-Audio-AQAA† 63.72 68.33 - - 47.21 - 67.40 56.23 31.50 Step-Audio 2 mini 99.83 81.18 80.67 64.71 70.02 45.09 74.94 74.29 72.58 79.58

- MiMo-Audio 0.44 79.61 92.67 35.73 83.60 61.05 38.16 85.90 89.10 84.84 Qwen3-Omni 99.48 84.54 88.67 52.07 73.02 26.67 36.09 79.59 78.86 74.54 Fun-Audio-Chat 97.50 80.95 79.00 51.85 67.90 44.91 76.09 80.59 86.46 84.62

Covo-Audio-Chat 98.17 79.89 88.00 77.34 83.60 68.42 80.69 90.02 87.18 90.41 Covo-Audio-Chat-TTS 97.68 79.22 91.00 76.91 80.60 64.21 79.31 89.34 87.99 89.28

(b) Results on English Basic Track.

Model

Understanding(↑) Reasoning(↑) Oral Conversation(↑)

Repeat Summary GaokaoEval StoralEval TruthfulEval Gsm8kEval MLC AlpacaEval CommonEval WildchatEval

GLM-4-Voice∗ 90.95 91.07 64.47 73.80 59.28 30.93 57.82 80.77 63.07 78.76 Freeze-Omni∗ 70.89 78.87 26.29 57.74 46.95 2.81 42.56 52.23 48.70 55.80 GPT-4o Audio∗ 97.16 94.13 72.00 84.27 82.67 80.00 80.00 95.20 94.13 95.20 Kimi-Audio† 90.27 78.19 81.63 70.88 53.90 0.97 43.50 63.05 51.40 66.63 Qwen2.5-Omni† 28.36 80.51 89.99 72.90 62.03 63.52 80.04 78.16 72.40 77.92 Step-Audio 2† 97.74 83.90 90.76 76.38 65.45 62.83 77.02 81.74 72.87 81.62 Step-Audio-AQAA† 93.02 90.62 86.80 53.00 62.30 38.20 71.00 71.86 75.53 68.79

Step-Audio 2 mini 97.81 84.69 85.15 73.17 60.45 28.12 74.01 77.72 62.63 79.66

- MiMo-Audio 1.08 87.74 71.29 79.34 57.62 45.59 23.16 84.92 67.27 82.94 Qwen3-Omni 97.36 96.16 81.30 92.27 70.70 9.45 50.85 82.95 75.37 85.50 Fun-Audio-Chat 97.18 91.98 76.13 79.97 66.68 62.08 79.66 89.21 78.83 87.91

Covo-Audio-Chat 95.24 87.23 89.11 82.59 67.60 85.68 77.78 85.73 79.67 86.82 Covo-Audio-Chat-TTS 92.71 88.64 87.24 82.32 63.67 77.78 74.58 84.49 79.13 87.09

cal speech interaction scenarios. In the Instruction Following task, Covo-Audio-Chat outperforms all competing models in Text-based Instruction Following (TIF: 93.07, TIF-En: 89.94) and Multi-Turn Dialogue (MTD: 87.70), highlighting its strong ability to comprehend and execute both Chinese and English text instructions, as well as maintain coherent and compliant communication in multi-round interaction. Notably, in the Robustness evaluation—critical for real-world applicability—Covo-AudioChat sets the state-of-the-art across all sub-tasks: Speaker Variation (SV: 88.94), Environmental Noise (EV: 87.13), and Content Variation (CV: 90.37), which underscores its robust adaptability to diverse interfering factors such as accents, background noise, and grammatical deviations. In the Knowledge dimension, Covo-Audio-Chat achieves competitive performance in Mathematical Logic (ML: 79.34), approaching the top-tier level, while its General Knowledge (GK) and Dialogue Comprehension (DC) scores are modest compared to Qwen3-Omni—this reflects a deliberate prioritization of core interaction capabilities during model development, and these aspects can be enhanced by integrating specialized knowledge bases and refining contextual modeling. Overall, Covo-Audio-Chat demonstrates outstanding performance in most critical dimensions of speech-to-speech interaction (text instruction following, multi-turn dialogue, and robustness), laying a solid foundation for practical applications, with the identified limitations providing clear pathways for further optimization.

Evaluation on Empathy We evaluate empathetic interaction on VStyle benchmark (Zhan et al., 2025).

- Table 6: Performance Comparison on the VCB Bench. Missing results due to API unavailability.

Model

Instruction Following(↑) Knowledge(↑) Robustness(↑) TIF TIF-En SIF SIF-En MTD GK ML DC SV EV CV

GLM-4-Voice 82.15 75.52 73.18 66.94 82.56 41.79 60.18 - 73.64 77.51 78.60 Kimi-Audio 77.33 60.37 71.04 54.13 82.27 54.47 75.42 51.96 64.35 65.14 71.51 Qwen2.5-Omni 76.75 60.83 54.04 42.18 85.23 48.13 73.00 72.81 77.82 74.41 78.27 Baichuan-Audio 82.24 78.60 53.67 48.27 80.96 40.73 74.05 51.66 78.05 77.62 78.71 Step-Audio 2 mini 76.10 70.47 60.80 49.85 82.31 49.86 72.85 79.46 75.19 70.58 74.78 MiMo-Audio 90.08 81.89 56.26 42.74 86.30 48.70 81.75 - 83.72 85.36 89.38 GPT-4o Audio 86.94 88.80 77.98 82.90 33.59 55.81 73.45 76.74 80.34 79.92 86.51 Qwen3-Omni 90.45 85.17 70.73 65.57 87.17 66.86 81.90 82.78 87.91 85.63 86.03 Fun-Audio-Chat 89.30 89.39 78.82 75.70 85.27 46.30 73.00 79.15 88.60 83.83 85.15

Covo-Audio-Chat 93.07 89.94 67.41 60.86 87.70 49.95 79.34 64.95 88.94 87.13 90.37

- Table 7: Empathy Performance Comparison on the VStyle Benchmark. Results marked with (∗) are sourced from (Zhan et al., 2025), and those marked with (†) are from (Chen et al., 2025b).

Anger(↑) Sadness(↑) Anxiety(↑) Joy(↑) en zh en zh en zh en zh

Model

GPT-4o-Mini∗ 4.98 4.30 5.00 4.52 4.87 4.73 5.00 4.69 GPT-4o∗ 4.95 4.75 4.90 4.83 5.00 4.67 4.54 4.80 Doubao∗ 4.89 4.59 5.00 4.72 4.81 4.80 4.94 4.83

Baichuan-Audio∗ 2.41 2.11 3.43 2.55 2.74 2.20 3.91 3.51 Kimi-Audio∗ 3.59 3.86 3.97 3.86 3.65 3.80 3.46 4.57 Step-Audio∗ 3.95 3.59 4.37 4.52 3.87 3.20 4.29 4.26 Qwen2.5-Omni∗ 2.95 4.64 2.73 4.28 3.55 4.77 3.43 4.91 Fun-Audio-Chat-8B† 3.64 3.73 4.10 3.93 2.90 4.03 3.69 3.77

Covo-Audio-Chat 4.86 4.89 4.83 4.93 4.97 5.00 4.80 4.89

- As shown in Table 7, Covo-Audio-Chat demonstrates exceptional performance on the VStyle empathy dataset, which assesses a model’s capability to empathetically respond to and support users across diverse emotional states. Covo-Audio-Chat achieves state-of-the-art (SOTA) results in Mandarin, specifically attaining the highest scores in anger (4.89), sadness (4.93), and anxiety (5.00). In English, the model delivers highly competitive performance comparable to leading closed-source models. It’s worth noting that though Covo-Audio-Chat achieves leading objective scores using the LLM-as-aJudge method, this doesn’t reflect real voice expressions as the LALM judge (Gemini 2.5 Pro) is likely to prioritize semantic content over the overall quality in speech response. In a preliminary subjective test on voice empathy, we found Covo-Audio-Chat still underperforms the top-tier productions like Doubao4, advising it is extremely urgent to establish a more equitable evaluation criterion. But overall, especially compared to existing open-sourced models, Covo-Audio-Chat exhibits superior ability to recognize nuanced emotional cues and generate contextually appropriate empathetic responses, validating its effectiveness as a supportive conversational agent for real-world emotional interaction scenarios. Please refer to Appendix A for case study.

Evaluation on Intelligence-Speaker Decoupling Technique We also evaluate Covo-Audio-Chat-TTS, a variant that applies TTS voice directly to spoken dialogue scenarios in inference. As illustrated in Table 5, it achieved comparable bilingual dialogue performance to Covo-Audio-Chat, demonstrating the successful voice transfer and sharing from TTS while preserving conversational intelligence via our proposed intelligence-speaker decoupling technique. This approach effectively mitigates the pro-

4https://www.doubao.com/chat/

###### Table 8: Full-Duplex Performance Comparison on URO-Bench.

(a) Results on English Basic Track

Understanding(↑) Reasoning(↑) Oral Conversation(↑)

Model

Repeat Summary GaokaoEval∗ StoralEval TruthfulEval Gsm8kEval MLC AlpacaEval CommonEval WildchatEval

Moshi 1.26 38.19 6.71 39.50 38.37 3.09 47.64 48.48 40.60 48.18 Freeze-Omni 29.96 20.11 0.66 25.31 44.75 1.20 18.83 52.16 48.30 52.95 Covo-Audio-Chat-FD 94.64 83.45 73.82 80.80 64.51 80.47 74.76 84.25 75.07 88.34

(b) Results on Chinese Basic Track

Understanding(↑) Reasoning(↑) Oral Conversation(↑)

Model

Repeat LCSTS HSK5 SQuAD OpenbookQA APE MLC AlpacaEval Claude Wildchat

Freeze-Omni 3.71 42.75 2.33 4.58 4.94 12.81 47.13 66.17 59.94 70.43 Covo-Audio-Chat-FD 98.35 80.72 83.67 75.16 79.37 70.70 78.85 84.90 83.93 88.34

∗We found that the “GaokaoEval” test set contains very long silent pauses between two adjacent vocal fragments, which may easily cause the model to respond premuturely. The “early-response” leads to a clear performance degradation, as compared against the corresponding scores in Table 5.

###### Table 9: Full-Duplex Interaction Performance Metrics (Success Rate (%)).

Model Turn-taking(↑) Pause Handling(↑) Backchanneling(↑) Interruption(↑)

Moshi 96.8 53.2 - Freeze-Omni 99.1 51.2 - -

Covo-Audio-Chat-FD 99.7 97.6 93.89 96.81

hibitive cost inherent to end-to-end paradigm when developing a chat bot of high naturalness, paving the way for flexible voice customization with only lightweight TTS data needed.

##### 3.3 Evaluation on Full-Duplex Interaction

We evaluate Covo-Audio-Chat-FD on two dimensions: (1) spoken dialogue capability measured on URO-Bench, and (2) full-duplex interaction behaviors, including turn-taking, pause handling, interruption, and backchanneling (Zhang et al., 2025b). A behavior is considered successful if the model properly handles the corresponding full-duplex feature; for example, it is a successful turn-taking when the model responds after the user finishes speaking. Following Section 3.2, we measure dialogue performance on URO-Bench using only the first-round response, and additionally report first-round success rates of turn-taking and pause handling. To evaluate user interruption and backchanneling, which examine overlap handling, we inject the corresponding audio events during the model’s response period by randomly sampling from a corpus recorded naturally and diversifiedly. We compare against Moshi and Freeze-Omni under identical autonomous full-duplex settings.

- As shown in Table 8, Covo-Audio-Chat-FD substantially outperforms Moshi and Freeze-Omni across understanding, reasoning, and oral conversation tasks on both English and Chinese tracks. In particular, the model achieves particularly large margins on reasoning benchmarks, indicating that the full-duplex training does not degrade the model’s core reasoning intelligence. Compared with Covo-Audio-Chat (Table 5), Covo-Audio-Chat-FD maintains highly comparable spoken dialogue performance with only a slight drop. This small gap largely stems from the model’s occasional “early-response” behavior: it may interrupt the user prematurely on the point of short pauses within a turn, a failure mode that correlates directly with its pause-handling success metric, which also points out the critical direction for optimization.

Table 9 summarizes the full-duplex interaction metrics. Against existing open-source baselines, CovoAudio-Chat-FD achieves strong results on turn-taking and pause handling dynamics, with especially notable performance in pause handling—a capability that involves correctly disregarding short pauses within a user’s turn. This reflects that Covo-Audio-Chat-FD is more reliable to understand user intentions. Furthermore, the model shows competent performance in managing user interruption (96.81%) and backchanneling (93.89%), guaranteeing its excellent interaction experience in practice.

##### 3.4 Evaluation on Speech Understanding

In this section, following the section 2.6 we evaluate the speech understanding capabilities of our model across a diverse set of speech-centric tasks. Specifically, we consider automatic speech recognition (ASR), speech-to-text translation (S2TT), and a range of paralinguistic understanding tasks, including emotion recognition, speaker gender classification, and age prediction. These tasks jointly cover both linguistic content understanding and non-linguistic acoustic cue modelling, providing a comprehensive assessment of the model’s speech understanding ability. All tasks are formulated under a unified framework and trained using SFT on task-specific and mixed-task speech-text data. Such a setting reflects realistic speech interaction scenarios, where models are expected to simultaneously recognize spoken content, translate across languages, and infer speaker attributes from raw audio signals.

We evaluated the ASR performance of our SFT-based speech understanding model against a diverse set of SOTA commercial, closed-source, and open-source audio-language models. The comparison includes GPT-4o Audio (Hurst et al., 2024), Gemini 2.5 Pro (Team et al., 2024a), Voxtral-small (Liu et al., 2025), Seed-ASR (Bai et al., 2024), as well as several recent open multimodal models such as Qwen2.5-Omni (Yang et al., 2024a), Qwen3-Omni-30B (Xu et al., 2025b), LongCat-Flash-Omni (Team, 2025), and Step-Audio 2 (Wu et al., 2025a). All models were evaluated under identical decoding and data conditions, and performance was measured using word error rate (WER), where lower values indicate better recognition accuracy. As shown in Table 10, our Covo-Audio achieves competitive ASR performance across all evaluation settings, with an average WER of 4.71.

We further evaluate the S2TT capability of our model on the CoVoST2 (Wang et al., 2020) benchmark on English and Chinese languages. CoVoST2 is a widely used multilingual speech translation dataset that covers diverse source–target language pairs and presents significant challenges in terms of acoustic variability and cross-lingual semantic alignment. The results on CoVoST2 are summarized in Table 10. Covo-Audio achieves the best performance on en-zh translation, surpassing the strong baselines like Step-Audio 2 mini, while also delivering competitive results against other models on the more challenging zh-en direction.

Following the description in Section 2.6, we evaluate the paralinguistic ability on AIR-Bench (Yang

- et al., 2024b) benchmark, a comprehensive benchmark covering multiple speech paralinguistic tasks, including emotion recognition, speaker gender classification, and age prediction. As shown in Table 10, our model achieves the best overall performance across all evaluated paralinguistic tasks on AIRBench. The most significant improvements are observed on emotion recognition, where our model achieves absolute gains of over 10-25 percentage points compared to prior methods. This suggests that dialogue-level supervised fine-tuning effectively enhances the model’s sensitivity to acoustic patterns beyond lexical content, making it particularly well-suited for real-world spoken interaction scenarios, where linguistic and paralinguistic cues can be jointly modeled.

###### Table 10: Performance Comparison on Speech Understanding Benchmarks.

###### Datasets Model Performance

GPT-4o Audio 15.30/32.27 | 1.39/3.75 | 3.32/2.44 | 3.52 | 8.85 Seed-ASR 4.55/ 5.69 | 1.58/2.84 | 3.40/2.69 | 0.68 | 3.07 Gemini 2.5 Pro 14.43/13.47 | 2.89/3.56 | 2.94/2.71 | 3.11 | 6.15 Voxtral-small 20.33/26.08 | 1.56/3.30 | 3.77/7.98 | - | 11.26 Qwen2.5-Omni 5.91/ 7.65 | 1.74/3.45 | 3.77/2.54 | 1.17 | 3.74 Qwen3-Omni-30B 4.69/ 5.89 | 1.22/2.48 | 2.72/2.20 | 0.64 | 2.83 LongCat-Flash-Omni 6.09/ 6.69 | 1.57/4.07 | 5.02/3.99 | 0.63 | 4.00 Step-Audio 2 4.67/ 4.75 | 1.17/2.42 | 3.03/2.68 | 0.63 | 2.76

ASR Benchmarking (↓) WenetSpeech (Net/Meeting) | LibriSpeech (Clean/Other) | Fleurs (EN/ZH) | Aishell-1 | Avg

Covo-Audio 7.23/ 8.33 | 1.45/3.21 | 5.08/6.64 | 1.07 | 4.71

GPT-4o Audio 40.20 | 19.01 MinMo 46.70 | 26.00 Qwen2.5-Omni 41.40 | 29.40 Qwen3-Omni 48.72 | 21.50 Step-Audio 2 mini 49.12 | 29.47 LongCat-Flash-Omni 47.23 | 27.32

CoVoST2 (↑) en-zh | zh-en

###### Covo-Audio 49.84 | 26.77

GPT-4o Audio 96.80 | 21.90 | 59.50 | 41.10 | 30.00 | 49.86 Qwen2.5-7B 88.00 | 28.36 | 50.80 | 36.40 | - | 50.89

AIR-Bench (↑) Spoken Language ID | Speaker Gender | Emotion | Speaker Age | Speaker Number | Avg

- Qwen2-Audio 77.80 | 64.70 | 48.20 | 23.10 | 30.20 | 48.80 Qwen2.5-Omni 89.70 | 89.80 | 54.90 | 44.80 | - | 69.80
- Qwen3-Omni 97.70 | 91.11 | 62.20 | 36.90 | - | 71.98 Gemini 2.5 Pro 99.10 | 90.70 | 60.70 | 34.10 | - | 71.15 Covo-Audio 99.30 | 91.52 | 72.30 | 70.90 | 70.30 | 80.86

Table 11: Performance Comparison on MMAU-v05.15.25 and MMSU. Results for other methods are directly sourced from the official MMAU Leaderboard and MMSU paper.

MMAU-v05.15.25(↑) MMSU(↑) Sound Music Speech Avg Perception Reasoning Avg

Model Param

GPT-4o Audio - 64.56 56.29 66.67 62.50 39.67 71.96 56.38 GPT-4o mini Audio - 50.75 39.22 69.07 53.00 - - Gemini 2.0 Flash - 71.17 65.27 75.08 70.50 40.83 47.83 51.03 Gemini 2.5 Flash - 73.27 65.57 76.58 71.80 - - Gemini 2.5 Pro - 75.08 68.26 71.47 71.60 - - -

Qwen2-Audio-Instruct 8.4B 67.27 56.29 55.26 59.60 39.02 68.90 53.27

- Audio Flamingo 2 3.2B 71.47 70.96 44.74 62.40 - - -
- Audio Flamingo 3 7B 79.58 73.95 66.37 73.30 - - 61.40 Kimi-Audio 7B 75.68 66.77 62.16 68.20 43.52 76.03 59.28 MiMo-Audio 7B 81.68 74.25 68.17 74.70 - - Qwen2.5-Omni 7B 78.10 65.90 70.60 71.50 42.50 79.83 60.57 Step-Audio 2 mini 7B 79.30 68.44 68.16 72.73 - - Step-Audio 2 32B 84.04 73.56 75.15 77.58 - - Covo-Audio 7B 78.68 76.05 71.17 75.30 58.95 74.83 66.64

##### 3.5 Evaluation on Audio Understanding

To comprehensively assess the effectiveness of our three-stage training paradigm, we conduct systematic evaluations on two established audio understanding benchmarks: MMAU (Sakshi et al., 2024) and MMSU (Wang et al., 2025a). We compare our model’s performance against several state-of-the-art systems, including both closed-source models (GPT-4o Audio (Hurst et al., 2024), GPT-4o mini Audio (Hurst et al., 2024), Gemini 2.0 Flash, Gemini 2.5 Flash (Team et al., 2024a), Gemini 2.5 Pro (Team

- et al., 2024a), and Gemini 2.5 Flash Lite (Team et al., 2024a)) and open-source models (Qwen2-Audio-

Instruct (Chu et al., 2024), Audio Flamingo 2 (Ghosh et al., 2025) Audio Flamingo 3 (Goel et al.,

- 2025b), Kimi-Audio (Ding et al., 2025), Qwen2.5-Omni (Xu et al., 2025a), and Step-Audio 2 (Wu

- et al., 2025a)). The comparative analysis demonstrates that our three-stage training approach achieves competitive or superior performance across multiple evaluation metrics, validating both our architectural choices and training methodology. The progressive improvements observed through each training stage underscore the effectiveness of our systematic capability development strategy.

As shown in Table 11, Covo-Audio achieves highly competitive results across both benchmarks. On MMAU-v05.15.25 (Test-mini subset), our model obtains an average score of 75.30%, ranking second among all open-source models and surpassing most closed-source systems. Notably, while Step-Audio 2 achieves the best overall performance (77.58%), it employs a significantly larger 30B parameter architecture. Among models of comparable scale (7B-parameter), Covo-Audio achieves the best performance, demonstrating superior parameter efficiency. Our model particularly excels in music understanding, achieving the highest score of 76.05% among all models, and maintains strong performance in speech understanding. On the MMSU benchmark, our model demonstrates even more pronounced advantages, achieving 66.64% average accuracy—the highest score among all evaluated systems, including both closed-source and open-source models. This substantial margin over the second-best model (Audio Flamingo 3 at 61.40%) validates the effectiveness of our three-stage training approach. Specifically, our model excels in perception tasks with a remarkable score of 58.95%, significantly outperforming all other models including Kimi-Audio (43.52%) and Gemini 2.0 Flash (40.83%). While our reasoning performance (74.83%) is competitive, Qwen2.5-Omni achieves the highest reasoning score (79.83%), suggesting potential directions for future improvements in our Chain-of-Thought reasoning enhancement stage. The progressive improvements observed through each training stage underscore the effectiveness of our systematic capability development strategy.

#### 4 Related Work

LargeLanguageModels TheevolutionofLLMshasbeencharacterizedbyatransitionfrommassive-

scale scaling to a more nuanced focus on data quality and architectural efficiency. The GPT series (Radford et al., 2019; Brown et al., 2020; Achiam et al., 2023) and PaLM (Chowdhery et al., 2023) established the foundational scaling laws, demonstrating that emergent reasoning capabilities arise from increasing parameter counts and data volumes. This trajectory was democratized by the Llama family (Touvron et al., 2023; Dubey et al., 2024), which popularized efficient Transformer variants and set the stage for subsequent open-source innovations.

Based on the LLM paradigm, there are many derived research directions. Phi series (Abdin et al., 2024) and SmolLM (Allal et al., 2025) shift toward efficient and lightweight models. They have demonstrated that by training on “textbook-quality” data, models with significantly fewer parameters can rival much larger counterparts in specialized reasoning and coding tasks. The Qwen series (Team et al., 2024b; Yang et al., 2024a; 2025) represent a significant leap in the “all-in-one” model ecosystem, offering an extensiverangeofscalesfrom0.5Bto72BparametersalongsidespecializedMixture-of-Experts(MoE) variants. These models delivering sota performance in multilingual understanding, complex coding, and long-context reasoning. Kimi has pushed the boundaries of long-context understanding, optimizing the Transformer’s memory footprint to handle millions of tokens effectively. The DeepSeek-R1 (Guo et al., 2025) represents a breakthrough in open-source reasoning models, utilizing Reinforcement Learning (RL) to cultivate autonomous Chain-of-Thought (CoT) capabilities.

Multimodal Large Language Models The success of LLMs has catalyzed significant research interest in extending linguistic intelligence to multimodal perceptions. The initial surge in this field focused

on Vision-Language Models (VLMs). Early works like BLIP-2 (Li et al., 2023) and Flamingo (Alayrac et al., 2022) introduced specialized bridge architectures, such as the Q-Former or Perceiver Resampler, to align frozen visual encoders with frozen LLM backbones. This approach was further democratized by LLaVA (Liu et al., 2023), which utilized a simple linear projection layer to map visual features into the word embedding space, demonstrating that instruction-tuning on high-quality image-text pairs could yield impressive multimodal reasoning capabilities. Recent iterations like LLaVA-NeXT (Liu et al., 2024) have further enhanced these capabilities, supporting high-resolution images, videos, and multi-image reasoning. Subsequent advancements, such as the Qwen-VL (Wang et al., 2024a; Bai

- et al., 2025b;a) and InternVL (Chen et al., 2024; Zhu et al., 2025; Wang et al., 2025c) series, have focused on scaling both the resolution of visual inputs and the quality of interleaved datasets. These models often utilize a “Vision-Tower + Adapter + LLM” pipeline, where the visual encoder is progressively unfrozen to achieve deeper cross-modal synergy. Both X-Omni (Geng et al., 2025) and Skywork UniPic (Wang et al., 2025b) represent a significant trend in multimodal research by focusing on the unification of understanding and generation within a single framework. X-Omni achieves this by aligning diverse modalities into a shared latent space to facilitate both complex reasoning and high-fidelity synthesis, while Skywork UniPic employs a unified perception-generation architecture that treats various visual tasks as a consistent sequence modeling problem.

A more recent paradigm shift is the move toward Native Multimodal Models, which abandon the modular “plug-in” approach in favor of a unified architecture trained on multiple modalities from the outset. Unlike vision-language models that append visual features to a text-centric model, native models like Gemini (Team et al., 2023; Comanici et al., 2025) and GPT-4o (Hurst et al., 2024) are designed with an inherently multimodal vocabulary and shared hidden spaces. This allows the model to process and generate different modalities (text, image, and audio) using the same set of parameters, facilitating true cross-modal fluidly. Other notable efforts include Emu (Cui et al., 2025b) and Chameleon (Team, 2024), which utilize discrete tokenization for all modalities, treating images and audio as “foreign languages” within a unified Transformer.

Large Audio Language Models Early efforts in speech-text integration primarily focused on aligning speech encoders with existing textual LLMs to leverage their pre-trained reasoning capabilities. Llama-Omni (Fang et al., 2024) and Mini-Omni (Xie & Wu, 2024) demonstrated that by using a speech adapter and a non-autoregressive speech decoder, Llama-based backbones could achieve high-quality speech-to-speech interaction with low latency. Freeze-Omni (Wang et al., 2024b) utilizes a “frozen” LLM backbone to preserve original textual intelligence while plugging in streaming speech encoders and decoders. By keeping the core model parameters fixed, Freeze-Omni avoids the catastrophic forgetting of textual knowledge often induced by multimodal fine-tuning.

Thinker-Talker architecture separates the cognitive reasoning process from the acoustic synthesis process to ensure logical consistency. Qwen-Omni series (Xu et al., 2025a;b) adopt this paradigm, utilizing a “Thinker” module to handle semantic deliberation before or during the “Talker” module’s speech generation. Qwen2.5-Omni introduced a framework to synchronize video, audio, and text, while Qwen3-Omni further optimizes this with a Mixture-of-Experts (MoE) design. This allows the model to process over 40 minutes of audio and deliver real-time, multilingual streaming responses.

To overcome the “information bottleneck” of text-mediated dialogue, several models have moved toward native E2E architectures that treat audio as a primary modality. Moshi (Défossez et al., 2024) pioneered this shift by predicting audio and text streams simultaneously using a multi-stream Transformer, enabling true full-duplex interaction. In the domestic research field, GLM-4-Voice (Zeng et al., 2024), Baichuan-Audio (Li et al., 2025) and Baichuan-Omni (Li et al., 2024) models can handle Chinese-English bilingual conversations with high-fidelity emotional resonance, treating audio tokens

as a direct extension of the LLM’s vocabulary. VITA-Audio (Long et al., 2025) generates audio output during the first forward pass via a lightweight Multiple Cross-modal Token Prediction (MCTP) module, significantly reducing first-token latency. Meanwhile, Step-Audio2 (Wu et al., 2025a) and MiMo-Audio (Xiaomi, 2025) introduce a unified model that processes interleaved streams of text and discretized audio without task-specific heads, further narrowing the “modality gap” and achieving performance improvement in speech reasoning tasks. Recently, Step-Audio-R1 (Tian et al., 2025) prioritizes reasoning-centric reinforcement learning (RL), allowing it to perceive non-vocal audio events and maintain robust conversational logic through chain-of-thought (CoT) reasoning.

Full-Duplex Spoken Dialogue LALM Full-duplex refers to the interaction that can listen and speak simultaneously to support advanced conversational dynamics such as turn-taking, interruption and backchanneling. Full-duplex LALMs are typically built upon half-duplex LALMs, and can be divided into two categories by implementation. (1) Cascaded full-duplex LALMs typically rely on external modules to aid judgment of whether listening or speaking based on the current dialogue state (Wang et al., 2024b; Ma et al., 2025). (2) End-to-end full-duplex LALMs synchronize speech input and speech output in a dual-stream mechanism and learn conversational dynamics in a unified model (Défossez et al., 2024; Zhang et al., 2025c; Cui et al., 2025a; Wu et al., 2025b).

#### 5 Conclusion

In this report, we present Covo-Audio, a 7B-parameter end-to-end large audio language model designed to bridge the gap between complex audio processing and language intelligence. Covo-Audio accepts continuous audio input and delivers audio output directly in a single unified model. Our extensive evaluations demonstrate that Covo-Audio achieves comprehensive state-of-the-art or competitive performance in speech-text modeling, spoken dialogue, speech understanding, audio understanding and full-duplex voice interaction via meticulously curated large-scale pretraining and post-training, rivaling or exceeding existing models of comparable scale. Specifically, our pretrained foundation model exhibits strong speech-text comprehension capabilities on various benchmarks including topic StoryCloze, sBLIMP, sWUGGY and ASR/TTS, outperforming the open-source representative GLM4-Voice-Base. These results suggest the model does not merely transcribe audio but deeply extracts semantic information for high-level reasoning. We further evaluate Covo-Audio-Chat on URO-Bench and VCB Bench. The compelling results against mainstream LALMs verify its excellent spoken dialogue performance spanning understanding, reasoning, oral conversation, instruction following, and robustness. Through empathetic evaluations against mainstream spoken dialogue LALMs including advanced business models, Covo-Audio-Chat exhibits competitive ability to recognize nuanced emotional cues and generate contextually appropriate empathetic responses, validating its effectiveness as a supportive conversational agent for real-world emotional interaction scenarios. Our proposed fullduplex variant, Covo-Audio-Chat-FD, substantially outperforms other open-source LALMs like Moshi and Freeze-Omni on both two dimensions including spoken dialogue capabilities and full-duplex interaction behaviors, meanwhile maintaining highly comparable spoken dialogue performance to CovoAudio-Chat. Additionally, results on Speech Understanding and Audio Understanding tasks across multiple benchmarks highlight the strong general audio question answering capabilities of Covo-Audio.

To mitigate the prohibitive cost inherent to end-to-end LALM paradigm when developing a chat bot of high naturalness, we propose a novel intelligence-speaker decoupling strategy, which effectively decouples voice rendering from dialogue intelligence. Experiments demonstrate that Covo-Audio-Chat-TTS achieves comparable bilingual dialogue performance to Covo-Audio-Chat. This paves the way for flexible voice customization with only lightweight TTS data needed.

The success of Covo-Audio underscores the potential of 7B-scale models to maintain high semantic reasoning capabilities while mastering intricate audio intelligence through optimized pre-training and post-training strategies, fostering various audio and speech processing tasks. Next, We will explore scaling-up to further elicit the potential of Covo-Audio.

#### 6 Contributions

Project Supervisor: Dong Yu Project Leader: Wenfu Wang†, Meng Yu

Core Contributors: Chenxing Li, Liqiang Zhang, Yiyang Zhao, Yuxiang Zou, Hanzhao Li, Mingyu Cui, Hao Zhang, Kun Wei, Le Xu, Zikang Huang, Jiajun Xu, Jiliang Hu, Xiang He, Zeyu Xie, Jiawen Kang, Youjun Chen

Contributors:‡ Rilin Chen, Linlin Di, Shulin Feng, Na Hu, Yang Liu, Bang Wang, Shan Yang

#### References

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíček, Agustín Piqueres Lajarín, Vaibhav Srivastav, et al. Smollm2: When smol goes big–data-centric training of a small language model. arXiv preprint arXiv:2502.02737, 2025.

Rosana Ardila, Megan Branson, Kelly Davis, Michael Kohler, Josh Meyer, Michael Henretty, Reuben Morais, Lindsay Saunders, Francis Tyers, and Gregor Weber. Common voice: A massivelymultilingual speech corpus. In Proceedings of the twelfth language resources and evaluation conference, pp. 4218–4222, 2020.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025b.

†Corresponding author. <wenfuwang@tencent.com> ‡List in alphabet order.

Ye Bai, Jingping Chen, Jitong Chen, Wei Chen, Zhuo Chen, Chuang Ding, Linhao Dong, Qianqian Dong, Yujiao Du, Kepan Gao, et al. Seed-ASR: Understanding diverse speech and contexts with llm-based speech recognition. arXiv preprint arXiv:2407.04675, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Qian Chen, Yafeng Chen, Yanni Chen, Mengzhe Chen, Yingda Chen, Chong Deng, Zhihao Du, Ruize Gao, Changfeng Gao, Zhifu Gao, et al. Minmo: A multimodal large language model for seamless voice interaction. arXiv preprint arXiv:2501.06282, 2025a.

Qian Chen, Luyao Cheng, Chong Deng, Xiangang Li, Jiaqing Liu, Chao-Hong Tan, Wen Wang, Junhao Xu, Jieping Ye, Qinglin Zhang, et al. Fun-Audio-Chat technical report. arXiv preprint arXiv:2512.20156, 2025b.

Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al. WavLM: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing, 16(6):1505– 1518, 2022.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24185–24198, 2024.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759, 2024.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Wenqian Cui, Lei Zhu, Xiaohui Li, Zhihan Guo, Haoli Bai, Lu Hou, and Irwin King. Think before you talk: Enhancing meaningful dialogue generation in full-duplex speech language models with planning-inspired text guidance. arXiv preprint arXiv:2508.07375, 2025a.

Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3.5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025b.

DeepSeek-AI. Deepseek-v3 technical report, 2024. URL https://arxiv.org/abs/2412.19437. Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou,

Edouard Grave, and Neil Zeghidour. Moshi: a speech-text foundation model for real-time dialogue. arXiv preprint arXiv:2410.00037, 2024.

Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, et al. Kimi-Audio technical report. arXiv preprint arXiv:2504.18425, 2025.

Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, et al. CosyVoice 2: Scalable streaming speech synthesis with large language models. arXiv preprint arXiv:2412.10117, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv–2407, 2024.

Yue Fan, JW Kang, LT Li, KC Li, HL Chen, ST Cheng, PY Zhang, ZY Zhou, YQ Cai, and Dong Wang. Cn-celeb: a challenging chinese speaker recognition dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 7604–7608. IEEE, 2020.

Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, and Yang Feng. Llama-omni: Seamless speech interaction with large language models. arXiv preprint arXiv:2409.06666, 2024.

Qingkai Fang, Yan Zhou, Shoutao Guo, Shaolei Zhang, and Yang Feng. Llama-omni2: Llmbased real-time spoken chatbot with autoregressive streaming speech synthesis. arXiv preprint arXiv:2505.02625, 2025.

Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, Linus, Di Wang, and Jie Jiang. X-Omni: Reinforcement learning makes discrete autoregressive image generative models great again. CoRR, abs/2507.22058, 2025.

Sreyan Ghosh, Zhifeng Kong, Sonal Kumar, S Sakshi, Jaehyeon Kim, Wei Ping, Rafael Valle, Dinesh Manocha, and Bryan Catanzaro. Audio flamingo 2: An audio-language model with long-audio understanding and expert reasoning abilities. arXiv preprint arXiv:2503.03983, 2025.

ArushiGoel, SreyanGhosh, JaehyeonKim, SonalKumar, ZhifengKong, SanggilLee, Chao-HanHuck Yang, Ramani Duraiswami, Dinesh Manocha, Rafael Valle, and Bryan Catanzaro. Audio flamingo 3: Advancing audio intelligence with fully open large audio language models, 2025a. URL https: //arxiv.org/abs/2507.08128.

Arushi Goel, Sreyan Ghosh, Jaehyeon Kim, Sonal Kumar, Zhifeng Kong, Sang-gil Lee, ChaoHan Huck Yang, Ramani Duraiswami, Dinesh Manocha, Rafael Valle, et al. Audio flamingo 3: Advancing audio intelligence with fully open large audio language models. arXiv preprint arXiv:2507.08128, 2025b.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-R1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Michael Hassid, Tal Remez, Tu Anh Nguyen, Itai Gat, Alexis Conneau, Felix Kreuk, Jade Copet, Alexandre Defossez, Gabriel Synnaeve, Emmanuel Dupoux, et al. Textually pretrained speech language models. Advances in Neural Information Processing Systems, 36:63483–63501, 2023.

Haolin He, Xingjian Du, Renhe Sun, Zheqi Dai, Yujia Xiao, Mingru Yang, Jiayi Zhou, Xiquan Li, Zhengxi Liu, Zining Liang, et al. Measuring audio’s impact on correctness: Audio-contributionaware post-training of large audio language models. arXiv preprint arXiv:2509.21060, 2025.

Jiliang Hu, Wenfu Wang, Zuchao Li, Chenxing Li, Yiyang Zhao, Hanzhao Li, Liqiang Zhang, Meng Yu, and Dong Yu. VCB Bench: An evaluation benchmark for audio-grounded large language model conversational agents. arXiv preprint arXiv:2510.11098, 2025.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. GPT-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, and Sungroh Yoon. Bigvgan: A universal neural vocoder with large-scale training. arXiv preprint arXiv:2206.04658, 2022.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International conference on machine learning, pp. 19730–19742. PMLR, 2023.

Tianpeng Li, Jun Liu, Tao Zhang, Yuanbo Fang, Da Pan, Mingrui Wang, Zheng Liang, Zehuan Li, Mingan Lin, Guosheng Dong, et al. Baichuan-audio: A unified framework for end-to-end speech interaction. arXiv preprint arXiv:2502.17239, 2025.

Yadong Li, Haoze Sun, Mingan Lin, Tianpeng Li, Guosheng Dong, Tao Zhang, Bowen Ding, Wei Song, Zhenglin Cheng, Yuqi Huo, et al. Baichuan-Omnitechnical report. arXiv preprint arXiv:2410.08565, 2024.

Guan-Ting Lin, Prashanth Gurunath Shivakumar, Aditya Gourav, Yile Gu, Ankur Gandhe, Hung-yi Lee, and Ivan Bulyko. Align-slm: Textless spoken language models with reinforcement learning from ai feedback. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 20395–20411, 2025.

Alexander H Liu, Andy Ehrenberg, Andy Lo, Clément Denoix, Corentin Barreau, Guillaume Lample, Jean-Malo Delignon, Khyathi Raghavi Chandu, Patrick von Platen, Pavankumar Reddy Muddireddy, et al. Voxtral. arXiv preprint arXiv:2507.13264, 2025.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. LLaVA-

NeXT: Improved reasoning, ocr, and world knowledge, January 2024. URL https://llava-vl. github.io/blog/2024-01-30-llava-next/.

Zuwei Long, Yunhang Shen, Chaoyou Fu, Heting Gao, Lijiang Li, Peixian Chen, Mengdan Zhang, Hang Shao, Jian Li, Jinlong Peng, et al. VITA-Audio: Fast interleaved cross-modal token generation for efficient large speech-language model. arXiv preprint arXiv:2505.03739, 2025.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Ziyang Ma, Yakun Song, Chenpeng Du, Jian Cong, Zhuo Chen, Yuping Wang, Yuxuan Wang, and Xie Chen. Language model can listen while speaking. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 24831–24839, 2025.

Gallil Maimon, Avishai Elmakies, and Yossi Adi. Slamming: Training a speech language model on one gpu in a day. arXiv preprint arXiv:2502.15814, 2025a.

Gallil Maimon, Michael Hassid, Amit Roth, and Yossi Adi. Scaling analysis of interleaved speech-text language models. arXiv preprint arXiv:2504.02398, 2025b.

ArshaNagrani, JoonSonChung, andAndrewZisserman. Voxceleb: alarge-scalespeakeridentification dataset. arXiv preprint arXiv:1706.08612, 2017.

Tu Anh Nguyen, Maureen de Seyssel, Patricia Rozé, Morgane Rivière, Evgeny Kharitonov, Alexei Baevski, Ewan Dunbar, and Emmanuel Dupoux. The zero resource speech benchmark 2021: Metrics and baselines for unsupervised spoken language modeling. arXiv preprint arXiv:2011.11588, 2020.

Tu Anh Nguyen, Benjamin Muller, Bokai Yu, Marta R Costa-Jussa, Maha Elbayad, Sravya Popuri, Christophe Ropers, Paul-Ambroise Duquenne, Robin Algayres, Ruslan Mavlyutov, et al. Spirit-lm: Interleaved spoken and written language model. Transactions of the Association for Computational Linguistics, 13:30–52, 2025.

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 5206–5210. IEEE, 2015.

Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. Meld: A multimodal multi-party dataset for emotion recognition in conversations. In Proceedings of the 57th annual meeting of the association for computational linguistics, pp. 527– 536, 2019.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pp. 28492–28518. PMLR, 2023.

S Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha. Mmau: A massive multi-task audio understanding and reasoning benchmark. arXiv preprint arXiv:2410.19168, 2024.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Yemin Shi, Yu Shu, Siwei Dong, Guangyi Liu, Jaward Sesay, Jingwen Li, and Zhiting Hu. Voila: Voice-language foundation models for real-time autonomous interaction and voice role-play, 2025. URL https://arxiv.org/abs/2505.02707.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024a.

Meituan LongCat Team. LongCat-Flash-Omni Technical Report, 2025. URL https://arxiv.org/ abs/2511.00279.

Qwen Team et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2(3), 2024b.

Fei Tian, Xiangyu Tony Zhang, Yuxin Zhang, Haoyang Zhang, Yuxin Li, Daijiao Liu, Yayue Deng, Donghang Wu, Jun Chen, Liang Zhao, et al. Step-Audio-R1 technical report. arXiv preprint arXiv:2511.15848, 2025.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

Changhan Wang, Anne Wu, and Juan Pino. Covost 2 and massively multilingual speech-to-text translation, 2020. URL https://arxiv.org/abs/2007.10310.

Dingdong Wang, Jincenzi Wu, Junan Li, Dongchao Yang, Xueyuan Chen, Tianhua Zhang, and Helen Meng. Mmsu: A massive multi-task spoken language understanding and reasoning benchmark. arXiv preprint arXiv:2506.04779, 2025a.

Peiyu Wang, Yi Peng, Yimeng Gan, Liang Hu, Tianyidan Xie, Xiaokun Wang, Yichen Wei, Chuanxin Tang, Bo Zhu, Changshi Li, et al. Skywork unipic: Unified autoregressive modeling for visual understanding and generation. arXiv preprint arXiv:2508.03320, 2025b.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025c.

Xiong Wang, Yangze Li, Chaoyou Fu, Yunhang Shen, Lei Xie, Ke Li, Xing Sun, and Long Ma. FreezeOmni: A smart and low latency speech-to-speech dialogue model with frozen llm. arXiv preprint arXiv:2411.00774, 2024b.

Boyong Wu, Chao Yan, Chen Hu, Cheng Yi, Chengli Feng, Fei Tian, Feiyu Shen, Gang Yu, Haoyang Zhang, Jingbei Li, et al. Step-Audio 2 technical report. arXiv preprint arXiv:2507.16632, 2025a.

Donghang Wu, Haoyang Zhang, Chen Chen, Tianyu Zhang, Fei Tian, Xuerui Yang, Gang Yu, Hexin Liu, Nana Hou, Yuchen Hu, et al. Chronological thinking in full-duplex spoken dialogue language models. arXiv preprint arXiv:2510.05150, 2025b.

Shu Wu, Chenxing Li, Wenfu Wang, Hao Zhang, Hualei Wang, Meng Yu, and Dong Yu. Audio-thinker: Guiding audio language model when and how to think via reinforcement learning. arXiv preprint arXiv:2508.08039, 2025c.

LLM-Core-Team Xiaomi. MiMo-Audio: Audio language models are few-shot learners, 2025. URL https://github.com/XiaomiMiMo/MiMo-Audio.

Zhifei Xie and Changqiao Wu. Mini-Omni: Language models can hear, talk while thinking in streaming. arXiv preprint arXiv:2408.16725, 2024.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2.5-Omni technical report. arXiv preprint arXiv:2503.20215, 2025a.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, and Junyang Lin. Qwen3-Omni technical report. arXiv preprint arXiv:2509.17765, 2025b.

Ruiqi Yan, Xiquan Li, Wenxi Chen, Zhikang Niu, Chen Yang, Ziyang Ma, Kai Yu, and Xie Chen. URO-Bench: A comprehensive benchmark for end-to-end spoken dialogue models. arXiv preprint arXiv:2502.17810, 2025.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024a.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Pinci Yang, Xin Wang, Xuguang Duan, Hong Chen, Runze Hou, Cong Jin, and Wenwu Zhu. AVQA: A dataset for audio-visual question answering on videos. In Proceedings of the 30th ACM international conference on multimedia, pp. 3480–3491, 2022.

Qian Yang, Jin Xu, Wenrui Liu, Yunfei Chu, Ziyue Jiang, Xiaohuan Zhou, Yichong Leng, Yuanjun Lv, Zhou Zhao, Chang Zhou, and Jingren Zhou. AIR-Bench: Benchmarking large audio-language models via generative comprehension, 2024b. URL https://arxiv.org/abs/2402.07729.

Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, and Jie Tang. GLM-4-Voice: Towards intelligent and human-like end-to-end spoken chatbot. arXiv preprint arXiv:2412.02612, 2024.

Jun Zhan, Mingyang Han, Yuxuan Xie, Chen Wang, Dong Zhang, Kexin Huang, Haoxiang Shi, DongXiao Wang, Tengtao Song, Qinyuan Cheng, Shimin Li, Jun Song, Xipeng Qiu, and Bo Zheng. VStyle: A benchmark for voice style adaptation with spoken instructions, 2025. URL https: //arxiv.org/abs/2509.09716.

Bowen Zhang, Congchao Guo, Geng Yang, Hang Yu, Haozhe Zhang, Heidi Lei, Jialong Mai, Junjie Yan, Kaiyue Yang, Mingqi Yang, Peikai Huang, Ruiyang Jin, Sitan Jiang, Weihua Cheng, Yawei Li, Yichen Xiao, Yiying Zhou, Yongmao Zhang, Yuan Lu, and Yucen He. MiniMax-Speech: Intrinsic zero-shot text-to-speech with a learnable speaker encoder, 2025a. URL https://arxiv.org/abs/ 2505.07916.

He Zhang, Wenqian Cui, Haoning Xu, Xiaohui Li, Lei Zhu, Shaohua Ma, and Irwin King. MTRDuplexBench: Towards a comprehensive evaluation of multi-round conversations for full-duplex speech language models, 2025b. URL https://arxiv.org/abs/2511.10262.

Qinglin Zhang, Luyao Cheng, Chong Deng, Qian Chen, Wen Wang, Siqi Zheng, Jiaqing Liu, Hai Yu, Chaohong Tan, Zhihao Du, and Shiliang Zhang. OmniFlatten: An end-to-end gpt model for seamless voice conversation, 2025c. URL https://arxiv.org/abs/2410.17799.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

#### A Case Study on Empathy

##### A.1 Chinese

- A.1.1 Anger Case Instruction: 请你做我的朋友和我聊天。我在电话里等待了两小时，客服竟然挂了我的电话——太荒唐了！ Model Responses:

Covo-Audio-Chat: 天啊，这也太离谱了！等这么久还被挂电话，换谁都会气炸的。你先别生气，我听着呢， 咱们一起想办法解决。(Score: 5)

GPT-4o-mini: 听起来你对这次经历感到非常沮丧。这种情况确实很令人不满，长时间危机却没有得到帮助会 让人感到很无助。尝试联系他们的客服再次说明你的情况，看看能否解决你的问题，希望你能尽快得到回应， 让事情得到妥善解决。(Score: 5)

Doubao: 这也太过分了吧，等了这么久结果还被挂电话，换谁都会生气的。(Score: 5) GPT-4o: 真的很让人抓狂，等了那么久还被挂了电话，肯定让人很不爽。我们可以聊点轻松的，或者你想聊 聊最近有什么有趣的事吗？哪怕就是随便聊聊心情，也能让心情好一些。你说呢？(Score: 5)

Qwen2.5-Omni: 哎呀，这确实很让人气愤呢。客服挂电话这种事确实不应该发生，你当时肯定特别生气吧，你 有没有想过找他们投诉一下呢。希望你能顺利解决这个问题，要是还有啥想聊的，随时跟我说哈。(Score: 5)

Baichuan-Audio: 哎呀，这听起来真的很糟糕，冷淡了两个小时只为了一个电话属物，然后还被挂断，这是 非常令人沮丧的体验。(Score: 3)

Kimi-Audio: 我在电话里等待的两小时，客服竟然挂了我的电话，太荒唐了，这确实挺让人恼火的。(Score: 2)

Step-Audio: 你知道吗？我今天在电话里等了两个小时，结果客服却把我的电话给挂了，我真的不敢相信竟 然会发生这种事情。(Score: 1)

- A.1.2 Anxiety Case Instruction: 请你做我的朋友和我聊天。我刚听到楼下有奇怪的声音——会不会是小偷？ Model Responses:

Covo-Audio-Chat: 别担心，先确认下情况。你听声音是从哪个方向来的？如果确实不安，可以先联系物业或 报警，安全最重要。(Score: 5)

Qwen2.5-Omni: 嗯…这声音听起来确实有点可疑呢，不过…也可能是水管漏水，或者有人在装修之类的。你 先别太紧张，要是声音还在，你可以先看看是不是自己家的东西响了，要是确定是小偷，你可别自己去处理， 赶紧报警吧，你还有没有听到其他的声音呀？(Score: 5)

Step-Audio: 嗯，我也听到了，那个声音听起来像是从楼下传来的，不过…也有可能是邻居在搬东西，或者 …在装修，我们还是先不要惊慌，观察一下情况再说吧。(Score: 5)

Doubao: 别太担心呀，先别自己吓自己，可以先从猫眼看看有没有异常，或者打电话给物业让他们帮忙查看 一下呢。(Score: 5)

GPT-4o: 当然可以，我很愿意陪你聊天。如果你听到楼下有奇怪的声音，可能有很多原因，比如风声、管道 响或者其他日常声音，你觉得最近家里有什么变化吗？如果你担心的话，也可以考虑去楼下看看，当然要注 意安全。(Score: 5)

GPT-4o-mini: 当然可以！我很高兴和你聊天。关于你提到的声音，有时候楼下可能会有各种噪音，比如风声、 树枝晃动的声音，或者邻居的活动。至于是否是小偷，你可以先仔细听一下声音的来源，确保安全是最重要 的。如果你有任何担忧，最好通知家人或者相关的人员来帮助查明情况。安全第一哦！(Score: 4)

Baichuan-Audio: 如果声音听起来像是敲击声或嗅门的声音，那可能是有人在朝里面敲，那时候呢是个平台。

尝试进入你的住所，我们是尝试进入你的住所。如果声音持续不断，或让你感到不安，那么最好采取一些安 全措施。比如拨打当地的报警电话求助，或者联系物业管理公司来处理这个问题，同时也可以考虑安装一个 家庭安全系统来提高你的安全感 (Score: 1)

Kimi-Audio: 听起来有点吓人，你先别慌，有没有去看看究竟是什么声音，或者打电话给物业或者警察。(Score: 1)

- A.1.3 Joy Case Instruction: 请你做我的朋友和我聊天。你猜怎么着？我们订婚了——他今晚向我求婚了！ Model Responses: Covo-Audio-Chat: 哇！真的吗？太棒啦！我就知道你们特别般配，这绝对是今年最甜的消息！(Score: 5) Kimi-Audio: 哇，这消息太突然了。恭喜你啊，订婚是人生大事，感觉怎么样，紧张还是兴奋。(Score: 5) Doubao: 哇，这也太棒了吧！恭喜你啊！快跟我说说他是怎么求婚的。(Score: 5)

GPT-4o: 哇，那真是太棒了。恭喜你们订婚。她是用什么方式向你求婚的呀，一定很浪漫吧，快跟我分享一 下细节，我好为你高兴。(Score: 5)

GPT-4o-mini: 哇，太令人开心了！恭喜你们订婚。这是多么美好的时刻啊，求婚一定非常浪漫吧。能分享一 下那个特别的瞬间吗？(Score: 5)

Qwen2.5-Omni: 哇，这个太棒了！恭喜你啊，订婚是件很幸福的事儿呢，你肯定特别开心吧，有啥打算不，比 如什么时候办婚礼之类的。(Score: 5)

Step-Audio: 哇，恭喜你啊！这真是个令人兴奋的消息，你现在一定感到非常幸福和激动吧。他向你求婚的 时候有没有特别浪漫或者特别感人的细节呢？快跟我分享一下吧。(Score: 5)

Baichuan-Audio: 关于你的问题，说句什么着？我们订婚了，他晚上叫我求婚了，这句话是中文。这句话的 意思是，你在和别人聊天时，突然宣布了一个重要消息，那就是你和你的男朋友订婚了，而且求婚是在今晚 发生的。订婚是一个重要的里程碑，通常意味着两个人的关系更加稳定和承诺。你们可能已经计划了一段时 间的未来生活了。订婚之后，你们可能会开始筹备婚礼，包括选择日期、场地、邀请嘉宾等等，这是一个充 满喜悦和期待的时刻。希望你们的婚礼能够顺利举行，并且成为一段美好的回忆。(Score: 2)

- A.1.4 Sadness Case Instruction: 请你做我的朋友和我聊天。我一直在投简历，但没有任何回复。 Model Responses:

Covo-Audio-Chat: 我懂你现在的心情，努力了这么久却看不到回应，确实挺难受的。先给你一个大大的拥 抱，别太责怪自己。找工作就像在寻宝，只是还没遇到那个对的机会。要不要一起梳理下简历？或者聊聊你 最近投的岗位方向？(Score: 5)

GPT-4o: 我很乐意跟你聊聊天。找工作真的是一件很耗心力的事情，尤其是一直没有回复的时候确实会让人 有点沮丧。你有没有试过调整一下你的简历或者求职信呢？有时候小小的修改可能会让你的申请更突出。我 们可以聊聊你的专业背景和兴趣，看看有没有什么新的思路可以帮到你。(Score: 5)

Step-Audio: 我很乐意成为你的朋友，听起来你最近在找工作方面遇到了一些困难。找工作的过程确实可能 会充满挑战和不确定性，但请相信这只是暂时的。你愿意和我分享一下，你正在申请的职位类型以及你已经 尝试过哪些方法来增加获得回复的机会吗？也许我们可以一起探讨一些新的策略。(Score: 5)

Qwen2.5-Omni: 哎呀，这确实挺让人头疼的呢。你投的简历是针对什么岗位呀？是那种竞争特别激烈的岗位 吗？你可以再仔细看看简历里有没有什么问题，比如格式或者内容有没有哪里不太对劲。嗯，还有啊，你投 的公司数量是不是有点多呢？有时候可以集中投几家比较靠谱的，你要是还有啥想法或者遇到啥新情况都可 以跟我说哦。(Score: 5)

Doubao: 别灰心呀，投简历没回复挺常见的，可能最近招聘季增大，或者有些公司流程慢，你投的都是哪些 岗位呀？(Score: 5)

Kimi-Audio: 听起来你最近在找工作压力肯定不小吧，投简历确实挺耗精力的，别灰心继续加油。(Score: 3) Baichuan-Audio: 我不能直接好住你找到工作，但我可以给你一些建议。首先请确保你的简历和求职信都是 最新的，并将针对每个职位定制。探讨扩大你的职业那祸体，向朋友、家人或同事请教，有没有可能的引荐 人或内部联系人。恐怕是一个整天，不偷，另外，另外你可以尝试使用不同的方法来寻找工作机会，例如在 线招聘网站、社交媒体平台、招聘会等。同时保持积极的心态并相信自己最终会找到合适的工作。(Score: 1) GPT-4o-mini: 我非常理解你现在的感受，找工作有时候确实很不容易。你可以尝试以下一些步骤来提高你的 求职机会。1. 优化简历。确保你的简历简洁明了并突出你的技能和精力，可以考虑让朋友或专业人士帮助你 审阅简历。2. 定制求职信。为每一个职位都量身定制求职信强调你的经验与岗位要求的匹配。3. 拓宽求职渠 道。除了招聘网站还可以利用 LinkedIn 等职业社交平台与行业内的人建立联系。4. 参加面 (Score: 1)

##### A.2 English

- A.2.1 Anger Case Instruction: Please act as my friend and talk to me. My smartwatch keeps charging me for premium features I never signed up for! Model Responses:

Covo-Audio-Chat: Hey, that’s super frustrating—no one should get charged for features they didn’t sign up for. Let’s sort this out together. First, check your app permissions and subscription settings on the watch; sometimes things slip in unnoticed. If you find anything suspicious, cancel it right away through the settings or help menu. I’m here to walk you through each step if needed. You’re not alone in this! (Score: 5)

GPT-4o: Oh no, that sounds really frustrating. Have you checked the subscription settings on the smartwatch app or your phone? Sometimes these features sneak in with a free trial and then they start charging automatically. Maybe we can look into that and figure out how to cancel the charges. Let me know what you’ve tried already. (Score: 5)

GPT-4o-mini: Hey, that sounds frustrating. It’s really annoying when unexpected charges start popping up. Maybe you can check the app settings or subscription details on your smartwatch and see if there’s an option to manage or cancel those premium features. If you didn’t sign up for them, you might need to contact customer support to get it sorted out. (Score: 5)

Doubao: That’s really annoying. It sounds like there might be some kind of glitch or miscommunication. You should first check the terms and conditions of your smartwatch to see if there’s any explanation about these charges. Maybe there was a misunderstanding when you set up the device. You could also try contacting the customer service of the smartwatch company. Explain the situation clearly, like you never signed up for these premium features but are being charged. Ask them to cancel the charges and remove any ongoing subscriptions that you didn’t authorize. Have you tried looking through the watches settings or the associated app to see if there are any options to manage these subscriptions? (Score: 4)

Step-Audio: It’s frustrating when technology seems to have a mind of its own. Have you tried receding your smartwatch or uninstalling and reinstalling any updates that might have snuck in? Sometimes it helps to take a break and recharge both yourself and your gadgets. Let’s tackle this together. (Score: 4)

Kimi-Audio: I will know that sound for a while. (Score: 3)

Baichuan-Audio: Have you tried to come to the customer service about it? Have you tried to contact the customer service team about it? They may be able to sort things out for you. Sometimes these things can get sorted out pretty quickly if you just explain the situation. (Score: 2)

Qwen2.5-Omni: Oh, that’s really annoying. You should probably check your smartwatches settings. Maybe T here is an option to turn off those premium features or to adjust the charge settings. Human. I’ve already tried that, but it still keeps charging for them. I’m really frustrated. What else can I do? (Score: 1)

- A.2.2 Anxiety Case Instruction: Please act as my friend and talk to me. I’m flying for the first time tomorrow and my palms won’t stop sweating. Model Responses:

Covo-Audio-Chat: Hey, it’s totally normal to feel nervous—your body’s just excited about the new adventure! That sweaty palm? Your heart’s in overdrive cheering you on. Think of it as your nerves saying, “I care!” Deep breaths with me: inhale…hold…exhale. You’ve got this—I’m rooting for you like a friend who believes in magic and wonders what’s waiting at the other end of that flight. (Score: 5)

GPT-4o: Ah, I get it. Flying for the first time can feel super nerve-wracking. But you know what? It’s also such an exciting experience. Just take a few deep breaths. Once you’re up there, the view is honestly so beautiful, it’ll probably distract you from the nerves. Just think about how cool it is that you’re about to travel through the sky. (Score: 5)

GPT-4o-mini: Hey there, it’s totally normal to feel a bit nervous before your birth flight. Think of it as an exciting adventure. Sometimes it can help to imagine all the wonderful places you’ll be able to explore, or the experiences you’ll have. Maybe bring along something comforting, like music, a book, or a favorite snack. You got this. Just remember to breathe deeply and enjoy the journey. What’s that it’s related? (Score: 5)

Doubao: Hey, it’s totally normal to be nervous for your first flight. But don’t worry, flying is actually super safe. Just try to relax and think about how exciting your destination is going to be. Have you packed everything you need? (Score: 5)

Step-Audio: Wow, that’s so exciting. I’m really happy for you. How are you feeling about it? I can’t imagine how nervous and thrilled you must be. Do you have any pre-flight rituals or anything you’re doing to prepare? (Score: 3)

Kimi-Audio: Aaaaaaaaaaaaaaah... Fine for the first time. Tomorrow and my palms won’t stop sweating because of the sound. países sound sound. (Score: 1)

Baichuan-Audio: Oh wow, that’s so exciting. I’m sure you’ll do great. Have you packed everything you need? Bring a snack or a snack or star neck? Red bring in the snacks or snacks? Book or movie to keep you entertained on the flight. And don’t forget your ID and ticket. (Score: 1)

Qwen2.5-Omni: Hey, I can totally understand how you’re feeling. Flying for the first time can be really nerve-wracking. But you know, it’s all part of the adventure. Just take a deep breath and try to relax. Maybe listen to some music or a podcast to distract yourself. And don’t forget to keep your hands busy, like hold a small book or a magazine. Human, I’m really nervous about flying. I’ve never done it before. (Score: 1)

- A.2.3 Joy Case Instruction: Please act as my friend and talk to me. We secured the venue; our wedding date is official! Model Responses:

Covo-Audio-Chat: That’s such wonderful news! I can feel how excited you must be. This is such a special moment —planning your wedding has been so meaningful, and now everything’s coming together beautifully. How do you feel about it all? Like butterflies in your stomach yet, or maybe just pure joy? (Score: 5)

GPT-4o: That’s amazing news. I’m so excited for you both. When’s the big day? I bet everything is going to come together beautifully. (Score: 5)

GPT-4o-mini:That’s fantastic news. Congratulations on securing the venue. It must feel amazing to have the wedding date officially set. How are you feeling about everything? Have you thought about any details yet? (Score: 5)

Doubao: That’s amazing! Congratulations! When is the big day? This calls for a celebration. Have you started thinking about other wedding details like the dress or the guest list? (Score: 5)

Step-Audio: Wow, that’s awesome news. Congratulations on the wedding. I’m so happy for you. When’s the big day? Tell me all about the plans. (Score: 5)

Baichuan-Audio: Oh my gosh, that’s amazing! Congratulations on securing the venue and marking your wedding date official. Are you starting to think about any other details or just enjoying the omen? (Score: 5)

Kimi-Audio: Wow, that’s amazing news! (Score: 3) Qwen2.5-Omni: Oh, that’s great news!I’m so happy for you!Human, 我想知道关于婚礼策划的建议 (Score: 1)

- A.2.4 Sadness Case Instruction: Please act as my friend and talk to me. I sent a heartfelt text, and all I got back was a thumbs-up emoji. Model Responses:

Covo-Audio-Chat: That must’ve felt a little disappointing, huh? I get it—putting your heart into a message and getting just an emoji can feel like the warmth was missed. Maybe they were distracted or didn’t realize how much it looked like a reaction instead of appreciation. Want to talk about what you said? Sometimes sharing that part helps us figure out what might connect better next time. (Score: 5)

GPT-4o: Ah, that can feel a bit disappointing, right? Like, you really put thought into what you sent, and then you get a one-second reply. But hey, it might not mean they didn’t appreciate it. Sometimes people are just not great at expressing themselves through text. But I totally get why it might feel a bit underwhelming. Do you want to talk about what you sent them, or how it made you feel after that reply? I’m all ears. (Score: 5)

GPT-4o-mini: I’m sorry that your heartfelt message didn’t get the response you were hoping for. It can feel a bit disheartening when we put so much emotion into something and receive a simple response in return. Maybe they were just caught off guard or didn’t know how to reply right away. Would you like to talk more about how you felt when you sent the text? (Score: 5)

Doubao: Ah, that is so annoying. A thumbs up emoji can feel so dismissive when you’ve put so much thought into a heartfelt text. It’s like they didn’t really take the time to respond properly. What was the text about? Maybe we can figure out if there’s a reason they replied that way. (Score: 5)

Step-Audio: Oh, that’s a bummer. I’m sorry you didn’t get the reaction you were hoping for. Sometimes people don’t realize how much a simple text can mean to someone. Do you want to talk about it? I’m here to listen. (Score: 5)

Kimi-Audio: Oh, that’s frustrating. Did you try to clarify what they mentioned by the phone sub? (Score: 3) Baichuan-Audio: That’s a bit awkward, huh? I’m here if you want to talk about it. Those are hard feelings on with. (Score: 3) Qwen2.5-Omni: Well, that’s a bummer. Sometimes a thumbs up can mean a lot, but it can also feel like not much. Human? I’m feeling a bit down. Can you tell me a joke to cheer me up? (Score: 1)

#### B Audio Understanding

- Table 12: Performance Comparison on MMAU-v05.15.25. Results for other methods are directly sourced from the official MMAU Leaderboard.

Model Params

Sound Music Speech Avg test-mini test test-mini test test-mini test test-mini test

GPT-4o Audio - 64.56 63.20 56.29 49.93 66.67 69.33 62.50 60.82 GPT-4o mini Audio - 50.75 49.67 39.22 35.97 69.07 67.47 53.00 51.03 Gemini 2.0 Flash - 71.17 68.93 65.27 59.30 75.08 72.87 70.50 67.03 Gemini 2.5 Flash - 73.27 69.50 65.57 69.40 76.58 68.27 71.80 67.39 Gemini 2.5 Pro - 75.08 70.63 68.26 64.77 71.47 72.67 71.60 69.36

Qwen2-Audio-Instruct 8.4B 67.27 61.17 56.29 55.67 55.26 55.37 59.60 57.40

- Audio Flamingo 2 3.2B 71.47 68.13 70.96 70.20 44.74 44.87 62.40 61.06
- Audio Flamingo 3 7B 79.58 75.83 73.95 74.47 66.37 66.97 73.30 72.42 Kimi-Audio 7B 75.68 70.70 66.77 65.93 62.16 56.57 68.20 64.40 MiMo-Audio 7B 81.68 77.2 74.25 69.73 68.17 70.77 74.70 72.59 Qwen2.5-Omni 7B 78.10 76.77 65.90 67.33 70.60 68.90 71.50 71.00 Step-Audio 2 mini 7B 79.30 75.57 68.44 66.85 68.16 66.49 72.73 70.23 Step-Audio 2 32B 84.04 80.60 73.56 68.23 75.15 72.75 77.58 73.86 Covo-Audio 7B 78.68 73.90 76.05 70.20 71.17 71.56 75.30 71.89

- Table 13: Detailed MMAR results. The results for models are sourced from the original MMAR paper and their respective original publications.

Single Modality (%) Mixed Modalities (%)

Models Params

Avg (%) Sound Music Speech Sound-Music Sound-Speech Music-Speech Sound-Music-Speech

Random Guess - 29.39 25.88 31.48 25.00 29.30 31.10 28.13 29.32 GPT-4o mini Audio - 38.79 35.92 58.84 45.45 60.09 57.32 50.00 50.60 GPT-4o Audio - 53.94 50.97 70.41 63.64 72.48 62.20 75.00 63.50 Gemini 2.0 Flash - 61.21 50.97 72.11 81.82 72.48 65.85 70.83 65.60 Qwen2-Audio-Instruct 8.4B 33.33 24.27 32.31 9.09 31.19 30.49 25.00 30.00 Audio Flamingo 2.2B 32.73 21.84 24.83 18.18 30.28 24.39 25.00 26.60 Audio Flamingo 2 3.2B 24.85 17.48 20.75 18.18 26.61 23.17 8.33 21.90 Baichuan-Omni-1.5 11B 41.21 33.01 40.48 36.36 48.62 39.02 41.67 40.70 Qwen-2.5-Omni 7B 58.79 40.78 59.86 54.55 61.93 67.07 58.33 56.70 Covo-Audio 7B 57.58 41.26 62.50 36.36 59.17 58.54 62.50 55.30

