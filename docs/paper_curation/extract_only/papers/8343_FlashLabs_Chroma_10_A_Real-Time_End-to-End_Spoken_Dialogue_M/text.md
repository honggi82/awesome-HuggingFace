# arXiv:2601.11141v1[cs.SD]16Jan2026

## FlashLabs Chroma 1.0: A Real-Time End-to-End Spoken Dialogue Model with Personalized Voice Cloning

### Tanyu Chen*, Tairan Chen*, Kai Shen*, Zhenghua Bao*,†, Zhihui Zhang, Man Yuan, Yi Shi† FlashLabs

Correspondence: zhenghua.bao@flashlabs.ai, chroma@flashlabs.ai

### Abstract

Recent end-to-end spoken dialogue systems leverage speech tokenizers and neural audio codecs to enable LLMs to operate directly on discrete speech representations. However, these models often exhibit limited speaker identity preservation, hindering personalized voice interaction. In this work, we present Chroma 1.0, the first open-source, real-time, end-toend spoken dialogue model that achieves both low-latency interaction and high-fidelity personalized voice cloning. Chroma achieves sub-second end-to-end latency through an interleaved text-audio token schedule (1:2) that supports streaming generation, while maintaining high-quality personalized voice synthesis across multi-turn conversations. Our experimental results demonstrate that Chroma achieves a 10.96% relative improvement in speaker similarity over the human baseline, with a Real-Time Factor of 0.43, while maintaining strong reasoning and dialogue capabilities. Our code and models are publicly available at GitHub and HuggingFace.

### 1 Introduction

Building real-time, interactive speech dialogue systems remains challenging. Most deployed systems follow a cascaded pipeline: input speech is first transcribed into text via automatic speech recognition (ASR), the transcript is then processed by a downstream large language model (LLM), and the generated response is finally converted back to speech by a text-to-speech (TTS) module. While effective for offline or latency-tolerant scenarios, such pipelines incur high end-to-end latency, suffer from error propagation, and tend to lose paralinguistic information such as speaker identity, speaking rate, timbre, emotion, and prosody. These limitations become especially limiting in real-time

*Equal contribution. †Corresponding author.

conversational settings, where naturalness, responsiveness, and speaker fidelity are essential.

Recent advances in speech tokenization and neural codecs (Hsu et al., 2021; Défossez et al., 2022; Ji et al., 2024) enable LLMs to operate directly on discrete speech representations, giving rise to speech-to-speech (S2S) systems that bypass explicit intermediate transcription. End-to-end large audio language models (LALMs) emerged following the success of GPT-4o (Hurst et al., 2024), which demonstrated the feasibility of end-to-end speech processing without explicit textual intermediates. Subsequent works (Défossez et al., 2024; Zeng et al., 2024; Xu et al., 2025a,b; Nguyen et al., 2025; Huang et al., 2025a,b; Wu et al., 2025) have adopted this principle, processing and generating speech directly through diverse architectural strategies: interleaved speech-text token sequences (Nguyen et al., 2025), unified audio-text representations (Huang et al., 2025a,b; Wu et al., 2025; Zeng et al., 2024), or parallel dual-stream architectures that decouple semantic reasoning from acoustic generation (Xu et al., 2025a,b).

However, existing end-to-end LALMs still face several critical limitations in both speech understanding and speech generation. Early models such as Spirit LM (Nguyen et al., 2025) and GLM4-Voice (Zeng et al., 2024) primarily focus on aligning semantic content from speech to text, often showing limited ability to capture paralinguistic cues that are essential for natural interaction. While more recent audio-understanding models like Qwen2-Audio (Chu et al., 2024) and KimiAudio (Ding et al., 2025) can comprehend such paralinguistic information, they generate only textual outputs, failing to leverage this understanding for expressive speech generation. Moreover, current S2S systems typically prioritize dialogue quality over personalized voice fidelity. While speech systems capable of voice cloning, such as VALL-E (Wang et al., 2023), the CosyVoice se-

[Figure 1]

- Figure 1: System workflow of Chroma 1.0. Chroma takes speech as input and produces speech as output, maintaining consistent speaker identity throughout the conversation.

ries (Du et al., 2024a,b, 2025), and industrial solutions (e.g., Qwen-TTS1 and ElevenLabs2), achieve high-quality speaker adaptation, they lack real-time streaming capabilities with consistent voice cloning across multi-turn conversations. Conversely, realtime dialogue models (Défossez et al., 2024; Xie and Wu, 2024a,b; Fang et al., 2024) sacrifice finegrained speaker control for low latency.

To address these limitations, we propose Chroma 1.0, the first open-source, real-time endto-end spoken dialogue model that achieves both low-latency interaction and high-fidelity personalized voice cloning. Our main contributions are as follows:

- • A streaming architecture that tightly couples speech understanding with speech generation through semantic state representations, enabling sub-second end-to-end latency.
- • High-fidelity voice cloning that conditions the generation model on audio embeddings from just a few seconds of reference audio, achieving 10.96% relative improvement in speaker similarity over human baseline.
- • An interleaved text-audio token schedule (1:2) that enables synchronized generation of acoustic codes with incremental text output, achieving real-time streaming synthesis.
- • Strong reasoning and dialogue capabilities with only 4B parameters, demonstrating competitive performance across understanding, reasoning, and oral conversation tasks.

- 1https://qwenlm.github.io/blog/qwen-tts/
- 2https://elevenlabs.io/voice-cloning

We release the complete codebase, training pipeline, and pretrained model weights to support reproducibility and further research3 4.

### 2 Related Works

2.1 Cascaded Speech–Text Pipelines vs. End-to-End S2S Conversation

Cascaded approaches remain widely deployed due to the maturity of ASR, LLM, and TTS modules, as well as established engineering practices (Zhang et al., 2023a; Radford et al., 2023; Wang et al., 2023; Chu et al., 2023; Tan et al., 2024; Huang et al., 2024; Lin et al., 2024; Fang et al., 2024). They offer flexibility but accumulate latency and errors, and often discard paralinguistic cues once speech is reduced to text, limiting downstream speaker fidelity and emotional expressiveness. In contrast, end-to-end S2S dialogue systems avoid explicit transcription by learning to map directly between discrete speech units. SeamlessM4T (Duquenne et al., 2023) established a multilingual S2S baseline, while GPT-4o (Hurst et al., 2024) demonstrated the commercial viability of native audio processing without intermediate text conversions. Recent research efforts have explored diverse architectural strategies for end-to-end speech modeling. Moshi (Défossez et al., 2024) introduces a full-duplex streaming architecture for real-time dialogue, while Spirit LM (Nguyen et al., 2025) employs interleaved speech-text token sequences to maintain multimodal alignment. GLM-4-Voice (Zeng et al., 2024)

- 3https://github.com/FlashLabs-AI-Corp/

FlashLabs-Chroma

- 4https://huggingface.co/FlashLabs/Chroma-4B

|Chroma Decoder<br><br>|0|
|---|
<br><br>|1|
|---|
<br><br>|2|
|---|
<br><br>···<br><br>|N|
|---|
<br><br>|1|
|---|
<br><br>|2|
|---|
<br><br>···<br><br>|N|
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|3|
|---|
<br><br>|N-1|
|---|
|
|---|

| |
|---|

| |
|---|

Audio Token

Pad Token

CodecDecoder

Audio Hidden

Pad Hidden

[Figure 2]

| |
|---|

| |
|---|

Audio Embedding

Text Token

Sample

Text Hidden

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

| |
|---|

|0|
|---|

|0|
|---|

|0|
|---|

|0|
|---|

|0|
|---|

|0|
|---|

|0|
|---|

|0|
|---|

|0|
|---|

|0|
|---|

|0|
|---|

###### ···

···

···

···

Chroma Backbone

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

···

···

Reference Text Reference Audio ······

Chroma Reasoner

- Figure 2: Overall architecture of Chroma 1.0. The Reasoner outputs text tokens and hidden states. These form an

interleaved text–audio embedding sequence (1:2) consumed by the Backbone to generate coarse acoustic codes c0t and hidden states ht. The Decoder predicts the remaining RVQ levels c1:t N−1, and the Codec Decoder reconstructs the full codebook sequence into a continuous waveform, enabling high-fidelity S2S generation.

and the early Step-Audio series (Huang et al., 2025a,b) adopt unified token representations that map speech and text into a shared vocabulary space, with Step-Audio-2 (Wu et al., 2025) further advancing generation quality using retrieval-augmented generation (RAG) and reinforcement learning techniques. Qwen2.5-Omni (Xu et al., 2025a) and Qwen3-Omni (Xu et al., 2025b) propose parallel dual-stream (Thinker-Talker) architectures that decouple semantic reasoning from acoustic generation. Specifically, Qwen3-Omni employs a multicodebook token prediction (MTP) module in the Talker to predict residual codebooks per frame. Models focused primarily on audio understanding—such as Qwen2-Audio (Chu et al., 2024) and Kimi-Audio (Ding et al., 2025)—generate textual responses, demonstrating the capability to comprehend paralinguistic information from speech inputs.

TTS from approximately 3 seconds of reference audio while preserving paralinguistic cues. VALLE X (Zhang et al., 2023b) extends this capability across languages. Diffusion and flow-matching approaches further advance fidelity and control. StyleTTS-2 (Li et al., 2023) introduces style diffusion achieving strong naturalness and zero-shot adaptation, NaturalSpeech 3 (Ju et al., 2024) factorizes content, prosody, and timbre for state-ofthe-art similarity, and Voicebox (Le et al., 2023) demonstrates high-speed non-autoregressive flowmatching. Open-source systems like CosyVoice series (Du et al., 2024a,b, 2025), and OpenVoice (Qin et al., 2023) demonstrate practical cloning with improved content consistency and fine-grained control, with later iterations adding streaming support. Commercial platforms (e.g. Elevenlabs) further validate few-second cloning at production scale.

### 3 Model Architecture

- 2.2 Voice Cloning and Personalized Speech Synthesis

Chroma 1.0 is built upon an end-to-end spoken dialogue architecture designed to achieve high-quality, real-time voice interaction through tight coupling of speech perception and language understanding. As illustrated in Fig. 2, the system consists of two integrated subsystems: (i) the Chroma Reasoner, responsible for multimodal input comprehension and textual response generation; and (ii) a speech

A major advancement in voice cloning came with neural codec language models (NCLMs), which cast TTS as discrete acoustic token generation. VALL-E (Wang et al., 2023) demonstrated that modeling EnCodec codes with a large conditional language model yields natural, expressive zero-shot

synthesis pipeline composed of the Chroma Backbone for acoustic modeling, the Chroma Decoder for audio codebook decoding, and the Chroma Codec Decoder for waveform reconstruction.

#### 3.1 Chroma Reasoner

The Chroma Reasoner, built upon the Thinker module (Xu et al., 2025a), performs multimodal understanding and semantic representation construction. It processes both text and audio inputs through the standard Qwen2-Audio encoding pipeline (Chu et al., 2024), generating high-level semantic representations that capture both linguistic content and acoustic characteristics.

The Reasoner employs a cross-modal attention mechanism to fuse text and audio features. The encoded representations are unified into a sequence of hidden states with temporal alignment through Time-aligned Multimodal Rotary Position Embedding (TM-RoPE) (Xu et al., 2025a). This fusion enables the model to leverage prosodic and rhythmic cues from speech alongside textual semantics, enhancing both dialogue understanding and contextual modeling for subsequent speech synthesis.

#### 3.2 Chroma Backbone

The Chroma Backbone adopts a 1B-parameter variant of the LLaMA (Touvron et al., 2023) architecture, designed to generate speech that matches the timbre of a given reference audio. To enable high-fidelity voice cloning, we encode the reference audio and its corresponding transcript into embedding prompts using CSM-1B5 and prepend them to the input sequence, explicitly conditioning the model on the target speaker’s acoustic characteristics. During inference, to ensure strict alignment between the audio output and the Reasoner’s text modality while maintaining low parameter count, we apply a shared token embedding strategy: the Reasoner’s token embeddings and hidden states are fed into the Backbone as unified textual context.

To support efficient streaming generation, we interleave text tokens with audio code tokens c0 at a fixed ratio of 1 : 2, meaning each text token is paired with two audio codes. This mechanism enables the Backbone to autoregressively generate audio sequences in parallel with the Reasoner’s incremental text generation, allowing the system to produce outputs without waiting for complete text sequences. Consequently, the Time-to-First-Token

5https://github.com/SesameAILabs/csm

(TTFT) is significantly reduced, enabling improved real-time interaction.

#### 3.3 Chroma Decoder

To substantially accelerate inference while preserving generation quality, we introduce a lightweight module—the Chroma Decoder—responsible for generating the remaining acoustic codes (c1t,...,cNt −1), rather than having the Backbone produce all codebooks directly. The Chroma Decoder is implemented as a variant of the LLaMA architecture with approximately 100M parameters. Unlike the Backbone, this module does not rely on the full history of text inputs or reference audio context; instead, it performs frame-synchronous inference conditioned only on the Backbone outputs at the current time step, thereby greatly reducing computational overhead associated with long-context processing.

Specifically, at each time step t, the Chroma Decoder takes as input the hidden-state features ht and the initial audio codebook c0t produced by the Backbone, and autoregressively generates the remaining RVQ codebooks cit (i = 1,...,N − 1) within each frame using level-specific projection heads conditioned on previously generated levels. This decoupled design not only reduces inference latency but also enables the Chroma Decoder to enrich finegrained acoustic attributes, such as prosody and articulation details, building upon the coarse semantic representation from the Backbone.

#### 3.4 Chroma Codec Decoder

As the final acoustic reconstruction module, the Chroma Codec Decoder maps the discrete codebook sequence into a continuous, high-fidelity speech waveform. At each time step, the module concatenates the coarse codebook (c0) and the refined acoustic codebooks (c1,...,cN−1) generated by the Chroma Decoder to form the complete discrete acoustic representation. Architecturally, this module follows the decoder design of the Mimi vocoder (Défossez et al., 2024) and employs a causal convolutional neural network (Causal CNN), ensuring strict temporal causality during waveform reconstruction to support streaming generation.

To meet real-time interaction requirements, we employ 8 codebooks (N = 8). This configuration significantly reduces the autoregressive refinement steps required by the Chroma Decoder, thereby improving inference efficiency.

#### 3.5 Training Datasets

Publicly available datasets lack high-quality speech dialogue data that meet our model’s requirements for semantic understanding and reasoning capabilities. To address this limitation, we propose a data generation pipeline that leverages the synergistic collaboration between LLMs and TTS systems.

Pipeline Workflow. The generation process consists of two stages: (1) Text Generation: User questions are fed into a Reasoner-like LLM module to generate corresponding textual responses. (2) Speech Synthesis: The textual responses are synthesized into speech using a TTS system, with timbre characteristics matching the reference audio. This synthesized speech serves as the training target, enabling the Backbone and Decoder modules to learn voice cloning and acoustic modeling.

#### 3.6 Training Objective

Our training strategy optimizes two primary components: the Backbone and the Decoder, while keeping the Reasoner frozen as a feature extractor. For each audio-text pair, the Reasoner provides fixed text embeddings and multimodal hidden states that serve as semantic and prosodic conditioning. The Backbone is trained to autoregressively predict the first layer of coarse acoustic codes (c0). To ensure causal alignment, it attends only to the prefix of the acoustic codes and the corresponding Reasoner representations. This objective enables the model to capture long-term temporal structure and align acoustic generation with the text progression. The Decoder refines the coarse acoustic representation by predicting the remaining Residual Vector Quantization (RVQ) levels (c1:N−1). Conditioned on the Backbone’s coarse code and hidden state, the Decoder operates via an intra-frame autoregressive process. This factorization allows it to progressively enhance acoustic fidelity while maintaining consistency with the coarse trajectory established by the Backbone. More details can be found in Appendix C.

### 4 Experiments

#### 4.1 Experimental Setup

Datasets. We evaluate our model on multiple benchmark datasets to assess different aspects of speech generation quality. For general performance evaluation, we use the CommonVoice dataset (Ardila et al., 2020), which provides diverse speakers and recording conditions for mea-

suring speaker similarity. To assess voice cloning capability, we conduct subjective experiments measuring naturalness and voice fidelity between generated and reference audio. To evaluate the model’s ability to understand and reason about input speech, we utilize a subset of URO-Bench (Yan et al., 2025), which provides speech-based questionanswering tasks requiring semantic comprehension. Unless otherwise specified, all experiments were conducted on an NVIDIA H200 GPU.

Training Configuration. Our implementation is based on PyTorch 2.7.1. We employ the AdamW optimizer (Loshchilov and Hutter, 2017) with a learning rate of 5 × 10−5 and a per-device batch size of 4. The model is trained for 100K steps on 8 NVIDIA H200 GPUs (141GB memory each), achieving convergence in approximately 6 hours. Gradient clipping with a maximum norm of 1.0 is applied to ensure training stability.

Evaluation Metrics. We conduct comprehensive evaluation using both objective and subjective metrics, covering speech quality, naturalness, speaker similarity, and system efficiency. For objective evaluation, we employ Speaker Similarity (SIM), Time-to-First-Token (TTFT), and Real-Time Factor (RTF). For subjective evaluation, we conduct Comparative Mean Opinion Score (CMOS) tests through human listeners. Our SIM evaluation is based on the SEED-TTS-EVAL framework6.

Speaker Similarity (SIM) evaluates voice cloning quality by measuring how closely the generated speech matches the target speaker’s voice characteristics. We use WavLM-Large (Chen et al., 2022a), a self-supervised speech representation model fine-tuned on speaker verification tasks (Chen et al., 2022b), to extract 192dimensional speaker embeddings from both generated and reference audio. Similarity is computed as the cosine similarity between these embeddings.

Comparative Mean Opinion Score (CMOS) measures naturalness (NCMOS) and speaker similarity (SCMOS) through pairwise comparisons. For NCMOS, participants compare two systems and select from four options: “A sounds more natural”, “B sounds more natural”, “About the same”, or “Hard to tell”. For SCMOS, participants first listen to reference audio, then compare which of two synthesized samples better matches the reference

6https://github.com/BytedanceSpeech/ seed-tts-eval

#### Model SIM ↑

Human Baseline 0.73 F5-TTS 0.64 Seed-TTS 0.76 FireRedTTS-2 0.66 Step-Audio-TTS 0.66 CosyVoice 3 0.72 Chroma 1.0 0.817

Table 1: Performance comparison of speech models on zero-shot voice cloning. Higher SIM indicates better speaker similarity.

speaker’s characteristics using the same four-option scale. The order of samples A and B is randomized to avoid position bias. CMOS scores are computed as the mean preference difference, where positive values indicate preference for our system.

Latency Metrics. We measure system efficiency using two complementary metrics: Time-to-FirstToken (TTFT) and Real-Time Factor (RTF). TTFT measures system responsiveness by recording the time elapsed from receiving the input to generating the first audio token, which is critical for natural conversation flow. RTF measures computational efficiency as the ratio of generation time to audio duration. An RTF below 1.0 indicates the system can generate speech faster than real-time playback. Both metrics are essential for evaluating real-time interactive performance, with lower values indicating better efficiency.

#### 4.2 Objective Voice Cloning Evaluation

We evaluate Chroma’s voice cloning capability by measuring SIM, the primary focus of this work. Given that Chroma currently generates Englishonly audio, we assess its performance in a zero-shot setting using English samples from the CommonVoice dataset, following the protocol established by Seed-TTS (Anastassiou et al., 2024). Our model is evaluated at its native sampling rate of 24kHz. Table 1 presents the comparative results against current state-of-the-art speech models and a human baseline.

As shown in Table 1, most contemporary TTS models achieve comparable speaker similarity scores, typically ranging 1–9% below the human

7Chroma operates at 24kHz sample rate, which better preserves speaker characteristics compared to 16kHz used by other models.

Metric Chroma ElevenLabs Deuce NCMOS 24.4% 57.2% 18.3% SCMOS 40.6% 42.4% 17.0%

- Table 2: Comparative evaluation between Chroma and ElevenLabs.

Preference Reference ElevenLabs Total 4 (8.0%) 46 (92.0%)

- Table 3: Comparison between ElevenLabs and reference (ground truth) audio.

baseline. A notable exception is Seed-TTS, which slightly exceeds the human baseline. Chroma, on the other hand, outperforms all competing models, achieving a relative improvement of 10.96% over the human baseline. This result suggests that Chroma effectively captures fine-grained paralinguistic features, enabling the generation of high-fidelity, personalized speech with exceptional speaker identity preservation.

#### 4.3 Subject Voice Cloning Evaluation

We conducted comparative experiments between Chroma and ElevenLabs, a state-of-the-art commercial voice cloning system. The experiments measured both NCMOS and SCMOS. We used their eleven_multilingual_v2 API8 to generate outputs. For each dimension, we selected 15 samples and generated corresponding outputs using both models, resulting in 30 comparative samples evaluated across 12 independent sessions. Responses of “About the same” or “Hard to tell” were grouped into a “Deuce” category, representing cases where no clear preference could be established.

Table 2 presents the comparative results. For NCMOS, ElevenLabs demonstrated superior performance, receiving 57.2% of preferences compared to Chroma’s 24.4% (18.3% Deuce), indicating significantly more natural-sounding speech. However, for SCMOS, where participants first listened to reference audio before comparing synthesized outputs, the results were remarkably close: ElevenLabs received 42.4% compared to Chroma’s 40.6% (17.0% Deuce). This near-tie, with only 1.8 percentage points difference, suggests comparable capability in capturing speaker-specific characteristics.

This phenomenon reflects fundamental differences in system design. ElevenLabs employs a two-

8https://elevenlabs.io/docs/overview/models

##### Component TTFT (ms) Avg Latency per Frame (ms) Total Duration (s)

Reasoner 119.12 26.03 3.74 Backbone 8.48 8.75 4.27 Decoder 19.27 17.56 8.57 Codec Decoder – 3.08 2.99

Overall Generation Latency 146.87 52.34 16.58 Generated Audio Length – – 38.80

Generation RTF 0.43

- Table 4: Latency breakdown for speech generation. The system achieves TTFT of 146.87ms and RTF of 0.43.

stage approach: first creating a voice profile from reference audio, then using it for TTS generation. In contrast, Chroma uses an end-to-end approach that directly processes reference audio in a single pass. While ElevenLabs optimizes for naturalness and clarity, its two-stage pipeline may lose finegrained speaker characteristics during voice profile extraction. Chroma’s end-to-end architecture preserves these nuanced features by maintaining direct access to reference audio throughout generation.

To investigate this trade-off, we conducted an additional experiment comparing ElevenLabs outputs directly with reference audio. Table 3 shows results from 5 sessions with 10 samples each, where evaluators chose which audio sounded more natural and human-like. Remarkably, evaluators overwhelmingly preferred ElevenLabs-generated audio (92.0%) over actual human recordings (8.0%). This reveals a critical insight: subjective listener preference does not necessarily align with speaker similarity.

This finding reframes our SCMOS interpretation. While ElevenLabs achieved marginally higher SCMOS preference (42.4% vs 40.6%), this advantage likely reflects listeners’ inherent bias toward naturalness rather than superior speaker fidelity. The overwhelming preference (92%) for synthesized audio over ground truth recordings demonstrates that naturalness dominates speaker similarity in subjective evaluations. Consequently, Chroma’s competitive SCMOS performance suggests stronger speaker fidelity than the raw percentages indicate, as it maintains competitiveness despite prioritizing faithful reproduction of speaker characteristicsincluding natural imperfections, speaking rate variations, and subtle prosodic features—over the perceptual naturalness that ElevenLabs optimizes for.

#### 4.4 Practical Generation Latency

The current Chroma architecture does not support batch processing, therefore, we measured its la-

tency under concurrency 1. Table 4 presents the latency breakdown for each component during response generation. The total generation latency is the sum of all individual component latencies. We demonstrate the measured latency on a real example generating a 38.80-second audio response.

Prefilling Strategy. Before generating speech, we prefill the prompt text and prompt audio to reduce TTFT. Specifically, we encode and concatenate the prompt text and prompt audio to obtain prompt embeddings, which are then fed into the Backbone to perform prefill computation and generate the corresponding KV cache (Pope et al., 2023). By constructing the KV cache before the generation phase, the model avoids reprocessing prompt content during inference, enabling immediate autoregressive generation upon receiving user input. This strategy effectively reduces response latency and substantially improves real-time interaction performance.

Component-wise Latency Analysis. The Reasoner TTFT (119.12ms) represents the time from processing input data to generating the first audio token. The Backbone TTFT (8.48ms) measures the time to produce the first audio hidden states after receiving the audio token from the Reasoner. The Decoder then generates the remaining 7 codebooks (c1:7) to capture fine-grained acoustic features, taking an average of 17.56ms per frame. Finally, the Codec Decoder reconstructs the waveform from the complete codebook sequence. Since we concatenate every 4 frames before passing them to the Codec Decoder for efficient batch processing, TTFT is not applicable for this component.

The overall system achieves a TTFT of 146.87ms, demonstrating sub-second responsiveness suitable for real-time interaction. The average latency per frame is 52.34ms, and with an RTF of 0.43, the system generates speech significantly faster than real-time playback, enabling smooth

Understanding Reasoning Oral Conv.

Models LLM Scale

Overall Rep.↑ Sum.↑ Gaokao↑ Storal↑ Truth.↑ Gsm8k↑ MLC↑ Alpaca↑ Common↑ Wild.↑

GLM-4-Voice 9B 90.95 91.07 64.47 73.80 59.28 30.93 57.82 80.77 63.07 78.76 69.09 LLaMA-Omni 8B 45.62 80.68 16.06 50.65 45.13 3.89 44.44 64.36 58.40 72.19 48.14 Freeze-Omni 7B 70.89 78.87 26.29 57.74 46.95 2.81 42.56 52.23 48.70 55.80 48.28

Mini-Omni 0.5B 5.07 32.20 0 23.25 25.06 0 2.82 30.99 29.80 31.42 18.06 Mini-Omni2 0.5B 8.10 40.06 0.66 28.49 26.92 0 6.97 34.81 30.70 36.43 21.31 SLAM-Omni 0.5B 12.26 66.21 1.32 36.95 34.65 0 21.85 48.98 41.03 52.61 31.59

Chroma (ours) 4B 69.05 74.12 38.61 71.14 51.69 22.74 60.26 60.47 62.07 64.24 57.44

- Table 5: Task accomplishment scores for end-to-end spoken dialogue models across understanding, reasoning, and oral conversation capabilities. Bold values indicate Chroma’s performance, which remains competitive across all dimensions despite using only 4B parameters.

streaming generation.

Real-Time Factor. The generation RTF is calculated by dividing the total generation latency by the length of the generated audio:

16.58s 38.80s ≈ 0.43. (1)

Tgeneration Taudio

RTF =

=

An RTF of 0.43 indicates that Chroma generates speech 2.3× faster than real-time playback, demonstrating strong performance for streaming applications. This efficiency allows the system to maintain low latency even during extended multi-turn conversations.

#### 4.5 Reasoning and Dialogue Capabilities

While Chroma’s primary focus is high-fidelity voice cloning, we evaluate its general dialogue capabilities to demonstrate that personalized voice generation does not compromise cognitive and conversational abilities. Table 5 compares Chroma against existing end-to-end spoken dialogue models across three dimensions using the basic track of URO-Bench (Yan et al., 2025).

Despite being optimized for voice cloning, Chroma demonstrates strong performance across all capabilities. In reasoning tasks, Chroma consistently achieves second-best performance: 71.14% on Storal (vs. 73.80% for GLM-4-Voice), 51.69% on TruthfulQA (vs. 59.28%), and 22.74% on GSM8K (vs. 30.93%). Notably, GLM-4-Voice uses 9B parameters, more than twice Chroma’s 4B scale. In oral conversation, Chroma achieves the highest scores on MLC (60.26%) and CommonVoice (62.07%), demonstrating natural dialogue flow. For understanding tasks, Chroma achieves competitive scores of 69.05% on repetition and 74.12% on summarization, ranking second among all models.

Critically, Chroma is the only model in this comparison with personalized voice cloning capability. All other models focus exclusively on dialogue and reasoning, without the ability to clone specific speaker characteristics. This makes Chroma’s competitive performance particularly noteworthy: it maintains strong cognitive and conversational abilities while simultaneously supporting high-fidelity voice personalization, a capability absent in all compared systems. Furthermore, with only 4B parameters, Chroma achieves efficiency advantages over larger models (7B-9B) and delivers superior performance compared to smaller models (0.5B) across all evaluated tasks.

### 5 Conclusion

We present Chroma 1.0, the first open-source, real-time end-to-end spoken dialogue model that achieves both low-latency interaction and highfidelity personalized voice cloning. Trained entirely on synthetic speech-to-speech data, Chroma achieves significant improvements in speaker similarity from only a few seconds of reference audio while maintaining real-time performance. Comparative evaluations demonstrate competitive performance against state-of-the-art commercial systems in voice cloning, alongside strong reasoning and dialogue capabilities. The system’s efficient architecture enables smooth streaming generation suitable for natural multi-turn conversations. By combining low-latency streaming with high-fidelity voice cloning, Chroma opens new possibilities for accessible, personalized voice AI applications across diverse domains. We release our code and models to facilitate further research in personalized spoken dialogue systems.

### 6 Acknowledgements

We thank Yusen Lin, Kaiwen Zhou, and Liujie Zhen for their contributions to the early stages of this work. We are grateful to the volunteers who participated in our human evaluation experiments.

### References

Philip Anastassiou, Jiawei Chen, Jitong Chen, Yuanzhe Chen, Zhuo Chen, Ziyi Chen, Jian Cong, Lelai Deng, Chuang Ding, Lu Gao, and 1 others. 2024. Seed-tts: A family of high-quality versatile speech generation models. arXiv preprint arXiv:2406.02430.

Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, and 1 others. 2022. Speecht5: Unifiedmodal encoder-decoder pre-training for spoken language processing. In Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: Long papers), pages 5723–5738.

Rosana Ardila, Megan Branson, Kelly Davis, Michael Kohler, Josh Meyer, Michael Henretty, Reuben Morais, Lindsay Saunders, Francis Tyers, and Gregor Weber. 2020. Common voice: A massivelymultilingual speech corpus. In Proceedings of the twelfth language resources and evaluation conference, pages 4218–4222.

Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, and 1 others. 2022a. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing, 16(6):1505–1518.

Zhengyang Chen, Sanyuan Chen, Yu Wu, Yao Qian, Chengyi Wang, Shujie Liu, Yanmin Qian, and Michael Zeng. 2022b. Large-scale self-supervised speech representation learning for automatic speaker verification. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6147–6151. IEEE.

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, and 1 others. 2024. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759.

Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, and Jingren Zhou. 2023. Qwen-audio: Advancing universal audio understanding via unified large-scale audiolanguage models. arXiv preprint arXiv:2311.07919.

Raffel Colin. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21.

Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. 2022. High fidelity neural audio compression. arXiv preprint arXiv:2210.13438.

Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. 2024. Moshi: a speechtext foundation model for real-time dialogue. arXiv preprint arXiv:2410.00037.

Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, and 1 others. 2025. Kimi-audio technical report. arXiv preprint arXiv:2504.18425.

Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, and 1 others. 2024a. Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. arXiv preprint arXiv:2407.05407.

Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, and 1 others. 2025. Cosyvoice 3: Towards in-the-wild speech generation via scaling-up and post-training. arXiv preprint arXiv:2505.17589.

Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, and 1 others. 2024b. Cosyvoice 2: Scalable streaming speech synthesis with large language models. arXiv preprint arXiv:2412.10117.

Paul-Ambroise Duquenne, H Elsahar, H Gong, K Heffernan, J Hoffman, C Klaiber, and 1 others. 2023. Seamlessm4t—massively multilingual and multimodal machine translation.

Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, and Yang Feng. 2024. Llama-omni: Seamless speech interaction with large language models. arXiv preprint arXiv:2409.06666.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM transactions on audio, speech, and language processing, 29:3451–3460.

Ailin Huang, Bingxin Li, Bruce Wang, Boyong Wu, Chao Yan, Chengli Feng, Heng Wang, Hongyu Zhou, Hongyuan Wang, Jingbei Li, and 1 others. 2025a. Step-audio-aqaa: a fully end-to-end expressive large audio language model. arXiv preprint arXiv:2506.08967.

Ailin Huang, Boyong Wu, Bruce Wang, Chao Yan, Chen Hu, Chengli Feng, Fei Tian, Feiyu Shen, Jingbei Li, Mingrui Chen, and 1 others. 2025b. Step-audio: Unified understanding and generation in intelligent speech interaction. arXiv preprint arXiv:2502.11946.

Rongjie Huang, Mingze Li, Dongchao Yang, Jiatong Shi, Xuankai Chang, Zhenhui Ye, Yuning Wu, Zhiqing Hong, Jiawei Huang, Jinglin Liu, and 1 others. 2024. Audiogpt: Understanding and generating

speech, music, sound, and talking head. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 23802–23804.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, and 1 others. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Shengpeng Ji, Ziyue Jiang, Wen Wang, Yifu Chen, Minghui Fang, Jialong Zuo, Qian Yang, Xize Cheng, Zehan Wang, Ruiqi Li, and 1 others. 2024. Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling. arXiv preprint arXiv:2408.16532.

Zeqian Ju, Yuancheng Wang, Kai Shen, Xu Tan, Detai Xin, Dongchao Yang, Yanqing Liu, Yichong Leng, Kaitao Song, Siliang Tang, and 1 others. 2024. Naturalspeech 3: Zero-shot speech synthesis with factorized codec and diffusion models. arXiv preprint arXiv:2403.03100.

Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, and 1 others. 2023. Voicebox: Text-guided multilingual universal speech generation at scale. Advances in neural information processing systems, 36:14005–14034.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 7871–7880.

Yinghao Aaron Li, Cong Han, Vinay Raghavan, Gavin Mischler, and Nima Mesgarani. 2023. Styletts 2: Towards human-level text-to-speech through style diffusion and adversarial training with large speech language models. Advances in Neural Information Processing Systems, 36:19594–19621.

Guan-Ting Lin, Cheng-Han Chiang, and Hung-yi Lee. 2024. Advancing large language models to capture varied speaking styles and respond properly in spoken conversations. arXiv preprint arXiv:2402.12786.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Tu Anh Nguyen, Benjamin Muller, Bokai Yu, Marta R Costa-Jussa, Maha Elbayad, Sravya Popuri, Christophe Ropers, Paul-Ambroise Duquenne, Robin Algayres, Ruslan Mavlyutov, and 1 others. 2025. Spirit-lm: Interleaved spoken and written language model. Transactions of the Association for Computational Linguistics, 13:30–52.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang,

Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2023. Efficiently scaling transformer inference. Proceedings of machine learning and systems, 5:606–624.

Zengyi Qin, Wenliang Zhao, Xumin Yu, and Xin Sun.

2023. Openvoice: Versatile instant voice cloning. arXiv preprint arXiv:2312.01479.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741.

Xu Tan, Jiawei Chen, Haohe Liu, Jian Cong, Chen Zhang, Yanqing Liu, Xi Wang, Yichong Leng, Yuanhao Yi, Lei He, and 1 others. 2024. Naturalspeech: End-to-end text-to-speech synthesis with humanlevel quality. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(6):4234–4245.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, and 1 others. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, and 1 others. 2023. Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint arXiv:2301.02111.

Boyong Wu, Chao Yan, Chen Hu, Cheng Yi, Chengli Feng, Fei Tian, Feiyu Shen, Gang Yu, Haoyang Zhang, Jingbei Li, and 1 others. 2025. Step-audio 2 technical report. arXiv preprint arXiv:2507.16632.

- Zhifei Xie and Changqiao Wu. 2024a. Mini-omni: Language models can hear, talk while thinking in streaming. arXiv preprint arXiv:2408.16725.
- Zhifei Xie and Changqiao Wu. 2024b. Mini-omni2: Towards open-source gpt-4o with vision, speech and duplex capabilities. arXiv preprint arXiv:2410.11190.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, and 1 others. 2025a. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, and 19 others. 2025b. Qwen3-omni technical report. Preprint, arXiv:2509.17765.

Ruiqi Yan, Xiquan Li, Wenxi Chen, Zhikang Niu, Chen Yang, Ziyang Ma, Kai Yu, and Xie Chen. 2025. Uro-bench: A comprehensive benchmark for end-to-end spoken dialogue models. arXiv preprint arXiv:2502.17810.

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, and 1 others. 2022. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5.

Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, and Jie Tang. 2024. Glm-4-voice: Towards intelligent and human-like end-to-end spoken chatbot. arXiv preprint arXiv:2412.02612.

Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, and Xipeng Qiu. 2023a. Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities. arXiv preprint arXiv:2305.11000.

Ziqiang Zhang, Long Zhou, Chengyi Wang, Sanyuan Chen, Yu Wu, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, and 1 others. 2023b. Speak foreign languages with your own voice: Crosslingual neural codec language modeling. arXiv preprint arXiv:2303.03926.

Appendix

- A Limitations and Future Work

The current system does not incorporate external tool use or task-specific post-training techniques such as reinforcement learning from human feedback (RLHF) (Ouyang et al., 2022) or direct preference optimization (DPO) (Rafailov et al., 2023). Integrating such methods could further improve dialogue quality, instruction following, and user preference alignment, particularly for conversational naturalness and contextual appropriateness.

Recent work on multi-codebook token prediction (MTP), such as Qwen3-Omni (Xu et al., 2025b), reduces first-packet latency by predicting residual codebooks in parallel rather than sequentially. Investigating whether MTP can be integrated into Chroma’s decoder stage without compromising voice cloning fidelity presents a promising avenue for further latency reduction.

Although Chroma’s speech reasoner supports multilingual input (currently Chinese and English), the system generates speech output only in English. Extending codec training and decoder modules to support multilingual output generation would broaden the system’s applicability across diverse linguistic contexts. Cross-lingual voice cloning, where input and output languages differ while preserving speaker identity, remains an important research direction.

While Chroma’s Backbone employs a decoderonly architecture aligning with recent trends in speech language modeling, encoder-decoder architectures have demonstrated strong performance across multiple domains, including machine translation (Colin, 2020), text generation (Lewis et al., 2020), text-to-image generation (Yu et al., 2022), and spoken language processing (Ao et al., 2022). These architectures may offer distinct advantages in controllability, cross-modal alignment, and explicit separation of understanding and generation processes. Exploring encoder-decoder designs for speech-to-speech dialogue could provide complementary benefits to our current approach, particularly for scenarios requiring fine-grained control over both semantic content and acoustic properties in voice cloning and streaming generation.

- B Ethical Considerations

While Chroma demonstrates substantial advancements in voice cloning technology, it also intro-

duces important ethical considerations. The capability to generate high-fidelity, personalized speech from minimal reference audio raises several risks, including the potential for impersonation, fraud, and the creation of misleading or harmful audio content without an individual’s consent.

To mitigate these risks, we recommend adopting strong technical and policy-based safeguards, such as:

- • Requiring explicit and verifiable consent for any form of voice cloning,
- • Developing and deploying reliable synthetic speech detection mechanisms,
- • Enforcing clear usage policies and access controls to prevent misuse,
- • Investigating watermarking or traceability techniques for generated audio.

When developed and used responsibly, voice cloning systems like Chroma can provide meaningful benefits, such as supporting voice restoration for individuals with speech impairments, enabling inclusive accessibility applications, and facilitating creative or personalized communication tools. We encourage the research community to advance ethical guidelines, regulatory frameworks, and robust detection technologies in parallel with progress in speech synthesis systems to ensure their safe and responsible deployment.

### C Training Objective

During training, the Reasoner is frozen and functions solely as a feature provider. For each training pair consisting of an audio sample xaudio and its corresponding transcription xtext, the Reasoner produces text embeddings Etext ∈ RT×d and multimodal hidden states Hreasoner ∈ RT×d, where T is the length of the text sequence and d is the hidden dimension. These representations remain fixed throughout optimization and serve as semantic and prosodic conditioning for the downstream acoustic models.

Conditioned on reference audio xref-audio and reference text xref-text, the Backbone autoregressively predicts a coarse discrete acoustic code sequence

c0 = {c0t | t = 1,...,L}, c0t ∈ {1,...,V },

where L denotes the number of audio frames and V is the codebook size.

The Chroma Decoder further refines each frame by producing the remaining N − 1 residual quantization levels of an RVQ representation. For each level j ∈ {1,...,N − 1}, the Decoder predicts

cj = {cjt | t = 1,...,L}, cjt ∈ {1,...,V }.

We define the complete N-level discrete acoustic representation as

c0:t N−1 = (c0t,c1t,...,cNt −1). Backbone Loss. To maintain causal alignment between text and audio generation, the Backbone at time t is allowed to attend only to prefix information. We define the causal conditioning set

z<t = c0<t, Etext,<t, Hreasoner,<t ,

where c0<t denotes the autoregressively generated acoustic prefix and Etext,<t, Hreasoner,<t represent the Reasoner-provided prefix signals.

The Backbone predicts the coarse code sequence c0 = {c0t}Lt=1, and its loss is defined as

L

1 L

log p c0t | z<t . (2)

Lbackbone = −

t=1

This objective trains the Backbone to capture inter-frame temporal structure and produce a coarse acoustic representation aligned with the text progression.

Decoder Loss. Given the coarse code c0t and the Backbone hidden state hBt , the Chroma Decoder refines each frame t via an intra-frame autoregressive process. Specifically, it predicts the residual quantization levels

c1:t N−1 = (c1t,c2t,...,cNt −1), cjt ∈ {1,...,V }.

The conditional distribution over refinement levels factorizes as

N−1

pθ cjt | c0t, hBt , c1:t j−1 ,

pθ c1:t N−1 | c0t, hBt =

j=1

(3)

where c1:t j−1 = (c1t,...,cjt−1) denotes the prefix for teacher forcing.

The training objective for the Chroma Decoder is the negative log-likelihood across all frames and refinement levels:

1 L

Ldecoder = −

L

t=1

N−1

log pθ cjt | c0t, hBt , c1:t j−1 .

j=1

(4)

This objective encourages the Decoder to progressively refine each frame’s acoustic representation while remaining consistent with the Backbone coarse code c0t and the contextual information encoded in hBt .

#### C.1 Training Strategy

We adopt a two-stage training strategy to stabilize optimization and progressively strengthen the refinement of acoustic representations.

In the first stage, we jointly train the Backbone and Decoder by setting the loss weight to λ = 0.5. This balanced weighting encourages the model to learn both the coarse acoustic code distribution and the residual quantization levels, helping the system establish consistent semantic-acoustic alignment in the early phase of training.

In the second stage, we freeze the Backbone parameters and set λ = 1, thereby shifting optimization entirely to the Decoder. This fine-tuning phase focuses on refining the higher-level quantization layers, enabling the model to capture finegrained speech characteristics such as timbre nuances, prosodic variations, and articulatory details. As a result, the final model achieves improved voice cloning fidelity and enhanced overall speech naturalness.

