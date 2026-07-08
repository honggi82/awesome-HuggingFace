# arXiv:2603.18090v2[cs.SD]20Mar2026

[Figure 1]

OpenMOSS

[Figure 2]

## MOSS-TTS Technical Report

SII-OpenMOSS Team*

Abstract

This technical report presents MOSS-TTS, a speech generation foundation model built on a scalable recipe: discrete audio tokens + autoregressive modeling + large-scale pretraining. Built on MOSS-Audio-Tokenizer, a causal Transformer tokenizer that compresses 24kHz audio to 12.5fps with variable-bitrate RVQ and unified semantic–acoustic representations, we release two complementary generators: MOSS-TTS, which emphasizes structural simplicity, scalability, and long-context/control-oriented deployment, and MOSS-TTS-Local-Transformer, which introduces a frame-local autoregressive module for higher modeling efficiency, stronger speaker preservation, and a shorter time to first audio. Across multilingual and open-domain settings, MOSS-TTS supports zero-shot voice cloning, token-level duration control, phoneme-/pinyin-level pronunciation control, smooth code-switching, and stable long-form generation. This report summarizes the design, training recipe, and empirical characteristics of the released models.

Homepage: https://mosi.cn/models/moss-tts Online Demo: https://huggingface.co/spaces/OpenMOSS-Team/MOSS-TTS AI Studio: https://studio.mosi.cn/voice-synthesis Hugging Face: https://huggingface.co/collections/OpenMOSS-Team/moss-tts GitHub: https://github.com/OpenMOSS/MOSS-TTS

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

### 1 Introduction

Text-to-speech (TTS) has evolved from task-specific pipelines into a broader paradigm of speech generation that is expected to behave like a foundation model: it should generalize across speakers, languages, speaking styles, and acoustic conditions; support controllable and low-latency synthesis; and remain stable over longform content [1–4]. Recent progress increasingly resembles the scaling trajectory of large language models, where model capacity and data scale unlock emergent capabilities beyond narrow benchmarks [5, 6].

At the same time, scaling speech generation is not a simple matter of “bigger models.” Modern approaches must reconcile competing requirements in representation learning and pretraining: (i) the discrete token representation must be compact enough for efficient sequence modeling yet expressive enough to preserve both semantic content and fine-grained acoustics; (ii) the generative model must remain stable over long sequences while staying compatible with streaming constraints; and (iii) the training signal must scale across diverse, noisy, real-world data without relying on brittle cascaded supervision. Much of the recent literature addresses these tensions by introducing multiple intermediate targets, external semantic teachers, refinement stages, or post-hoc alignment. Such designs can be effective, but they often complicate scaling because each additional module introduces a new supervision contract, new failure modes, and new latency budgets [2, 7–10].

∗Full contributors can be found in the Contributors section.

This report argues for a return to the core of speech generation: learn a high-quality audio tokenizer, train an autoregressive (AR) model over its tokens, and pretrain at scale. Concretely, we pursue the recipe discrete tokens + AR modeling + large-scale pretraining, and show that it provides a clean and scalable path to strong quality and controllability in practice. The key intuition is that a sufficiently capable tokenizer turns speech generation into a token prediction problem with a single, universal modeling objective—much like language modeling—thereby making it easier to scale data, compute, and downstream capabilities without continuously expanding the model stack.

MOSS-TTS combines three core components. (1) A high-quality audio tokenizer. We build on MOSSAudio-Tokenizer [11], a causal Transformer-based discrete tokenizer designed for large-scale AR modeling. It supports variable-bitrate residual vector quantization (RVQ), compressing 24kHz audio to 12.5fps and enabling streaming-friendly, frame-level encoding and decoding, while preserving high-fidelity reconstruction and semantically informative tokens. Unlike approaches that depend on external pretrained audio encoders or multi-stage distillation [7, 8, 12–14], MOSS-Audio-Tokenizer is trained end-to-end to jointly optimizeacousticreconstructionandsemanticalignment, aimingtomaximizescalabilityandminimizeinherited bottlenecks.

- (2) Large-scale, high-quality pretraining data. We build a large-scale, high-quality data pipeline that converts raw open-domain recordings into trainable single-speaker assets with cross-consistency gating (speaker consistency, language consistency, and transcript validity). The resulting corpus spans millions of hours in total, with the majority consisting of carefully filtered multilingual TTS-style supervision and targeted supplements for voice cloning and controllability. This data-centric foundation is essential for robustness across domains (podcasts, audiobooks, broadcast & news, film & drama, commentary, and online content) and for multilingual and code-switching behavior.
- (3) Refined discrete-token modeling for speech generation. On top of the tokenizer, we study and deploy discrete AR modeling strategies that remain efficient and stable for long-form synthesis. To serve both research reproducibility and practical deployment constraints, we explore two architectures with explicit tradeoffs. The Delay-Pattern model (MOSS-TTS) uses a single Transformer backbone with multiple prediction heads and an RVQ-aware delay schedule, prioritizing structural simplicity, scalability, and a clean longcontext operating point. The Global-Latent + Local Transformer model (MOSS-TTS-Local-Transformer) introduces an additional frame-local autoregressive module that is more complex but more learning-efficient, yielding stronger speaker preservation at smaller scale and a shorter time to first audio.

These components yield a practical speech generation foundation model with a broad capability set, including zero-shot voice cloning, token-level duration control, phoneme-/pinyin-level pronunciation control, multilingual synthesis with smooth code-switching (notably between Chinese and English), and stable longform generation up to hour-scale outputs.

Contributions. This technical report makes the following contributions:

- • We present MOSS-TTS, a discrete-token autoregressive speech generation foundation model built on a scalable discrete + AR + pretraining recipe.
- • We integrate and analyze MOSS-Audio-Tokenizer [11] as a universal, streaming-compatible audio tokenizer with variable bitrate and unified semantic-acoustic representations.
- • We present a large-scale, high-quality data pipeline that supports training on millions of hours of data and enables robust multilingual pretraining and controllable synthesis behavior.
- • WereleaseandcomparetwocomplementarydiscreteARarchitectures(Delay-Patternvs.Global-Latent

+ Local Transformer) that expose a clear tradeoff between structural simplicity/scalability and modeling efficiency/quality.

- • We demonstrate broad controllability features (voice cloning, duration control, pronunciation control) and strong empirical performance on speaker similarity and quality metrics.

Organization. The remainder of the report is organized as follows. We begin with an overview of related work. We then describe the audio tokenizer and the overall modeling architectures, followed by the pretraining data pipeline and training recipe. We next present the evaluation results before concluding.

### 2 Related Work

MOSS-TTS sits at the intersection of discrete audio tokenization, large-scale autoregressive sequence modeling, and speech generation foundation models. We review the most relevant directions below.

Neural audio codecs and discrete audio tokenization. Discrete representations have become a standard foundation for scalable audio generation, following the broader success of vector quantization in representation learning [15]. Neural codecs such as SoundStream [16] and subsequent high-fidelity compression models [17, 18] demonstrate that a learned encoder–quantizer–decoder stack can support low-bitrate reconstruction while remaining compatible with downstream sequence modeling. Recent toolkits and open implementations further accelerate codec research and adoption [19, 20]. For speech generation in particular, an effective tokenizer must not only reconstruct waveforms, but also expose tokens that are semantically aligned with text and robust under long-horizon generation. Several recent lines explore semantic shortcomings and the semantic–acoustic tradeoff in codec tokens [8, 14, 21], motivating tokenizers that better balance compression, perceptual quality, and text-aligned semantics.

Audio language modeling with discrete tokens. With discrete tokens, audio generation can be cast as token sequence modeling, enabling language-model-like scaling and training recipes [22–24]. Codec language models have been shown to produce intelligible speech and even zero-shot TTS behavior when trained autoregressively over discrete units [25]. Concurrently, a growing body of work studies how token choices and modeling decisions affect controllability, semantic fidelity, and efficiency [26, 27]. MOSS-TTS follows this trend but emphasizes a tokenizer and modeling stack designed to scale end-to-end without external pretrained audio teachers, aligning the discrete token representation with the requirements of AR speech generation.

TTS architectures: AR, NAR, diffusion/flow, and foundation-model scaling. Classical neural TTS systems progressed from AR acoustic modeling and neural vocoders [28–30] to faster and more controllable NAR frameworks [31, 32], flow-based and diffusion-based synthesis [33, 34]. End-to-end approaches such as VITS [35] further unified acoustic modeling and waveform generation, improving simplicity and sample quality. More recently, scaling-driven and token-centric systems increasingly combine discrete representations with AR backbones for robustness and controllability at scale [36], as reflected in recent open technical reports and large-scale systems such as Qwen3-TTS [1], CosyVoice [9], CosyVoice 3 [2], Seed-TTS [3], Fish-Speech [37], and FireRedTTS-2 [4]. Across these efforts, a recurring theme is that scaling data and model capacity alone is insufficient without a well-chosen discrete tokenizer and a model design that remains compatible with streaming, controllability, and long-context stability. MOSS-TTS complements this line of work by focusing on a fully discrete tokenization pipeline and token modeling strategies that remain efficient for long-form synthesis, while explicitly comparing two autoregressive architectures under the same tokenizer and largescale pretraining recipe.

Voice cloning and controllability. Practical TTS systems increasingly demand controllability beyond text content, including speaker identity (voice cloning), speaking rate/duration control, and fine-grained pronunciation control. Zero-shot voice cloning and multilingual universal generation have been explored via largescale generative models and conditioning mechanisms [38–40]. Token-centric systems also enable control signals to be expressed directly in the discrete domain, which can simplify modeling and improve stability compared to waveform-level control. MOSS-TTS emphasizes token-level duration control and phoneme/pinyin-level pronunciation interfaces, aiming to make control explicit and composable.

### 3 Audio Tokenizer

#### 3.1 Motivation and Design Principles

Audio tokenizers serve as the foundational bridge for native Audio Large Language Models (Audio LLMs), transforming continuous raw audio signals into discrete tokens that can be seamlessly processed within a unified generative framework. A unified audio tokenizer for speech LLMs must satisfy two primary requirements: enabling high-fidelity reconstruction of diverse audio signals and maintaining compatibility with the sequential nature of autoregressive modeling [8, 41, 42].

Existing approaches typically address these requirements through pretrained audio encoders (e.g., HuBERT, Whisper) [12–14, 43], multi-stage training pipelines [19, 44], or architecture-specific inductive biases such as specialized CNN structures [16–18]. These designs often introduce external dependencies and architectural constraints that hinder the seamless scaling of model capacity, data volume, and quantization levels. Drawing inspiration from the success of LLMs, where simple, scalable architectures trained on massive datasets have proven superior [5, 6], we posit that the performance ceiling of audio tokenizers can be raised by adopting a similar philosophy. We advocate for a simple, end-to-end scalable architecture that minimizes reliance on external priors or complex heuristics, emphasizing joint optimization and large-scale data exposure.

To address these limitations and support high-quality speech synthesis in MOSS-TTS, we use MOSS-AudioTokenizer, ahigh-performanceaudiotokenizerbasedontheCAT(CausalAudioTokenizerwithTransformer) architecture [11]. MOSS-Audio-Tokenizer is characterized by the following core strengths:

- • High Compression and Variable Bitrate: The model achieves a significant compression ratio, converting 24kHz audio into a discrete representation at only 12.5 frames per second (fps). Utilizing a 32-layer Residual Vector Quantization (RVQ) mechanism, it supports flexible bitrate adjustment from 0.125 to 4 kbps, catering to various high-fidelity reconstruction requirements.
- • Pure Transformer Architecture: Unlike traditional codecs that rely on complex, hand-crafted CNN or hybrid CNN-Transformer blocks, MOSS-Audio-Tokenizer adopts a minimalist causal Transformerbased design. This architecture is intentionally unencumbered by specialized inductive biases, making it remarkably simple to implement and highly efficient to scale up. With a substantial 1.6-billionparameter capacity, the model demonstrates superior representation power, while its inherently causal nature ensures seamless, frame-level streaming inference.
- • Universal Audio Representation: The model is pretrained on millions of hours of diverse audio data, including speech, music, and environmental sound effects, ensuring robust generalization across all audio domains.
- • Unified Semantic-Acoustic Modeling: The discrete tokens produced by MOSS-Audio-Tokenizer preserve strong reconstruction quality while inherently capturing rich semantic information, making them ideally suited for autoregressive LLM modeling.
- • End-to-End Joint Optimization: All components, including the encoder, quantizer, decoder, discriminators, and the LLM used for semantic alignment, are optimized jointly to maximize the model’s performance ceiling.

#### 3.2 Architecture

As illustrated in Figure 1, MOSS-Audio-Tokenizer adopts an RVQ-GAN framework for training. The model consists of five components: a causal encoder, a residual vector quantizer (RVQ), a causal decoder, a decoderonly LLM for semantic modeling, and adversarial discriminators.

Fully Transformer-based Encoder and Decoder. The encoder and decoder of MOSS-Audio-Tokenizer each consist of 68 causal Transformer blocks. To facilitate efficient streaming inference, both components use a 10second sliding-window attention mechanism. To progressively reduce the sequence length, the encoder

Discriminator

Real/Fake

[Figure 8]

Reconstruction Loss

[Figure 9]

[Figure 10]

VQ Loss

Audio

[Figure 11]

Speech

CausalTrm

CausalTrm

CausalTrm

CausalTrm

CausalTrm

CausalTrm

CausalTrm

CausalTrm

RVQ 32

[Figure 12]

Music

[Figure 13]

[Figure 14]

12.5 Hz

[Figure 15]

[Figure 16]

Sound

[Figure 17]

[Figure 18]

[Task_type] [Audio Hidden State]

Text

[Figure 19]

Decoder-only LLM Transcription / Caption

- Figure 1 Architecture of MOSS-Audio-Tokenizer. Both the encoder and decoder are built upon causal Transformers. All components, including the encoder, quantizer, decoder, decoder-only LLM, and discriminator, are optimized jointly in an end-to-end manner.

incorporates patchify operations [45] at the input stage and following layers 12, 24, and 36, with respective patch sizes of 240, 2, 2, and 2. Since these patchify operations alter the feature dimensionality, a linear projection is applied after each stage to map the hidden states to the corresponding Transformer block dimension. This configuration effectively downsamples raw 24kHz waveforms to a low frame rate of 12.5fps. The encoder is structured into four stages with hidden dimensions of 768, 768, 768, and 1280, containing 12, 12, 12, and 32 Transformer blocks, respectively. For each stage, the feed-forward network (FFN) dimension is set to four times the hidden dimension. The multi-head self-attention mechanism uses 12, 12, 12, and 20 attention heads across the four stages. All Transformer blocks employ rotary positional embeddings (RoPE) [46]. The decoder mirrors the encoder architecture in a fully causal manner. Both the encoder and decoder contain approximately 0.8B parameters and are trained from scratch.

Residual Vector Quantization. Discrete tokenization is performed using a 32-layer residual vector quantizer (RVQ). Each layer employs a codebook of size 1024 with factorized vector quantization (latent dimension 8) [18] and L2-normalized codes. To enable variable-bitrate tokenization, random quantizer dropout [16] with a probability of 1.0 is applied during training.

Semantic Supervision. To encourage the learning of semantically structured discrete representations, we attach a 0.5B decoder-only causal language model [47] as a semantic head. This head provides audio-to-text supervision by autoregressively predicting text conditioned on the quantizer outputs. The supervision tasks include Automatic Speech Recognition (ASR), multi-speaker ASR, and audio captioning.

Perceptual Modeling. To enhance the perceptual quality of the reconstructed audio, we employ a multiperiod discriminator [17] and a complex STFT discriminator [18] for adversarial training with the audio tokenizer.

#### 3.3 Training

MOSS-Audio-Tokenizer is trained on a massive dataset comprising millions of hours of both public and inthe-wild audio data. During training, we employ a multi-task learning framework to enable MOSS-AudioTokenizer to achieve both robust semantic alignment with text and high-fidelity audio reconstruction. The modeling approach for each component is detailed as follows.

Semantic Modeling via Audio-to-Text Tasks. To encourage the token representation to be semantically rich and aligned with text-based language modeling, we incorporate an auxiliary audio-to-text objective. Specifically, we employ a 0.5B-parameter decoder-only LLM [47] and condition it on the representations produced

by MOSS-Audio-Tokenizer. Concretely, we feed the hidden states from the quantizer output into the LLM, which then autoregressively predicts textual tokens. We consider a diverse set of audio-to-text tasks, including automatic speech recognition (ASR), multi-speaker ASR, and audio captioning. For audio samples that are paired with textual annotations, we apply the corresponding semantic modeling objective. Each task is specified by a fixed task tag 𝒯 , which is prepended to the LLM input. The semantic objective is optimized using a standard cross-entropy loss:

ℒsem = −

|s|

log 𝑝𝜃LLM (s𝑡 |𝒯 , q, s<𝑡) , (1)

𝑡=1

where s = (s1, . . . , s|s|) denotes the target text token sequence, q denotes the sequence of quantized audio representations produced by MOSS-Audio-Tokenizer, 𝒯 is a task-specific prompt token, and 𝜃LLM are the parameters of the causal language model.

QuantizerOptimization. Fortrainingsimplicityandstability, eachquantizationlayerinMOSS-Audio-Tokenizer

adopts factorized vector quantization [18], where codebooks are directly optimized via gradient descent, without relying on additional codebook update mechanisms [17]. We incorporate a commitment loss and a codebook loss to jointly optimize the encoder and the codebook entries:

𝑁𝑞

ℒcmt =

𝑐=1

z𝑐 − sg(𝑞𝑐(z𝑐)) 22 , (2)

𝑁𝑞

sg(z𝑐) − 𝑞𝑐(z𝑐) 22 , (3)

ℒcode =

𝑐=1

where z𝑐 denotes the input to the 𝑐-th quantization layer, 𝑞𝑐(z𝑐) is the corresponding quantized output, 𝑁𝑞 is the number of quantizers, and sg(·) denotes the stop-gradient operator [15].

Acoustic Modeling via Reconstruction Tasks. To ensure high-fidelity and domain-robust audio reconstruction, we adopt a multi-scale mel-spectrogram loss:

11

∥𝑆2𝑖(x) − 𝑆2𝑖(ˆx)∥1 , (4)

ℒrec =

𝑖=5

where 𝑆2𝑖(·) denotes the mel-spectrogram computed using a normalized short-time Fourier transform (STFT) with window size 2𝑖 and hop size 2𝑖−2. Here, x is the ground-truth waveform and xˆ is the reconstructed waveform generated by the decoder.

Adversarial Training. To further improve reconstruction fidelity and perceptual quality, we employ adversarial training with multiple discriminators. The discriminator loss follows the least squares GAN (LSGAN) formulation [48], given by:

𝐾

1 𝐾

(1 − 𝐷𝑘(x))2 + 𝐷𝑘2(ˆx), (5)

ℒD(x, xˆ) =

𝑘=1

where 𝐷𝑘 represents the 𝑘-th discriminator, 𝐾 is the total number of discriminators, x is the ground-truth audio, and xˆ is the predicted audio.

For the generator, we include an adversarial loss and a feature matching loss. The adversarial loss encourages

the generator to produce high-fidelity audio that is indistinguishable from real samples:

1 𝐾

ℒadv(ˆx) =

𝐾

(1 − 𝐷𝑘(ˆx))2. (6)

𝑘=1

Additionally, we incorporate a feature matching loss ℒfeat [49] to ensure structural similarity across multiple scales. It penalizes the ℓ1 distance between the intermediate feature maps of the discriminators for real and synthetic audio:

𝐷𝑘𝑙 (x) − 𝐷𝑘𝑙 (ˆx) 1 mean( 𝐷𝑘𝑙 (x) 1)

𝐿𝑘

𝐾

1 𝐾

1 𝐿𝑘

ℒfeat(x, xˆ) =

(7)

𝑘=1

𝑙=1

where 𝐷𝑘𝑙 denotes the feature representation from the 𝑙-th layer of the 𝑘-th discriminator, and 𝐿𝑘 is the number of layers in that discriminator.

Overall Training Objective. The overall generator objective is a weighted combination of all loss terms: ℒG = 𝜆semℒsem + 𝜆recℒrec + 𝜆cmtℒcmt + 𝜆codeℒcode + 𝜆advℒadv + 𝜆featℒfeat, (8)

where 𝜆sem, 𝜆rec, 𝜆cmt, 𝜆code, 𝜆adv, 𝜆feat are scalar hyperparameters controlling the relative contribution of each loss term.

During training, we set the hyperparameters to 𝜆sem=20, 𝜆rec=15, 𝜆cmt=0.25, 𝜆code=1.0, 𝜆adv=1.0, 𝜆feat=2.0. Due to computational constraints, we adopt a two-stage training schedule to improve training efficiency: nonadversarial pretraining without discriminator-related losses for 520k steps (batch size 1536, approximately 5 hours of audio per batch), followed by adversarial fine-tuning for 500k steps (batch size 768). All modules are optimized end-to-end without pretrained encoders or semantic teachers [7, 8, 12–14].

### 4 Architecture

MOSS-TTS is a speech generation foundation model built upon discrete audio tokens. To facilitate effective scaling and capitalize on the success of large language models (LLMs), we adopt a straightforward end-toend, purely autoregressive (AR) architecture. As illustrated in Figure 2, given a text sequence and an optional speech prompt, MOSS-TTS generates the target token sequence through next-token prediction. The central architectural question is not whether to use AR modeling, but how to handle the multi-stream discrete token block produced by the tokenizer. For a 32-layer RVQ tokenizer, the chosen token modeling pattern directly determines engineering complexity, scaling behavior, decoding latency, and final synthesis quality.

Rather than committing to a single token modeling pattern a priori, we train two architectures under the same tokenizer and large-scale pretraining recipe. This serves a concrete research purpose: to isolate the effect of the token modeling pattern itself in large-scale discrete speech modeling. In practice, the two designs expose a clear tradeoff. Delay Pattern uses a structurally simple single-backbone, multi-head parameterization, making it easier to scale to large model sizes, long contexts, and optimized inference backends. Local Transformer introduces an additional frame-local autoregressive module, which increases architectural complexity but improves modeling efficiency; during internal development, it exhibited consistently lower perlayer token losses, and the later voice-cloning evaluations show stronger speaker similarity at much smaller model scale. Correspondingly, the current report uses MOSS-TTS-Local-Transformer to highlight the quality advantage of the local pattern on standard cloning benchmarks (Table 3), while MOSS-TTS serves as the main architecture for duration control, pronunciation control, and ultra-long generation (Tables 5, 7, and 6).

The tokenizer emits 𝑁𝑞 = 32 RVQ layers. In our implementation, both architectures predict 𝑁ℎ = 𝑁𝑞 +1 = 33 channels at each aligned step: one text-or-pad channel 𝑦0,𝑡 and 32 audio channels 𝑦1,𝑡, . . . , 𝑦𝑁𝑞,𝑡, where 𝑦𝑗,𝑡 = a𝑗,𝑡 for 𝑗 ≥ 1. When step 𝑡 corresponds to an audio frame, 𝑦0,𝑡 is trained to emit a dedicated pad symbol; on

[Figure 20]

- Figure 2 Architecture of MOSS-TTS. The left panel illustrates the delay pattern as described in Section 4.1, while the right panel depicts the local transformer pattern as detailed in Section 4.2.

text-only steps, it emits the normal text token. We use the same head-wise weighted cross-entropy in both architectures, with

𝝀 = (1, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,

(9)

1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),

which up-weights the earliest coarse RVQ layers while keeping unit weight on the text-or-pad channel and the remaining finer layers.

#### 4.1 Delay Pattern

To model the RVQ hierarchy efficiently without increasing the sequence length to 𝑇 × 𝑁𝑞, we adopt a delay pattern [50]. Among the two architectures we study, this is the simpler and more scalable design: a single Transformer backbone carries the full sequence model, and each prediction channel is obtained by a lightweight head projection from the backbone hidden state.

Let s denote the input text sequence and A ∈ {1, . . . ,𝑉}𝑁𝑞×𝑇 be the audio token matrix, where 𝑉 denotes the codebook size of each RVQ layer, 𝑁𝑞 is the number of quantizers, and 𝑇 is the number of audio frames. Each element a𝑗,𝑡 ∈ {1, . . . ,𝑉} represents the token index at the 𝑗-th RVQ layer and 𝑡-th time frame. We apply a time-delay shift such that the 𝑗-th layer is shifted forward by 𝑗 − 1 frames. The delayed token matrix A˜ is defined as:

a˜𝑗,𝑡 = a𝑗,𝑡−(𝑗−1), 𝑡 ∈ {𝑗, . . . ,𝑇 + 𝑗 − 1}. (10)

Input Embedding. On the input side of the backbone LLM, we use 𝑁𝑞 distinct speech embedding tables. For each time step 𝑡 in the delayed sequence, the input audio representation vector h𝑡 ∈ ℝ𝐷 is the sum of embeddings across all layers:

𝑁𝑞

h𝑡 =

Emb𝑗(˜a𝑗,𝑡), (11)

𝑗=1

where Emb𝑗(·) denotes the embedding lookup for the 𝑗-th codebook and 𝐷 is the model hidden dimension. Text tokens are embedded with the standard text embedding table; the delay mechanism applies only to the RVQ audio streams. The resulting vector sequence of length 𝑇+𝑁𝑞 −1 is concatenated with text embeddings

- as the backbone input.

Modeling Objective. On the output side, the hidden state x𝑡 is passed through 𝑁ℎ = 33 heads: one text-orpad head and 32 audio heads. Let 𝑦˜0,𝑡 = 𝑦0,𝑡 and 𝑦˜𝑗,𝑡 = a𝑗,𝑡 for 𝑗 ≥ 1. The weighted training objective is

ℒdelay = −

𝑇+𝑁𝑞−1

𝑡=1

𝑁𝑞

𝑗=0

𝜆𝑗𝑚𝑗,𝑡 log 𝑝𝜃delay(˜𝑦𝑗,𝑡 | E, {˜𝑦𝑥,𝑦 : 𝑥 + 𝑦 < 𝑗 + 𝑡 + max(0, 1 − 𝑗)}), (12)

where 𝜃delay encompasses the parameters of the backbone, embeddings, and prediction heads; E represents the text token sequence; and 𝑚𝑗,𝑡 masks the invalid positions introduced by delay shifting and padding. Because every channel is predicted directly from the backbone state, the delay pattern keeps the decoding path simple: once x𝑡 is available, token generation only requires head projections. This simplicity is one of the main reasons it is easier to implement, scale, and deploy.

4.2 Local Transformer

We further explore a hierarchical token modeling design using a Local Transformer, inspired by the RQTransformer in Moshi [8]. Unlike the delay pattern, this approach models the token block without introducing temporal shifts: the backbone produces one global latent per aligned step, and a lightweight autoregressive module expands that latent into the within-step token block. This design is architecturally more complex, but it offers a stronger inductive bias for frame-level token modeling.

Input Embedding. On the input side, we directly sum the embeddings of all RVQ layers at each time step 𝑡 without any delay. The input hidden state h𝑡 to the backbone LLM is given by:

h𝑡 =

𝑁𝑞

𝑗=1

Emb𝑗(a𝑗,𝑡), (13)

where a𝑗,𝑡 is the token at the 𝑗-th RVQ layer and 𝑡-th time frame.

Hierarchical Decoding. On the output side, we employ a lightweight Local Transformer to autoregressively decode the full per-step token block. Specifically, let x𝑡 be the output hidden state from the backbone LLM

- at time 𝑡. The Local Transformer predicts the sequence (𝑦0,𝑡+1, 𝑦1,𝑡+1, . . . , 𝑦𝑁𝑞,𝑡+1) sequentially. The input to the Local Transformer when predicting channel 𝑗, denoted as z𝑗,𝑡, is defined as:

x𝑡, if 𝑗 = 0, Emb𝑗−1(𝑦𝑗−1,𝑡+1), if 1 ≤ 𝑗 ≤ 𝑁𝑞.

z𝑗,𝑡 =

(14)

The Local Transformer processes z𝑗,𝑡 and passes the resulting hidden states through the corresponding prediction head to emit 𝑦𝑗,𝑡+1.

Modeling Objective. The entire architecture, including the backbone and the local transformer, is trained end-to-end and optimized via

𝑇

ℒlocal = −

𝑡=1

𝑁𝑞

𝜆𝑗 log 𝑝𝜃local(𝑦𝑗,𝑡 | E, 𝑦<𝑗,𝑡, 𝑦:,<𝑡), (15)

𝑗=0

###### Raw Audio

###### Phase1: Preprocessing

###### Phase2: Filtering Phase3: Data Synthesis

[Figure 21]

- Stage ① Raw Audio Preprocessing

- Stage ② Diarization & Consolidation

- Stage ③ ASR & Transcript QC

- Stage ④ Joint Audio–Text Filtering

Timbre-Cloning Data

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Audiobook

Noise Reduction

[Figure 30]

###### [S1] [S2] [S3] [S2] [S3] [S2] [S1]

[Figure 31]

Format Alignment

Film & Drama

Pass? Rule-based Pre-ﬁltering

[Figure 32]

[Figure 33]

[Figure 34]

Most Similar!

[Figure 35]

[Figure 36]

[Figure 37]

Pass? LLM Reﬁnement

Volume Normalization

Target

Segments of Speakeri

Segment Pairs of Speakeri

Candidates

Pass? Single-speaker validation

Podcast

[Figure 38]

Supplementary Data

[Figure 39]

Broadcast & News

今天吃啥呀 Hello, how are you?

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Quality Scoring

LangID

[Figure 45]

Input Robustness Phonetic Input Short-Form

[S1] [S2] [S3] [S2]

Pass?

Acoustic Quality Filtering

Commentary

吃chi2啥sha2呀

今天，，吃啥呀

今天 吃

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Audio–text Language Pass? Consistency Filtering

Hello?//// how are you?

hɛloʊ, haʊ ɑr juː?

[Figure 50]

are zigzag

[S1] [S2] [S3] [S2]

Pass?

Length Consistency Filtering

Online Content

- Figure 3 Overview of the MOSS-TTS pretraining data pipeline, including preprocessing, filtering, and targeted data synthesis.

where 𝑦<𝑗,𝑡 denotes the preceding channels at the current aligned step, and 𝑦:,<𝑡 denotes all channels from previous steps. 𝜃local encompasses the parameters of the backbone LLM, the local transformer, embeddings, and prediction heads. Compared with the delay pattern, this design inserts an additional autoregressive loop of length 𝑁𝑞 + 1 inside each frame. As a result, it is computationally heavier in steady-state decoding, but it can start emitting audio earlier because it does not need to wait for delayed offsets to materialize the first frame. Empirically, its main advantage in this report is not architectural simplicity, but higher modeling efficiency and stronger speaker preservation.

Moreover, we incorporate Progressive Sequence Dropout as proposed in MOSS-Audio-Tokenizer [11] to support bitrate-controllable audio generation.

### 5 Pretraining

#### 5.1 Pretraining Data

Scaling TTS pretraining to millions of hours of speech data necessitates sourcing audio from naturally occurring, open-domain recordings—podcasts, audiobooks, broadcast & news, film & drama, commentary, and online content. Such recordings, however, rarely satisfy the conditions required for direct TTS supervision: they routinely contain multiple concurrent speakers, background music, ambient noise, and unreliable or missing transcription metadata. High-quality pretraining therefore demands that two fundamental properties hold for every training unit: (i) the audio is acoustically clean and contains the speech of a single speaker, free from overlapping voices, music, and significant background noise, and (ii) the paired transcript is linguistically well-formed and faithfully aligned to the spoken content. To enforce these properties at scale, we design a multi-stage data pipeline that progressively transforms raw web audio into curated, trainable speech–transcript pairs. As summarized in Figure 3, the pipeline is organized into three phases. The preprocessing phase (Stages 1–2) establishes a standardized acoustic foundation and extracts speaker-consistent segments via diarization. The filtering phase (Stages 3–4) first produces and refines transcripts with multilingual ASR, rule-based checks, and LLM-based quality control, and then retains only pairs that pass joint audio–transcript filtering, including acoustic quality checks, audio/text language-consistency checks, and duration-text consistency checks. The data synthesis phase supplements the corpus with targeted examples that introduce explicit speaker-conditioning structure for timbre transfer, broaden coverage of underrepresented input types, and strengthen robustness to diverse real-world input formats.

##### 5.1.1 Data Preprocessing

As shown in Figure 3, the preprocessing phase contains Stages 1–2 and prepares acoustically standardized, speaker-consistent segments before any transcript is produced.

- Stage 1: Raw audio preprocessing. Raw web-sourced recordings exhibit substantial heterogeneity in acoustic format: sampling rates vary widely across sources, loudness levels differ by tens of decibels, and many files carry environmental noise, music accompaniment, or reverberation. Left uncorrected, this variability degrades the reliability of both speaker diarization (Stage 2) and ASR (Stage 3), and introduces inconsistency into the acoustic features used during model training. We therefore apply the following preprocessing steps to each recording as the first pipeline stage:

- • Noise suppression. We apply MossFormer2-SE-48K [51], a neural speech enhancement model, to suppress stationary and non-stationary background noise. The purpose of denoising at this stage is not to produce the final training signal but to improve the reliability of downstream speaker diarization: a cleaner input yields more accurate voice activity detection and sharper speaker boundary estimates. Audio is resampled to 48kHz before enhancement, which is the native operating rate of the model.
- • Format standardization. Parameter alignment such as sample type, channel layout, and header metadata is enforced after enhancement, and the processed output is written to FLAC format, establishing a uniform format contract for all subsequent stages.
- • Volume normalization. To reduce inter-source level variation, we apply a two-stage gain procedure. First, we compute the RMS-based signal level in dBFS,

𝐿dBFS(x) = 20log10 𝑇1 𝑡 𝑥𝑡2 + 𝜖 ,

and apply a clamped gain 𝑔 = clip(−20 − 𝐿dBFS(x), −3, 3)dB, rescaling the waveform by 10𝑔/20. The target of −20dBFS and the ±3dB clipping range together prevent both over-compression of alreadyquiet recordings and excessive amplification of outliers. Second, peak normalization divides by the maximum absolute sample value to map the result to the [−1, 1] amplitude range, ensuring numerical consistency across the pipeline.

- Stage 2: Speaker diarization and segment consolidation. We run speaker diarization on the denoised audio to obtain a time-ordered sequence of speaker-labeled intervals,

𝒟 = {(𝑘𝑖, 𝑡𝑖st, 𝑡𝑖ed)}𝑁𝑖=1,

where 𝑘𝑖 is a recording-local speaker label (e.g., SPEAKER-00, SPEAKER-01, …) and 𝑡𝑖st < 𝑡𝑖ed are the start and end timestamps. Speaker labels are meaningful only within a single recording; we do not perform cross-

recording identity linking. We use DiariZen [52–54], an end-to-end neural diarization system, for this step.

Raw diarization output is typically fragmented: a single continuous speaking turn may be split into multiple short intervals separated by brief pauses or breath sounds. Training directly on such fragments would overrepresent short, sub-sentence units at the expense of longer, paragraph-level continuity. We therefore apply a two-step consolidation procedure to maximize contiguous single-speaker coverage:

- • Filtering and consecutive-speaker merging. Segments shorter than 𝜏min = 0.1s are discarded as unreliable diarization artifacts. The remaining segments are then scanned in chronological order: whenever two adjacent segments carry the same speaker label, they are merged into a single interval spanning both. This produces a consolidated sequence

𝒜 = {(𝑘𝑗, 𝑠𝑗, 𝑒𝑗)}𝑀𝑗=1, 𝑀 ≤ 𝑁, where each run of consecutive same-speaker segments in 𝒟 has been collapsed into one entry. There

is no gap threshold on merging: any two adjacent same-speaker segments are unified regardless of the intervening silence duration, because such gaps reflect natural pauses within a speaking turn rather than speaker changes.

- • Single-speaker truncation. We apply a hard one-hour limit to avoid unbounded unit lengths. Starting from 𝑠1 (the onset of 𝒜), we define a cutoff 𝑡lim = 𝑠1+3600s and emit segments from 𝒜 in order, clamping the endpoint of the last included segment to 𝑡lim and discarding any resulting segment shorter than 𝜏min. The output is a list of consolidated, speaker-labeled intervals drawn from at most one hour of the recording, starting from the onset of the first diarized segment.

##### 5.1.2 Data Filtering

As shown in Figure 3, the filtering phase corresponds to Stages 3–4: Stage 3 constructs and cleans transcripts, and Stage 4 keeps only pairs that are jointly consistent on the audio and transcript sides.

- Stage 3: ASR and transcript quality control. Each consolidated segment from Stage 2 passes through a sequential pipeline that transcribes the audio and then applies a series of quality-control steps to produce a clean, speaker-tagged transcript suitable for TTS training.

- • ASR transcription. We transcribe each segment using MOSS-Transcribe-Diarize [55], our proprietary multilingual ASR model. The model does not require an externally provided language label; it directly produces a multilingual diarization-aware transcript. The raw output follows a structured format in which every utterance span is prefixed by a recording-local speaker tag (e.g., [S1], [S2]) and may contain inline sound-event markers (e.g., [music], [laugh]). This structured output reflects the full diarizationaware recognition result and requires subsequent cleaning before use as a training transcript.
- • Rule-based pre-filtering. Before invoking the LLM, three lightweight rules discard clearly unusable transcripts to avoid wasting inference budget:

- – Empty content: the transcript is blank or consists only of whitespace after stripping.
- – Severe repetition loop: any phrase is repeated consecutively more than six times, which is a reliable signal of ASR model collapse.
- – Non-speech dominance: after removing all bracketed tags ([...]), the remaining linguistic content accounts for less than 20% of the total text length, indicating that the segment is predominantly noise, music, or other non-speech events.

Segments failing any rule are discarded without further processing.

- • LLM-based transcript refinement. Segments that pass pre-filtering are processed by a large language model using a structured two-stage prompt.

- – Diagnosis (filtering): the LLM first checks for two fatal defects. filter-1 targets identical spoken contentrepeatedtwoormoretimes(distinctfromthetriviallyrepeatingspeakertagsthatareanormal output of MOSS-Transcribe-Diarize). filter-2 targets sentence truncation, identified by a trailing hyphen or an abrupt mid-word termination. Segments receiving either code are discarded.
- – Correction (refinement): segments that pass diagnosis undergo sequential cleaning. refine-1 removes all non-speech event tags while preserving speaker tags and linguistic content. refine-2 deletes any speaker tag that is left with no following speech content. refine-3 applies minimal structural repair torestorethe standard [speaker]content format without modifying the recognized words. Not all steps are applied to every segment; segments already in correct form receive a nochange code and are passed through unmodified.

If the LLM call fails (e.g., due to a malformed response), the segment is discarded rather than falling back to the uncleaned transcript.

- • Single-speaker transcript validation. As a final check, we verify that the refined transcript contains only [S1] speaker tags. The presence of any [S2], [S3], or higher-indexed tag indicates that multiple speakers were detected within the segment at the transcription level, which is inconsistent with the single-speaker constraint enforced in Stage 2. Such segments are discarded.

- Stage 4: Joint audio–transcript filtering. Segments that survive Stage 3 undergo a second round of filtering that combines acoustic quality signals with audio–transcript consistency checks. Unlike Stage 3, which focuses on transcript quality, this stage treats the audio and transcript jointly; a segment is retained only if both its acoustic quality and its audio–transcript consistency fall within acceptable bounds.

- • Acoustic quality filtering. We compute DNSMOS [56] and Meta AudioBox Production Quality (PQ) [57] on the pre-denoising audio rather than on the enhanced signal, because speech enhancement can distort quality estimates and make the scores reflect the enhancer rather than the source recording. A segment is accepted only if its DNSMOS score exceeds 2.8 and its Meta AudioBox PQ score exceeds 6.5, retaining only segments with sufficiently clean and natural-sounding speech.
- • Audio–text language consistency filtering. We derive two language labels from different modalities and require them to agree. First, Whisper large-v3 [13] is applied to the audio to obtain an audio-side language label ℓˆaud. Second, a large language model reads the refined transcript and predicts a textside language label ℓˆtext. We retain a segment only if ℓˆaud = ℓˆtext. This removes pairs whose transcript language is inconsistent with the spoken content or whose ASR output is unreliable enough to confuse transcript-side language identification. For the remaining segments, we denote the agreed label by ℓˆ.
- • Audio–transcript length consistency filtering. A systematic mismatch between audio duration and transcript length indicates one of two failure modes: (i) the audio is far longer than the transcript, suggesting that large portions of the segment are silence or non-speech background; or (ii) the transcript is far longer than the audio, a reliable indicator of ASR hallucination. To detect both cases, we compute a language-specific character rate,

𝑟 = |𝑥′| 𝑑

,

where |𝑥′| is the character count of the refined transcript and 𝑑 is the segment duration in seconds. For each supported language ℓ, we define a valid rate interval [𝑟ℓmin, 𝑟ℓmax] derived from empirical statistics over a reference corpus; the agreed language label ℓˆ from the previous step is used to select the appropriate bounds. Segments whose rate 𝑟 ∉ [𝑟ℓmin, 𝑟ℓmax] are discarded.

##### 5.1.3 Data Synthesis

As shown in Figure 3, the final phase augments the naturally filtered corpus with targeted synthetic or transformed examples that cover capabilities not directly available from organic web audio.

Evenafterthepipelinedescribed above, thefilteredweb-sourcedcorpusstillleavesthreesystematicgapsthat cannot be closed by filtering alone. The most consequential is the absence of explicit speaker-conditioning structure: the filtered corpus pairs text with speech but provides no prompt audio, and a model trained on it alone has no mechanism to transfer timbre from a reference speaker. The remaining two gaps are on the text side: real user inputs often contain formatting noise that the model must handle gracefully , for example, inputs such as “Hello??!! are you there” or “Iwant to knowwhere it is”, and phonetic script input is absent from organic speech data yet required for fine-grained pronunciation control. We address all three through targeted data synthesis.

Timbre-cloning data construction. The goal of this construction is to provide (prompt audio, target audio) pairs in which both sides originate from the same speaker, enabling the model to learn prompt-conditioned timbre transfer from real recorded speech rather than from any generative process. Construction proceeds entirely from the filtered corpus produced by Stages 1–4.

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- Figure 4 Statistics of the MOSS-TTS pretraining corpus. Panel (a) shows the share of training hours by domain; panel (b) shows the language distribution as a donut chart (English/Chinese/Other) alongside a breakdown of the top minor languages; panel (c) shows the distribution of utterance duration by both hours (bars) and utterance count (line).

For each recording, we group the surviving segments by their diarization-assigned speaker identity. Let the segments attributed to a given speaker be {𝑠1, 𝑠2, . . . , 𝑠𝑛}. For each target segment 𝑠𝑖, we construct a prompt candidate pool as follows. For every other segment 𝑠𝑗 (𝑗 ≠ 𝑖), we draw five random temporal crops of 𝑠𝑗, each with independently sampled start and end timestamps subject to a maximum duration of 30s, yielding 5(𝑛 − 1) prompt candidates in total. We then score every candidate against 𝑠𝑖 by computing the cosine similarity between their speaker embeddings, extracted using the fine-tuned WavLM-Large model employed in Seed-TTS-eval [3]. The candidate attaining the highest similarity is selected as the prompt for 𝑠𝑖, producing the final training pair (prompt = 𝑠∗

𝑗,partial, target = 𝑠𝑖).

This construction strategy has two important properties. First, by evaluating similarity on the cropped partials rather than on full segments, the selection directly optimizes for how well the prompt conveys the speaker’s identity at inference-time durations, rather than for full-segment representativeness. Second, restricting the prompt to at most 30s and randomizing its boundaries encourages the model to extract stable timbre representations from varied temporal windows, improving robustness to prompt length and position at inference time.

Supplementary data. Three smaller supplements address remaining distributional gaps. For input robustness, we apply four noise transformations to the text of existing validated pairs—punctuation noise (consecutive marks, mixed full/half-width forms, malformed combinations), whitespace artifacts (extra spaces, misplaced line breaks), punctuation dropout, and sparse dirty-character injection—without modifying the audio. For phonetic script input, we support both full-sentence and partial (word- or phrase-level) replacement of orthographic text with phonetic notation: tone-marked pinyin for Chinese (e.g., nin2 hao3) and IPA enclosed in slashes for English (e.g., /hæŋ bæk/); training pairs are derived from filtered corpus segments by rule-based transcript conversion, with audio left unchanged. For dictionary-style short-form data, we supplement the corpus with single-character and single-word utterances, which are severely underrepresented in web-sourced recordings yet constitute a common real-world input pattern; without targeted coverage, a model trained on organic data alone tends to be unreliable on such ultra-short inputs.

- Figure 4 summarizes the composition of the resulting corpus across all data subsets, showing the distribution of training hours by domain, language, and utterance duration.

For duration control, every training asset is serialized into two parallel variants. In the duration-conditioned variant, the prompt field tokens stores an explicit integer equal to the target audio-token count; in the freeduration variant, the same field is set to None. This paired formatting is applied uniformly across the corpus, so explicit and implicit duration supervision are both present throughout pretraining rather than being introduced only in a specific phase.

Table 1 Four-phase pretraining schedule of MOSS-TTS.

Phase Max seq. LR Schedule and Data Mixture Updates

- P1 32k LR warmup to 2 × 10−4, then hold; use 𝒟basic only.
- P2 32k Hold LR at 2 × 10−4; enable all data subsets and strongly upsample 𝒟clone.
- P3 32k

Linearly decay LR from 2 × 10−4 to 2 × 10−6; keep all subsets active and restore the data mixture to normal proportions.

- P4 64k

Hold LR at 2 × 10−6; retain the rebalanced full-data mixture, expand the context window from 32k to 64k, and heavily upsample long-form data.

#### 5.2 Pretraining Stage

For brevity, we denote the five training data subsets used in the curriculum as follows: the main filtered corpus 𝒟basic, timbre-cloning pairs 𝒟clone, dictionary-style short-form data 𝒟dict, noisy-text augmentation data 𝒟noise, and phonetic augmentation data 𝒟phone. The curriculum design follows three principles. First, we begin with the highest-density and least ambiguous supervision to maximize early learning efficiency. Second, harder conditioning tasks such as voice cloning and pronunciation control are introduced while the optimizer is still in a high-learning-rate stable region, so that they become native behaviors rather than narrow later-stage patches. Third, long-context extension is deferred until the short-context model has largely converged, which substantially reduces optimization instability and preserves short-utterance quality. The full schedule follows a simple warmup–stable–decay (WSD) pattern [58]: the learning rate is warmed up only in Phase 1, held fixed in the other non-decay phases, and linearly decayed from 2×10−4 to 2×10−6 only in Phase 3.

Phase1: basicalignmentacquisition. Webeginwith𝒟basic only, coveringChinese, English, andlower-resource

languages under relatively clean and direct text–speech supervision. This phase includes the only explicit warmup in the whole training schedule: the learning rate is first increased to 2×10−4 and then held fixed for the remainder of the phase. Excluding more specialized objectives at this stage improves sample efficiency: the model first learns monotonic text-to-audio alignment, multilingual grapheme-to-acoustic mapping, and the basic semantics encoded by the tokenizer before it is asked to solve voice transfer or pronunciationcorrection tasks. Empirically, this stage produces a substantially stronger initialization for the subsequent mixed-data phases than training on the full heterogeneous mixture from step zero.

- Phase 2: capability expansion under stable high LR. After the base mapping is established, we switch to the

full data universe and deliberately assign a much higher sampling weight to 𝒟clone. The reason is strategic: prompt-conditioned timbre transfer is both harder and more fragile than ordinary text-to-speech, and if it is introduced too weakly it tends to remain a tail capability. Keeping the learning rate fixed at 2×10−4 while the model is exposed to the full control-oriented data mixture allows the backbone to absorb cloning, dictionary reading, noisy-text robustness, and phonetic prompting as first-class behaviors rather than as late patches.

- Phase 3: linear-decay mixture rebalancing and quality consolidation. Once the control capabilities are in place, we keep the full data universe active but restore the mixture to its normal proportions, while linearly decaying the learning rate from 2 × 10−4 to 2 × 10−6 over the entire phase. This step is critical. Oversampling timbre-cloning data for too long biases the model toward prompt copying and can suppress the relative influence of standard multilingual TTS, dictionary coverage, and robustness-oriented augmentations. Phase 3 therefore serves as the main quality-consolidation stage: the model revisits the full task distribution while the optimizer transitions from a still-flexible high-LR regime to a highly conservative low-LR regime. In WSD terms, this is the decay segment in which most of the final gains are consolidated [58]. Early in the phase, the remaining relatively large updates are still sufficient to repair mixture imbalance and absorb residual capability gaps; late in the phase, the much smaller updates improve stability, reduce hallucination-like failures, and sharpen the final tradeoff among intelligibility, speaker similarity, and controllability.

- Phase 4: long-context extension. In the final phase, we keep the learning rate fixed at 2 × 10−6, increase the maximum sequence length from 32k to 64k, and heavily upsample long-form data. We intentionally do not introduce this longer context earlier. Training with a very long window from the beginning is significantly less efficient, because most examples do not require it and because early optimization is better spent learning the core text–speech mapping than fitting long-range attention patterns. Instead, we follow a late contextextension strategy analogous to recent LLM and TTS systems: once the base distribution has converged at moderate context length, the model is adapted to longer contexts under a small learning rate, which preserves short-form quality while teaching paragraph-scale and hour-scale continuity [1, 59, 60]. The heavy upsampling of long-form data is important here: without it, the nominal 64k window would be underutilized by the natural length distribution of the corpus. This stage is intended to improve speaker consistency across long generations, reduce drift in prosody and content over extended passages, and enable the model to use longer prompt speech without destabilizing decoding.

Learning-rate shape and practical rationale. Viewed globally, the four phases form a simple WSD-style training program rather than four disconnected runs: a short warmup embedded in Phase 1 lifts the learning rate to 2 × 10−4, Phases 1–2 then share a stable plateau at 2 × 10−4, Phase 3 linearly decays the learning rate from 2 × 10−4 to 2 × 10−6, and Phase 4 holds the final low learning rate at 2 × 10−6 during long-context adaptation. This schedule combines the optimization efficiency of a long stable high-LR region with the reliability of a gradual decay into a low-LR refinement regime. The distinction matters in practice: the stable plateau is where the model acquires its main multilingual TTS and controllability behaviors, whereas the linear decay phase is where those behaviors are rebalanced and polished without the abrupt optimization shock that would come from dropping directly from 2 × 10−4 to 2 × 10−6. By the time training enters Phase 4, the optimizer is already in a conservative regime, which allows us to upsample long-form data and extend the context window with minimal damage to established short-form quality. Compared with a one-shot fulldata recipe, the staged curriculum yields a better division of labor across phases: P1 learns the multilingual TTS prior, P2 makes control abilities robust, P3 restores distributional balance while progressively refining the model, and P4 transfers the already-competent model to longer contexts. This progression serves as a practical compromise between training efficiency, controllability, and long-form robustness, and it forms the default pretraining recipe used for the full MOSS-TTS release.

### 6 Evaluation

We evaluate MOSS-TTS from two complementary perspectives: (i) the audio tokenizer—whether it provides high-fidelity and semantically usable units across bitrates and domains—and (ii) the speech generation model—whether discrete autoregressive modeling and large-scale pretraining yield strong zero-shot voice cloning, multilingual robustness, token-level duration control, phoneme-/pinyin-level pronunciation control, and ultra-long speech generation. For the speech generation model, we report results for both MOSSTTS and MOSS-TTS-Local-Transformer. Following influential TTS technical reports [1, 2, 61], we prioritize objective metrics that are easy to reproduce and interpret: content consistency measured by WER/CER using a fixed ASR backend, speaker similarity (SIM) measured by cosine similarity of pretrained speaker embeddings, and task-specific metrics for controllability and long-form behavior.

#### 6.1 Audio Tokenizer

We conduct a comprehensive evaluation of MOSS-Audio-Tokenizer, comparing it with current state-of-theart open-source audio tokenizers across various bitrate regimes. The baseline audio tokenizers include StableCodec[62], XCodec2.0[63], MiMo-Audio-Tokenizer[42], Higgs-Audio-Tokenizer[64], SpeechTokenizer[7], XY-Tokenizer [21], BigCodec [65], Mimi [8], DAC [18], Encodec [17], and Qwen3-TTS-Tokenizer [1]. Our evaluation encompasses speech, general audio, and music to assess the model’s versatility and reconstruction fidelity.

For speech reconstruction, we conduct evaluations on LibriSpeech test-clean (English) [66] and AISHELL-

- Table 2 Reconstruction quality comparison of open-source audio tokenizers on speech and audio/music data. Speech metrics are evaluated on LibriSpeech test-clean (English) and AISHELL-2 (Chinese) and reported as English/Chinese. Audiometricsareevaluatedonthe AudioSetevaluationsubset, whilemusic metricsareevaluatedontheMUSDB dataset; values are reported as audio/music. STFT-Dist. denotes the STFT distance. Higher is better for speech metrics, whereas

lower is better for audio/music metrics. 𝑵VQ denotes the number of quantizers. Bold entries indicate the best result within each bitrate regime.

Speech Audio / Music

Frame rate

Model bps

𝑵VQ

SIM↑ STOI↑ PESQ-NB↑ PESQ-WB↑ Mel-Loss↓ STFT-Dist.↓ StableCodec 700 25 2 0.62 / 0.45 0.91 / 0.86 2.91 / 2.50 2.24 / 1.93 – / – – / –

XCodec2.0 800 50 1 0.82 / 0.74 0.92 / 0.86 3.04 / 2.46 2.43 / 1.96 – / – – / –

MiMo-Audio-Tokenizer 850 25 4 0.80 / 0.74 0.91 / 0.87 2.94 / 2.62 2.39 / 2.14 0.82 / 0.81 2.33 / 2.23 Higgs-Audio-Tokenizer 1000 25 4 0.77 / 0.68 0.83 / 0.82 3.03 / 2.61 2.48 / 2.14 0.83 / 0.80 2.20 / 2.05 SpeechTokenizer 1000 50 2 0.36 / 0.25 0.77 / 0.68 1.59 / 1.38 1.25 / 1.17 – / – – / –

XY-Tokenizer 1000 12.5 8 0.85 / 0.79 0.92 / 0.87 3.10 / 2.63 2.50 / 2.12 – / – – / – BigCodec 1040 80 1 0.84 / 0.69 0.93 / 0.88 3.27 / 2.55 2.68 / 2.06 – / – – / –

Mimi 1100 12.5 8 0.74 / 0.59 0.91 / 0.85 2.80 / 2.24 2.25 / 1.78 1.24 / 1.19 2.62 / 2.49 MOSS-Audio-Tokenizer 750 12.5 6 0.82 / 0.75 0.93 / 0.89 3.14 / 2.73 2.60 / 2.22 0.86 / 0.85 2.21 / 2.10 MOSS-Audio-Tokenizer 1000 12.5 8 0.88 / 0.81 0.94 / 0.91 3.38 / 2.96 2.87 / 2.43 0.82 / 0.80 2.16 / 2.04

DAC 1500 75 2 0.48 / 0.41 0.83 / 0.79 1.87 / 1.67 1.48 / 1.37 – / – – / –

Encodec 1500 75 2 0.60 / 0.45 0.85 / 0.81 1.94 / 1.80 1.56 / 1.48 1.12 / 1.04 2.60 / 2.42 Higgs-Audio-Tokenizer 2000 25 8 0.90 / 0.83 0.85 / 0.85 3.59 / 3.22 3.11 / 2.73 0.74 / 0.70 2.07 / 1.92 SpeechTokenizer 2000 50 4 0.66 / 0.50 0.88 / 0.80 2.38 / 1.79 1.92 / 1.49 – / – – / – Qwen3-TTS-Tokenizer 2200 12.5 16 0.95 / 0.88 0.96 / 0.93 3.66 / 3.10 3.19 / 2.62 – / – – / –

MiMo-Audio-Tokenizer 2250 25 12 0.89 / 0.83 0.95 / 0.92 3.57 / 3.25 3.05 / 2.71 0.70 / 0.68 2.21 / 2.10

Mimi 2475 12.5 18 0.89 / 0.76 0.94 / 0.91 3.49 / 2.90 2.97 / 2.35 1.10 / 1.06 2.45 / 2.32 MOSS-Audio-Tokenizer 1500 12.5 12 0.92 / 0.86 0.95 / 0.93 3.64 / 3.27 3.20 / 2.74 0.77 / 0.74 2.08 / 1.96 MOSS-Audio-Tokenizer 2000 12.5 16 0.95 / 0.89 0.96 / 0.94 3.78 / 3.46 3.41 / 2.96 0.73 / 0.70 2.03 / 1.90

DAC 3000 75 4 0.74 / 0.67 0.90 / 0.88 2.76 / 2.47 2.31 / 2.07 0.86 / 0.83 2.23 / 2.10 MiMo-Audio-Tokenizer 3650 25 20 0.91 / 0.85 0.95 / 0.93 3.73 / 3.44 3.25 / 2.89 0.66 / 0.65 2.17 / 2.06 SpeechTokenizer 4000 50 8 0.85 / 0.69 0.92 / 0.85 3.05 / 2.20 2.60 / 1.87 – / – – / –

Mimi 4400 12.5 32 0.94 / 0.83 0.96 / 0.94 3.80 / 3.31 3.43 / 2.78 1.02 / 0.98 2.34 / 2.21 Encodec 4500 75 6 0.86 / 0.75 0.92 / 0.91 2.91 / 2.63 2.46 / 2.15 0.91 / 0.84 2.33 / 2.17

DAC 6000 75 8 0.89 / 0.84 0.95 / 0.94 3.75 / 3.57 3.41 / 3.20 0.65 / 0.63 1.97 / 1.87 MOSS-Audio-Tokenizer 3000 12.5 24 0.96 / 0.92 0.97 / 0.96 3.90 / 3.64 3.61 / 3.20 0.69 / 0.66 1.98 / 1.84 MOSS-Audio-Tokenizer 4000 12.5 32 0.97 / 0.93 0.97 / 0.96 3.95 / 3.71 3.69 / 3.30 0.68 / 0.64 1.96 / 1.82

2 (Chinese) [67]. We report speaker similarity (SIM), computed as the cosine similarity between speaker embeddings extracted from the original and reconstructed audio using a pretrained speaker verification model2. In addition, we report short-time objective intelligibility (STOI) [68] and perceptual evaluation of speech quality (PESQ) [69].

For sound and music reconstruction, following prior work [18], we evaluate on the AudioSet evaluation subset [70] and MUSDB [71]. We report mel-spectrogram distance and short-time Fourier transform (STFT) distance as objective metrics.

- Table 2 summarizes the objective reconstruction results across speech, general audio, and music benchmarks. We categorize the performance into low (750–1500bps), medium (1500–2500bps), and high (2500–6000bps) bitrate regimes. Additionally, Figure 5 illustrates the performance trajectory of MOSS-Audio-Tokenizer against other open-source alternatives within the 0–4 kbps range. Across all evaluated bitrates, MOSS-AudioTokenizer consistently outperforms the compared open-source baselines in speech reconstruction. On general audio and music benchmarks, the model maintains competitive performance. Notably, reconstruction 2UniSpeech speaker verification repository

[Figure 51]

- Figure 5 Comparison of objective reconstruction metrics between MOSS-Audio-Tokenizer and other state-of-the-art open-source audio tokenizers on the LibriSpeech test-clean dataset. Results are evaluated within the 0–4kbps bitrate range. The horizontal axis represents the bitrate, and the vertical axis denotes the corresponding objective reconstruction scores.

quality scales gracefully with the increase in bitrate, demonstrating that the model effectively leverages additional capacity and bitrate through its joint end-to-end optimization framework.

These results indicate strong modeling capacity for MOSS-Audio-Tokenizer across both low-bitrate and highbitrate regimes. By allowing a flexible selection of RVQ layers, the model can be adapted to diverse application requirements, spanning low-bitrate scenarios to high-fidelity audio generation. Overall, MOSS-AudioTokenizer provides a stable, high-fidelity, and standardized tokenizer for native audio generation models.

#### 6.2 Voice Cloning

- Table 3 compares MOSS-TTS with representative open and closed systems on Seed-TTS-eval. We report both architectures in both inference modes. Results for non-MOSS baselines are collected from the corresponding technical reports and reported as given in those sources.

For prompt-conditioned generation, we distinguish two inference paradigms throughout this section. In Clone, the user input explicitly provides a reference audio clip. In Continuation, we instead prepend the reference audio to the assistant-side speech prefix, prepend its ASR transcript to the requested text, and let the model continue generating the speech for the original text. We report both modes because they probe different uses of the same pretrained model: Clone measures explicit reference-audio conditioning, whereas Continuation tests whether native speech continuation already provides usable timbre transfer without relying on a dedicated clone-style prompt format.

On Seed-TTS-eval, speaker similarity is the more informative metric. Once WER/CER is already below

- Table 3 Zero-shot voice cloning on Seed-TTS-eval. We report English WER (↓), English speaker similarity (SIM, ↑), Chinese CER (↓), and Chinese SIM (↑). The evaluation results are from technical reports of other models, such as VoxCPM [61] and SparkTTS [72].

###### Model Mode Params Open EN WER ↓ EN SIM ↑ ZH CER ↓ ZH SIM ↑

DiTAR – 0.6B 1.69 73.50 1.02 75.30 FishAudio-S1 – 4B 1.72 62.57 1.22 72.10 CosyVoice3 – 1.5B 2.22 72.00 1.12 78.10 Seed-TTS – – 2.25 76.20 1.12 79.60 MiniMax-Speech – – 1.65 69.20 0.83 78.30

CosyVoice – 0.3B 4.29 60.90 3.63 72.30

- CosyVoice2 – 0.5B 3.09 65.90 1.38 75.70
- CosyVoice3 – 0.5B 2.02 71.80 1.16 78.00 F5-TTS – 0.3B 2.00 67.00 1.53 76.00 SparkTTS – 0.5B 3.14 57.30 1.54 66.00 FireRedTTS – 0.5B 3.82 46.00 1.51 63.50 FireRedTTS-2 – 1.5B 1.95 66.50 1.14 73.60 Qwen2.5-Omni – 7B 2.72 63.20 1.70 75.20 FishAudio-S1-mini – 0.5B 1.94 55.00 1.18 68.50 IndexTTS2 – 1.5B 2.23 70.60 1.03 76.50 VibeVoice – 1.5B 3.04 68.90 1.16 74.40 HiggsAudio-v2 – 3B 2.44 67.70 1.50 74.00 GLM-TTS – 1.5B 2.23 67.2 1.03 76.1 GLM-TTS-RL – 1.5B 1.91 68.1 0.89 76.4 VoxCPM – 0.5B 1.85 72.90 0.93 77.20

- Qwen3-TTS – 0.6B 1.68 70.39 1.23 76.40
- Qwen3-TTS – 1.7B 1.50 71.45 1.33 76.72

Clone 8B 1.92 69.31 1.46 76.21 Continuation 8B 1.84 70.86 1.37 76.98

MOSS-TTS

Clone 1.7B 1.87 71.74 1.33 77.24 Continuation 1.7B 1.93 73.28 1.44 79.62

MOSS-TTS-Local-Transformer

about 2, residual differences become hard to interpret: in our manual review, most remaining mismatches in that regime are ASR errors rather than audible pronunciation failures. Under this lens, MOSS-TTS is particularly strong on SIM. For both architectures, Continuation consistently improves speaker similarity over Clone, indicating that native speech continuation is an effective way to anchor speaker identity. MOSS-TTSLocal-Transformer is consistently stronger than MOSS-TTS on speaker preservation despite using only 1.7B parameters, and MOSS-TTS-Local-Transformer in Continuation achieves the highest Chinese and English similarity scores among the open-source models in the table. This matches the architectural tradeoff discussed in Section 4.1 and Section 4.2: MOSS-TTS-Local-Transformer is the more modeling-efficient architecture for speaker preservation, while MOSS-TTS remains the simpler long-context backbone used in the control-oriented evaluations below.

#### 6.3 Multilingual Voice Cloning

We evaluate the released pretrained checkpoints directly on the CV3-Eval multilingual voice cloning subset, without any task-specific fine-tuning or post-training for this benchmark. As shown in Table 4, this subset probes voice cloning across a larger language set than Seed-TTS-eval. We report Clone and Continuation separately for both released MOSS-TTS architectures. External baseline entries in Table 4 are filled only where corresponding values are provided in the cited reports.

- As shown in Table 4, even without benchmark-specific multilingual cloning training, MOSS-TTS remains competitive across several non-zh/en languages. Relative to strong open baselines, it shows stable perfor-

- Table 4 CER(%) and WER(%) on the CV3-Eval Multilingual Voice Cloning subset. “–” means the language is unsupported.

Model Mode zh en ja ko de es fr it ru

F5-TTS – 5.47 8.90 – – – – – – – Spark-TTS – 5.15 11.00 – – – – – – – GPT-SoVits – 7.34 12.50 – – – – – – – CosyVoice2 – 4.08 6.32 9.13 19.7 – – – – – CosyVoice2+DiffRO – 3.00 4.72 6.36 5.14 – – – – –

- CosyVoice3-0.5B – 3.89 5.24 10.4 12.8 7.41 4.25 12.9 6.68 6.77

- CosyVoice3-0.5B+DiffRO – 2.89 3.68 5.15 4.02 4.51 2.99 8.56 2.94 3.79
- CosyVoice3-1.5B – 3.91 4.99 7.57 5.69 6.43 4.47 11.8 10.5 6.64

- CosyVoice3-1.5B+DiffRO – 3.01 3.71 5.27 4.01 3.93 3.26 8.09 2.72 4.11

MOSS-TTS

Clone 4.42 4.92 10.72 6.33 4.70 4.36 11.17 5.46 6.37 Continuation 4.26 5.12 7.78 7.73 10.83 3.43 10.59 4.82 6.64

MOSS-TTS-Local-Transformer

Clone 3.95 4.35 10.10 5.95 4.28 3.98 10.32 5.02 5.90 Continuation 3.68 4.89 7.30 7.20 10.20 3.10 9.90 4.40 6.20

- Table 5 Token-level duration control. Relative duration error (%) across target-duration buckets. AbsErr Mean: mean absolute relative error; AbsErr P50/P90: 50th/90th percentile of absolute relative error; RMSE: root mean squared relative error.

Language Bucket AbsErr Mean (%) ↓ AbsErr P50 (%) ↓ AbsErr P90 (%) ↓ RMSE (%) ↓

zh 3s–10s 1.456 1.333 2.343 1.652 zh 10s–1m 0.359 0.254 0.647 0.502 zh 1m–10m 0.356 0.077 1.273 0.849 zh 10m–30m 0.678 0.061 1.859 1.228 zh overall 0.712 0.284 2.013 1.141

en 3s–10s 1.482 1.357 2.385 1.685 en 10s–1m 0.355 0.251 0.639 0.515 en 1m–10m 0.365 0.079 1.304 0.834 en 10m–30m 0.660 0.059 1.809 1.261 en overall 0.723 0.288 2.043 1.160

mance on de/es/it/ru, and the Continuation setting remains usable across the broader language set despite being a harder zero-shot transfer setting. Table 4 also shows that the largest gaps are concentrated in a few harder language pairs such as ja/ko and in some English continuation cases, which is consistent with the overall difficulty of this subset.

#### 6.4 Duration Control

From this subsection onward, we report only MOSS-TTS. The remaining three evaluations in this sectionduration control, ultra-long speech generation, and phoneme-/pinyin-level pronunciation control—stress explicit token conditioning and long-context continuation, where the delay architecture is the more practical release target because of its simpler single-backbone parameterization and better scalability at long sequence lengths. We therefore use MOSS-TTS-Local-Transformer primarily to characterize the similarity-quality tradeoff in the cloning benchmarks above.

We evaluate token-level duration control on MOSS-TTS by prompting the model with a target token count and measuring the relative duration error. Under our tokenizer, 1 second corresponds to 12.5 audio tokens. Given a target token count 𝑛, the target duration is𝑇target = 𝑛/12.5 seconds; we compute the realized duration 𝑇real from the generated waveform and report Err% = |𝑇real − 𝑇target|/𝑇target × 100%. We summarize errors by language and target-duration buckets.

- Table 6 Ultra-long speech generation on an internal evaluation set. Chinese reports CER (%) and English reports WER (%), each averaged over 10 prompts per bucket. SIM is reported in percentage form, where 100 corresponds to perfect cosine similarity. It is computed by averaging 3-second window scores within each utterance and then averaging over utterances in the same bucket. This internal set is used only to characterize expected behavior in ultra-long generation.

###### Language Bucket Clone CER/WER ↓ Continuation CER/WER ↓ Clone SIM (%) ↑ Continuation SIM (%) ↑

zh 10-100 0.83 0.65 69.3 69.9 zh 100-500 1.53 0.85 65.6 66.0 zh 500-2500 4.12 0.94 64.9 66.2 zh 2500-5000 3.46 1.19 63.4 66.3 zh 5000-10000 1.89 1.87 63.1 64.7 zh 10000+ 3.41 1.86 60.1 63.0

en 50-500 4.63 6.63 64.6 63.5 en 500-2500 3.65 4.08 60.9 60.2 en 2500-12500 3.75 4.05 60.3 60.0 en 12500-25000 3.76 3.75 55.5 56.5 en 25000-50000 4.58 6.50 54.8 53.3 en 50000+ 17.49 29.52 44.4 51.2

10-100

500-2500

5000-10000

100-500

2500-5000

10000+

Clone

Continuation

72

SpeakerSimilarity(%)

64

56

48

0 8 16 24 32 Elapsed Time (min)

0 8 16 24 32 Elapsed Time (min)

(a) Chinese

50-500

2500-12500

25000-50000

500-2500

12500-25000

50000+

Clone

Continuation

70

SpeakerSimilarity(%)

60

50

40

0 10 20 30 40 50 Elapsed Time (min)

0 8 16 24 32 40 Elapsed Time (min)

(b) English

- Figure 6 Speaker similarity drift under ultra-long generation. Each curve averages non-overlapping 3-second window similarities within a length bucket. For readability, the visualization reports 30-second bins and only keeps the time prefix where at least eight utterances remain in the bucket.

- As shown in Table 5, the model achieves consistently low relative duration errors from short to long utterances, with overall AbsErr Mean around 0.7% and strong percentile behavior. Notably, these results are obtained under a pretraining-only setup, indicating that effective token-level duration control can emerge without introducing a dedicated duration-control fine-tuning stage.

#### 6.5 Ultra-Long Speech Generation

We further build an internal ultra-long evaluation set for MOSS-TTS to estimate expected behavior when generation extends from short utterances to approximately one hour. The set covers Chinese and English, each with six language-specific text-length buckets and 10 prompts per bucket. For each prompt, we evaluate both Clone and Continuation, yielding 240 generated utterances in total. We transcribe each sample with MOSS-Transcribe-Diarize [55] and report CER for Chinese and WER for English, each computed per sample and averaged over the 10 prompts in each bucket. Speaker similarity (SIM) is computed as the mean cosine similarity over non-overlapping 3-second windows. This internal set is used only to characterize expected performance in ultra-long generation rather than to serve as a public benchmark.

- Table 6 is best read as a coarse bucket-level summary. Content fidelity remains usable through most buckets and degrades mainly at the longest horizons, while the average SIM values already suggest that speaker preservation weakens earlier than lexical accuracy. The more informative signal, however, is the temporal drift profile in Figure 6.

- Table 7 Phoneme-/pinyin-level pronunciation control on an internal evaluation set. We report span-only CER for Chinese and span-only WER for English. Lower is better.

Language Setting Replaced-Span CER/WER ↓

zh partial-replace 1.00 zh full-replace 1.65 en partial-replace 4.32 en full-replace 5.84

Figure 6 makes the failure mode explicit. In Chinese, most buckets begin in a narrow high-SIM band. Under Clone, the short and medium buckets stay fairly flat, but the 10000+ bucket shows a clear late-stage collapse near the tail. Under Continuation, the curves are much tighter and flatter: even the longest bucket stays close to the others for more than 30 minutes, indicating substantially better long-horizon speaker anchoring. English is harder. All buckets drift downward earlier, and the 50000+ bucket under Clone falls the fastest and separates from the shorter buckets after only a few minutes. Continuation does not remove this trend, but it clearly raises and smooths the long-bucket trajectories, especially for the 25000–50000 and 50000+ settings. The main conclusion from Figure 6 is therefore that ultra-long generation remains operational, but the dominant bottleneck is cumulative speaker drift over elapsed time rather than immediate lexical failure.

#### 6.6 Phoneme-/Pinyin-Level Pronunciation Control

We conduct a small internal functionality evaluation for phoneme-/pinyin-level pronunciation control using MOSS-TTS. For each language (Chinese and English), we construct two settings: partial-replace, where only a short target span is replaced by pinyin or IPA, and full-replace, where the entire sentence is specified in pinyin or IPA. Each language-setting pair contains 100 samples. Since the goal of this test is to verify controllable pronunciation editing, we evaluate only the controlled span rather than the full sentence.

We transcribe each generated utterance with MOSS-Transcribe-Diarize [55], align the transcript to the target text, and compute span-only CER (Chinese) and WER (English). As shown in Table 7, MOSS-TTS achieves low span error in all four settings on this internal set, indicating that phoneme-/pinyin-level control is already practically usable, including both local span replacement and full-sentence phoneme control.

### 7 Conclusion

In this technical report, we presented MOSS-TTS, an open speech generation foundation model built on a scalable recipe: a high-quality audio tokenizer, autoregressive next-token modeling, and large-scale multilingual pretraining. Built on MOSS-Audio-Tokenizer, MOSS-TTS formulates speech generation as autoregressive prediction over aligned text and speech tokens. On top of this tokenizer, MOSS-TTS and MOSSTTS-Local-Transformer instantiate two complementary operating points: the former emphasizes structural simplicity, scalability, and long-context/control-oriented deployment, while the latter emphasizes higher modeling efficiency, stronger speaker preservation, and a shorter time to first audio.

The empirical results support the central thesis of the report. MOSS-Audio-Tokenizer provides strong discreteaudiotokensacrossbitrateregimes, andthetwoarchitecturesexposeaclearandpracticallyusefultradeoff: MOSS-TTS-Local-Transformer is generally stronger on speaker similarity in zero-shot cloning, whereas MOSS-TTS is the more natural backbone for duration control and ultra-long generation. At the same time, the evaluation makes the remaining bottlenecks explicit. The hardest multilingual setting still leave room for improvement, and ultra-long generation shows that long-horizon speaker drift—rather than immediate lexical failure—is now the dominant failure mode, especially in English. We therefore view stronger long-context speaker anchoring, broader low-resource language coverage, and further improvement of finegrained controllability as the most important next directions.

Taken together, these results suggest that speech generation can benefit from the same principles that have

driven recent progress in open large language models: data quality, scale, and architectural simplicity. Rather than relying on increasingly elaborate cascades, MOSS-TTS shows that a strong tokenizer, a largescale high-quality data pipeline, and a unified autoregressive objective already provide a practical foundation for open speech generation. With the release of MOSS-Audio-Tokenizer, MOSS-TTS, and MOSS-TTSLocal-Transformer, we hope this report can serve both as a reproducible account of the current release and as a clean baseline for future work on open speech foundation models.

### Contributors

Core Contributors: Yitian Gong†, Botian Jiang†, Yiwei Zhao, Yucheng Yuan, Kuangwei Chen, Yaozhou Jiang, Cheng Chang, Dong Hong, Mingshu Chen, Ruixiao Li, Yiyang Zhang, Yang Gao, Hanfu Chen, Ke Chen, Songlin Wang, Xiaogui Yang∗ Contributors: Yuqian Zhang, Kexin Huang, ZhengYuan Lin, Kang Yu, Ziqi Chen, Jin Wang, Zhaoye Fei, Qinyuan Cheng, Shimin Li Advisors: Xipeng Qiu§

Affiliations: Shanghai Innovation Institute MOSI Intelligence Fudan University

†Equal contribution. ∗Project lead. §Corresponding author: xpqiu@fudan.edu.cn. We especially thank the Infrastructure and Data teams for their essential contributions to the MOSS-TTS release.

### References

- [1] Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, Xiong Wang, Zhifang Guo, Ziyue Jiang, Hongkun Hao, Zishan Guo, et al. Qwen3-tts technical report. arXiv preprint arXiv:2601.15621, 2026.
- [2] Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, et al. Cosyvoice 3: Towards in-the-wild speech generation via scaling-up and post-training. arXiv preprint arXiv:2505.17589, 2025.
- [3] Philip Anastassiou, Jiawei Chen, Jitong Chen, Yuanzhe Chen, Zhuo Chen, Ziyi Chen, Jian Cong, Lelai Deng, Chuang Ding, Lu Gao, et al. Seed-tts: A family of high-quality versatile speech generation models. arXiv preprint arXiv:2406.02430, 2024.
- [4] Kun Xie, Feiyu Shen, Junjie Li, Fenglong Xie, Xu Tang, and Yao Hu. Fireredtts-2: Towards long conversational speech generation for podcast and chatbot. arXiv preprint arXiv:2509.02020, 2025.
- [5] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- [6] Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020.
- [7] Xin Zhang, Dong Zhang, Shimin Li, Yaqian Zhou, and Xipeng Qiu. Speechtokenizer: Unified speech tokenizer for speech large language models. arXiv preprint arXiv:2308.16692, 2023.
- [8] Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. Moshi: a speech-text foundation model for real-time dialogue. arXiv preprint arXiv:2410.00037, 2024.
- [9] Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, et al. Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. arXiv preprint arXiv:2407.05407, 2024.
- [10] Yuancheng Wang, Haoyue Zhan, Liwei Liu, Ruihong Zeng, Haotian Guo, Jiachen Zheng, Qiang Zhang, Xueyao Zhang, Shunsi Zhang, and Zhizheng Wu. Maskgct: Zero-shot text-to-speech with masked generative codec transformer. arXiv preprint arXiv:2409.00750, 2024.
- [11] Yitian Gong, Kuangwei Chen, Zhaoye Fei, Xiaogui Yang, Ke Chen, Yang Wang, Kexin Huang, Mingshu Chen, Ruixiao Li, Qingyuan Cheng, et al. Moss-audio-tokenizer: Scaling audio tokenizers for future audio foundation models. arXiv preprint arXiv:2602.10934, 2026.
- [12] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM transactions on audio, speech, and language processing, 29:3451–3460, 2021.
- [13] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023.
- [14] Zhen Ye, Peiwen Sun, Jiahe Lei, Hongzhan Lin, Xu Tan, Zheqi Dai, Qiuqiang Kong, Jianyi Chen, Jiahao Pan, Qifeng Liu, et al. Codec does matter: Exploring the semantic shortcoming of codec for audio language model. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 25697–25705, 2025.
- [15] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.
- [16] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. Soundstream: An endto-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30:495–507, 2021.
- [17] Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. High fidelity neural audio compression. arXiv preprint arXiv:2210.13438, 2022.

- [18] Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, and Kundan Kumar. High-fidelity audio compression with improved rvqgan. Advances in Neural Information Processing Systems, 36:27980–27993, 2023.
- [19] Yi-Chiao Wu, Israel D Gebru, Dejan Marković, and Alexander Richard. Audiodec: An open-source streaming high-fidelity neural audio codec. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.
- [20] Zhihao Du, Shiliang Zhang, Kai Hu, and Siqi Zheng. Funcodec: A fundamental, reproducible and integrable opensource toolkit for neural speech codec. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 591–595. IEEE, 2024.
- [21] Yitian Gong, Luozhĳie Jin, Ruifan Deng, Dong Zhang, Xin Zhang, Qinyuan Cheng, Zhaoye Fei, Shimin Li, and Xipeng Qiu. Xy-tokenizer: Mitigating the semantic-acoustic conflict in low-bitrate speech codecs. arXiv preprint arXiv:2506.23325, 2025.
- [22] Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, et al. Audiolm: a language modeling approach to audio generation. IEEE/ACM transactions on audio, speech, and language processing, 31:2523–2533, 2023.
- [23] Haibin Wu, Xuanjun Chen, Yi-Cheng Lin, Kai-wei Chang, Ho-Lam Chung, Alexander H Liu, and Hung-yi Lee. Towards audio language modeling–an overview. arXiv preprint arXiv:2402.13236, 2024.
- [24] Siddique Latif, Moazzam Shoukat, Fahad Shamshad, Muhammad Usama, Yi Ren, Heriberto Cuayáhuitl, Wenwu Wang, Xulong Zhang, Roberto Togneri, Erik Cambria, et al. Sparks of large audio models: A survey and outlook. arXiv preprint arXiv:2308.12792, 2023.
- [25] Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, et al. Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint arXiv:2301.02111, 2023.
- [26] Dong Zhang, Xin Zhang, Jun Zhan, Shimin Li, Yaqian Zhou, and Xipeng Qiu. Speechgpt-gen: Scaling chain-ofinformation speech generation. arXiv preprint arXiv:2401.13527, 2024.
- [27] Dongchao Yang, Songxiang Liu, Haohan Guo, Jiankun Zhao, Yuanyuan Wang, Helin Wang, Zeqian Ju, Xubo Liu, Xueyuan Chen, Xu Tan, et al. Almtokenizer: A low-bitrate and semantic-rich audio codec tokenizer for audio language modeling. arXiv preprint arXiv:2504.10344, 2025.
- [28] Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. arXiv preprint arXiv:1609.03499, 2016.
- [29] Yuxuan Wang, RJ Skerry-Ryan, Daisy Stanton, Yonghui Wu, Ron J Weiss, Navdeep Jaitly, Zongheng Yang, Ying Xiao, Zhifeng Chen, Samy Bengio, and Quoc Le. Tacotron: Towards end-to-end speech synthesis. arXiv preprint arXiv:1703.10135, 2017.
- [30] Jonathan Shen, Ruoming Pang, Ron J Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, et al. Natural TTS synthesis by conditioning WaveNet on mel spectrogram predictions. arXiv preprint arXiv:1712.05884, 2018.
- [31] Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. FastSpeech: Fast, robust and controllable text to speech. arXiv preprint arXiv:1905.09263, 2019.
- [32] Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. FastSpeech 2: Fast and high-quality end-to-end text to speech. arXiv preprint arXiv:2006.04558, 2020.
- [33] Jaehyeon Kim, Sungwon Kim, Jungil Kong, and Juhee Son. Glow-TTS: A generative flow for text-to-speech via monotonic alignment search. arXiv preprint arXiv:2005.11129, 2020.
- [34] Vadim Popov, Ivan Vovk, Vardan Gogoryan, Tatiana Sadekova, and Mikhail Kudinov. Grad-TTS: A diffusion probabilistic model for text-to-speech. arXiv preprint arXiv:2105.06337, 2021.
- [35] Jaehyeon Kim, Jungil Kong, and Juhee Son. Conditional variational autoencoder with adversarial learning for endto-end text-to-speech. In International Conference on Machine Learning, pages 5530–5540. PMLR, 2021.

- [36] James Betker. Better speech synthesis through scaling. arXiv preprint arXiv:2305.07243, 2023.
- [37] Shĳia Liao, Yuxuan Wang, Tianyu Li, Yifan Cheng, Ruoyi Zhang, Rongzhi Zhou, and Yĳin Xing. Fish-speech: Leveraging large language models for advanced multilingual text-to-speech synthesis. arXiv preprint arXiv:2411.01156, 2024.
- [38] Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, Wei-Ning Hsu, et al. Voicebox: Text-guided multilingual universal speech generation at scale. arXiv preprint arXiv:2306.15687, 2023.
- [39] Edresson Casanova, Julian Weber, Christopher Shulby, Arnaldo Candido Junior, Eren Gölge, and Moacir Antonelli Ponti. YourTTS: Towards zero-shot multi-speaker TTS and zero-shot voice conversion for everyone. arXiv preprint arXiv:2112.02418, 2022.
- [40] Yinghao Aaron Li, Cong Han, Vinay S Raghavan, Gavin Mischler, and Nima Mesgarani. StyleTTS 2: Towards human-level text-to-speech through style diffusion and adversarial training with large speech language models. arXiv preprint arXiv:2306.07691, 2023.
- [41] Tianpeng Li, Jun Liu, Tao Zhang, Yuanbo Fang, Da Pan, Mingrui Wang, Zheng Liang, Zehuan Li, Mingan Lin, Guosheng Dong, et al. Baichuan-audio: A unified framework for end-to-end speech interaction. arXiv preprint arXiv:2502.17239, 2025.
- [42] Dong Zhang, Gang Wang, Jinlong Xue, Kai Fang, Liang Zhao, Rui Ma, Shuhuai Ren, Shuo Liu, Tao Guo, Weĳi Zhuang, et al. Mimo-audio: Audio language models are few-shot learners. arXiv preprint arXiv:2512.23808, 2025.
- [43] Jiaqi Li, Xiaolong Lin, Zhekai Li, Shixi Huang, Yuancheng Wang, Chaoren Wang, Zhenpeng Zhan, and Zhizheng Wu. Dualcodec: A low-frame-rate, semantically-enhanced neural audio codec for speech generation. arXiv preprint arXiv:2505.13000, 2025.
- [44] Simon Welker, Matthew Le, Ricky TQ Chen, Wei-Ning Hsu, Timo Gerkmann, Alexander Richard, and Yi-Chiao Wu. Flowdec: A flow-based full-band general audio codec with high perceptual quality. arXiv preprint arXiv:2503.01485, 2025.
- [45] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
- [46] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [47] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [48] Xudong Mao, Qing Li, Haoran Xie, Raymond YK Lau, Zhen Wang, and Stephen Paul Smolley. Least squares generative adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 2794–2802, 2017.
- [49] Kundan Kumar, Rithesh Kumar, Thibault De Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre De Brebisson, Yoshua Bengio, and Aaron C Courville. Melgan: Generative adversarial networks for conditional waveform synthesis. Advances in neural information processing systems, 32, 2019.
- [50] Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, and Alexandre Défossez. Simple and controllable music generation. Advances in Neural Information Processing Systems, 36:47704–47720, 2023.
- [51] Shengkui Zhao, Bin Ma, and Shinji Watanabe. MossFormer2: Combining transformer and RNN-free recurrent network for enhanced time-domain monaural speech separation. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 11766–11770. IEEE, 2024.
- [52] Jiangyu Han, Petr Pálka, Marc Delcroix, Federico Landini, Johan Rohdin, Jan Cernockỳ, and Lukáš Burget. Efficient and generalizable speaker diarization via structured pruning of self-supervised models. arXiv preprint arXiv:2506.18623, 2025.
- [53] Jiangyu Han, Federico Landini, Johan Rohdin, Anna Silnova, Mireia Diez, Jan Cernocky, and Lukas Burget. Finetune before structured pruning: Towards compact and accurate self-supervised models for speaker diarization. arXiv preprint arXiv:2505.24111, 2025.

- [54] Jiangyu Han, Federico Landini, Johan Rohdin, Anna Silnova, Mireia Diez, and Lukáš Burget. Leveraging selfsupervised learning for speaker diarization. In Proc. ICASSP, 2025.
- [55] Donghua Yu, Zhengyuan Lin, Chen Yang, Yiyang Zhang, Zhaoye Fei, Hanfu Chen, Jingqi Chen, Ke Chen, Qinyuan Cheng, Liwei Fan, et al. Moss transcribe diarize: Accurate transcription with speaker diarization. arXiv preprint arXiv:2601.01554, 2026.
- [56] Chandan K. A. Reddy, Vishak Gopal, and Ross Cutler. DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 886–890. IEEE, 2022.
- [57] Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hoffman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, et al. Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound. arXiv preprint arXiv:2502.05139, 2025.
- [58] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.
- [59] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [60] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [61] Yixuan Zhou, Guoyang Zeng, Xin Liu, Xiang Li, Renjie Yu, Ziyang Wang, Runchuan Ye, Weiyue Sun, Jiancheng Gui, Kehan Li, et al. Voxcpm: Tokenizer-free tts for context-aware speech generation and true-to-life voice cloning. arXiv preprint arXiv:2509.24650, 2025.
- [62] Julian D Parker, Anton Smirnov, Jordi Pons, CJ Carr, Zack Zukowski, Zach Evans, and Xubo Liu. Scaling transformers for low-bitrate high-quality speech coding. arXiv preprint arXiv:2411.19842, 2024.
- [63] Zhen Ye, Xinfa Zhu, Chi-Min Chan, Xinsheng Wang, Xu Tan, Jiahe Lei, Yi Peng, Haohe Liu, Yizhu Jin, Zheqi Dai, et al. Llasa: Scaling train-time and inference-time compute for llama-based speech synthesis. arXiv preprint arXiv:2502.04128, 2025.
- [64] BosonAI. Higgs audio v2: Redefining expressiveness in audio generation. https://github.com/boson-ai/higgs-audio, 2025.
- [65] Detai Xin, Xu Tan, Shinnosuke Takamichi, and Hiroshi Saruwatari. Bigcodec: Pushing the limits of low-bitrate neural speech codec. arXiv preprint arXiv:2409.05377, 2024.
- [66] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 5206–5210. IEEE, 2015.
- [67] Jiayu Du, Xingyu Na, Xuechen Liu, and Hui Bu. Aishell-2: Transforming mandarin asr research into industrial scale. arXiv preprint arXiv:1808.10583, 2018.
- [68] Cees H Taal, Richard C Hendriks, Richard Heusdens, and Jesper Jensen. A short-time objective intelligibility measure for time-frequency weighted noisy speech. In 2010 IEEE international conference on acoustics, speech and signal processing, pages 4214–4217. IEEE, 2010.
- [69] Antony W Rix, John G Beerends, Michael P Hollier, and Andries P Hekstra. Perceptual evaluation of speech quality (pesq)-a new method for speech quality assessment of telephone networks and codecs. In 2001 IEEE international conference on acoustics, speech, and signal processing. Proceedings (Cat. No. 01CH37221), volume 2, pages 749–752. IEEE, 2001.
- [70] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 776–780. IEEE, 2017.

- [71] Zafar Rafii, Antoine Liutkus, Fabian-Robert Stöter, Stylianos Ioannis Mimilakis, and Rachel Bittner. The musdb18 corpus for music separation. 2017.
- [72] Xinsheng Wang, Mingqi Jiang, Ziyang Ma, Ziyu Zhang, Songxiang Liu, Linqin Li, Zheng Liang, Qixi Zheng, Rui Wang, Xiaoqin Feng, et al. Spark-tts: An efficient llm-based text-to-speech model with single-stream decoupled speech tokens. arXiv preprint arXiv:2503.01710, 2025.

