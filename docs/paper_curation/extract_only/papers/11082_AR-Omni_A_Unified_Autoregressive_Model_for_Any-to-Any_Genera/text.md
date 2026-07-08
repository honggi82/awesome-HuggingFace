### arXiv:2601.17761v1[cs.LG]25Jan2026

2026-1-22

## AR-Omni: A Unified Autoregressive Model for Any-to-Any Generation

###### Dongjie Cheng1* Ruifeng Yuan1* Yongqi Li1† Runyang You1 Wenjie Wang2 Liqiang Nie3 Lei Zhang1 Wenjie Li1

1 The Hong Kong Polytechnic University 2 University of Science and Technology of China 3 Harbin Institute of Technology (Shenzhen)

{dong-jie.cheng,ruifeng.yuan}@connect.polyu.hk, liyongqi0@gmail.com

###### Abstract

Real-world perception and interaction are inherently multimodal, encompassing not only language but also vision and speech, which motivates the development of “Omni” MLLMs that support both multimodal inputs and multimodal outputs. While a sequence of omni MLLMs has emerged, most existing systems still rely on additional expert components to achieve multimodal generation, limiting the simplicity of unified training and inference. Autoregressive (AR) modeling, with a single token stream, a single next-token objective, and a single decoder, is an elegant and scalable foundation in the text domain. Motivated by this, we present AR-Omni, a unified any-to-any model in the autoregressive paradigm without any expert decoders. AR-Omni supports autoregressive text and image generation, as well as streaming speech generation, all under a single Transformer decoder. We further address three practical issues in unified AR modeling: modality imbalance via task-aware loss reweighting, visual fidelity via a lightweight token-level perceptual alignment loss for image tokens, and stability–creativity trade-offs via a finite-state decoding mechanism. Empirically, AR-Omni achieves strong quality across three modalities while remaining real-time, achieving a 0.88 real-time factor for speech generation.

Project Page: https://modalitydance.github.io/AR-Omni

#### 1 Introduction

Large Language Models (LLMs) have achieved strong performance in understanding and generating natural language [1]. However, their interface is largely limited to text. In contrast, real-world perception and interaction are inherently multimodal, encompassing not only language but also vision and speech. This interaction gap has motivated Multimodal Large Language Models (MLLMs), which extend LLMs to process multimodal capabilities. Early MLLMs extend LLMs with strong multimodal perception, enabling them to interpret multimodal inputs. A natural next step is to further equip MLLMs with multimodal generation capabilities, allowing them to respond to users in multiple modalities. This ability that supports both multimodal inputs and multimodal outputs is often

∗Equal contribution. †Corresponding author.

called “Omni” [2].

A sequence of omni MLLMs has progressively emerged. SpeechGPT [3] enables speech-text outputs by discretizing speech and training for cross-modal conversational instruction following. Chameleon [4] extends the same ambition to vision, proposing an early-fusion MLLM over a joint discrete vocabulary that supports both image and text generation. Building toward tri-modal any-to-any interaction, AnyGPT [5] and MIO [6] perform autoregressive modeling paired with expert diffusion decoders, thereby supporting flexible tri-modal understanding and generation.

Autoregressive (AR) modeling has proven to be an exceptionally elegant and scalable foundation in the text domain: with a single token stream, a single nexttoken objective, and a single decoder, it can already support powerful text generation. This simplicity is appealing not only conceptually but also practi-

Method Modality I/O Diffusion-free Streaming Real-time

Kosmos [7] T,I|T – – – Flamingo [8] T,I|T – – – Chameleon [4] T,I|T,I ✓ – – USLM [9] T,S|S – × ✓ AnyGPT [5] T,S,I|T,S,I × × × MiO [6] T,S,I|T,S,I × × × AR-Omni (7B) T,S,I|T,S,I ✓ ✓ ✓

- Table 1: High-level comparison of multimodal coverage and real-time streaming capability. Diffusion-free marks methods that do not rely on an external diffusion decoder for image synthesis. Streaming indicates early emission of playable audio chunks. Real-time indicates faster-than-real-time synthesis (RTF< 1) under our setup. “–” denotes not applicable. AR-Omni is the only model that achieves both Unified I/O and Real-time Streaming without external diffusion models.

cally, which offers the potential for unified training and inference for omni MLLMs. As summarized in Table 1, most existing omni MLLMs still rely on additional expert components (e.g., diffusion-style image decoders or non-AR speech generators) to achieve multimodal generation. Motivated by this, we aim to explore whether an omni model can be built with the same AR purity: a single autoregressive model that natively handles text, images, and speech.

We present AR-Omni, a unified any-to-any model in the autoregressive paradigm without any expert decoders. AR-Omni tokenizes text, images, and speech into discrete symbols and integrates them into a single joint vocabulary. After training, AR-Omni supports autoregressive text and image generation, as well as streaming speech generation, all under a single 7B-parameter Transformer backbone.

We further address three practical issues in unified AR modeling: 1) Modality imbalance. Unified AR training can be dominated by certain modalities/tasks, leading to skewed learning. We mitigate this via task-aware reweighting on response tails. 2) Visual fidelity. Autoregressively predicting image tokens may sacrifice perceptual quality in the reconstructed visuals. We introduce a lightweight tokenlevel perceptual alignment loss for image tokens. 3) Stability-creativity trade-offs. Different tasks prefer different decoding behaviors, yet a single decoding strategy can be suboptimal across tasks. We employ a finite-state decoding machine that automatically selects greedy decoding for ASR/TTS and sampling decoding for open-ended generation.

Empirically, AR-Omni achieves strong quality across modalities while remaining real-time: AROmni attains 146ms first-token latency and 0.88 real-time factor for speech generation, together with competitive objective accuracy (6.5 zero-shot TTS WER on VCTK and 9.4 ASR WER on LibriSpeech test-clean). Section 5 reports detailed quantitative results and qualitative examples demonstrating these any-to-any capabilities.

The key contributions are summarized:

- • We introduce AR-Omni, a unified autoregressive model that supports understanding and generation for text, image, and speech without any external task-specific decoders.
- • We equip AR-Omni with an efficient speech tokenizer, allowing it to decode audio after only a small number of speech tokens and thereby enabling streaming speech.
- • We address modality imbalance via task-aware reweighting on the response tail and improve visual fidelity with a token-level perceptual alignment loss for image tokens. We also address the trade-off between stability and creativity through a finite-state decoding machine.

#### 2 Related Work

###### 2.1 Multimodal Large Language Models

Early Multimodal Large Language Models (MLLMs) [10, 11] typically attach modality adapters to Large Language Models (LLMs), enabling multimodal perception and supporting

[Figure 1]

Text Token Audio Token Image Token

T to T S to S S+I to S S to I

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

AI is powered by advanced.. Hi! I’m AR-Omni… The sunset casts warm tones over the… Here you GO!

[Figure 7]

Discrete Multimodal Detokenizer

… … … …

…

[Figure 8]

# Unified Decoder

[Figure 9]

Discrete Multimodal Tokenizer

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

What makes AI so strong? Hi! How are you? Describe this image in detail. Can you show me a cat?

- Figure 1: Overview of AR-Omni. Text, speech, and image inputs are tokenized and embedded into a shared space. A single autoregressive decoder operates over a joint vocabulary to generate a unified token stream. T denotes text, S denotes speech, and I denotes image.

text response. Such approaches have witnessed impressive progress recently, as many more powerful MLLMs [12, 13] have emerged. Not satisfied with text-only generation, SpeechGPT [3] discretized speech and enables direct speech understanding and generation. NExT-GPT [14] connected an LLM with modality adaptors and diffusion-based decoders to support multimodal inputs and outputs. Building toward tri-modal any-to-any interaction, AnyGPT [5] unified text-speech-image understanding and generation. MIO [6] scaled the formulation to video generation under the same paradigm. However, they still rely on an external diffusion decoder for multimodal generation, which means the multimodal modeling responsibility is still mainly taken by external expert models.

in DALL·E [18]. They compressed images into a 2D grid of discrete codebook indices, which can be flattened for AR training. In practice, the SEED tokenizer [19, 20, 21] produced 1D visual codes and maps these codes to an image embedding that conditions a diffusion UNet to generate pixels.

Speech tokenizers. Neural audio codecs such as SoundStream [22] and EnCodec [23] quantized waveforms into discrete codes that can be decoded into high-fidelity audio. Another line [24, 9] factorized speech into semantic tokens for linguistic content and acoustic tokens for timbre. WavTokenizer [25] further explored single codebook tokenization for speech to better support streaming speech generation.

#### 3 AR-Omni

###### 2.2 Discrete Multimodal Tokenization

###### 3.1 Unified Autoregressive Modeling

The foundation of autoregressive multimodal modeling is to convert continuous modalities into discrete token sequences with vector-quantized (VQ) [15] tokenizers.

AR-Omni performs any-to-any generation by mapping all modalities into a shared discrete embedding space, allowing a single transformer-based backbone to generate multimodal data through next-token prediction. This unification is achieved by the joint vocabulary V = Vtext ∪ Vspeech ∪ Vimage that spans all

Image tokenizers. Representative designs include VQ-VAE-2 [16], VQGAN [17], and the dVAE used

modalities, where each sub-vocabulary contains the discrete tokens for its respective modality.

Text. We utilize the SentencePiece BPE tokenizer from Chameleon [4] to tokenize text.

Speech. Unlike prior methods that rely on dualcodebook (semantic and acoustic) tokenizers for easier modeling, we adopt a purely acoustic-based discrete tokenizer [25], eliminating the need for traditional semantic-to-acoustic modeling, thus bypassing the traditional "complete-then-decode" bottleneck and enabling streamed speech responses with low latency.

Image. Images are mapped into a sequence of discrete visual codes using a scene-aware VQ tokenizer [4]. Capturing essential geometric and semantic structures of the visual input in a causal 1D order, this aligns the visual generation process with the language and speech pathways without external diffusion decoders.

Interleaved modeling. To integrate these heterogeneous tokens into a single stream, we introduce a set of special tokens to manage modality transitions. Specifically, all tokens are concatenated into a single interleaved sequence x, where speech tokens are bracketed by <boa> and <eoa>, and image tokens by <boi> and <eoi>. Additionally, we use <eoh> to indicate the end of the input. For multiturn dialogues, we use <eom> to mark the end of the model’s response for the current turn, and <eos> to mark the end of the entire conversation. Text acts as the bridge that maintains semantic continuity across these markers. Prompt templates, marker usage, and examples are elaborated in Appendix C. This unified formatting reformulates any-to-any generation to a causal probabilistic modeling task:

∏︁T

pθ(xt | x<t), (1)

pθ(x) =

t=1

At inference time, our decoding strategy is taskaware: we use greedy decoding for deterministic subtasks such as Automatic Speech Recognition (ASR) and Text-to-Speech (TTS), and sampling for openended generation such as Text-to-Image generation (T2I). The effectiveness is further discussed in Section 6.2.

###### 3.2 Optimization

We employ a composite optimization strategy to address the challenges of multimodal generation. Specifically, we combine the Weighted Next-Token Prediction (Weighted NTP) objective to balance modality imbalance with a Perceptual Loss (PL) that encourages geometric consistency in the latent visual space.

Weighted NTP. To mitigate modality imbalance caused by heterogeneous token budgets in tokenized sequences (e.g., speech vs. text), we compute the loss as:

∑︁T

1 T

wt log pθ(xt | x<t), (2)

LwNTP = −

t=1

where wt is a scalar weight. In practice, for X2T tasks, including ASR and image captioning, we assign larger weights to the response text tokens. This amplifies supervision on the specific text outputs and reduces the tendency of modalities with longer sequences to dominate optimization. Experimental results in Section 6.2 demonstrate the effectiveness of the reweighting strategy.

Perceptual loss. While standard cross-entropy is effective for exact token matching, it lacks geometric awareness in the discrete latent space: it penalizes all non-target codes equally, ignoring the semantic similarity between visual tokens. To address this, we introduce a perceptual loss that aligns the lastlayer hidden states to a frozen, pretrained embedding space of target codes. This guides the model to produce visually coherent structures even when exact token matching fails.

Let E ∈ R|Vimage|×de be the fixed image embedding matrix, ht ∈ Rdh as the last-layer hidden state, and Wh ∈ Rde×dh a trainable projection. For timesteps T whose targets yt ∈ Vimage, we minimize

∑︁

1

⃦Wh ht − E[yt] ⃦2

2, (3)

Lperc =

|T |

t∈T

so the hidden states are encouraged to match the target code embedding, providing a smoother notion of similarity among visual codes than one-hot supervision.

Method Diffusion-free CIDEr ↑ Modality I/O

Kosmos [7] N/S 84.7 T,I | T Flamingo (9B) [8] N/S 79.4 T,I | T Flamingo (80B) N/S 84.3 T,I | T AnyGPT (8B) [5] × 107.5 T,S,I | T,S,I MIO (7B) [6] × 120.4 T,S,I | T,S,I

Chameleon (7B) [4] ✓ 13.72 T,I | T,I Anole (7B) [26] ✓ 15.07 T,I | T,I AR-Omni (7B) ✓ 56.53 T,S,I | T,S,I

- Table 2: Image captioning on the MS-COCO Karpathy test split. We report CIDEr. “Diffusion-free” indicates whether a model relies on an external diffusion decoder for image generation; N/S means the model does not support image generation. Modality I/O reports supported input (left) and output (right) modalities, separated by “|”; {T,S,I} denote {text, speech, image}.

Total objective. The total loss is defined as:

L = LwNTP + λperc Lperc, (4)

with a small λperc to balance NTP and perceptual gradients.

Training stabilization. Long interleaved multimodal sequences can be sensitive to optimization. We use residual-post-norm (swin-norm) [27], which applies normalization on the residual branch. Let x ∈ RT×d be the input hidden states of a Transformer block. Let Attn denote multi-head self-attention, FFN the feed-forward network, and Norm a normalization operator. The block updates are

h = x + Norm (Attn(x)). (5)

x′ = h + Norm (FFN(h)), (6)

where h is the post-attention state and x′ is the block output.

###### 3.3 Data

Pre-training data. Our pre-training data source is summarized in Table 3. To maintain a balanced mixture, we sample data with a ratio of 0.5 : 1 : 2 for text-only, text-image, and text-speech, respectively. We stop sampling once any modality subset is exhausted, ensuring a controlled modality composition without over-sampling scarce sources. Concretely,

Modality Dataset Text-only Ultra-FineWeb [28] Image–Text LAION-2B [29], LAION-Aesthetics [29], JourneyDB [30] Speech–Text GigaSpeech [31], Common Voice [32], MLS [33]

Table 3: Summary of pre-training corpora.

we use Ultra-FineWeb [28] for large-scale text pretraining; LAION and its Aesthetics subset [29] provide broad web-scale image–text pairs for general cross-modal alignment, while JourneyDB [30] adds higher-quality image–prompt data to improve textto-image generation. For speech–text supervision, we use GigaSpeech [31], Common Voice [32], and MLS [33]. Detailed data description can be found in Appendix B.

Omni-interleaved instruction data. We build our multimodal interleaved instruction data on AnyInstruct [5] by reformatting it under our unified tokenization and task definitions, preserving its interleaved text–image–speech structure. We further apply speech augmentation by mixing in indoor environmental noises from the DEMAND noise dataset [40]. We additionally incorporate VoiceAssistant-400K [41] to enrich speech-centric dialogues. To match the acoustic characteristics of pre-training, we extract the timbre from AROmni-Pretrain and synthesize assistant replies using CosyVoice2 [42], reducing the train–test distribution

Method Decoder Params Diffusion-free CLIPscore ↑ Modality I/O GILL [34] 900M × 0.67 T,I | T,I Emu [35] 900M × 0.66 T,I | T,I AnyGPT [5] 900M × 0.65 T,S,I | T,S,I MiO [6] 900M × 0.64 T,S,I | T,S,I

Chameleon (7B) [4] 41M ✓ N/S T,I | T,I Anole (7B) [26] 41M ✓ 0.27 T,I | T,I AR-Omni (7B) 41M ✓ 0.24 T,S,I | T,S,I

- Table 4: Text-to-image generation on MS-COCO. We report CLIPscore between generated images and reference captions. Decoder Params count only the modality-specific image generation module; diffusion-based methods use a latent diffusion UNet (∼900M params), while ours uses a VQGAN detokenizer (∼41M). “Diffusion-free” uses ✓/× to denote without/with external diffusion, and N/S indicates not supported. Modality I/O reports supported input (left) and output (right) modalities, separated by “|”; {T,S,I} denote {text, speech, image}.

Method WER ↓ Speech-in tok/s ↓ Codebook Modality I/O

Human-level [36] 5.8 N/S N/S S | T Wav2vec 2.0 [37] 2.7 N/S N/S S | T Whisper Large V2 [38] 2.7 N/S N/S S | T AnyGPT [5] 8.5 50 Dual T,S,I | T,S,I MiO [6] 6.3 200 Dual T,S,I | T,S,I

Anole (7B) [26] N/S N/S N/S T,I | T,I AR-Omni (7B) 9.4 40 Single T,S,I | T,S,I

- Table 5: ASR performance on LibriSpeech test-clean. We report WER. Speech-in tok/s is the number of discrete tokens per second produced by the speech tokenizer for the input audio; N/S indicates not supported. Codebook uses Dual for separate semantic and acoustic codebooks and Single for a unified codebook. Modality I/O reports supported input (left) and output (right) modalities, separated by “|”; {T,S,I} denote {text, speech, image}.

gap in speech. For text-only conversation, we include UltraChat [43] as a text-only dialogue source.

#### 4 Experimental Setup

###### 4.1 Training

The training process of AR-Omni consists of two stages: pre-training and fine-tuning. We initialize from Anole [26], a 7B autoregressive Transformer designed for interleaved image–text modeling. Stage 1 focuses on pre-training by optimizing the weighted NTP objective with the perceptual loss. Stage 2 performs fine-tuning on omni-interleaved instruction data, where the loss is applied exclusively to the re-

sponse tokens. Further technical details are provided in Appendix A.

###### 4.2 Image Evaluation

Image understanding. We evaluated image-to-text understanding on the MS-COCO 2014 captioning benchmark [44] using the Karpathy test split, following prior work [45]. We reported CIDEr in a zero-shot setting. CIDEr is computed by measuring the cosine similarity between TF-IDF–weighted n-gram vectors of the generated caption and the set of reference captions.

Image generation. We evaluated text-to-image generation on MS-COCO by randomly sampling 30k cap-

Method WER ↓ Speech-out tok/s ↓ FTL (ms) ↓ RTF ↓ Real-time (RTF < 1) Codebook Modality I/O

Ground Truth 1.9 N/S N/S N/S N/S N/S T,S | S VALL-E [39] 7.9 600 N/S N/S N/S Dual T,S | S USLM [9] 6.5 50 (+350) 11611 0.89 ✓ Dual T,S | S AnyGPT [5] 8.5 50 (+350) 6602 2.92 × Dual T,S,I | T,S,I MiO [6] 12.0 200 29079 48.90 × Dual T,S,I | T,S,I

Anole (7B) [26] N/S N/S N/S N/S N/S N/S T,I | T,I AR-Omni (7B) 6.5 40 146 0.88 ✓ Single T,S,I | T,S,I

- Table 6: Zero-shot TTS results on VCTK. We report WER, first-token latency (FTL), and real-time factor (RTF); N/S indicates not supported. Speech-out tok/s measures the length of the generated speech token stream per second; for dual-codebook methods, a(+b) denotes semantic tokens at a tok/s plus acoustic tokens at b tok/s. Codebook uses Dual for separate semantic and acoustic codebooks and Single for a unified codebook. Modality I/O reports supported input (left) and output (right) modalities, separated by “|”; {T,S,I} denote {text, speech, image}.

Weighted NTP

Perceptual

Total

5.00

0.12

4.75

5.0

0.10

4.50

4.5

0.08

4.25

Loss

0.06

4.00

4.0

3.75

0.04

3.5

3.50

0.02

3.0

3.25

0 20000 40000 60000 80000

0 20000 40000 60000 80000

0 20000 40000 60000 80000

Step

Step

Step

- Figure 2: Stage 1 pretraining losses of AR-Omni. From left to right: weighted NTP loss, perceptual loss, and total loss. Curves are smoothed for readability and plotted over 1k–93k training steps.

tions from the validation set, following prior protocols [20]. We generated images with AR-Omni and reported CLIPscore. CLIPscore is computed as the cosine similarity between the CLIP image embedding of a generated image and the CLIP text embedding of its caption using CLIP ViT-L, averaged over all samples.

###### 4.3 Speech Evaluation

ASR. We evaluated Automatic Speech Recognition (ASR) on the LibriSpeech test-clean split [46]. We reported Word Error Rate (WER). WER is computed as (substitutions + deletions + insertions) divided by the number of words in the reference transcript, based on Levenshtein alignment.

TTS. We evaluated zero-shot text-to-speech (TTS) on the VCTK [47] dataset by conditioning AR-Omni on text prompts. For TTS, WER is computed between the ASR transcript of the synthesized audio and the reference text. We used Whisper-Large-V2 [38] as the transcriber model. First Token Latency (FTL) is a theoretical lower bound defined as the latency to the first generated speech token that can be immediately decoded into audio. In the dual-codebook scheme, decoding is conditioned on aligned token pairs, so it typically starts once the corresponding tokens from both codebooks are available. Real Time Factor (RTF) is computed as the total wall-clock synthesis time divided by the duration of the generated audio.

#### 5 Main Results

- 5.1 Image Results

Image understanding. Table 2 presents zero-shot image captioning results. Within the diffusion-free autoregressive setting, AR-Omni outperforms the Anole [26] initialization on image-to-text generation. This suggests that extending a diffusion-free AR backbone to an any-to-any training regime does not compromise captioning quality, and can instead strengthen it.

Image generation. Table 4 presents text-to-image generation results. Compared with the Anole initialization, AR-Omni incurs only a slight drop on this benchmark, suggesting that any-to-any training largely preserves the ability to autoregressively generate image tokens. Meanwhile, diffusion-based systems score higher, consistent with the advantage of employing a diffusion-style image decoder for image synthesis. These results highlight a clear trade-off: AR-Omni maintains a diffusion-free, single-model generation pipeline, whereas diffusion decoders improve text-to-image quality at the cost of additional expert components.

5.2 Speech Results

ASR. Table 5 presents the ASR results along with the speech tokenization rate for the input audio. Compared to any-to-any baselines, AR-Omni achieves comparable recognition accuracy while using significantly fewer speech tokens. This demonstrates that a single-codebook, low-rate speech tokenization can serve as an effective interface for integrating speech into a unified AR model, without requiring separate modality-specific generators.

TTS. Table 6 reports zero-shot TTS results, including streaming-related metrics. Under the same evaluation setup, AR-Omni matches the best-performing token-based baseline in intelligibility, while exhibiting more favorable streaming characteristics in latency and throughput. In contrast, prior any-to-any systems often incur higher latency and slower generation, reflecting the overhead of heavier or more complex generation pipelines.

#### 6 Further Analysis

Method I2T ↑ T2I ↑ ASR ↓ TTS ↓

AR-Omni 0.5022 0.2406 0.1535 0.1105 w/o PL 0.4874 0.2356 0.1395 0.1175 w/o swin-norm 0.5488 0.2392 0.1329 0.1633 Simple NTP 0.5074 0.2248 0.2270 0.1589

Table 7: Ablation results on four omni-pretraining tasks at 40k steps. T denotes text, S denotes speech, and I denotes image. I2T reports CIDEr on image captioning. T2I reports CLIPscore on text-to-image generation. ASR and TTS report WER. ↑ indicates higher is better and ↓ indicates lower is better.

- 3.5

- 4.0

- 4.5

- 5.0

Loss

Ours

0 10000 20000 30000 40000 50000

Step

3.2

3.3

3.4

3.5

3.6

3.7

3.8

3.9

4.0

Simple NTP

Figure 3: Loss curves of AR-Omni (Ours) and the simple NTP training objective. The naive objective exhibits a sharp loss jump, whereas AR-Omni maintains a smooth and stable loss throughout training.

- 6.1 Analysis on Training Objectives

We analyze the stage-1 loss decomposition to understand how the composite pretraining objective is optimized. The results are shown in Figure 2. The weighted NTP term decreases smoothly throughout training, indicating a stable token-level learning signal under the unified setup. In contrast, the perceptual term converges quickly and saturates at a much smaller magnitude, so the total loss is largely governed by the weighted NTP trajectory; the residual fluctuations are therefore more consistent with mini-batch variance than optimization instability.

- 6.2 Ablation Study

0 10000 20000 30000 40000 50000

Step

To validate the training design of AR-Omni we conduct ablation analysis focusing on 1) the contribution of individual components to performance, and 2) their impact on long-term training stability.

Component effectiveness. We first assess the impact of key components by ablating them at 40k steps, with results summarized in Table 7. Removing the

[Figure 15]

[Figure 16]

[Figure 17]

AR-Omni: A Unified Autoregressive Model for Any-to-Any Generation

[Figure 18]

[Figure 19]

##### Speech + Image -> Speech

##### Multi-Turn Speech Conversation

a b

[Figure 20]

[Figure 21]

User: <speech in> Describe the image in detail?

- Turn0 User: <speech in> Hi, how are you ? AR-Omni : <speech out> I am doing well! How about your day today? Are there any ……

- Turn1 User: <speech in> Who is the best basketball player? AR-Omni : <speech out> The greatest basketball players of all time include Michael Jordan……

- Turn2 User: <speech in> Which club does he play for now? AR-Omni : <speech out> He currently … after retiring last year to become a coach ……

- Turn3 User: <speech in> Thank you, Byebye! AR-Omni : <speech out> Bye bye! You are very welcome to visit us anytime and stay ……

[Figure 22]

AR-Omni : <speech out> The sunset casts warm tones over the horizon and sea waves gently lapping against shorelines at golden hour.

[Figure 23]

Speech -> Image

c

[Figure 24]

User: <speech in> Can you show me a cat? AR-Omni : <image out>

[Figure 25]

Figure 4: Case studies of AR-Omni: (a) multi-turn speech conversation (S→S), (b) speech+image understanding with speech response (S+I→S), and (c) speech-to-image generation (S→I).

###### 6.3 Case Study

perceptual loss slightly reduces I2T and T2I, degrades TTS, and improves ASR. This indicates that PL mainly benefits vision generation and speech synthesis in our unified training framework. Dropping swin-norm improves I2T and ASR but substantially worsens TTS and slightly reduces T2I, suggesting swin-norm is crucial for maintaining speech quality. When removing Weighted NTP, PL, and swin-norm (hereby referred to as simple NTP), ASR and TTS errors increase sharply and T2I performance also drops, further demonstrating the effectiveness of the weighted NTP objective for mitigating modality imbalance.

After omni supervised instruction tuning, AR-Omni exhibits omni multimodal capabilities across text, images, and speech, including understanding and generation. Figure 4 presents representative case studies. In (a), AR-Omni maintains a multi-turn speech context and answers follow-up questions speech coherently. In (b), given a spoken instruction and an image as context, AR-Omni produces a grounded image description in speech; in (c), AR-Omni generates an image from a spoken prompt. Together, these cases illustrate unified perception, understanding, and generation across heterogeneous modality contexts. Longer-horizon and more fully interleaved demonstrations are provided in Appendix D.

7

Training stability. We further investigate the training dynamics of the Simple NTP variant over a longer horizon to assess stability. As illustrated in Figure 3, Simple NTP exhibits training instability, characterized by a loss rebound and abrupt spikes in the late stages, indicating a tendency toward model collapse. In contrast, our AR-Omni maintains a smooth and convergent loss trajectory throughout training. These results confirm that our proposed strategies not only enhance multimodal capabilities but also effectively prevent late-stage collapse, ensuring robust training stability.

#### 7 Conclusion and Future Work

We presented AR-Omni, a unified any-to-any model in the autoregressive paradigm without any expert decoders, which tokenizes text, images, and speech into a discrete token stream. With one Transformer backbone, AR-Omni supports autoregressive text and image generation as well as streaming speech generation. To make unified AR modeling practical, we mitigate modality imbalance with task-aware loss

reweighting, improve visual fidelity with perceptual loss, and adapt decoding behavior via a finite-state decoding machine. Experiments show competitive tri-modal capabilities while remaining real-time for streaming speech. A key limitation is that diffusionfree autoregressive image generation still lags behind diffusion-based systems in terms of image generation. Future work will focus on enhancing the quality of diffusion-free image generation while respecting the purity of the unified AR.

###### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [3] Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, and Xipeng Qiu. Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities. In Findings of EMNLP, 2023. URL https://aclanthology.org/2023. findings-emnlp.1055/.
- [4] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [5] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, Hang Yan, Jie Fu, Tao Gui, Tianxiang Sun, Yugang Jiang, and Xipeng Qiu. AnyGPT: Unified multimodal LLM with discrete sequence modeling. In Proceedings of ACL, 2024. URL https://arxiv.org/abs/2402.12226.
- [6] Zekun Wang, King Zhu, Chunpu Xu, Wangchunshu Zhou, Jiaheng Liu, Yibo Zhang, Jiashuo Wang, Ning Shi, Siyu Li, Yizhi Li, Haoran Que, Zhaoxiang Zhang, Yuanxing Zhang, Ge Zhang, Ke Xu, Jie Fu, and Wenhao Huang. MIO: A foundation model on multimodal tokens. arXiv preprint arXiv:2409.17692, 2024.
- [7] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, et al. Language is not all you need: Aligning perception with language models. Advances in Neural Informa-

- tion Processing Systems, 36:72096–72109, 2023.
- [8] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. ArXiv preprint, abs/2204.14198, 2022. URL https:// arxiv.org/abs/2204.14198.
- [9] Xin Zhang, Dong Zhang, Shimin Li, Yaqian Zhou, and Xipeng Qiu. Speechtokenizer: Unified speech tokenizer for speech large language models. arXiv preprint arXiv:2308.16692, 2023.
- [10] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.
- [11] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.
- [12] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [13] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [14] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023.
- [15] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NeurIPS, 2017.
- [16] Ali Razavi, Aaron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with vq-vae-

2. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

- [17] Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.
- [18] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation.

- arXiv preprint arXiv:2102.12092, 2021.
- [19] Yuying Ge, Yixiao Ge, Ziyun Zeng, Xintao Wang, and Ying Shan. Planting a seed of vision in large language model. arXiv preprint arXiv:2307.08041, 2023.
- [20] Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making llama see and draw with seed tokenizer. arXiv preprint arXiv:2310.01218, 2023.
- [21] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, 2022.
- [22] Neil Zeghidour, Andreas Luebs, Ahmed Omran, et al. Soundstream: An end-to-end neural audio codec. arXiv preprint arXiv:2107.03312, 2021.
- [23] Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. High fidelity neural audio compression. arXiv preprint arXiv:2210.13438, 2022.
- [24] Zalan Borsos et al. Audiolm: A language modeling approach to audio generation. arXiv preprint arXiv:2209.03143, 2022.
- [25] Shengpeng Ji, Ziyue Jiang, Wen Wang, Yifu Chen, Minghui Fang, Jialong Zuo, Qian Yang, Xize Cheng, Zehan Wang, Ruiqi Li, Ziang Zhang, Xiaoda Yang, Rongjie Huang, Yidi Jiang, Qian Chen, Siqi Zheng, Wen Wang, and Zhou Zhao. Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling. arXiv preprint arXiv:2408.16532, 2024.
- [26] Ethan Chern, Jiadi Su, Yan Ma, and Pengfei Liu. Anole: An open, autoregressive, native large multimodal models for interleaved image-text generation. arXiv preprint arXiv:2407.06135, 2024.
- [27] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, et al. Swin transformer v2: Scaling up capacity and resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12009–12019, 2022.
- [28] Yudong Wang, Zixuan Fu, Jie Cai, Peijun Tang, Hongya Lyu, Yewei Fang, Zhi Zheng, Jie Zhou, Guoyang Zeng, Chaojun Xiao, et al. Ultrafineweb: Efficient data filtering and verification for high-quality llm training data. arXiv preprint arXiv:2505.05427, 2025.
- [29] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open largescale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.

- [30] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. Advances in neural information processing systems, 36: 49659–49678, 2023.
- [31] Guoguo Chen, Shuzhou Chai, Guanbo Wang, Jiayu Du, Wei-Qiang Zhang, Chao Weng, Dan Su, Daniel Povey, Jan Trmal, Junbo Zhang, et al. Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio. arXiv preprint arXiv:2106.06909, 2021.
- [32] Rosana Ardila, Megan Branson, Kelly Davis, Michael Kohler, Josh Meyer, Michael Henretty, Reuben Morais, Lindsay Saunders, Francis Tyers, and Gregor Weber. Common voice: A massively-multilingual speech corpus. In Proceedings of the twelfth language resources and evaluation conference, pages 4218– 4222, 2020.
- [33] Vineel Pratap, Qiantong Xu, Anuroop Sriram, Gabriel Synnaeve, and Ronan Collobert. Mls: A large-scale multilingual dataset for speech research. arXiv preprint arXiv:2012.03411, 2020.
- [34] Jing Yu Koh, Daniel Fried, and Ruslan Salakhutdinov. Generating images with multimodal language models. ArXiv preprint, abs/2305.17216, 2023. URL https://arxiv.org/abs/2305.17216.
- [35] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023.
- [36] Dario Amodei, Sundaram Ananthanarayanan, Rishita Anubhai, Jingliang Bai, Eric Battenberg, Carl Case, Jared Casper, Bryan Catanzaro, Qiang Cheng, Guoliang Chen, et al. Deep speech 2: End-to-end speech recognition in english and mandarin. In International conference on machine learning, pages 173–

182. PMLR, 2016. URL https://arxiv.org/ abs/1512.02595.

- [37] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings. neurips.cc/paper/2020/hash/ 92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract. html.

- [38] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. ArXiv preprint, abs/2212.04356, 2022. URL https://arxiv.org/abs/2212.04356.
- [39] Chengyi Wang, Sanyuan Chen, Yu Wu, Zi-Hua Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, Lei He, Sheng Zhao, and Furu Wei. Neural codec language models are zero-shot text to speech synthesizers. ArXiv preprint, abs/2301.02111, 2023. URL https:// arxiv.org/abs/2301.02111.
- [40] Joachim Thiemann, Nobutaka Ito, and Emmanuel Vincent. Demand: a collection of multi-channel recordings of acoustic noise in diverse environments, June 2013. URL https://doi.org/10.5281/ zenodo.1227121.
- [41] Zhifei Xie and Changqiao Wu. Mini-omni: Language models can hear, talk while thinking in streaming. arXiv preprint arXiv:2408.16725, 2024.
- [42] Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, et al. Cosyvoice 2: Scalable streaming speech synthesis with large language models. arXiv preprint arXiv:2412.10117, 2024.
- [43] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233, 2023.
- [44] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision, 2014. URL https://api. semanticscholar.org/CorpusID:14113767.
- [45] Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable diffusion. ArXiv preprint, abs/2305.11846, 2023. URL https://arxiv. org/abs/2305.11846.
- [46] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: An ASR corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2015, South Brisbane, Queensland, Australia, April 19-24, 2015, pages 5206–5210. IEEE, 2015. doi: 10.1109/ ICASSP.2015.7178964. URL https://doi.org/ 10.1109/ICASSP.2015.7178964.
- [47] Christophe Veaux, Junichi Yamagishi, and Kirsten MacDonald. Cstr vctk corpus: English multi-speaker corpus for cstr voice cloning toolkit. 2017.

#### A Training Details

We train AR-Omni on 8 NVIDIA A100 GPUs. The pipeline consists of a pre-training stage followed by fine-tuning, both optimized with Adam and a linear learning-rate schedule with warmup. We apply global gradient clipping to stabilize unified multimodal training. Our implementation is developed on top of the Chameleon repository [4], including training and inference code. Table 8 lists the key hyperparameters for both stages.

Hyperparameter Pre-training Fine-tuning

GPU Type A100 A100 Optimizer Adam Adam LR Scheduler Linear Linear Gradient Clipping 1.0 1.0 Max Sequence Length 1300 3456 Warmup Ratio 0.05 0.05 Batch Size 480 64 Training Steps 140,000 18,000 Peak Learning Rate 6 × 10−5 2 × 10−5

- Table 8: Training hyperparameters for pre-training and fine-tuning. Batch size denotes the global batch size.

B Dataset Details

- Table 9 summarizes the pre-training corpora used for AR-Omni, covering image–text, speech–text, and text-only data. For each dataset, we report modality pairing, scale statistics, and brief notes on source and preprocessing to facilitate reproducibility. We use publicly available datasets and follow standard filtering and deduplication practices to reduce noise and unsafe content.

#### C Prompt Templates

We adopt a unified dialogue-style prompt format for any-to-any multimodal tasks (Table 10). Each user turn begins with <bos> and ends with <eoh>. Nontext modalities are explicitly bracketed by boundary tokens, including <boa>/<eoa> for audio tokens and <boi>/<eoi> for image tokens. For single-turn (nonchat) tasks, the assistant response is terminated by <eos>. For multi-turn chat settings, the assistant response is terminated by <eom>, and the dialogue

history is formed by concatenating previous turns in the same format.

This unified formatting casts diverse tasks into a single next-token prediction interface while keeping modality boundaries explicit for both inputs and outputs.

#### D Case Study

We provide additional qualitative examples illustrating AR-Omni’s any-to-any behavior. Figures 5 and 6 show multi-turn interactions with interleaved multimodal inputs and outputs. Figures 7–10 present text-to-image samples generated by AR-Omni without external diffusion decoders; images are obtained by decoding the autoregressively generated discrete image tokens with the detokenizer.

Dataset Details Image–Text LAION-2B-en Size: 2.32 billion image–text pairs

Language: English Content and source: Web image URLs and associated alternative text collected from Common Crawl

Notes: A Contrastive Language–Image Pre-training filtered subset of LAION-5B; LAION distributes index files (Uniform Resource Locators and metadata) rather than the image files themselves.

LAION-Aesthetics V2 Size: 600 million image–text pairs Language: English Content and source: English portion of LAION-5B ranked by a predicted aesthetic score; we select examples with predicted score at least 5 Notes: Aesthetic-score subsets are nested and overlapping.

JourneyDB Size: approximately 4.4 million image–prompt pairs Language: English prompts Content and source: High-resolution images generated by Midjourney with their corresponding text prompts Notes: The dataset also provides additional annotations (for example, captions and visual question answering style annotations) used for evaluation.

Speech–Text GigaSpeech Size: 10,000 hours of transcribed audio

Language: English Content and source: A mixture of read and spontaneous speech from audiobooks, podcasts, and online videos Notes: The corpus provides multiple predefined subsets that support training at different scales.

Common Voice Size: 33,150 hours total audio, including 22,108 hours validated Language: Multiple languages Content and source: Crowdsourced speech recordings paired with transcripts; optional speaker metadata is available Notes: Reported sizes depend on the release version; we report statistics for version 20.0.

Multilingual LibriSpeech Size: 44,500 hours of transcribed audio Language: English Content and source: Audiobook recordings from LibriVox aligned to transcripts Notes: We use only the English portion of Multilingual LibriSpeech.

Text-only Ultra-FineWeb Size: A subset sampled from the English split of Ultra-FineWeb. For reference, the full

English split contains approximately one trillion tokens and about 1.16 billion rows. Language: English Content and source: High-quality web text derived from Common Crawl through an efficient filtering and verification pipeline applied to FineWeb. Notes: We use only a portion of the English split and do not use the Chinese split.

Table 9: Detailed statistics of pre-training corpora.

###### Prompt Templates:

Image Generation (Text → Image)

<bos> USER: Create an image for: {caption} <eoh> ASSISTANT: <boi> {image} <eoi> <eos>

ASR (Audio → Text)

<bos> USER: Transcribe the audio. <boa> {audio} <eoa> <eoh> ASSISTANT: {transcript} <eos>

TTS (Text → Audio)

<bos> USER: Convert the text to speech: {text} <eoh> ASSISTANT: <boa> {audio} <eoa> <eos>

Image Captioning (Image → Text)

<bos> USER: Describe the image. <boi> {image} <eoi> <eoh> ASSISTANT: {caption} <eos>

Chat-Image Generation (Audio → Text → Image)

{history} <bos> USER: <boa> {audio} <eoa> Transcribe the audio and generate an image for the transcript. <eoh> ASSISTANT: {transcript} <boi> {image} <eoi> <eom>

Chat-Speech Conversation (Audio → Text)

{history} <bos> USER: <boa> {audio} <eoa> Reply in text. <eoh> ASSISTANT: {response} <eom>

- Table 10: Dialogue-style prompt templates with explicit modality boundaries. Single-turn tasks terminate assistant outputs with <eos>, while chat settings terminate each assistant message with <eom>. {history} denotes a concatenation of previous turns in the same format.

[Figure 26]

[Figure 27]

- Turn1 User: <speech in> Can you generate a picture of a building by the sea? AR-Omni : <image out>

- Turn2 User: <speech in> What do you see around the building in the image? AR-Omni : <speech out> I see a majestic structure surrounded by vast waves crashing against its rocky base.

- Turn1 User: <speech in> How can I pick a book that is easy to read? AR-Omni : <speech out> I think you should look for books with simple language and short chapters. Also try looking at some of our top bestsellers lists on websites like Goodreads or Nook.

- Turn2 User: <speech in> What kind of book do you suggest for that goal? AR-Omni : <speech out> A modern classic novel would be ideal; it balances difficulty level while engaging readers from diverse backgrounds.

- Turn3 User: <speech in> How much reading time should i aim for? AR-Omni : <speech out> Start with 5-10 minutes each session, gradually increasing as your stamina improves.

- Turn4 User: <speech in> If I lose interest, what should I try next? AR-Omni : <speech out> Try something new within your comfort zone. Maybe switch genres or authors who write in different styles.

[Figure 28]

[Figure 29]

- Turn1 User: <speech in> How can I plan a simple nature moment during the day? AR-Omni : <speech out> Choose a secluded spot with natural beauty nearby and bring water for hydration.

- Turn2 User: <speech in> Can you generate a "picture of a small waterfall in a forest"? AR-Omni : <image out>

- Turn3 User: <speech in> What do you notice near the water in the image? AR-Omni : <speech out> The river flows through lush greenery, creating an inviting atmosphere to relax amidst nature's tranquility.

- Turn4 User: <speech in> And how about this one from my hike last year?

[Figure 30]

[Figure 31]

- Turn1 User: <speech in> How can I start a hard talk with a friend? AR-Omni : <speech out> Begin by expressing your feelings calmly and listen to their perspective without judgment or defensiveness.

- Turn2 User: <speech in> What calm opening line do you suggest? AR-Omni : <speech out> "Let go of past regrets; embrace tomorrow with open arms."

- Turn3 User: <speech in> How can I ask my main question gently? AR-Omni : <speech out> Use "Can we discuss this in an open-minded way so that everyone feels heard?"

[Figure 32]

AR-Omni : <speech out> It shows clear blue waters reflecting trees above, enhancing visual appeal even more than still images.

Figure 5: Multi-turn interleaved conversation example.

[Figure 33]

[Figure 34]

- Turn1 User: <speech in> Can you generate a picture of a building by the sea? AR-Omni : <image out>

- Turn2 User: <speech in> What do you see around the building in the image? AR-Omni : <speech out> I see a majestic structure surrounded by vast waves crashing against its rocky base.

- Turn1 User: <speech in> How can I pick a book that is easy to read? AR-Omni : <speech out> I think you should look for books with simple language and short chapters. Also try looking at some of our top bestsellers lists on websites like Goodreads or Nook.

- Turn2 User: <speech in> What kind of book do you suggest for that goal? AR-Omni : <speech out> A modern classic novel would be ideal; it balances difficulty level while engaging readers from diverse backgrounds.

- Turn3 User: <speech in> How much reading time should i aim for? AR-Omni : <speech out> Start with 5-10 minutes each session, gradually increasing as your stamina improves.

- Turn4 User: <speech in> If I lose interest, what should I try next? AR-Omni : <speech out> Try something new within your comfort zone. Maybe switch genres or authors who write in different styles.

[Figure 36]

- Turn1 User: <speech in> How can I plan a simple nature moment during the day? AR-Omni : <speech out> Choose a secluded spot with natural beauty nearby and bring water for hydration.

- Turn2 User: <speech in> Can you generate a "picture of a small waterfall in a forest"? AR-Omni : <image out>

- Turn3 User: <speech in> What do you notice near the water in the image? AR-Omni : <speech out> The river flows through lush greenery, creating an inviting atmosphere to relax amidst nature's tranquility.

- Turn4 User: <speech in> And how about this one from my hike last year?

[Figure 37]

- Turn1 User: <speech in> How can I start a hard talk with a friend? AR-Omni : <speech out> Begin by expressing your feelings calmly and listen to their perspective without judgment or defensiveness.

- Turn2 User: <speech in> What calm opening line do you suggest? AR-Omni : <speech out> "Let go of past regrets; embrace tomorrow with open arms."

- Turn3 User: <speech in> How can I ask my main question gently? AR-Omni : <speech out> Use "Can we discuss this in an open-minded way so that everyone feels heard?"

AR-Omni : <speech out> It shows clear blue waters reflecting trees above, enhancing visual appeal even more than still images.

Figure 6: Multi-turn speech conversation example.

A foggy industrial basement in the...

A winter road at sunset A clean courtyard in the afternoon with muted...

A desert road at night

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

A foggy narrow alley in the morning with muted...

A minimal library interior at night with a quiet...

A brown dog on a blanket A cliffside lighthouse in the afternoon

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

A market street at sunset A small teddy bear on a shelf

A cliffside lighthouse at sunset

A foggy metro tunnel in the evening with strong...

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

A cold school hallway at night

A bright hospital corridor at night with a quiet...

A fluffy cat A playful cat

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

A mountain cabin at sunset

A mountain stream at sunrise

A green toy robot on a shelf

A cute hamster on a blanket

A wide hotel lobby in the morning with cool white...

A colorful teddy bear A forest campsite at dusk A snow globe with a tiny house inside

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

A quiet courtyard in the afternoon with distant...

A bunch of ripe strawberries on a plate

An echoing library interior in the afternoon with...

A dusty industrial corridor in the morning with...

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

A minimal narrow alley in the afternoon with soft...

A foggy narrow alley in the morning with muted...

A bowl of fresh grapes A small parrot by a window

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

A canyon at night A cobblestone alley at dusk

A stack of colorful books on a desk

A forest campsite at sunrise

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

A black dog on a pillow A Japanese garden at sunrise

A tram street at night A temple gate at sunrise

A beach path at night A shrine path at sunrise A chocolate cupcake A glass vase on a table

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

A red toy robot on the floor

A cherry blossom park at sunrise

A foggy subterranean hall at night

A fresh orange on a plate

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

A bowl of fresh blueberries on a plate

A cherry blossom park at sunrise

A cup of tea A red toy train on the floor

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

A cobblestone alley at night

A wooden toy robot An old marketplace in the afternoon

A bowl of fresh blueberries

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

A desert plain at night A cold glass of water An alarm clock A homemade loaf of bread on a table

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

An orange rabbit by a window

A fresh cheese sandwich on a table

A cute rabbit by a window A brown hamster on a blanket

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

A golden dog on a blanket

A black hamster A brown cat on a blanket A ceramic vase

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

A fresh red apple on a plate

A snow globe A cup of tea on a table A cold glass of lemonade on a table

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

A fresh chocolate cupcake on a table

A glass of water A bowl of fresh strawberries

A brown leather wallet

Figure 10: Qualitative image generation results with AR-Omni across diverse prompts and styles.

