# arXiv:2601.10547v2[cs.SD]26Jan2026

## HeartMuLa: A Family of Open Sourced Music Foundation Models

HeartMuLa Teams

GitHub: https://github.com/HeartMuLa/heartlib

Demo: https://heartmula.github.io/

### Abstract

We present a family of open-source Music Foundation Models designed to advance large-scale music understanding and generation across diverse tasks and modalities. Our framework consists of four major components: (1) HeartCLAP, an audiotext alignment model; (2) HeartTranscriptor, a robust lyric recognition model optimized for real-world music scenarios; and (3) HeartCodec, a low-frame-rate (12.5 Hz) yet high-fidelity music codec tokenizer that captures long-range musical structure while preserving fine-grained acoustic details and enabling efficient autoregressive modeling; (4) HeartMuLa, an LLM-based song generation model capable of synthesizing high-fidelity music under rich, user-controllable conditions (e.g., textual style descriptions, lyrics, and reference audio). In addition, it provides two specialized modes: (i) fine-grained musical attribute control, which allows users to specify the style of different song sections (e.g., intro, verse, chorus) using natural language prompts; and (ii) short, engaging music generation, which is suitable as background music for short videos. Lastly, HeartMuLa improves significantly when scaled to 7B parameters. For the first time, we show that a Suno-level, commercial-grade system can be reproduced using academic-scale data and GPU resources. We expect these foundation models to serve as strong baselines for future research and to facilitate practical applications in multimodal content production.

Suno v5

Suno v4.5

Mureka V7.6

Udio v1.5

Levo

YuE

DiffRhythm 2

Ace-step

HeartMuLa

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

###### Lyric Error Rate (PER)

###### Overall Quality (SongEval)

###### Style Consistency (Tag-Sim)

0.7

0.35

4.54 4.48

4.5

0.6

0.30

0.5

4.0

0.26 0.26

0.25

Score(1-5)

0.4

Similarity

3.5

###### PER

0.3

0.20

3.0

0.2

0.13

0.15

2.5

0.09

0.1

0.0

2.0

0.10

Sunov5Sunov4.5MurekaV7.6Udiov1.5 Levo YuEDiffRhythm2Ace-stepHeartMuLa

Sunov5Sunov4.5MurekaV7.6Udiov1.5 Levo YuEDiffRhythm2Ace-stepHeartMuLa

Sunov5Sunov4.5MurekaV7.6Udiov1.5 Levo YuEDiffRhythm2Ace-stepHeartMuLa

###### Audio Quality (AudioBox PQ)

###### Musicality (SongEval)

Structure Coherence

8.4

4.62

4.51

4.52

8.21

4.5

4.36

4.5

8.2

8.14

8.0

4.0

4.0

Score(1-5)

Score(1-5)

7.8

PQScore

3.5

3.5

7.6

3.0

7.4

3.0

2.5

7.2

7.0

2.0

2.5

Sunov5Sunov4.5MurekaV7.6Udiov1.5 Levo YuEDiffRhythm2Ace-stepHeartMuLa

Sunov5Sunov4.5MurekaV7.6Udiov1.5 Levo YuEDiffRhythm2Ace-stepHeartMuLa

Sunov5Sunov4.5MurekaV7.6Udiov1.5 Levo YuEDiffRhythm2Ace-stepHeartMuLa

Figure 1: Overall comparison of HeartMuLa-oss-3B with existing music foundation models.

Research technical report

### Contents

- 1 Introduction 4
- 2 HeartCodec 4

- 2.1 Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 2.2 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.3 Experiment Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 2.3.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 2.3.2 Reconstruction Quality . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 2.3.3 Ablation Study of the Latent Space . . . . . . . . . . . . . . . . . . . . . 9
- 2.3.4 Ablation Study of Training Stages . . . . . . . . . . . . . . . . . . . . . . 10

- 3 HeartMuLa 10

- 3.1 Hierarchical Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 3.2 Conditioning Mechanism . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 3.3 Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 3.3.1 Four-Stage Progressive Training Paradigm . . . . . . . . . . . . . . . . . 12
- 3.3.2 Optimization Objectives . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 3.3.3 Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 3.4 Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

- 3.4.1 Evaluation Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 3.4.2 Objective Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 3.4.3 Subjective Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 3.4.4 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- 3.5 Inference Acceleration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- 3.5.1 Overview of the Inference Pipeline . . . . . . . . . . . . . . . . . . . . . 19
- 3.5.2 Collaborative Optimizations: KV-Cache, FlashAttention, and CUDA Graph 19
- 3.5.3 Batching and Streaming Considerations . . . . . . . . . . . . . . . . . . . 20
- 3.5.4 Experimental Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- 4 HeartCLAP 21

- 4.1 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- 4.2 Experiment Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- 5 HeartTranscriptor 23

- 5.1 Dataset Construction Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- 5.2 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- 5.3 Experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- 6 Overall Training Datasets 24

- 6.1 Composition of the Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- 6.2 Music Style . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- 6.3 Music Structure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- 6.4 Data-Finegrained Style Annotation . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- 6.5 HeartBeats-Benchmark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- 7 Related Works 28

- 7.1 Audio Tokenizer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- 7.2 Music generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- 8 Conclusion 29
- 9 Ethics and Responsibility 29
- 10 Contributions 30

### 1 Introduction

Music generation and understanding have rapidly evolved with the emergence of large-scale multimodal foundation models Copet et al. [2023], Lei et al. [2025], Wu et al. [2025]. Recent advances in audio representation learning Xu et al. [2024], Won et al. [2023], text-audio alignment Elizalde et al. [2023], and autoregressive music generation Yang et al. [2023b], Copet et al. [2023], Agostinelli et al. [2023] as well as diffusion-based music synthesis Ning et al. [2025] have enabled impressive progress in music captioning, style transfer, and conditional generation. However, existing systems still face significant limitations. Many music models rely on proprietary datasets or closed-source pipelines, which limits reproducibility and downstream research. Others provide only coarse control over musical attributes, lack robust alignment between textual descriptions and acoustic realizations, or struggle to maintain long-range musical coherence beyond short segments Copet et al. [2023]. Furthermore, end-to-end controllable song generation jointly guided by style descriptions, lyrics, and reference audio remains an open challenge.

To address these limitations, we introduce a family of open-source Music Foundation Models designed to unify music understanding, alignment, and controllable generation within a single extensible ecosystem. Our framework integrates four key components:

- (1) HeartCLAP: an audio-text alignment model that learns a shared embedding space for music semantics, enabling accurate music tagging and cross-modal retrieval, and serving as a foundation for downstream generative tasks.
- (2) HeartTranscriptor: a robust lyric recognition model tailored to complex musical signals, providing accurate transcription of lyrical content.
- (3) HeartCodec: a low-frame-rate (12.5 Hz), high-fidelity music codec that captures both long-range structure and acoustic details. Its compact discrete representation enables high-quality reconstruction and efficient autoregressive modeling.
- (4) HeartMuLa: a multi-condition song generator that accepts flexible user inputs, including style descriptions, detailed lyrics, and reference audio, while offering fine-grained control over musical attributes such as genre, mood, rhythm, and expressive variations.

Our song generation model supports long-form music creation of up to six minutes, maintaining both structural coherence and expressive diversity over extended durations. In addition, it provides two specialized modes: (1) a short-music generation mode suitable for background music in short videos; and (2) a fine-grained style control mode that enables creators to control the style of different song parts (e.g., intro, verse, chorus) using natural language prompts.

Beyond the capabilities of individual models, the open-source nature of our ecosystem enables reproducibility, extensibility, and broad community adoption. By releasing model weights and evaluation protocols, we aim to provide a comprehensive foundation for future advancements in music intelligence.

In summary, our contributions are as follows:

- 1. We introduce a unified suite of open-source Music Foundation Models covering music–text alignment, music tokenization, lyric recognition, and controllable song generation.
- 2. We propose a novel music codec tokenizer that achieves high expressiveness at a low frame rate, enabling efficient and scalable modeling of long musical sequences.
- 3. We present a song generation framework with fine-grained musical control, supporting high-quality long-form generation of up to 6 minutes.

### 2 HeartCodec

In this section, we first present the architecture of HeartCodec in 2.1, detailing its semantic-rich encoder, ultra-low-frame-rate compressor, and high-fidelity reconstruction decoder (see Fig. 2). We then describe the training details in 2.2, including optimization objectives for different training stages and implementation specifics. Finally, 2.3 reports our experimental results, covering comparisons with baselines and ablation studies on design choices.

Encoder Decoder Compressor

| | |
|---|---|
| |Flow|
| |SQ|

+

Proj. & Cat.

| |
|---|

Matching

Layer K-1

[Figure 1]

Audio

Insert Query

+ Tokens

MuEncoder

- Layer 0

- Layer 1

Decoder

WavLM

Extract Query

[Figure 2]

Whisper

Figure 2: An illustration of our proposed HeartCodec. Left, middle, right are semantic-rich encoder, ultra-low frame rate compressor and high-fidelity reconstruction decoder, respectively.

#### 2.1 Architecture

HeartCodec compresses raw waveforms into discrete tokens through three parts, each contributing one of our core advantages: semantic-rich encoding, ultra-low-frame-rate tokenization, and high-fidelity reconstruction. We first employ pretrained audio encoders, including Whisper Radford et al. [2023], WavLM Chen et al. [2022], MuEncoder Xu et al. [2024], to produce semantically rich representations. Note that we use our training data to fine-tune the MuEncoder with the BEST-RQ Chiu et al. [2022], Won et al. [2023] loss. These representations are quantized via a query-based quantization strategy Yang et al. [2025b] to yield ultra-low-frame-rate discrete codes. Finally, a flow-matching-based decoder Liu et al. [2023] reconstructs the waveform from the quantized embeddings, achieving high-quality synthesis.

Semantic-Rich Encoder The inherent complexity of music has motivated prior work to demonstrate that leveraging representations across multiple abstraction levels, rather than relying solely on lowlevel acoustic features, leads to stronger downstream modeling performance Yuan et al. [2025], Lei et al. [2025], Xu et al. [2024], Lin et al. [2025]. Guided by this insight, we adopt a multi-encoder strategy and extract complementary representations from pretrained Whisper Radford et al. [2023], WavLM Chen et al. [2022], and fine-tuned Music Encoder models (MuEncoder), covering phonetic, music semantic, and acoustic levels. Specifically, we extract eleventh-layer features as music semantic representations from MuEncoder, capturing high-level musical attributes such as timbre, phrasing, and melodic structure, while second-layer features are used as acoustic representations encoding finegrained timbral and spectral details. To model phonetic-level cues related to vocal articulation and pronunciation, we use WavLM features obtained by averaging the outputs of layers 6 to 9, together with embeddings from the Whisper encoder. Formally, given an input stereo waveform x ∈ RTf

a×Ca with duration T, sample rate fa = 48kHz and channel Ca = 2, we denote by yi ∈ RTf

i×Ci the feature sequence extracted from the i-th pretrained module, where fi and Ci represent the corresponding frame rate and channel dimension. In particular, y1, y2, y3, and y4 correspond to the MuEncoder semantic features, WavLM phonetic features, Whisper embeddings, and MuEncoder acoustic features, respectively. Together, this set of representations {yi}4i=1 forms a coherent multilevel description that jointly captures phonetic, music semantic, and acoustic information, providing a rich and structured foundation for downstream modeling.

Ultra-Low Frame Rate Compressor The frame rate of audio tokenizer directly determines sequence length; however, recent LM-based music generation systems Copet et al. [2023], Yuan et al. [2025], Lei et al. [2025] still rely on audio codecs operating at relatively high frame rates (25–50 Hz), which limits scalability. In contrast, our tokenizer operates at a substantially lower frame rate of 12.5 Hz while jointly fusing multi-level representations. Concretely, for each multi-level feature sequence yi, we independently resample it to a common frame rate fh = 25 Hz. The temporally aligned features are fused into a single representation via channel-wise concatenation followed by a linear projection to dimension C, yielding yh ∈ RTf

h×C.. We then apply a query-based quantization strategy Yang et al. [2025b] on yh. After every two consecutive frames in yh, we insert a learnable query token, and process the resulting sequence with a Transformer encoder. The output embedding at each query-token position is retained as a learned summary of the two consecutive frames it follows,

while all non-query frame embeddings are discarded. This reduces the frame rate to fl = 12.5 Hz and produces a downsampled representation yl ∈ RTf

l×C, which is then discretized using a residual vector quantization (RVQ) module Zeghidour et al. [2022] with K = 8 codebooks of vocabulary size V = 8192, producing discrete indices A ∈ [V ]Tf

l×K, which serve as the audio token sequence for downstream language-model-based generation. The corresponding quantized representation is obtained by codebook lookup and summation across K codebooks, yielding yˆl ∈ RTf

l×C. We employ the standard RVQ commitment loss, averaged over frames

Tfl

1 Tfl

sg(yl,t) − yˆl,t 2. (1)

Lcommit =

t=1

To preserve the rich semantic information in the discretized representation, we employ upsamplers {Ui}4i=1 to align the quantized features yˆl with the reference features {yi}4i=1, yielding Ui(ˆyl) ∈ RTf

i×Ci. Alignment is encouraged by minimizing the feature alignment loss:

Tfi

Ui(ˆyl)⊤t yi,t ∥Ui(ˆyl)t∥∥yi,t∥

1 Tfi

L(aligni) = −

. (2)

log sigmoid

t=1

In practice, applying this loss to the MuEncoder semantic features and WavLM phonetic features (i = 1,2) yields the best results.

High-Fidelity Reconstruction Decoder Directly reconstructing waveforms from discretized representations typically results in lower fidelity than their continuous counterparts, due to quantizationinduced information loss. Following recent hybrid approaches Xu et al. [2024], Yang et al. [2025b], Liu et al. [2024], Lin et al. [2025], we reconstruct waveforms by first mapping discrete representations into a continuous latent space using a generative model, followed by waveform reconstruction using the corresponding decoder. Specifically, we employ a pretrained continuous audio tokenizer G and extract continuous latents z = GEnc(x) as reconstruction targets. We experiment with several continuous audio tokenizer Xu et al. [2024], Lei et al. [2025], Yang et al. [2024], and select a 25 Hz SQ-Codec Yang et al. [2024] based on our ablation results (See Sec 2.3). The latent distribution is modeled using flow matching Liu et al. [2023], where a vector field vθ transforms Gaussian noise z0 ∼ N(0,I) into the target latent z1 = z, conditioned on low-frame-rate discretized features yˆl. To enable seamless reconstruction across audio clips, we additionally condition the model on partial clean latents. Following Mehta et al. [2024], Le et al. [2023], we apply a binary mask m and provide the unmasked latent (1 − m) ⊙ z1 as input, while training the model to predict the masked portion m ⊙ z1. Together, the training objective becomes:

[∥vθ(zt,yˆl,(1 − m) ⊙ z1) − (m ⊙ (z1 − z0))∥2], (3)

Lfm = E

x,t,z0

where zt = tz1 + (1 − t)z0 is the noisy sample derived following schedule t ∼ U(0,1). We utilize a Diffusion Transformer Peebles and Xie [2023] backbone based on LLaMA architecture Team [2024]

for predicting vθ. All inputs zt,yˆl,(1 − m) ⊙ z1 are concated along feature channel. To further improve sampling efficiency, we adopt Reflow distillation Liu et al. [2023], reducing the number of sampling steps from 50 to 10. The distillation objective is defined as

Ldistill = E

x,t,z0

##### 0 ) 2 , (4)

−

−

−

−

1 − m ⊙ (zθ

1 − zθ

vθ zθ

t ,yˆl,(1 − m) ⊙ zθ

−

where θ− denotes a frozen copy of the trained vector field. The latent trajectories zθ

t are obtained by integrating the frozen vector field, zθ

t = z0 + 0 t vθ−(zs,s)ds. After distillation, the latent

−

distribution induced by z1θ still deviates from the ground-truth z1 = GEnc(x), where x ∼ pdata. To bridge this gap, we further fine-tune the SQ-Codec decoder, adapting it to the distilled latent

distribution by optimizing waveform reconstruction from x˜ = GDec(z1θ), yielding the following objective:

##### Lsq = Lrec(˜x,x) + λadv Ladv(˜x,x), (5)

where reconstruction loss Lrec consists of time-domain and frequency-domain terms, including an ℓ1 loss between the reconstructed waveform x˜ and the ground-truth waveform x, as well as an MSE loss computed on the STFT spectrogram. The adversarial loss Ladv is computed based on the discriminator outputs.

#### 2.2 Training Details

Our training pipeline is conducted in three stages. We begin by pretraining and fine-tuning HeartCodec from scratch, which establishes its fundamental modeling and reconstruction capabilities. Next, we perform ReFlow distillation to accelerate inference while maintaining generation quality. Finally, we fine-tune SQ-Codec on top of the distilled model to further improve reconstruction fidelity. The parameters of MuEncoder, WavLM and Whisper modules are kept frozen during all stages.

Pretraining and Finetuning The overall loss is formulated as a weighted sum of the terms in Eqs. 3, 1, and 2, yielding

L1 = λfm Lfm + λcommit Lcommit + λsem L(1)align + λpho L(2)align. (6)

We set λfm = 1, λcommit = 1, λsem = 0.1, and λpho = 0.1. The last two terms correspond to feature alignment losses regarding semantic features and phonetic (WavLM) features, respectively. We train our model on an internal dataset consisting of about 600,000 songs. During pretraining, we use a segment duration of T = 20.48s, while a longer duration of T = 29.76s is adopted for fine-tuning. All experiments are conducted on 8 NVIDIA A100 GPUs with a global batch size of 160. The model is trained for 15 epochs. We use the AdamW optimizer with a base learning rate of 1×10−4, together with a cosine learning rate scheduler that includes a warm-up period of the first 3% steps. We use HeartCodec (Pt.) and HeartCodec (Pt. & Ft.) to denote the model after pretraining only, and the model after pretraining followed by finetuning, respectively.

Reflow Distillation We perform reflow distillation on top of HeartCodec (Pt. & Ft.). During the ReFlow distillation stage, we curated a collection of 50,000 high-quality segments, each with

−

−

a duration of T = 29.76 s. Using the fine-tuned HeartCodec, we extracted triplets (el,zθ

0 ,zθ

1 ) consisting of the code, noise, and the model-predicted latent. Subsequently, we activated only the flow matching module and trained it on these 50,000 triplets according to Eq. (4). The training was conducted on 8 NVIDIA A100 GPUs for 2 epochs. We employed the AdamW optimizer with a learning rate of 5 × 10−6. We use HeartCodec (Reflow) to denote the model after reflow distillation.

Finetuning SQ-Codec Given the parameters of HeartCodec (Reflow), we further finetune SQCodec. In this stage, we freeze all parameters of HeartCodec except for the final SQ-Codec decoder. The model is optimized using the loss Lsq defined in Eq. 5. Training is performed on a filtered high-quality subset of approximately 20k samples, selected using objective metrics from AudioBox Aesthetics Tjandra et al. [2025] and SongEval Yao et al. [2025]. The model is trained for 3 epochs on 4 NVIDIA A100 GPUs. Both the generator and discriminator are optimized with the AdamW optimizer, using a learning rate of 2 × 10−6 and an exponential learning rate scheduler with decay factor γ = 0.999. The model after this stage is denoted as HeartCodec (SQ Ft.)

#### 2.3 Experiment Results

- 2.3.1 Experimental Setup We conduct objective and subjective studies to evaluate HeartCodec comprehensively. Objective Metrics. We assess performance across four critical dimensions:

- 1. Vocal fidelity, measured by Short-Time Objective Intelligibility (STOI), Perceptual Evaluation of Speech Quality (PESQ), Speaker Similarity (SPK_SIM) computed by ECAPA-TDNN model Desplanques et al. [2020], and Word Error Rate (WER) whose transcript is recognized by our HeartTranscriptor;
- 2. Overall reconstruction quality, evaluated using Virtual Speech Quality Objective Listener (VISQOL), Fréchet Audio Distance (FAD), and Fréchet Distance (FD);
- 3. Aesthetic metrics, including Content Evaluation (CE), Content Understanding (CU), and Perceptual Quality (PQ) from AudioBox Tjandra et al. [2025];
- 4. Musical Quality, including Coherence (Coh.), Musicality (Mus.), Memorability (Mem.), Clarity (Cla.), and Naturalness (Nat.). These metrics are computed by SongEval Yao et al. [2025] and used in Sec. 2.3.4 only.

- 5. Style Adherence, measured by Tag Similarity (Tag-Sim.) defined as the cosine similarity between the embeddings of the generated audio and the prompt style tags, extracted via the MuQ-MuLan model Zhu et al. [2025]. This metric is used in Sec. 2.3.4 only.
- 6. Computational efficiency, quantified by the Real-Time Factor (RTF), which measures the ratio of the processing time to the duration of the generated audio.

Subjective Metrics. We conducted a blind listening test with five expert listeners with musical backgrounds. Evaluators rated each sample on a scale of 1 to 5 across the following aspects:

- 1. Reconstruction Similarity, including Vocal Similarity (VS), Accompaniment Similarity (AS), and Mixing Similarity (MS), assessing how closely the components match the reference;
- 2. Acoustic Naturalness, comprising Vocal Naturalness (VN) and Melody Naturalness (MN), focusing on the absence of artifacts and the fluidity of the melody;
- 3. Perceptual Fidelity, involving Detail Retention (DR) and Overall Preference (Pref.), which reflect the model’s ability to preserve nuanced textures and the general listening satisfaction.

Implementation Details. The Flow Matching module in HeartCodec is configured with approximately 1.5B parameters. During the inference stage, the classifier-free guidance (CFG) scale is set to

- 1.25. For the sampling process, we employ 50 steps for the Pretrain and Finetune stages. To optimize inference efficiency, the sampling steps are reduced to 10 for the Reflow distillation and SQ-Finetune stages.
- 2.3.2 Reconstruction Quality

We conducted a comparative analysis against several representative music codecs, including SemantiCodec Liu et al. [2024], XCodec Ye et al. [2025], MuCodec Xu et al. [2024], and LeVo Lei et al. [2025]. The results of the objective evaluation are summarized in Table 1.

Table 1: Comparative Evaluation between Existing Codec Models.

Bitrate (kbps)

PESQ ↑ (WB/NB)

SPK_SIM ↑ WER ↓ CE ↑ CU ↑ PQ ↑ Ground Truth - - - - - - - - - 0.14 7.09 7.40 7.62 SemantiCodec

Model CodeBook Framerate

VISQOL ↑ FAD ↓ FD ↓ STOI ↑

1 x 32768 25 0.375 2.24 2.32 22.38 0.40 1.14/1.44 0.79 0.91 6.94 7.11 7.33 1 x 16384 100 1.40 2.28 1.92 16.90 0.53 1.30/1.82 0.89 0.52 6.91 7.12 7.31

- 1 x 1024 50 0.50 2.23 1.88 24.51 0.57 1.27/1.68 0.75 0.73 6.84 7.26 7.38
- 2 x 1024 50 1.00 2.28 1.20 18.30 0.64 1.43/1.95 0.84 0.52 6.86 7.21 7.41 4 x 1024 50 2.00 2.32 0.88 16.08 0.70 1.63/2.28 0.88 0.34 6.91 7.20 7.47 8 x 1024 50 4.00 2.35 0.70 14.78 0.74 1.87/2.62 0.91 0.27 6.91 7.22 7.50

XCodec

MuCodec 1 x 16384 25 0.35 3.07 1.02 14.73 0.45 1.12/1.36 0.76 0.54 6.99 7.26 7.59 LeVo (Mixed) 1 x 16384 25 0.35 3.24 1.03 19.44 0.49 1.14/1.44 0.79 0.47 7.09 7.33 7.63 LeVo (Dual-Track) 2 x 16384 25 0.70 3.26 1.45 19.96 0.56 1.21/1.61 0.82 0.35 7.20 7.54 7.88 HeartCodec (Pt.) 8 x 8192 12.5 1.30 3.57 0.52 12.33 0.64 1.45/2.02 0.90 0.26 7.06 7.26 7.48 HeartCodec (Pt. & Ft.) 8 x 8192 12.5 1.30 3.57 0.61 12.20 0.64 1.46/2.04 0.90 0.27 7.06 7.27 7.49 HeartCodec (Reflow) 8 x 8192 12.5 1.30 3.61 0.46 11.65 0.64 1.45/2.03 0.90 0.27 7.05 7.29 7.49 HeartCodec (SQ Ft.) 8 x 8192 12.5 1.30 3.72 0.27 11.06 0.66 1.52/2.10 0.90 0.26 7.05 7.36 7.57

Experimental results demonstrate that Reflow distillation using high-quality data yields marginal improvements in overall reconstruction metrics (VISQOL, FAD, and FD). Subsequent SQ-Finetune leads to significant gains across all indicators, establishing HeartCodec as the state-of-the-art music tokenizer. Notably, HeartCodec exhibits a decisive advantage in global reconstruction quality: its VISQOL substantially outperforms all baselines, while its FAD and FD are the lowest recorded. This underscores its superior capability in aligning both time-domain distributions and frequency-domain features with the original audio.

Regarding vocal fidelity, HeartCodec remains in the top tier with STOI of 0.66 and PESQ of 1.52/2.10. Moreover, it demonstrates exceptional semantic and identity preservation, achieving the lowest WER and a competitive SPK_SIM comparable to the high-bitrate XCodec. While XCodec achieves a higher STOI leveraging its higher bitrate and framerate, its integrated audio quality metrics like FAD and VISQOL still lag behind HeartCodec. In terms of aesthetics, the SQ-Finetune strategy allows HeartCodec to outperform most models in CE, CU, and PQ, trailing only slightly behind the dual-track architecture of LeVo.

Table 2: Comparative Evaluation of HeartCodec Under Varying Guidance Scales.

PESQ ↑ (WB/NB)

SPK_SIM ↑ WER ↓ CE ↑ CU ↑ PQ ↑

Model Guidance Scale VISQOL ↑ FAD ↓ FD ↓ STOI ↑

HeartCodec (Pt.) 1.25 3.57 0.52 12.33 0.64 1.45/2.02 0.90 0.26 7.06 7.26 7.48 HeartCodec (Pt. & Ft.) 1.25 3.57 0.61 12.20 0.64 1.46/2.04 0.90 0.27 7.06 7.27 7.49 HeartCodec (Pt.) 1.5 3.51 1.21 17.65 0.68 1.49/2.13 0.89 0.24 6.98 7.11 7.39 HeartCodec (Pt. & Ft.) 1.5 3.61 0.61 14.02 0.68 1.56/2.18 0.90 0.24 7.07 7.35 7.62

We further evaluate the performance of the fine-tuning stage across different guidance scales in Table 2. While the objective metrics show marginal improvement at a guidance scale of 1.25, the performance gains at a guidance scale of 1.5 are substantial, underscoring the effectiveness of our fine-tuning approach. However, despite the superior objective scores at guidance scale 1.5, subjective assessments reveal that a guidance scale of 1.25 yields a more natural and balanced auditory experience, with vocals and accompaniment sounding smoother and less harsh. Consequently, we select 1.25 as our default guidance scale.

As the ideal tokenizer for the HeartMuLa, HeartCodec’s strength lies not only in high reconstruction quality and rich semantic capture but also in its breakthrough regarding extremely low frame rate in music tokenizers. Given HeartMuLa’s Global-Local architecture, where a lightweight Local Transformer predicts a multi-layer RVQ, the computational latency added by increasing the RVQ layers is significantly lower than the linear growth caused by increasing the frame rate. HeartCodec’s low frame rate and multi-layer RVQ design enhance the scalability of HeartMuLa, providing a robust foundation for high-fidelity, streaming music generation.

#### 2.3.3 Ablation Study of the Latent Space

We utilize the bottleneck features from three distinct models, Mel VAE, 1D VAE, and SQ-Codec, as the target latents for the Flow Matching model. Specifically, 1D VAE and SQ-Codec reconstruct waveforms directly from their respective latents, whereas Mel VAE first recovers the Mel-spectrogram, followed by waveform synthesis via HiFi-GAN Kong et al. [2020]. We evaluate the reconstruction fidelity and computational efficiency of HeartCodec under these different objective latents upon completion of pre-training. This analysis aims to explore the impact of various latent representations on music reconstruction, with experimental results summarized in Table 3.

Table 3: Comparative Evaluation of HeartCodec Under Varying Latent Features.

PESQ ↑ (WB/NB)

Model VISQOL ↑ FAD ↓ FD ↓ STOI ↑

SPK_SIM ↑ WER ↓ RTF ↓

HeartCodec (Mel VAE) 3.43 0.51 8.15 0.54 1.27/1.74 0.88 0.34 0.397 HeartCodec (1D VAE) 3.57 0.84 12.95 0.63 1.46/2.06 0.89 0.32 0.116 HeartCodec (SQ-Codec) 3.57 0.52 12.33 0.64 1.45/2.02 0.90 0.26 0.121

Compared to 1D VAE, SQ-Codec achieves significantly superior scores in distribution-based metrics, specifically FAD and FD, while maintaining comparable performance across signal-level reconstruction metrics (VISQOL, STOI, and PESQ) and computational efficiency (RTF). Conversely, although Mel VAE exhibits lower FAD and FD, it is markedly inferior to SQ-Codec and 1D VAE in all other objective reconstruction indicators. Furthermore, the prohibitive RTF of Mel VAE (0.397) poses a substantial bottleneck for real-time streaming music generation.

Table 4: Subjective Evaluation of HeartCodec Under Different Latent Features.

Model VS ↑ AS ↑ MS ↑ VN ↑ MN ↑ DR ↑ Pref. ↑

HeartCodec (Mel VAE) 3.58 3.68 2.75 3.43 4.00 3.23 3.33 HeartCodec (1D VAE) 3.30 3.35 2.90 3.21 3.70 3.00 2.90 HeartCodec (1D SQ) 3.90 4.00 3.33 3.79 4.25 3.78 3.85

To further bridge the gap between objective metrics and human auditory perception, we assess the models’ subjective performance in Table 4. SQ-Codec outperforms its counterparts across all seven subjective dimensions. Notably, in terms of Vocal Similarity (VS) and Melody Naturalness (MN), SQ-Codec achieves scores of 3.90 and 4.25, respectively, significantly higher than 1D VAE. Taking all factors into consideration, we follow the setting of SimpleSpeech Yang et al. [2024, 2025a], and designate the bottleneck features of SQ-Codec as the target latent for the Flow Matching module.

#### 2.3.4 Ablation Study of Training Stages

To further assess the impact of reflow distillation and fine-tuning SQ-Codec on downstream music generation tasks, we conduct an additional set of experiments. Specifically, our music generation model, HeartMuLa (see Section 3.1), is trained to model token sequences produced by HeartCodec. During inference, we let HeartMuLa predict a sequence of tokens given lyrics and style tags as prompts, keep the predicted tokens fixed, and decode them using HeartCodec obtained at different training stages.

Specifically, we consider HeartCodec after pretraining and finetuning (Pt. & Ft.), after reflow distillation (Reflow), and after SQ-Codec fine-tuning (SQ Ft.). Notably, during both reflow distillation and SQ-Codec fine-tuning, the encoder and compressor components of HeartCodec are kept frozen. Therefore, using the same token sequence as input across different decoders is well justified and allows for a controlled comparison.

After decoding, we evaluate the generated music using metrics including aesthetic metrics produced by AudioBox Tjandra et al. [2025], musical quality measured by SongEval Yao et al. [2025], style alignment measured by tag similarity (Tag-Sim) and intelligibility of the vocal track measured by the phoneme error rate (PER). We report results on our HeartBeats-Benchmark (English) dataset (see Sec. 6.5), as summarized in Table 5. As shown in the results, compared to HeartCodec (Pt. & Ft.), the model after reflow distillation achieves noticeable improvements in both aesthetic quality and tag similarity, albeit with a degradation in vocal intelligibility. After further fine-tuning the SQ-Codec , all evaluation metrics consistently improve, yielding the best overall performance among the compared methods. These results demonstrate that the two additional training stages, reflow distillation and SQ-Codec fine-tuning, consistently enhance the quality of downstream music generation.

Table 5: Objective Evaluation of the Effects of Training Stages on Downstream Music Generation.

AudioBox ↑ SongEval ↑ Align ↑ Intel ↓ CE CU PQ Coh. Mus. Mem. Cla. Nat. Avg. Tag-Sim PER

Model

HeartCodec (Pt. & Ft.) 7.37 7.78 8.05 4.43 4.24 4.43 4.30 4.13 4.31 0.2339 0.1092 HeartCodec (Reflow) 7.42 7.78 8.04 4.46 4.28 4.46 4.32 4.18 4.34 0.2486 0.1258 HeartCodec (SQ Ft.) 7.45 7.78 8.07 4.52 4.36 4.54 4.38 4.25 4.41 0.2499 0.1005

### 3 HeartMuLa

In this section, we introduce HeartMuLa (Heart Music Language Model), a novel framework for music generation that operates on discrete audio tokens produced by our HeartCodec. HeartMuLa is designed to deliver long-form, high-fidelity, and controllable music synthesis. The section is organized as follows: Sec. 3.1 presents the hierarchical architecture, which enables efficient and high-fidelity generation; Sec. 3.2 details the integration of diverse conditions for precise control. The overall architecture of HeartMuLa is illustrated in Fig. 3; Sec. 3.3 describes the training strategy and implementation details; and Sec. 3.4 presents the evaluation of the model.

#### 3.1 Hierarchical Architecture

Following previous work Yang et al. [2023b], HeartMuLa employs a hierarchical factorization of the modeling process, initially capturing the coarse, long-range musical structure and subsequently incorporating fine-grained acoustic details. Specifically, we adopt our HeartCodec to tokenize audio into RVQ token sequence A = [a0,a1,...,aL−1] ∈ [V ]L×K, where L = Tfa denotes the number of frames, and each frame al = [al,0,...,al,K−1] consists of K RVQ tokens. Let hl,k denote the

Text Token

Generated Music

Contiunous Token

HeartCodec Detokenizer

[Figure 3]

Music Token

Music Hidden State

HeartMuLa Local Decoder

HeartMuLa Global Backbone

| | |
|---|---|
| | |

| | |
|---|---|
| | |

###### Text Tokenizer Audio Encoder HeartCodec Tokenizer

Reference Music Target Music (for Training)

###### Descriptions Lyrics

[Figure 4]

[Figure 5]

electronic, acoustic, synthesizer, soft

[Intro] [Verse] Hello HeartMuLa...

Figure 3: HeartMuLa Architecture

embedding of token al,k, and define hl = Kk=0−1 hl,k as the embedding of frame al. The modeling process is decomposed into global and local stages: a global transformer θglo which first models intra-frame dependencies by predicting al,0 at each frame l, conditioned on the preceding frame embeddings h<l, capturing the coarse semantic information encoded in the RVQ Layer 0 code, followed by a subsequent local transformer θloc predicting the remaining tokens al,k within this frame, conditioned on both the hidden state of the global transformer θglo(h<l) and local token embeddings hl,<k. The overall probability is given by:

K−1

p(al,k | hl,<k,θglo(h<l);θloc) (7)

p(al | h<l;θglo,θloc) = p(al,0 | h<l;θglo)

k=1

This hierarchical architecture simultaneously delivers computational efficiency and high-fidelity generation capabilities. Enhanced Efficiency is achieved through sequence factorization; the largescale global transformer is tasked solely with predicting the base tokens of layer 0 rather than the entire multi-layer hierarchy. This strategy significantly mitigates the computational overhead and modeling complexity associated with predicting multi-stream codebooks. Furthermore, the architecture leverages the global transformer to capture coarse-grained semantic and structural patterns across frames, while offloading the synthesis of intricate acoustic details to the local transformer. These components work in tandem to elevate the overall quality of the generated audio synergistically.

#### 3.2 Conditioning Mechanism

HeartMuLa uses lyrics, complemented by optional tags and reference audio as conditioning signals to enable precise control over the generation process.

Lyrics are annotated with structural markers such as [intro], [verse], and [chorus], which guide the model in identifying and preserving the song’s structure. These markers are retained during tokenization by the Llama-3.2 tokenizer Team [2024], yielding lyrics token sequence which is further embedded into Clyrics.

Tags capture high-level musical attributes, each falling under a specific category (e.g., genre, instrument), with varying levels of influence on the music. To prioritize categories with greater impact, we empirically assign a selection probability to each category, ensuring that more influential tags, such as genre, are given higher probabilities compared to less impactful ones, such as topic. The specific categories and their selection probabilities are provided in Table 6. These selected descriptions are then encapsulated within special tokens <tag> and </tag>, followed by tokenization via the Llama-3.2 tokenizer Team [2024] and embedding to form the tag sequence, represented as Ctag.

Reference Audio serves as a global stylistic cue. During training, we randomly sample a 10second segment from the ground-truth music and employ pre-trained MuQ-MuLan Zhu et al. [2025] embeddings 1 to characterize its musical style. This conditioning signal, denoted as Cmuq, is discarded with a 50% probability during training to facilitate unconditional modeling.

Together, Clyrics, Ctag and Cmuq form our overall condition sequence C, which is prepended before h0 and integrated as the prompt condition. This leads to a slight modification in the training objective compared to Eq. 7, and will be detailed in Sec. 3.3.2. Additionally, not all conditions are used during every training stage, as will be explained in Sec. 3.3.1.

Table 6: Selection Probabilities for Different Tag Categories Category Genre Timbre Gender Mood Instr. Scene Region Topic Prob. 0.95 0.5 0.375 0.325 0.25 0.2 0.125 0.1

#### 3.3 Training

We adopt a four-stage progressive training paradigm, as depicted in Fig. 4, which includes warm-up, pre-training, supervised fine-tuning, and reinforcement learning. We introduce each stage briefly in Sec. 3.3.1 and corresponding training objectives in Sec. 3.3.2. Implementation details are provided in Sec. 3.3.3.

#### 3.3.1 Four-Stage Progressive Training Paradigm

- Stage 1: Warmup. In this stage, we train HeartMuLa on 30-second music segments containing

lyrics. The input conditions C = [Cmuq,Clyrics] include the lyrics and the reference audio. The core objective of this stage is to facilitate rapid parameter convergence and establish a preliminary mastery of local acoustic texture modeling laws through dense training on short contexts, laying a foundational acoustic capability for subsequent long-sequence generation.

- Stage 2: Pretraining. In this stage, HeartMuLa is trained on a large-scale dataset of full songs,

using all three conditions C = [Ctag,Cmuq,Clyrics]. This stage aims to leverage massive data to force the model to learn long-range temporal dependencies and global musical structures under complete conditional inputs, thereby achieving precise adherence to lyrical content and effective control over the overall musical style.

- Stage 3: Supervised Finetuning. In this stage, we use AudioBox Tjandra et al. [2025] and SongEval Yao et al. [2025] to filter a high-quality subset from the full data. Then we fine-tune the model using

all conditions C = [Ctag,Cmuq,Clyrics]. This stage aims to improve both the synthesis quality and the fine-grained structural control of the generated music via fine-tuning optimization.

- Stage 4: Direct Preference Optimization. To further elevate the perceptual quality of the generated music, we use a reinforcement alignment stage utilizing Direct Preference Optimization (DPO) Rafailov et al. [2023]. By leveraging preference data constructed with metrics including tag similarity, Phonemes Error Rate (PER), and AudioBox Tjandra et al. [2025] scores, DPO enables the model to effectively distinguish and generate superior music. This stage is specifically designed to enhance multi-dimensional generation quality, resulting in significant improvements in vocal clarity, stylistic fidelity, and overall audio coherence.

1We do not want to release a model that can copy speaker’s timbre, and we find that MuQ-MuLan does not include any speaker timbre information. Thus, we choose MuQ-Mulan to extract the style embedding.

###### Stage 1 Stage 2 Stage 3 Stage 4

Warmup Pre-Training

Supervised Fine-tuning

Direct Preference Optimization

[Figure 6]

[Figure 7]

Short Music Data

[Figure 8]

Full Data

Music Emb

Full Data

|winner|
|---|

[Figure 9]

###### Tag HeartMuLa

Music Emb Tag

Target Music

Target Music

Music Emb

Filtering

|loser|
|---|

Music Emb

[Figure 10]

[Figure 11]

HeartMuLa Lyrics

HeartMuLa

Tag

HeartMuLa Lyrics

[Figure 12]

Lyrics

Preference Optimization

Lyrics

< 30s

full sequence

High Quality Subset

###### Stage 1 Stage 2 Stage 3 Stage 4

Warmup Pre-Training

Supervised Fine-tuning

Direct Preference Optimization

[Figure 13]

[Figure 14]

Short Music Data

[Figure 15]

Full Data

Music Emb

Full Data

winner loser

[Figure 16]

###### Tag HeartMuLa

Music Emb Tag

Target Music

Target Music

Music Emb

Filtering

Music Emb

[Figure 17]

[Figure 18]

HeartMuLa Lyrics

HeartMuLa

Tag

HeartMuLa Lyrics

[Figure 19]

Lyrics

###### Preference Optimization

Lyrics

< 30s

full sequence

High Quality Subset

Figure 4: Four-Stage Progressive Training Paradigm

#### 3.3.2 Optimization Objectives

Weighted CrossEntropy Loss Based on the hierarchical modeling architecture of HeartMuLa, our optimization objective is decomposed into a global loss L0 targeting long-term semantic modeling and residual terms Lk,(k ∈ [1,K − 1]) focusing on acoustic detail reconstruction. The total loss function Ltotal is defined as follows:

K−1

1 K − 1

λkLk (8)

Ltotal = λ0L0 +

k=1

where each term corresponds to the cross-entropy loss at the specific RVQ layer:

L−1

1 L − 1

log p(al,0 | h<l,C;θglo) (9)

L0 = −

l=1

L−1

1 L − 1

log p(al,k | hl,<k,θglo(h<l,C);θloc),∀k ∈ [1,K − 1] (10)

Lk = −

l=1

Here, λ0 is the weight for loss at layer 0, and λk for the loss at the k-th residual layer. We assign more weight to layer 0 by averaging the residual terms, as layer 0 captures the coarse semantic

information. The total loss function Ltotal integrates the prefix condition C into Eq. 7, applying the negative log-likelihood followed by weighted summation.

During the Warmup and Pretraining stages, we aim for the model to learn the acoustic feature distribution across all layers in a balanced manner. Consequently, we set the global loss coefficient to λ0 = 1.0 and assign uniform weights to all residual layers, i.e., λk = 1.0,∀k ∈ [1,K − 1]. During the Supervised Finetuning stage, we raise the weight of global loss to λ0 = 2.0 to emphasize musical structure and semantics, while weights for the remaining terms are set to λk = K10−k to attenuate their influence.

Direct Preference Optimization Loss Traditional Reinforcement Learning (RL) methods for LLMs (e.g., PPO) Schulman et al. [2017] rely on an explicit reward model and online sampling, leading to substantial computational overhead and training instability. We instead adopt Direct Preference Optimization (DPO) Rafailov et al. [2023], which casts the RL objective as supervised learning and directly optimizes the policy with preference pairs.

Given a dataset of preference pairs D = {(C,Awn,Als)}, where C denotes the prompt (including style description, structure lyrics and optional reference music), Awn represents the preferred (winning) audio sequence, and Als represents the dispreferred (losing) audio sequence, DPO derives a closed-form solution to the constrained reward maximization problem. The resulting objective bypasses explicit reward modeling by expressing the optimal policy directly in terms of the logprobability ratio between the policy model pθ and a reference model pref:

wn,Als)∼D [log σ (β · ∆θ(C,Awn,Als))], (11)

LDPO(θ) = −E(C,A

where σ(·) denotes the sigmoid function, β is a temperature hyperparameter controlling the deviation from the reference policy, and

∆θ(C,Awn,Als) = (log pθ(Awn|C) − log pref(Awn|C)) − (log pθ(Als|C) − log pref(Als|C)).

(12)

θ(A|C)

This formulation implicitly defines the reward as r(C,A) = β log p

pref(A|C), enabling the model to learn from preference comparisons without requiring reward labels or online generation during training.

HeartMuLa employs a hierarchical modeling process where layer 0 code is modelled by global transformer and residual codes by local transformer. This design necessitates a decomposition of the log-probability. Similar to Eq. 8, the log-probability is factorized as follows:

log pθ(A|C) =

L

L

log p(al,0 | h<l;θglo) +

l=1

l=1

K−1

log p(al,k | hl,<k,θglo(h<l);θloc). (13)

k=1

Substituting Eq. 13 into Eq. 11 yields the final DPO loss, which is naturally decomposed into two additive terms according to the hierarchical structure of HeartMuLa: one associated with the global semantic codes (layer-0) and another associated with the local residual acoustic codes (layers 1 to K − 1). Importantly, this decomposition implies that preference signals (rewards) can independently influence global semantic coherence and local acoustic details. As a result, the model can adjust its global semantic planning and local acoustic rendering in a disentangled manner, enabling more stable and targeted optimization compared to applying a single monolithic DPO objective over all tokens.

#### 3.3.3 Implementation Details

Built upon the Llama 3.2 architecture Team [2024], HeartMuLa integrates a 3B-parameter global backbone with a 300M-parameter local decoder, optimized using the training strategy detailed in Section 3.3.

Warmup Details. We initiated training with a 10,000-hour subset randomly sampled from our 100,000-hour high-quality music corpus, segmented into 30-second clips. This stage ran on 8 NVIDIA A100 80GB GPUs for 5 epochs. We utilized the AdamW optimizer with a Cosine scheduler, setting the learning rate to 2e − 4 and the CFG dropout probability to 0.02 to support Classifier-Free Guidance.

Pretraining Details. Subsequently, we scaled the training to the full 100,000-hour dataset on 64 NVIDIA A100 GPUs. The model was trained for another 5 epochs. While maintaining the optimizer configuration and CFG dropout, we annealed the learning rate to 2e − 5 to ensure stability.

SFT Details. This stage employed a curated dataset of 15,000 hours of high-quality music. Training was executed on 8 NVIDIA A100 GPUs for 3 epochs, using the same learning rate 2e − 5 and hyperparameters as the pre-training stage.

DPO Data Preparation. To construct the preference data for DPO, we employed a divergent sampling strategy. For each prompt input, the model generated four candidate audio samples with varying quality. Based on these candidates, we curated three distinct preference datasets, each prioritizing different metrics to guide the optimization process. In all cases, a preference pair (yw,yl) consists of a chosen winner sample and a loser sample. The specific criteria are as follows:

- • Muq-similarity-based Set: To enhance semantic alignment, the candidate with the highest Muq-similarity score was selected as yw, and the one with the lowest score as yl. We enforced a margin of simw −siml > 0.12 to ensure sufficient discriminability, and required simw > 0.3 to guarantee the fundamental quality of the positive sample.
- • PER-based Set: Focusing on articulation accuracy, we selected the sample with the lowest Phoneme Error Rate (PER) as yw and the highest as yl. A margin constraint of |PERw − PERl| > 0.1 was applied.
- • Audiobox & SongEval-based Set: To capture holistic audio quality, yw was identified as the sample achieving superior scores on both Audiobox Tjandra et al. [2025] and SongEval

Yao et al. [2025] metrics, while yl performed worst on both. We filtered pairs to ensure a SongEval score margin > 0.5 and an Audiobox average score margin > 0.8.

Training Configuration. For the DPO training phase, we set the learning rate to 1 × 10−7 and the KL penalty parameter β to 0.1. The model was trained for 3 epochs on a cluster of 8 NVIDIA A100 GPUs.

#### 3.4 Evaluation

- 3.4.1 Evaluation Setup We conduct both objective and subjective studies to comprehensively evaluate HeartMuLa.

Objective Metrics. To assess the performance of HeartMuLa, we employ a set of objective metrics focusing on three key aspects: Music Quality, Style Adherence, and Lyric Intelligibility.

- 1. Musical Quality: To assess the general audio quality and musical structure, we adopt two comprehensive metrics for generated music. SongEval Yao et al. [2025] evaluates structural and musical aspects, including Coherence (Coh.), Musicality (Mus.), Memorability (Mem.), Clarity (Cla.), and Naturalness (Nat.). AudioBox Tjandra et al. [2025] focuses on audio quality, providing scores for Content Evaluation (CE), Content Understanding (CU), and Perceptual Quality (PQ).
- 2. Style Adherence: To evaluate the controllability of our model, we utilize the Tag Similarity. This metric computes the cosine similarity between the embeddings of the generated audio and the target style tags, extracted via the MUQ-MuLan model Zhu et al. [2025].
- 3. Lyric Intelligibility: To measure how clearly the model generates lyrics, we calculate the Phoneme Error Rate (PER). Specifically, we extract vocals from the generated music using Demucs Défossez et al. [2019], transcribe them with our HeartTranscriptor in Section 5, and compute the phoneme-level discrepancy between the transcription and the reference lyrics.

Subjective Metrics. To capture perceptual nuances that objective metrics might miss, we conducted a blind listening test. Human evaluators rated the generated samples on a 5-point Mean Opinion Score (MOS) scale across six distinct dimensions:

- 1. Musicality: Focuses on the aesthetic appeal, specifically assessing the pleasantness of the melody and the depth of emotional expression.
- 2. Harmony: Evaluates the acoustic cohesion, measuring how naturally the vocals blend with the accompaniment.
- 3. Structure: Examines the organization of the song, checking for a clear musical progression and structural completeness.
- 4. Fidelity: Measures the pure audio quality, specifically the clarity of sound and the absence of background noise or artifacts.
- 5. Creativity: Assesses the novelty and inventiveness of the musical content.
- 6. Memorability: Reflects the "catchiness" of the song, indicating how easily the melody stays in the listener’s mind.

Implementation Details. During inference, we set the CFG scale to 1.5, temperature to 1.0, and top-k to 50. We use HeartCodec for detokenization. All evaluation experiments are conducted on NVIDIA A100 80GB GPUs.

#### 3.4.2 Objective Evaluation

We evaluated representative open-source models including LeVo Lei et al. [2025], DiffRhythm 2 Jiang et al. [2025], YuE Yuan et al. [2025], and ACE-Step Gong et al. [2025] and closed-source models including Suno-v5, Suno-v4.5, Mureka-V7.6, Udio-v1.5, and MiniMax-Music-2.0.

To conduct a comprehensive objective evaluation, we utilize the multilingual HeartBeats Benchmark introduced in Section 6.5. To ensure fair comparison, the input conditions are strictly limited to lyrics and textual style descriptions, as audio reference prompts are not supported by all baselines. Furthermore, models with inherent language constraints (e.g., LeVo Lei et al. [2025]) are evaluated solely on their supported languages. The detailed objective results are reported separately by language: Table 7 presents the performance on the English dataset, followed by Chinese in Table 8, Japanese in Table 9, Korean in Table 10, and Spanish in Table 11.

As shown in Tables 7 to 11, HeartMuLa demonstrates stable and competitive performance across all five languages. The most significant advantage of our model is lyric clarity, where it achieves

Table 7: Objective evaluation results on the HeartBeats Benchmark (English).

AudioBox ↑ SongEval ↑ Align ↑ Intel ↓ CE CU PC PQ Coh. Mus. Mem. Cla. Nat. Avg. Tag-Sim PER

Model

Suno-v5 7.65 7.83 6.46 8.21 4.62 4.51 4.63 4.51 4.44 4.54 0.26 0.13 Suno-v4.5 7.62 7.84 6.27 8.24 4.60 4.48 4.61 4.48 4.48 4.51 0.25 0.14 Mureka-V7.6 7.45 7.73 6.35 8.07 4.41 4.24 4.39 4.24 4.17 4.29 0.27 0.28 Udio-v1.5 7.52 7.69 6.18 7.98 4.12 3.92 4.08 3.88 3.82 3.97 0.23 0.25 MiniMax-2.0 7.73 7.98 6.45 8.35 4.60 4.49 4.59 4.48 4.39 4.51 0.26 0.13

LeVo 7.55 7.79 5.81 8.22 3.62 3.43 3.49 3.42 3.29 3.45 0.13 0.22 YuE 6.26 7.21 4.58 7.18 2.80 2.58 2.66 2.54 2.61 2.64 0.14 0.65

- DiffRhythm 2 7.23 7.58 6.37 7.88 4.00 3.81 4.03 3.82 3.67 3.87 0.32 0.28 ACE-Step 7.49 7.61 5.86 7.90 3.45 3.17 3.31 3.27 3.06 3.25 0.17 0.21 HeartMuLa (Ours) 7.55 7.82 5.89 8.14 4.52 4.36 4.54 4.35 4.24 4.48 0.26 0.09

Table 8: Objective evaluation results on the HeartBeats Benchmark (Chinese).

Model

AudioBox ↑ SongEval ↑ Align ↑ Intel ↓ CE CU PC PQ Coh. Mus. Mem. Cla. Nat. Avg. Tag-Sim PER

Suno-v5 7.59 7.80 6.44 8.24 4.66 4.55 4.65 4.57 4.49 4.58 0.29 0.23 Suno-v4.5 7.60 7.83 6.12 8.29 4.62 4.51 4.61 4.54 4.46 4.55 0.27 0.24 Mureka-V7.6 7.39 7.70 6.39 8.16 4.42 4.27 4.36 4.28 4.20 4.30 0.28 0.29 Udio-v1.5 7.35 7.57 6.16 8.02 4.14 3.95 4.05 3.94 3.87 3.99 0.26 0.50 MiniMax-2.0 7.66 7.86 6.36 8.34 4.59 4.49 4.54 4.48 4.40 4.50 0.27 0.19

LeVo 7.63 7.73 6.06 8.37 3.43 3.30 3.21 3.26 3.18 3.28 0.16 0.26 YuE 6.84 7.33 5.24 7.56 3.19 2.97 3.01 2.99 2.98 3.03 0.16 0.50

- DiffRhythm 2 7.23 7.59 6.25 8.07 3.92 3.72 3.85 3.76 3.60 3.77 0.29 0.32 ACE-Step 7.60 7.67 6.10 8.14 3.92 3.59 3.73 3.70 3.51 3.69 0.20 0.24 HeartMuLa (Ours) 7.46 7.78 5.97 8.12 4.57 4.45 4.58 4.45 4.36 4.48 0.24 0.12

Table 9: Objective evaluation results on the HeartBeats Benchmark (Japanese).

AudioBox ↑ SongEval ↑ Align ↑ Intel ↓ CE CU PC PQ Coh. Mus. Mem. Cla. Nat. Avg. Tag-Sim PER

Model

###### Suno-v5 7.47 7.71 6.57 8.21 4.63 4.54 4.60 4.51 4.44 4.54 0.32 0.35

- Suno-v4.5 7.44 7.73 6.44 8.22 4.58 4.48 4.56 4.48 4.42 4.51 0.27 0.29 Mureka-V7.6 7.52 7.77 6.39 8.34 4.52 4.38 4.49 4.37 4.32 4.42 0.29 0.30 Udio-v1.5 7.45 7.54 5.72 8.01 4.17 3.93 4.02 3.92 3.86 3.98 0.25 0.37 MiniMax-2.0 7.65 7.87 6.38 8.34 4.49 4.36 4.47 4.34 4.28 4.39 0.27 0.24

YuE 5.79 7.14 3.88 7.25 2.67 2.43 2.45 2.39 2.46 2.48 0.20 0.62 ACE-Step 7.62 7.69 6.14 8.23 3.71 3.38 3.49 3.42 3.27 3.45 0.23 0.33

HeartMuLa (Ours) 7.42 7.80 6.32 8.19 4.62 4.47 4.62 4.47 4.38 4.51 0.24 0.20

Table 10: Objective evaluation results on the HeartBeats Benchmark (Korean).

AudioBox ↑ SongEval ↑ Align ↑ Intel ↓ CE CU PC PQ Coh. Mus. Mem. Cla. Nat. Avg. Tag-Sim PER

Model

Suno-v5 7.44 7.75 6.40 8.15 4.61 4.52 4.61 4.52 4.47 4.54 0.30 0.38 Suno-v4.5 7.42 7.77 6.40 8.20 4.61 4.47 4.55 4.50 4.41 4.51 0.28 0.40 Mureka-V7.6 7.34 7.71 6.27 8.12 4.52 4.37 4.50 4.35 4.31 4.41 0.28 0.49 Udio-v1.5 7.20 7.37 5.61 7.79 3.91 3.71 3.83 3.61 3.63 3.74 0.20 0.57 MiniMax-2.0 7.62 7.92 6.22 8.38 4.57 4.43 4.51 4.40 4.33 4.45 0.26 0.40

YuE 5.21 6.98 3.61 7.12 2.68 2.43 2.55 2.41 2.44 2.50 0.14 0.56 ACE-Step 7.53 7.67 6.12 8.17 3.53 3.25 3.33 3.29 3.15 3.31 0.20 0.38

HeartMuLa (Ours) 7.51 7.79 6.09 8.16 4.56 4.45 4.56 4.43 4.34 4.47 0.21 0.16

Table 11: Objective evaluation results on the HeartBeats Benchmark (Spanish).

AudioBox ↑ SongEval ↑ Align ↑ Intel ↓ CE CU PC PQ Coh. Mus. Mem. Cla. Nat. Avg. Tag-Sim PER

Model

- Suno-v5 7.48 7.71 6.47 8.20 4.55 4.42 4.53 4.43 4.37 4.46 0.25 0.15

- Suno-v4.5 7.45 7.75 6.27 8.24 4.60 4.47 4.59 4.49 4.44 4.52 0.19 0.31 Mureka-V7.6 7.33 7.57 6.45 8.05 4.32 4.11 4.23 4.11 4.06 4.17 0.25 0.24 Udio-v1.5 7.31 7.45 5.95 7.85 4.00 3.79 3.86 3.77 3.67 3.82 0.17 0.37 MiniMax-2.0 7.71 7.91 6.50 8.35 4.49 4.35 4.47 4.32 4.25 4.38 0.22 0.12 ACE-Step 7.43 7.44 6.18 7.99 3.24 2.92 2.96 2.94 2.87 2.99 0.10 0.21 HeartMuLa (Ours) 7.39 7.72 6.12 8.16 4.46 4.25 4.46 4.29 4.22 4.34 0.22 0.13

the lowest Phoneme Error Rate (PER) in every language tested. For instance, in English (0.09) and Chinese (0.12), HeartMuLa outperforms top closed-source models like Suno-v5 and MiniMax-Music-

- 2.0, proving that it generates words much more clearly without blurring.

Regarding musical quality, HeartMuLa maintains high and consistent SongEval Yao et al. [2025] scores, with its structural and naturalness ratings matching professional standards. Unlike many open-source baselines that struggle with non-English tasks, HeartMuLa shows no performance drop across different regions. It maintains steady style adherence and audio quality in Chinese, Japanese, Korean, and Spanish alike. Overall, HeartMuLa provides a balanced combination of high-quality music production and industry-leading lyric accuracy for a global audience.

- 3.4.3 Subjective Evaluation

In this subjective evaluation experiment, 9 listeners participated in the test. Each rater evaluated 20 randomly selected samples from each model, including 10 English samples and 10 Chinese samples. To ensure fairness, a double-blind procedure with randomized trial order was adopted to minimize potential biases.

To refine the raw annotations, we employed the CrowdMOS framework to remove outliers and exclude annotators who did not listen to the full audio samples. Following common industry practice, all audio samples were loudness normalized to -14 dB LUFS prior to evaluation.

The subjective evaluation focused on seven perceptual dimensions:

- • Musicality: The overall aesthetic quality and emotional expressiveness of the song.
- • Harmony: The coherence and compatibility between vocals and accompaniment.
- • Structure: The logical progression and organization of the musical sections.
- • Fidelity: The perceived audio quality and absence of artifacts.
- • Creativity: The novelty and originality of the composition and performance.
- • Memorability: The catchiness and long-term recall of the melody.
- • Text Alignment: The semantic and rhythmic consistency between the generated lyrics and the musical content.

Participants were instructed to rate each stimulus according to the above criteria. The collected valid annotations were aggregated to compute the Mean Opinion Score (MOS) for each dimension.

All reported MOS values are computed as trimmed means by removing the highest and lowest ratings, with 95% confidence intervals calculated based on the trimmed samples.

#### 3.4.4 Ablation Study

The Effect of Different Training Phases. To understand the contribution of each training phase, we conduct an ablation study on the three stages of HeartMuLa: Pretraining, Supervised Fine-Tuning (SFT), and Direct Preference Optimization (DPO). The results are summarized in Table 15.

- Table 12: Subjective Evaluation Results (Trimmed MOS ± 95% CI) Across Different Dimensions Model Musicality Harmony Structure Fidelity Creativity Memorability Text Align. Suno-v4.5 78.10 ± 3.12 75.14 ± 4.16 78.80 ± 3.19 79.14 ± 2.65 71.26 ± 3.86 72.95 ± 3.81 77.17 ± 3.42

ACE-Step 67.42 ± 4.40 67.16 ± 5.10 69.07 ± 4.31 71.40 ± 3.76 62.04 ± 4.91 61.60 ± 5.21 67.94 ± 4.44 DiffRhythm 2 57.99 ± 5.38 57.13 ± 5.36 61.63 ± 4.62 62.58 ± 3.82 54.77 ± 5.06 53.15 ± 5.27 61.13 ± 4.45 YuE 56.95 ± 4.88 54.46 ± 5.37 60.07 ± 4.46 60.81 ± 3.99 55.68 ± 4.69 56.29 ± 4.80 61.25 ± 4.15 HeartMuLa (Ours) 69.55 ± 4.60 71.06 ± 4.57 73.44 ± 3.99 73.18 ± 3.74 66.73 ± 4.52 65.06 ± 4.93 70.51 ± 4.09

- Table 13: Overall MOS Ranking Across All Subjective Metrics with 95% Confidence Interval Rank Model Overall MOS 95% CI

- 1 Suno-v4.5 76.08 ±1.33
- 2 HeartMuLa (Ours) 69.93 ±1.66
- 3 ACE-Step 66.66 ±1.76
- 4 DiffRhythm2 58.33 ±1.86
- 5 YuE 57.93 ±1.76

The Effect of DPO on Various Dimensions. The experimental results shown in Table 16 demonstrate that Direct Preference Optimization (DPO) significantly enhances model performance across targeted dimensions, with each specialized preference dataset effectively optimizing its corresponding metric. Specifically, PER-DPO achieves a substantial reduction in Phoneme Error Rate (PER) to 0.0683, while Muq-DPO markedly improves semantic alignment, reaching a peak Tag-Sim score of 0.2839. Furthermore, the Audiobox & Songeval-DPO model shows the most comprehensive gains in objective audio quality and subjective song evaluation, yielding the highest average score of 4.448. These findings validate the efficacy of dimension-specific DPO in refining music generation and the model we ultimately adopted was obtained by linearly merging the models trained on the three dimensions mentioned above.

#### 3.5 Inference Acceleration

Long-form music generation poses significant challenges for efficient inference, as it requires modeling long-range musical structure while preserving fine-grained acoustic details over thousands of autoregressive steps. Under such conditions, inference latency is dominated mainly by repeated attention computation, Python-side scheduling overhead, and excessive GPU kernel launches. To address these issues, we introduce a set of collaborative system-level optimizations for HeartMuLa, targeting the batch size = 1 regime that is most relevant for interactive and streaming music generation. We employ a combination of KV-cache alignment, FlashAttention, and CUDA Graph to achieve efficient single-sample inference for long-form music generation, focusing on practical latency reduction and maintaining musical quality.

As summarized in Table 17, the proposed optimizations jointly reduce end-to-end generation time from 398.3s in the baseline configuration to 73.4s, achieving a 5.4× overall speedup. At the same time, the total number of GPU kernel launches is reduced from 1,561,161 to 979,149, representing a substantial decrease that directly alleviates Python-side dispatch overhead and GPU launch latency, which are critical bottlenecks in autoregressive decoding. The observed kernel launch reduction is slightly sub-linear relative to the latency reduction due to remaining Python-side scalar operations and dynamic control flow that limit the full reuse of kernels.

Table 14: Ablation study (English) of different training stages.

AudioBox ↑ SongEval ↑ Align ↑ Intel ↓ CE CU PC PQ Coh. Mus. Mem. Cla. Nat. Avg. Tag-Sim PER

Stage

HeartMuLa (Pretrain) 7.15 7.61 5.68 7.86 4.16 3.95 4.12 4.12 4.04 4.03 0.2377 0.1830 HeartMuLa (SFT) 7.45 7.78 6.04 8.07 4.52 4.36 4.54 4.38 4.25 4.41 0.2499 0.1005 HeartMuLa (DPO) 7.58 7.81 5.94 8.10 4.53 4.40 4.56 4.37 4.27 4.43 0.2573 0.0687

Table 15: Ablation study (Chinese) of different training stages.

AudioBox ↑ SongEval ↑ Align ↑ Intel ↓ CE CU PC PQ Coh. Mus. Mem. Cla. Nat. Avg. Tag-Sim PER

Stage

HeartMuLa (Pretrain) 7.19 7.62 5.77 7.89 4.24 4.03 4.19 4.09 3.96 4.10 0.2311 0.1174 HeartMuLa (SFT) 7.51 7.79 5.94 8.18 4.56 4.44 4.55 4.43 4.32 4.46 0.2442 0.1067 HeartMuLa (DPO) 7.49 7.78 5.97 8.18 4.52 4.37 4.50 4.38 4.26 4.40 0.2451 0.0806

- Table 16: DPO results trained on three preference datasets including Muq-similarity-based, PERbased and Audiobox&Songeval-based datasets.

AudioBox ↑ SongEval ↑ Align ↑ Intel ↓ CE CU PC PQ Coh. Mus. Mem. Cla. Nat. Avg. Tag-Sim PER

Model

Base (SFT) 7.45 7.78 6.04 8.07 4.52 4.36 4.54 4.38 4.25 4.41 0.2499 0.1005 PER-DPO 7.57 7.83 5.92 8.14 4.52 4.36 4.55 4.37 4.27 4.41 0.2538 0.0683 Muq-DPO 7.50 7.78 5.91 8.07 4.52 4.36 4.54 4.36 4.25 4.41 0.2839 0.0760 Aud. & Song.-DPO 7.57 7.84 6.00 8.15 4.55 4.40 4.59 4.40 4.29 4.45 0.2636 0.0820

These gains are enabled by carefully co-designing the inference pipeline around HeartMuLa’s cascaded decoding architecture, where a Global Transformer and a Local Transformer advance autoregressively in a tightly coupled manner Yang et al. [2023b]. While this architecture improves modeling capacity and temporal coherence, it also introduces non-trivial challenges for inference efficiency, including strict state synchronization, frequent cache updates, and complex control flow across modules.

All optimizations described in this section are verified to preserve musical quality. Objective evaluation results under different inference settings are summarized in Table 18, demonstrating that accelerated inference does not degrade generation quality. Minor fluctuations across configurations are within expected variance and do not affect perceptual fidelity. Streaming inference achieves the lowest end-to-end latency and the best intelligibility, reflecting the effectiveness of KV-cache alignment and tensor-only control flow in maintaining temporal coherence.

#### 3.5.1 Overview of the Inference Pipeline

HeartMuLa performs autoregressive generation over discrete acoustic tokens. At each decoding step, the model predicts one frame consisting of K RVQ code indices, conditioned on all previously generated frames and optional multimodal inputs such as lyrics or genre prompts. Inference proceeds in two stages: context prefill, which initializes hidden states and KV-caches using the prompt sequence, and autoregressive decoding, which incrementally generates new acoustic frames.

During autoregressive decoding, the model produces one acoustic frame per step, making the pipeline naturally compatible with online and streaming generation. Intermediate tokens or decoded audio segments can be emitted immediately once available. In this work, we focus on batch size = 1, which reflects practical interactive usage scenarios where end-to-end latency, responsiveness, and temporal stability are more important than throughput-oriented efficiency.

#### 3.5.2 Collaborative Optimizations: KV-Cache, FlashAttention, and CUDA Graph

The primary computational bottleneck in long-form music generation arises from repeated selfattention over an ever-growing token history. In cascaded decoding, this challenge is further exacerbated by the presence of multiple decoding modules whose internal states must remain strictly synchronized.

In the baseline implementation, frequent Python-side scalar operations (e.g., .item()) and dynamic control flow break computation graphs and lead to misaligned KV-cache updates. As a result, KV-cache reuse becomes a state consistency problem rather than a simple memory optimization, contributing directly to the high latency and excessive kernel launches observed in the baseline setting (Tables 17).

KV-Cache Alignment. To resolve these issues, we enforce strict alignment between token indices, positional encodings, attention masks, and KV-cache write locations. At decoding step t, the KVcache for each layer contains key-value pairs corresponding to tokens 0,...,t − 1, and the newly generated token is appended at position t:

Kt(+1l) = Append(Kt(l),kt(l)), Vt(+1l) = Append(Vt(l),vt(l)). (14)

All cache updates are implemented using pure tensor operations, completely avoiding Python-side indexing or scalar extraction. Positional encodings are derived from a global step counter, and separate positional buckets are used across modules when execution order differs. This design guarantees consistency among attention masks, cache contents, and positional information, enabling effective KV-cache reuse and the latency reduction observed when KV-cache is enabled.

FlashAttention. To further accelerate self-attention, we integrate FlashAttention Dao et al. [2022], Dao [2024] with KV-cache reuse. Only the valid prefix of cached keys and values is exposed to the attention kernel, and causal masks are precomputed outside the decoding loop. FlashAttention significantly reduces the number of kernel launches while maintaining stable speedups as sequence length increases, making it particularly effective for long-form music generation.

CUDA Graph. CUDA Graph is employed to reduce Python-side overhead by capturing and replaying static GPU execution graphs. We carefully separate static and dynamic components of the inference pipeline: the core transformer forward passes are captured within the graph, while dynamic elements such as sampled tokens, updated positions, and control decisions are injected externally before replay. This separation preserves shape invariance and correctness while enabling the substantial latency reduction reported in Table 17.

Engineering Considerations. Several practical constraints limit the applicability of whole-graph compilation in cascaded decoding:

- 1. Warm-up executions prior to graph capture may implicitly populate KV-caches, causing positional misalignment during replay.
- 2. Sampling operations (e.g., multinomial sampling) inside captured graphs freeze the random state, resulting in identical outputs across runs; therefore, sampling must remain outside the graph.
- 3. Sharing positional buckets between backbone and decoder modules was observed to introduce subtle phase and timbre drift, motivating the use of independent buckets.
- 4. Minor changes in prompt length or codebook configuration frequently invalidate previously captured graphs, necessitating re-capture.
- 5. Python-side scalar operations such as .item() consistently break graph capture and introduce cache inconsistencies.
- 6. Dynamic control flow arising from special decoding modes (e.g., temperature = 0 or variable CFG scales) cannot be safely included in captured graphs.

Collectively, these observations motivate a strict tensor-only control flow and a clear separation between static and dynamic elements within the inference pipeline.

#### 3.5.3 Batching and Streaming Considerations

The proposed inference pipeline supports both offline batch generation and online streaming generation. The optimizations described above are primarily designed for the batch size = 1 setting, which dominates interactive and streaming use cases. In this regime, end-to-end latency and temporal stability are more critical than raw throughput.

As batch size increases, optimizations such as KV-cache reuse and FlashAttention remain effective, but their relative benefits diminish due to increased inter-module synchronization overhead.

In contrast, the inference pipeline naturally supports streaming generation. Autoregressive decoding proceeds frame by frame, allowing intermediate acoustic tokens or decoded audio chunks to be

emitted immediately. KV-cache alignment and tensor-only control flow ensure temporal consistency across streamed outputs, while CUDA Graph capture is applied only to static subgraphs that do not interfere with online token emission. As a result, HeartMuLa achieves stable low-latency streaming inference without compromising output quality.

#### 3.5.4 Experimental Evaluation

To verify that inference acceleration does not degrade musical quality, we evaluate HeartMuLa under different inference configurations on our Benchmark. Table 18 reports objective metrics including AudioBox, SongEval, alignment (Align), and intelligibility (PER). Across base, optimized, streaming, and batch inference settings, the results remain highly consistent, indicating that the proposed optimizations preserve melody, harmony, rhythmic structure, and perceptual quality. Streaming inference achieves the best intelligibility (PER=0.0778) and the lowest end-to-end latency (67.95,s), demonstrating that real-time generation is feasible without significant quality degradation.

- Table 17: End-to-end inference latency and GPU kernel launches under different HeartMuLa inference configurations.

Inference Setting Avg. Latency (s) ↓ Min (s) ↓ Max (s) ↓ Kernel Launches ↓

Baseline 398.27 133.99 748.88 1,561,161 KV-cache 132.99 85.04 171.95 987,887 CUDA Graph 78.32 48.03 105.74 1,043,455 FlashAttention 76.00 48.16 99.56 979,149 All optimizations 73.41 46.24 98.88 979,149 Streaming 67.95 45.27 89.07 979,149

- Batch (BS=1) 73.41 46.24 98.88 979,149
- Batch (BS=2) 116.02 62.00 124.01 942,473 Batch (BS=4) 210.00 82.50 210.07 940,980 Batch (BS=8) 390.06 248.75 423.50 939,510

- Table 18: HeartMuLa generation quality and AudioBox/SongEval metrics under different inference configurations.

AudioBox ↑ SongEval ↑ Align ↑ Intel ↓ CE CU PC PQ Coh. Mus. Mem. Cla. Nat. Avg. Tag-Sim PER

Inference Setting

Baseline 7.478 7.789 5.901 8.167 4.574 4.445 4.579 4.441 4.348 4.477 0.2337 0.1008 KV-cache 7.620 7.843 6.056 8.269 4.625 4.507 4.645 4.512 4.420 4.542 0.2440 0.0868 CUDA Graph 7.576 7.822 5.932 8.211 4.550 4.429 4.561 4.431 4.334 4.461 0.2258 0.1235 FlashAttention 7.557 7.812 6.084 8.209 4.610 4.491 4.601 4.482 4.410 4.519 0.2422 0.1053 All optimizations 7.585 7.818 5.982 8.224 4.615 4.481 4.602 4.490 4.400 4.518 0.2520 0.1351 Streaming 7.606 7.830 6.111 8.255 4.618 4.486 4.609 4.481 4.392 4.517 0.2273 0.0778

### 4 HeartCLAP

The proposed HeartCLAP model architecture is designed to bridge the gap between musical audio and diverse textual descriptions. It comprises two primary components: a text encoder ET and a music encoder EM. Both backbones are initialized with pre-trained weights from MuQ-MuLan Zhu et al. [2025] to leverage robust prior musical knowledge. The high-level features extracted by both encoders are projected into a 1024-dimensional shared embedding space through separate linear projection layers.

To achieve cross-modal alignment, we employ Contrastive Learning using the InfoNCE loss Oord et al. [2018]:

N

exp(s(mi,ti)/τ) exp(s(mi,ti)/τ) + Nj=1,j̸=i exp(s(mi,tj)/τ) −

LCLAP = −

log

i=1

(15)

N

exp(s(mi,ti)/τ) exp(s(mi,ti)/τ) + Nj=1,j̸=i exp(s(mj,ti)/τ)

log

,

i=1

where s(·,·) denotes cosine similarity, τ is a learnable temperature parameter, and mi,ti denotes the embedding extracted from the i-th music-text pair via EM and ET respectively. This objective ensures that positive music-text pairs (mi,ti) are pulled together while mismatched pairs are pushed apart in the latent space.

#### 4.1 Training Details

To foster a comprehensive understanding of music, our training data encompasses a wide spectrum of attributes: genre, mood, instrumentation, singer timbre, gender, scene, and topic. We utilize a multi-format annotation strategy where each audio segment is paired with both tag-based descriptions separated by commas like "mood: soft, warm, genre: pop, ...", "soft, warm, pop, ...", and natural language sentences like "The music features a male gender, with hiphop genre, the instrument is drum, suitable for dance and workout scene". During training, one description format is randomly sampled per iteration to improve the model’s linguistic flexibility. To enhance robustness against incomplete user prompts-a common scenario in music generation-we implement two masking strategies:

- • Attribute-level masking: Entire attribute categories (e.g., all mood tags) are dropped with a probability pa.
- • Tag-level masking: Within a category containing multiple tags (e.g., several instruments), individual tags are randomly omitted with a probability pt.

The model was trained on 1 million music clips and corresponding description pairs using 8 NVIDIA A100 GPUs. We use the AdamW optimizer with a learning rate of 1 × 10−4 and a weight decay of 0.1. We employed a global batch size of 1024 and an epoch of 50.

#### 4.2 Experiment Results

We evaluated our HeartCLAP along with the baselines Laion-CLAP Elizalde et al. [2023] and MuQMuLan Zhu et al. [2025] on the WikiMT-X Wu et al. [2025] benchmark using music-description pairs for the Music-Text Retrieval task, as shown in Table 19. We employed Recall@K (R@K) and mAP@10 as evaluation metrics. R@K is 1 if the target-value item occurs in the first K retrieved items, and 0 otherwise. Meanwhile, mAP@10 calculates the average precision across all queries within the top 10 retrieved results.

Table 19: Comparative Evaluation of CLAP Models on WiKiMT-X Benchmark.

Text-to-Music Music-to-Text R@1 ↑ R@10 ↑ mAP@10 ↑ R@1 ↑ R@10 ↑ mAP@10 ↑

Model

Laion-CLAP 0.71 6.92 2.17 1.01 5.39 2.08 MuQ-MuLan 2.24 12.11 4.70 1.62 9.47 3.36 HeartCLAP 4.37 16.80 7.59 2.85 14.35 5.51

The experimental results demonstrate that HeartCLAP significantly outperforms both Laion-CLAP and the backbone MuQ-MuLan across all evaluation metrics. This substantial margin in R@K and mAP@10 suggests that our framework effectively facilitates a more precise alignment between music and multi-faceted textual descriptions. By leveraging pre-trained musical priors and incorporating diverse annotation formats-ranging from structured tags to natural language, HeartCLAP exhibits superior generalization capabilities in capturing both high-level semantic themes and fine-grained

acoustic attributes. The consistent performance gain in both Text-to-Music and Music-to-Text retrieval tasks indicates that the learned representations are mutually discriminative and robust for bi-directional retrieval scenarios.

### 5 HeartTranscriptor

Mainstream Automatic Speech Recognition (ASR) models, such as Whisper Radford et al. [2023], perform well on standard speech tasks. However, these systems struggle with music-specific ASR. Interference from complex instrumental accompaniments often degrades performance. Furthermore, the phonetic nuances of singing vocals differ significantly from spoken language. To address this, we introduce HeartTranscriptor, a framework designed for robust lyric recognition. Our approach involves fine-tuning Whisper on a high-quality dataset of musical audio. We adapt the model by aligning predicted probability distributions with ground truth labels. At each time step t, the decoder generates a probability distribution across the vocabulary. We then optimize model parameters using a cross-entropy loss function. The loss function LCE is defined as:

T

LCE = −

t=1

log P(yt|y<t,x;θ) (16)

where x represents the input audio features, yt is the target lyric token at time step t, y<t denotes the preceding ground truth tokens, and θ refers to the learnable model parameters. During training, we employ teacher forcing by conditioning on ground truth history. This objective maximizes the likelihood of generating the correct lyric sequence given the audio input.

#### 5.1 Dataset Construction Pipeline

To construct the dataset, we first selected multilingual songs, including Chinese, English, Korean, Japanese, Spanish, and various other languages. We then employed the Demucs model Rouard et al. [2023] to decouple the mixed audio tracks into isolated vocal and accompaniment tracks to mitigate interference. Subsequently, the Whisper-Medium model was utilized to perform lyric recognition on isolated vocal tracks. We then calculate the word error rates (WER) or character error rates (CER) over the transcribed results from Whisper-Medium. Crucially, to ensure data quality, we implemented a hierarchical filtering strategy: retaining Chinese and English samples with an error rate below 0.7, and other languages below 0.8. In summary, this pipeline provides a high signal-to-noise ratio training corpus. Fine-tuning our model on this curated data markedly improved overall recognition performance.

#### 5.2 Training Details

Based on the pipeline described above, we constructed a large-scale refined dataset of approximately 100,000 verified songs. These were further segmented into standardized 30-second audio slices, resulting in a cumulative duration of approximately 7,000 hours. This segment length aligns with the input window constraints of the Whisper model. By adhering to this duration, the model fully leverages available contextual features during training.

For the model training phase, we selected Whisper-Medium as the base model and employed a full fine-tuning strategy to effectively capture the complex acoustic textures present in music audio. The computational environment consisted of 8 NVIDIA A100 (80G) GPUs. We set the learning rate to 1 × 10−5, coupled with a 1000-step warmup strategy to prevent gradient instability during early training. To mitigate overfitting, the weight decay was set to 0.01, and the gradient clipping threshold was capped at 1.0. The training batch size was set to 16, with gradient accumulation steps utilized to increase the effective batch size to 16 × 4 × 8. Through this refined experimental configuration, we successfully developed a high-performance Music Automatic Speech Recognition model.

#### 5.3 Experiment

To evaluate the performance of our fine-tuned model, we selected two distinct benchmarks. The first is SSLD-200 from SongPrep Tan et al. [2025], which consists of 100 full-length songs in both

Chinese and English. The second is our internal HeartBeats-ASR-Bench, curated to test fine-grained capabilities with 200 audio slices (under 30 seconds) for each of the five target languages: English, Chinese, Korean, Japanese, and Spanish. Additionally, we utilized the Demucs model to process both datasets, isolating clean vocals from the background accompaniment. To quantitatively assess the accuracy of recognition, we adopted Word Error Rate (WER) for English and Spanish, and Character Error Rate (CER) for Chinese, Japanese, and Korean.

Table 20: Comparison of WER and CER across different models and datasets.

SSLD-200 HeartBeats-ASR-Bench Model 100en 100zh 200en 200zh 200ko 200ja 200es Whisper-Small 0.6802 0.5605 0.4172 0.3473 0.4612 0.5556 0.4391 Whisper-Medium 0.4695 0.4744 0.2206 0.3045 0.2789 0.3405 0.3520 Whisper-Turbo 0.4737 0.3726 0.3397 0.3020 0.2672 0.2630 0.3328 Whisper-Large-V3 0.3981 0.3724 0.2139 0.3316 0.1646 0.2146 0.2798 SongPrep 0.3460 0.1884 0.2075 0.1670 - - FireRedASR-aed - - 0.5176 0.3471 - - FireRedASR-llm - - 0.3266 0.1679 - - Qwen3-Omni-30B-A3B-Captioner - - 0.2610 0.2049 0.2394 0.2804 0.2320 HeartTranscriptor 0.2816 0.1438 0.1873 0.1077 0.1042 0.1801 0.2151

As shown in Table 20, HeartTranscriptor consistently achieves the lowest error rates across all datasets and languages, significantly outperforming both the baselines and other domain-specific models. It demonstrates exceptional robustness, particularly in processing full-length and multilingual songs. In comparison, the baseline Whisper models generally exhibit mediocre performance, with only Whisper-Large-V3 showing relatively better results among the vanilla versions. Regarding other specialized models, while SongPrep achieves competitive error rates, its applicability is strictly limited to Chinese and English. Furthermore, the FireRedASR series demonstrates average performance and is subject to strict input duration constraints (60s for aed, 30s for llm). Similarly, even after filtering out duration-induced hallucinations (error rate > 3) from Qwen3-Omni-30B-A3B-Captioner, it still underperforms compared to our proposed method.

### 6 Overall Training Datasets

#### 6.1 Composition of the Dataset

Our dataset consists of three parts: music with lyrics, instrumental music, and TTS datasets (LJSpeech Ito and Johnson [2017], LibriTTS Zen et al. [2019] and GigaSpeech Chen et al. [2021]). To account for musicality, we employed Audiobox-Aesthetic Tjandra et al. [2025] and SongEval Yao et al. [2025] to filter the dataset. To ensure the alignment between lyrics and music, we utilized HeartTranscriptor for Automatic Speech Recognition (ASR) to eliminate mismatched music-lyrics pairs. Additionally, we trained a baseline model using the entire dataset (e.g., a music model capable of generating coherent sound) and filtered the training data based on its Perplexity (PPL). Ultimately, we retained approximately 100,000 hours of high-quality training data.

#### 6.2 Music Style

For the music style in HeartMuLa, user inputs can vary widely, such as tags (e.g., K-pop), descriptions (e.g., "A lively music"), and more. Imagine you are a user: when selecting a style, the simplest and most direct approach would be to click a tag, forming your input. Furthermore, tags and descriptions can be interchanged via a Large Language Model (LLM). Thus, we have chosen a tag-based style approach. By analyzing user demand across various networks, we classified the tags into categories such as gender, genre, instrument, mood, scene, singer timbre, topic, and region. Based on these categories, we designed prompts that enable our fine-tuned Qwen2.5-Omni Xu et al. [2025] to understand the music style. The prompts are as follows:

#### Prompt Detail

You are a music researcher, music analysis assistant. Below I will give you a music, Please analyze this music. gender of the singer, genre of the music (ej, rock, hiphop), instrument used, mood of the music, scene(Applicable Scenarios for Music, excluding KTV,concert, party),

singer’s timbre, topic(Themes of Music), region, each result is retained within 1-3 words. must be in English. (genre, mood, scene, singer timbre, topic, region) these keys need contain **multiple** values.). You return the following format and explain why these labels were chosen.

{

"gender": "", "genre": [], "instrument": [], "mood": [], "scene": [], "singer_timbre": [], "topic": [], "region": []

}

**Reasoning:**

#### 6.3 Music Structure

Through extensive experiments, we found that performing structural analysis of music during the pre-training phase and incorporating it into the lyrics significantly enhances the structural coherence of the music during inference. We employed SongFormer Hao et al. [2025] to annotate the structure of the music. An example of a structurally annotated music piece is shown in the table below:

#### Music Structure

[Chorus] My chest is vibrating like a V8 motor Spinning faster, getting hotter Every beat is a piston stroke Leaving the past up in smoke

[Verse] Cold air intake, breathing deep Awake right now, no time for sleep The valves are open, the rhythm is true Driving this chassis straight to you

[Prechorus] Throttle wide open, floor to the mat No looking back, no time for that Ignition sparks the electric blue

[Chorus] My chest is vibrating like a V8 motor Spinning faster, getting hotter Every beat is a piston stroke Leaving the past up in smoke

[Bridge] Overheating in the red zone But I can’t stop this skin and bone From racing down this road alone

[Outro] Cut the engine. Fade to black.

#### 6.4 Data-Finegrained Style Annotation

To enable precise user control over intra-song generation, we propose a Fine-grained Style Annotation Pipeline. Building upon the structured lyrics generated in 6.3 as input, this pipeline employs our fine-tuned Qwen-2.5-omni multimodal large language model as its core analytical engine. During the annotation process, adhering to the principle of Audio Grounding-where raw audio signals serve

as the ultimate ground truth-the model integrates textual structural information to perform multidimensional feature deconstruction on each Structure Unit. Specifically, the analysis encompasses three orthogonal dimensions: Dynamics & Energy, Vocal & Technique, and Style & Vibe. The detailed fine-grained style annotation is illustrated in Box 6.4.

#### Music Finegrained Style Annotation

[Intro] [Subtle electronic pulse, atmospheric build, anticipatory mood, understated energy] Green leaves heavy on the bough The secret starts right here and now

[Verse] [Steady rhythmic foundation, introspective vocal delivery, narrative progression, moderate

intensity] No colors bursting in the spring air No petals dancing in the wind without a care People walk by and they don’t see the show But deep inside the skin,the sweetness starts to grow A mystery wrapped up in a humble disguise Hidden away from all the prying eyes

[Prechorus] [Gradual crescendo, rising vocal urgency, building anticipation, subtle harmonic shift] It happens in the dark,unseen Beneath the canopy of green Waiting for the perfect time to be

[Chorus] [Explosive melodic hook, triumphant vocal expression, high energy sustain, core thematic statement

] Oh,the flower blooms inside the fruit A hidden treasure,quiet and mute We don’t need the applause or the light To make something beautiful and bright Sweetness saved for those who know the truth The essence of an everlasting youth

[Bridge] [Melodic contrast, reflective vocal tone, shift in perspective, introspective pause] Let the others chase the butterfly Let them fade beneath the summer sky We hold our magic close to the core

[Outro] [Gradual fade, lingering melodic motif, sense of resolution, gentle conclusion] Open it up and see The secret of the fig tree

#### 6.5 HeartBeats-Benchmark

We developed HeartBeats-Benchmark using a "Human-in-the-Loop" curation strategy to rigorously evaluate music understanding. Drawing inspiration from the dataset construction protocols like YuE Yuan et al. [2025] and DiffRhythm Ning et al. [2025], we invited professional musicologists to design a comprehensive evaluation taxonomy. This benchmark aims to map acoustic signals to high-level semantics through a rigorously verified standard.

Evaluation Dimensions. To ensure a holistic assessment, the experts structured the analysis into three macro-perspectives covering six granular dimensions:

- • Acoustic Structure: This category covers Musical Style, delineating genre classification and stylistic roots. It also assesses Instrumentation, identifying rhythmic foundations and harmonic components within the arrangement.
- • Content Semantics: This aspect examines Vocal Texture, characterizing the singer’s timbre and acoustic properties. It further analyzes Narrative Theme, extracting subject matter and emotional focus from the lyric-melody interplay.

- • Contextual Atmosphere: We incorporate Emotional Valence to verify alignment between audio features and human affective cognition. Additionally, Usage Scenario categorizes the functional settings suitable for the track.

To enhance input diversity, we implemented a Random Dimension Dropout strategy. Specifically, for each data sample, we randomly mask two out of the six granular dimensions defined in our taxonomy. The tags from the remaining four dimensions are then extracted to the final input prompt sequence. To visualize this transformation, a representative sample of this expert-validated schema is visualized in Box 6.5, while Box 6.5 demonstrates the final inputs generated after applying the stochastic dropout strategy.

#### Box: Tag example

{

"Acoustic Structure": ["pop","Piano", "Strings", "acoustic guitar", "

synthesizer"], "Content Semantics": ["Sweet", "Love",] "Contextual Atmosphere": ["joyful","Dating"]

}

#### Box: Final Input Sequence (Post-Dropout)

[Pop, Piano, Strings, Acoustic Guitar, Synthesizer, Joyful, Sweet]

To ensure the professionalism and multilingual adaptability of the dataset, we established strict screening criteria for the lyrics section. The dataset includes 30 English, 20 Chinese, 10 Japanese, 10 Korean, and 10 Spanish songs. As demonstrated in Box 6.5, clear structural markers (e.g., [Intro], [Verse]) are embedded to segment the textual stream. Finally, the expert panel conducted a "Blind Validation" of all tag combinations, ensuring the dataset serves as an unbiased "Gold Standard" for model evaluation.

#### Box: Lyrics example

[Intro] [Verse] In the frozen North where the winter bites I hold a crimson seed through the endless nights It is hard as stone, preserved in the cold A memory of you that never grows old [Prechorus] You are down South where the spring rain falls Gathering the harvest by the garden walls Two different worlds, but the color is the same [Chorus] Red beans scattered across the miles One brings tears and the other brings smiles Yours are blooming in the warm embrace Mine are hidden in a secret place North and South, we plant the yearning deep Promises that we are sworn to keep

...

### 7 Related Works

#### 7.1 Audio Tokenizer

Discrete audio representations play a central role in audio language models, and existing approaches can be broadly categorized into semantic tokenizers and acoustic tokenizers, each exhibiting distinct trade-offs in representation capacity and modeling efficiency.

Semantic tokenizers typically rely on representations extracted from large-scale self-supervised learning (SSL) models, such as HuBERT Hsu et al. [2021] and WavLM Chen et al. [2022]. These models are trained to capture high-level linguistic and phonetic abstractions, making their hidden representations particularly amenable to discretization via clustering or vector quantization. Prior studies have shown that such semantic tokens exhibit strong compatibility with language modeling objectives, leading to improved modeling efficiency and generation stability in audio language models Borsos et al. [2023], Zeng et al. [2024], Du et al. [2024], Liu et al. [2024]. However, this advantage comes at the cost of substantial loss of fine-grained acoustic information. As a result, semantic token based systems typically require additional generative decoders, e.g. diffusion models Ho et al. [2020] or flow-matching frameworks Lipman et al. [2022] to reconstruct waveforms. These cascaded architectures significantly increase inference latency and system complexity, limiting their practicality for large-scale or real-time generation.

In contrast, acoustic tokenizers are derived from neural audio codec models that are explicitly optimized for waveform reconstruction. Such codecs generally consist of an encoder, a discrete quantizer, and a lightweight decoder, enabling efficient end-to-end audio reconstruction with low inference overhead. Representative models, including EnCodec Défossez et al. [2022], HiFi-Codec Yang et al. [2023a], and DAC Kumar et al. [2023], have demonstrated strong reconstruction fidelity across speech, music, and general audio domains. Owing to their rich acoustic expressiveness.

Among existing low-bitrate semantic-aware codecs, MimiCodec Défossez et al. [2024], MuCodec Xu et al. [2024] are most closely related to our work. However, MimiCodec focuses on speech data, and MuCodec has limited reconstruction performance. In contrast, our work targets a lowbitrate, semantic-rich, and high-fidelity music tokenizer. This design enables stronger alignment with language modeling objectives while retaining high-fidelity acoustic reconstruction.

#### 7.2 Music generation

Language Model Paradigm With the advent of Large Language Models (LLMs) and their demonstrated reasoning and scaling capabilities Team [2024], several works have adopted LLMs for end-to-end music generation Copet et al. [2023], Agostinelli et al. [2023]. A common framework involves encoding audio into discrete token sequences via Vector Quantized Variational Autoencoders (VQ-VAEs) Van Den Oord et al. [2017] or Residual Vector Quantization (RVQ) Zeghidour et al. [2022], which are subsequently modeled autoregressively.

Diffusion Models for music generation. Recently, diffusion models Ho et al. [2020], Lipman et al. [2022] have been introduced for their proficiency in modeling audio/music data Huang et al. [2023b,a].

Song Generation Song generation, which entails creating coherent vocals with instrumental accompaniment, presents unique challenges. Pioneering work Jukebox Dhariwal et al. [2020] uses a hierarchical VQ-VAE and transformer to model long sequences. Subsequent efforts like SongCreator Lei et al. [2024] and MusiCot Lam et al. [2025] employ dual-sequence models or coarse-grained style conditioning to improve vocal-accompaniment relationships and overall structure, though they remain susceptible to interference between tracks and limited token vocabulary. Alternative approaches, such as MelodyLM Li et al. [2024], adopt multi-stage pipelines to generate vocals and accompaniment separately. Recent models like YuE Yuan et al. [2025] and SongGen Liu et al. [2025] operate on interleaved vocal and accompaniment token sequences. On the diffusion front, DiffRhythm Ning et al. [2025] generates full songs in a continuous space. While industrial systems (e.g., Suno, Udio) show impressive results, their technical details are not public.

### 8 Conclusion

We presented HeartMuLa, an open-sourced family of music foundation models that unifies music-text alignment, music tokenization, lyrics recognition, and controllable music generation. By introducing a hierarchical audio language modeling framework built on ultra-low-frame-rate music tokens, our approach enables efficient and coherent long-form music generation with fine-grained controllability. Extensive experiments demonstrate consistent improvements over existing codecs and song generation baselines in both reconstruction quality, generation performance and modeling efficiency. We hope HeartMuLa serves as a strong foundation for future research in music understanding and generation, and facilitates broader applications in creative music production.

### 9 Ethics and Responsibility

HeartMuLa family is designed as an open-sourced initiative to advance music intelligence research. Our model operates on a transformative paradigm; it learns statistical acoustic representations to generate novel musical compositions rather than reproducing copyrighted source material. By utilizing an exceptionally diverse training dataset, explicitly enriched with culturally and linguistically diverse music content (see Section 6), our model can innovate and create within various musical styles effectively, thereby contributing to human musical artistry and cultural heritage.

However, responsible innovation requires more than just advanced architecture. We echo the advocacy of Ma et al. [2024] for labeling AI-generated/assisted content, thereby ensuring accountability for both creators and listeners. Notably, we implement a watermarking model to ensure audio security and facilitate content authentication. We believe this practice is essential for establishing a standardized responsibility protocol in the rapidly evolving field of AI music generation.

### 10 Contributions Co-First Authors

- • Yuxin Xie, Peking University
- • Yuguo Yin, Peking University
- • Zheyu Wang, Scale Global, Ario
- • Xiaoyu Yi, Peking University
- • Gongxi Zhu, Scale Global, Ario
- • Xiaolong Weng,Scale Global, Ario
- • Zihan Xiong, Scale Global, Ario
- • Yingzhe Ma, Scale Global, Ario
- • Dading Chong, Peking University
- • Dongchao Yang, The Chinese University of Hong Kong

#### Core Contributors

- • Jinliang Liu, Scale Global, Ario
- • Zihang Huang, Scale Global, Ario
- • Jinghan Ru, Peking University
- • Rongjie Huang, The Chinese University of Hong Kong

#### Contributors

- • Haoran Wan, Peixu Wang, Kuoxi Yu. Scale Global, Ario
- • Helin Wang, Liming Liang, Xianwei Zhuang. Peking University
- • Yuanyuan Wang, Haohan Guo, Dingdong Wang The Chinese University of Hong Kong
- • Junjie Cao, Zeqian Ju, Songxiang Liu, Yuewen Cao. Independent Researcher

#### Corresponding Authors

- • Yuexian Zou, Peking University
- • Heming Weng, Scale Global, Ario

#### Technical Leader

• Dongchao Yang, The Chinese University of Hong Kong

### References

Andrea Agostinelli, Timo I Denk, Zalán Borsos, Jesse Engel, Mauro Verzetti, Antoine Caillon, Qingqing Huang, Aren Jansen, Adam Roberts, Marco Tagliasacchi, et al. Musiclm: Generating music from text. arXiv preprint arXiv:2301.11325, 2023.

Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, et al. Audiolm: a language modeling approach to audio generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.

Guoguo Chen, Shuzhou Chai, Guanbo Wang, Jiayu Du, Wei Qiang Zhang, Chao Weng, Dan Su, Daniel Povey, Jan Trmal, Junbo Zhang, et al. GigaSpeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio. In INTERSPEECH, pages 4376–4380, 2021.

Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, and Furu Wei. Wavlm: Large-scale self-supervised pretraining for full stack speech processing. IEEE J. Sel. Top. Signal Process., 16(6):1505–1518, 2022. doi: 10.1109/JSTSP.2022.3188113. URL https://doi.org/10.1109/JSTSP.2022.3188113.

Chung-Cheng Chiu, James Qin, Yu Zhang, Jiahui Yu, and Yonghui Wu. Self-supervised learning with random-projection quantizer for speech recognition. In International Conference on Machine Learning, pages 3915–3924. PMLR, 2022.

Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, and Alexandre Défossez. Simple and controllable music generation. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 94b472a1842cd7c56dcb125fb2765fbd-Abstract-Conference.html.

Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Alexandre Défossez, Nicolas Usunier, Léon Bottou, and Francis Bach. Demucs: Deep extractor for music sources with extra unlabeled data remixed. arXiv preprint arXiv:1909.01174, 2019.

Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. High fidelity neural audio compression. arXiv preprint arXiv:2210.13438, 2022.

Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour. Moshi: a speech-text foundation model for real-time dialogue. arXiv preprint arXiv:2410.00037, 2024.

Brecht Desplanques, Jenthe Thienpondt, and Kris Demuynck. Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification. arXiv preprint arXiv:2005.07143, 2020.

Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, and Ilya Sutskever. Jukebox: A generative model for music. arXiv preprint arXiv:2005.00341, 2020.

Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, et al. Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. arXiv preprint arXiv:2407.05407, 2024.

Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang. Clap learning audio concepts from natural language supervision. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

Junmin Gong, Sean Zhao, Sen Wang, Shengyuan Xu, and Joe Guo. Ace-step: A step towards music generation foundation model. arXiv preprint arXiv:2506.00045, 2025.

Chunbo Hao, Ruibin Yuan, Jixun Yao, Qixin Deng, Xinyi Bai, Wei Xue, and Lei Xie. Songformer: Scaling music structure analysis with heterogeneous supervision, 2025. URL https://arxiv. org/abs/2510.02797.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3451–3460, 2021.

Jiawei Huang, Yi Ren, Rongjie Huang, Dongchao Yang, Zhenhui Ye, Chen Zhang, Jinglin Liu, Xiang Yin, Zejun Ma, and Zhou Zhao. Make-an-audio 2: Temporal-enhanced text-to-audio generation. arXiv preprint arXiv:2305.18474, 2023a.

Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, and Zhou Zhao. Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models. In International Conference on Machine Learning, pages 13916–13932. PMLR, 2023b.

Keith Ito and Linda Johnson. The LJ speech dataset, 2017. Yuepeng Jiang, Huakang Chen, Ziqian Ning, Jixun Yao, Zerui Han, Di Wu, Meng Meng, Jian Luan,

Zhonghua Fu, and Lei Xie. Diffrhythm 2: Efficient and high fidelity song generation via block flow matching. arXiv preprint arXiv:2510.22950, 2025.

Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. Advances in neural information processing systems, 33:17022–17033, 2020.

Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, and Kundan Kumar. Highfidelity audio compression with improved RVQGAN. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=qjnl1QUnFA.

Max WY Lam, Yijin Xing, Weiya You, Jingcheng Wu, Zongyu Yin, Fuqiang Jiang, Hangyu Liu, Feng Liu, Xingda Li, Wei-Tsung Lu, et al. Analyzable chain-of-musical-thought prompting for high-fidelity music generation. arXiv preprint arXiv:2503.19611, 2025.

Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, et al. Voicebox: Text-guided multilingual universal speech generation at scale. Advances in neural information processing systems, 36:14005–14034, 2023.

Shun Lei, Yixuan Zhou, Boshi Tang, Max WY Lam, Hangyu Liu, Jingcheng Wu, Shiyin Kang, Zhiyong Wu, Helen Meng, et al. Songcreator: Lyrics-based universal song generation. Advances in Neural Information Processing Systems, 37:80107–80140, 2024.

Shun Lei, Yaoxun Xu, Zhiwei Lin, Huaicheng Zhang, Wei Tan, Hangting Chen, Jianwei Yu, Yixuan Zhang, Chenyu Yang, Haina Zhu, Shuai Wang, Zhiyong Wu, and Dong Yu. Levo: High-quality song generation with multi-preference alignment. CoRR, abs/2506.07520, 2025. doi: 10.48550/ ARXIV.2506.07520. URL https://doi.org/10.48550/arXiv.2506.07520.

Ruiqi Li, Zhiqing Hong, Yongqi Wang, Lichao Zhang, Rongjie Huang, Siqi Zheng, and Zhou Zhao. Accompanied singing voice synthesis with fully text-controlled melody. arXiv preprint arXiv:2407.02049, 2024.

Rui Lin, Zhiyue Wu, Jiahe Le, Kangdi Wang, Weixiong Chen, Junyu Dai, and Tao Jiang. Duotok: Dual-track semantic music tokenizer for vocal-accompaniment generation, 2025. URL https://arxiv.org/abs/2511.20224.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Haohe Liu, Xuenan Xu, Yi Yuan, Mengyue Wu, Wenwu Wang, and Mark D. Plumbley. Semanticodec: An ultra low bitrate semantic audio codec for general sound. IEEE J. Sel. Top. Signal Process., 18 (8):1448–1461, 2024. doi: 10.1109/JSTSP.2024.3506286. URL https://doi.org/10.1109/ JSTSP.2024.3506286.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=XVjTT1nw5z.

Zihan Liu, Shuangrui Ding, Zhixiong Zhang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Songgen: A single stage auto-regressive transformer for text-to-song generation, 2025. URL https://arxiv.org/abs/2502.13128.

Yinghao Ma, Anders Øland, Anton Ragni, Bleiz MacSen Del Sette, Charalampos Saitis, Chris Donahue, Chenghua Lin, Christos Plachouras, Emmanouil Benetos, Elona Shatri, Fabio Morreale, Ge Zhang, György Fazekas, Gus Xia, Huan Zhang, Ilaria Manco, Jiawen Huang, Julien Guinot, Liwei Lin, Luca Marinelli, Max W. Y. Lam, Megha Sharma, Qiuqiang Kong, Roger B. Dannenberg, Ruibin Yuan, Shangda Wu, Shih-Lun Wu, Shuqi Dai, Shun Lei, Shiyin Kang, Simon Dixon, Wenhu Chen, Wenhao Huang, Xingjian Du, Xingwei Qu, Xu Tan, Yizhi Li, Zeyue Tian, Zhiyong Wu, Zhizheng Wu, Ziyang Ma, and Ziyu Wang. Foundation models for music: A survey, 2024. URL https://arxiv.org/abs/2408.14340.

Shivam Mehta, Ruibo Tu, Jonas Beskow, Éva Székely, and Gustav Eje Henter. Matcha-TTS: A fast TTS architecture with conditional flow matching. In Proc. ICASSP, 2024.

Ziqian Ning, Huakang Chen, Yuepeng Jiang, Chunbo Hao, Guobin Ma, Shuai Wang, Jixun Yao, and Lei Xie. Diffrhythm: Blazingly fast and embarrassingly simple end-to-end full-length song generation with latent diffusion. arXiv preprint arXiv:2503.01183, 2025.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 4172–4182. IEEE, 2023. doi: 10.1109/ICCV51070.2023.00387. URL https://doi.org/ 10.1109/ICCV51070.2023.00387.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 28492–28518. PMLR, 2023. URL https://proceedings.mlr.press/v202/radford23a.html.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

Simon Rouard, Francisco Massa, and Alexandre Défossez. Hybrid transformers for music source separation. In ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2023. doi: 10.1109/ICASSP49357.2023.10096956.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Wei Tan, Shun Lei, Huaicheng Zhang, Guangzheng Li, Yixuan Zhang, Hangting Chen, Jianwei Yu, Rongzhi Gu, and Dong Yu. Songprep: A preprocessing framework and end-to-end model for full-song structure parsing and lyrics transcription, 2025. URL https://arxiv.org/abs/2509. 17404.

Llama Team. The llama 3 herd of models. CoRR, abs/2407.21783, 2024. doi: 10.48550/ARXIV. 2407.21783. URL https://doi.org/10.48550/arXiv.2407.21783.

Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hoffman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, Carleigh Wood, Ann Lee, and Wei-Ning Hsu. Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound, 2025. URL https://arxiv.org/abs/2502.05139.

Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

Minz Won, Yun-Ning Hung, and Duc Le. A foundation model for music informatics. arXiv preprint arXiv:2311.03318, 2023.

Shangda Wu, Guo Zhancheng, Ruibin Yuan, Junyan Jiang, Seungheon Doh, Gus Xia, Juhan Nam, Xiaobing Li, Feng Yu, and Maosong Sun. Clamp 3: Universal music information retrieval across unaligned modalities and unseen languages. In Findings of the Association for Computational Linguistics: ACL 2025, pages 2605–2625, 2025.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. Qwen2.5-omni technical report, 2025. URL https://arxiv.org/abs/2503.20215.

Yaoxun Xu, Hangting Chen, Jianwei Yu, Wei Tan, Rongzhi Gu, Shun Lei, Zhiwei Lin, and Zhiyong Wu. Mucodec: Ultra low-bitrate music codec. CoRR, abs/2409.13216, 2024. doi: 10.48550/ ARXIV.2409.13216. URL https://doi.org/10.48550/arXiv.2409.13216.

Dongchao Yang, Songxiang Liu, Rongjie Huang, Jinchuan Tian, Chao Weng, and Yuexian Zou. Hifi-codec: Group-residual vector quantization for high fidelity audio codec. arXiv preprint arXiv:2305.02765, 2023a.

Dongchao Yang, Jinchuan Tian, Xu Tan, Rongjie Huang, Songxiang Liu, Xuankai Chang, Jiatong Shi, Sheng Zhao, Jiang Bian, Xixin Wu, et al. Uniaudio: An audio foundation model toward universal audio generation. arXiv preprint arXiv:2310.00704, 2023b.

Dongchao Yang, Dingdong Wang, Haohan Guo, Xueyuan Chen, Xixin Wu, and Helen Meng. Simplespeech: Towards simple and efficient text-to-speech with scalar latent transformer diffusion models. In Itshak Lapidot and Sharon Gannot, editors, 25th Annual Conference of the International Speech Communication Association, Interspeech 2024, Kos, Greece, September 1-5, 2024. ISCA, 2024. doi: 10.21437/INTERSPEECH.2024-1392. URL https://doi.org/10.21437/Interspeech. 2024-1392.

Dongchao Yang, Rongjie Huang, Yuanyuan Wang, Haohan Guo, Dading Chong, Songxiang Liu, Xixin Wu, and Helen Meng. Simplespeech 2: Towards simple and efficient text-to-speech with flow-based scalar latent transformer diffusion models. IEEE Transactions on Audio, Speech and Language Processing, 2025a.

Dongchao Yang, Songxiang Liu, Haohan Guo, Jiankun Zhao, Yuanyuan Wang, Helin Wang, Zeqian Ju, Xubo Liu, Xueyuan Chen, Xu Tan, Xixin Wu, and Helen M. Meng. Almtokenizer: A lowbitrate and semantic-rich audio codec tokenizer for audio language modeling. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025. OpenReview.net, 2025b. URL https://openreview.net/forum?id=QtMcq04xXd.

Jixun Yao, Guobin Ma, Huixin Xue, Huakang Chen, Chunbo Hao, Yuepeng Jiang, Haohe Liu, Ruibin Yuan, Jin Xu, Wei Xue, Hao Liu, and Lei Xie. Songeval: A benchmark dataset for song aesthetics evaluation, 2025. URL https://arxiv.org/abs/2505.10793.

Zhen Ye, Peiwen Sun, Jiahe Lei, Hongzhan Lin, Xu Tan, Zheqi Dai, Qiuqiang Kong, Jianyi Chen, Jiahao Pan, Qifeng Liu, Yike Guo, and Wei Xue. Codec does matter: Exploring the semantic shortcoming of codec for audio language model. In Toby Walsh, Julie Shah, and Zico Kolter, editors, AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, pages 25697–25705. AAAI Press, 2025. doi: 10.1609/AAAI.V39I24.34761. URL https://doi.org/10.1609/aaai.v39i24.34761.

Ruibin Yuan, Hanfeng Lin, Shuyue Guo, Ge Zhang, Jiahao Pan, Yongyi Zang, Haohe Liu, Yiming Liang, Wenye Ma, Xingjian Du, Xinrun Du, Zhen Ye, Tianyu Zheng, Yinghao Ma, Minghao Liu, Zeyue Tian, Ziya Zhou, Liumeng Xue, Xingwei Qu, Yizhi Li, Shangda Wu, Tianhao Shen, Ziyang Ma, Jun Zhan, Chunhui Wang, Yatian Wang, Xiaowei Chi, Xinyue Zhang, Zhenzhu Yang, Xiangzhou Wang, Shansong Liu, Lingrui Mei, Peng Li, Junjie Wang, Jianwei Yu, Guojian Pang, Xu Li, Zihao Wang, Xiaohuan Zhou, Lijun Yu, Emmanouil Benetos, Yong Chen, Chenghua Lin, Xie Chen, Gus Xia, Zhaoxiang Zhang, Chao Zhang, Wenhu Chen, Xinyu Zhou, Xipeng Qiu,

Roger B. Dannenberg, Zheng-Jia Liu, Jian Yang, Wenhao Huang, Wei Xue, Xu Tan, and Yike Guo. Yue: Scaling open foundation models for long-form music generation. CoRR, abs/2503.08638, 2025. doi: 10.48550/ARXIV.2503.08638. URL https://doi.org/10.48550/arXiv.2503.

##### 08638.

Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. Soundstream: An end-to-end neural audio codec. IEEE ACM Trans. Audio Speech Lang. Process., 30: 495–507, 2022. doi: 10.1109/TASLP.2021.3129994. URL https://doi.org/10.1109/TASLP. 2021.3129994.

Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J. Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu. Libritts: A corpus derived from librispeech for text-to-speech, 2019. URL https://arxiv.org/ abs/1904.02882.

Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, and Jie Tang. Glm-4-voice: Towards intelligent and human-like end-to-end spoken chatbot. arXiv preprint arXiv:2412.02612, 2024.

Haina Zhu, Yizhi Zhou, Hangting Chen, Jianwei Yu, Ziyang Ma, Rongzhi Gu, Yi Luo, Wei Tan, and Xie Chen. Muq: Self-supervised music representation learning with mel residual vector quantization. arXiv preprint arXiv:2501.01108, 2025.

