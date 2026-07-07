# arXiv:2605.30993v1[eess.AS]29May2026

## SwanVoice: Expressive Long-Form Zero-Shot Speech Synthesis for Both Monologue and Dialogue

[Figure 1]

Ruiqi Li1∗ Yu Zhang1∗ Changhao Pan1,2∗ Ke Lei1,2 Xiang Yin1† Cheng Yang1 1ByteDance, 2Zhejiang University {liruiqi.23,zhangyu.34,yinxiang.stephen}@bytedance.com

#### Abstract

Zero-shot text-to-speech (TTS) has improved substantially for single-speaker synthesis, yet expressive long-form multi-speaker dialogue remains difficult. A common workaround is to synthesize each turn with a monologue TTS model and stitch the outputs together. This adds inference cost and often breaks acoustic consistency, conversational coherence, and affective continuity across turns. Recent dialogue TTS systems have begun to address this setting, but they still struggle to keep expressive coherence, controllable speaker switching, and monologue quality at the same time. We present SwanData-Speech and SwanVoice. SwanDataSpeech builds monologue and dialogue corpora from in-the-wild audio, using Swan Forced Aligner for pause-aware word-level alignment and RobustMegaTTS3 for pronunciation-hard cases. Built on these data, SwanVoice is a zero-shot TTS model for 1–4 speakers, combining a 25 Hz VAE, raw-text conditioning with pause-aware symbols and pinyin substitution, and a flow-matching DiT with speaker-turn conditioning. Training starts from monologue speech, moves through mixed and real dialogue data, and then uses DiffusionNFT post-training with phone-level and speaker-similarity rewards. On SwanBench-Speech, SwanVoice obtains higher richness and hierarchy scores than all evaluated open-source baselines in both monologue and dialogue settings, while content accuracy remains the main limitation. Audio demos are available at https://swanaigc.github.io/#/swanvoice.

#### 1 Introduction

Recent advances in zero-shot text-to-speech (TTS) have made prompt-conditioned single-speaker synthesis increasingly reliable [3, 7, 11, 12, 19, 20, 25, 43, 45, 46]. Many speech-generation applications, however, require more than single-speaker narration. Short-form dramas, podcasts, and similar settings need TTS systems that treat a multi-party conversation as one generation problem [21, 57]. The common workaround is to synthesize one turn at a time and concatenate the waveforms. This can preserve each speaker locally, yet adjacent turns may disagree in room response, background ambience, speaking intensity, or pause timing. The result sounds assembled rather than recorded as a scene. A dialogue model therefore has to model full conversations, not isolated turns.

Recent dialogue-capable TTS models have shown end-to-end two-speaker generation and controllable speaker switching [21, 48, 64]. Long-form dialogue exposes failures that are less visible in short two-speaker generation: the acoustic environment should stay stable, speaker turns should remain separable even for similar voices, and affective continuity should carry across turns. At the same time, dialogue training should not degrade monologue synthesis. These failures are tightly coupled with data construction, since turn boundaries, pauses, and expressive labels shape turn control.

∗Equal contribution †Corresponding Author

Architecturally, modern zero-shot TTS systems combine speech representations, neural vocoders, Transformer-based text/audio modeling, and a generative module such as diffusion or flow matching. They can be roughly divided into autoregressive (AR) [11, 62] and non-autoregressive (NAR) [7, 19] formulations. Several dialogue TTS models use AR designs [21, 48]. In long dialogue, however, language-model-style AR generation brings sequential latency and exposure-bias failures such as word skipping or repetition [64]. NAR generative modeling is a better fit here because it reduces sequential decoding latency and conditions on the full text and speaker-turn sequence at once.

Two bottlenecks are central to this paper. 1) Dialogue data needs more than speaker labels. Expressive long-form synthesis needs speaker-consistent segments, pause-aware transcripts, quality filtering, and enough non-neutral speech to learn affective variation. These requirements interact: a speaker split error can corrupt turn control, while written-style punctuation can teach the model the wrong prosody. 2) Dialogue training should not erase monologue ability. Many dialogue models start from a monologue model and fine-tune on dialogue data with speaker-switch labels [21, 64]. This often improves turn control but can weaken monologue quality. The model also has to separate close voices, maintain a shared acoustic scene, and avoid pronunciation drift in long outputs.

We build SwanData-Speech, a pipeline for turning in-the-wild speech into monologue and dialogue training subsets. It is designed for sources such as podcasts, radio dramas, and film/TV content, where speakers, pauses, and acoustic conditions vary within long recordings. The pipeline includes: (i) a lightweight aligner, Swan Forced Aligner, for word-level timestamp alignment and pause-aware annotation; (ii) vocal separation and speaker segmentation modules built on existing methods; and (iii) quality and emotion filtering to retain clean expressive speech.

We then introduce SwanVoice, a zero-shot TTS model for 1–4 speakers. A 25 Hz VAE reduces the speech sequence length while preserving reconstruction quality. Raw text is kept as the main condition, with pause symbols and pinyin-substitution variants for pause control and Chinese pronunciation. The generator is a flow-matching DiT conditioned on speaker-turn IDs. SwanVoice is trained with a curriculum that moves from monologue speech to mixed and real conversational data, then post-trained with DiffusionNFT rewards for pronunciation robustness and speaker similarity.

#### 2 Data Processing Pipeline: SwanData-Speech

##### 2.1 Data Sources and Collection Scope

SwanData-Speech begins with a raw collection drawn mainly from internal resources, together with selected open-source Chinese and English datasets for broader linguistic and acoustic coverage. The raw collection contains approximately 2.59 million hours of audio, including about 2.24 million hours of Chinese data and 0.35 million hours of English data. We process this collection into task-specific subsets: SwanVoice uses the filtered monologue and dialogue subsets produced by the pipeline, while the 80K-hour subset in Appendix A is reserved for training and evaluating Swan Forced Aligner.

SwanVoice uses raw text as the conditioning input. This preserves richer semantic information, but it also increases sparsity for rare and polyphonic characters. A training corpus cannot exhaust all characters, pronunciation variants, or corner cases such as Chinese–English code-switching. Replacing all text with pinyin would reduce part of the sparsity problem, but it would also reduce readability and make authoring less convenient.

We therefore construct RobustMegaTTS3, a pronunciation-hard synthetic subset later rendered with MegaTTS 3. We collect the full word list from GCIDE 0.54 and the Level-1 and Level-2 character lists from the Table of General Standard Chinese Characters 3. An LLM (Qwen3-235B-A22BInstruct-2507) then generates five example sentences per entry.

We also use the LLM to create 20K Chinese hard cases and 20K English hard cases, covering polyphonic-character disambiguation in context, erhua, tone sandhi, onomatopoeic characters, homographs with different pronunciations, noun–verb stress shift, and irregular spellings. Another 100K Chinese–English code-switching texts span 13 scenarios and roles to stress mixed-language synthesis.

3http://www.moe.gov.cn/publicfiles/business/htmlfiles/moe/cmsmedia/other/2013/7/other98742.zip

To obtain accurate and standardized speech for these texts, we synthesize this portion of the audio with MegaTTS 3 [19], a phoneme-pronunciation-based model. RobustMegaTTS3 supplies dictionary-level pronunciation knowledge for rare and ambiguous pronunciations.

##### 2.2 Pipeline Overview

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

|[Figure 7]<br><br>Speaker3|
|---|

|[Figure 8]<br><br>Speaker1|
|---|

|[Figure 9]<br><br>[Figure 10]<br><br>Speaker2|
|---|

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Speech Enhancement

Speaker Diarization

Merge & Segmentation

In the Wild Data Clean Human Vocal Speaker Annotations

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

|[Figure 28]<br><br>[Figure 29]<br><br>| |
|---|---|
| | |
| | |
| | |

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

[Figure 35]

[Figure 36]

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

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

Quality Filtering

ASR Transcription

One Speaker Corpus

Segments with Transcripts

One Speaker Segments

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

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

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

Expressiveness Filtering

Punctuation Correction

[Figure 87]

[Figure 88]

[Figure 89]

Segments with Transcripts

Multiple Speaker Segments

Dialogue Corpus

Figure 1: Hierarchical data processing pipeline

The pipeline first applies speech enhancement and speaker diarization to raw audio. Based on speaker order, diarized segments are split into a monologue pool and a dialogue pool, and the two pools then go through ASR, punctuation refinement, and quality filtering separately. The output is two training datasets, one for monologue speech and one for dialogue conversations. We preserve the original sampling rate whenever possible during processing and resample all audio to 24 kHz only at the final stage. Figure 1 summarizes the hierarchical processing pipeline.

##### 2.3 Segmentation and Speaker-Aware Processing

- 2.3.1 Speech Enhancement We apply a vocal separation tool [4] to isolate the vocal component from all raw audio data.
- 2.3.2 Speaker Diarization

Most raw recordings are long, often more than ten hours per sample, and may contain multiple speakers in arbitrary order. We therefore split them into shorter speaker-ordered segments.

We use the open-source 3D-Speaker toolkit [8] for VAD, speaker embeddings, clustering, and diarization. It applies FSMN-Monophone VAD to split long audio into utterance-level chunks, then combines CAM++ [44] with spectral clustering for speaker-aware grouping.

After VAD and diarization, some segments are too short for stable training. We merge adjacent short segments from the same speaker when the silence between consecutive segments is at most 2 seconds. Segments shorter than 0.1 seconds, which are typically VAD artifacts, are removed, and each same-speaker merged sample is capped at 60 seconds.

For dialogue data, we merge consecutive multi-speaker segments up to 120 seconds. Each merged segment must contain 2–4 speakers, and no single silence interval may exceed 2 seconds. We use a sliding-window greedy merging strategy: starting from any monologue segment, a subsequent dialogue merge is kept as training data if it satisfies the constraints above. This partial overlap expands usable training data while preserving speaker order.

##### 2.4 Transcription and Alignment

- 2.4.1 ASR Transcription

We use SenseVoice-Small [2] for transcription and language identification, retaining only Chinese and English samples. Inverse text normalization (ITN) is disabled so that the model input stays closer to pronunciation; text normalization is left to a separate frontend model. Before pause correction, a small text Transformer restores punctuation for the transcribed text.

For dialogue utterances, we wrap the content of each speaker turn with special tokens of the form <S{id}> and </S{id}> to explicitly annotate the corresponding turn identity.

- 2.4.2 Punctuation Correction

The punctuation above is inferred from semantics. In conversational speech, however, semantic punctuation is often weakly correlated with actual pauses. A model trained on such text may learn to ignore punctuation and rely on dataset statistics for pause behavior, which leads to poor prosody in synthesized dialogue, especially around turn boundaries.

We revise punctuation in the transcribed text to better match acoustic pause patterns. A pretrained forced aligner first aligns the audio with the transcription and assigns a timestamp to each character. Pauses are then defined by the time gap between consecutive characters. Pauses shorter than 0.08 s are ignored. For pauses between 0.08 s and 0.18 s, we insert <|sp|>. For pauses between 0.18 s and 0.45 s, we use a comma. For pauses longer than 0.45 s, we use a period, exclamation mark, or question mark, depending on the original punctuation before correction; the default is a period. If punctuation appears where no pause is observed, it is removed. If a pause is observed without punctuation, punctuation is inserted. The aligner design and evaluation are reported in Appendix A.

2.5 Data Filtering

We score all audio samples with the non-intrusive DNSMOS metric [38]. PESQ [39] and STOI [50] are originally intrusive metrics, but we use the non-intrusive PESQ and STOI models from torchaudio-SQUIM [23] to score the full corpus.

After the initial filtering stage, emotion2vec+ [28] classifies the emotion of each sample and produces a confidence score. High-confidence non-neutral samples define the high-expressiveness subset.

- 3 Method: SwanVoice

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

VAE Decoder

VAE Decoder Generated Speech

[Figure 94]

Generated Latent

Generated Latent

Generated Speech

Target Length

Flow-based Transformer

Flow-based Transformer

[Figure 95]

- <S1>今天天气不错！
- <S2>诶，好像确实哦

| | |
|---|---|
| | |
| | |

- <S1>今天天气不错！
- <S2>诶，好像确实哦

|Tokenizer| | |
|---|---|---|
| | | |
| | | |

Reference Text

###### +

Text Token

###### +

- <S1>想出去玩吗？
- <S2>好啊好啊！

|Tokenizer|
|---|

Duration Model

1 1 2 2 1 1 2

1 1 2 2 1 1 2

- <S1>想出去玩吗？
- <S2>好啊好啊！

Text

Speaker Turn

+

+

Target Text

[Figure 96]

[Figure 97]

Noisy

[Figure 98]

[Figure 99]

[Figure 100]

VAE Encoder

VAE Encoder

[Figure 101]

Gaussian Noise

GT Wav Latent

GT Speech

Reference Speech

a) Training Procedure b) Inference Procedure

Figure 2: Overall training and inference procedure of SwanVoice.

##### 3.1 VAE

Given a speech waveform s, a variational encoder E maps s to a latent representation z, and a waveform decoder D reconstructs the signal as sˆ = D(z) = D(E(s)). To reduce computational cost and ease subsequent speech–text alignment, E temporally downsamples the input waveform by a factor of d. Architecturally, E follows the design in Ji et al. [18], while D is built upon HiFiGAN [22]. To capture high-frequency details and improve perceptual fidelity, we train with a set of

adversarial discriminators, including the multi-period discriminator (MPD), multi-scale discriminator (MSD), and multi-resolution discriminator (MRD) [17, 22]. The overall training objective is

L = Lrec + LKL + LAdv, (1)

where Lrec = ∥Φ(s) − Φ(ˆs)∥22 denotes the spectrogram-domain reconstruction loss computed by a feature extractor Φ, LKL is a lightly-weighted KL regularizer as in Rombach et al. [40], and LAdv is an LSGAN-style adversarial loss [29]. The compression rate is 25 latent frames per second.

##### 3.2 Tokenizer

We use the CosyVoice tokenizer [11] and feed raw text directly to the model. The text is tokenized by a BPE-based tokenizer, removing the need for a separate grapheme-to-phoneme (G2P) frontend. This simplifies preprocessing while allowing the model to learn context-dependent pronunciations end to end. For Chinese, the tokenizer provides a one-to-one character-level encoding, which prevents a single token from carrying an excessively long pronunciation and reduces sparse corner cases.

We add a dedicated pause token, <|sp|>, so the model can learn natural pausing patterns from text. For Chinese pronunciation control, the tokenizer vocabulary is augmented with 1,549 pinyin syllable combinations. During training, we randomly replace a subset of Chinese characters with pinyin forms extracted by pypinyin. This improves robustness to pronunciation variation. At inference time, pinyin hints can enforce the desired pronunciation of a character, which is useful for polyphonic characters and certain Northern Chinese dialect pronunciations.

For speaker annotation, we add a speaker-turn label sequence with the same length as the texttoken sequence. Each label indicates the speaker identity of the corresponding token. During text preprocessing, each speaker’s content is wrapped with turn-specific tags <S{id}> and </S{id}>. The speaker label sequence is constructed by detecting these tags and assigning the corresponding speaker ID to each token span.

##### 3.3 Flow-based Transformer

- As shown in Figure 2(a), the diffusion transformer (DiT) pads the text-token sequence and speakerturn embeddings to the temporal resolution of the waveform latent sequence. Instead of concatenating these heterogeneous conditions with the speech latent at the input, we first pass the padded text and turn representations through a lightweight Transformer stack. The model can therefore form text-side and turn-side features before they interact with the speech representation. Compared with naive early concatenation, this strategy improves in-context conditioning on the speech input [7].

The ground-truth waveform latent is constructed from a complete utterance. For monologue data, multiple short utterances from the same speaker may be concatenated into a longer sentence-level training example to improve long-form modeling. The waveform is encoded by the VAE into a latent sequence z⋆. We randomly split z⋆ into two contiguous parts: the first part is used as the reference segment, and the second part is the target segment to be generated. For dialogue data, the reference segment is required to contain at least a short span of speech from every speaker. Gaussian noise is injected into the target latent. The noised target, clean reference latent, processed text, and speaker-turn conditions are then fed into a flow-based Transformer. The Transformer is implemented as a deep stack of self-attention blocks and estimates the vector field over the latent trajectory.

We use RMSNorm [51] throughout the network and add AdaLN-based global adapters [35] to stabilize optimization and preserve long-form consistency in speaker timbre and recording conditions.

Following the standard flow-matching formulation, the model is trained to predict the velocity field between a noise sample and the clean target latent:

Lflow = Et∼U(0,1),z⋆∼pdata,ϵ∼N(0,I) ∥uθ(zt,t,c) − (z⋆ − ϵ)∥22 , (2) where

zt = (1 − t)ϵ + tz⋆, (3)

and c denotes the full conditioning information, including the processed text tokens, speaker-turn embeddings, and the reference speech latent used for conditioning.

##### 3.4 Curriculum Learning

Training directly on conversational data from scratch often produces unintelligible speech. The main difficulty is learning speech-text alignment from spoken conversations with multiple speakers, while still preserving strong monologue performance. We therefore use a three-stage curriculum that gradually moves from monologue data to real conversational data.

- 1) Monologue pretraining. We first train the model from scratch on monologue speech data. This stage uses approximately 2 million hours of monologue speech covering both Chinese and English. It establishes the basic synthesis ability, including high-fidelity acoustic modeling and reliable speech– text alignment. Starting the later stages from this pretrained model avoids many of the audio-quality and pronunciation failures that appear when training directly on complex conversational data.

This stage is also augmented with the pronunciation-hard and code-switching synthetic cases described in Section 2. These cases are difficult to collect at scale, and phoneme-based synthesis covers pronunciations that are rare in crawled speech.

- 2) Mixed conversational training. In the second stage, the pretrained monologue model is trained on monologue data together with concatenated 2–4-speaker conversational data. Since speaker diarization is imperfect, directly using real conversational data can make speaker transitions difficult to learn. The concatenated data provides an intermediate step in which the model learns to assign the correct speaker identity to each turn. Conversational examples are sampled more often than their raw-hour share so the model sees speaker switches frequently, while monologue examples remain in the mixture to prevent monologue degradation.
- 3) SFT training. In the third stage, the model is trained on monologue data together with real 2–4-speaker conversational data. By this point, it already has stable speaker-switching ability, so real conversational data can be used to learn higher-level dialogue consistency, including recordingenvironment consistency and emotional coherence. Monologue examples remain in the mixture to protect monologue performance. The real conversational data mainly comes from movies, TV dramas, and podcasts, which expose the model to richer affective and conversational variation.

##### 3.5 Post Training

After supervised training, the DiT-based TTS model still makes predictable errors: difficult words may be misread, and prompt speaker identity can drift. We address these errors with a post-training stage that optimizes model-generated samples against pronunciation and timbre rewards. Since usable reward models are available in our setting, we use online reinforcement learning. Instead of introducing an additional value model, we use a value-free optimization strategy and instantiate it with DiffusionNFT [61], which matches the flow-matching backbone. The rewards target phone-level consistency and speaker similarity, not recording-environment consistency or expressiveness.

Flow-GRPO [27] is an early attempt to bring online RL to flow-matching models. It converts the deterministic ODE sampling process into an equivalent SDE for stochastic exploration and uses a denoising-reduction strategy to lower training cost. DiffusionNFT is simpler for this setting: it performs policy optimization on the forward process through the flow-matching objective, addresses the forward-inconsistency issue of reverse-process RL, allows arbitrary black-box solvers, and only requires final clean samples with rewards rather than the full latent trajectory. DiffusionNFT also reports better efficiency than Flow-GRPO in head-to-head comparisons.

##### 3.5.1 Reward Models

The reward has two components: an ASR-based robustness reward for intelligibility and recognition errors, and a speaker-similarity reward for timbre preservation. Differentiable ASR-based optimization is possible in principle, but it would complicate the training pipeline and is not needed here. We use a reward-driven online RL formulation in which the model is updated from sampled utterances and their rewards without differentiating through the recognizers.

The first reward is the phone consistency reward rphone, which measures how well the generated speech matches the target text at the phoneme and tone levels. We apply an external phone recognizer to xˆ and compare the resulting phonetic sequence with the phonetic realization implied by y, yielding a normalized score in [0,1].

We remove punctuation and silence symbols on both sides, and merge each phone-tone pair into a single token, e.g., uj = phonej_tonej. Let uref and uhyp denote the resulting reference and predicted token sequences. The resulting WER and phone reward are

S + D + I max(1,|uref|)

WER(uref,uhyp) =

, (4) rphone = exp − WER(uref,uhyp) , (5)

- where S, D, and I are the numbers of substitutions, deletions, and insertions, respectively. We use a phone-based recognizer rather than the character- or word-based recognizers often used in ASR-derived rewards because the objective here is pronunciation accuracy and polyphonic-character disambiguation, especially in Chinese, rather than exact character identity.

The second reward is a speaker similarity reward rsim4, which compares the generated speech with the reference prompt in a pretrained speaker-embedding space:

rsim(ˆx,xref) = cos fspk(ˆx),fspk(xref) , (6) where fspk(·) is a frozen speaker encoder. We aggregate the two rewards as

- 1

- 2

rphone + rsim , (7)

r =

which is the default setting in our experiments. The framework also supports a weighted sum of multiple rewards when deployment priorities require different trade-offs.

##### 3.5.2 DiffusionNFT-style Policy Optimization

For each prompt, we draw multiple candidates from πold and compute their rewards. A prompt-wise advantage is formed by subtracting a within-prompt baseline:

K

1 K

rj, (8)

Ai = ri − r,¯ r¯ =

j=1

where K is the number of sampled candidates for the same condition. We clip the advantage and map it into a soft preference weight:

A ˜i 2Amax

- 1

- 2

A˜i = clip(Ai,−Amax,Amax), wi = clip

,0,1 . (9)

+

Let vθ(zt,t,c), vold(zt,t,c), and vref(zt,t,c) denote the denoising predictions of the online, old, and reference policies, respectively, under latent state zt, timestep t, and condition c = (y,xref). Following the NFT-style update rule, we construct positive and implicit-negative denoising branches:

vi+ = βNFTvθ + (1 − βNFT)sg(vold), (10) vi− = (1 + βNFT)sg(vold) − βNFTvθ, (11)

where sg(·) denotes stop-gradient and βNFT controls the interpolation strength. These predictions are converted to denoised latent estimates for the non-prompt region. The online policy is optimized

to prefer the positive branch when wi is large and the implicit negative branch when wi is small. The objective can be written as

1 − wi βNFT

wi βNFT

ℓ z ˆ0−,i,z0,i , (12)

ℓ z ˆ0+,i,z0,i +

LNFT = Ei

where ℓ(·,·) denotes the masked denoising loss on the generated target segment. To prevent the policy from drifting too far from the pretrained model, we add a reference-policy regularizer:

L = LNFT + λrefLref, Lref = E ∥vθ − sg(vref)∥22 . (13) This reference regularization preserves the speech quality and robustness inherited from supervised pretraining while still allowing reward-driven adaptation.

For post-training data, we collected 3K audio samples of real human conversations, transcribed them into text, and corrected the pause annotations. The post-training objective explicitly optimizes only phone-level WER and speaker similarity. In qualitative inspection, the resulting model also shows better recording-environment consistency and stronger expressiveness, which we treat as side effects.

4https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification

##### 3.6 Inference Procedure

- As shown in Figure 2(b), inference takes a reference speech segment and a target text sequence as input. The model synthesizes the target linguistic content while preserving the speaker identity and speaking style of the reference speech. We transcribe the reference speech with SenseVoice-Small [2] to obtain speaker-specific reference text. The target duration is estimated with a simple speaking-rate heuristic for each speaker in the reference speech. We also use sway sampling [7], which encourages the model to capture coarse speech contours in the early generation stage and refine fine-grained details later. The speech-text alignment is therefore largely determined by the first few denoising steps. Finally, the VAE decoder converts the target latent into a waveform.

We also introduce a staircase classifier-free guidance (CFG) strategy. It uses two guidance scales and three conditioning variants: a null condition, a full condition, and a text-only condition. The guided prediction is defined as

v˜t = v∅ + ωtext vtext − v∅ + ωref vfull − vtext ,

where ωtext and ωref denote the guidance scales for textual content and reference-dependent speaker/style information. The staircase formulation separates content guidance from reference guidance,

allowing the two effects to be controlled independently during inference. Increasing ωref moves the output toward the reference timbre and style without changing text guidance.

#### 4 Experiments

##### 4.1 Implementation Details

The main SwanVoice model has 2 billion parameters. Monologue pretraining uses 64 A100 GPUs for 500k steps, mixed conversational training uses 32 A100 GPUs for 600k steps, and supervised fine-tuning (SFT) uses 32 A100 GPUs for 300k steps. Post-training uses 8 A100 GPUs for 50 epochs.

##### 4.2 Evaluation Metrics

Following the evaluation protocol of SwanBench-Speech [34], we evaluate each model along three axes: acoustics, semantics, and expressiveness.

Acoustics For acoustics, we report timbre consistency, reverb consistency, and sound fidelity. Timbre consistency is measured with segment-based speaker similarity, computed as the average similarity of speaker embeddings 5 across segments. Reverb consistency follows the same idea: we compute the standard deviation of SRMR 6 values within sliding windows to measure the stability of the synthesized acoustic environment. Sound fidelity is measured by SQUIM-PESQ through the official torchaudio interface as a non-intrusive, reference-free metric 7.

Semantics For semantics, we evaluate content error rate and prosodic coherence. Content errors are measured by Character Error Rate (CER) on Chinese datasets and Word Error Rate (WER) on English datasets. Both are computed with FunASR-Nano [2] as the ASR model and JiWER as the calculation backend. For prosody, we use SpeechJudge [52], a Qwen2.5-Omni model fine-tuned for audio quality assessment. Prosodic coherence is rated on a 1.0–5.0 scale, where 1 means poor coherence and 5 means excellent coherence.

Expressiveness We evaluate expressiveness from two views: sentence-level expressive richness and paragraph-level expressive hierarchy for long-form speech. Because MOS prediction networks can correlate poorly with human perception [31], we use an MLLM-as-a-judge protocol with a large audio language model as the evaluator. For expressive richness, the audio waveform is segmented into non-overlapping 10-second chunks {ci}Mi=1. The evaluator assigns an expressiveness score si

to each chunk ci, and the final richness score is the arithmetic mean: Scorerich = ( Mi=1 si)/M. For expressive hierarchy, the full audio sequence is fed into the evaluator, which scores the speech

- 5https://huggingface.co/docs/transformers/en/model_doc/unispeech-sat
- 6https://github.com/jfsantos/SRMRpy
- 7https://docs.pytorch.org/audio/main/tutorials/squim_tutorial.html

- Table 1: Evaluation results of long-form TTS models across multi-dimensional metrics. Metrics cover Acoustics (Timbre/Reverb Consistency, Sound Fidelity), Semantics (Content Error, Prosodic Coherence), and Expressiveness (Richness, Hierarchy). The best and second-best results are marked in bold and underlined, respectively, for each metric.

Acoustics Semantics Expressiveness Model

Timbre(↑) Reverb(↓) Sound Fidelity(↑) Content Error(↓) Prosody(↑) Richness(↑) Hierarchy(↑) Open-Source Models

- CosyVoice-2 0.93 2.37 3.58 0.106 2.81 2.02 2.59

- CosyVoice-3 0.93 2.73 3.80 0.077 3.26 2.64 2.47 FishSpeech 0.93 2.00 4.09 0.066 3.77 2.37 2.90

F5TTS 0.92 2.12 2.60 0.085 2.87 2.77 2.97

GLM-TTS 0.94 1.64 3.90 0.074 3.28 1.57 2.39 IndexTTS-2 0.93 1.77 2.78 0.077 3.63 3.32 2.94 MegaTTS-3 0.93 2.07 3.52 0.072 3.22 2.40 3.01

SparkTTS 0.92 2.04 3.53 0.314 2.35 2.23 2.22 VibeVoice 0.92 2.45 3.47 0.092 3.75 3.42 3.06

ZipVoice 0.89 2.10 3.53 0.213 2.97 2.11 2.05 Average 0.92 2.13 3.48 0.12 3.19 2.49 2.66

SwanVoice 0.93 2.06 3.60 0.172 3.56 3.81 3.62

- Table 2: Results of dialogue generation models across SwanBench-Speech metrics. The best and second-best results are marked in bold and underlined, respectively, for each metric.

Acoustics Semantics Expressiveness Model

Timbre(↑) Reverb(↓) Sound Fidelity(↑) Content Error(↓) Prosody(↑) Richness(↑) Hierarchy(↑) Open-Source Models

FireRedTTS-2 0.91 3.54 2.54 0.148 2.93 2.52 2.65

MoonCast 0.90 3.29 2.60 0.284 2.93 2.42 2.54 MOSS-TTSD 0.89 3.52 2.83 0.227 2.57 3.04 2.86

SoulX-Podcast 0.92 3.23 3.98 0.101 3.89 2.80 3.15 VibeVoice 0.89 2.09 2.75 0.204 3.00 3.09 2.83 ZipVoice-Dialog 0.90 3.49 2.48 0.116 3.46 2.88 2.93 Average 0.90 3.19 2.86 0.180 3.13 2.79 2.83

SwanVoice 0.92 3.02 3.77 0.145 3.70 3.62 3.71

along three dimensions: Emotional Variation, Vocal Dynamics, and Scene Appropriateness. We use Gemini-3-Pro as the evaluator and report both scores on a 1–5 scale, where 1 is poor and 5 is excellent. The evaluator is given only the audio and a fixed scoring rubric, without system names, and samples from different systems are evaluated in a randomized order.

##### 4.3 Baselines

For monologue generation, we compare with ten open-source models: ZipVoice [64], SparkTTS [45], CosyVoice2-0.5B [10], CosyVoice3-0.5B [11], GLM-TTS [9], MegaTTS3 [19], IndexTTS2 [62], FishSpeech-1.5 [26], F5TTS [7], and VibeVoice [36].

For dialogue generation, we compare with six open-source long-form models: ZipVoice-Dialog [63], MoonCast [21], MOSS-TTSD [60], FireRedTTS2 [48], VibeVoice, and SoulX-Podcast [47].

##### 4.4 Zero-Shot Monologue TTS

Table 1 reports results on the Expressive Challenge subset of SwanBench-Speech. SwanVoice reaches

- 3.81 in richness and 3.62 in hierarchy, higher than all evaluated open-source baselines. Relative to VibeVoice, the strongest baseline on these two metrics, the gains are 0.39 and 0.56 points. The model is not the best on content error, but it keeps 0.93 timbre consistency, 3.60 sound fidelity, and 3.56 prosodic coherence, all at or above the open-source average.

##### 4.5 Zero-Shot Dialogue TTS

For dialogue, SwanVoice reaches 3.62/3.71 on richness/hierarchy, 0.53/0.56 points higher than the strongest baselines. Content error is below the baseline average but not the best in the table, and the demo page further includes 3–4-speaker cases.

#### 5 Conclusion

SwanVoice treats long-form dialogue as a full-context generation problem rather than a sequence of isolated turns. In our experiments, this matters most for expressiveness: SwanVoice obtains higher richness and hierarchy scores than all evaluated open-source baselines in both monologue and dialogue settings. The data pipeline contributes directly to this behavior. Speaker-aware segmentation, pause-aware alignment, pronunciation hard cases, and emotion-based filtering each address a failure mode that becomes audible in long speech.

The current model still has clear limitations. Content accuracy remains weaker than the best baselines in several settings, and speaker switching can still fail when the speakers are acoustically close or when the prompt is short. These errors suggest three directions for improvement: pronunciation control, alignment and pause modeling, and more robust speaker-turn conditioning. Future work should therefore focus on making long-form speech generation more reliable.

#### References

- [1] Amodei, D., Ananthanarayanan, S., Anubhai, R., Bai, J., Battenberg, E., Case, C., Casper, J., Catanzaro, B., Cheng, Q., Chen, G., et al. Deep speech 2: End-to-end speech recognition in english and mandarin. In International conference on machine learning, pp. 173–182. PMLR, 2016.
- [2] An, K., Chen, Q., Deng, C., Du, Z., Gao, C., Gao, Z., Gu, Y., He, T., Hu, H., Hu, K., et al. Funaudiollm: Voice understanding and generation foundation models for natural interaction between humans and llms. arXiv preprint arXiv:2407.04051, 2024.
- [3] Anastassiou, P., Chen, J., Chen, J., Chen, Y., Chen, Z., Chen, Z., Cong, J., Deng, L., Ding, C., Gao, L., et al. Seed-tts: A family of high-quality versatile speech generation models. arXiv preprint arXiv:2406.02430, 2024.
- [4] Anjok07 and aufr33. Ultimate vocal remover. https://github.com/Anjok07/ ultimatevocalremovergui, 2020.
- [5] Bain, M., Huh, J., Han, T., and Zisserman, A. Whisperx: Time-accurate speech transcription of long-form audio. arXiv preprint arXiv:2303.00747, 2023.
- [6] Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J., Kanda, N., Yoshioka, T., Xiao, X., et al. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing, 16(6):1505–1518, 2022.
- [7] Chen, Y., Niu, Z., Ma, Z., Deng, K., Wang, C., Zhao, J., Yu, K., and Chen, X. F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching. arXiv preprint arXiv:2410.06885, 2024.
- [8] Chen, Y., Zheng, S., Wang, H., Cheng, L., Zhu, T., Huang, R., Deng, C., Chen, Q., Zhang, S., Wang, W., et al. 3d-speaker-toolkit: An open-source toolkit for multimodal speaker verification and diarization. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2025.
- [9] Cui, J., Yang, Z., Li, N., Tian, J., Ma, X., Zhang, Y., Chen, G., Yang, R., Cheng, Y., Zhou, Y., et al. Glm-tts technical report. arXiv preprint arXiv:2512.14291, 2025.
- [10] Du, Z., Wang, Y., Chen, Q., Shi, X., Lv, X., Zhao, T., Gao, Z., Yang, Y., Gao, C., Wang, H., et al. Cosyvoice 2: Scalable streaming speech synthesis with large language models. arXiv preprint arXiv:2412.10117, 2024.
- [11] Du, Z., Gao, C., Wang, Y., Yu, F., Zhao, T., Wang, H., Lv, X., Wang, H., Ni, C., Shi, X., et al. Cosyvoice 3: Towards in-the-wild speech generation via scaling-up and post-training. arXiv preprint arXiv:2505.17589, 2025.
- [12] Guo, H.-H., Liu, K., Shen, F.-Y., Wu, Y.-C., Xie, F.-L., Xie, K., and Xu, K.-T. Fireredtts: A foundation text-to-speech framework for industry-level generative speech applications. arXiv preprint arXiv:2409.03283, 2024.
- [13] Guo, W., Pan, C., Zhu, Z., Hu, X., Zhang, Y., Tang, L., Yang, R., Wang, H., Zhang, Z., Wang, Y., Chen, Y., Xu, H., Xu, K., Fan, P., Chen, Z., Yu, Y., Huang, Q., Wu, F., and Zhao, Z. MRSAudio: A large-scale multimodal recorded spatial audio dataset with refined annotations. In Advances in Neural Information Processing Systems, 2025.
- [14] Guo, W., Zhang, Y., Pan, C., Huang, R., Tang, L., Li, R., Hong, Z., Wang, Y., and Zhao, Z. Techsinger: Technique controllable multilingual singing voice synthesis via flow matching. arXiv preprint arXiv:2502.12572, 2025.
- [15] Guo, W., Zhang, Y., Pan, C., Zhu, Z., Li, R., Chen, Z., Xu, W., Wu, F., and Zhao, Z. STARS: A unified framework for singing transcription, alignment, and refined style annotation. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 15081–15093, 2025.
- [16] Hu, K., Puvvada, K., Rastorgueva, E., Chen, Z., Huang, H., Ding, S., Dhawan, K., Xu, H., Balam, J., and Ginsburg, B. Word level timestamp generation for automatic speech recognition and translation. arXiv preprint arXiv:2505.15646, 2025.

- [17] Jang, W., Lim, D., Yoon, J., Kim, B., and Kim, J. Univnet: A neural vocoder with multiresolution spectrogram discriminators for high-fidelity waveform generation. arXiv preprint arXiv:2106.07889, 2021.
- [18] Ji, S., Jiang, Z., Wang, W., Chen, Y., Fang, M., Zuo, J., Yang, Q., Cheng, X., Wang, Z., Li, R., et al. Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling. arXiv preprint arXiv:2408.16532, 2024.
- [19] Jiang, Z., Ren, Y., Li, R., Ji, S., Zhang, B., Ye, Z., Zhang, C., Jionghao, B., Yang, X., Zuo, J., et al. Megatts 3: Sparse alignment enhanced latent diffusion transformer for zero-shot speech synthesis. arXiv preprint arXiv:2502.18924, 2025.
- [20] Ju, Z., Wang, Y., Shen, K., Tan, X., Xin, D., Yang, D., Liu, E., Leng, Y., Song, K., Tang, S., Wu, Z., Qin, T., Li, X., Ye, W., Zhang, S., Bian, J., He, L., Li, J., and sheng zhao. Naturalspeech 3: Zero-shot speech synthesis with factorized codec and diffusion models. In Proc. International Conference on Machine Learning (ICML), 2024.
- [21] Ju, Z., Yang, D., Yu, J., Shen, K., Leng, Y., Wang, Z., Tan, X., Zhou, X., Qin, T., and Li, X. Mooncast: High-quality zero-shot podcast generation. arXiv preprint arXiv:2503.14345, 2025.
- [22] Kong, J., Kim, J., and Bae, J. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. Advances in neural information processing systems, 33:17022–17033, 2020.
- [23] Kumar, A., Tan, K., Ni, Z., Manocha, P., Zhang, X., Henderson, E., and Xu, B. Torchaudiosquim: Reference-less speech quality and intelligibility measures in torchaudio. In ICASSP 2023

- 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5, 2023. doi: 10.1109/ICASSP49357.2023.10096680. URL https://doi.org/10. 1109/ICASSP49357.2023.10096680.

- [24] Li, R., Zhang, Y., Wang, Y., Hong, Z., Huang, R., and Zhao, Z. Robust singing voice transcription serves synthesis. arXiv preprint arXiv:2405.09940, 2024.
- [25] Li, Y., Zhou, X., Wang, J., Wang, L., Wu, Y., Zhou, S., Zhou, Y., and Shu, J. Indextts 2.5 technical report. arXiv preprint arXiv:2601.03888, 2026.
- [26] Liao, S., Wang, Y., Li, T., Cheng, Y., Zhang, R., Zhou, R., and Xing, Y. Fish-speech: Leveraging large language models for advanced multilingual text-to-speech synthesis. arXiv preprint arXiv:2411.01156, 2024.
- [27] Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.
- [28] Ma, Z., Zheng, Z., Ye, J., Li, J., Gao, Z., Zhang, S., and Chen, X. emotion2vec: Selfsupervised pre-training for speech emotion representation. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 15747–15760, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.931. URL https://aclanthology.org/2024.findings-acl.931/.
- [29] Mao, X., Li, Q., Xie, H., Lau, R. Y., Wang, Z., and Paul Smolley, S. Least squares generative adversarial networks. In Proceedings of the IEEE International Conference on Computer Vision, pp. 2794–2802, 2017.
- [30] McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., and Sonderegger, M. Montreal forced aligner: Trainable text-speech alignment using kaldi. In Proc. Interspeech, volume 2017, pp. 498–502, 2017.
- [31] Minixhofer, C., Klejch, O., and Bell, P. Ttsds2: resources and benchmark for evaluating human-quality text to speech systems. arXiv preprint arXiv:2506.19441, 2025.
- [32] Mu, B., Shi, X., Wang, X., Liu, H., Xu, J., and Xie, L. Llm-forcedaligner: A non-autoregressive and accurate llm-based forced aligner for multilingual and long-form speech. arXiv preprint arXiv:2601.18220, 2026.

- [33] Pan, C., Guo, W., Zhang, Y., Zhu, Z., Chen, Z., Wang, H., and Zhao, Z. A multimodal evaluation framework for spatial audio playback systems: From localization to listener preference. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 7006–7015, 2025. doi: 10.1145/3746027.3755571.
- [34] Pan, C., Yang, R., Wang, H., Zhou, Z., He, X., Guo, W., Jiang, Z., Li, R., Zhang, Y., Wen, C., Lei, K., Yin, X., Lu, J., Zhu, Z., and Zhao, Z. Comprehensive benchmarking of long-form speech generation in diverse scenarios, 2026.
- [35] Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.
- [36] Peng, Z., Yu, J., Wang, W., Chang, Y., Sun, Y., Dong, L., Zhu, Y., Xu, W., Bao, H., Wang, Z., Huang, S., Xia, Y., and Wei, F. Vibevoice technical report. arXiv preprint arXiv:2508.19205,

2025. doi: 10.48550/arXiv.2508.19205. URL https://arxiv.org/abs/2508.19205.

- [37] Rastorgueva, E., Lavrukhin, V., and Ginsburg, B. Nemo forced aligner and its application to word alignment for subtitle generation. In Interspeech, pp. 5257–5258, 2023.
- [38] Reddy, C. K. A., Gopal, V., and Cutler, R. Dnsmos: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors. In ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 6493–6497, 2021. doi: 10.1109/ICASSP39728.2021.9414878. URL https://doi.org/10.1109/ICASSP39728. 2021.9414878.
- [39] Rix, A. W., Beerends, J. G., Hollier, M. P., and Hekstra, A. P. Perceptual evaluation of speech quality (pesq)-a new method for speech quality assessment of telephone networks and codecs. In 2001 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), volume 2, pp. 749–752, 2001. doi: 10.1109/ICASSP.2001.941023. URL https: //doi.org/10.1109/ICASSP.2001.941023.
- [40] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.
- [41] Shi, X., Chen, Y., Zhang, S., and Yan, Z. Achieving timestamp prediction while recognizing with non-autoregressive end-to-end asr model. In National Conference on Man-Machine Speech Communication, pp. 89–100. Springer, 2022.
- [42] Strgar, L. and Harwath, D. Phoneme segmentation using self-supervised speech models. In 2022 IEEE Spoken Language Technology Workshop (SLT), pp. 1067–1073. IEEE, 2023.
- [43] Wang, C., Chen, S., Wu, Y., Zhang, Z., Zhou, L., Liu, S., Chen, Z., Liu, Y., Wang, H., Li, J., et al. Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint arXiv:2301.02111, 2023.
- [44] Wang, H., Zheng, S., Chen, Y., Cheng, L., and Chen, Q. Cam++: A fast and efficient network for speaker verification using context-aware masking. arXiv preprint arXiv:2303.00332, 2023.
- [45] Wang, X., Jiang, M., Ma, Z., Zhang, Z., Liu, S., Li, L., Liang, Z., Zheng, Q., Wang, R., Feng, X., et al. Spark-tts: An efficient llm-based text-to-speech model with single-stream decoupled speech tokens. arXiv preprint arXiv:2503.01710, 2025.
- [46] Wang, Y., Zhan, H., Liu, L., Zeng, R., Guo, H., Zheng, J., Zhang, Q., Zhang, X., Zhang, S., and Wu, Z. Maskgct: Zero-shot text-to-speech with masked generative codec transformer. arXiv preprint arXiv:2409.00750, 2024.
- [47] Xie, H., Lin, H., Cao, W., Guo, D., Tian, W., Wu, J., Wen, H., Shang, R., Liu, H., Jiang, Z., et al. Soulx-podcast: Towards realistic long-form podcasts with dialectal and paralinguistic diversity. arXiv preprint arXiv:2510.23541, 2025.
- [48] Xie, K., Shen, F., Li, J., Xie, F., Tang, X., and Hu, Y. Fireredtts-2: Towards long conversational speech generation for podcast and chatbot. arXiv preprint arXiv:2509.02020, 2025.

- [49] Xu, J., Guo, Z., Hu, H., Chu, Y., Wang, X., He, J., Wang, Y., Shi, X., He, T., Zhu, X., et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.
- [50] Zezario, R. E., Fu, S.-W., Fuh, C.-S., Tsao, Y., and Wang, H.-M. Stoi-net: A deep learning based non-intrusive speech intelligibility assessment model. In Asia-Pacific Signal and Information Processing Association Annual Summit and Conference, APSIPA 2020, Auckland, New Zealand, December 7–10, 2020, pp. 482–486. IEEE, 2020. URL https://ieeexplore.ieee.org/ document/9306495.
- [51] Zhang, B. and Sennrich, R. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.
- [52] Zhang, X., Wang, C., Liao, H., Li, Z., Wang, Y., Wang, L., Jia, D., Chen, Y., Li, X., Chen, Z., et al. Speechjudge: Towards human-level judgment for speech naturalness. arXiv preprint arXiv:2511.07931, 2025.
- [53] Zhang, Y., Huang, R., Li, R., He, J., Xia, Y., Chen, F., Duan, X., Huai, B., and Zhao, Z. Stylesinger: Style transfer for out-of-domain singing voice synthesis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19597–19605, 2024.
- [54] Zhang, Y., Jiang, Z., Li, R., Pan, C., He, J., Huang, R., Wang, C., and Zhao, Z. Tcsinger: Zeroshot singing voice synthesis with style transfer and multi-level style control. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 1960–1975, 2024.
- [55] Zhang, Y., Pan, C., Guo, W., Li, R., Zhu, Z., Wang, J., Xu, W., Lu, J., Hong, Z., Wang, C., et al. Gtsinger: A global multi-technique singing corpus with realistic music scores for all singing tasks. Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [56] Zhang, Y., Guo, W., Pan, C., Yao, D., Zhu, Z., Jiang, Z., Wang, Y., Jin, T., and Zhao, Z. TCSinger 2: Customizable multilingual zero-shot singing voice synthesis. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Proc. Annual Meeting of the Association for Computational Linguistics (ACL), pp. 13280–13294, Vienna, Austria, 2025.
- [57] Zhang, Y., Guo, W., Pan, C., Zhu, Z., Jin, T., and Zhao, Z. Isdrama: Immersive spatial drama generation through multimodal prompting. arXiv preprint arXiv:2504.20630, 2025.
- [58] Zhang, Y., Guo, W., Pan, C., Zhu, Z., Li, R., Lu, J., Huang, R., Zhang, R., Hong, Z., Jiang, Z., and Zhao, Z. Versatile framework for song generation with prompt-based control. In Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 195–219, 2025.
- [59] Zhang, Y., Tian, B., and Duan, Z. Conan: A chunkwise online network for zero-shot adaptive voice conversion. In Proceedings of the IEEE Automatic Speech Recognition and Understanding Workshop, 2025.
- [60] Zhao, X., Xu, Z., Cheng, Q., Fei, Z., Jin, L., Wang, Y., Chen, H., Jiang, Y., Gao, Q., Chen, K., et al. Moss-speech: Towards true speech-to-speech models without text guidance. arXiv preprint arXiv:2510.00499, 2025.
- [61] Zheng, K., Chen, H., Ye, H., Wang, H., Zhang, Q., Jiang, K., Su, H., Ermon, S., Zhu, J., and Liu, M.-Y. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.
- [62] Zhou, S., Zhou, Y., He, Y., Zhou, X., Wang, J., Deng, W., and Shu, J. Indextts2: A breakthrough in emotionally expressive and duration-controlled auto-regressive zero-shot text-to-speech.

- arXiv preprint arXiv:2506.21619, 2025.

[63] Zhu, H., Kang, W., Guo, L., Yao, Z., Kuang, F., Zhuang, W., Li, Z., Han, Z., Zhang, D., Zhang, X., et al. Zipvoice-dialog: Non-autoregressive spoken dialogue generation with flow matching.

- arXiv preprint arXiv:2507.09318, 2025.

- [64] Zhu, H., Kang, W., Yao, Z., Guo, L., Kuang, F., Li, Z., Zhuang, W., Lin, L., and Povey, D. Zipvoice: Fast and high-quality zero-shot text-to-speech with flow matching. arXiv preprint arXiv:2506.13053, 2025.

- [65] Zhu, J., Zhang, C., and Jurgens, D. Phone-to-audio alignment without text: A semi-supervised approach. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 8167–8171. IEEE, 2022.
- [66] Zhu, Z., Zhang, Y., Guo, W., Pan, C., and Zhao, Z. ASAudio: A survey of advanced spatial audio research. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, 2025.

## Appendices

### SwanVoice: Expressive Long-Form Zero-Shot Speech Synthesis for Both Monologue and Dialogue

[Figure 102]

#### A Swan Forced Aligner

##### A.1 Why Do We Need a Forced Aligner?

Modern ASR systems often output punctuated transcripts, either directly or through auxiliary punctuation restoration modules. This punctuation is usually optimized for readability and semantic plausibility, not for the acoustic pause structure of speech. As a result, ASR punctuation may correlate only weakly with real pauses, hesitations, or phrase boundaries in the waveform.

This mismatch is important when ASR-generated annotations are used to train downstream TTS systems. If punctuation does not reliably correspond to acoustic pauses, a TTS model may learn weak or inconsistent pause control: punctuation may fail to trigger a pause, while pauses may appear where no punctuation exists. These errors degrade downstream TTS prosody and controllability.

This motivates a dedicated forced aligner that grounds textual units in the speech signal and recovers word boundaries and pause structure from acoustic evidence rather than ASR punctuation conventions.

For expressive in-the-wild audio, raw recordings are rarely useful as training supervision without reliable transcripts, temporal boundaries, and fine-grained attribute labels. Otherwise, annotation errors are inherited by downstream generation models [15, 24, 55]. The problem becomes more pronounced in structured controllable audio generation, where the model must separate linguistic content, speaker identity, pronunciation, style, and expressive factors from imperfect supervision in large-scale real-world pipelines [14, 53, 54, 56].

##### A.2 Overview

Forced alignment aligns a transcript with a speech waveform and predicts temporal boundaries such as word-level start and end times. In practice, pauses, variable speaking rates, weak articulations, and annotation noise, including zero-duration or near-zero-duration labels, can degrade alignment quality. These issues are more difficult for aligners that rely on a single global blank representation or purely local frame classification without explicit sequence-structure control or learned transition constraints.

- • Traditional forced aligners such as Montreal Forced Aligner (MFA)[30] rely on pronunciation lexicons and Kaldi-style triphone acoustic modeling with speaker adaptation. They remain strong and widely used baselines, especially in lexicon-rich settings. Their modeling assumptions, however, make it less direct to add task-specific neural representations, structured blank modeling, and learned transition preferences.
- • A related line of work treats alignment or segmentation as frame-level boundary classification or segmentation [42, 65], which is especially relevant for phone-level alignment and boundary-sensitive tasks. Boundary detection and frame-wise classification, however, do not by themselves define a globally consistent word-to-speech alignment path. Transcriptconditioned word-level alignment often needs additional mechanisms to enforce monotonic occupancy, represent heterogeneous gap states, and stabilize ambiguous cases.
- • CTC-based systems [37] and ASR-alignment pipelines such as WhisperX [5] obtain timestamps from implicit CTC paths or auxiliary alignment stages. This works well as an engineering pipeline, but pause regions, blank handling, and transition preferences are spread across separate components rather than learned in one transcript-conditioned objective.
- • Recent methods such as Canary [16] and Qwen3-Omni [49] use large-scale neural models to predict timestamps directly. These models are typically large and autoregressive, which can be expensive for large-scale offline processing and online lyric/subtitle alignment services. Concurrent work, Qwen3 Forced Aligner [32], proposes a non-autoregressive parallel slotfilling approach that also leverages multilingual semantic knowledge from pretrained large

language models. Its design concatenates audio features, text, and time slots into one sequence, so activation and memory cost scale with the joint sequence length.

We focus on transcript-conditioned word-level forced alignment, especially when downstream speech generation or annotation refinement requires accurate pause-aware boundaries. Swan Forced Aligner combines (i) an explicit interleaved word/blank state topology, (ii) structured decoding with calibrated unary and transition scores, and (iii) an optional posterior-based decoding mode for locally ambiguous evidence in noisy long-form speech segments.

Compared with direct timestamp prediction, our model maintains an explicit alignment lattice with monotonic structural constraints, making the decoding process more controllable, interpretable, and diagnosable. The model is also computationally efficient, with compact parameterization, modest activation memory, and low-latency Viterbi decoding. Compared with conventional frameclassification aligners, it models both state emissions and state transitions in one alignment framework, and supports topology-constrained posterior decoding under uncertain or weak acoustic evidence.

In long-form structured audio modeling, small local timing errors can accumulate into content drift, unstable conditioning, or mismatches across the generated audio. Accurate time structure is therefore a practical requirement rather than a cosmetic annotation detail [13, 58, 59]. These difficulties also motivate broader evaluation protocols, since perceived quality depends on frame-level fidelity, consistency, preference, and synchronization over longer temporal contexts [33, 66].

[Figure 103]

[Figure 104]

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

AuxiliaryEncoder

[Figure 105]

AudioEncoder

Speech

[Figure 106]

Insert Blank Joint Cross Attention

Text Encoder

喝 杯 co ffe 吗

喝杯coffe吗？

Figure 3: Overview of Swan Forced Aligner.

#### B Method

##### B.1 Problem Setup

Let x denote an input speech waveform and let y = (y1,...,yN) denote its transcript. Our goal is to estimate the temporal boundary of each word in the transcript, i.e., a sequence of word-level intervals

{(si,ei)}Mi=1,

- where M is the number of aligned lexical words, and si and ei denote, respectively, the start and end times of the i-th word on the input waveform time axis.

We focus on transcript-conditioned word-level forced alignment. The transcript is assumed to be given, so the main challenge is not lexical recognition but robust boundary localization under pauses,

speaking-rate variation, weak articulations, and annotation noise. Realistic supervision may contain uncertain boundaries, heterogeneous blank regions between adjacent words, and zero-duration labels.

For training, each utterance may optionally be associated with word-level annotations

###### {(ˆsi,eˆi)}Mi=1,

and, when available, a confidence score cˆi ∈ [0,1] for each word annotation. These annotations are used to construct frame-level occupancy targets and duration supervision.

- B.2 Lexical Word Representation The transcript is represented as a sequence of lexical word units,

g = (g1,...,gM),

where each gi denotes one word to be aligned. Each word is then tokenized by a predefined text tokenizer into one or more subword tokens. The tokenizer used by SwanVoice does not merge multiple lexical words into a single token, but it may split a word into multiple tokens, especially for English. For example, a Chinese character is typically mapped to one token, while an English word may be decomposed into several subword pieces.

This tokenizer behavior is convenient for text modeling, but it creates a granularity mismatch for word-level forced alignment: the alignment target is a lexical word, whereas the text encoder operates on tokenizer-level units. We bridge this gap by inserting a dedicated anchor symbol <|wbd|> after each lexical word. Denote the tokenizer output of gi as

B(gi) = (ti,1,...,ti,n

i

), The final token sequence is

y˜ = (t1,1,...,t1,n

1

,<|wbd|>,...,tM,1,...,tM,n

M

,<|wbd|>).

The special token <|wbd|> acts as a word-level alignment anchor. It aggregates the contextual information of the preceding subword span into one hidden state representing the lexical word. Swan Forced Aligner therefore aligns word-anchor states extracted from the contextualized hidden states at <|wbd|> positions, rather than aligning every subword token independently.

- B.3 Backbone Encoders A pretrained acoustic encoder maps the input waveform x to frame-level features:

###### A(0) = Encpreaud(x) ∈ RT×d

###### ,

a

- where T is the number of acoustic frames after subsampling and da is the hidden dimension of the pretrained encoder. A lightweight Transformer encoder refines these projected features:

A = Encaud(Projaud(A(0))) ∈ RT×d. This yields the frame-level acoustic sequence A = (a1,...,aT) used by the structured aligner. On the text side, the text encoder maps the tokenized sequence y˜ to contextualized representations: H = Enctext(Embed(˜y)) ∈ RL×d.

The acoustic and text streams are not fully independent. The backbone allows text-conditioned acoustic encoding and audio-conditioned text encoding, so the resulting representations already carry cross-modal alignment cues before structured decoding.

Gathering the hidden states at the <|wbd|> positions gives the word-level text-anchor sequence:

###### W = (w1,...,wM), wi ∈ Rd.

Each anchor wi summarizes the full token span associated with lexical word gi, including the case where that word is decomposed into multiple subword tokens. These word-anchor representations are used as the text-side word states in the structured aligner.

##### B.4 Structured Alignment Topology

Swan Forced Aligner performs alignment over an explicit interleaved word–blank topology rather than predicting timestamps directly from a flat sequence representation. For a transcript with M lexical words, the latent state space is

S = (b0,w1,b1,w2,...,wM,bM),

where wi denotes the i-th word state and bi denotes the blank or gap state before, between, or after words. This topology represents both word occupancy and the pauses, silences, and transitional blank regions that appear in real speech.

Each word state wi is represented by the corresponding word-anchor embedding from the text encoder. For blank states, Swan Forced Aligner avoids a single global blank representation and models heterogeneous blank regions explicitly. Separate learnable parameters are used for the utterance-initial blank state b0 and the utterance-final blank state bM. For an internal blank between adjacent words, the prototype is conditioned on both neighboring word states:

bi = bbase + ∆(wi,wi+1), 1 ≤ i ≤ M − 1,

where bbase is a global blank prototype and ∆(·,·) is a small neural module that predicts a gap-specific residual from the adjacent word-state pair. This allows the model to distinguish short coarticulatory gaps, long pauses, and phrase-level boundaries.

A valid alignment path is represented by a latent-state sequence:

z = (z1,...,zT), zt ∈ S,

The path follows the interleaved topology monotonically and uses three transition types:

stay, adv1, adv2.

Here, stay keeps the current state, adv1 advances by one state along the topology, and adv2 skips across an intermediate blank when transitioning into a word state. These transitions enforce monotonic decoding that remains consistent with the transcript.

##### B.5 State Scoring and Stability-Oriented Calibration

Given acoustic frame features A = (a1,...,aT) and state representations in S, the model computes a frame-level unary score for each valid frame–state pair. Let hs ∈ Rd denote the representation of state s. The raw unary score at frame t for state s is defined as

###### ut,s = ϕ(at,hs),

where ϕ(·,·) is either cosine similarity or dot-product similarity.

In addition to frame-level state evidence, the model scores transition preferences between neighboring states. Transition scores are parameterized by lightweight neural heads conditioned on the destination state, with an additional pairwise module for skip transitions into word states. The incoming transition score for destination state s and transition type r is

τ(s,r), r ∈ {stay,adv1,adv2}.

This allows the model to score which state is locally plausible at a frame and how likely different monotonic advances are under the current alignment context.

One practical goal of Swan Forced Aligner is stable structured decoding across machines and execution environments. In our experiments, even with deterministic controls enabled, small numerical differences can alter the decoded Viterbi path when unary and transition scores are poorly calibrated. Score canonicalization and decoupled scaling make decoding robust to numerical noise.

For unary scores, we perform per-frame canonicalization over valid states:

###### u˜t,s = Canonu(ut,s),

where the normalization is applied only over valid states at frame t. This removes sample-dependent score offset and scale variation and makes the relative ordering among candidate states more stable.

For transition scores, the same canonicalization is applied to all valid transition entries in the sample: τ˜(s,r) = Canonτ(τ(s,r)).

This reduces variation in transition magnitude across utterances and prevents the decoder from becoming overly sensitive to implementation-dependent score scales.

Finally, we use separate learnable gains for unary and transition terms:

u∗t,s = γu u˜t,s, τ∗(s,r) = γτ τ˜(s,r),

where γu and γτ are independent learnable parameters. This dual-gamma design is more flexible than a single global temperature because occupancy and transition terms require separate calibration.

The final score of a valid state sequence z = (z1,...,zT) is defined as

T

T

u∗t,z

τ∗(zt,rt),

Score(z) =

+

t

t=1

t=2

where rt denotes the transition type used to enter state zt from zt−1. The calibrated unary and transition terms define the final transcript-conditioned alignment score.

##### B.6 Training Objectives

During training, word-level time annotations are converted into frame-level state supervision over the interleaved topology. Frames assigned to lexical words are supervised by their corresponding word states, while the remaining valid frames are assigned to blank states according to their positions relative to neighboring words. This yields targets on the inference lattice.

The primary frame-level supervision is a cross-entropy alignment loss over valid acoustic frames:

1 |Ω| t∈Ω

Lce =

αt CE(pt,zˆt),

where Ω is the set of valid acoustic frames, zˆt is the target state at frame t, pt is the predicted state distribution, and αt is an optional frame weight derived from annotation confidence.

To encourage globally consistent alignment paths, Swan Forced Aligner also optimizes a CRF objective over the same structured lattice. Let Z denote the set of all valid monotonic state paths and let Score(z) denote the path score defined in the previous subsection. For a target path zˆ derived from word-level time annotations, we use the CRF loss

exp(Score(ˆz)) z∈Z exp(Score(z))

Lcrf = −log

.

This objective complements frame-level cross-entropy by encouraging the gold alignment path to receive a high global score relative to all other valid monotonic paths.

To regularize state occupancy, Swan Forced Aligner uses duration supervision for both word states and blank states. Let dˆ(iw) and dˆ(ib) denote the target durations of word and blank states, and let d(iw) and d(ib) denote the predicted occupancies obtained by summing state posteriors over time. The two duration terms are combined as

Ldur = L(durw) + λbL(durb) , where both terms combine absolute-error and log-duration penalties to stabilize supervision across short and long segments during training on heterogeneous speech.

Swan Forced Aligner also includes a monotonicity regularization term that penalizes decreases in the expected word index over time. This encourages the word-state posterior mass to progress monotonically along the transcript and discourages locally inconsistent alignments under ambiguous evidence. Denoting this term by Lmono, the final training objective is

L = Lce + Lcrf + λdLdur + λmLmono. In our implementation, all major loss terms are combined with unit weight unless otherwise specified.

##### B.7 Inference Procedure

At inference time, Swan Forced Aligner computes calibrated unary and transition scores over the interleaved alignment lattice. The same monotonic topology is used for training and decoding.

The default decoding mode is Viterbi decoding, which finds the highest-scoring valid state path

z∗ = arg max z∈Z

Score(z),

where Z denotes the set of all valid monotonic paths. The word-level start and end times are then recovered from the frame ranges assigned to each word state.

Swan Forced Aligner also supports an optional posterior-based decoding mode. In this mode, forward–backward inference first computes state posteriors on the same structured lattice, and a topology-constrained path is decoded using posterior scores instead of raw path scores. This mode is more robust when local evidence is ambiguous because it incorporates path uncertainty rather than relying only on a single maximum-score explanation.

After decoding, the confidence of each aligned word is estimated by aggregating emission-state probabilities over the frames assigned to that word state. Once a word-aligned frame span is determined by the decoded path, the confidence score is computed as the average word-state probability over that span. The decoded path may depend on the inference mode, while the confidence itself is still derived from emission-side state probabilities.

Together with the explicit state path, these state probabilities provide diagnostic signals for downstream debugging and alignment-error analysis.

#### C Experiments

##### C.1 Experimental Setup

Datasets For the forced-aligner experiments in this appendix, we use a separate 80K-hour ChineseEnglish alignment-training subset from internal resources. It spans audiobooks, podcasts, conversational speech, meetings, and live stream recordings. All training sets are pre-annotated with pseudo-timestamps using the Montreal Forced Aligner (MFA). For evaluation, we use two humantimestamped sets: the Chinese subset of GTSinger-Speech [55] and Librispeech-Alignment [1].

Implementation Details We use WavLM[6] as the pretrained audio encoder. The auxiliary encoder is a 4-layer bidirectional Transformer with hidden size 512 and 8 attention heads. The text encoder is a 16-layer bidirectional Transformer with hidden size 512 and 8 attention heads. The model has about 400M parameters. Swan Forced Aligner is trained on the 80K-hour subset using 24 A100 GPUs, with a batch size of 4 hours for 80K steps. We optimize with AdamW, using a learning rate of 1.0e-5 and β = (0.9,0.999).

Evaluation Metrics We evaluate timestamp prediction with accumulated averaging shift (AAS), following prior work [41]. Lower AAS indicates more accurate timestamp prediction. AAS measures the average boundary deviation across all evaluated word slots:

1 N

AAS =

N

1 N

∥si − sˆi∥1 =

i=1

N

(|t(starti) − tˆ(starti) | + |t(endi) − tˆ(endi)|), (14)

i=1

- where N is the number of evaluated word slots, si = (t(starti) ,t(endi)) is the ground-truth boundary pair, and sˆi = (tˆ(starti) ,tˆ(endi)) is the predicted boundary pair.

Baselines We compare with five mainstream forced aligners: (1) Monotonic-Aligner [41], a nonautoregressive Paraformer-based aligner using a continuous integrate-and-fire mechanism, which supports only Chinese 8. (2) NeMo Forced Aligner [37], a tool for generating token-, word-, and segment-level timestamps of speech in audio using NeMo’s CTC-based ASR models. We use

8https://modelscope.cn/models/iic/speech_timestamp_prediction-v1-16k-offline

the official checkpoint 9 to perform English alignment. (3) WhisperX [5], a time-accurate speech recognition system with word-level timestamps utilizing voice activity detection and forced phoneme alignment. We use different checkpoints for Chinese and English speech following the official inference script 10; (4) Qwen3 Forced Aligner [32], a non-autoregressive aligner based on parallel slot filling and multilingual speech-language representations. We perform alignment using their official checkpoint 11; (5) LattifAI Aligner, a speech agent for millisecond-precision alignment. We use their official SDK 12 and the released Lattice-1 checkpoint 13 across all evaluation runs.

##### C.2 Experimental Results

Table 3: AAS (ms)↓ of Swan Forced Aligner and other forced aligners on Chinese and English test datasets. The best results are in bold and the second best are underlined. * denotes checkpoints that were not publicly released at evaluation time.

GTSinger-Speech-ZH LibriSpeech-Clean LibriSpeech-Others Monotonic-Aligner 61.98 - -

NeMo Forced Aligner - 87.05 91.85

WhisperX 221.29 87.02 96.64 Qwen3 Forced Aligner 47.31 27.84 29.74

LattifAI Aligner* 31.60 25.70 36.00 Swan Forced Aligner 45.19 27.67 29.92

As shown in Table 3, Swan Forced Aligner gives the best open-source AAS on the Chinese and LibriSpeech-Clean benchmarks. On LibriSpeech-Others, it is within 0.18 ms of Qwen3 Forced Aligner and about 10 ms behind LattifAI Aligner, the best proprietary system in this comparison.

- 9https://ngc.nvidia.com/models/nvidia:stt_en_fastconformer_hybrid_large_pc
- 10https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
- 11https://github.com/QwenLM/Qwen3-ASR
- 12https://github.com/lattifai/lattifai-python
- 13https://huggingface.co/LattifAI/Lattice-1

