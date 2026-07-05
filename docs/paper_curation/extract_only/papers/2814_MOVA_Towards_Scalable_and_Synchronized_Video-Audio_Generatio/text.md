OpenMOSS

[Figure 1]

## MOVA: Towards Scalable and Synchronized Video–Audio Generation

SII-OpenMOSS Team*

# arXiv:2602.08794v2[cs.CV]10Feb2026

Abstract

Audio is indispensable for real-world video, yet generation models have largely overlooked audio components. Current approaches to producing audio-visual content often rely on cascaded pipelines, which increase cost, accumulate errors, and degrade overall quality. While systems such as Veo 3 and Sora 2 emphasize the value of simultaneous generation, joint multimodal modeling introduces unique challenges in architecture, data, and training. Moreover, the closed-source nature of existing systems limits progress in the field. In this work, we introduce MOVA (MOSS Video and Audio), an open-source model capable of generating high-quality, synchronized audio-visual content, including realistic lip-synced speech, environment-aware sound effects, and content-aligned music. MOVA employs a Mixture-of-Experts (MoE) architecture, with a total of 32B parameters, of which 18B are active during inference. It supports IT2VA (Image-Text to Video-Audio) generation task. By releasing the model weights and code, we aim to advance research and foster a vibrant community of creators. The released codebase features comprehensive support for efficient inference, LoRA fine-tuning, and prompt enhancement.

Blog: https://mosi.cn/models/mova Model: https://huggingface.co/collections/OpenMOSS-Team/mova Code: https://github.com/OpenMOSS/MOVA

### 1 Introduction

Video generation has long been a pivotal domain in multimodal generative modeling. Historically, limitations in model capacity, data volume, and scalability hindered these models from achieving practically viable performance. The emergence of the scalable Diffusion Transformer architecture [1] has changed this landscape, giving rise to a series of models-including Sora [2], OpenSora [3, 4], Wan [5], and LTX [6]—that can generate high-fidelity, realistic videos. Beyond basic video synthesis, these models also demonstrate advanced capabilities such as long video generation [7], controllable synthesis [8], and even few-shot learning and visual reasoning [9, 10].

However, traditional video generation models often neglect the audio component, despite its critical role in multimedia content. Producing videos with synchronized sound typically requires a cascaded pipeline[11], e.g., generating video first and then synthesizing audio based on the visuals, or vice versa. Such pipelines inherently limit generation quality, as the audio and video modalities do not interact during synthesis. Endto-end models like Veo3 [12] have demonstrated that high-fidelity audio-visual generation with precise syn-

∗Full contributors can be found in the Contributors section.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

- Figure 1 Overview of MOVA capabilities. MOVA generates synchronized video and audio across diverse scenarios: multi-speaker speech with precise lip synchronization in both English and Chinese, physical sound effects aligned with visual events, and scene text generation. The model supports both 16:9 and 9:16 aspect ratios.

chronization is possible, highlighting the importance of joint audio-video modeling. Similarly, Sora2 [13] exhibits impressive capabilities in generating synchronized audio and video. Yet, all these state-of-the-art systems are closed-source, and subsequent releases such as Wan2.6 [14], Kling Video 3,0 [15], and Seedance

- 2.0 [16] remain proprietary. As a result, the development of high-quality video-audio generation models remains underexplored in the research community.

Compared to video-only generation models, video-audio generation models introduce three major challenges: 1. Data Pipeline: Incorporating the audio modality requires a fine-grained audio-video captioning pipeline. 2. Modality Fusion: To achieve harmonious and synchronized video-audio generation, these two modalities must mutually integrate information during the generation process. However, simultaneous bimodal generation faces challenges due to disparity in native information density between the two modalities, as well as challenges regarding fusion mechanisms and efficiency. 3. Scalability Verification: Most existing open-source models are evaluated on small-scale architectures and limited datasets [17–20], often falling short in achieving high-quality results across both modalities. It remains an open question whether videoaudio models can sustain continuous performance improvements with larger datasets and model scales.

In this work, we present MOVA (MOSS Video and Audio), a video-audio generation foundation model capable of synthesizing multilingual speech with high-quality lip synchronization, as well as environmental sounds with precise audio-visual alignment. To achieve high-quality video-audio generation, we propose an asymmetric dual-tower architecture, combining a pre-trained video tower with a pre-trained audio tower. For modality fusion, we employ a bidirectional cross-attention mechanism that enables rich interaction between the two modalities. To unleash the potential of this architecture, we construct a fine-grained videoaudio annotation pipeline and scale up the training data, resulting in strong performance in synchronized video-audio generation. We observe consistent improvements in both lip synchronization and video-audio alignment metrics as training progresses. As illustrated in Figure 1, MOVA realizes scalable and synchronized video-audio generation. To support research and foster community creation, we release all model weights along with training, inference, and fine-tuning code.

Our contributions can be summarized as follows:

• We develop MOVA, a synchronized video-audio generation model with an asymmetric dual-tower

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

- Figure 2 Model Structure Overview. MOVA couples an A14B video DiT backbone and a 1.3B audio DiT backbone via a

- 2.6B bidirectional Bridge module.

architecture that leverages pre-trained video and audio generation models.

- • We design a fine-grained audio-video captioning pipeline to produce high-quality bimodal training data at scale.
- • By scaling video-audio training, we achieve continuous improvements in synchronization performance across both modalities.

### 2 Model Architecture

##### 2.1 Preliminaries

We consider joint generation of temporally aligned video-audio pairs from a text prompt (optionally with a first-frame image for I2VA). Let 𝑉 ∈ ℝ(1+𝑇𝑣)×𝐻×𝑊×3 be a video of 𝑇𝑣 + 1 frames and 𝐴 ∈ ℝ𝑇𝑎×1 be the corresponding 48kHz mono audio waveform. We adopt pretrained VAEs to define compact latent spaces: the Wan2.1 video VAE [5] compresses 𝑉 into a spatiotemporal latent 𝑥𝑣, and a DAC-style audio VAE from HunyuanVideo-Foley [21] encodes 𝐴 into an audio latent 𝑥𝑎. All subsequent modules operate in these latent spaces. Given a condition 𝑐, our goal is to generate synchronized (𝑥𝑣,𝑥𝑎).

Our training objective follows the flow matching method [22, 23]. At each timestep 𝑡,

𝑣 𝑣,

𝑣 𝑡

𝑥𝑡𝑎 = (1 − 𝑡)𝑥𝑎 + 𝑡𝜀𝑎, (1) where 𝜀𝑣,𝜀𝑎 ∼ 𝒩(0,𝐼) and 𝑡 ∼ 𝒰(0,1).

The target velocity is given by the derivative of the interpolation path:

d𝑥𝑡𝑣 d𝑡

𝑣𝑡𝑣 ∶=

= 𝜀𝑣 − 𝑥𝑣,

d𝑥𝑡𝑎 d𝑡

𝑣𝑡𝑎 ∶=

= 𝜀𝑎 − 𝑥𝑎. (2)

The training loss is defined as a standard flow matching objective:

ℒFM = 𝔼𝑡,𝑐,𝑥𝑣,𝑥𝑎,𝜀[𝜆𝑣 ∥𝑣̂𝜃𝑣(𝑥𝑡𝑣,𝑥𝑡𝑎,𝑡,𝑐) − 𝑣𝑡𝑣∥2 + 𝜆𝑎 ∥𝑣̂𝜃𝑎(𝑥𝑡𝑣,𝑥𝑡𝑎,𝑡,𝑐) − 𝑣𝑡𝑎∥2 ], (3)

where 𝜃 denotes the model parameters, 𝑣̂𝜃 denotes the predicted cross-modal velocity field, 𝑐 denotes the conditioning signal (text, optionally augmented with an input image), and 𝜆𝑣,𝜆𝑎 are loss weights that balance the video- and audio-velocity regression terms, respectively.

- 2.2 Dual-Tower Diffusion Transformer

Our goal is to leverage powerful pretrained single-modality diffusion models and achieve synchronized video–audio generation with minimal additional cost. Specifically, we adopt Wan2.2 I2V A14B [5] as the video backbone (for image+text conditioned I2VA) and a 1.3B text-to-audio diffusion model with a Wan2.1style architecture as the audio backbone. We couple these two backbones through a lightweight dual-tower conditional Bridge, enabling bidirectional information exchange while keeping the core towers intact.

Bridge Module. The bridge operates at the hidden-state level of the two DiT backbones. At each interaction layer, two cross-attention blocks are added: one injects video hidden states into the audio DiT. and one injects audio hidden states into the video DiT.

Aligned RoPE. Video and audio latents live on different temporal grids: video latents are relatively coarse (due to temporal downsampling in the video VAE), while audio latents are much denser. If we apply crossattention with naive positional indices, tokens that represent the same physical time can be assigned to different positional slots (and conversely different physical times can share nearby indices), so queries and keys may correspond to mismatched physical times, causing audio–visual drift; this positional mismatch also breaks the translation invariance of the interaction process.

Similarly to MMAudio [24] and OVI [18], we modify standard RoPE [25] to align the two modalities on the same time grid.:

q̃(𝑚) = 𝑅(𝜃𝑝

𝑚

)q(𝑚), k̃(𝑚) = 𝑅(𝜃𝑝

𝑚

)k(𝑚), 𝑅(𝜙) = [

cos𝜙 −sin𝜙 sin𝜙 cos𝜙

].

Let 𝑓𝑣 and 𝑓𝑎 denote the latent frame rates of video and audio after their respective VAEs. We map video indices into the audio time units by scaling the temporal position with the ratio 𝑠 = 𝑓𝑎/𝑓𝑣:

𝑝𝑣(𝑖) = 𝑠 ⋅ 𝑖, 𝑝𝑎(𝑗) = 𝑗. This puts video and audio tokens on the same temporal scale.

- 3 Data Engineering

- 3.1 Overview

To transform heterogeneous and noisy raw videos into reliable resources for video-audio generation, we developasystematicdatacurationpipeline. Thepipelinefirstcleansandstandardizestheinputs, followedby

#### Stage 1: Video Preprocessing

#### Stage 3: Audio-Visual Captioning

#### Stage 2: Audio-Visual Quality Assessment

Standardization

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

speech

[Figure 30]

Audio Quality (Signal & Aesthetic)

Qwen3-Omni

Pad (9:16/16:9)

###### Resize

###### Resample

Singlemodality Caption

(720p)

(24fps)

nonspeech

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Detection

visual

###### Video Quality

MiMo-VL

[Figure 37]

[Figure 38]

(Technical & Aesthetic)

Voice Activation Scene Transition

[Figure 39]

|LLM Merging| |
|---|---|
| | |

(Silero VAD) (PySceneDetect)

[Figure 40]

[Figure 41]

[Figure 42]

GPT-OSS

[Figure 43]

Integrated Caption

[Figure 44]

A-V Alignment (Temporal & Semantic)

Segmentation

(8.05s)

[Figure 45]

[Figure 46]

[Figure 47]

High-quality videos with semantic-rich captions

Corrupted / Silent Low-quality / A-V Misaligned

- Figure 3 Data curation overview. Our data pipeline consists of three stages. In the first stage, we preprocess the raw data into fixed-length clips with a resolution of 720p, a frame rate of 24fps, and a duration of 8.05s. In the second stage, we filter these clips based on audio quality, video quality, and audio-visual alignment to obtain high-quality, synchronized clips. In the third stage, we utilize Qwen3-Omni and MiMo-VL to label the audio and visual information within the videos, respectively, and finally use GPT-OSS to merge these single-modality captions. Through our data pipeline, we have successfully curated high-quality audio-visual content along with corresponding, semantically rich captions.

the annotation of high-quality, richly-detailed captions across multiple modalities. By carefully structuring the data pipeline into three stages, we progressively filter out low-quality samples to retain high-fidelity data characterized by strong audio-visual consistency and coherent semantic labels.

Wecurateahigh-qualitysubsetofaudio-visualdatafromvariouspublicdatasets(datacollection, Section3.2). Our collection spans multiple video formats (e.g., movies, vlogs, and animations) and diverse themes ranging from education and sports to animation and interviews. The three-stage pipeline then processes the raw video data. In the first stage (video preprocessing, Section 3.3), we normalize aspect ratios, resize videos, resample streams, and segment videos into fixed-length clips based on VAD and scene transition annotations. In the second stage (audio-visual quality assessment, Section 3.4), we evaluate audio quality, video quality, and audio-visual alignment, retaining only clips that are both high-fidelity and well-aligned. In the third stage (audio-visual captioning, Section 3.5), we design structured and instructive prompts, generate modality-specific captions with MLLMs, and integrate them into unified annotations using a LLM. Our data pipeline is presented in Figure 3 and retention ratio is shown in Table 1.

Table 1 Retention Ratio of Total Dataset Duration

Raw Stage 1 (speech+nonspeech) Stage 1 (speech only) Stage 2

Retention Ratio (Relative to Raw Video)

100% 84.57% 58.75% 26.39%

##### 3.2 Data Collection

We use filtered HQ subsets of the following public video datasets in our work: VGGSound [26], AutoReCap [27], ChronoMagic-Pro [28], ACAV-100M [29], OpenHumanVid [30], SpeakerVid-5M [31], and OpenVid-

- 1M [32]. In addition, we utilize a large amount of in-house data. Our dataset encompasses a broad spectrum of domains (education, sports, and beauty, news, etc.), providing the distributional diversity necessary to enhance the model’s generalization across complex real-world scenarios.

##### 3.3 Video Preprocessing

We design and implement a scalable video preprocessing pipeline based on the Ray distributed framework, balancing data quality and processing efficiency. Initially, raw video data is filtered to remove samples with decoding failures or missing valid audio channels. Videos with unconventional container or encoding formats are remuxed or transcoded, respectively. Then, the pipeline generates segmentation metadata through three sequential steps:

- • Core Content Normalization: The FFmpeg cropdetect filter is applied to detect blank areas and retain the core visual content. The main content is then centered, resized to a 720p resolution, and symmetrically padded with black borders (pillarboxing or letterboxing) as necessary to conform to a 9:16 or 16:9 aspect ratio.
- • Voice Activity Detection: The audio track is extracted from each video and analyzed by the Silero Voice Activity Detection (VAD) [33] model to identify speech and non-speech intervals.
- • Scene Transition Analysis: PySceneDetect is employed to detect and record the timestamps of scene change points throughout the video.

By integrating the temporal information from VAD and scene transition analysis, we generate four types of fixed-duration (8.05-second) video segments: single-scene speech, single-scene non-speech, multi-scene speech, and multi-scene non-speech. This duration corresponds precisely to 193 video frames at 24 fps, calculated as the initial frame plus 8 seconds of video (1 + 8 × 24). For speech segments, the start time is adaptively adjusted to avoid truncating ongoing speech and ensure spoken-content continuity; the detailed algorithm is provided in Section A.3. Ultimately, only speech segments are selected for training, accounting for 69.47% of all preprocessed segments.

##### 3.4 Audio-Visual Quality Assessment

We conduct data quality assessment along three main dimensions: audio quality, video quality, and audiovisual alignment.

- • Audio quality: We compute signal-level metrics such as silence ratio and bandwidth, and further evaluate both signal and aesthetic aspects using the Audiobox-aesthetics audio quality assessment tool [34].
- • Video quality: We apply the DOVER video quality assessment tool [35] to assess the video from both technical and aesthetic perspectives.
- • Audio-visual alignment: We employ SynchFormer [36] to compute the temporal audio-visual synchronization of each video, and use ImageBind [37] to evaluate semantic audio-visual alignment.

To determine the filtering thresholds, we manually inspect the videos retained under different metric cutoffs and set reasonable thresholds for each dimension accordingly. In addition, we apply an audio classification model [38] to categorize audio and construct speech/non-speech subsets depending on the target capability (e.g., lip synchronization vs. general foley/ambience modeling). We provide the detailed filtering thresholds in Section A.4.

##### 3.5 Audio-Visual Captioning

We employ open-source models for audio-visual captioning and subsequently use a large language model (LLM) to merge the generated captions into coherent natural language descriptions, utilizing both NVIDIA GPUs and Ascend NPUs.

Bimodality Model. To annotate the filtered audio-visual contents, we employ distinct pipelines for video and audio modalities. For video annotation, we utilize the MiMo-VL-7B-RL model [39] to generate video descriptions, with explicit instructions focusing on video scene transitions. For audio annotation, we implement a dual-model strategy to separately handle speech and non-speech components. Specifically, Qwen3Omni-Instruct [40] was used for speech transcription, while Qwen3-Omni-Captioner [40] was applied to generate captions for non-speech sound and music. We then integrate these two subsets of annotations. This joint annotation strategy enables comprehensive coverage of both linguistic content and acoustic characteristics, reducing information loss and capturing multi-aspect audio semantics.

Caption Merging. For annotations generated by the separate modality pipelines, our primary goal is to integrate the content from the visual and audio dimensions. We employ the GPT-OSS-120B model [41] to merge the video captions with the aggregated audio annotations (comprising both speech and non-speech elements) while performing a consistency check to resolve potential cross-modal conflicts. Specifically, the model verifiesthealignmentbetweenvisual scenes and audio events toresolvepotentialconflicts. It then synthesizes these inputs into a cohesive, natural language description that seamlessly blends visual information with audio details, ensuring the final output is contextually unified and suitable for the target application. All prompts of our caption process and a detailed caption example are provided in Section A.5.

### 4 Training Strategy

##### 4.1 Overview

Our training pipeline consists of two stages, as illustrated in Figure 4. We initialize the video tower and video VAE from Wan2.2 weights [5], and adopt the audio VAE from HunyuanVideo-Foley [21]; both VAEs remain frozen throughout training. In the first stage (audio tower pretraining, Section 4.2), we train a standalone 1.3B text-to-audio DiT on diverse audio data covering music, general sounds, and speech. In the second stage (joint training, Section 4.4), the pretrained video tower (A14B), audio tower, and a randomly initialized Bridge module are optimized together with heterogeneous learning rates. This joint training further proceeds through three phases with progressively refined data and resolution (Section 4.3): Phase 1 (360p, diverse data), Phase 2 (360p, quality-filtered data), and Phase 3 (720p, highest-quality data).

##### 4.2 Audio Tower Training

Overview. To aligned with the video tower, we train an audio tower with the same architecture of Wan2.11.3B backbone. Relative to the original Wan2.1 setup, we replace Wan2.1’s 3D positional encoding over (𝑓,ℎ,𝑤) with a 1D positional encoding along the temporal axis. All other components and depth remain unchanged to maximally reuse engineering and training practices.

Training Data. The text-to-audio model is trained on three domains: (i) general sounds drawn from WavCaps and VGGSound [26, 42]; (ii) music from JamendoMaxCaps [43]; and (iii) our in-house TTS data. We train on fixed-length clips. Each clip is paired with a text prompt, which also carries an explicit duration token to control target length.

Evaluation Metrics. We assess the capability of our audio tower through various robustness metrics and compare with other size-matched baselines. Specifically, we consider fidelity, diversity, semantic alignment, and perceptual quality, which provide an overall view of audio generation performance. For fidelity, we use the Fréchet Distance (FD) with OpenL3 Embeddings [44], which measures how closely the generated

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

Figure 4 Training pipeline overview. (a) Audio tower pretraining: We train a 1.3B text-to-audio model with Wan2.1style architecture on music, general sounds, and TTS data. The audio VAE remains frozen during this stage. (b) Synchronous joint training: The video tower (A14B, blue) and audio tower (1.3B, orange) are connected via bidirectional Bridge cross-attention modules. (c) Video and audio timesteps are sampled independently, allowing each modality to follow its own noise schedule. (d) Bridge modules use a higher learning rate (𝜂br = 2 × 10−5) than backbone DiT blocks (𝜂b = 1×10−5) to accelerate cross-modal alignment while preserving pretrained priors. Both VAEs remain frozen throughout training.

audio matches real audio in distribution. Diversity is quantified using the Inception Score (IS)[45] which evaluates whether the model produces both varied and high-quality samples. Furthermore, robustness is captured by KL divergence with PaSST[46] in order to assess the stability of generated audio across different conditions. To assess the semantic consistency, we compute the CLAP score[47] to measure alignment between the conditioning text prompt and the generated audio. Beyond four metrics mentioned above, we incorporate perceptual evaluation using the AudioBox Aesthetic Benchmark[48]. The benchmark reports four human—preference-oriented indicators: aesthetics, naturalness, intelligibility and consistency. These dimensions capture overall appeal, absence of audible artifacts, clarity of linguistic or musical content, and the temporal coherence across the entire audio clip respectively.

Results. As shown in Table 2, our model achieves competitive performance on the AudioCaps [49] benchmark. It attains a strong IS of 10.54, clearly higher than AudioLDM2 (7.79), while maintaining a competitive CLAP score of 0.463. In terms of FD openl3, our result (72.25) is nearly identical to AudioLDM2 [50, 51] (72.04) and much better than TangoFLUX [52] (80.47), indicating stronger semantic fidelity. The KL divergence (1.47) also improves upon AudioLDM2 (1.66), suggesting better distribution alignment.

In Table 3, our method achieves the best results on Consistency (CU = 5.56) and Perceptual Quality (PQ = 6.20), surpassing advanced baselines such as AudioLDM2 and Tango2 [53]. Although it does not lead in CE or PC, the scores remain competitive. All these results demonstrate that our audio tower provides improvements in fidelity and coherence, while enhancing perceptual quality and semantic alignment compared to existing approaches.

- Table 2 Text-to-audio performance benchmark with AudioCaps Dataset. We report parameter size, number of function evaluations (NFE), Fréchet Distance with Openl3 embeddings (FD openl3), KL divergence with PaSST (KL passt), CLAP score, and Inception Score (IS).

Model Params NFE FD openl3↓ KL passt↓ CLAP score↑ IS↑ TangoFLUX [52] 516M 50 80.47 1.02 0.546 13.28 AudioLDM2 [51] 346M 200 72.04 1.66 0.409 7.79 Ours 1.3B 100 72.25 1.47 0.463 10.54

- Table 3 AudioBox Benchmark results. CE, CU, PC, and PQ denote the four evaluation indicators from AudioBox. Best results are highlighted in bold.

Model CE CU PC PQ AudioLDM [50] 3.27 5.10 3.23 5.82 AudioLDM2 [53] 3.48 5.54 3.00 6.09 Make-An-Audio 2 [54] 3.23 4.98 3.17 5.58 Tango 2 [53] 3.47 5.20 3.84 5.89 TangoFLUX [52] 3.54 5.07 3.64 5.78 Ours 3.41 5.56 3.04 6.20

##### 4.3 Progressive Joint Training

Our joint training follows a three-phase data and resolution curriculum, progressively refining both data quality and output resolution.

- Phase 1 (360p baseline): We initialize from pretrained Wan2.2 A14B (video) and a 1.3B audio tower, inserting the Bridge module with random initialization. Training proceeds at 360×640 resolution for 193 frames (8 seconds at 24 fps) on 1024 GPUs. We train on approximately 61,500 hours of diverse video-audio data drawn from SpeakerVid5M, Chinese drama, cartoon, movies, YouTube, and OpenHumanVid. We use asymmetric sigma-shift values with shift𝑣 = 5.0 and shift𝑎 = 1.0 to focus video learning on aggressive denoising while

keeping audio transitions smoother, and employ aggressive text dropout (𝑝textdrop = 0.5) to force Bridge-based alignment learning. This phase runs for 1 epoch (15 days).

- Phase 2 (quality-filtered alignment): We refine the noise schedule by aligning audio to match video, setting

shift𝑎 = 5.0 to match the video schedule. While Phase 1 uses a smoother audio schedule for stability, we find that high-fidelity timbre is sensitive to the noise schedule and denoising steps. Therefore, in Phase 2 we align

the audio sigma shift to the video setting (shift𝑎 = 5.0) to strengthen audio denoising and improve timbre fidelity. This alignment enables the audio tower to benefit from the same aggressive noise schedule that improves video quality, without requiring architectural changes—only the sigma-shift parameter is updated. We also reduce text dropout to 𝑝textdrop = 0.2 to allow text-guided refinement while retaining learned crossmodal priors, and introduce LUFS normalization to mitigate CFG-induced loudness explosion. This phase trains on approximately 37,600 hours of quality-filtered data for 1 epoch (7 days). To improve training quality, we curate the Phase 2 dataset using three complementary filters. First, we use OCR to identify videos without burned-in subtitles, retaining ∼9.5M clips and appending “This video has no subtitles.” to their prompts to teach the model this distinction. Second, we retain videos with LSE-D ≤ 9.5 and LSE-C ≥ 4.5, yielding ∼2.5M clips with high-quality lip-audio correspondence. Third, we apply DOVER technical quality score > 0.15 to select videos with superior visual fidelity, yielding ∼2.4M clips. The resulting dataset contains 16.8M clips (∼37,600 hours), balancing scale and quality.

- Phase 3 (720p fine-tuning): We upscale to 720×1280 resolution, training on approximately 11,000 hours of the

720p highest-quality subset (DOVER technical score > 0.14). The increased sequence length requires modified parallelism configuration: we increase context parallelism from CP=8 to CP=16. Checkpoint frequency increases to every 2000 steps to capture rapid convergence. This phase runs for 1 epoch (20 days).

Computational Resources. All three phases run on 1024 GPUs (128 nodes, 8 GPUs per node). For 360p training (Phases 1–2), we use CP=8, yielding effective batch size 128. For 720p fine-tuning (Phase 3), increased sequence length requires CP=16, reducing effective batch size to 64. The complete training spans 42 days, totaling approximately 43,000 GPU-days. See Appendix A.1 for the complete hyperparameter configuration.

##### 4.4 Optimization Details

Throughout all three phases, we optimize the full model end-to-end: the Bridge module and both pretrained towers are updated jointly from the first step. This differs from a two-stage warm-start that first trains the Bridge with frozen towers and then fine-tunes the full model. In our early experiments, the two-stage scheme reached an early performance plateau, which motivated end-to-end joint optimization. The main tension is that the Bridge must learn cross-modal correspondence quickly, while the large pretrained towers should remain stable and preserve their strong unimodal priors. We mitigate this tension with heterogeneous learning rates across module groups.

Heterogeneous Learning Rates. To balance fast Bridge convergence with tower stability, we use a higher learning rate for the Bridge (𝜂𝑏𝑟 = 2 × 10−5) than for the backbone towers (𝜂𝑏 = 1 × 10−5). This factor-of-two difference accelerates Bridge learning while reducing forgetting in the pretrained towers. In our experiments, a uniform learning rate either destabilizes the towers (if too high) or leaves the Bridge under-trained (if too low).

Dual Sigma Shift. Conventional joint diffusion that forces video and audio to share the same timestep can be suboptimal. The two modalities have different effective complexities: audio uses fewer tokens per second, yet timbre fidelity is highly sensitive to noise schedules. A single noise level may be too aggressive for one modality and too mild for the other, causing imbalanced gradients.

To enable this, we decouple the timestep sampling for each modality. During training, we independently draw 𝑡𝑣 and 𝑡𝑎 from 𝒰(0,1):

𝑡𝑣 𝑣 𝑣(𝑡𝑣 0𝑣 𝑣(𝑡𝑣 𝑣,

𝑧𝑡𝑎𝑎 = (1 − 𝜎𝑎(𝑡𝑎)) ⋅ 𝑧0𝑎 + 𝜎𝑎(𝑡𝑎) ⋅ 𝜖𝑎,

(4)

𝑚+𝑡(1−shift𝑚) controls the noise schedule for modality 𝑚 ∈ {𝑣,𝑎}.

where 𝜎𝑚(𝑡) = shift shift𝑚⋅𝑡

This decoupling provides two benefits. First, each modality follows its natural denoising trajectory—we set shift𝑣 = 5.0 (aggressive) and shift𝑎 = 1.0 (gradual) in Phase 1, then align them to shift𝑎 = 5.0 in Phase 2 to improve timbre fidelity. Second, noise schedules can be adjusted at inference time without retraining.

##### 4.5 Training Efficiency

For large-scale training, we shard model parameters with Fully Sharded Data Parallel (FSDP) [55] and adopt sequence parallelism via USP [56], achieving approximately 35% MFU. To eliminate redundant VAE computation introduced by sequence parallelism, we follow the approach in Wan [5]: for each CP group, the input preprocessing (primarily the VAE) is performed only once per CP step, and the preprocessed features are then broadcast from a designated rank to all other ranks within the same CP group before being fed into the DiT backbone, thereby avoiding duplicated computation. We use manual memory management to avoid the same Python’s garbage collection (GC) overhead reported in OpenSora2 [4]. We also port our training stack to Ascend NPUs and apply operator fusion for attention kernels, tensor layout transforms, and rotary embedding computation to reduce framework overhead. Under an 8-device configuration (CP=4, DP-shard=2), we measure 34.1 s/step on 8× Ascend 910A2; benchmark details are provided in Appendix A.2.

Because standard FSDP requires a consistent computation graph, for the A14B MoE video tower we adopt an alternating optimization strategy: on odd steps we sample high-noise timesteps for all samples and optimize the high-noise DiT, while on even steps we sample low-noise timesteps and optimize the low-noise DiT. In addition, the shared bridge and the audio tower are optimized at every step.

### 5 Inference

##### 5.1 Dual Classifier-Free Guidance

In the text-conditioned video-audio generation (T2VA) setting, we may view a paired video-audio sample as a single joint latent, where the only explicit condition is text. However, from a single-modality perspective, the other modality provides additional conditional information: when predicting video, audio serves as a condition, and vice versa. This perspective introduces extra controllability, because we can separately adjust the guidance strengths for text conditioning and cross-modal conditioning.

Following the dual Classifier-Free Guidance (dual CFG) proposed in InstructPix2Pix [57], we adapt CFG to joint audio-video models with two conditioning sources: the textual prompt 𝑐𝑇 and the cross-modal information induced by Bridge interactions 𝑐𝐵. We derive a principled dual CFG formulation by decomposing the joint posterior via Bayes’ rule:

𝑃(𝑐𝑇 ∣ 𝑐𝐵,𝑧)𝑃(𝑐𝐵 ∣ 𝑧)𝑃(𝑧) 𝑃(𝑐𝑇,𝑐𝐵)

𝑃(𝑧 ∣ 𝑐𝑇,𝑐𝐵) =

. (5)

Taking the score function (gradient of log-likelihood) and applying CFG scaling leads to:

𝑣̃𝜃 = 𝑣𝜃(𝑧𝑡,∅,∅)

𝐵 𝜃(𝑧𝑡,∅,𝑐𝐵 𝜃(𝑧𝑡,∅,∅)] + 𝑠𝑇 ⋅ [𝑣𝜃(𝑧𝑡,𝑐𝑇,𝑐𝐵) − 𝑣𝜃(𝑧𝑡,∅,𝑐𝐵)],

(6)

where ∅ denotes null conditioning, 𝑣𝜃(𝑧𝑡,𝑐𝑇,𝑐𝐵) is the model’s velocity prediction with both text and Bridge active, and 𝑣𝜃(𝑧𝑡,∅,∅) disables both (the “no-bridge” mode disables cross-modal injection).

By tuning 𝑠𝐵 and 𝑠𝑇, we control the trade-off between alignment and perceptual quality. In the most general setting, we use Dual CFG with three function evaluations per step (NFE= 3), and the two common special cases below reduce to NFE= 2:

- • Dual CFG (general) (𝑠𝐵 = 𝑠𝑏, 𝑠𝑇 = 𝑠𝑡): The full formulation that independently scales (i) modalityalignment guidance through the Bridge term and (ii) text guidance. This provides the most flexible control over the alignment–quality trade-off, at the cost of one additional model call (NFE= 3).
- • Text-only CFG (𝑠𝐵 = 1, 𝑠𝑇 = 𝑠): Standard formulation. Bridge remains active in both branches, so guidance does not explicitly amplify cross-modal alignment. Yields high semantic fidelity (e.g., ImageBind scores) but weaker temporal sync (higher DeSync). This is a two-branch guidance scheme (NFE= 2).
- • Text + modality CFG (𝑠𝐵 = 𝑠, 𝑠𝑇 = 𝑠): The unconditional branch disables Bridge injection, isolating the alignment signal. Produces stronger synchronization (lower DeSync, better lip-sync). This also uses two branches (NFE= 2).

##### 5.2 Generation Workflow

To address diverse user input formats and maintain style consistency with training data, which, in our opinion, can incentivize model’s full potential. Given a user-provided initial frame and a text prompt, our workflow generates coherent video and audio, as illustrated in Figure 5. The primary objective of this pipeline is prompt enhancement: rather than directly using raw user inputs which often lack descriptive detail, we

[Figure 77]

Figure 5 The overall workflow of MOVA for text-image and text-only to video-audio generation.

refine them to incentivize the model’s full generative potential. We observe that maintaining high performance is closely tied to the quality of the prompt; therefore, our multi-stage conditioning pipeline explicitly extracts visual grounding from the reference image and synthesizes an enriched narrative. This ensures that the final synthesis not only preserves the style, lighting, and cinematography established in the first frame but also aligns with the sophisticated data distribution the model was trained on.

Our pipeline consists of three primary stages. First, we utilize Qwen3-VL [58], a vision-language model, to extract a structured visual description. This extraction is guided by a curated prompt that constrains the model to four essential categories: (i) visual style, including color palette and lighting; (ii) cinematography, covering shot size, framing, and composition; (iii) visual elements, comprising subjects and their spatial relations; and (iv) OCR text, preserved exactly as it appears. This structured representation serves as an intermediate grounding that bridges the gap between the static image and the dynamic video.

Second, we synthesize a video generation prompt using LLMs (e.g. Gemini 2.5 Pro [59], conditioned on both the input text and the extracted visual description. The core design principle is to preserve the static attributes from the visual description while incorporating the temporal dynamics specified by the user text. We employ in-context learning [60] to ensure the generated prompt follows the narrative style and architecture of the training data.

Finally, MOVA generates video content by leveraging both the synthesized prompt and the initial frame as dual conditioning. This mechanism integrates narrative descriptions with visual grounding to ensure temporal synchronization while adhering to established visual priors. Furthermore, the versatility of our workflow allows for text-to-video-audio generation using a text prompt and an uninformative white image as input, thereby demonstrating MOVA’s capacity for high-quality, zero-shot video synthesis.

### 6 Evaluation

##### 6.1 Experiment Setup

Benchmarks. In this work, we use two benchmarks to test the model’s video generation capabilities. We adopt Verse-Bench [17], which consists of 600 image–text prompt pairs. To facilitate joint video–audio generation, we employ GPT-5 [61] to unify visual and audio descriptions into a single, cohesive prompt. While Verse-Bench [17] provides a large-scale collection of image–text prompts, it is not specifically designed for evaluating video–audio generation in various scenes. To further evaluate joint video–audio generation in

Figure 6 Dataset overview. (a) Category distribution of samples in the dataset, illustrating the relative proportions of different image categories. (b) Statistical distribution of prompt lengths.

realistic and challenging settings, we construct a dedicated evaluation benchmark covering diverse video generation scenarios. The benchmark is designed to assess capabilities that are critical for joint video–audio modeling, including temporal coherence, audio–visual synchronization, multi-character interaction, and dynamic scene evolution. Unlike Verse-Bench, which provides broad visual coverage, our benchmark adopts a finer-grained scenario taxonomy and explicitly categorizes samples into six representative video generation types, including multi-speaker interaction, movie-style narratives, sports competitions, game livestreams, camera motion sequences, and anime-style content. Detailed benchmark construction procedures and category descriptions are provided in Appendix A.6.

Evaluation metrics. We evaluate our method through both objective benchmarks and subjective human evaluations.

- • Objective evaluation: For objective evaluation, we report performance on Verse-Bench across several dimensions. We measure acoustic fidelity and diversity using the Inception Score (IS) computed with a PANNs [62] classifier, and assess speech quality with DNSMOS [63]. Cross-modal semantic alignment is quantified using the ImageBind score (IB-Score) [37] computed on joint audio-video embeddings. Temporal consistency is measured by the DeSync score predicted by Synchformer [36]. Furthermore, to rigorously evaluate lip-sync precision, we include the Lip Sync Error - Confidence (LSE-C) and Distance (LSE-D) metrics derived from SyncNet [64]. Additionally, we evaluate the concatenated minimum permutation character error rate (cpCER) score on the multi-speaker subset of our constructed benchmark. This metric is designed to assess whether the speaker identities and speaking content are correctly reflected in the generated outputs. To evaluate speaker timbre and dialogue content, we employ the MOSS Transcribe Diarize [65]. This model performs speaker diarization using explicit speaker tags such as [S01] and [S02], followed by automatic speech recognition that transcribes the corresponding spoken content for each identified speaker.
- • Subjective Evaluation: We conduct an arena-style preference study to evaluate human perception. The evaluation set comprises 732 samples, including 600 from Verse-Bench and 132 from our newly introduced benchmark, where half of the originally English-only Verse-Bench speech data was manually translated to construct a bilingual mix. For each comparison, participants are tasked with selecting the superior video across five dimensions: (i) prompt adherence, (ii) visual-audio synchrony, (iii) lip-sync accuracy, (iv) video quality, and (v) audio-speech fidelity. A standard ELO rating system is employed to compute model rankings based on pairwise human preference judgments. Following the official Chat-

- Table 4 Quantitative comparison of audio-visual generation performance on Verse-Bench. IS and AV-Align metrics are evaluated on all Verse-Bench subsets; DNSMOS and Lip Sync metrics are evaluated on Verse-Bench set3; ASR Acc is evaluated on the multi-speaker subset. Bold and underlined values denote the best and second-best results, respectively.

###### Model 𝑠𝐵

Audio-Speech AV-Align Lip Sync ASR Acc IS↑ DNSMOS↑ DeSync↓ IB-Score↑ LSE-D↓ LSE-C↑ cpCER↓

LTX-2 [66] - 3.066 3.635 0.451 0.213 7.261 6.109 0.220 Ovi [18] - 3.680 3.516 0.515 0.190 7.468 6.378 0.436 WAN2.1 + MMAudio - 4.036 – 0.260 0.317 – – –

MOVA-360p 1.0 4.269 3.797 0.475 0.286 8.098 6.278 0.177

w/ dual CFG 3.5 4.169 3.674 0.351 0.315 7.004 7.800 0.247 MOVA-720p 1.0 3.936 3.671 0.485 0.277 8.048 6.593 0.149

w/ dual CFG 3.5 3.814 3.751 0.370 0.297 7.094 7.452 0.218

bot Arena implementation 2, we set the initial ELO rating to 1000 with a K-factor of 4. The logistic scale and base are configured at 400 and 10, respectively. To ensure statistical robustness, we report the results using 1000 bootstrap iterations.

Baselines. We compare MOVA with three baseline models under a unified protocol. Specifically, the baselines span two paradigms: (i) synchronous audio–visual generators that produce video and speech jointly: LTX-2 [66] and Ovi [18]. (ii) a cascaded pipeline formed by coupling WAN2.1 [5] for video with MMAudio [24] for audio. For fair comparison, we standardize the spatial resolution at 720p and adopt the recommended configurations for all baseline models.

- • Ovi [18] is a single-stage audio–video generation model that jointly models both audio and video modalities within a single process. By employing two architecturally DiTs for audio and video, Ovi achieves natural audiovisual synchronization without relying on separate pipelines or post hoc alignment. In the fusion blocks, OVI uses a frozen T5 encoder to integrate the video model with a pretrained audio model capable of generating both speech and environmental sounds. This shared semantic conditioning allows the audio and video branches to be guided by the same semantic context, strengthening cross-modal coherence and synchronization.
- • LTX-2 [66] introduces an efficient audiovisual generation model that, like OVI, produces video and its synchronized audio within a single diffusion process. Unlike OVI, LTX-2 adopts an asymmetric dualstream Transformer architecture, enabling temporal alignment through bidirectional attention across all layers. The model employs modality-specific VAEs and positional encodings for audio and video, which preserve generation quality while significantly improving computational efficiency.
- • WAN2.1 + MMAudio is a cascaded baseline that decouples video and audio generation. Specifically, we use WAN2.1 [5] as the video backbone, followed by MMAudio [24] for video-conditioned audio synthesis. Unlike joint or synchronous modeling, this pipeline allocates the primary modeling burden to a strong video generator to capture spatiotemporal dynamics and kinematics, while the conditional audio generator focuses on acoustic consistency. This approach provides a competitive system-level baseline for audio-visual pairing.

- 6.2 Experimental Results 2https://colab.research.google.com/drive/1RAWb22-PFNI-X1gPVzc927SGUdfr6nsR

###### 6.2.1 Comparison with Baseline Methods

We evaluate the performance of MOVA against several competitive baselines, including LTX-2 [66], Ovi [18], and a cascaded pipeline (WAN2.1 + MMAudio). Table 4 presents a comprehensive quantitative comparison across four critical dimensions.

Audio Fidelity and Speech Quality. MOVA demonstrates a clear advantage in generating high-quality audio. Specifically, MOVA-360p achieves a state-of-the-art IS of 4.269 and a DNSMOS [63] of 3.797, significantly outperforming LTX-2 and Ovi. We observe that while the cascaded WAN2.1 + MMAudio baseline shows respectable IS (4.036), it lacks the capability to generate intelligible speech content. In contrast, MOVA maintains high speech naturalness and clarity even as we scale the resolution to 720p (DNSMOS of 3.671), suggesting that our unified modeling effectively captures the complex distribution of human speech and ambient sounds.

Audio-Visual Alignment. We evaluate audio-visual alignment through two lenses: temporal synchronization (DeSync [36]) and cross-modal semantic alignment (IB-Score [37]). Compared to contemporary unified models like LTX-2 and Ovi, MOVA exhibits a pronounced advantage. Specifically, MOVA-360p with dual CFG (𝑠𝐵 = 3.5) achieves a DeSync of 0.351 and an IB-Score of 0.315, significantly surpassing LTX-2 (0.451 / 0.213) and Ovi (0.515 / 0.190). This substantial margin, particularly the ∼50% improvement in IB-Score over Ovi, suggests that MOVA effectively binds auditory events to generated visual context. Notably, although the specialized cascaded pipeline (WAN2.1 + MMAudio) yields the best DeSync (0.260) due to its task-specific audio generator, MOVA virtually closes the gap in both temporal and semantic metrics. This demonstrates that our dual CFG strategy effectively amplifies the cross-modal alignment signal during inference, allowing a unified architecture to match the precision of modularized pipelines without sacrificing structural simplicity or end-to-end coherence.

Lip-SyncPrecision. The accuracyoffine-grainedlip synchronization is measured byLSE-D and LSE-C [64]. MOVA variants, particularly those equipped with dual CFG, exhibit a dominant performance in this category. MOVA-360p w/ dual CFG achieves the best LSE-D (7.004) and LSE-C (7.800), representing a substantial margin over LTX-2 (7.261/6.109) and Ovi (7.468/6.378). Interestingly, even without dual CFG, MOVA-720p maintains a competitive LSE-C of 6.593. This superior lip-sync performance confirms that our model has internalized a sophisticated mapping between phonetic features and labial dynamics, which is further refined by the explicit guidance of the Bridge interactions during inference.

Multi-Speaker Attribution. As illustrated in Table 4, MOVA-720p achieves the cpCER of 0.149, lower than LTX-2 (0.220) and Ovi (0.436). The high error rate of Ovi in this metric suggests a frequent ”voice-identity mismatch,” where the model fails to associate the correct speech with the corresponding subject. The low cpCER of MOVA demonstrates its ability to maintain high-fidelity identity consistency and correct audiovisual attribution in crowded scenes, a crucial requirement for realistic content generation. For both MOVA360p and MOVA-720p, the introduction of dual CFG leads to a slight increase in cpCER. This is primarily because dual CFG redistributes the guidance strength among multiple conditional branches during sampling, thereby diluting the relative weight of the text condition and reducing the model’s ability to follow explicit speaker instructions, such as speaker tags [S01]/[S02] and their corresponding speech content. In multi-speaker scenarios, this weakened instruction-following behavior more easily results in mismatches between speaker identities and transcribed content, which is reflected in higher cpCER. In addition, compared to MOVA-360p, MOVA-720p consistently achieves lower cpCER under both settings. This advantage mainly stems from differences in training data scale and training stages. MOVA-720p is further fine-tuned on top of MOVA-360p with broader data coverage and more extensive exposure to multi-speaker samples, leading to improved stability in speaker identity and speech content association.

Figure 7 Ablation study on human preference.

- Table 5 Ablation study of 𝑠𝐵. IS and AV-Align metrics are evaluated on all Verse-Bench subsets; DNSMOS and Lip Sync metrics are evaluated on Verse-Bench set3; ASR Acc is evaluated on the multi-speaker subset. Bold and underlined values denote the best and second-best results, respectively.

###### Model 𝑠𝐵

Audio-Speech AV-Align Lip Sync ASR Acc IS↑ DNSMOS↑ DeSync↓ IB-Score↑ LSE-D↓ LSE-C↑ cpCER↓

MOVA-360p 1.0 4.269 3.797 0.475 0.286 8.098 6.278 0.177 w/ dual CFG 2.0 4.222 3.748 0.421 0.305 7.323 7.331 0.185 3.0 4.319 3.686 0.388 0.312 7.014 7.774 0.188

- 3.5 4.169 3.674 0.351 0.315 7.004 7.800 0.247

- 4.0 4.225 3.631 0.365 0.316 6.957 7.891 0.264

###### 6.2.2 Ablation Study

We conduct a series of ablation experiments of MOVA to evaluate the impact of our training strategy and inference configurations, which is shown in Table 4, Table 5 and Table 6.

Scaling to High Resolution. We evaluate the empirical robustness of MOVA by scaling the generation resolution from 360p to 720p. As summarized in Table 4, the 720p variant demonstrates remarkawble consistency across diverse evaluation dimensions. Specifically, in terms of temporal and semantic alignment, MOVA-720p maintains a DeSync of 0.485 and an IB-Score of 0.277, showing negligible degradation compared to the 360p base model (0.475 and 0.286, respectively). This stability is particularly noteworthy as increasing visual resolution often introduces challenges in maintaining cross-modal coherence. Additionally, MOVA-720p achieves a Lip Sync LSE-C of 6.593, outperforming the 360p version’s 6.278. Similarly, it reports the best-ever cpCER (0.149) on the multi-speaker subset. While there is a marginal decrease in audio fidelity and speech quality metrics, which is a common trade-off when the model capacity is further distributed to handle increased visual complexity, the overall performance profile remains highly competitive. These results collectively validate our staged training strategy, confirming that MOVA can effectively scale to high-resolution synthesis while preserving its foundational audio-visual generative priors.

Effect of Dual Classifier-Free Guidance. We evaluate the influence of the dual CFG scale 𝑠𝐵 in Table 5. Our results demonstrate a synergistic improvement across all alignment-related metrics as 𝑠𝐵 increases from 1.0 to 4.0. Specifically, we observe a consistent reduction in DeSync and LSE-D, alongside a significant gain in IB-Score and LSE-C. For instance, as 𝑠𝐵 scales to 4.0, the LSE-C reaches a peak of 7.891 and the DeSync score is minimized to 0.365. This uniform progression across multiple distinct metrics suggests that strengthening the modality-alignment guidance effectively improves the synchronization between generated video and audio. However, this heightened alignment precision comes at a clear cost to speech quality and instruction following. As the guidance toward audio-visual alignment becomes more dominant, we observe a concurrent degradation in DNSMOS and a rise in cpCER (from 0.177 to 0.264). We interpret this phenomenon as a form of conditional interference: in the multi-branch sampling process, excessively high 𝑠𝐵 prioritizes the geometric constraints of synchronization over the generative fidelity of the speech signal. This ”over-regularization” potentially leads to a diminished sensitivity to textual instructions (e.g., speaker-specific tags), causing the model to prioritize how the speech aligns with the video at the expense of what is being said and how natural it sounds.

Emergent T2VA Capability. Interestingly, we find that MOVA exhibits a strong emergent capability for the T2VA task, which is summarized in Table 6. By substituting the reference image with a null placeholder (MOVA-360p-T2VA), we test whether the model can synthesize synchronized content driven solely by textual prompts. As summarized in Table 6, several intriguing observations emerge. Notably, the T2VA variant achieves a superior IS of 4.370 and a lower DeSync of 0.441 compared to the standard TI2VA baseline. This performance gain suggests that in the absence of explicit structural constraints from a reference image, the model can more freely explore the joint audio-visual manifold, leading to higher audio fidelity and improved temporal synchronization. Predictably, identity-dependent metrics like LSE-C and LSE-D show a marginal decline, as the null placeholder provides no lip geometry. However, the overall stability of the T2VA results is remarkable. This underscores that MOVA has internalized a robust, decoupled yet highly coordinated prior for video and audio synthesis, allowing it to maintain temporal synchronization and semantic alignment even when the visual conditioning is entirely removed.

###### 6.2.3 Arena-Based Human Evaluation

Given the limitations of current objective evaluation frameworks for audio-visual generation models, MOVA introduces an Arena-based human preference evaluation paradigm that includes the latest open-source audio-visual generation models worldwide. The evaluation collected over 5,000 valid votes and systematically analyzed the results. To ensure a fair comparison, all models utilize their respective official promptrefinement methods (e.g., our rewriter described in Section 5.2) to enhance the video generation prompt.

Subjective Comparison with Baseline Methods. As shown in Figure 8, MOVA demonstrates a clear superiority in human preference: it is more frequently selected by users in pairwise comparisons, achieving an ELO rating of 1113.8 (starting from an initial rating of 1000), significantly higher than all baseline models. MOVA consistently maintains a win rate exceeding 50%, with win rates surpassing 70% against Ovi and the cascaded system (WAN + MMAudio).

Subjective Ablation Study. As illustrated in Figure 7, we conduct an internal Arena to dissect the impact of prompt refinement, resolution scaling, and inference strategies on human preference. A primary observation is the critical role of the prompt rewriter. Variants utilizing refined prompts consistently achieve superior ELO ratings, with MOVA-720p (w/ rewriter) reaching the peak score of 1025.3. Compared to the standard MOVA-720p (982.9), this substantial ELO gain validates our motivation: user-provided inputs often vary in format and level of detail, creating a distribution gap with the model’s training data. By employing our multi-stage conditioning pipeline, which leverages LLMs to synthesize prompts that preserve visual grounding (e.g., style, cinematography) while incorporating temporal dynamics, we bridge this gap and effectively incentivize the model’s full generative potential. Regarding our inference strategy, we observe a subtle trade-off; while dual CFG (𝑠𝐵 = 3.5) significantly improves objective alignment metrics, it leads to a

(a) Human preference ELO ranking. (b) Win rates of MOVA against baseline models.

- Figure 8 Arena evaluation results showing MOVA’s performance in human preference studies. (a) ELO ratings demonstrate MOVA’s superiority over baseline models. (b) Pairwise win rates show MOVA consistently outperforms all competitors, particularly against Ovi and the WAN + MMAudio cascade.

Table 6 Evaluation of T2VA effectiveness. IS and AV-Align metrics are evaluated on all Verse-Bench subsets; DNSMOS and Lip Sync metrics are evaluated on Verse-Bench set3; ASR Acc is evaluated on the multi-speaker subset. Bold values denote the best results.

Model

Audio-Speech AV-Align Lip Sync ASR Acc IS↑ DNSMOS↑ DeSync↓ IB-Score↑ LSE-D↓ LSE-C↑ cpCER↓

MOVA-360p 4.269 3.797 0.475 0.286 8.098 6.278 0.177 MOVA-360p-T2VA 4.370 3.767 0.441 0.281 8.362 5.830 0.188

slight decrease in human preference scores, dropping from 1025.3 to 1014.5 in the rewriter-enhanced 720p models. We attribute this decrease to the formulation of dual CFG: by explicitly amplifying cross-modal alignment signals, the relative guidance scale for the primary text instruction is effectively diminished. This can occasionally result in reduced instruction-following.

- 6.3 Scaling to Lip Synchronization

Lip synchronization is among the most demanding audio–video generation tasks. Unlike discrete, onsetdriven events (e.g., “chopping fruit” or “hitting a drum”) where alignment depends on a few salient temporal onsets, speech requires continuous, fine-grained correspondence between mouth shapes and phonemes across long spans. We find that architectural mechanisms alone (e.g., Bridge modules for cross-modal attention) are insufficient to achieve high-quality lip synchronization—the model must also learn phoneme-toviseme mappings from data, which requires larger capacity and more training examples.

- Figure 9 shows the progression of LSE-C (higher is better) and LSE-D (lower is better) across our three-stage training process. In Stage 1, we train at 360p with aggressive video denoising, mild audio denoising, and high text dropout to force the model to rely on cross-modal bridging for alignment. LSE-D drops rapidly and LSE-C rises, indicating the model quickly learns basic synchronization patterns. Stage 2 maintains 360p but aligns the noise schedules across modalities for more stable cross-modal attention, reduces text dropout to refine semantic details, and applies loudness normalization to avoid CFG-induced volume distortion. LSE-D continues to decrease while LSE-C shows a notable jump, reflecting improved consistency and confidence in alignment. Finally, Stage 3 scales to 720p. With stable cross-modal alignment already established, the model can safely allocate capacity to higher resolution and finer spatial details without disrupting the learned synchronization structure. LSE-D further decreases and plateaus, while LSE-C stabilizes at a high level, indicat-

Figure 9 Training progression across three stages. Stage 1 (360p, aggressive bridging) establishes basic alignment; Stage

- 2 (360p, aligned schedules) refines consistency; Stage 3 (720p) scales to high resolution while preserving synchronization.

ing convergence to high-quality lip synchronization.

- 7 Discussion

- 7.1 Predefined sigma as an Implicit Synchronization Prior

Audio-visual synchronization inherently contains a tension in conditioning direction. For many discrete, event-driven cases, the most natural formulation is Video→Audio: the visual stream deterministically anchors when and where events occur, and most sound events (impacts, collisions, cuts, percussive gestures) are visually driven with clear temporal onsets. In these settings, generating audio conditioned on video aligns with both human intuition and the causal structure of the scene. In contrast, other synchronization tasks are more realistically Audio→Video and better match practical applications. Speech-driven generation is the canonical example: given an audio track, the target is to produce temporally consistent facial motion and lip articulation (Speech→Video), potentially with speaker-specific dynamics, language-dependent phoneme–viseme mappings, and context-dependent expressiveness. This directionality is often required in downstream pipelines such as dubbing, avatar animation, and talking-head generation, where audio is fixed and visuals must adapt.

This tension becomes more pronounced under the diffusion formulation. At each timestep, the noise levels for video and audio latents are governed by pre-defined schedules 𝜎𝑣(𝑡𝑣) and 𝜎𝑎(𝑡𝑎) (introduced in Dual Sigma Shift), which act as fixed priors and are not learnable during training. As a result, the relative corruption of the audio stream is largely fixed by design, whereas the effective uncertainty in the visual stream can vary substantially with object scale and visual dominance in the frame (as discussed in Stable Diffusion

- 3 [67]). Consequently, the same global schedule can implicitly bias the conditional direction: for close-up shots of lip motion where the target region occupies a large portion of the frame, the visual latent is relatively informative and the generation tends to behave like Video→Audio; conversely, when the relevant speaker occupies only a small region, the visual evidence becomes comparatively uncertain, and the process may naturally shift toward Audio→Video, letting speech provide the more reliable temporal anchor.

##### 7.2 Classifier-Free Guidance: factorization order

Our dual-CFG derivation starts from the factorization 𝑃(𝑧 ∣ 𝑐𝑇,𝑐𝐵) ∝ 𝑃(𝑐𝑇 ∣ 𝑐𝐵,𝑧)𝑃(𝑐𝐵 ∣ 𝑧)𝑃(𝑧), which induces a natural nesting structure: we first “turn on” cross-modal information (via 𝑐𝐵) and then apply text guidance on top of it (via 𝑐𝑇). This order is not the only valid choice. For example, one may alternatively

decompose the posterior as

𝑃(𝑧 ∣ 𝑐𝑇,𝑐𝐵) ∝ 𝑃(𝑐𝐵 ∣ 𝑐𝑇,𝑧)𝑃(𝑐𝑇 ∣ 𝑧)𝑃(𝑧), (7) and derive an analogous two-term guidance by swapping the roles of 𝑐𝑇 and 𝑐𝐵.

We adopt our order for a practical reason: it admits a clean reduction to the standard text-only CFG as a special case. Concretely, setting 𝑠𝐵 = 1 makes the Bridge condition appear in both the conditional and “unconditional” text branches, so the guidance collapses to the familiar text-CFG form while keeping crossmodal injection fixed. In other words, by tuning (𝑠𝐵,𝑠𝑇) we can continuously interpolate between “text-only CFG” and “text + modality CFG” without changing the sampling procedure.

In contrast, under the alternative factorization, the branch structure couples 𝑐𝐵 to the text-unconditional baseline in a way that prevents an equivalent “text-only CFG” limit: there is no choice of guidance weights that keeps the Bridge behavior identical across the two text branches while still producing the standard textCFG difference term. As a result, the swapped order does not provide an intuitive knob for “only amplify text while leaving cross-modal interactions unchanged.”

##### 7.3 Limitations

Audio Modeling Capacity and Coverage. Our audio tower follows the Wan2.1-1.3B backbone, which may limit the modeling capacity for acoustically rich or highly structured signals. In particular, we observe degraded performance on singing voice, complex sound textures, and music/instrumental content, where finegrained pitch/harmonic structure and long-range temporal dependencies are critical. More broadly, some audio–visual phenomena require stronger physical reasoning (e.g., correctly reflecting propagation delays such as the temporal offset between lightning and thunder), which is not explicitly enforced by our current objective and data.

Multi-speaker Synchronization and Annotation Reliability. While our model can handle single-speaker lip-sync reliably in many cases, multi-speaker scenes remain challenging. Rapid speaker turn-taking, overlapping speech, and ambiguous on-screen attribution can lead to incorrect mouth–audio assignment and temporal drift. This issue is compounded by the data pipeline: diarization errors and imperfect activespeaker labels can propagate to training, making the model conflate speakers or learn inconsistent supervision. Improving multi-speaker supervision (e.g., stronger active-speaker detection, cross-modal speaker tracking, and better filtering of noisy segments) is necessary for robust deployment.

Sequence Length and Computational Cost. Our current training and inference are constrained by sequence length. For example, a 720p, 8s clip at 24fps yields on the order of 1.6×105 tokens, resulting in high memory and compute costs. This limits throughput during training and increases latency at inference, especially when using the most general guidance setting (NFE= 3). Future work could address this bottleneck via more aggressive temporal/spatial compression, hierarchical or blockwise generation, and system-level optimizations tailored to long-context video tokens.

### 8 Related Work

Video Generation. Diffusion transformers (DiTs) [1, 68] have enabled large-scale video synthesis. Open models such as Wan [5] and HunyuanVideo [69] achieve near-photorealistic generation through efficient attention [70, 71] and transformer scaling [72]. Recent work extends these models to long-horizon generation [73], controllable camera motion [74], and high-resolution outputs exceeding 1080p. However, most textto-video systems remain video-only, leaving audio generation as a separate problem. Proprietary systems Veo3 [12] and Sora2 [13] demonstrate joint audio-video capabilities, but their closed-source nature limits reproducibility.

Audio Generation and Cascaded Pipelines. Latent diffusion enables scalable text-to-audio generation [50, 75]. Audio VAEs compress waveforms into compact latent representations: DAC [76] uses residual vector quantization for high-quality reconstruction, while Stable Audio [77] employs a stereo variational autoencoder with spectral losses. Cascaded video-to-audio (V2A) pipelines [24, 78, 79] represent a prevalent approach for audiovisual content creation. MMAudio [24] achieves temporal alignment by extracting features from video and using them as conditioning signals. While cascaded pipelines utilize strong singlemodality priors, the sequential factorization ignores bidirectional modality influence: audio cannot inform visual trajectory during sampling, and vice versa.

Joint Audio-Video Generation. End-to-end joint generation has been explored to overcome cascaded limitations [17–20, 80–82]. MMDisCo [83] uses discriminator-guided cooperative diffusion to align pretrained models, though adversarial training introduces instability at scale. MM-Diffusion [82] and JavisDiT [81] propose dual-stream architectures with cross-modal attention but are restricted to ambient sounds. MTV [80] explicitly separates audio into speech, effects, and music tracks with disentangled control over lip motion, event timing, and visual mood, achieving precise audio-visual synchronization across diverse audio types. UniVerse-1 [17] integrates Wan2.1 and Ace through a stitching-of-experts paradigm with independent noise sampling, but suffers from audio-video drift. Recent works such as Ovi [18], Harmony [19], and UniAVGen [20] adopt dual-tower architectures with RoPE-based positional encoding, achieving lip-synchronized video generation results without requiring prior separation of speech and environmental sounds. However, they have not scaled to general domains to demonstrate the full potential of the architecture. LTX-2 [66] successfully scales the dual-tower approach to cover both lip-synchronized speech and general domain sounds through large-scale data training, though the audio quality exhibits some electronic artifacts that require cleaner reconstruction. We address these limitations through capacity scaling with a 29B dual-tower architecture, achieving high-fidelity audio and strong performance on both lip-synchronized speech and general domain sounds across bilingual settings.

### 9 Conclusions

We presented MOVA, an open and scalable framework for joint video–audio generation with 32B total parameters (18B active). MOVA uses an asymmetric dual-tower design (an A14B video backbone and a 1.3B audio backbone) together with a 2.6B bidirectional bridge and Aligned RoPE to support fine-grained temporal audio–visual interaction.

Our work targets three key challenges in joint audio–video generation: data, modeling, and scaling. We curate over 100,000 hours of fine-grained audio–visual data with sound, music, and speech annotations aligned to visual content. We also propose training and architectural designs that improve the stability of largescale multimodal diffusion training, including decoupled timestep sampling to allow modality-specific noise schedules. Through controlled scaling studies, we find that increasing video model capacity and training data substantially improves lip synchronization, where smaller models show clear performance saturation.

In addition, we report practical system optimizations for large-scale training (Context Parallelism, FSDP2 strategies, and scheduled garbage collection), enabling stable 1024-GPU runs and achieving approximately 35% MFU. We will release the model weights, training code, and inference pipelines, and we hope MOVA can serve as a strong open baseline for future research on synchronized audio–video generation. Finally, important limitations remain, including high training cost and remaining failure cases in motion generation, which we leave for future work.

### Contributors

Core Contributors and Contributors are sorted alphabetically by first name, excluding advisors. Core Contributors:

Donghua Yu, Mingshu Chen, Qi Chen, Qi Luo, Qianyi Wu, Qinyuan Cheng*, Ruixiao Li, Tianyi Liang*, Wenbo Zhang, Wenming Tu, Xiangyu Peng, Yang Gao, Yanru Huo, Ying Zhu, Yinze Luo, Yiyang Zhang, Yuerong Song, Zhe Xu, Zhiyu Zhang

Contributors: Chenchen Yang, Cheng Chang, Chushu Zhou, Hanfu Chen, Hongnan Ma, Jiaxi Li, Jingqi Tong, Junxi Liu, Ke Chen, Shimin Li, Songlin Wang, Wei Jiang, Zhaoye Fei, Zhiyuan Ning

Advisors: Chunguo Li, Chenhui Li, Ziwei He, Zengfeng Huang, Xie Chen†, Xipeng Qiu†

Affiliations: Shanghai Innovation Institute MOSI Intelligence Fudan University Shanghai Jiao Tong University East China Normal University Tongji University Southeast University Xiamen University University of Electronic Science and Technology of China

∗Project Lead. †Corresponding authors: chenxie95@sjtu.edu.cn, xpqiu@fudan.edu.cn

### References

- [1] William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 4172–4182. IEEE, 2023. doi: 10.1109/ICCV51070.2023.00387. URL https://doi.org/10.1109/ICCV51070.2023.00387.

- [2] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. URL https://openai.com/research/video-generation-models-as-world-simulators.
- [3] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. CoRR, abs/2412.20404, 2024. doi: 10.48550/ARXIV.2412.20404. URL https://doi.org/10.48550/arXiv.2412.20404.

- [4] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, Yuhui Wang, Anbang Ye, Gang Ren, Qianran Ma, Wanying Liang, Xiang Lian, Xiwen Wu, Yuting Zhong, Zhuangyan Li, Chaoyu Gong, Guojun Lei, Leĳun Cheng, Limin Zhang, Minghao Li, Ruĳie Zhang, Silan Hu, Shĳie Huang, Xiaokang Wang, Yuanheng Zhao, Yuqi Wang, Ziang Wei, and Yang You. Open-sora 2.0: Training a commercial-level video generation model in $200k. CoRR, abs/2503.09642, 2025. doi: 10.48550/ ARXIV.2503.09642. URL https://doi.org/10.48550/arXiv.2503.09642.

- [5] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Xiaofeng Meng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, YangyuLv, YifeiLi, YĳingLiu, Yiming Wang, YingyaZhang, YitongHuang, YongLi, YouWu, YuLiu, YulinPan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. CoRR, abs/2503.20314, 2025. doi: 10.48550/ARXIV.2503.20314. URL https://doi.org/10.48550/arXiv.2503.20314.

- [6] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

- [7] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, and Song Hanand Yukang Chen. Longlive: Real-time interactive long video generation. 2025.
- [8] Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, and Xiangyu Yue. Ditctrl: Exploring attention control in multi-modal diffusion transformer for tuning-free multi-prompt longer video generation. arXiv:2412.18597, 2024.

- [9] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. CoRR, abs/2509.20328, 2025. doi: 10.48550/ARXIV.2509.20328. URL https://doi.org/10.48550/arXiv.2509.20328.

- [10] Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, Ming Zhang, Qiguang Chen, Tianyi Liang, Xiaomeng Hu, Yining Zheng, et al. Thinking with video: Video generation as a promising multimodal reasoning paradigm. arXiv preprint arXiv:2511.04570, 2025.

- [11] Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Dechao Meng, Jinwei Qi, Penchong Qiao, Zhen Shen, Yafei Song, Ke Sun, Linrui Tian, Guangyuan Wang, Qi Wang, Zhongjian Wang, Jiayu Xiao, Sheng Xu, Bang Zhang, Peng Zhang, Xindi Zhang, Zhe Zhang, Jingren Zhou, and Lian Zhuo. Wan-s2v: Audio-driven cinematic video generation,

2025. URL https://arxiv.org/abs/2508.18621.

- [12] Google / DeepMind. Veo3: A text-to-video generation system. Technical Report –, Google DeepMind, 2025. URL https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf. Accessed: 2025-09-28.
- [13] OpenAI. Sora 2 is here. https://openai.com/index/sora-2/, 2025. Accessed: 2026-01-13.
- [14] Wan AI. Wan 2.6 — ai video generation introduction. https://wan.video/introduction/wan2.6, 2025. Accessed: 2026-02-09.

- [15] Kling AI. Kling 3.0 ai video generator. https://kling3.io/, 2026. Accessed: 2026-02-09.
- [16] ByteDance / Seedance AI. Seedance 2.0 ai video generation platform. https://seedance2.com/, 2026. Accessed: 2026-02-09.
- [17] Duomin Wang, Wei Zuo, Aojie Li, Ling-Hao Chen, Xinyao Liao, Deyu Zhou, Zixin Yin, Xili Dai, Daxin Jiang, and Gang Yu. Universe-1: Unified audio-video generation via stitching of experts. arXiv preprint arXiv:2509.06155, 2025.

- [18] Chetwin Low, Weimin Wang, and Calder Katyal. Ovi: Twin backbone cross-modal fusion for audio-video generation. arXiv preprint arXiv:2510.01284, 2025.

- [19] Teng Hu, Zhentao Yu, Guozhen Zhang, Zihan Su, Zhengguang Zhou, Youliang Zhang, Yuan Zhou, Qinglin Lu, and Ran Yi. Harmony: Harmonizing audio and video generation through cross-task synergy. arXiv preprint arXiv:2511.21579, 2025.

- [20] Guozhen Zhang, Zixiang Zhou, Teng Hu, Ziqiao Peng, Youliang Zhang, Yi Chen, Yuan Zhou, Qinglin Lu, and Limin Wang. Uniavgen: Unified audio and video generation with asymmetric cross-modal interactions. arXiv preprint arXiv:2511.03334, 2025.

- [21] Sizhe Shan, Qiulin Li, Yutao Cui, Miles Yang, Yuehai Wang, Qun Yang, Jin Zhou, and Zhao Zhong. Hunyuanvideofoley: Multimodal diffusion with representation alignment for high-fidelity foley audio generation, 2025. URL https://arxiv.org/abs/2508.16930.
- [22] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

- [23] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

- [24] Ho Kei Cheng, Masato Ishii, Akio Hayakawa, Takashi Shibuya, Alexander Schwing, and Yuki Mitsufuji. Mmaudio: Taming multimodal joint training for high-quality video-to-audio synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28901–28911, 2025.

- [25] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2021.
- [26] Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audio-visual dataset. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2020.

- [27] Moayed Haji-Ali, Willi Menapace, Aliaksandr Siarohin, Guha Balakrishnan, and Vicente Ordonez. Taming data and transformers for audio generation. arXiv preprint arXiv:2406.19388, 2024.

- [28] Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Rui-Jie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. Advances in Neural Information Processing Systems, 37:21236–21270, 2024.

- [29] Sangho Lee, Jiwan Chung, Youngjae Yu, Gunhee Kim, Thomas M. Breuel, Gal Chechik, and Yale Song. ACAV100M: automatic curation of large-scale datasets for audio-visual video representation learning. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 10254–

10264. IEEE, 2021. doi: 10.1109/ICCV48922.2021.01011. URL https://doi.org/10.1109/ICCV48922.2021.01011.

- [30] Hui Li, Mingwang Xu, Yun Zhan, Shan Mu, Jiaye Li, Kaihui Cheng, Yuxuan Chen, Tan Chen, Mao Ye, Jingdong Wang, et al. Openhumanvid: A large-scale high-quality dataset for enhancing human-centric video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7752–7762, 2025.

- [31] Youliang Zhang, Zhaoyang Li, Duomin Wang, Jiahe Zhang, Deyu Zhou, Zixin Yin, Xili Dai, Gang Yu, and Xiu Li. Speakervid-5m: A large-scale high-quality dataset for audio-visual dyadic interactive human generation. arXiv preprint arXiv:2507.09862, 2025.

- [32] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhĳie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371, 2024.

- [33] Silero Team. Silero vad: pre-trained enterprise-grade voice activity detector (vad), number detector and language classifier. https://github.com/snakers4/silero-vad, 2024.
- [34] Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hoffman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, et al. Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound. arXiv preprint arXiv:2502.05139, 2025.

- [35] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou Hou, Annan Wang, Wenxiu Sun Sun, Qiong Yan, and Weisi Lin. Exploring video quality assessment on user generated contents from aesthetic and technical perspectives. In International Conference on Computer Vision (ICCV), 2023.

- [36] Vladimir Iashin, Weidi Xie, Esa Rahtu, and Andrew Zisserman. Synchformer: Efficient synchronization from sparse cues. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5325–5329. IEEE, 2024.

- [37] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

- [38] Wenxi Chen, Yuzhe Liang, Ziyang Ma, Zhisheng Zheng, and Xie Chen. Eat: Self-supervised pre-training with efficient audio transformer. arXiv preprint arXiv:2401.03497, 2024.

- [39] Xiaomi LLM-Core Team Zihao Yue, Zhenrui Lin, Yi-Hao Song, Weikun Wang, Shu-Qin Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, Lei Li, Kainan Bao, Hao Tian, Hailin Zhang, Gang Wang, Dawei Zhu, Cici, Chenhong He, Bowen Ye, Bowen Shen, Zihan Zhang, Zi-Ang Jiang, Zhixian Zheng, Zhichao Song, Zhen Luo, Yue Yu, Yudong Wang, Yu Tian, Yu Tu, Yihan Yan, Yi Huang, Xu Wang, Xin dan Xu, Xin Ran Song, Xing Zhang, Xing Yong, Xin Zhang, Xia Deng, Wenyu Yang, Wenhan Ma, Weiwei Lv, Weĳi Zhuang, Wei Liu, Sirui Deng, Shuo Liu, Shimao Chen, Shi liang Yu, Shao yang Liu, Shan yong Wang, Rui Ma, Qiantong Wang, Peng Wang, Nuo Chen, Menghang Zhu, Kang Zhou, Kang Zhou, Kai Fang, Jun-Miao Shi, Jinhao Dong, Jiebao Xiao, Jiaming Xu, Huaqiu Liu, Hongsheng Xu, Hengxu Qu, Hao-Song Zhao, Hanglong Lv, Guoan Wang, Duo Zhang, Dong Zhang, Di Zhang, Chong-Yi Ma, Chang Liu, Can Cai, and Bing Xia. Mimo-vl technical report. ArXiv, abs/2506.03569, 2025. URL https://api. semanticscholar.org/CorpusID:279155294.

- [40] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, and Junyang Lin. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025.

- [41] Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

- [42] Xinhao Mei, Chutong Meng, Haohe Liu, Qiuqiang Kong, Tom Ko, Chengqi Zhao, Mark D. Plumbley, Yuexian Zou, and Wenwu Wang. WavCaps: A ChatGPT-assisted weakly-labelled audio captioning dataset for audio-language multimodal research. IEEE/ACM Transactions on Audio, Speech, and Language Processing, pages 1–15, 2024.

- [43] Abhinaba Roy, Renhang Liu, Tongyu Lu, and Dorien Herremans. Jamendomaxcaps: A large scale music-caption dataset with imputed metadata. arXiv:2502.07461, 2025.

- [44] John Cramer, Hyungui Wu, Justin Salamon, and Juan Pablo Bello. Look, listen, and learn more: Design choices for deep audio embeddings. In Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856, 2019. doi: 10.1109/ICASSP.2019.8683142.

- [45] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems (NeurIPS), pages 2234–2242, 2016.

- [46] Khaled Koutini, Michael Moritz, Hamid Eghbal-Zadeh, and Gerhard Widmer. Efficient training of audio transformers with patchout. In Proc. Interspeech, pages 179–183, 2021. doi: 10.21437/Interspeech.2021-2041.

- [47] Yusong Wu, Haohe Liu, Ke Chen, Xin Wang, Qiuqiang Tian, Rui Chen, Qiuqiang Kong, Wenwu Huang, and Yuxuan Wang. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmenta-

- tion. In Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2023. doi: 10.1109/ICASSP49357.2023.10094846.
- [48] Bowen Zhang, Xinyu Wang, Shuo Chen, Chen Xu, Yuxuan Wang, and Qiuqiang Kong. Audiobox: Unified aesthetic quality assessment for audio generation. arXiv preprint arXiv:2309.07825, 2023. URL https://arxiv.org/abs/2309. 07825.

- [49] Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. Audiocaps: Generating captions for audios in the wild. In NAACL-HLT, 2019.

- [50] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley. AudioLDM: Text-to-audio generation with latent diffusion models. In Proceedings of the International Conference on Machine Learning, 2023.

- [51] Haohe Liu, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong, Qiao Tian, Yuping Wang, Wenwu Wang, Yuxuan Wang, and Mark D. Plumbley. Audioldm 2: Learning holistic audio generation with self-supervised pretraining. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32:2871–2883, 2024. doi: 10.1109/TASLP. 2024.3399607.

- [52] Chia-Yu Hung, Navonil Majumder, Zhifeng Kong, Ambuj Mehrish, Amir Ali Bagherzadeh, Chuan Li, Rafael Valle, Bryan Catanzaro, and Soujanya Poria. Tangoflux: Super fast and faithful text to audio generation with flow matching and clap-ranked preference optimization. arXiv preprint arXiv:2412.21037, 2024.

- [53] Navonil Majumder, Chia-Yu Hung, Deepanway Ghosal, Wei-Ning Hsu, Rada Mihalcea, and Soujanya Poria. Tango 2: Aligning diffusion-based text-to-audio generations through direct preference optimization, 2024. URL https: //arxiv.org/abs/2404.09956.
- [54] Jiawei Huang, Yi Ren, Rongjie Huang, Dongchao Yang, Zhenhui Ye, Chen Zhang, Jinglin Liu, Xiang Yin, Zejun Ma, and Zhou Zhao. Make-an-audio 2: Temporal-enhanced text-to-audio generation, 2023.
- [55] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

- [56] Jiarui Fang and Shangchun Zhao. Usp: A unified sequence parallelism approach for long context generative ai. arXiv preprint arXiv:2405.07719, 2024.

- [57] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.

- [58] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. CoRR, abs/2511.21631, 2025. doi: 10.48550/ ARXIV.2511.21631. URL https://doi.org/10.48550/arXiv.2511.21631.

- [59] Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf,

2025. Accessed: 2025-09-28.

- [60] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems

- 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html.
- [61] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

- [62] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. Panns: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE ACM Trans. Audio Speech Lang. Process., 28:2880–2894, 2020. doi: 10.1109/TASLP.2020.3030497. URL https://doi.org/10.1109/TASLP.2020.3030497.

- [63] Chandan K. A. Reddy, Vishak Gopal, and Ross Cutler. Dnsmos: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2021, Toronto, ON, Canada, June 6-11, 2021, pages 6493–6497. IEEE, 2021. doi: 10.1109/ICASSP39728.2021.

9414878. URL https://doi.org/10.1109/ICASSP39728.2021.9414878.

- [64] Joon Son Chung and Andrew Zisserman. Out of time: automated lip sync in the wild. In Asian conference on computer vision, pages 251–263. Springer, 2016.

- [65] Donghua Yu, Zhengyuan Lin, Chen Yang, Yiyang Zhang, Zhaoye Fei, Hanfu Chen, Jingqi Chen, Ke Chen, Qinyuan Cheng, Liwei Fan, et al. Moss transcribe diarize: Accurate transcription with speaker diarization. arXiv preprint arXiv:2601.01554, 2026.

- [66] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

- [67] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

- [68] JonathanHo, AjayJain, andPieterAbbeel. Denoisingdiffusionprobabilisticmodels. InProceedingsoftheAdvances in Neural Information Processing Systems, 2020.

- [69] Weĳie Kong, Qi Tian, Zĳian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [70] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

- [71] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.

- [72] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

- [73] Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shĳun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, and Tong Zhang. Longcat-video technical report. arXiv preprint arXiv:2510.22200,

2025. URL https://arxiv.org/abs/2510.22200.

- [74] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.

- [75] Haohe Liu, Qiao Tian, Yi Yuan, Xubo Liu, Xinhao Mei, Qiuqiang Kong, Yuping Wang, Wenwu Wang, Yuxuan Wang, and Mark D. Plumbley. AudioLDM 2: Learning holistic audio generation with self-supervised pretraining. arXiv preprint arXiv:2308.05734, 2023.

- [76] Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, and Kundan Kumar. High-fidelity audio compression with improved rvqgan. Advances in Neural Information Processing Systems, 36, 2024.

- [77] Zach Evans, Julian D Parker, CJ Carr, Zack Zukowski, Josiah Taylor, and Jordi Pons. Stable audio open. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025.

- [78] Simian Luo, Chuanhao Yan, Chenxu Hu, and Hang Zhao. Diff-foley: Synchronized video-to-audio synthesis with latent diffusion models. Advances in Neural Information Processing Systems, 36:48855–48876, 2023.

- [79] Yiming Zhang, Yicheng Gu, Yanhong Zeng, Zhening Xing, Yuancheng Wang, Zhizheng Wu, and Kai Chen. Foleycrafter: Bring silent videos to life with lifelike and synchronized sounds. arXiv preprint arXiv:2407.01494, 2024.

- [80] Shuchen Weng, Haojie Zheng, Zheng Chang, Si Li, Boxin Shi, and Xinlong Wang. Audio-sync video generation with multi-stream temporal control. NeurIPS, 2025.

- [81] Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Rongxin Jiang, Jiebo Luo, Hao Fei, et al. Javisdit: Joint audio-video diffusion transformer with hierarchical spatio-temporal prior synchronization. arXiv preprint arXiv:2503.23377, 2025.

- [82] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. Mm-diffusion: Learning multi-modal diffusion models for joint audio and video generation. In CVPR, 2023.

- [83] Akio Hayakawa, Masato Ishii, Takashi Shibuya, and Yuki Mitsufuji. Mmdisco: Multi-modal discriminator-guided cooperative diffusion for joint audio and video generation. arXiv preprint arXiv:2405.17842, 2024.

## Appendix

### Appendix Contents

A Appendix . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30

- A.1 Training Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- A.2 Ascend 910A2 Benchmark Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- A.3 Multi-shot and Single-shot Speech Window Generation . . . . . . . . . . . . . . . . . . . . . . 30
- A.4 Detailed Filtering Thresholds . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

- A.4.1 Threshold Specifications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- A.4.2 Subset Construction and Logic . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33

- A.5 Audio-visual Captioning Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- A.6 Benchmark Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37

### A Appendix

##### A.1 Training Hyperparameters

- Table 7 summarizes the complete training configuration across all three phases, including parallelism settings, learning rates, noise schedule parameters, and data curation details for reproducibility.

Table 7 Training hyperparameters for reproducibility. All phases use 1024 GPUs with DP replicate size 64.

Hyperparameter Phase 1 Phase 2 Phase 3

Resolution 360×640 360×640 720×1280 Frames 193 193 193 Batch size 128 128 64 Context Parallel (CP) 8 8 16

Learning rate (backbone) 1e-5 1e-5 1e-5 Learning rate (Bridge) 2e-5 2e-5 2e-5 Weight decay 0.001 0.001 0.001

Visual sigma shift 5.0 5.0 5.0 Audio sigma shift 1.0 5.0 5.0 Text dropout prob 0.5 0.2 0.2 Audio loss weight 0.2 0.2 0.2 LUFS normalization No Yes Yes

Training duration 15 days 7 days 20 days Checkpoint interval 5000 5000 2000

A.2 Ascend 910A2 Benchmark Details

We report a small system microbenchmark that measures end-to-end training step time under a fixed configuration (CP=4, DP-shard=2, 8 devices). The numbers depend on the software stack (driver/runtime versions, compiler and kernel coverage, and distributed communication settings), and should not be interpreted as a complete statement of training cost at scale.

- Table 8 Hardware summary and step time for an 8-device training microbenchmark on Ascend 910A2 (CP=4, DPshard=2).

Hardware FP16 (TFLOPs) VRAM (Consumption/GPU) Host RAM Step Time (s) 8× 910A2 376 ≈40GB ≥128GB 34.1

##### A.3 Multi-shot and Single-shot Speech Window Generation

In this section, we provide the pseudocode implementation for Multi-shot and Single-shot Speech Window Generation.

The multi-shot algorithm 1 advances over the speech segments from VAD. The upper bound of the window start time is set to the beginning of the current segment. The sampling range of the window start is further constrained such that it does not precede the end of the previous speech segment, does not precede the nearest scene split point before the current speech segment (to encourage natural scene transitions), and does not shift earlier than half of the window length relative to the current segment. A window start time is then randomly sampled within this constrained interval. Only windows whose temporal span contains at least one scene split point are kept.

- Algorithm 1 Multi-shot Speech Windows

Input: speech_segments: VAD results scene_split_spots: scene-detection results segment_duration = 8.05

- 1: 𝑖𝑑𝑥 ← 0
- 2: while 𝑖𝑑𝑥 < length(speech_segments) do
- 3: 𝑢𝑝𝑝𝑒𝑟_𝑏𝑜𝑢𝑛𝑑 ← speech_segments[𝑖𝑑𝑥].𝑠𝑡𝑎𝑟𝑡
- 4: if 𝑖𝑑𝑥 = 0 then
- 5: 𝑤𝑖𝑛𝑑𝑜𝑤_𝑠𝑡𝑎𝑟𝑡 ← 𝑢𝑝𝑝𝑒𝑟_𝑏𝑜𝑢𝑛𝑑
- 6: else
- 7: 𝑙𝑜𝑤𝑒𝑟_𝑏𝑜𝑢𝑛𝑑 ← max( speech_segments[𝑖𝑑𝑥 − 1].𝑒𝑛𝑑, max{𝑝 ∈ scene_split_spots ∣ 𝑝 < speech_segments[𝑖𝑑𝑥].𝑠𝑡𝑎𝑟𝑡}, speech_segments[𝑖𝑑𝑥].𝑠𝑡𝑎𝑟𝑡 − segment_duration/2 )
- 8: 𝑤𝑖𝑛𝑑𝑜𝑤_𝑠𝑡𝑎𝑟𝑡 ← RandomUniform(𝑙𝑜𝑤𝑒𝑟_𝑏𝑜𝑢𝑛𝑑,𝑢𝑝𝑝𝑒𝑟_𝑏𝑜𝑢𝑛𝑑)
- 9: end if
- 10: 𝑤𝑖𝑛𝑑𝑜𝑤_𝑒𝑛𝑑 ← 𝑤𝑖𝑛𝑑𝑜𝑤_𝑠𝑡𝑎𝑟𝑡 + segment_duration
- 11: if ∃𝑝 ∈ scene_split_spots such that 𝑤𝑖𝑛𝑑𝑜𝑤_𝑠𝑡𝑎𝑟𝑡 ≤ 𝑝 ≤ 𝑤𝑖𝑛𝑑𝑜𝑤_𝑒𝑛𝑑 then
- 12: append (𝑤𝑖𝑛𝑑𝑜𝑤_𝑠𝑡𝑎𝑟𝑡,𝑤𝑖𝑛𝑑𝑜𝑤_𝑒𝑛𝑑) to speech_multi_shot_segments
- 13: end if
- 14: 𝑖𝑑𝑥 ← next 𝑖𝑑𝑥 such that speech_segments[𝑖𝑑𝑥].𝑠𝑡𝑎𝑟𝑡 > 𝑤𝑖𝑛𝑑𝑜𝑤_𝑒𝑛𝑑
- 15: end while

The single-shot algorithm 2 iterates over consecutive scene intervals defined by scene split points. Within each scene interval, it identifies speech segments whose start time allows a fixed-length window to be fully contained in the scene. For each eligible speech segment, the upper bound of the window start time is set to the segment start, while the sampling range is constrained by the scene boundary, the end of the previous speech segment, and a half-window-length offset relative to the current segment. A single scene window is then generated by randomly selecting a start time within this range, while the index is advanced to skip overlapping windows, similar to the multi-shot algorithm above.

- Algorithm 2 Single-shot Speech Windows

Input: speech_segments: VAD results scene_split_spots: scene-detection results segment_duration = 8.05

- 1: 𝑖 ← 0
- 2: while 𝑖 < length(scene_split_spots) − 1 do
- 3: _𝑠𝑡𝑎𝑟𝑡 ← scene_split_spots[𝑖]
- 4: 𝑠𝑐𝑒𝑛𝑒_𝑒𝑛𝑑 ← scene_split_spots[𝑖 + 1]
- 5: Find the first index 𝑖𝑑𝑥 such that speech_segments[𝑖𝑑𝑥].𝑠𝑡𝑎𝑟𝑡 > 𝑠𝑐𝑒𝑛𝑒_𝑠𝑡𝑎𝑟𝑡 and speech_segments[𝑖𝑑𝑥].𝑠𝑡𝑎𝑟𝑡 + segment_duration < 𝑠𝑐𝑒𝑛𝑒_𝑒𝑛𝑑
- 6: if no such 𝑖𝑑𝑥 exists then
- 7: 𝑖 ← 𝑖 + 1
- 8: continue
- 9: end if
- 10: while 𝑖𝑑𝑥 < length(speech_segments) do
- 11: 𝑢𝑝𝑝𝑒𝑟_𝑏𝑜𝑢𝑛𝑑 ← speech_segments[𝑖𝑑𝑥].𝑠𝑡𝑎𝑟𝑡
- 12: if 𝑖𝑑𝑥 = 0 then
- 13: 𝑤𝑖𝑛𝑑𝑜𝑤_𝑠𝑡𝑎𝑟𝑡 ← 𝑢𝑝𝑝𝑒𝑟_𝑏𝑜𝑢𝑛𝑑
- 14: else
- 15: 𝑙𝑜𝑤𝑒𝑟_𝑏𝑜𝑢𝑛𝑑 ← max( speech_segments[𝑖𝑑𝑥 − 1].𝑒𝑛𝑑, 𝑠𝑐𝑒𝑛𝑒_𝑠𝑡𝑎𝑟𝑡, speech_segments[𝑖𝑑𝑥].𝑠𝑡𝑎𝑟𝑡 − segment_duration/2 )
- 16: 𝑤𝑖𝑛𝑑𝑜𝑤_𝑠𝑡𝑎𝑟𝑡 ← RandomUniform(𝑙𝑜𝑤𝑒𝑟_𝑏𝑜𝑢𝑛𝑑,𝑢𝑝𝑝𝑒𝑟_𝑏𝑜𝑢𝑛𝑑)
- 17: end if
- 18: 𝑤𝑖𝑛𝑑𝑜𝑤_𝑒𝑛𝑑 ← 𝑤𝑖𝑛𝑑𝑜𝑤_𝑠𝑡𝑎𝑟𝑡 + segment_duration
- 19: if 𝑤𝑖𝑛𝑑𝑜𝑤_𝑒𝑛𝑑 ≥ 𝑠𝑐𝑒𝑛𝑒_𝑒𝑛𝑑 then
- 20: break
- 21: end if
- 22: append (𝑤𝑖𝑛𝑑𝑜𝑤_𝑠𝑡𝑎𝑟𝑡,𝑤𝑖𝑛𝑑𝑜𝑤_𝑒𝑛𝑑) to speech_single_shot_segments
- 23: 𝑖𝑑𝑥 ← next 𝑖𝑑𝑥 such that speech_segments[𝑖𝑑𝑥].𝑠𝑡𝑎𝑟𝑡 > 𝑤𝑖𝑛𝑑𝑜𝑤_𝑒𝑛𝑑
- 24: end while
- 25: 𝑖 ← 𝑖 + 1
- 26: end while

##### A.4 Detailed Filtering Thresholds

In this section, we provide the specific filtering thresholds used during our data quality assessment process, as discussed in Sec. 3.4. These thresholds were determined by empirical observation to ensure a high-quality corpus while maintaining sufficient data diversity.

###### A.4.1 Threshold Specifications

- Table 9 summarizes the metrics and their corresponding cutoffs across the three dimensions: audio quality, video quality, and audio-visual alignment.

Table 9 Filtering thresholds for data quality assessment.

###### Dimension Metric Threshold

Silence Ratio < 0.8 Bandwidth (Hz) > 1,000 Audiobox-PQ (Production Quality) > 5.0 Audiobox-CU (Content Usefulness) > 4.5 Audiobox-CE (Content Enjoyment) > 2.5

Audio Quality

DOVER-Aesthetic > 0.85 DOVER-Technical > 0.05

Video Quality

ImageBind Score (IB-Score) ≥ 0.2 OR SynchFormer Offset (DeSync) ≤ 0.5

A-V Alignment

###### A.4.2 Subset Construction and Logic

- • Audio-Visual Alignment Logic: For alignment filtering, we apply a relaxed logical “OR” gate between semantic and temporal alignment. A video is retained if it satisfies either IB-Score ≥ 0.2 OR DeSync ≤ 0.5. This ensures that both semantically relevant ambient sounds and temporally synchronized speech/actions are preserved.
- • Speech Data Filtering: To construct specialized subsets for tasks such as lip synchronization, we utilize the EAT [38] classification model. Specifically, for the Speech Subset, we only retain samples where both EAT-contained-Speech and EAT-contained-Singing tags are evaluated as True (or satisfy the model’s positive classification confidence).

##### A.5 Audio-visual Captioning Details

We present detailed prompt design schemes tailored for our multimodal annotation pipeline, comprising threekeycomponents: (1)promptsforvideocaptioningutilizingtheMiMo-VL-7B-RLmodel[39]; (2)prompts for the dual-model audio strategy, specifically covering speech transcription with Qwen3-Omni-Instruct [40] and non-speech sound captioning with Qwen3-Omni-Captioner [40]; and (3) instructions for the GPT-OSS120B model [41] to facilitate consistency checks and caption merging.

###### Visual Description Prompt

You are a visual description annotator. Your sole purpose is to produce event-level annotations only for verifiably visible content (objects, actions, scenes, text). Ignore audio and inferred context entirely!

LAWS

- 1. LAW OF VISUAL TRUTH

- • Only describe visually verifiable elements (objects, actions, scenes, text).
- • Do not infer or hallucinate based on audio or context.

- 2. LAW OF VISUAL SILENCE

- • When no visual change is present, output null for visual_description.
- • When no on-screen text is present, output null for on_screen_text.

- 3. LAW OF VISUAL DYNAMICS

- • Detect all transitions: emerging/ceasing/moving visual elements.

- • Precisely document motion trajectories, speed changes, and visual rhythm.

###### OUTPUT FORMAT {

”video_visual_report”: { ”visual_description”: ”[Describe how visual elements evolve in detail. Ignore all text, subtitle and watermark in the video. null if no visual change is present]”, ”on_screen_text”: ”[Exact transcription of all visible text. null if no text is visually present]”

}, ”final_verification_audit”: {

”hallucination_check_passed”: true, ”visual_changes_verified”: true, ”comment”: ”All visual motion dynamics verifiably detected. No audio content or contextual inference included.”

} }

Audio Captioner Prompt Please describe the audio you hear in detail.

###### Speech Transcription Prompt

You are a speech transcription annotator. Your task is Automatic Speech Recognition! Your sole purpose is to produce event-level verbatim transcriptions of audible speech. Ignore non-speech sounds and music entirely. You must follow the OUTPUT FORMAT!

LAWS

- 1. LAW OF LANGUAGE FIDELITY

- • Preserve the original language.
- • No translation.

- 2. LAW OF SPEECH DYNAMICS

• When new speaker/language/intonation begins, create a new event. But you still need to follow the output format.

- 3. LAW OF SILENCE

• When no speech is present, output null in ‘speech_description‘ part.

###### OUTPUT FORMAT {

”speech_transcription_report”: { ”speech_description”: ”[Exact verbatim speech. null if none. Use [inaudible] for unclear parts]”

}, ”final_verification_audit”: {

”hallucination_check_passed”: true, ”speech_dynamics_verified”: true, ”comment”: ”All speech changes verifiably detected. No non-speech or music included.”

} }

###### Merge Caption Prompt

Act as a video description consolidator. Your task is to merge the provided video_description, audio_description, and speech_description fields into a single, fluent, and natural-sounding paragraph that vividly conveys the full multimedia experience to someone who cannot see or hear the original. Follow these rules precisely:

###### 1. Start with the visual narrative and anchor speaker context

- • If video_description is provided, begin with it, weaving in visual cues that ground speaker identities or positions (e.g., ”A woman in a red coat stands by the door; nearby, a man leans against the counter”).
- • Describe all visual content in specific, chronological, natural language—actions, objects, settings, lighting, movements, and scene changes—without summarizing. Use these details to contextualize speakers (e.g., ”The child pointing at the painting” or ”The older man gesturing emphatically”).

###### 2. Integrate speech as visually anchored dialogue with speaker dynamics

- • If speech_description is provided, embed spoken words as quoted dialogue tied to visual elements, usingspeechverbsthatreflecttone(e.g., ”snaps,””murmurs,””laughs”)andspeech rate (e.g., ”rushes,” ”drawls,” ”pauses between words”).
- • Use audio_description to inform:

- – Total number of speakers (e.g., ”two distinct voices” or ”a group of overlapping speakers”).
- – Speaker transitions, paired with visual cues when possible (e.g., ”As the camera pans to the window, a new voice cuts in sharply: ’Wait, that’s not right’”).
- – Voice characteristics (e.g., ”high-pitched,” ”gruff”) to align with visible subjects (e.g., ”the teenager in the corner” or ”the gray-haired woman”).

- • Example: ”The man slams his fist on the table. ’I told you this would happen!’ he shouts. Across the room, the woman in glasses crosses her arms and replies slowly, ’You never listened to my warnings.’”
- • Rely strictly on speech_description for the actual spoken content; use audio_description for speaker dynamics and video_description to anchor them visually.

###### 3. Describe non-speech audio concisely and distinctly

- • If audio_description is provided, include only non-speech elements, introduced with ”The audio includes…” or ”In the background…”.
- • Limit this to four categories:

- – Ambient/environmental sounds (e.g., rain tapping the window, a crowd’s murmur);

- – Music (e.g., a soft piano melody, upbeat jazz);
- – Audio themes/sources (e.g., a fire alarm blaring, waves crashing);
- – Structural audio changes (e.g., silence giving way to a crescendo).

- • Omit all references to voices, words, or speech—these belong exclusively to the speech section.

###### 4. Ensure coherence and avoid repetition

- • Never duplicate details across sections (e.g., don’t restate a speaker’s visual position in the audio section).
- • Link elements logically: e.g., a visual action (a character gasping) paired with speech (”’Oh no!’ she exclaims”) and ambient sound (a glass shattering in the background).

###### 5. Omit empty fields and maintain fluency

- • Skip missing/empty fields.
- • The final output should read as a seamless, human-like narrative—prioritizing flow, sensory immersion, and clarity over rigid structure.

We provide the caption examples generated by bimodality models, including the caption results of three dimensions: sound, music, and video, as well as the final total caption result, which is obtained by merging the dimension-specific captions mentioned above using a large language model [41]. Figure 10 presents the original video clips and audio mel-spectrogram corresponding to this example.

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Figure 10 Example video 1 for audio-visual captioning.

###### Visual Description

A close-up of an HP OfficeJet Pro X576dw MFP printer shows its ink cartridge bay with yellow (Y 971XL), blue (C 971XL), magenta (M 971XL), and an open black (X) slot. A person in a blue shirt and gray pants holds a black HP ink cartridge (labeled 970/971) and moves it toward the black slot. The printer’s display reads, “Close Ink Access Door” and “Please close the ink access door to continue.” A wider shot reveals the person from the back/side, continuing to align and insert the black cartridge into the X slot, with an office background (posters, a table, and blurred figures) visible. The camera alternates between close-ups of the hand/printer interaction and wider office-scene shots, documenting the cartridge insertion motion. After insertion, the hand finishes positioning the cartridge into the slot.

Audio Description The audio clip is a tightly produced, 8-second segment designed for commercial or corporate advertis-

ing, likely for a laser printer product or service. The recording quality is high, with a full frequency spectrum, no distortion, and a polished, modern sound profile. A single, mature male voice—clear, authoritative, and with a General American accent—delivers the narration in a calm, professional tone. The content is as follows:

- • “The final piece is simplicity.”
- • “Laser printers required us to replace imaging drums, transfer belts, and toner.”
- • “Now we only have to—”

The narration is delivered in a measured, declarative style, with a subtle rise in pitch at “simplicity” to signal a transition to a new point. The phrase “the final piece” is spoken with a slight emphasis, and the listing of printer components (“imaging drums, transfer belts, and toner”) is given with a rhythmic, evenly paced cadence. The concluding phrase “Now we only have to—” is abruptly cut off, suggesting the audio is an excerpt from a longer ad or commercial.

Throughout, a subtle, unobtrusive electronic background music track plays. It features synthesized pads, a soft, steady beat, and a simple melodic line, all in a minor key. The music is mixed low and never interferes with the narration, serving to create a modern, professional, and slightly optimistic atmosphere.

The recording is studio-clean, with no background noise or environmental sounds, and the voice has a slight reverb, indicating a controlled studio environment. The overall production style, linguistic choices, and content are all consistent with modern North American advertising aimed at businesses or professional consumers, highlighting the technological advancement and ease-of-use of a new printer or printer service. The abrupt ending reinforces that this clip is a fragment of a larger narrative, likely intended to leave a strong impression of progress and simplicity before the listener’s attention is drawn elsewhere.

###### Speech Description

The final piece is simplicity. Laser printers required us to replace imaging drums, transfer belts, and toner. Now we only have to...

###### Integrated Caption

A close-up reveals the ink cartridge bay of an HP OfficeJet Pro X576dw MFP, its yellow, blue and magenta cartridges neatly seated while the black slot hangs open; the printer’s display flashes the messages “Close Ink Access Door” and “Please close the ink access door to continue.” A person wearing a blue shirt and gray pants reaches in, holding a black HP ink cartridge labeled 970/971, and carefully aligns it with the empty X slot. The camera alternates between tight shots of the hand guiding the cartridge and wider views of the office, where posters, a table and blurred figures form the background, before the hand slides the cartridge fully into place. As this insertion unfolds, a calm, authoritative male voice—off-screen, mature and speaking with a clear General American accent—states, “The final piece is simplicity.” He continues, “Laser printers required us to replace imaging drums, transfer belts, and toner,” delivering the list with an even, rhythmic cadence, then adds, “Now we only have to—” before the narration is abruptly cut off. The audio includes a subtle, unobtrusive electronic background music track of synthesized pads, a soft steady beat and a simple melodic line in a minor key, mixed low so it underlies the narration without interfering.

##### A.6 Benchmark Details

This appendix provides additional details of the benchmark construction. Our benchmark is constructed from real-world videos by extracting the first-frame image and a corresponding prompt. Each prompt con-

cisely describes key visual elements, such as the scene setting, characters, and environmental conditions. Depending on the scenario, audio-related information is incorporated to form a unified prompt for joint video–audio generation. All samples are adapted to ensure temporal consistency and logical coherence of the generated videos.

The benchmark consists of six scenario categories, each targeting specific challenges in joint video–audio generation:

- • Multi-speaker scenarios, which evaluate the model’s ability to generate synchronized speech, facial expressions, and interactions among multiple characters.
- • Movie videos, which require film-level narrative generation with plots referencing the background of the original films.
- • Sports competitions, focusing on athletes’ performances, with some prompts including commentators’ narration.
- • Game livestream videos, covering shooting games, 3D games, and competitive games.
- • Camera motion sequences, designed to assess visual realism under camera panning, zooming, and rotation.
- • Anime-style videos, including both 2D anime and 3D animated content.

