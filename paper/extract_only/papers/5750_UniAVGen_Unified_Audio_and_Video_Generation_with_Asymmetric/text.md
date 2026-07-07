# arXiv:2511.03334v2[cs.CV]24Mar2026

## UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions

Guozhen Zhang1,†,∗ Zixiang Zhou2,† Teng Hu3 Ziqiao Peng4 Youliang Zhang5 Yi Chen2 Yuan Zhou2 Qinglin Lu2 Limin Wang1,6,‡ 1State Key Laboratory for Novel Software Technology, Nanjing University 2Tencent Hunyuan 3Shanghai Jiao Tong University 4Renmin University of China 5Tsinghua University 6Shanghai AI Lab zgzaacm@gmail.com lmwang@nju.edu.cn https://mcg-nju.github.io/UniAVGen/

Joint Audio-Video Generation Generation with Reference Audio

[Figure 1]

[Figure 2]

Ref. Audio

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

###### UniAVGen

Video-to-Audio Dubbing Audio-Driven Video Synthesis

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Figure 1. Multi-task compatibility of UniAVGen. Leveraging its robust design, UniAVGen can simultaneously tackle pivotal audiovisual tasks within a single model, eliminating the need for task-specific model designs.

#### Abstract

Due to the lack of effective cross-modal modeling, existing open-source audio-video generation methods often exhibit compromised lip synchronization and insufficient semantic consistency. To mitigate these drawbacks, we propose UniAVGen, a unified framework for human-centric joint audio and video generation. UniAVGen is anchored in a dual-branch joint synthesis architecture, incorporating two parallel Diffusion Transformers (DiTs) to build a cohesive cross-modal latent space. At its heart lies an Asymmetric Cross-Modal Interaction mechanism, which enables bidirectional, temporally aligned cross-attention, thus ensuring precise spatiotemporal synchronization and semantic consistency. Furthermore, this cross-modal interaction is augmented by a Face-Aware Modulation (FAM) module,

∗Work is done during internship at Tencent Hunyuan. †Guozhen Zhang and Zixiang Zhou contribute equally to this work. ‡Corresponding author.

which dynamically prioritizes salient regions in the interaction process. To enhance generative fidelity during inference, we additionally introduce Modality-Aware ClassifierFree Guidance (MA-CFG), a novel strategy that explicitly amplifies cross-modal correlation signals. Notably, UniAVGen’s robust joint synthesis design enables the seamless unification of pivotal audio-visual tasks within a single model. Furthermore, we demonstrate that joint multi-task training can further boost the performance of joint generation. Comprehensive experiments validate that, with far fewer training samples (1.3M vs. 30.1M), UniAVGen delivers overall advantages in audio-video synchronization, timbre consistency, and emotion consistency.

#### 1. Introduction

Joint audio-visual generation has emerged as a pivotal trend in state-of-the-art generative AI. Commercial solutions such as Veo3 [14], Sora2 [35], and Wan2.5 [8] have achieved

exceptional generation fidelity and demonstrated notable practical utility. However, most existing open-source methods [4, 5, 22, 28, 41, 45] still rely on decoupled pipelines, often leveraging a two-stage paradigm. One paradigm first generates a silent video, then performs separate audio synthesis for post-hoc dubbing [5, 45]; the other first generates an audio track to drive subsequent video synthesis [4, 22, 28]. Regardless of the order, such sequential frameworks inherently suffer from critical limitations: modality decoupling impedes cross-modal interplay during generation, resulting in inadequate semantic consistency and emotional alignment. Consequently, designing effective audio-video alignment in two-stage pipelines grows overly complex, often yielding suboptimal performance.

Recent works have also explored end-to-end joint audiovideo generation [19, 24, 30, 32, 44, 53]. However, existing methods are either confined to generating ambient sounds and fail to synthesize natural human speech [19, 24, 30, 44], or struggle to attain robust audio-visual alignment [32] and produce content lacking fine-grained temporal audio-visual synchronization [53]. Taken together, to date, there remains a lack of highly generalizable and well-aligned audio-video generation method for human-centric joint generation.

To address the aforementioned challenges, we introduce UniAVGen—a unified framework tailored for joint audiovideo generation. We prioritize human-centric audio-video generation not only because this direction remains underexplored in existing works, but also because of its significant practical utility. Specifically, UniAVGen is anchored in a symmetric dual-branch joint synthesis architecture, featuring two parallel Diffusion Transformer (DiT) [37, 56] streams—one dedicated to video, the other to audio—with identical architectural designs. Crucially, this symmetry establishes representational parity and fosters a cohesive latent space, pivotal for synchronizing joint audio-video generation. To better tackle the intricacies of audio-video alignment for efficient training, we augment this core architecture with three targeted innovations, as detailed below.

First and foremost, at the core of our framework lies an Asymmetric Cross-Modal Interaction mechanism, which enables bidirectional, temporally aligned cross-modal attention. Equipped with two modal-specifically designed aligners—audio-to-video and video-to-audio—this mechanism injects fine-grained audio semantics into the video stream for precise synchronization, while imparting the temporal dynamics and identity details from video to the audio. To further strengthen this cross-modal synergy and ground it in human-related features, we introduce a FaceAware Modulation module. Specifically, this component dynamically infers a mask for facial regions using decaying supervision signals, and constrains cross-modal interaction with a gradually relaxed scope. Additionally, to enhance the expressive fidelity of generated content, we pro-

pose Modality-Aware Classifier-Free Guidance—a novel strategy that explicitly amplifies cross-modal correlation signals during the classifier-free guidance stage. This targeted enhancement significantly boosts emotional intensity in audio and motion dynamics in video, enhancing the overall realism of the generated content.

As shown in Fig. 1, beyond joint generation, our framework can also seamlessly and efficiently adapt to different conditional generation tasks, such as video-to-audio dubbing and audio-driven video synthesis. This versatility enables us to unify pivotal audio-video generation tasks under a single paradigm, eliminating the need for taskspecific model designs. Furthermore, we experimentally demonstrate that a carefully designed multi-stage training strategy can further boost the performance of joint generation through joint multi-task training.

Our key contributions are summarized as follows:

- • We present UniAVGen, a unified audio-video generation framework anchored in a dual-branch joint synthesis architecture and an asymmetric cross-modal interaction mechanism, incorporating modal-specific designs to enhance cross-modal consistency in joint generation.
- • We propose a face-aware modulation module to dynamically constrain the regions of cross-modal interaction for more efficient and aligned cross-modal learning.
- • We present modality-aware classifier-free guidance, a novel strategy that selectively amplifies cross-modal dependencies during inference.
- • Leveraging its robust architectural design, UniAVGen can be seamlessly extended to multiple audio-video generation tasks and demonstrates state-of-the-art performance.

#### 2. Related Work

To enable aligned audio-video generation, the research community has explored three primary paradigms: audiodriven video synthesis, video-to-audio synthesis, and joint audio-video generation.

Audio-driven video synthesis. This dominant paradigm typically adopts a two-stage pipeline. First, a Text-toSpeech (TTS) model [2, 3, 16, 27, 38] synthesizes desired audio waveforms from speech content. Subsequently, a separate video synthesis model generates video conditioned on audio, with a focus on lip synchronization [4, 13, 28, 39, 40, 61]. While effective for lip synchronization, this cascaded design suffers from inherent modal decoupling: audio is generated without non-verbal cues, leading to poor semantic consistency between audio and video.

Video-to-audio synthesis. The reverse paradigm [5, 10, 25, 33, 42, 45, 47, 50, 55, 62] aims to generate aligned audio for silent videos. However, it retains two key limitations: first, current methods primarily focus on ambient audio dubbing and lack the ability to synthesize natural human audio; second, it inherits the critical flaw of modal de-

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

audio

###### Flow Matching Loss

video

###### Audio Head Video Head

###### Dual-Branch Joint Synthesis

speech content video caption

× 𝑁

× 𝑁

“This was the strangest of all things

He lowers his gaze to the phone he is holding, and uses his right thumb to swipe upwards on the screen several times.

𝑧 𝑎𝑟𝑒𝑓 𝑧 𝑎𝑐𝑜𝑛𝑑 𝑧𝑎 𝑧𝑣 𝑧 𝑣𝑐𝑜𝑛𝑑 𝑧 𝑣𝑟𝑒𝑓

Dynamic Mask Prediction

that ever came to the earth from

Asymmetric Cross-Modal Interaction A2V

Video2Audio Attention

Audio2Video Attention

KV V2A KV Aligner

Aligner

outer, He did find it soon.”

Q

Q

𝑧 𝑎𝑟𝑒𝑓 𝑧 𝑎𝑐𝑜𝑛𝑑 𝑧 𝑎 𝑧𝑣 𝑧 𝑣𝑐𝑜𝑛𝑑 𝑧 𝑣𝑟𝑒𝑓

ConvNeXt V2

###### DiT Layer DiT Layer

Blocks umT5

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

𝑧 𝑎𝑟𝑒𝑓 𝑧 𝑎𝑐𝑜𝑛𝑑 𝑧 𝑎

𝑧𝑣 𝑧 𝑣𝑐𝑜𝑛𝑑 𝑧 𝑣𝑟𝑒𝑓

ref audio cond. audio noisy audio noisy video cond. video ref image

- Figure 2. Architecture of UniAVGen: A dual-branch joint synthesis framework with asymmetric cross-modal interaction, augmented by face-aware modulation. Taking a reference image and text prompt as input, it enables coherent audio-video generation.

coupling—videos are generated in an “auditory vacuum,” unaware of the audio they will eventually pair with.

Joint audio-video generation. The most holistic paradigm synthesizes audio and video simultaneously within a single unified framework [19, 24, 30, 32, 44, 49, 50, 53, 54, 58, 60, 63]. Unfortunately, most prior open-source works [19, 24, 30, 44, 48, 54] target general joint audio-video generation rather than human specifically—failing to produce highquality human audio and offering limited practical value. Notably, recent concurrent works—UniVerse-1 [53] and Ovi [32]—have begun to support human audio generation. UniVerse-1 stitches two pre-trained audio and video generation models; due to architectural asymmetry, this stitching is complex and yields limited overall performance. Ovi employs a symmetric dual-tower architecture for joint generation, delivering strong performance. However, it lacks modal-specific cross-modal interaction designs and humanspecific modulation, resulting in limited generalization in out-of-domain. In contrast, our UniAVGen addresses these gaps by integrating asymmetric cross-modal interaction and face-aware modulation; it thus achieves superior semantic synchronization and robust generalization capabilities.

#### 3. Method

Our proposed method, UniAVGen, is a unified framework for high-fidelity audio-video generation. UniAVGen takes as input a reference speaker image Iref, a video prompt Tv (a caption describing the desired motion or expression), and speech content Ta (the text to be spoken). Additionally, it supports specifying a target voice via an optional reference audio clip Xa

ref, and enables continuation or conditional generation given reference audio Xa

cond and video Xv

cond.

##### 3.1. Overview

The architecture of UniAVGen is illustrated in Fig. 2. First, we introduce a dual-branch joint synthesis framework grounded in a symmetric design. For efficient training, we directly adopt the Wan 2.2-5B video generation model [52] as the backbone for the video branch. For the audio branch, we employ the architectural template of Wan 2.1-1.3B—this shares an identical overall structure with Wan 2.2-5B, differing only in the number of channels. This symmetric strategy ensures both branches start with equivalent representational capacity and establishes a natural correspondence between feature maps across all levels. Such structural parity serves as the cornerstone for enabling effective cross-modal interactions, thereby boosting both audio-video synchronization and overall generative quality. Video branch. The video branch operates entirely in the latent space. Specifically, videos are first processed at 16 frames per second and encoded into latent representations zv using the pre-trained Variational Autoencoder (VAE) from [52]. The reference speaker image Iref and conditional video are also encoded into latent embeddings zv

cond, respectively—with the video branch’s input formed by concatenating these three latent components ztvˆ = [zv

ref and zv

###### 0 ,zv

0 ,ztv]. For the video caption Tv, it is encoded via umT5 [6] into ev, and its embeddings are fed into the Diffusion Transformer (DiT) through crossattention. Following [52], we adopt the Flow Matching paradigm [29]: here, the model uθv is trained to predict the vector field vt, with the training objective formulated as:

ref

cond

Lv = vt(ztv) − uθv(ztvˆ,t,ev) 2 . (1)

Audio branch. Following the common practice in text-toaudio (TTS) [3], audios are first sampled at 24,000 Hz and

Audio Token Video Token

Non-Aligned

Limited Context

Aligned & Informative Context

Interpolation

Overlap

Interpolation

A2V

V2A A2V

V2A

A2V

V2A

(a) Symmetric Global

(b) Symmetric Temporal-Aligned

(c) Asymmetric Temporal-Aligned

Interaction

Interaction

Interaction (Ours)

- Figure 3. Comparison of cross-modal interaction mechanisms: (a) Global Interaction is simple but poses challenges for convergence; (b) Symmetric Time-Aligned Interaction converges quickly but has limited context utilization; (c) Our Asymmetric Cross-Modal Interaction achieves a superior balance between convergence speed and performance through modal-specific interaction design.

converted into Mel spectrograms, which serve as the audio latent representation za. Similarly, the reference audio Xa

cond are also transformed into their respective latent counterparts za

ref and conditional audio Xa

cond. These three latent components are then concatenated along the temporal dimension to form the audio branch’s input ztaˆ = [za

ref and za

###### 0 ,za

0 ,zta]. The training objective for the audio branch is formulated as:

ref

cond

La = vt(zta) − uθa(ztaˆ,t,ea) 2 , (2)

where ea denotes the features of the speech content Ta extracted via ConvNeXt [57] Blocks. These features are further injected into the DiT layers through cross-attention, ensuring the audio generation process is tightly coupled with the acoustic information of the target audio.

##### 3.2. Asymmetric Cross-Modal Interaction

While the dual-branch structure establishes structural parity, achieving robust audio-video synchronization demands deep cross-modal interaction. Prior works have primarily employed two designs for this: The first is global interaction [32, 53], as shown in Fig. 3(a), where each token of the current modality interacts with all tokens of the other. While simple, it requires high training costs to converge to strong performance due to lacking explicit temporal alignment. The second is symmetric time-aligned interaction [44], as shown in Fig. 3(b), where each video token reciprocally interacts with audio tokens in its corresponding interval. Such methods typically converge faster but access limited contextual information during interaction. To better balance convergence speed and performance, we introduce a novel Asymmetric Cross-Modal Interaction mechanism, comprising two specialized aligners tailored to each modality’s unique characteristics.

Audio-to-video (A2V) aligner. The A2V aligner ensures precise semantic synchronization by injecting fine-grained audio cues into the video branch. We first reshape the hidden features to align their temporal structure: the video tokens Hv ∈ RL

v×D are reshaped to Hˆv ∈ RT×N

v×D

(where T denotes the number of video latent frames and Nv is the number of spatial tokens per frame), and the audio tokens Ha ∈ RL

a×D are reshaped to Hˆa ∈ RT×N

a×D.

Unlike Fig. 3(b), we create a contextualized audio representation for each video frame, recognizing that visual articulation is also influenced by preceding and succeeding phonemes. For the i-th video latent, we construct an audio context window Cia = [Hˆia−w,...,Hˆia,...,Hˆia+w] by concatenating audio tokens from neighboring frames within a window of size w. Boundary frames are padded by replicating the features of the first or last frame. Subsequently, we perform frame-wise cross-attention, where the video latent for each frame queries the corresponding contextualized audio latent:

H¯iv = Wov[Hˆiv + CrossAttention(Q = WqHˆiv, K = WkaCia,V = WvaCia)]. (3)

Video-to-audio (V2A) aligner. Conversely, the V2A aligner aims to embed the audio features with semantics (e.g., timbre, emotion) derived from the visual cues. In A2V, each video latent i maps to a block of k audio tokens. In contrast, for V2A, audio must perceive more precise temporal positional information rather than being confined to a single video latent. To achieve granular alignment that captures smooth visual transitions, we propose a temporal neighbor interpolation strategy. For each audio token j (corresponding to video latent i = ⌊j/k⌋), we compute a unique interpolated video context Cjv—a weighted average of latents from two temporally adjacent video latents: frame i and the subsequent frame i + 1:

Cjv = (1 − α)Hˆiv + αHˆiv+1,

where α = (j mod k)/k. (4)

For the final block of audio tokens, we simply use Cjv = HˆTv−1. This interpolated context provides a smooth, timeaware visual signal. Finally, we perform cross-attention where each audio latent queries its corresponding interpo-

lated video context:

H¯ja = Woa H ˆja + CrossAttention Q = WqaHˆja, K = WkvCjv,V = WvvCjv . (5)

Finally, H¯a and H¯v are reshaped to match the dimensions of Ha and Hv, respectively, and injected back as additional features:

Hv = Hv + H¯v, (6) Ha = Ha + H¯a. (7)

To avoid compromising the generative capability of each modality at the start of training, the output matrices Woa and Wov are both zero-initialized.

##### 3.3. Face-aware modulation

For human-centric joint generation, the critical semantic coupling is mostly concentrated in the facial region. Forcing the interaction to process the entire scene is inefficient and risks introducing spurious correlations that destabilize background elements during early training. To address this, we propose a Face-Aware Modulation module that dynamically steers interaction toward the salient regions.

Dynamic mask prediction. We introduce a lightweight auxiliary mask-prediction head operating on video features Hv

l within each interaction layer l of the denoising network. This head applies layer normalization [1], a learned affine transformation [36], a linear projection, and sigmoid activation to generate a soft mask Ml ∈ (0,1)T×N

v: Ml = σ (Wm (γ ⊙ LayerNorm(Hv

) + β) + bm), (8)

l

where ⊙ is the element-wise product. To ensure the predicted mask provides a human-aware guide for interaction, we supervise it not only via the final denoising loss but also

with an mask loss λmLm = λm l Ml − Mgt 2 using the ground-truth face mask Mgt [15]. Meanwhile, to avoid

over-constraining cross-modal interaction in later training stages, λm gradually decays to 0 over time. More discussions are provided in Sec. 4.3.2.

Mask-guided cross-modal interaction. The predicted face mask Ml refines cross-modal attention in our asymmetric aligners through two distinct mechanisms: (1) A2V interaction: We employ the mask for selective updates:

+ Ml ⊙ H¯v

= Hv

Hv

, (9) where H¯v

l

l

l

l denotes the output of A2V cross-attention at layer l. This ensures audio information precisely modulates salient regions without disrupting backgrounds during early training. (2) V2A interaction: To enable Ml to strengthen information transfer from the video’s salient regions to the audio branch, we modulate the video features Hˆv

l as Hˆv

= Ml ⊙ Hˆv

l prior to computing Eq. (4).

l

##### 3.4. Modality-aware classifier-free guidance

Classifier-Free Guidance (CFG) [21] is a cornerstone technique for enhancing conditional fidelity in generative models. However, its conventional design is inherently unimodal. Naively applying it to joint synthesis—where each branch is independently guided by its text prompt—fails to amplify critical cross-modal dependencies. The guidance signal for audio-driven video or video-influenced audio is not explicitly enhanced, limiting the model’s audiovisual synchronization. To address this, we propose Modality-Aware Classifier-Free Guidance (MA-CFG), a novel scheme that repurposes the guidance mechanism to strengthen cross-modal conditioning. Our key insight is that a single, shared unconditional estimate can serve as the baseline for guiding both modalities simultaneously. This is achieved by performing one forward pass where the conditioning signals for both cross-modal interactions are nullified, which is equivalent to unimodal inference.

Specifically, we define the unconditional estimate for the audio and video modalities (without cross-modal interaction) as uθ

, and the estimate with cross-modal interaction as uθ

and uθ

a

v

. Then, MA-CFG for each modalities can be formulated as:

a,v

), (10) uˆa = uθ

a,v − uθ

uˆv = uθ

+ sv(uθ

v

v

), (11)

a,v − uθ

+ sa(uθ

a

a

where sv and sa are coefficients controlling the guidance strength for the video and audio modalities, respectively.

##### 3.5. Multi-task unification

As shown in Fig. 2, leveraging the symmetry and flexibility of UniAVGen’s overall design, we support multiple input combinations to handle distinct tasks: (1) Joint audiovideo generation: The default core task, which takes only text and a reference image as input to generate aligned audio and video. (2) Joint generation with reference audio: Compared to (1), it supports input of a custom reference audio to control the speaker’s timbre. Notably, latents of the reference audio skip cross-modal interaction to preserve the timbre consistency. (3) Joint audio-video continuation: It performs continuation given conditional audio and conditional video. For this task, conditional information also participates in cross-modal interaction to ensure temporal continuity, while its features remain unaffected by interaction to preserve conditional information. (4) Video-to-audio dubbing: When only conditional video is provided to the video branch, the model generates corresponding emotionand expression-aligned audio based on the video and text. A reference audio can be optionally provided to anchor timbre, and the reference image for the video branch is filled with the first frame of the conditional video. (5) audiodriven video synthesis: When only conditional audio is provided to the audio branch, the model generates expression-

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Universe-1

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

Ovi

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

UniAVGen (Ours)

[Figure 67]

[Figure 68]

(a) (b)

- Figure 4. Visual comparisons of UniAVGen against concurrent methods Ovi and UniVerse-1. Specifically, Example (a) uses an indistribution real human image: UniAVGen and Ovi generate high-fidelity, well-aligned audio-video, while UniVerse-1 is nearly static. Example (b) uses an out-of-distribution (OOD) anime image: Ovi lacks aligned lip/motions (poor generalization), UniVerse-1 stays static with noisy audio; in contrast, our model shows strong generalization, producing coherent, aligned audio-video matching the anime input.

###### Table 1. Quantitative comparison with different methods.

Audio Quality Video Quality Audio-Video Consistency PQ(↑) CU(↑) WER(↓) SC(↑) DD(↑) IQ(↑) LS(↑) TC(↑) EC(↑)

Methods Joint Training

Parameters (S+V)

Samples

Two-stage Generation

OmniAvatar [17] - 21.1B 8.15 7.41 0.152 0.987 0.000 0.721 6.34 0.454 0.349 Wan-S2V [18] - 16.6B 8.15 7.41 0.152 0.991 0.130 0.750 6.35 0.481 0.375

Joint Generation

JavisDiT [30] 10.1M 3.7B 5.21 3.93 0.986 0.965 0.373 0.716 1.23 0.776 0.388 Universe-1 [53] 6.4M 7.1B 4.56 4.29 0.296 0.985 0.080 0.733 1.21 0.573 0.300 Ovi [32] 30.7M 10.9B 6.03 6.01 0.216 0.972 0.360 0.774 6.48 0.828 0.558 UniAVGen 1.3M 7.1B 7.00 6.62 0.151 0.973 0.410 0.779 5.95 0.832 0.573

and motion-aligned video based on the audio and text. For deeper insights into how multi-task unification facilitates joint generation, we refer the reader to Sec. 4.3.4.

#### 4. Experiment

##### 4.1. Implementation details

UniAVGen is trained in three stages. Stage 1 focuses on training the audio branch in isolation: here, we only optimize the audio network using its dedicated objective La. The training data uses the English subset of the multilingual audio dataset Emilia [20]. We adopt a batch size of 256, a learning rate of 2 × 10−5, and the AdamW [31] optimizer with parameters β1 = 0.9, β2 = 0.999, ϵ = 1e−8, for a total of 160k training steps. Once the audio branch achieves robust generative performance, we proceed to Stage 2—endto-end joint training. In this phase, both branches are cooptimized via a composite loss Ljoint = Lv + La + λmLm, where λm is initialized to 0.1 and decays linearly to 0. The training data here uses an internally collected real human

audio-video dataset. We use a batch size of 32, a learning rate of 5e−6, and the same optimizer settings as Stage 1, for a total of 30k training steps. Stage 3 involves multi-task learning built on Stage 2, with training configurations consistent with Stage 2. In the training process, the ratio of the 5 tasks mentioned in Sec. 3.5 is set to 4:1:1:2:2, with a total of 10k training steps. Inference details are provided in the supplementary materials.

##### 4.2. Comparison with previous methods

Compared methods. We select representative methods from two categories of paradigms for comparison: (1) Twostage Generation: Since we focus on human=centric joint audio-video generation, we first generate audio using F5TTS [3], then generate video from audio with state-of-theart OmniAvatar [17] and Wan-S2V [18]. (2) Joint Generation: We select several latest open-source models for comparison: JavisDiT [30] focuses on general audio-video joint generation without human audio optimization, UniVerse1 [53] adopts dual pre-trained model stitching, and Ovi [32]

employs a symmetric dual-tower architecture with symmetric global cross-model interactions.

Evaluation setting. To mitigate test set leakage and better align with the objectives of audio-video generation, we constructed 100 test samples that are not sampled from existing videos. Each sample comprises a reference image, a video caption, and audio content. To comprehensively validate the model’s generalization capability—particularly across diverse visual domains—half of these reference images are real-world captures, while the remaining half consists of AIGC-generated content or anime-style visuals.

For evaluation, we measure model performance across three critical dimensions: (1) Audio Quality: Following [53], we adopt AudioBox-Aesthetics [51] to evaluate two core metrics: Production Quality (PQ) and Content Usefulness (CU). Additionally, we leverage the Whisper-large-

- v3 [43] model to compute the Word Error Rate (WER) of the generated audio. (2) Video Quality: We utilize VBench [23]—a widely recognized video evaluation benchmark—to assess video generation quality, focusing on three key metrics: Subject Consistency (SC), Dynamic Degree (DD), and Imaging Quality (IQ). (3) Audio-Video Consistency: Notably, this dimension encompasses three sub-aspects: Lip Synchronization (LS), Timbre Consistency (TC), and Emotion Consistency (EC). Specifically, we employ SyncNet [7]’s confidence score to evaluate lipsync consistency. For timbre and emotion consistency, as no open-source methodologies currently exist to quantify such cross-modal alignment, we instead leverage the multimodal large language model Gemini-2.5-Pro for evaluation. We set the outputs scores within the range [0,1]. A detailed system prompt (with implementation specifics provided in the supplementary materials) defines the scoring criteria, and the final score for each of these two metrics is computed as the average of three independent evaluations.

Quantitative comparison. Tab. 1 summarizes quantitative comparisons between our method and existing baselines: For audio quality, our method demonstrates significant superiority over other joint generation approaches in both acoustic quality and aesthetic metrics, with its WER further outperforming F5-TTS—underscoring stronger alignment with linguistic content. Turning to video quality, while twostage methods exhibit stronger identity consistency, their dynamism scores are near-zero, reflecting their inability to generate actions congruent with audio-driven emotions; in contrast, our method achieves the highest dynamism and aesthetic quality while retaining identity consistency comparable to state-of-the-art alternatives. Notably, for the critical audio-video consistency metric, our method—despite utilizing the fewest effective training samples—shows clear advantages over competitors in timbre and emotion alignment, while maintaining lip-sync performance on par with leading methods. Such training efficiency is attributed to

Table 2. Ablation studies on the design of interaction.

Interaction

LS(↑) TC(↑) EC(↑) A2V V2A

- (1) SGI SGI 3.46 0.667 0.459
- (2) STI STI 3.73 0.685 0.472
- (3) STI ATI 3.88 0.705 0.492
- (4) ATI STI 3.97 0.691 0.483

- (5) ATI ATI 4.09 0.725 0.504

[Figure 69]

Generated frames

[Figure 70]

Predicted masks w/ fixed 𝝀𝒎

[Figure 71]

Predicted masks w/ decaying 𝝀𝒎

Figure 5. Visual comparisons of predicted masks with fixed λm and decaying λm. Zoom in for the best view.

the proposed asymmetric cross-modal interaction mechanism and face-aware modulation.

Qualitative comparison. Fig. 4 presents visual comparisons of UniAVGen against recent concurrent methods Ovi and UniVerse-1. Specifically, Example (a) uses a real human image aligned with the training distribution: both UniAVGen and Ovi generate high-fidelity audio and videos, with motions and emotions tightly aligned to the audio, whereas UniVerse-1 exhibits near-static behavior. Example (b) employs an anime image—out-of-distribution (OOD) relative to the training set: Ovi fails to produce lip movements and motions aligned with the audio, highlighting its constrained generalization capacity; UniVerse-1 remains static and generates noisy audio. In contrast, our model exhibits robust generalization, generating coherent audio and motions that align with the input anime image.

##### 4.3. Ablations

For efficient ablation studies, unless otherwise specified, the following ablation results default to those from the first 10k steps of Stage 2 training. The colored background indicates our default setting.

###### 4.3.1. Cross-modal interaction design

As a core architectural component, we perform detailed ablation studies on the design of the cross-modal interaction module, as shown in Tab. 2. Consistent with the three mechanisms depicted in Fig. 3, this table denotes Symmetric Global Interaction as SGI, Symmetric TemporalAligned Interaction as STI, and our proposed Asymmetric Temporal-Aligned Interaction as ATI. SGI exhibits substantial performance deficits compared to STI with the same number of training steps—this confirms that temporalaligned designs more effectively facilitate model conver-

###### Table 3. Ablation studies on the face-aware modulation.

###### Settings LS(↑) TC(↑) EC(↑)

- (a) without FAM 3.89 0.705 0.489
- (b) unsupervised FAM 3.92 0.701 0.492
- (c) FAM with fixed λm 4.11 0.719 0.497

- (d) FAM with decaying λm 4.09 0.725 0.504

gence. Relative to STI, our proposed ATI delivers significant improvements in both A2V and V2A tasks: For A2V, ATI more robustly enhances timbre and emotion consistency between audio and video, validating that it indeed strengthens audio’s perception of facial expressions and movements across adjacent video frames; for V2A, it further boosts lip synchronization accuracy, confirming that it enables video frames to better capture information from adjacent audio segments.

###### 4.3.2. Effectiveness of face-aware modulation

We evaluate the effectiveness of Face-aware Modulation (FAM) through two key analyses. First, to confirm that our lightweight dynamic mask prediction module can reliably localize valid facial regions, we visualize average face masks predicted across layers with fixed λm in Fig. 5. This visualization demonstrates that our module effectively pinpoints face-salient regions. Additionally, when trained with decaying λm, the predicted masks still effectively capture facial regions while increasing weights on body regions—thereby enhancing the flexibility of cross-modal interactions. To further validate the FAM strategy, we compare performance under four configurations in Table 4: without FAM, unsupervised FAM, FAM with fixed λm, and FAM with decaying λm. Two critical insights emerge: (1) Supervised FAM yields significant improvements in overall audio-video consistency, indicating that constrained masks facilitate training convergence; (2) Decaying loss weights outperform fixed weights, indicating that gradually relaxing constraints on interaction locations during training further enhances the timbre and emotion consistency.

###### 4.3.3. Modality-aware classifier-free guidance

To demonstrate the effectiveness of MA-CFG, we provide visual comparisons in Fig. 6. Without MA-CFG, while audio and video remain generally consistent, the generated character’s emotions and body movements are insufficiently aligned with the audio’s emotional cues. With MA-CFG, by contrast, the jointly generated character exhibits facial expressions and body movements more tightly aligned with audio emotions, alongside more natural lip synchronization.

###### 4.3.4. Analysis of training strategies

As shown in Fig. 7, we compare the LC metric of our model under three distinct training strategies: train joint generation only (denoted as JGO), train joint generation first then multi-task learning (denoted as JFML), and multi-task training throughout (denoted as MTO). First, JGO exhibits a

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

- (a) w/ MA-CFG
- (b) w/o MA-CFG

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

###### Figure 6. Visual comparisons of joint generation results with and without MA-CFG. Zoom in for the best view.

6.0

5.5

5.0

###### LC

4.5

JGO

JFML

MTO

4.0

20k 25k 30k 35k 40k

# Training Steps

Figure 7. Comparisons of different training strategies.

lower performance ceiling than JFML, which we attribute to the ability of multi-task joint training to further strengthen cross-modal interaction. For instance, video-to-audio dubbing enhances the audio branch’s capture of conditional information from video, while audio-driven video synthesis deepens the video branch’s perception of the audio branch. Second, MTO demonstrates slower convergence speed than both JGO and JFML. This likely stems from the fact that joint generation is more task-intensive than conditional generation tasks—training the model with conditional tasks from the start may cause it to get trapped in local optima. In contrast, pre-training with joint generation lays a solid foundation for subsequent conditional tasks, allowing JFML to achieve the best overall performance.

#### 5. Conclusion

This work introduced UniAVGen, a unified framework for generating high-quality audio and video jointly. At its core lies the asymmetric cross-modal interaction (ATI). Unlike symmetric or global interaction designs, ATI enables modality-specific temporal alignment: it allows audio to efficiently perceive dynamics across adjacent video frames while empowering video frames to capture audio cues from neighboring audio segments. Complementing ATI, we further propose the Face-aware Modulation (FAM) module, which dynamically localizes facial regions and enhances interaction precision. Additionally, we introduce MA-CFG during inference to explicitly strengthen cross-modal influences. Overall, UniAVGen sets a new benchmark for audiovideo generation and paves the way for more practical and versatile multi-modal generation systems.

Acknowledgements. This work is supported by the Basic Research Program of Jiangsu (No. BK20250009), the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (No. JYB2025XDXM118), and the Collaborative Innovation Center of Novel Software Technology and Industrialization.

#### References

- [1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450,

2016. 5

- [2] Edresson Casanova, Kelly Davis, Eren G¨olge, G¨orkem G¨oknar, Iulian Gulea, Logan Hart, Aya Aljafari, Joshua Meyer, Reuben Morais, Samuel Olayemi, et al. Xtts: a massively multilingual zero-shot text-to-speech model. arXiv preprint arXiv:2406.04904, 2024. 2
- [3] Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, Jian Zhao, Kai Yu, and Xie Chen. F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching. arXiv preprint arXiv:2410.06885, 2024. 2, 3, 6
- [4] Yi Chen, Sen Liang, Zixiang Zhou, Ziyao Huang, Yifeng Ma, Junshu Tang, Qin Lin, Yuan Zhou, and Qinglin Lu. Hunyuanvideo-avatar: High-fidelity audio-driven human animation for multiple characters. arXiv preprint arXiv:2505.20156, 2025. 2
- [5] Ho Kei Cheng, Masato Ishii, Akio Hayakawa, Takashi Shibuya, Alexander Schwing, and Yuki Mitsufuji. Mmaudio: Taming multimodal joint training for high-quality video-toaudio synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28901–28911, 2025. 2
- [6] Hyung Won Chung, Noah Constant, Xavier Garcia, Adam Roberts, Yi Tay, Sharan Narang, and Orhan Firat. Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining. arXiv preprint arXiv:2304.09151,

2023. 3

- [7] Joon Son Chung and Andrew Zisserman. Out of time: automated lip sync in the wild. In Asian conference on computer vision, pages 251–263. Springer, 2016. 7, 2
- [8] Alibaba Cloud. Wan2.5. https://wan.video/, 2025. 1
- [9] Gaoxiang Cong, Yuankai Qi, Liang Li, Amin Beheshti, Zhedong Zhang, Anton Hengel, Ming-Hsuan Yang, Chenggang Yan, and Qingming Huang. Styledubber: Towards multiscale style learning for movie dubbing. In Findings of the Association for Computational Linguistics: ACL 2024, pages 6767–6779, 2024. 2
- [10] Gaoxiang Cong, Liang Li, Jiadong Pan, Zhedong Zhang, Amin Beheshti, Anton van den Hengel, Yuankai Qi, and Qingming Huang. Flowdubber: Movie dubbing with llmbased semantic-aware learning and flow matching based voice enhancing. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 905–914, 2025. 2
- [11] Gaoxiang Cong, Jiadong Pan, Liang Li, Yuankai Qi, Yuxin Peng, Anton van den Hengel, Jian Yang, and Qingming

- Huang. Emodubber: Towards high quality and emotion controllable movie dubbing. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15863– 15873, 2025. 2
- [12] Martin Cooke, Jon Barker, Stuart Cunningham, and Xu Shao. An audio-visual corpus for speech perception and automatic speech recognition. The Journal of the Acoustical Society of America, 120(5):2421–2424, 2006. 2
- [13] Jiahao Cui, Hui Li, Yun Zhan, Hanlin Shang, Kaihui Cheng, Yuqi Ma, Shan Mu, Hang Zhou, Jingdong Wang, and Siyu Zhu. Hallo3: Highly dynamic and realistic portrait image animation with diffusion transformer networks. arXiv e-prints, pages arXiv–2412, 2024. 2
- [14] Google DeepMind. Veo3. https://deepmind. google/models/veo/, 2025. 1
- [15] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-shot multilevel face localisation in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5203–5212, 2020. 5
- [16] Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, et al. Cosyvoice 2: Scalable streaming speech synthesis with large language models. arXiv preprint arXiv:2412.10117, 2024. 2
- [17] Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, and Steven Hoi. Omniavatar: Efficient audio-driven avatar video generation with adaptive body animation. arXiv preprint arXiv:2506.18866, 2025. 6, 2
- [18] Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Dechao Meng, Jinwei Qi, Penchong Qiao, Zhen Shen, Yafei Song, et al. Wan-s2v: Audio-driven cinematic video generation. arXiv preprint arXiv:2508.18621, 2025. 6, 2
- [19] Moayed Haji-Ali, Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Alper Canberk, Kwot Sin Lee, Vicente Ordonez, and Sergey Tulyakov. Av-link: Temporally-aligned diffusion features for cross-modal audio-video generation. arXiv preprint arXiv:2412.15191, 2024. 2, 3
- [20] Haorui He, Zengqiang Shang, Chaoren Wang, Xuyuan Li, Yicheng Gu, Hua Hua, Liwei Liu, Chen Yang, Jiaqi Li, Peiyang Shi, et al. Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation. In 2024 IEEE Spoken Language Technology Workshop (SLT), pages 885–890. IEEE, 2024. 6
- [21] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5
- [22] Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, and Qinglin Lu. Hunyuancustom: A multimodal-driven architecture for customized video generation. arXiv preprint arXiv:2505.04512, 2025. 2
- [23] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 7
- [24] Masato Ishii, Akio Hayakawa, Takashi Shibuya, and Yuki Mitsufuji. A simple but strong baseline for sounding

- video generation: Effective adaptation of audio and video diffusion models for joint generation. arXiv preprint arXiv:2409.17550, 2024. 2, 3
- [25] Yujin Jeong, Yunji Kim, Sanghyuk Chun, and Jiyoung Lee. Read, watch and scream! sound generation from text and video. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 17590–17598, 2025. 2
- [26] Tuomas Kynk¨a¨anniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. Advances in Neural Information Processing Systems, 37:122458–122483, 2024. 1
- [27] Yinghao Aaron Li, Cong Han, and Nima Mesgarani. Styletts: A style-based generative model for natural and diverse textto-speech synthesis. IEEE Journal of Selected Topics in Signal Processing, 2025. 2
- [28] Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, and Chao Liang. Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models. arXiv preprint arXiv:2502.01061, 2025. 2
- [29] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3
- [30] Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Rongxin Jiang, Jiebo Luo, Hao Fei, et al. Javisdit: Joint audio-video diffusion transformer with hierarchical spatio-temporal prior synchronization. arXiv preprint arXiv:2503.23377, 2025. 2, 3, 6
- [31] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6
- [32] Chetwin Low, Weimin Wang, and Calder Katyal. Ovi: Twin backbone cross-modal fusion for audio-video generation. arXiv preprint arXiv:2510.01284, 2025. 2, 3, 4, 6, 1
- [33] Simian Luo, Chuanhao Yan, Chenxu Hu, and Hang Zhao. Diff-foley: Synchronized video-to-audio synthesis with latent diffusion models. Advances in Neural Information Processing Systems, 36:48855–48876, 2023. 2
- [34] Rang Meng, Xingyu Zhang, Yuming Li, and Chenguang Ma. Echomimicv2: Towards striking, simplified, and semibody human animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5489–5498,

2025. 2

- [35] Openai. Sora2. https://openai.com/zh-HansCN/index/sora-2/, 2025. 1
- [36] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu. Semantic image synthesis with spatially-adaptive normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2337–2346,

2019. 5

- [37] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2

- [38] Puyuan Peng, Po-Yao Huang, Shang-Wen Li, Abdelrahman Mohamed, and David Harwath. Voicecraft: Zero-shot speech editing and text-to-speech in the wild. arXiv preprint arXiv:2403.16973, 2024. 2

- [39] Ziqiao Peng, Wentao Hu, Yue Shi, Xiangyu Zhu, Xiaomei Zhang, Hao Zhao, Jun He, Hongyan Liu, and Zhaoxin Fan. Synctalk: The devil is in the synchronization for talking head synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 666–676,

2024. 2

- [40] Ziqiao Peng, Wentao Hu, Junyuan Ma, Xiangyu Zhu, Xiaomei Zhang, Hao Zhao, Hui Tian, Jun He, Hongyan Liu, and Zhaoxin Fan. Synctalk++: High-fidelity and efficient synchronized talking heads synthesis using gaussian splatting. arXiv preprint arXiv:2506.14742, 2025. 2
- [41] Ziqiao Peng, Jiwen Liu, Haoxian Zhang, Xiaoqiang Liu, Songlin Tang, Pengfei Wan, Di Zhang, Hongyan Liu, and Jun He. Omnisync: Towards universal lip synchronization via diffusion transformers. arXiv preprint arXiv:2505.21448,

2025. 2

- [42] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

2024. 2

- [43] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023. 7
- [44] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. Mm-diffusion: Learning multi-modal diffusion models for joint audio and video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10219–10228, 2023. 2, 3, 4
- [45] Sizhe Shan, Qiulin Li, Yutao Cui, Miles Yang, Yuehai Wang, Qun Yang, Jin Zhou, and Zhao Zhong. Hunyuanvideofoley: Multimodal diffusion with representation alignment for high-fidelity foley audio generation. arXiv preprint arXiv:2508.16930, 2025. 2
- [46] Hubert Siuzdak. Vocos: Closing the gap between timedomain and fourier-based neural vocoders for high-quality audio synthesis. arXiv preprint arXiv:2306.00814, 2023. 1
- [47] Kim Sung-Bin, Jeongsoo Choi, Puyuan Peng, Joon Son Chung, Tae-Hyun Oh, and David Harwath. Voicecraft-dub: Automated video dubbing with neural codec language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14623–14632, 2025. 2
- [48] Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable diffusion. Advances in Neural Information Processing Systems, 36:16083–16099, 2023. 3
- [49] OpenMOSS Team, Donghua Yu, Mingshu Chen, Qi Chen, Qi Luo, Qianyi Wu, Qinyuan Cheng, Ruixiao Li, Tianyi Liang, Wenbo Zhang, et al. Mova: Towards scalable and synchronized video-audio generation. arXiv preprint arXiv:2602.08794, 2026. 3
- [50] Zeyue Tian, Yizhu Jin, Zhaoyang Liu, Ruibin Yuan, Xu Tan, Qifeng Chen, Wei Xue, and Yike Guo. Audiox: Diffusion transformer for anything-to-audio generation. arXiv preprint arXiv:2503.10522, 2025. 2, 3

- [51] Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hoffman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, et al. Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound. arXiv preprint arXiv:2502.05139, 2025. 7
- [52] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 3
- [53] Duomin Wang, Wei Zuo, Aojie Li, Ling-Hao Chen, Xinyao Liao, Deyu Zhou, Zixin Yin, Xili Dai, Daxin Jiang, and Gang Yu. Universe-1: Unified audio-video generation via stitching of experts. arXiv preprint arXiv:2509.06155, 2025. 2, 3, 4, 6, 7, 1
- [54] Kai Wang, Shijian Deng, Jing Shi, Dimitrios Hatzinakos, and Yapeng Tian. Av-dit: Taming image diffusion transformers for efficient joint audio and video generation. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 10486–10495, 2025. 3
- [55] Le Wang, Jun Wang, Chunyu Qiang, Feng Deng, Chen Zhang, Di Zhang, and Kun Gai. Audiogen-omni: A unified multimodal diffusion transformer for video-synchronized audio, speech, and song generation. arXiv preprint arXiv:2508.00733, 2025. 2
- [56] Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. Ddt: Decoupled diffusion transformer. arXiv preprint arXiv:2504.05741, 2025. 2
- [57] Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie. Convnext v2: Co-designing and scaling convnets with masked autoencoders. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16133– 16142, 2023. 4
- [58] Yazhou Xing, Yingqing He, Zeyue Tian, Xintao Wang, and Qifeng Chen. Seeing and hearing: Open-domain visualaudio generation with diffusion latent aligners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7151–7161, 2024. 3
- [59] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al. Qwen3-omni technical report. arXiv preprint arXiv:2509.17765, 2025. 2
- [60] Ruihan Yang, Hannes Gamper, and Sebastian Braun. Cmmd: Contrastive multi-modal diffusion for video-audio conditional modeling. In European Conference on Computer Vision, pages 214–226. Springer, 2024. 3
- [61] Guy Yariv, Itai Gat, Sagie Benaim, Lior Wolf, Idan Schwartz, and Yossi Adi. Diverse and aligned audio-to-video generation via text-to-video model adaptation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 6639– 6647, 2024. 2
- [62] Yiming Zhang, Yicheng Gu, Yanhong Zeng, Zhening Xing, Yuancheng Wang, Zhizheng Wu, and Kai Chen. Foleycrafter: Bring silent videos to life with lifelike and synchronized sounds. arXiv preprint arXiv:2407.01494, 2024. 2
- [63] Youliang Zhang, Zhaoyang Li, Duomin Wang, Jiahe Zhang, Deyu Zhou, Zixin Yin, Xili Dai, Gang Yu, and Xiu Li.

Speakervid-5m: A large-scale high-quality dataset for audiovisual dyadic interactive human generation. arXiv preprint arXiv:2507.09862, 2025. 3

## UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions

### Supplementary Material

#### 6. Additional implementation details

- 6.1. Context window size for A2V aligner

In the Asymmetric Cross-Modal Interaction mechanism (Sec. 3.2 of the main paper), the audio context window size w for A2V aligner is set to 12. Specifically, for the i-th video latent frame, the audio context window Cia concatenates audio tokens from i−12 to i+21 (i.e., 2 audio segments in total), ensuring sufficient contextual phoneme information for precise lip synchronization. Boundary frames (when i− 12 < 0 or i + 12 ≥ T) are padded by replicating the first or last audio frame’s features to avoid information loss.

- 6.2. Temporal alignment in interaction

Due to the design of existing video VAEs, each video latent, except the first one which corresponds to a single frame, is associated with four consecutive frames. To ensure precise temporal alignment during cross-modal interaction, we explicitly account for this characteristic of video latents. Specifically, for A2V alignment, we first compute the audio window size per frame by dividing the number of audio tokens by the actual number of video frames. We then determine the corresponding audio window for each video latent, meaning the effective window for the first latent is a quarter the size of those for subsequent latents. For V2A alignment, we first upsample the video latents to match audio’s fine-grained temporal resolution: each latent (except the first) is replicated four times. With this temporal alignment, we then compute the video context.

- 6.3. Inference details

We employ the Euler ODE solver with 50 sampling steps and leverage the Vocos vocoder [46] to convert generated log mel spectrograms into audio signals. For MA-CFG, we empirically set sv = 3 and sa = 2. To stabilize audiovisual quality while using CFG, we further adopt CFG interval [26], which restricts classifier-free guidance exclusively to the high-frequency generation phase with the interval set to [0.5,1]. Additionally, for efficiency, we set the text condition to empty during unimodal sampling in MACFG, which further reinforces text control.

#### 7. System prompt for evaluation

We use the following system prompts to evaluate Timbre Consistency (TC) and Emotion Consistency (EC) via Gemini-2.5-Pro. The prompt is designed to ensure objective, reproducible scoring (0-1 scale, 2 decimal places):

Table 4. User study statistics.

###### Methods AQ(↑) VQ(↑) AVC(↑)

Universe-1 [53] 2.35% 0.00% 0.00% Ovi [32] 28.75% 37.40% 25.70% UniAVGen (ours) 68.90% 62.60% 74.30%

You are an expert in audio and video understanding. Now you will receive an audio and video clip. Please judge the consistency between the timbre and emotion of the audio and video, and give a score between 0 and 1.

For timbre evaluation (score a), it is divided into 5 grades based on gender and age matching:

- 1. 0 points: Completely inconsistent (e.g., video shows a woman but audio is a man’s voice; age difference is extremely obvious)
- 2. 0.25 points: Severely inconsistent (one of gender or age is seriously mismatched, the other has slight inconsistency)
- 3. 0.5 points: Partially inconsistent (one of gender or age is mismatched, the other is consistent)
- 4. 0.75 points: Basically consistent (gender and age are roughly matched, with minor details inconsistent)
- 5. 1 point: Perfectly consistent (gender and age are completely matched without any differences)

For emotion evaluation (score b), it is divided into 5 grades based on frame-level emotion matching and body language correspondence:

- 1. 0 points: No correspondence at all (no frame matches, body language has nothing to do with audio)
- 2. 0.25 points: Rarely corresponding (very few frames match, body language basically does not correspond)
- 3. 0.5 points: Partially corresponding (about half of the frames match, body language partially corresponds)
- 4. 0.75 points: Basically corresponding (most frames match, body language roughly corresponds)
- 5. 1 point: Perfectly corresponding (every frame matches, body language fits audio perfectly) You should return the following JSON format: {"score":[a, b],"reason":"xxx"}

Where a is the timbre score, b is the emotion score, and reason is the specific reason for the score, which should not exceed 100 words.

Each sample is evaluated 3 times independently, and the average score is reported.

#### 8. User study

A comprehensive user study was also performed to further underscore the advantages of our method. Participants evaluated and selected the top-generated videos by assessing

###### Table 5. Results on GRID [12] under Dubbing Setting 3.0.

###### Methods LSE-C(↑) LSE-D(↓) WER(↓)

StyleDubber [9] 5.94 9.75 15.40 EmoDubber [11] 7.25 6.83 14.72 UniAVGen (ours) 7.59 6.11 10.64

###### Table 6. Results on EMTD [34].

###### Methods LSE-C(↑) LSE-D(↓) FID(↓) FVD(↓)

OmniAvatar [17] 7.19 6.90 45.02 459.44 Wan-S2V [18] 7.24 6.92 44.02 451.44 UniAVGen (ours) 7.05 6.85 43.97 469.85

audio quality (AQ), video quality (VQ), and overall audiovisual coherence (AVC). Results from 34 participants, presented in Tab. 4, reveal that our approach achieves superior overall audio-visual quality and enhanced consistency between audio and video compared to recent methods.

#### 9. Evaluation on conditional tasks

While UniAVGen is primarily designed for high-quality joint audio-visual generation, we further evaluate its performance on public benchmarks of other conditional generation tasks after multi-task joint training to ensure the completeness of this work. For video-to-audio dubbing, we test on the widely used GRID benchmark [12] with three metrics: LSE-C [7], LSE-D [7] and WER. As shown in Tab. 5, we compare performance under Dubbing Setting 3.0, which adopts unseen speakers as reference audio. Without complex or task-specific designs, our model achieves superior consistency and lower WER. For audio-to-video synthesis, we utilize the half-body animation benchmark EMTD [34] and compare against state-of-the-art audio-driven models [17, 18]. As presented in Tab. 6, our model attains nearSOTA performance with only simple multi-task fine-tuning. These results further validate the practicality and generalization capability of UniAVGen.

#### 10. Extended ablation studies

We supplement additional ablation experiments to validate the robustness of our core designs:

##### 10.1. Exploration of interaction insertion positions

Rationally integrating the interaction module is another critical consideration, which we address from two perspectives. First, at the layer-level (see Tab. 7), we explore four schemes: inserting into all layers, the first half of layers, the last half of layers, and interleaved insertion. Interleaved insertion yields the best results, indicating that appropriate yet not excessive cross-modal interaction better enhances the stability of multi-modal learning. Second, at the operationlevel: built on the DiT architecture of Wan2.2-5b, each DiT block comprises self-attention, text cross-attention, and

- Table 7. Ablation studies on the layer-level insertion. Settings LS(↑) TC(↑) EC(↑)

- (a) all layers 4.01 0.713 0.497
- (b) first half of layers 4.02 0.719 0.500
- (c) last half of layers 3.79 0.710 0.493

- (d) interleaved layers 4.09 0.725 0.504

- Table 8. Ablation studies on the operation-level insertion. Settings LS(↑) TC(↑) EC(↑)

- 1) before FFN 3.85 0.715 0.490
- 2) before cross-attention 3.98 0.721 0.499

- 3) before self-attention 4.09 0.725 0.504 Table 9. Ablation studies on the MA-CFG.

Settings LS(↑) TC(↑) EC(↑) IQ(↑)

- (a) no CFG 5.75 0.821 0.553 0.760
- (b) vanilla CFG 5.81 0.824 0.562 0.778
- (c) MA-CFG 6.29 0.841 0.580 0.752

- (d) MA-CFG under t ∈ [0.5,1] 5.95 0.832 0.573 0.779

FFN. We ablate the module insertion at three distinct positions: 1) before self-attention, 2) before cross-attention, and 3) before FFN. As shown in Tab. 8, position 1) achieves the optimal performance, suggesting that fully preserving the operational flow of each block facilitates better inheritance of pretrained capabilities.

##### 10.2. Validation of MA-CFG’s effectiveness

As shown in Tab. 9, we compare the performance of four testing strategies: no CFG, vanilla CFG, MA-CFG, and MA-CFG with the interval [0.5,1]. While vanilla CFG improves image quality, its enhancement on modal consistency is negligible. In contrast, MA-CFG significantly boosts audio-visual alignment metrics but slightly degrades image quality. By incorporating the constrained CFG interval, MA-CFG achieves simultaneous improvements in both image quality and modal alignment.

#### 11. Limitations

Currently, while UniAVGen performs well in speech-video generation, it lacks video-aligned ambient sound generation. Additionally, its ability to generate audio for multiperson scenarios remains constrained by the inflexible text encoder. For future efforts, we will first collecting more general high-quality audio-video data. Meanwhile, we plan to enhance the text encoder of the audio branch, specifically by adopting multi-modal large language models like QwenOmni3 [59], to enable multi-person scenarios generation.

