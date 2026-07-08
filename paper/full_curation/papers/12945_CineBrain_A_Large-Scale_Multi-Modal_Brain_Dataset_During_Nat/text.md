[Figure 1]

### CineBrain: A Large-Scale Multi-Modal Brain Dataset During Naturalistic Audiovisual Narrative Processing

##### Jianxiong Gao1, Yichang Liu1, Baofeng Yang1, Jianfeng Feng1, Yanwei Fu1,2 1Fudan University 2Shanghai Innovation Institute

[Figure 2]

[Figure 3]

## arXiv:2503.06940v2[cs.CV]13Dec2025

fMRI signals

EEG signals

# CineBrain

162,000 frames of 3T whole-brain fMRI

1000 Hz 64-channel EEG with ECG totaling 2,160 minutes

[Figure 4]

[Figure 5]

[Figure 6]

###### Audiovisual Series

[Figure 7]

[Figure 8]

MRI EEG

[Figure 9]

[Figure 10]

[Figure 11]

LCD Earphone

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Decode

Vision Auditory

[Figure 17]

[Figure 18]

Stimulus

540 minutes of human-centric videos

540 minutes of speech-centric audio

CineSync

[Figure 19]

[Figure 20]

Video Stimulus

Audio Stimulus

Figure 1. Overview of CineBrain. To leverage the complementary strengths of fMRI and EEG, CineBrain provides simultaneous audiovisual stimuli to participants while recording their EEG and fMRI signals. Engaging narrative-driven content from the television series The Big Bang Theory is utilized to facilitate the study of complex brain dynamics and multimodal neural decoding.

#### Abstract

ing dynamic video using a Multi-Modal Fusion Encoder and a Neural Latent Decoder. CineSync achieves stateof-the-art performance in dynamic reconstruction, leveraging the complementary strengths of fMRI and EEG to improve visual fidelity. Our analysis shows that auditory cortical activations enhance decoding accuracy, highlighting the role of auditory input in visual perception. Project Page: https://jianxgao.github.io/CineBrain.

Most research decoding brain signals into images, often using them as priors for generative models, has focused only on visual content. This overlooks the brain’s natural ability to integrate auditory and visual information, for instance, sound strongly influences how we perceive visual scenes. To investigate this, we propose a new task of reconstructing continuous video stimuli from multimodal brain signals recorded during audiovisual stimulation. To enable this, we introduce CineBrain, the first large-scale dataset that synchronizes fMRI and EEG during audiovisual viewing, featuring six hours of The Big Bang Theory episodes for crossmodal alignment. We also conduct the first systematic exploration of combining fMRI and EEG for video reconstruction and present CineSync, a framework for reconstruct-

#### 1. Introduction

In the vision community, considerable effort has been devoted to decoding brain signals—particularly fMRI—into images [7, 37, 38], videos [8, 23], and even 3D shapes [13], often using neural signals as priors to guide visual generative models. While this prior-guided approach is widely

adopted, it largely reduces neural decoding to a tool for controlling generative models rather than probing the principles of human perception. In contrast, the human brain, arguably the most intricate biological system, excels at integrating information across multiple sensory modalities. Humans naturally perceive the world through rich audiovisual experiences, seamlessly combining visual and auditory cues to form coherent, dynamic representations of their surroundings. A classic example is the McGurk effect [29], where conflicting auditory and visual speech cues produce a third, illusory percept, highlighting the fundamental interplay between vision and hearing. Beyond generative priors, understanding how the brain achieves such integration is central to bridging human cognition and machine perception.

Technically, most existing studies [7, 8, 13, 16, 28] in neural decoding focus on reconstructing visual content from visual-only stimuli, typically relying on a single neural modality such as fMRI or EEG. This limited scope overlooks two critical aspects of perception. First, auditory signals strongly modulate visual processing [31, 41], shaping attention, emotion, and context during naturalistic viewing. Second, fMRI and EEG provide complementary insights: fMRI offers fine-grained spatial localization, while EEG captures millisecond-level temporal dynamics. Ignoring the cross-modal and cross-modality interactions restricts the fidelity and realism of reconstructed visual stimuli.

To address this gap, we propose both a new task and a novel large-scale dataset. Specifically, we present the task of reconstructing continuous video stimuli from multimodal brain signals recorded during naturalistic audiovisual stimulation. This task requires capturing both the spatial and temporal aspects of neural encoding, enabling exploration of how auditory input influences visual perception. However, progress in this area has been limited by the lack of datasets that synchronize fMRI and EEG recordings under ecologically valid, temporally dynamic audiovisual conditions. To overcome this challenge, we present CineBrain, the first large-scale multimodal dataset that synchronizes fMRI and EEG during naturalistic audiovisual viewing. Drawing inspiration from neurocinematics research [17, 46], which demonstrates that narrative-driven content naturally sustains attention and elicits complex brain dynamics, CineBrain includes approximately six hours of The Big Bang Theory episodes, viewed by six participants under 3T fMRI scanning with concurrent EEG recordings captured using custom non-magnetic equipment. As illustrated in Fig. 2, the participants exhibit highly consistent activation patterns across both modalities, providing strong cross-modal alignment that is critical for multimodal decoding tasks.

Building on CineBrain, we conduct the first systematic exploration of fMRI–EEG integration for video reconstruction and present CineSync, a novel framework for reconstructing dynamic visual stimuli from multimodal brain

signals. CineSync employs a Multi-Modal Fusion Encoder with a dual-transformer design to independently process fMRI and EEG sequences before merging them via a learned fusion projector. To achieve semantic alignment across modalities, brain representations are anchored to visual and textual embeddings using a joint contrastive objective. The resulting multimodal embeddings are decoded by a Neural Latent Decoder, a diffusion-based model that fuses brain-derived features with latent noise to generate semantically coherent and perceptually realistic videos.

Evaluation using semantic and perceptual metrics demonstrates that CineSync achieves state-of-the-art performance on temporally dynamic reconstruction within CineBrain. It effectively leverages the complementary strengths of fMRI and EEG to enhance visual fidelity. Notably, auditory-related cortical activations improve decoding accuracy, highlighting cross-modal facilitation in natural perception, and ablation studies show that increasing EEG representational capacity further boosts reconstruction quality, underscoring EEG’s critical role in capturing rapid neural dynamics for fine-grained video reconstruction.

In summary, our contributions are as follows:

- • A novel task: We introduce the challenge of reconstructing continuous video stimuli from multimodal brain signals, modeling spatial and temporal neural encoding to reveal how auditory input influences visual perception.
- • A new dataset: We present CineBrain, the first largescale multimodal dataset that synchronizes fMRI and EEG during naturalistic audiovisual viewing, providing critical cross-modal alignment for decoding tasks.
- • A new framework: We propose CineSync, a framework for reconstructing dynamic video stimuli using a dualtransformer-based Multi-Modal Fusion Encoder (MFE) and a diffusion-based Neural Latent Decoder.
- • State-of-the-art performance: CineSync achieves superior reconstruction quality, leveraging fMRI–EEG complementarity, and highlighting the role of auditory-related cortical activations in improving decoding accuracy.

#### 2. Related Work

###### 2.1. Neuro-Stimulus Dataset

Data is fundamental to deep learning. To facilitate neural decoding experiments, various datasets have been proposed. For visual stimuli, NSD dataset [1] is a widely used large-scale fMRI dataset collected from image-based stimuli, while Wen dataset [47] and fMRI-Video-WebVid [23] contain fMRI data acquired from video stimuli. Additionally, the fMRI-Shape [13] and fMRI-Objaverse [14] datasets provide fMRI recordings from 3D video stimuli. For EEG data, the SEED-DV [27] dataset includes EEG recordings from video stimuli. Tang’s dataset [44] focuses specifically on auditory stimulation. However, these ex-

Table 1. Overview of the CineBrain Dataset. We present detailed statistics of our proposed CineBrain dataset and compare it with other existing video-based brain datasets. CineBrain provides comprehensive multimodal brain recordings during audiovisual stimulation. Each participant watched a total of 6 hours of audiovisual stimuli, corresponding to approximately 27,000 frames of fMRI data.

###### Dataset Stimulus Type Participants Gender (M/F) Duration Videos EEG fMRI

Wen [47] Video 3 0/3 3.07 h 5520 ✘ ✓ fMRI-Video-FCVID [23] Video 3 1/1 1.11 h 400 ✘ ✓ fMRI-Video-WebVid [23] Video 5 2/2 2.67 h 1200 ✘ ✓ SEED-DV [27] Video 20 10/10 0.76 h 1400 ✓ ✘

CineBrain (Ours) Audio+Video 6 2/4 6 h 5400 ✓ ✓

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

sub01 sub02 sub03 sub04

Figure 2. Visualization of fMRI and EEG Responses in CineBrain. fMRI and EEG responses of subjects 1–4 to identical stimuli, illustrating individual differences in brain activation.

isting datasets are limited because each dataset only captures single-modality brain signals responding to singlemodality stimuli. To address this limitation, in this paper, we propose a multimodal audiovisual stimulus dataset named CineBrain and conduct experiments based on it.

###### 2.2. Neural Decoding

Neural decoding aims to reconstruct external stimuli perceived by the human brain from recorded neural signals. This task remains challenging due to the complex nature of both neural encoding and stimulus reconstruction. Functional magnetic resonance imaging (fMRI), with its high spatial resolution and non-invasive acquisition, has shown strong performance in visual reconstruction. Recent works have achieved notable results in image reconstruction [7, 37, 42] and extended these advances to video [8, 15, 23, 43, 45] and 3D reconstruction [13, 14]. Electroencephalography (EEG), offering millisecond-level temporal resolution and continuous recording, has also proven effective for neural decoding. EEG-based methods have reconstructed images [3, 39, 40], videos [26, 28], 3D structures [16]. Despite these advances, most existing works have focused on single-modality decoding. The complementary characteristics of fMRI and EEG, where fMRI provides fine-grained spatial information and EEG offers precise temporal dynamics, have not yet been fully explored. In this paper, we investigate the integration of fMRI and EEG signals to exploit their complementary advantages and enhance the reconstruction of perceived stimuli.

###### 2.3. Diffusion Models

Diffusion models [19, 33, 36] are a powerful generative framework known for their capability to produce highquality images and can also be extended to other data modalities. The fundamental principle involves defining a forward diffusion process, in which clean data is progressively corrupted by adding Gaussian noise. A neural network, typically a U-Net architecture, is then trained to reverse this diffusion process, iteratively removing the noise to reconstruct high-quality data. DiT [32] has successfully scaled up diffusion models through transformer-based architectures, significantly improving their generative capabilities. Consequently, DiT has been effectively applied across multiple modalities, including text-to-image generation [10, 21, 32], text-to-video generation [4, 18, 48], and text-to-audio generation [6, 9]. In our study, motivated by the strong distribution modeling capabilities of diffusion models, we leverage diffusion models specifically trained on video and audio data to reconstruct these modalities from multimodal brain signals, including fMRI and EEG.

#### 3. Experimental Designs and Curated Dataset

In this section, we introduce CineBrain, the first multimodal brain dataset featuring synchronized EEG and fMRI recordings collected during story-based audiovisual stimulation. Tab. 1 summarizes key statistics of CineBrain and contrasts it with existing datasets primarily employing visual stimuli. In comparison, CineBrain provides richer data suitable for a broader range of downstream tasks. The dataset will be publicly available to promote research in story decoding and multimodal neural decoding.

###### 3.1. Subjects and Experiments

Six participants (ages 21–26; two males, four females), who are unaware of the study’s objectives, contributed to the CineBrain dataset. All participants have normal or corrected-to-normal vision and hearing. Written informed consent was obtained, and the experimental protocol received approval from the appropriate ethics committee.

Since prolonged MRI scanning can cause discomfort and reduced attention, participants cannot stay in the scanner for

long periods. Therefore, we select episodes from the popular TV series The Big Bang Theory as audiovisual stimuli. The show’s engaging narratives, familiarity with daily life, and rich audiovisual content help maintain participants’ attention throughout the 18-minute viewing sessions. To ensure diverse yet comparable stimuli, participants collectively view 20 different episodes. All participants watch the first 10 episodes of Season 7; in addition, Participants 1, 2, and 6 view 10 episodes from Season 9, while the remaining participants watch 10 episodes from Season 11.

During the experiment, the video resolution is downsampled to 720p to match the MRI’s in-bore LCD screen. Episodes of varying lengths are standardized to 18 minutes of viewing time, easing subsequent data processing. Each 18-minute viewing constitutes one “run”, yielding 20 runs per participant. To maintain data quality and reduce fatigue, runs are grouped into sessions of two or three episodes each, with no more than five runs conducted on a single day. In total, each participant contributes approximately 6 hours of concurrent fMRI and EEG alongside ECG.

Visual stimuli are presented on an LCD screen (8◦ × 8◦) positioned at the head of the scanner bed. Participants viewed the screen through a mirror attached to the RF coil, maintaining fixation on a central red dot (0.4◦ × 0.4◦).

CineBrain experimental setup features simultaneous fMRI and EEG recording, harnessing their complementary strengths—high spatial resolution from fMRI and high temporal resolution from EEG. Given the noisy MRI environment, participants use sponge earplugs and customdesigned non-magnetic headphones with soft padding to ensure comfort and clear audio delivery. An MRI-compatible EEG cap is also worn, enabling simultaneous collection of these complementary neural data types, significantly broadening dataset’s potential for diverse downstream analyses.

###### 3.2. Data Acquisition and Preprocessing

fMRI. The fMRI acquisition protocol follows established experimental standards [13, 14, 23]. A 3T scanner equipped with a 32-channel RF head coil is employed to obtain high-resolution T1-weighted structural images and functional data for the decoding experiments. T1-weighted images were obtained using an MPRAGE sequence (0.8-mm isotropic resolution, TR = 2500 ms, TE = 2.22 ms, flip angle 8◦). Functional data were recorded using a gradient-echo EPI sequence with whole-brain coverage (2-mm isotropic resolution, TR = 800 ms, TE = 37 ms, flip angle 52◦, multiband factor = 8). With a sampling frequency of 1.25 Hz, each video run generated 1350 functional MRI frames.

Preprocessing is performed using the widely adopted fMRIPrep pipeline [11, 12]. Unlike previous studies that focused exclusively on visual decoding [7, 8, 13], our experimental design additionally incorporates auditory stimuli, prompting the selection of both visual and auditory re-

[Figure 29]

Audio

Audio

Vision

Figure 3. ROIs from the fMRI signals used in our experiments.

gions of interest (ROIs). The selected ROIs are illustrated in Fig. 3. Specifically, the visual ROIs comprise 8,405 voxels, while the auditory ROIs comprise 10,541 voxels. Detailed ROI definitions are provided in the supplementary material.

To account for the inherent delay in BOLD responses, fMRI signals underwent z-scoring across vertices within each run, incorporating a 4-second lag.

EEG. EEG data are captured using an MRI-compatible 64channel cap at 1000 Hz, simultaneously recording ECG signals. Precise synchronization between EEG and fMRI data is ensured by logging fMRI TR timings. EEG preprocessing involves a multi-step artifact removal approach, targeting scanner-induced noise and biological interference while preserving neural activity. This includes bandpass filtering (0.1–30 Hz) to remove baseline drift and muscle artifacts, and a 50 Hz notch filter to reduce powerline interference [5, 22, 24, 25]. ECG artifacts are initially mitigated using QRS-based techniques, followed by independent component analysis to isolate residual artifacts. ECG recordings further support adaptive refinement of artifact removal, yielding clean EEG data for subsequent analysis.

Video Stimuli. The 24 fps, 720p video stimuli consist of 20 episodes, each lasting 18 minutes. For video decoding, we downsample the videos to a resolution of 480 × 720 and segment them into 4-second clips (33 frames per clip). Each participant thus contributes 5,400 clips in total, including 4,860 for training (from the first 18 episodes) and 540 for testing (from the final two episodes).

Audio Stimuli. Audio from each video episode is also segmented into corresponding 4-second clips, resulting in the same number of samples as the video clips, maintaining consistency for training and evaluation.

Text Descriptions. To support multimodal decoding experiments, textual descriptions are generated separately for each audio and video clip. Video descriptions are generated using Qwen2.5-VL [2] to facilitate contrastive learning. Audio clips are transcribed using Whisper-large-v3 [35], providing rich textual inputs aligned with auditory data for enhanced multimodal analyses.

###### 3.3. Extended Downstream Tasks

Although CineBrain is primarily collected to support multimodal brain-signal-based video decoding, it also provides a solid foundation for a range of extended downstream tasks

fMRI signals

EEG signals

fMRI signals

EEG signals

fMRI signals

EEG signals

fMRI signals

EEG signals

fMRI signals

EEG signals

Patchify

Patchify

Patchify

Patchify

Patchify

Patchify

Patchify

Patchify

Patchify

Patchify

###### Cross-Attn

fMRI Transformer

EEG Transformer

fMRI Transformer

EEG Transformer

fMRI Transformer

EEG Transformer

Multimodal Joint Transformer

fMRI Transformer

EEG Transformer

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Multimodal Transformer

c! !!

#! #!

c!

#! ## !"

c! #! #" !"

c! #! #" !"

c!

#! #" !"

C

C

C

C

C

Neuro Feature Fusion !

Neuro Feature Fusion !

Neuro Feature Fusion !

Neuro Feature Fusion !

Neuro Feature Fusion !

!!

!!

!!

!!

!!

(a) Joint Transformer

(b) Two-Stage Fusion

(c) Cross-Attention Fusion

(d) Spatial Concatenation

(e) Dual Transformer Fusion

Figure 4. Architectural exploration for integrating fMRI and EEG. We compare five encoder variants, each adopting a distinct fusion mechanism to integrate the complementary information from fMRI and EEG signals.

Cross-Attention

Table 2. Performance comparison of different multimodal encoder structures. We evaluate all variants using video-level semantic metrics and frame-level perceptual metrics. Bold denotes the best performance, while underlined denotes the second-best.

that may inspire broader multimodal research: (1) Auditory decoding: leveraging both fMRI and EEG signals to reconstruct auditory stimuli. (2) EEG-to-fMRI translation: exploring cross-modal mappings from EEG to fMRI representations, which is particularly meaningful given the high cost and difficulty of fMRI acquisition. (3) Stimulus-tobrain modeling: further extending the framework to predict fMRI or EEG responses directly from video or audio stimuli. Together, these tasks demonstrate the extensibility of CineBrain beyond video reconstruction. We envision CineBrain serving as a comprehensive benchmark for advancing research on how the human brain perceives, integrates, and reconstructs dynamic audiovisual experiences.

[Figure 36]

[Figure 37]

Semantic-level Perceptual-level

METHODS

2-way↑ 50-way↑ FVD↓ SSIM↑ PSNR↑

- (a) Joint Transformer 0.924 0.274 128.0 0.232 9.30

- (b) Two-Stage Fusion 0.921 0.278 120.0 0.242 9.60

- (c) Cross-Attn. Fusion 0.921 0.292 110.0 0.249 9.90

- (d) SpatialCat (f113-e113) 0.918 0.276 114.0 0.255 9.95 (d) SpatialCat (f34-e192) 0.928 0.311 106.0 0.250 10.10

- (d) SpatialCat (f24-e202) 0.926 0.307 108.0 0.247 10.00

- (e) Dual Trans. Fusion 0.929 0.324 51.53 0.249 12.03

(c) Cross-Attention Fusion: employs modality-specific Transformers where each block includes cross-attention for inter-modal information exchange. (d) Spatial Concatenation: uses different token counts for fMRI and EEG features, doubles the feature dimension, and spatially concatenates them before fusion. For a fair comparison, the total token count is halved. (e) Dual Transformer Fusion: extracts features from each modality with independent Transformers and aggregates them at the final stage. Tab. 2 indicates that separately encoding fMRI and EEG signals yields better performance. This finding reveals a substantial discrepancy between the two modalities, suggesting that directly applying fully shared self-attention is not suitable for their fusion. It also provides important insights that inform the design of our subsequent fusion module.

#### 4. Methods

Overall. To evaluate the effectiveness of our proposed CineBrain framework and the integration of fMRI and EEG signals to enhance downstream decoding tasks, we introduce an innovative and versatile approach comprising two main components: 1) Multi-Modal Fusion Encoder (MFE): This module extracts semantically aligned features from fMRI and EEG signals and subsequently fuses these modalities to produce effective representations for downstream tasks. 2) Neuro Latent Decoder (NLD): Utilizing fused signals from the MFE, this module employs a LoRAtuned, diffusion-based decoder to reconstruct corresponding stimuli (e.g., video or audio) from brain signals.

###### 4.1. Multi-Modal Fusion Encoder

Motivated by this observation, we design a dualtransformer architecture termed the Multi-Modal Fusion Encoder (MFE). Inspired by the success of contrastive learning [34] in neuro-decoding [8, 14, 23] and multi-modal representation learning, the MFE adopts the Vision Transformer (ViT) as the backbone with a class token to semantically align information across modalities. Formally, we denote the fMRI and EEG signals as xf and xe, respectively. The MFE is defined as E = {Ef,Ee}, where Ef and Ee

fMRI–EEG Fusion Exploration. As an initial exploration of fMRI–EEG integration, we evaluate five fusion architectures under comparable parameter budgets, as illustrated in Fig. 4. (a) Joint Transformer: concatenates fMRI and EEG features and feeds them into a unified Transformer for joint representation learning. (b) Two-Stage Fusion: processes fMRI and EEG separately with individual Transformers, followed by a joint Transformer for integration.

× N blocks

###### Modality Alignment

###### Multi-Modal Fusion Encoder

###### Transformer Block

[Figure 38]

[Figure 39]

[Figure 40]

The video features two distinct scenes set in different indoor environments…

[Figure 41]

[Figure 42]

''% ''&

The video features two distinct scenes set in different indoor environments…

[Figure 43]

Scale Scale

caption

Dense Feed-Forward "" + Δ"#

Patchify

Patchify

[Figure 44]

Vision Encoder

Text Encoder

Text Encoder

α'%,&'% α'&,&'&

[Figure 45]

[Figure 46]

Scale, Shift Scale, Shift

…

fMRI Transformer

EEG Transformer

'$%

'$&

[Figure 48]

[Figure 49]

Scale Scale

[Figure 50]

c! #! #" !"

###### C

Self-Attention

Neuro Feature Fusion !

contrastive

[Figure 51]

ℒ!&+ ℒ!( ℒ"&+ ℒ"( ℒ!"

Linear

!!

Linear

Linear

!! + Δ!!

!" + Δ!"

!# + Δ!#

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

###### Training Inputs

Neuro Latent Decoder

|N|ois|e|
|---|---|---|
| | | |
| | | |

3DCausalVAE

α$%,&$% α$&,&$&

[Figure 59]

[Figure 60]

Scale, Shift Scale, Shift

[Figure 61]

Transformer Block

Brain MLP Vision MLP

$%

| | |T| | |
|---|---|---|---|---|
| | | | | |

| | | |
|---|---|---|
| | | |

3D Causal VAE

[Figure 62]

[Figure 63]

Video Clip

!! ((

- Figure 5. Overview of the CineSync Framework. CineSync first employs a Multimodal Fusion Encoder to extract features from fMRI and EEG data, with a modality alignment module to align these features with semantic information. Subsequently, it utilizes a LoRA-tuned neural latent decoder to reconstruct videos based on the fused brain features. Note: The gray box is used only during training.

To further align fMRI and EEG representations, we introduce an additional cross-modal contrastive loss:

are modality-specific encoders for fMRI and EEG. Through the MFE, we obtain both latent features and class tokens:

###### zf,ze,cf,ce = E(xf,xe), (1)

Lfe = Lclip(cf,ce). (5) Thus, the overall contrastive objective is given by:

where zf and ze denote the latent representations extracted from fMRI and EEG, and cf and ce are their corresponding class tokens used for contrastive alignment.

Lc = Lfv + Lft + Lev + Let + Lfe. (6)

To integrate information across modalities, we introduce a fusion MLP ψ that combines the latent features:

During training, we optimize the MFE, the fusion MLP ψ, and the aggregation module φ, while keeping the pretrained encoders Ev and Et frozen. After extracting the fused brain representation zb, it is fed into the subsequent module for video reconstruction.

###### zb = ψ(zf,ze), (2)

where zb serves as the unified brain representation for subsequent video reconstruction.

For cross-modal alignment, we employ pre-trained contrastive encoders Ev and Et to extract embeddings from videos and their textual descriptions generated by the VLM (see Fig. 5). Given a video clip with n frames V = {I1,...,In}, frame-wise embeddings are obtained via Ev and aggregated by a temporal aggregation module φ to form a video-level representation cv. For text, we directly encode the caption using Et:

###### 4.2. Neuro Latent Decoder

To effectively reconstruct the temporal stimuli, we design a Neuro Latent Decoder (NLD), a general-purpose module that can be seamlessly integrated into any video diffusion model. Specifically, for video reconstruction, we adopt and adapt CogVideoX-5B [48] as the underlying diffusion model within our NLD. CogVideoX-5B generates videos at 8 fps with a resolution of 480 × 720, conditioned on text prompts. Given the central role of textual embeddings in video diffusion models, we replace the original text condition with our fused brain representation zb.

cv = φ {Ev(Ii)}ni=1 , ct = Et(Text). (3) We define the contrastive objectives as:

Lfv = Lclip(cf,cv), Lft = Lclip(cf,ct), Lev = Lclip(ce,cv), Let = Lclip(ce,ct).

During training, a video clip V is first encoded into the latent space using a 3D-VAE E, after which noise is added

(4)

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

GT

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

CineSync

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

CineSync-fMRI

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

CineSync-EEG

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

GT

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

CineSync

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

CineSync-fMRI

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

CineSync-EEG

- Figure 6. Qualitative comparison of our method with baselines. We compare the results of CineSync, CineSync-fMRI, and CineSyncEEG with the ground truth (GT). CineSync demonstrates higher accuracy, greater temporal consistency, and improved video quality.

according to the forward diffusion process: x0 = E(V ), xt = √αt x0 + √1 − αt ϵ, ϵ ∼ N(0,I),

(7) where t is uniformly sampled from {1,...,1000}. Following the training strategy of CogVideoX, we adopt explicit uniform timestep sampling. As illustrated in Fig. 5, the noised latent xt is concatenated with the fused brain feature zb and then fed into the diffusion model.

To efficiently adapt the model to brain-derived inputs, we employ LoRA fine-tuning on both the attention and feedforward layers within the DiT blocks of NLD, enabling effective integration of neural representations into the video generation process. The training objective of the NLD follows the standard diffusion loss:

L = EV,ϵ, t ∥ϵ − ϵθ(xt, zb, t)∥2 , ϵ ∼ N(0,I). (8)

#### 5. Experiments

Metrics. To comprehensively evaluate the model’s performance, we assess the reconstructed videos at two levels: semantic and perceptual. 1) Semantic-Level: Semantic similarity is essential for evaluating reconstruction quality. Following prior neuro-decoding studies [8, 13, 20, 23, 37], we

compute N-way top-K accuracy across entire videos and individual frames to measure feature similarity. Additionally, the Fréchet Video Distance (FVD) is calculated to quantify the distributional similarity between reconstructed and ground-truth videos. 2) Perceptual-Level: Beyond semantics, visual quality is also critical. We employ DINO temporal consistency (DTC) [30] and CLIP temporal consistency (CTC) [34] to assess video temporal coherence. Structural similarity (SSIM) and peak signal-to-noise ratio (PSNR) are further used to evaluate frame-level visual fidelity.

###### 5.1. Implementation Details

Both the fMRI and EEG transformers contain 12 layers with a hidden dimension of 2048 and a token length of 227 (226 spatial tokens plus one class token). The LoRA configuration in the NLD adopts a rank of 64 and a scaling factor α = 64. For modality alignment, we employ SigLIP [49] to extract visual and textual embeddings. Each input sample comprises 4-second multimodal brain signals, corresponding to 5 × 8,405 fMRI voxels and 64 × 4,000 EEG data points. We use the AdamW optimizer with β = (0.9,0.95) and an initial learning rate of 1 × 10−4. Further implementation details are provided in the supplementary material.

- Table 3. Performance comparison of CineSync with baselines. The average metrics across all subjects are reported. CineSync⋆ indicates the experiment that includes audio-related ROIs in fMRI. Bold denotes the best performance, while underlined denotes the second-best.

Semantic-level Perceptual-level

METHODS

Video-based Frame-based Video-based Frame-based

2-way↑ 50-way↑ FVD↓ 2-way↑ 50-way↑ DTC↑ CTC↑ SSIM↑ PSNR↑

EEG2Video [28] 0.786 0.162 146.23 0.815 0.134 0.683 0.707 0.109 6.218 CineSync-EEG 0.891 0.304 53.75 0.918 0.349 0.899 0.937 0.231 11.75

GLFA [23] 0.801 0.167 128.76 0.847 0.225 0.706 0.735 0.123 7.526 NeuroClips [15] 0.816 0.183 116.36 0.833 0.142 0.871 0.868 0.087 6.854 CineSync-fMRI 0.893 0.307 57.47 0.926 0.358 0.907 0.945 0.240 11.92

CineSync 0.909 0.319 52.78 0.937 0.398 0.915 0.967 0.262 11.99 CineSync⋆ 0.926 0.336 44.77 0.954 0.423 0.921 0.953 0.297 12.18

###### 5.2. Experimental Results

Encoder Comparison. As shown in Tab. 2, we observe that reducing the interaction between fMRI and EEG during the feature extraction stage leads to better overall performance. This finding suggests that the two modalities exhibit substantial representational differences, and excessive early fusion may hinder effective feature learning. Interestingly, experiment (d) shows that when the total token count is fixed at 226, moderately increasing the number of EEG tokens leads to performance gains in video reconstruction. However, we also observe that although this design expands the feature dimension, the reduced overall token count yields suboptimal performance compared with method (e), indicating that the number of tokens plays a more critical role.

Comparison with Baselines. To evaluate the effectiveness of our proposed CineSync, we compare it with several representative baselines, including EEG2Video [28], GLFA [23], and NeuroClips [15], as well as two simplified variants of our model: CineSync-fMRI and CineSyncEEG. All models are trained and evaluated on the same dataset for fair comparison. Tab. 3 reports the average performance across all subjects. Overall, the two simplified versions of CineSync substantially outperform the baselines across all metrics, validating the robustness of our framework even when using a single modality. Furthermore, the full CineSync model surpasses both CineSync-fMRI and CineSync-EEG, demonstrating that jointly modeling fMRI and EEG signals enables more comprehensive decoding of brain activity and leads to higher-quality video reconstruction. These results establish CineSync as a new state-ofthe-art framework for reconstructing dynamic visual stimuli from human brain signals. In addition, we conduct an extended experiment by incorporating auditory ROIs into the fMRI inputs for video reconstruction, while retaining all EEG data, resulting in a total of 18,946 voxels per frame. We refer to this variant as CineSync⋆. The results show that incorporating auditory ROIs improves video reconstruction performance, underscoring the benefits of multimodal and

cross-regional integration in brain decoding.

Qualitative Analysis. Qualitative results in Fig. 6 confirm that CineSync captures finer temporal dynamics and richer perceptual details than the baselines. This suggests that our framework effectively leverages the complementary advantages of fMRI and EEG to enhance reconstruction fidelity. Moreover, the consistently high reconstruction quality across subjects highlights the reliability and value of the CineBrain dataset in supporting multimodal brain-signal research, potentially facilitating further progress at the intersection of computer vision and cognitive neuroscience.

Ablation Studies. We conduct an ablation study to evaluate the effectiveness of the contrastive learning–based alignment employed in our framework. The results, presented in the supplementary material, demonstrate that our multi-level contrastive alignment strategy significantly contributes to the overall performance, confirming its effectiveness in aligning multimodal brain representations.

#### 6. Conclusion

In this paper, we present CineBrain, the first large-scale multimodal dataset simultaneously recording fMRI and EEG under naturalistic audiovisual stimulation. Building on this dataset, we introduce the task of reconstructing continuous video stimuli from multimodal brain signals. Furthermore, we explore effective strategies for integrating fMRI and EEG and propose CineSync, an innovative framework that integrates fMRI and EEG to enhance video reconstruction quality. CineSync achieves state-of-the-art performance, consistently outperforming baselines using singlemodal input, thereby demonstrating the benefits of multimodal brain representations. Experiments further show that incorporating auditory ROIs improves visual decoding, reinforcing the reliability and value of CineBrain dataset and highlighting its potential to advance multimodal brain decoding research. We hope our dataset and framework will serve as strong baselines and inspire future progress at the intersection of computer vision and cognitive neuroscience.

#### References

- [1] Emily J Allen, Ghislain St-Yves, Yihan Wu, Jesse L Breedlove, Jacob S Prince, Logan T Dowdle, Matthias Nau, Brad Caron, Franco Pestilli, Ian Charest, et al. A massive 7t fmri dataset to bridge cognitive neuroscience and artificial intelligence. Nature neuroscience, 25(1):116–126, 2022. 2
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 4, 1
- [3] Yunpeng Bai, Xintao Wang, Yan pei Cao, Yixiao Ge, Chun Yuan, and Ying Shan. Dreamdiffusion: Generating highquality images from brain eeg signals, 2023. 3
- [4] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models, 2024. 3
- [5] Jin Chen, Tony Ro, and Zhigang Zhu. Emotion recognition with audio, video, eeg, and emg: A dataset and baseline approaches. IEEE Access, 10:13229–13242, 2022. 4
- [6] Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, Jian Zhao, Kai Yu, and Xie Chen. F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching. arXiv preprint arXiv:2410.06885, 2024. 3
- [7] Zijiao Chen, Jiaxin Qing, Tiange Xiang, Wan Lin Yue, and Juan Helen Zhou. Seeing beyond the brain: Conditional diffusion model with sparse masked modeling for vision decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22710–22720,

2023. 1, 2, 3, 4

- [8] Zijiao Chen, Jiaxin Qing, and Juan Helen Zhou. Cinematic mindscapes: High-quality video reconstruction from brain activity. NeurIPS, 2023. 1, 2, 3, 4, 5, 7
- [9] Sefik Emre Eskimez, Xiaofei Wang, Manthan Thakker, Canrun Li, Chung-Hsien Tsai, Zhen Xiao, Hemin Yang, Zirun Zhu, Min Tang, Xu Tan, Yanqing Liu, Sheng Zhao, and Naoyuki Kanda. E2 tts: Embarrassingly easy fully nonautoregressive zero-shot tts. 2024. 3
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. 3
- [11] Oscar Esteban, Ross Blair, Christopher J. Markiewicz, Shoshana L. Berleant, Craig Moodie, Feilong Ma, Ayse Ilkay Isik, Asier Erramuzpe, Mathias Kent, James

- D. andGoncalves, Elizabeth DuPre, Kevin R. Sitek, Daniel
- E. P. Gomez, Daniel J. Lurie, Zhifang Ye, Russell A. Poldrack, and Krzysztof J. Gorgolewski. fmriprep. Software,

2018. 4

- [12] Oscar Esteban, Christopher Markiewicz, Ross W Blair, Craig Moodie, Ayse Ilkay Isik, Asier Erramuzpe Aliaga,

- James Kent, Mathias Goncalves, Elizabeth DuPre, Madeleine Snyder, Hiroyuki Oya, Satrajit Ghosh, Jessey Wright, Joke Durnez, Russell Poldrack, and Krzysztof Jacek Gorgolewski. fMRIPrep: a robust preprocessing pipeline for functional MRI. Nature Methods, 16:111–116, 2019. 4
- [13] Jianxiong Gao, Yuqian Fu, Yun Wang, Xuelin Qian, Jianfeng Feng, and Yanwei Fu. Mind-3d: Reconstruct high-quality 3d objects in human brain. arXiv preprint arXiv:2312.07485,

2023. 1, 2, 3, 4, 7

- [14] Jianxiong Gao, Yuqian Fu, Yun Wang, Xuelin Qian, Jianfeng Feng, and Yanwei Fu. fmri-3d: A comprehensive dataset for enhancing fmri-based 3d reconstruction. arXiv preprint arXiv:2409.11315, 2024. 2, 3, 4, 5
- [15] Zixuan Gong, Guangyin Bao, Qi Zhang, Zhongwei Wan, Duoqian Miao, Shoujin Wang, Lei Zhu, Changwei Wang, Rongtao Xu, Liang Hu, et al. Neuroclips: Towards highfidelity and smooth fmri-to-video reconstruction. arXiv preprint arXiv:2410.19452, 2024. 3, 8
- [16] Zhanqiang Guo, Jiamin Wu, Yonghao Song, Jiahui Bu, Weijian Mai, Qihao Zheng, Wanli Ouyang, and Chunfeng Song. Neuro-3d: Towards 3d visual decoding from eeg signals,

2024. 2, 3

- [17] Uri Hasson, Ohad Landesman, Barbara Knappmeyer, Ignacio Vallines, Nava Rubin, and David J Heeger. Neurocinematics: The neuroscience of film. Projections, 2(1):1–26,

2008. 2

- [18] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. 2022. 3
- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 3
- [20] Gautam Krishna, Co Tran, Yan Han, Mason Carnahan, and Ahmed H Tewfik. Speech synthesis using eeg. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1235–1238. IEEE, 2020. 7
- [21] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 3
- [22] Min-Ho Lee, Adai Shomanov, Balgyn Begim, Zhuldyz Kabidenova, Aruna Nyssanbay, Adnan Yazici, and SeongWhan Lee. Eav: Eeg-audio-video dataset for emotion recognition in conversational contexts. Scientific data, 11(1):1026,

2024. 4

- [23] Chong Li, Xuelin Qian, Yun Wang, Jingyang Huo, Xiangyang Xue, Yanwei Fu, and Jianfeng Feng. Enhancing cross-subject fmri-to-video decoding with global-local functional alignment. 1, 2, 3, 4, 5, 7, 8
- [24] Dongyang Li, Chen Wei, Shiying Li, Jiachen Zou, Haoyang Qin, and Quanying Liu. Visual decoding and reconstruction via eeg embeddings with guided diffusion. arXiv preprint arXiv:2403.07721, 2024. 4
- [25] Yamin Li, Ange Lou, Ziyuan Xu, Shengchao Zhang, Shiyu Wang, Dario Englot, Soheil Kolouri, Daniel Moyer, Roza Bayrak, and Catie Chang. Neurobolt: Resting-state eegto-fmri synthesis with multi-dimensional feature mapping. Advances in Neural Information Processing Systems, 37: 23378–23405, 2025. 4

- [26] Junxiang Liu, Junming Lin, Jiangtong Li, and Jie Li. Dynamind: Reconstructing dynamic visual scenes from eeg by aligning temporal dynamics and multimodal semantics to guided diffusion, 2025. 3
- [27] Xuan-Hao Liu, Yan-Kai Liu, Yansen Wang, Kan Ren, Hanwen Shi, Zilong Wang, Dongsheng Li, Bao-Liang Lu, and Wei-Long Zheng. EEG2video: Towards decoding dynamic visual perception from EEG signals. In The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS), 2024. 2, 3
- [28] Xuan-Hao Liu, Yan-Kai Liu, Yansen Wang, Kan Ren, Hanwen Shi, Zilong Wang, Dongsheng Li, Bao-Liang Lu, and Wei-Long Zheng. EEG2video: Towards decoding dynamic visual perception from EEG signals. In The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS), 2024. 2, 3, 8
- [29] HARRY MCGURK and JOHN MACDONALD. Hearing lips and seeing voices. Nature, 1976. 2
- [30] Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, ShangWen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023. 7
- [31] Hae-Jeong Park and Karl Friston. Structural and functional brain networks: from connections to cognition. Science, 342

(6158):1238411, 2013. 2

- [32] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022. 3
- [33] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024. 3
- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 5, 7
- [35] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision, 2022. 4
- [36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 3
- [37] Paul S. Scotti, Atmadeep Banerjee, Jimmie Goode, Stepan Shabalin, Alex Nguyen, Ethan Cohen, Aidan J. Dempster, Nathalie Verlinde, Elad Yundler, David Weisberg, Kenneth A. Norman, and Tanishq Mathew Abraham. Reconstructing the mind’s eye: fmri-to-image with contrastive learning and diffusion priors, 2023. 1, 3, 7
- [38] Paul S. Scotti, Mihir Tripathy, Cesar Kadir Torrico Villanueva, Reese Kneeland, Tong Chen, Ashutosh Narang,

- Charan Santhirasegaran, Jonathan Xu, Thomas Naselaris, Kenneth A. Norman, and Tanishq Mathew Abraham. Mindeye2: Shared-subject models enable fmri-to-image with 1 hour of data, 2024. 1
- [39] Prajwal Singh, Pankaj Pandey, Krishna Miyapuram, and Shanmuganathan Raman. Eeg2image: Image reconstruction from eeg brain signals, 2023. 3
- [40] Yonghao Song, Bingchuan Liu, Xiang Li, Nanlin Shi, Yijun Wang, and Xiaorong Gao. Decoding Natural Images from EEG for Object Recognition. In International Conference on Learning Representations, 2024. 3
- [41] Olaf Sporns. Brain connectivity. Scholarpedia, 2(10):4695,

2007. 2

- [42] Jingyuan Sun, Mingxiao Li, Zijiao Chen, Yunhao Zhang, Shaonan Wang, and Marie-Francine Moens. Contrast, attend and diffuse to decode high-resolution images from brain activities, 2023. 3
- [43] Jingyuan Sun, Mingxiao Li, and Marie-Francine Moens. Neuralflix: A simple while effective framework for semantic decoding of videos from non-invasive brain recordings. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7096–7104, 2025. 3
- [44] Jerry Tang, Amanda LeBel, Shailee Jain, and Alexander G Huth. Semantic reconstruction of continuous language from non-invasive brain recordings. Nature Neuroscience, 26(5): 858–866, 2023. 2
- [45] Haonan Wang, Qixiang Zhang, Lehan Wang, Xuanqi Huang, and Xiaomeng Li. Neurons: Emulating the human visual cortex improves fidelity and interpretability in fmri-to-video reconstruction, 2025. 3
- [46] Mitsuko Watabe-Uchida, Neir Eshel, and Naoshige Uchida. Neural circuitry of reward prediction error. Annual review of neuroscience, 40(1):373–394, 2017. 2
- [47] Haiguang Wen, Junxing Shi, Yizhen Zhang, Kun-Han Lu, Jiayue Cao, and Zhongming Liu. Neural encoding and decoding with deep learning for dynamic natural vision. Cerebral cortex, 28(12):4136–4160, 2018. 2, 3
- [48] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 3, 6
- [49] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training,

2023. 7

[Figure 128]

### CineBrain: A Large-Scale Multi-Modal Brain Dataset During Naturalistic

### Audiovisual Narrative Processing Supplementary Material

#### 7. Detailed Information on CineBrain

Users

Please watch the video and provide a detailed description of it, focusing on the main subjects, their actions, and the background scenes.

- 7.1. Detailed ROIs Selected in the Experiments

The regions of interest (ROIs) used in our study are summarized below. For the visual cortex, we include the following ROIs: V1, V2, V3, V3A, V3B, V3CD, V4, LO1, LO2, LO3, PIT, V4t, V6, V6A, V7, V8, PH, FFC, IP0, MT, MST, FST, VVC, VMV1, VMV2, VMV3, PHA1, PHA2, PHA3. For the auditory-related cortex, we select ROIs corresponding to: 4, 7AL, 7Am, 7m, 7PC, 7PL, 7Pm, 8Ad, 8Av, 8BM, 8C, 9a, 9p, 10d, 10v, 44, 45, 46, 47m, A1, IPS1, p32, PGp, s32, STGa, STSda, STSdp, STSva, STSvp, TPOJ1, TPOJ2, TPOJ3. In total, the selected visual ROIs include 8,405 voxels, while the auditory ROIs include 10,541 voxels.

- 7.2. Prompts for Generating Text Descriptions

Textual descriptions play an important role in our training pipeline. The prompts used to generate these descriptions with Qwen2.5-VL [2] are provided in Fig. 7.

- 7.3. EEG Experimental Device

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Qwen2.5-VL

The video depicts a lively office party scene with two men as the main focus. The setting is an office break room or similar communal area, characterized by a casual yet festive atmosphere. The background features a variety of people mingling, some holding drinks and others engaged in conversations, contributing to the social vibe. In the foreground, one man is dressed in a mustard-yellow corduroy blazer over a patterned sweater, paired with a white shirt. He appears to be in mid-conversation, his body language suggesting he is actively participating in the dialogue. His facial expression indicates he might be reacting to something the other man has said or done. The second man, wearing a gray suit jacket over a light blue shirt, is holding a glass of orange juice and seems to be in the middle of speaking. His posture and hand gestures suggest he is explaining something or making a point. The interaction between these two men is central to the scene, capturing a moment of friendly banter or discussion. The background reveals more details about the setting: a vending machine labeled \"COLD\" and \"drinks\" is visible, indicating the availability of refreshments for the attendees. A counter with various food items, including fruits and snacks, is also present, further emphasizing the party-like environment. The walls are adorned with colorful posters and artwork, adding a vibrant touch to the space. The overall ambiance suggests a relaxed and enjoyable gathering, typical of a workplace celebration or informal event.

- Figure 7. Prompt utilized for generating video descriptions.

E1

E2 E3

E4

E5

E6

E7

E8

- E9
- E10
- E11
- E12
- E13
- E14 E15

E16 E20

- E17
- E18
- E19

E21

E22

E23

E24E25

- E26
- E27

E28 E29

E30

E31

E32

E33

E34

E35

- E36
- E37

E38

E39

- E40
- E41

E42

- E43
- E44
- E45
- E46

E47

E48

E49

E50

E51

E52

E53

E54

E55

E56

E57

E58

E59

E60 E61

E63 E62

E64

- Figure 8. Electrode montage of a 64-channel EEG cap using the GSN-HydroCel-64_1.0 layout. Sensor positions are annotated with their corresponding channel labels.

We illustrate the electrode montage of a 64-channel EEG cap configured according to the GSN-HydroCel-64_1.0 layout in Fig. 8.

#### 8. Additional Experimental Results 8.1. Implementation Details

Due to GPU memory constraints, Our proposed CineSync is trained using a two-stage approach. In the first stage, only the multimodal fusion encoder is trained using contrastive loss with a batch size of 16 on a single H100 GPU. To enhance the effectiveness of contrastive learning, we augment the training dataset with diverse textual captions generated by Qwen2.5-VL-7B [2]. This pretraining phase lasts approximately 50 epochs. In the second stage, the pretrained encoder is integrated into the full model, which is finetuned for 5000 steps on 4 H100 GPUs using a batch size of 2 per GPU. The fMRI and EEG transformers each consist of 12 layers with a transformer dimension of 2048 and a token length of 227 (226 spatial tokens plus one class token). The LoRA configuration uses a rank of 64 and a scaling factor α=64. Input data consist of 4-second multimodal brain signals, yielding standard inputs of 5×8405 fMRI voxels and 64×4000 EEG data points.

###### 8.2. Ablation Study on Multimodal Alignment

To quantify the contribution of each alignment component in CineSync, we conduct ablations on the full model trained with both audio- and vision-related ROIs in fMRI. Specifically, w/o Vision, w/o Text, and w/o Across correspond to removing (i) the vision–fMRI alignment branch, (ii) the text–fMRI alignment branch, and (iii) the cross-

- Table 4. Ablation study on multimodal alignment in CineSync. We report the average performance across all subjects. CineSync⋆ indicates the experiment that includes audio-related ROIs in fMRI.

Semantic-level Perceptual-level

METHODS

Video-based Frame-based Video-based Frame-based

2-way↑ 50-way↑ FVD↓ 2-way↑ 50-way↑ DTC↑ CTC↑ SSIM↑ PSNR↑

w/o Vision 0.863 0.275 52.06 0.858 0.378 0.895 0.738 0.272 11.69 w/o Text 0.891 0.294 50.15 0.887 0.394 0.908 0.949 0.285 11.95 w/o Across 0.873 0.279 51.29 0.864 0.382 0.902 0.943 0.274 11.74

###### CineSync⋆ 0.926 0.336 44.77 0.954 0.423 0.921 0.953 0.297 12.18

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

GTSub01Sub02Sub06

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

- Figure 9. Video Reconstruction Results for Subjects 1, 2, and 6. We compare the reconstructed frames from Subjects 1, 2, and 6 with the corresponding ground-truth (GT) frames. The consistent semantic alignment and visual fidelity across subjects demonstrate the robustness and strong cross-subject generalization ability of CineSync.

###### 8.4. Video Reconstruction Across Subjects

modal alignment between EEG and fMRI, respectively. The averaged results across all subjects are reported in Tab. 4. Removing any of these alignment pathways leads to clear and consistent performance drops across both semanticand perceptual-level metrics, demonstrating that rich crossmodal alignment is critical for achieving high-quality brainto-video reconstruction.

To better illustrate the robustness of our model, we visualize the reconstructed videos from different subjects under the same visual stimuli. Representative results are shown in Fig. 9 and Fig. 10. Since Subjects 1, 2, and 6 share the same train–test split, and Subjects 3, 4, and 5 share another split, we compare the reconstruction quality within each group accordingly. Specifically, Fig. 9 presents the reconstructions of Subjects 1, 2, and 6, while Fig. 10 shows those from Subjects 3, 4, and 5. These results indicate that our model achieves stable performance across different individuals.

###### 8.3. Detailed Results for Each Subject

To further verify that our model remains robust across individuals, we report per-subject quantitative performance in Tab. 5. As shown, the results exhibit only minor variations across the six subjects, indicating that CineSync generalizes well to different brain patterns. Moreover, the averaged scores closely match the overall CineSync⋆ results, further confirming subject-invariant performance.

###### 8.5. More Results of Video Reconstruction

To further showcase the quality, semantic fidelity, and temporal coherence of the reconstructed videos, we present additional 12-frame examples in Fig. 11, Fig. 12, and Fig. 13. Across diverse scenes, our method consistently produces reconstructions that are semantically aligned with the stimuli

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

GTSub03Sub04Sub05

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

- Figure 10. Video Reconstruction Results for Subjects 3, 4, and 5. Reconstructed frames from Subjects 3, 4, and 5 are shown alongside the GT frames at matched timestamps. The results further verify that CineSync maintains stable reconstruction quality across different individuals.

Table 5. Performance of each subject in CineBrain. CineSync⋆ indicates the experiment that includes audio-related ROIs in fMRI.

Semantic-level Perceptual-level

METHODS

Video-based Frame-based Video-based Frame-based

2-way↑ 50-way↑ FVD↓ 2-way↑ 50-way↑ DTC↑ CTC↑ SSIM↑ PSNR↑

- Subject 1 0.921 0.332 46.12 0.951 0.419 0.918 0.949 0.294 12.11

- Subject 2 0.928 0.337 44.03 0.956 0.426 0.922 0.954 0.298 12.20

- Subject 3 0.923 0.334 45.87 0.952 0.421 0.917 0.950 0.295 12.14

- Subject 4 0.927 0.338 43.95 0.957 0.427 0.923 0.956 0.299 12.23

- Subject 5 0.929 0.339 44.62 0.955 0.425 0.920 0.952 0.296 12.17

- Subject 6 0.924 0.335 44.97 0.953 0.420 0.919 0.951 0.297 12.22 CineSync⋆ 0.926 0.336 44.77 0.954 0.423 0.921 0.953 0.297 12.18

and temporally smooth.

#### 9. Limitations and Future Work

Although our CineSync successfully leverages the spatial resolution of EEG to compensate for the temporal resolution limitations of fMRI, substantially improving video and audio reconstruction performance, our dataset currently does not explicitly provide additional fMRI data. This limitation restricts broader applications of our dataset and represents an area for future exploration.

Furthermore, our CineBrain dataset supports synchronized audiovisual stimuli. However, we have independently evaluated our primary contributions solely through separate video and audio reconstruction tasks. If we aim to expand into more complex applications such as embodied intelligence, it will be necessary to reconstruct multiple modalities simultaneously. Therefore, joint audiovisual recon-

struction represents another significant direction for our future research.

T=0 T=3 T=6 T=9 T=12 T=15

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

GT

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

Ours

T=18 T=21 T=24 T=27

T=30 T=32

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

GT

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Ours

Figure 11. More Results of CineSync: We present 12 frames with timestamps compared with the ground truth (GT).

T=0 T=3 T=6 T=9 T=12 T=15

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

GT

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Ours

T=18 T=21 T=24 T=27

T=30 T=32

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

GT

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Ours

Figure 12. More Results of CineSync: We present 12 frames with timestamps compared with the ground truth (GT).

T=0 T=3 T=6 T=9 T=12 T=15

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

GT

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

Ours

T=18 T=21 T=24 T=27

T=30 T=32

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

GT

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

Ours

Figure 13. More Results of CineSync: We present 12 frames with timestamps compared with the ground truth (GT).

