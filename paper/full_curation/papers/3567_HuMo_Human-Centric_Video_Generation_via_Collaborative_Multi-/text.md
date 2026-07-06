# arXiv:2509.08519v1[cs.CV]10Sep2025

[Figure 1]

[Figure 2]

## HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning

### Liyang Chen1,⋆, Tianxiang Ma2,⋆, Jiawei Liu2, Bingchuan Li2,†, Zhuowei Chen2, Lijie Liu2, Xu He1, Gen Li2, Qian He2, Zhiyong Wu1,§

1Tsinghua University, 2Intelligent Creation Lab, ByteDance ⋆Equal contribution, †Project lead, §Corresponding author

### Abstract

Human-Centric Video Generation (HCVG) methods seek to synthesize human videos from multimodal inputs, including text, image, and audio. Existing methods struggle to effectively coordinate these heterogeneous modalities due to two challenges: the scarcity of training data with paired triplet conditions and the difficulty of collaborating the sub-tasks of subject preservation and audio-visual sync with multimodal inputs. In this work, we present HuMo, a unified HCVG framework for collaborative multimodal control. For the first challenge, we construct a high-quality dataset with diverse and paired text, reference images, and audio. For the second challenge, we propose a two-stage progressive multimodal training paradigm with task-specific strategies. For the subject preservation task, to maintain the prompt following and visual generation abilities of the foundation model, we adopt the minimal-invasive image injection strategy. For the audio-visual sync task, besides the commonly adopted audio cross-attention layer, we propose a focus-by-predicting strategy that implicitly guides the model to associate audio with facial regions. For joint learning of controllabilities across multimodal inputs, building on previously acquired capabilities, we progressively incorporate the audio-visual sync task. During inference, for flexible and fine-grained multimodal control, we design a time-adaptive Classifier-Free Guidance strategy that dynamically adjusts guidance weights across denoising steps. Extensive experimental results demonstrate that HuMo surpasses specialized state-of-the-art methods in sub-tasks, establishing a unified framework for collaborative multimodal-conditioned HCVG.

Date: September 11, 2025 Project Page (Demo, Codes, Models, Data): https://phantom-video.github.io/HuMo

### 1 Introduction

Recent advances in foundational video generation models have significantly accelerated progress in HumanCentric Video Generation (HCVG), improving both the fidelity and controllability. Such progress is democratizing short video production. Traditional filmmaking, entailing scene setup, casting, styling, and scripting, is labor-intensive and demands collective expertise, incurring substantial time and financial costs. HCVG methods offers a disruptive alternative by leveraging multi-modal inputs: text for describing scenes and actions, images for defining human identity and subject appearance, and audio for character speaking. It greatly reduces manual overhead and enables scalable and efficient content creation.

| | |
|---|---|
|[Figure 3]||[Figure 25]<br><br>[Figure 26]|[Figure 27]<br><br>[Figure 28]|
|---|---|
|[Figure 29]<br><br>[Figure 30]| |
<br><br>|[Figure 20]<br><br>[Figure 21]|[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]|
|---|---|
| | |
<br><br>|[Figure 15]<br><br>[Figure 16]|
|---|
|

[Figure 17]

[Figure 18]

[Figure 19]

|[Figure 20]<br><br>[Figure 21]|[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]|
|---|---|
| | |

|[Figure 25]<br><br>[Figure 26]|[Figure 27]<br><br>[Figure 28]|
|---|---|
|[Figure 29]<br><br>[Figure 30]| |

[Figure 31]

[Figure 32]

[Figure 33]

|[Figure 34]<br><br>[Figure 35]|
|---|

|[Figure 36]<br><br>[Figure 37]|
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

|[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]|[Figure 45]|
|---|---|
| | |

|[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]|
|---|

|[Figure 49]<br><br>[Figure 50]|
|---|

|[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]| |
|---|---|
| | |

- Figure 1 We propose HuMo, a multimodal HCVG framework that supports flexible input compositions of text-image (first row), text-audio (second row), and text-image-audio (third row). HuMo generalizes to humans, humans with objects or animals, stylized humanoid artworks, and animations.

Prior HCVG methods [5, 19, 32] typically adopt a two-stage pipeline: a text-to-image (T2I) model [8] generates a subject-complete start frame containing all required elements (e.g., human subject, clothing, accessories or props, and scene background), followed by an image-to-video (I2V)-based model for audio-driven animation. However, this pipeline heavily depends on the subject-complete start frame, which inherently constrains the flexibility of text controls for refining or modifying the depicted subjects, as illustrated in Fig. 2(a)(b). Alternatively, subject-consistent video generation (S2V) methods [7, 22] leverage subject reference images and support flexible video customization through textual prompts. However, these methods are unable to incorporate an audio modality and cannot control what a character is speaking, as shown in Fig. 2(c). This significantly limits their application in scenarios requiring audio-visual collaboration. Recently, some methods [11, 34] attempt to integrate the audio-visual sync and subject preservation tasks mentioned above to achieve multimodal control. However, as shown in Fig. 2(d), they struggle to achieve effective collaboration among the triplet input modalities of text, reference images, and audio. For instance, emphasizing the influence of reference images usually degrades audio-visual sync. Conversely, prioritizing high sync tends to compromise either the text following or the subject preservation ability.

This work focuses on the HCVG task conditioned on text, reference images, and audio, aiming to achieve collaborative multimodal control. There exist two major challenges: 1) Data scarcity. Training such a model requires paired data with precisely aligned triplet conditions. However, there is a lack of publicly available datasets or standardized data preprocessing pipelines to effectively integrate these heterogeneous modalities. 2) Difficulty in collaborative multimodal control. Within a multi-task training framework, it is difficult to simultaneously achieve strong text following, subject consistency with reference images, and accurate audio-visual sync, as these abilities tend to compromise or undermine one another during learning. To tackle these challenges, we propose HuMo. Our key insight lies in the joint design of the data processing pipeline and training paradigm that enables the collaborative learning of multimodal controllability using the constructed multimodal data.

[Figure 54]

[Figure 55]

[Figure 56]

- (b)
- (c)
- (d)
- (e)

OmniHuman-1PhantomHunyuanCustomHuMo

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Start Frame

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Image Generation Model

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Reference Images

A man with wearing a white shirt and dark suit, joyfully plays with his black Labrador Retriever in front of a cozy house. Smiling brightly, he tosses a toy as the sleek, droopyeared dog responds eagerly on the well-kept lawn. Speaking Word [He, Me, Friend]

(a)

- Figure 2 Prior HCVG methods comparison. Reference images as inputs for Phantom [22], HunyuanCustom [11], and HuMo. OmniHuman-1 [19] takes the start frame as input, which is synthesized by an image generation model [8, 29]. OmniHuman-1 suffers from weak text adherence, unable to generate subjects (e.g., a toy) absent in the start frame, while the subject preservation is precariously dependent on the preceding image generator. Phantom lacks audio-driven articulation to synchronize mouth movements with the spoken words in the input audio. HunyuanCustom delivers unbalanced performance on both fronts. HuMo excels in collaborative performance across video quality, subject consistency, audio-visual sync, and text controllability.

To address the issue of data scarcity, we construct a two-stage multimodal data processing pipeline. Firstly, leveraging large-scale text–video samples [18, 33], we retrieve reference images with the same semantics but different visual attributes [3, 22] for each subject in the video samples from a billion-scale image corpus through detection, matching, filtering, and verification. This aims to ensure faithful subject preservation with the reference images while enabling flexible text editability on the subjects. Secondly, to achieve audiosynchronized video generation, we further filter video samples with synchronized audio tracks using speech enhancement and speech-lip alignment estimation [17]. Through this pipeline, we establish a high-quality multimodal dataset containing paired triplet conditions (text, reference images, and audio), which offers a strong foundation for subsequent learning of multimodal controllability.

Building upon the constructed dataset, we propose a progressive multimodal training paradigm consisting of two sub-tasks. 1) The Subject Preservation Task aims to enable collaborative control between text and image by preserving reference images without compromising the text-following capability of the foundational DiT-based backbone [31]. This is achieved through a minimally invasive image injection strategy that avoids structural modifications to the DiT backbone and confines parameter updates to a limited subset. 2) The Audio-Visual Sync Task introduces audio cross-attention layers to incorporate the audio modality. Unlike previous methods [32, 36] that directly localize audio influence, we propose a focus-by-predicting strategy that implicitly encourages the model to enhance synchronization between audio and human motions. To ensure that the previously acquired subject preservation ability will not be compromised during learning of audio-visual sync, we retain the first task while progressively incorporating the second task. This joint optimization

strategy facilitates collaborative learning of controllability across text, image, and audio modalities. By decoupling the learning of individual capabilities, this paradigm also enables flexible training on partially annotated datasets. For instance, enhancing subject preservation only requires video–image pairs, without the need for synced audio–video data.

During inference, to enable flexible and fine-grained control over triplet input modalities, we propose a time-adaptive classifier-free guidance (CFG) strategy, which dynamically adjusts the guidance strengths at different denoising steps. This allows for precise and efficient collaboration among text following, subject preservation, and audio-visual sync.

The main contributions can be summarized as follows:

- • Concept. We attribute the imbalanced multimodal controllability in existing HCVG methods to the absence of a comprehensive design across data processing and training paradigms for handling heterogeneous inputs. We propose HuMo, a unified framework that enables collaborative control across text, image, and audio modalities. It seamlessly supports various input compositions of text-image, text-audio, and text-image-audio.
- • Methodology. 1) We establish a multimodal data processing pipeline that produces a high-quality and diverse dataset with paired triplet conditions. 2) We propose a progressive multimodal training paradigm with task-specific strategies, which facilitates joint learning of controllabilities across triplet modalities. 3) We design a time-adaptive CFG strategy, enabling flexible, fine-grained, and collaborative multimodal control.
- • Significance. Extensive experimental results show HuMo surpasses the specialized state-of-the-art (SOTA) methods on both subject preservation and audio-visual sync sub-tasks. We further demonstrate its effectiveness and scalability by validating the framework on models of 1.7B and 17B parameters.

### 2 Related Works

#### 2.1 Audio-Driven Human Animation

Audio-driven human animation aims to generate videos from input human images to produce lip movements matching input speech signals. Built on the foundational video generation models [2, 9, 15, 27, 31], audio-driven human animation methods have achieved impressive performance. Hallo3 [5] is the first full-frame-size portrait animation application on a pre-trained DiT model, using cross-attention layers for audio control. To improve the motion dynamics, FantasyTalking [32] proposes to establish global motion at the clip level and optimize lip synchronization at the frame level. OmniHuman-1 [19] scales the training data by incorporating hybrid motion-related conditions to generate more realistic human videos. Despite their strong performance in facial animation and body motion, existing methods still require a subject-complete start frame with visible facial features, which limits user creativity.

#### 2.2 Subject-Consistent Video Generation

Subject-consistent video generation (S2V) aims to generate text-aligned videos with consistent subject appearances according to the input text prompts and reference images. Early approaches use pre-trained semantic encoders [25] to extract features from reference images, achieving identity-preserving video generation via adapters [10, 38] or DiT cross-attention layers [24]. In-context methods [7, 22] leverage the inherent consistency of pre-trained DiT-based models by concatenating reference image latents with noisy video latents. These methods achieve finer-grained subject consistency but weaken textual control. To preserve the textual control of the pre-trained model in our in-context method, we freeze the text-visual cross-attention layers and fine-tune only the self-attention layers. Moreover, existing methods support only image and text modalities and cannot support human speech input, which is critical for vivid and realistic video creation. Concurrent works such as InterActHuman [34] and HunyuanCustom [11] integrate subject-consistent video generation with audio-driven human animation. However, they still struggle to balance the influence of these three modalities. In contrast, we propose a progressive multimodal collaborative training paradigm and time-adaptive CFG to enable flexible multimodal control.

Progressive

[Figure 70]

Text Prompt

[Figure 71]

|Text + Ref Text + Audio Text + Ref +Audio<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]|
|---|

[Figure 76]

|[Figure 77]<br><br>[Figure 78]<br><br>Text Enc|
|---|

Man shoots photos on mountaintop …

- Stage 0
- Stage 1
- Stage 2 Ref Imgs

Text (Pre-trained)

[Figure 79]

2 1 0

|Audio-Attn<br><br>🔥|[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>N|
|---|---|
| | |

|[Figure 83]<br><br>[Figure 84]<br><br>Self-Attn<br><br>[Figure 85]<br><br>🔥|
|---|

|[Figure 86]<br><br>[Figure 87]<br><br>Text-Attn|
|---|

|[Figure 88]| |
|---|---|
| | |

[Figure 89]

[Figure 90]

Audio-Attn

[Figure 91]

|[Figure 92]|
|---|

Text-Attn

Self-Attn

Human Object

VAE

VAE

[Figure 93]

Ref Imgs

Text

[Figure 94]

[Figure 95]

[Figure 96]

Face Loc Loss

Ref Images

Text Human Audio

|[Figure 97]<br><br>[Figure 98]<br><br>Face Loc Predictor<br><br>[Figure 99]<br><br>🔥<br><br>[Figure 100]<br><br>[Figure 101]| | | |
|---|---|---|---|
| | | | |
| |Last 4 Block| | |

|[Figure 102]|
|---|

[Figure 103]

[Figure 104]

[Figure 105]

|[Figure 106]<br><br>[Figure 107]<br><br>Audio Enc<br><br>[Figure 108]<br><br>🔥|
|---|

[Figure 109]

[Figure 110]

Audio

- Figure 3 Overview of our framework. HuMo model (left) is trained based on the proposed data processing pipeline (right). Built upon a DiT-based T2V backbone from Stage 0, the model progressively learns subject preservation and audio-visual sync capabilities in Stages 1 and 2. HuMo achieves collaborative generation across different modality compositions.

### 3 Methodology

To address the challenges of data scarcity and compromised performance in multimodal conditioning, we propose HuMo, a Human-Centric Video Generation framework that enables collaborative control with Multimodal conditions. Given a textual prompt ctxt, reference images cimg, and an audio singal ca, HuMo aims to generate a video where the environment, human identity, appearance, accessories, clothing, as well as facial and body motions are semantically aligned with the input conditions, while remaining spatially and temporally coherent. We begin in Sec. 3.1 by describing the DiT-based T2V backbone [31]. Sec. 3.2 outlines the construction of the multimodal dataset. In Sec. 3.3, we present how a T2V model is extended to support triplet modalities. Finally, Sec. 3.4 describes our inference strategy for flexibly modality conditioning and fine-grained and collaborative generation.

#### 3.1 Preliminaries

In this work, we adopt a DiT-based T2V model [31] as our backbone. As shown in Fig. 3, it incorporates a 3D Variational Autoencoder (VAE) to compress video into a compact latent space. A text encoder [4] is utilized to encode textual information and then injected into the DiT backbone with cross-attention. For training, flow matching [20] is adopted by learning a continuous velocity field that transports samples from a simple prior to the data distribution along deterministic trajectories. HuMo inherits this DiT-based architecture and extends it by incorporating additional image and audio modalities, enabling multimodal conditioning for more controllable video generation. To effectively train the model under this multimodal setup, we formulate the flow matching objective as:

0,z1∥vθ(zt,t,c) − (z1 − z0)∥22, (1)

LFM(θ) = Et,z

where z0 is a sample from the simple prior, z1 is the latent of the target video sample, and c = {ctxt,cimg,ca} denotes the multimodal condition of text, image, and audio. The DiT model vθ takes the noisy latent zt = (1 − t)z0 + tz1, and learns to predict its velocity at any point t ∈ [0,1]. This formulation allows HuMo to efficiently learn a deterministic transport map from noise to data under complex multimodal constraints.

#### 3.2 Multimodal Data Processing Pipeline

High-quality, subject-consistent, and audio-visual synced multimodal data is crucial for HCVG but remains scarce and usually incomplete or misaligned. We propose a multimodal data processing pipeline to construct a diverse dataset of triplet conditions, which forms the foundation for the training paradigm we introduce later.

As shown in Fig. 3, our pipeline unfolds in several stages. In Stage 0, we begin with a large-scale video pool [18, 33] and leverage powerful VLMs [1, 30] for detailed text descriptions, ensuring basic textual modality for each video sample. In Stage 1, to avoid the common copy-paste issue [22] in subject preservation, we

follow prior works [3, 37] to sparsely sample video frames and retrieve cross-paired reference images from a billion-scale image corpus. For humans, we retrieve images with the same identity but varying in appearance (e.g., makeup, pose, background, clothing, age); for objects, we match the same semantic category but with varied visual attributes (e.g., color, shapes, viewpoint). This strategy improves text controllability for reference-guided generation by discouraging direct frame replication in training. In Stage 2, we supplement the data with audio modality by filtering for speech segments and creating tightly aligned audio-visual pairs with lip-sync analysis [17]. It is worth noting that in Stage 2, since audio-aligned data predominantly exists in human-centric videos, we can only retrieve human reference images.

Through this pipeline, we build a multimodal dataset consisting of detailed text prompts, consistent reference images, synced audio, and the target video. Stage 1 includes O(1)M video samples with text and reference images, while Stage 2 contains O(50)K samples with temporally-aligned audio. We will release the training data to facilitate future research on high-quality, controllable HCVG and to promote reproducibility in multimodal generation studies.

#### 3.3 Progressive Multimodal Training

We structure the acquisition of multimodal control capabilities into two distinct yet progressive training stages. Stage 1 establishes text-image controllability through the subject preservation task. Stage 2 realizes the joint learning of text-image-audio controllability via progressive training of the previous task and the audio-visual sync task. Training data for each stage is sourced from the corresponding stage of our data processing pipeline.

Subject Preservation. In Stage 1, we introduce reference images as additional inputs. To preserve the strong text following and image synthesis ability of the original DiT backbone [31], we adopt the minimal-invasive image injection strategy, which adheres to two key principles: avoiding structural modifications of the DiT backbone and confining parameter updating to a limited subset.

Specifically, without modifying the model architecture, we concatenate the VAE latents zimg of the reference images cimg with the noisy latent zt along the temporal dimension as inputs. cimg can be a single image of a human or an object, or the composition of multiple images. To prevent the model from misinterpreting the reference as the start frame and performing undesired image continuation, we always place the reference latents at the end of the video latent sequence, forming the input as [zt;zimg]. This design encourages the model to actively extract subject identity information from the reference images via self-attention and propagate it across all frames, enabling subject-consistent generation. The training is restricted to the self-attention layers of DiT to minimally affect its inherent text alignment and visual generation capabilities. Notably, text remains as input to ensure semantic consistency and controllability in generation.

Audio-Visual Sync. In stage 2, we extend the first-stage setup by incorporating audio modality. Following prior works [16, 32], we insert an audio cross-attention layer in each DiT block. Audio features are extracted via Whisper [26] for generalization across speakers and languages. Based on the observation that human motions are primarily aligned with temporally nearby audio cues [28], we construct the final audio embedding ca ∈ Rf×n×d by concatenating audio embeddings within a temporal window centered at each video frame stamp, where the window length n = 5 and and f same with the sequence length of the video latents. ca is conducted cross-attention calculation with hidden states frame-by-frame:

##### QzK⊤a

√

Attention(hz,ca) = softmax(

d

)Va, (2)

where Qz transformed from the hidden states hz ∈ Rf×h×w×d of video latent in DiT net, Ka and Va transformed from audio embedding.

The aforementioned cross-attention is computed between the audio signal and the full-frame noisy latent. As is known, audio cues are usually most correlated with localized human regions (e.g., face, lip). Consequently, this coarse-grained attention can lead to weak synchronization between audio and human motion. Prior I2V-based methods [36] address this by detecting facial regions in the start frame and restricting audio cross-attention exclusively to these areas. This strategy, however, is not applicable when the input is a full-frame facial image, as the model cannot predetermine the location of the face before denoising.

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

- (a) In slow motion, the scene transitions between summer and winter, highlighting the seasonal contrast.
- (b)a woman with sun-kissed skin and long dark hair walks along a golden beach in a red chiffon dress.

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Ref Images CFG Weights A CFG Weights B Time-Adaptive CFG

Figure 4 The proposed time-adaptive CFG balances text guidance and identity preservation.

We therefore propose a focus-by-predicting strategy to implicitly guide the model to focus more on the facial region. During training, we introduce a mask predictor Fmask to estimates the potential facial region distribution Mpred = Fmask(hz). Mpred is supervised by the ground-truth binary face mask Mgt using a binary cross-entropy (BCE) loss. Since early DiT blocks lack stable spatial representations, we only insert Fmask after the audio cross-attention modules in the last four DiT blocks and average these outputs as the final predicted mask. To further prevent the supervision from weakening when the face region is too small, we introduce a size-aware weight [36] by adaptively emphasizing the supervision on the face region:

hw

· BCE(Mpred,Mgt). (3)

Lmask =

w j=1 M(gti,j)

- h
- i=1

Unlike prior methods [34, 36] that apply hard gating on audio attention outputs, which is a truncation design we consider suboptimal. The focus-by-predicting strategy acts as a soft regularizer, which steers the model’s focus without crippling its representational capacity and retains the flexibility to model full-body kinematics and complex interactions.

Progressive Training. To ensure that the acquisition of audio-visual sync ability does not degrade the subject preservation ability, we employ a progressive task-weighting curriculum in stage 2. Initially, training is dominated by the subject preservation task (80% ratio, audio input as null) to reinforce existing ability, while the audio-visual sync task constitutes the remaining 20%. As training progresses, we gradually increase the proportion of the audio-visual sync task to 50%. Throughout this stage, we still adhere to the finetuning principles in stage 1 by only updating the self-attention layers and audio-related modules. This progressive strategy facilitates a smooth transition for the model from bi-modal (text, image) to tri-modal (text, image, audio) inputs, ensuring stable training and the collaboration of multimodal learning.

#### 3.4 Inference Strategies

Flexible Multimodal Control. For flexible, independent control at inference, we adapt CFG with separate guidance scales (λtxt, λimg, λa) for each modality:

vθ(zt,t,c) = λa[vθ(ctxt,cimg,ca) − vθ(ctxt,cimg,∅)]

+ λimg[vθ(ctxt,cimg,∅) − vθ(ctxt,∅,∅)]

+ λtxt[vθ(ctxt,∅,∅) − vθ(∅,∅,∅)]

+ vθ(∅,∅,∅). (4)

HuMo can generate coherent results even when modalities are absent, enabling condition combination of [ctxt,cimg], [ctxt,ca], and [ctxt,cimg,ca]. Missing conditions are simply replaced by a null token ∅.

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

- (a)

A handsome man in a black suit and white shirt, with neatly side-swept hair, gracefully puts on deep-brown leather gloves with fine stitching and ruched wrists, fully focused on the task.

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

- (b)

A gray-haired man, a suited man with glasses, a bearded man, and a woman in a black top, step into an ancient Chinese Buddhist temple. A long shot slowly moving to a close up.

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

(c) A young witch with a red bow, in a black top and white apron, flies on a broomstick with a black kitten wearing a red bow. They glide through sunlit green trees under blue sky with white clouds.

Ref Images Kling 1.6 MAGREF HunyuanCustom Phantom Ours

Figure 5 Qualitative comparison for the subject preservation task. Zoom in for details.

Video Quality Text Following Subject Consistency AES↑ IQA↑ HSP↑ TVA↑ ID-Cur↑ ID-Glink↑ CLIP-I↑ DINO-I↑

Method

Kling 1.6 [14] 0.645 0.714 3.792 2.564 0.470 0.501 0.639 0.394 MAGREF [7] 0.622 0.708 3.331 2.852 0.334 0.359 0.665 0.416 HunyuanCustom [11] 0.592 0.705 3.705 1.777 0.309 0.335 0.649 0.426 Phantom [22] 0.608 0.150 3.612 2.877 0.649 0.674 0.677 0.426

Ours-1.7B 0.586 0.680 3.432 3.222 0.609 0.668 0.660 0.414 Ours-17B 0.657 0.717 3.906 3.939 0.731 0.757 0.687 0.447

Table 1 Quantitative results for the subject preservation task with text and reference images as inputs.

Time-Adaptive CFG. We observe that the influence of each modality shifts throughout the denoising process: early steps tend to construct the overall semantic structure and spatial layout guided by text, while later steps focus on fine-grained details (e.g., identity similarity, audio-visual sync). A static CFG configuration is suboptimal for the shifting modal dynamics. We therefore propose a time-adaptive CFG that dynamically switches between two configurations of guidance scales. From timestep 1.0 to 0.98, we adopt the text/imagedominant configuration to establish a stable semantic layout (e.g., character and scene composition). From timestep 0.98 to 0, we shift to parameters that emphasize audio and image control. Empirical results demonstrate that this strategy significantly improves multimodal collaboration and enhances overall video quality.

### 4 Experiment

Implementation Details. We build our method on the backbones of Wan-2.1-1.3B and Wan-2.1-14B [31]. All video samples are resampled to 25 fps with a 480×832 resolution. We employ a two-stage training strategy. Stage 1 involves training for 40k steps, with all audio-related modules disabled. Stage 2 continues for another 40k steps, where we enable the audio-related modules.

Comparative Methods. We compare HuMo with two categories of baselines. 1) S2V methods that take text and subject reference images as input, including open-source models MAGREF [7], HunyuanCustom [11], Phantom-Wan-14B [22], and the commercial closed-source method Kling 1.6 [14]. Comparison is conducted on an in-house benchmark of 100 test cases involving humans, objects, and animals. 2) Audio-visual synced methods that take text, image, and audio as inputs, including open-source methods Hallo3 [5], FantasyTalking [32], HunyuanCustom [11], and the commercial closed-source OmniHuman-1 [19]. Comparison is conducted on the MoCha benchmark [35]. Notably, HunyuanCustom is the only method that supports reference images,

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

| |
|---|

(a)

A man in a checkered shirt and headphones sings, plays a silver guitar, and speaks to the camera in a recording studio. A static front shot captures his rhythmic movements and deeply focused, emotionally engaged expression against a lit, card-decorated black wall. Speaking Word [ Gas /ɡæs/ ]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

| |
|---|

A woman in a bulky space suit with a large helmet speaks to the camera against a blurred Mars backdrop. Her face remains clearly visible, framed by loose strands of tied-back hair and dark, determined eyes. A static close-up captures her expressive face and upper body as she continues speaking. Speaking Word [ One /wʌn/ ]

- (b)

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

| |
|---|

- (c)

A gray-haired, bearded man in his 60s speaks thoughtfully to the camera at a Paris café. Wearing a wool coat, button-down shirt, brown beret, and glasses, he has a professorial look. A static close-up captures his mostly still upper body, lit by cinematic golden light with Parisian streets in the background. Speaking Word [ Power /'paʊər/ ]

Start/Ref Image OmniHuman-1 Hallo3 FantasyTalking HunyuanCustom Ours

Figure 6 Qualitative comparison for audio-visual sync task. For the I2V-based methods, we use the first frame generated by MoCha [35] as the input start frame. For HunyuanCustom and Ours, the cropped face from the start frame is utilized as input. The official OmniHuman-1 [19] website API does not support text input. Zoom in for details.

Video Quality Text Following SubjectSubject Consistency Audio-Visual Sync

Method

AES↑ IQA↑ HSP↑ TVA↑ ID-Cur↑ ID-Glink↑ Sync-C↑ Sync-D↓ OmniHuman [19] 0.545 0.682 4.503 - 0.677 0.727 6.526 7.784 Hallo3 [5] 0.381 0.634 4.200 6.117 0.726 0.727 5.189 9.212 FantasyTalking [32] 0.455 0.652 4.444 6.209 0.703 0.729 3.202 10.914 HunyuanCustom [11] 0.358 0.619 4.370 6.246 0.729 0.716 4.562 9.892 Ours-1.7B 0.322 0.661 4.350 5.865 0.721 0.729 6.005 8.648 Ours-17B 0.589 0.718 4.537 6.508 0.747 0.740 6.252 8.577

Table 2 Quantitative results for the audio-visual sync task on MoCha benmark.

while all other baselines rely on a subject-complete start frame and follow the I2V generation paradigm. The latter methods are easier to achieve higher visual quality.

Evaluation Metrics. We conduct comprehensive evaluation using multiple objective metrics covering four key aspects. 1) Video Quality. We evaluate visual appeal and perceptual quality using aesthetics (AES) and image quality assessment (IQA) from the widely-used VBench [13]. Specifically for human videos, we leverage the SOTA VLM, Gemini-2.5-Pro [30], to estimate the human structure plausibility (HSP). 2) Text-Video Alignment (TVA). Semantic consistency between the input text prompt and generated video is measured via the VLM-based reward model [21]. 3) Subject Consistency. For human identity, we detect and crop faces from generated frames, and compute similarity with the reference image using Face-Cur [12] and Face-Glink [6]. For non-facial objects, we use DINO-I [23] and CLIP-I [25] scores to compute average embedding similarity. 4) Audio-Visual Sync. We adopt Sync-C and Sync-D [17] to quantify alignment between input audio and facial motion.

#### 4.1 Multimodal Conditioned Comparison

Subject Preservation Task. 1) Qualitatively. As shown in Fig. 5, HuMo demonstrates superior performance in text following ability compared to other methods. For instance, in case (b), where the prompt is “step into a temple", other methods fail to generate the correspondence. Our method shows strong subject preservation and generalizes well to unseen four-subject cases. It accurately maintains four distinct human identities in case (b), while other methods suffer from missing persons or human identity drift. HuMo also excels in

Variants AES ↑ TVA ↑ ID-Cur ↑ Sync-C ↑

Full Finetune 0.529 6.157 0.749 6.250 w/o Progressive Training 0.541 6.375 0.724 6.106 w/o Focus-by-Predicting 0.587 6.507 0.730 5.946

Ours-17B 0.589 6.508 0.747 6.252

Table 3 Quantitative ablation study on MoCha.

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

|?|
|---|

- (a) Man in a suit and sunglasses, an airplane, smiling as he talks on his phone. Speaking Word [ Life /laɪf/ ]

- (b) Sunlit beach, seagulls, a man energetically plays a brown leather football. Speaking Word [ Protect /prəˈtekt/ ]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Ref Images Full Finetune w/o Progressive w/o Face Loc Ours

Figure 7 Qualitative ablation study. Zoom in for details.

visual aesthetics and human structure plausibility. In case (a), HuMo generates a person wearing gloves without structural artifacts, whereas baselines show noticeable limb degradation. In case (c), HuMo seamlessly integrates a background image into the generation, despite not being trained on any background images. 2) Quantitatively. Tab. 1 reveals that HuMo outperforms other methods in aesthetic and overall image fidelity. HuMo achieves a strong capability in modeling human pose and body integrity with the highest HSP score. For text following and subject consistency with reference images, HuMo also achieves the SOTA performance, which is consistent with the qualitative observation. This highlights our model’s capabilities to achieve strong textual editability without compromising subject consistency.

Audio-Visual Sync Task. 1) Qualitatively. Fig. 6 first reveals HuMo’s superior capability in text following. In case (a) and (c), HuMo successfully synthesizes the “silver guitar" and “golden light in the background" respectively, whereas other methods fail to render these specific details. This observation underscores a fundamental weakness of I2V-based methods - their limited ability for re-editing the provided subject-complete start frame. Furthermore, in case (b), when provided with a dimly lit facial image, other methods are unable to generate the “clearly visible face" as required by the text prompt. HuMo, by contrast, not only synthesizes a clear face but also preserves the identity of the reference face image. 2) Quantitatively. As shown in Tab. 2, HuMo attains the highese scores for aesthetic quality and text follwing. In terms of HSP and identity similarity, our method outperforms HunyuanCustom. Notably, HuMo also surpasses other I2V-based methods, despite their inherent advantage with stronger priors on body layout and facial structure. For audio-visual sync, our 1.7B model already outperforms several open-source specialized models and trails only slightly behind the commercial method OmniHuman-1. It is crucial to note that for this evaluation, we only utilize a single reference face image. As illustrated in Fig. 1, HuMo supports a combined input of audio and multiple reference images (e.g., facial photos, clothing, animal), offering enhanced controllability and customization.

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

- Figure 8 Given the same reference subject reference image but different text prompts, our method achieves collaborative text-image controllability.

#### 4.2 Method Analysis

Ablation Study for Training Strategies. 1) Full Finetuning. We update all parameters of the DiT model instead of confining parameter updating to a limited subset. It leads to significant drops of AES and TVA scores in Tab. 3. Visually, Fig. 7(a) fails to generate the “phone", and case (b) exhibits noticeable artifacts. Full finetuning disrupts the DiT’s pretrained capabilities for high-quality video synthesis and text-image alignment. 2) w/o Progressive Training. We train two tasks in a single stage. This change leads to a degradation across most evaluation metrics. The generated character exhibits low identity similarity to the reference image. This suggests that without progressively building different capabilities, the model struggles to achieve effective coordination across modalities. 3) w/o Focus-by-Predicting. We remove the focus-by-predicting strategy. It results in a decline in Sync-C score, and the generated lip movements are misaligned with the spoken word. This highlights the importance of face location prediction for learning audio-visual correspondence. Furthermore, we observe a decline in ID-Cur score, indicating that this strategy also implicitly contributes to better facial identity consistency.

Ablation Study for Inference Strategy. As shown in Fig. 4, CFG parameters A focus on text-driven layout control, while CFG parameters B emphasize identity similarity. Our proposed time-adaptive CFG dynamically adjusts the weights of the two parameters at different generation stages: in the early time steps, it prioritizes textual guidance to ensure the generated content aligns with the prompt structure; in later time steps, it increases the emphasis on identity preservation. This adaptive strategy effectively balances semantic controllability and identity consistency, leading to more precise and stable character generation.

Text controllability. As shown in Fig. 8, we input the same reference image and use different text prompts to vary the character’s clothing, accessories, and makeup in order to evaluate text controllability. We observe that while the character’s identity remains consistent, their appearance changes accordingly, demonstrating the effectiveness of our method in achieving text-image collaboration. Notably, previous subject-preserving methods [11, 22] primarily focus on compositionality, which embeds reference images within the semantic context of the text, while overlooking the “editability" of the text, i.e., the ability to adjust the degree of information injected from the reference image. In this work, we specifically enhance this aspect.

Image controllability. As shown in Fig. 9, to showcase the movie-level, highly customizable generation capabilities of HuMo, we re-create scenes from the renowned TV series “Game of Thrones” using both text-audio (TA) and text-image-audio (TIA) modes. In the TIA setting, to demonstrate flexible controllability of reference images, we condition the model on a reference portrait of an actor different from the one in the original footage. Across both TA and TIA modes, HuMo produces videos that preserve the similar layout and visual elements of the original scenes. In TIA mode, the reference identity is seamlessly integrated into the target semantic context, enabling short-form production from a single, unadorned headshot of the actor.

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

###### TATIAOriginal

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

|[Figure 191]|
|---|

|[Figure 192]|
|---|

|[Figure 193]|
|---|

[Figure 194]

[Figure 195]

|[Figure 196]|
|---|

- Figure 9 Re-creation of Game of Thrones (Season 3), named as Faceless Thrones. We extract captions [1] and audio from the original video and generate new videos with HuMo using two modes: text–audio (TA) and text–image–audio (TIA). The reference identity image for TIA mode is displayed at the top-left corner.

### 5 Conclusion

We propose HuMo, a novel human-centric video generation framework with multimodal conditions. HuMo establishes a multimodal data processing pipeline to produce high-quality and diverse datasets with paired text prompts, reference images, and audio. The proposed progressive multimodal training paradigm successfully integrates the control capabilities of text, image, and audio modalities into a unified model. Taking advantage of the proposed time-adaptive CFG strategy, our model enables flexible, fine-grained, and collaborative control over all three modalities during inference. HuMo satisfies the multiple requirements of text prompt following, subject preservation, and audio-visual sync in human-centric short video creation.

### 6 Ethical Considerations

The development of HuMo for Human-Centric Video Generation may raise several ethical concerns. First, the ability to synthesize realistic human videos from multimodal inputs (text, image, and audio) may lead to misuse, such as deepfakes or non-consensual content creation. Ensuring informed consent and protecting individuals’ likenesses are critical. Second, fine-grained control over generated content calls for responsible usage guidelines to prevent manipulation or misinformation. Developers and users must adhere to ethical standards, including transparency, data privacy, and the prevention of harm.

### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, and et al. Qwen2.5-vl technical report, 2025.
- [2] Shoufa Chen, Chongjian Ge, Yuqi Zhang, Yida Zhang, Fengda Zhu, Hao Yang, Hongxiang Hao, Hui Wu, Zhichao Lai, and et al. Goku: Flow based video generative foundation models. arXiv preprint arXiv:2502.04896, 2025.

- [3] Zhuowei Chen, Bingchuan Li, Tianxiang Ma, Lijie Liu, Mingcong Liu, Yi Zhang, Gen Li, Xinghui Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom-data : Towards a general subject-consistent video generation dataset, 2025.
- [4] Hyung Won Chung, Xavier Garcia, Adam Roberts, Yi Tay, Orhan Firat, Sharan Narang, and Noah Constant. Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum? id=kXwdL1cWOAi.

- [5] Jiahao Cui, Hui Li, Yun Zhan, Hanlin Shang, Kaihui Cheng, Yuqi Ma, Shan Mu, Hang Zhou, Jingdong Wang, and Siyu Zhu. Hallo3: Highly dynamic and realistic portrait image animation with video diffusion transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21086–21095, June 2025.

- [6] Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Cotsia, and Stefanos P Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, page 1–1, 2021.

- [7] Yufan Deng, Xun Guo, Yuanyang Yin, Jacob Zhiyuan Fang, Yiding Yang, Yizhi Wang, Shenghai Yuan, Angtian Wang, Bo Liu, Haibin Huang, and Chongyang Ma. Magref: Masked guidance for any-reference video generation, 2025.
- [8] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, and et al. Seedream 3.0 technical report, 2025.
- [9] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, and et al. Seedance 1.0: Exploring the boundaries of video generation models, 2025. URL https://arxiv.org/abs/2506.09113.
- [10] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024.

- [11] Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, and Qinglin Lu. Hunyuancustom: A multimodal-driven architecture for customized video generation, 2025.
- [12] Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. Curricularface: Adaptive curriculum learning loss for deep face recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun 2020.

- [13] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21807–21818, June 2024.

- [14] Kling. Multi-reference images to video generation feature. https://app.klingai.com/cn/release-notes/ 2025-07-24, 2025.
- [15] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [16] Zhe Kong, Feng Gao, Yong Zhang, Zhuoliang Kang, Xiaoming Wei, Xunliang Cai, Guanying Chen, and Wenhan Luo. Let them talk: Audio-driven multi-person conversational video generation, 2025.
- [17] Chunyu Li, Chao Zhang, Weikai Xu, Jingyu Lin, Jinghui Xie, Weiguo Feng, Bingyue Peng, Cunjian Chen, and Weiwei Xing. Latentsync: Taming audio-conditioned latent diffusion models for lip sync with syncnet supervision,

2025. URL https://arxiv.org/abs/2412.09262.

- [18] Hui Li, Mingwang Xu, Yun Zhan, Shan Mu, Jiaye Li, Kaihui Cheng, Yuxuan Chen, Tan Chen, Mao Ye, Jingdong Wang, and Siyu Zhu. Openhumanvid: A large-scale high-quality dataset for enhancing human-centric video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7752–7762, June 2025.

- [19] Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, and Chao Liang. Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

- [20] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

- [21] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, Xintao Wang, Xiaohong Liu, Fei Yang, Pengfei Wan, Di Zhang, Kun Gai, Yujiu Yang, and Wanli Ouyang. Improving video generation with human feedback, 2025.
- [22] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

- [23] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, page 38–55, 2024.

- [24] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

- [25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139, pages 8748–8763, 2021.

- [26] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pages 28492–28518, 2023.

- [27] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, and et al. Seaweed-7b: Cost-effective training of video generation foundation model, 2025. URL https://arxiv.org/abs/2504.08685.
- [28] Supasorn Suwajanakorn, Steven M. Seitz, and Ira Kemelmacher-Shlizerman. Synthesizing obama: learning lip sync from audio. ACM Trans. Graph., 36(4), 2017.

- [29] Gemini Team. Gemini 2.5 flash image. https://aistudio.google.com/prompts/new_chat?model=gemini-2. 5-flash-image-preview/, 2025.
- [30] Gemini Team. Gemini 2.5: Our most intelligent ai model. https://blog.google/technology/google-deepmind/ gemini-model-thinking-updates-march-2025/, 2025.
- [31] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, and et al. Wan: Open and advanced large-scale video generative models, 2025.
- [32] Mengchao Wang, Qiang Wang, Fan Jiang, Yaqi Fan, Yunpeng Zhang, Yonggang Qi, Kun Zhao, and Mu Xu. Fantasytalking: Realistic talking portrait generation via coherent motion synthesis. In Proceedings of the 33th ACM International Conference on Multimedia, 2025.

- [33] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, Fei Yang, Pengfei Wan, and Di Zhang. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8428–8437, June 2025.

- [34] Zhenzhi Wang, Jiaqi Yang, Jianwen Jiang, Chao Liang, Gaojie Lin, Zerong Zheng, Ceyuan Yang, and Dahua Lin. Interacthuman: Multi-concept human animation with layout-aligned audio conditions, 2025.

- [35] Cong Wei, Bo Sun, Haoyu Ma, Ji Hou, Felix Juefei-Xu, Zecheng He, Xiaoliang Dai, Luxin Zhang, Kunpeng Li, Tingbo Hou, Animesh Sinha, Peter Vajda, and Wenhu Chen. Mocha: Towards movie-grade talking character synthesis, 2025.
- [36] Hongwei Yi, Tian Ye, Shitong Shao, Xuancheng Yang, Jiantong Zhao, Hanzhong Guo, Terrance Wang, Qingyu Yin, Zeke Xie, Lei Zhu, Wei Li, Michael Lingelbach, and Daquan Zhou. Magicinfinite: Generating infinite talking videos with your words and voice, 2025.
- [37] Shenghai Yuan, Xianyi He, Yufan Deng, Yang Ye, Jinfa Huang, Bin Lin, Jiebo Luo, and Li Yuan. Opens2v-nexus: A detailed benchmark and million-scale dataset for subject-to-video generation, 2025.
- [38] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identitypreserving text-to-video generation by frequency decomposition. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12978–12988, 2025.

