# arXiv:2602.12160v1[cs.CV]12Feb2026

[Figure 1]

[Figure 2]

## DreamID-Omni: Unified Framework for Controllable Human-Centric Audio-Video Generation

### Xu Guo1,⋆, Fulong Ye2,⋆, Qichao Sun2,⋆,†, Liyang Chen1, Bingchuan Li2,†, Pengze Zhang2, Jiawei Liu2, Songtao Zhao2,§, Qian He2, Xiangwang Hou1,§

1Tsinghua University, 2Intelligent Creation Lab, ByteDance ⋆Equal contribution, †Project Lead, §Corresponding author

### Abstract

Recent advancements in foundation models have revolutionized joint audio-video generation. However, existing approaches typically treat human-centric tasks including reference-based audiovideo generation (R2AV), video editing (RV2AV) and audio-driven video animation (RA2V) as isolated objectives. Furthermore, achieving precise, disentangled control over multiple character identities and voice timbres within a single framework remains an open challenge. In this paper, we propose DreamID-Omni, a unified framework for controllable human-centric audio-video generation. Specifically, we design a Symmetric Conditional Diffusion Transformer that integrates heterogeneous conditioning signals via a symmetric conditional injection scheme. To resolve the pervasive identity-timbre binding failures and speaker confusion in multi-person scenarios, we introduce a Dual-Level Disentanglement strategy: Synchronized RoPE at the signal level to ensure rigid attention-space binding, and Structured Captions at the semantic level to establish explicit attribute-subject mappings. Furthermore, we devise a Multi-Task Progressive Training scheme that leverages weakly-constrained generative priors to regularize strongly-constrained tasks, preventing overfitting and harmonizing disparate objectives. Extensive experiments demonstrate that DreamID-Omni achieves comprehensive state-of-the-art performance across video, audio, and audio-visual consistency, even outperforming leading proprietary commercial models. We will release our code to bridge the gap between academic research and commercial-grade applications.

Date: February 13, 2026 Project Page (Demo, Codes, Models): https://guoxu1233.github.io/DreamID-Omni/

### 1 Introduction

Recently, joint audio-video generation has seen rapid progress, with many breakthrough works emerging. For example, commercial models such as Veo3, Sora2, Wan 2.6 [47] and Seedance 1.5 Pro [4] have achieved impressive results. In the open-source community, models like Ovi [36] and LTX-2 [15] have also demonstrated promising performance. These advances have greatly promoted the development of joint audio-video generation. However, in real-world applications, supporting more controllable generation particularly within human-centric scenarios is crucial.

Controllable human-centric generation has advanced in several directions. Works such as Phantom [35] and Wan2.6 [47] utilize reference images or voice timbres for video (R2V) or audio-video (R2AV) generation, which rely solely on text prompts as a weakly-constrained guidance. To achieve higher controllability, other

Human-Reference Audio-Video Generation (R2AV)

|Ref voice1<br><br>[Figure 3]<br><br>Ref ID1<br><br>[Figure 4]<br><br>|[Figure 5]|
|---|
|
|---|

|Ref voice2<br><br>Ref ID2<br><br>|[Figure 6]|
|---|
<br><br>[Figure 7]<br><br>[Figure 8]|
|---|

|Ref voice<br><br>[Figure 9]<br><br>Ref ID<br><br>|[Figure 10]|
|---|
<br><br>[Figure 11]|
|---|

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

Text:

<sub1>... brown jacket, in open plains... said: “Sorry... I'm going to end your life right now.”

... in office, ... <sub1> ... and says: “We need ...”. <sub2> replies: “Yes, Let me ...”

Text:

|Ref voice1<br><br>[Figure 23]<br><br>Ref ID1<br><br>|[Figure 24]|
|---|
<br><br>[Figure 25]|
|---|

|Ref voice1<br><br>[Figure 26]<br><br>Ref ID2<br><br>|[Figure 27]|
|---|
<br><br>[Figure 28]|
|---|

|Ref oice3<br><br>[Figure 29]<br><br>Ref ID3<br><br>[Figure 30]<br><br>|[Figure 31]|
|---|
|
|---|

|Ref voice1<br><br>[Figure 32]<br><br>Ref ID1<br><br>|[Figure 33]|
|---|
<br><br>[Figure 34]|
|---|

|Ref voice2<br><br>[Figure 35]<br><br>Ref ID2<br><br>|[Figure 36]|
|---|
<br><br>[Figure 37]|
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Text:

In meeting room, ... <sub1> asserts: “We should ...”. <sub2> replies: “I agree ...”, <sub3> suggests: “Let's...”

... rustic cabin, ... <sub1> put down ... says: “I'm so glad ...”. <sub2> smiles and saids: “Me too ...”

Text:

Human-Reference Video Editing (RV2AV) Human-Reference Audio-Driven Video Animation (RA2V)

|Ref ID<br><br>|[Figure 52]|
|---|
|
|---|

||[Figure 53]|
|---|
<br><br>Ref voice<br><br>[Figure 54]<br><br>Ref ID<br><br>[Figure 55]|
|---|

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

|Source Video|Face swap result|
|---|---|

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Driving Audio

Text:

<sub1> in... said loudly: “Nice work... get a fire team to...”

... in a modern indoor <sub1> ... said: “...You used your child as bait...”

Text:

|Ref ID<br><br>|[Figure 66]|
|---|
|
|---|

[Figure 67]

[Figure 68]

[Figure 69]

||[Figure 70]|
|---|
<br><br>Ref ID|
|---|

[Figure 71]

|Source Video|EditHumanresultswap result|
|---|---|

[Figure 72]

[Figure 73]

[Figure 74]

Driving Audio

[Figure 75]

Driving Audio

...in office, <sub1> wearing black suit, said: “... It’s coming from your...”

Text:

Text:

<sub1>... light grey crewneck shirt... outdoors... speak: “really bad guy, someone ...”

- Figure 1 Showcase of DreamID-Omni. DreamID-Omni seamlessly unifies reference-based audio-video generation (R2AV), video editing (RV2AV), and audio-driven video animation (RA2V).

approaches introduce stronger supervision, such as source videos or driving audio, for strongly-constrained generation. For instance, Humo [2] animates videos (RA2V) based on reference identities and driving audio, while works like HunyuanCustom [21] and VACE [25] perform video editing given a reference identity and source video, which can be further extended to replace the corresponding audio (RV2AV). Despite these advancements, these capabilities are largely treated as isolated tasks. Researchers in the video-only domain have begun to shift toward unified architectures [19, 25, 30, 40, 59, 61] to enhance task flexibility and reduce the operational overhead of deploying multiple models. However, the joint audio-video domain still lacks a unified perspective. Fundamentally, we observe that R2AV, RV2AV, and RA2V all share an identical objective: mapping a static identity anchor (image and audio) onto a dynamic spatio-temporal canvas (text, source video, or driving audio). Based on this insight, these tasks are inherently amenable to a unified framework trained on a consistent data source, transcending the limitations of task-specific silos.

Nevertheless, developing this unified framework presents several challenges: (1) How to build a unified model framework that supports generation, editing and animation; (2) How to address identity-timbre binding and speaker confusion in multi-person generation; (3) How to design effective training strategies to prevent conflicts among multiple tasks.

To address these challenges, we introduce DreamID-Omni, which integrates reference-based generation, editing, and animation into a single paradigm. DreamID-Omni builds upon a dual-stream Diffusion Transformer [38] (DiT) architecture, where video and audio streams interact via bidirectional cross-attention for fine-grained synchronization. We propose a Symmetric Conditional DiT design that unifies heterogeneous conditioning signals—reference images, voice timbres, source videos, and driving audio—into a shared latent space, enabling seamless task switching without architectural changes.

To resolve multi-person confusion, we propose a Dual-Level Disentanglement strategy. At the signal level, Synchronized Rotary Positional Embeddings (Syn-RoPE) is introduced to bind reference identities with their corresponding voice timbres within the attention space. At the semantic level, Structured Captions utilize anchor tokens paired with fine-grained descriptions to establish explicit mappings between specific subjects and their respective attributes or speech content.

Finally, we devise a Multi-Task Progressive Training strategy to harmonize the three tasks. In the initial

two stages, we focus exclusively on the weakly-constrained R2AV task, employing in-pair reconstruction and cross-pair disentanglement to enhance identity and timbre fidelity while encouraging the model to learn robust reference representations. In the final stage, strongly-constrained tasks (RV2AV and RA2V) are introduced for joint training with R2AV. This approach prevents the model from overfitting to strongly-constrained tasks, thereby maintaining superior performance on the weakly-constrained generation task.

In summary, our contributions are as follows: (1) We propose DreamID-Omni, a novel human-centric controllable generation framework based on a Symmetric Conditional DiT, which seamlessly integrates R2AV, RV2AV, and RA2V tasks. (2) We introduce Dual-Level Disentanglement, which addresses identity-timbre binding and speaker confusion in multi-person generation via Syn-RoPE and Structured Captions. (3) We present a Multi-Task Progressive Training strategy that effectively harmonizes diverse tasks with varying constraint strengths. (4) Extensive experiments demonstrate that DreamID-Omni achieves comprehensive state-of-the-art performance across video, audio, and audio-visual consistency, even when compared to leading proprietary commercial models.

### 2 Related Work

- 2.1 Joint Audio-Video Generation

Recent advancements in diffusion-based foundation models in video generation [12, 26, 47] and audio generation [13, 32] have significantly expanded the frontier of joint audio-video synthesis. While pioneering works [42] use coupled U-Net backbones, current DiT-based approaches dominate the field. These methods typically employ either dual-stream architectures [15, 17, 33, 34, 36, 48] with specialized fusion layers (e.g., cross-attention) or unified DiT structures [22, 49, 50, 63] with joint self-attention to achieve synchronized multi-modal alignment. Despite their impressive generative fidelity, these models are primarily designed for vanilla text-to-audio-video or first-frame-conditioned synthesis. They lack the capability to condition the generative process on external identity or voice timbre references. This limitation restricts their utility in scenarios requiring persistent identity and timbre consistency.

- 2.2 Controllable Video Generation Model

Reference-based Generation. To enhance controllability, reference-based video generation has emerged as a prominent research direction, focusing on maintaining identity consistency by integrating reference features into the diffusion process. While initial efforts [18, 39, 62] were primarily tailored for single-identity scenarios, subsequent research has extended these capabilities to multi-subject settings [1, 10, 21, 23, 29, 35, 65]. However, these works are typically video-centric and do not support audio generation.

Video Editing and Animation. In terms of temporal control, tasks can be categorized into video editing and audio-driven video animation. Editing frameworks [5, 7, 14, 25, 37, 43, 52, 58] allow for the modification of identity attributes within the source video. Audio-driven video animation [2, 31, 51, 54, 57] aims to generate videos from reference images to produce lip movements matching input speech signals. Despite their success, these models are all task-specific, and no existing model attempts to unify reference-based generation, editing, and animation.

- 3 Methodology

- 3.1 Problem Formulation

We unify the landscape of controllable human-centric generation into a single probabilistic framework. Given a text prompt T , a set of reference identities I = {I1,...,IN}, and corresponding reference voice timbres A = {A1,...,AN}, the goal is to synthesize a synchronized video-audio stream Y = {Yvideo,Yaudio}.

To support reference-based editing and animation tasks, we introduce two optional structural conditions: a source video context Vsrc and a driving audio stream Adri. The framework models the conditional distribution:

P(Y | T ,I,A,Vsrc,Adri) (1)

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

###### Ref Image Caption:

- <img1>: A man with dark hair..., identified as <sub1>.
- <img2>: A smiling woman with dark hair..., identified as <sub2>.

###### Target Video Caption:

A coffee shop with warm ... ,<sub1> is on the left, wearing dark... <sub2> is on the right, wearing a simple...,<sub1> looks at ... <sub2> smiling...

###### Target Audio Caption:

The gentle murmur of coffee shop chatter and the clinking of cups...

###### Target Joint Caption:

- <sub1> ... says, <S>I was hoping...<E>
- <sub2>... replies, <S>Me too. It's ...<E>

Video

Self-Attention Audio

Self-Attention

Text-Video

CrossAttention Text-Audio

CrossAttention

Video-Audio

CrossAttention Audio-Video

CrossAttention

...

... ...

N x Symmetric Conditional DiT Blocks

...

Video VAE Encoder

Audio VAE Encoder

t = M * N

Add Noise

Syn-RoPE Video:

... ...

Add Noise

Syn-RoPE Audio:

Dri Aud (For RA2V)

... Add

...

Src Vid (For RV2AV)

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

... ...

[Figure 82]

Structured Caption from MLLM

[Figure 83]

[Figure 84]

[Figure 85]

Target Vid

Target Aud

Ref Img 1

Ref Aud 1

Ref Img N

Ref Aud N

t = M * 1 t = M * N

t = 0 t = M * 1

t = 0

t = Lv

AudioVAE

Dncoder VideoVAE

Dncoder

Gen Audio

Gen Video

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

...

Add

|[Figure 95]|
|---|

|[Figure 96]|
|---|

Stage 3

Cross-pair Data for R2AVTasK

Stage 2

Stage 1

In-pair Data for R2AV TasK

Multi-Task Progressive Training

[Figure 97]

[Figure 98]

[Figure 99]

Target Vid

Target Aud

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Ref Img 1

Ref Aud 1

Ref Img 2

Ref Aud 2

|[Figure 105]|
|---|

|[Figure 106]|
|---|

[Figure 107]

[Figure 108]

Target Vid

Target Aud

Ref Img

Ref Aud

[Figure 109]

|[Figure 110]|
|---|

[Figure 111]

|[Figure 112]|
|---|

Dri Aud (For RA2V)

Src Vid (For RV2AV) Target Vid

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Target Aud

Ref Img 1

Ref Aud 1

Ref Img 2

Ref Aud 2

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

|[Figure 126]|
|---|

|[Figure 127]|
|---|

Omni-Task Fine-tuning

t = La*𝛾

RoPE Margin

RoPE Margin

RoPE Margin

RoPE Margin

[Figure 128]

[Figure 129]

[Figure 130]

Figure 2 Overview of DreamID-Omni framework. We integrate reference-based generation (R2AV), editing (RV2AV), and animation (RA2V) using a Symmetric Conditional DiT trained via a multi-task progressive training strategy. Structured Caption and Syn-RoPE ensure robust dual-level disentanglement in multi-person scenarios.

By selectively providing these conditions, our framework seamlessly transitions between three distinct tasks, as summarized in Table 1.

Task Input Output Goal

Human-Reference Audio-Video Generation (R2AV) T ,I,A Generate with references I,A. Human-Reference Video Editing (RV2AV) T ,I,A,Vsrc Edit identity and audio in Vsrc. Human-Reference Audio-Driven Video Animation (RA2V) T ,I,Adri Animate identity I using Adri.

- Table 1 Task Unification in DreamID-Omni. Our framework unifies R2AV, RV2AV and RA2V by toggling input conditions.

- 3.2 Framework

To address the diverse tasks defined in Section 3.1, we propose DreamID-Omni, a unified framework built upon a dual-stream DiT, as illustrated in Figure 2. The architecture consists of two parallel backbones: a video stream for visual synthesis and an audio stream for acoustic synthesis. These streams interact via bidirectional cross-attention layers, enabling fine-grained temporal synchronization and semantic alignment between the visual and auditory modalities.

###### 3.2.1 Symmetric Conditional DiT

A core architectural contribution of DreamID-Omni is the Symmetric Conditional DiT, designed to seamlessly integrate reference-based generation, editing, and animation within a unified framework. This is achieved through a symmetric dual-stream conditioning strategy that composes heterogeneous control signals in the latent space with structural parity. Let zv and za represent the noisy target video and target audio latents, respectively. To guide the denoising process, we construct two comprehensive conditional sequences, Xv and

Xa, which integrate both identity-specific and structural guidance: Xv = [zv;Ev(I)] + [Ev(Vsrc);0E

v(I)] (2) Xa = [za;Ea(A)] + [Ea(Adri);0E

a(A)] (3)

where [·;·] denotes concatenation along the sequence dimension, 0T represents a zero tensor with the same shape as T, and Ev,Ea are the respective VAE encoders. In this symmetric formulation, the reference features (Ev(I),Ea(A)) are concatenated to the noisy latents, allowing the DiT blocks to extract and disentangle high-level identity and timbre priors. Simultaneously, the structural conditions (Vsrc,Adri) are injected via element-wise addition, serving as a structural canvas that enforces spatial and temporal consistency. This dual-injection strategy effectively decouples the conditioning into identity-preservation and structural-guidance channels.

The inherent flexibility of this design enables seamless task switching. As detailed in Table 1, providing a null input for the structural conditions (Vsrc or Adri) effectively nullifies the additive term in Eqs. 2 or 3. Consequently, the model adaptively transitions between R2AV, RV2AV, and RA2V based on the available conditional modalities, maintaining a unified parameter set across all functional modes.

###### 3.2.2 Dual-Level Disentanglement

A critical challenge in multi-person generation is the confusion between subjects, which manifests in two forms: identity-timbre mismatch (e.g., subject A speaks with the voice of subject B) and attribute-content misattribution (e.g., subject A erroneously inheriting the visual attributes and dialogue of subject B). We posit that these failures stem from entanglement at two distinct levels. At the signal level, standard attention mechanisms fail to bind the visual features of an identity to its corresponding voice timbre. At the semantic level, unstructured text captions provide insufficient granularity to explicitly link specific subjects to their respective visual attributes, motions, and speech content. To address this, we propose a Dual-Level Disentanglement strategy. We introduce Syn-RoPE to enforce a rigid binding at the signal level, and a Structured Captioning scheme to resolve ambiguity at the semantic level.

Syn-RoPE. Recent works [27] have explored using Rotary Position Embedding [44] for spatial localization within video frames. However, such a spatially-grounded approach is incompatible with the more challenging task like R2AV, where character positions are synthesized dynamically by the model. To overcome this limitation, we propose Syn-RoPE, an identity-grounded mechanism that operates by assigning distinct, non-overlapping temporal positional segments to different semantic inputs within the model’s attention space. As illustrated in

- Figure 2, inspired by [36], we synchronize the video and audio streams by scaling the RoPE frequencies of the

target audio latents by a factor γ = Lv/La, where Lv and La denote the sequence lengths of the target video and audio latents, respectively. More crucially, Syn-RoPE partitions the absolute temporal positional index space into reserved “RoPE Margins” for the target sequence and each reference identity. Specifically, the target video and audio latents occupy the initial positional range [0,L − 1], where L denotes the maximum temporal length. We define a fixed margin M such that M ≫ L to serve as the base interval for each identity slot. Subsequently, the latent features of the k-th reference identity (both image Ik and audio Ak) are assigned to the k-th reserved segment, [k · M,(k + 1) · M − 1]. This strategy offers two fundamental advantages: (i) Inter-Identity Decoupling: By leveraging the periodicity of RoPE, each identity’s features are projected into a distinct rotational subspace, naturally suppressing cross-identity attention scores and preventing feature entanglement. (ii) Intra-Identity Synchronization: By mapping the visual and acoustic features of the same identity to identical positional segments, we achieve a robust, implicit cross-modal synchronization at the signal level. This design provides a unified mechanism for robust identity binding across all generation, editing, and animation tasks.

Structured Caption. At the semantic level, ambiguity in multi-subject scenarios typically arises when standard prompts fail to explicitly associate visual attributes, motions, and speech content with specific individuals. To resolve this, we introduce a Structured Captioning scheme that establishes an unambiguous mapping between each reference identity Ik and a unique anchor token, denoted as ⟨subk⟩. The process begins by generating a fine-grained attribute description for each identity to initialize the anchor tokens. Building upon this foundation, the target video content is synthesized into a comprehensive “script” partitioned into distinct

semantic fields: video caption, audio caption, and joint caption. Crucially, all references to individuals across these fields consistently utilize the predefined anchor tokens ⟨subk⟩. This format provides the model with an explicit grounding that resolves semantic-level entanglement, which is critical for the success of all three core tasks.

#### 3.3 Multi-Task Progressive Training

Training a unified model for R2AV, RV2AV, and RA2V presents a complex optimization challenge. A naive joint training approach often suffers from conflicting learning objectives, where the generative objective of creating diverse content can interfere with the fidelity objective of adhering to strong conditional constraints. To circumvent this, we introduce a Multi-Task Progressive Training Strategy, a three-stage curriculum designed to incrementally build model capabilities, ensuring stable convergence and synergistic learning.

In-pair Reconstruction. The initial stage aims to establish a robust generative prior for controllable generation. We train the model exclusively on the R2AV task, using an in-pair reconstruction objective. For each training sample Y , we extract the reference identity I and the reference timbre A from the sample itself. The model is then tasked with reconstructing the full data stream conditioned on these internal references and the text prompt T . To prevent the model from trivially copying the reference segments and to encourage true conditional synthesis, we introduce a masked reconstruction loss. Let Mv and Ma be binary masks identifying the spatio-temporal regions of I and A within the ground truth latents. The loss is computed only on the unmasked regions, forcing the model to generate, rather than merely copy, the content corresponding to the references. The objective is defined as:

Linpair = Ez,t,C λv∥(1 − Mv) ⊙ (ϵv − ϵˆθ(zv,t,t,C))∥22

(4)

+ λa∥(1 − Ma) ⊙ (ϵa − ϵˆθ(za,t,t,C))∥22

where the conditioning set for this stage is C = {I,A,T }, ϵ is the ground truth noise, ϵˆθ is the model’s prediction, and ⊙ denotes element-wise multiplication.

Cross-pair Disentanglement. To enhance the model’s generalization capabilities and force it to learn a truly disentangled representation of identity and timbre, we advance to a cross-pair training stage. In this phase, the reference identity I and timbre A are sourced from a different video clip than the target video-audio stream Y . This more challenging objective compels the model to synthesize content based on abstract identity and timbre concepts, rather than relying on low-level correlations present in the source. The training objective for this stage, Lcross, reuses the same formulation as Linpair (Eq. 4). However, a key distinction is that the masks are nullified by setting Mv = 0 and Ma = 0. This modification ensures the loss is computed over the entire data stream, pushing the model towards a more robust disentanglement.

Omni-Task Fine-tuning. The final stage unifies all tasks by fine-tuning the model on a mixed dataset comprising R2AV, RV2AV, and RA2V samples. RV2AV samples are constructed by providing a masked version of the target video as structural context (Vsrc), while RA2V samples supply the target audio as the driving signal (Adri). By training on this composite dataset, the model learns to seamlessly switch between generation, editing, and animation based on the provided conditions, as formulated in Eq. 1. This progressive, three-stage curriculum is crucial. We observe that by first mastering the weakly-constrained R2AV task, the model develops a powerful and diverse generative prior. This prior then serves as a robust foundation for the strongly-constrained RV2AV and RA2V tasks, allowing the model to learn high-fidelity conditional control without sacrificing generative quality, leading to a truly unified and capable omni-purpose model.

#### 3.4 Inference Pipeline

At inference time, we employ a multi-condition Classifier-Free Guidance (CFG) [20] strategy, which is applied independently to the video and audio streams, but follows the same unified formulation:

ϵˆfinal = ϵˆθ(zt,∅,∅) + wT · (ˆϵθ(zt,T ,∅) − ϵˆθ(zt,∅,∅)) + wS · (ˆϵθ(zt,T ,S) − ϵˆθ(zt,T ,∅))

(5)

|[Figure 131]|
|---|

[Figure 132]

... rainy day, <sub1> hold a cat, Said sorrowfully: “I will miss you forever, Everyday and every moment of my life...”

- (a)
- (b)

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

<sub1>

[Figure 137]

[Figure 138]

[Figure 139]

Wan-2.6

###### Ours

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Phantom

（Not suppot Audio Gen）

[Figure 145]

[Figure 146]

[Figure 147]

Qwen-Image + Ovi

（Not suppot Voice ref）

[Figure 148]

[Figure 149]

[Figure 150]

VACE

[Figure 151]

（Not suppot Audio Gen）

[Figure 152]

Qwen-Image + LTX-2

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Hunyuan Custom

（Not suppot Voice ref）

（Not suppot Audio Gen）

[Figure 158]

|[Figure 159]|
|---|

|[Figure 160]|
|---|

[Figure 161]

[Figure 162]

In the coffee shop on the street-side... <sub1> is on the left ... Says: “Simplicity is the ultimate sophistication.”, <sub2> is on the right, wearing a light-colored polo shirt, replies: “I completely agree, it's about the core idea.”

[Figure 163]

[Figure 164]

[Figure 165]

<sub1> <sub2>

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

Wan-2.6

###### Ours

| | |
|---|---|

| |
|---|

| |[Figure 176]|
|---|---|

| |
|---|

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Qwen-Image + Ovi

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Phantom

（Not suppot Audio Gen）

（Not suppot Voice ref）

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Qwen-Image + LTX-2

VACE

（Not suppot Audio Gen）

（Not suppot Voice ref）

[Figure 195]

- Figure 3 Qualitative comparison with state-of-the-art (SOTA) methods on R2AV. Please zoom in for more details.

Method

Support Video Audio Audio-Visual Consistency

Video Audio AES ↑ ViCLIP ↑ ID-Sim. (S/M) ↑ PQ ↑ CLAP ↑ WER ↓ T-Sim. (S/M) ↑ Sync-C ↑ Sync-D ↓ Spk-Conf. ↓

Phantom ✓ × 0.604 13.791 0.657/0.572 - - - - - - VACE ✓ × 0.613 11.091 0.664/0.395 - - - - - - HunyuanCustom ✓ × 0.589 12.159 0.659/- - - - - - - -

Qwen-Image + LTX-2 ✓ ✓ 0.611 8.548 0.571/0.349 6.247 0.144 0.093 - 3.706 10.003 0.340 Qwen-Image + Ovi ✓ ✓ 0.606 8.974 0.459/0.336 5.826 0.203 0.097 - 5.857 8.407 0.380 Wan2.6 ✓ ✓ 0.632 13.410 0.523/0.455 6.391 0.236 0.534 0.391/0.217 6.026 8.352 0.380 Ours ✓ ✓ 0.618 13.911 0.674/0.603 6.290 0.278 0.052 0.493/0.402 6.226 7.791 0.080

Table 2 Quantitative comparison of R2AV on our proposed benchmark. Best results are in bold, second best are underlined. The S/M notation in ID-Sim. and T-Sim. refers to results on single-person and multi-person scenarios, respectively.

where ϵˆθ(zt,T ,S) is the model’s prediction under text condition T and a stream-specific condition S. For the video stream, S = I , while for the audio stream, S = A. The terms wT and wS are their respective guidance scales. This chained application ensures that identity and timbre guidance operates on a text-aligned basis, leading to more stable and coherent results. The MLLM system prompt for Structured Caption is shown in Fig. 8.

- 4 Experiments

Method AES ↑ ViCLIP ↑ ID-Sim. ↑ WER ↓ T-Sim. ↑ Sync-C ↑

VACE 0.560 14.353 0.565 - - HunyuanCustom 0.538 14.576 0.590 - - -

Ours 0.584 14.832 0.635 0.065 0.513 6.241

Table 3 Comparison with SOTA methods on RV2AV.

Method AES ↑ ViCLIP ↑ ID-Sim. ↑ Sync-C ↑ Sync-D ↓

Humo 0.550 14.859 0.609 6.114 8.323 HunyuanCustom 0.567 13.027 0.611 5.786 9.071 Ours 0.591 16.618 0.623 6.325 8.659

Table 4 Comparison with SOTA methods on RA2V.

#### 4.1 Setup

IDBench-Omni. We introduce IDBench-Omni, a new comprehensive benchmark for controllable human-centric audio-video generation. The benchmark comprises three specialized test sets, totaling 200 high-quality data instances, designed to evaluate a model’s omni-purpose capabilities: (1) 100 identity-timbre-caption triplets for evaluating generation task; (2) 50 masked videos with target identity and timbre for evaluating controlled video editing; and (3) 50 driving audios with reference identities for evaluating audio-driven animation. These sets cover a diverse range of challenging scenarios, including complex multi-person dialogues, significant variations in identity and timbre, and in-the-wild recording conditions. IDBench-Omni provides a rigorous and holistic platform for evaluating the generation, editing, and animation capabilities of unified audio-video models.

Implementation Details. We initialize our model from Ovi [36] and train on audio-video data from [28] (construction details in Sec. A.2). During training, we set the learning rate to 1.0 × 10−5, with a global batch size of 32 and Rope Margin M = 150. The training curriculum begins with the In-pair Reconstruction for 10,000 steps, followed by the Cross-pair Disentanglement and Omni-Task Fine-tuning stages, which involve 20,000 iterations each. In the final Omni-Task stage, we sample R2AV, RV2AV, and RA2V data with a ratio of 4:3:3.

Evaluation Metrics. We evaluate our model across three key dimensions. For video, we assess fidelity and coherence through the aesthetics score (AES) from VBench [24] for video quality, the text-video similarity from ViCLIP [53] for text following, and ArcFace [9] for Identity Similarity (ID-Sim.). For audio, we evaluate its quality and fidelity from multiple aspects. We gauge audio quality via the Production Quality (PQ) score from AudioBox-Aesthetics [46] and semantic consistency using CLAP [56]. Additionally, we compute the Word Error Rate (WER) by transcribing the generated audio with Whisper-large-v3 [41] and comparing it against the ground-truth transcript, while Timbre Similarity (T-Sim.) is determined by the cosine similarity of speaker embeddings from WavLM [3]. For audio-visual consistency, we focus on synchronization and attribution. Lip-sync accuracy relies on the standard confidence (Sync-C) and distance (Sync-D) scores from SyncNet [8]. Finally, Speaker Confusion (Spk-Conf.), a critical metric for multi-person dialogues, is evaluated by the Gemini-2.5-Pro model [45], a detailed system prompt provided in Sec. A.3.

#### 4.2 Comparison

Comparison on R2AV. As there are no open-source methods that directly support the R2AV task, we establish a set of strong baselines for comparison. We compare our method with the closed-source model Wan2.6 [47] and two cascaded pipelines constructed by first generating an initial frame with Qwen-Image [55] and then animating it with LTX-2 [15] and Ovi [36]. Additionally, for video-centric metrics, we include leading R2V models: Phantom [35], VACE [25], and HunyuanCustom [21]. As demonstrated in Table 2, our method achieves superior or comparable results across the video, audio, and audio-visual consistency dimensions. For qualitative comparison in Fig. 3, in case (a), our model delivers the most realistic visual results compared to baselines such as Wan2.6, while exhibiting superior identity consistency with the reference identities relative to Ovi and LTX-2. In case (b), only ours successfully achieves correct binding between specific identities and their corresponding timbres, whereas baselines like Wan2.6 suffer from identity-timbre mismatch. See Sec. A.4 for user study details.

Comparison on RV2AV. We compare our method with SOTA video editing methods, VACE [25] and HunyuanCustom [21] on RV2AV. The quantitative results are presented in Table 3. Since the compared methods do not support audio generation, audio-related metrics are reported exclusively for our model. The results demonstrate that our method not only achieves SOTA performance on video-centric metrics (AES, ViCLIP,

|[Figure 196]|
|---|

|[Figure 197]|
|---|

... <sub1> wearing thick-rimmed glasses, said: “since ... merger with BMC.”

...<sub1> wearing a dark red jacket ... said to... “... I do need you …”

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

|[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]<br><br>[Figure 206]<br><br>[Figure 207]|
|---|

|[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]<br><br>[Figure 211]<br><br>[Figure 212]<br><br>[Figure 213]|
|---|

Source Video

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

Ours

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

VACE

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

Hunyuan Custom

(a) (b)

###### Figure 4 Qualitative comparison with SOTA methods on RV2AV. Please zoom in for more details.

|[Figure 232]|
|---|

|[Figure 233]|
|---|

...room... green walls, <sub1> said to...: “... I do...that... ”

...dark..., man behind, <sub1>... said: “... How about...be dead? ”

[Figure 234]

[Figure 235]

Driving Audio

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

###### Ours

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

HuMo

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

Hunyuan Custom

(a) (b)

Figure 5 Qualitative comparison with SOTA methods on RA2V. Please zoom in for more details.

and ID-Sim.), but also exhibits excellent audio generating capabilities, as evidenced by the strong WER, T-Sim., and Sync-C scores. Qualitative results are illustrated in Fig. 4. In case (a), our model delivers higher identity similarity and superior visual quality; in case (b), it demonstrates improved text-following capabilities compared to the baselines.

Comparison on RA2V. For the RA2V task, we compare our method with Humo [2] and HunyuanCustom [21]. As shown in Table 4, our method achieves comparable lip-sync accuracy to Humo and leading performance on video-related metrics. Qualitative comparisons are provided in Fig. 5. Notably, in scenarios involving multiple subjects, both Humo and HunyuanCustom frequently exhibit speaker misattribution errors. In contrast, our model animates the correct subject by precisely following the structured captions.

#### 4.3 Ablation Studies

Ablation on Dual-level Disentanglement. To validate the effectiveness of our dual-level disentanglement design, we conduct an ablation study on the challenging multi-person dialogue scenario of the R2AV task. The

<sub1>

<sub1> <sub2>

|[Figure 254]|
|---|

[Figure 255]

... <sub1> in the left, wearing wine-colored ... said: “We shold... ”, <sub2> in the right, wearing grey suit ... black sweater, replies:”absolutely...”

|[Figure 256]|
|---|

|[Figure 257]|
|---|

...a medium shot, <sub1>... holds a cappuccino , said: “... perfect cappuccino... ”

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

Only IR

w/o SC

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Only CD

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

w/o Syn-RoPE

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

MT (w/o OFT)

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

Ours

[Figure 307]

[Figure 308]

Ours

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

(b) (a)

Figure 6 Qualitative results of our ablation studies. (a) Ablation on our dual-level disentanglement design. (b) Ablation on multi-task progressive training.

Method ViCLIP ↑ T-Sim. ↑ Sync-C ↑ Sync-D ↓ Spk-Conf. ↓

w/o Syn-RoPE 13.179 0.211 4.192 10.411 0.12 w/o SC 11.381 0.378 5.943 8.064 0.26

Ours 13.613 0.402 6.074 8.027 0.08

Table 5 Ablation study on Dual-level Disentanglement.

Method ViCLIP ↑ ID-Sim.↑ AQ ↑ T-Sim. ↑ CLAP ↑

Only IR 11.931 0.692 5.576 0.504 0.225 Only CD 14.044 0.543 6.072 0.471 0.287 MT (w/o OFT) 9.518 0.638 4.449 0.442 0.104 Ours 14.573 0.674 6.313 0.493 0.282

Table 6 Ablation study on Multi-Task Progressive Training.

quantitative results are presented in Table 5, with a qualitative comparison in Fig. 6 (a). Our analysis highlights the distinct contributions of each component: (1)w/o SC: Following Ovi [36], we replace the Structured Caption (SC) with standard unstructured joint caption (i.e., a single global description for both target video and audio), the model’s ability to follow textual instructions is significantly impaired, resulting in the lowest ViCLIP score. More critically, this leads to a dramatic increase in speaker confusion, with the Spk-Conf. rate more than tripling from 0.08 to 0.26. This underscores the crucial role of SC in explicitly associating visual attributes and dialogue content with specific subjects in multi-person scenarios. As illustrated in the first row of Fig. 6 (a), without SC, both the visual attributes and content of ⟨sub1⟩ and ⟨sub2⟩ suffer from severe mismatch. (2)w/o Syn-RoPE: Removing Syn-RoPE, which is designed to bind specific speakers to their corresponding timbres, leads to a severe degradation in timbre preservation, as indicated by the sharp drop in the T-Sim. score. The identity-timbre mismatch also negatively impacts lip-sync accuracy (Sync-C/D). As shown in the second row of Fig. 6 (a), without Syn-RoPE, ⟨sub1⟩ is erroneously bound to the voice timbre of ⟨sub2⟩.

Ablation on Multi-Task Progressive Training. We conduct an ablation study on our multi-task progressive training strategy, with the results on the single-person R2AV scenario presented in Table 6. A qualitative comparison is provided in Fig. 6 (b) Only IR: Training exclusively with In-pair Reconstruction (IR) leads to severe copy-paste issues, as illustrated in Fig. 6 (b) (the first row). While this results in deceptively high ID-Sim. and T-Sim. scores, the model fails to learn meaningful conditional synthesis, leading to poor text-following ability (ViCLIP) and audio quality (AQ). (2) Only CD: Conversely, training only with Cross-pair Disentanglement (CD) from the start proves too challenging. The model struggles to learn fundamental representations, resulting in very low ID-Sim. and T-Sim. scores, as shown in the second row in Fig. 6 (b). (3) MT (w/o OFT): This experiment validates our progressive training philosophy by attempting to train all tasks (R2AV, RV2AV, RA2V) jointly from scratch without Omni-Task Fine-tuning (OFT). This naive multi-task (MT) approach yields suboptimal performance on the R2AV task, particularly in text-following as indicated by ViCLIP (third row of Fig. 6 (b)). This confirms our hypothesis that when training a unified model, it is crucial to first establish a strong generative prior on weakly-constrained tasks (like R2AV) before introducing strongly-constrained tasks (like RV2AV/RA2V). Without this progression, the model tends to “shortcut” the

learning process by overfitting to the easier, strongly-constrained tasks, ultimately failing to generalize on the more complex, weakly-constrained generation tasks.

### 5 Conclusion

In this paper, we have presented DreamID-Omni, a unified framework for controllable human-centric audiovideo generation. By integrating reference-based generation, editing, and animation into a single paradigm, DreamID-Omni addresses the limitations of previous task-specific models. To tackle the critical challenge of multi-person confusion, we introduced Syn-RoPE for signal-level identity-timbre binding and Structured Captioning for semantic-level disentanglement. Furthermore, our proposed Multi-Task Progressive Training strategy effectively harmonizes disparate objectives. Extensive experiments on our new benchmark, IDBenchOmni, demonstrate that DreamID-Omni achieves SOTA performance across multiple tasks.

### References

- [1] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.

- [2] Liyang Chen, Tianxiang Ma, Jiawei Liu, Bingchuan Li, Zhuowei Chen, Lijie Liu, Xu He, Gen Li, Qian He, and Zhiyong Wu. Humo: Human-centric video generation via collaborative multi-modal conditioning. arXiv preprint arXiv:2509.08519, 2025.

- [3] Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, and Furu Wei. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. 2021.
- [4] Siyan Chen, Yanfei Chen, Ying Chen, Zhuo Chen, Feng Cheng, Xuyan Chi, Jian Cong, Qinpeng Cui, Qide Dong, Junliang Fan, et al. Seedance 1.5 pro: A native audio-visual joint generation foundation model. arXiv preprint arXiv:2512.13507, 2025.

- [5] Xu Chen, Keke He, Junwei Zhu, Yanhao Ge, Wei Li, and Chengjie Wang. Hifivfs: High fidelity video face swapping. arXiv preprint arXiv:2411.18293, 2024.

- [6] Zhuowei Chen, Bingchuan Li, Tianxiang Ma, Lijie Liu, Mingcong Liu, Yi Zhang, Gen Li, Xinghui Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom-data: Towards a general subject-consistent video generation dataset. arXiv preprint arXiv:2506.18851, 2025.

- [7] Gang Cheng, Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Ju Li, Dechao Meng, Jinwei Qi, Penchong Qiao, et al. Wan-animate: Unified character animation and replacement with holistic replication. arXiv preprint arXiv:2509.14055, 2025.

- [8] Joon Son Chung and Andrew Zisserman. Out of time: automated lip sync in the wild. In Asian conference on computer vision, pages 251–263. Springer, 2016.

- [9] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

- [10] Yufan Deng, Xun Guo, Yuanyang Yin, Jacob Zhiyuan Fang, Yiding Yang, Yizhi Wang, Shenghai Yuan, Angtian Wang, Bo Liu, Haibin Huang, et al. Magref: Masked guidance for any-reference video generation. arXiv preprint arXiv:2505.23742, 2025.

- [11] Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, et al. Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. arXiv preprint arXiv:2407.05407, 2024.

- [12] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.

- [13] Junmin Gong, Sean Zhao, Sen Wang, Shengyuan Xu, and Joe Guo. Ace-step: A step towards music generation foundation model. arXiv preprint arXiv:2506.00045, 2025.

- [14] Xu Guo, Fulong Ye, Xinghui Li, Pengqi Tu, Pengze Zhang, Qichao Sun, Songtao Zhao, Xiangwang Hou, and Qian He. Dreamid-v:bridging the image-to-video gap for high-fidelity face swapping via diffusion transformer, 2026.
- [15] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, et al. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2026.

- [16] Jiangyu Han, Federico Landini, Johan Rohdin, Anna Silnova, Mireia Diez, and Lukáš Burget. Leveraging self-supervised learning for speaker diarization. In Proc. ICASSP, 2025.

- [17] Akio Hayakawa, Masato Ishii, Takashi Shibuya, and Yuki Mitsufuji. Mmdisco: Multi-modal discriminator-guided cooperative diffusion for joint audio and video generation. arXiv preprint arXiv:2405.17842, 2024.

- [18] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024.

- [19] Xuanhua He, Quande Liu, Zixuan Ye, Weicai Ye, Qiulin Wang, Xintao Wang, Qifeng Chen, Pengfei Wan, Di Zhang, and Kun Gai. Fulldit2: Efficient in-context conditioning for video diffusion transformers. arXiv preprint arXiv:2506.04213, 2025.

- [20] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

- [21] Teng Hu, Zhentao Yu, Zhengguang Zhou, Sen Liang, Yuan Zhou, Qin Lin, and Qinglin Lu. Hunyuancustom: A multimodal-driven architecture for customized video generation. arXiv preprint arXiv:2505.04512, 2025.

- [22] Xiaohu Huang, Hao Zhou, Qiangpeng Yang, Shilei Wen, and Kai Han. Jova: Unified multimodal learning for joint video-audio generation. arXiv preprint, 2025.

- [23] Yuzhou Huang, Ziyang Yuan, Quande Liu, Qiulin Wang, Xintao Wang, Ruimao Zhang, Pengfei Wan, Di Zhang, and Kun Gai. Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning. arXiv preprint arXiv:2501.04698, 2025.

- [24] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

- [25] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025.

- [26] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [27] Zhe Kong, Feng Gao, Yong Zhang, Zhuoliang Kang, Xiaoming Wei, Xunliang Cai, Guanying Chen, and Wenhan Luo. Let them talk: Audio-driven multi-person conversational video generation. arXiv preprint arXiv:2505.22647, 2025.

- [28] Hui Li, Mingwang Xu, Yun Zhan, Shan Mu, Jiaye Li, Kaihui Cheng, Yuxuan Chen, Tan Chen, Mao Ye, Jingdong Wang, et al. Openhumanvid: A large-scale high-quality dataset for enhancing human-centric video generation. arXiv preprint arXiv:2412.00115, 2024.

- [29] Zhaoyang Li, Dongjun Qian, Kai Su, Qishuai Diao, Xiangyang Xia, Chang Liu, Wenfei Yang, Tianzhu Zhang, and Zehuan Yuan. Bindweave: Subject-consistent video generation via cross-modal integration. arXiv preprint arXiv:2510.00438, 2025.

- [30] Sen Liang, Zhentao Yu, Zhengguang Zhou, Teng Hu, Hongmei Wang, Yi Chen, Qin Lin, Yuan Zhou, Xin Li, Qinglin Lu, et al. Omniv2v: Versatile video generation and editing via dynamic content manipulation. arXiv preprint arXiv:2506.01801, 2025.

- [31] Gaojie Lin, Jianwen Jiang, Jiaqi Yang, Zerong Zheng, Chao Liang, Yuan Zhang, and Jingtuo Liu. Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13847–13858, 2025.

- [32] Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley. AudioLDM: Text-to-audio generation with latent diffusion models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 21450–21474. PMLR, 23–29 Jul 2023.

- [33] Haohe Liu, Gael Le Lan, Xinhao Mei, Zhaoheng Ni, Anurag Kumar, Varun Nagaraja, Wenwu Wang, Mark D Plumbley, Yangyang Shi, and Vikas Chandra. Syncflow: Toward temporally aligned joint audio-video generation from text. arXiv preprint arXiv:2412.15220, 2024.

- [34] Kai Liu, Wei Li, Lai Chen, Shengqiong Wu, Yanhao Zheng, Jiayi Ji, Fan Zhou, Rongxin Jiang, Jiebo Luo, Hao Fei, and Tat-Seng Chua. Javisdit: Joint audio-video diffusion transformer with hierarchical spatio-temporal prior synchronization. In arxiv, 2025.

- [35] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. arXiv preprint arXiv:2502.11079, 2025.

- [36] Chetwin Low, Weimin Wang, and Calder Katyal. Ovi: Twin backbone cross-modal fusion for audio-video generation. arXiv preprint arXiv:2510.01284, 2025.

- [37] Xiangyang Luo, Ye Zhu, Yunfei Liu, Lijian Lin, Cong Wan, Zijian Cai, Shao-Lun Huang, and Yu Li. Canonswap: High-fidelity and consistent video face swapping via canonical space modulation. arXiv preprint arXiv:2507.02691, 2025.

- [38] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.

- [39] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

- [40] Leigang Qu, Feng Cheng, Ziyan Yang, Qi Zhao, Shanchuan Lin, Yichun Shi, Yicong Li, Wenjie Wang, Tat-Seng Chua, and Lu Jiang. Vincie: Unlocking in-context image editing from video. arXiv preprint arXiv:2506.10941, 2025.

- [41] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023.

- [42] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. Mm-diffusion: Learning multi-modal diffusion models for joint audio and video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10219–10228, 2023.

- [43] Hao Shao, Shulun Wang, Yang Zhou, Guanglu Song, Dailan He, Zhuofan Zong, Shuo Qin, Yu Liu, and Hongsheng Li. Vividface: A robost and high-fidelity video face swapping framework. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

- [44] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

- [45] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- [46] Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hoffman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, Carleigh Wood, Ann Lee, and Wei-Ning Hsu. Meta audiobox aesthetics: Unified automatic quality assessment for speech, music, and sound. 2025.
- [47] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [48] Duomin Wang, Wei Zuo, Aojie Li, Ling-Hao Chen, Xinyao Liao, Deyu Zhou, Zixin Yin, Xili Dai, Daxin Jiang, and Gang Yu. Universe-1: Unified audio-video generation via stitching of experts. arXiv preprint arXiv:2509.06155, 2025.

- [49] Jun Wang, Chunyu Qiang, Yuxin Guo, Yiran Wang, Xijuan Zeng, Chen Zhang, and Pengfei Wan. Klear: Unified multi-task audio-video joint generation. arXiv preprint arXiv:2601.04151, 2026.

- [50] Kai Wang, Shijian Deng, Jing Shi, Dimitrios Hatzinakos, and Yapeng Tian. Av-dit: Efficient audio-visual diffusion transformer for joint audio and video generation. arXiv preprint arXiv:2406.07686, 2024.

- [51] Mengchao Wang, Qiang Wang, Fan Jiang, Yaqi Fan, Yunpeng Zhang, Yonggang Qi, Kun Zhao, and Mu Xu. Fantasytalking: Realistic talking portrait generation via coherent motion synthesis. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 9891–9900, 2025.

- [52] Runqi Wang, Yang Chen, Sijie Xu, Tianyao He, Wei Zhu, Dejia Song, Nemo Chen, Xu Tang, and Yao Hu. Dynamicface: High-quality and consistent face swapping for image and video using composable 3d facial priors. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13438–13447, 2025.

- [53] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinyuan Chen, Yaohui Wang, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Yu Qiao. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023.

- [54] Huawei Wei, Zejun Yang, and Zhisheng Wang. Aniportrait: Audio-driven synthesis of photorealistic portrait animations, 2024.

- [55] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

- [56] Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP, 2023.

- [57] Mingwang Xu, Hui Li, Qingkun Su, Hanlin Shang, Liwei Zhang, Ce Liu, Jingdong Wang, Yao Yao, and Siyu zhu. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation, 2024.
- [58] Zhengbo Xu, Jie Ma, Ziheng Wang, Zhan Peng, Jun Liang, and Jing Li. End-to-end video character replacement without structural guidance. arXiv preprint arXiv:2601.08587, 2026.

- [59] Tao Yang, Ruibin Li, Yangming Shi, Yuqi Zhang, Qide Dong, Haoran Cheng, Weiguo Feng, Shilei Wen, Bingyue Peng, and Lei Zhang. Many-for-many: Unify the training of multiple video and image generation and manipulation tasks. arXiv preprint arXiv:2506.01758, 2025.

- [60] Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. Effective whole-body pose estimation with two-stages distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4210–4220, 2023.

- [61] Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, and Wenhan Luo. Unic: Unified in-context video editing. arXiv preprint arXiv:2506.04216, 2025.

- [62] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identitypreserving text-to-video generation by frequency decomposition. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12978–12988, 2025.

- [63] Lei Zhao, Linfeng Feng, Dongxu Ge, Rujin Chen, Fangqiu Yi, Chi Zhang, Xiao-Lei Zhang, and Xuelong Li. Uniform: A unified multi-task diffusion transformer for audio-video generation. arXiv preprint arXiv:2502.03897, 2025.

- [64] Shengkui Zhao, Zexu Pan, and Bin Ma. Clearervoice-studio: Bridging advanced speech processing research and practical deployment. arXiv preprint arXiv:2506.19398, 2025.

- [65] Yong Zhong, Zhuoyi Yang, Jiayan Teng, Xiaotao Gu, and Chongxuan Li. Concat-id: Towards universal identitypreserving video synthesis. arXiv preprint arXiv:2503.14151, 2025.

### A Appendix

In the supplementary material, the sections are organized as follows:

- • We provide qualitative comparisons with baselines on RV2AV and RA2V in Sec. A.1.
- • We provide the details of our data construction pipeline in Sec. A.2.
- • We provide the MLLM-based judge prompt in Sec. A.3.
- • We provide more details regarding the user study in Sec. A.4.
- • We provide more qualitative results for R2AV, RV2AV, and RA2V tasks in Sec. A.5.

#### A.1 Comparison Results on RV2AV and RA2V

Figs. 4 and 5 compare DreamID-Omni with SOTA methods on RV2AV and RA2V, respectively, where the qualitative results clearly demonstrate our superior performance.

#### A.2 Data Construction Details

Our full dataset consists of approximately 1M high-quality audio-video pairs. As illustrated in Fig. 7, our data construction pipeline is categorized into two primary stages: In-pair data construction. We process each video clip to extract its internal references. The reference voice timbre set A is created by applying DiariZen [16] for speaker diarization to obtain precise timestamps. Concurrently, the reference identity set I is formed by using DWPose [60] to detect and crop face regions from keyframes. Cross-pair data construction. For the audio branch, we construct the reference timbre A through a multi-stage pipeline. First, DiariZen [16] and Gemini [45] are combined to accurately label speaker segments in multi-person dialogues. Subsequently, CosyVoice [11] is employed to clone a clean voice for each speaker, which is then purified using ClearerVoice [64] for final denoising. For the video branch, the reference identity I is constructed following the established Phantom-Data [6] pipeline.

##### A.3 MLLM-Based Judge We employ Gemini-2.5-Pro as a MLLM-based judge for Speaker Confusion, Fig. 9 presents the system prompt.

|[Figure 313]<br><br>[Figure 314]<br><br>[Figure 315]<br><br>[Figure 316]<br><br>[Figure 317]<br><br>[Figure 318]<br><br>face embedding<br><br>detec crop<br><br>arcface data base<br><br>search Remove Duplicates<br><br>[Figure 319]<br><br>[Figure 320]<br><br>|[Figure 321]|
|---|
<br><br>|[Figure 322]|
|---|
<br><br>[Figure 323]<br><br>[Figure 324]<br><br><sub1> <sub2><br><br>DiariZen Coyvice Clearvice<br><br>[Figure 325]<br><br>[Figure 326]<br><br>[Figure 327]<br><br>Cross-pair data construction|
|---|

[Figure 328]

###### Structured Caption

|[Figure 329]<br><br>[Figure 330]<br><br>[Figure 331]<br><br>[Figure 332]<br><br>[Figure 333]<br><br>detec crop<br><br>[Figure 334]<br><br>DiariZen<br><br>[Figure 335]<br><br>[Figure 336]<br><br>[Figure 337]<br><br>[Figure 338]<br><br>augumentation<br><br>|[Figure 339]|
|---|
<br><br><sub1><br><br>[Figure 340]<br><br>In-pair data construction|
|---|

###### Ref Image Caption:

- <img1>: A young man with dark hair smiling, identified as <sub1>.
- <img2>: A young woman with light brown hair, identified as <sub2>.

###### Target Video Caption:

Overall Environment/Scene: A cozy, warmly lit living room in the evening. The shot is an upper-body close-up of both subjects, who are sitting on a couch. Main Characters/Subjects Appearance: <sub1> is on the left, wearing a casual grey sweater. <sub2> is on the right, in a cream-colored knit top. Main Characters/Subjects Actions: <sub1> looks at <sub2> and speaks with an animated expression. <sub2> listens intently with a gentle smile, then leans in slightly to reply.

###### Target Audio Caption:

A quiet, comfortable room ambiance. <sub1> speaks in an upbeat, friendly tone. <sub2> has a warm and thoughtful voice.

###### Target Joint Caption:

<sub1> smiles and asks, <S>Are you ready for the movie night?<E> <sub2> nods enthusiastically and replies, <S>Yes! I have the popcorn all ready.<E>

Figure 7 Data construction pipeline.

|Role: You are specialized in the deep understanding and joint analysis of input image and user prompt. Your core responsibility is to act as a professional prompt engineering, merging complex visual and user prompt into a coherent, insightful caption for joint video-audio generation.<br><br>Input Data:<br><br>1. Reference Images: One or more images, typically containing one or more persons.<br>2. User Prompt: The text input by the user serves as a prompt for the joint generation of audio and video, including basic visual descriptions, and may also includes audio descriptions and speech content descriptions,The content of the speech is marked in double quotation marks.<br><br><br>Task Definition: Your core objective is to generate a structured output containing the following four parts:<br><br>1. Ref Image Caption:<br><br>* Generate a concise description for each reference image Within 20 words in English. Focus on identifying and describing the main persons or subjects in the images<br>* Use a special tagging format to identify images and subjects, like `<imgN>` and `<subN>`, where `N` is a sequential number starting from 1.<br><br><br>2. Target Video Caption:<br><br>* Optimize based on the provided Reference Images and User Prompt.<br>* The primary task is to identify and replace all descriptive terms in the "user prompt" that refer to subjects matching those in the reference images with the corresponding special subject tag (e.g., `<sub1>`).<br>* Utilize your understanding of the reference images and user prompt to refine the visual description part of the "user prompt"**, ensuring its accuracy, richness of detail, and relevance to the reference images.<br>*. Please strictly follow this description order: Overall Environment/Scene, Main Character Appearance, Main Character Actions<br><br><br>3. Target Audio Caption:<br><br>* Optimize based on the provided Reference Images and User Prompt.<br>* Auditory Description (Non-Speech Section): If no Auditory Description is provided in the user prompt, then create a reasonable concise audio description and character emotions, voice descriptions based on the Target Video Caption.<br><br><br>4. Target Joint Caption:<br><br><br>* Optimize based on the provided Reference Images and User Prompt.<br>* Please strictly follow this description order: Main Character Actions and Speech content descriptions<br>* Speech content descriptions: please use the format `<S>pure speech content<E>`. Don't delete or modify the the content of the speech in user prompt.<br><br><br>Example output prompt: "Ref Image Caption: <img1>: A man with dark hair..., identified as <sub1>. <img2>: A smiling woman with dark hair..., identified as <sub2>. Target Video Caption: A coffee shop with warm ... ,<sub1> is on the left, wearing dark... <sub2> is on the right, wearing a simple...,<sub1> looks at ... <sub2> smiling... 、 Target Audio Caption: The gentle murmur of coffee shop chatter and the clinking of cups... Target Joint Caption: <sub1> ... says, <S>I was hoping...<E> <sub2>... replies, <S>Me too. It's ...<E>" """|
|---|

Figure 8 MLLM system prompt for Structured Caption.

#### A.4 User Study

We conduct a user study as part of the evaluation on IDBench-Omni. Specifically, we invited 30 professional video creators to serve as evaluators. Users rate each video on seven dimensions on a 1–5 scale, and we average the ratings to obtain the final scores. The user study was carried out in a blinded setting. Table 7 indicates that our approach performs strongly across multiple dimensions.

|Role: You are a professional forensic video analyst specializing in multi-modal human-centric synthesis. Your task is to detect Speaker Confusion in generated videos where multiple identities and audio streams are present.<br><br>Input Data:<br><br>• Generated Video<br>• Reference Images<br>• Structured Caption<br><br>Task Definition: Given a generated video, a set of reference images (ID images), and a structured caption, you must determine if the "Speaker Attribution" is correct. Speaker Confusion (Output: 1) occurs if the following conditions are met:<br><br>• Subject A is performing the lip movements or facial expressions corresponding to the dialogue lines assigned to Subject B in the caption.<br>• Subject A's visual appearance (e.g., clothing, face) is partially or fully blended with Subject B while they are speaking.<br><br>Output Format:<br><br>• 1 if Speaker Confusion is detected<br>• 0 if the video is consistent with the references and caption.<br>|
|---|

###### Figure 9 MLLM system prompt for speaker confusion detection.

Method Text-Video Alignment ID-Sim. Video Quality Text-Audio Alignment Timbre-Sim. Audio Quality Lip-sync

Phantom 3.62 3.55 3.35 - - - VACE 3.45 3.47 3.28 - - - Qwen-Image + LTX-2 3.32 3.09 3.14 4.18 2.41 3.73 2.91 Qwen-Image + Ovi 3.70 3.05 3.64 4.23 2.41 3.77 3.32 Wan2.6 3.51 3.18 3.77 3.57 2.95 4.08 3.12 Ours 3.86 3.95 3.68 4.75 3.50 4.23 4.50

Table 7 User Study with state-of-the-art methods on R2AV. Best results are in bold, second best are underlined.

#### A.5 More Visual Results

As shown in Fig. 10-13, we provide more qualitative results of DreamID-Omni on R2AV, RV2AV and RA2V task.

...In a open-kitchen café, <sub1>... tastes the sauce with a spoon, said: “I keep telling myself I’m fine...”

|[Figure 341]|
|---|

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

... A modern gym, <sub1>... lift the dumbbells, ... said: “I am very tired now, but I don't intend to stop.”

|[Figure 350]|
|---|

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

... in a clothing store, <sub1> wearing blue tank top, holds up a pink tank top and said: “This tank top is exactly...”

|[Figure 359]|
|---|

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

... in a recording studio, <sub1>... in front of a microphone... said: “Hey everyone, welcome back to the channel! ...”

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

|[Figure 373]|
|---|

[Figure 374]

[Figure 375]

[Figure 376]

... a playful room with colorful toys, <sub1>... stands ... a powerful pose... said: “I'm a superhero, let me handle...”

|[Figure 377]|
|---|

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

... living room, <sub1>... sitting at a desk with a laptop ... and speaking: “... everyone, glad you could join..”

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

|[Figure 391]|
|---|

[Figure 392]

[Figure 393]

[Figure 394]

###### Figure 10 More qualitative results on R2AV I

...A fashion magazine's editorial office, ... <sub1> on the left... suggests: “We should start ...”, <sub1> <sub2> <sub2> is on the right... replies: “I've already been ...”

|[Figure 395]|
|---|

[Figure 396]

|[Figure 397]|
|---|

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

...at a family wedding reception, ... <sub1> on the left... said: “Look, how magnificent it is!”, <sub2> ... smiles warmly and replies: “Yes, I just can't believe it.”

<sub1> <sub2>

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

|[Figure 411]|
|---|

|[Figure 412]|
|---|

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

...in a garden, ... <sub1> wearing yellow dress on the left... said: “Look over there! It's...”, <sub2> is in a simple white blouse and replies: “Oh wow, it's so beautiful...”

<sub1> <sub2>

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

|[Figure 423]|
|---|

|[Figure 424]|
|---|

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

...warmly lit living room, ... <sub1> is on the left... said: “Are you ready for the movie night?”, <sub2> is on the right ... replies: “Yes! I have the popcorn all ready.”

<sub1> <sub2>

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

|[Figure 435]|
|---|

|[Figure 436]|
|---|

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

...meeting space, ... <sub1> on the left... says: “We need to...”, <sub2> on the right... responds: “Option A gives...”, <sub3> in the center... says: “I agree. Let ...”

<sub2>

<sub1> <sub2>

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

|[Figure 447]|
|---|

|[Figure 448]|
|---|

|[Figure 449]|
|---|

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

###### Figure 11 More qualitative results on R2AV II

... dimly lit with some blurred lights,... <sub1> ... said: “Dried blood on the outside of that bag...”

|[Figure 459]<br><br>[Figure 460]<br><br>[Figure 461]<br><br>[Figure 462]<br><br>[Figure 463]<br><br>[Figure 464]<br><br>[Figure 465]<br><br>[Figure 466]<br><br>[Figure 467]<br><br>[Figure 468]|
|---|

Source Video

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

|[Figure 474]|
|---|

[Figure 475]

[Figure 476]

[Figure 477]

... in a warmly room,... <sub1> ... dark blue suit and a light blue shirt, said: “here, I believe...”

|[Figure 478]<br><br>[Figure 479]<br><br>[Figure 480]<br><br>[Figure 481]<br><br>[Figure 482]<br><br>[Figure 483]<br><br>[Figure 484]<br><br>[Figure 485]<br><br>[Figure 486]<br><br>[Figure 487]|
|---|

Source Video

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

|[Figure 493]|
|---|

[Figure 494]

[Figure 495]

[Figure 496]

###### Figure 12 More qualitative results on RV2AV

...in an unlit space, <sub1> ... said: “Just figured I'd be by your side when it happened”

|[Figure 497]|
|---|

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

Driving Audio

... outdoors, <sub1>...beige jacket, ... said: “bout you. About how you're changing.”

|[Figure 504]|
|---|

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

Driving Audio

... in a dimly lit room, ...blue glow of a computer monitor, <sub1> said: “increasingly powerful...”

|[Figure 511]|
|---|

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

Driving Audio

###### Figure 13 More qualitative results on RA2V

