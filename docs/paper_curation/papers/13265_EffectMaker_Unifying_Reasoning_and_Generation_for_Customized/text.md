## EffectMaker: Unifying Reasoning and Generation for Customized Visual Effect Creation

Shiyuan Yang1,2,† Ruihuang Li1,† Jiale Tao1 Shuai Shao1,∗ Qinglin Lu1,‡ Jing Liao2,‡ 1Tencent Hunyuan 2City University of Hong Kong

†Equal contribution. ∗ Project lead. ‡Corresponding authors.

# arXiv:2603.06014v1[cs.CV]6Mar2026

[Figure 1]

|[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>User image Generated Video<br><br>[Figure 12]|[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>User image Generated Video<br><br>[Figure 22]|[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>User image Generated Video<br><br>[Figure 31]|
|---|---|---|
|[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>User image Generated Video<br><br>[Figure 41]|[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>User image Generated Video<br><br>[Figure 52]|[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>User image Generated Video<br><br>[Figure 63]|
|[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]|[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]|[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]|

ReferenceVideoReferenceVideoReferenceVideo

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

User image Generated Video User image Generated Video User image Generated Video

Figure 1. Given a reference video with visual effect (top row in each grid), and a user-specified target image (wrapped by shadow box), our EffectMaker transfers the reference effect to user image to create vivid video (bottom row in each grid) with the same effect pattern.

### Abstract

of modeling supernatural or stylized effects. Moreover, these approaches often require per-effect fine-tuning, which severely limits their scalability and generalization to novel VFX. In this work, we present EffectMaker, a unified reasoning–generation framework that enables reference-based VFX customization. EffectMaker employs a multimodal large language model to interpret high-level effect semantics and reason about how they should adapt to a target

Visual effects (VFX) are essential for enhancing the expressiveness and creativity of video content, yet producing high-quality effects typically requires expert knowledge and costly production pipelines. Existing AIGC systems face significant challenges in VFX generation due to the scarcity of effect-specific data and the inherent difficulty

subject, while a diffusion transformer leverages in-context learning to capture fine-grained visual cues from reference videos. These two components form a semantic–visual dual-path guidance mechanism that enables accurate, controllable, and effect-consistent synthesis without per-effect fine-tuning. Furthermore, we construct EffectData, the largest and high-quality synthetic dataset containing 130k videos across 3k VFX categories, to improve generalization and scalability. Experiments show that EffectMaker achieves superior visual quality and effect consistency over state-of-the-art baselines, offering a scalable and flexible paradigm for customized VFX generation. Project page: https://effectmaker.github.io.

### 1. Introduction

Visual effects (VFX) are designed to enhance the visual expressiveness and aesthetic appeal of a video by introducing artistic or physical augmentations beyond the real world. They play a crucial role in film, advertising, and gaming, serving as a creative medium to enrich narrative experience and visual immersion. However, producing high-quality VFX typically requires professional expertise and expensive production pipelines, making it one of the most resource-intensive components in modern video creation. As generative AI technologies evolve, leveraging AIGC (AI-generated content) for VFX generation holds great promise for reducing production costs while enabling accessible, creative, and controllable video effect design.

Despite remarkable advances in video generative models, current AIGC systems remain underexplored for visual effect generation. Although large-scale diffusion-based video models trained on massive real-world datasets can generate visually realistic content, they struggle with supernatural, non-physical, and exaggerated effects — which are inherently out-of-domain from real-world video distributions. Moreover, VFX-specific data are scarce and costly to obtain, limiting systematic research in this direction. Existing attempts often rely on small, closed-set VFX categories and require either fine-tuning a dedicated LoRA for each effect [33] — which is inefficient and hard to scale, or training a mixture of LoRAs on a limited effect set [35], which lacks generalization to open-set unseen effects.

A natural alternative for open-set VFX generation is to describe target effects via text prompts, as in text-to-video systems. However, visual effects are often abstract, multilayered, and stylistically complex, making them difficult to capture precisely with language alone. Even for professional designers, textual descriptions may not always be a good choice as text often fail to convey the nuanced texture, motion dynamics, and atmosphere of a desired effect. In practice, designers often rely on visual references—transferring the “look and feel” of one effect into

another visual context—an intuitive and visually grounded process that textual prompting cannot achieve. Therefore, beyond the data scarcity challenge, a key technical problem lies in accurately extracting and transferring effects from reference videos.

In this work, we present EffectMaker, a unified reasoning–generation framework designed for effect-driven video synthesis. On the reasoning side, a multimodal large language model (MLLM) first comprehends the effect’s highlevel semantics from the reference, and then reasons about how to adaptively apply this effect to the novel subject in user’s image. This frees users from explicitly describing complex visual effects. On the generation side, a diffusion transformer (DiT) [38] leverages in-context learning to capture fine-grained visual details from the reference. Together, these two components form a semantic–visual dualpath guidance mechanism that jointly steers the model toward semantically aligned and visually consistent VFX synthesis. By framing effect creation as a reference-based task and leveraging an MLLM for enhanced visual understanding and reasoning, our method offers an intuitive, scalable, and highly controllable alternative to previous LoRA-based approaches, eliminating the need for effect-specific finetuning.

To address the VFX data scarcity, we construct EffectData, the largest VFX dataset to date, comprising over 130k videos across 3k diverse effect categories, including atmospheric, transformation, stylistic effects. EffectData expands the effect category by an order of magnitude compared with existing datasets and offers paired annotations. We will release the dataset to support future research in VFX generation and editing. We validate EffectMaker against state-of-the-art effect generation methods. Extensive experiments demonstrate that our approach achieves superior visual quality and effect consistency.

In summary, our main contributions are as follows:

- • We propose EffectMaker, a reference-based framework for VFX customization. It unifies multimodal understanding and controllable generation in a single model to apply an effect from a reference video onto a target image.
- • We introduce semantic–visual dual-path guidance mechanism, which combines MLLM-based understanding with a Diffusion Transformer’s in-context learning ability, enabling accurate and controllable VFX transfer.
- • We construct EffectData, the largest high-quality VFX dataset to date, containing 3k effect classes, providing a valuable resource for future research on VFX generation.

### 2. Related work

General video generation. Recent advances in diffusion models have significantly advanced text-to-video (T2V) generation. Early T2V approaches extended 2D image diffusion models into the temporal domain using U-Net–based

backbones equipped with spatio-temporal attention or motion modules [5, 6, 18], enabling controllable short-video generation. More recently, the field has shifted toward Diffusion Transformers (DiTs) [38], which offer greater scalability and stronger generative capacity. Prominent DiTbased systems such as CogVideo [49], Wan [40], HunyuanVideo [23], Sora [7], and Veo [2] have achieved unprecedented realism and fidelity in open-domain video generation, which has greatly benefited down-stream applications in creative visual creation [9, 27, 44, 45, 48, 54].

Generation with understanding. With the rapid development of MLLMs in the field of multi-modal understanding [29, 30, 52, 53], increasing attention has been devoted to unifying vision understanding and visual generation within a single reasoning–generation framework. Existing approaches can be grouped into three paradigms. The first extends auto-regressive LLMs to visual domains by predicting discrete visual tokens [11, 20, 39, 41, 51], but typically suffers from limited fidelity and weak spatial coherence. The second, LLM–diffusion hybrids [10, 14, 37, 43], uses the LLM for high-level reasoning while delegating image synthesis to an external diffusion decoder, achieving a practical balance between semantic capability and visual quality. The third paradigm explores unified architectures [12, 47, 56] that merge LLM and diffusion into a single transformer, offering tighter semantic–visual coupling but with high computational cost and limited scalability for video. Following recent understanding–generation frameworks [36, 46, 50], we adopt the second paradigm for its lightweight yet effective integration of understanding and generation in reference-based visual effect synthesis.

Visual effect generation. Visual effect generation remains a relatively underexplored problem in the video generation community. The pioneering AIGC-based method MagicVFX [17] synthesizes effects by directly copying pixel-level content from a reference video to a target video within a user-specified region, followed by noise injection for refinement. However, this copy–paste manner lacks flexibility when the reference and target scenes differ significantly, and it requires extensive manual adjustment. VFXCreator [33] enables effect transfer by fine-tuning a separate LoRA for each effect type, achieving reasonable results for single effects but suffering from poor scalability and generalization to unseen ones. Omni-Effects [35] improves scalability via a mixture-of-LoRA (LoRA-MoE) strategy trained on 55 effect categories, yet the limited effect diversity constrains domain generalization. Two concurrent works, Video-as-Prompt [4] and VFXMaster [26], adopt a reference-based paradigm similar to ours and demonstrate better generalization to unseen effects. Nevertheless, their models lacks reasoning ability, relies on carefully and

Table 1. Comparison of existing VFX datasets. Our dataset significantly expands the number of effect classes and provides detailed captions and editing instructions.

Dataset # Classes # Videos Label Caption Instruction

VFX-307[17] – 190 ✗ ✗ ✗ Open-VFX[33] 52 634 ✓ ✗ ✗ Omni-VFX[35] 55 1700 ✓ ✗ ✗ HiggisField[1] 145 1060 ✓ ✓ ✗ VAP-Data[4] 100 90k ✓ ✗ ✗ EffectData (Ours) 3000 130k ✓ ✓ ✓

manually-crafted effect prompts, making them unfriendly for interactive use. Moreover, existing VFX datasets cover limited range of effect categories (dozens to a few hundred), whereas our EffectData increases this scale by an order of magnitude. It also provides per-video annotations including labels, captions, and instructions. A statistic comparison of VFX-related dataset is shown in Tab. 1.

### 3. Method

#### 3.1. Overview

In this work, we propose EffectMaker, a unified referencebased visual effect generation model that seamlessly integrates the sophisticated visual effect understanding and reasoning capabilities of MLLMs with the in-context generative power of video DiT model. We also introduce EffectData, the largest paired VFX dataset with comprehensive annotations. Our task is formulated as follows: given a single reference VFX video and a target image, EffectMaker analyzes and comprehends the VFX in the reference video and transfers them to the target image, generating visually compelling videos with the same stylistic effects.

An overview of our model architecture is illustrated in Fig. 2. Our model consists of two main components: on the understanding side, an MLLM model is responsible for semantically understanding and reasoning about the reference visual effects; On the generation side, a video DiT synthesizes the target video conditioned on the extracted effect representations.

#### 3.2. Effect understanding

On the understanding side, our MLLM is initialized from the Qwen3-VL-8B model [3]. On the input side, it encodes multimodal information including the system prompt, user instruction, reference video, and user image. The user instruction guides the MLLM to first analyze the visual effects presented in the reference video, then examine the content of the user image, and reason how the effect should adapt to a new target, especially those with major differences in shape, and finally imagine and describe the appearance of the target image after transferring the specified visual effect. On the output side, to fully exploit the model’s multimodal

Denoise

[Figure 101]

[Figure 102]

[Figure 103]

Reference video

User image

System Prompt

User Instruction

MLLM hidden state

Reasoning token

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

🔥 ×𝑁 blocks

[Figure 113]

###### Video DiT

Reference latent

Target latent

×𝑇 steps

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Noisy video latent

[Figure 119]

[Figure 120]

[Figure 121]

VAE Encode

[Figure 122]

[Figure 123]

Self-attn

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

FFN

Cross-attn1

Cross-attn2

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

VAE Decode

Tokenizer

ViT Encode

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

MLLM

AR

[Figure 139]

[Figure 140]

Output video

[Figure 141]

[Figure 142]

Connector

[Figure 143]

🔥

[Figure 144]

[Figure 145]

Effect Understanding Effect Generation

Figure 2. Overview of our model architecture. Given a reference VFX video and a target image, on the reasoning side, an MLLM extracts high-level semantic cues of the reference video, providing abstract effect descriptions that serve as semantic guidance. On the generation side, a video DiT model leverages in-context generation to capture fine-grained visual details from the reference, and generates a target video with consistent visual effect.

Semantic conditioning via decoupled cross-attention. The semantic-level conditioning is derived from the understanding stage described above, encompassing both MLLM’s understanding and reasoning features. A straightforward fusion strategy is to concatenate these two modalities and feed them into the DiT’s cross-attention layers. However, we found that such a direct combination compromises the model’s representational capacity. To better preserve modality-specific information, we employ a decoupled cross-attention mechanism. Concretely, reasoning features are encoded using the original T5 text encoder of DiT and injected through the standard cross-attention layers. For understanding features, we introduce an additional cross-attention branch that processes them independently. Both branches share the same query representation but use distinct key and value projections for the textual and visual features. The outputs from the two branches are then fused together by direct feature addition. To prevent unintended semantic interference with the reference stream, the crossattention is performed exclusively between the target video tokens and the semantic conditions.

understanding and reasoning capabilities, we extract two complementary types of features: semantic-understanding features and semantic-reasoning features. The semanticunderstanding features are obtained from the hidden states of the last layer of the MLLM, encoding rich multimodal representations of the reference inputs. However, this feature only captures the current input semantics through a single forward pass and does not involve any reasoning about how the reference effect should be applied to the user’s image. To complement this, we further extract semanticreasoning features from the auto-regressively predicted text token sequence. These features summarize the model’s previous understanding and encapsulate its reasoning process to infer the user-desired final outcome. The two feature types are then jointly used as conditioning signals for the video DiT. Since the feature spaces of the MLLM and the DiT are not aligned, we introduce a light connector module to bridge this modality gap. The overall workflow is illustrated in the left part of Fig. 2.

#### 3.3. Effect generation

Visual conditioning via in-context learning. While semantic conditioning provides global guidance on what effect to generate, it lacks the fine-grained spatial and temporal details required for visually faithful effect cloning. To address this limitation, we further leverage the in-context learning capability of DiT for visual-level conditioning, following prior studies [21, 34]. Specifically, both the reference and target videos are encoded into latent representa-

On the generation side, we employ an image-to-video base model to transfer the reference effect onto a user-provided subject image, thereby synthesizing a dynamic visual effect video. Specifically, we adopt Wan2.2-TI2V-5B [40] as our DiT-based backbone. To achieve faithful effect cloning, we introduce a semantic–visual dual-conditioning strategy that injects reference effect information from both the semantic and visual levels.

Transition prompt

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

“You’re a VFX expert, help me construct an editing instruction based on <Effect_name>”...

"As the girl slowly lowers her hand from shielding the sun, her gaze becomes focused and determined, and a scorching energy begins to gather in her outstretched palm, swirling and condensing into a pulsating sphere of fire that seems to hold immense power."

[Figure 151]

Ice / ice spike ice crystal …

[Figure 152]

###### Fireball

VLM

[Figure 153]

[Figure 154]

Fire / fire breath fire dragon …

|[Figure 155]<br><br>[Figure 156]|
|---|

In the person's raised palm, a glowing, pulsating spherical fireball is coalescing, as if ready to be cast.

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

LLM

###### fireball

[Figure 162]

###### Lightning /

…

[Figure 163]

[Figure 164]

Effect taxonomy

Editing prompt

Paired images

First-last frame to video

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Video generation

[Figure 171]

[Figure 172]

[Figure 173]

|[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]|
|---|

Image Editor

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Filtering

[Figure 184]

Subject editing

[Figure 185]

Source image

Target image

Target video

Subject images

(a) EffectData construction pipeline (b) Some VFX examples from EffectData

Figure 3. (a) Illustration of our EffectData construction pipeline. (b) Some examples from the EffectData dataset.

tions using a shared VAE encoder. After patchifying and flattening, the reference and target latents are concatenated along the sequence dimension and jointly processed by the DiT blocks. Notably, we modify the self-attention into a dual-stream scheme, where reference and target tokens are projected through separate query, key, and value projections to disentangle their representation spaces, while still allowing bi-directional attention across the combined token sequence. The attention operation is formulated as:

images, collected from internal sources and public datasets PPR10K [28]. We preprocess the data by filtering out images containing text, multiple or unclear subjects.

- Step 2: VFX taxonomy. We build systematic VFX taxonomy by defining a series of orthogonal attribute sets including VFX elements (e.g., ice, fire, magic), geometric patterns (e.g., particle, wave, ring), and attachment regions (e.g., face, arms, full body). This structured taxonomy supports compositional generation of diverse effect classes.
- Step 3: Instruction generation. For each VFX class, we prompt an LLM to generate multiple editing instructions describing how to transform the source image into its effectenhanced counterpart.
- Step 4: Subject editing. Given a source image from Step 1 and its corresponding instruction from Step 3, we employ image editing model to synthesize the target image with the desired effect.
- Step 5: Video generation. For each source-target image pair, we then prompt an MLLM to describe the dynamic transition between the two images, and feed this prompt along with the frame pair into first-last-frame-tovideo model to synthesize temporally coherent VFX videos.

Or = SA(Qr,[Kr;Kt],[Vr;Vt]) Ot = SA(Qt,[Kr;Kt],[Vr;Vt])

, (1)

where SA(·) denotes the self-attention operation, Qr,Kr,Vr and Qt,Kt,Vt are the query, key, and value features of the reference and target video tokens, respectively. [·;·] denotes concatenation along the sequence dimension. Or and Ot are the output features for the reference and target streams, which are then jointly processed by subsequent feed-forward layers.

Biased RoPE. To differentiate between the target and reference videos, we employ a biased 3D Rotary Position Embedding (RoPE) for the reference video. Specifically, we align the spatial components of the RoPE to those of the target video, while introducing a constant offset to the temporal dimension. This temporal bias separates the positional encoding spaces of the two videos, ensuring a safe margin to prevent interference between their RoPE.

### 4. Experiment

#### 4.1. Experiment setup

Dataset. Our datasets are collected from multiple sources, including EffectData synthesized using our proposed data pipeline (Sec. 3.4), supplemented by additional samples from OpenVFX dataset [33] and the Higgsfield website [1].

#### 3.4. Data construction

Existing VFX video datasets are scarce in both scale and diversity. To overcome this, we build a data synthesis pipeline and construct a large-scale, high-quality paired VFX data. As shown in Fig. 3, the detailed steps are as follows:

Implementation details. During training, the reference and target videos are randomly sampled from the same VFX class. To reduce computational cost, we temporally downsample each reference video to 17 frames. Each frame is then resized such that the shorter side equals 448 pixels.

Step 1: Subject collection. We first construct a subject dataset mainly composed of human portraits and animal

[Figure 186]

Figure 4. Qualitative comparison with related baselines on OpenVFX dataset.

Table 2. Quantitative comparison with related baselines on the OpenVFX dataset. VQ=visual quality, MQ=motion quality, TA=text alignment, CAS=class alignment score. All metrics are model-based; higher values indicate better performance.

Metrics Methods Cake-ify Crumble Decapitate Deflate Dissolve Explode Eye-pop Harley Inflate Levitate Melt Squish Ta-da Venom Avg.

VFX-Creator 2.71 2.41 2.68 2.33 1.69 1.38 2.13 0.97 2.41 2.50 1.96 1.77 2.28 1.62 2.06 Omni-Effect 3.11 2.49 2.62 2.39 1.93 1.84 2.23 1.48 2.55 2.27 2.20 2.27 2.59 1.82 2.27 Wan2.2-FT 2.15 1.42 2.63 2.15 1.64 0.91 2.23 1.71 1.89 2.12 1.74 2.11 2.04 1.69 1.89 Ours 3.39 2.94 3.01 2.99 1.93 2.23 2.44 2.17 3.19 3.70 2.53 2.79 3.99 2.46 2.84

VQ↑

VFX-Creator -0.05 -0.15 -0.05 0.13 -0.17 -0.35 0.11 -0.26 -0.14 0.48 -0.16 -0.41 0.77 -0.21 -0.03 Omni-Effect 0.12 -0.27 -0.08 0.02 -0.09 -0.33 0.04 -0.28 -0.11 -0.04 -0.14 -0.37 -0.09 -0.34 -0.14 Wan2.2-FT 0.22 -0.16 0.79 0.41 0.06 -0.30 0.45 0.27 0.04 0.28 0.09 -0.02 0.63 0.01 0.20 Ours 0.59 0.02 0.11 0.51 -0.06 -0.13 0.73 0.25 0.09 0.65 0.26 -0.05 0.52 0.04 0.25

MQ↑

VFX-Creator -2.63 1.54 -1.82 -3.38 -3.50 1.44 0.90 1.51 -2.95 -1.29 -0.58 0.99 -3.36 0.26 -0.92 Omni-Effect -2.23 1.80 -1.99 -3.10 -3.32 1.77 1.27 1.51 -1.71 -1.04 -0.22 0.87 -2.43 0.28 -0.61 Wan2.2-FT -0.56 1.43 -2.20 -2.51 -1.38 0.87 0.75 -1.22 -0.04 -1.03 -1.50 0.94 -2.62 -0.74 -0.70 Ours -1.89 1.40 -1.60 -2.31 -0.55 1.11 1.29 1.00 0.41 -0.47 -0.17 0.80 -2.35 -0.08 -0.24

TA↑

VFX-Creator 4.00 5.00 5.00 5.00 5.00 4.20 5.00 5.00 5.00 1.00 5.00 5.00 1.80 4.80 4.34 Omni-Effect 2.80 5.00 4.00 2.80 4.80 4.80 5.00 5.00 5.00 4.00 4.60 5.00 3.80 5.00 4.40 Wan2.2-FT 2.20 5.00 0.00 5.00 5.00 3.00 4.40 4.20 3.00 4.60 1.00 4.00 3.80 5.00 3.59 Ours 3.60 4.60 5.00 4.20 5.00 3.80 5.00 5.00 4.60 4.80 5.00 5.00 4.20 5.00 4.63

CAS↑

For the target video, we set the frame length to 81 frames with the shorter side fixed to 704 pixels, and the longer side is proportionally adjusted to maintain the aspect ratio of the user-provided first frame. We train our model for approximately 50k steps on 32 NVIDIA H20 GPUs using the Adam optimizer [22] with a learning rate of 2 × 10−5.

Metrics. We evaluate model performance from two complementary perspectives: video quality and effect consis-

tency. For quality assessment, we employ reward-modelbased metrics from VideoAlign [32], including Visual Quality (VQ) score, which evaluates appearance fidelity, and the Motion Quality (MQ) score, which measures temporal smoothness and motion realism. For effect consistency, we use the Text Alignment (TA) score from VideoAlign, which quantifies the alignment between generated videos and textual descriptions. We further leverage Gemini 2.5 API to rate a 0-5 Effect Class Alignment Score (CAS) and an Ef-

fect Reference Alignment Score (RAS) that reflect how well the effect in the generated video matches the intended VFX class name and reference VFX.

Baselines. We compare our method with previous stateof-the-art VFX generation methods including VFXCreator [33] and Omni-Effects [35], we also finetuned our base model Wan2.2-TI2V-5B on our dataset as a baseline method, referred as Wan2.2-FT.

#### 4.2. Qualitative results

Close-set comparison. Since the baseline methods VFXCreator and Omni-Effects were trained on the OpenVFX dataset, which contains a closed set of visual effects, we also trained our model on the same dataset and selected the common categories as our testing benchmark for fair comparison. In Fig. 4, we provide side-by-side qualitative results of different methods. Notably, the reference video in the last row is solely provided for our method only. It is evident from the Inflate and Ta-da cases that our method produces more accurate and visually consistent effects compared to the Wan2.2-FT and Omni-Effect. For instance, in the Inflate case, both Omni-Effect and Wan2.2FT failed to generate the expected inflated and floating boy, whereas our model achieved a realistic and coherent transformation. Although VFXCreator achieves comparable visual quality in most cases, it requires finetuning per-effect LoRA models, making it less flexible and scalable than our feed-forward approach. Benefiting from strong visual understanding and in-context learning, our method also successfully transfers spatial and temporal effect patterns from the reference video, as seen with the explosion flames and flying debris in the Explode effect.

Open-set comparison. To evaluate the generalization capability of our model, we further compare it with OmniEffects and Wan2.2-FT on open-set visual effects that were not included in the training set. As illustrated in Fig. 5, we present several examples of unseen effects, including portal, plastic model, and green tree glowing. It is evident that Omni-Effects fails to generate plausible effects for unseen categories, while Wan2.2-FT can produce roughly correct patterns but with noticeably lower similarity to the reference effect. This suggests that text-only conditioning struggles to capture the fine-grained visual cues and temporal dynamics of complex effects. In contrast, our method, guided by reference videos, successfully reproduces these unseen effects with higher consistency relative to the reference.

Reference rephrasing. Since our framework builds upon the TI2V model, which inherently supports both I2V and T2V generation, it naturally extends to generating effect videos without an input image. In this setting, we leverage

[Figure 187]

Figure 5. Qualitative comparison on unseen visual effects.

[Figure 188]

Figure 6. Reference rephrasing results. Given only a reference effect video, our model reproduces the target video with similar visual content and effect.

the MLLM to describe both the content as well as the VFX in the reference video, and then employ the DiT to render a new video that reproduces the content and effect. We refer to this application as reference rephrasing. We show several examples in Fig. 6. The results demonstrate that our semantic–visual dual-path guidance mechanism can effectively rephrase the reference video cues.

#### 4.3. Quantitative results

Following Omni-Effect [35], we conduct a quantitative comparison across 14 visual effect classes from the OpenVFX dataset. For each class, we test on 10 subject images and report the average scores for four metrics: VQ, MQ, TA, and CAS, as summarized in Tab. 2. The results demonstrate that our method not only achieves superior visual and motion quality but also surpasses existing state-ofthe-art VFX generation methods in terms of effect alignment and responsiveness. Notably, the comparison with the Wan2.2-FT baseline further highlights the advantage of the reference-based paradigm over purely text-driven generation, as the former contributes more effectively to modeling

[Figure 189]

Figure 7. Ablation on different types of conditioning.

dynamic and complex visual patterns.

#### 4.4. Ablation study

Conditioning design. To verify our dual-conditioning strategy, i.e. semantic conditioning and visual in-context conditioning, we conduct an ablation study and report the quantitative results in Tab. 3, along with qualitative comparisons in Fig. 7. As shown, using either semantic conditioning or visual in-context conditioning alone leads to inferior performance. When conditioned solely on the visual reference video, the model tends to replicate only low-level appearance cues, such as color and texture, while failing to capture complex effect structures (e.g., it cannot reproduce the DNA-like geometry as seen in Fig. 7(c)). In contrast, relying only on semantic conditioning yields semantically correct effects but lacks fine-grained spatial details. Notably, combining both conditioning achieves the most faithful performance, demonstrating the effectiveness of our proposed semantic-visual conditioning strategy.

Table 3. Ablation study on different conditioning types.

Conditioning VQ↑ MQ↑ TA↑ CAS↑ RAS↑

Semantic 2.78 0.16 1.06 4.20 3.84 Visual 2.48 0.12 -0.38 2.24 1.48 Semantic+Visual 2.92 0.21 1.24 4.40 4.16

Data scaling. We further examine the impact of scaling up the training data on the model’s generalization capability. Specifically, we train our model with two different orders of magnitude of effect classes (i.e. 100, and 1000) and evaluate them on open-set effects. The results are reported in Tab. 4. As shown, increasing the data scale consistently improves both quality and consistency metrics. We attribute this improvement to the broader domain coverage provided by a larger training set, which enables better interpolation and extrapolation when the model encounters new effects.

Attention design. We compare our dual-stream attention design mentioned in Sec. 3.3 with a single-stream variant where reference and target tokens share the same projection matrices. We find that switching to single-stream attention degrades the effect transfer capability, with TA decreasing

Table 4. Effect of scaling up training set on model performance.

# VFX Class VQ ↑ MQ ↑ TA ↑ CAS ↑ RAS ↑

100 2.76 0.13 0.94 3.76 3.22 1000 2.89 0.19 1.21 4.22 4.04

[Figure 190]

Figure 8. Results of human preference percentage.

from 1.24 to 0.81, CAS from 4.40 to 3.30, and RAS from 4.16 to 2.84. This is likely because reference and target tokens lie in heterogeneous (clean vs. noisy) domains, where separate projections better handle the distribution gap.

#### 4.5. User study

We conduct a user study on OpenVFX results, involving 30 participants and 28 questions (two per effect). Participants compared side-by-side video results with the effect name and reference video, and selected their preferred result based on three criteria: effect quality, class alignment, and reference alignment. The display order was randomized to reduce bias. As shown in Fig. 8, our method achieves the highest preference rates across all criteria, demonstrating superior perceptual quality and effect consistency.

### 5. Conclusion

We present EffectMaker, a reasoning–generation framework for customized visual effect cloning. By integrating the advanced reasoning capability of MLLM with the incontext learning ability of DiT, EffectMaker achieves superior performance in replicating complex reference effects across diverse scenarios. In addition, we introduce EffectData, the largest and high-quality paired VFX dataset to date, which provides systematic captions and serves as a valuable resource for future research and applications in art, gaming, and advertising communities.

Limitation. Despite the promising results, EffectMaker still has several limitations. First, our approach may struggle with extremely complex effects involving rapid or large motions, primarily due to the base model’s capacity bound. Second, the reliance on synthetic training data may introduce biases and may not fully capture the diversity and realism of real-world VFX scenarios (e.g., noisy, ambiguous, or multiple overlapping effects). Future work will explore stronger base models and incorporate diverse real-world data sources to further improve generalization and realism.

### References

- [1] Higgsfield. https://higgsfield.ai/, 2025. Accessed: June 1, 2025. 3, 5
- [2] Veo3. https://veo3.im/, 2025. Accessed: June 18,

2025. 3

- [3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, et al. Qwen3-vl technical report, 2025. 3
- [4] Yuxuan Bian, Xin Chen, Zenan Li, Tiancheng Zhi, Shen Sang, Linjie Luo, and Qiang Xu. Video-as-prompt: Unified semantic control for video generation. arXiv preprint, arXiv:2510.20888, 2025. 3
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [6] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 3
- [7] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. Technical report, OpenAI, 2024. 3
- [8] ByteDance. Jimeng ai. https://jimeng.jianying. com/, 2025. 15
- [9] Liyi Chen, Ruihuang Li, Guowen Zhang, Pengfei Wang, and Lei Zhang. Fast multi-view consistent 3d editing with video priors. arXiv preprint arXiv:2511.23172, 2025. 3
- [10] SiXiang Chen, Jianyu Lai, Jialin Gao, Tian Ye, Haoyu Chen, Hengyu Shi, Shitong Shao, Yunlong Lin, Song Fei, Zhaohu Xing, et al. Postercraft: Rethinking high-quality aesthetic poster generation in a unified framework. arXiv preprint arXiv:2506.10741, 2025. 3
- [11] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint, arXiv:2501.17811,

2025. 3

- [12] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint, arXiv:2505.14683, 2025. 3
- [13] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 12

- [14] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehen-

- sion and generation. arXiv preprint arXiv:2404.14396, 2024. 3
- [15] Google. Nano banana - ai image editor. https:// nanobanana.ai/, 2025. 14
- [16] Google. Google gemini. https://gemini.google. com/app/, 2025. 13, 14
- [17] Jiaqi Guo, Lianli Gao, Junchen Zhu, Jiaxin Zhang, Siyang Li, and Jingkuan Song. Magicvfx: Visual effects synthesis in just minutes. In Proceedings of the 32nd ACM International Conference on Multimedia (MM ’24), October 28–November 1, 2024, Melbourne, VIC, Australia, pages 8238–8246,

2024. 3

- [18] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 3
- [19] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 12
- [20] Yang Jiao, Haibo Qiu, Zequn Jie, Shaoxiang Chen, Jingjing Chen, Lin Ma, and Yu-Gang Jiang. Unitoken: Harmonizing multimodal understanding and generation through unified visual encoding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3600–3610, 2025. 3
- [21] Xuan Ju, Weicai Ye, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Qiang Xu. Fulldit: Video generative foundation models with multimodal control via full attention. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15737–15747, 2025. 4
- [22] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

2014. 6

- [23] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, and et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603,

2024. 3

- [24] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2023. 14
- [25] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space,

2025. 14

- [26] Baolu Li, Yiming Zhang, Qinghe Wang, Liqian Ma, Xiaoyu Shi, Xintao Wang, Pengfei Wan, Zhenfei Yin, Yunzhi Zhuge, Huchuan Lu, and Xu Jia. Vfxmaster: Unlocking dynamic visual effect generation via in-context learning. arXiv preprint, arXiv:2510.25772, 2025. 3
- [27] Ruihuang Li, Liyi Chen, Zhengqiang Zhang, Varun Jampani, Vishal M Patel, and Lei Zhang. Syncnoise: Geometrically consistent noise prediction for instruction-based 3d editing. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4905–4913, 2025. 3

- [28] Jie Liang, Hui Zeng, Miaomiao Cui, Xuansong Xie, and Lei Zhang. Ppr10k: A large-scale portrait photo retouching dataset with human-region mask and group-level consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 653–661, 2021. 5, 13
- [29] Yunlong Lin, Zixu Lin, Kunjie Lin, Jinbin Bai, Panwang Pan, Chenxin Li, Haoyu Chen, Zhongdao Wang, Xinghao Ding, Wenbo Li, et al. Jarvisart: Liberating human artistic creativity via an intelligent photo retouching agent. arXiv preprint arXiv:2506.17612, 2025. 3
- [30] Yunlong Lin, Linqing Wang, Kunjie Lin, Zixu Lin, Kaixiong Gong, Wenbo Li, Bin Lin, Zhenxi Li, Shiyi Zhang, Yuyang Peng, et al. Jarvisevo: Towards a self-evolving photo editing agent with synergistic editor-evaluator optimization. arXiv preprint arXiv:2511.23002, 2025. 3
- [31] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 12
- [32] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025. 6, 12
- [33] Xinyu Liu, Ailing Zeng, Wei Xue, Harry Yang, Wenhan Luo, Qifeng Liu, and Yike Guo. Vfx creator: Animated visual effect generation with controllable diffusion transformer. arXiv preprint, arXiv:2502.05979, 2025. 2, 3, 5, 7
- [34] Yawen Luo, Jianhong Bai, Xiaoyu Shi, Menghan Xia, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Tianfan Xue. Camclonemaster: Enabling reference-based camera control for video generation. arXiv preprint arXiv:2506.03140,

2025. 4

- [35] Fangyuan Mao, Aiming Hao, Jintao Chen, Dongxia Liu, Xiaokun Feng, Jiashu Zhu, Meiqi Wu, Chubin Chen, Jiahong Wu, and Xiangxiang Chu. Omni-effects: Unified and spatially-controllable visual effects generation. arXiv preprint arXiv:2508.07981, 2025. 2, 3, 7
- [36] Chong Mou, Qichao Sun, Yanze Wu, Pengze Zhang, Xinghui Li, Fulong Ye, Songtao Zhao, and Qian He. Instructx: Towards unified visual editing with mllm guidance,

2025. 3

- [37] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025. 3
- [38] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2, 3

- [39] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

- 3

[40] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 3,

- 4, 15

- [41] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 3
- [42] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 14
- [43] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint, arXiv:2506.18871, 2025. 3
- [44] Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi Xie, and Lei Zhang. Insvie-1m: Effective instruction-based video editing with elaborate dataset construction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16692–16701, 2025. 3
- [45] Chaodong Xiao, Zhengqiang Zhang, and Lei Zhang. BinaryAttention: One-bit qk-attention for vision and diffusion transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. 3
- [46] Yicheng Xiao, Lin Song, Yukang Chen, Yingmin Luo, Yuxin Chen, Yukang Gan, Wei Huang, Xiu Li, Xiaojuan Qi, and Ying Shan. Mindomni: Unleashing reasoning generation in vision language models with rgpo. arXiv preprint arXiv:2505.13031, 2025. 3
- [47] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 3
- [48] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with userdirected camera movement and object motion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 3
- [49] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 3
- [50] Shoubin Yu, Difan Liu, Ziqiao Ma, Yicong Hong, Yang Zhou, Hao Tan, Joyce Chai, and Mohit Bansal. Veggie: Instructional editing and reasoning video concepts with grounded generation. arXiv preprint, arXiv:2503.14350,

2025. 3

- [51] Peiying Zhang, Nanxuan Zhao, Matthew Fisher, Yiran Xu, Jing Liao, and Difan Liu. Duetsvg: Unified multimodal svg generation with internal visual guidance. arXiv preprint arXiv:2512.10894, 2025. 3
- [52] Shiyi Zhang, Wenxun Dai, Sujia Wang, Xiangwei Shen, Jiwen Lu, Jie Zhou, and Yansong Tang. Logo: A long-form video dataset for group action quality assessment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2405–2414, 2023. 3

- [53] Shiyi Zhang, Sule Bai, Guangyi Chen, Lei Chen, Jiwen Lu, Junle Wang, and Yansong Tang. Narrative action evaluation with prompt-guided multimodal interaction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18430–18439, 2024. 3
- [54] Shiyi Zhang, Junhao Zhuang, Zhaoyang Zhang, Ying Shan, and Yansong Tang. Flexiact: Towards flexible action control in heterogeneous scenarios. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11, 2025. 3
- [55] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. Advances in Neural Information Processing Systems, 36:49842–49869, 2023. 12
- [56] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 3

### A. Additional implementation details

Multi-resolution training. Since our framework does not require the reference video and the target video to share the same spatial resolution, we adopt a multi-resolution training strategy to improve robustness to variable input sizes. For the target video, we standardize the shorter side to 704 pixels, while the longer side is adaptively scaled to the nearest multiple of 32—consistent with the VAE’s downsampling factor (but no more than 1280 pixels), ensuring minimal distortion of the original aspect ratio. To reduce computational overhead, we downsample the reference video such that its shorter side is resized to 448 pixels, with the longer side similarly adjusted to a multiple of 32 while capped at 832 pixels.

Timestep sampling. For timestep sampling, instead of uniform timestep sampling, we adopt a semi-logit-normal distribution similar to the strategy used in Stable Diffusion 3 [13]. Specifically, the timestep t is sampled as follows:

1 1 + e−1.5z , z ∼ N(0,1) (2)

t ∼ T ·

where T denotes the total number of timesteps. This scheme biases the sampling density toward the middle timesteps, which has been shown to outperform uniform sampling in rectified flow-based diffusion models.

Timestep embedding. During the forward noising process at timestep t, we keep all reference tokens and the first-frame tokens of the target video noise-free. For these clean tokens, we always apply the zero timestep embedding. Noise is added only to the non–first-frame tokens of the target video, and we use the corresponding timestep embeddings for these noised tokens.

DiT sampling. The DiT model takes as input the concatenated latent sequence consisting of the target image latent (encoded from the user-provided image), the noise latent (sampled from Gaussian noise), and the encoded reference latent. Conditioned on the semantic features extracted by the MLLM, the DiT progressively denoises this sequence and predicts the target video using flow-matching–based sampling [31]. We adopt the UniPC sampler [55] with 30 denoising steps and apply classifier-free guidance with a scale of 6 [19]. The noise schedule shift parameter is set to 16.

Computational cost and efficiency Under the aforementioned configuration, the MLLM-based understanding stage requires 2 seconds, while the DiT-based generation stage takes 2 minutes on an H20 GPU. The end-to-end inference

consumes 48 GB of GPU memory. The memory can be further reduced if applying optimization techniques (e.g. model sharding, CPU offloading, sequence parallelism, quantization etc.).

""" # **Role** You are a multimodal evaluator specializing in visual effects (VFX) analysis. Your job is to assess how well the given video matches the described VFX.

# **Input**

- 1. Effect name: {VFX_NAME}
- 2. Effect description: {VFX_DESCRIPTION}
- 3. Video clip (the evaluation target)

# **Task** Evaluate the degree to which the video visually matches the described VFX, and assign an **Effect Match Score (0–5)** based on the following criteria:

Scoring rules:

- - 5: Effect perfectly matches the description; clear and well-executed.
- - 4: Effect mostly matches; minor deviations.
- - 3: Effect partially matches; moderate accuracy or clarity.
- - 2: Effect appears slightly; overall weak presence.
- - 1: Very faint traces of the effect, barely visible.
- - 0: No visible effect or completely irrelevant.

# **Output** Return the result in JSON format: {

"effect name": "The given effect name", “effect match score”: an integer score (0-5), "reason": "One-sentence reason for the score (≤20 words)"

} """

Figure 9. Prompt template for automatic Effect Class Alignment (CAS) Scoring.

Metrics. To evaluate video generation quality, we employ VQ (Visual Quality), MQ (Motion Quality), and TA (Text Alignment) as primary metrics. Derived from the most recent VideoAlign model [32], a multi-dimensional reward model built on the Qwen2-VL backbone that was trained on 182k human-annotated triplets using a Bradley-Terry-withTies (BTT) objective, allowing it to robustly learn relative preferences and ”tie” conditions from professional annotators. These metrics are designed to systematically mirror human judgment across distinct dimensions. VQ assesses the static visual fidelity of the video frames, focusing on clarity, aesthetic creativity, and logical reasonableness, independent of the textual context. MQ exclusively evaluates dynamic characteristics, measuring motion smoothness, physical plausibility, and temporal stability. Finally, TA serves as a context-aware metric that gauges the semantic consistency between the generated video and the input prompt, verifying the accurate representation of subjects, actions, and environments. Together, these indicators provide a comprehensive assessment that aligns closely with human perception

In addition to the above metrics, we further introduce the Effect Class Alignment Score (CAS) and Reference

[Figure 191]

Figure 10. Screen shot of user study questionnaire.

[Figure 192]

Figure 11. Detailed human preference percentages for each VFX class and each evaluation dimension.

Alignment Score (RAS) to assess how well a generated video conforms to the intended visual effect. Specifically, we feed the generated video together with the target effect name (or reference frames) into Gemini2.5 [16], instructing the model to evaluate the degree of alignment between the video and the specified effect class and to assign a score from 0 to 5. Here we show the CAS scoring prompt in Fig. 9.

User study details We conducted a user study to evaluate the perceptual quality of different baselines across the 14 effect categories in the OpenVFX dataset. The study included 28 questions (two per effect) and involved 30 participants. For each question, participants were shown sideby-side video results generated by different methods, along with the effect class name and a reference video. They were asked to choose their preferred result according to three criteria: (1) Effect quality: the overall visual fidelity judged purely by subjective perception; (2) Class alignment: how well the generated effect matches the given class name; and (3) Reference alignment: how closely the result resembles the reference video. To reduce selection bias, the display order in each question was randomly shuffled. We show a

screenshot of our user study questionnaire in Fig. 10.

We show detailed human preference percentages for each VFX class and each evaluation dimension in Fig. 11. The results show that for most of effect clasess, our method consistently achieved the highest or comparable preference rates across all evaluation dimensions, demonstrating its superior perceptual quality and effect consistency.

### B. Dataset details

Compared to conventional real-world video datasets, existing VFX datasets remain highly limited in both scale and diversity. To overcome this bottleneck, we develop a VFX synthesis pipeline that generates large-scale, highquality paired data suitable for supervised VFX learning. The pipeline consists of five major steps.

Subject collection. We first construct a diverse subject dataset primarily containing human portraits and animal images. Human portraits are sourced from a mixture of internal datasets and publicly available collections such as PPR10K [28]. Since these portraits are mostly female images, we additionally synthesize a subset of both male and

[Figure 193]

Figure 12. Word cloud of atmospheric VFX elements, showing the diversity of effect classes.

animal images using FLUX [24]. Given the raw subject image pool, we perform three preprocessing steps: (1) Superresolution: Images whose shorter side is below 704 pixels are upsampled via a high-fidelity super-resolution model. (2) Text erasure: We apply an OCR system to detect images with noticeable overlaid text and remove the text using Flux-Kontext [25]. (3) Quality filtering: A VLM-based filter is used to discard multi-person images and samples where the primary subject occupies only a small portion of the frame, ensuring clean, centered subjects for downstream generation.

VFX taxonomy. To synthesize a broad spectrum of VFX types, we construct a systematic and structured VFX taxonomy. Overall, our effects fall into two major categories: atmospheric effects and transformation effects.

Atmospheric effects refer to supernatural visual elements that originate from or surround the subject, enhancing mood, ambiance, or character attributes. To systematically describe such effects, we define orthogonal attribute sets that include: (1) VFX elements (e.g. fire, ice, light), (2) geometric patterns (e.g., particles, waves, rings), and (3) attachment regions (e.g., face, ear, arms, full body). These attributes combine to form a rich space of atmospheric effects. Fig. 12 shows the word cloud of atmospheric effect group.

Transformation effects indicate substantial modifications to the subject’s appearance, including outfit transformation (e.g. hairstyle, headwear, clothing, accessories etc.), identity transformation (e.g. gender swap, aging, rejuvenation, or transformation into animals, robots, or mythical creatures), and physical transformation (e.g. shattering, melting, explosion etc.).

[Figure 194]

Figure 13. We show four groups of paired images under the same VFX. Each group contains three paired images with different subjects but the same effect.

Instruction generation. For each atmospheric effect configuration, we prompt an LLM to combine the chosen effect element, geometric pattern, and attachment region into multiple diverse editing instructions that describe how to modify the source image into its VFX-enhanced counterpart. A prompt example is shown in Fig. 14. For transformation effects, we similarly prompt the LLM to expand each basic transformation into a detailed editing instruction suitable for image editing models.

Subject editing. For each editing instruction, we randomly sample multiple subject images from the subject image bank. We then employ an image editing model to synthesize a target image that incorporates the desired effect. We experimented with several editing models, including Flux-Kontext [25], Qwen-Image-Edit [42], and NanoBanana [15]. These models can produce consistent VFX patterns across different subject images when given the same editing instruction, as shown in Fig. 13, which is crucial for generating coherent paired videos. Beyond executing the core VFX instruction, we further prompt the model to adjust the subject’s pose, body motion, or facial expression to better match the semantics of the effect. The edited image and its corresponding original images collectively form a paired VFX group, from which we can sample a reference-target sample pair for supervised in-context training.

Video generation. For each source–target pair, we horizontally concatenate the two images and feed the combined image along with the effect description into Gemini 2.5 Pro [16]. We instruct the model to imagine the left image as

You are now a creative visual effects (VFX) designer who specializes in AI-based image editing, particularly in adding expressive atmospheric effects to character images. “Atmospheric effects” refer to supernatural visual elements generated by the character or surrounding them, enhancing emotion, mood, or personality—such as energy fields, glows, particle flows, elemental forces, etc. Assume that the source image is already provided, and each image contains a single clear main character.

The VFX element to be used is: {vfx_element}. Based on this VFX element, create five unique and imaginative image editing instructions. These instructions will be used to guide an AI image editing model to naturally integrate the effect into the scene while preserving the original character’s structure. Each instruction **must** include the following elements:

- - **VFX element type:** {vfx_element}.
- - **Effect attachment region:** Specify which part of the character the effect is applied to and how it appears. Examples include: emitted from the eyes, flowing from the eyes, exhaled from the mouth, growing from the ears, igniting from the hair, radiating from the head, released from the palms,

forming around the fists, appearing behind the body, rising from the feet, wrapping around the arms, surrounding the whole body, etc. Feel free to explore beyond these examples—be imaginative.

- **Optional shape or spatial form:** When helpful, describe the geometric or symbolic structure of the effect, such as: point-like, particle cluster, wave-like, linear, ring-shaped, shield-like, field-like, radial, explosive, tree-like, flowing, swirling, spherical, vortex, spiral, or figurative shapes such as dragon, tiger, serpent, blade, tidal wave, lightning, etc.

Important: The effect scope and application method must be appropriate for the given VFX element ({vfx_element}). Do *not* mechanically list keywords—embed them naturally into vivid, concrete, and actionable descriptions.

You must output five editing instructions, each in the following JSON format (this is an example; text after `//` is for explanation only and should not appear in the output): ```json { "vfx": "Fireball", // the name of the VFX element "instruction": "Add a round-shaped fireball into the person's hand.", // editing instruction "abstract": "Fireball on hand" // concise abstract with minimal words } ```

Figure 14. Prompt template for generating VFX editing instruction.

the first frame and the right image as the last frame, and to describe the dynamic transition between them. Finally, we use the generated transition description together with the paired images as input to first–last-frame-to-video models, such as Jimeng3.0 [8] and Wan-FLF2V-14B [40] to synthesize complete VFX videos.

Scalable data generation. To reduce the cost of largescale data synthesis with closed-source models, we adopt a scalable data expansion strategy. We first generate a moderate amount of high-quality data using the pipeline above and then fine-tune an instruction-based VFX generation model on an image-to-video backbone. The resulting model can produce consistent effects across different subjects given the same VFX instruction, enabling efficient large-scale data generation without repeatedly invoking expensive image editing and first–last-frame video models.

### C. Additional comparison

We provide additional visual comparisons with related baselines on OpenVFX dataset in Fig. 15. Please refer to our project page for dynamic results.

### D. Additional visual results

We present additional VFX visual results other than OpenVFX dataset, including those from Higgsfield and EffectData, as shown in Fig. 16, Fig. 17, Fig. 18, Fig. 19, Fig. 20. Please refer to our project page for dynamic results.

### E. Real-world VFX adaptation.

While our model was trained on synthetic VFX data, it still has the potential to generalize to real-world VFX transfer. We manually collected some real-world VFX movie clips and evaluated our model. As shown in Fig. 21, despite being trained only on synthetic data, it achieves successful zeroshot generalization to real VFX. However, a synthetic-toreal gap remains in fine-grained details. We plan to bridge this in future work by incorporating real-world VFX data.

### F. Failure cases

Despite the effectiveness of our proposed framework, there are still certain limitations and failure cases to be discussed. First, due to the inherent capacity limitations of the base model, our approach may struggle with extremely complex effects involving rapid or large-magnitude motions. As shown in Fig. 22(a), when the reference effect is flying and the subject undergoes a sudden take-off motion, the model occasionally fails to maintain subject fidelity (see red circle). Second, the framework assumes that the provided firstframe image is semantically compatible with the reference effect. When this assumption is violated, the transferred effect may become incoherent or semantically meaningless. As illustrated in Fig. 22(b), the reference effect is sunflower blooming, but the user supplies a dog image as the first frame. Under such mismatch, the generated video cannot faithfully manifest the intended VFX semantics. Note that such issue can be avoided by proper user interactions. Our future work will employ more powerful base models and incorporate diverse real-world data sources to further enhance generalization and realism.

[Figure 195]

###### Figure 15. Additional visual comparison with related baselines on OpenVFX dataset.

[Figure 196]

###### Figure 16. Additional visual results. In each grid, top row is the reference video, bottom row is our generated result.

[Figure 197]

###### Figure 17. Additional visual results. In each grid, top row is the reference video, bottom row is our generated result.

[Figure 198]

###### Figure 18. Additional visual results. In each grid, top row is the reference video, bottom row is our generated result.

[Figure 199]

###### Figure 19. Additional visual results. In each grid, top row is the reference video, bottom row is our generated result.

[Figure 200]

###### Figure 20. Additional visual results. In each grid, top row is the reference video, bottom row is our generated result.

[Figure 201]

Figure 21. Real-world VFX transfer examples.

[Figure 202]

Figure 22. Failure cases of our method. (a) Due to the limited capacity of the base model, subject fidelity may degrade under large and rapid motions (e.g., sudden upward take-off). (b) When the user provides a first-frame image that is semantically incompatible with the reference effect, the transferred effect becomes incoherent and fails to reflect the intended VFX.

