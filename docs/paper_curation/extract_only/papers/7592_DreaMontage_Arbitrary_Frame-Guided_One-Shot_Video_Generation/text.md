# arXiv:2512.21252v2[cs.CV]25Dec2025

[Figure 1]

[Figure 2]

## DreaMontage: Arbitrary Frame-Guided One-Shot Video Generation

Jiawei Liu∗†, Junqiao Li∗, Jiangfan Deng∗†, Gen Li∗, Siyu Zhou, Zetao Fang, Shanshan Lao, Zengde Deng, Jianing Zhu, Tingting Ma, Jiayi Li, Yunqiu Wang, Qian He, Xinglong Wu

Intelligence Creation Team, ByteDance

### Abstract

The "one-shot" technique represents a distinct and sophisticated aesthetic in filmmaking. However, its practical realization is often hindered by prohibitive costs and complex real-world constraints. Although emerging video generation models offer a virtual alternative, existing approaches typically rely on naïve clip concatenation, which frequently fails to maintain visual smoothness and temporal coherence. In this paper, we introduce DreaMontage, a comprehensive framework designed for arbitrary frame-guided generation, capable of synthesizing seamless, expressive, and long-duration one-shot videos from diverse user-provided inputs. To achieve this, we address the challenge through three primary dimensions. (i) We integrate a lightweight intermediate-conditioning mechanism into the DiT architecture. By employing an Adaptive Tuning strategy that effectively leverages base training data, we unlock robust arbitrary-frame control capabilities. (ii) To enhance visual fidelity and cinematic expressiveness, we curate a high-quality dataset and implement a Visual Expression SFT stage. In addressing critical issues such as subject motion rationality and transition smoothness, we apply a Tailored DPO scheme, which significantly improves the success rate and usability of the generated content. (iii) To facilitate the production of extended sequences, we design a Segment-wise Auto-Regressive (SAR) inference strategy that operates in a memoryefficient manner. Extensive experiments demonstrate that our approach achieves visually striking and seamlessly coherent one-shot effects while maintaining computational efficiency, empowering users to transform fragmented visual materials into vivid, cohesive one-shot cinematic experiences.

Date: December 25, 2025 Project Page: https://dreamontage.github.io/DreaMontage

### 1 Introduction

The "one-shot" (or "long take") technique represents a distinct aesthetic orientation in filmmaking, celebrated for its immersive continuity. However, executing a physical one-shot video incurs substantial costs in set design, staging, and post-production, while demanding exceptional professional skill. Furthermore, traditional filmmaking is strictly bound by physical space limitations, constraining imaginative scope and creative freedom. Recently, the burgeoning capabilities of video generation models [3, 6, 12, 16, 18, 27, 29] have opened new avenues for this task, potentially allowing users to leverage existing visual materials to produce coherent one-shot long videos. A prevalent practice involves concatenating multiple clips generated via first-last frame conditioning. However, this approach fails to fundamentally guarantee the smoothness and coherence of the video content, often resulting in disjointed transitions. Consequently, an arbitrary frame-conditioning video

∗Equal contributions †Corresponding authors: {liujiawei.cc22, dengjiangfan}@bytedance.com

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

t=16.0s

t=1.0s

Prompt Prompt Prompt Prompt

t=7.0s～10s

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

··· t～60.0s

[Figure 21]

Generated One-Shot, Long-Form Video

- Figure 1 DreaMontage: Flexible Dreams, Seamless Montage. Our model generates one-shot, long-form videos guided by arbitrary keyframes or video clips anchored at precise temporal locations.

generator has become a critical requirement. Ideally, such a system should produce seamless, cohesive, and engaging one-shot videos while affording precise temporal control.

We first formulate an explicit definition of this task: given a set of images and/or video clips alongside their temporal positions, the model generates a single continuous shot that obeys user instructions and ensures coherent transitions between conditioning contents. Typically, modern video generation models comprise a Variational Autoencoder (VAE) [11] and a Diffusion Transformer (DiT) [17]. From a technical perspective, equipping existing frameworks with such capabilities faces three primary obstacles. First, the 3D VAE [16] encoder usually adopts a causal mechanism with temporal down-sampling. Consequently, intermediate reference images cannot be intuitively represented in the latent space, hindering precise frame-level control. Second, the content between conditioning frames often exhibits significant semantic or visual shifts, posing severe challenges to the base model’s continuity and leading to undesirable abrupt cuts. Third, one-shot videos inherently require longer durations, whereas modern DiT-based models demand substantial memory and computational resources. Satisfying these overheads for long-video generation is non-trivial.

In this work, we propose DreaMontage, an arbitrary frame-guided one-shot video generator. To address the aforementioned challenges, we introduce innovations across three dimensions. As demonstrated in Fig. 2, first, an intermediate-conditioning mechanism is designed: we follow the channel-wise concatenation mode used in the typical image-to-video task and propose an extra Shared-RoPE condition strategy in the super-resolution DiT. By filtering the base model’s training data, we employ an Adaptive Training scheme to mitigate frame-wise misalignment, thereby equipping the generator with basic arbitrary frame-conditioning capabilities. Second, to enhance the expressiveness of one-shot generation, we carefully categorize video types and curate a small-scale, high-quality dataset. We then leverage Supervised Fine-Tuning (SFT) for visual expression. To tackle issues like abrupt cuts and unnatural subject motion, we construct specific pairwise datasets and apply Direct Preference Optimization (DPO) [21, 28, 31]. Finally, for inference, we design a Segment-wise Auto-Regressive (SAR) generation mechanism. This strategy decouples long video generation from strict computational and memory constraints while preserving the integrity of the one-shot content. Through this holistic design, DreaMontage empowers creators to orchestrate complex narratives where disparate visual assets merge into a unified sequence, maintaining high visual fidelity and temporal coherence.

| |
|---|

###### TRAINING PIPELINE INFERENCE PIPELINE

[Figure 22]

[Figure 23]

INPUTS: Ref Images/Videos, Insertion Times, Prompts

Pre-trained VFM （Base & SR DiT）

[Figure 24]

High-Res Ref Latents

[Figure 25]

[Figure 26]

Ref Images/Videos

Video VAE

[Figure 27]

Low-Res Ref Latents

Channel-Concat

Interm-Cond Adaptation （Base & SR DiT）

& Shared-RoPE

[Figure 28]

[Figure 29]

Prompts Text Encoder Text Embeddings

[Figure 30]

[Figure 31]

Manual-Collected

Visual Expression SFT

High-Quality Data

（Base DiT）

[Figure 32]

Auto-Regressive Long Video Generation

[Figure 33]

[Figure 34]

[Figure 35]

Base DiT

Specific Pair-Wise Data

Tailored DPO （Base DiT）

Text Embeddings

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Auto-Regressive Long Video

Final Models （Base & SR DiT）

SR DiT

Super-Resolution

- Figure 2 Overview of the DreaMontage. The left panel illustrates the multi-stage training pipeline, progressing from the Adaptive Tuning to the Visual Expression SFT and Tailored DPO. The right panel depicts the inference pipeline, where reference (condition) images/videos and rephrased prompts guide the generation process, supporting auto-regressive long-video generation.

In summary, our contributions are mainly three-fold:

- • We propose a simple yet efficient method for inserting intermediate conditions—including independent frames and video clips, unlocking arbitrary frame-conditioning capabilities within a DiT-based model at a relatively low cost.
- • We curate high-quality datasets and implement a progressive training pipeline involving SFT and DPO, empowering the model to generate seamless, coherent, and vivid one-shot videos.
- • We design a specialized Segment-wise Auto-Regressive (SAR) generation mechanism to enable long one-shot video production, striking an optimal balance between performance and efficiency.

### 2 Related Work

With the rapid advancement of diffusion models, the performance of AI-driven video generation has been continuously evolving. Commercial systems such as Sora [16], Veo [3], Kling [27], Seaweed [22] and Seedance [6], as well as open-source counterparts like Wan [29], Hunyuan Video [12], and CogVideoX [32], have demonstrated remarkable video quality and temporal consistency. Among various downstream tasks [1, 2, 4, 5, 7, 8, 14, 33], image-to-video (I2V) generation has attracted particular attention due to its controllability. However, relying solely on the first and/or last frame as the conditioning signal remains restrictive, lacking fine-grained control over motion evolution and intermediate transformations. To address this limitation, we extend the task boundaries, enabling the model to support combinations of multiple image and video conditions that can be placed at arbitrary positions along the temporal axis. This enhances the foundation model’s flexibility and controllability, allowing it to generate long take videos with richer visual variations.

In the I2V task, different foundation models adopt distinct mechanisms for injecting conditioning. Models such as Open-Sora [34], LTX-Video [9] and LongCat [26] employ timestep-based conditioning control, where the noise tokens and conditioning tokens are assigned different timesteps, enabling the model to learn to distinguish conditioning signals based on timestep information. Beyond timestep-based conditioning, Hunyuan Video further introduces in-context control for image conditions. It integrates a semantic image injection

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

[Figure 52]

[Figure 53]

F0 F1 F2 F3 F4 F5 F6

F0 F1 F2 F3 F4 F5 F6

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Video Latents

Video Latents

|[Figure 58]<br><br>re-sample<br><br>|
|---|

[Figure 59]

[Figure 60]

VideoVAE

VideoVAE

Encoder

Encoder

(2x temporal downsample)

(2x temporal downsample)

F0 F1,2 F3,4 F5,6

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

Aligned: Condition matches single-frame encoding at inference stage.

[Figure 72]

[Figure 73]

F2 C DiT

[Figure 74]

TemporalCondition latentMismatch:contains C DiT feature of multiple frames, not just F2.

F2

video condition

Channel-wise Concat

(a) I2V-like Interm-Cond Training (b) Proposed Interm-Cond Adaptation

- Figure 3 The Interm-Cond Adaptation strategy. (a) Due to the Causalty VAE’s temporal downsampling, an intermediate latent aggregates information from multiple frames, making it an imprecise condition for a specific timestamp. (b) To resolve this, we align the training distribution with inference. Each single condition frame (or the initial frame of a condition video) is re-encoded while the subsequent frames of the condition video are re-sampled from the latent distribution.

module to extract the semantic tokens from the input conditioning image, which are then injected through a combination of dual-stream and single-stream DiT blocks. This approach increases the length of the contextual token sequence, which not only raises computational costs but also limits the feasibility of using long sequences as conditioning inputs. Models such as Wan and Open-Sora Plan [13] prepare guidance frames that match the shape of the target video, with non-conditioning regions set to zero, and then feed them into the VideoVAE to obtain the latent representation. Consequently, regardless of how many conditioning frames are used, the model must encode a conditional video of the same length as the target video. Due to the use of Conv3D in the VideoVAE, this results in high computational cost and substantial redundant computation. In contrast, we adopt a more efficient approach by extracting the VAE latent from the condition images/videos and concatenating it with the noise latent along the channel dimension. This approach is simple and intuitive, and our experiments demonstrate that this channel-wise concatenation mechanism can effectively handle the multi-intermediate conditions.

### 3 Method

#### 3.1 Interm-Cond Adaptation

In this work, we adopt Seedance 1.0 [6], a video generation framework based on DiT architecture. As illustrated in the right half of Fig. 2, the pipeline starts from a 3D Video Variational Autoencoder (VideoVAE) compressing images and videos into a compact latent space. A text encoder is utilized to encode textual information, which is subsequently integrated into the DiT backbone via cross-attention. The generation process consists of two steps: first, a base DiT model generates video latents at a low resolution (480p); then, a super-resolution (SR) DiT model enhances the primary latents to the desired higher resolution (720p/1080p). In DreaMontage, our modifications are aimed at facilitating intermediate conditioning (Interm-Cond) and ensuring the efficacy of arbitrary-frame guided generation.

##### 3.1.1 Model Design

Base Model. In the original image-to-video task of the base model, the initial frame is channel-wise concatenated on the noise latent. We follow the same approach when conditioning intermediate frames. However, due to the

causality inherent in the VAE encoder, this channel-wise concatenation mode encounters an correspondence issue. As illustrated in Fig. 3-(a), in case of the independent frame conditioning, its VAE latent corresponds to multiple frames of the video to be generated. In the case of video conditioning (which can be deemed as multiple continuous frames) , the first frame faces a similar dilemma. In our work, we find that this problem can largely be solved through lightweight tuning with the Interm-Cond setup. Therefore, we construct a one-shot subset from the base model’s training data, and then devise an efficient Adaptive Tuning strategy to enable the generation from intermediately inserted images and videos (details provided in the next section).

Super-Resolution Model. In the super-resolution (SR), conditional frames are inserted as signals for highresolution guidance. However, in case of Interm-Cond, we frequently observe flickering and cross-frame color shifts. These artifacts are primarily attributed to the amplification of discrepancies between the "conditions" and the "generated contents" by the SR model, in which the conditions are channel-wise concatenated. In order to mitigate this problem, we propose Shared-RoPE: for each reference image, besides the channel-wise conditioning, an extra sequence-wise conditioning is applied. As shown in Fig. 4, the VAE latent is directly concatenated along the noise sequence, with the value of Rotary Position Embedding (RoPE) [25] being set the same as those at the corresponding position. Especially, in the case of video condition, the Shared-RoPE is only applied to the first frame to avoid excessive computational overhead.

##### 3.1.2 Adaptive Tuning

Token Concat

|Noised Latent<br><br>Condition Image<br><br>Condition Video<br><br>Zero Padding<br><br>|
|---|

Data Filtering. We start by constructing a one-shot subset from the base corpus through a data filtering pipeline. Specifically, a VLM-based scene detection model is adopted to exclude multi-shot videos. The cosine similarity is calcuated between the CLIP [20] features of the first and last frame for each video, filtering out those with high similarity to retain videos exhibiting large visual variations. We use Q-Align [30] to assess aesthetic scores, thereby eliminating data with low aesthetic quality. To ensure adequate motion intensity, a 2D optical flow predictor is applied to estimate the video’s motion strength. Additionally, the RTMPose [10] is utilized to filter out high-quality human-centric videos with clear pose structure. Through this meticulous data filtering process, we finally obtain a one-shot subset featuring large variations, strong motion, and high aesthetic quality. To emphasize the multi-action characters in the one-shot data above, we adopt an internally trained action recognizer to identify action intervals within each video, followed by a VLM captioner that generates dense descriptions for each action, resulting in structured, action-wise video annotations.

Channel Concat

[Figure 75]

[Figure 76]

SR-DiT

Shared-RoPE

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Positional

[Figure 84]

ids

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| |𝑡0|𝑡1| |𝑡2|𝑡3| |𝑡4| |𝑡5| |

Generated Video Tokens 𝑪𝒗𝟎

𝑪𝒊

Figure 4 The Shared-RoPE strategy for the superresolution model. In addition to channel-wise concatenation, we introduce a sequence-wise conditioning mechanism to eliminate artifacts. Condition frames are appended to the tail of the sequence while share the same RoPE value as the target frames they guide (e.g., Ci shares the RoPE of t1). In the case of video condition, this strategy is only applied to the first frame.

Training. During training, we propose an efficient way to support arbitrary conditions and make full use of the data. Primarily, the VAE latents of all videos are pre-calculated to ensure computational efficiency. As shown in Fig. 3-(b), for single image condition, the boundary frames of each action are treated as conditioning inputs. To enhance the input diversity and improve model robustness, we additionally sample random intermediate frames within each video to serve as supplementary conditions. These frames are re-encoded with the single-image mode by the VAE encoder. For video condition, segments with varying lengths are randomly extracted from the latents. It is noteworthy that these "latent segments" are not a

typical encoding result due to the causality of the VAE (the pre-calculated latents are acquired by causal encoding from the start of the whole video). We mitigate this problem by adopting an approximate approach: for each segment, the first frame is re-encoded while the subsequent frames are re-sampled from the latent distribution.

#### 3.2 Visual Expression SFT

To further enhance the model’s instruction-following ability and motion expressiveness, we perform Supervised Fine-Tuning on newly collected, category-balanced data with stronger one-shot characteristics.

Data Collection. We begin by conducting a fine-grained analysis of the model’s underperforming cases and categorizing them into five major classes: Camera Shots, Visual Effects, Sport, Spatial Perception, and Advanced Transitions. Each major class is further divided into multiple subclasses to ensure precise targeting. For instance, the "Camera Shots" class includes subclasses such as "Basic Camera Movements – Dolly In" and "Shooting Technique – First-Person View (FPV)", while the "Visual Effects" class covers "Generation – Light" and "Transformation – Animal Metamorphosis". This hierarchical classification system sets a clear direction for subsequent data collection. Based on the taxonomy, we collect a small-scale dataset consisting of videos for each subclass. Samples in this dataset are carefully selected to capture the core characteristics of each scenario while prioritizing high motion dynamics. Compared with the adaptive tuning data, videos for SFT have longer durations (up to 20 seconds) and cover a greater number of seamless scene transitions.

Training. The SFT is conducted upon the model weights from the previous adaptive tuning. We follow the similar training strategies and random condition settings in the adaptive tuning stage. After the SFT, the model acquires remarkable enhancement for motion dynamics and instruction-following of the video contents.

#### 3.3 Tailored DPO

The model encounters two typical issues when generating one-shot videos. On the one hand, if there is a significant difference between adjacent conditions, the video generated may experience undesired abrupt cuts. On the other hand, motion of subjects (such as human, animals and vehicles) in the video often deviates from physical laws and is prone to distortions. To mitigate these problems, we design tailored pipelines to produce contrastive pair-data and conduct specified Direct Preference Optimization (DPO) [21, 28] training, as shown in Fig. 5.

Abrupt Cuts. The basic idea of solving the "abrupt cuts" issue is guiding the model to learn a preference against cuts. We make two steps in achieving this goal. First, a vision-language model (VLM) is trained to recognize abrupts cuts within a video. Specifically, we generate a dataset comprising 10k video clips using the base model through the fist-last frame conditioning (the simplest special case of arbitray-conditioning), where each clip is categorized into five distinct levels based on "the severity of the cuts". In this process, the initial categorization is performed by GPT-4o [15] and subsequently refined by human annotators. This dataset is then utilized to fine-tune a VLM, enabling the model to effectively identify cuts. After acquisition of this "abrupt cut discriminator", we leverage it to construct pairs of "cut" and "non-cut" videos. Particularly, the base generation model is employed again to produce a large number of video "groups", each containing videos generated from the same prompt (first/last condition frames and texts) but with different seeds. The VLM is then applied to these groups to select the "best" and "worst" videos in terms of the cut severity mentioned above, thereby generating a substantial number of contrastive pairs. Finally, these pairs are utilized in DPO training, steering the generator to avoid producing videos that contain abrupt cuts.

Subject Motion. The approach to mitigating the subject motion problem follows a similar path. Primarily, we carefully identify and collect a set of common subjects (humans, vehicles, etc.) along with their frequently occurring problematic actions (jumping, rotating, turning, etc.) within the context of video generation. After that, a text-to-image generator Seedream [23] and a VLM-based rephraser are adopted to create a set of image-to-video prompts, each comprising a "first frame", a "last frame" and a "descriptive text". This prompt set is then utilized to generate video groups using the base generation model, adhering to the analogous methodology described above. In contrast to the abrupt cuts issue, we observe that current top vision-language models struggle to recognize the unreasonable motion problem. Therefore, we directly select high-contrast

Pipeline A: Abrupt Cuts

Pipeline B: Subject Motion

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

VLM Rephraser

GPT-4o & Human Anno

& T2I Generator

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Problematic Subject & Motion

Base Model

Cut VLM

Images & Prompts

[Figure 99]

Videos

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Human Anno

Different Seeds

Different Seeds

[Figure 117]

VLM Selects Best & Worst

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Video Groups

Video Groups

Pos/Neg Pairs

Pos/Neg Pairs

Base Model

[Figure 123]

Tailored DPO Training

[Figure 124]

| | |
|---|---|
|Policy 𝜋𝜃| |
| | |

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Initialized From Policy 𝜋𝜃

Final Base Model

[Figure 129]

| |[Figure 130]<br><br>|
|---|---|
|Ref Model 𝜋𝑟𝑒𝑓 (SFT)| |
| | |

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Positive ↑ Negative ↓

- Figure 5 Illustration of the Tailored DPO. To eliminate specific generation artifacts, we construct preference pairs via two distinct pipelines: Pipeline A addresses abrupt cuts by leveraging a trained VLM discriminator to automatically select positive/negative samples, whereas Pipeline B targets subject motion rationality through human-annotated screening of challenging cases. These pairs subsequently drive the DPO training to optimize the policy πθ against the reference model πref, ensuring smoother transitions and physically plausible motions.

pairs with the assistance of human annotators. The resulting pairs are then incorporated into DPO training, guiding the generator to produce more reasonable motion for the subjects.

Optimization Objective. Our Tailored DPO aims to strictly penalize the identified artifacts. We skip complex reward modeling and directly optimize the policy πθ against the reference model πref (initialized from the SFT weights). The objective is to minimize the standard DPO loss:

πθ(vw|c) πref(vw|c) − β log

πθ(vl|c) πref(vl|c)

LDPO = −E(c,v

w,vl)∼D log σ β log

, (1)

where c denotes the visual and textual conditions, and β controls the deviation from πref. By maximizing the margin between the likelihood of plausible samples (vw) and artifact-prone ones (vl), the model is effectively calibrated to produce coherent long-take videos while preserving the diversity learned during SFT.

#### 3.4 Segment-wise Auto-Regressive Generation

To address the challenge of making long one-shot videos that exceed the computational capacity of single-pass generation, we propose Segment-wise Auto-Regressive (SAR) generation strategy. Specifically, a variablelength sliding window is applied in the latent space to partition the target video into multiple consecutive segments, during which the user-provided conditions are treated as candidate boundaries. As the window slides, the current segment is terminated on the latest boundary once the window exceeds a predetermined maximum length. After the partitioning, an auto-regressive mechanism is used in successively generating these segments.

Formally, we define the generation of the n-th segment sn as a conditional mapping process parameterized by θ. Specifically, the model synthesizes the current latent sequence by conditioning on the tail latents of the preceding segment — extracted via a temporal operator τ(·) — alongside the set of local guidance signals Cn:

sn = Gθ (τ(sn−1),Cn), (2)

where Cn = {c(1)n ,...,c(nm)} denotes the collection of heterogeneous conditions (e.g., images or video clips) falling within the current window. By explicitly conditioning on τ(sn−1), the generator maintains rigorous pixel-level continuity across segment boundaries. This auto-regressive mechanism ensures that the distribution of the current segment is strictly aligned with the boundary context of the previous one, maintaining temporal consistency across the entire sequence.

The process above is repeated until all segments are generated. Finally, we fuse the overlapping latent frames between adjacent segments and obtain the final latent sequence. This sequence is then processed by a VAE decoder, yielding a naturally coherent long video with smooth visual continuity. It is noteworthy that our segment-wise generation is entirely performed in the latent space, yielding smoother transitions than pixel-based approaches. Moreover, benefiting from the model’s training during the Adaptive Tuning and the Visual Expression SFT, the model inherently maintains visual consistency when extending videos based on conditioning frames, avoiding artifacts such as frame flickering and abrupt jumps.

### 4 Experiment

In this section, we conduct a systematic evaluation of DreaMontage. First, we demonstrate our unique function of arbitrary frame-guided one-shot generation. As this capability is largely absent in existing models, we focus on qualitative demonstrations to highlight the narrative coherence and temporal control achieved by our methods. Second, by formulating first-last and multi-keyframe conditioning as special sub-cases, we benchmark against current SOTA models to evidence our superiority in visual and motion quality through rigorous quantitative and subjective comparisons. Finally, we conduct ablation studies to verify the critical contributions of optimizations in our work.

#### 4.1 Experimental Settings

Implementation Details. As demonstrated in previous sections, our training pipeline comprises three progressive stages. In the Adaptive Training stage, the filtered one-shot dataset contains 300k filtered video clips, on which the model is trained for 30k steps. In the Visual Expression SFT stage, we collect nearly 1k high-quality samples and the model is trained for another 15k steps. Finally, in the Tailored DPO stage, we construct 1k preference pairs for each task and train the model for 10k steps. For the super-resolution, we train two specialized models (based on the same DiT architecture) targeting 720p and 1080p resolutions, respectively.

Evaluation Dataset. To comprehensively assess the generation capabilities across varying complexities, we curate a large-scale internal test set covering a wide spectrum of common scenarios, encompassing diverse themes, styles, subjects, and semantic contents. The target video durations in this dataset range from 5 to 60 seconds, where each test sample is originally constructed with complex sequences of multiple image and video conditions. Furthermore, to make fair comparisons with existing models that lack support for arbitrary-frame conditioninig, we derive two specialized subsets from this foundational dataset: a Multi-Keyframe Benchmark, which excludes video insertion to focus on multi-keyframe conditioning, and a First-Last Frame Benchmark, which retains only the initial and final frames to align with the standard first-last conditioning setting.

Evaluation Metrics. To comprehensively evaluate perceptual quality and generation capabilities, we employ the Good/Same/Bad (GSB) protocol for human evaluation. In this pairwise comparison setting, human experts are presented with two videos side-by-side generated by the models under comparison, sharing identical inputs. To ensure fairness, the display order is randomized and blindly annotated. Evaluators are instructed to judge the results across four key dimensions: Visual Quality, Motion Effects, Prompt Following, and Overall Preference. For each pair and dimension, the evaluator assigns a rating of "Good" if the evaluated model outperforms the counterpart, "Bad" if it falls short, or "Same" if the quality is indistinguishable. For each comparison pair, by aggregating all rating data, we obtain the counts of Wins, Losses, and Ties. The final GSB

score is then calculated as (Wins - Losses) / (Wins + Losses + Ties). This method allows us to quantify the preference rate and capture subtle nuances in generation capabilities that automated metrics may overlook.

#### 4.2 One-shot Video Generation

We qualitatively evaluate the core capability of DreaMontage: generating long-duration, one-shot videos guided by arbitrary intermediate conditions. As visualized in Fig. 6, we present six distinct scenarios ranging from single-image constraints to complex mixed-media narratives. The frames marked with red borders represent user-provided conditions — anchored at precise timestamps.

Complex Narrative Construction. Rows (b) and (c) demonstrate the model’s ability to "connect the dots" between disparate visual concepts. In the Multi-keyframe setting (b), DreaMontage generates a seamless transition from a realistic train interior to a futuristic cyberpunk city, naturally handling the shattering of the window as a bridge between the two distinct styles. More strikingly, in row (c), the model executes a sophisticated "match cut" and continuous zoom: transitioning from an extreme close-up of an eye, flying through the pupil to reveal a street scene, and eventually shifting to a meadow. This proves that our method can handle drastic scale changes and semantic jumps without producing the morphing artifacts or ghosting effects common in baseline interpolation methods.

Video-Guided Transition and Extension. Beyond static images, DreaMontage excels at handling video clips as conditions. Row (d) illustrates a Video-condition (Transition) task, where the model takes a skiing clip as one of the conditions and logically evolves the environment from snow to sea, transforming the athlete’s motion into surfing (another condition) while preserving physical momentum. In the Extension case (e), given a starting clip of a cat riding a bike, the model hallucinates a complex interaction where the cat jumps from the bike to the back of a horse, effectively extending the narrative scope while maintaining character consistency.

Flexible Hybrid Control. Finally, row (f) showcases the Mix Image-Video Condition, where a static image and two dynamic clips are placed at the start, the middle and the end of the timeline respectively. DreaMontage successfully generates a coherent trajectory, depicting the rider taking off a helmet, ascending into the sky, and morphing into an astronaut, thereby translating a static storyboard into a vivid, multi-stage cinematic experience.

Please refer to our project page for more high-resolution video demonstrations.

#### 4.3 Comparison with existing Methods

To the best of our knowledge, no existing method supports inserting video clips as intermediate conditions. Instead, current approaches are restricted to multi-keyframe or first-last frame conditioning, both of which constitute special cases of our DreaMontage. Therefore, we benchmark against state-of-the-art models under these two respective sub-settings to ensure a fair comparison.

Multi-Keyframe Mode. We benchmark DreaMontage against leading models capable of handling multi-keyframe conditions: Vidu Q2 [24] and Pixverse V5 [19]. As shown in Fig. 7, DreaMontage achieves a commanding lead in overall preference, surpassing Vidu Q2 by +15.79% and Pixverse V5 by +28.95%. Notably, the most significant gap is observed in prompt following, where our model outperforms both competitors by a substantial margin of +23.68%. This confirms that our Adaptive Tuning and Visual Expression SFT enable DreaMontage to faithfully respect complex user instructions and semantic constraints, whereas counterparts often struggle to align the generated content with the text prompts in multi-condition settings. Regarding motion effects, our model demonstrates superior performance compared to Vidu Q2 (+7.89%) and remains competitive with Pixverse V5 (-2.63%), striking a robust balance between dynamic movement and stability. While visual quality shows a slight trade-off compared to Vidu Q2 (-2.63%) and a tie with Pixverse V5 (0.00%), the decisive advantage in overall user preference indicates that our model’s superior narrative coherence and instruction adherence are more critical for one-shot video generation tasks.

First-Last Frame Mode. We compare our approach with the widely recognized model Kling 2.5 [27]. Although Kling 2.5 is a formidable contender renowned for its high-definition generation, our model proves to be robust. The GSB statistics reveal a draw in visual quality (0.00%), suggesting that DreaMontage matches

|[Figure 135]|
|---|

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

- (a) Last-frame-condition. “[0, 5] An airplane flies quickly across the sky, leaving a smoke trail that spells out the word ‘Travel’”

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

- (e) Video-condition (Extension). “[3, 10] A horse walks in. The kitten jumps from a bike onto the horse's back. Camera pans left; the kitten rides the horse to a stream.”

- (b) Multi-keyframe-condition. “[0, 10] Train moves forward, static window frame; [10, 15] Window shatters into digital fragments; [15, 20] Camera flies through the window into a future cyberpunk city.”

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

- (c) Multi-keyframe-condition. “[0, 5] Zoom into extreme close-up of a man's eye; camera flies through the right pupil to reveal a man walking fast; [5, 10] Man walks forward; a golden butterfly enters. He follows it with his gaze until its wings fully obscure the lens; [10, 15] Butterfly flies away revealing a golden meadow; a boy runs in to chase it; [15, 20] The boy turns around, faces the camera directly, smiling.”
- (d) Video-condition (Transition). “[3, 7] Camera follows a man skiing. The snow ahead transitions into the sea. His snowboard transforms into a yellow surfboard, and he starts surfing.”

- (f) Mix Image-Video Condition. “[0, 3] The man takes off his helmet; [5, 10] The man rides the motorcycle into the sky, flies into outer space, and morphs into an astronaut.”

|[Figure 152]|
|---|

|[Figure 153]|
|---|

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

|[Figure 158]|
|---|

|[Figure 159]|
|---|

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

|[Figure 170]|
|---|

|[Figure 171]|
|---|

[Figure 172]

[Figure 173]

[Figure 174]

|[Figure 175]|
|---|

|[Figure 176]|
|---|

|[Figure 177]|
|---|

[Figure 178]

[Figure 179]

[Figure 180]

|[Figure 181]|
|---|

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

|[Figure 185]|
|---|

[Figure 186]

[Figure 187]

[Figure 188]

|[Figure 189]|
|---|

|[Figure 190]|
|---|

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

###### Figure 6 Qualitative visualization of arbitrary frame-guided generation. Red-bordered frames denote the user-provided conditions anchored at specific timestamps, while the other frames represent the generated content.

|0 0.2 0.4 0.6 0.8 1<br><br>Overall<br><br>Prompt Following<br><br>Motion Effects<br><br>Visual Quality<br><br>DreaMontage vs. Vidu Q2<br><br>(Multi-Keyframe Mode)<br><br>GSB: -2.63%<br><br>GSB: +7.89%<br><br>GSB: +23.68%<br><br>GSB: +15.79%| |0 0.2 0.4 0.6 0.8 1<br><br>DreaMontage vs. Pixverse V5<br><br>(Multi-Keyframe Mode)<br><br>GSB: 0.00%<br><br>GSB: -2.63%<br><br>GSB: +23.68%<br><br>GSB: +28.95%|0 0.2 0.4 0.6 0.8 1<br><br>DreaMontage vs. Kling 2.5<br><br>(First-Last Frame Mode)<br><br>GSB: 0.00%<br><br>GSB: +4.64%<br><br>GSB: +4.64%<br><br>GSB: +3.97%|
|---|---|---|---|

- Figure 7 Quantitative comparison with state-of-the-art methods. The green bars represent the percentage of cases where our result is visually superior, red favors the competitor, and gray indicates comparable quality. Reported numbers are the GSB scores.

the visual fidelity of top-tier commercial models. However, we secure consistent wins in both motion effects (+4.64%) and prompt following (also +4.64%). Consequently, DreaMontage achieves a higher overall preference (+3.97%), confirming that it excels not only in its unique multi-frame capability but also maintains SOTA-level performance in standard first-last frame generation task.

#### 4.4 Ablation Study

Effectiveness of Visual Expression SFT. As presented in Table 1, comparing the SFT-enhanced model against the Base model reveals that the primary contribution of this stage lies in motion dynamics. Specifically, while visual quality remains comparable (0.00%), the model achieves a remarkable improvement in motion effects (+24.58%). This indicates that the SFT stage effectively activates the model’s ability to generate vivid, high-magnitude movements. Coupled with a modest gain in prompt following (+5.93%), the Visual Expression SFT leads to a significant boost in overall preference (+20.34%), confirming its necessity for transforming basic generation capabilities into cinematic storytelling.

Effectiveness of Tailored DPO. To rigorously assess the impact of our Tailored DPO, we train a unified model using preference pairs from both the "Abrupt Cuts" and "Subject Motion" pipelines. The performance gains are evaluated separately on two specialized dimensions to isolate the improvements for each objective.

Performance on Abrupt Cuts. As shown in the middle section of Table 1, compared to the SFT baseline, the DPO-enhanced model achieves a GSB score of +12.59%. Qualitative analysis confirms that by penalizing negative samples with hard cuts, the model successfully learns to synthesize smoother narrative bridges between disparate conditioning frames, effectively resolving the transition discontinuity issue.

Performance on Subject Motion. On this specific task, the same DPO model yields a preference score of +13.44%. Visual inspection demonstrates that the model effectively generalizes the preference for natural motion, significantly mitigating anatomical distortions and unnatural movements (e.g., sliding feet or impossible limb rotations) that were occasionally observed in the SFT stage.

Shared-RoPE in Super-Resolution. The proposed Shared-RoPE mechanism achieves a dominant preference rate (+53.55%). As shown in the last line of Table 1, in the baseline setting (SR Base), the SR model frequently interprets the minor discrepancies between the low-res latent and the high-res condition as semantic conflicts, resulting in severe temporal flickering and color shifts. By enforcing alignment through shared rotary position embeddings, our method ensures that the high-resolution generation remains strictly aligned with the conditioning signals, thereby eliminating artifacts and ensuring temporal stability in the final output.

#### 4.5 Exploration on Applications

Beyond standard benchmarking, DreaMontage demonstrates immense potential in real-world creative workflows, particularly where precise control over narrative flow and visual consistency is paramount. We highlight three promising application scenarios enabled by our unique multi-condition architecture.

Table 1 Ablation study investigating the performance gains attributed to each optimization strategy. Numbers represent the GSB score (%).

Comparison Setting Visual Quality Motion Effects Prompt Following Overall SFT vs. Base (only adaptive training) 0.00 +24.58 +5.93 +20.34

Abrupt Cuts - +12.59 - +12.59 Subject Motion +13.44 - - +13.44

SFT+DPO vs. SFT

Shared-RoPE vs. SR Base +53.55 - - +53.55

Cinematic Trailer and Montage Creation. Traditional video generation models are limited to extending a single image or text prompt, often losing coherence over time. DreaMontage, however, acts as a "neural editor." By accepting a sequence of mixed inputs—such as static character concept art, keyframe illustrations, and pre-existing video clips—it can synthesize a cohesive cinematic trailer. Users can define the narrative structure by placing specific "anchor" frames or clips at different timelines, and our model seamlessly fills the gaps with semantically aligned transitions. This capability significantly streamlines the pre-visualization (pre-viz) process for filmmakers, allowing them to iterate on storyboards with motion and temporal dynamics instantly.

Infinite Long-Video Generation. While current models struggle with degradation in long-form generation, DreaMontage’s architecture naturally supports autoregressive video extension without quality decay. By treating the tail latents of a generated segment as the initial condition for the next, our model can produce theoretically infinite videos while maintaining strict character and stylistic consistency, making it ideal for creating continuous vlogs, nature documentaries, or endless loop animations.

Game Cutscenes and Dynamic Advertising. In the gaming and advertising industries, assets often exist in mixed formats—high-resolution promotional posters (images) and gameplay recordings (videos). DreaMontage can bridge these distinct modalities. For instance, it can animate a static product poster into a dynamic sequence that seamlessly transitions into actual usage footage, all guided by a unified text prompt. This "hybrid-condition" capability allows creators to repurpose existing static assets into engaging video content, reducing production costs while ensuring that the generated motion remains faithful to the brand’s visual style.

### 5 Conclusion

In this work, we present DreaMontage, a unified framework that democratizes the creation of cinematic one-shot videos. By empowering DiT architectures with arbitrary frame-guided conditioning and a memoryefficient Segment-wise Auto-Regressive strategy, we effectively overcome the rigidity of existing first-last frame paradigms. Our progressive training pipeline — integrating Adapive Tuning, Visual Expression SFT and Tailored DPO — further ensures both narrative coherence and physical plausibility. Extensive evaluations demonstrate that DreaMontage achieves superior controllability and visual fidelity compared to state-of-the-art baselines, providing a scalable and efficient solution for next-generation visual storytelling. We hope this work serves as a foundational step toward more flexible and expressive storytelling in the era of generative AI.

### References

- [1] Liyang Chen, Tianxiang Ma, Jiawei Liu, Bingchuan Li, Zhuowei Chen, Lijie Liu, Xu He, Gen Li, Qian He, and Zhiyong Wu. Humo: Human-centric video generation via collaborative multi-modal conditioning. arXiv preprint arXiv:2509.08519, 2025.

- [2] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In The Twelfth International Conference on Learning Representations, 2023.

- [3] Google DeepMind. Veo: Our most capable generative video model. Google DeepMind Blog, 2024. URL https://deepmind.google/technologies/veo/.

- [4] Wanquan Feng, Jiawei Liu, Pengqi Tu, Tianhao Qi, Mingzhen Sun, Tianxiang Ma, Songtao Zhao, Siyu Zhou, and Qian He. I2vcontrol-camera: Precise video camera control with adjustable motion strength. arXiv preprint arXiv:2411.06525, 2024.

- [5] Wanquan Feng, Tianhao Qi, Jiawei Liu, Mingzhen Sun, Pengqi Tu, Tianxiang Ma, Fei Dai, Songtao Zhao, Siyu Zhou, and Qian He. I2vcontrol: Disentangled and unified video motion synthesis control. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14051–14060, 2025.

- [6] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.

- [7] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023.

- [8] Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Chongyang Ma, Weiming Hu, Zhengjun Zha, Haibin Huang, Pengfei Wan, et al. I2v-adapter: A general image-to-video adapter for video diffusion models. CoRR, 2023.

- [9] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

- [10] Tao Jiang, Peng Lu, Li Zhang, Ningsheng Ma, Rui Han, Chengqi Lyu, Yining Li, and Kai Chen. Rtmpose: Real-time multi-person pose estimation based on mmpose. arXiv preprint arXiv:2303.07399, 2023.

- [11] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

- [12] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [13] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.

- [14] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. arXiv preprint arXiv:2502.11079, 2025.

- [15] OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/, 2024. Accessed: 2024-05-13.
- [16] OpenAI. Video generation models as world simulators. OpenAI Technical Report, 2024. URL https://openai

.com/research/video-generation-models-as-world-simulators.

- [17] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

- [18] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, et al. Open-sora 2.0: Training a commercial-level video generation model in $200 k. arXiv preprint arXiv:2503.09642, 2025.

- [19] Pixverse AI. Pixverse V5: Next-generation cinematographic video synthesis. Pixverse Blog, 2025. URL https://pixverse.ai/.

- [20] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

- [21] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

- [22] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, Feng Cheng, Feilong Zuo, Xuejiao Zeng, Ziyan Yang, Fangyuan Kong, Meng Wei, Zhiwu Qing, Fei Xiao, Tuyen Hoang, Siyu Zhang, Peihao Zhu, Qi Zhao, Jiangqiao Yan, Liangke Gui, Sheng Bi, Jiashi Li, Yuxi Ren, Rui Wang, Huixia Li, Xuefeng Xiao, Shu Liu, Feng Ling, Heng Zhang, Houmin Wei, Huafeng Kuang, Jerry Duncan, Junda Zhang, Junru Zheng, Li Sun, Manlin Zhang, Renfei Sun, Xiaobin Zhuang, Xiaojie Li, Xin Xia, Xuyan Chi, Yanghua Peng, Yuping Wang, Yuxuan Wang, Zhongkai Zhao, Zhuo Chen, Zuquan Song, Zhenheng Yang, Jiashi Feng, Jianchao Yang, and Lu Jiang. Seaweed-7b: Cost-effective training of video generation foundation model,

2025. URL https://arxiv.org/abs/2504.08685.

- [23] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.

- [24] ShengShu Technology. Vidu Q2: A highly consistent and dynamic video generation model. Vidu Project Page,

2025. URL https://www.vidu.studio/.

- [25] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

- [26] Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shijun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, et al. Longcat-video technical report. arXiv preprint arXiv:2510.22200, 2025.

- [27] Kuaishou Technology. Kling ai: High-quality video generation with physical world simulation. Kling AI Project Page, 2024. URL https://kling.kuaishou.com/.

- [28] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.

- [29] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [30] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023.

- [31] Teng Xiao, Yige Yuan, Huaisheng Zhu, Mingxiao Li, and Vasant G Honavar. Cal-dpo: Calibrated direct preference optimization for language model alignment. Advances in Neural Information Processing Systems, 37: 114289–114320, 2024.

- [32] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

- [33] Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, and Hang Li. Make pixels dance: High-dynamic video generation. arXiv:2311.10982, 2023.

- [34] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.

