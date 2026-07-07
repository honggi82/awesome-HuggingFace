# arXiv:2509.17627v1[cs.CV]22Sep2025

[Figure 1]

[Figure 2]

## OmniInsert: Mask-Free Video Insertion of Any Reference via Diffusion Transformer Models

##### Jinshu Chen ∗ Xinghui Li ∗† Xu Bai ∗ Tianxiang Ma Pengze Zhang Zhuowei Chen Gen Li Lijie Liu Songtao Zhao † Bingchuan Li ‡ Qian He

Intelligent Creation Lab, ByteDance

##### Abstract

Recent advances in video insertion based on diffusion models are impressive. However, existing methods rely on complex control signals but struggle with subject consistency, limiting their practical applicability. In this paper, we focus on the task of Mask-free Video Insertion, and aim to resolve three key challenges: data scarcity, subject–scene equilibrium, and insertion harmonization. To address the data scarcity, we propose a new data pipeline InsertPipe, constructing diverse cross-pair data automatically. Building upon our data pipeline, we develop OmniInsert, a novel unified framework for mask-free video insertion from both single and multiple subject references. Specifically, to maintain subject-scene equilibrium, we introduce a simple yet effective ConditionSpecific Feature Injection mechanism to distinctly inject multi-source conditions and propose a novel Progressive Training strategy that enables the model to balance feature injection from subjects and source video. Meanwhile, we design the Subject-Focused Loss to improve the detailed appearance of the subjects. To further enhance insertion harmonization, we propose an Insertive Preference Optimization methodology to optimize the model by simulating human preferences, and incorporate a Context-Aware Rephraser module during reference to seamlessly integrate the subject into the original scenes. To address the lack of a benchmark for the field, we introduce InsertBench, a comprehensive benchmark comprising diverse scenes with meticulously selected subjects. Evaluation on InsertBench indicates OmniInsert outperforms state-of-the-art closed-source commercial solutions. Code will be released.

Date: September 23, 2025 Project Page: https://phantom-video.github.io/OmniInsert/

##### 1 Introduction

The emergence of diffusion models [11, 38] has significantly advanced the field of video generation. In recent years, numerous open-source and commercial video generation models have been developed [8, 13, 46], providing powerful tools for creating video content. Beyond generation, video editing naturally arises as an extension, focusing on re-generate a reference video under fine-grained control signals like depth [6, 29], pose [15, 20, 35, 51], motion [7, 18, 27, 30] and point [5, 44]. Among various editing techniques, referenceimage-based editing [19, 23] plays a key role by providing explicit visual cues to guide the editing process. Specially, the Video Insertion (VI) task of inserting a reference subject into the source video has sparked considerable research interest, due to its significant potential for practical applications in film production, advertising, and artistic design.

* Equal Contribution, † Corresponding author, ‡ Project lead.

Real-world Human Insertion

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

Add a man kissing the woman

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Real-world Object Insertion

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Add a raccoon into the box on the table

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Figure 1 Showcase of OmniInsert in Real-World scenarios.

#### Reality-virtuality-mixed Insertion

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

Add a pink monster in the trunk

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

#### Multi-object Insertion

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

Figure 2 Showcase of OmniInsert in Reality-virtuality-mixed and Multi-object scenarios.

Prior approaches [23, 54] employ DDIM inversion for noise initialization to maintain the primary structure of the reference video and inject subject features during the denoising steps. However, these methods introduce an extra inference stage that doubles the computational time and frequently cause discontinuity and unnaturalness in the inserted subject. Recently, VideoAnyDoor [45] and GetInVideo [55] propose end-to-end video insertion frameworks to process reference images and condition videos simultaneously. Nevertheless, they rely on complex control signals (e.g., masks or points) to guide insertion position and motion and struggle with subject consistency. Very recently, some unified video editing works [28, 53] support video insertion based on instruction. However, they still suffer from issues of subject inconsistency and unnatural interactions, limiting practical applicability.

In this work, our focus is on the task of Mask-free Video Insertion (MVI), inserting user-defined characters into a reference video according to the customized prompt. The task poses several challenges, including: 1) Data Scarcity: Lacking paired videos before and after insertion along with corresponding subject references; 2) Subject-scene Equilibrium: Ensuring consistency of inserted subject while maintaining invariance of unedited areas of the reference video; 3) Insertion Harmonization: Achieving plausible positioning and motion of inserted subjects to ensure natural interactions.

To address the Data Scarcity, we propose a new data pipeline InsertPipe, producing training data consisting of reference subjects paired with appropriately edited videos and textual prompt. Specifically, we introduce three data curation pipelines: RealCapture Pipe, SynthGen Pipe, and SimInteract Pipe, aiming to enhance data diversity, thereby improving the model’s robustness across complex scenarios. The RealCapture Pipe leverages existing real-world videos, constructing paired videos through detection, tracking, and video erasing tools. Unlike prior methods [3, 16] that extract key frames from videos as reference subjects, we focus on constructing cross-video subject pairs to prevent copy-paste issue. To ensure comprehensive scene diversity, the SynthGen Pipe utilizes Large Language Model (LLM) [37] to generate varied prompts. These prompts are combined with techniques like image generation, image editing, video generation, and subject removal to automatically construct large-scale cross-pair datasets. For the scarcity of complex scene interaction data, we design the SimInteract Pipe to generate customized data based on the rendering engine.

Building upon our data pipeline, we develop OmniInsert, a novel unified framework for mask-free video insertion, supporting both single and multiple subject references with customized prompts. To maintain Subject-scene Equilibrium, we introduce three targeted improvements across model architecture, training strategy, and supervision. Firstly, we introduce a simple yet effective Condition-Specific Feature Injection (CFI) mechanism. It distinctly injects conditions from the source video and reference subject while preserving efficiency, requiring only minimal adjustments to the video foundation model. Secondly, we propose a novel Progressive Training (PT) strategy, which enables the model to balance multi-condition injection through multi-stage optimization. Thirdly, we design a Subject-Focused Loss (SL) to aid the model in focusing on capturing the detailed appearance of the subjects.

To enhance Insertion Harmonization in complex visual scenarios, we propose two complementary techniques. First, Insertive Preference Optimization (IPO) guides the model to learn context-aware insertion strategies using a curated set of paired videos that reflect human preferences across diverse scenes. Second, the ContextAware Rephraser (CAR) enriches user prompts at inference time by injecting fine-grained scene details (such as object textures, spatial layout, and interaction cues) into the instruction. This allows inserted content to better align with both the semantics and visual structure of the scene.

Through the aforementioned improvements, our approach enables effective insertion of customized subjects into reference videos, as illustrated in Fig. 1 and Fig. 2. To address the lack of a benchmark for MVI, we introduce a comprehensive benchmark, InsertBench, which consists of 120 videos paired with meticulously selected subjects (suitable for insertion in each video) and the corresponding prompts. Leveraging this benchmark, we conduct extensive evaluations demonstrating that OmniInsert achieves clear advantages over state-of-the-art commercial solutions, both quantitatively and qualitatively. Overall, our contributions are summarized as follows.

Technology. 1) We develop InsertPipe, a systematic data curation framework featuring multiple data pipelines to automatically generate high-quality and diverse data; 2) We propose OmniInsert, a unified mask-free

Real Pipe: Cross-pair Video + Subject Remove Rendering Pipe

[Figure 77]

Track-Erase Toolkit

[Figure 78]

[Figure 79]

[Figure 80]

###### Houdini

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Rendering Tool

###### SpatialLM

[Figure 87]

Cross-pairVideo

###### VLM

###### VLM Captioner DINO

[Figure 88]

Scence Layout Provider

…

[Figure 89]

###### Filter

SameID

…

###### SAM

Minimax Remover

[Figure 90]

[Figure 91]

Target Video

Source Video

“Describe the detailed categories of the main subje ct in the video…”

Detector

Segmenter

[Figure 92]

[Figure 93]

[Figure 94]

∙Video Quality ?

[Figure 95]

[Figure 96]

…

###### Houdini

∙Artifacts ?

[Figure 97]

∙Residual subject? …

[Figure 98]

###### Rendering Tool

…

Source Video Reference Image

…

3D Stored Assets

Reference Image (Stored Assets)

Target Video

- Gen Pipe 1: T2I + I2V + Subject Remove

- Gen Pipe 2: Video Editing

✂

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

###### Flux

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

###### Text-to-Image Model

###### VLM

###### VLM

On the lawn In the desert On the ship

A man A hedgehog An apple

###### Seedance

Track-Erase Toolkit

###### Filter

###### Filter

Image-to-Video Model

“Generate a diptych with two side−by−side images of same hedgehog. Left:A close−up of the hedgehog;

∙IP Similarity ?

…

…

∙Video Quality ?

∙Image Quality?

∙Artifacts ?

Subject Bucket Scence Bucket

…

…

∙Plausible Scene?

∙Residual subject?

Right:Thehedgehogsitsonaship…” Reference Image

Target Video

###### Source Video

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

###### VLM

###### VLM

###### Captioner

Step1X-Edit

###### DINO

Video-Editing Model

###### Filter

Frame Interpolation

Image Editing Model

“Imagine that the main object appears in the given video, return the detailed description.”

Detector

∙Video Quality ?

∙Artifacts ?

Reference Image

…

∙Residual subject?

…

…

∙Motion Coherence

KeyFrame

Source Video

Target Video

Coarse Results

Detection Bounding Box

Bounding Box Frames

Figure 3 Overview of InsertPipe. It consists of three data construction pipelines: Real Pipe, Rendering Pipe, and Gen Pipe.

architecture capable of seamlessly inserting both single and multiple reference subjects into videos; 3) We introduce InsertBench, a comprehensive benchmark tailored for MVI task.

Significance. 1) OmniInsert demonstrates superior generation quality, bridging the gap between academic research and commercial-grade applications; 2) We present a comprehensive study of the MVI task—including data, model, and benchmark—will be publicly released to support future research and development.

##### 2 Related Work

Video Foundation Model. The development of diffusion models [11] has significantly advanced video foundation model research. Early models [2, 9] extended pre-trained Text-to-Image (T2I) models for continuous video generation by adding temporal modules. VDM [12] extends 2D U-Net to 3D, while AnimateDiff [9] integrates 1D temporal attention into 2D spatial blocks for efficiency. These strategies allow them to leverage powerful image priors, but their capabilities can be constrained by the original image-centric architecture. Recently, the emergence of Diffusion Transformer(DiT) [38]-based methods for video generation has exhibited superior performance in quality and consistency. These methods treat video as a sequence of spatiotemporal patches, processing them in a unified manner with a Transformer. These powerful scaling transformers [8, 22, 33, 34, 46, 52] enable generating longer and higher-quality videos, paving the way for various downstream video generation tasks.

Video Insertion. Video Insertion (VI) aims to insert a user-provided reference subject into source video while maintaining the consistency of unedited regions. Prior approaches [23, 54] employ DDIM inversion to initialize the generation noise based on the reference video and inject subject features during the denoising steps. Due to they inherently require an additional stage for inversion, computational overhead and inference costs increase significantly. VideoAnyDoor [45] proposes an end-to-end framework that precisely modifies both content and motion according to user-specified mask and point conditions. GetInVideo [55] employs a 3D full-attention diffusion transformer architecture to jointly process reference images, condition videos, and masks, maintaining temporal coherence. However, these methods rely on complex control signals but struggle with subject consistency. Recently, some unified video editing works [28, 53] support mask-free video insertion but still encounter subject infidelity and incoherent interactions in complex scenarios. Consider commercial software [21, 39] currently maintains state-of-the-art in MVI capabilities, it’s critical to bridge the

gap between academic research and commercialization.

##### 3 Preliminary

The Diffusion Transformer (DiT) [38] model employs a transformer as the denoising network to refine diffusion latent. Our method inherits the video diffusion transformers trained using Flow Matching [31], which conducts the forward process by linearly interpolating between noise and data in a straight line. At the time step t, latent zt is defined as: zt = (1 − t)z0 + tϵ, where z0 is the clean video, and ϵ ∼ N(0,1) is the Gaussian noise. The model is trained to directly regress the target velocity:

LFM = E[∥(z0 − ϵ) − vθ(zt,t,y)∥2], (1)

where vθ refers to the diffusion model output and y denotes condition. Our diffusion model is composed of stacked DiT blocks, which perform 2D self-attention for spatial modeling and 3D self-attention for spatiotemporal fusion.

##### 4 Methodology

Given reference subjects and a source video, we aim to generate a new composited video showcasing a natural interaction between the inserted contents and the original scenes. We first introduce a new data pipeline InsertPipe to construct diverse paired data (Sec. 4.1). Building upon our data pipeline, we develop a novel framework OmniInsert employing a core Condition-Specific Feature Injection (CFI) module for mask-free video insertion (Sec. 4.2). To further enhance insertion stability and quality, we design a Progressive Training (PT) strategy incorporating the Subject-Focused Loss (SL) and Insertive Preference Optimization (IPO) during training (Sec. 4.3). Additionally, we propose a Context-Aware Rephraser (CAR) module during inference to strengthen insertion harmonization (Sec. 4.4).

###### 4.1 InsertPipe

To address the challenge of data scarcity, we introduce a comprehensive data pipeline InsertPipe to produce diverse data for the MVI task, encompassing RealCapture Pipe, SynthGen Pipe, and SimInteract Pipe, as shown in Fig. 3. The resulting training samples are formatted as {prompt, reference images, source video, target video}.

RealCapture Pipe. Using long videos from [48] and proprietary data, we segment videos into single-scene clips as target videos with AutoShot/PySceneDetect. The Vision-Language Model (VLM) [37] then captions these clips, detailing subject appearance, scenes, and interactions. These captions are processed by LLM [37] to extract subject categories/appearances for detection/tracking prompts [41], generating mask sequences. Furthermore, we apply video erasing techniques [56] to remove target subjects to create source videos, with a VLM-based filtering mechanism that avoids visual artifacts of inpainting. For subject conditions, we construct cross-video pairs (avoiding same-video keyframes [3, 16]) to prevent copy-paste issues. Subjects are matched across CLIP [42] and facial embeddings [4], discarding pairs with extreme similarity (copy-paste) or dissimilarity (inconsistency).

SynthGen Pipe. Considering the limited scene diversity, we introduce a generation-based data pipeline comprising two complementary sub-flows. 1) We first meticulously construct subject and scene buckets with 300 categories and 1000 settings, respectively. Then, diverse subject-scene pairs generated by LLM are combined with text templates to produce cross-pair references via T2I models [24, 25]. To ensure consistency, VLM is further employed to score the appearance consistency and detail preservation within each pair. Subsequently, we synthesize target videos depicting natural interactions using the Image-to-Video (I2V) foundation models [8], while source videos are derived using the tracking-erase toolkit with the VLM filtering strategy. 2) Given videos depict empty scenes, we employ VLM to determine suitable inserted subjects and generate corresponding prompts. Next, we sample key frames and apply instruction-based image editing methods [32] to get target coarse positions. Leveraging temporal interpolation and video inpainting to synthesize the target videos, with VLM filtering ensuring temporal coherence and subject consistency.

Training Inference

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

′A woolly sheep grazes peacefully on the vast green meadow ...′

[Figure 132]

′Add this dog into this trash.′

Target Video

Source Video*

Reference Image

Source Video

Reference Image

User Instruction

Prompt

[Figure 133]

VAE Encoder

###### ContextAware Rephraser

Condition-Specific Feature Injection

′A happy-looking Shiba Inu with a red collar is inside the trash can,which is obvious .…′

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

⊗ ⊗

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

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

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

[Figure 181]

VAE Encoder

Prompt

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Channel

0 0 0 0 0 1

Spatial

[Figure 194]

Diffusion Model

Temporal

[Figure 195]

[Figure 196]

[Figure 197]

DiT Block

LoRA

VAE Decoder

…

[Figure 198]

[Figure 199]

DiT Block LoRA

[Figure 200]

[Figure 201]

[Figure 202]

Target Predicted

Subject-Focused Loss*

###### IPO Loss*

Figure 4 Overview of OmniInsert. Note that the parts marked with * are only enabled in specific phases.

SimInteract Pipe. Despite enhanced diversity, complex subject-scene interactions (e.g., a man waving behind slowly opening doors) remain scarce. To address this, we develop a rendering pipeline based on [14]. Specifically, we construct extensive asset libraries for subjects and scenes in the rendering environment. SpatialLM [36] then generates 350 layout priors to drive stochastic asset placement. Leveraging rigged assets with predefined motion bindings, we synthesize interactions by retrieving motion library animations through predefined prompts. Finally, we can leverage strategically positioned cameras to render target videos while paired source videos are derived via subject removal under identical cameras.

###### 4.2 OmniInsert Framework

Mask-free Video insertion is a challenging task that requires accurate preservation of both subject identity and background consistency. A straightforward solution is to inject VAE features of the references along the temporal dimension or to concatenate reference visual tokens after patchification. However, such approaches impose high computational overhead, and fail to account for the distinct alignment requirements of different conditions: reference videos demand frame-wise alignment with latent noise, while subject references require full temporal feature interaction.

To address these challenges, we propose OmniInsert, a novel end-to-end framework for mask-free video insertion using single or multiple subject references guided by text prompts, as shown in Fig. 4. At its core, we introduce a simple yet effective Condition-Specific Feature Injection (CFI) mechanism that injects both video

and subject features in a unified yet efficient manner, tailored to their respective temporal alignment needs. Condition-Specific Feature Injection. CFI adopts targeted designs in both the injection mechanism and channel-level flags to accommodate the distinct characteristics of video and object conditions. Specifically, for the video condition, source videos are concatenated with latent noise along the channel dimension, accompanied by dedicated flags, formally expressed as:

zVidt = Concat([zTt ,zS,fS],dim = 1), (2)

where noisy target video latent zTt = (1 − t)zT + tϵ, zT ∈ Rf×c×h×w denotes the clean target video latent. zS ∈ Rf×c×h×w represents reference source video latent, fS ∈ Rf×1×h×w is an all-zero vector. The choice of channel-wise concatenation aligns well with the background condition, facilitating effective spatial alignment.

In contrast, for the subject condition, latents are concatenated along the temporal dimension to better capture dynamic changes across frames. As shown in Fig. 4, the subject latent is constructed as:

zSubt = Concat([zIt,zI,fI],dim = 1), (3)

where zI,zIt ∈ Rn×c×h×w denote the clean and noisy subject latents, with zIt = (1 − t)zI + tϵ, and n denotes the number of inserted subjects. fI ∈ R1×1×h×w is an all-one flag map. The overall diffusion input is finally constructed by concatenating the video and subject condition latents along the frame dimension: zt = Concat([zVidt ,zSubt ],dim = 0). This temporal concatenation effectively captures subject motion and continuity.

Thanks to our differentiated condition injection strategies, CFI effectively achieves seamless subject insertion and background preservation while maintaining network efficiency. Furthermore, we integrate the LoRA mechanism into the DiT blocks to avoid expensive full-parameter updates, thereby preserving the model’s original text-alignment capabilities and visual fidelity.

###### 4.3 OmniInsert Training Pipeline

Progressive Training. A significant challenge in training video insertion models lies in the inherent imbalance of learning difficulty: preserving the source video is mainly a copy-paste task, which is much simpler than the complex temporal and spatial modeling involved in dynamic subject insertion. Consequently, naive single-stage training tends to bias the model toward the simpler task of background preservation, often resulting in failed object insertion, as illustrated in Fig 7. To mitigate these issues, we introduce a four-stage progressive training strategy:

- Phase 1. We begin by training the model to inject reference subjects based on subject features and text prompts, while discarding the source video condition. This isolates subject modeling and helps the model develop strong subject representation and motion generation capabilities.
- Phase 2. Based on the phase 1 model, we introduce the source video and conduct pretraining on the complete MVI task across the full dataset. This enables the model to acquire an initial capacity for aligning subjects with video backgrounds. However, models at this stage often suffer from identity inconsistency and show limited performance in complex scenes.
- Phase 3. To address the above limitations, we conduct a refinement stage using a curated dataset composed of high-fidelity portraits and synthetic renderings. A short period of fine-tuning on this data significantly enhances identity preservation and improves the model’s robustness in visually complex environments.
- Phase 4. To further reduce physical implausibilities (e.g., unnatural poses, model penetration) and visual artifacts, we develop Insertion Preference Optimization (IPO), a fine-tuning stage inspired by [40]. By leveraging a small number of human-annotated preference pairs, IPO guides the model toward more visually and physically plausible insertions, yielding notable improvements in generation quality.

Through progressive optimization, the model achieves balanced feature injection from subjects and source video, enhancing insertion stability and fidelity. Notably, the success of this progressive strategy also critically depends on carefully designed losses, which are described below.

|Method|Subject Consistency<br><br>|Text-Video Alignment|Video Quality|
|---|---|---|---|
| |CLIP-I∗ ↑ DINO-I∗ ↑ FaceSim ↑|ViCLIP-T ↑|Dynamic ↑ Image-Quality ↑ Aesthetics ↑ Consistency ↑|

Pika-Pro [39] 0.682 0.543 0.422 24.721 0.873 0.655 0.524 0.910 Kling [21] 0.664 0.513 0.539 23.091 0.823 0.667 0.521 0.921 Ours 0.745 0.639 0.488 25.945 0.825 0.704 0.556 0.930

Table 1 Quantitative comparisons with baseline methods, and ∗ indicates the consistency of the segmented subject area.

Method ConsistencySubject ↑ AlignPromptwith↑ RationalityInsertion ↑ ComprehensiveEvaluation ↑ Pika-Pro [39] 5.08% 9.00% 10.67% 8.58% Kling [21] 29.42% 22.67% 24.92% 23.08%

Ours 65.50% 68.33% 64.41% 68.34%

Table 2 User study with baseline methods.

Subject-Focused Loss. In Phases 1–3, the flow matching loss (LFM) serves as the primary training objective. However, since it treats all regions of the synthesized video equally, it often fails to maintain consistency for subjects that occupy relatively small spatial areas. To address this issue, we propose the Subject-Focused Loss (SL), a modification of LFM that directs the model’s attention more strongly toward the subject regions. Formally, the loss is defined as:

LSL = E[∥M · [(z0 − ϵ) − Vθ(zt,t,y)]∥2], (4)

where M denotes spatially downsampled masks derived from tracking. Thus, the overall loss L for Phase 1-3 is:

L = λ1LFM + λ2LSL. (5)

IPO Loss. In Phase 4, we propose a preference optimization mechanism to improve visual stability and physical plausibility. Specifically, the well-trained phase-3 model serves as the reference model πref, while an additional LoRA module is initialized as the trainable policy πθ. For preference data construction, we curate paired samples (preferred yw and dispreferred yl) through human selection of model inference outputs. Similar to [50], the loss is:

LIPO = LDPO + λ · E (−log πθ(yl|x) − γ)2 , (6)

πθ(yw|x) πref(yw|x) − β log

πθ(yl|x) πref(yl|x)

, (7)

LDPO = −E log σ β log

where x denotes the model input for brevity and σ represents the standard Sigmoid function. Notably, we find that a limited dataset of only 500 preference pairs suffices to achieve substantial performance gains at this stage. Note that the IPO loss is exclusively applied in phase 4 as the sole objective function.

###### 4.4 OmniInsert Inference Pipeline

Classifier-free guidance [10] balances sample quality and diversity in diffusion models through joint conditional and unconditional training. During inference, we design a joint classifier-free guidance to balance multiple conditions:

vˆθ(zt, cp, ci, cv) = vθ(zt, ∅, ∅, ∅)

- + S1 · (vθ(zt, cp, ∅, ∅) − vθ(zt, ∅, ∅, ∅))
- + S2 · (vθ(zt, cp, ci, ∅) − vθ(zt, cp, ∅, ∅))
- + S3 · (vθ(zt, cp, ci, cv) − vθ(zt, cp, ci, ∅)) ,

(8)

where cp, ci, cv represent prompt condition, reference subjects, and source video. S1, S2 and S3 are guidance scales.

Context-Aware Rephraser. To enhance the coherence of subject insertion in complex visual scenes, we introduce a Context-Aware Rephraser (CAR) module at inference time, as shown in Fig 4. CAR leverages the

VLM to generate detailed, context-aware prompts that guide the model toward more seamless and plausible integration of the inserted subject into the source scene. Specifically, CAR first prompts the VLM to produce fine-grained descriptions of both the source environment and the reference subject. It then provides the VLM with creative constraints—such as preserving the original video setting, encouraging imaginative yet coherent insertion of the subject, and maintaining spatial consistency in terms of size and position, etc. As a result, CAR produces prompts that more accurately reflect the user’s creative intent, particularly in terms of visual effects (VFX), ultimately improving the realism and harmony of insertions achieved by OmniInsert.

##### 5 Experiments

###### 5.1 Setup

InsertBench. We present InsertBench, a comprehensive benchmark for evaluating MVI capabilities. The benchmark comprises 120 videos spanning various environments, including indoor, natural landscapes, wearable scenarios, and even animated scenes. Each video is approximately 5 seconds long with 121 frames, paired with meticulously selected subjects to ensure visually compatible insertion. Such a diverse benchmark would be beneficial for the community.

Implementation Details. The training protocol comprises four phases: 1) 70k iterations for subject-to-video task in phase 1; 2) 30k iterations for MVI task pretraining in phase 2; 3) 10k iterations for model refinement in phase 3; 4) 8k iterations for preference optimization in phase 4. The total computational resources consumed approximately 7000 GPU-hours on A100. During inference, we use the Euler sampler with 50 steps and set S1,S2,S3 to 7.5, 3.5 and 1.5, respectively. For fair comparison, all experiments are conducted at the resolution of 480p. Please refer to the supplementary materials for more details.

Baselines. For the MVI task, the existing available state-of-the-art methods are closed-source commercial software. Therefore, we evaluate and compare the latest capabilities of Pika-Pro [39] and Kling [21].

Evaluation Metrics. We assess three dimensions: CLIP-I, DINO-I and FaceSim [4] for Subject Consistency; ViCLIP-T [49] for Text-Video Alignment; Dynamic Quality, Image-Quality, Aesthetics and Consistency [17] for Video Quality.

###### 5.2 Qualitative Analysis

Fig. 5 and Fig. 6 present visual comparisons between our method and baseline approaches. OmniInsert demonstrates superior subject/background consistency, whereas baseline methods struggle to balance multi-source feature injection. Specifically, Pika exhibits significant deficiencies in subject fidelity and insertion plausibility, particularly in wearable and human-centric scenarios. Kling suffers from obvious scene inconsistency while encountering copy-paste issues in portrait scenarios. Compared with baselines, our method ensures the natural integration of inserted subjects within source scenes. Since the compared methods lack multi-reference support, additional results of multiple subjects are provided in the supplementary materials.

###### 5.3 Quantitative Comparisons

Metric Evaluation. For subject consistency, we uniformly sample 10 frames per video and compute their similarity to reference subjects. For prompt following, we measure text-video cosine similarity. Tab. 1 shows OmniInsert leads in metrics of CLIP-I∗, DINO-I∗ and ViCLIP-T. Although Kling achieves higher face similarity in portrait scenarios, they encounter copy-paste issues (1st line in Fig. 5 and Fig. 6). Moreover, our method shows superiority in video quality.

User Study. We conduct a user study to evaluate the generation quality of our model. We assigned 40 test samples and invited 30 volunteers to select the most preferred result under four criteria: subject consistency, text alignment, insertion rationality and comprehensive evaluation. Tab. 2 shows that OmniInsert achieves significant advantages.

Add a woman cuddling up to the man

| |[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]|[Figure 206]| |
|---|---|---|---|

|[Figure 207]|
|---|

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

𝑶𝒖𝒓𝒔

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

𝑲𝒍𝒊𝒏𝒈

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

𝑷𝒊𝒌𝒂-𝑷𝒓𝒐

Add a pair of sunglasses to the man

|[Figure 220]<br><br>[Figure 221]| | |[Figure 222]<br><br>[Figure 223]|
|---|---|---|---|

|[Figure 224]|
|---|

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

𝑶𝒖𝒓𝒔

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

𝑲𝒍𝒊𝒏𝒈

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

𝑷𝒊𝒌𝒂-𝑷𝒓𝒐

Add a bear standing next to the sheep

|[Figure 237]| |[Figure 238]|[Figure 239]<br><br>[Figure 240]|
|---|---|---|---|

|[Figure 241]|
|---|

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

𝑶𝒖𝒓𝒔

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

𝑲𝒍𝒊𝒏𝒈

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

𝑷𝒊𝒌𝒂-𝑷𝒓𝒐

- Figure 5 Qualitative comparisons I with state-of-the-art methods. Please zoom in for more details.

Add a woman standing by the flower

|[Figure 254]|
|---|

| |[Figure 255]<br><br>[Figure 256]<br><br>[Figure 257]|[Figure 258]| |
|---|---|---|---|

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

𝑶𝒖𝒓𝒔

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

𝑲𝒍𝒊𝒏𝒈

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

𝑷𝒊𝒌𝒂-𝑷𝒓𝒐

Add a dog sitting inside the trolley

|[Figure 271]|
|---|

|[Figure 272]| |[Figure 273]<br><br>[Figure 274]<br><br>[Figure 275]| |
|---|---|---|---|

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

𝑶𝒖𝒓𝒔

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

𝑲𝒍𝒊𝒏𝒈

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

𝑷𝒊𝒌𝒂-𝑷𝒓𝒐

Add a Wombat lying on a sea turtle's back

|[Figure 288]<br><br>[Figure 289]| | |[Figure 290]<br><br>[Figure 291]|
|---|---|---|---|

|[Figure 292]|
|---|

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

𝑶𝒖𝒓𝒔

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

𝑲𝒍𝒊𝒏𝒈

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

𝑷𝒊𝒌𝒂-𝑷𝒓𝒐

- Figure 6 Qualitative comparisons II with state-of-the-art methods. Please zoom in for more details.

###### 5.4 Ablation Studies

To demonstrate the effectiveness of our proposed method, we conducted the following ablation studies: a). Directly training the MVI task on the full dataset without the Progressive Training (PT) method of phases 1-3 (denoted as w/o PT). b). Not using the Subject-Focused Loss (SL) during training (denoted as w/o SL). c). Simply using the user instruction without Context-Aware Rephraser (CAR) during inference (denoted as w/o CAR). d). Not using the Insertion Preference Optimization (IPO) phase during training (denoted as w/o IPO).

As shown in Fig. 7 and Table. 3, we demonstrate the effectiveness of our proposed modules: a). PT substantially facilitates subject-scene equilibrium through multi-stage optimization, enhancing insertion stability; b). SL guides the model to focus on more critical yet harder-to-fit subject regions, improving subject preservation; c). CAR generates rich descriptions of subject-scene interactions, facilitating more natural compositions; d). IPO effectively enhances insertion rationality while mitigating visual artifacts.

|(𝑒). 𝑶𝒖𝒓𝒔|
|---|

|(𝑎). 𝑤/𝑜 𝑷𝑻|
|---|

|(𝑏). 𝑤/𝑜 𝑺𝑳|
|---|

|(𝑑). 𝑤/𝑜 𝑰𝑷𝑶|
|---|

|(𝑐). 𝑤/𝑜 𝑪𝑨𝑹|
|---|

|𝑅𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒 𝐼𝑚𝑎𝑔𝑒|
|---|

|𝑆𝑜𝑢𝑟𝑐𝑒 𝑉𝑖𝑑𝑒𝑜|
|---|

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

###### Figure 7 Ablation results of our method.

Method CLIP-I∗ ↑ DINO-I∗ ↑ ViCLIP-T ↑ Aesthetic ↑

- a) w/o PT 0.642 0.533 22.883 0.520

- b) w/o SL 0.657 0.587 25.908 0.542

- c) w/o CAR 0.689 0.609 23.052 0.532

- d) w/o IPO 0.732 0.625 24.537 0.507

- e) Ours 0.745 0.639 25.945 0.556 Table 3 Ablation study of our method.

##### 6 Conclusion

This paper presents a comprehensive study of the task, i.e., Mask-free Video Insertion. To address the data scarcity, we introduce a new data pipe InsertPipe to produce diverse paired data. Furthermore, we develop a novel unified framework OmniInsert, leveraging a Condition-Specific Feature Injection mechanism and incorporating a Progressive Training strategy with Subject-Focused Loss to achieve subject-scene equilibrium. Additionally, we propose an Insertive Preference Optimization methodology and a Context-Aware Rephraser module to enhance insertion harmonization. Finally, to address the lack of a benchmark, we build InsertBench and experiments show that OmniInsert outperforms closed-source commercial solutions.

##### References

- [1] Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pages 4447–4455. PMLR, 2024.

- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

- [3] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey Tulyakov. Multi-subject open-set personalization in video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6099–6110, 2025.

- [4] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2019.

- [5] Yufan Deng, Ruida Wang, Yuhao Zhang, Yu-Wing Tai, and Chi-Keung Tang. Dragvideo: Interactive drag-style video editing. In European Conference on Computer Vision, pages 183–199. Springer, 2024.

- [6] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7346–7356, 2023.

- [7] Ruoyu Feng, Wenming Weng, Yanhui Wang, Yuhui Yuan, Jianmin Bao, Chong Luo, Zhibo Chen, and Baining Guo. Ccedit: Creative and controllable video editing via diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6712–6722, 2024.

- [8] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.

- [9] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

- [10] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

- [12] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022.

- [13] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

- [14] houdini. houdini-engine. https://www.sidefx.com/products/houdini-engine/, 2025.
- [15] Li Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024.

- [16] Yuzhou Huang, Ziyang Yuan, Quande Liu, Qiulin Wang, Xintao Wang, Ruimao Zhang, Pengfei Wan, Di Zhang, and Kun Gai. Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning. arXiv preprint arXiv:2501.04698, 2025.

- [17] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024.

- [18] Hyeonho Jeong and Jong Chul Ye. Ground-a-video: Zero-shot grounded video editing using text-to-image diffusion models. arXiv preprint arXiv:2310.01107, 2023.

- [19] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025.

- [20] Johanna Karras, Aleksander Holynski, Ting-Chun Wang, and Ira Kemelmacher-Shlizerman. Dreampose: Fashion video synthesis with stable diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22680–22690, 2023.

- [21] Keling. Image to video elements feature. https://app.klingai.com/cn/multimodal-to-video/add-object/new, 2025.
- [22] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [23] Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. Anyv2v: A tuning-free framework for any video-to-video editing tasks. arXiv preprint arXiv:2403.14468, 2024.

- [24] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [25] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. URL https: //arxiv.org/abs/2506.15742.
- [26] Feng Liang, Akio Kodaira, Chenfeng Xu, Masayoshi Tomizuka, Kurt Keutzer, and Diana Marculescu. Looking backward: Streaming video-to-video translation with feature banks. arXiv preprint arXiv:2405.15757, 2024.

- [27] Feng Liang, Bichen Wu, Jialiang Wang, Licheng Yu, Kunpeng Li, Yinan Zhao, Ishan Misra, Jia-Bin Huang, Peizhao Zhang, Peter Vajda, et al. Flowvid: Taming imperfect optical flows for consistent video-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8207–8216, 2024.

- [28] Sen Liang, Zhentao Yu, Zhengguang Zhou, Teng Hu, Hongmei Wang, Yi Chen, Qin Lin, Yuan Zhou, Xin Li, Qinglin Lu, et al. Omniv2v: Versatile video generation and editing via dynamic content manipulation. arXiv preprint arXiv:2506.01801, 2025.

- [29] Miao Liao, Feixiang Lu, Dingfu Zhou, Sibo Zhang, Wei Li, and Ruigang Yang. Dvi: Depth guided video inpainting for autonomous driving. In European conference on computer vision, pages 1–17. Springer, 2020.

- [30] Jun Hao Liew, Hanshu Yan, Jianfeng Zhang, Zhongcong Xu, and Jiashi Feng. Magicedit: High-fidelity and temporally coherent video editing. arXiv preprint arXiv:2308.14749, 2023.

- [31] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

- [32] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

- [33] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024.

- [34] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.

- [35] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Pose-guided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 4117–4125, 2024.

- [36] Yongsen Mao, Junhao Zhong, Chuan Fang, Jia Zheng, Rui Tang, Hao Zhu, Ping Tan, and Zihan Zhou. Spatiallm: Training large language models for structured indoor modeling. arXiv preprint arXiv:2506.07491, 2025.

- [37] OpenAI. Chatgpt (gpt-4 version). https://chat.openai.com/, 2024.
- [38] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

- [39] Pika. Pikaadd. https://pika.art/pikadditions, 2025.
- [40] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

- [41] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. URL https://arxiv.org/abs/2408.00714.

- [42] Shihao Shao and Qinghua Cui. 1st place solution in google universal images embedding. arXiv preprint arXiv:2210.08473, 2022.

- [43] Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, and Houfeng Wang. Preference ranking optimization for human alignment. Proceedings of the AAAI Conference on Artificial Intelligence, 38

(17):18990–18998, Mar. 2024. doi: 10.1609/aaai.v38i17.29865. URL https://ojs.aaai.org/index.php/AAAI/ article/view/29865.

- [44] Yao Teng, Enze Xie, Yue Wu, Haoyu Han, Zhenguo Li, and Xihui Liu. Drag-a-video: Non-rigid video editing with point-based interaction. arXiv preprint arXiv:2312.02936, 2023.

- [45] Yuanpeng Tu, Hao Luo, Xi Chen, Sihui Ji, Xiang Bai, and Hengshuang Zhao. Videoanydoor: High-fidelity video object insertion with precise motion control. arXiv preprint arXiv:2501.01427, 2025.

- [46] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [47] Fu-Yun Wang, Zhaoyang Huang, Weikang Bian, Xiaoyu Shi, Keqiang Sun, Guanglu Song, Yu Liu, and Hongsheng Li. Animatelcm: Computation-efficient personalized style video generation without personalized video data. In SIGGRAPH Asia 2024 Technical Communications, pages 1–5. 2024.

- [48] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8428–8437, 2025.

- [49] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, and Yu Qiao. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022.

- [50] Teng Xiao, Yige Yuan, Huaisheng Zhu, Mingxiao Li, and Vasant G Honavar. Cal-dpo: Calibrated direct preference optimization for language model alignment. Advances in Neural Information Processing Systems, 37: 114289–114320, 2024.

- [51] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1481–1490, 2024.

- [52] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

- [53] Zixuan Ye, Xuanhua He, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Qifeng Chen, and Wenhan Luo. Unic: Unified in-context video editing. arXiv preprint arXiv:2506.04216, 2025.

- [54] Yuyang Zhao, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Make-a-protagonist: Generic video editing with an ensemble of experts. arXiv preprint arXiv:2305.08850, 2023.

- [55] Shaobin Zhuang, Zhipeng Huang, Binxin Yang, Ying Zhang, Fangyikang Wang, Canmiao Fu, Chong Sun, Zheng-Jun Zha, Chen Li, and Yali Wang. Get in video: Add anything you want to the video. arXiv preprint arXiv:2503.06268, 2025.

###### [56] Bojia Zi, Weixuan Peng, Xianbiao Qi, Jianan Wang, Shihao Zhao, Rong Xiao, and Kam-Fai Wong. Minimaxremover: Taming bad noise helps video object removal. arXiv preprint arXiv:2505.24873, 2025.

##### A Implementation Details

- A.1 Detailed Parameters The hyper-parameters used in our experiments are set as follows:

- • For the IPO loss, we set the hyper-parameters γ, λ and β to 10, 1, 1, respectively, optimizing the model in phase 4.
- • For the Subject-Focused Loss (SL), we set λ1 and λ2 as 1, 1, respectively.

- A.2 Training and Inference Details

Training time. The phase-1 subject-to-video training takes about 70k iterations (nearly 2700 A100 GPU hours). The phase-2 pretraining for the MVI task costs about 30k iterations (nearly 1500 A100 GPU hours), and the phase-3 refinement costs about 10k iterations (nearly 500 A100 GPU hours). The last phase-4 preference optimization takes about 8k iterations (nearly 2300 A100 GPU hours).

Training Parameters. To avoid the computational expense of full fine-tuning, we integrate the LoRA mechanism into the DiT blocks, configuring it with a rank of 256. This results in a total of 600M trainable parameters.

Training Data Details. For the training of the phase-1 subject-to-video model and the phase-2 MVI task pretraining, we use a large-scale training dataset containing about 1M samples. Mark data generated by the introduced RealCapture Pipe, SynthGen Pipe (T2I + I2V + subject remove), SynthGen Pipe (video editing), and SimInteract Pipe as (a), (b), (c), (d), respectively. The actual ratio of different types of data (type (a) : type (b) : type (c) : type (d)) in phase 1 and phase 2 is set to 5:2:2:1. For the training of phase-3 refinement, the dataset contains about 50k samples and the ratio of different data is 3:3:3:1. For the training of phase-4 preference optimization, the training dataset has about 0.5k good-bad paired data, and the ratio is set to 3:3:3:1.

Inference time. Generating a single 5-second 480P video (121 frames) with our proposed model takes approximately 90 seconds using 8 NVIDIA A100 GPUs. Further discussion on potential improvements to inference speed can be found in Sec. D.

- A.3 User Study

To compare with the baseline methods, we conduct a user study as part of the evaluation. The survey randomly presented 40 sets of generated results to each participant. Fig. 8 displays a screenshot from our user study, showcasing a set of generated results. From left to right, it shows the reference subject, the source video, and the results from three competing methods in random order, paired with the corresponding text prompt. We ask each participant four questions:

- 1. Which result appears to have the highest consistency with reference subjects?
- 2. Which result best matches the prompt ‘[prompt]’?
- 3. Which result appears to have the highest insertion rationality?
- 4. Which result matches your best choice based on comprehensive considerations?

For each set of results displayed in the survey, we ensured that their order was randomly shuffled to prevent bias. Responses where all answers had the same selection and responses with completely identical answers were considered invalid. Finally, we obtained a total of 30 valid surveys to evaluate the model.

##### B InsertBench Details

To address the lack of a benchmark for Mask-free Video Insertion (MVI), we introduce a comprehensive benchmark, InsertBench, which consists of 120 videos paired with meticulously selected subjects (suitable for insertion in each video) and the corresponding prompts. Our collected videos span diverse categories, including natural landscapes, indoor environments, transportation scenes, dynamic interactions, animated

[Figure 318]

Figure 8 Screenshot of user study.

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

Add a boy playing with Doraemon

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

Add a girl next to the man, recording with a mobile phone

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

Add a monkey to the wooden box

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

Add a necklace to the man

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

Add a wild boar in the restaurant

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

Figure 9 Examples of InsertBench.

19

Add a dog in the car

contexts and wearable scenarios, each comprising 121 frames at 24 fps. As shown in Fig. 9, we carefully select suitable subjects for each video to insert, while considering harmonization and creativity.

##### C More Visual Results

In this section, we provide more visual results of OmniInsert. Fig. 10 and Fig. 11 present our inference results, Fig. 12 and Fig. 13 show comparisons with baselines. Corresponding videos for all results (main paper and supplement) are available in supplementary attachments. We strongly recommend viewing the mentioned dynamic results for a more intuitive understanding of the practical effectiveness of our proposed method.

All results are generated at 480P resolution based on our proposed benchmark InsertBench. Supplemental results demonstrate enhanced capabilities in diverse aspect ratios (e.g., vertical video) and multi-subject scenarios, further improving our method can achieve stable, semantically coherent insertions with strong prompt fidelity. Thanks to our designed data curation pipeline InsertPipe, which covers a wide range of scenes and subject categories, the model exhibits excellent robustness across various resolutions and aspect ratios. Our designed Condition-Specific Feature Injection (CFI) mechanism injects both video and subject features in a unified yet efficient manner, ensuring differentiated condition injection. And we integrate the LoRA mechanism into the DiT blocks, preserving the model’s original prompt following capabilities. Furthermore, the Progressive Training (PT) strategy enables the model to effectively achieve subject-scene equilibrium through multi-stage optimization, further enhancing insertion stability. As shown in Fig. 12 and Fig. 13, both Pika-Pro [39] and Kling [21] exhibit issues with insertion failures and unnatural insertion artifacts, whereas our method demonstrates superior robustness. Additionally, our model is capable of dealing with multi-subject scenarios, a capability that other baseline models either lack or perform poorly. This is achieved through our CFI design, where the IP information of multiple subjects is concatenated along the temporal dimension, enabling a unified training framework with the common single-subject insertion.

##### D Limitation and Future Work

In this section, we discuss the limitations of our method and potential directions for future work.

Color Fidelity and Physical Plausibility. We observe that our generated results may occasionally exhibit slight color discrepancies compared to the reference video, as well as minor physically implausible phenomena, as illustrated in Fig. 14. Notably, such issues are not unique to our method and are commonly observed across competitive baselines. Although our designed Insertive Preference Optimization (IPO) in Phase-4 has significantly reduced the challenges, complete resolution is still difficult. Due to the dependence of the IPO mechanism on the model’s intrinsic high-quality sample generation capability, the pool of selectable high-quality examples may become constrained when such artifacts are prevalent, especially in diversity. Therefore, incorporating more advanced preference optimization techniques [1, 43] is a crucial direction for future improvement.

Inference Speed. Our method adheres to a standard video diffusion model baseline, with an inference time of approximately 90 seconds for a 480P video with 121 frames. In contrast, the inference speeds of the compared bashlines are all above 180s. (Notably, as our compared baselines are all closed-source, we are unable to conduct a fair comparison by running their models offline.) Although our method demonstrates superior inference speed, its performance could be further enhanced by applying general acceleration techniques [26, 47] for video diffusion models.

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

###### Figure 10 More qualitative results I.

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

###### Figure 11 More qualitative results II.

Add a necklace to the man

|[Figure 413]<br><br>[Figure 414]|[Figure 415]| |[Figure 416]|
|---|---|---|---|

|[Figure 417]|
|---|

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

𝑶𝒖𝒓𝒔

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

𝑲𝒍𝒊𝒏𝒈

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

𝑷𝒊𝒌𝒂-𝑷𝒓𝒐

Add a woman next to the man, recording with a mobile phone

|[Figure 430]<br><br>[Figure 431]|[Figure 432]|[Figure 433]| |
|---|---|---|---|

|[Figure 434]|
|---|

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

𝑶𝒖𝒓𝒔

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

𝑲𝒍𝒊𝒏𝒈

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

𝑷𝒊𝒌𝒂-𝑷𝒓𝒐

Add a groundhog wearing a pair of glasses next to the tree

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

𝑶𝒖𝒓𝒔

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

𝑲𝒍𝒊𝒏𝒈

###### Figure 12 More qualitative comparisons I.

###### Add a wild boar in the restaurant

|[Figure 461]<br><br>[Figure 462]| | |[Figure 463]<br><br>[Figure 464]|
|---|---|---|---|

|[Figure 465]|
|---|

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

𝑶𝒖𝒓𝒔

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

𝑲𝒍𝒊𝒏𝒈

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

𝑷𝒊𝒌𝒂-𝑷𝒓𝒐

Add a bird flying upon the tree

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

|[Figure 482]|
|---|

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

𝑶𝒖𝒓𝒔

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

𝑲𝒍𝒊𝒏𝒈

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

𝑷𝒊𝒌𝒂-𝑷𝒓𝒐

Figure 13 More qualitative comparisons II.

[Figure 495]

[Figure 496]

[Figure 497]

### 𝒂 𝒃

Figure 14 The failure cases. (a) shows the physically implausible bad cases, such as model interpenetration, and (b) shows the slight color discrepancy between the source video and the inference result.

