### Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset

# arXiv:2510.15742v2[cs.CV]17Dec2025

Qingyan Bai1,2, Qiuyu Wang2, Hao Ouyang2, Yue Yu1,2, Hanlin Wang1,2, Wen Wang2,3, Ka Leong Cheng2, Shuailei Ma2,4, Yanhong Zeng2, Zichen Liu1,2, Yinghao Xu2, Yujun Shen2, Qifeng Chen1 1HKUST 2Ant Group 3Zhejiang University 4Northeastern University

[Figure 1]

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

[Figure 18]

Reimagine the industrial setting in a cybernetic brain lab where the tablet is a neural interface.

Replace the black dog with a white fox sitting calmly beside them.

Make it in the style of Japanese anime.

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

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Add a glowing vintage streetlamp casting a warm yellow hue on the pavement near the couple.

Transform the scene into a whimsical pastry dream world where the cake is a floating island.

Imitate the look of the 3D Chibi style.

Figure 1. Our proposed synthetic data generation pipeline can automatically produce high-quality and highly diverse video editing data, encompassing both global and local editing tasks. We highly recommend the readers see the supplementary video samples.

#### Abstract

Instruction-based video editing promises to democratize content creation, yet its progress is severely hampered by the scarcity of large-scale, high-quality training data. We introduce Ditto, a holistic framework designed to tackle this fundamental challenge. At its heart, Ditto features a novel data generation pipeline that fuses the creative diversity of a leading image editor with an in-context video generator, overcoming the limited scope of existing models. To make this process viable, our framework resolves the prohibitive cost-quality trade-off by employing an efficient, distilled model architecture augmented by a temporal en-

hancer, which simultaneously reduces computational overhead and improves temporal coherence. Finally, to achieve full scalability, this entire pipeline is driven by an intelligent agent that crafts diverse instructions and rigorously filters the output, ensuring quality control at scale. Using this framework, we invested over 12,000 GPU-days to build Ditto-1M, a new dataset of one million high-fidelity video editing examples. We trained our model, Editto, on Ditto1M with a curriculum learning strategy. The results demonstrate superior instruction-following ability and establish a new state-of-the-art in instruction-based video editing. The data, model, and code can be found at the project page.

#### 1. Introduction

Recently, the field of visual generative models has witnessed a remarkable divergence: while instruction-based image editing has achieved unprecedented levels of precision and user-friendliness with models like InstructPix2Pix [6], FLUX.1 Kontext [3], Qwen-Image [47], and Gemini 2.5 Flash Image (Nano-Banana) [16], its video counterpart has lagged significantly behind. This growing capabilities gap stems from the inherent complexities of the temporal dimension. Editing a video requires not only modifying content but also ensuring these changes are propagated coherently across frames—a challenge that has proven formidable. The primary obstacle impeding progress is a well-understood but unsolved problem: the profound scarcity of large-scale, high-quality, and diverse paired data for training end-to-end video editing models [53, 54].

Existing works have attempted to address this data scarcity challenge through various synthetic data generation strategies. Earlier approaches either relied on computationally prohibitive per-video optimization methods [36] or adopted training-free image-to-video propagation techniques [49, 53]. However, these pipelines suffer from a persistent trade-off: they sacrifice editing diversity, temporal consistency, and visual quality for scalability, or vice-versa. A scalable, cost-efficient data pipeline that generate highfidelity results remains an open challenge.

To address these shortcomings, we introduce Ditto1, a scalable and cost-efficient data synthesis pipeline architected to systematically dismantle these trade-offs. Our approach first tackles the challenge of editing fidelity and diversity. Capitalizing on the advanced maturity of instruction-based image editors, the pipeline generates a high-quality edited reference frame that acts as a strong visual prior. This anchor frame then guides an in-context video generator [20] to synthesize a temporally coherent video that faithfully matches the edit, overcoming the quality limitations of previous methods. Second, to resolve the critical efficiency-coherence trade-off where high-fidelity generation is prohibitively expensive, our pipeline integrates a distilled video model with a temporal enhancer. This innovative combination reduces computational costs to just 20% of the original while preserving temporal stability and avoiding visual artifacts. Finally, to achieve true scalability and eliminate the bottleneck of manual curation, we deploy an autonomous Vision-Language Model (VLM) agent. This agent carries dual responsibilities: programmatically generating diverse instructions for both local and global edits, and serving as a flaw-detection mechanism to automatically filter out low-quality or failed video pairs, en-

1The name “Ditto” is chosen to reflect the model’s core function: making the output video a faithful reflection, or “ditto,” of the user’s textual instruction.

suring the final dataset’s integrity.

We invested over 12,000 GPU-days using this pipeline to construct Ditto-1M, a new large-scale dataset comprising over one million source-instruction-edited video triplets, as demonstrated in Fig. 1. The dataset is meticulously structured to cover a wide spectrum of editing tasks and is curated via our VLM agent to ensure instruction consistency and high aesthetic quality.

With the proposed dataset, we train our final editing model, Editto. To bridge the gap between our visuallyguided data synthesis and the goal of purely instructiondriven inference, we propose a modality curriculum learning strategy [4]. Our curriculum begins by providing the model with both the text instruction and the edited reference image as a “scaffold.” As training progresses, we gradually anneal the visual guidance, compelling the model to learn the more difficult, abstract mapping from text instruction alone.

Our contributions are as follows:

- • A novel, scalable synthesis pipeline, Ditto, that efficiently generates high-fidelity and temporally coherent video editing data.
- • The Ditto-1M Dataset, a million-scale, open-source collection of instruction-video pairs to facilitate community research.
- • A state-of-the-art editing model, trained on Ditto-1M, that demonstrates superior performance on established benchmarks.
- • A modality curriculum learning strategy that effectively enables a visually-conditioned model to perform language-driven editing.

#### 2. Related Work

###### 2.1. Instruction-based Image Editing

Visual generative models have advanced rapidly [1, 5, 7, 12, 13, 15, 17–19, 23–25, 27, 30, 37, 39–45, 50, 56]. Instruction-based image editing has also rapidly evolved, moving beyond simple text-to-image generation to enable nuanced, user-guided modifications. Early and influential methods like InstructPix2Pix [6] demonstrated the feasibility of fine-tuning diffusion models on generated datasets of image triplets (source image, instruction, edited image) to perform edits. This was achieved by ingeniously combining a large language model (GPT-3) to generate textual edit instructions and a text-to-image model (Stable Diffusion) to synthesize the corresponding image pairs, creating a largescale training corpus without manual annotation. More recent advancements, particularly with the advent of powerful models like FLUX.1 Kontext [3], Qwen-Image [47], and Gemini 2.5 Flash Image (Nano-Banana) [16], have unlocked even more sophisticated capabilities. These models can process both text and reference images as inputs,

enabling targeted local edits, robust character consistency across multiple turns, and complex scene transformations within a unified architecture, often without requiring finetuning. Our work builds upon this progress by integrating a state-of-the-art instruction-based image editor as a critical component in our data synthesis pipeline, using it to manipulate keyframes that guide the subsequent video-level edit.

###### 2.2. Instruction-based Video Editing

Video editing has gained remarkable progress [8, 9, 28, 46, 51] with the development of the base generative models. Extending instruction-based editing to video requires maintaining temporal consistency and preserving background content. Current approaches fall into two main categories:

Inversion-based Methods. These methods avoid paired video-text-edit data but are computationally intensive. Tune-A-Video [48] fine-tunes a text-to-image model on a single video, enabling personalized edits but lacking scalability. Zero-shot techniques like TokenFlow [14] and FateZero [35] use DDIM inversion and feature propagation to enforce the consistency of the edited video. However, their quality relies on inversion fidelity and often struggles with complex motion or occlusions.

Feed-forward Methods. These end-to-end models aim to overcome inversion-based limitations but face the fundamental challenge of data scarcity. The development of feed-forward approaches is tightly coupled with the creation of synthetic datasets, as large-scale human-annotated video edit data is notoriously scarce. Early works [36, 55] attempted to synthesize data using computationally expensive one-shot tuning methods [32, 48], which limited the scale and quality of the resulting datasets. More recent paradigms have sought greater scalability with the approach of “lift and propagate”, employed by methods like VEGGIE [53] and InsViE [49]. These methods edit a single keyframe and then use an image-to-video model to propagate the change, but often suffer from temporal inconsistencies as the quality is capped by the propagation model. Se˜norita [57] achieves notable progress with a sophisticated “expert system” paradigm. It systematically categorizes editing tasks into 18 sub-classes and employs a large suite of specialized expert models - some newly trained - to generate highquality data for each specific task. While ensuring quality for predefined tasks, this approach is less scalable and requires significant effort to develop and maintain numerous task-specific models. In stark contrast to these specialized or propagation-based pipelines, our work introduces a unified and scalable ”All-in-One” data synthesis framework. Our pipeline is centered around a single in-context video generator that conditions on a reference edited frame from a single image editor and a depth-derived motion representation, enabling more direct and high-quality video synthesis without relying on a multitude of disparate expert models.

Crucially, our contribution extends beyond the dataset itself. We propose a novel Modality Curriculum Learning (MCL) strategy during training. This allows our final model, Editto, to perform edits based purely on text instructions at inference time, bridging the gap between multi-modal data synthesis and single-modality deployment. Finally, the concurrent work EditVerse [21] also explores in-context learning to unify editing tasks. Instead, we leverage in-context generation primarily for high-quality data synthesis.

#### 3. Ditto-1M

Our methodology begins with the construction of a largescale, high-quality dataset. We designed a novel, scalable data generation pipeline to synthesize over one million instruction-video triplets, as in Fig. 2. The architecture of this pipeline was specifically engineered to address four critical challenges inherent to existing data synthesis approaches:

- 1. Overcoming Limited Editing Diversity and Fidelity. Current data pipelines of instruction-based video editing often rely on training-free inversion techniques [36, 54], which tend to yield synthetic data of limited quality. To address this, we propose to leverage an in-context video generator to produce high-quality editing samples with visual contexts. Capitalizing on the more advanced development of image-based editing models, we incorporate strong priors from these image editors to serve context and guide the video generation for better editing quality. This is combined with depth-guided video context to ensure spatiotemporal coherence, significantly improving the diversity and fidelity of generated edits.
- 2. Resolving the Efficiency-Quality Trade-off. A major technical hurdle is the trade-off between generation cost and data quality. Current high-fidelity methods are prohibitively expensive (e.g., 50 GPU-minutes per sample on a single GPU), while faster, distilled models often introduce artifacts like temporal flickering. Our pipeline is designed with a cost-aware workflow that significantly reduces computational overhead without compromising the temporal coherence of the videos.
- 3. Automating Instruction Generation and Quality Control. To achieve true scalability, manual creation of instructions and verification of outputs is infeasible. Our pipeline integrates an automated agent with two primary responsibilities: (a) programmatically generating diverse and meaningful instructions for both local and global edits, and (b) serving as a flaw-detection mechanism to automatically filter out generated pairs that are of low quality or fail to follow the instructions.
- 4. Ensuring High Aesthetic and Motion Quality. Unlike general-purpose video datasets (e.g., Panda-70M), which are not optimized for editing tasks, our pipeline prioritizes the generation of content with high aesthetic value

##### 1. Pre-process (~60GPU-Days) 2. Generation (~6000GPU-Days) 3. Post-process (~6000GPU-Days)

[Figure 37]

Source video

[Figure 38]

Source Videos

[Figure 39]

[Figure 40]

Depth Predictor

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

VLM Filtering

[Figure 50]

Tags

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Scene Subject

[Figure 57]

[Figure 58]

- 1. Video Caption
- 2. Edit Instructions

[Figure 59]

[Figure 60]

VLM

[Figure 61]

###### VLM

[Figure 62]

[Figure 63]

[Figure 64]

…

Rule-based Filtering of Edited Videos

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Representations

[Figure 72]

Low noise

[Figure 73]

[Figure 74]

[Figure 75]

Vision Encoder

|[Figure 76]|
|---|

[Figure 77]

[Figure 78]

…

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Image Editor

Extraction of Tags and Representations

Key-Frames

Edited Key-Frames

[Figure 93]

Delete >

[Figure 94]

[Figure 95]

Depth Videos

끫룊7

끫룊1

Denoising Enhancer

[Figure 96]

끫룊4

[Figure 97]

[Figure 98]

[Figure 99]

끫룊6

Edited Key-Frames

끫룊0

Edit Instructions

끫룊3 끫룊2

[Figure 100]

끫룊5

끫룊

[Figure 101]

In-Context Video Generator

[Figure 102]

[Figure 103]

[Figure 104]

Deduplication with Representations

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

|[Figure 113]|
|---|

|[Figure 114]|
|---|

[Figure 115]

[Figure 116]

Point Tracking

[Figure 117]

[Figure 118]

Motion Filtering via Tracking Quality Enhancing of Edited Videos

Edited Videos

- Figure 2. Our scalable data synthesis pipeline. (1) Pre-processing: A diverse video pool is curated via automated deduplication and motion filtering. (2) The core engine synthesizes video triplets, conditioning an in-context generator on automated instructions, appearance context from edited key-frames, and structural context from depth maps. (3) Post-processing: Final visual quality is guaranteed by a VLM-based filter and a denoising enhancer. For the sake of reproducibility, every component in our pipeline employs an open-source model.

and natural motion dynamics. This focus ensures the resulting dataset, and the models trained upon it, are wellaligned with real-world usage scenarios where visual appeal is paramount.

The following sections will detail the architecture of our data generation pipeline, explaining how each component systematically addresses these challenges.

###### 3.1. Source Video Filtering

The Ditto-1M dataset is built exclusively from highresolution videos sourced from Pexels [34], a platform for professional-grade footage under the Pexels License. Unlike datasets derived from uncurated web scrapes, this strategy provides a foundation of superior aesthetic and technical quality, suitable for video editing tasks. We also first apply a rigorous filtering and pre-processing protocol. This protocol examines videos in the following aspects:

Near-Duplicate Removal: To prevent dataset redundancy and ensure broad content diversity, we implement a rigorous deduplication process. We employ a powerful visual encoder [31] to extract compact feature representations for each video. Pairwise similarity between these feature vectors is then computed. Videos exceeding a pre-defined similarity threshold are systematically filtered out, guaranteeing the uniqueness of each source video in our collection.

Motion Scale: Videos that contain little or no motion over time—such as fixed-camera surveillance footage, still nature scenes, or unmoving interior shots—are considered less valuable for video editing tasks because they lack dynamic visual changes. To automatically identify such lowdynamic content, we employ a tracking-based method that analyzes frame-to-frame motion across the video sequence. Specifically, for each video, we first sample points on a grid layout and then use CoTracker3 [22] to track these points, obtaining their trajectories. We then compute the average of the cumulative displacements of all tracked points over the entire video as the motion score of the video. By setting a threshold, we filter out videos with low motion scores, effectively removing those with negligible temporal variation. Videos that pass this filtering stage are then standardized. Each video is resized to a uniform resolution and its frame rate is converted to 20 FPS. This standardization simplifies the training process and ensures consistency across the entire dataset.

###### 3.2. Instruction Generation

For each filtered source video Vs, we generate a set of corresponding editing instructions p. We employ a powerful VLM Qwen2.5 VL [2] for this task with a two-step prompting strategy. First, we prompt the VLM to generate a dense

caption c that describes the video’s content, subjects, and scenery:

###### c = VLM(Vs,pcaption). (1)

This caption serves as a semantic anchor. Next, we feed both the video Vs and its caption c back into the VLM, prompting it to devise a creative and plausible editing instruction p:

###### p = VLM(Vs,c,pinstruct). (2)

This conditioned approach ensures that instructions are contextually grounded in the video’s content, yielding a diverse set of commands ranging from global style transformations to specific, localized object modifications.

###### 3.3. Visual Context Preparation

Our generation process is heavily guided by a rich visual context, which consists of two key components: an edited reference frame that specifies the target appearance, and a depth video that enforces spatiotemporal consistency.

Key-Frame Editing for Appearance Guidance. We first select a key-frame fk from the source video Vs as the anchor for the editing. This frame is then edited by the instructionguided image editor Qwen-Image [47] Eimg, using the instruction p generated in the previous step:

fk′ = Eimg(fk,p). (3)

This resulting frame fk′, serves as the visual prototype for the edit, defining the final appearance including style and

textures.

Depth Video Prediction for Spatiotemporal Structure. To preserve the geometric structure and motion dynamics of the original scene, we extract a dense depth video Vd from Vs with a video depth predictor D [10]. The predicted depth video acts as a dynamic structural scaffold, providing an explicit, frame-by-frame guide for the structure and geometry of the scene during the video generation.

###### 3.4. In-Context Video Generation With the editing instruction p and the multi-modal visual

context fk′ and Vd prepared, we synthesize the edited video Ve with the in-context video generator [20], which is denoted as G. VACE is a feed-forward video generative model designed to condition its generation on rich visual prompts such as images, masks, and videos by learning a context branch beyond the base generative model [43]. In our design, we adopt G to synthesize the edited video by taking the textual prompt p as a high-level semantic guide, the edited key-frame fk′ as the primary appearance condition, and the depth video Vd as a strict spatiotemporal constraint. This generation process is formulated as:

Ve = G(Vd,fk′,p). (4)

By integrating these three modalities with the attention mechanism, VACE can faithfully propagate the edit defined in fk′ across the entire sequence, adhering to the motion and structure laid out by Vd, while ensuring the result is semantically aligned with the instruction p. Our pipeline achieves high-quality and coherent video edits without costly pervideo optimization. Please refer to the supplementary materials for additional analysis of the data generator. To facilitate scalable synthetic data generation and further reduce the computational burden, we employ model quantization and knowledge distillation techniques [52]. We apply post-training quantization to reduce the model’s memory footprint and inference cost with minimal impact on output quality. Furthermore, we adopt the generative video model [52] distilled from the teacher model, preserving editing fidelity while significantly accelerating the generation process with few-step inference. This optimized pipeline is crucial for producing large-scale video editing data efficiently.

###### 3.5. Edited Video Curation and Enhancing

To guarantee the highest quality, the generated triplets (Vs, p, Ve) undergo a final two-stage curation and refinement including VLM filtering and denoiser enhancing.

VLM-Based Curation. We first use a VLM [2] as an automated judge to perform rejection sampling. Each triplet is evaluated against two criteria: (1) Instruction Fidelity: whether the edit in Ve accurately reflects the prompt p. (2) Fidelity: whether Ve preserves the semantic and motion from Vs. (3) Visual quality: whether the videos are visual appealing without significant distortion or artifacts. (4) Safety & Appropriateness: whether the content has unsafe or inappropriate material, such as pornography, violence, or horror, ensuring the dataset is ethically compliant and suitable. Triplets that fail to meet our quality thresholds on these criteria are discarded.

Quality Enhancement via Denoising. The curated edited videos are then enhanced using the state-of-the-art opensource Text-to-Video (T2V) model, Wan2.2 [43]. Unlike post-processing in prior work that performs simple upscaling, our objective is to achieve perceptual refinement without introducing semantic deviations from edited content of Ve. This requirement aligns perfectly with the specialized design of Wan2.2’s Mixture-of-Experts (MoE) architecture, which employs a coarse denoiser for structural and semantic formation under high noise, and a fine denoiser specialized in later-stage refinement under low noise. We specifically leverage the fine denoiser for a short, 4-step reverse process. For each video Ve, we first add a small amount of Gaussian noise. The fine denoiser then inverts this process utilizing its expert prior to remove subtle artifacts and enhance textural details precisely because it is optimized for making minimal, semantic-preserving adjustments to nearly-complete

Context Branch Edit Instruction

Input Video

|[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>|
|---|

“Make it an Ukiyo-e style video.”

Context

Block Context

## …

Block

Block

VAE Encoder

|[Figure 123]|
|---|

[Figure 124]

[Figure 125]

Reference Frame

Text Encoder

|[Figure 126]<br><br>|[Figure 127]|[Figure 128]| | | |[Figure 129]<br><br>|
|---|---|---|---|---|---|---|

…

Block

Block

| |[Figure 130]<br><br>[Figure 131]<br><br>| | |[Figure 132]<br><br>[Figure 133]| | |
|---|---|---|---|---|---|---|

VAE Encoder

DiT

DiT

DiT

+

VAE Decoder

Edited Video

[Figure 134]

[Figure 135]

[Figure 136]

|[Figure 137]<br><br>[Figure 138]| |[Figure 139]|
|---|---|---|

Edited Video

Noise

Main Branch

- Figure 3. Model training pipeline. We train the context blocks based on the in-context video generator with curriculum learning by gradually annealing and eventually dropping the reference frame.

Table 1. Quantitative comparisons with prior arts. The best results are bolded.

###### Automatic Metric Human Evaluation

Method CLIP-T ↑ CLIP-F ↑ VLM ↑ Edit-Acc ↑ Temp-Con ↑ Overall ↑

TokenFlow [14] 23.63 98.43 7.10 1.70 1.97 1.70 InsV2V [11] 22.49 97.99 6.55 2.17 1.96 2.07 InsViE [49] 23.56 98.78 7.35 2.28 2.30 2.36 Ours 25.54 99.03 8.10 3.85 3.76 3.86

videos. This yields a high-quality output with improved resolution and visual fidelity that remains strictly semantically consistent with our initial edit.

###### 3.6. Details of Ditto-1M

We collected a total of over 200k source videos, approximately half of which feature human activities. After undergoing a filtering process, these videos were edited using editing instructions generated by a VLM, followed by an additional round of filtering. This pipeline ultimately yielded approximately 1M edited videos. Among these, about 700k video triplets involve global editing (including changes to style, environment, etc.), while roughly 300k pertain to local editing (encompassing object replacing, adding, and removal). The final enhanced videos have a resolution of 1280x720, each comprising 101 frames at 20 FPS. The visual quality of the final samples significantly surpasses that of previous datasets - we strongly recommend reviewing the video samples provided on the project page.

#### 4. Modality Curriculum Model Learning

We select the in-context video generator VACE [20] as our backbone, inspired by its strong prior for generating videos that are spatially and structurally aligned with a source video. VACE’s original capability is to condition its generation on two visual contexts (and prompts): a source video and a reference image. Our goal is to repurpose this powerful visual generator into a proficient editor that operates on abstract textual instructions. However, directly fine-tuning

the model to bridge the vast semantic gap from visual to textual conditioning is prone to instability. We therefore adapt its architecture, as shown in Fig. 3. It consists of a Context Branch for extracting spatiotemporal features from the source video and reference frame, and a DiT-based [33] Main Branch that synthesizes the edited video under the joint guidance of the visual context and the new textual embeddings from the instruction.

To ease the training difficulty and stably bridge this modality gap, we introduce a modality curriculum learning (MCL) strategy. The core idea is to leverage the model’s inherent ability to process the reference image context as a temporary aid. In the initial training phase, we provide the edited reference frame as a strong visual ”scaffold” alongside the new text instruction. As training progresses, we gradually anneal the probability of providing this visual scaffold, eventually dropping it entirely. This process compels the model to shift its dependency from the concrete visual target it already understands to the more abstract textual instruction, transforming it into a purely instructionbased video editing model. We train the model using the flow matching [26] objective:

0,c∥vt(zt,t,c) − (z0 − zt)∥2, (5)

L = Et,z

where z0 is the clean latent encoded from the target edited video, zt is its noised version at timestep t, c represents the conditioning from text and visual contexts, and vt is the model’s predicted vector field pointing from zt to z0.

Source TokenFlow InsV2V InsViE Gen4-Aleph Ours

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Renderitinthestyleofpixelart.Replacetheman'sclothtoblacksuit.MakeittheLEGOtoystyle.

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

Figure 4. Qualitative comparisons with prior arts TokenFlow [14], InsV2V [11], InsViE [49] and Gen4-Aleph [38].

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Figure 5. Our data and learned model enable the translation from synthetic videos to the real domain.

#### 5. Experiments

ing rate of 1e-4 on a cluster of 64 GPUs. We employ our modality curriculum learning strategy, where the initial 5,000 steps serve as a curriculum warm-up phase.

###### 5.1. Experimental Settings

Our model is built upon the pre-trained in-context video generator [20, 43] and is fine-tuned on our newly proposed large-scale dataset, which comprises over one million highquality video triplets. To maintain the strong generative prior of the base model and ensure training efficiency, we freeze the majority of the pre-trained model’s parameters, and only fine-tune the linear projection layers of context blocks. The model is trained for approximately 16,000 steps using the AdamW optimizer [29] with a constant learn-

###### 5.2. Experimental Results

Quantitative Comparison. We perform quantitative comparisons using automatic metrics and a user study, summarized in Tab. 1. To ensure a fair comparison, our test set consists of 50 videos collected from various online sources, deliberately excluding Pexels videos to ensure the data is out-of-distribution relative to our training set. For each video, we provide 5 distinct editing instructions. For au-

Source Ours Data Generator

RobotarmsPencilsketch

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

Figure 6. Unlike the original data generator, which fails to handle newly emerging information beyond key frames, our model - trained with filtering and scaling techniques - outperforms it.

Source ~60K Samples ~120K Samples ~250K Samples ~500K Samples

Source w/o MCL w/ MCL

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

OrigamiPixel

BearChineseclothes

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Figure 7. Ablation studies on training data scale and modality curriculum learning (MCL).

ator from our pipeline, demonstrating superior handling of newly emerged content as in Fig. 6. This superiority stems from our scaled training regimen, including the curriculum learning and exposure to the filtered, high-quality data.

tomatic evaluation, we employ three metrics: CLIP-T measures the CLIP text-video similarity to assess how well the edit follows the instruction; CLIP-F calculates the average inter-frame CLIP similarity to gauge temporal consistency; and a VLM score provides a holistic assessment of edit effectiveness, semantic preservation, and overall aesthetic quality. A user study based on 1,000 votes from postgraduates and researchers also rates instruction-following (Edit-Acc), temporal consistency (Temp-Con), and overall quality (Overall). As shown, our method significantly outperforms all baselines across metrics, achieving the highest automatic scores and a strong preference in human evaluations, which confirms its superior instruction adherence, temporal smoothness, and visual quality. Please refer to the supplementary materials for details of the user study.

###### 5.3. Ablation Studies

We conduct ablation studies to validate the key components of our framework, with results presented in Fig. 7. We find that our model’s performance scales effectively with the training data - as the number of samples increases, both the quality of the stylistic edits and the fidelity to the original video’s content and motion improve significantly, confirming the value of our large-scale dataset. Furthermore, we ablate our modality curriculum learning (MCL) strategy and find that, without MCL, the model often struggles to interpret the instruction’s full semantic intent. Therefore it is crucial for bridging the modality gap and learning to follow instructions.

Qualitative Comparison. As shown in Fig. 4, our method consistently produces visually superior results that better adhere to edit instructions compared to prior arts. For complex stylizations, our model generates temporally coherent videos that accurately match the target style, while competitors often yield blurry or inconsistent results. For local attribute changes (e.g., “black suit”), our method precisely edits the target object while preserving identity and background details, a task where Gen4-Aleph slightly changes the man’s identity and other methods largely fail. We recommend reviewing the video samples provided on the project page for a better understanding.

#### 6. Conclusion

We have presented Ditto, a scalable framework that significantly advances instruction-based video editing by systematically addressing the core challenge of data scarcity through a new paradigm for large-scale data synthesis. Our synthetic data generation pipeline overcomes the fidelitydiversity and efficiency-coherence trade-offs plaguing prior methods by leveraging strong image-editing priors, a distilled in-context video generator with a temporal enhancer, and autonomous VLM-based quality control. This enables the creation of the large-scale, high-quality Ditto-1M dataset. The proposed modality curriculum learning strategy further ensures our model Editto achieves state-of-theart performance by effectively transitioning from visualtextual conditioning to purely instruction-driven inference.

Additional Results. We showcase the synthetic-to-real (syn2real) capability in Fig. 5 benefited from our data by training the model to map the stylized videos in our dataset back to their original, real-world source videos. This successful transfer highlights the rich and photorealistic information contained within our dataset, demonstrating its utility beyond standard editing tasks. Also, our final trained model substantially outperforms the raw data gener-

#### References

- [1] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647, 2025. 2
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025. 4, 5
- [3] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. FLUX. 1 Kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, 2025. 2
- [4] Yoshua Bengio, J´erˆome Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Int. Conf. Mach. Learn.,

2009. 2

- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. InstructPix2Pix: Learning to follow image editing instructions. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2
- [7] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Shijie Huang, Zhaohui Hou, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025. 2
- [8] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2Video: Video editing using image diffusion. In Int. Conf. Comput. Vis., 2023. 3
- [9] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. StableVideo: Text-driven consistency-aware diffusion video editing. In Int. Conf. Comput. Vis., 2023. 3
- [10] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. In IEEE Conf. Comput. Vis. Pattern Recog., 2025. 5
- [11] Jiaxin Cheng, Tianjun Xiao, and Tong He. Consistent videoto-video transfer using synthetic dataset. In Int. Conf. Learn. Represent., 2024. 6, 7
- [12] Ruihang Chu, Yefei He, Zhekai Chen, Shiwei Zhang, Xiaogang Xu, Bin Xia, Dingdong Wang, Hongwei Yi, Xihui Liu, Hengshuang Zhao, et al. Wan-move: Motioncontrollable video generation via latent trajectory guidance. arXiv preprint arXiv:2512.08765, 2025. 2
- [13] Junyu Gao, Kunlin Yang, Xuan Yao, and Yufan Hu. Unity in diversity: Video editing via gradient-latent purification. In IEEE Conf. Comput. Vis. Pattern Recog., 2025. 2
- [14] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. TokenFlow: Consistent diffusion features for consistent video editing. In Int. Conf. Learn. Represent., 2024. 3, 6, 7

- [15] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In IEEE Conf. Comput. Vis. Pattern Recog., 2023. 2
- [16] Google. Gemini 2.5 Flash Image. https : / / aistudio.google.com/models/gemini-2-5flash-image, 2025. 2
- [17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. AnimateDiff: Animate your personalized text-toimage diffusion models without specific tuning. In Int. Conf. Learn. Represent., 2024. 2
- [18] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.
- [19] Yi Huang, Wei Xiong, He Zhang, Chaoqi Chen, Jianzhuang Liu, Mingfu Yan, and Shifeng Chen. Dive: Taming dino for subject-driven video editing. In Int. Conf. Comput. Vis.,

2025. 2

- [20] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. VACE: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025. 2, 5, 6, 7
- [21] Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Nanxuan Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, et al. Editverse: Unifying image and video editing and generation with in-context learning. arXiv preprint arXiv:2509.20360, 2025. 3
- [22] Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker3: Simpler and better point tracking by pseudolabelling real videos. arXiv preprint arXiv:2410.11831,

2024. 4

- [23] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. HunyuanVideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603,

- 2024. 2

[24] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. arXiv preprint arXiv:2501.08316,

- 2025.

- [25] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350,

2025. 2

- [26] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 6
- [27] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025. 2
- [28] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-P2P: Video editing with cross-attention control. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 3

- [29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7
- [30] Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025. 2
- [31] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, ShangWen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision, 2023. 4
- [32] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. CoDeF: Content deformation fields for temporally consistent video processing. In IEEE Conf. Comput. Vis. Pattern Recog., 2024. 3
- [33] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Int. Conf. Comput. Vis., 2023. 6
- [34] Pexels. Pexels. https://www.pexels.com/, 2025. 4
- [35] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. FateZero: Fusing attentions for zero-shot text-based video editing. In Int. Conf. Comput. Vis., 2023. 3
- [36] Bosheng Qin, Juncheng Li, Siliang Tang, Tat-Seng Chua, and Yueting Zhuang. Instructvid2vid: Controllable video editing with natural language instructions. In Int. Conf. Multimedia and Expo, 2024. 2, 3
- [37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2022. 2
- [38] Runway. Introducing Runway Gen-4. https : / / runwayml . com / research / introducing runway-gen-4, 2025. 7
- [39] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In Adv. Neural Inform. Process. Syst., 2022. 2
- [40] Tiancheng Shen, Zilong Huang, Xiangtai Li, Zhijie Lin, Jiyang Liu, Yitong Wang, Jiashi Feng, Ming-Hsuan Yang, and Jun Hao Liew. Qk-edit: Revisiting attention-based injection in mm-dit for image and video editing. In Int. Conf. Comput. Vis., 2025.
- [41] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189, 2023.
- [42] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. In Int. Conf. Comput. Vis., 2025.

- [43] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 5, 7
- [44] Shengzhi Wang, Yingkang Zhong, Jiangchuan Mu, Kai Wu, Mingliang Xiong, Wen Fang, Mingqing Liu, Hao Deng, Bin He, Gang Li, et al. Align-a-video: Deterministic reward tuning of image diffusion models for consistent video editing. In IEEE Conf. Comput. Vis. Pattern Recog., 2025.
- [45] Yukun Wang, Longguang Wang, Zhiyuan Ma, Qibin Hu, Kai Xu, and Yulan Guo. Videodirector: Precise video editing via text-to-video models. In IEEE Conf. Comput. Vis. Pattern Recog., 2025. 2
- [46] Cong Wei, Quande Liu, Zixuan Ye, Qiulin Wang, Xintao Wang, Pengfei Wan, Kun Gai, and Wenhu Chen. Univideo: Unified understanding, generation, and editing for videos. arXiv preprint arXiv:2510.08377, 2025. 3
- [47] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 2, 5
- [48] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-A-Video: One-shot tuning of image diffusion models for text-to-video generation. In Int. Conf. Comput. Vis., 2023. 3
- [49] Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi Xie, and Lei Zhang. InsViE-1M: Effective instruction-based video editing with elaborate dataset construction. In Int. Conf. Comput. Vis., 2025. 2, 3, 6, 7
- [50] Weihan Xu, Yimeng Ma, Jingyue Huang, Yang Li, Wenye Ma, Taylor Berg-Kirkpatrick, Julian McAuley, Paul Pu Liang, and Hao-Wen Dong. Regen: Multimodal retrievalembedded generation for long-to-short video editing. arXiv preprint arXiv:2505.18880, 2025. 2
- [51] Xiangpeng Yang, Linchao Zhu, Hehe Fan, and Yi Yang. VideoGrain: Modulating space-time attention for multigrained video editing. In Int. Conf. Learn. Represent., 2025. 3
- [52] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In IEEE Conf. Comput. Vis. Pattern Recog., 2025. 5
- [53] Shoubin Yu, Difan Liu, Ziqiao Ma, Yicong Hong, Yang Zhou, Hao Tan, Joyce Chai, and Mohit Bansal. VEGGIE: Instructional editing and reasoning video concepts with

grounded generation. arXiv preprint arXiv:2503.14350,

2025. 2, 3

- [54] Chi Zhang, Chengjian Feng, Feng Yan, Qiming Zhang, Mingjin Zhang, Yujie Zhong, Jing Zhang, and Lin Ma. InstructVEdit: A holistic approach for instructional video editing. arXiv preprint arXiv:2503.17641, 2025. 2, 3
- [55] Zhenghao Zhang, Zuozhuo Dai, Long Qin, and Weizhi Wang. EffiVED: Efficient video editing via text-instruction diffusion models. arXiv preprint arXiv:2403.11568, 2024. 3
- [56] Yixuan Zhu, Haolin Wang, Shilin Ma, Wenliang Zhao, Yansong Tang, Lei Chen, and Jie Zhou. Fade: Frequency-aware diffusion model factorization for video editing. In IEEE Conf. Comput. Vis. Pattern Recog., 2025. 2
- [57] Bojia Zi, Penghui Ruan, Marco Chen, Xianbiao Qi, Shaozhe Hao, Shihao Zhao, Youze Huang, Bin Liang, Rong Xiao, and Kam-Fai Wong. Se˜norita-2M: A high-quality instructionbased dataset for general video editing by video specialists. arXiv preprint arXiv:2502.06734, 2025. 3

