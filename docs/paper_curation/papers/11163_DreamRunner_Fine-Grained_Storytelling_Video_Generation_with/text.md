## DREAMRUNNER: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation

[Figure 1]

[Figure 2]

### Zun Wang1 Jialu Li1 Han Lin1 Jaehong Yoon2 Mohit Bansal1

1UNC Chapel Hill 2Nanyang Technological University {zunwang,jialuli,hanlincs,mbansal}@cs.unc.edu jaehong.yoon@ntu.edu.sg https://zunwang1.github.io/DreamRunner

. . .

[Figure 3]

[Figure 4]

[Figure 5]

- (2.1) Motion Retrieval and Prior Learning

[Figure 6]

- (2.2) Subject Prior Learning

||[Figure 7]|
|---|
<br><br>|[Figure 8]|
|---|
<br><br>witch ..|
|---|

Character Priors

|..<br><br>|[Figure 9]|
|---|
<br><br>|[Figure 10]|
|---|
<br><br>cat|
|---|

Reference Images

Fine-Tuning

# arXiv:2411.16657v4[cs.CV]14Nov2025

…

[Figure 11]

(1) Plan Generation

###### Fine-Grained Plan

|High-Level Plan<br><br>scene1, motion1, narration1|LLM<br><br>|
|---|---|
|LLM Planning<br><br>scene2: the study room motion2: writing, sipping, sitting narration2:<br><br>.. the witch is writing..and sipping tea.. her cat sitting..| |
| |Planning|
|scene3, motion3, narration3 …| |

..the witch reads her spellbook, gestures with her hands to cast a charm with her cat lying behind..

Background: study room

[Figure 12]

(3) Video Generation with Region-Based Diffusion

[Figure 13]

[Figure 14]

Frm1: (cap, bbox) pairs

[Figure 15]

…

###### Prompt

|Frm6: (cap, bbox) pairs cat sits.. study room.. notes..<br><br>witch sips..<br><br>|
|---|

Diffusion Model with SR3AI

LLM Planning

Write a story of the witch and her cat

..the witch writes about her adventures in her notes, sipping a calming tea, her cat is sitting beside..

[Figure 16]

Query

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

###### Motion Query

|[Figure 22]|Video|
|---|---|
|[Figure 23]| |
|[Figure 24]| |

Database

…

Sipping …

.

Fine-Tuning

…

.

Retrieval

a woman is sipping a cup of coffee

a man is sipping while wearing a sweater

.

Motion Priors

[Figure 25]

..the witch walks among the moonlit flowers in her garden with her cat, whispering a soft farewell..

Figure 1: Overall pipeline for DREAMRUNNER. (1) plan generation stage: we employ an LLM to craft a hierarchical video plan (i.elet@tokeneonedot, “High-Level Plan” and “Fine-Grained Plan”) from a user-provided generic story narration. (2.1) motion retrieval and prior learning stage: we retrieve videos relevant to the desired motions from a video database for learning the motion prior through test-time fine-tuning. (2.2) subject prior learning stage: we use reference images for learning the subject prior through test-time fine-tuning. (3) video generation with region-based diffusion stage: we equip diffusion

[Figure 26]

[Figure 27]

Motion Textual Query Sitting

|woman sipping a cup of coffee<br><br>model|
|---|

[Figure 28]

Query

with a novel spatial-temporal region-based 3D attention and prior injection module (i.elet@tokeneonedot, SR3AI) for video generation with fine-grained control.

[Figure 29]

[Figure 30]

…

Retrieval

##### Abstract

injection module SR3AI for fine-grained object-motion binding and frame-by-frame spatial-temporal semantic control. We compare DREAMRUNNER with various SVG baselines, demonstrating state-of-the-art performance in character consistency, text alignment, and smooth transitions. Additionally, DREAMRUNNER exhibits strong fine-grained conditionfollowing ability in compositional text-to-video generation, significantly outperforming baselines on T2V-ComBench. Finally, we domonstrate DREAMRUNNER’s ability to generate multi-character interactions with qualitative examples.

Storytelling video generation (SVG) aims to produce coherent and visually rich multi-scene videos that follow a structured narrative. Existing methods primarily employ LLM for high-level planning to decompose a story into scenelevel descriptions, which are then independently generated and stitched together. However, these approaches struggle with generating high-quality videos aligned with the complex single-scene description, as visualizing such complex description involves coherent composition of multiple objects/events, complex motion synthesis and character customization with sequential motions. To address these challenges, we propose DREAMRUNNER, a novel story-to-video generation method: First, we structure the input script using a large language model (LLM) to facilitate both coarse-grained scene planning as well as fine-grained object-level layout planning. Next, DREAMRUNNER presents retrieval-augmented test-time adaptation to capture target motion priors for objects in each scene, supporting diverse motion customization based on retrieved videos, thus facilitating the generation of new videos with complex, scripted motions. Lastly, we propose a novel spatial-temporal region-based 3D attention and prior

### 1 Introduction

Advancing storytelling video generation (SVG) is crucial for real-world video generation applications, enabling the creation of rich, immersive narratives with multiple realistic scenes, characters, and interactive events. Unlike existing short-form video generation approaches (Singer et al. 2022; Hong et al. 2022; Zhang et al. 2023b; Chen et al. 2023; Wang

- et al. 2023a; Girdhar et al. 2023; Khachatryan et al. 2023; Weng et al. 2024; Qing et al. 2024; Zhang et al. 2024; Fei
- et al. 2024; Bar-Tal et al. 2024), these models allow char-

acters and objects to evolve across scenes, enhancing the coherence of generated content to align more closely with human storytelling. Such capabilities hold vast potential in media, gaming, and interactive storytelling.

Existing SVG methods (He et al. 2023; Zhuang et al. 2024; Oh et al. 2025; Zheng and Fu 2024; He et al. 2024; Zhao et al. 2024) primarily employ high-level planning with a large language model (LLM), breaking down a story into multiple key scene descriptions. Each scene is then generated independently as a separate video and later stitched together to form a complete long-form storytelling video. Generating high-quality single-scene video exhibits three key challenges: 1) Coherent composition: As a highly complex textual form, a story, even at the single-scene level (e.g. “Lucy on the left and a man on the right is walking towards each other, they meet in the middle and start ballroom dancing”), typically involves multiple objects/characters with distinct motion trajectories, attributes, and sequentially occurring events, all of which must be coherently composed in the generated video. 2) Complex motion synthesis: The complex scene descriptions often feature intricate character motions (e.g. “ballroom dancing”) that are difficult to generate from the base text-to-video (T2V) models. 3) Character customization with sequential events: These descriptions usually involve characters with pre-defined reference images (e.g. Lucy), with sequential motions (e.g. walking to ballroom dancing), making it challenging to maintain both temporal coherence and visual consistency with the character. However, recent SVG methods often feed singlescene descriptions directly as textual conditions to the T2V model with limited constraints, resulting in suboptimal fidelity, missed events/objects, unclear motions, etc.

To address the above challenges, we propose DREAMRUNNER, a novel SVG framework that enhances finegrained alignment between scene descriptions and generated videos. Beyond high-level planning, DREAMRUNNER uses LLM-based compositional reasoning to decompose complex scenes into frame-by-frame layout plans of multiple entities with sequential motions/events, followed by regionbased attention for coherent composition. For complex motion synthesis, we adopt retrieval-augmented prior learning, injecting priors only into relevant regions to support character customization with sequential motions. Specifically, DREAMRUNNER presents three essential processes in the framework: (1) Dual-Level Video Plan Generation, (2) Motion Retrieval and Subject/Motion Prior Learning, and (3) Spatial-Temporal Region-Based 3D Attention and Prior Injection (SR3AI). In (1) plan generation stage, given a user-provided story narration (e.g. “write a story of the witch and her cat’s one day”), we employ an LLM for hierarchical planning: first generate a high-level plan with character-driven, motion-rich event descriptions across scenes, then decompose the scene descriptions into detailed, entity-specific frame-level layout plans within each scene. The generated frame-level plan serves as the fine-grained guidance for T2V. In (2) prior learning stage, we learn both subject and motion priors to enhance character consistency and motion fidelity. Subject priors are learned from character reference images using customization techniques (Ruiz et al.

- 2023) to adapt the model to specific appearances. Then we treat complex motion synthesis as a customization problem and learn motion priors to capture the visual patterns of target motions. To this end, we introduce an automatic retrieval pipeline that selects motion-relevant videos from a largescale dataset (Wang et al. 2023b) as references. We then apply test-time fine-tuning (Zhao et al. 2023) to learn customized motion priors. We use per-video prompts—rather than a shared one as in prior methods—to improve motion specificity, and learn both priors via LoRA-based tuning (Hu et al. 2021) on specific layers of DiT (Peebles and Xie 2023). In (3) video generation stage, we introduce SR3AI, a novel spatial-temporal region-based 3D attention and prior injection module that enables fine-grained control without additional training. Unlike prior methods that support only spatial (Lin et al. 2023; Lian et al. 2024; Yang et al. 2024a; Jain et al. 2024) or temporal (Bansal et al. 2024) control, SR3AI leverages frame-level layout plans to enable spatialtemporal control over sequential events, object attributes, trajectories, and spatial relationships. We first encode multiple conditions from the fine-grained plan. SR3AI then computes visual latents for each condition based on its spatialtemporal layout and enforces attention masking so that each condition attends only to its designated region. This ensures precise control and coherent composition of multiple objects and motions. Moreover, we extend this region-based design to inject learned character and motion priors into their corresponding regions in the diffusion model, enabling coherent character and motion customization.

We validate the effectiveness of DREAMRUNNER on two tasks: story-to-video generation and compositional text-tovideo generation. For SVG, we collect a story dataset, DreamStorySet, and compare DREAMRUNNER with SoTA methods (VideoDirectGPT (Lin et al. 2023) and VLogger (Zhuang et al. 2024)). DREAMRUNNER achieves a 13.1% relative improvement in character consistency score and an 8.56% gain in text alignment score. It also improves sequential event generation within a single scene, with a 27.2% boost for smoother multi-event transitions. Qualitative results further show strong generalization to multi-character settings. In compositional T2V generation, DREAMRUNNER outperforms baseline methods on T2VCompBench (Sun et al. 2024) across all dimensions, demonstrating its strength in compositional generation. Notably, despite being based on open-source models (Yang et al.

- 2024c), DREAMRUNNER achieves the highest scores in dynamic attribute binding and object interaction, along with comparable results in spatial relationships and motion binding to closed-source models, showing our method’s potential to bridge the performance gap between open- and closedsource models. In summary, our main contributions include:

- • A retrieval-augmented prior learning approach to enhance the synthesis of complex motions.
- • A spatiotemporal region-based attention module for coherent composition of multiple objects and sequential events, along with a region-based LoRA injection design for character and sequential motion customization.
- • SoTA performance in both compositional T2V and SVG.

### 2 Related Work

Storytelling Video Generation. Storytelling video generation (Oh et al. 2025; Zheng and Fu 2024; Long et al. 2024) aims to produce multi-scene videos from input scripts. Existing approaches use either high-level LLM planning for step-by-step decomposition and generation (Zhuang et al.

- 2024; He et al. 2023; Lin et al. 2023)or keyframe generation with text-to-image models followed by video animation (He et al. 2024; Zhao et al. 2024; Chai et al. 2023). Reference-based customization methods (Ruiz et al. 2023; Gal et al. 2023; Kumari et al. 2023a; Sohn et al. 2023; Park et al. 2024; Zheng et al. 2024; Ye et al. 2023; Wei et al. 2023b; Li, Li, and Hoi 2023; Chen et al. 2025; Huang et al.
- 2025; Liu et al. 2025) to preserve character identity across scenes are also adopted. Our work targets the video-centric challenge of generating multi-character, motion-rich videos with smooth, natural transitions. Compositional Generation. Recent advances in diffusion models have enhanced compositional T2V generation by improving coherence, semantic alignment, and user control. Several methods leverage LLMs for scene planning (Lian et al. 2024; Lin et al. 2023; Fei et al. 2024; Zheng et al. 2023; Qu et al. 2023), while others employ regional masks for multi-object control (Jain et al. 2024; Tian et al. 2024; Yang et al. 2024a; Yu et al. 2024; Wei et al. 2024) or frame-level semantic conditioning (Bansal et al. 2024; Xing et al. 2024). Additionally, LoRA-based compositional techniques integrate diverse concepts within the generation process (Yang et al. 2024b; Li et al. 2024; Gu et al. 2024; Zhong et al. 2024). However, these approaches do not explicitly bind objects to their corresponding actions/events spatialtemporally. Our method ensures fine-grained control over both objects and motions, maintaining a cohesive objectaction link throughout the video. Motion Customization. Motion customization remains a key challenge in video generation. One line of work focuses on pixel-level motion learning for video editing, aiming to replicate fine-grained motions from reference frames while preserving temporal consistency (Wu et al. 2023a; Zhang et al. 2023a; Ren et al. 2024; Jeong, Park, and Ye 2023; Lin et al. 2024). Another learns generic motion priors—e.g., human or camera movements—from curated datasets (Zhao et al. 2023; Wu et al. 2023b; Zhu et al. 2024; Wei et al.

- 2023a), capturing high-level semantics for realistic motion synthesis. Most approaches rely on test-time fine-tuning with motion LoRAs or adapters. Our method builds on this pipeline by retrieving motion-relevant videos from largescale databases for more diverse and context-aware motion priors. We also replace the standard single-prompt conditioning with per-video detailed prompts to improve motion specificity and generation quality.

### 3 Methodology

Task Setup. Storytelling Video Generation focuses on creating multi-scene, character-driven videos based on a given topic. The characters are defined by reference images (e.g. images of a witch), and the topic is presented as an instructional prompt (e.g. "witch’s one day"). The generated videos

should align with the given topic and accurately reflect the characteristics and behavior of the characters.

Method Overview. Our approach employs a hierarchical system where an LLM generates event-based scripts across multiple scenes, followed by detailed plans specifying the layout and motion transitions of key objects per scene (Section 3.1). A video diffusion model then synthesizes each scene step by step. We train motion priors from retrieval videos aligned with the LLM-generated plans, sourced from a large-scale video-language database, and character priors using the reference images (Section 3.2). Finally, we inject these priors and detailed plans into the video generation process in a zero-shot manner using our spatial-temporal regional diffusion module SR3AI (Section 3.3).

Base Generation Model. We leverages CogVideoX2B (Yang et al. 2024c) as the base text-to-video model. CogVideoX-2B employs a DiT-based architecture that integrates full 3D attention, and generates 6-second videos at 8 fps conditioned on input text. In our method, we extend CogVideoX-2B by training character and motion priors in distinct layers (see Sec. 3.2) and by modifying its 3D attention (see Sec. 3.3) for better motion and character binding.

#### 3.1 Generating Dual-Level Plans with LLMs

Story-Level Coarse-Grained Planning. We prompt an LLM (GPT-4o (OpenAI 2024)) to generate 6„8 characterdriven, motion-rich scene descriptions based on the story topic, task requirements, and a single in-context example. Each description follows a structured format: scene, motions, and narrations, where motions are defined first, followed by corresponding event narrations. This sequence forms a high-level plan that guides story progression across scenes, ensuring narrative coherence.

Scene-Level Fine-Grained Planning. After generating a list of single-scene descriptions with narrations, we create detailed, entity-level plans for each latent frame. Each plan consists of an overall background description followed by entity-specific details for each latent frame. As shown in the yellow Frame-Level Plan box at the top of Figure 2, the background provides a global scene description (e.g., “A large garden"), formatted as Background: background description. Entity-level details specify each entity’s description, motion (e.g., "A [v1] witch is walking among the moonlit flowers in her garden"), and bounding box layout, formatted as: Frame: [entity name, entity motion, entity description], [x0,y0,x1,y1]. Here, [x0,y0,x1,y1] denotes the top-left and bottom-right corners of the bounding box, with coordinates normalized to r0,1s. Entities without motion are labeled "none". When bounding boxes overlap, we prompt the LLM to generate a unified caption that integrates the descriptions of all entities within the overlapping region. Each scene includes plans for six key frames, with each frame guiding one second of video generation (we interpolate key frames to match the #frames of visual latents), resulting in a six-second output using CogVideoX. Detailed prompt templates for both levels’ planning are in Appendix I.

|The rock|
|---|

|The forest|
|---|

|The rock|
|---|

|The rock|
|---|

|The rock|
|---|

|The forest|
|---|

|The forest|
|---|

|The forest|
|---|

|A dirt path in the forest|
|---|

|A dirt path in the forest|
|---|

|A dirt path in the forest|
|---|

|A dirt path in the forest|
|---|

|[V*] teddy is sitting on the rock|
|---|

|[V*] teddy is strolling in the forest|
|---|

|[V*] teddy is strolling in the forest|
|---|

|[V*] teddy is strolling in the forest|
|---|

|Frame-Level Fine-Grained Plan + Conditions<br><br>|Frame 6: [witch, whispering, [v1] witch whispering to ..], [0.6, 0.0, 1.0, 0.8], [cat, walking, [v2] cat walking ..], [0.0, 0.2, 0.4, 0.8], ...<br><br>Background: A large garden with moonlit flowers<br><br>Frame-Level Plan<br><br>…<br><br>Frame 1: [witch, walking, [v1] witch walking in ..], [0.6, 0.0, 1.0, 0.8], [cat, walking, [v2] cat walking among..], [0.0, 0.2, 0.4, 0.8], ...|
|---|
<br><br>|C1: garden with ..<br>C2: moonlit flowers.. C3: [v2] cat walking..<br><br><br>C4: [v1] witch walking among the moonlit flower.. C5: [v1] witch whispering ..<br><br>Conditions|
|---|
|
|---|

|Single-Scene Narration<br><br>The witch walks among the moonlit flowers in her garden with her cat, whispering a farewell..|
|---|

Video Database

Motion Textual Query

.

Query

LLM

.

. Sitting

Retrieval

|visual latent attends its corresponding textual embeddings (in same color) and all visual latents (including itself) textual embeddings attends itself and its corresponding visual latents (in same color)<br><br>|Visual Latents<br><br>Latent Frame 1 Latent Frame 2-3 Latent Frame 4-6|
|---|
<br><br>Spatial-Temporal-Region-Based 3D Attention (with Frame-Level Plan)<br><br>||A large garden with moonlit flowers|
|---|
<br><br>|A [v1] witch is whispering farewell to the moonlit flowers|
|---|
<br><br>|Moonlit flowers in the garden|
|---|
<br><br>|A [v1] witch is walking among the moonlit flowers in her garden|
|---|
<br><br>|A [v2] cat near the witch among the moonlit flowers|
|---|
<br><br>Textual Condition embeddings|
|---|
|
|---|

|<br><br>[Figure 32]<br><br>…<br><br>|video 1: a person is sitting at an outdoor campfire<br>video 2: the person is sitting on a simulator for racing<br><br><br>Captions|
|---|
<br><br>video 1 video 2|
|---|

Diffusion Model

Conditions Text Encoder

Latents 𝑍 𝑇 … 𝑇

……

𝑍

[Figure 33]

Region-Based 3D-Attn

|||garden..|
|---|
<br><br>|flower..|
|---|
<br><br>|cat..|
|---|
<br><br>|witch whispering farwell..|
|---|
<br><br>|witch is walking among..|
|---|
<br><br>𝑀𝑎𝑠𝑘<br><br>|garden..|
|---|
<br><br>|flower..|
|---|
<br><br>|cat..|
|---|
<br><br>|witch whispering farwell..|
|---|
<br><br>|witch is walking among..|
|---|
<br><br>Character<br><br>LoRA Masks<br><br>𝑀𝑎𝑠𝑘|
|---|
<br><br>||garden..|
|---|
<br><br>|flower..|
|---|
<br><br>|cat..|
|---|
<br><br>|witch whispering farwell..|
|---|
<br><br>|witch is walking among..|
|---|
<br><br>|garden..|
|---|
<br><br>|flower..|
|---|
<br><br>|cat..|
|---|
<br><br>|witch whispering farwell..|
|---|
<br><br>|witch is walking among..|
|---|
<br><br>𝑀𝑎𝑠𝑘 𝑀𝑎𝑠𝑘<br><br>Motion<br><br>LoRA Masks|
|---|
<br><br>|𝑊x = 𝑊 𝑥 + 𝐴 𝐵 (𝑀𝑎𝑠𝑘 𝑥) +𝐴 𝐵 (𝑀𝑎𝑠𝑘 𝑥)<br><br>Character LoRA Injection|
|---|
<br><br>|𝑊𝑥 = 𝑊 𝑥 + 𝐴 𝐵 (𝑀𝑎𝑠𝑘 𝑥)<br><br>+ 𝐴 𝐵 (𝑀𝑎𝑠𝑘 𝑥)<br><br>Motion LoRA Injection|
|---|
<br><br>Spatial-Temporal-Region-Based LoRA Injection (with Frame-Level Plan)|
|---|

Feed Forword

𝑍

Region-Based 3D-Attn

Feed Forword

𝑍

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Denoising

Latents 𝑍

Figure 2: Implementation details for region-based diffusion. We extend the vanilla self-attention mechanism to spatialtemporal-region-based 3D attention (see upper orange part), which is capable of aligning different regions with their respective text descriptions via region-specific masks. The region-based character and motion LoRAs (see lower yellow and blue parts) are then injected interleavingly to the attention and FFN layers in each transformer block (see the right part). Note that though we resize the visual latents into sequential 2D latent frames for better visualization, they are flattened and concatenated with all conditions when performing region-based attention. Fig. 3 and Appendix A.3 provide example of the region-based attention mask and more details of region-based LoRA injection, respectively.

#### 3.2 Motion Retrieval and Prior Learning

video temporal LoRAs by injecting it into temporal attention layers with video-specific spatial LoRAs into spatial layers (Wang et al. 2023a; Zer 2023). Only temporal LoRAs are used during inference. Since our approach is based on CogVideoX (Yang et al. 2024c), which employs 3D full attention instead of separate spatial-temporal attention, we manually designate even layers as “spatial” and odd layers as “temporal” to separate the learning of spatial and temporal LoRAs. We train LoRAs on top-ranked retrieved videos with all backbone parameters frozen, optimizing both diffusion loss for frame reconstruction and an appearancedebiased temporal loss to focus on motion-specific learning. Note that unlike previous methods using a single-prompt condition for all the retrieved videos, we utilize video caption provided from the database as per-video prompt. This helps the model implicitly separate motion-unrelated backgrounds, appearances, etc, allowing it to focus on motionspecific patterns. More design details are in Appendix A.2.

Retrieving Motion-Related Videos from Database. We employ a retrieval-augmented approach to fine-tune motion priors at test time for complex motion synthesis. Based on motion descriptions generated from the LLM planning, we retrieve relevant videos from a large-scale video database (Wang et al. 2023b). Our retrieval pipeline first uses BM25 (Robertson, Zaragoza et al. 2009) for initial textbased retrieval, followed by attribute-based filtering and clip segmentation via object tracking (Jocher 2020). We then compute semantic similarity scores using CLIP (Radford et al. 2021) and ViCLIP (Wang et al. 2023b) to refine the selection, ensuring high-quality motion-aligned videos (see Appendix A.1 for details). By following this process, we retrieve 4 „ 20 video clips per motion, which are then used as reference videos for learning motion priors.

Motion Prior Training. We follow recent motion customization methods (Zhao et al. 2023) with test-time finetuning for learning motion priors. Typically, reference videos are used to learn an appearance-debiased cross-

Subject Prior Learning. We learn the subject’s appearance by injecting LoRA modules into the spatial transformer lay-

ers. To train these LoRAs, we create videos by repeating reference images multiple times (48 time, similar to the output frame number of CogVideoX) and focus on reconstructing the first frame of the video during training, preventing overfitting to the static, repeated video. Notably, the subject priors are learned within spatial LoRAs, while the motion priors are learned within temporal LoRAs. Since their injections target different layers, there is no overlap, effectively avoiding conflicts between multiple LoRAs.

#### 3.3 Spatial-Temporal-Region-Based Diffusion

Region-Based 3D Attention. We build our model on CogVideoX (Yang et al. 2024c), a text-to-video generation model designed on top of a Diffusion Transformer (DiT). Unlike methods that use separate spatial and temporal attention for efficient video modeling, CogVideoX employs a 3D full attention module, integrating self-attention across concatenated embeddings of all visual latents and the text condition embeddings. We extend this module to enable regionspecific conditioning via masking, aligning different regions with their respective text descriptions. Specifically, given a fine-grained plan with N region-specific text descriptions C1,C2,...,CN and corresponding layouts L1,L2,...,LN across frames, we encode each text condition Ci to produce embeddings T1,T2,...,TN (Figure 2 top right). At each attention layer, we identify the visual latents corresponding to each layout Li in the latent space. We then perform masked self-attention on the concatenation of T1,T2,...,TN and L1,L2,...,LN. The self-attention mask is defined as follows: for each region’s visual latents Li, attention is allowed to its corresponding text condition embeddings Ti and all visual latents L1,L2,...,LN. Conversely, for each condition embeddings Ti, attention is restricted to itself and its corresponding latents Li. This design ensures each region is conditioned on its specific textual description while maintaining interactions among visual latents through unmasked attention among L1,L2,...,LN. No modifications are made to other modules in the base model, preserving the integrity of its original architecture. A visualization example of such masking strategy is contained in Fig. 3.

Region-Based LoRA Injection. We adopt a similar regionbased strategy for injecting LoRA priors into diffusion models. For each LoRA, we first identify the corresponding regions of latent tokens based on the associated text description and layout information. LoRA injection is then applied exclusively to these regions, ensuring precise alignment between the priors and their designated areas. This approach enables handling multiple LoRAs simultaneously while avoiding conflicts between them, preserving the integrity of each injected prior. Appendix A.3 provides details of this strategy with equation derivations, explanations, etc.

### 4 Experiments

In this section, we first introduce the evaluation datasets and evaluation metrics details in Section 4.1, then compare our DREAMRUNNER with prior methods on story-to-video generation in Section 4.2. Next, we present detailed ablation studies on the necessity of RAG and effectiveness of

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

[Figure 54]

[Figure 55]

… …

Witch pours Witch stirs Cat walks portion room

[Figure 56]

[Figure 57]

[Figure 58]

| | | | | | | | |…| | | |…| | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | |…| | | |…| | | |
| | | | | | | | |…| | | |…| | | |
| | | | | | | | |…| | | |…| | | |
| | | | | | | | |…| | | |…| | | |
| | | | | | | | |…| | | |…| | | |
| | | | | | | | |…| | | |…| | | |
| | | | | | | | |…| | | |…| | | |
| | | | | | | | |…| | | |…| | |…|
| | | | | | | | |…| | | |…| | | |
| | | | | | | | |…| | | |…| | | |
| | | | | | | | |…| | | |…| | | |
| | | | | | | | |…| | | |…| |… …|… …|
| | | | | | | | |…| | | |…| | | |
| | | | | | | | |…| | | |…| | | |
| | | | | | | | |…| | | |…| | | |

Witch

pours

Witch

stirs

Cat

walks

portion

room

… …

… …

… …

… …

… …

… …

… …

… …

……

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

##### Figure 3: Visualization of spatial-temporal region-based

- 3D attention mask. Different text colors represent different conditions, while the white region indicates masked areas. For simplicity, we reduce each condition to two words, each frame to three segments, and display only three conditions and two frames in the figure. In practice, conditions can be longer and more numerous, frames can have more segments, and there are 12 latent frames in total.

SR3AI in Section 4.3, and demonstrate the generalizability of our DREAMRUNNER to improve compositional text-tovideo generation on T2V-CompBench (Sun et al. 2024) in Section 4.5. Moreover, we show the effectiveness of RAG for learning the motion prior on a more comprehensive motion dataset we collect in Section 4.6. Lastly, we present qualitative comparison between our DREAMRUNNER and previous approaches in Section 4.4.

4.1 Experimental Setups

Evaluation Datasets. We evaluate DREAMRUNNER on two tasks: (1) story-to-video generation, and (2) compositional text-to-video generation. The first task focuses on the model’s ability to follow the text closely while maintaining character and scene consistency throughout the story. The second task assesses various aspects of compositionality in video generation. For (1) story-to-video generation, we collect and introduce a new benchmark dataset, DreamStorySet. Specifically, we collect 10 characters, including 6 from existing customization datasets (CustomConcept101 (Kumari et al. 2023b) and Dreambooth (Ruiz et al. 2023)), and

- 4 with generation models (FLUX (flu 2024)). (featuring two motions per scene) and three multi-character stories (featuring two or three motions per scene). Each story comprises
- 5 to 8 scenes, incorporating a total of 64 diverse motions throughout. We focus on single-character stories for quantitative evaluation of SVG models and reserve multi-character

Character Fine-Grained Text Full Text Transition Visual Quality

Method

CLIP DINO CLIP ViCLIP CLIP ViCLIP DINO Aesthetics Imaging Smoothness

VideoDirectorGPT (Lin et al. 2023) 54.3 9.5 23.7 21.7 22.4 22.5 63.5 42.3 60.3 94.3 VLogger (Zhuang et al. 2024) 62.5 41.3 23.5 23.1 22.5 22.2 73.6 43.4 61.2 96.2

###### DREAMRUNNER (Ours) 70.7

###### 55.1

###### 24.7

###### 23.7

###### 24.2

###### 24.1

###### 93.6

###### 55.4

###### 62.1

###### 98.1

(+13.1%)

(+33.4%)

(+5.11%)

(+2.60%)

(+7.56%)

(+8.56%)

(+27.2%)

(+27.6%)

(+1.47%)

(+1.98%)

- Table 1: Evaluation of story-to-video generation on DreamStorySet. We compare ours with VideoDirectorGPT and VLogger on character consistency (CLIP and DINO scores), text instructions following and full prompt adherence (CLIP and ViCLIP scores), and event transitions smoothness (DINO score). Our relative improvement over VLogger is highlighted in blue.

Model Consist-attr Dynamic-attr Spatial Motion Action Interaction

Gen-3 (gen 2024) 0.7045 0.2078 0.5533 0.3111 0.6280 0.7900 Dreamina (Dre 2024) 0.8220 0.2114 0.6083 0.2391 0.6660 0.8175 PixVerse (Pix 2024) 0.7370 0.1738 0.5874 0.2178 0.6960 0.8275 Kling (kli 2024) 0.8045 0.2256 0.6150 0.2448 0.6460 0.8475

VideoCrafter2 (Chen et al. 2024) 0.6750 0.1850 0.4891 0.2233 0.5800 0.7600 Open-Sora 1.2 (hpcaitech 2024) 0.6600 0.1714 0.5406 0.2388 0.5717 0.7400 Open-Sora-Plan v1.1.0 (Lab and etc. 2024) 0.7413 0.1770 0.5587 0.2187 0.6780 0.7275 VideoTetris (Tian et al. 2024) 0.7125 0.2066 0.5148 0.2204 0.5280 0.7600 LVD (Lian et al. 2023) 0.5595 0.1499 0.5469 0.2699 0.4960 0.6100 CogVideoX-2B (Yang et al. 2024c) 0.6775 0.2118 0.4848 0.2379 0.5700 0.7250 CogVideoX-2B+SR3A (Ours) 0.7350 (+8.5%) 0.2672 (+26.2%) 0.6123 (+26.3%) 0.2608 (+9.6%) 0.5840 (+2.5%) 0.7625 (+5.2%)

CogVideoX-5B (Yang et al. 2024c) 0.7232 0.2250 0.5845 0.2551 0.6040 0.7995 CogVideoX-5B+SR3A (Ours) 0.7650 (+5.8%) 0.2832 (+25.9%) 0.6875 (+17.5%) 0.3041 (+19.2%) 0.6340 (+5.0%) 0.8725 (+9.1%)

- Table 2: T2V-CompBench evaluation results. Best/2nd best scores for open-sourced models are bolded/underlined. gray indicates close-sourced models, and yellow indicates the best score for close-sourced models.

RAG SR3AI

Fine-Grained Text Full Text Trans. Quality CLIP ViCLIP CLIP ViCLIP DINO Asth. Img. Smth.

ˆ ˆ 23.8 22.5 22.2 22.1 87.1 54.3 61.3 94.3 ˆ ✓ 23.9 23.1 23.5 22.4 92.5 55.4 61.9 98.0 ✓ ˆ 24.7 23.5 23.9 24.0 84.6 55.6 61.9 98.1 ✓ ✓ 24.7 23.7 24.2 24.1 93.6 55.4 62.1 98.1

- Table 3: Ablation studies for the effectiveness of RAG and SR3AI in DREAMRUNNER. Our full model achieves the best text-following ability and event transition smoothness.

#### 4.2 Story-To-Video Generation Evaluation

We compare DREAMRUNNER with prior SoTAs (VideoDirectorGPT (Lin et al. 2023) and VLogger (Zhuang et al. 2024)) on our DreamStorySet dataset for story-to-video generation. For fairness, each scene narration is split into two single-motion descriptions, with corresponding videos later merged into a single-scene video. As shown in Table 1, DREAMRUNNER improves CLIP/DINO scores by 13.1%/33.4% over VLogger, demonstrating the effectiveness of our learned subject prior and region-based LoRA injection for character consistency. To evaluate textfollowing capability, we assess both full-prompt adherence and fine-grained event alignment. DREAMRUNNER improves CLIP/ViCLIP scores consistently on both settings, showing superior alignment with both full-scene and finegrained event descriptions. For transition quality, we compute the DINO-based transition score to measure scene and event consistency. DREAMRUNNER improves transitions by 27.2% over VLogger, highlighting the effectiveness of SR3AI in generating sequential events in a single scene. Lastly, we evaluate visual quality across aesthetic quality, imaging quality, and motion smoothness. DREAMRUNNER enhances aesthetics while slightly improving the other two, demonstrating its capability to generate high-quality videos adhere to complex scene descriptions with smooth event transitions. We provide three additional quality scores from VBench and qualitative examples in the Appendix.

stories for qualitative evaluation. For (2) compositional textto-video generation, we use the T2V-CompBench (Sun et al.

- 2024) to benchmark the performance of DREAMRUNNER, where we select six dimensions except numeracy.

Evaluation Metrics. We evaluate our storytelling videos across multiple dimensions: Character Consistency (Frameto-Reference CLIP/DINO), Text Alignment at both the full narration and fine-grained scene level (Image/Video-toText CLIP/ViCLIP), and Transition Smoothness (Frame-toFrame DINO). For visual quality, we adopt three representative metrics from VBench (Huang et al. 2023): aesthetic quality, imaging quality, and video smoothness (Li et al.

- 2023), with full quality results and metric details provided in the Appendix. We follow similar evaluation metrics to T2VComBench (Sun et al. 2024) for compositional T2V.

#### 4.3 Ablation Studies

Implimentations. We use CogVideoX-2B as our base model for SVG. Test-time-finetuning each prior requires 5min on a single A6000 GPU. For compositional T2V we evaluate with both CogVideoX-2B and CogVideoX-5B.

In this section, we demonstrate the effectiveness of RAG for automatic video retrieval in motion prior learning and SR3AI for fine-grained control over objects and their motions to achieve coherent composition. As shown in Ta-

###### Single-Scene Description References

Single-Scene Description

###### References

[Figure 68]

[Figure 69]

[Figure 70]

In the ware afternoon, teddy bear stands in the garden, surrounded by colorful flowers, basking in the gentle sunlight. Then, he bends over to carefully examine a delicate flower, observing its intricate petals and soft texture. Meanwhile, the robot toy is playing a guitar nearby, filling the air with a soothing melody.

The Mermaid enters a dense kelp forest, wandering through the tall, flowing

.. ..

.. plants. She pauses to inspect some unique shells caught on a kelp strand.

[Figure 71]

[Figure 72]

Row 1: CogVideoX-2B with single-scene description

Row 1: Vlogger

[Figure 73]

[Figure 74]

Row 2: CogVideoX-2B with single-scene description + Character LoRA (globally injected)

Row 2: CogVideoX-2B with single-scene description + Character LoRAs (globally injected)

[Figure 75]

[Figure 76]

Row 3: LLM-generated fine-grained layout plan

Row 3: LLM-generated fine-grained layout plan

[Figure 77]

[Figure 78]

Row 4: DreamRunner (with hard region-based attention between visual latents)

Row 4: CogVideoX-2B + SR3AI (with LLM plan in Row 3 + Character LoRA)

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Row 5: DreamRunner (with full attention between visual latents)

Row 5: CogVideoX-2B + SR3AI (with LLM plan in Row 3 + Character LoRA + RAG Motion LoRAs)

(a) Multi-Character Qualitative Ablation + SoTA Comparison (b) Single-Character Qualitative Ablation

Figure 4: Qualitative comparison and ablations of DREAMRUNNER on SVG. In (a) multi-character example, DREAMRUNNER produces significantly better character consistency compared to other strong baselines, while others fail to maintain object consistency (e.g., VLogger), or fail to generate multiple objects ((a) Row 2,4). In (b) single-character setting, integrating SR3AI and locally-injected priors consistently improve overall quality, complex motion synthesis and coherent composition. Note that in the overlapped regions in (b) row 3, the caption is a merge of the two. For cleaner visualization, we don’t show it here.

ble 3, using SR3AI for enhanced multi-object multi-event binding (2nd row) significantly improves event transition smoothness within a single scene, as well as visual quality and text alignment—likely because decomposing attention into region-based components enables the model to “divide and conquer” more effectively. Incorporating retrievalaugmented motion prior learning (3rd row) further improves video-text similarity for both fine-grained and full-prompt alignment, showing its importance for motion enhancement. Finally, combining both(last row) yields the best results across event transition, text alignment, and visual quality. We also provide additional ablations on RAG pipeline, Layer-seperation strategy for motion prior learning, and computational cost comparison in Appendix B.

a robot-like teddy bear and blurry compositions, highlighting the necessity of region-based LoRA injection. To study attention design, we ablate our region-based attention by comparing hard-regional attention (row 4) and full attention (row 5). While hard-regional attention strictly follows the layout plan (row 3), it limits spatial-temporal continuity due to lack of inter-region interaction. In contrast, our full attention mechanism enables smooth transitions while maintaining spatial-region constraints, supporting high-quality multicharacter customization.

In Fig. 4(b), we present single-character ablations. Using the base CogVideoX-2B with only scene-level text (row 1) leads to vague character/background and missing actions. Injecting global character LoRA (row 2) improves character appearance but still fails on action and transition quality, and degrades background fidelity (e.g., cartoonish kelp forest). Applying SR3AI with layout plans (row 4) improves trajectory control and preserves background fidelity through localized injection, but motion remains limited. Injecting RAGlearned motion priors (row 5) enables clear, fine-grained motion execution (e.g., the mermaid stooping and interacting with kelp), demonstrating the benefit of our motion prior learning and injection strategy. Overall, our full model combines coherent composition with strong motion quality,

#### 4.4 Qualitative Ablations and Comparisons

We provide qualitative comparisons and ablations in Fig. 4.

In Fig. 4(a), we compare DREAMRUNNER with VLogger(Zhuang et al. 2024) for multi-character generation and analyze the effects of region-based attention and LoRA injection. Our method (row 5) generates coherent multicharacter composition and motions, outperforming VLogger (row 1). Using CogVideoX-2B with character LoRAs injected globally (row 2) results in interference, producing

|Method|CLIP ViCLIP|
|---|---|
|CogVideoX-2B<br><br>|23.39 20.84|
|CogVideoX-2B + RAG (w/ single prompt for all videos) CogVideoX-2B + RAG (w/ per-video prompt)|24.01 22.02 24.67 23.04<br><br>|

- Table 4: Effect of RAG and per-video prompt for motion prior learning.

showing the effectiveness of SR3AI and retrieval-augmented prior learning for complex, multi-entity video generation.

#### 4.5 Compositional T2V Generalization

In this section, we demonstrate how our spatial-temporal region-based attention module (SR3A) enhances compositional T2V, as evaluated on T2V-CompBench (Sun et al. 2024). We use SR3A (no LoRA injection) as no customization is required. Given a prompt, we use GPT-4o to generate layout plans, and SR3A ensures coherent composition of objects and events. As shown in Table 2, SR3A significantly improves both CogVideoX-2B and CogVideoX5B (Yang et al. 2024c) across all categories. Specifically, it boosts dynamic attribute binding by over 25%, spatial binding by over 15%, and motion binding by at least 10%, highlighting SR3A’s ability to maintain coherent multi-object compositions, trajectories, and sequential events. It also improves scores on other fine-grained aspects, demonstrating strong control capabilities. Notably, DREAMRUNNER built on CogVideoX-5B achieves SoTA results in five dimensions among open-source models, and surpasses all closed-source models in dynamic attribute binding, spatial binding, and object interactions, highlighting its ability to close the openclosed source model gap and adapt to stronger base models. Qualitative examples per dimension are in Appendix E.

#### 4.6 Effect of RAG and Per-Caption Prompt

We investigate the effectiveness of retrieval-augmented testtime fine-tuning and our per-caption prompt design for learning an enhanced motion prior. Specifically, for each motion in the 64-motion set, we use GPT-4o to generate six prompts and evaluate the average CLIP/ViCLIP scores. As shown in Table 4, applying our approach to CogVideoX-2B improves both scores, with the significant ViCLIP gain indicating better story-video alignment and enhanced motion accuracy. Rows 2–3 further show that per-video prompts outperform single prompts, suggesting that video-specific conditioning helps the model ignore unrelated visual cues and better capture motion-specific patterns. These results confirm that RAG effectively retrieves motion-relevant videos and facilitates the learning of more accurate motion priors.

### 5 Conclusion

In this work, we present DREAMRUNNER, a novel framework for story-to-video generation. Specifically, DREAMRUNNER utilizes a LLM to structure a hierarchical video plan, then introduces retrieval-augmented test-time adaptation to capture target motion priors, and finally generates videos using a novel region-based 3D attention and prior injection module for coherent composition. Experiments on both story-to-video and compositional T2V generation

benchmarks show that DREAMRUNNER outperforms strong baselines and SoTAs in tackling fine-grained complex motions, maintaining multi-scene consistency of multiple objects, and ensuring seamless scene transitions.

### Acknowledgments

This work was supported by DARPA ECOLE Program No. HR00112390060, NSF-AI Engage Institute DRL2112635, DARPA Machine Commonsense (MCS) Grant N6600119-2-4031, ARO Award W911NF2110220, ONR Grant N00014-23-1-2356, Accelerate Foundation Models Research program, and a Bloomberg Data Science PhD Fellowship. The views contained in this article are those of the authors and not of the funding agency.

### References

- 2023. Zeroscope. https://huggingface.co/cerspense/zeroscope_v2_ 576w.
- 2024. Dreamina. https://dreamina.capcut.com/ai-tool/platform.

2024. Flux. https://github.com/black-forest-labs/flux.

2024. Gen-3. https://runwayml.com/blog/introducing-gen-3alpha/.

2024. Kling. https://kling.kuaishou.com/.

2024. PixVerse. https://app.pixverse.ai.

Bansal, H.; Bitton, Y.; Yarom, M.; Szpektor, I.; Grover, A.; and Chang, K.-W. 2024. TALC: Time-Aligned Captions for Multi-Scene Text-to-Video Generation. arXiv preprint arXiv:2405.04682.

Bar-Tal, O.; Chefer, H.; Tov, O.; Herrmann, C.; Paiss, R.; Zada, S.; Ephrat, A.; Hur, J.; Liu, G.; Raj, A.; et al. 2024. Lumiere: A space-time diffusion model for video generation. arXiv preprint arXiv:2401.12945.

Chai, W.; Guo, X.; Wang, G.; and Lu, Y. 2023. Stablevideo: Textdriven consistency-aware diffusion video editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 23040–23050.

Chen, H.; Zhang, Y.; Cun, X.; Xia, M.; Wang, X.; Weng, C.; and Shan, Y. 2024. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. arXiv preprint arXiv:2401.09047.

Chen, T.-S.; Siarohin, A.; Menapace, W.; Fang, Y.; Lee, K. S.; Skorokhodov, I.; Aberman, K.; Zhu, J.-Y.; Yang, M.-H.; and Tulyakov, S. 2025. Multi-subject open-set personalization in video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 6099–6110.

Chen, W.; Ji, Y.; Wu, J.; Wu, H.; Xie, P.; Li, J.; Xia, X.; Xiao, X.; and Lin, L. 2023. Control-a-video: Controllable text-to-video generation with diffusion models. arXiv preprint arXiv:2305.13840.

Fang, Y.; Zhu, H.; Zeng, Y.; Ma, K.; and Wang, Z. 2020. Perceptual Quality Assessment of Smartphone Photography. In IEEE Conference on Computer Vision and Pattern Recognition, 3677–3686.

Fei, H.; Wu, S.; Ji, W.; Zhang, H.; and Chua, T.-S. 2024. Dysen-VDM: Empowering Dynamics-aware Text-to-Video Diffusion with LLMs. In Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR).

Gal, R.; Alaluf, Y.; Atzmon, Y.; Patashnik, O.; Bermano, A. H.; Chechik, G.; and Cohen-Or, D. 2023. An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. In Proceedings of the International Conference on Learning Representations (ICLR).

Girdhar, R.; Singh, M.; Brown, A.; Duval, Q.; Azadi, S.; Rambhatla, S. S.; Shah, A.; Yin, X.; Parikh, D.; and Misra, I. 2023. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709.

Gu, Y.; Wang, X.; Wu, J. Z.; Shi, Y.; Chen, Y.; Fan, Z.; Xiao, W.; Zhao, R.; Chang, S.; Wu, W.; et al. 2024. Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models. Advances in Neural Information Processing Systems, 36.

He, H.; Yang, H.; Tuo, Z.; Zhou, Y.; Wang, Q.; Zhang, Y.; Liu, Z.; Huang, W.; Chao, H.; and Yin, J. 2024. DreamStory: Open-Domain Story Visualization by LLM-Guided Multi-Subject Consistent Diffusion. arXiv preprint arXiv:2407.12899.

He, Y.; Xia, M.; Chen, H.; Cun, X.; Gong, Y.; Xing, J.; Zhang, Y.;

- Wang, X.; Weng, C.; Shan, Y.; et al. 2023. Animate-a-story: Storytelling with retrieval-augmented video generation. arXiv preprint

- arXiv:2307.06940.

Hessel, J.; Holtzman, A.; Forbes, M.; Bras, R. L.; and Choi, Y. 2021. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718.

Hong, W.; Ding, M.; Zheng, W.; Liu, X.; and Tang, J. 2022. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868.

hpcaitech. 2024. Open-sora: Democratizing efficient video production for all.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

- Huang, Y.; Yuan, Z.; Liu, Q.; Wang, Q.; Wang, X.; Zhang, R.; Wan,

- P.; Zhang, D.; and Gai, K. 2025. Conceptmaster: Multi-concept video customization on diffusion transformer models without testtime tuning. arXiv preprint arXiv:2501.04698.

- Huang, Z.; He, Y.; Yu, J.; Zhang, F.; Si, C.; Jiang, Y.; Zhang, Y.; Wu, T.; Jin, Q.; Chanpaisit, N.; et al. 2023. Vbench: Comprehensive benchmark suite for video generative models. arXiv preprint

- arXiv:2311.17982.

Jain, Y.; Nasery, A.; Vineet, V.; and Behl, H. 2024. Peekaboo: Interactive video generation via masked-diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8079–8088.

Jeong, H.; Park, G. Y.; and Ye, J. C. 2023. VMC: Video Motion Customization using Temporal Attention Adaption for Textto-Video Diffusion Models. arXiv preprint arXiv:2312.00845.

Jocher, G. 2020. YOLOv5 by Ultralytics.

Ke, J.; Wang, Q.; Wang, Y.; Milanfar, P.; and Yang, F. 2021. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, 5148– 5157.

Khachatryan, L.; Movsisyan, A.; Tadevosyan, V.; Henschel, R.; Wang, Z.; Navasardyan, S.; and Shi, H. 2023. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. In Proceedings of the International Conference on Computer Vision (ICCV).

Kumari, N.; Zhang, B.; Zhang, R.; Shechtman, E.; and Zhu, J.

- 2023a. Multi-Concept Customization of Text-to-Image Diffusion. In Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR). Kumari, N.; Zhang, B.; Zhang, R.; Shechtman, E.; and Zhu, J.-Y.
- 2023b. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1931–1941.

Lab, P.-Y.; and etc., T. A. 2024. Open-Sora-Plan.

Li, D.; Li, J.; and Hoi, S. C. H. 2023. BLIP-Diffusion: Pre-trained Subject Representation for Controllable Text-to-Image Generation and Editing. In Advances in Neural Information Processing Systems (NeurIPS).

Li, J.; Cho, J.; Sung, Y.-L.; Yoon, J.; and Bansal, M. 2024. SELMA: Learning and Merging Skill-Specific Text-to-Image Experts with Auto-Generated Data. arXiv preprint arXiv:2403.06952.

Li, Z.; Zhu, Z.-L.; Han, L.-H.; Hou, Q.; Guo, C.-L.; and Cheng, M.-M. 2023. Amt: All-pairs multi-field transforms for efficient frame interpolation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9801–9810.

- Lian, L.; Shi, B.; Yala, A.; Darrell, T.; and Li, B. 2023. Llm-grounded video diffusion models. arXiv preprint arXiv:2309.17444.
- Lian, L.; Shi, B.; Yala, A.; Darrell, T.; and Li, B. 2024. LLMgrounded Video Diffusion Models. In The Twelfth International Conference on Learning Representations.

Lin, H.; Cho, J.; Zala, A.; and Bansal, M. 2024. Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model. arXiv preprint arXiv:2404.09967.

Lin, H.; Zala, A.; Cho, J.; and Bansal, M. 2023. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. arXiv preprint arXiv:2309.15091.

Liu, L.; Ma, T.; Li, B.; Chen, Z.; Liu, J.; Li, G.; Zhou, S.; He, Q.; and Wu, X. 2025. Phantom: Subject-consistent video generation via cross-modal alignment. arXiv preprint arXiv:2502.11079.

Long, F.; Qiu, Z.; Yao, T.; and Mei, T. 2024. VideoStudio: Generating Consistent-Content and Multi-Scene Videos. arXiv:2401.01256.

Meral, T. H. S.; Simsar, E.; Tombari, F.; and Yanardag, P. 2025. Contrastive Test-Time Composition of Multiple LoRA Models for Image Generation. ICCV.

Oh, G.; Jeong, J.; Kim, S.; Byeon, W.; Kim, J.; Kim, S.; and Kim, S. 2025. MEVG: Multi-event Video Generation with Text-to-Video Models. In European Conference on Computer Vision, 401–418. Springer.

OpenAI. 2024. Hello, GPT-4 Turbo.

Oquab, M.; Darcet, T.; Moutakanni, T.; Vo, H.; Szafraniec, M.; Khalidov, V.; Fernandez, P.; Haziza, D.; Massa, F.; El-Nouby, A.; et al. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193.

Park, J. W.; Park, S. H.; Koh, J. Y.; Lee, J.; and Song, M. 2024. CAT: Contrastive Adapter Training for Personalized Image Generation. CoRR, abs/2404.07554.

Peebles, W.; and Xie, S. 2023. Scalable diffusion models with transformers. In Proceedings of the International Conference on Computer Vision (ICCV).

Qing, Z.; Zhang, S.; Wang, J.; Wang, X.; Wei, Y.; Zhang, Y.; Gao, C.; and Sang, N. 2024. Hierarchical spatio-temporal decoupling for text-to-video generation. In Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR). Qu, L.; Wu, S.; Fei, H.; Nie, L.; and Chua, T.-S. 2023. Layoutllmt2i: Eliciting layout guidance from llm for text-to-image generation. In Proceedings of the 31st ACM International Conference on Multimedia, 643–654.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning (ICML).

Ren, Y.; Zhou, Y.; Yang, J.; Shi, J.; Liu, D.; Liu, F.; Kwon, M.; and Shrivastava, A. 2024. Customize-a-video: One-shot motion customization of text-to-video diffusion models. arXiv preprint

- arXiv:2402.14780.

Robertson, S.; Zaragoza, H.; et al. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends® in Information Retrieval, 3(4): 333–389.

Ruiz, N.; Li, Y.; Jampani, V.; Pritch, Y.; Rubinstein, M.; and Aberman, K. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22500–22510.

Schuhmann, C.; and Beaumont, R. 2022. LAION-Aesthetics Predictor V1. https://github.com/LAION-AI/aesthetic-predictor. Accessed: 2025-03-07.

Singer, U.; Polyak, A.; Hayes, T.; Yin, X.; An, J.; Zhang, S.; Hu,

- Q.; Yang, H.; Ashual, O.; Gafni, O.; et al. 2022. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792.

Sohn, K.; Jiang, L.; Barber, J.; Lee, K.; Ruiz, N.; Krishnan, D.; Chang, H.; Li, Y.; Essa, I.; Rubinstein, M.; Hao, Y.; Entis, G.; Blok, I.; and Chin, D. C. 2023. StyleDrop: Text-to-Image Synthesis of Any Style. In Advances in Neural Information Processing Systems (NeurIPS).

Sun, K.; Huang, K.; Liu, X.; Wu, Y.; Xu, Z.; Li, Z.; and Liu, X. 2024. T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation. arXiv preprint arXiv:2407.14505.

Tian, Y.; Yang, L.; Yang, H.; Gao, Y.; Deng, Y.; Chen, J.; Wang,

- X.; Yu, Z.; Tao, X.; Wan, P.; Zhang, D.; and Cui, B. 2024. VideoTetris: Towards Compositional Text-to-Video Generation. arXiv:2406.04277.

Wang, J.; Yuan, H.; Chen, D.; Zhang, Y.; Wang, X.; and Zhang, S. 2023a. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571.

Wang, Y.; He, Y.; Li, Y.; Li, K.; Yu, J.; Ma, X.; Li, X.; Chen, G.; Chen, X.; Wang, Y.; et al. 2023b. Internvid: A large-scale videotext dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942.

Wang, Y.; Li, K.; Li, Y.; He, Y.; Huang, B.; Zhao, Z.; Zhang, H.; Xu, J.; Liu, Y.; Wang, Z.; Xing, S.; Chen, G.; Pan, J.; Yu, J.; Wang,

- Y.; Wang, L.; and Qiao, Y. 2022. InternVideo: General Video Foundation Models via Generative and Discriminative Learning. arXiv preprint arXiv:2212.03191.

Wei, Y.; Zhang, S.; Qing, Z.; Yuan, H.; Liu, Z.; Liu, Y.; Zhang, Y.; Zhou, J.; and Shan, H. 2023a. Dreamvideo: Composing your dream videos with customized subject and motion. arXiv preprint

- arXiv:2312.04433. Wei, Y.; Zhang, S.; Yuan, H.; Wang, X.; Qiu, H.; Zhao, R.; Feng,

- Y.; Liu, F.; Huang, Z.; Ye, J.; et al. 2024. DreamVideo-2: Zero-Shot Subject-Driven Video Customization with Precise Motion Control. arXiv preprint arXiv:2410.13830.

Wei, Y.; Zhang, Y.; Ji, Z.; Bai, J.; Zhang, L.; and Zuo, W. 2023b. ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation. In Proceedings of the International Conference on Computer Vision (ICCV).

Weng, W.; Feng, R.; Wang, Y.; Dai, Q.; Wang, C.; Yin, D.; Zhao,

- Z.; Qiu, K.; Bao, J.; Yuan, Y.; et al. 2024. ART-V: Auto-Regressive Text-to-Video Generation with Diffusion Models. In Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR).

Wu, J. Z.; Ge, Y.; Wang, X.; Lei, S. W.; Gu, Y.; Shi, Y.; Hsu, W.; Shan, Y.; Qie, X.; and Shou, M. Z. 2023a. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 7623–7633.

Wu, R.; Chen, L.; Yang, T.; Guo, C.; Li, C.; and Zhang, X. 2023b. Lamp: Learn a motion pattern for few-shot-based video generation. arXiv preprint arXiv:2310.10769.

Xing, Z.; Dai, Q.; Weng, Z.; Wu, Z.; and Jiang, Y.-G. 2024. AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction. arXiv preprint arXiv:2406.06465.

Yang, S.; Hou, L.; Huang, H.; Ma, C.; Wan, P.; Zhang, D.; Chen, X.; and Liao, J. 2024a. Direct-a-Video: Customized Video Generation with User-Directed Camera Movement and Object Motion. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers ’24 (SIGGRAPH Conference Papers ’24), 12. New York, NY, USA: ACM.

- Yang, Y.; Wang, W.; Peng, L.; Song, C.; Chen, Y.; Li, H.; Yang,

- X.; Lu, Q.; Cai, D.; Wu, B.; and Liu, W. 2024b. LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models. arXiv preprint arXiv: 2403.11627.

Yang, Z.; Teng, J.; Zheng, W.; Ding, M.; Huang, S.; Xu, J.; Yang,

- Y.; Hong, W.; Zhang, X.; Feng, G.; et al. 2024c. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072.

Ye, H.; Zhang, J.; Liu, S.; Han, X.; and Yang, W. 2023. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. CoRR, abs/2308.06721.

Yu, S.; Fang, J. Z.; Zheng, S.; Sigurdsson, G.; Ordonez, V.; Piramuthu, R.; and Bansal, M. 2024. Zero-shot controllable image-tovideo animation via motion decomposition. ACM, Multimedia.

Zhang, D. J.; Wu, J. Z.; Liu, J.-W.; Zhao, R.; Ran, L.; Gu, Y.; Gao, D.; and Shou, M. Z. 2024. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. International Journal of Computer Vision, 1–15.

Zhang, Y.; Tang, F.; Huang, N.; Huang, H.; Ma, C.; Dong, W.; and Xu, C. 2023a. MotionCrafter: One-Shot Motion Customization of Diffusion Models. arXiv preprint arXiv:2312.05288.

Zhang, Y.; Wei, Y.; Jiang, D.; Zhang, X.; Zuo, W.; and Tian, Q. 2023b. ControlVideo: Training-free Controllable Text-to-Video Generation. arXiv preprint arXiv:2305.13077.

Zhao, C.; Liu, M.; Wang, W.; Yuan, J.; Chen, H.; Zhang, B.; and Shen, C. 2024. MovieDreamer: Hierarchical Generation for Coherent Long Visual Sequence. arXiv preprint arXiv:2407.16655.

Zhao, R.; Gu, Y.; Wu, J. Z.; Zhang, D. J.; Liu, J.; Wu, W.; Keppo, J.; and Shou, M. Z. 2023. MotionDirector: Motion Customization of Text-to-Video Diffusion Models. arXiv preprint arXiv:2310.08465.

Zheng, G.; Zhou, X.; Li, X.; Qi, Z.; Shan, Y.; and Li, X. 2023. Layoutdiffusion: Controllable diffusion model for layout-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22490–22499.

Zheng, M.; Simsar, E.; Yesiltepe, H.; Tombari, F.; Simon, J.; and Yanardag, P. 2024. Stylebreeder: Exploring and Democratizing Artistic Styles through Text-to-Image Models. CoRR, abs/2406.14599.

Zheng, S.; and Fu, Y. 2024. TemporalStory: Enhancing Consistency in Story Visualization using Spatial-Temporal Attention. arXiv preprint arXiv:2407.09774.

Zhong, M.; Shen, Y.; Wang, S.; Lu, Y.; Jiao, Y.; Ouyang, S.; Yu, D.; Han, J.; and Chen, W. 2024. Multi-lora composition for image generation. arXiv preprint arXiv:2402.16843.

Zhu, J.; Gao, L.; Song, J.; et al. 2024. EchoReel: Enhancing Action Generation of Existing Video Diffusion Models. arXiv preprint

- arXiv:2403.11535.

Zhuang, S.; Li, K.; Chen, X.; Wang, Y.; Liu, Z.; Qiao, Y.; and Wang, Y. 2024. Vlogger: Make your dream a vlog. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8806–8817.

### A Additional Method Details

#### A.1 Retrieval Processes

Our retrieval process consists of the following steps (using the example “sitting” as the query motion):

- 1) Initial Retrieval with BM25: We use the text-only BM25 score (Robertson, Zaragoza et al. 2009) based on video captions in the dataset to retrieve 400 candidate videos for the query. To ensure the retrieved videos are human motion-centric, we add "person is" at the beginning of the query ("person is sitting").
- 2) Attribute-Based Filtering: We refine the candidate pool by filtering videos based on key attributes such as duration (at least 2 second), frame count (at least 40 frames), and aspect ratio (width/height at least 0.9). This ensures that the selected videos align with the requirements of the video generator, excluding videos that are too short or have extreme aspect ratios.
- 3) Clip Segmentation via Object Tracking: We track individuals within videos using YOLOv5 (Jocher 2020) and segment clips into human-centered segments based on the tracking results, keeping meaningful human-focused content.
- 4) Scoring with CLIP and ViCLIP (Radford et al. 2021; Wang et al. 2022): To ensure the fidelity between the segmented video clips and the query, we compute semantic similarity scores to the query text (e.g. “the person is sitting”) using CLIP and ViCLIP for each segmented clip. The CLIP score is computed by sampling eight frames and averaging frame-query scores, while the ViCLIP score is directly computed on the full video and query. We select the top 20 videos that satisfy the average scores of CLIP and ViCLIP > 0.2. We retain the top four videos based on their ranking if fewer than four videos meet this threshold.

#### A.2 Motion Prior Training

We train the LoRAs on our filtered top-ranked videos with all other backbone parameters frozen, using two diffusion losses: a standard diffusion loss Lorg, which is a reconstruction loss of all the video frames, and an appearance-debiased temporal loss Lad, which decouples the motion space from appearance space in the latent space, focusing on only reconstructing the motions in the videos. Formally,

Lorg “ Ez0,y,ϵ„Np0,1q,t„Up0,Tqr||ϵ ´ ϵθpzt, t, yq||2s (1)

where z0 is the latent encoding of the training videos, y is the text prompt condition, ϵ is the Gaussian noise added to the latent space, ϵθ is the predicted noise, and t is the denoising time step. The appearance-debiased temporal loss optimizes the normalized latent space:

ϕpϵq “ aβ2 ` 1ϵ ´ βϵanchor (2)

where ϵanchor is the anchor among the frames from the same training data, and β is the strength factor that controls the strength of the debiasing. Lad is defined as:

Lad “ Ez0,y,ϵ„Np0,1q,t„Up0,Tqr||ϕpϵq ´ ϕpϵθpzt, t, yqq||2s (3)

In the end, we update the model using a combined motion loss function defined as Lmotion “ Lorg ` Lad. Notably, we do not apply scaling to each loss term throughout experiments in the paper, highlighting the robustness and simplicity of hyperparameter selection in DREAMRUNNER.

#### A.3 Region-Based LoRA Injection

Low-Rank Adaptation (LoRA) (Hu et al. 2021) was introduced to efficiently adapt large pre-trained language models to various downstream tasks. More recently, LoRA has been extended to textto-image and text-to-video generation, enabling lightweight subject and motion customization (Ruiz et al. 2023; Zhao et al. 2023). Instead of updating the full weight matrix W, LoRA applies a lowrank decomposition, modifying it as:

###### W “ W0 ` ∆W “ W0 ` BA, (4)

where W0 P Rdˆk denotes the original model weights, and B P Rdˆr, A P Rrˆk are the learnable low-rank matrices with r ! d, k.

During inference, the LoRA layer’s output is the summation of the original layer’s output W0x and low-rank matrices’ output BAx:

Wx “ W0x ` BAx (5)

where x P Rkˆc (c is the number of latent tokens, while k is the dimension of each token) is the latents. By restricting updates to these low-rank factors, LoRA significantly reduces computational overhead compared to full fine-tuning approaches. Additionally, its plug-and-play nature makes it highly efficient for deployment and sharing across different pre-trained models.

To inject LoRA locally into specific regions, we compute masks for the latents corresponding to each LoRA and mask the input of other regions. For example, considering the character LoRAs in main paper’s Fig. 1, we compute the output when forwarding the latents x to this layer as:

Wx “ W0x ` AwitchBwitchpMaskwitch ¨ xq ` AcatBcatpMaskcat ¨ xq “ pW0 ` AwitchBwitchqpMaskwitch ¨ xq ` pW0 ` AcatBcatqpMaskcat ¨ xq ` W0pp1c ´ Maskwitch ´ Maskcatq ¨ xq

(6)

where 1c is a c dimension all-ones vector and ¨ denotes the dot product, so Mask ¨ x denotes the dot product between the c dimension boolen mask and the k ˆ c latent. In this case, the cat’s LoRA only acts on the cat-related region (Maskcat ¨ x), ensuring that the cat-region output is similar to the output obtained when applying a classical cat LoRA globally:

WxMaskwitch“1 “pW0 ` AwitchBwitchqpMaskwitch ¨ xMaskwitch“1q `pW0 ` AcatBcatqpMaskcat ¨ xMaskwitch“1q `W0pp1c ´ Maskwitch ´ Maskcatq ¨ xMaskwitch“1q “pW0 ` AwitchBwitchqp1 ¨ xMaskwitch“1q `pW0 ` AcatBcatqp0 ¨ xMaskwitch“1q `W0p0 ¨ xMaskwitch“1q “pW0 ` AwitchBwitchqxMaskwitch“1

(7)

Similarly, the witch’s LoRA only modifies the witch-related region (Maskwitch¨x), preserving consistency with traditional LoRAbased adaptation for the witch. For other regions unrelated to both the witch and the cat, the output remains unchanged, as the mask

ensures that no LoRA modifications are applied, making it equivalent to the output without any LoRA injection. For the Motion LoRAs in main paper’s Fig. 1„ we follow the similar style to inject them locally:

Wx “ W0x ` AwalkingBwalkingpMaskwalking ¨ xq `AwhisperingBwhisperingpMaskwhispering ¨ xq

(8)

Such a region-based LoRA injection design allows multiple LoRAs to act on different regions, ensuring seamless multi-LoRA integration while preventing interference between unrelated entities or motions, thereby enabling precise multi-character, multi-motion customization.

### B Additional Comparisons

#### B.1 Ablations of RAG pipeline

We provide additional experiments in Section 3.2 to demonstrate the effectiveness of our data processing approach within the retrieval pipeline. Table 5 primarily evaluates the impact of the maximal number of retrieved videos and the use of CLIP and ViCLIP for filtering. For efficiency, we evaluate a subset of eight motions selected from the full pool of 64 motions. The results indicate that, compared to the CogVideoX zero-shot baseline (Row 1), retrieving videos without any filtering (Row 2) improves performance, even though BM-25 (Robertson, Zaragoza et al. 2009) retrieval introduces some noise. This highlights the importance of retrieval itself. Furthermore, adding CLIP and ViCLIP as filters (Row 4) further enhances performance, showcasing the benefit of using semantically aligned videos for improved motion learning. Additionally, retrieving a sufficient number of videos is critical, as evidenced by Row 3, where limiting retrieval to a maximum of three videos results in poorer performance compared to retrieving 20 videos (Row 4).

|Max. #Retrieval CLIP+ViCLIP filter<br><br>|CLIP ViCLIP|
|---|---|
|0 ˆ|23.42 20.56<br><br>|
|20 ˆ<br><br>|24.01 22.51|
|3 ✓|24.45 22.80<br><br>|
|20 ✓<br><br>|25.47 23.66|

Table 5: Pipeline component ablation on retrievalaugmented test-time adaptation for learning a better motion prior.

#### B.2 Ablations of Odd-Even Layer Separation for Motion Prior Learning

During motion prior learning, since our backbone does not explicitly separate spatial and temporal modeling, we manually designate even layers as spatial and odd layers as temporal to decouple the learning of spatial and temporal LoRAs. We hypothesize that this design helps learn appearance-debiased, motion-focused priors, while the interleaved structure allows adaptation across lowlevel structures and high-level semantics, as both spatial and temporal LoRAs are injected throughout the model. We validate this hypothesis in Table 6. Specifically, removing spatial LoRAs (row 1) degrades performance compared to row 3, indicating that separating spatial and temporal LoRAs enhances motion-focused learning. Additionally, the Half-Half approach (injecting spatial LoRAs into the first half of layers and temporal LoRAs into the second half, row 2) underperforms compared to interleaved injection, highlighting the importance of multi-level learning across the entire model.

|Method<br><br>|CLIP ViCLIP|
|---|---|
|No Appearance-Debiased|24.4 22.5<br><br>|
|Half-Half Injection<br><br>|24.9 23.1|
|Interleaved Injection (Ours)<br><br>|25.5 23.7|

Table 6: LoRA injection ablations.

#### B.3 Computational Cost Comparison

Since our method is training-free and only involves test-time finetuning (TTF), we report the optimization time solely for each method for reference. Specifically, our methods’ TTF takes approximately 0.2 GPU hours per motion, totaling 3–4 GPU hours for a full story with 15–20 motions.

In comparison, training-intensive baselines, VLogger and VideoDirectorGPT, require around 6K and 400 GPU hours, respectively. Notably, our method still outperforms these baselines even without RAG and TTF (Table 3), with only spatial-temporal regional attention involved.

### C Evaluation Metrics

We provide detailed evaluation metrics for assessing generated storytelling videos across multiple dimensions in Section 4.1.

- • Similarity to Reference Images: We assess the alignment between the generated videos and reference images using Image-Image CLIP (Hessel et al. 2021) and DINO (Oquab et al. 2023) scores. The evaluation is conducted by averaging the CLIP/DINO similarity scores between each frame of the generated video and the reference images, with CLIPL14 (336px) (Hessel et al. 2021) and DINOv2-B16 (Oquab et al.

2023) as image encoders.

- • Similarity to Full Narration per Scene: We measure the alignment between the full narration and the generated videos for each scene using Image-Text CLIP (Hessel et al. 2021) and VideoText ViCLIP (Wang et al. 2023b) scores. For the CLIP score, we uniformly sample eight frames from the single-scene video and compute the average score between each frame and the full narration. For the ViCLIP score, we directly use the alignment score between the video and the narration. Per-scene scores are averaged to obtain the overall score for the multi-scene storytelling video.
- • Fine-Grained Text Alignment per Scene: We also assess fine-grained alignment to textual descriptions using Image-Text CLIP (Hessel et al. 2021) and Video-Text ViCLIP (Wang et al. 2023b) scores. In our story, each narration contains two motions, which we decouple into two consecutive single-motion descriptions using an LLM (OpenAI 2024). For each generated singlescene video, we divide it into two segments at the temporal midpoint. We then compute the CLIP/ViCLIP scores between the first segment and the first description and between the second segment and the second description. The average of these two scores provides the per-scene score, and the per-scene scores are further averaged to compute the overall score for the storytelling video.
- • Transition Quality: We assess whether the single-scene video achieves smooth transitions between two motions. To evaluate this, we uniformly sample four frames from the video per scene and calculate the average DINO similarity between adjacent frames. A higher transition score indicates smoother transitions, as it reflects minimal changes in the background across frames.
- • Motion Smoothness: Following (Huang et al. 2023), we utilize motion priors from a video frame interpolation model (Li et al.

2023) to quantify the smoothness of motion in generated videos (see Supplementary for details), specifically, given one video, we drop the odd-number frames and then use the interpolation

model to infer the dropped frames. Then we compute the Mean Absolute Error between the estimated frames and the dropped frames.

- • Aesthetic Quality: We evaluate the artistic and visual appeal of each generated video frame using the LAION aesthetic predictor (Schuhmann and Beaumont 2022) frame-by-frame. This metric reflects aesthetic aspects such as composition, color richness and harmony, photorealism, naturalness, and overall artistic quality.
- • Imaging Quality: Imaging quality measures distortions such as over-exposure, noise, and blur in generated frames. We evaluate this using the MUSIQ (Ke et al. 2021) image quality predictor frame-by-frame, which is trained on the SPAQ dataset (Fang et al. 2020).

### D Visual Quality Evaluation for SVG

We evaluate the generated videos using a comprehensive set of visual quality and consistency metrics, including aesthetics, imaging quality, motion smoothness, temporal flickering, subject consistency, and background consistency, as shown in Table 7 and Table 8. Our method achieves the best overall performance across all metrics, demonstrating its effectiveness in producing high-quality and consistent videos. Moreover, the ablation results indicate that introducing the proposed RAG and SR3AI modules does not harm the visual quality, as the full model maintains or improves performance compared to its variants.

### E Compositional T2V Examples

In this section, we present qualitative examples demonstrating the capabilities of DREAMRUNNER in compositional text-to-video generation. DREAMRUNNER effectively generates videos with accurate action binding to different characters, consistent attributes across objects, dynamic attribute changes, motion control, object interactions, and appropriate spatial relationships between objects. Note that in the overlapped regions, the caption is a merge of all involved captions. For cleaner visualization, we don’t show the merged caption in the layout plans.

For action binding, as shown in Figure 6, DREAMRUNNER generates distinct motions for two objects (e.g., A monkey in a pilot jacket and a parrot with aviator goggles ready for flight), ensuring the actions are correctly bound to their respective objects without interference.

For consistent attribute binding, as illustrated in Figure 7, our approach maintains separate attributes for different objects (e.g., Rectangular briefcase swinging near a hexagonal fountain) without any overlap or inconsistency.

For dynamic attribute changes, as shown in Figure 8, DREAMRUNNER naturally transitions object attributes over time (e.g., A timelapse of a piece of bread initially fresh, then growing moldy).

For motion control, as depicted in Figure 9, DREAMRUNNER successfully directs object movements in different trajectories (e.g., A squirrel climbs upward on a tree and A drone is gradually descending to the ground in a park).

For object interactions, as illustrated in Figure 10, interactions are accurately modeled, adhering to the real world rules (e.g., Cat’s paw presses on a soft pillow).

Finally, for spatial relationships, as shown in Figure 11, DREAMRUNNER generates rare or imaginative configurations (e.g., a duck positioned under a spacecraft) while maintaining spatial coherence.

These examples highlight the strong performance of DREAMRUNNER in generating high-quality compositional text-to-video outputs.

### F Characters

We provide character examples in Figure 12, where the first four are generated using FLUX (flu 2024), and the others are collected from existing customization datasets (CustomConcept101 (Kumari et al. 2023b) and Dreambooth (Ruiz et al. 2023)).

### G Single-Character Examples

In this section, we present qualitative examples of video generation featuring a single main character. As shown in Figure 14 and Figure 13, DREAMRUNNER generates consistent characters throughout the entire story. Additionally, in each scene, DREAMRUNNER effectively captures multiple events, such as the mermaid first wandering through the plants and then examining unique shells.

### H Multi-Character Examples

In this section, we present qualitative examples of video generation featuring multiple characters. As illustrated in Figure 15 and Figure 16, DREAMRUNNER generates multi-scene, multi-character videos where each character retains its own motion and interacts seamlessly with others, without any interference. For instance, in Figure 15, the witch is shown pouring ingredients while the cat wanders around the room. Even as the cat approaches the witch, their motions remain independent, and the appearance of both characters is consistently preserved throughout the scene.

### I LLM Prompts

We provide detailed LLM prompts for both high-level plans and fine-grained plans in Listing 1 and Listing 3. respectively. For highlevel plans, we use a simple in-context example with instructions, while fine-grained plans require reasoning before generating the output. Example outputs are shown in Listings 2 and 4. We also provide our example input output for merging captions in overlapped regions in Listing 5. For multi-character scenarios, we adapt character-specific words and examples, limit required motions to a maximum of four, and enforce non-overlapping spatial regions for each character.

### J Multi-Character Video Generation with Overlapping Regions

We show an example of DREAMRUNNER handling overlapped regions in a multi-character setting (see Fig. 5). In this case, the bounding boxes of the cat and the witch slightly overlap, leading to visual overlap between the characters themselves. This demonstrates DREAMRUNNER’s ability to manage such interactions.

### K Limitation and Future Work

Limitations. DREAMRUNNER build s on a diffusion-based video generation framework and trains lightweight adapters on frozen backbones. As such, its performance and output quality are inherently constrained by the capabilities of the underlying backbone models. If the backbone struggles with rare compositions, complex motions, or fine-grained details, DREAMRUNNER may inherit these weaknesses.

Future work. Future improvements may come from adopting stronger and more generalizable backbone models to enhance the overall visual quality. Additionally, advanced LoRA-merging techniques such as CLoRA (Meral et al. 2025) can be explored to better coordinate multiple character streams and improve quality in overlapping regions. More broadly, developing architectures with stronger compositional reasoning and multi-entity awareness remains an important direction.

[Figure 83]

[Figure 84]

Figure 5: Generated multi-character videos with slightly overlapping regions.

Method Aesthetics Imaging Smoothness Temp. Flickering Subject Consistency BG Consistency Overall

VideoDirectorGPT (Lin et al. 2023) 42.3 60.3 94.3 91.5 75.6 87.8 75.3 VLogger (Zhuang et al. 2024) 43.4 61.2 96.2 95.7 84.1 91.0 78.6 DREAMRUNNER (Ours) 55.4 (+27.65%) 62.1 (+1.47%) 98.1 (+1.98%) 96.2 (+0.52%) 91.2 (+8.44%) 92.3 (+1.43%) 82.6 (+5.03%)

- Table 7: Evaluation of visual quality and consistency. Metrics include aesthetics, imaging, motion smoothness, temporal stability, subject/background consistency, and overall quality. DREAMRUNNER Achieves the best quality compared with others.

RAG SR3AI Aesthetics Imaging Smoothness Temp. Flickering Subject Consistency BG Consistency Overall ˆ ˆ 54.3 61.3 94.3 95.9 91.0 92.2 81.50 ˆ ✓ 55.4 61.9 98.0 96.3 90.9 92.3 82.47 ✓ ˆ 55.6 61.9 98.1 96.0 91.1 92.0 82.45 ✓ ✓ 55.4 62.1 98.1 96.2 91.2 92.3 82.55

- Table 8: Ablation on quality dimensions. Overall is the mean of all six metrics. Adding RAG and SR3AI doesn’t harm the overall visual quality of the backbone model. The last row is our full method DREAMRUNNER.

[Figure 85]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A monkey in a pilot jacket and a parrot with aviator goggles ready for flight

[Figure 86]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A dolphin jumps next to a speeding boat

[Figure 87]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A squirrel gathers nuts and a bat hangs from a tree branch

[Figure 88]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A kid and a penguin watch a movie in the cinema

- Figure 6: Qualitative results of DREAMRUNNER generated with prompts characterizing action binding. SR3A denotes our spatial-temporal region-based attention module.

[Figure 89]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

Rectangular briefcase swinging near a hexagonal fountain

[Figure 90]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

Wooden car speeding past a porcelain tree

[Figure 91]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

White tractor plowing near a green farmhouse

[Figure 92]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

Big hearts and small stars floating upwards

- Figure 7: Qualitative results of DREAMRUNNER generated with prompts characterizing consistent attribute binding. SR3A

[Figure 93]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A cat's green eyes turn a red

[Figure 94]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A timelapse of a piece of bread initially fresh, then growing moldy

[Figure 95]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A timelapse of a flower bud blooming into a full flower

[Figure 96]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

Clear blue sky turns stormy gray

- Figure 8: Qualitative results of DREAMRUNNER generated with prompts characterizing dynamic attribute binding. SR3A

[Figure 97]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A drone is gradually descending to the ground in a park

[Figure 98]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A squirrel climbs upward on a tree

[Figure 99]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A paper airplane gliding to the right across a classroom

[Figure 100]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A bird flies upwards in a garden

- Figure 9: Qualitative results of DREAMRUNNER generated with prompts characterizing motion binding. SR3A denotes our spatial-temporal region-based attention module.

[Figure 101]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

Cat's paw presses on a soft pillow

[Figure 102]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

Sorcerer summons water from thin air, flooding room

[Figure 103]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

Coach motivates athlete during a tough workout

[Figure 104]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

Tiger coach whistles at soccer practice

- Figure 10: Qualitative results of DREAMRUNNER generated with prompts characterizing object interactions. SR3A denotes

[Figure 105]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A llama walking on the right side of a motorcycle in a city street

[Figure 106]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A chandelier hanging above a grand piano in a busy hotel lobby

[Figure 107]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A duck waddling below a spacecraft

[Figure 108]

CogVideoX 5B

CogVideoX 5B + SR3A

LLM Layout Plan

A lion sitting behind a chicken

- Figure 11: Qualitative results of DREAMRUNNER generated with prompts characterizing spatial relationships. SR3A denotes

Characters Generated by FLUX

[Figure 109]

[Figure 110]

warrior

witch

[Figure 111]

[Figure 112]

mermaid astronaut

Characters Collected from Web

[Figure 113]

[Figure 114]

cat 1 cat 2

[Figure 115]

[Figure 116]

dog 1 dog 2

[Figure 117]

[Figure 118]

teddy bear robot toy

Figure 12: Qualitative results of DREAMRUNNER generated with a single character (mermaid).

[Figure 119]

The astronaut prepares for landing on the unknown planet. He carefully examines the spaceship’s control panel, ensuring all systems are ready, and stretches his arms, feeling the thrill of discovery ahead.

[Figure 120]

After a smooth landing, the astronaut steps out onto the dusty, alien soil. He plants a flag beside his ship, marking his arrival on this strange new world with a sense of pride.

[Figure 121]

Moving cautiously, the astronaut climbs over jagged rocks, making his way through the rough terrain. He pauses atop a high ledge, using the surveying machine on the alien landscape stretching endlessly before him.

[Figure 122]

The astronaut enters a mysterious forest filled with strange, luminescent plants. He kneels down to inspect a glowing flower and collects a small soil sample, documenting his discoveries.

[Figure 123]

Stumbling upon ancient ruins, the astronaut brushes sand off a strange relic carved with unknown symbols. He takes a photo for further study, intrigued by the mysteries this planet holds.

[Figure 124]

As the day ends, the astronaut finds a quiet cliff and sits on a rock, watching a stunning alien sunset. He reflects on his journey, feeling a deep connection to this unfamiliar world.

Figure 13: Qualitative results of DREAMRUNNER generated with a single character (astronaut).

[Figure 125]

The Mermaid begins her day through the vibrant coral reef. She swims smoothly between colorful corals and touches the corals with her curiosity.

[Figure 126]

The Mermaid diving toward the old, mysterious sunken ship, exploring its hidden corners. Her movements are careful as she examines every detail, wondering what secrets the ship might reveal.

[Figure 127]

In the open ocean, the Mermaid glides effortlessly through the water. She encounters a friendly dolphin and waves as it playfully circles around her before swimming off.

[Figure 128]

The Mermaid enters a dense kelp forest, wandering through the tall, flowing plants. She pauses to inspect some unique shells caught on a kelp strand.

[Figure 129]

The Mermaid ventures into a shadowy sea cave, swimming cautiously. She finds a glowing stone on the cave floor and picks it up, mesmerized by its mysterious light.

[Figure 130]

The Mermaid finishes her journey by a secluded lagoon, lying near the shore. With a playful motion, she splashes water around, basking in the warm sunlight to end her day.

Figure 14: Qualitative results of DREAMRUNNER generated with a single character (mermaid).

[Figure 131]

The witch begins her day in her potion room. As she pours ingredients into a bubbling cauldron, her cat wanders curiously around the room. The witch stirs carefully as the mixture begins to glow, casting a mystical light across the space.

[Figure 132]

At a mystical lake deep in the forest, the witch kneels by the water's edge. Her cat swimming through the shallows, splashing playfully and sending ripples across the surface. The witch watches as shimmering visions emerge beneath the water, offering guidance for her next magical journey.

[Figure 133]

- As night falls, the witch returns home to her cozy study. Her cat sits gracefully on the windowsill, watching the witch. The witch writes about her magical adventures in her journal, sipping a calming tea as she reflects on the day's enchantments and mysteries.
- At the edge of a quiet village, the witch takes out her spellbook and gestures with her wand, casting a protective charm over the nearby homes, ensuring the villagers' safety from lurking shadows. Nearby, her cat lies stretched out on a warm patch of grass,

watching her intently with half-closed eyes.

[Figure 134]

[Figure 135]

Under the glow of the moon and surrounded by luminous flowers, the witch walks through her mysterious forest, while her cat quietly arranges flowers nearby. As they move through the magical scenery, the witch pauses to wave her hands toward the cat, sharing a moment of connection amidst the enchanting glow.

[Figure 136]

To end her day, the witch walks among the moonlit flowers in her garden, whispering a farewell to each plant. Her cat walks softly beside her, its tail swaying gently with each step. With one last glance at the shimmering stars above, she heads inside, her heart content from a day filled with magic.

Figure 15: Qualitative results of DREAMRUNNER generated with multiple characters (witch and cat 1).

[Figure 137]

The warrior arrives at the training grounds early in the morning with his dog, who wanders nearby. He stretches his arms to warm up, then picks up his sword and swings it in practice, honing his technique.

[Figure 138]

In the courtyard, the warrior stands poised to begin his kung fu practice. His dog lies to rest a short distance away, watching quietly. The warrior practice kung fu with precision and strength, grounding himself with every step and feeling the energy flow through him.

[Figure 139]

After hours of training, the warrior heads to the riverbank. His dog eagerly swims in the river with joy. The warrior fills his flask with fresh water from the river and drinks deeply, refreshing himself before continuing with his routine.

[Figure 140]

Amid the tranquil bamboo grove, the warrior draws his bow, each arrow flying true. Nearby, his dog walks leisurely among the bamboo, its steps soft and unhurried. After a series of shots, the warrior sits down on the ground to rest, taking in the quiet and steadying his breath.

[Figure 141]

As night falls, the warrior sets up camp and lights a small fire, with His dog sitting faithfully by his side. The warrior sharpens his sword with a whetstone, ensuring it is ready for another day of training and adventure.

Figure 16: Qualitative results of DREAMRUNNER generated with multiple characters (warrior and dog 2).

- Listing 1: The LLM prompt for high-level planning (Section 3.1, Story-Level Croase-Grained Planning). User may input the high-level planning to generate at the [Input] highlighted in blue. We highlight scene, motions, narrations using same colors as Section 3.1.

Consider you are an expert in writing stories. I will provide you with a topic, and you need to create a multi-scene story with 5 to 8 scenes. Each scene should describe the events taking place, emphasizing highlighting human actions or motions. Limit each scene to a maximum of 2 distinct human motions. Your output should include both the scene and motions as

well as the narration, where the motion should be in present progressive. [Input] Teddy’s one day [Output]

- Scene 1: bedroom Motions: waking up, stretching arms Narration: Teddy starts his day in his bedroom. He wakes up from the bed and stretches his arms beside it, getting ready for a new day.
- Scene 2: kitchen Motions: pouring cereal, drinking Narration: Teddy is in the kitchen for breakfast. He pours cereal into a bowl in the kitchen, and drinks

the soup, preparing his morning meal.

- Scene 3: backyard Motions: watering plants, examining flower Narration: Teddy spends some time outside in the backyard. He waters the garden plants and kneels down to examine a flower, enjoying the fresh air and tending to the plants.
- Scene 4: park Motions: jogging, breathing Narration: Teddy visits the park for some exercise. He jogs in the park, then he becomes very tired so he breaths heavily, regaining his energy.
- Scene 5: forest Motions: hiking, sitting Narration: Teddy goes hiking on a forest trail. He hikes up the trail and then sits on a rock to rest, appreciating the tranquility of nature.
- Scene 6: living room Motions: sitting, reclining Narration: Teddy winds down at the end of the day in the living room. He sits on the couch and then reclines to relax, ending his day in comfort and peace. [Input] Mermaid’s Adventure

- Listing 2: The example of the output from the high-level planning prompt, from Listing 1. We highlight scene, motions, narrations using same colors as Section 3.1. [Output]

- Scene 1: Coral Reef Motions: swimming, touching Narration: The Mermaid begins her day through the vibrant coral reef. She swims smoothly between colorful corals and touches the corals with her curiosity.
- Scene 2: Sunken Ship Motions: diving, exploring Narration: The Mermaid diving toward the old, mysterious sunken ship, exploring its hidden corners. Her movements are careful as she examines every detail, wondering what secrets the ship might reveal.
- Scene 3: Open Ocean Motions: gliding, waving Narration: In the open ocean, the Mermaid glides effortlessly through the water. She encounters a friendly dolphin and waves as it playfully circles around her before swimming off.
- Scene 4: Kelp Forest Motions: wandering, inspecting Narration: The Mermaid enters a dense kelp forest, wandering through the tall, flowing plants. She pauses to inspect some unique shells caught on a kelp strand.
- Scene 5: Sea Cave Motions: swimming, picking up Narration: The Mermaid ventures into a shadowy sea cave, swimming cautiously. She finds a glowing stone on the cave floor and picks it up, mesmerized by its mysterious light.
- Scene 6: Lagoon Motions: lying, splashing Narration: The Mermaid finishes her journey by a secluded lagoon, lying near the shore. With a playful motion, she splashes water around, basking in the warm sunlight to end her day.

- Listing 3: LLM prompt for fine-grained frame-level plan (Section 3.1, Scene-Level Fine-Grained Planning) per narration. We highlight Background, [entity, motion, caption] and layout [left, top, right, bottom] using same colors as Section 3.1.

Assuming the frame size is normalized to the range 0-1, you need to give a possible 6-frame layout plan at 1fps with the relevant regions, containing entity with corresponding motion, caption and bounding box involved in the input example motion and narration. You should follow these instructions:

- 1. [Background and Regions] You need to give a background of the videos, then list all regions with related entity, motion, caption, and bounding box, for each frame.
- 2. [Bounding Box Size] Each bounding box of the region is one rectangle or square box in the layout, and the size of boxes should be AS LARGE AS POSSIBLE. The width and height of each bounding box should be at least 0.2.
- 3. [Bounding Box Format] Every region should contain one related motion and caption, with the bounding boxes in the format of [[entity1, motion1, caption1], [left, top, right, bottom]].

If the entity doesn’t involve any motion, use "none" as the motion (e.g. ["table", "none", "a table in the forest"]).

- 4. [Captions and Motions)] "IMPORTANT" For each entity, you should give a caption containing the entity and motion, with the provided background.
- 5. [Allowing Overlaps for Interaction Contexts] Regions can overlap if necessary, particularly when entities are interacting. For example, in "Teddy is pouring water into a bowl," you can have a region for "a bowl" overlapping or near Teddy’s region. Similarly, in " Teddy is sitting by the river," you can have a region for "river bank" overlapping or positioned beneath Teddy’s region.

- 5. [Allowing Overlaps for Interaction Contexts] Regions can overlap if necessary, particularly when entities are interacting. For example, in "Teddy is pouring water into a bowl," you can have a region for "a bowl" overlapping or near Teddy’s region. Similarly, in " Teddy is sitting by the river," you can have a region for "river bank" overlapping or positioned beneath Teddy’s region.
- 6. [Interaction with Objects] For an object listed in previous frames, when the entity is interacting with it in the current frames, you can still list the interacted object, then also use a whole region describing the character interacting with the object (e.g. [["rock", "none", "a rock in the forest"], [0.0, 0.8, 0.2, 1.0]] before entity Teddy is interacting with it, [["rock", "none", "a rock in the forest"], [0.0, 0.8, 0.2, 1.0]], [["Teddy", " sitting", "Teddy is sitting on the rock in the forest"], [0.0, 0.0, 0.4, 1.0]] when interacting).
- 7. [Static Interaction Objects] In most cases, the interacted objects should be static. For example, "Teddy is greeting to people" then "people" should have no bounding box changes. For

some non-static objects like "the bottle" of "Teddy is drinking water", ignore such objects to be listed seperatedly.

- 8. [Motion Limitation] You should not use other motions outside the provided narration. The motion and caption should all be in PRESENT PROGRESSIVE.
- 9. [Motion Duration] One motion should last at least two frames.
- 10. [Reasoning] Add reasoning before you generate the region plan, explaining how you will allocate different events/motions to different frames and how the entity will be moving (e.g

., left to right or staying static).

- 11. [Smart Motion Allocation] Allocate motions smartly by reasoning the frames they need (e.g

., static motions like standing will require fewer frames but walking may need more). The background should not contain information about the characters (e.g., Teddy’s xxx is not allowed).

- 12. [Common Sense in Layout] Make sure the locations of the generated bounding boxes are consistent with common sense. You need to generate layouts from the close-up camera view of the event. The layout difference between two adjacent frames should not be too large, considering the small interval.
- 13. [Motions for Large Position Changes] If you want to move the entity largely by changing its bounding boxes (e.g., from most right to most left), make sure the motion naturally involves position changes (e.g., walking, running, flying, riding a bike). Otherwise, avoid big changes in related bounding boxes (e.g., cooking, jumping, playing guitar, etc., where bounding boxes should not change significantly).
- 14. [Big Regions Preference] We prioritize using large regions for main entities. For example , [0.2, 0.0, 0.8, 1.0] for a main teddy bear, as long as it fits the scene. Do not use small regions (like with only 0.2~0.3 width and height) as possible as you can.
- 15. [Motion Caption Consistency] Ensure that each caption accurately reflects the stated motion. For example, if the motion is "walking" the caption should match this exactly, such

as "Teddy is walking in the park." Avoid any inconsistencies where the motion described in the caption does not match the stated motion, such as "Teddy is running in the park" when the

motion is "walking." This consistency maintains clarity and alignment with the narration.

Use format:

- *Reasoning* reason
- *Plan* Background: background

- Frame_1: [[entity1, motion1, caption1], [left, top, right, bottom]], [[entity2, motion2, caption2], [left, top, right, bottom]], ..., [[entity3, motion3, caption3], [left, top, right, bottom]]
- Frame_2: [[entity1, motion1, caption1], [left, top, right, bottom]], [[entity2, motion2, caption2], [left, top, right, bottom]], ..., [[entity3, motion3, caption3], [left, top, right, bottom]]

... Frame_6: [[entity1, motion1, caption1], [left, top, right, bottom]], [[entity2, motion2, caption2], [left, top, right, bottom]], ..., [[entity3, motion3, caption3], [left, top, right, bottom]]

Reasoning: ... Example 1: [Input] Motion: walking, sitting Narration: Teddy goes to a forest. He walks on the trail and then sits on a rock to rest, appreciating the tranquility of nature. [Output]

*Reasoning* Listed motions are: walking, sitting. Related entities and motions: Teddy (from walking to sitting), rock (will be sit by Teddy in the end, but no related motions). Motion frames allocation: The main entity involve motion changes is Teddy. Teddy’s motion changes from walking to sitting. As sitting takes less time, we should allocate more frames to walking. so the plan is 4 frames for walking and 2 frames for sitting. (Following bullet point 11 [Smart Motion Allocation]) Bounding box changes: For Teddy, the first motion is walking, which involves position changes

a little bit. And the second motion is sitting, which should be a bounding-box-static motion

. So the Teddy bear can have bounding boxes changes at first and finally interacts with the For the rock, it doesn’t involve any motion, so it should be static from far from to close to , and finally interacting with Teddy. (Following bullet point 13 [Motions for Large Position Changes]) Interaction: This input narration involve interaction between the rock and Teddy and happens in late frames, so the rock and Teddy will have one merged-region where the entity is Teddy and the caption is about Teddy sitting on the rock. ALso the rock will also be listed from frame 1 to 3, while maintained from 4 to 6, as a static object. (Following bullet point 5 [ Allowing Overlaps for Interaction Contexts], 6 [Interaction with Objects], 7 [Static Interaction Objects])

In conclusion, the plan can be: Teddy is at the right side at the beginning of the video, and

there’s also a rock on the left right corner. From Frame 1 to Frame 4, the Teddy bear is hiking on the trail moving from right to left to approaching the rock. In Frame 5 and Frame 6, it reaches the rock and sits on the rock (which is on the left corner) to rest, in the left side of the video.

*Plan* Background: the forest

- Frame_1: [["Teddy", "hiking", "Teddy is hiking on a trail in the forest"], [0.6, 0.0, 1.0, 1.0]], [["rock", "none", "a rock in the forest"], [0.0, 0.8, 0.2, 1.0]]
- Frame_2: [["Teddy", "hiking", "Teddy is hiking on a trail in the forest"], [0.47, 0.0, 0.87,

- 1.0]], [["rock", "none", "a rock in the forest"], [0.0, 0.8, 0.2, 1.0]]
- Frame_3: [["Teddy", "hiking", "Teddy is hiking on a trail in the forest"], [0.33, 0.0, 0.73, 1.0]], [["rock", "none", "a rock in the forest"], [0.0, 0.8, 0.2, 1.0]]
- Frame_4: [["Teddy", "hiking", "Teddy is hiking on a trail in the forest"], [0.2, 0.0, 0.6, 1.0]], [["rock", "none", "a rock in the forest"], [0.0, 0.8, 0.2, 1.0]]
- Frame_5: [["Teddy", "sitting", "Teddy is sitting on the rock in the forest"], [0.0, 0.0, 0.4, 1.0]], [["rock", "none", "a rock in the forest"], [0.0, 0.8, 0.2, 1.0]]
- Frame_6: [["Teddy", "sitting", "Teddy is sitting on the rock in the forest"], [0.0, 0.0, 0.4, 1.0]], [["rock", "none", "a rock in the forest"], [0.0, 0.8, 0.2, 1.0]]

[Input] Motions: swimming, touching Narration: The Mermaid begins her day through the vibrant coral reef. She swims smoothly between colorful corals and touches the corals with her curiosity.

- Listing 4: The example output of fine-grained planning prompt 3. We highlight Background, [entity, motion, caption] and layout [left, top, right, bottom] using same colors as Section 3.1.

[Output]

*Reasoning* Listed motions are: swimming, touching. Related entities and motions: Mermaid (from swimming to touching), corals (being touched by the mermaid). Motion frames allocation: The main entity involving motion changes is the Mermaid. The first motion, swimming, can last relatively longer with more frame coverage as it involves position

changes. Touching corals happens afterward and should occupy fewer frames with minimal position changes. Bounding box changes: The swimming motion involves position changes, so the bounding box for the Mermaid can have slight shifts, swimming through the corals from left to right. The touching motion of the Mermaid should be static, focusing on the interaction at the coral, which is near the center of the frame. Interaction: The interaction between the Mermaid and the corals happens later in the narration, so a combined region featuring the Mermaid touching the corals will be involved in

the last two frames. Non-interacting corals will appear in all frames, having stable bounding boxes across the video to show the environment and the Mermaid’s interaction context.

The plan is as follows: The Mermaid swims through the vibrant coral reef, starting from near the left, moving towards the center, where the corals are located. On reaching the center, she begins touching the corals with fascination.

*Plan* Background: the vibrant coral reef full of colors and life

- Frame_1: [["Mermaid", "swimming", "The Mermaid is swimming through the vibrant coral reef"], [0.0, 0.0, 0.4, 1.0]], [["corals", "none", "Colorful corals in the reef"], [0.5, 0.3, 0.8, 0.6]]
- Frame_2: [["Mermaid", "swimming", "The Mermaid is swimming through the vibrant coral reef"], [0.15, 0.0, 0.55, 1.0]], [["corals", "none", "Colorful corals in the reef"], [0.5, 0.3, 0.8, 0.6]]
- Frame_3: [["Mermaid", "swimming", "The Mermaid is swimming through the vibrant coral reef"], [0.3, 0.0, 0.7, 1.0]], [["corals", "none", "Colorful corals in the reef"], [0.5, 0.3, 0.8, 0.6]]
- Frame_4: [["Mermaid", "swimming", "The Mermaid is swimming through the vibrant coral reef"], [0.45, 0.0, 0.85, 1.0]], [["corals", "none", "Colorful corals in the reef"], [0.5, 0.3, 0.8, 0.6]]
- Frame_5: [["Mermaid", "touching", "The Mermaid is touching the corals with her curiosity"], [0.4, 0.1, 0.7, 0.9]], [["corals", "none", "Colorful corals in the reef"], [0.5, 0.3, 0.8, 0.6]]
- Frame_6: [["Mermaid", "touching", "The Mermaid is touching the corals with her curiosity"], [0.4, 0.1, 0.7, 0.9]], [["corals", "none", "Colorful corals in the reef"], [0.5,

- 0.3, 0.8, 0.6]]

Listing 5: The example input output for getting merged prompt for overlapping regions. The model input is in black while the output is in blue

Given multiple captions describing the same scene, merge them into a single, coherent caption

that includes all key actions or entities, avoids redundancy, and maintains natural fluency. [Captions]

- 1. A man is throwing a frisbee in the park.
- 2. A dog is running toward the frisbee with its tongue out.

[Merged Caption] A man throws a frisbee in the park while a dog runs toward it with its tongue out.

