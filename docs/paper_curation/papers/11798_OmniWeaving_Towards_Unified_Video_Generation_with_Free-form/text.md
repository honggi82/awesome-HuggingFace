# arXiv:2603.24458v2[cs.CV]2Apr2026

[Figure 1]

2026-03-26

## OmniWeaving: Towards Unified Video Generation with Free-form Composition and Reasoning

Kaihang Pan1,2,∗,‡, Qi Tian2,∗, Jianwei Zhang2, Weijie Kong2, Jiangfeng Xiong2, Yanxin Long2, Shixue Zhang2, Haiyi Qiu1, Tan Wang3, Zheqi Lv1, Yue Wu2,§, Liefeng Bo2, Siliang Tang1,§, Zhao Zhong2,† 1Zhejiang University 2Tencent Hunyuan 3Nanyang Technological University ∗ Equal Contribution, § Corresponding Authors, † Project Leader

Project Page: https://omniweaving.github.io/ Github: https://github.com/Tencent-Hunyuan/OmniWeaving

[Figure 2]

Model: https://huggingface.co/tencent/HY-OmniWeaving

#### Abstract

While proprietary systems such as Seedance-2.0 have achieved remarkable success in omnicapable video generation, open-source alternatives significantly lag behind. Most academic models remain heavily fragmented, and the few existing efforts toward unified video generation still struggle to seamlessly integrate diverse tasks within a single framework. To bridge this gap, we propose OmniWeaving, an omni-level video generation model featuring powerful multimodal composition and reasoning-informed capabilities. By leveraging a massive-scale pretraining dataset that encompasses diverse compositional and reasoning-augmented scenarios, OmniWeaving learns to temporally bind interleaved text, multi-image, and video inputs while acting as an intelligent agent to infer complex user intentions for sophisticated video creation. Furthermore, we introduce IntelligentVBench, the first comprehensive benchmark designed to rigorously assess next-level intelligent unified video generation. Extensive experiments demonstrate that OmniWeaving achieves SoTA performance among open-source unified models. The codes and model have already been publicly available. Project Page: https://omniweaving.github.io.

#### 1 Introduction

The pursuit of artificial general intelligence has driven the evolution of visual generation models from task-specific experts to unified generalists (OpenAI et al., 2024; Xiao et al., 2025; Pan et al., 2025a; 2024; Xia et al., 2025). In the image domain, this paradigm shift was significantly catalyzed by GPT-4o (OpenAI et al., 2024) and NanoBanana (Google, 2025a), proprietary models that seamlessly integrated image understanding and generation within a single framework. Their unprecedented success in executing omni-level generation has sparked a vigorous response from the open-source community. Consequently, academic models like BAGEL (Deng et al., 2025a) and OmniGen2 (Wu et al., 2025c) have rapidly emerged, natively coupling visual comprehension with generative modules to enable unified image synthesis with free-form multimodal inputs.

As the field naturally progresses toward the temporal domain, video generation also reaches a pivotal juncture requiring a more unified framework. Recently, proprietary systems such as Seedance-2.0 (Seed, 2026) have redefined the landscape, establishing that next-generation models must be genuinely “omni-capable” by synergizing two foundational pillars: (1) Multimodal composition, which enables the seamless spatio-temporal binding of free-form, interleaved text, image, and video inputs; and (2) Abstract reasoning, which empowers models to act as active agents capable of inferring complex user intentions and mastering the underlying semantic logic of dynamic scenes.

However, unlike the flourishing open-source ecosystem in image generation, academic progress in unified video generation significantly lags behind proprietary systems, revealing a substantial capability gap. First, the current landscape of video generation remains dominated by fragmented approaches (Wu et al., 2025a; Wan et al., 2025; Kong et al., 2024) narrowly tailored for text-to-video, image-to-video synthesis, or video-to-video synthesis (He et al., 2025), relying on task-specific modules that impede scaling and integration. Furthermore, while recent open-source models such as VACE (Jiang et al., 2025), UniVideo (Wei et al., 2025), VINO (Chen et al., 2026) attempt to unify video generation tasks, they either focus primarily on basic task combinations or fail to leverage deep visual understanding to drive unified generation. Consequently, they still struggle to effectively address multimodal composition and reasoning-informed video synthesis.

We argue that bridging this substantial capability gap of unified video models relies on three key drivers. First, the model architecture must integrate both visual comprehension and generation into a single framework to explicitly activate abstract reasoning, evolving models from passive renders into “thinking-guided” generators. Second, a transition toward free-form, multi-task pretraining is essential to move beyond rigid prompt-video pairs and capture

‡ Work done when interning at Tencent Hunyuan.

Foundational Video Generation Tasks

Text to Video First-Frame to Video

Close-up of a serious man in aviator sunglasses with a deeply furrowed brow. He is speaking.

The couple is celebrating a joyful moment together.

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

|[Figure 9]|
|---|

Key-Frames to Video

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

|[Figure 14]|
|---|

|[Figure 15]|
|---|

Man rises, dribbles, turns, and shoots backward as sunset lights him.

###### Video Editing

[Figure 16]

[Figure 17]

[Figure 18]

Place the bun-haired person in a festive green and red striped sweater at the table, right of the man in red, performing expressive movements.

|[Figure 19]|[Figure 20]|[Figure 21]|
|---|---|---|

Multimodal Composition Tasks

Compositional Multi-Image-to-Video

[Figure 22]

[Figure 23]

[Figure 24]

|[Figure 25]|
|---|

The bear in the first image holds the jar of raspberry jam in the third image steady while using the spoon in the second image to stir. It scoops up jam with the spoon, lifts it, and opens its mouth to taste.

|[Figure 26]|
|---|

|[Figure 27]|
|---|

[Figure 28]

[Figure 29]

[Figure 30]

The drone in the third image rotates steadily clockwise. The background is in the fourth image.

|[Figure 31]|
|---|

|[Figure 32]|
|---|

The man in the first image and the woman in the second image talk with mouths opening and closing.

|[Figure 33]|
|---|

|[Figure 34]|
|---|

Text-Image-Video-to-Video

[Figure 35]

[Figure 36]

[Figure 37]

Replace color of the car in the video with the color of car in the image.

|[Figure 38]|[Figure 39]|[Figure 40]|
|---|---|---|

|[Figure 41]|
|---|

[Figure 42]

[Figure 43]

[Figure 44]

Replace the background with the scene in the image.

| |[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]| |
|---|---|---|

|[Figure 48]|
|---|

Reasoning-Augmented Video Generation

###### Image-to-Video with Implicit Intent

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

|[Figure 53]|
|---|

Two girls were reunited with their long-lost dog.

Reasoning: Two girls were happily greeted by their long-lost dog. The girl in the school uniform immediately turns around and kneels down, opening her arms to welcome it. The other girl in the white coat quickly joins. Their expressions transform from initial surprise to pure joy as they pet and finally hug the long-lost poodle tightly, faces pressed together and smiling happily at the camera.

###### Event-Deductive Multi-Image-to-Video

Reasoning: The camera transitions from a low-angle view of a traffic light displaying a green arrow, signaling vehicles to proceed straight ahead, to a high-angle view of a grand, historic building, emphasizing its architectural details and the bustling street scene around it.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

|[Figure 58]|
|---|

|[Figure 59]|
|---|

From traffic light to historic building, capturing city life in motion.

Figure 1: Showcase of OmniWeaving across diverse video generation scenarios, such as foundational tasks, multimodal composition tasks, and reasoning-augmented scenarios.

the intricate semantic relationships across diverse modalities. Finally, since existing benchmarks are largely limited to simplistic tasks with monolithic input formats, the community requires a more complex and comprehensive evaluation suite to foster the development of truly “omni-capable” video systems.

To address these critical challenges, we propose OmniWeaving, an omni-level video generation framework capable of both multimodal composition and abstract reasoning. Based on a unified architecture integrating visual comprehension and generation, we introduce a massive-scale training dataset that spans a broad spectrum of scenarios and diverse input formats, including both multimodal composition and reasoning-augmented tasks. Through a meticulous three-stage training strategy, as shown in Figure 1, OmniWeaving could adeptly handle diverse video generation scenarios, effectively “weaving” free-form text, image, and video inputs into a coherent spatio-temporal narrative.

To rigorously evaluate unified video generation from heterogeneous, free-form inputs, we introduce IntelligentVBench, a novel benchmark employing a “VLM-as-a-judge” paradigm to assess abstract reasoning and compositional capabilities across four distinct tasks. Extensive experiments demonstrate that our proposed OmniWeaving framework achieves state-of-the-art performance among existing open-source alternatives. In summary, our main contributions are threefold:

- • We propose OmniWeaving, a unified framework that seamlessly integrates visual understanding to achieve omni-level video generation from free-form inputs with strong composition and reasoning capabilities.
- • We introduce a massive-scale dataset, spanning a broad spectrum of generative scenarios including both composition- and reasoning-related training tasks.
- • We present IntelligentVBench, the first benchmark dedicated to measuring multimodal composition and abstract reasoning in unified video generation.

#### 2 Related Work

Unified Video Generation. While proprietary systems such as Seedance-2 (Seed, 2026), Kling-O1 (Team et al., 2025), SORA, and Veo3 (Google, 2025b) have largely realized next-level, “omni-capable” intelligent video generation, their underlying techniques remain undisclosed, leaving a significant capability gap in the research community. Currently, the open-source video generation landscape is dominated by fragmented approaches narrowly tailored for specific tasks, such as text-to-video, image-to-video (Wan et al., 2025; Wu et al., 2025a), or video-tovideo synthesis (Bai et al., 2025a; He et al., 2025), that typically rely on isolated models and disjointed pipelines. Furthermore, genuine unified video generation fundamentally relies on robust multimodal composition and abstract reasoning, and recent open-source efforts attempting such unification still exhibit notable shortcomings. For instance, OmniVideo (Tan et al., 2025) and OmniVideo2 (Yang et al., 2026) merely incorporate two related video generation capabilities—text-to-video and video editing—into a single framework. Although models like VACE (Jiang et al., 2025), UniVideo (Wei et al., 2025), and VINO (Chen et al., 2026) expand the variety of supported tasks, they fail to fully leverage deep visual understanding to drive unified generation and lack the cohesive integration of multi-task capabilities within a single architecture. To address these challenges, OmniWeaving aims to explore better strategies across multiple dimensions, including architecture, data, and training paradigms, to provide a robust reference for next-generation unified video synthesis.

Video Generation Benchmarks. As video generation models rapidly advance, traditional benchmarks struggle to capture their true capabilities due to two primary limitations. (1) Lack of complexity: Most benchmarks are highly task-specific with rigid input formats. For instance, VBench (Huang et al., 2024) and VBench++ (Huang et al., 2025) strictly evaluate foundational text- or image-to-video generation, restricted to single-shot scenarios. TGVE+ (Singer

- et al., 2024) and OpenVE-Bench (He et al., 2025) focus on Video-to-Video editing tasks. While VACE-Bench (Jiang
- et al., 2025) attempts to incorporate various downstream tasks, the input structures still remain inflexible. (2) Lack of comprehensiveness: Current benchmarks primarily assess foundational video rendering in simplistic scenes, largely neglecting higher-order abilities such as composition and reasoning. Although benchmarks like OpenS2V (Yuan et al., 2025) and VACE-Bench (Jiang et al., 2025) include test cases for multimodal composition, they are insufficient in scale and completely omit reasoning evaluations. Furthermore, most benchmarks rely on small, specialized tool models for assessment, unable to measure whether the generated videos truly align with user intentions in complex scenarios. In contrast, designed with both complexity and comprehensiveness in mind, our IntelligentVBench encompasses diverse tasks, supports free-form inputs across multiple modalities, explicitly evaluates reasoning and compositional skills, and leverages a VLM-as-a-Judge (Zheng et al., 2023) paradigm to ensure a robust evaluation.

#### 3 Training Data

While conventional text-video paired data provides useful supervision, it falls short in supporting complex in-context reasoning and composition that involves interleaved text, images, and video inputs. Models trained exclusively on

such data often struggle to capture nuanced semantic relationships across modalities. To address these limitations, we incorporate large-scale vision-text interleaved data into our training corpus to enable richer multimodal interactions, utilizing videos sourced from both real-world and synthetic domains. In this section, we detail the training data sources, training tasks, and the data construction process in Section 3.1, 3.2, and 3.3, respectively.

###### 3.1 Training Data Source

To construct a robust training corpus, we curate data from two complementary sources based on video provenance: real-world and synthetic domains. Real-world data encompasses a broad spectrum of visual content essential for capturing rich appearances, naturalistic motions, and complex scene dynamics; crucially, this anchors the generated videos to natural distributions and mitigates noticeable generative artifacts. However, because naturally occurring paired videos for highly conditioned tasks, such as video editing, are frequently sparse or inherently noisy, we incorporate synthetic data by leveraging off-the-shelf generation models (Wu et al., 2025a; Wan et al., 2025; Google, 2025b) to rapidly synthesize target videos aligned with specific input conditions. While relying exclusively on synthetic data tends to introduce pronounced artificial biases, combining these two domains creates a synergistic effect that perfectly balances natural realism with task-specific conditioning density.

###### 3.2 Training Tasks

To ensure our training tasks facilitate richer multimodal interactions, we establish two core design principles: comprehensive coverage of diverse multimodal scenarios and the systematic optimization of hierarchical model capabilities. Accordingly, we structure our training framework around three primary competencies, each encompassing a diverse array of task formats.

Foundational Video Generation Tasks: This category integrates some foundational generation and editing tasks across three primary domains: (a) Text-to-image and text-to-video synthesis with text-video or text-image paired data; (b) Instruction-guided video-to-video editing for both local and global modifications, such as background replacement, style transfer, object manipulation (addition, removal, or replacement), and text rendering; and (c) Key-frame(s)-to-video generation, which synthesizes continuous temporal sequences either from a single initial frame or via interpolation across multiple key-frames.

Multimodal Composition Tasks. Multimodal composition requires extracting and integrating distinct subjects or scenes from diverse inputs to synthesize a coherent video without unnatural blending artifacts. We formulate two primary tasks: (a) Interleaved Text-and-Multi-Image-to-Video generation, where the inputs contain multiple reference images (capturing key visual elements such as subjects or scenes) interleaved with text, requiring the model to accurately compose these elements into a cohesive video sequence; and (b) Text-Image-Video-to-Video generation, where the inputs consist of three modalities (image, text, and video), requiring the model to seamlessly integrate target visual elements extracted from reference images into the temporal dynamics of a source video.

Reasoning-Augmented Tasks: When user inputs are ambiguous, reasoning is essential to decipher the intended video content. Accordingly, we construct a reasoning-augmented dataset encompassing three main tasks: (a) Text-to-Video generation, where the model is trained to deduce comprehensive descriptions from brief, ambiguous input text queries prior to synthesis; (b) Intent-Driven Image-to-Video generation, where the model learns to formulate a reasoning trace detailing the temporal progression when visual and textual inputs lack explicit linkage (e.g., the text outlines abstract intents); and (c) Event-Deductive Multi-Image-to-Video generation, given several highly disparate reference images as the key-frames, the model is optimized to bridge disparate reference key-frames by first uncovering implicit temporal dynamics via providing transition descriptions for these key-frames, before generating temporally coherent videos.

###### 3.3 Training Data Construction

To construct our training tasks, we employ a dual-pipeline data construction strategy: output-first and input-first. In the output-first pipeline, we curate a diverse array of real-world videos from sources such as YouTube, cinematic clips, live-stream excerpts, and social media platforms to serve as ground-truth target videos. Subsequently, an ensemble of auxiliary models is utilized to extract corresponding images or generate descriptive texts that act as task-specific inputs. Conversely, the input-first pipeline begins by formulating the input conditions, leveraging video generation models, augmented by various tool models, to synthesize the corresponding ground-truth videos. To facilitate both pipelines, we integrate a robust suite of models, such as Qwen3 (Yang et al., 2025), Qwen3-VL (Bai et al., 2025b), Gemini2.5-Pro (Comanici et al., 2025), SAM3 (Carion et al., 2025), FLUX2 (Labs, 2025), and various video generation models (Wu et al., 2025a; Wan et al., 2025; Google, 2025b), with Qwen3-VL additionally serving as an evaluator for rigorous data quality filtering. Next, we will provide a detailed exposition of the training data construction pipeline for each task.

Foundational Video Generation Tasks: Our primary training corpus for Text-to-Image (T2I), Text-to-Video (T2V), and Key-Frame(s)-to-Video generation (I2V) tasks is predominantly derived from extensive in-house datasets

Pipeline for Interleaved Text-and-Multi-Image-to-Video Generation Task

[Figure 60]

Extracted Frame

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

SAM3

[Figure 66]

[Figure 67]

[Figure 68]

[IMG1] [IMG2]

###### Qwen3-VL

The bears from [IMG1] and [IMG2] playfully intercepting the man on a

###### Qwen3-VL

Extracted Objects

[Figure 69]

[Figure 70]

- Entity1: The

bear on the left

- Entity2: The

bear on the right

- Entity3: The man on the bicycle

[Figure 71]

[Figure 72]

bicycle in [IMG3] over the background in [IMG4]. The bear from [IMG1] raises a paw to halt his progress, the other from [IMG2] stands closely behind the bike.

FLUX2

[Figure 73]

- 1. Object Verification
- 2. Prompt Rewriting
- 3. Semantic Verification

Entity Extraction & Filtering

Two bears are playfully intercepting a man on a bicycle in the middle of a forest trail. One bear raises a paw to halt his progress, the other stands closely behind the bike.

Object & Background Isolation

[Figure 74]

[Figure 75]

Extracted Background

[IMG3]

###### [IMG4]

Input Text(Image)-Video

Interleaved Text-andMulti-Image Input

Pipeline for Text-Image-Video-to-Video Generation Task

[Figure 76]

[Figure 77]

Target Object: The person on the left of the dog

Qwen3-VL

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Modification Analysis

###### FLUX2

[Video] [IMG]

###### Qwen3-VL

[Figure 85]

[Figure 86]

Add the person from [IMG] into the [Video], walking on the grassy path and on the left of the dog, facing away from the camera toward the treeline.

[Figure 87]

- 1. Prompt Rewriting
- 2. Semantic Verification

Isolate Changed Elements

Add a person walking on the grassy path, facing away from the camera toward the treeline.

Extracted Reference Image

Text-Image-Video Input

Input Video-to-Video Case

Extract 1st Frame

Figure 2: Training data construction pipeline for Multimodal Composition Tasks.

utilizing an output-first pipeline. Specifically, the videos are mainly collected from web-sourced clips, and we leverage Qwen3-VL-235B and Gemini2.5-Pro to generate high-quality textual annotations as the user input. To ensure the annotations align with specific task requirements, we design tailored prompting strategies: T2I and T2V prompts focus on visual semantic descriptions, whereas I2V prompts emphasize the dynamic transitions originating from the initial frame or the characterization of spatio-temporal offsets across multiple key-frames. Beyond this output-first paradigm, we also integrate an input-first strategy to produce a set of synthetic video data. Specifically, we first construct a carefully curated set of textual prompts and key-frames, and subsequently query Veo3 (Google, 2025b) for high-quality video synthesis.

Furthermore, the training task for Instruction-guided video-to-video editing (V2V) encompasses both global and local modifications. Global edits primarily focus on background transformations and style changes, while local edits include fine-grained operations such as object addition, removal, replacement, and text manipulation within the video. To construct a robust training corpus, we aggregate data from existing datasets, specifically OpenVE-3M (He et al., 2025) and Ditto (Bai et al., 2025a), and synthesize additional samples following their established pipelines. Finally, to guarantee high dataset fidelity, the entirety of this collected corpus undergoes rigorous quality filtration via Qwen3-VL-235B, ensuring the elimination of unsuccessful or low-quality edits.

Notably, we observe that existing pipelines for local object addition V2V frequently yield physical inconsistencies, where newly introduced objects appear detached from the scene’s underlying geometry and lighting. To rectify this, we invert the local object removal process, treating the post-removal video as the source input and the original, unedited video as the ground-truth target. By adapting the corresponding instructions from “removal” to “addition”, we successfully generate high-quality V2V training samples for local object addition, characterized by physical realism and seamless environmental integration.

Multimodal Composition Tasks. The Multimodal Composition Tasks encompass two primary sub-tasks: Interleaved Text-and-Multi-Image-to-Video generation and Text-Image-Video-to-Video generation. The respective data construction pipelines are detailed below, also shown in Figure 2.

Specifically, to facilitate Interleaved Text-and-Multi-Image-to-Video generation, we construct our dataset by systematically processing our image-to-video (I2V) task data. Given a detailed video caption and its corresponding key-frame, we employ Qwen3-VL-235B to identify moving objects, representing them as named entities enriched with precise descriptive modifiers to ensure accurate localization. Qwen3-VL also acts as a filter to discard entities that feature ambiguous descriptions, isolated body parts, or objects absent from the key-frame. Subsequently, for each validated entity, we utilize SAM3 to estimate its spatial location and temporal presence across the video sequence, followed by the application of FLUX2 to extract the corresponding object. To ensure appearance diversity and avoid pose duplication with the first frame, we leverage FLUX2 to extract these entity images from subsequent frames. Moreover, recognizing the potential extraction inaccuracies of FLUX2, we formulate four distinct prompt variations for each target object, leveraging Qwen3-VL as a verifier to confirm the identity alignment between the subject in the extracted image and the specified entity within the key-frame. In addition to entity extraction, we also isolate the background from the first frame with FLUX2. We then instruct Qwen3-VL to synthesize a comprehensive, reorganized prompt that integrates the extracted objects and the background, thereby formulating the interleaved text-and-multi-image input. Ultimately, following a rigorous final verification by Qwen3-VL to ensure semantic consistency between the constructed interleaved input and the ground-truth video, we construct a large-scale, high-quality dataset tailored for Interleaved Text-and-Multi-Image-to-Video generation.

In contrast, we formulate the Text-Image-Video-to-Video generation task by repurposing existing video-to-video editing datasets. Given a source and target video pair, we first employ Qwen3-VL-235B, in conjunction with the original editing instruction, to analyze the specific modifications made in the target video, generating precise descriptive terms for the altered elements, such as a specific localized object or the overall video background. We then extract the first frame of the target video and apply FLUX2 to isolate the specific visual elements that have changed relative to the source video to serve as our reference images. Finally, we prompt Qwen3-VL a second time to rewrite the editing instructions and verify the semantic alignment between the input and output, thereby ensuring that the target elements detailed in the prompt are explicitly grounded in the extracted reference images.

Reasoning-Augmented Tasks. The Reasoning-Augmented Tasks comprise three primary sub-tasks: Text-toVideo generation, Intent-Driven Image-to-Video generation, and Event-Deductive Multi-Image-to-Video generation. In addition to standard user inputs and ground-truth videos, each task incorporates a reasoning trace that explicitly bridges the input to the corresponding output. Next, we detail the pipelines for constructing the training data across these three tasks.

For Text-to-Video generation, we initialize our pipeline with a collection of brief, ambiguous text queries, where a subset of these queries has already been accompanied by ground-truth videos. To process the queries without corresponding videos, we leverage Qwen3-30B to generate detailed prompts acting as query-guidance. These detailed prompts are subsequently fed into HunyuanVideo-1.5 (Wu et al., 2025a) to generate the corresponding target videos. Furthermore, for queries that already possess matching videos, we employ Qwen3-VL-235B to deeply analyze the video content and expand the initial brief query into a comprehensive, detailed prompt. Ultimately, this dual-pathway approach yields robust training triplets, comprising the initial query, the detailed prompt guidance, and the associated video. Therefore, it enables video generation models to ground video generation in sophisticated language-based reasoning.

To advance Intent-Driven Image-to-Video generation, we curate a dataset of cinematic text-video pairs wherein the text describes the action intent or behavioral premise driving the corresponding video scene. For each sample, we leverage Qwen3-VL-235B to synthesize a detailed description of the temporal motion unfolding from the initial frame. To ensure rigorous data quality, we further employ Qwen3-VL to strictly filter out instances where the generated motion lacks causal alignment with the driving intent. On this basis, the action intent or behavioral premise serves as the user query, and the generated motion description functions as an intermediate reasoning trace. It effectively enables the model to infer the reasoning trace prior to generating temporally coherent videos conditioned on both the first frame and the user query.

To facilitate Event-Deductive Multi-Image-to-Video generation, we process our collected video dataset by extracting key-frames and rigorously filtering them to ensure significant visual distinctions among key-frames. Then we leverage Qwen3-VL-235B to generate two distinct types of textual annotations: a concise, single-sentence prompt that outlines the underlying intent or premise of the video, and a detailed prompt that intricately describes the dynamic temporal transitions across the given key-frames. In formulating the training data, each instance incorporates the ordered key-frames and the corresponding ground-truth video, alongside a user query that either utilizes the concise prompt or relies solely on a generic instruction devoid of any explicit event cues, i.e., “generate a complete video based on the provided key-frames”. Crucially, the detailed prompt acts as an explicit reasoning trace, thereby conditioning the model to logically deduce the underlying event trajectory implied by the key-frames before synthesizing the final video.

#### 4 Model: OmniWeaving

In this section, we introduce OmniWeaving, a unified framework for video generation and editing conditioned on free-form text, image, and video inputs. Crucially, the model possesses reasoning and compositional capabilities that are essential for intelligent unified video generation. We detail the model architecture, training strategy, and implementation details in Section 4.1, 4.2, and 4.3, respectively. More Details are given in Appendix A.

###### 4.1 Model Architecture

Recognizing that robust visual comprehension is fundamental to unified free-form video generation, OmniWeaving is designed as an integrated architecture for multimodal understanding and generation. As shown in Figure 3, it comprises three core components: a Multimodal Large Language Model (MLLM), a Multimodal Diffusion Transformer (MMDiT), and a Variational Autoencoder (VAE).

First, the MLLM acts as the core semantic parser. It projects free-form multimodal inputs into a high-level semantic space and routes the last-layer hidden states to the MMDiT via an MLP connector. Second, the VAE serves as a visual tokenizer, compressing input visions into low-level latents to provide fine-grained reconstruction signals. Finally, the MMDiT operates as the backbone diffusion model: its conditioning branch encodes the MLLM semantics, while the generative branch integrates VAE latents with noise, ultimately generating semantically aligned,

### MMDiT

| | |
|---|---|
| | |
| | |
| | |

BlockN

<think> The image only has a man in a hallway. As he walks through the hallway, an office show in the left. He turns the handle and enters. The shot transitions smoothly into the office, where a manager in a suit sits behind a desk. The man approaches and handle over the briefcase to the manager. </think>

- Block1

- Block2

...

Self Attention

- Block3 MLP

[Figure 88]

MLLM

##### VAE

Image Video Noisy Latents

Text Tokenizer

|[Figure 89]|
|---|

[Figure 90]

[Figure 91]

[Figure 92]

ViT

Text Tokenizer

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

System Prompt

Image Video

Text

[Figure 98]

[Figure 99]

[Figure 100]

Generate a video of a man handing over a briefcase to his manager in the office.

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Figure 3: OmniWeaving consists of an MLLM for multimodal understanding and an MMDiT for generation. On this basis, we activate the thinking mode of the MLLM and further introduce the DeepStacking mechanism.

high-fidelity videos. On this basis, we further introduce two extra improvements tailored for advanced reasoning and composition.

- (1) Activating Thinking Mode of the MLLM: Direct MLLM encoding of interleaved visual-text inputs often yields semantic ambiguity due to weak intra-correlations and unclear video creation intents. To address this issue, we elevate the MLLM from a passive feature extractor to an active reasoner. By activating the thinking mode of the MLLM to generate intermediate reasoning steps, it autonomously deduces a semantically precise, enhanced prompt. The hidden states of this enhanced prompt are then forwarded alongside the original MLLM features to condition the MMDiT, effectively bridging the cognitive gap between abstract user intent and pixel-level generation.
- (2) Hidden States DeepStacking: Compositional video generation involving multiple subjects or intricate scenes often relies on both low- and high-level semantic representations. Therefore, as shown in Figure. 3, we draw inspiration from the DeepStacking mechanism in Qwen3-VL (Bai et al., 2025b), extracting hidden states from a broader range of intermediate MLLM layers to capture a rich semantic spectrum spanning from fine-grained details to high-level abstractions. An MLP connector projects these multi-level features into the MMDiT embedding space. These projected features are then directly added to the corresponding hidden states within the first three layers of the MMDiT conditioning branch, effectively injecting multi-granular semantic guidance into the generative process.

###### 4.2 Training Strategy

Leveraging the proposed architecture and training data, we formulate a progressive, three-stage training paradigm, elevating OmniWeaving from a passive pixel renderer to an active and intelligent generalist.

- Stage 1: Modality Alignment Training. As our generative backbone, i.e., HunyuanVideo-1.5 (Wu et al., 2025a), is highly proficient as a text-to-video expert but lacks prior exposure to multimodal hidden states produced by our MLLM, the primary objective of this initial stage is modality alignment between MLLM and MMDiT. We restrict the training to the most fundamental tasks: Text-to-Video (T2V) and Image-to-Video (I2V). We keep the MLLM parameters strictly frozen and exclusively finetune the MMDiT and the MLP connector with high-quality T2I and T2V samples. After this stage, OmniWeaving achieves performance comparable to the MMDiT backbone that uses its own text encoder.
- Stage 2: Multi-Task Free-Form Pretraining. Having aligned the foundational modalities, we scale the training to encompass complex, heterogeneous inputs. Specifically, this stage incorporates all tasks detailed in Section 3.2, excluding the Reasoning-Augmented Tasks. However, we observe that introducing training tasks involving video input, such as video editing, at the very onset of the training impedes the model’s learning efficiency on tasks that require processing interleaved image-text inputs. Thus, we further partition this training phase into two distinct sub-stages: we initially exclude tasks involving video inputs, later integrating them for joint mixed training in the second sub-stage. Throughout this training stage, the MLLM parameters still remain frozen to focus exclusively on fine-tuning the MMDiT. Ultimately, we empower OmniWeaving to unify diverse video generation and editing tasks via multimodal instructions, establishing it as a versatile generalist framework with robust free-form compositional capabilities.
- Stage 3: Reasoning-Augmented Fine-Tuning. In the final stage, we further elevate OmniWeaving’s cognitive capabilities by introducing the reasoning-augmented tasks, jointly co-training alongside a curated high-quality subset from Stage 2. Crucially, we unfreeze the MLLM in this phase for end-to-end optimization alongside the

Table 1: Training recipe of OmniWeaving.

| |Stage1 Stage2<br><br>Stage3 MLLM MMDiT<br><br>|
|---|---|
|Hyperparameters Learning rate LR scheduler Min Learning rate Weight decay Gradient norm clip Optimizer GPUs Batch Size Loss weight (CE : MSE) Warm-up steps Training steps Resolution Text dropout ratio Diffusion timestep shift|2 × 10−5 2.0 × 10−5 3.0 × 10−5 1.0 × 10−5<br><br>Constant Constant Cosine Constant<br><br>2 × 10−5 2.0 × 10−5 2.0 × 10−6 1.0 × 10−5<br><br>0.01 0.01 0.05 0.01<br><br>1.0 1.0 1.0 1.0<br><br><br>Muon Muon AdamW Muon<br><br>128× H20s 400× H20s 400× H20s 128 400 400<br><br>- - 0.25 : 1<br><br>250 1200 250<br><br>5K 50K 3k 480p 480p 480p 0.1 0.1 0.1<br><br>1.0 for image and 4.0 for video<br><br>|
|Data sampling ratio Text-to-Image Text-to-Video First-Frame-to-Video Key-Frames-to-Video Video-to-Video Interleaved Text-and-Multi-Image-to-Video Text-Image-Video-to-Video Reasoning-Augmented Tasks<br><br>|0.10 0.05 → 0.03 0.00 0.40 0.15 → 0.07 0.05 0.50 0.15 → 0.06 0.05 0.00 0.25 → 0.14 0.05 0.00 0.00 → 0.25 0.15 0.00 0.40 → 0.20 0.15 0.00 0.00 → 0.25 0.15 0.00 0.00 0.40|

MMDiT. Beyond the standard diffusion loss, the reasoning traces within the augmented data also introduce a next-token-prediction loss specifically designed to enhance the MLLM’s reasoning proficiency. This establishes a “comprehend-then-generate” paradigm. Specifically, when confronted with ambiguous inputs, the MLLM learns to activate a “thinking mode” to extract explicit requirements via visual comprehension, thereby better guiding the MMDiT to synthesize highly aligned videos. Consequently, OmniWeaving evolves into an integrated, proactive reasoner and generator, functioning as an intelligent agent adept at managing diverse video generation tasks.

###### 4.3 Implementation Details

We implement OmniWeaving by integrating Qwen2.5-VL (Bai et al., 2025c) as the foundational MLLM and HunyuanVideo-1.5 (Wu et al., 2025a) as the generative MMDiT. To instantiate our DeepStacking mechanism, we extract the hidden states from the 8th, 16th, and 24th layers of the MLLM. These states are projected through a trainable MLP and subsequently added directly to the corresponding hidden states of the first three conditioning layers within the MMDiT. To mitigate the computational overhead of attention in the MMDiT, we adopt the sparse attention implementation from SSTA (Selective and Sliding Tile Attention) (Wu et al., 2025a). Furthermore, we employ the Muon optimizer (Liu et al., 2025a) to accelerate training convergence. The training process is structured into three distinct stages, spanning 5k, 50k, and 3k steps, with learning rates set to 2e-5, 2e-5, and 1e-5, respectively. During the second (multi-task free-form pretraining) stage, we initially exclude tasks involving video inputs for the first 20k steps. In the subsequent 30k steps, we incorporate these tasks involving video inputs, encompassing all tasks detailed in Section 3.2, with the exception of the Reasoning-Augmented Tasks. These Reasoning-Augmented Tasks are exclusively introduced during the third training stage (Reasoning-Augmented Fine-Tuning). To facilitate large-scale optimization, we train our model with 400 NVIDIA H20 GPUs, maintaining a global batch size of 400. Our primary focus is on generating videos at a 480p resolution; accordingly, all training videos contain between 33 and 161 frames, with aspect ratios strictly bounded between 0.25 and 4.0. The training hyperparameters and data sampling ratio for each stage are summarized in Table 1.

#### 5 Benchmark: IntelligentVBench

As video generation models rapidly advance, traditional benchmarks struggle to evaluate their true capabilities. As discussed in Section 2, they often lack complexity by confining evaluations to isolated, single-shot tasks, and lack comprehensiveness by prioritizing basic rendering in simplistic scenes over the crucial roles of multi-element composition and abstract reasoning. To bridge this gap, we introduce IntelligentVBench, a novel evaluation suite designed to benchmark next-level intelligent video generation by integrating abstract reasoning and free-form composition across a diverse array of complex tasks. A comparison between IntelligentVBench and existing video generation benchmarks is shown in Table 2. In Figure 4, we further present examples of each task type in IntelligentVBench.

Table 2: Compare IntelligentVBench with existing video generation benchmarks.

|Benchmark<br><br>|#Size<br><br>Multi- Interleaved Task Input|Input modality Image Multi-Images Video<br><br>|Ability to Measure Foundation Composition Reasoning<br><br>|VLM-as-a-Judge|
|---|---|---|---|---|
|VBench (Huang et al., 2024) VBench++ (Huang et al., 2025) OpenVE-Bench (He et al., 2025) TGVE+ (Singer et al., 2024) OpenS2V-Eval (Yuan et al., 2025) VACE-Bench (Jiang et al., 2025)|946 ✗ ✗ 2384 ✓ ✗ 431 ✗ ✗<br><br>1417 ✗ ✗ 180 ✗ ✓ 480 ✓ ✓<br><br>|✗ ✗ ✗<br><br>✓ ✗ ✗<br><br>✗ ✗ ✓<br><br>✗ ✗ ✓<br><br><br>✓ ✓ ✗ ✓ ✓ ✓<br><br>|✓ ✗ ✗ ✓ ✗ ✗ ✓ ✓ ✗ ✓ ✓ ✗ ✓ ✓ ✗ ✓ ✓ ✗<br><br>|✗ ✗ ✓ ✗ ✗ ✗|
|IntelligentVBench<br><br>|1030 ✓ ✓|✓ ✓ ✓|✓ ✓ ✓<br><br>|✓|

Implicit Image-to-Video (Implicit I2V)

Abstract Intent The girls are celebrating the end of school, expressing freedom.

Lack of Explicit Correlation The armed men initiated a sudden attack or raid, causing chaos.

###### Causal Deduction

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

The soldier is struck by a bullet.

Interpolative Dual-Image-to-Video (Interpolative DI2V)

Drastic Spatial Disparities Camera follows woman crossing street, then pans right onto sidewalk past parked cars.

###### Varying Subjects across Frames Local State Variations

Scissors slice litchi; hand peels back skin to reveal juicy white flesh.

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]|
|---|

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

Camera shifts from woman at vanity to man applying lotion.

Compositional Multi-Image-to-Video (Compositional MI2V)

- 1 Subject The man in the image stands still as his mouth opens and closes while talking. Another woman standing behind him remains motionless.

|[Figure 119]|
|---|

- 2 Subject

1 Subject + 1 Background

The camera follows the man in the first image over the background shown in the second image. He runs forward with his eyes fixed ahead and arms swinging.

|[Figure 120]|
|---|

|[Figure 121]|
|---|

The woman in the first image kneels by an open suitcase, sorting clothes. The woman in the second image sits nearby, watching.

|[Figure 122]|
|---|

|[Figure 123]|
|---|

2 Subject + 1 Background

The fox in the first image opens its mouth to talk, raises its right hand over the background in the third image. The bird in the second image flutters gently in place.

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

- 3 Subject The man in the first image starts talking. The woman in the second image turns her head away, revealing her profile. The woman in the third image stands still with her arms crossed.

###### 3 Subject + 1 Background

The woman with green curly hair in the first image holds the perfume bottle in the second image in one hand and the small jar with a silver lid in the third image in the other, over the background shown in the fourth image. She blinks, purses her lips, and looks around.

|[Figure 127]|
|---|

|[Figure 128]|
|---|

|[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

Text-Image-Video-to-Video (TIV2V)

Background Change Local Addition Local Replacement

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

Replace the background with the scene shown in the image.

Add the white car in the image driving on the dirt road.

Replace the dog in the orange sweater with the dog in the blue winter jacket in the image.

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Figure 4: Examples of each task type in IntelligentVBench.

###### 5.1 Design Principles

To comprehensively evaluate the complex capabilities required of unified video generation models, the design of IntelligentVBench is guided by two core principles, closely mirroring the demands of real-world human queries.

Heterogeneous and Free-Form Inputs. To better reflect the complexities of real-world human queries, our benchmark aims to employ test cases featuring free-form, interleaved vision-text inputs. Specifically, inputs should consist of diverse combinations of multiple images, text, and, when necessary, reference video contexts. Crucially, images within the prompt serve multifaceted roles: Some act as spatial anchors, providing specific objects intended for scene composition, while others serve as temporal anchors to delineate event progression. This design compels models to dynamically interpret unstructured, multi-modal conditions rather than over-fitting to rigid input templates.

Evaluating Composition and Reasoning: We consider composition and reasoning to be the two most critical capabilities for advanced intelligent video generation, thus structuring our evaluation around them. Specifically, for (a) Composition, given complex inputs comprising diverse subjects and scenes, we require the model to not only achieve accurate static spatial composition, but also to temporally synthesize the dynamic interactions among these entities and their interplay with contextual background elements throughout the generated frames. Furthermore, regarding (b) Reasoning, many cases within our benchmark cannot be resolved through direct explicit scene rendering; instead, the model may first need to deduce temporal correlations across significantly disparate reference images, interpret textual descriptions of object behaviors that remain absent from the visual inputs, or process prompts that merely articulate the abstract intent underlying an event.

###### 5.2 Task Taxonomy

Guided by the design principles, we construct four distinct tasks within IntelligentVBench: Implicit Image-to-Video (Implicit I2V), Interpolative Dual-Image-to-Video (Interpolative DI2V), Compositional Multi-Image-to-Video

(Compositional MI2V), Text-Image-Video-to-Video (TIV2V). The first two are explicitly designed to evaluate reasoning, whereas the latter two focus on composition.

Implicit I2V: Although structurally similar to standard I2V tasks, where an image provides the initial frame and a text prompt guides the generation, this task emphasizes implicit reasoning. It requires the model to logically unfold temporal events under high ambiguity, e.g., when the prompt expresses abstract intent without concrete actions, or when the described behavior lacks explicit grounding in the input image, or when text provides only preliminary assumptions requiring causal deduction.

Interpolative DI2V: It requires synthesizing a temporally-coherent video conditioned on two boundary frames and a text prompt, presenting three primary challenges: (a) large spatial disparities, requiring the generation of plausible physical trajectories or complex camera motions; (b) multi-subject dynamics, necessitating the inference of complex intermediate states for various characters; and (c) local environmental variations, which demand deducing the underlying events that drive these visual transformations.

Compositional MI2V: Given 1-4 reference images of subjects or backgrounds, alongside text prompts that may introduce additional objects, the objective is to seamlessly integrate these disparate elements into a coherent video while preserving the given visual identities.

TIV2V: This task assesses cross-modal compositional editing. Given a source video, reference images, and a text instruction, the model must extract specific concepts from the references and integrate them into the source video, while preserving the temporal consistency and visual fidelity of non-target elements.

In total, our benchmark comprises 1,030 test cases. Specifically, the Implicit I2V task contains 250 test cases; to construct highly challenging data, all input images are curated from movie clips featuring complex motions, with PhD experts formulating implicit instructions based on the corresponding cinematic plots. The Interpolative DI2V task also includes 250 test cases, where the paired input images are sourced from social networks or film segments to ensure they are contextually related yet exhibit significant visual variance. Furthermore, the Compositional MI2V task encompasses 320 test cases, each accepting one to four input images. And one to three images define the subject(s) and up to one image specifying the scene context. We further categorize this task into three subtasks based on the number of subject input images: 130 cases feature a single subject image input (with 78 cases of these including an additional scene image), 120 cases feature two subject image inputs (with 40 cases including an additional scene image), and 70 cases feature three subject image inputs (with 36 cases including an additional scene image). Finally, the TIV2V task comprises a total of 210 test cases spanning three distinct operational objectives: inserting an object from a reference image into the source video (71 test cases), substituting a specific visual element within the video with one from the reference image (73 test cases), and replacing the video’s original background with the scene depicted in the reference image (66 test cases).

- 5.3 Evaluation Metrics To comprehensively assess model performance on IntelligentVBench, we adopt the “VLM-as-a-Judge” (Zheng

- et al., 2023) paradigm, employing Gemini2.5-Pro (Comanici et al., 2025), as an automatic evaluator to establish three essential metrics for each task:

Instruction Following assesses how well the video executes the semantic and logical intent of the text prompt. For example, for the Implicit I2V task, it evaluates whether the model successfully captures the semantic correlation between the input image and the prompt to generate videos that accurately align with the user’s underlying intent. For the interpolative DI2V task, it gauges the intuitive transition logic between key-frames. For the compositional MI2V task, it examines whether the model accurately constructs all target visual elements with plausible interactions among multiple subjects following the prompt. Finally, for the TIV2V task, it measures the model’s capacity to seamlessly integrate the provided video and image in strict adherence to the instruction.

Condition Preserving evaluates the fidelity with which a generated video reconstructs and retains the identity, structure, and intricate details of provided visual references. Specifically, in Implicit I2V and Interpolative DI2V tasks, this metric assesses frame consistency by verifying whether the initial and final frames anchor flawlessly to the conditioning images without mutational drift. For Compositional MI2V, it measures subject and scene consistency by assessing whether the generated output faithfully preserves the multiple distinct identities and spatial layouts dictated by the input references. Finally, for TIV2V, the metric serves a dual purpose: it gauges the preservation of the source video’s unedited regions and original temporal dynamics, while simultaneously verifying that the newly introduced objects or scenes faithfully conform to the provided reference images.

Overall Visual Quality evaluates the general aesthetic quality, temporal consistency, motion smoothness, and physical naturalness of the generated video. This metric focuses on ensuring the outputs are fluid, obey real-world physics, and are devoid of noticeable AI artifacts or high-frequency flickering. Importantly, in the TIV2V task, if the source video inherently suffers from compromised aesthetic quality and the accompanying text prompt does not explicitly mandate aesthetic restoration, it should not be penalized during evaluation for retaining visual artifacts that were already present in the input video.

- Table 3: Main results of the Implicit I2V, Interpolative DI2V, and TIV2V tasks in IntelligentVBench. The best results are marked in bold for specialized- and unified-models, respectively. And the second best results in each category are underlined.

Implicit I2V Interpolative DI2V TIV2V

Model #Params

###### IF ↑ CP ↑ VQ ↑ MIN AVG IF ↑ CP ↑ VQ ↑ MIN AVG IF ↑ CP ↑ VQ ↑ MIN AVG

Specialized Video Generation Models

CogVideoX-I2V 5B 3.53 3.74 2.90 2.68 3.39 - - - - - - - - - Wan2.1-I2V 14B 3.76 3.33 3.34 2.78 3.48 - - - - - - - - - -

- Wan2.1-FLF2V 14B - - - - - 4.48 4.28 4.49 3.98 4.42 - - - - HunyuanVideo-I2V 13B 3.37 4.17 3.87 3.00 3.80 - - - - - - - - - HunyuanVideo1.5-I2V 8.3B 3.98 3.44 3.67 3.12 3.70 - - - - - - - - - -

- Wan2.2-I2V 14B 4.21 3.70 3.68 3.30 3.86 - - - - - - - - - Unified Video Generation Models

|VACE-Wan2.1 14B VACE-LTX 2B VINO 13B+4B UniVideo(query) 13B+7B UniVideo(hidden) 13B+7B<br><br>|3.66 3.22 3.32 2.76 3.40 3.04 3.28 2.28 2.20 2.87 3.70 2.64 2.96 2.28 3.10 3.66 3.37 3.16 2.77 3.39 3.74 3.82 3.28 2.95 3.61|4.01 3.61 3.96 3.32 3.86 3.51 3.13 2.67 2.59 3.10 2.38 1.73 3.29 1.65 2.47 2.31 1.85 3.27 1.76 2.48 2.36 1.98 3.40 1.84 2.58<br><br>|1.46 1.42 1.71 1.27 1.53<br><br>1.43 1.36 1.25 1.20 1.35<br>2.86 2.90 2.52 2.11 2.76<br>3.22 3.91 3.26 2.66 3.46 3.13 4.01 2.93 2.56 3.36<br><br><br>|
|---|---|---|---|
|Ours (w/o think) 8.3B+7B Ours (think) 8.3B+7B<br><br>|4.05 3.75 3.36 3.11 3.72 4.33 3.82 3.63 3.34 3.93<br><br>|4.60 4.22 4.52 3.99 4.45 4.74 4.38 4.50 4.11 4.54<br><br>|4.00 4.04 3.65 3.31 3.89 - - - - -|

Each metric is rated on a 1–5 scale based on carefully crafted prompts to ensure precise scoring. Furthermore, we establish two metrics to evaluate overall performance, formulated as AVG = (IF + CP + VQ)/3 and MIN = Min(IF, CP, VQ). The evaluation prompt template for Implicit I2V task is detailed in Figure 16 in Appendix C. The evaluation prompt template for Interpolative DI2V task is detailed in Figure 17. The evaluation prompt templates for Compositional MI2V task are detailed in Figure 18 and Figure 19. Furthermore, we design three distinct evaluation prompt templates tailored to the specific sub-tasks within the TIV2V task, as illustrated in Figure 20, Figure 21, and Figure 22.

#### 6 Experiments

###### 6.1 Experimental Setup

All evaluations are conducted in a zero-shot setup. In our comprehensive evaluation across both the proposed IntelligentVBench and existing benchmarks, we categorize baseline models into two primary paradigms: unified video generation models and task-specific video generation models. Specifically, we evaluate two architectural variants of VACE (Jiang et al., 2025), distinguished by their underlying backbones: VACE-Wan2.1 (Wan et al., 2025) and VACE-LTX (HaCohen et al., 2024). Furthermore, both VINO (Chen et al., 2026) and UniVideo (Wei et al., 2025) employ a unified architecture that couples an MLLM (Bai et al., 2025b) with a downstream diffusion model (Kong et al., 2024). Building on this, we assess two variants of UniVideo: UniVideo(query), which utilizes a query token mechanism (Pan et al., 2025b) to extract conditional embeddings from the MLLM, and UniVideo(hidden), which directly leverages the MLLM’s final hidden states to condition the diffusion model.

To establish task-specific baselines, we evaluate a diverse suite of models tailored to each task within IntelligentVBench. For Implicit I2V, we include CogVideoX-I2V (Yang et al., 2024), Wan2.1-I2V (Wan et al., 2025), HunyuanVideo-I2V (Kong et al., 2024), HunyuanVideo1.5-I2V (Wu et al., 2025a), and Wan2.2-I2V (Wan et al., 2025) as the baselines. For Interpolative DI2V, we employ Wan2.1-FLF2V (Wan et al., 2025) as the baseline. And for Compositional MI2V, we utilize SkyReels-A2 (Fei et al., 2025), SkyReels-V3 (Li et al., 2026), MAGREF (Deng et al., 2025b), and Phantom (Liu et al., 2025b). Notably, no specialized models currently exist for the TIV2V task. Furthermore, for video editing task on OpenVE-Bench, we incorporate six specialized editing models: OmniVideo (Tan et al., 2025), InsViE (Wu et al., 2025d), Lucy-Edit (Team, 2025), ICVE (Liao et al., 2025), Ditto (Bai et al., 2025a), and OpenVE-Edit (He et al., 2025). Finally, for text-to-video generation task on VBench, we include four task-specific baselines, i.e., StepVideo (Ma et al., 2025), CogVideoX (Yang et al., 2024), HunyuanVideo (Kong et al., 2024), and Wan2.1 (Wan et al., 2025).

###### 6.2 Main Results on IntelligentVBench

We conduct zero-shot evaluations on IntelligentVBench for both SoTA open-source task-specific specialized models and unified models. The primary results for Implicit I2V, Interpolative DI2V, and TIV2V are summarized in Table 3, while the performance for Compositional MI2V is detailed in Table 4. We identify three key findings regarding the limitations of existing methods:

(1) Current unified models exhibit significant performance imbalances across subtasks. For instance, VINO and VACE achieve competitive results in Compositional MI2V, yet fail significantly in Interpolative DI2V and TIV2V, respectively. Such discrepancies underscore that existing open-source models are far from truly “omni-capable”, lacking the robustness to handle diverse input formats while maintaining simultaneous reasoning and compositional proficiency. (2) Across individual subtasks, specialized models still surpass unified frameworks in reasoningintensive scenarios, such as Implicit I2V and Interpolative DI2V. In contrast, unified models exhibit superior perfor-

- Table 4: Main results of Compositional MI2V task in IntelligentVBench. We define 3 sub-categories based on the number of subjects presented in the 1–4 input images.

###### 1Subject (with BKG) 2Subjects (with BKG) 3Subjects (with BKG) IF ↑ CP ↑ VQ ↑ MIN AVG IF ↑ CP ↑ VQ ↑ MIN AVG IF ↑ CP ↑ VQ ↑ MIN AVG

Model #Params

Specialized Video Generation Models SkyReels-A2 14B 3.51 4.08 4.46 3.24 4.02 3.22 3.76 4.37 2.97 3.78 1.64 1.76 2.50 1.56 1.97 SkyReels-V3 14B 3.46 3.71 4.65 2.98 3.94 3.28 3.84 4.44 3.04 3.86 2.59 3.10 4.30 2.37 3.33 MAGREF 14B 3.15 2.48 4.32 2.18 3.32 3.04 2.81 4.33 2.44 3.39 2.50 2.21 4.46 2.07 3.06 Phantom 14B 3.21 2.95 4.29 2.47 3.48 2.88 3.42 4.38 2.62 3.55 2.36 2.79 4.21 2.20 3.12 Unified Video Generation Models

|VACE-Wan2.1 14B VACE-LTX 2B VINO 13B+4B UniVideo(query) 13B+7B UniVideo(hidden) 13B+7B<br><br>|3.88 4.48 4.68 3.62 4.35<br><br>2.74 2.86 2.89 2.25 2.83<br>3.72 4.22 4.46 3.42 4.13 3.35 3.90 4.41 3.01 3.89 3.33 4.18 4.38 3.08 3.97<br>|3.31 4.03 4.51 3.08 3.95<br><br>2.12 2.26 2.49 1.98 2.29<br><br>3.56 4.34 4.58 3.39 4.16<br><br><br>2.98 3.73 4.18 2.76 3.63<br><br>3.22 4.12 4.36 3.09 3.90<br>|2.60 3.03 4.40 2.50 3.34<br><br>1.94 2.06 2.41 1.89 2.14<br>2.63 2.97 4.24 2.53 3.28<br><br><br>2.30 2.50 3.89 2.20 2.89<br><br>2.31 2.83 3.94 2.29 3.03<br>|
|---|---|---|---|
|Ours 8.3B+7B<br><br>|4.35 4.53 4.58 4.01 4.49<br><br>|4.08 4.22 4.52 3.61 4.27<br><br>|3.53 4.01 4.54 3.26 4.03|

Table 5: Main results on OpenVE-Bench.

Model #Params Global-Style↑ BKG-Change↑ Local-Change↑ Local-RM↑ Local-Add↑ Subtitle-Edit↑ Overall↑ Specialized Video Generation Models

OmniVideo 1.3B 1.11 1.18 1.14 1.14 1.36 1.00 1.16 InsViE 2B 2.20 1.06 1.48 1.36 1.17 2.18 1.58 Lucy-Edit 5B 2.27 1.57 3.20 1.75 2.30 1.61 2.12 ICVE 13B 2.22 1.62 2.57 2.51 1.97 2.09 2.16 Ditto 14B 4.01 1.68 2.03 1.53 1.41 2.81 2.25 OpenVE-Edit 5B 3.16 2.36 2.98 1.85 2.15 2.91 2.57

Unified Video Generation Models

|VACE 14B VINO 13B+4B UniVideo(query) 13B+7B UniVideo(hidden) 13B+7B<br><br>|1.49 1.55 2.07 1.46 1.26 1.48 4.32 2.37 3.65 3.19 2.71 2.94 3.47 2.47 3.86 2.48 2.87 3.13 3.55 2.42 3.67 2.89 2.90 2.99<br><br>|1.55 3.11 3.05 3.09<br><br>|
|---|---|---|
|Ours 8.3B+7B|3.91 2.30 3.65 3.22 2.30 3.53<br><br>|3.15|

mance in Compositional MI2V. This discrepancy suggests that while current multi-task training effectively captures compositional attributes, the capacity for reasoning-informed generation remains underdeveloped. (3) Across the four tasks, TIV2V—requiring the simultaneous synthesis of text, image, and video modalities—demonstrates the lowest peak performance. The current absence of specialized models for this task further underscores a significant research gap, suggesting that future advancements in video generation must prioritize the seamless integration and coordination of increasingly diverse multimodal inputs.

Compared to existing models, Omni-Weaving achieves the following advantages: (1) Synergistic Multi-Task Integration. Across all four tasks, Omni-Weaving consistently achieves SoTA performance under both MIN and AVG overall metrics. Unlike previous unified models, it demonstrates a synergistic integration that mitigates intertask competition and mutual suppression. (2) Enhanced Multimodal Composition. Whether processing complex multi-subject image combinations or heterogeneous tri-modal inputs (image, video, and text), Omni-Weaving outperforms all baselines, showcasing superior compositional flexibility for video generation. (3) ComprehensionGuided Generation. For reasoning-related tasks like Implicit I2V and Interpolative DI2V, enabling the “thinking mode” of MLLM yields consistently superior results compared to direct generation. Notably, on Implicit I2V, OmniWeaving underperforms Wan2.2 without thinking; however, activating thinking enables a performance reversal to establish a new SoTA. This confirms that our framework successfully unifies comprehension and generation, leveraging deep reasoning to elevate visual synthesis.

###### 6.3 Main Results on Existing Benchmarks (T2V and V2V)

We conduct zero-shot evaluations on VBench (Huang et al., 2024) for text-to-video (T2V) generation and OpenVE-Bench (He et al., 2025) for video-to-video (V2V) editing, comparing OmniWeaving against both unified video generation models and specialized models.

Table 6: Main Results on VBench.

Model #Params Quality↑ Semantic↑ Total↑ Specialized Video Generation Models StepVideo 30B 84.46 71.28 81.83 CogVideoX 5B 83.05 77.33 81.91 HunyuanVideo 13B 85.09 75.82 83.24 Wan2.1 14B 85.59 76.11 83.69 Unified Video Generation Models

Table 6 presents the main VBench results for text-tovideo (T2V) generation, with baseline scores sourced from Chen et al. (2026). We can see that OmniWeaving outperforms existing unified frameworks on VBench, achieving performance comparable to specialized models like HunyuanVideo. Given that T2V data constitutes less than 10% of our training corpus, OmniWeaving successfully preserves its foundational generation capabilities without significant degradation.

|VINO 13B+4B UniVideo 13B+7B|84.00 78.01 - -<br><br>|82.80 82.58|
|---|---|---|
|Ours 8.3B+7B|84.37 78.03|83.10|

[Figure 146]

Figure 5: (a) AVG performance when enabling or disabling the thinking mode of OmniWeaving. (b) AVG performance with different DeepStacking strategies. (c) Performance visualization across different input formats for each unified video generation model.

- Table 5 presents the main V2V editing results on OpenVE-Bench. We adopt most baseline scores directly from He et al. (2025), whereas the results for Univideo and VINO are obtained through our own evaluations. We can see that OmniWeaving yields an average score of 3.15 across the six OpenVE-Bench subtasks, surpassing specialized models and unified counterparts such as UniVideo and VINO. Notably, our model demonstrates well-balanced proficiency across global, local, and text-based editing. However, we do observe a relative performance dip in the “local-add” subtask. We find that this discrepancy arises because this subtask primarily features animated content addition with minimal emphasis on natural integration, whereas our training explicitly prioritizes realistic object addition for this editing type.

###### 6.4 Qualitative Comparisons

As illustrated in Figure 6, we conduct a qualitative comparison of VINO, UniVideo, and OmniWeaving. In both Intent-Driven I2V and Interpolative DI2V tasks, the two baselines frequently produce start or end frames that are misaligned with the provided reference images. Moreover, in Compositional MI2V, both baselines struggle to successfully integrate all specified visual elements; for instance, in the first case, the videos synthesized by both VINO and UniVideo contain only two individuals, whereas in the second case, although UniVideo manages to incorporate all three subjects, it entirely disregards the background constraints. Finally, in TIV2V, the baselines often introduce unintended modifications to the unedited regions of the source video, while sometimes failing to accurately respond to the user’s prompts. In contrast, OmniWeaving effectively mitigates these issues, showing higher-quality generation across all cases. See more qualitative examples of OmniWeaving in Figures 11, 12, 13, 14, and 15 within Appendix B.

###### 6.5 In-Depth Analysis

Effect of Reasoning-Augmented Video Generation. As shown in Figure 5(a), we evaluate the AVG performance of OmniWeaving and its precursor prior to the final training stage on Implicit I2V and Interpolative DI2V tasks, both with and without “think mode” activated. Notably, before training stage 3, enabling “think mode” severely degrades performance, revealing a lack of synergy between comprehension and generation. In contrast, after Reasoning-Augmented Fine-Tuning, “think mode” significantly boosts overall results, demonstrating that enhanced reasoning effectively drives higher-quality video generation.

Effect of DeepStacking. To evaluate the impact of DeepStacking, we fine-tune the Stage-1 model on a multimodalcomposition subset with various layer-selection strategies. Figure 5(b) compares their average performance on Compositional MI2V. When fixing the selection to three layers, sampling across a broad depth (e.g., [8, 16, 24]) outperforms concentrations in exclusively shallow ([2, 4, 6]) or deep ([24, 25, 27]) layers, with all configurations surpassing non-DeepStacking. Furthermore, this 3-layer setup achieves superior results compared to 2-layer ([12, 24]) or 4-layer ([6, 12, 18, 24]) alternatives. These findings suggest that integrating a balanced spectrum of semantic features—spanning low-level to high-level—optimizes compositional video generation.

Capability Visualization. Figure 5(c) presents a radar chart visualizing the average performance scores across diverse input formats on established test cases for each unified model. It intuitively shows that OmniWeaving consistently outperforms baselines, exhibiting superior versatility and robustness when processing a wide variety of input formats.

Assessment of Evaluation Protocol. To validate the evaluation reliability for IntelligentVBench, we conducted a user study comparing human expert ratings with evaluations from several frontier

Table 7: Pearson correlation with human ratings for each VLM.

|VLMs<br><br>|IF ↑ CP ↑ VQ ↑ AVG|
|---|---|
|Qwen3-VL-235B Seed-1.6 GPT-5|0.66 0.54 0.59 0.60 0.72 0.63 0.67 0.67 0.75 0.72 0.71 0.73<br><br>|
|Gemini2.5-Pro<br><br>|0.81 0.74 0.72 0.76|

###### Model Implicit I2V Interpolative DI2V

|[Figure 147]|
|---|

|[Figure 148]|
|---|

|[Figure 149]|
|---|

She is fleeing from a danger, seeking cover near the building.

Scissors cut into red circles on paper, partially slicing them out.

Input

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

VINO

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

UniVideo

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

OmniWeaving

|[Figure 180]|
|---|

|[Figure 181]|
|---|

|[Figure 182]|
|---|

Two girls were reunited with their long-lost dog.

Woman bites burger, chews, lifts and rotates it for camera close-up.

Input

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

VINO

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

UniVideo

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

OmniWeaving

Compositional Multi-Image-to-Video

The man in the first image starts talking. The woman in the second image stands still. The woman in the third image also stands still with her arms crossed.

The man in the first image and woman in the second image talk with mouths opening and closing over the background in the fourth image. The drone in the third image rotates steadily clockwise.

[Figure 213]

|[Figure 214]|
|---|

|[Figure 215]|
|---|

[Figure 216]

Input

|[Figure 217]|
|---|

|[Figure 218]|
|---|

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

VINO

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

UniVideo

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

OmniWeaving

Text-Image-Video-to-Video

|[Figure 250]|[Figure 251]|[Figure 252]|[Figure 253]|[Figure 254]|
|---|---|---|---|---|

|[Figure 255]|[Figure 256]|[Figure 257]|[Figure 258]|[Figure 259]|
|---|---|---|---|---|

Input

|[Figure 260]|
|---|

[Figure 261]

Replace the sofa in the video with the brown sofa in the image.

Replace the background of the video with the scene shown in the image.

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

VINO

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

UniVideo

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

OmniWeaving

|[Figure 292]|[Figure 293]|[Figure 294]|[Figure 295]|[Figure 296]|
|---|---|---|---|---|

|[Figure 297]|[Figure 298]|[Figure 299]|[Figure 300]|[Figure 301]|
|---|---|---|---|---|

Input

Add the woman in the image seated next to the person in the wheelchair in the video.

Add the seated person in the image back into the hospital bed in the video, facing the doctor.

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

VINO

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

UniVideo

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

OmniWeaving

Figure 6: Qualitative comparison of VINO, UniVideo, and OmniWeaving.

VLMs, including Gemini2.5-Pro (Comanici et al., 2025), GPT-5 (OpenAI, 2025), Seed-1.6 (Guo et al., 2025), and Qwen3-VL (Bai et al., 2025b). By calculating the Pearson correlation coefficient across randomly sampled representative cases from each subtask, we assessed the alignment between automated scores and human judgment. As shown in Table 7, Gemini2.5-Pro consistently achieves the highest correlation across all metrics, demonstrating its superior proficiency in accurately reflecting model performance.

#### 7 Conclusion and Future Work

In this paper, we introduce OmniWeaving, an omni-level video generation model featuring powerful multimodal composition and reasoning-informed generation capabilities. OmniWeaving can accommodate free-form, multimodal interleaved inputs to generate compliant videos. It is also capable of leveraging deep visual understanding to actively guide the generative process with a unified visual comprehension and generation framework. Furthermore, we propose IntelligentVBench, a comprehensive benchmark specifically designed to evaluate next-level intelligent video generation, containing diverse tasks. Extensive experiments show that OmniWeaving achieves SoTA performance across existing open-source unified frameworks and even surpasses specialized models.

Furthermore, it is imperative to clarify that OmniWeaving is not designed to entirely bridge the performance gap between open-source and closed-source models. We concede that a substantial disparity still persists between it and proprietary counterparts such as Seedance-2.0. This gap is evident not only in overall model performance but also in the diversity of supported input modalities and format flexibility, largely because closed-source models benefit from significantly greater computational resources and training data. Closing this gap is neither our objective for OmniWeaving nor a realistic short-term expectation. Instead, through the exploration of OmniWeaving and its open-source release, we aim to provide the community with a viable reference point to help guide the future trajectory of unified video generation models. Moving forward, we will continue to explore unified video generation. Our future efforts will focus on supporting more complex inputs, such as interleaved multiple-image-and-video sequences, and incorporating additional modalities, such as audio input and output, to ultimately achieve fully omni-modal, audio-visually synchronized video generation.

#### References

Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, et al. Scaling instruction-based video editing with a high-quality synthetic dataset. arXiv preprint arXiv:2510.15742, 2025a.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025b.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025c. URL https://arxiv.org/abs/2502.13923.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

Junyi Chen, Tong He, Zhoujie Fu, Pengfei Wan, Kun Gai, and Weicai Ye. Vino: A unified visual generator with interleaved omnimodal context. arXiv preprint arXiv:2601.02358, 2026.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025a.

Yufan Deng, Xun Guo, Yuanyang Yin, Jacob Zhiyuan Fang, Yiding Yang, Yizhi Wang, Shenghai Yuan, Angtian Wang, Bo Liu, Haibin Huang, et al. Magref: Masked guidance for any-reference video generation. arXiv e-prints, pp. arXiv–2505, 2025b.

Zhengcong Fei, Debang Li, Di Qiu, Jiahua Wang, Yikun Dou, Rui Wang, Jingtao Xu, Mingyuan Fan, Guibin Chen, Yang Li, et al. Skyreels-a2: Compose anything in video diffusion transformers. arXiv preprint arXiv:2504.02436, 2025.

Google. Introducing gemini 2.5 flash image, our state-of-the-art image model. https://developers. googleblog.com/en/introducing-gemini-2-5-flash-image/?utm_source=chatgpt. com, 2025a.

Google. Gemini ai video generator powered by veo 3.1. https://gemini.google/overview/ video-generation/, 2025b.

Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

Haoyang He, Jie Wang, Jiangning Zhang, Zhucun Xue, Xingyuan Bu, Qiangpeng Yang, Shilei Wen, and Lei Xie. Openve-3m: A large-scale high-quality dataset for instruction-guided video editing. arXiv preprint arXiv:2512.07826, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21807–21818, 2024.

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17191–17202, 2025.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Black Forest Labs. Flux.2 - next generation image generation. https://bfl.ai/models/flux-2, 2025. Debang Li, Zhengcong Fei, Tuanhui Li, Yikun Dou, Zheng Chen, Jiangping Yang, Mingyuan Fan, Jingtao Xu,

Jiahua Wang, Baoxuan Gu, et al. Skyreels-v3 technique report. arXiv preprint arXiv:2601.17323, 2026. Xinyao Liao, Xianfang Zeng, Ziye Song, Zhoujie Fu, Gang Yu, and Guosheng Lin. In-context learning with unpaired clips for instruction-based video editing. arXiv preprint arXiv:2510.14648, 2025. Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982, 2025a.

Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14951–14961, 2025b.

Guoqing Ma, Haoyang Huang, Kun Yan, Liangyu Chen, Nan Duan, Shengming Yin, Changyi Wan, Ranchen Ming, Xiaoniu Song, Xing Chen, et al. Step-video-t2v technical report: The practice, challenges, and future of video foundation model. arXiv preprint arXiv:2502.10248, 2025.

OpenAI. Introducing GPT-5. https://openai.com/index/introducing-gpt-5/, 2025. Accessed: 2026-03-05.

OpenAI, Aaron Hurst, Adam Lerer, et al. Gpt-4o system card. Technical report, OpenAI, 2024. URL https: //arxiv.org/abs/2410.21276.

Kaihang Pan, Siliang Tang, Juncheng Li, Zhaoyu Fan, Wei Chow, Shuicheng Yan, Tat-Seng Chua, Yueting Zhuang, and Hanwang Zhang. Auto-encoding morph-tokens for multimodal llm. arXiv preprint arXiv:2405.01926, 2024.

Kaihang Pan, Wang Lin, Zhongqi Yue, Tenglong Ao, Liyu Jia, Wei Zhao, Juncheng Li, Siliang Tang, and Hanwang Zhang. Generative multimodal pretraining with discrete diffusion timestep tokens. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26136–26146, 2025a.

Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025b.

ByteDance Seed. Seedance 2.0. https://seed.bytedance.com/en/seedance2_0, 2026. Uriel Singer, Amit Zohar, Yuval Kirstain, Shelly Sheynin, Adam Polyak, Devi Parikh, and Yaniv Taigman. Video

editing via factorized diffusion distillation. In European Conference on Computer Vision, pp. 450–466. Springer, 2024.

Zhiyu Tan, Hao Yang, Luozheng Qin, Jia Gong, Mengping Yang, and Hao Li. Omni-video: Democratizing unified

video understanding and generation. arXiv preprint arXiv:2507.06119, 2025. DecartAI Team. Lucy edit: Open-weight text-guided video editing, 2025. Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He,

Kang He, et al. Kling-omni technical report. arXiv preprint arXiv:2512.16776, 2025.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Cong Wei, Quande Liu, Zixuan Ye, Qiulin Wang, Xintao Wang, Pengfei Wan, Kun Gai, and Wenhu Chen. Univideo: Unified understanding, generation, and editing for videos. arXiv preprint arXiv:2510.08377, 2025.

Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Jack Peng, Jianbing Wu, Jiangfeng Xiong, Jie Jiang, et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025a.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025b.

Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025c.

Yuhui Wu, Liyi Chen, Ruibin Li, Shihao Wang, Chenxi Xie, and Lei Zhang. Insvie-1m: Effective instruction-based video editing with elaborate dataset construction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16692–16701, 2025d.

Bin Xia, Yuechen Zhang, Jingyao Li, Chengyao Wang, Yitong Wang, Xinglong Wu, Bei Yu, and Jiaya Jia. Dreamomni: Unified image generation and editing. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 28533–28543, 2025.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13294–13304, 2025.

Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. Byt5: Towards a token-free future with pre-trained byte-to-byte models. Transactions of the Association for Computational Linguistics, 10:291–306, 2022.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Hao Yang, Zhiyu Tan, Jia Gong, Luozheng Qin, Hesen Chen, Xiaomeng Yang, Yuqing Sun, Yuetan Lin, Mengping Yang, and Hao Li. Omni-video 2: Scaling mllm-conditioned diffusion for unified video generation and editing. arXiv preprint arXiv:2602.08820, 2026.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Shenghai Yuan, Xianyi He, Yufan Deng, Yang Ye, Jinfa Huang, Bin Lin, Jiebo Luo, and Li Yuan. Opens2v-nexus: A detailed benchmark and million-scale dataset for subject-to-video generation. arXiv preprint arXiv:2505.20292, 2025.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11975–11986, 2023.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Appendix

- A Model Details

Building upon the general architecture outlined in Section 4.1, we detail the specific design and implementation of our model by instantiating OmniWeaving with Qwen2.5-VL (Bai et al., 2025c) as the MLLM and HunyuanVideo1.5 (Wu et al., 2025a) as the core generative engine. Because HunyuanVideo-1.5 utilizes a multilingual Glyph-ByT5 text encoder (Xue et al., 2022) for precise cross-lingual text rendering and a SigLIP (Zhai et al., 2023) encoder for robust vision-text semantic alignment, we retain both modules to preserve architectural consistency with the generative backbone. Specifically, in scenarios involving multiple visual inputs, the global SigLIP feature is computed as the mean embedding across all provided images. Similarly, for video inputs, we extract a representative set of frames and average their corresponding SigLIP embeddings to yield a unified semantic representation. Notably, both the Glyph-ByT5 and SigLIP encoders remain strictly frozen throughout the entire training process.

Furthermore, a key aspect of multitask training is that the representations provided by the MLLM should enable the MMDiT to understand exactly what task needs to be performed (Wu et al., 2025b). To this end, we set distinct MLLM system prompts for different tasks to facilitate better differentiation. The system prompts for each video generation task are shown in Figure 7, Figure 8, Figure 9, and Figure 10. For reasoning-augmented tasks, we append a specialized reasoning-guidance prompt to the user input (i.e., based on the provided image and text, please generate a detailed description that explicitly articulates the visual, temporal, and semantic characteristics of the target video). This reasoning-guidance prompt explicitly directs the MLLM to perform reasoning, yielding an enhanced prompt that provides a granular description of the target video’s spatial composition and temporal dynamics. Subsequently, the hidden states corresponding to this enhanced prompt are concatenated with those of the original user input; this combined representation is then fed into the MMDiT to effectively steer the generative process.

- B More Experimental Results

More qualitative results for OmniWeaving’s video generation across various task scenarios are presented in Figure 11, Figure 12, Figure 13, Figure 14, and Figure 15. As can be seen, OmniWeaving integrates diverse video generation capabilities within a single model, enabling it to effectively handle various video generation tasks.

- C Benchmark Evaluation Prompts

This section presents the evaluation prompt templates tailored for each task in IntelligentVBench. Overall, we utilize Gemini 2.5 Pro as an automated evaluator to assess three essential metrics per task, with each metric scored on a 1–5 scale. Specifically, the evaluation prompt template for Implicit I2V is detailed in Figure 16. The evaluation prompt template for Interpolative DI2V is detailed in Figure 17. The evaluation prompt templates for Compositional MI2V are detailed in Figure 19 and Figure 18. Furthermore, we design three distinct evaluation prompt templates tailored to the specific sub-tasks within the TIV2V task, as shown in Figure 20, Figure 21, and Figure 22.

System Prompt for Text-to-Video Generation Task

<|im_start|>system

You are a helpful assistant. Describe the video by detailing the following aspects: (1) The main content and theme of the video. (2) The color, shape, size, texture, quantity, text, and spatial relationships of the objects. (3) Actions, events, behaviors temporal relationships, physical movement changes of the objects. (4) Background environment, light, style and atmosphere. (5) camera angles, movements, and transitions used in the video.<|im_end|>

<|im_start|>user <|user_text|><|im_end|> <|im_start|>assistant

System Prompt for First-Frame-to-Video Generation Task

<|im_start|>system

You are a helpful assistant. Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user’s text instruction should alter the image to introduce motion and evolution over time. Generate a video using this image as the first frame that meets the user’s requirements, ensuring the specified elements evolve or move in a way that fulfills the text description while maintaining consistency. <|im_end|>

<|im_start|>user <|user_img|><|user_text|><|im_end|> <|im_start|>assistant

- Figure 7: System prompts for Text-to-Video and First-Frame-to-Video generation tasks, where <|user img|> is the user input image and <|user text|> is the user input prompt.

System Prompt for Key-Frames-to-Video Generation Task

<|im_start|>system

You are a helpful assistant. Given a text instruction and multiple images (greater than 1) as key-frames of the video, you need to analyze the visual trajectory required to transition from the start to the end across these key-frames. Determine how the elements in each key-frame must evolve, move, or transform to align with the subsequent key-frame based on the text instruction. Generate a video that seamlessly connects these key-frames, ensuring the motion and evolution between them fulfill the text description while maintaining temporal consistency. <|im_end|>

<|im_start|>user <|key_frame_1|>...<|key_frame_n|><|user_text|><|im_end|> <|im_start|>assistant

System Prompt for Video-to-Video Editing Task

<|im_start|>system

You are a helpful assistant. Given a text instruction and an input video, you need to analyze the visual content and temporal dynamics of the input video, and then explain how the user’s text instruction should modify the video’s visual style, objects, or scene composition. Generate an edited video that meets the user’s requirements, ensuring the specified modifications are applied consistently across frames while preserving the original motion flow and coherence.<|im_end|>

<|im_start|>user <|user_video|><|user_text|><|im_end|> <|im_start|>assistant

- Figure 8: System prompts for Key-Frame-to-Video generation and Video-to-Video editing tasks, where <|user text|> is the user input prompt, <|key frame i|> is the user input image, and <|user video|> is the user input video.

System Prompt for Interleaved Text-and-Multi-Image-to-Video Generation Task

<|im_start|>system

You are a helpful assistant. Given text instructions and one or more input images, you need to explain how to extract and combine key information from the input images to construct a new image as the video’s first frame, and then how the user’s text instruction should alter the image to introduce motion and evolution over time. Generate a video that meets the user’s requirements, ensuring the specified elements evolve or move in a way that fulfills the text description while maintaining consistency.<|im_end|>

<|im_start|>user <|user_img_1|><|user_text_1|>...<|user_img_n|><|user_text_n|><|im_end|> <|im_start|>assistant

System Prompt for Text-Image-Video-to-Video Generation Task

<|im_start|>system

You are a helpful assistant. Given a text instruction, one or more reference images, and an input video, you need to analyze the visual content and temporal dynamics of the input video, alongside the scene or subject characteristics of the reference images. Explain how the user’s text instruction directs the application of the reference images’ visual attributes onto the input video. Generate an edited video that meets the user’s requirements, ensuring the specified modifications are applied consistently across frames while preserving the original motion flow and coherence. <|im_end|>

<|im_start|>user <|user_video|><|user_img_1|>...<|user_img_n|><|user_text|><|im_end|> <|im_start|>assistant

- Figure 9: System prompts for Interleaved Text-and-Multi-Image-to-Video and Text-Image-Video-to-Video generation tasks, where <|user image i|> is the user input images, and <|user video|> is the user input video.

System Prompt for Reasoning-augmented Task (Example 1)

<|im_start|>system

You are a helpful assistant. Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user’s text instruction should alter the image to introduce motion and evolution over time. Note that the text instruction may implicitly express the intent of video generation. Generate a video using this image as the first frame that meets the user’s requirements, ensuring the specified elements evolve or move in a way that fulfills the text description while maintaining consistency. <|im_end|>

<|im_start|>user <|user_img|><|user_text|> Based on the provided image and text, please generate a detailed description that explicitly articulates the visual, temporal, and semantic characteristics of the target video.<|im_end|> <|im_start|>assistant

System Prompt for Reasoning-augmented Task (Example 2)

<|im_start|>system

You are a helpful assistant. Given a text instruction and multiple images (greater than 1) as key-frames of the video, you need to analyze the visual trajectory required to transition from the start to the end across these key-frames. Determine how the elements in each key-frame must evolve, move, or transform to align with the subsequent key-frame based on the text instruction. Note that the text instruction may implicitly express the intent of video generation. Generate a video that seamlessly connects these key-frames, ensuring the motion and evolution between them fulfill the text description while maintaining temporal consistency. <|im_end|>

<|im_start|>user <|key_frame_1|>...<|key_frame_n|><|user_text|> Based on the provided image and text, please generate a detailed description that explicitly articulates the visual, temporal, and semantic characteristics of the target video.<|im_end|> <|im_start|>assistant

Figure 10: Two System prompt examples for reasoning-augmented video generation tasks.

###### Text-to-Video First-Frame-to-Video

A man stands before a burning plane at dusk.

[Figure 334]

While a woman is tending to a, two curious children suddenly burst into the scene with surprised expressions.

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

A man in a suit looking around nervously.

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

The woman smiles, closes her eyes, and leans her head toward the man. The man also leans his head toward the woman, and they kiss.

Cars driving on a highway bridge.

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

A British Shorthair cat adjusting its gaze, camera zooms in.

[Figure 353]

A man in a floral shirt dances energetically by a poolside at night, camera pans right

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

###### Key-Frames-to-Video

[Figure 361]

In a sunny park, a young woman and a man in sportswear run on a dirt path. She leads, he follows, smiling

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

The scene shifts from workers hurriedly moving crates amid swirling dust to a man in a suit directing operations beside a goggled companion

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

Paddington shifts from surprise to contemplation while clutching honey jar.

[Figure 378]

[Figure 379]

Generate a complete video based on the provided first-frame and last-frame.

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

Figure 11: Qualitative results for OmniWeaving on Text-to-Video, First-Frame-to-Video, and Key-Frames-to-Video generation tasks.

###### Video-to-Video Editing

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

Convert the sketchanimated video to a live-action, photorealistic format.

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

Turn the video into an oil painting style.

[Figure 397]

[Figure 398]

[Figure 399]

Replace the futuristic cyber garage with a industrial garage environment

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

Replace the background with a futuristic scene

Add a bearded man wearing a baseball cap, standing in front of stacked wooden barrels, making hand gestures.

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

x

Add a person lying backward on the red exercise ball, with her back arched over the ball.

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

x

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

Remove the firefighter clad in a vibrant red uniform.

Change man‘s hair color to platinum blonde while keeping hair shape.

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

Add "What if that closed shutter just creaked open?" subtitles at the center of the video.

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

Figure 12: Qualitative results for OmniWeaving on Video-to-Video editing tasks.

###### Compositional Multi-Image-to-Video

A woman holds up the light blue shirt in the first image, spreading it open with both hands over the background in the second image.

[Figure 439]

[Figure 440]

[Figure 441]

The woman in the image is sitting in the car talking, while the man seated next to her remains motionless as he drives.

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

The woman in the first image kneels by an open suitcase, sorting clothes. The woman in the second image sits nearby, watching.

The camera slowly zooms in on the pile of red beans in the first image nestled in a woven basket, with the spoon in the second image placed upon them.

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

The woman in the first image holds the handbag in the second image up towards the camera while speaking over the background in the third image.

[Figure 459]

The cat in the first image, appearing angry over the background of the third image, looks at the mouse from the second image that he is holding.

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

The man in the first image starts talking. The woman in the second image stands still. The woman in the third image also stands still with her arms crossed.

The cars in the first, second, and third images are driving on the road.

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

The pink handbag in the first image, the green handbag in the second image, and the blue handbag in the third image remain stationary but shift slightly in the frame due to the camera's slow upward movement over the background in the fourth image.

[Figure 483]

[Figure 484]

[Figure 485]

The man in the first image talks animatedly to the camera against the background in the fourth image. The man in the second image smiles. The woman in the third image stands between them and keeps her thumbs-up.

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

Figure 13: Qualitative results for OmniWeaving on Compositional Multi-Image-to-Video generation tasks.

###### Text-Image-Video-to-Video

[Figure 496]

Replace the man in the video with the man in the image.

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

Replace the green tree in the video with the autumn-colored tree in the image.

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

Replace the white refrigerator in the video with the refrigerator in the image.

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

Add the white car in the image that moves with the shadow on the left of the centerline in the video.

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

Let the woman on the far left in the video wear the hat shown in the image.

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

Replace the background of the video with the scene shown in the image.

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

Replace the background of the video with the scene shown in the image.

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

Figure 14: Qualitative results for OmniWeaving on Text-Image-Video-to-Video generation tasks.

###### Reasoning-Augmented Video Generation

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

The armed men initiated a sudden attack or raid, causing chaos.

Reason: The video begins with a narrow cobblestone street lined with weathered stone buildings, evoking a historical or Mediterranean ambiance. Then two armed men in dark clothing emerge from behind a corner, moving quickly and aggressively toward the center of the frame. Their presence disrupts the calm, causing panic among the seated individuals who scatter in fear. The camera follows their movement, capturing the chaos as they advance through the crowd, with some individuals falling or being knocked down by the sudden violence. The scene conveys urgency and danger, with the armed men’s actions altering the tranquil market atmosphere into one of turmoil and conflict.

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

The crowd led by the man had previously suffered unfair treatment.

Reason: The man and the crowd in the protest scene are dressed in black, with many wearing headscarves, and are holding up signs and raising their fists in unison. The man in the foreground has a beard and is looking upward with a serious expression. Behind him, other protesters are shouting and gesturing emphatically. The background features a statue and a building under an overcast sky, suggesting a somber or tense atmosphere. The camera remains static throughout the clip, capturing the intensity of the moment without any movement or change in framing.

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

Generate a video based on the provided first and last frames.

[Figure 559]

[Figure 560]

Reason: A child uses a red pump to inflate a basketball, which grows visibly larger as the scene progresses, while another basketball remains deflated nearby.

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

Generate a video based on the provided first and last frames.

[Figure 565]

[Figure 566]

Reason: Starting with a close-up of the turquoise car's wheel well, the camera smoothly tilts upward, revealing the car’s hood and then the dark grey Volvo station wagon parked on grass in the background, with both license plates visible under soft daylight.

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

Generate a video based on the provided first and last frames.

[Figure 571]

[Figure 572]

Reason: A gloved hand enters from the left, sprinkling red chili flakes onto the grated cheese topping on the meatballs, enhancing their appearance against the rustic wooden platter and warm wooden table.

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

Generate a video based on the provided first and last frames.

[Figure 577]

[Figure 578]

Reason: The camera begins focused on a crosswalk where pedestrians are crossing, then smoothly pans right, revealing parked cars lining the sidewalk, and finally settles on a parking sign, capturing the urban street scene.

Figure 15: Qualitative results for OmniWeaving on Reasoning-Augmented video generation.

Evaluation Prompt Template for Implicit I2V Task

You are an expert data rater specializing in evaluating Image-to-Video (I2V) generation. You will be provided with a reference image (serving as the first frame), a natural language instruction, and the generated video. Note that the instruction may describe abstract intent, introduce new characters or objects not present in the input image, or may lack an explicit association with the input image. Your task is to evaluate the generated video on a 5-point scale across three dimensions:

- 1. The first score: Condition Preserving Objective: Evaluate whether the first frame of the generated video perfectly anchors to the input image. Maintaining style consistency and coherence in subsequent frames relative to the first frame is considered a secondary, advanced requirement for achieving the highest scores.

- 5: Perfect Consistency. The first frame is an identical reconstruction of the input image. Additionally, as an advanced requirement, subsequent frames maintain impeccable style consistency and visual coherence with the first frame without abrupt mutations. - 4: High Fidelity. The first frame is highly consistent with the input image, with only negligible differences (e.g., micro-scaling or slight cropping). Minor stylistic drifting or small visual jumps in subsequent frames are acceptable, provided the first frame is nearly perfect. - 3: Moderate Consistency. The first frame is recognizable as the input image but shows visible shifts in color, sharpness, or structural details. The score drops here primarily because the starting point is flawed, making subsequent frame behavior less relevant. - 2: Low Fidelity. The first frame deviates significantly from the input (e.g., distorted composition, major style shifts, or missing key elements). The sequence is heavily penalized due to the initial frame mismatch, regardless of what happens next. - 1: Total Dissociation. The video completely fails to use the input image as the first frame. There is zero visual connection to the reference from the very beginning.

- 2. The second score: Instruction Following Objective: Assess how well the video executes the semantic intent of the prompt, including the ”silky” integration of new elements and the logical evolution of the scene.

- 5: Perfect Alignment. Every detail of the instruction is perfectly rendered. If new characters or objects are introduced, they appear through logical, seamless transitions rather than abrupt popping. The semantic progression is intuitive and creative. - 4: Good Alignment. The core semantics of the prompt are successfully captured. New elements are integrated well, though there might be minor imperfections in the pacing of the motion or slight omissions of secondary descriptors. - 3: Partial Alignment. The model captures the main idea but ignores specific nuances. Transitions involving new elements may feel slightly awkward or forced, and the video evolution only partially matches the intended narrative. - 2: Weak Alignment. The video mostly ignores the prompt or only reflects isolated keywords. The movement or introduction of elements feels chaotic, illogical, or contradicts the instruction’s intent. - 1: No Alignment. The video content has no relevance to the natural language instruction provided.

- 3. The third score: Overall Visual Quality Objective: Evaluate the general aesthetic quality, temporal consistency (flicker-free), motion smoothness, and physical naturalness (absence of severe artifacts or morphing) of the generated video, independent of the input image or prompt.

- - 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. Motions are fluid and natural (no jitter or dropped frames), physical transformations obey real-world physics, and there is zero flickering or artifacts. - 4: Good. High overall quality. Smooth and coherent, with only minor, easily forgivable artifacts or very slight flickering occurring in complex motion areas or fine edges. Overall viewing experience remains pleasant.
- - 3: Fair. Acceptable but noticeably flawed. Visible visual anomalies are present, such as noticeable flickering, mild stuttering/jittering, moderate morphological distortions during movement, or obvious AI artifacts. - 2: Poor. Significantly degraded quality. Severe visual and temporal defects dominate the video, including high-frequency flickering, severe motion rigidity, grotesque ”melting” or distortion of objects during transitions, and heavy noise. - 1: Unacceptable. Completely collapsed visual integrity. The video lacks any temporal coherence. It is heavily corrupted with extreme artifacts, tearing, or noise, making it unwatchable and devoid of aesthetic value.

Example Response Format: You are required to return a dictionary structured as follows: {”Condition Preserving”: [A number from 1 to 5], ”Instruction Following”: [A number from 1 to 5], ”Overall Visual Quality”: [A number from 1 to 5]}.

<|user_input|>

Figure 16: Evaluation prompt template for Implicit I2V task.

Evaluation Prompt Template for Interpolative DI2V Task

You are an expert data rater specializing in evaluating video generation conditioned on first and last frames. You will be provided with two reference images, the target first frame and the target last frame, along with a text instruction, and the generated video. The instruction requires the video to transition logically from the first frame to the last frame based on the provided text. Your task is to evaluate the generated video on a 5-point scale from three perspectives:

- 1. The first score: Condition Preserving Objective: Evaluate how accurately the generated video’s first and last frames reconstruct the two provided input images (including color, lighting, composition, and fine details).

- 5: Perfect Match. The first and last frames of the generated video perfectly replicate the provided images. There is no perceptible cropping, or color shift. - 4: High Fidelity. The first and last frames are highly consistent with the provided images. The main subjects are identical, with only microscopic, easily overlooked flaws (e.g., slight lighting variance or minuscule distortion at the extreme edges). - 3: Moderate Fidelity. The first and last frames are generally recognizable as the input images but contain visible discrepancies. Noticeable issues such as mild color distortion, blurred background details, or slight morphological changes are present, though the core identity remains intact. - 2: Low Fidelity. The first or the frame shows severe deviation from the provided image. There is significant distortion, missing key elements, or massive color shifts. The original image traits are barely identifiable. - 1: Complete Failure. The first or last frame is completely irrelevant to the provided image, or the video fails to anchor to the first frame or the last frame entirely.

- 2. The second score: Instruction Following Objective: Assess how well the generated video follows the natural language instruction to logically and semantically bridge the first and last frames.

- 5: Perfect Alignment. The transition logic from the first frame to the last frame in the generated video is flawless and highly intuitive. The video executes the text prompt perfectly. The bridge between the first and the last frames is seamless with no logical gaps. - 4: Good Alignment: The transition from the first frame to the last frame is logical and cohesive. The core semantics of the prompt are successfully rendered, but there may be minor imperfections in secondary motion details, pacing, or amplitude. - 3: Partial Alignment. The transition from the first frame to the last frame is somewhat disjointed. The model clearly attempts to follow the prompt to connect the keyframes, but exhibits noticeable logical leaps, introduces unwarranted hallucinatory actions, or ignores a portion of the prompt. - 2: Weak Alignment. The transition from the first frame to the last frame is abrupt or chaotic. Most of the text instructions are ignored. The connection between keyframes feels like an unnatural cross-fade (morphing without physical logic), or the semantic evolution strongly violates common sense. - 1: No Alignment: The video completely ignores the prompt. The intermediate generation deviates entirely from the text description, or the semantic logic is fundamentally broken and contradictory to the instruction.

- 3. The third score: Overall Visual Quality Objective: Evaluate the general aesthetic quality, temporal consistency (flicker-free), motion smoothness, and physical naturalness (absence of severe artifacts or morphing) of the generated video, independent of the input images or prompt.

- - 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. Motions are fluid and natural (no jitter or dropped frames), physical transformations obey real-world physics, and there is zero flickering or artifacts. - 4: Good. High overall quality. Smooth and coherent, with only minor, easily forgivable artifacts or very slight flickering occurring in complex motion areas or fine edges. Overall viewing experience remains pleasant.
- - 3: Fair. Acceptable but noticeably flawed. Visible visual anomalies are present, such as noticeable flickering, mild stuttering/jittering, moderate morphological distortions during movement, or obvious AI artifacts. - 2: Poor. Significantly degraded quality. Severe visual and temporal defects dominate the video, including high-frequency flickering, severe motion rigidity, grotesque ”melting” or distortion of objects during transitions, and heavy noise. - 1: Unacceptable. Completely collapsed visual integrity. The video lacks any temporal coherence. It is heavily corrupted with extreme artifacts, tearing, or noise, making it unwatchable and devoid of aesthetic value.

Example Response Format: You are required to return a dictionary structured as follows: {”Condition Preserving”: [A number from 1 to 5], ”Instruction Following”: [A number from 1 to 5], ”Overall Visual Quality”: [A number from 1 to 5]}.

<|user_input|>

Figure 17: Evaluation prompt template for Interpolative DI2V task.

Evaluation Prompt Template for Compositional MI2V Task with One Input Image)

You are an expert data rater specializing in evaluating subject-driven video generation. You will be given a reference image containing a specific subject, a prompt, and the generated video. The prompt requires the generated video to accurately feature the exact subject from the reference image while performing actions or interacting with the environment based on the provided text. Your task is to evaluate the generated video on a 5-point scale from three perspectives:

- 1. The first score: Condition Preserving Objective: Evaluate how accurately the generated video retains the identity of the subject from the reference image throughout the entire video duration.

- 5: Perfect Preservation. The subject’s identity are flawlessly maintained and highly stable. Regardless of camera angle or complex motion, the subject exactly matches the reference image with zero feature loss or morphological distortion. - 4: High Preservation. The subject is highly consistent. The identity is clearly recognizable, with easily overlooked losses of detail or slight warping occurring during complex trajectories. 3: Moderate Preservation. The subject is generally recognizable but exhibits noticeable flaws. Visible loss of fine detail, color shifts, or moderate morphological distortions (e.g., slight ”melting” or facial warping during movement) are present. - 2: Low Preservation. The subject suffers from severe identity drift. While there might be a vague resemblance initially, the subject undergoes massive distortion, grotesque AI morphing, or noticeably transforms into a different entity as the video progresses. - 1: Complete Failure. No relation to the reference. The subject in the generated video is completely irrelevant to the reference image, or the requested subject is entirely absent from the scene.

- 2. The second score: Instruction Following Objective: Assess how accurately the video executes the text prompt, such as the subject’s actions, the surrounding environment, object interactions, and camera movements.

- 5: Perfect Alignment. The video is a flawless semantic match to the text. It accurately executes the core actions and perfectly captures all described background settings, lighting constraints, secondary objects, and specific cinematography (e.g., panning, zooming), provided these elements are described in the prompt. - 4: Good Alignment. The core intent is successfully rendered. The subject performs the main instructed actions in the correct setting, but the video misses minor secondary details (e.g., a small prop). - 3: Partial Alignment. The execution is somewhat disjointed. The video captures the general concept but exhibits flawed action logic, fails to complete the motion, or entirely misses a significant environmental or action constraint mentioned in the prompt. - 2: Weak Alignment. The video severely deviates from the instruction. It only captures isolated keywords (e.g., the subject is present but doing the wrong thing), and the scenario or behavior strongly contradicts the prompt’s description. - 1: No Alignment. Complete ignorance of the prompt. Aside from potentially including the subject, the events, actions, or environments in the video have absolutely no relation to the text prompt, representing a total hallucination.

- 3. The third score: Overall Visual Quality Objective: Evaluate the general aesthetic quality, temporal consistency (flicker-free), motion smoothness, and physical naturalness of the generated video.

- 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. Motions are fluid and natural (no jitter or dropped frames), physical transformations obey real-world physics, and there is zero flickering or artifacts. - 4: Good. High overall quality. Smooth and coherent, with only minor, easily forgivable artifacts or very slight flickering occurring in complex motion areas or fine edges. Overall viewing experience remains pleasant. - 3: Fair. Acceptable but noticeably flawed. Visible visual anomalies are present, such as noticeable flickering, mild stuttering/jittering, moderate morphological distortions during movement, or obvious AI artifacts. - 2: Poor. Significantly degraded quality. Severe visual and temporal defects dominate the video, including high-frequency flickering, severe motion rigidity, grotesque ”melting” or distortion of objects during transitions, and heavy noise. - 1: Unacceptable. Completely collapsed visual integrity. The video lacks any temporal coherence. It is heavily corrupted with extreme artifacts, tearing, or noise, making it unwatchable and devoid of aesthetic value.

Example Response Format: You are required to return a dictionary structured as follows: {”Condition Preserving”: [A number from 1 to 5], ”Instruction Following”: [A number from 1 to 5], ”Overall Visual Quality”: [A number from 1 to 5]}.

<|user_input|>

Figure 18: Evaluation prompt template for Compositional MI2V task with one input image.

Evaluation Prompt Template for Compositional MI2V Task with Multiple Input Images

You are an expert data rater specializing in evaluating multi-subject-and-background-conditioned video generation. You will be given multiple reference images containing specific subjects, a reference background image, a prompt, and the generated video. The prompt requires the generated video to seamlessly integrate all exact subjects from the references into the specified background, while performing dynamic actions and interactions based on the provided text. Your task is to evaluate the video on a 5-point scale from three perspectives:

- 1. The first score: Condition Preserving Objective: Evaluate how accurately the generated video retains the identity of all subjects and the background from the reference images, maintaining their stability throughout the video without identity bleeding or missing subjects.

- 5: Perfect Preservation. All subjects’ identity are flawlessly maintained and highly stable. Regardless of camera angle or complex motion, each subject exactly matches the reference image with zero feature loss or morphological distortion. - 4: High Preservation. All subjects and background are highly consistent with those in the given images. There is only microscopic detail loss in subjects or minuscule blurring in extreme background areas during highly complex occlusions or sweeping camera movements. - 3: Moderate Preservation. Generally recognizable but flawed. No subjects are missing, and the background matches the reference broadly. However, visible issues exist: mild identity bleeding between subjects, or the background undergoes moderate morphological changes over time (e.g., specific background elements shifting, mutating, or disappearing). - 2: Low Preservation. The video omits a required subject, exhibits severe identity fusion, or the background significantly deviates from the provided reference (e.g., the structural layout collapses or morphs into a different environment entirely). - 1: Complete Failure. Irrelevant or severe omissions. The majority of subjects are missing, and the background bears no resemblance to the provided reference, with a total loss of input.

- 2. The second score: Instruction Following Objective: Assess how accurately the video executes the text prompt, focusing on the specific actions of each subject and the logical interactions between them within the background.

- 5: Perfect Alignment. The video is a flawless semantic match to the text. It accurately executes the core actions within the specific background. Also, the multi-subject interactions described in the prompt (e.g., conversing, holding hands, fighting) are rendered perfectly and logically. - 4: Good Alignment. The core intent is successfully rendered. Main interactions and actions are executed correctly within the background, but the video misses minor secondary behavioral details or subtle narrative elements. - 3: Partial Alignment. Execution is somewhat disjointed. The video captures the concept of the subjects being together, but the interactions are flawed or stiff (e.g., prompting ”hugging” but generating them just ”standing next to each other”), or a major action constraint is missed. - 2: Weak Alignment. The video severely deviates from the instruction. The model captures the presence of the subjects but completely fails to demonstrate the requested interaction or actions, with subjects acting independently and contrary to the prompt. - 1: No Alignment. Complete ignorance of the prompt. Aside from containing the subjects, the events, actions, or environments have absolutely no relation to the text prompt, representing a total AI hallucination.

- 3. The third score: Overall Visual Quality Objective: Evaluate the general aesthetic quality, temporal consistency (flicker-free), motion smoothness, and physical naturalness of the generated video.

- - 5: Excellent. Exceptional visual aesthetics and perfect temporal consistency. Motions are fluid and natural (no jitter or dropped frames), physical transformations obey real-world physics, and there is zero flickering or artifacts. - 4: Good. High overall quality. Smooth and coherent, with only minor, easily forgivable artifacts or very slight flickering occurring in complex motion areas or fine edges. Overall viewing experience remains pleasant.
- - 3: Fair. Acceptable but noticeably flawed. Visible visual anomalies are present, such as noticeable flickering, mild stuttering/jittering, moderate morphological distortions during movement, or obvious AI artifacts. - 2: Poor. Significantly degraded quality. Severe visual and temporal defects dominate the video, including high-frequency flickering, severe motion rigidity, grotesque ”melting” or distortion of objects during transitions, and heavy noise. - 1: Unacceptable. Completely collapsed visual integrity. The video lacks any temporal coherence. It is heavily corrupted with extreme artifacts, tearing, or noise, making it unwatchable and devoid of aesthetic value.

Example Response Format: You are required to return a dictionary structured as follows: {”Condition Preserving”: [A number from 1 to 5], ”Instruction Following”: [A number from 1 to 5], ”Overall Visual Quality”: [A number from 1 to 5]}.

<|user_input|>

Figure 19: Evaluation prompt template for Compositional MI2V task with multiple input images.

Evaluation Prompt Template for TIV2V Task (local object addition)

You are a data rater specializing in grading video object addition editing. You will be given two videos (before and after editing), an reference image and the corresponding editing instructions. The instructions describe how to add a specific subject from the reference image into a designated location within the video. Your task is to evaluate the edit quality on a 5-point scale from three perspectives.

The first score: Instruction Following Objective: Evaluates whether the edit correctly follows the instruction to add the subject from the reference image into the designated location, and whether the newly added object naturally integrates into the original scene rather than looking like a rigid insertion.

- - 1: Fail. No edit performed, the video is completely corrupted, or the edit is fundamentally wrong. - 2: Poor. Wrong object/class added, the target is only partially added, an unrelated object is added alongside it, or it is placed in a completely incorrect location. - 3: Fair. Correct object added to the general location, but with significant attribute errors (e.g., identity mismatch with the reference image). Integration check: The object feels rigidly or stiffly ”pasted” into the environment and does not logically or naturally fit the context of the original scene. - 4: Good. The correct object is added with main attributes and location correct; only minor details slightly mismatch the reference image. Integration check: The object integrates almost naturally into the original scene context, with only minor signs of artificial placement. - 5: Perfect. All and only the requested objects are added exactly as instructed. The object perfectly matches the reference image, is placed in the exact correct position, and is integrated so naturally that it looks as though it was always meant to be part of the scene. The second score: Condition Preserving Objective: Evaluates two critical preservation aspects: (A) How well the newly added object retains the identity and intricate details of the subject in the reference image, and (B) How perfectly the unedited regions, non-target objects, and the original temporal dynamics from the original video are preserved.
- - 1: Fail. The new object bears no resemblance to the reference image, OR the unedited regions/temporal dynamics are completely destroyed or frozen. - 2: Poor. The added object lacks key intricate details from the reference image, OR there are obvious, distracting alterations to the unedited regions in the video or their original temporal flow (e.g., previously moving unedited regions suddenly freeze). - 3: Fair. The added object retains its basic identity but loses some finer details from the reference. The unedited regions in the video are mostly preserved, but some spatial details or original temporal variations (e.g., altered camera shifts or smoothed-out motion in the unedited regions) are noticeably affected. - 4: Good. The added object matches the reference image closely with only minor detail discrepancies. The unedited regions in the video and their temporal dynamics are almost maintained, with only minor deviations visible. - 5: Perfect. Flawless preservation. The new object exactly matches the reference subject’s details, AND NO changes occurred in the unedited areas, e.g., the original background and its temporal changes/motion are perfectly intact. The third score: Overall Visual Quality Objective: Evaluates the physical realism, temporal stability, and visual harmony of the new object within the video. CRITICAL EXCEPTION: Do NOT deduct points for visual quality issues, noise, or artifacts that were already present in the original, unedited video.
- - 1: Fail. Severe new artifacts or physical errors (e.g., the added object floats, has completely wrong perspective/lighting). The added object causes severe flickering/jittering or blocks key original elements, breaking visual coherence. - 2: Poor. Obvious paste marks, severe style/resolution mismatches, or poor handling of physics (e.g., terrible occlusion, lack of contact with surfaces). The added region temporally jitters or remains static against a moving environment. - 3: Fair. The general style, lighting, and physics of the added object are fundamentally consistent with the scene, but noticeable flaws remain (e.g., slight temporal disharmony, mismatched shadows, awkward motion handling, or minor edge artifacts). - 4: Good. Style, lighting, shadows, and reflections of the new object are believable and move correctly with the scene. Minor physical mismatches or slight temporal instability (e.g., faint flicker) are visible. - 5: Perfect. Seamlessly integrated and physically flawless. The added object exhibits precise highlights, shadows, and motion effects that perfectly match the scene’s physics. It is temporally stable and visually indistinguishable from a real object interacting within that specific environment.

Example Response Format: You are required to return a dictionary structured as follows: {”Condition Preserving”: [A number from 1 to 5], ”Instruction Following”: [A number from 1 to 5], ”Overall Visual Quality”: [A number from 1 to 5]}.

<|user_input|>

Figure 20: Evaluation prompt template for TIV2V task (local object addition).

Evaluation Prompt Template for TIV2V task (background change)

You are a data rater specializing in grading video background editing. You will be given two videos (before and after editing), an reference image and the corresponding editing instructions. The instructions requires to replace the video background with the scene depicted in the reference image. Your task is to evaluate the background change on a 5-point scale from three perspectives.

The first score: Instruction Following Objective: Evaluates whether the background is correctly replaced and matches the target scene provided in the reference image.

- - 1: Fail. No background change at all, or the new background is entirely unrelated to the reference image. - 2: Poor. Background is only partially replaced, or exhibits major deviations in style, content, or layout compared to the reference image. - 3: Fair. The main background is replaced, but some key elements from the reference image are missing, extraneous, or placed incorrectly. - 4: Good. Requested background is fully present and largely consistent with the reference image in content and style, with only minor discrepancies. - 5: Perfect. Perfect compliance; the new background perfectly matches the scene shown in the reference image (content, style, and layout). The second score: Condition Preserving Objective: Evaluates whether the non-background elements (foreground subjects, objects) and their original motion are kept completely intact without distortion.
- - 1: Fail. Massive distortion, loss of foreground objects, or original motion is completely ruined/frozen. - 2: Poor. Significant alteration of foreground objects (e.g., missing body parts, severe structural artifacts) or noticeably altered/unnatural motion. - 3: Fair. Minor changes to the foreground objects (e.g., slight blurring of internal details, small localized artifacts) or slight jitter in their original motion. - 4: Good. Foreground objects and their motion are almost perfectly preserved, with only barely noticeable minute artifacts upon close inspection. - 5: Perfect. All non-background elements and their original motion remain exactly as they were in the source video. The third score: Overall Visual Quality Objective: Evaluates the synthesis of Visual & Temporal Seamlessness (Edge, Blend & Stability) and Physical Consistency (Lighting, Perspective, Motion & Depth) of the video after editing. CRITICAL EXCEPTION: Do NOT deduct points for visual quality issues, noise, or artifacts that were already present in the original, unedited video.
- - 1: Fail. Severe composite errors (large tearing, flickering, jittering edges) OR severe physical mismatch (conflicting light, floating subject, wrong horizon, no parallax during camera motion). Obvious fake at a glance.
- - 2: Poor. Clear cut-out halos, obvious edge ’boiling’, or noticeable inconsistencies in lighting, scale, and perspective shifts during motion. - 3: Fair. Acceptable overall but visible flaws on closer inspection: slight edge blur, minor temporal shimmer, or small errors in shadows, depth, and perspective tracking. - 4: Good. Nearly invisible seams; edges are stable, lighting/scale/depth are well-matched, and perspective tracks convincingly with camera motion. Only minor issues visible. - 5: Perfect. Flawless composite and physical realism. Edges, light, shadows, perspective, and atmospheric depth are perfectly coherent and temporally stable throughout the video.

Example Response Format: You are required to return a dictionary structured as follows: {”Condition Preserving”: [A number from 1 to 5], ”Instruction Following”: [A number from 1 to 5], ”Overall Visual Quality”: [A number from 1 to 5]}.

<|user_input|>

Figure 21: Evaluation prompt template for TIV2V task (background change).

Evaluation Prompt Template for TIV2V Task (local object replacement)

You are an expert data rater specializing in grading video object replacement edits. You will be provided with an original video, a reference image, the edited video, and the corresponding editing instructions. The instructions dictate replacing a specific object in the original video with a subject from the reference image. Your task is to evaluate the editing performance on a 5-point scale across three key dimensions, paying close attention to temporal consistency.

The first score: Instruction Following Objective: Evaluates whether the model successfully executed the editing instruction (e.g., replacing the correct target object with the specified new object) for the correct duration and at the correct position, and whether the newly added object naturally integrates into the original scene rather than looking like a rigid insertion.

- - 1: Fail. The target was not replaced, or a completely unrelated object/region of the video was edited. - 2: Poor. Attempted replacement, but the instruction was poorly executed (e.g., replacement only happens in a few frames, or the edit occurs in the wrong spatial location). - 3: Fair. The basic instruction was followed (target replaced), but with noticeable errors in execution (e.g., incorrect count, wrong pose/scale, or failure to maintain the replacement consistently across all frames), or the new object feels stiffly, abruptly, or unnaturally forced into the scene. - 4: Good. The instruction was followed correctly for the entire duration. The correct target was replaced with accurate placement and scale, with only minor deviations. And it integrates almost naturally into the original scene’s context, with only minor stiffness or slight contextual mismatch. - 5: Perfect. Flawless execution. All specified objects were replaced perfectly. The new object fits so naturally into the scene’s context that it feels exactly as if it originally belonged there, fulfilling the prompt perfectly. The second score: Condition Preserving Objective: Evaluates two critical preservation aspects: (A) How well the newly added object retains the identity and intricate details of the subject in the reference image, and (B) How perfectly the unedited regions, non-target objects, and the original temporal dynamics (e.g., original background motion, natural evolving lighting, camera shifts) from the original video are preserved.
- - 1: Fail. The new object bears no resemblance to the reference image, OR the unedited regions/temporal dynamics are completely destroyed or frozen. - 2: Poor. Major identity loss in the new object, OR significant warping, distortion, or loss of original temporal motion in the unedited regions. - 3: Fair. The new object captures the general idea but has obvious attribute discrepancies, OR there are visible, distracting changes to the unedited regions, such as background elements losing their original natural movement or temporal flow. - 4: Good. The new object closely matches the reference image with only minor attribute errors, AND the unedited regions almost preserve their original appearance and temporal dynamics, with minor changes. - 5: Perfect. Flawless preservation. The new object exactly matches the reference subject’s details, AND absolutely zero changes occurred in the unedited areas, e.g., the original background and its temporal changes/motion are perfectly intact. The third score: Overall Visual Quality Objective: A comprehensive evaluation of Visual Naturalness, Temporal Stability, Physical/Motion Integrity, and how seamlessly the new object blends into the scene (lighting, shadows, perspective). CRITICAL EXCEPTION: Do NOT deduct points for visual quality issues, noise, or artifacts that were already present in the original, unedited video.
- - 1: Fail. Severe visual degradation introduced by the edit. Video heavily broken or the new object is severely deformed, flickers uncontrollably, or floats/slides unnaturally (complete motion tracking failure). - 2: Poor. Obvious artifacts introduced. Noticeable flickering seams, missing/static shadows, poor occlusion, or the new object’s motion clearly disconnects from the camera/scene movement. Fails to blend into the environment. - 3: Fair. Acceptable quality but with visible temporal or physical inconsistencies introduced by the edit. Minor flickering, edge fuzzying, lighting shifts over time, or small tolerable drifts in motion tracking. Integration is passable but imperfect. - 4: Good. High visual quality and stable style. The new object’s motion is well-tracked, interacts realistically with the scene (accurate shadows/reflections), and naturally blends in. Only Tiny temporal artifacts are visible. - 5: Perfect. Completely seamless, temporally stable, and dynamically flawless. Perfect motion tracking, perspective, and lighting integration. The new object blends perfectly into the scene in every frame, making the edit completely undetectable to a casual viewer.

Example Response Format: You are required to return a dictionary structured as follows: {”Condition Preserving”: [A number from 1 to 5], ”Instruction Following”: [A number from 1 to 5], ”Overall Visual Quality”: [A number from 1 to 5]}.

<|user_input|>

Figure 22: Evaluation prompt template for TIV2V task (local object replacement).

