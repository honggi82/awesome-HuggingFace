# AnimeShooter: A Multi-Shot Animation Dataset for Reference-Guided Video Generation

### Lu Qiu1, Yizhuo Li1, Yuying Ge†,2, Yixiao Ge2, Ying Shan2, Xihui Liu†,1

1 The University of Hong Kong 2 ARC Lab, Tencent PCG † Corresponding authors Project Page: https://qiulu66.github.io/animeshooter/

arXiv:2506.03126v1[cs.CV]3Jun2025

###### Story-level Annotation Shot-level Annotation

[Figure 1]

Shot-1 00:00-00:05

[Figure 2]

###### Sample Info

Scene: Living Room Main Characters: [Bride, Groom]

Video ID: … Frame Index: [1299, 1692]

###### Visual Captions

[Figure 3]

[Figure 4]

###### Storyline

Narrative Caption: The bride confronts the groom… Descriptive Caption: The bride and the dog enter…

A newly married couple discovers that they are actually made of clay…

###### Audio Descriptions

[Figure 5]

###### Main Characters

Description: High-pitched, rapid barks… Visual Source: The dog is barking, panting…

[Figure 6]

[Figure 7]

[Figure 8]

Bride - A clay figure with red hair, wearing a white wedding dress…

Groom - A slender clay figure with dark hair, wearing a dark suit…

[Figure 9]

Shot-2 00:05-00:12

[Figure 10]

Scene: Dining Room Main Characters: [Bride]

###### Main Scenes

Living Room - An interior room with tan walls, and a large light tan sofa…

###### Visual Captions

[Figure 11]

Narrative Caption: The excited dog barks… Descriptive Caption: Bride enters a dining room…

Dining Room – It appears in warm colors, with kitchen utensils and other objects on the table…

###### Audio Descriptions

[Figure 12]

[Figure 13]

Description: Short, low-pitched, breathy sighs… Visual Source: The bride sighs and moves…

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Shot-3 00:12-00:16

Scene: Living Room Main Characters: [Bride, Groom]

###### Visual Captions

[Figure 18]

Bride

[Figure 19]

Narrative Caption: The groom sits on the sofa… Descriptive Caption: The groom with a dark suit…

[Figure 20]

###### Audio Descriptions

Description: Faint music and indistinct dialogue… Visual Source: The television playing a show …

[Figure 21]

[Figure 22]

[Figure 23]

Groom

###### ……

- Figure 1: Overview of AnimeShooter. It is a reference-guided multi-shot animation dataset featuring comprehensive hierarchical annotations and strong coherence across shots. At the story level, each sample includes an overall storyline, main scene descriptions, and detailed character profiles with reference images. At the shot level, consecutive shots are annotated with specific scenes, involved characters, and rich visual captions (both narrative and descriptive). A specific subset, AnimeShooter-audio, additionally provides synchronized audios for each shot with corresponding audio descriptions and sound sources.

*The work was done during the author’s internship at ARC Lab, Tencent PCG.

Preprint. Under review.

## Abstract

Recent advances in AI-generated content (AIGC) have significantly accelerated animation production. To produce engaging animations, it is essential to generate coherent multi-shot video clips with narrative scripts and character references. However, existing public datasets primarily focus on real-world scenarios with global descriptions, and lack reference images for consistent character guidance. To bridge this gap, we present AnimeShooter, a reference-guided multi-shot animation dataset. AnimeShooter features comprehensive hierarchical annotations and strong visual consistency across shots through an automated pipeline. Storylevel annotations provide an overview of the narrative, including the storyline, key scenes, and main character profiles with reference images, while shot-level annotations decompose the story into consecutive shots, each annotated with scene, characters, and both narrative and descriptive visual captions. Additionally, a dedicated subset, AnimeShooter-audio, offers synchronized audio tracks for each shot, along with audio descriptions and sound sources. To demonstrate the effectiveness of AnimeShooter and establish a baseline for the reference-guided multi-shot video generation task, we introduce AnimeShooterGen, which leverages Multimodal Large Language Models (MLLMs) and video diffusion models. The reference image and previously generated shots are first processed by MLLM to produce representations aware of both reference and context, which are then used as the condition for the diffusion model to decode the subsequent shot. Experimental results show that the model trained on AnimeShooter achieves superior cross-shot visual consistency and adherence to reference visual guidance, which highlight the value of our dataset for coherent animated video generation.

## 1 Introduction

The animation industry plays a pivotal role in modern entertainment and education [42]. Recent advances in AI-generated content (AIGC) have revolutionized animation production through automated creation of complex visual narratives. Professional animation workflows necessitate the generation of coherent multi-shot video sequences that maintain visual consistency and adhere to predefined character designs. This reveals a substantial gap stemming from three fundamental limitations in existing public video datasets [2, 47, 4, 40, 25, 46, 39]: (1) focus on real-world scenario with easily obtainable web video content, (2) reliance on global captions inadequate for multi-shot narration, and (3) absence of reference images essential for consistent character guidance across sequential shots.

In this paper, we present AnimeShooter, a reference-guided multi-shot animation dataset featuring comprehensive hierarchical annotations and strong visual consistency across consecutive shots. Storylevel annotations define an overall storyline, main scene descriptions, and detailed character profiles with reference images. The entire story is then decomposed into ordered consecutive shots. For each shot, the shot-level annotation specifies scene, involved characters, and detailed visual captions in both narrative and descriptive forms. AnimeShooter-audio is a subset which offers additional annotations of synchronized audio for each shot, along with audio descriptions and sound sources. The dataset is constructed using an automated curation pipeline as shown in Figure 2: we first collect and filter a diverse range of large-scale animation films sourced from YouTube, then utilize Gemini [8] to generate hierarchical story scripts comprising story-level and shot-level annotations. Character reference images are extracted by sampling keyframes, segmenting characters with Sa2VA [50] which is prompted by character ID/appearance, and ensuring quality with InternVL [6] filtering.

To demonstrate the efficacy of AnimeShooter and establish a baseline model for this challenging task, we propose AnimeShooterGen, a reference-guided multi-shot video generation model based on MLLM and diffusion model. It can generate consecutive shots in an autoregressive manner. At each generation step, both the reference image and preceding video shots are encoded by the MLLM to produce representations that simultaneously capture character identity features and visual context. We design a multi-stage training strategy to bridge the real-to-animation domain gap and achieve autoregressive multi-shot video generation. Experiments on a custom evaluation dataset comprising multiple Intellectual Properties (IPs) and extensive evaluations demonstrate that models trained on AnimeShooter effectively learn cross-shot visual consistency and adhere to specified references.

To the best of our knowledge, this is the first reference-guided multi-shot animation dataset. Through large-scale multi-shots with visual consistency, accurate reference images for character identity, and comprehensive story and shot-level annotations, we hope AnimeShooter will facilitate research and development in narrative animation generation.

## 2 Related Work

Video-Text Datasets. Existing text-to-video datasets present notable limitations for multi-shot animation generation. The widely adopted WebVid-10M [2] relies on readily available online video titles and primarily comprises short video clips. While dense-captioning datasets such as Vript [47], ActivityNet [3], Panda-70M [4], and InternVid [40] offer temporally segmented clips with localized descriptions, structurally akin to multi-shot annotation via timestamp-caption pairs. However, they exhibit critical narrative deficiencies, failing to maintain coherent plot progression and suffering from temporal fragmentation with abrupt gaps or redundant overlaps. In the animation domain specifically, while efforts like AnimeCeleb [26], Sakuga-42M [35], and AniSora [23] aim to build animation datasets, they are typically restricted to single-shot content, focus narrowly on character heads, or are not publicly available, thereby limiting their utility for multi-shot animation generation.

Video Customization Datasets. Video customization techniques facilitate the synthesis of videos centered on specific concepts, such as individuals, pets, or objects. A critical requirement for storytelling and animation generation is multi-shot video customization: the ability to synthesize a sequence of shots that maintain the consistent appearance of a predefined character. ID-Animator [18] focuses on human face synthesis, utilizing facial regions as reference images. The VideoBooth dataset [24], derived from WebVid [2], augments textual prompts with image prompts generated by segmenting subjects from initial video frames via Grounded-SAM [31, 27]. Many other related efforts in video customization [5, 21, 9, 30] with similar data construction pipelines are predominantly address single-shot synthesis, and target non-animation applications.

Multi-Shot Storytelling and Animation Generation. Generating multi-shot videos for storytelling and animation often follows a staged pipeline. Anim-Director [29] uses image generators for reference designs, which then guide keyframe generation and subsequent I2V animation. This pipeline is shared by works like VideoStudio [33] and DynamiCrafter [45]. Except injecting reference images via multi-modal cross-attention (e.g., Anystory [17], VideoStudio [33]), some works attempt to maintain character appearance by optimization strategies (e.g., TaleCrafter [11], DreamRunner [41]). For example, MovieAgent [43] leverages an LLM [12, 15] for script/layout generation with ROICtrl [14] and ED-LoRA [13] for character injection. Despite these advancements, the dominant per-shot generation paradigm inherently struggles with cross-shot consistency. Recent work on Long Context Tuning (LCT) [16] validates that autoregressive architectures can achieve enhanced holistic visual appearance and temporal coherence by recursively conditioning each shot on preceding visual contexts. But it also addresses real-world domains and faces limitations such as prohibitive training costs and a lack of explicit architectural mechanisms for reference image conditioning.

## 3 AnimeShooter Dataset

This section describes the generation of AnimeShooter’s structured multi-shot story script and corresponding reference images. The construciton pipeline is shown in Figure 2. Please refer to supplementary files for the construction of additional subset AnimeShooter-audio.

### 3.1 Data Collection and Filtering

Our dataset collection begins by sourcing large-scale, diverse animated content from YouTube using keywords (e.g., "short animation", "cartoon short film"). To ensure content relevance and minimize visual-linguistic interference, we first filter these results. 16 uniformly sampled frames from each video are analyzed using InternVL [6] to exclude non-animated materials (such as tutorials or film reviews) and videos containing embedded subtitles. Prolonged animated content often exhibits temporal variations in character appearance, while intricate storyline with multiple characters create cognitive overload. To mitigate these issues, we implement a duration-based filtering protocol to preserve character consistency and reduce narrative complexity. Videos exceeding 20 minutes are removed firstly. The remaining videos are then cut into segments with around 1-minute durations

- (a) Data Collection and Filtering
- (b) Multi-Shot Captioning and Reference Image Generation

- • Wrong Character
- • Incomplete
- • Exaggerate
- • Fragmented
- • Too Small / Large

Storyline: Brian is taking an exam and struggles to answer…

[Figure 24]

[Figure 25]

Main Characters: ID: Brian Appearance: Brian is a cartoon character with brown hair…

[Figure 26]

ID: Brain Appearance: The brain is a pink, cartoonish creature…

[Figure 27]

- 00:01

00:00

- 00:02

[Figure 28]

Raw Data Filter & Split

[Figure 29]

Large Scale

Animation Duration

Diversity

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

1-min Segment (One Story)

[Figure 36]

- 00:03

###### Main Scenes:

Classroom: The classroom has desks and chairs… Playground: It is a brightly colored outdoor space…

[Figure 37]

Segment & Filter

###### -------------------------------------------------------------------------------

- Shot-1 00:00-00:04: Scene: null Main Characters: [] Caption: The screen is completely black for four seconds…
- Shot-2 00:04-00:06: ……

[Figure 38]

[Figure 39]

[Figure 40]

Sampled Frames Story Script by Gemini

- Figure 2: Video collection and annotation pipeline. We curate relevant videos from YouTube and segment them into 1-minute segments using boundary detection. Each segment serves as an individual sample representing a self-contained narrative unit (one story). We use Gemini to further decompose the story into consecutive shots with visual consistency based on transitions, and generate structured story script. Corresponding reference images are generated by Sa2VA and InternVL.

using PySceneDetect [1] algorithm to ensure coherent segment boundaries and narrative continuity. Each segment serves as an individual sample representing a self-contained narrative unit (one story).

### 3.2 Multi-Shot Captioning

To maintain the narrative cohesion and avoid referential ambiguity, we design a top-down multi-shot captioning strategy with three systematic phases: (1) Story-level annotation. This phase establishes a global narrative context by summarizing a succinct, coherent storyline and identifying 1-3 main characters and scenes, including detailed descriptions of their appearance and environment. (2) Shot decomposition. The entire story is subsequently decomposed into consecutive, non-overlapping shots delineated by shot transitions. (3) Shot-level annotation. For each shot, annotations identify the scene and characters, alongside two caption types: a narrative caption articulating plot progression (e.g., “The girl said goodbye to the bear”) and a descriptive caption conveying visual details (e.g., “A girl in red standing in front of a brown bear”). We utilize Gemini-2.0-flash [8] to process 1-minute segments and generate the hierarchical story script through these three phases above.

### 3.3 Reference Image Generation

Directly extracting frames containing a specific character is an intuitive but often inadequate strategy for obtaining reference images from animated films. The frequent co-occurrence of multiple characters and the presence of complex backgrounds can significantly hinder accurate character identification and isolation. We implements a robust model-assisted segmentation and filtering workflow. The process commences by leveraging pre-extracted story scripts to retrieve all related shots. From these shots, frames are sampled at 1 fps. Candidate frames are then fed into Sa2VA [50], which generates initial segmentation masks based on character IDs and appearance descriptions provided as text prompts. These raw masks are refined by morphological operations to fill holes and smooth contours, contour analysis to discard masks exhibiting excessive disconnected regions, and size filtering to exclude masks that occupy less than 5% or more than 90% area of the image.

Table 1: Statistics of AnimeShooter. “Num.” for number, “Dur.” for duration, “Chars.” for characters.

Statistics Level Total Num. Avg. Dur.(s) Avg. Caption(w) Avg. Chars. Avg. Scenes

Video-level 29K 286.57 - - Story-level 148K 56.72 33.55 2.26 2.20 Shot-level 2.2M 3.85 41.42 - -

To guarantee the final quality of reference images, InternVL [6] performs a secondary verification, enforcing structural completeness of the segmented character, semantic coherence between the segmented region and the provided ID/appearance prompts, consistent appearance across instances relative to the character’s textual description, avoidance of frames with extreme poses or expressions, and maintenance of high image resolution without motion blur.

### 3.4 Dataset Statistics

To ensure annotation fidelity within automated pipeline, we integrate human verification checkpoints on a small subset, validating the accuracy of story scripts and reference images. The statistical overview of the AnimeShooter dataset is presented in Table 1. The dataset contains 29K videos, each with an average duration of 286.57 seconds. Videos are typically divided into 5.07 segments. Each segment is approximately one minute long and serves as an individual sample representing one story. These story units average 56.72 seconds and feature an average of 2.26 main characters, 2.2 main scenes, and 14.82 shots. Each shot averages 3.85 seconds and is enriched with both a 10.62-word narrative caption and a 30.8-word descriptive caption, summing to 41.42 words.

## 4 Method

To validate the utility of AnimeShooter and establish a baseline model for animation generation, we introduce AnimeShooterGen. Inspired by prior works in world model [44, 20], AnimeShooter operates in an autoregressive fashion for reference-guided multi-shot video generation.

### 4.1 Model Design of AnimeShooterGen

Given the character reference image Iref, the previous context of the story and a natural language caption for the current shot, AnimeShooterGen predicts the current i-th video shot, denoted as Si.

- Figure 3 gives an overview of the model architecture. The model has two core components: the autoregressive backbone stemming from a pretrained MLLM [32] and a video generator based on pretrained DiT [36, 19]. An adapter (Q-Former [28]) is added to stitch these two components. For the generation of Si, the MLLM backbone fMLLM first processes a set of inputs: a reference image Iref provided by user, the accumulated previous context C<i, and the textual caption Ti for the current shot. The previous context C<i encapsulates the long-term memory from preceding shots and is composed of a visual context V<i and a textual context T<i:

V<i = {Fj,end | j = 1,...,i − 1} (1) T<i = {Tj | j = 1,...,i − 1} (2)

where Fj,end represents the last frame of the previously generated j-th shot Sj, and Tj is its corresponding caption. We set a sequence of learnable queries as the input of MLLM, and generate a

conditioning signal Condi:

Condi = fMLLM(Iref,C<i,Ti) (3) This Condi effectively combines character visual cues from Iref, long-term memory from C<i, and current textual guidance from Ti. Subsequently, the video generator synthesizes the current shot Si, and the training objective can be formulated as follows:

0∼pdata,ϵ∼N(0,I) ∥ϵ − ϵθ(xt,Condi)∥22 (4)

Et,x

min

θ

During the inference stage, upon the successful generation of shot Si, its last frame Fi,end and its caption Ti are incorporated into the previous context to form C<i+1 = (V<i ∪ {Fi,end},T<i ∪ {Ti}), which is then used for generating the subsequent shot Si+1. This autoregressive update mechanism allows the model to maintain coherence and narrative flow across multiple shots.

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

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

LoRA

LoRA

[Figure 58]

[Figure 59]

[Figure 60]

Diffusion

Diffusion

[Figure 61]

[Figure 62]

Adapter shot-2 (generated video)

Adapter shot-1 (generated video)

[Figure 63]

reference image (provided by user)

[Figure 64]

LoRA

Multimodal Large Language Model

[Figure 65]

……

[Figure 66]

[Figure 67]

reference image caption-1 learnable queries last frame-1 caption-2 learnable queries

last frame-2

- Figure 3: Overview of the model architecture. The two core components include the autoregressive backbone stemming from pretrained MLLM, and a video generator initialized from a pretrained DiT. To stitch these two components, we add a Q-Former as the adapter. This framework can generate multi-shot video in autoregressive manner.

### 4.2 Staged Training

The training of AnimeShooterGen is conducted in a multi-stage fashion. The detailed training strategies and implementation details can be found in supplementary files.

Condition Alignment: The initial stage focuses on aligning the MLLM’s output conditioning signal with the text encoder of the pretrained diffusion model. MLLM processes the first frame of a ground truth video clip and corresponding caption to generate an MLLM condition. We then minimize the MSE loss between this MLLM condition and the embedding of the caption extracted from the diffusion model’s text encoder. In this stage, only the adapter and learnable queries are trainable.

Single-Shot Training: This stage aims to bridge the real-to-animation domain gap, and train MLLM to extract character visual attributes from the reference image. MLLM receives Iref and Ti as input, and is optimized to produce an effective condition for generating the target shot Si. In this stage, LoRA weights of MLLM, the adapter, and learnable queries are trainable.

Multi-Shot Training: To foster consistency across multiple shots in terms of visual appearance, style, and color palettes, this stage extends the training to sequences. MLLM now processes the reference image Iref, the current caption Ti, and the previous context C<i. LoRA weights of MLLM, the adapter, and learnable queries are trainable.

LoRA Enhancement: In all preceding stages, the core diffusion model remains frozen. To further enhance the character and style consistency provided by MLLM and refine overall video quality, this final stage involves test-time finetuning. Given a few video clips from a particular IP, we freeze all other model components and exclusively train LoRA weights added to the diffusion model.

### 4.3 Integration of Audio Generation Capabilities

To further augment the immersive quality, we integrate AnimeShooterGen with Text-to-Audio (TTA) model TangoFlux [22]. The corresponding audio prompts are generated by GPT4o [34]. These prompts guide TangoFlux in synthesizing audios, which are then merged with the video sequences. Audio generation remains outside the scope of this paper, please refer to supplementary for details.

## 5 Experiments

### 5.1 Experiment Setting

Baselines: We compare two mainstream methods in storytelling field. The first approach employs a short video generation model capable of IP customization to produce individual video shots. To ensure a fair comparison and highlight the benefits of our multi-shot framework, we finetune the same pretrained diffusion model, CogVideo-2B [19], on the same IP-specific dataset. The second approach first generates a series of IP-consistent keyframes, which are then transformed into video

Table 2: Quantitative comparisons of automatic metrics.

Shot-level Story-level

Model Metric

Shot-1 Shot-2 Shot-3 Shot-4 Mean HarMeanP IP-Adapter + I2V

0.8004 0.7814 0.7891 0.7947 0.7914 0.5901 Cogvideo-LoRA 0.7297 0.7200 0.7417 0.7413 0.7332 0.5028

CLIP ↑

AnimeShooterGen 0.8022 0.7949 0.7970 0.7986 0.7982 0.6121 IP-Adapter + I2V

0.3679 0.4169 0.4047 0.3870 0.3941 0.6818 Cogvideo-LoRA 0.4777 0.5060 0.4842 0.4864 0.4886 0.7759

DreamSim ↓

AnimeShooterGen 0.3484 0.3820 0.3799 0.3764 0.3717 0.6413

Table 3: Quantitative comparisons of MLLM evaluation and user studies.

OQ ↑ CRC ↑ MSC ↑ MCC ↑ GPT Gem. Hum. GPT Gem. Hum. GPT Gem. Hum. GPT Gem. Hum.

Model

IP-Adapter + I2V 6.76 6.15 2.36 7.19 5.44 2.26 6.53 6.66 3.18 6.07 5.51 2.71 Cogvideo-LoRA 6.96 4.82 2.83 6.82 2.57 2.31 6.64 4.50 2.75 6.30 3.64 2.39

AnimeShooterGen 7.19 6.88 4.23 7.87 6.54 4.72 7.15 8.24 4.63 6.68 7.07 4.52

shots using an I2V model. Keyframe generation employs SDXL [37], augmented with IP-Adapter [49] to integrate reference image features, and CogVideo-5B is utilized for the I2V conversion.

Evaluation Dataset: We collect 20 animation films with distinct IPs. For each IP, we manually annotate 5-6 short clips for model fine-tuning. To evaluate multi-shot generation performance, we employ DeepSeek [15] to generate 10 unique narrative prompts per IP. Each prompt describes a story comprising 4 coherent shots. This process yielded a test set of 200 stories, totaling 800 video shots.

Metrics: Following prior works[43][7][48], we evaluate models using automatic metrics, advanced MLLM assessments and user studies. For automatic metrics, we employ CLIP score [38] and DreamSim [10] to quantify the consistency between generated characters and the reference image at shot-level and story-level. We also leverage GPT-4o and Gemini 2.5 Pro as MLLM-based judges, and conduct user studies to align with human preferences. The evaluation dimensions include Overall Quality <OQ>, Character-Reference Consistency <CRC>, Multi-Shot Style Consistency <MSC> and Multi-Shot Contextual Consistency <MCC>. More detailed information are in the supplementary.

### 5.2 Quantitative Comparisons

As shown in Table 2, automatic metrics evaluating character-reference alignment demonstrate that AnimeShooterGen outperforms both comparison methods. Notably, despite being trained on sequences of only 3 consecutive shots, AnimeShooterGen generalizes robustly to longer sequences during testing. Table 3 reveals additional advantages through MLLM evaluation and user studies. Beyond achieving superior CRC which also evaluates character-reference alignment, AnimeShooterGen exceeds comparison methods in MSC and MCC. Results underscore its dual strengths: (1) Enhanced reference image alignment. AnimeShooterGen achieves markedly higher character consistency than CogVideo-LoRA which shares the same diffusion architecture, proving that MLLM conditions effectively encode reference image features. (2) Cross-shot visual coherence. The MLLM’s memory mechanism retains historical context across shots, enabling high-level semantic alignment to guide the diffusion process in generating stylistically and contextually consistent new shots.

### 5.3 Qualitative Comparisons

- Figure 4 illustrates the visual outcomes of different methods on multi-shot animation generation task. The IP-Adapter + I2V approach struggles to maintain fidelity to the provided reference images due to weak control over IP-specific features. For instance, in the right-hand example, the generated character exhibits significant discrepancies in hairstyle, clothing, and facial structure compared to the reference image. CogVideo-LoRA also fails to achieve alignment with the reference images. Critically, both comparison methods generate individual shots as independent processes, leading to

- Shot-1: A wolf stands in a cluttered workshop, organizing tools on a table.
- Shot-2: He sits in an office, staring at a flickering computer screen.
- Shot-3: He walks through a quiet park, nodding at empty benches.
- Shot-4: He enters a small café, operating a café machine.

- Shot-1: A young girl stands at the edge of a frozen lake.
- Shot-2: She walks across the icy surface, slipping slightly.
- Shot-3: She kneels on the ice, looking down.
- Shot-4: She finds a cozy little cabin glowing warmly in the snow.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

- Shot-1
- Shot-2
- Shot-3
- Shot-4

AnimeShooterGen(Ours)IP-Adapter+I2VCogVideo-LoRA

[Figure 72]

[Figure 73]

- Shot-1
- Shot-2
- Shot-3
- Shot-4

[Figure 74]

[Figure 75]

- Shot-1
- Shot-2
- Shot-3
- Shot-4

#### Figure 4: Qualitatively comparisons on multi-shot animation generation. Our method delivers the best visual quality, including character-reference consistency and multi-shot consistency.

Reference image Shot-1 Shot-2 Shot-3 Shot-4

[Figure 76]

[Figure 77]

[Figure 78]

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

[Figure 93]

- Figure 5: Visualization of using different references in MLLM (before LoRA Enhancement). Shared caption: “Shot-1: The man walks down a cobblestone street lined with blooming cherry trees, holding a vintage leather journal under his arm.”, “Shot-2: He pauses at a flower shop, steps inside, and begins carefully selecting flowers.”, “Shot-3: At the counter, he wraps the bouquet in paper.”, “Shot-4: He tucks the flowers into his bicycle basket and pedals away past pastel-colored storefronts.”

glaring inconsistencies between shots. In the left-hand example, both comparison methods depict the wolf in its natural animal form in the third shot, with anthropomorphic representations generated in the remaining shots. In contrast, AnimeShooterGen achieves superior reference fidelity, and sustains cross-shot consistency in style and environmental elements, as demonstrated by the invariant morphology of snow-covered trees across multiple shots. It contribute to the autoregressive generation strategy, where previously generated shots directly condition subsequent ones. This mechanism ensures robust style uniformity and contextual coherence throughout the multi-shot sequence.

- 5.4 Investigating the Impact of Reference Images

To isolate and understand the direct influence of reference images on the generation process, we omit the LoRA enhancement phase. The model is conditioned on same captions paired with distinct reference images. As illustrated in Figure 5, the reference images inject coarse-grained visual cues into the MLLM condition, influencing both character appearance and artistic style: in rows 1 and 2, the generated characters adopt clothing colors and silhouettes that closely correspond to their respective reference images; the minimalist sketch style in row 3 directly mirrors the reference image’s aesthetic. Crucially, even in the absence of LoRA enhancement, the autoregressive nature of our framework maintains strong multi-shot consistency. This observation underscores the inherent capability of the autoregressive architecture to enforce shot-to-shot coherence.

- 6 Conclusion and Limitation

This paper introduces AnimeShooter, a comprehensive dataset for reference-guided multi-shot animation generation, featuring comprehensive hierarchical annotations and strong visual consistency across shots. Story-level annotations provide the storyline, key scenes, and main character profiles with reference images, while shot-level annotations decompose the story into consecutive shots, each annotated with scene, characters, and both narrative and descriptive visual captions. We also propose AnimeShooterGen which can generate reference-guided multi-shot animation in an autoregressive manner. Experiments demonstrate that being trained on AnimeShooter’s multi-shot annotations promotes cross-shot consistency and adherence to predefined references. Current limitations include the restriction of AnimeShooterGen from open-domain generation due to computational demands, the requirement for test-time fine-tuning to enhance character consistency, and suboptimal audiovisual synchronization resulting from a naive zero-shot audio generation approach. We anticipate AnimeShooter will serve as a valuable resource for future work aimed at developing more robust open-domain models with improved audio-visual alignment and character fidelity.

## References

- [1] Pyscenedetect.
- [2] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1728–1738, 2021.
- [3] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961–970, 2015.
- [4] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024.
- [5] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey Tulyakov. Multisubject open-set personalization in video generation. arXiv preprint arXiv:2501.06187, 2025.
- [6] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024.
- [7] Junhao Cheng, Yuying Ge, Yixiao Ge, Jing Liao, and Ying Shan. Animegamer: Infinite anime life simulation with next game state prediction. arXiv preprint arXiv:2504.01014, 2025.
- [8] Google DeepMind. Gemini, 2024.
- [9] Yufan Deng, Xun Guo, Yizhi Wang, Jacob Zhiyuan Fang, Angtian Wang, Shenghai Yuan, Yiding Yang, Bo Liu, Haibin Huang, and Chongyang Ma. Cinema: Coherent multi-subject video generation via mllm-based guidance. arXiv preprint arXiv:2503.10391, 2025.
- [10] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344, 2023.
- [11] Yuan Gong, Youxin Pang, Xiaodong Cun, Menghan Xia, Yingqing He, Haoxin Chen, Longyue Wang, Yong Zhang, Xintao Wang, Ying Shan, et al. Talecrafter: Interactive story visualization with multiple characters. arXiv preprint arXiv:2305.18247, 2023.
- [12] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [13] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models. Advances in Neural Information Processing Systems, 36:15890–15902, 2023.
- [14] Yuchao Gu, Yipin Zhou, Yunfan Ye, Yixin Nie, Licheng Yu, Pingchuan Ma, Kevin Qinghong Lin, and Mike Zheng Shou. Roictrl: Boosting instance control for visual generation. arXiv preprint arXiv:2411.17949, 2024.
- [15] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [16] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv preprint arXiv:2503.10589, 2025.

- [17] Junjie He, Yuxiang Tuo, Binghui Chen, Chongyang Zhong, Yifeng Geng, and Liefeng Bo. Anystory: Towards unified single and multiple subject personalization in text-to-image generation. arXiv preprint arXiv:2501.09503, 2025.
- [18] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024.
- [19] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.
- [20] Yuanhui Huang, Wenzhao Zheng, Yuan Gao, Xin Tao, Pengfei Wan, Di Zhang, Jie Zhou, and Jiwen Lu. Owl-1: Omni world model for consistent long video generation. arXiv preprint arXiv:2412.09600, 2024.
- [21] Yuzhou Huang, Ziyang Yuan, Quande Liu, Qiulin Wang, Xintao Wang, Ruimao Zhang, Pengfei Wan, Di Zhang, and Kun Gai. Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning. arXiv preprint arXiv:2501.04698, 2025.
- [22] Chia-Yu Hung, Navonil Majumder, Zhifeng Kong, Ambuj Mehrish, Amir Ali Bagherzadeh, Chuan Li, Rafael Valle, Bryan Catanzaro, and Soujanya Poria. Tangoflux: Super fast and faithful text to audio generation with flow matching and clap-ranked preference optimization. arXiv preprint arXiv:2412.21037, 2024.
- [23] Yudong Jiang, Baohan Xu, Siqian Yang, Mingyu Yin, Jing Liu, Chao Xu, Siqi Wang, Yidi Wu, Bingwen Zhu, Jixuan Xu, et al. Exploring the frontiers of animation video generation in the sora era: Method, dataset and benchmark. arXiv preprint arXiv:2412.10255, 2024.
- [24] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. Videobooth: Diffusion-based video generation with image prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6689–6700, 2024.
- [25] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions. Advances in Neural Information Processing Systems, 37:48955–48970, 2024.
- [26] Kangyeol Kim, Sunghyun Park, Jaeseong Lee, Sunghyo Chung, Junsoo Lee, and Jaegul Choo. Animeceleb: Large-scale animation celebheads dataset for head reenactment. In European Conference on Computer Vision, pages 414–430. Springer, 2022.
- [27] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023.
- [28] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.
- [29] Yunxin Li, Haoyuan Shi, Baotian Hu, Longyue Wang, Jiashun Zhu, Jinyi Xu, Zhen Zhao, and Min Zhang. Anim-director: A large multimodal model powered agent for controllable animation video generation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024.
- [30] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via cross-modal alignment. arXiv preprint arXiv:2502.11079, 2025.
- [31] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pages 38–55. Springer, 2024.

- [32] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al. Nvila: Efficient frontier visual language models. arXiv preprint arXiv:2412.04468, 2024.
- [33] Fuchen Long, Zhaofan Qiu, Ting Yao, and Tao Mei. Videostudio: Generating consistent-content and multi-scene videos. In European Conference on Computer Vision, pages 468–485. Springer, 2024.
- [34] OpenAI. Gpt-4o: Multimodal large language model, 2025.
- [35] Zhenglin Pan. Sakuga-42m dataset: Scaling up cartoon research. arXiv preprint arXiv:2405.07425, 2024.
- [36] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [37] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [39] Wenjing Wang, Huan Yang, Zixi Tuo, Huiguo He, Junchen Zhu, Jianlong Fu, and Jiaying Liu. Videofactory: Swap attention in spatiotemporal diffusions for text-to-video generation. 2023.
- [40] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023.
- [41] Zun Wang, Jialu Li, Han Lin, Jaehong Yoon, and Mohit Bansal. Dreamrunner: Fine-grained storytelling video generation with retrieval-augmented motion adaptation. arXiv preprint arXiv:2411.16657, 2024.
- [42] Paul Wells. Understanding animation. Routledge, 2013.
- [43] Weijia Wu, Zeyu Zhu, and Mike Zheng Shou. Automated movie generation via multi-agent cot planning. arXiv preprint arXiv:2503.07314, 2025.
- [44] Jiannan Xiang, Guangyi Liu, Yi Gu, Qiyue Gao, Yuting Ning, Yuheng Zha, Zeyu Feng, Tianhua Tao, Shibo Hao, Yemin Shi, et al. Pandora: Towards general world model with natural language actions and video states. arXiv preprint arXiv:2406.09455, 2024.
- [45] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pages 399–417. Springer, 2024.
- [46] Tianwei Xiong, Yuqing Wang, Daquan Zhou, Zhijie Lin, Jiashi Feng, and Xihui Liu. Lvd-2m: A long-take video dataset with temporally dense captions. arXiv preprint arXiv:2410.10816, 2024.
- [47] Dongjie Yang, Suyuan Huang, Chengqiang Lu, Xiaodong Han, Haoxin Zhang, Yan Gao, Yao Hu, and Hai Zhao. Vript: A video is worth thousands of words. Advances in Neural Information Processing Systems, 37:57240–57261, 2024.
- [48] Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen. Seed-story: Multimodal long story generation with large language model. arXiv preprint arXiv:2407.08683, 2024.
- [49] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

#### [50] Haobo Yuan, Xiangtai Li, Tao Zhang, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, and Ming-Hsuan Yang. Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos. arXiv preprint arXiv:2501.04001, 2025.

## A Details for Data Curation

### A.1 Multi-Shot Captioning

In Figure 6, we present the detailed prompt designed for multi-shot captioning (Section 3.2) with Gemini-2.0-flash. Initially, Gemini-2.0-flash is prompted to establish a global narrative context by summarizing a succinct, coherent storyline and identifying main characters and main scenes. Following this analysis, it decomposes the whole story into consecutive shots and provides detailed shot-level annotations.

USER: Given a video, your task is to succinctly summarize the storyline and hierarchically decompose it into shots according to the following guidelines:

- 1. Ensure that the storyline is succinct and coherent.
- 2. You should identify the main characters and the main scenes (where the story takes place). Provide detailed descriptions of the character's physical appearances, and the environment, style and color of the scenes.
- 3. The main character list should include only one to three important characters, those who appear repeatedly throughout the storyline. Characters who appear only a few times and are not important to the storyline should be omitted.
- 4. Decompose this video into non-overlapping shots based on transitions. For each shot, provide timestamps that cover the entire duration without skipping any part of the video. If a shot is longer than 4 seconds, in addition to the full shot duration, select a coherent and complete 4-second segment that best represents the corresponding plot and caption. Make sure all the start time and end time is connected between adjacent shots.
- 5. For each shot, provide a plot which focus on the action and story plot instead of visual descriptions. For example, "The girl said goodbye to the bear" is preferred over "A girl in red standing in front of a brown bear".
- 6. For each shot, also provide a detailed descriptive caption of the scene/environment, characters, and their actions or movements, even if these details have been mentioned in previous shots. For example, "By the bank of a river covered with trees, a girl in red stand in front of a brown bear, smiling and waving to him".
- 7. The story may contain prologue and epilogue, which are usually characterized by a pure background, the appearance of the story title, or a lot of text to indicate creation information. Please mark these meaningless prologue and epilogue that do not advance the plot in your response. The output format is JSON. Here is an example:

{

"storyline": ..., "main characters": [

{

"ID": //name of character1, "appearance": ...

},...

], "main scenes": [

{

"ID": //name of scene1, "environment": ...

},...

], "shots": [

{

"start time": "MM:SS", "end time": "MM:SS", "extra 4-second segment": "MM:SS-MM:SS", //optional, only if the shot is longer than 4 seconds "is_prologue_or_epilogue": prologue / epilogue, //optional, only if the shot is prologue or epilogue "main characters": [//list IDs of main characters who appear in the current shot] "scene": //write ID of the scene of the current shot "plot": //write the story plot "caption": //write the detailed descriptive caption

},... ]

} <VIDEO>

Figure 6: The prompt used for multi-shot captioning.

### A.2 Reference Image Generation

To address the challenge of ambiguous segmentation in images where multiple characters or nontarget individuals are present, we use the prompt template for Sa2VA (Figure 7, top subfigure) that explicitly incorporates character IDs and appearance descriptions from the shot-level character list, followed by a clear specification of the target ID for segmentation. For post-processing, we first apply morphological opening and closing operations with a kernel size of 5 to smooth boundaries and remove noise, and then eliminate masks containing over 15 contours or 5 disconnected components. Finally, we retain only the largest connected region, and discard masks occupying less than 5% or exceeding 90% of the total image area. The prompt shown in the bottom subfigure of Figure 7 is provided to InternVL for the secondary verification.

USER: <IMAGE> This image may contain characters as follows:

- 1. Girl, A young girl with short dark hair, wearing a pink bow.

- 2. Boy, A toddler with short dark hair, wearing a yellow overall.

- 3. Caregiver, An adult male with brown hair, a mustache, and glasses. Please segment Caregiver.

USER: You are tasked with evaluating whether an image can serve as a proper character reference image. Given an image and a character description, analyze the following criteria:

- 1. Character Completeness:

- - The character is segmented from video frames, the segmented character should be complete, not fragmented
- - The character's key features and details should be clearly visible

- 2. Description-Image Consistency:

- - Compare the character's appearance in the image with the provided description
- - Check if physical attributes (hair color, clothing, body type, etc.) match
- - Verify if distinctive features mentioned in the description are present
- - Note any inconsistencies or missing elements

- 3. Text-based Appearance Consistency:

- - Ensure the character's appearance is consistent throughout the description
- - Check if the description maintains the same physical attributes across different moments
- - Reject cases where the description shows inconsistency (e.g., wearing red clothes at one time and blue clothes at another)

- 4. Pose Evaluation:

- - The character should be in a neutral or natural standing pose
- - Avoid images with exaggerated expressions or dynamic action poses
- - The viewing angle should provide a clear understanding of the character

- 5. Image Quality:

- - The image should be clear and sharp, not blurry
- - Details should be easily distinguishable
- - Avoid images with motion blur or poor resolution

Remember: A good character reference image should serve as a clear, accurate representation of the character's standard appearance and design.

Based on these criteria, provide:

- - Verdict: yes/no on whether the image is suitable as a reference
- - Explanation: a brief explanation of your verdict

Here is the character description: <CHARACTER APPEARANCE>

Here is the image: <IMAGE>

Figure 7: The prompt used for reference image generation. Top subfigure: Segmentation prompt for Sa2VA. Bottom subfigure: Filtering prompt for InternVL.

### A.3 Audio Annotation for AnimeShooter-audio

Distinct from the visually-focused annotation in Section 3.2, this section aims to generate visualaudio annotations by addressing the inherent asynchrony between modalities (e.g., audio transitions lingering beyond visual shot cuts). To ground the model’s analysis and maintain consistency with pre-existing holistic information, thus avoiding redundant reference image generation, video segments alongside their story-level annotations are supplied to Gemini-2.5-Pro. Gemini-2.5-Pro is required to firstly exclude video segments with prolonged background music or human speech. Following this, the model executes shot decomposition and shot-level annotation. It performs a joint analysis of visual and auditory cues to detect optimal clip boundaries where both modalities exhibit coherent transitions. For each clip, it generates: (1) two types of visual captions, (2) audio descriptors (categorizing sound types and describing tones), and (3) source attribution (mapping sounds to visual elements). Notably, these clips do not strictly adhere to single visual shot boundaries, as decomposition is determined by the joint consideration of both visual and auditory transitions. For terminological consistency, these audio-visually coherent clips are still referred to as ’shots’. The prompt used for audio annotation is shown in Figure 8.

## B Details for Model Training

### B.1 Model Framework

AnimeShooterGen leverages NVILA-8B[32] as the pretrained MLLM backbone and CogVideo2B[19] as the pretrained video diffusion model. An adapter, crucial for inter-model communication, is implemented using a QFormer[28] with 12 layers. The length of learnable queries is set to 64. In AnimeShooterGen, for a sequence including n shots, MLLM receives an input formatted as "<Reference><Caption1><LearnableQuery><Frame1>...<Captionn><LearnableQuery>",

where each <Captioni> represents textual guidance for the i-th shot and <LearnableQuery> serves as a placeholder for contextual feature extraction. During training, the <Frame1> to <Framen−1> tokens are populated with the last frames from their corresponding ground truth video shots, and all n shots contribute jointly to the diffusion loss through backpropagation. At inference time, the model operates autoregressively: it first generates a 1-shot sequence using only the initial reference, then replaces the <Frame1> token with the final frame of the newly generated shot to condition the next iteration, recursively extending the sequence until reaching the target length n.

### B.2 Implementation Details

For Condition Alignment, we train the model using video-caption pairs from the large-scale WebVid10M dataset[2]. MLLM takes the first frame of a ground-truth video clip, its corresponding caption, and learnable queries as inputs. We minimize the MSE loss between the MLLM’s output condition and the T5 features of the caption. During this phase, only the adapter and learnable queries are trainable, optimized with a batch size of 32 for 1.8 M steps.

For Single-Shot Training, we utilize samples from the AnimeShooter dataset containing reference images and descriptive captions for individual shots. Here, the adapter, learnable queries, and LoRA parameters of the MLLM are fine-tuned with a batch size of 24 for 17K steps. Classifier-Free Guidance (CFG) is applied to enhance multimodal control, with independent dropout probabilities of 0.05 for the reference image, caption, or both modalities.

For Multi-Shot Training, we curate 3-shot sequences from the AnimeShooter dataset. The same trainable components (adapter, learnable queries, and MLLM LoRA) are updated with a batch size of 8 for 8K steps. A simplified CFG strategy is adopted, where both the reference image and caption inputs are simultaneously replaced with blank content at a 0.05 probability, eliminating modality-specific dropout.

For LoRA enhancement, given 5-6 separate video clips from a particular IP, these are randomly sampled and combined to form training sequences of 3 consecutive shots. This targeted finetuning is performed for 1K to 2K steps with batch size of 2.

Analyze the input animation video and perform visual-audio annotation following these steps: # Video Verification Ensure the video does not contain extended segments of background music (BGM) or human speech. The audio of this video should feature only sound effects (e.g., footsteps, alarms, animal sounds, environmental sounds).

# Shot Segmentation Split the video into non-overlapping multi-shot structure using combined visual+audio boundaries where:

- - Make sure all the start time and end time is connected between adjacent shots.
- - Each shot maintains continuous visual narrative and audio context.
- - Transition points must show simultaneous visual+audio changes.
- - Each shot lasts more than 3 seconds.
- - For each shot, provide start/end timestamps (MM:SS).

# Shot Annotation Based on the given video-level annotation (main characters, main scenes and storyline), you should create shot-level annotations for each shot:

- (1) List main characters present in this shot and the scene of this shot.
- (2) The video may contain prologue and epilogue, which are usually characterized by a pure background, the appearance of the story title, or a lot of text to indicate creation information. Please mark these meaningless prologue and epilogue that do not advance the plot in your response.
- (3) Visual annotation: For each shot, provide a plot which focus on the action and story plot instead of visual descriptions. For example, "The girl said goodbye to the bear" is preferred over "A girl in red standing in front of a brown bear". For each shot, also provide a detailed descriptive caption of the scene/environment, characters, and their actions or movements, even if these details have been mentioned in previous shots. For example, "By the bank of a river covered with trees, a girl in red stand in front of a brown bear, smiling and waving to him".
- (4) Audio annotation: For each shot, provide three description layers for only sound effects. The first is "type", it means sound effect category, e.g., alarm, magical plink, scampering. The second is "sound description", it means technical audio characteristics (pitch, intensity, temporal pattern, etc.). The third is "visual context", it means on-screen source/trigger of sound.

# Output Format Structure as hierarchical JSON. If the video contains extended segments of background music (BGM) or human speech, return empty JSON object. json {

"storyline": ..., "main characters": [

{

"ID": ..., "appearance": ...

},...

], "main scenes": [

{

"ID": ..., "environment": ...

},...

], "shots": [

{

"start time": "MM:SS", "end time": "MM:SS", "is_prologue_or_epilogue": prologue / epilogue, //optional, only if the shot is prologue or epilogue "main characters": [//list IDs of main characters who appear in the current shot], "scene": ..., //write ID of the scene of the current shot "visual annotation": {

"plot": ..., "caption": ...

}, "audio annotation": {

"type": ..., "sound description": ..., "visual context": ...

} },... ]

} # Now, please analyze the following case. <VIDEO> <VIDEO LEVEL ANNOTATION>

Figure 8: The prompt used for constructing AnimeShooter-audio.

## C Details for Experiments

### C.1 Baselines

For fair comparison with CogVideo-LoRA, we fine-tune the same pretrained diffusion model (CogVideo-2B) on the same IP-specific dataset and iterations as AnimeShooterGen. While AnimeShooterGen is trained in multi-shot mode (batch size=2 per step, with each sample containing 3 shots, totaling 6 shots per step), CogVideo-LoRA supports only single-shot training. To match the computational scale, we set its batch size to 6 (6 shots per step). For the training-free baseline IP-Adapter + I2V, we utilize stable-diffusion-xl-base-1.0 with its IP-Adapter to generate keyframes conditioned on reference images and captions. These keyframes and captions are then fed to the CogVideo-5B I2V model (replacing CogVideo-2B due to its lack of I2V capability) to synthesize video shots.

### C.2 Evaluation Dataset Construction

We construct a manually annotated evaluation dataset of 20 animated films featuring distinct IPs to support LoRA enhancement for AnimeShooterGen and finetuning for CogVideo-LoRA. For each IP, alongside a reference image, we curate 5–6 short video clips (each lasting several seconds) exclusively depicting the main character, ensuring maximal diversity in actions and scenes, as shown in Figure 9. To evaluate multi-shot generation performance, we employ DeepSeek with prompt shown in Figure 10 to generate 10 unique narrative prompts per IP.

Timestamps (s): 20 - 23 Caption: A young girl walked through a dense jungle, surrounded by trees and vegetation…

Timestamps (s): 27.5 – 29.5 Caption: A young girl stood amidst the dense jungle, intently studying the map in her hands…

Timestamps (s): 48 – 52 Caption: A young girl wandered through the brick ruins hidden deep within the jungle…

[Figure 94]

[Figure 95]

[Figure 96]

Timestamps (s): 75 - 77 Caption: A close-up portrait of a young girl standing amidst dense jungle foliage...

Timestamps (s): 140 - 149 Caption: At twilight, a young girl stumbles and falls onto the forest floor blanketed with leaves…

Timestamps (s): 242 - 244 Caption: A young girl stood before a castle crafted from sunlit yellow bricks, her figure turning around…

[Figure 97]

[Figure 98]

[Figure 99]

Figure 9: Example of IP-specific dataset for evaluation.

### C.3 Automatic Metrics

We employ CLIP score and DreamSim to quantify the visual similarity between generated characters and the reference image. We uniformly sample 5 frames from each shot and compute the average similarity score as the shot-level metric. To assess story-level consistency across 4 shots, we introduce two metrics: (1) Mean Similarity (Mean), the arithmetic mean of the 4 shot-level similarity scores. (2) Penalized Harmonic Mean Similarity (HarMeanP). Recognizing that even a single poorly generated shot can disrupt viewer immersion, this metric penalizes the worst-performing shot. This metric first calculates the harmonic mean of all 4 shots’ similarity scores (chosen for its sensitivity to extremely low values), then multiplies this result by the lowest similarity score as an additional penalty term.

Generate 10 prompts for evaluation cases. Each prompt must contain 4 consecutive scenes with simple captions following these rules:

- 1. Storytelling format with scene continuity but distinct background changes and different events.

- - Fixed character reference (it will be provided)
- - Only one character in all scenes

- 2. Keep captions simple.

- - Use basic verbs and no complex interactions.
- - Use simple language without complicated adjectives or descriptive phrases.
- - Without small objects or detailed props.

- 3. Follow this exact JSON format: {

"prompts": [ [

- "scene1 description",
- "scene2 description",
- "scene3 description",
- "scene4 description"

], // 9 more prompts

]

} Now, please generate prompts for a young girl.

Figure 10: The prompt used for constructing evaluation dataset.

### C.4 MLLM Assessment

We leverage GPT-4o and Gemini 2.5 Pro as MLLM-based judges. To mitigate ordering bias in evaluation, we employ the prompt template shown in Figure 11 and change the presentation order across three independent evaluation rounds. The final results reported in Section 5.2 represent the averaged metrics from these three trials, ensuring robustness against positional preferences. MLLM uses 1-10 scoring (1=worst, 10=best) with one-point increments. The evaluation dimensions including:

- • Overall quality <OQ>. A holistic assessment considering aesthetic appeal, image quality, visual consistency and so on.
- • Character-Reference Consistency <CRC>. Visual fidelity of the generated character to the provided reference image.
- • Multi-Shot Style Consistency <MSC>. Coherence of artistic style, color palette, and texture across all shots within a story.
- • Multi-Shot Contextual Consistency <MCC>. Continuity of the narrative and context across shots, e.g., ensuring the character maintains a consistent appearance appropriate to the unfolding story.

### C.5 Human Evaluation

For the human evaluation, we recruit 10 participants who hold at least a bachelor’s degree and have prior experience in image or video generation. A total of 15 stories with 60 shots are presented to the participants. The evaluation metrics align with those defined in Section C.4. Participants are instructed to score each dimension (1: lowest, 5: highest) based on the reference image, corresponding captions, and multi-shot videos generated by the three models. The final performance of each model is calculated as the average scores across all responses.

### C.6 Additional Qualitative Results

We present additional qualitative results in Figure 12 and Figure 13. Both comparison methods exhibit limitations in preserving character consistency with the reference image and cross-shot coherence. As illustrated in the right panel case of Figure 13, the IP-Adapter+I2V framework erroneously transforms the flashlight-equipped helmet into a color-mismatched hat. In the left panel case, the first shot

Please act as an impartial judge and evaluate three AI-generated multi-shot video samples (4 shots each) based on the following inputs and requirements.

# Input

- 1. Character reference image (provided first)
- 2. Text captions for each of the 4 shots (provided second)
- 3. Three models' outputs - Each model’s output is a 4-row grid, where: Each row corresponds to one shot (Shot 1 to Shot 4 from top to bottom). Within each row, 3 sampled frames are placed horizontally (left to right).

# Evaluation Dimensions Use 1-10 scoring (1=worst, 10=best) with one-point increments allowed:

- 1. Overall Quality <OQ> Conduct a comprehensive assessment of the overall quality for each model-generated sample, evaluating multiple dimensions such as visual consistency, aesthetic appeal, image quality, and other relevant factors.
- 2. Character-Reference Consistency <CRC> Evaluate visual fidelity and consistency between generated characters and the provided reference image.
- 3. Multi-Shot Style Consistency <MSC> Assess style, color palette and texture coherence across all shots.
- 4. Multi-Shot Contextual Consistency <MCC> Verify contextual continuity. E.g., evaluate whether the character maintain a stable appearance, behavior, and role throughout shots.

# Requirements & Output Format Avoid any bias, ensure that the order of presentation does not affect your decision. Do not favor certain agent names. Reflect the score differences as much as possible.

Return JSON format with detailed feedback: {

- "Model_A": { "scores": {

"OQ": 4, "CRC": 7, "MSC": 3, "MCC": 1

}, "feedback": {

"strengths": ["Consistent eye detail", "Good lighting continuity"], "weaknesses": ["Hand proportions vary in Shot 3", "Background mismatch in Shot 4"]

} },

- "Model_B": {...},
- "Model_C": {...}, "comparative_analysis": "Model A excels in character consistency and performs best...”

}

Figure 11: The prompt used for MLLM assessment.

generated by CogVideo-LoRA demonstrates a visually discordant art style. In contrast, our proposed AnimeShooterGen achieves superior preservation of character identity, color palette continuity, and stylistic consistency across generated sequences.

## D Integration of Audio Generation Capabilities for AnimeShooterGen

To further augment the immersive quality, we integrate AnimeShooterGen with zero-shot Text-toAudio (TTA) generation using TangoFlux [22]. The workflow involves processing the video captions and keyframes with GPT-4o to generate descriptive audio captions with the prompt shown in Figure 14. These audio captions subsequently guide TangoFlux in synthesizing audio tracks, which are then merged with the video sequences. However, results reveal substantial limitations in current simplistic zero-shot audio generation paradigms. Primarily, the decoupled generation processes for visual and auditory modalities result in inherent inter-modality synchronization failures. For example, footstep sounds lag behind walking animations, or character facial expressions mismatch

- Shot-1: A witch stands in a moonlit forest clearing, holding a staff.
- Shot-2: She stops where the forest ended. There is a massive stone castle.
- Shot-3: She sits by a stone fireplace. Flames burst to life.
- Shot-4: She stands on a tower's roof, with snowy mountains behind her.

- Shot-1: A yellow dog runs across a grassy field under a clear sky.
- Shot-2: It walks along a narrow dirt path surrounded by wildflowers.
- Shot-3: It naps in a patch of sunlight near a flowerbed.
- Shot-4: It lies down under a large tree at the edge of the field.

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

- Shot-1
- Shot-2
- Shot-3
- Shot-4

AnimeShooterGen(Ours)IP-Adapter+I2VCogVideo-LoRA

[Figure 104]

[Figure 105]

- Shot-1
- Shot-2
- Shot-3
- Shot-4

[Figure 106]

[Figure 107]

- Shot-1
- Shot-2
- Shot-3

- Shot-1: A man in a suit stands in an office, staring at a computer.
- Shot-2: He steps into a warehouse with stacked wooden crates.
- Shot-3: He leans against a concrete barrier on a highway overpass.
- Shot-4: In a park, he sits on a bench by the fountain to take a break.

Shot-1: A boy stands on a rooftop, looking at city lights. Shot-2: He walks down a narrow alley with graffiti walls. Shot-3: He pauses by a flickering streetlamp, glancing around. Shot-4: He sits on a park bench, resting his feet.

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

- Shot-1
- Shot-2
- Shot-3
- Shot-4

AnimeShooterGen(Ours)IP-Adapter+I2VCogVideo-LoRA

[Figure 112]

[Figure 113]

- Shot-1
- Shot-2
- Shot-3
- Shot-4

[Figure 114]

[Figure 115]

- Shot-1
- Shot-2
- Shot-3

with voice. Furthermore, constrained by existing text-to-audio models’ performance, environmental sound effects such as gentle breezes, engine roars, and mechanical hums fail to achieve sufficient perceptual distinctiveness, thereby compromising the immersive experience. We propose an audioannotated subset named AnimeShooter-audio, hoping to facilitate and encourage future research into the development of more sophisticated audio-visual co-generation models capable of achieving tighter synchronization and semantic coherence.

USER: Analyze the three provided animation frames and their corresponding text caption. Generate a descriptive audio prompt for a text-to-audio model to create corresponding ambient sounds.

# Requirements:

- - Prioritize sounds logically implied by the visuals and caption, e.g., dog barks if a dog is shown.
- - Specify the source of each sound explicitly, e.g., whirring of a coffee machine, rustling of leaves.
- - Sound effects may include: ambience (e.g., wind, city noise), action sounds (e.g., footsteps, door creaks)...
- - Avoid abstract metaphors, focus on concrete sounds a model can generate.

# Example Output: Only one phrase representing one type of sound. Directly output the audio prompt, no other text. For example: "a cow is mooning" "A stream flows"

# Generate the audio prompt now. <CAPTION> <FRAMES>

Figure 14: The prompt used for generating descriptive audio captions.

## E Potential Negative Social Impacts

While animation generation models democratize content creation, they risk enabling malicious applications such as generating deepfakes for disinformation, infringing intellectual property, or producing harmful content. To address these risks, the research community and policymakers must adopt proactive safeguards. Technical countermeasures include embedding watermarks in generated videos for provenance tracking, deploying AI-driven detectors to flag synthetic content, and implementing strict input/output filters to block unethical prompts.

## F Data Access and License

The AnimeShooter and AnimeShooter-audio datasets are released under the CC BY-NC 4.0 License and the data is collected from publicly accessible sources. These released datasets include annotated scripts, reference image masks, and corresponding video IDs, while the original source videos must be obtained independently from YouTube using the provided IDs. AnimeShooterGen is built upon two pretrained models: NVILA [32] and CogVideo [19]. We will release the code and model weights of AnimeShooterGen under the Apache 2.0 License, and provide a copy of the original licenses of NVILA and CogVideo in our GitHub repository.

To utilize the AnimeShooter and AnimeShooter-audio datasets, users must first download the source videos from YouTube via the specified video IDs or URLs using the yt-dlp tool. Subsequently, video segments should be extracted based on the provided start and end frame indices through ffmpeg processing. The extracted segments can then be analyzed using the accompanying story-level and shot-level annotations. Additionally, binary masks and corresponding frame indices are supplied for reference image generation. These reference images can be obtained by applying the provided masks to the cropped video frames through pixel-wise multiplication.

