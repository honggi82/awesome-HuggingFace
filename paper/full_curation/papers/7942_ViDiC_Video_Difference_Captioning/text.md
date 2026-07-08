# arXiv:2512.03405v3[cs.CV]24Mar2026

[Figure 1]

2026-03-25

[Figure 2]

## ViDiC: Video Difference Captioning

Jiangtao Wu1∗, Shihao Li1∗, Zhaozhou Bian1∗, Jialu Chen2, Runzhe Wen1, An Ping1, Yiwen He1, Jiakai Wang2, Yuanxing Zhang2,†, Jiaheng Liu1,†

1 NJU-LINK Team, Nanjing University 2 Kling Team, Kuaishou Technology jiangtaowu@smail.nju.edu.cn liujiaheng@nju.edu.cn

### Abstract

Understanding visual differences between dynamic scenes requires the comparative perception of compositional, spatial, and temporal changes—a capability that remains underexplored in existing vision-language systems. While prior work on Image Difference Captioning (IDC) has enabled models to describe semantic changes between static images, these approaches fail to capture motion continuity, event evolution, or video editing consistency over time. We introduce the ViDiC (Video Difference Captioning) task and its corresponding ViDiC-1K dataset, designed to evaluate the ability of Multimodal Large Language Models (MLLMs) to provide fine-grained descriptions of similarities and differences between video pairs. ViDiC-1K comprises 1,000 curated video pairs annotated with 3720 comparative checklist items, covering seven categories: subject, style, background, camera work, motion, position, and playback techniques. To ensure reliable evaluation, we propose a dual-checklist framework that measures the accuracy of similarity and difference separately, based on the LLM-as-a-Judge protocol. Experiments on 17 representative multimodal models reveal a significant performance gap in their comparative description and difference perception abilities. We hope ViDiC-1K can be a challenging benchmark that lays a solid foundation for advancing video understanding, edit awareness, and comparative reasoning in multimodal intelligence. The dataset is now open-source and available at https://huggingface.

co/datasets/NJU-LINK/ViDiC-1K.

- Video A
- Video B

- Video A
- Video B

Subject Change

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Are the tractors in the two videos different?

Does video B have one more boy than video A?

###### Camera Change

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

###### Style Change

###### Playback Technique Change

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Is the little girl’s clothing color different in the two videos? Is video A in an oil painting style while video B is not?

Are the people in the two videos wearing different clothes? Is video A played normally, while video B is reversed?

###### Motion Change

###### Position Change

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

[Figure 37]

[Figure 38]

Is the man playing the guitar different in the two videos? Is the man shown in a side view in video B, but not in video A?

Background Change

[Figure 39]

[Figure 40]

[Figure 41]

- Video A
- Video B

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Caption Generation

[Figure 46]

- Video A
- Video B

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Automatic evaluation based on checklists

MLLMs

Are the two videos shot from different camera angles?

Is the location in the two videos different? Does the man check his watch in video B but not in video A?

Are the positions of the lake opposite in the two videos?

Video Difference Captioning

Similarities: Both were filmed from the same fixed camera position over a rice paddy during sunset.

Differences: Video A features a visual sun, creating a strong reflection and a red-orange sky. In contrast, Video B's cloud-obscured sun produces a soft, and diffused reflection.

Checklist Example Similarity Question: Are the two videos filmed at the different locations?

[Figure 53]

Correct Answer: No Model Answer: No Judge Result: Yes

Difference Question: Is the sun unobstructed at the end of video A, while in video B it remains covered?

[Figure 54]

[Figure 55]

[Figure 56]

Correct Answer: Yes Model Answer: Yes Judge Result: Yes

Figure 1: Illustration of the seven categories of video-pair variations in our study: Subject, Style, Playback Technique, Camera Work, Position, Motion, and Background. Using “Background Change” as an exemplar, we showcase our Video Difference Captioning task, where a model generates a caption detailing similarities and differences. The caption’s accuracy is then assessed against a fine-grained checklist.

* Equal Contribution. † Corresponding Author.

### 1 Introduction

Understanding and describing differences between visual inputs is a fundamental capability of human perception and a cornerstone of visual reasoning. While recent progress in image difference captioning (IDC) (Jhamtani and Berg-Kirkpatrick, 2018; Park et al., 2019; Yao et al., 2022; Di et al., 2025; Liu et al., 2026) has enabled models to articulate semantic changes between pairs of static images, these methods remain inherently limited: they operate on snapshots, ignoring the temporal evolution and motion cues that define real-world visual experiences. In dynamic scenes, differences are not only found in static frames but also emerge over time — arising from variations in actions, events, camera movements, or stylistic transitions across time. To bridge this gap, as shown in Figure 1, we introduce Video Difference Captioning (ViDiC), a new task that extends difference captioning into the video domain. Specifically, the ViDiC task requires models to generate natural language descriptions that accurately capture differences in both their static visual content and their temporal dynamics between two video clips while maintaining coherence and factual grounding. This formulation moves beyond traditional video similarity (Zeng et al., 2022; Liberatori et al., 2025) or video editing evaluation (Wu et al., 2023; Argaw et al., 2022; Ju et al., 2025) tasks, focusing instead on edit understanding rather than edit execution.

However, constructing such a benchmark for the ViDiC task presents several challenges. First, video annotation is costly and ambiguous: differences may arise from subtle temporal cues or stylistic variations not easily expressible in simple labels. Second, existing video editing datasets emphasize task completion metrics (e.g., edit fidelity) (Chen et al., 2025a; Sun et al., 2024; Li et al., 2025a; Pan et al., 2026; Chen et al., 2025b), which fail to capture descriptive capabilities. Finally, scalable benchmarking requires standardized evaluation protocols to ensure consistency across diverse models and data sources.

To address these issues, we introduce a dual-checklist evaluation framework and a high-quality video pair dataset called ViDiC-1K, which is designed explicitly for video difference captioning. Our dataset comprises 1,000 curated video pairs (both real and synthetic), annotated with 3720 fine-grained comparative questions spanning seven semantic dimensions: Subject, Style, Background, Camera Work, Motion, Position, and Playback Technique. Each pair is accompanied by both similarity and difference checklists, enabling a detailed, interpretable assessment that transcends single-score metrics. Beyond data design, we propose an LLM-assisted evaluation protocol where a large judge model (GPT-5-Mini) quantifies factual accuracy by comparing generated captions against human-verified ground truths. This scalable, model-agnostic evaluation paradigm ensures reliable comparison across systems without requiring direct visual access during judgment.

In summary, our contributions are threefold:

- • We introduce the Video Difference Captioning task by unifying descriptive, comparative, and temporal understanding, which generalizes image-level difference captioning into the temporal domain and establishes a foundation for advancing multimodal models toward more robust and explainable video reasoning.
- • To evaluate the capabilities of existing MLLMs, we first propose the ViDiC-1K benchmark, comprising 1,000 annotated video pairs with structured similarity–difference checklists across seven spatio-temporal dimensions, and introduce a scalable evaluation framework leveraging LLM-as-a-judge for factual, interpretable, and reproducible benchmarking of MLLMs.
- • Through extensive experiments on existing models, we demonstrate that ViDiC exposes crucial performance gaps in fine-grained temporal reasoning and edit interpretation, and reveals domainspecific weaknesses that remain unsolved even for leading MLLMs.
- • To enhance the model’s capability in ViDiC, we propose a training set, a video difference understanding dataset featuring over 60,000 diverse video pairs across various scenarios. By fine-tuning Qwen-2.5-VL-7B-Instruct on this dataset, we achieve significant performance improvements over the baseline. These results validate the effectiveness of our data and establish a strong foundation for future research in video difference captioning.

### 2 Related Works

Visual Difference Understanding. Visual difference understanding has evolved from low-level pixel comparisons, exemplified by Change Detection(Caye Daudt et al., 2018; Chen and Shi, 2020) using Siamese architectures, to high-level semantic reasoning(Jhamtani and Berg-Kirkpatrick, 2018; Wu et al., 2025). This shift underscores the necessity of interpreting visual semantics in natural language, a domain where the current generation of MLLMs demonstrates remarkable proficiency across a spectrum of tasks, ranging from canonical captioning and VQA (Lin et al., 2015; Agrawal et al., 2019; Hudson and Manning, 2019; Marino et al., 2019; Singh et al., 2019) to holistic reasoning (Liu et al., 2024a; Li et al., 2024).

However, despite these advancements, MLLMs typically operate on a single visual input, restricting their ability to perform comparative understanding. This limitation creates a disconnect with real-world applications—such as video editing verification, content forensics, and intelligent surveillance—where distinguishing fine-grained discrepancies between reference and target footage is essential. Consequently, there is an urgent need to bridge the gap between these practical multi-video requirements and the current deficiency in comparative reasoning capabilities (Peng et al., 2025).

Image Difference Captioning. Image Difference Captioning task focuses on describing semantic changes between two images. Early studies, such as Spot-the-Diff (Jhamtani and Berg-Kirkpatrick, 2018) introduced datasets for learning to verbalize visual differences. Recent works attempt on synthetic data generation and preference-based selection (Wu et al., 2025; Ju et al., 2025; Dunlap et al., 2024). Despite these advances, IDC methods rely on static image pairs and thus fail to capture temporal dynamics or motion consistency. In contrast, ViDiC-1K extends the task into the temporal domain by introducing a benchmark for video difference captioning, where models must reason over both spatial and temporal variations between two video clips. Compared to IDC, this task requires understanding event evolution and motion patterns over time, providing a more comprehensive evaluation framework for video understanding and video editing models.

Video Editing Datasets. The scarcity of high-quality training data remains a primary bottleneck for video editing, constrained by the difficulty of maintaining spatio-temporal consistency and semantic alignment. Although recent works have turned to large-scale synthetic curation (Ju et al., 2025), ensuring the quality of these automated instruction-edit pairs is difficult without a reliable verification loop. This necessitates the development of Video Difference Captioning models, which are essential for scaling up data production by automatically verifying edit success or generating precise edit instructions from video pairs. Unlike prior datasets focused on edit fidelity (Wu et al., 2023; Argaw et al., 2022), ViDiC-1K addresses this need by establishing a benchmark specifically for fine-grained video difference understanding.

### 3 ViDiC-1K

To evaluate video comparison for editing, we introduce the ViDiC-1K benchmark, built on a new framework derived from editing workflows. The framework organizes comparison criteria into seven super-categories (Figure 1) and uses a dual-checklist design assessing both similarities and differences for each video pair. This enables a granular evaluation that overcomes the limitations of single, coarsegrained similarity scores.

#### 3.1 Data Collection

##### 3.1.1 Video Collection

To establish a benchmark with broad coverage, we constructed ViDiC-1K with 1,000 video pairs by aggregating data from existing public sources while also generating videos via our proprietary pipeline. The approximate proportions of specific data sources are shown in Figure 4f. To maintain a high standard of data quality, all videos were uniformly filtered to remove duplicates, videos containing significant artifacts, videos with negligible motion, and those exhibiting excessively large inter-video differences.

- • Externally Sourced Video Collection: The external data was sourced from two primary channels: public academic datasets (VidDiffBench (Burgess et al., 2025), IF-VidCap (Li et al., 2025b), VACE (Jiang et al., 2025), PKU-DyMVHumans (Zheng et al., 2024), ToCaDa (Malon et al., 2018) and DVSC (Pizzi et al., 2023)) and web platforms (YouTube, and LMArena (Zheng et al., 2023)). For the IF-Vidcap dataset and a YouTube subset, we employ a temporal bisection strategy, selecting continuous long takes and manually dividing each into two consecutive segments to generate similar video pairs.
- • Controlled Synthetic Generation via Frame Splicing: To acquire video samples for categories that are difficult to collect naturally, such as dynamic weather transitions, we designed the synthetic pipeline illustrated in Figure 2. The core process involves stacking boundary frames, utilizing the Veo3 model (Wiedemer et al., 2025) to synthesize a composite video, and subsequently splitting it. This synthetic data constitutes only a small fraction of our self-collected dataset. The specific prompts and detailed generation workflow are provided in the supplementary material.
- • CV and Rendering-Based Video Augmentation: As illustrated by the examples in Figure 3, we utilize a variety of computer vision and rendering techniques to achieve fine-grained editing of

video content. This approach allows for precise modifications, including: (1) altering camera perspectives via ReCamMaster (Bai et al., 2025a); (2) modifying artistic styles with stylization tools (Ye et al., 2024); (3) adding or removing specific subjects using SAM-2 (Ravi et al., 2024) combined with inpainting; and (4) re-animating subject actions within rendering engines such as Unreal Engine.

First Frame Last Frame

|[Figure 57]<br><br>[Figure 58]|[Figure 59]<br><br>[Figure 60]|
|---|---|

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

- Video1
- Video2

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Vertical Stitching Crop the Video Vertically

T2V Generation

Figure 2: Synthetic Generation via Frame Splicing.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

###### add a person delete a thing change background

[Figure 83]

[Figure 84]

Figure 3: Edit via CV and Rendering tools.

##### 3.1.2 Annotation Pipeline

Our annotation pipeline employs a two-stage process that combines automated generation with expert validation to ensure quality.

- Stage 1: Automated Draft Generation. This stage begins with the generation of comparative descriptions for video pairs. To mitigate single-model bias and diversify linguistic patterns, we distribute this description task across a pool of MLLMs: GPT-5 (Singh et al., 2025), Gemini-2.5-Pro (Comanici et al., 2025), Qwen3-VL-plus (Yang et al., 2025), InternVL-3.5-241B (Wang et al., 2025a), and Doubao-Seed1.8(ByteDance Seed, 2025). These models are explicitly prompted to prioritize high-level semantic discrepancies over low-level pixel artifacts. This constraint ensures that the generated descriptions capture meaningful content variations rather than technical imperfections. Subsequently, we employ a diverse set of models, including Gemini-2.5-Pro, GPT-5-mini (Singh et al., 2025), and DeepSeek-V3 (DeepSeek-AI et al., 2025), to synthesize these insights into a unified and robust checklist.
- Stage 2: Human Validation. A team of six trained professional annotators meticulously refined the draft checklists. Each list was independently reviewed and corrected by two annotators based on unified criteria, targeting issues such as factual errors, logical contradictions, misclassifications, or excessive subjectivity. Crucially, the review scrutinized semantic validity, explicitly discarding items that were devoid of meaningful content or involved visual distinctions too subtle for humans to perceive. Any disagreements were resolved through a consensus-driven discussion mediated by a third senior annotator. This rigorous, multi-annotator protocol resulted in only 16.32% of the initial model-generated items being retained verbatim; the remainder were either substantially revised or discarded entirely. This process ensures that every item in the final checklists is factually accurate, and precisely aligned with human judgment.

#### 3.2 Dataset Statistics

Overall Statistics Our benchmark comprises 1,000 video pairs annotated with 3720 comparative checklist items (982 similarity, 2738 difference). Figure 4c illustrates the number of video pairs corresponding to each checklist length. The source videos are curated for diversity, with durations predominantly ranging from 2 to 12 seconds, reflecting the typical length observed in modern video editing(Figure 4d), a varied set of resolutions (Figure 4e), and a broad spectrum of topics to ensure generalizability (Figure 4b).

Our comparative checklist items are categorized according to a multifaceted taxonomy, depicted in Figure 4a. This system encompasses seven key dimensions: 1) Subject, covering its type, count, and detailed attributes from appearance to pose; 2) Style, utilizing a constrained list of objective descriptors (e.g., Anime, Oil Painting) to ensure consistency; 3) Background, describing background objects, weather, location, atmosphere, and lighting; 4) Camera Work, analyzing cinematographic elements like movement and scale; 5) Subject Motion, detailing action dynamics; 6) Positional Relationship, focusing on spatial arrangements; and 7) Playback Technique, identifying basic effects like slow-motion. Statistically, Subject, Background and Style account for approximately 57% to align with mainstream editing types, while the 35% allocated to dynamic categories emphasizes temporal changes, distinguishing ViDiC from IDC.

Comparison with Other Benchmarks Current benchmarks for visual comparison suffer from critical fragmentation, focusing either on static images or on isolated tasks. A comprehensive overview in

###### Subject

300

278

250

217

198

200

150

101

100

78 73 69

49

50

24

0

CountClothingAppearanceColorOCRTypePoseStateMaterial

###### Style

35

33

30

25

21

19 18

19

20

15 14

15

12

10 9 8

10

6

5

0

RealisticAnimeFlat CGB&WOilPaintUkiyo-eCyberpunkComicSketchInkOthers

###### Background

500

445

400

300

259

200

84

100

63

19

0

ObjectsLightingLocationWeatherAtmosphere

###### Camera

120

110 107

100

86

80

66

60

54

40

34

26 24

20

0

ScaleMovementOrientationCompositionDOFAngle PerspectiveShots

###### Motion

300

264

250

200

150

107

100

78

50

28 27 21

0

TypesInteractionDirectionSpeedAmplitudeTrajectory

###### Position

150

143

120

95

90

60

30

8

0

LayoutInteractionFlip

###### Playback Tech.

34

35

30

25

20

17

14

15

10

5

0

ReverseSlowFast

- (a) Statistical distribution across various categories including Subject, Style, Background, Camera, Motion, Position, and Playback Technique.

Perspective

Shot Framing

Angle

CameraMovement

Composition

Optics

Scenario

Indoors

Urban

Nature

Virtual

Action

Movement

Transit

Daily Tasks

Interaction

Entertainment

Sports

Professional

Tutorial

Subject

Person

Object

Animal

Landscape

Traffic

Theme

Life

Factuals

Knowledges

Skills

Creative

Entertainment

Commercial

- (b) Hierarchical classification of video content.

Total Items (Same + Diff)

NumberofVideoPairs

Different Items

500

Same Items

400

Self-curated

300

200

100

0

0 2 4 6 8 10 12 14

Number of Checklist Items

(c) Distribution of the number of checklist items per video pair.

LMArena

2-4s 1.7%

[480p-720p) 6.2%

IF-Vidcap

>12s 3.4%

6-8s 11.3%

<480p 18.5%

Youtube

10-12s 11.9%

4-6s 52.4%

Viddiffbench PKU-DyMVHumans VACE Bench and DVSC ToCaDa

[720p-1080p) 54.2% >=1080p 21.1%

8-10s 19.3%

(d) Distribution of video durations.

(e) Distribution of video resolutions.

(f) Distribution of video source datasets.

Figure 4: An overview of the statistical analysis of our dataset across multiple dimensions.

Table 1: Comparison of benchmarks for image and video difference captioning and related tasks. We compare our ViDiC-1K with existing image difference captioning datasets, including Spot-the-Diff (Jhamtani and Berg-Kirkpatrick, 2018), CLEVR-Change (Park et al., 2019), and OmniDiff (Liu et al., 2026), as well as the video action differencing benchmark VidDiffBench (Burgess et al., 2025). “Syn.” and “Real” denote synthetic and real-world data sources, respectively. The size denotes the number of samples in the test set. Reference-based evaluation is a metric that measures model output quality by comparison with pre-defined reference answers (e.g., BLEU, CIDEr).

Benchmark Source Task Category Count Size Evaluation

Spot-the-Diff Real Image Difference Captioning 1 1,400 Reference-based CLEVR-Change Syn. Image Difference Captioning 5 7,970 Reference-based OmniDiff Real and Syn. Image Difference Captioning 12 1,560 Reference-based VidDiffBench Real Video Action Differencing 5 549 Ground Truth + LLM

ViDiC-1K (Ours) Real and Syn. Video Difference Captioning 35 1,000 Checklist + LLM

- Table 1 highlights this gap. A critical comparison with the concurrent VidDiffBench (Burgess et al., 2025) further highlights the distinct value of ViDiC-1K. While VidDiffBench pioneers action differencing, it is strictly confined to skill assessment within five narrow domains (e.g., fitness, surgery), significantly limiting its applicability to identifying specific motion deviations. In contrast, ViDiC-1K captures a comprehensive spectrum of visual variations—including subject consistency, cinematography, and background changes—essential for universal video understanding. This breadth enables critical realworld applications beyond simple coaching, such as validating consistency in generative video editing, identifying video plagiarism for copyright protection, and analyzing complex scenes in intelligent surveillance. To overcome the domain-specific limitations of previous works, ViDiC-1K establishes a holistic framework with 35 fine-grained categories. Notably, regarding data scale, while VidDiffBench relies solely on 549 existing clips, we double the volume to 1,000 by integrating both diverse collected footage and self-produced samples. Finally, by benchmarking 17 diverse MLLMs—versus only five in VidDiffBench—ViDiC-1K serves as a robust, general-purpose foundation for the community.

#### 3.3 Evaluation Methodology

- 3.3.1 Evaluation Framework

Traditional metrics fall short of the evaluation for complex descriptive tasks, as they measure textual similarity rather than factual correctness. To overcome this, we propose a framework to directly quantify factual accuracy using a human-annotated checklist. This checklist is composed of a set of binary (yes/no) questions, denoted as Q, derived from predefined evaluation dimensions. Each question has a corresponding ground-truth answer, forming a ground-truth answer set, AGT. During evaluation, the model under review M, is prompted with a given video pair and the evaluation dimensions to generate a description, D. Subsequently, a powerful and separate Judge model, J (we use GPT-5-Mini), must answer the questions in Q based solely on the information present in D, without access to the videos. This process yields the Judge’s answer set, AJ . The factual accuracy of the description is then determined by the consistency between AJ and AGT, providing a direct and reliable measure of the model’s ability to articulate facts.

The rationale for our evaluation method is twofold. First, we align with the binary judgment paradigm validated by MME(Fu et al., 2025), POPE(Li et al., 2023), and HallusionBench(Guan et al., 2024), which has proven effective in probing fine-grained details and diagnosing hallucinations. Second, to strictly prevent information leakage, the model under test is exposed exclusively to the open-ended prompts and remains blind to the specific judgment questions or verification criteria.

- 3.3.2 Evaluation Metric We formulate the fine-grained video comparison metric as the Accuracy over the question set Q:

|Q|

1

∑

I AJ,i = AGT,i (1)

Accuracy =

|Q|

i=1

Here, I(·) is an indicator function that equals 1 if the model’s answer for question i exactly matches the ground-truth answer, and 0 otherwise. Given the distinct design objectives of similarity and difference questions, we adopt tailored evaluation strategies for each type.

Similarity Questions To penalize hallucination over omission (since enumerating all shared attributes in similar video pairs is impractical), questions about similarities are framed inversely (See Figure 1 for an example.). A response is considered correct if it either confirms the similarity or omits the attribute, thereby only penalizing hallucinated differences.

Difference Questions Conversely, to enforce descriptive accuracy, Difference questions are framed as verifiable propositions about specific differences (See Figure 1 for an example). The model must correctly affirm these true statements, with any failure to verify or omission of the specified details being penalized.

#### 3.4 Training Dataset Construction

To enhance the model’s capability in video difference captioning, we constructed a training dataset comprising over 60k video pairs. The construction methodology mirrors that of our test set; therefore, we omit repetitive descriptions here.

Data Sourcing and Filtering. Our data is aggregated from six diverse sources: Ditto-1M(Bai et al., 2025b), LMArena, Miradata(Ju et al., 2024), Ego-Exo4D(Grauman et al., 2024), MultiCamVideo(Bai et al., 2024), and Vript(Yang et al., 2024). Before processing, we utilized hash mapping to identify and remove any samples overlapping with the evaluation set. Subsequently, we implemented a two-stage filtering pipeline to ensure data quality. First, we applied heuristic filters based on hard metrics, calculating blurriness via Laplacian variance and motion dynamics via optical flow magnitude to remove low-quality samples. Second, specifically for datasets exhibiting large internal quality variance, we utilized Qwen3-VL-32B to perform a secondary semantic screening.

Similarity and Difference Annotation. We employed Qwen3-VL-32B to generate training targets that explicitly capture semantic similarities and differences. This model was selected for its top-tier performance on our benchmark and its open-source availability, enabling high-quality annotations at lower cost.

Comprehensive details regarding the specific video construction strategies for each video source, the quality screening prompts, and the annotation prompts are provided in the supplementary material.

### 4 Experiments

#### 4.1 Main Results

We evaluate 17 popular models including Gemini-3.0-Flash, Gemini-2.5-Pro (Comanici et al., 2025), Gemini-2.5-Flash (Comanici et al., 2025), GPT-5 (Singh et al., 2025), InternVL3.5 (Wang et al., 2025a), Qwen2.5-VL (Bai et al., 2025c), Qwen3-VL (Yang et al., 2025), Keye-VL-1.5 (Team et al., 2025a), Mimo-VLSFT (Team et al., 2025b), Kimi-VL-A3B (Team et al., 2025c), GLM-4.1V (glm, 2026), InternVideo2.5 (Wang et al., 2025b) and LLaVA-v1.6-Vicuna (Liu et al., 2024b). To establish a human baseline, we also invited independent evaluators unaffiliated with this project to participate in the evaluation, resulting in a human performance score. Additionally, we fine-tune Qwen2.5-VL-7B-Instruct on our self-constructed dataset to create ViDiC-Qwen. The main results are presented in Table 2, which lead to the following key observations:

- 1. Model Performance Gap. Our dataset reveals a clear performance hierarchy among models. While proprietary models still lead, open-source models like Qwen3-VL-32B are now outperforming some closed-source rivals, demonstrating rapid progress. Additionally, performance consistently scales with model size within a given model family.
- 2. Semantic Dimension Variations. Models excel at Style recognition and perform reasonably on Subject, Position, and Background. However, Motion, Camera work and Playback technique detection remain particularly weak, especially for open-source models, indicating critical limitations in temporal artifact identification.
- 3. Similarity and Difference Trade-off. High Similarity scores indicate low hallucination, but low Difference scores reveal weak fine-grained perception. Qwen3-VL-8B achieves 80.24% on Similarity but only 49.43% on Difference, capturing coarse distinctions while missing subtle details. Balancing both remains a critical challenge.
- 4. Thinking Mode Impact. Thinking mode improves both Difference and Similarity scores, revealing enhanced fine-grained perception and reduced hallucinations on identical content.
- 5. Incompatibility with Dual-Video Inputs. Furthermore, MLLMs like LLaVA-v1.6-Vicuna-7B exhibited pathological behaviors on dual-video inputs, such as generating repetitive, nonterminating text.
- 6. Training Effectiveness. Our trained model achieves a remarkable average improvement of 11.75 points (50.43 vs. 38.68). This significant boost effectively verifies the effectiveness of our constructed training set. We also further evaluated the model on external benchmarks. As shown in Table ??, it outperforms the baseline on LVBench(Wang et al., 2025c) (38.67 vs. 35.64), with improvements across specific fine-grained categories. Evaluation results on other benchmarks are provided in the supplementary material.

#### 4.2 Further Analysis

Judge Consistency Analysis. To select the most suitable LLM as an automated judge, we conducted a human–model inter-rater reliability analysis. To balance efficiency and reliability, we randomly sampled 750 video pairs, accounting for 75% of our dataset. For this subset, we aggregated responses generated by multiple models, including Gemini-2.5-pro (Comanici et al., 2025), Qwen2.5-VL (Bai et al., 2025c), and GPT-5 (Singh et al., 2025). These responses were then independently assessed by both human annotators (serving as the baseline) and three candidate LLM judges: GPT-5 Mini (Singh et al., 2025), DeepSeek-V3 (DeepSeek-AI et al., 2025), and Qwen3-32B (Yang et al., 2025). The concordance rates, which quantify the alignment of each LLM’s judgments with the human evaluation standards, are summarized in Table ??. The results indicate a strong correlation, particularly for GPT-5 Mini, validating the potential of using LLMs for scalable and consistent evaluation. This is also the reason why we select GPT-5 Mini.

To ensure reliability and rule out stochastic fluctuations, we conducted five repeated judging rounds using GPT-5-mini on captions generated by Gemini-2.5-Pro. The accuracy remained stable (69.30%–69.62%) with high human agreement (94.38%–95.42%), confirming the framework’s robustness.

Effect of Different Video Parameters. We conducted a sensitivity analysis on Qwen2.5-VL-7B-Instruct and Mimo-VL-SFT to evaluate the impact of temporal sampling and spatial resolution. To isolate these effects, we fixed the frame rate at 2 fps while varying the resolution, and fixed the resolution at 720 × 1280 while varying the FPS. As illustrated in Figure 5a, accuracy for both models exhibits a continuous upward trend across both dimensions. This outcome is expected, as increasing data density in both time and space provides more comprehensive visual cues for accurate captioning. However, the

- Table 2: Results of different models on our benchmark across overall metrics and fine-grained categories. Diff., Sim., and Avg. stand for Difference, Similarity, and Average scores(all in %). Pos., Backgr., and Tech. denote Position, Background, and Playback Technique, respectively(all in %). Param. denotes the parameter scale. A superscript lightbulb icon ( ) indicates a “thinking” mode.

Model Param.

Overall Metrics Category Performance

Avg. Diff. Sim. Subject Motion Pos. Backgr. Cam. Style Tech. Human 94.46 92.99 98.57 96.36 94.36 90.14 96.70 92.90 82.31 97.18 Closed-Source

- Gemini-2.5-Pro 69.33 66.84 76.27 71.95 61.71 70.42 75.47 60.41 79.27 66.20

- Gemini-3.0-Flash 65.81 60.04 81.87 66.17 57.78 68.31 69.31 63.88 77.44 61.97 Gemini-2.5-Flash 63.73 57.87 80.04 66.60 56.92 64.79 66.12 58.99 81.71 42.25 GPT-5 62.26 62.03 62.90 62.63 56.79 68.05 74.03 49.75 61.18 40.62

Open-Source

Qwen3-VL 32B 63.90 62.75 67.11 66.64 55.38 69.01 70.30 58.20 62.20 45.07 Qwen3-VL 8B 57.57 49.43 80.24 59.70 48.03 59.86 62.27 54.73 71.95 26.76 Qwen3-VL 8B 55.75 50.99 69.04 56.76 46.84 58.45 64.91 49.21 62.20 29.58 InternVL-3.5 38B 53.62 47.64 70.26 54.48 43.42 53.17 64.36 49.21 54.27 26.76 Mimo-VL-SFT 7B 51.26 41.20 79.33 51.72 39.32 49.30 55.78 53.31 71.95 26.76 Qwen2.5-VL-Instruct 72B 46.22 38.04 69.01 45.00 35.10 47.54 53.74 45.34 62.80 23.94 InternVL-3.5 38B 45.85 36.83 70.98 45.11 40.00 45.94 51.71 42.74 61.59 21.13 InternVL-3.5 8B 45.78 37.80 68.02 46.23 33.68 46.48 53.14 42.74 66.46 21.13 Qwen2.5-VL-Instruct 32B 45.30 35.55 72.48 45.28 35.62 46.83 52.53 43.92 52.44 22.54 Keye-VL-1.5 8B 45.24 30.94 85.12 43.99 35.89 45.55 50.72 45.76 63.98 21.43 Mimo-VL-SFT 7B 43.09 33.27 70.47 45.67 32.82 43.31 44.88 45.58 51.83 22.54 Qwen2.5-VL-Instruct 7B 38.68 25.90 74.31 35.95 32.53 35.21 41.52 43.44 57.32 22.54 InternVL-3.5 8B 38.18 29.34 62.83 39.33 30.48 38.03 43.89 33.12 54.88 18.31 Keye-VL-1.5 8B 38.12 28.94 63.74 38.51 31.53 34.52 43.72 35.52 51.55 21.43 GLM-4.1V 9B 36.51 29.04 57.33 38.30 30.94 33.10 40.81 34.38 41.46 21.13 Kimi-VL-A3B 16B 34.82 21.23 72.71 33.21 28.03 31.34 35.97 40.69 52.44 21.13 InternVideo2.5 7B 34.18 16.95 82.26 29.76 30.60 32.75 33.00 42.74 57.32 21.13 LLaVA-V1.6-Vicuna 7B 25.19 0.58 93.79 17.89 25.98 22.18 17.16 43.69 49.39 22.54

ViDiC-Qwen (Ours) 7B 50.43 41.72 74.69 50.37 38.70 52.11 57.38 48.73 68.10 26.76

- Table 3: Concordance Rates of LLM Judges with a Human Baseline. The table compares agreement percentages.

##### LLMs Average Similarity Difference

GPT-5-mini 95.22 95.90 94.97 DeepSeek-V3 89.37 90.84 88.84 Qwen3-32B 87.23 88.98 86.60

magnitude of improvement differs significantly. The performance gain driven by FPS is much steeper than that of resolution. Notably, Mimo-VL-SFT suffers a significant performance drop at the extremely low sampling rate of 0.5 fps, indicating a critical threshold for temporal information. In comparison, while higher resolutions yield consistent gains, the growth rate is notably smaller and more gradual relative to FPS. This suggests that Video Difference Captioning tasks depend more heavily on motion continuity than fine-grained spatial details, distinguishing Video Difference Captioning from Image Difference Captioning (IDC), which typically rely more on spatial fidelity.

Effect of Input Video Noise To evaluate the robustness of VLMs against common visual corruptions, we conduct experiments on Qwen2.5-VL-7B and Qwen2.5-VL-32B models. We systematically apply blur, noise, and color saturation augmentations at three distinct intensity levels (light, medium, and heavy), visualized in Figure 5b. Detailed augmentation parameters are provided in supplementary material. As shown in Figure 5c, the Difference scores exhibit a consistent decline as augmentation intensity increases across all distortion types. Notably, under light and medium augmentation conditions, the overall semantic content remains largely preserved, and the semantic differences between video

- Table 4: Comparison on fine-grained metrics. Baseline denotes Qwen2.5VL-7B; Ours denotes ViDiCQwen-7B.

Model Entity Recognition Temporal Grounding Summarization Key Info Retrieval

Baseline 32.35 31.36 31.03 36.77 Ours 37.96 36.82 34.48 39.18

Origin Light Medium Heavy

- 18

- 19

- 20

- 21

- 22

- 23

- 24

- 25

- 26

- 27

- 28

- 29

|Original<br><br>| |
|---|
<br><br>Light<br><br>| |
|---|
<br><br>Medium<br><br>| |
|---|
<br><br>Heavy|
|---|

[Figure 85]

50

DifferenceScore(7B)

###### Accuracy(%)

BLUR

45

Mimo-VL-SFT

Qwen2.5-VL-Instruct

40

[Figure 86]

35

0.5 1 2 4 6

Blur Noise Saturation

Augmentation Method

FPS

- 26

- 27

- 28

- 29

- 30

- 31

- 32

- 33

- 34

- 35

- 36

- 37

- 38

- 39

NOISE

Original Light Medium Heavy

50

DifferenceScore(32B)

###### Accuracy(%)

45

Mimo-VL-SFT

[Figure 87]

Qwen2.5-VL-Instruct

40

COLOR

35

320x240 1280x720 1920x1080 2560x1440 3840x2160

Blur Noise Saturation

Resolution

Augmentation Method

(a) Comparison of accuracy for two models: Mimo-VL-SFT and Qwen2.5-VL-7B across varying fps and resolutions.

(b) Three video augmentations (blur, noise, color saturation) applied at light, medium, and heavy intensities, showing their progressive effects on frames.

(c) Difference detection scores across augmentation types and intensity levels for Qwen2.5-VL-7B (top) and Qwen2.5-VL-32B (bottom).

- Figure 5: Overview of video parameters and data augmentation effects. (a) Model accuracy with respect to fps and resolutions. (b) Examples of video augmentations at different intensity levels. (c) Quantitative impact of augmentation intensity on model performance metrics.

pairs remain clearly perceivable to human observers. However, the models’ Difference scores decrease substantially. We posit that this performance shift stems from a reduction in the model’s fine-grained perceptual acuity; the introduced visual interference masks subtle, pixel-level details, thereby impairing its ability to discern minute distinctions. Through qualitative analysis of model outputs, we observe that augmented inputs lead to the omission of fine-grained details in the generated descriptions, such as gender attributes, clothing characteristics, and other nuanced visual elements. This suggests that while the models retain their capacity for high-level semantic understanding, their sensitivity to low-level visual features is significantly compromised under distortion.

Order Sensitivity. To investigate whether the order of video inputs affects model performance, we conduct experiments comparing two settings: Forward (inputting Video A followed by Video B) and Reverse (inputting Video B followed by Video A). We evaluate four representative models: Qwen3VL-32B, Qwen2.5-VL-72B, InternVL3.5-38B , and Keye-VL-1.5-8B . Our results reveal distinct patterns of order sensitivity across different model architectures. The Qwen series demonstrates remarkable robustness to input sequence variations, with Qwen3-VL-32B achieving nearly identical performance (63.90 in Forward vs. 63.04 in Reverse) and Qwen2.5-VL-72B showing similar stability (46.22 vs. 45.45). In sharp contrast, InternVL3.5-38B and Keye-VL-1.5-8B exhibit notable sensitivity to input order. Specifically, InternVL3.5-38B experiences a performance drop from 45.85 in the Forward setting to 42.66 in the Reverse setting, while Keye-VL-1.5-8B declines from 45.24 to 41.67. We attribute this performance gap to differences in model architectures and attention mechanisms. The attention mechanisms in these susceptible models may exhibit an uneven focus on the visual inputs, disproportionately emphasizing one video while potentially neglecting the other, leading to inconsistent reasoning results when the video order is swapped.

Fine-Grained Category Accuracy Analysis. To uncover distinct model capabilities often masked by aggregate-level analysis, we conducted a fine-grained evaluation of four representative multimodal models: Mimo-VL-SFT , Gemini-2.5-pro, InternVL3.5-38B , and Qwen2.5-VL-72B-Instruct. This diverse set, spanning proprietary and open-source multimodal models of varying scales, allows for a multifaceted

###### OVERALL CAPABILITIES

###### SUBJECT

###### STYLE

###### BACKGROUND

Appearance

Anime

Subject

Objects

Ukiyo-e

B&W

Type

Clothing

100

100

100

100

Playback

Style

75

75

75

75

Sketch

CG

50

50

50

50

Weather

Atmosphere

25

25

25

25

State

Color

Realism

Comic

Background

Position

Pose

Count

Oil Paint

Cyberpunk

Location Lighting

Motion Camera

OCR Material

Ink Flat

###### CAMERA

###### MOTION

###### POSITION

###### PLAYBACK

Angle

Amplitude

Flip

Fast

100

100

100

|Shots<br><br>|Composition<br><br>25<br><br>50<br><br>75<br><br>100|
|---|---|
|Perspective|Movement<br><br>|

S

ion

75

75

75

Types

Direction

50

50

50

25

25

25

Scale

DOF

Trajectory

Interaction Layout

Interaction

Slow Reverse

Per

nt

Orientation

Speed

Mimo-VL-SFT Gemini2.5-Pro InternVL3.5-38B Qwen2.5-VL-72B

- Figure 6: Detailed performance analysis. The top-left ‘Overall’ chart summarizes model accuracy across seven high-level categories. The other seven charts offer a fine-grained breakdown for each of these categories, detailing performance on specific sub-tasks.

comparison as illustrated in Figure 6.

Although the models generally exhibit strong capabilities in recognizing core attributes such as Type, and Location, their performance on several fine-grained categories remains exceptionally challenging. These difficult areas reveal critical gaps in current model capabilities. Poor performance on tasks like OCR, combined with struggles in visual aspects such as expression and depth of field, not only differentiate models but also map out the precise frontiers where progress is most needed. By identifying these specific weaknesses, our analysis provides targeted guidance for future research priorities.

Error Analysis. An analysis of our model’s failure cases reveals three recurring and distinct error patterns, as illustrated in Figure 7: (1) hallucinating non-existent differences on identical content; (2) generating self-contradictory responses, simultaneously claiming sameness while describing a change; and most frequently, (3) incomplete or imprecise difference detection, where the model either fails to perceive a salient change or describes it in a vague, uninformative manner. Taking Gemini-2.5-Pro as a representative case, quantitative analysis shows that Missed Detection is the most prevalent error (577 cases, 50.6%), followed by Self-Contradiction (323 cases, 28.3%) and Hallucination (191 cases, 16.7%). Detailed results for other models and experimental configurations are provided in the supplementary material. These errors collectively highlight current limitations in the model’s fine-grained perception, logical consistency, and robust visual grounding, pointing to clear avenues for future improvement.

The first video has a noticeable zoom-in compared to the second video, but the model ignores this difference.

Both videos have a similar warm tone with no clear atmospheric or lighting differences, the model’s response is overly subjective.

In the second video the leaves are falling, unlike in the first, yet the model says they’re identical.

[Figure 88]

[Figure 89]

[Figure 90]

Error Analysis

Generated Caption

[Figure 91]

[Figure 92]

Similarities: Both videos feature the same rabbit eating an orange, with floating still leaves in the background.

Differences: The first video has a warmer tone, while the second video has a cooler tone. The lighting in the second video is also noticeably brighter than in the first.

|[Figure 93]<br><br>[Figure 94]|
|---|

[Figure 95]

[Figure 96]

[Figure 97]

|[Figure 98]<br><br>[Figure 99]|
|---|

Similarities: Both video clips show the 3D animated character with

a blown hair, looking at a mobile phone with a single-color background.

Differences: The bird icons differ - the first is blue while the second is white. The background differs - the second video has a yellow background with white in the center, while the first one is only blue.

The first video shows a finger pointing at the phone screen; the second does not, but the model ignores this difference.

The bird icons are identical, but the model wrongly claimed they differ The first is blue, while the second is white.

The model said both backgrounds are single-color, yet claimed the second is yellow with white in the center, which is self-contradictory.

|[Figure 100]<br><br>[Figure 101]<br><br>|
|---|

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Error Analysis

Generated Caption

[Figure 106]

[Figure 107]

[Figure 108]

|[Figure 109]<br><br>[Figure 110]|
|---|

- Figure 7: We present two illustrative failure cases of the model. In each example, the left column displays two representative frames from the video pair. The top panel shows the generated caption, while the bottom panel identifies the errors and their underlying causes. For clarity, the model outputs have been simplified.

Furthermore, the rabbit example on the right of Figure 7 exposes a deeper, modality-specific challenge that distinguishes ViDiC from static Image Difference Captioning. In this scenario, the model fails to capture the temporal evolution of the scene, resulting in two critical omissions: it completely neglects the camera zoom-in operation in the first video and misinterprets the dynamic falling leaves in the second video as being stationary. These failures suggest that the model currently struggles to perceive changes that occur over time rather than just in visual appearance. This underscores the essential significance of the ViDiC task, as it demands a higher-level perception of temporal continuity and camera intent, capabilities that are indispensable for comprehensive video understanding but are absent in static image comparisons.

### 5 Conclusion

This paper introduces Video Difference Captioning (ViDiC), a novel task aimed at advancing comparative spatio-temporal video understanding. To support this task, we present a large-scale dataset comprising over 60,000 training samples and the ViDiC-1K benchmark, which spans seven critical dimensions. Furthermore, we propose the Dual-Checklist framework, an LLM-as-a-judge approach designed for rigorous and factually accurate assessment. Our evaluation of 17 representative multimodal models reveals significant bottlenecks in comparative perception, particularly in analyzing cinematography and motion. By highlighting these limitations, ViDiC provides a robust foundation and a new trajectory for developing fine-grained video understanding models.

### 6 Future Directions

This work establishes a foundational benchmark for the novel task of Video Difference Captioning (ViDiC), introducing a high-quality test set (ViDiC-1K) and a dual-checklist evaluation framework that reveals the comparative spatio-temporal understanding capabilities of current MLLMs. Looking forward, we envision leveraging this methodology to advance video editing research. Specifically, our ViDiC framework can be adapted to construct large-scale, high-quality training datasets for video editing tasks. By systematically capturing fine-grained differences between original and edited videos, this approach can generate precise edit descriptions and annotations that significantly improve the quality of video editing training data. This direction holds promise for developing more capable video editing models that understand and execute complex editing instructions with greater accuracy and nuance.

### References

Harsh Jhamtani and Taylor Berg-Kirkpatrick. Learning to describe differences between pairs of similar images, 2018. URL https://arxiv.org/abs/1808.10584.

Dong Huk Park, Trevor Darrell, and Anna Rohrbach. Robust change captioning, 2019. URL https: //arxiv.org/abs/1901.02527.

Linli Yao, Weiying Wang, and Qin Jin. Image difference captioning with pre-training and contrastive learning, 2022. URL https://arxiv.org/abs/2202.04298.

Zonglin Di, Jing Shi, Yifei Fan, Hao Tan, Alexander Black, John Collomosse, and Yang Liu. Difftell: A high-quality dataset for describing image manipulation changes. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 24580–24590, October 2025.

Yuan Liu, Saihui Hou, Saijie Hou, Jiabao Du, Shibei Meng, and Yongzhen Huang. Omnidiff: A comprehensive benchmark for fine-grained image difference captioning, 2026. URL https://arxiv.org/abs/ 2503.11093.

Zhaoyang Zeng, Yongsheng Luo, Zhenhua Liu, Fengyun Rao, Dian Li, Weidong Guo, and Zhen Wen. Tencent-mvse: A large-scale benchmark dataset for multi-modal video similarity evaluation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 3128–3137. IEEE, 2022. doi: 10.1109/CVPR52688.2022.00314. URL https: //doi.org/10.1109/CVPR52688.2022.00314.

Benedetta Liberatori, Alessandro Conti, Lorenzo Vaquero, Yiming Wang, Elisa Ricci, and Paolo Rota. Convis-bench: Estimating video similarity through semantic concepts, 2025. URL https://arxiv.org/ abs/2509.19245.

Jay Zhangjie Wu, Xiuyu Li, Difei Gao, Zhen Dong, Jinbin Bai, Aishani Singh, Xiaoyu Xiang, Youzeng Li, Zuwei Huang, Yuanxi Sun, Rui He, Feng Hu, Junhua Hu, Hai Huang, Hanyu Zhu, Xu Cheng, Jie Tang, Mike Zheng Shou, Kurt Keutzer, and Forrest Iandola. Cvpr 2023 text guided video editing competition, 2023. URL https://arxiv.org/abs/2310.16003.

Dawit Mureja Argaw, Fabian Caba Heilbron, Joon-Young Lee, Markus Woodson, and In So Kweon. The anatomy of video editing: A dataset and benchmark suite for ai-assisted video editing, 2022. URL https://arxiv.org/abs/2207.09812.

Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Nanxuan Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, Daniil Pakhomov, Zhe Lin, Soo Ye Kim, and Qiang Xu. Editverse: Unifying image and video editing and generation with in-context learning, 2025. URL https://arxiv.org/abs/ 2509.20360.

Yinan Chen, Jiangning Zhang, Teng Hu, Yuxiang Zeng, Zhucun Xue, Qingdong He, Chengjie Wang, Yong Liu, Xiaobin Hu, and Shuicheng Yan. Ivebench: Modern benchmark suite for instruction-guided video editing assessment, 2025a. URL https://arxiv.org/abs/2510.11647.

Shangkun Sun, Xiaoyu Liang, Songlin Fan, Wenxu Gao, and Wei Gao. Ve-bench: Subjective-aligned benchmark suite for text-driven video editing quality assessment, 2024. URL https://arxiv.org/abs/ 2408.11481.

Caorui Li, Yu Chen, Yiyan Ji, Jin Xu, Zhenyu Cui, Shihao Li, Yuanxing Zhang, Jiafu Tang, Zhenghao Song, Dingling Zhang, Ying He, Haoxiang Liu, Yuxuan Wang, Qiufeng Wang, Zhenhe Wu, Jiehui Luo, Zhiyu Pan, Weihao Xie, Chenchen Zhang, Zhaohui Wang, Jiayi Tian, Yanghai Wang, Zhe Cao, Minxin Dai, Ke Wang, Runzhe Wen, Yinghao Ma, Yaning Pan, Sungkyun Chang, Termeh Taheri, Haiwen Xia, Christos Plachouras, Emmanouil Benetos, Yizhi Li, Ge Zhang, Jian Yang, Tianhao Peng, Zili Wang, Minghao Liu, Junran Peng, Zhaoxiang Zhang, and Jiaheng Liu. Omnivideobench: Towards audiovisual understanding evaluation for omni mllms, 2025a. URL https://arxiv.org/abs/2510.10689.

Yaning Pan, Qianqian Xie, Guohui Zhang, Zekun Wang, Yongqian Wen, Yuanxing Zhang, Haoxuan Hu, Zhiyu Pan, Yibing Huang, Zhidong Gan, Yonghong Lin, An Ping, Shihao Li, Yanghai Wang, Tianhao Peng, and Jiaheng Liu. Mt-video-bench: A holistic video understanding benchmark for evaluating multimodal llms in multi-turn dialogues, 2026. URL https://arxiv.org/abs/2510.17722.

Xinlong Chen, Yuanxing Zhang, Chongling Rao, Yushuo Guan, Jiaheng Liu, Fuzheng Zhang, Chengru Song, Qiang Liu, Di Zhang, and Tieniu Tan. Vidcapbench: A comprehensive benchmark of video captioning for controllable text-to-video generation, 2025b. URL https://arxiv.org/abs/2502.12782.

Rodrigo Caye Daudt, Bertr Le Saux, and Alexandre Boulch. Fully convolutional siamese networks for change detection. In 2018 25th IEEE International Conference on Image Processing (ICIP), pages 4063–4067,

2018. doi: 10.1109/ICIP.2018.8451652.

Hao Chen and Zhenwei Shi. A spatial-temporal attention-based method and a new dataset for remote sensing image change detection. Remote. Sens., 12:1662, 2020. URL https://api.semanticscholar.org/ CorpusID:219512939.

Keming Wu, Sicong Jiang, Max Ku, Ping Nie, Minghao Liu, and Wenhu Chen. Editreward: A humanaligned reward model for instruction-guided image editing, 2025. URL https://arxiv.org/abs/2509. 26346.

Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. Microsoft coco: Common objects in context,

2015. URL https://arxiv.org/abs/1405.0312.

Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. nocaps: novel object captioning at scale. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 8947–8956, 2019. doi: 10.1109/ICCV.2019. 00904.

Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering, 2019. URL https://arxiv.org/abs/1902.09506.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question

answering benchmark requiring external knowledge, 2019. URL https://arxiv.org/abs/1906.00067. Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and

Marcus Rohrbach. Towards vqa models that can read, 2019. URL https://arxiv.org/abs/1904.08920.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player?, 2024a. URL https://arxiv.org/abs/2307.06281.

Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension, 2024. URL https://arxiv.org/abs/2404.16790.

Tianhao Peng, Haochen Wang, Yuanxing Zhang, Zekun Wang, Zili Wang, Gavin Chang, Jian Yang, Shihao Li, Yanghai Wang, Xintao Wang, Houyi Li, Wei Ji, Pengfei Wan, Steven Huang, Zhaoxiang Zhang, and Jiaheng Liu. Mvu-eval: Towards multi-video understanding evaluation for multimodal llms, 2025. URL https://arxiv.org/abs/2511.07250.

Lisa Dunlap, Yuhui Zhang, Xiaohan Wang, Ruiqi Zhong, Trevor Darrell, Jacob Steinhardt, Joseph E. Gonzalez, and Serena Yeung-Levy. Describing differences in image sets with natural language, 2024. URL https://arxiv.org/abs/2312.02974.

James Burgess, Xiaohan Wang, Yuhui Zhang, Anita Rau, Alejandro Lozano, Lisa Dunlap, Trevor Darrell, and Serena Yeung-Levy. Video action differencing, 2025. URL https://arxiv.org/abs/2503.07860.

Shihao Li, Yuanxing Zhang, Jiangtao Wu, Zhide Lei, Yiwen He, Runzhe Wen, Chenxi Liao, Chengkang Jiang, An Ping, Shuo Gao, Suhan Wang, Zhaozhou Bian, Zijun Zhou, Jingyi Xie, Jiayi Zhou, Jing Wang, Yifan Yao, Weihao Xie, Yingshui Tan, Yanghai Wang, Qianqian Xie, Zhaoxiang Zhang, and Jiaheng Liu. If-vidcap: Can video caption models follow instructions?, 2025b. URL https://arxiv.org/abs/2510. 18726.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing, 2025. URL https://arxiv.org/abs/2503.07598.

Xiaoyun Zheng, Liwei Liao, Xufeng Li, Jianbo Jiao, Rongjie Wang, Feng Gao, Shiqi Wang, and Ronggang Wang. Pku-dymvhumans: A multi-view video benchmark for high-fidelity dynamic human modeling,

2024. URL https://arxiv.org/abs/2403.16080.

Thierry Malon, Geoffrey Roman-Jimenez, Patrice Guyot, Sylvie Chambon, Vincent Charvillat, Alain Crouzil, André Péninou, Julien Pinquier, Florence Sèdes, and Christine Sénac. Toulouse campus surveillance dataset: scenarios, soundtracks, synchronized videos with overlapping and disjoint views. In Proceedings of the 9th ACM Multimedia Systems Conference, MMSys ’18, page 393–398, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450351928. doi: 10.1145/3204949.3208133. URL https://doi.org/10.1145/3204949.3208133.

Ed Pizzi, Giorgos Kordopatis-Zilos, Hiral Patel, Gheorghe Postelnicu, Sugosh Nagavara Ravindra, Akshay Gupta, Symeon Papadopoulos, Giorgos Tolias, and Matthijs Douze. The 2023 video similarity dataset and challenge, 2023. URL https://arxiv.org/abs/2306.09489.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. URL https://arxiv.org/abs/2306.05685.

Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners, 2025. URL https://arxiv.org/abs/2509.20328.

Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, and Di Zhang. Recammaster: Camera-controlled generative rendering from a single video, 2025a. URL https://arxiv.org/abs/2503.11647.

Zixuan Ye, Huijuan Huang, Xintao Wang, Pengfei Wan, Di Zhang, and Wenhan Luo. Stylemaster: Stylize your video with artistic generation and translation, 2024. URL https://arxiv.org/abs/2412.07744.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos, 2024. URL https://arxiv.org/abs/2408.00714.

Aaditya Singh et al. Openai gpt-5 system card, 2025. URL https://arxiv.org/abs/2601.03267. Gheorghe Comanici et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long

context, and next generation agentic capabilities, 2025. URL https://arxiv.org/abs/2507.06261.

An Yang et al. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388. Weiyun Wang et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning,

and efficiency, 2025a. URL https://arxiv.org/abs/2508.18265. ByteDance Seed. Seed1.8 model card: Towards generalized real-world agency. Technical report, December

2025. URL https://lf3-static.bytednsdoc.com/obj/eden-cn/lapzild-tss/ljhwZthlaukjlkulzlp/ research/Seed-1.8-Modelcard.pdf.

DeepSeek-AI et al. Deepseek-v3 technical report, 2025. URL https://arxiv.org/abs/2412.19437.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2025. URL https://arxiv.org/abs/ 2306.13394.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models, 2023. URL https://arxiv.org/abs/2305.10355.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models, 2024. URL https://arxiv.org/abs/2310.14566.

Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, Yinghao Xu, Yujun Shen, and Qifeng Chen. Scaling instruction-based video editing with a high-quality synthetic dataset, 2025b. URL https://arxiv.org/abs/2510.15742.

Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions,

2024. URL https://arxiv.org/abs/2407.06358. Kristen Grauman et al. Ego-exo4d: Understanding skilled human activity from first- and third-person perspectives, 2024. URL https://arxiv.org/abs/2311.18259.

Jianhong Bai, Menghan Xia, Xintao Wang, Ziyang Yuan, Xiao Fu, Zuozhu Liu, Haoji Hu, Pengfei Wan, and Di Zhang. Syncammaster: Synchronizing multi-camera video generation from diverse viewpoints. arXiv preprint arXiv:2412.07760, 2024.

Dongjie Yang, Suyuan Huang, Chengqiang Lu, Xiaodong Han, Haoxin Zhang, Yan Gao, Yao Hu, and Hai Zhao. Vript: A video is worth thousands of words, 2024. URL https://arxiv.org/abs/2406.06040.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025c. URL https://arxiv.

org/abs/2502.13923. Kwai Keye Team et al. Kwai keye-vl technical report, 2025a. URL https://arxiv.org/abs/2507.01949. Core Team et al. Mimo-vl technical report, 2025b. URL https://arxiv.org/abs/2506.03569. Kimi Team et al. Kimi-vl technical report, 2025c. URL https://arxiv.org/abs/2504.07491. Glm-4.5v and others, 2026. URL https://arxiv.org/abs/2507.01006.

Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, Min Dou, Kai Chen, Wenhai Wang, Yu Qiao, Yali Wang, and Limin Wang. Internvideo2.5: Empowering video mllms with long and rich context modeling, 2025b. URL https://arxiv.org/abs/2501.12386.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2024b. URL https://arxiv.org/abs/2310.03744.

Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Lvbench: An extreme long video understanding benchmark, 2025c. URL https://arxiv.org/abs/2406.08035.

### A Real-world Applications of ViDiC

ViDiC moves beyond simple similarity scoring to offer actionable, semantic insights into video discrepancies. By articulating what has changed, this capability empowers diverse domains:

- • Content Integrity and Forensics: ViDiC facilitates copyright protection by detecting subtle obfuscations. Furthermore, it combats disinformation by explicitly identifying and describing manipulated regions, acting as an interpretable tool for fact-checking.
- • Video Editing Verification: Serving as a rigorous evaluator, ViDiC validates whether editing models adhere to user prompts. It ensures precise modifications without introducing unintended artifacts or background alterations by explicitly captioning discrepancies between source and edited footage.
- • Automated Video Production: In collaborative workflows, the system streamlines post-production by generating automatic Change Logs. These summaries of editorial adjustments effectively bridge the communication gap between human editors and AI tools.
- • Skill Assessment and Rehabilitation: For applications in sports training or physical therapy, ViDiC provides granular feedback on pose deviations and temporal misalignments by comparing user performance against standard reference videos.
- • Scientific Monitoring: It automates change detection in longitudinal observation footage, converting visual shifts in dynamic environments into structured descriptive reports for efficient analysis.
- • Intelligent Surveillance: Surpassing conventional motion sensing, semantic analysis distinguishes between environmental interference and relevant activity, ensuring that alerts are triggered only by specific objects or actionable security behaviors.

### B Construction of the Training Set

In this section, we detail the construction strategies employed for the diverse data sources within our training set. First, for the large-scale open-source datasets, we apply a quality filtering pipeline based on four key quantitative metrics: resolution, blurriness, brightness, and motion intensity. This rigorous screening removes samples with insufficient visual clarity, extreme lighting conditions, or lack of motion. Beyond these general metrics, we implement tailored strategies to accommodate the distinct properties of data sources. For Ditto-1M, which focuses on video editing, we aim to eliminate artifacts introduced during the editing process. We leverage the large multimodal model Qwen3-VL-32B to assess postedit quality, filtering out videos that exhibit temporal instability or incoherence. Finally, for Vript and Miradata, we employ a long-shot segmentation strategy. By cutting continuous long-take footage into shorter segments, we successfully extract naturally cohesive and visually similar video pairs.

[Figure 111]

- Figure 8: The annotation interface used for the final inspection. Annotators utilize this tool to simultaneously filter out unqualified videos and validate the rationality of the checklist and standard answers.

### C Dataset Samples

To better illustrate our dataset, this section presents two specific examples. Each example features key frames from a pair of similar videos, along with a corresponding checklist of questions.

Example 1

[Figure 112]

[Figure 113]

Example 2

[Figure 114]

[Figure 115]

Similarity Question

- • Class: Camera Category: Shot Scale Question: Is the camera’s shot scale different in both videos? Correct Answer: no
- • Class: Playback Technique Question: Is the playback technique different in both videos? Correct Answer: no

Difference Question

- • Class: Motion Category: Motion Type Question: Does the man in the Video B touch his head with his hand, a motion that is not seen in the Video A? Correct Answer: yes
- • Class: Background Category: Lighting Question: Is the lighting in the Video B white-toned, in contrast to the warm, golden-toned lighting in the Video A? Correct Answer: yes

Similarity Question

- • Class: Background Category: Key Background Object Question: Is the man in black in the background different in the two videos? Correct Answer: no
- • Class: Style Question: Is the style different between the Video A and B? Correct Answer: no

Difference Question

- • Class: Subject Category: Gender Question: Is the foreground subject in the Video A a male, in contrast to the foreground subject in the Video B who is a female? Correct Answer: yes
- • Class: Subject Category: Clothing and Accessories Question: Does the foreground subject in the Video A wear a blue long-sleeved garment, while the one in the Video B wears a red sleeveless dress? Correct Answer: yes

### D Construction of the Test Set

#### D.1 Rigorous Data Curation and Dataset Composition

To refine our video collection, we implemented a rigorous three-stage cleaning pipeline that transformed 8,756 raw candidate pairs into 1,000 high-quality samples. First, an automated temporal filter removed outliers outside the 2s–40s range, reducing the pool to 5,420 pairs. Second, annotators manually removed trivial (identical) or invalid (excessively disparate) pairs, narrowing the candidates down to 1,850. Finally, to guarantee the reliability of the benchmark, trained professionals performed a comprehensive validation using the custom interface shown in Figure 8. In this stage, annotators simultaneously evaluated visual dynamics and annotation quality. Specifically, they discarded samples exhibiting static scenes or negligible motion, while concurrently verifying the rationality of the checklists and the correctness of the ground truth answers to ensure strict alignment with the video content. This process, conducted over one month, eliminated samples with either low-quality footage or ambiguous annotations to yield the final 1,000 pairs.

The final test set exhibits diverse source composition: CV and Rendered Synthesis contributes the largest portion at 340 samples (34.00%), followed by LMArena with 157 samples (15.70%), IF-VidCap with 131

samples (13.10%), YouTube with 116 samples (11.60%), and Splice-based Synthesis with 85 samples (8.5%). Smaller but significant contributions come from VidDiffBench (50 samples, 5.00%), PKU-DyMVHumans (49 samples, 4.90%), ToCaDa (28 samples, 2.80%), VSC (27 samples, 2.70%), and VACE_Bench (17 samples, 1.70%). This distribution ensures the dataset retains diverse high-quality samples ranging from synthetic edits to real-world footage.

#### D.2 Controllable Synthesis Pipeline via Frame Splicing

For this part, we detail the workflow for the controlled synthetic generation pipeline via Frame Splicing, designed to produce high-quality, temporally consistent video pairs. The process begins with GPT-5 generating precise, attribute-specific editing prompts based on a source image. These prompts aim to modify specific semantics (e.g., weather or style) while strictly maintaining the original scene structure. A comprehensive list of these prompts is provided in the final "Prompt Summary" section. Guided by these prompts, Nano Banana creates a static edited counterpart, modifying only the target attributes while rigorously preserving the remaining visual content and layout. To facilitate similar motion dynamics during video generation, we employ a frame splicing strategy where the original and edited images are vertically stitched into a single composite frame. This composite frame serves as the initial frame input for the Veo3 model, where spatial-temporal attention mechanisms encourage highly similar motion trajectories across the upper and lower segments. Finally, the output is cropped along the stitching seam to yield video pairs with closely matched dynamics but distinct visual attributes.

To ensure data reliability, we implemented a rigorous manual validation phase for all Veo3 outputs. Expert annotators reviewed the decoupled pairs against two criteria: (1) motion dynamics must remain highly consistent throughout the duration, and (2) visual differences must strictly reflect the intended semantic transformations rather than generation artifacts. Pairs exhibiting significant motion discrepancies, unnatural distortions, or synthesis flaws were systematically discarded. This protocol ensures our dataset evaluates a model’s comprehension of complex video content and semantic shifts, rather than its sensitivity to low-level visual noise.

### E Experiment

#### E.1 Evaluation Settings

We provide the detailed settings for all evaluated open-source models in Table 5. Most models are tested under default settings. Closed-source models are accessed via API calls, using the default configuration.

#### E.2 Training Settings

We fine-tuned the Qwen2.5-VL-7B-Instruct model using a full-parameter fine-tuning strategy to specialize it as a structured video comparative analysis engine. The training was conducted on a cluster of NVIDIA GPUs utilizing DeepSpeed ZeRO-3 optimization to manage the memory footprint of the 7B parameters. We employed the BFloat16 precision format to ensure numerical stability while maintaining computational efficiency. The model was trained for three epochs with a peak learning rate of 1e-5, following a cosine decay schedule with a 3% warmup phase and a minimum learning rate of 1e-6. To accommodate the high-dimensional spatial-temporal data of dual-video inputs, the maximum sequence length was extended to 10,240 tokens, supported by gradient checkpointing and a global batch size of 64. Our training objective focused on the Comparison Mode task, supervising the model to generate strictly formatted JSON outputs that characterize similarities and differences across seven distinct visual dimensions: subject, style, background, camera work, motion, positional relationship, and playback technique.

#### E.3 Evaluation results on other benchmarks

To verify the generalization capabilities of ViDiC-Qwen and ensure that the enhancements in fine-grained tasks do not come at the cost of overfitting, we further evaluated the model on several open-source benchmarks. As shown in the results, ViDiC-Qwen demonstrates robust performance across different evaluation protocols. Specifically, we observe performance gains on MMBench-Video (1.94 vs. 1.77) and Video-MMMU (32.33 vs. 31.67). Meanwhile, the performance on Video-MME remains comparable to the baseline (54.48 vs. 55.0). These results suggest that our training strategy effectively boosts specific video understanding capabilities while preserving the model’s generalizability.

- Table 5: Evaluation metrics for locally deployed open-source models. The "Frame" column represents the frame rate (float) or fixed frame number (integer). "None" in the table means disabled. "Auto" means determined by the model’s default configuration. A superscript lightbulb icon ( ) indicates a "thinking" mode.

Models Params Resolution Frame Temperature Top_p Repetition Penalty

InternVL-3.5 38B 448 × 448 32 0.6 0.9 1.05 InternVL-3.5 38B 448 × 448 32 0.1 0.9 1.05 InternVL-3.5 8B 448 × 448 32 0.6 0.9 1.05 InternVL-3.5 8B 448 × 448 32 0.1 0.9 1.05 Qwen2.5-VL-Instruct 72B Auto 2.0 0.7 0.9 1.05 Qwen2.5-VL-Instruct 32B Auto 2.0 0.7 0.9 1.05 Qwen2.5-VL-Instruct 7B Auto 2.0 0.7 0.9 1.05 ViDiC-Qwen 7B Auto 2.0 0.7 0.9 1.05 Qwen3-VL 32B Auto 2.0 0.7 0.9 Auto Qwen3-VL 8B Auto 2.0 0.7 0.9 Auto Qwen3-VL 8B Auto 2.0 0.7 0.9 Auto Keye-VL-1.5 8B Auto 2.0 0.1 0.001 1.05 Keye-VL-1.5 8B Auto 2.0 0.1 0.001 1.05 LlaVA-V1.6-Vicuna 7B 448 × 448 32 None None Auto Kimi-VL-A3B 16B Auto 32 0.7 0.9 Auto InternVideo2.5 7B 448 × 448 32 None None Auto GLM-4.1V 9B Auto 32 0.1 Auto Auto Mimo-VL-SFT 7B Auto 2.0 0.3 0.95 Auto Mimo-VL-SFT 7B Auto 2.0 0.3 0.95 Auto

#### E.4 Robustness Experiment Details

Three distortion types (Gaussian noise, Gaussian blur, and color saturation) are applied at Light, Medium, and Heavy intensities using OpenCV. Unlike random perturbation strategies, we apply deterministic parameters for each intensity level to ensure strict reproducibility.

Gaussian Noise: We inject additive noise sampled from a normal distribution N (0, σ2), with standard deviations σ set to 15, 30, and 45 for the Light, Medium, and Heavy levels, respectively. To ensure identical noise patterns for a given video, a fixed random seed (42) is employed during generation. Gaussian Blur: We apply a smoothing filter with kernel dimensions of 5 × 5 (Light), 9 × 9 (Medium), and 15 × 15 (Heavy). The standard deviation of the Gaussian kernel is automatically determined based on the kernel size to ensure consistent blurring effects. Color Saturation: We scale the S-channel in the HSV color space by multiplicative factors of 0.75 (Light), 0.50 (Medium), and 0.25 (Heavy), effectively reducing the color vividness at increasing intensity levels.

Crucially, to ensure that the observed similarity metrics reflect the model’s intrinsic robustness rather than artificial disparities, identical perturbation parameters are rigorously applied to both videos within a comparison pair.

### F Error Analysis

This section presents a comprehensive error analysis through quantitative distributions and qualitative examples. To examine current vision-language models’ failure modes, we evaluated four representative models: Gemini-3.0-flash, Qwen3-VL-32B, InternVL-3.5-38B , and Mimo-VL-SFT . We employed GPT5mini to categorize errors by analyzing model responses alongside judge verdicts and rationales (The prompt is provided in the final "Prompt Summary" section.) As shown in Table 6, Missed Detection emerges as the primary error source across all models.

To provide deeper insights into these error patterns, we present several representative cases from different models in the following examples. For clarity of presentation, the displayed model outputs have been simplified.

- Table 6: Detailed breakdown of error type distribution across four representative vision-language models. The table reports both the absolute occurrence count and the relative percentage for three primary error categories: Missed Detection, Self-Contradiction, and Hallucination.

##### Model Missed Detection Self-Contradiction Hallucination

Count Percent Count Percent Count Percent

Gemini-3.0-flash 680 53.5 394 31.0 135 10.6 InternVL-3.5-38B 901 52.2 504 29.2 241 14.0 Mimo-VL-SFT 978 53.9 594 32.8 167 9.2 Qwen3-VL-32B 600 44.7 413 30.8 258 19.2

##### Example 1

[Figure 116]

[Figure 117]

Generated caption: The key distinctions lie in subject movement. In the Video A, the subject is seated with hands on lap and performs a vertical head nod, while in the Video B, the subject stands with crossed arms and turns their head horizontally.

Error Analysis: The response claims the man shifts from seated to standing and changes head movements. In reality, the central subject’s actions are identical. The analysis fabricates these differences while ignoring the actual change: a person walking a dog in Video B.

##### Example 2

[Figure 118]

[Figure 119]

Generated caption: The video demonstrates how the same anime character appears in two completely different styles. The Video A has a soft, pastel-colored aesthetic, depicting an animated urban setting.

Error Analysis: The core error is a critical omission of information. The model only analyzes the style of the Video A, while completely failing to mention the Video B’s style—specifically the "cyberpunk" style. This leaves the response half-finished, failing to provide the requested comparison or a complete analysis.

##### Example 3

[Figure 120]

[Figure 121]

Generated caption: The Video B shows the same room as the Video A: a man in a patterned short-sleeve shirt and shorts is seated at an electronic keyboard, with microphones and stands nearby. Windows line the back wall. Chairs and musical equipment are arranged around the room in the same overall layout.

Error Analysis: The analysis fails to account for the impact of camera perspective on object visibility. Although the actual scene layout remains unchanged, the Video B reveals more chairs due to a higher angle and shifted camera orientation, demonstrating a limited understanding of how camera variations alter the quantity of visible objects in the frame.

##### Example 4

[Figure 122]

[Figure 123]

Generated caption: In both videos, the subject walks steadily along the identical walkway, maintaining consistent motion. The only variation lies in the framing and field of view. Due to the difference in perspective, the Video B intentionally features fewer cars to create a cleaner shot.

Error Analysis: The model’s response is incomplete because it fails to mention the two most important consequences of the perspective change. Specifically, it misses how the new angle changes the view of the walkway from horizontal to longitudinal, creating a strong sense of depth, and how this in turn shifts the subject’s perceived motion from moving sideways to moving directly toward the camera.

### G Prompts

#### G.1 Comparative Descriptions

During the Automated Draft Generation stage, we used diverse models to produce detailed comparative descriptions highlighting similarities and differences between video pairs. The prompt is as follows:

Video Analyst Prompt

#### ROLE & TASK

You are a professional, detail-oriented video analyst. Your task is to perform a comprehensive, objective, side-by-side comparison of two videos: Video A and Video B. Your entire existence is governed by a single principle: radical objectivity. You are a machine for observing and reporting, not for interpreting. Before you write a single word, you must ask yourself: "Is this statement an undeniable, verifiable visual fact?" If the answer is anything less than a 100% certain "yes", you must discard the statement. This principle overrides all other instructions.

#### CRITICAL INSTRUCTIONS

You must strictly follow these rules for your entire response:

- 1. ENGLISH ONLY: Your entire response MUST be in English.
- 2. Strict Objectivity and Certainty: Your analysis must be based only on clear, verifiable visual evidence.

- • No Subjective Language: Do not use interpretive words (e.g., "beautiful," "skillfully shot").
- • No Ambiguity or Speculation: If an element is unclear, you MUST omit it entirely.
- • No Fabrication: Do not add details that do not exist.

- 3. ‘Diff’ sections are for differences only: Under any Diff heading, you MUST only describe aspects that are different.
- 4. skip ‘Diff’ if identical: If a category is completely identical, describe the similarities under Same and then completely omit the Diff section.
- 5. Semantic Priority: Evaluate the video based strictly on semantic content, such as subject identity and action logic. Completely disregard non-semantic AI-generated flaws.
- 6. Carefully verify which parts are truly the same before marking them as ‘Same’.

#### OUTPUT FORMAT INSTRUCTIONS

You MUST format your entire response in Markdown. Adhere strictly to the following structure:

- • Main Categories: Use Level 2 Headings.
- • Comparison Labels: Use bold for **Same** and **Diff**.
- • Difference Sub-Categories: Under **Diff**, use bolded sub-headings.
- • Details: Use nested bullet points for Video A and Video B. ANALYSIS FRAMEWORK

- 1. Subject Analyze the core subjects present in the videos.

Same

- • State if subject categories, count, or core attributes are identical. Note if high-level categories are the same. Diff (Only mention if different; otherwise, omit this section)
- • Subject Category: Differences in broad categories (Person, Animal, Object, Text).

- – Video A:
- – Video B:

- • Subject Count: Differences in total count and count per category.

- – Video A:
- – Video B:

- • Subject Attributes: Differences in appearance (age, gender, clothing) and state/pose.

- – Video A:
- – Video B:

##### 2. Style

Analyze the overall visual style using only the restricted list of descriptors: American comic style, Ukiyo-e, Anime, Pixel Art, Ghibli Style, Cyberpunk, Steampunk, Low Poly, Voxel Art, Minimalist, Flat Design, Retro, Oil Painting, Watercolor, Sketch, Graffiti, Ink Wash Painting, Black and White, Monochromatic, CG Rendering, realistic(un-stylized).

Same

- • If the overall style is identical, describe it here. Diff (Only mention if different; otherwise, omit this section)
- • Describe each video’s style based on objective characteristics.

- – Video A:
- – Video B:

###### 3. Scene & Background Analyze the environment and setting. Same

- • If the scene (location, lighting) is identical, describe it here. Diff (Only mention if different; otherwise, omit this section)
- • Location: indoor/outdoor, urban/natural.
- • Lighting: daylight/artificial, time of day.
- • Major Elements: key background objects.

###### 4. Camera Work / Cinematography Analyze camera language and shooting techniques. Same

- • If aspects of cinematography are identical, describe it here. Diff (Only mention if different; otherwise, omit this section)
- • Shot Scale Sequence: long, medium, close-up.
- • Camera Movement Sequence: pan, tilt, dolly, static.
- • Angle Sequence: eye-level, high-angle, low-angle.
- • Subject’s Orientation to Camera: front, profile, back.

###### 5. Subject Motion Analyze the dynamic performance and actions of the subjects. Same

- • If core dynamic events and motion types are identical, describe them here. Diff (Only mention if different; otherwise, omit this section)
- • Core Dynamic Event/Motion Type: running, jumping, expression changes.
- • Interaction: Between subjects or with objects.
- • Motion Details: Direction, speed, trajectory.

###### 6. Positional Relationship Analyze the relative spatial relationship between elements. Same

- • If relative positions are identical, describe it here. Diff (Only mention if different; otherwise, omit this section)
- • Indicate if the scene is flipped or mirrored.
- • Spatial Arrangement: Changes in relative positions between subjects and background elements.

##### 7. Playback Technique

Analyze special playback manipulations. Use only: "slow-motion", "fast-forward", "reverse", or "no special playing techniques".

##### Same

- • If identical, do not include the Same section at all.

Diff (Only mention if different; otherwise, omit this section)

##### • Playback Manipulation:

- – Video A:
- – Video B:

#### G.2 Checklist

In the checklist generation stage, we leveraged diverse models. Specifically, we fed contrastive descriptions and prompts into these models to generate an initial checklist. The prompt is as follows:

Checklist Generation Prompt

#### ROLE & TASK

You are a highly precise Assessment Creator AI. Your task is to convert a comparative analysis text into a comprehension quiz. The quiz must rigorously test a user’s understanding of specific and concrete differences between two items. Your entire existence is governed by a single principle: radical objectivity. You are a machine for observing and reporting, not for interpreting. Before you write a single word, you must ask yourself: "Is this statement an undeniable, verifiable visual fact?" If the answer is anything less than a 100% certain "yes", you must discard the statement. This principle overrides all other instructions.

#### CRITICAL INSTRUCTIONS

- 1. Mandatory Comparative Framing

- • Every single question MUST explicitly mention and compare both items (e.g., "Video A" and "Video B") within the question itself.
- • Forbidden: Questions that only ask about one item are strictly prohibited.
- • Correct Example: Is it true that in the Video A and B, the person is assembling the same wooden chair?
- • Forbidden Example: Is the person in the Video A assembling a wooden chair?

- 2. AVOID LOGICAL CONTRADICTIONS Absolute prohibition of contradictions is strictly enforced! All annotations must maintain complete logical consistency.

- • Zero tolerance for conflicting information
- • Immediate flagging of logical inconsistencies The following is a contradiction that is not allowed:

{"class": "style", "question": "Is the visual style different in Video A and B?", "correct_answer": "no"} {"class": "style", "question": "Does the Video A have a black and white style , in contrast to the cyberpunk style of the Video B?", "correct_answer": "yes"}

Explanation: These statements conflict. The first claims the style is the same, while the second confirms a specific, significant difference.

- 3. Strict Adherence to Source Text: You MUST base every question and answer exclusively on the provided Input Text.
- 4. Question-Based Format: All items MUST be direct, closed-ended questions that can be definitively answered with "Yes" or "No".
- 5. Adaptive Question Quantity

- • For a brief ‘diff’ description, generate 0–2 questions.
- • For a detailed ‘diff’ description, generate 1–3 questions, each targeting a different specific detail.

- 6. Every question generated MUST be followed by its correct standard answer: Yes or No.
- 7. Module Design Requirements Organize the quiz into two modules: Module 1 (Similarities) and Module 2 (Differences). The

Similarities module must have fewer questions. Both modules must follow the Guidelines for asking questions.

##### Module 1: Similarities (Same) Design

Purpose: Module 1 performs a reverse check. Questions are generated from similarities but are framed to ask “Are [they] different?”, so the correct answer is always "no".

Critical Rule for Module 1: No examples allowed. Do not use phrases like ‘In terms of...’, ‘involving...’, or ‘such as...’. Questions must be abstract and must not reveal specific details.

Question Characteristics:

- • Address categories, not specific attributes. Questions must be abstract.
- • Prefer subcategories over major categories.
- • Sparingly ask about playback techniques or perspective.

Question Source and Answer Rule: Questions are derived from same information and phrased as a query about difference. Therefore, the answer must always be "no".

##### Module 2: Differences (Diff) Design

- • Focus on specific, narrow points of primary difference.
- • FORBIDDEN: Broad, general comparison questions.
- • All Module 2 answers must be “yes”.
- • The difficulty MUST be layered (include simple and hard questions).
- • Principle of Specificity:

- – Correct: Regarding assembly tools, does the Video A feature a manual screwdriver while the Video B features a power drill?
- – Forbidden: Are the assembly methods different? (Too general).

#### Core Principle: Avoid Subjectivity

Questions must be based on observable facts. A question is wrong if it is:

- • Subjective: Uses words like "better" or "clearer."
- • Lacks a Clear Standard: Uses comparisons like "more" or "straighter."
- • Focused on Trivialities: Asks about a negligible difference.

#### INPUT

You will be given an Input Text in Markdown, divided into sections like ## Same and ## Diff.

#### OUTPUT

You MUST format your entire response as a single JSON object. The keys and nesting format are non-negotiable. The value for the class key must be from the provided guidelines.

{

"Similarities": [ {

"class": "subject", "question": "Is the subject category different between both

videos?", "correct_answer": "no", "answer": "", "explanation": ""

}, {

"class": "style", "question": "Do both videos have a different visual style?", "correct_answer": "no", "answer": "", "explanation": ""

}, {

"class": "camera", "question": "Are different cinematography techniques used in both

videos?", "correct_answer": "no",

"answer": "", "explanation": ""

}

], "Differences": [

{

"class": "subject", "question": "Does the Video A contain two subjects while the Video

B contains only one?", "correct_answer": "yes", "answer": ""

}, {

"class": "motion", "question": "In the Video A, is the guitarist actively playing

while in the Video B he holds a static pose?", "correct_answer": "yes", "answer": ""

}, {

"class": "background", "question": "Does the Video A show a whiteboard while the Video B

shows a nighttime window?", "correct_answer": "yes", "answer": ""

} ]

}

#### Guidelines for asking questions

- 1. Subject ("class": "subject") Can be asked from aspects like Type, Quantity, and Attributes (Appearance, Pose/State).

Core Principles: Avoid subjective language. Dynamic actions do not belong here. Questions about similarities and differences must not contradict each other.

- 2. Style ("class": "style") Use descriptors from a predefined list only (e.g., Anime, Cyberpunk, Oil Painting, Sketch, realistic).

Core Principles: Strict objectivity. Focus on the one or two most significant features. In Module 1, similarity questions must never target a specific style.

- 3. Background ("class": "background")

Analyzes environment, setting, and key background elements (Location, Atmosphere, Lighting, Weather, Key Objects).

Core Principles: Overlook minor details. Focus on major, unambiguous features. In Module 1, questions must not target a specific background.

- 4. Camera Work ("class": "camera")

Analyzes camera techniques (Perspective, Movement, Angle, Scale, Composition, Depth of Field, Orientation).

Core Principles: Overlook minor details. Only ask about significant differences. Base descriptions on identifiable techniques, not artistic "feeling."

- 5. Subject Motion ("class": "motion") Asks about motion and dynamics (Core events, event details, interactions).

Core Principles: Overlook minor details. Describe physical movement, not emotional intent (e.g., "raising a fist," not "threatening").

- 6. Positional Relationship ("class": "position")

Asks about the relative position between key subjects or between a subject and a significant background element.

Core Principles: Overlook minor details. Use clear, objective positional language relative to the viewer’s frame.

- 7. Playback Technique ("class": "playback technique") Concerns manipulation like "slow-motion," "fast-forward," and "reverse."

Core Principles: Use only the three standard terms. If both videos are at normal speed, avoid asking about this to prevent redundancy.

#### G.3 Evaluation

The following prompt instructs the model under test to generate similarities and differences between given items.

Prompt for the model under test to generate similarities and differences ROLE & TASK You are a Professional Video Analyst. Your task is to perform a comprehensive comparison of Video A and Video B. CRITICAL INSTRUCTIONS You must strictly follow these rules:

- 1. NO SPECULATION: If an element is unclear or ambiguous, omit it. Do not guess.
- 2. NO FABRICATION: Do not add details that do not exist in the videos.
- 3. CONSISTENCY: Ensure similarities and differences do not contradict each other.
- 4. DIFFERENTIAL DETAIL: Keep Similarities concise and direct. However, for Differences, you must provide comprehensive detail.
- 5. OPEN VOCABULARY: The items listed in parentheses within the framework are illustrative examples, not a restrictive list of options. Do not limit your description to these terms; use the most precise vocabulary available to describe what is actually seen.
- 6. Ignore negligible pixel-level noise or compression artifacts; focus on semantic and structural differences.

ANALYSIS FRAMEWORK Analyze the videos based on the following 7 dimensions. Use this framework to populate your response:

- 1. Subject Type & Quantity: Person, animal, object, vehicle, architecture, text/logo (OCR: Transcribe visible text verbatim), etc. Count the subjects. Attributes: Age, gender, ethnicity, clothing, accessories, color, material, physical features. State: Pose, facial expression, object state.
- 2. Style Description: Analyze the visual style. Recommended Vocabulary (Reference Only): You may use terms such as: Realistic (unstylized), American comic, Ukiyo-e, Anime, Pixel Art, Ghibli Style, Cyberpunk, Steampunk, Low Poly, Minimalist, Flat Design, Retro, Oil Painting, Watercolor, Sketch, Graffiti, Ink Wash, Black and White, Monochromatic, CG Rendering.
- 3. Background Setting: Location type (indoor room, office, street, park, outdoor). Environment: Lighting (natural/artificial, bright/dim), Weather (sunny, rainy), Background objects (furniture, buildings, vehicles), Atmosphere.
- 4. Camera Specs: Perspective (1st/3rd person), Angle (high/low/eye-level), Shot Scale (close-up, medium, wide), Depth of Field, Shot Structure (Continuous Shot, Multi-shot Sequence,

- Transition), View (Front View, Side View, Back View and so on). Movement: Static, Pan, Tilt, Zoom, Dolly, Tracking.
- 5. Motion Action: Primary movements type, direction, speed, amplitude and trajectory of the subject. Interaction: How subjects interact with each other or objects. Event Sequence: The chronological order of key actions or changes in state.
- 6. Position Layout (Frame Composition): Center, left, right, foreground, background, Spatial Flipping (Horizontal Flip, Vertical Flip). Relation: Spatial relationship between the subject and key background elements.
- 7. Playback Technique Technique: Identify if the video uses slow-motion, fast-forward, reverse, or no special playing techniques (Play forward at normal speed).

OUTPUT FORMAT Your response must be organized by the categories below. Only include categories where relevant details or differences exist.

[Category Name] Similarities: [Describe what is consistent between the two videos] Differences: In Video A, [Description]. In Video B, [Description].

#### G.4 Judge

During the evaluation stage, we employed the GPT5-mini model to perform judgment tasks. The prompt is as follows:

Judge Protocol

Based on the description generated by the model, determine whether the answer to the following question should be “yes” or “no”, and provide a brief reason. Judgment Principles:

- 1. Default to Same: Unless the description explicitly states that there is a difference, you must default to considering it as the same, and answer the question based on this assumption.
- 2. Validating Differences: To conclude that something is different, rely on explicit content or reasonable logical inference. Strictly avoid over-interpretation.
- 3. Handling Generalizations: If the question uses broad or general adjectives (e.g., “general”, “overall”), focus on the holistic content and main idea rather than specific details or minor discrepancies.

Output Format: The entire response will be a single JSON object in the following format:

{

"answer": "yes/no", "explanation": "reason"

}

Input Format: [Model Description] – The model-generated description will be provided here [Question] – The specific question to be answered will be provided here

#### G.5 Error Analyst

During the error analysis stage, we employed the GPT5-mini model to categorize prediction errors. The prompt is as follows:

##### Error Analysis Protocol

My purpose is to act as an error analysis agent for video understanding models. Given a "Correct Answer", "Model’s Description", and "Model’s Predicted Answer", I will categorize the type of error and provide a brief justification.

My operational process is as follows: Input I will receive a [Correct Answer], [Model’s Description], and [Model’s Predicted Answer]. Analysis

I will compare the model’s output against the ground truth to identify the error type. Error Categories

I will classify errors into one of the following categories:

- 1. Hallucination (Non-existent Difference): The model claims a difference exists when the ground truth indicates similarity. The model invents details not supported by the actual video content.
- 2. Missed Detection (Incomplete): The ground truth confirms a specific difference exists, but the model’s description fails to mention it or explicitly claims the videos are the same.
- 3. Self-Contradiction: The model’s description contains information supporting the correct answer, but reaches the wrong conclusion. Or the description contains internally inconsistent statements.
- 4. Vague / Imprecise: The model detects some change but provides insufficient detail to answer the specific question. The description is too generic to address what was asked.
- 5. Reasoning Error: The model correctly identifies all visual elements, but fails in the logical inference step to reach the correct answer. The evidence is accurate but the final judgment is flawed.

##### Output

My entire response will be a single JSON object in the following format:

{

"error_type": "category_name", "explanation": "detailed_reasoning"

}

I am now ready to receive the [Correct Answer], [Model’s Description], and [Model’s Predicted Answer] to perform this analysis.

- G.6 Training set Prompts We automatically annotated the training data with the following prompt:

AI Video Analysis Engine (Comparison Mode)

SYSTEM PROMPT: AI Video Analysis Engine

ROLE

You are a highly specialized AI Video Analysis Engine. Your sole purpose is to compare Video A and Video B and generate a strictly formatted JSON output. You must act as a dispassionate observer, recording only visually verifiable facts.

##### CRITICAL CONSTRAINTS

- 1. Output Format: Return ONLY a single, valid JSON object. No markdown code blocks, no conversational text, no preambles.
- 2. JSON Safety: Do NOT use double quotes (") inside the specific string values. Use single quotes (’) if necessary to quote text inside the description. Ensure the JSON is valid and parsable.

- 3. Objectivity: Describe ONLY what is visually verifiable. Do not infer emotions (e.g., ‘happy’, ‘scary’). Use descriptive terms (e.g., ‘smiling’, ‘low-key lighting’).
- 4. No Timestamps: Never mention specific seconds or frame numbers.
- 5. Language: English ONLY.

LOGIC FOR ANALYSIS Similarity Logic (Macro-Aggregation)

- • Goal: Summarize shared elements concisely.
- • Rule: If a Main Dimension (e.g., Background) is functionally identical in both videos, describe it as a unified whole (e.g., ‘Both videos take place in a sunny park’). Do not list every sub-detail unless necessary to establish the context.

##### Difference Logic (Structured Contrast)

- • Goal: Highlight specific changes accurately.
- • Rule: You must identify which of the 7 Dimensions have changed.
- • Precision (Granularity): Be extremely specific. Avoid generic terms. (e.g., ‘Video A features a casual white t-shirt vs Video B features a formal black tuxedo’).
- • Format: For each changed dimension, use the strict pattern: [Dimension Name]: Video A [description] vs Video B [description].

##### ANALYSIS FRAMEWORK (7 DIMENSIONS)

- 1. Subject: Type, Quantity, Appearance (Age, clothes, colors), Pose/Expression.
- 2. Style (STRICT VOCABULARY): Realistic, Anime, Cyberpunk, CG Rendering, etc.
- 3. Background: Location, Lighting, Weather, Key static objects.
- 4. Camera Work: Perspective, Angle, Shot Scale, Movement, Depth of Field.
- 5. Motion: Specific actions, Speed, Direction of movement.
- 6. Positional Relationship: Static spatial layout (Left side, Center, Behind object).
- 7. Playback Technique: [ ‘slow-motion’, ‘fast-forward’, ‘reverse’, ‘normal speed’ ]

JSON OUTPUT TEMPLATE { "similarity": "...", "difference": "..." }

#### G.7 Video Filtering Prompts

For the Ditto-1M dataset, we utilized Qwen3-VL-32B to assist in video filtering, using the following prompt:

Post-Edit Video QA

Objective: Assess video quality. Output strictly in JSON. Rejection Standards (Result: “NO”):

- • Significant Deviation: The target video content drastically differs from the source video, or exhibits severe visual inconsistencies between scenes.
- • Visual Artifacts: Visible pixelation, macro-blocking, heavy compression, color banding, or unintentional blurring.
- • Playback Glitches: Tearing, flickering, random black/green frames, stuttering, or content that remains static.
- • Editing Errors: Jarring cuts, unfinished transitions, incorrect aspect ratios, or obstructive overlays.

JSON Output Format: {

"result": "YES", // YES if clean, NO if defects found "reason": "Brief explanation of the defect or Pass"

}

- G.8 Frame Stitching Video Generation This section details all the prompts used during the controlled synthetic generation pipeline.

##### System Prompt: Visual Style Architect

Role: You are an AI expert in Visual Style Transfer designed for Image-to-Image editing workflows. Your goal is to analyze the input image internally and generate a concise style transformation prompt. Core Directives:

- 1. Internal Analysis: Identify the main subject (e.g., "a cat", "a building") and current lighting purely to select the best matching style.
- 2. Style Selection: Pick the single most suitable style from the library that enhances the subject’s characteristics.
- 3. Prompt Construction: Generate a prompt that combines the subject with strong stylistic keywords. Crucially, append strict constraints to prevent structural changes.

##### Available Styles:

American Comic, Ukiyo-e, Anime, Pixel Art, Ghibli, Cyberpunk, Steampunk, Low Poly, Voxel Art, Minimalist, Flat Design, Retro, Oil Painting, Watercolor, Sketch, Graffiti, Ink Wash, Black and White, Monochromatic, CG Rendering, Photorealistic.

Output Format: Selected Style: [Style Name] Editing Prompt: "[Simple Subject Keyword] in [Selected Style] style, [3-4 Style-specific Visual Descriptors], strict adherence to original composition and poses." Example: Input: (Image of a car on a road) Selected Style: Cyberpunk Editing Prompt: "A car on a road in Cyberpunk style, neon lighting, chrome reflections, futuristic atmosphere, strict adherence to original composition and poses."

System Prompt: Visual Weather Architect Role: You are an Image Editing Prompt Generator. Your task is to analyze an uploaded image and generate a descriptive English prompt to change the weather. Core Logic:

- 1. Analyze: Identify the subject, pose, clothing, and composition.
- 2. Select Weather: Choose a specific weather condition from the library below.
- 3. Generate: Create a prompt that combines the original scene with the new weather atmosphere.

##### Weather Library:

- • Clear Sunny: Clear blue sky, harsh bright sunlight.
- • Heavy Rain: Dark grey sky, pouring rain, splashing water, puddles.
- • Cloudy: Flat grey clouds covering the sky, soft diffused light.
- • Snowy: Falling snowflakes, ground covered in white snow.
- • Sunny Rain: Bright sunshine shining while it is raining.

- • Foggy: Thick mist obscuring the background, low visibility.

Prompt Formula: [Description of Subject and Scene] + [New Weather Details] + [Structure Preservation Constraint] Preservation Rules: Use natural phrases to keep the original content:

- • "...while strictly maintaining the exact same composition, pose, and layout."
- • "...keeping the original subject and camera angle unchanged."

##### Output Format:

- • Target Weather: [Selected Weather]
- • Prompt: [The Final English Prompt]

We utilized the generated prompts and the original image to generate the edited image. Then, we stitched these images together to create a video, using the image as the first frame. The prompt is as follows:

##### Image-to-Video Generation Prompt

Cinematic motion, bring the image to life naturally. Subtle and fluid movement, maintaining strict fidelity to the input image. High temporal consistency, seamless visual flow, no morphing, no distortion. Consistent lighting and exposure, preserved subject identity, smooth camera motion, natural physics, continuous action.

### H Category Framework

The detailed category framework is presented in Tables 7 and 8. Table 7 covers Subject, Style, and Background attributes. Table 8 presents Motion, Position, Camera, and Playback Technique attributes.

Table 7: Fine-grained subcategories for Subject, and Style.

Subject

Category Description Type Subject classification: persons, animals, plants, vehicles, buildings, virtual characters. Count Total number of subjects and per-category counts. Appearance Physical attributes: age, gender, ethnicity, physique, facial features, hairstyle, makeup. Clothing Attire and accessories: hats, glasses, jewelry, masks. Pose Body posture (standing, sitting), hand gestures, facial expressions. State Physical condition of inanimate objects (open/closed, broken). Material Material composition and texture: metal, wood, fabric, plastic. Color Dominant colors of objects, clothing, or skin tones. OCR Visible text transcription: signage, subtitles, logos, documents.

Style

Category Description Restricted Style Descriptors

Each video is categorized into one predefined style: Traditional & Fine Art: Oil Painting, Watercolor, Sketch, Ink Wash, Ukiyo-e, Graffiti. Digital & Graphical: CG Rendering, Pixel Art, Voxel Art, Low Poly, Minimalist, Flat Design. Pop Culture & Thematic: Anime, Ghibli, American Comic, Cyberpunk, Steampunk, Retro. Visual Tone: Black and White, Monochromatic, Realistic.

Table 8: Subcategories for Background, Motion, Position, Camera, and Playback Technique.

Background

Category Description Location Scene type: office, street, park, indoor/outdoor environment. Atmosphere Mood or ambiance: festive, quiet, tense. Lighting Light source, time of day, style, brightness level. Weather Environmental conditions: sunny, rainy, snowy, foggy. Key Background Objects

Prominent objects forming the main visual setting.

###### Motion

Category Description Type Primary movement or action sequence. Interaction Relationships between subjects. Direction Spatial orientation: moving left, approaching, receding. Speed Velocity: static, slow, fast, accelerating. Amplitude Magnitude, range, or force of motion. Trajectory Path of motion: linear, curved, circular.

Position

Category Description Subject-Object Relation

Spatial arrangement between subject and background elements.

Relative positions and distances between multiple subjects.

Subject-Subject Relation

Spatial Flipping Geometric transformations: horizontal or vertical mirroring.

###### Camera

Category Description Scale Subject’s frame size: close-up, medium, wide shot. Movement Camera motion: pan, tilt, zoom, dolly. Orientation Subject’s direction relative to camera: front-facing, side profile. Angle Vertical camera position: eye-level, high angle, overhead. Composition Subject placement within frame: centered, left/right-aligned. Depth of Field Background clarity: blurred (shallow), sharp (deep). Perspective Point of view: first-person, third-person. Shot Count Single continuous shot vs. multi-shot sequence.

###### Playback Technique

Category Description Slow Motion Reduced playback speed to emphasize details. Fast Motion Increased playback speed to compress timeline. Reverse Backward playback from end to start.

