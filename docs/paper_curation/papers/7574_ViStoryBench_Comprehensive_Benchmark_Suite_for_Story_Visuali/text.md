#### ViStoryBench: Comprehensive Benchmark Suite for Story Visualization

Cailin Zhuang1,2,3,∗ Ailin Huang2,∗,† Yaoqi Hu3,∗ Jingwei Wu2 Wei Cheng2,† Jiaqi Liao2,4 Hongyuan Wang2 Xinyao Liao2 Weiwei Cai2 Hengyuan Xu2 Xuanyang Zhang2 Xianfang Zeng2 Zhewei Huang2,‡ Gang Yu2,‡ Chi Zhang4,‡

1 ShanghaiTech University 2 StepFun 3 AIGC Research 4 AGI Lab, Westlake University

##### Dataset Code Project Page Story Explorer

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

# arXiv:2505.24862v5[cs.CV]29Mar2026

[Figure 5]

## 🎞

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

###### Dataset Construction Benchmark Evaluation

Story Visualization Benchmark

MLLM-driven Structured Script Creation

Multi-expert Omni-dimension Assessment

###### Raw Story

[Figure 13]

[Figure 14]

CIDS Score

[Figure 15]

[Figure 16]

[Figure 17]

The original global short story

Succeed Matches

###### Cross Character ID Similarity

Character Description

Bbox: [p1, p2, p3, p4] Cross-Sim Score: 0.85 Self-Sim Score: 0.91

[Figure 18]

[Figure 19]

Generated and reference id similarity

[Figure 20]

[Figure 21]

[Figure 22]

Global character appearance description

Grounding

###### Self Character ID Consistency

|Multi-grained Task Decomposition|
|---|

[Figure 23]

Generated and generated id consistency

Shot Perspective Design

[Figure 24]

recognition OCCM

[Figure 25]

[Figure 26]

MLLM Human

###### Onstage Character Count Matching

[Figure 27]

[Figure 28]

The current shot size, camera

Superﬂuous Match Omissive

Correctness of the number of generated characters

[Figure 29]

###### angle, etc Characters Appearing

###### Scene-level Consistency

[Figure 30]

[Figure 31]

Scene setting prompt alignment

###### Cross Style Similarity

|Professional Fields Knowledge|
|---|

###### CSD Score

The characters appearing in Setting Description the current shot

Gen-ref style similarity

###### Shot-level Consistency

Cross-Sim Score: 0.88 Self-Sim Score: 0.90

[Figure 32]

[Figure 33]

###### Self Style Consistency

Shot design prompt alignment

Gen-gen style consistency

|Multi-dimensional Information Isolation|
|---|

###### Interaction Actions Consistency

The scene setting of the current shot

[Figure 34]

Global actions prompt alignment

###### Shot VLM Score

###### Scene VLM Score

###### Individual Action Consistency

[Figure 35]

###### Static Shot Description

3/5

2/5

[Figure 36]

Local action prompt alignment

[Figure 37]

[Figure 38]

|Visual-friendly Description|
|---|

Objective event and action description of the current shot

[Figure 39]

[Figure 40]

###### Inter. Action Score

Ind. Action Score

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

3/5

4/5

[Figure 45]

###### Plot Correspondence

|Contextual Narrative Modeling|
|---|

[Figure 46]

[Figure 47]

The subjective plot description and dialogue of the current shot

Aesthetic Quality Copy-Paste Detection Diversity

Human Evaluation

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

###### Character Reference Images Generated Shots

Figure 1. Overview of ViStoryBench. Dataset Construction : We employ structured prompt engineering with 5 strategies to convert an LLM into a controllable visual narrative script generator, with human proofreading for reasonableness and manual collection of character reference images. Benchmark Evaluation : Based on this dataset, we develop metrics for character/style similarity and multi-grained prompt alignment, evaluated through a hybrid framework combining expert models and VLMs.

##### Abstract

features richly annotated multi-shot scripts derived from curated stories spanning literature, film, and folklore. Large language models assist in story summarization and script generation, with all outputs verified by humans for coherence and fidelity. Character references are carefully curated to maintain consistency across different artistic styles. ViStoryBench proposes a suite of multi-dimensional automated metrics to evaluate character consistency, style similarity, prompt alignment, aesthetic quality, and artifacts like copy-paste behavior. These metrics are validated through human studies and used to assess a broad range of opensource and commercial models, enabling systematic analysis and encouraging advances in visual storytelling.

Story visualization aims to generate coherent image sequences that faithfully represent a narrative and match given character references. Despite progress in generative models, existing benchmarks remain narrow in scope, often limited to short prompts, lacking character references, or single-image cases, failing to reflect real-world narrative complexity and obscuring true model performance. We introduce ViStoryBench, a comprehensive benchmark designed to evaluate story visualization models across varied narrative structures, visual styles, and character settings. It

* Equal contribution. † Project leads. ‡ Corresponding authors.

##### 1. Introduction

In recent years, story visualization [6, 96, 111] has emerged as a rapidly evolving research field, aiming to generate sequences of visually consistent images that faithfully convey a narrative while adhering to character references, thereby creating an engaging storytelling experience. This capability holds broad potential in areas such as entertainment and education, especially with recent advances in generative models [41, 48, 49, 57, 74, 84, 94, 95] enabling opendomain visual creation.

Compared to general image or video generation evaluation [26, 29, 30, 77, 80], story visualization evaluation involves more dimensions, including not only visual quality and diversity, but also character consistency and narrative alignment. However, recent studies [89, 92, 96] often adopt limited evaluation metrics without a unified standard, and their constrained test scenarios fail to reflect real-world applicability. There is thus a pressing need for a comprehensive benchmark to systematically evaluate and advance story visualization methods under in the wild. In this paper, we present ViStoryBench, a comprehensive benchmark for story visualization with the following key features:

Multifaceted Dataset Creation: We construct a diverse dataset spanning multiple story genres (e.g., comedy, horror) and visual styles (e.g., anime, 3D rendering) to enable comprehensive model evaluation across narrative and aesthetic dimensions. We carefully curate 80 story segments with 344 characters to balance narrative structures and visual elements. ViStoryBench includes stories with both single and multiple protagonists, testing the models’ ability to maintain character consistency. Additionally, it features complex plots and intricate world-building, challenging models to generate accurate visual content.

Comprehensive Evaluation Metrics: In addition to conventional image generation metrics, such as image quality, diversity, and prompt alignment, we also quantify several attributes particularly vital to story visualization. These encompass stylistic consistency across the generated sequence, alignment of character actions and interactions with textual descriptions, visual novelty (beyond mere copypasting of references), and correctness of character sets in each scene. Our benchmark incorporates 12 automated evaluation metrics. This structured and multifaceted framework enables researchers to thoroughly identify both the strengths and weaknesses of different models, thus foster-

ing targeted improvements.

Extensive Model Evaluation: We evaluate over 30 methods (including 25 base methods and their variants), analyzing the alignment between automated metrics and human evaluations to provide reliable model insights.

We will release the complete benchmark, detailed prompts for data construction, evaluation results for each model, and code to ensure reproducibility, aiming to facilitate future advancements in story visualization.

##### 2. Related Work

###### 2.1. Story Visualization

Recent advances in diffusion and auto-regressive models have significantly advanced multimodal long-story generation, with growing emphasis on visual consistency and cross-modal coherence.

For story image generation [3, 27, 40, 51, 55, 63, 70, 75, 76, 105, 106], methods have evolved from GAN-based methods [42] to diffusion-based techniques. Training-free methods like StoryDiffusion [111] leverage consistent selfattention, while UNO [90] and USO [88] promote content and style customization. Recent works integrate LLMs/VLMs for character and scene planning [6, 72, 79, 82], or use CoT for scene understanding [59], with extensions to story continuation [10, 98], next-shot generation [28], and comic creation [4, 5, 87]. Unified multimodal models [20, 66, 86] further support diverse generation tasks.

In story video generation [23, 31, 32, 37, 81, 102, 104, 107, 108], systems like MMStoryAgent [96], AnimDirector [43], MovieAgent [85] and FilmMaster [33] orchestrate multi-component pipelines. Recent trends shift toward holistic scene-level synthesis, with Captain Cinema [93], ShotAdapter [38], HoloCine [56], Sora2 [60] and Seedance [18] generating multi-shot narratives in single passes. We focus on the image-level evaluation of these systems, by extracting video keyframes as storyboards.

The field expands into 3D storytelling [35, 83] while commercial platforms [20, 54, 58] and closed-source MLLMs [9, 11, 21, 36] bridge research with applications. Despite progress, challenges persist in multi-image coherence, long-range dependency modeling, fine-grained control, and complex narrative alignment. ViStoryBench addresses these gaps through standardized evaluation, enabling systematic analysis and driving future advancements

- Table 1. Comparison with Related Benchmarks. ViStoryBench offers the most comprehensive coverage in terms of styles, evaluation metrics, and the number of methods evaluated, highlighting its suitability for diverse storytelling assessments.

Benchmark #Stories #Prompts / Shots #Ref Images #Styles #Metrics #Methods Evaluated

DreamBench++ [64] - 1,350 150 3 2 7 OmniContext [86] - 400 790 2 2 6 VinaBench [17] 85 1,181 - 6 12 3

###### ViStoryBench 80 1,317 509 10 12 25

|Range: 1 to 10| |
|---|---|
| | |
|Median: 4| |
| | |
| | |
| | |

Full Dataset Lite Dataset

Horror War Adventure Mystery Historical Sci-Fi Social Issues

10

ReferenceCharacters

Realistic

Chinese

Photorealistic

8

6

Indian

Action manga Anime

4

Fantasy Romance Fairy Tale

Euro-American

Unrealistic

Line art Cartoon Vintage 3D/Voxel Flat vector illustration Photoreal fantasy CG Religious art

2

Islamic Japanese African

Fable

0

0 5 10

0 10 20 30 40 50 60 70 80

Stories

Count

(a) Cultural and Style Distribution.

(b) Theme Distribution (Lite vs. Full).

(c) Casts per Story.

- Figure 2. ViStoryBench Dataset Statistics Overview. (a) Distribution of story style and cultural origin of our dataset. (b) Theme distribution comparison of Full and Lite subset, showing high statistical similarity. (c) Distribution of reference characters number per story. See Appendix C, E for details. in visual storytelling.

are consistent with each other, and Ci = (Ti,Si). Next, we provide m storyboard shot descriptions: Shot1, Shot2, ..., Shotm. Each Shoti includes a text description that contains the following components:

- 2.2. Datasets and Benchmarks

We summarize story visualization datasets [24, 34, 42, 45, 98–100] in Appendix C, showing significant advances in scale, resolution, and diversity.

Several benchmarks have been proposed for video generation and visual storytelling. StoryBench [1] evaluates text-to-video models across multiple tasks, while VBench [110] and Video-Bench [25] focus on single-shot videos. ShotBench [46] introduces camera-focused QA evaluation, and DreamSim [16] provides perceptual similarity metrics. Recent benchmarks address specialized aspects: SFD [19] targets long-form genre-diverse stories, StoryEval measures story completion using vision-language models, MovieBench [91] supports long-video analysis, RISEBench [109] covers reasoning-informed editing, and VinaBench [17] enhances visual narrative fidelity with commonsense and discourse annotations.

Compared to existing benchmarks (Table 1), ViStoryBench differs significantly from DreamBench++ [64] and OmniContext [86], which focus on single-image generation with limited style variation. While VinaBench also targets storytelling, it lacks character references and supports fewer styles. ViStoryBench introduces a reference-based, multiimage benchmark with broader style coverage and stronger support for character consistency for comprehensive story visualization evaluation.

- 3. ViStoryBench

- ⃝1 Setting Description: A description of the current scene’s setting
- ⃝2 Plot Correspondence: The story segment from the original narrative corresponding to this shot.
- ⃝3 Onstage Characters: A list of characters present in the current shot.
- ⃝4 Static Shot Description: A description of the static actions or positions of characters and objects in the frame, representing a fixed visual state.
- ⃝5 Shot Perspective Design: Cinematographic information, including shot size, shot type, and camera angle.

The objective of the story visualization task is to generate a sequence of images I1...Im that faithfully represent the described storyboard shots, adhering to the provided prompts and character information. This involves accurately depicting the characters, their actions, the scene settings, and the specified camera perspectives. The specific quantitative evaluation methods are described later.

###### 3.2. Source Data

Story and Script. To ensure narrative diversity, we manually curate 80 story segments from a wide range of sources, including film and television scripts, literary classics, world folklore, novels, and picture books. For lengthy stories, we employ LLMs to assist in summarization, producing concise versions of several hundred words. Each story is then adapted into a script containing character descriptions and storyboards, also with LLM assistance. All LLM-generated content is manually reviewed to ensure narrative coherence and logical consistency.

- 3.1. Problem Definition

Based on some prior work [42, 44, 98], we design a story generation task with comprehensive input conditions. Given a story script, we first provide the appearance descriptions T1, T2, ..., Tn and corresponding images S1, S2, ..., Sn for n characters C1, C2, ..., Cn, where Ti and Si

Character Reference Images. For each character, we manually collect reference images from the Internet that

[Figure 55]

[Figure 56]

[Figure 57]

Copy Paste Detection

###### Onstage Character Count Matching

###### Prompt Alignment

[Figure 58]

|[Figure 59]<br><br>[Figure 60]|
|---|

|[Figure 61]|
|---|

Reference images

[Figure 62]

[Figure 63]

Copy-paste Non-Copy-paste

|1|Genera|[Figure 64]<br><br>[Figure 65]<br><br>ted Shot|
|---|---|---|

|[Figure 66]<br><br>[Figure 67]<br><br>1| |2<br><br>G|ene|rated Sho|3<br><br>t|
|---|---|---|---|---|---|

|[Figure 68]<br><br>[Figure 69]|
|---|

|[Figure 70]<br><br>|[Figure 71]|
|---|
|
|---|

[Figure 72]

[Figure 73]

###### <On-stage Characters>

[Figure 74]

1. Aladdin

Qingqing Mother

[Figure 75]

Mismatch: Superfluous

Amount Match

[Figure 76]

[Figure 77]

[Figure 78]

###### <Static Shot Description>

|[Figure 79]<br><br>[Figure 80]|
|---|

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

|[Figure 87]<br><br>[Figure 88]|
|---|

[Figure 89]

Qingqing sits by the hospital bed, holding Mother's hand, her expression sad and concerned. Mother lies in bed, pale-faced, her eyes gentle yet determined.

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

1 2

<On-stage Characters>

###### 1 2

| | |
|---|---|

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

- 1. Snow White
- 2. Prince
- 3. New Queen

3

Prompt Mismatch

Generated Shot

Generated Shot

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Qingqing lies in bed Mother sits by the hospital bed

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Mismatch: Omissive

Amount Match

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Self-Similarity

[Figure 114]

[Figure 115]

|[Figure 116]<br><br>[Figure 117]<br><br>|Qingqing|
|---|
<br><br>|Mother|
|---|
<br><br>Generated Shot|
|---|

Lack of style self-similarity Lack of character self-similarity

[Figure 118]

[Figure 119]

###### High style self-similarity

- Figure 3. Case Study of Failure in Story Visualization. Prompt Alignment : Alignment of character interaction, individual actions, camera design, and scene setting descriptions. Onstage Character Count Matching : Whether the number of characters in the generated shots matches the script. Copy-Paste Detection : Quantifies the tendency to replicate reference images rather than generating novel instances. Self & Cross Similarity : Style and identity consistency of generated characters with reference images and across shots.

align with the textual descriptions, ensuring a consistent visual style within each story. A small subset of images is generated using SDXL [65]. The final dataset comprises 344 characters and 509 reference images, categorized the 80 stories into 10 distinct style genres for stylistic diversity. Details on data collection, style distribution, and statistics are provided in Appendix C and visualized in Figure 2.

103] serves as our open-set object detector for cropping character regions. For feature extraction, we use ArcFace [12], AdaFace [39], and FaceNet [69] for realistic characters, and CLIP [67] otherwise. The CLIP-based model from CSD [71] is employed for style feature extraction. We use the Inception Score for diversity and the Aesthetic Predictor V2.5 for quality evaluation.

Next, we briefly describe the calculation procedure for each metric, with additional details provided in Appendix H, I, J. For clarity in presentation, both the cosine similarity scores and the original 0 − 4 rating scale are converted to a 100-point scale.

###### 3.3. Evaluation Metrics

Overview. We present a comprehensive suite of metrics to assess story visualization models across multiple key dimensions. To provide intuitive motivation for our evaluation metrics, Figure 3 illustrates typical failure cases in story visualization. The metrics cover the following aspects:

- ⃝1 Cross- and Self-Similarity: Assessing the resemblance between generated and reference images, as well as the consistency within the generated images themselves.
- ⃝2 Prompt Alignment: Measuring how well the generated images align with the storyboard descriptions provided in the prompts.
- ⃝3 Character Matching: We calculate the accuracy of the number of characters in each generated image.
- ⃝4 Aesthetic: We evaluate the aesthetic metric, generation quality and diversity of the generated results.
- ⃝5 Copy-Paste: We specifically design a Copy-Paste Detection metric to check if the model excessively replicates the character reference images.

Character Identification Similarity (CIDS). The CIDS computation pipeline comprises four stages:

- ⃝1 Character Detection: Grounding DINO crops character regions from reference and generated images. For character reference, Grounding DINO typically trims the edges. For generated images, however, it may fail entirely, returning an empty result;
- ⃝2 Feature Extraction: CLIP or face models convert crops to 512-d feature vectors;
- ⃝3 Character Matching: Similarity computation and bipartite matching to find optimal character correspondences;
- ⃝4 Scoring: Average cosine similarity of matched pairs yields the final score.

Preliminaries. We briefly introduce the core models and tools used in our evaluation pipeline. Grounding DINO [47,

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Reference Images

[Figure 125]

Shot1 Bboxs

Embedding Extractor

###### Embedding Extractor

Character A

|[Figure 126]<br><br>[Figure 127]|
|---|

|[Figure 128]<br><br>[Figure 129]|
|---|

|[Figure 130]<br><br>[Figure 131]|
|---|

|[Figure 132]<br><br>[Figure 133]<br><br>B|
|---|

Character B

[Figure 134]

[Figure 135]

Shot2 Bboxs

Image & Prompt

Character A

Character B

[Figure 136]

[Figure 137]

Character A

Character

Ref embedding

CrossSimilarity

###### Grounding-DINO

[Figure 138]

[Figure 139]

[Figure 140]

Self Similarity

Generated Shots

[Figure 141]

Image Crops CIDS Score

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Succeed Matches

[Figure 147]

Realistic? Non-Realistic?

Shot 1

Bbox: [p1, p2, p3, p4] Cross-Sim Score: 0.85 Self-Sim Score: 0.91

Matching

Shot 1 embedding Shot 2 embedding

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

Shot 2

CLIP

ArcFace

[Figure 154]

###### CIDS Score

[Figure 155]

Failed Matches

[Figure 156]

[Figure 157]

Bbox: null Cross-Sim Score: 0.0 Self-Sim Score: 0.0

[Figure 158]

Embedding Extractor

- Figure 4. Character Identification Similarity (CIDS) Metric. Evaluating both cross-similarity and self-consistency by detecting and cropping character regions from reference and generated images, then computing cosine similarity between matched character features.

Style Similarity. The CSD computation pipeline comprises four stages:

the low variance of our evaluation scores. The detailed methodology and results are provided in Appendix H.6.

- ⃝1 Image Encoding: Encoding each image using a styletrained CLIP vision encoder;
- ⃝2 Feature Extraction: Extracting style features through CSD layers;
- ⃝3 Similarity Computing: Computing pairwise cosine similarity between all style embeddings;
- ⃝4 Scoring: Averaging the scores over all valid pairs to produce the final score.

Onstage Character Count Matching (OCCM). We observe that various models struggle to generate the correct number of onstage characters as specified in the script, often leading to superfluous additions (hallucinations) or omitted figures. To quantify this critical capability, we introduce the Onstage Character Count Matching (OCCM) score. It is important to note that this metric relies on an upstream character detector to obtain the detected character count (D). Nevertheless, the OCCM formula itself is designed to provide a fair and robust measure of numerical consistency. The score is calculated as:

Prompt Alignment. Regarding the similarity between the generated images and the provided storyboard descriptions, we categorize them as follows:

- ⃝1 Scene Score: Overall correspondence between the generated scene and the narrative details provided in the storyboard’s static shot description, including setting, mood, and layout.
- ⃝2 Shot Score: Consistency between the depicted camera perspective (e.g., close-up, wide shot, over-theshoulder) in the generated image and the specified shot design in the storyboard.
- ⃝3 Character Interaction (CI): Alignment between the group-level interactions of characters in the generated image and the intended interactions described in the storyboard’s static shot description.
- ⃝4 Individual Actions (IA): Accuracy of the gestures, expressions, and poses of each character in the generated image relative to their described behavior in the static shot description.

|D − E| ϵ + E × 100% (1)

OCCM = exp −

where D is the detected number of characters, E is the expected count from the storyboard, and ϵ = 10−6 is a small smoothing factor to prevent division by zero. The detailed explanation of performance bounds and the design principle is provided in Appendix K.

Copy-Paste Detection. A common shortcut in story visualization is directly reusing the input character reference images, which compromises generation diversity and fidelity to the prompt. To quantify this behavior, we propose a geometrically-normalized Copy-Paste Rate metric. This metric assesses whether a generated character feature g is closer to its specific input reference feature r or to a feature from a different reference image t of the same character. Here, the second reference t serves as a proxy for a generalized target, helping to determine if the model is merely copying the input r instead of learning the character’s general appearance. The detailed explanation of the computational principle is provided in Appendix L.

We primarily use Gemini-3-Pro for automated evaluation. As shown in Figure 5, we have established Likert-scale questionnaires to evaluate the consistency of each generated image on a scale from 0 to 4.

To validate the reliability of this VLM-based approach, we conducted a rigorous stability analysis, which confirms

[Figure 159]

[Figure 160]

Evaluation Pipeline

###### Shots Cropped Images

###### Generated Shots

###### Gemini/Qwen

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

…

[Figure 165]

[Figure 166]

Prompt Alignment (0 - 4)

|Shot 1|Shot 2 Shot N|
|---|---|
|<Plot Correspondence> <Shot Perspective Design> <On-stage Characters> <Setting Description> <Static Shot Description><br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]| |

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Scene score Camera score Character interaction score Individual action score

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Scoring Samples

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

###### <Shot Perspective Design>

###### <Setting Description>

###### <Static Shot Description>

###### <Static Shot Description>

Daytime, Aunt Polly's room, contains

Rostov curls up by the campﬁre, shivering,expression pained, attempting to sleep

Wide shot, high-angle shot

The big rabbit picks up the little rabbit and swings him overhis head.

a bed, a closet, and a table with a jar of preserves on it.

|[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>UNO|
|---|

|[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]<br><br>[Figure 193]<br><br>UNO|
|---|

|[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>GPT-4o|
|---|

|[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>StoryDiffusion|
|---|

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

###### Scene score: 4

Camera score: 4

Character interaction score: 4

###### Individual action score: 4

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

|[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>[Figure 218]<br><br>StoryAdapter|
|---|

|[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]<br><br>[Figure 222]<br><br>[Figure 223]<br><br>Seed Story|
|---|

|[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]<br><br>[Figure 228]<br><br>StoryDiffusion|
|---|

|[Figure 229]<br><br>[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]<br><br>UNO|
|---|

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

Scene score: 2

Camera score: 2

Character interaction score: 0

Individual action score: 2

[Figure 241]

[Figure 242]

[Figure 243]

- Figure 5. Prompt Alignment Evaluation. Based on the descriptions, Scene Score, Shot Score, Character Interaction and Individual Action are evaluated via Gemini-3-Pro for best evaluation accuracy, Qwen3-VL [97] for reproducibility.

Image Quality. We evaluate visual quality through two complementary metrics averaged across all images:

For image generation, we assess StoryGen [45], TheaterGen [6], StoryDiffusion [111], SEED-Story [98], StoryAdapter [52], UNO [89], OmniGen2 [86], CharaConsist [82], and QwenImageEdit-2509 [66]. We also evaluate multi-modal large models, including GPT-4o [36], Gemini2.0 [11], Gemini-2.5 [9] (Nano Banana), and Seedream4.0 [20]. For video generation, we test Vlogger [114], AnimDirector [43], MMStoryAgent [96], MovieAgent [92], and the native multi-shot model Sora2 [60]. Additionally, we include a simple Copy-Paste Baseline that programmatically places reference images on a 1080p canvas. Due to resource constraints, commercial software results, including MOKI [54], MorphicStudio [58], AIbrm [50], ShenBi [53], Typemovie [78], and Doubao [2], are reported only on the Lite version of the dataset using their May 2025 releases.

- • Inception Score (Inc) [68] measures diversity and clarity using an Inception V3 backbone.
- • Aesthetic Score [13] assesses aesthetic perceptual quality on a 1-10 scale via Aesthetic Predictor V2.5 [13] (a SigLIP-based predictor [101]), where scores below 5.5 typically indicate blurry, noisy, or unappealing images.

##### 4. Experiments and Analysis

###### 4.1. Experimental Setup

To address the high costs associated with user studies and commercial platform evaluations, we introduce ViStoryBench-Lite, a strategically curated subset for efficient yet representative assessment. This one-quarter subset of the full ViStoryBench was meticulously constructed to preserve statistical alignment with the full dataset in text styles and character reference distributions, as illustrated by the theme distribution alignment in Figure 2. It includes 20 carefully selected stories with 36 animated, 41 realistic, and 43 non-human characters, closely mirroring the full dataset’s composition. As detailed in Appendix E, performance correlation between the Lite and full versions is consistently high, validating its reliability as a cost-effective alternative for large-scale evaluation.

Detailed adaptations for the varying method (Image and video) requirements are discussed in Appendix F. For video methods without intermediate images, we use the keyframe of each shot. Most methods produce 1080p outputs, except for some like Gemini-2.0 with limited resolution control.

A public leaderboard will be maintained to foster community competition, ranking models by averaging metric ranks for balanced evaluation.

User Study. To evaluate the consistency and aesthetic quality of the generated images, we conduct a user study involving participants who assessed the results across three dimensions: Environment Consistency, Character Identifi-

In our main experiments (Table 2), we evaluate a diverse range of image and video generation methods.

- Table 2. Quantitative Results of Various Story Visualization Methods on ViStoryBench and ViStoryBench-Lite. Results highlighted with a gray background are excluded from ranking. the Copy-Paste Baseline directly pastes the character reference image as the output. For certain methods, we evaluate multiple inference configurations and report all corresponding results. indicate the first, second, third, fourth, and fifth performance, respectively. We list full results on ViStoryBench-Lite in Appendix E. CSD: Style Similarity; CIDS: Character Similarity; PA: Prompt Alignment Score (CI: Character Interaction, IA: Individual Action); CM: OCCM; Inc: Inception Score; Aes: Aesthetics Score; CP: Copy-Paste. : With image reference; : Only text input; : Auto-regressive mode; superscript k means scale=k. Note: PA scores are based on Gemini-3 Pro.

CSD↑ CIDS↑ PA↑ CM↑ Inc↑ Aes↑ CP Cross Self Cross Self Scene Shot CI IA Avg.

Method Model

The following results are obtained on ViStoryBench Copy-Paste Baseline - 0.728 0.712 0.929 0.984 0.34 1.75 0.70 0.81 0.90 89.4 6.71 4.48 0.474

###### Story Image Method

StoryGen [45] SD1.5 0.379 0.540 0.428 0.576 0.58 1.95 0.53 0.49 0.89 51.1 8.73 4.02 0.246 StoryGen [45] SD1.5 0.371 0.531 0.417 0.568 0.63 2.05 0.53 0.40 0.90 50.8 8.89 4.02 0.228 StoryGen [45] SD1.5 0.283 0.580 0.414 0.593 0.53 2.05 0.48 0.39 0.86 41.1 7.31 3.74 0.227 TheaterGen [6] SD1.5 0.184 0.392 0.348 0.578 1.99 1.86 0.46 0.33 1.16 55.4 14.89 4.90 0.189 StoryDiffusion [111] SDXL 0.269 0.628 0.397 0.622 2.62 2.30 1.33 1.15 1.85 62.9 15.72 5.76 0.181 StoryDiffusion [111] SDXL 0.340 0.547 0.436 0.565 1.34 2.54 1.39 1.30 1.64 57.4 10.06 5.13 0.216 SEED-Story [98] SDXL 0.227 0.748 0.287 0.587 1.43 1.64 0.41 0.22 0.93 44.4 6.30 3.82 0.186 Story-Adapter [52] 0 SD1.5 0.456 0.548 0.460 0.605 1.52 2.60 1.81 1.66 1.90 69.0 12.98 4.99 0.198 Story-Adapter [52] 5 SD1.5 0.325 0.737 0.401 0.626 1.42 2.56 1.58 1.43 1.75 59.3 13.73 4.89 0.194 Story-Adapter [52] 0 SD1.5 0.280 0.462 0.400 0.538 1.57 2.70 1.77 1.63 1.92 63.1 16.34 5.17 0.195 Story-Adapter [52] 5 SD1.5 0.318 0.733 0.395 0.624 1.41 2.54 1.53 1.36 1.71 59.6 13.13 4.90 0.193 UNO [90] FLUX1 0.391 0.602 0.485 0.620 3.11 2.40 1.92 1.78 2.30 74.2 12.40 5.23 0.234 OmniGen2 [86] DiT 0.454 0.600 0.548 0.647 3.16 2.68 2.14 1.98 2.49 70.2 11.05 5.25 0.275 CharaConsist [82] FLUX1 0.282 0.553 0.315 0.519 3.15 2.45 1.83 1.53 2.24 57.9 13.80 5.88 0.172 QwenImageEdit-2509 [66] DiT 0.381 0.593 0.475 0.574 3.21 2.32 2.39 2.13 2.51 59.8 13.42 5.50 0.218

[Figure 244]

###### Story Video Method

Vlogger [114] SD1.4 0.201 0.407 0.346 0.548 1.15 2.32 1.56 1.34 1.59 75.4 10.29 4.28 0.194 Vlogger [114] SD1.4 0.259 0.453 0.362 0.554 1.18 2.28 1.55 1.39 1.60 76.6 9.77 4.28 0.200 AnimDirector [43] SD3 0.288 0.510 0.401 0.578 3.30 2.34 2.48 2.09 2.55 67.4 12.02 5.59 0.212 MMStoryAgent [96] SDXL 0.238 0.669 0.388 0.596 2.41 2.12 1.25 1.00 1.69 61.5 9.09 5.88 0.198 MovieAgent [85] SD1.5 0.193 0.502 0.360 0.560 0.92 1.83 0.76 0.61 1.03 63.6 11.61 4.63 0.198 MovieAgent [85] SD3 0.299 0.479 0.400 0.544 3.14 2.50 2.53 2.00 2.54 64.6 14.99 5.32 0.209

The following results are obtained on ViStoryBench-Lite Copy-Paste Baseline - 0.911 0.994 0.550 0.735 0.42 1.60 0.76 0.85 0.91 92.76 5.46 4.39 1.000

###### Commercial Platform

MOKI [54] - 0.214 0.694 0.372 0.621 2.29 1.56 0.58 0.45 1.22 45.96 10.36 5.79 0.211 MorphicStudio [58] - 0.577 0.628 0.603 0.677 3.01 2.31 1.77 1.32 2.10 60.79 9.00 4.96 0.234 AIbrm [50] - 0.412 0.730 0.557 0.740 3.06 2.18 1.67 1.55 2.12 75.53 9.53 5.72 0.223 ShenBi [53] - 0.275 0.575 0.418 0.585 3.49 2.35 2.65 2.11 2.65 61.33 11.60 5.07 0.197 Typemovie [78] - 0.325 0.646 0.464 0.621 2.34 2.17 1.62 1.40 1.88 74.14 11.15 5.32 0.168 Doubao [2] - 0.367 0.695 0.446 0.642 3.88 2.41 3.23 2.65 3.04 65.23 9.88 5.61 0.255

###### Multi-modal Large Model

GPT-4o∗ [36] - 0.481 0.680 0.420 0.522 3.82 2.82 3.58 3.12 3.34 69.33 9.02 5.49 0.209 Gemini-2.0∗ [11] - 0.361 0.573 0.573 0.677 3.26 2.46 2.43 2.00 2.54 74.82 10.12 4.91 0.266

- Gemini-2.5∗ [21] - 0.447 0.657 0.553 0.642 3.89 2.32 3.26 2.90 3.09 64.86 10.54 5.61 0.255

- Gemini-3.0 Pro∗ [22] - 0.385 0.622 0.581 0.653 3.94 2.58 3.71 3.25 3.37 59.97 12.50 5.54 0.244 Seedream-4.0 [20] - 0.369 0.585 0.280 0.539 3.82 2.59 3.20 2.68 3.07 49.45 12.12 5.21 0.201 Sora2∗ [60] - 0.515 0.713 0.766 0.839 3.08 2.76 2.91 2.50 2.81 81.42 6.53 4.72 0.158 Sora2∗ [60] - 0.365 0.685 0.364 0.561 3.01 2.83 2.96 2.33 2.78 72.67 9.68 4.52 0.158

cation Consistency, and Subjective Aesthetics Score. Environment Consistency focused on whether scenes corresponding to the same environment description appeared visually cohesive. For Character Identification Consistency,

participants rated how consistently the main characters were identifiable and coherent throughout the story. Subjective Aesthetics Score assessed the overall artistic appeal, detail richness, and storytelling effectiveness of the visualizations.

- Table 3. Correlation Analysis. Kendall’s τ, Spearman’s ρ, and Pearson’s σ coefficients between human evaluations and automated metrics are reported.

Metrics Self CSD Self CIDS Aesthetics

τ 0.4165 0.4978 0.2565 ρ 0.5648 0.6759 0.3966 σ 0.6042 0.7956 0.5411

Detailed scoring criteria for each dimension are provided in Appendix G. The detailed scoring results will also be open-sourced. We list the top four methods in each category based on user ratings (0-4 scores, scale to 100%):

- ⃝1 Environment Consistency: UNO (3.28 82.0%), GPT4o (3.25 81.2%), Doubao (3.22 80.4%), StoryAdapter (3.18 79.6%)
- ⃝2 Character Identification Consistency: Doubao (3.70 92.6%), AIbrm (3.54 88.4%), UNO (3.36 84.0%), GPT4o (3.26 81.6%)
- ⃝3 Subjective Aesthetics Score: GPT-4o (3.42 85.6%), Doubao (3.4 85.0%), AIbrm (3.32 83.0%), UNO (3.21 80.4%)

We establish one-to-one correspondences between three human evaluation metrics and automated metrics, similar to Theatergen [6]: Character Identification Consistency corresponds to CIDS, Environment Consistency corresponds to Style Similarity, and Subjective Aesthetics Score to Aesthetic Quality. Subsequently, we evaluate the correlation coefficients between automated metrics and human evaluation results, including Kendall’s τ, Spearman’s ρ, and Pearson’s σ, as shown in Table 3. These demonstrate that automated metrics can effectively reflect human preferences.

Due to time constraints, the user study was limited to commercial method versions available until mid-May 2025, and we excluded obvious outliers like StoryGen [45] from correlation analysis to ensure valid metric validation. As detailed in Appendix G.

- 4.2. Discussion: Insights and Limitations

Insights. Based on the comprehensive evaluation results across ViStoryBench and ViStoryBench-Lite, we derive and briefly outline the following key findings:

- ⃝1 Multi-modal large models (e.g., GPT-4o [36]) excel in narrative alignment but lag in visual quality;
- ⃝2 Commercial tools exhibit strengths in aesthetics and style sim, yet often lack fine-grained narrative control;
- ⃝3 Story image methods achieve high character consistency but show limited generalization across scenarios;
- ⃝4 Story video methods face challenges in per-frame quality when modeling temporal dynamics;
- ⃝5 Multi-shot video methods exhibit high self-consistency due to training on movie data, while showing weaker customization capability;

- ⃝6 A clear trade-off exists between consistency and diversity, underscoring the need for balanced evaluation;
- ⃝7 This Real-world oriented evaluation reveals that current methods still struggle to jointly optimize semantic and visual qualities.

These insights emphasize the value of task-specific model selection and highlight future directions such as multi-shot data, unified multi-modal architectures, and more holistic evaluation metrics. Details are provided in Appendix B

Limitations. The design and scope of ViStoryBench are subject to several limitations:

- ⃝1 Focus on multi-image consistency: Although synchronized audio-visual storytelling is the long-term goal, current evaluation is limited to multi-image generation with emphasis on inter-frame consistency.
- ⃝2 Lack of background-aware evaluation: Due to the absence of background-reference support in existing open-source methods, the benchmark does not include background reference images or scene-level image-andimage similarity evaluation.
- ⃝3 Trade-offs in evaluation metrics: Our hybrid evaluation strategy employs expert models for stability and VLMbased scoring for semantic richness. However, expert models may underperform in complex contexts, while VLMs remain prone to hallucinations.

Despite our extensive efforts to select the most reliable evaluation method for each metric, these limitations remain.

##### 5. Conclusion

We present ViStoryBench, a high-fidelity benchmark for evaluating story visualization under diverse scenarios. It contains 80 multi-shot stories across 10 visual styles, each with character references and script annotations. The benchmark introduces 12 human-validated automated metrics assessing character similarity, style similarity, prompt alignment, aesthetics, artifacts, and copy-paste behavior. Extensive experiments on open-source and commercial systems offer the first large-scale, multi-dimensional comparison of SOTA methods, revealing key strengths, limitations, and future directions.

Due to space limitations, we have placed a large number of technical details in the appendix and supplementary materials, such as the video display on our homepage, the evaluation of multi-shot video generation models, and the prompt alignment evaluation based on Qwen.

##### Acknowledgement

This work was supported by the National Natural Science Foundation of China (No. 6250070674) and the Zhejiang Leading Innovative and Entrepreneur Team Introduction Program (2024R01007).

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

###### Character Plot Scene Shot Action

- Figure S1. Words Cloud. Visualization of narrative elements from the stories in our ViStoryBench dataset: spanning character traits (e.g., black hair, middle-aged, uniform), plot points (e.g., says, finally, love), scene settings (e.g., daytime, atmosphere, outside, sunlight, living room), shot types (e.g., eye-level, close-up, high-angle), and character actions (e.g., expression, stands, gaze).

### Appendix

##### A. Broader Limitations and Societal Impact

The design and scope of ViStoryBench are subject to several limitations, which reflect current technological constraints and evaluation challenges in the field of storyoriented generation:

Focus on multi-image consistency: While the long-term objective of this research community is synchronized audiovisual storytelling with full scene dynamics, the current benchmark is deliberately scoped to multi-image generation with an emphasis on inter-shot consistency. This focus allows us to address the most immediate challenges in visual narrative coherence without introducing additional complexity from temporal modeling or audio alignment, which are active areas of research in their own right. Since established benchmarks like VBench [110] already provide comprehensive metrics for single-shot video temporal modeling, we avoid redundant efforts in this area.

Lack of background-reference evaluation: Due to limited support for background-conditioned generation in current open-source story visualization models, ViStoryBench does not incorporate background reference images or include scene-level image-to-image similarity evaluation, with only a minimal number of single-scene stories included. Instead, scene descriptions are provided via text prompts, and scene consistency is assessed through prompt alignment. Future work will expand the dataset to include more single-scene multi-shot stories along with corresponding scene reference images to enhance the scope and functionality of the benchmark.

Inherent trade-offs in evaluation metrics: The benchmark employs a hybrid evaluation strategy that combines

expert models (for stability and continuity) and VLMbased scorers (for semantic richness and narrative alignment). However, this approach involves fundamental compromises: expert models may lack adaptability in visually or narratively complex scenarios, while VLMs remain susceptible to hallucination and may not always align with human perceptual judgments. These trade-offs underline the lack of a universally optimal automated metric for all aspects of story visualization.

Dataset Limitations and Copyright Concerns. Some images in our dataset are derived from well-known movies, TV shows, and animations. While these samples are used solely for academic research and benchmarking purposes, they may raise copyright concerns. We do not claim ownership of any copyrighted material, and all third-party content is included under fair use principles for non-commercial research and analysis. Nevertheless, users of our dataset should be aware of potential legal constraints when repurposing or redistributing the data. Additionally, the inclusion of well-known visual content may cause certain metrics to become overfitted to these familiar styles or characters, potentially leading to metric manipulation or overoptimization.

Language Sensitivity. Our benchmark supports both Chinese and English story prompts. While we select the appropriate language for each model based on its design and documentation, we do not control for potential discrepancies in generation quality caused by language differences. This may introduce variation that is not attributable to model capability alone.

Scope of Evaluation. Our benchmark currently does not support accurate evaluation of comic-style or manga generation tasks that involve multi-panel layouts within a single image, due to the lack of a robust panel segmentation

method. Similarly, we do not assess inference efficiency or runtime performance across models. For video-based story generation methods, our benchmark does not yet include a comprehensive evaluation of temporal coherence, framelevel consistency, or other video-specific quality metrics. These remain important future directions.

Societal Impact. We envision story visualization models as promising tools for education, creativity, and cultural preservation. In curating the dataset, we made conscious efforts to include diverse narratives from multiple cultures and regions. However, generative models are still susceptible to reproducing stereotypes and amplifying data-driven biases. It is vital that these tools are developed and deployed responsibly.

Conclusion. Despite extensive efforts to align metric selection with the requirements of each evaluation dimension, these limitations reflect persistent challenges in the current evaluation paradigm. They also indicate meaningful directions for future work in developing more comprehensive and reliable story visualization benchmarks.

Finally, we emphasize that generative models should not be used to create or disseminate false or misleading content. Addressing such risks requires active collaboration between researchers, platform providers, and policymakers to ensure safe and ethical applications.

##### B. Overall Insights

Based on the table of automated test results on ViStoryBench and ViStoryBench-Lite, we have analyzed the performance of various story visualization methods across multiple metrics. Here are the key insights and patterns observed from a research perspective:

Performance of Multi-modal Large Models is Dominant, Especially GPT-4o.

- ⃝1 GPT-4o [36] achieves the best or second-best performance in critical metrics like Alignment Score (3.673), OCCM Score (93.5), CIDS (Cross: 0.571 and Self: 0.679), and CSD (Cross: 0.481 and Self: 0.680), indicating strong narrative understanding, character consistency and style similarity. This suggests that largescale multi-modal models excel at high-level semantic alignment and coherence, likely due to their extensive pre-training on diverse data.
- ⃝2 However, GPT-4o [36] lags in Inception Score (9.02) and Aesthetics Score (5.49), which measure image diversity and visual quality. This implies a trade-off: while LLMs handle narrative complexity well, they may struggle with low-level visual fidelity compared to specialized methods.

Commercial Software Shows Strengths in Visual Quality but Inconsistencies in Narrative Tasks.

- ⃝1 Commercial tools like MorphicStudio [58] excel in style consistency (CSD Cross: 0.653), while Doubao [2] perform well in alignment (3.494). This highlights their optimization for production-ready output.

- ⃝2 However, they exhibit variability: for example, MOKI [54] has high aesthetics (5.79) but poor character consistency (CIDS Cross: 0.214). This indicates that commercial tools may prioritize aesthetic appeal over fine-grained narrative controls, leading to imbalances in evaluation dimensions.

Story Image Methods Excel in Specific Niches, but Lack Uniformity.

- ⃝1 Methods like OmniGen2 [86] lead in character consistency (CIDS Self: 0.537) and OCCM (90.8), demonstrating strengths in maintaining identity across frames. Story-Adapter [52] variants achieve high style consistency (CSD Cross: 0.456), showing progress in specialized tasks.

- ⃝2 However, performance varies widely: SEED-Story [98] and TheaterGen [6] score low on multiple metrics (e.g., Alignment <2.00 and CIDS Cross <0.35), indicating that some methods overfit to specific scenarios or lack generalization. The reliance on reference images (e.g., Story-Adapter [52] with image-ref) often boosts consistency but may limit creativity.

Story Video Methods Underperform in Key Metrics, Revealing Challenges in Temporal Modeling.

- ⃝1 Video-based Methods like Vlogger [114] and MovieAgent [85] generally score lower in style and character consistency (e.g., CSD Cross <0.3) compared to image-based methods. This suggests that temporal modeling introduces additional complexity, hindering per-frame quality.

- ⃝2 An exception is MovieAgent [85] (SD3), which achieves strong alignment (3.16), implying that leveraging advanced image-based diffusion models (e.g., SD3 [15]) can mitigate some issues. Yet, overall, video methods lag in metrics like Inception Score, indicating limited diversity in generated sequences.

Excellent Performance in Multi-shot Video Models, Especially Sora2.

⃝1 Multi-shot video generation models excel in crossshot character and scene consistency, as evidenced by high self-similarity score (e.g., CIDS Self 0.813 and CSD Self 0.713 for Sora2 [60]), outperforming many image-based methos. This likely results from training on large-scale character-consistent multi-shot movie

data. In contrast, most image-based methods lack specialized training on multi-shot storyboard data.

⃝2 However, current multi-shot video models struggle with visual reference adherence, indicated by relatively lower cross-similarity scores (e.g., CIDS Cross 0.738 and CSD Cross 0.515 for Sora2 [60]), revealing a gap in style and scene constraint preservation compared to reference-based image methods.

Trade-offs Between Consistency and Creativity Are Evident. Methods with high character consistency and high copy-paste score (lower is better) often have lower diversity (e.g., GPT-4o). This underscores a fundamental tension in story visualization: optimizing for one dimension may compromise another.

In story visualization tasks, comprehensive evaluation metrics are extremely important. For instance, the simple Copy-Paste Baseline achieves optimal results across numerous metrics. However, its alignment score is notably low. Although IS can generally measure the quality and diversity of image generation, it is quite challenging to compare different models by examining the IS metric alone. When using only text as input, StoryDiffusion [111] and Story-Adapter [52] achieve excellent IS and aesthetic quality. However, relying solely on text input clearly cannot produce results that resemble the features and styles of the character reference images.

Our quantitative metrics demonstrate alignment with qualitative observations. For Story-Adapter [52], the scoring consistency between automated metrics and human evaluation is particularly evident: (1) In text-only mode (its native setting), the overall quality score (scale=5) systematically surpasses the baseline (scale=0), as theoretically expected; (2) When using image references, scale=0 achieves higher cross-similarity but lower self-similarity compared to scale=5 in both CIDS and CSD.

ViStoryBench-Lite Reveals Real-World Gaps Results on ViStoryBench-Lite (focused on practical scenarios) show that commercial and LLM methods perform well in alignment but struggle with low-level metrics (e.g., Gemini2.0 [11] has Align: 3.150 but CSD Cross: 0.361). This indicates that real-world applications require balancing semantic and visual qualities, and current methods may not fully address this.

###### Conclusion and Future Directions:

⃝1 No single method dominates all metrics, emphasizing the need for task-specific model selection. For narrativeheavy tasks, LLMs like GPT-4o [36] are preferable; for

visual quality, commercial tools or specialized image methods may suffice.

- ⃝2 Future work should focus on hybrid approaches: integrating LLMs for planning with diffusion models for visual quality, and improving evaluation metrics to better capture storytelling aspects like pacing and emotion.
- ⃝3 A promising direction for story visualization is to combine the temporal coherence of video models with the precise reference alignment of image-based generators, enabling customized multi-shot output with both consistency and control.
- ⃝4 The variability in results underscores the value of benchmarks like ViStoryBench for guiding progress. Researchers should prioritize methods that balance consistency, diversity, and alignment.

##### C. Details of Dataset Collection and Statistics

Story visualization datasets have shown notable growth and evolution in terms of scale, image resolution, automated generation pipelines, and stylistic diversity—reflecting both technological advancements and the expanding scope of research interests. We summarize story visualization datasets [24, 34, 42, 45, 98, 99] in Table S3. A key distinction of ViStoryBench compared to existing datasets lies in our construction methodology: rather than extracting captions from visual keyframes to construct a narrative, we take a top-down approach by generating structured shot descriptions directly from full textual stories. Additionally, when curating benchmarks, we place particular emphasis on ensuring a broad range of styles and thematic diversity. Table S1 presents a detailed breakdown of the 10 visual styles curated in our dataset, which are classified based on their character reference images.

Table S1. Distribution of the 10 Visual Styles in ViStoryBench. The classification is based on the visual style of the character reference images for each story.

###### Style Category Number of Stories

Photorealistic / Live-action photo 39 Anime / Cel-shading style 14 Children’s book / Cartoon 7 Classic fairy tale illustration / Vintage 5 Classical oil painting / Religious art 4 Flat vector illustration 4 3D / Voxel / Claymation style 3 Chinese ink painting / Line art / Silhouette 2 Action manga / American comics style 1 CG realistic fantasy illustration 1

Total 80

The collected dataset spans a wide array of genres, including 13 folktales, 10 romance stories, 4 suspense/crime

[Figure 250]

[Figure 251]

[Figure 252]

###### Dataset Construction Casts Dataset Statistics

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

<Raw Story> Little brown rabbit, who was going to bed, held on right to Big brown rabbit's very long ears. He wanted to be sure that Big brown rabbit was listening.…

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

Number of Characters

[Figure 263]

<Character Name> Little Brown Rabbit <Description> A little rabbit with chestnut fur, bright eyes, small and …

[Figure 264]

Story 03: Chinese Ancient Court

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

<Character Name> Big Brown Rabbit <Description> A warm and kind big rabbit with chestnut fur, tall and ...

[Figure 275]

[Figure 276]

Number of Shots

|[Figure 277]<br><br>[Figure 278]<br><br>[Figure 279]<br><br>Big Brow|[Figure 280]<br><br>n Rabbit|
|---|---|

|[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]<br><br>Little Brow|[Figure 284]<br><br>n Rabbit|
|---|---|

[Figure 285]

[Figure 286]

Manual Collection

[Figure 287]

Story 09: German Fairy Tales

[Figure 288]

AI Generation

[Figure 289]

[Figure 290]

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

···

|[Figure 301]<br><br>Shot 1|Shot 2 Shot 3 Shot N<br><br>|
|---|---|
|[Figure 302]<br><br>[Figure 303]<br><br><Plot Correspondence> It's time for the little rabbit to go to bed, but he tightly holds onto the big rabbit's ears and refuses to let go.<br><br>[Figure 304]<br><br><Setting Description> Nighttime, bedroom, cozy atmosphere, soft bedding, a small night light on the bedside table…<br><br>[Figure 305]<br><br><Shot Perspective Design> Medium shot, eye level shot. <Characters Appearing> Little Brown Rabbit, Big Brown Rabbit.<br><br>[Figure 306]<br><br>[Figure 307]<br><br><Static Shot Description> The little brown rabbit sits on the bed, tightly holding the big brown rabbit‘s ears, with a playful and cute …| |

[Figure 308]

Story 26: Horror

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

Story 29: Ancient Japanese Legends

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

Theme Distribution

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

Category Gender

[Figure 332]

Story 52: Fantasy

- Figure S2. Overview of ViStoryBench Dataset. Dataset Construction: We build a story generation pipeline powered by large language models (LLMs), followed by human verification to ensure quality and consistency. Casts: Reference images for characters are manually curated to maintain a consistent visual style. Dataset Statistics: ViStoryBench dataset exhibits a broad distribution across story categories, stylistic variations, and character diversity, enabling comprehensive evaluation of storytelling generation models.

|[Figure 333]| | | | |[Figure 334]| | | | | | |[Figure 335]<br><br>[Figure 336]| | | |[Figure 337]| | | |[Figure 338]| | | |[Figure 339]|[Figure 340]| | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 341]| |[Figure 342]|[Figure 343]| | |[Figure 344]| | | | | | | |[Figure 345]| | | |[Figure 346]| | |[Figure 347]| |[Figure 348]| | | |[Figure 349]|
|[Figure 350]| |[Figure 351]| |[Figure 352]| | |[Figure 353]| | | | |[Figure 354]| | | |[Figure 355]| | |[Figure 356]| | |[Figure 357]| | | |[Figure 358]| |
|[Figure 359]|[Figure 360]| | | | | | | | |[Figure 361]<br><br>[Figure 362]| | | | | |[Figure 363]|[Figure 364]| | |[Figure 365]| | | |[Figure 366]| | |[Figure 367]|

[Figure 368]

- Figure S3. Random Character Reference Samples from the Dataset. The reference images include full-body shots, half-body shots, or portraits, spanning diverse visual styles from photorealistic to a variety of animation-inspired designs.

stories, 3 horror narratives, 6 historical tales, 10 fantasy stories, 7 science fiction stories, 3 war stories, 10 stories about social life, 3 survival/adventure stories, and 11 fairy tales. These stories are segmented and adapted into detailed shot scripts using the in-house LLM model. The full dataset contains 1,317 shots in total, with each story comprising between 4 and 30 shots (averaging 16.5 shots per story). Basic statistics shown in Figure S2

To support a wide range of methods, all test-related textual prompts are provided in both English and Chinese. For methods that only support Chinese, or perform significantly better with Chinese input, we use the Chinese version; otherwise, English inputs are used. Each individual shot is annotated with five structured fields: Setting Description, Plot Correspondence, Onstage Characters, Static Shot Description, and Shot Perspective Design.

In our curation process, we made a conscious effort to incorporate narratives from diverse cultural backgrounds. The cultural origins of the 80 stories are distributed across several major regions, including Chinese (39 stories), EuroAmerican (27), Japanese (8), African (3), Islamic/Middle Eastern (2), and Indian (1). This diversity is complemented by a wide array of thematic genres: the dataset includes 13 folktales, 10 romance stories, 4 suspense/crime stories,

- 3 horror narratives, 6 historical tales, 10 fantasy stories, 7 science fiction stories, 3 war stories, 10 stories about social life, 3 survival/adventure stories, and 11 fairy tales.

###### C.1. Character Reference Image

For most well-known stories in our dataset, character reference images are directly sourced from existing visual works such as movies, animated films, or television series. For lesser-known or original stories, we adopt one of two strategies to obtain reference images for the main characters: (1) retrieving representative screenshots from films or TV shows with similar settings and styles (covering 16 stories), or (2) using the SDXL model [65] to generate high-quality stylized animation portraits (covering 7 stories).

In total, our dataset includes 344 unique characters, which can be categorized into 190 real humans, 135 virtual humans (e.g., animated or game characters), and 19 non-human entities (e.g., animals or creatures). Regarding gender annotation, there are 210 male, 108 female, and 26 characters who are either genderless or non-binary. Each character is associated with between 1 and 10 reference images, resulting in a total of 509 reference images, with 89 characters having more than one image. A selection of these reference images is visualized in Figure S3.

Furthermore, we categorize all 80 stories into two distinct types based on the visual style of the main character references: realistic stories and unrealistic stories. The realistic category includes 39 stories whose characters are portrayed using photographic or cinematic images. These characters are additionally labeled with ethnicity information following prior works [7, 8, 62]. The remaining 41 stories are labeled as unrealistic, typically involving animation, stylized art, or fantasy characters.

This classification allows us to conduct stratified evaluations and analyze how different generation methods perform across story types with distinct visual and semantic characteristics.

###### C.2. Prompts for Dataset Construction

To automate the transformation of story narratives into detailed multi-shot descriptions, we leverage a large language model (LLM) as a shot planner. This LLM is tasked with segmenting each narrative into a coherent sequence of visual shots, ensuring consistency in character presence, camera composition, environment description, and story pro-

gression.

We propose a structured prompt engineering approach for data generation, which systematically decomposes the complex story visualization task into well-defined, verifiable subtasks to achieve precise control over LLM outputs. This methodology aligns with rigorous benchmark construction standards in academic research and offers a valuable technical framework for building complex, structured multimodal datasets. Our approach converts an MLLM into a controllable visual narrative script generator and employs five core strategies:

- ⃝1 Multi-grained task decomposition, breaking the task into five structured modules—Plot Correspondence, Setting Description, Shot Perspective Design, Onstage Characters, and Static Shot Description—enabling the LLM to focus on simpler subproblems for improved accuracy and stability.
- ⃝2 Professional knowledge infusion, incorporating cinematic expertise such as standardized shot terminology (e.g., Wide Shot, Low-angle) and narrative principles (e.g., Smooth Narrative Transitions, Emotional Resonance).
- ⃝3 Multi-dimensional information isolation, enforcing modality separation (e.g., excluding characters from setting descriptions) to prevent spurious correlations and support combinatory generalization.
- ⃝4 Visual-friendly description, ensuring all textual content is concrete and directly depictable (e.g., avoiding abstract expressions in favor of visually grounded descriptions).
- ⃝5 Contextual narrative modeling, maintaining coherence across shot sequences by considering preceding and subsequent shots. Thus, our prompt set defines an LLM-powered automated data generation framework, enabling efficient construction of a high-quality, consistent, and domain-informed benchmark for visual storytelling.

Below, we present the specific system prompt utilized for this purpose.

##### D. Qualitative Results

We provide the visualization generation results of the methods tested on Story 09 and Story 01, as shown in Figure S4 and Figure S5. We sample the first five shots of the story for display to offer a concise yet representative comparison. At the top of the figure, we include a “Copy-Paste Baseline” that visually presents the ground-truth character presence in each frame using manually cropped reference images. This serves as a reference for evaluating the accuracy of character depiction across methods. Below the baseline, we showcase the generation results from all 18 evaluated open-source methods (including their key variants), as well as several leading commercial tools. These compar-

- Table S2. Comparison between Lite and Full Settings across Multiple Evaluation Dimensions. Quantitative comparison between the full dataset and the lite subset under a single method. The close alignment across all metrics, including style and character consistency, generative quality, diversity, and prompt alignment, which demonstrates that the lite subset serves as a representative proxy for the full dataset, enabling efficient yet reliable evaluation.

Style Consistency Character Consistency Character Matching Generative Quality Diversity Prompt Alignment

Cross Self Cross Self OCCM Aesthetic Inception Scene Shot CI IA

Full 0.325 0.569 0.427 0.600 62.487 4.894 11.510 2.331 2.891 2.176 2.101 Lite 0.373 0.626 0.469 0.620 65.351 5.027 9.787 2.779 2.965 2.564 2.290

Diff 13.78% 9.55% 9.34% 3.21% 4.48% 2.68% 16.17% 17.57% 2.52% 16.34% 8.62%

- Table S3. Comparison of Multi-modal Storytelling Datasets. Representative story datasets are summarized in terms of annotation method, image scale, resolution, number of shots per story, and visual style.

###### Datasets Caption # Images Resolution # Shots Style

VIST [34] Manual 146k - 5 Realistic Flintstones [24] Manual 123k 128 × 128 5 Anime Pororo [42] Manual 74k 128 × 128 5 Anime StorySalon [45] ASR 160k 432 × 803 14 Anime StoryStream [98] Generated 258k 480 × 854 30 Anime OpenStory [99] Generated 107M 720p & 1080p 28 Realistic

isons highlight differences in prompt alignment, character consistency, and visual quality across models.

To facilitate better examination of the visual outputs, we also provide a full frame-by-frame visualization of the entire story. (Due to policy restrictions, the demo video was recorded anonymously and submitted as supplementary material.)

This detailed visualization enables a more comprehensive evaluation of each method’s performance over the complete narrative sequence, especially in terms of temporal coherence, character persistence, and scene transitions. We encourage readers to explore this link for in-depth visual analysis beyond the summarized frames shown in the main figure.

More results can be found on the website: https: //huggingface.co/datasets/ViStoryBench/ ViStoryBenchResult

##### E. ViStoryBench-Lite Benchmark

###### E.1. Effectiveness of ViStoryBench-Lite

To assess the representativeness and reliability of ViStoryBench-Lite, we conduct a comprehensive comparison between the full dataset and the Lite subset. We first perform a distributional analysis over story categories to examine the content diversity across both subsets. As shown in Figure 2, the Lite subset exhibits a highly similar category distribution to the full dataset, suggesting a well-preserved narrative and visual diversity.

Further, we apply a unified generation method across both subsets and report quantitative results over all ma-

jor evaluation dimensions. The detailed comparison is presented in Table S2, which includes metrics on style consistency, character consistency, generative quality, diversity, and prompt alignment. The results demonstrate that the performance difference between the Lite and full datasets is minimal across most dimensions. Notably, only marginal discrepancies are observed in certain VLM-based prompt alignment metrics, such as scene-level consistency and Character Interaction, while the overall alignment score remains within a narrow error margin.

These findings validate the effectiveness of ViStoryBench-Lite as a representative subset of the full benchmark. This is particularly important for settings where large-scale human involvement (e.g. user study) or commercial API evaluation is required, such as user studies or commercial platform assessments. By using ViStoryBench-Lite, researchers and practitioners can achieve efficient and cost-effective evaluation without compromising result reliability.

###### E.2. Full Evaluation on ViStoryBench-Lite

We present the results of several selected methods on the ViStoryBench-Lite benchmark in the main paper. To facilitate a more comprehensive comparison, we provide the complete evaluation results for all the methods on the lite set, as shown in Table S4. These results serve as a thorough reference for assessing the performance of different approaches from multiple perspectives. Additionally, to ensure reproducibility, we provide the complete prompt alignment evaluation results using Qwen3-VL-8B-Instruct, deployed via vLLM offline with a fixed seed of 42, as shown

You are a seasoned film script artist skilled at transforming descriptive text from novel scripts into visual content descriptions. You also adapt scripts into static shot scripts. Your designs must incorporate a wide variety of compositions to perfectly capture the script’s content through imagery, ensuring the storyline is effectively conveyed in the visual descriptions. In addition, you also need to provide a comprehensive introduction to the characters that appear in the shot scripts, mainly describing their appearance and clothing.

Task Description You are required to write shot scripts based on the user’s input story, totaling <num of shots> shots. Each shot can feature 0 to 3 characters, meaning it can be a scene shot, single-character shot, two-character shot, or three-character shot. For each shot in the shot script, you need to output <Plot Correspondence>, <Setting Description>, <Shot Perspective Design>, <Onstage characters>, <Static Shot Description>. The <Plot Correspondence>section requires dividing the original plot into <num of shots> scenes, presenting the plot of each scene in the form of narration and dialogue. Note that when dividing the input plot into different scenes, the rationality of the plot needs to be considered.

###### Concept Explanation of Various Fields in Shot Scripts

- • Setting Description: The story will be divided into <num of shots> scenes. The setting refers to the environmental setup of each scene. It should not include any characters. You need to describe all elements in the environment in detail in this field, so that the scene where the story takes place can be vividly recreated. Standard writing format: time, location, atmosphere description, other elements in the environment, lighting effects.

- • Shot Perspective Design: Shot Perspective Design refers to information from several dimensions: shot distance, camera angle, and camera type.
- • Onstage characters: Please select the characters appearing in this scene from the character list. The number of characters should be controlled between 0-3. If no characters appear, leave it blank.
- • Static Shot Description: This part describes the static actions or positions of characters and items in the scene, ensuring that it describes a fixed state. Writing format: <character position>, <character expression>, <character action>, <position of elements in the scene>.

###### Requirements for Creating Setting Description in Shot Scripts

- • You need to design a shot script for the user’s input story. Break the story into <num of shots> main scenes and write scene descriptions for each of these <num of shots> scenes.

- • Note that character descriptions should not appear in Setting Descriptions. Only describe the scene itself, ensuring that the scene is consistent with the original story. Do not include backgrounds, items, or other elements that are not present in the story.
- • Think from the perspective of the visuals, using the visuals to drive the content of the shots. Ensure that all plot elements can be directly depicted through the visuals. Avoid thinking from the perspective of a screenwriter’s script and refrain from using abstract or metaphorical expressions.
- • Pay attention to the consistency between the characters’ locations and the Setting Descriptions.
- • When writing the background content, do not directly use the expressions from the origin story. Use clear and concise sentences to describe in detail all the elements included in the background visuals and their relationships.
- • If there are characters in the visuals, clearly express their facial expressions, demeanor, and actions.
- • Pay attention to the visual narrative continuity between adjacent shot panels.

###### Requirements for Creating Shot Perspective Design in Shot Scripts

- • Continuity in Shot Composition: Adjacent shots should maintain coherence in shot composition and camera angles. By employing a diverse yet consistent combination of shot compositions and camera angles, create a viewing experience that is both spatially immersive and visually engaging.
- • Shot Distance Selection: Choose the most appropriate shot distance from ”wide shot, full shot, medium long shot, medium shot, medium close up, close up” based on the emotional atmosphere of the current scene. A smaller subject size results in a more relaxed emotional tone, while a larger subject size creates a tenser atmosphere. Consider the shot distance design of preceding and following shots as well.

- • Guidelines for Shot Composition Combinations:

- – Smooth Narrative Transitions: The way shot compositions are connected impacts narrative fluidity. Effective transitions involve gradual tightening or loosening. Moving from wider to tighter shots is called ”tightening,” while moving from tighter to wider shots is called ”loosening.” Avoid abrupt shifts from Wide Shot to extreme close-ups or vice versa.
- – Avoid Repetitive Compositions: Ensure that adjacent shots have different compositions to prevent visual monotony.
- – Emotional Resonance: Since shot composition affects emotional tone, match shot compositions with the emotional intensity of the plot. For instance, intense emotional scenes are better suited for tighter shots.
- – Rhythmic Pacing: The combination of shot compositions influences the visual rhythm of the scene. Use an appropriate mix of shot compositions within each episode to convey the narrative’s pace effectively.

- • Camera Angle Selection: Opt for the camera angle—”front view, side view, back view”—that best suits the current scene’s visual requirements.
- • Camera Type Selection: Choose the camera type—”Eye level shot, low-angle shot, high-angle shot, bird’s eye view shot, dutch angle shot, foreshortening, inverted shot”—that aligns with the scene’s content and emotional tone. Consider the camera types used in preceding and following shots for continuity.
- • Camera Type Combinations: Select camera types that complement the scene’s content and emotional context. Pay attention to how different camera types interact to enhance the visual storytelling.
- • Please refer to the provided materials to choose the most fitting camera type for the current scene’s content and emotional tone, ensuring that the combination of camera types supports the overall narrative effectively.

Reference Materials for Camera Design For Shot Distance selection:

- • Wide Shot: Displays the relationship between characters and their environment, commonly used to showcase scenes and background settings.
- • Full shot: Shows the entire body of a character, often used to present full actions or the overall view of a scene.
- • Medium long shot: Captures from above the character’s knees.
- • Medium shot: Captures from above the character’s waist.
- • Medium close up: Captures from above the character’s chest.
- • Close up: focus on a close-up of the character’s head or face, with the background and environment typically blurred or entirely out of view.

For determining the relationship between Camera Type and content:

- • Eye level shot: The camera is positioned at the same level as the eyeline.
- • Low-angle shot: The camera shoots upward from below the eyeline, enhancing the subject’s authority or size, often conveying power or intimidation. Suitable for emphasizing an individual’s dominance or creating visual pressure, such as highlighting a hero or villain.
- • High-angle shot: The camera looks down from above the eyeline, showing the breadth of a scene or diminishing the visual importance of the subject. Effectively reduces the visual scale of characters or objects, used to depict a sense of isolation or helplessness in characters, or to present vast landscapes.
- • Bird’s eye view shot: The camera shoots downward from a high altitude, providing a top-down perspective, usually covering extensive geographical areas. Highly effective when a global view of events or environments is needed.
- • Dutch angle shot: The camera is deliberately tilted during filming, often used to create a sense of imbalance or tension. Particularly effective in portraying scenes of chaos, tension, or psychological instability.
- • Foreshortening: Emphasizes depth of field through perspective techniques, making the relationship between foreground and background more prominent. Suitable for highlighting spatial relationships and depth, commonly used to enhance visual guidance and a sense of depth.
- • Inverted shot: The image is filmed upside down, challenging the audience’s visual habits, often used to represent confusion or an unstable mental state.

###### Notes

• Each Static Shot Description in the shot scripts should correspond sequentially to each segment of the story, providing detailed descriptions of characters’ expressions, actions, and states.

###### Notes

- • Do not include any characters in the Setting Description. Ignore the characters in the story and only describe the environmental setting.
- • Do not introduce items or backgrounds in the Static Shot Description that are not mentioned in the story. Ensure that the location of actions aligns with the story.
- • Each sentence in the plot has context. Use this context to determine the best shot design.
- • Ensure that the transitions between different shots in the shot scripts follow creative requirements.
- • Approach the creation of shot content from a visual perspective. Consider the best way to present the script visually.
- • Only provide content that meets the output format requirements; do not include any explanations.
- • When dividing the original plot content, rewrite it into a format suitable for presentation.
- • Strictly follow the format and requirements of the output example when writing.

###### A Sample output

{

"Shot 1": {

"Plot Correspondence": "Fern is performing a magical ritual at the bell tower to awaken an ancient power, which is crucial for her team’s quest to protect the town from impending danger.",

"Setting Description": "Twilight, stone street leading to the bell tower, bell tower with arches, town streets, distant bell tower with arches, romantic atmosphere, mysterious atmosphere, glowing arrow on the ground, fallen leaves,

pigeons perched on rooftops, soft golden light", "Shot Perspective Design": "Medium shot, eye level shot", "Onstage characters": ["Fern"], "Static Shot Description": "Fern is pressed against the bell tower pillar, with an expectant and serious expression. Her right index and middle fingers are

together, suspended in mid-air, and fine golden particles are pouring out from her fingertips."

}, "Characters": {

"Fern": "A beautiful girl with long purple hair and purple eyes, wearing a silver butterfly hair accessory, a black coat and black boots, and a white dress under the coat. She holds a wooden staff in her hand.",

"Frieren": "A young white-haired female elf with twin ponytails, blue-green eyes, pointed elf ears, a pair of red earrings, a white wizard robe with gold trim and brown boots, holding a staff with a round ruby on the top.",

"Himmel": "A young boy with short blue hair and blue eyes, wearing a blue knight uniform and a white cape, holding a long sword in his hand, with a gentle yet firm expression."

} }

in Table S5.

To further interpret the results, we visualize the normalized performance of each evaluated SOTA method across all twelve dimensions using polar coordinates in Table S6. Each radius is normalized relative to the best score achieved among all models for the corresponding dimension.

Performance Analysis of Various Methods. On ViStoryBench-Lite, we observe a wide spectrum of

performance among the evaluated methods.

As evidenced by the quantitative results in the tables, GPT-4o [36] and Gemini-2.5 [9] demonstrate superior prompt alignment capabilities, which can be attributed to their fine-grained comprehension abilities rooted in their LLM foundations. Meanwhile, Sora2 [60] achieves the best balance across the four key metrics of character consistency (cross and self) and style consistency (cross and self), likely benefiting from its extensive training on visually co-

[Figure 369]

|[Figure 370]|[Figure 371]|[Figure 372]|[Figure 373]|[Figure 374]|
|---|---|---|---|---|

[Figure 375]

Copy-Paste

Baseline

[Figure 376]

[Figure 377]

|[Figure 378]<br><br>[Figure 379]<br><br>[Figure 380]|[Figure 381]| |[Figure 382]<br><br>[Figure 383]<br><br>[Figure 384]<br><br>[Figure 385]<br><br>[Figure 386]|[Figure 387]|
|---|---|---|---|---|

[Figure 388]

Gemini

|[Figure 389]<br><br>[Figure 390]<br><br>[Figure 391]|[Figure 392]|[Figure 393]<br><br>[Figure 394]|[Figure 395]|[Figure 396]<br><br>[Figure 397]<br><br>[Figure 398]|
|---|---|---|---|---|

[Figure 399]

GPT4o StoryDiffusion

[Figure 400]

[Figure 401]

|[Figure 402]|[Figure 403]<br><br>[Figure 404]<br><br>[Figure 405]<br><br>[Figure 406]|[Figure 407]|[Figure 408]|[Figure 409]<br><br>[Figure 410]<br><br>[Figure 411]|
|---|---|---|---|---|

(Img-refver.)

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

###### StoryDiffusion

|[Figure 416]|[Figure 417]<br><br>[Figure 418]|[Figure 419]<br><br>[Figure 420]<br><br>[Figure 421]<br><br>[Figure 422]|[Figure 423]<br><br>[Figure 424]|[Figure 425]|
|---|---|---|---|---|

(Text-onlyver.)

[Figure 426]

[Figure 427]

|[Figure 428]|[Figure 429]<br><br>[Figure 430]|[Figure 431]<br><br>[Figure 432]<br><br>[Figure 433]<br><br>[Figure 434]|[Figure 435]|[Figure 436]<br><br>[Figure 437]|
|---|---|---|---|---|

[Figure 438]

###### UNO StoryAdapter

[Figure 439]

[Figure 440]

(Img-refscale0)

|[Figure 441]|[Figure 442]<br><br>[Figure 443]<br><br>[Figure 444]<br><br>[Figure 445]|[Figure 446]<br><br>[Figure 447]|[Figure 448]<br><br>[Figure 449]|[Figure 450]|
|---|---|---|---|---|

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

(Img-refscale5)

StoryAdapter

|[Figure 455]<br><br>[Figure 456]<br><br>[Figure 457]|[Figure 458]|[Figure 459]<br><br>[Figure 460]|[Figure 461]|[Figure 462]<br><br>[Figure 463]<br><br>[Figure 464]|
|---|---|---|---|---|

[Figure 465]

[Figure 466]

[Figure 467]

(Auto-Regressive)

|[Figure 468]|[Figure 469]<br><br>[Figure 470]<br><br>[Figure 471]<br><br>[Figure 472]|[Figure 473]|[Figure 474]<br><br>[Figure 475]<br><br>[Figure 476]|[Figure 477]|
|---|---|---|---|---|

[Figure 478]

StoryGen

[Figure 479]

[Figure 480]

|[Figure 481]|[Figure 482]<br><br>[Figure 483]<br><br>[Figure 484]<br><br>[Figure 485]|[Figure 486]<br><br>[Figure 487]|[Figure 488]|[Figure 489]<br><br>[Figure 490]|
|---|---|---|---|---|

[Figure 491]

###### TheaterGen StoryGen

(Mixture)

[Figure 492]

[Figure 493]

|[Figure 494]|[Figure 495]<br><br>[Figure 496]|[Figure 497]<br><br>[Figure 498]|[Figure 499]<br><br>[Figure 500]|[Figure 501]<br><br>[Figure 502]|
|---|---|---|---|---|

|[Figure 503]<br><br>[Figure 504]<br><br>[Figure 505]|[Figure 506]|[Figure 507]|[Figure 508]<br><br>[Figure 509]|[Figure 510]<br><br>[Figure 511]<br><br>[Figure 512]|
|---|---|---|---|---|

[Figure 513]

StudioAIbrm

|[Figure 514]|[Figure 515]<br><br>[Figure 516]|[Figure 517]<br><br>[Figure 518]<br><br>[Figure 519]<br><br>[Figure 520]|[Figure 521]<br><br>[Figure 522]|[Figure 523]|
|---|---|---|---|---|

[Figure 524]

Doubao Morphic

|[Figure 525]|[Figure 526]<br><br>[Figure 527]<br><br>[Figure 528]<br><br>[Figure 529]|[Figure 530]<br><br>[Figure 531]|[Figure 532]<br><br>[Figure 533]|[Figure 534]|
|---|---|---|---|---|

[Figure 535]

MOKI

|[Figure 536]<br><br>[Figure 537]|[Figure 538]|[Figure 539]<br><br>[Figure 540]|[Figure 541]<br><br>[Figure 542]<br><br>[Figure 543]<br><br>[Figure 544]|[Figure 545]|
|---|---|---|---|---|

[Figure 546]

[Figure 547]

|[Figure 548]<br><br>[Figure 549]<br><br>[Figure 550]|[Figure 551]|[Figure 552]|[Figure 553]<br><br>[Figure 554]|[Figure 555]<br><br>[Figure 556]<br><br>[Figure 557]|
|---|---|---|---|---|

Typemovie

[Figure 558]

Vlogger AnimDirector

|[Figure 559]<br><br>[Figure 560]|[Figure 561]|[Figure 562]<br><br>[Figure 563]<br><br>[Figure 564]|[Figure 565]|[Figure 566]<br><br>[Figure 567]<br><br>[Figure 568]|
|---|---|---|---|---|

(SD3ver.)

[Figure 569]

[Figure 570]

|[Figure 571]<br><br>[Figure 572]<br><br>[Figure 573]|[Figure 574]<br><br>[Figure 575]|[Figure 576]| |[Figure 577]<br><br>[Figure 578]<br><br>[Figure 579]<br><br>[Figure 580]|
|---|---|---|---|---|

MMStoryAgent MovieAgent

(ROICtrlver.)

[Figure 581]

|[Figure 582]<br><br>[Figure 583]|[Figure 584]|[Figure 585]<br><br>[Figure 586]<br><br>[Figure 587]<br><br>[Figure 588]|[Figure 589]<br><br>[Figure 590]|[Figure 591]|
|---|---|---|---|---|

###### MovieAgent

(SD3ver.)

|[Figure 592]|[Figure 593]<br><br>[Figure 594]<br><br>[Figure 595]<br><br>[Figure 596]|[Figure 597]|[Figure 598]|[Figure 599]<br><br>[Figure 600]<br><br>[Figure 601]|
|---|---|---|---|---|

[Figure 602]

[Figure 603]

|[Figure 604]<br><br>[Figure 605]|[Figure 606]<br><br>[Figure 607]|[Figure 608]|[Figure 609]<br><br>[Figure 610]<br><br>[Figure 611]|[Figure 612]|
|---|---|---|---|---|

|[Figure 613]|[Figure 614]<br><br>[Figure 615]|[Figure 616]<br><br>[Figure 617]|[Figure 618]<br><br>[Figure 619]<br><br>[Figure 620]<br><br>[Figure 621]|[Figure 622]|
|---|---|---|---|---|

[Figure 623]

CharaConsist

[Figure 624]

|[Figure 625]<br><br>[Figure 626]<br><br>[Figure 627]|[Figure 628]|[Figure 629]|[Figure 630]<br><br>[Figure 631]<br><br>[Figure 632]<br><br>[Figure 633]|[Figure 634]|
|---|---|---|---|---|

###### OmniGen2

[Figure 635]

|[Figure 636]<br><br>[Figure 637]<br><br>[Figure 638]|[Figure 639]<br><br>[Figure 640]|[Figure 641]<br><br>[Figure 642]|[Figure 643]<br><br>[Figure 644]|[Figure 645]|
|---|---|---|---|---|

Seedream4

[Figure 646]

|[Figure 647]<br><br>[Figure 648]<br><br>[Figure 649]|[Figure 650]<br><br>[Figure 651]|[Figure 652]|[Figure 653]|[Figure 654]<br><br>[Figure 655]<br><br>[Figure 656]|
|---|---|---|---|---|

[Figure 657]

Edit(2509)

QwenImg

[Figure 658]

|[Figure 659]|[Figure 660]<br><br>[Figure 661]<br><br>[Figure 662]<br><br>[Figure 663]|[Figure 664]<br><br>[Figure 665]|[Figure 666]<br><br>[Figure 667]|[Figure 668]|
|---|---|---|---|---|

[Figure 669]

###### Gemini-2.5

NanoBanana

[Figure 670]

[Figure 671]

[Figure 672]

- Figure S4. Qualitative Result on Story 09. From left to right are shot1 to shot5. Reference images of each shot’s onstage characters is shown in Copy-Paste baseline results.

herent multi-shot data. The performance gap between earlier methods (e.g., StoryGen [45], TheaterGen [6], Vlogger [114]) and more recent approaches (e.g., UNO [90], OmniGen2 [86]) reflects the natural progression and collective advancement within the research community.

Models such as Storydiffusion [111] and StoryAdapter [52] exhibit strong prompt alignment, while maintaining balanced performance in generation quality and character similarity. MovieAgent (SD-3) [85] and AnimDirector [43] achieve consistently high scores across most dimensions, with particularly notable strengths in aesthetics and diversity, indicating their advantages in generating visually appealing and varied outputs. Interestingly, SEED-Story [98], which focuses on story continuation tasks, exhibits excellent self-consistency in style but shows weaker performance in other metrics. This observation highlights a potential trade-off between maintaining visual fidelity and introducing meaningful variation in generated content.

Among the commercial and proprietary systems, we observe competitive and often leading performance. AIbrm [50] and Doubao [2], two commercial Chinese platforms, show highly stable performance across almost all

metrics. Doubao, in particular, demonstrates outstanding results in both prompt alignment and generation quality, with a strong balance between character rendering fidelity and stylistic coherence. These results suggest that closedsource commercial tools have made significant strides in multi-shot storytelling, although their internal pipelines remain opaque.

Overall, these results highlight the multidimensional nature of high-quality story visualization and suggest that no single model excels uniformly across all criteria. These quantitative findings align consistently with our qualitative visual observations and common sense, thereby validating the credibility and reliability of our proposed evaluation metrics for story visualization tasks.

##### F. Method Evaluation Detail on ViStoryBench

In this section, we report how we adapt each method to the ViStoryBench test. In general, we strive to implement reasonable inputs for reference character images and shot script prompts on each work as much as possible. We make efforts to standardize inputs, such as adjusting output resolution to a 16:9 aspect ratio whenever feasible. To guarantee reproducibility, we fix the random seed. Additionally,

[Figure 673]

|[Figure 674]|[Figure 675]<br><br>[Figure 676]<br><br>[Figure 677]<br><br>[Figure 678]|[Figure 679]|[Figure 680]|[Figure 681]<br><br>[Figure 682]<br><br>[Figure 683]|
|---|---|---|---|---|

[Figure 684]

Copy-Paste

Baseline Gemini-2.5

[Figure 685]

[Figure 686]

|[Figure 687]<br><br>[Figure 688]|[Figure 689]|[Figure 690]<br><br>[Figure 691]|[Figure 692]<br><br>[Figure 693]<br><br>[Figure 694]<br><br>[Figure 695]|[Figure 696]|
|---|---|---|---|---|

[Figure 697]

(Image-ref)

[Figure 698]

Sora2

[Figure 699]

[Figure 700]

|[Figure 701]|[Figure 702]<br><br>[Figure 703]<br><br>[Figure 704]<br><br>[Figure 705]|[Figure 706]|[Figure 707]<br><br>[Figure 708]<br><br>[Figure 709]|[Figure 710]|
|---|---|---|---|---|

[Figure 711]

[Figure 712]

(Text-only)

Sora2

[Figure 713]

[Figure 714]

|[Figure 715]|[Figure 716]<br><br>[Figure 717]<br><br>[Figure 718]<br><br>[Figure 719]|[Figure 720]|[Figure 721]<br><br>[Figure 722]<br><br>[Figure 723]|[Figure 724]|
|---|---|---|---|---|

[Figure 725]

CharaConsist

[Figure 726]

|[Figure 727]|[Figure 728]<br><br>[Figure 729]<br><br>[Figure 730]<br><br>[Figure 731]|[Figure 732]|[Figure 733]|[Figure 734]<br><br>[Figure 735]<br><br>[Figure 736]|
|---|---|---|---|---|

OmniGen2Edit(2509)

[Figure 737]

|[Figure 738]<br><br>[Figure 739]|[Figure 740]|[Figure 741]<br><br>[Figure 742]<br><br>[Figure 743]<br><br>[Figure 744]|[Figure 745]|[Figure 746]<br><br>[Figure 747]|
|---|---|---|---|---|

Seedream4

[Figure 748]

|[Figure 749]|[Figure 750]<br><br>[Figure 751]<br><br>[Figure 752]|[Figure 753]|[Figure 754]<br><br>[Figure 755]<br><br>[Figure 756]<br><br>[Figure 757]|[Figure 758]|
|---|---|---|---|---|

[Figure 759]

QwenImg

[Figure 760]

|[Figure 761]|[Figure 762]<br><br>[Figure 763]|[Figure 764]<br><br>[Figure 765]|[Figure 766]<br><br>[Figure 767]<br><br>[Figure 768]<br><br>[Figure 769]|[Figure 770]|
|---|---|---|---|---|

[Figure 771]

NanoBanana

[Figure 772]

[Figure 773]

[Figure 774]

- Figure S5. Qualitative Result on Story 01. From left to right are shot1 to shot5. Reference images of each shot’s onstage characters is shown in Copy-Paste baseline results.

we implement mechanisms for inputting reference images and adapting lengthy shot script prompts. These adaptations enable methods to generate continuous image results of stories.

###### F.1. Copy-Paste Baseline

The Copy-Paste Baseline aims to verify the correctness of certain metrics by constructing the simplest possible gener-

ation method, which involves directly copying and pasting characters into images. For example, this baseline demonstrates that the metrics in ViStoryBench can perform an accurate calculation in areas such as CIDS, CSD, Copy-paste rate, and Matched Character Count calculation.

We obtain the image for the current shot by simply stitching together the images of the onstage characters in each shot, resulting in the image outcome for the current shot. A

- Table S4. Quantitative Results of Various Story Visualization Methods on ViStoryBench-Lite. Results highlighted with a gray background are excluded from ranking, for example, SEED-Story is trained on only three animations and does not aim for generalization, while the Copy-Paste Baseline directly pastes the character reference image as the output. For certain methods, we evaluate multiple inference configurations and report all corresponding results. indicate the first, second, third, fourth, and fifth performance, respectively. CSD: Style Similarity; CIDS: Character Similarity; PA: Prompt Alignment Score (CI: Character Interaction, IA: Individual Action); CM: OCCM; Inc: Inception Score; Aes: Aesthetics Score; CP: Copy-Paste. : With image reference; : Only text input; : Auto-regressive mode; superscript k means scale=k. Note: PA scores are based on Gemini-3 Pro.

CSD↑ CIDS↑ PA↑ CM↑ Inc↑ Aes↑ CP Cross Self Cross Self Scene Shot CI IA Avg.

Method Model

Copy-Paste Baseline - 0.735 0.770 0.911 0.993 0.42 1.60 0.76 0.85 0.91 92.76 5.46 4.39 0.550

###### Story Image Method

StoryGen [45] SD1.5 0.405 0.562 0.405 0.591 0.59 1.78 0.58 0.55 0.88 52.98 7.15 4.09 0.277 StoryGen [45] SD1.5 0.396 0.551 0.396 0.602 0.59 1.94 0.55 0.25 0.83 52.53 7.67 4.09 0.224 StoryGen [45] SD1.5 0.316 0.617 0.316 0.610 0.46 2.02 0.52 0.30 0.82 40.13 6.25 3.86 0.240 TheaterGen [6] SD1.5 0.221 0.411 0.354 0.537 1.99 1.92 0.49 0.40 1.20 54.93 13.60 4.94 0.204 StoryDiffusion [111] SDXL 0.293 0.680 0.409 0.641 2.61 2.02 1.25 1.05 1.73 67.07 12.99 5.83 0.186 StoryDiffusion [111] SDXL 0.409 0.611 0.460 0.575 1.35 2.68 1.34 1.25 1.66 62.48 8.18 5.21 0.251 SEED-Story [98] SDXL 0.258 0.763 0.559 0.656 1.44 1.67 0.32 0.16 0.90 64.33 4.90 3.81 0.306 Story-Adapter [52] 0 SD1.5 0.518 0.609 0.490 0.605 1.42 2.53 1.75 1.45 1.79 70.34 11.49 4.89 0.250 Story-Adapter [52] 5 SD1.5 0.371 0.758 0.425 0.619 1.35 2.40 1.46 1.40 1.65 61.39 12.03 4.80 0.217 Story-Adapter [52] 0 SD1.5 0.343 0.515 0.430 0.547 1.48 2.67 1.76 1.60 1.88 65.32 12.72 5.12 0.203 Story-Adapter [52] 5 SD1.5 0.353 0.752 0.416 0.634 1.31 2.44 1.40 1.30 1.61 61.57 10.59 4.85 0.220 UNO [90] FLUX1 0.425 0.648 0.512 0.630 3.12 2.25 1.98 1.75 2.28 70.88 10.50 5.13 0.287 OmniGen2 [86] DiT 0.491 0.648 0.576 0.668 3.18 2.38 2.15 1.90 2.40 73.44 8.21 5.21 0.298 CharaConsist [82] FLUX1 0.333 0.646 0.347 0.539 3.24 2.37 1.82 1.63 2.27 62.10 10.84 5.78 0.216 QwenImageEdit-2509 [66] DiT 0.404 0.614 0.482 0.541 3.37 2.10 2.45 2.20 2.53 61.27 10.56 5.46 0.249

[Figure 775]

###### Story Video Method

Vlogger [114] SD1.4 0.240 0.462 0.369 0.524 1.12 2.25 1.60 1.35 1.58 77.13 8.41 4.24 0.209 Vlogger [114] SD1.4 0.299 0.497 0.373 0.519 1.14 2.24 1.53 1.35 1.57 79.14 8.83 4.24 0.211 AnimDirector [43] SD3 0.305 0.558 0.423 0.593 3.27 2.11 2.44 2.05 2.47 72.03 9.94 5.60 0.206 MMStoryAgent [96] SDXL 0.261 0.661 0.385 0.598 2.44 1.94 1.14 0.95 1.62 58.24 8.09 5.91 0.189 MovieAgent [85] SD1.5 0.236 0.564 0.372 0.568 0.93 1.81 0.88 0.85 1.12 64.41 10.06 4.69 0.261 MovieAgent [85] SD3 0.346 0.539 0.433 0.582 3.09 2.28 2.45 1.90 2.43 67.29 12.04 5.32 0.199

###### Commercial Platform

MOKI [54] - 0.214 0.694 0.372 0.621 2.29 1.56 0.58 0.45 1.22 45.96 10.36 5.79 0.211 MorphicStudio [58] - 0.577 0.628 0.603 0.677 3.01 2.31 1.77 1.32 2.10 60.79 9.00 4.96 0.234 AIbrm [50] - 0.412 0.730 0.557 0.740 3.06 2.18 1.67 1.55 2.12 75.53 9.53 5.72 0.223 ShenBi [53] - 0.275 0.575 0.418 0.585 3.49 2.35 2.65 2.11 2.65 61.33 11.60 5.07 0.197 Typemovie [78] - 0.325 0.646 0.464 0.621 2.34 2.17 1.62 1.40 1.88 74.14 11.15 5.32 0.168 Doubao [2] - 0.367 0.695 0.446 0.642 3.88 2.41 3.23 2.65 3.04 65.23 9.88 5.61 0.255

###### Multi-modal Large Model (Language, Image and Video)

GPT-4o∗ [36] - 0.481 0.680 0.420 0.522 3.82 2.82 3.58 3.12 3.34 69.33 9.02 5.49 0.209 Gemini-2.0∗ [11] - 0.361 0.573 0.573 0.677 3.26 2.46 2.43 2.00 2.54 74.82 10.12 4.91 0.266

- Gemini-2.5∗ [21] - 0.447 0.657 0.553 0.642 3.89 2.32 3.26 2.90 3.09 64.86 10.54 5.61 0.255

- Gemini-3.0 Pro∗ [22] - 0.385 0.622 0.581 0.653 3.94 2.58 3.71 3.25 3.37 59.97 12.50 5.54 0.244 Seedream-4.0 [20] - 0.369 0.585 0.280 0.539 3.82 2.59 3.20 2.68 3.07 49.45 12.12 5.21 0.201 Sora2∗ [60] - 0.515 0.713 0.766 0.839 3.08 2.76 2.91 2.50 2.81 81.42 6.53 4.72 0.158 Sora2∗ [60] - 0.365 0.685 0.364 0.561 3.01 2.83 2.96 2.33 2.78 72.67 9.68 4.52 0.158

sample is shown in Figure S6.

###### F.2. Story Image Methods

###### F.2.1. StoryDiffusion

The original StoryDiffusion [111] is primarily designed for image generation using multiple reference images of differ-

ent characters and multi-shot prompts, where each prompt includes a character name and description. We implement the following adaptations for ViStoryBench evaluation:

⃝1 Since the native implementation can only handle short texts that cannot exceed the model’s maximum sequence length limit (77 tokens), similarly to StoryGen adaption,

- Table S5. Qwen-Based Prompt Alignment Evaluation of Story Visualization Methods on ViStoryBench and ViStoryBench-Lite. While the Copy-Paste Baseline directly pastes the character reference image as the output. For certain methods, we evaluate multiple inference configurations and report all corresponding results. indicate the first, second, third, fourth, and fifth performance, respectively. PA: Prompt Alignment Score (CI: Character Interaction, IA: Individual Action); : With image reference; : Only text input; : Auto-regressive mode; superscript k means scale=k.

PA (lite)↑ PA (full)↑ Scene Shot CI IA Avg. Scene Shot CI IA Avg. Copy-Paste Baseline - 0.30 2.29 0.80 1.11 1.12 0.30 2.29 0.80 1.11 1.12

Method Model

###### Story Image Method

StoryGen [45] SD1.5 0.66 2.75 0.85 1.12 1.35 0.65 2.62 0.84 1.10 1.30 StoryGen [45] SD1.5 0.71 2.57 0.82 0.98 1.27 0.72 2.65 0.85 1.07 1.32 StoryGen [45] SD1.5 0.48 2.73 0.74 1.18 1.28 0.56 2.68 0.83 1.14 1.30 TheaterGen [6] SD1.5 2.28 2.32 0.55 0.73 1.47 2.16 2.26 0.49 0.67 1.39 StoryDiffusion [111] SDXL 2.73 3.52 1.23 1.38 2.21 2.77 3.48 1.40 1.45 2.27 StoryDiffusion [111] SDXL 1.40 3.43 1.62 1.82 2.07 1.38 3.35 1.73 1.72 2.04 SEED-Story [98] SDXL 1.62 2.79 0.32 0.44 1.29 1.64 2.57 0.54 0.51 1.32 Story-Adapter [52] 0 SD1.5 1.60 3.48 2.00 2.05 2.28 1.68 3.47 2.17 2.07 2.35 Story-Adapter [52] 5 SD1.5 1.65 3.48 1.83 1.74 2.17 1.67 3.50 1.99 1.86 2.26 Story-Adapter [52] 0 SD1.5 1.81 3.54 2.35 2.12 2.45 1.83 3.51 2.28 2.11 2.43 Story-Adapter [52] 5 SD1.5 1.57 3.46 1.84 1.77 2.16 1.64 3.49 1.99 1.87 2.25 UNO [90] FLUX1 3.11 3.46 2.38 1.94 2.72 3.03 3.53 2.22 1.88 2.67 OmniGen2 [86] DiT 3.16 3.56 2.43 2.03 2.80 3.09 3.59 2.48 2.07 2.81 CharaConsist [82] FLUX1 2.54 3.59 1.56 1.40 2.27 2.50 3.53 1.61 1.39 2.26 QwenImageEdit-2509 [66] DiT 3.24 3.55 2.64 2.22 2.91 2.98 3.53 2.46 2.07 2.76

[Figure 776]

###### Story Video Method

Vlogger [114] SD1.4 1.41 3.31 2.12 1.95 2.20 1.36 3.27 2.10 1.88 2.15 Vlogger [114] SD1.4 1.37 3.28 1.96 1.81 2.10 1.35 3.24 2.07 1.90 2.14 AnimDirector [43] SD3 3.37 3.59 2.98 2.17 3.03 3.32 3.61 2.97 2.28 3.04 MMStoryAgent [96] SDXL 2.64 3.19 1.10 1.25 2.04 2.55 3.30 1.34 1.24 2.11 MovieAgent [85] SD1.5 0.96 3.07 0.86 0.96 1.46 0.95 3.10 0.80 0.79 1.41 MovieAgent [85] SD3 3.09 3.59 2.90 2.31 2.97 3.14 3.58 2.96 2.32 3.00

###### Commercial Platform

MOKI [54] - 2.49 2.32 0.54 0.60 1.49 - - - - MorphicStudio [58] - 3.00 3.47 2.12 1.88 2.62 - - - - AIbrm [50] - 2.92 3.43 1.73 1.64 2.43 - - - - ShenBi [53] - 3.48 3.51 3.11 2.25 3.09 - - - - Typemovie [78] - 2.34 3.51 1.94 1.82 2.40 - - - - Doubao [2] - 3.80 3.60 3.51 2.73 3.41 - - - - -

###### Multi-modal Large Model (Language, Image and Video)

GPT-4o∗ [36] - 3.68 3.79 3.55 2.95 3.49 - - - - Gemini-2.0∗ [11] - 3.08 3.55 2.68 2.07 2.84 - - - - Gemini-2.5∗ [21] - 3.61 3.48 3.19 2.72 3.25 - - - - Seedream-4.0 [20] - 3.61 3.58 3.23 2.75 3.29 - - - - Sora2∗ [60] - 3.05 3.56 3.14 2.20 2.99 - - - - Sora2∗ [60] - 2.83 3.62 2.92 2.21 2.89 - - - - -

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

###### Figure S6. Sample Results of Copy-Paste Baseline.

###### Table S6. Best Normalized Polar Visualization of SOTA Methods on ViStoryBench-Lite. Anti-Clockwisely, Character Similarity:

Cross-similarity; Style Similarity:

###### Cross-similarity; Prompt Alignment:

Shot, Character Interaction (CI),

Self-similarity,

Self-similarity,

Scene,

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

###### Average Score; Generation Quality:

Aesthetics; Diversity:

Individual Action (IA),

Inception Score.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Story Image Method

StoryGen (Multi-image) TheaterGen StoryDiffusion (Img-ref) SeedStory StoryAdapter (Img-ref, scale=0)

| |0.2 0.8|
|---|---|
| | |

| |0.2 0.8|
|---|---|
| | |

| |0.2 0.8|
|---|---|
| | |

| |0.2 0.8|
|---|---|
| | |

| |0.2 0.8|
|---|---|
| | |

UNO OmniGen2 CharaConsist QwenImageEdit-2509

| |0.2 0.8|
|---|---|
| | |

| |0.2 0.8|
|---|---|
| | |

| |0.2 0.8|
|---|---|
| | |

| |0.2 0.8|
|---|---|
| | |

###### Story Video Method & Commercial Platform

Vlogger (Text-only) AnimDirector (SD3) MMStoryAgent MovieAgent (SD3) MOKI

| |0.2 0.8|
|---|---|
| | |

| |0.2 0.8<br><br>|
|---|---|
| | |

| |0.2 0.8<br><br>|
|---|---|
| | |

| |0.2 0.8|
|---|---|
| | |

| |0.2 0.8<br><br>|
|---|---|
| | |

MorphicStudio AIbrm ShenBi TypeMovie Doubao

| |0.2 0.8|
|---|---|
| | |

| |0.2 0.8|
|---|---|
| | |

| |0.2 0.8|
|---|---|
| | |

| |0.2 0.8<br><br>|
|---|---|
| | |

| |0.2 0.8|
|---|---|
| | |

###### Multi-modal Large Model

GPT-4o Gemini-2.0 Gemini-3 (NanoBananaPro) Seedream-4.0 Sora2 (Img-ref)

| |0.2 0.8<br><br>|
|---|---|
| | |

| |0.2 0.8|
|---|---|
| | |

| |0.2 0.8<br><br>|
|---|---|
| | |

| |0.2 0.8<br><br>|
|---|---|
| | |

| |0.2 0.8<br><br>|
|---|---|
| | |

we employ the grouped encoder sd embed [112] to address this issue.

- ⃝2 Given StoryDiffusion does not support inputting reference images of multiple characters while generating one

shot, we sort characters based on their appearance frequency in the full shot script, from highest to lowest, and only introduced the highest-priority characters in one shot.

- ⃝3 StoryDiffusion predefines various style templates, each containing detailed style descriptions, including quality words. We insert the character prompt into the specified position within the style description, i.e., between the style word and the quality word, to achieve richer style expression.
- ⃝4 During testing, we obtain two types of results: those based on image reference (img ref) and those based solely on text (text only). We report the metrics for both types of results separately.

###### F.2.2. Story-Adapter

The original Story-Adapter [52] generates multiple images through multiple prompts and does not inherently support input image references. We implement the following adaptations for ViStoryBench evaluation:

- ⃝1 Incorporating Reference Images: Story-Adapter performs multiple rounds of iterative inference. In the first round, it generates all the storyboard images, which are then used as image references for regeneration in subsequent rounds. This multi-round process enhances both prompt alignment and style consistency of the images. We leverage this inherent image-referencing capability of Story-Adapter by inputting the image of the current onstage character into the pipeline during the first round of generation, thereby achieving image generation with reference images.
- ⃝2 We test the results of different iteration rounds (scale 0 and scale 5) under two modes: text-only and image-Ref, and reported the results of all four methods.

###### F.2.3. UNO

UNO [90]’s original implementation supports various reasoning methods such as one2one, one2many, many2one, and many2many. Among these, the many2many approach involves inputting multiple images along with prompts that provide simple descriptions of the images. We exclusively evaluate the many2many method, which is capable of completing the ViStoryBench task. For each shot, we generate the corresponding image result by inputting the shot’s prompt and the images of the onstage characters.

###### F.2.4. OmniGen2

As a general-purpose image generation model, OmniGen2 is evaluated for its visual consistency capability. Scene, plot, action, and shot design descriptions are concatenated into a prompt. The first reference image of every character appearing in the shot is supplied as a visual reference. Each shot is generated individually. The previous shot’s image is excluded from the context, as empirical tests indicated performance degradation when including such references.

###### F.2.5. CharaConsist

CharacterConsist enables training-free, cross-shot character identity “locking”. We evaluate its performance on key

metrics such as CIDS Self.

- ⃝1 “Setting Description” serves as the background prompt, “Character Description” as the foreground prompt, and “Static Shot Description” + “Shot Perspective Design” as the action prompt. Since CharacterConsist does not support explicit character references, no reference images are injected.
- ⃝2 The most frequently appearing character in each story is selected as the protagonist, with “prompt en” used as the unified foreground character descriptor across all shots.

- ⃝3 The first shot is generated as the identity reference. Subsequent frames reuse the “id fg mask” and “id bg mask” from the first frame. Each generation first performs a pre-run (“is pre run”=True) before formal generation. Spatial parameters (“spatial kwargs”) are propagated throughout to ensure character and background consistency across shots.

###### F.2.6. QwenImageEdit-2509

This model is specifically designed for multi-image consistency and is evaluated for its visual coherence capability. Scene, plot, action, and shot design descriptions are concatenated as the prompt. The first reference image of each character in the shot is provided as a visual reference. Each shot is generated sequentially. The image from the previous shot is omitted from the input, as tests demonstrated that including it led to output degradation.

###### F.2.7. StoryGen

The original StoryGen [45] supports the integration of previously generated image-text pairs as a context to construct image sequences that align with the input shot scripts. We implement the following adaptations for ViStoryBench evaluation:

- ⃝1 Mix Inference Method: The mix method behaves similarly to the multi-image-condition method for the first frame, while subsequent frame generation follows the auto-regressive method.
- ⃝2 When image references are absent (e.g., no onstage characters), the first shot uses aggregated story-wide character descriptions and images for consistent initial representation. For subsequent shots, a dynamic sliding window blends historical generation results as input, maintaining temporal coherence and mitigating quality degradation from long-range dependencies.
- ⃝3 Grouped Encoder for Ultra Long Prompts: To handle ultra-long prompts and prevent information loss due to truncation in CLIP models, we employ a grouped encoder sd embed [112] to address this issue.

- ⃝4 Resolution Adjustment: We modify the resolution settings to approximate a 9:16 aspect ratio, specifically 512 × 912, to align with the generation resolutions of other methods.
- ⃝5 Deterministic Random Seed Strategy: To enhance test

controllability, we adopt a deterministic random seed strategy.

###### F.2.8. TheaterGen

The original TheaterGen [6] only supports text input, where an LLM is used to parse the description of each character and the overall image of the text. Subsequently, each character’s independent image is generated through the IPAdapter, and then these images are placed in their corresponding positions using detection and segmentation models. Finally, the final image is generated under the guidance of the character images.

Since the method does not open-source the code for invoking the LLM part, we supplement the corresponding code to primarily obtain the bounding box coordinates of each IP image required for model input. Afterward, following TheaterGen’s setup, we place the character reference images in our dataset into the corresponding positions to generate final results.

###### F.2.9. SEED-Story

The original SEED-Story [98] focus on generating long multi-modal stories and their visualization through an autoregressive approach. SEED-Story requires the user to provide an initial image and text description, then proceeds with the story generation, using each output as the next input. Due to the story-continuation nature, it is different from other approaches (SEED-Story was only trained on the StoryStream dataset [98], which contains data from only three cartoon series, the model does not have generalization).

Input Handling: We only adapt our data in the visualization stage, and there is no need for the previous story generation. For the first input image (000start), select the first reference image of the first character. In the pre-visualization input data, add the prompt words of this role to the beginning of the prompt list to ensure the first shot’s prompts are not compromised.

###### F.3. Story Video Method

###### F.3.1. MovieAgent

Similar to Vlogger, MovieAgent [85] also receives a story and utilizes an LLM to generate a series of fine-grained descriptions. Additionally, ROIctrl includes the bounding boxes of characters. We directly map shot prompts onto Vlogger’s fine-grained descriptions for generation. We only perform the step from prompt to image, without further generating a video, and use this image as the final result.

###### F.3.2. AnimDirector

AnimDirector [43] does not support input reference images. It utilizes LLM to supplement character and scene descriptions based on simple prompts provided by users, refining them into complete story sequences by scenes. Visual images are generated through the Stable Diffusion 3 [14]

model, filtered by VLM, and subsequently used to generate videos. We adopt the following strategies to complete the test on ViStoryBench:

- ⃝1 We directly input the prompt of each shot from ViStoryBench as the story sequence prompt to generate images. After filtering by VLM, we obtain the results without proceeding with the subsequent video generation steps.
- ⃝2 To accommodate ultra-long prompts, we employ grouped encoder sd embed [112] to address this issue.

- ⃝3 We modify the original resolution of 1024 × 1024 to a resolution close to 16:9, specifically 768 × 1344.
- ⃝4 We fix the random seed to ensure reproducibility.

###### F.3.3. MMStoryAgent

MMStoryAgent [96] does not support the input of reference images. We only utilize the Image Agent mentioned in the code to generate images from prompts, without further generating a video. As MMStoryAgent is based on StoryDiffusion, we also employ grouped encoder sd embed [112] to adapt to long prompts.

###### F.3.4. Vlogger

The original Vlogger [114] takes a short story and utilizes an LLM to generate a series of fine-grained descriptions. These descriptions encompass character and object descriptions, video script descriptions (in both Chinese and English), characters and objects present, duration settings, etc. We directly map shot prompts onto Vlogger’s fine-grained descriptions, selecting the first frame of each shot from the generated video as the result. Similar to StoryDiffusion, vlogger does not support inputting reference images of multiple characters while generating one shot. We adopt a similar solution to that of StoryDiffusion.

###### F.4. Commercial Software

Given the absence of API or similar call methods for the following commercial softwares and their intricate interaction processes, we employ external annotators to generate image results for all the commercial software mentioned below. The annotators adhered to a predefined collection instruction and protocol, and all generated results subsequently underwent a rigorous quality check by the authors. All the following methods were tested between May 1 and May 7, 2025. F.4.1. MOKI

When generating shots on MOKI [54], it is necessary to select one of the provided painting style options. To best replicate the effects of real human usage, we instruct the annotators to choose the option that most closely aligns with the style of the character reference images for each script.

Given that MOKI restricts the maximum number of reference characters in a story to three, we sort the characters based on their appearance frequency among all shots. The

three characters with the highest frequency were then added as reference characters to maximize the performance of the model.

We generate images of MOKI using Chinese version dataset. MOKI has a limit on the length of each shot’s prompt, capped at 60 Chinese characters. Consequently, we make certain omissions in the prompt for each shot. For instance, we only used the Static Shot Description and Plot Correspondence sections as input. If the character count still exceeded the limit, we employ an LLM to abbreviate the text while preserving key information.

For each shot, the platform generated four images at a time. We select the first image as the final result.

###### F.4.2. MorphicStudio

MorphicStudio [58] restricts each shot to include only one reference character. Consequently, we rank the frequency of character appearances in the script and use this order as a priority to select the current onstage reference character for each shot.

MorphicStudio allows uploading 2 to 10 images for the same reference character. For characters with only one reference image, we upload an additional identical image to meet the minimum requirement of two images. For characters with multiple reference images, we upload all available reference images. Therefore, we do not provide the CopyPaste Rate for MorphicStudio because all reference images were uploaded and used for generation, making it incalculable.

For each shot, the platform generates four images at a time. We select the first image as the final result.

###### F.4.3. AIbrm

AIbrm [50] restricts each shot to a maximum of two reference characters. We employ the same sorting methodology as utilized in MorphicStudio.

The character creation process in AIbrm involves uploading a real human image, selecting a style from the provided options, and inputting a character prompt to generate the character. For the real human category in our dataset, we upload images and chose the ”realistic” style. For virtual human and non-human categories, since image uploads were not feasible, we select the closest available style. Subsequently, we input the character prompts from our dataset across all styles to complete the character creation.

###### F.4.4. ShenBi

Firstly, when creating a project, ShenBi [53] requires selecting a generation style from a list. We choose the style that is closest to the reference images in the dataset.

When creating characters, we upload the character images along with their corresponding prompts to generate new character images, which are subsequently utilized within the method to produce shot images.

ShenBi restricts each shot to a maximum of three reference characters. We employ the same sorting methodology as utilized in MorphicStudio.

The output of ShenBi is in the form of videos. We extract the first frame of each scene video as the final result.

###### F.4.5. Typemovie

Firstly, when creating a project, Typemovie [78] requires selecting a generation style from a list. We choose the style that is closest to the reference images in the dataset.

The character creation process in Typemovie involves uploading a real human image, selecting a style from the provided options, and inputting a character prompt to generate the character. For the real human category in our dataset, we simply upload reference images. For virtual human and non-human categories, since image uploads were not feasible, we select the closest available style. Subsequently, we input the character prompts from our dataset across all styles to complete the character creation.

Typemovie restricts each shot to include only one reference character. We employ the same sorting methodology as utilized in MorphicStudio.

###### F.4.6. Doubao

We conduct our tests using the grayscale test version of the ”Image Generation” model on the Doubao homepage [2], dated April 27, 2025.

Since this image generation model only supports uploading a single image, we employ a sorting method similar to that used in MorphicStudio to prioritize characters and selected the highest-priority character to upload as a single reference image.

The prompt used during generation was (translated to English): ”This is an image of the protagonist <character name>. Next, please generate storyboard scenes based on the protagonist’s image and the script I provide. The script for the first scene is as follows: <shot prompt>.” We perform multiple rounds of generation in one session to obtain the desired consistent image results for one story.

###### F.5. Multi-modal Large Models

To integrate Multi-modal Large Models into the ViStoryBench evaluation framework, we adopt the following key adaptation strategies:

- ⃝1 Atomic Shot Processing: Each “shot” defined within ViStoryBench was treated as an independent generation request to the model, ensuring focused processing for individual narrative segments.
- ⃝2 Comprehensive Prompt Engineering: For every shot, a structured textual prompt was meticulously crafted. This prompt amalgamated all critical textual information provided by ViStoryBench, including plot details, scene descriptions, character portrayals, camera per-

spective guidelines, and the desired aspect ratio for the output image.

- ⃝3 Direct Visual Referencing: Character reference images, after undergoing a standardization pre-processing pipeline (e.g., resizing, color space conversion), were directly incorporated as visual inputs for each shot’s generation request. This aimed to guide the model in rendering characters consistent with their specified appearances.
- ⃝4 Conversational Context Continuation: To foster narrative coherence across sequential shots, the model’s inherent capability to process conversational history was leveraged. The most recent interaction cycles, encompassing the prompt for the preceding shot and the model’s response (which includes the generated image), served as contextual information for the subsequent shot’s generation task.

###### F.5.1. GPT-4o

OpenAI’s GPT-4o [36] represents a general-purpose multimodal understanding and generation model. We evaluate whether its prompt alignment capability directly translates to high-quality visual storytelling.

Scene, plot, action, and shot design descriptions are concatenated into a unified prompt. The image from the previous shot and the first reference image of all characters appearing in the current shot are used as visual references. Each shot is generated sequentially.

###### F.5.2. Gemini-2.0

Google’s Gemini-2.0 [11] represents a general-purpose multimodal understanding and generation model. We evaluate whether its prompt alignment capability directly translates to high-quality visual storytelling.

Scene, plot, action, and shot design descriptions are concatenated into a unified prompt. The image from the previous shot and the first reference image of all characters appearing in the current shot are used as visual references. Each shot is generated sequentially.

###### F.5.3. Gemini-2.5 (NanoBanana)

Gemini-2.5 [9] represents a general-purpose multimodal understanding and generation model. We evaluate whether its prompt alignment capability directly translates to highquality visual storytelling.

Scene, plot, action, and shot design descriptions are concatenated into a unified prompt. The image from the previous shot and the first reference image of all characters appearing in the current shot are used as visual references. Each shot is generated sequentially.

###### F.5.4. Seedream-4.0

Seedream-4.0 [20] represents a multimodal Image understanding and generation model. Scene, plot, action, and shot design descriptions are concatenated into a unified prompt.

The image from the previous shot and the first reference image of all characters appearing in the current shot are used as visual references. Each shot is generated sequentially.

###### F.5.5. Sora2

Sora2 [60] represents a long-video generation model with native multi-shot capability. We aim to evaluate its longrange narrative comprehension and multi-shot visual consistency.

- ⃝1 Scene, plot, action, and shot design descriptions for each shot are concatenated to form the input prompt.
- ⃝2 In “img ref” mode, the first reference image of every character in the story is provided to define overall visual style and character baselines. In text only mode, image references are disabled.

- ⃝3 TransNetV2 [73] is used for shot boundary detection and keyframe extraction, followed by manual verification to match keyframes to their corresponding shots.
- ⃝4 Stories that fail to generate due to copyright restrictions or realistic portrait style limitations are excluded; only successfully generated stories are included in average metric calculations.

##### G. Details of User Study

The user study component of our research was time-limited, concluding in mid-May 2025 due to project constraints. Its main purpose was to validate our automated evaluation metrics. As the user feedback showed a strong correlation with the automated scores, we confirmed the reliability of the automated method. Therefore, subsequent model evaluations were predominantly carried out using this approach.

In correlation analysis, we exclude results from StoryGen [45], as its limited generation quality caused human evaluators to penalize character consistency scores, This is due to the generated character deviating significantly from the distribution of typical characters, leading to mismatched human expectations.

To comprehensively assess the visual quality and consistency of story visualization results, we conducted a structured human evaluation on the ViStoryBench-Lite benchmark. This benchmark contains a diverse subset of stories across multiple genres and character settings, making it suitable for evaluating both identity consistency and visual storytelling fidelity.

Evaluation Dimensions. Each story visualization result was evaluated on three key dimensions:

- • Character Identification Consistency: Measures whether the main characters remain visually consistent and recognizable across different shots in the story.
- • Environment Consistency: Assesses the consistency of environmental elements—such as furniture, architecture,

Scoring Criterial and Interface of User Study Character Identification Consistency: Based on the provided story visualization results, please assess the character id consistency of characters throughout the story and provide a score. Scoring Criteria:

- • 0: There is a lack of fundamental ID consistency, with nearly every image featuring different characters, indicating an almost complete absence of images with matching characters.
- • 1: In a smaller subset of images (about 10-30%), the main characters demonstrate mutual consistency.
- • 2: In a moderate number of images (around 30-60%), the main characters can be treated as having mutual consistency.
- • 3: In a substantial subset of images (approximately 60-80%), the main characters exhibit mutual consistency. However, a minor portion of images still shows inconsistencies in character representation.
- • 4: The main characters are consistently identifiable across the vast majority of images.

Environment Consistency: Based on the provided story visualization results, please assess the environment consistency throughout the story and provide a score. Scoring Criteria:

- • 0: There is a lack of fundamental environmental consistency; under the same environmental description, the generated scenes exhibit neither consistent style nor content.
- • 1: At a glance, the scenes appear to have some level of consistency, such as similar styles. However, upon closer inspection, the content is entirely different and lacks any coherence.
- • 2: There is a certain level of consistency in the style and semantic information of the image scenes, such as the presence of similarly styled beds and windows. However, inconsistencies exist in either the style or specific content, for instance, while tables and desk lamps are present in both, the desk lamps themselves are not similar.
- • 3: The majority of image have consistent semantic information, with style and specific content being largely uniform.
- • 4: Nearly all image scenes exhibit strong consistency in both style and specific content, akin to the effect of video recording within the same scene over a continuous timeframe.

Subjective Aesthetics Score: Based on the provided results, please assess the aesthetics of the story and provide a score. Scoring Criteria:

- • 0: Most characters have very obvious generation problems, such as distorted faces, extra/missing limbs, or the painting style is very uncomfortable for humans to watch. Or the image quality is extremely poor.
- • 1: Characters have obvious generation problems, such as extra/missing limbs, distortion, etc., but there is no discomforting content. Or the image quality is poor.
- • 2: Over 80% of the characters have no obvious physical problems, and there is no obvious content that causes physical discomfort, but the visual experience is poor. Almost all the content of the images, such as character poses, is completely the same, lacking variation.
- • 3: Over 80% of the characters have no obvious physical problems. Mediocre picture books with ordinary visual experience, lacking variation and storytelling in images.
- • 4: Over 80% of the characters have no obvious physical problems. Excellent and beautiful picture books that can be commercialized, with rich content, beautiful details, diversity, and interest, and obvious storytelling.

[Figure 785]

- Table S7. Results of User Study. For certain methods, we evaluate multiple inference configurations and report all corresponding results.

indicate the first, second, third, fourth, and fifth performance, respectively. : With image reference; : Only text input; : Auto-regressive mode; superscript k means scale=k.

[Figure 786]

Character Identification Environment Subjective

Method Model

Consistency ↑ Consistency ↑ Aesthetics ↑

###### Story Image Method

StoryGen [45] SD1.5 0.10 0.12 0.05 StoryGen [45] SD1.5 0.18 0.27 0.13 StoryGen [45] SD1.5 0.37 0.22 0.10 TheaterGen [6] SD1.5 0.35 0.55 0.30 StoryDiffusion [111] SDXL 2.72 2.73 2.45 StoryDiffusion [111] SDXL 2.62 2.33 2.30 SEED-Story [98] SDXL 2.05 2.11 1.05 Story-Adapter [52] 0 SD1.5 2.55 2.62 2.80

[Figure 787]

- Story-Adapter [52] 5 SD1.5 2.90 2.98 2.68 Story-Adapter [52] 0 SD1.5 2.33 2.23 2.50

- Story-Adapter [52] 5 SD1.5 3.10 2.67 2.68 UNO [90] FLUX1 3.20 3.10 3.02

###### Story Video Method

- Vlogger [114] SD1.4 0.87 1.07 0.67

- Vlogger [114] SD1.4 1.30 1.33 1.08 AnimDirector [43] SD3 2.52 2.28 1.77 MMStoryAgent [96] SDXL 2.27 2.78 2.55 MovieAgent [85] SD1.5 1.90 1.95 1.55 MovieAgent [85] SD3 2.45 2.47 1.83

###### Commercial Platform

MOKI [54] - 1.73 2.20 2.55 MorphicStudio [58] - 2.60 2.53 2.39 AIbrm [50] - 3.42 2.97 3.15 ShenBi [53] - 2.74 2.89 2.48 Typemovie [78] - 2.25 2.35 2.00 Doubao [2] - 3.63 3.02 3.25

###### Multi-modal Large Model (Language, Image and Video)

GPT-4o∗ [36] - 3.08 3.06 3.28 Gemini-2.0∗ [11] - 2.84 2.84 2.26

or background settings—across the sequence of generated images.

• Subjective Aesthetics Score: Evaluates the overall visual appeal of the story, including character quality, composition, artistic style, and storytelling clarity.

Each dimension was scored on a 5-point Likert scale (from 0 to 4), based on clear qualitative criteria provided to the annotators as shown in the scoring interface below. For example, a score of 0 in Character Consistency indicates a complete lack of identifiable characters across shots, whereas a score of 4 indicates nearly perfect character continuity.

Annotation Interface and Process. We developed a web-based annotation interface below that displays:

- • The full set of generated images for a story.
- • Relevant textual annotations, such as the story prompt and

shot descriptions.

• A structured table for scoring each of the three criteria.

The interface was designed to facilitate efficient and focused annotation, allowing annotators to toggle between image sequences and text prompts while assigning scores.

Annotator Pool and Assignment Strategy. We recruited 20 human annotators with prior experience in visual content evaluation, including graduate students and crowd workers trained on our scoring rubric. Given the large number of models, stories, and dimensions to evaluate, we adopted a balanced partial assignment strategy: each annotator was assigned only a subset of the full evaluation set, but we ensured that:

- • Every model-story pair received at least 10 independent ratings per dimension.
- • Annotators were randomly assigned different stories and

methods to avoid bias.

- • Each task contained only a manageable number of stories (typically 6–8), to reduce fatigue.

This design helped scale the annotation process while maintaining evaluation reliability.

Aggregation and Analysis. For each model and story, we aggregated the scores by taking the mean across annotators. We also report standard deviation across annotators to reflect inter-rater variability. The collected annotations form the human evaluation benchmark for comparing model performance on ViStoryBench-Lite in terms of character coherence, environmental stability, and visual storytelling quality.

##### H. Details of Prompt Alignment Evaluation

To evaluate how well the generated images align with the input shot prompts, we employ GPT-4.1 [61] as an automatic evaluator. The LLM is prompted to assign a Likertscale rating (0–4) for each image-prompt pair across several semantic dimensions, including scene correctness, camera composition, and character actions. The average of these subtask scores yields the final Alignment Score used in Table S4.

To better analyze the capabilities and limitations of each method, we further break down the Alignment Score into four interpretable sub-scores: Scene Score, Shot Score, Character Interaction, and Individual Action, with the final score computed as an equally weighted average of these components. Each dimension focus on a specific aspect of visual-textual alignment. For example, the Scene Score evaluates the match between background or scene attributes and the prompt, while the Camera Score assesses adherence to cinematographic framing (e.g., close-up, long shot). The action scores reflect whether the actions of characters (either collectively or individually) are faithful to the described narrative.

From the results, we observe that recent video generation methods adapted for story visualization, such as AnimDirector and MovieAgent (SD3), achieve the highest alignment scores across most categories. AnimDirector leads in Scene Score (3.61) and Character Interaction (3.24), while MovieAgent (SD3) excels in Individual Action (2.50). In contrast, conventional image-generation methods like StoryGen or SEED-Story generally perform poorly, with significantly lower scores across all metrics.

This detailed analysis reveals that high-quality story visualization requires coherent handling of both low-level visual elements (like camera and scene) and high-level semantics (like character intent and interaction), underscoring the necessity of specialized multi-modal reasoning for prompt alignment.

###### H.1. Character Interaction

To assess fine-grained alignment between visual content and textual prompts—particularly focusing on interactions between characters—we introduce a semantic consistency evaluation protocol targeting Character Interaction. This task evaluates whether the generated image accurately captures the described relational dynamics between two or more characters, such as hugging, fighting, handing over an object, or sitting together. Such interactions are crucial for evaluating story-level coherence and the model’s ability to capture nuanced inter-character behavior.

###### H.2. Shooting Method (Shot)

To assess the framing and compositional accuracy of generated images from a cinematic perspective, we propose a shot-type alignment evaluation. This task examines how well the image conforms to the specified camera distance (e.g., close-up, medium shot, wide shot) and camera angle (e.g., eye-level, high-angle, low-angle) provided in the textual prompt. Accurate shot framing is essential for conveying narrative focus, emotional tone, and spatial arrangement—hallmarks of professional visual storytelling.

###### H.3. Static Shot Description (Scene)

To evaluate the fidelity of background and environment rendering, we introduce a static shot grounding task. Unlike the character-centric evaluations, this task focus on non-character elements—including environmental context, background objects, spatial layout, and overall ambient mood. It measures whether these visual elements align semantically with the scene descriptions in the prompt, such as ”a classroom with a blackboard and wooden desks” or ”a cozy bedroom with warm lighting and starry wallpaper.” This evaluation is critical for assessing the model’s holistic scene understanding.

###### H.4. Individual Action

To further probe character-level grounding and behavioral fidelity, we propose an individual action consistency evaluation. This task isolates the action of a specific named character described in the prompt—such as ”Tom is raising his right hand”—and checks whether the generated image faithfully represents this behavior. We crop the character from the image using a detected bounding box and perform feature-based comparisons for evaluation. This task offers a focused lens on the model’s ability to correctly associate and render discrete physical actions with the correct character identity.

###### H.5. Effectiveness of Prompt Alignment Metric

To ensure the evaluation protocol accurately captures the nuanced alignment between generated visualizations and

|Plot Correspondence|It's time for the little rabbit to go to bed, but he tightly holds onto the big rabbit's ears and refuses to let go.|Ennis knows he must move today because the farm has been sold.|Carrying his memories of Jack, Ennis embarks on a new journey.|Daedalus, commissioned by King Minos, built an incredibly complex labyrinth.|The old man became wealthy because of Princess Kaguya.|They boarded a small boat and left the island. Hagrid waved his wand, and the boat shot forward like an arrow, the splashing waves glittering like crystals in the sunlight.|
|---|---|---|---|---|---|---|
|Setting Description|Nighttime, the little rabbit's bedroom, cozy atmosphere, soft bedding, a small night light on the bedside table, a starry sky painting on the wall, the moon and stars visible outside the window, soft yellow lighting|Early morning, outside an old trailer, surrounded by a desolate farm, wind howling, Ennis stands by the trailer, preparing to pack his belongings.|Early morning, outside an old trailer, surrounded by a desolate farm, wind howling, Ennis sits in the truck’s driver seat, ready to depart.|Daytime, inside the labyrinth, gloomy atmosphere, stone walls covered with ivy, moss growing on the ground. Flickering light streams through narrow windows, faint echoes resonate from the depths of the labyrinth, dim lighting|Daytime, Sanuki no Takamuro’s new residence, a spacious courtyard, opulent house, bustling servants, bright lighting|Daytime, at sea, the small boat speeding forward, waves splashing, sunlight shining on the water, the outline of the island in the distance, bright lighting.|
|Shot Perspective Design|Medium shot, eye level shot|Full shot, low-angle shot|Medium shot, eye level shot|Medium shot, high-angle shot|Full shot, high-angle shot|Wide shot, rear view, bird's eye view shot|
|Characters Appearing|Little Brown Rabbit，Big Brown Rabbit|Ennis Del Mar|Ennis Del Mar|Daedalus|Sanuki no Takamuro|Harry Potter, Hagrid|
|Static Shot Description|The little brown rabbit sits on the bed, tightly holding the big brown rabbit's ears, with a playful and cute expression. The big brown rabbit sits by the bed, looking gentle, slightly lowering his head to look at the little rabbit|Ennis stands by the trailer, looking resolute yet helpless, loading his belongings onto the truck, the background showing the desolate farm and howling wind.|Ennis sits in the truck’s driver seat, looking resolute yet sorrowful, gripping the steering wheel, gazing into the distance through the windshield, the background showing the desolate farm and howling wind.|Daedalus stands at the center of the labyrinth, his expression focused, holding design blueprints in his hand as he surveys the surrounding walls|Sanuki no Takamuro stands in the center of the courtyard, looking around with a contented expression, his hands clasped behind his back.|Harry and Hagrid sit in the boat. Hagrid waves his wand, and the boat speeds forward. Harry grips the side tightly.|

|[Figure 788]<br><br>[Figure 789]<br><br>[Figure 790]|
|---|

|[Figure 791]<br><br>[Figure 792]<br><br>[Figure 793]|
|---|

|[Figure 794]<br><br>[Figure 795]<br><br>[Figure 796]|
|---|

|[Figure 797]<br><br>[Figure 798]<br><br>[Figure 799]|
|---|

|[Figure 800]<br><br>[Figure 801]<br><br>[Figure 802]|
|---|

|[Figure 803]<br><br>[Figure 804]<br><br>[Figure 805]|
|---|

High CLIP Score

|[Figure 806]<br><br>[Figure 807]<br><br>[Figure 808]|
|---|

|[Figure 809]<br><br>[Figure 810]<br><br>[Figure 811]|
|---|

|[Figure 812]<br><br>[Figure 813]<br><br>[Figure 814]|
|---|

|[Figure 815]<br><br>[Figure 816]<br><br>[Figure 817]|
|---|

|[Figure 818]<br><br>[Figure 819]<br><br>[Figure 820]|
|---|

|[Figure 821]<br><br>[Figure 822]<br><br>[Figure 823]|
|---|

High GPT Score (VinaBench)

|[Figure 824]<br><br>[Figure 825]<br><br>[Figure 826]|
|---|

|[Figure 827]<br><br>[Figure 828]<br><br>[Figure 829]|
|---|

|[Figure 830]<br><br>[Figure 831]<br><br>[Figure 832]|
|---|

|[Figure 833]<br><br>[Figure 834]<br><br>[Figure 835]|
|---|

|[Figure 836]<br><br>[Figure 837]<br><br>[Figure 838]|
|---|

|[Figure 839]<br><br>[Figure 840]<br><br>[Figure 841]|
|---|

High GPT Score (ViStoryBench)

[Figure 842]

⭐

- Figure S7. Case Study of Prompt Alignment. We compare the alignment between generated visualizations and the detailed prompt components—including plot correspondence, setting, shot perspective, characters, and static shot description—across six different shots. Each row provides expert annotations on prompt elements, while the bottom three rows indicate whether the generated image achieves a high CLIP score, a high GPT score according to VinaBench, or a high GPT score under the proposed ViStoryBench evaluation. ViStoryBench consistently shows better alignment with visual-semantic fidelity and shot design, as reflected by the green checks.

textual prompts, we conducted an in-depth case study analyzing prompt alignment across a diverse set of generated shots. As illustrated in Figure S7, we dissected the prompt into multiple semantically rich components—such as plot correspondence, setting, camera perspective, character presence, and static shot descriptions—and compared how different evaluation methods respond to these elements.

For each shot, we collected expert annotations to indicate the relevance and completeness of the generated content with respect to these prompt components. We then compared three types of automatic scoring signals: CLIPbased similarity, GPT-based scoring from VinaBench [17], and the proposed ViStoryBench evaluation protocol. The case study revealed that while CLIP and VinaBench scores often misalign with semantic and stylistic fidelity, our ViStoryBench protocol consistently correlates better with expert judgments, particularly in terms of narrative consistency and cinematographic correctness (as highlighted by the green checkmarks).

This iterative analysis informed the design of ViStoryBench’s evaluation prompts and scoring rubric, enabling a more reliable and interpretable assessment of visual story generation quality.

###### H.6. Stability of VLM-based Automatic Evaluation

A common concern with evaluations that leverage Large Language Models (LLMs), such as GPT-4.1 [61], is the potential for variability in their outputs, which could affect the reliability of the results. To rigorously validate the stability and robustness of our automated evaluation framework for prompt alignment, we conducted a detailed stability analysis.

Experimental Setup for Stability Test We performed 3 to 5 independent evaluation runs for each prompt-adherence metric (Scene Score, Character Interaction, Individual Action, and Camera Score). This analysis was conducted on a representative subset of our benchmark, comprising 5 diverse stories (01, 09, 27, 41, 53) and the results from three

key methods: UNO, StoryDiffusion, and Story-Adapter. By repeatedly evaluating the same set of generated images, we can precisely quantify the variance attributable to the LLM evaluator itself.

Results and Analysis The results demonstrate exceptionally low variance across all evaluation runs. The standard deviation for each metric was consistently an order of magnitude smaller than the performance gaps observed between different models in our main experiments. For instance, after aggregating the scores across the selected methods and stories, we observed the following mean scores and standard deviations:

- • Scene Score: 2.82 ± 0.03
- • Character Interaction: 2.57 ± 0.04
- • Individual Action: 2.40 ± 0.08
- • Camera Score: 3.23 ± 0.03 These minimal standard deviations confirm that the random fluctuations in the GPT-4.1 [61] evaluator are negligible. This high level of consistency ensures that the performance differences we report in our benchmark are meaningful reflections of model capabilities, rather than artifacts of evaluation noise. This finding strongly supports the reliability of using our VLM-based protocol for large-scale, automated story visualization assessment.

###### H.7. Correlation Analysis of Qwen-base and GPTbased Evaluation

Based on the Prompt Alignment scoring results from the ViStoryBench-Lite dataset, we conducted a correlation analysis between the scores provided by GPT-4.1 [61] and Qwen3-VL-8B-Instruct [97]. This analysis aimed to reveal the consistency between these two VLMs in evaluating story visualization methods. We focused on the Prompt Alignment Score (PA) average (Avg.) as it represents the comprehensive indicator of overall performance. The PA score includes sub-items: Scene (scene description), Shot (shot perspective), CI (character interaction), IA (individual action), and Avg. (average). Below is a detailed analysis report.

We extracted the PA Avg. scores (GPT-based and Qwenbased) for each method from Table S4 and Table S5, forming a paired dataset with scores ranging from 0 to 4 points. Statistical correlation analysis was performed, calculating the Pearson correlation coefficient (measuring linear correlation) and the Spearman rank correlation coefficient (measuring monotonic correlation based on rankings). The calculations and analyses are as follows:

Pearson Correlation Analysis for PA Average. The Pearson correlation coefficient is calculated using the for-

mula:

(Xi − X¯)(Yi − Y¯) (Xi − X¯)2 (Yi − Y¯)2

,

r =

where X represents GPT-based scores and Y represents Qwen-based scores.

Based on 34 data points, the Pearson correlation coefficient is r ≈ 0.93. This value close to 1 indicates a strong positive linear correlation between GPT-based and Qwenbased PA Avg. ratings. That is, when GPT scores are higher, Qwen scores tend to be higher as well, and vice versa.

Spearman Correlation Analysis for PA Average. The Spearman correlation coefficient is calculated using the formula:

6 d2i n(n2 − 1)

ρ = 1 −

,

where di is the difference in ranks, and n is the sample size.

After ranking the scores (higher scores receive higher ranks) and computing the rank differences, the Spearman correlation coefficient is ρ ≈ 0.89. This value also close to 1 indicates high consistency in the ranking order between the two ratings, demonstrating significant monotonic correlation. This implies that both models similarly evaluate the relative performance of different methods.

Correlation Analysis for PA Sub-items. For comprehensiveness, we also calculated the correlation coefficients for the PA sub-items (Scene, Shot, CI, IA). The average Pearson correlation coefficient ranges from approximately 0.85 to 0.90, and the average Spearman coefficient ranges from 0.82 to 0.88, both indicating moderate to strong correlations.

Conclusion of Correlation Analysis. Through correlation analysis of the Prompt Alignment scores on the ViStoryBench-Lite dataset, we found that the ratings from GPT-4.1 and Qwen3-VL are highly correlated (Pearson r ≈ 0.93, Spearman ρ ≈ 0.89). Both GPT-4.1 and Qwen3-VL are advanced VLMs with powerful multimodal understanding capabilities. They are trained on similar datasets, resulting in similar evaluation criteria for prompt alignment. The strong correlation indicates that in story visualization tasks, both models perform consistently when evaluating prompt alignment capability, demonstrating interchangeability between them.

##### I. Details of Character Identification Similarity

###### I.1. Calculation

Figure 4 illustrates the computation pipeline of our Character Identification Similarity (CIDS) metric. CIDS quantifies both self-similarity (within generated images) and

cross-similarity (between generated and reference images) through a four-stage computational pipeline:

- ⃝1 Character Detection. Grounding DINO localizes character regions using text prompts.

- ◦ Reference images: Crops character regions with edge trimming (minor boundary adjustments).
- ◦ Generated images: May fail to detect characters (returns empty result), indicating identity inconsistency.

- ⃝2 Feature Extraction. Extracts 512D embeddings from cropped regions:

- ◦ Realistic characters: ArcFace/AdaFace/FaceNet trimodel ensemble (robust facial features).
- ◦ Stylized characters: CLIP ViT-L/14 (semantic alignment).

- ⃝3 Bipartite Matching. Solves optimal character correspondence via Hungarian algorithm:

- ◦ Computes cosine similarity matrix between reference/generated features.
- ◦ Matches characters maximizing global similarity (excludes failed detections).

- ⃝4 Scoring. Final metric: CIDS = N1 Ni=1 cos(vref(i),vgen(i)) where N = number of matched pairs, v = feature vectors.

###### I.2. Impact of Reference Image Selection on CrossCIDS Metric

- Table S8. Cross-CIDS Metric with Different Reference Image. ”Dataset Reference” refers to results calculated with reference images in ViStoryBench dataset, ”Generated Reference” refers to results calculated with generated reference images of methods. All results below are obtained on ViStoryBench-Lite.

###### Method Dataset Reference Generated Reference

MOKI [54] 0.292 0.338 AIbrm [50] 0.559 0.683 ShenBi [53] 0.347 0.389

In certain methods, the character reference images or features used for generation are not directly sourced from our dataset but are instead synthesized through an additional generation stage. A common scenario involves converting real-person reference images from our dataset into stylized versions—such as anime-style characters—resulting in visual appearances that may differ substantially from the original subjects. These discrepancies in reference image selection can significantly impact the Cross-CIDS metric. In the main results tables, we report scores based on the original reference images provided in the dataset. For a more comprehensive comparison, we additionally report Cross-CIDS scores computed using the synthesized reference characters, as shown in Table S8.

##### J. Details of Style Similarity Calculation

Figure S8 illustrates the computation pipeline of our Style Similarity metric. Adapted from CSD [71, 113], this metric captures both self-similarity (within generated images) and cross-similarity (between generated and reference images) by analyzing style-specific features extracted via CSDCLIP [71].

The computation consists of three key steps:

- ⃝1 Each image is encoded into visual embeddings using a CLIP [67] vision encoder pre-trained on large-scale style datasets;
- ⃝2 The extracted features are passed through CSD layers to disentangle content and style representations, retaining only the style components;
- ⃝3 Pairwise cosine similarity is computed between the resulting style embeddings to measure stylistic alignment.

This design enables fine-grained comparison of artistic and stylistic consistency, independent of content semantics.

##### K. Details of OCCM

Onstage Character Count Matching (OCCM) metric relies on an upstream character detector to obtain the detected character count (D). Consequently, the accuracy of the OCCM score is inherently bounded by the detector’s performance, which may be affected by factors like heavy occlusion or extreme artistic styles. We opted for expert models as the upstream detector. While expert models can misjudge, VLMs perform worse in hallucination counting. The design of OCCM formula is based on two core principles:

- 1. Scale Normalization: By dividing the absolute error |D − E| by the expected count E, we convert the error into a relative percentage. This allows the metric to fairly evaluate scenes of varying scales, ensuring that the same relative error receives a similar penalty, whether in a single-character scene (E = 1) or a multi-character scene (E = 10).
- 2. Non-linear Penalty: We employ an exponential function to penalize the relative error. Unlike a linear penalty, the exponential function leads to a gentle score decay for small errors but a sharp drop for larger ones. This characteristic better aligns with human perception: missing one out of ten characters is a minor flaw, whereas missing five constitutes a critical failure.

##### L. Details of Copy-Paste Detection

To rigorously evaluate whether the generated image is merely a replication of a specific reference image (denoted as the anchor or target reference r0) rather than a generalized synthesis from the provided character concept, we employ a Softmax-based Copy-Paste Score.

Let g be the unit-normalized feature vector of the generated image, and R = {r0,r1,...,rN} be the set of unit-

[Figure 843]

[Figure 844]

[Figure 845]

###### Reference Images

[Figure 846]

###### CSD-CLIP

CSD-CLIP

|[Figure 847]<br><br>[Figure 848]|
|---|

|[Figure 849]<br><br>[Figure 850]|
|---|

|[Figure 851]<br><br>[Figure 852]|
|---|

|[Figure 853]<br><br>[Figure 854]<br><br>B|
|---|

Image

[Figure 855]

[Figure 856]

Character A

Character

Ref embedding CLIP Vision

CrossSimilarity

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

Self Similarity

Generated Shots

###### Style Decoupling

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

Shot 1

Calculation

Shot 1 embedding Shot 2 embedding

[Figure 865]

[Figure 866]

[Figure 867]

Shot 2

[Figure 868]

Style embedding

[Figure 869]

[Figure 870]

###### CSD Score

Cross-Sim Score: 0.88 Self-Sim Score: 0.90

CSD-CLIP

- Figure S8. Style Similarity Calculation Pipeline. Evaluating both cross-similarity and self-consistency by computing cosine similarity between the style features of two images.

Table S9. Computational Efficiency of Evaluation Metrics.

normalized feature vectors for the input reference images, where r0 represents the primary reference subject to copypaste detection, and {r1,...,rN} serves as the set of auxiliary references for the same character.

Metric Scope Time Notes

Aesthetics Score Single image 0.026s per generated image Style Similarity Pair images 0.046s cross or self Character Similarity Pair images 0.450s cross or self Inception Score Total data 8.057s full dataset Prompt Alignment* Single image 25.173s per generated image

We first calculate the cosine similarity between the generated image and each reference image in the set R. To quantify the exclusivity of the match between g and the target r0 relative to other references, we formulate the score as a probability distribution using a temperature-scaled Softmax function:

* Longest computation due to LLM inference constraints

we ensure consistent measurement conditions across metrics.

exp(g⊤r0/τ)

CopyRate(g|R) =

(S1)

N k=0 exp(g⊤rk/τ)

Table S9 reports the average computational cost associated with each evaluation metric. These metrics differ significantly in scope and computational complexity. For example, aesthetic scoring is performed on individual images and is highly efficient, averaging only 0.026 seconds per image. Style and character similarity metrics, which operate on image pairs, are slightly more demanding, especially character similarity (0.450 seconds per pair), likely due to the use of deep feature extractors.

where τ is a temperature hyperparameter set to 0.01. This low temperature value sharpens the distribution, making the metric highly sensitive to the nearest neighbor in the feature space.

The resulting Copy-Paste Rate ranges from 0 to 1. A score approaching 1 indicates that the generated image g is significantly more similar to the specific reference r0 than to any other provided references, suggesting a ”copy-paste” overfitting behavior. Conversely, a lower score implies that the generated features are either distributed among multiple references or have successfully generalized beyond the specific appearance of r0. The final metric is averaged across all generated samples for each character.

In contrast, metrics such as the Inception Score are computed over the entire dataset and thus have a higher runtime (8.057 seconds), though only once per dataset. The most computationally intensive metric is the Prompt Alignment score, requiring over 25 seconds per image. This is primarily due to the involvement of large language model (LLM) inference, which introduces latency constraints.

##### M. Benchmark Evaluation Efficiency

To ensure the reproducibility and transparency of our computational experiments, we provide detailed information regarding the hardware setup and evaluation runtime. All experiments are conducted using high-performance GPU accelerators (e.g., NVIDIA H800 with 80GB memory), and

###### Prompt for Character Interactions Task Definition

You will be provided with an image and a text prompt describing the main character’s action. As an experienced evaluator, your task is to evaluate the semantic consistency between the image and the text prompt, according to the scoring criteria. This evaluation focus specifically on whether the action of the main character in the image aligns with the action described in the text.

Scoring Criteria When evaluating the semantic consistency between an image and its corresponding text prompt, the following aspects are crucial:

- • Relevance: Does the image show the main character performing the action or behavior mentioned in the text? The action in the image should match the core description provided in the text.
- • Accuracy: Does the image depict the action correctly according to the text prompt? Any specific details related to the action, such as gestures, posture, or environment, should align with the description.
- • Completeness: Does the image show the main character completing the entire action as described in the text? The image should not omit important parts of the action or behavior.

Scoring Range Based on these criteria, you will assign a score from 0 to 4 that reflects the degree of semantic consistency between the image and the text prompt:

- • Very Poor (0): No correlation. The image does not reflect any aspect of the action described in the text prompt.
- • Poor (1): Weak correlation. The image addresses the text in a very general way but misses most details and accuracy of the action.
- • Fair (2): Moderate correlation. The image depicts the action to some extent, but there are several inaccuracies or missing details.
- • Good (3): Strong correlation. The image accurately portrays most elements of the action with minor inaccuracies or omissions.
- • Excellent (4): Near-perfect correlation. The image closely aligns with the text prompt and portrays the main character’s action with high accuracy and precision.

###### Input format

Every time you will receive a text prompt and an image. Please carefully review the image and text prompt. Before giving a score, please provide a brief analysis of the above evaluation criteria, which should be very concise and accurate.

Output Format Analysis: <Your analysis> Score: <Your Score>

###### Prompt for Shooting Evaluation Task Definition

You will be provided with an image and a text prompt describing the shot type of the image. As an experienced evaluator, your task is to assess whether the generated image meets the specified shot requirements based on the evaluation criteria.

Additional Material Instruction: You are a professor evaluator. Below is information about different shot types and shot distances. Please evaluate whether the generated image meets the requested shot type.

###### Shot Distance Descriptions

- • Long Shot: Shows the relationship between characters and their environment, typically used to display the scene or environment.
- • Full Shot: Shows the full body of a character, commonly used to display movement or the full scene.
- • Medium Long Shot: Starts from above the character’s knees, capturing part of the environment.
- • Medium Shot: Captures the character from the waist up.
- • Close-Up: Captures the character from the chest up.
- • Extreme Close-Up: focus on the character’s head or face, with the background and environment typically blurred or not visible.

###### Angle Descriptions

- • Eye Level Shot: The camera is positioned at the subject’s eye level.
- • Low Angle Shot: The camera is positioned below eye level, shooting upward, emphasizing the character’s power or size.
- • High Angle Shot: The camera is positioned above eye level, shooting downward, often minimizing the subject’s significance.
- • Bird’s Eye View: Camera shot taken from directly above, providing an overview of the scene.
- • Tilted Shot: The camera is intentionally tilted to create a sense of imbalance or tension.
- • Perspective Compression: A technique that emphasizes depth and the relationship between foreground and background through perspective.

Scoring Range A score between 0 and 4 will be assigned based on how well the shot type aligns with the content described in the prompt:

- • Very Poor (0): The image does not meet any shot or angle requirements.
- • Poor (1): The image meets some but not most of the shot or angle requirements.
- • Fair (2): The image partially meets the shot or angle requirements, but some elements are off.
- • Good (3): The image meets most of the shot or angle requirements.
- • Excellent (4): The image fully meets all of the shot and angle requirements.

Input Format You will receive a text prompt and an image. Please carefully review the image and text prompt. Provide an analysis followed by a score.

Output Format Analysis: <Your analysis > Score: <Your score>

###### Prompt for Static Shot Evaluation Task Definition

You will be provided with an image and a text prompt that describes the background, objects, and mood of the scene (excluding characters). Your task is to evaluate the consistency between the background and objects described in the prompt and what is visually represented in the image.

###### Evaluation Criteria

When assessing the semantic consistency between the image and the text prompt, focus on how well the background and non-character elements in the image match the description provided in the text. The evaluation should be based on the following aspects:

- • Relevance: The image should clearly relate to the primary background elements and objects described in the text. It should reflect the main setting and environment described, without introducing irrelevant or unrelated features.
- • Accuracy: Check if the specific details mentioned in the text are correctly represented in the image. This includes any mentioned objects, scenery, environmental conditions (e.g., weather, lighting), and relevant background elements.
- • Completeness: Evaluate whether the image accurately includes all critical background elements described in the text. The image should reflect the key details and setting, not leaving out essential aspects of the described background or scene.
- • Context: The image should maintain the context of the description. If the text describes a specific environment or atmosphere, the image must capture that context appropriately, considering the described mood and setting elements.

Scoring Criteria Based on these factors, the image will be assigned a score from 0 to 4, indicating the degree of consistency between the image and the description in the text:

- • Very Poor (0): No correlation. The image completely fails to reflect the background or objects described in the text.
- • Poor (1): Weak correlation. The image touches on the background or objects in a very general sense but misses most of the important details or has significant inaccuracies.
- • Fair (2): Moderate correlation. The image contains some relevant background and objects, but several important details are missing or inaccurately represented.
- • Good (3): Strong correlation. The image accurately represents most of the described background and objects with minor omissions or inaccuracies.
- • Excellent (4): Near-perfect correlation. The image perfectly captures the background and objects as described in the text, leaving no significant details missing or inaccurate.

Input Format You will receive a text prompt and an image. Please carefully review the image and text prompt. Provide an analysis followed by a score.

Output Format Analysis: <Your analysis> Score: <Your Score>

Prompt for Individual Action Evaluation Task Definition

For each evaluation, you will receive a text prompt, an image, and a character name. Your task is to first extract the individual action or behavior of the specified character from the text prompt, then determine whether the image accurately reflects this description for that character, and finally assign a score based on the criteria.

Evaluation Process

- • Extract Action Information: Carefully extract the specific action or behavior described for the given character (character name) from the text prompt.
- • Image Comparison: Examine the image to determine whether the specified character’s action matches the extracted description.
- • Analyze and Score: Analyze the match according to the scoring criteria and assign a score.

Scoring Criteria Focus on the following aspects when evaluating:

- • Relevance: Does the image show the specified character performing the action or behavior described in the text?
- • Accuracy: Are the details of the character’s action in the image (such as posture, gestures, environment) consistent with the text description?
- • Completeness: Does the image fully depict the character completing the entire action as described, without omitting important parts?

Assign a score from 0 to 4 based on the degree of semantic consistency:

- • 0 (Very Poor): No correlation. The image does not reflect any aspect of the described action.
- • 1 (Poor): Weak correlation. The image only generally addresses the text, missing most details and accuracy.
- • 2 (Fair): Moderate correlation. The image depicts the action to some extent but with several inaccuracies or missing details.
- • 3 (Good): Strong correlation. The image accurately portrays most elements of the action, with only minor inaccuracies or omissions.
- • 4 (Excellent): Near-perfect correlation. The image closely aligns with the text prompt and depicts the character’s action with high accuracy and completeness.

Output Format Analysis: <Your analysis> Score: <Your Score>

##### References

- [1] Emanuele Bugliarello, H Hernan Moraldo, Ruben Villegas, Mohammad Babaeizadeh, Mohammad Taghi Saffar, Han Zhang, Dumitru Erhan, Vittorio Ferrari, Pieter-Jan Kindermans, and Paul Voigtlaender. Storybench: A multifaceted benchmark for continuous story visualization. NeurIPS,

2023. 3

- [2] ByteDance Inc. Doubao ai assistant. https://www. doubao.com/, 2024. Accessed: 2025-04-16. 6, 7, 10, 20, 22, 23, 27, 30
- [3] Hong Chen, Rujun Han, Te-Lin Wu, Hideki Nakayama, and Nanyun Peng. Character-centric story visualization via visual planning and token alignment. arXiv preprint arXiv:2210.08465, 2022. 2
- [4] Siyu Chen, Dengjie Li, Zenghao Bao, Yao Zhou, Lingfeng Tan, Yujie Zhong, and Zheng Zhao. Manga generation via layout-controllable diffusion. arXiv preprint arXiv:2412.19303, 2024. 2
- [5] Junhao Cheng, Xi Lu, Hanhui Li, Khun Loun Zai, Baiqiao Yin, Yuhao Cheng, Yiqiang Yan, and Xiaodan Liang. Autostudio: Crafting consistent subjects in multi-turn interactive image generation. arXiv preprint arXiv:2406.01388,

2024. 2

- [6] Junhao Cheng, Baiqiao Yin, Kaixin Cai, Minbin Huang, Hanhui Li, Yuxin He, Xi Lu, Yue Li, Yifei Li, Yuhao Cheng, et al. Theatergen: Character management with llm for consistent multi-turn image generation. arXiv preprint arXiv:2404.18919, 2024. 2, 6, 7, 8, 10, 20, 22, 23, 26, 30
- [7] Wei Cheng, Su Xu, Jingtan Piao, Chen Qian, Wayne Wu, Kwan-Yee Lin, and Hongsheng Li. Generalizable neural performer: Learning robust radiance fields for human novel view synthesis. arXiv preprint arXiv:2204.11798, 2022. 13
- [8] Wei Cheng, Ruixiang Chen, Siming Fan, Wanqi Yin, Keyu Chen, Zhongang Cai, Jingbo Wang, Yang Gao, Zhengming Yu, Zhengyu Lin, et al. Dna-rendering: A diverse neural actor repository for high-fidelity human-centric rendering. In ICCV, 2023. 13
- [9] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 2, 6, 17, 28
- [10] Karan Dalal, Daniel Koceja, Gashon Hussein, Jiarui Xu, Yue Zhao, Youjin Song, Shihao Han, Ka Chun Cheung, Jan Kautz, Carlos Guestrin, Tatsunori Hashimoto, Sanmi Koyejo, Yejin Choi, Yu Sun, and Xiaolong Wang. Oneminute video generation with test-time training. arXiv preprint arXiv:2504.05298, 2025. 2
- [11] Google DeepMind. Gemini 2.0 flash: Native image generation in google ai studio. https : / / developers . googleblog . com / en / experiment - with - gemini - 20 - flash native- image- generation/, 2025. Accessed: 2025-04-16. 2, 6, 7, 11, 22, 23, 28, 30
- [12] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos

- Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, 2019. 4
- [13] discus0434. Aesthetic predictor v2.5. https : / / github . com / discus0434 / aesthetic predictor-v2-5, 2024. Accessed: 2025-05-10. 6
- [14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 26
- [15] Patrick Esser, Sumith Kulal, Ajay Jitkasan, Andkhuja Rakhimov, Dawid Filip, Meng Du, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024. 10
- [16] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344,

2023. 3

- [17] Silin Gao, Sheryl Mathew, Li Mi, Sepideh Mamooler, Mengjie Zhao, Hiromi Wakaki, Yuki Mitsufuji, Syrielle Montariol, and Antoine Bosselut. Vinabench: Benchmark for faithful and consistent visual narratives. arXiv preprint arXiv:2503.20871, 2025. 2, 3, 32
- [18] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, Xunsong Li, Yifu Li, Shanchuan Lin, Zhijie Lin, Jiawei Liu, Shu Liu, Xiaonan Nie, Zhiwu Qing, Yuxi Ren, Li Sun, Zhi Tian, Rui Wang, Sen Wang, Guoqiang Wei, Guohong Wu, Jie Wu, Ruiqi Xia, Fei Xiao, Xuefeng Xiao, Jiangqiao Yan, Ceyuan Yang, Jianchao Yang, Runkai Yang, Tao Yang, Yihang Yang, Zilyu Ye, Xuejiao Zeng, Yan Zeng, Heng Zhang, Yang Zhao, Xiaozheng Zheng, Peihao Zhu, Jiaxin Zou, and Feilong Zuo. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025. 2
- [19] Ridouane Ghermi, Xi Wang, Vicky Kalogeiton, and Ivan Laptev. Short film dataset (sfd): A benchmark for story-level video understanding. arXiv preprint arXiv:2406.10221, 2024. 3
- [20] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025. 2, 6, 7, 22, 23, 28
- [21] Google. Nano Banana: Image editing in Gemini just got a major upgrade. https://gemini.google.com/ nano-banana, 2025. 2, 7, 22, 23
- [22] Google. Nano Banana Pro: Image editing in Gemini just got a major upgrade. https://deepmind.google/ models/gemini-image/pro/, 2025. 7, 22
- [23] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv preprint arXiv:2503.10589, 2025. 2
- [24] Tanmay Gupta, Dustin Schwenk, Ali Farhadi, Derek Hoiem, and Aniruddha Kembhavi. Imagine this! scripts to compositions to videos. In ECCV, 2018. 3, 11, 14

- [25] Hui Han, Siyuan Li, Jiaqi Chen, Yiwen Yuan, Yuling Wu, Chak Tou Leong, Hanwen Du, Junchen Fu, Youhua Li, Jie Zhang, et al. Video-bench: Human-aligned video generation benchmark. arXiv preprint arXiv:2504.04907, 2025. 3
- [26] Sebastian Hartwig, Dominik Engel, Leon Sick, Hannah Kniesel, Tristan Payer, Poonam Poonam, Michael Gl¨ockler, Alex B¨auerle, and Timo Ropinski. A survey on quality metrics for text-to-image generation. arXiv preprint arXiv:2403.11821, 2024. 2
- [27] Huiguo He, Huan Yang, Zixi Tuo, Yuan Zhou, Qiuyue Wang, Yuhang Zhang, Zeyu Liu, Wenhao Huang, Hongyang Chao, and Jian Yin. Dreamstory: Open-domain story visualization by llm-guided multi-subject consistent diffusion. arXiv preprint arXiv:2407.12899, 2024. 2
- [28] Jingwen He, Hongbo Liu, Jiajun Li, Ziqi Huang, Yu Qiao, Wanli Ouyang, and Ziwei Liu. Cut2next: Generating next shot via in-context tuning. arXiv preprint arXiv:2508.08244, 2025. 2
- [29] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021. 2
- [30] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017. 2
- [31] Panwen Hu, Jin Jiang, Jianqi Chen, Mingfei Han, Shengcai Liao, Xiaojun Chang, and Xiaodan Liang. Storyagent: Customized storytelling video generation via multi-agent collaboration. arXiv preprint arXiv:2411.04925, 2024. 2
- [32] Haoyang Huang, Guoqing Ma, Nan Duan, Xing Chen, Changyi Wan, Ranchen Ming, Tianyu Wang, Bo Wang, Zhiying Lu, Aojie Li, et al. Step-video-ti2v technical report: A state-of-the-art text-driven image-to-video generation model. arXiv preprint arXiv:2503.11251, 2025. 2
- [33] Kaiyi Huang, Yukun Huang, Xintao Wang, Zinan Lin, Xuefei Ning, Pengfei Wan, Di Zhang, Yu Wang, and Xihui Liu. FilMaster: Bridging cinematic principles and generative ai for automated film generation, 2025. 2
- [34] Ting-Hao Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Aishwarya Agrawal, Jacob Devlin, Ross Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, et al. Visual storytelling. In NAACL, 2016. 3, 11, 14
- [35] Yuzhou Huang, Yiran Qin, Shunlin Lu, Xintao Wang, Rui Huang, Ying Shan, and Ruimao Zhang. Story3d-agent: Exploring 3d storytelling visualization with large language models. arXiv preprint arXiv:2408.11801, 2024. 2
- [36] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 2, 6, 7, 8, 10, 11, 17, 22, 23, 28, 30
- [37] Taewon Kang, Divya Kothandaraman, and Ming C. Lin. Text2story: Advancing video storytelling with text guidance. arXiv preprint arXiv:2503.06310, 2025. 2
- [38] Ozgur Kara, Krishna Kumar Singh, Feng Liu, Duygu Ceylan, James M. Rehg, and Tobias Hinz. Shotadapter: Textto-multi-shot video generation with diffusion models. In

- Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 2
- [39] Minchul Kim, Anil K Jain, and Xiaoming Liu. Adaface: Quality adaptive margin for face recognition. In CVPR,

2022. 4

- [40] Bowen Li and Thomas Lukasiewicz. Word-level fine-grained story visualization. arXiv preprint arXiv:2208.02341, 2022. 2
- [41] Junzhe Li, Sifan Zhou, Liya Guo, Xuerui Qiu, Linrui Xu, Delin Qu, Tingting Long, Chun Fan, Ming Li, Hehe Fan, Jun Liu, and Shuicheng Yan. Uniface: A unified finegrained face understanding and generation model. arXiv preprint arXiv:2503.08120, 2025. 2
- [42] Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, Lawrence Carin, David Carlson, and Jianfeng Gao. Storygan: A sequential conditional gan for story visualization. In CVPR, 2019. 2, 3, 11, 14
- [43] Yunxin Li, Haoyuan Shi, Baotian Hu, Longyue Wang, Jiashun Zhu, Jinyi Xu, Zhen Zhao, and Min Zhang. Animdirector: A large multimodal model powered agent for controllable animation video generation. In SIGGRAPH Asia,

2024. 2, 6, 7, 20, 22, 23, 26, 30

- [44] Zuzeng Lin, Ailin Huang, and Zhewei Huang. Collaborative neural rendering using anime character sheets. arXiv preprint arXiv:2207.05378, 2022. 3
- [45] Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yanfeng Wang, and Weidi Xie. Intelligent grimm - openended visual storytelling via latent diffusion models. In CVPR, 2024. 3, 6, 7, 8, 11, 14, 20, 22, 23, 25, 28, 30
- [46] Hongbo Liu, Jingwen He, Yi Jin, Dian Zheng, Yuhao Dong, Fan Zhang, Ziqi Huang, Yinan He, Yangguang Li, Weichao Chen, Yu Qiao, Wanli Ouyang, Shengjie Zhao, and Ziwei Liu. Shotbench: Expert-level cinematic understanding in vision-language models, 2025. 3
- [47] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, 2024. 4
- [48] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024. 2
- [49] Hongbo Ma, Fei Shen, Hongbin Xu, Xiaoce Wang, Gang Xu, Jinkai Zheng, Liangqiong Qu, and Ming Li. Styletailor: Towards personalized fashion styling via hierarchical negative feedback. arXiv preprint arXiv:2508.06555, 2025. 2
- [50] MagicLight AI. Brmgo: Ai-powered tool for story script generation. https://brmgo.cn/, 2025. Accessed: 2025-04-16. 6, 7, 20, 22, 23, 27, 30, 34
- [51] Adyasha Maharana and Mohit Bansal. Integrating visuospatial, linguistic and commonsense structure into story visualization. arXiv preprint arXiv:2110.10834, 2021. 2
- [52] Jiawei Mao, Xiaoke Huang, Yunfei Xie, Yuanqi Chang, Mude Hui, Bingjie Xu, and Yuyin Zhou. Story-Adapter:

- A Training-free Iterative Framework for Long Story Visualization, 2024. 6, 7, 10, 11, 20, 22, 23, 25, 30
- [53] Maoyan Entertainment. Shenbi: Ai-powered scriptwriting tool by maoyan. https://shenbi.maoyan.com/,

2025. Accessed: 2025-04-16. 6, 7, 22, 23, 27, 30, 34

- [54] Meitu Inc. Moki: Ai short film creation tool. https: //www.moki.cn, 2024. 2, 6, 7, 10, 22, 23, 26, 30, 34
- [55] Chutian Meng, Fan Ma, Chi Zhang, Jiaxu Miao, Yi Yang, and Yueting Zhuang. Logistory: A logic-aware framework for multi-image story visualization. In The Fourteenth International Conference on Learning Representations, 2026. 2
- [56] Yihao Meng, Hao Ouyang, Yue Yu, Qiuyu Wang, Wen Wang, Ka Leong Cheng, Hanlin Wang, Yixuan Li, Cheng Chen, Yanhong Zeng, Yujun Shen, and Huamin Qu. Holocine: Holistic generation of cinematic multi-shot long video narratives. arXiv preprint arXiv:2510.20822, 2025. 2
- [57] Ruibo Ming, Zhewei Huang, Zhuoxuan Ju, Jianming Hu, Lihui Peng, and Shuchang Zhou. A survey on future frame synthesis: Bridging deterministic and generative approaches. arXiv preprint arXiv:2401.14718, 2024. 2
- [58] Morphic, Inc. Introducing morphic studio. https:// www.morphic.com/, 2024. Accessed: 2025-04-16. 2, 6, 7, 10, 22, 23, 27, 30
- [59] Daniel A. P. Oliveira and David Martins de Matos. Storyreasoning dataset: Using chain-of-thought for scene understanding and grounded story generation, 2025. 2
- [60] OpenAI. Sora 2. https://sora.chatgpt.com/,

2025. Accessed: 2025-11-11. 2, 6, 7, 10, 11, 17, 22, 23, 28

- [61] OpenAI. Gpt-4.1: Advanced large language model for natural language understanding and generation. https: //openai.com/research/gpt-4-1, 2025. Accessed: 2025-04-16. 31, 32, 33
- [62] Dongwei Pan, Long Zhuo, Jingtan Piao, Huiwen Luo, Wei Cheng, Yuxin Wang, Siming Fan, Shengqi Liu, Lei Yang, Bo Dai, et al. Renderme-360: a large digital asset library and benchmarks towards high-fidelity head avatars. NeurIPS, 2023. 13
- [63] Xichen Pan, Pengda Qin, Yuhong Li, Hui Xue, and Wenhu Chen. Synthesizing coherent story with auto-regressive latent diffusion models. In WACV, 2024. 2
- [64] Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai, Chunrui Han, Zheng Ge, Xiangyu Zhang, and Shu-Tao Xia. Dreambench++: A human-aligned benchmark for personalized image generation. arXiv preprint arXiv:2406.16855, 2024. 2, 3
- [65] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 4, 13
- [66] Qwen Team. Qwen-image technical report. https:// qwen.ai/blog/qwen-image, 2025. 2, 6, 7, 22, 23
- [67] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 4, 34

- [68] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In NeurIPS, 2016. 6
- [69] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In CVPR, 2015. 4
- [70] Fei Shen, Hu Ye, Sibo Liu, Jun Zhang, Cong Wang, Xiao Han, and Wei Yang. Boosting consistency in story visualization with rich-contextual conditional diffusion models. arXiv preprint arXiv:2407.02482, 2024. 2
- [71] Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Measuring style similarity in diffusion models. arXiv preprint arXiv:2404.01292, 2024. 4, 34
- [72] Quanjian Song, Donghao Zhou, Jingyu Lin, Fei Shen, Jiaze Wang, Xiaowei Hu, Cunjian Chen, and Pheng-Ann Heng. Scenedecorator: Towards scene-oriented story generation with scene planning and scene consistency. arXiv preprint arXiv:2510.22994, 2025. 2
- [73] Tom´aˇs Souˇcek and Jakub Lokoˇc. Transnet v2: An effective deep network architecture for fast shot transition detection. arXiv preprint arXiv:2008.04838, 2020. 28
- [74] Zihan Su, Xuerui Qiu, Hongbin Xu, Tangyu Jiang, Junhao Zhuang, Chun Yuan, Ming Li, Shengfeng He, and Fei Richard Yu. Safe-sora: Safe text-to-video generation via graphical watermarking. arXiv preprint arXiv:2505.12667, 2025. 2
- [75] Wenzhang Sun, Zhenyu Wang, Zhangchi Hu, Chunfeng Wang, Hao Li, and Wei Chen. Muse: A multi-agent framework for unconstrained story envisioning via closed-loop cognitive orchestration. arXiv preprint arXiv:2602.03028,

2026. 2

- [76] Ming Tao, Bing-Kun Bao, Hao Tang, Yaowei Wang, and Changsheng Xu. Storyimager: A unified and efficient framework for coherent story visualization and completion. arXiv preprint arXiv:2404.05979, 2024. 2
- [77] Chongjun Tu, Lin Zhang, Pengtao Chen, Peng Ye, Xianfang Zeng, Wei Cheng, Gang Yu, and Tao Chen. Favorbench: A comprehensive benchmark for fine-grained video motion understanding. arXiv preprint arXiv:2503.14935,

2025. 2

- [78] TypeMovie Team. Typemovie: Text-to-video storytelling with style and rhythm. https://typemovie.art,

2024. Accessed: 2025-04-16. 6, 7, 22, 23, 27, 30

- [79] Bo Wang, Haoyang Huang, Zhiying Lu, Fengyuan Liu, Guoqing Ma, Jianlong Yuan, Yuan Zhang, Nan Duan, and Daxin Jiang. Storyanchors: Generating consistent multiscene story frames for long-form narratives. arXiv preprint arXiv:2505.08350, 2025. 2
- [80] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In AAAI, 2023. 2
- [81] Jiuniu Wang, Zehua Du, Yuyuan Zhao, Bo Yuan, Kexiang Wang, Jian Liang, Yaxi Zhao, Yihen Lu, Gengliang Li, Junlong Gao, Xin Tu, and Zhenyu Guo. Aesopagent: Agentdriven evolutionary system on story-to-video production. arXiv preprint arXiv:2403.07952, 2024. 2

- [82] Mengyu Wang, Henghui Ding, Jianing Peng, Yao Zhao, Yunpeng Chen, and Yunchao Wei. Characonsist: Finegrained consistent character generation. arXiv preprint arXiv:2507.11533, 2025. 2, 6, 7, 22, 23
- [83] Wenjia Wang, Liang Pan, Zhiyang Dou, Jidong Mei, Zhouyingcheng Liao, Yuke Lou, Yifan Wu, Lei Yang, Jingbo Wang, and Taku Komura. Sims: Simulating stylized human-scene interactions with retrieval-augmented script generation. arXiv preprint arXiv:2411.19921, 2024. 2
- [84] Ziyi Wang, Songbai Tan, Gang Xu, Xuerui Qiu, Hongbin Xu, Xin Meng, Ming Li, and Fei Richard Yu. Safe-var: Safe visual autoregressive model for text-to-image generative watermarking. arXiv preprint arXiv:2503.11324, 2025. 2
- [85] Mike Zheng Shou Weijia Wu, Zeyu Zhu. Automated movie generation via multi-agent cot planning, 2025. 2, 7, 10, 20, 22, 23, 26, 30
- [86] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871,

2025. 2, 3, 6, 7, 10, 20, 22, 23

- [87] Jianzong Wu, Chao Tang, Jingbo Wang, Yanhong Zeng, Xiangtai Li, and Yunhai Tong. Diffsensei: Bridging multimodal llms and diffusion models for customized manga generation. arXiv preprint arXiv:2412.07589, 2024. 2
- [88] Shaojin Wu, Mengqi Huang, Yufeng Cheng, Wenxu Wu, Jiahe Tian, Yiming Luo, Fei Ding, and Qian He. Uso: Unified style and subject-driven generation via disentangled and reward learning. 2025. 2
- [89] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation.

- arXiv preprint arXiv:2504.02160, 2025. 2, 6

[90] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation.

- arXiv preprint arXiv:2504.02160, 2025. 2, 7, 20, 22, 23, 25, 30

- [91] Weijia Wu, Mingyu Liu, Zeyu Zhu, Xi Xia, Haoen Feng, Wen Wang, Kevin Qinghong Lin, Chunhua Shen, and Mike Zheng Shou. Moviebench: A hierarchical movie level dataset for long video generation. arXiv preprint arXiv:2411.15262, 2024. 3
- [92] Weijia Wu, Zeyu Zhu, and Mike Zheng Shou. Automated movie generation via multi-agent cot planning. arXiv preprint arXiv:2503.07314, 2025. 2, 6
- [93] Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan Yuille, and Lu Jiang. Captain cinema: Towards short movie generation. arXiv preprint arXiv:2507.18634, 2025. 2
- [94] Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, and Yu-Gang Jiang. A survey on video diffusion models. ACM Computing Surveys, 2023. 2
- [95] Hongbin Xu, Chaohui Yu, Feng Xiao, Jiazheng Xing, Hai Ci, Weitao Chen, Fan Wang, and Ming Li. Cyc3d: Fine-

- grained controllable 3d generation via cycle consistency regularization. arXiv preprint arXiv:2504.14975, 2025. 2
- [96] Xuenan Xu, Jiahao Mei, Chenliang Li, Yuning Wu, Ming Yan, Shaopeng Lai, Ji Zhang, and Mengyue Wu. Mmstoryagent: Immersive narrated storybook video generation with a multi-agent paradigm across text, image and audio. arXiv preprint arXiv:2503.05242, 2024. 2, 6, 7, 22, 23, 26, 30
- [97] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 6, 33
- [98] Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen. Seed-story: Multimodal long story generation with large language model. arXiv preprint arXiv:2407.08683, 2024. 2, 3, 6, 7, 10, 11, 14, 20, 22, 23, 26, 30
- [99] Zilyu Ye, Jinxiu Liu, JinJin Cao, Zhiyang Chen, Ziwei Xuan, Mingyuan Zhou, Qi Liu, and Guo-Jun Qi. Openstory: A large-scale open-domain dataset for subject-driven visual storytelling. In CVPR, 2024. 11, 14
- [100] Xingxi Yin, Yicheng Li, Gong Yan, Chenglin Li, Jian Zhao, Cong Huang, Yue Deng, and Yin Zhang. 2kcharacters-10k-stories: A quality-gated stylized narrative dataset with disentangled control and sequence consistency. arXiv preprint arXiv:2512.05557, 2025. 3
- [101] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023. 6
- [102] Chi Zhang, Yuanzhi Liang, Xi Qiu, Fangqiu Yi, and Xuelong Li. Vast 1.0: A unified framework for controllable and consistent video generation. arXiv preprint arXiv:2412.16677, 2024. 2
- [103] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection, 2022. 4
- [104] Jiaxu Zhang, Tianshu Hu, Yuan Zhang, Zenan Li, Linjie Luo, Guosheng Lin, and Xin Chen. Bridging imagination with audio-video generation via a unified director. arXiv:2512.23222, 2025. 2
- [105] Jinlu Zhang, Jiji Tang, Rongsheng Zhang, Tangjie Lv, and Xiaoshuai Sun. Storyweaver: A unified world model for knowledge-enhanced story character customization. AAAI,

- 2025. 2

[106] Jinlu Zhang, Qiyun Wang, Baoxiang Du, Jiayi Ji, Jing He, Rongsheng Zhang, Tangjie Lv, Xiaoshuai Sun, and Rongrong Ji. Persistent story world simulation with continuous character customization. arXiv preprint arXiv:2603.16285,

- 2026. 2

- [107] Kaiwen Zhang, Liming Jiang, Angtian Wang, Jacob Zhiyuan Fang, Tiancheng Zhi, Qing Yan, Hao Kang, Xin Lu, and Xingang Pan. Storymem: Multi-shot long video storytelling with memory. arXiv preprint arXiv:2512.19539, 2025. 2
- [108] Canyu Zhao, Mingyu Liu, Wen Wang, Jianlong Yuan, Hao Chen, Bo Zhang, and Chunhua Shen. Moviedreamer: Hi-

erarchical generation for coherent long visual sequence,

2024. 2

- [109] Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Hao Li, Zicheng Zhang, Guangtao Zhai, Junchi Yan, Hua Yang, Xue Yang, and Haodong Duan. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025. 3
- [110] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025. 3, 9
- [111] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. NeurIPS, 2024. 2, 6, 7, 11, 20, 22, 23, 30
- [112] Shudong Zhu(Andrew Zhu). Long prompt weighted stable diffusion embedding. https://github.com/ xhinker/sd_embed, 2024. 24, 25, 26
- [113] Cailin Zhuang, Yaoqi Hu, Xuanyang Zhang, Wei Cheng, Jiacheng Bao, Shengqi Liu, Yiying Yang, Xianfang Zeng, Gang Yu, and Ming Li. Styleme3d: Stylization with disentangled priors by multiple encoders on 3d gaussians. arXiv preprint arXiv:2504.15281, 2025. 34
- [114] Shaobin Zhuang, Kunchang Li, Xinyuan Chen, Yaohui Wang, Ziwei Liu, Yu Qiao, and Yali Wang. Vlogger: Make your dream a vlog. In CVPR, 2024. 6, 7, 10, 20, 22, 23, 26, 30

