# arXiv:2605.30557v1[cs.CV]28May2026

## Seeing Isn’t Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)?

#### Yue Zhang1 Zun Wang1 Han Lin1 Yonatan Bitton2 Idan Szpektor2 Mohit Bansal1

1UNC Chapel Hill 2Google Research https://spatialuncertain.github.io

### Abstract

Spatial reasoning is a fundamental capability for vision-language models deployed in real-world environments. However, visual observations are inherently limited representations of a 3D world: occlusion can render objects invisible, and perspective can make geometric properties misleading. Despite this, existing spatial reasoning benchmarks typically assume that observations are sufficient and reliable, focusing on whether models produce correct answers rather than whether they recognize when a question cannot be answered and what additional observations would be needed. In this work, we challenge this assumption by constructing a controlled evaluation framework, SPATIALUNCERTAIN, based on 3D simulated environments. We introduce two types of observation challenges: (1) occlusion, which hides target information, and (2) perspective ambiguity, which produces misleading visual cues. For each configuration, we design spatial questions that are answerable under clean observations but require abstention under the introduced challenges. We further evaluate whether models can identify which additional viewpoints would resolve perspective ambiguity. Our results across a diverse set of frontier open- and closed-source vision-language models (e.g., GPT-4o, GPT-5.4, Gemini-3.0-Flash, Qwen2.5-VL, InternVL) reveal two consistent failure modes. First, models are prone to overconfident answering, attempting to solve spatial reasoning tasks even when visual evidence is incomplete or misleading, with average accuracy around 30% under occlusion and below 10% under perspective ambiguity. Second, even when additional views are available, some models perform near random chance in identifying which would provide reliable evidence. We further show that visual input is beneficial when information is missing, but can actively mislead models under perspective ambiguity. To investigate whether these failures can be mitigated, we compare prompting strategies and fine-tuning approaches. Structured prompting partially improves abstention but introduces a trade-off with answerable accuracy. In contrast, fine-tuning on diverse ambiguity conditions yields more robust observational uncertainty, suggesting that this capability is learnable but requires exposure to different uncertainty signals. Together, our findings call for moving beyond answer correctness toward evaluating whether models know when to abstain and how to seek reliable evidence.

### 1 Introduction

Recent advances in Multimodal Large Language Models (MLLMs) [Liu et al., 2023, Singh et al., 2025, Deepmind, 2025a] have enabled intelligent agents to perceive and interact with their environments, bringing us closer to practical embodied systems [Zhang et al., 2024a]. A fundamental capability underlying these systems is spatial reasoning, which has been extensively studied through a growing number of benchmarks [Yang et al., 2025a,b, Pothiraj et al., 2025, Wang et al., 2024a, Liu et al., 2025,

Preprint.

(b) Occlusion

###### (a) Clean View

###### Multiple 2D Observations

(c) Perspective Ambiguity

Unreliable Information

Sufficient Information

Missing Information

###### 3D Scene

“Locker Room”

[Figure 1]_images/imageFile1.png>)

###### Recognize Unreliable View Identify Informative View

[Figure 2]_images/imageFile2.png>)

[Figure 3]_images/imageFile3.png>)

[Figure 4]_images/imageFile4.png>)

[Figure 5]_images/imageFile5.png>)

[Figure 6]_images/imageFile6.png>)

[Figure 7]_images/imageFile7.png>)

| | |
|---|---|
| | |

[Figure 8]_images/imageFile8.png>)

[Figure 9]_images/imageFile9.png>)

[Figure 10]_images/imageFile10.png>)

[Figure 11]_images/imageFile11.png>)

View 1 View 2

[Figure 12]_images/imageFile12.png>)

[Figure 13]_images/imageFile13.png>)

###### …

[Figure 14]_images/imageFile14.png>)

[Figure 15]_images/imageFile15.png>)

[Figure 16]_images/imageFile16.png>)

View 3 View 4

Q: What’s the relationship between the bench and the door?

Q: What’s the relation between the locker unit and the bench?

###### Top View

Q: Are the two benches the same size?

Q: Which view can help answer the question?

“Locker Room”

[Figure 17]_images/imageFile17.png>)

[Figure 18]_images/imageFile18.png>)

###### Answerable Unanswerable

###### Unanswerable

[Figure 19]_images/imageFile19.png>)

Can select

[Figure 20]_images/imageFile20.png>)

[Figure 21]_images/imageFile21.png>)

[Figure 22]_images/imageFile22.png>)

The left “bench” appears wider due to perspective

Clear visual evidence of “locker and bench”

“Door” is visually occluded.

If Cannot informative view? determine

[Figure 23]_images/imageFile23.png>)

[Figure 24]_images/imageFile24.png>)

Model

###### Model

Model

###### Model

[Figure 25]_images/imageFile25.png>)

[Figure 26]_images/imageFile26.png>)

[Figure 27]_images/imageFile27.png>)

[Figure 28]_images/imageFile28.png>)

[Figure 29]_images/imageFile29.png>)

“Cannot Determine”

“Cannot Determine”

[Figure 30]_images/imageFile30.png>)

“Left/right”

“Yes/No”

“View 1”

“Left/right”

- Figure 1: Visual observations are inherently 2D projections of a 3D world and may provide sufficient, missing, or unreliable information for spatial reasoning. (a) Under clean views, questions are answerable from direct visual evidence. (b) Under occlusion, target information becomes invisible, requiring models to abstain with Cannot determine. (c) Under perspective ambiguity, geometric appearance becomes unreliable due to viewpoint bias, requiring models not only to recognize uncertainty but also to identify an informative reference view for reliable reasoning.

Jia et al., 2025, Stogiannidis et al., 2025]. These benchmarks have driven significant progress by measuring models’ ability to answer spatial questions (e.g., object relations, distance, or object size) from visual observations such as images or videos, but typically assume that visual input provides sufficient and reliable information (see Fig. 1(a)).

However, in practice, this assumption often breaks down. Visual observations are inherently 2D projections of a 3D world, where occlusion can hide critical objects and perspective can distort geometric properties, making spatial evidence incomplete or misleading (see Fig. 1(b) and (c)). Such unreliable observations are particularly challenging for embodied agents [Zhang et al., 2024a, Duan et al., 2022, Zhang and Kordjamshidi, 2023, Yu et al., 2026a], where acting on missing or misleading visual evidence can lead to incorrect action decisions or unsafe behaviors. Ideally, when visual evidence is incomplete or misleading, the appropriate behavior is not to guess, but to abstain, defer judgment, or actively seek additional observations. A similar shift has recently emerged in language modeling, where models are encouraged to express uncertainty or abstain when evidence is insufficient [Manakul et al., 2023, Stengel-Eskin et al., 2024, Wen et al., 2025]. In contrast, uncertainty awareness remains largely underexplored in visual spatial reasoning, where evaluation still predominantly focuses on answer correctness alone.

To address this gap, we introduce SPATIALUNCERTAIN, a controlled evaluation framework that evaluates whether models can recognize when visual observations are unreliable, and whether they can identify additional informative evidence rather than answer blindly. Specifically, we begin from clean 3D scenes where relevant spatial evidence is fully observable, ensuring that spatial questions are answerable under reliable observations. We then introduce two controlled observation perturbations. First, we simulate occlusion by inserting objects between the camera and the target, creating partial or full invisibility conditions that lead to missing information (Fig. 2(top)). Second, we introduce perspective-induced ambiguity by shifting the camera closer to one object, resulting in misleading visual cues that bias geometric perception (Fig. 2(bottom)). This setup allows the same spatial question to transition from answerable to unanswerable depending on the observation condition. Under occlusion or ambiguous perspectives, certain spatial questions can no longer be reliably resolved from the available visual evidence, so the appropriate behavior is not to guess, but to abstain or express uncertainty. Beyond recognizing unreliable observations, effective spatial reasoning also requires identifying what additional viewpoints are needed to resolve such unreliable visual evidence. Therefore, we introduce two complementary evaluation tasks: ViewSel, which directly measures viewpoint selection ability in isolation, and AbstainViewSel, which jointly evaluates whether models can first recognize an unreliable observation and then select an informative alternative viewpoint.

Using this controlled setup, we evaluate eight vision-language models spanning open-source (Qwen2.5-VL-7B, Qwen2.5-VL-32B, InternVL3-8B) and closed-source (GPT-4o, GPT-5-mini, GPT5.4, Gemini-2.5-Flash, Gemini-3.0-Flash) families. Our results reveal two major limitations in spatial reasoning under unreliable observations (Sec. 4.2): (i) While models achieve strong performance

when visual evidence is sufficient, they tend to produce confident answers even when observations are incomplete or misleading. (ii) models struggle to identify which additional viewpoints would provide reliable evidence, revealing limitations not only in abstention but also in actively acquiring informative observations. Beyond these failure modes, we uncover an additional asymmetry in how models use visual input (Sec. 4.3). Visual information is beneficial when evidence is missing, improving both answering and abstention under occlusion, but is far less reliable under perspective ambiguity. That said, when visual cues become misleading, adding visual input often degrades models’ ability to recognize unanswerable cases. We further explore whether these limitations can be mitigated (Sec. 4.4). We find that structured prompting can partially improve abstention, but introduces a trade-off with answerable accuracy, indicating that prompting alone is insufficient. In contrast, fine-tuning results suggest that abstention is a learnable capability, but only when models are trained on diverse forms of visual ambiguity. Together, our findings suggest that current MLLMs lack a unified understanding of observational reliability in spatial reasoning.

### 2 Related Work

Spatial reasoning in MLLMs. Spatial reasoning has emerged as a fundamental capability for multimodal large language models (MLLMs) [Chen et al., 2024, Cheng et al., 2024, Zhang et al.,

- 2024b], and a growing body of work has proposed benchmarks to evaluate it [Yang et al., 2025b,a, Yu et al., 2026b, Daxberger et al., 2025, Wang et al., 2024b, Xu et al., 2025, Rajabi and Kosecka, 2024, Yang et al., 2025c, Ma et al., 2022]. Early efforts focus on evaluating basic spatial relations such as relative relations, depth ordering, and size comparison using image/video-based question answering datasets. More recent benchmarks aim to provide broader and more systematic evaluations, including large-scale and multi-task settings such as SpatialEval [Yin et al., 2023] and OmniSpatial [Jia et al.,
- 2025], which cover diverse spatial reasoning skills ranging from object relations to complex scene understanding. Several works further emphasize the importance of controlled evaluation [Pothiraj et al., 2025, Liu et al., 2023, Johnson et al., 2017]. For example, What’sUp [Kamath et al., 2023] constructs minimally varying image pairs to isolate spatial relations. Despite these advances, existing benchmarks primarily evaluate whether models produce correct answers, but do not explicitly assess whether a question is answerable given the observation. In contrast, we evaluate spatial reasoning under varying observation conditions (e.g., occlusion and perspective ambiguity), focusing on whether models can recognize when the available evidence is reliable for spatial questions.

Observational uncertainty and abstention. Uncertainty estimation and abstention are important for building reliable models. Classical work on calibration and selective prediction shows that neural networks can be overconfident, and that models should sometimes abstain when their predictions are uncertain [Guo et al., 2017, Hendrycks and Gimpel, 2016, Geifman and El-Yaniv, 2017, Whitehead et al., 2022]. This problem has become especially important for large language models, which often generate fluent but unsupported answers. Recent work therefore studies truthfulness, self-knowledge, confidence elicitation, hallucination detection, and calibrated expressions of uncertainty [Lin et al.,

- 2022, Kadavath et al., 2022, Yin et al., 2023, Tian et al., 2023, Xiong et al., 2024, Manakul et al.,
- 2023, Stengel-Eskin et al., 2024, Wen et al., 2025]. While uncertainty and abstention have been widely explored in language models, they remain less studied in vision-language models. Related efforts examine object hallucination, unanswerable visual questions, and selective VQA, encouraging models to abstain rather than answer incorrectly [Rohrbach et al., 2018, Li et al., 2023, Sun et al.,
- 2024, Guan et al., 2024, Gurari et al., 2018, Guo et al., 2024, He et al., 2024, Eisenschlos et al., 2024]. However, these works primarily focus on factual uncertainty, object existence, or generic unanswerability, and typically assume that visual observations provide reliable evidence for reasoning. In contrast, we study observation-dependent uncertainty in spatial reasoning, where answerability is determined by the viewpoint.

### 3 SPATIALUNCERTAIN: Controlled Evaluation Framework

We construct a controlled evaluation framework using 3D simulated environments, and the pipeline is shown in Fig. 2. Starting from diverse indoor scenes (Sec. 3.1), we introduce two types of challenges: occlusion (Sec. 3.2) and perspective ambiguity (Sec. 3.3). On top of these configurations, we design spatial reasoning tasks whose answerability varies systematically with observation conditions (Sec. 3.4). All configurations undergo human validation to ensure quality (Sec. 3.5).

/Users/zhangyue/Desktop/Holodeck /clean_scene_layout/a_buffet-202604-07-18-45-15-036915/a_buffet.js on

###### QA Generation

###### Occlusion Pipeline

[Figure 31]_images/imageFile31.png>)

[Figure 32]_images/imageFile32.png>)

Q: From this view, is the target visible? Scene A: A. yes B.no

Fridge is partially occluded

Visibility

Select Target and Occluder

Inject Occluder

Partially Occluded

Clean View

[Figure 33]_images/imageFile33.png>)

[Figure 34]_images/imageFile34.png>)

Q: From this camera viewpoint, where is the target relative to the table? A: A. left B. right. C. cannot determine

[Figure 35]_images/imageFile35.png>)

[Figure 36]_images/imageFile36.png>)

Relative Relation

| | |
|---|---|

[Figure 37]_images/imageFile37.png>)

Answerable

Q:Which object is closer to the camera: the fridge or the table? A: A. cannot determine B full length mirror C. painting.

[Figure 38]_images/imageFile38.png>)

Depth Ordering

Fridge is fully occluded

Fully

[Figure 39]_images/imageFile39.png>)

Answerable Occluded

Q:Compared to the painting, is the full length mirror larger or smaller? A: A. large B. cannot determine C smaller D. the same.

Size Comp.

“A kitchen”

Target: “fridge” Occluder: “storage cabinet”

[Figure 40]_images/imageFile40.png>)

Unanswerable

QA Generation

###### ViewSelect (ViewSel)-> Single Stage

Reference View

Perspective Pipeline

[Figure 41]_images/imageFile41.png>)

[Figure 42]_images/imageFile42.png>)

[Figure 43]_images/imageFile43.png>)

[Figure 44]_images/imageFile44.png>)

Q: How many artworks can you see in this image? A: A. one B. two C. more than two

[Figure 45]_images/imageFile45.png>)

###### …

Visibility

Scene Select Object Pairs

[Figure 46]_images/imageFile46.png>)

[Figure 47]_images/imageFile47.png>)

Q: Based on this image, which one is horizontal longer? A: A. left B. right C. almost same D. cannot determine

Size Comp.

[Figure 48]_images/imageFile48.png>)

Answerable

Q: Which image can best decide physical size of two paintings?

Ambiguous View

Q: Based on this image, do they share similar proportions? A: A. cannot determine B. right is wider B: left is wider

Abstain-then-ViewSelect (AbstainViewSel)-> Two-Stage

Shape Comp.

[Figure 49]_images/imageFile49.png>)

|[Figure 50]_images/imageFile50.png>)<br><br>[Figure 51]_images/imageFile51.png>)<br><br>…<br><br>Q: Then select the view that best supports to answer.|
|---|

If Abstain

[Figure 52]_images/imageFile52.png>)

Answer or Cannot Determine

Q:Which object is closer to the table? A: A left B. cannot determine C. right D. the same.

Relative Position

“A buffet” Target:“Two artworks”

[Figure 53]_images/imageFile53.png>)

Unanswerable

[Figure 54]_images/imageFile54.png>)

Unanswerable

- Figure 2: Overview of our evaluation framework of SPATIALUNCERTAIN. (Top) Occlusion: A target object is occluded to create partial or full occlusion configurations, each paired with a clean reference. (Bottom) Perspective: Same-category object pairs are viewed from a reference (equidistant) and an ambiguous (shifted) camera position. We further introduce ViewSel (single-stage view selection) and AbstainViewSel (two-stage: abstain then select), evaluating whether models can identify informative viewpoints. We design four types of spatial reasoning questions. For questions highlighted in green, the correct behavior under fully occluded or ambiguous views is to abstain with Cannot determine.

#### 3.1 3D Scene Collection

We generate 3D indoor scenes using Holodeck [Yang et al., 2024], an LLM-based automated layout generation system. Given a natural language prompt (e.g., "a bedroom" or "a kitchen"), Holodeck uses a large language model (GPT-4o [OpenAI, 2024]) to plan object selection and placement, producing diverse and realistic room configurations. For each scene, Holodeck provides full 3D asset placement with object positions, orientations, and bounding box information, which we use to automatically derive ground truth annotations without the need for manual labeling. All scenes are rendered using AI2-THOR [Kolve et al., 2017], which supports controllable camera placement and produces photo-realistic RGB images. This controlled rendering environment is central to our framework: by fixing the 3D scene and varying only the camera viewpoint or object configuration, we can isolate how changes in observation reliability affect model reasoning.

#### 3.2 Occlusion Configurations

Target-occluder selection. For each clean scene, we select target objects satisfying two criteria: (1) visibility: the object must be clearly visible from the camera viewpoint, measured by its projected bounding box area; and (2) uniqueness: the object must be the only instance of its category in the scene, ensuring unambiguous reference in generated questions. We retain the top-k (k=3) targets per scene ranked by visibility score (more details are in the Appendix A.1). To construct the occluded scene, for each target, we select an occluder from the remaining objects based on two factors: (1) spatial proximity to the target, and (2) physical size sufficient to plausibly occlude it. In our setting, this procedure results in a diverse collection of target–occluder pairs, with targets spanning 225 unique object categories (e.g., bookshelf, coffee table, ottoman) and occluders spanning 286 categories (e.g., storage cabinet, bookshelf, armchair), covering a wide range of object types and occlusion scenarios.

Occlusion scene camera placement. Given a selected target-occluder pair, we place the occluder along the line of sight between the camera and the target, ensuring it is closer to the camera than the target (shown in Fig. 3a left). The placement is subject to several geometric constraints to ensure physical plausibility: (1) the occluder must not penetrate other objects or room boundaries, (2) a minimum depth separation is maintained between the occluder and target to avoid clipping, and (3) the occluder must remain within the same room area as the target. The modified scene layout is then re-rendered using AI2-THOR to produce the occluded view. In practice, due to irregular object geometry and shape variations, the realized occlusion may deviate from the intended configuration, resulting in both partial and full occlusion cases under similar placement conditions. To ensure

Question Type Occlusion Perspective

2,596

Visibility

2500

Relative

Answerable (A) Unanswerable (U)

2,292

###### (a) Occlusion (b) Perspective Ambiguity

Depth

| |
|---|

###### NumberofQuestions

Size / Shape

2000

1,857 1,857

Target Target Object A Object B

1,720

1500

1000

Occluder

[Figure 55]_images/imageFile55.png>)

[Figure 56]_images/imageFile56.png>)

500

View 1 View 2

[Figure 57]_images/imageFile57.png>)

[Figure 58]_images/imageFile58.png>)

[Figure 59]_images/imageFile59.png>)

0

Clean View Occluded View

Clean Partial Occ.

Full Occ.

Clean Perspective View

Reference View

(a) Camera placement under different conditions.

(b) Distribution of answerable vs. unanswerable.

- Figure 3: Camera placement under different conditions (left) and the resulting distribution of answerable vs. unanswerable questions across configurations (right).

accurate categorization, we perform human annotation to determine whether the target remains visible, and label each configuration as partial or full occlusion accordingly. Examples of the partial and full occlusion scenes are shown in Fig. 4(a).

#### 3.3 Perspective Ambiguity Configuration

Object pair selection. To induce perspective ambiguity, we select pairs of same-category objects with similar physical size, such that they appear comparable under neutral viewpoints but exhibit large appearance differences under perspective ambiguity. We consider two types of object pairs. Floor pairs consist of two floor-standing objects of the same type (e.g., two chairs) that are spatially proximate and share similar orientations, ensuring that they are comparable under a neutral viewpoint. Wall pairs consist of two wall-mounted objects (e.g., two paintings) placed on the same or adjacent walls, whose sizes are visually comparable, either along the horizontal or vertical dimension.

Perspective camera placement. As shown in Fig. 3a right, for each object pair, we generate two types of views while keeping the underlying 3D scene fixed. The reference view places the camera on the perpendicular bisector of the two objects at an equidistant position, ensuring both objects are fully visible and at equal distances from the camera. The perspective view translates the camera laterally along the axis connecting the two objects, bringing it closer to one object while keeping both within the field of view. This change in viewpoint induces systematic appearance differences without altering the underlying geometry. Specifically, for floor pairs, the nearer object appears larger due to size–distance effects. For wall pairs, the object viewed at an oblique angle appears foreshortened, altering its apparent proportions. As a result, objects with identical physical properties can exhibit conflicting visual cues under different viewpoints.

#### 3.4 Task Design

Spatial question design and answerability. We construct a unified set of spatial questions that are applied across all configurations, with ground truth derived automatically from 3D scene geometry. We consider four question types: Visibility, Relative position, Depth ordering, and Size/Shape. Visibility asks whether an object is observable from the current viewpoint, relative position, depth ordering capture spatial relationships, and size/shape probes geometric properties such as object size or proportions (see examples in Fig. 2). A key design principle is that answerability varies systematically with observation conditions. As shown in Fig. 3b, under clean observations, all questions are answerable, as visual evidence is complete and reliable. Under partial occlusion, questions remain answerable since the target is still partially observable. Under full occlusion, answerability depends on the question type: visibility remains answerable, while questions requiring access to the hidden target (relative position, depth, and size/shape) become unanswerable, with the correct response being Cannot determine. This setting introduces missing information. In contrast, under perspective ambiguity, visual information is not missing but can become unreliable. Under the reference view, all questions are answerable. However, under the perspective view, questions about size and shape cannot be reliably answered from visual appearance alone, as the apparent geometry no longer reflects the true physical properties. Meanwhile, visibility and relative position remain answerable, as they depend on geometric properties that are preserved under viewpoint changes.

/Users/zhangyue/Desktop/Holodeck /rendered_scene/occlusion_scene_ layout/an_art_studio-2026-04-08-14 -20-06-462834

an_office-2026-04-07-03-34-19-749 088

/Users/zhangyue/Desktop/Holodeck /view_selection_scenes_layout/a_li ving_room-2026-04-06-22-15-11-21 0277

/Users/zhangyue/Desktop/Holodeck /view_selection_scenes_layout/a_lo bby-2026-04-07-04-32-06-637407

/Users/zhangyue/Desktop/Holodeck /rendered_scene/occlusion_scene_ layout/a_garage-2026-04-07-00-1639-565497/occlusion_ladder-0_gar age_garage_fridge-0_garage

/Users/zhangyue/Desktop/Holodeck /rendered_scene/clean_scene_layo ut/a_gym-2026-02-02-20-31-04-487 623

/Users/zhangyue/Desktop/Holodeck /view_selection_scenes_layout/a_d eli-2026-04-07-13-47-53-404455

- (a) Occlusion Scenes

- (b) Perspective Scenes

Target:side table Target:Oven

Target: coffee station

Target: fridge

Target: barbell rack Target: stool seat

[Figure 60]_images/imageFile60.png>)

[Figure 61]_images/imageFile61.png>)

[Figure 62]_images/imageFile62.png>)

[Figure 63]_images/imageFile63.png>)

[Figure 64]_images/imageFile64.png>)

[Figure 65]_images/imageFile65.png>)

Clean Scenes

Occluder:visitor chair Occluder:cabinet Occluder:filing cabinet

Occluder:ladder Occluder:storage rack

Occluder: floor lamp

[Figure 66]_images/imageFile66.png>)

[Figure 67]_images/imageFile67.png>)

[Figure 68]_images/imageFile68.png>)

[Figure 69]_images/imageFile69.png>)

[Figure 70]_images/imageFile70.png>)

[Figure 71]_images/imageFile71.png>)

Occlusion Scenes (Full/Partial)

###### Full Occlusion Scenes Partial Occlusion Scenes

|[Figure 72]_images/imageFile72.png>)|
|---|

[Figure 73]_images/imageFile73.png>)

[Figure 74]_images/imageFile74.png>)

[Figure 75]_images/imageFile75.png>)

[Figure 76]_images/imageFile76.png>)

[Figure 77]_images/imageFile77.png>)

[Figure 78]_images/imageFile78.png>)

[Figure 79]_images/imageFile79.png>)

[Figure 80]_images/imageFile80.png>)

[Figure 81]_images/imageFile81.png>)

Object Pair: planter

Object Pair: dinner table

|[Figure 82]_images/imageFile82.png>)|
|---|

|[Figure 83]_images/imageFile83.png>)|
|---|

[Figure 84]_images/imageFile84.png>)

[Figure 85]_images/imageFile85.png>)

[Figure 86]_images/imageFile86.png>)

[Figure 87]_images/imageFile87.png>)

[Figure 88]_images/imageFile88.png>)

[Figure 89]_images/imageFile89.png>)

[Figure 90]_images/imageFile90.png>)

[Figure 91]_images/imageFile91.png>)

Object Pair: wall shelves

Object Pair: painting

Reference views are highlighted in red boxes.

- Figure 4: Examples of our controlled evaluation scenes. (a) Occlusion scenes: inserted objects create partial or full occlusion. (b) Perspective scenes: camera shifts introduce misleading views.

View selection under perspective ambiguity. Beyond recognizing when a question cannot be answered under an ambiguous viewpoint, a reliable model should also identify which additional viewpoint would provide sufficient evidence for reasoning. Therefore, we introduce two complementary tasks for viewpoint assessment under perspective ambiguity. ViewSelect (ViewSel): the model is presented with five candidate views (one informative reference view and four ambiguous alternatives) and asked to identify the view that best supports answering a spatial reasoning question about physical size. This metric evaluates pure viewpoint selection ability in isolation, independent of abstention behavior. Abstain-then-ViewSelect (AbstainViewSel): We further introduce a two-stage evaluation that jointly measures abstention and viewpoint selection. In Stage 1, the model is shown only the biased view and asked to answer the original question, including the option to abstain with Cannot determine. Stage 2 is triggered only if the model abstains. The model is then presented with the five candidate views and asked which would allow reliable reasoning. A prediction is counted as correct only if the model both correctly abstains in Stage 1 and successfully identifies the informative reference view in Stage 2.

#### 3.5 Human Validation and Statistics

We collect 240 unique scenes spanning 43 room types, including bedrooms, living rooms, buffets, museums, nurseries, and other common indoor environments. From these scenes, we construct 1,222 occlusion configurations (649 partial, 573 full) and 701 perspective object pairs across 390 scenes (334 floor pairs, 367 wall pairs). More examples of two types of scenes are shown in Fig. 4. Based on these controlled scenes, we finally generate 10,322 QA pairs: 6,608 from occlusion configurations (across 4 question types) and 3,714 from perspective configurations (across 4 question types). The distribution of answerable and unanswerable questions across conditions is summarized in Fig. 3b.

All scenes undergo human validation through a dedicated annotation interface. For occlusion scenes, annotators are presented with paired clean and occluded views side by side, with target and occluder objects labeled by name, and classify each configuration as no occlusion, partial occlusion, or full occlusion; configurations with no meaningful occlusion are discarded. For perspective scenes, annotators verify that the reference view provides sufficient visual evidence while the perspective view introduces visible geometric ambiguity, discarding configurations that fail this check. Full annotation interface details are in the Appendix A.1.1.

4 Experimental Results

#### 4.1 Evaluation Models and Protocol

We evaluate eight vision-language models spanning both open-source and closed-source families. Open-source models include Qwen2.5-VL-7B [Bai et al., 2025], Qwen2.5-VL-32B [Bai et al., 2025],

Table 1: Performance under occlusion and perspective ambiguity challenges. Ans. denotes accuracy on answerable questions, while Unans. measures the ability to correctly identify unanswerable cases. ViewS. and AbsViewS. correspond to the ViewSel and AbstainViewSel tasks, respectively, evaluating viewpoint selection with and without the abstention stage.

Model Occlusion Perspective Ambiguity

Ans. Unans. All Ans. Unans. All ViewS AbsViewS Random 32.3 23.3 30.0 25.0 25.9 25.0 20.0 4.0 Open-source

Qwen2.5-VL-7B [Bai et al., 2025] 51.1 39.3 48.0 62.4 41.5 57.8 24.6 8.6 Qwen2.5-VL-32B [Bai et al., 2025] 51.7 40.0 48.6 69.0 21.7 58.5 20.7 4.6 InternVL3-8B [Zhu et al., 2025] 61.7 7.3 47.5 70.4 1.1 55.1 18.5 0.0 Closed-source

GPT-4o [OpenAI, 2024] 53.9 32.8 48.4 35.2 36.3 35.4 39.3 22.1 GPT-5-mini [Singh et al., 2025] 64.7 7.8 49.9 76.1 15.2 62.2 53.7 18.0 GPT-5.4 [OpenAI, 2026] 58.2 19.5 48.1 69.5 22.6 59.2 70.9 22.6

- Gemini-2.5-Flash [Deepmind, 2025a] 56.1 45.0 53.2 66.4 2.4 52.2 18.5 6.7

- Gemini-3.0-Flash [Deepmind, 2025b] 61.7 44.1 57.1 64.0 6.3 51.3 50.3 2.4

and InternVL3-8B [Zhu et al., 2025]. Closed-source models include GPT-4o [OpenAI, 2024], GPT-5-mini [Singh et al., 2025], GPT-5.4 [OpenAI, 2026], Gemini-2.5-Flash [Deepmind, 2025a], and Gemini-3.0-Flash [Deepmind, 2025b]. All models are evaluated in a zero-shot setting using a standardized multiple-choice prompt (see Appendix A.2.1). For each question, we provide a single image selected from the oracle viewpoint: the camera position verified to have clear visibility of the target object. Models are presented with multiple-choice questions and required to select exactly one option, including Cannot determine where applicable. We report three metrics: Ans. (accuracy on answerable questions), Unans. (accuracy on unanswerable questions, i.e., correctly selecting Cannot determine), and All (micro-averaged accuracy over all questions). For the view selection task, we additionally report ViewSel accuracy. A random baseline is included for reference. More details about evaluation metrics are discussed in Appendix Sec. A.2.2.

- 4.2 Results on Observational Uncertainty

Table 1 presents model performance under occlusion and perspective ambiguity, and Fig. 5 provides a task-level breakdown. We summarize the following two failure modes.

Failure to abstain under unreliable observations. Across all models, performance on answerable questions consistently exceeds random, indicating that models can perform meaningful spatial reasoning when visual evidence is sufficient. However, their behavior diverges sharply on unanswerable cases, revealing three consistent patterns. (1) Answer–abstention trade-off. Models that perform better on identifying unanswerable cases tend to sacrifice accuracy on answerable questions. For example, Gemini-2.5-Flash achieves high Occ-Unans. (45.0) but relatively lower Occ-Ans. (56.1), while GPT-5-mini achieves the highest Perspective Ans. (76.1) but only 15.2 Unans. This suggests a fundamental trade-off between answering and abstention, rather than a unified notion of uncertainty awareness. (2) Inconsistent uncertainty behavior across models. There is no consistent pattern of abstention across model families. Gemini-2.5-Flash achieves Occ-Unans. 45.0 but collapses under perspective ambiguity (Unans. 2.4), while GPT-4o maintains more balanced performance across both conditions (32.8 vs. 36.3). GPT-5.4 achieves strong view selection (70.9) but only moderate Unans. performance (19.5 under occlusion). This inconsistency indicates that current VLMs do not learn a generalizable notion of when visual evidence is unreliable. (3) Sensitivity to observation uncertainty. Model performance degrades systematically as the reliability of visual observations decreases. Under occlusion, accuracy drops progressively from clean to partial and full occlusion, with the largest degradation when critical visual evidence is entirely missing. Notably, even partial occlusion leads to a consistent performance drop despite the target remaining visible, indicating that models are not robust to incomplete observations and rely heavily on near-complete visual evidence. Under perspective ambiguity, performance collapses on questions that depend on appearance-based cues (e.g., size and shape), where visual evidence becomes misleading, while tasks relying on viewpoint-invariant properties (e.g., visibility, relative position) remain largely stable. Together, these results show that models struggle when observations are incomplete or unreliable,

GPT-4o GPT-5.4 Gemini 2.5-Flash Gemini 3-Flash

Answerable (A)

Unanswerable (U)

| |
|---|

| |
|---|

Visibility

Relative Position

###### Depth Ordering

###### Size Comparison

100

A A A

A A U

|A A U| | | |
|---|---|---|---|
| | | | |
|| |
|---|
<br><br>| | | |
|| |
|---|
<br><br>| | | |
|Random (25%)| | | |
| | | | |
|| |
|---|
<br><br>| | | |
| | | | |

|A A U| | | |
|---|---|---|---|
| | | | |
|| |
|---|
<br><br>| | | |
|| |
|---|
<br><br>| | | |
|Random (25%)<br><br>| |
|---|
<br><br>| | | |
| | | | |
| | | | |
| | | | |

80

| |
|---|

Accuracy(%)

Occlusion

| |
|---|

60

Random (50%)

| |
|---|

| |
|---|

40

Random (20%)

20

0

Clean Partial Full

Clean Partial Full

Clean Partial Full

Clean Partial Full

Visibility

Relative Position

###### Size Comparison

Shape Comparison

100

A A

A A

A U

|A U| | |
|---|---|---|
| | | |
| | | |
| | | |
|Random (25%)<br><br>| | |
| | | |
| | | |
| | | |

80

Perspective

Accuracy(%)

60

40

Random (25%)

Random (25%)

Random (20%)

20

0

Clean Distorted

Clean Distorted

Clean Distorted

Clean Distorted

- Figure 5: Model accuracy across question types under occlusion (top) and perspective ambiguity (bottom). Blue and orange backgrounds indicate answerable (A) and unanswerable (U) conditions, respectively. Dashed lines show random baselines. Bold lines highlight the strongest closed-source models in each setting (Gemini-3.0-Flash for occlusion and GPT-5.4 for perspective).

suggesting that current VLMs fail to reason about the validity of visual evidence rather than the spatial relations themselves.

Failure to identify informative viewpoints. Results on the view selection tasks further reveal that current models struggle not only to recognize unreliable observations, but also to identify which additional viewpoints would provide reliable evidence. On ViewSel, which evaluates viewpoint selection in isolation, stronger models such as GPT-5.4 (70.9) and GPT-5-mini (53.7) achieve substantially above-random performance, indicating that models can often identify informative views when explicitly prompted to do so. However, performance drops sharply on AbstainViewSel, which additionally requires models to first recognize the current ambiguous view as uninformative before selecting a better viewpoint. For example, GPT-5.4 decreases from 70.9 to 22.6, GPT-5-mini from 53.7 to 18.0, and Gemini-3.0-Flash from 50.3 to 2.4. This large gap suggests that models face challenges at both stages: they struggle to recognize when the current observation is unreliable, and even when explicitly asked to select an informative viewpoint, their performance remains limited. Overall, these results suggest that informative viewpoint selection emerges only in stronger models, and even these models struggle to determine when their current observations are unreliable.

#### 4.3 Effect of Visual Input on Observational Uncertainty

We compare text-only (T) and vision-enabled (T+V) performance across answerable (Ans) and unanswerable (Unans) questions under occlusion and perspective ambiguity, as shown in Table 2. Adding visual input consistently improves answerable performance across all models, confirming that visual observations provide useful information when evidence is sufficient. Under occlusion, visual input also improves unanswerable performance for some models (e.g., GPT-5.4: Occ-Unans +6.4, Gemini-3.0-Flash: +29.8), suggesting that visual signals help detect missing evidence. Under perspective ambiguity, however, the effect reverses: both models show substantial drops in unanswerable performance when visual input is added (e.g., GPT-5.4: Pers-Unans -21.7, Gemini-3.0-Flash: -35.8), indicating that misleading visual cues actively suppress appropriate abstention. Overall, these results reveal a clear asymmetry: visual input is beneficial when information is missing, but can actively mislead models when observations are unreliable, highlighting that current models struggle to assess the reliability of visual evidence.

#### 4.4 Toward Improving Observational Uncertainty

Prompting helps but remains limited. To investigate whether abstention failures can be mitigated through prompting, we compare two strategies (see Appendix Sec. A.2.1

Table 2: Effect of visual input (T vs T+V).

Model Setting Occ-Ans Occ-Unans Pers-Ans Pers-Unans GPT-5.4 T 52.8 13.1 21.0 44.3

T+V 58.2 ↑5.4 19.5 ↑6.4 69.5 ↑48.5 22.6 ↓21.7 Gemini-3.0-Flash T 41.6 14.3 35.2 42.1

T+V 61.7 ↑20.1 44.1 ↑29.8 64.0 ↑28.8 6.3 ↓35.8

Table 3: Effect of fine-tuning strategies.

Model Occ-Ans Occ-Unans Pers-Ans Pers-Unans Base (Qwen2.5-VL-7B) 49.9 41.0 59.6 42.9 LoRA-Occ 67.2↑ 39.3↓ 55.3↓ 38.5↓ LoRA-Pers 54.4↑ 7.7↓ 84.8↑ 86.8↑

LoRA-Mixed 70.3↑ 62.8↑ 88.8↑ 76.9↑

for full details): a standard prompt that instructs the model to commit to a specific answer based on visible evidence, and a structured reasoning prompt that guides the model to first assess object visibility and viewpoint reliability before selecting an answer.

As shown in Table 4, structured prompting improves unanswerable performance for both models, but the effect is uneven. GPT-5-mini shows a substantial gain on Occ-Unans (7.8→30.4), while Gemini-2.5-Flash improves only slightly (45.0→48.7). However, this improvement comes at the cost of answerable accuracy: GPT-5-mini drops from 64.7 to 54.7, and Gemini-2.5-Flash from 56.1 to 50.4. Overall, structured prompting can encourage abstention, but does not reliably improve observational uncertainty: gains are model-dependent and introduce an answer-abstention trade-off that prompting alone cannot resolve.

Table 4: Effect of prompting on answerable and unanswerable cases under occlusion.

Model Prompt Occ-Ans Occ-Unans GPT-5-mini Standard 64.7 7.8

Structured 54.7 30.4 Gemini-2.5-Flash Standard 56.1 45.0

Structured 50.4 48.7

Can fine-tuning improve observational uncertainty? We further investigate whether fine-tuning can enable models to acquire a generalizable abstention capability under observation uncertainty. We fine-tune Qwen2.5-VL-7B-Instruct with LoRA [Hu et al., 2021] (rank 16, α 32) on our training split, holding out 10% of scenes for testing and 10% for validation. We train four variants: base (no adaptation), LoRA-Occ (trained on occlusion data only), LoRA-Pers (trained on perspective data only), and LoRA-Mixed (trained on both). As shown in Table 3, we observe two key findings. (1) Abstention is learnable but requires diversity. LoRA-Mixed substantially improves both answerable and unanswerable performance across occlusion and perspective conditions, demonstrating that models can acquire abstention behavior when trained on diverse forms of visual ambiguity. Importantly, this resolves the abstention–accuracy trade-off observed with prompting alone. (2) Single-condition training fails to generalize. Domain-specific fine-tuning does not transfer across ambiguity types. LoRA-Occ fails to meaningfully improve occlusion unanswerable performance (39.3 vs. base 41.0), while LoRA-Pers causes a dramatic drop in occlusion unanswerable performance (7.7), indicating negative transfer across ambiguity types. Together, these results suggest that generalizable abstention requires exposure to diverse forms of observation uncertainty during training.

### 5 Conclusion

We present SPATIALUNCERTAIN, a controlled diagnostic framework for evaluating observational awareness in VLMs under two projection-induced challenges: occlusion and perspective ambiguity. Our evaluation across eight VLMs reveals two consistent failure modes: models are systematically overconfident when visual evidence is incomplete or misleading, and perform near random chance when identifying informative viewpoints. We further show that prompting alone cannot resolve these failures, while fine-tuning on diverse ambiguity conditions substantially improves observational awareness. We hope SPATIALUNCERTAINmotivatesfuture work on VLMs that can assess the reliability of their own observations and actively seek additional evidence when needed.

### 6 Acknowledgement

We would like to thank Zengqi Zhao for his help in the human verification process. This work was supported by NSF-AI Engage Institute DRL-2112635, ARO Award W911NF2110220, and ONR Grant N00014-23-1-2356. The views contained in this article are those of the authors and not of the funding agency.

### References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. ArXiv, abs/2502.13923, 2025. URL https://api.semanticscholar.org/CorpusID:276449796.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024.

An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision language model. ArXiv, abs/2406.01584, 2024. URL https://api.semanticscholar.org/CorpusID:270215984.

Erik Daxberger, Nina Wenzel, David Griffiths, Haiming Gang, Justin Lazarow, Gefen Kohavi, Kai Kang, Marcin Eichner, Yinfei Yang, Afshin Dehghan, et al. Mm-spatial: Exploring 3d spatial understanding in multimodal llms. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7395–7408, 2025.

- Deepmind. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. https://arxiv.org/abs/2507.06261, 2025a.
- Deepmind. Gemini 3 flash: Frontier intelligence built for speed. https://blog.google/produc

##### ts/gemini/gemini-3-flash/, 2025b.

Jiafei Duan, Samson Yu, Hui Li Tan, Hongyuan Zhu, and Cheston Tan. A survey of embodied ai: From simulators to research tasks. IEEE Transactions on Emerging Topics in Computational Intelligence, 6(2):230–244, 2022.

Julian Eisenschlos, Hernán Maina, Guido Ivetta, and Luciana Benotti. Selectively answering visual questions. In Findings of the Association for Computational Linguistics: ACL 2024, pages 4219–4229, 2024.

Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. Advances in neural information processing systems, 30, 2017.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14375–14385, 2024.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR, 2017.

Yangyang Guo, Fangkai Jiao, Zhiqi Shen, Liqiang Nie, and Mohan Kankanhalli. Unk-vqa: A dataset and a probe into the abstention ability of multi-modal large models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):10284–10296, 2024.

Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617, 2018.

Xingwei He, Qianru Zhang, A Jin, Yuan Yuan, Siu-Ming Yiu, et al. Tubench: Benchmarking large vision-language models on trustworthiness with unanswerable questions. arXiv preprint arXiv:2410.04107, 2024.

Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. arXiv preprint arXiv:1610.02136, 2016.

J. Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. ArXiv, abs/2106.09685, 2021. URL https://api.semanticscholar.org/CorpusID:235458009.

Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135, 2025.

Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Amita Kamath, Jack Hessel, and Kai-Wei Chang. What’s “up” with vision-language models? investigating their struggle with spatial reasoning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9161–9175, 2023.

Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, et al. Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474, 2017.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 292–305, 2023.

Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: long papers), pages 3214–3252, 2022.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Jingping Liu, Ziyan Liu, Zhedong Cen, Yan Zhou, Yinan Zou, Weiyan Zhang, Haiyun Jiang, and Tong Ruan. Can multimodal large language models understand spatial relations? In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 620–632, 2025.

Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. Sqa3d: Situated question answering in 3d scenes. arXiv preprint arXiv:2210.07474, 2022.

Potsawee Manakul, Adian Liusie, and Mark Gales. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 9004–9017, 2023.

OpenAI. Hello GPT-4o, 2024. URL https://openai.com/index/hello-gpt-4o. OpenAI. Openai: Gpt-5.4 model, 2026. URL https://developers.openai.com/api/docs/m

odels/gpt-5.4.

Atin Pothiraj, Elias Stengel-Eskin, Jaemin Cho, and Mohit Bansal. Capture: Evaluating spatial reasoning in vision language models via occluded object counting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8001–8010, 2025.

Navid Rajabi and Jana Kosecka. Gsr-bench: A benchmark for grounded spatial reasoning evaluation via multimodal llms. ArXiv, abs/2406.13246, 2024. URL https://api.semanticscholar.or g/CorpusID:270619607.

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4035–4045, 2018.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Elias Stengel-Eskin, Peter Hase, and Mohit Bansal. Lacie: Listener-aware finetuning for calibration in large language models. Advances in Neural Information Processing Systems, 37:43080–43106, 2024.

Ilias Stogiannidis, Steven McDonagh, and Sotirios A Tsaftaris. Mind the gap: Benchmarking spatial reasoning in vision-language models. arXiv preprint arXiv:2503.19707, 2025.

Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liangyan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. In Findings of the Association for Computational Linguistics: ACL 2024, pages 13088–13110, 2024.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5433–5442, 2023.

Jiayu Wang, Yifei Ming, Zhenmei Shi, Vibhav Vineet, Xin Wang, Yixuan Li, and Neel Joshi. Is a picture worth a thousand words? delving into spatial reasoning for vision language models. Advances in Neural Information Processing Systems, 37:75392–75421, 2024a.

Tai Wang, Xiaohan Mao, Chenming Zhu, Runsen Xu, Ruiyuan Lyu, Peisen Li, Xiao Chen, Wenwei Zhang, Kai Chen, Tianfan Xue, et al. Embodiedscan: A holistic multi-modal 3d perception suite towards embodied ai. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19757–19767, 2024b.

Bingbing Wen, Jihan Yao, Shangbin Feng, Chenjun Xu, Yulia Tsvetkov, Bill Howe, and Lucy Lu Wang. Know your limits: A survey of abstention in large language models. Transactions of the Association for Computational Linguistics, 13:529–556, 2025.

Spencer Whitehead, Suzanne Petryk, Vedaad Shakib, Joseph Gonzalez, Trevor Darrell, Anna Rohrbach, and Marcus Rohrbach. Reliable visual question answering: Abstain rather than answer incorrectly. In European Conference on Computer Vision, pages 148–166. Springer, 2022.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, YIFEI LI, Jie Fu, Junxian He, and Bryan Hooi. Can LLMs express their uncertainty? an empirical evaluation of confidence elicitation in LLMs. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=gjeQKFxFpZ.

Peiran Xu, Sudong Wang, Yao Zhu, Jianing Li, Gege Qi, and Yunjian Zhang. Spatialbench: Benchmarking multimodal large language models for spatial cognition. arXiv preprint arXiv:2511.21471, 2025.

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025a.

Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis L Brown II, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, et al. Cambrian-s: Towards spatial supersensing in video. In The Fourteenth International Conference on Learning Representations, 2025b.

Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025c.

Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, et al. Holodeck: Language guided generation of 3d embodied ai environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16227–16237, 2024.

Zhangyue Yin, Qiushi Sun, Qipeng Guo, Jiawen Wu, Xipeng Qiu, and Xuan-Jing Huang. Do large language models know what they don’t know? In Findings of the association for Computational Linguistics: ACL 2023, pages 8653–8665, 2023.

Shoubin Yu, Yue Zhang, Zun Wang, Jaehong Yoon, Huaxiu Yao, Mingyu Ding, and Mohit Bansal. When and how much to imagine: Adaptive test-time scaling with world models for visual spatial reasoning. ArXiv, abs/2602.08236, 2026a. URL https://api.semanticscholar.org/Corp usID:285452504.

Shoubin Yu, Yue Zhang, Zun Wang, Jaehong Yoon, Huaxiu Yao, Mingyu Ding, and Mohit Bansal. When and how much to imagine: Adaptive test-time scaling with world models for visual spatial reasoning. arXiv preprint arXiv:2602.08236, 2026b.

Yue Zhang and Parisa Kordjamshidi. Vln-trans: Translator for the vision and language navigation agent. In Annual Meeting of the Association for Computational Linguistics, 2023. URL https: //api.semanticscholar.org/CorpusID:257038436.

Yue Zhang, Ziqiao Ma, Jialu Li, Yanyuan Qiao, Zun Wang, Joyce Chai, Qi Wu, Mohit Bansal, and Parisa Kordjamshidi. Vision-and-language navigation today and tomorrow: A survey in the era of foundation models. arXiv preprint arXiv:2407.07035, 2024a.

Yue Zhang, Zhiyang Xu, Ying Shen, Parisa Kordjamshidi, and Lifu Huang. Spartun3d: Situated spatial understanding of 3d world in large language models. arXiv preprint arXiv:2410.03878, 2024b.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Yue Cao, Yangzhou Liu, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Han Lv, De-Hua Chen, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Ying Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Lijun Wu, Kai Zhang, Hui Deng, Jiaye Ge, Kaiming Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: Exploring advanced training and testtime recipes for open-source multimodal models. ArXiv, abs/2504.10479, 2025. URL https: //api.semanticscholar.org/CorpusID:277780955.

### A Appendix

#### A.1 SPATIALUNCERTAIN Construction Details

Target occluder in occlusion configuration We score each object based on its visibility from the camera viewpoint, combining two factors: angular centrality (how close the object is to the camera’s optical axis) and apparent size (the object’s projected size relative to its depth). Objects that are partially occluded by other scene objects receive a penalty. We retain the top-k (k=3) objects per scene as target candidates, further requiring scene-level uniqueness.

Object pair selection in perspective configuration. To induce perspective ambiguity, we identify pairs of objects that are physically comparable but visually sensitive to viewpoint changes. Specifically, we select pairs of same-category objects with similar physical size, ensuring that they are expected to appear comparable under neutral viewpoints but can exhibit large appearance differences under perspective distortion. We consider two types of object pairs. Floor pairs consist of two floorstanding objects of the same type (e.g., chairs), whose centers are below 1.2m and are separated by at least 2.5m to allow for significant depth variation. Wall pairs consist of two wall-mounted objects (e.g., paintings) placed on the same or adjacent walls, separated by at least 1.2m, and matched in aspect ratio within a 20% tolerance to ensure similar physical proportions. For each scene, we sample up to 3 floor pairs and 2 wall pairs, promoting diversity while maintaining controlled geometric conditions.

- A.1.1 Human Annotation

The annotation interface is shown in Fig. 6. A total of 7 annotators participated in the validation process, each independently reviewing assigned configurations.

Occlusion Annotation. Annotators are presented with paired clean and occluded views side by side, with target and occluder objects labeled by name. Each configuration is classified into one of three categories: no occlusion (the occluder does not meaningfully block the target), partial occlusion (the target is partially visible), or full occlusion (the target is entirely hidden). Configurations classified as no occlusion are discarded. Approximately one-third of the generated occlusion configurations were discarded after annotation, reflecting the difficulty of achieving meaningful occlusion under geometric and physical constraints.

Perspective Annotation. For perspective scenes, annotators verify two conditions: (1) the reference view provides sufficient visual evidence to answer the spatial questions, and (2) the perspective view introduces visible geometric ambiguity that makes the questions unanswerable from that viewpoint. Configurations that fail either check are discarded. Similarly, approximately one-third of generated perspective configurations were discarded, primarily due to insufficient visual ambiguity in the perspective view or inadequate evidence in the reference view.

A.2 Evaluation Setup

- A.2.1 Prompt Templates

We use two prompt variants in our experiments: a standard multiple-choice prompt and a structured reasoning prompt.

Standard Prompt. The standard prompt instructs the model to select the best answer based on visible evidence, without any explicit guidance on assessing observation reliability. It permits abstention via “Cannot determine” but does not actively encourage it. This prompt serves as our primary evaluation setting.

Structured Reasoning Prompt. The structured reasoning prompt explicitly guides the model to assess observation reliability before selecting an answer. It decomposes the reasoning process into two explicit checks: whether the target is visible, and whether the viewpoint is reliable. Only if both checks pass does the model proceed to select a specific answer; otherwise, it defaults to “Cannot

determine.” This prompt is used in our prompting analysis to investigate whether explicit reasoning guidance can improve observational awareness.

You are answering a visual multiple-choice question about the provided image(s). Look carefully at the image and choose the best answer based on visible evidence. Rules:

- • You must choose exactly one option.
- • Choose “Cannot determine” if the image lacks sufficient visual evidence to decide reliably.
- • Reply with ONLY one letter.

Question: {question}

Options: {options} Reply with ONLY one letter from: A, B, C, ...

#### Structured Reasoning Prompt.

You are answering a visual multiple-choice question about the provided image(s). Follow these steps before selecting your answer:

- Step 1: Is the target object fully visible? Answer Yes or No.
- Step 2: Is the viewpoint reliable for answering the question? Answer Yes or No.
- Step 3: If both answers are Yes, select the correct option. Otherwise, select {cannot_letter}) Cannot determine. Question: {question} Options: {options} Write your step-by-step reasoning, then on a new line write: Answer: <letter from A, B, C, ...>

[Figure 92]_images/imageFile92.png>)

[Figure 93]_images/imageFile93.png>)

(a) Occlusion annotation examples. (b) Perspective ambiguity examples.

Figure 6: Annotation Interface for Occlusion and Perspective Scenes.

#### A.2.2 Evaluation Metrics

Models are presented with multiple-choice questions and required to select exactly one option, including Cannot determine where applicable. We report the following metrics: Answerable Accuracy

(Ans.) is calculated by Ans. = ##correcttotal (ans)(ans). Unanswerable Accuracy (Unans.) is calculated by Unans. = ##correcttotal (unans)(unans). Overall Accuracy (All) is calculated by All = ##correcttotal (ans)(ans)+#+# correcttotal (unans)(unans), ViewSel is calculated by ViewSel = ##totalcorrectlyview selectionselectedquestionsviews , and AbstainViewSel is calculated by AbstainViewSel = ##correcttotal unanswerableabstain-and-selectquestionscases.

#### A.2.3 Implementation Details

We fine-tune Qwen2.5-VL-7B-Instruct with LoRA adapters on the occlusion and perspective training splits separately to study cross-condition abstention transfer. LoRA is applied to all linear projections

in the language tower (r=16, α=32, dropout 0.05), while the vision encoder remains frozen. Training uses bf16, gradient checkpointing, cosine learning-rate scheduling, and a warm-up ratio of 0.03. Loss is computed only on assistant response tokens. The occlusion adapter is trained on 5.2K samples for 1 epoch with learning rate 3e−5, batch size 4, and gradient accumulation 2. The perspective adapter is trained on 3.0K samples for 2 epochs with learning rate 1e−4, batch size 2, and gradient accumulation 8. Training is conducted on 2×A100 80GB GPUs per adapter. At evaluation, each adapter is tested on held-out scenes from both benchmarks to measure in-domain and cross-domain abstention transfer.

#### A.3 Limitation

Our framework relies on controlled synthetic 3D environments, which enable systematic manipulation of observational conditions but may not fully capture the complexity and diversity of real-world scenes. In addition, our work focuses on observational uncertainty arising from occlusion and ambiguous viewpoints. While these settings isolate important challenges in spatial reasoning, real embodied environments may involve more complex and dynamic sources of uncertainty, such as motion, temporal changes, or sensor noise. Furthermore, our evaluation focuses on single-step spatial reasoning and viewpoint assessment, rather than full interactive exploration. Extending observational awareness to long-horizon embodied decision making remains an important direction for future work.

#### A.4 Licenses and External Assets

We use AI2-THOR (Apache 2.0) for simulation, open-source models (e.g., Qwen2.5-VL, InternVL3) under their respective licenses, and proprietary models (e.g., GPT and Gemini) via official APIs in accordance with their terms of service.

#### A.5 Broader Impact

This work studies observational uncertainty in vision-language models and highlights their tendency to produce confident spatial reasoning under incomplete or misleading observations. Improving awareness of unreliable visual evidence may benefit reliability-critical applications such as embodied agents and robotic systems. At the same time, our findings suggest that current models can make overconfident decisions under ambiguous viewpoints, which may lead to unsafe behaviors if deployed without appropriate safeguards. We hope this work encourages future research on uncertainty-aware and more reliable multimodal reasoning systems.

