[Figure 1]

## Seeing from Another Perspective: Evaluating Multi-View Understanding in MLLMs

Chun-Hsiao Yeh1* Chenyu Wang2* Shengbang Tong3 Ta-Ying Cheng4 Ruoyu Wang2 Tianzhe Chu6 Yuexiang Zhai1 Yubei Chen5 Shenghua Gao2,6 Yi Ma1,2,6

# arXiv:2504.15280v2[cs.CV]27Apr2025

1UC Berkeley 2TranscEngram 3NYU 4University of Oxford 5UC Davis 6HKU

View 3

Camera Pose Estimation

[Figure 2]

###### Can MLLMs identify the same attribute across views?

[Figure 3]

[Figure 4]

Counting

[Figure 5]

[Figure 6]

View4

[Figure 7]

Who stands closest to the spot where camera View 2 was positioned, and can we find the same person in camera View 1?

100 80

[Figure 8]

60 40

###### Can MLLMs track the object moving from another view?

[Figure 9]

[Figure 10]

20

[Figure 11]

Attribute Identification

Object Manipulation

[Figure 12]

In View 2, what if the man in a red shirt starts walking toward the left side of camera view, can you tell me his trajectory in View 1?

[Figure 13]

###### Can MLLMs estimate coarse camera poses?

[Figure 14]

[Figure 15]

Relative Direction

Relative Distance

[Figure 16]

If I randomly shuffle View 1 to View 4, can you sort them in clockwise order from topdown view?

View2

View1

Human Level Gemini-2.0-Flash Claude-3.7-Sonnet

GPT-4o Qwen2.5-VL-72B InternVL2.5-38B

Figure 1. We present All-Angles Bench, a rich-annotated benchmark with over 2,100 Q&A pairs from 90 diverse scenes for evaluating multi-view understanding of MLLMs. Left and Middle: An example question setup of multiple views capturing the same scene and the corresponding questions. Right: Accuracies of six notable MLLMs across different question categories.

specifically test model’s geometric correspondence and the capacity to align information consistently across views. Our extensive experiments, benchmark on 27 representative MLLMs including Gemini-2.0-Flash, Claude-3.7-Sonnet, and GPT-4o against human evaluators reveals a substantial performance gap, indicating that current MLLMs remain far from human-level proficiency. Through in-depth analysis, we show that MLLMs are particularly underperforming under two aspects: (1) cross-view correspondence for partially occluded views and (2) establishing the coarse camera poses. These findings highlight the necessity of domain-specific refinements or modules that embed stronger multi-view awareness. We believe that our AllAngles Bench offers valuable insights and contribute to bridging the gap between MLLMs and human-level multiview understanding. The project and benchmark are publicly available at https://danielchyeh.github. io/All-Angles-Bench/.

### Abstract

Multi-view understanding, the ability to reconcile visual information across diverse viewpoints for effective navigation, manipulation, and 3D scene comprehension, is a fundamental challenge in Multi-Modal Large Language Models (MLLMs) to be used as embodied agents. While recent MLLMs have shown impressive advances in high-level reasoning and planning, they frequently fall short when confronted with multi-view geometric consistency and crossview correspondence. To comprehensively evaluate the challenges of MLLMs in multi-view scene reasoning, we propose All-Angles Bench, a benchmark of over 2,100 human carefully annotated multi-view question–answer pairs across 90 diverse real-world scenes. Our six tasks (counting, attribute identification, relative distance, relative direction, object manipulation, and camera pose estimation)

∗Equal Contribution

[Figure 17]

- Figure 2. Overview of All-Angles Bench. Our benchmark targets a comprehensive view of multi-view understanding, spanning six primary question types. These question types are designed to investigate several major aspects of 3D scene understanding, from creating correspondence between objects to associating relative object and camera poses.

### 1. Introduction

Multi-view understanding is a fundamental challenge in bridging machine and human-level understanding [13, 19, 58] because it underpins an agent’s ability to perceive the environment consistently from diverse viewpoints. By ensuring geometric coherence and cross-view consistency, agents can accurately reconstruct scene layouts and object relationships — capabilities critical for effective navigation, manipulation, and interaction in the real world [43, 44]. The recent advancement in Multimodal Large Language Models (MLLMs) demonstrates strong capabilities in high-level reasoning and task planning [1, 4, 9, 17, 21, 27], and thus the feasibility of directly using MLLMs as embodied agents is an intriguing research challenge [14, 20, 25, 31, 39, 60]. However, such capacities alone are insufficient for generalist embodied agents operating in the real world, where a comprehensive 3D scene understanding and robust multiview reasoning are pivotal [10, 22, 23]. Recent studies survey that MLLMs lacking multi-view scene understanding often commit agent manipulation and navigation errors such as misjudge the target distance, skip partially occluded obstacles — stemming from limited awareness of multi-view geometry and object relationships [57, 65]. Since these models must navigate, manipulate, and make decisions in real world environments, it is vital to evaluate (and ultimately strengthen) their multi-view understanding capabilities. Yet, this aspect remains underexplored in details.

To this end, we raise two questions: (1) Do MLLMs possess the ability to understand multiple viewpoints simultaneously? and (2) What are the key challenges in MLLMs to gain better multi-view understanding?

To address these questions and in light of the lack of benchmarks to evaluate multi-view reasoning, we introduce AllAngles Bench, comprising over 2,100 carefully humanannotated question-answer pairs across 90 diverse multiview scenes in real world [18, 24]. We define six tasks — counting, attribute identification, relative distance, relative direction, manipulation, and camera pose estimation — with a focus on evaluating MLLM’s geometric understanding and its ability to align information consistently across multi-view scenes. To better evaluate whether models truly possess multi-view capabilities, we also propose a paired question scheme by creating a second question with the same content but with slightly changed wording/order of views. We benchmark 27 representative MLLMs (including Gemini-2.0 [45], Claude-3.7 [1], and GPT-4o [40]) against human evaluators. As revealed in Figure 1, a substantial performance gap persists between current MLLMs and human evaluators.

To better understand why MLLMs fall short of human-level multi-view reasoning, we conduct an in-depth analysis of commonly failed questions and tasks, and derive two key findings. First, MLLMs struggle to identify the same object across multiple views. We further test whether chain-ofthought prompting — a technique that has proven effective in other reasoning tasks [42, 55, 62] — could address this limitation. However, our experiments reveal that these linguistic strategies do not provide consistent improvements across models for multi-view reasoning. This suggests that more fundamental domain-specific refinements to multiview awareness modules or training data are necessary for MLLMs to fully internalize cross-view consistency. Second, MLLMs often fail to establish correspondence between

[Figure 18]

[Figure 19]

Multi-View Scenes

[Figure 20]

View 1

[Figure 21]

|[Figure 22]| | |View 2|
|---|---|---|---|
| | | |[Figure 23]|
| |[Figure 24]<br><br>[Figure 25]| | |

Human Annotation & Quality Check

Question Creation

###### Paired-Question Generation

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

the dark red button-up shirt and

Question: What if the man with the dark red button-up shirt and black pant in camera View 1 starts walking away from the spot where camera View 1 was positioned, what would be the trajectory of the man in camera View 3?

Question: What if the man with black pant in camera View 1 starts walking toward the view,

Question: What if the man with black pant in camera View 1 starts walking toward the view, what would be the trajectory of the man in camera View 3?

the spot where camera View 1 was positioned

what would be the trajectory of the man in camera View 3?

View 4 View 3

Paired Q&A

[Figure 30]

[Figure 31]

- (A) walking toward the left side of View 3
- (B) walking toward the right side of View 3
- (C) walking toward the window Answer: (B)

- 1. Counting
- 2. Attribute Identification
- 3. Relative Distance

- 4. Relative Direction
- 5. Object Manipulation
- 6. Camera Pose Estimation

- (A) walking toward the left side of View 3
- (B) walking toward the right side of View 3
- (C) walking toward the window

- (A) walking toward the left side of View 3
- (B) walking toward the right side of View 3
- (C) walking toward the spot where camera View 3 was positioned Answer: (A)

the spot where camera View 3 was positioned

Question Type Design

[Figure 32]

human annotated answer

- Figure 3. All-Angles Bench construction pipeline. (1) We collect and curate 90 diverse multi-view scenes and design six tasks that emphasize multi-view reasoning. (2) We generate initial questions via an MLLM, then refine and validate them through human annotation to ensure correctness, clarity, and domain relevance. (3) We create paired questions by systematically rephrasing or altering each view perspective while preserving their underlying visual correspondences to evaluate model’s cross-view consistency. A final quality-control step removes inconsistent or ambiguous pairs. Note that counting and camera pose estimation tasks utilize all available views per query, whereas other tasks employ two randomly selected viewpoints.

different viewpoints. We visualize how models infer scene layouts from multiple perspectives, revealing a consistent inability to accurately estimate camera poses, which in turn impedes performance on tasks like relative direction and object manipulation. We hope these insights will be helpful to future research towards bringing more better multi-view capabilities in MLLMs.

changes in object positions, orientations, or configurations across views; (6) Camera Pose Estimation: Evaluating the capacity to estimate viewpoint arrangements or scene layouts from multi-view inputs. Each task addresses a specific dimension of multi-view reasoning, ensuring a thorough assessment of MLLMs’ geometric understanding and their ability to align information across perspectives.

Our All-Angles Bench is derived from a curated selection of 90 diverse multi-view scenes sourced from Ego-Exo4D [18] and EgoHumans [24], totaling 2,132 question–answer pairs. Each question is structured as a multiple-choice questionnaire with three options, only one of which is correct. By collecting multi-view data from varying directions and viewpoints, we generate paired question-answers that highlight differences in appearance, occlusion, and spatial relationships across perspectives while preserving the underlying visual correspondences. The benchmark scenes encompass a wide range of activities (e.g., basketball, soccer, cooking, music playing) and environments (e.g., offices, gym, repair store, kitchen, playground) to ensure broad coverage of real-world scenarios where cross-view reasoning is essential. As shown in Figure 2, each question targets one of the six task categories outlined above to provide a challenging yet realistic platform for evaluating MLLMs’ geometric understanding and multi-view correspondence.

### 2. All-Angles Bench

The ability to integrate observations of the scene layout from multiple viewpoints is critical for the geometric understanding of MLLMs, which can significantly help with capturing and anticipating interaction outcomes of real-world complex environments safely.

#### 2.1. Overview of All-Angles Bench

Most existing benchmarks to evaluate MLLMs primarily rely on single-view or egocentric data, leaving the multiview consistency and correspondence capabilities of current MLLMs largely unexamined. To address this gap, we introduce All-Angles Bench, which comprehensively evaluates MLLMs’ performance across six task categories in multi-view scenarios: (1) Counting: Enumerating objects across viewpoints without double-counting or overlooking occluded elements; (2) Attribute Identification: Recognizing key properties (e.g., pose, color, shape, orientation) consistently across different viewing perspectives; (3) Relative Distance: Estimating object distances when presented with multiple views; (4) Relative Direction: Testing the understanding of directional relationships between objects across different views; (5) Object Manipulation: Inferring

#### 2.2. Benchmark Collection Process

We build a benchmark collection pipeline to effectively generate high quality question-answer pairs for multi-view understanding, as shown in Figure 3. To ensure the benchmark quality, all questions were manually annotated by human

annotators after collecting and clipping the raw questions.

Data Collection & Question Type Design. We begin by manually selecting 90 diverse multi-view scenes from EgoExo4D [18] and EgoHumans [24], covering a broad spectrum of activities and environments (e.g., indoor settings, residential areas, industrial spaces) to ensure varied visual contexts. Since the focus of this benchmark is on multiview analysis, we ensure each scene includes footage captured from at least three viewpoints. We then manually design six task categories spanning fundamental aspects of multi-view understanding: from enumerating and identifying objects across multiple viewpoints (counting, attribute identification), to capturing spatial relationships (relative distance, relative direction), and analyzing how objects change across views or camera perspectives (object manipulation, camera pose estimation). Please see Appendix for further details on the specific question design.

Question Creation & Human Annotation. After collecting our multi-view scenes and designing question templates for each task category, we leverage an MLLM [40] to generate initial questions grounded in the multi-view visual data. Specifically, we generate three questions per category for each multi-view scene except generating one question for camera pose estimation. We utilize all available views per query for counting and camera pose estimation tasks, whereas other tasks employ two randomly selected viewpoints. We hire eight human annotators who carefully examine each question along with the associated multiview images, removing invalid entries and refining question phrasing in Figure 3 (middle). This meticulous manual process also involves revising incorrect answer choices and finally annotating the single correct answer. For example, in attribute identification, the MLLM might inconsistently describe an object across two different camera views. In relative direction, it might offer contradictory options — e.g., “facing the right side of the camera view” vs. “facing the stove” — that actually reference the same orientation. Detailed instructions and guidelines for human annotator can be found in the Appendix.

Paired-Question Generation & Human Quality Check. To rigorously evaluate whether MLLMs truly grasp multiview concepts, we generate paired questions by systematically rephrasing (i.e., orientation) or altering the original queries (e.g., views) while preserving their underlying visual correspondences and the question structure. For instance, an attribute identification question such as “Is there a man wearing a yellow hoodie in View 1? Identify him in View 2.” can be paired with “There is a man wearing a yellow hoodie in View 2? Identify him in View 1,” ensuring both questions reference the same individual despite different viewpoint. Likewise, for relative direction, we swap orientations (e.g., left vs. right) and reference views (View 1

8.3%Cam. Pose

Primary Q&A Paired Q&A

Rel. Dist. 23.2%

Counting 11.8%

300

250

200

###### All-Angles Bench

Rel.Dir. 16.5%

150

100

22.3%Manipul.

50

Attribute18.0%

0

Attribute Cam. PoseManipul. Rel. Dir. Rel. Dist.

Figure 4. Statistical overview of All-Angles Bench. The pie chart shows the distribution of 6 sub-tasks of multi-view understanding. The bar plot illustrates the percentage breakdown by primary and paired question-answers of each sub-task.

vs. View 2). This process is similar to language manipulation in [56, 66] but requires careful verification of view-toview consistency. A final human quality check ensures geometric alignment between the paired questions, resulting in 85.3% of questions having paired counterparts (counting task is not involved) — thereby testing whether MLLMs genuinely understand multi-view scenarios or merely guess answers. The statistics of benchmark is shown in Figure 4.

### 3. MLLMs Have Multi-View Understanding?

#### 3.1. Evaluation Setup

Benchmark Models. We evaluate a broad spectrum of MLLMs spanning diverse model families, parameter scales, and training paradigms. On the closed-source side, we include three of the most prominent model families — Gemini-2.0 [45], Claude-3.7 [1], and GPT4o [40]. For open-source models, we examine recent breakthroughs from Deepseek-VL2 [52], Qwen2.5-VL [4], InternVL2.5 [8], Cambrian [46], LLaVA-OneVision [28], LLaVA-NeXT-Video [63], and OVIS [36]. In all experiments, we follow standard protocols and set the temperature to zero unless otherwise specified.

Human Evaluation. We randomly select a subset of 250 questions from our All-Angles Bench — encompassing all six task categories for evaluation by human annotators, each of whom independently answers every question. For fair comparison, we also report performance of Gemini-2.0Flash, Claude-3.7-Sonnet, GPT-4o, Qwen2.5-VL-72B, and InternVL2.5-38B on this subset.

#### 3.2. Results

As the primary results shown in Table 1, there remains a substantial performance gap between both of closed- and open-source MLLMs and human-level multi-view understanding. We post several findings we observe.

|Methods<br><br>|Avg.|Attribute Cam.Pose Counting Manipul. Rel.Dir. Rel.Dist.<br><br>Multiple-Choice Answer|
|---|---|---|
|Performance Against Human (250 Q&As) Human Level GPT-4o<br><br>Gemini-2.0-Flash Claude-3.7-Sonnet<br><br>InternVL2.5-38B Qwen2.5-VL-72B|82.0 52.4 58.4 52.8 60.8 58.4<br><br>|93.3 88.9 86.3 72.0 79.5 95.7 66.7 16.7 52.9 40.0 53.8 63.8 62.2 38.9 64.7 48.0 56.4 68.1 60.0 38.9 37.3 38.0 56.4 80.9 73.3 27.8 70.6 42.0 64.1 68.1 73.3 22.2 52.9 44.0 61.5 76.6|
|Closed-source Models<br><br>GPT-4o Gemini-1.5-Pro<br><br>Gemini-1.5-Flash<br><br>Gemini-2.0-Flash<br><br><br>Claude-3.5-Sonnet Claude-3.7-Sonnet<br><br>|47.8 47.4 46.6 52.3 48.2 50.0<br><br>|66.8 35.8 43.0 42.6 38.9 51.2 59.8 33.5 39.4 45.2 38.6 55.1<br><br>62.9 43.8 35.9 43.9 33.2 52.4 68.4 33.0 64.9 41.0 41.8 58.9<br><br>63.2 33.0 41.8 41.2 43.5 55.3 68.4 35.8 41.4 40.1 46.9 56.7<br><br><br>|
|Open-source Models DeepSeek-VL2-Small<br><br>DeepSeek-VL2 InternVL2.5-2B InternVL2.5-4B InternVL2.5-8B<br><br>InternVL2.5-38B InternVL2.5-78B Qwen2.5-VL-3B<br><br>Qwen2.5-VL-72B Ovis2-2B Ovis2-4B Ovis2-8B<br><br>Ovis2-16B Ovis2-34B<br><br>Cambrian-8B Cambrian-13B Cambrian-34B<br><br>LLaVA-Onevision-Qwen2-7B LLaVA-Onevision-Qwen2-72B<br><br>LLaVA-Video-Qwen2-7B LLaVA-Video-Qwen2-72B|45.5 47.8 41.0 45.8 49.9 55.6 52.5 45.2 55.7 46.2 46.6 49.1 53.2 55.3 39.2 36.5 41.9 45.9 52.5 42.8 53.1<br><br>|65.3 27.8 39.0 42.6 32.7 51.6<br><br>70.5 24.4 39.0 46.2 33.5 54.7 59.5 15.9 42.6 34.2 30.7 48.8<br><br>66.6 18.2 47.8 36.6 35.8 54.7<br><br><br>73.9 28.4 48.6 41.6 40.3 54.5 80.4 31.3 56.6 45.2 49.7 58.7 79.4 27.3 52.6 39.7 43.5 59.3<br><br>62.7 22.2 45.0 37.2 36.4 53.8<br><br>77.5 29.5 55.4 43.7 54.3 60.7 61.9 26.7 49.0 42.0 35.5 51.4 65.5 21.6 53.4 34.0 36.1 56.9 70.5 17.0 49.4 43.5 41.2 54.7 75.5 29.5 56.6 44.3 46.3 56.1 79.4 26.7 53.8 46.2 50.6 59.7 59.8 19.9 33.1 33.0 33.0 43.5 59.0 25.6 30.7 27.3 32.1 37.9<br><br>63.7 20.5 38.2 37.2 35.2 43.7<br><br>64.5 22.2 39.4 44.5 35.2 52.0<br><br><br>73.4 26.7 45.4 45.6 46.3 60.3 64.8 12.5 42.2 32.6 37.2 50.8 73.6 27.8 46.2 45.2 46.6 61.9<br><br>|

Table 1. Evaluation results for 27 MLLMs. We consolidate performance from both closed-source and open-source MLLM evaluations. We use deeper-gray to highlight the top result among all models in each sub-task, while light-gray marks the second-best result.

Finding 1: Simple task for human like coarse camera pose estimation poses challenges for MLLMs.

While humans approaching near-perfect accuracy on multiple tasks in our All-Angles Bench, both open- and closedsource models often struggle. For example, in camera pose estimation, human annotators achieve 88.9% accuracy when ordering multiple camera perspectives, whereas stateof-the-art MLLMs such as Gemini-2.0-Flash, Qwen2.5VL-72B, and InternVL2.5-38B trail behind over 50% margins. Many open-source MLLMs perform even worse than random guessing. Common errors include failures to reconcile viewpoint transitions and misinterpretations of geometric relationships, underscoring the persistent gap between human-level capabilities and current MLLM performance.

Finding 2: Certain open-source MLLMs surpass closed-source ones in orientation-sensitive tasks.

Interestingly, Ovis2-34B [36] and Qwen2.5-VL-72B [4] outperform leading closed-source models such as Gemini2.0 [45] and Claude-3.7-Sonnet [1] on object manipulation and relative direction. We observe that Qwen2.5-VL-72B integrates robust video understanding and fine-grained visual grounding modules (as highlighted in its model report), positioning it well to capture how objects re-orient across different viewpoints. The specialized, video-focused training regimes observed in these open-source models, which emphasize frame-by-frame orientation tracking and spatial grounding — crucial for handling multi-view scenes. While it is unclear whether closed-source models train with similar strategies, this findings can still be a good indicator that

domain-specific refinement can yield better performances in tasks tackling orientation and geometric reasoning.

#### 3.3. MLLM’s Robustness on Paired Questions

While the correctness of one single question indicates how often a model answers an isolated question correctly, it does not capture whether the model remains consistent when presented with semantically equivalent queries from different viewpoints or rephrasings. To investigate this, we also propose to look into the proportions of questions where the answers are inconsistent with one another.

First, we classify each paired instance into three scenarios: 1) CC (Both Correct) when the model answers both the primary and paired question correctly, 2) WW (Both Wrong) when it fails both versions, and 3) IC (Inconsistent) when the model answers one version correctly but fails the other. We are particularly interested in the case of IC, as this shows the number of questions where the model answered correctly but does not in fact reflect correct multi-view understanding, as simply changing the order or rephrasing the question leads to a wrong answer.

As shown in Figure 5, we report the proportions of IC (inconsistent) outcomes across six leading MLLMs — three open-source (Ovis2-34B, Qwen2.5-VL-72B, InternVL2.538B) and three closed-source (GPT-4o, Gemini-2.0-Flash, Claude-3.7-Sonnet). We have several observations: 1) GPT4o exhibits notably high inconsistency score IC (around 70%) on relative distance tasks, whereas the other five models generally have around 40% inconsistency in this category, 2) All models struggle with relative direction; all surpasses 40% inconsistency IC, highlighting the challenge of reasoning about orientation shifts in multi-view scenarios, 3) Gemini-2.0-Flash and Claude-3.7-Sonnet remain fairly balanced inconsistency across overall question types, while Ovis2-34B and GPT-4o vary significantly across tasks.

### 4. Why Do MLLMs Struggle with Multi-View Understanding?

To investigate specific weaknesses of MLLMs in multiview comprehension, we evaluate each question type in our All-Angles Bench. We select the top-performing closedsource and open-source MLLMs in our benchmark and systematically identify where these models succeed or fail in understanding multi-view scenarios.

#### 4.1. Failure of Multi-View Correspondence

We first investigate the multi-view counting task since we are curious about the discrepancy between egocentric view and multi-view counting. We begin our analysis by examining counting questions especially counting on how many people in total are in the scene. We find that MLLMs typically succeed in the complete-visibility in one view sce-

nario (i.e., when all individuals are visible within a single view), but frequently fail in the partial-visibility across multiple views scenario when partial information is distributed across multiple viewpoints (e.g., Person A and B in View 1, and Person C and D in View 2). As illustrated in Figure 6, GPT-4o occasionally handles these scenarios by simply counting the number of people per view and choosing the highest count, neglecting to reconcile individuals across different perspectives and thus leading to errors.

Can Reasoning Injection Improve MLLM’s Ability? To investigate whether linguistic reasoning can enhance MLLMs’ multi-view understanding, we randomly select 55 scenes from our 90-scene All-Angles Bench, excluding those with only a single person or with insufficient partialvisibility. In each chosen scene, all individuals are visible in at least one camera view (see Figure 6, left). We then create a paired version of these scenes by manually cropping footage so that key information is split across multiple viewpoints (e.g., Person A and B in View 1, and Person C and D in View 2). This setup enables a fair comparison of MLLMs’ performance under the same set of completevisibility versus partial-visibility conditions.

Prompting techniques have shown promise in enhancing the reasoning and problem-solving capabilities of large models across diverse tasks. Motivated by these findings, we explore whether such linguistic prompts can also bolster the visual-spatial proficiency of MLLMs in multi-view settings. Specifically, we introduce an Identification CoT strategy, which instructs the model to (1) provide a detailed description of each visible individual — noting appearance, clothing, orientation, and interactions with nearby people or objects, (2) cross-reference these descriptions across all views to avoid double-counting, and (3) provide a final tally of unique entities. The detailed prompt of Identification CoT could be found in Appendix. We also report two additional CoT strategies, Zero-Shot CoT and Self Consistency /w CoT which were used in [55], for comparison.

We evaluate three prompting strategies — Zero-Shot CoT, Self-Consistency, and Identification CoT — across three leading MLLMs: GPT-4o [40], Ovis2-34B [36], and InternVL2.5-38B [9] chosen for their varying levels of counting proficiency. As GPT-4o’s results shown in Figure 7, both Zero-Shot CoT and Self-Consistency yield relative gains of approximately 15% each over the noprompting baseline. Notably, Identification CoT provides a substantial improvement under partial-visibility conditions, suggesting that explicit entity descriptions and crossview consistency checks are pivotal for accurate reasoning when some individuals or objects are only partially visible across different views. However, when the model already possesses robust multi-view counting capabilities (e.g., InternVL2.5-38B), the benefits of additional prompt-

[Figure 33]

###### Inconsistent (𝑰𝑪) Both Correct (𝑪𝑪) + Both Wrong (𝑾𝑾) Averaged 𝑰𝑪

- Figure 5. Paired question-answers inconsistency across 6 MLLMs. We report the proportions of IC and CC + WW. Notably, GPT-4o struggles with relative distance (around 70% inconsistency). Gemini-2.0-Flash and Claude-3.7-Sonnet exhibit more balanced performance, whereas Ovis2-34B and GPT-4o vary considerably across tasks.

[Figure 34]

Complete-Visibility in One View Partial-Visibility across Multiple Views

Question: Here are multiple camera views which are pointing to the same scene, could you count how many people in total are in the scene?

Answer: 5 individuals in View 1, 3 individuals in View 2. I can count a total of 5 people in the scene.

Answer: I can see there are 3 people in View 1, 2 people in View 2. Therefore, the correct answer is 3.

[Figure 35]

A B

C

D E

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

| | |
|---|---|
| | |

A

C B

| |
|---|

A

| |
|---|

D C

[Figure 40]

| |
|---|

| |
|---|

D

| |
|---|

E

[Figure 41]

- View 1
- View 2

- View 1
- View 2

- Figure 6. Complete- and Partial-visibility counting. While MLLMs often succeed when everyone is visible in one viewpoint, they sometimes fail to reconcile fragmented information across views, as shown by GPT-4o occasionally picks the largest per-view count rather than reconciling individuals across views.

[Figure 42]

###### GPT-4o Ovis2-34B InternVL2.5-38B

Figure 7. Analysis of reasoning prompt strategies. We report the effectiveness of Zero-Shot CoT, Self-Consistency, and Identification CoT — across GPT-4o, Ovis2-34B, and InternVL2.5-38B under complete-view and partial-view settings. While CoT variations delivers notable gains in partial-visibility scenarios in GPT4o, its impact diminishes for models already be robust at multiview counting (e.g., InternVL2.5-38B). These results indicate that refining reasoning prompt alone is insufficient; specialized multiview training may be necessary to excel on All-Angles Bench.

ing diminish and can even degrade performance, as observed with InternVL2.5-38B. This phenomenon echoes findings in [55], where CoT methods offered limited advantages for strong spatial-reasoning models such as Gemini1.5 [45]. We hypothesize that, beyond these prompt reasoning strategies, architectures or training methods specialized for multi-view scenarios — incorporating domain-specific data or spatial-aware modules may be necessary to further advance MLLMs’ performance, rather than relying solely on enhanced prompt engineering.

#### 4.2. Failure with Coarse Camera Estimation

We also observe that MLLMs often struggle with orientation-sensitive challenges (as highlighted in Table 1), such as estimating camera poses, object directions, and tracking object trajectories across multiple viewpoints. To investigate how these shortcomings manifest, we design a

visualization prompt inspired by [55], wherein each object’s center is mapped to a 10 × 10 grid and camera view poses are depicted as dot with directional arrow. Specifically, we task GPT-4o [21] and Gemini-2.0-Flash [45] with inferring both object and camera’s positions and orientations from multi-view images (see Appendix for prompt details).

As illustrated in Figure 8 (object manipulation) and Figure 9 (camera pose estimation), many orientation-related errors stem from the model’s inability to reconcile viewpoint transformations. Instead of maintaining consistent object correspondences, the model frequently misaligns camera coordinates or overlooks background cues critical for geometric reasoning. This not only impacts camera pose estimation but also complicates downstream tasks such as relative direction or object manipulation where fine-grained rotational and positional cues are essential. These observations echo the findings from Section 4.1, suggesting that

GPT-4o Gemini-2.0-Flash Annotated Ground Truth

View 1 View 2

|[Figure 43]|[Figure 44]<br><br>Ca|[Figure 45]<br><br>mer|[Figure 46]<br><br>a V|[Figure 47]<br><br>iew|[Figure 48]<br><br>2|[Figure 49]|[Figure 50]|[Figure 51]|[Figure 52]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 53]|[Figure 54]|[Figure 55]<br><br>[Figure 56]|[Figure 57]|[Figure 58]|[Figure 59]|[Figure 60]|[Figure 61]|[Figure 62]|[Figure 63]|
|[Figure 64]|[Figure 65]|[Figure 66]|[Figure 67]<br><br>*|[Figure 68]<br><br>Hum|[Figure 69]<br><br>an|[Figure 70]<br><br>1|[Figure 71]|[Figure 72]|[Figure 73]|
|[Figure 74]|[Figure 75]|[Figure 76]|[Figure 77]|[Figure 78]|[Figure 79]|[Figure 80]<br><br>|[Figure 81]<br><br>Hu|[Figure 82]<br><br>ma|[Figure 83]<br><br>n 2|
|[Figure 84]<br><br>[Figure 85]|[Figure 86]<br><br>[Figure 87]|[Figure 88]<br><br>[Figure 89]|[Figure 90]<br><br>[Figure 91]|[Figure 92]<br><br>[Figure 93]|[Figure 94]<br><br>[Figure 95]|[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]|[Figure 99]<br><br>[Figure 100]|[Figure 101]<br><br>[Figure 102]|[Figure 103]<br><br>[Figure 104]|
|[Figure 105]|[Figure 106]|[Figure 107]|[Figure 108]<br><br>Le|[Figure 109]<br><br>go|[Figure 110]|[Figure 111]|[Figure 112]<br><br>Hu|[Figure 113]<br><br>ma|[Figure 114]<br><br>n 3|
|[Figure 115]|[Figure 116]|[Figure 117]<br><br>[Figure 118]|[Figure 119]|[Figure 120]|[Figure 121]|[Figure 122]|[Figure 123]|[Figure 124]<br><br>T|[Figure 125]<br><br>V|
|[Figure 126]|[Figure 127]|[Figure 128]<br><br>|[Figure 129]|[Figure 130]|[Figure 131]|[Figure 132]|[Figure 133]|[Figure 134]|[Figure 135]|
|[Figure 136]|[Figure 137]<br><br>Ca|[Figure 138]<br><br>mer|[Figure 139]<br><br>a V|[Figure 140]<br><br>iew|[Figure 141]<br><br>1|[Figure 142]|[Figure 143]|[Figure 144]|[Figure 145]|

|[Figure 146]|[Figure 147]|[Figure 148]|[Figure 149]|[Figure 150]|[Figure 151]|[Figure 152]|[Figure 153]|[Figure 154]|[Figure 155]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 156]|[Figure 157]|[Figure 158]|[Figure 159]|[Figure 160]|[Figure 161]|[Figure 162]|[Figure 163]|[Figure 164]|[Figure 165]|
|[Figure 166]|[Figure 167]|[Figure 168]|[Figure 169]|[Figure 170]|[Figure 171]|[Figure 172]|[Figure 173]|[Figure 174]|[Figure 175]|
|[Figure 176]|[Figure 177]<br><br>*H|[Figure 178]<br><br>um|[Figure 179]<br><br>an|[Figure 180]<br><br>1|[Figure 181]|[Figure 182]|[Figure 183]<br><br>Hu|[Figure 184]<br><br>ma|[Figure 185]<br><br>n 2|
|[Figure 186]|[Figure 187]<br><br>|[Figure 188]|[Figure 189]|[Figure 190]|[Figure 191]<br><br>Leg|[Figure 192]<br><br>o|[Figure 193]|[Figure 194]|[Figure 195]<br><br>TV|
|[Figure 196]<br><br>ra V|[Figure 197]<br><br>iew<br><br>[Figure 198]|[Figure 199]<br><br>2|[Figure 200]|[Figure 201]|[Figure 202]|[Figure 203]|[Figure 204]|[Figure 205]|[Figure 206]|
|[Figure 207]|[Figure 208]|[Figure 209]|[Figure 210]|[Figure 211]|[Figure 212]|[Figure 213]|[Figure 214]<br><br>Hu|[Figure 215]<br><br>ma|[Figure 216]<br><br>n 3|
|[Figure 217]|[Figure 218]|[Figure 219]|[Figure 220]|[Figure 221]<br><br>[Figure 222]|[Figure 223]<br><br>[Figure 224]|[Figure 225]|[Figure 226]|[Figure 227]|[Figure 228]|
|[Figure 229]|[Figure 230]|[Figure 231]|[Figure 232]|[Figure 233]|[Figure 234]<br><br>|[Figure 235]|[Figure 236]|[Figure 237]|[Figure 238]|
|[Figure 239]|[Figure 240]|[Figure 241]|[Figure 242]|[Figure 243]<br><br>Ca|[Figure 244]<br><br>me|[Figure 245]<br><br>ra V|[Figure 246]<br><br>iew|[Figure 247]<br><br>1|[Figure 248]|

|[Figure 249]|[Figure 250]|[Figure 251]|[Figure 252]|[Figure 253]|[Figure 254]|[Figure 255]|[Figure 256]|[Figure 257]|[Figure 258]|
|---|---|---|---|---|---|---|---|---|---|
|[Figure 259]|[Figure 260]|[Figure 261]|[Figure 262]|[Figure 263]|[Figure 264]|[Figure 265]|[Figure 266]<br><br>TV|[Figure 267]|[Figure 268]|
|[Figure 269]|[Figure 270]|[Figure 271]|[Figure 272]|[Figure 273]|[Figure 274]|[Figure 275]|[Figure 276]|[Figure 277]|[Figure 278]<br><br>|
|[Figure 279]|[Figure 280]|[Figure 281]|[Figure 282]|[Figure 283]|[Figure 284]|[Figure 285]|[Figure 286]|[Figure 287]<br><br>[Figure 288]|[Figure 289]|
|[Figure 290]|[Figure 291]|[Figure 292]<br><br>Hu|[Figure 293]<br><br>ma|[Figure 294]<br><br>n 2|[Figure 295]|[Figure 296]|[Figure 297]<br><br>Hu|[Figure 298]<br><br>ma|[Figure 299]<br><br>n 3|
|[Figure 300]|[Figure 301]|[Figure 302]|[Figure 303]|[Figure 304]|[Figure 305]<br><br>|[Figure 306]|[Figure 307]|[Figure 308]|[Figure 309]|
|[Figure 310]|[Figure 311]|[Figure 312]|[Figure 313]|[Figure 314]|[Figure 315]|[Figure 316]<br><br>Le|[Figure 317]<br><br>go|[Figure 318]|[Figure 319]|
|[Figure 320]|[Figure 321]|[Figure 322]|[Figure 323]|[Figure 324]|[Figure 325]<br><br>[Figure 326]<br><br>[Figure 327]<br><br>|[Figure 328]|[Figure 329]|[Figure 330]|[Figure 331]|
|[Figure 332]|[Figure 333]<br><br>*H|[Figure 334]<br><br>um|[Figure 335]<br><br>an 1|[Figure 336]|[Figure 337]<br><br>|[Figure 338]|[Figure 339]|[Figure 340]|[Figure 341]|
|[Figure 342]|[Figure 343]|[Figure 344]|[Figure 345]|[Figure 346]<br><br>Ca|[Figure 347]<br><br>me|[Figure 348]<br><br>ra V|[Figure 349]<br><br>iew|[Figure 350]<br><br>1|[Figure 351]|

[Figure 352]

[Figure 353]

Camera View 2

Came

[Figure 354]

[Figure 355]

Camera Orientation ( )

Object Trajectory ( )

[Figure 356]

###### Object Manipulation

GPT-4o: The visualization represents the spatial arrangement of key objects in the scene using a 10x10 grid, along with the camera positions and orientations.… In View 1, his trajectory in View 1 is from the left side of the image toward the bottom-right side where Camera 1 is placed. … The man in the dark red shirt was

What if the man with the dark red button-up shirt and black pant in camera View1 starts walking toward the spot where camera View 1 is positioned, what would be the trajectory of the man in camera View 2?

initially positioned near the left side of the frame in View 2. If he moves toward the bottom-right in View 1, then from the perspective of Camera 2, he would appear to be moving toward the left side of the frame in View 2. Thus, the final answer is (A)

Gemini-2.0-Flash: Here is the representation of the scene within a 10x10 grid. The question asks about the

- (A) walking toward the left side of View 2
- (B) walking toward the right side of View 2
- (C) walking toward the spot where camera View 2 is positioned

[Figure 357]

trajectory of the man with the red shirt if he walks towards the position of Camera View 1. In Camera View 1, the man is on the left side of the frame. Since Camera View 1 comes from the right, the trajectory of the man with the red shirt would be toward the right side of view in Camera View 2. Answer: (B) walking toward the

right side of view.

- Figure 8. Visualization of multi-view scene reconstruction and camera pose alignment. Although GPT-4o and Gemini-2.0-Flash both display moderate proficiency in single-view scene reconstruction, they struggle when aligning two different camera perspectives. Misidentifying camera poses leads to incorrect directional reasoning — such as tracking a person’s trajectory from View 1 to View 2 which needs multi-view consistency in current MLLMs.

View 1 View 2 Gemini-2.0-Flash: Now, let's analyze how these features appear in each image and deduce the camera positions: View-1 and View-2: These cameras are positioned relatively close to each other, with View-2 slightly to the left of View-1. * View-2 and View-3: These cameras are positioned on opposite sides of the man, with View3 further to the left. View-3 and

View-4: These cameras are positioned relatively close to each other, with View-4 slightly to the right of View-

3. View-4 and View-1: These cameras are positioned on opposite sides of the man, with View-1 further to the right. Here is my visual estimation of camera poses:

```

Tool Shelves

| | View-3 ---- Man ---- View-4 | |

View-2 ---- View-1

| |

Repair Stand

```

Clockwise Order of Views (top-down): View-1 → View-2 → View-3 → View-4

View 1, View 2, View 3, View 4 are four different input camera views that represent the same scene. Could you reconstruct the camera view layout from the top-bottom view, and answer the following question?

Please also provide the reasons.

Camera Pose Estimation

[Figure 358]

View 4 View 3

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

- Figure 9. Visualization of camera pose estimation. When asked to order the camera poses in clockwise order, MLLMs fail completely despite providing detailed reasonings.

on video understanding and, more broadly, exploring model capabilities for embodied real-world tasks. Our work contributes to this growing area by: 1) providing a timely evaluation benchmark for assessing current and future models’ abilities in multi-view perception — a fundamental capability for 3D and 4D tasks; and 2) offering an analysis of why current models struggle with multi-view understanding.

Benchmarking Visual Spatial Ability. There are recently more works [5, 16, 29, 30, 37, 55] studying video and visual spatial ability of MLLMs. Our work is most relevant to VideoMME [16], VSI-Bench [55] and MV-Bench [29]. VideoMME extensively evaluates video understanding but emphasizes temporal reasoning [16]. VSI-Bench specifically targets spatial intelligence through egocentric video scenarios [55]. MV-Bench also focuses on comprehensive multi-modal temporal understanding [29]. Our work focuses on multi-view understanding, a cornerstone for robust 3D and 4D reasoning in MLLMs. Unlike previous work that primarily assess single-view or temporal reasoning, we explicitly evaluate how models align geometric and semantic information across multiple viewpoints. We further provide a detailed breakdown analysis that dissects model deficiencies in multi-view understanding.

domain-specific training or architectural refinements, especially those emphasizing viewpoint consistency may necessary to close the gap in multi-view understanding.

### 5. Related Works

Multimodal Large Language Models. MLLMs [3, 12, 17, 21, 28, 32, 33, 46] have demonstrated impressive capabilities across various tasks [26, 34, 35, 38, 48, 59, 60] and applications [2, 11, 47, 49, 53, 61, 64]. In particular, an increasing number of studies [6, 7, 14, 19] are focusing

### 6. Conclusion

In conclusion, we introduce All-Angles Bench, a comprehensive benchmark to evaluate MLLMs’ multi-view understanding. Our evaluation of 27 representative models across over 2,100 annotated multi-view question-answer pairs in the six tasks, we reveal significant limitations in geomet-

ric consistency and cross-view correspondence, particularly in cross-view identification and camera pose estimation. These findings highlight the need for domain-specific training to enhance MLLMs’ multi-view reasoning, providing insights toward achieving human-level performance.

### References

- [1] Anthropic. Claude, 2024. 2, 4, 5
- [2] Hao Bai, Yifei Zhou, Jiayi Pan, Mert Cemri, Alane Suhr, Sergey Levine, and Aviral Kumar. Digirl: Training in-thewild device-control agents with autonomous reinforcement learning. Advances in Neural Information Processing Systems, 37:12461–12495, 2025. 8
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. 2023. 8
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 4, 5
- [5] Keshigeyan Chandrasegaran, Agrim Gupta, Lea M Hadzic, Taran Kota, Jimming He, Crist´obal Eyzaguirre, Zane Durante, Manling Li, Jiajun Wu, and Fei-Fei Li. Hourvideo: 1-hour video-language understanding. Advances in Neural Information Processing Systems, 37:53168–53197, 2025. 8
- [6] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465,

2024. 8

- [7] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. arXiv preprint arXiv:2408.10188, 2024. 8
- [8] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 4, 3
- [9] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 2, 6
- [10] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37:135062–135093, 2025. 2
- [11] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine,

- and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025. 8
- [12] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. In NeurIPS, 2024. 8
- [13] Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan Lee, Devi Parikh, and Dhruv Batra. Embodied question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1–10, 2018. 2
- [14] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, et al. Palm-e: An embodied multimodal language model. 2023. 2, 8
- [15] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198–11201, 2024. 1
- [16] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 8
- [17] Google. Gemini, 2023. 2, 8
- [18] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19383–19400, 2024. 2, 3, 4, 1
- [19] Yining Hong, Chunru Lin, Yilun Du, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. 3d concept learning and reasoning from multi-view images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9202–9212, 2023. 2, 8
- [20] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In International conference on machine learning, pages 9118–9147. PMLR, 2022. 2
- [21] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 2, 7, 8
- [22] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Alaa Maalouf, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, et al. Conceptfusion: Open-set multimodal 3d mapping. arXiv preprint arXiv:2302.07241, 2023. 2
- [23] Baoxiong Jia, Yixin Chen, Huangyue Yu, Yan Wang, Xuesong Niu, Tengyu Liu, Qing Li, and Siyuan Huang. Sceneverse: Scaling 3d vision-language learning for

- grounded scene understanding. In European Conference on Computer Vision, pages 289–310. Springer, 2024. 2
- [24] Rawal Khirodkar, Aayush Bansal, Lingni Ma, Richard Newcombe, Minh Vo, and Kris Kitani. Ego-humans: An egocentric 3d multi-human benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19807–19819, 2023. 2, 3, 4, 1
- [25] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 2
- [26] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024. 8
- [27] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2
- [28] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 4, 8
- [29] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195– 22206, 2024. 8
- [30] Manling Li, Shiyu Zhao, Qineng Wang, Kangrui Wang, Yu Zhou, Sanjana Srivastava, Cem Gokmen, Tony Lee, Erran Li Li, Ruohan Zhang, et al. Embodied agent interface: Benchmarking llms for embodied decision making. Advances in Neural Information Processing Systems, 37: 100428–100534, 2025. 8
- [31] Fangchen Liu, Kuan Fang, Pieter Abbeel, and Sergey Levine. Moka: Open-vocabulary robotic manipulation through mark-based visual prompting. In First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, 2024. 2
- [32] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 8
- [33] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR,

2024. 8

- [34] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In ECCV, 2024. 8
- [35] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In

ICLR, 2023. 8

- [36] Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Han-Jia Ye. Ovis: Structural embedding alignment for multimodal large language model. arXiv preprint arXiv:2405.20797, 2024. 4, 5, 6
- [37] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023. 8
- [38] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In ACL, 2022. 8
- [39] Dantong Niu, Yuvan Sharma, Giscard Biamby, Jerome Quenum, Yutong Bai, Baifeng Shi, Trevor Darrell, and Roei Herzig. Llarva: Vision-action instruction tuning enhances robot learning. arXiv preprint arXiv:2406.11815, 2024. 2
- [40] OpenAI. gpt4o, 2024. 2, 4, 6, 1, 3
- [41] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020. 2
- [42] William Rudman, Michal Golovanesky, Amir Bar, Vedant Palit, Yann LeCun, Carsten Eickhoff, and Ritambhara Singh. Forgotten polygons: Multimodal large language models are shape-blind. arXiv preprint arXiv:2502.15969, 2025. 2
- [43] Chan Hee Song, Jihyung Kil, Tai-Yu Pan, Brian M Sadler, Wei-Lun Chao, and Yu Su. One step at a time: Long-horizon vision-and-language navigation with milestones. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15482–15491, 2022. 2
- [44] Alessandro Suglia, Qiaozi Gao, Jesse Thomason, Govind Thattai, and Gaurav Sukhatme. Embodied bert: A transformer model for embodied, language-guided visual task completion. arXiv preprint arXiv:2108.04927, 2021. 2
- [45] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2, 4, 5, 7
- [46] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. In NeurIPS, 2024. 4, 8
- [47] Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024. 8
- [48] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In CVPR, 2024. 8
- [49] Chenyu Wang, Weixin Luo, Qianyu Chen, Haonan Mai, Jindi Guo, Sixun Dong, Zhengxin Li, Lin Ma, Shenghua Gao, et al. Mllm-tool: A multimodal large language model for tool agent learning. arXiv preprint arXiv:2401.10727,

2024. 8

- [50] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In ICLR, 2023. 3
- [51] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022. 2
- [52] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-ofexperts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024. 4
- [53] Jingwei Xu, Zibo Zhao, Chenyu Wang, Wen Liu, Yi Ma, and Shenghua Gao. Cad-mllm: Unifying multimodalityconditioned cad generation with mllm. arXiv preprint

- arXiv:2411.04954, 2024. 8

[54] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint

- arXiv:2412.15115, 2024. 1

- [55] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. arXiv preprint arXiv:2412.14171, 2024. 2, 6, 7, 8
- [56] Yue Yang, Shuibai Zhang, Wenqi Shao, Kaipeng Zhang, Yi Bin, Yu Wang, and Ping Luo. Dynamic multimodal evaluation with flexible complexity by vision-language bootstrapping. arXiv preprint arXiv:2410.08695, 2024. 4
- [57] Hanxun Yu, Wentong Li, Song Wang, Junbo Chen, and Jianke Zhu. Inst3d-lmm: Instance-aware 3d scene understanding with multi-modal instruction tuning, 2025. 2
- [58] Licheng Yu, Xinlei Chen, Georgia Gkioxari, Mohit Bansal, Tamara L Berg, and Dhruv Batra. Multi-target embodied question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6309–6318, 2019. 2
- [59] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024. 8
- [60] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 2, 8
- [61] Yuexiang Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Shengbang Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, et al. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. In NeurIPS, 2024. 8
- [62] Jianshu Zhang, Dongyu Yao, Renjie Pi, Paul Pu Liang, et al. Vlm 2-bench: A closer look at how well vlms implicitly link explicit matching visual cues. arXiv preprint arXiv:2502.12084, 2025. 2

- [63] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. 4
- [64] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 8
- [65] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. arXiv preprint arXiv:2409.18125, 2024. 2
- [66] Kaijie Zhu, Jindong Wang, Qinlin Zhao, Ruochen Xu, and Xing Xie. Dynamic evaluation of large language models by meta probing agents. arXiv preprint arXiv:2402.14865,

2024. 4

[Figure 363]

## Seeing from Another Perspective:

## Evaluating Multi-View Understanding in MLLMs Supplementary Material

In these supplementary materials, we provide the following:

- • Details on the construction and annotation pipeline of AllAngles Bench (Section 7);
- • Evaluation setup, implementation detail of CoT methods, evaluation results visualization and complete evaluation results for the tiny All-Angles Bench (Section 8);
- • Additional visualization results and prompts (Section 9).

### 7. Construction and Annotation Pipeline

#### 7.1. Dataset Collection

We manually selected 83 scenes from Ego-Exo4D [18] and 7 scenes from EgoHumans [24] to ensure the diversity of scenes. Given the high density of viewpoints in some EgoHumans scenes, we carefully curated a subset of more spatially dispersed views to avoid excessive redundancy. As a result, we retained 4 - 5 views per scene. All multi-view images were standardized to a resolution of 796 × 448 pixels.

#### 7.2. Question Creation

For each generated question, we recorded the following key attributes: question index, source dataset, task category, image list path, question text, and multiple-choice options. Since the questions would undergo a human-in-the-loop quality review and verification process, letter-based answer choices were not generated at this stage.

Among the six task categories in All-Angles Bench, five were generated using an MLLM [40]. For the Camera Pose Estimation task, however, we designed a dedicated question template to structure the question generation process. The system prompt, task-specific prompts for the five generated tasks, and the camera pose estimation question template are illustrated in Figures 25, 26, and 27, respectively.

#### 7.3. Human Annotation and Quality Check

In this study, we hired eight Ph.D. students in STEM fields specializing in natural language processing or computer vision, to manually annotate all questions. To ensure consistency, we provided a comprehensive annotation guideline, which was refined and structured into a streamlined version, as illustrated in Figure 10.

Additionally, to maintain high annotation quality, we implemented a multi-stage verification process. Before the formal annotation process, annotators were required to com-

plete a set of sample questions to familiarize themselves with the standards and guidelines. To minimize errors and ambiguities, each annotation was then cross-checked by at least one other annotator, with any disagreements resolved through group discussions. Figure 12 presents an example comparing an initial annotation with its final version after cross-checking. Such ambiguous or unclear instances were flagged for review and collaboratively examined in team meetings, ensuring a standardized and consistent annotation process. Furthermore, we adopted a random sampling review mechanism, periodically evaluating a subset of annotated data to ensure strict adherence to the guidelines. The finalized benchmark is stored in JSON format, with an example visualization provided in Figure 11.

To streamline the annotation and quality control process, we developed a GUI-based annotation platform, as shown in Figure 11. This platform provides an intuitive interface enabling annotators to inspect and edit annotations efficiently. Annotators can seamlessly browse multi-view input images, modify questions, and adjust answer options with ease, ensuring both accuracy and consistency in the annotations.

Notably, the entire dataset collection and processing required over 300 person-hours, reflecting our meticulous attention to detail in ensuring the benchmark’s high reliability and quality for the relevant research community.

### 8. Experiment Details

#### 8.1. Evaluation Setup

Our evaluation is conducted using the VLMEvalKit [15] framework. In order to ensure the reproducibility of our evaluation, we employ a greedy decoding strategy for all models, setting the temperature to 0 unless otherwise stated. The text input follows a standardized format: [Question][Options][Post-prompt], where the post-prompt instructs: “Answer with the option’s letter from the given choices directly.”

To ensure that all final predictions are formatted as singleletter outputs, facilitating subsequent evaluation against the ground-truth answers and minimizing errors due to fuzzy matching, we leverage the open-source LLM, Qwen2.532B [54] to extract the predicted options accurately. The corresponding prompt is shown in Figure 24.

For human-level performance evaluation on the tiny 250-

We first have to say thank for your efforts in annotating the All-Angles Bench benchmark. Your contribution is invaluable and plays a crucial role in advancing this research.

Your Task: Review, refine, and finalize multiple-choice questions (MCQs) generated by an MLLM. Ensure they are clear, accurate, and correctly answered. What to Check:

- 1. Clarity & Relevance – The question must match the image/context and be easy to understand.
- 2. Grammar & Precision – No errors or vague phrasing.
- 3. Answer Quality – The correct answer must be factually accurate, and distractors should be plausible yet clearly incorrect.

Example Before & After: Original Question:

“What if the man with black pants in camera View 1 starts walking toward the view, what would be the trajectory of the man in camera View 3?”

✗ Issue: “toward the view” is unclear. Revised Question:

“What if the man with black pants in camera View 1 starts walking toward the camera position of the view, what would be the trajectory of the man in camera View 3?” ✓ Fix: Provide a more specific description of the subject’s movement.

✓ Final Answer: (B) walking toward the right side of View 3. How to Annotate

- 1. Verify the questions and options for plausibility and accuracy based on the images/data provided.
- 2. Correct unclear wording, grammatical errors, or inaccuracies in the questions.
- 3. Finalize the correct answer. Common Mistakes to Avoid

- ✗ Vague phrasing (e.g., “toward the view”).
- ✗ Misleading or contradictory options.
- ✗ MLLM hallucinations (made-up facts or objects)

- Figure 10. The streamlined version of annotation guideline for annotators to follow. It outlines key verification steps, common pitfalls, and examples to help annotators improve question clarity, accuracy, and answer quality.

question benchmark, we invited two additional Ph.D. students in STEM fields who were not involved in the annotation process to answer the questions. Each evaluator was assigned 125 questions and given unlimited time to answer with their best effort. Their combined scores serve as the human performance baseline for this tiny benchmark.

To eliminate potential biases introduced by contextual cues, we exclude paired data from this subset, preventing evaluators from leveraging strong prior knowledge. Additionally, for questions involving only two views, we ensure consistency with the MLLM setup by displaying only the relevant input views rather than all available ones. To further prevent evaluators from unintentionally deriving answers from sequentially presented images, we randomly shuffle the question order, ensuring independent assessment of each query.

#### 8.2. Implementation Details of CoT Methods

Inspired by [55], we evaluate three distinct reasoning-based prompting strategies on our benchmark: Zero-Shot CoT, Self-Consistency, and our proposed Identification CoT. Below, we outline the implementation details.

Notably, after generating intermediate reasoning steps and predictions using the three CoT approaches, we apply a standardized post-processing step. Specifically, we leverage an additional open-source LLM to explicitly extract the final answer from the generated response, as described in Section 8.1.

• Zero-Shot CoT: Building on prior works [41, 51], we enhance step-by-step reasoning in the MLLM by appending the phrase, “Let’s think step by step”, to each question in

[Figure 364]

[Figure 365]

- Figure 11. Left: A structured JSON representation of a question-answer pair. Right: A snapshot of the GUI-based Annotation Platform used for reviewing and refining annotations. Best viewed zoomed in for details.

What if the fried pan on the kitchen island is moved to the edge near to the window in View 2, what would be the trajectory of the same fried pan in View 1?

- (A) moving toward the left side of View 1
- (B) moving toward the right side of View 1
- (C) moving toward the center in View 1

Object Manipulation

[Figure 366]

[Figure 367]

[Figure 368]

- View 1
- View 2

Before Cross Checking Version

Finalized Version

What if the fried pan on the kitchen island is moved to

near the uncovered window located at the top center-left

side in View 2, what would be the trajectory of the same fried pan in View 1?

- (A) moving toward the top left side of View 1
- (B) moving toward the top right side of View 1
- (C) moving toward the bottom left side of View 1

Object Manipulation

[Figure 369]

Issues:

- ✗ ‘the window in View 2’ is unclear
- ✗ Option A and B should explicitly include both directional and depth cues
- ✗ avoid potential ambiguity between options A and C

[Figure 370]

- Figure 12. Comparison of a flagged ambiguous case before modification and its finalized version after cross-checking. The initial annotation was reviewed by multiple annotators, with ambiguities resolved through discussions to ensure clarity and consistency.

adopts the same settings as Zero-Shot CoT, using temperature equals 0 and a single inference pass to generate the final prediction.

Figures 14 and 15 illustrate model outputs for two MLLMs — GPT-4o [40] and InternVL2.5-38B [8] — under the three prompting strategies.

#### 8.3. Evaluation Results Visualization

Figure 23 visualizes model performance across six task categories, where color intensity represents precision levels cooler colors indicate lower accuracy, while warmer colors denote higher accuracy.

#### 8.4. More Evaluation Results

Table 2 presents the evaluation results of 27 MLLMs, encompassing both closed-source and open-source models on the 250-question benchmark. The findings remain consistent with those in the main text, confirming that human performance significantly surpasses that of all MLLMs.

the post-prompt. The decoding parameter, temperature, is set to 0 to ensure deterministic inference.

### 9. Visualization Results

- • Self-Consistency: Following the Self Consistency approach [50], we prompt MLLMs to generate multiple independent responses for each question. To encourage diversity, we set temperature to 0.6 and conduct five independent inference runs, selecting the most frequently occurring prediction as the final answer.
- • Identification CoT: Designed specifically for counting tasks, Identification CoT prompts the MLLM to list each target entity visible across all views, mitigating the risk of double-counting and improving accuracy. The corresponding prompt is shown in Figure 13. This method

In this section, we present a comprehensive visualization of the scene across all available views for convenience and consistency. While some questions do not require every view, we ensure that only the relevant ones are provided as input during inference.

#### 9.1. Benchmark Examples

Figures 16 and 17 showcase additional primary questionand-answer examples. These illustrations highlight the multi-view image inputs alongside their corresponding

Identification CoT

Let’s think step by step before answering the question! Your goals:

- 1. Internally generate a detailed description of each visible person|note their appearance, clothing, orientation, movements, and any nearby detailed described objects or people.
- 2. Cross-check these detailed descriptions across all views to avoid double-counting the same individual.
- 3. Arrive at the total number of unique people in the scene. Important Instructions:

- You should perform your step-by-step reasoning privately and not reveal it to the user.

- Figure 13. Our proposed Identification CoT prompt. To design for counting tasks with partial-visibility, our prompt guides the MLLM to systematically list each target entity across all views.

correctly identify the answer, and their reasoning processes are coherent and logically sound, demonstrating their ability to follow step-by-step inference when the task aligns well with their learned knowledge.

#### 9.4. Human Evaluation Failure Cases

Figure 22 presents three questions that human evaluators answered incorrectly. Analyzing these errors underscores the robustness of our annotation process while also highlighting the challenge and complexity of our benchmark. Additionally, the errors highlight the challenge and complexity of our benchmark, demonstrating its effectiveness in evaluating MLLMs’ ability to understand spatial relationships in multi-view images.

#### 9.5. Visualization Prompt

Figure 28 displays the prompt used to visualize scene reconstruction and camera pose alignment, as detailed in our paper. This prompt allows us to assess how well GPT4o and Gemini-2.0-Flash handle orientation-sensitive challenges, further validating their spatial reasoning capabilities.

tasks and Q&A pairs, demonstrating the diversity and complexity of our benchmark.

#### 9.2. Pair Data Examples

Figures 18 and 19 provide more examples of generated paired data. Each figure presents both the primary Q&A and the generated pair Q&A, presented side by side to illustrate their structural alignment and transformation process. This comparison emphasizes how the paired Q&A is derived from the primary one, reinforcing the dataset’s consistency and utility.

#### 9.3. Reasoning Examples

Figures 20 and 21 depict three reasoning cases evaluated using GPT-4o [40] and Gemini-2.0-Flash [45]. We modify the post-prompt to instruct the models not only to generate answers but also to provide detailed reasoning. Our analysis reveals distinct reasoning patterns across the three evaluated cases. In case 1, both GPT-4o and Gemini-2.0-Flash select incorrect answers, indicating challenges in understanding the underlying spatial relationships. In case 2, while GPT4o arrives at the correct answer, its reasoning process contains logical inconsistencies, suggesting that the model may have relied on heuristic shortcuts rather than fully comprehending the question. In contrast, Gemini-2.0-Flash fails to produce the correct response. While in case 3, both models

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

View 2 View 3 View 4

View 1

|Question: Here are multiple camera views which are pointing to the same scene, could you count how many people in total are in the scene?|
|---|

|GPT-4o|
|---|

|InternVL2.5-38B|
|---|

Baseline

|I have some observations from Views:<br><br>1.Basketball player (1 person) appears in multiple images; 2.Two seated individuals (2 people) consistently visible on the bench. The same basketball player is visible across multiple images. The two seated individuals remain unchanged.<br><br>Final Count: 3 unique people.<br><br>|1. Basketball player (1 person) appears in multiple images; 2. Two seated individuals (2 people) consistently visible on the bench; 3. Two standing individuals (2 people) seen in the<br><br>background in at least one view.<br><br>The same basketball player is visible across multiple images; The two seated individuals remain unchanged; The two standing individuals appear in at least one frame but are the same across views. Final Count: 5 unique people.<br><br>|
|---|---|

Zero-Shot CoT

|View 1 Analysis: One person in black (basketball player) near the hoop; Two people sitting on the bench; A tripod is present (not a person).<br>View 2 Analysis: The same basketball player is visible; Two new individuals standing in the background, not seen in View 1; The seated individuals are not clearly visible here.<br>View 3 Analysis: Confirms the basketball player; The two seated individuals are back in view; The two standing individuals from View 2 are faintly visible.<br>View 4 Analysis: Clearly shows the two seated individuals again; The basketball player is not visible in this frame. Final Cross-Check & Count: Basketball player (1) seen across multiple views. Two seated individuals (2) consistently visible. Two standing individuals (2) appear in at least one view. Final Count: 5 unique people.<br><br><br>|Identify People in Each View:<br><br>View 1: One basketball player, two seated individuals.<br>View 2: Same basketball player, two additional standing people in the background.<br>View 3: Confirms the basketball player, two seated people, and the standing people from View 2.<br>View 4: Confirms the two seated individuals.<br><br><br>Cross-Check to Avoid Double Counting: The basketball player appears in multiple views but is the same person; The two seated individuals remain the same across images; The two standing individuals from View 2 also appear faintly in View 3.<br><br>Final Count: Basketball player (1); Two seated individuals (2); Two standing individuals (2) Total = 5 unique people<br><br>|
|---|---|

Identification CoT

|Step 1: Detailed Description of Each View<br><br>View 1: One person in the foreground: Wears a black outfit (black T-shirt, black pants, black sneakers with white soles);Holding a basketball; Facing the basketball hoop, preparing to shoot. Two people sitting on the bench in the background: One wears a white hoodie and shorts; The other wears dark clothing.<br>View 2: Same person from View 1 (black outfit, holding a basketball); Now visible from a slightly different angle, still preparing to shoot. Additional two people near the left side of the image (not seen in View 1): One wears a black T-shirt and dark pants; The other wears a gray T-shirt and dark pants; The two people sitting on the bench from View 1 are not clearly visible in this image.<br>View 3: Same person in black outfit, holding a basketball, still near the hoop. The two people sitting on the bench (white hoodie & dark clothing) appear again. No additional new people spotted.<br>View 4: Focuses on the bench area: The two people sitting on the bench (white hoodie & dark clothing) are now clearly visible. No visible players or standing individuals in this frame.<br><br><br>Step 2: Merging Process to Prevent Double Counting Start with View 1: Identify 3 people (1 player near the hoop + 2 sitting on the bench).<br><br>In View 2: Recognize that the basketball player is the same person as in View 1; The two new people standing near the left side were not in View 1, so they are counted; The bench people are not visible but already counted.<br>In View 3: The basketball player is still the same person, no new people appear; The bench sitters appear again but are already counted.<br>In View 4: The bench sitters appear again, but they are already counted in previous views. Final Count<br><br><br>1 basketball player (black outfit, seen in multiple views but counted once).<br>2 people sitting on the bench (white hoodie & dark clothing, counted from View 1). 2 additional standing people in View 2 (not seen in View 1, so they are unique additions). Total Unique People in the Scene: 5<br><br><br>|Here is a step-by-step breakdown of the reasoning process to count the number of unique people<br><br>in the scene:<br><br>Step 1: Analyze Each View Individually Each image provides a different perspective of the same gymnasium scene. To avoid doublecounting individuals, I will analyze each image separately while noting the distinct clothing, posture, and positioning of the people.<br><br>View 1 A person in black clothing (black T-shirt, black pants, black shoes) is holding a basketball near the hoop. Two people sitting on the bench in the background, one wearing a white hoodie and another in darker clothing. A tripod is also visible but does not count as a person.<br><br>View 2 The same person in black clothing is visible near the basketball hoop. Two additional individuals are standing in the background, wearing different clothing (possibly one in a grey shirt and another in dark clothing). The two seated individuals from the previous image are not clearly visible in this view.<br>View 3 The person in black clothing is again seen near the basketball hoop. The two seated individuals are visible in the background again. The people from View 2 standing near the back are also faintly visible. Tripods are still visible.<br><br>View 4 The two seated individuals on the bench are clearly visible. The person in black (basketball player) is not visible in this frame. The background does not show the standing individuals from View 2.<br><br><br>Step 2: Cross-Checking Across Views Basketball player (Black T-shirt, Black pants, holding a ball) appears in Views 1, 2, and 3. Two seated individuals appear in Views 1, 3, and 4. Two standing individuals in the background appear in View 2 and faintly in View 3.<br>Step 3: Confirming Unique Individuals By tracking each person’s unique features and ensuring they are not double-counted: Basketball player (visible in Views 1, 2, and 3). Seated person in white hoodie (visible in Views 1, 3, and 4). Seated person in dark clothing (visible in Views 1, 3, and 4). Standing person near back (grey or white shirt) (visible in Views 2 and faintly in View 3). Standing person near back (dark clothing) (visible in Views 2 and faintly in View 3). Thus, the final count of unique people in the scene is 5.<br><br><br>|
|---|---|

Figure 14. Comparison of model outputs on the same questions under different prompting methods. (Case 1)

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

View 2 View 3 View 4

View 1

|Question: Here are multiple camera views which are pointing to the same scene, could you count how many people in total are in the scene?|
|---|

|GPT-4o|
|---|

|InternVL2.5-38B|
|---|

Baseline

|After analyzing the images, I observed that View 1 shows 2 individuals, View 2 captures 1, View 3 includes 2, and View 4 displays 1 visible person. By cross-referencing distinct features such as clothing, posture, and positioning across views, I conclude that the total number of unique individuals in the scene is 2.<br><br>|By cross-referencing these details and ensuring that no individuals are counted multiple times, I<br><br>have determined the total number of distinct people present in the scene. Based on this assessment, my final conclusion is that there are three unique individuals in the scene.<br><br>|
|---|---|

Zero-Shot CoT

|Step 1: Identify and Describe People in Each Camera View<br><br>View 1: One person in a white T-shirt and blue jeans bending over, holding red plastic blocks.<br>View 2: A person in a dark red button-up shirt and black pants is walking towards the scene, another person in a white T-shirt is partially visible.<br>View 3: A clearer view of two people, one in a white T-shirt and another in a beige outfit, interacting with large plastic blocks.<br>View 4: A wider view, showing the person in a white T-shirt bending over and another person standing near him, plus additional tripods.<br><br><br>Step 2: Avoid Double Counting Since the same individuals appear in multiple images, we must ensure we do not count them multiple times. The total number of distinct individuals identified in the scene is 3. Final Answer: 3<br><br><br>|Step 1: Analyzing Each Camera View<br><br>View 1: A person in a white T-shirt and blue jeans is bending over, holding red plastic blocks. Background includes a TV, cabinet, tripods, and a blue couch.<br>View 2: A person in a dark red/brown long-sleeve shirt and black pants is walking toward the center. Another person in a white shirt and light-colored pants is near the blue couch, holding green blocks.<br>View 3: The individual in a white T-shirt and blue jeans is interacting with plastic blocks. The person in a white shirt and light-colored pants is standing nearby.<br>View 4: The same two individuals from View 3 are visible again, with multiple tripods along the right side. The room layout remains consistent.<br><br><br>Step 2: Identifying Unique Individuals<br><br>Person 1: White T-shirt, blue jeans, bending down (Views 1, 3, 4).<br>Person 2: White shirt, light-colored pants, standing and holding a block (Views 2, 3, 4).<br>Person 3: Dark red/brown long-sleeve shirt, black pants (only in View 2). Final Prediction: The total number of unique people in the scene is 3.<br><br><br>|
|---|---|

Identification CoT

|Let's carefully identify and describe each individual in the images to ensure an accurate count of unique people.<br><br>Step 1: Identify People in Each Camera View<br><br>View 1: One person is clearly visible in the foreground. Description: Wearing a white T-shirt and blue jeans, bending forward while holding red plastic blocks. Other details: Their face is not visible as they are facing downward. Background: A TV on a stand, a tripod, and a blue couch are visible.<br>View 2: A new person appears prominently in this image. Description: Wearing a dark red button-up shirt and black pants, facing away from the camera, walking toward the scene. Other people visible: The person in the white T-shirt from View 1 is partially visible, still engaging with the blocks. Background: Kitchen area, counter, and a laptop.<br>View 3: A wider view of the scene shows two people interacting. Person 1 (previously identified): White T-shirt, blue jeans, bending over while handling large plastic blocks. Person 2 (new): Wearing a beige outfit (long-sleeved shirt and pants), standing near the blocks. Other details: The person in the white T-shirt is still bending forward, while the person in beige is standing upright.<br>View 4: Provides an even wider view of the room. Person 1 (previously identified): White T-shirt, blue jeans, still bending over. Person 2 (previously identified): Beige outfit, standing and holding a green plastic block. Additional details: Three tripods set up along the window, ensuring multiple camera angles of the same scene.<br><br><br>Step 2: Cross-Check Individuals Across Views White T-shirt, blue jeans (bending over, handling blocks) → Appears in all images. Dark red button-up shirt, black pants (walking toward the scene) → Seen in Image 2, not clearly visible in others. Beige long-sleeved shirt, pants (standing near the blocks) → Clearly visible in Images 3 and 4.<br>Step 3: Avoid Double Counting The same individuals appear in different images from varying angles, but we confirm that there are three unique people in total. Final Answer: 3<br><br><br>|Step 1: Analyze Each View Individually<br><br>View 1: A person in a white T-shirt and blue jeans is prominently in the foreground, bending down while holding red plastic blocks. Their face is not visible due to their posture. The background includes a TV on a cabinet, tripods, and a blue couch.<br>View 2: A person in a dark red/brown long-sleeve shirt and black pants is seen from behind, walking towards the center of the scene. Another individual, wearing a white shirt and lightcolored pants, is in the background holding green blocks near the blue couch. The kitchen counter, tripods, and part of the play area are visible in the background.<br>View 3: The individual in the white T-shirt and blue jeans is again visible, bending and interacting with large plastic blocks. The person in the white shirt and light-colored pants is standing and holding a green block. Background elements such as the TV, cabinet, and tripods are visible.<br>View 4: The individual in the white T-shirt and blue jeans is seen again, continuing to engage with the plastic blocks. The person in the white shirt and light-colored pants remains in the scene. Multiple tripods are positioned along the right side. The room layout, including the window, TV, couch, and play area, is evident.<br><br><br>Step 2: Cross-Checking Across Views<br><br>Person 1: Wearing a white T-shirt and blue jeans, seen bending down in multiple views.<br>Person 2: Wearing a white shirt and light-colored pants, standing and holding a green block.<br>Person 3: Wearing a dark red/brown long-sleeve shirt and black pants, only visible in View 2. The final answer for the number of unique people present is 3.<br><br><br>|
|---|---|

Figure 15. Comparison of model outputs on the same questions under different prompting methods. (Case 2)

- View 2 and View 3 are two different views that represent Counting the same scene. In View 3, there is a person with blue jeans. Could you identify the same person in View 2?

- (A) the person holding the basketball
- (B) the person on the most left in View 2
- (C) the person on the most right in View 2

Attribute Identification

View 1 and View 3 are two different camera views that represent the same scene. In which view, the man in the purple shirt is farther from the spot where the camera view was positioned?

- (A) farther from the spot where camera View 1 was positioned
- (B) farther from the spot where camera View 3 was positioned (C) distance to the spot where camera View 1 and View 3 were positioned is equal

Relative Distance View 3 and View 4 are two different views that represent the same scene. In View 4, the man in a green hoodie and blue jeans is facing away from the spot where camera view was positioned. Which direction is the same person facing in View 3?

- (A) facing to the left side of the View 3
- (B) facing to the right side of the View 3
- (C) facing to the spot where camera View 3 was positioned

Relative Direction

What if the person wearing purple t-shirt is walking away from the spot where camera view is positioned in camera View 4, what would be the trajectory of the same person

in View 2?

- (A) walking to the left side of the View 2
- (B) walking to the spot where camera View 2 was positioned
- (C) walking to the right side of the View 2

Object Manipulation View 1, View 2, View 3, View 4 are four different input camera views that represent the same scene. Could you reconstruct the camera view layout from the top-bottom view, and answer the following question?

- (A) in clockwise order, the layout is View 1, View 2, View 3, View 4
- (B) in clockwise order, the layout is View 2, View 1, View 4, View 3
- (C) in clockwise order, the layout is View 2, View 3, View 1, View 4

Camera Pose Estimation

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

|View 1<br><br>[Figure 391]|View 2<br><br>[Figure 392]|
|---|---|
|View 4<br><br>[Figure 393]|View 3<br><br>[Figure 394]|

Here are multiple camera views which are pointing to the same scene, could you count how many volleyball nets in

total are in the scene?

(A) 1 (B) 2 (C) 3

Counting

View 1 and View 3 are two different views that represent

the same scene. In View 1, there is a person wearing a red shirt and black pants standing near the center of the

court. Could you identify the same person in View 3?

- (A) the person near the center in View 3
- (B) the person on the most left in View 3
- (C) the person on the most right in View 3

Attribute Identification

- View 3 and View 1 are two different camera views that represent the same scene. In which view, the net is closer to the spot where the camera view was positioned?

Here are multiple camera views which are pointing to the same scene, could you count how many people in total are in the scene?

|View 1<br><br>[Figure 395]|View 2<br><br>[Figure 396]|
|---|---|
|View 4<br><br>[Figure 397]|View 3<br><br>[Figure 398]|

(A) 5 (B) 4 (C) 3

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

###### Relative Direction

[Figure 404]

Relative Distance Viewthe same4 andscene.ViewIn1 areViewtwo4, thedifferentplayerviewsstandingthatnearrepresentthe far-

right side of the net is facing to the spot where camera

view was positioned. Which direction is the same player facing in View 1?

[Figure 405]

- (A) facing to the left side of the View 1
- (B) facing to the right side of the View 1
- (C) facing to the spot where camera View 1 was positioned

- (A) closer to the spot where camera View 1 was positioned
- (B) closer to the spot where camera View 3 was positioned
- (C) distance to the spot where camera View 1 and View 3 were positioned is equal

[Figure 406]

[Figure 407]

###### Camera Pose Estimation

[Figure 408]

Object Manipulation View 1, View 2, View 3, View 4 are four different input camera views that represent the same scene. Could you reconstruct the camera view layout from the top-bottom view, and answer the following question?

What if the volleyball net is being pulled slightly closer to the camera position in camera View 4, what would be the trajectory of the same net in camera View 1?

- (A) in clockwise order, the layout is View 1, View 2, View 3, View 4
- (B) in clockwise order, the layout is View 2, View 3, View 1, View 4
- (C) in clockwise order, the layout is View 1, View 3, View 2, View 4

- (A) moving toward the left side of the View 1
- (B) moving toward the spot where camera View 1 was positioned
- (C) moving toward the right side of the View 1

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

###### Counting

Here are multiple camera views which are pointing to the same scene, could you count how many people in total are in the scene?

- View 1 and View 3 are two different views that represent the same scene. In View 3, there is a person wearing a pink shirt standing on the far-right side. Could you identify the same person in View 1?

- (A) the person sitting on the bench
- (B) the person standing near the net in View 1
- (C) the person not visible in View 1

Attribute Identification

- View 2 and View 3 are two different camera views that represent the same scene. In which view, the tennis net is farther from the position of camera view?

[Figure 413]

|View 1<br><br>[Figure 414]|View 2<br><br>[Figure 415]|
|---|---|
|View 4<br><br>[Figure 416]|View 3<br><br>[Figure 417]|

(A) 3 (B) 4 (C) 5

[Figure 418]

[Figure 419]

###### Relative Direction

Relative Distance Viewthe same3 andscene.ViewIn1 areViewtwo3, thedifferentman wearingviews thatwhiterepresentcup is facing towards the right side of the view. Which direction is the same person facing in View 1?

[Figure 420]

- (A) facing to the left side of the view
- (B) facing to the right side of the view
- (C) facing to the spot where camera View 1 was positioned

[Figure 421]

[Figure 422]

- (A) farther from the spot where camera View 2 was positioned
- (B) farther from the spot where camera View 3 was positioned
- (C) distance to the spot where camera View 2 and View 23were positioned is equal

[Figure 423]

Camera Pose Estimation

[Figure 424]

Object Manipulation View 1, View 2, View 3, View 4 are four different input

camera views that represent the same scene. Could you

What if the person wearing the gray t-shirt is running

reconstruct the camera view layout from the top-bottom view, and answer the following question?

toward the tennis net in camera View 2, what would be the trajectory of the same person in View 4?

[Figure 425]

[Figure 426]

- (A) in clockwise order, the layout is View 1, View 2, View 3, View 4
- (B) in clockwise order, the layout is View 2, View 1, View 3, View 4
- (C) in clockwise order, the layout is View 2, View 3, View 1, View 4

- (A) moving toward the left side of the view
- (B) moving toward the right side of the view
- (C) standing without moving

- Figure 16. All-Angles Bench Samples (Part I)

###### Attribute Identification

View 3 and View 2 are two different views that represent Counting the same scene. In View 3, there is a person in a black outfit on the left side. Could you identify the same person in View 2?

Here are multiple camera views which are pointing to the same scene, could you count how many people in total are in the scene?

|View 1<br><br>[Figure 427]|View 2<br><br>[Figure 428]|
|---|---|
|View 4<br><br>[Figure 429]|View 3<br><br>[Figure 430]|

(A) 5 (B) 6 (C) 7

- (A) the person closer to the center of View 2
- (B) the person near the left side of View 2
- (C) the person not visible in View 2

- View 1 and View 4 are two different camera views that represent the same scene. In which view, the player in the red jersey dribbling the ball is closer to the spot where the camera view was positioned?

- (A) closer to the spot where camera View 1 was positioned
- (B) closer to the spot where camera View 4 was positioned
- (C) distance to the spot where camera View 1 and View 4 were positioned is equal

Relative Distance Viewthe same4 andscene.ViewIn1 areViewtwo4, thedifferentman withviewsthethatredrepresentshirt and white socks is facing slightly towards the right side of the view. Which direction is the same person facing in View 1?

- (A) facing slightly to the left side of the View 1
- (B) facing slightly to the right side of the View 1
- (C) facing directly away from the spot where camera View 1 was positioned

Relative Direction

What if the person in the red shirt kicking the ball in View 4 starts moving toward the goal, what would be the trajectory of the same person in View 2?

- (A) moving toward the left side of the view
- (B) moving toward the right side of the view
- (C) moving toward the spot where camera View 2 was positioned

Object Manipulation Viewcamera1, Viewviews2,thatViewrepresent3, Viewthe4 aresamefourscene.differentCouldinputyou reconstruct the camera view layout from the top-bottom view, and answer the following question?

- (A) in clockwise order, the layout is View 1, View 2, View 3, View 4
- (B) in clockwise order, the layout is View 4, View 3, View 2, View 1
- (C) in clockwise order, the layout is View 2, View 3, View 1, View 4

Camera Pose Estimation

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

|View 1<br><br>[Figure 443]|View 2<br><br>[Figure 444]|
|---|---|
|View 4<br><br>[Figure 445]|View 3<br><br>[Figure 446]|

Here are multiple camera views which are pointing to the

same scene, could you count how many basketball hoops in

total are in the scene?

(A) 2 (B) 3 (C) 4

Counting

- View 2 and View 4 are two different views that represent

- the same scene. In View 2, there is a person in a white tshirt standing under the basketball hoop. Could you identify the same person in View 4?

- (A) the person near the basketball
- (B) the person in a blue t-shirt
- (C) the person closer to the center of View 2

Attribute Identification

- View 2 and View 1 are two different camera views that represent the same scene. In which view, the basketball hoop is farther from the spot where the camera view was positioned?

- (A) farther from the spot where camera View 1 was positioned
- (B) farther from the spot where camera View 2 was positioned
- (C) distance to the spot where camera View 1 and View 2 were positioned is equal

Relative Distance

View 3 and View 4 are two different views that represent the same scene. In View 4, the man with the blue t-shirt is facing away from the spot where camera View 4 was positioned. Which direction is the same person facing in View 3?

- (A) facing to the left side of the view
- (B) facing to the right side of the view
- (C) facing away from the spot where camera View 1 was positioned

Relative Direction

What if the basketball is thrown toward the spot where

camera view is positioned in camera View 4, what would be the trajectory of the basketball in View 3?

- (A) moving toward the left side of the View 3
- (B) moving toward the right side of the View 3
- (C) moving toward the spot where camera View 3 was positioned

Object Manipulation View 1, View 2, View 3, View 4 are four different input camera views that represent the same scene. Could you reconstruct the camera view layout from the top-bottom view, and answer the following question?

- (A) in clockwise order, the layout is View 1, View 4, View 3, View 2
- (B) in clockwise order, the layout is View 1, View 2, View 3, View 4
- (C) in clockwise order, the layout is View 2, View 3, View 1, View 4

Camera Pose Estimation

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

[Figure 457]

[Figure 458]

|[Figure 459]<br><br>View 1|[Figure 460]<br><br>View 2|
|---|---|
|[Figure 461]<br><br>View 4|[Figure 462]<br><br>View 3|

Here are multiple camera views which are pointing to the same scene, could you count how many people in total are

in the scene?

(A) 3 (B) 4 (C) 5

Counting

- View 3 and View 1 are two different views that represent

- the same scene. In View 3, there is a frying pan with a wooden spatula on the stove. Could you identify the same frying pan in View 1?

[Figure 463]

[Figure 464]

###### Attribute Identification

[Figure 465]

- (A) next to the cutting board on the counter
- (B) on the background of View 1
- (C) inside the sink

[Figure 466]

[Figure 467]

###### Relative Direction

[Figure 468]

Relative Distance Viewthe same2 andscene.ViewIn4 areViewtwo2, thedifferentrefrigeratorviews thatis facingrepresenttoward

the camera view. Which direction is the same refrigerator

View 1 and View 2 are two different camera views that

facing in View 4?

represent the same scene. In which view, the white chair with a brown seat is closer to spot where the camera view was positioned?

[Figure 469]

- (A) facing to the left side of the View 4
- (B) facing to the right side of the View 4
- (C) facing to the spot where camera View 4 was positioned

- (A) closer to the spot where camera View 1 was positioned
- (B) closer to the spot where camera View 2 was positioned
- (C) distance to the spot where camera View 1 and View 2 were positioned is equal

[Figure 470]

[Figure 471]

Camera Pose Estimation

View 1, View 2, View 3, View 4 are four different input camera views that represent the same scene. Could you reconstruct the camera view layout from the top-bottom view, and answer the following question?

[Figure 472]

Object Manipulation

What if the person sitting on the chair in camera View 4 moves to stand behind the person cooking, what would be the trajectory of the same person in camera View 3?

- (A) in clockwise order, the layout is View 1, View 2, View 3, View 4
- (B) in clockwise order, the layout is View 1, View 3, View 4, View 2
- (C) in clockwise order, the layout is View 2, View 3, View 1, View 4

[Figure 473]

- (A) moving toward the right side of View 3
- (B) moving toward the left side of View 3
- (C) moving to the spot where camera View 3 was positioned

[Figure 474]

- Figure 17. All-Angles Bench Samples (Part II)

###### Attribute Identification

|View 1<br><br>[Figure 475]|View 2<br><br>[Figure 476]|
|---|---|
|View 4<br><br>[Figure 477]|View 3<br><br>[Figure 478]|

- View 1 and View 2 are two different views that represent the same scene. In View 2, there is a person balancing a soccer ball on his foot. Could you identify the same person in View 1?

- (A) the person in the white t-shirt
- (B) the person in the green t-shirt
- (C) the person sitting on the ground

View 1 and View 2 are two different views that represent the same scene. In View 1, there is a person in the white tshirt. Could you identify the same person in View 2?

(A) the person balancing a soccer ball on his foot (B) the person on the most left in View 2 (C) the person sitting on the ground

Primary Q&A Pair Q&A

[Figure 479]

[Figure 480]

Relative Distance

- View 2 and View 3 are two different camera views that represent the same scene. In which view, the man in the white t-shirt and black shorts closer to the spot where the camera view was positioned?

[Figure 481]

[Figure 482]

Primary Q&A Pair Q&A

View 2 and View 3 are two different camera views that represent the same scene. In which view, the man in the white

t-shirt and black shorts farther from the spot where the camera

view was positioned?

- (A) farther from the spot where camera View 3 was positioned
- (B) farther from the spot where camera View 2 was positioned
- (C) distance to the spot where camera View 3 and View 2 were positioned is nearly equal

- (A) closer to the spot where camera View 3 was positioned
- (B) closer to the spot where camera View 2 was positioned
- (C) distance to the spot where camera View 3 and View 2 were positioned is nearly equal

[Figure 483]

[Figure 484]

[Figure 485]

###### Object Manipulation

Primary Q&A Pair Q&A

|View 1<br><br>[Figure 486]|View 2<br><br>[Figure 487]|
|---|---|
|View 4<br><br>[Figure 488]|View 3<br><br>[Figure 489]|

What if the green scissors are moved away from the spot where camera view is positioned in camera View 3, what would be the trajectory of the same green scissors in View 2?

What if the green scissors are moved closer to the spot where camera view is positioned in camera View 1, what would be the trajectory of the same green scissors in View 2?

[Figure 490]

- (A) moving toward the left side of the view
- (B) moving toward the right side of the view
- (C) remain in the same place

- (A) moving toward the left side of the view
- (B) moving toward the right side of the view
- (C) remain in the same place

[Figure 491]

[Figure 492]

###### Relative Direction

Primary Q&A Pair Q&A

View 4 and View 3 are two different views that represent

View 4 and View 3 are two different views that represent the same scene. In View 3, the green scissors are facing towards the spot where camera view was positioned. Which direction are the green scissors facing in View 4?

the same scene. In View 4, the green scissors are facing to the left side of the view. Which direction are the green scissors facing in View 3?

[Figure 493]

- (A) facing to the left side of the View 3
- (B) facing to the right side of the View 3
- (C) facing to the spot where camera View 3 was positioned

- (A) facing to the left side of the View 4
- (B) facing to the right side of the View 4
- (C) facing to the spot where camera View 4 was positioned

[Figure 494]

|View 1<br><br>[Figure 495]|View 2<br><br>[Figure 496]|
|---|---|
|View 4<br><br>[Figure 497]|View 3<br><br>[Figure 498]|

[Figure 499]

###### Camera Pose Estimation

Primary Q&A Pair Q&A

View 1, View 2, View 3, View 4 are four different input camera views that represent the same scene. Could you reconstruct the camera view layout from the top-bottom

View 1, View 2, View 3, View 4 are four different input camera views that represent the same scene. Could you reconstruct the camera view layout from the top-bottom view, and answer the following question?

view, and answer the following question?

- (A) in counterclockwise order, the layout is View 4, View 3, View 2, View 1
- (B) in counterclockwise order, the layout is View 1, View 2, View 3, View 4
- (C) in counterclockwise order, the layout is View 4, View 1, View 3, View 2

- (A) in clockwise order, the layout is View 1, View 2, View 3, View 4
- (B) in clockwise order, the layout is View 4, View 3, View 2, View 1
- (C) in clockwise order, the layout is View 2, View 3, View 1, View 4

[Figure 500]

[Figure 501]

- Figure 18. Paired Data Samples (Part I)

###### Attribute Identification

|View 1<br><br>[Figure 502]|View 2<br><br>[Figure 503]|
|---|---|
|View 4<br><br>[Figure 504]|View 3<br><br>[Figure 505]|

- View 1 and View 2 are two different views that represent the same scene. In View 1, there is a black bag placed on the floor under the basketball hoop. Could you identify the same black bag in View 2?

- (A) the black bag placed on the right side in View 2
- (B) the black bag placed on the center left side in View 2 (C) the black bag placed near the yellow bin in View 2

[Figure 506]

View 1 and View 2 are two different views that represent the same scene. In View 2, there is a black bag placed on the center left side. Could you identify the same black bag

in View 1?

- (A) the black bag placed on the floor under the basketball hoop
- (B) the black bag placed near the yellow bin in View 1
- (C) the black bag placed near the spot where camera View 1 was positioned

Primary Q&A Pair Q&A

[Figure 507]

[Figure 508]

Object Manipulation

[Figure 509]

What if the person wearing the pink shirt moves to the right side in camera View 3, what would be the trajectory of the same person in camera View 4?

- (A) moving toward the left side of View 4
- (B) moving toward the right side of View 4
- (C) moving toward the background

[Figure 510]

What if the person wearing the pink shirt moves to the left side in camera View 3, what would be the trajectory of the same person in camera View 4?

- (A) moving toward the left side of View 4
- (B) moving toward the right side of View 4
- (C) moving toward the background

Primary Q&A Pair Q&A

[Figure 511]

|View 1<br><br>[Figure 512]|View 2<br><br>[Figure 513]|
|---|---|
|View 4<br><br>[Figure 514]|View 3<br><br>[Figure 515]|

Relative Direction

- View 1 and View 3 are two different views that represent the same scene. In View 1, a woman in the black shirt is facing away from the spot where camera view was positioned. Which direction is the same woman facing in View 3?

[Figure 516]

###### Relative Distance

Primary Q&A Pair Q&A

|View 1<br><br>[Figure 517]|View 2<br><br>[Figure 518]|
|---|---|
|View 4<br><br>[Figure 519]|View 3<br><br>[Figure 520]|

View 1 and View 3 are two different camera views that represent the same scene. In which view, the white milk jug on the counter farther from the spot where the camera view was positioned?

View 1 and View 3 are two different camera views that represent the same scene. In which view, the white milk jug on the counter

closer to the spot where the camera view was positioned?

[Figure 521]

- (A) farther from the spot where camera View 1 was positioned
- (B) farther from the spot where camera View 3 was positioned
- (C) distance to the spot where camera View 1 and View 3 were positioned is equal

(A) closer to the spot where camera View 1 was positioned (B) closer to the spot where camera View 3 was positioned (C) distance to the spot where camera View 1 and View 3 were positioned is equal

[Figure 522]

[Figure 523]

###### Camera Pose Estimation

Primary Q&A Pair Q&A

View 1, View 2, View 3, View 4 are four different input camera views that represent the same scene. Could you reconstruct the camera view layout from the top-bottom

View 1, View 2, View 3, View 4 are four different input camera views that represent the same scene. Could you reconstruct the camera view layout from the top-bottom view, and answer the following question?

view, and answer the following question?

[Figure 524]

[Figure 525]

- (A) in counterclockwise order, the layout is View 4, View 3, View 2, View 1
- (B) in counterclockwise order, the layout is View 1, View 2, View 3, View 4 (C) in counterclockwise order, the layout is View 4, View 1, View 3, View 2

- (A) in clockwise order, the layout is View 1, View 2, View 3, View 4
- (B) in clockwise order, the layout is View 4, View 3, View 2, View 1
- (C) in clockwise order, the layout is View 2, View 3, View 1, View 4

[Figure 526]

Primary Q&A Pair Q&A

View 1 and View 3 are two different views that represent the same scene. In View 3, a woman in the black shirt is facing to the spot where camera view was positioned. Which direction is the same woman facing in View 1?

- (A) facing to the left side of the View 1
- (B) facing to the right side of the View 1
- (C) facing away from the spot where camera View 1 was positioned

(A) facing to the left side of the View 3 (B) facing to the right side of the View 3 (C) facing to the spot where camera View 3 was positioned

[Figure 527]

[Figure 528]

- Figure 19. Paired Data Samples (Part II)

|View 1<br><br>[Figure 529]|View 2<br><br>[Figure 530]|
|---|---|
|View 4<br><br>[Figure 531]|View 3<br><br>[Figure 532]|

[Figure 533]

Camera Pose Estimation

View 1, View 2, View 3, View 4 are four different input camera views that represent the same scene. Could you reconstruct the camera view layout from the top-bottom view, and answer the following question?

Please provide a detailed description of the relative positions of

each view and the spatial relationships between them.

Additionally, create a sketch illustrating the layout of the images. Finally, sort all the input images in clockwise order

Answer: In clockwise order, the layout is View 1, View 3, View 4, View 2

|Gemini- 2.0-Flash|
|---|

|GPT-4o|
|---|

|Okay, I can help you reconstruct the camera layout and sort the images in clockwise order. Here's my analysis:<br><br>1. Identifying Common Elements: The Person: The person is the central figure in all images. The Stove: The stove is visible in all images and is a key reference point. The Kitchen Cabinets: The white kitchen cabinets are visible in all images and provide a backdrop. Tripods: There are several tripods with cameras set up around the kitchen.<br>2. Analyzing Relative Positions:<br><br>View 1: This view is from the front-left of the person, showing their face and the stove from an angle. We can see another person in the background on the left.<br><br>View 2: This view is from the left side of the person, showing the side of their face and the stove from a more side-on angle.<br>View 3: This view is from the back of the person, showing their back and the stove directly in front of them.<br>View 4: This view is from the back-left of the person, showing their back and the stove from an angle.<br><br><br>3. Reconstructing the Layout (Top-Down View): Based on the relative positions, here's a sketch of the camera layout:<br>4. Clockwise Order: Starting from the front-left (Image-1) and moving clockwise around the person: View 1 (Front-Left); View 2 (Left); View 4 (Back-Left); View 3 (Back) Therefore, the images in clockwise order are: View 1, View 2, View 4, View 3<br><br><br>View 1 (Front-Left)<br>View 2 (Left)<br><br><br>Person<br><br>View 3 (Back)<br>View 4 (Back-Left)<br><br><br>|To analyze the scene and determine the relative positions of the camera views, let's examine each image and figure out the camera angle. By doing so, we can reconstruct the layout from a top-down perspective and then arrange the images in clockwise order.<br><br>Analysis:<br><br>View 1: - The scene shows a person working on the stove, with the view primarily focused on the right side of the person.<br><br>- The cabinets are visible to the right, and the stove is on the left of the image.<br>- There's another person visible on the far left, indicating that the photo is taken from the left<br><br><br>side of the kitchen.<br><br>View 2: - The person is facing the stove, with a camera tripod clearly visible in the scene.<br><br>- The refrigerator is visible further back to the right side.<br>- This image is taken from the right side towards the central part of the kitchen.<br><br><br>View 3: - The person is standing directly in front of the stove. This is a rear view, showing the back of the person.<br><br>- The stove and cabinets are directly in front of them.<br><br>View 4: - This image is taken from the back-left of the person.<br><br><br>- The window is clearly visible to the left, suggesting the camera is positioned from the<br><br>back-left corner of the kitchen. Sketch and Positioning: Based on the observations, the layout can be visualized in the kitchen as follows:<br><br>Clockwise Order: View 4(Top-left corner view); View 3 (Direct rear view); View 2 (Right side view); View 1 (Left side view)<br><br>So, based on the layout and analysis:<br><br>- Clockwise order starting from the top-left corner is: View 4, View 3, View 2, View 1<br><br><br>View 1<br>View 2<br><br><br>View 3<br>View 4<br><br><br>Refrigerator<br><br>Window<br><br>Stove<br><br>|
|---|---|

- Figure 20. Visualization of the reasoning process for two MLLMs (Part I). In this case, both MLLMs choose incorrect options due to errors in their reasoning process.

[Figure 534]

|View 1<br><br>[Figure 535]|View 2<br><br>[Figure 536]|
|---|---|
|View 4<br><br>[Figure 537]|View 3<br><br>[Figure 538]|

Object Manipulation

What if piano bench is pushed backward in View 3, what would be the trajectory of the same piano bench in View 1? Please select the correct answer from the following options and provide a detailed reasoning process.

(A) moving toward the left side of the View 1 (B) moving toward the right side of the View 1 (C) moving toward the spot where camera View 1 was positioned

[Figure 539]

|GPT-4o|
|---|

|Gemini- 2.0-Flash|
|---|

|Here's the breakdown of the answer: Understanding "Backward”: In the context of the image, "backward" relative to the piano player means away from the piano. Relating to View 1: Considering the camera angle in View 1, moving the bench away from the piano would mean the bench is moving toward the left side of the image. Therefore, the correct answer is: A. moving toward the left side of the View 1 Here's the reasoning: View 3 shows the piano bench is aligned with the center of the piano.<br><br>View 1 shows the piano bench is aligned with the center of the piano.<br><br>The camera position is different in both images, therefore the trajectory of the bench is moving toward the left side of the image.**<br><br>Therefore, the answer is A.<br><br>|To determine the trajectory of the piano bench when pushed backward, we need to consider the perspective and orientation of the two images.<br><br>In View 3, the camera is positioned to the left of the pianist, capturing a side view of the grand piano and the pianist. The piano bench is positioned in front of the pianist.<br><br>In View 1, the camera is positioned to the right side of the pianist, capturing a more direct side view of the pianist and the piano.<br><br>If the bench is pushed backward from the pianist's perspective in View 3, it will move away from the piano, towards the camera position in View 1.<br><br>In View 1, the camera is capturing the pianist from the opposite side compared to View 3. Therefore, the bench moving backward from the pianist's perspective in View 3 corresponds to moving toward the camera position in View 3, which aligns with moving toward the spot where camera View 1 was positioned.<br><br>Thus, the correct answer is: C. moving toward the spot where camera View 1 was positioned|
|---|---|

[Figure 540]

|View 1<br><br>[Figure 541]|View 2<br><br>[Figure 542]|
|---|---|
|View 4<br><br>[Figure 543]|View 3<br><br>[Figure 544]|

Relative Direction

View 4 and View 1 are two different views that represent the same scene. In View 4, the man in the long-sleeve shirt is kicking the ball towards the goal. Which direction is the same man facing in View 1?

Please select the correct answer from the following options and provide a detailed reasoning process.

[Figure 545]

(A) facing to the left side of the View 1 (B) facing to the right side of the View 1 (C) facing to the spot where camera View 1 was positioned

|GPT-4o|
|---|

|Gemini- 2.0-Flash|
|---|

|Here's the reasoning to determine the direction the man in the long-sleeve shirt is facing in View 1: View 4 Analysis:<br><br>In View 4, we see the man kicking the ball towards the goal, which is positioned to the left of him. We can infer that he's facing somewhat towards the left of the image. View 1 Analysis: View 1 is taken from a different angle, but we can still see the man kicking the ball. The goal is now visible on the left side of the image. Since the man is kicking towards the goal, he must be facing in that direction. Therefore, the correct answer is: A. facing to the left side of the View 1|To determine the direction the man in the long-sleeve shirt is facing in View 1, we need to analyze both images and their perspectives.<br><br>In View 4, the man in the long-sleeve shirt is kicking the ball towards the goal. The camera is positioned behind him and to the right of the goal, capturing the moment of the kick. We can observe the arrangement of the field, the goal net, and the buildings in the background.<br><br>In View 1, the scene is viewed from a different angle. The camera is positioned on the opposite side of the goal compared to View 4, looking back towards the goal. The man in the long-sleeve shirt is seen from a side view as he kicks the ball. By comparing the two images:<br><br>- In View 4, the man is facing the goal, which is towards the right side of the image.<br>- In View 1, the man is still facing the goal, but because the perspective has shifted, he is facing towards the left side of the image. Therefore, in View 1, the man in the long-sleeve shirt is facing to the left side of the View 1. The correct answer is: A. facing to the left side of the View 1<br>|
|---|---|

- Figure 21. Visualization of the reasoning process for two MLLMs (Part II). In the above case, GPT-4o selects the correct option but contain errors in its reasoning process. In the case below, both GPT-4o and Gemini-2.0-Flash follow a correct reasoning process and ultimately select the right answer.

[Figure 546]

|View 1<br><br>[Figure 547]|View 2<br><br>[Figure 548]|
|---|---|
|View 4<br><br>[Figure 549]|View 3<br><br>[Figure 550]|

###### Relative Distance

View 3 and View 1 are two different camera views that represent the same scene. In which view, the woman is closer to the spot where the camera view was positioned?

- (A) closer to the spot where camera View 1 was positioned
- (B) closer to the spot where camera View 3 was positioned
- (C) distance to the spot where camera View 1 and View 3 were positioned is nearly equal

|Evaluator Prediction: (A) Answer: (C) Error Review: Due to differences in viewing angles, a perspective illusion makes it appear that the woman is closer to the camera in View 1. However, considering the relative positions of the cameras, the woman is actually positioned approximately midway between<br><br>View 1 and View 3. Therefore, option C is the more reasonable choice.|
|---|

[Figure 551]

|View 1<br><br>[Figure 552]|View 2<br><br>[Figure 553]|
|---|---|
|View 4<br><br>[Figure 554]|View 3<br><br>[Figure 555]|

Relative Direction

View 2 and View 4 are two different views that represent the same scene. In View 2, the man in the blue t-shirt near the back wall is facing toward the right side of the view. Which direction is the same man facing in camera View 4?

- (A) facing to the left side of the View 4
- (B) facing to the right side of the View 4
- (C) facing to the spot where camera View 4 was positioned

|Evaluator Prediction: (C) Answer: (B) Error Review: When observing only View 4, the presence of the bicycle repair stand introduces a visual occlusion, which may create the illusion that the man is facing the camera. However, by examining View 2, we can clearly determine his actual orientation—he is facing the opposite direction of the white door. Additionally, based on the known information provided in the question, we know that in View 2, he is facing to the right. Consequently, we can conclude that in View 4, the man is also facing to the right, making option B the correct choice.|
|---|

|View 1<br><br>[Figure 556]|View 2<br><br>[Figure 557]|
|---|---|
|View 4<br><br>[Figure 558]|View 3<br><br>[Figure 559]|

[Figure 560]

Counting

Here are multiple camera views which are pointing to the

same scene, could you count how many basketball hoops in total are in the scene?

(A) 2 (B) 3 (C) 4

|Evaluator Prediction: (B) Answer: (C) Error Review: In View 1, we can clearly see a man wearing a white shirt standing under a basketball hoop, with another basketball hoop visible behind him. In View 4, we can confirm that there is also a basketball hoop on the left side of the same man’s location. In View 3, in addition to the two previously mentioned hoops, there is another basketball hoop visible in the background, which is easy to overlook. This indicates that there are a total of two basketball courts, each equipped with a pair of basketball hoops. Therefore, option C is the correct choice.|
|---|

- Figure 22. Questions that evaluators answered incorrectly, along with a detailed review of their reasoning for selecting the incorrect options.

[Figure 561]

##### Figure 23. The visualization of all model performance across the 6 task categories in All-Angles Bench.

|Methods|Avg.<br><br>|Attribute Cam.Pose Counting Manipul. Rel.Dir. Rel.Dist.<br><br>Multiple-Choice Answer|
|---|---|---|
|Human Level Closed-source Models<br><br>GPT-4o Gemini-1.5-Pro<br><br>Gemini-1.5-Flash<br><br>Gemini-2.0-Flash<br><br><br>Claude-3.5-Sonnet Claude-3.7-Sonnet<br><br>|82.0 52.4 50.8 50.0 58.4 50.0 52.8<br><br>|93.3 88.9 86.3 72.0 79.5 95.7 66.7 16.7 52.9 40.0 53.8 63.8 60.0 22.2 35.3 44.0 51.3 76.6 55.6 22.2 27.5 50.0 56.4 74.5 62.2 38.9 64.7 48.0 56.4 68.1 57.8 61.1 52.9 22.0 51.3 63.8 60.0 38.9 37.3 38.0 56.4 80.9|
|Open-source Models<br><br>InternVL2.5-2B InternVL2.5-4B InternVL2.5-8B<br><br>InternVL2.5-38B InternVL2.5-78B<br><br>DeepSeek-VL2-Small<br><br>DeepSeek-VL2 Qwen2.5-VL-3B<br><br>Qwen2.5-VL-72B Ovis2-2B Ovis2-4B Ovis2-8B<br><br>Ovis2-16B Ovis2-34B<br><br>Cambrian-8B Cambrian-13B Cambrian-34B<br><br>LLaVA-Onevision-Qwen2-7B LLaVA-Onevision-Qwen2-72B<br><br>LLaVA-Video-Qwen2-7B LLaVA-Video-Qwen2-72B|45.2 47.2 52.4 60.8 54.4 48.0 51.6 52.4 58.4 50.8 54.0 54.4 60.4 59.2 40.4 39.2 47.2 53.6 57.2 46.8 54.0<br><br>|64.4 5.6 41.2 34.0 33.3 68.1 62.2 16.7 39.2 34.0 33.3 78.7 60.0 16.7 54.9 34.0 48.7 78.7 73.3 27.8 70.6 42.0 64.1 68.1 77.8 16.7 52.9 30.0 59.0 70.2 62.2 38.9 45.1 32.0 35.9 68.1 62.2 38.9 51.0 48.0 38.5 61.7 68.9 22.2 43.1 42.0 46.2 74.5 73.3 22.2 52.9 44.0 61.5 76.6 57.8 38.9 43.1 38.0 46.2 74.5 73.3 38.9 51.0 26.0 61.5 68.1 68.9 5.6 47.1 44.0 59.0 74.5 66.7 50.0 58.8 44.0 59.0 78.7 71.1 11.1 52.9 52.0 61.5 78.7 57.8 22.2 33.3 34.0 38.5 46.8 48.9 27.8 31.4 28.0 35.9 57.4 68.9 16.7 43.1 44.0 33.3 57.4 64.4 16.7 43.1 44.0 53.8 78.7 68.9 11.1 43.1 56.0 64.1 74.5 62.2 5.6 43.1 26.0 53.8 68.1 60.0 11.1 43.1 50.0 56.4 78.7|

Table 2. Evaluation results for 27 MLLMs on 250 Q&A tiny benchmark.

Multiple-Choice Answer Extraction Prompt

Given a prediction for a multiple-choice question, directly extract the selected answer while skipping the reasoning process. If the prediction explicitly chooses option A, B, or C, return the corresponding letter. If the prediction does not specify a choice or indicates that none of the options are correct, return None.

Figure 24. The prompt used for extracting multiple-choice answers from predictions.

System Prompt

You are the expert for designing questions and answers for creating a benchmark for multi-camera view scenarios.

Here I will first give you several images from different camera views which are pointing to the same scene. Please answer the following questions based on the specific task description and example provide. Ensure the questions and answers are based on the objects and their relationships visible across the given camera views.

Figure 25. The system prompt used for generating five tasks with the MLLM.

Task Specific Prompt

- 1. Counting: Counting across multi-views Task Description: Could you count the amount on specific object (e.g., people, chair, cup) in the scene based on all the input camera views? Example: Question: Here are multiple camera views which are pointing to the same scene, could you count how many people in total are in the scene? Option: (A) 5, (B) 3, (C) 1
- 2. Attribute Identification: Attribute identification across multiple views Task Description: Could you identify the same object / attribute across multiple camera views? Example: Question: View 1 and View 2 are two different views that represent the same scene. In View 1, there is a person in fencing who is facing to the spot where camera View 1 was positioned. Could you identify the same person in View 2? Option: (A) the person on the left side of View 2, (B) the person closer to the center of View 2, (C) the person sitting on the stairs
- 3. Relative Distance: Object-camera relative distance Task Description: Could you measure the relative distance between the seen object and camera view position in the cross view scenario? Example: Question: View 1 and View 2 are two different camera views that represent the same scene. In which view, the basketball is closer to the spot where the camera view was positioned? Option: (A) closer to the spot where camera View 1 was positioned, (B) closer to the spot where camera View 2 was positioned, (C) distance to the spot where camera View 1 and View 2 were positioned is equal
- 4. Relative Direction: Object-camera relative direction Task Description: Could you measure the relative direction between the seen object and camera view position in the cross view scenario? Example: Question: View 1 and View 2 are two different camera views that represent the same scene. In View 1, the man in red t-shirt is facing to the spot where View 1 was positioned, which direction is the man facing in View 2? Option: (A) facing to the left side of the view, (B) facing to the right side of the view, (C) facing to the spot where camera View 2 was positioned
- 5. Manipulation: Relative object manipulation across views Task Description: Could you manipulate the seen objects (e.g., what if the person is walking to the left of the view), and design the Q&A based on the trajectory of object across views? Example: Question: What if the person wearing the gray t-shirt is walking toward the spot where camera View 1 was positioned, what would be the trajectory of the same person in View 2? Option: (A) walking toward the left side of View 2, (B) walking toward the right side of View 2, (C) remain in the same position

- Figure 26. The task-specific prompts used for generating five tasks with the MLLM.

Camera Pose Estimation Question Template

Question: View 1, View 2, View 3, View 4 are four different input camera views that represent the same scene. Could you reconstruct the camera view layout from the top-bottom view, and answer the following question?

Option: (A) in clockwise order, the layout is View 1, View 2, View 3, View 4, (B) in clockwise order, the layout is View 4, View 3, View 2, View 1, (C) in clockwise order, the layout is View 2, View 3, View 1, View 4

- Figure 27. The question template designed for the Camera Pose Estimation task.

[Task] These images capture the same scene. Your objective is to identify specific objects within each image, understand the spatial arrangement of the scene, and estimate the center point of each object, assuming the entire scene is represented by a 10x10 grid.

[Rule] We provide the primary categories about in this scene: (e.g., human, lego, tv, sofa, stair...) Focus ONLY on these categories (ignore small object or categories). Estimate the center location of each instance within the provided categories, assuming the entire scene is represented by a 10x10 grid. If a category contains multiple instances, include all of them (e.g., Person A, Person B, Person C..). Each object’s estimated location should reflect its real position in the scene while preserving the relative spatial relationships. Combine and merge information from the images since they are pointing to the same scene, calibrating the object locations accordingly.

[Output] Please generate a 10x10 grid visualization that includes: The predicted locations of each object (e.g., humans, basketball, etc.). The camera view position The camera’s viewpoint direction represented as an arrow indicating its view orientation The trajectory of the object in the question (e.g., object movement), indicating its motion path. Please also provide your answer and your reasons step by step in details.

Figure 28. Visualization prompt designed to evaluate MLLMs on orientation-sensitive challenges.

