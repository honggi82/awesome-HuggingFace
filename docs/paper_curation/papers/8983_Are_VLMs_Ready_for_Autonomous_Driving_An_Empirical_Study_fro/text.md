[Figure 1]

### Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability, Data, and Metric Perspectives

Shaoyuan Xie† Lingdong Kong‡,♢,∗ Yuhao Dong‡,§ Chonghao Sima‡,▽ Wenwei Zhang‡ Qi Alfred Chen† Ziwei Liu§ Liang Pan‡,

†University of California, Irvine ‡Shanghai AI Laboratory ♢National University of Singapore §S-Lab, Nanyang Technological University ▽The University of Hong Kong

# arXiv:2501.04003v1[cs.CV]7Jan2025

Code & Demo: drive-bench.github.io Dataset & Benchmark: huggingface.co/datasets/drive-bench/arena

[Figure 2]

[Figure 3]

Perception

Prediction

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

|[Figure 10]<br><br>[Figure 11]<br><br>Text Only<br><br>H.265<br><br>Quant<br><br>Bit Error<br><br>[Figure 12]<br><br>Dri|[Figure 13]<br><br>Bright<br><br>Dark<br><br>Snow<br><br>Fog<br><br>[Figure 14]<br><br>ve|
|---|---|
|[Figure 15]<br><br>Zoom Blur<br><br>Motion Blur<br><br>Frame<br><br>Saturate<br><br>[Figure 16]<br><br>[Figure 17]<br><br>Ben|[Figure 18]<br><br>[Figure 19]<br><br>Rain<br><br>Splash<br><br>Lens Crash<br><br>[Figure 20]<br><br>ch|

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Q: Will the object be in ego moving direction?

Q: What is the moving status of the object? A: It is appear to be a car and is going ahead.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

A: No, it is moving toward another way.

[Figure 29]

[Figure 30]

Q: What action should

[Figure 31]

[Figure 32]

Q: Predict the speed and steering of the ego.

the ego vehicle take ? A: It might keep going

[Figure 33]

[Figure 34]

A: The ego is going straight and driving fast.

###### but at a lower speed.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Behavior

Planning

Figure 1. Overview of DriveBench. Our benchmark evaluates the reliability and visual grounding of Vision-Language Models (VLMs) in autonomous driving across four mainstream driving tasks – perception, prediction, planning, and explanation – under a diverse spectrum of 17 settings (clean, corrupted, and text-only inputs). It includes 19, 200 frames and 20, 498 QA pairs spanning three question types: multiple-choice, open-ended, and visual grounding. By addressing diverse tasks and conditions, we aim to reveal VLM limitations and promote reliable, interpretable autonomous driving.

##### Abstract

through natural language. However, the assumption that VLMs inherently provide visually grounded, reliable, and interpretable explanations for driving remains largely unexamined. To address this gap, we introduce DriveBench, a benchmark dataset designed to evaluate VLM reliability across 17 settings (clean, corrupted, and text-only inputs), encompassing 19,200 frames, 20,498 question-

Recent advancements in Vision-Language Models (VLMs) have sparked interest in their use for autonomous driving, particularly in generating interpretable driving decisions

(∗) Project lead. ( ) Corresponding author.

answer pairs, three question types, four mainstream driving tasks, and a total of 12 popular VLMs. Our findings reveal that VLMs often generate plausible responses derived from general knowledge or textual cues rather than true visual grounding, especially under degraded or missing visual inputs. This behavior, concealed by dataset imbalances and insufficient evaluation metrics, poses significant risks in safety-critical scenarios like autonomous driving. We further observe that VLMs struggle with multi-modal reasoning and display heightened sensitivity to input corruptions, leading to inconsistencies in performance. To address these challenges, we propose refined evaluation metrics that prioritize robust visual grounding and multi-modal understanding. Additionally, we highlight the potential of leveraging VLMs’ awareness of corruptions to enhance their reliability, offering a roadmap for developing more trustworthy and interpretable decision-making systems in realworld autonomous driving contexts. The benchmark toolkit is publicly accessible.

[Figure 43]

What is the moving status of object at (480,520)? Please select the correct answer from the following options: A. Going Ahead B. Turn Left C. Turn Right

Question

[Figure 44]

The object, which appears to be a car, is positioned in the center of the lane and is facing forward, indicating that it is moving straight ahead. The moving status of object is: A. Going Ahead.

[Figure 45]

Clean Image

[Figure 46]

Based on the coordinate (480, 520), which is near the center of the image, the object likely represents a vehicle on the road. the most probable moving status is: A. Going Ahead.

VLM

No Image

[Figure 47]

|Ground Truth: Turn.<br><br>|
|---|

[Figure 48]

Accuracy: 95% Accuracy: 90%

[Figure 49]

|Ground Truth: Going Ahead.|
|---|

Dataset

[Figure 50]

GPT Score: 65 GPT Score: 75

Rate my answer based on the similarity. Here is <answer>, and here is <GT>.

GPT

Figure 2. Do VLMs provide reliable explanations based on visual cues in driving? We study this from perspectives on reliability, data, and metrics. We find VLMs can fabricate quality answers to driving questions when visual information is absent. The fabricated answers can bypass current metrics, even GPT scores, due to imbalance, lack of context dataset, and problematical evaluation protocols. Our observations challenge the passive assumption that VLMs are more reliable than task-specific models in driving decisions [26] because of visual-grounded interpretable responses.

##### 1. Introduction

With recent advancements in Vision-Language Models (VLMs) [1, 5, 12, 13, 42–44, 67], there has been increasing research interest in applying VLMs to autonomous driving applications [20, 48, 58, 59, 62, 68, 69, 74]. These efforts span both the design of end-to-end frameworks [58, 59, 62, 74] and the integration of VLMs to facilitate interpretable interactions and decisions through natural language [28, 48, 53]. Such interpretability is believed to enhance transparency, trustworthiness, and user confidence in autonomous systems [76].

dation to its limits by using text-only prompts. Surprisingly, VLMs demonstrate comparable performance to their outputs under “clean” visual inputs, even when no visual cues are available (as illustrated in Fig. 2). This observation starkly contrasts with human drivers, who would struggle under such degraded conditions. A closer analysis reveals that this apparent “resilience” is often a byproduct of imbalanced datasets and suboptimal evaluation protocols rather than inherent model robustness.

However, previous studies highlight significant limitations in evaluating end-to-end autonomous driving models in open-loop settings [39]. Instead of focusing on trajectory prediction with potentially unreliable open-loop end-to-end VLMs [28, 50, 59, 74], we address another fundamental – yet underexplored – question that has been widely assumed [50, 58, 62, 76]:

Datasets. We perform an in-depth analysis of existing “Driving with Language” benchmarks [9, 29, 55, 59, 72] and identify critical shortcomings, particularly concerning dataset imbalance. Many of these benchmarks, built on popular driving datasets such as nuScenes [55], BDD [78], and Waymo Open [61], inherit limitations from their original designs [39]. For instance, imbalanced data distributions skew evaluations, enabling overly simplistic answers such as Going Ahead” to achieve over 90% accuracy for motion-related queries. Furthermore, the reliance on singleframe questions – often reliant on temporal context – creates challenges even for human annotators. Consequently, these benchmarks exhibit inherent biases and persistent negative samples, which diminish the interpretability and reliability

“Are existing VLMs capable of providing reliable explanations grounded on visual cues for driving?”

To investigate, we examine whether driving decisions generated by VLMs are genuinely grounded in sensory information from the physical environment or reflect general knowledge and fabricated responses from textual cues.

Model Reliability. To address the fundamental question, we examine VLM reliability through an out-ofdistribution (OoD) robustness lens. For this purpose, we introduce DriveBench, a benchmark encompassing four mainstream driving tasks and 15 types of data corruptions, including 19,200 images and 20,498 QA pairs tailored to real-world autonomous driving scenarios. To assess robustness under extreme conditions, we push visual degra-

of evaluation outcomes.

Metrics. We also reevaluate existing metric designs critically. Language-driven interactions in driving applications are often assessed using traditional pattern-matching metrics such as ROUGE [40], BLEU [54], and CIDEr [65], which were originally developed for summarization and translation tasks. However, as noted in [3, 4, 18, 63], these metrics face significant limitations in evaluating nuanced language-based driving decisions. Even modern evaluators like GPT-based scoring [10, 22, 45, 59] provide limited insights without task-specific rubrics. These constraints underscore the need for metrics that effectively capture reasoning, contextual understanding, and safety-critical aspects. We advocate for the development of advanced evaluation metrics that incorporate task-specific rubrics, structured question formats, and contextual driving information to more accurately assess VLMs in the real world.

Through a series of comprehensive experiments, we derive several key insights from our analysis, spanning 17 settings (i.e., clean, text-only, and various corrupted inputs), 12 VLMs (including both open-sourced and commercial models), 5 tasks (perception, prediction, planning, behavior, and corruption identification), and 3 evaluation metrics (accuracy scores, traditional language metrics [40, 54], and GPT scores). These findings shed light on the current challenges in integrating VLMs into driving scenarios:

- 1 Fabricated responses under degradation: VLMs often produce plausible yet fabricated responses under degraded visual conditions, including scenarios where no visual cues exist. This raises concerns about their reliability and trustworthiness, as such behaviors are difficult to detect using existing datasets and evaluation protocols.

- 2 Awareness of visual corruptions: While VLMs exhibit some awareness of visual corruptions, they only explicitly acknowledge these issues when directly prompted. This highlights the models’ limited capacity to autonomously assess the reliability of visual inputs and provide scenariospecific, safety-focused responses.

- 3 Impact of dataset biases: Highly biased datasets and suboptimal evaluation protocols can create misleading perceptions of VLM performance. In many cases, VLMs rely on general knowledge rather than actual visual cues to generate responses, which can unexpectedly achieve high scores with existing metrics.

- 4 Need for tailored metrics: Current evaluation metrics, including traditional language-based metrics [40, 54] and GPT scores [10, 59], fail to capture the nuanced requirements of autonomous driving tasks. There is an urgent need for the development of specialized metrics that account for reasoning, contextual understanding, and safety-critical aspects to evaluate VLMs more effectively.

Our findings through DriveBench not only highlight the need for improved datasets and evaluation protocols but also

pave the way for developing safer, more interpretable VLMs for real-world autonomous systems.

##### 2. Related Work

Driving with Language. VLMs [1, 5, 42–44, 67] have demonstrated remarkable human-level reasoning and understanding across diverse domains [6, 8, 11, 15, 17, 25, 41, 47, 60, 62, 73, 75]. This capability has raised the prospect of utilizing VLMs to manage complex and unpredictable scenarios in autonomous driving [76]. Additionally, the language-based interaction that VLMs offer can help mitigate the black-box nature of deep neural networks by providing explanatory feedback that accompanies their decisions. Driven by these advantages, a growing body of research has begun investigating the deployment of VLMs in autonomous driving [20, 48, 59, 62, 69, 74]. Early works [20, 69] leveraged LLMs for decision-making in simplified driving simulators (e.g., HighwayEnv [34]) by offering context-driven descriptions. More recent advancements in VLM architectures [48, 59, 62, 74] enable these models to interact directly with environments through multimodal (visual and language) inputs. However, despite these advancements, the robustness and reliability of VLMs in complex, real-world autonomous driving tasks remain largely untested, especially given that reliable performance across diverse driving situations is a fundamental requirement for their application in autonomous driving.

Datasets & Metrics. To support the integration of VLMs in autonomous driving, several multimodal datasets have been proposed [16, 29, 55, 59, 70, 72]. These datasets typically augment established driving benchmarks, such as BDD [78] and nuScenes [55], with language-based annotations that enable language-driven perception and decisionmaking. Certain datasets, such as DriveLM [59], incorporate advanced structures like graphs [55, 59] to further support the reasoning capabilities of VLMs, thus offering a richer context for nuanced decision-making. DriveLM [59] is notable for its extensive, human-annotated language data based on the nuScenes dataset, covering the full range of autonomous driving tasks, including perception, prediction, planning, and control. Nonetheless, despite these advances, current datasets and metrics may still lack the comprehensive scope needed to capture the full spectrum of real-world driving complexities, particularly in evaluating open-ended questions generated by VLMs, where responses require a detailed understanding of diverse scenarios.

VLM Reliability. Deep neural networks have historically struggled with out-of-distribution (OoD) data, a limitation of particular concern in autonomous driving, where failing to handle rare or unexpected scenarios could result in severe safety risks [30, 31, 71]. Large-scale models such as CLIP [56], trained on extensive internet-sourced datasets, have shown enhanced robustness to such challenging corner

a

b

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

FastFast Normal

FastFast Normal

Very

Very

[Figure 56]

175

11 35 409 41 14

2 5 119 13 0

800

Pred: Going Ahead

Pred: Going Ahead

700

150

GT Action: Turn Left

GT Action: Turn Right

[Figure 57]

[Figure 58]

21 89 855 129 37

1 17 189 15 6

[Figure 59]

[Figure 60]

600

125

| |
|---|

SpeedSlow Not

SpeedSlow Not

500

100

58 61 489 84 69

10 15 78 27 9

400

75

300

40 60 579 50 62

9 14 107 18 10

50

200

c

d

[Figure 61]

[Figure 62]

Moving

Moving

[Figure 63]

[Figure 64]

25

0 0 879 0 0

0 0 135 0 0

Pred: Going Ahead

Pred: Turn Left

100

0

0

GT Action: Turn Right

GT Action: Turn Right

[Figure 65]

[Figure 66]

LeftSlightlyLeftGoingStraightSlightlyRightRight

LeftSlightlyLeftGoingStraightSlightlyRightRight

[Figure 67]

[Figure 68]

| |
|---|

(a) Training Set

(b) Validation Set

- Figure 3. The behavior distributions of steering and speed in DriveLM-nuScenes [59]. The majority actions of vehicle behaviors are “Going Ahead”, which has also been noted in [39].

|[Figure 69]| |[Figure 70]| | | |93|.33| | |
|---|---|---|---|---|---|---|---|---|---|
|27|.55|[Figure 71]| | | | | | | |
|[Figure 72]| | | | | |9|5.5|6| |
| | | | |[Figure 73]<br><br>64|.91| | | | |
|[Figure 74]| | | | | |90|.72| | |

0 25 50 75 100

Human Random Clean

No Pix. No Fea.

Acc (%)

(a) Perception Task

| |[Figure 75]|[Figure 76]| |6|9.51| | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
|23.|[Figure 77]<br><br>40| | | | | | | | | |
| | | | |[Figure 78]<br><br>8.8|2| | | | | |
|[Figure 79]<br><br>28|.16| | | | | | | | | |
|[Figure 80]|5.9|1| | | | | | | | |

0 25 50 75 100

Human Random Clean

No Pix. No Fea.

Acc (%)

(b) Behavior Task

- Figure 4. The accuracy scores of perception and behavior tasks under different visual inputs. The results are from DriveLMAgent [59]. No Pix. and No Fea. denote zero image pixel and zero feature, respectively.

Figure 5. Challenging cases in existing dataset. The results are from GPT4-o [2]. (a): The black sedan is turning left, indicated by the turn lights. (b): The black sedan is turning right. The model predicts both Going Ahead. The examples show challenging cases for Turn choice, where the visual cues are too subtle or rely on temporal context for correct predictions. (c) and (d) are both Turning Right, but the model fails to locate the objects based on center pixel positions due to the existence of overlapping or occlusion. Zoomed-in for more details.

for driving with languages, considering the impact that the dataset serves as one of the benchmarks in Foundation Models for Autonomous Systems Workshop [14, 52]. The dataset spans five tasks, including perception, prediction, planning, behavior, and control. For each task, different sets of questions are applied, such as multiple-choice questions (MCQs), and visual question answering (VQA). The comparison between our dataset and related benchmarks can be seen in Tab. 1.

[Figure 81]

[Figure 82]

cases [19, 51], which suggests that VLMs, trained on vast, diverse datasets – may have the inherent common-sense reasoning capabilities to address these challenges better than traditional, task-specific models [35, 76]. However, this hypothesis remains under-investigated in two critical areas: 1) the reliability of VLMs to maintain accurate reasoning when exposed to visual corruptions, and 2) their capacity to detect and interpret potential visual anomalies that could impact safe vehicle maneuvering. In this work, we provide a systematic evaluation of the reliability of current VLMs under conditions of visual corruption, identifying potential limitations that impact their applicability in real-world driving contexts.

Distribution Bias. Through detailed examination, we identify a significant distribution bias in the dataset, which is naturally inherited from the nuScenes dataset [7, 39]. Specifically, in behavior-based multiple-choice questions (MCQs) that inquire about the future movement of the ego vehicles, approximately 78.6% of responses are labeled as Going Straight as shown in Fig. 3. Consequently, randomly selecting Going Straight as an answer can yield accuracy levels exceeding 70%, which is concerning since the fine-tuning processes [59] further encourage the model to memorize majority choices. To address this imbalance, in DriveBench, we carefully re-sampled the data to create a more balanced distribution among different options.

Challenging Cases. Furthermore, we evaluate the advanced GPT-4o [2] and analyze the failure cases, as illustrated in Fig. 5. We find that annotations such as Turn Left or Turn Right are factually accurate but often require temporal context or subtle indicators (e.g., turn signal lights) to be correctly interpreted. Additionally, vehicles overlap with each other in some cases, making the pixel position-based question too nuanced for current VLMs to distinguish (e.g., What’s the object at (540, 600) and object at (530, 610), respectively). As a result, existing VLMs demonstrate con-

##### 3. DriveBench: Driving with VLMs

In this section, we detail the construction of our benchmark designed to assess the reliability of VLMs within the domain of autonomous driving.

###### 3.1. Datasets

In this section, we start building our DriveBench with representative driving with language datasets [59]. We choose DriveLM [59] as it is one of the most representative datasets

- Table 1. Comparisons among evaluation benchmarks for driving. “Per.”, “Pre.”, “Beh.”, “Pla.”, “Rob.” refer to the Perception, Prediction, Behavior, Planning, and Robustness tasks, respectively. GPTctx represents GPT scores augmented with context information.

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

# Frames # QA Pairs

Benchmark

Logic Evaluation Metrics

Per. Pre. Beh. Pla. Rob. (Test) (Test)

BDD-X [29] ✓ ✗ ✗ ✗ ✗ - - None Language BDD-OIA [72] ✓ ✗ ✓ ✗ ✗ - - None F1 Score

nuScenes-QA [55] ✓ ✗ ✗ ✗ ✗ 36,114 83,337 None Acc Talk2Car [16] ✓ ✗ ✗ ✓ ✗ ∼ 1.8k 2,447 None Acc

nuPrompt [70] ✓ ✗ ✗ ✗ ✗ ∼ 36k ∼ 6k None AMOTA DRAMA [49] ✓ ✗ ✗ ✓ ✗ - ∼ 14k Chain Language Rank2Tel [57] ✓ ✗ ✗ ✓ ✗ - - Chain Acc, Language

DirveMLLM [23] ✓ ✗ ✗ ✗ ✗ 880 - None Acc DriveVLM [62] ✓ ✗ ✓ ✓ ✗ - - None GPTctx DriveLM [59] ✓ ✓ ✓ ✓ ✗ 4,794 15,480 Graph Language, GPT

###### DriveBench ✓ ✓ ✓ ✓ ✓ 19,200 20,498 Graph Acc, Language, GPT, GPTctx

[Figure 88]

- Figure 6. The word cloud collected from the QA pairs in the proposed benchmark, highlighting the main focus on different autonomous driving tasks in DriveBench. The larger the font size, the higher the frequency of occurrence.

siderable difficulty in accurately interpreting cues that rely on temporal information or subtle visual indicators, leading to a disproportionate number of negative outcomes in evaluations. To prevent such samples from skewing our findings, we eliminate instances highly dependent on temporal context or present significant interpretive challenges for current VLMs. Our data selection strategy prioritizes instances that GPT-4o [2] can correctly interpret, thereby indicating the availability of sufficient visual cues for single-framebased predictions. We also analyze other failure cases of GPT-4o [2], as shown in Fig. 7. In these examples, humans can answer with correct answers while GPT-4o [2] heavily relies on the object relation position to the frames for decision-making. Thus, we keep those examples in our dataset.

###### 3.2. Driving Tasks

Our DriveBench covers four mainstream driving tasks, including perception, prediction, planning, and

[Figure 89]

[Figure 90]

[Figure 91]

behavior, examples are shown in Fig. 1.

[Figure 92]

The perception task focuses on identifying the moving status of targeted objects and analyzing the surrounding environments. The prediction task focuses on predicting the future movement of these objects. The planning task focuses on suggesting actions to navigate the vehicle safely and efficiently in complex driving scenarios. Finally, the behavior task is designed to predict the future steering and speed of the ego vehicles. By addressing these tasks, the dataset ensures comprehensive coverage of the critical aspects required for evaluating the capabilities of driving VLMs in handling complex and dynamic scenarios.

###### 3.3. Corruption Data

Autonomous driving is an application where numerous types of corruption can exist [24, 30–32, 37, 71]. Previous works find that task-specific models are inherently vulnerable to OoD corruptions, which prohibit precise perception of surroundings. On the other hand, large models [19, 51] are shown to be resilient toward OoD corruption, given their vast amount of training data [56]. Therefore, given the safety-critical applications and promising robustness of large models, one natural question is, how reliable are existing VLMs towards visual corruption in driving?

Our DriveBench crafts a total of 15 different corruption types (cf. Fig. 1), spanning across weather conditions (1Brightness, 2Dark, 3Fog, 4Snow, and 5Rain), external disturbances (6Water Splash and 7Lens Obstacle), sensor failures (8Camera Crash, 9Frame Lost, and 10Saturate), motion blurs (11Motion Blur and 12Zoom Blur), and data transmission errors (13Bit Error, 14Color Quant,

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

#### a

#### GT Action: Going Ahead b GT Action: Turn Right c GT Action: Turn Right

The object at (0.5739, 0.6361) is a pedestrian. Based on the image, the pedestrian is moving towards the left

The object at (0.3231, 0.6787) is a car that appears to be turning left. Therefore, the correct answer is: C. Turn left.

###### The moving status of object <c1,CAM_FRONT,0.1906,0.6139> is: C. Turn left.

side of the frame. Therefore, the correct answer is: B. Turn left.

[Figure 101]

[Figure 102]

[Figure 103]

Explanation: The pedestrian's orientation and movement direction suggest turning left relative to the camera's perspective.

Explanation: The car is positioned at an angle suggesting turning left, as indicated by its orientation relative to road markings.

Explanation: The object in question is positioned at the intersection and is oriented towards the left, indicating a left turn.

[Figure 104]

[Figure 105]

[Figure 106]

#### d GT Action: Turn Right

#### e GT Action: Turn Right

#### f GT Action: Turn Left

The object at coordinates (0.5466, 0.7209) appears to be a car. Based on

The object at coordinates (0.4245, 0.6148) appears to be a pedestrian moving across the road. Therefore, the correct answer is: C. Going ahead.

###### The moving status of object <c1,CAM_BACK,0.6958,0.6241> is: C. Turn right.

the image, the car is facing towards the left. Therefore, the correct answer is: B. Turn left.

[Figure 107]

[Figure 108]

[Figure 109]

Explanation: The car is positioned with its front facing slightly towards left, suggesting it is in the process of turning left.

Explanation: The pedestrian is walking straight across the road, indicating forward movement.

Explanation: The red car is positioned in a way that indicates it is turning right, based on its orientation and the road markings.

- Figure 7. GPT-4o failure cases. (a): GPT-4o reasons the moving status of the pedestrian by the moving position related to the frame, instead of the coordinate of the moving object itself, thus leading to wrong perception results. (b): The model struggles to distinguish the correct direction based on the coordinate of the target object. (c): GPT-4o reasons the moving status of the SUV by the relative location of the object to the current frame causes the wrong perception results. (d): GPT-4o fails to perceive the orientation of the car. (e): The dataset contains examples that need multiple frames to reason successfully, GPT-4o fails to address these examples with a single image input. (f): GPT-4o reasons the moving status of the SUV by the relative location of the object to the current frame causes the wrong perception results.

and 15H.265 Compression). The corruption encompasses a range of potential OoD scenarios the vehicles might encounter. From a reliability perspective, These corruptions are the key to our evaluation and insights into VLMs’ visual-grounded driving capabilities. For more detailed definitions and examples, please refer to Appendix A.1.

as language-based driving decisions are naturally linked to physical and context-specific real-world scenarios.

###### 3.4. Vision-Language Models (VLMs)

Recent advancements in VLMs applied to autonomous driving mainly include: 1) leveraging VLMs to generate and interpret driving decisions in high-level language [28, 48, 62], and 2) employing VLMs for end-to-end autonomous driving in low-level commands in open-loop [28, 50, 59, 62, 68, 74] or closed-loop [58] settings. A fundamental motivation and underlying assumption across this domain is that VLMs can generate interpretable and explainable re-

It should be noted that while existing research has explored VLM hallucinations and trustworthiness [27, 36, 64, 66], it has not yet been rigorously examined within the context of driving applications. Autonomous driving requires approaching the reliability of VLMs from different aspects,

- Table 2. Evaluations of VLMs across different driving tasks (perception, prediction, planning, and behavior). “Clean” represents clean image inputs. “Corr.” represents corruption image inputs, averaged across fifteen corruptions. “T.O.” represents text-only evaluation. For humans, we only evaluate MCQ questions in perception and behavior tasks. The evaluations are based on GPT scores, where we tailored detailed rubrics for each task and question type. We highlight scores higher than clean performance under corruption.

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Perception Prediction Planning Behavior

Method Size Type

Clean Corr. T.O. Clean Corr. T.O. Clean Corr. T.O. Clean Corr. T.O. Human - - 47.67 38.32 - - - - - - - 69.51 54.09 -

[Figure 114]

GPT-4o [2] - Commercial 35.37 35.25 36.48 51.30 49.94 49.05 75.75 75.36 73.21 45.40 44.33 50.03

LLaVA-1.5 [43] 7 B Open 23.22 22.95 22.31 22.02 17.54 14.64 29.15 31.51 32.45 13.60 13.62 14.91 LLaVA-1.5 [43] 13 B Open 23.35 23.37 22.37 36.98 37.78 23.98 34.26 34.99 38.85 32.99 32.43 32.79

LLaVA-NeXT [44] 7 B Open 24.15 19.62 13.86 35.07 35.89 28.36 45.27 44.36 27.58 48.16 39.44 11.92 InternVL2 [12] 8 B Open 32.36 32.68 33.60 45.52 37.93 48.89 53.27 55.25 34.56 54.58 40.78 20.14

Phi-3 [1] 4.2 B Open 22.88 23.93 28.26 40.11 37.27 22.61 60.03 61.31 46.88 45.20 44.57 28.22 Phi-3.5 [1] 4.2 B Open 27.52 27.51 28.26 45.13 38.21 4.92 31.91 28.36 46.30 37.89 49.13 39.16

Oryx [46] 7 B Open 17.02 15.97 18.47 48.13 46.63 12.77 53.57 55.76 48.26 33.92 33.81 23.94 Qwen2-VL [67] 7 B Open 28.99 27.85 35.16 37.89 39.55 37.77 57.04 54.78 41.66 49.07 47.68 54.48 Qwen2-VL [67] 72 B Open 30.13 26.92 17.70 49.35 43.49 5.57 61.30 63.07 53.35 51.26 49.78 39.46

DriveLM [59] 7 B Specialist 16.85 16.00 8.75 44.33 39.71 4.70 68.71 67.60 65.24 42.78 40.37 27.83 Dolphins [48] 7 B Specialist 9.59 10.84 11.01 32.66 29.88 39.98 52.91 53.77 60.98 8.81 8.25 11.92

Models

PerformanceChange

Human GPT-4o LLaVA1.5-7B

0.0

0.1

- LLaVA1.5-13B

- LLaVA1.6-7B

0.2

InternVL2-8B

0.3

Phi3

0.4

Phi-3.5

Weather External Sensor Motion Transmission

Qwen2-7B

Qwen2-72B

Figure 8. Illustration of performance degradation. After applying each corruption, we evaluate the MCQs accuracy changes compared with clean inputs. We observe that human performance largely decreases while most VLMs remain unchanged.

sponses, thereby reducing the opaque “black-box” nature inherent in traditional task-specific models [26, 38].

To encompass the full scope of existing advanced VLMs, the current version of DriveBench evaluates a diverse set of 12 popular VLMs, including both commercial and opensource models, as well as models fine-tuned specifically for autonomous driving applications [48, 59]. This selection reflects the latest developments in state-of-the-art VLMs for driving. To ensure consistency, we apply a standardized system prompt across all models (further details are provided in the Appendix C.2). The prompt explicitly instructs the VLMs to generate auxiliary explanations, enabling GPT-based evaluation of MCQs, which only have a selection alone by default, as detailed in the next section.

###### 3.5. Evaluation Metrics

In DriveBench, we consider several metrics following [59], including accuracy, BLEU [54], ROUGE-L [40], and GPT score [10, 59]. For MCQs, we utilize both accuracy, as the most direct measure, and GPT scores to capture nuances

in the explanatory quality beyond simple answer selection. We employ GPT-3.5-turbo for GPT Score evaluation. To better capture subtleties between responses, we prompt the model with detailed rubrics that account for answer correctness, coherence, and the alignment of explanations with the final answer. Rubrics are adapted for each specific task and question type to better reflect human-preferred responses. Detailed information on the evaluation prompts and rubrics can be found in Appendix C.3.

##### 4. Experiments

We conduct extensive benchmark experiments and analyses in DriveBench, with detailed discussions leading to our observations and conclusions by step.

###### 4.1. Experimental Setups

We set the temperature to 0.2 and top-p to 0.2, with a maximum output token limit of 512. For DriveLMAgent [59], we adhere to the configurations outlined in [14]. Specifically, we utilize LLaMA-Adapter-V2 [21] as the base model, fine-tuned on the DriveLM-nuScenes dataset. The fine-tuning process is conducted on eight A800 GPUs with a batch size of 4, over 4 epochs. For other open-source models, we download the official model weight from HuggingFace and inference using the vLLM [33] framework. More details about the used model weight can be found in Appendix F. For GPT-4o, we query the official APIs from OpenAI with the same configuration mentioned above.

###### 4.2. Observations & Discussions

###### 4.2.1. Corruption Resilience

The primary results, evaluated using GPT, are summarized in Tab. 2 and Fig. 9. These findings reveal that, even in the presence of image corruption, the model performance remains largely unaffected, demonstrating notable resilience to such perturbations. Specifically, as illustrated in Fig. 9, the performance trends align closely with those observed in other benchmarks. For instance, GPT-4o and Qwen2VL72B consistently achieve state-of-the-art results. Furthermore, when evaluating the resilience against corruption, most vision-language models (VLMs) maintain comparable performance to that observed with clean image inputs, even in open-ended visual question answering (VQA) tasks. To understand the source of this resilience, we investigate whether it stems from the inherent robustness of these VLMs, due to their pre-training on extensive webscale datasets, or if other factors contribute to this phenomenon.

Human Evaluations. To further validate that the applied corruptions indeed impact the driving scenario and to explore the performance gap between humans and VLMs, we conduct a human evaluation. Specifically, we sub-sample the dataset and design a user interface to facilitate human performance assessment (more details in Appendix C.4).

The results are shown in Tab. 2. Since human responses include only a choice without a time-intensive written explanation, GPT scores, which we design to reward detailed explanations, may introduce unfairness in comparison. Therefore, we also report accuracy degradation, as shown in Fig. 8. Interestingly, we observe a significant accuracy drop for human participants under corrupted conditions, whereas most VLMs exhibit subtle performance variations across different corruption types. We next explore if it is due to model robustness [19] or other possible reasons. Text-Only Prompts. Given the above results, we further investigate the effects of extreme corruption by providing VLMs with fully black images, reducing the input to text-only prompts with no visual information. The results, shown in Tab. 2, reveals an intriguing pattern: GPT scores for text-only prompts are closely aligned with those obtained with clean image inputs. This trend persists across different tasks and models, suggesting that the phenomenon is not solely due to model robustness.

Considering the GPT score also takes the quality of explanations into account in addition to answer correctness, we further analyze accuracy on MCQs to isolate the potential scoring advantages due to explanations (discussed in Sec. 3.4). Given that MCQs in perception tasks have three answer choices and MCQs in behavior tasks have four, the expected random-guessing accuracy would be approximately 33% and 25%, respectively. We report results only

InternVL DriveLM

Phi-3.5 Qwen2-VL72B

LLaVA-1.513B Dolphin

[Figure 115]

GPT4-o Fog

[Figure 116]

Rain

Snow

[Figure 117]

[Figure 118]

[Figure 119]

Lens Obstacle

Dark

[Figure 120]

[Figure 121]

Water Splash

[Figure 122]

Bright

[Figure 123]

Clean (Base)

[Figure 124]

Crash

[Figure 125]

[Figure 126]

Frame Lost

H.265

Compress Color

[Figure 127]

[Figure 128]

Saturate

Quant Bit Error

[Figure 129]

[Figure 130]

[Figure 131]

Motion Blur

Zoom Blur

Figure 9. Radar chart comparisons among different models. The performance for each input corruption type is averaged across all the 1, 261 questions spanning four different driving tasks. The evaluation metric used here is the GPT score.

for models that exceed this random baseline on clean inputs, as shown in Tab. 3. Interestingly, a significant portion of the models show minimal or no accuracy degradation, even in the absence of visual cues. The results are even more concerning when considering open-ended questions (i.e., prediction and planning). For instance, responses from the state-of-the-art GPT-4o [2] under text-only conditions maintain approximately 95% of the performance seen with clean image prompts.

Upon further examination shown in Fig. 10, we observe that the high performance of VLMs under text-only conditions is likely influenced by the extensive general knowledge acquired during training. For instance, the models can guess the moving status of one surrounding object based on text cues referring to which camera it has been seen and the corresponding position in that image (examples can be found in Fig. 2). These observations yield two key insights:

- • VLMs are capable of producing plausible responses to driving-related questions based solely on natural language prompts. This capability is likely attributed to the extensive general knowledge and common sense reasoning capabilities developed during their pre-training.
- • The current evaluation protocols for assessing VLMs in autonomous driving reveal significant shortcomings. Even advanced evaluation methods, such as those leveraging GPT-based scoring, fail to capture the nuances of the reliability of VLMs’ responses.

To investigate the first insight further, we pose the ques-

- Table 3. Comparisons of accuracy scores between “clean” and fully “black” (no image) inputs. We observe a large portion of models have no clear performance degradation even when the visual information is absent, suggesting the VLMs response might mainly be based on majority biases (e.g., Going Ahead, in most driving scenarios), instead of leveraging visual cues from sensors.

Task Image Human GPT-4o [2] LLaVA-NeXT [44] LLaVA-1.513B [43] Phi-3 [1] Phi-3.5 [1] Qwen2-VL7B [67] Qwen2-VL72B [67] Perception

Clean 93.3 59.0 55.0 50.0 54.5 56.5 59.0 60.0

No Image - 59.5 ↑ 0.5 34.5 ↓ 20.5 50.0 ↓ 0.0 17.5 ↓ 37.0 58.5 ↑ 2.0 56.5 ↓ 2.5 23.5 ↓ 36.5 Behavior

Clean 69.5 25.5 33.5 32.5 26.5 36.5 30.0 23.0 No Image - 24.0 ↓ 1.5 24.0 ↓ 9.5 33.0 ↑ 0.5 30.0 ↑ 3.5 40.0 ↑ 3.5 23.0 ↓ 7.0 36.5 ↑ 13.5

- Table 4. Comparisons of accuracy changes before and after prompting VLMs with explicit corruption context. We notice a clear trend of performance degradation after mentioning the corruption type in the question. The results suggest VLMs are aware of the current corruption and acknowledge they can not respond to the severely degraded visual information when explicitly prompted.

Method Bright Dark Snow Fog Rain Lens Water Cam Frame Saturate Motion Zoom Bit Quant H.265

GPT-4o −8.69 −12.98 −8.25 −9.00 −6.00 −3.81 −5.82 −12.94 −10.99 −8.52 −6.98 0.57 −8.22 −4.79 −14.30

LLaVA-1.57B 0.26 1.04 0.25 0.00 0.00 1.40 2.60 −2.79 −8.97 0.51 −0.52 2.57 2.22 −1.32 −2.66 LLaVA-1.513B 0.26 1.04 0.25 0.00 0.00 1.96 2.60 −1.27 −0.26 0.51 1.04 2.57 2.22 −0.26 −2.07 LLaVA-NeXT −5.83 −20.63 −31.95 −14.00 −18.50 −31.39 −36.97 −6.13 −18.29 −17.67 −24.85 −33.29 −19.50 5.89 −21.19

InternVL8B −7.24 −8.92 −10.74 −9.50 −7.50 −7.54 −6.24 −17.51 −0.23 −2.46 −2.35 −7.00 −6.67 −7.71 −4.65 Phi-3.5 −9.78 −7.48 −7.75 −9.00 −8.50 −8.60 −7.48 −16.37 −9.31 −9.50 −8.48 −8.07 −6.94 −11.29 −11.16 Phi-3 -4.22 8.67 0.75 −5.00 −10.00 −11.31 −33.22 3.03 8.29 −8.51 −5.42 3.57 17.89 −18.81 -13.12

Qwen2-VL7B −9.74 −7.96 −9.75 −9.50 −9.00 −5.93 −6.98 −20.94 −29.85 −8.49 −8.46 −3.00 −5.06 −9.38 −11.07 Qwen2-VL72B −6.70 −8.96 −8.25 −9.50 −11.00 −8.04 −6.90 7.19 11.01 −10.51 −7.44 −2.93 −6.61 −9.29 −13.07

Cam Front Left

Cam Front

Cam Front Right

0.40

ROUGE-L BLEU-4

Models

| |
|---|

| |
|---|

| |
|---|

GPT-4o LLaVA1.5-7B

| |
|---|

0.35

| |
|---|

0.5

- LLaVA1.5-13B

- LLaVA1.6-7B

0.30

InternVL2-8B

ROUGE-LScore

BLEU-4Score

Phi3

0.4

0.25

Phi-3.5

Qwen2-7B

Cam Back Left

Cam Back

Cam Back Right

0.20

Qwen2-72B

| |
|---|

| |
|---|

|Going ahead<br><br>Turn left<br><br>Turn right|
|---|

0.3

DriveLM

0.15

Dolphin

0.10

0.2

0.05

25 30 35 40 45 50

Figure 10. Prediction spatial distribution of Qwen2-VL7B [67] under text-only prompts. The model can potentially “guess” the MCQ answers without visual information by leveraging text cues, e.g., camera and coordinate positions in the questions.

GPT Score

Figure 11. Evaluation results when using different metrics. The language metrics, such as ROUGE-L [40] and BLEU-4 [54], exhibit high consistency; while the GPT score demonstrates noticeable gaps. We also observe that fine-tuned process benefits DriveLM [21, 59] significantly in regulating its response format, thus leading to misleading high performance under language metrics.

tion: “Are driving VLMs aware of the underlying corruptions in images when they fabricate their answers?” This question serves as the basis for the expanded analysis presented below.

VLMs are explicitly prompted with references to visual corruption. As shown in Tab. 4, the results demonstrate a notable trend of decreasing accuracy across various models and corruption types. Certain models exhibit substantial performance declines in the presence of corruption prompts; for example, LLaVA-NeXT7B [44] experiences an accuracy reduction of approximately 19.62%. A closer examination of model responses reveals increased uncertainty when the corruption context is included in the prompt. For instance, the model may respond with statements such as “based on the image provided, it is not possible to accurately de-

###### 4.2.2. Corruption Awareness

We explore whether the fabricated “reasonable” answer of VLMs under corruption might stem from a lack of awareness regarding potential visual corruptions. To investigate this, we conduct two experiments: E-1) involves explicit corruption reference when prompting the model, e.g., “what are the important objects in the snowy day”, and E2) we directly ask the model to identify the current type of image corruption, e.g., “what is the current corruption”.

In E-1, we analyze changes in MCQ accuracy when

- Table 5. Evaluations on corruption awareness. “MCQ” represents questions that directly ask about the current corruption types. “VQA” represents questions in perception, prediction, and planning but augmented with explicit corruption context information, averaged across three driving tasks. “CAP” represents captioning questions that ask detailed descriptions at both object-level and corruption-level.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Weather External Sensor Motion Transmission

Method

MCQ VQA CAP MCQ VQA CAP MCQ VQA CAP MCQ VQA CAP MCQ VQA CAP GPT-4o [2] 57.20 57.28 54.90 29.25 56.60 61.98 44.25 54.95 56.53 34.25 59.20 56.25 36.83 53.95 57.57

LLaVA-1.57B [43] 69.70 35.49 35.91 26.50 29.17 34.95 18.83 30.64 33.15 71.25 33.43 35.18 10.17 27.28 34.38 LLaVA-1.513B [43] 61.60 39.76 37.76 15.50 34.55 37.83 24.08 35.48 36.08 79.75 36.46 36.42 15.50 32.53 34.33 LLaVA-NeXT [44] 69.70 36.96 48.52 48.50 30.32 57.18 21.83 30.40 44.37 66.00 34.20 50.44 11.83 29.43 53.50

InternVL2 [12] 59.90 48.72 48.60 50.75 47.74 57.82 29.92 45.06 51.14 68.25 49.51 49.67 30.00 43.42 54.24

Phi-3 [1] 40.00 40.59 45.61 25.00 31.44 45.99 16.83 35.58 43.71 31.25 42.92 48.43 27.67 33.04 41.35 Phi-3.5 [1] 60.60 41.82 45.97 21.25 36.89 30.95 25.58 34.66 39.30 33.00 46.03 49.33 39.67 33.47 39.67

Oryx [46] 53.20 40.43 48.95 45.00 40.68 56.06 50.50 36.71 48.55 72.50 40.01 48.33 39.67 36.98 49.87 Qwen2-VL7B [67] 76.70 49.33 45.12 37.50 47.62 51.24 22.83 39.45 47.23 57.00 47.40 47.74 35.83 42.31 48.60

Qwen2-VL72B [67] 59.80 51.05 48.55 45.50 50.57 57.25 52.25 45.89 48.59 58.25 50.85 47.88 44.83 46.23 50.50 DriveLM [59] 21.20 42.86 20.04 21.25 37.49 21.92 9.00 36.68 15.56 22.25 42.05 17.07 17.50 39.56 10.37 Dolphins [48] 54.30 30.21 31.08 3.00 30.42 29.38 9.42 26.83 26.30 9.25 29.82 28.05 21.50 28.86 27.65

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.6

0.5

0.4

ROUGE-L

0.3

0.2

0.1

0.0

0.0 0.1 0.2 0.3 0.4 0.5

GPT Score

(a) Open-Ended Questions

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.6

0.5

0.4

Accuracy

0.3

0.2

0.1

0.0

0.0 0.1 0.2 0.3 0.4 0.5

GPT score

(b) Multiple Choice Questions

- Figure 12. Correlations when using different metrics. We study how well accuracy or ROUGE-L [40] matches the GPT scores for open-ended questions and multiple-choice questions (MCQs), respectively. We find that ROUGE-L [40] fails to reflect semantic information (e.g., key object) that is critical in driving. Contrarily, accuracy aligns well with the GPT score for MCQ while the GPT score can further capture nuanced differences in explanation when the answer is correct.

termine the moving status of the object (480, 520) given the camera crash corruption”. These findings suggest that some models exhibit a degree of corruption awareness when explicitly prompted, recognizing potential unreliability in their responses under conditions of severe visual degradation. The differences in responses between prompts with and without explicit corruption references lead to two significant insights:

- • VLMs demonstrate a limited capacity to identify visual corruption when not explicitly prompted.
- • In the absence of corruption-specific prompts, VLMs frequently rely on general knowledge rather than on the degraded visual information in the current driving scene, often generating responses that feign an understanding of the visual context.

These insights raise critical concerns, as reliable au-

tonomous driving depends on accurate situational awareness and precise interpretation of environmental cues. The observed behavior suggests that current VLMs may lack the capacity for actively corruption-aware reasoning, defaulting instead to generalized responses and common sense knowledge (e.g., “going ahead”, “maintain a safe distance”, “be cautious”, etc.). This tendency can be considered a form of hallucination [27], which undermines the visual grounding reliability of VLMs within the safety-critical context of autonomous driving.

Conversely, models such as LLaVA-1.5 [43] with 7B and 13B parameters exhibit minimal performance changes even when corruption-specific prompts are provided. This observation, when combined with the previous findings, suggests two possible explanations: 1) these models may lack the capability to detect image corruption, or 2) while aware of the corruption, their responses remain dominated by general knowledge rather than visual cues, even in clean situations, thus lead to unchanged performance.

To investigate the first hypothesis, we conduct E-2, in which we explicitly prompt the VLMs to identify the type of visual corruption or determine the number of corrupted cameras in scenarios involving camera crash and frame lost corruptions [71]. The results, summarized under the MCQ columns in Tab. 5, indicate that LLaVA-1.5 [43] achieves high accuracy in identifying corruption types, particularly in weather and motion corruption scenarios, suggesting that it indeed possesses corruption awareness.

To evaluate the second hypothesis, we analyze the confusion matrix of responses from LLaVA-1.5 [43]. Remarkably, the model consistently outputs Going ahead across all corruption scenarios, regardless of the actual visual context (shown in Fig. 21f in Appendix). This uniformity in answering indicates the model response is based on general knowledge rather than counting on visual inputs. Combining with the findings in Sec. 4.2.1, we conclude below:

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

100

100

100

80

80

80

Question-unawareEval

Context-unawareEval

Rubric-unawareEval

60

60

60

40

40

40

20

20

20

0

0

0

0 20 40 60 80 100

0 20 40 60 80 100

0 20 40 60 80 100

Rubirc-aware Eval

Question-aware Eval

Context-aware Eval

(a) Rubric-Aware GPT Evaluation

(b) Question-Aware GPT Evaluation

(c) Context-Aware GPT Evaluation

- Figure 13. Comparisons among different evaluation types (rubric, question-aware, and context-aware). The GPT scores vary depending on the rubric, question, and physical driving context. With more information added, the results become more distinguishable.

Q: In this scenario, what are safe actions to take for the ego vehicle?

Planning

[Figure 137]

[Figure 138]

[Figure 139]

Q: What are the important objects in the current scene?

Q: What object should be

noticed first when getting to the next possible location?

Perception Prediction

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Q: Predict the behavior of the ego vehicle in the current scene.

Behavior

[Figure 144]

A: The ego vehicle is steering to the right side of the road. The ego vehicle is driving with normal speed in the current condition.

A: The recommended safe actions are (1) it is getting

dark, ensure headlights are on, (2) there is a pedestrian

on the left sidewalk, maintain awareness, (3) maintain lane discipline and be cautious of vehicles approaching from the opposite direction.

A: The ego vehicle should notice (1) the pedestrian on the crosswalk from front-left

image, as it is is walking across the road, (2) the oncoming vehicle from front-

right image, as it is approaching from the opposite direction.

A: The important objects to consider for decision are:

- • Bus, near the center of the front image, and is moving slowly.

- • Pedestrian, near the left side of the front-left image, and is walking on the sidewalk.

[Figure 145]

Total Score: 74/100

Brief Summary: The answer provides a reasonable analysis of the objects in the scene and suggests appropriate actions. However, there are

some inaccuracies in identifying the objects and their states, as well as minor issues with clarity.

Total Score: 73/100

Brief Summary: The answer provides a detailed analysis of objects in the scene, showing good awareness of potential hazards. However, there are some inaccuracies in the motion precision and situational awareness aspects.

[Figure 146]

Total Score: 87/100 Total Score: 37/100 Brief Summary: The answer demonstrates a high level of accuracy in predicting safe actions based on the scenario. The reasoning and justification are logical and align well with safety considerations. It could benefit from slightly more conciseness.

Brief Summary: The predicted answer did not match the correct answer. However, the explanation demonstrated a good understanding of the ego vehicle's behavior and provided clear reasoning and justification based on the environmental cues.

- Figure 14. Examples of GPT-4o [2] responses to four tasks and the corresponding evaluation results under the dark condition. We observe that GPT-4o [2] is aware of the low-light environment and can identify the bus and pedestrian from the image, showing resilience.

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

###### Perception Prediction Planning Behavior

Q: What is the moving status of object in the front view camera image?

Q: What object should be

Q: What actions taken by the ego vehicle can lead to a collision with the object?

Q: Predict the behavior of the ego vehicle in the current scene.

[Figure 154]

noticed first when getting to the next possible location?

A: The object at coordinates (0.5073, 0.5778) appears to be turning left. The image shows a vehicle moving from the left side of the image towards the center, indicating a leftward motion status.

A: The ego vehicle should notice the car in the front image, as it is moving faster than the ego vehicle. The ego vehicle should maintain its lane and monitor the rearview camera for any changes in the approaching car.

A: The object appears to be a vehicle in the center of the image. Actions that could lead to a collision include sudden speeding up and lane change. To avoid a collision, the ego should maintain a steady speed and signal intentions clearly.

A: The ego vehicle is slightly steering to the left side of the road. The ego vehicle is driving fast in the current scene.

[Figure 155]

###### Total Score: 0/100

###### Total Score: 63/100

###### Total Score: 87/100 Total Score: 100/100 Brief Summary:

Brief Summary: The predicted answer does not align with the correct answer, and there are inaccuracies in object recognition, location,

Brief Summary: The answer correctly identifies the object but fails to prioritize according to the instructions. The description of the object and state are generally accurate, but there are inaccuracies and lack of clarity in the recommended actions for the ego vehicle.

Brief Summary: The predicted answer receives a perfect score of 100 as it exactly matches the correct answer, demonstrates a clear understanding of behavior and details, and provides logical reasoning based on environmental clues.

The answer demonstrates a strong understanding of potential actions that could lead to a collision with the identified object. The

[Figure 156]

orientation, environmental conditions, and reasoning.

reasoning provided is logical and aligns well with safety considerations.

- Figure 15. Examples of GPT-4o [2] responses to four tasks and the corresponding evaluation results under the motion blur condition. We observe that GPT-4o [2] are influenced by this type of corruption and tend to predict “driving fast” based on it. The example shows the potential of visual corruption to influence high-level driving decisions.

- • Advanced VLMs tend to rely predominantly on textbased cues to generate responses under conditions of visual degradation, even though they are aware of it.
- • Less advanced models demonstrate a stronger dependence on general knowledge acquired during training, resulting in responses dominated by learned priors rather than situational visual information.

growing body of research dedicated to this area [48, 59, 62].

Specifically, we select DriveLM [59] and Dolphin [48] as representative models for our analysis, as both exhibit promising results and are explicitly fine-tuned to enhance visual-grounded driving decision-making abilities. DriveLM is fine-tuned on the nuScenes dataset [55], while Dolphin is fine-tuned on the BDD dataset [78]. This distinction in fine-tuning datasets offers a unique opportunity to investigate the transferability of driving-specific VLMs across different datasets, as well as their answer reliability to visual corruption.

###### 4.2.3. Fine-Tuned VLM Models

Thus far, our analysis has primarily focused on generalpurpose VLMs applied in driving. While this focus is essential, given that these models serve as foundational architectures for task-specific fine-tuning, it remains insufficient for evaluating driving-specific capabilities, as these general models are not tailored or optimized for autonomous driving tasks. In this section, we shift our attention to VLMs fine-tuned specifically on driving datasets, reflecting the

The main results are summarized in Tab. 2 and Tab. 5. A key observation is that Dolphin [48], which is primarily fine-tuned on the BDD [78] dataset, demonstrates significant difficulty in answering questions from the nuScenes [55] dataset. Given the general capabilities of VLMs to ad-

[Figure 157]

[Figure 158]

[Figure 159]

Q: What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision making.

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Ground Truth: In this scene, there is a black pickup truck to the back, a white truck to the front right, a yellow truck to the front, and a black SUV to the back of the ego vehicle . . .

[Figure 164]

###### GPT4-o Qwen2-VL LLaVA-NeXT DriveLM-Agent

A: In the current scene, the important objects to consider are:

A: In the current scene, the important objects are:

A: In the current scene, the important objects include:

A: There is a white truck to the front of the ego vehicle, a white sedan to the back of the ego vehicle, a white

- • Trucks, including a large white truck in the front and a Ryder truck in the back.

- • Buildings, on the left and right sides of the road.

- • Trees, on the right side of the road.

- • Road markings, the road has clear lane markings and a pedestrian crossing.

- • The white semi-truck in the center of the image.

- • The Ryder truck on the right side of the image.

- • The white truck on the left side of the image.

- • The fire hydrant on the sidewalk.

- • The traffic signs and signals, including the stop sign and the traffic light.

- • Trucks, seen in front, front-left, front-right, and back-right images.

- • Buildings & fences, seen in front-left and frontright images.

- • Road signs, seen in the back-right image.

- • Pedestrian, visible in the front image.

sedan to the back of the ego vehicle, and a red light to the front of the ego vehicle.

[Figure 165]

- 1. Action Alignment:15
- 2. Motion Precision: 0
- 3. Driving Context Appropriateness: 10
- 4. Situational Aware: 10
- 5. Concise & Clarity: 15
- 6. Grammar: 10 Total Score: 60/100

- 1. Action Alignment:15
- 2. Motion Precision: 0
- 3. Driving Context Appropriateness: 10
- 4. Situational Aware: 10
- 5. Concise & Clarity: 15
- 6. Grammar: 10 Total Score: 60/100

- 1. Action Alignment:15
- 2. Motion Precision: 0
- 3. Driving Context Appropriateness: 10
- 4. Situational Aware: 10
- 5. Concise & Clarity: 15
- 6. Grammar: 8 Total Score: 58/100

- 1. Action Alignment:10
- 2. Motion Precision: 0
- 3. Driving Context Appropriateness: 5
- 4. Situational Aware: 5
- 5. Concise & Clarity: 10
- 6. Grammar: 10 Total Score: 40/100

[Figure 166]

Brief Summary: The predicted answer provides a detailed list of important objects in the scene, aligning well with the driving context and demonstrating good situational awareness. However, it lacks any prediction of future actions or motions based on these objects.

Brief Summary: The predicted answer accurately identifies the important objects in the scene, such as trucks, buildings, trees, and road markings. However, it lacks precision in matching the correct objects and their positions. The predicted answer does not provide any information on motion.

Brief Summary: The predicted answer provides a detailed list of important objects in the scene. The motion precision is missing entirely. The driving context appropriateness and

Brief Summary: The predicted answer accurately identified some objects in the scene but failed to match the correct objects and their positions. The motion precision was completely off, and the driving context appropriateness and situational awareness were partially addressed.

situational awareness are decent, but there are some inaccuracies.

- Figure 16. Examples of different VLM responses under the frame lost condition. We observe that GPT-4o [2] responses with visible objects while LLaVA-NeXT [44] and DriveLM [59] tend to hallucinate objects that cannot be seen from the provided images.

dress questions across diverse domains, this result is both surprising and concerning, highlighting the limited generalizability of driving-specific VLMs when exposed to datasets or question formats that differ from their fine-tuning conditions. Regarding DriveLM [59], we further investigate how the model benefits from in-distribution fine-tuning in the following section. This analysis aims to elucidate the poten-

tial advantages and limitations of fine-tuning on a specific language-annotated driving dataset.

###### 4.2.4. Metrics

Language Metrics. The insights presented thus far rely primarily on accuracy and GPT-based scores. However, due to the prohibitive costs associated with large-scale evalua-

[Figure 167]

[Figure 168]

[Figure 169]

Q: What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision making.

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Ground Truth: In this scene, there is a black SUV to the front, a black SUV to the back, a black sedan to the front, a pedestrian riding a bike to the front, and a red light to the front of the ego vehicle . . .

[Figure 174]

###### GPT4-o Qwen2-VL LLaVA-NeXT DriveLM-Agent

A: Based on the images, the important objects for driving decision-making are:

A: In the current scene, the important objects are:

A: In the current scene, the important objects include:

A: There is a black car to the back of the ego vehicle, a black sedan to the front of the ego vehicle, a black

- • Vehicles, there are multiple on the road, which are moving and need to be tracked for safe navigation.

- • Traffic lights, which will dictate when it is safe to proceed or stop.

- • Pedestrians, who may cross the road and need to be accounted for.

- • The car in the foreground, which is the vehicle the autonomous driving assistant is controlling.

- • The traffic lights, which are relevant for traffic rules and signals.

- • The pedestrians, which are important for ensuring safety.

- • The other vehicles.

- • Vehicles, visible in the lane ahead, parked or moving slowly on the left side of the road, and visible behind.

- • Traffic signals, visible in the distance.

- • Road conditions, which is wet and indicating the need for cautious driving.

sedan to the front of the ego vehicle, and a white sedan to the front of the ego vehicle.

[Figure 175]

- 1. Action Alignment:15
- 2. Motion Precision: 0
- 3. Driving Context Appropriateness: 10
- 4. Situational Aware: 10
- 5. Concise & Clarity: 15
- 6. Grammar: 10 Total Score: 60/100

- 1. Action Alignment:15
- 2. Motion Precision: 0
- 3. Driving Context Appropriateness: 10
- 4. Situational Aware: 10
- 5. Concise & Clarity: 15
- 6. Grammar: 10 Total Score: 60/100

- 1. Action Alignment:15
- 2. Motion Precision: 0
- 3. Driving Context Appropriateness: 10
- 4. Situational Aware: 10
- 5. Concise & Clarity: 15
- 6. Grammar: 8 Total Score: 58/100

- 1. Action Alignment:10
- 2. Motion Precision: 0
- 3. Driving Context Appropriateness: 5
- 4. Situational Aware: 5
- 5. Concise & Clarity: 10
- 6. Grammar: 10 Total Score: 40/100

[Figure 176]

Brief Summary: The predicted answer provides a detailed list of important objects in the current scene, aligning well with the driving context and demonstrating good situational awareness. However, there is a lack of motion precision and failure to mention all objects mentioned in the answer.

Brief Summary: The predicted answer provides a detailed list of important objects in the scene, aligning well with the

Brief Summary: The predicted answer provides a detailed list of important objects in the scene, aligning well with the driving context and demonstrating situational awareness. However, the predicted motion is missing.

Brief Summary: The predicted answer accurately identified some of the important objects in the scene but missed crucial details such as the presence of a pedestrian riding a bike and a red light. The motion precision was completely off, as it did not mention any motion-related information.

driving context and demonstrating good situational awareness.

However, there is a lack of specific motion predictions and inaccuracies in identifying all the objects.

- Figure 17. Examples of different VLM responses under the water splash condition. We observe that, under severe visual corruptions, VLMs respond with ambiguous and general answers based on their learned knowledge, without referring to the visual information. Most responses include traffic signals and pedestrians, even though they are not visible in the provided images.

tions using GPT APIs, traditional language metrics, such as BLEU [54] and ROUGE-L [40] remain widely employed in existing benchmarks [29, 59]. To better understand their applicability, we evaluate the validity of these metrics in the context of language-based driving decision tasks. In Fig. 11, we present the prediction task performance un-

der clean image inputs, evaluated across different metrics. As anticipated, the language metrics demonstrate high internal consistency: models with high BLEU scores also tend to achieve high ROUGE-L scores, as both metrics emphasize pattern-matching between predicted responses and ground-truth answers. Furthermore, we visualize how

the same responses are scored under ROUGE-L and GPT scores at scale in Fig. 12a. The results further reveal how the language score fails to reflect underlying key information: when ROUGE-L remains around 0.2, GPT scores vary across a large range. DriveLM [59], while surpassing other VLMs with large margins under ROUGE-L evaluation, still lags behind Qwen2-VL72B [67] and GPT-4o [2] in GPT evaluation. The observation indicates that the main improvement of in-distribution fine-tuning on the current small-scale driving dataset largely comes from the answering template.

Accuracy. In terms of accuracy, we also study how accuracy and GPT scores are related. The results are presented in Fig 12b. The GPT evaluation highly aligns with accuracy since we prompt the GPT to assign certain scores if the answer is correct. The divination is because GPT assigns another portion of scores to the coherence of explanation dimensions, capturing nuanced differences between answers. GPT Evaluation. A critical question remains: is GPT evaluation currently the optimal approach? The answer is nuanced. GPT-based scoring can effectively capture human preferences and emphasize critical elements in driving scenarios, yet this capability is highly contingent on the provision of comprehensive driving contextual information. We empirically compare how the same response is scored given different information, shown in Fig. 13a. When GPT evaluation is prompted solely with GT and model response, the resulting scores are highly homogeneous while the inclusion of specific rubrics, questions, and specific-driving context yields greater score diversity.

##### 5. Conclusion

This work identifies and addresses key challenges in the deployment of Vision-Language Models (VLMs) for autonomous driving, with an emphasis on their visualgrounding reliability in complex real-world scenarios. Our findings reveal that VLMs frequently generate plausible yet unsupported responses when subjected to severe visual degradation, casting doubt on their reliability in critical decision-making tasks. Furthermore, imbalanced datasets and suboptimal evaluation protocols amplify these concerns, contributing to an overestimation of VLM reliabilities. To mitigate these challenges, we advocate for future efforts to prioritize the development of well-balanced, context-aware datasets and advanced evaluation metrics that rigorously assess the quality, contextual reasoning, and safety of driving decisions.

##### Appendix

A. Benchmark Setup 15 A.1. Benchmark Construction . . . . . . . . . . . 15

- A.2. Corruption Definitions . . . . . . . . . . . . 15
- A.3. Overall Statistics . . . . . . . . . . . . . . . 17
- A.4. License . . . . . . . . . . . . . . . . . . . . 17

- B. Benchmark Study 17

- B.1. DriveLM-nuScenes . . . . . . . . . . . . . . 17
- B.2. BDD-X . . . . . . . . . . . . . . . . . . . . 17

- C. Additional Implementation Details 17

- C.1. VLM Configurations . . . . . . . . . . . . . 17
- C.2. VLM Prompts . . . . . . . . . . . . . . . . 18
- C.3. GPT Evaluations . . . . . . . . . . . . . . . 18
- C.4. Human Evaluations . . . . . . . . . . . . . . 19

- D. Detailed Experiment Results 19

- D.1. GPT Scores . . . . . . . . . . . . . . . . . . 19
- D.2. Accuracy Scores . . . . . . . . . . . . . . . 19
- D.3. ROUGE-L Scores . . . . . . . . . . . . . . 19

- E. Broader Impact & Limitations 37

- E.1. Broader Impact . . . . . . . . . . . . . . . . 37
- E.2. Potential Limitations . . . . . . . . . . . . . 37

- F. Public Resource Used 37

##### A. Benchmark Setup

In this section, we elaborate in detail on the procedures and protocols used to establish the DriveBench in this work.

###### A.1. Benchmark Construction

We detailed the benchmark construction process in this section. Our DriveBench is primarily inspired by DriveLM [59] given its impact and representativeness. Given its public availability, we subsample 200 keyframes from the DriveLM [59] training dataset. These keyframes are selected to balance the ground truth distribution, which can more accurately reflect the model’s performance and prevent bias in most common cases (e.g., “Going Ahead”). Each keyframe has multiple questions related to different tasks, spanning perception, prediction, planning, and behavior. For each task, we follow the question type design in DriveLM [59], including multiple-choice questions (MCQs) and visual question answering (VQAs), as shown in Tab. 6. When evaluating corruption awareness, we add information about corruption context to the question and modify the answer accordingly if necessary. We generate the corruption-related question-answering pairs by prompting GPT4 based on original QA pairs, which we refer to the robustness dataset as shown in Tab. 7.

###### A.2. Corruption Definitions

In this section, we detailed our settings for generating image corruption. DriveBench encompasses five distinctive

<RUBRICS> Here is the question: <QUESTION> Here is the ground truth object visual description: <DESC> Here is the correct answer: <GT> Here is the model answer: <ANS>

<RUBRICS> Here is the question: <QUESTION> Here is the correct answer: <GT> Here is the model answer: <ANS>

<RUBRICS> Here is the correct answer: <GT> Here is the model answer: <ANS>

-----------------------------------------------------------------------------------

-----------------------------------------------------------------------------------

-----------------------------------------------------------------------------------

- 1. Action Prediction Accuracy: 25/40
- 2. Reasoning and Justification: 15/20
- 3. Probability or Confidence: 10/15
- 4. Contextual Awareness and Safety Considerations: 10/15
- 5. Conciseness and Clarity: 5/10

- 1. Action Prediction Accuracy: 35/40
- 2. Reasoning and Justification: 18/20
- 3. Probability or Confidence: 12/15
- 4. Contextual Awareness and Safety Considerations: 13/15
- 5. Conciseness and Clarity: 8/10

- 1. Action Prediction Accuracy: 20/40
- 2. Reasoning and Justification: 15/20
- 3. Probability or Confidence: 10/15
- 4. Contextual Awareness and Safety Considerations: 10/15
- 5. Conciseness and Clarity: 5/10

Score: 86

###### Score: 65 Score: 60

- Figure 18. Examples of GPT evaluators with different information. The evaluator can revise the score and give a more accurate evaluation based on more contextual information in the driving scenarios. For more details please refer to Fig. 32.

###### Table 6. Detailed distribution of the curated benchmark dataset. # Driving Task Question Type # Samples Total

mark, we include the 6Water Splash and 7Lens Obstacle corruptions.

- • Sensor Failures (3 Types): The simulations of sensor failures. In this benchmark, we include the 8Camera Crash, 9Frame Lost, and 10Saturate corruptions.

[Figure 177]

- • Motion Blurs (2 Types): The simulations of the blurs caused by the ego vehicle’s high-speed motion. In this benchmark, we include the 11Motion Blur and 12Zoom Blur corruptions.

[Figure 178]

- • Data Transmission Errors (3 Types): The simulations of the errors happening during the video transmission process. In this benchmark, we include the 13Bit Error, 14Color Quantization, and 15H.265 Compression corruptions.

[Figure 179]

- ◦ Perception MCQ & VQA 400

- ◦ Prediction VQA 61 1,261

[Figure 180]

[Figure 181]

- ◦ Planning VQA 600

[Figure 182]

- ◦ Behavior MCQ 200

###### Table 7. Detailed distribution of the proposed robustness benchmark dataset. The total number is summed across all the corruption types. “Corrupt. Rec.” represents corruption recognition, which asks the model to identify the current corruption types. “Corrupt. Desc.” represents questions related to the description of the current corrupted environment.

[Figure 183]

###### # Driving Task Question Type # Samples Total

[Figure 184]

- ◦ Corrupt. Rec. MCQ 4,000

[Figure 185]

- ◦ Perception MCQ & VQA 5,475

- ◦ Prediction VQA 799 19,237

[Figure 186]

[Figure 187]

- ◦ Planning VQA 5,999

[Figure 188]

- ◦ Corrupt. Desc. CAP 3,000

All the 15 corruption types are generated by applying high-fidelity image processing algorithms developed in previous works [30, 31, 71, 77]. Here, we detail how each corruptions are synthesized as follows:

- • 1Brightness: Adjusts the brightness values of the camera images by scaling pixel intensity upwards.
- • 2Dark: Simulates low-light conditions by scaling down the image’s brightness using a gamma-adjusted mapping. Additionally, it introduces Poisson noise to mimic photon shot noise and Gaussian noise to simulate sensor noise.
- • 3Snow: Generates a synthetic snow layer using random noise, applies motion blur to simulate falling snow, and blends it with the original image.
- • 4Fog: Simulates fog by blending a fractal noise-based fog layer over the image.
- • 5Rain: Adds streak-like artifacts to the image, created through line patterns combined with motion blur, to sim-

corruption categories, each with multiple different types of corruptions reflecting the real-world scenarios.

- • Weather & Lighting Conditions (5 Types): The simulations of diverse environmental weather and lighting conditions are used in the driving scenarios. In this benchmark, we include the 1Brightness, 2Dark, 3Snow, 4Fog, and 5Rain corruptions.

[Figure 189]

- • External Disturbances (2 Types): The simulations of situations where camera lenses are occluded by external objects or stains. In this bench-

[Figure 190]

ulate rain.

- • 6Water Splash: Simulates water splashes by overlaying transparent circular droplet patterns on the image, followed by a Gaussian blur to mimic water distortion effects.
- • 7Lens Obstacle: Creates lens obstruction effects by blending blurred and unblurred regions of the image using a randomly placed and shaped elliptical mask to emulate obstructions on the lens surface.
- • 8Camera Crash: Simulates a camera crash by replacing the affected image frames with black frames, representing a complete loss of data for specific viewpoints or cameras.
- • 9Frame Lost: Emulates frame loss by randomly setting some frames to black, indicating partial data corruption or a temporary transmission failure.
- • 10Saturate: Modifies the image’s color saturation by manipulating the saturation channel in the HSV color space, either enhancing or reducing the vibrancy of colors.
- • 11Motion Blur: Applies a linear motion blur to the image, simulating movement during exposure, with the blur radius and direction determined by severity.
- • 12Zoom Blur: Applies a radial zoom effect to the image, creating a focused blur that simulates rapid movement toward or away from the lens, controlled by severity.
- • 13Bit Error: Introduces random bit-level noise in the image data, mimicking digital corruption, with severity influencing the extent of errors.
- • 14Color Quantization: Reduces the image’s color palette to a limited set of levels, simulating low-quality color quantization, where severity controls the number of colors.
- • 15H.265 Compression: Applies heavy H.265 video compression artifacts to the image, with severity increasing the compression level and artifact visibility.

###### A.3. Overall Statistics

The results in the main paper are based on the curated datasets, which contain 1,461 questions, the detailed distribution of the questions is shown in Tab. 6. Specifically, each keyframe has two perception questions: one for MCQ and the other for VQA, four VQA questions for the planning task, and one behavior MCQ for the behavior task. In terms of the prediction task, not all keyframes have corresponding prediction questions.

###### A.4. License

The proposed benchmark is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 In-

ternational License1.

##### B. Benchmark Study

In this section, we include detailed information on the dataset distribution of the representative driving-withlanguage dataset. These datasets advance the development of driving with language models.

- B.1. DriveLM-nuScenes

We visualize the dataset distribution in perception and behavior tasks in the DriveLM-nuScenes dataset [59], as shown in Fig. 3. The majority of choices for vehicle behaviors are going straight, which is also shown in [39]. Along with the DriveLM-Agent results in the main paper, we find that highly imbalanced data can cause several problems. When fine-tuning VLMs on this dataset, the model tends to memorize the majority of choices, and thus answer with it during inference even if the visual cues are absent, which prevents effective evaluation of the model reliability. The dataset mainly adopts language metrics [40, 54] and naive GPT score: prompt with only the answer and ground truth, for evaluation. Our results in the main paper also show the limitations of these metrics in evaluating language-based driving decisions.

- B.2. BDD-X

To show the general existence of the limitations of existing benchmarks, we also study the BDD-X [29] dataset. We observe similar limitations as those in DriveLM [59] dataset, where the data is highly imbalanced. Most actions of the car are “Stop” or “Going Ahead”, where the random guessing of VLMs can achieve high accuracy since we’ve shown they potentially guessed the answer based on common knowledge and general case in Sec. 4.2.1. The observation shows that the limitations observed in our work are not individual but general drawbacks of existing drivingwith-language benchmarks.

##### C. Additional Implementation Details

In this section, we provide more details in terms of implementations and evaluation to facilitate the reproduction of this work.

###### C.1. VLM Configurations

• GPT-4o [2], developed by OpenAI’s, offers GPT-4-level intelligence with enhanced speed and multimodal capabilities, including voice, text, and image processing. It is designed to provide faster and more efficient responses across various applications.

1https://creativecommons.org/licenses/by-nc-sa/ 4.0.

Distribution of Car Actions

2000

is stopped

isdrivingforward

movesforwardthencomestoastop

1750

1500

acceleratestoaconstantspeed

accelerates

isstoppedatanintersection

isdrivingdownthestreet

brakesandcomestoastop

istravellingdowntheroad.

slowsdown

1250

isdrivingdownthehighway

proceedsdownthestreet

acceleratesdowntheroad

istravellingdowntheroad

isheadingdownthestreet

Frequency

isdrivingforwardslowly

isslowingtoastop

headsdownthestreet

slowstoastop

1000

headsdowntheroad

isnotmoving

isaccelerating

isslowingdown

comestoastop

drivesforward

isstopping

750

isturningright

maintainsspeed

stops

isturningleft

isstationary

nan

hasstopped

isslowing

isstopped.

turnsright

500

slows

turnsleft

250

0

0 5 10 15 20 25 30 35

Actions

- Figure 19. BDD-X dataset [29]: detailed distribution of car actions. Only the actions with a frequency larger than 80 are visualized. The Stop actions and Turn actions are highlighted. We observe similar data distribution in balance as those in DriveLM [59], where turning actions only account for a small portion of all actions.

- • Phi-3 [1] is a language model developed by Microsoft, focusing on efficiency and performance in natural language understanding and generation tasks. It is designed to handle a wide range of applications, from conversational agents to content creation.
- • Phi-3.5 [1] is an advanced version of Microsoft’s Phi series, offering improved reasoning and mathematical capabilities comparable to larger models like GPT-4o. It maintains efficiency while managing complex AI tasks across different languages.
- • LLaVA-1.5 [43] is an open-source large multimodal model that integrates vision and language understanding. With 13 billion parameters, it achieves state-of-the-art performance across multiple benchmarks, rivaling models like GPT-4.
- • LLaVA-NeXT [44] is an evolution of the LLaVA series, enhancing multimodal capabilities by supporting multiimage, video, and 3D tasks within a unified large language model. It achieves state-of-the-art performance on a wide range of benchmarks, demonstrating strong video understanding through task transfer from images.
- • InterVL [12] is an open-source multimodal dialogue model developed by OpenGVLab. It closely approximates the performance of proprietary models like GPT4o, excelling in tasks that integrate visual and linguistic information, such as visual question answering and image captioning.
- • Oryx [17] is a unified multimodal architecture created by researchers from Tsinghua University and Tencent. It is designed for spatial-temporal understanding of images, videos, and multi-view 3D scenes, offering on-demand processing of visual inputs with arbitrary spatial sizes and temporal lengths.
- • Qwen2-VL [5, 67] is a large language model developed

- by Alibaba Cloud, available in both chat and pretrained versions. It delivers high-quality language generation and understanding capabilities, optimized for tasks requiring nuanced comprehension and generation of human language.
- • DriveLM-Agent [59] is a model from OpenDriveLab tailored for autonomous driving applications, focusing on graph-based visual question answering. It addresses challenges in driving scenarios by integrating language understanding with visual perception, enhancing decisionmaking processes in autonomous systems.
- • Dolphins [48] is a multimodal language model developed by NVIDIA for driving applications. It adeptly processes inputs such as video data, text instructions, and historical control signals to generate informed outputs, facilitating a comprehensive understanding of complex driving scenarios.

###### C.2. VLM Prompts

We use the same system prompt for all the candidate VLMs, as shown in Fig. 20. We prompt the VLMs to explain for MCQs to facilitate the GPT evaluation based on their explanations.

###### C.3. GPT Evaluations

We include the detailed prompt we used for GPT evaluation here, we use the following prompt to evaluate MCQ questions in the perception task as shown in Fig. 22. The DESC is used to prompt context information for accurate evaluation. Limited to the current drive-with-language dataset, we extract the natural language description of critical objects in the current environment to provide context information.

The prompt for the open-ended perception task is shown in Fig. 23. Since the ground truth for open-ended questions

You are a smart autonomous driving assistant responsible for analyzing and responding to driving scenarios. You are provided with up to six camera images in the sequence [CAM FRONT, CAM FRONT LEFT, CAM FRONT RIGHT, CAM BACK, CAM BACK LEFT, CAM BACK RIGHT]. Each image has normalized coordinates from [0, 1], with (0,0) at the top left and (1,1) at the bottom right.

Instructions:

- 1. Answer Requirements:

- • For multiple-choice questions, provide the selected answer choice along with an explanation.

- • For “is” or “is not” questions, respond with a “Yes” or “No”, along with an explanation.
- • For open-ended perception and prediction questions, related objects to which the camera.

- 2. Key Information for Driving Context:

• When answering, focus on object attributes (e.g., categories, statuses, visual descriptions) and motions (e.g., speed, action, acceleration) relevant to driving decision-making

Use the images and coordinate information to respond accurately to questions related to perception, prediction, planning, or behavior, based on the question requirements.

Figure 20. Inference system prompt.

already included the visual description and moving status of important objects, we only prompt with PRED and GT with detailed rubrics.

###### C.4. Human Evaluations

In this subsection, we elaborate in more detail on how we conduct the human evaluation experiments in our benchmark.

Procedures. Considering the large number of questions in the dataset, we subsample 15 out of 200 keyframes from our curated dataset. To ensure no overlaps between different corruptions, which might cause information leakage, we lower the probability if the same keyframes are sampled before. Then, we design a user interface for human evaluation, focusing on MCQs. The interface is shown in Fig. 24. As with evaluating VLMs, we only prompt single-view images if the questions are related to only one of the cameras.

Ethic Declaration. According to the Federal Policy of Human Subjective Research2, our research involves conducting anonymous visual recognition tasks, where participants respond to questions about visual stimuli without any interventions or demographic data collection. It qualifies for Exempt Research 2(i)3 because it solely involves surveylike procedures with no physical or psychological risks to

- 2https://www.federalregister.gov/d/2017-01058/

p-1315

- 3https://www.federalregister.gov/d/2017-01058/

p-1375

participants. Specifically, it meets the requirements of Exempt Research 2(i) as the data is recorded in a manner ensuring that participants’ identities cannot be readily ascertained, directly or through linked identifiers. It does not fall under Exempt Research 2(ii) or 2(iii) because no identifiable information is recorded, and no IRB review is required to ensure these protections. We also submit an IRB review records to the corresponding institutions and receive the official confirmation of IRB review exemption.

##### D. Detailed Experiment Results

In this section, we include the detailed benchmark results evaluated by the GPT score metric. We also include the prediction spatial distribution in Fig. 21 for each model.

###### D.1. GPT Scores

We include the detailed GPT scores in Tab. 8, Tab. 10, Tab. 12, Tab. 13, and Tab. 15 for different tasks. The observation and conclusion in the main paper are primarily derived from GPT scores. Therefore, we focus more on the discussion of accuracy and language scores in the following sections.

###### D.2. Accuracy Scores

We include accuracy scores for MCQs in addition to GPT scores in Tab. 9 and Tab. 11. Compared with GPT scores, we find that the accuracy score metric is more homogeneous. For example, the LLaVA-1.5 models have 50% accuracy under all the input types, suggesting they are merely output “Going Ahead” for perception MCQs, which is also observed in the prediction spatial distribution in Fig. 21e and 21f. Moreover, we find that most models have no accuracy degradation under corruptions or even text-only inputs. This raises concerns about whether VLMs are indeed leveraging visual information to make decisions about the specified spatial location or naively guessing based on their general knowledge.

###### D.3. ROUGE-L Scores

We also present the detailed language scores, i.e., ROUGEL [40] here for open-ended questions in Tab. 14 and Tab. 16. As discussed in the main paper, we find the fine-tuning process can significantly benefit the ROUGE-L score, as indicated by the fact that DriveLM [59] output performs other models with a large margin. On the contrary, GPT-4o [2], which generates more detailed answers punished by the answer length. The ROUGE-L score of GPT-4o [2] is lower than most of the models even though the GPT scores are much higher.

Cam Front Left

Cam Front

Cam Front Right

Cam Front Left

Cam Front

Cam Front Right

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

Cam Back Left

Cam Back

Cam Back Right

Cam Back Left

Cam Back

Cam Back Right

| |
|---|

| |
|---|

|Going ahead<br><br>Turn left<br><br>Turn right|
|---|

| |
|---|

| |
|---|

|Going ahead<br><br>Turn left<br><br>Turn right|
|---|

(a) Ground Truth

(b) GPT-4o [2]

Cam Front Left

Cam Front

Cam Front Right

Cam Front Left

Cam Front

Cam Front Right

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

Cam Back Left

Cam Back

Cam Back Right

Cam Back Left

Cam Back

Cam Back Right

| |
|---|

| |
|---|

|Going ahead<br><br>Turn left<br><br>Turn right|
|---|

| |
|---|

| |
|---|

|Going ahead<br><br>Turn left<br><br>Turn right|
|---|

(c) Phi-3 [1]

(d) Phi-3.5 [1]

Cam Front Left

Cam Front

Cam Front Right

Cam Front Left

Cam Front

Cam Front Right

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

Cam Back Left

Cam Back

Cam Back Right

Cam Back Left

Cam Back

Cam Back Right

| |
|---|

| |
|---|

|Going ahead<br><br>Turn left<br><br>Turn right|
|---|

| |
|---|

| |
|---|

|Going ahead<br><br>Turn left<br><br>Turn right|
|---|

(e) LLaVA-1.57B [43]

(f) LLaVA-1.513B [43]

Cam Front Left

Cam Front

Cam Front Right

Cam Front Left

Cam Front

Cam Front Right

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

Cam Back Left

Cam Back

Cam Back Right

Cam Back Left

Cam Back

Cam Back Right

| |
|---|

| |
|---|

|Going ahead<br><br>Turn left<br><br>Turn right|
|---|

| |
|---|

| |
|---|

|Going ahead<br><br>Turn left<br><br>Turn right|
|---|

(g) Qwen2-VL72B [67]

(h) InterVL8B [12]

Figure 21. Prediction spatial distributions from VLMs. The locations represent the object positions in the image within each camera, which is input to the model as a text description. We only visualize the data point where the model response aligns with the provided multiple choices (e.g., “Going ahead”, “Turn Left”, and “Turn Right”).

Please evaluate the multiple-choice answer on a scale from 0 to 100, where a higher score reflects precise alignment with the correct answer and well-supported reasoning. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

###### 1. Answer Correctness (50 points):

- - Exact Match (50 points): Assign 50 points if the predicted answer exactly matches the correct answer.
- - No Match (0 points): Assign 0 points if the predicted answer does not match the correct answer, regardless of explanation quality. 2. Object Recognition (10 points):
- - Award up to 5 points for accurately identifying all relevant object(s) in the scene.
- - Award up to 5 points for correct descriptions of the identified object(s), including attributes like colors, materials, sizes, or shapes.
- - Guideline: Deduct points for any missing, misidentified, or irrelevant objects, particularly if they are crucial to the driving context. Deduct points if any important visual details are missing, incorrect, or overly generalized, especially if they affect comprehension or recognition. 3. Object Location and Orientation (15 points):
- - Score up to 5 points for a precise description of the object’s location, orientation, or position relative to the ego vehicle.
- - Award up to 5 points for acknowledging environmental factors, such as lighting, visibility, and other conditions that influence perception.
- - Score up to 5 points based on how well the answer reflects an understanding of situational context, such as obstacles, traffic flow, or potential hazards.
- - Guideline: Deduct points for inaccuracies or omissions in spatial information that could affect scene understanding. Deduct points if the answer fails to consider factors impacting object visibility or situational awareness. Deduct points for overlooked or misinterpreted contextual factors that may impact driving decisions.

###### 4. Environmental Condition Awareness (15 points):

- - Award up to 15 points if the explanation considers environmental conditions (e.g., weather or sensor limitations) that could impact perception.
- - Guideline: Deduct points if relevant environmental conditions are ignored or inadequately addressed. 5. Clarity of Reasoning (10 points):
- - Award up to 5 points for clear, logically structured reasoning that is easy to understand.
- - Assign up to 5 points for grammatical accuracy and coherent structure.
- - Guideline: Deduct points for vague or confusing explanations that hinder comprehension. Deduct points for grammar or syntax issues that impact clarity or logical flow. Assign 0 points from criteria 2 to 5 if no explanation is provided.

Here is the multiple-choice question: QUESTION Here is the ground truth object visual description: DESC Here is the correct answer: GT Here is the predicted answer and explanation (if any): PRED

###### Please fill in the following scoring sheet, and then provide a brief summary supporting the score:

- 1. Answer Correctness (50 points):
- 2. Object Recognition (10 points):
- 3. Object Location and Orientation (15 points):
- 4. Environmental Condition Awareness (15 points):
- 5. Clarity of Reasoning (10 points):

###### Total Score: Brief Summary:

Figure 22. GPT evaluation prompts for MCQs in our benchmark.

Please evaluate the predicted answer on a scale from 0 to 100, where a higher score reflects precise alignment with the correct answer and well-supported reasoning. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

###### 1. Action Alignment (20 points):

- - Assign up to 20 points based on how accurately the predicted action (e.g., forward, turn left, turn right) matches the correct answer.
- - Guideline: Award full points only for exact matches or highly similar actions. Deduct points for any inaccuracies or missing elements. Assign 0 points if no action prediction is provided.

###### 2. Motion Precision (20 points):

- - Award up to 20 points based on how closely the predicted motion (e.g., speed up, decelerate) aligns with the correct motion in the answer.
- - Guideline: Deduct points if the predicted motion fails to match the type or intensity of the correct answer. Ensure that the intended speed or deceleration aligns accurately with the driving context. Assign 0 points if no motion prediction is provided. 3. Driving Context Appropriateness (15 points):
- - Score up to 15 points for the relevance of the predicted answer to the driving context implied by the correct answer, emphasizing logical alignment with the situation.
- - Guideline: Award higher scores only if the answer fully reflects an accurate understanding of the driving context. Deduct points if the action or motion is illogical or does not align with the scenario’s requirements. 4. Situational Awareness (15 points):
- - Award up to 15 points for demonstrated awareness of environmental factors (e.g., traffic participants, obstacles) relevant to the action or motion.
- - Guideline: Deduct points if the answer misses key situational details that may lead to unsafe or incorrect predictions. 5. Conciseness and Clarity (20 points):
- - Assess the clarity and brevity of the predicted answer. Answers should be concise, clear, and easy to understand, effectively communicating the intended actions and motions.
- - Guideline: Deduct points for verbosity, ambiguity, or lack of focus that could hinder quick comprehension. 6. Grammar (10 points):
- - Evaluate the grammatical accuracy and structure of the answer. Assign up to 5 points for clarity and logical flow, and up to 5 points for grammatical accuracy.
- - Guideline: Deduct points for grammar or syntax issues that reduce readability or coherence. Here is the predicted answer: PRED Here is the correct answer: GT Please fill in the following scoring sheet, and then provide a brief summary supporting the score:

- 1. Action Alignment (20 points):
- 2. Motion Precision (20 points):
- 3. Driving Context Appropriateness (15 points):
- 4. Situational Awareness (15 points):
- 5. Conciseness and Clarity (20 points):
- 6. Grammar (10 points):

###### Total Score: Brief Summary:

Figure 23. GPT evaluation prompts for Open-Ended Questions in our benchmark.

[Figure 191]

[Figure 192]

(a) (b)

[Figure 193]

[Figure 194]

(c) (d)

[Figure 195]

[Figure 196]

(e) (f)

Figure 24. Illustrative examples from our human evaluation interfaces.

Clean Bright Dark

Snow

Fog

Rain Lens

Water Crash Frame

Saturate

Motion

Zoom

Bit

Quant H.265

| | |
|---|---|
|0|20 40 60 80|

(a) GPT-4o

| | |
|---|---|
|0|20 40 60 80|

(b) Phi3

| | |
|---|---|
|0|20 40 60 80|

(c) Phi-3.5

| | |
|---|---|
|0|20 40 60 80|

(d) LLaVA-1.57B

| | |
|---|---|
|0|20 40 60 80|

(g) InternVL8B

| | |
|---|---|
|0|20 40 60 80|

(e) LLaVA-1.513B

| | |
|---|---|
|0|20 40 60 80|

(h) Oryx

| | |
|---|---|
|0|20 40 60 80|

(f) LLaVA-NeXT

| | |
|---|---|
|0|20 40 60 80|

(i) Qwen2VL7B

| | |
|---|---|
|0|20 40 60 80|

(j) Qwen2VL72B

| | |
|---|---|
|0|20 40 60 80|

(k) Dolphin

| | |
|---|---|
|0|20 40 60 80|

(l) DriveLM

- Figure 25. Model performance comparisons using radar graphs. The performance for each input corruption type is averaged across all the 1, 261 questions spanning four different tasks using GPT scores. The gray dash line represents the performance of text-only input. We observe VLMs have subtle performance changes under corruptions. For some models, the GPT scores under only text input are even higher than the performance when the visual information is available.

###### What is the moving status of object <c2,CAM_BACK,0.5443,0.5639>? Please select the correct answer from the following options: A. Turn left. B. Going ahead. C. Turn right.

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

B. Going ahead. The object is located in the center of the image and appears to be moving forward, as there are no visible indicators of turning or stopping.

B. Going ahead. The object is a car that is positioned in the lane ahead of the ego vehicle and appears to be moving in the same direction as the ego vehicle.

- 1. Answer Correctness (50 points): 50
- 2. Object Recognition (10 points): 5
- 3. Location & Orientation (15 points): 5
- 4. Environmental Condition Awareness (15 points): 10
- 5. Clarity of Reasoning (10 points): 5 Total Score: 75/100

- 1. Answer Correctness (50 points): 50
- 2. Object Recognition (10 points): 5
- 3. Location & Orientation (15 points): 5
- 4. Environmental Condition Awareness (15 points): 10
- 5. Clarity of Reasoning (10 points): 5 Total Score: 75/100

[Figure 201]

Brief Summary: The answer provided accurately identifies the object as a car and correctly states its moving status as "Going ahead." The explanation includes details about the object's position relative to the ego vehicle and considers environmental factors that could impact perception. The reasoning is clear and logically structured, earning a high score of 75.

Brief Summary: The answer provided accurately identifies the moving status of the object as "Going ahead" based on the visual description of a gray car. The explanation is clear, logically structured, and considers environmental conditions that could impact perception, earning a total score of 75.

[Figure 202]

- Figure 26. Example of Qwen2-VL72B for the Perception task. Under the text-only condition (right example), the model can fabricate answers based on the coordinate information in the question. The GPT evaluator can not reflect the fabricated answer even with detailed rubrics, resulting in the same score as prompted with a clean image.

What object should the ego vehicle notice first when the ego vehicle is getting to the next possible location? What is the state of the object that is first noticed by the ego vehicle and what action should the ego vehicle take? What object should the ego vehicle notice second . . .

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

###### First Object Noticed:

###### First Object Noticed:

- • Object: The black SUV in CAM_FRONT_RIGHT.

- • State: It is parked on the right side of the road.

- • Action: The ego vehicle should maintain its current speed and trajectory, as the parked SUV does not pose an immediate threat.

- • Object: The white car in CAM_FRONT_RIGHT.

- • State: It is parked on the right side of the road.

- • Action: The ego vehicle should maintain its current speed and trajectory, as there is no immediate need to slow down or change direction.

[Figure 216]

###### Second Object Noticed:

###### Second Object Noticed:

- • Object: The white van in CAM_FRONT_RIGHT.

- • State: It is parked on the right side of the road.

- • Action: The ego vehicle should maintain its current speed and trajectory, as the parked van does not pose an immediate threat.

- • Object: The black car in CAM_FRONT_LEFT.

- • State: It is parked on the left side of the road.

- • Action: The ego vehicle should continue to maintain its current speed and trajectory, as there is no immediate obstacle in its path.

###### Third Object Noticed:

###### Third Object Noticed:

- • Object: The white car in CAM_FRONT_RIGHT.

- • State: It is parked on the right side of the road.

- • Action: The ego vehicle should maintain its current speed and trajectory, as the parked car does not pose an immediate threat.

- • Object: The building in CAM_FRONT.

- • State: It is stationary on the right side of the road.

- • Action: The ego vehicle should continue to maintain its current speed and trajectory, as the building does not pose an immediate threat to its path.

- 1. Object Identification & Priority Order: 15/20
- 2. Object Category & Visual Description: 10/20
- 3. State of the Object: 5/15
- 4. Recommended Action for Ego Vehicle: 10/15
- 5. Logical Flow & Reasonableness: 10/20
- 6. Clarity and Grammar: 8/10 Total Score: 58/100

- 1. Object Identification & Priority Order: 15/20
- 2. Object Category & Visual Description: 18/20
- 3. State of the Object: 12/15
- 4. Recommended Action for Ego Vehicle: 12/15
- 5. Logical Flow & Reasonableness: 16/20
- 6. Clarity and Grammar: 8/10 Total Score: 81/100

[Figure 217]

Brief Summary: The predicted answer correctly identifies the objects in the correct priority order, but there are inaccuracies in the visual descriptions and states of the objects. The recommended actions ... The logical flow ... The clarity and grammar ... Overall, the predicted answer needs improvement in providing accurate visual descriptions, states of objects, and recommended actions for the ego vehicle to align better with the correct answer.

Brief Summary: The predicted answer demonstrates a good understanding of the scenario and accurately identifies the objects in the correct priority order. However, there are minor inaccuracies in the description of the object states and recommended actions for the ego vehicle. The logical flow ... Overall, the predicted answer provides a reasonable response but could benefit from more precise details and clearer explanations.

[Figure 218]

- Figure 27. Example of Qwen2-VL72B for the Prediction task. Under the snow condition (right example), the model can still observe some visible objects in the images. Since most objects are severely occluded, the third object noticed becomes more general: the building, compared to the clean inputs, where all the objects are visible vehicles. However, the GPT score is even higher. The example illustrates that even the GPT evaluator can not accurately reflect the language quality generated by VLMs.

[Figure 219]

###### What actions could the ego vehicle take based on <c1,CAM_BACK_LEFT,0.7995,0.5463>? Why take this action and what's the probability?

[Figure 220]

[Figure 221]

[Figure 222]

The ego vehicle could take the action of stopping at the stop sign located at (0.7995, 0.5463) in the image. This action is necessary because the stop sign indicates that the vehicle must come to a complete stop before proceeding. The probability of taking this action is high, as it is a legal requirement to obey traffic signs and ensure safety at intersections.

The ego vehicle could take the action of slowing down or stopping. This is because the object at (0.7995, 0.5463) in the back left view is likely a pedestrian or a stationary object on the road. Taking this action would help avoid a potential collision and ensure the safety of the pedestrian or the vehicle. The probability of needing to take this action is high, as the object is in close proximity to the ego vehicle's path.

- 1. Action Prediction Accuracy: 10/40
- 2. Reasoning & Justification: 15/20
- 3. Probability or Confidence Level: 5/15
- 4. Contextual & Safety Considerations: 5/15
- 5. Conciseness & Clarity: 8/10 Total Score: 43/100

- 1. Action Prediction Accuracy: 30/40
- 2. Reasoning & Justification: 15/20
- 3. Probability or Confidence Level: 12/15
- 4. Contextual & Safety Considerations:12/15
- 5. Conciseness & Clarity: 8/10 Total Score: 77/100

[Figure 223]

Brief Summary: The predicted answer does not accurately align with the correct action of keeping going at the same speed. The reasoning provided is somewhat clear but lacks a direct connection to the scenario. The probability level is mentioned but not fully supported by the context. Contextual awareness and safety considerations are minimal. The answer is relatively concise and clear, earning some points in that category. Overall, the predicted answer falls short in accurately predicting the correct action and lacks strong reasoning and contextual awareness.

Brief Summary: The predicted answer accurately identifies the need to slow down or stop based on the presence of a pedestrian or stationary object in the back left camera view. The reasoning provided is logical and aligns with safety considerations. The probability assessment is reasonable but could be more specific. The answer

demonstrates good contextual awareness and safety considerations. However, it could be more concise and focused on key points.

[Figure 224]

- Figure 28. Example of Qwen2-VL72B for Planning task. Under the text-only condition (right example), the model fails to perceive the motorcycle to stop sign, giving their close distance on the image. When the image is absent, the model guesses the object to be a pedestrian or a stationary object on the road, unexpectedly leading to an answer with higher scores.

[Figure 225]

###### What is the moving status of object <c2,CAM_BACK,0.5443,0.5639>? Please select the correct answer from the following options: A. Turn left. B. Going ahead. C. Turn right.

[Figure 226]

[Figure 227]

[Figure 228]

B. Going ahead. Explanation: The object is positioned in the

B. Going ahead. Explanation: The object is located slightly to the right of the center of the image, which suggests it is aligned with the road. Given the perspective of the image, the object appears to be moving straight along the road. Therefore, the correct answer is B.

center of the lane and appears to be moving straight ahead, as indicated by its alignment with the road markings and the absence of any

visible turning action.

- 1. Answer Correctness: 50 points
- 2. Object Recognition: 10 points
- 3. Object Location and Orientation: 15 points
- 4. Environmental Condition Awareness: 15 points
- 5. Clarity of Reasoning: 10 points Total Score: 100/100

- 1. Answer Correctness: 50 points
- 2. Object Recognition: 5 points
- 3. Object Location and Orientation: 10 points
- 4. Environmental Condition Awareness: 10 points
- 5. Clarity of Reasoning: 5 points Total Score: 80/100

Brief Summary: The answer provided is accurate, correctly identifying the moving status of the object as "Going ahead" based on its position and alignment with the road markings. The explanation is clear, logically structured, and considers environmental conditions that could impact perception. Overall, the answer meets all criteria and earns a perfect score of 100.

Brief Summary: The answer provided accurately identifies the moving status of the object as "Going ahead" based on its position and alignment with the road. The reasoning is clear and logically structured, supporting the correct answer. However, there were some minor inaccuracies in object recognition and environmental condition awareness, which slightly impacted the overall score.

[Figure 229]

[Figure 230]

- Figure 29. Example of GPT-4o for Perception task. Under the bit error condition (right example), the visual information for the target object is completely corrupted but the model can guess the answer correctly. However, the GPT evaluator here captures the nuance in the explanation and assigns a higher score to the more detailed answer with clean image inputs.

[Figure 231]

###### In this scenario, what are safe actions to take for the ego vehicle?

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

- • Maintain Speed: Keep a steady speed to ensure safe distance from vehicles behind.

- • Prepare to Steer Right: As the road curves, be ready to steer smoothly to the right.

- • Monitor Bus Movement: Keep an eye on the bus in the front-right camera to anticipate any merging or stopping.

- • Check Mirrors: Regularly check side mirrors for the position of the motorcycle and other vehicles.

- • Signal Intentions: Use turn signals if planning to change lanes or make a turn to alert surrounding vehicles.

- • Slow Down: Reduce speed to assess the situation and prepare for any necessary maneuvers.

- • Stop or Maneuver: If the object in the front cannot be safely bypassed, stop the vehicle.

- • Check Surroundings: Use all cameras to ensure no pedestrians or vehicles are approaching before making any turns or reversing.

- • Signal Intentions: Use turn signals if planning to maneuver around the object or change lanes.

[Figure 244]

Turn left, decelerate gradually without braking, are safe actions to take for the ego vehicle in this scenario.

Slightly offset to the left, change to the left lane, and turn left.

[Figure 245]

- 1. Action Prediction Accuracy: 35
- 2. Reasoning & Justification: 18
- 3. Probability or Confidence Level: 12
- 4. Contextual Awareness & Safety : 13
- 5. Conciseness and Clarity: 8 Total Score: 86/100

- 1. Action Prediction Accuracy: 30
- 2. Reasoning & Justification: 15
- 3. Probability or Confidence Level: 10
- 4. Contextual Awareness & Safety : 12
- 5. Conciseness and Clarity: 8 Total Score: 75/100

Brief Summary: The predicted answer demonstrates a strong understanding of the scenario, accurately identifying key observations and recommending appropriate actions for the ego vehicle. While there are minor discrepancies and areas for improvement in conciseness and clarity, the response overall aligns well with the correct answer and provides a logical justification for the chosen actions.

Brief Summary:

[Figure 246]

The predicted answer demonstrates a good understanding of the driving scenario and provides relevant actions for the ego vehicle. The reasoning and justification are clear, although some minor details could be further elaborated. The contextual awareness and safety considerations are mostly accurate, with a few areas for improvement. The answer is concise and clear but could benefit from a more focused explanation. Overall, the predicted answer is well-structured and aligns with the correct answer, earning a score of 75.

[Figure 247]

- Figure 30. Example of GPT-4o for Planning task. In both low-light (left example) and normal (right example) conditions, the ground truth is to turn left. However, in the left case, the vehicle is prepared to turn left. Turning right will cause a potential collision with the white sedan on the front right camera. On the right image, the road is empty and the predicted slow-down action will not cause safety issues. However, the GPT evaluator assigns a higher score on the left case, ignoring the potential safety consequences as there are no high-fidelity physical context prompts.

[Figure 248]

###### What actions could the ego vehicle take based on <c1,CAM_BACK,0.6323,0.6314>? Why take this action and what is the probability?

[Figure 249]

| |
|---|

Based on the image, the ego vehicle is at an intersection with a motorcycle and a car visible behind it.

- 1. Maintain Current Speed: The motorcycle and car appear to be at a safe distance behind the ego vehicle. There are no immediate obstacles or hazards directly in the path. Probability: High, as maintaining speed is often the default action when no immediate threats are present.

- 2. Prepare to Stop: If the ego vehicle is approaching a stop sign or traffic light (not visible in the image), it may need to slow down or stop. Probability: Moderate, depending on traffic signals or signs not visible in the image.

- 3. Change Lanes: If the ego vehicle needs to turn or avoid an obstacle ahead (not visible in the image), it might consider changing lanes. Probability: Low, as there is no immediate indication of a need to change lanes.

[Figure 250]

Conclusion: The most likely action is to maintain current speed, given the apparent safe distance from vehicles behind and no visible immediate obstacles.

The action is to keep going at the same speed. The reason is that there is no safety issue, and the probability is high.

[Figure 251]

- 1. Action Prediction Accuracy (40 points): Award full points only for exact or highly similar action matches. Deduct points for inaccuracies or actions that do not match the correct answer, especially if they could compromise safety or appropriateness in context.
- 2. Reasoning & Justification (20 points): Deduct points if the reasoning lacks clarity, omits relevant details, or includes contradictions. The explanation should justify the action in a way that is suitable for the scenario provided.
- 3. Probability or Confidence Level (15 points): Deduct points if the probability level is missing, implausible, or does not align with the action or reasoning provided.
- 4. Contextual Awareness & Safety (15 points): Deduct points for failing to consider contextual factors that may impact the ego vehicle's decision, especially if they could lead to unsafe actions.
- 5. Conciseness & Clarity (10 points): Deduct points for verbosity, ambiguity, or lack of focus that could hinder quick comprehension. Assign 0 points if no explanation is provided.

Rate my answer based on the correct answer out of 100, with higher scores indicating that the answer is closer to the correct answer, and you should be accurate to single digits like 62, 78, 41, etc. Output the number only.

Here is the correct answer: <GT> Here is the model answer: <ANS>

[Figure 252]

Here is the question: <QUESTION> Here is the ground truth object visual description: <DESC> Here is the correct answer: <GT> Here is the model answer: <ANS>

###### Total Score: 73/100 Total Score: 92/100

- Figure 31. Examples of the GPT evaluator with different prompts. With the same question and answer, the GPT evaluator assigns different scores based on the rubrics. Without detailed information on the physical context, the scoring point, the evaluator fails to capture critical information for driving decisions in the answer but simply based on semantic similarities.

[Figure 253]

###### What actions could the ego vehicle take based on <c1,CAM_BACK,0.5073,0.5778>? Why take this action and what is the probability?

[Figure 254]

Based on the image and the coordinates provided, the ego vehicle is facing a situation where a large truck is making a turn and partially blocking the road. The coordinates (0.5073, 0.5778) likely correspond to the position of the truck, which is a critical object in this scenario.

Possible Actions:

- 1. Slow Down or Stop: The truck is obstructing the path, and there is an oncoming vehicle. Slowing down or stopping allows the truck to complete its turn safely and avoids potential collisions.

- 2. Wait for the Truck to Clear: Once the truck has cleared the path, the ego vehicle can proceed. This ensures that the road is clear and safe to continue driving

- 3. Prepare to Yield: If the oncoming vehicle has the right of way, yielding ensures compliance with traffic rules and safety.

[Figure 255]

Conclusion: The most immediate and probable action is to slow down or stop to allow the truck to complete its turn safely. This action minimizes risk and ensures the safety of all vehicles involved.

###### The action is to keep going at the same speed. The reason is that there is no safety issue, and the probability is high

[Figure 256]

<RUBRICS> Here is the correct answer: <GT> Here is the model answer: <ANS>

<RUBRICS> Here is the question: <QUESTION> Here is the correct answer: <GT> Here is the model answer: <ANS>

<RUBRICS> Here is the question: <QUESTION> Here is the ground truth object visual description: <DESC> Here is the correct answer: <GT> Here is the model answer: <ANS> -------------------------------------------------------------------

-------------------------------------------------------------------

-------------------------------------------------------------------

[Figure 257]

- 1. Action Prediction Accuracy: 35/40
- 2. Reasoning & Justification: 18/20
- 3. Probability or Confidence Level: 12/15
- 4. Contextual Awareness and Safety: 13/15
- 5. Conciseness & Clarity: 8/10

- 1. Action Prediction Accuracy: 25/40
- 2. Reasoning & Justification: 15/20
- 3. Probability or Confidence Level: 10/15
- 4. Contextual Awareness & Safety: 10/15
- 5. Conciseness & Clarity: 5/10

- 1. Action Prediction Accuracy: 20/40
- 2. Reasoning & Justification: 15/20
- 3. Probability or Confidence Level: 10/15
- 4. Contextual Awareness & Safety: 10/15
- 5. Conciseness & Clarity: 5/10

Total Score: 86/100

###### Total Score: 65/100 Total Score: 60/100

- Figure 32. Examples of the GPT evaluators with different information. We gradually add more information about the question and the visual description of the target objects. The evaluator gives a more accurate score based on more information.

- Table 8. Detailed GPT score results of MCQs for the Perception task. “Clean” represents clean image inputs. “T.O.” represents textonly evaluation. The “Corrupt” settings range from weather conditions, external disturbances, sensor failures, motion blur, and transmission errors. The benchmarked VLMs include commercial, open-sourced, and driving specialist models, respectively.

[Figure 258]

Method

Clean•

T.O.•

Brightness•

Dark•

Snow•

Fog•

Rain•

LensObstacle•

WaterSplash•

CameraCrash•

FrameLost•

Saturate•

MotionBlur•

ZoomBlur•

BitError•

ColorQuant•

H.265Compression• Human 47.67 - 43.33 26.67 45.00 18.33 40.00 35.33 37.33 42.00 20.00 25.00 43.00 20.00 33.67 25.33 31.33 GPT-4o 41.87 43.59 43.84 44.82 45.18 44.30 46.20 45.69 44.10 38.12 40.08 39.32 41.40 36.67 37.54 38.22 39.37

[Figure 259]

Phi-3 35.51 32.65 35.28 34.15 38.88 38.22 37.70 38.39 36.75 34.93 33.89 37.53 37.80 39.72 36.49 37.15 37.03 Phi-3.5 40.22 38.33 39.22 36.46 41.41 43.04 41.40 40.66 40.83 37.59 36.91 39.86 40.33 42.30 36.71 41.49 41.48

LLaVA-1.57B 32.40 32.68 32.48 32.95 31.95 32.43 32.30 32.88 32.18 32.93 31.63 32.50 32.43 32.93 32.48 32.18 32.63 LLaVA-1.513B 33.58 33.25 33.25 33.25 33.15 33.50 33.53 32.95 32.93 33.48 33.40 33.25 33.45 33.68 33.33 33.25 33.38 LLaVA-NeXT 32.98 4.20 33.85 20.43 11.62 16.58 27.33 18.24 32.30 26.80 20.83 23.50 34.00 17.75 24.50 18.03 26.24

InternVL8B 46.60 52.46 43.65 44.15 43.58 46.02 42.38 41.48 43.38 45.32 49.08 43.98 41.30 41.50 38.25 44.84 42.13

Oryx 17.98 20.87 16.48 16.88 16.63 16.79 14.31 16.35 15.85 16.49 21.44 21.38 16.36 21.04 17.65 18.13 19.51 Qwen2VL7B 42.64 37.76 43.08 37.29 39.72 41.67 40.87 40.69 39.89 39.75 39.17 40.85 41.32 39.62 34.28 39.90 41.20

Qwen2VL72B 38.15 21.53 36.24 37.77 35.91 35.78 37.13 38.14 38.97 29.48 25.63 36.87 36.91 37.01 30.90 36.05 41.48

Dolphin 6.50 8.35 10.18 11.08 10.70 9.53 10.58 9.93 9.80 10.08 9.95 11.20 9.85 10.10 8.80 10.00 11.10 DriveLM 22.38 12.45 20.78 25.30 18.98 24.43 25.95 22.03 21.03 21.95 16.28 19.38 22.98 20.93 19.90 16.25 26.48

- Table 9. Detailed Accuracy score results of MCQs for the Perception task. “Clean” represents clean image inputs. “T.O.” represents text-only evaluation. The “Corrupt” settings range from weather conditions, external disturbances, sensor failures, motion blur, and transmission errors. The benchmarked VLMs include commercial, open-sourced, and driving specialist models, respectively.

[Figure 260]

H.265Compression•

LensObstacle•

CameraCrash•

WaterSplash•

MotionBlur•

ColorQuant•

FrameLost•

ZoomBlur•

Brightness•

BitError•

Saturate•

Clean•

Snow•

Dark•

Rain•

T.O.•

Fog•

Method

[Figure 261]

Human 93.33 - 80.00 53.33 80.00 33.33 80.00 66.67 73.33 80.00 40.00 46.67 80.00 40.00 60.00 46.67 53.33 GPT-4o 59.00 59.50 60.50 63.50 59.00 61.00 59.50 58.00 59.00 50.00 51.50 56.50 57.50 52.00 51.00 54.00 57.50

Phi3 54.50 17.50 55.00 33.00 50.00 55.00 59.00 56.00 57.50 32.50 23.50 58.00 57.50 49.00 36.00 49.50 57.50 Phi-3.5 56.50 58.50 59.00 58.00 58.00 59.00 58.50 60.00 59.50 58.50 57.00 59.50 59.00 59.50 57.50 60.50 58.50

LLaVA-1.57B 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 LLaVA-1.513B 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 50.00 LLaVA-NeXT 55.00 34.50 53.50 42.50 41.50 44.00 55.00 42.00 52.00 31.00 26.50 50.50 53.50 39.00 44.50 37.50 49.00

InternVL8B 56.50 23.50 57.50 61.00 62.00 59.50 59.50 59.50 60.00 50.50 31.00 56.50 56.00 59.00 60.00 58.50 58.50

Oryx 51.00 19.00 53.50 50.50 52.00 52.50 52.50 50.00 48.00 32.50 29.00 52.00 48.00 36.00 50.50 52.00 52.00 Qwen2VL7B 59.00 56.50 60.00 59.00 60.00 59.50 59.00 59.00 59.00 58.00 56.00 59.00 59.50 55.00 54.50 57.00 59.00

Qwen2VL72B 60.00 23.50 58.00 60.00 58.50 59.50 61.50 60.00 59.50 39.00 29.50 60.00 59.00 55.50 55.50 58.50 61.00

Dolphin 7.00 9.00 5.50 5.50 5.50 6.00 4.50 6.50 5.50 5.50 7.50 6.50 6.00 5.50 6.00 5.50 5.50 DriveLM 40.00 23.00 37.00 45.00 35.00 46.50 45.50 40.50 37.50 39.50 29.50 34.00 42.00 36.50 36.00 30.50 47.00

- Table 10. Detailed GPT score results of MCQs for the Behavior task. “Clean” represents clean image inputs. “T.O.” represents textonly evaluation. The “Corrupt” settings range from weather conditions, external disturbances, sensor failures, motion blur, and transmission errors. The benchmarked VLMs include commercial, open-sourced, and driving specialist models, respectively.

[Figure 262]

Method

Clean•

T.O.•

Brightness•

Dark•

Snow•

Fog•

Rain•

LensObstacle•

WaterSplash•

CameraCrash•

FrameLost•

Saturate•

MotionBlur•

ZoomBlur•

BitError•

ColorQuant•

H.265Compression• Human 65.00 - 40.00 40.67 66.67 51.00 73.33 53.33 56.00 60.00 36.67 54.00 67.67 34.67 53.33 73.33 53.33 GPT-4o 45.40 50.03 46.27 49.20 42.54 41.79 46.55 45.30 47.78 45.28 40.44 47.89 39.91 32.25 48.40 50.35 41.07

[Figure 263]

Phi-3 45.20 40.91 45.98 44.48 47.91 45.17 47.45 44.22 44.02 43.65 44.01 43.51 42.48 41.05 43.83 46.60 44.30 Phi-3.5 36.75 39.16 37.15 38.14 37.19 39.53 38.40 36.79 37.36 36.83 37.98 39.09 37.70 38.91 38.27 38.23 36.85

LLaVA-1.57B 13.60 14.91 12.79 12.83 15.57 12.63 14.06 13.99 12.79 14.68 13.65 13.12 13.55 13.83 13.98 13.44 13.48 LLaVA-1.513B 32.99 32.79 33.34 33.10 33.10 31.96 32.44 32.56 32.49 31.87 31.55 31.84 33.17 31.14 33.40 31.78 33.72 LLaVA-NeXT 48.16 11.92 48.84 38.82 15.90 39.13 47.07 20.72 47.02 48.20 36.67 39.69 47.36 39.60 46.99 28.13 47.55

InternVL8B 54.58 20.14 32.54 36.95 42.10 56.72 31.53 31.09 41.65 50.17 32.77 43.66 34.82 34.90 50.41 50.78 41.66

Oryx 33.92 23.94 34.19 37.77 33.02 32.89 32.56 34.16 34.83 34.51 34.82 34.05 33.95 29.61 35.25 33.27 32.33 Qwen2VL7B 49.07 46.93 46.81 48.75 48.04 47.64 48.45 46.80 49.24 47.95 47.19 49.58 48.83 41.27 49.72 47.07 47.90

Qwen2VL72B 51.26 39.46 52.13 51.24 51.64 49.75 53.18 52.46 50.81 51.25 47.44 51.22 48.87 35.72 52.76 49.77 48.52

Dolphin 8.81 7.11 7.17 9.54 9.02 6.48 8.05 7.95 7.10 9.29 8.94 8.02 8.02 9.42 8.37 10.07 6.32 DriveLM 42.78 27.83 47.18 36.30 40.70 39.18 40.93 43.30 40.98 39.95 38.23 40.08 45.68 38.88 41.10 33.50 39.65

- Table 11. Detailed Accuracy score results of MCQs for the Behavior task. “Clean” represents clean image inputs. “T.O.” represents text-only evaluation. The “Corrupt” settings range from weather conditions, external disturbances, sensor failures, motion blur, and transmission errors. The benchmarked VLMs include commercial, open-sourced, and driving specialist models, respectively.

[Figure 264]

H.265Compression•

LensObstacle•

CameraCrash•

WaterSplash•

MotionBlur•

ColorQuant•

FrameLost•

ZoomBlur•

Brightness•

BitError•

Saturate•

Clean•

Snow•

Dark•

Rain•

T.O.•

Fog•

Method

[Figure 265]

Human 66.67 - 40.00 46.67 66.67 53.33 73.33 53.33 53.33 60.00 40.00 53.33 66.67 33.33 53.33 73.33 53.33 GPT-4o 25.50 24.00 25.50 25.00 21.50 23.50 25.00 26.00 24.00 26.50 28.50 24.50 23.50 24.00 26.00 22.50 21.50

Phi3 26.50 30.00 29.50 29.50 28.00 29.50 28.50 27.00 30.00 32.50 31.00 28.50 29.50 23.50 27.50 31.50 27.50 Phi-3.5 36.50 40.00 37.00 36.00 37.00 38.50 38.00 36.50 35.50 37.00 39.00 39.00 37.50 36.50 39.00 36.50 36.00

LLaVA1.57B 10.00 9.50 8.50 8.00 8.00 7.50 8.00 8.50 8.00 11.00 10.00 7.50 9.00 7.50 11.00 8.50 8.00 LLaVA1.513B 32.50 33.00 33.00 33.00 32.50 32.50 32.00 32.50 32.50 32.00 32.50 32.50 33.00 31.00 34.00 31.00 33.50 LLaVA-NeXT 23.00 15.00 23.00 24.00 22.00 23.00 22.50 24.50 25.50 27.50 24.50 26.50 24.00 23.00 24.00 21.50 25.50

InternVL8B 27.50 21.50 9.00 14.50 20.50 25.50 13.00 11.50 15.00 25.00 17.50 21.00 11.50 12.00 28.00 23.50 18.50

Oryx 21.00 21.00 21.50 21.50 21.50 20.50 21.50 21.50 21.50 21.50 22.00 21.50 22.00 21.00 22.50 21.50 21.50 Qwen2VL7B 30.00 23.00 29.00 28.00 25.00 28.50 27.50 25.00 28.50 31.50 28.50 33.50 26.00 21.50 27.00 28.50 30.00

Qwen2VL72B 23.00 36.50 25.50 24.50 25.50 22.00 29.50 26.00 22.50 27.00 25.00 26.00 22.50 22.00 28.50 23.50 23.50

Dolphin 0.50 3.50 1.50 0.00 1.00 0.00 0.00 1.00 1.00 1.50 2.50 0.50 1.00 1.00 1.00 0.50 0.50 DriveLM 44.00 25.50 48.50 37.00 41.50 40.00 42.50 43.50 41.00 41.00 39.50 40.50 46.50 40.00 43.00 35.00 41.50

- Table 12. Detailed GPT score results of open-ended questions for the Perception task. “Clean” represents clean image inputs. “T.O.” represents text-only evaluation. The “Corrupt” settings range from weather conditions, external disturbances, sensor failures, motions, and transmission errors. The benchmarked VLMs include commercial, open-sourced, and driving specialist models, respectively.

[Figure 266]

Method

Clean•

T.O.•

Brightness•

Dark•

Snow•

Fog•

Rain•

LensObstacle•

WaterSplash•

CameraCrash•

FrameLost•

Saturate•

MotionBlur•

ZoomBlur•

BitError•

ColorQuant•

H.265Compression•

GPT-4o 28.87 29.37 29.51 28.15 30.19 28.89 28.63 28.49 29.42 27.12 28.16 27.96 29.82 32.79 27.25 26.44 29.89

Phi-3 10.26 10.38 10.44 11.27 10.97 10.81 10.99 10.64 11.32 10.67 10.01 11.03 11.37 12.31 11.38 10.33 10.58 Phi-3.5 14.83 18.20 14.24 13.08 15.36 16.09 18.04 16.53 14.39 16.42 14.47 12.88 14.97 12.96 14.74 13.27 18.66

LLaVA-1.57B 14.03 11.94 13.53 13.31 13.31 13.61 13.75 13.91 13.48 13.95 12.90 12.98 13.35 13.05 13.36 12.83 14.28 LLaVA-1.513B 13.13 11.50 12.72 13.39 12.86 13.38 13.05 13.23 13.59 15.13 14.98 12.39 13.60 13.85 12.72 13.43 12.26 LLaVA-NeXT 15.33 23.53 15.49 14.95 16.61 15.62 16.05 15.66 15.66 15.19 16.04 15.16 16.06 17.92 15.27 15.10 15.86

InternVL8B 18.12 14.73 20.05 18.82 23.14 14.28 27.75 23.43 26.67 11.88 17.11 26.26 22.02 27.25 11.54 29.57 29.77

Oryx 16.07 16.07 15.72 13.46 11.98 15.54 13.73 17.75 15.86 13.35 13.29 14.45 13.74 13.90 13.17 13.40 14.66 Qwen2VL7B 15.33 32.56 16.75 14.95 15.16 15.66 15.02 15.66 15.48 17.31 17.65 14.52 15.29 15.20 15.17 14.63 17.89

Qwen2VL72B 22.10 13.88 20.59 17.00 14.64 18.77 20.00 24.19 20.34 17.55 15.45 19.13 16.10 18.58 15.97 18.78 16.36

Dolphin 12.68 13.67 12.07 10.34 12.37 11.46 11.73 12.33 11.04 12.34 11.39 11.47 11.34 10.17 11.40 11.35 11.65 DriveLM 11.32 5.05 11.30 10.03 11.21 9.71 10.22 10.71 11.01 11.14 10.13 9.38 10.97 9.03 11.39 10.50 10.73

- Table 13. Detailed GPT score results of the open-ended questions for Prediction. “Clean” represents clean image inputs. “T.O.” represents text-only evaluation. The “Corrupt” settings range from weather conditions, external disturbances, sensor failures, motions, and transmission errors. The benchmarked VLMs include commercial, open-sourced, and driving specialist models, respectively.

[Figure 267]

H.265Compression•

LensObstacle•

CameraCrash•

WaterSplash•

MotionBlur•

ColorQuant•

FrameLost•

ZoomBlur•

Brightness•

BitError•

Saturate•

Clean•

Snow•

Dark•

Rain•

T.O.•

Fog•

Method

GPT-4o 51.30 49.05 52.15 50.28 47.97 49.66 51.39 50.49 53.30 46.62 45.95 49.18 51.90 49.75 48.59 47.56 54.36

Phi-3 40.11 22.61 45.21 26.28 35.54 44.05 34.87 40.90 33.56 44.26 38.74 41.39 35.08 36.46 32.25 32.82 37.57 Phi-3.5 45.13 4.92 46.57 9.67 48.02 45.08 52.16 42.75 47.02 47.18 24.66 23.26 49.16 28.38 41.39 18.02 49.87

LLaVA-1.57B 22.02 14.64 24.79 20.95 15.97 15.30 18.98 16.28 24.16 6.11 13.90 20.30 25.20 10.56 11.10 15.61 23.92 LLaVA-1.513B 36.98 23.98 36.00 35.59 40.51 39.23 38.90 38.11 38.92 36.25 36.54 37.57 38.10 39.74 34.28 37.16 36.31 LLaVA-NeXT 35.07 28.36 37.15 35.31 37.59 37.62 35.44 37.00 35.87 36.25 30.10 40.56 34.66 39.36 31.74 34.07 35.66

InternVL8B 45.52 48.89 45.73 40.71 35.75 38.43 33.18 38.69 40.71 45.00 45.75 30.03 39.52 36.55 40.12 28.40 30.31

Oryx 48.13 12.77 49.52 44.33 47.67 45.77 47.20 45.90 50.30 42.18 44.59 42.26 48.21 52.54 45.56 43.77 49.61 Qwen2VL7B 37.89 37.77 40.82 35.90 38.92 44.15 40.15 41.89 41.57 36.61 35.87 41.25 40.89 39.23 36.69 40.52 38.84

Qwen2VL72B 49.35 5.57 43.89 43.25 43.74 44.49 45.57 41.61 47.46 40.89 34.75 39.38 48.21 51.15 40.49 40.82 46.67

Dolphin 32.66 39.98 29.85 32.31 24.64 29.92 31.38 33.41 31.79 29.05 30.93 30.49 31.59 26.38 30.13 25.64 30.62 DriveLM 44.33 4.70 46.82 43.90 42.33 35.84 44.13 44.00 42.59 46.25 33.56 29.69 42.15 19.00 38.20 44.33 42.87

- Table 14. Detailed ROUGE-L score results of open-ended questions for the Predicion task. “Clean” represents clean image inputs. “T.O.” represents text-only evaluation. The “Corrupt” settings range from weather conditions, external disturbances, sensor failures, motion blur, and transmission errors. The benchmarked VLMs include commercial, open-sourced, and driving specialist models, respectively.

[Figure 268]

Method

Clean•

T.O.•

Brightness•

Dark•

Snow•

Fog•

Rain•

LensObstacle•

WaterSplash•

CameraCrash•

FrameLost•

Saturate•

MotionBlur•

ZoomBlur•

BitError•

ColorQuant•

H.265Compression•

GPT-4o 19.74 18.58 19.77 19.58 19.67 19.71 19.69 20.22 19.94 19.89 19.66 19.70 19.39 19.48 19.47 19.90 19.67

Phi3 17.76 14.71 17.21 21.55 25.81 15.28 23.77 18.56 19.27 17.59 17.42 16.90 16.75 14.36 25.23 26.88 22.55 Phi-3.5 19.36 18.76 18.37 5.78 17.28 15.74 18.24 18.38 17.98 17.10 12.63 10.84 17.85 12.46 17.12 8.50 17.03

LLaVA1.57B 21.18 23.21 21.75 22.64 22.21 22.17 22.19 19.61 22.05 22.53 22.36 22.65 21.71 20.28 22.46 22.57 21.17 LLaVA1.513B 24.12 24.79 24.03 24.04 24.09 24.37 24.14 24.18 24.35 23.91 23.91 23.84 24.04 24.59 23.93 24.31 24.12

qwen2-7b 25.49 24.15 25.64 24.80 25.21 24.82 24.86 25.17 25.64 25.62 25.99 24.74 25.11 23.73 25.09 25.31 25.35 Qwen2VL72B 23.42 16.10 20.05 18.46 17.97 19.08 19.59 18.68 18.51 22.05 21.17 17.97 18.98 19.03 18.89 19.35 18.13 LLaVA1.6-7b 16.09 13.93 16.29 16.99 17.02 17.14 16.54 16.27 17.08 16.25 16.59 18.05 16.69 16.93 16.58 17.33 16.46

InternVL8B 13.92 13.25 20.57 15.14 14.67 13.64 15.15 16.39 15.45 13.92 16.12 14.47 16.29 14.79 14.02 14.26 14.07

Oryx 21.03 15.43 18.79 17.45 16.99 18.04 18.19 17.81 17.68 20.38 19.56 16.54 18.21 17.84 18.20 17.92 17.03 Qwen2VL7B 25.49 23.15 25.64 24.80 25.21 24.82 24.86 25.17 25.64 25.62 25.99 24.74 25.11 23.73 25.09 25.31 25.35

Qwen2VL72B 23.42 16.10 20.05 18.46 17.97 19.08 19.59 18.68 18.51 22.05 21.17 17.97 18.98 19.03 18.89 19.35 18.13

Dolphin 25.18 23.64 24.71 24.84 25.95 24.98 26.19 25.09 25.18 24.52 24.75 24.48 24.99 24.63 24.87 24.34 24.95 DriveLM 40.00 23.00 37.00 45.00 35.00 46.50 45.50 40.50 37.50 39.50 29.50 34.00 42.00 36.50 36.00 30.50 47.00

- Table 15. Detailed GPT score results of the open-ended questions for the Planning. “Clean” represents clean image inputs. “T.O.” represents text-only evaluation. The “Corrupt” settings range from weather conditions, external disturbances, sensor failures, motions, and transmission errors. The benchmarked VLMs include commercial, open-sourced, and driving specialist models, respectively.

[Figure 269]

H.265Compression•

LensObstacle•

CameraCrash•

WaterSplash•

MotionBlur•

ColorQuant•

FrameLost•

ZoomBlur•

Brightness•

BitError•

Saturate•

Clean•

Snow•

Dark•

Rain•

T.O.•

Fog•

Method

GPT-4o 75.75 73.21 77.56 74.33 73.00 76.58 76.53 75.14 74.65 84.08 74.54 74.84 77.34 73.09 74.30 76.36 76.82

Phi-3 60.03 46.88 61.48 59.14 64.59 64.81 63.47 61.83 62.31 58.32 54.63 63.04 61.61 58.93 58.11 64.04 63.40 Phi-3.5 31.91 46.30 30.57 28.64 32.88 28.38 34.70 30.88 29.53 23.16 24.53 25.47 32.73 23.00 28.24 23.42 29.31

LLaVA-1.57B 29.15 32.45 31.52 31.49 32.58 32.42 31.00 29.81 31.72 33.70 35.95 29.93 31.20 30.65 30.05 30.00 30.61 LLaVA-1.513B 34.26 38.85 33.63 34.36 35.06 39.43 35.18 33.01 34.80 37.61 39.32 32.77 34.74 32.99 33.75 33.33 33.67 LLaVA-NeXT 45.27 27.58 45.64 44.54 43.55 44.17 45.08 44.62 44.21 45.69 44.51 41.17 45.30 44.57 43.67 43.57 45.13

InternVL8B 53.27 34.56 54.70 60.02 63.89 53.69 60.64 54.31 56.68 52.14 46.12 54.44 55.94 54.93 49.08 57.02 55.13

Oyrx 53.57 48.26 55.46 58.60 58.91 57.93 55.04 57.01 58.35 52.28 51.05 55.76 55.88 53.63 52.94 56.00 57.57 Qwen2VL7B 57.04 41.66 54.19 58.37 52.70 58.18 55.85 53.98 55.64 54.71 53.18 51.93 55.22 51.73 56.79 53.16 56.04

Qwen2VL72B 61.30 53.35 62.01 65.31 66.69 68.15 67.83 65.67 65.26 57.42 56.34 62.06 64.20 61.66 58.23 59.86 65.31

Dolphin 52.91 60.98 51.85 55.39 53.09 54.78 53.92 51.79 53.57 55.73 57.81 55.78 51.42 50.95 53.35 54.89 52.17 DriveLM 68.71 65.24 67.25 67.52 65.72 63.08 69.60 69.04 67.97 67.85 66.47 66.25 67.93 70.17 68.46 68.30 68.59

[Figure 270]

- Table 16. Detailed ROUGE-L score results of open-ended questions for the Planning task. “Clean” represents clean image inputs. “T.O.” represents text-only evaluation. The “Corrupt” settings range from weather conditions, external disturbances, sensor failures, motion blur, and transmission errors. The benchmarked VLMs include commercial, open-sourced, and driving specialist models, respectively.

H.265Compression•

LensObstacle•

CameraCrash•

WaterSplash•

MotionBlur•

ColorQuant•

FrameLost•

ZoomBlur•

Brightness•

BitError•

Saturate•

Clean•

Snow•

Dark•

Rain•

T.O.•

Fog•

Method

GPT-4o 6.54 6.42 6.54 6.34 6.54 6.54 6.52 6.40 6.52 6.47 6.35 6.39 6.39 6.40 6.35 6.37 6.67

Phi3 9.39 10.05 9.56 9.26 9.51 9.62 9.47 9.65 9.60 10.01 10.01 9.90 9.40 9.42 9.07 10.02 9.36 Phi-3.5 6.19 8.27 6.52 7.97 6.30 6.10 6.12 6.57 6.59 7.18 7.59 5.98 6.17 7.87 6.64 6.04 6.37

LLaVA1.57B 8.76 9.98 8.93 8.99 9.18 8.96 8.76 9.00 9.04 9.06 9.24 8.88 8.94 9.27 9.25 9.16 8.85 LLaVA1.513B 8.06 8.68 8.14 8.09 8.10 8.06 8.10 8.03 7.99 8.68 8.64 8.43 8.10 8.04 8.39 8.33 8.24

LLaVA-NeXT 5.69 7.75 5.72 5.66 5.50 5.49 5.60 5.67 5.60 5.62 5.88 5.59 5.64 5.45 5.64 5.64 5.58 InternVL8B 4.06 9.45 10.37 11.65 12.07 4.05 10.81 10.97 9.88 4.16 9.29 11.02 10.04 9.23 4.03 10.24 9.65

Oryx 21.03 15.43 18.79 17.45 16.99 18.04 18.19 17.81 17.68 20.38 19.56 16.54 18.21 17.84 18.20 17.92 17.03 Qwen2VL7B 9.61 8.33 9.31 8.70 8.14 8.33 8.69 8.32 8.82 9.15 8.81 8.69 8.64 9.05 9.13 8.77 7.92 Qwen2VL72B 12.26 13.17 12.13 11.60 11.45 10.97 11.62 11.50 11.67 12.01 11.43 12.13 11.85 11.55 12.06 11.84 11.93

Dolphin 12.90 14.82 12.90 13.01 12.61 12.82 12.45 12.81 12.74 13.56 13.91 12.82 12.86 12.89 13.22 13.29 12.69 DriveLM 53.12 46.83 51.55 51.26 49.26 46.74 52.02 53.38 52.77 52.44 50.91 48.66 53.28 52.20 49.88 52.79 51.43

##### E. Broader Impact & Limitations

In this section, we discuss the broader implications of our study and acknowledge its potential limitations.

###### E.1. Broader Impact

Our research focuses on evaluating the reliability of VLMs in autonomous driving, emphasizing three critical perspectives: the model’s robustness, data quality, and evaluation metrics. The findings reveal a concerning tendency of VLMs to fabricate explanations, particularly under conditions of visual degradation. This issue is not limited to autonomous driving but is likely relevant to other VLMembodied systems, such as robotics and other safety-critical cyber-physical systems. For example, VLM-based robots could generate misleading task explanations or actions based on hallucinated understanding, potentially compromising safety and operational reliability.

The implications of our work extend beyond autonomous driving, calling for a reassessment of benchmark and metric designs to better evaluate the trustworthiness of VLMs in real-world applications. Current benchmarks often fail to account for the complexity and variability of real-world scenarios, particularly in environments where system malfunctions could result in life-threatening consequences. Our study highlights the urgency of addressing these gaps to develop robust, reliable, and interpretable VLMs that can be safely integrated into such systems.

Finally, the design of benchmarks, testbeds, and evaluation metrics that accurately capture the reliability and safety implications of applying VLMs to real-world physical systems is of paramount importance. These tools must go beyond traditional performance metrics to consider the nuanced requirements of autonomous systems, such as contextual understanding, interpretability, and robustness against adversarial conditions.

###### E.2. Potential Limitations

While this study provides valuable insights, it is essential to recognize its limitations to contextualize the findings:

- • The experimental results are derived exclusively from the DriveLM [59] dataset due to the prohibitive computational cost of large-scale VLM inference and GPT-based evaluations. While DriveLM is a comprehensive dataset, its scope may limit the generalizability of our findings to other driving benchmarks or real-world settings. Future work should expand the analysis to additional datasets and environments to validate the observed trends.
- • The lack of detailed contextual data in the DriveLM dataset poses a constraint on our evaluations. For instance, the GPT-based assessments rely on limited visual descriptions of key objects, which may not comprehensively capture the broader situational context required for accurate and nuanced evaluations. Expanding datasets

- to include richer temporal and spatial contexts could improve evaluation fidelity.
- • This study primarily investigates the language-based explanations generated by VLMs. While these insights are crucial for understanding VLM reliability, it remains unclear whether the observations generalize to action models that generate vehicle trajectories or other nonlanguage outputs. Exploring how VLMs’ visual grounding affects action-oriented tasks, such as trajectory prediction or manipulation control, represents an important direction for future research.
- • The study evaluates a finite set of 12 VLMs across specific tasks, metrics, and settings. Although the insights are significant, the scalability of these findings to emerging VLM architectures or more diverse driving scenarios warrants further investigation.

By addressing these limitations in future studies, we aim to build a more comprehensive understanding of the challenges and opportunities in applying VLMs to autonomous driving and other safety-critical domains.

##### F. Public Resource Used

In this section, we acknowledge the use of the following public resources, during the course of this work:

- • nuScenes4 . . . . . . . . . . . . . . . . . . . . . . . . CC BY-NC-SA 4.0
- • nuScenes-devkit5 . . . . . . . . . . . . . . . . . Apache License 2.0
- • DriveLM6 . . . . . . . . . . . . . . . . . . . . . . . .Apache License 2.0
- • Phi-3.5-vision-instruct7 . . . . . . . . . . . . . . . . . .MIT License
- • Phi-3-mini-4k-instruct8 . . . . . . . . . . . . . . . . . .MIT License
- • LLaVA-1.5-7B-hf9 . . . . . . . . . . . . . . . .Apache License 2.0
- • LLaVA-1.5-13B-hf10 . . . . . . . . . . . . . .Apache License 2.0
- • LLaVA-v1.6-mistral-7B11 . . . . . . . . . Apache License 2.0
- • InternVL2-8B12 . . . . . . . . . . . . . . . . . . Apache License 2.0
- • Qwen2-VL-7B13 . . . . . . . . . . . . . . . . . Apache License 2.0
- • Qwen2-VL-72B14 . . . . . . . . . . . . . . . . Apache License 2.0

- 4https://www.nuscenes.org/nuscenes.
- 5https://github.com/nutonomy/nuscenes-devkit.
- 6https://github.com/OpenDriveLab/DriveLM
- 7https : / / huggingface . co / microsoft / Phi - 3 . 5 -

vision-instruct

- 8https://huggingface.co/microsoft/Phi-3-mini-

4k-instruct

- 9https://huggingface.co/llava-hf/llava-1.5-7b-

hf

- 10https://huggingface.co/llava- hf/llava- 1.5-

13b-hf

- 11https://huggingface.co/liuhaotian/llava-v1.6-

mistral-7b

- 12https://huggingface.co/OpenGVLab/InternVL2-8B
- 13https : / / huggingface . co / Qwen / Qwen2 - VL - 7B -

Instruct

- 14https : / / huggingface . co / Qwen / Qwen2 - VL - 72B -

Instruct

##### References

- [1] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024. 2, 3, 7, 9, 10, 18, 20
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 20

- [3] Mousumi Akter, Naman Bansal, and Shubhra Kanti Karmaker. Revisiting automatic evaluation of extractive summarization task: Can we do better than rouge? In Findings of the Association for Computational Linguistics, pages 1547– 1560, 2022. 3
- [4] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In European Conference on Computer Vision, pages 382–398, 2016. 3
- [5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 2, 3, 18
- [6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 3
- [7] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11621–11631, 2020. 4
- [8] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024. 3
- [9] Kai Chen, Yanze Li, Wenhua Zhang, Yanxin Liu, Pengxiang Li, Ruiyuan Gao, Lanqing Hong, Meng Tian, Xinhai Zhao, Zhenguo Li, Dit-Yan Yeung, Huchuan Lu, and Xu Jia. Automated evaluation of large vision-language models on self-driving corner cases. arXiv preprint arXiv:2404.10595,

2024. 2

- [10] Long Chen, Oleg Sinavski, Jan H¨unermann, Alice Karnsund, Andrew James Willmott, Danny Birch, Daniel Maund, and

- Jamie Shotton. Driving with llms: Fusing object-level vector modality for explainable autonomous driving. In IEEE International Conference on Robotics and Automation, pages 14093–14100, 2024. 3, 7
- [11] Yuntao Chen, Yuqi Wang, and Zhaoxiang Zhang. Drivinggpt: Unifying driving world modeling and planning with multi-modal autoregressive transformers. arXiv preprint arXiv:2412.18607, 2024. 3
- [12] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 2, 7, 10, 18, 20
- [13] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 2
- [14] DriveLM contributors. Drivelm: Driving with graph visual question answering. https://github.com/ OpenDriveLab/DriveLM, 2023. 4, 7
- [15] Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, and Ziran Wang. Drive as you speak: Enabling human-like interaction with large language models in autonomous vehicles. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 902–909, 2024. 3
- [16] Thierry Deruyttere, Simon Vandenhende, Dusan Grujicic, Luc Van Gool, and Marie-Francine Moens. Talk2car: Taking control of your self-driving car. In Conference on Empirical Methods in Natural Language Processing, pages 2088–2098,

2019. 3, 5

- [17] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-v: Exploring long-chain visual reasoning with multimodal large language models. arXiv preprint arXiv:2411.14432, 2024. 3, 18
- [18] Mikhail Evtikhiev, Egor Bogomolov, Yaroslav Sokolov, and Timofey Bryksin. Out of the bleu: how should we assess quality of the code generation models? Journal of Systems and Software, 203:111741, 2023. 3
- [19] Alex Fang, Gabriel Ilharco, Mitchell Wortsman, Yuhao Wan, Vaishaal Shankar, Achal Dave, and Ludwig Schmidt. Data determines distributional robustness in contrastive language image pre-training (clip). In International Conference on Machine Learning, pages 6216–6234. PMLR, 2022. 4, 5, 8
- [20] Daocheng Fu, Xin Li, Licheng Wen, Min Dou, Pinlong Cai, Botian Shi, and Yu Qiao. Drive like a human: Rethinking autonomous driving with large language models. In IEEE/CVF Winter Conference on Applications of Computer Vision, pages 910–919, 2024. 2, 3
- [21] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010,

2023. 7, 9

- [22] Ali Goli and Amandeep Singh. Frontiers: Can large language models capture human preferences? Marketing Science, 2024. 3
- [23] Xianda Guo, Ruijun Zhang, Yiqun Duan, Yuhang He, Chenming Zhang, Shuai Liu, and Long Chen. Drivemllm: A benchmark for spatial understanding with multimodal large language models in autonomous driving. arXiv preprint arXiv:2411.13112, 2024. 5
- [24] Xiaoshuai Hao, Mengchuan Wei, Yifan Yang, Haimei Zhao, Hui Zhang, Yi Zhou, Qiang Wang, Weiming Li, Lingdong Kong, and Jing Zhang. Is your hd map constructor reliable under sensor corruptions? In Advances in Neural Information Processing Systems, 2024. 5
- [25] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281–14290, 2024. 3
- [26] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17853–17862, 2023. 2, 7
- [27] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023. 6, 10
- [28] Bo Jiang, Shaoyu Chen, Bencheng Liao, Xingyu Zhang, Wei Yin, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Senna: Bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313, 2024. 2, 6
- [29] Jinkyu Kim, Anna Rohrbach, Trevor Darrell, John Canny, and Zeynep Akata. Textual explanations for self-driving vehicles. In European Conference on Computer Vision, pages 563–578, 2018. 2, 3, 5, 14, 17, 18
- [30] Lingdong Kong, Youquan Liu, Xin Li, Runnan Chen, Wenwei Zhang, Jiawei Ren, Liang Pan, Kai Chen, and Ziwei Liu. Robo3d: Towards robust and reliable 3d perception against corruptions. In IEEE/CVF International Conference on Computer Vision, pages 19994–20006, 2023. 3, 5, 16
- [31] Lingdong Kong, Shaoyuan Xie, Hanjiang Hu, Lai Xing Ng, Benoit Cottereau, and Wei Tsang Ooi. Robodepth: Robust out-of-distribution depth estimation under corruptions. Advances in Neural Information Processing Systems, 36, 2024. 3, 16
- [32] Lingdong Kong, Shaoyuan Xie, Hanjiang Hu, Yaru Niu, Wei Tsang Ooi, Benoit R. Cottereau, Lai Xing Ng, Yuexin Ma, Wenwei Zhang, Liang Pan, Kai Chen, Ziwei Liu, Weichao Qiu, Wei Zhang, Xu Cao, Hao Lu, Ying-Cong Chen, Caixin Kang, Xinning Zhou, Chengyang Ying, Wentao Shang, Xingxing Wei, Yinpeng Dong, Bo Yang, Shengyin Jiang, Zeliang Ma, Dengyi Ji, Haiwen Li, Xingliang Huang, Yu Tian, Genghua Kou, Fan Jia, Yingfei Liu, Tiancai Wang, Ying Li, Xiaoshuai Hao, Yifan Yang, Hui Zhang, Mengchuan Wei, Yi Zhou, Haimei Zhao, Jing Zhang, Jinke

- Li, Xiao He, Xiaoqiang Cheng, Bingyang Zhang, Lirong Zhao, Dianlei Ding, Fangsheng Liu, Yixiang Yan, Hongming Wang, Nanfei Ye, Lun Luo, Yubo Tian, Yiwei Zuo, Zhe Cao, Yi Ren, Yunfan Li, Wenjie Liu, Xun Wu, Yifan Mao, Ming Li, Jian Liu, Jiayang Liu, Zihan Qin, Cunxi Chu, Jialei Xu, Wenbo Zhao, Junjun Jiang, Xianming Liu, Ziyan Wang, Chiwei Li, Shilong Li, Chendong Yuan, Songyue Yang, Wentao Liu, Peng Chen, Bin Zhou, Yubo Wang, Chi Zhang, Jianhang Sun, Hai Chen, Xiao Yang, Lizhong Wang, Dongyi Fu, Yongchun Lin, Huitong Yang, Haoang Li, Yadan Luo, Xianjing Cheng, and Yong Xu. The robodrive challenge: Drive anytime anywhere in any condition. arXiv preprint arXiv:2405.08816, 2024. 5
- [33] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. 7
- [34] Edouard Leurent. An environment for autonomous driving decision-making. https://github.com/eleurent/ highway-env, 2018. 3
- [35] Chunyi Li, Jianbo Zhang, Zicheng Zhang, Haoning Wu, Yuan Tian, Wei Sun, Guo Lu, Xiaohong Liu, Xiongkuo Min, Weisi Lin, et al. R-bench: Are your large multimodal model robust to real-world corruptions? arXiv preprint arXiv:2410.05474, 2024. 4
- [36] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Conference on Empirical Methods in Natural Language Processing, pages 292–305,

2023. 6

- [37] Ye Li, Lingdong Kong, Hanjiang Hu, Xiaohao Xu, and Xiaonan Huang. Is your lidar placement optimized for 3d scene understanding? In Advances in Neural Information Processing Systems, 2024. 5
- [38] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, and Jifeng Dai. Bevformer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers. In European Conference on Computer Vision, pages 1–18, 2022. 7
- [39] Zhiqi Li, Zhiding Yu, Shiyi Lan, Jiahan Li, Jan Kautz, Tong Lu, and Jose M Alvarez. Is ego status all you need for openloop end-to-end autonomous driving? In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14864–14873, 2024. 2, 4, 17
- [40] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, 2004. 3, 7, 9, 10, 14, 17, 19
- [41] Benlin Liu, Yuhao Dong, Yiqin Wang, Yongming Rao, Yansong Tang, Wei-Chiu Ma, and Ranjay Krishna. Coarse correspondences elicit 3d spacetime understanding in multimodal language model. arXiv preprint arXiv:2408.00754, 2024. 3
- [42] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, pages 34892–34916, 2023. 2, 3
- [43] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning.

- In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 7, 9, 10, 18, 20
- [44] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 2, 3, 7, 9, 10, 13, 18
- [45] Yinhong Liu, Han Zhou, Zhijiang Guo, Ehsan Shareghi, Ivan Vuli´c, Anna Korhonen, and Nigel Collier. Aligning with human judgement: The role of pairwise preference in large language model evaluators. arXiv preprint arXiv:2403.16950,

2024. 3

- [46] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Oryx mllm: On-demand spatial-temporal understanding at arbitrary resolution. arXiv preprint arXiv:2409.12961, 2024. 7, 10
- [47] Zuyan Liu, Yuhao Dong, Yongming Rao, Jie Zhou, and Jiwen Lu. Chain-of-spot: Interactive reasoning improves large vision-language models. arXiv preprint arXiv:2403.12966,

2024. 3

- [48] Yingzi Ma, Yulong Cao, Jiachen Sun, Marco Pavone, and Chaowei Xiao. Dolphins: Multimodal language model for driving. arXiv preprint arXiv:2312.00438, 2023. 2, 3, 6, 7, 10, 12, 18
- [49] Srikanth Malla, Chiho Choi, Isht Dwivedi, Joon Hee Choi, and Jiachen Li. Drama: Joint risk localization and captioning in driving. In IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1043–1052, 2023. 5
- [50] Jiageng Mao, Yuxi Qian, Junjie Ye, Hang Zhao, and Yue Wang. Gpt-driver: Learning to drive with gpt. arXiv preprint arXiv:2310.01415, 2023. 2, 6
- [51] Thao Nguyen, Gabriel Ilharco, Mitchell Wortsman, Sewoong Oh, and Ludwig Schmidt. Quality not quantity: On the interaction between dataset design and robustness of clip. Advances in Neural Information Processing Systems, 35:21455–21469, 2022. 4, 5
- [52] OpenDriveLab. Foundation models for autonomous systems,

2024. Accessed: 2024-11-11. 4

- [53] Chenbin Pan, Burhaneddin Yaman, Tommaso Nesti, Abhirup Mallik, Alessandro G Allievi, Senem Velipasalar, and Liu Ren. Vlp: Vision language planning for autonomous driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14760–14769, 2024. 2
- [54] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Annual Meeting of the Association for Computational Linguistics, pages 311–318, 2002. 3, 7, 9, 14, 17
- [55] Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, and Yu-Gang Jiang. Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario. In AAAI Conference on Artificial Intelligence, pages 4542– 4550, 2024. 2, 3, 5, 12
- [56] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR,

2021. 3, 5

- [57] Enna Sachdeva, Nakul Agarwal, Suhas Chundi, Sean Roelofs, Jiachen Li, Mykel Kochenderfer, Chiho Choi, and Behzad Dariush. Rank2tell: A multimodal driving dataset for joint importance ranking and reasoning. In IEEE/CVF Winter Conference on Applications of Computer Vision, pages 7513–7522, 2024. 5
- [58] Hao Shao, Yuxuan Hu, Letian Wang, Guanglu Song, Steven L Waslander, Yu Liu, and Hongsheng Li. Lmdrive: Closed-loop end-to-end driving with large language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15120–15130, 2024. 2, 6
- [59] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. In European Conference on Computer Vision, pages 256–274, 2024. 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 17, 18, 19, 37
- [60] Austin Stone, Ted Xiao, Yao Lu, Keerthana Gopalakrishnan, Kuang-Huei Lee, Quan Vuong, Paul Wohlhart, Sean Kirmani, Brianna Zitkovich, Fei Xia, et al. Open-world object manipulation using pre-trained vision-language models. arXiv preprint arXiv:2303.00905, 2023. 3
- [61] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2446–2454, 2020. 2
- [62] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289, 2024. 2, 3, 5, 6, 12
- [63] Ngoc Tran, Hieu Tran, Son Nguyen, Hoan Nguyen, and Tien Nguyen. Does bleu score work for code migration? In IEEE/ACM International Conference on Program Comprehension, pages 165–176, 2019. 3
- [64] Haoqin Tu, Chenhang Cui, Zijun Wang, Yiyang Zhou, Bingchen Zhao, Junlin Han, Wangchunshu Zhou, Huaxiu Yao, and Cihang Xie. How many unicorns are in this image? a safety evaluation benchmark for vision llms. arXiv preprint arXiv:2311.16101, 2023. 6
- [65] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4566–4575, 2015. 3
- [66] Junyang Wang, Yiyang Zhou, Guohai Xu, Pengcheng Shi, Chenlin Zhao, Haiyang Xu, Qinghao Ye, Ming Yan, Ji Zhang, Jihua Zhu, et al. Evaluation and analysis of hallucination in large vision-language models. arXiv preprint arXiv:2308.15126, 2023. 6
- [67] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2, 3, 7, 9, 10, 15, 18, 20
- [68] Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez.

- Omnidrive: A holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning. arXiv preprint arXiv:2405.01533, 2024. 2, 6
- [69] Licheng Wen, Daocheng Fu, Xin Li, Xinyu Cai, Tao Ma, Pinlong Cai, Min Dou, Botian Shi, Liang He, and Yu Qiao. Dilu: A knowledge-driven approach to autonomous driving with large language models. In International Conference on Learning Representations, 2024. 2, 3
- [70] Dongming Wu, Wencheng Han, Tiancai Wang, Yingfei Liu, Xiangyu Zhang, and Jianbing Shen. Language prompt for autonomous driving. arXiv preprint arXiv:2309.04379,

2023. 3, 5

- [71] Shaoyuan Xie, Lingdong Kong, Wenwei Zhang, Jiawei Ren, Liang Pan, Kai Chen, and Ziwei Liu. Benchmarking and improving bird’s eye view perception robustness in autonomous driving. arXiv preprint arXiv:2405.17426, 2024. 3, 5, 10, 16
- [72] Yiran Xu, Xiaoyin Yang, Lihang Gong, Hsuan-Chu Lin, TzYing Wu, Yunsheng Li, and Nuno Vasconcelos. Explainable object-induced action decision for autonomous vehicles. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9523–9532, 2020. 2, 3, 5
- [73] Yi Xu, Yuxin Hu, Zaiwei Zhang, Gregory P Meyer, Siva Karthik Mustikovela, Siddhartha Srinivasa, Eric M Wolff, and Xin Huang. Vlm-ad: End-to-end autonomous driving through vision-language model supervision. arXiv preprint arXiv:2412.14446, 2024. 3
- [74] Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters, 9(10):8186–8193, 2024. 2, 3, 6
- [75] Jingkang Yang, Yuhao Dong, Shuai Liu, Bo Li, Ziyue Wang, Haoran Tan, Chencheng Jiang, Jiamu Kang, Yuanhan Zhang, Kaiyang Zhou, et al. Octopus: Embodied vision-language programmer from environmental feedback. In European Conference on Computer Vision, pages 20–38. Springer,

2025. 3

- [76] Zhenjie Yang, Xiaosong Jia, Hongyang Li, and Junchi Yan. A survey of large language models for autonomous driving. arXiv preprint arXiv:2311.01043, 2023. 2, 3, 4
- [77] Chenyu Yi, Siyuan Yang, Haoliang Li, Yap-peng Tan, and Alex Kot. Benchmarking the robustness of spatial-temporal models against corruptions. In Advances in Neural Information Processing Systems, 2021. 16
- [78] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. Bdd100k: A diverse driving dataset for heterogeneous multitask learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2636–2645, 2020. 2, 3, 12

