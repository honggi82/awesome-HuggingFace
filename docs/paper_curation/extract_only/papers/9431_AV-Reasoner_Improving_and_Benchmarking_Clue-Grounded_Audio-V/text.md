# arXiv:2506.05328v2[cs.CV]22Jul2025

## AV-Reasoner: Improving and Benchmarking Clue-Grounded Audio-Visual Counting for MLLMs

##### Lidong Lu*, Guo Chen*, Zhiqi Li, Yicheng Liu, Tong Lu

Nanjing University https://av-reasoner.github.io

#### Abstract

Despite progress in video understanding, current MLLMs struggle with counting tasks. Existing benchmarks are limited by short videos, close-set queries, lack of clue annotations, and weak multimodal coverage. In this paper, we introduce CG-AV-Counting, a manually-annotated clue-grounded counting benchmark with 1,027 multimodal questions and 5,845 annotated clues over 497 long videos. It supports both black-box and white-box evaluation, serving as a comprehensive testbed for both end-to-end and reasoning-based counting. To explore ways to improve model’s counting capability, we propose AV-Reasoner, a model trained with GRPO and curriculum learning to generalize counting ability from related tasks. AV-Reasoner achieves state-of-the-art results across multiple benchmarks, demonstrating the effectiveness of reinforcement learning. However, experiments show that on out-of-domain benchmarks, reasoning in the language space fails to bring performance gains.

#### 1 Introduction

Counting is a crucial indicator of fine-grained alignment and reasoning in Multimodal Large Language Models (MLLMs). Unlike coarse-grained tasks, counting requires precise temporal and spatial grounding. Models must detect, localize, and accumulate instances across frames or scenes, often under challenging visual conditions. This makes it a valuable proxy task for evaluating a model’s ability to align vision, audio and language at a detailed level. Moreover, counting has broad practical applications across a variety of real-world scenarios, including crowd analysis, object inventory, and repetitive action recognition.

Despite recent progress in tasks like video question answering, temporal grounding, etc. [1–14], MLLMs’ performance on counting task remains limited. Besides, existing counting benchmarks often suffer from the following shortcomings:

- • Short Video Duration: Most benchmarks [15–19] focus on videos shorter than one minute, which restricts the evaluation of counting and grounding performance over longer temporal contexts.
- • Closed-Set Queries: Most benchmarks have a close set of queries [17–19], limiting their generalization. Although some recent benchmarks support open-vocabulary queries [15], their use of short video clips makes it easy for models to exploit shortcuts.
- • Lack of Clue Annotations: Existing datasets typically lack annotations for counting clues [16–18], making it difficult to assess whether models perform genuine reasoning or rely on guesswork.
- • Single-Modality Evaluation: With the emergence of omni-modal MLLMs, most benchmarks still focus solely on visual inputs, failing to test how these models process and combine information from different modalities.

*Equal contribution. Corresponding author.

Preprint. Under review.

###### Clue-Grounded Annotation for Audio-Visual Counting

###### Evaluation Protocals

###### Event Counting:

How many times do the three men in black in the video yell as they make preparations before jumping into the water?

Acc OBOA MAE RMSE Acc OBOA MAE RMSE

End-to-End Counting (Black Box)

Long Acc

Full Video

Trimmed Video

Object Counting: How many chickens are kept in the man's yard?

Ref Acc

Reasoning Counting (White Box)

Temporal Segements

Event

Attribute Counting: How many different pot colors are there in the bedroom shown in the video?

Full Video

Bounding Box

Clue Acc

WCS

Annotated Frames

Clusters

Object / Attribute

###### AV-Reasoner Curriculum-based Reinforcement Finetuning with GRPO

###### Abilities

MCQ & Close-set QA

AVQA AVE AVVP ARIG AV-Counting

SRM

Completions

Temporal & Spatial Grounding

Policy Model

SRM

GRPO

KL

Training Datasets

Counting

Verifiable Rewards

Reference Model

Full Task

- Figure 1: Overview of CG-AV-Counting benchmark and AV-Reasoner baseline. Our benchmark covers three types of counting targets, providing end-to-end counting evaluation based on black-box testing and reasoing counting evaluation based on white-box testing, which can comprehensively evaluate the accuracy and interpretability of the model counts.AV-Reasoner uses GRPO algorithms based on course learning for intensive fine-tuning, introduces the stage review mechanism (SRM) and the Full Task RL to alleviate forgetting between tasks. AV-Reasoner has various capabilities such as AVQA, AVE, AVVP, ARIG, AV-Counting, etc.

To address these limitations, we propose CG-AV-Counting, a manually-annotated clue-grounded benchmark for evaluating MLLMs’ counting ability in long videos with multimodal queries. Based on 497 videos from CG-Bench [20] spanning over ten content categories, our dataset includes 1,027 annotated questions across five reference-query modalities: visual-only, audio-only, visual-reference audio-query, audio-reference visual-query, and joint audio-visual. Counting targets cover events, objects, and attributes (e.g., gender, color), each with annotated reference intervals and couting clues. Besides, we design both white-box and black-box evaluations to assess both end-to-end and reasoning-based counting.

We evaluate a range of closed-source and open-source MLLMs using this benchmark. The result shows that most models struggle with complex video counting tasks, especially under white-box evaluation, and open-source audio-visual models often perform worse than vision-only models. It indicates that current MLLMs still have significant room for improvement in this task.

In an effort to explore strategies for enhancing counting ability, and given the lack of countingspecific training data, we propose a capability transfer approach. Based on Ola-Omni [21], we develop AV-Reasoner, trained with GRPO [22] and curriculum learning [23]. Instead of relying solely on limited counting annotations, AV-Reasoner progressively learns tasks closely related to counting, such as audio-visual understanding, temporal grounding, and spatial grounding, enabling it to acquire counting ability through ability generalization. This approach achieves SOTA performance on general audio-visual understanding tasks such as Audio Visual Question Answering (AVQA), Audio Visual Temporal Grounding (AVTG), and Audio Refered Image Grounding (ARIG), and significantly improves counting performance compared to the base model.

#### 2 Related Work

Omni-modal Large Language Models. Omni-MLLMs are designed to process diverse inputs, such as text, images, audio, and video, within a unified framework. Models like AnyGPT [24] and

Unified-IO 2 [25] map all modalities to a shared feature space, while Mini-Omni2 [26], OMCAT [27], Baichuan-Omni [28], Ola-Omni [21], Crab [29], and VideoLLaMA 2 [6] use addtional encoders to inject new modalities. For audio-video alignment, Video-SALMONN [30] utilizes fine-grained synchronization, Meerkat [31] employs optimal transport for feature matching, and PAVE [32] integrates audio into visual features using a lightweight module. In this paper, we introduce AVReasoner, an enhanced version of Ola-Omni [21] that improves counting performance through GRPO training strategy. Beyond counting, AV-Reasoner also achieves new SOTA results on most audio-visual understanding tasks.

Video Counting Benchmarks. To the best of our knowledge, there is currently no comprehensive benchmark specifically designed to evaluate MLLMs’ video counting capabilities. DVDCounting [16] and VideoNIAH [33] use synthetic data for object counting. They have limited counting target variety and do not have long videos. Other benchmarks, such as MVBench [34] and WorldSense [35], include real-world videos, but counting is only a subtask of the overall evaluation, resulting in a smaller number of samples. Datasets for repetitive action counting [15, 17–19] feature short videos and simple queries, making them unsuitable for evaluating MLLMs. Besides, most benchmarks only have visual queries, which limits their ability to fully evaluate Omni-MLLMs.

To fill these gaps, as is shown in Tab. 1, we introduce CG-AV-Counting, a benchmark focused on counting tasks in videos longer than 10 minutes. Unlike previous datasets, it has both audio and visual modalities in the queries. The benchmark includes 1,027 samples with various counting targets like objects, events, and attributes. It also provide fine-grained clue annotations for more interpretable evaluation of MLLMs.

Large Language Model Reasoning. Recent advancements in LLM reasoning have focused on improving complex problem-solving abilities. Chain-of-Thought (CoT) [36] helps models break down complex tasks into logical steps, improving performance in arithmetic, commonsense, and symbolic reasoning. Unlike previous supervised methods, DeepSeek R1 [22] introduces a GRPObased reinforcement learning approach, guiding models to optimize reasoning strategies without relying on step-by-step annotations. Inspired by this, GRPO-based MLLMs, like Visual-RFT [37], Video-R1 [38], and Video-Chat-R1 [39], have achieved significant results in visual reasoning tasks.

#### 3 CG-AV-Counting

##### 3.1 Dataset Construction

- As shown in Fig. 2, the construction of CG-AV-Counting follows three stages:

In the question proposal stage, Gemini 2.0 Flash [40] is used to generate candidate questions and query intervals for each video. Annotators use these as references to improve efficiency.

During the answer annotation stage, human annotators first preview the whole video, select and refine questions that can be grounded globally, draft answers, and determine the interval to localize the query (reference interval). If needed, they can create new questions.

The clue annotation stage provides fine-grained annotations to verify the draft answers and evaluate model’s interpretable coutning ability. Human annotators first label counting clues from the video, which is categorized into three types according to the counting target:

- • For event counting, they annotate the start and end timestamps of each event.
- • For object counting, they annotate bounding boxes when objects appear simultaneously or annotate the first appearance of each object. This approach aligns with intuitive human counting practices and reduces annotation errors.
- • For attribute counting, they first perform object counting clue annotation and then group objects with the same query attribute.

Finally, the clues are compared with the answer drafted in stage 2. If they match, the corresponding sample is added to the candidate pool, and the final reference interval is determined by combining the draft interval with the clue intervals. Otherwise, arbitration is conducted to decide whether the sample should be retained.

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

- Figure 2: A Three-Stage Data Annotation Pipeline for CG-AV-Counting. It is designed to improve both annotation efficiency and accuracy.

V (637) A (13)

V2A (100)

A2V (134)

AV 143

Question (1027)

Object (768)

Event (245)

Attribute (14)

Question (1027)

0

20

40

60

80

100

120

140

160

180

200

1 3 5 7 9 11 13 15 17 19 21 23 25

NumberofQuestions

Count Number

≥25

0

50

100

150

200

250

5 10 15 20 25 30 35 40 45 50 55 60

NumberofQuestions

Reference Interval (seconds)

(a) Distribution of Query Modality (b) Distribution of Counting Target (c) Distribution of Reference Interval Length (5s per bin) (d) Distribution of Count Number

- Figure 3: Statistics of Our Proposed Benchmark. Our benchmark features diverse query modalities and a wide range of counting targets. The count distribution follows a long-tail pattern, providing a challenging testbed for evaluating models’ generalization and compositional reasoning abilities.

##### 3.2 Dataset Statistics

- As shown in Fig. 3, CG-AV-Counting is based on a subset of 497 videos from CG-Bench [20]. The benchmark includes 1,027 multimodal-query questions and 5,845 fine-grained manually-annotated clue annotations. Nearly 40% of the samples require the model to use both audio and visual modalities for counting, while others only require the visual modality. This design ensures that the benchmark is applicable to both visual models and audio-visual models. The benchmark includes object, event, and attribute counting target. Among them, attribute counting is more challenging because it requires grouping objects with the same attribute based on the query.

This benchmark spans a numerical range from 1 to 76, with a long-tail distribution, where most counts fall between 1 and 20. Video content includes over 10 categories, such as sports, life record, humor, tutorials, etc., offering greater domain diversity than existing benchmarks. All videos in the benchmark exceed 10 minutes, and reference intervals range from seconds to minutes, covering both short-term and long-range dependencies.

In summary, as shown in Tab. 1, unlike previous counting benchmarks, our benchmark incorporates both audio and visual modalities, features more complex queries, and provides fine-grained counting clues to jointly evaluate models’ abilities in both end-to-end and reasoning-based counting.

##### 3.3 Evaluation Metrics

We follow CG-Bench’s settings and use a dual evaluation protocol. It includes black-box and white-box evaluations to comprehensively assess MLLMs’ counting ability. Black-box Evaluation assesses model’s end-to-end counting ability under two settings:

- • Long Acc: The model is provided with the entire video and must perform both temporal localization and counting, reflecting its holistic reasoning ability.
- • Ref Acc: The model is given a trimmed segment based on the reference interval, isolating its counting ability from temporal localization.

For both settings, we use four complementary metrics to measure counting ability:

- • Accuracy (Acc): It evaluates the model’s ability to predict the exact number of instances, reflecting strict counting precision.
- • Off-By-One Accuracy (OBOA): It allows a small margin of error in counting. A prediction is considered correct if it is off by at most one.
- • Mean Absolute Error (MAE): It quantifies the average size of counting errors, providing a direct measure of overall counting consistency.

- • Root Mean Square Error (RMSE): It penalizes larger deviations more heavily, highlighting the model’s stability and reliability in avoiding severe counting mistakes.

White-box Evaluation assesses the model’s ability to localize counting targets with explicit evidence: for event counting, the model predicts temporal segments for each event. We use temporal IoU (tIoU) to measure alignment with ground-truth intervals. For object counting, the model is given specific video frames and must locate the first appearance of each object using bounding boxes. Spatial alignment is measured by Intersection over Union (IoU). For attribute counting, predicted bounding boxes are clustered according to the query, and each predicted cluster is evaluated against the corresponding ground-truth cluster using IoU.

To account for both localization and count accuracy, we add a counting accuracy penalty for deviations from the ground-truth. So the White-box Counting Score (WCS) is defined as:

K

1 K

WCS =

LAk × CAPk × 100%

k=1

|GTk|

1 |GTk|

LAk =

IoU(Predk,GTk)

j=1

||Predk| − |GTk|| |GTk|

CAPk = max 0,1 −

(1)

where K is the number of instance clusters: one for event and object counting, and the number of attribute types for attribute counting. IoU refers to tIoU or IoU depending on the task.

LAk (Localization Accuracy) measures the average overlap between predicted and ground-truth instances within the k-th cluster, reflecting how accurately the model localizes instances. To fairly evaluate this metric, a greedy matching algorithm is used to establish an optimal one-to-one correspondence between predicted and ground-truth instances, maximizing the total IoU even when their numbers or order differ.

CAPk (Counting Accuracy Penalty) penalizes the discrepancy between the number of predicted instances and ground-truth instances in the k-th cluster. It scales from 0 to 1, with a value of 1 indicating perfect count matching and values closer to 0 indicating larger count deviations.

The score increases when predictions are both well-localized and correctly counted, and drops as the count error or localization error grows. It is bounded between 0 and 100. A score of 100 indicates perfect performance, where all predicted instances are accurately localized and the predicted count exactly matches the ground-truth. In contrast, a score of 0 reflects either a severe count mismatch, where the absolute error exceeds the ground-truth count, or a failure to follow the required output format. In practice, when the counting accuracy is low, the penalty tends to dominate the overall score, resulting in a sharp drop regardless of localization quality. Conversely, when the predicted count is close to the ground-truth, the localization accuracy becomes the main factor influencing the final score.

In addition, we report Instruction-Following Accuracy (IFA) to measure the proportion of samples for which the model’s output matches the required format. This metric complements WCS by explicitly evaluating the model’s ability to follow task-specific output instructions, which is essential for reliable and interpretable predictions.

##### 3.4 Comparison with Existing Video Counting Benchmarks

As summarized in Table 1, existing video counting benchmarks typically suffer from limited modality, content diversity, and reasoning complexity.

Most prior datasets, such as DVD-Counting [16], VideoNIAH [33], and MVBench [34], only contain visual-only samples. In contrast, CG-AV-Counting introduces a richer query structure with audiovisual interactions, supporting audio-referenced visual queries, visual-referenced audio queries, and joint audio-visual counting—enabling evaluation of complex multimodal reasoning scenarios.

- Table 1: Comparison with existing video counting benchmarks. "✓/✗" indicates partial satisfaction. "#Test Samples" indicates the number of samples related to the counting task within each benchmark.

Benchmark #Test Samples Modaility Real scene Long Video Complex Query Clue Event Object Attribute DVD-Counting [16] 200 V ✗ ✗ ✓ ✗ ✗ ✓ ✗

VideoNIAH [33] 450 V ✗ ✗ ✓ ✗ ✗ ✓ ✗ MVBench [34] 400 V ✓/✗ ✓/✗ ✓ ✗ ✓ ✗ ✗ RepCount [19] 152 V ✓ ✓ ✗ ✓ ✓ ✗ ✗

CountixAV [18] 563 A+V ✓ ✗ ✗ ✗ ✓ ✗ ✗

OVR [15] 12452 V ✓ ✗ ✓ ✗ ✓ ✗ ✗ WorldSense [35] 476 A+V ✓ ✓ ✓ ✗ ✓ ✓ ✗

CG-AV-Conuting (Ours) 1027 A+V ✓ ✓ ✓ ✓ ✓ ✓ ✓

Second, regarding video length, most existing benchmarks rely on short clips (typically under 1 minute) [15, 16, 18, 33, 34]. CG-AV-Counting leverages long-form videos (all exceeding 10 minutes), requiring sustained temporal reasoning and long-range clue localization.

Third, in terms of counting targets, previous datasets mostly focus on object or event counting [15, 16, 18, 19, 33, 34]. In contrast, CG-AV-Counting comprehensively covers object, event, and attribute counting, enabling a more fine-grained and versatile evaluation.

Fourth, while prior datasets [16, 18, 33–35] typically only provide final count labels, CG-AV-Counting includes manually annotated fine-grained counting clues that clearly indicate where and how evidence for the count appears across modalities and time. These annotations not only improve dataset transparency, but also support interpretable and diagnostic evaluation of model behavior.

Finally, beyond standard black-box evaluation of end-to-end counting accuracy, CG-AV-Counting introduces white-box evaluation protocols to assess models’ intermediate reasoning steps. This dual evaluation protocols allow for a more comprehensive and explainable assessment of multimodal counting capabilities.

Overall, CG-AV-Counting significantly broadens the scope and realism of video counting evaluation, establishing a more challenging and representative bnchmark for future multimodal reasoning research.

##### 3.5 Evaluation Results

As shown in Tab. 2, all models struggle with the counting task, significantly underperforming compared to human performance. Close-source MLLMs generally outperform open-source ones across most metrics. However, even the best-performing close-source models fall far short of humanlevel accuracy, indicating that accurate counting remains a major challenge for current MLLMs.

Notably, Gemini 2.5 Pro [40] and Gemini 2.5 Flash [40], which utilize both audio and visual modalities, achieve the strongest results, which reflects their relative robustness across different evaluation settings. Among open-source models, Eagle2.5-8B [3] is a rare outlier, outperforming several close-source models.

Nevertheless, despite the inclusion of audio-visual samples in the benchmark, the improvement from A+V models is inconsistent. For example, open-source audio-visual models such as UnifiedIO-2 XXL [25] and VideoLLaMA2.1-7B-AV [6] perform significantly worse than vision-only opensource models, suggesting that the mere inclusion of audio is not sufficient. The limited gains from multimodal inputs can likely be attributed to two key factors:

- • Imperfect Audio-Visual Alignment: Many models may not be well-trained on temporally synchronized audio and video inputs. Misaligned representations hinder the ability to leverage audio cues for grounding or disambiguation, especially in dynamic counting scenarios.
- • Insufficient Training on Grounding Tasks: The counting task inherently requires fine-grained spatio-temporal reasoning and object tracking, which are not emphasized in the pretraining of most MLLMs. The absence of dedicated supervision on such tasks limits their generalization to grounding-sensitive benchmarks.

In particular, the consistently low WCS scores across all models highlight a fundamental challenge: the task requires not only accurate grounding and precise counting, but also the ability to produce

- Table 2: Performance of various MLLMs on CG-AV-Counting. We use different colors to indicate the model categories: Vision Language Models and Audio-Visual Language Models . Bold indicates the best scores, while underline denotes the second-best scores.

Black-box Evaluation (Long Acc) Black-box Evaluation (Ref Acc) White-box Evaluation

Model Modality

Acc↑ OBOA↑ MAE↓ RMSE↓ Acc↑ OBOA↑ MAE↓ RMSE↓ WCS↑ IFA↑

Random - 1.56 4.97 30.35 36.66 - - - - 0.25 100.00 Human (full-video) A+V 85.00 96.49 0.65 0.95 91.53 98.05 0.23 0.45 71.93 100.00 GPT-4.1 (text) [41] - 6.04 13.15 8.90 17.60 - - - - 0.00 76.24

###### Close-Source MLLMs

GPT-4.1 [41] V 27.17 49.95 2.68 4.78 37.39 64.85 1.77 3.53 2.78 98.73 GPT-4o [42] V 22.30 45.57 3.25 5.82 32.91 58.62 2.06 4.37 2.59 98.73 SEED-1.5 VL [43] V 27.85 52.19 2.93 6.39 36.12 57.64 2.35 4.45 2.46 99.03 Gemini 2.5 Flash [40] A+V 36.90 61.05 2.71 6.32 41.48 65.53 1.86 4.08 4.20 95.03 Gemini 2.5 Pro [40] A+V 40.80 65.82 2.33 7.20 47.42 72.25 1.47 3.53 6.71 95.03

###### Open-Source MLLMs

VideoLLaMA3-7B [4] V 12.46 30.57 4.52 12.60 18.50 41.48 3.23 5.68 1.03 91.72 Eagle2-9B [44] V 12.46 34.08 3.84 6.33 21.91 45.37 3.24 5.77 0.57 76.14 Qwen2.5-VL-7B [1] V 20.84 48.00 4.08 7.91 27.45 52.68 2.76 6.17 0.87 85.78 MiniCPM-V 2.6 [45] V 13.83 33.20 4.06 6.58 18.99 39.44 3.62 6.74 0.57 67.67 InternVL3-8B [2] V 17.92 43.43 3.21 5.55 30.09 56.67 2.57 5.76 0.71 97.57 InternVideo2.5-8B [46] V 22.20 48.30 3.17 5.92 28.33 56.47 2.75 5.70 - VideoChat-Flash-7B [47] V 19.86 43.91 3.49 6.17 25.41 46.54 3.53 6.48 - Eagle2.5-8B [3] V 28.04 51.80 2.76 5.23 34.47 63.97 2.07 4.39 2.59 83.35 UnifiedIO-2 XXL [25] A+V 10.61 30.48 3.99 6.30 15.29 37.49 3.38 5.79 0.00 2.24 VideoLLaMA2.1-7B-AV [6] A+V 5.06 13.73 5.11 7.34 8.57 18.40 4.93 7.24 0.11 13.83 Qwen2.5-Omni-7B [48] A+V 22.30 48.69 3.92 8.49 33.05 60.20 3.05 5.79 1.17 95.52 Ola-7B [21] A+V 17.92 38.85 4.57 10.52 25.33 46.53 3.30 5.98 0.84 75.66

- Figure 4: Overview of Our Training Strategy. Our strategy consists of three stages. During the curriculum-based RL stage, we introduce a Stage Review Mechanism, which mitigates forgetting of previously trained tasks by mixing in a proportion of earlier samples.

answers in a predefined format. This jointly stresses the models’ spatial understanding, numerical reasoning, and controllable output generation.

In summary, the counting task remains a difficult benchmark due to its dependence on precise grounding and multimodal reasoning. These results underscore the need for more targeted training strategies and benchmarks to foster genuine multimodal understanding rather than surface-level fusion.

#### 4 A Strong Baseline: AV-Reasoner

##### 4.1 Overview

Due to the lack of enough data for counting tasks, directly improving counting performance with existing data is suboptimal. To explore a way to improve models’ counting performance without relying on large-scale annotated counting data, we introduce AV-Reasoner, built upon the Ola-Omni7B[21] base model, which improves counting performance by training on tasks strongly related to counting, such as temporal and spatial grounding. This enables AV-Reasoner to enhance its counting ability through task-related exploration.

##### 4.2 Training Dataset

To tackle audio-visual counting tasks, the model needs core capabilities in audio-visual understanding, temporal grounding, and spatial grounding. As shown in Tab. 3, we use original annotations from datasets like AVQA [49], MUSIC-AVQA [50], AVE [51], UnAV [52], DVD-Counting [16], and RepCount [19]. For the LLP [53] dataset, we use pseudo-labels from AVUIE [29]. For ARIG, we convert segmentation masks from AVSS [54] into bounding boxes. Unlike AVUIE [29], which normalizes coordinates, our method retains the original image resolution.

##### 4.3 Verifiable Rewards Design

We explore GRPO-based training for three major audio-visual tasks: QA, Grounding, and Counting. For each, we design a verifiable reward to guide learning.

Table 3: The datasets used during training.

Tasks Datasets AVQA AVQA [49], Music AVQA [50] AVTG AVE [51], UnAV [52], LLP [53] ARIG AVSS-ARIG [54] Counting DVD-Counting [16], RepCount [19]

General Format Reward RGFormat. For QA and Counting tasks, we use the format reward function from DeepSeek-R1 [22] to ensure the model’s output follows the expected format. The model must output thinking process within a <think>...</think> block and the final answer in a <answer>...</answer> block. We verify the format using regular expression matching. If correct, a reward of 1 is given; otherwise, 0.

Json Format Reward RJFormat. For grounding tasks, we require the model to output in JSON format within the <answer>...</answer> block. Based on the general format reward, we introduce a JSON validity check. First, we attempt to parse the content between <answer> and </answer> as a standard JSON object. If parsing succeeds, the multiplier m = 1.0. If it fails, we use a bracketmatching algorithm to extract a potential JSON structure; if successful, m = 0.5. If both approaches fail, m = 0. Next, we define the key completeness score s = Ncomplete/Ntotal, where Ncomplete is the number of JSON items with all required keys, and Ntotal is the total number of items in the parsed list. The final JSON Format Reward RJFormat = m × s.

Accuracy Reward Racc. For multiple-choice and closed-set question answering tasks, the reward is 1.0 for a correct prediction and 0.0 for an incorrect prediction.

IoU Reward RIoU. For object and event grounding tasks, we calculate the tIoU or cIoU [55] between predicted and ground truth sets of temporal segments or spatial bounding boxes. The RIoU is the average of the IoU scores across all groups, where the best matches are found using a greedy algorithm for optimal alignment. If both the prediction and reference sets are empty, the reward is set to 1.0.

rMAE Reward RrMAE. For counting tasks, we use relative MAE (Mean Absolute Error) to compute the reward. The calculation method is as follows:

RrMAE = 1 − min(1.0,|Pred − GT|/GT) (2) Specially, if the ground truth is 0, we use accuracy-based Racc as the RrMAE.

##### 4.4 Training Strategy

As shown in Fig. 4, our training pipeline follows a three-stage process: cold-start SFT, curriculumbased RL with stage review mechanism, and full-task RL.

Cold-start SFT. To improve Ola-Omni’s [21] performance on counting and grounding tasks, we conduct a cold-start SFT, training it on AVTG, ARIG, and counting tasks. This phase also enhances its ability to generate structured JSON outputs for rule-based rewards in these tasks.

Curriculum-based RL (CB-RL) with Stage Review Mechanism. Training all tasks simultaneously from scratch can negatively impact performance on complex tasks like counting, which suffers from limited data availability. To mitigate this, we use a curriculum learning strategy, organizing tasks into three levels of increasing difficulty: (1) question answering, (2) temporal and spatial grounding, and (3) counting. The model is trained on each level for two epochs in sequence using GRPO, allowing it to progressively acquire necessary skills. To improve training efficiency, we introduce an offline

- Table 4: Comparison with SoTA MLLMs on Various Audio-Visual Benchmarks. We use different colors to indicate the model categories: Vision Language Models ,

existing Audio-Visual Language Models and our proposed model .

MusicAVQA LLP UnAV ARIG DVD-Counting CG-AV-Counting AVHBench AV-Odyssey OmniBench WorldSense acc Segment Level Event Level mAP cIoU AUC acc long ref WCS A2V V2A acc acc acc Close-Source MLLMs

Model

GPT-4.1 [41] - - - - - - - 27.17 37.39 2.78 - - - - GPT-4o [42] - - - - - - - 22.30 32.91 2.59 - - 34.50 51.14 42.60 Claude 3.5 Sonnet [56] - - - - - - - - - - - - - 59.37 34.8 Gemini 2.5 Pro [40] - - - - - - - 40.80 47.42 6.71 - - - - Gemini 2.5 Flash [40] - - - - - - - 36.90 41.48 4.20 - - - - Gemini 1.5 Pro [40] - - - - - - - - - - - - 30.80 42.91 48.00 Gemini 1.5 Flash [40] - - - - - - - - - - 83.30 63.00 27.80 - -

###### Open-Source MLLMs

Video-R1 [38] - - - - - - 34.50 - - - - - - - Qwen2.5-Omni [48] - - - - - - - 22.30 33.05 1.17 - - - 56.13 45.40 UnifiedIO2-XXL [25] - - - - - - - 10.61 15.29 0.00 - - 27.20 33.98 25.90 VideoLLaMA 2 [6] - - - - - - - 5.06 8.57 0.11 75.20 74.20 26.80 - 25.40 Video-SALMONN [30] 47.60 - - - - - - - - - 78.10 65.20 - 35.64 OneLLM [57] 47.60 - - - - - - - - - 53.70 44.30 27.40 - 22.80 GroundingGPT [58] - - - - 44.02 0.45 - - - - - - - - VALOR [59] 78.90 - - - - - - - - - - - - - X-InstructBLIP [60] 44.50 - - - - - - - - - 18.10 16.30 - - MEERKAT [31] 79.15 54.96 - - - - - - - - - - - - AVicuna [61] 49.60 - - 60.30 - - - - - - - - - - PAVE [32] 82.30 - - - - - - - - - - - - - Crab [29] 78.94 59.00 54.44 - 41.78 0.42 - - - - - - - - AV-Reasoner (Ours) 83.39 62.40 59.40 64.69 45.06 0.47 43.50 22.30 35.83 1.11 84.45 80.63 36.38 48.95 45.84 AV-Reasoner-Thinking (Ours) 85.01 64.70 62.00 65.18 46.73 0.49 44.00 21.03 34.08 1.68 82.45 80.28 34.29 48.34 44.36

data filtering pipeline: before each epoch, we perform five rollouts per sample using the reference model. For QA and counting tasks, we discard samples where all rollouts yield correct answers; for grounding tasks, we discard those with the average IoU greater than 0.9.

Additionally, to address performance degradation caused by forgetting earlier tasks during later training stages, we introduce a Stage Review Mechanism (SRM) that mixes in 20% of previously seen samples to maintain stability across tasks.

Full-task RL (FT-RL). Despite using the SRM, some performance degradation across tasks remains. To address this, we conduct a full-task training stage after curriculum learning. We sample data from each dataset, ensuring balanced distribution across tasks and difficulty levels, with difficulty defined by the pass rate over five rollouts. Using these samples, we perform another round of reinforcement learning to enhance the model’s overall performance.

#### 5 Expirements

##### 5.1 Comparison with State-of-the-Art MLLMs

As shown in Tab. 4, our model, AV-Reasoner, demonstrates strong performance across various audiovisual tasks. AV-Reasoner-Thinking achieves an accuracy of 85.01 on MusicAVQA [50], surpassing PAVE’s 82.30. For event localization, it achieves 64.70 (segment-level) and 62.00 (event-level) on LLP [53], and 65.18 mAP on UnAV [52], all significantly surpassing the current SOTA results. In spatial grounding, AV-Reasoner-Thinking outperforms GroundingGPT [58] and Crab [29], with 46.73 cIoU and 0.49 AUC on ARIG [54]. On DVD-Counting [16], AV-Reasoner achieves 44.00 accuracy, surpassing Video-R1 [38] by 9.50 points.

Beyond task-specific benchmarks, AV-Reasoner-Thinking also excels in comprehensive benchmarks. In AV-Odyssey [62], it reaches 34.29 accuracy, ranking just below GPT-4o [42]. When not forced to output thinking, AV-Reasoner surpasses GPT-4o by 1.88 points. On WorldSense [35], AV-Reasoner scores 45.84 without thinking output, second only to Gemini 1.5 Pro [40], while dropping to 44.36 with thinking output. In OmniBench [63], AV-Reasoner scores 48.95, though it still lags behind Qwen2.5-Omni [48], it outperforms Gemini 1.5 Pro [40]. The performance difference between explicitly outputting the thinking process and not will be discussed in the ablation study section.

- Table 5: Comparison with Base Model on Counting Tasks. Models trained with our proposed strategy consistently achieve higher accuracy across all counting benchmarks. The setting without explicit reasoning outputs yields even better performance.

|Model|DVD-Counting<br><br>Acc OBOA<br><br>|WorldSense Object Action Audio<br><br>|CG-AV-Counting (Long Acc) Acc OBOA MAE RMSE|CG-AV-Counting (Ref Acc)<br><br>Acc OBOA MAE RMSE|CG-AV-Counting (White Box)<br><br>WCS IFA|
|---|---|---|---|---|---|
|Ola-Omni (Base Model)<br><br>|16.50 51.50<br><br>|36.10 30.91 44.44|17.92 38.85 4.57 10.52<br><br>|25.33 46.53 3.30 5.98<br><br>|0.84 75.66|
|AV-Reasoner-Thinking (Ours)<br><br>|44.00 81.00 (+27.50) (+29.50)|36.59 34.55 46.67 (+0.49) (+3.64) (+2.23)<br><br>|21.03 48.78 3.26 8.20 (+3.11) (+9.93) (-1.31) (-2.32)<br><br>|34.08 60.95 2.40 4.66 (+8.75) (+14.42) (-0.90) (-1.32)|1.68 79.65 (+0.84) (+3.99)<br><br>|
|AV-Reasoner (Ours)<br><br>|43.50 80.50 (+27.00) (+29.00)<br><br>|37.07 32.12 47.78 (+0.97) (+1.21) (+3.34)|22.30 48.30 3.15 5.89 (+4.38) (+9.45) (-1.42) (-4.63)<br><br>|35.83 61.44 2.38 4.44 (+10.50) (+14.91) (-0.92) (-1.54)<br><br>|1.11 80.82 (+0.27) (+5.16)|

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

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

- Figure 5: Example Ouputs on Various Tasks. AV-Reasoner not noly outputs the correct answer, but also give the thinking process.

- 5.2 Comparison with Base Model on Counting Tasks

Tab. 5 highlights the effectiveness of GRPO in improving counting capabilities. Compared to the base model Ola-Omni [21], both AV-Reasoner variants achieve consistent improvements across all counting benchmarks. This demonstrates that our GRPO-based training strategy not only leverages existing grounding and QA abilities, but also enables the model to generalize them to counting.

- 5.3 Ablation Studies In this section, we aim to address the following key questions through a series of experiments:

- Q1: How does each stage of our training strategy enhance the model’s counting capability?

- As shown in Tab. 6, SFT improves in-domain DVD-Counting [16] accuracy but suffers from overfitting, leading to degraded out-of-domain performance. QA-based RFT mitigates overfitting and improves generalization, but its gains on counting are limited. In contrast, grounding-based training greatly improves counting by focusing on the skills needed for it. And counting-specific and full-task RL further improve performance by reinforcing task-relevant abilities.

Q2: What is the impact of curriculum-based RL and the review mechanism?

- As shown in Tab. 7, the base model performs poorly on grounding tasks due to its weak ability to follow JSON-structured output formats. SFT mitigates this issue by explicitly guiding the model to

###### Table 6: Impact of Each Training Stage on Counting Performance. SRM is applied to CB-RL stages.

DVD-Counting WorldSense CG-AV-Counting(Long Acc) CG-AV-Counting(Ref Acc)

Stage

Acc OBOA MAE RMSE Object Action Audio Acc OBOA MAE RMSE Acc OBOA MAE RMSE

Base Model 16.50 51.50 1.86 2.45 36.10 30.91 44.44 17.92 38.85 4.57 10.52 25.33 46.53 3.30 5.98 SFT 41.50 75.00 0.99 1.51 27.80 30.91 35.52 15.00 35.25 3.79 8.24 23.89 44.34 2.91 4.68 SFT + CB-RLQA 23.00 54.00 1.77 2.44 33.66 31.52 41.46 16.55 37.39 4.75 8.27 24.76 45.68 2.73 4.68 SFT + CB-RLQA + CB-RLGrounding 34.50 70.00 1.32 1.96 34.63 33.17 45.37 18.21 45.37 4.07 8.26 28.40 51.04 2.54 4.67 SFT + CB-RLQA + CB-RLGrounding + CB-RLCounting 43.00 80.00 0.85 1.28 35.12 33.66 45.85 20.84 46.54 3.31 8.22 34.00 59.60 2.50 4.68 SFT + CB-RLQA + CB-RLGrounding + CB-RLCounting + FT-RL 44.00 81.00 0.84 1.29 36.59 34.55 46.67 21.03 48.78 3.26 8.20 34.08 60.95 2.40 4.66

###### Table 7: Effect of Curriculum Learning and Review Mechanism on Model Performance.

QA Grounding Counting

Method

AVQA MUSIC-AVQA AVE LLPsegment LLPevent UnAV ARIG DVD CG-AVlong CG-AVref CG-AVWCS WorldSenseobj WorldSenseact WorldSenseaud

Base model 82.61 79.47 19.43 14.5 14.5 0.28 0.19 16.50 17.92 25.33 0.84 36.10 30.91 44.44 CB-RL(w/ SRM) + FT-RL 92.97 84.59 78.56 51.90 49.80 55.45 45.55 29.50 18.31 28.41 1.15 36.59 31.71 45.37 SFT 78.73 78.10 80.02 48.70 47.10 52.67 45.48 41.50 15.00 23.89 0.83 27.80 30.91 35.52 SFT + GRPO with all data 93.25 85.68 79.82 58.10 55.80 60.09 48.43 31.50 10.42 17.16 0.77 24.39 22.42 32.20 SFT + CB-RL(w/o SRM) 89.24 78.93 80.03 60.30 58.30 63.91 45.97 42.50 20.84 33.07 1.19 34.63 30.73 42.44 SFT + CB-RL(w/o SRM) + FT-RL 93.13 83.44 80.10 61.30 60.90 64.28 46.72 43.50 20.74 33.68 1.61 35.61 33.17 46.34 SFT + CB-RL(w/ SRM) 93.14 84.79 80.15 64.50 61.60 65.07 46.62 43.00 20.84 34.00 1.66 35.12 33.66 45.85 SFT + CB-RL(w/ SRM) + FT-RL 93.17 85.01 81.26 64.70 62.00 65.18 46.73 44.00 21.03 34.08 1.68 36.59 34.55 46.67

produce well-formed outputs, which is critical for grounding evaluation. While directly applying RL without SFT can improve performance on simpler tasks like QA, it brings limited benefits on more complex tasks that demand reasoning and precise grounding.

Training with GRPO on all tasks at once highlights the model’s poor generalization under limited data, especially for counting tasks. CB-RL helps by learning tasks of varying difficulty progressively, but degrades performance on previously learned tasks. Our SRM addresses this by periodically revisiting earlier data. When combined with FT-RL, it achieves the best trade-off between learning new capabilities and preserving prior knowledge.

##### Q3: Is explicit thinking output truly beneficial for improving model performance?

As shown in Tab. 4, GRPO enhances reasoning ability, but explicitly output thinking at inference does not always improve performance, which may introduce errors and increase hallucinations, particularly in out-of-domain settings. This is evident in benchmarks like WorldSense and AVOdyssey hallucination subsets, and AVHBench, where explicit thinking lowers accuracy. Specifically, as shown in Tab. 8, AVHBench audio-driven video hallucination accuracy drops from 84.45 to 82.45, and WorldSense Hallucination subset drops from 45.56 to 35.56. These findings emphasize the importance of balancing reasoning transparency with robustness to reduce hallucinations.

Table 8: Comparison of Hallucination on Benchmark Subset with and without Explicit Thinking.

Benchmark Thinking No Thinking

AVHBench A2V 82.45 84.45 AVHBench V2A 80.28 80.63 WorldSense Hall. 35.56 45.56 AVOdyssey Hall. 31.00 32.50

##### 5.4 Qualitative Results

Fig. 5 visualizes AV-Reasoner’s outputs on tasks including MCQ, QA, AVTG, ARIG, and Counting. The model not only provides correct answers but also generates coherent reasoning. For example, in AVTG, it identifies the sounds and determines the primary one, and in ARIG, it explains how the sounding object is located based on shape and auditory cues.

#### 6 Conclusion

In this paper, we introduce CG-AV-Counting, a manually-annotated multimodal benchmark for evaluating counting in long videos with clue annotations. It reveals that current MLLMs struggle in both black-box and white-box settings, particularly in reasoning counting. In order to explore ways to improve model’s counting performance, we propose a GRPO-based training strategy. Our model AV-Reasoner achieves strong performance in counting and other audio-visual understanding tasks.

Limitation. Firstly, white-box evaluation of object and attribute counting typically depends on predefined clue frames due to sparse clue annotations. Moreover, inconsistencies between the model’s reasoning process and its final answers often lead to performance drops when the model is required

to explicitly output its reasoning at the interface, highlighting the need to enforce reasoning-answer consistency during training.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [2] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [3] Guo Chen, Zhiqi Li, Shihao Wang, Jindong Jiang, Yicheng Liu, Lidong Lu, De-An Huang, Wonmin Byeon, Matthieu Le, Tuomas Rintamaki, et al. Eagle 2.5: Boosting long-context post-training for frontier vision-language models. arXiv preprint arXiv:2504.15271, 2025.
- [4] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025.
- [5] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023 - System Demonstrations, Singapore, December 6-10, 2023, pages 543–553. Association for Computational Linguistics, 2023.
- [6] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024.
- [7] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.
- [8] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023.
- [9] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024.
- [10] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, and Yu Qiao. Internvideo: General video foundation models via generative and discriminative learning. CoRR, abs/2212.03191, 2022.
- [11] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. arXiv preprint arXiv:2403.15377, 2024.
- [12] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Song XiXuan, et al. Cogvlm: Visual expert for pretrained language models. Advances in Neural Information Processing Systems, 37:121475–121499, 2024.
- [13] Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500, 2024.
- [14] Guo Chen, Yin-Dong Zheng, Jiahao Wang, Jilan Xu, Yifei Huang, Junting Pan, Yi Wang, Yali Wang, Yu Qiao, Tong Lu, et al. Videollm: Modeling video sequence with large language models. arXiv preprint arXiv:2305.13292, 2023.
- [15] Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, and Andrew Zisserman. Ovr: A dataset for open vocabulary temporal repetition counting in videos. arXiv preprint arXiv:2407.17085, 2024.
- [16] Video-R1. Dvd-counting, 2025. URL https://huggingface.co/datasets/Video-R1/ DVD-counting.

- [17] Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. Counting out time: Class agnostic video repetition counting in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10387–10396, 2020.
- [18] Yunhua Zhang, Ling Shao, and Cees GM Snoek. Repetitive activity counting by sight and sound. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14070–14079, 2021.
- [19] Huazhang Hu, Sixun Dong, Yiqun Zhao, Dongze Lian, Zhengxin Li, and Shenghua Gao. Transrac: Encoding multi-scale temporal correlation with transformers for repetitive action counting. arXiv preprint arXiv:2204.01018, 2022.
- [20] Guo Chen, Yicheng Liu, Yifei Huang, Yuping He, Baoqi Pei, Jilan Xu, Yali Wang, Tong Lu, and Limin Wang. Cg-bench: Clue-grounded question answering benchmark for long video understanding. arXiv preprint arXiv:2412.12075, 2024.
- [21] Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Ola: Pushing the frontiers of omni-modal language model with progressive modality alignment. arXiv preprint arXiv:2502.04328, 2025.
- [22] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [23] Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko Sinapov, Matthew E Taylor, and Peter Stone. Curriculum learning for reinforcement learning domains: A framework and survey. Journal of Machine Learning Research, 21(181):1–50, 2020.
- [24] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, et al. Anygpt: Unified multimodal llm with discrete sequence modeling. arXiv preprint arXiv:2402.12226, 2024.
- [25] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26439–26455, 2024.
- [26] Zhifei Xie and Changqiao Wu. Mini-omni2: Towards open-source gpt-4o with vision, speech and duplex capabilities. arXiv preprint arXiv:2410.11190, 2024.
- [27] Arushi Goel, Karan Sapra, Matthieu Le, Rafael Valle, Andrew Tao, and Bryan Catanzaro. Omcat: Omni context aware transformer. arXiv preprint arXiv:2410.12109, 2024.
- [28] Yadong Li, Jun Liu, Tao Zhang, Song Chen, Tianpeng Li, Zehuan Li, Lijun Liu, Lingfeng Ming, Guosheng Dong, Da Pan, et al. Baichuan-omni-1.5 technical report. arXiv preprint arXiv:2501.15368, 2025.
- [29] Henghui Du, Guangyao Li, Chang Zhou, Chunjie Zhang, Alan Zhao, and Di Hu. Crab: A unified audio-visual scene understanding model with explicit cooperation. arXiv preprint arXiv:2503.13068, 2025.
- [30] Guangzhi Sun, Wenyi Yu, Changli Tang, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Yuxuan Wang, and Chao Zhang. video-salmonn: Speech-enhanced audio-visual large language models. arXiv preprint arXiv:2406.15704, 2024.
- [31] Sanjoy Chowdhury, Sayan Nag, Subhrajyoti Dasgupta, Jun Chen, Mohamed Elhoseiny, Ruohan Gao, and Dinesh Manocha. Meerkat: Audio-visual large language model for grounding in space and time. In European Conference on Computer Vision, pages 52–70. Springer, 2024.
- [32] Zhuoming Liu, Yiquan Li, Khoi Duc Nguyen, Yiwu Zhong, and Yin Li. Pave: Patching and adapting video large language models, 2025. URL https://arxiv.org/abs/2503.19794.
- [33] Zijia Zhao, Haoyu Lu, Yuqi Huo, Yifan Du, Tongtian Yue, Longteng Guo, Bingning Wang, Weipeng Chen, and Jing Liu. Needle in a video haystack: A scalable synthetic evaluator for video mllms. arXiv preprint arXiv:2406.09367, 2024.
- [34] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024.

- [35] Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. arXiv preprint arXiv:2502.04326, 2025.
- [36] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [37] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.
- [38] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.
- [39] Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025.
- [40] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [41] OpenAI. Introducing gpt-4.1 in the api, 2025. URL https://openai.com/index/gpt-4-1 /.
- [42] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [43] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.
- [44] Zhiqi Li, Guo Chen, Shilong Liu, Shihao Wang, Vibashan VS, Yishen Ji, Shiyi Lan, Hao Zhang, Yilin Zhao, Subhashree Radhakrishnan, et al. Eagle 2: Building post-training data strategies from scratch for frontier vision-language models. arXiv preprint arXiv:2501.14818, 2025.
- [45] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.
- [46] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, Min Dou, Kai Chen, Wenhai Wang, Yu Qiao, Yali Wang, and Limin Wang. Internvideo2.5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386, 2025.
- [47] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. Videochat-flash: Hierarchical compression for long-context video modeling. arXiv preprint arXiv:2501.00574, 2024.
- [48] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025.
- [49] Pinci Yang, Xin Wang, Xuguang Duan, Hong Chen, Runze Hou, Cong Jin, and Wenwu Zhu. Avqa: A dataset for audio-visual question answering on videos. In Proceedings of the 30th ACM international conference on multimedia, pages 3480–3491, 2022.
- [50] Guangyao Li, Yake Wei, Yapeng Tian, Chenliang Xu, Ji-Rong Wen, and Di Hu. Learning to answer questions in dynamic audio-visual scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19108–19118, 2022.
- [51] Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chenliang Xu. Audio-visual event localization in unconstrained videos. In ECCV, 2018.
- [52] Tiantian Geng, Teng Wang, Jinming Duan, Runmin Cong, and Feng Zheng. Dense-localizing audio-visual events in untrimmed videos: A large-scale benchmark and baseline. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22942–22951, 2023.

- [53] Jatin Lamba, Abhishek, Jayaprakash Akula, Rishabh Dabral, Preethi Jyothi, and Ganesh Ramakrishnan. Cross-modal learning for audio-visual video parsing. 2021. URL https: //arxiv.org/abs/2104.04598.
- [54] Jinxing Zhou, Xuyang Shen, Jianyuan Wang, Jiayi Zhang, Weixuan Sun, Jing Zhang, Stan Birchfield, Dan Guo, Lingpeng Kong, Meng Wang, and Yiran Zhong. Audio-visual segmentation with semantics. International Journal of Computer Vision, pages 1–21, 2024.
- [55] Zhaohui Zheng, Ping Wang, Dongwei Ren, Wei Liu, Rongguang Ye, Qinghua Hu, and Wangmeng Zuo. Enhancing geometric factors in model learning and inference for object detection and instance segmentation. IEEE transactions on cybernetics, 52(8):8574–8586, 2021.
- [56] Anthropic. The Claude 3 Model Family: Opus, Sonnet, Haiku Anthropic. URL https: //www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_ Card_Claude_3.pdf.
- [57] Jiaming Han, Kaixiong Gong, Yiyuan Zhang, Jiaqi Wang, Kaipeng Zhang, Dahua Lin, Yu Qiao, Peng Gao, and Xiangyu Yue. Onellm: One framework to align all modalities with language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26584–26595, 2024.
- [58] Zhaowei Li, Qi Xu, Dong Zhang, Hang Song, Yiqing Cai, Qi Qi, Ran Zhou, Junting Pan, Zefeng Li, Van Tu Vu, et al. Groundinggpt: Language enhanced multi-modal grounding model. arXiv preprint arXiv:2401.06071, 2024.
- [59] Jing Liu, Sihan Chen, Xingjian He, Longteng Guo, Xinxin Zhu, Weining Wang, and Jinhui Tang. Valor: Vision-audio-language omni-perception pretraining model and dataset. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
- [60] Artemis Panagopoulou, Le Xue, Ning Yu, Junnan Li, Dongxu Li, Shafiq Joty, Ran Xu, Silvio Savarese, Caiming Xiong, and Juan Carlos Niebles. X-instructblip: A framework for aligning image, 3d, audio, video to llms and its emergent cross-modal reasoning. In European Conference on Computer Vision, pages 177–197. Springer, 2024.
- [61] Yunlong Tang, Daiki Shimada, Jing Bi, and Chenliang Xu. Avicuna: Audio-visual llm with interleaver and context-boundary alignment for temporal referential dialogue. arXiv e-prints, pages arXiv–2403, 2024.
- [62] Kaixiong Gong, Kaituo Feng, Bohao Li, Yibing Wang, Mofan Cheng, Shijia Yang, Jiaming Han, Benyou Wang, Yutong Bai, Zhuoran Yang, et al. Av-odyssey bench: Can your multimodal llms really understand audio-visual information? arXiv preprint arXiv:2412.02611, 2024.
- [63] Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Zekun Wang, Jian Yang, et al. Omnibench: Towards the future of universal omni-language models. arXiv preprint arXiv:2409.15272, 2024.
- [64] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.
- [65] Jiashuo Yu, Ying Cheng, Rui-Wei Zhao, Rui Feng, and Yuejie Zhang. Mm-pyramid: Multimodal pyramid attentional network for audio-visual event localization and video parsing. In Proceedings of the 30th ACM international conference on multimedia, pages 6241–6249, 2022.

#### A Benchmark Construction

##### A.1 Reference-Query Modality Definition

To support comprehensive evaluation of multimodal counting capabilities, our benchmark defines five distinct modal settings based on the combination of reference and query modalities. Each setting reflects a different reasoning requirement, depending on which modality is used to locate the counting target and query interval:

- • Visual-only: The model is required to both locate and count using only visual input. This setting targets scenarios where the counting question and answer are entirely grounded in visual content.
- • Audio-only: The model must rely solely on audio cues to identify the relevant segments and determine the count. This setting emphasizes sound-based reasoning, such as counting distinct audio events or speaker turns.
- • Visual-reference, Audio-query: The model uses visual input to locate the relevant temporal segment and then performs counting based on audio information within that scope (e.g., “How many people spoke in the scene showing the conference table?”).
- • Audio-reference, Visual-query: The model leverages audio to identify the query interval and subsequently counts visually observable targets in the corresponding segments (e.g., “How many people are visible when the sound of clapping occurs?”).
- • Joint Audio-Visual: Both audio and visual modalities are required to solve the task effectively. The model must integrate information across modalities to interpret the question, localize relevant segments, and produce an accurate count. This setting reflects complex scenarios where neither modality alone is sufficient, or where combining both provides more reliable counting signals.

##### A.2 Counting Target Definition

We define three types of counting targets in our benchmark to evaluate model capabilities from multiple perspectives:

- • Event: A temporally localized activity or incident in the video. Event counting requires the model to recognize distinct occurrences over time.
- • Object: A visually identifiable entity in the scene. Object counting involves detecting and enumerating instances. The model must avoid double-counting due to camera motion, scene transitions, or repeated appearances of the same entity.
- • Attribute: A clustering of objects based on shared visual or semantic properties. Rather than counting individual instances, the model must identify groupings according to the query attribute (e.g., “How many different clothing colors are worn by people?”).

##### A.3 More Dataset Statistics

As shown in Fig. 6 and Fig. 7, the videos selected from CG-Bench cover a diverse range of topics, including Life Record, Sports, Instruction, and TV Show, among more than ten categories. All videos are longer than 10 minutes, with the majority ranging from 20 to 30 minutes. This design encourages models to perform counting over long temporal contexts, which is essential for evaluating their capability in long-range temporal grounding and accumulation-based reasoning. This addresses a key limitation of existing counting benchmarks, which often rely on short clips that fail to capture the complexity of long-range dependencies.

##### A.4 Prompt for Generating Initial Question Proposals

You are given a video, and your task is to generate **count-based audio-visual reasoning questions** that can be answered by

analyzing a clearly defined segment of the video. ### **Definition of Count-Based Audio-Visual Reasoning Question** Each question should involve counting something that is either **audibly heard**, **visually seen**, or both, during a localized event in the video. The question must be specific, grounded in real scenes, and the answer must be **objectively verifiable** within a given time span. ### **Modalities and Reasoning Types**

Driving (19)

Special Scenes (34)

Animal & Pet (18)

Life Record (122)

Electonic/Social Gaming

(26)

Security & Health (15)

### Video (497)

Humor/funny (25)

Music & TV show (58)

Embodied Expert (34)

GUI (17)

Instruction & Knowledge (43) News (6)

Art & Culture (60)

Sports & Exercise (20)

###### Figure 6: Statistics of Video Content Categories.

30

25

NumberofVideos

20

15

10

5

0

10

12

14

16

18

20

22

24

26

28

30

32

34

36

38

40

42

44

46

48

50

52

55

57

59

Duration (minutes)

Figure 7: Statistics of Video Duration.

- - **A2V (Audio to Visual)**: The question is triggered by an **audio cue** (e.g., a sound, noise, or dialogue) which helps the model **identify the relevant time segment** in the video. Once the time segment is located using the audio cue, the question asks about **visual information** (e.g., objects, people, or actions) within that segment. For example, "When the dog starts barking, how many people are visible in the background?" The audio cue identifies the time segment (e.g., when the dog barks), and the question asks about what is seen visually during that segment.

- - **V2A (Visual to Audio)**: The question is triggered by a **visual cue** (e.g., someone entering the frame, a person performing an action) which helps the model **identify the relevant time segment** in the video. The question then asks about **auditory events** (e.g., how many sounds or spoken words are heard) that occur during that time segment. For example, "When the firefighter enters the building, how many sirens can be heard in the background?" Here, the visual cue (the firefighter entering) locates the time segment, and the question asks about sounds during that segment.
- - **AV (Audio + Visual)**: The question requires both **audio and visual cues** to accurately count objects/events. Both cues help locate the relevant time segment in the video, and then the question asks for a count based on the interaction between both modalities. For example, "When the presenter gestures to the audience, how many people respond verbally?" Both audio (the people speaking) and visual (the presenter’s gesture) cues are needed to define the time segment for counting.

- - **A (Audio Only)**: The question is based solely on **audio cues** to identify a relevant time segment, and the task is to count auditory events within that segment. For example, "How many door slams can be heard in the scene?" The audio cue (the door slams) defines the time segment, and the model counts the auditory events within that period.

- - **V (Visual Only)**: The question is based solely on **visual cues**, with no need for audio. The task involves counting visible objects or actions in the defined segment. For example, "How many people are wearing blue shirts in the crowd at the

park?" This is based purely on visual observation. ### **Constraints**

- 1. Each question must correspond to a **specific and bounded video segment**.

- 2. The **answer must be a count** (e.g., number of visual objects, number of auditory events, number of multimodal occurrences, etc.).

- 3. The answer must be **clearly determined** and not ambiguous within the video span.

- 4. The difficulty of the question should be specified based on perceptual complexity (e.g., occlusion, background noise, overlapping motion/sound).

- 5. Include a **diverse mix** of A2V, V2A, AV, A, and V questions in the output.

- 6. Avoid redundant or trivial questions.

- 7. The question should not include specific timepoints.

### **Output Format** Return your result as a **JSON array**, where each entry is a dictionary with the following fields:

- - ‘"question"‘: A clear and specific counting question.

- - ‘"type"‘: One of ‘"A2V"‘, ‘"V2A"‘, ‘"AV"‘, ‘"A"‘, or ‘"V"‘.

- - ‘"start_time"‘: Start of relevant video segment (‘"MM:SS"‘).

- - ‘"end_time"‘: End of relevant video segment (‘"MM:SS"‘).

- - ‘"counting_result"‘: The correct count answer.

### **Example Output** Enclose the JSON block within ‘<json></json>‘ tags.

<json> [

{

"question": "When the dog starts barking, how many people are visible in the background?", "type": "A2V", "start_time": "01:15", "end_time": "01:45", "counting_result": 3

}, {

"question": "When the firefighter enters the building, how many sirens can be heard in the background?", "type": "V2A", "start_time": "02:00", "end_time": "02:30", "counting_result": 2

}, {

"question": "When the presenter gestures to the audience, how many people respond verbally?", "type": "AV", "start_time": "03:00", "end_time": "03:40", "counting_result": 4

}, {

"question": "How many door slams can be heard in the scene?", "type": "A", "start_time": "05:15", "end_time": "05:45",

- "counting_result": 3

}, {

"question": "How many people are wearing different colors of clothes in the park?", "type": "V", "start_time": "02:00", "end_time": "02:30",

- "counting_result": 4

}

] </json>

- A.5 Evaluation Prompts

- A.5.1 Black-Box Evaluation

Watch the video and answer the question ’{Question Here}’ with a number. Just output the number itself, don’t output anything else.

- A.5.2 White-Box Evaluation (Event)

Watch the video and provide your answer to the question ’{Question Here}’, including the start and end timestamps for each event. Format your answer in JSON, enclosed in <answer> and </answer> tags. The output should look like this: <answer>[[\" start_time\", \"end_time\"], ...]</answer>. Ensure each timestamp is in seconds (e.g., ’xx.xx’).

- A.5.3 White-Box Evaluation (Object)

According to the given video frames, answer the question ’{Question Here}’, including the bounding box for the query object in the first frame where it appears. For subsequent frames where the object appears, do not provide the bounding box again. Format your answer in JSON, enclosed within <answer> and </answer> tags. The output should look like this: <answer>{{\" Frame1\": [[x_min, y_min, x_max, y_max]], \"Frame2\": [...],...}}</answer>. In the output, each frame should either contain the bounding box of the object (if it appears for the first time in that frame) or an empty list ‘[]‘ (if the object does not appear or it has already been labeled in a previous frame). Ensure that bounding boxes are listed as [x_min, y_min,

- x_max, y_max].

A.5.4 White-Box Evaluation (Attribute)

According to the given video frames, answer the question ’{Question Here}’, clustering the objects based on the question. For each unique cluster, assign a unique label and return the bounding box for each object in the first frame where it appears. For subsequent frames where the object appears, do not output anything. Format your answer in JSON, enclosed within

<answer> and </answer> tags. The output should look like this: <answer>{{\"Frame 1\": [{{\"bbox\": [x_min, y_min, x_max,

- y_max], ’label’: \"Label 1\"}}], \"Frame 2\": [...], ...}}</answer>. In the output, each frame should either contain the bounding box and label for the object (if it appears for the first time in that frame) or an empty list ‘[]‘ (if the object has already been labeled or does not appear in that frame). The label should correspond to a unique object cluster according

to the question.

|0<br><br>0.1<br><br>0.2<br><br>0.3<br><br>0.4<br><br>0.5<br><br>Object Event Attribute<br><br>Acc<br><br>Count Target<br><br>GPT-4o GPT-4.1 Gemini 2.5 Flash Gemini 2.5 Pro SEED-1.5 VL<br><br>InternVL3-8B Eagle2.5-8B Eagle2-9B Qwen2.5-VL-7B VideoLLaMA3-7B<br><br>InternVideo2.5-8B LLaVA-NeXT-Video-7B MiniCPM-V 2.6 UnifiedIO-2 XXL VideoLLaMA2.1-7B-AV<br><br>Ola-7B Qwen2.5-Omni-7B AV-Reasoner (Ours)<br><br>|
|---|

Acc

###### Figure 8: Model accuracy across different count targets.

0.6

0.5

0.4

Acc

0.3

0.2

0.1

0

0~5 5~10 10~15 15~20 20~25 ≥25

Count Number

GPT-4o GPT-4.1 Gemini 2.5 Flash Gemini 2.5 Pro

SEED-1.5 VL InternVL3-8B Eagle2.5-8B Eagle2-9B

Qwen2.5-VL-7B VideoLLaMA3-7B VideoChat-Flash-7B InternVideo2.5-8B

MiniCPM-V 2.6 UnifiedIO-2 XXL VideoLLaMA2.1-7B-AV Ola-7B

Qwen2.5-Omni-7B AV-Reasoner (Ours)

Figure 9: Model accuracy across different count ranges.

- A.6 More Evaluations

- A.6.1 Model Performance across Different Counting Targets

Fig. 8 highlights a clear trend across both close-source and open-source MLLMs: models consistently perform better on object counting tasks compared to event and attribute counting. Most models demonstrate a noticeable advantage when counting concrete, visually grounded entities like objects. This performance gap is especially prominent in open-source models, where accuracy on event and attribute targets often drops substantially.

The relative ease of object counting can be attributed to the more direct visual correspondence between input and target. Objects are typically well-defined spatially, consistently annotated in vision-language pretraining data, and often associated with discrete visual regions. In contrast, events may unfold over time and require temporal reasoning, while attributes tend to be abstract, context-dependent, or even implicit, making them harder to detect and quantify reliably.

These findings suggest that current MLLMs are more adept at processing perceptually salient elements. Addressing the challenges posed by more abstract or temporally extended targets like events and attributes may require stronger temporal modeling, better multimodal alignment, or targeted supervision in future model designs.

- A.6.2 Model Performance across Different Counting Numbers

As shown in the Fig. 9, model performance clearly declines as the count range increases. Open-source models exhibit relatively high accuracy when the count is ≤5, but their performance becomes more erratic and less reliable when the count exceeds 5.

|0<br><br>0.1<br><br>0.2<br><br>0.3<br><br>0.4<br><br>0.5<br><br>0.6<br><br>0.7<br><br>A A2V AV V V2A<br><br>Acc<br><br>Query Modality<br><br>GPT-4o GPT-4.1 Gemini 2.5 Flash Gemini 2.5 Pro SEED-1.5 VL<br><br>InternVL3-8B Eagle2.5-8B Eagle2-9B Qwen2.5-VL-7B VideoLLaMA3-7B<br><br>VideoChat-Flash-7B InternVideo2.5-8B MiniCPM-V 2.6 UnifiedIO-2 XXL VideoLLaMA2.1-7B-AV<br><br>Ola-7B Qwen2.5-Omni-7B AV-Reasoner (Ours)<br><br>|
|---|

Acc

###### Figure 10: Model accuracy across different query modalities.

0.7

40

35

0.6

30

0.5

25

0.4

###### WCS

20

0.3

15

0.2

10

0.1

5

0

0

Event

Object Attribute

GPT-4o GPT-4.1 Gemini 2.5 Flash Gemini 2.5 Pro

SEED-1.5 VL InternVL3-8B Eagle2.5-8B Eagle2-9B

Qwen2.5-VL-7B VideoLLaMA3-7B MiniCPM-V 2.6 UnifiedIO-2 XXL

VideoLLaMA2.1-7B-AV Ola-7B Qwen2.5-Omni-7B AV-Reasoner (Ours)

Figure 11: Model’s White-box Evaluation Performance across Different Counting Targets.

- A.6.3 Model Performance across Different Query Modalities

- As shown in Fig. 10, model performance varies significantly across different query modalities. When the query modality is audio (A), most models—especially omni-MLLMs like Qwen2.5-Omni-7B and Ola-7B achieve relatively high accuracy and low error, indicating strong alignment with audio inputs. However, performance drops noticeably in cross-modal settings such as A2V and V2A, where both MAE and RMSE increase substantially. Notably, V2A and AV emerges as the most challenging configuration across all metrics. Overall, omni-MLLMs (purple) show clear advantages over VLMs (yellow) in audio-involved queries, but this advantage diminishes in purely visual or AV scenarios.

A.6.4 Model’s White-box Evaluation Performance across Different Counting Targets

- As shown in Fig. 11, in the white-box evaluation, most models demonstrate relatively strong reasoning counting abilities in the event counting, but perform poorly in the object and attribute counting. Notably, only MiniCPM-V 2.6 [64] achieves a score in the attribute counting task. This suggests that current models are relatively proficient in temporal grounding, where understanding the sequence and timing of events is crucial, but perform poorly in spatial grounding, which requires precise localization and differentiation of objects or attributes.

#### B Experiment Details

##### B.1 Experimental Setups

All experiments are conducted using the TRL framework on 8 NVIDIA A100 GPUs (each with 80GB memory). The base model used is Ola-Omni-7B. Training was performed with bf16 enabled to improve memory efficiency and computational performance.

We adopted the GRPO algorithm for fine-tuning. During the rollout phase, the generation temperature was set to 1.0, the maximum number of generated tokens is 256 or 1024 according to different tasks, and each sample was expanded into 8 rollout trajectories. The KL divergence penalty coefficient β was set to 0.1 to ensure controlled deviation from the initial policy.

Table 9: Our detailed training settings. Data indicated in parentheses refers to that used in the SRM.

Cold-start SFT Curriculum-based RL (QA) Curriculum-based RL (Grounding) Curriculum-based RL (Counting) Full-task RL Data

Dataset AVTG+ARIG+Counting AVQA AVTG+ARIG (AVQA) Counting (AVTG+ARIG+AVQA) AVQA+AVTG+ARIG+Counting #Samples 78K 72K 72K (14K) 6K (1K) 10K

Thinking ✗ ✓ Max New Tokens - 256 1024

Training

Table 10: Performance comparison across AVQA and AVE Benchmarks.

Benchmark MM-Pyramid [65] MEERKAT [31] PAVE [32] Crab [29] AV-Reasoner (Ours) AV-Reasoner-Thinking (Ours)

AVQA Acc (%) - 87.17 93.80 - 93.02 93.17 AVE Acc (%) 77.80 - - 80.15 82.86 81.26

##### B.2 Detailed Training Settings

Tab. 9 summarizes the training configurations across different stages. The curriculum-based RL process is divided into three subtasks: QA, grounding, and counting, each trained with its corresponding dataset. To mitigate forgetting and maintain performance across tasks, we adopt a stage review mechanism (SRM), which involves mixing a portion of previously seen samples during training. The number of such samples is shown in parentheses.

#### C Evaluation on More Benchmarks

The performance of our model is also evaluated on the AVQA and AVE benchmarks. The test results are shown in the Tab. 10.

