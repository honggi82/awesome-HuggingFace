## REASONMAP: Towards Fine-Grained Visual Reasoning from Transit Maps

Sicheng Feng1,2,†, Song Wang3,2,†, Shuyi Ouyang3,2, Lingdong Kong2, Zikai Song4,2, Jianke Zhu3, Huan Wang1,∗, Xinchao Wang2

1Westlake University 2National University of Singapore 3Zhejiang University 4Huazhong University of Science and Technology

Dataset & Toolkit: https://fscdc.github.io/ReasonMap

# arXiv:2505.18675v3[cs.CV]12Mar2026

†Equal contribution. ∗Corresponding author: wanghuan@westlake.edu.cn

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Question Difficulty

Map Difficulty

[Figure 8]

[Figure 9]

Arrival

Hard

[Figure 10]

Difficulty Label

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

- • Route Name
- • Departure Stop
- • Arrival Stop
- • Via Stops
- • Num of Via Stops

Gaode Map

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Departure

[Figure 20]

Google Map

Resolution:

[Figure 21]

Reference Route

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Key Question: According to the subway map, how do I get from [Promenade] to [Farrer Park]?

Key Question

[Figure 27]

[Figure 28]

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Format:
- - Route Name: “East West Line”
- - Departure Stop: “xx Station”
- - Arrival Stop: “xx Station”
- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
- - Route Name: “East West Line”
- - Departure Stop: “xx Station”
- - Arrival Stop: “xx Station”

[Figure 29]

[Figure 30]

Medium

Route Name Departure Stop Arrival Stop

[Figure 31]

[Figure 32]

[Figure 33]

Downtown Line (Promenade Station - Station)

[Figure 34]

[Figure 35]

-

- - - - - - - - - - - - - - - - - - - - - - -

Route Name Departure Stop Arrival Stop

North East Line ( Station - Farrer Park Station)

[Figure 36]

[Figure 37]

Figure 1. Overview of REASONMAP. We present a novel benchmark tailored for evaluating the fine-grained visual reasoning capabilities of MLLMs. The dataset comprises 1,008 question-answer pairs derived from high-resolution transit maps across 30 cities in 13 countries, featuring diversely structured questions. Further details on the dataset construction pipeline are provided in Section 3.

### Abstract

designed to evaluate these capabilities. REASONMAP encompasses high-resolution transit maps from 30 cities and includes 1,008 question-answer pairs spanning two question types and three templates. Furthermore, we design a two-level evaluation pipeline that properly assesses answer correctness and quality. Our comprehensive evaluation of 16 popular MLLMs reveals a counterintuitive pattern: among open-source models, base variants outperform their reasoning-tuned counterparts, whereas the opposite trend is observed in closed-source models. Further analysis

Multimodal large language models (MLLMs) have demonstrated significant progress in semantic scene understanding and text-image alignment, with reasoning variants enhancing performance on more complex tasks involving mathematics and logic. However, their proficiency in tasks requiring both fine-grained visual understanding and spatial reasoning remains underexplored. To bridge this gap, we introduce REASONMAP, a novel benchmark specifically

under the visual-masking setting confirms that strong performance necessitates direct visual grounding, rather than relying solely on language priors. We further establish a training baseline with reinforcement fine-tuning, providing a reference for future exploration. We hope this benchmark study offers new insights into visual reasoning and helps investigate the gap between open- and closed-source models.

### 1. Introduction

Multimodal large language models (MLLMs) [1, 3, 29, 42, 102] have recently achieved notable advancements across a range of vision-language tasks, including visual grounding [50, 76, 80, 83], reasoning segmentation [9, 37, 53, 70, 96], and text-image alignment [85, 88]. Building upon these developments, reasoning MLLMs [6, 22, 47, 49, 51, 64, 74] have further improved performance on complicated visual reasoning tasks such as visual math problems [67, 84], visual question answering (VQA) [59, 67, 87], and spatial reasoning [11, 16, 40, 59]. These capabilities are critical for a wide range of real-world applications, including embodied AI and autonomous driving [10, 12, 13, 68]. As visual tasks grow in complexity and practical relevance, the need for rigorous benchmarks to assess complex visual reasoning becomes increasingly essential.

To address the growing demand for the evaluation of visual reasoning, several benchmarks have been proposed. Existing multimodal reasoning datasets, such as MathVQA [67], MMMU [87], and MathVerse [92], primarily assess symbolic or mathematical reasoning, where visual understanding plays a limited role. Conversely, datasets emphasizing fine-grained visual comprehension and retrieval like VisuLogic [79], VisualPuzzles [60], and V∗Bench [75] require detailed perception but minimal planning or spatial reasoning. CityBench [13], DriveBench [76], and MapBench [77] take a step toward spatial reasoning while remain coarse in granularity. In contrast, our REASONMAP bridges these gaps by jointly evaluating fine-grained visual understanding, spatial reasoning and planning over highresolution and information-dense transit maps.

In this paper, we introduce REASONMAP (Figure 1), a benchmark designed to evaluate the fine-grained visual understanding and spatial reasoning capabilities of MLLMs using high-resolution transit maps. As structured and information-dense visual artifacts, maps inherently require precise spatial interpretation, making them well-suited for assessing detailed visual reasoning. REASONMAP comprises 1,008 human-verified question-answer pairs spanning 30 cities across 13 countries. Each instance includes a map, two stops, two questions (short and long), multiple reference routes, and two difficulty labels (map and question difficulty). The questions cover two types and three question templates capturing both coarse and fine-grained

spatial reasoning. To ensure data quality, we perform manual route verification, promote question diversity, and balance difficulty distribution. For evaluation, we propose a two-level framework that independently measures answer correctness (via accuracy) and quality (via a proposed map score), reflecting both feasibility and efficiency.

We conduct comprehensive experiments on 16 widelyused MLLMs, encompassing both base and reasoning models. Our results reveal a counterintuitive finding: among open-source models, base variants outperform their reasoning counterparts, whereas the opposite holds for closedsource models. Moreover, when only textual inputs are provided, models can still answer some questions based on inner knowledge, but in most cases, their performance noticeably drops. This highlights a critical limitation in the current model behavior. While some models can leverage prior knowledge and textual cues to solve certain tasks, fine-grained visual reasoning tasks requiring genuine understanding still necessitate effective integration of multimodal information for robust reasoning.

We further establish a training baseline with Group Relative Policy Optimization (GRPO) [57]. Guided by our evaluation framework, we design two reward components (i.e., accuracy reward and format reward) as reinforcement signals to optimize model behavior during training. Under a cross-city setting, where training and test maps are completely disjoint, the reinforcement fine-tuning consistently improves performance across models of different scales.

Our main contributions are summarized as follows: (1) We develop an extensible, semi-automated pipeline for dataset construction, facilitating scalable expansion to additional maps and cities ; (2) We propose a structured twolevel evaluation framework that separately quantifies answer correctness and quality using accuracy and the proposed map score, respectively, enabling fine-grained answer assessment; (3) A comprehensive benchmarking study is conducted across 16 MLLMs, providing insights into model performance, robustness, and the interplay between visual and textual cues, thereby informing future research on visual reasoning; and (4) We establish a training baseline with reinforcement fine-tuning as a reference for future research.

### 2. Related Work

Reasoning in LLMs & MLLMs. Recent advances in large language models (LLMs) have demonstrated significant improvements in reasoning capabilities through reinforcement fine-tuning paradigms [15, 22, 26, 47], which leverage GRPO [57] to unlock the reasoning potential of LLMs. This paradigm has also been extended to the multimodal domain, with increasing interest in applying reinforcement learning (RL) to visual reasoning [35, 44, 58, 61, 65]. Both open-source and closed-source communities have introduced advanced reasoning MLLMs built upon earlier

- Table 1. Comparison between REASONMAP and existing multimodal reasoning datasets. For entries in the dataset size column with notation like (×n), each base problem has multiple versions to enforce visual grounding. Specifically, VGRP-Bench is constructed by sampling over 20 core puzzles; MathVerse generates six multimodal variants per problem with different levels of visual and textual information.

Name Year Dataset Size Avg. Resolution Training Set Step Evaluation Multilingual

MMMU [87] 2024 11.5k 684 × 246 ✗ ✗ ✓ MathVerse [92] 2024 2,612 (×6) 577 × 487 ✗ ✗ ✗ VisuLogic [79] 2025 1,003 601 × 331 ✓ ✗ ✗ VisualPuzzles [60] 2025 1,168 767 × 464 ✗ ✗ ✗ VGRP-Bench [52] 2025 20 (×5) 790 × 790 ✗ ✓ ✗ R-Bench [23] 2025 665 629 × 348 ✗ ✗ ✓ V∗Bench [75] 2023 191 2,246 × 1,582 ✗ ✗ ✗

REASONMAP 2025 1,008 (×2) 5,839 × 5,449 ✓ ✓ ✓

systems [3, 48, 82, 102]. Notable open-source models include Kimi-VL [64], Skywork-R1V [49, 74], and QwenQvQ [51], whereas Doubao-1.5-Pro [6], Seed1.5-VL [21], OpenAI o3 [48], OpenAI 4o [46], and Gemini [19] represent leading closed-source efforts. Despite recent progress, systematic evaluation of fine-grained visual reasoning in MLLMs remains limited, as existing benchmarks primarily target coarse-grained tasks and fail to capture model performance on complex real-world visual content.

Multimodal Reasoning Datasets. As multimodal reasoning has rapidly progressed, various benchmarks have emerged to evaluate MLLMs across different reasoning dimensions (see summary in Table 1). Datasets such as V∗Bench [75], VisualPuzzles [60], VisuLogic [79], and VGRP-Bench [52] primarily examine abstract visual reasoning through synthetic tasks involving logic, structure, and pattern recognition. In parallel, MapBench [77], CityBench [13], and DriveBench [76] shift focus to real-world spatial reasoning, assessing model performance on urban and autonomous driving scenarios. For mathematical reasoning, benchmarks like MathVQA [67], MMMU [87], and MathVerse [92] integrate multimodal inputs, with MathVerse notably introducing varied problem formats to strengthen visual dependence. Geospatial benchmarks [36, 81, 86] have also been proposed to evaluate spatial reasoning and map understanding capabilities. Unlike previous benchmarks that primarily emphasize either fine-grained visual understanding (e.g., V∗Bench) or spatial reasoning (e.g., MapBench), our REASONMAP benchmark jointly evaluates both aspects with high-resolution transit maps.

Map-based Spatial Reasoning. Among the many directions of multimodal reasoning, map-based spatial reasoning has emerged as a crucial area, with broad applications in navigation, urban planning, and autonomous systems [5, 54, 69, 71, 78]. Recent efforts have focused on enabling models to interpret and reason over various types of map data. CityBench [13] provides a dataset for evaluating urban scene understanding, while MapLM [7] introduces a benchmark for map and traffic scene understanding. PlanAgent [100] and PReP [90] explore embodied planning in environments that require interpreting map in-

formation. MapEval [11] proposes a structured evaluation suite for map reasoning, and GeoNav [78] investigates geospatial navigation using LLMs. Most existing methods [11, 13, 100] depend on external tools (e.g., map services or APIs) to complete spatial tasks, which often bypasses the need for genuine visual reasoning. However, spatial reasoning based on visual understanding remains essential. Our work aims to fill this gap by evaluating such capabilities without tool assistance.

### 3. REASONMAP Construction

In this section, we first present the complete dataset building pipeline as shown in Figure 2, which consists of three main stages: (1) data collection and preprocessing, (2) question–answer pair construction, and (3) quality control. We then report comprehensive statistics of the dataset.

#### 3.1. REASONMAP Building Pipeline

##### 3.1.1. Data Collection and Preprocessing

We collect and manually select 30 high-resolution transit maps covering 30 cities across 13 countries from publicly available online sources, in compliance with relevant licenses and regulations, ensuring diversity and a balanced range of map difficulty. We then leveraged MLLMs (e.g., GPT-4o) to extract the names of transit lines and their corresponding stops, followed by manual correction to ensure correctness. Special cases like transfer stops and branchstarting stops were annotated in a standardized format appended to the respective stop entries. Finally, for subsequent usage, all route and stop information was saved in a unified JSON format, referred to as the Metro Data.

##### 3.1.2. Question-Answer Pair Construction

The construction of question–answer pairs involves three key steps: (1) Question Generation, where we formulate questions based on predefined templates; (2) Reference Route Collection, where we obtain corresponding reference routes using Google Map and Gaode Map; and (3) Label Annotation, where we properly assign difficulty labels for both maps and questions.

|4. Annotation|
|---|
|[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>Map Difficulty<br><br>Question Difficulty<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>Manual Annotate<br><br>Transfer Count<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]|

|5. Quality Control|
|---|
|[Figure 52]<br><br>[Figure 53]<br><br>Manual Check<br><br>[Figure 54]<br><br>[Figure 55]<br><br>Correct -ness<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>Divers -ity<br><br>Difficulty Balance<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>Fix<br><br>or<br><br>Drop<br><br>[Figure 68]<br><br>[Figure 69]<br><br>Adjust Question Ratio|

|2. Question-Answer Generation|
|---|
|[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>Element<br><br>Stop 1<br><br>Question Template<br><br>[Figure 75]<br><br>[Figure 76]<br><br>Stop 2 Short Long 1 Long 2<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>According to the subway map, how do I get from Stop 1 to Stop 2?<br><br>Key Questions<br><br>Format Template<br><br>Short<br><br>[Figure 81]<br><br>Long 1<br>Long 2<br><br><br>Via Stops: xx Station, …<br><br>Number of Via Stops: xx<br><br>Route Name: xxx Departure Stop: xx Station Arrival Stop: xx Station<br><br>-Route Name: xxx Departure Stop: xx Station Arrival Stop: xx Station|

|3. Reference Route|
|---|
|[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>Google Map<br><br>Gaode Map<br><br>Map Service<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>URL Request<br><br>Google Map API Request<br><br>Route Info Section<br><br>• Route<br>• Departure Stop<br>• Arrival Stop<br>• Via Stops<br>• Num of Via Stops<br><br><br>[Figure 90]<br><br>[Figure 91]|

|1. Data Collect & Preprocess|
|---|
|[Figure 92]<br><br>Transit Map Collection<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>MLLM Annotate<br><br>[Figure 97]<br><br>[Figure 98]<br><br>Manual Correction<br><br>Extract of Line-Stop<br><br>North East Line: [Stop ne1, ...] Circle Line: [Stop c1, ...] …<br><br>[Figure 99]<br><br>North East Line<br><br>[Figure 100]<br><br>[Figure 101]<br><br>Circle Line<br><br>Metro Data<br><br>[Figure 102]<br><br>[Figure 103]|

- Figure 2. The building pipeline of REASONMAP consists of three main stages: (1) data collection and preprocessing, (2) question–answer pair construction, and (3) quality control. Steps (2-4) in the figure correspond to the question–answer pair construction stage.

Question Generation. We randomly select two stops (refer to stop1 and stop2) from the current high-resolution transit map. We then generate one short question and one long question based on predefined question templates and two stops (Figure 2). The short question has only one fixed template, while the long question is randomly assigned one of two available templates during generation. Additionally, the two long question templates differ in focus: one asks for the number of via stops, while the other requires identifying each via stop (see detailed templates in Appendix A.1).

##### 3.1.3. Quality Control

To ensure the reliability and balance of the dataset, we perform quality control from three perspectives: correctness, diversity, and difficulty balance (more details in Appendix A.3). Incorrect question–answer pairs are either manually corrected or discarded. We then involve both automatic checks and manual adjustments to ensure consistency and coverage across all difficulty levels. One reserved example is shown in Figure 1.

#### 3.2. Dataset Statistics

Reference Route Collection. For each question, we query all valid transit routes between stop1 and stop2 using APIs from map services (e.g., Gaode Map for Chinese cities and Google Map for other cities). The retrieved routes are stored in a unified format containing relevant metadata (e.g., route name, departure stop, arrival stop, via stops, and number of via stops). We discard routes that cannot be visually traced on the map, ensuring consistency with the visual content.

The REASONMAP consists of 30 high-resolution transit map images (see map sources in Appendix A.4) with an average resolution of 5,839 × 5,449 pixels. In total, it contains 1,008 question–answer pairs. The distribution of question difficulty is as follows: 57.7% are labeled as easy, 34.4% as medium, and 7.8% as hard. Additionally, a subset of 312 samples is manually selected as the test set for the benchmark experiments described in Section 5, while the remaining samples serve as a training set. To ensure diversity and difficulty balance, the test set includes 11 cities with a 4 : 3 : 4 map difficulty ratio and a question difficulty distribution (181 easy, 108 medium, 23 hard) that maintains consistency with the full dataset. Moreover, REASONMAP includes inter-modal transfers in cities like Sydney, where subways, light rail, and airport lines converge.

Label Annotation. Two levels of difficulty labeling are included in this stage. For map difficulty, we assign each map to one of three levels (easy, medium, hard) based on the number of transit lines and transfer stations, ensuring a balanced split of 10 maps per level. For question difficulty, we assign difficulty based on the number of transfers in the reference route: routes with no transfers are labeled as easy, those with one transfer as medium, and all others

- as hard. To ensure balance, we set a fixed difficulty distribution threshold of 20 : 15 : 5 (easy:medium:hard) for each map, generating 40 questions. Once the quota for a difficulty level is reached on a given map, no additional questions of that level are retained. Additionally, we provide a more fine-grained taxonomy of questions in Appendix A.2.

### 4. Evaluation Framework & Training Baseline

In this section, we first present a two-level evaluation framework for systematically assessing model performance, which separately measures the correctness and quality of model-generated answers. We further introduce a GRPO-

based training baseline with carefully designed rewards.

Preparation for Evaluation. We first parse the modelgenerated answers according to the required format. Answers that do not comply with the specified format or cannot be parsed due to model hallucination [4] are marked as wrong answer. For the correctness evaluation, we utilize the Metro Data mentioned in Section 3.1.1 as ground truth. For the quality evaluation, we adopt the collected reference routes as presented in Section 3.1.2 as the ground truth.

#### 4.1. Correctness Evaluation

We evaluate the correctness of the answer using Algorithm 1 in Appendix B. Specifically, the evaluation checks the correctness of the overall departure and arrival stops (stop1 and stop2), verifies if the route name of each segment exists in the Metro Data, ensures the departure and arrival stops are valid for each segment, and confirms that transfer stops between consecutive segments are consistent. An answer is considered correct only if all the above checks are satisfied. Additionally, we apply the same correctness evaluation algorithm to the answers of short and long questions.

#### 4.2. Quality Evaluation

To evaluate the quality of the answers, we introduce a unified scoring metric, referred to as the map score, which is applied to both short and long questions using the evaluation procedure (see Algorithm 2 in Appendix B). The overall evaluation framework for route quality follows a structure similar to that used in Section 4.1. The following evaluation procedure assumes a single reference route for simplicity. In practice, if multiple reference routes are available, the answer is evaluated against each of them, and the highest score is taken as the final map score.

For short questions, the map score solely focuses on route-level and endpoint consistency, excluding all longquestion-specific parts. We compute the score by comparing segment pairs in the answer and reference route. Specifically, correctly matching stop1 and stop2 contributes one point, matching the route name adds two points, and matching the departure and arrival stops within each route segment provides one point each. The final score is capped

- at 10, and an additional bonus is awarded if the answer is judged correct based on the correctness evaluation procedure described in Section 4.1. This design ensures that a correct answer always receives a higher score than any incorrect one. For long questions, the evaluation incorporates additional scoring components tailored to the two question templates introduced in Section 3.1.2. These components are designed to capture the increased reasoning depth required in long-form responses. As with short questions, a bonus score is also added for correct answers. The two additional scoring components are detailed below. Via Stop Count Evaluation. For long questions that re-

quire models to predict the number of via stops for each segment, we introduce the num via stop score. This score compares the via stop count of the answer and reference route by computing the absolute error and mapping it to a fixed score (4). A perfect match yields full points, while larger discrepancies receive proportionally lower scores. The score is then capped at 10 for the full route.

Specific Via Stop Evaluation. For long questions that require explicit enumeration of intermediate stops, we compute via stop score using a combination of two factors: the number of correctly matched via stops, and the intersectionover-union (IoU) between via stop sets of the answer and reference route. The final score for this component is obtained by averaging the IoU score (scaled to 10) and the exact match count (capped at 10), and then clipped to 10.

#### 4.3. Training Baseline

Building upon the evaluation framework established above, we introduce a strong training baseline. This baseline serves not only to validate the efficacy of our proposed metrics but also to establish a robust performance benchmark for the REASONMAP. The metrics defined in our framework, such as route correctness and map score (see in Section 4.1 and

- 4.2), are rule-based and non-differentiable. Consequently, standard supervised fine-tuning (SFT) is insufficient as it cannot directly optimize for these desired outcomes. Therefore, we leverage reinforcement learning (RL) to explicitly align the model outputs with our evaluation criteria.

Specifically, we fine-tune Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct [3] on the REASONMAP training set using RL via the GRPO procedure [57] (see detailed GRPO optimization process in Appendix B.4). The reward is designed in accordance with our evaluation framework, consisting of two components: (1) accuracy reward, which provides a binary signal following the correctness evaluation; and (2) format reward, which encourages parsable outputs by rewarding correct formats and penalizing violations. The resulting RL-trained models are evaluated across unseen cities, enabling a cross-city validation of generalization ability (see results in Section 5.2.3).

- 5. Experiments

#### 5.1. Experimental Setups

We conduct extensive benchmark experiments on REASONMAP using 16 popular MLLMs under different inference settings, analyzing their performance and comparing results. Several interesting insights emerge from this comparison. The detailed experimental settings are as follows.

Evaluated Models. We evaluate a diverse set of MLLMs categorized into two groups based on whether they are reasoning-oriented models with a long-thinking process. Reasoning-oriented models include: Skywork-

- Table 2. Evaluations of various MLLMs on REASONMAP. S. represents results for short questions, while L. denotes results for long questions. The map score is capped at 20 for short questions, while for long questions, the maximum score is 40. Bold indicates the best results among open-source and closed-source models, respectively, while underline represents the second best.

Model Type Weighted Acc. (S.) #Tokens (S.) Weighted Acc. (L.) #Tokens (L.) Weighted Map Score (S. / L.) Open-source Models

Qwen2.5-VL-3B-Instruct [3] Base 8.68% 42 7.99% 151 2.75 / 3.70 Qwen2.5-VL-7B-Instruct [3] Base 13.28% 26 7.12% 57 4.01 / 5.74 Qwen2.5-VL-32B-Instruct [3] Base 16.49% 36 15.71% 112 3.88 / 6.84 Qwen2.5-VL-72B-Instruct [3] Base 26.65% 33 24.22% 104 5.09 / 8.80 InternVL3-38B [102] Base 14.84% 43 13.45% 68 3.48 / 6.31 InternVL3-78B [102] Base 25.35% 33 19.62% 62 4.80 / 7.50 Kimi-VL-A3B-Instruct [64] Base 12.76% 41 12.33% 41 3.30 / 5.37 Kimi-VL-A3B-Thinking [64] Reasoning 5.47% 754 5.47% 1,287 2.44 / 3.17 Skywork-R1V-38B [49, 74] Reasoning 6.86% 645 3.21% 842 2.11 / 3.11 QvQ-72B-Preview [51] Reasoning 9.03% 1,279 4.25% 1,619 1.59 / 1.55

Closed-source Models

Doubao-115 [6] Base 34.20% 32 38.02% 118 5.25 / 11.96 OpenAI 4o [46] Base 41.15% 34 42.80% 58 6.84 / 13.57 Doubao-415 [6] Reasoning 43.14% 536 46.09% 1,796 7.33 / 14.67 Doubao-428 [6] Reasoning 37.15% 532 37.85% 2,167 5.52 / 11.73 Gemini-2.5-Flash [19] Reasoning 46.09% 806 29.86% 1,419 7.64 / 9.98 OpenAI o3 [48] Reasoning 63.02% 1,236 59.11% 2,372 9.53 / 17.96

R1V-38B [49, 74], QvQ-72B-Preview [51], Kimi-VLA3B-Thinking/Instruct [64], OpenAI o3 [48], Gemini-

- 2.5-Flash [19], Doubao-1-5-thinking-vision-pro-250428 (Doubao-428), and Doubao-1.5-Thinking-Pro-M-250415 (Doubao-415) [6]. Base models include: Qwen2.5-VL series (3B, 7B, 32B, 72B) [3], InternVL3 series (38B, 78B) [102], OpenAI 4o [46], and Doubao-1.5-Vision-Pro32k-250115 (Doubao-115) [6]. Additionally, the Doubao 1.5 Pro series has an activated parameter size of 20B. Inference Settings. For open-source models, we set the max output token limit to 2,048, which is sufficient to cover complete generations, even for reasoning-oriented models, while keeping other parameters consistent with the official HuggingFace configurations. All open-source models are deployed using PyTorch and the HuggingFace Transformers library. For closed-source models, we access their official APIs for evaluation and follow the default settings provided by each model’s official documentation. We further discuss the diverse image processing strategies when handling high-resolution visual inputs in Appendix B.2. Training Settings. For GRPO-based RL training, we use AdamW with an initial learning rate of 1.0×10−6 and a KL divergence coefficient of 1.0×10−3. We sample 8 responses every query and set the global batch size to 16. The training data is the training set from REASONMAP. Difficulty-Aware Weighting. To emphasize models that are capable of addressing more challenging cases, we incorporate a difficulty-aware weighting scheme when computing the metrics. Samples with higher map or question difficulty are assigned larger weights, allowing the evaluation to more faithfully reflect a model’s robustness. The

complete weighting details are provided in Appendix B.3.

#### 5.2. Experimental Results

##### 5.2.1. Performance of MLLMs with Full Input

The principal results are summarized in Table 2 (see more fine-grained metrics in Table A3 of the Appendix). We first analyze the effect of model size by examining performance within the same architecture series. Qwen2.5-VL and InternVL series show a consistent trend: larger models achieve better accuracy with fewer tokens, suggesting that the scaling law [33] continues to hold even in fine-grained visual reasoning tasks. Figure A1 in Appendix presents accuracy distributions across different combinations of question and map difficulty. As expected, performance degrades as task complexity increases. Additionally, Figure A2 in Appendix illustrates accuracy variation across cities. We observe a negative correlation between map difficulty and accuracy. Moreover, model performance varies notably even among cities with comparable map difficulty levels. This disparity can be partially attributed to factors such as city prominence and the language used for stop names (see the ablation study on language in Appendix C.3), both of which are closely tied to the model’s pretrained knowledge.

Notably, we observe a counterintuitive phenomenon: among open-source models, reasoning models consistently underperform their base counterparts, whereas the opposite holds in the closed-source models1. Recent works argue that RL improves sample efficiency rather than foster-

1Although the comparison across closed-source models may not be fair due to lack of transparency in details, the reasoning variants exhibit consistently stronger performance in this category.

- Table 3. Evaluations of MLLMs on REASONMAP w/o visual inputs. S. denotes results for short questions and L. denotes results for long questions. The map score is capped at 20 for short questions, while for long questions, the maximum score is 40. Bold indicates the best results among open-source and closed-source models, respectively, while underline represents the second best. Green highlights improved results compared to the full input setting (Table 2), while red indicates performance drops.

Model Type Weighted Acc. (S.) #Tokens (S.) Weighted Acc. (L.) #Tokens (L.) Weighted Map Score (S. / L.) Open-source Models

Qwen2.5-VL-3B-Instruct [3] Base 9.38%↑0.7% 47 9.72%↑1.73% 147 2.93↑0.18 / 4.51↑0.81 Qwen2.5-VL-72B-Instruct [3] Base 16.41%↓10.24% 28 15.71%↓8.51% 108 4.03↓1.06 / 6.49↓2.31 Kimi-VL-A3B-Instruct [64] Base 11.81%↓0.95% 41 9.81%↓2.52% 49 3.37↑0.07 / 5.32↓0.05 Kimi-VL-A3B-Thinking [64] Reasoning 4.17%↓1.30% 1,039 2.08%↓3.39% 1,755 2.06↓0.38 / 1.64↓1.53

Closed-source Models

Doubao-115 [6] Base 13.72%↓20.48% 34 13.98%↓24.04% 99 3.50↓1.75 / 6.48↓5.48 Doubao-415 [6] Reasoning 21.53%↓21.61% 352 17.19%↓28.90% 1,047 4.85↓2.48 / 7.68↓6.99

ing new reasoning abilities [73, 89, 95], as it biases models toward high-reward responses, limits exploration, and, because most open-source reasoning models are fine-tuned mainly on math, physics, and logic tasks [49, 57], further reinforces domain-specific reasoning behaviors. However, these do not fully explain why closed-source reasoning models outperform their base variants. To further investigate, we conduct a case analysis. Among open-source models, we observe that reasoning variants often engage in repeated trial-and-verification during their thinking process. This iterative reasoning frequently introduces visual confusion; for instance, a model may initially identify the correct route but later override it with an incorrect one during self-reflection, failing to recover the earlier correct reasoning (see the case by Kimi-VL-A3B-Thinking). In contrast, closed-source models exhibit stronger visual grounding: even when visual confusion occurs, they tend to selfcorrect within the reasoning chain, ultimately producing the correct answer as illustrated in Figure A3 of the Appendix. Overall, this suggests that the better performance of closed-source reasoning models is primarily driven by their stronger visual grounding capability, which enables effective error correction throughout the reasoning process.

##### Wrong Case by Kimi-VL-A3B-Thinking (green denotes correct & red represents wrong)

< think > ... Is Augustins on Line 18? Yes, it is on Line 18. ... Wait, maybe I got the location wrong ... Let me confirm, steps should be like: Augustins (Line 14) - ... < /think > ... Route: Line 14 ...

##### 5.2.2. Performance of MLLMs without Visual Input

Recent studies suggest that multimodal models may rely on internal knowledge priors rather than truly attending to visual inputs [20, 24, 31, 92]. To examine this phenomenon, we further evaluate representative open- and closed-source models under the visual-masking setting. The results are reported in Table 3. We observe that while most models can

leverage prior knowledge to answer questions, their performance generally declines to varying degrees when visual input is removed, with the decline being more pronounced among closed-source models. Model performance is positively correlated with the performance drop after masking visual inputs, indicating effective use of visual information. In contrast, models like Qwen2.5-VL-3B-I show minimal or even improved performance, suggesting a reliance on prior knowledge rather than real visual reasoning. We further conduct non-vision experiments by replacing maps with their symbolic representations in Appendix C.4.

##### 5.2.3. Results of Training Baseline

As shown in Table 4, the training baseline with the GRPObased RL scheme consistently improves overall accuracy and map reasoning quality across model scales, while also reducing token usage. These results demonstrate that incorporating the reward signals can effectively enhance reasoning efficiency and improve answer quality.

#### 5.3. Error Analysis

Figure 3 presents representative failure cases from REASONMAP, revealing several recurring error types. A common issue is visual confusion, where the model misidentifies the transit line due to similar colors or adjacent layouts, for instance, mistaking Line 9 for Line 16 (OpenAI

- o3, left column; Doubao-428, right column). Another frequent problem is format errors, where responses deviate from the required structure, making them unprocessable despite containing correct route information (Doubao-115 and QvQ-72B-Preview, left column). We also observe instances
- of hallucination [4], where the model repeats the correct answer (Kimi-VL-A3B-Thinking, middle column) or generates information that is not present in the input, such as mentioning image generation, as seen in Skywork-R1V38B (right column). Refusal cases are also present, where models explicitly decline to answer (Skywork-R1V-38B, middle and right column). Notably, these errors may occasionally co-occur within a single response (Skywork-R1V-

[Figure 104]

[Figure 105]

- Table 4. Evaluations of RL-trained models on REASONMAP. S. represents results for short questions, while L. denotes results for long questions. The map score is capped at 20 for short questions, while for long questions, the maximum score is 40.

Model Type Weighted Acc. (S.) #Tokens (S.) Weighted Acc. (L.) #Tokens (L.) Weighted Map Score (S. / L.) Qwen2.5-VL-3B-Instruct [3] Base 8.68% 42 7.99% 151 2.75 / 3.70

+ Reinforcement Fine-tuning Base 11.46%↑2.78% 25 10.50%↑2.51% 93 3.81↑1.06 / 6.09↑2.39 Qwen2.5-VL-7B-Instruct [3] Base 13.28% 26 7.12% 57 4.01 / 5.74

[Figure 106]

+ Reinforcement Fine-tuning Base 26.22%↑12.94% 25 26.04%↑18.92% 34 5.52↑1.51 / 9.52↑3.78

Departure

Arrival

Departure

Departure

Arrival

Arrival

[Figure 107]

[Figure 108]

Q: 7th St/Metro Ctr to Little Tokyo / Arts Dist + Question Template

Q: Ponte Mammolo to Gardenie + Question Template

Q: 良乡南关站 to 丰台东⼤街站

+ Question Template

|A: Route Name: Line B Departure Stop: 7th St / Metro Ctr Station Arrival Stop: Little Tokyo / Arts District Station<br><br>Visual Confusion|
|---|

[Figure 109]

Visual Confusion

Refusal

A: <think> …丰台东⼤街, is on Line 16 … However, I mistakenly thought it might be on Line 14 at first. After checking, I confirm it is definitely on Line 16 … </think> … Route Name: 16号线 … Arrival Stop: 丰台东⼤街站

[Figure 110]

A: <think> … </think> I‘m sorry, I can’t execute that request. I‘m an AI assistant designed to provide textbased assistance, and I don’t have the capability to generate or display images ...

Hallucination

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Qwen 2.5-VL

OpenAI o3

Skywork R1V

72B-I

|A: <think> ... <\think> Route Name: Line A\nDeparture Stop: 7th St…District Station\n--\n Route Name: Line A\nDeparture Stop: 7th St…District Station<br><br>Hallucination|
|---|

38B

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

|A: Route Name: 房⼭线 -- 9号线 Departure Stop: 良乡南关 Arrival Stop: 丰台东⼤街<br><br>Format<br><br>Error|
|---|

Kimi-VL

Visual

A: <think> Looking at the map, San Giovanni is marked with M B and M C, so it is the transfer station for Line B and Line C. </think> Route Name: Line B … \n-\nRoute Name: Line C \n Departure Stop: San Giovanni Station \n Arrival Stop: Gardenie Station

A3B Thinking

Confusion

Doubao

[Figure 119]

-115

[Figure 120]

A: <think> …how to get from 7th St / Metro Ctr to Little Tokyo / Arts District Station using the subway map… <\think> I'm sorry, I can't help with that question. Maybe someone else can.

A: <think> … </think> Route Name: Fangshan Line to Line 9 Departure Stop: 良乡南关站 (Fangshan Line), Arrival Stop: 丰台东⼤街站 (Line 9), Transfer at: 郭公庄站

Format

Refusal

[Figure 121]

[Figure 122]

Doubao

[Figure 123]

[Figure 124]

Error

-428

Skywork R1V

QvQ

72B Preview

38B

- Figure 3. Error case analyses of various MLLMs using REASONMAP. For reasoning models, the reasoning process is explicitly marked with <think> and </think> tags. We highlight error contents in the answers with red and categorize them accordingly.

38B, right column). Furthermore, we conduct a systematic analysis of failure causes from a model capability perspective (e.g., optical character recognition (OCR), visual grounding, and spatial reasoning) in Appendix C.5. The above error types highlight the limitations in visual grounding and response robustness, especially when handling finegrained visual details (see more cases in Appendix D).

semi-automated and scalable data building pipeline, we curate a diverse set of human-verified question-answer pairs across 30 cities from 13 countries. Our two-level evaluation framework enables a nuanced assessment of both correctness and quality. Comprehensive experiments on 16 widely used MLLMs reveal key insights into model behavior, highlighting performance gaps between base and reasoning models, as well as the crucial role of visual input. To further strengthen the benchmark and facilitate future research, we provide a GRPO-based RL baseline. Error analyses also reveal recurring failure patterns, highlighting visual weaknesses of current MLLMs. Collectively, these findings underscore the need for more rigorous evaluation and training approaches to advance visual reasoning.

### 6. Conclusion

In this work, we introduce REASONMAP, a new benchmark specifically designed to evaluate the fine-grained visual understanding and spatial reasoning capabilities of MLLMs using high-resolution transit maps. Through a

### Acknowledgement

This paper is supported by Young Scientists Fund of the National Natural Science Foundation of China (NSFC) (No. 62506305), Zhejiang Leading Innovative and Entrepreneur Team Introduction Program (No. 2024R01007), Key Research and Development Program of Zhejiang Province (No. 2025C01026), Scientific Research Project of Westlake University (No. WU2025WF003), Chinese Association for Artificial Intelligence (CAAI) & Ant Group Research Fund - AGI Track (No. 2025CAAI-ANT-13). It is also supported by the research funds of the National Talent Program and Hangzhou Municipal Talent Program.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 2
- [2] Zhenxin Ai, Huilan Luo, and Jianqin Wang. A lightweight multistream framework for salient object detection in optical remote sensing. IEEE Transactions on Geoscience and Remote Sensing, 63:1–15, 2025. 21
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 3, 5, 6, 7, 8, 15, 16, 18, 19, 22
- [4] Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024. 5, 7
- [5] Zhibin Bao, Sabir Hossain, Haoxiang Lang, and Xianke Lin. A review of high-definition map creation methods for autonomous driving. Engineering Applications of Artificial Intelligence, 122:106125, 2023. 3
- [6] ByteDance. doubao-1.5-pro. https : / / seed . bytedance.com/en/special/doubao_1_5_ pro, 2025. 2, 3, 6, 7, 15, 16, 18, 19, 22
- [7] Xu Cao, Tong Zhou, Yunsheng Ma, Wenqian Ye, Can Cui, Kun Tang, Zhipeng Cao, Kaizhao Liang, Ziran Wang, James M Rehg, et al. Maplm: A real-world large-scale vision-language benchmark for map and traffic scene understanding. In CVPR, 2024. 3
- [8] Xinghao Chen, Anhao Zhao, Heming Xia, Xuan Lu, Hanlin Wang, Yanjun Chen, Wei Zhang, Jian Wang, Wenjie Li, and Xiaoyu Shen. Reasoning beyond language: A comprehensive survey on latent chain-of-thought reasoning. arXiv preprint arXiv:2505.16782, 2025. 21
- [9] Yi-Chia Chen, Wei-Hua Li, Cheng Sun, Yu-Chiang Frank Wang, and Chu-Song Chen. Sam4mllm: Enhance multimodal large language model for referring expression segmentation. In ECCV, 2024. 2
- [10] Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Yang Zhou, Kaizhao Liang, Jintai Chen, Juanwu Lu, Zichong Yang, Kuei-Da Liao, et al. A survey on multimodal large language models for autonomous driving. In WACV, 2024. 2

- [11] Mahir Labib Dihan, Md Tanvir Hassan, Md Tanvir Parvez, Md Hasebul Hasan, Md Almash Alam, Muhammad Aamir Cheema, Mohammed Eunus Ali, and Md Rizwan Parvez. Mapeval: A map-based evaluation of geo-spatial reasoning in foundation models. arXiv preprint arXiv:2501.00316,

2024. 2, 3

- [12] Jiafei Duan, Samson Yu, Hui Li Tan, Hongyuan Zhu, and Cheston Tan. A survey of embodied ai: From simulators to research tasks. TETCI, 6(2):230–244, 2022. 2
- [13] Jie Feng, Jun Zhang, Junbo Yan, Xin Zhang, Tianjian Ouyang, Tianhui Liu, Yuwei Du, Siqi Guo, and Yong Li. Citybench: Evaluating the capabilities of large language model as world model. arXiv preprint arXiv:2406.13945,

2024. 2, 3, 18

- [14] Sicheng Feng, Keda Tao, and Huan Wang. Is oracle pruning the true oracle? arXiv preprint arXiv:2412.00143, 2024. 21
- [15] Sicheng Feng, Gongfan Fang, Xinyin Ma, and Xinchao Wang. Efficient reasoning models: A survey. TMLR, 2025. 2
- [16] Sicheng Feng, Kaiwen Tuo, Song Wang, Lingdong Kong, Jianke Zhu, and Huan Wang. Rewardmap: Tackling sparse rewards in fine-grained visual reasoning via multi-stage reinforcement learning. arXiv preprint arXiv:2510.02240,

2025. 2

- [17] Sicheng Feng, Zigeng Chen, Xinyin Ma, Gongfan Fang, and Xinchao Wang. dvoting: Fast voting for dllms. arXiv preprint arXiv:2602.12153, 2026. 20
- [18] Junqi Ge, Ziyi Chen, Jintao Lin, Jinguo Zhu, Xihui Liu, Jifeng Dai, and Xizhou Zhu. V2pe: Improving multimodal long-context capability of vision-language models with variable visual position encoding. arXiv preprint arXiv:2412.09616, 2024. 15
- [19] Gemini, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 3, 6, 15, 16, 18, 22
- [20] Aarti Ghatkesar, Uddeshya Upadhyay, and Ganesh Venkatesh. Looking beyond language priors: Enhancing visual comprehension and attention in multimodal models. arXiv preprint arXiv:2505.05626, 2025. 7
- [21] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025. 3, 19
- [22] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2
- [23] Meng-Hao Guo, Jiajun Xu, Yi Zhang, Jiaxi Song, Haoyang Peng, Yi-Xuan Deng, Xinzhi Dong, Kiyohiro Nakayama, Zhengyang Geng, Chen Wang, et al. Rbench: Graduate-level multi-disciplinary benchmarks for llm & mllm complex reasoning evaluation. arXiv preprint arXiv:2505.02018, 2025. 3
- [24] Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms

- reason in multimodality? emma: An enhanced multimodal reasoning benchmark. arXiv preprint arXiv:2501.05444, 2025. 7
- [25] James Hays and Alexei A Efros. Im2gps: estimating geographic information from a single image. In CVPR, 2008. 20
- [26] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In ICLR,

2021. 2

- [27] Byeongho Heo, Song Park, Dongyoon Han, and Sangdoo Yun. Rotary position embedding for vision transformer. In ECCV, 2024. 15
- [28] Yining Hong, Rui Sun, Bingxuan Li, Xingcheng Yao, Maxine Wu, Alexander Chien, Da Yin, Ying Nian Wu, Zhecan James Wang, and Kai-Wei Chang. Embodied web agents: Bridging physical-digital realms for integrated agent intelligence. arXiv preprint arXiv:2506.15677, 2025. 20
- [29] Yangliu Hu, Zikai Song, Na Feng, Yawei Luo, Junqing Yu, Yi-Ping Phoebe Chen, and Wei Yang. Sf2t: Self-supervised fragment finetuning of video-llms for fine-grained understanding. arXiv preprint arXiv:2504.07745, 2025. 2
- [30] Jingyuan Huang, Jen-tse Huang, Ziyi Liu, Xiaoyuan Liu, Wenxuan Wang, and Jieyu Zhao. Vlms as geoguessr masters: Exceptional performance, hidden biases, and privacy risks. arXiv preprint arXiv:2502.11163, 2025. 20
- [31] Botian Jiang, Lei Li, Xiaonan Li, Zhaowei Li, Xiachong Feng, Lingpeng Kong, Qi Liu, and Xipeng Qiu. Understanding the role of llms in multimodal evaluation benchmarks. arXiv preprint arXiv:2410.12329, 2024. 7
- [32] Xin Jin, Siyuan Li, Siyong Jian, Kai Yu, and Huan Wang. Mergemix: A unified augmentation paradigm for visual and multi-modal understanding. arXiv preprint arXiv:2510.23479, 2025. 21
- [33] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361,

2020. 6

- [34] Lingdong Kong, Wesley Yang, Jianbiao Mei, Youquan Liu, Ao Liang, Dekai Zhu, Dongyue Lu, Wei Yin, Xiaotao Hu, Mingkai Jia, et al. 3d and 4d world modeling: A survey. arXiv preprint arXiv:2509.07996, 2025. 21
- [35] EvolvingLMMs Lab. Open R1 Multimodal. https: //github.com/EvolvingLMMs-Lab/open-r1multimodal, 2025. 2
- [36] Alexandre Lacoste, Nils Lehmann, Pau Rodriguez, Evan Sherwin, Hannah Kerner, Bj¨orn L¨utjens, Jeremy Irvin, David Dao, Hamed Alemohammad, Alexandre Drouin, et al. Geo-bench: Toward foundation models for earth monitoring. 2023. 3
- [37] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In CVPR, 2024. 2
- [38] Qinqian Lei, Bo Wang, and Robby T. Tan. Ez-hoi: Vlm adaptation via guided prompt learning for zero-shot hoi detection. In NeurIPS, 2024. 21

- [39] Qinqian Lei, Bo Wang, and Robby T. Tan. Hola: Zeroshot hoi detection with low-rank decomposed vlm feature adaptation. In ICCV, 2025. 21
- [40] Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vuli´c, and Furu Wei. Imagine while reasoning in space: Multimodal visualization-ofthought. arXiv preprint arXiv:2501.07542, 2025. 2
- [41] Qi Li and Xinchao Wang. Sponge tool attack: Stealthy denial-of-efficiency against tool-augmented agentic reasoning. arXiv preprint arXiv:2601.17566, 2026. 21
- [42] Wentong Li, Yuqian Yuan, Jian Liu, Dongqi Tang, Song Wang, Jie Qin, Jianke Zhu, and Lei Zhang. Tokenpacker: Efficient visual projector for multimodal llm. IJCV, pages 1–19, 2025. 2
- [43] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024. 18
- [44] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visualrft: Visual reinforcement fine-tuning. In CVPR, 2025. 2
- [45] Utkarsh Mall, Kevin Matzen, Bharath Hariharan, Noah Snavely, and Kavita Bala. Geostyle: Discovering fashion trends and events. In ICCV, 2019. 20
- [46] OpenAI. Hello gpt4-o. https://openai.com/ index/hello-gpt-4o/, 2024. 3, 6, 18, 19, 22
- [47] OpenAI. OpenAI o1. https://openai.com/o1/,

2024. 2

- [48] OpenAI. OpenAI o3 and o4-mini System Card. https: / / cdn . openai . com / pdf / 2221c875 - 02dc 4789-800b-e7758f3722c1/o3-and-o4-minisystem-card.pdf, 2025. 3, 6, 18, 22
- [49] Yi Peng, Xiaokun Wang, Yichen Wei, Jiangbo Pei, Weijie Qiu, Ai Jian, Yunzhuo Hao, Jiachun Pan, Tianyidan Xie, Li Ge, et al. Skywork r1v: pioneering multimodal reasoning with chain-of-thought. arXiv preprint arXiv:2504.05599,

2025. 2, 3, 6, 7, 15, 16, 21

- [50] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 2
- [51] Qwen Team. Qvq: To see the world with wisdom. https://qwenlm.github.io/blog/qvq-72bpreview/, 2024. 2, 3, 6, 15, 16, 18, 22
- [52] Yufan Ren, Konstantinos Tertikas, Shalini Maiti, Junlin Han, Tong Zhang, Sabine S¨usstrunk, and Filippos Kokkinos. Vgrp-bench: Visual grid reasoning puzzle benchmark for large vision-language models. arXiv preprint arXiv:2503.23064, 2025. 3
- [53] Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel reasoning with large multimodal model. In CVPR, 2024. 2
- [54] Ari Seff and Jianxiong Xiao. Learning from maps: Visual common sense for autonomous driving. arXiv preprint arXiv:1611.08583, 2016. 3

- [55] Kele Shao, Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Holitom: Holistic token merging for fast video large language models. arXiv preprint arXiv:2505.21334, 2025. 21
- [56] Kele Shao, Keda Tao, Kejia Zhang, Sicheng Feng, Mu Cai, Yuzhang Shang, Haoxuan You, Can Qin, Yang Sui, and Huan Wang. When tokens talk too much: A survey of multimodal long-context token compression across images, videos, and audios. arXiv preprint arXiv:2507.20198,

2025. 21

- [57] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2, 5, 7, 16
- [58] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025. 2
- [59] Fatemeh Shiri, Xiao-Yu Guo, Mona Golestan Far, Xin Yu, Gholamreza Haffari, and Yuan-Fang Li. An empirical analysis on spatial reasoning capabilities of large multimodal models. arXiv preprint arXiv:2411.06048, 2024. 2
- [60] Yueqi Song, Tianyue Ou, Yibo Kong, Zecheng Li, Graham Neubig, and Xiang Yue. Visualpuzzles: Decoupling multimodal reasoning evaluation from domain knowledge. arXiv preprint arXiv:2504.10342, 2025. 2, 3
- [61] Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning. In NeurIPS, 2025. 2
- [62] Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Dycoke: Dynamic compression of tokens for fast video large language models. In CVPR, 2025. 21
- [63] Keda Tao, Kele Shao, Bohan Yu, Weiqiang Wang, Huan Wang, et al. Omnizip: Audio-guided dynamic token compression for fast omnimodal large language models. arXiv preprint arXiv:2511.14582, 2025. 21
- [64] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025. 2, 3, 6, 7, 15, 16, 18, 19, 21
- [65] R1-V Team. R1-V. https://github.com/DeepAgent/R1-V?tab=readme-ov-file, 2025. 2
- [66] Guangnian Wan, Xinyin Ma, Gongfan Fang, and Xinchao Wang. Invisible safety threat: Malicious finetuning for llm via steganography. In ICLR, 2026. 21
- [67] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with mathvision dataset. In NeurIPS, 2024. 2, 3
- [68] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6): 186345, 2024. 2

- [69] Song Wang, Wentong Li, Wenyu Liu, Xiaolu Liu, and Jianke Zhu. Lidar2map: In defense of lidar-based semantic map construction using online camera distillation. In CVPR, 2023. 3
- [70] Song Wang, Gongfan Fang, Lingdong Kong, Xiangtai Li, Jianyun Xu, Sheng Yang, Qiang Li, Jianke Zhu, and Xinchao Wang. Pixelthink: Towards efficient chain-of-pixel reasoning. arXiv preprint arXiv:2505.23727, 2025. 2
- [71] Song Wang, Lingdong Kong, Xiaolu Liu, Hao Shi, Wentong Li, Jianke Zhu, and Steven CH Hoi. Forging spatial intelligence: A roadmap of multi-modal data pre-training for autonomous systems. arXiv preprint arXiv:2512.24385,

2025. 3

- [72] Song Wang, Xiaolu Liu, Lingdong Kong, Jianyun Xu, Chunyong Hu, Gongfan Fang, Wentong Li, Jianke Zhu, and Xinchao Wang. Pointlora: Low-rank adaptation with token selection for point cloud learning. In CVPR, 2025. 21
- [73] Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Lucas Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, et al. Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571, 2025. 7
- [74] Yichen Wei, Yi Peng, Xiaokun Wang, Weijie Qiu, Wei Shen, Tianyidan Xie, Jiangbo Pei, Jianhao Zhang, Yunzhuo Hao, Xuchen Song, et al. Skywork r1v2: Multimodal hybrid reinforcement learning for reasoning. arXiv preprint arXiv:2504.16656, 2025. 2, 3, 6, 15, 16, 18, 21
- [75] Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. In CVPR, 2024. 2, 3
- [76] Shaoyuan Xie, Lingdong Kong, Yuhao Dong, Chonghao Sima, Wenwei Zhang, Qi Alfred Chen, Ziwei Liu, and Liang Pan. Are vlms ready for autonomous driving? an empirical study from the reliability, data and metric perspectives. In ICCV, 2025. 2, 3
- [77] Shuo Xing, Zezhou Sun, Shuangyu Xie, Kaiyuan Chen, Yanjia Huang, Yuping Wang, Jiachen Li, Dezhen Song, and Zhengzhong Tu. Can large vision language models read maps like a human? arXiv preprint arXiv:2503.14607,

- 2025. 2, 3, 18

[78] Haotian Xu, Yue Hu, Chen Gao, Zhengqiu Zhu, Yong Zhao, Yong Li, and Quanjun Yin. Geonav: Empowering mllms with explicit geospatial reasoning abilities for languagegoal aerial navigation. arXiv preprint arXiv:2504.09587,

- 2025. 3

- [79] Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025. 2, 3
- [80] Yunqiu Xu, Linchao Zhu, and Yi Yang. Mc-bench: A benchmark for multi-context visual grounding in the era of mllms. In ICCV, 2025. 2
- [81] Yibo Yan and Joey Lee. Georeasoner: Reasoning on geospatially grounded context for natural language understanding. In CIKM, 2024. 3
- [82] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng

- Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 3
- [83] Jingru Yang, Huan Yu, Yang Jingxin, Chentianye Xu, Yin Biao, Yu Sun, and Shengfeng He. Visual-linguistic agent: Towards collaborative contextual object reasoning. arXiv preprint arXiv:2411.10252, 2024. 2
- [84] Zhen Yang, Jinhao Chen, Zhengxiao Du, Wenmeng Yu, Weihan Wang, Wenyi Hong, Zhihuan Jiang, Bin Xu, and Jie Tang. Mathglm-vision: Solving mathematical problems with multi-modal large language model. arXiv preprint arXiv:2409.13729, 2024. 2
- [85] Michal Yarom, Yonatan Bitton, Soravit Changpinyo, Roee Aharoni, Jonathan Herzig, Oran Lang, Eran Ofek, and Idan Szpektor. What you see is what you read? improving textimage alignment evaluation. In NeurIPS, 2023. 2
- [86] Sahiti Yerramilli, Nilay Pande, Rynaa Grover, and Jayant Sravan Tamarapalli. Geochain: Multimodal chainof-thought for geographic reasoning. arXiv preprint arXiv:2506.00785, 2025. 3
- [87] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024. 2, 3
- [88] Xinli Yue, JianHui Sun, Junda Lu, Liangchao Yao, Fan Xia, Tianyi Wang, Fengyun Rao, Jing Lyu, and Yuetang Deng. Instruction-augmented multimodal alignment for image-text and element matching. arXiv preprint arXiv:2504.12018, 2025. 2
- [89] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837,

2025. 7

- [90] Qingbin Zeng, Qinglong Yang, Shunan Dong, Heming Du, Liang Zheng, Fengli Xu, and Yong Li. Perceive, reflect, and plan: Designing llm agent for goal-directed city navigation without instructions. arXiv preprint arXiv:2408.04168,

2024. 3

- [91] Kejia Zhang, Keda Tao, Jiasheng Tang, and Huan Wang. Poison as cure: Visual noise for mitigating object hallucinations in lvms. In NeurIPS, 2025. 21
- [92] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In ECCV, 2024. 2, 3, 7
- [93] Shihua Zhang, Zizhuo Li, Yuan Gao, and Jiayi Ma. Dematch: Deep decomposition of motion field for two-view correspondence learning. In CVPR, 2024. 21
- [94] Shihua Zhang, Zizhuo Li, and Jiayi Ma. Dematch++: Twoview correspondence learning via deep motion field decomposition and respective local-context aggregation. IEEE Transactions on Pattern Analysis and Machine Intelligence,

2025. 21

- [95] Sheng Zhang, Qianchu Liu, Guanghui Qin, Tristan Naumann, and Hoifung Poon. Med-rlvr: Emerging medical

- reasoning from a 3b base model via reinforcement learning. arXiv preprint arXiv:2502.19655, 2025. 7
- [96] Tao Zhang, Xiangtai Li, Hao Fei, Haobo Yuan, Shengqiong Wu, Shunping Ji, Chen Change Loy, and Shuicheng Yan. Omg-llava: Bridging image-level, object-level, pixel-level reasoning and understanding. In NeurIPS, 2024. 2
- [97] Xinglang Zhang, Yunyao Zhang, ZeLiang Chen, Junqing Yu, Wei Yang, and Zikai Song. Logical phase transitions: Understanding collapse in llm logical reasoning. arXiv preprint arXiv:2601.02902, 2026. 21
- [98] Yunyao Zhang, Xinglang Zhang, Junxi Sheng, Wenbing Li, Junqing Yu, Wei Yang, and Zikai Song. From ambiguity to verdict: A semiotic-grounded multi-perspective agent for llm logical reasoning. arXiv preprint arXiv:2509.24765,

2025. 21

- [99] Shitian Zhao, Haoquan Zhang, Shaoheng Lin, Ming Li, Qilong Wu, Kaipeng Zhang, and Chen Wei. Pyvision: Agentic vision with dynamic tooling. arXiv preprint arXiv:2507.07998, 2025. 20
- [100] Yupeng Zheng, Zebin Xing, Qichao Zhang, Bu Jin, Pengfei Li, Yuhang Zheng, Zhongpu Xia, Kun Zhan, Xianpeng Lang, Yaran Chen, et al. Planagent: A multi-modal large language agent for closed-loop vehicle motion planning. arXiv preprint arXiv:2406.01587, 2024. 3
- [101] Junhan Zhu, Hesong Wang, Mingluo Su, Zefang Wang, and Huan Wang. Obs-diff: Accurate pruning for diffusion models in one-shot. arXiv preprint arXiv:2510.06751, 2025. 21
- [102] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 2, 3, 6, 15, 16, 18, 19, 22

## REASONMAP: Towards Fine-Grained Visual Reasoning from Transit Maps Supplementary Material

### Appendix

We provide a comprehensive overview in the Appendix, covering key details of our dataset, methodology, evaluation, training baseline, analysis, and further discussions. Specifically, we include the question templates, quality control details, a fine-grained taxonomy of difficulty, and sources of transit maps from 30 cities for REASONMAP construction in Appendix A. We then report detailed descriptions of the evaluation algorithm, experimental setup, and GRPO training in Appendix B. In Appendix C, we include supplementary results and conduct more experiments, including supplementary results, evaluation of symbolic representation and an ablation study about languages. We also provide the results of fine-grained error analysis metrics and systematically analyze failure causes. In Appendix D, we further extend case analysis by providing more classical cases. In addition, we further discuss the stated limitations, future directions, and potential broader impacts of our work in Appendix E. We finally present public implementation for the MLLMs used in our experiments, LLM usage statement, and ethical statement (see Appendix F).

- A. Dataset Construction Details 13

- A.1. Question Template Summary . . . . . . . . 13
- A.2. A More Fine-grained Taxonomy of Difficulty 14
- A.3. Quality Control Details . . . . . . . . . . . . 14
- A.4. Map Source . . . . . . . . . . . . . . . . . . 14

- B. Details of Evaluation and Training Baseline 14

- B.1. Correctness and Quality Evaluation . . . . . 14
- B.2. High-Resolution Image Preprocessing. . . . 14
- B.3. Details about Difficulty-Aware Weighting. . 16
- B.4. Details of GRPO RL Training . . . . . . . . 16

- C. Supplementary Experiments 16

- C.1. Supplementary Results . . . . . . . . . . . . 16
- C.2. Fine-grained Error Analysis Metric Summary 16
- C.3. Further Experiments about Languages . . . . 16
- C.4. Further Experiments about Symbolic Representation of Maps . . . . . . . . . . . . . . 17
- C.5. Further Systematic Analysis on Failure Causes 18

- D. Case Analysis 19
- E. Further Discussions 20

- E.1. Limitations and Future Work . . . . . . . . . 20
- E.2. Broader Impact . . . . . . . . . . . . . . . . 21

##### F. Further Statement 21

- F.1. Public Implementation . . . . . . . . . . . . 21
- F.2. Large Language Model Usage Statement . . 22
- F.3. Ethics Statement . . . . . . . . . . . . . . . 22

### A. Dataset Construction Details

#### A.1. Question Template Summary

We present one short question template and two long question templates as follows.

##### Short Question Template

According to the subway map, how do I get from [Stop 1] to [Stop 2]? Provide only one optimal route, with only the line name and the departure and arrival stations. The format should be strictly followed:

Route Name: Line x Departure Stop: xx Station Arrival Stop: xx Station

-Route Name: Line x Departure Stop: xx Station Arrival Stop: xx Station

##### Long Question Template 1

According to the subway map, how do I get from [Stop 1] to [Stop 2]? Provide only one optimal route, and include the number of via stops for each route section (excluding the departure and arrival stops). The format should be strictly followed:

Route Name: Line x Departure Stop: xx Station Arrival Stop: xx Station Number of Via Stops: x

-Route Name: Line x Departure Stop: xx Station Arrival Stop: xx Station Number of Via Stops: x

Long Question Template 2 According to the subway map, how do I get from [Stop 1] to [Stop 2]? Provide only one optimal route, including all the via stops. The format should be strictly followed: Route Name: Line x Departure Stop: xx Station Arrival Stop: xx Station Via Stops: xx Station, xx Station

-Route Name: Line x Departure Stop: xx Station Arrival Stop: xx Station Via Stops: xx Station

#### A.2. A More Fine-grained Taxonomy of Difficulty

Beyond the easy, middle, and hard categorization for map and question difficulty, we provide three additional difficulty aware labels: 1) city line count, the total number of lines in a city (i.e., a proxy for map difficulty); 2) city transfer count, the total number of transfer stations in a city (i.e., a proxy for map difficulty); and 3) question transfer count, the number of transfers in the queried route (i.e., a proxy for question difficulty). These labels enable fine-grained category design and filtering in subsequent analyses.

#### A.3. Quality Control Details

Our quality control combines automated checks with manual refinement. Specifically, we first validate route correctness (e.g., start stop, arrival stop, and connectivity), followed by manual checks to ensure visual consistency (i.e., routes can be inferred from maps). Questions or GTs with issues are corrected or removed. Three domain experts reviewed the data and identified an error rate of ∼16%, after which all questions/GTs were corrected and verified to be accurate. Finally, we systematically adjust the difficulty distributions to prevent bias and ensure a balanced evaluation benchmark.

#### A.4. Map Source

We provide the sources of all maps included in REASONMAP for further reference (Table A1).

### B. Details of Evaluation and Training Baseline

#### B.1. Correctness and Quality Evaluation

We present the detailed algorithms for evaluating answer correctness and quality (Algorithm 1 for correctness evaluation and Algorithm 2 for quality evaluation).

- Table A1. Source links to the city transit maps used in the REASONMAP dataset. We present a total of 30 cities sourced from 13 countries.

City Source City Source City Source Budapest [Link] Oslo [Link] Rome [Link] Lisboa [Link] Geneva [Link] Dubai [Link] Auckland [Link] Sydney [Link] Singapore [Link] Kuala Lumpur [Link] Los Angeles [Link] Miami [Link] New York [Link] Toronto [Link] Washington [Link] Guiyang [Link] Shanghai [Link] Huhehaote (Hohhot) [Link] Nanchang [Link] Nanning [Link] Shenzhen [Link] Hangzhou [Link] Dalian [Link] Kunming [Link] Hefei [Link] Beijing [Link] Changzhou [Link] Jinan [Link] Xi’an [Link] Changshang [Link]

For matching (e.g., =) in the algorithms, we apply rulebased corrections on top of string matching to account for semantically irrelevant formatting variations (e.g., Line 1” = Route 1” = “1”), preventing evaluation failures caused solely by stylistic or linguistic differences. These corrections are deliberately limited to remain consistent with the format requirements of each question.

Additionally, for multilingual maps, the pipeline is identical to that of English maps, except that route and station names are retained in their local language. During evaluation, we accept both the local language and its English translation as correct, prioritizing semantic correctness.

Algorithm 1: Correctness Evaluation Initialize acc ← 1; if departure stop of first segment ̸= stop1 or arrival

stop of last segment ̸= stop2 then

acc ← 0;

foreach segment in predicted route do if route name not in the Metro Data then

acc ← 0;

if departure or arrival stop not in the stop list of

the route then

acc ← 0;

if not the last segment then

if arrival stop of current segment ̸=

departure stop of next segment then

acc ← 0; return acc

B.2. High-Resolution Image Preprocessing.

We compare how different Multimodal Large Language Models (MLLMs) handle high-resolution image inputs in

- Table A2. Specifically, we examine three key components in their preprocessing pipelines: dynamic resolution han-

Algorithm 2: Quality Evaluation Initialize map score ← 0; if departure stop of first segment = stop1 and arrival stop of last segment = stop2 then

map score ← map score + 1;

/* Long-question-specific part */ Initialize Vunion, Vintersection ← ∅; Initialize via stop score, num via stop score ← 0;

foreach segment pair (answer route, reference route) do if answer route name = reference route name then

map score ← map score + 2;

if answer departure stop = reference departure stop then

map score ← map score + 1;

if answer arrival stop = reference arrival stop then

map score ← map score + 1;

/* Long-question-specific part */ Calculate absolute difference (error) in the number of via stops; num via stop score ← num via stop score +

max(0, 4 − error/max(number of answer via stops, number of reference via stops) × 4); if answer route name = reference route name then

Update Vunion, Vintersection with answer and reference via stops respectively; via stop score ← via stop score + number of correctly matched via stops;

/* Long-question-specific part */ via stop score ← min(10,via stop score); num via stop score ← min(10,num via stop score); via stop score ← average( |Vintersection|/|Vunion| × 10, via stop score)

map score ← map score + Option(via stop score or num via stop score);

/* 10 for short question; 20 for long question */ map score ← min(10,map score)/min(20,map score) ; if correctness evaluation (acc) = 1 then

map score ← map score + 10/map score + 20; return map score;

dling, positional encoding, and token compression.

- 1. Dynamic resolution handling refers to whether the model can directly accept images of arbitrary sizes without resizing or cropping. Most recent models support native resolution processing, enabling them to preserve fine-grained spatial information. In contrast, models like Gemini [19] rely on image tiling and resizing to fit fixed input constraints.
- 2. Positional encoding helps the model retain spatial structure among visual tokens. Common strategies include 2D Rotary Positional Encoding (2D-RoPE) [27], as seen in Qwen2.5-VL [3] and Doubao [6], or flexible alternatives like V2PE [18] in InternVL3 [102]. Some mod-

els (e.g., Gemini, Skywork-R1V [49, 74]) do not explicitly disclose their positional encoding scheme, which we mark as “–” in the table.

3. Token compression aims to reduce the number of visual tokens for more efficient processing. Different models adopt different strategies: Qwen2.5-VL and QVQ [51] compress tokens via 2 × 2 patch concatenation followed by an MLP; InternVL3 [102] and Kimi-VL [64] utilize spatial transformations like pixel unshuffle or shuffle, also followed by MLPs; Doubao averages over 2 × 2 patches before projection. Models without token compression may incur higher memory and computation costs when processing high-resolution inputs.

- Table A2. Comparison of high-resolution image preprocessing strategies across different MLLMs. We use “−” to denote unspecified or unclear content.

Model Dynamic Resolution Handling Positional Encoding Token Compression

Qwen2.5-VL series [3] ✓ 2D-RoPE ✓ (2 × 2 Concat + MLP) QVQ-72B-Preview [51] ✓ 2D-RoPE ✓ (2 × 2 Concat + MLP) InternVL3 series [102] ✓ V2PE ✓ (Unshuffle + MLP) Kimi-VL series [64] ✓ 2D-RoPE ✓ (Shuffle + MLP) Skywork-R1V-38B [49, 74] ✓ - ✗ Gemini [19] ✗ (Tiling+Resize) - ✗ Doubao-1.5-Pro series [6] ✓ 2D-RoPE ✓ (2 × 2 Pooling + MLP)

#### B.3. Details about Difficulty-Aware Weighting.

Each difficulty pair is assigned a predefined weight that reflects its relative challenge level. The full weight matrix is shown below, where the first element in each pair denotes the question difficulty and the second denotes the map difficulty:

Difficulty Pair Weight (“easy”, “easy”) 1.0

(“medium”, “easy”) 1.5

(“hard”, “easy”) 2.0 (“easy”, “medium”) 1.5

(“medium”, “medium”) 2.0 (“hard”, “medium”) 2.5

(“easy”, “hard”) 2.0 (“medium”, “hard”) 2.5

(“hard”, “hard”) 3.0

This weighting scheme rewards models more for correctly solving harder question–map combinations, reflecting the increased reasoning complexity they entail, while maintaining moderate differences between buckets to prevent excessive score variance and preserve evaluation stability.

#### B.4. Details of GRPO RL Training

GRPO [57] extends standard policy gradient methods by normalizing rewards within a sampled group, which stabilizes optimization and encourages relative preference learning. Specifically, given an input x and a group of K sampled outputs G = {yi}Ki=1 with their corresponding scalar rewards {ri}Ki=1, the centered group advantage Aˆi is computed as the deviation of each sample’s reward from the group mean:

K

1 K

Aˆi = ri −

rj. (1)

j=1

The policy parameters θ are then updated to maximize the following objective:

L(θ) =

max

θ

K

Aˆi log πθ(yi | x), (2)

i=1

where πθ(yi | x) denotes the model likelihood of generating yi under parameters θ. This objective encourages the model

to increase the probability of outputs with above-average rewards while suppressing those with below-average ones. In our implementation, the reward ri is composed of an accuracy component and a format component.

### C. Supplementary Experiments

#### C.1. Supplementary Results

We provide additional visualization results in Figure A1 and A2 to better illustrate our evaluation on REASONMAP as follows. Figure A1 and A2 illustrate the model’s accuracy across different difficulty levels and cities, respectively. As shown, accuracy decreases with increasing difficulty and varies considerably across cities.

#### C.2. Fine-grained Error Analysis Metric Summary

We report multiple fine-grained error analysis metrics in Table A3 as follows: (1) dep − arr score: +1 if both the start and end stations are correct; (2) route name score: +2 for each correctly identified line name along the route; (3) stops score: +1 for each correctly identified intermediate stop; (4) num via stop score (only for long questions): computed by taking the absolute difference between the number of via stops in the answer and the reference route, and mapping it to a score from 0 to 4; (5) via stop score (only for long questions): calculated by averaging the number of correctly matched via stops (up to 10) and the Intersection-over-Union (IoU) between the via stop sets of the answer and reference route (scaled to 10).

We further provide a sensitivity analysis with two additional weighting schemes for the components of the map score (C1 & C2 in Table A4). Based on the results in Tab. A3 & A4, performance ranking remains consistent across different weighting schemes, while all components of the map score exhibit similar increasing or decreasing trends.

#### C.3. Further Experiments about Languages

We conduct an ablation study under the textualized representation paradigm (as mentioned in Appendix C.4). In this setting, visual images are not involved, which allows us to safely replace all non-English station names with unique English aliases without introducing visual inconsistencies. This approach isolates the language prior factor and avoids

[Figure 125]

[Figure 126]

(a) Accuracy for short questions (b) Accuracy for long questions

- Figure A1. Accuracy across difficulty combinations for four representative MLLMs (Qwen2.5-VL-72B-I, InternVL3-78B, OpenAI o3, and Doubao-415). Each difficulty combination is denoted by a pair (e.g., easy-hard), where the first term indicates question difficulty and the second term represents map difficulty. The pair (hard-middle) contains only one sample, leading to an accuracy of 100%. We summarize the number of evaluation samples in each difficulty bucket: 55 samples for easy-easy, 46 for easy-middle, 28 for middle-easy, 7 for hard-easy, 23 for middle-middle, 80 for easy-hard, 1 for hard-middle, 57 for middle-hard, and 15 for hard-hard.

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

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

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

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

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

(a) Accuracy for short questions (b) Accuracy for long questions

- Figure A2. Accuracy across different cities for four representative MLLMs (Qwen2.5-VL-72B-I, InternVL3-78B, OpenAI o3, and Doubao-415). Each city is marked with the corresponding map difficulty and the country flag. Each city in the test set provides a specific number of samples per model: 32 samples for Auckland, 34 for Los Angeles, 7 for Miami, 35 for Lisboa, 18 for Geneva, 40 for Beijing, 39 for Hangzhou, 17 for Budapest, 39 for Singapore, 40 for Rome, and 11 for Toronto.

any potential confounding effects from visual modifications. Concretely, we manually replace all Chinese station names in Beijing and Hangzhou with unique English station names (e.g., mapping them to New York stops: ‘zhichunli’ <-> 86 St), preserving the original transit map structure. The results under this setting are as follows.

Overall, we observe from the results in Table A5 that using English labels leads to performance improvements, particularly for long-form questions. This suggests that the model indeed exhibits a language bias, with English showing an advantage over Chinese, which may be attributed to differences in pre-training data distributions.

#### C.4. Further Experiments about Symbolic Representation of Maps

We conduct further experiments about deterministic baselines derived from symbolic representations of the maps. This setting can serve as a theoretical performance ceiling, independent of perceptual challenges faced by MLLMs. We replace the visual input with symbolic representations extracted from the underlying map structure. Specifically, we convert all routes and station information into textual form to represent the topological structure of the map. This textualized representation is then used for evaluation. Specifically, we provide the model with textualized representations and the question as input, without including any visual maps.

- Table A3. Fine-grained error analysis metrics of various MLLMs. S. represents results for short questions, while L. denotes results for long questions. Bold indicates the best results among open-source and closed-source models, respectively.

Model Type Dep-Arr Score (S. / L.)Route Name Score (S. / L.)Stops Score (S. / L.)Num. Via Stop Score (L.)Via Stop Score (L.) Open-source Models

Qwen2.5-VL-3B-Instruct [3] Base 0.86 / 0.78 0.03 / 0.02 1.03 / 0.96 0.42 0.00 Qwen2.5-VL-32B-Instruct [3] Base 0.95 / 0.92 0.09 / 0.10 1.16 / 1.19 1.57 0.01 Qwen2.5-VL-72B-Instruct [3] Base 0.96 / 0.95 0.22 / 0.24 1.23 / 1.22 1.56 0.04 InternVL3-38B [102] Base 0.87 / 0.84 0.06 / 0.10 1.08 / 1.12 1.63 0.00 InternVL3-78B [102] Base 0.96 / 0.89 0.15 / 0.17 1.15 / 1.12 1.46 0.01 Kimi-VL-A3B-Instruct [64] Base 0.89 / 0.88 0.07 / 0.07 1.06 / 1.11 0.91 0.02 Kimi-VL-A3B-Thinking [64] Reasoning 0.80 / 0.65 0.08 / 0.10 0.99 / 0.79 0.50 0.00 Skywork-R1V-38B [74] Reasoning 0.60 / 0.62 0.06 / 0.09 0.74 / 0.71 1.00 0.00 QvQ-72B-Preview [51] Reasoning 0.35 / 0.22 0.03 / 0.02 0.42 / 0.29 0.20 0.01

Closed-source Models

Doubao-115 [6] Base 0.78 / 0.96 0.08 / 0.18 1.08 / 1.31 1.94 0.06 OpenAI 4o [46] Base 0.97 / 0.95 0.22 / 0.29 1.49 / 1.53 2.22 0.04 Doubao-415 [6] Reasoning 0.98 / 0.98 0.33 / 0.30 1.57 / 1.65 2.37 0.08 Doubao-428 [6] Reasoning 0.73 / 0.75 0.00 / 0.03 1.19 / 1.27 2.27 0.00 Gemini-2.5-Flash [19] Reasoning 0.93 / 0.67 0.27 / 0.29 1.67 / 1.22 1.82 0.05 OpenAI o3 [48] Reasoning 0.99 / 0.91 0.32 / 0.16 1.77 / 1.73 3.31 0.03

- Table A4. Ablation on the map score. MS denotes the map score.

reasoning (e.g., OCR, grounding, and spatial reasoning). To assess OCR capabilities, we collect metrics of 9 representative MLLMs on OCRBench [43]. Comparing these with their performance on REASONMAP as shown in Table A7 in the paper, we observe no clear correlation between OCR ability and REASONMAP accuracy. Notably, this trend holds across both open-source and closed-source models, suggesting that stronger OCR performance alone does not lead to better fine-grained visual reasoning. For instance, among open-source models, InternVL3-78B achieves the highest OCRBench scores, but underperforms Qwen2.5VL-72B-Instruct on REASONMAP.

- C1 uses the avg scheme, while C2 excludes the Dep-Arr score part.

Model Weighted MS (S./L.) C1 MS (S./L.) C2 MS (S./L.) Qwen2.5-VL-32B-I 3.88 / 6.84 3.85 / 5.80 2.90 / 4.88 Qwen2.5-VL-72B-I 5.09 / 8.80 5.08 / 8.83 4.12 / 7.88 OpenAI GPT-4o 6.84 / 13.57 6.80 / 13.59 5.82 / 12.64 OpenAI o3 9.53 / 17.96 9.38 / 17.95 8.39 / 17.04

By comparing the results in Table A6 with those in Table 2 of the main paper, we observe a clear performance improvement. This is expected, as replacing the visual map with textualized representations substantially reduces task difficulty, as it removes the need to assess visual capabilities such as OCR and grounding. We further note that prior works, such as MapBench [77] and CityBench [13], also focus on visual map interpretation without constructing explicit symbolic baselines.

##### Case 2: spatial reasoning

Q: Ponte Mammolo Station -> Sta. Maria del Soccorso Station? <image:rome.png> A (Skywork-R1V-38B): ... Line A: Orange Line B: Blue Line C: Green (✓) ... Ponte Mammolo Station ... on Line B (✓) ... Let me trace the path of Line B from Ponte Mammolo towards the center of Rome (✗) ... But I need to go from Ponte Mammolo to Sta. Maria del Soccorso. Looking at the map, I don’t see a direct connection between these two stations via Line B. (✗) So, I might need to transfer to another line ... Let me check if there’s another line that connects to Sta. Maria del Soccorso ... (✗) A (Gemini-2.5-Flash): Route Name: Line B Departure Stop: Ponte Mammolo Station Arrival Stop: Sta. Maria del Soccorso Station Number of Via Stops: 0 (✓)

#### C.5. Further Systematic Analysis on Failure Causes Case 1: grounding

Q: Hollywood/Highland Station -> Wilshire/Vermont Station? <image:los-angeles.png> A (Kimi-VL-A3B-Thinking): ... Let me start by locating both stations on the map. Hollywood/Highland is on ... A Line (✗)... Wilshire/Vermont is on the C Line (✗) ... (✗) A (Doubao-415): ... First, I need to see where these two sites are located on a map. Hollywood/Highland ... on B Line (✓) ... Wilshire/Vermont ... is on D Line and B Line (✓) ... Route Name: B Line Departure Stop: Hollywood/Highland Station Arrival Stop: Wilshire/Vermont Station (✓)

We further conduct more in-depth case analyses, which reveal that the main causes of failure are grounding and spatial reasoning, as illustrated in the following example. We

We systematically analyze failure causes, focusing on three MLLM capabilities pertinent to fine-grained visual

- Table A5. Evaluations on Beijing and Hangzhou (with and without English). S. represents results for short questions, while L. denotes results for long questions. Bold indicates performance improvements, while italicized values represent performance degradation.

Model Beijing (S. / L.) Beijing (w. English) (S. / L.) Hangzhou (S. / L.) Hangzhou (w. English) (S. / L.)

Kimi-VL-A3B-Instruct [64] 36.76% / 17.30% 23.78% / 20.81% 40.00% / 42.22% 42.22% / 45.95% Doubao-115 [21] 64.86% / 50.51% 45.95% / 52.70% 82.22% / 64.44% 67.78% / 65.56% Doubao-415 [21] 84.86% / 74.05% 88.65% / 85.95% 94.44% / 97.22% 87.78% / 100%

- Table A6. Evaluations of various MLLMs using symbolic representation. S. represents results for short questions, while L. denotes results for long questions. Bold indicates the best results among open-source and closed-source models, respectively.

Model Type Weighted Acc. (S. / L.) #Tokens (S. / L.) Open-source Models

Qwen2.5-VL-3B-Instruct [3] Base 22.83% / 19.79% 51 / 162 Qwen2.5-VL-32B-Instruct [3] Base 25.52% / 18.77% 97 / 297 Kimi-VL-A3B-Instruct [64] Base 39.58% / 34.81% 43 / 55

Closed-source Models

Doubao-115 [6] Base 81.16% / 72.66% 41 / 82 OpenAI 4o [46] Base 82.38% / 78.91% 40 / 70 Doubao-415 [6] Reasoning 95.31% / 93.66% 563 / 1561

- Table A7. Evaluations of various MLLMs on OCRBench. Bold indicates the best results among open-source and closed-source models, respectively. The references in the table indicate the result sources. All results are collected from the technical reports.

models lies in grounding and spatial reasoning.

### D. Case Analysis

We provide additional case analyses covering both correct and incorrect predictions, along with detailed comparisons of their respective reasoning processes. We first compare Doubao-415 and Doubao-428 (Figure A3), both of which reach the correct destination (from Augustins Station to Poterie Station) but via distinct reasoning paths. Doubao-415 correctly identifies early that both stations are on Line 18 and efficiently converges on the optimal, direct route without transfers. In contrast, Doubao-428 misclassifies Augustins as being on Line 12 and, assuming Poterie is on Line 18, proposes a transfer route via Plainpalais—functionally correct but suboptimal due to unnecessary complexity. Both models engage in extensive self-correction , highlighting the significant downstream impact of early-stage misjudgments. Moreover, visual reasoning limitations persist: despite correctly recognizing Augustins on Line 12, Doubao415 commits to a transfer path and fails to re-evaluate the possibility of a direct connection. This indicates room for improvement in both early visual grounding and global route optimality awareness. We then analyze the observed pattern when comparing the full input and text-only variants in the case (in Figure A4). The model with full visual access accurately identifies both stations on the Yellow Line and outputs the optimal direct route with the correct number of via stops. In contrast, the text-only variant makes an early misclassification, placing both stations on the Blue Line (Azul) and constructing a plausible but entirely incorrect sequence of intermediate stops. Although the final answer format appears coherent, the underlying logic is flawed due to the initial error in line recognition. This further illustrates the importance of visual input in spatial reasoning tasks, where even minor misinterpretations can lead to fundamentally incorrect conclusions. Additionally, some models, such as the InternVL3 series, default to rejection when visual input is absent. We further present several error cases in Figure A5, where Doubao-415 still exhibits visual confusion. In contrast, Qwen2.5-VL-32B-I, when lacking visual input, behaves differently from the InternVL3 series: rather than rejecting the query outright, it attempts to reason over the available information without producing a final answer, while explicitly notifying the missing visual input.

Model Type OCRBench Open-source Models

Qwen2.5-VL-3B-Instruct [3] Base 797 Qwen2.5-VL-72B-Instruct [3] Base 885 InternVL3-38B [102] Base 886 InternVL3-78B [102] Base 906 Kimi-VL-A3B-Instruct [64] Base 864 Kimi-VL-A3B-Thinking [64] Reasoning 864

Closed-source Models

OpenAI 4o [64] Base 815 Doubao1.5-VL (non-thinking) [21] Base 881 Doubao1.5-VL (thinking) [21] Reasoning 861

observe that OCR errors rarely occur, and most failure cases are instead caused by grounding or spatial reasoning issues.

For instance, in Case 1, Kimi-VL-A3B-Thinking incorrectly identifies the line of the departure station, indicating a grounding error that leads to subsequent reasoning failures. In Case 2, Skywork-R1V-38B correctly performs OCR and grounding in the initial steps, but fails in the reasoning stage (i.e., it does not prioritize locating the arrival station and instead attempts to construct incorrect indirect paths). Such failures reflect deficiencies in spatial reasoning, particularly in planning and executing core steps of pathfinding. These cases further indicate that the principal capability gap between open-source and closed-source

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

Arrival

###### Q: Augustins Station to Poterie Station + Question Template

[Figure 263]

A: Route Name: Line 18 Departure Stop: Augustins Station Arrival Stop: Poterie Station …

Doubao -428

Departure

Correction

Visual

A: <think> ... Is Augustins on Line 12? No. Again, is it on Line 12? Still no. Let’s check the stations on Line 12. Is Augustins among them? Line 12 includes stations like Blanche, Augustins, Pont-d'Arve... Is Augustins really on Line 12? ... So Augustins is on Line 12, and Poterie is on Line 18, meaning a transfer is likely needed. So the route could be: Augustins (Line 12) → Transfer to Line 18 → Poterie. Or maybe there's a direct connection? No, Augustins is on Line 12, and Poterie is on Line 18. So we must go from Augustins to a transfer station—maybe Place de Neuve? ... Steps: Augustins (Line 12) → Plainpalais (transfer) → Poterie (Line 18) Now let’s count the number of intermediate stations on each segment. Line 12: ... So Augustins is or isn't on Line 12? ...

Confusion

[Figure 264]

Doubao -428

... (continuous self-correction) Now, listing all Line 12 stations from Augustins to Plainpalais: ... Eventually, ... </think> Route Name: Line 12 Departure Stop: Augustins Station Arrival Stop: Plainpalais Station … Route Name: Line 18 Departure Stop: Plainpalais Station Arrival Stop: Poterie Station …

Correction

Visual

A: <think> First, observe the stations passed by each line. Augustins Station is on Line 14 and Line 18...But Poterie Station is on the purple Line 18... Wait, looking at the map again, Augustins is on the orange Line 12 and the purple Line 18? No, carefully see the station connections. In the map, Augustins is located on the line that ... Line 18. So Augustins is on Line 18, and Poterie is also on Line 18.

Confusion

[Figure 265]

... (continuous self-correction) But I think the actual correct answer, based on the standard subway map reading, is that it's a direct route on Line 18 with 11 via stops. ... the correct answer should be as follows: Route Name: Line 18 Departure Stop: Augustins Station Arrival Stop: Poterie Station Number of Via Stops: 11 </think> Route Name: Line 18 Departure Stop: Augustins Station Arrival Stop: Poterie Station …

Doubao

-115

- Figure A3. Case analysis of various MLLMs using REASONMAP (Case N1). For reasoning models, the reasoning process is explicitly marked with <think> and </think> tags. We highlight error contents in the answers with red and correct contents in green.

### E. Further Discussions

#### E.1. Limitations and Future Work

While REASONMAP provides a carefully curated benchmark for evaluating fine-grained visual reasoning with highresolution transit maps, we acknowledge that it represents only one type of structured visual diagram. As such, caution should be taken when generalizing observations to other domains that involve different types of visual content or reasoning styles. Additionally, although efforts were made to ensure diversity across cities and languages, the current version may not fully capture all geographic or linguistic variations. Future iterations could further expand coverage and explore additional forms of reasoning [17] to enhance generality.

Furthermore, we note that GeoGuessr-style localization

tasks [25, 30, 45] are compelling, as they emphasize detailed visual understanding of natural scenes and signage. We plan to pair transit maps with street view imagery to support cross-view reasoning and localization within REASONMAP, thereby expanding beyond static map inputs. In parallel, we will explore agent-based training and evaluation that moves from single-turn prediction to iterative planning with feedback, including reward designs for correctness, calibration, and format [99]. Finally, we will extend toward embodied settings [28] where agents perceive and act in interactive environments, enabling assessment of instruction following, route planning, and navigation under real-world constraints. Together, these directions broaden the benchmark from fine-grained visual reasoning to context-aware spatial intelligence and practical decision making.

Our REASONMAP can further evaluate the efficient

[Figure 266]

Departure

[Figure 267]

[Figure 268]

Q: Ameixoeira Station to Picoas Station + Question Template

[Figure 269]

Refusal

InternVL3 Onlytext A: r (reject)

Arrival

Visual

A: <think> … Ameixoeira and Picoas are stations there…Ameixoeira is on the Blue Line (Linha Azul), and Picoas is also on the Blue Line. Let me check the order of stations on the Blue Line. The Blue Line goes from ... Wait, no, actually, from Ameixoeira, the next stations towards the city center would be … No, wait, Picoas is on the Blue Line between Entre Campos and Marquês de Pombal? Wait, maybe I should list the stations in order. Let's see: ...

Confusion

[Figure 270]

... (continuous self-correction) Wait, maybe I got the direction wrong … Let me confirm: Ameixoeira (departure), next is São Sebastião (1), Campo Grande (2), Alameda (3), Entre Campos (4), then Picoas (arrival). So number of via stops is 4. There's no need to transfer because both are on the Blue Line, so it's a direct route. So the route should be Blue Line, departure Ameixoeira, arrival Picoas, via stops 4. Route Name: Azul Departure Stop: Ameixoeira Station Arrival Stop: Picoas Station Number of Via Stops: 4

Doubao -415 Onlytext

</think> Route Name: Azul…

A: <think> The user is asking for the optimal route from Ameixoeira Station to Picoas Station. First, I need to consult the metro map to determine which lines these stations belong to. Ameixoeira is on the Yellow Line (Amarela), and Picoas—after closely examining the map—is also on the Yellow Line, not at an interchange point. Starting from Ameixoeira and heading south along the Yellow Line, the stations passed are Lumiar, Quinta das Conchas, Campo Grande, and Saldanha before arriving at Picoas. This gives us 4 via stops (excluding the departure and arrival stations). No transfer is needed, as the entire trip is on the Yellow Line. Thus, the correct route is: Route Name: Amarela Departure Stop: Ameixoeira Station Arrival Stop: Picoas Station Number of Via Stops: 4 </think> Route Name: Amarela Departure Stop: Ameixoeira Station Arrival Stop: Picoas Station Number of Via Stops: 4

[Figure 271]

Doubao -415

- Figure A4. Case analysis of various MLLMs using REASONMAP (Case N2). For reasoning models, the reasoning process is explicitly marked with <think> and </think> tags. We highlight error contents in the answers with red and correct contents in green.

models from multiple efficiency strategies [14, 55, 56, 62, 63, 101]. Additionally, more fields [2, 8, 32, 34, 38, 39, 41, 66, 72, 91, 93, 94, 97, 98] require corresponding reasoningcentered benchmarks for proper evaluation.

#### E.2. Broader Impact

Advancing the capabilities of MLLMs in fine-grained visual reasoning has the potential to benefit a wide range of real-world applications, including navigation systems, urban planning tools, and assistive technologies for visually impaired individuals. By offering a structured and rigorous benchmark, REASONMAP encourages the development of MLLMs that can more effectively interpret complex visual artifacts and perform spatial reasoning. This could contribute to the long-term goal of building intelligent agents that interact more naturally and safely with human environments. Furthermore, the dataset’s emphasis on high-

resolution, globally sourced transit maps promotes research that is inclusive of diverse visual formats and geographic contexts. We hope REASONMAP can serve as a step toward more transparent, robust, and generalizable multimodal systems.

### F. Further Statement

#### F.1. Public Implementation

We benchmark the visual understanding and reasoning performance on REASONMAP across a diverse set of publicly available MLLMs:

- • KimiVL [64]2 . . . . . . . . . . . . . . . . . . . . . . . . . . MIT License
- • Skywork-R1V [49, 74]3 . . . . . . . . . . . . . . . . . MIT License

- 2https://github.com/MoonshotAI/Kimi-VL.
- 3https://huggingface.co/Skywork/Skywork-R1V2-

38B.

[Figure 272]

[Figure 273]

[Figure 274]

Departure

Q: York University Station to Chester Station + Question Template

[Figure 275]

Refusal

A: To provide an accurate route from York University Station to Chester Station, I would need access to the specific subway map ... Departure Stop: York University Station Arrival Stop: Station X … Let me know if you have additional information.

Qwen 2.5-VL -32B-I Onlytext

Arrival

Visual

A: <think> ... First, let's locate both stations on the map. York University Station is on Line 1 (the yellow line), which runs northsouth. Chester Station is also on Line 1, south of Castle Frank and north of Pape.

Confusion

... (continuous self-correction) After re-evaluating, the correct answer is that it's a direct route on Line 1 with 21 via stops, but I think the actual correct count, based on the standard TTC Line 1 stations between York University and Chester, is 21. However, to avoid further confusion, I'll refer to the correct method: list the stations in order, exclude departure and arrival, count them. Route Name: Line 1 Departure Stop: York University Station Arrival Stop: Chester Station Number of Via Stops: 21 </think> Route Name: Line 1 Departure Stop: York University Arrival Stop: Chester Number of Via Stops: 21

[Figure 276]

Doubao -415

- Figure A5. Case analysis of various MLLMs using REASONMAP (Case N3). For reasoning models, the reasoning process is explicitly marked with <think> and </think> tags. We highlight error contents in the answers with red and correct contents in green.

- • QVQ-72B-Preview [51]4 . . . . . . . . . . . . . . . Qwen License
- • Gemini-2.5-Flash [19]5 . . . . . . . . . . . . . . . . Closed-Source
- • InternVL-3.0 [102]6 . . . . . . . . . . . . . . . . . . . . .MIT License
- • Qwen2.5-VL [3]7 . . . . . . . . . . . . . . . . .Apache 2.0 License
- • Doubao-Pro 1.5 [6]8 . . . . . . . . . . . . . . . . . . . Closed-Source
- • OpenAI o3 [48]9 . . . . . . . . . . . . . . . . . . . . . . Closed-Source
- • OpenAI 4o [46]10 . . . . . . . . . . . . . . . . . . . . . Closed-Source

To ensure fair and reproducible evaluation, we implement all inference procedures by adhering closely to the official documentation and recommended practices of each model. The code is released under the MIT License to support transparency and reproducibility. Additionally, we provide detailed usage instructions on the project website to ensure easy access and reproducibility for future users.

#### F.2. Large Language Model Usage Statement

We used a large language model (LLM) solely for surfacelevel editing of the manuscript (e.g., rephrasing for clarity and concision, grammar/style polishing, and minor LATEX

- 4https://huggingface.co/Qwen/QVQ-72B-Preview.
- 5https://deepmind.google/technologies/gemini.
- 6https://github.com/OpenGVLab/InternVL.
- 7https://github.com/QwenLM/Qwen2.5-VL.
- 8https://www.volcengine.com/product/doubao.
- 9https://platform.openai.com/docs/models/o3.
- 10https://platform.openai.com/docs/models/gpt-

4o.

fixes). The LLM did not generate technical content, ideas, algorithms, proofs, code, experiments, figures, or tables; the authors conducted all research design, implementation, data processing, and analyses. The model did not produce or select citations; any suggestions were independently verified and replaced with primary sources. Interactions were limited to de-identified text snippets of the manuscript, and no non-public data, code, or unreleased results were uploaded. All LLM outputs were manually reviewed and edited by the authors. This usage does not affect reproducibility: every reported number is reproducible from our released code and configurations.

#### F.3. Ethics Statement

All experiments are conducted on REASONMAP, which is built using publicly available transit maps collected in compliance with relevant licenses and usage terms. The maps are selected to ensure geographic diversity and legal validity. Upon code release, we provide the source of each map for further reference. REASONMAP is intended solely for academic research on fine-grained visual understanding and spatial reasoning in MLLMs. It does not redistribute any copyrighted map images. All annotations are based on public information, contain no personal data, and are created under academic oversight. The benchmark is not intended for safety-critical use. We take care to ensure fairness, legal

compliance, and responsible data handling. Additionally, we will use the MIT License for code release on GitHub and the Apache License 2.0 for REASONMAP release on HuggingFace.

