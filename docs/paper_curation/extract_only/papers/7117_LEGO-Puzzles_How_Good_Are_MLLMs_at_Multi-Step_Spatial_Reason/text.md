arXiv:2503.19990v3[cs.AI]20Jun2025

# LEGO-Puzzles: How Good Are MLLMs at Multi-Step Spatial Reasoning?

## Kexian Tang1,2* Junyao Gao1,2* Yanhong Zeng1† Haodong Duan1† Yanan Sun1 Zhening Xing1 Wenran Liu1 Kaifeng Lyu3‡ Kai Chen1‡ Shanghai AI Laboratory1 Tongji University2 Tsinghua University3

{tangkexian, duanhaodong}@pjlab.org.cn

Project Page Code LEGO-Puzzles

## Abstract

Multi-step spatial reasoning entails understanding and reasoning about spatial relationships across multiple sequential steps, which is crucial for tackling complex realworld applications, such as robotic manipulation, autonomous navigation, and automated assembly. To assess how well current Multimodal Large Language Models (MLLMs) have acquired this fundamental capability, we introduce LEGO-Puzzles, a scalable benchmark designed to evaluate both spatial understanding and sequential reasoning in MLLMs through LEGO-based tasks. LEGOPuzzles consists of 1,100 carefully curated visual questionanswering (VQA) samples spanning 11 distinct tasks, ranging from basic spatial understanding to complex multi-step reasoning. Based on LEGO-Puzzles, we conduct a comprehensive evaluation of 20 state-of-the-art MLLMs and uncover significant limitations in their spatial reasoning capabilities: even the most powerful MLLMs can answer only about half of the test cases, whereas human participants achieve over 90% accuracy. Furthermore, based on LEGOPuzzles, we design generation tasks to investigate whether MLLMs can transfer their spatial understanding and reasoning abilities to image generation. Our experiments show that only GPT-4o and Gemini-2.0-Flash exhibit a limited ability to follow these instructions, while other MLLMs either replicate the input image or generate completely irrelevant outputs. Overall, LEGO-Puzzles exposes critical deficiencies in existing MLLMs’ spatial understanding and sequential reasoning capabilities, and underscores the need for further advancements in multimodal spatial reasoning.

*Equal contribution; work done during internships in Shanghai AI Lab-

oratory. †Project Leads. ‡Corresponding Authors.

## 1. Introduction

Spatial intelligence [5] has attracted growing attention due to its significance in various applications, including robotics control [21, 27], autonomous driving [17, 51], and automated assembly [12]. These complex real-world applications inherently require advanced multi-step spatial reasoning capabilities, which involve perceiving 3D-aware spatial relationships and reasoning about them across multiple sequential steps [5, 43, 56]. With the rapid advancement of Large Language Models (LLMs) [3, 16, 40, 48], Multimodal Large Language Models (MLLMs) [8, 33, 41, 47, 49] have also witnessed significant progress in perceiving visual information and interacting with humans through natural language. While MLLMs have made remarkable strides in fundamental tasks such as object recognition [13, 28] and optical character recognition [15, 35, 38, 45], existing evaluations [30, 36] suggest that their spatial reasoning abilities are still limited.

Research on evaluating MLLMs’ multi-step spatial reasoning capabilities remains largely unexplored. Existing studies primarily focus on assessing the spatial understanding capability, which pertains to the comprehension of a static scene. Some works [20, 29, 52] employ synthetic environments to render multiple simple 3D objects and then query the spatial relationships between them. However, such tasks tend to be overly simplistic for MLLMs to handle, lacking the diversity and complexity of real-world scenarios. Other studies [30, 37] construct spatial understanding tasks based on natural images, but this approach often involves manual annotations, which may limit scalability. Moreover, most existing evaluations rarely evaluate reasoning over sequences of spatial transformations or actions, leaving the multi-step aspect of spatial reasoning largely unaddressed.

In this work, we take inspiration from a common recreational activity, LEGO construction, to design a comprehensive evaluation framework for assessing the multi-step

|Dependency|
|---|
|Question: Please select the correct option (A, B, C, or D) that shows the LEGO part required to make the transition from the first state <image 1> to the second <image 2>. Options: A.<br><br><image 1> <image 2><br><br>B. C. D.<br><br>[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]|

|Ordering|
|---|
|Question: You will be provided with the current assembly state image <image 1>, the target assembly state image <image 2>, and four step images labeled A, B, C, and D. Your goal is to arrange the four step images in the correct order that transitions from the current state to the target state. Options: A.<br><br>Answer: CBAD <image 1> <image 2><br><br>B. C. D.<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]|

|Height|
|---|
|Question: Which LEGO object is shorter in 3D space? Options:<br><br>A. The LEGO piece pointed by the blue arrow<br>B. The LEGO piece pointed by the red arrow<br>C. They are the same height<br><br><br>[Figure 13]<br><br>|

|Adjacency|
|---|
|Question: Are the LEGO pieces pointed to by the two red arrows adjoining or separated Options:<br><br>A. adjoining<br>B. seperated<br><br><br>[Figure 14]<br><br>|

|Next Step|
|---|
|Question: Please select the correct option (A, B, C, D, or E) that shows the assembly state after adding the next part <image 1> onto the current state <image2>. The next part is a step toward achieving the final product <image 3>. Options: A.<br><br><image 1> <image 2><br><br>B.<br><br>E.<br><br><image 3><br><br>C. D.<br><br>[Figure 15]<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]|

|Outlier|
|---|
|Question: Please select the correct option (A, B, C, D, or E) that does NOT represent a step in transitioning from the current state <image 1> to the target state <image 2>. Options: A.<br><br>C.<br><br><image 1> <image 2><br><br>B.<br><br>D. E.<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]|

|Multiview|
|---|
|Question: Based on the LEGO piece shown in the reference image, which of the following images shows the LEGO piece from a left-toright perspective? Options: A.<br><br>C.<br><br>B.<br><br>D.<br><br>reference image<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]|

|Rotation Status|
|---|
|Question: Does the additional piece <image 1> need to be rotated to attach it to the base structure <image 2> in order to form the final structure <image 3>? Options:<br><br>A. Yes<br><br>B. No <image 1> <image 2> <image 3><br><br><br>[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]|

|Backwards|
|---|
|Question: This is the target assembly image: <image 1>Which option (A, B, C, or D) correctly shows the correct assembly step? Options: A.<br><br>C.<br><br>B.<br><br>D.<br><br><image 1><br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]|

|Rotation|
|---|
|Question: From a top-down perspective, how many degrees has the LEGO figure in <image 2> rotated clockwise around its center relative to <image 1>? Options:<br><br>A. 30°<br><br>B. 60°<br>C. 90°<br>D. 120° <image 1> <image 2><br><br><br>[Figure 44]<br><br>[Figure 45]|

|Position|
|---|
|Question: Based on the current state <image 1>, the next part <image 2> to install, and the state after installation <image 3>, which of the following images shows the correct installation point? Options: A. B<br><br>.<br><br><image 1> <image 2> <image 3><br><br>C<br><br>.<br><br>D<br><br>.<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]|

Figure 1. Task examples of LEGO-Puzzles. From left to right, the columns represent tasks in Spatial Understanding, Single-Step Sequential Reasoning, and Multi-Step Sequential Reasoning. Note: The questions above are slightly simplified for clarity and brevity.

spatial reasoning capabilities of MLLMs. The assembly process of a complete LEGO model typically encompasses dozens or even hundreds of discrete construction steps, providing an ideal foundation for testing sequential reasoning abilities. Each step requires accurate comprehension of geometry, orientation, and connection mechanisms of LEGO pieces to successfully follow the provided illustrations. Based on publicly available LEGO projects with detailed step-by-step assembly instructions, we introduce LEGO-Puzzles, a benchmark specifically engineered to evaluate MLLMs’ multi-step spatial reasoning capabilities. In total, LEGO-Puzzles encompasses a diverse collection of over 1,100 carefully curated visual question-answering (VQA) pairs spanning 11 distinct tasks, grouped into three categories (Fig. 1): a set of fundamental tests to assess MLLMs’ basic spatial understanding capabilities, including recognition of height relationships, rotation angles, 3D viewpoints, etc. Building upon this foundation, we construct both single-step and multi-step sequential reasoning evaluations based on LEGO assembly sequences to examine models’ sequential reasoning ability. In addition, a single LEGO instruction manual can yield hundreds of evaluation samples, enabling efficient benchmark expansion, while the availability of tens of thousands of open-source LEGO models across diverse categories ensures high scalability and diversity of the benchmark.

Leveraging LEGO-Puzzles, we conduct comprehensive evaluations of 20 state-of-the-art MLLMs, including proprietary models such as GPT-4o and Gemini-2.0-Flash, as well as leading open-source alternatives [8, 42, 49, 55]. Our experimental results reveal a substantial gap between current MLLMs and human-level proficiency. Even the strongest models struggle with basic spatial understanding tasks, such as accurately identifying the height relationships in 3D space. Among open-source models, only a few achieve performance notably above random guessing across different tasks.

To further investigate whether MLLMs can transfer there spatial understanding and reasoning abilities to image generation, we design generation tasks based on LEGOPuzzles. For instance, given an assembly illustration, an MLLM is tasked with generating an image of the intermediate state following the specified assembly operation. In these generation tests, most of the evaluated models fail completely, either disregarding the provided instructions or generating images that are entirely irrelevant to the intended LEGO configuration.

In summary, LEGO-Puzzles provides a comprehensive evaluation of the spatial understanding and sequential reasoning capabilities of MLLMs. Our main contributions are as follows:

#### • Evaluation for multi-step spatial reasoning. Built upon

LEGO’s step-by-step building process, LEGO-Puzzles is the first benchmark explicitly designed to assess multistep spatial reasoning, where each task requires reasoning over up to 7 LEGO construction steps.

- • Progressive and comprehensive task coverage. Our benchmark includes a diverse set of tasks spanning basic spatial understanding, single-step reasoning, and multistep reasoning. This enables systematic evaluation of MLLMs’ reasoning capabilities across increasing levels of spatial and sequential complexity.
- • Exploratory evaluation of reasoning transfer to image generation. Building upon LEGO-Puzzles, we further conduct exploratory experiments to examine whether MLLMs can transfer their spatial understanding and reasoning abilities to image generation tasks. This sheds light on the models’ capacity for multimodal generalization beyond recognition and question answering.

## 2. Related Work

General Multi-Modal Evaluation Benchmarks. Recent years have seen significant advancements in multimodal large language models (MLLMs), accompanied by a surge in benchmark datasets evaluating their visual understanding. Several comprehensive benchmarks have been introduced to assess various multimodal capabilities. MME [13] provides a systematic evaluation of 14 image-centric tasks, revealing persistent challenges such as object hallucination and spatial reasoning failures. MMBench [36] introduces a bilingual multiple-choice format for fine-grained multimodal assessment. Moving beyond static images, SEEDBench [24] evaluates generative comprehension across 19K Q&A pairs spanning both image and video reasoning, showing that temporal understanding remains a major limitation. For expert-level reasoning, MMMU [58] presents a discipline-specific benchmark across 183 subtopics, revealing substantial knowledge gaps even in leading MLLMs even in leading MLLMs, such as GPT-4o and Gemini. Overall, these benchmarks reveal that while MLLMs have made progress, they still struggle with spatial understanding, temporal coherence, multimodal integration, and highlevel reasoning, presenting clear directions for future research.

Visual-Spatial Understanding in MLLMs. Multimodal large language models (MLLMs) have made significant strides in vision-and-language tasks, yet they still struggle with 3D spatial understanding. Benchmarks such as 3DSRBench [37] show that even the most advanced models achieve only 45–50% accuracy on 3D spatial tasks and experience substantial performance drops under unusual camera angles. To enhance spatial reasoning, several studies have explored Chain-of-Thought (CoT) prompting. For example, Park et al. [43] demonstrate that combining CoT with explicit image-to-text conversion can improve gener-

alization from simple to hard visual reasoning tasks. However, beyond such tailored interventions, traditional CoT prompting alone has generally failed to improve spatial reasoning performance [56]. In response, alternative approaches have emerged. Spatially enriched datasets, such as Spatial Aptitude Training (SAT) [44], significantly boost zero-shot performance across real-image benchmarks. Architectural innovations like CAD-GPT [50], which embeds 3D coordinates into language representations, and MVoT [26], which introduces visual sketching during inference, further expand the solution space. Additionally, lightweight strategies like Coarse Correspondences [32] improve spatial understanding without requiring model fine-tuning. Despite these advances, achieving human-level 3D spatial reasoning in MLLMs remains an open challenge.

## 3. LEGO-Puzzles

In this section, we introduce LEGO-Puzzles, a diverse and comprehensive benchmark designed to evaluate the multi-step spatial reasoning capability of MLLMs in detail. Specifically, we first introduce the motivation and definition of each task in Sec. 3.1. Then, we introduce our dataset curation process, including data collection, question-answer generation, and quality control, in Sec. 3.2.

### 3.1. Task Definition

To enable a more comprehensive and progressively structured evaluation of multi-step spatial reasoning in MLLMs, we define three categories of tasks. This framework is grounded in insights from cognitive psychology and human developmental stages in acquiring spatial intelligence [5, 39, 54]. Using LEGO building as a concrete and intuitive example, we observe that humans typically develop spatial reasoning abilities in stages—from basic spatial understanding, to reasoning through individual assembly steps, and ultimately to reasoning across multiple sequential steps. Based on this developmental trajectory, our benchmark is divided into three categories, as illustrated in Fig. 2 (a).

#### Type 1: Spatial Understanding. This category fo-

cuses on the ability to understand the spatial relationships between each LEGO piece and how these pieces relate from different perspectives in 3D space: (1) Height: Distinguish the relative heights of LEGO objects. (2) Adjacency: Determine whether LEGO objects are adjacent or separated. (3) Rotation: Calculate the angle of rotation between a LEGO object and its corresponding rotated version. (4) Multiview: Predict the current LEGO status from different viewpoints.

#### Type 2: Single-Step Sequential Reasoning. Build-

ing upon spatial understanding, this category evaluates the model’s ability to reason through the dependencies and assembly logic. Tasks are designed based on single-step actions, mirroring how humans typically progress from spatial perception to carrying out one assembly step at a time

during the building process: (5) Rotation Status: Determine whether a LEGO piece requires rotation before installation. In contrast to Task 3, this task focuses on reasoning from an assembly perspective, with finer granularity. (6) Position: Identify the correct assembly position to place the next LEGO piece. (7) Next-Step: Predict the next LEGO status based on the current status and the new pieces. (8) Dependency: Identify which pieces are necessary to transition from the current to the next assembly stage.

Type 3: Multi-Step Sequential Reasoning. The final category assesses the ability to reason across multiple steps in the assembly process. These tasks build upon the singlestep reasoning skills from Type 2 and require planning over extended sequences (involving up to 7 intermediate stages): (9) Backwards: Identify the correct intermediate stage of a LEGO build from the full assembly state. (10) Ordering: Determine the correct assembly order of the provided final LEGO images. (11) Outlier: Detect the LEGO status that does not belong to the provided assembly sequence.

In conclusion, LEGO-Puzzles consists of over 1,100 visual question-answering (VQA) pairs derived from 407 LEGO building instructions, encompassing 11 tasks across spatial understanding, single-step and multi-step sequential reasoning. Each task contains 100 samples to ensure balanced evaluation across categories.

### 3.2. Dataset Curation

As illustrated in Fig. 2, our pipeline consists of three key steps: data collection, question-answer generation, and quality control. This design ensures the scalability, accuracy, and reliability of our data.

Data Collection. Data collection consists of three stages. (1). LEGO Project Collection. We collect a diverse set of open-source LEGO source files from the Internet, each containing detailed step-by-step building instructions and part lists. To ensure suitable task complexity, we filter for projects with moderate final size. Extremely large builds exhibit high structural complexity, and the small visual changes introduced by added pieces make it difficult for models to detect step-wise differences. Conversely, overly small builds are filtered out for low spatial complexity and insufficient steps for multi-step reasoning. We also ensure category diversity, covering animals, furniture, vehicles, and more, to increase task variety. (2). Rendering and Transformation. We render each project into PDF format using the publicly available software Studio1, keeping the camera viewpoint fixed across steps to maintain spatial and temporal consistency. The tool allows flexible editing of the source files, enabling us to modify part attributes such as type, quantity, color, and position as needed for task construction. For example, in the Rotation and Multiview task,

1https://www.bricklink.com/v3/studio/download. page

we apply POV-Ray style rendering and adjust lighting to simulate different viewing angles. In the Backward task, we introduce deliberate errors in part attributes to generate incorrect assembly states. (3).Metadata Extraction and Unification. We employ PDF-Extract2 to extract structured information from the rendered PDFs, including individual LEGO pieces and assembled objects. All visual assets are processed under a unified naming convention and organized for downstream question-answer generation across all defined task types.

Question-Answer Generation. To support scalable and structured data construction, we manually design taskspecific templates tailored to each task. Each example includes an instruction defining the model’s role, a question referencing input images using tokens like <image x>, and a ground-truth answer. For example, in the Position task, the model is provided with the current assembly state, the part to install, the resulting state, and several candidate placements. This standardized template design allows flexibility in the number of input images required by each task, making it suitable for both single-step and multi-step reasoning scenarios.

Quality Control. To ensure data quality and reliability, we implement a multi-stage human-in-the-loop review process. 1)Duplicate filtering. We perform duplication checks by computing similarity scores across rendered images, identifying visually redundant or recolored samples. These are further reviewed manually, and duplicates are removed to reduce redundancy and improve evaluation quality. 2)Image Quality and Format Check. We manually verify all rendered images to ensure consistency in camera perspective, correctness of part attributes (e.g., color, shape, quantity), and adherence to naming conventions. Any files with errors are either corrected or discarded. 3)Template and annotation validation. Each question-answer pair is verified by three trained annotators. Reviewers confirm that the images referenced by tokens (<image 1>, <image 2>, ...) appear in the correct order. Samples with unresolved disagreements are either revised or removed.

Given the vast number of high-quality open-source LEGO projects available, our pipeline is inherently scalable and can be extended in an semi-automated manner to support larger and more diverse benchmarks in the future.

## 4. Experiment

### 4.1. Experimental Setting

Benchmark Models. We extensively evaluate 20 models, covering a diverse range of architectures, sizes, and training processes for Spatial Understanding and Sequential Reasoning tasks. For open-source models, we evaluate MiniCPM-V2.6 [57], Qwen2-VL-[7B/72B] [49],

2https://github.com/opendatalab/PDF-Extract-Kit

Rendering & Transformation

###### Metadata Extraction

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

###### DataCollection

[Figure 57]

Angle Viewpoint

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Number Piece Object

Color Position

[Figure 63]

[Figure 64]

Control

cross-validation

QA GenerationQuality

Duplicate Filtering

Image Quality Image Format

Template Validation Annotation Validation

annotator annotator

| | |
|---|---|
| | |
| | |

Task-Specific Templates Image List

Question: Based on the current state <image 1>… <image 2> … <image 3>, which of the following …?

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Options: A.<image 4> B.<image 5> C.<image 6> D.<image 7>

[Figure 71]

Ground Truth Answer: C

(a). Problem statistics in LEGO-Puzzles

(b). Data curation pipeline

Figure 2. Problem statistics and data curation in LEGO-Puzzles.

Qwen2.5-VL-[7B/72B] [4], InternVL2.5-[8B/78B] [7], VILA1.5-13B [31], Idefics3-8B [23], DeepSeek-VL2[Tiny/Small] [55], Pixtral-12B [1], LLaVA-OneVision7B [25], and EMU3 [53]. For proprietary models, we evaluate Claude-3.5-Sonnet [2], Gemini-1.5-Flash, Gemini-1.5Pro, Gemini-2.0-Flash [47], GPT-4o (20241120), and GPT4o-mini [41]. Moreover, all evaluations are conducted in a zero-shot setting for a fair comparison.

There is a significant gap between open-source and proprietary MLLMs in both spatial understanding and sequential reasoning abilities as shown in Tab. 1. Most opensource MLLMs perform only marginally better than Random, while leading proprietary models, such as Gemini-2.0Flash and GPT-4o, exhibit strong spatial reasoning capabilities, achieving overall accuracies of 54.0% and 57.7%, respectively.

Baselines. We provide two baselines for comparison:

Gap between Human and MLLMs. To study the performance gap between human and MLLMs on LEGO-Puzzles tasks, we randomly select 20 questions from each task to create LEGO-Puzzles-Lite, resulting in a total of 220 QA pairs, and use this subset for investigation. We invited 30 human experts to solve these questions, ensuring a diverse and representative evaluation of human-level reasoning. Our findings indicate that human experts consistently achieve significantly higher overall performance (93.6%), as shown in Tab. 2. In contrast, current MLLMs fall short, with even the most advanced models, Gemini-2.0-Flash and GPT-4o, trailing over 30% behind human performance across all tasks. This persistent gap highlights the need for comprehensive and substantial improvements in our LEGOPuzzles.

- • Random indicates the accuracy of random selection for each question, assuming equal probability for all options.
- • p-value-based critical value indicates the minimum accuracy required to statistically surpass random guessing at a given significance level (p = 0.05).

Evaluation Metrics. For all questions in LEGO-Puzzles, we adopt accuracy (%) as the evaluation metric. For multiple-choice questions, we follow VLMEvalKit [11], applying rule-based matching as a first step, and resort to LLM-based choice matching using ChatGPT [41] when heuristic matching fails. For the Ordering task, we adopt the same strategy: rule-based extraction of the predicted step sequence from the model’s response, followed by LLMbased matching when necessary.

Challenges of LEGO-Puzzles. 1). Spatial Understanding. Our results reveal that MLLMs struggle significantly with tasks requiring true 3D spatial reasoning. In the Height task, we deliberately construct cases where interpreting the image from a 2D viewpoint leads to a different answer than interpreting it from a true 3D perspective. As shown

### 4.2. Main Results

We include evaluation results for spatial understanding and sequential reasoning in Tab. 1 and Tab. 2. We summarize key findings as below.

Gap between Open-source and Proprietary Models.

- Table 1. Full evaluation results of 20 MLLMs on LEGO-Puzzles. Dark Gray and Light Gray indicates the best performance for each task among all models and open-source models respectively. We highlight the top 3 models by overall performace using Dark Green ,

Medium Green , and Light Green , respectively.

Models

Spatial Understanding Single-Step Reasoning Multi-Step Reasoning

Overall Height Adjacency Rotation Multiview Next-Step Dependency Rotation Stat. Position Backwards Ordering Outlier

Proprietary

Claude-3.5-Sonnet 39.0 60.0 42.0 48.0 61.0 78.0 58.0 37.0 49.0 54.0 64.0 53.6 Gemini-1.5-Flash 29.0 58.0 28.0 45.0 57.0 77.0 57.0 32.0 28.0 20.0 51.0 43.8

- Gemini-1.5-Pro 35.0 58.0 38.0 56.0 59.0 84.0 61.0 39.0 35.0 44.0 59.0 51.6

- Gemini-2.0-Flash 35.0 70.0 49.0 45.0 69.0 81.0 54.0 46.0 56.0 46.0 43.0 54.0 GPT-4o 49.0 66.0 41.0 51.0 65.0 87.0 51.0 51.0 53.0 72.0 49.0 57.7 GPT-4o-mini 31.0 53.0 26.0 51.0 27.0 71.0 57.0 32.0 50.0 7.0 27.0 39.3 Open-source MiniCPM-V2.6 26.0 56.0 22.0 44.0 34.0 50.0 51.0 29.0 23.0 0.0 19.0 32.2 Qwen2-VL-7B 31.0 57.0 30.0 40.0 44.0 70.0 48.0 26.0 13.0 9.0 28.0 36.0 Qwen2.5-VL-7B 22.0 54.0 30.0 43.0 43.0 66.0 53.0 28.0 19.0 9.0 27.0 35.8 InternVL2.5-8B 35.0 53.0 23.0 37.0 38.0 48.0 64.0 25.0 35.0 0.0 29.0 35.2 VILA1.5-13B 26.0 55.0 26.0 35.0 17.0 34.0 48.0 26.0 12.0 4.0 22.0 27.7 Idefics3-8B 29.0 51.0 23.0 23.0 18.0 20.0 47.0 30.0 24.0 4.0 24.0 26.6 InternVL2.5-78B 41.0 62.0 32.0 47.0 60.0 79.0 58.0 32.0 40.0 15.0 37.0 45.7 Qwen2-VL-72B 40.0 62.0 37.0 51.0 57.0 79.0 49.0 43.0 34.0 26.0 31.0 46.3 Qwen2.5-VL-72B 41.0 60.0 43.0 37.0 54.0 75.0 59.0 53.0 74.0 46.0 41.0 53.0 DeepSeek-VL2-Small 31.0 52.0 36.0 41.0 38.0 57.0 59.0 28.0 41.0 3.0 26.0 37.5 DeepSeek-VL2-Tiny 32.0 52.0 36.0 24.0 27.0 25.0 47.0 27.0 26.0 4.0 16.0 28.7 Pixtral-12B 31.0 68.0 24.0 24.0 21.0 38.0 53.0 21.0 24.0 3.0 37.0 31.3 LLaVA-OneVision-7B 42.0 59.0 21.0 41.0 30.0 50.0 59.0 26.0 20.0 0.0 22.0 33.6 EMU3 31.0 52.0 24.0 25.0 17.0 25.0 47.0 25.0 24.0 0.0 20.0 26.4 Baseline Random Guessing 33.0 50.0 25.0 25.0 20.0 25.0 50.0 25.0 25.0 4.2 20.0 27.5 ↑ Random (p < 0.05) 42.0 59.0 33.0 33.0 28.0 33.0 59.0 33.0 33.0 9.0 28.0 35.5

- Table 2. Comparing top-performing MLLMs with human proficiency on LEGO-Puzzles-Lite. The best results are marked in bold. The top 3 overall performances are highlighted in Dark Green , Medium Green , and Light Green , respectively.

Spatial Understanding Single-Step Reasoning Multi-Step Reasoning

Models

Overall Height Adjacency Rotation Multiview Next-Step Dependency Rotation Stat. Position Backwards Ordering Outlier

###### LEGO-Puzzles-Lite

Human proficiency 70.0 95.0 95.0 100.0 90.0 100.0 100.0 95.0 95.0 95.0 95.0 93.6 Claude-3.5-Sonnet 40.0 55.0 50.0 50.0 60.0 75.0 55.0 35.0 60.0 55.0 60.0 54.1 Gemini-2.0-Flash 30.0 65.0 55.0 40.0 80.0 85.0 60.0 40.0 60.0 50.0 45.0 55.5 GPT-4o 35.0 75.0 45.0 50.0 60.0 85.0 60.0 60.0 55.0 60.0 65.0 59.1 InternVL2.5-78B 40.0 55.0 30.0 45.0 60.0 85.0 55.0 30.0 25.0 20.0 50.0 45.0 Qwen2-VL-72B 30.0 65.0 45.0 50.0 55.0 80.0 45.0 35.0 30.0 15.0 35.0 44.1 Qwen2.5-VL-72B 50.0 65.0 40.0 45.0 55.0 75.0 65.0 55.0 80.0 25.0 50.0 55.0

in Tab. 1, most models (11/20) perform worse than Random. These results suggest that MLLMs tend to defaultly answer questions based on a 2D projection rather than a true 3D perspective. This observation highlights their reliance on 2D spatial priors during inference, underscoring the need for further research on equipping models with ro-

bust 3D spatial reasoning capabilities. Similarly, in the Rotation task, which requires identifying the rotation angle of an object, 6 out of 20 models fall below Random, with most failing to reach even 40% accuracy. This further indicates that current MLLMs struggle to perceive and distinguish object orientation changes reliably. 2). Sequential

Reasoning. LEGO-Puzzles also reveals substantial limitations in MLLMs’ sequential reasoning capabilities, particularly when multiple reasoning steps are required. While performance on single-step tasks such as Dependency and Next-Step is relatively stronger, models struggle considerably with multi-step tasks. As shown in Tab. 1, almost half of the models score below ↑ Random in the Ordering task, with some models (e.g., InternVL2.5-8B, LLaVAOneVision-7B, EMU3) failing completely. Similar trends are observed in the Backwards task, where 8 out of 14 opensource models perform below ↑ Random. To further understand the effect of step length, we conduct the Next-kStep experiment in Sec. 4.4 which systematically evaluates model accuracy under increasing reasoning steps.

In conclusion, our LEGO-Puzzles highlights both the spatial understanding and sequential reasoning abilities of MLLMs. The overall results suggest significant room for improvement, particularly in domains involving relative relationships, rotation perception, and long-range sequential reasoning.

### 4.3. Image Generation Evaluation

As mentioned in Sec. 1, to further investigate whether MLLMs can transfer their spatial understanding and reasoning abilities to image generation, we construct a set of generation tasks based on LEGO-Puzzles. Specifically, we convert the original multiple-choice format in LEGO-Puzzles into image generation tasks, where models are required to directly produce a visual output rather than selecting from given options. We select five tasks—Rotation* and Multiview* from spatial understanding, and Position*, Dependency*, and Next-Step* from single-step sequential reasoning—resulting in a total of 100 questions.

We evaluate the open-source models Emu2 [46], GILL [22], and Anole [9], as well as the proprietary models GPT-4o△, GPT-4o and Gemini-2.0-Flash, all of which support long-range sequence input and image output. For evaluation, traditional metrics such as FID [19], CLIPScore [14, 18], and X-IQE [6] mainly assess image fidelity or cross-modal alignment, often relying on pretrained model priors or fixed scoring heuristics. They struggle to capture the fine-grained spatial accuracy required in LEGO assembly tasks, where even small errors—such as misaligned parts or incorrect orientations—can invalidate the result. Furthermore, many recent multimodal evaluation metrics depend on GPT-based models [34], introducing uncontrollable bias into the evaluation process. Therefore, we enlist 5 human experts to assess model performance across two dimensions: appearance similarity and instruction following. Each aspect is rated on a scale from 0 to 3.

Tab. 3 presents the human evaluation results across five generation tasks. Overall, proprietary models outperform

△refers to the version of GPT-4o released prior to March 6, 2025.

open-source ones in both appearance consistency (App) and instruction adherence (IF). Among them, GPT-4o achieves the highest overall scores (App: 2.25, IF: 1.77), followed by Gemini-2.0-Flash (App: 2.15, IF: 1.08). However, both models show clear room for improvement in instruction following, scoring only 1.77 and 1.08 out of 3, respectively. For GPT-4o△, the results suggest that the model may not directly edit the input image, but instead reinterpret the scene semantically and regenerate it based on textual understanding. This leads to lower appearance consistency (App: 1.24), reflecting a conceptual reconstruction process rather than precise visual editing. Among open-source models, Emu2 shows some ability to preserve visual appearance (App: 0.89) but fails almost entirely in instruction following (IF: 0.05), treating the task more as image replication than reasoning-based generation. GILL and Anole perform the worst, with near-zero scores across all tasks and frequently irrelevant outputs. The qualitative results are shown in Fig. 3.

Overall, these results show that current models—especially open-source ones—struggle significantly with instruction-grounded image generation, highlighting the challenges of spatially grounded visual synthesis.

### 4.4. Exploring Multi-Step Sequential Reasoning

Experiments in Sec. 4.2 show that MLLMs perform noticeably worse on multi-step sequential reasoning tasks compared to single-step ones. To systematically investigate how reasoning performance is affected by the number of steps involved, we design a fine-grained task called Next-k-Step. This task extends the original Next-Step setting in LEGOPuzzles by explicitly controlling the number of steps k required to reach the target assembly state.

Experimental Setup. In contrast to Next-Step in LEGOPuzzles, Next-k-Step requires MLLMs to identify the correct LEGO object after sequentially adding k additional LEGO pieces to the current LEGO object. We set k = 1,2,3,4,5 and construct 20 test cases for each k value. Specifically, each input includes the current LEGO object (x1), the next k LEGO pieces (x2,x3,...,xk+1) and the target LEGO object (xk+2), along with the corresponding text instructions. The model is expected to select the correct answer from four options (A, B, C, D). To further explore whether the widely used Chain-of-Thought (CoT) prompting strategy can enhance performance in multi-step reasoning, we evaluate model accuracy under two prompting conditions: standard prompting (without CoT) and explicit step-by-step reasoning (with CoT). We conduct this experiment using the four top-performing models from the original Next-Step task: GPT-4o, Gemini-2.0-Flash, Qwen2.5VL-72B, and InternVL2.5-78B.

Performance Degradation when k Increases. As shown in Tab. 4, GPT-4o and Gemini-2.0-Flash exhibit clear per-

- Table 3. Evaluation on generation. We conduct human-based evaluation to assess the “Appearance” (App) and “Instruction Following” (IF) scores of GPT-4o, Gemini-2.0-Flash, GPT-4o△, Emu2, GILL, and Anole, using a scoring scale from 0 to 3 for both dimensions.

|Task \MLLM<br><br>|GPT-4o Gemini-2.0-Flash GPT-4o△ Emu2 GILL Anole| | | | | |
|---|---|---|---|---|---|---|
| |App IF<br><br>|App IF<br><br>|App IF|App IF<br><br>|App IF<br><br>|App IF|
|Rotation* Multiview* Position* Dependency* Next-Step*<br><br>|2.35 1.75 2.30 2.00 2.30 1.65 2.05 2.00 2.25 1.45<br><br>|2.05 1.45 2.40 1.65 2.85 1.30 1.70 0.95 1.75 0.05|0.65 0.50<br><br>1.95 0.40<br><br>2.95 1.00<br><br><br>0.35 0.05 0.30 0.00<br><br>|1.70 0.00 1.65 0.25 0.50 0.00 0.55 0.00 0.05 0.00<br><br>|0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00|0.10 0.00 0.05 0.00 0.00 0.00 0.00 0.00 0.00 0.00<br><br>|
|Overall<br><br>|2.25 1.77|2.15 1.08|1.24 0.39<br><br>|0.89 0.05|0.00 0.00<br><br>|0.03 0.00|

Dependency*

Rotation* Position*

Multiview*

Question: Based on the current assembly state <image 1>. the next assembly state <image 2>, generate an image that shows the LEGO part required to make the transition from the current state to the next state.

<image 1> <image 2> Emu2 GPT-4o△ Gemini-2.0-Flash GPT-4o

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Question: Based on reference image, generate an image of the LEGO object rotated clockwise by 60 degrees from a top-down perspective around its center.

[Figure 78]

Reference Image Emu2 GPT-4o△ Gemini-2.0-Flash GPT-4o

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Question: Based on reference image, generate an image showing the LEGO piece from a top-to-down perspective.

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Reference Image Emu2 Gemini-2.0-Flash GPT-4o

Question: Based on the current state <image1>, the next part to install <iamge2>, and the state after installation <image 3>, generate an image showing the correct assembly point for the next part to be installed, with the installation point clearly labeled on the current assembly state.

<image 1> <image 2> <image 3> Emu2 GPT-4o△ Gemini-2.0-Flash GPT-4o

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

GPT-4o△

Figure 3. Qualitative visual results for image generation tasks. Note: The questions above are slightly simplified for clarity and brevity.

- Table 4. Evaluation on Next-k-Step. k represents the number of steps, and CoT refers to adding a “Think step by step before answering” instruction in QA pairs, similar to those in LLMs.

Limited Effectiveness of Chain-of-Thought (CoT). As shown in Tab. 4, CoT provides clear gains at k = 1 for GPT-4o (↑30%) and InternVL2.5-78B (↑20%), and a modest improvement for GPT-4o at k = 2 (↑10%). However, for k ≥ 2, its effect vanishes or worsens—GPT-4o drops to 0% by k = 4, and InternVL2.5-78B stays near or below the random baseline (25%). For Gemini-2.0-Flash and Qwen2.5-VL-72B, CoT yields no consistent gain. In particular, Qwen2.5-VL-72B performs stably with or without CoT, suggesting robustness without reliance on explicit step-by-step prompting. Overall, CoT offers limited help in multi-step spatial reasoning, especially as task complexity increases.

|Setting|GPT-4o Gemini-2.0-Flash Qwen2.5-VL-72B InternVL2.5-78B| | | |
|---|---|---|---|---|
| |w/o CoT w. CoT<br><br>|w/o CoT w. CoT<br><br>|w/o CoT w. CoT<br><br>|w/o CoT w. CoT|
|k = 1<br>k = 2<br>k = 3<br>k = 4<br>k = 5<br>|45.0 75.0 15.0 25.0 5.0 5.0 5.0 0.0 5.0 0.0<br><br>|85.0 60.0 45.0 50.0 35.0 40.0 35.0 50.0 20.0 25.0<br><br>|65.0 65.0 60.0 55.0 75.0 75.0 65.0 65.0 65.0 65.0|35.0 55.0 30.0 20.0 10.0 20.0 20.0 5.0<br><br>25.0 10.0|

formance degradation as the number of reasoning steps k increases, reflecting their difficulty in handling multi-step sequential reasoning. In contrast, Qwen2.5-VL-72B maintains stable accuracy ( 65%) across all k values, suggesting stronger robustness to multi-step reasoning. InternVL2.578B performs near the random baseline (25%) regardless of k, indicating limited overall effectiveness. These results suggest that most current MLLMs lack the capacity to track and integrate spatial transformations over multiple steps, with accumulated errors leading to inconsistent predictions in longer reasoning chains.

## 5. Discussion

While LEGO-Puzzles is built on rendered data, it aims to evaluate fundamental spatial reasoning capabilities that are also essential in real-world scenarios. To assess its generalizability beyond synthetic environments, we compare model performance on LEGO-Puzzles with 3DSRBench [37], a benchmark based on natural images. Both

- Table 5. Pearson correlation coefficients (PCC) and p-values for height and adjacency tasks.

Task PCC P-value Height 0.93 0.00723 Adjacency 0.98 0.00046

datasets contain conceptually similar tasks—specifically, the Height task in LEGO-Puzzles aligns with Height in 3DSRBench, and Adjacency in LEGO-Puzzles corresponds to the Location task in 3DSRBench.

We evaluate all proprietary models tested in LEGOPuzzles on the corresponding tasks in 3DSRBench and compute the Pearson correlation coefficient [10] to measure consistency in performance across the two datasets. As shown in Tab. 5, the results reveal strong positive correlations: 0.93 for Height and 0.98 for Adjacency, both statistically significant (p < 0.01).

These findings suggest that LEGO-Puzzles not only offers high scalability and precise control in synthetic settings but also captures spatial reasoning patterns that generalize well to natural images. This validates its utility as a proxy for evaluating real-world spatial understanding.

- 6. Conclusion

We introduce LEGO-Puzzles, a novel benchmark specifically designed to evaluate spatial understanding, as well as single-step and multi-step sequential reasoning in MLLMs. Inspired by human cognitive patterns in LEGO construction, we create a dataset that includes over 1,100 carefully curated visual question-answering (VQA) samples across 11 distinct tasks, providing diverse scenarios to assess multimodal visual reasoning. We conduct comprehensive experiments with 20 advanced MLLMs, revealing substantial performance gaps compared to humans, particularly in extended sequential reasoning and the generation of spatially coherent visual outputs. These findings underscore the urgent need to enhance the spatial understanding and sequential reasoning capabilities of multimodal AI.

## References

- [1] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, et al. Pixtral 12b. arXiv preprint arXiv:2410.07073, 2024. 5
- [2] Anthropic. The claude 3 model family: Opus, sonnet, haiku.

2024. 5

- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei

- Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 1
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 5
- [5] Marc H Bornstein. Frames of mind: The theory of multiple intelligences. Journal of Aesthetic Education, 20(2), 1986. 1, 3
- [6] Yixiong Chen, Li Liu, and Chris Ding. X-iqe: explainable image quality evaluation for text-to-image generation with visual large language models. arXiv preprint arXiv:2305.10843, 2023. 7
- [7] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 5
- [8] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Chao Xu, Bin Wang, Xingjian Wei, Wei Li, Wenjian Zhang, Bo Zhang, Pinlong Cai, Licheng Wen, Xiangchao Yan, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. How far are we to gpt-4v? closing the gap to commercial multimodal models with opensource suites, 2024. 1, 2
- [9] Ethan Chern, Jiadi Su, Yan Ma, and Pengfei Liu. Anole: An open, autoregressive, native large multimodal models for interleaved image-text generation. arXiv preprint arXiv:2407.06135, 2024. 7
- [10] Israel Cohen, Yiteng Huang, Jingdong Chen, Jacob Benesty, Jacob Benesty, Jingdong Chen, Yiteng Huang, and Israel Cohen. Pearson correlation coefficient. Noise reduction in speech processing, pages 1–4, 2009. 9
- [11] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198–11201, 2024. 5
- [12] Jianguo Duan, Liwen Zhuang, Qinglei Zhang, Ying Zhou, and Jiyun Qin. Multimodal perception-fusion-control and human–robot collaboration in manufacturing: A review. The International Journal of Advanced Manufacturing Technology, 132(3):1071–1093, 2024. 1
- [13] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. ArXiv, abs/2306.13394, 2023. 1, 3
- [14] Junyao Gao, Yanchen Liu, Yanan Sun, Yinhao Tang, Yanhong Zeng, Kai Chen, and Cairong Zhao. Styleshot: A snap-

- shot on any style. arXiv preprint arXiv:2407.01414, 2024. 7
- [15] Junyao Gao, Yanan Sun, Fei Shen, Xin Jiang, Zhening Xing, Kai Chen, and Cairong Zhao. Faceshot: Bring any character into life. arXiv preprint arXiv:2503.00740, 2025. 1
- [16] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1
- [17] Xianda Guo, Ruijun Zhang, Yiqun Duan, Yuhang He, Chenming Zhang, Shuai Liu, and Long Chen. Drivemllm: A benchmark for spatial understanding with multimodal large language models in autonomous driving. arXiv preprint arXiv:2411.13112, 2024. 1
- [18] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718,

2021. 7

- [19] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 7
- [20] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017. 1
- [21] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 1
- [22] Jing Yu Koh, Daniel Fried, and Russ R Salakhutdinov. Generating images with multimodal language models. Advances in Neural Information Processing Systems, 36:21487–21506,

2023. 7

- [23] Hugo Lauren¸con, Andr´es Marafioti, Victor Sanh, and L´eo Tronchon. Building and better understanding visionlanguage models: insights and future directions. In Workshop on Responsibly Building the Next Generation of Multimodal Foundational Models, 2024. 5
- [24] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 3
- [25] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 5
- [26] Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vuli´c, and Furu Wei. Imagine while reasoning in space: Multimodal visualization-ofthought. arXiv preprint arXiv:2501.07542, 2025. 3
- [27] Xiaoqi Li, Mingxu Zhang, Yiran Geng, Haoran Geng, Yuxing Long, Yan Shen, Renrui Zhang, Jiaming Liu, and Hao

- Dong. Manipllm: Embodied multimodal large language model for object-centric robotic manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18061–18070, 2024. 1
- [28] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023. 1
- [29] Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14963–14973, 2023. 1
- [30] Yuan-Hong Liao, Rafid Mahmood, Sanja Fidler, and David Acuna. Reasoning paths with reference objects elicit quantitative spatial reasoning in large vision-language models. arXiv preprint arXiv:2409.09788, 2024. 1
- [31] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26689–26699, 2024. 5
- [32] Benlin Liu, Yuhao Dong, Yiqin Wang, Zixian Ma, Yansong Tang, Luming Tang, Yongming Rao, Wei-Chiu Ma, and Ranjay Krishna. Coarse correspondences boost spatial-temporal reasoning in multimodal language models. arXiv preprint arXiv:2408.00754, 2024. 3
- [33] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

2023. 1

- [34] Minqian Liu, Zhiyang Xu, Zihao Lin, Trevor Ashby, Joy Rimchala, Jiaxin Zhang, and Lifu Huang. Holistic evaluation for interleaved text-and-image generation. arXiv preprint arXiv:2406.14643, 2024. 7
- [35] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023. 1
- [36] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 1, 3
- [37] Wufei Ma, Haoyu Chen, Guofeng Zhang, Celso M de Melo, Alan Yuille, and Jieneng Chen. 3DSRBench: A comprehensive 3d spatial reasoning benchmark. arXiv preprint arXiv:2412.07825, 2024. 1, 3, 8
- [38] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pages 947–

952. IEEE, 2019. 1

- [39] Nora S Newcombe and Andrea Frick. Early education for spatial intelligence: Why, what, and how. Mind, Brain, and Education, 4(3):102–111, 2010. 3

- [40] OpenAI. Chatgpt. https://openai.com/blog/ chatgpt, 2023. 1
- [41] OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774,

2023. 1, 5

- [42] OpenBMB. Minicpm: Unveiling the potential of end-side large language models, 2024. 2
- [43] Simon Park, Abhishek Panigrahi, Yun Cheng, Dingli Yu, Anirudh Goyal, and Sanjeev Arora. Generalizing from simple to hard visual reasoning: Can we mitigate modality imbalance in vlms? arXiv preprint arXiv:2501.02669, 2025. 1, 3
- [44] Arijit Ray, Jiafei Duan, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A. Plummer, Ranjay Krishna, Kuo-Hao Zeng, and Kate Saenko. SAT: Spatial aptitude training for multimodal language models. arXiv preprint arXiv:2412.07755, 2024. 3
- [45] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 1
- [46] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, et al. Generative multimodal models are in-context learners. arXiv preprint arXiv:2312.13286, 2023. 7
- [47] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1, 5
- [48] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1
- [49] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 2, 4
- [50] Siyu Wang, Cailian Chen, Xinyi Le, Qimin Xu, Lei Xu, Yanzhou Zhang, and Jie Yang. CAD-GPT: Synthesising cad construction sequence with spatial reasoning-enhanced multimodal llms. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2025. 3
- [51] Wenhai Wang, Jiangwei Xie, ChuanYang Hu, Haoming Zou, Jianan Fan, Wenwen Tong, Yang Wen, Silei Wu, Hanming Deng, Zhiqi Li, et al. Drivemlm: Aligning multi-modal large language models with behavioral planning states for autonomous driving. arXiv preprint arXiv:2312.09245, 2023. 1
- [52] Xingrui Wang, Wufei Ma, Zhuowan Li, Adam Kortylewski, and Alan L Yuille. 3d-aware visual question answering about parts, poses and occlusions. Advances in Neural Information Processing Systems, 36:58717–58735, 2023. 1
- [53] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang,

- Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 5
- [54] Marcy Willard. What is sequential reasoning in childhood?,

2022. 3

- [55] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, et al. Deepseek-vl2: Mixture-ofexperts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024. 2, 5
- [56] Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. arXiv preprint arXiv:2412.14171, 2024. 1, 3
- [57] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 4
- [58] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 3

