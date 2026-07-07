## Creation-MMBench: Assessing Context-Aware Creative Intelligence in MLLMs

### Xinyu Fang1,2*, Zhijian Chen3*, Kai Lan3, Lixin Ma3, Shengyuan Ding2,4, Yingji Liang5, Xiangyu Zhao2,6, Farong Wen6, Zicheng Zhang2,6, Guofeng Zhang1, Haodong Duan2†, Kai Chen2†, Dahua Lin2,7 Zhejiang University1 Shanghai AI Laboratory2 Tongji University3 Nanjing University4 East China Normal University5 Shanghai Jiaotong University6 The Chinese University of Hong Kong7

# arXiv:2503.14478v2[cs.CV]19Mar2025

Assume you are an experienced screenwriter….. This is a scene photo of two people talking. Please follow the requirements below to write a conversation about two people talking face-to-face. 1. Role positioning:….. 2. Scene association:….

###### Q & A

Analytical Intelligence

Creative Multimodal Understanding

[Figure 1]

[Figure 2]

[Figure 3]

Creative Intelligence

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Partially: MLLM-Bench (2023) Only 20 Creation Samples

SEED-Bench (2023) MMBench (2024) MMMU (2024)

Professional Functional Writing

Scene: Office Meeting Room .. Dialogue: Alex: (enthusiastic, gesturing towards the laptop): “…” Mr. Tan (calm, analytical): (He leans back slightly...)

Scene: A modern office …. Dialogue: John: (Nods) Those are solid ideas. Let‘s discuss this further with the team. …

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Creative

[Figure 17]

[Figure 18]

Creative Challenge

Entirely:

Common Functional Writing

###### Good Fair

Practical Intelligence

Triarchic Theory of Intelligence

[Figure 19]

[Figure 20]

Creation-MMBench

[Figure 21]

[Figure 22]

[Figure 23]

Does the person to the left of the man appear to be sitting? (yes)

[Figure 24]

[Figure 25]

[Figure 26]

51 tasks 765 Creation Samples !

[Figure 27]

VAB (2024) VLABench (2024) EmbodiedBench (2025)

[Figure 28]

Literary Writing

[Figure 29]

[Figure 30]

Yes, the person to the left of the man appears to be sitting.

[Figure 31]

Practical Analytical

Easy Simple

Lack of Multimodal creative benchmarks

MLLMs have certain shortcomings in dealing with creative tasks in daily situations

Figure 1. Our Motivation for Creation-MMBench. The triarchic theory of intelligence divides intelligence into three forms. Current MLLM benchmarks have significant gaps in evaluating visual-creative intelligence compared to the other forms. Additionally, existing benchmarks feature simple questions that fail to assess model performance in real-life creative tasks. Therefore, we proposed CreationMMBench, which includes four categories, more creative and discriminative questions, and better evaluation of visual creative intelligence.

### Abstract

vides valuable insights for advancing MLLM creativity and establishes a foundation for future improvements in multimodal generative intelligence. Full data and evaluation code is released on https://github.com/opencompass/Creation-MMBench.

Creativity is a fundamental aspect of intelligence, involving the ability to generate novel and appropriate solutions across diverse contexts. While Large Language Models (LLMs) have been extensively evaluated for their creative capabilities, the assessment of Multimodal Large Language Models (MLLMs) in this domain remains largely unexplored. To address this gap, we introduce CreationMMBench, a multimodal benchmark specifically designed to evaluate the creative capabilities of MLLMs in realworld, image-based tasks. The benchmark comprises 765 test cases spanning 51 fine-grained tasks. To ensure rigorous evaluation, we define instance-specific evaluation criteria for each test case, guiding the assessment of both general response quality and factual consistency with visual inputs. Experimental results reveal that current open-source MLLMs significantly underperform compared to proprietary models in creative tasks. Furthermore, our analysis demonstrates that visual fine-tuning can negatively impact the base LLM’s creative abilities. Creation-MMBench pro-

### 1. Introduction

Creativity is the ability to generate novel and appropriate solutions to complex problems across various contexts[1, 17]. With the rapid advancement of Large Language Models (LLMs), numerous benchmarks have been proposed to assess their capabilities across different dimensions of intelligence, including comprehension, reasoning, and creativity [12, 18, 21, 22, 25]. These benchmarks have significantly contributed to a deeper understanding of LLM intelligence and have played a crucial role in driving their improvement. Meanwhile, Multimodal Large Language Models (MLLMs) [2, 4, 14] have also benefited from advancements in LLMs, achieving notable progress in perception, reasoning, and other cognitive abilities [3, 16, 32].

As a well-established theory in psychology, the Triarchic Theory of Intelligence [23] comprises three subtheories: the

1Equal Contribution. 2Corresponding Author.

Concentration, Planning, Problem Solving

analytical subtheory, the contextual subtheory, and the creative subtheory. The analytical subtheory primarily focuses on information processing and problem-solving skills based on domain-specific knowledge and can be assessed through various knowledge and reasoning benchmarks [10, 32]. The contextual subtheory, on the other hand, emphasizes practical intelligence in real-world scenarios and is typically evaluated using agent-based or embodied AI benchmarks [28, 33]. Despite the significance of the creative subtheory in intelligence, evaluations of MLLMs’ creative capabilities remain highly inadequate and lag significantly behind those conducted for LLMs [8, 18]. Moreover, constructing benchmarks to assess visual creativity presents inherent challenges. Cognitive science research suggests that creativity arises from a distributed cortical network involving the coordination of multiple brain regions. As illustrated in Fig. 2, creativity is closely associated with functions of the frontal lobe, such as concentration, planning, and problem-solving [11]. Within the context of MLLM evaluation, assessing creative capabilities requires benchmarks that encompass a broader range of fundamental cognitive abilities compared to those needed for other types of intelligence assessment [15, 31].

[Figure 32]

Language

Vision

Visual Creation Task

[Figure 33]

Figure 2. Brain regions related to creativity and their respective functions [6, 11].

to integrate contextual and visual information effectively. Using these tailored criteria, an MLLM-generated response is compared against a reference answer, and preferences are assigned accordingly. In addition to the preference obtained through pairwise comparison, we introduce a visual factuality score to evaluate whether the MLLM’s response aligns with key facts present in the visual input. This factual score is determined through unitary evaluation conducted by the GPT-4o judge model. Both Unitary Scoring and Pairwise Comparison offer a comprehensive assessment of creative quality and factual accuracy.

To address this significant gap, we introduce CreationMMBench, a novel benchmark specifically designed to assess the creative capabilities of MLLMs in image-based tasks across authentic real-world scenarios. The benchmark consists of 765 test cases spanning 51 fine-grained tasks, which are categorized into four major groups: Literary Writing, Common Functional Writing, Professional Functional Writing, and Creative Multimodal Understanding. Additionally, the benchmark is accompanied by rich context to facilitate comprehensive evaluation. In each task, an MLLM is provided with one or more images along with a detailed context specifying the assigned role, necessary background information, and clear task instructions. The model then follow the instruction and leverage the visual input to accomplish various creative tasks, such as composing artwork-inspired prose, developing structured lesson plans, or interpreting the conceptual foundations of advertisements. The approach enables a systematic assessment of MLLMs’ capacity to integrate visual perception with creative expression in contextually appropriate ways.

Based on Creation-MMBench, we conduct a comprehensive evaluation of mainstream MLLMs. The results indicate that current open-source MLLMs generally underperform compared to advanced proprietary models (e.g., Gemini2.0-Pro, GPT-4o) in terms of context-aware creativity. To further explore the impact of visual instruction tuning, we transformed Creation-MMBench into a text-only variant, Creation-MMBench-TO, by replacing image inputs with corresponding textual descriptions. The results reveal a negative effect of visual fine-tuning on the creative abilities of the base LLM, suggesting potential trade-offs introduced by multimodal adaptation.

In summary, our main contributions are three-fold:

- • Development of Creation-MMBench, a multimodal benchmark specifically designed to evaluate the creative capabilities of MLLMs. The benchmark incorporates a diverse set of image sources, spans a wide range of topics and task types across real-world scenarios, and features highquality, original human-written instructions.
- • Design of a robust evaluation methodology that includes carefully crafted instance-specific criteria for each test case, enabling assessment of both general response quality and visual-factual alignment in model-generated content.
- • A comprehensive assessment of various MLLMs on Creation-MMBench, providing detailed insights into their performance. The results highlight the current limitations of MLLMs in context-aware creativity and vision-based language generation, offering valuable guidance for future research and development.

Unlike ground-truth based evaluations, creative responses generated by models resist rule-based assessment methods. In our evaluation framework, we implement the widely adopted MLLM-as-a-Judge methodology, utilizing GPT-4o to assess the quality of model-generated responses. Given the diverse task types and stylistic variations across Creation-MMBench, a single-criterion evaluation model cannot reliably assess all tasks. To this end, we define instance-level evaluation criteria for each test case, ensuring that responses are assessed based on their ability

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

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Recipe infer and guide

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Clothing match design

Art inspired prose

Story continue

Museum guide creation UI design analysis

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Professional Functional Writing

Literary Writing

[Figure 64]

[Figure 65]

[Figure 66]

Creation MMBench

Scientific diagram understanding

Teaching plan

Historical story creation

Landscape to Poem

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Common Functional Writing

Creative Multimodal Understanding

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Short video scripts

Holiday card writing

Photograph Appreciation

Snapshot analysis

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Store experience review

Travel Itinerary Planning and Recommendations

Public welfare activity initiative

[Figure 105]

Document understanding

Advertisement Explanation

Letter of application

Meme explanation

Figure 3. Overview of Creation-MMBench. Contains four task categories, each category consists of multiple tasks, and the types of images are diverse. Only a few representative tasks of each category are shown here. Complete list of tasks is detailed in the Appendix A.

### 2. Related Work

ity, and elaboration in creative tasks [9].

Brainstorming techniques, commonly used to boost creativity, have been applied to evaluate LLMs’ creative abilities. RPGBench [30] uses role-playing games to assess creativity, and LiveIdeaBench [22] evaluates scientific creativity using single-keyword prompts, focusing on novelty, feasibility, fluency, and flexibility. Other benchmarks like LLM-Evolve [29] test problem-solving and adaptability, while SimulBench [13] evaluates creative simulations like acting as a Linux terminal. These benchmarks offer a comprehensive evaluation of LLMs’ creative and simulation capabilities, inspiring further exploration of MLLMs’ creative potential.

Evaluating Creative Capabilities of LLMs. To evaluate the creative writing capabilities of large language models (LLMs), several benchmark tests have been introduced. One example is the LLM Creative Story-Writing Benchmark [18], where 26 LLMs generate 500 short stories each, incorporating random elements, for a total of 13,000 stories. Six models then assess these stories based on 16 criteria related to character development, plot, and narrative structure. Another test [26] challenges models and humans to create stories based on specific prompts. These benchmarks assess not only the writing quality but also the diversity and complexity of the generated content.

Advancing the Evaluation of Creative Intelligence in MLLMs. The advancement of MLLMs has led to the development of various benchmarks to evaluate their intelligence. MMBench [15] covers 20 distinct ability dimensions, focusing on MLLMs’ general capability. MMMU [32] evaluates advanced perception and reasoning with domain-specific knowledge, featuring 11,500 multimodal questions across 6 disciplines. These benchmarks mainly focus on the analytical intelligence of MLLMs. For assessing MLLMs’ contextual intelligence, agent-

In addition to creative writing tasks, psychological tests commonly used to assess human creativity have also been adapted for evaluating LLMs. The Alternative Uses Test (AUT) evaluates a model’s ability to propose novel uses for everyday items within a time limit, as demonstrated in the assessment of GPT-3’s creativity [24]. Another benchmark introduces a small-scale test with a leaderboard to evaluate how four LLMs generate alternative uses for objects [20]. The Torrance Tests of Creative Thinking (TTCT) have also been applied to LLMs to assess fluency, flexibility, original-

Num of Creative Questions

Specific Role for each Questions

Visual Factuality Check

Criteria Level

multi-images task

Benchmarks

VisIT-Bench 65 benchmark ✓ ✗ ✓ MLLM-Bench 20 instance ✗ ✗ ✓ Touch-Stone 189 benchmark ✓ ✗ ✗ AlignMMbench 353 task ✗ ✗ ✗

Creation-MMBench 765 instance ✓ ✓ ✓

- Table 1. Comparison of Creation-MMBench with other partial-creation MLLM benchmarks.

based or embodied AI benchmarks are commonly used. VLABench [33] provides 100 categories of tasks to evaluate robotics’ language-conditioned manipulation ability, while EmbodiedBench [28] offers a comprehensive evaluation on models’ problem-solving ability with 1,128 tasks across 4 environments.

While the evaluation of MLLMs’ analytical and contextual intelligence has become relatively mature, the assessment of their creative intelligence remains insufficient. Existing partial-creation benchmarks, such as MLLMBench [7] and AlignMMBench [27], lack a systematic and comprehensive evaluation, often failing to assess models’ capabilities in complex, real-world scenarios. Furthermore, a dedicated benchmark designed specifically to evaluate MLLMs’ creativity has yet to be developed. Therefore, there is a pressing need for a comprehensive and practical benchmark to bridge this gap. Creation-MMBench aims to establish a dedicated benchmark for creative ability evaluation by incorporating a diverse set of real-world tasks, offering a novel perspective on evaluating MLLMs’ creative intelligence.

### 3. Creation-MMBench

This section describes the construction process of CreationMMBench, covering aspects such as task design, data collection, annotation, quality control, and evaluation. As shown in Fig. 3, the dataset includes diverse categories, reflecting the complexity and breadth of the tasks involved. Additionally, we introduce the data format and the indicators used to assess model capabilities.

#### 3.1. Benchmark construction

Task Design. We began with a brainstorming session to explore creative tasks in daily scenarios and designed a prototype task set encompassing both routine (e.g., writing common emails) and professional tasks (e.g., designing teaching plans). Leveraging a large language model, we then expanded this set to generate a diverse range of candidate tasks. Finally, through manual refinement and integration, a well-defined set of 51 tasks was established.

Task Categorization. We divided the 51 tasks into four

Figure 4. Evaluation Result of MLLMs w/o visual input.

main categories:

- 1. Literary Writing: Focus on literary creation (poetry, dialogues, stories, etc.)
- 2. Common Functional Writing: Focus on functional writing in daily life (social media writing, daily affairs inquiry, etc.)
- 3. Professional Functional Writing: Focus on functional writing and creative problem-solving in professional domains (analyzing design, developing lesson plans, etc.)
- 4. Creative Multimodal Understanding: Focus on the integration of visual understanding and creativity (formatted visual content analysis, image appreciation, etc.)

Data Composition. For each task, 15 carefully crafted test cases are collected. Each test case comprises two major components:

- • Visual Content: One or more images that contain the necessary information required to accomplish the test case.
- • Query: Include Role (the identity models need to play), Background (prior knowledge that is not duplicated by the visual content and is difficult to acquire, Instruction (operations that models need to perform), and Requirement (constraints or additional considerations).

All queries are organized into a complete format using a unified template and sent to MLLMs with visual content. Instance-specified criteria are defined to make the evaluation more reasonable. The criteria can be mainly divided into two groups:

- • General Subjective Criteria: Assess models’ expressive capability (structure, style, fluency), execution ability for queries (compliance with requirements, roles, and instructions), and deep reflection on visual content.
- • Visual Factuality Criteria: Assess models’ ability to perceive objective visual content and utilize visual information effectively.

Data Annotation and Quality Control. After task design and definition of data composition, we proceeded with data annotation (including questions and criteria) and quality control. To make the annotator easier to understand, we first built an example question for each task with detailed annotation, then asked volunteers to annotate 15 sam-

|[Figure 106]<br><br>Role: an interior space designer, good at finding existing housing problems according to the actual situation of users, and giving relevant opinions. Background: This picture is the floor plan of the house you have to design. This home is about to welcome a family of four with two children. Instruction: analyze the floor plan of the house Requirements:<br><br>1.Combine the characteristics of the occupants and your expertise to list the advantages and disadvantages of the house.<br>2.Provide suggestions for improving the house's shortcomings along with a preliminary renovation plan.<br><br><br>General Subjective Criteria:<br><br>1. Completeness of Analysis: 1.1 Does the response identify …?<br><br>1.2 Are the identified advantages and disadvantages reasonable?<br>2. Feasibility and Relevance of Suggestions: 2.1 Do the suggestions directly address the identified shortcomings of the house? 2.2 Are the suggestions practical, family-friendly, and … ? …<br><br><br>Visual Factuality Criteria:<br><br>1. Consistency with the Floor Plan and Requirements: 1.1 Does the response accurately identify key issues such as insufficient bedrooms, lack of children's spaces, and bathroom accessibility problems? …<br><br>OCR Knowledge–based Creation Spatial Understanding<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>Query|
|---|
|[Figure 111]<br><br>Role: a renowned writer known for rich imagination<br><br>and expertise in crafting short stories and micronovels, capable of drawing inspiration from images for literary creation.<br><br>Background: This picture was taken in London in 2015. Instruction: write a short story or micro-novel based on this image Requirements:<br><br>1.Develop plot ideas and engage in literary creation centered around the main content of the given image.<br>2.The setting, background, and character design of the story or novel can be freely crafted.<br>3.The work must have a clear theme and focus, …<br>4.The piece should include an … title.<br><br><br>General Subjective Criteria:<br><br>1. Completeness and Clarity: 1.1 Does the story create a compelling narrative…? 1.2 Is the writing smooth and coherent, allowing the plot to unfold logically and engagingly? 1.3 …<br>2. Imagination and Creativity: 2.1 Are the writing style and plot interesting enough to … ? 2.2 …<br>3. Clear and Profound Central Theme: 3.1 Does the story maintain a focused and coherent theme? 3.2 …<br><br><br>Visual Factuality Criteria:<br><br>1. Urban Interaction: 1.1 Does the story accurately represent the urban setting of London in 2015…? 1.2<br><br>Are the central characters in the image developed with realistic emotions and interactions, reflecting their implied roles (i.e. in business attire, …)?\n? ….<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>Query<br><br>[Figure 115]<br><br>Scene Perception Divergent Thinking Relation Understanding<br><br>|

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

- (a) Distribution of query lengths.

[Figure 116]

- (b) Roles in Creation-MMBench.

(c) Example Case of Creation-MMBench.

Figure 5. Statistics and Cases of Creation-MMBench. Compared to other widely used MLLM benchmarks, Creation-MMBench features a more comprehensive query design to capture abundant creative contexts. Diverse roles are introduced into the queries to stimulate MLLMs’ utilization of disciplinary and prior knowledge. As an MLLM benchmark, Creation-MMBench includes a rich variety of images to thoroughly evaluate multiple capabilities of MLLMs.

ple questions for each task with the example and guideline provided below:

assign numerical values to the pairwise comparison results: {A>>B = +2, A>B = +1, A=B = 0, A<B = -1, A<<B = -2}. For better interpretability, we multiply this average score by 50 and normalize it to the range of -100 to +100, forming a metric as Reward. To mitigate the inherent position bias in the MLLM-as-a-judge approach, we conduct a Dual Evaluation, swapping the response positions. The final result is obtained by averaging the outcomes of both evaluations. Detailed evaluation prompt is shown in Appendix B.

- 1. The visual content of questions should be semantic rich, and the query should not contain any explicit information in the visual content.
- 2. You are encouraged to formulate diverse queries within the task scope, like diverse roles and background settings, matching the visual content.
- 3. The ideal answer should be open-ended, creative, but the quality of the response can be assessed using criteria.
- 4. Ensure each requirement is clear and avoids redundancy. Keep the Visual Factuality Criteria concise and direct.

#### 3.2. Dataset Statistics

To better understand the composition of CreationMMBench, we conducted a statistical analysis.

Benchmark Comparison Tab. 1 shows the comparison result of Creation-MMBench and four widely used partialcreation MLLM benchmarks. As a dedicated benchmark for evaluating creativity, Creation-MMBench features a significantly richer set of creative questions and adopts a multiimage format. Each question is designed with specific roles to stimulate MLLMs’ creative capabilities. Unlike other benchmarks that apply the same evaluation criteria across an entire benchmark or task, Creation-MMBench customizes assessment criteria for each question, taking into account both subjective creativity and visual factuality. This tailored approach enables a more comprehensive evaluation of MLLMs’ creative abilities.

After initial labeling, we conducted cross-verification among volunteers, followed by expert review to ensure data quality.

Evaluation Strategy. We employ the MLLM-as-a-judge approach, which consists of two forms: Unitary Scoring and Pairwise Comparison. In Unitary Scoring, the judging model assigns a score between 1 and 10 to the response of the evaluated model based on the Visual Factuality Criteria. The Visual Factuality Score is the average score across all questions. In Pairwise Comparison, the evaluated model is designated as model A, while the baseline model (GPT4o-1120) is designated as model B. The judging model assesses the responses based on General Subjective Criteria and visual content, selecting from the set {A>>B, A>B, A=B, A<B, A<<B}. To facilitate further computation, we

Statistics and Cases Fig. 5 presents several statistics and cases of Creation-MMBench. As depicted in Fig. 5a, we analyzed the query length distributions of Creation-

|Model|Overall|LW|CFW<br><br>|PFW|CMU|OC Score|Avg Tokens|
|---|---|---|---|---|---|---|---|
| |VFS Reward|VFS Reward|VFS Reward|VFS Reward|VFS Reward| | |

Proprietary MLLMs

Gemini-2.0-pro-exp 8.53 4.48 8.66 -1.88 8.98 12.71 8.01 3.33 8.65 -8.06 73.4 718 GPT-4o-1120[Baseline] 8.72 0.00 8.86 0.00 8.93 0.00 8.26 0.00 9.38 0.00 72.0 497 Gemini-1.5-pro-002 8.41 -5.49 8.66 -6.04 8.59 -2.04 8.05 -4.82 8.75 -17.22 72.2 444 GPT-4.5-0227 8.54 -5.88 8.63 -4.38 8.76 -8.33 8.05 -5.88 9.29 -0.56 / 394 GPT-4o-mini 8.07 -13.56 8.30 -4.38 8.44 -15.28 7.50 -16.05 8.40 -12.78 64.1 436 Doubao-VL 8.38 -14.09 8.28 -19.17 9.01 -3.33 7.65 -18.72 8.77 -25.00 / 516 Claude-3.5-Sonnet 7.96 -15.46 8.44 -16.46 7.45 -21.57 7.98 -11.14 8.88 -9.44 70.6 336 Moonshot-v1-32k-vision 7.43 -20.58 7.30 -21.46 8.20 -8.80 6.91 -26.50 6.91 -36.11 / 485

Open-Source MLLMs

Qwen2.5-VL-72B-Instruct 8.33 -5.82 8.04 -10.83 8.91 4.44 7.68 -11.49 8.86 -11.94 76.1 553 InternVL2.5-78B-MPO 8.06 -12.55 8.22 -9.17 8.60 -5.00 7.45 -16.32 8.22 -27.78 77.0 461 InternVL2.5-8B-MPO 7.65 -15.10 8.09 -16.25 8.30 -3.80 6.80 -23.95 7.88 -19.44 70.3 548 InternVL2.5-78B 7.91 -16.43 8.05 -17.50 8.45 -7.69 7.26 -20.53 8.18 -28.33 75.2 473 Qwen2-VL-72B-instruct 7.87 -22.45 7.75 -24.58 8.17 -15.56 7.42 -26.84 8.43 -26.39 74.8 439 InternVL2.5-8B 7.38 -25.42 7.91 -23.33 7.95 -15.83 6.62 -33.95 7.45 -30.00 68.1 500 Qwen2.5-VL-7B-Instruct 7.55 -29.80 7.34 -39.38 8.40 -21.67 6.71 -33.25 7.78 -30.56 70.9 510 MiniCPM-o-2.6 7.49 -34.77 7.79 -35.42 7.95 -27.31 6.76 -40.88 8.08 -36.94 70.2 389 DeepSeek-VL2 7.24 -38.52 7.58 -33.75 7.58 -32.50 6.61 -44.02 7.81 -45.56 66.4 440 LLaVA-OneVision-72B 7.16 -39.87 7.26 -36.32 7.72 -30.61 6.43 -47.98 7.62 -46.37 68.0 315 LLaVA-OneVision-7B 6.75 -43.49 7.36 -43.54 7.27 -31.85 6.04 -50.53 6.82 -56.11 60.2 373 Qwen2-VL-7B-instruct 7.12 -43.76 6.99 -55.83 7.67 -36.30 6.57 -45.26 7.25 -45.28 67.1 456

- Table 2. Evaluation Result of MLLMs on Creation-MMBench. VFS stands for Visual Factuality Score. LW, CFW, PFW, and CMU stand for four categories in Creation-MMBench: Literary Writing, Common Functional Writing, Professional Functional Writing, and Creative Multimodal Understanding. OC Score represents the average score of the OpenVLM Leaderboard and mainly demonstrates the objective performance of the model. The token number is calculated with tiktoken GPT-4o-1120 tokenizer.

MMBench in comparison with two partial-creation benchmarks (MLLM-Bench, AlignMMBench) and two widely used general benchmarks (MM-Vet, MMBench). The results indicate that our benchmark features more comprehensive and complex query designs. The majority of queries exceed a length of 500 tokens, which facilitates models in capturing richer creative contexts. Fig. 5b illustrates the diversity of roles present in the queries (e.g., writer, artist, Michelin chef, etc.), reflecting the richness of the questions. As an MLLM benchmark, our dataset contains a total of

- 1,001 images spanning more than 25 different categories, with some questions incorporating up to 9 images. Fig. 5c displays the example cases in Creation-MMBench.

Vision Indispensability To verify the necessity of visual content in Creation-MMBench, we selected three MLLMs with varying capability levels (Gemini-1.5-Pro-

- 002, Qwen2-VL-72B-instruct, and MiniCPM-o-2.6) and examined their performance after removing visual input. In Fig. 4, we observe that when the visual information is removed, the same models exhibit significant declines in Reward. This finding verifies the necessity of visual content in evaluating model performance.

### 4. Experiment

Using Creation-MMBench, we evaluate various Multimodal Large Language Models (MLLMs), with a focus on image-based MLLMs that support multiple image inputs. Additionally, we adapted our benchmark into a text-only version (Creation-MMBench-TO) by replacing the visual inputs with corresponding textual descriptions and tested multiple Large Language Models (LLMs) to gain deeper insights into their creative capabilities. All evaluations were conducted based on VLMEvalKit [5], employing greedy decoding during inference with the maximum output tokens set to 4096.

#### 4.1. Main Results

We evaluated 20 current powerful MLLMs on CreationMMBench, results are shown on Tab. 2.

Proprietary MLLMs. Gemini-2.0-Pro performs similarly to GPT-4o, particularly in common functional writing, where it excels in producing content with a conversational tone and effectively integrates images. Its strong pre-existing knowledge also helps in professional functional writing tasks, but there is a slight gap in perception,

|VLM|Corresponding LLM|Text Input w. LLM|Text Input w. VLM<br><br>|Vision+Text Input w. VLM|
|---|---|---|---|---|
| | |VFS Reward<br><br>|VFS Reward|VFS Reward|
|GPT-4o-1120 Gemini-2.0-pro-exp Qwen2.5-VL-72B-Instruct Qwen2.5-VL-7B-Instruct MiniCPM-o-2.6 InternVL2.5-8B|GPT-4o-1120 Gemini-2.0-pro-exp Qwen2.5-72B-Instruct Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct InternLM2.5-7B-Chat|8.71 6.96 8.49 4.08 8.55 0.82 8.18 -19.18 8.18 -19.18 7.83 -22.19|8.71 6.96 8.49 4.08 8.51 -4.05 7.97 -27.50 7.78 -36.57 7.92 -28.73|8.72 0.36 8.53 4.48 8.33 -5.82 7.55 -29.80 7.49 -34.77 7.38 -25.42|

- Table 3. LLM performance on Creation-MMBench-TO and Visual Instruction Tuning Impact on VLM creation capability. The image descriptions provided by GPT-4o are general. For the proprietary models, we point to themselves as corresponding LLM and report the performance with image descriptions and questions.

especially in tasks like document and snapshot analysis. The smaller GPT-4o-mini outperforms proprietary models like Claude but struggles with professional functional writing due to its limited disciplinary knowledge. DoubaoVL stands out in common functional writing tasks, achieving the highest visual factuality score in this area.

Open-Source MLLMs. Among open-source MLLMs, Qwen2.5-VL-72B stands out, performing similarly to advanced proprietary models like Gemini-1.5-Pro and outperforming GPT-4o-mini across all four major categories. This highlights the potential of open-source models in visual creation. The InternVL series also shows strong performance across different model sizes, indicating potential advantages in data and training strategies. The mixed preference optimized (MPO) model demonstrates impressive results in smaller models, with particular strengths in creative multimodal understanding, suggesting that MPO can effectively guide models to better align with human preferences.

Category-level Evaluation Results. Across all four categories, professional functional writing shows relatively weaker performance, while common functional writing performs the best. This may be due to the greater difficulty of tasks in the former, which require extensive disciplinary knowledge and a deeper understanding of image content. These tasks are more complex and demand higher cognitive abilities. In contrast, common functional writing typically involves simpler, everyday tasks that require less advanced image understanding, making them easier to complete. In the Multimodal Content Understanding and Creation category, while all models show basic content understanding, their ability to generate more creative content is limited. This highlights the gap between the models’ objective interpretation abilities and their human-aligned visual creativity, further qualitative cases are provided in Appendix G.

Comparison of Model Performance on Objective Tasks and Creation-MMBench. To better compare the models’ objective performance with their visual creativity, we use the OC Score to represent the overall objective performance. As shown in Fig. 6, proprietary models perform well both in objective tasks and visual creativity. However,

Figure 6. Comparing OC Score and Creation-MMBench Reward. This figure shows the model performance on the OpenVLM Leaderboard and Creation-MMBench, highlighting a significant gap between objective performance and visual creativity in some open-source models.

some open-source models, despite showing strong objective performance, struggle with open-ended visual creativity tasks. These models tend to excel in tasks with definitive answers but fall short in generating creative, contextually relevant content. This discrepancy emphasizes the need for a more comprehensive evaluation approach, as traditional objective metrics alone may not fully capture a model’s creative abilities in complex, real-world scenarios.

#### 4.2. Evaluating LLMs on Creation-MMBench-TO

Current creation benchmarks for Large Language Models mostly focus on specific topics (e.g., LiveIdeaBench [22]), but fail to reveal their creation capability in multiple daily scenarios. To investigate it, we build Creation-MMBenchTO and GPT-4o was used to make the image descriptions with the prompt shown in Appendix E. As shown in Tab. 3, proprietary LLMs showed slightly better contextual creativity than open-source LLMs, though the gap was smaller than that between MLLMs. Large-scale language models performed better at understanding context and expressing ideas compared to smaller models. Additionally, the visual factuality score improved because GPT-4o’s image descriptions helped LLMs better interpret the image in comparison to MLLMs. Surprisingly, GPT-4o performed better in

|Judger|MLLM|Dual Eval<br><br>| | | |Single Eval| | | |
|---|---|---|---|---|---|---|---|---|---|
| | |MAE↓| |Cons.↑| |MAE↓| |Cons.↑| |
|Gemini-2P|Gemini Qwen MiniCPM|0.65 0.51 0.61|0.59<br><br>|82.83 91.00 86.14|86.67|0.78 0.67 0.69|0.72|74.75 80.00 81.19|78.67|
|Claude-3.5|Gemini Qwen MiniCPM|0.56 0.46 0.47<br><br>|0.50|89.90 92.00 89.90|90.60|0.61 0.59 0.57|0.59|83.84 85.00 86.87|85.23|
|GPT-4o|Gemini Qwen MiniCPM|0.53 0.42 0.53|0.50|92.08 96.08 88.24|92.13|0.57 0.46 0.59|0.54|89.11 91.18 86.27|88.85|

Question

[Figure 117]

Assume you are a master of meme interpretation with a good sense of humor. Here is an Internet meme picture. Please follow the requirements below to analyze it. 1.First, please describe the content of the given image in as much detail as possible. 2.Second, if…, please provide an introduction to that knowledge or background. …..

Assistant A Assistant B

[Figure 118]

[Figure 119]

- Table 4. The Alignment Between Different Evaluation Strategies and Human Preference.

21The image features a close-up of 12121 Gru, …... The background

The character is from the animated 0000 film “The Nightmare Before Christmas.” The meme references a common issue in Australia, where discarded cigarettes can ignite grass fires, especially during dry seasons. … The text humorously contrasts the typical reaction of people to the reckless behavior of an Australian who carelessly throws a cigarette into the grass. ….

knowledge is that Australia is prone to bushfires, especially during its hot and dry seasons. The meme humorously

suggests that an Australian who carelessly discards a cigarette into the grass would immediately realize the

visual creativity on Creation-MMBench-TO. This could be because the model can focus more on divergent thinking and creation with the help of descriptions, which may minimize the negative impact of the basic visual content on creativity.

potential consequences of their action …

Evaluation

Clear and Direct Description

Insufficient overall coherence

#### 4.3. Impact of Visual instruction tuning on creation capability of MLLM

Effectively Explain Background Knowledge

Incorrect Identification of Character

Identify Character in correct

Explanation Less Effective Clear but Lack engaging Quality

Existing research indicates that visual instruction tuning procedures may adversely affect the language encoder’s capacity to process and model text-only inputs. To further investigate this, we conducted three experiments under different settings, as shown in Tab. 3. The results indicate that the open-source MLLM, after visual instruction tuning, consistently performs worse compared to the corresponding LLM on Creation-MMBench-TO. This could be due to the instructions used during tuning being of similar length, which restricts the model’s ability to grasp detailed content in longer texts, resulting in a lower visual factuality score. The lack of creative data that combines images further contributes to a significant drop in the reward score. Although some proprietary models have shown stronger performance on Creation-MMBench, the performance gap of most MLLMs on Creation-MMBench-TO and CreationMMBench highlights the need for improvement in the perceptual capabilities of MLLMs.

Logic and Coherent Explanation

Figure 7. Qualitative study Case between InternVL-2.5-78B and Reference Answer (GPT4o-1120).

results indicate that for all judging models, Dual Evaluation outperforms Single Evaluation, verifying the necessity of Dual Evaluation. Among all the judging models, GPT-4o achieves the best performance in terms of MAE and Consistency, exhibiting the highest alignment with human preferences. Finally, we selected Dual Evaluation, and GPT-4o as the evaluation strategy for Creation-MMBench.

#### 4.5. Qualitative Study

To further explore the differences between models on Creation-MMBench, we conducted a detailed qualitative study by combining model responses with evaluations. As shown in Fig. 7, InternVL2.5 exhibited limitations in visual perception, particularly in accurately identifying characters due to insufficient latent knowledge. Additionally, InternVL2.5 showed certain weaknesses in the fluency and engagement of its language expression. In contrast, GPT4o was favored by the evaluation model, which provided a more balanced assessment. This highlights that open-source models still have considerable space for improvement, particularly in visual creativity tasks.

#### 4.4. Evaluation Strategy Selection

The goal of MLLM-as-a-judge is always to achieve a higher alignment with human preferences. Therefore, we randomly sampled a subset of questions (51 tasks × 2 questions) and recruited four volunteers to do the pairwise comparison. We selected three models (Gemini-1.5-pro-002, Qwen2-VL-72B, MiniCPM-o-2.6) as Model A, used the baseline model (GPT-4o-1120) as Model B, randomizing the responses’ position to avoid human biases. Details of the human evaluation process are provided in Appendix F.

### 5. Conclusion

We then selected three advanced MLLMs (Gemini-2.0Pro, Claude-3.5-Sonnet, GPT-4o) as judging models, and used MAE and Consistency as metrics to reflect the alignment degree. Tab. 4 presents the alignment degree between different evaluation strategies and human preferences. The

We present Creation-MMBench, a novel benchmark designed to assess the creative capabilities of MLLMs in realworld scenarios. The benchmark consists of 765 cases across 51 detailed tasks. For each case, we develop instance-specific criteria to evaluate both the subjective

quality of responses and visual-factual alignment. Additionally, we create a text-only version, Creation-MMBenchTO, by substituting image inputs with corresponding textual descriptions. Extensive experiments on both benchmarks enable a thorough assessment of mainstream MLLMs’ creative abilities and allow us to examine the negative impact of visual instruction tuning.

### References

- [1] Teresa M Amabile. Creativity in context: Update to the social psychology of creativity. Routledge, 2018. 1
- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 1
- [3] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330,

2024. 1

- [4] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 1
- [5] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pages 11198–11201, 2024. 6
- [6] Zhenni Gao, Xiaojin Liu, Delong Zhang, Ming Liu, and Ning Hao. Subcortical structures and visual divergent thinking: a resting-state functional mri analysis. Brain Structure and Function, 226(8):2617–2627, 2021. 2
- [7] Wentao Ge, Shunian Chen, Guiming Hardy Chen, Junying Chen, Zhihong Chen, Nuo Chen, Wenya Xie, Shuo Yan, Chenghao Zhu, Ziyue Lin, et al. Mllm-bench: evaluating multimodal llms with per-sample criteria. arXiv preprint arXiv:2311.13951, 2023. 4
- [8] Sikun Guo, Amir Hassan Shariatmadari, Guangzhi Xiong, Albert Huang, Eric Xie, Stefan Bekiranov, and Aidong Zhang. Ideabench: Benchmarking large language models for research idea generation. arXiv preprint arXiv:2411.02429,

2024. 2

- [9] Erik E Guzik, Christian Byrge, and Christian Gilde. The originality of machines: Ai takes the torrance test. Journal of Creativity, 33(3):100065, 2023. 3
- [10] Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark. arXiv preprint arXiv:2501.05444,

2025. 2

- [11] Kenneth M Heilman. Possible brain mechanisms of creativity. Archives of Clinical Neuropsychology, 31(4):285–296,

2016. 2

- [12] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. 1
- [13] Qi Jia, Xiang Yue, Tianyu Zheng, Jie Huang, and Bill Yuchen Lin. Simulbench: Evaluating language models with creative simulation tasks. arXiv preprint arXiv:2409.07641, 2024. 3

- [14] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

2023. 1

- [15] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 2, 3
- [16] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 1
- [17] Richard E Mayer. Fifty years of creativity research. Handbook of creativity, pages 449–460, 1999. 1
- [18] Lech Mazur. Llm creative story-writing benchmark. https : / / github . com / lechmazur / writing,

2025. 1, 2, 3

- [19] Yuxuan Qiao, Haodong Duan, Xinyu Fang, Junming Yang, Lin Chen, Songyang Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. Prism: A framework for decoupling and assessing the capabilities of vlms. Advances in Neural Information Processing Systems, 37:111863–111898, 2025. 2
- [20] Abdullah Al Rabeyah, Fabr´ıcio G´oes, Marco Volpe, and Talles Medeiros. Do llms agree on the creativity evaluation of alternative uses? arXiv preprint arXiv:2411.15560, 2024. 3
- [21] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level googleproof q&a benchmark. In First Conference on Language Modeling, 2024. 1
- [22] Kai Ruan, Xuan Wang, Jixiang Hong, and Hao Sun. Liveideabench: Evaluating llms’ scientific creativity and idea generation with minimal context. arXiv preprint arXiv:2412.17596, 2024. 1, 3, 7
- [23] Robert J Sternberg. The triarchic theory of intelligence.

1997. 1

- [24] Claire Stevenson, Iris Smal, Matthijs Baas, Raoul Grasman, and Han van der Maas. Putting gpt-3’s creativity to the (alternative uses) test. arXiv preprint arXiv:2206.08932, 2022. 3
- [25] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track,

2024. 1

- [26] Paul Williams and Carlos G´omez-Rodr´ıguez. A confederacy of models: A comprehensive evaluation of llms on creative writing. In UniSC Research Conference. University of the Sunshine Coast, 2024. 3
- [27] Yuhang Wu, Wenmeng Yu, Yean Cheng, Yan Wang, Xiaohan Zhang, Jiazheng Xu, Ming Ding, and Yuxiao Dong. Alignmmbench: Evaluating chinese multimodal alignment in large vision-language models. arXiv preprint arXiv:2406.09295, 2024. 4

- [28] Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, et al. Embodiedbench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents. arXiv preprint arXiv:2502.09560, 2025. 2, 4
- [29] Jiaxuan You, Mingjie Liu, Shrimai Prabhumoye, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. Llmevolve: Evaluation for llm’s evolving capability on benchmarks. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 16937– 16942, 2024. 3
- [30] Pengfei Yu, Dongming Shen, Silin Meng, Jaewon Lee, Weisu Yin, Andrea Yaoyun Cui, Zhenlin Xu, Yi Zhu, Xingjian Shi, Mu Li, et al. Rpgbench: Evaluating large language models as role-playing game engines. arXiv preprint arXiv:2502.00595, 2025. 3
- [31] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 2
- [32] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 1, 2, 3
- [33] Shiduo Zhang, Zhe Xu, Peiju Liu, Xiaopeng Yu, Yuan Li, Qinghui Gao, Zhaoye Fei, Zhangyue Yin, Zuxuan Wu, YuGang Jiang, et al. Vlabench: A large-scale benchmark for language-conditioned robotics manipulation with longhorizon reasoning tasks. arXiv preprint arXiv:2412.18194,

2024. 2, 4

- [34] Zicheng Zhang, Xiangyu Zhao, Xinyu Fang, Chunyi Li, Xiaohong Liu, Xiongkuo Min, Haodong Duan, Kai Chen, and Guangtao Zhai. Redundancy principles for mllms benchmarks, 2025. 1

## Creation-MMBench: Assessing Context-Aware Creative Intelligence in MLLMs Supplementary Material

### A Overview of Tasks in Creation-MMBench

Creation-MMBench consists of four main categories and 51 tasks, as shown in Fig. 8. The Literary Writing category includes 8 tasks, focusing on visual literary creation using images such as photographs, illustrations, and paintings. The Common Functional Writing category comprises 18 tasks, addressing everyday functional creation across various genres and image types. The Professional Functional Writing category contains 19 tasks, focusing on creation tasks that require specific domain knowledge. Finally, the Creative Multimodal Understanding category includes 6 tasks, which involve interpreting implied content from images with rich textual information. For each category, we provide four examples, as illustrated in Fig. 14 - Fig. 29.

### B Query and Judge Prompt Template for Creation-MMBench

#### B.1 Query Template

For each test case, the query is formatted using the template shown in Fig. 30. In Creation-MMBench-TO, we replace visual content with generated descriptions, as no images are provided to the LLM.

#### B.2 Judge Template

For pairwise comparison, the General Subjective Criteria are essential for a fair assessment. Visual content helps the judge model better understand the predictions and prevents arbitrary conclusions based solely on linguistic strengths. As shown in Fig. 33, the predictions from different models are presented side by side with the criteria to minimize position bias, with instructions also provided to the judging model. Although changing the hypothetical positions helps reduce positional bias, dual evaluation remains necessary. The format restrictions for evaluating model responses facilitate the extraction of the final verdict through regular matching methods.

For Unitary Scoring, we provide Visual Factuality Criteria along with the model’s predictions, the reference answer, and the query, as outlined in Fig. 34. In test cases with a corresponding GroundTruth, this is included to ensure accurate judgment. Each criterion includes several main points, which may be further subdivided into subpoints. The evaluation model scores based on the completeness of these points, with a total score of 10.

### C Main Experiment Analysis on Win Rate

In Creation-MMBench, we adopt the MLLM-as-a-judge approach and introduce two metrics, Visual Factuality Score and Reward, to assess the creative capabilities of MLLMs. In this section, we propose a new metric, Win Rate, to provide a more comprehensive evaluation of MLLMs’ performance.

#### C.1 Win Rate Definition

Win Rate is defined as the proportion of instances in which the response generated by the evaluated model surpasses that of the baseline model in the Pairwise Comparison.

#### C.2 Main results on Win Rate

Tab. 5 presents the Win Rate and detailed judgment counts of MLLMs on Creation-MMBench. Among Proprietary MLLMs, Gemini-2.0-pro-exp demonstrates the best performance in terms of Win Rate, exhibiting the highest number of Much Better and Better cases. In contrast, GPT-4omini performs the worst, with only 53 Better cases. Among Open-Source MLLMs, Qwen2.5-VL-72B-Instruct achieves the best performance, with only 42 Much Worse cases. However, most models perform poorly, lacking any Much Better cases. A noticeable performance gap remains between Open-Source and Proprietary MLLMs in terms of Win Rate.

### D Advanced Analysis of Creation-MMBench

#### D.1 Redundancy Analysis

Following the [34], we compute the correlation coefficients between the model evaluation results of CreationMMBench and other representative objective benchmarks to investigate the redundancy of Creation-MMBench. Fig. 9 presents the Spearman’s Rank Correlation Coefficient (SRCC) and the coefficient of determination (R²) between the benchmarks. As shown in the figure, CreationMMBench exhibits a low correlation with MathVista, AI2D, and OCRBench in both SRCC and R². This is likely because these three benchmarks primarily assess objective capabilities such as mathematical reasoning, information extraction, and simple logical inference, with most queries presented in multiple-choice format—an evaluation focus that differs significantly from that of Creation-MMBench.

In contrast, MMMU and MM-Vet show a certain degree of correlation with Creation-MMBench. This may be attributed to the fact that both benchmarks incorporate a portion of creativity-oriented testing, such as the Art & Design

[Figure 120]

Figure 8. Overview of Creation-MMBench Complete Task. Contains four task categories, each category consists of multiple tasks.

section in MMMU-Val and the summarization task in MMVet. In general, Creation-MMBench shows low redundancy with existing MLLM Benchmarks, which reflects the novelty and uniqueness of our benchmark.

#### D.2 Other Statistics

Fig. 10 presents supplementary statistics for CreationMMBench. As shown in Fig. 10a, we compare the reference answer lengths of Creation-MMBench with four widely used MLLM benchmarks. It is evident that our benchmark exhibits a significantly higher proportion of long answers exceeding 1,500 tokens, which reflects the greater complexity of our tasks. Fig. 10b illustrates the richness of instructions within Creation-MMBench, reflecting the diversity of tasks. The analysis of image categories in Fig. 10c demonstrates the rich visual content incorporated in our benchmark. This diversity ensures a comprehensive evaluation of the model’s perceptual capabilities, further solidifying Creation-MMBench as a rigorous MLLM benchmark.

### E Query-Specific Experiments on CreationMMBench-TO

For Creation-MMBench-TO, the instructions for visual content description are crucial as they are designed to fully

stimulate the model to interpret the content of the image as detailed and rich as possible. To avoid the loss of some fine-grained content caused by generic visual descriptions, which could affect the performance of LLM’s creative ability, we additionally used Query-Specific Instruction generated by GPT-4o to guide the visual description [19].

As shown in Fig. 31, Generic instruction is a standardized, universal instruction aimed at extracting and describing the basic elements present in an image. Query-specific instruction is a combination of generic instruction and incremental instruction that directs the VLM to provide a detailed account of the visual information relevant to the question. The incremental instruction is crafted by the GPT-4o given the text-only question and the few-shot prompt template shown in Fig. 32.

Results on Tab. 6 reveal that query-specific descriptions can help LLMs gain a better understanding of visual content, resulting in a higher Visual Factuality Score and Reward. However, GPT-4o exhibits an inverse trend, which may be because fine-grained descriptions can mislead the attention of the models and may generate too much detailed creative content that does not fully meet the criteria.

Model VFS Reward WR MB Better Tie Worse MW Fail Proprietary MLLMs

Gemini-2.0-pro-exp 8.53 4.48 26.75% 9 400 898 163 59 1 GPT-4o-1120 8.72 0.00 - - - - - - Gemini-1.5-pro-002 8.41 -5.49 11.37% 6 168 1032 300 24 0 GPT-4.5-0227 8.54 -5.88 5.36% 7 75 1186 255 7 0 GPT-4o-mini 8.07 -13.56 3.79% 5 53 1022 422 28 0 Doubao-VL 8.38 -14.09 9.22% 4 137 850 500 38 1 Claude-3.5-Sonnet 7.96 -15.46 12.55% 4 188 843 321 174 0 Moonshot-v1-32k-vision 7.43 -20.58 6.09% 1 92 822 500 111 4

Open-Source MLLMs Qwen2.5-VL-72B-Instruct 8.33 -5.82 13.2% 6 196 984 302 42 0 InternVL2.5-78B-MPO 8.06 -12.55 8.76% 6 128 917 434 45 0 InternVL2.5-8B-MPO 7.65 -15.10 10.33% 0 158 843 438 91 0 InternVL2.5-78B 7.91 -16.43 7.25% 4 107 863 494 62 0 Qwen2-VL-72B-instruct 7.87 -22.45 4.64% 0 71 764 632 63 0 InternVL2.5-8B 7.38 -25.42 5.62% 2 84 699 624 121 0 Qwen2.5-VL-7B-Instruct 7.55 -29.80 4.25% 0 65 620 713 132 0 MiniCPM-o-2.6 7.49 -34.77 2.29% 2 33 545 799 151 0 DeepSeek-VL2 7.24 -38.52 1.77% 0 27 504 791 207 1 LLaVA-OneVision-72B 7.16 -39.87 1.72% 0 26 448 842 194 20 LLaVA-OneVision-7B 6.75 -43.49 1.96% 1 29 411 816 273 0 Qwen2-VL-7B-instruct 7.12 -43.76 1.57% 0 24 402 845 259 0

- Table 5. Win Rate Result of MLLMs on Creation-MMBench. WR, MB, MW stands for Win Rate, Much Better and Much Worse

|LLM<br><br>|Generic<br><br>|Query-Specific|
|---|---|---|
| |VFS Reward|VFS Reward|
|GPT-4o-1120 Qwen2.5-72B-Instruct InternLM2.5-7B-Chat|8.71 6.96 8.55 0.82 7.83 -22.19|8.88 3.33 8.82 4.80 8.33 -15.29|

- Table 6. Comparison on Generic Descriptions and QuerySpecific Descriptions on Creation-MMBench-TO.

#### F.1 The process of Human Pairwise Comparison

For human evaluation, We sampled two questions from each task in Creation-MMBench to construct a set of 102 questions. Four volunteers were recruited to perform pairwise comparisons on this question set. Fig. 11 illustrates the user interface used by human evaluators for this task. To mitigate potential bias, we randomized both the order of the questions and the positions of Model A (Gemini-1.5-pro002, Qwen2-VL-72B, MiniCPM-o-2.6) and Model B (baseline, i.e. GPT-4o-1120)’s responses. Evaluators were provided with the corresponding visual content, related questions, and assessment criteria to compare the quality of the responses presented on the left and right. Their selections were recorded to generate preference results.

### F Human Alignment

In this section, we provide a detailed examination of Human Alignment, covering the process of pairwise comparison conducted by human evaluators, the definition of the evaluation metrics, and the comprehensive results of ModelHuman and Human-Human alignment.

#### F.2 The Definition of MAE and Consistency

Eq (1) and (2) present the metrics used to evaluate the degree of alignment, specifically MAE and Consistency. In

###### SRCC Correlation Heatmap

###### R2 Correlation Heatmap

1.0

1.0

[Figure 121]

[Figure 122]

1.00 0.30 0.46 0.68 0.83 0.77 0.46 0.26 0.62 0.75

1.00 0.09 0.20 0.29 0.75 0.69 0.17 0.03 0.54 0.53

MathVista

MathVista

0.30 1.00 0.59 0.10 0.66 0.39 0.78 0.84 0.68 0.75

0.09 1.00 0.51 0.14 0.39 0.11 0.70 0.75 0.31 0.60

MMMU-VAL

MMMU-VAL

0.8

0.8

0.46 0.59 1.00 0.53 0.72 0.57 0.83 0.46 0.65 0.85

0.20 0.51 1.00 0.56 0.37 0.26 0.79 0.32 0.34 0.79

HallusionBench

HallusionBench

0.68 0.10 0.53 1.00 0.50 0.64 0.41 0.20 0.36 0.61

0.29 0.14 0.56 1.00 0.21 0.29 0.44 0.14 0.16 0.58

OCRBench

OCRBench

0.6

0.6

0.83 0.66 0.72 0.50 1.00 0.80 0.74 0.51 0.91 0.93

0.75 0.39 0.37 0.21 1.00 0.72 0.37 0.19 0.86 0.74

MMStar

MMStar

0.77 0.39 0.57 0.64 0.80 1.00 0.48 0.19 0.82 0.74

0.69 0.11 0.26 0.29 0.72 1.00 0.18 0.01 0.83 0.56

AI2D

AI2D

0.4

0.4

0.46 0.78 0.83 0.41 0.74 0.48 1.00 0.79 0.67 0.88

0.17 0.70 0.79 0.44 0.37 0.18 1.00 0.67 0.28 0.78

MMVet

MMVet

0.26 0.84 0.46 0.20 0.51 0.19 0.79 1.00 0.43 0.64

0.03 0.75 0.32 0.14 0.19 0.01 0.67 1.00 0.09 0.40

Creation-MMBench

Creation-MMBench

0.2

0.2

0.62 0.68 0.65 0.36 0.91 0.82 0.67 0.43 1.00 0.84

0.54 0.31 0.34 0.16 0.86 0.83 0.28 0.09 1.00 0.64

MMBench-V11-TEST

MMBench-V11-TEST

0.75 0.75 0.85 0.61 0.93 0.74 0.88 0.64 0.84 1.00

0.53 0.60 0.79 0.58 0.74 0.56 0.78 0.40 0.64 1.00

OC Score

OC Score

0.0

0.0

MathVistaMMMU-VALHallusionBenchOCRBenchMMStar AI2DCreation-MMBenchMMVetMMBench-V11-TESTOCScore

MathVistaMMMU-VALHallusionBenchOCRBenchMMStar AI2DCreation-MMBenchMMVetMMBench-V11-TESTOCScore

(a) Distribution of query lengths.

(b) Roles in Creation-MMBench.

Figure 9. Redundancy Analysis of Creation-MMBench with other widely used MLLM Benchmarks.

these equations, J represents the pairwise comparison results from a specific judging model or human evaluator, while P denotes the corresponding reference value (average of human ratings).

1 n

MAE =

n

|Ji − Pi| (1)

i=1

Consistency =

n

1 n

i=1

1, if |Ji − Pi| ≤ 1 0, otherwise

(2)

#### F.3 Full Results

We conducted experiments to study both Model-Human Alignment and Human-Human Alignment. For the former, J refers to the judging model’s comparison result, while P represents the average human preference. For the latter, J refers to the comparison result of an individual human, with P being the average preference of the remaining humans. Tab. 7 presents the detailed alignment results.

It can be observed that for all judging models, MLLMas-a-judge outperforms LLM-as-a-judge in terms of MAE and Consistency. This may be because the incorporation of visual content allows the judging models to conduct a more comprehensive evaluation. Regarding Human-Human Alignment, human preferences are not highly consistent with one another, which reflects the subjective nature of our benchmark.

### G Category Qualitative Case Study

We conducted a qualitative analysis of the common situations that occur in some task categories. Fig. 12 mainly

focuses on the category of Professional Functional Writing. It can be significantly observed that Qwen2.5-VL misjudged the swimlane diagram as a data flow diagram due to insufficient understanding of the domain-specific knowledge, leading to subsequent errors in diagram analysis. In contrast, GPT-4o-1120 effectively avoided this mistake, and its overall language is more professional and structured, demonstrating a more accurate and detailed explanation of the diagram, thus gaining the preference of the judge model. This example also reflects the important role of specific disciplinary knowledge and a detailed understanding of image content in this category of tasks.

For Creative Multimodal Understanding tasks, as shown in Fig. 13, both models gain full scores in visual factuality and exhibit similar performance in basic visual content understanding and information extraction. However, GPT-4o-1120 gives a more comprehensive plan with clear scheduling and reasonable arrangement, thus winning the preference of the judging model.

- (a) Distribution of reference answers lengths.

[Figure 123]

- (b) Instructions in Creation-MMBench.

|[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>Top 15 Image Categories in Creation-MMBench<br><br>Animation & Comics 9% People 9% Product 8% Architecture 8% Event 8%<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>|
|---|
|Education 7% Art 5% Food & Beverage 5% Nature 5% Science & Technology 4%<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>|
|News & Newspaper 4% UI 4% Interior 3% History & Culture 3% Statistical Data 3%<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]|

(c) Top 15 Image Categories in Creation-MMBench.

###### Figure 10. Other Statistics of Creation-MMBench.

[Figure 169]

###### Figure 11. The Process of Human Pairwise Comparison.

|Judging Method|Judging Model/Human|MLLM|Dual Evaluation<br><br>| | | |Non-Dual Evaluation| | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | |MAE↓| |Consistency↑<br><br>| |MAE↓| |Consistency↑| |
|LLM-as-a-judge|Gemini-2.0-Pro|Gemini-1.5-pro-002 Qwen2-VL-72B MiniCPM-o-2.6|0.67 0.59 0.61|0.62|83.17 84.16 85.15|84.16|0.75 0.65 0.67|0.69|77.23 78.22 82.18<br><br>|79.21|
| |GPT-4o-mini|Gemini-1.5-pro-002 Qwen2-VL-72B MiniCPM-o-2.6<br><br>|0.67 0.59 0.52|0.59|83.17 85.29 90.20|86.23|0.79 0.67 0.66|0.71|74.26 76.47 81.37|77.38|
| |Claude-3.5-Sonnet|Gemini-1.5-pro-002 Qwen2-VL-72B MiniCPM-o-2.6|0.63 0.46 0.46|0.52<br><br>|89.11 94.12 92.16|91.80|0.73 0.58 0.58|0.63|78.22 82.35 85.29|81.97|
| |GPT-4o|Gemini-1.5-pro-002 Qwen2-VL-72B MiniCPM-o-2.6|0.56 0.46 0.51|0.51|93.07 92.16 89.22<br><br>|91.48|0.56 0.54 0.58|0.56|90.10 87.25 85.29|87.54|
|MLLM-as-a-judge|Gemini-2.0-Pro|Gemini-1.5-pro-002 Qwen2-VL-72B MiniCPM-o-2.6|0.65 0.51 0.61|0.59|82.83 91.00 86.14|86.67|0.78 0.67 0.69|0.72|74.75 80.00 81.19|78.67<br><br>|
| |GPT-4o-mini|Gemini-1.5-pro-002 Qwen2-VL-72B MiniCPM-o-2.6|0.64 0.53 0.49|0.55<br><br>|84.16 93.14 91.18|89.51|0.71 0.65 0.61|0.66|76.24 82.35 82.35|80.33|
| |Claude-3.5-Sonnet|Gemini-1.5-pro-002 Qwen2-VL-72B MiniCPM-o-2.6|0.56 0.46 0.47|0.50<br><br>|89.90 92.00 89.90|90.60|0.61 0.59 0.57|0.59|83.84 85.00 86.87|85.23|
| |GPT-4o|Gemini-1.5-pro-002 Qwen2-VL-72B MiniCPM-o-2.6|0.53 0.42 0.53|0.50|92.08 96.08 88.24|92.13|0.57 0.46 0.59|0.54|89.11 91.18 86.27|88.85<br><br>|
|Human-as-a-judge|H1|Gemini-1.5-pro-002 Qwen2-VL-72B MiniCPM-o-2.6|/ / /|/|/ / /|/|0.65 0.60 0.66<br><br>|0.64|84.16 90.20 87.25|87.21|
| |H2|Gemini-1.5-pro-002 Qwen2-VL-72B MiniCPM-o-2.6|/ / /<br><br>|/|/ / /|/|0.82 0.72 0.73|0.75|74.26 82.35 79.41|78.69|
| |H3<br><br>|Gemini-1.5-pro-002 Qwen2-VL-72B MiniCPM-o-2.6|/ / /|/|/ / /|/|0.74 0.62 0.72|0.68|76.24 80.39 90.20|82.30|
| |H4|Gemini-1.5-pro-002 Qwen2-VL-72B MiniCPM-o-2.6|/ / /|/|/ / /|/|0.64 0.61 0.65|0.63|87.13 89.22 87.25|87.87|

###### Table 7. The Results of Model-Human Alignment and Human-Human Alignment.

Question

[Figure 170]

Assume you are an experienced software development engineer,…. This is a software engineering diagram that you encounter in your work. Please follow the requirements below to draft a document to ensure that … can understand the type of diagram, the business process, and the division of work after reading it.

- 1. Ensure the document is concise and clear.
- 2. The document should clearly specify the type of diagram, its purpose, and the stage in software engineering where it is typically used.

Assistant B

Assistant A

[Figure 171]

1. The diagram shown is a Swimlane Diagram, a type of …. It visually represents a business process by dividing it

[Figure 172]

- 1. … The diagram is a Data Flow Diagram (DFD).
- 2. Business Process Illustrated: This specific DFD

into lanes, where each lane corresponds to a specific role, …

illustrates the credit approval process for a sales transaction. Here's a step-by-step breakdown of the process: 1. Customer buys product: The process begins

- 2. Purpose of the Diagram: The purpose of this Swimlane Diagram is to: Illustrate the flow of a business process: It shows how tasks and decisions move between different roles or departments. Clarify responsibilities: Each lane represents a …
- 3. Stage in Software Engineering: This diagram is typically used during the Requirements Analysis or System Design stage of software engineering. It helps: Understand the current ….
- 4. Explanation of the Business Process: The diagram represents a credit approval process for a customer purchasing a product. Below is a step-by-step explanation: Roles/Departments …

when a customer decides to buy a product. 2. Credit form: The customer fills out a credit form, which is then sent to the sales department. …

3. Division of Work: The DFD also shows how the work is divided among different departments:

- - **Customer:** Initiates the process by buying a product and filling out a credit form.
- - **Sales Department:** Handles the sales call, order entry, and order form creation. … By following this DFD, each department knows its role and responsibilities in the credit approval process, ensuring a smooth and efficient workflow.

6. Key Takeaways: This Swimlane Diagram provides a clear, visual representation of the credit approval process. By following this document, even a beginner can understand the purpose, structure, and workflow depicted in the diagram.

Evaluation： A Much Better Than B

Assistant A correctly identifies the diagram as a Swimlane Diagram, which is accurate based on the image. The response is clear and concise, making it accessible to beginners. It effectively explains the purpose of the diagram, highlighting its role in illustrating business processes, clarifying responsibilities, and identifying dependencies. The document also accurately places the diagram in the Requirements Analysis or System Design stage of software engineering, which is appropriate. The explanation of the business process is detailed and aligns well with the image, covering the roles involved and the flow of tasks. The language is professional and structured, suitable for both technical and non-technical readers.

Assistant B Evaluation:\n\nAssistant B incorrectly identifies the diagram as a Data Flow Diagram (DFD), which does not align with the image. This misidentification affects the clarity and accuracy of the response. While the explanation of the business process is detailed, it is based on the incorrect assumption of the diagram type. The response does cover the division of work and provides a step-by-step breakdown, but the initial error in identifying the diagram type undermines the overall quality. The language is clear and professional, but the misidentification impacts the completeness and accuracy of the document.

- Figure 12. Qualitative Case in Professional Functional Writing. This case comes from Software Engineering Diagram Explanation Task, Assistant A is GPT-4o-1120, assistant B is Qwen2.5-VL-72B.

Question

[Figure 173]

Assume you are an experienced and wellreputed tour guide can flexibly customize travel plans … This is a map marked with popular attractions. The travelers you are about to host are art and architecture enthusiasts who are eager to explore Gaudí’s masterpieces.….. … design a 2-day travel plan based on the provided map and the travelers' preferences.

1. The plan should include the detailed schedule ….. 2. It is necessary to select some representative attractions …..

Assistant B

Assistant A

[Figure 174]

[Figure 175]

### 2-Day Travel Plan for Art and Architecture Enthusiasts in Barcelona. This 2-day itinerary is designed

### 2-Day Travel Plan for Art and Architecture Enthusiasts in Barcelona

to immerse travelers in the architectural wonders of Antoni Gaudí, the vibrant street life of Barcelona, and the rich Catalan culture. The plan prioritizes efficiency, cultural depth, and flexibility. Day 1: Gaudí’s Masterpieces and Catalan Culture Morning: La Sagrada Familia (Stop 4 on the map) … Why Visit: La Sagrada Familia is Gaudí’s most iconic work ….Special Considerations: Book tickets online in advance to avoid long queues. Opt for a guided tour to understand the symbolism behind the design. Day 2: Las Ramblas, Gothic Quarter, and La Pedrera Morning: Las Ramblas and Gothic Quarter (Stops 1 and 2 on the map) … Additional Notes: Transportation: …. Flexibility: … Souvenirs: Pick up unique gifts like Gaudí-inspired ceramics or Catalan wines from local shops.

- #### Day 1: Gaudí's Masterpieces and Gothic Quarter

**Morning:**: **La Sagrada Familia (Attraction 4)** Reason for Selection:** One of Gaudí's most iconic works…

**Afternoon:**: **Lunch in the Gothic Quarter**

**Reason for Selection:** The Gothic Quarter is known for its narrow medieval streets and is a great place to enjoy local Catalan cuisine.

- #### Day 2: Modernist Architecture and Local Experiences: **Morning:**: **Park Güell (Attraction 3)**

**Reason for Selection:** Another iconic Gaudí landmark, Park Güell offers a unique blend of architecture and nature. ### Summary This 2-day travel plan is designed to cater to the travelers' interests in art, architecture, and local …

Evaluation： A Better Than B

Assistant A’s … Score: 10/10 Assistant A Evaluation: The plan is clear and well-structured, with specific times and activities outlined for each day. It … providing essential details about each attraction.\n\n

Assistant B’s … Score: 10/10 Assistant B Evaluation: The plan is clear, with a detailed schedule and descriptions of each attraction. … some details are less comprehensive than Assistant A's.\n Background information is generally concise but could be more detailed in some areas.

2. Efficiency and Comfort: The itinerary is efficiently planned, with attractions grouped by proximity to minimize travel time. Sufficient buffer time is included between activities, ensuring a relaxed pace. … Assistant A provides a more comprehensive and well-organized itinerary, with clear scheduling and detailed background information. The plan is efficient and diverse, aligning closely with the travelers' interests.

2. Efficiency and Comfort: … logically organized, with attractions planned to minimize travel time. However, …. might not be optimal given their locations … Assistant B's response is also strong but slightly less detailed and efficient in terms of travel route optimization.

- Figure 13. Qualitative Case in Creative Multimodal Understanding. This case comes from Travel Itinerary Planning and Recommendations Task, Assistant A is GPT-4o-1120, assistant B is InternVL2.5-78B.

[Figure 176]

###### Figure 14. Example Case of Literary Writing, from Task story continue.

[Figure 177]

###### Figure 15. Example Case of Literary Writing, from Task daily conversation creation.

[Figure 178]

###### Figure 16. Example Case of Literary Writing, from Task landscape to poem.

[Figure 179]

###### Figure 17. Example Case of Literary Writing, from Task historical story creation.

[Figure 180]

###### Figure 18. Example Case of Common Functional Writing, from Task daily achievement show off.

[Figure 181]

###### Figure 19. Example Case of Common Functional Writing, from Task social media travel content.

[Figure 182]

###### Figure 20. Example Case of Common Functional Writing, from Task daily affairs inquiries.

[Figure 183]

###### Figure 21. Example Case of Common Functional Writing, from Task personal event summaries.

[Figure 184]

###### Figure 22. Example Case of Professional Functional Writing, from Task teaching plan.

[Figure 185]

Figure 23. Example Case of Professional Functional Writing, from Task product marketing strategy.

[Figure 186]

Figure 24. Example Case of Professional Functional Writing, from Task nutritional formulation of recipe.

[Figure 187]

###### Figure 25. Example Case of Professional Functional Writing, from Task clothing match design.

[Figure 188]

###### Figure 26. Example Case of Creative Multimodal Understanding, from Task advertisement explanation.

[Figure 189]

###### Figure 27. Example Case of Creative Multimodal Understanding, from Task document understanding.

[Figure 190]

###### Figure 28. Example Case of Creative Multimodal Understanding, from Task snapshot analysis.

[Figure 191]

###### Figure 29. Example Case of Creative Multimodal Understanding, from Task travel itinerary planning and recommendations.

Query

Creation-MMBench: Assume you are <Role> <Background> Please follow the requirements below to <Instruction>. <Requirement> Creation-MMBench-TO: Assume you are <Role> <Background> Please follow the requirements below to <Instruction>. <Requirement> This question does not provide images, only descriptions of images by a large language model. Please answer based on the descriptions. Description of the image: <Image Description>

Figure 30. Query Template of Creation-MMBench and Creation-MMBench-TO

Descrption

Generic Instruction: Please carefully describe the content of each incoming image, starting with the number of images. For each image, first provide a general introduction to the content, then describe the image type, characters and objects, scene and atmosphere, the relationships between people and objects, and any text on the image.

Query-Specific Instruction: Please carefully describe the content of each incoming image, starting with the number of images. For each image, first provide a general introduction to the content, then describe the image type, characters and objects, scene and atmosphere, the relationships between people and objects, and any text on the image. Please pay special attention to the following aspects: <query-specific part>.

Figure 31. Generic Instruction vs. Query-Specific Instruction of Image Description

Few-Shot Prompt Template

Your task is to give a concise instruction about what basic elements are needed to be described based on the given question. Ensure that your instructions do not cover the raw question, options, or thought process of answering the question.

##Question##: Assume you are an expert at parsing documents, extracting key points and core ideas from documents, and condensing long documents into one or two paragraphs to summarize them. I come from the environmental protection department. What conclusions can I draw from this report? Please provide key evidence from the document to support your answer, within 100 words. Please follow the requirements below to Answer my question.

- 1. Your answer should fit the content of the reference material.
- 2. Provide a concise answer with sufficient thinking, removing unnecessary details. ##Contents to observe##: All text and related charts on the picture show the trend of changes

##Question##: Assume you are a world traveler. You went to this particular country, Chile, and this is what you saw. Please follow the requirements below to write a Reddit post with these pictures, titled ”Am I the only one who feels Chile is extremely underrated as a travel destination?”

- 1. Combine the picture content to show the beautiful scenery of Chile.
- 2. Attract other users to reply ##Contents to observe##: The uniqueness of the main scenery in the picture and the emotions conveyed by the characters in the picture

##Question##: Assume you are an experienced Reddit post creator seeking help and guidance. There is a problem while uninstalling a software from the iPhone 12. Please follow the requirements below to identify what the problem is based on this image and write a Reddit post asking for help.

- 1. The post needs a concise title that clearly summarizes the problem
- 2. The post should include tags closely related to the problem to increase the likelihood of it being seen or answered
- 3. The post requires a detailed body to provide more information for potential responders ##Contents to observe##: The situation on the interface and the main issues identified

##Question##: Assume you are experienced UI/UX designer, skilled in analyzing and optimizing interface designs to enhance usability and visual appeal. Please follow the requirements below to analyze and propose optimization suggestions for the current UI design based on the provided image.

- 1. The goal of the optimization is to improve the interface layout, user interaction flow, and overall visual aesthetics while ensuring a seamless user experience.
- 2. After optimization, the application should have strong visual appeal, easy navigation, and provide users with an enjoyable experience, ultimately increasing user engagement and retention rates. ##Contents to observe##: Interface layout and related information on the interface

##Question##: <Question>. ##Contents to observe##:

Figure 32. The Prompt Template for the GPT-4o to Generate the ”Query-Specific Part”.

Subjective Judge

<Image Content> Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt below, considering both the provided criteria and the image. Your task is to carefully assess each response based on how well it meets the evaluation criteria, incorporating the visual context from the image. The criteria should be the primary basis for your judgment, with the image serving to complement and inform your analysis. Steps for Evaluation:

- 1. Review Both Responses Independently: Carefully analyze Assistant A’s and Assistant B’s responses with the criteria and the image. Do not assume any

response is better just because it is listed first. Each response should be independently assessed based on the criteria and aided by images to help understand the context.

- 2. Compare the Strengths and Weaknesses: After evaluating each response independently, compare the two. Consider both the quality of the content and how

closely it aligns with the criteria and image. Identify the strengths and weaknesses of each response, and highlight the key differences.

- 3. Ensure Fairness: To avoid positional bias, swap the positions of Assistant A and Assistant B after the first evaluation (i.e., make

Assistant A become Assistant B and vice versa) and repeat the analysis and comparison. This ensures that each response is evaluated impartially under the same criteria.

- 4. Provide a Conclusion Based on Both Evaluations: After completing both evaluations (original and swapped positions), combine your analysis to provide a final

verdict. If the responses are similar, with only minimal differences, your judgment should reflect that and indicate a tie. Possible Verdict Options:

- • If Assistant A is clearly better in both evaluations: [[A>>B]]
- • If Assistant A is slightly better in both evaluations: [[A>B]]
- • If both responses are nearly identical, showing minimal differences and no clear advantage: [[A=B]]
- • If Assistant B is slightly better in both evaluations: [[B>A]]
- • If Assistant B is clearly better in both evaluations: [[B>>A]] Instructions to the AI Assistants: [INSTRUCTIONS] <instructions> [END INSTRUCTIONS]

- Assistant A Response:

- [ASSISTANT A] <Reference Answer> [END ASSISTANT A] Evaluation Criteria: [CRITERIA] <Subjective Criteria> [END CRITERIA]

Assistant B Response:

- [ASSISTANT B] <Model Prediction> [END ASSISTANT B] Output Format: Your output should include:

- 1. Evaluation of Assistant A’s Response: Provide a detailed qualitative evaluation, focusing on how well Assistant

- A’s response aligns with the criteria and the image.

2. Evaluation of Assistant B’s Response: Provide a detailed qualitative evaluation, focusing on how well Assistant

- B’s response aligns with the criteria and the image.

- 3. Final Verdict: After considering both evaluations, select one of the following verdicts and justify it based on your analysis: Your output format should end like this:

- Assistant A Evaluation: [qualitative comment]
- Assistant B Evaluation: [qualitative comment] Final Verdict is: [[VERDICT]]

Figure 33. Subjective Judge Prompt Template of Creation-MMBench

Visual Judge

With GroundTruth: Please act as an impartial judge and evaluate the Visual Factuality of the responses provided by two AI assistants to the user prompt displayed below. The responses were generated based on the provided instructions and visual input from images. There is a provided ground truth for the instructions, but the ground truth was not given to the AI assistants when generating their responses. Take this context into account when making your judgment. Steps for Evaluation:

- 1. Evaluate visual factuality for both responses based on the provided ground truth and visual factuality criteria.

- • If the visual factuality criteria consist of **X aspects**, each aspect is worth **10/X points**.
- • For each aspect, there may be multiple small criteria. If there are **Y small criteria in one aspect**, each small criterion is worth **10/X/Y points**.

- 2. Assign a total score out of 10 for each response. Instructions to the AI assistants: [INSTRUCTIONS] <instructions> [END INSTRUCTIONS]

- Assistant A response:

- [ASSISTANT A] <Reference Answer> [END ASSISTANT A] Visual Factuality Criteria: [VISUAL FACTUALITY CRITERIA] <Visual Factuality Criteria> [END CRITERIA]

Assistant B response:

- [ASSISTANT B] <Model Prediction> [END ASSISTANT B] Ground truth: [GROUND TRUTH] <GroundTruth> [END GROUND TRUTH] Your output should evaluate visual factuality scores for each assistant and end like this:

- Response A Visual Factuality Score: X/10
- Response B Visual Factuality Score: Y/10 Without GroundTruth: Please act as an impartial judge and evaluate the Visual Factuality of the responses provided by two AI assistants to the user prompt displayed below. The responses were generated based on the provided instructions and visual input from images. Take this context into account when making your judgment. Steps for Evaluation:

1. Evaluate visual factuality for both responses based on the visual factuality criteria.

- • If the visual factuality criteria consist of **X aspects**, each aspect is worth **10/X points**.
- • For each aspect, there may be multiple small criteria. If there are **Y small criteria in one aspect**, each small

criterion is worth **10/X/Y points**.

2. Assign a total score out of 10 for each response. Instructions to the AI assistants: [INSTRUCTIONS] <instructions> [END INSTRUCTIONS]

- Assistant A response:

- [ASSISTANT A] <Reference Answer> [END ASSISTANT A] Visual Factuality Criteria: [VISUAL FACTUALITY CRITERIA] <Visual Factuality Criteria> [END CRITERIA]

Assistant B response:

- [ASSISTANT B] <Model Prediction> [END ASSISTANT B] Your output should evaluate visual factuality scores for each assistant and end like this: Response A Visual Factuality Score: X/10 Response B Visual Factuality Score: Y/10

Figure 34. Visual Factuality Judge Prompt Template of Creation-MMBench

