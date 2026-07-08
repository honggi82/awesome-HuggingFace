# arXiv:2503.11557v1[cs.CV]14Mar2025

## VERIFY: A Benchmark of Visual Explanation and Reasoning for Investigating Multimodal Reasoning Fidelity

Jing Bi1 Junjia Guo1 Susan Liang1 Guangyu Sun2 Luchuan Song1 Yunlong Tang1 Jinxi He1 Jiarui Wu1 Ali Vosoughi1 Chen Chen2 Chenliang Xu1 1University of Rochester 2University of Central Florida

{jing.bi, jguo40, sliang22, lsong11, yunlong.tang, ali.vosoughi, chenliang.xu}@rochester.edu {jhe44, jwu114}@u.rochester.edu guangyu@ucf.edu, chen.chen@crcv.ucf.edu

#### Abstract

Visual reasoning is central to human cognition, enabling individuals to interpret and abstractly understand their environment. Although recent Multimodal Large Language Models (MLLMs) have demonstrated impressive performance across language and vision-language tasks, existing benchmarks primarily measure recognition-based skills and inadequately assess true visual reasoning capabilities. To bridge this critical gap, we introduce VERIFY, a benchmark explicitly designed to isolate and rigorously evaluate the visual reasoning capabilities of state-of-the-art MLLMs. VERIFY compels models to reason primarily from visual information, providing minimal textual context to reduce reliance on domain-specific knowledge and linguistic biases. Each problem is accompanied by a human-annotated reasoning path, making it the first to provide in-depth evaluation of model decision-making processes. Additionally, we propose novel metrics that assess visual reasoning fidelity beyond mere accuracy, highlighting critical imbalances in current model reasoning patterns. Our comprehensive benchmarking of leading MLLMs uncovers significant limitations, underscoring the need for a balanced and holistic approach to both perception and reasoning. For more teaser and testing, visit our project page.

#### 1. Introduction

Visual reasoning is a key aspect of human intelligence, shaping how individuals acquire knowledge and interpret the world [20, 26, 40]. As a key mechanism for concept formation and understanding, it has been the focus of extensive research [35]. However, classic models that overlook reasoning abilities, often trained on small-scale datasets with limited access to formalized reasoning, struggle to generalize beyond specific tasks and domains [27, 35, 36, 44].

In contrast, recent advances in Multimodal Large Lan-

###### Question

###### Human Reasoning Path

Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

Since the composition of the figures differs and there is no obvious attribute rule, priority should be given to considering numerical rules. The first figure in the question stem contains a single curve, so we consider the number of curves. Upon observation, it is found that each figure in the question stem has only one curve, hence the ? (blank) figure should also have only one curve. Only option A meets this criterion.

[Figure 1]

Answer: A Pattern: Line Property OPENAI-o1 Correctness: ❌ Recognition: 0, Abstraction: 2, Deduction: 1

A concise way to see why “FAIR” fits is to note that in each of the given words, if you drop the first letter you still (more or less) get a recognizable English word:

- • THIS → HIS
- • HELLO → ELLO (“ello” is at least a colloquial form)
- • HAIR → AIR Of the answer choices, only “FAIR” continues that pattern (FAIR → AIR). The others (“BOOK,” “EYE,” “BYE”) do not reduce to a standard English word by removing just the first letter. Hence “FAIR” is the best fit.

This answer reveals Semantic Dominance Bias in model reasoning. To overcome this, letters must be perceived as shapes rather than symbols. Like the well-known “How many ‘r’s in strawberry?” puzzle, the model struggles recognizing letters visually.

Figure 1. This example demonstrates that current MLLMs primarily depend on straightforward visual signals (e.g., letters) for reasoning, frequently neglecting patterns based on other characteristics, such as shapes or line properties. VERIFY delivers humanannotated reasoning paths to enhance the evaluation and comprehension of why and when models fail.

guage Models (MLLM) [6, 28, 32, 34, 47, 60, 62] have achieved near-human performance in various visionlanguage tasks [7, 17, 29, 45, 48]. Nevertheless, existing benchmarks predominantly evaluate recognition-based skills, such as object detection, image captioning, and optical character recognition (OCR), which primarily rely on perceptual abilities [11, 31, 54, 56]. Recent studies have started evaluating MLLM performance in more cognitively demanding domains, including mathematical reasoning and chart-based visual understanding [33, 37], which require

reasoning of numerical and logical operations. However, these benchmarks often conflate reasoning ability with domain knowledge drawn from the underlying LLM, making it hard to isolate and accurately assess reasoning based solely on visual information. Additionally, most evaluations emphasize accuracy as their primary metric while neglecting the decision-making quality and the depth of the reasoning process, such that even when models correctly identify the visual pattern, they often struggle to pinpoint the right answer. This gap raises an essential research question: To what extent can MLLMs genuinely perform visual reasoning and demonstrate systematic cognitive understanding?

To address this gap, we introduce VERIFY (Visual Explanation and Reasoning for Investigating Multimodal Reasoning Fidelity), a new benchmark designed to rigorously assess the reasoning fidelity of leading MLLMs, covering both open-source and proprietary systems. Inspired by The Raven’s Progressive Matrices[5], VERIFY isolates visual reasoning ability by minimizing reliance on textual input, compelling models to reason primarily through visual information. As shown in Figure 1, each problem includes a concise textual question and one image with answer options. This design isolates visual reasoning by requiring models to predominantly rely on non-textual cues.

By providing human-annotated reasoning paths, VERIFY not only highlights when such biases occur but also offers insights into why models might misinterpret or overemphasize specific elements. This approach ultimately aims to foster a more balanced evaluation of visual reasoning, encouraging models to integrate both semantic and abstract visual cues for a more accurate and holistic understanding.

VERIFY encompasses a diverse and challenging set of visual reasoning tasks requiring generalization and abstraction, as exemplified in Figure 2, surpassing existing benchmarks in both diversity and difficulty as shown in Table 1.

Leveraging this benchmark, we evaluate leading MLLMs from OpenAI, Google, and various most recent open-source models to provide a comprehensive analysis of their visual reasoning abilities. Our results reveal that even the most advanced models achieve an accuracy of only 21.7%, which is below random chance (25%). Furthermore, we identify shortcomings in current automated evaluation methods [51], noting their insufficiency in capturing nuanced differences between models. Inspired by foundational studies in human visual reasoning [5], we propose novel metrics designed to evaluate the reasoning process itself, providing a more refined assessment beyond accuracy. Our findings emphasize that for advanced visual reasoning models, the reasoning path holds greater importance than merely obtaining the correct answer, as models typically explore multiple reasoning paths during problem-solving.

Extensive qualitative analyses of the models’ reasoning patterns reveal a critical imbalance, where models profi-

cient in perceiving visual details often struggle to deduce correct answers, while those excelling in abstract reasoning frequently overlook essential visual cues, highlighting a deficiency in balanced reasoning capabilities. In summary, our contributions are as follows:

- • VERIFY: We introduce a novel visual reasoning benchmark isolating reasoning ability by minimizing domainspecific and linguistic biases. To the best of our knowledge, it is the first to provide human-annotated, clear, and evaluable reasoning trajectories for each question.
- • Beyond Accuracy: We proposal novel metrics that assess the reasoning process itself, providing a more nuanced and comprehensive evaluation beyond traditional accuracy, capturing the depth and quality of reasoning.
- • Comprehensive Benchmarking: VERIFY systematically evaluates leading MLLMs on reasoning capabilities, perceptual alignment, and logical consistency, highlighting the impact of Chain-of-Thought (CoT) reasoning and the challenges in transitioning from abstraction to answer deduction, offering valuable insights.

#### 2. Related Work

Visual Reasoning. Visual reasoning involves comprehending and manipulating visual information to infer logical relationships [3, 15]. Early tasks like VQAv1 and VQAv2

- [3, 15] focused on object-centric question answering, while CLEVR [25] and GQA [19] introduced compositional programmatic queries for structured reasoning. Beyond static images, video-based datasets such as CoPhy and Space
- [4, 12, 21] emphasize physical interactions and causal reasoning in motion. Our work aligns with abstract visual reasoning that does not require external domain knowledge, like RAVEN, SVRT, CVR, Bongard-HOI, and BongardLOGO [13, 23, 38, 58, 59], which explore relational patterns among shapes. However, as Van der Maas et al. [52] note, many existing words have rigid configurations that hinder generalization to complex problems.

Reasoning Benchmarks. Most popular language reasoning benchmarks focus on text-based QA, mathematics, or code inference [10, 16, 42, 43], where step-by-step solutions can be programmatically verified. In the visual domain, benchmarks like CLEVR [25], GQA [19], CLEVRER [55], Bongard [38], RAVEN [59], MMComposition [18], and VidComposition [46] introduce compositional reasoning over images or videos. Recent efforts [22] incorporate multi-modal CoT prompting, often blending external knowledge with visual cues. Among the most relevant is MM-IQ [8], which targets abstract visual puzzles but lacks step-by-step annotations to clarify reasoning transitions. In contrast, our work provides explicit human-annotated reasoning paths, offering a more transparent, interpretable framework for fine-grained evaluation beyond the accuracy.

Reasoning Path

###### Positional Relationships

###### Gradual Change

###### Spatial Position

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Q: Select an option whose 3D shape can be folded from the given paperboard on the left.

Q: Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

Q: Choose the most appropriate option to fill in the question mark so that both sets of figures share similarities while maintaining their distinct characteristics.

###### Line Properties

[Figure 6]

Explanation

###### Symmetry and Axes

###### Element Operations

[Figure 7]

[Figure 8]

Observing the movement, we find that the black ball moves one position clockwise each time, alternating between appearing inside and outside the box. The cross moves two positions clockwise each time, also alternating between appearing outside and inside the box. The white ball moves three positions clockwise each time, following the same alternating pattern. Only option D meets these conditions. Therefore, the correct answer is D.

A. ①③⑤; ②④⑥

B. ①②⑤; ③④⑥ C. ①④⑤; ②③⑥ D. ①③④; ②⑤⑥

Visual Aspect 1: translation

| | |
|---|---|
|Entities|Transformations|
|black ball|moves 1 space clockwise|
|cross sign|moves 2 spaces clockwise|
|white ball|moves 3 spaces clockwise|

Q: Divide the following six figures into two groups based on their pattern and select the correct classification method.

A. ①③④; ②⑤⑥ B. ①③⑤; ②④⑥ C. ①②⑥; ③④⑤ D. ①④⑥; ②③⑤

###### Structural Composition & Spatial Division

[Figure 9]

Q: Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

Q: Divide the following six figures into two groups based on their pattern and select the correct classification method.

Visual Aspect 2: Alternating positions

###### Geometric Transformations

###### Mathematics

| | |
|---|---|
|Entities|Transformations|
|black ball|alternates between inside and outside|
|cross sign|alternates between outside and inside|
|white ball|alternates between inside and outside|

[Figure 10]

[Figure 11]

A. ①②③; ④⑤⑥ B. ①③⑤; ②④⑥ C. ①②⑤; ③④⑥ D. ①④⑤; ②③⑥

Q: Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

Q: Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

Q: Divide the following six figures into two groups based on their pattern and select the correct classification method.

- Figure 2. Categories from the VERIFY dataset cover a range of patterns, from logical operations to 3D geometry and mathematics. The right panel presents a human reasoning path, demonstrating how visual transformations, rotations, and inside-outside shifts lead to the final answer. We encourage readers to test these examples with MLLM models (e.g., o1 or Gemini ) to assess their reasoning capabilities.

Automatic Reasoning-Path Evaluation. Interest is growing in evaluating not just the accuracy a model provides, but also how it draws its conclusions. 1. Embedding Methods. Golovneva et al. [14] use embedding similarities to compare reasoning chains, effective for text but weaker for capturing geometric or visual relationships. 2. Symbolic Methods. Structured parsing, such as subject-verb-object [39] or formal proofs [41], struggles with complex visual puzzles due to inconsistent symbolic representation of spatial relations. 3. Process Reward Models. Lightman et al. [30] introduce PRM800K for mathematical reasoning, but training such models requires vast amounts of data, making large-scale visual adaptation costly. 4. Prompting Methods. LLMs can verify reasoning by prompting without ground-truth references, depending solely on internal coherence. [51, 57] Applying these methods on visual reasoning yields poor results, as LLMs struggle to differentiate visual nuances.

#### 3. Data Collection

Our dataset stands apart from existing ones that rely on rendering engines to ensure compositionality. While synthesized data can be abundant, it often lacks the richness and complexity inherent in real-world scenarios. We curate our dataset primarily from publicly available questions in China’s National Civil Servants Examination, a rigor-

- ous postgraduate-level assessment. To mitigate any po-

tential bias, we have incorporated questions from different provinces to ensure a diverse and comprehensive dataset. Focusing on problems that require logical reasoning rather than simple recognition, we deliberately select complex questions demanding trial and error with multiple visual patterns. This ensures both correctness and validity while establishing a higher standard of difficulty than existing datasets. To ensure quality and clarity, we included only questions with definitive answers, filtering out ambiguous ones, those with multiple correct answers, or requiring external knowledge. Details about selection criteria are in the supplementary materials.

##### 3.1. Reasoning Path

One of the key contributions of our dataset is the reasoning path, which consists of a sequence of logical steps leading to the correct answer. The common practice [8, 61] is to rely on models to generate reasoning paths, which are then reviewed and evaluated by human annotators or an external verifier. While existing methods may work for moderately difficult tasks, they fail when faced with more challenging reasoning problems like ours. In our initial experiments, we found that state-of-the-art models struggle to generate correct reasoning paths, even as a starting point. As a result, we manually annotate each reasoning path to ensure accuracy. Each path is independently labeled by three annotators, and any discrepancies are resolved through discussion. Further details will be provided in supplementary materials.

[Figure 12]

- Figure 3. We divide the reasoning process into four key stages inspired by human visual reasoning: perception, recognition, abstraction, and deduction. Unlike general visual tasks, where perception involves detecting raw visual features, humans often have implicit perception because the provided visual elements are already structured for direct recognition of useful components. Even for shown complex problems, a model with strong visual abilities—like Gemini—can effectively analyze patterns and logical structures to determine the correct answer.

##### 3.2. Visual Reasoning Pattern

To solve the proposed visual reasoning problem, humans must first identify the crucial visual elements necessary for detecting patterns. However, these elements are not always immediately apparent, and we often fall prey to Semantic Dominance Bias, which causes us to focus too much on the semantic meaning of a figure rather than carefully analyzing its shape or other visual properties. To systematically approach the problem, we define the reasoning process through four stages, as shown in Figure 3, which also enables a structured evaluation. I. Perception, where the raw visual input is processed, and features such as shapes, colors, and orientations are detected; II. Recognition, which involves extracting useful visual aspects from Perception, selecting the most relevant features that contribute to understanding the pattern; III. Abstraction, where these extracted visual aspects are used to identify patterns by filtering out unnecessary details and focusing on meaningful relationships and IV. Deduction, the final stage, in which logical reasoning is applied to infer missing details or predict patterns based on the extracted abstractions.

For humans, the Perception stage is often implicit and may even be bypassed, as some initially relevant visual elements may later prove irrelevant. As a result, Recognition becomes the first explicit step, where annotators identify only the essential visual aspects and structure the reasoning path accordingly. After annotating the reasoning path, we summarize and identify key visual reasoning patterns for subsequent evaluation, as illustrated in Figure 2. While MARVEL [24] and MM-IQ [8] provided useful insights, they proved insufficient for our dataset.

To address this limitation, we extracted refined patterns

- as follows that build upon existing ones while introducing unique structural distinctions. Notably, we separated sym-

metry from geometric transformations to enhance clarity. I. Spatial Position (SP): How elements are positioned (e.g., top-bottom, center-corner) and how they change through movement, rotation, or progression. II. Element Operations (EO): Focuses on interactions between shapes, such as layering, merging, splitting, and reassembly. III. Geometric Transformations (GT): Involves transformations like rotation, translation, and reflection, often occurring in sequences. IV. Gradual Change (GC): Tracks progressive changes in shape, size, or position, forming logical patterns. V. Mathematics (MA): Observes numerical aspects, including element counts and repetitive patterns. VI. Symmetry and Axes (SA): Identifies different types of symmetry, including axial, central, and rotational. VII. Line Properties (LP): Analyzes stroke characteristics such as type, angle, continuity, and curvature. VIII. Structural Composition and Spatial Division (SC): Examines enclosures, layering, connections, and shading to uncover deeper structural patterns. IX. Positional Relationships (PR): Investigates how elements relate to each other in terms of position, alignment, and directional shifts.

[Figure 13]

Figure 4. Category distribution of VERIFY: This chart illustrates the distribution of various mathematical and geometric concepts, with the largest segment, “Mathematics,” accounting for 24.3%. The remaining segments are fairly evenly distributed, reflecting a balanced emphasis on all visual patterns.

##### 3.3. Dataset Statistics

The table1 compares various datasets used for reasoning tasks, highlighting differences in size, source, difficulty,

Name Size Source RP Difficulty Year SVRT [13] 23 Synthesized × Easy 2011 RAVEN [59] 14,000 Synthesized × Easy 2019 ARC [9] 600 Synthesized × Medium 2019 DOPT [53] 95,200 Synthesized × Easy 2020 IQTest [33] 228 Mixed × Medium 2023 MARVEL [24] 770 Web × Medium 2024 MM-IQ [8] 2,710 Mixed × Medium 2025 VERIFY 600 National Exam ✓ Hard 2025

Table 1. Comparison of datasets from the corresponding paper. ”RP” indicates the inclusion of a reasoning path, and the difficulty level is based on both the human score and the participants’ education level as reported in the original study.

and reasoning complexity. Among them, VERIFY stands

- out as the most challenging due to its high quality, realworld origin (national exams), deep reasoning complexity, and rigorous difficulty level. Compared to existing datasets, VERIFY is more challenging and diverse. While it has fewer samples than large-scale synthetic datasets, it surpasses other handcrafted datasets in both the number of samples and the diversity of reasoning patterns. Unlike synthetic datasets that focus on pattern recognition, VERIFY includes real-world, high-stakes reasoning tasks, making it a more reliable benchmark for evaluating cognitive abilities.

#### 4. Evaluation

Given a visual reasoning problem q, a generated reasoning path hˆ = {hˆ1,...,hˆN}, and a generated answer aˆ produced by MLLMs, our goal is to evaluate the quality of both the reasoning process and the final answer. In our case, the ground truth answer a is available, along with reference solution steps h = {h1,...,hM}, as a reliable benchmark.

Unlike previous works that focus solely on answer matching—ignoring the entire chain of reasoning—we independently evaluate both the final answer and the reasoning path. This approach provides more informative insights into the model’s reasoning process. Our empirical study reveals that models can sometimes generate the correct answer with an incorrect reasoning path or produce an incorrect answer despite following a correct reasoning process. By separately assessing these components, we gain a more comprehensive understanding of models.

##### 4.1. Failure case of existing evaluation metrics

Recent studies [51, 57] have explored the use of open source LLMs as automated evaluators for reasoning processes, by comparing each step of the whole path against a reference reasoning trajectory. This trend aligns with the growing interest in using LLMs as judges to provide consistent and scalable evaluation frameworks. During the early stages of designing our evaluation protocol, we tested this methodology using three of the latest highcapacity LLMs as judges: DeepSeek-R1:32B, DeepSeekR1:70B, and LLaMA3.3-70B. We randomly selected 40 out

of 600 samples, using the o1 response as the model response, and evaluated reasoning paths by comparing them against multiple reference responses I. o1: The reasoning path and final answer produced by the OpenAI-o1 model. II. Ground Truth Reasoning (GTR): A human-annotated reasoning path that correctly leads to the answer. III. Expanded Ground Truth Reasoning (GTR+): Leverage LLM to expand and segment the GTR into three detailed stages: Recognition, Abstraction and Deduction. IV. Expanded Ground Truth Reasoning with Pattern (GTR+P): A refined version that incorporates human-labeled pattern annotations, offering additional structural insights into the reasoning process. Scores were assigned on a scale from 0 to 10 based on fidelity to human reasoning. The evaluation process involved assessing the alignment of model-generated reasoning paths with ground truth references. Additionally, we conducted two types of comparisons: I. r+c: The model’s response includes both its reasoning path and the final selected choice, with the ground truth answer. II. r: The model’s response includes only the reasoning path, without the final choice. Based on an accuracy of 0.15 for o1 across 40 problems, several key observations emerge:

r1 r2 Model Score (r+c) Score

- o1 GTR+P deepseek-r1:32b 7.85 6.40
- o1 GTR+ deepseek-r1:32b 7.60 6.80
- o1 GTR deepseek-r1:32b 7.80 6.35 GTR GTR+P deepseek-r1:32b 8.45 7.15 GTR GTR+ deepseek-r1:32b 8.70 8.00
- o1 GTR+P deepseek-r1:70b 8.10 7.80
- o1 GTR+ deepseek-r1:70b 8.35 8.10
- o1 GTR deepseek-r1:70b 7.65 6.60 GTR GTR+P deepseek-r1:70b 8.65 8.65 GTR GTR+ deepseek-r1:70b 8.80 8.65
- o1 GTR+P llama3.3 8.05 8.20
- o1 GTR+ llama3.3 8.15 8.35
- o1 GTR llama3.3 7.95 8.20 GTR GTR+P llama3.3 9.05 9.40 GTR GTR+ llama3.3 9.25 9.85

Table 2. Using an LLM as a judge to compare the relevance scores of two reasoning paths often falls short in correctness. r1 and r2 are the two reasoning paths we compare against.

Overestimation in LLM Judges. Table 2 shows a striking discrepancy: human evaluators marked only 6 out of 40 answers as correct, yet LLMs consistently assigned scores in the range of 6.35–9.85. In our experiment, we ask each LLM to compare two reasoning paths and score them on a scale from 1 to 10 based on how well they align. This disparity indicates that LLMs tend to overestimate reasoning quality, often relying on superficial similarity rather than a rigorous evaluation of logical correctness. Notably, LLaMA3.3-70B assigns comparably high scores to both

[Figure 14]

- Figure 5. This diagram illustrates three proposed evaluation metrics: Agreement, which extracts common elements across model answers and compares their consistency; Match, which assesses alignment between model responses and ground truth across Recognition, Abstraction, and Deduction; and Perception Similarity, which measures the similarity of extracted perception attributes across different models.

GTR and its structured variants (GTR+ and GTR+P), while it awards slightly lower scores to o1 relative to the ground truth. Nonetheless, the overestimation bias persists, underscoring the challenge LLMs face in adequately penalizing flawed reasoning.

Bias Toward Completed Choices. Surprisingly, LLMs tend to assign higher scores when the final choice is included (r+c), even though they should recognize that the answer is incorrect and penalize it accordingly. We suspect that including the choice makes the reasoning path feel more complete, biasing LLM judges toward a more favorable answer.

Our experiment reveals that LLMs prioritize completeness over actual correctness, favoring coherence even at the expense of reasoning accuracy. This bias results in a failure to properly penalize flawed logic, leading to overestimation and missed nuances in visual reasoning evaluation.

##### 4.2. Evaluation Framework

Our evaluation framework is designed to systematically assess the reasoning capabilities of models by decomposing their responses into distinct cognitive stages. To achieve this, we extract structured elements from both the ground truth and model-generated answers, facilitating a detailed comparison.

###### 4.2.1. Decomposition of Reasoning Stages

Based on the ground truth, we categorize the reasoning process into three key stages:

Recognition (R): Identifying key visual elements necessary for solving the problem. Abstraction (A): Inferring higher-level patterns or relationships from R. Deduction (D): Applying logical reasoning to the abstracted patterns to reach a conclusion.

For model responses, we further introduce a Perception (Pˆ) stage, which represents the raw visual elements detected by the model before recognition and reasoning take place. This leads to an extracted model reasoning path:

hˆ = {P,ˆ R,ˆ A,ˆ Dˆ}

###### 4.2.2. Evaluation of Reasoning Stages

For R, A, and D, we perform direct matching against the ground truth using a strong reasoning model, QWQ[50]. This ensures a robust comparison while accounting for linguistic variability in model-generated responses.

###### 4.2.3. Perception Evaluation

Unlike R, A, and D, the evaluation of perception (Pˆ) lacks a well-defined ground truth, as manually verifying each re-

Model COT GT MA GC LP EO PR SA SC SP ALL Open-Source MLLMs Qwen2.5-70b × 0.210 0.180 0.230 0.120 0.240 0.110 0.170 0.205 0.190 0.195 QVQ-72b ✓ 0.197 0.158 0.196 0.091 0.267 0.147 0.200 0.212 0.100 0.177 Mulberry-8b ✓ 0.230 0.240 0.157 0.145 0.253 0.067 0.164 0.135 0.233 0.187 LLaVA-CoT-11b ✓ 0.280 0.200 0.300 0.150 0.260 0.100 0.140 0.190 0.220 0.210 Proprietary MLLMs GPT-4o × 0.262 0.178 0.353 0.091 0.200 0.093 0.164 0.250 0.200 0.192 Gemini ✓ 0.311 0.185 0.275 0.145 0.280 0.093 0.109 0.115 0.267 0.193 OpenAI-o1 ✓ 0.279 0.233 0.275 0.164 0.293 0.120 0.145 0.192 0.200 0.215

Table 3. Compact performance comparison of MLLMs with or without CoT support.

ti-choice questions. 2. Computing Multi-Choice Answer Accuracy: Accuracy is measured across reasoning categories.

sponse is labor-intensive. To address this, we propose two unsupervised evaluation methods:

- 3. Assessing Reasoning Path: Evaluates reasoning based on

- our proposed evaluation metrics. 5.1. Analysis of Accuracy

With a random-choice baseline of 0.25 for a 1-out-of4 selection, all models underperform relative to chance. Our proposed benchmark reveals a significant challenge in uncovering the subtle intricacies of visual reasoning. Among open-source MLLMs, LLaVA-CoT (11B) achieves the highest overall accuracy, closely followed by Qwen2.5 (72B). Notably, despite lacking Chain-ofThought (CoT) support, Qwen2.5 (72B) remains highly competitive. Mulberry (8B) and QVQ (72B) trail slightly behind. In the proprietary category, o1 demonstrates the best performance, surpassing both GPT-4o (0.192) and Gemini 2.0 Flash Thinking (0.193). Interestingly, despite lacking explicit CoT support, GPT-4o still performs at a comparable level to Gemini. We include a detailed QVQ response in the supplementary material to showcase its ability to switch reasoning paths.

Impact of CoT. Models equipped with CoT reasoning generally exhibit improved performance in reasoning-intensive tasks. Gemini 2.0 Flash Thinking and o1 outperform GPT4o , reinforcing the hypothesis that step-by-step reasoning enhances accuracy. A similar trend is observed among open-source models, where LLaVA-CoT (11B) leads the group in accuracy. Proprietary models continue to lead in performance over open-source alternatives, benefiting from extensive training data, optimized architectures, and greater computational resources. However, among opensource models, LLaVA-CoT (11B) emerges as a strong contender, demonstrating balanced accuracy across vari-

- ous benchmarks. Notably, models incorporating Chainof-Thought (CoT) reasoning consistently excel in complex problem-solving, underscoring the significance of structured, step-by-step methodologies. Despite this trend, some open-source models, such as Qwen2.5 (72B), achieve re-

Common Element Agreement. We prompt a large language model (LLM) to extract common visual elements across all model responses. The agreement score for each model is then computed based on the frequency with which it includes these shared elements, offering a measure of consistency across models.

Inter-Model Perception Similarity. To understand how models interpret visual elements relative to one another, we compute pairwise similarity scores between model responses. This results in an n × n matrix that quantifies perceptual alignment between models.

Our evaluation framework provides a scalable and structured approach to assess both the structured reasoning process and raw perception abilities of models. This enables a more comprehensive understanding of their cognitive performance across multiple dimensions. More details and examples will be included in the supplementary material.

#### 5. Experiments and Results

In this section, we evaluate the visual reasoning abilities of MLLMs with or without CoT. During the initial phase of our evaluation, we tested a range of open-source and smaller models (13B, 32B) on a randomly selected set of 40 questions. The results revealed that these models struggled, exhibiting both low accuracy and flawed reasoning. Recognizing these limitations, we refined our approach by selecting the most advanced open-source models specifically designed for reasoning: LLaVA-CoT (11B), Mulberry (8B), and QVQ (72B) [49], alongside Qwen2.5-72B, a cuttingedge vision model. For proprietary models, we included GPT-4o, as well as two of the most recent leading models: OpenAI-o1 [2] and Gemini 2.0 Flash Thinking [1].

To ensure a fair comparison, all models were evaluated under identical conditions with the same default hyperparameters. The evaluation process consisted of three steps. 1. Generating Responses: Models produce answers to mul-

[Figure 15]

- Figure 6. The first row visualizes the correlation between Recognition (R) and Abstraction (A), while the second examines the transition from Abstraction (A) to Deduction (D). Bubble sizes represent confidence levels. The analysis highlights inter-model variability, with stronger correlations in R → A than A → D, indicating challenges in applying logical deduction even when abstraction is well-formed. The last column aggregates all models within the same figure, providing an overall trend.

sults on par with proprietary counterparts, even without explicit CoT support, suggesting that advancements in model CoT strategy can help bridge the performance gap. Our analysis indicates that prioritizing the quality of the reasoning process can reveal crucial diagnostic insights that overall accuracy alone does not capture. In this work, the dataset and benchmark are strategically designed to highlight discrepancies along the reasoning path, thereby offering a robust framework to assess and enhance model performance.

##### 5.2. Analysis of Reasoning Path

The result of matching between model over Recognition, Abstraction, and Deduction as shown in Figure 6, providing insight into how well models transition from one stage to the next. The first row compares Recognition to Abstraction, evaluating how effectively models generalize visual elements into higher-level patterns. The second row compares Abstraction to Deduction, assessing the models’ ability to apply logical reasoning to abstracted representations. The bubble size represents the count of the same score.

Recognition vs. Abstraction. The scatter plots in the first row reveal a strong positive correlation between Recognition (R) and Abstraction (A), highlighting that models adept

- at recognizing key visual elements also tend to excel at forming higher-level abstractions. However, models like GPT-4o and Mulberry exhibit greater variance, suggesting inconsistency in abstraction despite strong recognition capabilities. This inconsistency may indicate limitations in generalizing patterns beyond direct visual cues. In contrast, models such as QVQ and Gemini demonstrate more stable performance, implying a robust ability to translate recognized elements into coherent abstract representations. This consistency makes them more reliable in tasks requiring structured pattern recognition.

[Figure 16]

Figure 7. Pairwise Similarity Matrix of Model Responses

[Figure 17]

Figure 8. Agreement scores compare each model’s consistency in identifying common visual elements.

Abstraction vs. Deduction. The second row illustrates the relationship between Abstraction (A) and Deduction (D), revealing a slightly weaker correlation in some models. This suggests that even when models successfully abstract information, their ability to apply logical deduction remains inconsistent. Notably, GPT-4o and QVQ show high variance in abstraction-to-deduction transitions, indicating unpredictability in logical reasoning. This suggests that while these models can form abstract representations, their deductive processes do not always follow a clear or systematic pattern. Conversely, models like LLaVA-CoT and Qwen2.5 maintain a more compact distribution, signify-

ing a structured and consistent approach to deduction when abstraction is well-formed. Their ability to apply logical reasoning in a predictable manner makes them more reliable for tasks requiring systematic inference.

Perception similarity between models. As shown in Figure 7, LLaVA-CoT and Mulberry (4.13), as well as Gemini and GPT-4o (4.12), exhibit the strongest perceptual alignment, suggesting these pairs process visual data similarly. QVQ shows the weakest alignment with LLaVACoT and o1 , indicating a distinct interpretation of visual inputs. GPT-4o and Qwen2.5 show relatively strong alignment with multiple models, suggesting a broader generalization. o1 maintains moderate alignment across models but does not dominate in similarity with any particular one. Performance trends indicate that proprietary models hold an edge, but open-source models like LLaVA-CoT and Qwen2.5 remain highly competitive.

Perception agreement between models. As shown in Figure8, the agreement score provides insight into the consistency of different models in identifying common visual elements across responses. Notably, proprietary models such as GPT-4o and o1 exhibit higher consistency, as indicated by their larger proportional representation in the agreement, suggesting that these models more frequently include commonly mentioned elements in the responses. Moreover, models like Qwen2.5 and QVQ demonstrate moderate agreement, indicating a balance between consistency and diversity in interpretations.

#### 6. Conclusion

In this paper, we introduce VERIFY, the first benchmark designed to assess the fidelity of visual reasoning paths in MLLMs. Through comprehensive experiments and analysis, we highlight the limitations of existing reasoning path evaluation methods and propose new metrics for a more thorough assessment of leading models. Our findings reveal significant shortcomings in current MLLMs’ ability to perform visual reasoning, emphasizing their tendency to excel in either perceptual grounding or logical reasoning, but rarely both. While models with CoT reasoning generally achieve higher accuracy, challenges persist in the transition from abstraction to deduction, limiting overall reasoning capabilities. Additionally, we observe strong correlations between recognition and abstraction, but inconsistencies arise in models’ ability to apply logical inference systematically. By providing a structured evaluation framework, we aim to bridge this gap and pave the way for future advancements in MLLM development.

#### References

- [1] Gemini 2.0 flash thinking. 7
- [2] Learning to reason with llms. 7
- [3] Aishwarya Agrawal, Jiasen Lu, Stanislaw Antol, Margaret Mitchell, C. Lawrence Zitnick, Dhruv Batra, and Devi Parikh. Vqa: Visual question answering, 2016. 2
- [4] Fabien Baradel, Natalia Neverova, Julien Mille, Greg Mori, and Christian Wolf. Cophy: Counterfactual learning of physical dynamics, 2020. 2
- [5] Lawrence W Barsalou. Perceptual symbol systems. Behavioral and brain sciences, 22(4):577–660, 1999. 2
- [6] Jing Bi, Yunlong Tang, Luchuan Song, Ali Vosoughi, Nguyen Nguyen, and Chenliang Xu. Eagle: Egocentric aggregated language-video engine. arXiv preprint arXiv:2409.17523, 2024. 1
- [7] Davide Caffagni, Federico Cocchi, Luca Barsellotti, Nicholas Moratelli, Sara Sarto, Lorenzo Baraldi, Marcella Cornia, and Rita Cucchiara. The revolution of multimodal large language models: a survey. arXiv preprint arXiv:2402.12451, 2024. 1
- [8] Huanqia Cai, Yijun Yang, and Winston Hu. Mm-iq: Benchmarking human-like abstraction and reasoning in multimodal models, 2025. 2, 3, 4, 5
- [9] Fran¸cois Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019. 5
- [10] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 2
- [11] Hongyuan Dong, Jiawen Li, Bohong Wu, Jiacong Wang, Yuan Zhang, and Haoyuan Guo. Benchmarking and improving detail image caption, 2024. 1
- [12] Jiafei Duan, Samson Yu Bai Jian, and Cheston Tan. Space: A simulator for physical interactions and causal learning in 3d environments, 2021. 2
- [13] Fran¸cois Fleuret, Ting Li, Charles Dubout, Emma K Wampler, Steven Yantis, and Donald Geman. Comparing machines and humans on a visual categorization test. Proceedings of the National Academy of Sciences, 108(43): 17621–17625, 2011. 2, 5
- [14] Olga Golovneva, Moya Chen, Spencer Poff, Martin Corredor, Luke Zettlemoyer, Maryam Fazel-Zarandi, and Asli Celikyilmaz. Roscoe: A suite of metrics for scoring step-bystep reasoning. arXiv preprint arXiv:2212.07919, 2022. 3
- [15] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering, 2017. 2
- [16] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. 2
- [17] Hang Hua, Yunlong Tang, Chenliang Xu, and Jiebo Luo. V2xum-llm: Cross-modal video summarization with temporal prompt instruction tuning. arXiv preprint arXiv:2404.12353, 2024. 1

- [18] Hang Hua, Yunlong Tang, Ziyun Zeng, Liangliang Cao, Zhengyuan Yang, Hangfeng He, Chenliang Xu, and Jiebo Luo. Mmcomposition: Revisiting the compositionality of pre-trained vision-language models. arXiv preprint arXiv:2410.09733, 2024. 2
- [19] Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering, 2019. 2
- [20] William H Ittelson. Visual space perception. 1960. 1
- [21] Steeven Janny, Fabien Baradel, Natalia Neverova, Madiha Nadri, Greg Mori, and Christian Wolf. Filtered-cophy: Unsupervised learning of counterfactual physics in pixel space,

2022. 2

- [22] Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanwei Li, Yu Qi, Xinyan Chen, Liuhui Wang, Jianhan Jin, Claire Guo, Shen Yan, et al. Mme-cot: Benchmarking chain-of-thought in large multimodal models for reasoning quality, robustness, and efficiency. arXiv preprint arXiv:2502.09621, 2025. 2
- [23] Huaizu Jiang, Xiaojian Ma, Weili Nie, Zhiding Yu, Yuke Zhu, Song-Chun Zhu, and Anima Anandkumar. Bongardhoi: Benchmarking few-shot visual reasoning for humanobject interactions, 2023. 2
- [24] Yifan Jiang, Jiarui Zhang, Kexuan Sun, Zhivar Sourati, Kian Ahrabian, Kaixin Ma, Filip Ilievski, and Jay Pujara. Marvel: Multidimensional abstraction and reasoning through visual evaluation and learning, 2024. 4, 5
- [25] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017. 2
- [26] Stephen M Kosslyn. Image and brain: The resolution of the imagery debate. MIT press, 1996. 1
- [27] Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. Building machines that learn and think like people. Behavioral and brain sciences, 40:e253,

2017. 1

- [28] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 1

- [29] Zijing Liang, Yanjie Xu, Yifan Hong, Penghui Shang, Qi Wang, Qiang Fu, and Ke Liu. A survey of multimodel large language models. In Proceedings of the 3rd International Conference on Computer, Artificial Intelligence and Control Engineering, pages 405–409, 2024. 1
- [30] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023. 3
- [31] Weifeng Lin, Xinyu Wei, Ruichuan An, Peng Gao, Bocheng Zou, Yulin Luo, Siyuan Huang, Shanghang Zhang, and Hongsheng Li. Draw-and-understand: Leveraging visual prompts to enable mllms to comprehend what you want. arXiv preprint arXiv:2403.20271, 2024. 1

- [32] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 1
- [33] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts, 2024. 1, 5
- [34] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 1
- [35] Mikołaj Małki´nski and Jacek Ma´ndziuk. Deep learning methods for abstract visual reasoning: A survey on raven’s progressive matrices. ACM Computing Surveys, 57(7):1–36,

2025. 1

- [36] Gary Marcus. Deep learning: A critical appraisal. arXiv preprint arXiv:1801.00631, 2018. 1
- [37] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning, 2022. 1
- [38] Weili Nie, Zhiding Yu, Lei Mao, Ankit B Patel, Yuke Zhu, and Anima Anandkumar. Bongard-logo: A new benchmark for human-level concept learning and reasoning. Advances in Neural Information Processing Systems, 33:16468–16480,

2020. 2

- [39] Archiki Prasad, Swarnadeep Saha, Xiang Zhou, and Mohit Bansal. Receval: Evaluating reasoning chains via correctness and informativeness. arXiv preprint arXiv:2304.10703,

2023. 3

- [40] Zenon W Pylyshyn. Seeing and visualizing: It’s not what you think. MIT press, 2003. 1
- [41] Abulhair Saparov and He He. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought. arXiv preprint arXiv:2210.01240, 2022. 3
- [42] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adri`a GarrigaAlonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022. 2
- [43] Mirac Suzgun, Nathan Scales, Nathanael Sch¨arli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022. 2
- [44] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013. 1
- [45] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, et al. Video understanding with large language models: A survey. arXiv preprint arXiv:2312.17432, 2023. 1
- [46] Yunlong Tang, Junjia Guo, Hang Hua, Susan Liang, Mingqian Feng, Xinyang Li, Rui Mao, Chao Huang, Jing

- Bi, Zeliang Zhang, et al. Vidcomposition: Can mllms analyze compositions in compiled videos? arXiv preprint arXiv:2411.10979, 2024. 2
- [47] Yunlong Tang, Daiki Shimada, Jing Bi, Mingqian Feng, Hang Hua, and Chenliang Xu. Empowering llms with pseudo-untrimmed videos for audio-visual temporal understanding. arXiv preprint arXiv:2403.16276, 2024. 1
- [48] Yunlong Tang, Junjia Guo, Pinxin Liu, Zhiyuan Wang, Hang Hua, Jia-Xing Zhong, Yunzhong Xiao, Chao Huang, Luchuan Song, Susan Liang, et al. Generative ai for celanimation: A survey. arXiv preprint arXiv:2501.06250,

2025. 1

- [49] Qwen Team. Qvq: To see the world with wisdom, 2024. 7
- [50] Qwen Team. Qwq: Reflect deeply on the boundaries of the unknown, 2024. 6
- [51] Gladys Tyen, Hassan Mansoor, Victor C˘arbune, Peter Chen, and Tony Mak. Llms cannot find reasoning errors, but can correct them given the error location. arXiv preprint arXiv:2311.08516, 2023. 2, 3, 5
- [52] Han LJ Van der Maas, Lukas Snoek, and Claire E Stevenson. How much intelligence is there in artificial intelligence? a 2020 update. Intelligence, 87:101548, 2021. 2
- [53] Taylor Webb, Zachary Dulberg, Steven Frankland, Alexander Petrov, Randall O’Reilly, and Jonathan Cohen. Learning representations that support extrapolation. In International conference on machine learning, pages 10136–10146. PMLR, 2020. 5
- [54] Penghao Wu and Saining Xie. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13084–13094, 2024. 1
- [55] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019. 2
- [56] Yuhang Zang, Wei Li, Jun Han, Kaiyang Zhou, and Chen Change Loy. Contextual object detection with multimodal large language models. International Journal of Computer Vision, pages 1–19, 2024. 1
- [57] Zhongshen Zeng, Pengguang Chen, Shu Liu, Haiyun Jiang, and Jiaya Jia. Mr-gsm8k: A meta-reasoning benchmark for large language model evaluation. arXiv preprint arXiv:2312.17080, 2023. 3, 5
- [58] Aimen Zerroug, Mohit Vaishnav, Julien Colin, Sebastian Musslick, and Thomas Serre. A benchmark for compositional visual reasoning. arXiv preprint arXiv:2206.05379,

2022. 2

- [59] Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and SongChun Zhu. Raven: A dataset for relational and analogical visual reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5317– 5327, 2019. 2, 5
- [60] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 1
- [61] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models, 2024. 3

[62] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 1

## VERIFY: A Benchmark of Visual Explanation and Reasoning for Investigating Multimodal Reasoning Fidelity

### Supplementary Material

#### 1. Qualitative result

Below is a detailed case study of the question, including reasoning paths from different models and human analysis. We identified a few common mistake patterns: I. Misinterpreting Features: Errors arise from incorrect counting of elements or misjudging numerical patterns. II. Reliance on Intuition: Mistakes occur when relying on surface-level visual trends instead of logical transformations. III. Failure to Verify All Options: Incorrect answers result from selecting seemingly correct choices without checking alternatives.

###### Question

Human Reasoning Path

Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

This is a figure subtraction problem. Upon observing the first set of figures, it can be noticed that subtracting the second figure from the first results in the third figure. Following this pattern, only answer C conforms to this rule.

[Figure 18]

Answer: C Pattern: Logical Operation QVQ Correctness: ✅ Recognition: 3, Abstraction: 5, Deduction: 4

... I think the most plausible explanation is that the sequence is star, triangle left, triangle right, triangle up, and so on, with the question mark being triangle right, which is C. Therefore, the answer is C.

This answer indicates that a wrong reasoning path leads to a correct result. The correct pattern behind the question is related to logical operations, but the model tries to summarize the hidden directional pattern of the sequence of shapes.

Gemini Correctness: ✅ Recognition: 2, Abstraction: 4, Deduction: 5

... Consider the Number of Vertices/Sides:

- • Star (roughly 5 points)
- • Polygon (5 sides/vertices)
- • Triangle (3 sides/vertices) - Option C The number of vertices or sides seems to be decreasing, further supporting option C as the next step in simplification ...

This answer indicates that a wrong reasoning path leads to a correct result. The model concludes that the pattern is that the shapes become simpler and simpler from left to right.

LLaVA-CoT Correctness: ✅ Recognition: 2, Abstraction: 4, Deduction: 3

The pattern in the shapes appears to be based on the number of sides of the polygons. In the first set, the shapes are a circle (0 sides), a square (4 sides), and a star (6 sides). In the second set, the shapes are a pentagon (5 sides), a hexagon (6 sides), and the question mark. The question mark should follow the pattern of increasing the number of sides. Therefore, the next shape should have 7 sides. Comparing the options, option c is a heptagon (7 sides), which fits the pattern.

This answer indicates that a wrong reasoning path leads to a correct result. The model concludes that the pattern is related to the number of sides. Also, there is a hallucination in recognizing the shape.

Figure 1. Incorrect reasoning paths still lead to the correct answer

###### Question

###### Human Reasoning Path

Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

Upon observing the pattern in each row of figures, it is noticed that each figure has three rows of black dots. The first row of black dots decreases by one dot from right to left. Applying this rule to the second row confirms its correctness. Therefore, based on the identified pattern, the answer to this question is option A.

[Figure 19]

Answer: A Pattern: Sequence and Gradient OPENAI-o1 Correctness: ❌ Recognition: 2, Abstraction: 3, Deduction: 1

A succinct way to see why the missing die-face must be the “five-pip” option (B) is to notice that in the bottom row all the other squares show five dots arranged in slightly different orientations. Hence the last square must also be a five-pip face to keep that row consistent. In other words, each row focuses on the same number of pips (top row all “threes,” next row all “fours,” etc.), and the bottom row is all “fives.” That forces the question-mark square to be the five-dot arrangement (option B).

The model struggles with accurately counting the points and, once it determines an answer of five, fails to recognize that none of the four answer choices contain a die with five points. This indicates a limitation in verifying its own conclusions against the provided options.

qwen2.5 Correctness: ❌ Recognition: 8, Abstraction: 7, Deduction: 9

... Following the same pattern, the number of dots should decrease by 1 from the second cell to the third cell. Therefore, the third cell should have 4 dots ... Step 2: Verify the Pattern with Options Now, let's check which option has 4 dots: Option A: 4 dots; Option B: 5 dots; Option C: 2 dots; Option D: 3 dots; Only Option A has 4 dots.

The model partially identifies the correct pattern, but the final answer is determined by chance due to a miscount of the dice points in the answer choices. This mistake leads to an accidental correct selection rather than a fully reasoned solution.

Figure 2. With human reasoning favoring option A due to a decreasing dot pattern. o1 incorrectly identifies the answer due to miscounting and failing to verify against options, while Qwen2.5 accidentally selects the correct answer despite a miscount.

###### Question

###### Human Reasoning Path

Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

It is known that each row of the given figures contains six elements: large white five-pointed stars, small white five-pointed stars, small black five-pointed stars, large black squares, small black squares, and white diamonds. The third row of figures should also contain these elements, while the first two figures already include five of these six elements, leaving only one large black square. Through analysis, it can be determined that only option D's figure is consistent with this requirement.

[Figure 20]

Answer: D Pattern: Sequence and Gradient OPENAI-o1 Correctness: ❌ Recognition: 6, Abstraction: 4, Deduction: 3

A concise way to see why the missing cell must contain one star and one black diamond (option A) is to track how diamonds and stars distribute by row and column:

- • In each row, there are exactly three diamonds total—always two black and one outline. Checking rows 1 and 2 confirms this pattern; row 3 so far has one black diamond in its first cell, one outline diamond in its second, so the missing cell must contain the second black diamond.
- • Likewise, counting stars by row shows that row 1 has three stars in total, row 2 has four, and row 3 already has four from its first two cells (two stars each). To maintain a simple “3, 4, 5” progression in the rows, the last cell in row 3 must add exactly one more star. Hence the final cell must be “one star + one black diamond,” which is choice A.

This answer correctly identifies the method of solving the problem by tracking shape distributions but makes an error in counting. While the approach of ensuring each row contains the required six elements is valid, the specific count of stars and diamonds in row 3 is miscalculated.

qwen2.5 Correctness: ❌ Recognition: 2, Abstraction: 4, Deduction: 1

... Step 3: Determine the Pattern The pattern suggests that each shape in the grid is part of a repeating sequence. Specifically, the third column should follow the same pattern as the first and second columns.

...

This answer correctly attempts to identify a pattern but fails to confirm its validity, leading to the wrong conclusion. While the approach of analyzing rows and columns is appropriate, the identified pattern is incorrect, resulting in an inaccurate choice for the missing shape.

Figure 3. The question asks for the missing figure in a sequence, following a structured pattern of shapes. The correct answer (D) is determined through consistent shape distribution across rows.

###### Question

###### Human Reasoning Path

Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

- Figure 1 can be transformed into
- Figure 2 through a clockwise rotation of 90 degrees, while Figure 2 can be transformed into Figure 3 through left-right flipping. This pattern holds true for the second row as well. Therefore, based on this logic, the figure in the question mark position should be obtained by flipping Figure 2 in the third row from left to right. Only option C satisfies this condition. Hence, the correct answer is indeed C.

[Figure 21]

Answer: C Pattern: Line Property OPENAI-o1 Correctness: ❌ recognition: 0, Abstraction: 2, Deduction: 1

A concise way to see why the answer is A is to notice that in each column the “featured” symbol (T in row 1, C in row 2, arrow in row 3) occupies the same half of the square:

- • In column 1 it always fills the left two quadrants.
- • In column 2 it always fills the top two quadrants.
- • In column 3 it always fills the right two quadrants. Hence for the missing bottom-right cell (row 3, column 3), the arrows must occupy the right-hand half of the square, which is exactly option A.

It only found the most intuitive element relationship, without more logic, and the lack of logic was the biggest problem of its failure.

- Figure 4. A logical pattern recognition problem where the missing figure follows a rotation and flipping rule. o1 incorrectly chose A by relying on an intuitive pattern without full logical reasoning.

Question

Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

Human Reasoning Path

The first figure remains unchanged and is placed at the bottom. Then, by applying different transformations to the first figure - flipping it upside down and placing it on top, rotating it clockwise and placing it on the left, and rotating it counterclockwise before placing it on the right - the third figure is obtained. Applying this rule to the second set of figures corresponds to option B. Therefore, the correct answer is B.

Answer: B Pattern: Geometric Transformation OPENAI-o1 Correctness: ❌ Recognition: 0, Abstraction: 2, Deduction: 1

- • The first figure (a striped triangle) is “one triangle.”
- • The second figure (a striped diamond) can be seen as “two of those triangles” joined along an edge.
- • It follows that the next figure should be “four of those triangles” joined to form a square. Among the choices, only figure A is essentially a square composed of four triangular regions (the diagonals partition it into four triangles), mirroring exactly the “4 triangles” step in the progression. Hence A is the best fit.

Only the shape of the lines is considered, not the geometric relationship between them. Only the dependence between the shape of the triangle and the number of lines is found. These are not inherent logical relationships, indicating that the model does not have the ability of logical reasoning.

[Figure 22]

- Figure 5. The correct answer is determined through geometric transformations, while o1 incorrectly deduces a numerical pattern in shapes.

###### Question

Human Reasoning Path

The elements are composed differently and do not exhibit an obvious attribute rule. Upon observation, it is discovered that the figures in the question stem are all composed of black blocks and white blocks, with an even distribution of colored blocks, giving priority to the area of black and white blocks. The areas of the black blocks in the question stem figures are successively 1/6, 1/5, 1/4, 1/3, and 1/2 of the total area of the figure. Therefore, the black block area of the figure at the ? position should be equal to the entire area of the figure, meaning the entire figure should be completely black. Only option D satisfies this condition.

Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

[Figure 23]

Answer: D Pattern: Sequences and Gradients OPENAI-o1 Correctness: ✅ Recognition: 3, Abstraction: 5, Deduction: 4

A concise way to see the logic is to notice that the figures come in pairs that use the same number of black-white subdivisions but in a different arrangement:

- • The first two each split into six triangles (the hexagon vs. the “folded” version).
- • The next two each split into three triangles.
- • The fifth figure is split into two parts (the black arrow and white rectangle). Hence we expect the sixth figure to be a single undivided shape. Of the answer choices, only D (the solid black rectangle) is one piece, so D is the correct choice.

The reasoning process contains small counting mistakes (number of triangles is incorrect) that, however, do not affect the final reasoning results.

- Figure 6. Pattern-based reasoning puzzle where black area increases progressively; the correct answer (D) is a fully black rectangle, completing the sequence.

Question

Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

Human Reasoning Path

Since the first set increases by 1 each time, the second set should follow the same pattern. The known values in the second set are 6 and 7, so adding another increment of 1 would result in 8. Therefore, the unknown value must be B.

Answer: B Pattern: Line Property OPENAI-o1 Correctness: Recognition: 0, Abstraction: 2, Deduction: 1

A concise way to see why the missing triangle must be option D is to note that, across each row, the shapes keep adding “cuts” (lines) and hence more subdivided regions. In the top row, the third square has the most lines of its trio, so in the bottom row the third triangle should likewise have the most lines of its trio. Among A–D, only D has enough internal lines to continue that increasingcomplexity pattern, so D is the best fit.

It seems that only the number of line segments is simply analyzed here, while the position relationship is ignored, and the inherent logic between lines is not considered.

[Figure 24]

- Figure 7. The missing shape follows an incremental line property, with answer choice B completing the sequence logically.

###### Question

###### Human Reasoning Path

Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

Since the lines in the shapes intersect prominently, we consider the number of intersections. The intersection points in the question stem occur in a sequence of 2, 3, 4, 5, 6, ?, implying that the next shape should have 7 intersection points. Among the options, the number of intersection points is sequentially 18, 8, 7, 8; thus, option C matches this criterion. Therefore, the correct answer is C.

[Figure 25]

Answer: C Pattern: Structural Composition OPENAI-o1 Correctness: ❌ Recognition: 0, Abstraction: 2, Deduction: 1

A succinct way to see why (D) fits is to notice that the sequence keeps returning to “a circle plus a polygon.” The second panel is a circle with an inscribed triangle; the third is a circle partitioned by chords (effectively another polygonal subdivision), and then the fourth panel returns to overlapping circles. The natural “next” shape in that cycle is again a circle with a new regular polygon—in this case, the inscribed square of choice (D). None of the other options continues that alternating circle-plus-polygon pattern. Hence (D) is the best completion.

It does not find geometric relationships, but simply analyzes the relevant properties of the pattern, which goes against basic logic.

- Figure 8. A reasoning-based pattern recognition question where the correct answer (C) follows an intersection count sequence. o1 ’s incorrect answer (D) misinterprets the geometric pattern, focusing on shape alternation rather than structural composition.

Question

Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

Human Reasoning Path

All three figures in the first set are two-dimensional, and the first two figures in the second set are threedimensional. Therefore, it can be inferred that the third figure in the second set should also be threedimensional. Only option A is correct.

Answer: A Pattern: Line Property OPENAI-o1 Correctness: ❌ Recognition: 6, Abstraction: 7, Deduction: 5

A concise way to see the intended pattern is that each 2D figure is being “extruded” into its natural 3D solid. A circle becomes a cylinder, a square becomes a cube, and so an octagon should become an octagonal prism. Of the four answer-choices given, only (B) -- which shows an octagon -- corresponds to that octagonal prism (it is the top/bottom face of the prism). Hence (B) is the correct fill-in for the question mark.

This reasoning process indicates overthinking of the model. It believes the shape of option B is the top/bottom face of a 3D solid instead of a 2D shape.

[Figure 26]

- Figure 9. Illustration of a pattern recognition puzzle where 2D shapes transform into their corresponding 3D forms, with the correct answer (A) requiring a three-dimensional figure. o1 overthinks, selecting (B) based on a misinterpretation of shape extrusion.

###### Question

###### Human Reasoning Path

Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

By observing the first row, it is noticed that the relationship between the first and second figures is a 180° rotation around the vertical axis, and the relationship between the second and third figures is a 180° rotation around the horizontal axis. Therefore, the answer to this question is option C.

[Figure 27]

Answer: C Pattern: Geometric Transformation OPENAI-o1 Correctness: ❌ Recognition: 4, Abstraction: 6, Deduction: 5

A succinct way to see why the missing arrow must point up is to track each column as a simple “turn” of the arrow from one row to the next. For example:

- • Column 1 goes “up” (top row) → “left” (middle row); that is a 90° turn one way (counterclockwise). One more 90° the same way gives “down” for the bottom row.
- • Column 2 goes “up” (top) → “right” (middle); that is a 90° turn the other way (clockwise). One more 90° clockwise gives “down.”
- • Column 3 goes “down” (top) → “right” (middle); again a 90° clockwise turn. One more 90° clockwise from “right” is “up.” Hence the question-mark cell (bottom row, right column) must be an up arrow, which corresponds to choice C.

Although the answer is correct, a closer examination of the analysis reveals that the model fails to correctly interpret the angle of rotation—it cannot accurately distinguish an arrow pointing "left-down" or at 45°. This suggests a limitation in recognizing precise directional changes, leading to potential misinterpretations in similar pattern-based problems.

Figure 10. The correct answer (C) follows a sequence of 180° and 90° rotations, though analysis reveals challenges in accurately interpreting directional changes.

Model response

Extract Reasoning

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

WEB UI

[Figure 33]

Figure 11. Example of extracted reasoning and collaborative annotation interface.

#### 2. Model Comparison

We also include a radar chart for a quick comparison of model performance, as shown in Figure 12. The plot highlights variations across key reasoning and perception dimensions. Notably, models such as Qwen2.5 and GPT-4o exhibit higher agreement scores, suggesting stronger consensus in visual perception. Meanwhile, o1 demonstrates relatively higher abstraction and deduction scores, indicating superior reasoning capabilities. Differences in recognition performance suggest that models vary in their ability to identify key visual elements, which may impact subsequent reasoning stages.

We also present a new data accuracy evaluation in Table 4, which excludes invalid responses and computes the mean of the remaining data. This provides a more accurate assessment of models like QVQ, which utilizes all tokens for reasoning without explicitly producing a final answer.

[Figure 34]

- Figure 12. Holistic comparison of Model Reasoning and Perception Abilities

#### 3. Data Process

Our selection process ensures the highest quality and clarity by following strict criteria. First, we remove all ambiguous questions, excluding those with multiple correct answers or requiring external knowledge beyond the provided visual content. To maintain relevance, we only include data from the past 5-8 years, up to 2024. Additionally, we collect problems from various provinces, prioritizing those known for higher difficulty levels, such as Beijing and Shanghai, to ensure a rigorous dataset. Each question is handcrafted and filtered by annotators with at least a master’s degree or higher, guaranteeing clear and precise final selections.

To ensure a clear and structured reasoning path for each problem, we follow a systematic approach. First, annotators

attempt to solve the problem independently. If successful, we collect a concise description of their solution process. If an annotator is unable to solve the problem, we provide the correct answer as a hint to determine whether a reasoning path can be constructed. If both steps fail, we assess whether the problem is inherently difficult or if it follows an exceptionally rare pattern. Any problems deemed too rare are removed, and only solutions provided by annotators serve as the gold standard for reasoning clarity.

#### 4. Example of Data Extraction

Below, we present the model’s response alongside the extracted reasoning elements, as illustrated in Figure 11. The extracted reasoning captures the core logical structure of the response, ensuring key insights are conveyed effectively. Additionally, we showcase the web-based UI used for collaboration with annotators. We will share these resources with the community to facilitate further research.

#### 5. Analysis of QVQ 5.1. How the reasoning works with visual

The reasoning process in multi-modal models remains an open challenge, particularly in proprietary MLLMs such as o1 , which do not reveal their internal reasoning paths. Meanwhile, weaker open-source models lack the ability to explore different strategies when analyzing visual patterns. In contrast, QVQ exhibits a unique characteristic: it systematically attempts multiple reasoning strategies to solve a given problem rather than following a single fixed approach, as shown in Figure 16.

Our observations indicate that QVQ genuinely explores various paths to reasoning, often exceeding 10 steps in an effort to find the correct answer.

This characteristic differentiates it from other models, as it does not settle on a single reasoning trajectory prematurely. However, this exhaustive approach contributes to its low performance, as the model frequently exceeds the available 16k context length before converging on a final answer. Instead of providing a clear and concise response, QVQ continues its reasoning process until it runs out of tokens, leading to incomplete or inconclusive outputs.

One notable observation is that QVQ consistently uses “Alternatively” to introduce a shift in reasoning, allowing us to identify alternative considerations as distinct reasoning steps. This enables us to analyze the step count and token distribution for each step.

Building on this observation, Figure 13 reveals that while the majority of QVQ tasks require a relatively small number of reasoning steps, a long-tail distribution is evident, with some tasks demanding an excessive number of steps. This suggests that for more complex problems, QVQ struggles to

Model COT GT MA GC LP EO PR SA SC SP ALL Open-Source LMMs qwen2.5 × 0.230 0.179 0.235 0.091 0.240 0.080 0.182 0.115 0.241 0.174 qvq ✓ 0.226 0.172 0.208 0.104 0.290 0.172 0.234 0.268 0.107 0.199 mulberry ✓ 0.233 0.245 0.157 0.154 0.257 0.069 0.173 0.137 0.241 0.192 llava-cot ✓ 0.180 0.186 0.235 0.091 0.253 0.160 0.093 0.216 0.267 0.184 Proprietary LMMs o1 ✓ 0.279 0.236 0.280 0.164 0.293 0.120 0.145 0.196 0.207 0.217 gemini ✓ 0.311 0.189 0.275 0.148 0.280 0.093 0.109 0.115 0.267 0.195 gpt-4o × 0.262 0.178 0.360 0.096 0.200 0.093 0.167 0.255 0.207 0.194

Table 4. Unlike the previous evaluation, which considered all invalid cases as incorrect, this updated assessment excludes invalid cases from the computation. By removing invalid responses from the accuracy calculation, the reported scores better reflect the models’ true performance on valid instances.

[Figure 35]

- Figure 13. Histogram showing the distribution of reasoning steps. The x-axis represents the number of reasoning steps, while the yaxis represents the count of occurrences.

[Figure 36]

Figure 14. Histogram showing the distribution of the length of reasoning steps in words. The x-axis represents the length of reasoning steps, while the y-axis represents the count of occurrences.

Figure 15. Comparison of reasoning step distributions: (a) Number of steps and (b) Length of steps in words.

streamline its reasoning and instead resorts to prolonged deliberation without efficiently converging on a solution. The presence of tasks requiring over 30 or even 50 steps highlights the difficulty of the problem, as the model fails to resolve them within a reasonable step count.

Similarly, Figure 14 illustrates the distribution of token usage per reasoning step. While most steps remain relatively short, the steep long-tail distribution shows that some reasoning steps extend to several hundred or even thousands of tokens. This suggests that QVQ, when faced with difficult problems, often generates disproportionately lengthy reasoning sequences in an attempt to work through them. However, rather than leading to a clear resolution, these extended steps further contribute to inefficiency and excessive token consumption.

Together, these findings indicate that QVQ struggles with complex reasoning tasks, as evidenced by both excessive step counts and highly variable step lengths. The model’s inability to converge efficiently suggests that cer-

tain problems remain unsolved within a reasonable context window. This inefficiency highlights a fundamental limitation in QVQ’s reasoning approach—rather than refining its logic, it frequently resorts to verbosity and fragmented steps, ultimately failing to reach a satisfactory conclusion.

Interestingly, while o1 does not return intermediate reasoning steps, we can infer that it undergoes a similar iterative process. Based on token usage, we estimate that O1 requires an average of 6-8k tokens before reaching a decision. This suggests that proprietary models may employ comparable multi-step reasoning strategies but optimize their token allocation more efficiently than QVQ.

Overall, while QVQ demonstrates an impressive ability to explore diverse reasoning strategies, its lack of an effective stopping mechanism and excessive token consumption hinder its performance. Future improvements should focus on optimizing its reasoning efficiency—potentially by refining the model’s ability to prioritize promising reasoning paths early on and reducing unnecessary exploration.

###### Question

Human Reasoning Path

The composition of the elements is the same, with priority given to positional patterns. The 3x3 grid figures are examined row by row. Upon observing the first row, it is noticed that Figure 1 is transformed into Figure 2 through a clockwise rotation, and then Figure 2 is transformed into Figure 3 through a left-right flip. This pattern is verified in the second row, where the same rule applies. Therefore, the figure in the question mark position should be obtained by flipping Figure 2 in the third row from left to right. Among the four options, only option C satisfies this condition. Hence, the correct answer is C.

Choose the most appropriate option to fill in the question mark so that the figures follow a certain pattern.

[Figure 37]

So I've got this puzzle here, and I need to figure out which option—A, B, C, or D—fits into the question mark in the grid. The grid has three rows and three columns, with various shapes and patterns in each cell. The bottom row has four options, and I need to choose the one that makes sense in the context of the whole grid. First, I should look for patterns or rules that govern how the shapes and patterns are arranged in the grid. Let's look at the first row: • The first cell has three rectangles, each with horizontal lines, and the letters "T" and "I" below them. • The second cell has three rectangles: one with horizontal lines, one with vertical lines, and one with diagonal lines. There are also some symbols below them, but they're not entirely clear. • The third cell has three rectangles: one with horizontal lines, one with vertical lines, and one with diagonal lines, similar to the second cell, but with different symbols below. Moving to the second row: • The first cell has three circles, each divided into different sections with different patterns: one has a curved line, another has a straight line, and the third has a dotted line or something similar. • The second cell has three circles with similar divisions and patterns. • The third cell has three circles with similar divisions and patterns, but again, with different specific symbols or lines within them. Now, the third row is incomplete, with the question mark in the last cell. Before that, there are two cells: • The first cell has a square divided into four smaller squares, each with different patterns and some symbols inside. • The second cell has a similar setup: a square divided into four smaller squares with different patterns and symbols. So, the bottom row shows squares divided into four quadrants, each with their own patterns and symbols. Looking at the options A, B, C, and D: • All options show a square divided into four quadrants. • Each quadrant has a combination of patterns and symbols, such as arrows or leaves. I need to find the option that continues the pattern or follows the rules established in the previous cells of the grid. First, I should try to identify any consistent patterns or rules across the rows or columns. Looking at the first column: • First cell: Three rectangles with horizontal lines and letters "T" and "I". • Second cell: Three circles with different patterns. • Third cell: Square divided into four quadrants with different patterns and symbols. It seems like there's a progression from rectangles to circles to squares, which makes sense as the grid progresses from simpler shapes to more complex ones. Looking at the second column: • First cell: Three rectangles with mixed patterns (horizontal, vertical, diagonal lines) and some symbols. • Second cell: Three circles with similar patterns and symbols. • Third cell: Square divided into quadrants with different patterns and symbols. Again, a similar progression from rectangles to circles to squares. Looking at the third column: • First cell: Three rectangles with mixed patterns and symbols. • Second cell: Three circles with similar patterns and symbols. • Third cell: Square divided into quadrants with different patterns and symbols. Same pattern continues. Now, focusing on the third row, specifically the squares divided into quadrants. In the third row, the first two cells have squares with four quadrants, each containing different patterns and symbols. Looking closely at these quadrants: • In the first cell, the quadrants have patterns like horizontal lines, vertical lines, diagonal lines, and maybe another type of pattern, each combined with a small symbol inside. • In the second cell, similar setup with different patterns and symbols. So, the question is, what should the third cell in the third row look like? Given that the previous cells in the row have squares divided into quadrants with specific patterns and symbols, the options A, B, C, and D all fit that general description. Therefore, I need to look for more specific patterns or rules that differentiate between these options. Let's consider the patterns within the quadrants: • Option A: Quadrants have patterns like horizontal lines, vertical lines, diagonal lines, and perhaps a curved line or another distinct pattern. • Option B: Similar patterns, but the arrangement or the specific symbols might differ. • Option C: Again, similar patterns with possible variations in arrangement or symbols. • Option D: Same as above. I need to find a pattern that matches the established rule in the grid. Perhaps, the patterns in the quadrants follow a specific sequence or rotation. Looking at the first two cells of the third row: • In the first cell, the quadrants have patterns arranged in a certain order. • In the second cell, the patterns are similar but possibly rotated or mirrored. If there's a consistent rule, perhaps the patterns are rotating 90 degrees with each cell. Alternatively, maybe the symbols inside the quadrants are following a specific sequence. Another thought: perhaps the patterns in the quadrants correspond to the patterns in the previous rows. For example, the first row has rectangles with horizontal lines, so maybe horizontal lines represent rectangles, and so on. But that might be too complicated. Alternatively, perhaps the symbols inside the quadrants relate to the letters or other symbols in the previous cells. However, without knowing the exact meanings of all the symbols, that might not be helpful. Let's consider the positions of the symbols within the quadrants. In option A, the symbols are positioned in a certain way within the quadrants. Similarly, in options B, C, and D, the symbols are positioned differently. Perhaps the positions follow a specific pattern or rotation. For instance, if in option A, the symbols are all pointing in one direction, and in option B, they're pointing in another direction, that could be a clue. Looking at the symbols, they seem to be arrows or leaves pointing in different directions. In option A, the arrows point upwards, left, right, and downwards. In option B, they might point in a different sequence. Similarly, options C and D have their own sequences. Perhaps the sequence of arrow directions follows a specific pattern or rotation. If, for example, the arrows are rotating 90 degrees clockwise with each cell,

- then option B might be the correct choice. Alternatively, if the sequence is following a specific order, such as up, right, down, left, repeating, then option C might fit. But I need to look back at the grid to see what the previous cells have. Looking at the third row, first cell: arrows pointing up, right, down, left. Second cell: arrows pointing right, down, left, up. Third cell: should follow a certain rule based on this pattern. If the arrows are rotating 90 degrees clockwise with each cell, then the third cell should have arrows pointing down, left, up, right. Looking at the options: • Option A: up, right, down, left • Option B: right, down, left, up • Option C: down, left, up, right

• Option D: left, up, right, down So, if the rotation is 90 degrees clockwise each time, then the first cell has up, right, down, left; the second cell has right, down, left, up; and the third cell should have down, left, up, right. That would correspond to option C. Alternatively, if the sequence is simply shifting positions, like a cyclic permutation, then option C would fit. However, I need to confirm if this is the only possible rule or if there are other patterns at play. Another possible rule could be that the patterns in the quadrants alternate in a certain way. For example, horizontal lines alternate with vertical lines, or diagonal lines alternate in direction. But looking at the options, all have a mix of patterns, so that might not be the distinguishing factor. Alternatively, perhaps the symbols inside the quadrants follow a specific pattern based on their shapes or types. For instance, maybe certain symbols always appear in quadrants with specific patterns. But without knowing the exact meanings of the symbols, that's hard to determine. Another angle: perhaps the grid is following a mirror image pattern, either horizontally or vertically. If I look at the first two cells in the third row, the patterns in the quadrants might be mirrored in some way. However, given that the quadrants themselves are arranged in a specific order, a simple mirror might not apply. Alternatively, perhaps the patterns are being inverted or negated in some way. But that seems less likely. Let's consider the possibility that the patterns are being combined or overlapped in some fashion. For example, if two adjacent quadrants share a common pattern, that might indicate a specific relationship. But again, without more information, that's speculative. Given that, perhaps the simplest explanation is the rotation or shifting of the arrow directions. If the arrows are rotating 90 degrees clockwise with each cell,

- then option C would be the correct choice for the third cell. Alternatively, if the sequence is being reversed or arranged in a different order, that could also be a possibility. However, based on the pattern observed in the first two cells of the third row, the rotation seems plausible. Therefore, I would tentatively select option C as the most appropriate choice for the question mark. To ensure that this is the correct choice, I should check if there are any other patterns or rules that could point towards a different option. For example, perhaps the patterns in the quadrants are being modified based on the row or column positions. Alternatively, maybe the symbols inside the quadrants are following a specific sequence unrelated to the arrow directions. However, without additional context or information about the symbols, it's difficult to determine any more complex rules. Therefore, based on the visible pattern of arrow directions rotating 90 degrees clockwise with each cell, option C is the most logical choice for the third cell in the third row. Final Answer [ \boxed{C} ]

Figure 16. The example reasoning path of QVQ shows that QVQ needs extensive tokens to solve the problem.

