# arXiv:2512.11995v2[cs.CV]9Jun2026

## V-REX: Benchmarking Exploratory Visual Reasoning via Chain-of-Questions

###### Chenrui Fan⋆,1, Yijun Liang⋆,1, Shweta Bhardwaj⋆,1 Ming Li1, Kwesi Cobbina1, Tianyi Zhou2

{cfan42,yliang17,shweta12}@umd.edu Project: https://github.com/tianyi-lab/VREX ⋆These authors contributed equally to this work.

1University of Maryland, College Park 2Mohamed bin Zayed University of Artificial Intelligence

Abstract. While many vision language models (VLMs) are developed to passively answer well-defined, straightforward questions with highly specified targets, as in most benchmarks, they often struggle in practice with complex open-ended tasks, which usually require multiple rounds of active exploration and reasoning in the visual space. Such visual thinking paths not only provide step-by-step exploration and verification as an AI detective but also produce better interpretations of the final answers. However, these paths are challenging to evaluate due to the large exploration space of intermediate steps. To bridge the gap, we develop an evaluation suite, “Visual Reasoning with multi-step EXploration (V-REX)”, which is composed of a benchmark of challenging visual reasoning tasks requiring native multi-step exploration and an evaluation protocol. V-REX casts the multi-step exploratory reasoning into a Chain-of-Questions (CoQ) and disentangles VLMs’ capability to (1) Planning: determining the exploratory direction to gather information at each step for solving the task; and (2) Following: answering curated CoQ sequentially to collect information for deriving the final answer. By curating finite options of questions and answers per step, V-REX achieves a reliable quantitative and finegrained analysis of the intermediate steps. By assessing SOTA proprietary and open-sourced VLMs, we reveal consistent scaling trends, significant differences between planning and following abilities, and substantial room for improvement in multi-step exploratory reasoning.

##### 1 Introduction

Various practical applications of vision language models (VLMs) need to perform sophisticated multi-step visual reasoning [2, 6, 19, 20, 22, 43, 45, 48] to solve the user queries. Recent studies reveal the weakness of existing VLMs on exploratory reasoning tasks, showing that they often rely on brute-force search in the input image to allocate the potential objects of interest, and rarely adjust their plans to be adaptive to collected clues [4, 18, 32]. This weakness substantially limits the application of VLMs in challenging open environments where the goals cannot be fully

Planning Task Following Task

### V-REX

[Figure 1]

|What is the black car doing?<br><br>How many cars are visible?<br><br>Is the ground wet or dry?<br><br>What color is the sign?|
|---|

What is the black car doing?

Original Question: Who is mainly responsible for the accident?

Tasks

[Figure 2]

[Figure 3]

|Backing<br><br>Driving Forward<br><br>Parking<br><br>Turning Left|
|---|

Backing 6 Wet Red

[Figure 4]

|What is the sign about?<br><br>Is it still raining?<br><br>Are there people walking?<br><br>What is the silver car doing?|
|---|

What is the silver car doing?

[Figure 5]

|Turning Right<br><br>Parking<br><br>Driving Forward<br><br>Backing|
|---|

Driving Forward

Not sure Not Sure No

CoQ GT Question Based on the previous results, who is mainly responsible for the accident? CoQ GT Answer

Distractor Question Distractor Answer

[Figure 6]

Following Evaluation Step Planning Evaluation Step

|The black car The silver car|
|---|

[Figure 7]

Correct Answer for Distractor Question

- Fig. 2: An example from V-REX with corresponding planning and following tasks. In the planning task, the model is given the original question and asked to select a sub-question in each step that is necessary and helpful for solving the original question. In the following task, the model is asked to answer the ground truth sub-questions step-by-step.

specified at the very beginning but need progressive planning on the fly. For example, guessing the location based on a street view image [22], detecting cheating from posted images [18], or simply complicated tasks [34]. Tasks like these require multiple rounds of active exploration, sub-goal proposal, and answering the subquestions to collect sufficient contextual clues and identify the final targets, while poor exploration may significantly undermine or distract such reasoning processes.

However, recent visual reasoning and benchmarks mainly focus on math problems with visual contexts [23, 44, 46] or puzzle games with toy environments [24, 33, 37], which can be addressed by single-round QA or language space reasoning. These questions are usually straightforward with highly specified instructions or targets. In addition, their input prompt and context are sufficient for the models to make a concrete plan in advance and execute it without further exploration. Hence, they do not heavily rely on exploration skills, which, in contrast, are critical to open-ended tasks that need to actively collect clues given only high-level initial intents. Moreover, most of them merely evaluate the final answers and do not investigate the intermediate exploration steps, which can reflect important capabilities or fundamental flaws, such as shortcut solutions or redundancy, making the

Original Question

- Question-1

Answer-1

Answer-2

- Question-2

Question-3 (Final Question)

Answer-3 (Final Answer)

Planning Task

Ground Truth QA-Chain

Following Task

|Distractor Answers<br><br>Distractor Questions<br><br>Correct Answers for Distractor Questions<br><br>CoQ GT Questions<br><br>CoQ GT Answers<br><br>Model Choice<br><br>Provided Choice|
|---|

Fig. 1: Overview of Chain-of-Questions (CoQ). The left represents the manually formulated ground truth QA chain. The middle represents the Planning task, evaluating the model’s capability in selecting sub-questions that are helpful to answer the original question. The right represents the Following task, evaluating the model’s capability in answering each sub-question.

Flowchart Pattern Property Relationship

#### V-REX

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

|[Figure 12]|
|---|

Deduction Guessing

Scenarios

How many people participated in the survey feel they are struggling due to wage freezes?

What is the complexity of the algorithm?

Counting

Navigation Retrieval

[Figure 13]

In which area of the picture does the liquid flow fastest?

Which option would fit into the missing part the best?

Responsibility Intention Location Time Topic

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

How many Golden State Warriors players are positioned in or just around the paint area?

Based on the cover, what is the subject of the book?

Who is mainly responsible for the accident?

What is the player in white jersey at the back post trying to do?

What is the location of the picture?

What year was this photo taken?

Map GUI Traffic Trend Word Puzzle

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

A passenger arrives at Gates E20-E28 and wants to …. Which of these routes most efficiently incorporates these stops before reaching the Sunflower Garden?

What sequence of events and current situation best explains the child's state?

What is the most likely primary goal of the user at this moment?

Is the driver permitted to make a right turn at this intersection?

Which word is in this puzzle?

- Fig. 3: Scenarios in V-REX. With various samples, V-REX spans 15 real-world scenarios across 4 reasoning categories (Deduction, Guessing, Navigation, & Retrieval), covering diverse settings such as diagrams, time estimation, GUI interpretation, and others.

whole evaluation suite a black box. Hence, it is still unclear how each step affects the subsequent reasoning and the final answer, how robust existing VLMs are to intermediate-step errors, and whether they can recover from these errors.

To bridge the gap, we investigate the quality and impact of the intermediate exploration steps and develop a novel evaluation suite, “Visual Reasoning with multi-step EXploration (V-REX)”, which is composed of challenging visual reasoning tasks requiring native multi-step exploration and subgoal decomposition. Specifically, we formulate the exploratory reasoning process in a novel form called “Chain-of-Questions (CoQ)”, which represents a sequence of interconnected sub-questions and corresponding answers that serve as the precondition for the final answer, as shown in Figure 1 (left). In CoQ, a model is prompted to ask a sequence of sub-questions and answer them sequentially to collect sufficient context, thereby creating a QA chain progressively on the fly to address the original question. It is worth clarifying that CoQ is not a prompting method(e.g. CoT) to improve models’ performance, but a way to conduct finegrained evaluation of the exploratory visual reasoning ability of model.

A key challenge of evaluating multi-step exploratory reasoning is the prohibitively large space for free exploration of VLMs. CoQ can reduce it to a finite space with limited choices of questions and answers per step, providing a tractable and controllable assessment. Specifically, CoQ enables us to disentangle the reasoning capabilities into Planning and Following when facing complex tasks that require multiple rounds of exploration with sub-goal proposals. The Planning of CoQ evaluates the model’s exploratory ability to determine effective directions by selecting informative subquestions that guide information gathering toward the final answer. For example, in Figure 2 (middle), when given the question “Who is mainly responsible for the accident?”, the model should raise exploratory questions about the conditions of the two vehicles, rather than questions about the number of

vehicles. The Following of CoQ evaluates the model’s ability to answer the subquestions to collect sufficient contextual clues and identify the final targets, as shown in Figure 2 (right). An overview of CoQ is shown in Figure 1, where the two dimensions serve as the complementary parts of the exploratory reasoning process, indicating a holistic and fine-grained diagnosis of the reasoning capabilities of VLMs.

V-REX comprises 702 samples and 2,504 questions spanning 4 reasoning categories and 15 scenarios as shown in Figure 3. Each sample contains 2 to 6 reasoning steps, with 3.57 steps on average. In our experiments, we extensively evaluate VLMs of different model families, ranging from open-source to proprietary models, including relatively small models and larger ones. The evaluation lead to some unrevealed observations and shed novel insights to improving the exploratory reasoning capability of VLMs.

Contributions: We introduce V-REX, the first benchmark for assessing the multi-step exploratory reasoning capability of VLMs with a novel form of Chainof-Questions (CoQ). In our setting, the model’s planning and following abilities can be disentangled and evaluated separately, providing a more fine-grained and interpretable evaluation of the reasoning capabilities of VLMs.

###### Key Findings:

- 1. By following the hints of CoQ, VLMs consistently achieve better performance on final questions, demonstrating the importance of exploration in visual reasoning.
- 2. The scaling law holds for V-REX, and models of the same size show much less variance in Following performance than in Planning performance; this indicates a need to improve VLMs’ exploratory abilities.
- 3. Following and Planning capabilities both contribute positively to the overall performance of the model.
- 4. Smaller models are better at Following than Planning while larger models have more balanced performance.
- 5. VLMs are better at recovering from failed Planning than from failed Following.

##### 2 Related Work

###### 2.1 Exploratory Visual Reasoning

Inspired by recent advances on the reasoning ability of LLMs [14, 27], efforts have been made to replicate similar success on the multimodal domain. [36, 38, 49] adopt GRPO algorithm to train R1-style VLMs. [15] designs an effective training pipeline to elicit VLM’s ability to explore in visual space and general long and complex chain-of-thought. Meanwhile, multiple benchmarks have been proposed to evaluate the visual reasoning ability of VLMs. VisuLogic [46] measures the ability of VLMs on genuine vision-centric reasoning tasks. ZeroBench [34] evaluates VLMs on extremely difficult tasks. CaughtCheating [18] investigates the VLM’s capability on cheating detection tasks, revealing the weakness of existing models on real detective-level tasks. More benchmarks evaluate VLMs’ visual reasoning abilities from different aspects, including cognitive reasoning [31], visual grid

reasoning [33], visual comparison reasoning [3], cultural reasoning [35], multiimage reasoning [9], and color-related reasoning [21]. However, although these benchmarks cover a wide range of visual reasoning tasks, they mainly focus on the exploration skills of VLMs on answer space and neglect the exploration of question space, that is, the model’s ability to propose helpful sub-questions, set subgoals that would guide the reasoning process effectively. Our benchmark, V-REX, fills the gap by evaluating both exploration of answer space and question space in a disentangled manner.

- 2.2 Evaluation with Intermediate Steps

Most visual reasoning benchmarks [9, 31, 34, 46] evaluate the ability of VLMs in end-to-end approach. Given complex visual reasoning tasks, the models are prompted to directly produce the final answer, and performance is measured solely by answer accuracy. Chen et al. [5] introduces a process reward model (PRM) to score intermediate reasoning steps. But they rely on Monte-Carlo estimation and LLM-as-a-Judge as PRM for intermediate steps, which are not reliable for benchmark evaluation. LlamaV-o1 [39] adopts a fine-grained evaluation framework that compares each generated reasoning step to a ground-truth trace, but it also relies on LLM-based judgments when assessing alignment or correctness. By breaking down the complex visual reasoning tasks into multiple sub-tasks and evaluating the intermediate steps, V-REX can more accurately assess the ability of VLMs to reason step-by-step in Planning and Following dimensions.

- 3 Exploratory Reasoning in Visual Space

- 3.1 Chain-of-Questions (CoQ)

To investigate the intermediate exploratory reasoning processes of vision language models (VLMs), we conceptualize their reasoning pathways in the CoQ format as a QA chain with a sequence of interconnected sub-questions Qt and corresponding answers AQ

. The reasoning process can then be denoted as: {(Q1,AQ

t

)}, where T is the number of reasoning steps. This formulation helps formulate the model’s exploratory reasoning process as an iterative alternation between generating exploratory sub-questions and providing corresponding answers. It allows us to decouple the model’s reasoning capability into two fundamental dimensions: the proficiency in formulating informative intermediate questions (Planning), and the accuracy in addressing these sub-questions (Following). However, evaluating these abilities in the infinite exploratory space is not feasible. It is challenging to design a reliable evaluation protocol because it involves directly rating the quality of open-ended reasoning processes. To address this, we propose to evaluate the model’s ability to answer a series of multiple-choice questions (MCQ) with the CoQ process, reformulating the evaluation into a more tractable and controllable assessment in finite space.

###### ),(Q2,AQ

),...,(QT,AQ

1

2

T

###### 3.2 Planning of CoQ

An important dimension of exploratory reasoning capability is Planning: the capacity to strategically chart a course of exploration through the reasoning space. Effective planning entails the model’s ability to dynamically adapt its sequence of inquiry based on evolving intermediate evidence and to accurately select the most informative next step at each stage. As illustrated in the middle of Figure 2, we branch over the question node at each step of the pre-defined ground-truth QA chain by adding several uninformative, distracting questions. The planning evaluation then becomes asking the model p to identify the helpful sub-question at step t conditioned on the previous CoQ history:

Qt ∼ p(·|Qorigin, Q1, A∗Q1, Q2, A∗Q2, . . . , Qt−1, A∗Qt−1) (1)

where Qorigin is the original problem, Qt is the question selected by model at step t, and A∗Q

is the ground-truth answer for Qt. For simplicity, we omit the notation of the image in the above equation. By directly providing the corresponding ground-truth answer A∗Q

t

to the model at each step, we ensure that the model does not need to solve the sub-questions itself but only needs to decide which question to ask next. This design explicitly isolates the Planning ability from Following.

t

###### 3.3 Following of CoQ

For the Following dimension, our objective is to evaluate the model’s ability to accurately answer intermediate questions based on both visual cues from different regions of the image and knowledge obtained through preceding reasoning steps. Specifically, we present the VLM with a pre-constructed, sequential chain of QA steps and require the model to provide answers at each stage en route to the final conclusion. As depicted in the right part of Figure 2, we introduce distractor options at each answer node, transforming each step into a multiple-choice setting. Specifically, at reasoning step t in the chain, we augment the correct answer with several incorrect alternatives. The model p is tasked with selecting the correct answer to question Q∗t at t step, conditioned on the image and the accumulated conversation history:

, . . . , Q∗t−1, AQ∗

, Q∗t) (2)

, Q∗2, AQ∗

t ∼ p(·|Q∗1, AQ∗

AQ∗

t−1

2

1

is the corresponding model-generated response. This methodology enables a principled evaluation of the model’s capacity to faithfully follow an intended reasoning path.

where Q∗t is the ground-truth question at each step and AQ∗

t

##### 4 V-REX Benchmark

V-REX consists of 702 samples and 2,504 questions, with each sample containing 2 to 6 reasoning steps and 3.57 steps on average. The detailed distributions of questions and reasoning steps are illustrated in Figure 4. The dataset spans

- 4 reasoning categories and 15 application scenarios, supporting a fine-grained evaluation of VLMs’ exploratory reasoning capabilities.

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

- Fig. 4: Statistics of V-REX, including question distributions (left) and the distribution of reasoning steps (right).

###### 4.1 Taxonomy

To thoroughly evaluate exploratory reasoning, V-REX defines 4 reasoning categories where exploration plays a central role: Deduction, Guessing, Navigation, and Retrieval. Each category reflects a different way VLMs explore and reason across multi-step problems.

Deduction category measures model’s ability to discover, infer, and apply logical or causal relationships from structured information. It includes 4 application scenarios: (i) Flowchart, (ii) Pattern, (iii) Property, and (iv) Relationship. Guessing category involves uncertainty and incomplete information, requiring models to infer hidden or missing factors through hypothesis exploration. It includes 5 scenarios: (i) Responsibility, (ii) Intention, (iii) Location, (iv) Time, and (v) Topic. Navigation category measures the model’s ability to explore by traversing spatial layouts or procedural paths, planning step-wise movements through the scene while maintaining global consistency. It includes 4 scenarios: (i) Map, (ii) GUI, (iii) Traffic, and (iv) Trend. Retrieval category measures the model’s ability to locate, gather dispersed information by thoroughly exploring the visual input. It includes 2 scenarios: (i) Counting and (ii) Word Puzzle.

###### 4.2 Data Curation

We begin by elaborating on the methodology for constructing the ground-truth QA chains for each sample, followed by an explanation of how these chains are systematically adapted to probe the distinct following and planning capabilities of VLMs within the reasoning space.

Ground-Truth QA Chain As discussed in Section 3, we use QA chains to model the reasoning traces of VLMs when solving multi-step reasoning problems. We denote the ground-truth QA chain as a sequence {(Q∗1,A∗Q∗

)},

###### ),(Q∗2,A∗Q∗

),...,(Q∗T,A∗Q∗

1

2

T

where Q∗T is the final question and A∗Q∗

is the final answer, a valid QA chain should have the following properties:

T

- – Helpfulness. ∀t ∈ [1,T − 1], ∃t′ > t, A∗Q∗

t

is helpful for answering Q∗t

2

. It means that every question should be helpful for the following questions.

- – Correctly Ordered. ∀t ∈ [2,T], ∀t′ < t, Q∗t′ does not depend on A∗Q∗

. It means that every question should not depend on its subsequent question.

t

We rely on human experts to collect high-quality ground-truth QA chains on visual reasoning tasks across all categories. The images are sourced from websites and publicly available benchmarks, with detailed sources provided in Appendix A. Specifically, five PhD-level annotators are initially assigned three scenarios each for ground-truth QA chain construction. We then have two rounds of cross-verification. For each round, annotators are randomly shuffled to three other unseen scenarios. They are required to ensure the validity of the reasoning chain, helpfulness of the intermediate steps, and make sure they are in the correct order. At the end of each verification round, they would give feedback to the original annotator and the previous verifier.

Planning Task To evaluate VLMs’ exploratory ability in the question space, we construct samples for the Planning dimension based on ground-truth QA chains. We develop a two-stage automatic generation pipeline, consisting of Distractor Construction and Question Integration, which leverages LLM for generation, filtering, and integration.

Distractor Construction stage aims to create alternative questions that are contextually relevant but unhelpful for solving the final question. We design two complementary strategies to capture local and global variability in reasoning: step-level and chain-level distractor generation. In step-level generation, LLM (GPT-5) produces distractors for each reasoning step that are locally plausible but deliberately misleading. In contrast, chain-level generation considers the entire distracting QA chain holistically. These chains are self-consistent yet subtly deviate from the ground truth, allowing evaluation of whether models can distinguish globally misleading reasoning paths from the true reasoning chain. Question Integration stage aims to integrate the generated distractors with the groundtruth chain into a unified dataset through automatic filtering and refinement, creating a controlled benchmark for evaluating models’ exploration reasoning in complex question spaces. More details are provided in the Appendix C.

Following Task For Following dimension, we manually create a few plausible distractor choices for each answer step. These distractors are combined with the ground-truth answer to form a series of multiple-choice questions.

###### 4.3 Evaluation Metrics

We use intermediate accuracy to measure both Following and Planning ability of the model. Specifically, for a QA chain of T steps, Planning ability is quantified as:

T−1

1 T − 1

I[Q∗t = Qt] (3)

t=1

where Q∗t is the ground-truth question at step t and Qt is the model’s chosen question at step t. Similarly, Following ability is quantified as:

T−1

1 T − 1

###### I[A∗Q∗

t ] (4) where I[·] is the indicator function, AQ∗

= AQ∗

t

t=1

is the model’s chosen answer for the ground-truth question Q∗t at step t, and A∗Q∗

t

is the ground-truth answer.

t

##### 5 Results

###### 5.1 Main Results

Table 1 summarizes VLM performance across the Following and Planning tasks, revealing consistent scaling trends and distinct strengths across model families. Here, the performance refers to Following and Planning abilities. Overall, model performance increases steadily with size. Larger models show stronger multimodal reasoning and coordination capabilities. GPT-5 and o3 achieve the highest average performance on both tasks. Notably, the performance gap between large open-sourced and proprietary models has narrowed considerably. Several large open-source VLMs match or even exceed proprietary models in some reasoning categories, particularly Deduction and Navigation.

While zooming into different reasoning categories, the result reveals that VLMs exhibit distinct areas of expertise across model scales. Smaller models (<7B and 7B–10B) show highly variable strengths, with some excelling in Deduction or Guessing and others in Navigation or Retrieval. As the model size increases beyond 10B, the expertise distribution becomes more concentrated. On the Following task, InternVL2.5-38B achieves the best accuracy in 3 of 4 categories, and InternVL3-38B similarly dominates in 3 categories on the Planning task. Proprietary models such as GPT-5 and o3 also achieve top accuracy across multiple categories. These patterns suggest that larger models develop more unified and generalizable reasoning capabilities, rather than relying on specialized strengths in isolated categories.

###### 5.2 Main Findings Finding 1

By following the hints of CoQ, VLMs consistently achieve better performance on final questions, demonstrating the importance of exploration in visual reasoning.

###### Table 1: Performance of various VLMs (grouped by size) on V-REX. The best performance in each VLM group is highlighted in bold.

###### Planning Following

Model Deduction Guessing Navigation Retrieval Average Deduction Guessing Navigation Retrieval Average VLMs: <7B

LLaVA-OV-1B [17] 31.5 37.1 27.9 43.9 35.1 33.8 49.3 64.0 56.1 50.8

- InternVL3-1B [50] 37.1 47.3 42.5 42.4 42.3 37.4 50.2 67.1 61.0 53.9

- InternVL3.5-1B [42] 42.4 43.5 42.2 41.1 42.3 44.9 45.2 69.8 62.2 55.5 Qwen3-VL-2B-IT [47] 31.9 37.5 34.1 28.4 33.0 50.2 61.0 80.3 48.0 59.9 Qwen3-VL-2B-Think [47] 44.4 60.6 48.4 41.9 48.8 39.2 51.6 72.9 48.4 53.0

InternVL3-2B [50] 52.7 53.8 64.1 36.9 51.9 51.6 60.8 80.7 61.0 63.5

- InternVL3.5-2B [42] 63.1 60.8 63.2 42.7 57.5 59.3 51.5 71.3 60.2 60.5 Qwen2.5-VL-3B [1] 47.9 59.3 51.0 37.3 48.9 61.2 57.7 83.4 66.7 67.2 Qwen3-VL-4B-IT [47] 62.5 65.4 70.9 36.8 58.9 55.7 63.8 81.6 56.9 64.5 Qwen3-VL-4B-Think [47] 58.8 83.4 63.6 42.2 62.0 56.2 57.8 82.7 60.2 64.2 InternVL2.5-4B [8] 62.5 57.9 69.0 55.4 61.2 55.3 60.9 77.6 65.9 64.9 InternVL3.5-4B [42] 69.6 65.4 66.8 51.8 63.4 63.7 58.7 81.6 72.4 69.1

VLMs: 7B−10B

LLaVA-OV-7B [17] 53.0 42.7 61.0 35.4 48.0 58.6 62.1 71.9 74.0 66.6 Qwen2.5-VL-7B [1] 59.1 67.6 64.7 36.7 57.1 65.8 64.5 81.0 73.2 71.1 Qwen3-VL-8B-IT [47] 74.7 84.7 80.0 46.8 71.6 59.2 64.2 88.3 57.7 67.3 Qwen3-VL-8B-Think [47] 66.8 86.7 69.7 48.7 68.0 63.0 60.3 84.9 58.5 66.7 InternVL2.5-8B [8] 70.1 75.1 71.8 50.6 66.9 57.9 59.9 81.4 66.7 66.5

- InternVL3-8B [50] 53.5 44.5 66.8 40.1 51.2 54.9 58.8 81.6 69.1 66.1 InternVL3.5-8B [42] 66.2 69.6 73.1 47.3 64.1 55.6 61.0 82.8 61.8 65.3

- InternVL3-9B [50] 69.1 81.6 84.1 57.5 73.1 56.0 66.3 82.9 69.1 68.6 VLMs: >10B

InternVL3-14B [50] 76.4 86.4 79.8 56.7 74.8 68.8 65.9 85.7 69.1 72.4 InternVL3.5-14B [42] 78.6 73.0 77.9 53.3 70.7 64.9 62.3 85.1 72.4 71.2 InternVL2.5-26B [8] 76.4 71.1 79.9 52.6 70.0 64.4 61.0 85.5 69.9 70.2 InternVL2.5-38B [8] 85.6 85.0 85.0 59.8 78.8 71.3 69.0 90.8 74.8 76.5 InternVL3-38B [50] 83.2 91.3 86.6 65.2 81.6 68.1 65.2 91.8 69.1 73.5 InternVL3.5-38B [42] 81.7 85.3 84.3 58.0 77.3 65.9 63.0 84.1 64.2 69.3

VLMs: Proprietary

- GPT-4o [30] 77.9 69.6 79.7 63.7 72.7 73.1 78.0 87.1 67.5 76.4

- GPT-5 [28] 93.9 91.8 93.9 58.1 84.4 76.6 86.2 90.9 84.6 84.6 O1 [27] 94.4 93.2 92.4 61.3 85.3 73.0 80.4 90.7 74.0 79.5 O3 [29] 94.8 94.4 93.3 62.1 86.1 80.2 83.0 89.9 83.7 84.2 Gemini 2.0 Flash [13] 86.2 93.4 85.9 54.6 80.0 74.3 67.6 88.7 78.9 77.4 Gemini 2.5 Flash [11] 87.8 89.7 84.5 49.6 77.9 69.8 74.2 87.1 78.0 77.3

To assess the effect of the hints in CoQ in VLMs, we compare the accuracy performance on final questions with and without these hints across all models. For quantitative analysis, we compute the performance changing ratio for each VLM and reasoning category as (AccCoQ − Acc)/Acc. The detailed accuracies on 4 categories for each VLM can be found in Appendix E. Figure 5 illustrates the distribution of performance changing ratios across all models for the Following and Planning tasks. Most ratios are positive, demonstrating that the model achieves higher accuracy on final questions when intermediate steps are introduced. This indicates that the partial hints we provide to the models on both tasks (i.e., questions in Following task and answers in Planning task) help VLMs in reasoning, which justifies the validity of our manually constructed ground-truth QA chains.

The Retrieval category shows relatively minor improvements in both tasks, likely because it depends more on factual recall and direct visual matching than on complex reasoning. Consequently, the CoQ process contributes less to tasks where success relies primarily on precise information retrieval rather than the

###### Planning Task

Following Task

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.8

PerformanceChangingRatio

0.6

0.4

0.2

0.0

0.2

0.4

0.6

DeductionGuessingNavigationRetrieval

DeductionGuessingNavigationRetrieval

- Fig. 5: The ratio of change on final accuracy brought by CoQ evaluation across all VLMs. The X-axis denotes task categories, and the Y-axis represents the performance changing ratio.

hierarchical organization of reasoning processes. Another possible reason is that the human-designed reasoning paths may diverge from the model’s intrinsic retrieval strategy. This mismatch suggests that while hints in CoQ effectively help structures with deductive or sequential reasoning, it may not optimally support tasks dominated by information lookup or dense visual-textual association.

Moreover, when comparing the Following and Planning tasks, a few VLMs exhibit decreased performance when provided with CoQ hints across all categories. For some models, the introduction of intermediate steps can generate distractive reasoning paths or omit essential reasoning cues in Planning task settings, ultimately disrupting their decision-making process. As a result, instead of improving reasoning consistency, adding exploration steps may introduce additional cognitive noise that hinders accurate question or answer selection. The failure cases are shown in Appendix J.

###### Finding 2

Scaling laws persist on V-REX, while performance variance among models of the same size is smaller in Following than in Planning.

To investigate whether the general scaling law holds on our Following and Planning tasks, we plot the model’ abilities against parameter size for both tasks. As in Figure 6, colors denote the sample categories and marker shapes represent different model families. We also use dashed lines with shaded bands to aggregate the ability of models of the same scale. We observe an upward ability trend as the model size increases, suggesting that the scaling law continues to hold. However, the ability variance among model families of the same size is smaller for the Following task than for the Planning task, indicating that the Following ability is more stable and converged across model families. The relatively uniform following ability across models indicates that, given a well-structured question, models of the same size can effectively leverage visual cues and contextual information to

###### Planning Task

###### Following Task

0.9

0.9

0.8

0.8

Performance

Performance

0.7

0.7

0.6

0.6

0.5

Category

Model Family

0.5

Deduction

- Internvl2.5

- Internvl3

- Qwen2.5 Vl Instruct

- Qwen3 Vl Thinking

0.4

Guessing Retrieval Navigation

0.4

Internvl3.5

Qwen3 Vl Instruct

0.3

Llava Ov

1 2 3 4 7 8 9 14 26 38

1 2 3 4 7 8 9 14 26 38

Model Size (Billions of Parameters)

Model Size (Billions of Parameters)

- Fig. 6: Following and Planning ability on models of different sizes (logarithmic scale for the x-axis). Overall, the model’s ability on both tasks positively correlates with model size. Notably, the variance of following ability among same-sized models is smaller than that of Planning ability.

reach similar outcomes. This finding underscores the critical role of Planning as a distinguishing factor in multi-step visual reasoning, particularly for problems that require more open-ended exploration strategies.

Finding 3 Following and Planning abilities both contribute positively to the overall performance of the model.

0.4 0.5 0.6 0.7 0.8

Planning Ability

0.4

0.5

0.6

0.7

0.8

OverallPerformance

R=0.858 p<1e-4

0.5 0.6 0.7 0.8

Following Ability

- 0.4

- 0.5

- 0.6

- 0.7

- 0.8 R=0.948 p<1e-4

- Fig. 7: Correlation between Following and Overall performance (left), and between Planning and Overall performance (right). Both Following and Planning abilities are positively correlated with the overall performance of models.

We examine how a model’s Planning and Following abilities relate to its overall performance. Here, overall performance refers to model accuracy in directly answering the final question of each sample, which captures its end-to-end task competence. As depicted in Figure 7, both the Following-overall and Plan-

Following > Planning

Balanced

- Fig. 8: The ratio between Following and Planning ability at different model sizes. The abilities of Planning and Following are more balanced when the ratios are closer to 1.0. Smaller models are better at Following than Planning while larger models have more balanced ability.

ning-overall correlations are markedly strong, indicating that improvements in either dimension tend to yield gains in overall performance. Quantitatively, the Pearson correlation coefficient between Following ability and overall performance is 0.948, while that between Planning ability and overall performance is 0.858. These results indicate that Following ability remains a primary determinant of model end-to-end capability. However, the larger variance in Planning ability across VLMs suggests that current models exhibit more pronounced disparities in their capacity for strategic reasoning.

###### Finding 4

Smaller models are better at Following than Planning while larger models have more balanced performance.

Figure 8 presents the ratio of Following to Planning ability as a function of model size. The results indicate that smaller models exhibit a pronounced imbalance, with substantially greater proficiency in Following compared to Planning. As model size increases, however, this ratio approaches unity and stabilizes, reflecting that larger models achieve a more balanced performance across both reasoning dimensions. This trend suggests that scaling model capacity enhances not only overall performance but also enables models to more evenly develop planning-oriented reasoning capabilities alongside Following skills.

###### Finding 5

VLMs are better at recovering from failed planning steps than from failed following steps.

We assess models’ capability to recover from failure, defined as their ability to reach the correct final answer despite making at least one incorrect intermediate step. This ability is crucial in real-world settings, where models often take

suboptimal reasoning paths. Table 2 reports failure-recovery performance for models with more than 10B parameters. The results indicate that all models exhibit some ability to recover, with recovery from failed planning generally exceeding recovery from failed following. This suggests that models are relatively more robust to suboptimal plans. Distracting questions provide fewer clues but do not necessarily derail the final answer. In contrast, errors in the answer space are more likely to propagate to an incorrect final answer.

Another notable observation is that recovery from the failed following process does not differ significantly between smaller open-source models and larger proprietary models. By contrast, recovery from failed planning is markedly stronger in larger proprietary models. This indicates that larger proprietary models are less susceptible to suboptimal plans and to distraction from uninformative cues along the reasoning chain.

###### 5.3 Additional Results

Complementary results and experimental details are provided in the Appendix. We establish a blindfold baseline in Appendix F to verify the visual-centric nature of our benchmark. To accommodate future model advancements, we investigate increased distractor complexity in Appendix G, demonstrating the benchmark’s potential for extensibility. Finally, Appendix J presents fine-grained case studies to highlight the diagnostic capabilities of our benchmark.

Table 2: VLMs’ ability to recover from exploration failures. The percentages of each model’s responses that contain at least one wrong intermediate step but still arrive at correct final answers.

Model From Failed Planning From Failed Following VLMs: >10B

InternVL3-14B 65.9 55.9 InternVL3.5-14B 65.5 52.6 InternVL2.5-26B 66.9 62.1 InternVL2.5-38B 69.2 63.2 InternVL3-38B 65.1 52.9 InternVL3.5-38B 65.4 45.6

VLMs: Proprietary

- GPT-4o 78.9 56.5

- GPT-5 84.3 53.6

- o1 79.0 59.5

- o3 80.6 57.3 Gemini 2.0 Flash 69.5 59.4 Gemini 2.5 Flash 72.7 51.8

##### 6 Conclusion

We introduced V-REX, a benchmark that evaluates multi-step exploratory visual reasoning through a structured Chain-of-Questions (CoQ) formulation. By disentangling reasoning into Planning and Following, V-REX provides finegrained insights that are hidden in end-to-end evaluation. Experiments on different VLMs reveal consistent scaling trends, strong correlations between intermediate abilities and final performance, and a notable imbalance in smaller models, favoring Following over Planning. Our findings highlight exploratory reasoning, especially effective planning, as a key challenge for future VLMs and establish V-REX as a foundation for advancing deliberate, stepwise visual reasoning.

## Bibliography

- [1] Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report (2025), https://arxiv.org/abs/2502.13923
- [2] Bai, T., Hu, Z., Sun, F., Jiantao, Q., Jiang, Y., He, G., Zeng, B., He, C., Yuan, B., Zhang, W.: Multi-step visual reasoning with visual tokens scaling and verification. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025), https://openreview.net/forum? id=y60FhgO07j
- [3] Cai, J., Yang, K., Fu, L., Ding, J., Li, J., Sun, H., Xing, D., Shen, J., Meng, Z.: Comparebench: A benchmark for visual comparison reasoning in vision-language models (2025), https://arxiv.org/abs/2509.22737
- [4] Campbell, D.I., Rane, S., Giallanza, T., Sabbata, C.N.D., Ghods, K., Joshi, A., Ku, A., Frankland, S.M., Griffiths, T.L., Cohen, J.D., Webb, T.W.: Understanding the limits of vision language models through the lens of the binding problem. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024), https://openreview.net/forum? id=Q5RYn6jagC
- [5] Chen, H., Lou, X., Feng, X., Huang, K., Wang, X.: Unveiling chain of step reasoning for vision-language models with fine-grained rewards (2025), https://arxiv.org/abs/2509.19003
- [6] Chen, J., Li, M., Kil, J., Wang, C., Yu, T., Rossi, R., Zhou, T., Chen, C., Zhang, R.: Visr-bench: An empirical study on visual retrieval-augmented generation for multilingual long document understanding (2025), https: //arxiv.org/abs/2508.07493
- [7] Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al.: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271 (2024)
- [8] Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al.: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 24185–24198

(2024)

- [9] Cheng, Z., Xu, B., Gong, L., Song, Z., Zhou, T., Zhong, S., Ren, S., Chen, M., Meng, X., Zhang, Y., Li, Y., Ren, L., Chen, W., Huang, Z., Zhan, M., Wang, X., Feng, F.: Evaluating mllms with multimodal multi-image reasoning benchmark (2025), https://arxiv.org/abs/2506.04280
- [10] Chow, W., Mao, J., Li, B., Seita, D., Guizilini, V., Wang, Y.: Physbench: Benchmarking and enhancing vision-language models for physical world understanding. arXiv preprint arXiv:2501.16411 (2025)

- [11] Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261

(2025)

- [12] Dalal, S.: timeguessr (2025), https://huggingface.co/datasets/sd16/ timeguessr
- [13] DeepMind, G.: Gemini 2.0 flash (2025), https://deepmind.google/ technologies/gemini/flash/
- [14] DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning (2025), https://arxiv.org/abs/2501.12948
- [15] Dong, Y., Liu, Z., Sun, H.L., Yang, J., Hu, W., Rao, Y., Liu, Z.: Insightv: Exploring long-chain visual reasoning with multimodal large language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 9062–9072 (June 2025)
- [16] Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al.: Openai o1 system card. arXiv preprint arXiv:2412.16720 (2024)
- [17] Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., Li, C.: Llava-onevision: Easy visual task transfer (2024), https://arxiv.org/abs/2408.03326
- [18] Li, M., Wang, C., Liang, Y., Wang, X., Zhou, Y., Wu, X., Zhang, Y., Zhang, R., Zhou, T.: Caughtcheating: Is your mllm a good cheating detective? exploring the boundary of visual perception and reasoning (2025), https: //arxiv.org/abs/2507.00045
- [19] Li, M., Zhang, R., Chen, J., Wang, C., Gu, J., Zhou, Y., Dernoncourt, F., Zhu, W., Zhou, T., Sun, T.: Towards visual text grounding of multimodal large language model (2025), https://arxiv.org/abs/2504.04974
- [20] Li, Y., Liu, Z., Li, Z., Zhang, X., Xu, Z., Chen, X., Shi, H., Jiang, S., Wang, X., Wang, J., Huang, S., Zhao, X., Jiang, B., Hong, L., Wang, L., Tian, Z., Huai, B., Luo, W., Luo, W., Zhang, Z., Hu, B., Zhang, M.: Perception, reason, think, and plan: A survey on large multimodal reasoning models

(2025), https://arxiv.org/abs/2505.04921

- [21] Liang, Y., Li, M., Fan, C., Li, Z., Nguyen, D., Cobbina, K., Bhardwaj, S., Chen, J., Liu, F., Zhou, T.: Colorbench: Can vlms see and understand the colorful world? a comprehensive benchmark for color perception, reasoning, and robustness (2025), https://arxiv.org/abs/2504.10514
- [22] Liu, Y., Ding, J., Deng, G., Li, Y., Zhang, T., Sun, W., Zheng, Y., Ge, J., Liu, Y.: Image-based geolocation using large vision-language models (2024), https://arxiv.org/abs/2408.09474
- [23] Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.W., Galley, M., Gao, J.: Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts (2024), https://arxiv.org/abs/ 2310.02255

- [24] Lyu, Z., Zhang, D., Ye, W., Li, F., Jiang, Z., Yang, Y.: Jigsaw-puzzles: From seeing to understanding to reasoning in vision-language models (2025), https://arxiv.org/abs/2505.20728
- [25] Mathew, M., Bagal, V., Tito, R., Karatzas, D., Valveny, E., Jawahar, C.: Infographicvqa. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 1697–1706 (2022)
- [26] Moreno, M.: geoguessr (2024), https://huggingface.co/datasets/ marcelomoreno26/geoguessr
- [27] OpenAI, :, Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., et al.: Openai o1 system card (2024), https://arxiv. org/abs/2412.16720
- [28] OpenAI: Gpt-5 system card (August 2025), https://cdn.openai.com/gpt5-system-card.pdf
- [29] OpenAI: Openai o3 and o4-mini system card. Tech. rep., OpenAI (April 2025), https://cdn.openai.com/pdf/2221c875-02dc-4789-800be7758f3722c1/o3-and-o4-mini-system-card.pdf
- [30] OpenAI, Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., etc.: Gpt-4o system card (2024), https://arxiv.org/abs/2410.21276
- [31] Pandya, P., Gupta, V., Talwarr, A.S., Kataria, T., Roth, D., Gupta, V.: NTSEBENCH: Cognitive reasoning benchmark for vision language models. In: Chiruzzo, L., Ritter, A., Wang, L. (eds.) Findings of the Association for Computational Linguistics: NAACL 2025. pp. 3680–3708. Association for Computational Linguistics, Albuquerque, New Mexico (Apr 2025). https://doi.org/10.18653/v1/2025.findings-naacl.204, https://aclanthology.org/2025.findings-naacl.204/
- [32] Pothiraj, A., Stengel-Eskin, E., Cho, J., Bansal, M.: Capture: Evaluating spatial reasoning in vision language models via occluded object counting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 8001–8010 (October 2025)
- [33] Ren, Y., Tertikas, K., Maiti, S., Han, J., Zhang, T., Süsstrunk, S., Kokkinos, F.: Vgrp-bench: Visual grid reasoning puzzle benchmark for large visionlanguage models (2025), https://arxiv.org/abs/2503.23064
- [34] Roberts, J., Taesiri, M.R., Sharma, A., Gupta, A., Roberts, S., Croitoru,

I., Bogolin, S.V., Tang, J., Langer, F., Raina, V., Raina, V., Xiong, H., Udandarao, V., Lu, J., Chen, S., Purkis, S., Yan, T., Lin, W., Shin, G., Yang, Q., Nguyen, A.T., Atkinson, D.I., Baranwal, A., Coca, A., Dang, M., Dziadzio, S., Kunz, J.D., Liang, K., Lo, A., Pulfer, B., Walton, S., Yang, C., Han, K., Albanie, S.: Zerobench: An impossible visual benchmark for contemporary large multimodal models (2025), https://arxiv.org/abs/2502.09696

- [35] Satar, B., Ma, Z., Irawan, P.A., Mulyawan, W.A., Jiang, J., Lim, E.P., Ngo, C.W.: Seeing culture: A benchmark for visual reasoning and grounding. In: Christodoulopoulos, C., Chakraborty, T., Rose, C., Peng, V. (eds.) Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 22238–22254. Association for Computational Linguistics,

- Suzhou, China (Nov 2025). https://doi.org/10.18653/v1/2025.emnlpmain.1131, https://aclanthology.org/2025.emnlp-main.1131/
- [36] Shen, H., Liu, P., Li, J., Fang, C., Ma, Y., Liao, J., Shen, Q., Zhang, Z., Zhao, K., Zhang, Q., Xu, R., Zhao, T.: Vlm-r1: A stable and generalizable r1-style large vision-language model (2025), https://arxiv.org/abs/2504.07615
- [37] Song, Y., Ou, T., Kong, Y., Li, Z., Neubig, G., Yue, X.: Visualpuzzles: Decoupling multimodal reasoning evaluation from domain knowledge (2025), https://arxiv.org/abs/2504.10342
- [38] Tan, H., Ji, Y., Hao, X., Chen, X., Wang, P., Wang, Z., Zhang, S.: Reason-rft: Reinforcement fine-tuning for visual reasoning of vision language models

(2025), https://arxiv.org/abs/2503.20752

- [39] Thawakar, O., Dissanayake, D., More, K.P., Thawkar, R., Heakl, A., Ahsan, N., Li, Y., Zumri, I.Z.M., Lahoud, J., Anwer, R.M., Cholakkal, H., Laptev, I., Shah, M., Khan, F.S., Khan, S.: LlamaV-o1: Rethinking step-by-step visual reasoning in LLMs. In: Che, W., Nabende, J., Shutova, E., Pilehvar, M.T. (eds.) Findings of the Association for Computational Linguistics: ACL 2025. pp. 24290–24315. Association for Computational Linguistics, Vienna, Austria (Jul 2025). https://doi.org/10.18653/v1/2025.findings-acl.1247, https://aclanthology.org/2025.findings-acl.1247/
- [40] Tong, P., Brown, E., Wu, P., Woo, S., IYER, A.J.V., Akula, S.C., Yang, S., Yang, J., Middepogu, M., Wang, Z., et al.: Cambrian-1: A fully open, visioncentric exploration of multimodal llms. Advances in Neural Information Processing Systems 37, 87310–87356 (2024)
- [41] Wang, K., Pan, J., Shi, W., Lu, Z., Ren, H., Zhou, A., Zhan, M., Li, H.: Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems 37, 95095–95169 (2024)
- [42] Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al.: Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265 (2025)
- [43] Wang, Y., Wu, S., Zhang, Y., Yan, S., Liu, Z., Luo, J., Fei, H.: Multimodal chain-of-thought reasoning: A comprehensive survey (2025), https://arxiv. org/abs/2503.12605
- [44] Wang, Z., Sun, J., Zhang, W., Hu, Z., Li, X., Wang, F., Zhao, D.: Benchmarking multimodal mathematical reasoning with explicit visual dependency

(2025), https://arxiv.org/abs/2504.18589

- [45] Xu, G., Jin, P., Wu, Z., Li, H., Song, Y., Sun, L., Yuan, L.: Llava-cot: Let vision language models reason step-by-step. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 2087– 2098 (October 2025)
- [46] Xu, W., Wang, J., Wang, W., Chen, Z., Zhou, W., Yang, A., Lu, L., Li, H., Wang, X., Zhu, X., Wang, W., Dai, J., Zhu, J.: Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models (2025), https://arxiv.org/abs/2504.15279
- [47] Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025)

- [48] You, H., Sun, R., Wang, Z., Chen, L., Wang, G., Ayyubi, H., Chang, K.W., Chang, S.F.: IdealGPT: Iteratively decomposing vision and language reasoning via large language models. In: Bouamor, H., Pino, J., Bali, K. (eds.) Findings of the Association for Computational Linguistics: EMNLP 2023. pp. 11289–11303. Association for Computational Linguistics, Singapore (Dec 2023). https://doi.org/10.18653/v1/2023.findings-emnlp.755, https://aclanthology.org/2023.findings-emnlp.755/
- [49] Zhou, H., Li, X., Wang, R., Cheng, M., Zhou, T., Hsieh, C.J.: R1-zero’s "aha moment" in visual reasoning on a 2b non-sft model (2025), https: //arxiv.org/abs/2503.05132
- [50] Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., et al.: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479

(2025)

##### A Image Sources

We collect images for V-REX from a combination of website sources and publicly available visual reasoning benchmarks. Specifically, our dataset incorporates images from PhysBench[10], MathVision[41], CV-Bench[40], and InfographicQA[25]. In addition, we include real-world images sourced from the online platforms GeoGuessr[26] and TimeGuessr[12]. Together, these diverse sources ensure broad coverage across the 15 application scenarios represented in V-REX.

##### B Detailed Taxonomy

Deduction. This category addresses rule-based and logical reasoning, requiring models to explore potential causal or relational patterns to reach consistent conclusions. Exploration involves hypothesizing, verifying, and generalizing implicit rules from limited observations. Flowchart Deduction assesses whether models can reason through multi-step logical structures by tracing dependencies in flow diagrams. Pattern Deduction examines the ability to identify mathematical or visual regularities and extrapolate unseen elements. Property Deduction tests the capacity to infer physical or material properties based on observed patterns, and Relationship Deduction measures understanding of inter-object relationships and causal interactions in structured visual data. Together, these tasks capture the model’s exploratory capacity to discover and apply underlying rules within multi-step reasoning processes.

Guessing. This category examines reasoning under uncertainty, requiring models to explore incomplete or ambiguous evidence to infer missing information. Exploration is reflected in the model’s ability to generate and evaluate multiple hypotheses before settling on the most plausible one. Intention Guessing measures a model’s ability to infer human or object intentions from dynamic visual cues. Location Guessing requires identifying plausible locations based on contextual evidence, while Responsibility Guessing examines causal reasoning in scenarios such as traffic incidents or shared actions. Time Guessing tests temporal inference based on environmental cues, and Topic Guessing evaluates the ability to infer abstract or thematic concepts from multimodal contexts. These tasks together assess exploratory reasoning through uncertainty resolution and context-driven hypothesis testing.

Navigation. This category highlights spatial and sequential reasoning, requiring models to explore action sequences, spatial configurations, and reasoning trajectories. Exploration is reflected in the model’s ability to plan, simulate, and adjust its reasoning path while preserving overall coherence. Map Navigation tests spatial reasoning and route planning, while GUI Navigation evaluates the model’s ability to reason through interface layouts and procedural tasks. Traffic Navigation focuses on understanding dynamic interactions under traffic scenarios and rules. Trend Navigation examines the ability to interpret sequential or time-evolving visual patterns. Together, these tasks assess exploratory

###### QA-Chain Step-Level Chain-Level

Chain 1 Chain 2

[Figure 28]

[Figure 29]

[Figure 30]

Qorigin

|…| |
|---|---|
| | |

|…| |
|---|---|
| | |

| | |
|---|---|
|…| |
| | |

… …

Qi-1 Ai-1

Distracting Chains

Distracting Questions

Step i

Qi Ai

| | |
|---|---|
|…| |
| | |

| | |
|---|---|
|…| |
| | |

… …

…

Qfinal Afinal

GPT-5 Input GPT-5 Output

CoQ Question CoQ Answer

Distractor Question Correct Answer for Distractor Question

Human Annotate

- Fig. 9: Generation Pipeline for Planning. For each QA chain, GPT-5 generates candidate distractors using two complementary strategies: (1) step-level, which introduces diverse distracting questions at each reasoning step, and (2) chain-level, which constructs full distracting reasoning chains. For each sample, we select 1 most challenging distracting chain and 2 diverse distracting questions per step to form the final distractor set.

reasoning as models must simulate and adjust multi-step plans in dynamic spatial or temporal settings.

Retrieval. This category emphasizes global evidence integration, prompting the model to explore multiple visual or textual components. Exploration emerges through the model’s capacity to locate, select, and combine dispersed cues into a coherent reasoning outcome. Counting Retrieval tests whether models can locate and aggregate quantitative information across visual regions, while Word Puzzle Retrieval evaluates the ability to identify, interpret, and reason over embedded textual elements. These tasks together measure exploratory reasoning through multi-step evidence gathering and integration, assessing how effectively models combine local details into coherent global understanding.

Together, these four categories and fifteen sub-tasks form a comprehensive taxonomy for evaluating exploratory reasoning. They capture complementary aspects of how VLMs hypothesize, search, plan, and integrate knowledge across diverse multimodal reasoning scenarios.

##### C Data Curation for Planning

###### C.1 Generation Process

To evaluate VLMs’ exploratory abilities within the question space, we design a Planning task in which models must navigate reasoning chains while confronted with multiple plausible yet distracting questions. This setup probes whether a model can select the most informative sub-questions amid competing alternatives.

The dataset is constructed using a two-stage pipeline composed of Distractor Construction and Question Integration. The pipeline uses powerful LLMs for generation, filtering, and refinement, resulting in question sets that are semantically rich, diverse, and challenging for VLMs.

Distractor Construction. This stage focuses on producing alternative questions that appear contextually relevant but deviate from the intended reasoning path. We design two complementary strategies to capture both local and global variability within reasoning chains. The first strategy, step-level distractor generation, focuses on each reasoning step individually. At every step, the LLM is prompted with the preceding context, including ground-truth questions and previously generated distractors, to propose new step-specific questions that remain plausible yet lead to divergent reasoning directions. This encourages local diversity and tests a model’s ability to identify the correct continuation among multiple confounding options. The second strategy, chain-level distractor generation, operates at a global level. Given an entire ground-truth QA chain, the LLM generates multiple challenger chains that are logically coherent within themselves but deviate from the original reasoning trajectory. These distractors evaluate whether a model can differentiate between globally consistent yet semantically incorrect reasoning paths. To ensure higher-quality distractors, we employ GPT-5 for all question generation.

Question Integration. After generating both step-level and chain-level distractors, we integrate them into a unified Planning dataset through a refinement and selection process. First, we use a strong open-sourced LLM (Qwen3-VL-32BInstruct) to perform inference over a mixture of chain-level distractors and ground-truth questions, and automatically identify the chain that causes the highest confusion or reasoning inconsistency for each sample. This model-guided evaluation highlights the most challenging and coherent distractor chains, which are then retained for integration. Next, GPT-5 is prompted to select the most diverse and difficult step-level distractors that differ from the retained chains while preserving contextual plausibility. These selected distractors are then merged with the original ground-truth questions to form the final multi-choice Planning dataset, where each reasoning step includes both correct and misleading alternatives.

Through this two-stage generation and integration pipeline, the Planning task systematically probes VLMs’ ability to explore and reason within complex question spaces. It reveals whether models can plan consistent reasoning trajectories when confronted with subtle yet semantically misleading alternatives.

###### C.2 Generation Prompt

To support high-quality reasoning across both step-level and chain-level distractor generation, we design two structured system prompts that guide the closed-source GPT-5[28] model in producing challenging and misleading multi-step reasoning traces. The prompts require the model to base every generated question on

explicit visual evidence and to articulate a coherent progression of reasoning across steps, ensuring that each distractor remains contextually grounded while deviating from the intended reasoning path. To accommodate different forms of reasoning complexity, the prompts specify detailed instructions for paraphrasing gold questions, generating plausible yet strategically irrelevant alternatives, and preserving internal coherence within each distractor chain. Additionally, the prompts introduce explicit guidelines for separating task-relevant cues from unhelpful or misleading ones, preventing the model from inadvertently producing reasoning that supports the correct answer.

The prompts are further enhanced with internal checklists, stylistic constraints, and strict prohibitions against helpful reasoning patterns, enabling robust and controlled distractor generation. By enforcing consistent output formats, grounding each question in the visual scene, and providing precise instructions for multi-step dependencies, these prompts facilitate the production of reliable paraphrased questions, step-level distractors, and globally coherent distracting chains. This design supports interpretable and reproducible distractor construction, ensuring that the generated alternatives effectively challenge the reasoning capabilities of modern VLMs.

###### System Prompt for Step-Level Generation (Part 1)

You will be given:

- 1. A final question that must be answered based on an image.
- 2. (Optional) Intermediate questions and confusing questions for earlier steps.

- - If this is step 1, you will only receive the final question.
- - If this is step N > 1, you will receive all intermediate and confusing questions

from steps 1 to N-1 as context.

Your task: For the current reasoning step , generate 5 confusing or

misleading questions that:

- - Are visually and contextually relevant to the scene.
- - Sound appropriate regardless of previous intermediate choices.
- - Are NOT logically useful for solving the final answer

.

- - Should confuse the model ’s step planning.
- - If the final question contains a condition ("If the input is xxx"),

repeat that condition in the confusing questions for the first step.

In other words: The confusing questions must appear contextually

coherent but strategically irrelevant. They should make an incorrect planning choice seem superficially reasonable.

For each unhelpful question , provide:

- - A short factual answer.
- - An unhelpfulness_score (1-3).

###### System Prompt for Step-Level Generation (Part 2)

INTERNAL REASONING CHECKLIST (do not output):

- 1. Coherent with any prior intermediate steps.
- 2. Shares entities or context with the main task.
- 3. Does NOT provide causal , counting , spatial , or comparative reasoning.
- 4. Focuses on irrelevant attributes or secondary objects.
- 5. Appears plausible as a next -step question. FORBIDDEN QUESTION TYPES:

- - Overlap with any intermediate question ’s reasoning step.
- - Provide causal , temporal , or numerical clues needed for the final answer.
- - Help the model disambiguate the correct reasoning step.

Prefer "side -path" confusion questions:

- - About background objects or irrelevant actions.
- - About minor attributes (pose , color , expression , small details).
- - About goals or comparisons unrelated to the final task. OUTPUT RULES:
- - Keep grounded , natural , image -based.
- - No hallucinations.
- - Maintain stylistic consistency.
- - Return ONLY valid JSON:

{

"question_N": [ {" paraphrased_intermediate_question ": "string", "

answer": "string", "unhelpfulness_score ": int}, {"unhelpful_question": "string", "answer": "string

", "unhelpfulness_score": int}, {"unhelpful_question": "string", "answer": "string ", "unhelpfulness_score": int}, {"unhelpful_question": "string", "answer": "string ", "unhelpfulness_score": int},

... ]

}

###### System Prompt for Chain-Level Distractor Generation (Part 1)

You will be given:

- 1. An image.
- 2. A final question that must be answered based on the image.
- 3. Several intermediate questions that form the correct reasoning chain.

Your task: Generate 2 confusing reasoning chains. For each gold

step i, produce:

- - A paraphrase of the gold intermediate question_i.
- - One misleading question (and answer) for Chain 1.
- - One misleading question (and answer) for Chain 2. Each confusing question must:
- - Be visually and contextually relevant to the image.
- - Sound coherent and stylistically close to the intermediate/final question.
- - Remain consistent across steps (a plausible but incorrect chain).
- - Not be logically useful for solving the final question.
- - Confuse the model ’s step planning by offering a tempting side -path.
- - Repeat any conditional clause from the final question (e.g., "If the input is X")

in the confusing questions for the first step. INTERNAL REASONING CHECKLIST (do not output):

- 1. Shares scene context with the gold question.
- 2. Maintains a consistent distractor storyline.
- 3. Avoids reasoning patterns (causal , temporal , spatial , numerical , comparative)

that lead to the correct answer.

- 4. Focuses on irrelevant or secondary attributes , actions , or objects.

###### System Prompt for Chain-Level Distractor Generation (Part 2)

FORBIDDEN HELPFUL QUESTION TYPES:

- - Repeating or clarifying any gold reasoning step.
- - Providing causal , numerical , or spatial clues related to the final answer.
- - Helping to disambiguate or verify the correct reasoning path.

PREFERRED QUESTION STYLE:

- - "Side -path" questions about background details or secondary objects.
- - Attributes and interactions unrelated to the final goal.
- - Natural , grounded , and plausible but strategically irrelevant.

### OUTPUT RULES:

- - Keep questions grounded and factual (no hallucinations).
- - Avoid trivial perceptual details unless used as distractors.
- - Avoid generating questions semantically identical to the original intermediate ones.
- - PARAPHRASE the intermediate question to ensure stylistic consistency , return the original answer as

the corresponding answer.

- - Return ONLY valid JSON following the schema below. ### JSON OUTPUT SCHEMA:

{ "paraphrased_gt_questions ": {"question_1": "string", "answer_1": string", ...},

- "consistent_confusing_questions1 ": {"question_1": " string", "answer_1": string", ...},
- "consistent_confusing_questions2 ": {"question_1": " string", "answer_1": string", ...}

}

##### D Experiment Details

###### D.1 Implementation Details

To comprehensively evaluate the exploratory capabilities of VLMs, we assess a total of 32 models spanning a broad range of state-of-the-art families, including both proprietary and open-source systems. The evaluated models include GPT4o[30], GPT-5[28], O1[16], O3[29], Gemini-2 Flash[13], Gemini-2.5 Flash[11], LLaVA-OV[17], InternVL2.5[7], InternVL3-Instruct[50], InternVL3.5-Instruct[42], Qwen2.5-VL[1], and Qwen3-VL-Instruct (IT) / Think[47]. OpenAI and Gemini models are accessed through their official APIs, while all open-source models are locally deployed for evaluation.

This model suite spans a wide range of architectural designs and parameter scales, from 1B to 38B parameters, covering both multimodal proprietary systems and fully open-source implementations. Such diversity enables a balanced and comprehensive assessment of exploratory reasoning behavior across different computational budgets and training paradigms. To ensure a fair comparison, all open-source models are evaluated under a standardized experimental environment using a single NVIDIA A100 GPU (80GB), with identical decoding and temperature settings across all experiments.

###### D.2 Evaluation Prompts

We provide 2 prompts to assess model behavior under the Following and Planning tasks.

For the Planning task, the model is given the final question at the beginning of the full interaction. At each step, it receives the accumulated conversation history, the image, and a set of candidate intermediate questions consisting of both gold questions and distractors. The prompt requires the model to independently select the most helpful intermediate question from these alternatives and to output its choice in a precise, constrained format.

For the Following task, the model is provided with the conversation history, the image, and the corresponding question and answer candidates at each step. The prompt instructs the model to identify the correct answer for the given question, ensuring strict adherence to the human-designed reasoning chain.

###### Evaluation Prompt for Planning task

Choose the single most helpful intermediate question from the options below. On the FIRST line , output ONLY the option letter in parentheses (e.g., (A)). That first line must match exactly: ^\\([A-F]\\)$ After a blank line , you MAY add brief reasoning.

###### Evaluation Prompt for Following task

You need to choose the most likely answer from the following options , make sure the option letter is in the parentheses like (X).:

##### E Final Accuracy for VLMs on V-REX

The accuracies on final questions under different settings (w/o CoQ, under the Planning task, and under the Following task) are shown in Table 3. Table 3 reports the final-question accuracies of all 32 evaluated VLMs across these three evaluation modes.

Under the w/o CoQ setting (Acc), the model answers the final question directly without relying on any intermediate reasoning steps. AccPlan reflects the final-question accuracy after the model explores the question space by selecting the most informative intermediate question at each step in the presence of distractors. AccFollow measures the final-question accuracy after the model navigates the answer space by choosing the correct answer to each intermediate question along the ground-truth reasoning chain.

Together, these results provide a comprehensive view of how model performance changes when guided by CoQ-based reasoning, highlighting the role of structured exploration in improving visual reasoning.

##### F Blindfold Experiment on V-REX

To investigate the extent to which V-REX relies on visual information, we conducted a “blindfold” experiment where models were evaluated on the benchmark without access to the input images. We evaluate the overall performance of the models on our tasks and the results are presented in Table 4.

Image-Critical Nature of V-REX The results demonstrate that for the majority of categories, specifically Guessing, Deduction, and Retrieval, model performance drops significantly in the absence of visual input. Most models achieve scores close to random guessing (e.g., around 25% for 4-option questions or lower depending on the task format), indicating that these tasks are heavily dependent on visual understanding. The models cannot effectively solve these problems using only the textual questions and options, confirming that V-REX effectively benchmarks visual reasoning capabilities.

###### Table 3: Final accuracy over 4 categories and 32 VLMs with and without CoQ.

###### Model Deduction Guessing Navigation Retrieval Overall

Acc AccPlan AccFollow Acc AccPlan AccFollow Acc AccPlan AccFollow Acc AccPlan AccFollow Acc AccPlan AccFollow VLMs: <7B LLaVA-OV-1B 26.4 27.5 30.8 50.4 49.6 47.9 58.2 55.1 59.2 48.8 48.8 48.8 45.9 45.2 46.7

- InternVL3-1B 26.4 31.9 27.5 49.6 49.6 50.4 60.2 63.3 69.4 24.4 22.0 36.6 40.1 41.7 46.0

- InternVL3.5-1B 29.7 30.8 37.4 43.0 34.7 48.8 59.2 56.1 77.6 41.5 29.3 41.5 43.3 37.7 51.3 Qwen3-VL-2B-IT 24.2 20.9 40.7 66.1 38.8 63.6 61.2 37.8 77.6 26.8 34.1 34.1 44.6 32.9 54.0 Qwen3-VL-2B-Think 33.0 42.9 40.7 44.6 51.2 51.2 39.8 55.1 62.2 43.9 53.7 41.5 40.3 50.7 48.9

InternVL3-2B 40.7 47.3 44.0 54.5 62.8 61.2 67.3 74.5 78.6 39.0 43.9 48.8 50.4 57.1 58.1

- InternVL3.5-2B 36.3 40.7 40.7 42.1 54.5 53.7 52.0 80.6 70.4 51.2 43.9 31.7 45.4 54.9 49.1 Qwen2.5-VL-3B 47.5 40.7 55.4 62.6 58.7 60.2 71.4 73.5 79.6 56.1 58.5 53.7 59.4 57.8 62.2 Qwen3-VL-4B-IT 47.3 44.0 54.9 65.3 63.6 68.6 67.3 76.5 82.7 58.5 58.5 48.8 59.6 60.7 63.7 Qwen3-VL-4B-Think 51.6 54.9 48.4 57.0 56.2 61.2 63.3 69.4 71.4 46.3 53.7 63.4 54.6 58.5 61.1 InternVL2.5-4B 36.3 41.8 42.9 54.5 57.0 62.0 68.4 81.6 80.6 41.5 58.5 43.9 50.2 59.7 57.3 InternVL3.5-4B 45.1 49.5 48.4 49.6 66.1 59.5 73.5 84.7 85.7 58.5 56.1 61.0 56.7 64.1 63.6

VLMs: 7B−10B

LLaVA-OV-7B 44.0 52.7 48.4 57.9 62.8 66.1 68.4 74.5 68.4 61.0 53.7 70.7 57.8 60.9 63.4 Qwen2.5-VL-7B 60.8 35.2 60.8 64.2 67.8 68.3 70.4 72.4 78.6 48.8 56.1 65.9 61.1 57.9 68.4 Qwen3-VL-8B-IT 40.7 42.9 57.1 63.6 62.8 69.4 73.5 83.7 89.8 63.4 68.3 53.7 60.3 64.4 67.5 Qwen3-VL-8B-Think 59.3 54.9 59.3 57.9 48.8 65.3 66.3 74.5 78.6 61.0 58.5 65.9 61.1 59.2 67.3 InternVL2.5-8B 39.6 53.8 53.8 55.4 67.8 63.6 69.4 82.7 80.6 46.3 48.8 51.2 52.7 63.3 62.3

- InternVL3-8B 41.8 47.3 50.5 57.0 67.8 68.6 66.3 81.6 80.6 43.9 53.7 53.7 52.3 62.6 63.4 InternVL3.5-8B 47.3 62.6 53.8 52.9 62.0 57.9 67.3 87.8 85.7 56.1 61.0 48.8 55.9 68.3 61.5

- InternVL3-9B 46.2 56.0 54.9 61.2 71.9 65.3 73.5 82.7 83.7 48.8 56.1 51.2 57.4 66.7 63.8 VLMs: >10B

InternVL3-14B 52.7 57.1 63.7 56.2 77.7 64.5 80.6 88.8 90.8 53.7 48.8 63.4 60.8 68.1 70.6 InternVL3.5-14B 45.1 59.3 54.9 48.8 65.3 64.5 69.4 85.7 91.8 58.5 48.8 58.5 55.4 64.8 67.4 InternVL2.5-26B 53.8 62.6 56.0 60.3 72.7 71.9 76.5 84.7 87.8 56.1 53.7 68.3 61.7 68.4 71.0 InternVL2.5-38B 61.4 64.8 69.2 61.8 74.4 72.7 82.7 87.8 95.9 65.9 63.4 65.9 67.9 72.6 75.9 InternVL3-38B 61.5 65.9 72.5 66.1 77.7 66.1 78.6 85.7 93.9 63.4 58.5 58.5 67.4 72.0 72.8 InternVL3.5-38B 53.8 59.3 60.4 57.9 75.2 63.6 81.6 88.8 74.5 63.4 63.4 63.4 64.2 71.7 65.5

VLMs: Proprietary

- GPT-4o 68.3 59.3 70.3 59.3 83.5 74.0 84.7 94.9 94.9 56.1 70.7 63.4 67.1 77.1 75.6

- GPT-5 67.0 87.9 68.1 77.7 84.3 80.2 85.7 93.9 91.8 78.0 78.0 85.4 77.1 86.0 81.4 O1 62.6 76.9 68.1 77.7 89.3 77.7 90.8 93.9 93.9 75.6 68.3 82.9 76.7 82.1 80.7 O3 76.2 79.1 74.3 72.4 86.8 80.5 86.7 93.9 93.9 80.5 78.0 80.5 79.0 84.5 82.3 Gemini 2.0 Flash 73.3 67.0 73.3 67.5 78.5 69.9 79.6 87.8 93.9 63.4 70.7 73.2 70.9 76.0 77.6 Gemini 2.5 Flash 60.4 68.1 61.5 72.7 76.0 71.9 86.7 91.8 91.8 70.7 75.6 78.0 72.7 77.9 75.8

Analysis of Navigation Performance An notable exception is the Navigation category, where models consistently achieve higher scores even without visual context.This suggests that while visual information aids navigation, the textual components of these questions likely contain significant spatial cues or commonsense patterns that advanced language models can exploit. In this specific category, visual information may serve more as supplementary evidence rather than the sole source of critical information required for decision-making.

##### G Exploring Task Difficulty on Planning

To investigate how the size of the candidate space affects the difficulty of the Planning task, we expand the answer space by introducing four additional distracting options for each intermediate question. This increases the number of candidate options that the model must consider during exploration process. We evaluate multiple VLMs under this setting and report the results in Table 5.

As shown in the table, model performance consistently decreases when the number of candidate options increases. This trend holds across models of different sizes and architectures, indicating that larger candidate sets substantially increase

Table 4: Blindfold model overall performance on VREX tasks.

Model Guessing Deduction Retrieval Navigation qwen3-vl-2B-instruct 34.71 21.98 19.51 55.10 qwen3-vl-4B-instruct 33.88 30.77 14.63 56.12 qwen3-vl-8B-instruct 24.79 30.77 7.32 57.14

- internvl3.5-1B 37.19 29.67 34.15 39.80
- internvl3.5-2B 19.01 19.78 21.95 44.90 internvl3.5-4B 29.75 28.57 9.76 63.27 internvl3.5-8B 12.40 34.07 14.63 51.02 internvl3.5-14B 30.58 26.37 24.39 68.37

- internvl3-1B 29.75 30.77 14.63 43.88
- internvl3-2B 25.62 17.58 17.07 48.98 internvl3-8B 31.40 30.77 17.07 62.24 internvl3-14B 25.62 38.46 31.71 68.37 internvl3-38B 31.40 35.16 26.83 71.43

- GPT-4o 10.74 42.86 21.95 61.22
- GPT-5 36.36 40.66 29.27 77.55

- o1 35.54 35.16 24.39 74.49
- o3 33.06 39.56 31.71 69.39

the difficulty of the task. The performance drop suggests that models struggle more when they must select the correct reasoning step from a larger set of plausible alternatives.

Moreover, although absolute performance decreases with more options, the relative ranking of models remains largely consistent with the results obtained using fewer options. This indicates that the current configuration with fewer candidate options is already sufficient to approximate the comparative exploration abilities of different models, while the expanded candidate set mainly increases task difficulty without significantly altering the relative performance ordering.

##### H Stepwise Recovery from Failure Analysis

To study the model’s ability to recover from failure, we investigate the relationship between the number of wrong planning or following steps in CoQ versus the final accuracy of the model, averaged over all models. As shown in Figure 10 and Figure 11, they display slightly different patterns. The final accuracy decreases monotonically as the number of wrong following steps increases. This reflects that more misinformation in intermediate steps would generally lead to worse results. However, the final accuracy of failed planning is not monotonic. We hypothesize that when the number of wrong planning steps is large, the model might take an entirely different approach and might still arrive at a correct conclusion. Also, the accuracy drops more sharply in Figure 11 than in Figure 10, which further verifies that models are generally more robust to wrong following steps than wrong planning steps.

Table 5: Planning performance with expanded candidate space. We increase task difficulty by expanding the candidate options for each intermediate question from 4 to 8. Results across Deduction, Guessing, Navigation, and Retrieval show consistent performance drops across models as the candidate space grows, indicating increased exploration difficulty. Despite this, the relative ranking of models remains largely stable.

Deduction Guessing Navigation Retrieval Average

Model 4 opts 8 opts 4 opts 8 opts 4 opts 8 opts 4 opts 8 opts 4 opts 8 opts VLMs: <7B

- InternVL3-1B 37.09 16.82 47.32 28.89 42.48 22.72 42.38 22.78 42.32 22.80

- InternVL3.5-1B 42.36 20.45 43.48 19.53 42.25 20.06 41.11 22.30 42.30 20.58 Qwen3-VL-2B-IT 31.90 20.92 37.45 11.30 34.09 15.44 28.41 9.33 32.96 14.25

InternVL3-2B 52.73 32.30 53.84 29.37 64.09 30.85 36.87 21.03 51.88 28.39

- InternVL3.5-2B 63.13 37.29 60.76 41.63 63.20 35.42 42.74 21.67 57.46 34.00 Qwen3-VL-4B-IT 62.54 46.03 65.45 46.32 70.93 43.02 36.79 26.59 58.93 40.49 InternVL3.5-4B 69.59 52.86 65.44 50.36 66.76 40.75 51.83 24.68 63.40 42.16

VLMs: 7B–10B Qwen3-VL-8B-IT 74.69 51.19 84.75 61.41 79.96 53.68 46.83 31.31 71.56 49.40

- InternVL3-8B 53.50 32.43 44.55 29.11 66.78 32.43 40.08 20.24 51.23 28.55 InternVL3.5-8B 66.23 52.91 69.63 54.26 73.08 48.50 47.26 36.87 64.05 48.13

- InternVL3-9B 69.08 50.69 81.57 52.66 84.09 55.61 57.50 41.55 73.06 50.13 VLMs: >10B

InternVL3-14B 76.36 55.05 86.35 54.87 79.80 57.41 56.71 33.53 74.81 50.21 InternVL3.5-14B 78.55 61.44 73.01 53.84 77.91 53.20 53.29 40.36 70.69 52.21 InternVL3-38B 83.21 59.80 91.31 68.55 86.56 57.56 65.20 46.43 81.57 58.08 InternVL3.5-38B 81.70 54.52 85.30 62.01 84.29 57.30 57.98 45.67 77.32 54.88

VLMs: Proprietary

- GPT-4o 77.92 47.32 69.61 36.29 79.65 46.63 63.69 34.01 72.72 41.06

- GPT-5 93.89 56.91 91.82 69.86 93.90 61.79 58.13 34.60 84.44 55.79 O1 94.43 62.91 93.24 70.76 92.38 64.11 61.27 44.33 85.33 60.53 O3 94.76 68.60 94.45 68.86 93.28 59.63 62.10 40.71 86.15 59.45

##### I Shared Failures Analysis among Models

Although several strong models (e.g. GPT-5 and O1/O3) achieve high average accuracies on Planning and Following tasks, such averages can mask process-level failures along intermediate steps. Moreover, final-answer accuracy can be high due to recovery/shortcuts; therefore, we define sample difficulty using intermediatestep performance. To reveal whether the benchmark contains systematic difficulty (rather than isolated model-specific errors), we conduct a step-wise shared failure analysis across six model families.

We stratify samples into easy, moderate and hard based on intermediate-step performance across six model families. Let a sample contain T intermediate steps and let N denote the number of models evaluated (namely, Qwen3-VL-8Binstruct, Gemini 2.5 Flash, GPT-5, InternVL3-9B, LLaVA-OV-7B and O1). For each step t ∈ {1,...,T}, we compute the number of models that answer step t correctly. We use two sample-level statistics: (i) AvgCorrect, mean intermediate-

Final Accuracy by Number of Wrong Planning Steps

Accuracy(AveragedoverAllModels)

0.8

0.724

0.7

0.633

0.6

0.553

0.525

0.5

0.455

0.429

0.4

0.3

0.2

0.1

0.0

0 1 2 3 4 5

Number of Wrong Planning Steps

- Fig. 10: Stepwise result of recovery from failed planning

0 1 2 3 4 5

Number of Wrong Following Steps

0.0

0.2

0.4

0.6

0.8

1.0

Accuracy(AveragedoverAllModels)

0.880

0.626

0.468

0.324

0.204

0.000

Final Accuracy by Number of Wrong Following Steps

- Fig. 11: Stepwise result of recovery from failed following

step accuracy across both steps and models (each model’s step outcome is binary here), and (ii) UnanimousFail, an indicator of whether there exists at least one intermediate step that all models fail. Using these statistics, we assign difficulty

- as follows: Easy if AvgCorrect = 1; Hard if UnanimousFail = 1 or AvgCorrect < 0.5; and Moderate otherwise. Across the Following dataset (375 samples), we obtain 69 hard, 224 moderate, and 82 easy samples; across the Planning dataset (351 samples), we obtain 43 hard, 287 moderate, and 21 easy samples. Notably, Planning contains fewer hard samples under our strict definition (which targets near-universal step failures), yet overall Planning accuracy can still be lower because question selection is inherently more error-prone than answering a provided step in Following.

We then measure shared failures step-wise by computing, at each step position, the fraction of samples for which a supermajority of model families fail that step (fail rate ≥ 75%, i.e., at least 5 out of 6 models fail). Figures 12 and 13 compare this quantity for hard versus moderate samples across categories for Following and Planning tasks, respectively. While higher shared failure on hard samples is expected given our stratification, the key observation is that failures concentrate

- at specific step positions: moderate samples rarely exhibit such near-universal step failures (curves remain close to zero), whereas hard samples show clear peaks where most model families fail simultaneously, indicating systematic bottlenecks

| |Sample Difficulty<br><br>moderate<br><br>hard| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| |Sample Difficulty<br><br>moderate<br><br>hard| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

withfailurerate75%

withfailurerate75%

Fractionofsamples

Fractionofsamples

0.4

0.4

0.2

0.2

0.0

0.0

1 2 3 4 5 6

1 2 3 4

Step position in reasoning chain

Step position in reasoning chain

- Fig. 12: Step-wise shared failure on Following (hard vs. moderate samples). For each step, we report the fraction of samples whose failure rate is at least 75%, where the failure rate is the proportion of evaluated models that answer the step incorrectly (here N = 6, so ≥ 75% means at least 5/6 models fail). Curves are shown separately for hard and moderate samples across task categories.

shared across diverse model architectures. The peak locations vary by category (e.g., earlier bottlenecks in Retrieval/Navigation/Deduction and later bottlenecks in Guessing), suggesting that both early grounding and later integration/decision stages remain challenging and can propagate errors through the chain. As a result, the benchmark retains meaningful headroom even when average accuracy appears high, and it remains informative over time: progress can be measured not only by final answers but also by reductions in shared step-level failures and improved process reliability.

##### J Cases Study

###### J.1 Success cases

We showcase representative success cases enabled by CoQ across different task settings in Figure 14 and 15. Decomposing the final question into sub-questions helps the model correct perceptual mistakes or stay aligned with the intended reasoning path, ultimately enabling it to reach the correct final answer.

###### J.2 Failure cases of CoQ

While CoQ helps organize multi-step reasoning, its advantages are limited for Retrieval category. These problems depend largely on straightforward factual

| |Sample Difficulty<br><br>moderate<br><br>hard| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| |Sample Difficulty<br><br>moderate<br><br>hard| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

withfailurerate75%

withfailurerate75%

Fractionofsamples

Fractionofsamples

0.4

0.4

0.2

0.2

0.0

0.0

1 2 3 4 5 6

1 2 3 4

Step position in reasoning chain

Step position in reasoning chain

- Fig. 13: Step-wise shared failure on Planning Task (hard vs. moderate samples). For each step, we report the fraction of samples whose failure rate is at least 75%, where the failure rate is the proportion of evaluated models that answer the step incorrectly (here N = 6, so ≥ 75% means at least 5/6 models fail). Curves are shown separately for hard and moderate samples across task categories.

lookup or direct visual identification, rather than layered reasoning or extended inference. Since Retrieval questions typically involve minimal reasoning depth, adding a CoQ chain often introduces extra steps that offer little value and may not influence the final prediction.

- Figure 16 illustrates a failure case where the model correctly answers the question without CoQ. However, when the model is required to engage in stepwise exploration, it identifies the appropriate sub-questions yet still fails to reach the correct final answer. This indicates that the bottleneck lies not in hierarchical reasoning or exploration, but in the model’s fundamental perceptual ability to extract fine-grained information from a visually dense image.
- Figure 17 further shows a mismatch between the human-designed reasoning path and the model’s natural reasoning trajectory. Under the Planning task, the model struggles to align its question selection with the intended human chain. Even under the Following task where it is required to adhere to the predefined CoQ, the model still fails to generate the correct final answer. This suggests that when the solution does not depend on multi-step abstraction, the CoQ structure can conflict with the model’s native retrieval process.

###### J.3 Failure cases of Planning

Although hierarchical question planning is intended to guide models toward more structured and interpretable reasoning, we observe that some VLMs experience

[Figure 31]

Final Question: How many individual cubes are used in total to build this structure?

Final Answer: 30

|Without CoQ Inference|
|---|
|To determine the total number of cubes, count the number of pillars and the number of cubes in each pillar.<br><br>The image shows 5 pillars, and each pillar consists of 8 cubes. Therefore, the total number of cubes is: 5 pillars × 8 cubes per pillar = 40 cubes Thus, the most likely answer is (E) 40.|

|Model’s Planning w/ CoQ|
|---|

[Figure 32]

How many towers are there in total?

6

[Figure 33]

How many cubes are in each tower?

5

How many individual cubes are used in total to build this structure?

[Figure 34]

30

- Fig. 14: Success case for Planning task. Without CoQ, the model fails to reach the correct final answer due to perceptual and counting errors. Under the Planning setting, however, once the model selects the correct intermediate question, it receives the corresponding answer, which compensates for its perceptual limitations and enables it to arrive at the correct final prediction.

performance degradation when intermediate CoQ steps are introduced. The expanded reasoning space can create distracting side-paths, causing models to focus on visually salient but semantically irrelevant sub-questions or to overlook key reasoning cues that are necessary for solving the task. In such situations, the additional steps fail to strengthen the intended chain of thought and instead introduce extra cognitive noise that disrupts both step selection and final answer prediction. As shown in Figure 18 and Figure 19, models may shift their attention toward superficial objects or answer-option patterns, diverging from the human-designed chain and ultimately producing incorrect conclusions even with correct initial steps.

###### J.4 Failure cases of Following

Although the human-designed CoQ path is intended to guide the model to answer the final question with manually-crafted decomposition steps, we observe that sometimes the model would fail to answer the final question correctly after following the CoQ path. Figure 20 and Figure 21 show two failure cases in the Following task that represent two typical failure modes. As shown in Figure 20, the model correctly answers all the intermediate questions about the book cover, but it fails to leverage the information extracted from previous reasoning steps to answer the final question correctly. Different from the previous case, the model in Figure 21 incorrectly answers the intermediate question about the least numbers of protusions and indentations of the missing puzzle piece, leading to an incorrect final answer.

[Figure 35]

Final Question: What is the complexity of the algorithm described in the flowchart?

Final Answer: O(N logN)

|Without CoQ Inference|
|---|

|(A) O(N).|
|---|

|Model’s Following w/ CoQ|
|---|

What is the algorithm described in the flowchart?

[Figure 36]

Merge sort

How many individual cubes are used in total to build this structure?

[Figure 37]

30

###### Fig. 15: Success case for Following task. By decomposing the final question into informative sub-questions, the model correctly identifies the algorithm in the flowchart and uses this intermediate insight to arrive at the correct final answer.

[Figure 38]

Final Question: Which word is in this puzzle?

Final Answer: (C) Heart

|Without CoQ Inference|
|---|
|(B) Whistle|

|Human-designed CoQ|
|---|

|Model’s Planning w/ CoQ|
|---|

|Model’s Following w/ CoQ|
|---|

[Figure 39]

Which word's first-two-letter combination is in this puzzle?

Which word's first-two-letter combination is in this puzzle?

Which word's first-two-letter combination is in this puzzle?

[Figure 40]

Hearts and Whistle

Hearts and Whistle

All of them

[Figure 41]

Which word's first-three-letter combination is in this puzzle?

Which word's first-three-letter combination is in this puzzle?

Which word's first-three-letter combination is in this puzzle?

[Figure 42]

All of them

All of them

All of them

[Figure 43]

Which word's first-four-letter combination is in this puzzle?

Which word's first-four-letter combination is in this puzzle?

Which word's first-four-letter combination is in this puzzle?

[Figure 44]

Hearts

Hearts

Whistle

Which word is in this puzzle?

Which word is in this puzzle?

Which word is in this puzzle?

[Figure 45]

[Figure 46]

Heart

Whistle

Whistle

###### Fig. 16: Failure case in the Retrieval category where CoQ provides limited benefit. The model can directly identify the correct word from the puzzle without CoQ. However, when required to follow or plan using the CoQ chain, the model becomes misled by intermediate steps and incorrectly answers “Whistle.” The example shows how structured CoQ exploration can interfere with tasks that primarily require precise visual matching rather than sequential reasoning.

|Without CoQ Inference|
|---|
|To determine the number of completely visible cars in the image, we need to identify which cars are fully within the frame and not partially obscured by other objects.<br><br>1. The black SUV is fully visible in the lower left part of the image.<br>2. The silver sedan is fully visible in the lower right part of the image.<br>3. The silver SUV is also fully visible in the lower right part of the image.<br>4. The black car is partially visible on the left side of the image, but only the front part is visible. Thus, there are three cars that are completely visible: the black SUV, the silver sedan, and the silver SUV. Therefore, the correct answer is (B) 3 cars are fully visible.<br>|

[Figure 47]

Final Question: How many cars are completely visible in this image scene?

Final Answer: (B) 3 cars are fully visible

|Human-designed CoQ|
|---|

|Model’s Planning w/ CoQ|
|---|

|Model’s Following w/ CoQ|
|---|

[Figure 48]

Where is the scene depicted in the image taking place?

Is there an advertising poster or billboard near the sidewalk?

Where is the scene depicted in the image taking place?

[Figure 49]

Next to a hilly park along a road

Yes. a poster board is visible on right side

Next to a hilly park along a road

[Figure 50]

Is there a large floral car shaped design visible on the hillside?

Is there a giant clock face made of plants on the slope?

Is there a large floral car shaped design visible on the hillside?

[Figure 51]

No

Yes

Yes

[Figure 52]

Are there cars parked on the road next to the green hill?

Is there a metal fence running along the edge of the lawn above the sidewalk?

Are there cars parked on the road next to the green hill?

[Figure 53]

Yes. A few cars are parked on the road

Yes

Yes. A few cars are parked on the road

How many cars are completely visible in this image scene?

How many cars are completely visible in this image scene?

How many cars are completely visible in this image scene?

[Figure 54]

[Figure 55]

3 cars are fully visible

2 cars are fully visible

2 cars are fully visible

- Fig. 17: Failure case in the Retrieval category where CoQ provides limited benefit. Without CoQ, the model naturally checks each car one by one and correctly identifies three fully visible cars. Under Planning, the imposed CoQ chain forces the model into a reasoning path that diverges from its native strategy, leading to incorrect intermediate questions and an incorrect final count. Even under Following, the model cannot recover, showing that the human-designed chain is incompatible with the model’s retrieval-oriented reasoning.

|Without CoQ Inference|
|---|
|To determine the fourteenth animal the magician will pull out of his hat, we need to identify the repeating pattern and then find the position of the fourteenth animal within that pattern. The pattern of the animals is as follows:<br><br>1. Mouse. 2. Snail. 3. Yellow bird. 4. Frog. 5. Red bird This pattern repeats every 5 animals. To find the fourteenth animal, we can use modular arithmetic: \[ 14 \mod 5 = 4 \] This means that the fourteenth animal corresponds to the fourth animal in the repeating pattern. According to the pattern: 1st animal: Mouse. 2nd animal: Snail. 3rd animal: Yellow bird. 4th animal: Frog. 5th animal: Red bird. Therefore, the fourteenth animal is a Frog.|

[Figure 56]

Final Question: A magician takes animals out of his hat always in the same order, as shown below. The pattern of the figure is repeated every five animals. What will be the fourteenth animal he will pull out of his hat?

Final Answer: A (Frog)

|Human-designed CoQ|
|---|

|Model’s Planning w/ CoQ|
|---|

[Figure 57]

The pattern of the figure is repeated every few animals. What is the length of the repeating cycle?

The pattern of the figure is repeated every few animals. What is the length of the repeating cycle?

5

5

[Figure 58]

What is the position of the fourteenth animal within its repeating cycle?

What is the position of the fourteenth animal within its repeating cycle?

4th

4th

[Figure 59]

Which animal occupies that position in the repeating cycle?

How many of the answer options depict birds?

A (Frog)

2

What will be the fourteenth animal he will pull out of his hat?

What will be the fourteenth animal he will pull out of his hat?

[Figure 60]

A (Frog)

D (Bird)

###### Fig. 18: Failure case in the Planning task. Although the model correctly identifies the repeating cycle and the position of the fourteenth animal, its third step selects an irrelevant question about birds. This detour misleads the final prediction, causing the model to answer with a bird rather than the correct frog.

Final Question: Which word best describes this image?

[Figure 61]

Final Answer: Spam

|Without CoQ Inference|
|---|
|(B) Spam|

|Human-designed CoQ|
|---|

|Model’s Planning w/ CoQ|
|---|

[Figure 62]

What object is sketched on the left side of the drawing?

What color outlines the big rectangular shape on the left?

A can of processed luncheon meat

Blue

[Figure 63]

What object is sketched on the left side of the drawing?

Inside that left rectangle, what color is the wide horizontal bar?

A pig envisioning a letter

Purple

Which word best describes this image?

Which word best describes this image?

[Figure 64]

Spam

Bacon‑Mail

###### Fig. 19: Failure case in the Planning task. The model selects a chain of CoQ steps centered on color descriptions, which are uninformative for determining the intended concept of the drawing. This distractive chain introduces additional noise and prevents the model from connecting the pig and canned meat sketches to the correct answer “Spam.”

[Figure 65]

Final Question: Based on the cover, what is the subject of the book?

Final Answer: (C) Quantum Mechanics

|Without CoQ Inference|
|---|
|The cover is likely related to Quantum Mechanics, as David J. Griffiths is known for writing a textbook on this subject. Therefore, the most likely subject of the book is: C. Quantum Mechanics (C)|

###### Human-designed CoQ

###### Model’s Following w/ CoQ

What might be the drawing on the cover?

What might be the drawing on the cover?

[Figure 66]

A cat

A cat

Who is listed as the author of the book?

Who is listed as the author of the book?

[Figure 67]

David J. Griffiths

David J. Griffiths

What subject is this author best known for?

What subject is this author best known for?

[Figure 68]

Quantum Mechanics

Quantum Mechanics

Based on the cover, what is the subject of the book?

Based on the cover, what is the subject of the book?

[Figure 69]

Quantum Mechanics

Electromagnetism

###### Fig. 20: Failure case in the Following task. Although the model correctly answers all the intermediate questions about the book cover, it fails to leverage the information extracted from previous reasoning steps to answer the final question correctly.

Final Question: Which one of the pieces numbered 1-6 completes the middle puzzle by filling the missing slot?

[Figure 70]

Final Answer: (B) 2

|Without CoQ Inference|
|---|
|(B) 2|

|Human-designed CoQ|
|---|

|Model’s Following w/ CoQ|
|---|

At least how many protrusions are there in the middle piece according to the incomplete puzzle?

At least how many protrusions are there in the middle piece according to the incomplete puzzle?

[Figure 71]

- 1

At least how many indentations are there in the middle piece according to the incomplete puzzle?

- 2

2

At least how many indentations are there in the middle piece according to the incomplete puzzle?

[Figure 72]

0

Which one of the pieces numbered 1-6 completes the middle puzzle by filling the missing slot?

Which one of the pieces numbered 1-6 completes the middle puzzle by filling the missing slot?

[Figure 73]

3

2

###### Fig. 21: Failure case in the Following task. The model incorrectly answers the intermediate question about the least numbers of protusions and indentations of the missing puzzle piece, leading to an incorrect final answer.

