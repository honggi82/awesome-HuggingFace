## RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark

Yang Shi1,2⋄ Yuhao Dong3⋄♠ Yue Ding4⋄ Yuran Wang1⋄ Xuanyu Zhu1⋄ Sheng Zhou5⋄ Wenting Liu1⋄ Haochen Tian3⋄ Rundong Wang6 Huanqian Wang7 Zuyan Liu7 Bohan Zeng1 Ruizhe Chen8 Qixun Wang1 Zhuoran Zhang1 Xinlong Chen4 Chengzhuo Tong1 Bozhou Li1 Qiang Liu4 Haotian Wang7‡ Wenjing Yang Yuanxing Zhang2‡ Pengfei Wan2 Yi-Fan Zhang4‡ Ziwei Liu3‡ 1PKU 2Kling Team 3NTU 4CASIA 5NUS 6USTC 7THU 8ZJU ⋄ Core Contributor ♠ Project Leader ‡ Corresponding Author https://github.com/FrankYang-17/RealUnify

# arXiv:2509.24897v2[cs.AI]20Mar2026

##### Understanding Generation

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Stage 3: Synergetic Evaluation

Stage 1: Direct GenEval

|[Figure 7]|
|---|

###### MMBench

|[Figure 8]|
|---|

Question: What are the colors of the four chairs in the whole picture from left to right?

[Figure 9]

|[Figure 10]|
|---|

GEU

Which one is the correct caption of this image?

a photo of a red cake and a purple chair.

[Figure 11]

Mental Reconstruction

[Figure 12]

|Hint: Restore the<br><br>image that has been shuffled by patches<br><br>|[Figure 13]|
|---|
|
|---|

Stage 1.5: Combination

Answer: White, Yellow, Blue, Red

|[Figure 14]|
|---|

|[Figure 15]|
|---|

Does this artwork belong to the type of mythological?

Add a mouse on the floor in front of the cat.

[Figure 16]

|Hint: Refine it to ensure the generated image fully aligns with the given conditions.|
|---|

Prompt: There are three cats in a row. The black cat is not on the far right, the white cat is to the left of the black cat, and the gray cat is to the right of the white cat.

[Figure 17]

[Figure 18]

Stage 2: Reasoning

WISE

LogicVista

[Figure 19]

|[Figure 20]|
|---|

|[Figure 21]|
|---|

UEG

|[Figure 22]|
|---|

Which one is the correct caption of this image?

Logical Reasoning

Einstein's favorite musical instrument

Figure 1. Illustration of RealUnify. Unlike benchmarks focused on either understanding or generation (Stage 1), those that merely integrate both capabilities (Stage 1.5), or even those that preliminarily explore the mutual enhancement between understanding and generation (Stage 2), RealUnify stands as the first benchmark to comprehensively evaluate and fully harness the synergy between these capabilities, making it a pioneering effort in assessing ability synergy for unified models.

###### Abstract

a benchmark specifically designed to evaluate bidirectional capability synergy. RealUnify comprises 1,000 meticulously human-annotated instances spanning 10 categories and 32 subtasks. It is structured around two core axes: 1) Understanding Enhances Generation, which requires reasoning (e.g., commonsense, logic) to guide image generation, and 2) Generation Enhances Understanding, which necessitates mental simulation or reconstruction (e.g., of transformed or disordered visual inputs) to solve reasoning tasks. A key contribution is our dual-evaluation protocol, which combines direct end-to-end assessment with a diagnostic stepwise evaluation that decomposes tasks into distinct understanding and generation phases. This protocol allows us to precisely dis-

The integration of visual understanding and generation into unified models represents a significant stride toward generalpurpose AI. However, a fundamental question remains unanswered by existing benchmarks: does this architectural unification actually enable synergetic interaction between the constituent capabilities? Existing evaluation paradigms, which primarily assess understanding and generation in isolation, are insufficient for determining whether a unified model can leverage its understanding to enhance its generation or use generative simulation to facilitate deeper comprehension. To address this gap, we introduce RealUnify,

cern whether performance bottlenecks stem from deficiencies in core abilities or from a failure to integrate them. Through large-scale evaluations of 12 leading unified models and 6 specialized baselines, we find that current unified models still struggle to achieve effective synergy, indicating that architectural unification alone is insufficient. These results highlight the need for new training strategies and inductive biases to fully unlock the potential of unified modeling.

###### 1. Introduction

The field of multimodal artificial intelligence has undergone a paradigm shift with the rise of unified models that integrate both visual understanding (e.g., visual question answering) and generation (e.g., text-to-image synthesis) within a single neural architecture [8, 21, 25, 43]. While such unification offers architectural elegance, its most compelling promise lies in the potential for synergetic effects between capabilities: leveraging knowledge and reasoning from understanding to guide more accurate generation, and employing internal generative simulation (e.g., “thinking with images”) to facilitate more complex understanding. A fundamental open question remains: do the two core capabilities, i.e., understanding and generation, mutually enhance each other? This question further motivates a reconsideration of architectural design: should we pursue a unified model that integrates both, or co-located models without functional synergy?

A primary obstacle in answering this question is the lack of a suitable benchmark. As illustrated in Figure 1, current evaluation frameworks predominantly assess understanding and generation in isolation (Stage 1), and some benchmarks [19, 27, 42] combine tasks from both domains to evaluate capabilities simultaneously (Stage 1.5). Recent efforts like T2I-CoReBench [18] and WISE [24] have begun exploring whether understanding enhances generation quality, but do not explicitly test whether success on a task depends on the interaction of both capabilities. Thus, there remains a pronounced lack of rigorous benchmarks with systematic design to probe the very essence of unification: bidirectional capability synergy. Moreover, while existing text-to-image benchmarks [11, 24] focus on aspects such as image aesthetics or textual relevance, which are areas that current unified models aim to improve through largescale, high-quality training data. However, these capabilities are not sufficient for addressing complex, real-world problems and fail to reflect the true level of intelligence a model possesses.

To address this gap, we introduce RealUnify, the first comprehensive benchmark aimed at answering the fundamental question: Can unified models effectively leverage their synergy between understanding and generation abilities to solve complex tasks? The core innovation of RealUnify lies in its meticulously designed task suite, where each in-

stance requires an intricate interplay between understanding and generation. RealUnify structures its evaluation around two core tracks: (1) Understanding Enhances Generation (UEG), which tests whether knowledge and reasoning improve generation accuracy, and (2) Generation Enhances Understanding (GEU), which examines whether generative reconstruction and visualization can support more effective visual reasoning.

As shown in Table 1, RealUnify’s 1,000 instances span 10 categories and 32 manually crafted and validated subtasks, which in particular requires synergy between understanding and generation, such as tasks that require mathematical computation prior (understanding) to generate images and visual tracking of multi-step transformations (generation) to answer questions (understanding). Unlike traditional benchmarks [11, 24] that focus on aspects such as the aesthetic quality of generated images or textual relevance, RealUnify shifts its focus to the model’s ability to solve real-world tasks. Notably, all tasks in RealUnify are carefully designed with a focus on the synergy between understanding and generation, ensuring that each instance requires an intricate interplay of both capabilities. Moreover, a cornerstone of RealUnify is its dual-evaluation protocol including the direct evaluation and stepwise evaluation, enabling precise diagnosis of whether unified models achieve genuine capability synergy or merely functional coexistence. Specifically, direct evaluation tests whether models can achieve end-to-end synergy in a realistic setting (closer to the intrinsic capability of models during the realistic deployment), whereas stepwise evaluation decomposes tasks into understanding and generation, revealing whether performance limits arise from weak individual capabilities or from the lack of genuine synergy.

Through extensive evaluations of 12 leading unified models and 6 state-of-the-art specialized baselines using our dual-evaluation protocol, we uncover a striking conclusion: despite their unified architecture, current models still struggle to synergize understanding and generation capabilities effectively. This finding is robustly supported by three key empirical patterns. First, under direct evaluation, models perform poorly on both UEG (average 37.5% for best opensource) and GEU tasks, indicating their inability to spontaneously integrate capabilities in end-to-end scenarios. Second, and more diagnostically, the stepwise evaluation reveals a revealing dissociation: when UEG tasks are decomposed into “understanding-then-generation” stages, performance improves significantly (e.g., BAGEL [8] improves from 32.7% to 47.7%), demonstrating that models possess the required knowledge but cannot seamlessly integrate it. Conversely, decomposing GEU tasks into “generation-thenunderstanding” stages causes performance to degrade, suggesting that models default to relying on understanding shortcuts rather than effectively leveraging generation. Third, when we construct an “oracle” model by combining the

- Table 1. Comparisons on RealUnify and other benchmarks. RealUnify is designed to provide a comprehensive evaluation of unified models across multiple dimensions. It is entirely human-annotated and integrates both direct and stepwise evaluation protocols. RealUnify centers on evaluating whether the synergy between generation and understanding can be effectively harnessed to solve complex tasks.

###### Benchmark Category #QA #Tasks #Subtasks Annotation Eval Und. Gen. Ability Synergy

MME-Unify [42] Unified 4,104 14 - Mixed Direct ✓ ✓ ✗ UniEval [19] Unified 1,234 13 81 (M)LLM Direct ✓ ✓ ✗ T2I-CoReBench [18] T2I 1,080 2 12 (M)LLM Direct ✗ ✓ ✗ Science-T2I [17] T2I 898 3 16 Human Direct ✗ ✓ ✗ T2I-ReasonBench [29] T2I 800 4 26 (M)LLM Direct ✗ ✓ ✗ WISE [24] T2I 1,000 3 25 Mixed Direct ✗ ✓ ✗ MMBench [22] I2T 3,217 6 20 Mixed Direct ✓ ✗ ✗ LogicVista [40] I2T 448 5 9 Human Direct ✓ ✗ ✗

RealUnify Unified 1,000 10 32 Human Direct/Step ✓ ✓ ✓

best specialist models (Gemini 2.5 Pro [7] for understanding and GPT-Image-1 [25] for generation) in a stepwise manner, it achieves 72.7% on UEG tasks, establishing a highperformance upper bound that current unified models fall far short of. Collectively, these results indicate that architectural unification alone is insufficient. To fully realize the potential of capability synergy, unified models require more advanced training schemes and stronger inductive biases.

###### 2. Related Work

Unified Models. Recently, unified models [5, 21, 31, 34, 38] have emerged as a central research direction in multimodal intelligence. These frameworks integrate both visual understanding and generation within a single architecture and have demonstrated competitive performance. Early studies [5, 6, 30, 34] primarily emphasize functional integration, ensuring that a single model could simultaneously perform understanding and generation. With the evolution of model capabilities, research interest has shifted toward examining whether unification itself can yield additional benefits or even give rise to emergent abilities. For example, Liquid [39] provides empirical evidence that training data from either understanding or generation tasks can enhance performance on the other, indicating reciprocal benefits between the two. Building on this finding, UniFluid [9] shows that well-designed training strategies can further reinforce such cross-task gains. Beyond performance improvements, BAGEL [8] uncovers the emergence of complex compositional behaviors, such as multimodal generation with long-context reasoning.

Benchmarks for Unified Models. Research on unified models has recently emerged, driving the need for benchmarks tailored to their evaluation. Among existing efforts, MME-Unify [42] is the first benchmark to jointly assess multimodal comprehension, generation, and mixed-modality tasks. Building on this direction, UniEval [19] enables evaluation without auxiliary models or human annotations. Despite these advances, these benchmarks still fall short of assessing whether integrating understanding and generation ac-

tually produces measurable performance gains. Complementary to these works, several text-to-image (T2I) benchmarks, such as MMMG [23], T2I-CoReBench [18], and WISE [24], can also be adapted for evaluating unified models, but their emphasis on T2I tasks provides limited evidence of reciprocal benefits between understanding and generation.

###### 3. RealUnify

The tasks in RealUnify fall into two categories: assessing whether model understanding enhances generation (UEG) (Section 3.1), and whether generative ability supports understanding (GEU) (Section 3.2). As shown in Figure 2, these categories jointly evaluate the extent to which crosscapability transfer improves complex task performance and overall model competence. Details on dataset construction and evaluation are provided in Section 3.3.

###### 3.1. Understanding Enhances Generation (UEG)

For the UEG tasks, we focus on a thorough evaluation of the image generation capabilities of current unified models. To emphasize the role of understanding in the image generation process, we design 6 categories in which the model must first interpret the prompt before producing the output.

World Knowledge. This task category targets image generation grounded in objective world knowledge. The goal is to examine whether unified models can accurately produce visual content that aligns with established facts. To ensure comprehensive coverage, the tasks span 7 major domains of knowledge: Animals & Plants, Food, Architecture, Culture, Sports, Technology, and Lifestyle, enabling a broad assessment of models’ ability to leverage world knowledge.

Commonsense Reasoning. Commonsense reasoning requires the model to generate images that reflect everyday phenomena observed in the real world. In this category, the model is given prompts describing widely recognized real-world situations, and it should produce corresponding images. The evaluation focuses on whether models demonstrate commonsense intelligence, such as understanding basic physical laws, human activities, and common objects.

[Figure 23]

[Figure 24]

###### UNDERSTANDING ENHANCES GENERATION (UEG) GENERATION ENHANCES UNDERSTANDING (GEU)

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

###### World Knowledge

###### Commonsense Reasoning

###### Mental Reconstruction

###### Mental Tracking

[Figure 29]

Question: Turn all black segments into orange, then turn all yellow into orange, then turn all green into red. Which digits are formed by the orange segments? A. “1,7” B. “4,7”. C. ”4,6”. D. “7,6”

Prompt: A slice of butter melting unevenly on hot toast.

Question: Including the photographer, how many cars are there in the photo? A. 3 B. 1 C. 4 D. 2

[Figure 30]

[Figure 31]

Prompt: The largest feline animal in terms of body size.

[Figure 32]

[Figure 33]

[Figure 34]

User

User

User

User

Question:

- 1. Is there a slice of butter present?
- 2. Is the butter placed on a toast?
- 3. Is the butter shown melting unevenly rather than in a uniform manner?

[Figure 35]

Question:

[Figure 36]

[Figure 37]

[Figure 38]

- 1. Does this image show a Tiger?
- 2. Can you identify a Tiger in this image?

[Figure 39]

[Figure 40]

Hint: Restore the image that has been shuffled by patches…

Evaluation

[Figure 41]

Evaluation

[Figure 42]

[Figure 43]

[Figure 44]

Hint: Apply the transformations to the contents of the image.

User

[Figure 45]

[Figure 46]

###### Mathematical Reasoning

###### Logical Reasoning

User

[Figure 47]

[Figure 48]

Answer: A

Prompt: Three birds, one blue and one gray, are lined up on a telephone pole. The blue bird is not in the middle, and the adjacent birds are different colors.

Answer: D

Prompt: A table with some books, which could be arranged into four stacks with 2 books each. Draw all the books on the table.

[Figure 49]

[Figure 50]

Evaluation

Evaluation

User

User

[Figure 51]

[Figure 52]

###### Attentional Focusing

###### Cognitive Navigation

Question:

Question:

[Figure 53]

[Figure 54]

- 1. Is the blue bird not in the middle?
- 2. Are the adjacent different colors?

1. Are there a total of exactly 8 books on the table?

[Figure 55]

Question: What is the text written on the blue golf ball holder?

Evaluation

Evaluation

Question: On the shortest path from Penguin to Polar, which of the following animals can we see?? A. Rabbit B. Monkey C. Aviary D. Lion

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

###### Scientific Reasoning Code To Image

User

User

- A. MUTUAL INSURANC
- B. NEW YORK MUTUAL
- C. NEW MEXICO MUTUAL
- D. NEW MEXICO INSURANCE

Prompt: Code: num = int(input()) if num > 0: print("A pair of shoes") elif num < 0: print("A pink pig rolling in the mud.")

[Figure 64]

Prompt: ,A litmus solution is exposed to a carbon dioxide (CO₂) environment.

[Figure 65]

User

User

else: print("A fluffy sheep with a bell around its neck.")

Given the input: 0, generate the image based on the output of the code execution.

Question:

- 1. Is the solution red?
- 2. Is the solution blue?
- 3. Is the solution purple?

[Figure 66]

[Figure 67]

[Figure 68]

Hint: Highlight the regions of the image that are relevant to the question.

[Figure 69]

Hint: Mark the path(s) in the image that are relevant to the question.

Question:

✘

[Figure 70]

Evaluation

User

User

1. Does the image show a fluffy sheep with a bell around its neck?

✘

Evaluation

[Figure 71]

[Figure 72]

[Figure 73]

Answer: C

Answer: B

[Figure 74]

|Refine it to ensure the generated image fully aligns with the given conditions.|
|---|

Hint for UEG:

Evaluation

Evaluation

User

Figure 2. Overview of RealUnify. The benchmark includes 2 task categories: Understanding Enhances Generation (UEG) and Generation Enhances Understanding (GEU), encompassing 10 task types. Hints are provided to guide task decomposition in the stepwise evaluation.

Mathematical Reasoning. Mathematical reasoning and computation represent fundamental yet essential abilities for intelligent systems. In this category, models need to perform the necessary calculations implied by the image-generation instructions in order to produce correct results. The tasks cover 4 subtasks, including Numerical Calculation (SingleStep & Multi-Step), Probability Estimation, Proportional Reasoning, and Constraint-based Equation Solving.

Logical Reasoning. Logical reasoning is a cornerstone of intelligence, enabling models to trace dependencies, combine multiple conditions into consistent outcomes, and adapt their outputs to hypothetical changes. This category assesses whether models can reason over explicit or implicit conditions to ensure generated outputs satisfy logical constraints.

Scientific Reasoning. Tasks in this category require reasoning grounded in specialized scientific principles across 4 domains: Physics, Chemistry, Biology, and Geography. The goal is to assess whether models can correctly apply established scientific principles to reason about given scenarios and generate outputs consistent with real-world phenomena.

Code-to-Image. This category evaluates the model’s ability to bridge symbolic code and visual generation. Specifically, the model must parse the provided code, reason over its logic in conjunction with the given input, and infer the correct textual instruction implied by the execution outcome. It is then required to generate an image that faithfully reflects this inferred instruction.

###### 3.2. Generation Enhances Understanding (GEU)

For the GEU tasks, we focus on questions that require the model to leverage its generative capabilities to simplify problem-solving and thereby improve overall accuracy. To this end, we design 4 customized tasks that evaluate the model’s understanding capability.

Mental Reconstruction. This task type evaluates models’ ability to reason over and reconstruct disrupted visual inputs. Images are divided into patches of varying granularity and then shuffled. The model is tasked with answering specific questions based on this disordered image. Achieving accurate responses often relies on understanding spatial, matching, and relational cues. Success in these tasks necessitates accurate reconstruction of the original image.

Mental Tracking. This task aims to test the model’s ability to trace and update visual states through a sequence of transformations. The input consists of digits constructed from colored line segments, and the model is instructed to perform diverse modifications—for example, “first

changing all blue segments to green, then turning all green segments into yellow”, and so on. The model is then queried about the digit represented by a specific color. Accomplishing such tasks requires the model to internally track and memorize how different regions evolve under successive changes.

Attentional Focusing. This category examines whether unified models can effectively concentrate on critical regions within complex visual inputs, a paradigm often associated with the notion of “thinking with images” [28, 45]. Common techniques for emphasizing salient content include cropping, bounding-box annotation, or super-resolution. For unified models, however, a key challenge is whether they can leverage their native visual generation ability to highlight target regions directly, thereby facilitating more accurate visual understanding. This task category encompasses 3 sub-tasks, involving Quantity Recognition, OCR Recognition, and Attribute Recognition, which collectively assess a model’s capacity to extract and reason over essential information.

Cognitive Navigation. Navigation is an essential task in real-world scenarios, where models must proceed step by

step to ultimately reach a defined goal. In this category, we distinguish between two types of navigation tasks. The first type is maze navigation, in which we synthesize multiple mazes and require models to answer questions that involve solving them. The second type is map navigation, which we consider more representative of real-world conditions. Beyond simply solving the map, this task evaluates the model’s ability to identify the shortest route or follow specific paths under given constraints, thereby providing a more comprehensive assessment of its navigation capabilities.

in the text-to-image setting, we further employ a question list to poll the outputs, ensuring that the visual content aligns with the intended target.

|Prompt: A platinum wire is dipped into barium chloride crystals and placed in a flame.|
|---|

[Figure 75]

[Figure 76]

Model

###### 3.3. Benchmark Construction

Is there a platinum wire in the image?

[Figure 77]

[Figure 78]

|yes|
|---|

Data Collection and Annotation. We collect data from multiple sources. For the UEG tasks, all prompts are manually curated by 10 human experts. After collection, we perform a cross-check in which three additional reviewers independently validate the prompts, and only those agreed upon by all reviewers are retained. For the GEU tasks, we develop an automated script to generate samples for Mental Reconstruction and Mental Tracking tasks, which are then annotated by human experts. A cross-checking procedure is similarly applied to ensure data correctness. For the Attentional Focusing task, we sample data from BLINK [10] and HR-Bench [33]. For the Cognitive Navigation task, mazes are generated automatically, with human experts providing the corresponding answers. Additionally, map images are sourced via the Google Search API, and the associated questions and answers are created by human experts. The detailed annotation and verification process can be found in the supplementary materials.

User

Gemini

Is the platinum wire placed in the flame?

[Figure 79]

[Figure 80]

|yes|
|---|

User

Gemini

|Is the flame green?|
|---|

[Figure 81]

[Figure 82]

|yes|
|---|

User

Gemini

Figure 3. Illustration of polling evaluation. To assess the accuracy of the generated images, we design verification questions and employ Gemini 2.5 Pro as the judge in a polling-based evaluation.

Stepwise Evaluation. In stepwise evaluation, tasks in RealUnify are explicitly decomposed into 2 sequential stages. For the Understanding Enhances Generation setting, the unified model must first solve the problem in pure text form and then use the obtained response as the instruction for subsequent image generation, which is consistent with the “first understanding, then generation” paradigm widely adopted in text-to-image models [8, 13, 20]. In contrast, for the Generation Enhances Understanding setting, the model is required to first produce an intermediate image based on the given input and then answer the corresponding question using this image, which is consistent with the “first generation, then understanding” strategy commonly explored in works on “thinking with images” [15, 26, 32, 44, 45]. This protocol not only enables a finer-grained analysis of potential bottlenecks in unified models but also provides clearer evidence of whether unification leads to genuine performance gains.

Evaluation Criteria. To further investigate whether unified models can truly benefit from unification, we design two complementary evaluation protocols: direct evaluation and stepwise evaluation. Direct evaluation focuses on measuring the overall performance of unified models, examining whether unification leads to notable gains in an endto-end manner. In contrast, stepwise evaluation explicitly decomposes each task into separate stages of understanding and generation, allowing a fine-grained analysis of model strengths and weaknesses and providing clearer evidence of whether unification contributes to improved capability in solving complex tasks.

Direct Evaluation. In direct evaluation, the model is required to perform the tasks described in Sections 3.1 and

Benchmark Statistics. After a rigorous process of question construction, selection, and subsequent annotation and verification by domain experts, we compile a dataset consisting of 1,000 questions, with 600 UEG tasks and 400 GEU tasks. As illustrated in Figure 4, these questions cover a wide range of categories, resulting in 32 distinct subtasks distributed in multiple domains. This dataset provides a systematic framework for evaluating the capabilities of unified models in a synergetic manner, offering insight into their effectiveness in complex real-world tasks.

- 3.2 without intermediate decomposition. For the Understanding Enhances Generation setting, tasks take the form of text-to-image generation, where the model must directly produce a target image given a problem statement. For the Generation Enhances Understanding setting, tasks adopt an image-to-text format, in which the model is provided with an image, a question, and multiple candidate options, and is expected to select the most appropriate option. As is shown in Figure 3, to verify the correctness of the generated images

[Figure 83]

- Figure 4. Statistics of RealUnify. The tasks span 10 categories, divided into two groups: UEG and GEU, including 32 subtasks.

###### 4. Experiments

###### 4.1. Evaluation on RealUnify

Evaluation Setup. For the evaluation on RealUnify, we consider a total of 12 unified models: 11 state-of-the-art opensource models and 1 cutting-edge proprietary model. Among the open-source models, we select representative candidates, including BAGEL-7B [8], OmniGen2 [38], Ovis-U13B [31], UniWorld-V1 [21], UniPic2-Metaquery-9B [36], OneCAT-3B [16], MIO [35], ILLUME+ [14], Show-o2 [41], Janus-Pro [5], and BLIP3-o [4]. In addition, we evaluate the closed-source Gemini-2.5-Flash-Image [12], also referred to as “Nano Banana”, which serves as a strong baseline.

To better quantify the performance gap between unified and state-of-the-art specialized models in visual understanding and generation, we evaluate 3 high-performing models from each domain. For visual understanding, we include Gemini-2.5-Pro [7], GPT-4.1 [1], and Qwen2.5-VL-7B [2]. For image generation, we assess FLUX.1 Kontext [3], QwenImage [37], and GPT-Image-1 [25]. Taken together, these evaluation results provide a comprehensive overview of the capabilities of current unified models as well as cutting-edge understanding and generation models on RealUnify.

Main Results. We evaluate all unified models on RealUnify using both direct and stepwise evaluation, as described in Section 3.3. The results for each task category, as well as the overall performance, are summarized in Table 2. All performance values are reported as accuracy percentages.

As shown in Table 2, under direct evaluation on RealUnify, existing unified models perform poorly on both UEG and GEU tasks, underscoring the gap between current unified approaches and true task unification. In particular, UEG tasks reveal a marked performance disparity between open-

source and proprietary models. While the best open-source model achieves 37.5, the proprietary Nano Banana [12] reaches 63.0. This highlights the difficulty open-source unified models face in leveraging their understanding capabilities to support generation inherently. In contrast, the GEU tasks reveal a different pattern. Although all models still perform poorly, open-source models demonstrate notably stronger understanding capabilities than proprietary models. This further confirms our earlier conclusion. Despite their promising comprehension abilities, current unified models struggle to effectively incorporate such understanding into the generation process. Bridging this gap between understanding and generation is crucial for enhancing the performance of these models, particularly when dealing with complex generation tasks.

To further explore the capabilities of current unified models, we conduct experiments using the stepwise evaluation framework. Since certain models do not support image editing, their results on the GEU tasks are not reported. By decoupling both UEG and GEU tasks, our goal is to uncover the true potential of these models and to further analyze their stepwise performance on tasks that demand both understanding and generation abilities. The results reveal a quite surprising pattern. For UEG tasks, all models benefit from stepwise decomposition, with BAGEL showing the most substantial improvement (+15). These results suggest that current unified models can internally retain the knowledge required for complex generation tasks. However, they struggle to inherently leverage this knowledge in practice in the UEG tasks, indicating that they remain far from achieving genuine task unification. In the case of GEU tasks, the situation differs considerably. After stepwise decomposition, all models exhibit reduced performance. This outcome indicates that although current unified models possess adequate generative capabilities, they still lack the ability to effectively apply these capabilities to real-world problem-solving. The observed degradation further suggests that, in direct evaluation, these models tend to rely primarily on their understanding abilities while overlooking the fact that GEU tasks demand a synergetic integration of both generation and understanding.

Taken together, these results suggest that although current unified models possess sufficient understanding and generation capabilities individually, they fall short on tasks that require a synergetic integration of both. This shortcoming highlights a persistent performance gap between existing approaches and the goal of achieving true unification.

Judge Reliability Validation. To verify the reliability of Gemini 2.5 Pro [7] as a judge model for evaluating generated images, we additionally employ Qwen2.5-VL [2], one of the most advanced open-source models, as an alternative judge. To establish a reliable baseline, we invite 4 human experts to conduct manual evaluations and report the averaged scores. As shown in Table 4, Gemini 2.5 Pro exhibits stronger agree-

- Table 2. Evaluation results on RealUnify. WR: World Knowledge; CR: Commonsense Reasoning; MR-I: Mathematical Reasoning; LR: Logical Reasoning; SR: Scientific Reasoning; C2T: Code-to-Image; MR-II: Mental Reconstruction; MT: Mental Tracking; AF: Attentional Focusing; CN: Cognitive Navigation. For each task, we present both direct and stepwise evaluation results, reported in the format direct/step. The best performance on each task is in blue .

Model

Understanding Enhances Generation Generation Enhances Understanding

Total WK CR MR-I LR SR C2I Avg MR-II MT AF CN Avg

Proprietary Models

Nano Banana 89 / - 86 / - 34 / - 65 / - 48 / - 56 / - 63.0 / - 34 / - 27 / - 36 / - 30 / - 31.8 / - 50.5 / -

Open-Source Unified Models

MIO 24 / 35 26 / 33 18 / 13 9 / 10 10 / 11 0 / 8 14.5 / 18.3 26 / 23 19 / 18 35 / 19 23 / 21 25.8 / 20.3 19.0 / 19.1 Janus-Pro 25 / 26 77 / 71 16 / 7 13 / 17 16 / 20 3 / 10 25.0 / 25.2 21 / - 23 / - 28 / - 29 / - 25.3 / - 25.1 / ILLUME+ 44 / 52 62 / 62 22 / 22 23 / 25 26 / 26 1 / 7 29.7 / 32.3 27 / 27 19 / 20 35 / 38 30 / 25 27.8 / 27.5 28.9 / 30.4 Show-o2 30 / 42 56 / 50 25 / 25 21 / 21 18 / 20 18 / 19 28.0 / 29.5 36 / - 28 / - 36 / - 21 / - 30.3 / - 28.9 / -

OmniGen2 36 / 55 61 / 60 21 / 26 29 / 28 16 / 20 19 / 6 30.3 / 32.5 30 / 42 21 / 24 51 / 38 28 / 19 32.5 / 30.8 31.2 / 31.8 UniPic2 61 / 62 73 / 72 31 / 30 28 / 38 25 / 26 7 / 15 37.5 / 40.5 26 / 28 20 / 24 27 / 27 23 / 16 24.0 / 23.8 32.1 / 33.8 UniWorld-V1 51 / 56 64 / 59 26 / 26 33 / 37 21 / 24 15 / 9 35.0 / 35.2 29 / 33 19 / 25 57 / 36 24 / 20 32.3 / 28.5 33.9 / 32.5 Ovis-U1 37 / 59 72 / 71 28 / 30 23 / 34 15 / 17 12 / 25 31.2 / 39.3 32 / 38 28 / 25 60 / 31 36 / 24 39.0 / 29.5 34.3 / 35.4 BLIP3-o 57 / 62 71 / 74 21 / 24 19 / 25 28 / 22 2 / 9 33.0 / 36.0 36 / - 25 / - 57 / - 32 / - 37.5 / - 34.8 / OneCAT 61 / 64 70 / 65 32 / 20 29 / 27 24 / 31 9 / 27 37.5 / 39.0 26 / 29 25 / 26 43 / 26 31 / 36 31.3 / 29.3 35.0 / 35.1 BAGEL 46 / 74 70 / 80 23 / 26 29 / 37 21 / 29 7 / 40 32.7 / 47.7 37 / 38 31 / 25 50 / 52 39 / 28 39.3 / 35.8 35.3 / 42.9

- Table 3. Performance comparison of unified models and specialized models. We report results by selecting the top-3 performing unified models based on their overall performance in UEG and GEU and comparing them against specialized models.

Table 4. Comparisons of different judges. We assess the quality of the models’ generated images with different judges, and the results are reported in the direct/step format.

Judge Nano Banana BAGEL OneCAT OmniGen2

(a) Understanding Enhances Generation (UEG)

Gemini 2.5 Pro 63.0 / - 32.7 / 47.7 37.5 / 39.0 30.3 / 32.5 Qwen2.5-VL 70.7 / - 35.3 / 59.0 35.5 / 44.7 33.5 / 38.8 Human Expert 59.3 / - 31.5 / 44.2 36.0 / 38.8 27.7 / 30.3

Model WK CR MR-I LR SR C2I Total Specialized Models GPT-Image-1 90 87 31 69 48 48 62.2 Qwen-Image 66 83 28 44 25 67 52.2 FLUX.1 Kontext 53 73 25 27 25 37 40.0 Unified Models

###### Table 5. Comparisons with Gen-Und SOTA.

Model WK CR MR-I LR SR C2T Total Nano Banana 89 86 34 65 48 56 63 Und→Gen (SOTA) 93 86 43 70 53 91 72.7 Model MR-II MT AF CN Total

Nano Banana 89 86 34 65 48 56 63.0 UniPic2 61 73 31 28 25 7 37.5 OneCAT 61 70 32 29 24 9 37.5

(b) Generation Enhances Understanding (GEU)

BAGEL 37 31 50 39 39.3 Gen→Und (SOTA) 29 27 21 50 31.8

Model MR-II MT AF CN Total Specialized Models Gemini 2.5 Pro 30 73 73 43 54.8 GPT-4.1 38 23 56 37 38.5 Qwen2.5-VL 35 23 44 36 34.5 Unified Models BAGEL 37 31 50 39 39.3 Ovis-U1 32 28 60 36 39.0 BLIP3-o 36 25 57 32 37.5

###### 4.2. Analysis Experiments

Comparisons with Specialists. We then compare the top 3 unified models, ranked by their overall performance on the UEG and GEU tasks, against the leading specialized models. As shown in Table 3, unified models demonstrate competitive results on UEG tasks, even surpassing state-of-the-art image generation models in some cases. This suggests that incorporating understanding capabilities can indeed facilitate complex generation tasks. Nevertheless, the comparison also reveals that, for challenging image generation tasks, current open-source models still exhibit a substantial performance gap relative to proprietary counterparts, indicating that further progress in model architecture and large-scale training is required to close this gap. On GEU tasks, by contrast, open-source unified models already demonstrate strong un-

ment with human expert evaluations, while Qwen2.5-VL displays a greater divergence. Therefore, the evaluation based on Gemini 2.5 Pro as the judge is relatively reliable and can largely align with human expert assessments.

derstanding abilities, even outperforming certain proprietary specialist models. These findings underscore the value of synergetic unified frameworks, which not only exploit strong understanding capabilities to improve generation, narrowing the gap with proprietary generation models, but also integrate generative components to enrich understanding, making the overall process more intuitive and efficient.

how unified models address tasks and to highlight the performance gap relative to ideal settings.

- Figure 5 shows two examples revealing how Bagel [8]

benefits from stepwise execution, which compels it to integrate both understanding and generative capabilities. For UEG tasks, by explicitly leveraging its understanding ability, the model correctly infers that the object required is a lightsaber and is then able to generate an appropriate image. For GEU tasks, although the model does not perfectly reconstruct the disordered image patches, the intermediate reconstruction still guides it toward the correct answer. These examples reveal how unified models can benefit from the synergy between understanding and generation, enabling them to solve problems that neither capability alone could address.

We further examine the oracle settings and provide two illustrative examples to demonstrate their performance when supplied with ground-truth intermediate results. As shown in

- Figure 6, Bagel continues to underperform even under stepwise evaluation. However, the performance improves when the ground-truth intermediate step is given. This observation suggests that existing unified models still lack essential internal capabilities. This limitation in individual capabilities may also hinder the effective integration of understanding and generation in real-world problem-solving, underscoring the need to strengthen the core capacities of unified models as a prerequisite for true unification.

How well can unified models be? To better examine the potential upper bound of current unified models, we conduct experiments by combining two of the most powerful models for generation and understanding: Gemini 2.5 Pro [7] and GPT-Image-1 [25]. This combination is evaluated in a stepwise manner to approximate the maximum performance that unified models could achieve in task unification.

As shown in Table 5, integrating these two strong models yields an impressive score of 72.7 on UEG tasks. This result not only demonstrates that our UEG tasks require substantial reasoning capabilities for successful generation, but also highlights that, although current unified models benefit from stepwise decomposition on UEG tasks, they remain far from achieving comparable performance in a truly synergetic fashion. For GEU tasks, the results further support our earlier conclusion that, although current state-of-the-art generative models can produce photorealistic images, they remain inadequate for addressing real-world problems. When GPT-Image-1 [25] is integrated with Gemini 2.5 Pro [7], we observe substantial performance degradation, underscoring that adapting generative capabilities for practical problemsolving remains a significant challenge for unified models. Strengthening the generalization capacity of generative models is thus a key step toward developing more reliable and effective unified frameworks.

###### 5. Conclusion

In this paper, we introduce RealUnify, the first comprehensive benchmark explicitly designed to investigate capability synergy between understanding and generation in unified models. Unified models should not merely represent the coexistence of understanding and generation; rather, they should enable a synergistic interaction between these two capabilities, fostering mutual enhancement to achieve a higher level of intelligence. RealUnify systematically evaluates this synergy through two complementary settings—Understanding Enhances Generation and Generation Enhances Understanding—spanning 10 diverse task categories. Extensive experiments and analysis reveal that current unified models are still far from achieving genuine synergy, and there remains a significant gap when compared to the state-of-the-art specialized models for understanding or generation. Although they can accomplish tasks under stepwise decomposition, indicating substantial potential, their inability to succeed in end-to-end scenarios highlights the absence of true synergy. Realizing such synergy in unified models, and thus empowering them to tackle complex real-world tasks, remains a pressing research direction.

###### Protocol 1: Direct

|Question:<br><br>How many cartoon toy cars are in the picture? A. 3 B. 2 C. 1 D. 0<br><br>[Figure 84]|
|---|

|Prompt: The most iconic handheld weapon in Star Wars.|
|---|

Protocol 1: Direct

[Figure 85]

[Figure 86]

[Figure 87]

|B|
|---|

###### BAGEL

BAGEL

✘ blaster rifle

[Figure 88]

[Figure 89]

[Figure 90]

Refine it to ensure the generated image fully aligns with the given conditions.

[Figure 91]

Restore the image that has been shuffled by patches…

User

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

A lightsaber, the iconic handheld weapon from Star Wars.

|C|
|---|

BAGEL BAGEL

lightsaber

User

BAGEL

BEGAL

[Figure 98]

- Figure 5. Effective examples of stepwise execution in task solving. Through the unified model’s inherent understanding and generation abilities, the model is able to implement complex tasks.

[Figure 99]

|Prompt: The fastestrunning animal on land.|
|---|

[Figure 100]

[Figure 101]

BAGEL BAGEL

The fastest land animal.

✘ wrong

[Figure 102]

|A|
|---|

[Figure 103]

[Figure 104]

[Figure 105]

BAGEL GT

A Cheetah.

correct

[Figure 106]

[Figure 107]

|Question: How many people are there in the picture? A. 8 B. 7 C. 6 D. 5<br><br>[Figure 108]|
|---|

[Figure 109]

BAGEL

[Figure 110]

GT

[Figure 111]

[Figure 112]

[Figure 113]

BAGEL

[Figure 114]

BAGEL

|B|
|---|

- Figure 6. Challenging examples of stepwise execution in task solving. Despite using a stepwise approach, the unified model struggles to complete complex tasks, only succeeding with intermediate results based on the given ground truth.

Case Study. We present qualitative results to illustrate

###### Acknowledgement

This research is supported by cash and in-kind funding from NTU S-Lab and industry partner(s). This study is also supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOE-T2EP20221-0012, MOET2EP20223-0002).

###### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 6
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 6
- [3] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv–2506,

2025. 6

- [4] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 6, 1
- [5] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025. 3, 6, 1
- [6] Zhihong Chen, Xuehai Bai, Yang Shi, Chaoyou Fu, Huanyu Zhang, Haotian Wang, Xiaoyan Sun, Zhang Zhang, Liang Wang, Yuanxing Zhang, et al. Opengpt-4o-image: A comprehensive dataset for advanced image generation and editing. arXiv preprint arXiv:2509.24900, 2025. 3
- [7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 3, 6, 8, 1
- [8] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 2, 3, 5, 6, 8, 1
- [9] Lijie Fan, Luming Tang, Siyang Qin, Tianhong Li, Xuan Yang, Siyuan Qiao, Andreas Steiner, Chen Sun, Yuanzhen Li, Tao Zhu, et al. Unified autoregressive visual generation and understanding with continuous tokens. arXiv preprint arXiv:2503.13436, 2025. 3
- [10] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and

- Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer, 2024. 5, 1
- [11] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-toimage alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 2
- [12] Google. Introducing gemini 2.5 flash image, our stateof-the-art image model. https://developers. googleblog.com/en/introducing-gemini-25-flash-image/, 2025. 6, 1
- [13] Ziyu Guo, Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Rui Huang, Haoquan Zhang, Manyuan Zhang, Jiaming Liu, Shanghang Zhang, Peng Gao, et al. Can we generate images with cot? let’s verify and reinforce image generation step by step. arXiv preprint arXiv:2501.13926, 2025. 5
- [14] Runhui Huang, Chunwei Wang, Junwei Yang, Guansong Lu, Yunlong Yuan, Jianhua Han, Lu Hou, Wei Zhang, Lanqing Hong, Hengshuang Zhao, et al. Illume+: Illuminating unified mllm with dual visual tokenization and diffusion refinement. arXiv preprint arXiv:2504.01934, 2025. 6, 1
- [15] Xinyu Huang, Yuhao Dong, Weiwei Tian, Bo Li, Rui Feng, and Ziwei Liu. High-resolution visual reasoning via multiturn grounding-based reinforcement learning. arXiv preprint arXiv:2507.05920, 2025. 5
- [16] Han Li, Xinyu Peng, Yaoming Wang, Zelin Peng, Xin Chen, Rongxiang Weng, Jingang Wang, Xunliang Cai, Wenrui Dai, and Hongkai Xiong. Onecat: Decoder-only auto-regressive model for unified understanding and generation. arXiv preprint arXiv:2509.03498, 2025. 6, 1
- [17] Jialuo Li, Wenhao Chai, Xingyu Fu, Haiyang Xu, and Saining Xie. Science-t2i: Addressing scientific illusions in image synthesis, 2025. 3
- [18] Ouxiang Li, Yuan Wang, Xinting Hu, Huijuan Huang, Rui Chen, Jiarong Ou, Xin Tao, Pengfei Wan, and Fuli Feng. Easier painting than thinking: Can text-to-image models set the stage, but not direct the play? arXiv preprint arXiv:2509.03516, 2025. 2, 3
- [19] Yi Li, Haonan Wang, Qixiang Zhang, Boyu Xiao, Chenchang Hu, Hualiang Wang, and Xiaomeng Li. Unieval: Unified holistic evaluation for unified multimodal understanding and generation. arXiv preprint arXiv:2505.10483, 2025. 2, 3
- [20] Jiaqi Liao, Zhengyuan Yang, Linjie Li, Dianqi Li, Kevin Lin, Yu Cheng, and Lijuan Wang. Imagegen-cot: Enhancing textto-image in-context learning with chain-of-thought reasoning. arXiv preprint arXiv:2503.19312, 2025. 5
- [21] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025. 2, 3, 6, 1
- [22] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 3
- [23] Yuxuan Luo, Yuhui Yuan, Junwen Chen, Haonan Cai, Ziyi Yue, Yuwei Yang, Fatima Zohra Daha, Ji Li, and Zhouhui

- Lian. Mmmg: A massive, multidisciplinary, multi-tier generation benchmark for text-to-image reasoning. arXiv preprint arXiv:2506.10963, 2025. 3
- [24] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025. 2, 3
- [25] OpenAI. Introducing 4o image generation. https: //openai.com/index/introducing-4o-imagegeneration/, 2025. 2, 3, 6, 8
- [26] Yang Shi, Jiaheng Liu, Yushuo Guan, Zhenhua Wu, Yuanxing Zhang, Zihao Wang, Weihong Lin, Jingyun Hua, Zekun Wang, Xinlong Chen, et al. Mavors: Multi-granularity video representation for multimodal large language model. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 10994–11003, 2025. 5
- [27] Yang Shi, Huanqian Wang, Wulin Xie, Huanyao Zhang, Lijie Zhao, Yi-Fan Zhang, Xinfeng Li, Chaoyou Fu, Zhuoer Wen, Wenting Liu, et al. Mme-videoocr: Evaluating ocr-based capabilities of multimodal llms in video scenarios. arXiv preprint arXiv:2505.21333, 2025. 2
- [28] Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025. 4
- [29] Kaiyue Sun, Rongyao Fang, Chengqi Duan, Xian Liu, and Xihui Liu. T2i-reasonbench: Benchmarking reasoning-informed text-to-image generation. arXiv preprint arXiv:2508.17472,

2025. 3

- [30] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 3
- [31] Guo-Hua Wang, Shanshan Zhao, Xinjie Zhang, Liangfu Cao, Pengxin Zhan, Lunhao Duan, Shiyin Lu, Minghao Fu, Xiaohao Chen, Jianshan Zhao, et al. Ovis-u1 technical report. arXiv preprint arXiv:2506.23044, 2025. 3, 6, 1
- [32] Qixun Wang, Yang Shi, Yifei Wang, Yuanxing Zhang, Pengfei Wan, Kun Gai, Xianghua Ying, and Yisen Wang. Monet: Reasoning in latent visual space beyond images and language. arXiv preprint arXiv:2511.21395, 2025. 5
- [33] Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7907–7915, 2025. 5, 1
- [34] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 3
- [35] Zekun Wang, King Zhu, Chunpu Xu, Wangchunshu Zhou, Jiaheng Liu, Yibo Zhang, Jiashuo Wang, Ning Shi, Siyu Li, Yizhi Li, et al. Mio: A foundation model on multimodal tokens. arXiv preprint arXiv:2409.17692, 2024. 6, 1

- [36] Hongyang Wei, Baixin Xu, Hongbo Liu, Cyrus Wu, Jie Liu, Yi Peng, Peiyu Wang, Zexiang Liu, Jingwen He, Yidan Xietian, Chuanxin Tang, Zidong Wang, Yichen Wei, Liang Hu, Boyi Jiang, William Li, Ying He, Yang Liu, Xuchen Song, Eric Li, and Yahui Zhou. Skywork unipic 2.0: Building kontext model with online rl for unified multimodal model, 2025. 6, 1
- [37] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 6
- [38] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 3, 6, 1
- [39] Junfeng Wu, Yi Jiang, Chuofan Ma, Yuliang Liu, Hengshuang Zhao, Zehuan Yuan, Song Bai, and Xiang Bai. Liquid: Language models are scalable and unified multi-modal generators. arXiv preprint arXiv:2412.04332, 2024. 3
- [40] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024. 3
- [41] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Showo2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025. 6, 1
- [42] Wulin Xie, Yi-Fan Zhang, Chaoyou Fu, Yang Shi, Bingyan Nie, Hongkai Chen, Zhang Zhang, Liang Wang, and Tieniu Tan. Mme-unify: A comprehensive benchmark for unified multimodal understanding and generation models. arXiv preprint arXiv:2504.03641, 2025. 2, 3
- [43] Yan Yang, Haochen Tian, Yang Shi, Wulin Xie, Yi-Fan Zhang, Yuhao Dong, Yibo Hu, Liang Wang, Ran He, Caifeng Shan, et al. A survey of unified multimodal understanding and generation: Advances and challenges. Authorea Preprints,

2025. 2

- [44] Zeyuan Yang, Xueyang Yu, Delin Chen, Maohao Shen, and Chuang Gan. Machine mental imagery: Empower multimodal reasoning with latent visual tokens. arXiv preprint arXiv:2506.17218, 2025. 5
- [45] Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025. 4, 5

## RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark

### Supplementary Material

We provide visualization results of representative examples for each subtask of RealUnify, along with the overall task distribution and the manual annotation and verification process in Section A. In addition, implementation details are outlined in Section B to enhance the reproducibility of our results. Finally, in Section C, we present common failure modes of unified models in generation tasks.

###### A. Benchmark Details

###### A.1. Representative Examples from RealUnify

In order to comprehensively convey the characteristics of tasks in RealUnify, two representative examples are presented for each task. Figure 7, 8, 9, and 10 present examples of the Understanding Enhances Generation (UEG) tasks and Generation Enhances Understanding (GEU) tasks, respectively.

###### A.2. Task Distribution

Table 6 presents the distribution of task instances across different categories in RealUnify. Each task is evaluated under both direct and stepwise settings. In the latter, the evaluation is decomposed into two parts: one focusing on visual understanding and the other on generation, thereby allowing a more fine-grained assessment of the model’s reasoning process.

###### A.3. Dataset Annotation and Verification

To construct the UEG benchmark, we recruit 10 human experts to manually design all prompts and their corresponding question lists. These contributors consist of senior undergraduate and doctoral students specializing in artificial intelligence, each possessing substantial expertise in multimodal understanding and image generation. After data collection, every sample undergoes a rigorous validation process, where 3 independent reviewers examine its correctness, objectivity, and answer uniqueness. The reviewers are themselves doctoral students in artificial intelligence, ensuring a high level of professional scrutiny and annotation reliability.

For the GEU tasks—including Mental Reconstruction, Mental Tracking, and Cognitive Navigation—we design automated data-construction pipelines and complement them with manual filtering and verification to ensure both diversity and correctness of the samples. In particular, the maps used in the Cognitive Navigation task originate from the Google Search API. Each sample is further examined by three independent reviewers, and it is retained only if all reviewers

approve it. For the Attentional Focusing task, we sample instances from two existing benchmarks, BLINK [10] and HR-Bench [33], followed by an additional round of human verification to ensure annotation reliability.

###### B. Experiment Details

###### B.1. Evaluation Setup

We evaluate a total of 12 unified models on RealUnify, including 11 leading open-source models and 1 cutting-edge proprietary model.

For the proprietary model, we evaluate Gemini 2.5 Flash Image (also known as “Nano Banana”) [12] using the official API, gemini-2.5-flash-image-preview.

For open-source models, we select BAGEL-7B [8], OmniGen2 [38], Ovis-U1-3B [31], UniWorld-V1 [21], UniPic2-Metaquery-9B [36], OneCAT-3B [16], MIO [35], ILLUME+ [14], Show-o2 [41], Janus-Pro [5], and BLIP3o [4]. All models are evaluated using the official default or recommended settings for inference.

In the Understanding Ehances Generation (UEG) tasks, we use the state-of-the-art Gemini 2.5 Pro [7] as the judge model to evaluate the generated images through a pollingbased method. The evaluation is performed through the official gemini-2.5-pro API.

###### B.2. Evaluation Prompt

For the Understanding Enhances Generation (UEG) tasks, when polling the generated images using Gemini 2.5 Pro [7], we use the prompt provided in Table 7.

For the Generation Ehances Understanding (GEU) tasks, since the tasks are presented in the multiple-choice format, we provide the prompt for the multiple-choice questions in

- Table 8. In the stepwise evaluation of the Understanding Enhances

Generation (UEG) tasks, the models first need to refine the original prompt. The corresponding prompt is provided in

- Table 9. In the stepwise evaluation of the Generation Enhances

Understanding (GEU) task, each task is decomposed, with image generation (editing) performed first, followed by visual understanding. Tables 10, 11, 12, and 13 present the prompts used for image generation (editing) in the Mental Reconstruction, Mental Tracking, Attentional Focusing, and Cognitive Navigation tasks, respectively.

[Figure 118]

###### UEG World Knowledge

UEG Commonsense Reasoning

###### Protocol 1: Direct

Protocol 1: Direct

|Prompt:<br><br>The animal on the left side of Australia's national emblem.|
|---|

|Prompt:<br><br>A glass jar dropping on the floor and shattering, with water splashing outward.|
|---|

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Model

Model

✘ Intact, no shattering

✘ Boar

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Refine it to ensure the generated image fully aligns with the given conditions.

Refine it to ensure the generated image fully aligns with the given conditions.

User

User

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

A glass jar…, its bottom shattering into many sharp fragments, with water…

A kangaroo, the animal on the left side of Australia's national emblem.

Model Model

Model Model

Kangaroo

Shattering

[Figure 133]

[Figure 134]

(a) World Knowledge

(b) Commonsense Reasoning

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

###### UEG Mathematical Reasoning

UEG Logical Reasoning

###### Protocol 1: Direct

Protocol 1: Direct

|Prompt:<br><br>A tree with 8 birds perched on its branches, and another 6 birds are leaving the tree. Draw the birds on the tree.|
|---|

|Prompt:<br><br>Three books in a row on a table. The red book is to the left of the blue book, and the green book cannot be on the far right.|
|---|

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Model

Model

✘ Numerous birds

✘ On the far right

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Refine it to ensure the generated image fully aligns with the given conditions.

Refine it to ensure the generated image fully aligns with the given conditions.

User

User

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

Three books in a row on a

|Two birds perched on a tree.|
|---|

table: the red, the green & the blue from left to right.

Model Model

Model Model

Two birds

Not on the far right

[Figure 153]

[Figure 154]

(c) Mathematical Reasoning

(d) Logical Reasoning

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

###### UEG Scientific Reasoning

###### UEG Code-to-Image

###### Protocol 1: Direct

|Prompt:<br><br>Phenolphthalein indicator is added to a potassium hydroxide (KOH) solution.|
|---|

###### Protocol 1: Direct

|Prompt:<br><br>Code: num = int(input()) result = (num * num) % 4 + 2 things = ['pencils.', 'pen.', 'cats.', 'dogs.’] print(result,things[result])<br><br>Given the input: 37, generate the image based on the output of code execution.|
|---|

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Model

Model

✘ Nonsense text

✘ Blue powder

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Refine it to ensure the generated image fully aligns with the given conditions.

Refine it to ensure the generated image fully aligns with the given conditions.

User

User

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

|Generate an image that shows exactly three cats.|
|---|

|A beaker filled with…, showing bright pink-color.|
|---|

Model

Model Model

Three cats

[Figure 173]

Bright pink solution

[Figure 174]

Model

(e) Scientific Reasoning

(f) Code-to-Image

Figure 7. Examples of Understanding Enhances Generation (UEG) tasks in RealUnify.

###### C. Common Failure Modes of Unified Models in Generation Tasks

Even state-of-the-art unified models still exhibit typical failure modes during image generation, including attribute en-

[Figure 178]

GEU Mental Reconstruction

GEU Mental Tracking

|Question:<br><br>Among the people standing in the back row, when counting from right to left, which position is the shortest person? A. 2 B. 3. C. 4 D. 5<br><br>[Figure 179]|
|---|

|Question:<br><br>Simultaneously change the yellow segments to pink, the black to red,…, what digits do the resulting red segments form? A. 4, 2. B. 1, 2. C. 9, 6. D. 9, 0<br><br>[Figure 180]|
|---|

Protocol 1: Direct

Protocol 1: Direct

[Figure 181]

[Figure 182]

|B|
|---|

|D|
|---|

Model

Model

[Figure 183]

[Figure 184]

[Figure 185]

Restore the image that has been shuffled by patches…

Apply the transformations and modifications to the contents of the image.

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

|D|
|---|

|B|
|---|

User Model

Model

User Model

Model

(a) Mental Reconstruction

(b) Mental Tracking

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

GEU Attentional Focusing

GEU Cognitive Navigation

|Question:<br><br>How many green train carts are in the picture? A. 3 B. 1 C. 2 D. 0<br><br>[Figure 197]|
|---|

|Question:<br><br>[Figure 198]<br><br>Which point on the path from the orange triangle to the blue triangle is closest to the destination? A. A B. B C. C D. D|
|---|

Protocol 1: Direct

Protocol 1: Direct

[Figure 199]

[Figure 200]

|D|
|---|

|C|
|---|

Model

Model

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Highlight the regions of the image that are relevant to the question.

Mark the path(s) in the image that are relevant to the question.

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

|B|
|---|

|D|
|---|

User

Model

Model

User Model

Model

(c) Attentional Focusing

(d) Cognitive Navigation

Figure 8. Examples of Generation Enhances Understanding (GEU) tasks in RealUnify.

tanglement, inaccurate quantity, attribute fidelity errors, and confused spatial relationships. We illustrate these common failure modes in Figure 11 and Figure 12.

confusion, such as generating features of a male lion instead. These errors reveal the clear shortcomings and typical failure modes of unified models in the generation process, limiting their performance on more complex tasks. In particular, for challenging tasks such as RealUnify, which require the synergy of multiple capabilities, these issues may become significant bottlenecks.

As shown in Figure 11, when the instruction involves generating multiple objects or objects of different types with distinct attributes, the model often exhibits attribute mixing between different objects and mismatches in object quantity. In addition, when the objects to be generated have specific or complex attributes and structures, the model is also prone to insufficient fidelity. Moreover, the accurate realization of spatial relationships among multiple objects remains a common issue for the model. Figure 12 exposes several other problems of the model. First, in generating fine-grained features such as fingers and text, the model often suffers from detail loss, distortion, and deformation. Second, the model is also prone to generating scenes that violate common sense and physical laws. Finally, even for common and clearly defined objects (e.g., a lioness), the model shows severe

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

UEG Commonsense Reasoning

###### UEG World Knowledge

Protocol 1: Direct

|Prompt:<br><br>Someone opening a freezer on a humid day, with a small puff of fog spilling out.|
|---|

###### Protocol 1: Direct

|Prompt:<br><br>The most well-known egglaying mammal commonly found in Australia.|
|---|

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Model

Model

✘ Not ”opening”

✘ hamster

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

Refine it to ensure the generated image fully aligns with the given conditions.

Refine it to ensure the generated image fully aligns with the given conditions.

[Figure 223]

User

User

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

A person standing…, using

A platypus, …with its ducklike bill, webbed feet, and brown fur clearly visible.

one hand to pull the freezer door open…

Model Model

Model Model

opening

[Figure 229]

platypus

[Figure 230]

(a) World Knowledge

(b) Commonsense Reasoning

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

###### UEG Mathematical Reasoning

UEG Logical Reasoning

###### Protocol 1: Direct

Protocol 1: Direct

|Prompt:<br><br>A table with some oranges, which could be divided into 3 baskets with 2 oranges each. Draw all the oranges on the table.|
|---|

|Prompt:<br><br>Three trees on a lawn, one with red leaves, one with green leaves, and one with yellow leaves. The green tree is shorter than the yellow tree but taller than the red tree.|
|---|

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Model

Model

✘ Numerous oranges

✘ Green tree is shortest

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

Refine it to ensure the generated image fully aligns with the given conditions.

Refine it to ensure the generated image fully aligns with the given conditions.

User

User

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

…the first tree has red leaves, the second…the trees get taller.

|A wooden table with six fresh oranges...|
|---|

Model Model

Model Model

Six oranges

Red tree is shortest

[Figure 249]

[Figure 250]

(c) Mathematical Reasoning

(d) Logical Reasoning

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

###### UEG Scientific Reasoning

###### UEG Code-to-Image

###### Protocol 1: Direct

|Prompt:<br><br>Phenolphthalein indicator is added to a potassium hydroxide (KOH) solution.|
|---|

###### Protocol 1: Direct

|Prompt:<br><br>Code: animal_tuple = ('cats', 'dogs','rabbits', 'birds') w=int(input())%4 + 2 for idx, animal in enumerate(animal_tuple):<br><br>if idx == w : print(f"{w+1} white {animal}")<br><br>Given the input: 4, generate the image based on the output of code execution.|
|---|

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

Model

Model

✘ Numerous animals

✘ Blue powder

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

Refine it to ensure the generated image fully aligns with the given conditions.

Refine it to ensure the generated image fully aligns with the given conditions.

User

User

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

|A beaker filled with…, showing bright pink-color.|
|---|

|An image of 3 white rabbits.|
|---|

Model

Model Model

Three rabbits

[Figure 269]

Bright pink solution

Model

[Figure 270]

(e) Scientific Reasoning

(f) Code-to-Image

###### Figure 9. Examples of Understanding Enhances Generation (UEG) tasks in RealUnify.

[Figure 271]

[Figure 272]

[Figure 273]

GEU Mental Tracking

[Figure 274]

GEU Mental Reconstruction

|Question:<br><br>Simultaneously change the yellow segments to pink, the black to red,…, what digits do the resulting red segments form? A. 4, 2. B. 1, 2. C. 9, 6. D. 9, 0<br><br>[Figure 275]|
|---|

Protocol 1: Direct

|Question:<br><br>How many birds are there in the picture?<br><br>A. 7 B. 5 C. 6 D. 4<br><br>[Figure 276]|
|---|

Protocol 1: Direct

[Figure 277]

[Figure 278]

|D|
|---|

|A|
|---|

Model

Model

[Figure 279]

[Figure 280]

[Figure 281]

Apply the transformations and modifications to the contents of the image.

[Figure 282]

Restore the image that has been shuffled by patches…

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

|B|
|---|

|C|
|---|

User Model

Model

User Model

Model

(a) Mental Reconstruction

(b) Mental Tracking

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

GEU Attentional Focusing

GEU Cognitive Navigation

|Question:<br><br>What are the hazard numbers displayed on the left side storage container's label? A. 2,3 B. 1,2 C. 1 D. 1,3<br><br>[Figure 293]|
|---|

|Question:<br><br>On the shortest path from Africa Forest to Carnivore Forest, which place will we pass by? A. Asian Forest B. Europe Forest C. Australia Forest D. Kids Zone<br><br>[Figure 294]|
|---|

Protocol 1: Direct

Protocol 1: Direct

[Figure 295]

[Figure 296]

|C|
|---|

|C|
|---|

Model

Model

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

Highlight the regions of the image that are relevant to the question.

Mark the path(s) in the image that are relevant to the question.

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

|D|
|---|

|B|
|---|

User

Model

Model

User Model

Model

(c) Attentional Focusing

(d) Cognitive Navigation

###### Figure 10. Examples of Generation Enhances Understanding (GEU) tasks in RealUnify.

Table 6. Distribution of task instances across different categories in RealUnify. Each task is evaluated under both direct and stepwise settings, where stepwise evaluation further decomposes the process into a visual understanding problem and a generation problem.

#### Task Category Task #Number (Direct / Stepwise)

World Knowledge 100 / 100 Commonsense Reasoning 100 / 100 Mathematical Reasoning 100 / 100 Logical Reasoning 100 / 100 Scientific Reasoning 100 / 100 Code-to-Image 100 / 100

Understanding Enhances Generation

Mental Reconstruction 100 / 100 Mental Tracking 100 / 100 Attentional Focusing 100 / 100 Cognitive Navigation 100 / 100

Generation Enhances Understanding

Total - 1,000 / 1,000

- Table 7. Polling prompt using Gemini 2.5 Pro as the judge model in UEG tasks.

[Image] Please answer the following question based on the image: Question: [Question]

You should only reply yes or no, and do not provide any other extra content.

- Table 8. Evaluation prompt for the multiple-choice question in GEU tasks.

[Image] Select the best answer to the following multiple-option question based on the image. Respond with only the letter (A, B, C, or D) of the correct option. Question: [Question] Option:

- A. [Option A]
- B. [Option B]
- C. [Option C]
- D. [Option D] The best answer is:

- Table 9. Prompt for Understanding Enhances Generation (UEG) tasks.

Here is the prompt for image generation: [Prompt] Please refine it into a simple, direct, and unambiguous form to ensure the generated image fully aligns with the given description and conditions. Respond only with the refined prompt, without adding anything else.

- Table 10. Prompt for stepwise evaluation of Mental Reconstruction tasks.

[Image] Please restore the image that has been shuffled by patches, without adding extra content or altering the original image.

- Table 11. Prompt for stepwise evaluation of Mental Tracking tasks.

[Image] Here is the question: [Question] Please apply the corresponding transformations and modifications to the contents of the image according to the question.

- Table 12. Prompt for stepwise evaluation of Attentional Focusing tasks.

[Image] Here is the question: [Question] Please highlight the regions of the image that are relevant to the question.

- Table 13. Prompt for stepwise evaluation of Cognitive Navigation tasks.

###### [Image] Here is the question: [Question] Please mark the path(s) in the image that are relevant to the question.

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

BAGEL OneCAT UniWorld-V1

UniPic2

Attribute Entanglement (rabbits and chickens)

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

BAGEL

OneCAT UniWorld-V1 UniPic2

Quantity Accuracy (8 flowers)

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

BAGEL

OneCAT BLIP3-o OmniGen2

Attribute Fidelity (Olympic Rings)

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

BAGEL

Show-o2 UniWorld-V1 Nano Banana

Positional Alignment (the green book cannot be on the far right)

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

BAGEL UniPic2 UniWorld-V1

Ovis-U1

Fine-grained Detail (hands and fingers)

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

BAGEL

Show-o2 UniWorld-V1 Ovis-U1

Physical Law (the potato should sink to the bottom)

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

BAGEL

OneCAT BLIP3-o Ovis-U1

Text Distortion (font distortion, warping, and meaningless content)

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

BAGEL

Janus-Pro MIO ILLUME+

Object Misclassification (the right side should be a lioness)

