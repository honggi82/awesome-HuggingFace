# arXiv:2407.01284v1[cs.AI]1Jul2024

## WE-MATH: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?

Runqi Qiao1∗, Qiuna Tan1∗, Guanting Dong1, Minhui Wu2, Chong Sun2, Xiaoshuai Song1, Zhuoma GongQue1, Shanglin Lei3, Zhe Wei1, Miaoxuan Zhang1, Runfeng Qiao4, Yifan Zhang1, Xiao Zong1, Yida Xu1, Muxi Diao1, Zhimin Bao2, Chen Li2, Honggang Zhang1† 1Beijing University of Posts and Telecommunications,2Wechat, Tencent Inc., 3Huazhong University of Science and Technology, 4Beijing Institute of Technology https://We-Math.github.io

#### Abstract

Visual mathematical reasoning, as a fundamental visual reasoning ability, has received widespread attention from the Large Multimodal Models (LMMs) community. Existing benchmarks focus more on the result-oriented performance, but neglecting the underlying principles in knowledge acquisition and generalization. Inspired by human-like mathematical reasoning, we introduce WE-MATH, the first benchmark specifically designed to explore the problem-solving principles beyond the end-to-end performance. We meticulously collect and categorize 6.5K visual math problems, spanning 67 hierarchical knowledge concepts and 5 layers of knowledge granularity. We firstly decompose composite problems into sub-problems according to the required knowledge concepts and introduce a novel four-dimensional metric, namely Insufficient Knowledge (IK), Inadequate Generalization (IG), Complete Mastery (CM), and Rote Memorization (RM) to hierarchically assess inherent issues in LMMs’ reasoning process. With WE-MATH, we conduct a thorough evaluation of existing LMMs in visual mathematical reasoning and reveal a negative correlation between solving step and problem-specific performance. We confirm the IK issue of LMMs can be effectively improved via knowledge augmentation strategy. More notably, the primary challenge of GPT-4o has significantly transitioned from IK to IG, establishing it as the first LMM advancing towards the knowledge generalization stage. In contrast, other LMMs exhibit a marked inclination towards Rote Memorization – they correctly solve composite problems involving multiple knowledge concepts, yet fail in answering sub-problems. We anticipate that WE-MATH will open new pathways for advancements in visual mathematical reasoning for LMMs. The WE-MATH data and evaluation code are available at https://github.com/We-Math/We-Math.

#### 1 Introduction

“I think, therefore I am.” — René Descartes

Human cognitive and reasoning patterns have profoundly shaped the progress of deep learning [1]. Initially, the design of neural networks [2] is inspired by the brain’s neuronal mechanisms. It uses convolution kernels and hierarchical network to mimic human cognitive process of knowledge acquisition. Recently, Transformers [3] employ attention mechanisms to handle multiple information

∗Equal contribution. †Corresponding author

Preprint. Under review.

| | |
|---|---|
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |

GPT-4o Gemini-1.5-Pro G-LLaVA-13B LLaVA-NeXT-110B

MiniCPM-Llama3-V 2.5

InternVL-Chat-V1.5

[Figure 1]

Understanding and Conversion of Units

Angles and Length

Position

Correspondence of Coordinates and Positions

Calculation of Solid Figures

Understanding of Solid Figures

Route Map

Calculation of Plane Figures

Direction

Understanding of Plane Figures Basic Transformations of Figures

Cutting and Combining of Figures

Figure 1: Overview of LMMs’ performances on WE-MATH. Figures from left to right illustrates the (1) accuracy of different LMMs on various problem-solving steps, (2) the performance in different visual mathematics categories and (3) the result in knowledge based reasoning evaluation.

flows and quickly focus on critical content, thereby achieving more efficient and in-depth sequential learning. Owing the scalability of the Transformer architecture and pre-training techniques, Large Language Models (LLMs) [4, 5, 6, 7] and Large Multimodal Models (LMMs) [8, 9, 10, 11, 12, 13, 14, 15, 16] showcases strong reasoning abilities that parallel human performance across a wide range of tasks and provide a glimpse into the early outlines of Artificial General Intelligence (AGI).

Mathematical reasoning is a critical capability of foundational models. Existing methods employ Chain of Thought (COT) [17], Program of Thought (POT) [18, 19], Tool-integrated techniques [20, 21] and data augmentation strategies [22, 23, 24, 25] to guide LLMs towards emulating human-like reasoning patterns. In a more challenging scenario, Visual mathematical reasoning requires the model to accurately decode the visual information in image and perform reasoning based on the textual problem. With the rapid advancements of large multimodal models (LMMs) [26, 27], researchers progressively utilize the LMMs for solving visual mathematical problems [28, 29]. These studies provide valuable insights into the ongoing improvements in multi-modal logical thinking capabilities.

To systematically evaluate visual mathematical reasoning capabilities, previous efforts [30, 31, 32, 33] have focused on challenging geometric problems. Recently, several benchmarks [34, 35] expand the scope to include a wider range of disciplines. However, these benchmarks rely solely on end-to-end results for assessment, which fails to identify inherent issues within the LMMs’ reasoning process. Moreover, MathVerse [35] attempt to directly evaluate reasoning paths based on reference answers, but limitations remain due to the knowledge-intensive nature of mathematical reasoning. While noticing that humans solve complex math problems through gradually mastering and generalizing the knowledge concepts [36], we claim a fair evaluation of a model’s reasoning process should be based on knowledge concepts. Therefore, we pose two questions about mathematical reasoning evaluation:

- Q1: Does the correct answer truly reflect LMM’s ability to reason through such problems accurately?
- Q2: Does an incorrect answer suggest a lack of foundational knowledge in LMM’s reasoning process?

As the response, we present WE-MATH, a pioneering benchmark for conducting an in-depth analysis of the underlying principles of LMMs in visual mathematical reasoning. WE-MATH consists of over 6.5K meticulously selected visual math problems, which can be categorized into 5 layers of knowledge granularity across 67 knowledge concepts for ensuring a comprehensive coverage. We observe that real-world math problems typically encompass multiple foundational knowledge concepts, and their difficulty is directly related to the number of concepts involved. Upon this, we decouple the model’s ability to solve composite problems with k knowledge concepts into two stages:

- 1) LMMs can solve k individual sub-problems corresponding its knowledge concept;
- 2) LMMs reason out the final answer by integrating the k individual knowledge concepts. The above process can be formulated as follows:

k

P(yi|xi) · Preason, (1)

P(Y |X) =

i=1

where (X,Y ) and (xi,yi) denote the (question,answer) pairs in a composite mathematical problem and the i-th sub-problem, respectively. Preason stands for the LMMs’ reasoning capacity. It is evident that assessing the reasoning process cannot be based solely on final answers. To decompose a composite problem into individual sub-problems according to the invoked knowledge concept, we select 1.5k high-quality problems with multiple knowledge concepts in WE-MATH. Following equation 1, these composite problems are gradually decomposed by expert annotators into one-step problem (xi,yi). Motivated by human reasoning patterns, WE-MATH further introduces a fourdimensional metric to precisely evaluate the inherent gaps in LMMs’ problem-solving abilities, namely Insufficient Knowledge (IK), Inadequate Generalization (IG), Complete Mastery (CM), and Rote Memorization (RM). To further tackle the fundamental IK issue, we propose a heuristic knowledge concept augmented (KCA) strategy, constructing descriptions for 67 knowledge concepts from Wikipedia [37] and textbooks, thereby providing essential knowledge for LMMs’ reasoning.

- Figure 1 illustrates our overview experimental results. Not surprisingly, GPT-4o [38] achieves the best overall performance across different visual mathematics categories. Closed-source LLMs (GPT-4V, Gemini 1.5 Pro) and LMMs with larger parameter scales (LLaVA-NeXT-110B [39]) generally exhibit superior visual mathematical reasoning capabilities. However, most LMMs perform significantly worse on multi-step problems compared to one-step problems, suggesting that the number of knowledge concepts is positively correlated with the question’s difficulty and negatively correlated with LMM performance. In specialized disciplines, most LMMs excel in calculation but consistently struggle with fine-grained visual measurement ("Angles and Length").

For reasoning evaluation, we emphasize that mastery of knowledge concepts is fundamental. Unfortunately, most LMMs still suffer from Insufficient Knowledge issue, especially smaller-scale models (e.g., over 350 IK issues in LLaVA-1.6-7B and DeepSeek-VL-1.3B). GPT-4o significantly addresses this knowledge gap, establishing it as the first LMM advancing towards the knowledge generalization stage. More notably, several LMMs still exhibit a marked inclination towards Rote Memorization (e.g. G-LLaVA-13B nearly 36% in RM (Loose)), raising doubts about whether current LMMs truely possess the mathematical reasoning capability. In addition, our proposed KCA strategy substantially reduces the IK issue in LMMs, and error analysis further provides empirical guidance towards human-like reasoning. We anticipate that WE-MATH will open new pathways for advancements in visual mathematical reasoning in LMMs.

#### 2 WE-MATH

Overview of WE-MATH. As previously mentioned, existing benchmarks tend to be result-oriented, while overlooking the essence of solving mathematical problems. This leads to the generation of some counterintuitive evaluation conclusions. For example, conclusions in MathVista [34] indicate that LMMs exhibit superior performance on university-level problems compared to elementary-level ones. Different from existing benchmarks, as shown in Figure 2, WE-MATH is constructed around textbook knowledge units, decomposing composite problem solutions into sub-problems based on the knowledge concepts. WE-MATH has the following characteristics:

- (1) Hierarchical Knowledge Structure. WE-MATH strictly adheres to the knowledge presented in mathematics textbooks, featuring a rigorous hierarchical and multi-category architecture. It ensures the independence of knowledge concepts within the same level, while establishing logical relationships among concepts at different hierarchical levels.
- (2) Knowledge based Reasoning Evaluation. WE-MATH is designed to explore how LMMs solve problems. Drawing upon that humans tackle problems incrementally by leveraging fundamental knowledge concepts, we break down complex mathematical problems into more manageable subproblems. Furthermore, we employ diverse measurement dimensions for meticulous evaluations.
- (3) Knowledge Concept Augmentation. To alleviate the inherent issues during the problem-solving process, we heuristically introduce descriptions for 67 knowledge concepts from Wikipedia and textbooks, thereby providing essential knowledge support for the reasoning processes of LMMs.

##### 2.1 Hierachial Structured Dataset Composition

Hierachial Knowledge Structure. WE-MATH emphasizes fundamental math skills, believing that complex mathematical reasoning is built upon foundation of basic mathematical reasoning processes.

[Figure 2]

###### We-Math Knowledge Card

[Figure 3]

[Figure 4]

[Figure 5]

###### Total samples

Decomposable samples

Knowledge system

Understanding

Direction RouteMap

Nodes Layers

99 5

Newly collected images 6307 Unique images 6524

1674 Unique questions 1740

[Figure 6]

Conversion and Units of

Newly collected questions

[Figure 7]

Correspondence of Coordinates and Positions

[Figure 8]

[Figure 9]

Angles and

·Southeast, Southwest, Northeast, Northwest Directions ·Cardinal Directions (East, South, West, North)

·Front-Back Position ·Up-Down Position ·Left-Right Position

[Figure 10]

Length

·Determining the Position of Objects Based on Direction, Angle, and Distance ·Describing Simple Routes Based on Direction and Distance

·Representing Positions Using Ordered Pairs ·Finding Positions Based on Ordered Pairs

[Figure 11]

[Figure 12]

Position

·Combining and Dividing Solids ·Division of Plane Figures ·Combining Plane Figures ·Tessellation of Figures ·Folding Problems of Figures ·Observing Figures ·Properties and Understanding of Parallelograms ·Properties and Understanding of Triangles ·Properties and Understanding of Trapezoids ·Properties and Understanding of Rectangles ·Properties and Understanding of Squares ·Understanding Triangular Rulers ·Parallel ·Understanding and Representing Angles

[Figure 13]

[Figure 14]

·Axial Symmetry ·Translation ·Rotation

of Plane FiguresUnderstanding

ofSolidFigures

Understanding

·Sum of Interior Angles of Other Polygons ·Sum of Interior Angles of Triangles ·Calculation and Comparison of Angles ·Area of Parallelograms ·Area of Squares ·Area of Triangles ·Area of Trapezoids ·Area of Circles ·Area of Rectangles ·Perimeter of Parallelograms ·Perimeter of Triangles ·Perimeter of Trapezoids ·Perimeter of Rectangles ·Perimeter of Squares ·Perimeter of Circles

[Figure 15]

[Figure 16]

[Figure 17]

###### We-Math

[Figure 18]

Plane FiguresCalculation of

Calculation of Solid Figures

·Perpendicularity ·Understanding Circles

·Distance Between Two Points ·Area of Squares ·Understanding Line Segments, Lines, and Rays ·Understanding Sectors

[Figure 19]

[Figure 20]

·Observing Objects ·Expanded View of Cylinders ·Expanded View of Rectangular Cuboids ·Expanded View of Cubes ·Properties of Cylinders ·Properties of Cones ·Properties and Understanding of Rectangular Cuboids ·Properties and Understanding of Cubes

[Figure 21]

Cuttingand Combiningof

[Figure 22]

·Surface Area of Rectangular Cuboids ·Surface Area of Cylinders ·Volume and Capacity of Cubes ·Surface Area of Cubes ·Volume and Capacity of Cylinders ·Volume and Capacity of Cones ·Volume and Capacity of Rectangular Cuboids

Basic Transformations ofFigures

Figures

[Figure 23]

[Figure 24]

[Figure 25]

·Conversion Rates and Calculations Between Area Units ·Conversion Rates and Calculations Between Volume Units ·Conversion Rates and Calculations Between Length Units

[Figure 26]

·Understanding Angles (Using a Protractor) ·Understanding Length (Using a Ruler)

- Figure 2: Overview diagram and the statistics of WE-MATH. The left and right side shows the first two layers of WE-MATH’s categories and information of different samples and terminal nodes.

Based on extensive research, mathematical problems are categorized into five distinct types, namely Plane Figures, Solid Figures, Transformations and Movements of Shapes, Positions and Directions, Measurements. These five categories can be decomposed into 12 typical problems, which are further decomposed as 67 knowledge concepts (terminal nodes in the structure). We collect problems according to this tree structure and constrain that each terminal node contains a strict range of 10-40 samples. This rule ensures data balance across domains.

Data Collection and Annotation. All problems (6.5K) in WE-MATH are sourced from publicly authoritative mathematics websites and subsequently organized based on our defined knowledge structure. We employ three expert annotators to manually label each question with knowledge concepts. Cross-validation is performed to ensure at least two experts have identical annotations for the same question. Samples with notably inconsistent labels will be considered of low quality and subsequently excluded. To prepare for the subsequent decomposition of problems, we further annotate problem-solving steps based on the knowledge concepts labels. We categorize each problem into three distinct classes: "One-Step", "Two-Step", and "Three-Step". This categorization enables us to gain a deeper understanding on how LMMs solve problems. Further details about annotation can be found in Appendix. After the annotation, all problems are double checked by an expert team in terms of three aspaces: (1) The consistency between the questions and dagrams; (2) The correctness of the answers to the questions; (3) The alignments between problems and the 67 knowledge concepts.

##### 2.2 Knowledge based Reasoning Evaluation

Problem Definition. For the visual mathematical reasoning task, given text question Qi, image Ii and corresponding answer Ai. We define the LMMs evaluation dataset Deval = {(Qi,Ii,Ai)|Ki,Ci}Ni=1.

where Ki and Ci are two prior constraints for question Qi. In detail, Ki = {ki}Mi=1 denote M knowledge concepts within the question. Ci represents the prerequisite conditions needed to solve the problem Qi (see Figure 3 for example). For the convenience of presenting, we define the problem containing k knowledge concepts as a "k-step problem" in our paper.

Knowledge-based Data Decomposition. Real-world mathematical problems are composed of multiple atomic knowledge concepts. However, existing benchmarks usually overlook this information, leading to unreasonable evaluation results. Inspired by Euclid’s Elements [36], we argue that the evaluation of mathematical reasoning ability in LMMs essentially involves assessing their mastery of fundamental knowledge concepts. It is quite a natural and objective way to exploit basic knowledge concepts for reasoning evaluation of LMMs. Given an i-th test sample {(Qi,Ii,Ai)|Ki,Ci} ∈ DWE-MATH with M concepts Ki = {kim}Mm=1, we ask human experts to decompose each problem step by step into M sub-problems based on knowledge concepts, which can be formulated as:

{(qim,imi ,ami )|kim,cmi }Mm=1 = Decompose

{(Qi,Ii,Ai)|Ki,Ci} (2)

(Qi,Ii,Ai)∈DWE-MATH

[Figure 27]

- Figure 3: The pipeline of knowledge-based data decomposition (an example of a three-step problem in WE-MATH).

where ki, ci denote the individual knowledge and prior condition for the sub-problem. “Decompose” represents the Human decomposition process based on M knowledge concepts. To ensure logical

coherence of decomposition, the condition cmi is initialized as Ci. Then it is recursively computed by concatenating the answer ami −1 and condition cmi −1 of the m − 1-th concept:

cmi = cmi −1 + am−1

i for m = 2,3,...,M (3)

qiM = Qi aMi = Ai

where “+” denotes the concatenation operation. In addition, the equation

must

be satisfied, which is also a constraint for logical coherence. Finally, we can obtain the original multi-step problem and M one-step sub-problems for reasoning evaluation. The overall pipeline of Knowledge-based Data Decomposition are displayed in the left side of Figure 3.

##### Metric for Reasoning Evaluation.

Based on the decomposed multi-step problems, we further reveal the inherent issues of LMMs in problem-solving process. We feed both the M one-step sub-problems and the original problem into LMMs, and classifying the responses into four categories:

- 1. Insufficient Knowledge (IK): Part of one-step problems contain errors, and the multi-step problem is wrong. It is reasonable because model’s insufficient grasp of single knowledge concept may lead to errors in multi-step problem.
- 2. Inadequate Generalization (IG): One-Step problems are all correct, but the multi-step problem is incorrect. This is also considered reasonable. While LMMs are capable of understanding individual knowledge concepts, they may struggle to generalize that knowledge to solve composite problems.

Evaluation

Evaluation

Strict Metric

###### Loose Metric

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

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 56]

[Figure 57]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Metric for Reasoning Evaluation

Metric for Reasoning Evaluation

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Insufficient Knowledge

Inadequate Generalization

Complete Mastery

Rote Memorization

Insufficient Knowledge

Inadequate Generalization

Complete Mastery

Rote Memorization

Insufficient Knowledge Insufficient Knowledge

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Two-Step problem

Two-Step problem

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

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

[Figure 105]

Step2

Step2

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

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

Step1

Step1

- Figure 4: An example of the four-dimensional metrics for evaluating a two-step problem, using both loose and strict settings.

- 3. Complete Mastery (CM): One-Step problems are all correct, and multi-step problem is also answered correctly. This result demonstrates that the model’s results are both reliable and accurate.
- 4. Rote Memorization (RM): One-Step problems contain errors, but the multi-step problem is answered correctly, which contradicts human logical thinking. If a model can solve composite multi-step problems but fails to answer the one-step problems needed in the process, it raises doubts about the model’s reliability.

Considering IK, IG, and CM, it is evident that results falling under the IG category are generally more preferred compared to those classified as IK. The reason is that IK reflects the model’s struggle with both single and multiple knowledge concepts, while IG shows the model’s proficiency one-step problem. By enhancing the model’s generalization ability in reasoning process, we can potentially shift results from IG to CM. Therefore, we establish a reasoning capability hierarchy as IK < IG < CM. We believe that RM is an unreasonable scenario (models can solve multi-step problems without mastering one-step problems, which completely contradicts human reasoning intuition).

In light of the model’s instability, the current criteria for determining whether a result belongs RM is strict. We thus propose a more flexible loose metric. As illustrated in Figure 4, the TFT and FTT situations in the two-step problems are regard as CM (rather than RM), according to the loose metric. We also discuss the situation of four-dimensional metrics on three-problem in Appendix C. We propose the following metric to judge the reliability of the model’s reasoning process:

SIK =

NIK N

, SIG =

NIG N

, SCM =

NCM N

, SRM =

NRM NRM + NCM

(4)

where N denotes the total number of samples and NIK, NIG, NCM, NRM represents the number of samples for a specific situation. Therefore, we obtain our final reasoning confidence scores:

Scoreaverage = αSIK + βSIG + SCM (5)

where α,β denotes the weight for each case. To ensure the reasoning capability hierarchy is "IK < IG < CM", we control the params α < β < 1, and set the default value of α to 0.0 and β to 0.5.

- 2.3 Knowledge Concept Augmentation

In the previous section, we identify the Insufficient Knowledge (IK) as the foundation challenge in mathematical reasoning. To heuristically tackle this issue, we enlist human experts to create 67 knowledge concept cards, which is essential for LMM’s reasoning process. Initially, expert annotators offer precise summaries derived from the definitions in Euclid’s Elements [36], Wikipedia and textbooks. Subsequently, these experts further condense the content examined by a series of questions related to a specific knowledge concept, extracting crucial knowledge hints for incorporation into the knowledge cards. After several rounds of review, we confirm the accuracy and utility of each card.

- Figure 5 showcases typical knowledge concept cases and their descriptions. Consequently, with a

###### given problem Qi and its respective knowledge concept Ki, LMMs utilize the relevant knowledge cards to deduce the answer Ai. The detailed information of KCA can be found in Appendix.

[Figure 138]

Cutting and Combining of Figures

Tessellation of Figures

###### Folding Problems of Figures Description:

Description:

- 1. The folded figure exhibits the properties of a symmetrical figure, with the fold line as the axis of symmetry, and the two parts of the figure before and after folding are symmetrical about the fold line.
- 2. The two parts of the figure before and after folding are identical.
- 3. The lines connecting corresponding points before and after folding are perpendicular to the fold line and equidistant from the fold line.

- 1. Tiling, or tessellation of a plane, refers to the process of covering a plane using one or more types of shapes that are identical in size and shape, without leaving any gaps or overlapping.
- 2. Common shapes for tiling: square, rectangle, triangle, parallelogram, regular hexagon.
- 3. The sum of the angles at each vertex point of the tiling must equal a full angle (360°). A plane can only be tiled if the sum of the angles at each vertex point equals a full angle.
- 4. Since the sum of the interior angles of a triangle is 180°, using several congruent triangles, each angle can be used twice to form a full angle, so triangles can always tile a plane.
- 5. The sum of the interior angles of any quadrilateral is 360°, and at each vertex point of the tiling, the four angles just form a full angle, so any quadrilateral can tile a plane.
- 6. Each interior angle of a regular hexagon is 120°, and they can also form a full angle, so regular hexagons can tile a plane.

Division of Plane Figures Description:

- 1. Dividing a geometric figure into several parts according to certain requirements is called the partitioning of the figure.
- 2. When a plane figure is divided into two or more new geometric figures, the area of the original figure equals the sum of the areas of the new figures, but the perimeter of the original figure will be shorter than the sum of the perimeters of the resulting figures. If the total length of the dividing lines is a, then the increase in the perimeter after division is 2a.

a a

Combining and Dividing Solids Description:

Combining Plane Figures Description:

- 1. The combination of plane figures refers to the process of combining two or more plane figures to form a new plane figure. This tessellation can be achieved by connecting the edges of the plane figures to form a whole.
- 2. When two or more figures are combined to form a new geometric figure, its area equals the sum of the areas of the original figures, but the perimeter will be shorter than the sum of the perimeters of the original figures. If the total length of the joined edges is a, then the decrease in the perimeter after joining is 2a.

- 1. When two or more geometric solids are combined to form a new solid, the volume of the new solid equals the sum of the volumes of the original solids. However, the surface area of the new solid is less than the sum of the surface areas of the original solids. If the overlapping area is S, the decrease in surface area is 2S.
- 2. When a geometric solid is cut into parts, the sum of the volumes of the parts equals the volume of the original solid. However, the sum of the surface areas of the parts is greater than the surface area of the original solid. If the cutting area is S, the increase in surface area is 2S.

a

S

Figure 5: The cases of knowledge concept cards.

#### 3 Experiment

Evaluation Protocols. To accelerate the evaluation speed, WE-MATH comprises a testmini set with 1740 samples, including 1215 one-step samples, 360 two-step samples, and 165 three-step samples. In subsequent experiments, we utilize the WE-MATH testmini subset for evaluation. For automated evaluation, we standardize all samples into a multiple-choice format. We use regex to match the LMMs’ predictions and then calculate their accuracy against the ground-truth answers for main results. For analyses in section 3.2 and 3.3, we utilize the four-dimensional metric described in section 2.2 for assessment. To avoid LMMs deduce answers from options, we introduce an extra uncertain option to mitigate this issue.

Evaluation Models. We examine the performance of foundation models across two distinct categories on WE-MATH: (a) Closed-source LMMs: GPT-4o [38], GPT-4V [26], Gemini 1.5 Pro [40], Qwen-VL-Max [13], (b) Open-source LMMs: LLaVA-NeXT-110B, LLaVA-NeXT-70B [39], LLaVA1.6-13B, LLaVA-1.6-7B [41], DeepSeek-VL-1.3B, DeepSeek-VL-7B [42], Phi3-Vision-4.2B [43], MiniCPM-Llama3-V 2.5 [44], InternLM-XComposer2-VL-7B [45], InternVL-Chat-V1.5 [46], GLM4V-9B [47], LongVA [48], G-LLaVA-13B [29].

##### 3.1 Main Result

- Table 1 shows the overall performance of different LMMs on One-Step / Two-Step / Three-Step problems and different problem domains. We have the following observations:

The Nums of Knowledge Concepts are negatively correlated with LMMs’ Performance. Regarding problems of varying complexities (one-step vs. two-step vs. three-step), GPT-4o consistently achieve an advantage across all settings. Other closed-source models, such as GPT-4V and Gemini 1.5 Pro, also demonstrate competitive performance. However, most LMMs perform significantly worse on multi-step problems compared to one-step problems. For instance, GPT-4o’s accuracy drops from 72.84% to 43.64%. This trend is even more pronounced in stronger models like LLaVA-NeXT-110B and InternVL-Chat-V1.5. These observations suggest that the number of knowledge concepts in a question is positively correlated with its difficulty and negatively correlated with LMMs’ performance, supporting the rationale for decomposing questions to a certain extent.

- Table 1: Accuracy scores of LMMs on the testmini subset of WE-MATH. The first 3 columns report the overall performance on one-step, two-step, three-step problems, while the other columns display the result on one-step problems in different problem categories. The highest accuracy for closed-source and open-source LMMs is marked in blue and green respectively. (S1: one-step problem, S2: two-step problem, S3: three-step problem, Mem: Measurement, PF: Plane Figures, SF: Solid Figures, TMF: Transformations and Motion of Figures, PD: Position and Direction. AL: Angles and Length, UCU: Understanding and Conversion of Units, CPF: Calculation of Plane Figures, UPF: Understanding of Plane Figures, CSF: Calculation of Solid Figures, USF: Understanding of Solid Figures, BTF: Basic Transformations of Figures, CCF: Cutting and Combining of Figures, Dir: Direction, Pos: Position, RoM: Route Map, CCP: Correspondence of Coordinates and Positions).

Mem PF SF TMF PD

Model S1 S2 S3

UCU AL CPF UPF CSF USF BTF CCF Dir Pos RoM CCP Closed-source

GPT-4o 72.84 58.06 43.64 86.61 39.12 77.35 71.56 84.50 62.27 58.74 69.37 93.10 72.67 47.53 73.33 GPT-4V 65.51 49.17 38.18 82.54 38.42 70.67 60.22 76.58 56.32 57.76 67.67 79.29 57.48 47.80 63.33

Gemini 1.5 Pro 56.13 51.39 33.94 50.99 31.23 61.75 45.03 69.95 57.54 39.24 62.65 68.81 54.13 40.66 60.00 Qwen-VL-Max 40.82 30.28 20.61 19.35 25.26 39.82 41.44 43.64 48.02 43.82 43.39 41.43 35.09 40.66 26.67

Open-source LLaVA-NeXT-110B 53.74 36.94 31.52 39.48 57.72 59.48 53.06 52.25 50.22 54.09 50.76 54.76 55.86 40.11 40.00

LLaVA-NeXT-72B 42.88 35.56 30.91 31.65 25.26 43.25 42.39 46.14 41.76 44.22 51.02 44.29 38.93 32.97 36.67 InternVL-Chat-V1.5 49.38 30.56 28.48 43.95 29.82 52.23 52.06 44.19 48.15 47.05 46.82 65.71 50.47 36.54 36.67

LLaVA-1.6-13B 29.38 25.28 32.73 21.73 23.16 23.37 34.72 25.26 26.36 37.52 41.65 26.90 28.87 37.09 30.00 G-LLaVA-13B 32.43 30.56 32.73 33.33 29.12 32.04 37.88 19.57 33.51 37.12 32.79 31.19 33.21 25.55 40.00

GLM-4V-9B 47.33 37.22 38.18 53.37 37.02 51.32 46.52 50.60 38.22 44.09 45.22 40.95 49.27 36.81 53.33 MiniCPM-LLaMA3-V 2.5 39.75 31.11 29.70 28.57 37.02 40.81 39.82 40.97 38.61 31.96 42.66 40.95 42.70 43.96 43.33

LongVA-7B 43.54 30.56 28.48 24.50 39.82 45.09 40.75 51.85 42.49 45.60 44.56 44.52 40.74 47.53 20.00 LLaVA-1.6-7B 22.96 20.83 15.76 18.45 20.53 16.92 29.63 15.57 18.60 42.67 24.05 17.62 43.31 28.85 26.67

DeepSeek-VL-7B 32.59 26.67 25.45 16.57 35.09 27.27 38.01 24.18 38.65 50.02 30.09 24.52 41.01 51.65 23.33 InternLM-XComposer2-VL-7B 47.00 33.06 33.33 31.25 46.49 47.70 42.57 51.44 43.87 41.13 50.58 65.48 53.87 55.22 40.00

Phi3-Vision-4.2B 42.14 34.17 27.88 28.67 15.96 47.23 38.83 49.99 44.41 28.76 31.22 48.57 49.19 26.37 50.00 DeepSeek-VL-1.3B 31.44 27.78 23.03 27.78 23.86 22.76 36.92 30.36 34.18 44.46 28.29 48.10 41.77 37.09 33.33

MiniCPM-V-2.5 LLaVA1.6-13B G-LLaVA-13B LLaVA1.6-7B DeepSeek-VL-7B InternLM-XC2 Phi-3-Vision DeepSeek-VL-1.3B

GPT-4o GPT-4V Gemini 1.5 Pro Qwen-VL-Max

LLaVA-NeXT-110B LLaVA-NeXT-72B InternVL-Chat-V1.5

LongVA-7B

GLM-4V-9B

[Figure 139]

[Figure 140]

[Figure 141]

Understanding and Conversion of Units

Understanding and Conversion of Units

Understanding and Conversion of Units

Position

Angles and Length

Position

Angles and Length

Position

Angles and Length

Correspondence of Coordinates and Positions

Correspondence of Coordinates and Positions

Correspondence of Coordinates and Positions

Calculation of Solid Figures

Calculation of Solid Figures

Calculation of Solid Figures

Understanding of Solid Figures

Understanding of Solid Figures

Understanding of Solid Figures

Route Map

Route Map

Route Map

Calculation of Plane Figures

Calculation of Plane Figures

Calculation of Plane Figures

Direction

Direction

Direction

Understanding of Plane Figures Basic Transformations of Figures

Understanding of Plane Figures Basic Transformations of Figures

Understanding of Plane Figures Basic Transformations of Figures

Cutting and Combining of Figures

Cutting and Combining of Figures

Cutting and Combining of Figures

(a) Closed-Source Models

(b) Open-Source Models (Larger)

(c) Open-Source Models (Smaller)

Figure 6: The visualization of different LMMs’ performances on each category.

Larger Parameter Scales in LLMs generally achieve Better Generalization Abilites. To explore what role LLM plays in LMMs, we conduct pairwise comparisons on the LMMs with the same LLM backbone (e.g. LLaVA-NeXT-110B vs LLaVA-NeXT-72B; DeepSeek-VL-7B vs DeepSeekVL-1.3B). Focusing on the strict metric, we observe that larger parameter scales in LLMs generally perform better, which reveals that the parameter scales in the text decoder is a key factor in achieving the generalization ability in visual mathematical reasoning.

LMMs excel in Calculation but struggle with Fine-grained Visual Measurement. Focusing on different math categories, GPT-4o still maintains impressive results across various subfields. In contrast, as shown in Figure 6, other LMMs generally struggle with "Angle Measurement" and "Unit Conversion". After analyzing these cases, we reveals that the main challenge for LMMs lies in their inability to perform precise visual angle and unit measurements. Furthermore, most LMMs demonstrate better proficiency in calculation (e.g., Calculations of Solid Figures, Calculations of Plane Figures) compared to conceptual understanding (e.g., Understanding of Solid Figures, Understanding of Plane Figures), which indicates that current LMMs excel at directly applying formulas based on given conditions on but are still limited in understanding and comprehensively applying knowledge.

- Table 2: The performance of different LMMs on four-dimensional metrics for reasoning evaluation. The best and the two worst performances are marked in blue and red (Avg: Scoreaverage).

Strict Loose Avg (↑) IK (↓) IG (↓) CM (↑) RM (↓) Avg (↑) IK (↓) IG (↓) CM (↑) RM (↓)

Model

Closed-source

GPT-4o 42.86% 31.24% (164) 15.24% (80) 35.24% (185) 34.16% (96) 60.57% 31.24% (164) 15.24% (80) 52.95% (278) 1.07% (3) GPT-4V 31.05% 39.81% (209) 14.48% (76) 23.81% (125) 47.92% (115) 51.43% 39.81% (209) 14.48% (76) 44.19% (232) 3.33% (8)

Gemini 1.5 Pro 26.38% 42.86% (225) 11.24% (59) 20.76% (109) 54.77% (132) 46.00% 42.86% (225) 11.24% (59) 40.38% (212) 12.03% (29) Qwen-VL-Max 10.48% 65.14% (342) 7.62% (40) 6.67% (35) 75.52% (108) 25.52% 65.14% (342) 7.62% (40) 21.71% (114) 20.28% (29)

Open-source LLaVA-NeXT-110B 19.24% 50.29% (264) 14.48% (76) 12.00% (63) 65.95% (122) 37.90% 50.29% (264) 14.48% (76) 30.67% (161) 12.97% (24)

LLaVA-NeXT-72B 13.43% 58.86% (309) 7.05% (37) 9.90% (52) 70.95% (127) 31.52% 58.86% (309) 7.05% (37) 28.00% (147) 17.88% (32) InternVL-Chat-V1.5 14.95% 56.19% (295) 13.90% (73) 8.00% (42) 73.25% (115) 32.67% 56.19% (295) 13.90% (73) 25.71% (135) 14.01% (22)

LLaVA-1.6-13B 5.24% 69.14% (363) 3.24% (17) 3.62% (19) 86.90% (126) 22.00% 69.14% (363) 3.24% (17) 20.38% (107) 26.21% (38) G-LLaVA-13B 6.48% 64.19% (337) 4.57% (24) 4.19% (22) 86.59% (142) 22.29% 64.19% (337) 4.57% (24) 20.00% (105) 35.98% (59)

GLM-4V-9B 14.86% 52.95% (278) 9.52% (50) 10.10% (53) 73.10% (144) 35.05% 52.95% (278) 9.52% (50) 30.29% (159) 19.29% (38) MiniCPM-LLaMA3-V 2.5 9.52% 60.19% (316) 9.14% (48) 4.95% (26) 83.85% (135) 28.00% 60.19% (316) 9.14% (48) 23.43% (123) 23.60% (38)

LongVA-7B 11.52% 61.14% (321) 8.95% (47) 7.05% (37) 76.43% (120) 27.71% 61.14% (321) 8.95% (47) 23.24% (122) 22.29% (35) LLaVA-1.6-7B 3.33% 78.29% (411) 2.48% (13) 2.10% (11) 89.11% (90) 13.81% 78.29% (411) 2.48% (13) 12.57% (66) 34.65% (35)

DeepSeek-VL-7B 6.29% 69.14% (363) 4.57% (24) 4.00% (21) 84.78% (117) 20.95% 69.14% (363) 4.57% (24) 18.67% (98) 28.99% (40) InternLM-XComposer2-VL-7B 12.67% 56.38% (296) 10.48% (55) 7.43% (39) 77.59% (135) 30.95% 56.38% (296) 10.48% (55) 25.71% (135) 22.41% (39)

Phi3-Vision-4.2B 10.57% 58.86% (309) 8.95% (47) 6.10% (32) 81.07% (137) 29.81% 58.86% (309) 8.95% (47) 25.33% (133) 21.30% (36) DeepSeek-VL-1.3B 5.90% 71.05% (373) 2.67% (14) 4.57% (24) 82.61% (114) 21.52% 71.05% (373) 2.67% (14) 20.19% (106) 23.19% (32)

###### (a) Strict Metric (b) Loose Metric

Figure 7: The Leaderboard of different LMMs under the strict and loose metric (average score %). "∼" represents an approximate estimate of the total parameters nums in LMMs.

LMMs exhibit Strong Potential for Parameter Compression. In terms of different LMMs, LLaVANeXT-110B demonstrates performance closest to GPT-4. Surprisingly, despite having smaller parameter scales, Phi3-Vision-4.2B and MiniCPM-Llama3-V 2.5 also show competitive performance compared to LLaVA-NeXT-72B. Moreover, the recent GLM-4V-9B and InternVL-Chat-V1.5 have allocated a larger proportion of parameters to the visual encoder (as shown in Table 8), thereby demonstrating notable capabilities. This underscores the importance of optimizing visual representations and suggests that LMMs still have significant potential for parameter compression.significant potential for parameter compression.

##### 3.2 Knowledge based Reasoning Analysis

- Table 2 and Figure 7, 8, 9 illustrate the results of knowledge based reasoning evaluation, including four distinct conditions (IK, IG, CM, RM). We have the following observations:

IK is the Greatest Vulnerability of LMMs. All LMMs consistently demonstrate an Insufficient Knowledge issue during the reasoning process, especially in models with smaller parameter scales (LLaVA-1.6-7B, DeepSeek-VL-1.3B). As discussed in section 2.2, addressing IK is crucial for progressing towards Inadequate Generalization (IG) and Complete Mastery (CM). This knowledge gap in solving one-step problems hinders further progress in reasoning about more composite mathematical problems. This finding also supports the rationale behind our proposed KCA strategy.

GPT-4o’s Main Challenge has gradually shifted from IK to IG, highlighting it as the First LMM towards the Knowledge Generalization Stage. Focusing on IK and IG, GPT-4o exhibits a substantial lead in addressing the IK issue, but the weakest performance in IG. Further analyzing

500

Insufficient Knowledge Inadequate Generalization Complete Mastery Rote Memorization

400

300

Number

200

100

0

LongVA -7B

GPT-4o GPT-4V Gemini

LLaVA -NeXT-110B

GLM -4V-9B

InternVL -Chat-V1.5

InternLM -XC2

LLaVA -NeXT-72B

Phi-3 -Vision

MiniCPM -V 2.5

G-LLaVA -13B

Qwen-VL -Max

LLaVA -1.6-13B

DeepSeek -VL-7B

DeepSeek -VL-1.3B

LLaVA -1.6-7B

-1.5-Pro

- Figure 8: The performance of different LMMs on four-dimensional metrics under strict metric.

GPT-4o GPT-4V Gemini 1.5 Pro

LLaVA -NeXT-110B

GLM -4V-9B

InternVL -Chat-V1.5

InternLM -XC2

LLaVA -NeXT-72B

Phi-3 -Vision

MiniCPM -V 2.5

LongVA -7B

G-LLaVA -13B

Qwen-VL -Max

LLaVA -1.6-13B

DeepSeek -VL-7B

DeepSeek -VL-1.3B

LLaVA -1.6-7B

0

100

200

300

400

500

Number

Insufficient Knowledge Inadequate Generalization Complete Mastery Rote Memorization

- Figure 9: The performance of different LMMs on four-dimensional metrics under loose metric.

GPT-4oLLaVA-NeXT-110BInternVL-Chat-V1-5

InternLM-XC2 MiniCPM

-V2.5

- 0

10

20

30

40

50

60

70

Score(%)

(Strict) Scoreaverage

w/o KCA

KCA

GPT-4oLLaVA-NeXT-110BInternVL-Chat-V1-5

InternLM-XC2 MiniCPM

-V2.5

0

10

20

30

40

50

60

70

Score(%)

(Loose) Scoreaverage

w/o KCA

KCA

GPT-4oLLaVA-NeXT-110BInternVL-Chat-V1-5 InternLM-XC2 MiniCPM-V2.5

0

10

20

30

40

50

60

70

80

90

Score(%)

Insufficient Knowledge (IK)

w/o KCA

KCA

GPT-4oLLaVA-NeXT-110BInternVL-Chat-V1-5 InternLM-XC2 MiniCPM-V2.5

0

10

20

30

40

50

60

70

80

90

Score(%)

Inadequate Generalization (IG)

w/o KCA

KCA

- Figure 10: Quantitative Analysis on KCA. The left two figures show the impact of KCA on the average performance of LMMs under strict and loose conditions. The right two figures compare the results between IK and IG.

the logical relationships between IK, IG, and CM (IK → IG → CM), we are pleasantly surprised to find that GPT-4o is markedly superior to the open-sourced LLaVA-NeXT-110B in IK (19.05%), suggesting it has successfully converted a considerable amount of IK into IG issue. This revelation indicates that GPT-4o’s challenges in reasoning have shifted from addressing Insufficient Knowledge in one-step problems to the knowledge generalization stage, leading us to speculate that there may have been groundbreaking changes in GPT-4o’s training strategy. However, other LMMs remain stuck at the IK phase. We argue that it is pointless to compare IG without a solid grasp of IK, highlighting the significance of our hierarchical metrics (IK < IG < CM).

The Unreasonable RM issue remains widespread across Most LMMs. GPT-4o achieves a significant lead on the RM issue, particularly on the loose metric (SRM < 2%). However, other LMMs still exhibit nearly 25% SRM on the loose metric. When focusing on the changes in SRM between strict and loose metrics, several models (LLaVA-NeXT-110B, GLM-4V-9B, DeepSeekVL-1.3B, MiniCPM-Llama3-V 2.5) show significant variations. This is a beneficial phenomenon, indicating that these models possess a certain ability to solve one-step problems, but their performance fluctuates due to external factors such as prompting templates and hyper-parameters.

##### 3.3 Quantitative Analysis

[Figure 142]

The Effectiveness on KCA. Figure 10 displays the quantitative analysis of the LMMs with knowledge concept augmented (KCA). We find that LMMs with different parameter scales show consistent performance improvements on both strict and loose metrics after introducing the KCA strategy. Moreover, the KCA strategy significantly alleviates IK issues but does not noticeably improve IG. This aligns with human intuition, as the knowledge descriptions primarily address gaps in reasoning knowledge. Nevertheless, alleviating IG issues requires a comprehensive enhancement of the LMMs’ knowledge generalization abilities, which we consider a direction for future exploration.

Understanding Angles (Using a Protractor)

Sum of Interior Angles of Other Polygons

No Errors: 38

No Errors: 13

Division of Plane Figures

No Errors: 40

No Errors: 52

Rotation

Figure 11: Error analysis of GPT-4o, The definitions of 4 types errors are listed in Appendix.

Error Anaysis. Figure 11 shows the occurrence of the four types of errors across the 67 knowledge concepts. Knowledge errors are the most frequent, appearing in over 45 knowledge concepts. Notably, although visual errors are the second most common, they are more concentrated in specific concepts (e.g., "Understanding Angles" >10), and over 38 concepts have no visual errors. This finding underscores the urgent need to enhance the fine-grained measurement capabilities of visual encoders in LMMs for mathematical reasoning, rather than blindly improving their overall capabilities.

#### 4 Related Work

Mathematical Reasoning Benchmarks. Assessing mathematical reasoning abilities is crucial for the development of large foundational models (LLMs and LMMs). Early efforts, such as MathQA [49], focus on solving mathematical word problems and highlight the importance of operation-based reasoning. Following this, datasets like GSM8K [50] and MATH [51] set the stage for evaluating textbased mathematical problems at various difficulty levels. Other benchmarks, such as MMLU [52] and MT-Bench [53], also consider mathematical evaluation as a key part of assessing LLMs. Beyond textonly evaluations, datasets like GeoQA [32], UniGeo [33], and Geometry3K [30] have pioneered the evaluation of geometric problems. Recently, several benchmarks [34] [54] have expanded their scope to cover a broader range of subjects. Additionally, MathVerse [35] aims to evaluate reasoning paths based on reference answers. However, challenges remain due to the complex nature of mathematical reasoning. In this paper, we introduce WE-MATH, a comprehensive benchmark designed to evaluate the reasoning abilities of LMMs across a wide range of mathematical categories.

Benchmarks for Large Multimodal Model. The rapid advancement of Large Language Models (LLMs) and Large Multimodal Models (LMMs) have highlighted the necessity for more comprehensive evaluation benchmarks. At first, the emergence of a series of text-only benchmarks and evaluations give us a clearer understanding of the strengths and weaknesses of large language models [50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62]. Focusing on the visual aspect, early benchmarks predominantly focused on narrow tasks like Visual Question Answering (VQA) [63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75] and image captioning [76, 77, 78], showcasing significant progress but not fully addressing the broader spectrum of multimodal perception and reasoning. This gap has driven recent research to assess LMMs from multiple angles. Notable efforts include MMBench [79] and SEED-bench [80, 81], which probe models’ abilities through commonsense queries incorporating multiple-choice formats. For domain-specific expertise, MMMU [54] utilize academic content to gauge deeper knowledge levels. Yet, benchmark such as MMStar [82] reveals that certain evaluations allow models to respond without images, risking data leakage and failing to adequately measure logic and reasoning skills. The challenge of understanding image implications, requiring multi-hop reasoning and theory of mind (ToM) [83, 84, 85, 85, 86, 87], underscores this shortfall. In parallel, the intersection of large language models (LLMs) and Large Multimodal Models (LMMs) has surged, extending the applicability of LMMs evaluations across diverse modalities including 2D images [88, 89, 90], 3D point clouds [91, 92, 93], audio [94, 95, 96, 97], and video [98, 99, 100]. Moreover, a series of works have positioned LMMs as agents with various tools, such as APIs [101, 102, 103], retrievers [104, 105] , thereby broadening the development avenues for the model evaluation community [106, 107, 108, 109].

#### 5 Conclusion

In this paper, we propose WE-MATH, a comprehensive benchmark for in-depth analysis of LMMs in visual mathematical reasoning. WE-MATH encompasses 6.5K visual math problems, covering 5 layers and 67 knowledge concepts. Moreover, we pioneeringly decompose composite problems into sub-problems according to the required knowledge concepts and introduce a novel four-dimensional metric for fine-grained reasoning evaluation. With WE-MATH, we thoroughly evaluate existing LMMs in visual mathematical reasoning and reveal a negative correlation between solving steps and problem-specific performance. Furthermore, we identify IK issues as the greatest vulnerability of LMMs. However, GPT-4o’s main challenge has shifted from IK to IG, highlighting it the first LMM towards the next stage. Lastly, analyses on KCA strategy and error cases further heuristically guides existing LMMs towards human-like visual mathematical reasoning.

#### References

- [1] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436–444, 2015.
- [2] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.
- [3] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [4] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022.
- [5] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [6] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [7] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.
- [8] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024.
- [9] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems, 36, 2024.
- [10] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.
- [11] Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023.
- [12] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010, 2023.

- [13] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023.
- [14] Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355, 2023.
- [15] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.
- [16] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [17] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023.
- [18] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. CoRR, 2022.
- [19] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: program-aided language models. In Proceedings of the 40th International Conference on Machine Learning, pages 10764–10799, 2023.
- [20] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving, 2024.
- [21] Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, and Hongsheng Li. Mathcoder: Seamless code integration in llms for enhanced mathematical reasoning. CoRR, 2023.
- [22] Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct, 2023.
- [23] Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling relationship on learning mathematical reasoning with large language models, 2023.
- [24] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models, 2024.
- [25] Chengpeng Li, Zheng Yuan, Guanting Dong, Keming Lu, Jiancan Wu, Chuanqi Tan, Xiang Wang, and Chang Zhou. Query and response augmentation cannot help out-of-domain math reasoning generalization. arXiv preprint arXiv:2310.05506, 2023.
- [26] R OpenAI. Gpt-4v (ision) system card. Citekey: gptvision, 2023.
- [27] R. Gemini Team. Gemini: A family of highly capable multimodal models, 2024.
- [28] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421, 9(1):1, 2023.
- [29] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370, 2023.

- [30] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165, 2021.
- [31] Minjoon Seo, Hannaneh Hajishirzi, Ali Farhadi, Oren Etzioni, and Clint Malcolm. Solving geometry problems: Combining text and diagram interpretation. In Proceedings of the 2015 conference on empirical methods in natural language processing, pages 1466–1476, 2015.
- [32] Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric P Xing, and Liang Lin. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. arXiv preprint arXiv:2105.14517, 2021.
- [33] Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. arXiv preprint arXiv:2212.02746, 2022.
- [34] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.
- [35] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024.
- [36] Richard Fitzpatrick. Euclid’s elements of geometry, 2008.
- [37] Wikipedia contributors. Wikipedia, 2004.
- [38] OpenAI. Hello gpt-4o, 2024.
- [39] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.
- [40] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [41] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023.
- [42] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, and Chong Ruan. Deepseek-vl: Towards real-world vision-language understanding, 2024.
- [43] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024.
- [44] Jinyi Hu, Yuan Yao, Chongyi Wang, Shan Wang, Yinxu Pan, Qianyu Chen, Tianyu Yu, Hanghao Wu, Yue Zhao, Haoye Zhang, et al. Large multilingual models pivot zero-shot multimodal learning across languages. arXiv preprint arXiv:2308.12038, 2023.
- [45] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024.
- [46] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Zhong Muyan, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023.

- [47] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and Zihan Wang. Chatglm: A family of large language models from glm-130b to glm-4 all tools, 2024.
- [48] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision, 2024.
- [49] Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. MathQA: Towards interpretable math word problem solving with operationbased formalisms. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2357–2367, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
- [50] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [51] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- [52] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021.
- [53] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.
- [54] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023.
- [55] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models, 2023.
- [56] Guanting Dong, Jinxu Zhao, Tingfeng Hui, Daichi Guo, Wenlong Wan, Boqi Feng, Yueyan Qiu, Zhuoma Gongque, Keqing He, Zechen Wang, and Weiran Xu. Revisit input perturbation problems for llms: A unified robustness evaluation framework for noisy slot filling task, 2023.
- [57] Xiaoshuai Song, Muxi Diao, Guanting Dong, Zhengyang Wang, Yujia Fu, Runqi Qiao, Zhexu Wang, Dayuan Fu, Huangxuan Wu, Bin Liang, Weihao Zeng, Yejie Wang, Zhuoma GongQue, Jianing Yu, Qiuna Tan, and Weiran Xu. Cs-bench: A comprehensive benchmark for large language models towards computer science mastery, 2024.
- [58] Xiaoshuai Song, Keqing He, Pei Wang, Guanting Dong, Yutao Mou, Jingang Wang, Yunsen Xian, Xunliang Cai, and Weiran Xu. Large language models meet open-world intent discovery and recognition: An evaluation of chatgpt, 2023.
- [59] Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen Wang, Zhicheng Dou, and Ji-Rong Wen. Understand what llm needs: Dual preference alignment for retrieval-augmented generation, 2024.

- [60] Mingfeng Xue, Dayiheng Liu, Kexin Yang, Guanting Dong, Wenqiang Lei, Zheng Yuan, Chang Zhou, and Jingren Zhou. Occuquest: Mitigating occupational bias for inclusive large language models, 2023.
- [61] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence?, 2019.
- [62] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks, 2020.
- [63] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433, 2015.
- [64] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017.
- [65] Kushal Kafle and Christopher Kanan. An analysis of visual question answering algorithms. In Proceedings of the IEEE international conference on computer vision, pages 1965–1973, 2017.
- [66] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.
- [67] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.
- [68] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.
- [69] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019.
- [70] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marçal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4291–4301, 2019.
- [71] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.
- [72] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pages 947–952. IEEE, 2019.
- [73] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2024.
- [74] Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samual White, et al. Vizwiz: nearly real-time answers to visual questions. In Proceedings of the 23nd annual ACM symposium on User interface software and technology, pages 333–342, 2010.
- [75] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022.

- [76] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.
- [77] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8948–8957, 2019.
- [78] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pages 2641–2649, 2015.
- [79] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.
- [80] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seedbench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.
- [81] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024.
- [82] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large visionlanguage models? arXiv preprint arXiv:2403.20330, 2024.
- [83] Poorav Desai, Tanmoy Chakraborty, and Md Shad Akhtar. Nice perfume. how long did you marinate in it? multimodal sarcasm explanation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 10563–10571, 2022.
- [84] Jack Hessel, Ana Marasovi´c, Jena D Hwang, Lillian Lee, Jeff Da, Rowan Zellers, Robert Mankoff, and Yejin Choi. Do androids laugh at electric sheep? humor" understanding" benchmarks from the new yorker caption contest. arXiv preprint arXiv:2209.06293, 2022.
- [85] Winnie Street, John Oliver Siy, Geoff Keeling, Adrien Baranes, Benjamin Barnett, Michael McKibben, Tatenda Kanyere, Alison Lentz, Robin IM Dunbar, et al. Llms achieve adult human performance on higher-order theory of mind tasks. arXiv preprint arXiv:2405.18870, 2024.
- [86] Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, and Jingren Zhou. Self-play with execution feedback: Improving instruction-following capabilities of large language models, 2024.
- [87] Ziqiang Liu, Feiteng Fang, Xi Feng, Xinrun Du, Chenhao Zhang, Zekun Wang, Yuelin Bai, Qixuan Zhao, Liyang Fan, Chengguang Gan, et al. Ii-bench: An image implication understanding benchmark for multimodal large language models. arXiv preprint arXiv:2406.05862, 2024.
- [88] Chenshuang Zhang, Fei Pan, Junmo Kim, In So Kweon, and Chengzhi Mao. Imagenet-d: Benchmarking neural network robustness on diffusion synthetic object, 2024.
- [89] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. 2009 IEEE conference on computer vision and pattern recognition, pages 248–255, 2009.
- [90] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88(2):303–338, 2010.

- [91] Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 3d shapenets: A deep representation for volumetric shapes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1912–1920, 2015.
- [92] Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu. Shapenet: An information-rich 3d model repository, 2015.
- [93] Andreas Geiger, Philipp Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. International Journal of Robotics Research (IJRR), 32(11):1231–1237, 2013.
- [94] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5206–5210. IEEE, 2015.
- [95] Christophe Veaux, Junichi Yamagishi, and Kirsten MacDonald. The voice bank corpus: Design, collection and data analysis of a large regional accent speech database. In 12th Annual Conference of the International Speech Communication Association (Interspeech), pages 121–125, 2017.
- [96] Qian Yang, Jin Xu, Wenrui Liu, Yunfei Chu, Ziyue Jiang, Xiaohuan Zhou, Yichong Leng, Yuanjun Lv, Zhou Zhao, Chang Zhou, and Jingren Zhou. Air-bench: Benchmarking large audio-language models via generative comprehension, 2024.
- [97] Bin Wang, Xunlong Zou, Geyu Lin, Shuo Sun, Zhuohan Liu, Wenyu Zhang, Zhengyuan Liu, AiTi Aw, and Nancy F. Chen. Audiobench: A universal benchmark for audio large language models, 2024.
- [98] Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating video-based large language models, 2023.
- [99] Sami Abu-El-Haija, Nisarg Kothari, Joonseok Lee, Paul Natsev, George Toderici, Balakrishnan Varadarajan, and Sudheendra Vijayanarasimhan. Youtube-8m: A large-scale video classification benchmark. In arXiv preprint arXiv:1609.08675, 2016.
- [100] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models, 2023.
- [101] Junlin Xie, Zhihong Chen, Ruifei Zhang, Xiang Wan, and Guanbin Li. Large multimodal agents: A survey. arXiv preprint arXiv:2402.15116, 2024.
- [102] Chenyu Wang, Weixin Luo, Qianyu Chen, Haonan Mai, Jindi Guo, Sixun Dong, Xiaohua, Xuan, Zhengxin Li, Lin Ma, and Shenghua Gao. Mllm-tool: A multimodal large language model for tool agent learning, 2024.
- [103] Xiao Liu, Jianfeng Lin, and Jiawei Zhang. Beyond text: Unveiling multimodal proficiency of large language models with multiapi benchmark, 2023.
- [104] Xinwei Long, Jiali Zeng, Fandong Meng, Zhiyuan Ma, Kaiyan Zhang, Bowen Zhou, and Jie Zhou. Generative multi-modal knowledge retrieval with large language models, 2024.
- [105] Ruochen Zhao, Hailin Chen, Weishi Wang, Fangkai Jiao, Xuan Long Do, Chengwei Qin, Bosheng Ding, Xiaobao Guo, Minzhi Li, Xingxuan Li, and Shafiq Joty. Retrieving multimodal information for augmented generation: A survey. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 4736–4756, Singapore, December 2023. Association for Computational Linguistics.
- [106] Paul Pu Liang, Yiwei Lyu, Xiang Fan, Zetian Wu, Yun Cheng, Jason Wu, Leslie Chen, Peter Wu, Michelle A. Lee, Yuke Zhu, Ruslan Salakhutdinov, and Louis-Philippe Morency. Multibench: Multiscale benchmarks for multimodal representation learning, 2021.

- [107] Wentao Ge, Shunian Chen, Guiming Hardy Chen, Zhihong Chen, Junying Chen, Shuo Yan, Chenghao Zhu, Ziyue Lin, Wenya Xie, Xinyi Zhang, Yichen Chai, Xiaoyu Liu, Dingjie Song, Xidong Wang, Anningzhe Gao, Zhiyi Zhang, Jianquan Li, Xiang Wan, and Benyou Wang. Mllm-bench: Evaluating multimodal llms with per-sample criteria, 2024.
- [108] Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, Jiayi Lei, Quanfeng Lu, Runjian Chen, Peng Xu, Renrui Zhang, Haozhe Zhang, Peng Gao, Yali Wang, Yu Qiao, Ping Luo, Kaipeng Zhang, and Wenqi Shao. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi, 2024.
- [109] Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. Uniir: Training and benchmarking universal multimodal information retrievers, 2023.

#### Contents

### Appendix

- 1 Introduction 1
- 2 WE-MATH 3

- 2.1 Hierachial Structured Dataset Composition . . . . . . . . . . . . . . . . . . . . . 3
- 2.2 Knowledge based Reasoning Evaluation . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.3 Knowledge Concept Augmentation . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 3 Experiment 7

- 3.1 Main Result . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.2 Knowledge based Reasoning Analysis . . . . . . . . . . . . . . . . . . . . . . . . 9
- 3.3 Quantitative Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- 4 Related Work 11
- 5 Conclusion 12

- A Broaden Impact 21
- B More Details on WE-MATH 21

- B.1 Hierarchical Knowledge Structure . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- B.2 Knowledge-based Data Decomposition . . . . . . . . . . . . . . . . . . . . . . . . 21
- B.3 Knowledge Concepts Augmentation . . . . . . . . . . . . . . . . . . . . . . . . . 26
- B.4 Details of Data Collection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- B.5 Details of Data Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- C More Details on the Metrics 27
- D More Details on Experiment Setup 29

- D.1 Details of the Evaluated Models . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- D.2 Details of the Model Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . 29

- E More Details on Experiment Results 31

- E.1 Details of Model Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- E.2 Specific Error Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36

- F Example Demonstration 39

- F.1 Description of the Knowledge Concepts . . . . . . . . . . . . . . . . . . . . . . . 39
- F.2 Data Sources of WE-MATH . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45

#### A Broaden Impact

Bridging Human-Like Inspiration and Reliability. As previously mentioned, works such as neural networks [2] and attention mechanisms [3] draw their design inspiration from human thinking patterns. This is fundamentally because the purpose of designing AI is to assist humans. Currently, LMMs have already been helping people in various scenarios, which was unimaginable in the past. Therefore, we firmly believe that a new era is coming, where people will focus not only on the performance of models in specific fields but also on the reliability of a model. In some fundamental scenarios, a reliable model is more important, which is one of the primary motivations behind the creation of WE-MATH. Furthermore, after completing our experiments, we find that in a loose setting, GPT-4o’s RM metric is only 1.07%, showing us the possibility of a reliable and accurate model emerging in the future.

Fine-grained Evaluation and Versatile Applications. From the model’s perspective, WE-MATH can provide LMMs with an assessment of mathematical abilities. Additionally, WE-MATH’s IK, IG, and CM metrics offer a fine-grained evaluation of the model’s capabilities. Furthermore, the RM metric reflects a model’s reliability to address our concern of not desiring a model that can solve complex problems but makes errors on sub-problems within the solution process. Ultimately, we introduce the Scoreaverage metric to quantify the model’s overall performance. Moreover, since WE-MATH is constructed from the decomposition of a multi-step problem’s necessary solution process, it provides new perspectives for interactive tasks (multi-turn dialogues), self-supervised learning, information extraction, and other tasks. It also offers crucial references and support for the deployment of models in education and other fields.

Ethics Statement. We ensure that WE-MATH complies with legal and ethical guidelines throughout its construction process, with no violations. We provide fair compensation to all annotators involved. WE-MATH focuses on elementary mathematics problems, and during its construction, data collection was sourced from publicly available test questions, textbooks, and professional websites. Since mathematics problems inherently have standard answers, they are not subject to cultural differences. Additionally, we guarantee that WE-MATH is solely for academic research purposes, and we uphold the strict prohibition of any commercial use. Additionally, we declare that we will bear full responsibility in the event of any rights violations and confirm the data license.

#### B More Details on WE-MATH

##### B.1 Hierarchical Knowledge Structure

Figure 12, 13 shows the detailed hierarchical structure of WE-MATH, which includes 5 levels, 99 nodes, and 67 leaf nodes.

In the initial stages of constructing the benchmark, we aimed to address two key objectives. We believe that the purpose of designing a benchmark is to evaluate the performance of models and provide guidance on areas that need improvement. However, existing benchmarks offer only broad guides in these aspects. Additionally, the core contribution mentioned earlier is that WE-MATH is the first benchmark specifically designed to study the mathematical problem-solving mechanisms of models. Inspired by the learning paradigm of humans, which is based on knowledge concepts, WE-MATH constructs its dataset with knowledge concepts as the basic unit, resulting in evaluations with rigorous scientific accuracy and better guidance.

##### B.2 Knowledge-based Data Decomposition

Figures 14, 15 illustrate the process of Knowledge-based Data Decomposition. Collection. In each example, the Collection section presents specific information about each multistep problem in the dataset.

Human reasoning. The Human reasoning section shows the process required before decomposing each multi-step problem, where educational experts extract the key information needed for each sub-problem based on the reasoning path for the knowledge concepts included in the multi-step problem.

Decompose. The Decompose section uses the key information extracted in the Human reasoning section to formulate sub-problems, refine the options, and ultimately achieve the decomposition of a multi-step problem.

It is necessary to further explain that to ensure each sub-problem has a rigorous logical relationship and is independent, the text condition for the first sub-problem is derived from the text condition of the multi-step problem, and the image condition for the first sub-problem is the same as the image condition of the multi-step problem.

Furthermore, in constructing the second sub-problem, two situations may arise. The first situation is where the answer of the first sub-problem is injected as a key condition into the image condition of the second sub-problem, presenting the information visually. The second situation is where the answer of the first sub-problem is injected as a key condition into the text condition of the second sub-problem, while the image condition remains unchanged.

In WE-MATH, the vast majority of cases are of the first type. However, for some information that is extremely difficult to present in images, we opt for the second type, presenting the information in text form. To ensure fairness in the decomposition of the problems, only one of these situations will occur in the decomposition of the same multi-step problem. This approach ensures that the question of the final sub-problem will match the original multi-step problem, completing the decomposition.

[Figure 143]

###### Figure 12: The Hierarchical Knowledge Structure of WE-MATH (1).

[Figure 144]

###### Figure 13: The Hierarchical Knowledge Structure of WE-MATH (2).

[Figure 145]

Two-Step problem

Collection

Condition： As shown in the diagram, quadrilateral ABCD is a parallelogram. A circle with center A has a circumference of 36 cm, and the arc length EF is 6 cm. Question: What is the measure of ∠C? ( )° Option: A. 30°;B. 60°;C. 90°;D. 45°; E. No correct answer Answer: B. 60°

Knowledge concepts: · Understanding Sectors, · Properties and Understanding of Parallelograms

D C

F

A E B

[Figure 146]

Human reasoning

Extracting Key Information Based on Knowledge Concepts

Step1: Understanding Sectors Step2: Properties and Understanding of Parallelograms

I know ∠A = 60°, In parallelogram ABCD, I can determine the measure of ∠C.

I know the circumference of a circle is 36cm, I can calculate the measure of ∠A corresponding to an arc length of 6cm.

Answer: ∠A = 60°

Final Answer: ∠C = 60°

[Figure 147]

Decompose

Step-by-Step Based Knowledge Concept

c1：As shown in the diagram, quadrilateral ABCD is a parallelogram. If the circumference of a circle is 36cm, and the arc length EF is 6 cm,

c2：As shown in the diagram, quadrilateral ABCD is a parallelogram. If the circumference of a circle is 36cm, and the arc length EF is 6 cm. In parallelogram ABCD,

q1: the measure of ∠A corresponding to an arc length EF equals ( )° Option: A. 30°;B. 60°;C. 90°;D. 45°; E. No correct answer

q2: what is the measure of ∠C? Option: A. 30°; B. 60°; C. 90°; D. 45°; E. No correct answer

D C

D C F F

60°

A E B

A E B

a1 : B. 60° (∠A = 60°) a2 : B. 60° (∠C = 60°)

Figure 14: An example of a two-step problem in WE-MATH.

[Figure 148]

Three-Step problem

Collection

Condition： As shown in the figure, quadrilateral ABCD is a rectangle with an area of 20 cm². Taking B as the center and AB as the radius, the sector intersects the length BC of the rectangle at point E. Square FBHG is drawn inside the sector.

Knowledge concepts: · Area of Rectangles, · Understanding Sectors, · Area of Squares

- A D
- B H E C

F G

Question: What is the area of square FBHG in cm²? Option: A. 16;B. 8;C. 6;D. 4; E. No correct answer Answer: B. 8

###### 5cm

[Figure 149]

[Figure 150]

Human reasoning

Step-by-step based knowledge concept

Step1: Area of Rectangles Step2: Understanding Sectors Step3: Area of Squares

I know in rectangle ABCD, a sector with center B and radius BA intersects BC at point E, Both AB and BG are radii of the circle. AB = 4cm;

I know BG = 4 cm, BG is the diagonal of the square; I can calculate the area of square FBHG .

I know the quadrilateral ABCD is a rectangle with an area of 20 cm², The

length of BC is 5 cm.; I can calculate the length of AB.

I can determine the length of BG.

Answer: AB = 4 cm Answer: BG = 4 cm Final Answer: 8 cm²

[Figure 151]

Decompose

Step-by-step based knowledge concept

c1： As shown in the figure, quadrilateral ABCD is a rectangle with an area of 20 cm².

c2： As shown in the figure, in rectangle ABCD, a sector with center B and radius BA intersects BC at point E. The square FBHG is inside the sector.

c3： As shown in the figure, given the length of the diagonal BG of square FBHG, BG = 4cm.

q1: What is the length of AB in cm? Option: A. 4; B. 3; C. 2; D. 5; E. No correct answer

q3: What is the area of square FBHG in cm²? Option: A. 16; B. 8; C. 6; D. 4; E. No correct answer

q2: What is the length of BG in cm? Option: A. 4;B. 3;C. 2;D. 5; E. No correct answer

- A D
- B H E C

- A D
- B H E C

- A D
- B H E C

F G

F G 4cm

F G

4cm

5cm

a1 : A. 4 a2 : A. 4 a3 : B. 8

Figure 15: An example of a three-step problem in WE-MATH.

Table 3: Prompt templates for response generations.

Type Prompt Template

Now, we require you to solve a multiple-choice math question. Please briefly describe your thought process and provide the final answer(option). Question: <Question> Option: <Option> Regarding the format, please answer following the template below, and be sure to include two <> symbols: <Thought process>: <<your thought process>> <Answer>: <<your option>>

Multiple Choice

Now, we require you to solve a multiple-choice math question. We will provide you with the relevant knowledge concepts of this question for your reference. Please briefly describe your thought process and provide the final answer(option). Knowledge concept: <Knowledge concept> Question: <Question> Option: <Option> Regarding the format, please answer following the template below, and be sure to include two <> symbols: <Thought process>: <<your thought process>> <Answer>: <<your option>>

Knowledge Concept Augmentation

##### B.3 Knowledge Concepts Augmentation

Table 3 report the prompt templates in our experiments. We concatenate the textual descriptions into the prompt. Additionally, each knowledge concept description is accompanied by its corresponding visual content, which helps the experimenter understand and facilitates further enhancement when models can incorporate sufficient visual information as part of the prompt in the future.

In section F.1, we illustrates the specific content of descriptions for 67 knowledge concepts. For example, as shown in Figure 46, for the knowledge concept "Perimeter of Squares," it is necessary to know that "c=4a", relying solely on textual descriptions is insufficient for understanding this concept, so we include visual information to aid comprehension.

##### B.4 Details of Data Collection

With the hierarchical knowledge structure, we select geometric problems with images from publicly authoritative mathematics websites from various countries, including professional exams and practice tests (detailed sources list can be found in section F.2). To ensure comprehensive coverage of fundamental and critical areas in primary math, we select the five most foundational and prevalent domains within the field of primary geometry, including:

- • Plane figures: Questions involving identification and properties of two-dimensional shapes.
- • Solid figures: Questions related to the recognition and characteristics of three-dimensional objects.
- • Transformation and motion of figures: Problems focusing on geometric transformations such as translation, rotation, and reflection.
- • Position and direction: Questions that involve understanding spatial relationships and directions.
- • Measurement: Problems requiring the measurement of length, area, volume, and angles.

The selection criteria are as follows: (1) The problems include multiple knowledge concepts and can be decomposed into steps for solution. (2) The problems and images are consistent. (3) The correct answer is unique, and the distractor options are highly confusing.

##### B.5 Details of Data Statistics

[Figure 152]

Figure 16: The distribution of the number of words per question in WE-MATH. Questions with a length greater than 80 are categorized as 81 for visualization simplicity.

Question distribution. The WE-MATH consists entirely of English questions, all newly collected from public authoritative mathematics websites, and presented in the format of multiple-choice questions. As illustrated in Table 4, the average number of words in the English questions within WE-MATH is 25.81, with the maximum number of words in a question reaching 143. Figure 16 further elaborates on the distribution of word counts, highlighting the diverse patterns of the questions.

Table 4: Key statistics of WE-MATH.

##### Statistic Number

Total questions 6,524 Newly collected questions 6,524 Multiple-choice questions 6,524

- -First-layer nodes 5
- -Second-layer nodes 12
- -Terminal nodes 67 Question options

- -Total options 25,178
- -Average options 3.859
- -Proportion of answer A 6,524 (25.9%)
- -Proportion of answer B 6,524 (25.9%)
- -Proportion of answer C 6,505 (25.8%)
- -Proportion of answer D 4,419 (17.6%)
- -Proportion of answer E 1,198 (4.8%)
- -Proportion of answer F&G 11 (0.04%) Question length

- -Maximum length (word) 143
- -Maximum length (character) 852
- -Average length (word) 25.8
- -Average length (character) 135.3

Advantages of Multiple-Choice Questions. In WE-MATH, all problems are presented as multiple-choice questions. Even if some problems did not originally conform to the multiple-choice format during the initial selection, our researchers manually converted

- them into the format. Using multiple-choice questions offers several advantages:

Standardization: Ensures uniformity across all questions, facilitating consistent assessment and comparison across different hierarchical subjects.

Objective Grading: The use of single correct answers eliminates subjectivity in grading, enhancing the reliability of the evaluation.

Efficiency: Allows for rapid and scalable assessment, suitable for large datasets and automated systems.

Focused Assessment: Carefully designed distractors help in accurately identifying specific knowledge gaps and common misconceptions.

#### C More Details on the Metrics

Distinguishing Metric. Considering the model’s instability, Figure 4 and Figure 17, 18 illustrate the two metrics we propose for distinguishing between RM and CM metrics. Figure 4 represents the

Evaluation

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

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Metric for Reasoning Evaluation

[Figure 167]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Insufficient Knowledge

Inadequate Generalization

Complete Mastery

Rote Memorization

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Insufficient Knowledge

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

- Step1

Two-Step problem

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

Strict Metric

- Step2

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

Step2

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

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

Figure 17: Diagram illustrating strict metric in three-step problem.

Evaluation

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Metric for Reasoning Evaluation

[Figure 294]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Insufficient Knowledge

Inadequate Generalization

Complete Mastery

Rote Memorization

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

Insufficient Knowledge

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

- Step1

Two-Step problem

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

Loose Metric

- Step2

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

Step2

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

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

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

Figure 18: Diagram illustrating loose metric in three-step problem.

two-step problem, while Figures 17 and Figures 18 represent the three-step problem. Specifically, under the strict metric, if there is any error in the corresponding sub-problems of a multi-step problem that is answered correctly, it is classified as RM (Rote Memorization). Only if all corresponding sub-problems are answered correctly (TTTT, TTT) is it classified as CM (Complete Master). Under the loose metric, it is classified as RM only if the model answers all sub-problems incorrectly (FFFT, FFT), otherwise, it is classified as CM. Therefore, the Scoreaverage under the loose metric is slightly higher. We hope to see models like GPT-4o [38] and GPT-4V [26], which have already performed nearly perfectly under the loose metric and are far ahead of other models, bring us even greater surprises under the strict metric in the next update.

Metrics’ Intrinsic Logic. As shown in Figure 4, 17, 18, it is evident in the Metric for Reasoning Evaluation Section that IK, IG, and CM have a logical relationship. In the early stages of constructing WE-MATH, we recorded all the model’s responses and analyzed the answers to each multi-step problem and its corresponding sub-problems. We believe that for both humans and models, a reasonable learning process should involve first mastering each knowledge concept individually and

- then learning to comprehensively apply them to achieve complete mastery. The situation where the multi-step problem is answered correctly but the sub-problems are answered incorrectly (RM) is an unreasonable phenomenon. Therefore, we developed a four-dimensional fine-grained metric to further evaluate the model’s performance.

Table 5: The release time and model source of LMMs used in WE-MATH

Model Release Time Source

GPT-4o [38] 2024-05 https://gpt4o.ai/ GPT-4V [26] 2024-04 https://openai.com/index/gpt-4v-system-card/ Gemini 1.5 Pro [40] 2024-05 https://deepmind.google/technologies/gemini/pro/ Qwen-VL-Max [13] 2024-01 https://huggingface.co/spaces/Qwen/Qwen-VL-Max/ LLaVA-NeXT-110B [39] 2024-05 https://huggingface.co/lmms-lab/llava-next-110b/ LLaVA-NeXT-72B [39] 2024-05 https://huggingface.co/lmms-lab/llava-next-72b/ LLaVA-1.6-13B [41] 2024-03 https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf/ LLaVA-1.6-7B [41] 2024-03 https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf/ DeepSeek-VL-1.3B [42] 2024-03 https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat/ DeepSeek-VL-7B [42] 2024-03 https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat/ Phi3-Vision-4.2B [43] 2024-05 https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/ MiniCPM-LLaMA3-V 2.5 [44] 2024-05 https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5/ InternLM-XComposer2-VL-7B [45] 2024-04 https://huggingface.co/internlm/internlm-xcomposer2-vl-7b/ InternVL-Chat-V1.5 [46] 2024-04 https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5/ GLM-4V-9B [47] 2024-06 https://huggingface.co/THUDM/glm-4v-9b LongVA [48] 2024-06 https://huggingface.co/lmms-lab/LongVA-7B G-LLaVA-13B [29] 2024-03 https://huggingface.co/renjiepi/G-LLaVA-13B/

Table 6: Generating parameters for Closed-Source LMMs.

Model Generation Setup GPT-4o "model" : "gpt-4o", "temperature" : 0, "max_tokens" : 1024 GPT-4V "model" : "gpt-4-turbo", "temperature" : 0, "max_tokens" : 1024

Gemini 1.5 Pro "model" : "gemini-1.5-pro-latest", "temperature" : 0, "max_tokens" : 1024 Qwen-VL-Max "model" : "qwen-vl-max", "temperature" : 0, "max_tokens" : 1024

#### D More Details on Experiment Setup

##### D.1 Details of the Evaluated Models

To evaluate the mathematical reasoning abilities of various LMMs, we selected their latest model versions. Table 5 presents their release dates and specific sources. Given the intuition that smaller models (with parameters of 7B or less) perform poorly on Insufficient Knowledge (IK), we also included evaluations of the latest models with 7B, 4.2B, and 1.3B parameters. This was done to explore whether these models could achieve significant improvement under the KCA strategy.

##### D.2 Details of the Model Hyperparameters

For all closed-sourced models with API access, we adopt the generation scheme shown in Table 6 and simply run the inference with CPUs, which typically completes within a day. For all open-source models, we utilize a cluster with 8 NVIDIA A800-SXM4-80GB GPUs to run the inference, and we follow the hyper-parameter settings specified in the model source’s inference samples. If no specific instructions are provided, we use the default settings. Table 7 details the specific generation parameters.

Table 7: Generating parameters for Open-Source LMMs.

Model Generation Setup

LLaVA-NeXT-110B do_sample = False, temperature = 0, max_new_tokens = 1024 LLaVA-NeXT-72B do_sample = False, temperature = 0, max_new_tokens = 1024 LLaVA-1.6-13B do_sample = False, temperature = 0, max_new_tokens = 1024 LLaVA-1.6-7B do_sample = False, temperature = 0, max_new_tokens = 1024

DeepSeek-VL-1.3B do_sample = False, max_new_tokens = 1024 DeepSeek-VL-7B do_sample = False, max_new_tokens = 1024 Phi3-Vision-4.2B do_sample = False, temperature = 0, max_new_tokens = 1024

MiniCPM-LLaMA3-V 2.5 sampling = True, temperature = 0.7 InternLM-XComposer2-VL-7B do_sample = False

InternVL-Chat-V1.5 num_beams = 1, do_sample = False, max_new_tokens = 1024 GLM-4V-9B do_sample = True, max_length = 1024, top_k = 1

LongVA do_sample = False, temperature = 0, max_new_tokens = 1024, num_beams = 1 G-LLaVA-13B do_sample = True, temperature = 0.2, max_new_tokens = 1024

Table 8: Model architecture of 17 LMMs evaluated on We-Math.

Models LLM Vision Encoder

GPT-4o - GPT-4V - Gemini 1.5 Pro - Qwen-VL-Max - LLaVA-NeXT-110B Qwen1.5-110B-Chat CLIP-ViT-L-P14-336 LLaVA-NeXT-72B Qwen1.5-72B-Chat CLIP-ViT-L-P14-336 LLaVA-1.6-13B Vicuna-13B-v1.5 CLIP-ViT-L-P14-336 LLaVA-1.6-7B Vicuna-7B-v1-5 CLIP-ViT-L-P14-336 DeepSeek-VL-1.3B DeepSeek-LLM-1.3B-base SigLIp-L-P16-384 DeepSeek-VL-7B DeepSeek-LLM-7B-base SigLIp-L-P16-384 & SAM-B Phi3-Vision-4.2B Phi-3-mini-128K-instruct CLIP-ViT-L-P14-336 MiniCPM-LLaMA3-V 2.5 Llama3-8B-Instruct SigLIp-L-P14-384 InternLM-XComposer2-VL-7B InternLM2-7B-ChatSFT CLIP-ViT-L-P14-336 InternVL-Chat-V1.5 InternLM2-Chat-20B InternViT-6B-448px-V1-5 (6B) GLM-4V-9B GLM-9B EVA_02_CLIP-E-P14 (4.7B) LongVA Qwen2-7B-Instruct CLIP-ViT-L-P14-336 G-LLaVA-13B Vicuna-13B-v1.5 CLIP-ViT-L-P14-336

#### E More Details on Experiment Results

##### E.1 Details of Model Performance

The Leaderboard on WE-MATH. We present the visualization results of the Scoreaverage under the loose and strict metric in Figure 7, respectively. GPT-4o shows a significant lead under both metric, and LLaVA-NeXT-110B performs the best among open-source models. Impressively, InternVL-ChatV1.5 and GLM-4V-9B achieved excellent scores, surpassing the closed-source model Qwen-VLMax. Additionally, some recently proposed smaller models (such as Phi-3-Vision-4.2B, InternLMXComposer2-VL-7B, and MiniCPM-LLaMA3-V 2.5) also demonstrated outstanding performance, suggesting that optimizing training methods might partially substitute for the performance gains typically achieved by increasing the parameter count.

Detailed Performance of Four-Dimensional Metrics. Figure 8 and Figure 9 display the specific performance of LMMs under both loose and strict metric across four metrics. Focusing on the IK metric, GPT-4o has the fewest instances under both metric, indicating that GPT-4o has the best grasp of the knowledge concepts. Furthermore, for the IG metric, we find that GPT-4o and GPT-4V have the highest exposure compared to other models. As discussed in the previous Section C, IG issues only arise after addressing IK issues, which further indicates that GPT-4 is progressing to the next stage. Focusing on the CM and RM metrics, GPT-4o and GPT-4V continue to show significant leadership. Both models excel in the CM metric, where the number of correctly answered multi-step problems and their corresponding sub-questions is significantly higher than that of other models. Additionally, comparing GPT-4o and GPT-4V under strict metric, GPT-4o consistently outperforms GPT-4V, aligning with the Scoreaverage results.

Detailed Performance on Each Category. In Figure 6, we present the performance of open-source and closed-source models under the second-level nodes. In Figure 19 to Figure 35, we detail the specific performance of 17 models across 67 knowledge concepts (based on statistics from one-step problem questions). It is evident that GPT-4o consistently leads in overall performance, but its main issue lies in measurement-related tasks. Notably, some open-source models perform worse on the simpler "Understanding and Conversion of Units" knowledge concepts compared to "Angles and Length" related concepts, while InternVL-Chat-V1.5 and MiniCPM-LLaMA3-V 2.5 exhibit more logically consistent results.

[Figure 407]

- Figure 19: Detailed performance of GPT-4o across 67 knowledge concepts.

- Figure 20: Detailed performance of GPT-4V across 67 knowledge concepts.
- Figure 21: Detailed performance of Gemini 1.5 Pro across 67 knowledge concepts.

[Figure 409]

- Figure 22: Detailed performance of Qwen-VL-Max across 67 knowledge concepts.

[Figure 410]

- Figure 23: Detailed performance of LLaVA-NeXT-110B across 67 knowledge concepts.

[Figure 411]

###### Figure 24: Detailed performance of LLaVA-NeXT-72B across 67 knowledge concepts.

[Figure 413]

###### Figure 25: Detailed performance of InternVL-Chat-V 1.5 across 67 knowledge concepts.

[Figure 414]

###### Figure 26: Detailed performance of LLaVA-1.6-13B across 67 knowledge concepts.

[Figure 415]

###### Figure 27: Detailed performance of G-LLaVA-13B across 67 knowledge concepts.

- Figure 28: Detailed performance of GLM-4V-9B across 67 knowledge concepts.

[Figure 417]

Figure 29: Detailed performance of MiniCPM-LLama3-V 2.5 across 67 knowledge concepts.

[Figure 418]

- Figure 30: Detailed performance of LongVA-7B across 67 knowledge concepts.

[Figure 419]

- Figure 31: Detailed performance of LLaVA-1.6-7B across 67 knowledge concepts.

- Figure 32: Detailed performance of DeepSeek-VL-7B across 67 knowledge concepts.

[Figure 421]

Figure 33: Detailed performance of InternLM-XComposer2-VL-7B across 67 knowledge concepts.

[Figure 422]

- Figure 34: Detailed performance of Phi3-Vision-4.2B across 67 knowledge concepts.

[Figure 423]

- Figure 35: Detailed performance of DeepSeek-VL-1.3B across 67 knowledge concepts.

- E.2 Specific Error Analysis Table 9: Detailed Descriptions of Error Types.

##### Error Type Explanation

For a specific knowledge concept, the model is unclear or confused about it, or it misuses another knowledge concept to solve the problem.

Knowledge Error

Errors that occur in the logical reasoning process while using knowledge concepts to solve the problem step by step.

Reason Error

Errors in visual perception, where the model incorrectly identifies shapes or numbers in an image.

Visual Error

The thought process introduces factors that are not consistent with the facts, which are not mentioned in the context of the image or question.

Hallucination

Error Types. To delve into the failure cases of models, we detailed four typical error types in Table 9. Furthermore, to facilitate a better understanding of each error type, we provide examples of each error made by GPT-4o from Figure 36 to Figure 39. Since a single thought process in a problem can involve multiple errors and a single logical error is enough to derail a much larger solution, we consider the first error that occurs in the reasoning steps as the key error and include only this error in our statistics.

Correspondence of Errors in Multi-Step and One-Step Problems. Focusing on Insufficient Knowledge, the errors in multi-step problems often correspond to those in one-step problems. This supports our approach of decomposing problems to accurately associate error types with specific knowledge concepts. Furthermore, we observe a positive correlation between the quantity of knowledge concepts and the errors in the reasoning process. As the complexity of knowledge concepts increases, the difficulty for the model to perform multi-step reasoning also increases, leading

- to a higher likelihood of visual recognition errors and incorrect application of knowledge concepts.

Visual Error

Visual Error

Question:

Question:

[Figure 424]

[Figure 425]

As shown in the diagram, using a protractor to measure the angle, what is the size of∠1? ( )

As shown in the diagram, how do you transform figure A into figure B?

[Figure 426]

A

B

O

1

6cm

A. Translate 10 units to the right; B. Translate 9 units to the right; C. Translate 8 units to the right; D. Translate 7 units to the right; E. No correct answer

A. 30°;B. 60°;C. 55°;D. 90°; E. No correct answer

Answer : A Knowledge Concept : Understanding Angles (Using a Protractor) Response for GPT-4o:

###### Answer : A Knowledge Concept : Translation Response for GPT-4o:

<Thought process>: To determine the size of∠1, I will use the protractor shown in the diagram. The protractor is aligned with the baseline at 0°. The

<Thought process>: To determine how to transform figure A into figure B, I need to count the number of units figure A needs to be translated to the right to align with figure B. By counting the grid squares horizontally from the

angle∠1 is measured from the baseline to the inclined line. By observing the protractor, the inclined line intersects the protractor at the 60° mark. Therefore, the size of∠1 is 60°. <Answer>: B. 60°

leftmost point of figure A to the leftmost point of figure B, I can determine the correct translation distance.

<Answer>: C. Translate 8 units to the right.

Figure 36: Specific examples of Visual Error.

Reason Error

Reason Error

Question:

Question:

As shown in the diagram, there is an equilateral triangle within a rectangle. If∠2 = 60°, then the degree measure of∠1 is ( ).

As shown in the diagram, there is a circular cake with a radius of 5 cm, which is divided into four parts. Compared to the original circumference of the cake, the additional edge length is ( )

|2<br><br>13°<br><br>1|
|---|

A C

D

B

A. 13°;B. 12°;C. 17°;D. 30°; E. No correct answer

###### Answer : C Knowledge Concept : Calculation and Comparison of Angles Response for GPT-4o:

A. 4AB+4CD; B. AB+CD; C. 2AB+2CD; D. No correct answer

###### Answer : C Knowledge Concept : Translation Response for GPT-4o:

<Thought process>: Given that the triangle is equilateral, all its internal angles are 60°. Since∠2 is given as 60°, it confirms the triangle's equilateral nature. The angle adjacent to∠2 within the rectangle is 13°. The sum of angles around a point is 360°. Therefore, the sum of the angles∠1,∠2, and the 13°angle should be 180° (since they form a straight line). Thus,∠1 + 60°+ 13°= 180°. Solving for∠1, we get∠1 = 180°- 60°- 13°= 107°. However, this does not match any of the given options. <Answer>: E. No correct answer

<Thought process>: The original circumference of the cake is given by the formula \(2\pi r\), where \(r\) is the radius. For a radius of 5 cm, the original circumference is \(2\pi \times 5 = 10\pi\) cm. When the cake is divided into four parts, each part will have two additional edges: one along the radius (AB or CD) and one along the arc. The additional edge length for each part is AB + CD. Since there are four parts, the total additional edge length is 4(AB + CD). <Answer>: A. 4AB+4CD

Figure 37: Specific examples of Reason Error.

Knowledge Error

Knowledge Error

Question:

Question:

The diagram shows a cylindrical roller. The area covered by the roller after rolling 10 circles is equivalent to ( ) dm².(π = 3.14)

As shown in the diagram, two triangular boards are arranged to form a shape.∠1 +∠2 = ( )

A

[Figure 427]

|314cm2|
|---|

10

- 1
- 2

C

B

A. 3.14;B. 62.8;C. 314;D. 31.4; E. No correct answer

- Answer : A Knowledge Concept : Conversion Rates and Calculations Between Area Units Response for GPT-4o: <Thought process>: To determine the area covered by the cylindrical roller after rolling 10 circles, we need to calculate the area covered in one complete revolution and then multiply it by 10. The area covered in one revolution is the circumference of the roller times its width. Given that the total area covered after 10 circles is 314 cm², we can divide this by 10 to find the area covered in one circle, which is 31.4 cm². Therefore, the correct answer is 31.4 dm².

D

A. 120°; B. 160°; C. 145°; D. 90°; E. No correct answer

###### Answer : D Knowledge Concept : perpendicularity Response for GPT-4o:

<Thought process>: The two triangular boards are arranged such that they share a common side, CB. The angles∠1 and ∠2 are adjacent to each other and form a straight line along CB. Since a straight line measures 180°, the sum of∠1 and∠2 must be 180°. <Answer>: E. No correct answer

- <Answer>: D. 31.4

Figure 38: Specific examples of Knowledge Error.

Hallucination

Hallucination

Question:

Question:

As shown in the diagram, HG is the axis of symmetry for trapezoid ABCD. Then, AC equals ( ) cm, GC equals ( ) cm, and AH equals ( ) cm.

The white part in the diagram is composed of 4 identical sectors, which can be assembled into a circle. The area of the white part is ( ).

2cm

H

A B

[Figure 428]

5cm

C D

G

5cm

A. 5, 4, 2;B. 5, 5, 2;C. 5, 4, 3;D. 5, 5, 3;E. No correct answer

- Answer : B Knowledge Concept : Symmetry Response for GPT-4o: <Thought process>: Given that HG is the axis of symmetry for trapezoid ABCD, it means that the trapezoid is symmetric about this axis. Therefore, the lengths on either side of the axis are equal.

A. π;B. 3π;C. 9π;D. 36π; E. No correct answer

###### Answer : D Knowledge Concept : Area of a Circle Response for GPT-4o:

<Thought process>: The diagram shows a square with a side length of 6 units. The white part is composed of 4 identical sectors, which can be assembled into a circle. The radius of each sector is 3 units (half of the side length of the square). The area of one sector is (1/4)πr², where r is the radius. Therefore, the area of one sector is (1/4)π(3)² = (1/4)π(9) = (9/4)π. Since there are 4 sectors, the total area of the white part is 4 * (9/4)π

- 1. Since HG is the axis of symmetry, AC and BD are equal in length. Given that BD is 5 cm, AC must also be 5 cm.
- 2. Since H is the midpoint of AB, and AB is 4 cm (2 cm on each side of H), GC must be half of AC. Therefore, GC is 5 cm / 2 = 2.5 cm.
- 3. Since H is the midpoint of AB, AH is half of AB. Therefore, AH is 2 cm.

= 9π. <Answer>: E. No correct answer

- <Answer>: E. No correct answer

Figure 39: Specific examples of Hallucination.

- F Example Demonstration F.1 Description of the Knowledge Concepts Figure 5 and Figure 40 to 49 illustrate the the detailed information of knowledge concepts.

[Figure 429]

###### Understanding and Conversion of Units

Conversion Rates and Calculations Between Volume Units (Including Liters and Milliliters)

Description:

- 1. Conversion rates between volume units: 1 cubic meter (m³) = 1000 cubic decimeters (dm³); 1 cubic decimeter (dm³) = 1000 cubic centimeters (cm³); 1 cubic decimeter (dm³) = 1 liter (L); 1 cubic centimeter (cm³) = 1 milliliter (mL)
- 2. Converting between Different Units: To convert from a larger unit to a smaller unit, multiply by the conversion rate. To convert from a smaller unit to a larger unit, divide by the conversion rate.
- 3. To convert cubic meters to cubic decimeters: multiply by 1000.
- 4. To convert cubic decimeters to cubic centimeters: multiply by 1000.
- 5. To convert cubic decimeters to liters: 1 cubic decimeter equals 1 liter.
- 6. To convert cubic centimeters to milliliters: 1 cubic centimeter equals 1 milliliter.

[Figure 430]

[Figure 431]

1m3 1dm3 1cm3 1L 1mL

Conversion Rates and Calculations Between Area Units Description:

###### Conversion Rates and Calculations Between Length Units Description:

1. Conversion rates between area units: 1 square kilometer (km²) = 1,000,000 square meters (m²);

- 1. Conversion Rates between Length Units: 1 kilometer (km) = 1000 meters (m); 1 meter (m) = 10 decimeters (dm); 1 decimeter (dm) = 10 centimeters (cm); 1 centimeter (cm) = 10 millimeters (mm); 1 millimeter (mm) = 1000 nanometers (nm)
- 2. Converting between Different Units: To convert from a larger unit to a smaller unit, multiply by the conversion rate. To convert from a smaller unit to a larger unit, divide by the conversion rate.
- 3. To convert kilometers to meters: multiply by 1000.
- 4. To convert meters to decimeters: multiply by 10.
- 5. To convert decimeters to centimeters: multiply by 10.
- 6. To convert centimeters to millimeters: multiply by 10.
- 7. To convert millimeters to nanometers: multiply by 1000.
- 8. To convert nanometers to millimeters: divide by 1000.
- 9. To convert millimeters to centimeters: divide by 10.
- 10. To convert centimeters to decimeters: divide by 10.
- 11. To convert decimeters to meters: divide by 10.
- 12. To convert meters to kilometers: divide by 1000.

- 1 square meter (m²) = 100 square decimeters (dm²); 1 square decimeter (dm²) = 100 square centimeters (cm²)
- 2. To convert from a larger unit to a smaller unit, multiply by the conversion rate.
- 3. To convert from a smaller unit to a larger unit, divide by the conversion rate.
- 4. To convert square kilometers to square meters: multiply by 1,000,000.
- 5. To convert square meters to square decimeters: multiply by 100.
- 6. To convert square decimeters to square centimeters: multiply by 100.
- 7. To convert square centimeters to square decimeters: divide by 100.
- 8. To convert square decimeters to square meters: divide by 100.
- 9. To convert square meters to square kilometers: divide by 1,000,000.

10cm

1cm2

…

1km

10cm 1dm2

…

1m 1m 1m 1m 1m

1000

###### Figure 40: The description of the knowledge concept "Understanding and Conversion of Units"

[Figure 432]

###### Angles and Length

Understanding Angles (Using a Protractor) Description:

###### Understanding Length (Using a Ruler)

Description:

- 1. Steps to measure an angle: Align the center of the protractor with the vertex of the angle, and align the 0°baseline of the protractor with one side of the angle. The degree measure of the angle is indicated by the other side of the angle on the protractor. When reading the degree measure, pay attention to whether to use the outer scale or the inner scale. If the angle opens to the left, use the outer scale; if the angle opens to the right, use the inner scale.
- 2. The size of an angle is not related to the length of its two sides but to the amount they open. The wider the two sides open, the larger the angle; the narrower they open, the smaller the angle. Full angle > straight angle > obtuse angle > right angle > acute angle..

- 1. Length is usually measured using a ruler.
- 2. When using a ruler for measurement, first determine the measurement unit of the ruler. If there is no special indication, the default unit is centimeters.
- 3. If one end of the line segment being measured is aligned with the 0 mark on the ruler, the length of the line segment is the direct reading from the other end.
- 4. If one end of the line segment being measured is not aligned with the 0 mark, the length of the line segment is the difference between the readings at both ends.

[Figure 433]

0 1 2 3 4 5 6 7 8 9 10

Figure 41: The description of the knowledge concept "Angles and Length"

[Figure 434]

###### Basic Transformations of Figures

###### Translation Description: 1. A symmetry axis is a straight line. Description:

Axial Symmetry

- 2. In a symmetrical figure, the distance from corresponding points on either side of the symmetry axis to the axis is equal.
- 3. In a symmetrical figure, folding along the symmetry axis results in complete overlap of the left and right sides.
- 4. If two figures are symmetric about a certain line, then that line is the symmetry axis, and the line segment connecting corresponding points is bisected perpendicular to the symmetry axis.

- 1. Translation refers to the movement of all points of a figure in the same plane by the same distance in a given direction. This type of motion is called translation of the figure.
- 2. Translation does not change the shape, size, or orientation of the figure.
- 3. The shape and size of the figure remain unchanged after translation; only the position changes.
- 4. After translation, the line segments connecting corresponding points are parallel (or collinear) and equal in length.
- 5. Multiple consecutive translations are equivalent to a single translation.
- 6. A figure after an even number of symmetries is equivalent to the figure after a translation.
- 7. Translation is determined by direction and distance.
- 8. After translation, corresponding line segments are parallel (or collinear) and equal in length, corresponding angles are equal, and the line segments connecting corresponding points are parallel (or collinear) and equal in length.

Rotation Description:

- 1. The rotation of a figure involves the movement of every point on the figure in the plane around a fixed point by a fixed angle. It can be described by three elements: the center of rotation, the direction of rotation, and the angle of rotation.
- 2. The distances from corresponding points to the center of rotation are equal.
- 3. The angle between the line segment connecting corresponding points and the center of rotation equals the angle of rotation.
- 4. The figures before and after rotation are congruent, meaning their size and shape remain unchanged.
- 5. The center of rotation is the unique fixed point.

O

###### Figure 42: The description of the knowledge concept "Basic Transformations of Figures"

[Figure 435]

[Figure 436]

###### Direction

###### Position

###### Cardinal Directions (East, South, West, North) Description:

Up-Down Position

Description:

- 1. Maps usually use "up" to represent north, "down" to represent south, "left" to represent west, and "right" to represent east.
- 2. South is opposite to north, and west is opposite to east; northwest is opposite to southeast, and northeast is opposite to southwest.
- 3. East, south, west, and north are arranged in a clockwise direction.

North

| |
|---|
| |
| |

1. Up-down position Refers to the vertical positional relationship of objects.

East

West

Front-Back Position

South

Description:

[Figure 437]

1. Front-back position refers to the longitudinal positional relationship of objects in a horizontal direction.

###### Southeast, Southwest, Northeast, Northwest Directions Description:

North

Northwest Northeast

Left-Right Position

1.Northeast lies 45 degrees north of due east, southeast lies 45 degrees south of due east, northwest lies 45 degrees west of due north, and southwest lies 45 degrees west of due south.

Description:

East

West

1. Left-right position refers to the lateral positional relationship of objects in a horizontal direction.

Southwest Southeast

South

- Figure 43: The description of the knowledge concepts "Direction" and "Position"

[Figure 438]

###### Calculation of Solid Figures

Surface Area of Cylinders

Volume and Capacity of Cones Description:

Description:

- 1. Volume formula: The formula for calculating the volume of a cone is V = (1/3)πr²h, where r is the radius of the base of the cone and h is the height of the cone. This formula indicates that the volume of a cone is one-third the volume of a cylinder with the same base and height.
- 2. Relationship between height and volume: If the volume of a cone is known, the height of the cone can be calculated using the formula h = 3V/(πr²), where V is the volume of the cone and r is the radius of the base.
- 3. Relationship between base area and volume: Similarly, if the volume of a cone is known, the base area of the cone can be calculated using the formula A = 3V/h, where V is the volume of the cone and h is the height.
- 4. The capacity of a cone usually refers to the amount of space a conical container can hold. In the absence of special instructions, capacity and volume can be considered the same concept, using the same units and calculation methods.
- 5. Generally, π is taken as 3.14.

r

- 1. The lateral surface area of a cylinder = circumference of the base ×height.
- 2. The area of the base of a cylinder = π×radius squared.
- 3. The surface area of a cylinder refers to the sum of its lateral surface area and the areas of its two bases.

###### h

h

Surface Area of Rectangular Cuboids Description:

- 1. Definition of Surface Area: The total area of the 6 faces of a rectangular cuboid or a cube is called its surface area.
- 2. Surface area of a rectangular cuboid = length×width×2 + length ×height×2 + width×height×2.
- 3. Surface area of a rectangular cuboid = (length×width + length× height + width×height)×2.
- 4. Surface Area of a Rectangular Cuboid: If the letters a, b, and h represent the length, width, and height of a rectangular cuboid respectively, and S represents the surface area of the rectangular cuboid, then S = 2ab + 2ah + 2bh or S = 2(ab + ah + bh)..

r

h

b a

Volume and Capacity of Cylinders Description:

- 1. The formula for calculating the volume of a cylinder: The volume of a cylinder = base area×height.
- 2. In terms of letters: If V represents the volume of the cylinder, where r is the radius of the base and h is the height of the cylinder, then the formula for the volume of the cylinder is: V = πr²h.
- 3. The capacity of a cylinder usually refers to the amount of space a cylindrical container can hold. In the absence of special instructions, capacity and volume can be considered the same concept, using the same units and calculation methods.
- 4. π is generally taken as 3.14.

###### Surface Area of Cubes

Description:

- 1. Surface Area of a Cube: The total area of the 6 faces of a cube is its surface area, S = 6s.
- 2. Method of Calculating the Surface Area of a Cube: Surface area of a cube = edge length×edge length×6.
- 3. S = 6*a*a, where S represents the surface area of the cube, and a

h

| |
|---|

r

a

a a

represents the edge length of the cube.

Volume and Capacity of Rectangular Cuboids Description:

Volume and Capacity of Cubes Description:

- 1. The formula for calculating the volume of a rectangular cuboid: The volume of a rectangular cubed = length×width×height.
- 2. In terms of letters: If V represents the volume of the rectangular cuboid, and a, b, h represent the length, width, and height of the rectangular cubed respectively, then the formula for the volume of the rectangular cubed can be written as: V = abh.
- 3. The volume of a rectangular cuboid is equal to the base area multiplied by the height.
- 4. The capacity of a rectangular cuboid usually refers to the amount of space a rectangular container can hold. In the absence of special instructions, capacity and volume can be considered the same concept, using the same units and calculation methods.

- 1. The formula for calculating the volume of a cube: The volume of a cube = side length×side length×side length.
- 2. In terms of letters: If V represents the volume of the cube and a represents the side length of the cube, then the formula for the volume of the cube can be written as: V = a³.
- 3. The volume of a cube is equal to the base area multiplied by the height.
- 4. The capacity of a cube usually refers to the amount of space a cubic container can hold. In the absence of special instructions, capacity and volume can be considered the same concept, using the same units and calculation methods.

a

h

a

b a

a

- Figure 44: The description of the knowledge concept "Calculation of Solid Figures"

- 1. When observing an object from different angles, you will see two or three adjacent faces, and the shape observed on each face is different.
- 2. The top, front, and side faces of a rectangular cuboid or cube are defined by convention. The face facing upward is called the top face, the face facing the observer is called the front face, and the two side faces are called the side faces. It is impossible to see opposite faces of a rectangular cuboid or cube at the same time.
- 3. When observing a combination of objects, not only should the shape of each object be considered, but also the positional relationships between the objects. If an object is blocked by another object, the blocked part will be difficult to observe.
- 4. No matter which face a cube or sphere is observed from, the shape seen is always the same.

Description:

|Front View|
|---|

Top View

Side View

Expanded View of a Rectangular Cubiod

- 1. The expanded view of a rectangular cuboid is a twodimensional figure obtained by unfolding the faces of the rectangular cuboid and laying them flat.
- 2. A rectangular cuboid has 6 faces, and the expanded view consists of these 6 faces. Each face is a rectangle. Opposite faces of the rectangular cuboid are equal, so the expanded view will have three pairs of equal rectangles.

Description:

| |
|---|
| |

Expanded View of Rectangular Cuboids

- 1. Formation of a Cylinder: A cylinder can be formed by rotating a rectangle around one of its edges, or by rolling a rectangle into a cylindrical shape.
- 2. Components of a Cylinder: A cylinder consists of bases and a lateral surface. The bases are two equal circular surfaces, referred to as the upper base and the lower base; the lateral surface is a curved surface that connects the upper and lower bases.
- 3. Unfolded Lateral Surface of a Cylinder: When the lateral surface of a cylinder is unfolded, it forms a rectangle or a square. If the unfolded surface is a rectangle, its length is the circumference of the circular base, and its width is the height of the cylinder; if the unfolded surface is a square, it means the height of the cylinder is equal to the circumference of the circular base.

Description:

Expanded View of Cylinders

- 1. An expanded view of a cube is a two-dimensional representation of the cube, formed by unfolding the cube along its edges into a plane. It shows the arrangement of all six faces of the cube on the same plane.
- 2. An expanded view of a cube can have different arrangements, but it usually consists of one central square surrounded by four adjacent squares, with an additional square connected to the central square. Common shapes include "T" shape, "cross" shape, and "L" shape.
- 3. Each square in the expanded view represents one face of the cube, and every two adjacent squares in the expanded view are also adjacent faces in the cube.
- 4. There are a total of 6 squares in the expanded view, and each square has equal side lengths.

Description:

Expanded View of Cubes

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

- 1. The base of a cone is a circle, and the lateral surface of a cone is a curved surface.
- 2. The distance from the apex of the cone to the center of the base is the height of the cone.
- 3. When the lateral surface of a cone is unfolded, it forms a sector.
- 4. The line segment from the apex of the cone to any point on the edge of the base is called the slant height of the cone, and all slant heights are equal in length.

Properties of Cones Description:

- 1. The top and bottom surfaces of a cylinder are called the bases.
- 2. A cylinder has a curved surface called the lateral surface.
- 3. The distance between the two bases of a cylinder is called the height.
- 4. A cylinder is formed by rotating a rectangle 180°around one of its edges.
- 5. The bases of a cylinder are circular.
- 6. The heights of all cylinders are equal.

Properties of Cylinders Description:

- 1. A rectangular cuboid has six faces, all of which are rectangles.
- 2. The opposite faces of a rectangular cubed have equal area, and of the 12 edges, the lengths of the 4 opposite edges are equal.
- 3. A rectangular cuboid has 8 vertices.
- 4. The lengths of the three edges intersecting at one vertex of a rectangular cuboid are called length, width, and height.
- 5. The edge where two faces of a rectangular cuboid meet is called an edge.
- 6. The point where three edges of a rectangular cuboid meet is called a vertex.
- 7. When placed on a table, a rectangular cuboid can show at most three faces.
- 8. The total area of the six faces of a rectangular cuboid is called its surface area.

Description:

b a

h

- 1. A cube has six faces, all of which are squares.
- 2. All six faces of a cube have equal area.
- 3. A cube has 12 edges, and all edges are of equal length.
- 4. A cube has 8 vertices.
- 5. A cube can be considered a special type of rectangular cuboid.

Properties and Understanding of Cubes Description:

a

a

a

[Figure 439]

Understanding of Solid Figures

Observing Objects

Properties and Understanding of Rectangular Cuboids

- Figure 45: The description of the knowledge concept "Understanding of Solid Figures"

[Figure 440]

###### Calculation of Plane Figures

###### Sum of Interior Angles of Triangles Description:

Area of Parallelograms

Description:

- 1. The sum of the interior angles of any triangle is 180°.
- 2. According to the exterior angle theorem, an exterior angle of a triangle is equal to the sum of the two non-adjacent interior angles.

1. The area of a parallelogram is equal to the base times the height, S = ah.

h a

Area of Triangles

Sum of Interior Angles of Other Polygons

Description:

###### Description:

- 1. If the base is a and the height is h, then the area of a triangle is S = 1/2 ah.
- 2. The area of a right triangle is equal to the product of its two legs divided by two.
- 3. If the area and height of a triangle are known, its base length can be calculated. Similarly, if the area and base length of a triangle are known, its height can be determined.

- 1. The sum of the interior angles of a quadrilateral is 360°, and the sum of the interior angles of a pentagon is 540°.
- 2. The sum of the interior angles of a polygon = 180°×(number of sides - 2).
- 3. The sum of the interior angles of an n-sided polygon = (n - 2)× 180°.
- 4. A polygon with all sides of equal length and all interior angles equal is called a regular polygon.

h a

b

108°

a

Area of Sectors

Calculation and Comparison of Angles

Description:

- 1. Since the area of a sector with a central angle of 360°is the area of the circle, S = πr², the area of a sector with a central angle of n°is: S = nπr²÷360.
- 2. There is another formula for the area of a sector: S = 1/2 lr, where l is the arc length and r is the radius. The arc length l = nπr÷180
- 3. Generally, π is taken as 3.14.

Description:

- 1. Sum and difference of angles: Adding multiple angles gives the total angle, and subtracting gives the difference angle.
- 2. The size of an angle is not related to the length of its two sides, but only to the size of the angle's opening.
- 3. Comparison of angles: The size of angles can be compared directly by their degrees. Straight angle > obtuse angle > right angle > acute angle.

l

n° r

180°

Perimeter of Parallelograms

Description:

Area of Trapezoids Description:

1. A parallelogram has equal opposite sides, and its perimeter is twice the sum of its adjacent sides. The formula for the perimeter is C = 2(a + b), where a and b are the lengths of the sides of the parallelogram.

b

a

1. Using the letters a and b to represent the upper base and the lower base of a trapezoid, and the letter h to represent the height of the trapezoid, the formula for the area of a trapezoid can be expressed as S = 1/2 (a + b)×h.

a

h b

Perimeter of Triangles

Description:

- 1. The perimeter of a triangle is the sum of its three sides. Scalene triangle: C = a + b + c (where a, b, and c are the lengths of the three sides of the triangle).
- 2. Isosceles triangle: C = 2a + b (where a is the length of the equal sides, and b is the length of the base).
- 3. Equilateral triangle: C = 3a (where a is the length of any one side).

Area of Circles Description:

- a
- b c a a a

a a b

- 1. The area of a circle = pi×radius×radius. S = πr² = π( d/2 )²
- 2. Generally, π is taken as 3.14.

r

###### Perimeter of Trapezoids

- 1. In a trapezoid, the parallel sides are called the bases. The longer base is called the lower base, and the shorter base is called the upper base. The other two sides are called the legs. The perimeter of a trapezoid is the sum of the upper base, lower base, and the two legs. The formula for the perimeter is: upper base + lower base + leg + leg, denoted as L = a + b + c + d.
- 2. The formula for the perimeter of an isosceles trapezoid is: upper base + lower base + 2 legs, denoted as L = a + c + 2b.

Description:

a b d c

Area of Rectangles Description:

| |
|---|

a b b c

1. The area of a rectangle is equal to its length×width, expressed as: S = ab.

b

a

Circumference of Circles

Area of Squares Description:

Description:

- 1. Since the area of a sector with a central angle of 360°is the area of the circle, S = πr², the area of a sector with a central angle of n°is: S

= nπr²÷360.

- 2. There is another formula for the area of a sector: S = 1/2 lr, where l is the arc length and r is the radius. The arc length l = nπr÷180
- 3. Generally, π is taken as 3.14.

a

| |
|---|

- 1. The area of a square is equal to the square of its side length: S = a * a.
- 2. The area of a square is equal to the square of the length of its diagonal divided by two.

r

O

- a a

d

a

Perimeter of Squares Description:

Perimeter of Rectangles Description:

a

| |
|---|

| |
|---|

1. A rectangle has equal opposite sides, and the perimeter of a rectangle = (length + width)×2 (C = 2(a+b)).

1. A square has four equal sides, and the perimeter of a square = side length×4 (C = 4a).

b a

a

a

a

Figure 46: The description of the knowledge concept "Calculation of Plane Figures"

[Figure 441]

###### Understanding of Plane Figures

Properties and Understanding of Trapezoids Description:

Properties and Understanding of Squares Description:

- 1. A trapezoid (or trapezium) is a quadrilateral with only one pair of opposite sides parallel. The parallel sides are called the bases of the trapezoid: the longer base is called the lower base, and the shorter base is called the upper base; the other two sides are called the legs; the perpendicular segment between the two bases is called the height of the trapezoid.
- 2. A trapezoid with one leg perpendicular to the bases is called a right trapezoid.
- 3. A trapezoid with both legs equal in length is called an isosceles trapezoid.
- 4. The height of a trapezoid is the distance between the upper base and the lower base.

- 1. A square is a special type of parallelogram. A parallelogram with one pair of adjacent sides equal and one right angle is called a square, also known as a regular quadrilateral.
- 2. Both pairs of opposite sides are parallel; all four sides are equal; adjacent sides are perpendicular to each other.
- 3. All four angles are 90°, and the sum of the interior angles is 360°.
- 4. The diagonals are perpendicular to each other; the diagonals are equal in length and bisect each other; each diagonal bisects a pair of opposite angles.
- 5. A square is both a centrally symmetric figure and an axisymmetric figure (with four lines of symmetry).

a

| |
|---|

h

a

a

a

###### Understanding Circles Description:

- 1. In a plane, a circle is defined as the set of all points that are at a fixed distance from a fixed point.
- 2. A circle is an axisymmetric figure.
- 3. The center of the circle is called the center, usually denoted by the letter O. A line segment connecting the center to any point on the circle is called the radius, usually denoted by the letter r.
- 4. A line segment that passes through the center and has its endpoints on the circle is called the diameter, usually denoted by the letter d. The diameter is the longest line segment within the circle, and it is twice the length of the radius.
- 5. All radii of a circle are equal in length, and all diameters are equal in length.

Understanding Triangular Rulers

Description:

r

O

- 1. A triangular ruler is a triangular-shaped tool commonly used for measuring angles, drawing straight lines, and performing geometric constructions.
- 2. There are two types of triangular ruler: Isosceles right triangular ruler: One angle is 90°, and the other two angles are 45°and 45°. Scalene right triangular ruler: One angle is 90°, and the other two angles are 30°and 60°.

45° 45°

60°

###### Properties and Understanding of Rectangles Description:

30°

- 1. A rectangle is a plane figure and a parallelogram with one right angle. A rectangle is also defined as a parallelogram with all four angles being right angles. A square is a special type of rectangle where all four sides are equal in length.
- 2. Properties of a rectangle: The two diagonals are equal in length; the two diagonals bisect each other; both pairs of opposite sides are parallel; both pairs of opposite sides are equal in length; all four angles are right angles; there are two axes of symmetry; it is unstable (prone to deformation); the square of the length of a rectangle's diagonal is equal to the sum of the squares of its two sides; the quadrilateral formed by sequentially connecting the midpoints of a rectangle's sides is a rhombus.

Distance Between Two Points

| | |
|---|---|
| | |

Description:

- 1. The line segment between two points is the shortest distance between them.
- 2. In a plane, the length of the line segment with these two points as endpoints is the distance between the two points.
- 3. If two lines are parallel, then the shortest distance between them is perpendicular to both lines.

- a b

###### Understanding and Representing Angles Description:

###### Properties and Understanding of Parallelograms

###### Description:

- 1. A parallelogram is a closed shape in a plane formed by two pairs of parallel line segments.
- 2. A quadrilateral with two pairs of opposite sides parallel is called a parallelogram.
- 3. Rectangles, rhombuses, and squares are special types of parallelograms.
- 4. In a parallelogram, both pairs of opposite sides are equal in length, both pairs of opposite angles are equal, adjacent angles are supplementary, the height (distance between the parallel lines) is the same everywhere, and the diagonals bisect each other.
- 5. The height of a parallelogram is the perpendicular segment drawn from a point on one side to the opposite side, and this segment is called the height of the parallelogram.
- 6. To draw a parallelogram, first draw two parallel and equal-length line segments, then connect the endpoints of these two segments to form a quadrilateral.

- 1. Understanding angles from a static perspective: An angle is a figure formed by two rays originating from a single point.
- 2. Understanding angles from a dynamic perspective: When a ray rotates around its vertex to another position, the figure formed by these two rays is called an angle. Two rays with a common endpoint form an angle, this common endpoint is called the vertex of the angle, and the two rays are called the sides of the angle.
- 3. Since rays extend infinitely in one direction, the length of the sides of an angle is irrelevant to the size of the angle.
- 4. The size of an angle can be measured and compared.
- 5. Straight angle: A 180°angle. When the two sides of an angle are on the same line, the angle formed is called a straight angle. Specifically, when the ray OA rotates around point O, and the terminal side is on the extension line of the initial side OA in the opposite direction, it forms a straight angle.
- 6. Right angle: A 90°angle. When the ray OA rotates around point O, and the terminal side is perpendicular to the initial side, it forms a right angle. Half of a straight angle is called a right angle.
- 7. Acute angle: An angle greater than 0°and less than 90°. An angle smaller than a right angle is called an acute angle.
- 8. Obtuse angle: An angle greater than 90°and less than 180°. An angle greater than a right angle and less than a straight angle is called an obtuse angle.
- 9. Full angle: A 360°angle. When the ray OA rotates around point O, and the terminal side coincides with the initial side, it forms a full angle..

O A

O A

Understanding Line Segments, Lines, and Rays

O A

Description:

- 1. A line has no endpoints and can extend infinitely in both directions.
- 2. A ray has one endpoint and can extend infinitely in one direction.
- 3. A line segment has two endpoints, a fixed length, and can be measured.
- 4. Among all the lines connecting two points, the line segment is the shortest.

180°

O A

360°

###### O A（B）

Understanding Sectors

Description:

Properties and Understanding of Triangles Description:

- 1. A sector is a shape formed by a circular arc and the two radii connecting the endpoints of the arc to the center of the circle.
- 2. All radii in a sector are equal in length.
- 3. The part of the circle between two points A and B is called an "arc".
- 4. An angle with its vertex at the center of the circle is called a "central angle".

- A
- B

r o

- 1. A triangle is a closed geometric figure formed by three line segments connected end-to-end in sequence.
- 2. By angle classification: Acute triangle: All three interior angles of the triangle are less than 90 degrees. Right triangle: One of the three interior angles of the triangle is exactly 90 degrees. Obtuse triangle: One of the three interior angles of the triangle is greater than 90 degrees.
- 3. By side classification: Scalene triangle: A triangle where all three sides are of different lengths. Isosceles triangle: A triangle with two equal sides. Equilateral triangle: A triangle with all three sides equal.
- 4. The sum of the interior angles of a triangle in a plane is 180 degrees.
- 5. The sum of the lengths of any two sides of a triangle is greater than the length of the third side, and the difference between the lengths of any two sides is less than the length of the third side.
- 6. The height of a triangle is the perpendicular line segment drawn from a vertex to the opposite side.
- 7. To draw the height of a triangle: Choose one side as the base. Find the vertex opposite the base. Using a ruler and compass, draw a perpendicular line from the vertex to the base. Ensure to mark the right angle symbol, indicating it is the height.
- 8. Once the lengths of the three sides of a triangle are determined, the triangle is fixed. The angles and the area enclosed by the three sides do not change.

arc r

h

Parallel Description:

h

- 1. The positional relationship between two lines in the same plane is either parallel or intersecting.
- 2. Two lines in the same plane that do not intersect are called parallel lines. It can also be said that these two lines are parallel to each other.
- 3. Through a point outside a line, only one line can be drawn parallel to the given line.

h

Perpendicularity Description:

- 1. Perpendicular lines intersect at right angles, and their intersection point is called the foot of the perpendicular. If a point on one line intersects another line forming a right angle, then this line is perpendicular to the other line.
- 2. A line is perpendicular to another line if the angle between them is 90 degrees. Similarly, a line is perpendicular to a plane if it forms a 90-degree angle with any line lying in that plane.
- 3. The perpendicular segment from a point to a line is the shortest distance from the point to the line.
- 4. From a given point on a line and a point not on the line, only one perpendicular line can be drawn to the given line.

Observing Figures Description:

90°

[Figure 442]

1. Count or identify various shapes within a plane figure. For example, by observing the length and number of smaller unit shapes, you can determine the side length or area of a larger shape.

Figure 47: The description of the knowledge concept "Understanding of Plane Figures"

[Figure 443]

###### Route Map

Determining the Positions of Objects Based on Direction, Angle, and Distance Description:

North

- 1. Determining Direction: Common basic directions include east, west, south, and north, as well as intermediate directions such as southeast, southwest, northeast, and northwest. In mathematics, "up" usually represents north, "down" represents south, "left" represents west, and "right" represents east to indicate directions.
- 2. Determining Angle: Angles are used to describe the relative position between objects or the deviation of an object from a standard direction. For example, "30 degrees west of north" describes a direction that is 30 degrees to the left of the north direction.
- 3. Determining Distance: Distance refers to the straight-line length from one point to another. When describing a position, it is necessary to specify the distance along a certain direction from a reference point.
- 4. Describing Position: To describe the position of an object, first determine a reference point, then describe the direction, angle, and distance from the reference point to the object based on its actual location. For example, "The hospital is located 100 meters in the direction 30 degrees west of north from the library."

hospital

100m 30°

library

West

Describing Simple Routes Based on Direction and Distance Description:

- 1. Determining Direction: Commonly used basic directions include east, west, south, and north, as well as intermediate directions such as southeast, southwest, northeast, and northwest. In teaching, the convention is to use "up" for north, "down" for south, "left" for west, and "right" for east to identify directions.
- 2. Determining Angles: Angles are used to describe the relative position between objects or the deviation of an object from a standard direction. For example, ”45 degrees west by south" describes a direction 45 degrees to the left of due south.
- 3. Determining Distance: Distance refers to the straight-line length from one point to another. When describing a route, it is necessary to indicate how far to travel from the starting point along a certain direction.
- 4. Describing Routes: To describe a simple route, first determine the starting position, then describe the direction, angle, and distance from the starting point to the destination based on the actual path taken. For example, "Starting from the school, walk 50 meters east to the traffic light, then walk 100 meters in a direction 45 degrees south by west to reach the library."

[Figure 444]

[Figure 445]

east 50m

southwest 45° 100m

[Figure 446]

Figure 48: The description of the knowledge concept "Route Map"

[Figure 447]

###### Correspondence of Coordinates and Positions

###### Finding Positions Based on Ordered Pairs

Representing Positions Using Ordered Pairs Description:

Description: 1. Definition of an Ordered Pair: An ordered pair is a combination of two numbers, typically used to represent the position of a point in a Cartesian coordinate system. The two numbers in the ordered pair represent values on different directions or axes.

- 1. Representation of an Ordered Pair: An ordered pair consists of two numbers, with the first number representing the position on the horizontal axis (column) and the second number representing the position on the vertical axis (row). For example, the ordered pair (2, 4) represents the position in column 2, row 4.
- 2. Method of Determining Position: To find a position based on an ordered pair, first identify a reference point on the plane. Then move horizontally according to the first number in the ordered pair, and move vertically according to the second number. The final position reached is the point represented by the ordered pair.

[Figure 448]

[Figure 449]

- 2. Composition of an Ordered Pair: In an ordered pair (x, y), the first number x represents the column number, which is the horizontal position; the second number y represents the row number, which is the vertical position. This notation follows the "horizontal first, then vertical" convention.
- 3. Notation of an Ordered Pair: When using an ordered pair to represent a point's position, the numbers or letters representing the column and row need to be enclosed in parentheses, for example, (2, 5) represents the position at column 2, row 5.
- 4. Uniqueness of an Ordered Pair: Each ordered pair uniquely determines a position, and conversely, a position can be uniquely represented by an ordered pair.

Figure 49: The description of the knowledge concept "Correspondence of Coordinates and Positions"

F.2 Data Sources of WE-MATH Table 10 to 14 illustrate the the detailed data source lists of WE-MATH.

- Table 10: The data sources of WE-MATH (Part1, Source 1 to 50).

|Number|Data Source|
|---|---|
|1|[Beijing Chaoyang New Target Detection] 2022 Edition of People’s Education Edition Grade 3 Volume 1: Mathematics|
|2|[Beijing New Target Detection] 2023 Printing People’s Education Edition Grade 3 Volume 1: Mathematics|
|3<br><br>|[Beijing Chaoyang New Target Detection] 2023 Printing People’s Education Edition Grade 3 Volume 2: Mathematics|
|4<br><br>|[Learning Objectives and Assessment] 2023-2024 Academic Year People’s Education Edition Grade 3 Volume 2: Mathematics|
|5|[Beijing Chaoyang New Target Detection] 2020 Edition of the People’s Education Edition Grade 4 Volume 1: Mathematics|
|6<br><br>|[New Target Detection] 2022 Edition of the People’s Education Press Grade 4 Volume 2: Mathematics|
|7<br><br>|[Beijing Chaoyang New Target Detection] 2022 Edition of People’s Education Edition Grade 4 Volume 1: Mathematics|
|8|[Beijing Chaoyang New Target Detection] 2023 Printing People’s Education Edition Grade 4 Volume 2: Mathematics|
|9|[Beijing New Target Detection] 2023 Printing People’s Education Edition Grade 4 Volume 1: Mathematics|
|10|[Learning Objectives and Assessment] 2023-2024 Academic Year People’s Education Press Grade 4 Volume 2: Mathematics|
|11|[Beijing Chaoyang New Target Detection] 2021 Edition of the People’s Education Edition Grade 5 Volume 1: Mathematics|
|12<br><br>|[New Target Detection] 2022 Edition of the People’s Education Edition Grade 5 Volume 2: Mathematics|
|13|[Beijing Chaoyang New Target Detection] 2022 Edition of the People’s Education Edition Grade 5 Volume 1: Mathematics|
|14|[Beijing Chaoyang New Target Detection] 2023 Printing People’s Education Edition Grade 5 Volume 2: Mathematics|
|15<br><br>|[Beijing New Target Detection] 2023 Printing People’s Education Edition Grade 5 Volume 1: Mathematics|
|16<br><br>|[Learning Objectives and Tests] 2023-2024 Academic Year People’s Education Edition Grade 5 Volume 2: Mathematics|
|17<br><br>|[Beijing Chaoyang New Target Detection] 2023 Printing People’s Education Edition Sixth Grade Volume 2: Mathematics|
|18<br><br>|[New Target Detection] 2022 Edition of the People’s Education Press Grade 6 Volume 2: Mathematics|
|19|[Beijing Chaoyang New Target Detection] 2022 Edition of People’s Education Edition Grade 6 Volume 1: Mathematics|
|20|[Beijing Chaoyang New Target Detection] 2023 Edition of the People’s Education Press Sixth Grade Volume 1: Mathematics|
|21|[Learning Objectives and Tests] 2023-2024 Academic Year Sixth Grade People’s Education Press Volume 2: Mathematics|
|22|[Beijing Xicheng Learning Exploration Diagnosis] 2022 Edition of People’s Education Edition Grade 3 Volume 1: Mathematics|
|23<br><br>|[Beijing Learning Inquiry Diagnosis] 2023 Printing People’s Education Press Grade 3 Volume 2: Mathematics|
|24|[Beijing Learning Inquiry Diagnosis] 2023 Printing People’s Education Press Grade 3 Volume 1: Mathematics|
|25<br><br>|[Learning Exploration Diagnosis] 2023-2024 Academic Year People’s Education Edition Grade 3 Volume 2: Mathematics|
|26<br><br>|[Learning Exploration Diagnosis] 2022 Edition of the People’s Education Press Grade 4 Volume 2: Mathematics|
|27<br><br>|[Beijing Xicheng Learning Exploration Diagnosis] 2022 Edition of People’s Education Edition Grade 4 Volume 1: Mathematics|
|28<br><br>|[Beijing Learning Inquiry Diagnosis] 2023 Printing People’s Education Press Grade 4 Volume 2: Mathematics|
|29|[Beijing Learning Inquiry Diagnosis] 2023 Printing People’s Education Press Grade 4 Volume 1: Mathematics|
|30<br><br>|[Learning Exploration Diagnosis] 2023-2024 Academic Year People’s Education Edition Grade 4 Volume 2: Mathematics|
|31<br><br>|[Learning Exploration Diagnosis] 2022 Edition of the People’s Education Press Grade 5 Volume 2: Mathematics|
|32<br><br>|[Beijing Xicheng Learning Exploration Diagnosis] 2022 Edition of the People’s Education Press Fifth Grade Volume 1: Mathematics|
|33|[Beijing Learning Inquiry Diagnosis] 2023 Printing People’s Education Press Grade 5 Volume 2: Mathematics|
|34|[Beijing Learning Inquiry Diagnosis] 2023 Printing People’s Education Press Grade 5 Volume 1: Mathematics|
|35|[Learning Exploration Diagnosis] 2023-2024 Academic Year People’s Education Edition Grade 5 Volume 2: Mathematics|
|36|[Learning Exploration Diagnosis] 2022 Edition of the People’s Education Press Grade 6 Volume 2: Mathematics|
|37<br><br>|[Beijing Learning Inquiry Diagnosis] 2022 Edition of People’s Education Press Grade 6 Volume 1: Mathematics|
|38|[Beijing Learning Exploration Diagnosis] 2023 Printing People’s Education Press Grade 6 Volume 2: Mathematics|
|39|[Beijing Learning Inquiry Diagnosis] 2023 Printing People’s Education Press Sixth Grade Volume 1: Mathematics|
|40<br><br>|[Learning Exploration Diagnosis] 2023-2024 Academic Year People’s Education Edition Sixth Grade Volume 2: Mathematics|
|41<br><br>|[Beijing Haidian famous teachers accompany you to study and practice synchronously] 2022 Beijing Normal University Edition Grade 3 Volume 1: Mathematics|
|42<br><br>|[Beijing Haidian famous teachers accompany you to study and practice synchronously] 2022 print Beijing Normal University edition Grade 3, Volume 2: Mathematics|
|43|[Beijing Companion Learning Synchronous Learning Handbook] 2023 Printing Beijing Normal University Edition Grade 3 Volume 1: Mathematics|
|44<br><br>|[Synchronous Learning Handbook for You] 2023-2024 Academic Year Beijing Normal University Edition Grade 3 Volume 2: Mathematics|
|45<br><br>|[Famous teachers from Haidian accompany you to study, practice and test] 2022 Beijing Normal University Grade 4 Volume 2: Mathematics|
|46<br><br>|[Beijing Haidian famous teachers accompany you to study and practice synchronously] 2022 Beijing Normal University Edition Grade 4 Volume 1: Mathematics|
|47<br><br>|[Beijing Haidian famous teachers accompany you to study and practice synchronously] 2022 Beijing Normal University Edition Grade 4 Volume 2: Mathematics|
|48|[Beijing Companion Learning Synchronous Learning Handbook] 2023 Printing Beijing Normal University Edition Grade 4 Volume 1: Mathematics|
|49|[Synchronous Learning Handbook for You] 2023-2024 Academic Year Beijing Normal University Edition Grade 4 Volume 2: Mathematics|
|50|[Haidian famous teachers accompany you to study synchronous learning and practice book] 2021 edition of fifth grade volume 1: Mathematics|

###### Table 11: The data sources of WE-MATH (Part2, Source 51 to 100).

|Number|Data Source|
|---|---|
|51<br><br>|[Haidian famous teachers accompany you to study and practice synchronously] 2022 edition of Beijing Normal University Grade 5 Volume 2: Mathematics|
|52|[Beijing Haidian famous teachers accompany you to study synchronously] 2022 Beijing Normal University Edition Grade 5 Volume 2: Mathematics|
|53|[Beijing Companion Learning Synchronous Learning Handbook] 2023 Printing Beijing Normal University Edition Grade 5 Volume 1: Mathematics|
|54|[Synchronous Learning Handbook for You] 2023-2024 Beijing Normal University Edition Grade 5 Volume 2: Mathematics|
|55|[Haidian famous teachers accompany you to study synchronous learning and practice book] 2021 edition of sixth grade volume: Mathematics|
|56<br><br>|[Haidian famous teachers accompany you to study and practice synchronously] 2022 Beijing Normal University Grade 6 Volume 2: Mathematics|
|57|[Beijing Haidian Famous Teacher] 2022 Beijing Normal University Edition Sixth Grade Volume 1: Mathematics|
|58|[Beijing Haidian famous teachers accompany you to study and practice synchronously] 2022 Beijing Normal University Edition Grade 6 Volume 2: Mathematics|
|59|[Beijing Companion Learning Synchronous Learning Handbook] 2023 Printing Beijing Normal University Edition Sixth Grade Volume 1: Mathematics|
|60<br><br>|[Synchronous Learning Handbook for You] 2023-2024 Academic Year Beijing Normal University Edition Sixth Grade Volume 2: Mathematics|
|61<br><br>|[Beijing Dongcheng Formative Independent Evaluation] 2022 Edition of People’s Education Edition Grade 3 Volume 1: Mathematics|
|62|[Formative Self-Evaluation] 2023-2024 Academic Year Grade 3, Volume 2: Mathematics|
|63<br><br>|[Beijing Formative Independent Assessment] 2023 Printing People’s Education Press Grade 3 Volume 1: Mathematics|
|64|[Beijing Formative Independent Assessment] 2023 Printing People’s Education Press Grade 3 Volume 1: Mathematics|
|65|[Beijing Dongcheng Formative Independent Evaluation] 2022 Edition of the People’s Education Press Grade 4 Volume 1: Mathematics|
|66|[Formative Self-Evaluation] 2023-2024 Academic Year People’s Education Edition Grade 4 Volume 2: Mathematics|
|67<br><br>|[Beijing Formative Independent Assessment] 2023 Printing People’s Education Press Grade 4 Volume 1: Mathematics|
|68<br><br>|[Formative Self-Evaluation] 2022 Edition of the People’s Education Press Grade 5 Volume 2: Mathematics|
|69|[Beijing Dongcheng Formative Independent Evaluation] 2022 Edition of the People’s Education Press Fifth Grade Volume 1: Mathematics|
|70<br><br>|[Formative Self-Evaluation] 2023-2024 Academic Year People’s Education Edition Grade 5 Volume 2: Mathematics|
|71|[Beijing Formative Independent Assessment] 2023 Printing People’s Education Edition Grade 5 Volume 1: Mathematics|
|72|[Formative Self-Evaluation] 2022 Edition of the People’s Education Press Grade 6 Volume 2: Mathematics|
|73|[Beijing Dongcheng Formative Independent Evaluation] 2022 Edition of the People’s Education Press Sixth Grade Volume 1: Mathematics|
|74<br><br>|[Formative Self-Evaluation] 2023-2024 Academic Year Sixth Grade People’s Education Edition Volume 2: Mathematics|
|75|[Beijing Formative Independent Assessment] 2023 Printing People’s Education Press Sixth Grade Volume 1: Mathematics|
|76|[All-in-one study, practice and examination] 2023-2024 school year Beijing edition Grade 3, Volume 2: Mathematics|
|77<br><br>|[All-in-one study, practice and examination] 2023-2024 school year People’s Education Press Grade 3 Volume 2: Mathematics|
|78<br><br>|[Beijing all-in-one study, practice and examination] 2023 Printing People’s Education Press Grade 3 Volume 1: Mathematics|
|79<br><br>|[Beijing all-in-one study, practice and examination] 2023 Beijing Edition Grade 3 Volume 2: Mathematics|
|80<br><br>|[Beijing all-in-one study, practice and examination] 2022 Beijing Edition Grade 3 Volume 2: Mathematics|
|81|[Beijing all-in-one study, practice and examination] 2022 Edition of People’s Education Press Grade 3 Volume 2: Mathematics|
|82<br><br>|[Beijing all-in-one study, practice and examination] 2022 Beijing Normal University Edition Grade 3 Volume 1: Mathematics|
|83|[All-in-one study, practice and examination] 2023-2024 school year Beijing Edition Grade 4 Volume 2: Mathematics|
|84|[All-in-one study, practice and examination] 2023-2024 school year People’s Education Press Grade 4 Volume 2: Mathematics|
|85<br><br>|[Beijing all-in-one study, practice and examination] 2023 Beijing Edition Grade 4 Volume 1: Mathematics|
|86|[Beijing all-in-one study, practice and examination] 2023 Printing People’s Education Press Grade 4 Volume 1: Mathematics|
|87|[Beijing all-in-one study, practice and examination] 2022 Beijing Edition Grade 4 Volume 2: Mathematics|
|88<br><br>|[Beijing all-in-one study, practice and examination] 2022 Edition of People’s Education Press Grade 4 Volume 2: Mathematics|
|89|[Beijing all-in-one study, practice and examination] 2022 Beijing Normal University Edition Grade 4 Volume 1: Mathematics|
|90|[Beijing all-in-one study, practice and examination] 2021 Edition Grade 4 Volume 1: Mathematics|
|91|[All-in-one study, practice and examination] 2023-2024 school year Beijing Edition Grade 5 Volume 2: Mathematics|
|92<br><br>|[All-in-one study, practice and examination] 2023-2024 school year People’s Education Press Grade 5 Volume 2: Mathematics|
|93|[Beijing all-in-one study, practice and examination] 2023 Beijing Edition Grade 5 Volume 1: Mathematics|
|94|[Beijing all-in-one study, practice and examination] 2023 Printing People’s Education Edition Grade 5 Volume 1: Mathematics|
|95<br><br>|[Beijing all-in-one study, practice and examination] 2022 Beijing Edition Grade 5, Volume 2: Mathematics|
|96|[Beijing all-in-one study, practice and examination] 2022 Edition of People’s Education Edition Grade 5 Volume 2: Mathematics|
|97|[All-in-one study, practice and examination] 2022 Beijing Normal University Edition Grade 5 Volume 1: Mathematics|
|98|[Beijing all-in-one study, practice and examination] 2021 Edition Grade 5 Volume 1: Mathematics|
|99<br><br>|[All-in-one study, practice and examination] 2023-2024 school year Beijing edition sixth grade volume 2: Mathematics|
|100|[All-in-one study, practice and examination] 2023-2024 school year People’s Education Press Grade 6 Volume 2: Mathematics|

###### Table 12: The data sources of WE-MATH (Part3, Source 101 to 150).

|Number<br><br>|Data Source|
|---|---|
|101<br><br>|[Beijing all-in-one study, practice and examination] 2023 Beijing Edition Sixth Grade Volume 1: Mathematics|
|102|[Beijing all-in-one study, practice and examination] 2023 Printing People’s Education Press Sixth Grade Volume 1: Mathematics|
|103|[Beijing all-in-one study, practice and examination] 2022 Beijing Edition Grade 6 Volume 2: Mathematics|
|104|[Beijing all-in-one study, practice and examination] 2022 Edition of People’s Education Press Sixth Grade Volume 2: Mathematics|
|105|[Beijing all-in-one study, practice and examination] 2022 Printing Beijing Normal University Edition Grade 6 Volume 2: Mathematics|
|106<br><br>|[Beijing all-in-one study, practice and examination] 2022 Beijing Normal University Edition Sixth Grade Volume 1: Mathematics|
|107|[Beijing all-in-one study, practice and examination] 2021 Edition Sixth Grade Volume 1: Mathematics|
|108|[Beijing Class Workbook] 2023 Beijing Edition Grade 4 Volume 1: Mathematics|
|109<br><br>|[Class Workbook] 2023-2024 Academic Year Beijing Edition Grade 4 Volume 2: Mathematics|
|110|[Zhejiang Class Workbook] 2022 Edition People’s Education Edition Grade 4 Volume 1: Mathematics|
|111|[Class Workbook] 2023-2024 Beijing Edition Grade 5, Volume 2: Mathematics|
|112|[Beijing Class Workbook] 2023 Beijing Edition Grade 5 Volume 1: Mathematics|
|113<br><br>|[Zhejiang Class Workbook] 2022 Beijing Normal University Edition Grade 5 Volume 1: Mathematics|
|114<br><br>|[Zhejiang Class Workbook] 2022 Edition People’s Education Edition Grade 5 Volume 1: Mathematics|
|115<br><br>|[Beijing Class Workbook] 2023 Beijing Edition Grade 6 Volume 1: Mathematics|
|116<br><br>|[Class Workbook] 2023-2024 Academic Year Beijing Edition Grade 6 Volume 2: Mathematics|
|117|[Zhejiang Class Workbook] 2022 Beijing Normal University Edition Sixth Grade Volume 1: Mathematics|
|118|[Zhejiang Class Workbook] 2022 Edition People’s Education Edition Sixth Grade Volume 1: Mathematics|
|119|[Mathematics Textbook] 2023-2024 Academic Year People’s Education Press Grade 3 Volume 2: Mathematics|
|120|[Mathematics Textbook] 2022 Shanghai Education Edition Grade 3 Volume 1: Mathematics|
|121<br><br>|[Beijing Mathematics Textbook] 2022 Beijing Edition Grade 3 Volume 1: Mathematics|
|122|[Beijing Mathematics Textbook] 2022 Beijing Normal University Edition Grade 3 Volume 1: Mathematics|
|123|[Shanghai Mathematics Textbook] 2021 Shanghai Education Edition Grade 3 Volume 2: Mathematics|
|124|[Beijing Mathematics Textbook] 2021 Beijing Edition Grade 3 Volume 2: Mathematics|
|125<br><br>|[Mathematics Textbook] 2020 Beijing Normal University Edition Grade 3 Volume 2: Mathematics|
|126<br><br>|[Beijing Mathematics Textbook] 2020 Edition People’s Education Press Grade 3 Volume 1: Mathematics|
|127|[Mathematics Textbook] 2023-2024 Academic Year People’s Education Press Grade 4 Volume 2: Mathematics|
|128|[Shanghai Mathematics Textbook] 2022 Shanghai Education Edition Grade 4 Volume 1: Mathematics|
|129|[Beijing Mathematics Textbook] 2022 Beijing Edition Grade 4 Volume 1: Mathematics|
|130<br><br>|[Beijing Mathematics Textbook] 2022 Beijing Normal University Edition Grade 4 Volume 1: Mathematics|
|131<br><br>|[Shanghai Mathematics Textbook] 2021 Shanghai Education Edition Grade 4 Volume 2: Mathematics|
|132<br><br>|[Beijing Mathematics Textbook] 2021 Beijing Edition Grade 4 Volume 2: Mathematics|
|133|[Beijing Mathematics Textbook] 2020 Beijing Normal University Edition Grade 4 Volume 2: Mathematics|
|134<br><br>|[Beijing Mathematics Textbook] 2019 Edition People’s Education Press Grade 4 Volume 1: Mathematics|
|135<br><br>|[Mathematics Textbook] 2023-2024 Academic Year People’s Education Edition Grade 5 Volume 2: Mathematics|
|136<br><br>|[Shanghai Mathematics Textbook] 2022 Shanghai Education Edition Grade 5 Volume 1: Mathematics|
|137<br><br>|[Mathematics Textbook] 2022 Beijing Normal University Edition Grade 5 Volume 1: Mathematics|
|138|[Shanghai Mathematics Textbook] 2022 Shanghai Education Edition Grade 5 Volume 1: Mathematics|
|139|[Mathematics Textbook] 2022 Beijing Normal University Edition Grade 5 Volume 1: Mathematics|
|140<br><br>|[Beijing Mathematics Textbook] 2022 Beijing Edition Grade 5 Volume 1: Mathematics|
|141|[Beijing Mathematics Textbook] 2021 Beijing Edition Grade 5 Volume 2: Mathematics|
|142<br><br>|[Shanghai Mathematics Textbook] 2020 Shanghai Education Edition Grade 5 Volume 2: Mathematics|
|143<br><br>|[Beijing Mathematics Textbook] 2020 Beijing Normal University Edition Grade 5 Volume 2: Mathematics|
|144|[Beijing Mathematics Textbook] 2020 Edition People’s Education Edition Grade 5 Volume 1: Mathematics|
|145<br><br>|[Beijing Mathematics Textbook] 2022 Edition People’s Education Press Sixth Grade Volume 2: Mathematics|
|146|[Beijing Mathematics Textbook] 2022 Beijing Edition Sixth Grade Volume 1: Mathematics|
|147|[Mathematics Textbook] 2022 Beijing Normal University Edition Grade 6 Volume 1: Mathematics|
|148|[Beijing Mathematics Textbook] 2021 Beijing Normal University Edition Grade 6 Volume 2: Mathematics|
|149<br><br>|[Beijing Mathematics Textbook] 2021 Beijing Edition Sixth Grade Volume 2: Mathematics|
|150|[Beijing Mathematics Textbook] 2018 Edition People’s Education Press Sixth Grade Volume 1: Mathematics|

###### Table 13: The data sources of WE-MATH (Part4, Source 151 to 200).

|Number<br><br>|Data Source|
|---|---|
|151<br><br>|2022 Sichuan Liangshan Primary School to Junior High School Examination Paper (People’s Education Edition): Mathematics|
|152|2022 Chongqing Yubei District Primary School to Junior High School Examination Paper (People’s Education Edition Examination): Mathematics|
|153<br><br>|2022 Guizhou Qiandongnan Primary School to Junior High School Examination Paper (People’s Education Edition Examination): Mathematics|
|154<br><br>|2022 Anhui Fuyang Taihe County Primary School to Junior High School Examination Paper (Beijing Normal University Edition Examination): Mathematics|
|155<br><br>|2022 Guangdong Huizhou Huiyang District Primary School to Junior High School Examination Paper (Beijing Normal University Edition): Mathematics|
|156|2022 Guangdong Shaoguan Xinfeng County Primary School to Junior High School Examination Paper (People’s Education Edition): Mathematics|
|157|2022 Guangdong Zhanjiang Mazhang District Primary School to Junior High School Examination Paper (Beijing Normal University Edition Examination): Mathematics|
|158|2022 Gansu Dingxi Minxian Primary School to Junior High School Examination Paper (Beijing Normal University Edition): Mathematics|
|159|2022 Guangdong Jieyang Jiedong District Primary School to Junior High School Examination Paper (Beijing Normal University Edition Examination): Mathematics|
|160<br><br>|2022 Henan Luohe Wuyang County Primary School Entrance Examination Paper (People’s Education Edition): Mathematics|
|161<br><br>|2022 Tianjin Primary School to Junior High School Examination Paper (Primary School to Junior High School in Some Districts): Mathematics|
|162|2022 Hebei Tangshan Lunan District Primary School Entrance Examination Paper (Hebei Education Edition Examination): Mathematics|
|163<br><br>|2022 Hebei Baoding Qingyuan District Primary School to Junior High School Examination Paper (People’s Education Edition): Mathematics|
|164|2022 Xinjiang Turpan Primary School to Junior High School Examination Paper (People’s Education Edition): Mathematics|
|165|2022 Hebei Shijiazhuang Luquan District Primary School Entrance Examination Paper (Hebei Education Edition): Mathematics|
|166|2022 Hainan Ledong Li Autonomous County Primary School to Junior High School Examination Paper (People’s Education Edition Examination): Mathematics|
|167|2022 Henan Jiyuan Primary School to Junior High School Examination Paper (People’s Education Edition): Mathematics|
|168|2021 Beijing Fengtai District Primary School to Junior High School Examination Paper (People’s Education Edition Examination): Mathematics|
|169|2021 Yunnan Kunming Wuhua District Primary School to Junior High School Examination Paper: Mathematics|
|170<br><br>|2021 Yunnan Kunming Xishan District Primary School Entrance Examination Paper: Mathematics|
|171|2021 Shaanxi Xi’an Beilin District Primary School to Junior High School Examination Paper (Part 2): Mathematics|
|172<br><br>|2021 Shaanxi Xi’an Beilin District Primary School to Junior High School Examination Paper: Mathematics|
|173<br><br>|2021 Zhejiang Ningbo Haishu District Primary School to Junior High School Examination Paper: Mathematics|
|174|2021 Shaanxi Xi’an Weiyang District Primary School to Junior High School Examination Paper: Mathematics|
|175|2021 Shaanxi Xi’an Yanta District Primary School to Junior High School Examination Paper (Part 5): Mathematics|
|176|2021 Shaanxi Xi’an Beilin District Primary School to Junior High School Examination Paper: Mathematics|
|177|"2021 Primary to Junior High School Examination Paper in Baqiao District, Xi’an, Shaanxi: Mathematics|
|178<br><br>|2021 Shaanxi Xi’an Weiyang District Primary School to Junior High School Examination Paper (Part 3): Mathematics|
|179<br><br>|"2021 Primary to Junior High School Examination Paper in Erqi District, Zhengzhou, Henan: Mathematics|
|180|2021 Jiangsu Nantong Primary School to Junior High School Examination Paper (Main Urban Area): Mathematics|
|181<br><br>|2021 Jiangsu Suzhou Xiangcheng District Primary School to Junior High School Examination Paper 5 to 6 Direct Promotion Class: Mathematics|
|182<br><br>|"2021 Primary to Junior High School Examination Paper for Baqiao District, Xi’an, Shaanxi (Part 5): Mathematics|
|183<br><br>|2021 Hunan Changsha Yuhua District Primary School to Junior High School Examination Paper: Mathematics|
|184|"2021 Primary to Junior High School Examination Paper of Jingxing County, Shijiazhuang, Hebei (Hebei Education Edition): Mathematics|
|185|2021 Hebei Shijiazhuang Pingshan County Primary School Entrance Examination Paper (Hebei Education Edition): Mathematics|
|186|2021 Hebei Shijiazhuang Lingshou County Primary School to Junior High School Examination Paper (Hebei Education Edition): Mathematics|
|187|"2021 Primary to Junior High School Examination Paper in Yanta District, Xi’an, Shaanxi: Mathematics|
|188|2021 Shaanxi Xi’an Xincheng District Primary School to Junior High School Examination Paper (Part 3): Mathematics|
|189<br><br>|"2021 Primary to Junior High School Examination Paper of Yuanshi County, Shijiazhuang, Hebei (Hebei Education Edition): Mathematics|
|190|"2021 Primary to Junior High School Examination Paper in Zhengding County, Shijiazhuang, Hebei (Hebei Education Edition): Mathematics|
|191<br><br>|2021 Shaanxi Xi’an Yanta District Primary School to Junior High School Examination Paper (Part 14): Mathematics|
|192|2021 Chongqing Shapingba District Chongqing Nankai Middle School Primary School to Junior High School Examination Paper (Part 3): Mathematics|
|193|2021 Shaanxi Xi’an Yanta District Primary School to Junior High School Examination Paper (Part 4A): Mathematics|
|194|"2021 Primary to Junior High School Examination Paper in Yanta District, Xi’an, Shaanxi: Mathematics|
|195<br><br>|2021 Shaanxi Xi’an Beilin District Primary School to Junior High School Examination Paper (II) (Hanguang Campus): Mathematics|
|196|2020 Beijing Chaoyang District Primary School to Junior High School Examination Paper: Mathematics|
|197<br><br>|2020 Beijing Haidian District Primary School to Junior High School Examination Paper (Paper A): Mathematics|
|198<br><br>|2020 Beijing Haidian District Primary School to Junior High School Examination Paper (Paper B): Mathematics|
|199<br><br>|2020 Beijing Haidian District Primary School to Junior High School Examination Paper: Mathematics|
|200|2020 Beijing Changping District Primary School to Junior High School Examination Paper: Mathematics|

###### Table 14: The data sources of WE-MATH (Part5, Source 201 to 223).

|Number<br><br>|Data Source|
|---|---|
|201<br><br>|2020 Hunan Changsha Yuhua District Yali Experimental Middle School Primary to Junior High School Mathematics Test Paper|
|202<br><br>|2020 Shenzhen Futian District Shenzhen Senior High School Primary to Junior High School Mathematics Test Paper|
|203<br><br>|2020 Heilongjiang Jixi Hulin City Primary School Entrance Examination Paper: Mathematics|
|204|2020 Heilongjiang Qiqihar Primary School to Junior High School Examination Paper: Mathematics|
|205<br><br>|2020 Ningxia Wuzhong Hongsibao District Primary School Entrance Examination Paper (People’s Education Edition): Mathematics|
|206<br><br>|2020 Liaoning Fushun Wanghua District Primary School to Junior High School Examination Paper: Mathematics|
|207<br><br>|"2020 Primary to Junior High School Examination Paper in Xincheng District, Hohhot, Inner Mongolia: Mathematics|
|208<br><br>|2020 Guangdong Zhaoqing Huaiji County Primary School to Junior High School Examination Paper: Mathematics|
|209|2020 Guangdong Zhaoqing Huaiji County Primary School to Junior High School Examination Paper: Mathematics|
|210|2020 Fujian Quanzhou Licheng District Primary School Entrance Examination Paper: Mathematics|
|211|"2020 Primary to Junior High School Examination Paper in Huimin District, Hohhot, Inner Mongolia: Mathematics|
|212|2020 Hebei Baoding Jingxiu District Primary School to Junior High School Examination Paper: Mathematics|
|213|2020 Sichuan Chengdu Pidu District Primary School to Junior High School Examination Paper (Part 1): Mathematics|
|214<br><br>|2020 Sichuan Mianyang Youxian District Primary School Entrance Examination Paper (Part 6): Mathematics|
|215|2020 Guangdong Shaoguan Zhenjiang District Primary School to Junior High School Examination Paper: Mathematics|
|216<br><br>|2020 Sichuan Mianyang Fucheng District Primary School to Junior High School Examination Paper (Part 10): Mathematics|
|217|2020 Sichuan Mianyang Fucheng District Primary School to Junior High School Examination Paper (Part 5): Mathematics|
|218<br><br>|2020 Sichuan Chengdu Pidu District Primary School to Junior High School Examination Paper (Part 5): Mathematics|
|219<br><br>|2020 Hunan Changsha Yuelu District Primary School to Junior High School Examination Paper: Mathematics|
|220<br><br>|2020 Hunan Changsha Yuelu District Primary School to Junior High School Examination Paper: Mathematics|
|221|2020 Gansu Lanzhou Chengguan District Primary School to Junior High School Examination Paper (Class Examination): Mathematics|
|222|2020 Hunan Changsha Primary School to Junior High School Examination Paper: Mathematics|
|223<br><br>|2020 Liaoning Shenyang Primary School Primary Exam Paper: Mathematics|

