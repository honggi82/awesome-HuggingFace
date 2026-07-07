# arXiv:2403.14624v2[cs.CV]18Aug2024

1

MATHVERSE:

[Figure 1]

## Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?

Renrui Zhang∗‡1,2, Dongzhi Jiang∗1, Yichi Zhang∗2, Haokun Lin2, Ziyu Guo2, Pengshuo Qiu2 Aojun Zhou1, Pan Lu3, Kai-Wei Chang3, Peng Gao†2, Hongsheng Li†1

1CUHK MMLab 2Shanghai Artificial Intelligence Laboratory 3University of California, Los Angeles {zhangrenrui, dzjiang, ziyuguo}@link.cuhk.edu.hk lupantech@gmail.com, gaopeng@pjlab.org.cn, hsli@ee.cuhk.edu.hk

### Abstract

The remarkable progress of Multi-modal Large Language Models (MLLMs) has garnered unparalleled attention, due to their superior performance in visual contexts. However, their capabilities in visual math problem-solving remain insufficiently evaluated and understood. We investigate current benchmarks to incorporate excessive visual content within textual questions, which potentially assist MLLMs in deducing answers without truly interpreting the input diagrams. To this end, we introduce MATHVERSE, an all-around visual math benchmark designed for an equitable and in-depth evaluation of MLLMs. We meticulously collect 2,612 high-quality, multi-subject math problems with diagrams from publicly available sources. Each problem is then transformed by human annotators into six distinct versions, each offering varying degrees of information content in multi-modality, contributing to 15K test samples in total. This approach allows MATHVERSE to comprehensively assess whether and how much MLLMs can truly understand the visual diagrams for mathematical reasoning. In addition, we propose a Chainof-Thought (CoT) evaluation strategy for a fine-grained assessment of the output answers. Rather than naively judging True or False, we employ GPT-4(V) to adaptively extract crucial reasoning steps, and then score each step with detailed error analysis, which can reveal the intermediate CoT reasoning quality by MLLMs. With MATHVERSE, we unveil that, most existing MLLMs struggle to understand math diagrams, relying heavily on textual questions. Surprisingly, some of them even achieve 5%+ higher accuracy without the visual input, e.g., Qwen-VL-Max and InternLM-XComposer2. In contrast, GPT-4V and ShareGPT4V demonstrate relatively better comprehension of the visual content for mathematical reasoning. We hope MATHVERSE may provide unique insights to guide the future development of MLLMs. Project page: https://mathverse-cuhk.github.io.

### 1 Introduction

With the substantial advances of big data and computational power, Large Language Models (LLMs) [4, 28, 55, 56, 13], such as ChatGPT [45] and GPT-4 [46], have emerged as a central point of interest in both industry and academia. To broaden their applicability across diverse contexts, Multi-modal Large Language Models (MLLMs) [66, 20, 52, 11, 61, 70] have recently become

1∗ Equal contribution ‡ Project lead † Corresponding author

Preprint. Under review.

[Figure 2]

[Figure 3]

###### GeoQA MathVista MMMU

Original Questions w/o Text Redundancy w/o Diagram Input

[Figure 4]

[Figure 5]

[Figure 6]

| |
|---|

AverageAccuracy(%)

51.7

43.3

35.8

###### Question:

###### Question:

Question:

33.3

29.2

As shown in the figure, AB is parallel to CD, and a straight line EF

AB is the diameter of ⊙O, C is the point on ⊙O, passing point C is the tangent of ⊙O and intersects the extended line of AB at point E, OD ⊥ AC at point D, if ∠E = 30°, CE = 6.0, the value of OD is ()

The curve y = f(x) and the line y = -3, as shown in the figure, intersect at the points (0, -3), (𝑎, -3), and (𝑏, -3). The sum of the area of the shaded region enclosed by the curve and the line is given by ()

25.0

intersects AB at point E, intersects CD at point F, EG bisects angle BEF, and it intersects CD at point G, angle 1 = 50°, angle 2 is equal to ()

Open-source MLLMs

Closed-source MLLMs

(a) Text Redundancy within Existing Benchmarks (b) Ablation Study

- Figure 1: (a) We showcase three examples of Text Redundancy (highlighted in red) within existing visual math benchmarks [9, 41, 63]. (b) We report an ablation study by respectively removing the redundant texts and input diagrams on 120 randomly selected problems, for closed-sourced [47, 22, 3] and open-sourced [21, 38, 16] MLLMs.

a fast-evolving track, exemplified by the latest GPT-4V [47], Gemini [22], and the open-source LLaVA [39, 34, 32, 30] and SPHINX [36, 21]. Concurrently, a diverse array of evaluation benchmarks [17, 40, 33, 18, 53] are curated to assess their visual comprehension performance across different domains. Notably, the capability to solve mathematical problems involving diagrams serves as a critical measure, offering insights into the multi-modal logical thinking prowess of MLLMs. This task demands MLLMs to accurately decode the visual elements within input diagrams (characters and figures), and correlate them with the condition specified by textual questions for mathematical reasoning. Previous efforts [42, 51], e.g., GeoQA [9, 5] and UniGeo [7], concentrate on the challenging geometric problems, while the recent MathVista [41] and MMMU [63] expand the scope to encompass broader disciplines, including functions, charts, and scientific problems.

However, through our comprehensive observation and analysis, we identify three primary issues in current mathematical benchmarks for evaluating MLLMs:

- i. Do MLLMs truly see the math diagrams in evaluation? This is the most fundamental question concerning the accurate assessment of visual math problem-solving. In Figure 1 (a), we showcase three examples from current benchmarks. We observe their texts contain too much duplicate information (highlighted in red) that is also depicted in the diagram. This redundancy might inadvertently provide MLLMs with a shortcut to resolve the problem by mostly reading the text, rather than interpreting the diagram. Our hypothesis gains support from the experiment in Figure 1 (b). For 40 randomly sampled problems from each benchmark, we remove such redundant texts from the question, challenging MLLMs to capture the corresponding information exclusively from visual inputs. The results reveal a significant drop in accuracy among most MLLMs (the blue column), even falling below the scores without taking diagrams as input (the grey column). This outcome suggests that MLLMs primarily depend on textual cues rather than the visual diagrams themselves to solve these problems in evaluation. Given this, we demonstrate that current visual math benchmarks might not be comprehensive enough to assess the genuine multi-modal mathematical reasoning capabilities of MLLMs.
- ii. Is it equitable to assess solely by the final answer? Most existing multi-modal benchmarks directly compare model outputs with ground truths to derive a binary evaluation result. While this approach may suffice for general visual contexts, it falls short in math problems that require intricate step-by-step reasoning. In Figure 2, we examine three model outputs. Although they all arrive at incorrect answers in the end, they demonstrate varying levels of precision in the intermediate reasoning processes. Merely categorizing these outputs as ‘Incorrect’ fails to capture the nuanced differences in the reasoning quality of MLLMs.

###### Question: For what x does f reach its local maximum? GPT-4V: Gemini-Pro: SPHINX-MoE:

✓

✓

###### Answer: x = 3

The graph shown is the derivative of the function f, denoted by f′(x).

The graph shows two functions: f(x) in blue, and f’(x) in orange …

✓

The graph shows f(x) and its derivative f’(x) …

[Figure 7]

✓

✓

The derivative of a function is zero at the relative extrema of the original function …

To find f(x)’s local maximum, we look at where f'(x) is zero.

×

f(x) is a parabola with a vertical axis of symmetry.

✓

The graph of f'(x) crosses the xaxis at two points, and a local maximum is where the derivative changes from positive to negative …

×

This means that f(x) will have a U-shape, with the vertex at the point (4, 0).

×

Since f′(x) is positive on the interval 2<x<7, we know that f is increasing over that interval …

×

The local maximum of f(x) occurs when x = 4, the point where the graph of f’(x) intersects the x-axis.

×

×

f has a local minimum at x=2 and a local maximum at x=7 …

This occurs just to the left of the y-axis, a little before x=0 …

×

×

f reaches a local maximum at x=7.

It appears to be around x = -1.

✓ Correct × Incorrect

- Figure 2: Comparison of Visual Mathematical Reasoning by Three MLLMs. Despite the incorrect final answer, GPT-4V [47], Gemini-Pro [22], and SPHINX-MoE [21] exhibit different levels of quality in the intermediate reasoning process.

iii. Do they specialize in mathematical reasoning evaluation? GeoQA, UniGeo, and other previous works narrowly target specific aspects of plane geometry. This limits the evaluation of broader mathematical capabilities, e.g., functions and solid geometry. Instead, MathVista expands its scope by including a wide array of peripheral tasks (19 out of 28), encompassing natural images, statistic plots, and charts, which do not directly evaluate professional math skills. Furthermore, the math problems in MMMU are of college-level complexity with extensive domain-specific knowledge, potentially hindering MLLMs from fully demonstrating their reasoning capacity.

Therefore, in light of the issues discussed, we present MATHVERSE, a holistic and specialized visual math benchmark crafted to evaluate the multi-modal mathematical reasoning skills of MLLMs. This benchmark encompasses a meticulously collected dataset of 2,612 visual math problems, with 1,236 newly acquired from public question repositories and 1,376 selected from existing benchmarks, ensuring a diverse range of challenges. To specialize in mathematical reasoning, MATHVERSE spans three primary areas: plane geometry, solid geometry, and functions. Each problem has been rigorously reviewed by expert annotators and classified into twelve detailed categories, emphasizing different fine-grained problem-solving capabilities. Notably, MATHVERSE distinguishes itself by introducing two novel strategies for evaluating MLLMs.

First, we investigate the influence of textual redundancy and validate whether MLLMs can interpret the diagrams for mathematical reasoning. As illustrated in Figure 3 (Left), we categorize the textual content within the questions into three different types: Descriptive Information, Implicit Property, and Essential Condition. These categories, arranged in ascending order of significance for problemsolving, correspond to information directly observable from the diagram, implicit spatial properties that demand advanced visual perception, and specific measurements crucial for computing the solution, respectively. Based on this problem formulation, expert annotators progressively remove the textual information from the questions in MATHVERSE, while incrementally incorporating elements into the visual diagrams to ensure problems are adequately defined. As shown in Figure 3 (Right), this process results in six unique versions of each problem characterized by a reduction in textual content and an enhancement in visual elements, creating a total of 15K test samples. These delicately curated problems can indicate the various multi-modal capabilities of MLLMs, such as geometric element understanding, function curve perception, and numerical value recognition, which thoroughly unveils whether and how much they comprehend the visual diagram for mathematical reasoning.

Second, to rigorously assess the visual Chain-of-Thought (CoT) capabilities [58], we propose a CoT Evaluation strategy for the step-by-step reasoning assessment of MLLMs. For each model’s output, we leverage GPT-4 to first extract several crucial steps exclusively from the solving process, deliberately omitting the input of the question and answer. This approach aims to mitigate the bias towards GPT-4’s inherent question-answering propensities. Then, the corresponding question, diagram, and ground-truth answer are fed into GPT-4 to evaluate each identified critical step, and provide detailed error analysis. Finally, the overall score is obtained by considering every single step within reasoning. Note that, we do not pre-define a ground-truth key-step template, since each

[Figure 8]

###### Plane Geometry: Solid Geometry: Functions:

Three Categories of Texts:

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Descriptive Information

Implicit Property

Essential Condition Question:

Question:

###### Question:

[Figure 15]

📖

[Figure 16]

Decrease Text Content

The graph shows 𝑦 = 𝑥 passing (0,0) and a vertical or horizontal translation 𝑦 passing (2,0). Write an equation for 𝑦 as shown.

Find the surface area of the cylinder shown. The height is 10 cm and the radius is 6 cm. Give your answer to two decimal places.

AB and CD are two diameters of circle O, chord DE parallel AB,

🔍 Increase Vision Content

[Figure 17]

arc DE is the arc of 50°, then angle BOC is (). Choices: …

[Figure 18]

Six Versions of a Problem:

Vision-intensive Vision-dominant Vision-only

Text-dominant Text-lite Text-only

A: 115° Answer: 603.19 cm Answer: 𝑦 = 𝑥 + 2

Answer:

Subject: Angle Subject: Length Subject: Expression

[Figure 19]

[Figure 20]

[Figure 21]

- Figure 3: Three Categories of Question Texts in MATHVERSE. According to the significance for problem-solving, we categorize the question texts into three categories, and transform each problem into six versions for evaluation, with varying content in multi-modality. We present three examples in MATHVERSE for illustration.

math problem may encompass a variety of solution pathways, and different MLLMs tend to exhibit variable reasoning lengths. With CoT scoring, MATHVERSE showcases a fine-grained evaluation of the intermediate logical deduction of MLLMs, demonstrating visual mathematical CoT capabilities.

We conduct extensive experiments on MATHVERSE with popular closed-source [47, 3, 22] and open-source [37, 38, 16, 21] MLLMs. Comparing different problem versions, we unveil that, most existing MLLMs struggle to understand math diagrams, relying heavily on textual questions. Therein, GPT-4V [47] achieves the best overall performance across different problem versions and subjects. Surprisingly, some of the MLLMs even attain much higher results without the diagram input, e.g., +5.1% for Qwen-VL-Max [3] and +5.6% for InternLM-XComposer2 [16]. With the fine-grained error analysis produced by our CoT evaluation strategy, we demonstrate such results are due to their deficient visual encoding capacity for mathematical diagrams, which instead acts as a distraction for problem-solving. In contrast, GPT-4V and ShareGPT4V [12] demonstrate relatively better comprehension of the visual content for mathematical reasoning. Our experimental results suggest that inadequate mathematical visual interpretation capabilities represent the most significant impediment for MLLMs in addressing multi-modal math problems, indicating substantial potential for advancement.

The contributions of this paper are summarized as follows:

- • We investigate primary issues within existing benchmarks and introduce MATHVERSE, an all-around multi-modal benchmark evaluating the visual mathematical reasoning of MLLMs. The meticulously curated dataset contains 20K test problems with diagrams for a comprehensive assessment.
- • By modifying problems with varying information content in multi-modality, we explore whether and how much MLLMs can understand the visual diagrams for mathematical reasoning, rather than relying on question texts.
- • We propose a CoT evaluation strategy with GPT-4 to extract and assess each key step in the reasoning process of MLLMs, which provides a detailed error analysis and fine-grained evaluation of their multi-modal mathematical CoT capabilities.

### 2 MATHVERSE

In Section 2.1, we first present an overview of the curated visual math dataset in MATHVERSE. Then, in Section 2.2, we introduce our data formulation approach for investigating the visual mathematical comprehension of Multi-modal Large Language Models (MLLMs). Finally, in Section 2.3, we elaborate on the methodology of our proposed Chain-of-Thought (CoT) evaluation strategy.

##### Figure 4: Subject Distribution of MATHVERSE. Solid G: Solid Geometry, Plane G: Plane Geometry.

#### Table 1: Key Statistics of MATHVERSE.

Statistic Number Total questions 2,612

- - Multiple-choice questions 1,631 (62.4%)
- - Free-form questions 981 (37.6%)
- - Newly collected questions 1,236 (47.3%)
- - Existing-dataset questions 1,376 (52.7%)
- - Questions with explanations 1,236 (47.3%) Total test samples 15,672

- - Newly annotated samples 10,448 (66.7%)
- - Samples of each version 2,612 (16.7%)

Analytic

Property Applied

Area

Expression Coordinate

Applied

Functions 20.5%

Volume Area Length

Plane G 66.8%

Solid G 12.7%

Number of unique images 2,420 (92.6%) Number of unique questions 2,573 (98.5%) Number of unique answers 847 (32.4%)

Length

Maximum question length 203 Maximum answer length 17 Average question length 35.7 Average answer length 1.4

Angle

er num

#### 2.1 Visual Math Dataset

To thoroughly assess visual mathematical proficiency, we compile a comprehensive problem set covering a broad spectrum of math subjects, diagram patterns, and specialized knowledge domains. This widespread collection for MATHVERSE aims to pose diverse challenges to MLLMs, ensuring a robust evaluation of their capabilities in visual contexts.

Data Composition and Categorization. MATHVERSE comprises a total of 2,612 visual math problems, which contribute to the final created 15K test samples. Detailed statistics for data composition are presented in Table 1. This meticulously collected dataset covers three fundamental math subjects, i.e., plane geometry (1,746), solid geometry (332), and functions (534), where the latter two are all composed of newly collected problems. The choice of these three subjects is not only due to their rigorous demands on multi-modal reasoning, but also for two other considerations. For one thing, as we specialize MATHVERSE in mathematical problem-solving, other peripheral tasks in MathVista [41] are not included, e.g., statistical reasoning, table question-answering, and puzzle tests. For another, we expect the evaluation to fully display the reasoning capabilities of MLLMs with moderate-level mathematical knowledge. This avoids limiting their performance with overly complex domain-specific theorems or prior commonsense knowledge. Therefore, we deliberately focus the collected problems on the high school level, excluding advanced college-level disciplines like calculus and graph theory featured in MMMU [63]. Furthermore, expert annotators subdivide the problems into twelve fine-grained categories, as depicted in Figure 4, showcasing various dimensions of visual mathematical skills.

Data Collection and Review Process. Our collection procedure for high-quality visual math problems involves a rigorous selection from both pre-existing datasets and public question repositories. In the domain of plane geometry, we initially select 750 problems from GeoQA [9], 119 from GEOS [51], and 507 from Geometry3K [42], based on their original data quality and distribution. We exclude questions that are extremely simple or excessively complex, as well as those that appear dubious or lack necessary conditions. To enhance the diversity of question types and diagram styles, we further enrich our dataset with additional 370 plane geometry problems by manually collecting from other sources1,2,3. Given the scarcity of solid geometry and function-related problems in existing benchmarks, we purposefully gather these two types of problems (332 and 534, respectively) from new sources1,2,3 to address this gap. Problems that include multiple diagrams or require visual illustrations within solutions are excluded, considering the current limitations of MLLMs in resolving such information. Note that, all the newly collected problems (1,236) accompany detailed

- 1https://homework.study.com
- 2https://www.ixl.com/math
- 3https://mathspace.co/us

[Figure 22]

Descriptive Information Implicit Property Essential Condition

Text Dominant Text Lite Text Only Vision Intensive Vision Dominant Vision Only

[Figure 23]

AB and CD are two diameters of circle O, chord DE parallel AB,

AB and CD are two diameters of circle O, chord DE parallel AB,

Text Input

📖

[Figure 24]

Chord DE parallel AB,

Chord DE parallel AB, then angle BOC is (). Choices: …

Arc DE is the arc of 50°, then angle BOC is (). Choices: …

arc DE is the arc of 50°, then angle BOC is (). Choices: …

arc DE is the arc of 50°, then angle BOC is (). Choices: …

arc DE is the arc of 50°, then angle BOC is (). Choices: …

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Chord DE parallel AB, then angle BOC is (). Choices: …

Vision Input

[Figure 29]

50°

🔍

[Figure 30]

50°

Text Dominant Text Lite Vision Dominant Text Dominant Text Lite Vision Dominant

[Figure 31]

[Figure 32]

The graph shows 𝑦 = 𝑥 passing (0,0) and a vertical or horizontal translation 𝑦 passing (2,0). Write an equation for 𝑦 as shown.

The graph shows 𝑦 = 𝑥 and a vertical or horizontal translation 𝑦 . Write an equation for 𝑦 as shown.

The graph shows 𝑦 and a vertical or horizontal translation 𝑦 . Write an equation for 𝑦 as shown.

Find the surface area of the cylinder shown. The height is 10 cm and the radius is 6 cm. Give your answer to two decimal places.

Find the surface area of the cylinder shown. and the radius is 6 cm. Give your answer to two decimal places.

Find the surface area of the cylinder shown. Give your answer to two decimal places.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

6 cm

y y = x

- Figure 5: Six Versions of Each Problem in MATHVERSE. Expert annotators meticulously transform each visual math problem within MATHVERSE into six versions. They contain different visionlanguage content for a holistic visual mathematical evaluation.

explanations. After the preliminary collection, we undertake a comprehensive review to verify the accuracy of the answers, ensure consistency between questions and diagrams, and confirm the relevance of each problem to the defined twelve categories. This meticulous review guarantees the dataset’s quality and precision.

#### 2.2 Whether MLLMs Truly See the Diagrams?

In this section, we detail our data formulation approach to transform each problem in MATHVERSE into six different versions with varying information content in multi-modality. In this way, we explore the visual diagram understanding capabilities of MLLMs for mathematical reasoning.

Three Types of Textual Information. Considering the textual redundancy in original math problems, we first define three distinct categories for the textual information within the questions, as illustrated in Figure 3 and the following:

- • Descriptive Information (DI) refers to the directly observable and clearly portrayed content in the diagram. It depicts the basic figure composition, spatial arrangement, and annotated entities, such as the presence of geometric shapes or intersection points of functions. These sentences normally help establish the context and frame the problem to orient the solver. Nevertheless, such information is repetitive to the visual components present in the diagram, thus regarded as redundant information for problem-solving. More importantly, it may assist MLLMs in bypassing the process of diagram interpretation, thereby undermining the assessment for visual mathematical reasoning, as evidenced in Figure 1.
- • Implicit Property (IP) involves the information that requires a higher level of visual perception but less mathematical knowledge to discern from the diagram. It signifies strong visual conditions for problem-solving, such as the parallelism and perpendicularity between lines, the similarity and congruence among triangles, and the category and periodicity of functions. They can, in theory, be fully extracted from the diagrams alone, giving adequate capability for visual recognition and comprehension of MLLMs.

- • Essential Condition (EC) denotes the specific numerical or algebraic measurements, which are indispensable conditions to derive the solution and cannot be derived from the visual diagram. This category encompasses precise values of angles, lengths, and function expressions, such as an angle being 45 degrees, the length of BC being 6 units, and the functional equation f(x) = x2 + 3. Without these details in textual information, solving the visual math problem would be impossible.

Creating Six Versions of Each Problem. Based on the three categories, expert annotators systematically remove different textual information within questions, and incrementally incorporate the critical elements into diagrams. This approach can progressively reduce textual redundancy and information content, thereby increasingly compelling MLLMs to capture mathematical conditions from the visual input. As compared in Figure 5, we generate six versions of each problem in MATHVERSE, obtaining 15,672 test instances. With this curated problem set, we can provide a holistic evaluation of the genuine visual comprehension of MLLMs, and whether it can facilitate multi-modal mathematical reasoning. The details of each problem version are as follows:

- • Text-dominant Version retains the entire textual content, including the three types of textual information and the question statement. If the original problem contains limited Descriptive Information, we manually add it within the textual content. This version may induce MLLMs to regard the text as the primary source of information, treating the diagram more as a supplementary visual aid. This serves as the baseline point for evaluation.

[Figure 39]

Text: DI + IP + EC + Question Vision: Diagram (1)

[Figure 40]

- • Text-lite Version diminishes the Descriptive Information from the Text-dominant version, assuming this information can be observed from the diagram. This creates a condensed question without redundancy, forcing MLLMs to interpret the diagram for basic information.

[Figure 41]

Text: IP + EC + Question Vision: Diagram (2)

[Figure 42]

- • Text-only Version directly discards the diagram input from the Text-dominant version. Comparing this to the Text-lite version helps identify where MLLMs mainly obtain the contextual visual information for problem-solving, the Descriptive Information or diagram.

[Figure 43]

Text: DI + IP + EC + Question Vision: ∅ (3)

[Figure 44]

- • Vision-intensive Version further removes the Implicit Property from the Text-lite version. Without the strong visual condition in texts, MLLMs are challenged to intensively leverage their visual interpretation skills to gather sufficient cues for mathematical reasoning. The outcome demonstrates their proficiency in understanding mathematical relationships visually.

[Figure 45]

Text: EC + Question Vision: Diagram (4)

[Figure 46]

- • Vision-dominant Version, building upon the Text-lite version, excludes the Essential Condition from texts, instead annotating these measurements visually in diagrams. The textual content is narrowed down to Implicit Property and question statements. It demands MLLMs to recognize the Essential Condition exclusively from diagrams, and accurately correlate it with corresponding visual elements for problem-solving.

[Figure 47]

Text: IP + Question Vision: Diagram + EC (5)

[Figure 48]

- • Vision-only Version strips away the entire textual input, conveying the problem solely through the diagram. We render the remaining textual content in the Vision-dominant version onto the diagram, and minimize the text to an almost negligible prompt. This compels MLLMs to extract figures and recognize questions both from the visual information, standing as an ultimate test for their visual reasoning capabilities in mathematics.

[Figure 49]

[Figure 50]

Text: ∅ Vision: Diagram + EC + IP + Question (6)

[Figure 51]

Model Output

Multi-step Scoring

The line passes through the points (-3, 0) and (0, -6). We can use these points to find the equation of the line. Let‘s use the slopeintercept form of a linear equation: y = mx + b, where m is the slope and b is the y-intercept. First, we’ll find the slope (m) …

✓ 1/1

- 1. The model correctly states that the slope-intercept form of a linear equation is y = mx + b.
- 2. The model correctly calculates the slope (m) using the given points (-3, 0) and (0, -6).
- 3. The model correctly finds the y-intercept (b) using one of the points and the slope-intercept form of the equation.
- 4. The equation of the line is correctly stated as y = -2x - 6.
- 5. The model incorrectly states that the region below the line is shaded. However, based on the diagram and the ground truth answer, the region above the line is shaded.
- 6. The final inequality provided by the model is y ≤ -2x - 6, which does not match the ground truth answer of y ≥ -2x - 6.

✓ 1/1 ✓ 1/1 ✓ 1/1

Key-step Extraction

###### Question:

- 1. Use the slope-intercept form of a linear equation: y = mx + b.
- 2. Find the slope (m) using the formula: m = (y2 - y1) / (x2 - x1). Using the points (-3, 0) and (0, -6): m = (-6 - 0) / (0 - (-3)), m = -2.
- 3. Find the y-intercept (b) using one of the points and the slope-intercept form of the equation: 0 = -2 * (-3) + b, b = -6.
- 4. Since the line is solid, the inequality is either y ≤ -2x - 6 or y ≥ -2x - 6.
- 5. The region below the line is shaded, which means the values of y should be less than or equal to -2x - 6.
- 6. The inequality describing the region drawn in the plane is: y ≤ -2x – 6.

The x-intercept of the line is -3 and the y-intercept is -6. State the inequality that describes the region drawn in the plane.

× 0/1

× 0/1

Answer: y ≥ -2x - 6. Subject: Expression Version: Text Dominant

Multi-step Score: (1 + 1 + 1 + 1 + 0 + 0) / 6 = 2/3 ≈ 0.67 Final CoT Evaluation Score: 0.67 × 0.7 + 0 × 0.3 = 0.47

[Figure 52]

Model Output

[Figure 53]

Multi-step Scoring

Since B C and E D are parallel, they form a proportion. Therefore, we can set up the following equation: A B / B C = C D / D E. Substituting the given values, we get: 16 / (x - 3) = (x + 5) / 20. Cross-multiplying, we get: 16(x + 5) = 20(x - 3). Expanding and simplifying, we get: 16x + 80 = 20x – 60, 4x = 140, x = 35. …

✓ 1/1

- 1. The model correctly identifies that BC and ED are parallel.
- 2. The equation AB / BC = CD / DE is incorrect because it doesn't reflect the correct sides according to similar triangles.
- 3. The calculation of the equation is correctly performed, but since the equation itself is incorrect, this step is also incorrect.
- 4. The model correctly solves for x based on the incorrect equation. Since the equation is wrong, this step is also incorrect.
- 5. The model substitutes x into CD = x + 5 to find CD, which would be correct if the value of x was correct. Since the value of x is based on an incorrect equation, this step is also incorrect.
- 6. The final answer provided by the model is D: 40, which matches the ground truth answer. Despite the incorrect steps, the final answer is coincidentally correct.

###### × ×

0/1

0/1

Key-step Extraction

Question:

###### ×

0/1

BC and ED are parallel. Find CD if AC = x - 3, BE = 20, AB = 16, and CD = x + 5. Choices: A:32 B:35 C:36 D:40

- 1. Identify that BC and ED are parallel, forming a proportion equation.
- 2. Set up the equation: AB / BC = CD / DE.
- 3. Substitute the given values into it: 16 / (x - 3) = (x + 5) / 20. Expand and simplify the equation to get: 16x + 80 = 20x – 60.
- 4. Solve for x: 4x = 140, x = 35.
- 5. Substitute x into CD = x + 5 to find CD: CD = 35 + 5 = 40.
- 6. The final answer is D: 40.

###### ×

0/1

✓

1/1

Answer: D: 40 Subject:

[Figure 54]

Length Version: Text Lite

Multi-step Score: (1 + 0 + 0 + 0 + 0 + 1) / 6 = 1/3 ≈ 0.33 Final CoT Evaluation Score: 0.33 × 0.7 + 1 × 0.3 = 0.53

- Figure 6: Examples of the CoT Evaluation Strategy for MATHVERSE. We present two outputs from Qwen-VL-Max [3] with our CoT evaluation strategy, which assesses the fine-grained reasoning capabilities with a detailed explanation for error analysis.

#### 2.3 CoT Evaluation Strategy

Compared to visual question-answering in general scenarios, the solving process of MLLMs for mathematical problems requires nuanced, step-by-step CoT reasoning. Considering two cases in Figure 6, one arrives at the correct solution albeit through incorrect intermediary steps, while the other demonstrates the opposite phenomenon. Therefore, the binary ‘Correct’ or ‘Incorrect’ evaluative approach of existing benchmarks is inadequate to examine the depth and precision of the multi-step reasoning process. To this end, we propose a CoT evaluation strategy to thoroughly assess their mathematical CoT skills in visual contexts, involving two prompting phases with GPT-4(V) [47, 46].

Key-step Extraction. Given the output of an MLLM, we first employ GPT-4, the languageonly version, to extract N pivotal steps within the reasoning sequence, denoted as [s1,s2,...,sN], including the final answer sA. Such key steps include significant computational outcomes, the identification of visual components, and critical immediate inferences. Note that, we only prompt GPT-4 with the MLLM’s output, deliberately omitting the original questions, diagrams, and groundtruth answers. This approach aims to mitigate the inherent bias of GPT-4 itself towards problemsolving and visual diagram interpretation, thereby concentrating solely on the logical coherence of the model output. In addition, we do not pre-define a ground-truth key-step template for each problem, but perform the extraction adaptively for the unique output of every MLLM. Since the problem potentially encompasses diverse possible solution pathways, and different MLLMs exhibit varying reasoning lengths and styles, the rigid template would harm the CoT evaluation accuracy.

Multi-step Scoring. After the extraction phase, we utilize GPT-4V, the multi-modal version, to evaluate each critical step and culminate a comprehensive score. We feed the extracted key steps, the original questions, diagrams, and ground-truth answers all into GPT-4V, contributing to a holistic assessment, e.g., numerical computations, logical deductions, and visual interpretations. Therein, we observe that GPT-4V occasionally struggles with accurately recognizing elements within functional diagrams, leading to unstable evaluation for related problems. We thereby annotate additional

- Table 2: Mathematical Evaluation on Six Problem Versions in MATHVERSE’s testmini Set. We calculate the ‘All’ score without averaging the ‘Text Only’ version. ‘CoT-E’ or ‘Acc’ denotes whether to employ the proposed CoT evaluation strategy or not. The highest accuracy for closed-source and

open-source MLLMs is marked in red and blue respectively.

Text Dominant

Text Lite

Text Only

Vision Intensive

Vision Dominant

Vision Only

All

Model

CoT-E Acc CoT-E Acc CoT-E Acc CoT-E Acc CoT-E Acc CoT-E Acc CoT-E Acc Baselines

Random Chance - 12.4 - 12.4 - 12.4 - 12.4 - 12.4 - 12.4 - 12.4 Human - 64.9 - 71.2 - 70.9 - 41.7 - 61.4 - 68.3 - 66.7

LLMs

ChatGPT [48] - - 51.3 33.3 38.5 18.9 51.3 33.3 - - - - - GPT-4 [46] - - 63.4 46.5 40.7 20.7 63.4 46.5 - - - - - -

Closed-source MLLMs

Qwen-VL-Plus [3] 21.3 11.8 26.0 15.7 21.2 11.1 25.2 14.5 18.5 9.0 19.1 13.0 21.8 10.0 Gemini-Pro [22] 35.3 23.5 39.8 26.3 34.7 23.5 44.5 27.3 32.0 23.0 36.8 22.3 33.3 22.2 Qwen-VL-Max [3] 37.2 25.3 42.8 30.7 37.7 26.1 47.9 28.9 33.6 24.1 35.9 24.1 35.9 21.4 GPT-4V [47] 54.4 39.4 63.1 54.7 56.6 41.4 60.3 48.7 51.4 34.9 50.8 34.4 50.3 31.6

Open-source MLLMs

LLaMA-Adapter V2 [20] 5.8 5.7 7.8 6.2 6.3 5.9 3.9 2.7 6.2 6.1 4.5 4.2 4.4 6.1 ImageBind-LLM [24] 10.0 9.2 13.2 11.4 11.6 11.3 12.9 11.7 9.8 8.9 11.8 11.2 3.5 3.4 mPLUG-Owl2 [62] 10.3 5.9 11.6 6.6 11.4 6.3 13.8 6.1 11.1 6.3 9.4 5.6 8.0 4.9 MiniGPT-v2 [11] 10.9 11.0 13.2 12.1 12.7 12.0 15.3 11.7 11.1 13.1 11.3 10.3 6.4 7.4 LLaVA-1.5 [37] 12.7 7.6 17.1 8.8 12.0 7.6 22.6 11.5 12.6 7.4 12.7 7.4 9.0 6.9 SPHINX-Plus [21] 14.0 12.2 16.3 13.9 12.8 11.6 15.8 14.9 12.9 11.6 14.7 13.5 13.2 10.4

G-LLaVA [19] 15.7 16.6 22.2 20.9 20.4 20.7 21.6 21.1 16.5 17.2 12.7 14.6 6.6 9.4 LLaVA-NeXT [38] 17.2 15.6 21.6 19.4 19.7 15.2 25.1 18.1 17.6 16.8 14.9 15.2 12.1 11.3 ShareGPT4V [12] 17.4 13.1 21.8 16.2 20.6 16.2 14.6 6.6 18.6 15.5 16.2 13.8 9.7 3.7 SPHINX-MoE [21] 22.8 15.0 33.3 22.2 21.9 16.4 40.7 18.3 21.1 14.8 19.6 12.6 18.3 9.1 InternLM-XC2. [16] 25.9 16.5 36.9 22.3 28.3 17.0 42.5 16.5 20.1 15.7 24.4 16.4 19.8 11.0

information for function problems and together feed into GPT-4V, ensuring the quality of visual evaluation. Specifically, GPT-4V assesses each N intermediate step with a binary score of ‘1’ (correct) or ‘0’ (incorrect), and derives the overall score by aggregating the correctness of the final answer. We formulate the scoring process as

N

1 N

Scorefinal = α

Score(si) + (1 − α)Score(sA), (7) where α denotes a balancing factor between the intermediate steps and the final answer sA. We set α

i=1

- as 0.7 by default to underscore the significance of CoT reasoning. As exemplified in Figure 6, besides the fine-grained scoring, the CoT evaluation can also provide a detailed error analysis of each step, which is valuable and instructive for the development of MLLMs in the field.

### 3 Experiments

In this section, we conduct a systematic evaluation of existing Multi-modal Large Language Models (MLLMs) on MATHVERSE. We first introduce the experimental setup in Section 3.1. Then, we detail the quantitative results in Section 3.2 and narrate the error analysis in Section 3.3.

#### 3.1 Experimental Setup

Division of the testmini Subset. MATHVERSE encompasses a comprehensive collection of 2,612 visual math problems, alongside 15,672 corresponding test instances. To enable faster evaluation and model development validation, we extract a smaller subset termed testmini including 788 problems and 4,728 instances. In constructing testmini, we employ a random sampling strategy across different subfields, maintaining a sample size proportional to the overall dataset to preserve its statistical representativeness. The remaining test set features 1,824 problems and 10,944 samples will be utilized for standard evaluation and publicly released in the future. In the subsequent experiments, all quantitative results are assessed using the testmini subset of MATHVERSE.

Evaluation Models. We examine the performance of foundation models across three distinct categories on MATHVERSE: (a) Large Language Models (LLMs) as the text-only baseline, which only take textual questions as input, including ChatGPT [45] and GPT-4 [46], (b) Closed-source

- Table 3: Mathematical Evaluation on Different Subjects and Subfields in MATHVERSE’s testmini Set. We report the scores averaging five problem versions except for the ‘Text Only’ version, and employ the CoT evaluation strategy by default. Len: Length; Anal: Analytic; Apply: Applied; Vol: Volume; Coord: Coordinate; Prop: Property; Exp: Expression; Apply: Applied. The highest accuracy for closed-source and open-source MLLMs is marked in red and blue respectively.

Plane Geometry Solid Geometry Functions All Len Area Angle Anal Apply All Len Area Vol All Coord Prop Exp Apply

Model All

Closed-source MLLMs

Qwen-VL-Plus [3] 21.3 17.3 19.1 16.4 16.1 23.6 13.2 24.8 18.1 18.7 33.4 31.3 52.5 25.1 10.8 50.3 Gemini-Pro [22] 35.3 33.0 32.2 42.6 28.4 30.2 32.3 33.4 35.0 29.3 36.1 28.3 25.7 26.6 10.8 51.3 Qwen-VL-Max [3] 37.2 38.4 41.7 46.4 32.6 40.6 38.7 33.7 25.4 28.3 42.6 38.4 43.7 35.5 13.6 61.0 GPT-4V [47] 54.4 56.9 60.8 63.4 52.6 48.5 60.9 50.2 54.8 39.9 56.8 52.8 72.3 47.1 30.9 70.1

Open-source MLLMs LLaMA-Adapter V2 [20] 5.8 5.9 4.0 5.9 6.6 13.4 3.3 4.6 5.3 3.1 5.7 6.2 6.7 6.1 4.5 7.9 ImageBind-LLM [24] 10.0 9.7 12.1 9.9 9.2 10.2 4.8 4.6 4.9 3.5 5.3 14.9 12.3 13.8 4.6 25.9 mPLUG-Owl2 [62] 10.3 7.7 8.2 6.0 5.7 12.4 10.6 11.0 9.2 6.7 15.7 17.4 22.8 18.6 5.3 22.2 MiniGPT-v2 [11] 10.9 11.6 10.0 9.8 14.3 9.1 11.8 1.7 2.2 1.6 0.5 11.2 4.2 15.7 4.0 21.1 LLaVA-1.5 [37] 12.7 11.8 13.1 15.1 9.7 9.4 13.2 10.6 12.1 8.7 11.6 14.8 18.8 12.7 9.5 23.7 SPHINX-Plus [21] 14.0 14.4 14.2 10.5 14.1 16.5 16.8 7.0 7.2 6.1 7.6 17.9 11.1 19.1 6.3 27.7 G-LLaVA [19] 15.7 20.2 17.3 13.6 26.5 5.9 23.1 5.0 10.3 4.4 3.1 9.2 9.1 9.1 1.3 15.5 LLaVA-NeXT [38] 17.2 15.9 14.8 13.1 16.3 17.7 17.8 19.6 33.3 11.7 12.6 23.1 24.5 23.4 8.0 33.1 ShareGPT4V [12] 17.4 16.9 16.2 17.9 16.9 12.2 21.1 15.0 13.6 10.9 19.7 20.2 19.9 22.2 8.4 25.8 SPHINX-MoE [21] 22.8 24.5 26.3 28.4 21.1 26.6 24.4 15.8 9.4 10.7 26.3 19.5 23.5 19.3 9.2 30.3 InternLM-XC2. [16] 25.9 26.2 27.1 29.7 20.6 18.5 22.2 20.1 34.5 14.1 25.2 23.7 24.4 24.9 10.6 36.3

MLLMs, represented by models like GPT-4V [47], Gemini-Pro [22], Qwen-VL-Max [3], and QwenVL-Plus, and (c) Open-source MLLMs, featuring models such as LLaVA-1.5 [37] (Vicuna-13B [13]), LLaVA-NeXT [38] (Vicuna-13B), SPHINX-MoE [21] (Mixtral-8×7B [28]), SPHINX-Plus (LLaMA213B [56]), InternLM-XComposer2 [16] (InternLM2-7B [54]), LLaMA-Adapter V2 [20] (LLaMA7B [55]), ImageBind-LLM [24] (LLaMA-7B), MiniGPT-v2 [11] (LLaMA2-7B), mPLUG-Owl2 [62] (LLaMA-7B), G-LLaVA [19] (LLaMA2-7B), and ShareGPT-4V [12] (Vicuna-13B).

Implementation Details. All our experiments are conducted under a zero-shot setting, showcasing the generalization capacity of MLLMs for mathematical reasoning, without few-shot prompting or further fine-tuning. By default, we employ the Chain-of-Thought (CoT) prompting technique [58], which encourages MLLMs to perform complete reasoning steps for a fine-grained evaluation. A baseline representing random chance is established for comparison, for which we select one option

- at random for multiple-choice questions and utilize empty for free-form questions. In addition, we recruit ten qualified college students, and ask them to solve the problems in MATHVERSE independently, serving as a baseline for human performance. We conduct all experiments on NVIDIA A100 GPUs. As the text-only LLMs can only take text questions as input, we evaluate them with the first three problem versions, i.e., Text Dominant, Text Lite, and Text Only. For the ‘w/o’ results, we utilize the template in MathVista [41] to prompt GPT-4 [46] for answer extraction, and directly score the final answer without the intermediate reasoning process.

#### 3.2 Experimental Analysis

To best investigate the visual mathematical reasoning capabilities, we report the evaluation results of different models on MATHVERSE for the six transformed problem versions in Table 2 and twelve detailed subjects in Table 3. We mainly analyze the performance by the proposed Chain-of-Though (CoT) evaluation, and derive the following observations.

MLLMs Rely More on DI than Seeing Diagrams. Comparing the Text-dominant and Text-only versions, with the elimination of visual input, most MLLMs even obtain an unexpected performance improvement, e.g., +5.1% for Qwen-VL-Max and +5.6% for InternLM-XComposer2. This suggests that the unsatisfactory visual encoding for mathematical diagrams instead severely harms the original problem-solving capacity of MLLMs. As exemplified in Figure 7, from the error analysis of our CoT evaluation strategy, we observe that Gemini-Pro can deduce the correct answer exclusively by the visual information within the Descriptive Information. Instead, the inaccurate visual perception of mathematical elements directly interferes with the outcome of problem-solving, turning correct answers into incorrect ones. In contrast, GPT-4V and ShareGPT-4V achieve better results in Text Dominant than in Text Only, indicating their relatively better visual encoding, which would not degrade the performance. However, they still encounter a larger performance drop by removing the

[Figure 55]

Text Dominant

Text Only

- 1. Correct, the horizontal line test is a valid method to determine if a function has an inverse.
- 2. Correct, based on the provided information, it appears that no horizontal line would intersect the function more than once.
- 3. Correct, following the logic of steps 1 and 2, the conclusion is valid.
- 4. Correct, the answer A corresponds to the conclusion that the function does have an inverse.

- 1. This step is correct as the model correctly states the definition of a one-to-one function, which is necessary to determine if a function has an inverse.
- 2. Analyze the function graph. This step is incorrect. The model_output incorrectly states that the value 0 on the y-axis corresponds to both the value 4 and the value -4 on the x-axis, which is not supported by the image.
- 3. Conclude that the function does not have an inverse function because it is not one-to-one. This conclusion is based on the incorrect analysis in step 2. Since step 2 is incorrect, this conclusion is also incorrect.
- 4. The final answer provided by the model_output is incorrect because the function appears to be one-to-one based on the image.

✓ 1/1

✓ 1/1

Visual Perception Error

[Figure 56]

✓ 1/1

× 0/1

Question:

✓ 1/1 ✓ 1/1

× 0/1

For the curve graphed below, whose asymptote is x-axis and x=4, determine if it has an inverse function. Choices: A:Yes B:No

× 0/1

Multi-step Score: (1 + 0 + 0 + 0) / 4 = 0.25 Final CoT Evaluation Score: 0.25 × 0.7 + 0 × 0 = 0.175

Multi-step Score: 1 Final CoT Evaluation Score: 1

Answer: A: Yes Subject: Property

[Figure 57]

- Figure 7: A Typical Visual Perception Error by our CoT Evaluation Strategy. The example is an output from Gemini-Pro [22], where the correct reasoning of the Text-only version is distracted by the visual perception error within the diagram.

redundant Descriptive Information than the diagram input, e.g., GPT-4V and ShareGPT-4V. This pattern demonstrates that they tend to capture more visual information for mathematical reasoning from the text content, instead of seeing the diagram itself.

MLLMs are Moderately Effective at Perceiving IP. By discarding the Implicit Property in question texts, a negligible decline in accuracy is noted from the Text-lite to Vision-intensive versions for most MLLMs. This is because the Implicit Property mainly encompasses the spatial layouts and geometric relationships, which demand minimal mathematical domain knowledge for interpretation. This outcome underscores the favorable visual perception skills of MLLMs for non-mathematical elements, which is not the primary obstacle hindering MLLMs in solving visual math problems.

MLLMs are Challenged to interpret EC from Diagrams. Incorporating the Essential Condition within diagrams challenges MLLMs to accurately identify and understand these conditions in vision modality for mathematical problem-solving. Evidence from the Vision-dominant results indicates a notable decline in the performance of most MLLMs compared to the Text-lite accuracy, such as -5.8% for GPT-4V and -3.9% for InterLM-XComposer2. This reveals their inaccurate identification of mathematical symbols and an insufficient grasp of domain-specific knowledge required to associate identified measurements with relevant concepts.

MLLMs struggle to Solve Problems Entirely by Diagrams. The scenario of Vision-only problems aligns more closely with real-world applications, where capturing an image is often more convenient than transcribing the problem into text. However, by rendering the whole question within the diagram, the mathematical problem-solving capacity of MLLMs is further diminished. This experiment unveils the great challenge for MLLMs to simultaneously understand mathematical conditions, questions, and figures from the visual input alone.

Closed-source MLLMs are Better-performed. From the performance in both tables, we observe a consistently better performance achieved by closed-source MLLMs than open-sourced ones. Despite the gap with humans, GPT-4V attains the leading position among MLLMs, showcasing superior mathematical capabilities over problem versions and subjects, especially the challenging subfields like ‘Coord’ and ‘Prop’ (the property and coordinate solving of function problems). InternLMXComposer2 and SPHINX-MoE are the best-performing open-source MLLMs, while still lagging behind Gemini-Pro with a margin of 9.4% and 12.5% overall accuracy, respectively, suggesting large improvement space.

LLMs Achieve Competitive Results to MLLMs. Utilizing solely question texts as input, two LLMs, i.e., GPT-4 and ChatGPT, attain superior accuracy to most MLLMs in Text Dominant and Lite versions. Even in the absence of redundant Descriptive Information within Text-lite problems, GPT-4 outperforms InternLM-XComposer2 and SPHINX-MoE by substantial margins of 12.4% and 18.8%, respectively. These findings not only indicate the strong mathematical reasoning skills of LLMs, but

Results with and without CoT Evaluation

|w/o CoT evaluation<br><br>Increase by CoT evaluation<br><br>Decrease by CoT evaluation|
|---|

60

40

20

0

Gemini-Pro Qwen-VL-Max GPT-4V LLaMA-Adapter V2ImageBind-LLMmPLUG-Owl2 MiniGPT-v2 LLaVA-1.5 SPHNIX-Plus G-LLaVA LLaVA-NeXT ShareGPT4V SPHINX-MoE InternLM-XC2

Qwen-VL-Plus

- Figure 8: Results with and without CoT Evaluation in MATHVERSE. Referring to Table 2, we denote the ‘w/o’ results in blue pillars, and highlight the increase and decrease magnitude with ‘CoT-E’ by green and red colors, respectively.

further emphasize the deficiencies in diagram interpretation of existing MLLMs. Importantly, the performance of GPT-4 is only exceeded by GPT-4V, which demonstrates that a satisfactory diagram perception capability can enhance problem-solving for visual mathematics.

GPT-4(V) Beats Human in the Text-only Version. Without the visual content provided in diagrams, human solvers often face challenges in deducing the correct answers due to the lack of sufficient information, e.g., 41.7% ‘w/o’ scores in Text-only problems. In contrast, GPT-4V and GPT-

- 4 achieve the ‘w/o’ scores of 41.1% and 46.5%, respectively, which surpass the human performance. This comparison highlights their advanced reasoning capabilities in handling extreme scenarios, exhibiting more robustness for mathematical problem-solving given missing visual conditions.

Mathematical Training Benefits the Performance. In addition to foundational visual instructionfollowing datasets, both SPHINX-MoE and InternLM-XComposer2 extend their training regimes to include specialized mathematical problems that are either text-only or visual, such as MathQA [2], Geometry3K [43], and MathInstruct [64]. This approach of math-specific tuning contributes to their leading performance in MATHVERSE. Furthermore, G-LLaVA fine-tunes LLaVA-1.5 by a large-scale visual geometric dataset containing 170K enriched problems. This targeted refinement can improve several fields (‘Len’, ‘Angle’, and ‘Apply’) within the plane geometry subject. However, since G-LLaVA’s fine-tuning data does not include problems of analytic geometry, solid geometry, and functions, it harms the related results of LLaVA-1.5 due to catastrophic forgetting, e.g., -3.5% in ‘Anal’, -5.6% in ‘Solid Geometry’, and -5.6% in ‘Functions’. This phenomenon underscores the critical role of developing extensive, high-quality visual math data for effectively training MLLMs.

Discrepancy Between ‘CoT-E’ and ‘w/o’ Scores. As illustrated by Table 2, the ‘CoT-E’ scores for MLLMs, in most cases, are much higher than ‘w/o’ scores, e.g., +16.1% for GPT-4V and +9.6% for InternLM-XComposer2. This observation demonstrates that our proposed CoT evaluation strategy identifies numerous correct intermediate reasoning steps, despite the final incorrect answer, highlighting the effectiveness of fine-grained assessment. In Figure 8, we present the statistics of variance between ‘CoT-E’ and ‘w/o’ scores within different MLLMs. Although GPT-4V attains top-tier performance, it exhibits a pronounced gap of 16.1% concerning the evaluation of CoT reasoning quality, similar to the 12.4% gap of Qwen-VL-Max. Conversely, SPHINX-MoE showcases favorable precision among open-source MLLMs, while preserving a relatively lower variance of two evaluation methods, i.e., 6.0% compared to InternLM-XComposer’s 9.6%. This indicates its consistent step-by-step reasoning throughout the problem-solving process.

#### 3.3 Error Analysis

To delve into the fine-grained predictions, we select the best-performing MLLM, GPT-4V [47], to understand its modes of success and failure. Our proposed CoT evaluation strategy has produced a detailed assessment of model output, including step-wise scores and explanation, reducing extensive manual effort in identifying and analyzing errors. We conduct our analysis on the two-step output from the CoT evaluation across the entire dataset, focusing on two key dimensions.

Text Dominant

Text Lite

Text Only

[Figure 58]

[Figure 59]

[Figure 60]

× Ans. × Rea. 3.3%

× Ans. × Rea. 2.6%

× Ans. × Rea. 18.5%

× Ans. Partial × Rea. 44.4%

× Ans. Partial × Rea.

× Ans. Partial × Rea. 31.9%

✓ Ans. 55.6%

✓ Ans. ✓ Rea. 37.1%

✓ Ans. ✓ Rea. 35.1%

✓ Rea. 25.8%

✓ Ans. Partial × Rea. 14.6%

× Ans. ✓ Rea. 2.6%

× Ans. ✓ Rea. 7.9%

✓ Ans. × Rea. 0.7%

✓ Ans. Partial × Rea. 11.3%

✓ Ans. × Rea. 0.0%

✓ Ans. Partial × Rea. 4.6%

✓ Ans. × Rea. 0.0%

× Ans. ✓ Rea. 4.0%

Vision Intensive

Vision Dominant

Vision Only

[Figure 61]

[Figure 62]

[Figure 63]

× Ans. × Rea. 6.6%

× Ans. × Rea. 6.0%

× Ans. × Rea. 24%

× Ans. Partial × Rea.

× Ans. Partial × Rea.

✓ Ans. 54.3%

✓ Ans. ✓ Rea. 24.8%

✓ Ans. 54.3%

× Ans. Partial × Rea. 39.7%

✓ Rea. 27.2%

✓ Rea. 29.8%

✓ Ans. Partial × Rea. 4.1%

× Ans. ✓ Rea. 1.3%

× Ans. ✓ Rea. 2.0%

✓ Ans. Partial × Rea. 6.6%

✓ Ans. Partial × Rea. 10.6%

✓ Ans. × Rea. 0.0%

✓ Ans. × Rea. 1.3%

✓ Ans. × Rea. 0.0%

× Ans. ✓ Rea. 7.4%

- Figure 9: Distribution of GPT-4V’s [47] Errors in Reasoning and Answers. For the six problem versions in MATHVERSE, we provide the statistics of errors made by GPT-4V based on their occurrence in answers (‘Ans.’) and reasoning processes (‘Rea.’).

Errors in Reasoning or Answer? In Figure 9, we showcase the statistics of different error distributions in six problem versions of MATHVERSE. We define the following six error categories: correct final answer with correct/partially correct/incorrect CoT reasoning and incorrect final answer with correct/partially correct/incorrect CoT reasoning. For all six versions, the incorrect final answers are mostly caused by the partially incorrect reasoning process. In addition, a number of problems with correct answers are accompanied by partially or entirely incorrect reasoning, e.g., 15.3% in Text Dominant, which cannot be detected by the traditional True or False evaluation. As we remove the content within textual questions and enrich the visual diagram, e.g., from Text Dominant and Lite to Vision Dominant and Only, we observe a progressive increase in the error rate of ‘incorrect final answer with incorrect CoT reasoning’, indicating that MLLMs are challenged to conduct high-quality intermediate reasoning by capturing more information from the visual input.

What Types of Errors? To further investigate the specific error types, we survey the problems with errors that occur either within the reasoning process or the final answer. As depicted in Figure 10, we divide the errors of GPT-4V into four distinct types: visual perception error, reasoning error, knowledge error, and calculation error. Consistent with our findings in the main paper, the primary source of errors in problem-solving attributes to the inaccurate interpretation of mathematical diagrams, which significantly impedes the performance of MLLMs. For the problem versions that demand advanced diagram interpretation, e.g., Vision Dominant and Only, we observe a notable increase in the rate of visual perception errors, demonstrating an urgent need for stronger visual encoders in MLLMs. Moreover, reasoning errors also account for a considerable percentage, indicating that the logical deduction skills of MLLMs still require improvement. As expected, knowledge errors do not significantly hinder the mathematical reasoning capabilities of MLLMs in MATHVERSE.

Text Dominant Text Lite Text Only

[Figure 64]

[Figure 65]

[Figure 66]

Knowledge Error 10.3%

Knowledge Error 5.9%

Knowledge Error 8.8%

Calculation Error 17.6%

Calculation Error 20.6%

Calculation Error 37.9%

Reasoning Error 32.4%

Visual Perception Error 32.4%

Visual Perception Error 44.1%

Reasoning Error 51.8%

Reasoning Error 38.2%

Visual Perception Error 0.0%

Vision Intensive Vision Dominant Vision Only

[Figure 67]

[Figure 68]

Calculation Error 11.8%

[Figure 69]

Calculation Error 9.1%

Knowledge Error 4.5%

Knowledge Error 5.9%

Knowledge Error 12.1%

Calculation Error 22.7%

Reasoning Error 35.2%

Reasoning Error 36.4%

Reasoning Error 27.3%

Visual Perception Error 42.4%

Visual Perception Error 47.1%

Visual Perception Error 45.5%

- Figure 10: Distribution of GPT-4V’s [47] Errors within Different Types. We present the statistics of four error types by GPT-4V in the six problem versions, i.e., Visual Perception Error, Reasoning Error, Calculation Error, and Knowledge Error.

### 4 Conclusion

In this paper, we propose a comprehensive and specialized benchmark, MATHVERSE, for the visual mathematical problem-solving capacity of MLLMs. We meticulously collect high-quality math problems with diagrams spanning three primary subjects and twelve subfields. Given the issues within current benchmarks, we transform each problem into six versions, investigating whether and how much MLLMs can interpret the visual math diagrams. We also propose a CoT evaluation strategy for finer-grained assessment of the intermediate reasoning process of MLLMs. By evaluating various closed-source and open-source models, MATHVERSE unveils that most existing MLLMs struggle to accurately understand mathematical diagrams, and even attain higher results without visual input. This indicates the potential of developing more advanced math-specific vision encoders for stronger multi-modal mathematical reasoning.

### References

- [1] Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems 35, 23716–23736 (2022)
- [2] Amini, A., Gabriel, S., Lin, P., Koncel-Kedziorski, R., Choi, Y., Hajishirzi, H.: Mathqa: Towards interpretable math word problem solving with operation-based formalisms. arXiv preprint arXiv:1905.13319 (2019)
- [3] Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., Zhou, J.: Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966 (2023)
- [4] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. In: Advances in neural information processing systems. pp. 1877–1901 (2020)

- [5] Cao, J., Xiao, J.: An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In: Proceedings of the 29th International Conference on Computational Linguistics. pp. 1511–1520 (2022)
- [6] Chen, G., Zheng, Y.D., Wang, J., Xu, J., Huang, Y., Pan, J., Wang, Y., Wang, Y., Qiao, Y., Lu, T., et al.: Videollm: Modeling video sequence with large language models. arXiv preprint arXiv:2305.13292 (2023)
- [7] Chen, J., Li, T., Qin, J., Lu, P., Lin, L., Chen, C., Liang, X.: Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. arXiv preprint arXiv:2212.02746 (2022)
- [8] Chen, J., Li, T., Qin, J., Lu, P., Lin, L., Chen, C., Liang, X.: Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. ArXiv abs/2212.02746 (2022)
- [9] Chen, J., Tang, J., Qin, J., Liang, X., Liu, L., Xing, E.P., Lin, L.: Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. arXiv preprint arXiv:2105.14517

(2021)

- [10] Chen, J., Tang, J., Qin, J., Liang, X., Liu, L., Xing, E.P., Lin, L.: Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. ArXiv abs/2105.14517 (2021), https://api.semanticscholar.org/CorpusID:235253782
- [11] Chen, J., Li, D.Z.X.S.X., Zhang, Z.L.P., Xiong, R.K.V.C.Y., Elhoseiny, M.: Minigpt-v2: Large language model as a unified interface for vision-language multi-task learning. arXiv preprint arXiv:2310.09478 (2023)
- [12] Chen, L., Li, J., wen Dong, X., Zhang, P., He, C., Wang, J., Zhao, F., Lin, D.: Sharegpt4v: Improving large multi-modal models with better captions. ArXiv abs/2311.12793 (2023), https://api.semanticscholar.org/CorpusID:265308687
- [13] Chiang, W.L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J.E., Stoica, I., Xing, E.P.: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. https://lmsys.org/blog/2023-03-30-vicuna/ (March 2023)
- [14] Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al.: Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168 (2021)
- [15] Dai, W., Li, J., Li, D., Tiong, A.M.H., Zhao, J., Wang, W., Li, B., Fung, P., Hoi, S.: Instructblip: Towards general-purpose vision-language models with instruction tuning (2023)
- [16] Dong, X., Zhang, P., Zang, Y., Cao, Y., Wang, B., Ouyang, L., Wei, X., Zhang, S., Duan, H., Cao, M., et al.: Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420 (2024)
- [17] Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., Wu, Y., Ji, R.: Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394 (2023)
- [18] Fu, C., Zhang, R., Lin, H., Wang, Z., Gao, T., Luo, Y., Huang, Y., Zhang, Z., Qiu, L., Ye, G., et al.: A challenger to gpt-4v? early explorations of gemini in visual expertise. arXiv preprint arXiv:2312.12436 (2023)
- [19] Gao, J., Pi, R., Zhang, J., Ye, J., Zhong, W., Wang, Y., Hong, L., Han, J., Xu, H., Li, Z., et al.: G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370 (2023)
- [20] Gao, P., Han, J., Zhang, R., Lin, Z., Geng, S., Zhou, A., Zhang, W., Lu, P., He, C., Yue, X., Li, H., Qiao, Y.: Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010 (2023)
- [21] Gao, P., Zhang, R., Liu, C., Qiu, L., Huang, S., Lin, W., Zhao, S., Geng, S., Lin, Z., Jin, P., et al.: Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. arXiv preprint arXiv:2402.05935 (2024)

- [22] Gemini Team, G.: Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023)
- [23] Guo, Z., Zhang, R., Zhu, X., Tang, Y., Ma, X., Han, J., Chen, K., Gao, P., Li, X., Li, H., et al.: Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following. arXiv preprint arXiv:2309.00615 (2023)
- [24] Han, J., Zhang, R., Shao, W., Gao, P., Xu, P., Xiao, H., Zhang, K., Liu, C., Wen, S., Guo, Z., et al.: Imagebind-llm: Multi-modality instruction tuning. arXiv preprint arXiv:2309.03905

(2023)

- [25] Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., Steinhardt, J.: Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR) (2021)
- [26] Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., Steinhardt, J.: Measuring mathematical problem solving with the math dataset. NeurIPS (2021)
- [27] Hong, Y., Zhen, H., Chen, P., Zheng, S., Du, Y., Chen, Z., Gan, C.: 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems 36

(2024)

- [28] Jiang, A.Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D.S., de Las Casas, D., Hanna, E.B., Bressand, F., Lengyel, G., Bour, G., Lample, G., Lavaud, L.R., Saulnier, L., Lachaux, M., Stock, P., Subramanian, S., Yang, S., Antoniak, S., Scao, T.L., Gervet, T., Lavril, T., Wang, T., Lacroix, T., Sayed, W.E.: Mixtral of experts. Arxiv 2401.04088 (2024)
- [29] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. arXiv preprint arXiv:2304.02643 (2023)
- [30] Li, B., Zhang, K., Zhang, H., Guo, D., Zhang, R., Li, F., Zhang, Y., Liu, Z., Li, C.: Llava-next: Stronger llms supercharge multimodal capabilities in the wild. https://llavavl.github.io/blog/2024-05-10-llava-next-stronger-llms/ (2024)
- [31] Li, B., Zhang, Y., Chen, L., Wang, J., Pu, F., Yang, J., Li, C., Liu, Z.: Mimic-it: Multi-modal in-context instruction tuning. arXiv preprint arXiv:2306.05425 (2023)
- [32] Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Li, Y., Liu, Z., Li, C.: Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 (2024)
- [33] Li, B., Wang, R., Wang, G., Ge, Y., Ge, Y., Shan, Y.: Seed-bench: Benchmarking multimodal llms with generative comprehension. ArXiv abs/2307.16125 (2023)
- [34] Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W., Ma, Z., Li, C.: Llava-nextinterleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895 (2024)
- [35] Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: International Conference on Machine Learning. pp. 12888–12900. PMLR (2022)
- [36] Lin, Z., Liu, C., Zhang, R., Gao, P., Qiu, L., Xiao, H., Qiu, H., Lin, C., Shao, W., Chen, K., et al.: Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575 (2023)
- [37] Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning (2023)
- [38] Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., Lee, Y.J.: Llava-next: Improved reasoning, ocr, and world knowledge (January 2024), https://llava-vl.github.io/blog/ 2024-01-30-llava-next/
- [39] Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. In: NeurIPS (2023)

- [40] Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281 (2023)
- [41] Lu, P., Bansal, H., Xia, T., Liu, J., yue Li, C., Hajishirzi, H., Cheng, H., Chang, K.W., Galley, M., Gao, J.: Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. ArXiv abs/2310.02255 (2023)
- [42] Lu, P., Gong, R., Jiang, S., Qiu, L., Huang, S., Liang, X., Zhu, S.C.: Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv preprint arXiv:2105.04165 (2021)
- [43] Lu, P., Gong, R., Jiang, S., Qiu, L., Huang, S., Liang, X., Zhu, S.C.: Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In: Annual Meeting of the Association for Computational Linguistics (2021), https://api.semanticscholar. org/CorpusID:234337054
- [44] Luo, H., Sun, Q., Xu, C., Zhao, P., Lou, J., Tao, C., Geng, X., Lin, Q., Chen, S., Zhang, D.: Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583 (2023)
- [45] OpenAI: Chatgpt. https://chat.openai.com (2023)
- [46] OpenAI: Gpt-4 technical report. ArXiv abs/2303.08774 (2023)
- [47] OpenAI: GPT-4V(ision) system card (2023), https://openai.com/research/ gpt-4v-system-card
- [48] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Gray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., Lowe, R.: Training language models to follow instructions with human feedback. In: Advances in Neural Information Processing Systems (2022)
- [49] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning (2021), https://api.semanticscholar.org/CorpusID:231591445
- [50] Roy, S., Roth, D.: Solving general arithmetic word problems. ArXiv abs/1608.01413 (2016), https://api.semanticscholar.org/CorpusID:560565
- [51] Seo, M., Hajishirzi, H., Farhadi, A., Etzioni, O., Malcolm, C.: Solving geometry problems: Combining text and diagram interpretation. In: Proceedings of the 2015 conference on empirical methods in natural language processing. pp. 1466–1476 (2015)
- [52] Su, Y., Lan, T., Li, H., Xu, J., Wang, Y., Cai, D.: Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355 (2023)
- [53] Sun, K., Pan, J., Ge, Y., Li, H., Duan, H., Wu, X., Zhang, R., Zhou, A., Qin, Z., Wang, Y., et al.: Journeydb: A benchmark for generative image understanding. Advances in Neural Information Processing Systems 36 (2024)
- [54] Team, I.: Internlm: A multilingual language model with progressively enhanced capabilities

(2023)

- [55] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023)
- [56] Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023)

- [57] Wang, K., Ren, H., Zhou, A., Lu, Z., Luo, S., Shi, W., Zhang, R., Song, L., Zhan, M., Li, H.: Mathcoder: Seamless code integration in LLMs for enhanced mathematical reasoning. In: The Twelfth International Conference on Learning Representations (2024), https://openreview. net/forum?id=z8TW0ttBPp
- [58] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems 35, 24824–24837 (2022)
- [59] Xu, P., Shao, W., Zhang, K., Gao, P., Liu, S., Lei, M., Meng, F., Huang, S., Qiao, Y., Luo, P.: Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models. arXiv preprint arXiv:2306.09265 (2023)
- [60] Xu, R., Wang, X., Wang, T., Chen, Y., Pang, J., Lin, D.: Pointllm: Empowering large language models to understand point clouds. arXiv preprint arXiv:2308.16911 (2023)
- [61] Ye, Q., Xu, H., Xu, G., Ye, J., Yan, M., Zhou, Y., Wang, J., Hu, A., Shi, P., Shi, Y., Jiang, C., Li, C., Xu, Y., Chen, H., Tian, J., Qian, Q., Zhang, J., Huang, F.: mplug-owl: Modularization empowers large language models with multimodality (2023)
- [62] Ye, Q., Xu, H., Ye, J., Yan, M., Hu, A., Liu, H., Qian, Q., Zhang, J., Huang, F., Zhou, J.: mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration

(2023)

- [63] Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., Wei, C., Yu, B., Yuan, R., Sun, R., Yin, M., Zheng, B., Yang, Z., Liu, Y., Huang, W., Sun, H., Su, Y., Chen, W.: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502 (2023)
- [64] Yue, X., Qu, X., Zhang, G., Fu, Y., Huang, W., Sun, H., Su, Y., Chen, W.: Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653

(2023)

- [65] Zhang, H., Li, X., Bing, L.: Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858 (2023)
- [66] Zhang, R., Han, J., Zhou, A., Hu, X., Yan, S., Lu, P., Li, H., Gao, P., Qiao, Y.: LLaMA-adapter: Efficient fine-tuning of large language models with zero-initialized attention. In: The Twelfth International Conference on Learning Representations (2024), https://openreview.net/ forum?id=d4UiXAHN2W
- [67] Zhang, R., Hu, X., Li, B., Huang, S., Deng, H., Li, H., Qiao, Y., Gao, P.: Prompt, generate, then cache: Cascade of foundation models makes strong few-shot learners. CVPR 2023 (2023)
- [68] Zhang, R., Jiang, Z., Guo, Z., Yan, S., Pan, J., Dong, H., Gao, P., Li, H.: Personalize segment anything model with one shot. ICLR 2024 (2023)
- [69] Zhang, R., Wang, L., Qiao, Y., Gao, P., Li, H.: Learning 3d representations from 2d pre-trained models via image-to-point masked autoencoders. CVPR 2023 (2023)
- [70] Zhang, R., Wei, X., Jiang, D., Zhang, Y., Guo, Z., Tong, C., Liu, J., Zhou, A., Wei, B., Zhang, S., et al.: Mavis: Mathematical visual instruction tuning. arXiv preprint arXiv:2407.08739 (2024)
- [71] Zhou, A., Wang, K., Lu, Z., Shi, W., Luo, S., Qin, Z., Lu, S., Jia, A., Song, L., Zhan, M., et al.: Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. arXiv preprint arXiv:2308.07921 (2023)
- [72] Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M.: Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592 (2023)

### Appendix Overview

- • Section A: Related work.
- • Section B: Additional experimental details.
- • Section C: More dataset details.
- • Section D: Comparison to current benchmarks.
- • Section E: Limitation and future work.
- • Section F: Qualitative examples.

### A Related Work

Multi-modal Large Language Models (MLLMs), building upon the prevalence of Large Language Models (LLMs) [55, 56, 45, 28, 4] and large vision models [49, 29, 68, 67, 69], have become increasingly prominent in the field. They extend LLMs to tackle a diverse range of tasks and domains, including the mainstream 2D images [35, 15, 1, 31] and other modalities, such as 3D point clouds [23, 60, 27], audio [24, 52], and video [65, 6]. Noteworthy examples like OpenAI’s GPT-4V [47] and Google’s Gemini [22] exhibit exceptional visual understanding and reasoning capabilities, setting new benchmarks in multi-modal performance. However, their closed-source nature poses a barrier to the broader application and development of MLLMs. Concurrently, another line of work is dedicated to exploring advanced MLLMs open-source to the community. Prior efforts like LLaMA-Adapter [66, 20], LLaVA [39, 38, 37], and MiniGPT-4 [72, 11] leverage a frozen CLIP [49] model for image encoding, and inject the visual cues into LLaMA [55] for multi-modal instruction tuning. The subsequent mPLUG-Owl [61, 62], Qwen-VL [3], InternLM-XComposer [16], and SPHINX [36, 21] further push the frontier of MLLMs in understanding and generalizing across visual contexts. Despite comprehensive benchmarks [17, 40, 33, 59] on general visual instruction-following scenarios, the specific potential of MLLMs for visual mathematical problem-solving remains underexplored. In this paper, we introduce the MATHVERSE benchmark to comprehensively evaluate the visual mathematical reasoning and diagram understanding skills of MLLMs, providing unique perspectives for future research directions.

Mathematical Reasoning Benchmarks have emerged as a significant area of focus, posing considerable challenges for large foundational models, e.g., LLMs and MLLMs. Initially, datasets in this realm are designed to address basic algebraic [26] and arithmetic [50] word problems, which are relatively limited in scope and volume. Subsequent efforts, including MATH [26], GSM8K [14], and MMLU [25], expand the range and quality of textual mathematical problems. These datasets feature a broader spectrum of difficulties, establishing a robust benchmark for the evaluation of general and math-specific LLMs [71, 64, 57, 19, 44]. Besides the text-only assessment, there is a growing demand for comparable, high-quality benchmarks for evaluating mathematical problemsolving in visual contexts, with the rapid progress of MLLMs. There are prior attempts, such as GeoQA [10], UniGeo [8], and Geometry3K [43], which focused exclusively on geometric problems. The recently proposed MathVista [41] broadens the scope to incorporate a variety of multi-modal tasks involving mathematical reasoning, and MMMU [63] covers college-level questions demanding intricate, domain-specific knowledge. However, our analysis identifies three main shortcomings within the current visual math benchmarks, as elaborated in Section 1 of the main paper. Therefore, we propose MATHVERSE specialized in the multi-modal mathematical evaluation of MLLMs, comprising twelve subjects, six problem versions, and 20K test samples. Our objective is to thoroughly investigate whether and how much MLLMs genuinely interpret visual diagrams for mathematical reasoning.

### B Additional Experimental Details

Model Sources. For different MLLMs, we select their latest models and best-performing configurations for evaluation to fully reveal their visual mathematical proficiency. Table 4 presents the release time and model sources of MLLMs used in MATHVERSE.

#### Table 4: The Release Time and Model Source of MLLMs Used in MATHVERSE.

Release Time

Source

Model

ChatGPT [48] 2022-11 https://platform.openai.com/ GPT-4 [46] 2023-03 https://platform.openai.com/

https://help.aliyun.com/zh/ dashscope/developer-reference/ vl-plus-quick-start

Qwen-VL-Plus [3] 2023-11

Gemini-Pro [22] 2023-12 https://ai.google.dev/

https://help.aliyun.com/zh/ dashscope/developer-reference/ vl-plus-quick-start

Qwen-VL-Max [3] 2024-01

GPT-4V [47] 2023-09 https://platform.openai.com/

https://github.com/OpenGVLab/ LLaMA-Adapter/tree/main/llama_ adapter_v2_multimodal7b

LLaMA-Adapter V2 [20] 2023-04

LLaVA-1.5 [37] 2023-10 https://huggingface.co/

liuhaotian/llava-v1.5-13b MiniGPT-v2 [11] 2023-10 https://github.com/

Vision-CAIR/MiniGPT-4

mPLUG-Owl2 [62] 2023-11 https://huggingface.co/

MAGAer13/mplug-owl2-llama2-7b G-LLaVA [19] 2023-12 https://github.com/pipilurj/

G-LLaVA/tree/main

https://github.com/OpenGVLab/ LLaMA-Adapter/tree/main/ imagebind_LLM

ImageBind-LLM [24] 2023-05

ShareGPT4V [12] 2023-11 https://huggingface.co/ Lin-Chen/ShareGPT4V-13B

https://huggingface.co/ Alpha-VLLM/LLaMA2-Accessory/ tree/main/finetune/mm/SPHINX/ SPHINX-v2-1k

SPHINX-Plus [36] 2023-11

https://huggingface.co/ liuhaotian/llava-v1. 6-vicuna-13b

LLaVA-NeXT [38] 2024-01

https://huggingface.co/ Alpha-VLLM/LLaMA2-Accessory/ tree/main/finetune/mm/SPHINX/ SPHINX-MoE

SPHINX-MoE [21] 2024-01

https://huggingface. co/internlm/ internlm-xcomposer2-vl-7b

InternLM-XComposer2 [16] 2024-01

- Table 5: Input Prompt of MLLMs for Response Generation. We adopt two different prompts for the free-form and multiple-choice questions. Note that these prompts are used for five problem versions except for the Vision-only version.

Question Prompt

Free-form Question

Please first conduct reasoning, and then answer the question and provide the final value, e.g., 1, 2.5, 300, at the end.

– Question: {question}

Multiple-choice Question

Please first conduct reasoning, and then answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.

– Question: {question}

- Table 6: Input Prompt for Vision-only Problems. Especially for the Vision-only version without textual input, we add “According to the question shown in the image” at the beginning of the prompt, and remove the “Question:” at the end.

#### Question Prompt

According to the question shown in the image, please first conduct reasoning, and then answer the question and provide the final value, e.g., 1, 2.5, 300, at the end.

Free-form Question

According to the question shown in the image, please first conduct reasoning, and then answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.

Multiple-choice Question

Prompt for Response Generation. We adopt two types of prompts respectively for the free-form and multiple-choice questions, as shown in Table 5. We inspire the Chain-of-Thought (CoT) reasoning capabilities of MLLMs by using the phrase “first conduct reasoning”. Especially for the Vision-only problem version in Table 6, we add “According to the question shown in the image” at the beginning to remind MLLMs to read the questions rendered within diagrams, where the textual input for MLLMs only contains the prompt itself.

Prompt for the CoT Evaluation. Our proposed CoT evaluation contains two steps, i.e., key-step extraction and multi-step scoring, which prompt GPT-4 [46] and GPT-4V [47], respectively. The input configuration is listed in Table 7. We utilize the text-only GPT-4 in the first step to extract multiple key steps within the model’s unstructured output, without feeding the question information. In the second step, we input the extracted key-step reasoning and all the available content related to the problem into GPT-4V, allowing for a holistic assessment, including diagram interpretation, logical reasoning, and numerical computation. In Figure 11, we showcase the manual annotation for critical information within functional diagrams, e.g., function expression and properties. This assists GPT-4V in evaluating the visual perception accuracy of MLLMs for function graphs.

Human Performance Assessment. We recruit ten qualified college students specifically for the evaluation of human performance on MATHVERSE. These individuals are kept separate from the data curation stage, eliminating the possibility of them encountering the solutions beforehand. We allocate to each student the questions from a specific problem version. This strategy is to prevent them from gaining additional information from another version to answer questions, e.g., leveraging the textual Implicit Property from the Text-lite version to solve Text-intensive problems. They are asked to directly provide the final answer without detailed reasoning. Therefore, we do not report the CoT evaluation results for human performance, alongside the ‘Random Chance’ baseline.

- Table 7: Configuration for the CoT Evaluation Strategy. We conduct two evaluation phases, respectively by prompting the text-only GPT-4 [46] and GPT-4V [47]. The symbol ‘XXX’ denotes the given one-shot sample, abbreviated for brevity. The ‘Annotation’ (represented in grey) in the second phase is only required for function problems.

Phase Input Prompt

I will give you a detailed solving procedure or a single answer for a math problem.

If it is a procedure, you need to extract the key solution steps and list them accordingly in markdown syntax. If it is just a single answer, output the answer directly. Here are examples:

Key-step Extraction (GPT-4)

Model Output

- – Model output: XXX
- – Extracted: 1. XXX 2. XXX 3. XXX
- – Model output: 2.2
- – Extracted: The single answer is 2.2 Here is what you need to extract:
- – Model output: {model output}
- – Extracted:

I will first give you a visual math problem, including the question, diagram, ground-truth answer, and detailed annotation of the diagram, and then give you a model output containing multiple key solution steps.

Please think step by step and output the Average score, along with the Final answer score in the end, as described below:

- – Average score: Evaluate, based on the given question, answer, diagram, and diagram annotation, whether each solution step is correct in logical reasoning, visual perception, and numerical computation, with an incorrect score of 0 and a correct score of 1. Then, calculate the average score of multiple steps.
- – Final answer score: Match the model’s final answer with the ground truth answer, scoring 1 if it matches and 0 if it doesn’t.
- – If the model output only includes a single step or answer, the Average score and Final answer score are the same.
- – Question: {question}
- – Ground-truth answer: {answer}
- – Diagram annotation: {annotation}
- – Model output: {extracted steps}
- – Average score:
- – Final answer score:

Extracted Steps Question Diagram Answer Annotation

Multi-step Scoring (GPT-4V)

###### Annotation:

Annotation: Annotation:

A piecewise function 𝑓: when 𝑥 <= 3, 𝑓 𝑥 = −𝑥 + 1, when 𝑥 > 3, 𝑓(𝑥) = 𝑥 − 5.

A quadratic function 𝑦 = 𝑥 − 2 − 4, goes through (0, 0), (4, 0), opens upward with a vertex (2, 4).

A U-shape curve, passes the origin (0, 0), opens upward.

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Annotation: Annotation:

A linear function 𝑦 = 0.5𝑥 + 2, passes (-4, 0), (0, 2) and (2, 3), the shaded region is above the line.

Two quadratic functions going through (1, 0) and (7, 0), 𝑓(𝑥) = −(1/3)(𝑥 − 1)(𝑥 − 7) in blue with a vertex is (4, 3), 𝑔(𝑥) = −(2/3)(𝑥 − 1)(𝑥 − 7) in dashed red with a vertex is (4, 6).

[Figure 74]

[Figure 75]

- Figure 11: Manual Annotations for Function Problems in MATHVERSE. We provide detailed annotations, e.g., function expression and properties, for the diagrams of 534 function problems, which benefits the accuracy of GPT-4V [47] for CoT evaluation.

### C More Dataset Details

#### C.1 Data Curation

This paper engages twelve expert annotators for data curation, consisting of senior undergraduate and graduate students from across the globe with a strong background in science. In collaboration with the authors, they are required to mainly complete five tasks concerning data collection, categorization, quality review, problem version transformation, and function diagram annotation.

Data Collection. We comprehensively collect visual math problems from existing datasets [43, 9, 51] and public question repositories1,2,3. We specifically select high-quality plane geometric problems from current benchmarks, which showcase various question types, moderate question length, diverse diagram styles, and appropriate solving difficulty. For the manually collected problems of three subjects (plane geometry, solid geometry, and functions), we apply the Mathpix tool4 to accurately extract the question texts, diagrams, explanations, and answers from the website. We strictly comply with copyright and licensing rules, ensuring that we refrain from using data from sites that forbid copying and redistribution. After the initial collection, we obtain around 3.5K visual math problems, with 1.5K from existing datasets and 2K newly collected.

- 1https://homework.study.com
- 2https://www.ixl.com/math
- 3https://mathspace.co/us
- 4https://mathpix.com

Descriptive Information Original Question: Text Dominant:

[Figure 76]

[Figure 77]

In ⊙G, CE and AB are both diameters. ∠AGD is a right angle and ∠AGC is 60°. Find ∠BGE. Choices: A:60° B:120° C:240° D:360°.

Add

∠AGC is 60°. Find ∠BGE. Choices: A:60° B:120° C:240° D:360°.

[Figure 78]

[Figure 79]

Add

A polynomial function is graphed, which goes through (-2, 1), (0, 2), and (3, 1). Determine the least possible degree of the function.

Determine the least possible degree of the polynomial function shown.

[Figure 80]

Add

The figure shows ¾ of a cylinder. The height is 15 centimeters and the radius is 5 centimeters. Find the volume of the figure shown, correct to two decimal places.

[Figure 81]

Find the volume of the figure shown with a radius of 5 centimeters, correct to two decimal places.

- Figure 12: Manual Annotations for Descriptive Information in MATHVERSE. For some collected problems, we are required to supplement additional Descriptive Information (highlighted in red) to distinguish the Text-dominant version.

Data Categorization and Review. We first ask the human annotators to categorize the problems into three primary subjects, i.e., plane geometry, solid geometry, and functions. Within each subject, according to the definitions in Section C.2, the math problems are further divided into twelve finegrained categories. At the same time, we meticulously review the collected dataset. We manually rectify the problems with incorrect answers and discard the problems with multiple diagrams, visual solutions, and too much similar content to others. Finally, 2,612 high-quality math problems with paired diagrams are preserved for MATHVERSE, spanning diverse subjects and subfields.

Transformation of Problem Versions. Given the three types of textual information within questions, human annotators rigorously transform each problem into six different versions as discussed in Section 2.2 of the main paper. We utilize Microsoft PowerPoint to annotate the diagrams in the Vision-dominant version, and employ Matplotlib to render the questions onto the diagrams in the Vision-only version. As illustrated in Figure 12, for problems with minimal Descriptive Information, we manually enhance the question text with additional contextual description about the diagram to differentiate the Text-dominant version. In the case of questions in Figure 13, where the Essential Condition has been fully depicted in the diagrams, we remove some of this content from the diagram and incorporate it into the text to mark the Vision-dominant version.

#### C.2 Subject and Subfield Definition

The visual math problems within MATHVERSE encompass three primary subjects, plane geometry, solid geometry, and functions, alongside twelve finer-grained subfields, which comprehensively evaluate the diagram understanding and mathematical reasoning capabilities of MLLMs.

Plane Geometry is a fundamental area that explores the properties and relations of points, lines, and surfaces in a two-dimensional plane. This subject delves into concepts such as angles, triangles, circles, and polygons, offering a rich context for assessing the spatial comprehension and logical deduction skills of MLLMs. We divide it into five subfields, as exemplified in Figure 14:

Original Question: Text Dominant:

Essential Condition

[Figure 82]

[Figure 83]

Delete x

Add Essential Condition

In the right triangle, an acute angle is 70 degrees and the hypotenuse is x length. Find x to the nearest hundredth. Choices: A:5.13 B:5.46 C:15.96 D:43.86.

Find x to the nearest hundredth. Choices: A:5.13 B:5.46 C:15.96 D:43.86.

[Figure 84]

[Figure 85]

Delete 6

Add Essential Condition

In trapezoid JKLM, JM = 6, KL = 6, and ∠JML is 80 degrees. Find ∠K. Choices: A:6° B:60° C:100° D:180°.

Find ∠K. Choices: A:6° B:60° C:100° D:180°.

- Figure 13: Manual Modification for Textual Essential Condition in MATHVERSE. For the original problems shown, we transfer some of the Essential Condition from diagrams to question texts (highlighted in green) to mark the Vision-dominant version.

- • Length focuses on the measurement and comparison of distances between points. This subfield includes understanding the properties of lines, segments, and their use in determining the perimeters of geometric shapes, which is foundational for MLLMs to solve plane geometry problems.
- • Area examines the size of two-dimensional surfaces. It encompasses calculating the areas of various shapes, such as triangles, rectangles, circles, and more complex polygons, by applying specific formulas and principles, which is crucial for comprehending the concept of space within geometry.
- • Angle involves the study of angles and their properties, including different types of angles (acute, right, and obtuse), angle measurement, and the relationships between angles, particularly in polygons. This subfield demands the advanced spatial perception capacity of MLLMs.
- • Analytic Geometry, also known as coordinate geometry, merges algebra and geometry to solve geometric problems using coordinate systems, exploring the calculation and reasoning of equations for geometric shapes. MLLMs are evaluated on their coordinate identification and algebraic capabilities.
- • Applied Geometry relate to the application of geometric principles to solve real-world and theoretical problems. It challenges MLLMs to first understand the background information within questions, and apply their knowledge of lengths, areas, angles, and analytic geometry for problem-solving.

Solid Geometry focuses on the study of three-dimensional objects that have depth, length, and width, thereby offering a more complex and enriched exploration of spatial structures. This subject investigates a variety of shapes such as cubes, cylinders, spheres, and pyramids, and assesses MLLMs to tackle questions concerning the volume, surface area, and geometric properties of these solids. This subject contains three subfields, as exemplified in Figure 15:

• Length, extending from the 2D counterpart, focuses on measuring the edges and curves that define three-dimensional objects. It involves determining the linear distance between

[Figure 86]

###### Plane Geometry:

###### Length

###### Area

###### Angle

###### Analytic

###### Applied

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Question:

Question:

Question:

Question:

Question:

As shown in the figure, passing point P to draw the tangent PC of circle O, if AO = OB = PB = 1.0, then the length of PC is (). Choices: A:1 B:√3 C:2 D:√5.

As shown in the figure, circle O is tangent to AB at point C, angle BCE = 60.0, DC = 6.0, DE = 4.0, then 𝑆 is (). Choices: A:6√5 B:6√3 C:6√2 D:6.

Consider the points on the diagram P, M, and S, where S denotes the south cardinal point. Using the diagram, find the true bearing of P from M.

What are the coordinates of point A? Enter an exact value or round to the nearest hundredth.

A fighter jet, flying at an altitude of 2000 m is … find the distance covered by the jet in one minute. Round your answer to the nearest meter.

√2 2

√2 2

(

,

)

Answer: 230∘T

Answer: B:√3

Answer: B:6√3

Answer:

Answer:

1688

- Figure 14: Examples of Five Subfields in Plane Geometry, spanning Length, Area, Angle, Analytic, and Applied Geometry problems. We showcase the Text-lite version.

Question:

Answer: B:√3

Length

[Figure 92]

As shown in the figure, passing point P to draw the tangent PC of circle O, if AO = OB = PB = 1.0, then the length of PC is (). Choices: A:1 B:√3 C:2 D:√5.

Question:

Answer: B:6√3

Area

[Figure 93]

As shown in the figure, circle O is tangent to AB at point C, angle BCE = 60.0, DC = 6.0, DE = 4.0, then 𝑆 is (). Choices: A:6√5 B:6√3 C:6√2 D:6.

Question:

Answer: 230∘T

Angle

[Figure 94]

Consider the points on the diagram P, M, and S, where S denotes the south cardinal point. Using the diagram, find the true bearing of P from M.

Question:

Answer:

Analytic

What are the coordinates of point A? Enter an exact value or round to the nearest hundredth.

(

√2 2

,

√2 2

)

[Figure 95]

Question:

Answer:

Applied

[Figure 96]

A fighter jet, flying at an altitude of 2000 m is … find the distance covered by the jet in one minute. Round your answer to the nearest meter.

1688

Plane Geometry:

[Figure 97]

Question:

Answer: 10 m

Length

Question:

Answer: 124.85 𝑐𝑚

Area

Question:

Answer: 63.3 g

Volume

Solid Geometry:

[Figure 98]

[Figure 99]

The perpendicular height of the cone is 12 m. Hence, find the length of the diameter of the cone's base.

[Figure 100]

The length of CD side is 12 cm. Calculate the area of the triangular divider correct to two decimal places.

[Figure 101]

The length of the pipe is 13 cm. Calculate the weight of the pipe if 1 𝑐𝑚 of the metal weighs 5.3 g, giving your answer correct to one decimal place.

Question:

Answer: −1

Coordinate

Question:

Answer:

Property

Question:

Answer: 63.3 g

Expression

Functions:

[Figure 102]

[Figure 103]

10 ⋅ 𝑓 7 + 9 ⋅ 𝑔(−1) =

B: Zero

Is the discriminant of 𝑓 positive, zero, or negative? Choices: A:Positive B:Zero C:Negative.

[Figure 104]

[Figure 105]

Write the equation for g(x).

Question:

Answer: D:When …

Applied

[Figure 106]

The function A models the rectangle's area (in square meters) ... Which of these statements are true? Choices: A:Greater width always relates …

- Figure 15: Examples of Three Subfields in Solid Geometry, spanning Length, Area, and Volume problems. We showcase the Text-lite version.

[Figure 107]

###### Solid Geometry:

###### Length

###### Area

Volume

[Figure 108]

[Figure 109]

[Figure 110]

Question:

###### Question:

Question:

The length of the pipe is 13 cm. Calculate the weight of the pipe if 1 𝑐𝑚 of the metal weighs 5.3 g, giving your answer correct to one decimal place.

The perpendicular height of the cone is 12 m. Hence, find the length of the diameter of the cone's base.

The length of CD side is 12 cm. Calculate the area of the triangular divider correct to two decimal places.

Answer: 124.85 𝑐𝑚

Answer: 10 m

Answer: 63.3 g

[Figure 111]

###### Functions:

###### Coordinate

###### Property

Expression

###### Applied

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

points in space, the perimeters of bases of solids, and the height or depth of objects. This measurement is a foundational element for MLLMs in analyzing geometric solids.

- • Area encompasses the calculation of the total area covered by the outer surfaces of solids. This normally requires MLLMs to break down complex shapes into several simpler components for area calculation in plane geometry, assessing their spatial and logical reasoning performance.
- • Volume pertains to measuring the space enclosed within three-dimensional objects. This demands MLLMs to precisely identify the geometric solids and apply accurate formulas to calculate the volume, which evaluates their mathematical knowledge application and calculation skills.

Question:

Question:

Question:

###### Question:

The function A models the rectangle's area (in square meters) ... Which of these statements are true? Choices: A:Greater width always relates …

Is the discriminant of 𝑓 positive, zero, or negative? Choices: A:Positive B:Zero C:Negative.

10 ⋅ 𝑓 7 + 9 ⋅ 𝑔(−1) =

Write the equation for g(x).

Answer: −1

Answer:

Answer: 63.3 g

Answer: D:When …

B: Zero

Functions involve analyzing mathematical functions to understand the relationship between variables. These challenges range from simple tasks, like calculating a function value for a given input, to more complex scenarios, such as exploring the behavior and representation of various function types. We evaluate MLLMs by four types of function problems, exemplified in Figure 16:

• Function Coordinate focuses on interpreting and extracting coordinate-level information from graphical representations of functions. It includes tasks such as identifying specific coordinate values of points on the graph and observing intersection points between functions and axes, which test the MLLM’s basic proficiency in functional visual perception.

###### Plane Geometry:

###### Length

###### Area

###### Angle

###### Analytic

###### Applied

Question:

Question:

Question:

Question:

Question:

As shown in the figure, passing point P to draw the tangent PC of circle O, if AO = OB = PB = 1.0, then the length of PC is (). Choices: A:1 B:√3 C:2 D:√5.

As shown in the figure, circle O is tangent to AB at point C, angle BCE = 60.0, DC = 6.0, DE = 4.0, then 𝑆 is (). Choices: A:6√5 B:6√3 C:6√2 D:6.

Consider the points on the diagram P, M, and S, where S denotes the south cardinal point. Using the diagram, find the true bearing of P from M.

What are the coordinates of point A? Enter an exact value or round to the nearest hundredth.

A fighter jet, flying at an altitude of 2000 m is … find the distance covered by the jet in one minute. Round your answer to the nearest meter.

√2 2

√2 2

(

,

)

Answer: 230∘T

Answer: B:√3

Answer: B:6√3

Answer:

Answer:

1688

###### Solid Geometry:

###### Length

###### Area

Volume

Question:

###### Question:

Question:

The length of the pipe is 13 cm. Calculate the weight of the pipe if 1 𝑐𝑚 of the metal weighs 5.3 g, giving your answer correct to one decimal place.

The perpendicular height of the cone is 12 m. Hence, find the length of the diameter of the cone's base.

The length of CD side is 12 cm. Calculate the area of the triangular divider correct to two decimal places.

Answer: 124.85 𝑐𝑚

Answer: 10 m

Answer: 63.3 g

[Figure 126]

###### Functions:

###### Coordinate

###### Property

Expression

###### Applied

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Question:

Question:

Question:

###### Question:

The function A models the rectangle's area (in square meters) ... Which of these statements are true? Choices: A:Greater width always relates …

Is the discriminant of 𝑓 positive, zero, or negative? Choices: A:Positive B:Zero C:Negative.

10 ⋅ 𝑓 7 + 9 ⋅ 𝑔(−1) =

Write the equation for g(x).

Answer: −1

Answer:

Answer: 63.3 g

Answer: D:When …

B: Zero

- Figure 16: Examples of Four Subfields in Functions, spanning Function Coordinate, Property, Expression, and Applied problems. We showcase the Text-lite version.

- • Function Property emphasizes the model’s capacity to discern and deduce the inherent properties of functions from their graphs, such as symmetry, asymptotes, extrema (maximum and minimum points), and intervals of increase or decrease. These problems can reveal the understanding of MLLMs for the deeper characteristics of functions.
- • Function Expression refers to the direct analysis using the algebraic expressions of functions, widely including linear, quadratic, polynomial, exponential, logarithmic, and piecewise functions. It challenges MLLMs to extract specific function expressions and apply transformations, bridging the gap between abstract mathematical reasoning and visual interpretation.
- • Applied Function, similar to the applied geometry problems, requires MLLMs to leverage their functional knowledge and theorems in real-world scenarios, e.g., modeling economic data, predicting physical phenomena, and calculating probabilities. This assesses the MLLM’s capabilities to understand functions in both theoretical and practical contexts.

#### C.3 Detailed Statistics of MATHVERSE

More Data Statistics. In Table 8, we provide a more detailed data statistics of MATHVERSE. Therein, the 534 newly annotated questions refer to all the function problems, for which we meticulously annotate critical functional information, as depicted in Figure 11. The number of newly annotated diagrams represents the 5,224 math problems in the Vision-dominant and Vision-only versions. For these problems, we respectively integrate the Essential Condition and all textual content with the diagrams. We also list the numbers of multiple-choice answers, where A, B, C, and D are almost uniformly distributed.

Problem Length Variance. In Table 9, we highlight the variance in question and answer lengths across the five problem versions in MATHVERSE, excluding the Vision-only category due to its absence of text. For both word and character levels, as we remove the pre-defined textual elements (Descriptive Information, Implicit Property, and Essential Condition), the maximum and average lengths of questions decrease accordingly, while the answer lengths remain the same. In Figure 17, we visualize the word-level variation of question length for the three problem versions: Text Dominant (blue), Text Lite (green), and Vision Dominant (red). By progressively omitting Descriptive Information and Essential Condition from the Text-dominant version, we observe a clear downward trajectory for the question length distribution and their average values.

#### Table 8: Statistics of MATHVERSE.

Statistic Number Total questions 2,612

- - Subjects/subfields 3/12
- - Multiple-choice questions 1,631 (62.4%)
- - Free-form questions 981 (37.6%)
- - Newly collected questions 1,236 (47.3%)
- - Existing-dataset questions 1,376 (52.7%)
- - Questions with explanations 1,236 (47.3%)
- - Newly annotated questions 534 (20.4%) Multiple-choice question

- - Proportion of answer A 585 (22.4%)
- - Proportion of answer B 828 (31.7%)
- - Proportion of answer C 703 (26.9%)
- - Proportion of answer D 444 (17.0%)
- - Proportion of answer E&F 52 (2.0%) Total test samples 15,672

- - Newly annotated samples 10,448 (66.7%)
- - Newly annotated diagrams 5,224 (33.3%)
- - Samples of each version 2,612 (16.7%)

Number of unique images 2,420 (92.6%) Number of unique questions 2,573 (98.5%) Number of unique answers 847 (32.4%)

#### Table 9: Length of Different Problem Versions in MATHVERSE.

Problem Version Word Character Text Dominant & Text Only

- - Maximum question length 203 1,311
- - Maximum answer length 17 102
- - Average question length 35.7 204.8
- - Average answer length 1.4 6.3 Text Lite

- - Maximum question length 179 1,173
- - Maximum answer length 17 102
- - Average question length 22 133.8
- - Average answer length 1.4 6.3 Vision Intensive

- - Maximum question length 171 1,126
- - Maximum answer length 17 102
- - Average question length 18.8 116.8
- - Average answer length 1.4 6.3 Vision Dominant

- - Maximum question length 176 1,132
- - Maximum answer length 17 102
- - Average question length 17.6 123.5
- - Average answer length 1.4 6.3

Distribution of Question Lengths (Word)

| |Text Dominant<br><br>Text Lite<br><br>Vision Dominant<br><br>Text Dominant Average|
|---|---|

0.20

Percentage

0.15

Text Lite Average Vision Dominant Average

0.10

0.05

0.00

0 20 40 60 80 100

- Figure 17: Distribution of Question Length for Three Problem Versions. We exclude the Descriptive Information and Essential Condition from the Text-dominant problems, respectively creating the Text-lite and Vision-dominant versions.

### D Comparison to Current Benchmarks

In this section, we offer a detailed comparison between MATHVERSE and existing multi-modal mathematical benchmarks, i.e., geometry-specific benchmarks [9, 5, 7, 42, 51], MathVista [41], and MMMU [63], from the following four aspects:

The Investigation of Diagram Interpretation Capacity. As discussed in Figure 1 of the main paper, the math problems in most existing datasets contain excessive redundant information in textual content, which is repetitive to the visual elements in diagrams. This issue enables MLLMs to potentially bypass the process of visual understanding, and thereby cannot determine whether and how much MLLMs truly interpret the math diagram. In contrast, our MATHVERSE includes six problem versions with different information content across text and vision. By comparing the performance variance between different problem versions, we can thoroughly investigate the mathematical diagram interpretation capabilities of MLLMs for the first time.

Evaluation Approach. Previous benchmarks adopt a simple True or False metric to score the response from MLLMs, which lacks fine-grained information and intermediate reasoning assessment, as analyzed in Figure 2 of the main paper. In contrast, MATHVERSE adopts a unique CoT evaluation

[Figure 131]

Table QA:

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Textbook and Science QA:

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Plot and Chart QA:

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

IQ Test and Synthetic QA:

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

General and Icon QA:

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

- Figure 18: Diagram Examples of Math-related Tasks in MathVista [41]. These tasks are not strongly correlated to the mathematical reasoning skills of MLLMs, probably skewing the assessment emphasis towards visual math problems.

Required Knowledge:

Alternating Group

Lagrangian Multipliers

Nonlinear Programming

Calculus

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Diagram:

[Figure 166]

Required Knowledge:

Cayley Diagram

Hopcroft Algorithm

Chromatic Index

Dijkstra Algorithm

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Diagram:

[Figure 171]

- Figure 19: Diagram Examples with Required Knowledge in MMMU [63]. These math problems demand MLLMs to comprehend college-level domain knowledge, potentially hindering them from fully exerting mathematical reasoning skills.

strategy by examining each crucial solution step within the model output. This approach not only unveils the CoT reasoning quality of MLLMs, but also provides detailed error analysis, serving as valuable guidance for future enhancement.

The Depth and Width in Math Problems. The geometry-specific benchmarks evaluate only a limited dimension of mathematical skills in MLLMs. MathVista instead incorporates a variety of mathrelated question-answering tasks, e.g., textbook figures, tables, plots, charts, puzzles, and synthetic scenes, as exemplified in Figure 18. However, the integration of these peripheral tasks (covering more than 70%) might divert the focus from the specialized mathematical evaluation of MLLMs. In addition, MMMU focuses on college-level complexity, requiring advanced domain-specific knowledge, as depicted in Figure 19. Given this, the lack of profound mathematical theorems would restrict the performance of MLLMs, biasing the evaluation of logical reasoning and visual perception proficiency. Therefore, our MATHVERSE concentrates on specialized visual math problems (plane geometry, solid geometry, and functions) with a moderate difficulty (high-school level), aiming to fully exert the capabilities of MLLMs.

Total Volume of Test Samples. We summarize the size of test instances for different datasets in Table 10. As demonstrated, our MATHVERSE offers a considerably larger number of samples than others, nearly three times to MathVista and twenty times to GeoQA+, including meticulously annotated six versions of visual math problems. This contributes to a comprehensive and robust evaluation of visual mathematical reasoning capabilities.

#### Table 10: Number of Test Samples in Different Benchmarks.

MMMUMath

Benchmark GEOS Geo3K GeoQA+ MathVista

MATHVERSE Test Samples 119 601 755 6,141 540 15,672

### E Limitation and Future Work

While our MATHVERSE takes a step forward in the field of visual mathematical evaluation for MLLMs, it is important to recognize several limitations as follows.

We have categorized the math problems in MATHVERSE by various criteria, including subjects, subfields, and versions featuring differing degrees of multi-modal content. These categorization approaches evaluate the capabilities of MLLMs from multiple dimensions. Nevertheless, it is also meaningful to further divide the problems based on their difficulty levels, akin to MATH [26], a text-only benchmark defining five levels of difficulty. This additional layer of differentiation can provide deeper insights into the problem-solving abilities of MLLMs across a spectrum of challenges, which we leave as future work.

The curated dataset in MATHVERSE focuses on math problems in the high school level with moderate difficulty, which aims to fully demonstrate the mathematical reasoning skills within current MLLMs. However, with the advancement of architecture and training methodologies, future MLLMs have the potential to grasp more complex knowledge and theorems across a variety of domains. Therefore, there is significant value in further augmenting MATHVERSE with problems spanning broader complexity and disciplines, including those at the college level and within scientific fields. By transforming the expanded problems into different versions, we can facilitate a more comprehensive and robust evaluation of MLLMs for their diagram interpretation and reasoning capabilities.

Moreover, the problems in MATHVERSE and other current mathematical benchmarks are mainly in English. Given that some multilingual MLLMs [24, 3] have been developed, existing evaluation cannot reveal their full capabilities when confined to a single language. The incorporation of multilingual visual math problems would not only extend the dataset’s global applicability, but also enhance the assessment of MLLMs for linguistic diversity and understanding.

### F Qualitative Examples

To ease the understanding, we offer a variety of qualitative examples in MATHVERSE. In Section F.1, we showcase the meticulously transformed six versions of visual math problems. In Section F.2, we compare the response of different MLLMs on Text-lite problems, including GPT-4V [47], LLaVANeXT [38], and SPHINX-MoE [21]. Specifically, we present the key-step extraction output by the CoT evaluation, and mark the multi-step scoring results aside. In Section F.3, we provide the response comparison of GPT-4V for three problem versions in MATHVERSE, i.e., Text Dominant, Text Lite, and Vision Dominant.

- F.1 Comparison of Six Problem Versions Please refer to Figures 13∼15.
- F.2 Response of Different MLLMs Please refer to Figures 16∼21.
- F.3 Response of Different Problem Versions Please refer to Figures 22∼27.

[Figure 172]

Descriptive Information Implicit Property Essential Condition

📖 Text Input 🔍 Vision Input

📖 Text Input 🔍 Vision Input

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

The logarithm function

[Figure 177]

[Figure 178]

[Figure 179]

Write an equation for the transformation of f(x)=|x| as shown in the figure, which passes (3, 3) and is piecewise.

- f(x)=ln(x) passing (1,4) and logarithm function
- g(x) passing (1,0) are shown below. g(x) is a transformation

Text Dominant

- of f(x). Hence state the equation of g(x).

The function f(x)=ln(x) and g(x) are shown, hence state the equation

- of g(x).

The logarithm function

[Figure 180]

[Figure 181]

- f(x)=ln(x) and logarithm function g(x) are shown below.
- g(x) is a transformation of f(x). Hence state the equation of g(x).

Write an equation for the transformation of f(x)=|x| as shown in the figure.

Text Lite

The logarithm function

Write an equation for the transformation of f(x)=|x| as shown in the figure, which passes (3, 3) and are piecewise.

- f(x)=ln(x) passing (1,4) and logarithm function
- g(x) passing (1,0) are shown below. g(x) is a transformation of f(x). Hence state the equation of g(x).

Text Only

[Figure 182]

[Figure 183]

Vision Intensive

Write an equation for the function.

The logarithm function f(x) and logarithm function g(x) are shown below.

[Figure 184]

[Figure 185]

Write an equation for the transformation of f(x)=|x| as shown in the figure.

Vision Dominant

g(x) is a transformation of f(x). Hence state the equation of g(x).

[Figure 186]

[Figure 187]

Vision Only

#### Figure 20: Comparison of Six Problem Versions in MATHVERSE.

[Figure 188]

Descriptive Information Implicit Property Essential Condition

📖 Text Input 🔍 Vision Input

📖 Text Input 🔍 Vision Input

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

A square pyramid is shown in the image. The height of this square pyramid is 5x+7. Length L is the side of the base of the pyramid. Write down an expression for L, in terms of the variable x.

Asoft drink can has a height of 13 cm and a radius of 3 cm. Find L, the length of the longest straw that can fit into the can (so that the straw is not bent and fits entirely inside the can).

[Figure 193]

[Figure 194]

Text Dominant

A square pyramid is shown in the inage. The height of this square pyramid is 5x+7. Write down an expression for L, in terms of the variable x.

[Figure 195]

Text Lite

[Figure 196]

The radius is 3 cm. Find L.

Asoft drink can has a height of 13 cm and a radius of 3 cm. Find L, the length of the longest straw that can fit into the can (so that the straw is not bent and fits entirely inside the can).

A square pyramid is shown in the image. The height of this square pyramid is 5x+7. Length L is the side of the base of the pyramid. Write down an expression for L, in terms of the variable x.

Text Only

[Figure 197]

[Figure 198]

Vision Intensive

The height of this solid is 5x+7. Write down an expression for L, in terms of the variable x.

The radius is 3 cm. Find L.

[Figure 199]

A square pyramid is shown in the image. Find an expression for L, in terms of the variable x.

[Figure 200]

Vision Dominant

Find L.

[Figure 201]

[Figure 202]

Vision Only

#### Figure 21: Comparison of Six Problem Versions in MATHVERSE.

[Figure 203]

Descriptive Information Implicit Property Essential Condition

📖 Text Input 🔍 Vision Input

📖 Text Input 🔍 Vision Input

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

As shown in the figure, it is known that AB and AD are the chords of circle O, angle BOD = 50° then the degree of angle BAD is (). Choices: ..

As shown in figure, in right triangle ABC, it is known that angle A = 90°, AC = 3.0,

[Figure 208]

[Figure 209]

Text Dominant

- AB = 4.0, then Sin B is. Choices: …

[Figure 210]

As shown in figure, in right triangle ABC,

- AC = 3.0, AB = 4.0, then Sin B is equal to. Choices: …

As shown in the figure, angle BOD = 50° then the degree of angle BAD is (). Choices: ..

[Figure 211]

Text Lite

As shown in the figure, it is known that AB and AD are the chords of circle O, angle BOD = 50° then the degree of angle BAD is (). Choices: ..

As shown in figure, in right triangle ABC, it is known that angle A = 90°, AC = 3.0, AB = 4.0, then Sin B is. Choices: …

Text Only

[Figure 212]

As shown in the figure, angle BOD = 50° then the degree of angle BAD is (). Choices: ..

As shown in figure, AC = 3.0, AB = 4.0, then Sin B is equal to.

[Figure 213]

Vision Intensive

Choices: …

[Figure 214]

[Figure 215]

As shown in the figure, the degree of angle BAD is (). Choices: ..

As shown in figure, in right triangle ABC, then Sin B is equal to. Choices: …

Vision Dominant

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

Vision Only

#### Figure 22: Comparison of Six Problem Versions in MATHVERSE.

[Figure 220]

A square prism has a side GF of length 15 cm as shown. Now, we want to find y, the length of the diagonal DF. Calculate y to two decimal places.

Answer: y = 21.61 Subject: Subfield:

Solid Geometry Length

[Figure 221]

###### GPT-4V:

✓ 1/1 ✓ 1/1

- 1. Consider the square prism as a combination of two right-angled triangles. Triangle DEF is one of them, and triangle DCF is the other.
- 2. Calculate the length of the diagonal EF of the bottom square face EFGH using the Pythagorean theorem. Since EF is the diagonal of a square of side length 15cm (side GF), we have: EF^2 = GF^2 + GE^2, EF^2 = 450. Now taking the square root to find EF: EF = √450 = 15√2 centimeters.
- 3. Find the diagonal DF which runs through the rectangular face DCEF. Triangle DEF is a right-angled triangle, hence we again apply Pythagorean theorem, with EF (just calculated) and DE being the legs, and DF being the hypotenuse. Since DE is equivalent in length to AB which is given as 11 cm, we have: DF^2 = DE^2 + EF^2 = 571.Taking the square root to find DF: DF = √571. Let's calculate the numeric value to two decimal places: DF ≈ 23.89 centimeters.
- 4. Therefore, y, the length of the diagonal DF, is approximately 23.89 cm.

× 0/1 × 0/1

[Figure 222]

###### Visual Perception Error

Multi-step Score: 0.5 Final CoT Evaluation Score: 0.35

LLaVA-NeXT:

- 1. Use the Pythagorean theorem to find the length of the diagonal. The theorem states that the square of the length of the hypotenuse (the diagonal in this case) is equal to the sum of the squares of the lengths of the legs. In this case, the legs are the sides of the square prism, which are of equal length.
- 2. Calculate the square of the lengths of the legs: y^2 = 11^2 + 11^2, y^2 = 121 + 121, y^2 = 242.
- 3. Take the square root of both sides to find the length of the diagonal: y = √242.
- 4. Round the square root to two decimal places and then multiply by itself to find the length of y to two decimal places: y = √242 ≈ 15.5, y * y ≈ 25.25.
- 5. The length of the diagonal DF is approximately 15.5 cm.

✓ 1/1

× 0/1 × 0/1 × 0/1 × 0/1

[Figure 223]

Numerical Calculation Error

Multi-step Score: 0.2 Final CoT Evaluation Score: 0.14

###### SPHINX-MoE:

✓ 1/1 ✓ 1/1

- 1.Use the Pythagorean theorem to find the length of the diagonal DF.
- 2.The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the lengths of the other two sides.
- 3.In this case, we have a right triangle with sides of length 11cm and 15cm. Let's call the length of the hypotenuse y.

.

- 4. Using the Pythagorean theorem, we have: y^2 = 11^2 + 15^2, y^2 = 121 + 225, y^2 = 346.
- 5. Taking the square root of both sides, we get: y = √346, y ≈ 18.60.
- 6. Therefore, the length of the diagonal DF is approximately 18.60 cm.

× 0/1 × 0/1 × 0/1 × 0/1

Multi-step Score: 0.33

Final CoT Evaluation Score: 0.23

- Figure 23: Response Comparison of GPT-4V [47], LLaVA-NeXT [38], and SPHINX-MoE [21]. We adopt the Text-lite version of the problem, and highlight the key-step extraction and scoring by the CoT evaluation strategy.

[Figure 224]

Find the volume of the cone shown. The perpendicular height is 6 cm. Round your answer to two decimal places.

Answer: Volume = 25.13 cm3 Subject:

Solid Geometry Volume

[Figure 225]

###### Subfield:

###### GPT-4V:

✓ 1/1

- 1.Use the formula for the volume of a cone: Volume = 1/3 * 𝜋 * r^2 * h, where r is the radius and h is the height of the cone.
- 2. Substitute the given values into the formula: Volume = 1/3 * 𝜋 * (2 cm)^2 * 6 cm, Volume = 1/3 * 𝜋 * 4 cm2 * 6 cm, Volume = 𝜋 * 4 cm2

* 2 cm, Volume = 8 𝜋 cm3.

- 3. Compute the volume numerically using 𝜋 ≈ 3.14159: Volume ≈ 8 * 3.14159, Volume ≈ 25.13272 cm3.
- 4. Round the result to two decimal places: Volume ≈ 25.13 cm3.
- 5. Final value: 25.13 cm3.

✓ 1/1 ✓ 1/1 ✓ 1/1 ✓ 1/1

Multi-step Score: 1 Final CoT Evaluation Score: 1

LLaVA-NeXT:

Reasoning Error

[Figure 226]

✓ 1/1

- 1. Use the formula for the volume of a cone: V = (1/3) 𝜋 r^2 h, where r is the radius and h is the height of the cone.
- 2. Convert the height from centimeters to meters. There are 100 centimeters in a meter, so 6 centimeters is equal to 0.06 meters.
- 3. Plug the values into the formula: V = (1/3) 𝜋 (2^2) (0.06), V = (1/3) 𝜋 (4) (0.06), V = (1/3) 𝜋 (4) (0.06) m3, V = 0.00064 m3.
- 4. Round to two decimal places. The volume of the cone is approximately 0.00064 cubic meters.

× 0/1 × 0/1 × 0/1

Multi-step Score: 0.25 Final CoT Evaluation Score: 0.175

SPHINX-MoE:

Knowledge Error

[Figure 227]

✓ 1/1

- 1. Use the formula for the volume of a cone: V = (1/3) 𝜋 r^2 h, where V is the volume, r is the radius, and h is the height.
- 2. Given the height h = 6 cm but not the radius r. Use the fact that the radius is equal to the height of the triangle formed by the cone's slant height and the height of the cone.
- 3. The slant height is the hypotenuse of a right triangle with legs of length 6 cm and 2 cm (the radius of the base of the cone). Use the Pythagorean theorem to find the slant height: a^2 = b^2 + c^2, where a is the hypotenuse (slant height), b is the height of the cone (6 cm), and c is the radius of the base of the cone.
- 4. Calculate the slant height: a^2 = 6^2 + 2^2, a^2 = 36 + 4, a^2 = 40, a = √40.
- 5. Use the Pythagorean theorem again to find the radius (r = 2 cm): r^2 = a^2 - b^2, r^2 = (√40)^2 - 6^2 = 40 - 36, r^2 = 4, r = √4.
- 6. Now that we have the radius (r = 2 cm), we can find the volume of the cone: V = (1/3) 𝜋 r^2 h, V = (1/3) 𝜋 (2)^2 (6), V = (1/3) 𝜋

(4)(6), V = (1/3) 𝜋 (24), V = 8𝜋.

- 7. The volume of the cone is approximately 25.13 cubic centimeters.

× 0/1 × 0/1 × 0/1 × 0/1 × 0/1

✓ 1/1

Multi-step Score: 0.29 Final CoT Evaluation Score: 0.5

- Figure 24: Response Comparison of GPT-4V [47], LLaVA-NeXT [38], and SPHINX-MoE [21]. We adopt the Text-lite version of the problem, and highlight the key-step extraction and scoring by the CoT evaluation strategy.

Question:

[Figure 228]

There is an isosceles triangle. Find the area of the figure. Choices: A:30 B:60 C:120 D:240

Answer: B Subject:

Plane Geometry Area

[Figure 229]

###### Subfield:

###### GPT-4V:

✓ 1/1

- 1. Use the formula for the area of a triangle, which is ( 1/2 * base * height).
- 2. Determine the height h of the isosceles triangle with two equal sides (each with length 13) and a base (length 10).
- 3. Drop a perpendicular from the vertex between the equal sides to the midpoint of the base to create two right-angled triangles. This splits the base into two segments of length 5 each.
- 4. Use the Pythagorean theorem for one of these right-angled triangles: h^2 + 5^2 = 13^2, h^2 = 13^2 - 5^2, h^2 = 169 – 25, h^2 = 144. So, h = √144. Therefore, h = 12.
- 5. Find the area of the isosceles triangle: Area = (1/2 * base * height), Area = (1/2 * 10 * 12), Area = 5 * 12, Area = 60.
- 6. The area of the isosceles triangle is 60. The correct option is B: 60.

✓ 1/1 ✓ 1/1 ✓ 1/1 ✓ 1/1

✓ 1/1

- Multi-step Score: 0.5 Final CoT Evaluation Score: 0.35
- Multi-step Score: 1 Final CoT Evaluation Score: 1

LLaVA-NeXT:

✓ 1/1

- 1. Calculate the area of an isosceles triangle using the formula: Area = (1/2) * Base * Height.
- 2. In the given problem, the base of the triangle is 10 units long, and the height is 13 units long.
- 3. Substitute the given values into the formula: Area = (1/2) * 10 * 13, Area = 13 * 7, Area = 91.
- 4. The final answer is 91.

✓ 1/1

× 0/1 × 0/1

[Figure 230]

Numerical Calculation Error

SPHINX-MoE:

Visual Perception Error

[Figure 231]

✓ 1/1

- 1. The area of an isosceles triangle is given by the formula A = (1/2) * b * h, where b is the base length and h is the height.
- 2. In this case, the base length is 10 and the height is 13.
- 3. Plugging these values into the formula, we get A = (1/2) * 10 * 13 = 65.
- 4. Therefore, the area of the triangle is 65. The correct option is B: 60.

× 0/1 × 0/1 × 0/1

Multi-step Score: 0.25 Final CoT Evaluation Score: 0.475

- Figure 25: Response Comparison of GPT-4V [47], LLaVA-NeXT [38], and SPHINX-MoE [21]. We adopt the Text-lite version of the problem, and highlight the key-step extraction and scoring by the CoT evaluation strategy.

[Figure 232]

Question:

Find the value of h to the nearest metre.

Answer: 9 Subject:

Plane Geometry Applied

[Figure 233]

###### Subfield:

###### GPT-4V:

- 1. Use trigonometry, specifically the tangent function, to relate the opposite side of the angle in the triangle (which is h, the height of the tower) to the adjacent side (which is the horizontal distance from the base of the tower).
- 2. For the smaller angle of 10 degrees, the tower with the known height creates the opposite side, and the 29-meter distance is the adjacent side. However, we need to find the height of the taller tower, for which we have a 22-degree angle.
- 3. Use the tangent function for the 22-degree angle: tan(22°) = opposite/adjacent tan(22°) = h/29.
- 4. Solve for h: h = 29 * tan(22°).
- 5. Calculate this with a calculator: h ≈ 29 * 0.4040 (tan(22°) ≈ 0.4040) h ≈ 11.716 metres.
- 6. Round to the nearest metre: h ≈ 12 metres Final value: 12.

✓ 1/1

✓ 1/1 ✓ 1/1 ✓ 1/1 ✓ 1/1

✓ 1/1

- Multi-step Score: 0 Final CoT Evaluation Score: 0

Multi-step Score: 0.3 Final CoT Evaluation Score: 0.2

- Multi-step Score: 1 Final CoT Evaluation Score: 1

LLaVA-NeXT:

✓ 1/1

- 1. Determine the height of the tower relative to the person's height by measuring the distance between the person's shadow and the tower.
- 2. Assume that the distance between the person and the tower is equal to the length of the shadow.
- 3. Use the formula to find the length of the shadow: Shadow length = (person's height) x tan(angle of light).
- 4. In this case, the angle of light is approximately 90 degrees, so the shadow length is: Shadow length = 2.2 meters x tan(90 degrees)

= 2.2 meters x 1 = 2.2 meters.

- 5. Find the height of the tower relative to the person's height by subtracting the shadow length from the tower's height: h = 2.5 meters - 2.2 meters = 0.3 meters.
- 6. The value of h is approximately 0.3 meters, or 30 centimeters.

× 0/1

✓ 1/1

× 0/1

[Figure 234]

Visual Perception Error

× 0/1 × 0/1

[Figure 235]

Reasoning Error

SPHINX-MoE:

× 0/1

1. The single answer is that the value of h is 30.

- Figure 26: Response Comparison of GPT-4V [47], LLaVA-NeXT [38], and SPHINX-MoE [21]. We adopt the Text-lite version of the problem, and highlight the key-step extraction and scoring by the CoT evaluation strategy.

Question:

[Figure 236]

A polynomial p is graphed. What could be the equation of p ? Choices:

- A: p(x)=(x+1)2(2x+5)2(x-3)2
- B: p(x)=(x+1)2(2x+5)(x-3)2
- C: p(x)=(x+1)2(2x+5)(x-3)
- D: p(x)=(x+1)(2x+5)(x-3)2

[Figure 237]

Answer: B Subject: Subfield:

Functions Expression

###### GPT-4V:

✓ 1/1

- 1. Analyze the graph for key features such as x-intercepts (roots) and the behavior of the graph at those intercepts.
- 2. Identify the x-intercepts: at x = -1, x = -2.5 (which is -5/2 when converted to a fraction), and x = 3. This tells us that the factors of the polynomial must include (x + 1), (x + 5/2), and (x - 3).
- 3. Look at the behavior of the graph at each intercept to determine the multiplicity of each root. The graph touches the x-axis and turns around at x = -1 and x = 3, which indicates that these roots are of even multiplicity. The graph crosses the x-axis at x = -2.5, which indicates that this root is of odd multiplicity.
- 4. Match the graph to the correct equation by comparing the multiplicity of the roots with the given options.
- 5. The correct equation that matches the graph is: B: p(x)=(x+1)2(2x+5)(x-3)2.

✓ 1/1

✓ 1/1

✓ 1/1 ✓ 1/1

- Multi-step Score: 0.2 Final CoT Evaluation Score: 0.14
- Multi-step Score: 1 Final CoT Evaluation Score: 1

LLaVA-NeXT:

Visual Perception Error

[Figure 238]

✓ 1/1

- 1. Analyze the graph and identify the degree of the terms in the polynomial.
- 2. Identify that the polynomial has a degree of 2 for the terms with the highest degree, which are the squares of the linear terms.
- 3. Identify the highest degree term that is not squared is the term with the variable raised to the second power, which is (x+1)2.
- 4. Identify the next highest degree term is (2x+5)2, and the lowest degree term is (x-3)2.
- 5. Conclude that the equation of the polynomial p(x) should be of the form: p(x) = (x+1)2(2x+5)2(x-3)2 This is option A.

× 0/1

× 0/1 × 0/1 × 0/1

###### SPHINX-MoE:

× 0/1

- 1. The graph of p has a vertical tangent at x=-1.
- 2.The graph has a horizontal tangent at x=3.
- 3. The graph has a vertical tangent at x=-5/2.
- 4.The graph of p also passes through the point (-1,0).
- 5. The equation of p is p(x)=(x+1)2(2 x+5)(x-3)2.
- 6. The correct option is A.

Visual Perception Error

[Figure 239]

× 0/1 × 0/1

✓ 1/1

× 0/1 × 0/1

Multi-step Score: 0.17 Final CoT Evaluation Score: 0.12

- Figure 27: Response Comparison of GPT-4V [47], LLaVA-NeXT [38], and SPHINX-MoE [21]. We adopt the Text-lite version of the problem, and highlight the key-step extraction and scoring by the CoT evaluation strategy.

[Figure 240]

Question:

The function A models the rectangle‘s area (in square meters) as a function of its width (in meters). Which of these statements are true? Choices:

A: Greater width always relates to smaller area. B: Greater width relates to smaller area as long as the width is less than 10 m. C: When there is no width, the area is 20 m2. D: When there is no width, the area is 0 m2.

[Figure 241]

Answer: D

Subject: Functions

###### Subfield: Applied

###### GPT-4V:

✓ 1/1 1.Analyze the graph of the function A, which models the area of a rectangle as a function of its width.

- 2. Evaluate each statement based on the graph:

- - A: Greater width always relates to a smaller area.
- - B: The area is always increasing with the width.
- - C: When the width is 0, the area is not 0.
- - D: When there is no width, the area is 0 m2.

- 3. Determine that statement D is true because the graph starts at the origin (0,0), indicating that when there is no width, there is no area, which is consistent with the properties of a rectangle.
- 4. The correct option is: D: When there is no width, the area is 0 m2.

✓ 1/1

✓ 1/1

✓ 1/1

Multi-step Score: 1 Final CoT Evaluation Score: 1

LLaVA-NeXT:

[Figure 242]

Visual Perception Error

✓ 1/1

- 1. Analyze the given information and the graph of the quadratic function A.
- 2. Observe that the vertex of the quadratic function is at (10, 100), indicating that the function has a minimum point at this point.
- 3. Note that the quadratic function intersects the x-axis at (0,0) and (20,0), which means the area is zero when the width is zero.
- 4. Consider the given statements and determine their correctness based on the observations from the graph.
- 5. Statement A is incorrect as the area does not always decrease with increasing width.
- 6. Statement B is correct as the area decreases with increasing width up to the vertex, which is at 10 meters.
- 7. Statement C is incorrect as the area is zero when the width is zero, not 20 square meters.
- 8. Statement D is correct as the area is zero when the width is zero.
- 9. The correct statement is B: Greater width relates to smaller area as long as the width is less than 10 m.

× 0/1

✓ 1/1

✓ 1/1

✓ 1/1

× 0/1

✓ 1/1 ✓ 1/1

× 0/1

- Multi-step Score: 0.67 Final CoT Evaluation Score: 0.47
- Multi-step Score: 1 Final CoT Evaluation Score: 1

SPHINX-MoE:

✓ 1/1

1. The single answer is D.

- Figure 28: Response Comparison of GPT-4V [47], LLaVA-NeXT [38], and SPHINX-MoE [21]. We adopt the Text-lite version of the problem, and highlight the key-step extraction and scoring by the CoT evaluation strategy.

[Figure 243]

(Text Dominant)

###### Question:

There is a curve goes through (-4,1), (-1,3), and (0,0). Estimate the intervals on which the polynomial function is increasing or decreasing.

Answer: The function is increasing on (-∞,-2.5) ∪ (1, ∞), and decreasing on (-2.5,1) Subject: Subfield:

Functions

[Figure 244]

Property

###### Text Dominant:

✓ 1/1

- 1. Determine the intervals on which the function is increasing or decreasing by looking at the slope of the curve on the graph.
- 2. The function starts off decreasing from the point (-5, something above 6) to the point (-4, 1).
- 3. The function then starts increasing at the point (-4, 1) and continues to increase until it reaches a peak somewhere between -2 and -1 on the x-axis.
- 4. After reaching the peak, the function starts to decrease again, and it continues to decrease past the point (-1, 3) until it reaches the point (0, 0).
- 5. The function reaches its minimum at (0, 0) and then starts to increase again. It continues to increase past the point (2, something below -3) and keeps increasing as it moves off to the right of the graph.
- 6. The intervals on which the function is increasing or decreasing are as follows:

- - Decreasing: from the start of the graph to x = -4, and from just before x = -2 to x = 0.
- - Increasing: from x = -4 to just before x = -2, and from x = 0 to the end of the graph.

- 7. More precisely:

- - The function is increasing on the intervals (-4, -1.5) and (0, ∞).
- - The function is decreasing on the intervals (-∞, -4) and (-1.5, 0).

- 8. Note that the exact turning points are estimated since the graph does not provide precise values, and the intervals are given in terms of x-values. The exact values of the turning points cannot be determined without more information or a more precise graph.

× 0/1 × 0/1

× 0/1 × 0/1

× 0/1 × 0/1

✓ 1/1

Multi-step Score: 0.25 Final CoT Evaluation Score: 0.175

###### Text Lite:

✓ 1/1 ✓ 1/1

- 1. Determine the slope of the function as it moves from left to right along the x-axis.
- 2. Identify the points where the function reaches its peaks and its lowest points.
- 3. The function is increasing from the leftmost part of the graph until it reaches its first peak, which is approximately at x = -3.
- 4. The function starts decreasing after this peak until it reaches its lowest point, which is approximately at x = 0.
- 5. The function then starts increasing again from x = 0 until it reaches its next peak, which is approximately at x = 2.
- 6. The function decreases again after x = 2 towards the rightmost part of the graph.
- 7. The intervals on which the function is increasing are from x = -5 to x = -3 and from x = 0 to x = 2. The intervals on which the function is decreasing are from x = -3 to x = 0 and from x = 2 to x = 4.
- 8. Final values: Increasing intervals: (-5, -3) and (0, 2), Decreasing intervals: (-3, 0) and (2, 4).

× 0/1 × 0/1 × 0/1 × 0/1

× 0/1

× 0/1

Multi-step Score: 0.25 Final CoT Evaluation Score: 0.175

Vision Dominant:

✓ 1/1

- 1. Determine the intervals on which the function is increasing or decreasing by looking at the slope of the function as it moves from left to right along the x-axis.
- 2. From the graph, identify that the function is increasing from the leftmost part of the graph until it reaches its first peak. This occurs from approximately x = -5 to x = -2.5.
- 3. After reaching the first peak, the function starts to decrease until it reaches the bottom of the first trough. This occurs from approximately x = -2.5 to x = 0.
- 4. The function then starts to increase again from the bottom of the trough until it reaches the second peak. This occurs from

- approximately x = 0 to x = 1.

5. Finally, the function decreases again after the second peak towards the rightmost part of the graph. This occurs from

- approximately x = 1 to x = 4.

- 6. Conclude that the function is increasing on the intervals: (-5, -2.5) and (0, 1)
- 7. And the function is decreasing on the intervals: (-2.5, 0) and (1, 4)

× 0/1 × 0/1 × 0/1 × 0/1 × 0/1 × 0/1

Multi-step Score: 0.14 Final CoT Evaluation Score: 0.1

- Figure 29: Response of Different Problem Versions by GPT-4V [47]. By comparing the output from three versions, we observe increasing reasoning errors when more textual content is visually depicted in diagrams. We only showcase the Text-dominant version of the problem for brevity.

(Text Dominant)

Question:

[Figure 245]

Sia is swinging from a chandelier. The horizontal distance between Sia and the wall, in meters, is modeled by D(t) where t is the time in seconds. The function is graphed below, along with one segment highlighted, a red double arrow from (3.75,3), (3.75,5). What is the meaning of the highlighted segment? Choices:

- A:The chandelier is hung 2 meters from the wall.
- B:The furthest Sia gets from the point where the chandelier hangs is 2 meters.
- C:The chandelier is 2 meters wide.
- D:Sia completes 2 swing cycles per second.

[Figure 246]

Answer: B Subject: Functions Subfield: Applied

###### Text Dominant:

✓ 1/1

- 1. The graph shows a function D(t) that represents the horizontal distance between Sia and the wall over time.
- 2. The highlighted segment with a red double arrow spans vertically from the point (3.75, 3) to the point (3.75, 5) on the graph.
- 3. This segment is at a constant time of 3.75 seconds and shows a change in the distance D(t) from 3 meters to 5 meters.
- 4. The segment does not provide information about the width of the chandelier or the number of swing cycles per second. Instead, it indicates a change in distance at a specific moment in time.
- 5. Since the distance changes from 3 to 5 meters, this represents a 2-meter difference in the horizontal distance from the wall during the swing.
- 6. Therefore, the meaning of the highlighted segment is that the furthest Sia gets from the point where the chandelier hangs is 2 meters.
- 7. The correct option is: B: The furthest Sia gets from the point where the chandelier hangs is 2 meters.

✓ 1/1

× 0/1

× 0/1

✓ 1/1

✓ 1/1

✓ 1/1

Multi-step Score: 0.71 Final CoT Evaluation Score: 0.8

###### Text Lite:

✓ 1/1 ✓ 1/1

- 1. Analyze the graph which shows a periodic function, representing the back-and-forth motion of Sia swinging from the chandelier. The horizontal axis is labeled with time t in seconds, and the vertical axis is labeled with distance D in meters. The highlighted segment on the graph indicates a change in the distance from one point to another over time.
- 2. Evaluate each choice:

- - A: The chandelier is hung 2 meters from the wall. This statement would imply a fixed distance from the wall, which the graph does not provide information about.
- - B: The furthest Sia gets from the point where the chandelier hangs is 2 meters. This statement seems to be about the amplitude of the swing, but the highlighted segment does not indicate the amplitude; it indicates a change in distance over time.
- - C: The chandelier is 2 meters wide. The width of the chandelier is not something that can be determined from this graph, which shows distance from the wall over time.
- - D: Sia completes 2 swing cycles per second. The highlighted segment shows a change in distance over time, but it does not provide information about the number of cycles per second.

- 3. Conclude that none of the options A, B, C, or D directly describe the meaning of the highlighted segment in terms of the graph's representation of distance over time.
- 4. Given the options provided and the information from the graph, none of the choices accurately describe the meaning of the highlighted segment. Therefore, based on the information provided and the reasoning above, there is no correct option to choose from A, B, C, or D.

✓ 1/1

× 0/1

✓ 1/1 ✓ 1/1

× 0/1

× 0/1

Multi-step Score: 0.63 Final CoT Evaluation Score: 0.44

Vision Dominant:

✓ 1/1

- 1. The graph shows a periodic function, representing the motion of a pendulum or a swing. The horizontal axis represents time (t), and the vertical axis represents the horizontal distance (D) from the wall. The highlighted segment indicates a change in the distance D over time.
- 2. The highlighted segment is a vertical line, showing a change in distance at a particular moment in time, from a lower value to a higher value on the D(t) function.
- 3. None of the options A, B, C, or D directly relate to the meaning of a single vertical segment on the graph. The closest interpretation of the highlighted segment would be the change in distance from the wall at a specific moment, which is not explicitly described in any of the options provided.
- 4. Given the information and the graph, none of the options A, B, C, or D correctly describe the meaning of the highlighted segment.

× 0/1 × 0/1

× 0/1

Multi-step Score: 0.25 Final CoT Evaluation Score: 0.175

- Figure 30: Response of Different Problem Versions by GPT-4V [47]. By comparing the output from three versions, we observe increasing reasoning errors when more textual content is visually depicted in diagrams. We only showcase the Text-dominant version of the problem for brevity.

[Figure 247]

(Text Dominant)

Question:

As shown in the figure, Xiaoming designed two right angles to measure the width of the river BC, he measured AB = 2.0, BD = 7/3, CE = 9.0, then the width of the river BC is () Choices:

- A:7 / 2
- B:17 / 2
- C:40 / 7
- D:11

[Figure 248]

Answer: C Subject: Plane Geometry Subfield:

Applied

###### Text Dominant:

✓ 1/1

- 1. Identify that the triangles ABD and CEB are similar because they both have a right angle, and they share the angle at point B.
- 2. Use the property of similar triangles that the ratios of their corresponding sides are equal:AB/BD = BC/CE.
- 3. Substitute the given values into the ratio: 2.0 / (7.0/3.0) = BC / 9.0.
- 4. Cross-multiply to solve for BC: 2.0 * 9.0 = BC * (7.0/3.0).
- 5. Solve for BC by dividing both sides by (7.0/3.0): BC = 18.0 / (7.0/3.0).
- 6. Simplify the equation: BC = 18.0 * (3.0/7.0).
- 7. Calculate the final value: BC = 54.0 / 7.0.
- 8. Round the result to the closest answer: BC = 7.71428571...
- 9. The correct option is: C: 40 / 7 meters.

✓ 1/1

✓ 1/1 ✓ 1/1 ✓ 1/1 ✓ 1/1 ✓ 1/1 ✓ 1/1 ✓ 1/1

Multi-step Score: 1 Final CoT Evaluation Score: 1

###### Text Lite:

✓ 1/1 ✓ 1/1

- 1. Identify that the triangles ABD and EBC are similar because they both have a right angle, and they share the angle at point B.
- 2. Find the ratio between the sides of the triangles ABD and EBC using the given lengths AB and BD. The ratio of AB to BD is 2.0 / (7.0/3.0) = 6.0/7.0.
- 3. Set up a proportion using the sides of the similar triangles: AB/BD = BC/CE.
- 4. Substitute the known values and the ratio into the proportion: (6.0/7.0) = BC/9.0.
- 5. Solve for BC: BC = (6.0/7.0) * 9.0, BC = 54.0/7.0, BC = 7.71428571...
- 6. Compare the calculated value with the provided options. The closest value to this result among the choices given is 40/7 meters, which is approximately 5.71428571 meters. However, none of the options exactly match the calculated value. There might be a mistake in the calculation or in the provided options. Given the options, the closest one is C: 40/7 m.

✓ 1/1 ✓ 1/1

✓ 1/1 × 0/1

Multi-step Score: 0.83 Final CoT Evaluation Score: 0.58

###### Vision Dominant:

✓ 1/1

- 1. Identify the similar triangles in the figure (Triangle ABD and triangle EBC).
- 2. Note the lengths of the known sides in each triangle.
- 3. Since the triangles are similar, the corresponding sides are proportional. Write the proportions as: AB/BD = CE/BC.
- 4. Plug in the known values and solve for BC: BC = 9.0 * (7.0/3.0) / 2.0 BC = 10.5.
- 5. The correct answer is not directly listed among the choices. However, since 10.5 meters can be written as 21/2 meters, which is identical to 7/2 times 3 meters, it seems that there might have been a miscalculation in the options or I'm misinterpreting the figure based on the given variables. Please verify the options provided or the figure details to ensure an accurate answer can be given. Assuming the options provided contain a typo and closest to 10.5 (or 21/2) meters is the fraction that also involves 7, the most sensible choice if we round 10.5 to the nearest half is:

- A: 7/2 meters (which is 3.5 meters and not the correct answer).

6. Given the absence of an accurate option and assuming the calculations are correct, there should be an option that reflects 21/2 meters, which is not present in the given choices. If we are to choose from the closest available option, it would be:

- B: 17/2 meters.

- 7. However, 17/2 meters equals 8.5 meters, which is not the result we have calculated. Therefore, there seems to be an error in the options or in the figures provided which needs to be addressed for an accurate selection. Please review the question or the choices for any discrepancies.

✓ 1/1 ✓ 1/1

× 0/1 × 0/1

× 0/1

× 0/1

Multi-step Score: 0.43 Final CoT Evaluation Score: 0.3

- Figure 31: Response of Different Problem Versions by GPT-4V [47]. By comparing the output from three versions, we observe increasing reasoning errors when more textual content is visually depicted in diagrams. We only showcase the Text-dominant version of the problem for brevity.

(Text Dominant)

[Figure 249]

Question:

As shown in the figure, triangle ABC is inscribed in circle O, OC perpendicular OB, OD perpendicular AB intersects AC at point E. Knowing that the radius of circle O is 1.0, then what is the value of AE2 + CE2? Choices:

- A:1
- B:2
- C:3
- D:4

[Figure 250]

Answer: B Subject: Plane Geometry Subfield:

Length

###### Text Dominant:

✓ 1/1

- 1. Identify that OC is perpendicular to OB, making triangle OCB a right triangle with OC and OB as its legs. Since OC and OB are radii of the circle and the radius is given as 1.0, OC = OB = 1.0. Therefore, triangle OCB is an isosceles right triangle.
- 2. Identify that OD is perpendicular to AB, which means that OD is the altitude of triangle AOB from point O to side AB. Since triangle AOB is also isosceles (OA = OB = 1.0), OD bisects AB. This means that AD = DB.
- 3. Consider triangle AOD and triangle BOD. Since OD is perpendicular to AB, both triangles AOD and BOD are right triangles. Also, since AD = DB, triangle AOD is congruent to triangle BOD. Therefore, AE = EB.
- 4. Since AE = EB and AC = AB (because triangle ABC is inscribed in the circle and AC and AB are both radii), we can say that CE = EB.
- 5. To find AE^2 + CE^2, since AE = CE, we can say that AE^2 + CE^2 = 2 * AE^2.
- 6. To find AE, use the Pythagorean theorem in triangle AOD (or BOD, since they are congruent). We have: AO^2 = AD^2 + OD^2.

Since AO = 1.0 (radius of the circle) and AD = 1/2 * AB (because AD = DB and AB is the diameter of the circle, which is twice the radius), we have:

- 1.0^2 = (1/2 * AB)^2 + OD^2.

AB is the diameter of the circle, so AB = 2 * radius = 2 * 1.0 = 2.0. Therefore: 1.0^2 = (1/2 * 2.0)^2 + OD^2 OD^2 = 0 Since OD^2 = 0, OD must be 0, which means that point D coincides with point O, and AE = EB = 1.0 (since AE = EB and AB = 2 * radius =

- 2.0). Therefore, AE^2 + CE^2 = 2 * AE^2 = 2 * 1.0^2 = 2.

- 7. The correct answer is: B: 2

✓ 1/1

✓ 1/1

× 0/1 × 0/1 × 0/1

✓ 1/1

Multi-step Score: 0.57 Final CoT Evaluation Score: 0.7

###### Text Lite:

✓ 1/1 ✓ 1/1

- 1. Use the properties of a circle and the Pythagorean theorem.
- 2. Since OC is perpendicular to OB and both are radii of the circle, triangle OCB is an isosceles right triangle. Therefore, the length of BC can be found using the Pythagorean theorem: BC = √2
- 3. Similarly, since OD is perpendicular to AB, triangle OAD is also an isosceles right triangle. Therefore, AD = OD = 1.
- 4. Using the Pythagorean theorem again, AO = √2. Since AO is the diameter of the circle, AC is also the diameter, and thus AC = AO = √2.
- 5. We have AC = √2 and BC = √2. Since E is a point on AC, AE + EC = AC.
- 6. We want to find AE^2 + CE^2. Let's denote AE as x and CE as (√2 - x). Then we can express AE^2 + CE^2 as: 2x^2 - 2√2x + 2.
- 7. To minimize this expression, we can take the derivative with respect to x and set it to zero to find the minimum value. However, since we are given that AE and CE are segments of a line, and we know that the sum of their squares in a right triangle is constant, we can infer that the sum of the squares of AE and CE will be equal to the square of AC.
- 8. Therefore, AE^2 + CE^2 = AC^2 = (√2)^2 = 2.
- 9. The correct answer is: B: 2

✓ 1/1

× 0/1

× 0/1

× 0/1

× 0/1

✓ 1/1

✓ 1/1

Multi-step Score: 0.56 Final CoT Evaluation Score: 0.69

Vision Dominant:

✓ 1/1

- 1. Given that the radius of circle O is 1, this means that OA, OE, and OC are all radii of the circle, and thus they all have a length of 1.
- 2. Use the Pythagorean theorem (a^2 + b^2 = c^2) on triangle AOE, where AE is the hypotenuse, and OA and OE are the two legs: AE^2 = OA^2 + OE^2, AE^2 = 2.
- 3. By the same logic, apply the Pythagorean theorem to triangle COE, where CE is the hypotenuse: CE^2 = OC^2 + OE^2, CE^2 = 2.
- 4. Now, according to the question, find the value of AE^2 + CE^2: AE^2 + CE^2 = 2 + 2 AE^2 + CE^2 = 4.
- 5. Therefore, the value of AE^2 + CE^2 is 4. The correct option is D: 4.

✓ 1/1

✓ 1/1

× 0/1

× 0/1

Multi-step Score: 0.6 Final CoT Evaluation Score: 0.42

- Figure 32: Response of Different Problem Versions by GPT-4V [47]. By comparing the output from three versions, we observe increasing reasoning errors when more textual content is visually depicted in diagrams. We only showcase the Text-dominant version of the problem for brevity.

[Figure 251]

(Text Dominant)

Question:

Consider the solid pictured and answer the following: Hence what is the total surface area? The outer radius is 11 cm, inner radius is 10 cm and length 24 cm. Give your answer to the nearest two decimal places.

Answer: SA = 3298.67 cm2 Subject: Subfield:

Solid Geometry Area

[Figure 252]

Text Dominant

✓ 1/1 ✓ 1/1

- 1. Calculate the surface area of the outer side of the cylinder using the formula for the lateral surface area of a cylinder: 2𝜋 rh, where r is the outer radius and h is the length (height) of the cylinder. Outer surface area = 2 𝜋 * 11 * 24.
- 2. Calculate the surface area of the inner side of the cylinder using the same formula, but with the inner radius. Inner surface area = 2 𝜋 * 10 * 24.
- 3. Calculate the surface area of both circular ring-shaped ends. This is the area of the outer circle minus the area of the inner circle, multiplied by 2 (since there are two ends). Area for one end = Area of outer circle - Area of inner circle = 𝜋router^2 - 𝜋rinner^2 = 𝜋11^2 𝜋10^2. Total area for both ends = 2 * (𝜋(11^2 - 10^2)).
- 4. Add all these together to get the total surface area: Total surface area = 528𝜋 + 480𝜋 + 42𝜋 = 1050𝜋.
- 5. Use the approximate value of π, which is 3.14, to calculate the total surface area in decimal form: Total surface area ≈ 1050 * 3.14.
- 6. Round the result to the nearest two decimal places: 3297.00.
- 7. The total surface area of the solid, to the nearest two decimal places, is approximately 3297.00 square centimeters.

✓ 1/1

× 0/1

✓ 1/1

× 0/1

× 0/1

Multi-step Score: 0.57 Final CoT Evaluation Score: 0.4

Text Lite

- 1. Calculate the area A of a circle using the formula A = πr^2, where r is the radius. In this case, the radius r is 5 cm. Therefore, the area of one circular base is π(5^2) or 25π.
- 2. Since a cylinder has two bases (top and bottom), multiply the area of one base by 2 to find the total area of both bases: 2 × 25π = 50π cm².
- 3. To find the area of the curved surface of the cylinder, use the formula for the lateral surface area of a cylinder, which is A = 2πrh, where h is the height or length of the cylinder. In this case, the height h is 24 cm. The lateral surface area would therefore be 2πrh = 2π(5)(24) = 240π cm².
- 4. Add both calculated areas together to find the total surface area: Total surface area = area of both bases + lateral surface area. Total surface area = 50π + 240π = 290π cm².
- 5. Use the approximation π ≈ 3.14159 to compute the numerical answer: Total surface area ≈ 290 × 3.14159. Total surface area ≈ 910.8611 cm².

× 0/1 × 0/1 × 0/1 × 0/1 × 0/1

× 0/1 6.Roundedtothenearesttwodecimalplaces,thetotalsurfaceareais910.86cm².

Multi-step Score: 0 Final CoT Evaluation Score: 0

Vision Dominant

- 1. Find the area of one circular base using the formula A = πr², where r is the radius of the base. In this case, the radius is 5 cm, so the area of one base is π * 25 cm² = 78.54 cm² (to two decimal places).
- 2. Since there are two circular bases, double the result to get the total area of the bases: 2 * 78.54 cm² = 157.08 cm².
- 3. Calculate the rectangular surface area, which is the curved surface that wraps around the cylinder. The length of the rectangle is the same as the circumference of the base circle, which can be found with the formula C = 2πr. In this case, the circumference is 2 * π * 5 cm = 31.42 cm (to two decimal places).
- 4. The width of the rectangle is the same as the height (length) of the cylinder, which is given as 24 cm. So the area of the rectangle is 31.42 cm * 24 cm = 753.96 cm² (to two decimal places).
- 5. Add the areas of both circles and the rectangular wrap to get the total surface area: 157.08 cm² + 753.96 cm² = 911.04 cm².
- 6. The total surface area of the cylinder to the nearest two decimal places is 911.04 cm² .

× 0/1 × 0/1 × 0/1 × 0/1 × 0/1 × 0/1

Multi-step Score: 0

Final CoT Evaluation Score: 0

- Figure 33: Response of Different Problem Versions by GPT-4V [47]. By comparing the output from three versions, we observe increasing reasoning errors when more textual content is visually depicted in diagrams. We only showcase the Text-dominant version of the problem for brevity.

(Text Dominant)

Question:

[Figure 253]

Find the volume of the cone pictured. The perpendicular height is 6 and radius is 3. (Give your answer correct to 2 decimal places.)

Answer: 56.55 Subject: Subfield:

Solid Geometry Volume

[Figure 254]

Text Dominant

- 1. Use the formula for the volume of a cone: V = 1/3𝜋r^2h , where V is the volume, r is the radius of the base, and h is the height of the cone.
- 2. Given the radius r = 3 and the height h = 6, plug these values into the formula: V = 1/3𝜋(3)^2 (6) , V = 1/3 𝜋 ∗ 9 * 6 , V = 18 𝜋.
- 3. Calculate the numerical value using the approximation 𝜋 ≈ 3.14159: V ≈ 18 * 3.14159 , V ≈ 56.54862 𝜋.
- 4. Round to two decimal places, the volume of the cone is approximately: V ≈ 56.55.
- 5. Final Value: 56.55.

✓ 1/1 ✓ 1/1 ✓ 1/1 ✓ 1/1 ✓ 1/1

Multi-step Score: 1 Final CoT Evaluation Score: 1

Text Lite

✓ 1/1 ✓ 1/1 ✓ 1/1 ✓ 1/1 ✓ 1/1 ✓ 1/1

- 1. Use the formula for the volume of a cone: Volume = 1/3 𝜋 r^2 h, where r is the radius of the base of the cone, and h is the height.
- 2. Assume the "3" refers to the radius of the base of the cone, and the perpendicular height provided in the question text is "6". These values are the radius and height respectively.
- 3. Substitute r = 3 and h = 6 into the formula: Volume = 1/3 𝜋 (3)^2 (6), Volume = 1/3 𝜋 (9) (6), Volume = 1/3 𝜋 (54), Volume = 18 𝜋.
- 4. Calculate the numeric value using the approximation 𝜋 ≈ 3.14159: Volume = 18 * 3.14159, Volume ≈ 56.54862.
- 5. Round to two decimal places, the volume of the cone is: Volume ≈ 56.55.
- 6. Final value: 56.55.

Multi-step Score: 1 Final CoT Evaluation Score: 1

Vision Dominant

✓ 1/1 ✓ 1/1

- 1. Identify the shape of the solid figure. In this case, it is a cone.
- 2. Use the formula for the volume of a cone: V = 1/3𝜋r^2 h, where V is the volume, r is the radius of the base, h is the height of the cone, and \pi is the constant approximately equal to 3.14159.
- 3. Identify the values for r and h from the image. Here, h = 6 and the diameter of the base is 3, which means the radius r = 3/2 = 1.5.
- 4. Substitute the values into the formula: V = 1/3𝜋(1.5)^2 * 6, V = 1/3𝜋* 2.25 * 6, V = 1/3𝜋* 13.5, V = 𝜋 * 4.5, V ≈ 3.14159 * 4.5, V ≈ 14.13765.
- 5. Round the final answer to two decimal places. The volume is 14.14 cubic units.
- 6. Final answer: 14.14.

× 0/1 × 0/1

✓ 1/1

× 0/1

Multi-step Score: 0.5

Final CoT Evaluation Score: 0.35

- Figure 34: Response of Different Problem Versions by GPT-4V [47]. By comparing the output from three versions, we observe increasing reasoning errors when more textual content is visually depicted in diagrams. We only showcase the Text-dominant version of the problem for brevity.

