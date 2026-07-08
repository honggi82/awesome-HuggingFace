## GenExam: A Multidisciplinary Text-to-Image Exam

# arXiv:2509.14232v5[cs.CV]14May2026

Zhaokai Wang12* Penghao Yin32* Xiangyu Zhao12 Changyao Tian42 Yu Qiao2 Wenhai Wang42 Jifeng Dai3 Gen Luo2

### Abstract

Exams are a fundamental test of expert-level intelligence and require integrated understanding, reasoning, and generation. Existing exam-style benchmarks mainly focus on understanding and reasoning tasks, and current generation benchmarks emphasize the illustration of world knowledge and visual concepts, neglecting the evaluation of rigorous drawing exams. We introduce GenExam, the first benchmark for multidisciplinary text-to-image exams, featuring 1,000 samples across 10 subjects with exam-style prompts organized under a four-level taxonomy. Each problem is equipped with ground-truth images and fine-grained scoring points to enable a precise evaluation of semantic correctness and visual plausibility. Experiments on 17 text-to-image and unified models demonstrate the great challenge of GenExam and the huge gap where open-source models consistently lag behind the leading closedsource ones. By framing image generation as an exam, GenExam offers a rigorous assessment of models’ ability to integrate understanding, reasoning, and generation, providing insights for on the path to intelligent generative models. Our benchmark and evaluation code are released at https: //github.com/OpenGVLab/GenExam.

### 1. Introduction

Exams are the ultimate test of expert-level intelligence. They are not merely about recalling knowledge points, but serve as a comprehensive assessment of understanding, reasoning, and generation. The ability to solve multidisciplinary exam problems indicates that a model has expert-level intelligence surpassing that of most adults. Therefore, exam-style bench-

*Equal contribution

1Shanghai Jiao Tong University 2Shanghai AI Laboratory 3Tsinghua University 4The Chinese University of Hong Kong. Correspondence to: Gen Luo <luogen@pjlab.org.cn>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

marks naturally serve as a crucial yardstick to evaluate the progress of artificial general intelligence (AGI) towards domain expertise (Morris et al., 2024; Yue et al., 2024; Phan et al., 2025).

However, existing multidisciplinary benchmarks are primarily focused on understanding tasks, including pure text benchmarks (Wang et al., 2024b; Zhong et al., 2024) and multimodal benchmarks (Yue et al., 2024; 2025), but rarely focus on image generation tasks. In the field of text-toimage generation (T2I), current benchmarks focus more on general world knowledge reasoning (Niu et al., 2026; Sun et al., 2025). Some benchmarks have explored the domain of subject-specific images but are largely limited to concept illustration (Luo et al., 2025; Chang et al., 2025b;a), a relatively simple task with loose diagnostic criteria, similar to “illustrating concepts through image generation” rather than “solving a drawing exam”.

Upon a closer inspection of graph-drawing questions in common multidisciplinary exams, such as AP (CollegeBoard, 1952), A-level (Cambridge, 1951), and IB (IBO, 1968), we find that text-to-image exams present far distinct challenges against common image generation tasks:

- 1. The questions (prompts) are typically more complex, precise, and diverse, with strict and explicit constraints on drawing (generation);
- 2. Each question is equipped with a reference answer (ground truth image) and detailed scoring criteria (scoring points), enabling rigorous evaluation of the correctness of drawn images;
- 3. Knowledge scope is broader and deeper and can be systematically organized into a hierarchical structure (taxonomy);
- 4. Solving them demands rigorous semantic accuracy in drawn images, requiring integration of understanding, reasoning, and generation.

As shown in Fig. 1, even state-of-the-art text-to-image models, such as Nano Banana Pro (Google, 2025b) and GPTImage-1.5 (OpenAI, 2025c) excel at producing images with superficially plausible overall structure but often fall short in portraying correct exam details.

Based on this motivation, we introduce GenExam, the first

[Figure 1]

###### Key Points of GenExam

###### Multidisciplinary Text-to-Image Exam

- 1. Complex and diverse questions (prompts) with strict constraints on drawing (generation)
- 2. Reference answer (ground truth image) & customized and detailed scoring criteria (scoring points)
- 3. Deep and broad knowledge scope organized in hierarchy (taxonomy)
- 4. Integration of understanding, reasoning, and generation

[Figure 2]

Prompt: Draw an undirected weighted graph with 7 vertices labeled A, B, C, D, E, F, and G. Connect the vertices with edges as follows, and label each edge with its corresponding weight: AB (3), AF (6), AG (4), BC (2), BG (3), CD (10), CG (6), DE (10), DG (5), EF (7), EG (9), FG (8). Do not include other edges.

Taxonomy: Computer/Data Structures and Algorithms/Graph/Undirected Graph

[Figure 3]

[Figure 4]

[Figure 5]

BC=2?

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

No Duplicated Nodes?

[Figure 10]

No Extra Edges?

[Figure 11]

[Figure 12]

Ground Truth Nano Banana Pro GPT-Image-1.5 Seedream 4.5

- Figure 1. Examples of state-of-the-art generative models on the GenExam benchmark. Orange dashed circles indicate scoring points. GenExam contains complex and diverse prompts resembling human exams, and pose great challenge to existing models.

benchmark dedicated to multidisciplinary text-to-image exams. As shown in Fig. 2, GenExam asks the model to accomplish the subject-specific drawing exam with a series of detailed, rigorous and knowledgeable requirements. It includes 1,000 samples of 10 subjects with detailed fourlevel taxonomy, collected from college-level exams and open-source benchmarks and carefully reviewed by PhD annotators. In addition, GenExam adopts a strict prompt generation process through cooperation of GPT-5 (OpenAI,

42.5%, respectively, most models can only obtain strict scores less than 15%, and many latest T2I and unified multimodal large language models (MLLMs) like FLUX.2 dev (Labs, 2025), Qwen-Image-2512 (Wu et al., 2025) and BAGEL (Deng et al., 2025) even yield strict scores less than 3%. These results highlight the great challenge of GenExam and the huge gap between closed-source and open-source models, suggesting potential directions for existing models towards AGI. Overall, our contributions can be summarized in three folds:

- 2025a) and humans, with each prompt designed in the style of real exams (CollegeBoard, 1952; IBO, 1968). Given these designs, models are required to seamlessly integrate their understanding, reasoning and generation capabilities to solve the problem of GenExam.

- • We propose GenExam, the first benchmark for multidisciplinary text-to-image exams, with carefully curated prompts and ground truth images. It aims to serve as a real exam to test whether models can seamlessly integrate understanding, reasoning and generation.
- • We design an evaluation framework tailored for multidisciplinary images, assessing the semantic correctness and visual plausibility of disciplinary images through scoring points customized for each sample.
- • Comprehensive experiments on existing T2I models and unified MLLMs highlight the challenge of GenExam. We provide analysis and insights on their capabilities and areas for improvement.

The evaluation of multidisciplinary images also poses challenges, as they prioritize semantic correctness over photorealism or aesthetic quality, different from natural images. Relying solely on a single instruction for MLLM-as-a-judge evaluation (Zhang et al., 2025c) makes it difficult to cover all requirements in the prompt. Therefore, we design customized evaluation criteria for each individual sample, similar to the exam marking process. As shown in in Fig. 1, scoring points of each sample are produced through a rigorous review process of GPT-5 and humans, with the ground truth image as the evaluation reference. Leveraging the most advanced commercial MLLM, i.e. GPT-5, we can automatically diagnose each scoring point through visual question answering. In addition to the correctness evaluation, GenExam also includes additional metrics to assess the visual plausibility, namely encompassing spelling, readability, and logical consistency. Based on the above metrics, we calculate two final scores to satisfy different evaluation requirements, with strict and relaxed standards respectively.

### 2. Related Work

2.1. Generative Models

Text-to-image generation (T2I) has been a primary research focus in generative AI in recent years. The dominant approaches are based on diffusion (Ho et al., 2020; Dhariwal & Nichol, 2021; Rombach et al., 2022; Labs, 2024) or autoregressive objects (Ramesh et al., 2021; Tian et al., 2024; Sun et al., 2024), which excel at generating images with high aesthetic quality and strong semantic consistency with

Our experimental results show that although Nano Banana Pro and GPT-Image-1.5 achieves strict scores of 70.2% and

Computer Science Biology

Math Chemistry

Computer Science Biology

Math Chemistry

Draw a schematic diagram in a eukaryotic cell, including: (1) a labeled cell with mitochondrion and cytosol, (2) three stages of cellular respiration (glycolysis, Krebs cycle, electron transport chain) in correct locations, (3) main inputs (food/glucose, oxygen) and outputs (ATP, CO2, water), and (4) the flow of glucose and NADH between stages. Use arrows for directions and label the components.

Starting from benzene, draw a stepwise reaction scheme (skeletal structures) to obtain the final product using the reagents in the given order: 1) propanoyl chloride (CH3CH2COCl), AlCl3; 2) Br2/FeBr3; 3) H2/Pd; 4) N-bromosuccinimide (NBS), (PhCO2)2, CCl4. Show all the intermediates, respecting directing effects, reduction chemoselectivity, and radical benzylic bromination, and depict the final product structure.

Draw the complete recursion and merge tree for the Merge Sort algorithm when sorting the array [38, 27, 43, 3, 9, 82, 10]. Clearly show each step of the division (splitting into subarrays) and the merging process, labeling the arrays at each node of the tree.

In a rectangle ABCD, draw a circle centered at vertex A. Let the circle be tangent to diagonal BD at point E. From vertex C, draw a tangent CG to the circle, with G as the point of tangency. Let CG be perpendicular to BD at point F. Connect AE, AG, and AC. Clearly label all points A, B, C, D, E, F, and G.

[Figure 13]

[Figure 14]

Taxonomy: Computer/ Data Structures and Algorithms/ Sorting

Taxonomy: Biology/Physiology_and Molecular Process/ Molecular Mechanisms/ Material Metabolism

[Figure 15]

Taxonomy: Mathematics/ Plane Geometry/ Circles/Tangent

[Figure 16]

Taxonomy: Chemistry/ Chemical_Reaction/ Organic_Reaction

Engineering

Physics

Engineering

Physics

[Figure 17]

Draw a schematic diagram of a regenerative Rankine cycle with an open feedwater heater. Clearly indicate the boiler, turbine, condenser, two pumps, and the open feedwater heater. Show the extraction of steam from the turbine for feedwater heating, and label all main state points (1 through 7) and the mass flow split (y and 1-y) at the extraction point. Use arrows to indicate the direction of fluid flow.

Construct a precise ray diagram for a thin, symmetric biconvex lens on a horizontal optical axis. Place an upright object (arrow) to the left at distance v > focal length f, and form the real image to the right at distance b. From the object’s tip, draw the three principal rays: (1) a ray parallel to the axis refracting through the far focal point, (2) a ray passing through the lens center undeviated, and (3) a ray aimed at the near focal point emerging parallel to the axis.

[Figure 18]

[Figure 19]

Taxonomy: Engineering/Thermodynamics and Energy/ Thermal Cycles

Taxonomy: Physics/Optics and Waves/ Optical Path Diagrams/Lens

Geography Music History

Economics

Economics

Geography

Music History

Draw a thematic map of the Italian Peninsula in the 8th century BCE. Delineate and color five territorial zones: Etruscans (orange) in northcentral Italy, Gauls (pale yellow) in the far north, Sabines (green) inland northeast of Rome, Latins (purple) on the mid-western coast around the lower Tiber, and Greeks (pink) along southern Italy and Sicily’s coasts. Show surrounding seas. Include a simple legend.

Draw an AD-AS model graph showing the effect of an increase in health insurance premiums paid by firms by shifting the SRAS curve, while keeping the AD curve unchanged. Clearly label the all the curves, the initial equilibrium (Y1, P1), and the new equilibrium (Y2, P2). Indicate the direction of the SRAS shift and annotate the cause of the shift.

Draw the C major scale in the treble clef, labeling each note (C, D, E, F, G, A, B) with its corresponding scale degree (1st to 7th) and Roman numeral (I to VII). Highlight the 6th degree (A) as the submediant by enclosing it in a box and labeling it 'SUBMEDIANT'.

Draw a schematic diagram of the Earth's global atmospheric circulation, labeling and illustrating the Hadley cell, Ferrel cell, and Polar cell. Indicate the direction of the winds, westerlies, and polar easterlies, and mark the positions of subpolar lows (Polar front) and polar highs.

[Figure 20]

Taxonomy: Music/Notes and Rests

[Figure 21]

[Figure 22]

Taxonomy: Geography/ Climate and Cycles/ Natural Cycles/ Atmospheric Circulation

[Figure 23]

Taxonomy: Economics/Macroeconomics/ AD_AS_Model

Taxonomy: History/Historical Maps/ Territory Maps

- Figure 2. Examples from GenExam. GenExam contains 1,000 exam-style prompts that span over 10 subjects and corresponding reference images for multidisciplinary text-to-image exam.

prompts. Recently, unified MLLMs have emerged as a popular research topic, which supports multimodal understanding and image generation with a single model (Wang et al., 2024a; Deng et al., 2025; Xie et al., 2025; Li et al., 2025; Tian et al., 2026). The core advantage of unified MLLMs lies in providing powerful multimodal understanding and reasoning capabilities for image generation. The latest state-of-the-art models like GPT-Image-1.5 (OpenAI, 2025c) and Nano Banana Pro (Google, 2025b) show strong capabilities in generating plausible natural and aesthetic images. However, as shown in our experiments, their ability on multidisciplinary text-to-image exams is still limited.

with benchmarks like GenEval (Ghosh et al., 2023) and metrics like CLIP score (Hessel et al., 2021) and VQA score (Lin et al., 2024). Recent works move towards reasoning-informed generation (Niu et al., 2026; Zhao et al., 2025; Meng et al., 2024; Sun et al., 2025; Zhang et al., 2025a), and typically rely on MLLM-as-a-judge (Zhang et al., 2025c) for evaluation. However, their focus is mainly on world knowledge or commonsense reasoning, with domains restricted to natural or synthetic images.

Meanwhile, benchmarks such as MMMG (Luo et al., 2025), OneIG-Bench (Chang et al., 2025a), and SridBench (Chang et al., 2025b) explore multidisciplinary image generation, but primarily in the form of knowledge-concept illustration, a relatively simple task with loose diagnostic criteria compared to exam-style questions. Existing benchmarks for multidisciplinary exams are mainly for understanding tasks (Yue et al., 2024; Xia et al., 2025; Phan et al., 2025), but rarely focus on image generation. In contrast, our GenExam targets multidisciplinary text-to-image exams, with complex, precise, and diverse prompts and strict and explicit constraints on generation. By designing fine-grained scoring points, GenExam provides a more rigorous evaluation of reasoning-informed generation in multidisciplinary images.

#### 2.2. Image Generation Benchmarks

Generation benchmarks (Zhao et al., 2025; Ghosh et al., 2023; Niu et al., 2026; Xia et al., 2025; Liu et al., 2026a; Li et al., 2026; Liu et al., 2026b) are of great importance in guiding models towards intelligent generative models. The early evaluation for image generation is mainly based on image similarity metrics (Heusel et al., 2017), which fails to capture the higher-level semantics. The current evaluation focuses primarily on literal prompt-image alignment,

### 3. GenExam

#### 3.1. Overview

Fig. 2 provides an overview of our GenExam benchmark. It contains 1,000 samples that span over 10 subjects: mathematics, physics, chemistry, biology, computer science, geography, economics, music, history, and engineering. Each sample contains a ground truth image, a four-level taxonomy, and several scoring points.

Scoring Points. One critical challenge in the evaluation of exam-style image generation is that it is difficult to judge the semantic correctness of the generated image. This becomes evident when using MLLM-as-a-judge (Zhang et al., 2025c) with a single instruction template, where MLLMs often fail to cover all the requirements specified in the prompt. Examples include the specific structure of organic molecules (e.g., the number of each type of atom, functional groups, chemical bonds), positional relationships in geometric figures, all notes in a musical score, etc. Therefore, inspired by scoring criteria for graph-drawing questions in human exams, we design several scoring points for each prompt, in the form of questions. By answering the questions, we can decide whether the generated image is correct, e.g. “Does the molecule contain exactly 8 carbon atoms?”. This ensures that each sample has a unique evaluation criterion, which improves the accuracy and stability of the evaluation. Each scoring point has a predefined score that sums up to 1, similar to the grading system in exams. The detailed evaluation protocol is given in Sec. 3.2.

Taxonomy. In addition to the top-level subject, we also provide a four-level taxonomy, such as “Mathematics/Plane Geometry/Circles/Tangent” in Fig. 2. The taxonomy annotations are built based on ISCED-F (UNESCO, 2013) fields, with the aim of improving the academic strictness of GenExam for seamless integration with multidisciplinary research. Detailed taxonomy is provided in Appendix D.

Statistics. In Tab. 1, the prompts in GenExam exhibit an average length of 74.8 words, providing dense and detailed requirements for the generated images, resembling difficult graph-drawing questions in real human exams. Each prompt has an average of 6.9 scoring points to thoroughly assess the generated image from all aspects based on the input prompt. In Fig. 3, we visualize the distributions of the image types, which demonstrate the diversity of images in our benchmark. We also provide a subset to facilitate an efficient evaluation. More statistics are provided in Appendix A.

Data Curation. We first consider possible subjects and domains of multidisciplinary images and curate a four-level taxonomy, and use it to generated keywords for web image collection. Images from existing subject-knowledge-related multimodal understanding datasets are also collected. We then conduct automatic filtering by utilizing GPT-5 to rate

images in terms of text richness, image domain, complexity and subject knowledge density, and set a threshold to remove undesirable images. Subsequently, prompts and scoring points are curated through GPT-5 drafting and rigorous human refinement. Detailed data curation pipeline is provided in Sec. B.

#### 3.2. Evaluation Protocol

Unlike the evaluation of natural images, multidisciplinary images focus more on correctness than photorealism or aesthetic quality, and a simple flaw can cause significant errors. For instance, if a single atom is drawn mistakenly in a chemical structure or some arrows are depicted in the opposite direction in a diagram, the entire image is incorrect. In practice, common failure cases of current T2I models (Fig. 6) are either from semantic consistency with the prompt or plausibility of the image itself (e.g. spelling errors). Therefore, to thoroughly judge the correctness and overall quality of the generated image, we consider two dimensions: semantic correctness and visual plausibility. The evaluation pipeline is shown in Fig. 4.

Semantic Correctness. This dimension assesses the image’s consistency with the input prompt. The scoring points mentioned in Sec. 3.1 provide detailed criteria to judge each specific prompt. We use the MLLM judge to answer each scoring points questions by “Yes” or “No” like visual question answering. Since the answers should all be positive for a correct image, we can compute semantic correctness score as the sum of scores of all the questions whose answers are “Yes”. The range of semantic correctness is 0-1. Since judging certain subject knowledge-related aspects can be difficult, we add the ground truth image as input, so that the MLLM judge can use it as reference.

Visual Plausibility. This dimension focuses on the image itself, consisting of three sub-dimensions:

- 1. Spelling: Whether the spelling of the text in the image is correct, including notation and equations. This assesses the model’s text rendering capability, which has become the focus in recent models (Google, 2025b; Wu et al., 2025).
- 2. Logical Consistency: Whether all marks, text, musical notes, etc. are logically consistent, e.g. the labeled coordinates and the actual position. This evaluates the model’s ability to maintain internal coherence in the generated content, reflecting its capacity to integrate multiple components in a logically unified manner.
- 3. Readability: Whether all components are clearly readable and identifiable, without incorrectly placed labels and marks, overlap, occlusion, and unlabeled key components. This requires the model to design an appropriate layout and place the components, e.g. through reasoning.

Statistics Value Taxonomy Count (level 1/2/3/4) 10/40/132/236 Subject Knowledge Difficulty (easy:medium:hard) 24%: 38%: 38% Minimum Number of Scoring Points 3 Maximum Number of Scoring Points 14 Average Number of Scoring Points 6.9

[Figure 24]

Minimum Prompt Length (words) 24 Maximum Prompt Length (words) 173 Average Prompt Length (words) 74.8

Table 1. Key statistics of GenExam. Figure 3. Distribution of image types.

[Figure 25]

[Figure 26]

Draw the chemical reaction scheme for the acid-catalyzed hydration process (using H2SO4 and H2O) of 1-methylcyclohexene. Show the reactant, the reaction conditions, and the major product, indicating the position of the hydroxyl group according to Markovnikov's rule. Ground Truth Image Generated Image

2

GPT-5

[Figure 27]

[Figure 28]

Spelling

[Figure 29]

[Figure 30]

0

rate

2

Logical Consistency

Strict Score

Prompt

GPT-5

[Figure 31]

[Figure 32]

[Figure 33]

answer

0.65

2

Is the reactant 1-methylcyclohexene (a six-membered ring containing a double bond and a methyl group at the 1-position)?

Readability

Relaxed Score

[Figure 34]

(Score: 0.2)

[Figure 35]

[Figure 36]

sum of scores 0.5

Is the reaction condition H2SO4 and H2O labeled? (Score: 0.1)

…

...

Semantic Correctness

Scoring Points

Answers

Figure 4. Evaluation protocol of GenExam. Employing MLLM-as-a-judge, we use scoring points of each prompt to calculate semantic correctness (0-1), and also rate the image in terms of spelling, logical consistency, and readability (0/1/2). The four dimensions are used to calculate a strict score and a relaxed score.

All the three sub-dimensions are in range 0-2, where 2 indicates almost perfect (tiny errors are allowed), 1 indicates a few flaws that slightly hinder the understanding of the key information, and 0 indicates critical errors that significantly hinder the understanding of the key information.

et al., 2026), and the values are 0.7, 0.1, 0.1 and 0.1 respectively. We encourage future research to report both strict and relaxed scores for comparison.

Evaluator Model. We utilize MLLM-as-a-judge (Zhang et al., 2025c) for evaluation with carefully crafted prompts, provided in Appendix E.3. Specifically, we adopt GPT-5 (OpenAI, 2025a), which shows strong multimodal understanding capabilities in multidisciplinary images. In Tab. 3 and Appendix Tab. 4, we also validate the robustness of Gemini-3-Flash and other MLLMs.

Strict and Relaxed Scores. Based on the two dimensions, we first calculate a strict score as the percentage of correct images among all generated images. An image is considered correct if and only if semantic correctness is 1 (i.e. all scoring points are correct) and the scores of spelling, logical consistency and readability are all 2, indicating full satisfaction of all evaluation dimensions, similar to RISEBench (Zhao et al., 2025). It is only more aligned with multidisciplinary image generation where a tiny flaw could lead to a technically wrong image, posing great challenges for T2I models.

### 4. Experiments

In this section, we evaluate the performance of representative methods on our GenExam. We also validate the alignment between MLLM-as-a-judge and human evaluation, and offer analysis on the failure cases of existing methods. We use GPT-5 (OpenAI, 2025a) as the evaluator model and set the reasoning effort as “low”. More experiments are provided in Appendix C, including results categorized by subject knowledge difficulty and level-2 taxonomy, detailed scores on each dimension, and ablations on the evaluators.

However, as shown in Tab. 2, existing methods struggle to generate strictly correct images on GenExam. To highlight the difference between these models, we additionally calculate a relaxed score as the weighted average of the semantic correctness, spelling, logical consistency and readability. The weights are selected based on alignment with human preferences, similar to the process in WiScore (Niu

Math Phy Chem Bio Geo Comp Eng Econ Music Hist Overall Str Rel Str Rel Str Rel Str Rel Str Rel Str Rel Str Rel Str Rel Str Rel Str Rel Str Rel Closed-source Models

Model

Nano Banana Pro 55.6 86.3 75.2 95.1 60.2 88.7 75.6 95.9 75.8 96.5 65.7 91.7 71.2 95.1 88.3 97.2 61.5 91.0 97.6 99.9 70.2 93.0 GPT-Image-1.5 26.5 65.8 46.0 85.4 39.0 78.1 56.4 91.9 60.6 92.5 36.3 75.8 44.1 86.4 42.9 85.5 29.2 70.8 51.2 90.9 42.5 81.5 GPT-Image-1 7.9 51.9 13.3 66.4 13.6 53.4 22.8 73.7 15.9 73.9 10.3 55.6 13.1 65.5 13.0 65.8 9.2 52.7 2.4 67.4 13.1 62.2 Seedream 4.5 5.3 44.7 11.5 63.4 7.6 48.9 25.0 75.8 12.1 67.6 12.7 57.9 15.3 69.7 15.6 67.3 3.1 38.0 4.9 55.0 12.3 59.5 FLUX.2 max 6.6 49.0 8.8 63.2 6.8 54.0 11.6 74.5 15.2 76.3 8.8 56.5 10.8 68.9 2.6 61.5 6.2 47.0 7.3 68.0 8.6 61.6 Seedream 4.0 2.6 39.8 3.5 49.0 5.9 46.1 18.6 71.0 10.6 65.1 6.9 52.2 11.7 60.0 5.2 56.0 0.0 34.5 7.3 56.7 7.8 53.2 Imagen-4-Ultra 2.6 35.9 9.7 57.4 9.3 44.5 14.7 68.1 7.6 66.9 2.9 40.1 12.6 65.6 9.1 59.7 0.0 38.4 0.0 57.8 7.8 53.0 Open-source T2I Models

FLUX.2 dev 2.6 31.6 1.8 42.7 4.2 33.2 3.8 54.8 3.0 62.6 1.0 31.1 2.7 48.9 1.3 43.6 0.0 33.4 0.0 47.5 2.4 42.3 Qwen-Image-2512 0.0 27.9 2.7 41.3 0.8 23.2 1.3 44.4 6.1 56.6 0.0 24.1 4.5 42.9 0.0 32.3 0.0 28.3 0.0 37.0 1.5 35.3 HiDream-I1-Full 0.0 16.7 0.0 17.7 0.0 13.5 0.0 27.3 0.0 36.2 0.0 15.4 0.0 24.4 0.0 18.8 0.0 21.3 0.0 31.8 0.0 21.2 Stable Diffusion 3.5 Large 0.0 12.2 0.0 13.2 0.0 10.7 0.0 21.8 0.0 38.8 0.0 6.6 0.0 16.3 0.0 8.0 0.0 24.1 0.0 18.0 0.0 15.9 Open-source Unified MLLMs

BAGEL (thinking) 0.0 11.7 0.0 13.8 0.0 11.9 0.0 15.2 0.0 28.5 0.0 6.2 0.0 10.7 0.0 6.3 0.0 14.7 0.0 16.0 0.0 12.9 BAGEL 0.0 14.7 0.0 10.6 0.0 7.9 0.0 10.8 0.0 24.5 0.0 6.8 0.0 10.2 0.0 5.3 0.0 13.7 0.0 14.4 0.0 11.4 Show-o2-7B 0.0 10.8 0.0 11.9 0.0 4.8 0.0 12.8 0.0 33.3 0.0 4.7 0.0 11.8 0.0 7.0 0.0 8.8 0.0 14.5 0.0 11.2 Show-o2-1.5B-HQ 0.0 7.3 0.0 7.5 0.0 6.2 0.0 15.0 0.0 25.3 0.0 4.3 0.0 9.3 0.0 7.3 0.0 7.6 0.0 19.8 0.0 10.1 BLIP3o-NEXT-GRPO-Text-3B 0.0 15.5 0.0 10.5 0.0 9.2 0.0 15.5 0.0 23.7 0.0 8.2 0.0 10.1 0.0 8.1 0.0 15.2 0.0 10.2 0.0 12.6 BLIP3o-8B 0.0 6.4 0.0 5.5 0.0 4.7 0.0 7.0 0.0 16.7 0.0 3.6 0.0 8.4 0.0 2.5 0.0 6.0 0.0 11.2 0.0 6.7 Janus-Pro 0.0 13.7 0.0 8.8 0.0 8.2 0.0 7.2 0.0 18.8 0.0 3.9 0.0 10.5 0.0 4.2 0.0 14.5 0.0 6.6 0.0 9.5 Emu3 0.0 11.3 0.0 0.6 0.0 0.6 0.0 5.6 0.0 34.6 0.0 5.1 0.0 16.5 0.0 1.9 0.0 5.8 0.0 6.2 0.0 8.8

Table 2. Strict scores (Str) and relaxed scores (Rel) on GenExam.

#### 4.1. Baselines

We select 17 representative models for comparison, including closed-source proprietary models, open-source T2I models and unified MLLMs. We use the default T2I configuration for each of the models for inference.

Closed-source Models: Nano Banana Pro (Google, 2025b), GPT-Image-1.5 (OpenAI, 2025c), GPT-Image-1 (OpenAI,

- 2025b), Seedream 4.5 (Seed, 2025), FLUX.2 max (Labs, 2025), and Imagen-4-Ultra (Deepmind, 2025). These proprietary models represent the state-of-the-art methods in text-to-image generation.

Open-source T2I Models: Qwen-Image-2512 (Wu et al., 2025), FLUX.2 dev (Labs, 2025), HiDream-I1-Full (Cai et al., 2025), and Stable Diffusion 3.5 Large (Rombach et al., 2022). These models are dedicated to the T2I task.

Open-source Unified MLLMs: Show-o2-7B, Show-o21.5B-HQ (Xie et al., 2025), BAGEL (Deng et al., 2025) (thinking & non-thinking), Janus-Pro (Chen et al., 2025c), Emu3 (Wang et al., 2024a), BLIP3o-8B (Chen et al., 2025a), and BLIP3o-NEXT-GRPO-Text-3B (Chen et al., 2025b). These models employ a unified architecture for T2I and multimodal understanding.

#### 4.2. Main Results

Results on GenExam are provided in Tab. 2. The strict scores demonstrate the challenge of our benchmark. While Nano Banana Pro and GPT-Image-1.5 show relatively strong performance of 70.2% and 42.5%, other closed-source methods like Seedream 4.5 and FLUX.2 max can only achieve strict scores less than 15%. For open-source models, even the latest Qwen-Image-2512 and FLUX.2 dev fail in almost all cases, with strict scores less than 5%. This shows a huge gap between open-source and closed-source models in multidisciplinary text-to-image exams, suggesting potential direction for future improvement.

The differences between the models are highlighted with relaxed scores. The performance gap between open-source and closed-source models still exists. FLUX.2 dev and Qwen-Image-2512 achieve the highest relaxed scores of 42.3 and 35.3 among open-source methods. However, current unified MLLMs like BAGEL and Show-o2 still lag behind dedicated T2I models, suggesting a significant challenge for unified MLLMs to utilize its multidisciplinary knowledge and reasoning ability into image generation.

- 0.0

0.5

1.0

- 1.5

- 2.0 1.96 1.95

- 0.0

0.5

1.0

- 1.5

- 2.0 1.95 1.87

1.0

2.0

0.92

1.81

1.73

1.70

1.66

0.78

1.58

1.50

1.5

1.18

0.57 0.55

1.11

1.10

0.99

0.5

1.0

0.89

0.41

0.35

0.63

0.53

0.47

0.5

0.0

0.0

Semantic Correctness

Spelling

Logic Consistency

Readability

Nano Banana Pro

GPT-Image-1.5

GPT-Image-1

Seedream 4.5

Flux-2-Dev

Qwen-Image-2512

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

###### Figure 5. Comparison across models on four evaluation dimensions.

Kendall Spearman Pearson

Metric

Semantic Correctness

Logical Consistency

τ p ρ p r p

Spelling

Readability

Relaxed Score By GPT-5 0.6746 <0.0001 0.8217 <0.0001 0.8444 <0.0001 Relaxed Score By Gemini-3-Flash 0.6608 <0.0001 0.8200 <0.0001 0.8261 <0.0001 Semantic Correctness 0.6333 <0.0001 0.7862 <0.0001 0.8062 <0.0001 VQA Score 0.1452 0.0009 0.2189 0.0005 0.1786 0.0046 CLIP Score 0.1158 0.0072 0.1620 0.0103 0.1647 0.0091

Range 0-1 0-2 0-2 0-2 MAE 0.10 0.11 0.28 0.20

(a) MAE

(b) Correlation Table 3. Mean absolute error (MAE) and correlations between human preferences and automatic evaluation.

#### 4.3. Human Alignment

We conduct a human alignment experiment to examine the validity of MLLM-as-a-judge, following the paradigms in (Zhao et al., 2025; Sun et al., 2025). Five experts are asked to score 250 randomly sampled images generated by GPTImage-1.

They first provide an overall rating between 1 and 10, and then follow the automatic evaluation protocol to annotate scoring points, spelling, logical consistency and readability. The average scores among all experts are used to calculate the mean absolute error (MAE) and correlations between MLLM-judge predicted scores and human scores. In Tab. 3(a), the results show a low MAE across the four dimensions, indicating high precision of our evaluation.

We then use the overall rating to calculate correlations (Kendall’s τ, Spearman’s ρ and Pearson’s r) between human scores and four automatic metrics: our relaxed score, our semantic correctness, VQA score (Lin et al., 2024), and CLIP score (Hessel et al., 2021). In Tab. 3(b), our relaxed score evaluated by GPT-5 achieves high correlations with human judgments with p < 0.05, slightly better than using solely semantic correctness without visual plausibility, while CLIP score and VQA score fail to capture the correctness of multidisciplinary images. When using Gemini-3-Flash (Google, 2025a) as the evaluator for relaxed scores, the results still show strong human alignment.

In Appendix Tab. 4, we further examine other closed-source and open-source evaluators. In Appendix Tab. 5, we extend to generated images of five other models to confirm strong correlations with human ratings across all models, and provide standard deviations to demonstrate the stability of the evaluation.

#### 4.4. Error Analysis

To gain deeper insights into the strengths and limitations of the models, we provide a comparison of representative models on each evaluation dimension in Fig. 5. Specifically, Nano Banana Pro and GPT-Image-1.5 surpass other models across all four dimensions, notably with near-perfect spelling and readability, and Nano Banana Pro shows even stronger semantic correctness and logic consistency than GPT-Image-1.5. Open-source models Qwen-Image-2512 and FLUX.2 dev lag behind closed-source models across all dimensions, especially on spelling and logic consistency, suggesting that open-source models should first focus on basic drawing before tackling high-level semantics and reasoning.

Examples of generated images are shown in Fig. 6. As shown, state-of-the-art closed-source models such as Nano Banana Pro can generate relatively high-quality images that adheres to prompt specifications, despite some minor flaws. In contrast, open-source models often struggle with fundamental rendering skills, frequently generating incoherent

Math

Index Scoring Point Question Score

- 1 Are the x- and y-axes of the Cartesian plane shown? 0.08
- 2 Is the parabola y = x^2 drawn opening upward and passing through (0,0) and (1,1)? 0.18
- 3 Is the line y = 2x − 1 drawn with slope 2, crossing the x-axis at x = 0.5 and passing through (1,1)? 0.2
- 4 Is the bounded region exactly the set enclosed by y = 0, y = x^2, and y = 2x − 1 with vertices (0,0), (0.5,0), and (1,1)? 0.22
- 5 Is only this bounded region shaded (with no extra areas shaded)? 0.12
- 6 Are the intersection points (1,1) indicated? 0.08
- 7

Prompt: Draw the region in the xy-plane bounded by the curves y = x^2, y = 2x - 1, and the x-axis. Clearly indicate the intersection point of the curves and shade the area enclosed between y = x^2 and y = 2x - 1 above the x-axis.

Are the boundaries of shade along the x-axis from 0 to 0.5, along the line from (0.5,0) to (1,1), and along the parabola from (1,1) back to (0,0)?

0.12

Taxonomy: Mathematics/ Analytic Geometry/ Definite Integral Area

[Figure 37]

Ground Truth:

[Figure 38]

Nano Banana Pro

- 0.08 2 0 2

Strict: 0 Relaxed: 0.256

- 1. 2. 3. 4. 5. 7.

- 0.28 2 0 2

Strict: 0 Relaxed: 0.396

- 1. 2. 3. 4. 5. 7.

- 0.08 1 0 0

Strict: 0 Relaxed: 0.106

- 1. 2. 3. 4. 5.

- 0.54 1 0 2

Strict: 0 Relaxed: 0.528

GPT-Image-1.5 Seedream 4.5 Qwen-Image-2512 Flux.2 dev

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

- 1. 2. 3. 4. 5. 7.

- 0.08 1 0 2

Strict: 0 Relaxed: 0.206

- 1. 2. 3. 4. 5. 7.

[Figure 43]

6. 6. 6. 6. 6. 7.

[Figure 44]

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell

History

Prompt: Create a simplified world map (Africa–Europe–Asia only) for a world history exam. Shade, in the same color, the four earliest rivervalley civilization core regions circa 3000–1500 BCE and place letter markers: A at the Nile Valley in northeastern Africa near the delta, B in Mesopotamia near the Tigris River, C along the Indus River in northwestern South Asia, and D along the Yellow (Huang He) River in northern China. Include coastlines and the three major surrounding oceans (Atlantic to the west of Africa, Indian south of South Asia, Pacific to the east of East Asia). Add a small legend box showing the fill color used for “ancient civilizations.”

Index Scoring Point Question Score

- 1 Is there a recognizable continental map showing Africa, Europe, and Asia in correct relative positions? 0.1
- 2 Is a single consistent color used to shade four distinct civilization core regions? 0.06
- 3 Is the letter marker 'A' placed at the Nile Valley in northeastern Africa near the river’s delta? 0.18
- 4 Is the letter marker 'B' placed in Mesopotamia near the Tigris River in southwestern Asia? 0.18
- 5 Is the letter marker 'C' placed along the Indus River in northwestern South Asia? 0.18
- 6 Is the letter marker 'D' placed along the Yellow (Huang He) River in northern China? 0.18
- 7 Is the Atlantic Ocean shown west of Africa? 0.02
- 8 Is the Indian Ocean shown south of the Indian subcontinent? 0.02
- 9 Is the Pacific Ocean shown east of East Asia? 0.02
- 10 Is there a small legend box indicating the color used for ancient civilizations? 0.06

Taxonomy: History/Historical Map/ Territory Map

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Ground Truth:

[Figure 50]

Nano Banana Pro GPT-Image-1.5 Seedream 4.5 Qwen-Image-2512 Flux.2 dev

1. 2. 3. 4. 5. 7. 8. 9. 10.

- 0.34 2 1 2

Strict: 0 Relaxed: 0.404

- 1. 2. 3. 4. 5.

- 0.4 1 0 1

Strict: 0 Relaxed: 0.38

- 1. 2. 3. 4. 5.

1. 2. 3. 4. 5. 7. 8. 9. 10.

- 0.18 0 0 0

Strict: 0 Relaxed: 0.304

- 1. 2. 3. 4. 5.

6. 6. 6. 6. 6. 7. 8. 9. 10.

7. 8. 9. 10.

7. 8. 9. 10.

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Semantic Read Logic Spell 1.0 2 2 2

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell 0.58 2 2 2

Semantic Read Logic Spell

Strict: 1 Relaxed: 1.0

Strict: 1 Relaxed: 1.0

Figure 6. Examples of images generated by top-performing models. Semantic, Read, Logic, and Spell denote semantic correctness, logical consistency and spelling, respectively.

figures characterized by legibility errors (e.g., the blurred label “C” in Qwen-Image-2512 in History), logical inconsistencies (e.g., redundant axes in Qwen-Image-2512 Math), and spelling inaccuracies (e.g., ”Ea?ies ?e’ws” in FLUX.2 dev in History). These visual defects not only diminish visual plausibility scores but also impede semantic evaluation.

discrepancy suggests that the bottleneck lies not in the lack of knowledge, but in the failure to execute visual reasoning based on that knowledge. Consequently, enhancing the ability to transform internal knowledge into visual outputs and improving the alignment between conceptual understanding and generative execution could improve open-source models to a new stage of performance. These specific failures further underscore the value of GenExam’s fine-grained scoring criteria, which effectively capture nuanced flaws that broad metrics such as photorealism or general aesthetics would typically overlook. Additional qualitative results are provided in Appendix F.

It is worth noting that the inferior performance of opensource models often results not from a lack of necessary disciplinary knowledge, but from an inability to accurately translate that knowledge into readable visual figures. For instance, in the result of FLUX.2 dev in history, the model correctly identifies the regions of Egypt, Iran, India, and China (although with excessive highlighting) but fails to place the appropriate graphics within those regions. This

### 5. Conclusion

We presented GenExam, the first benchmark for multidisciplinary text-to-image exams, designed to test whether models can integrate understanding, reasoning and generation to truly solve graph-drawing problems. GenExam offers diverse and challenging prompts with ground truth images and fine-grained scoring points, reflecting the rigor of real exams. Our comprehensive experiments on 17 existing models demonstrate the difficulty of the benchmark and the gap between closed-source and open-source models. We hope GenExam will guide the advancement of generative models toward expert-level intelligence in a multidisciplinary direction.

### References

Bai, L., Cai, Z., Cao, M., Cao, W., Chen, C., Chen, H., Chen, K., Chen, P., Chen, Y., Chen, Y., Cheng, Y., Cheng, Y., Chu, P., Chu, T., Cui, E., Cui, G., Cui, L., Cui, Z., Deng, N., Ding, N., Dong, N., Dong, P., Dou, S., Du, S., Duan, H., Fan, C., Gao, B., Gao, C., Gao, J., Gao, S., Gao, Y., Gao, Z., Ge, J., Ge, Q., Gu, L., Gu, Y., Guo, A., Guo, Q., Guo, X., He, C., He, J., Hong, Y., Hou, S., Hu, C., Hu, H., Hu, J., Hu, M., Hua, Z., Huang, H., Huang, J., Huang,

- X., Huang, Z., Jiang, Z., Kong, L., Li, L., Li, P., Li, P., Li, S., Li, T., Li, W., Li, Y., Lin, D., Lin, J., Lin, T., Lin, Z., Liu, H., Liu, J., Liu, J., Liu, J., Liu, K., Liu, K., Liu, K., Liu, S., Liu, S., Liu, W., Liu, X., Liu, Y., Liu, Z., Lu,
- Y., Lv, H., Lv, H., Lv, H., Lv, Q., Lv, Y., Lyu, C., Ma, C., Ma, J., Ma, R., Ma, R., Ma, R., Ma, X., Ma, Y., Ma, Z., Mi, S., Ning, J., Ning, W., Pang, X., Peng, J., Peng, R., Qiao, Y., Qiu, J., Qu, X., Qu, Y., Ren, Y., Shang, F., Shao, W., Shen, J., Shen, S., Song, C., Song, D., Song, D., Su, C., Su, W., Sun, W., Sun, Y., Tan, Q., Tang, C., Tang, H., Tang, K., Tang, S., Tong, J., Wang, A., Wang, B., Wang, D., Wang, L., Wang, R., Wang, W., Wang, W., Wang, Y., Wang, Z., Wu, L.-I., Wu, W., Wu, Y., Wu, Z., Xiao, L., Xing, S., Xu, C., Xu, H., Xu, J., Xu, R., Xu, W., Yang, G., Yang, Y., Ye, H., Ye, J., Ye, S., Yu, J., Yu, J., Yu, J., Yuan, F., Zhang, B., Zhang, C., Zhang, C., Zhang, H., Zhang, J., Zhang, Q., Zhang, Q., Zhang, S., Zhang, T., Zhang, W., Zhang, W., Zhang, Y., Zhang, Z., Zhao, H., Zhao, Q., Zhao, X., Zhao, X., Zhou, B., Zhou, D., Zhou, P., Zhou, Y., Zhou, Y., Zhu, D., Zhu, L., and Zou, Y. Intern-s1: A scientific multimodal foundation model, 2025a. URL https://arxiv.org/abs/2508.15763.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Cai, Q., Chen, J., Chen, Y., Li, Y., Long, F., Pan, Y., Qiu, Z., Zhang, Y., Gao, F., Xu, P., et al. Hidream-i1: A highefficient image generative foundation model with sparse

diffusion transformer. arXiv preprint arXiv:2505.22705, 2025.

Cambridge. Cambridge international as & a levels. https://www.cambridgeinternational. org/programmes-and-qualifications/ cambridge-advanced/cambridgeinternational-as-and-a-levels/ subjects/, 1951.

Chang, J., Fang, Y., Xing, P., Wu, S., Cheng, W., Wang, R., Zeng, X., Yu, G., and Chen, H.-B. Oneig-bench: Omni-dimensional nuanced evaluation for image generation. Advances in Neural Information Processing Systems, 2025a.

Chang, Y., Feng, Y., Sun, J., Ai, J., Li, C., Zhou, S. K., and Zhang, K. Sridbench: Benchmark of scientific research illustration drawing of image generation model. arXiv preprint arXiv:2505.22126, 2025b.

Chen, J., Xu, Z., Pan, X., Hu, Y., Qin, C., Goldstein, T., Huang, L., Zhou, T., Xie, S., Savarese, S., Xue, L., Xiong, C., and Xu, R. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset, 2025a. URL https://arxiv.org/abs/ 2505.09568.

Chen, J., Xu, Z., Pan, X., Yang, S., Qin, C., Yan, A., Zhou, H., Chen, Z., Zhou, T., Savarese, S., Xue, L., Xiong, C., and Xu, R. Blip3o-next: A next-generation multimodal foundation model, Aug 2025b. URL https://jiuhaichen.github.io/ BLIP3o-NEXT.github.io/.

Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., and Ruan, C. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025c.

CollegeBoard. Advanced placement. https://ap. collegeboard.org/, 1952.

Deepmind, G. Imagen. https://deepmind.google/ models/imagen/, 2025.

Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Google. Gemini 3 flash: frontier intelligence built for speed. https://blog.google/products-andplatforms/products/gemini/gemini-3flash/, 2025a.

Google. Introducing nano banana pro. https: //blog.google/innovation-and-ai/ products/nano-banana-pro/, 2025b.

Guo, M.-H., Xu, J., Zhang, Y., Song, J., Peng, H., Deng, Y.X., Dong, X., Nakayama, K., Geng, Z., Wang, C., et al. Rbench: Graduate-level multi-disciplinary benchmarks for llm & mllm complex reasoning evaluation. International Conference on Machine Learning, 2025.

Hastings, J., Owen, G., Dekker, A., Ennis, M., Kale, N., Muthukrishnan, V., Turner, S., Swainston, N., Mendes, P., and Steinbeck, C. Chebi in 2016: Improved services and an expanding collection of metabolites. Nucleic acids research, 44(D1):D1214–D1219, 2016.

Hessel, J., Holtzman, A., Forbes, M., Bras, R. L., and Choi, Y. Clipscore: A reference-free evaluation metric for image captioning. Conference on Empirical Methods in Natural Language Processing, 2021.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

IBO. Assessment and exams - international baccalaureate. https://www.ibo.org/programmes/ diploma-programme/assessment-andexams/, 1968.

Kembhavi, A., Seo, M., Schwenk, D., Choi, J., Farhadi, A., and Hajishirzi, H. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4999–5007, 2017.

Labs, B. F. Flux. https://github.com/blackforest-labs/flux, 2024.

Labs, B. F. FLUX.2: Frontier Visual Intelligence. https: //bfl.ai/blog/flux-2, 2025.

Li, H., Tian, C., Shao, J., Zhu, X., Wang, Z., Zhu, J., Dou, W., Wang, X., Li, H., Lu, L., et al. Synergen-vl: Towards synergistic image understanding and generation with vision experts and token folding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 29767–29779, 2025.

Li, Y., Zeng, Z., Zhou, Z., Gao, X., Tian, M., Yang, Y., Cheng, M., Dai, Q., Yang, Y., Qiu, L., et al. Bizgeneval: A systematic benchmark for commercial visual content generation. arXiv preprint arXiv:2603.25732, 2026.

Lin, Z., Pathak, D., Li, B., Li, J., Xia, X., Neubig, G., Zhang, P., and Ramanan, D. Evaluating text-to-visual generation with image-to-text generation. In European Conference on Computer Vision, pp. 366–384. Springer, 2024.

Liu, M., Fan, Z., Wang, Z., Gu, L., Zhu, Z., He, Y., Yang, Y., Tian, C., Zhao, X., Liao, N., et al. Grade: Benchmarking discipline-informed reasoning in image editing. arXiv preprint arXiv:2603.12264, 2026a.

Liu, M., Ma, S., Meng, S., Zhao, X., Zhang, Z., Zhang, S., Zhong, Z., Chen, P., Cao, H., Sun, X., et al. Rise-video: Can video generators decode implicit world rules? arXiv preprint arXiv:2602.05986, 2026b.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.

Luo, Y., Yuan, Y., Chen, J., Cai, H., Yue, Z., Yang, Y., Daha, F. Z., Li, J., and Lian, Z. Mmmg: A massive, multidisciplinary, multi-tier generation benchmark for textto-image reasoning. Advances in Neural Information Processing Systems, 2025.

Meng, F., Shao, W., Luo, L., Wang, Y., Chen, Y., Lu, Q., Yang, Y., Yang, T., Zhang, K., Qiao, Y., et al. Phybench: A physical commonsense benchmark for evaluating textto-image models. arXiv preprint arXiv:2406.11802, 2024.

Morris, M. R., Sohl-Dickstein, J., Fiedel, N., Warkentin, T., Dafoe, A., Faust, A., Farabet, C., and Legg, S. Levels of agi for operationalizing progress on the path to agi. International Conference on Machine Learning, 2024.

Niu, Y., Ning, M., Zheng, M., Jin, W., Lin, B., Jin, P., Liao, J., Feng, C., Ning, K., Zhu, B., et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. International Conference on Machine Learning, 2026.

OpenAI. Gpt-5 system card. https://cdn.openai. com/pdf/8124a3ce-ab78-4f06-96eb49ea29ffb52f/gpt5-system-card-aug7. pdf, 2025a.

OpenAI. Gpt-image-1. https://openai.com/index/image-generation-api/, 2025b.

OpenAI. Gpt-image-1.5. https://openai.com/zh-Hans-CN/index/new-chatgpt-images-is-here/, 2025c.

- Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024a.

- Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 2024b.

OpenAI. Gpt-4o system card. https://openai.com/ index/gpt-4o-system-card/, 2025d.

Phan, L., Gatti, A., Han, Z., Li, N., Hu, J., Zhang, H., Zhang, C. B. C., Shaaban, M., Ling, J., Shi, S., et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.

Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I. Zero-shot text-to-image generation. In International conference on machine learning, pp. 8821–8831. Pmlr, 2021.

Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.-m., Bai, S., Xu, X., Chen, Y., et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

Xia, P., Han, S., Qiu, S., Zhou, Y., Wang, Z., Zheng, W., Chen, Z., Cui, C., Ding, M., Li, L., et al. Mmie: Massive multimodal interleaved comprehension benchmark for large vision-language models. The International Conference on Learning Representations, 2025.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Xie, J., Yang, Z., and Shou, M. Z. Show-o2: Improved native unified multimodal models. Advances in Neural Information Processing Systems, 2025.

Seed, B. Seedream 4.0. https://seed.bytedance.com/en/seedream4_0, 2025.

Sun, K., Fang, R., Duan, C., Liu, X., and Liu, X. T2ireasonbench: Benchmarking reasoning-informed textto-image generation. arXiv preprint arXiv:2508.17472, 2025.

Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng, B., Luo, P., and Yuan, Z. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

Tian, C., Yang, D., Chen, G., Cui, E., Wang, Z., Duan, Y., Yin, P., Chen, S., Yang, G., Liu, M., Zhu, Z., Fan, Z., Gu, L., Wang, H., Wei, Q., Yin, J., Yang, X., Zhong, Z., Qin, Q., Xin, Y., Fu, B., Liu, Y., Ge, J., Guo, Q., Luo, G., Li, H., Qiao, Y., Chen, K., and Zhang, H. Internvlu: Democratizing unified multimodal models for understanding, reasoning, generation and editing. 2026. URL https://arxiv.org/abs/2603.09877.

Tian, K., Jiang, Y., Yuan, Z., Peng, B., and Wang, L. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.

UNESCO. Fields of education and training 2013 (isced-f 2013). https://uis.unesco.org/ en/topic/international-standardclassification-education-isced, 2013.

Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.

Yue, X., Zheng, T., Ni, Y., Wang, Y., Zhang, K., Tong, S., Sun, Y., Yin, M., Yu, B., Zhang, G., et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2025.

Zhang, D., Jiang, C., Xu, R., Chen, B., Jin, Z., Lu, Y., Zhang, J., Yong, L., Luo, J., and Luo, S. Worldgenbench: A world-knowledge-integrated benchmark for reasoning-driven text-to-image generation. arXiv preprint arXiv:2505.01490, 2025a.

Zhang, R., Wei, X., Jiang, D., Zhang, Y., Guo, Z., Tong, C., Liu, J., Zhou, A., Wei, B., Zhang, S., et al. Mavis: Mathematical visual instruction tuning. The International Conference on Learning Representations, 2025b.

Zhang, Z., Wang, J., Wen, F., Guo, Y., Zhao, X., Fang, X., Ding, S., Jia, Z., Xiao, J., Shen, Y., Zheng, Y., Zhu, X., Wu, Y., Jiao, Z., Sun, W., Chen, Z., Zhang, K., Fu, K., Cao, Y., Hu, M., Zhou, Y., Zhou, X., Cao, J., Zhou, W., Cao, J., Li, R., Zhou, D., Tian, Y., Zhu, X., Li, C., Wu, H., Liu, X., He, J., Zhou, Y., Liu, H., Zhang, L., Wang, Z., Duan, H., Zhou, Y., Min, X., Jia, Q., Zhou, D., Zhang, W., Cao, J., Yang, X., Yu, J., Zhang, S.,

Duan, H., and Zhai, G. Large multimodal models evaluation: A survey. https://github.com/aibench/LMM-Evaluation-Survey, 2025c. Project

Page: AIBench, available online.

Zhao, X., Zhang, P., Tang, K., Zhu, X., Li, H., Chai, W., Zhang, Z., Xia, R., Zhai, G., Yan, J., et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. Advances in neural information processing systems, 2025.

Zhong, W., Cui, R., Guo, Y., Liang, Y., Lu, S., Wang, Y., Saied, A., Chen, W., and Duan, N. Agieval: A humancentric benchmark for evaluating foundation models. Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2024.

### A. More Statistics

[Figure 56]

(5)

(32)

)6 (

dn a

)7 (

m

Data

yrtsim

Equilibriu

(68)

Reaction

Hard ware

yrtsim

sciteni K

Algorith ms

Structures and

ter

ehcom

ehcortcelE

Networking and

Theory and

Ma

lacim

Evolution (13)

mical

Architecture

mical

reh T

Of

Ecology (5)

ehC

(50)

Structure

Che

Microeconomics (30)

Che

AI (24)

Physiology and Molecular Proces (49)

Systems (11)

Genetics and

(17)

Macroeconomics (42)

(118)

Co

Finance and Decision (5)

mputer (102)

mistry

Structure and Morphology (89)

Economics (77)

Mechanical Engineering (33)

Che

Biology (156)

Civil and Architecture (19)

Engineering (111)

Mechanics and Dynamics (9)

Quantum Mechanics (10)

Surveying and Cartography (9)

Physics (113)

Optics and Waves (11)

Geography (66)

Thermodynamics and Energy (30)

Electromagnetism (15)

History (41)

Music (65)

Thermodynamics (13)

Mechanics of Materials (11)

Math(151)

Circuits and Electronics (36)

Maps (25)

Key

Earth Science (37)

Mechanics (28)

Signatures and Time

Ecology (4)

Map (32)

Change (7)

Chord

Interval

Notes and

Structures and

citylanA oe G

Human and

Solid Geo metry

Signatures (8)

Diagrams (10)

Plane Geo metry

Historical

(2)

Data

Rests (47)

Others

Historical

yrtem )65 (

(11)

(84)

Figure 7. Level-2 taxonomy of GenExam. Detailed four-level taxonomy is in Sec D.

Fig. 7 shows all the level-2 taxonomy in GenExam, and Fig. 8(a) and (b) show the distribution of subjects in GenExam-Full and GenExam-Mini. We include more samples for science and engineering subjects while also covering subjects related to humanity and social science, such as Economics, Music, and History. GenExam-Mini is constructed using stratified sampling on level-3 taxonomy, which ensures similar distributions between GenExam-Full and GenExam-Mini.

The proportion of image sources is presented in Fig. 8(c), demonstrating the diversity of images in our benchmark. Word clouds of prompts in each subject are shown in Fig. 9, where we observe keywords with dense subject knowledge similar to human exams.

[Figure 57]

[Figure 58]

[Figure 59]

(a) (b) (c)

Figure 8. More statistics: (a) Subject distribution on GenExam-Full; (b) Subject distribution on GenExam-Mini; (c) Proportion of each image source.

[Figure 60]

[Figure 61]

(a) Biology (b) Chemistry

[Figure 62]

[Figure 63]

(c) Computer Science (d) Economics

[Figure 64]

[Figure 65]

(e) Engineering (f) Geography

[Figure 66]

[Figure 67]

(g) History (h) Math

[Figure 68]

[Figure 69]

###### (i) Music (j) Physics Figure 9. Word clouds of prompts in each subject.

Manual Check and Filtering

Annotating

Data Collection

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Existing Datasets

Web Images

Taxonomy List

[Figure 78]

[Figure 79]

Annotate Modify & Filter

###### Search

Filter

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Keywords

Prompt Scoring Points

Original Descriptions

Final Data

Images

Figure 10. Data curation pipeline. We use pre-defined taxonomy to collect web images and existing datasets, and conduct annotating and filtering based on GPT-5 and manual check.

### B. Data Curation Pipeline

Data Collection. The data curation pipeline is shown in Fig. 10. To construct a general benchmark for multidisciplinary text-to-image exam, we first consider possible subjects and domains of multidisciplinary images and curate a four-level taxonomy assisted by GPT-5 (OpenAI, 2025a) and through manual check. We then use these taxonomy to generate keywords related to specific subject knowledge, and collect web images from exams and textbooks based on Google search. Images from existing subject-knowledge-related datasets are also collected: MMMU (Yue et al., 2024), ScienceQA (Lu et al., 2022), TextbookQA (Kembhavi et al., 2017), MAVIS (Zhang et al., 2025b), ChEBI (Hastings et al., 2016), HLE (Phan et al., 2025) and R-Bench (Guo et al., 2025). We obtain 40K images from the data sources with their original text descriptions (e.g. webpage content for web images, lecture notes in TextbookQA, or question and answers in MMMU).

Automatic Filtering. To obtain a high-quality dataset, we employ an automatic filtering process on the collected images. We first design heuristic rules to remove images with duplicated content, low resolution, watermarks or non-English text. Then, we leverage GPT-5 to rate each image in terms of its text richness, image domain, image complexity, and subject knowledge density. We set a threshold for each of these aspects to filter out images that do not meet the requirements of our task, including those with:

- • High text richness: e.g. tables or screenshots of pure text. Such images are dominated by textual information and primarily examines the text rendering ability, which deviates from the core requirement of more general “image generation” for visual content.
- • Undesirable image domain: e.g. photographs, pathological images or satellite images. These domains rely more on realistic details or professional equipment for collection, which is inconsistent with real graph-drawing questions in exams. They hardly reflect the ability to visually transform interdisciplinary knowledge.
- • High complexity: e.g. too many components or sub-figures. Excessive components or sub-figures lead to redundant visual information, which interferes with the model’s generation of core visual elements. Additionally, complex structures often exceed the model’s current generation capabilities and pose overly high challenges to the model.
- • Low subject knowledge density: e.g simple plots and charts without subject knowledge. Such images lack core information in interdisciplinary scenarios.

We obtain 6.5K images from automatic filtering for annotating.

Annotating. We employ GPT-5 to generate the initial version of prompts and scoring points. Prompts that require subject knowledge and reasoning are generated, similar to graph-drawing questions in exams. The original text description from the data source is used for reference to improve the precision, bu should be treated carefully since they may contain irrelevant information. Subsequently, scoring points (questions and scores) are generated based on the prompt. We enforce through prompt engineering that the questions cannot be too general (e.g. “Does the image show the correct structure of benzene?”) or too complex (can be split into multiple sub-questions). Scores are based on the difficulty of the questions, where basic questions should have low scores and harder ones have high scores. We provide some few-shot examples to enhance the labeling process. The complete prompts are provided in Appendix E.2.

Manual Check and Filtering. Three PhD annotators perform rigorous manual inspection on all the images, prompt, and

Time (s/img) Closed-source MLLMs

Ground Truth Input

Scoring Points Accuracy ↑

Plausibility MAE ↓

Kendall τ ↑

Spearman ρ ↑

Pearson r ↑

API Cost ($/img)

Evaluator Model

Gemini-3-Flash Img 86.4 - 0.6608 0.8300 0.8261 0.063 27 GPT-5 + low reasoning effort Img 88.5 0.20 0.6746 0.8217 0.8444 0.0242 72 GPT-5 + low reasoning effort − 86.2 0.23 0.6141 0.7753 0.7976 0.0231 59 GPT-5 + low reasoning effort Text 87.8 0.21 0.6559 0.8103 0.8336 0.0234 63 GPT-5 + low reasoning effort Img, Text 88.5 0.20 0.6693 0.8345 0.8427 0.0250 75 GPT-5 + minimal reasoning effort Img 84.2 0.28 0.6066 0.7521 0.7860 0.0144 35 GPT-5 + medium reasoning effort Img 86.7 0.21 0.6617 0.8224 0.8358 0.0389 101 GPT-5 + high reasoning effort Img 87.6 0.20 0.6647 0.8190 0.8304 0.0654 186 GPT-5-mini + minimal reasoning effort Img 79.0 0.35 0.5446 0.6996 0.7185 0.0039 11 GPT-5-mini + low reasoning effort Img 84.3 0.29 0.6002 0.7729 0.7806 0.0052 19 GPT-5-mini + medium reasoning effort Img 84.2 0.25 0.6313 0.8025 0.8165 0.0079 36 GPT-5-mini + high reasoning effort Img 83.4 0.26 0.5906 0.7611 0.7811 0.0196 119

- o3 Img 82.0 0.23 0.6082 0.7706 0.7895 0.0229 39

- o4-mini Img 84.7 0.35 0.6300 0.7915 0.8092 0.0156 31 GPT-4.1 Img 71.2 0.39 0.4417 0.5745 0.5723 0.0142 29 GPT-4o Img 68.5 0.39 0.4303 0.5700 0.5610 0.0134 23 Open-source MLLMs Qwen3-VL-235B-A22B-Thinking Img 75.1 0.38 0.4606 0.6021 0.5759 - 105 Qwen3-VL-235B-A22B-Instruct Img 71.6 0.43 0.4279 0.5703 0.6182 - 35 InternVL3.5-241B-A28B (thinking) Img 63.3 0.41 0.2856 0.3538 0.3866 – 29 Intern-S1 (241B, thinking) Img 57.0 0.51 0.0664* 0.0833* 0.2088* – 39 Intern-S1-mini (8B, thinking) Img 56.6 0.53 0.0027* 0.0023* 0.0158* – 18

Table 4. Ablation on the evaluator model. Settings are identical to Tab. 3. *: The p-value is larger than 0.05, indicating the metric using this evaluator has no statistical significance.

scoring points, and cross-validate each other’s decision. Images with duplicate domain or undesirable text richness and complexity not recognized by automatic filtering are first manually removed. Subsequently, imprecise prompts and are manually modified to align with the ground truth images and include sufficient disciplinary knowledge. Inaccurate scoring points are removed or updated to match the prompt. We finally obtain 1K images as the final dataset.

### C. Additional Experiments

#### C.1. Ablation on Evaluator Models

We further study the selection of evaluator model by examining different MLLMs, including closed-source models (Gemini3-Flash (Google, 2025a), GPT-5, GPT-5-mini, o3, o4-mini, GPT-4.1, GPT-4o (OpenAI, 2025a;d)) and open-source models (Qwen3-VL (Bai et al., 2025b), InternVL3.5 (Wang et al., 2025), Intern-S1, Intern-S1-mini (Bai et al., 2025a)). On the same manually evaluated samples as in Tab. 3, we run each of the models as evaluator and calculate the accuracy of answering scoring points, MAE of visual plausibility, and three correlation metrics (Kendall’s τ, Spearman’s ρ, and Pearson’s r).

As shown in Tab. 4, the latest model GPT-5 with low reasoning effort achieves the best performance with acceptable cost and time, while using medium or high reasoning effort leads to a potential decrease in performance and higher costs. In addition, using ground truth images as reference help to improve the evaluation precision. Early closed-source models have relatively low performance, especially those without reasoning abilities.

For open-source models, Qwen3-VL demonstrates relative strong performance, but still lags behind GPT-5. Other models have a larger gap between closed-source models, where some scoring point accuracy is close to random guess (50%) and some correlations are even of no statistical significance, suggesting these evaluators are not applicable. This is because multidisciplinary images are still difficult for current multimodal understanding models (Yue et al., 2025). Therefore, it is necessary to use advanced MLLMs to evaluate the generated images for higher precision and robustness.

Semantic Spelling Logic Readability

Model

Kendall τ Spearman ρ Pearson r MAE std MAE std MAE std MAE std

Nano Banana Pro 0.068 0.0142 0.071 0.0448 0.158 0.0573 0.118 0.0451 0.7246 0.8621 0.8893 GPT-1.5-Image 0.082 0.0169 0.086 0.0537 0.191 0.0692 0.143 0.0548 0.7013 0.8416 0.8679 Seedream 4.5 0.108 0.0224 0.113 0.0701 0.261 0.0946 0.195 0.0759 0.6672 0.8129 0.8248 FLUX.2 dev 0.126 0.0258 0.132 0.0817 0.301 0.1085 0.225 0.0853 0.6461 0.7928 0.7986 Qwen-Image-2512 0.134 0.0273 0.140 0.0864 0.317 0.1149 0.238 0.0901 0.6319 0.7784 0.7816

Table 5. Human alignment on images generated by more models.

We observe that using ground truth images as reference helps to improve the evaluation precision. We also test using captions of the ground truth images generated by GPT-5 as the ground truth, and observe that text-only ground truth is slightly worse than image-only ground truth. This is because ground truth images have better spatial/detail fidelity than textual descriptions (which often omit critical spatial details) and lower ambiguity. Using both images and textual descriptions shows no significant improvement than image-only, as images already contain sufficient detailed information and incorporating text becomes unnecessary.

#### C.2. Detailed Comparison of Existing Methods

- In Tab. 5, we extend the human alignment experiment to 500 additional samples covering five other models. The results show that relaxed scores maintain strong correlations with human ratings across all models, confirming the metric’s generalizability. The standard deviation (std) values demonstrate high inter-rater reliability.
- In Tab. 6 (a), we show the results on GenExam-Mini. The performance of the models is consistent with the results from the full dataset, which can be considered as a valid subset for efficient and affordable validation.

Results categorized by subject knowlege difficulty are presented in Tab. 6 (b). We observe that models tend to have lower performance on samples with higher subject knowledge difficulty, demonstrating the challenge of integrate profound multidisciplinary knowledge into generation.

We provide results categorized by image types in Tab. 7. We observe that models tend to have lower performance on samples with certain image types like chemical structures, geometric shapes, sheet music and trees & graphs. These suggest potential directions for future improvements.

The results on each level-2 taxonomy are provided in Tab. 8. To highlight the difference between models, and since some models can only obtain a 0% strict score, we only report relaxed scores.

Tab. 9 shows the scores of each dimension (semantic correctness, spelling, logical consistency, readability) on each subject.

Model Strict Relaxed Closed-source Models

Easy Medium Hard Strict Relaxed Strict Relaxed Strict Relaxed Closed-source Models

Model

Nano Banana Pro 70.8 92.8 GPT-Image-1.5 41.2 82.2 GPT-Image-1 12.6 62.9 Seedream 4.5 12.0 61.6 FLUX.2 max 10.4 62.4 Seedream 4.0 8.8 55.0

Nano Banana Pro 73.1 93.8 71.8 92.8 66.8 92.7 GPT-Image-1.5 50.8 85.6 41.4 79.1 38.3 81.3 GPT-Image-1 18.6 68.1 12.8 62.2 9.9 58.5 Seedream 4.5 12.8 59.9 9.8 56.9 14.5 61.9 FLUX.2 max 14.5 66.0 7.7 61.0 5.8 59.3 Imagen-4-Ultra 9.9 58.6 7.1 50.7 7.1 51.8 Open-source T2I Models

- Imagen-4-Ultra 5.6 53.4 Open-source T2I Models FLUX.2 dev 0.4 42.1 Qwen-Image-2512 1.2 36.4 HiDream-I1-Full 0.0 21.2 Stable Diffusion 3.5 Large 0.0 16.0 Open-source Unified MLLMs BAGEL (thinking) 0.0 13.5 BAGEL 0.0 11.7 Show-o2-7B 0.0 12.0 Show-o2-1.5B-HQ 0.0 10.6 BLIP3o-NEXT-GRPO-Text-3B 0.0 12.6 BLIP3o-8B 0.0 6.8 Janus-Pro 0.0 9.8 Emu3 0.0 8.8

FLUX.2 dev 2.9 48.9 2.1 40.4 2.4 39.9 Qwen-Image 0.4 34.6 0.3 25.4 0.0 21.3 HiDream-I1-Full 0.0 29.7 0.0 19.8 0.0 17.1 Stable Diffusion 3.5 Large 0.0 25.0 0.0 14.9 0.0 11.2 Open-source Unified MLLMs

BAGEL (thinking) 0.0 21.1 0.0 12.8 0.0 7.7 BAGEL 0.0 17.9 0.0 11.5 0.0 7.1 Show-o2-7B 0.0 18.1 0.0 10.2 0.0 7.9 Show-o2-1.5B-HQ 0.0 17.1 0.0 8.7 0.0 6.9 BLIP3o-NEXT-GRPO-Text-3B 0.0 20.2 0.0 12.3 0.0 8.1 BLIP3o-8B 0.0 10.6 0.0 5.8 0.0 5.1 Janus-Pro 0.0 13.6 0.0 9.9 0.0 6.5 Emu3 0.0 14.2 0.0 7.5 0.0 6.0

(b) Subject Knowledge Difficulty

(a) GenExam-Mini

###### Table 6. Results on GenExam-Mini and on different subject knowledge difficulty levels.

Chemical Structures

Geometric Shapes

Plots & Charts

Sheet Music

Trees & Graphs

Diagrams

Maps

Other

Model

Str Rel Str Rel Str Rel Str Rel Str Rel Str Rel Str Rel Str Rel Closed-source Models

Nano Banana Pro 51.2 85.2 71.6 94.5 56.5 84.2 85.7 97.6 82.6 96.3 61.7 90.8 63.9 89.3 65.5 89.8 GPT-Image-1.5 31.0 73.7 46.5 85.0 29.0 66.4 53.6 92.7 44.9 84.2 30.0 71.1 27.8 66.2 41.4 77.9 GPT-Image-1 7.1 45.5 15.6 66.3 7.3 54.3 8.0 68.5 14.1 64.3 10.0 54.0 8.3 52.0 13.8 61.1 Seedream 4.5 4.8 44.8 16.4 66.2 1.6 42.4 0.0 54.2 15.2 63.8 3.3 38.1 11.1 52.0 10.3 60.1 FLUX.2 max 6.0 46.1 9.7 67.2 8.1 56.1 7.1 67.6 6.5 57.5 6.7 48.5 11.1 51.4 10.3 60.8

- Imagen-4-Ultra 6.0 35.6 10.5 60.4 3.2 37.9 5.4 57.7 7.2 52.5 0.0 39.5 0.0 32.2 6.9 47.9 Open-source T2I Models FLUX.2 dev 4.8 28.5 2.4 47.7 1.6 32.2 3.6 52.1 2.9 37.0 0.0 34.4 0.0 28.8 0.0 43.3 Qwen-Image 0.0 12.9 0.4 29.4 0.0 22.6 0.0 41.5 0.0 20.0 0.0 23.9 0.0 18.9 0.0 22.6 HiDream-I1-Full 0.0 10.3 0.0 22.7 0.0 21.2 0.0 34.2 0.0 17.7 0.0 21.9 0.0 17.0 0.0 20.0 Stable Diffusion 3.5 Large 0.0 8.1 0.0 17.7 0.0 13.1 0.0 29.3 0.0 9.1 0.0 24.9 0.0 7.8 0.0 11.2 Open-source Unified MLLMs BAGEL (thinking) 0.0 10.5 0.0 13.8 0.0 13.6 0.0 22.5 0.0 7.5 0.0 15.5 0.0 8.4 0.0 9.0 BAGEL 0.0 7.9 0.0 11.7 0.0 17.1 0.0 19.1 0.0 6.4 0.0 14.8 0.0 9.8 0.0 6.8 Show-o2-7B 0.0 4.3 0.0 13.1 0.0 10.6 0.0 22.8 0.0 7.2 0.0 8.9 0.0 3.8 0.0 9.3 Show-o2-1.5B-HQ 0.0 4.4 0.0 11.5 0.0 6.8 0.0 22.6 0.0 7.3 0.0 7.8 0.0 4.3 0.0 7.8 BLIP3o-NEXT-GRPO-Text-3B 0.0 7.8 0.0 12.9 0.0 16.6 0.0 16.7 0.0 8.6 0.0 16.4 0.0 13.4 0.0 14.9 BLIP3o-8B 0.0 4.3 0.0 7.2 0.0 8.1 0.0 16.5 0.0 2.6 0.0 6.5 0.0 3.4 0.0 5.6 Janus-Pro 0.0 9.3 0.0 9.5 0.0 15.4 0.0 11.9 0.0 4.3 0.0 15.0 0.0 4.7 0.0 11.4 Emu3 0.0 0.0 0.0 12.7 0.0 5.0 0.0 13.7 0.0 4.0 0.0 5.8 0.0 30.0 0.0 0.0

Table 7. Strict scores (Str) and relaxed scores (Rel) on GenExam for different image types.

BAGEL(thinking)

NanoBananaPro

HiDream-I1-Full

Imagen-4-Ultra

GPT-Image-1.5

GPT-Image-1

Seedream4.5

FLUX.2max

Qwen-Image

Show-o2-7B

FLUX.2dev

Level-2 Taxonomy

Biology/Ecology 100.0 86.9 47.1 26.1 47.8 77.5 40.6 19.7 24.0 21.4 8.7 Biology/Genetics and Evolution 94.9 87.6 66.8 75.9 64.4 57.7 36.2 22.3 10.6 8.4 6.2 Biology/Physiology and Molecular Process 97.8 92.5 73.0 75.0 75.4 62.1 47.8 23.2 20.5 9.3 8.6 Biology/Structure and Morphology 94.7 92.5 76.5 79.0 77.0 72.4 62.2 39.0 33.6 19.1 16.2 Chemistry/Chemical Equilibrium 100.0 97.3 88.7 83.7 67.0 50.8 56.3 29.1 15.1 15.4 12.4 Chemistry/Chemical Kinetics and Thermochemistry 99.0 97.5 87.8 76.8 86.4 73.5 62.3 34.9 22.1 11.6 1.4 Chemistry/Chemical Reaction 83.4 66.3 34.2 34.1 34.5 30.2 20.0 3.5 4.9 2.3 0.3 Chemistry/Electrochemistry 100.0 87.8 74.4 71.6 75.6 70.4 39.9 24.1 16.8 12.3 2.2 Chemistry/Structure Of Matter 88.4 79.5 54.5 48.5 56.9 45.4 34.1 17.0 16.2 16.2 6.9 Computer/Data Structures and Algorithms 88.4 67.4 52.1 51.2 49.0 34.0 27.4 20.3 17.4 7.6 3.9 Computer/Hardware Architecture 96.0 77.4 48.2 62.1 63.3 41.7 36.6 20.1 12.2 2.4 3.0 Computer/Networking and Systems 95.9 83.1 59.8 61.6 57.3 50.6 35.0 10.9 12.9 1.9 2.7 Computer/Theory and AI 93.7 88.9 66.2 67.0 66.8 46.6 32.9 19.1 14.5 7.9 8.7 Economics/Finance and Decision 92.0 86.8 74.4 68.1 72.6 47.9 37.6 32.4 17.2 13.5 7.0 Economics/Macroeconomics 97.7 83.5 65.5 67.4 61.3 59.4 43.9 19.9 21.0 5.2 7.7 Economics/Microeconomics 97.3 88.2 64.8 67.1 60.0 62.1 44.2 18.9 15.9 6.7 6.1 Engineering/Civil and Architecture 95.8 87.5 66.4 67.5 67.3 64.7 51.3 34.9 30.2 13.9 16.0 Engineering/Mechanical Engineering 93.8 87.5 64.4 69.8 69.7 68.3 55.5 38.0 27.4 15.5 11.5 Engineering/Mechanics and Dynamics 95.6 84.0 61.7 70.5 40.2 45.6 44.9 15.3 9.7 0.6 8.5 Engineering/Mechanics of Materials 96.1 89.5 69.8 68.4 68.4 49.9 33.8 17.2 21.8 8.7 4.7 Engineering/Surveying and Cartography 98.9 84.6 74.5 68.7 80.0 72.0 60.1 37.8 33.7 12.8 20.9 Engineering/Thermodynamics and Energy 94.5 84.7 63.1 71.6 74.4 72.9 43.4 32.4 20.0 6.6 10.3 Geography/Earth Science 97.8 94.5 77.9 75.8 82.8 75.7 68.7 56.0 38.5 30.2 37.6 Geography/Human and Ecology 93.5 98.2 61.9 51.7 50.2 36.1 22.9 20.4 17.3 1.2 6.2 Geography/Maps 94.9 88.7 69.8 58.1 70.8 58.9 60.0 44.7 35.7 30.2 31.2 History/Historical Data Change 100.0 85.3 62.3 71.8 77.7 57.9 48.6 31.0 25.3 15.5 8.6 History/Historical Map 99.8 94.8 68.6 50.1 64.5 56.4 47.4 40.5 33.9 16.3 15.9 History/Others 100.0 49.2 65.0 74.8 91.1 78.8 45.2 34.8 19.5 12.8 13.1 Mathematics/Analytic Geometry 90.7 68.1 49.1 45.7 38.7 28.7 25.3 16.4 12.4 6.5 10.0 Mathematics/Plane Geometry 83.7 63.0 52.1 43.4 54.0 36.2 33.8 18.2 18.3 14.4 11.0 Mathematics/Solid Geometry 83.9 76.0 65.1 50.0 63.6 70.5 47.1 37.3 27.3 16.7 13.5 Music/Chord Structures and Interval Diagrams 93.4 52.4 36.6 48.1 36.4 21.2 19.1 8.1 13.5 1.0 1.0 Music/Key Signatures and Time Signatures 85.4 55.7 43.0 29.7 37.5 28.6 28.3 20.4 20.7 12.9 7.1 Music/Notes and Rests 91.5 77.2 57.7 37.3 50.9 43.7 37.3 27.2 23.1 17.9 10.7 Physics/Circuits and Electronics 90.2 73.2 55.1 55.6 47.1 39.1 29.1 14.7 8.8 7.2 4.6 Physics/Electromagnetism 97.5 91.9 71.9 66.1 77.4 66.0 53.0 35.1 26.0 24.6 18.2 Physics/Mechanics 99.4 92.9 74.9 68.2 76.7 71.5 50.7 37.7 23.6 14.0 16.2 Physics/Optics and Waves 95.2 87.7 74.5 74.9 71.1 69.8 54.7 28.8 24.8 20.7 19.0 Physics/Quantum Mechanics 94.9 85.1 63.4 63.5 66.4 64.8 46.1 28.6 25.1 24.1 19.5 Physics/Thermodynamics 96.6 93.5 68.8 61.9 53.4 51.6 38.9 22.9 8.2 5.1 7.5

###### Table 8. Relaxed scores on level-2 raxonomy.

BAGEL(thinking)

NanoBananaPro

HiDream-I1-Full

Imagen-4-Ultra

GPT-Image-1.5

GPT-Image-1

Seedream4.5

FLUX.2max

Qwen-Image

Show-o2-7B

FLUX.2dev

Subject Dimension

Semantic Correctness 0.96 0.91 0.72 0.74 0.78 0.69 0.59 0.32 0.26 0.1 0.12 Spelling 1.9 1.95 1.53 1.72 1.05 1.21 0.7 0.28 0.25 0.17 0.04 Logical Consistency 1.91 1.76 1.32 1.32 1.3 1.17 0.73 0.51 0.31 0.39 0.4 Readability 1.98 1.94 1.74 1.74 1.69 1.63 1.28 1.15 1.24 1.01 0.37

Biology

Semantic Correctness 0.86 0.75 0.46 0.42 0.53 0.43 0.3 0.14 0.1 0.08 0.04 Spelling 1.97 1.87 1.66 1.59 1.21 1.06 0.88 0.23 0.27 0.37 0.03 Logical Consistency 1.72 1.37 0.86 0.69 0.72 0.57 0.46 0.14 0.12 0.37 0.21

Chemistry

- Readability 1.96 1.94 1.72 1.6 1.43 1.27 1.15 0.77 0.86 0.58 0.19

Economics

Semantic Correctness 0.97 0.84 0.64 0.65 0.64 0.61 0.44 0.2 0.18 0.05 0.07 Spelling 2.0 1.94 1.66 1.88 1.35 1.34 1.06 0.39 0.19 0.0 0.01 Logical Consistency 1.95 1.52 1.06 0.88 0.69 0.78 0.44 0.16 0.19 0.19 0.29

- Readability 1.97 1.82 1.53 1.57 1.29 1.27 1.1 0.69 0.88 0.36 0.14

Semantic Correctness 0.95 0.84 0.62 0.68 0.71 0.67 0.5 0.33 0.19 0.08 0.11 Spelling 1.95 1.97 1.64 1.79 1.29 1.36 1.0 0.5 0.66 0.23 0.12 Logical Consistency 1.85 1.64 1.17 1.08 1.05 1.01 0.59 0.41 0.35 0.34 0.41

Engineering

- Readability 1.95 1.86 1.64 1.56 1.46 1.41 1.19 0.92 1.15 0.48 0.31

Geography

Semantic Correctness 0.96 0.9 0.72 0.65 0.78 0.68 0.66 0.49 0.3 0.2 0.32 Spelling 1.95 1.97 1.51 1.61 1.3 1.08 0.82 0.65 0.73 0.68 0.39 Logical Consistency 1.89 1.88 1.41 1.12 1.32 1.18 1.03 0.91 0.82 0.91 0.92

- Readability 1.95 2.0 1.8 1.65 1.7 1.55 1.47 1.45 1.45 1.24 0.83

Semantic Correctness 1.0 0.89 0.66 0.51 0.71 0.6 0.52 0.41 0.32 0.11 0.14 Spelling 2.0 2.0 1.41 1.56 1.1 0.83 0.56 0.39 0.27 0.15 0.12 Logical Consistency 2.0 1.76 1.15 0.83 0.98 0.8 0.56 0.39 0.39 0.66 0.49 Readability 2.0 1.95 1.63 1.49 1.66 1.46 1.12 1.24 1.2 0.9 0.34

History

Semantic Correctness 0.84 0.6 0.43 0.37 0.46 0.3 0.26 0.13 0.09 0.04 0.08 Spelling 1.97 1.95 1.77 1.73 1.53 1.38 1.28 0.81 0.83 0.53 0.18 Logical Consistency 1.66 1.12 0.94 0.59 0.61 0.4 0.3 0.19 0.27 0.58 0.52 Readability 1.89 1.7 1.62 1.45 1.3 1.16 1.08 0.9 0.93 0.59 0.38

Math

Semantic Correctness 0.9 0.64 0.42 0.28 0.36 0.27 0.23 0.13 0.08 0.07 0.05 Spelling 1.98 1.97 1.74 1.62 1.63 1.51 1.38 0.91 0.88 0.86 0.57 Logical Consistency 1.74 1.26 0.98 0.54 1.02 0.74 0.57 0.42 0.75 0.23 0.18 Readability 1.92 1.89 1.89 1.57 1.71 1.63 1.54 1.49 1.54 0.86 0.28

Music

Semantic Correctness 0.95 0.83 0.62 0.6 0.63 0.57 0.42 0.25 0.12 0.09 0.12 Spelling 1.97 1.96 1.74 1.82 1.42 1.32 1.03 0.45 0.52 0.42 0.16 Logical Consistency 1.84 1.65 1.16 0.85 0.88 0.83 0.5 0.42 0.25 0.42 0.31 Readability 1.96 1.9 1.75 1.64 1.52 1.36 1.12 0.91 1.06 0.66 0.26

Physics

Semantic Correctness 0.91 0.72 0.48 0.53 0.54 0.38 0.27 0.16 0.11 0.03 0.05 Spelling 1.98 1.94 1.75 1.84 1.49 1.25 1.1 0.66 0.52 0.29 0.05 Logical Consistency 1.75 1.31 1.05 0.8 0.84 0.47 0.33 0.21 0.25 0.24 0.14 Readability 1.93 1.84 1.66 1.51 1.42 1.04 0.97 0.75 0.81 0.29 0.11

Computer Science

###### Table 9. Scores of each dimension for all subjects.

### D. Full Taxonomy List

In the table below, we show the complete list of four-level taxonomy and numbers of samples (denoted in parentheses). Blue text indicates mapping with ISCED-F (UNESCO, 2013) codes.

Level 1 Level 2 Level 3 Level 4

Food Chain and Food Web (4) Niche (1)

Ecology (5) 0521 Environmental sciences Ecosystem (5)

Evolutionary Tree (2) Pedigree Dominant and Recessive Genetic Diseases (1)

Evolution and Population Genetics (3)

Gene Linkage Map (1) Gene Structure (1) Mendelian Genetics (7) Transcription and Splicing Mechanisms (1)

Genetics and Evolution (13) 0511 Biology

Genetics (10)

Cell Physiology (3) Cell Division (3)

Metabolism (1) Microbiology (1) Signaling and Regulation (2)

Molecular Mechanism (4)

Genetic Information Transmission (12) Material Metabolism (3)

Molecular Mechanisms (15)

General Adaptation Syndrome (1) Infection Spread (2) Pharmacological Dose Response Curve (1) Stress Response Model (1) Tumor and Inflammation (1)

Physiology and Molecular Process (49)

- 0511 Biology
- 0512 Biochemistry 0912 Medicine 0916 Pharmacy

Pathology and Pharmacology (6)

Blood Coagulation (1) Germ Layer Differentiation (1) Immunity (3) Nervous System (13) Photosynthesis (2) Respiration and Gas Exchange (1)

Systemic Physiology (21)

Biology (156)

Basic Cell Structure (18) Microbial Morphology (4) Special Cells (5)

Cell Structure (27)

Biomacromolecules (4) Biomolecules (2)

Molecular Structure (6)

Anal Canal (1) Bladder (1) Brain (16) Digestive Tract (3) Ear (1) Eye (4) Heart (4) Joint Structure (3) Kidney (2) Larynx (1) Lung (1) Muscular and Skeletal System (2) Nervous System (1) Pancreas (1) Plant Seed (1) Plant Stem Tissue (2) Skin (1) Vascular System (3)

Structure and Morphology (89)

- 0511 Biology
- 0512 Biochemistry

Organ Structure (48)

Others (2) Optical Microscope Structure (2)

Epithelial Tissue (3) Plant Vascular Tissue (3)

Tissue Structure (6)

Chemical Equilibrium (5) Acid Base Titration (2)

Solubility (3) Chemical Kinetics and Thermochemistry (7)

Inorganic Reaction (2) Organic Reaction (27) Reaction Mechanism (3) Electrochemistry (6)

Chemical Reaction (32)

Chemistry (118) 0531 Chemistry

Atomic Model (6) Electron Configuration (8) Crystal Structure (3) Element Abundance Distribution (2)

Atomic Structure (14)

Structure Of Matter (68)

Electron Configuration and Intermolecular Forces (8)

Isomers (4) Molecular Structure Diagram (11) Organic Compound (26)

Molecular Structure (49)

ER Diagram (2)

Adjacency List (3) Directed Graph (5) Graph Algorithms (4) Maximum Flow (1) Shortest Path (1) Undirected Graph (10) Linked List (2) Queue (1) Sorting (2)

Graph (24)

Data Structures and Algorithms (50) 0613 Software and applications development and analysis

B Tree (1) Others (2) Search Tree (7) Syntax Tree (2) Traversal (6)

Tree (18)

Bus Structure (3) Cache (3) Others (1)

Hardware Architecture (17) 0613 Software and applications development and analysis

Logic Gates (5) Multiplexer Application (2) Instruction Format (1) Pipeline (3)

Digital Circuits (7)

Computer (102)

Networking and Systems (11)

- 0612 Database and network design and

administration

Computer Networks (7)

Packet Structure (3) TCP and IP Protocol Stack (2) Topology (2)

Operating System (4)

Deadlock Resource Allocation Diagram (1)

Memory Paging (2) Process Scheduling (1)

Theory and AI (24)

- 0613 Software and

Compiler Principles (2) Finite Automaton (8)

Learning Rate Impact (2) Multicollinearity (1) Neural Networks (3) Overfitting and Underfitting Diagram (1) ROC Curve (1) Sampling Methods (1) Training and Testing Curves (4) Variance Bias Tradeoff (1)

applications development and analysis 0619 Artificial Intelligence

Machine Learning (14)

CAPM (2) Beta Coefficient Scatter Plot (1) Risk Return Chart (3)

Finance and Decision (5) 0412 Finance, banking and insurance

AD AS Model Aggregate Demand and Aggregate Supply (29)

Business Cycle (1) Equilibrium Unemployment (1) Exchange Rates and Monetary Policy (1) IS LM Model (1) Laffer Curve (1) Lorenz Curve (1) Other Economic Statistical Charts (3) Other Macroeconomic Theories (1)

Macroeconomics (42) 0311 Economics

Economics (77)

Phillips Curve Inflation and Unemployment (3)

Cost Curves (4) Expected Utility Curve (1) Game Theory (1)

Budget Constraint Line (1) Indifference Curve Utility Maximization (6)

PPF and Utility Curves (7)

Microeconomics (30) 0311 Economics

Consumer and Producer Surplus (2) Elasticity Price and Income (4) Equilibrium Price (9) Price Ceiling (1) Tax Impact (1)

Supply Demand Curves (17)

Building Structure (4)

Detail Drawings (4) Plan View (1) Sectional and Profile Views (6) Geotechnical Engineering (3) Roads and Bridges (1) Mechanical Engineering (33) 0715 Mechanics and metal trades

Civil and Architecture (19) 0731 Architecture and town planning 0732 Building and civil engineering

Engineering Drawings (11)

Schematic Diagram (18) Section View (14) Machine Parts (1) Simple Parts (1)

Engineering Drawings (32)

Block Diagram (4) Damping and Vibration (5) Mechanics of Materials (11) 0715 Mechanics and metal trades

Mechanics and Dynamics (9) 0715 Mechanics and metal trades

Vibration and Control (9)

Engineering (111)

Stress Strain Relationship (11)

Data Processing and Adjustment (3) Maps and Diagrams (2) Sectional and Profile Views (4)

Surveying and Cartography (9) 0715 Mechanics and metal trades

Energy (2) Energy Changes (5) States and Phase Changes (10) Thermal Cycle (13)

Thermodynamics and Energy (30) 0713 Electricity and energy

Astronomy (3)

Atmospheric Circulation (2) Climate Chart (2) Energy Balance (1) Temperature Change (1) Water Cycle (10) Geological Age (1) Geomagnetic Field (1) Greenhouse Effect (1) Landforms and Geology (14) Latitude and Longitude (1)

Climate and Circulation (16)

Earth Science (37) 0532 Earth sciences

Geography (66)

Population and City (3) Urban Planning Map (1)

Human and Ecology (4) 0314 Sociology and cultural studies

Earthquake Belt Distribution (2) Tropical Temperate and Frigid Zones (2) World and Regional Map (21)

Maps (25) 0314 Sociology and cultural studies

Others (2) Population (3) Unemployment Rate (2)

Historical Data Change (7)

History (41) 0222 History and archaeology

Others (6) Trade (6)

Historical Map (32) Route Map (12)

Territory Map (20) Others (2)

Absolute Value Function (1) Definite Integral Area (12) Exponential and Logarithmic Function (2) Geometric Meaning Of Derivative (1)

Inequality Region (2) Linear Programming (2)

Analytic Geometry (56)

Linear Function (3) Other Function (14) Parametric Equation and Polar Curve (4) Piecewise Function (8) Quadratic Function (6) Trigonometric Function (3)

Angle (3)

Chord (8) Inscribed and Circumscribed Circle (6) Others (4) Tangent (4) Complex Geometry Problem (23)

Circle (22)

Mathematics (151) 0541 Mathematics

Other Polygon (2) Other Quadrilateral (5) Parallelogram (2) Pentagon (2) Rectangle (10) Regular Hexagon (1) Trapezoid (3)

Rectangle and Polygon (25)

Plane Geometry (84)

Altitude (1) Angle Bisector (1) Congruence (1) Others (1) Perpendicular Bisector (1) Right Triangle (4) Similarity (2)

Triangle (11)

Cylinder and Cone (4)

Oblique Prism (1) Right Prism (1) Pyramid (2) Regular Pyramid (2) Section (2) Straight Cut (2) Sphere (1) Tangent Plane (1)

Prism (2)

Solid Geometry (11)

Perfect Fifth and Major Third (3) Triads and Seventh Chords (7)

Chord Structures and Interval Diagrams (10)

Circle Of Fifths (3) Compound Triple Time (1) Key Signature (4) Notes and Rests (47)

Music (65) 0215 Music and performing arts

Key Signatures and Time Signatures (8)

Curves In Circuit (2) DC and AC Circuit (19) Kirchhoff Law Application (2) Op Amp Circuit (2)

Circuit Diagram (25)

Resistor (1) Transistor (2)

Circuits and Electronics (36)

Component Symbols and Combinations (3)

Filter (1) Spectrum Diagram (2) Waveform Diagram (5)

Signal and Electronics (8)

Electric and Magnetic Field (8)

Ampere Force (1) Energy Loss and Hysteresis Loop (2) Lorentz Force (4)

Electromagnetism (15)

Electromagnetic Induction (7)

Gas Pressure (3) Liquid Pressure (15) Principle (5) Kinematics (4) Newtonian Mechanics (1)

Fluid Mechanics (23)

Physics (113) 0533 Physics 0714 Electronics and automation

Mechanics (28)

Interference and Diffraction (2) Lens Imaging (3) Reflection and Refraction (4) Wave (2)

Ray Diagram (9)

Optics and Waves (11)

Atomic Structure (2) Electromagnetic Wave (1) Photoelectric Effect (2) Potential Well (1) Spectrum (4)

Quantum Mechanics (10)

Intermolecular Forces (2) Molecular Speed Distribution (1) Phase Change (8) Pressure Volume Temperature Relationship (2)

Thermodynamics (13)

### E. Prompts

#### E.1. Automatic Filtering

You are an expert-level visual reasoning evaluator. You will be given an image. Your task is to assess a given image for its suitability as a basis for multidisciplinary (i.e. requires subject knowledge such as math, physics, chemistry, biology, etc.) image generation. From each of the perspectives below, you should assign a score with a confidence score and a short explanation.

## Perspectives

- 1. Text Richness Score (range: 0-10): If the image content is pure table/text without any other content (e.g. graphs, diagrams, plots, icons, geometry, etc.), it is 0. If the image does not contain any text, it is 10. For other cases, the score is based on the complexity of the image content.
- 2. Image Domain (range: 0-1): If the image is a natural image (i.e., real image) / pathological image (also including body scans, MRI, CT scans, and X-rays), it is 0, otherwise it is 1.
- 3. Image Complexity (range: 1-10): Whether the image is too complex to draw (consider the number and complexity of components,objects, text, etc.), very complex is 1, very simple is 10.
- 4. Subject Knowledge (range: 1-10): Whether the model needs **subject knowledge and reasoning** to generate this image (note: the prompt is not a complete description of the image, it is related to subject knowledge), no need is 1, need a lot is 10. ## Instructions For each domain, provide:

- - **Score**: An integer, based on the range defined above.
- - **Confidence Score**: An integer between 1 and 5.
- - **Short Explanation**: 1-2 sentences justifying the score, referencing the image content and its alignment with the domain definition. Make sure your explanation is concise. ## Output Format Respond STRICTLY in the following format: {{

“Text Richness Score”: {{ “score”: 0, “confidence”: 0, “explanation”: “”

}}, “Image Domain”: {{

“score”: 0, “confidence”: 0, “explanation”: “”

}}, “Image Complexity”: {{

“score”: 0, “confidence”: 0, “explanation”: “”

}}, “Subject Knowledge”: {{

“score”: 0, “confidence”: 0, “explanation”: “”

}},

}} Make sure your response follows the format strictly and can be parsed by Python json.loads. Do not include any other explanation.

#### E.2. Annotating

You are an expert-level prompt generator for multidisciplinary text-to-image generation. You are given an image, a related conversation between a user and a model, and its corresponding taxonomy.

Your job is to provide a **prompt (like an academic question)** for multidisciplinary text-to-image generation corresponding to this image. You will also generate some **scoring points** when evaluating whether some model generates the correct image corresponding to this prompt.

## Instructions ### Prompt

- 1. The prompt should **NOT** be a complete description of the image. It should be in a style similar to an **academic question** that **requires some subject knowledge and reasoning** to generate the image. If a model does not have such subject knowledge, it is expected to fail to generate the correct image.
- 2. The prompt should be precise and concise (less than 200 words) and should be in English.
- 3. The provided conversation and text in the image can sometimes help you to generate the prompt, but you should consider them carefully since they may contain some irrelevant information. ## Scoring Points

- 1. The scoring points are in the form of a list of questions and their corresponding scores, answered by “Yes” or “No” only. They are designed so that by answering the questions, we can evaluate whether the model can generate the correct image corresponding to the prompt. It is expected that the answers are all “Yes” if the model generates the correct image.
- 2. The number of scoring points should be no more than 20. If the image is simple and requires little subject knowledge and reasoning, the number of scoring points can be small.
- 3. The questions should be based on the prompt instead of the image, i.e. since the model is required to generate the image based on the prompt, the questions should not focus on information or components that are in the ground truth image but not in the prompt.
- 4. The questions should not be too general. It should provide **details** rather than “Is xxx correct?”, e.g. the structure of a molecule (bonds, position of atoms, etc.), characteristics that a curve should satisfy, etc. It should not be too complex as you should break it down into several sub-questions.
- 5. The score of each scoring point should be a number between 0 and 1, indicating the proportion of this question in the total score of the image.**The sum of all scores should be 1**.
- 6. Typically, the score of the most basic question (e.g. checking the overall structure of the image) should be small, such as 0.1 or 0.2. For complex questions, questions requiring subject knowledge, or questions about details, the score should be large. ## Examples

- 1. Prompt: “Generate a planar molecular structure diagram of benzene (C6H6), showing its conjugated double bond structure, and distinguish carbon atoms and hydrogen atoms with different colors.”, Scoring points: [{{“question”: “Is the number of carbon atoms 6?”, “score”: 0.2}}, {{“question”: “Is the number of hydrogen atoms 6?”, “score”: 0.2}}, {{“question”: “Is the number of double bonds 3?”, “score”: 0.2}}, {{“question”: “Is the number of single bonds 3?”, “score”: 0.2}}, {{“question”: “Are carbon atoms and hydrogen atoms in different colors?”, “score”: 0.2}}]
- 2. Prompt: “Generate the graph of the function y = ex.”, Scoring points: [{{“question”: “Does the image shows a graph of a function?”, “score”: 0.1}}, {{“question”: “Does the curve pass through the point (0, 1)?”, “score”: 0.2}}, {{“question”: “Does the curve asymptotically approach the x-axis when x approaches negative infinity?”, “score”: 0.2}}, {{“question”: “Does the curve tends to positive infinity when x approaches positive infinity?”, “score”: 0.2}}, {{“question”: “Is the curve smooth and continuous?”, “score”: 0.1}}, {{“question”: “Is the slope always positive and increasing when x increases?”, “score”: 0.2}}]
- 3. Prompt: “Generate a schematic diagram of an animal cell structure. Label cell membrane, cytoplasm, nucleus, and mitochondria.”, Scoring points: [{{“question”: “Does the image contain a cell structure?”, “score”: 0.1}}, {{“question”: “Does the cell membrane label correspond to the correct position in the cell?”, “score”: 0.2}}, {{“question”: “Does the cytoplasm label correspond to the correct position in the cell?”, “score”: 0.2}}, {{“question”: “Does the nucleus label correspond to the correct position in the cell?”, “score”: 0.2}}, {{“question”: “Does the mitochondria label correspond to the correct position in the cell?”, “score”: 0.3}}] ## Output Format Respond STRICTLY in the following format: {{

“prompt”: “”, “scoring points”: [

{{“question”: “question1”, “score”: 0.1}},

... ],

}} Make sure your response follows the format strictly and can be parsed by Python json.loads. Do not include any other explanation. ## Taxonomy of this image {taxonomy}

## Conversation {conversations}

#### E.3. Evaluation

You are an expert-level evaluator for multidisciplinary text-to-image generation. You will be given an input prompt for text-to-image generation, an image generated by a model (the first image), a ground truth image for reference (the second image), and several scoring point questions and their corresponding scores. Your task is to evaluate the generated image by answering the scoring point questions and evaluating the image from some global perspectives.

## Instructions

- 1. Remember that the first image is the generated image from a model, and the second image is the ground truth image. You should **only evaluate the generated image**, while the ground truth image can be used for reference.
- 2. First, give a detailed description of all the components in the generated image.
- 3. For each scoring point, you should use the ground truth image as **reference information** to make more accurate judgments. You should generate a **detailed** reasoning **step by step**. It should first analyze the question, what subject knowledge it requires and what information you need to refer to from the ground truth image, then analyze whether the generated image satisfies the scoring point based on the scoring point, prompt and reference information. Finally, answer 1 if the generated image fully satisfies the scoring point, otherwise answer 0. For example, if a scoring point is ”Does the cell membrane label correspond to the correct position in the cell?”, you can refer to the ground truth image to check the correct position of the cell membrane label as an example, by **explicitly analyzing the ground truth image** in your reasoning.
- 4. You should also evaluate the generated image from some global perspectives provided below. For each perspective, first provide a **detailed step-by-step** reasoning, e.g. recognize all the text labels for spelling, then give a score in the defined range. ## Global Perspectives

- 1. Spelling (range: 0-2): The spelling of the text in the image, including the notations and equations. You should first recognize the text in the image in the reasoning, then check the spelling of the text. Specifically:

- - 0: There are critical errors in spelling, notations or equations which significantly hinders the understanding of the image.
- - 1: There are some errors in spelling, notations or equations that somehow hinder the understanding of key information in the image.
- - 2: All or almost all the spelling, notations and equations are correct. Tiny errors like commas, capital letters, etc. are allowed.

- 2. Readability (range: 0-2): The readability of the image. Each component in the image should be clearly readable and identifiable. All text labels and marks should in the right place and are not overlapped or occluded by other elements. If the image is geometry or diagram, there should not be unlabeled points or lines or duplicated labels. If the image is a plot or chart, the axis should be labeled and the ticks should be carefully checked. You should first identify the components, labels, and marks in the image in the reasoning, then check their readability. Specifically:

- - 0: There are critical readability issues which significantly hinders the understanding of the image e.g. some components are impossible to distinguish, or some labels are fully overlapped or occluded, or many key information are not labeled.
- - 1: There are some readability issues, e.g. some components are not clearly readable and identifiable, or some labels are overlapped or occluded, or some key information are not labeled.
- - 2: The readability is perfect or almost perfect. Components are clearly readable and identifiable and necessary labels and marks are present. Tiny errors are allowed.

- 3. Logical Consistency (range: 0-2): The logical consistency of the image. Check the correctness of all the marks, text, musical notes, etc. If the image is geometry, check the correctness of each marked angles, lengths, coordinates, etc. If the image is a plot or chart, check the correctness of each data point, the axis, the ticks, the legend, etc. You should first identify the components, labels, and marks in the image in the reasoning, then check their logical consistency. Specifically:

- - 0: There are critical logical consistency issues which significantly hinders the understanding of the image, e.g. some key marks are not correct, some angles/lengths/data points are significantly inconsistent with the text label, etc.
- - 1: There are some logical consistency issues that somehow hinder the understanding of key information in the image, e.g. some marks are not correct, some angles/lengths/data points are inconsistent with the text label, etc.
- - 2: The logical consistency is perfect or almost perfect. Marks, text, etc. are correct and consistent. Tiny errors are allowed. ## Output Format Respond STRICTLY in the following format: {{

”description”: ”...”, ”answers”: [

{{

”reasoning”: ”...”, ”answer”: 1

}},

...

], ”global evaluation”: {{

”Spelling”: {{ ”reasoning”: ”...”, ”score”: 0,

}}, ”Readability”: {{

”reasoning”: ”...”,

”score”: 0, }}, ”Logical Consistency”: {{

”reasoning”: ”...”, ”score”: 0,

}}, }}

}} Make sure your response follows the format strictly and can be parsed by Python json.loads. Do not include any other explanation. ## Prompt {prompt}

## Scoring Points {scoring points}

- F. More Visualization More visualization of generated images are given in Figs. 11, 12 and 13.

Computer Science

Prompt: Draw a resource allocation graph for an operating system scenario where there are two processes (Process 1 and Process 2) and two resources (Resource 1 and Resource 2). Process 1 is assigned to

Index Scoring Point Question Score

- 1 Are there exactly two processes and two resources shown in the diagram? 0.1
- 2 Is Process 1 assigned to Resource 1? 0.1
- 3 Is Process 1 waiting for Resource 2? 0.15
- 4 Is Process 2 assigned to Resource 2? 0.1
- 5 Is Process 2 waiting for Resource 1? 0.15
- 6 Are the assignment and waiting relationships clearly indicated with arrows in the correct directions? 0.2
- 7 Is the deadlock state explicitly labeled in the diagram? 0.2

- Resource 1 and is waiting for Resource 2, while Process 2 is assigned to
- Resource 2 and is waiting for Resource 1. Clearly indicate the direction of assignment and waiting relationships, and label the deadlock state in the diagram.

Taxonomy: Computer/Networking and Systems/Operating System/ Deadlock Resource Allocation Diagram

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Ground Truth:

[Figure 91]

Nano Banana Pro GPT-Image-1.5 Seedream 4.5 Qwen-Image-2512 Flux.2 dev

- 0.3 1 0 2

Strict: 0 Relaxed: 0.36

- 1. 2. 3. 4. 5. 7.

- 0.8 2 2 2

Strict: 0 Relaxed: 0.86

- 1. 2. 3. 4. 5. 7.

- 0.3 1 0 1

Strict: 0 Relaxed: 0.31

- 1. 2. 3. 4. 5.

1. 2. 3. 4. 5. 7.

- 0.4 1 0 0

Strict: 0 Relaxed: 0.33

- 1. 2. 3. 4. 5. 7.

6. 6. 6. 6. 6. 7.

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell 1.0 2 1 2

Semantic Read Logic Spell

Strict: 0 Relaxed: 0.9

Economics

Prompt: Draw a diagram comparing the lifetime wage profiles of two individuals: one who quits after high school and one who goes to college. On the vertical axis, label wage, and on the horizontal axis, label age. Mark the ages 18, 22, and 65 on the age axis. Show that the high school graduate earns a constant wage w_HS from age 18 to 65, while the college graduate earns zero wage from 18 to 22 (with a negative value H representing college costs), then earns a higher constant wage w_C from age 22 to 65. Clearly label the areas L, M, and N corresponding to the opportunity cost, direct cost, and wage premium, respectively.

Index Scoring Point Question Score

- 1 Does the diagram have wage on the vertical axis and age on the horizontal axis? 0.1
- 2 Are the ages 18, 22, and 65 marked on the age axis? 0.1
- 3 Is there a horizontal line at wage w_HS from age 18 to 65 for the high school graduate? 0.1
- 4 Is there a segment from age 18 to 22 for the college graduate at a negative value -H? 0.1
- 5 Is there a horizontal line at wage w_C from age 22 to 65 for the college graduate? 0.1
- 6

Are the areas L, M, and N clearly labeled in the correct regions (L: foregone earnings, M: direct cost, N: wage premium)?

0.2

- 7 Is the area N above w_HS and between ages 22 and 65? 0.1
- 8 Is the area L between w_HS and zero from age 18 to 22? 0.1
- 9 Is the area M below zero (at -H) from age 18 to 22? 0.1

Taxonomy: Economics/ Microeconomics/ Cost Curves

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Ground Truth:

[Figure 97]

Nano Banana Pro GPT-Image-1.5 Seedream 4.5 Qwen-Image-2512 Flux.2 dev

- 0.9 2 1 2

Strict: 0 Relaxed: 0.88

- 1. 2. 3. 4. 5.

- 0.4 1 0 1

Strict: 0 Relaxed: 0.38

- 1. 2. 3. 4. 5.

- 0.3 1 0 1

Strict: 0 Relaxed: 0.31

- 1. 2. 3. 4. 5.

1. 2. 3. 4. 5. 7. 8. 9.

1. 2. 3. 4. 5. 7. 8. 9.

6. 6. 6. 6. 6. 7. 8. 9.

7. 8. 9.

7. 8. 9.

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell 1.0 2 1 2

Semantic Read Logic Spell 0 1 0 1

Strict: 0 Relaxed: 0.95

Strict: 0 Relaxed: 0.1

###### Figure 11. Visualization of generated images.

Biology

Prompt: Draw a tundra food web for Nunavut. Depict the following organisms in a rocky, snowy tundra landscape: bilberry, bear sedge, lichen, mushroom, earthworm, brown lemming, short-tailed weasel, Arctic fox, snowy owl, rough-legged hawk, parasitic jaeger, barrenground caribou, and grizzly bear. Use directed arrows that point from the food to the consumer. Required trophic links: bilberry → brown lemming; bear sedge → brown lemming; lichen → barren-ground caribou; brown lemming → short-tailed weasel; brown lemming → Arctic fox; brown lemming → snowy owl; brown lemming → rough-legged hawk; brown lemming → parasitic jaeger; earthworm → rough-legged hawk; barrenground caribou → grizzly bear. Do not add other links.

Index Scoring Point Question Score

- 1 Are all listed organisms depicted together in a single rocky, snowy tundra scene? 0.15
- 2 Is there an arrow from bilberry to brown lemming? 0.08
- 3 Is there an arrow from bear sedge to brown lemming? 0.08
- 4 Is there an arrow from lichen to barren-ground caribou? 0.08
- 5 Is there an arrow from brown lemming to short-tailed weasel? 0.08
- 6 Is there an arrow from brown lemming to Arctic fox? 0.08
- 7 Is there an arrow from brown lemming to snowy owl? 0.08
- 8 Is there an arrow from brown lemming to rough-legged hawk? 0.08
- 9 Is there an arrow from brown lemming to parasitic jaeger? 0.08
- 10 Is there an arrow from earthworm to rough-legged hawk? 0.08
- 11 Is there an arrow from barren-ground caribou to grizzly bear? 0.11
- 12 Do all arrows point from food to consumer, and are no additional (unlisted) trophic links present? 0.02

[Figure 98]

[Figure 99]

Taxonomy: Biology/Ecology/ Ecosystem/ Food Chain and Food Web

[Figure 100]

[Figure 101]

[Figure 102]

Ground Truth:

[Figure 103]

Nano Banana Pro GPT-Image-1.5 Seedream 4.5 Qwen-Image-2512 Flux.2 dev

1. 2. 3. 4. 5. 6. 6. 6. 6. 6. 7. 8. 9. 10.

4.

- 0.55 2 2 2

Strict: 0 Relaxed: 0.385

- 1. 3. 5.

- 0 1 0 0

Strict: 0 Relaxed: 0.05

- 1. 2. 3. 4. 5.

1. 2. 3. 5. 7. 8. 9. 10.

2. 4.

1. 2. 3. 4. 5. 7. 8. 9. 10.

7. 8.

9. 10.

7. 8. 9. 10.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

11.

12. 11. 12. 11. 12. 11. 12.

11. 12.

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

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell 0.32 1 0 0

Semantic Read Logic Spell 0.91 2 2 2

Semantic Read Logic Spell 0.4 1 0 0

Strict: 0 Relaxed: 0.274

Strict: 0 Relaxed: 0.637

Strict: 0 Relaxed: 0.33

Geography

Index Scoring Point Question Score

Prompt: Draw a cross‑sectional diagram illustrating the process of cavern formation in a limestone region, showing the unsaturated zone, water table, groundwater zone, and the direction of groundwater flow. Indicate where caverns are forming and the presence of a surface stream above the cavern. Label all key features.

- 1 Is there a distinct unsaturated zone above a saturated groundwater zone separated by a water table boundary? 0.3
- 2 Is there a surface stream depicted above the cavern location, as required by the prompt? 0.15
- 3 Is the direction of groundwater flow clearly indicated in the saturated (groundwater) zone with flowlines or arrows? 0.35
- 4

Are caverns shown forming within the limestone, and is their location indicated with a surface stream positioned above them?

0.15

- 5 Is the overall scene a lateral cross‑section through karst bedrock rather than a plan view or non‑karst landscape? 0.05

Taxonomy: Geography/Earth Science/ Landforms and Geology

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Ground Truth:

[Figure 124]

Nano Banana Pro GPT‑Image‑1.5 Seedream 4.5 Qwen‑Image‑2512 Flux.2 dev

1. 2. 3. 4. 5.

1. 2. 3. 4. 5.

1. 2. 3. 4. 5.

1. 2. 3. 4. 5.

- 0.65 2 1 1

Strict: 0 Relaxed: 0.655

- 1. 2. 3. 4. 5.

Semantic Read Logic Spell 1.0 2 2 2

Semantic Read Logic Spell 1.0 2 2 1

Semantic Read Logic Spell 1.0 1 1 1

Semantic Read Logic Spell 1.0 2 2 2

Semantic Read Logic Spell

Strict: 1 Relaxed: 1.0

Strict: 0 Relaxed: 0.95

Strict: 0 Relaxed: 0.85

Strict: 1 Relaxed: 1.0

Music

Prompt: Create an instructional poster for music theory that demonstrates transposition using the Circle of Semitones. Draw the Circle of Semitones labeled with major-key tonics around the circle in correct order. Also annotate two specific interval moves on the circle: from D to E is up a whole step (clockwise), and from B to G is down a major third (counterclockwise). Beneath the circle, show two staff examples: (1) treble clef D major (two sharps) transposed up a whole step to treble clef E major (four sharps); (2) treble clef B minor (two sharps) transposed down a major third to bass clef G minor (two flats), with arrows between each pair to show the direction of transposition.

Index Scoring Point Question Score

- 1 Are the key names arranged on a Circle of Semitones in the correct clockwise order? 0.33
- 2 Is the interval from D to E explicitly depicted as an upward whole step on the circle (clockwise)? 0.1
- 3 Is the interval from B to G depicted as a downward major third on the circle (counterclockwise)? 0.28
- 4 Does the first staff example show treble clef D major with two sharps? 0.1
- 5 Does the transposed partner show treble clef E major with four sharps? 0.1
- 6 In the second example, is B minor shown on treble clef with two sharps? 0.05
- 7 Is the transposed partner G minor shown on bass clef with two flats? 0.05

Taxonomy: Music/Key Signatures and Time Signatures/ Circle Of Fifths

[Figure 125]

Ground Truth:

[Figure 126]

Nano Banana Pro GPT-Image-1.5 Seedream 4.5 Qwen-Image-2512 Flux.2 dev

- 0.625 2 1 2

Strict: 0 Relaxed: 0.688

- 1. 2. 3. 4. 5. 7.

- 0 2 0 1

Strict: 0 Relaxed: 0.15

- 1. 2. 3. 4. 5. 7.

- 0.15 1 0 0

Strict: 0 Relaxed: 0.155

- 1. 2. 3. 4. 5.

- 0.675 2 1 2

Strict: 0 Relaxed: 0.723

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

- 1. 2. 3. 4. 5. 7.

- 0.3 0 0 0

Strict: 0 Relaxed: 0.21

- 1. 2. 3. 4. 5. 7.

6. 6. 6. 6. 6. 7.

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell

###### Figure 12. Visualization of generated images.

Chemistry

Index Scoring Point Question Score 1 Are there two separate beakers forming two half-cells connected by a U-shaped salt bridge? 0.12 2 Does one half-cell contain a Zn(s) electrode immersed in 1.0 M Zn(NO3)2(aq) with correct labeling? 0.18 3 Does the other half-cell contain a Cr(s) electrode immersed in 1.0 M Cr(NO3)3(aq) with correct labeling? 0.18 4 Is the salt bridge labeled as KNO3(aq) and shown connecting the two solutions? 0.18 5 Are the electrodes connected externally by wires to a voltmeter? 0.14 6 Are both electrodes depicted as solid metals immersed in their respective solutions? 0.1 7 Are both electrolytes clearly aqueous and shown with the stated concentration of 1.0 M in each beaker? 0.1

Prompt: Draw a labeled diagram of a galvanic cell constructed from a zinc electrode in 1.0 M Zn(NO3)2(aq) and a chromium electrode in 1.0 M Cr(NO3)3(aq), connected by a KNO3(aq) salt bridge. Clearly indicate the placement of the Zn(s) and Cr(s) electrodes, the voltmeter, and the salt bridge. Label all solutions and components.

Taxonomy: Chemistry/ Electrochemistry

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Ground Truth:

[Figure 136]

Nano Banana Pro GPT-Image-1.5 Seedream 4.5 Qwen-Image-2512 Flux.2 dev

1. 2. 3. 4. 5. 7.

- 0.7 1 1 2

Strict: 0 Relaxed: 0.69

- 1. 2. 3. 4. 5. 7.

- 0.64 2 1 0

Strict: 0 Relaxed: 0.598

- 1. 2. 3. 4. 5.

1. 2. 3. 4. 5. 7.

- 0.36 2 0 0

Strict: 0 Relaxed: 0.352

- 1. 2. 3. 4. 5. 7.

6. 6. 6. 6. 6. 7.

Semantic Read Logic Spell 1.0 2 2 2

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell 1.0 2 2 2

Semantic Read Logic Spell

Strict: 1 Relaxed: 1.0

Strict: 1 Relaxed: 1.0

Engineering

Index Scoring Point Question Score

- 1 Are five identical cylindrical rods shown as separate elements? 0.1
- 2

Is one rod depicted under axial compression with arrows along the cylinder axis pointing inward toward the rod (opposed forces pushing toward the center)?

0.15

- 3

Is one rod depicted under axial tension with arrows along the cylinder axis pointing outward away from the rod (opposed forces pulling apart)?

0.15

- 4

Is one rod depicted under shear with a pair of equal and opposite parallel forces acting tangentially and perpendicular to the rod’s axis (sliding tendency on opposite faces)?

0.15

- 5

Is one rod depicted under torsion with a torque about the rod’s axis, indicated by opposite rotational arrows at the ends or tangential force couples producing twist?

0.2

- 6

Is one rod depicted under bending with transverse forces producing a bending moment (e.g., opposite-direction arrows perpendicular to the axis at separated locations or an equivalent load/support arrangement)?

0.2

- 7 Do all rods use clear directional arrows to indicate the nature and direction of applied forces or moments unambiguously? 0.05

Prompt: Draw a schematic diagram showing five identical cylindrical rods, each subjected to a different fundamental type of mechanical loading: compression, tension, shear, torsion, and bending. For each cylinder, use arrows to clearly indicate the direction and nature of the applied forces or moments, and label each type of loading accordingly.

[Figure 137]

[Figure 138]

Taxonomy: Engineering/ Mechanics of Materials/ Stress Strain Relationship

[Figure 139]

[Figure 140]

[Figure 141]

Ground Truth:

[Figure 142]

Nano Banana Pro GPT-Image-1.5 Seedream 4.5 Qwen-Image-2512 Flux.2 dev

- 0.8 2 1 2

Strict: 0 Relaxed: 0.81

- 1. 2. 3. 4. 5. 7.

- 0.6 2 1 2

Strict: 0 Relaxed: 0.67

- 1. 2. 3. 4. 5. 7.

- 0.65 1 0 1

Strict: 0 Relaxed: 0.555

- 1. 2. 3. 4. 5.

1. 2. 3. 4. 5. 7.

1. 2. 3. 4. 5. 7.

6. 6. 6. 6. 6. 7.

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell

Semantic Read Logic Spell 0.85 1 1 2

Semantic Read Logic Spell 0.1 2 0 1

Strict: 0 Relaxed: 0.795

Strict: 0 Relaxed: 0.22

Physics

Index Scoring Point Question Score

Prompt: Draw a comparative table with two columns: the left column lists the four types of Fourier transforms (Fourier Transform, Fourier Series, Discrete Time Fourier Transform, Discrete Fourier Transform), each labeled with the type of signals they apply to (continuous/aperiodic, continuous/periodic, discrete/aperiodic, discrete/periodic). In the right column, for each type, illustrate an example signal waveform that matches the corresponding signal type (continuous aperiodic, continuous periodic, discrete aperiodic, discrete periodic).

- 1

Is the layout a clear two-column, four-row table, with the left column listing the four transforms (Fourier Transform, Fourier Series, Discrete-Time Fourier Transform, Discrete Fourier Transform) along with their applicable signal-type labels, and the right column showing corresponding example waveforms?

0.1

- 2 For the Fourier Transform row, does the example depict a continuous-time aperiodic waveform (a smooth, nonrepeating burst) 0.175
- 3 For the Fourier Series row, does the example depict a continuous-time periodic waveform that clearly repeats across multiple cycles? 0.175
- 4

For the Discrete-Time Fourier Transform row, does the example depict a discrete-time aperiodic sequence using isolated sample markers without repetition?

0.15

- 5

For the Discrete Fourier Transform row, does the example depict a discrete-time periodic sequence using sample markers that repeat a short pattern over time?

0.15

- 6

Are continuous signals rendered with continuous lines while discrete signals are rendered with discrete markers (dots or squares) rather than lines?

0.15

- 7 Are the four examples visually aligned to distinct rows so their categories are unambiguous and non-overlapping? 0.1

Taxonomy: Physics/ Circuits and Electronics/ Signal and Electronics/ Waveform Diagram

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Ground Truth:

[Figure 148]

Nano Banana Pro GPT-Image-1.5 Seedream 4.5 Qwen-Image-2512 Flux.2 dev

1. 2. 3. 4. 5. 7.

1. 2. 3. 4. 5. 7.

1. 2. 3. 4. 5.

1. 2. 3. 4. 5. 7.

1. 2. 3. 4. 5. 7.

6. 6. 6. 6. 6. 7.

Semantic Read Logic Spell 1.0 2 2 2

Semantic Read Logic Spell 0.55 2 0 2

Semantic Read Logic Spell 0.35 2 0 0

Semantic Read Logic Spell 1.0 2 2 2

Semantic Read Logic Spell 0.275 2 0 1

Strict: 1 Relaxed: 1.0

Strict: 0 Relaxed: 0.585

Strict: 0 Relaxed: 0.345

Strict: 0 Relaxed: 1.0

Strict: 0 Relaxed: 0.343

###### Figure 13. Visualization of generated images.

