### MORPHOBENCH: A Benchmark with Difficulty Adaptive to Model Reasoning

Xukai Wang1,4*, Xuanbo Liu1,3*, Mingrui Chen1,4*, Haitian Zhong1,4*, Xuanlin Yang1,2*, Bohan Zeng2*, Jinbo Hu2*, Hao Liang1,2, Junbo Niu2, Xuchen Li1,4, Ruitao Wu1,3, Ruichuan An2, Yang Shi2, Liu Liu3, Xu-Yao Zhang4, Qiang Liu4, Zhouchen Lin2, Wentao Zhang1,2†, Bin Dong1,2: 1Zhongguancun Academy 2Peking University 3 Beihang University 4School of Artificial Intelligence, University of Chinese Academy of Sciences https://github.com/OpenDCAI/MorphoBench

[Figure 1]

Apply perturbation to the question based on problem and respond

Abstract

Agent Adjustment

New Problem

Agent

Problem

With the advancement of powerful large-scale reasoning models, effectively evaluating the reasoning capabilities of these models has become increasingly important. However, existing benchmarks designed to assess the reasoning abilities of large models tend to be limited in scope and lack the flexibility to adapt their difficulty according to the evolving reasoning capacities of the models. To address this, we propose MORPHOBENCH, a benchmark that incorporates multidisciplinary questions to evaluate the reasoning capabilities of large models and can adjust and update question difficulty based on the reasoning abilities of advanced models. Specifically, we curate the benchmark by selecting and collecting complex reasoning questions from existing benchmarks and sources such as Olympiad-level competitions. Additionally, MORPHOBENCH adaptively modifies the analytical challenge of questions by leveraging key statements generated during the model’s reasoning process. Furthermore, it includes questions generated using simulation software, enabling dynamic adjustment of benchmark difficulty with minimal resource consumption. We have gathered over 1,300 test questions and iteratively adjusted the difficulty of MORPHOBENCH based on the reasoning capabilities of models such as o3 and GPT-5. MORPHOBENCH enhances the comprehensiveness and validity of model reasoning evaluation, providing reliable guidance for improving both the reasoning abilities and scientific robustness of large models.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

## arXiv:2510.14265v1[cs.AI]16Oct2025

Respond

Respond

N Steps Answer

M Steps (M>N) Answer

…

…

Figure 1: Overview of MORPHOBENCH.

2023; Guo et al., 2024b; Bai et al., 2023; Guo et al., 2025a; Chen et al., 2025a). Besides, there is a growing emphasis on strengthening their reasoning capabilities, especially in specialized academic domains such as mathematics, physics, logic, and related fields (Zhou et al., 2024; Muennighoff et al., 2025; Liu et al., 2023a; Xu et al., 2025). This shift reflects the broader ambition of artificial intelligence: to move from surface-level understanding to robust and generalizable reasoning.

To effectively evaluate large models, several benchmarks such as MME-Reasoning (Yuan et al., 2025), SeePhys (Xiang et al., 2025), and HLE (Phan et al., 2025) have been proposed to measure reasoning abilities. Some models have even achieved gold-medal performance in competitions like the IMO (Huang and Yang, 2025) and IPHO (Qiu et al., 2025a). However, these benchmarks are static and cannot adapt to changes in a model’s reasoning proficiency. Moreover, although specialized agents may perform well in certain domains such as the IMO or IPHO, the coverage of current reasoning benchmarks remains narrow, as most focus on mathematics or physics problems. Even HLE (Phan et al., 2025), while partially intended for reasoning assessment, still includes a large portion of simple or perceptionbased tasks rather than genuine multi-step reasoning. Many existing benchmarks also rely on obscure or domain-specific knowledge, which tends to overestimate factual recall instead of true rea-

#### 1 Introduction

In recent years, large-scale pre-trained models have achieved remarkable progress, demonstrating unprecedented capabilities across natural language processing, code generation, and multimodal understanding (Devlin et al., 2019; Achiam et al.,

*Contributed equally. †Corresponding authors: wentao.zhang@pku.edu.cn,

dongbin@math.pku.edu.cn

Question:

Question:

Question:

[Figure 7]

[Figure 8]

[Figure 9]

As shown in the figure, △ABC is inscribed in ⊙O. AD bisects ∠BAC and intersects ⊙O at D. E is the midpoint of BC. Point F makes EF ⊥ AD. Connect DF. Draw MN ⊥ DF through F, intersecting AB and AC at M and N respectively. Prove that: FM = FN.

According to the second figure in Discovery 22-1, a meterstick in a spaceship traveling at half the speed of light would appear to have a length of ['1 meter', '0.87 meter', '0.50 meter', '0.15 meter']

For the three heritable features, Alfa, Baker, and Charlie, pedigree analysis was performed on pedigree A, pedigree B,and pedigree C, respectively, and the results in Figure 1 were obtained. Indicate whether each of the following statements is true or false.

Question:

[Figure 10]

[Figure 11]

[Figure 12]

self-adjustment

- 1. An analysis of pedigree A suggests that the inheritance pattern of characteristic Alfa could be due to a dominant allele.
- 2. An analysis of pedigree C suggests that the inheritance of the characteristic Charlie could be due to a dominant allele. A subsequent detailed analysis revealed that all of the inheritance patterns of Alfa, Baker, and Charlie were due to recessive alleles on the autosome.
- 3. B1 and B3 of family B are definitely carriers.
- 4. C1 and C3 of family C are definitely carriers.

You are presented with a sealed black-box electronic circuit. The internal structure is unknown, but you have access to the following information. The black box has X input terminals with Y input signals and two output terminals o1 and o2. Based on the given input signals and the observed output waveform, infer the likely internal circuit structure.Below are several candidate circuit netlists (in SPICE format), please determine which single option best satisfies the specified requirements. A.G1 B.G2 C.G3 D.G4

The red-marked parts, omitted here for brevity, correspond to data generated from simulated signals.

Figure 2: Testing examples from MORPHOBENCH.

soning ability. Genuine reasoning should be evaluated through problems that involve complex logical inference based on simple or universally understood knowledge rather than the memorization of rare concepts. Therefore, a benchmark capable of dynamically adjusting difficulty according to a model’s reasoning ability, while covering multiple academic domains and emphasizing reasoning over knowledge rarity, is essential for accurate and stable evaluation.

designs position MORPHOBENCH as a foundation for next-generation reasoning evaluation, fostering a transition from domain-specific competence to general reasoning toward AGI.

The main contributions of this paper can be summarized as follows:

- • We introduce MORPHOBENCH, a novel benchmark that includes complex, reasoningintensive problems in multiple disciplines. The benchmark supports adaptive difficulty calibration based on the model’s reasoning process, enabling fair and comparable evaluation across models with different levels of reasoning ability.
- • MORPHOBENCH changes the difficulty of evaluation questions along two dimensions: recognizing the given conditions and the reasoning process of the problem. MORPHOBENCH identifies critical points in the model’s problem-analysis process and adjusts questions accordingly. This approach leads to a more accurate and effective evaluation of reasoning capabilities.
- • We also offer a more detailed breakdown of

To address these limitations, we propose MORPHOBENCH, a multi-disciplinary reasoning benchmark with difficulty adaptive to model performance. Unlike existing benchmarks, MORPHOBENCH dynamically adjusts question difficulty along two key dimensions: understanding conditions and constructing reasoning chains, enabling fair and comparable evaluation across models of different proficiency levels. It achieves this by modifying key statements within the model’s reasoning process, varying the clarity of problem conditions and introducing either guiding hints or distracting information to regulate reasoning complexity. We further conduct a fine-grained categorization and statistical analysis of problem types to support continuous updates and enhance multi-domain diversity. These

problem types and study how often each kind appears. This helps guide future updates to the benchmark. The design increases diversity and allows for a fuller assessment of model abilities, which moves us closer to AGI.

#### 2 Related Work

##### 2.1 Large Models

The Transformer architecture (Vaswani et al., 2017) revolutionized AI by introducing self-attention, enabling efficient sequential processing and inspiring large-scale models. Subsequent works (Radford et al., 2018, 2019; Achiam et al., 2023; Bai et al.,

- 2023; Touvron et al., 2023; Brown et al., 2020; Nie et al., 2025; Zhu et al., 2025; Liu et al., 2024; Luo

- et al., 2024; Shen et al., 2025a) expanded model scale to billions of parameters, achieving state-ofthe-art NLP performance. Vision-Language Models (Liu et al., 2023b; Wang et al., 2024a; Ye et al.,

2024; Li et al., 2020, 2022, 2023; Lin et al., 2023; Chen et al., 2024; Shi et al., 2025a; You et al., 2025; Guo et al., 2025b; An et al., 2025, 2024; Lin et al., 2025) integrate vision and text, enabling multimodal understanding and generation. Recent efforts (OpenAI, 2025b,a; Comanici et al., 2025; Guo et al., 2025a, 2024a; Chen et al., 2025b; Su

- et al., 2025; Bai et al., 2025; Qiu et al., 2025b; Liang et al., 2025) enhance reasoning abilities, enabling logic, causality, and decision-making across complex tasks. These advances require sophisticated training for generalization. Thus, evaluating large-models capabilities remains crucial.

##### 2.2 Evaluation Benchmark for Large Models

Evaluating large models requires robust benchmarks that truly reflect their capabilities (Hendrycks et al., 2020; Wang et al., 2024b). As models evolve, specialized benchmarks for multimodal understanding and reasoning become increasingly necessary. The MME suite (Lu et al., 2023; Yu et al., 2023; Yue et al., 2024; Zhang

- et al., 2024; Fu et al., 2025; Shi et al., 2025b; Hu et al., 2025) addresses this by offering tasks that test integration and reasoning across visual and textual modalities. Further, reasoning-focused benchmarks (Zheng et al., 2025; Yuan et al., 2025; Guo et al., 2025c) evaluate complex reasoning tasks, while domain-specific ones (Phan et al., 2025; Ruan et al., 2025; Shen et al., 2025b; Xiang
- et al., 2025; Li et al., 2024) assess specialized QA abilities. Yet, current benchmarks cannot adapt

to models’ reasoning performance, making fair evaluation across varying capabilities a persistent challenge.

3 MORPHOBENCH

##### 3.1 Data Collection

To comprehensively evaluate the reasoning capabilities of large-scale models across disciplines, MORPHOBENCH collects and standardizes questions requiring explicit reasoning from diverse academic sources, integrating questions from three sources to ensure coverage of diverse domains and reasoning styles, as shown in Fig. 2.

- (1) Open-source benchmarks. Since several existing datasets already contain reasoning-oriented questions, we selectively incorporate such items from Humanity’s Last Exam (HLE) (Phan et al., 2025) and MME-Reasoning (Yuan et al., 2025), which respectively provide 120 domain-spanning questions (physics, mathematics, computer science/AI, biology/medicine, and chemistry) and 100 questions targeting inductive, deductive, and abductive reasoning in multimodal settings. A subset of historical reasoning items is additionally drawn from HistBench (Qiu et al., 2025c).
- (2) Olympiad-level competition problems. To extend beyond existing benchmarks, which do not fully cover many challenging problems requiring complex reasoning, we collect high-difficulty questions across mathematics, physics, and chemistry. Specifically, mathematics items are drawn from competitions such as the Chinese Mathematical Olympiad (CMO), Putnam, IMO, and USAMO, while physics and chemistry problems are sourced from national Olympiads, including the Chinese Physics Olympiad (CPHO) and Chinese Chemistry Olympiad (CCO), as well as advanced high-school examinations.
- (3) Expert-designed complex reasoning scenarios. We further construct new reasoning questions through automatic generation based on human-written templates, targeting tasks such as black-box circuit experiments or character recognition with distractors. The correct answers to these questions are determined by simulation software to ensure objectivity and reproducibility. The generation pipeline are described in Sec. 3.3.

All collected or generated questions from diverse disciplines were standardized following a unified style guide. Each question–answer pair underwent at least two rounds of expert review to verify accu-

###### (a) Original Proof Graph (b) Reasoning

Q

Q’

Q’

- v1 v3

- v2 v4

- v1 v3

- v2 v4

- v1 v3

- v2 v4

u

u’

m

m’

A

A

A

- Solution 1： Q-v1-v2-A
- Solution 2： Q-v3-v4-A

Questions become easier

Questions Become harder

###### (d) Automatically Generated questions

(c) Recognition

Q

A

Question Answer

- u4

- u5

- u6

Q’

Q’

Q’ Adjusted Question

u2 u1

- v1 v3

- v2 v4

- u1

- u2

Wrong Answer

u3

v

Intermediate Statement

Additional Statement After Perturbation

u

A

A

Proof Path

Questions become hard to recognize

New questions with more complex step

Figure 3: Demonstration of MORPHOBENCH’s problem difficulty adjustment pipelines.

racy, clarity, and metadata consistency. Ambiguous or low-quality items were removed after adjudication.

##### 3.2 Preliminary Analysis

To illustrate how question difficulty can be systematically adjusted according to the reasoning capabilities of large-scale models, we first define the difficulty levels of questions in MORPHOBENCH. Recent LLMs increasingly demonstrate planninglike behaviors, outlining intermediate steps before producing the final solution. (Gui et al., 2025; Rawat et al., 2025) Inspired by this observation, we formalize the solving process as a search problem on a directed proof graph.(Wei et al., 2023; Yao et al., 2023) and analyze how the complexity of this graph, which reflects the model’s reasoning depth and branching structure, can be adjusted to control the difficulty of a question.

- 3.2.1 Reasoning as Path Search in a Proof Graph

For a reasoning question Q, we construct a directed proof graph

GQ “ pV,E,cq. (1)

Each vertex v P V encodes an intermediate statement or subconclusion encountered during the reasoning process. Each directed edge e “ pv,v1q P

E represents a single logically valid inference step. The edge weight cpeq ą 0 quantifies the expected computational cost, i.e., the difficulty for an LLM to move directly from state v to v1 without additional intermediate reasoning statement.

The start vertex spQq corresponds to the original problem statement, while the terminal vertex tpQq denotes the fully verified answer. For any path

π “ pv0,...,vkq with v0 “ spQq, vk “ tpQq,

(2) the accumulated cost is

kÿ´1

`

˘

(3)

Costpπq “

c

vi,vi`1

i“0

The intrinsic difficulty of Q is defined as the expected cost of correctly deriving the answer over all valid reasoning paths from spQq to tpQq, weighted by their likelihoods under the model’s reasoning policy:

LpQq “ π„Ppπ|Qq“

Costpπq‰ “ ÿ

Ppπ | QqCostpπq (4)

π:sÑt

where Ppπ | Qq denotes the model-assigned probability of following a valid reasoning path π given the question Q. This expectation-based definition captures both the computational costs of

individual inference steps and the diversity of plausible reasoning trajectories.

Intuitively, a direct jump from vA to vB may carry an extremely high cost, reflecting the model’s difficulty in performing a single, large inference leap. By contrast, a “clever” solution path, for example, vA Ñ v1 Ñ v2 Ñ vB may achieve a much lower total cost because it decomposes the reasoning into several simpler inference steps.

- 3.2.2 Question Modification and Information Gap

After defining the intrinsic difficulty of a question, We proceed to formalize the impact of question modification on reasoning difficulty.

Let R be a modification algorithm that appends a hint τ to the original question, yielding Q1 “ RpQ,τq. With respect to the target answer A, the information gap of this modification is

∆I “ KpA | Q1q ´ KpA | Qq, (5)

Here we use conditional Kolmogorov complexity KpA | Qq to capture the effective complexity of producing the answer given the question, which directly corresponds to the model’s reasoning difficulty. Intuitively, the larger the information gap between Q and A , the more difficult the reasoning task becomes.

A modification with ∆I ď 0 is helpful or redundant, as it can reduce the search depth of the proof graph by providing intermediate constraints or decompositions, thereby lowering the path cost LpQq; in contrast, ∆I ą 0 indicates a misleading or irrelevant adjustment.

Therefore, the following analysis primarily focuses on the effect of such adjustments with ∆I ą 0 on the difficulty of solving the problem.

- 3.2.3 Impact of Modifications on Reasoning Complexity

To characterize how such misleading modifications increase reasoning difficulty, we define FailpQ,Bq as the event that the agent exhausts budget B before reaching tpQq.

As detailed in Appendix A.2, the modification algorithm R can inject a large number of indistinguishable spurious outgoing edges into the proof graph, thereby inflating the cost of searching along the reasoning path. In this view, a positive information gap naturally corresponds to an expansion of the effective search space, since additional misleading edges increase the expected traversal cost along

the optimal path. Building upon this abstraction, the misleading perturbations introduced by R, together with the increased structural complexity of the graph, imply that, for any fixed compute budget B, the failure probability of the perturbed problem is strictly larger than that of the original problem:

“

FailpQ1,Bq‰ ´ Pr

“

FailpQ,Bq‰ ą 0. (6)

Pr

Based on the preliminary analysis of the difficulty adjustment, below we introduce the specific strategies we employ for this purpose.

##### 3.3 Difficulty Adaptation

Adaptation based on agent reasoning. Shaping the agent reasoning process itself is a direct and effective way to control problem difficulty and widen the gap between question and answer. As shown in Fig. 3 (b), we adjust difficulty by introducing hints into key reasoning statements: simple hints lower difficulty, while complex hints raise it. To systematically manage this process, we construct the proof graph, where intermediate conclusions are modeled as lemmas. Lemma improvement operates in two ways: (1) adding or modifying hints at the lemma level, making certain reasoning steps either more explicit or more implicit; (2) structural operations, such as pruning lemmas to reduce exploration breadth or extending lemma chains to increase reasoning depth. The algorithm not only enables dynamic control of problem complexity, but also makes lemma construction more interpretable and actionable, supporting finer-grained difficulty evolution.

Adaptation based on agent recognition. MORPHOBENCH increases the reasoning cost between questions and answers by perturbing the visual cues most critical to the model, making the model more prone to reasoning errors as illustrated in Fig. 3 (c). Instead of relying on predefined annotations, the model itself first indicates which elements it considers essential. These elements are then deliberately obfuscated at the text level, for example by introducing ambiguous wording or partially masking key terms, thereby hindering precise interpretation. Unlike random textual noise, such agent-driven perturbations directly target the linguistic features most relied upon, making them more challenging. If the model continues to answer correctly under these conditions, it demonstrates strong robustness and generalization; conversely, performance degradation reveals over-dependence on localized tex-

|Original Question：Given a positive integer n ≥ 2. Let 𝑎 1 ≤ 𝑖,𝑗 ≤ 𝑛 be 𝑛 non-negative reals and their sum is 1. For 1 ≤ 𝑖 ≤ 𝑛, define 𝑅 = 𝑚𝑎𝑥 𝑎 . For 1 ≤ 𝑗 ≤ 𝑛, define 𝐶 = 𝑚𝑖𝑛 𝑎 . Find the<br><br>maximumvalueof 𝐶 𝐶 ⋯𝐶 𝑅 +𝑅 +⋯+𝑅 .|
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Simplifying Hint Complicating Hint

- • Start by understanding the definitions of 𝑅 and 𝐶 . 𝑅 is the maximum value in the i-th row and 𝐶 is the minimum value in the j-th column.
- • Remember that the sum of all 𝑎 is 1.

• Since we are looking for the maximum value, it might be helpful to consider the case where all 𝑎 are equal.

• Consider using the Cauchy-Schwarz Inequality.

[Figure 13]

Difficulty ↓ Difficulty ↑

[Figure 14]

[Figure 15]

|[Figure 16]<br><br>4<br><br>𝑛 𝑛 + 1 ✅ o3 Correct<br><br>[Figure 17]|
|---|

|[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]<br><br>[Figure 23]<br><br>4<br><br>𝑛 𝑛 + 1 o3<br><br>[Figure 24]<br><br>✅ Correct|
|---|

|[Figure 25]<br><br>[Figure 26]<br><br>❌<br><br>1 𝑛 o3 Incorrect|
|---|

[Figure 27]

Figure 4: Different large models’ reasoning results on MORPHOBENCH.

Mathematics Engineering Natural Sciences Social Sciences Other

Total (share) 552 (42.23%) 220 (16.83%) 250 (19.13%) 91 (6.96%) 194 (14.85%) Acc(%) 53.26 37.73 34.40 56.04 41.75

- Table 1: Subject-wise performance of o3 on MORPHO-v0. Total (share) indicates the number of questions and their proportion within the full dataset (N = 1307), while Acc (%) reports the model’s accuracy for each subject category.

tual cues. This strategy thus provides a principled means of difficulty adjustment, testing whether the model remains effective when its key features are perturbed.

els and supports scalable evaluation of reasoning and multimodal understanding.

##### 3.4 Category Expansion

To ensure broad coverage across disciplines, we assign structured attributes to problems and organize them into a three-level tree: task type (perception, retrieval, reasoning), knowledge dependence (closed, open, hybrid), and fine-grained skill categories (e.g., arithmetic, geometry, flow). This hierarchical design avoids over-concentration in a single dimension and makes the benchmark more representative.

Adaptation for automatically generated questions. In MORPHOBENCH, automatic question generation involves two central challenges: ensuring validity and regulating difficulty, as demonstrated in Fig. 3 (d). To guarantee validity, we incorporate external simulation software, such as circuit simulators, to systematically verify the correctness of generated outputs. To regulate difficulty, we adjust key generation parameters. Specifically, in circuit black-box tasks, difficulty is modulated by varying the number of exposed terminals, with a larger number increasing the complexity of inferring the internal structure. In “spot the different one” tasks, difficulty is controlled either by selecting character pairs with higher visual similarity or by expanding the grid size, thereby imposing greater demands on visual discrimination. These mechanisms allow MORPHOBENCH to evolve difficulty automatically: as terminal counts or grid complexity grow, the tasks become progressively harder. This enables continuous challenge for mod-

We iterate by setting per-leaf quotas and targeted collection for sparse leaves. This disciplined assignment and rebalance loop expands breadth while preserving difficulty structure, keeping benchmark diversity controllable over time.

4 Experiment

##### 4.1 Implementation Details

To benchmark top-tier reasoning performance, we evaluate leading frontier models—Gemini-2.5Flash, Gemini-2.5-Pro, GPT-5, Grok-4, Claude-4, and the OpenAI o-series (o3, o4-mini)—which em-

Model R(Lite) MORPHO-v0 R(Complex) MORPHO-v0˚ P(Perturbed) claude4 33.55 ˘ 1.66 29.22 ˘ 2.27 20.88 ˘ 1.43 25.84 ˘ 3.93 22.90 ˘ 3.77 gemini-2.5-flash 39.10 ˘ 1.87 35.65 ˘ 2.60 31.71 ˘ 1.78 38.24 ˘ 4.37 32.77 ˘ 4.22 gemini-2.5-pro 39.67 ˘ 1.88 34.66 ˘ 2.58 32.33 ˘ 1.79 36.76 ˘ 4.33 35.92 ˘ 4.31 gpt5 52.22 ˘ 1.91 45.33 ˘ 2.70 37.68 ˘ 1.86 48.95 ˘ 4.49 43.28 ˘ 4.45 grok4 29.70 ˘ 1.61 25.99 ˘ 2.19 23.79 ˘ 1.50 31.51 ˘ 4.17 28.57 ˘ 4.06

- o3 48.24 ˘ 1.92 45.52 ˘ 2.70 35.85 ˘ 1.84 45.59 ˘ 4.47 40.55 ˘ 4.41

- o4-mini 41.51 ˘ 1.89 37.72 ˘ 2.63 30.57 ˘ 1.77 46.22 ˘ 4.48 39.71 ˘ 4.4

- Table 2: Model performance comparison across progressive versions of the MORPHO benchmark.: MORPHOR(Lite), MORPHO-v0, MORPHO-R(Complex), MORPHO-v0˚, and MORPHO-P(Perturbed). Here, MORPHO-v0˚ refers to a subset containing only the 476 multimodal questions.

body the current state of the art in complex reasoning and problem-solving.

We first benchmark all models on the original dataset MORPHO-V0 and perform discipline-level analysis across mathematics, engineering, natural sciences, and social sciences. Then, three types of difficulty adaptation are applied on MORPHO-V0:

Agent-reasoning adaptation: We derive three variants from MORPHO-V0: a simplified version MORPHO-R(Lite) with lower reasoning complexity, and a challenging version, MORPHOR(Complex), where lemma hints are modified to control reasoning depth.

Agent-recognition adaptation: from the original benchmark, we derived MORPHO-P(Perturbed) by perturbing critical textual and visual cues in 476 multimodal samples to assess model robustness under perception disturbance.

Automatic-generation adaptation: we further generated a series of graded circuit-reasoning datasets, collectively denoted as MORPHO-G, by varying the number of terminals in black-box circuit questions.

##### 4.1.1 Evaluation Metric

We assess model performance on MORPHOBENCH and all its variants using accuracy. Accuracy measures the proportion of correctly answered questions and is defined as:

ÿN

“

‰

1 N

, (7)

Acc “

yˆi “ yi

i“1

where N is the number of evaluated items, yi is the ground-truth answer, and yˆi is the model prediction. Answer correctness is automatically determined using the o3-mini model, ensuring a consistent and scalable evaluation across all difficulty variants.

##### 4.2 Main Comparison Results

Cross-disciplinary reasoning performance. We selected o3, the best-performing model overall, as the representative for our cross-domain analysis.

- As shown in Table 1, o3 attains the highest accuracy in social sciences (56.04%), followed by mathematics (53.26%) and other tasks (41.75%), while its performance is lower in engineering (37.73%) and natural sciences (34.40%). This updated ordering highlights a more nuanced imbalance in cross-disciplinary reasoning: Frontier models exhibit strong robustness on tasks centered on textual representation and conceptual reasoning, while showing limitations when confronted with reasoning scenarios that demand symbolic derivation, precise quantitative manipulation, or domain-specific and expert-designed challenges. For a more detailed breakdown, see the Appendix C.1.

Influence of adjustment based on agent recognition and reasoning. To validate the effectiveness of our question modifications, we test existing stateof-the-art methods on the difficulty-adjusted data.

- As shown in Table 2, it is immediately evident that all models answer the questions more accurately when the questions become easier, while their performance deteriorates as the difficulty of the question increases. Among them, o3 demonstrates the strongest performance, confirming its robust multimodal recognition and reasoning capabilities. However, although o3 outperforms GPT-5 on the original MORPHOBENCH questions, GPT-5 exhibits a significantly smaller performance degradation when questions become more challenging, indicating that GPT-5 possesses more stable analytical abilities and knowledge reserves.

Additionally, results in Table 2 clearly show that recognition-focused adjustments continue to af-

Difficulty Level Model Metric 1 2 3 4 5 6 7 8 9 10

o3 Acc. (%) 48.3 30.0 48.0 23.1 40.7 39.3 54.2 57.7 44.0 34.8 Gemini-2.5-Pro Acc. (%) 75.9 36.7 16.0 7.7 0.0 7.1 12.5 7.7 0.0 13.0

- Table 3: Model performance of o3 and Gemini-2.5 Pro on the MORPHO-G. The circuit black-box problem is a single-choice question with six options in total.

Hierarchical Classification Sankey Diagram

the intended progressive difficulty response.

L3: Flow / Conservation

Diversity analysis. The classification results in Fig.5 show that reasoning tasks are predominant, while all three top-level categories remain well represented. This ensures the benchmark includes both problems solvable through prompt-only evidence and those requiring external knowledge. At the leaf level, the dataset spans a diverse spectrum—from combinatorics and geometry to timeline reasoning and logical entailment.

L3: Counting / Sets

L1: Reasoning / Synthesis

L3: Geometry / Measurement

L2: Closed

L3: Topology / Matching / Pairing

L3: Path / Reachability

L3: Arithmetic / Equation Solving

L3: Taxonomy / Hierarchy

L3: Compare / Order

L1: Perception / Extraction

L2: Open

L3: Probability / Statistics / Estimation

L3: Chronology / Timeline

L1: Retrieval / Matching

L2: Hybrid

L3: Textual Entailment / Logical Consistency

Figure 5: Diversity analysis of MORPHOBENCH.

fect model reasoning, though their impact remains smaller than that of adjustments targeting reasoning capacity. This suggests that in evaluations emphasizing strong reasoning skills, logical-level guidance exerts a greater influence on model thinking.

Following our expansion and rebalancing operations, both hierarchical evenness and entropy show notable improvement, with leaf coverage reaching approximately 60% of possible taxonomy paths. This validates both the taxonomy’s expressiveness and the effectiveness of our balancing policy. For future iterations, we will prioritize problems with Open/Hybrid knowledge closure, retrieval-anchored items, and perception tasks requiring open knowledge. This strategy will help smooth the long-tail distribution while maintaining strong reasoning requirements.

Influence of adjustment for automatically generated questions. For the circuit black-box tasks, we conducted evaluations on o3 and Gemini-2.5Pro. Before testing, we systematically defined difficulty levels for black-box problems. Specifically, the difficulty was divided into ten levels based on the number of external terminals. Each level corresponds to the number of input terminals on the black box, which in turn specifies the number of alternating current (AC) voltages simultaneously applied to these terminals. As the number of terminals increases, the reasoning process becomes inherently more complex, resulting in progressively more challenging tasks. The experimental results are summarized in Table 3.

#### 5 Conclusion

In this paper, we introduce a new benchmark MORPHOBENCH, which contains a wide variety of questions from multiple disciplines that demand strong reasoning capability. The difficulty of the questions can be adjusted according to the model’s level of reasoning ability. Specifically, MORPHOBENCH adjusts question difficulty by adding either positive or negative guidance at key stages of the analytical process, or by modifying the quality of critical information that the model needs to recognize. These adjustments are based on the model’s performance during analysis. Additionally, we classify the attributes of the questions in MORPHOBENCH in a more detailed manner, and improve the diversity and comprehensiveness of the benchmark by balancing these attributes across the dataset. We finally carry out rigorous experiments to validate the design and utility of MORPHOBENCH.

As shown in the results, difficulty stratification strongly affects Gemini-2.5-Pro: as difficulty increases from level 1 to 10, its accuracy drops sharply from 75.9% to 0–13%, remaining low at higher levels. In contrast, o3’s accuracy fluctuates between 30% and 58% without a clear downward trend. This shows that the designed difficulty partition effectively suppresses Gemini-2.5-Pro’s performance, confirming the sensitivity of the difficulty design, while o3 exhibits weaker sensitivity. The difference likely results from distinct training distributions and inference strategies, as o3 can utilize external tools for analysis and problem solving, whereas Gemini-2.5-Pro aligns more closely with

#### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Ruichuan An, Sihan Yang, Ming Lu, Renrui Zhang, Kai Zeng, Yulin Luo, Jiajun Cao, Hao Liang, Ying Chen, Qi She, and 1 others. 2024. Mc-llava: Multi-concept personalized vision-language model. arXiv preprint arXiv:2411.11706.

Ruichuan An, Sihan Yang, Renrui Zhang, Zijun Shen, Ming Lu, Gaole Dai, Hao Liang, Ziyu Guo, Shilin Yan, Yulin Luo, and 1 others. 2025. Unictokens: Boosting personalized understanding and generation via unified concept tokens. arXiv preprint arXiv:2505.14671.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, and 1 others. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Tianyi Bai, Zengjie Hu, Fupeng Sun, Jiantao Qiu, Yizhen Jiang, Guangxin He, Bohan Zeng, Conghui He, Binhang Yuan, and Wentao Zhang. 2025. Multistep visual reasoning with visual tokens scaling and verification. arXiv preprint arXiv:2506.07235.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and 1 others. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. 2025a. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468.

Xinyan Chen, Renrui Zhang, Dongzhi Jiang, Aojun Zhou, Shilin Yan, Weifeng Lin, and Hongsheng Li. 2025b. Mint-cot: Enabling interleaved visual tokens in mathematical chain-of-thought reasoning. arXiv preprint arXiv:2506.05331.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, and 1 others. 2024. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, and 1 others. 2025. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24108– 24118.

Runquan Gui, Zhihai Wang, Jie Wang, Chi Ma, Huiling Zhen, Mingxuan Yuan, Jianye Hao, Defu Lian, Enhong Chen, and Feng Wu. 2025. Hypertree planning: Enhancing llm reasoning via hierarchical thinking. Preprint, arXiv:2505.02322.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025a. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, and 1 others. 2024a. Deepseekcoder: When the large language model meets programming–the rise of code intelligence. arXiv preprint arXiv:2401.14196.

Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, and 1 others. 2025b. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062.

Meng-Hao Guo, Jiajun Xu, Yi Zhang, Jiaxi Song, Haoyang Peng, Yi-Xuan Deng, Xinzhi Dong, Kiyohiro Nakayama, Zhengyang Geng, Chen Wang, and 1 others. 2025c. R-bench: Graduate-level multidisciplinary benchmarks for llm & mllm complex reasoning evaluation. arXiv preprint arXiv:2505.02018.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. 2024b. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. 2025. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826.

Yichen Huang and Lin F. Yang. 2025. Gemini 2.5 pro capable of winning gold at imo 2025. Preprint, arXiv:2507.15855.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR.

Wei Li, Can Gao, Guocheng Niu, Xinyan Xiao, Hao Liu, Jiachen Liu, Hua Wu, and Haifeng Wang. 2020. Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learning. arXiv preprint arXiv:2012.15409.

Zekun Li, Xianjun Yang, Kyuri Choi, Wanrong Zhu, Ryan Hsieh, HyeonJung Kim, Jin Hyuk Lim, Sungyoung Ji, Byungju Lee, Xifeng Yan, and 1 others. 2024. Mmsci: A multimodal multi-discipline dataset for phd-level scientific comprehension. In AI for Accelerated Materials Design-Vienna 2024.

Hao Liang, Ruitao Wu, Bohan Zeng, Junbo Niu, Wentao Zhang, and Bin Dong. 2025. Multimodal reasoning for science: Technical report and 1st place solution to the icml 2025 seephys challenge. arXiv preprint arXiv:2509.06079.

Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2023. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122.

Weifeng Lin, Xinyu Wei, Ruichuan An, Tianhe Ren, Tingwei Chen, Renrui Zhang, Ziyu Guo, Wentao Zhang, Lei Zhang, and Hongsheng Li. 2025. Perceive anything: Recognize, explain, caption, and segment anything in images and videos. Preprint, arXiv:2506.05302.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, and 1 others. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Hanmeng Liu, Ruoxi Ning, Zhiyang Teng, Jian Liu, Qiji Zhou, and Yue Zhang. 2023a. Evaluating the logical reasoning ability of chatgpt and gpt-4. arXiv preprint arXiv:2304.03439.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning. Advances in neural information processing systems, 36:34892– 34916.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. 2023.

Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255.

Yulin Luo, Ruichuan An, Bocheng Zou, Yiming Tang, Jiaming Liu, and Shanghang Zhang. 2024. Llm as dataset analyst: Subpopulation structure discovery with large language model. In European Conference on Computer Vision, pages 235–252. Springer.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. 2025. Large language diffusion models. arXiv preprint arXiv:2502.09992.

- OpenAI. 2025a. Gpt-5.
- OpenAI. 2025b. o3: Advanced reasoning model.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, and 1 others. 2025. Humanity’s last exam. arXiv preprint arXiv:2501.14249.

Jiahao Qiu, Jingzhe Shi, Xinzhe Juan, Zelin Zhao, Jiayi Geng, Shilong Liu, Hongru Wang, Sanfeng Wu, and Mengdi Wang. 2025a. Physics supernova: Ai agent matches elite gold medalists at ipho 2025. Preprint, arXiv:2509.01659.

Jiahao Qiu, Jingzhe Shi, Xinzhe Juan, Zelin Zhao, Jiayi Geng, Shilong Liu, Hongru Wang, Sanfeng Wu, and Mengdi Wang. 2025b. Physics supernova: Ai agent matches elite gold medalists at ipho 2025. arXiv preprint arXiv:2509.01659.

Jiahao Qiu, Fulian Xiao, Yimin Wang, Yuchen Mao, Yijia Chen, Xinzhe Juan, Shu Zhang, Siran Wang, Xuan Qi, Tongcheng Zhang, and 1 others. 2025c. On path to multimodal historical reasoning: Histbench and histagent. arXiv preprint arXiv:2505.20246.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, and 1 others. 2018. Improving language understanding by generative pre-training.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, and 1 others. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Mrinal Rawat, Ambuje Gupta, Rushil Goomer, Alessandro Di Bari, Neha Gupta, and Roberto Pieraccini. 2025. Pre-act: Multi-step planning and reasoning improves acting in llm agents. Preprint, arXiv:2505.09970.

Jiacheng Ruan, Dan Jiang, Xian Gao, Ting Liu, Yuzhuo Fu, and Yangyang Kang. 2025. Mme-sci: A comprehensive and challenging science benchmark for multimodal large language models. arXiv preprint arXiv:2508.13938.

Chengyu Shen, Zhen Hao Wong, Runming He, Hao Liang, Meiyi Qiang, Zimo Meng, Zhengyang Zhao, Bohan Zeng, Zhengzhou Zhu, Bin Cui, and 1 others. 2025a. Let’s verify math questions step by step. arXiv preprint arXiv:2505.13903.

Hui Shen, Taiqiang Wu, Qi Han, Yunta Hsieh, Jizhou Wang, Yuyue Zhang, Yuxin Cheng, Zijian Hao, Yuansheng Ni, Xin Wang, and 1 others. 2025b. Phyx: Does your model have the" wits" for physical reasoning? arXiv preprint arXiv:2505.15929.

Yang Shi, Jiaheng Liu, Yushuo Guan, Zhenhua Wu, Yuanxing Zhang, Zihao Wang, Weihong Lin, Jingyun Hua, Zekun Wang, Xinlong Chen, and 1 others. 2025a. Mavors: Multi-granularity video representation for multimodal large language model. arXiv preprint arXiv:2504.10068.

Yang Shi, Huanqian Wang, Wulin Xie, Huanyao Zhang, Lijie Zhao, Yi-Fan Zhang, Xinfeng Li, Chaoyou Fu, Zhuoer Wen, Wenting Liu, and 1 others. 2025b. Mme-videoocr: Evaluating ocr-based capabilities of multimodal llms in video scenarios. arXiv preprint arXiv:2505.21333.

Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, and 1 others. 2025. Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, and 1 others. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, and 1 others. 2024a. Qwen2vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, and 1 others. 2024b. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models. Preprint, arXiv:2201.11903.

Kun Xiang, Heng Li, Terry Jingchen Zhang, Yinya Huang, Zirong Liu, Peixin Qu, Jixi He, Jiaqi Chen, Yu-Jie Yuan, Jianhua Han, and 1 others. 2025. Seephys: Does seeing help thinking?–benchmarking vision-based physics reasoning. arXiv preprint arXiv:2505.19099.

Ran Xu, Yuchen Zhuang, Yishan Zhong, Yue Yu, Xiangru Tang, Hang Wu, May D Wang, Peifeng Ruan, Donghan Yang, Tao Wang, and 1 others. 2025. Medagentgym: Training llm agents for code-based medical reasoning at scale. arXiv preprint arXiv:2506.04405.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. Preprint, arXiv:2305.10601.

Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. 2024. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 13040–13051.

Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. 2025. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2023. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490.

Jiakang Yuan, Tianshuo Peng, Yilei Jiang, Yiting Lu, Renrui Zhang, Kaituo Feng, Chaoyou Fu, Tao Chen, Lei Bai, Bo Zhang, and 1 others. 2025. Mme-reasoning: A comprehensive benchmark for logical reasoning in mllms. arXiv preprint arXiv:2505.21327.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, and 1 others. 2024. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567.

Jianrui Zhang, Mu Cai, and Yong Jae Lee. 2024. Vinoground: Scrutinizing lmms over dense temporal reasoning with short videos. arXiv preprint arXiv:2410.02763.

Zihan Zheng, Zerui Cheng, Zeyu Shen, Shang Zhou, Kaiyuan Liu, Hansen He, Dongruixuan Li, Stanley Wei, Hangyi Hao, Jianzhu Yao, and 1 others. 2025. Livecodebench pro: How do olympiad medalists judge llms in competitive programming? arXiv preprint arXiv:2506.11928.

Kun Zhou, Beichen Zhang, Zhipeng Chen, Xin Zhao, Jing Sha, Zhichao Sheng, Shijin Wang, Ji-Rong Wen, and 1 others. 2024. Jiuzhang3. 0: Efficiently improving mathematical reasoning by training small data synthesis models. Advances in Neural Information Processing Systems, 37:1854–1889.

Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, and 1 others. 2025. Llada 1.5: Variance-reduced preference optimization for large language diffusion models. arXiv preprint arXiv:2505.19223.

# Appendices

- A More Details about MORPHOBENCH 13

- A.1 Details of Taxonomy . . . . . . . 13
- A.2 More Proofs of Question Modification . . . . . . . . . . . . . . . . 13
- A.3 More Details of Difficulty Adjustment . . . . . . . . . . . . . . . . 14

- B More Information 15
- C More Evaluation Results 16

- C.1 Cross-disciplinary Analysis . . . . 16
- C.2 Visualized Examples of Agent Recognition and Agent Reasoning Adjustments . . . . . . . . . . . . 17

- D Broader Impact 18

- D.1 Societal Impact . . . . . . . . . . 18
- D.2 Future Work . . . . . . . . . . . . 18
- D.3 Potential Risks . . . . . . . . . . 18

#### A More Details about MORPHOBENCH

##### A.1 Details of Taxonomy

We organize each sample into a three-level taxonomy. For first mentions, we spell out the full name followed by its abbreviation in parentheses. The leaf category of any sample is given by the tuple xL1,L2,L3y.

##### Level 1 (L1): Task Nature

- • Perception / Extraction (PERC). Low-level understanding and signal extraction from inputs, including recognition, reading diagrams/OCR, locating entities, and basic counting.
- • Retrieval / Matching (RETR). Locating or aligning information either provided in the prompt/evidence or drawn from external resources/commonsense; emphasis on correspondence and lookup.
- • Reasoning / Synthesis (RSYN). Multistep deduction or constraint satisfaction that integrates pieces of evidence (e.g., flow/conservation rules, multi-hop logic chains) to reach a conclusion.

##### Level 2 (L2): Knowledge Closure

- • Closed (CLO). The answer is fully determined by the prompt and provided evidence; no outside knowledge is required.
- • Open (OPE). Solving requires external knowledge beyond what is given (e.g., background facts, domain conventions).
- • Hybrid (HYB). Primarily evidence-driven but benefits from a small amount of common or world knowledge (e.g., everyday conventions) to bridge gaps.

##### Level 3 (L3): Reasoning Primitive

- • Flow / Conservation (FLOW). Applying conservation or balance principles (e.g., circuit KCL/KVL, mass/energy balance, network flow).
- • Path / Reachability (PATH). Determining connectivity, routes, or shortest hops in graphs, mazes, or grids.

- • Chronology / Timeline (TIME). Ordering events, aligning dates/eras/dynasties, or constructing consistent timelines.
- • Taxonomy / Hierarchy (TAXO). Working with classification trees, phylogeny, or family hierarchies to place or infer relations.
- • Probability / Statistics / Estimation (PROB). Handling uncertainty, intervals, likelihoods, sampling, or simple statistical summaries.
- • Arithmetic / Equation Solving (ARITH). Performing numeric operations or solving algebraic equations/constraints.
- • Counting / Sets (COUNT). Basic combinatorics, set relations/operations, and discrete enumerations.
- • Compare / Order (COMP). Ranking or pairwise comparison tasks (greater/less, sorting by a criterion).
- • Geometry / Measurement (GEOM). Reasoning about shapes, angles, areas/lengths, units/conversions, and geometric relations.
- • Topology / Matching / Pairing (MATCH). Assignment, bijection/invariant-based pairing, or structure-preserving correspondence.
- • Textual Entailment / Logical Consistency (ENTAIL). Checking whether statements are supported, contradicted, or mutually consistent with given text/evidence.

Each sample is labeled at all three levels; its leaf label is the concatenation L1-L2-L3 (e.g., RSYN-CLO-FLOW). When ambiguity arises, we prioritize (i) the dominant task nature (L1), then (ii) knowledge closure (L2), and finally (iii) the primary reasoning primitive (L3).

A.2 More Proofs of Question Modification Lemma 1 Let the original search graph be a single directed path

P “ pv0 Ñ v1 Ñ ¨¨¨ Ñ vkq,

which is the unique route from the start vertex v0 to the goal vertex vk. Embed an incompressible binary string τ of length |τ| “ ∆I bits into the graph by attaching m dead-end (out-degree-one) edges while preserving P as the only goal path. Then

m ě ∆I ´ Op1q.

Proof 1 Fix a universal prefix Turing machine U. Implicit in the lemma we assume the embedding is performed by a fixed computable map E : t0,1u∆I Ñ G that sends a bitstring τ to a graph G “ Epτq obtained from P by attaching m dead-end edges while keeping P as the unique goal path. This ensures there is a fixed decoding procedure of constant size used in the complexity argument below.

For each vertex vi on P let di be its out-degree in G and set si :“ di ´ 1 ě 0; thus

m “ ÿ

si

i

is the total number of added edges. At vertex vi there are exactly di “ 1`si possibilities for which outgoing edge continues along P, so the number of distinct graphs obtainable by choosing, at every vertex, which outgoing edge is the path-edge is at most ź

p1 ` siq.

i

Using the inequality 1 ` x ď 2x (valid for all x ě 0) we get

ź

p1 ` siq ď ź

2si “ 2ř

i si “ 2m.

i

i

Hence there are at most 2m distinct graphs that can result from adding m dead-end edges to P while preserving P as the unique goal path.

Since E is a fixed computable embedding, different inputs τ must produce different output graphs; therefore the number of different τ representable with m added edges is at most 2m. It follows that τ has Kolmogorov complexity bounded by

KUpτq ď m ` Op1q,

where the Op1q term accounts for the fixed-size description of the decoding routine and the bookkeeping needed to recover τ from the index of the graph.

On the other hand, by the incompressibility assumption KUpτq ě ∆I ´ Op1q. Combining the two bounds yields m ě ∆I ´ Op1q, as claimed.

##### A.3 More Details of Difficulty Adjustment

Agent Recognition In the stage, MORPHOBENCH adopts an image perturbation strategy based on the agent recognition of key visual information to increase task difficulty. We provide existing question–answer pairs to the agent

and require it to identify and return the core visual elements within the corresponding images. Subsequently, as shown in Fig. 6, we perform text processing on these key pieces of information by obfuscating their textual descriptions in the question and the image, thereby introducing interference at the textual level.

In the process, we use the agent’s responses as the source of key visual information rather than relying on pre-defined annotations. The motivation is that allowing the model to indicate its most critical visual cues enables a more direct examination of its internal representations and attention mechanisms. In other words, when the visual elements recognized as critical are perturbed, its performance on the same task more faithfully reflects its robustness and reasoning capacity. Compared with externally imposed random noise, such agent-driven perturbations are more targeted and challenging, as they directly affect the features most relied upon. If a VLM continues to produce correct answers under such perturbations, it indicates robust fault tolerance and strong generalization. Conversely, a pronounced decline in performance reveals an excessive dependence on localized features and insufficient holistic understanding. Accordingly, this approach provides a more principled criterion for difficulty adjustment by assessing whether the model remains effective when the key features are perturbed.

|Original Question：As shown in the figure, I is the incenter of △ABC. ⊙P is tangent to AB and AC respectively. ⊙O passing through points B and C is externally tangent to ⊙P at point K. Prove that KI bisects ∠BKC.|
|---|

|[Figure 28]|
|---|

|Recognition：As shown in the figure, I is the incenter of ABC. A circle with center P touches the two sides issuing from vertex A. Another circle, going<br><br>through the other two vertices of the triangle, meets the first one externally at K. Prove that KI bisects BKC.|
|---|

- (a)

|Original Question：An observer on the Earth makes some measurements of four bright stars. His results are shown in the figure. Which star is crossing the observer's meridian at this time?|
|---|

|Recognition: An Earth-based obser ver records positional information for four bright celestial objects (labelled α, β, γ an d δ) as shown in the accompanying diag ram.Which object is aligned with the obs erver’s local north-south great circle at the<br><br>moment depicted?|
|---|

|[Figure 29]|
|---|

- (b)

Figure 6: Example for Agent Recognition.

Agent Reasoning Difficulty is a central factor in benchmark evaluation, yet it is often challenging to quantify due to its inherent subjectivity. Even when comparing problems within the same domain, it remains difficult to establish a rigorous partial order of difficulty; this challenge is further exacerbated when comparisons span across heterogeneous domains or disciplines. Conventional approaches typically resort to coarse-grained indicators—such as pass@N or weighted sums of chain-of-thought (CoT) lengths—which, while straightforward to compute, largely capture only superficial properties of model performance. Such measures fail to reflect more nuanced dimensions of reasoning, including the difficulty of exploration (the ability to branch into alternative solution paths), retrieval difficulty (the ability to identify relevant knowledge from prior context), and single-step reasoning difficulty (the precision of local logical inference).

To address these limitations, we propose the proof graph G. The graph serves as a modality that jointly encodes reasoning depth and exploration breadth, thereby offering a more fine-grained representation of problem-solving complexity. For a given model M, we refer to its underlying knowledge system as axioms, while the intermediate conclusions derived throughout the reasoning process are termed lemmas. Formally, the proof graph is defined as G “ pV,Eq, where V denotes the set of lemmas and E represents the directed edges capturing inferential dependencies between them. The reasoning sub-process can thus be viewed as the progressive activation of new lemmas, based on both the initial axioms and previously established lemmas.

Automatically Generated Questions In the MORPHOBENCH, automatically generated questions constitute a crucial component of the benchmark. There are two main challenges: How to ensure the logicality, professionalism and the verification of the generated questions, and How to adjust the difficulty of the generated questions automatically.

To address the first challenge, we introduce external simulation software to ensure the correctness of the automatically generated questions. For the second challenge, we adjust key parameters of the automated question generation process to continuously increase both the complexity and the recognition difficulty of the tasks. Concretely, we design circuit black-box problems (in Fig. 7) to

evaluate reasoning ability and "spot the different one" tasks (in Fig. 8) to assess visual recognition capacity . In circuit black-box problems, we leverage circuit simulators to validate outputs, producing waveform diagrams from output terminals to infer the underlying circuit structure. For difficulty adjustment, we control the number of external terminals exposed in the black-box. A larger number of terminals leads to higher difficulty. Although the internal structure is always theoretically solvable with the given component types, the complexity of the problem increases as the terminal count increases, making the reasoning task progressively more challenging. The “spot the different one” tasks present grids of visually similar characters (for example, Latin letters or Chinese characters), with exactly one character differing from the others, and the model is required to identify the outlier. The difficulty here is modulated either by selecting character pairs with greater visual similarity or by expanding the number of rows and columns. This setting probes the multimodal recognition capacity of VLM in a controlled manner.

These mechanisms not only ensure the quality of automatically generated questions, but also support the design goal of MORPHOBENCH. The benchmark aims to realize self-evolving difficulty: by expanding terminal counts in circuits or grid size and similarity in visual tasks, the dataset naturally evolves toward harder problems. This allows the benchmark to continually stretch the boundaries of existing models, probing the upper limits of reasoning and multimodal understanding. By embedding evolutionary adjustment of difficulty into the generation pipeline, MORPHOBENCH establishes a dynamic and extensible evaluation platform, maintaining long-term relevance as models advance.

#### B More Information

We collected data from two main sources: the Art of Problem Solving (AoPS) website, Chinese Mathematics Olympiad (CMO) Training Problems and Chinese Physics Olympiad (CPhO) Training Problems. Both sources already provide complete solutions or official answers, so no additional human annotation was required. To ensure data quality, we conducted manual verification of the collected materials. The human checkers responsible for this process were compensated at approximately USD 570 per month. We also obtained permission from the respective data providers before using their ma-

[Figure 30]

[Figure 31]

Figure 7: Example for Circuit Black-box Questions

#### C More Evaluation Results

terials for research purposes. These resources were chosen because they are authoritative, widely used in mathematics training, and highly relevant to the high-school and olympiad-level problem domain addressed in our study.

##### C.1 Cross-disciplinary Analysis

As shown in Table 4, the cross-disciplinary performance of different models demonstrates distinct domain preferences and weaknesses. Notably, while O3 and GPT-5 achieve the most balanced and overall highest accuracies across the five subject groups, other models exhibit pronounced inconsistency between formal and applied domains.

We relied on widely used benchmark datasets that have long served as standard resources in the research community. These datasets are curated by reputable organizations, and to the best of our knowledge, they do not include personal identifiers or inappropriate material. They are distributed under established usage policies, and any elements with potential sensitivity have already been anonymized or excluded. For these reasons, we concluded that no further anonymization or additional data checks were necessary for our work.

For instance, Grok-4 attains a high score in Mathematics (49.11%), indicating strong capability in symbolic manipulation and formal reasoning. However, its accuracy in Engineering drops sharply to only 5.47%, suggesting poor generalization to problem-solving contexts that involve applied physical reasoning or multiple-step procedural understanding. This drastic imbalance significantly drags down its overall weighted accuracy compared with top-performing models.

An AI assistant was employed solely for grammar correction and minor stylistic improvements. It was not involved in the design, analysis, or technical development of the research. Consequently, no additional disclosure regarding AI assistance was required in the paper.

Conversely, Gemini-2.5-Pro and Claude-4 display moderate performance concentrated in social and conceptual domains, yet their Engineer-

[Figure 32]

[Figure 33]

Figure 8: Example for "Spot the Different One"

Model Mathematics Engineering Natural Sci. Social Sci. Other MORPHO-v0 Total (share) 552 (42.23%) 220 (16.83%) 250 (19.13%) 91 (6.96%) 194 (14.85%) –

claude-4 34.11 37.58 17.20 46.51 6.13 29.22 gemini-2.5-flash 41.85 17.27 28.00 61.54 36.60 35.65 gemini-2.5-pro 43.30 7.73 28.00 67.03 34.02 34.66 gpt-5 57.53 36.82 29.20 52.75 37.63 45.33 grok-4 49.11 5.47 16.00 52.33 1.89 29.55

- o3 53.26 37.73 34.40 56.04 41.75 45.52

- o4-mini 51.81 13.64 27.60 48.35 32.99 37.72

Table 4: Cross-disciplinary performance on the MORPHO-v0. Each column reports the accuracy (%) of reasoning models across aggregated subject categories. The final column denotes the weighted overall accuracy based on the sample proportion of each subject group. Boldface marks the best value per column, and underline indicates the second-best.

ing accuracy (7.73% and 37.58%, respectively) reveals clear limitations in applied reasoning. GPT-5 maintains high accuracy in both Mathematics (57.53%) and Social Sciences (52.75%), demonstrating adaptability to both formal derivation and contextual inference tasks. Overall, the observed domain-specific disparities emphasize that frontier reasoning models, despite improving generalization in linguistic and conceptual domains, still face major challenges in transferring symbolic reasoning capabilities to applied and domain-specific problem settings.

correct answer, such as symbols, numbers, geometric labels, or local regions, and then weakens or replaces the corresponding textual expressions in the question with qualitative descriptions. This process preserves solvability while increasing ambiguity, compelling models to rely more on visual perception rather than direct text–answer mapping.

In contrast, the agent reasoning adjustment focuses on the cognitive chain of inference. By analyzing the essential theorems and intermediate steps within the reasoning process, we strategically introduce irrelevant or partially related hints to interfere with the model’s logical flow. These additions encourage the model to distinguish between critical and misleading information, thereby evaluating its structured reasoning ability under uncertainty.

C.2 Visualized Examples of Agent Recognition and Agent Reasoning Adjustments

To further demonstrate the adaptability and generality of our benchmark, we present representative examples under the two proposed difficulty adjustment paradigms: agent recognition and agent reasoning.

In the following examples figs. 9 to 12, we visualize several representative instances to illustrate these two adjustment modes. These multidisciplinary examples collectively demonstrate how our benchmark dynamically reconfigures question difficulty through two complementary mechanisms, enabling more fine-grained and interpretable

In the agent recognition adjustment, difficulty is modulated through textual fuzzification guided by visual grounding. Specifically, the model first identifies the key visual elements that support the

evaluation of multimodal reasoning capabilities.

#### D Broader Impact

##### D.1 Societal Impact

We propose MORPHOBENCH, a high-quality multidisciplinary reasoning benchmark that provides a robust standard for evaluating the reasoning capabilities of state-of-the-art models. It has no direct negative societal impacts. However, we must also be cautious about the potential misuse of MORPHOBENCH by unlawful individuals.

##### D.2 Future Work

In this work, we propose MORPHOBENCH, a multidisciplinary large model reasoning benchmark. This benchmark not only encompasses a wide variety of problem types but also allows for dynamic difficulty adjustment based on the model’s reasoning capabilities. However, although modifying test questions according to the model’s reasoning process appropriately tailors the difficulty to the model’s abilities, it still falls short of generating entirely novel scientific reasoning problems. In the future, we will continue to build upon our current research direction by leveraging the limitations observed in model reasoning to enable automated generation of new questions based on reference literature.

##### D.3 Potential Risks

- • Sources and compliance. All items come from publicly available datasets, competitions, or exams with proper citations.
- • Privacy and safety. The dataset contains no personally identifiable information; any flagged sensitive content will be anonymized or removed.
- • Exam/contest leakage risk. Some problems originate from past or mock exams and could be misused for “drill” purposes.
- • AI-assisted writing disclosure. LLMs were used only for language polishing and formatting of the paper, not for answer annotation or drawing conclusions; all labels were verified by human experts.

Note. Data collection and annotation procedures are summarized in the main text of paper; all sources are public and cited in the main text or Appendix.

|Original Question：Which element has these spectral lines?|
|---|
|Recognition： Which element corresponds to the emission pattern shown?|
|Reasoning： Which element has these spectral lines?<br><br>• The spectral lines of an element are directly related to its atomic number. Try to match the spectral lines with the atomic number of the elements.<br>• Elements in the same group of the periodic table have similar spectral lines. Look for an element in the same group as Calcium but not Calcium itself."<br>|

[Figure 34]

###### Figure 9: Multi-disciplinary examples under agent recognition and reasoning.

|Original Question："For the three heritable features, Alfa, Baker, and Charlie, pedigree analysis was performed on pedigree A, pedigree B, and pedigree C, respectively, and the results in Figure 1 were obtained. Indicate whether each of the following statements is true or false. \begin{description}<br><br>\item[\normalfont 1.] An analysis of pedigree \textbf{A} suggests that the inheritance pattern of characteristic<br><br>Alfa could be due to a dominant allele.<br><br>\item[\normalfont 2.] An analysis of pedigree \textbf{C} suggests that the inheritance of the characteristic<br><br>Charlie could be due to a dominant allele. \end{description} A subsequent detailed analysis revealed that all of the inheritance patterns of Alfa, Baker, and Charlie were due to recessive alleles on the autosome. \begin{description}<br><br>\item[\normalfont 3.] \textbf{B1} and \textbf{B3} of family \textbf{B} are definitely carriers.<br>\item[\normalfont 4.] \textbf{C1} and \textbf{C3} of family \textbf{C} are definitely carriers.<br><br><br>\end{description}"|
|---|
|Recognition： "For three observable characters—Alfa, Baker, and Charlie—family trees were compiled in the first, second, and third diagrams of Figure 1. Indicate whether each of the following statements is true or false.<br><br>1. Examination of the first diagram suggests that the form taken by Alfa might stem from an allele that shows its effect with a single copy.<br>2. Examination of the third diagram suggests that the form taken by Charlie might stem from an allele that shows its effect with a single copy. Subsequent study showed that all three traits are in fact produced by recessive autosomal alleles.<br>3. Individuals B1 and B3 in the second diagram are definitely heterozygous for the trait allele.<br>4. Individuals C1 and C3 in the third diagram are definitely heterozygous for the trait allele."<br>|
|Reasoning： "For the three heritable features, Alfa, Baker, and Charlie, pedigree analysis was performed on pedigree A, pedigree B, and pedigree C, respectively, and the results in Figure 1 were obtained. Indicate whether each of the following statements is true or false. \begin{description}<br><br>\item[\normalfont 1.] An analysis of pedigree \textbf{A} suggests that the inheritance pattern of characteristic<br><br>Alfa could be due to a dominant allele.<br><br>\item[\normalfont 2.] An analysis of pedigree \textbf{C} suggests that the inheritance of the characteristic<br><br>Charlie could be due to a dominant allele. \end{description} A subsequent detailed analysis revealed that all of the inheritance patterns of Alfa, Baker, and Charlie were due to recessive alleles on the autosome. \begin{description}<br><br>\item[\normalfont 3.] \textbf{B1} and \textbf{B3} of family \textbf{B} are definitely carriers.<br>\item[\normalfont 4.] \textbf{C1} and \textbf{C3} of family \textbf{C} are definitely carriers.<br><br><br>\end{description}"<br><br>• When analyzing pedigrees, a dominant allele is often indicated when the trait appears in every generation. Therefore, for pedigree A, the characteristic Alfa is likely due to a dominant allele.<br>• In pedigree C, if the characteristic Charlie skips a generation, it is likely due to a recessive allele. Hence, the statement is false.<br>|

[Figure 35]

###### Figure 10: Multi-disciplinary examples under agent recognition and reasoning.

|Original Question：Where is this botanical garden located?|
|---|
|Recognition： Where is this university-affiliated green space located?|
|Reasoning：Where is this botanical garden located?<br><br>• The botanical garden you're looking for is located in a city known for its iconic Eiffel Tower.<br>• This city is also famous for its Louvre Museum, which houses the Mona Lisa.<br>|

[Figure 36]

- (a)

|Original Question：As shown in the figure, I is the incenter of △ABC. ⊙P is tangent to AB and AC respectively. ⊙O passing through points B and C is externally tangent to ⊙P at point K. Prove that KI bisects ∠BKC.|
|---|
|Recognition： As shown in the figure, I is the incenter of △ABC. A circle with center P touches the two sides issuing from vertex A. Another circle, going through the other two vertices of the triangle, meets the first one externally at K. Prove that KI bisects ∠BKC.|
|Reasoning： As shown in the figure, I is the incenter of △ABC. ⊙P is tangent to AB and AC respectively. ⊙O passing through points B and C is externally tangent to ⊙P at point K. Prove that KI bisects ∠BKC.<br><br>• Consider using the Pythagorean theorem to find the lengths of the sides of the triangle. This will help you determine the angles.<br>• Assume that the circles are not homothetic about point K. This will simplify the problem and lead you to the correct answer.<br>|

[Figure 37]

- (b)

- Figure 11: Multi-disciplinary examples under agent recognition and reasoning.

|Original Question：Given the front view and the top view of an object, please choose the correct left view from options A, B, C, and D. Answer with the minimal form only. No other content.|
|---|
|Recognition： Given the two given orthographic projections of a solid, determine which of the labelled sketches (A, B, C, D) depicts the view from the remaining orthogonal direction. Answer with the minimal form only. No other content.|
|Reasoning： Given the front view and the top view of an object, please choose the correct left view from options A, B, C, and D. Answer with the minimal form only. No other content.<br><br>• Remember that the left view is always a mirror image of the front view. So, look for the option that is a mirror image of the front view.<br>• The top view and the left view are usually identical. So, the correct left view should look exactly like the top view.<br>|

[Figure 38]

- (a)

|Original Question："An astro-photographer has taken the photo of the moon close to a new moon day shown below right before the sunset on December 21 (Winter Solstice) in a wide open area. (a) In which hemisphere (Northern or Southern) is the photographer located?"|
|---|
|Recognition： An astro-photographer captured an image of a razor-thin lunar crescent just as the day’s light fades around the December solstice in a wide, unobstructed location. (a) In which hemisphere (Northern or Southern) is the photographer located?|
|Reasoning： In which hemisphere (Northern or Southern) is the photographer located?<br><br>• Consider the date of the photograph. December 21 is the Winter Solstice, which is the shortest day of the year in the Northern Hemisphere. Therefore, the photographer must be in the Northern Hemisphere.<br>• Remember that the moon is closer to the horizon during a new moon. This would suggest that the photographer is in the Northern Hemisphere, where the moon appears lower in the sky during winter."<br>|

[Figure 39]

- (b)

- Figure 12: Multi-disciplinary examples under agent recognition and reasoning.

