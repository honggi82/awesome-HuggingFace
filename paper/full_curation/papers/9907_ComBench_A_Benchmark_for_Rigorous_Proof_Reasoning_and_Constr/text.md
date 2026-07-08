# arXiv:2606.10479v1[cs.AI]9Jun2026

## ComBench: A Benchmark for Rigorous Proof Reasoning and Constructive Realization in Olympiad-Level Combinatorics

Shunkai Zhang2,1,* Haoran Zhang3,1,* Yun Luo1,† Qianjia Cheng1

#### Haodi Lei1 Yizhuo Li1 Runzhe Zhan1 Zhilin Wang1 Bangjie Xu4 Yucheng Su4 Xinmiao Han4 Xiaoye Qu1 Dongrui Liu1 Zhouchen Lin2 Yu Qiao1 Ning Ding4, 1 Yafu Li1, 5,† Yu Cheng1,†

1Shanghai AI Laboratory 2Peking University 3Shanghai Jiao Tong University 4Tsinghua University 5The Chinese University of Hong Kong

*Equal contribution. †Corresponding authors.

ABSTRACT

Combinatorics is central to Olympiad-level mathematical problem solving, requiring deep discrete reasoning, creative constructions, and rigorous structural insight. Recent evidence suggests that even today’s strongest frontier models remain uneven on Olympiad combinatorics, revealing a gap in creative mathematical reasoning. We introduce ComBench, an Olympiad-level combinatorics benchmark for evaluating and diagnosing the combinatorial reasoning capabilities of large language models. ComBench contains 100 human-annotated competition-level problems organized around two complementary settings: analysis-centric problems, which primarily require rigorous mathematical arguments, and construction-centric problems, which require explicit constructions in addition to correctness justifications. The evaluation protocol combines rubric-guided proof grading with deterministic construction verification, exposing cases where proof quality and construction validity diverge. Experiments on frontier open- and closed-source models show that ComBench is far from saturated: the strongest model reaches 65.4% overall Avg. and 75.3% overall Best@4. We further find that Rigorous Proof Reasoning and Constructive Realization are distinct capabilities: Kimi-K2.6 trails GPT-5.5 on analysis-centric proof grading but surpasses it on construction-centric Best@4, while Existence and Construction problems remain consistently hardest across representative frontier models.

Project Page Code

### 1 Introduction

Large language models (LLMs) have recently demonstrated rapid progress on mathematical reasoning, with frontier systems approaching elite human performance on several competition-style benchmarks (Luong et al., 2025; An et al., 2025). Nevertheless, their success remains uneven on Olympiad-level combinatorics, a domain that requires not only long-horizon logical deduction but also discrete structural insight and creative construction. For example, Gemini Deep Think and DeepSeekMath-V2(Shao et al., 2025) achieved gold-level performance at IMO 2025, solving five of six problems, yet failed on the hardest combinatorics problem (Luong et al., 2025). It suggests that current models may still lack key ingredients of constructive and creative mathematical reasoning.

Existing Olympiad-level benchmarks (Luong et al., 2025; An et al., 2025) provide important tools for studying such progress, but they do not isolate the constructive nature of combinatorics. Answerbased benchmarks can scale evaluation, but reveal little about whether a model has produced a valid mathematical argument. IMO-Bench-style proof evaluation addresses this limitation by judging proof quality and partial progress, making it well-suited for measuring Rigorous Proof Reasoning: the ability to write complete solutions aligned with Olympiad grading standards. However, many combinatorial problems require an additional capability. A model must not only argue that an object exists, but also realize it explicitly.

|Problem Statement: Consider a 2025×2025 grid of unit squares. Matilda wishes to place on the grid some rectangular tiles, possibly of different sizes, such that each side of every tile lies on a grid line and every unit square is covered by at most one tile.<br><br>Determine the minimum number of tiles Matilda needs to place so that each row and each column of the grid has exactly one unit square that is not covered by any tile.<br><br>[Figure 1]<br><br>IMO 2025 Problem 6<br><br>[Figure 2]|
|---|

|[Figure 3]<br><br>[Figure 4]<br><br>Answer and Proof<br><br>Answer: 2112. In general, the answer turns out to be ⌈𝑛 + 2 𝑛 − 3 ⌉, but when 𝑛 is not a perfect square the solution is more complicated. Proof: Bound. There are several approaches (all hard), but the shortest proof seems to be the following one that exploits the 𝐸𝑟𝑑ő𝑠-Szekeres theorem; it is Solution 6 in the shortlist. The theorem being quoted is: Let 𝑛 ≥ 1 be an integer. Given a permutation of (1,…,𝑛), if a longest increasing subsequence (LIS) has length 𝑎 and a longest decreasing subsequence (LDS) has length 𝑏, then 𝑎𝑏 ≥ 𝑛.<br><br>...[MORE CONTENT]|
|---|

|[Figure 5]<br><br>[Figure 6]<br><br>Construction<br><br>[Figure 7]<br><br>A construction solution to the problem in grid size 25×25.|
|---|

Figure 1: IMO 2025 P6, a challenging combinatorics problem unsolved by all evaluated models. The figure shows the original problem statement, the reference answer, and a schematic illustration of the reference construction.

We refer to this second capability as Constructive Realization: the ability to turn an existential or constructive argument into a complete, unambiguous, and machine-checkable discrete witness. Such witnesses may be colorings, tilings, graphs, set families, strategies, matrices, permutations, or counterexamples. Natural-language descriptions of these objects are often difficult to evaluate reliably, since their validity depends on global constraints over the entire object. For example, in IMO 2025 Problem 6 (Figure 1), a complete solution requires not only proving the optimal value, but also exhibiting a tile distribution that attains it.

In this study, we propose ComBench, a benchmark of 100 Olympiad-level combinatorics problems designed to evaluate both Rigorous Proof Reasoning and Constructive Realization. ComBench extends IMO-Bench-style proof evaluation with executable construction verification. It contains two types of records. Analysis-centric records evaluate the original mathematical solution using rubric-guided proof grading. Construction-centric records retain the original proof task but add a construction task, where the model must output a concrete witness in a prescribed representation, and an item-specific verifier checks the witness deterministically. The distinction lies in the evaluation target.

ComBench is curated from major Olympiad-style competitions, including IMO, USAMO, team selection tests, regional Olympiads, and IMO Shortlist problems. Each problem is annotated with grading guidelines following a 0/1/6/7-point rubric. Construction-centric records are further equipped with construction instructions, reference witnesses, verifier plans, and deterministic Python verifiers. Multiple math experts provide rigorous and carefully justified annotations concerning grading guidelines, combinatorial construction instructions, and deterministic verifiers. This design lets us evaluate proof quality and construction validity separately, and then combine them through verifier-gated scoring when a construction is required.

We evaluate a range of frontier closed-source and open-source LLMs on ComBench, with several findings illustrating the benchmark’s diagnostic value. Overall, Olympiad-level combinatorics remains far from saturated: the strongest model, GPT-5.5, achieves 65.4% overall Avg. and 75.3% overall Best@4 under our rubric-guided and verifier-gated evaluation. Model behavior also differs sharply across the two target capabilities. Kimi-K2.6 is substantially weaker than GPT-5.5 on analysis-centric proof grading, yet achieves the best construction-centric Best@4 score, surpassing GPT-5.5. At the category level, Existence and Construction problems are consistently the hardest for representative frontier models, while Counting and Graph Theory problems are comparatively easier. These findings suggest that Rigorous Proof Reasoning and Constructive Realization are related but distinct capabilities, and that progress on Olympiad-level combinatorics requires both proof-level rigor and explicit witness realization.

Contributions. This work makes the following contributions:

- • We introduce ComBench, a 100-problem Olympiad-level combinatorics benchmark for Rigorous Proof Reasoning and Constructive Realization.
- • We develop an evaluation protocol that combines IMO-Bench-style rubric-guided proof grading with deterministic verification of explicit construction witnesses.
- • We provide a systematic empirical evaluation of frontier closed-source and open-source LLMs on ComBench, revealing a clear separation between proof-level reasoning and explicit construction ability, as well as category-specific weaknesses in Olympiad-level combinatorics.

### 2 Related Work

Evaluation of Mathematical Reasoning. Mathematical benchmarks have evolved from GSM8K and MATH (Cobbe et al., 2021; Hendrycks et al., 2021) to Olympiad- and frontier-level evaluations (He et al., 2024; Balunovi´c et al., 2026; Glazer et al., 2025; Phan et al., 2025; Wang et al., 2026), dominated by three protocols: answer-only matching as in AMO-Bench and OlymMATH (An et al., 2025; Sun et al., 2026), rubric-guided 0–7 proof grading as in IMO-Bench and ProofGrader (Luong et al., 2025; Ma et al., 2026), and Lean-based formal proving benchmarks such as CombiBench (Liu

- et al., 2025), which formalizes combinatorial problems in Lean 4 and evaluates theorem-proving and fill-in-the-blank formal solutions. Yet Olympiad combinatorics, where solutions hinge on globally constrained witnesses such as colorings, set families, or counter-examples, remains underserved by these protocols: answer matching cannot expose invalid witnesses, and proof graders rate the prose without checking the construction itself. Complementing the Lean-formalization line, ComBench retains natural-language solutions but adds executable verification of explicit witnesses.

LLMs for Olympiad-level Reasoning. After early neuro-symbolic and formal-proof systems reached medal-level IMO performance (Chervonyi et al., 2025; Hubert et al., 2025), large-scale posttraining with reinforcement learning, further refined by recipes around scaling, training strategies, and tool-integrated reasoning, has substantially lifted the mathematical capabilities of general-purpose LLMs (Guo et al., 2025; Team et al., 2026; Yang et al., 2025; Cheng et al., 2026; Zhang et al., 2026). Building on this trajectory, recent thinking systems have reached IMO-gold-level natural-language performance under contest conditions (Luong et al., 2025; Huang and Yang, 2025; Chen et al., 2025; Yang et al., 2026; Li et al., 2026), yet the hardest combinatorics problems remain unsolved: no model received credit on IMO 2025 P6. ComBench is designed to disentangle and measure precisely this gap.

### 3 ComBench Dataset

#### 3.1 Overview

We introduce ComBench, a benchmark of 100 Olympiadlevel combinatorics problems designed to evaluate mathematical reasoning beyond final-answer correctness. It is balanced between 50 analysis-centric problems and 50 construction-centric problems. The benchmark spans 15 competition sources from 2000 to 2025, including the International Mathematical Olympiad (IMO), United States of America Mathematical Olympiad (USAMO), team selection tests, regional Olympiads, and IMO Shortlist problems. This source diversity combines canonical international problems with national and selection-test problems that broaden coverage of constructive, extremal, graph-theoretic, counting, and strategy-driven combinatorics. Figure 2 visualizes this composition: the 100 problems are split evenly between analysis-centric and construction-centric records, and are organized into five primary combinatorics categories.

The two problem types operationalize the paper’s two target capabilities: analysis-centric records evaluate Rigorous

7% 3% 7% 5%

6%

28%

7%

16%

7% 17%

Figure 2: Distribution of ComBench categories.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

LLM

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Assemble JSONL data

Semantic Audit

Inputs

specification issue

LLM

[Figure 19]

- • problem
- • reference answer
- • human intent revise

- • instruction
- • reference construction
- • grading guidelines

[Figure 20]

###### Code Verifier

[Figure 21]

[Figure 22]

Reference Check

[Figure 23]

The Python code to verify the combinatorial construction

debug

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

with human efforts

Format Structure Math

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Human Check

[Figure 38]

Final Record

[Figure 39]

[Figure 40]

Code verifier generation

scope & validity rubrics

benchmark data

LLM

verifier issue

Figure 3: Construction-centric data builder pipeline for ComBench.

Proof Reasoning, while construction-centric records additionally evaluate Constructive Realization through an explicit, mechanically checkable witness. Aggregate source, year, category, and problem-type statistics are provided in Appendix A.

#### 3.2 Record Design and Problem Taxonomy

We select Olympiad-level combinatorics problems centered on discrete structures, extremal reasoning, counting, graph-theoretic arguments, strategies, or explicit constructions. The selected problems include standard Olympiad-style proof tasks as well as tasks whose natural solutions require explicit witnesses such as colorings, permutations, graphs, tables, set systems, tilings, strategies, or counterexamples. To better target combinatorial reasoning, we exclude problems whose core difficulty is non-combinatorial, even if they involve minor counting or discrete elements.

Each ComBench record contains the information needed for rubric-guided proof evaluation: the original problem statement, source metadata, reference answer, reference solution when available, primary problem category, problem type, and problem-specific grading guidelines. The grading guidelines follow a 0/1/6/7-point rubric and specify what counts as no progress, minimal progress, nearly complete progress, and a complete solution. Construction-centric records contain additional fields that define the executable construction task: an item-specific construction instruction, a reference witness, an informal verifier plan, and deterministic Python verifier code. Appendix C.2 gives a complete formatted example of one construction-centric record.

For each construction-centric problem, the witness is the concrete discrete object requested by the construction instruction. It is not a proof sketch or natural-language justification; rather, it is the object whose representation and mathematical validity can be checked by a verifier. This design makes the construction task explicit, rather than relying on free-form object descriptions that may be difficult to parse or reproduce. Representative witness formats are shown in Appendix C.3.

ComBench uses five primary problem categories: extremal problems, existence and construction, operations and strategies, graph theory, and counting. These categories provide a stable way to analyze recurring reasoning patterns rather than mutually exclusive mathematical subfields. Category definitions and the category-by-type breakdown are provided in Appendix A.

#### 3.3 Annotation Pipeline

Both analysis-centric and construction-centric records require annotation. For analysis-centric records, the main annotation is the problem-specific 0/1/6/7-point grading guideline used for rubric-guided proof evaluation. Construction-centric records use the same proof-rubric annotation, but additionally require a construction instruction, a reference witness, an informal verifier plan, deterministic verifier code, and semantic audit. Because rubric construction is shared by both record types, we focus below on the full construction-centric pipeline, which extends proof-rubric annotation with witness specification and executable verification, as shown in Figure 3.

- Stage 1: specification and rubric construction. The inputs include the original problem, reference answer, and human-expert intent for the customized construction task, together with a reference witness. The intent specifies what kind of witness should be requested and what mathematical target it must realize. Given these inputs, the LLM drafts an item-specific construction instruction, standardizes the reference construction into the required format, and produces problem-specific grading guidelines under the 0/1/6/7-point rubric. Human reviewers then check whether the construction instruction preserves the original problem meaning, whether the reference construction satisfies the intended target, and whether the grading guideline captures meaningful Olympiad-style proof progress.
- Stage 2: verifier generation and semantic audit. After human review, the problem statement and construction instruction are used to generate deterministic Python verifier code. The verifier reads only the raw construction payload and checks both the declared representation and the required mathematical constraints. We then use an LLM-assisted semantic audit to inspect the full chain from the original problem to the executable verifier. The audit checks whether the construction target, instruction, witness, verifier plan, and code remain aligned with the original mathematical goal, and flags either an upstream specification issue or a verifier-generation issue when they diverge.
- Stage 3: record assembly and executable reference check. Records passing the semantic audit are assembled into JSONL format. As an executable reference check, the reference witness is passed to the verifier through the same raw-standard-input interface used for model submissions. A record is accepted only if this check succeeds. It then receives a final human inspection of verifier strictness, coverage, consistency, and diagnostic clarity. The full record schema, verifier I/O contract, and human audit checklist are described in Appendix B, C, and D.

#### 3.4 Quality Control and Transparency

Quality control in ComBench targets three main risks. First, we check target fidelity: the problem statement, reference answer, grading guideline, construction instruction, and reference witness must refer to the same mathematical claim. Second, we check verifier adequacy: each reference witness must be parseable and accepted by its verifier, and each verifier must enforce the intended global combinatorial constraints rather than only superficial format properties. Third, we check metadata and evaluation consistency: source identifiers, categories, problem types, record counts, and final JSONL files must match the files used by the evaluation pipeline. Additional record-schema and quality-control details are provided in Appendix C.

We also check the overlap between ComBench and IMO-Bench. We compare all 100 problems using source metadata, problem-statement similarity, and manual inspection of high-similarity candidates. This process identifies 14 overlapping or substantially modified problems, yielding an overlap rate of 14%; the remaining 86 problems show no high-confidence match. The matching protocol and released overlap metadata are described in Appendix H.

[Figure 41]

Model Response

[Figure 42]

- Q1 Solution

- Q2 Construction: <construct>...</construct>

#### 3.5 Combining Proof and Construction Scores

[Figure 43]

Proof Judge

ComBench keeps proof grading and construction verification as separate signals so that Rigorous Proof Reasoning and Constructive Realization can be analyzed independently. For model ranking, however, construction-centric records also require a single score. Analysis-centric records use the proof score returned by the rubric-guided judge. For construction-centric records, we obtain a proof score p ∈ {0,1,6,7} and a binary construction score c ∈ {0,1}. If construction passes, the final score remains p. If construction fails, high proof scores are demoted: 7  → 6 and 6  → 1, while 0 and 1 remain unchanged.

[Figure 44]

Input: Q1 Solution Output: Proof Score {0, 1, 6, 7}

[Figure 45]

[Figure 46]

Code Verifier

[Figure 47]

Input: Combinatorial Construction Output: Pass or not

[Figure 48]

Verifier-Gated Demotion

[Figure 49]

If construction passes: keep proof score If construction fails: 7→6, 6→1, 1→1, 0→0

[Figure 50]

[Figure 51]

Final Score

Figure 4 illustrates this verifier-gated rule. The rule is motivated by manual inspection of construction-centric responses: when a high-scoring proof fails executable construction verification, the proof often contains a hidden construction gap, such as asserting

Used for Avg. and Pass@4

Figure 4: Verifier-gated scoring rule for construction-centric records.

existence at a high level, leaving the witness under-specified, or failing to instantiate the object in a checkable form. Thus a valid witness is not treated as an extra bonus; it confirms that a high-scoring constructive solution can actually be realized. Conversely, failed construction should not erase all proof progress, since the response may still contain meaningful mathematical reasoning. The demotion rule therefore calibrates high proof scores using construction validity while preserving low proof scores that already indicate insufficient progress. Appendix J provides examples of this scoring rationale.

### 4 Experiments

The experiments evaluate what current models can and cannot do under the two capabilities introduced earlier: Rigorous Proof Reasoning and Constructive Realization. We first report overall performance on ComBench, then examine whether success on proof reasoning transfers to construction realization, which combinatorial categories remain difficult, and what mathematical proof errors dominate below-full-credit solutions.

#### 4.1 Evaluation Configuration

Models and sampling. We evaluate ten open- and closed-source models: GPT-5.5 (OpenAI, 2026), Gemini-3.1-Pro-Preview-Thinking (Google DeepMind, 2026), Kimi-K2.6 (Moonshot AI, 2026), DeepSeek-V4-Pro (DeepSeek-AI, 2026), Nemotron-Cascade-2-30B-A3B (Yang et al., 2026), GLM-5.1 (GLM-5-Team et al., 2026), Qwen3.6-35B-A3B (Qwen Team, 2026), Qwen3.6-MaxPreview (Qwen Team, 2026), Google-Gemma-4-31B-IT (Google DeepMind, 2026), and SU-01(Li et al., 2026). We use temperature 0.6, model-specific maximum token limits, and four sampled solutions per item.

Prompt format. The prompt follows the record type. Analysis-centric records ask for a complete Olympiad-style solution to the original problem. Construction-centric records contain two questions: the original proof task and an item-specific construction task. The construction answer must provide one <construct>...</construct> payload, read by the executable verifier. Representative prompt templates are provided in Appendix E.2.

Evaluation components. Proofs are scored by a rubric-guided LLM judge using the problem statement, reference solution when available, and problem-specific grading guidelines. The judge returns one of 0,1,6,7 points out of 7. For construction-centric records, the extracted construction payload is additionally passed to the item-specific Python verifier and receives a binary pass/fail result. The verifier-gated score used for construction-centric records follows Section 3.5. We further conduct a manual audit of sampled proof-judge decisions in Appendix I.

Metrics. We report Avg., the average normalized score over all generated solutions, and Best@4, which takes the best score among four sampled solutions for each record before averaging across records. Avg. measures overall model capability across sampled solutions, while Best@4 measures potential performance under repeated sampling. Full prompt templates, parsing rules, evaluator details, and failure handling are provided in Appendix E.

#### 4.2 Main Results

The main results report analysis-centric, construction-centric, and overall performance; all values are percentages, and construction-centric scores use the verifier-gated rule in Section 3.5. Table 1 shows that ComBench is far from saturated: the strongest model, GPT-5.5, reaches only 65.4% overall Avg. and 75.3% overall Best@4, leaving substantial headroom even under best-of-four sampling. Thus Olympiad-level combinatorics remains difficult under rubric-guided and verifier-gated evaluation.

The rankings reveal both model tiers and capability separation. GPT-5.5 leads overall and on analysiscentric records, while Gemini-3.1-Pro remains close in overall Best@4. Kimi-K2.6, however, achieves the highest construction-centric Best@4 despite a lower analysis-centric score, showing that Constructive Realization is not merely a by-product of stronger Rigorous Proof Reasoning. DeepSeek-V4-Pro forms a middle tier below the top three but above the remaining models. The lower-scoring group, including Qwen3.6-Max, SU-01, GLM-5.1, Qwen3.6-35B, Nemotron-Cascade,

Analysis-centric Construction-centric Overall Avg.(%) Best@4(%) Avg.(%) Best@4(%) Avg.(%) Best@4(%)

Model

GPT-5.5 62.4 72.9 68.4 77.7 65.4 75.3 Gemini-3.1-Pro 56.1 69.7 64.5 78.3 60.3 74.0 Kimi-K2.6 43.5 60.6 63.4 83.7 53.5 72.1 DeepSeek-V4-Pro 37.8 56.6 52.6 67.7 45.2 62.1 Qwen3.6-Max 21.4 32.9 28.4 39.1 24.9 36.0 SU-01 20.9 30.3 28.8 41.1 24.8 35.7 GLM-5.1 21.6 36.0 25.6 37.1 23.6 36.6 Qwen3.6-35B 17.9 26.6 22.7 32.0 20.3 29.3 Nemotron-Cascade 21.8 32.9 17.4 28.0 19.6 30.4 Gemma-4-31B-IT 16.1 24.3 17.5 30.9 16.8 27.6

Table 1: Main evaluation results on ComBench.

and Gemma-4-31B-IT, further shows that ComBench remains challenging for smaller or more specialized models. SU-01 is especially informative: its overall score is limited by format and instruction-following failures, yet parseable responses still show nontrivial mathematical performance on construction-centric records.

These patterns motivate verifier-gated evaluation: a fluent proof can receive high natural-language credit without producing the claimed object in a checkable form. The results suggest that future mathematical models must improve not only proof fluency, but also target identification, structural planning, and explicit witness synthesis. Additional per-category and per-problem results are reported in Appendix F.

#### 4.3 Subset Analysis

We next separate the main results into two views: construction-centric behavior and category-level behavior. The first view tests whether proof success is accompanied by a valid witness, while the second view asks which forms of combinatorial reasoning remain most difficult.

Construction-centric behavior. Table 2 focuses on the 50 construction-centric records, with all values reported as percentages. The proofscore column measures Rigorous Proof Reasoning, the construction pass rate measures Constructive Realization, and the verifier-gated score measures whether the two are jointly present. Across models, verifier-gated scores are lower than proof scores, showing that high-scoring proofs are not always accompanied by valid executable constructions.

The construction pass rate further isolates witness realization in the prescribed representation. The strongest models reach a 75.0% per-sample construction pass rate, still leaving many failed witnesses even among frontier systems. KimiK2.6 matches the top construction pass rate and achieves a strong verifier-gated score, despite weaker analysis-centric performance in Table 1. This reinforces the main-results observation that construction-centric records test more than proof difficulty: they require translating a mathematical target into a complete discrete object satisfying global constraints. Appendix J gives two GPT-5.5 examples where high proof scores are demoted because the response lacks an accepted mechanically checkable witness.

Model Proof Construction Score

GPT-5.5 72.3 75.0 68.4 Gemini-3.1-Pro 68.9 75.0 64.5 Kimi-K2.6 66.3 75.0 63.4 DeepSeek-V4-Pro 56.1 62.5 52.6 SU-01 29.6 27.5 28.8 Qwen3.6-Max 31.5 38.0 28.4 GLM-5.1 29.0 27.0 25.6 Qwen3.6-35B 24.9 40.5 22.7 Gemma-4-31B-IT 20.9 15.0 17.5 Nemotron-Cascade 18.4 20.5 17.4

Table 2: Construction-centric subset analysis. All values are average percentages.

GPT-5.5 Gemini Kimi DeepSeek

100

Avg.score(%)

80

60

40

20

0

Extremal Exist./Constr. Operations Graph Counting

Figure 5: Category-level performance of four representative frontier models.

Missing Core Mechanism Wrong Mathematical Target

[Figure 52]

[Figure 53]

[Figure 54]

Misses the key idea needed to complete the proof.

Proves the wrong statement or optimizes the wrong quantity.

False Lemma

Faulty Induction

[Figure 55]

[Figure 56]

Uses induction incorrectly with a broken recursive step.

Relies on a false claim or invalid intermediate fact.

[Figure 57]

Unjustified Leap Incomplete Case Analysis

[Figure 58]

[Figure 59]

Skips intermediate reasoning between given conditions and result.

Misses necessary cases or boundary conditions.

[Figure 60]

Other Mathematical Error

Format or Unscored

[Figure 61]

[Figure 62]

Rare low-frequency errors merged for visualization.

Unscorable due to malformed output or evaluation failure.

Figure 6: Distribution of primary proof error reasons over all below-full-credit proof samples. Primary reasons are mutually exclusive by construction, while secondary reason tags may overlap.

Category-level behavior. Figure 5 breaks down category-level performance for the four strongest representative models. Analysis-centric records use proof scores, while construction-centric records use verifier-gated scores. Across all four models, Existence and Construction is the hardest category, with scores below the corresponding model’s performance on the other categories. This suggests that current LLMs remain weak at existence-style combinatorial reasoning: they often struggle to identify the right global object, maintain all constraints simultaneously, and turn an existence argument into a complete witness.

Counting and Graph Theory obtain the highest scores, indicating that current models are relatively stronger when the solution can be organized around enumerative structure, graph reformulation, or local structural constraints. By contrast, Existence and Construction, Operations and Strategies, and Extremal Problems often require selecting the correct global target, maintaining an invariant across a process, or matching a sharp bound with a concrete construction. This imbalance suggests that models have acquired some local structural reasoning, but remain weak at global combinatorial planning: they can often manipulate a given representation, yet struggle to choose the right object, invariant, or extremal structure before the proof begins.

#### 4.4 Error Analysis

This section analyzes proof-side failures only. A generated solution is included if its proof score is 0, 1, or 6, or if the proof component is unscored because of parsing or evaluation failure. Construction failure is analyzed separately through the construction pass rate and verifier-gated score in Table 2. The two analyses are complementary: proof errors diagnose failures in Rigorous Proof Reasoning, while construction failures diagnose whether the model can realize the required witness in a complete and checkable form.

We assign one primary reason to each below-full-credit proof critique using mathematically grounded proof-error types. Each case is first annotated with a fine-grained category, and mathematical categories below the global 2% frequency threshold are merged into Other Mathematical Error for the paper-facing distribution. Primary reasons are mutually exclusive by construction, while secondary tags may overlap. Figure 6 shows that most failures are not merely formatting errors. The dominant failure mode is Missing Core Mechanism, accounting for 41.2% of below-full-credit proof solutions. A further 20.0% fall under Wrong Mathematical Target, where the model proves or optimizes the wrong statement. Format or Unscored accounts for 11.0%, while false lemmas, faulty induction, unjustified leaps, and incomplete case analysis together account for 25.5%. Full definitions, fine-grained counts, and per-model error distributions are provided in Appendix G.

These failures indicate that current models are not limited only by response formatting or proof presentation. The dominant errors point to deeper weaknesses in global proof planning: models often miss the central combinatorial mechanism, optimize or prove the wrong target, or leave a crucial global obligation unclosed. In combination with the verifier results, this suggests that stronger Olympiad-level combinatorics systems need more than local algebraic or pattern-matching skill; they must improve target selection, invariant discovery, proof closure, and executable witness synthesis.

### 5 Conclusion

We present ComBench, a 100-problem Olympiad combinatorics benchmark that decouples Rigorous Proof Reasoning from Constructive Realization via a verifier-gated protocol combining rubric proof grading with deterministic witness checking. Frontier models exhibit a clear dissociation between the two capabilities—strong proofs can lack valid witnesses, while weaker proof grading can coexist with top construction performance—and below-credit proof failures concentrate on missing core mechanisms and wrong mathematical targets, calling for advances in global proof planning and explicit witness synthesis.

### Limitations

ComBench is designed to make constructive combinatorial reasoning more directly evaluable, but it has several limitations. First, the deterministic verifier can only check a construction target that has been explicitly formalized in advance. This makes the evaluation reliable for the prescribed witness interface, but it does not cover every valid natural-language construction a model might describe. A model may have a mathematically meaningful construction idea that fails because it is not expressed in the required representation, while the verifier cannot judge alternative formulations outside its specification.

Second, ComBench focuses on Olympiad-level combinatorics. This focus is intentional, since combinatorics strongly stresses discrete structure, extremal reasoning, strategies, and explicit constructions. However, performance on ComBench should not be interpreted as a complete measure of Olympiad mathematics ability. Other domains such as algebra, geometry, and number theory require different forms of abstraction, symbolic manipulation, and proof technique, which are outside the scope of this benchmark.

### Ethical Considerations

ComBench is built from public Olympiad-style combinatorics problems and does not contain personal, demographic, or sensitive user data. The main risks of this benchmark are therefore not privacy-related, but concern interpretation and use. Scores on ComBench should not be taken as a complete measure of mathematical intelligence, since the benchmark focuses on combinatorial reasoning and constructioncentric tasks. In addition, some public problems may overlap with existing benchmarks or model training data, so we document overlap with IMO-Bench and encourage cautious interpretation of results on public-source problems.

All annotators involved in problem annotation, grading-guideline construction, construction specification, and verifier checking received compensation corresponding to their annotation workload.

The automatic evaluation pipeline also requires care. The LLM-based proof judge may disagree with expert human graders, and executable verifiers only check pre-specified construction targets rather than arbitrary natural-language mathematical reasoning. We therefore treat ComBench as a diagnostic benchmark for mathematical reasoning and witness realization, rather than as the sole basis for claims about model capability.

### References

DeepSeek-AI. DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence. 2026. Google DeepMind. Gemma 4: Byte for Byte, the Most Capable Open Models. https://blog.

google/innovation-and-ai/technology/developers-tools/gemma-4/,

2026. Google Keyword Blog. Accessed: 2026-05-26. Google DeepMind. Gemini 3.1 Pro. https://deepmind.google/models/gemini/pro/,

2026. Accessed: 2026-05-26. Moonshot AI. Kimi K2.6: Advancing Open-Source Coding. https://www.kimi.com/blog/ kimi-k2-6, 2026. Moonshot AI Tech Blog. Accessed: 2026-05-26.

OpenAI. Introducing GPT-5.5. 2026. URL https://openai.com/index/ introducing-gpt-5-5/. Accessed: 2026-05-26.

Qwen Team. Qwen3.6-35B-A3B: Agentic Coding Power, Now Open to All. 2026. URL https: //qwen.ai/blog?id=qwen3.6-35b-a3b.

Qwen Team. Qwen3.6-Max-Preview: Smarter, Sharper, Still Evolving. 2026. URL https: //qwen.ai/blog?id=qwen3.6-max-preview.

Shengnan An and Xunliang Cai and Xuezhi Cao and Xiaoyu Li and Yehao Lin and Junlin Liu and Xinxuan Lv and Dan Ma and Xuanlin Wang and Ziwen Wang and Shuang Zhou. AMOBench: Large Language Models Still Struggle in High School Math Competitions. 2025. URL https://arxiv.org/abs/2510.26768.

Mislav Balunovi´c and Jasper Dekoninck and Ivo Petrov and Nikola Jovanovi´c and Martin Vechev. MathArena: Evaluating LLMs on Uncontaminated Math Competitions. 2026. URL https: //arxiv.org/abs/2505.23281.

Jiangjie Chen and Wenxiang Chen and Jiacheng Du and Jinyi Hu and Zhicheng Jiang and Allan Jie and Xiaoran Jin and Xing Jin and Chenggang Li and Wenlei Shi and Zhihong Wang and Mingxuan Wang and Chenrui Wei and Shufa Wei and Huajian Xin and Fan Yang and Weihao Gao and Zheng Yuan and Tianyang Zhan and Zeyu Zheng and Tianxi Zhou and Thomas Hanwen Zhu. Seed-Prover 1.5: Mastering Undergraduate-Level Theorem Proving via Learning from Experience. 2025. URL https://arxiv.org/abs/2512.17260.

Qianjia Cheng and Yuchen Zhang and Zhilin Wang and Yuxin Zuo and Shunkai Zhang and Yuchen Fan and Yu Qiao and Bowen Zhou and Ning Ding and Yu Cheng and Yun Luo and Ganqu Cui. Teaching Thinking Models to Reason with Tools: A Full-Pipeline Recipe for Tool-Integrated Reasoning. 2026. URL https://arxiv.org/abs/2605.06326.

Yuri Chervonyi and Trieu H. Trinh and Miroslav Olˇs´ak and Xiaomeng Yang and Hoang Nguyen and Marcelo Menegali and Junehyuk Jung and Junsu Kim and Vikas Verma and Quoc V. Le and Thang Luong. Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2. 2025. URL https://arxiv.org/abs/2502.03544.

Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Mark Chen and Heewoo Jun and Lukasz Kaiser and Matthias Plappert and Jerry Tworek and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and John Schulman. Training Verifiers to Solve Math Word Problems. 2021. URL https://arxiv.org/abs/2110.14168.

Elliot Glazer and Ege Erdil and Tamay Besiroglu and Diego Chicharro and Evan Chen and Alex Gunning and Caroline Falkman Olsson and Jean-Stanislas Denain and Anson Ho and Emily de Oliveira Santos and Olli J¨arviniemi and Matthew Barnett and Robert Sandler and Matej Vrzala and Jaime Sevilla and Qiuyu Ren and Elizabeth Pratt and Lionel Levine and Grant Barkley and Natalie Stewart and Bogdan Grechuk and Tetiana Grechuk and Shreepranav Varma Enugandla and Mark Wildon. FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI. 2025. URL https://arxiv.org/abs/2411.04872.

GLM-5-Team and : and Aohan Zeng and Xin Lv and Zhenyu Hou and Zhengxiao Du and Qinkai Zheng and Bin Chen and Da Yin and Chendi Ge and Chenghua Huang and Chengxing Xie and Chenzheng Zhu and Congfeng Yin and Cunxiang Wang and Gengzheng Pan and Hao Zeng and Haoke Zhang and Haoran Wang and Huilong Chen and Jiajie Zhang and Jian Jiao and Jiaqi Guo and Jingsen Wang and Jingzhao Du and Jinzhu Wu and Kedong Wang and Lei Li and Lin Fan and Lucen Zhong and Mingdao Liu and Mingming Zhao and Pengfan Du and Qian Dong and Rui Lu and Shuang-Li and Shulin Cao and Song Liu and Ting Jiang and Xiaodong Chen and Xiaohan Zhang and Xuancheng Huang and Xuezhen Dong and Yabo Xu and Yao Wei and Yifan An and Yilin Niu and Yitong Zhu and Yuanhao Wen and Yukuo Cen and Yushi Bai and Zhongpei Qiao and Zihan Wang and Zikang Wang and Zilin Zhu and Ziqiang Liu and Zixuan Li and Bojie Wang and Bosi Wen and Can Huang and Changpeng Cai and Chao Yu and Chen Li and Chengwei Hu and Chenhui Zhang and Dan Zhang and Daoyan Lin and Dayong Yang and Di Wang and Ding Ai and Erle Zhu and Fangzhou Yi and Feiyu Chen and Guohong Wen and Hailong Sun and Haisha Zhao and Haiyi Hu and Hanchen Zhang and Hanrui Liu and Hanyu Zhang and Hao Peng and

Hao Tai and Haobo Zhang and He Liu and Hongwei Wang and Hongxi Yan and Hongyu Ge and Huan Liu and Huanpeng Chu and Jia’ni Zhao and Jiachen Wang and Jiajing Zhao and Jiamin Ren and Jiapeng Wang and Jiaxin Zhang and Jiayi Gui and Jiayue Zhao and Jijie Li and Jing An and Jing Li and Jingwei Yuan and Jinhua Du and Jinxin Liu and Junkai Zhi and Junwen Duan and Kaiyue Zhou and Kangjian Wei and Ke Wang and Keyun Luo and Laiqiang Zhang and Leigang Sha and Liang Xu and Lindong Wu and Lintao Ding and Lu Chen and Minghao Li and Nianyi Lin and Pan Ta and Qiang Zou and Rongjun Song and Ruiqi Yang and Shangqing Tu and Shangtong Yang and Shaoxiang Wu and Shengyan Zhang and Shijie Li and Shuang Li and Shuyi Fan and Wei Qin and Wei Tian and Weining Zhang and Wenbo Yu and Wenjie Liang and Xiang Kuang and Xiangmeng Cheng and Xiangyang Li and Xiaoquan Yan and Xiaowei Hu and Xiaoying Ling and Xing Fan and Xingye Xia and Xinyuan Zhang and Xinze Zhang and Xirui Pan and Xu Zou and Xunkai Zhang and Yadi Liu and Yandong Wu and Yanfu Li and Yidong Wang and Yifan Zhu and Yijun Tan and Yilin Zhou and Yiming Pan and Ying Zhang and Yinpei Su and Yipeng Geng and Yong Yan and Yonglin Tan and Yuean Bi and Yuhan Shen and Yuhao Yang and Yujiang Li and Yunan Liu and Yunqing Wang and Yuntao Li and Yurong Wu and Yutao Zhang and Yuxi Duan and Yuxuan Zhang and Zezhen Liu and Zhengtao Jiang and Zhenhe Yan and Zheyu Zhang and Zhixiang Wei and Zhuo Chen and Zhuoer Feng and Zijun Yao and Ziwei Chai and Ziyuan Wang and Zuzhou Zhang and Bin Xu and Minlie Huang and Hongning Wang and Juanzi Li and Yuxiao Dong and Jie Tang. GLM-5: from Vibe Coding to Agentic Engineering. 2026. URL https://arxiv.org/abs/2602.15763.

Guo, Daya and Yang, Dejian and Zhang, Haowei and Song, Junxiao and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Zhang, Ruoyu and Ma, Shirong and Bi, Xiao and others. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 2025.

Chaoqun He and Renjie Luo and Yuzhuo Bai and Shengding Hu and Zhen Leng Thai and Junhao Shen and Jinyi Hu and Xu Han and Yujie Huang and Yuxiang Zhang and Jie Liu and Lei Qi and Zhiyuan Liu and Maosong Sun. OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems. 2024. URL https:

//arxiv.org/abs/2402.14008.

Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt. Measuring Mathematical Problem Solving With the MATH Dataset. 2021. URL https://arxiv.org/abs/2103.03874.

Yichen Huang and Lin F. Yang. Winning Gold at IMO 2025 with a Model-Agnostic Verification-andRefinement Pipeline. 2025. URL https://arxiv.org/abs/2507.15855.

Hubert, Thomas and Mehta, Rishi and Sartran, Laurent and Horv´ath, Mikl´os Z and Zuˇˇ zi´c, Goran and Wieser, Eric and Huang, Aja and Schrittwieser, Julian and Schroecker, Yannick and Masoom, Hussain and others. Olympiad-level formal mathematical reasoning with reinforcement learning. Nature, 2025.

Yafu Li and Runzhe Zhan and Haoran Zhang and Shunkai Zhang and Yizhuo Li and Zhilin Wang and Jiacheng Chen and Futing Wang and Xuyang Hu and Yuchen Fan and Bangjie Xu and Yucheng Su and Xinmiao Han and Chenxi Li and Haodi Lei and Yufeng Zhao and Zejin Lin and Qianjia Cheng and Tong Zhu and Xiaoye Qu and Ganqu Cui and Peng Ye and Yun Luo and Zhouchen Lin and Yu Qiao and Bowen Zhou and Ning Ding and Yu Cheng. Achieving Gold-Medal-Level Olympiad Reasoning via Simple and Unified Scaling. 2026. URL https:

//arxiv.org/abs/2605.13301.

Junqi Liu and Xiaohan Lin and Jonas Bayer and Yael Dillies and Weijie Jiang and Xiaodan Liang and Roman Soletskyi and Haiming Wang and Yunzhou Xie and Beibei Xiong and Zhengfeng Yang and Jujian Zhang and Lihong Zhi and Jia Li and Zhengying Liu. CombiBench: Benchmarking LLM Capability for Combinatorial Mathematics. 2025. URL https://arxiv.org/abs/2505.

03171.

Thang Luong and Dawsen Hwang and Hoang H. Nguyen and Golnaz Ghiasi and Yuri Chervonyi and Insuk Seo and Junsu Kim and Garrett Bingham and Jonathan Lee and Swaroop Mishra and Alex Zhai and Clara Huiyi Hu and Henryk Michalewski and Jimin Kim and Jeonghyun Ahn and Junhwi Bae and Xingyou Song and Trieu H. Trinh and Quoc V. Le and Junehyuk Jung. Towards Robust Mathematical Reasoning. 2025. URL https://arxiv.org/abs/2511.01846.

Wenjie Ma and Andrei Cojocaru and Neel Kolhe and Bradley Louie and Robin Said Sharif and Haihan Zhang and Vincent Zhuang and Matei Zaharia and Sewon Min. Reliable Fine-Grained Evaluation of Natural Language Math Proofs. 2026. URL https://arxiv.org/abs/2510.13888.

Phan, Long and Gatti, Alice and Han, Ziwen and Li, Nathaniel and Hu, Josephina and Zhang, Hugh and Zhang, Chen Bo Calvin and Shaaban, Mohamed and Ling, John and Shi, Sean and others. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.

Zhihong Shao and Yuxiang Luo and Chengda Lu and Z. Z. Ren and Jiewen Hu and Tian Ye and Zhibin Gou and Shirong Ma and Xiaokang Zhang. DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning. 2025. URL https://arxiv.org/abs/2511.22570.

Haoxiang Sun and Yingqian Min and Zhipeng Chen and Wayne Xin Zhao and Ji-Rong Wen. Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models. 2026. URL https://arxiv.org/abs/2503.21380.

Kimi Team and Yifan Bai and Yiping Bao and Y. Charles and Cheng Chen and Guanduo Chen and Haiting Chen and Huarong Chen and Jiahao Chen and Ningxin Chen and Ruijue Chen and Yanru Chen and Yuankun Chen and Yutian Chen and Zhuofu Chen and Jialei Cui and Hao Ding and Mengnan Dong and Angang Du and Chenzhuang Du and Dikang Du and Yulun Du and Yu Fan and Yichen Feng and Kelin Fu and Bofei Gao and Chenxiao Gao and Hongcheng Gao and Peizhong Gao and Tong Gao and Yuyao Ge and Shangyi Geng and Qizheng Gu and Xinran Gu and Longyu Guan and Haiqing Guo and Jianhang Guo and Xiaoru Hao and Tianhong He and Weiran He and Wenyang He and Yunjia He and Chao Hong and Hao Hu and Yangyang Hu and Zhenxing Hu and Weixiao Huang and Zhiqi Huang and Zihao Huang and Tao Jiang and Zhejun Jiang and Xinyi Jin and Yongsheng Kang and Guokun Lai and Cheng Li and Fang Li and Haoyang Li and Ming Li and Wentao Li and Yang Li and Yanhao Li and Yiwei Li and Zhaowei Li and Zheming Li and Hongzhan Lin and Xiaohan Lin and Zongyu Lin and Chengyin Liu and Chenyu Liu and Hongzhang Liu and Jingyuan Liu and Junqi Liu and Liang Liu and Shaowei Liu and T. Y. Liu and Tianwei Liu and Weizhou Liu and Yangyang Liu and Yibo Liu and Yiping Liu and Yue Liu and Zhengying Liu and Enzhe Lu and Haoyu Lu and Lijun Lu and Yashuo Luo and Shengling Ma and Xinyu Ma and Yingwei Ma and Shaoguang Mao and Jie Mei and Xin Men and Yibo Miao and Siyuan Pan and Yebo Peng and Ruoyu Qin and Zeyu Qin and Bowen Qu and Zeyu Shang and Lidong Shi and Shengyuan Shi and Feifan Song and Jianlin Su and Zhengyuan Su and Lin Sui and Xinjie Sun and Flood Sung and Yunpeng Tai and Heyi Tang and Jiawen Tao and Qifeng Teng and Chaoran Tian and Chensi Wang and Dinglu Wang and Feng Wang and Hailong Wang and Haiming Wang and Jianzhou Wang and Jiaxing Wang and Jinhong Wang and Shengjie Wang and Shuyi Wang and Si Wang and Xinyuan Wang and Yao Wang and Yejie Wang and Yiqin Wang and Yuxin Wang and Yuzhi Wang and Zhaoji Wang and Zhengtao Wang and Zhengtao Wang and Zhexu Wang and Chu Wei and Qianqian Wei and Haoning Wu and Wenhao Wu and Xingzhe Wu and Yuxin Wu and Chenjun Xiao and Jin Xie and Xiaotong Xie and Weimin Xiong and Boyu Xu and Jinjing Xu and L. H. Xu and Lin Xu and Suting Xu and Weixin Xu and Xinran Xu and Yangchuan Xu and Ziyao Xu and Jing Xu and Jing Xu and Junjie Yan and Yuzi Yan and Hao Yang and Xiaofei Yang and Yi Yang and Ying Yang and Zhen Yang and Zhilin Yang and Zonghan Yang and Haotian Yao and Xingcheng Yao and Wenjie Ye and Zhuorui Ye and Bohong Yin and Longhui Yu and Enming Yuan and Hongbang Yuan and Mengjie Yuan and Siyu Yuan and Haobing Zhan and Dehao Zhang and Hao Zhang and Wanlu Zhang and Xiaobin Zhang and Yadong Zhang and Yangkun Zhang and Yichi Zhang and Yizhi Zhang and Yongting Zhang and Yu Zhang and Yutao Zhang and Yutong Zhang and Zheng Zhang and Haotian Zhao and Yikai Zhao and Zijia Zhao and Huabin Zheng and Shaojie Zheng and Longguang Zhong and Jianren Zhou and Xinyu Zhou and Zaida Zhou and Jinguo Zhu and Zhen Zhu and Weiyu Zhuang and Xinxing Zu. Kimi K2: Open Agentic Intelligence. 2026. URL https://arxiv.org/abs/2507.20534.

Erik Y. Wang and Sumeet Motwani and James V. Roggeveen and Eliot Hodges and Dulhan Jayalath and Charles London and Kalyan Ramakrishnan and Flaviu Cipcigan and Philip Torr and Alessandro Abate. HorizonMath: Measuring AI Progress Toward Mathematical Discovery with Automatic Verification. 2026. URL https://arxiv.org/abs/2603.15617.

An Yang and Anfeng Li and Baosong Yang and Beichen Zhang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Gao and Chengen Huang and Chenxu Lv and Chujie Zheng and Dayiheng Liu and Fan Zhou and Fei Huang and Feng Hu and Hao Ge and Haoran Wei and Huan Lin and

Jialong Tang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Yang and Jiaxi Yang and Jing Zhou and Jingren Zhou and Junyang Lin and Kai Dang and Keqin Bao and Kexin Yang and Le Yu and Lianghao Deng and Mei Li and Mingfeng Xue and Mingze Li and Pei Zhang and Peng Wang and Qin Zhu and Rui Men and Ruize Gao and Shixuan Liu and Shuang Luo and Tianhao Li and Tianyi Tang and Wenbiao Yin and Xingzhang Ren and Xinyu Wang and Xinyu Zhang and Xuancheng Ren and Yang Fan and Yang Su and Yichang Zhang and Yinger Zhang and Yu Wan and Yuqiong Liu and Zekun Wang and Zeyu Cui and Zhenru Zhang and Zhipeng Zhou and Zihan Qiu. Qwen3 Technical Report. 2025. URL https://arxiv.org/abs/2505.09388.

Zhuolin Yang and Zihan Liu and Yang Chen and Wenliang Dai and Boxin Wang and Sheng-Chieh Lin and Chankyu Lee and Yangyi Chen and Dongfu Jiang and Jiafan He and Renjie Pi and Grace Lam and Nayeon Lee and Alexander Bukharin and Mohammad Shoeybi and Bryan Catanzaro and Wei Ping. Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation. 2026. URL https://arxiv.org/abs/2603.19220.

Yang, Zhuolin and Liu, Zihan and Chen, Yang and Dai, Wenliang and Wang, Boxin and Lin, Sheng-Chieh and Lee, Chankyu and Chen, Yangyi and Jiang, Dongfu and He, Jiafan and Pi, Renjie and Lam, Grace and Lee, Nayeon and Bukharin, Alexander and Shoeybi, Mohammad and Catanzaro, Bryan and Ping, Wei. Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation. 2026.

Zhang, Haoran and Li, Yafu and Wang, Zhi and Wang, Zhilin and Zhang, Shunkai and Qu, Xiaoye and Cheng, Yu. Characterizing, Evaluating, and Optimizing Complex Reasoning. arXiv preprint arXiv:2602.08498, 2026.

### A Dataset Details

This appendix provides the aggregate dataset statistics supporting Section 3. We report the competition source distribution, year coverage, and category-by-type breakdown used in the main text. Full problem-level metadata, including the complete problem-to-category mapping, will be released with the public dataset rather than reproduced as a 100-row appendix table.

Statistic Count Total problems 100 Analysis-centric records 50 Construction-centric records 50 Primary categories 5 Competition sources 15 Year range 2000–2025

Table 3: Dataset summary statistics.

#### A.1 Source and Year Distribution

- Table 4 reports the competition sources. The dataset combines canonical international problems with national and selection-test problems so that the benchmark is not dominated by a single contest family.
- Table 5 reports the year coverage. The distribution emphasizes recent Olympiad-style problems while retaining earlier problems that are useful for classical combinatorial themes.

A.2 Problem Type and Category Breakdown

- Table 6 defines the five primary categories used in ComBench. Table 7 gives the category-by-type breakdown. The construction-centric portion is intentionally concentrated in categories where explicit witnesses naturally arise, especially extremal problems and existence-style tasks.

Source Count Source Count IMO 32 CMO 3 TSTST 14 CGMO 3 USAMO 13 IMOSLC 3 USATST 8 USEMO 2 RMO 6 CTST 2 CSMO 4 ELMO 2 EGMO 3 EMC 2 APMO 3

Table 4: Competition source distribution in ComBench.

Year range Count 2000–2009 12 2010–2014 11 2015–2019 32 2020–2023 33 2024–2025 12

Table 5: Year distribution in ComBench.

#### A.3 Released Metadata Fields

The public dataset release will include problem-level metadata and annotations needed to reproduce the aggregate analyses in this paper. Each released record will include the problem identifier, source, year, problem statement, problem type, primary category, reference answer, reference solution when available, and grading guidelines. Construction-centric records will additionally include the construction instruction, reference witness, verifier plan, and verifier code. These fields allow users to aggregate results by source, year, problem category, and evaluation target without requiring a long problem-level table in the paper appendix.

### B Construction Annotation Pipeline

This appendix expands the three-stage construction-centric annotation pipeline summarized in Section 3. The pipeline converts a constructive combinatorics problem into a record containing a proof-grading rubric and an executable witness-verification interface.

#### B.1 Stage 1: Specification and Rubric Construction

- Stage 1 starts from the original problem, reference answer, reference solution, human construction intent, and a reference witness. The construction intent specifies the type of object to be requested and the mathematical target that the object must realize. The reference witness provides an initial concrete construction before standardization.

Given these inputs, the LLM drafts three construction-side artifacts and one proof-side artifact: an item-specific construction instruction, a standardized reference construction, an informal verifier plan, and 0/1/6/7-style grading guidelines. The instruction defines the expected witness and its representation. The standardized reference construction is the payload that should pass the final verifier. The verifier plan lists the format checks and mathematical constraints that the verifier should enforce. The grading guidelines define proof progress for the original problem.

Human review checks whether the construction instruction preserves the original problem meaning, whether the standardized reference construction satisfies the intended target, and whether the grading guidelines reflect meaningful Olympiad-style proof progress. Items that fail this review are returned to intent revision or Stage 1 generation.

Category Description Extremal Problems Ask for maximum or minimum values under combinatorial con-

straints, often requiring both a sharp bound and a matching construction.

Existence and Construction Ask whether an object exists or require an explicit witness satisfying a global condition. Operations and Strategies Involve games, processes, transformations, invariants, or strategic choices. Graph Theory Focus on graph structures, paths, colorings, connectivity, or discrete networks. Counting Require enumerative or combinatorial counting arguments, often with careful case distinctions or structural decompositions.

Table 6: Primary category definitions used in ComBench.

Category Analysis Construction Total Extremal 7 28 35 Existence 17 7 24 Operations 16 7 23 Graph Theory 6 5 11 Counting 4 3 7 Total 50 50 100

- Table 7: Problem type and category breakdown. Analysis and Construction denote analysis-centric and construction-centric records.

#### B.2 Stage 2: Verifier Generation and Semantic Audit

- Stage 2 generates deterministic Python verifier code from the reviewed problem statement, construction instruction, reference construction, and verifier plan. The verifier reads only the raw witness payload and checks both representation validity and mathematical validity. It should reject malformed payloads, incomplete objects, objects satisfying only local constraints, and objects that miss the intended global combinatorial condition.

After verifier generation, we run an LLM-assisted semantic audit over the full annotation chain. The audit compares the original problem, reference answer, construction intent, construction instruction, reference construction, verifier plan, and verifier code. Its goal is to determine whether the executable check still matches the same mathematical claim as the original problem.

Records with step1 issue are returned to the specification stage, since the task definition itself is unreliable. Records with step2 issue are returned to verifier generation, since the construction target is acceptable but the executable checker is not.

#### B.3 Stage 3: Assembly and Executable Reference Check

Records passing semantic audit are assembled into the final JSONL format. Before acceptance, the standardized reference construction is passed to the verifier through the same raw-standard-input interface later used for model submissions. This executable reference check ensures that the stored construction, verifier code, and payload interface are mutually consistent.

The final human review focuses on verifier strictness, coverage, consistency, and diagnostic clarity. In particular, reviewers check that the verifier enforces global combinatorial constraints rather than merely validating syntax, rejects plausible but incomplete witnesses, and provides diagnostics that are useful for debugging the record. Additional verifier input and output details are provided in Appendix D.

Component Role Problem and reference answer Define the original mathematical target and final claim. Reference solution Supports rubric construction and proof-progress calibration. Human intent Specifies the desired witness type and construction target. Reference witness Provides a concrete construction to be standardized and checked.

Table 8: Stage 1 inputs.

Artifact Purpose Instruction Specifies what witness the model must output and in what

format. Reference construction Provides a valid payload under the declared representation. Verifier plan Lists the intended format and mathematical checks. Grading guidelines Defines the 0/1/6/7-point proof rubric.

Table 9: Stage 1 generated artifacts.

### C Record Schema and Quality Control

This appendix supports the record-design and quality-control claims in Section 3. ComBench records contain common fields for proof evaluation and additional fields for construction-centric evaluation.

#### C.1 Record Schema

The construction-centric fields define an executable interface rather than an additional naturallanguage hint. During evaluation, the model receives the construction instruction, submits one payload, and the verifier checks only that payload. The reference construction is used for dataset validation and is not provided to the response model.

#### C.2 Example Construction-Centric Record

This subsection shows the complete information content of one construction-centric record. The released dataset stores these fields as JSONL; here they are formatted across paragraphs and code blocks for readability. The example is based on IMO 2020 Problem 4 with the fixed instance n = 33.

Problem statement. There is an integer n > 1. There are n2 stations on a slope of a mountain, all at different altitudes. Each of two cable car companies, A and B, operates k cable cars; each cable car provides a transfer from one of the stations to a higher one, with no intermediate stops. The k cable cars of A have k different starting points and k different finishing points, and a cable car which starts higher also finishes higher. The same conditions hold for B. We say that two stations are linked by a company if one can start from the lower station and reach the higher one by using one or more cars of that company, with no other movements between stations allowed. Determine the smallest positive integer k for which one can guarantee that there are two stations that are linked by both companies.

Reference answer. The reference answer is k = n2 − n + 1.

Construction instruction. For the specific case n = 33, determine the maximum number of cable cars k that each company can operate without creating any pair of stations linked by both companies. Provide an explicit construction for this maximum case. The 332 = 1089 stations are numbered 1 to 1089 from lowest to highest altitude. The output must be a single Python expression evaluating to a tuple of two sets, (A lines, B lines). The sets A lines and B lines contain tuples (i, j), representing a cable car line from station i to station j for companies A and B respectively. The configuration must satisfy all validity constraints for cable-car operations: distinct start points, distinct end points, and the monotonicity condition that a cable car starting higher also finishes higher. Each company must have exactly k = 1056 cars. The model must put only the Python expression inside the construction block, without markdown, explanations, or surrounding tags other than <construct>...</construct>.

Audit target Main check Construction target Does it preserve the original mathematical claim? Instruction Is the requested witness precise and unambiguous? Reference construction Does it satisfy both the instruction and original target? Verifier plan Does it cover the necessary format and mathematical con-

straints? Verifier code Does it implement the intended checks without overfitting the reference construction?

Table 10: Semantic audit checklist.

Audit outcome Meaning pass The construction target, instruction, witness, plan, and verifier are seman-

tically aligned.

- step1 issue The construction intent, instruction, or reference construction changes,

weakens, or distorts the original target.

- step2 issue The verifier misses constraints, checks the wrong property, conflicts with

the instruction, or overfits the reference construction.

Table 11: Semantic audit outcomes.

#### Reference construction.

(

{(i, i + 1) for i in range(1, 1090) if i % 33 != 0}, {(i, i + 33) for i in range(1, 1057)}

)

Listing 1: Reference construction payload for the example record.

Verifier code. The verifier reads the raw payload, parses it as a Python expression, checks the cable-car constraints for both companies, computes transitive closures, and rejects the construction if any linked pair appears for both companies.

import sys def verify():

payload = sys.stdin.read().strip() if not payload:

print("Empty input") return

try: # Parse the payload directly as a Python expression. # eval is used because the reference construction uses comprehensions and range(). data = eval(payload)

except Exception as e: print(f"Failed to parse Python expression: {e}") return

if type(data) is not tuple or len(data) != 2: print("Parsed object is not a tuple of exactly two elements") return

A_lines, B_lines = data for name, lines in [("A_lines", A_lines), ("B_lines", B_lines)]:

if type(lines) is not set: print(f"{name} is not a set") return

for item in lines:

if type(item) is not tuple or len(item) != 2: print(f"An item in {name} is not a tuple of exactly two elements") return

u, v = item if type(u) is not int or type(v) is not int:

print(f"An item in {name} contains non-integers") return

- if len(A_lines) != 1056: print(f"A_lines has {len(A_lines)} elements, expected 1056") return

- if len(B_lines) != 1056:

Field Applies to Purpose problem id, source, year All Identify the original problem and support

source/year analysis. Problem statement All Provides the task given to the model. Reference answer All Specifies the target answer or claim. Reference solution All Supports rubric construction and proof judging

when available. Primary category All Supports category-level analysis. Problem type All Indicates analysis-centric or construction-centric

evaluation. Grading guidelines All Defines the 0/1/6/7-point proof rubric. Construction intent Constr. Specifies the desired witness type and target. Construction instruction Constr. Defines the model-facing witness request and out-

put format. Reference construction Constr. Provides the standardized witness used for executable reference checking. Verifier plan Constr. Describes intended format and mathematical checks. Verifier code Constr. Implements deterministic witness verification.

Table 12: ComBench record schema.

Field Example value id IMO-2020-P4-33 Source metadata IMO 2020 Problem 4, instantiated with n = 33. Problem type Construction-centric. Primary category Graph Theory. Reference answer k = n2 − n + 1. Verifier target Check that both companies have 1056 valid monotone cable-

car lines and no station pair is linked by both companies.

Table 13: Formatted metadata for an example construction-centric record.

print(f"B_lines has {len(B_lines)} elements, expected 1056") return

for name, lines in [("A_lines", A_lines), ("B_lines", B_lines)]:

- u_set = set()

- v_set = set() sorted_lines = sorted(list(lines), key=lambda x: x[0]) prev_v = -1

for u, v in sorted_lines:

if not (1 <= u <= 1089 and 1 <= v <= 1089): print(f"Station out of range in {name}: ({u}, {v})") return

- if u >= v: print(f"Invalid transfer in {name}: ({u}, {v}), must be u < v") return

- if u in u_set: print(f"Duplicate starting point {u} in {name}") return

- u_set.add(u)

if v in v_set: print(f"Duplicate finishing point {v} in {name}") return

- v_set.add(v)

- if v <= prev_v: print(f"Monotonicity violated in {name}: sequence of v is not strictly increasing.") return

prev_v = v

def get_transitive_closure(lines): adj = {u: v for u, v in lines} closure = set() for start_node in adj:

curr = start_node while curr in adj:

nxt = adj[curr] closure.add((start_node, nxt)) curr = nxt

return closure

- A_closure = get_transitive_closure(A_lines)

- B_closure = get_transitive_closure(B_lines)

intersection = A_closure.intersection(B_closure) if intersection:

example = list(intersection)[0] print(f"Intersection is not empty. Found linked pairs in both companies, e.g., {example}.") return

print("True") if __name__ == "__main__":

verify()

Listing 2: Complete verifier code for the example record.

Grading guidelines. Partial credit is assigned for the following forms of progress: (1) modeling each company’s system as a directed graph on the stations and correctly deducing from the orderpreserving and distinct start/endpoint conditions that each connected component is a path, possibly with isolated vertices; (2) deriving the component-count formula for one company, namely that with k edges on n2 vertices, the number of connected components is n2 − k; and (3) giving the correct extremal pigeonhole step, using n2 = (n − 1)(n + 1) + 1 and n − 1 components to conclude that some component has at least n + 1 vertices.

Almost-correct credit is assigned for solutions that prove sufficiency of k = n2 − n + 1 using the path-component structure and pigeonhole principle but make a small counting slip; or that present the full sufficiency proof and the idea of a counterexample for k = n2 − n but do not fully describe or verify the construction; or that are otherwise almost complete with non-negligible minor mistakes.

#### Reference solution. The answer is k = n2 − n + 1.

When k = n2 − n, a valid construction exists and shows that this value is not sufficient. For n = 33, the reference construction above realizes this counterexample.

To prove that k = n2 − n + 1 is sufficient, view A and B as graphs whose vertices are the n2 stations and whose edges are the cable cars. For each company, the condition that a cable car starting higher also finishes higher, together with distinct starting points and distinct finishing points, implies that the connected components are paths, possibly with isolated vertices; edge direction is irrelevant for this component-counting argument.

If k = n2 − n + 1, then each company has exactly n2 − k = n − 1 connected components. Since n2 = (n − 1)(n + 1) + 1, the pigeonhole principle implies that some component of A has at least n + 1 vertices. Since B has only n − 1 connected components, this component of A must contain two vertices that belong to the same component of B. These two stations are therefore linked by both companies.

#### C.3 Representative Witness Forms

Construction-centric records use different payload styles depending on the mathematical object being requested. Table 14 summarizes representative forms, and the following code blocks show the corresponding reference constructions.

Record Witness form Payload meaning USAMO-2024-P4 Explicit binary matrix A fully enumerated 0/1 construction. USAMO-2020-P4-10 Ordered-pair list A sequence of concrete ordered pairs. TSTST-2024-P9-20 Comprehension table A large table represented by a compact rule. IMOSLC-2023-P4-150 Semantic dictionary Structured fields with mathematical roles. RMO-2019-P15 Rule-based witness Decision rule + supporting partitions.

Table 14: Representative witness payload forms in construction-centric records.

#### Explicit binary matrix.

- [[0, 0, 0, 0],

- [0, 0, 0, 1],

- [0, 0, 1, 1],

- [0, 1, 1, 1],

- [1, 1, 1, 1]] Listing 3: Explicit binary-matrix witness from USAMO-2024-P4.

#### Explicit ordered-pair list.

- [[1, 0], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1]]

Listing 4: Ordered-pair witness from USAMO-2020-P4-10.

#### Comprehension-defined table.

[[((j - i) % 20) * 20 + i + 1 for j in range(20)] for i in range(20)]

Listing 5: Comprehension-defined table from TSTST-2024-P9-20.

#### Semantic dictionary.

{

"endpoints": [150] + [ val for i in range(2, 151) for val in ((i-1)*150 + i - 1, i*150)

], "rows": [[1]] + [[2*i - 1, 2*i - 2]

for i in range(2, 151)] }

Listing 6: Dictionary-structured witness from IMOSLC-2023-P4-150.

#### Rule-based witness.

{

"rule": "lambda team: sum(1 for x in team if x in {1, 2, 3}) % 2 == 1", "partition_1": [

[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]

], "partition_3": [

- [1, 4, 5, 6, 7, 8],

- [2, 9, 10, 11, 12, 13],

- [3, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]

] }

Listing 7: Rule-based witness from RMO-2019-P15.

#### C.4 Quality-Control Checks

For construction-centric records, the most important risk is target drift: a construction task may accidentally ask for a weaker, stronger, or different object than the original problem requires. We therefore check the construction instruction and verifier against the original problem and reference answer, rather than only against the reference construction.

Verifier adequacy is checked at two levels. At the interface level, the verifier must accept the stored reference construction through the same raw-standard-input interface used for model submissions. At

Check type Required condition Target fidelity The problem statement, reference answer, reference solution, grading

guideline, construction instruction, and reference construction must target the same mathematical claim.

Verifier adequacy The reference construction must be parseable and accepted by the verifier, and the verifier must enforce mathematical constraints rather than only format-level properties.

Dataset integrity Source identifiers, problem types, categories, record counts, and final JSONL files must be mutually consistent.

Table 15: Quality-control checks.

the mathematical level, it must enforce the global constraints specified by the construction instruction, not merely validate syntax or local consistency. Records that fail these checks are returned to the relevant stage of the annotation pipeline.

Dataset integrity checks ensure that the paper statistics match the evaluation files. We cross-check problem identifiers, source labels, year labels, problem categories, problem types, and constructionfield availability against the final JSONL records used in the experiments.

#### C.5 Human Audit Checklist

The final human audit is applied after automatic generation and executable reference checking. Reviewers verify that the construction task preserves the original mathematical target, that the verifier checks the intended constraints rather than superficial syntax, and that the record metadata is consistent with the released dataset. Table 16 summarizes the questions used in this review.

Audit item Question checked by reviewers Problem meaning Does the construction task preserve the original mathemat-

ical claim? Target strength Is the requested witness neither weaker nor stronger than

intended? Output contract Is the required representation explicit and unambiguous? Reference construction Does the stored construction satisfy the instruction and the

original target? Verifier coverage Does the verifier check all essential mathematical constraints? Verifier robustness Does the verifier reject plausible invalid or incomplete witnesses? Overfitting risk Does the verifier check a class of valid objects rather than the stored reference only? Diagnostics Are failure messages sufficient for debugging annotation errors? Metadata consistency Do source, year, category, type, and construction fields match the released record?

Table 16: Human audit checklist.

### D Executable Verifier Details

This appendix describes the executable verifier used for construction-centric records. Here, a verifier refers to the deterministic Python code associated with an individual record. It is distinct from the rubric-guided proof judge: the proof judge evaluates natural-language reasoning, while the executable verifier checks only the submitted construction payload.

#### D.1 Input and Output Contract

For construction-centric records, the model response must contain exactly one payload inside a <construct>...</construct> block. The evaluation pipeline extracts the interior of this block and passes it as raw standard input to the item-specific Python verifier. The verifier does not read the proof text, the problem statement, or any surrounding explanation in the model response.

Component Evaluator Role Proof solution Proof judge Scores Rigorous Proof Reasoning. Construct payload

Code verifier Checks Constructive Realization.

Reference construction

Code verifier Validates the record through executable reference checking.

Table 17: Separation between proof judging and executable construction verification.

The verifier returns a binary construction result. A construction passes if the verifier prints True; otherwise it fails. Missing, duplicated, or malformed <construct> blocks receive construction score 0 before verifier execution. During dataset validation, the stored reference construction is passed through the same raw-standard-input interface used for model submissions.

#### D.2 Verifier Scope

The executable verifier is not a general theorem prover and does not evaluate the natural-language proof. Its scope is limited to the construction target formalized by the item-specific instruction. Within this scope, the verifier checks three kinds of conditions.

Responsibility Typical checks Format validity The payload can be parsed and has the declared type, di-

mensions, length, symbols, or value ranges.

Structural constraints The submitted object satisfies representation-level constraints such as distinctness, legality of edges, valid coordinates, or complete assignments.

Mathematical correctness The object satisfies the global combinatorial properties required by the construction instruction, such as coverage, extremality, non-conflict, invariance, or counterexample validity.

Table 18: Verifier responsibilities.

This design intentionally separates proof quality from witness validity. A model may receive partial proof credit while failing the verifier, and a valid witness does not by itself guarantee a complete proof. The combined score for construction-centric records is defined in Section 3.5.

#### D.3 Construction Object Types

Construction-centric records use several recurring witness representations. Table 19 summarizes representative object types and the corresponding verifier checks.

### E Evaluation Protocol

This appendix expands the evaluation configuration in Section 4. It specifies the response-model sampling setup, prompt templates, parsing rules, failure handling, and metric definitions used to reproduce the reported results.

Object type Typical verifier checks Matrices or grids Shape, allowed entries, row/column constraints, coverage,

adjacency, or path-count constraints.

Colorings Domain coverage, color validity, forbidden monochromatic patterns, adjacency constraints, or extremal requirements.

Graphs Vertex and edge validity, degree constraints, connectivity, subgraph avoidance, paths, cycles, or coloring properties. Selected cell sets Coordinate validity, distinctness, board constraints, cover-

age, domination, or packing properties. Permutations or sequences Length, element set, uniqueness, recurrence conditions, monotonicity, or extremal objective values. Set systems Membership validity, intersection constraints, covering properties, and cardinality requirements. Strategies Legal move sequence, game-state transition validity, terminal condition, and claimed guarantee. Counterexamples Format validity and direct falsification of the target statement under the specified parameters.

Table 19: Representative construction object types.

#### E.1 Response Models and Sampling

The response models are the models reported in Table 1. For each record, we sample four independent solutions using temperature 0.6 and model-specific maximum token limits. All response models are evaluated with the same downstream protocol.

Proof evaluation uses a unified rubric-guided Gemini-3.1-Pro-Preview-Thinking judge for all response models. The judge receives the problem statement, the reference solution when available, and the problem-specific grading guidelines. Construction evaluation uses the item-specific deterministic Python verifier associated with each construction-centric record.

#### E.2 Prompt Templates

The prompt format depends on the record type. The following templates summarize the exact structure used in evaluation; bracketed fields are filled with the corresponding record-specific content.

#### Analysis-centric records.

You are solving one IMO-style math problem. ## Problem """ Please answer the problem adhering to the following rules:

- 1. Please use LaTeX format to represent the variables and formulas used in the solution process and results.

- 2. Please provide a complete and explicit solution process in the response.

- 3. In the end of the response:

- - If the problem has final answer(s), put them in \boxed{}.

- - If the problem requires multiple answers, list them in order, each in a separate \boxed{}.

- - If the problem is a proof or does not require a final numerical answer, provide a complete and rigorous proof, and do not use \boxed{}.

Problem: [Original problem statement] """

You must answer using exactly the following output structure: ## Solution [Your complete solution, including the final answer if applicable.] Rules:

- - Output the solution heading exactly as written above.

- - Do not use ‘## Solution to Question 1‘ or ‘## Solution to Question 2‘.

- - Do not use any <construct>...</construct> block.

#### Construction-centric records.

You are solving one IMO-style math problem with an additional construction deliverable.

- ## Question 1 """ [Original problem statement] """

- ## Question 2 """ [Item-specific construction instruction] """ You must answer using exactly the following output structure:

- ## Solution to Question 1 [Your proof and final answer to Question 1 only.]

- ## Solution to Question 2 [Your response to Question 2. This section must include exactly one construction block.] <construct> [Construction payload] </construct> Rules:

- - Output the two solution headings exactly as written above.

- - Do not put any <construct> block in Solution to Question 1.

- - Put exactly one <construct>...</construct> block in Solution to Question 2.

- - Inside <construct>...</construct>, put ONLY the construction payload and nothing else.

- - Do not use code fences.

The proof part evaluates Rigorous Proof Reasoning. The construction payload evaluates Constructive Realization.

Rubric-guided proof evaluation. All response models are graded with the same proof-evaluation template. The judge receives the problem statement, the ground-truth solution when available, the item-specific grading guidelines, and the proposed solution. It must return exactly one parsed score from the four-level rubric:

You are an expert grader for the International Mathematics Olympiad (IMO). Evaluate the proposed solution strictly and rigorously.

Inputs:

- - Problem Statement: [Original problem statement]

- - Ground-Truth Solution: [Reference solution, if available]

- - Specific Grading Guidelines: [Problem-specific 0/1/6/7 rubric]

- - Proposed Solution: [Model response]

Output the final score using exactly one of: <points>7 out of 7</points> <points>6 out of 7</points> <points>1 out of 7</points> <points>0 out of 7</points>

We do not use final-answer equivalence grading as the main proof score; the reported proof scores use the rubric-guided proof judge.

#### E.3 Parsing and Failure Handling

For analysis-centric records, the parser extracts the model’s proof solution and sends it to the proof judge. For construction-centric records, the parser extracts the proof section and the unique <construct>...</construct> block. Only the block interior is passed to the executable verifier.

If a construction-centric response has no construction block, multiple construction blocks, or a malformed block, its construction score is 0 without verifier execution. The proof text is still judged when it can be parsed. Missing or non-parseable outputs remain in the denominator for aggregate metrics. For SU-01, only outputs satisfying the evaluation format are treated as valid parsed outputs; missing or non-parseable outputs are counted under the same full-denominator accounting.

#### E.4 Scoring and Metrics

The proof judge returns one of 0,1,6,7 points out of 7. The construction verifier returns a binary result: 1 if the submitted payload is accepted and 0 otherwise. For construction-centric records, the aggregate score uses the verifier-gated rule defined in Section 3.5.

The main text reports two metrics. Avg. is the average normalized score over all generated solutions. Best@4 first takes the highest score among four sampled solutions for each record and then averages these per-record best scores. Additional reports may include Pass@4, the percentage of records solved in at least one of four sampled solutions; Passˆ4, the percentage of records solved in all four sampled solutions; and construction pass rate, the per-sample percentage of construction payloads accepted by the verifier.

### F Additional Experimental Results

This appendix provides supplementary results for the main-result and subset-analysis claims in Section 4. The main text reports the most compact tables and figures; here we include additional category-level, problem-level, and repeated-sampling summaries.

#### F.1 Category-Level Summary

- Table 20 summarizes average category performance across the evaluated models. The values use the same scoring rule as Table 1: analysis-centric records use proof scores, while construction-centric records use verifier-gated scores.

Category # records Mean Avg. Best model Best Avg. Extremal Problems 35 34.1 GPT-5.5 69.5 Existence and Construction 24 22.9 GPT-5.5 48.1 Operations and Strategies 23 31.6 GPT-5.5 65.5 Graph Theory 11 59.6 Gemini-3.1-Pro 75.3 Counting 7 59.4 GPT-5.5 94.4

Table 20: Supplementary category-level performance summary.

The category-level averages reinforce the trend shown in Figure 5. Existence and Construction is the lowest-scoring category on average, even though it is not the largest category. In contrast, Counting and Graph Theory have substantially higher average scores, suggesting that current models are more reliable on locally structured enumeration or graph-structural arguments than on global existence and witness-realization problems.

#### F.2 Problem-Level Difficulty

Problem-level success counts are computed over four sampled solutions per model. For analysiscentric records, a sampled solution is counted as solved when the proof score is 7. For constructioncentric records, strict success requires both proof score 7 and construction score 1.

For analysis-centric records, the easiest problems by total full-proof samples include IMO2008 5 and TSTST2023 4, each with 32 full-score samples across the evaluated models. The hardest analysiscentric problems include IMO2024 5, IMO2018 3, IMO2017 3, IMO2013 2, IMO2012 3, IMO2009 6, USAMO2025 6, and TSTST2022 5, each with zero full-score samples.

For construction-centric records, the easiest strict problems include RMO-2019-P15 with 31 strict successful samples, APMO-2012-P2 and CSMO-2018-P14-789 with 27, and USAMO-2000-P4 and CSMO-2004-P7 with 25. The hardest strict construction-centric problems include IMO-2025-P6-23, IMO-2022-P6, IMO-2004-P3, USAMO-2022-P6-202, USAMO-2009-P3-20-21, CTST-2022-P7-518, RMO-2018-P4-81, and RMO-2009-P7, each with zero strict successful samples.

#### F.3 Construction-Centric Stability

The construction-centric subset shows a large gap between one-success and all-success repeatedsampling metrics. For example, GPT-5.5 reaches 68.0% verifier-gated Pass@4 but only 42.0% Passˆ4. This indicates that many construction-centric records can be solved by at least one sampled solution,

but are not solved consistently across all four samples. The gap supports the use of both average scores and best-of-four scores in the main text: the former measures typical performance, while the latter measures whether the model can find a correct solution under repeated sampling.

### G Proof Error Taxonomy

This appendix supports the error analysis in Section 4. The taxonomy applies only to proof-side critiques, and therefore diagnoses failures in Rigorous Proof Reasoning. Failures of Constructive Realization are analyzed separately through construction pass rates, verifier-gated scores, and the case studies in Appendix J.

#### G.1 Scope and Annotation Rule

We analyze every below-full-credit proof sample, including proofs scored 0, 1, or 6, as well as proof components that are unscored because of parsing or evaluation failure. Each sample is assigned exactly one primary reason. Primary reasons are mutually exclusive by construction, while secondary tags may overlap. For the paper-facing distribution, mathematical categories below the global 2% frequency threshold are merged into Other Mathematical Error. Format or Unscored is kept separate because it is not a mathematical proof error.

#### G.2 Taxonomy Definitions

Table 21 defines the proof-error categories. The examples are schematic: they illustrate the kind of mathematical failure covered by each category, rather than quoting a particular model response.

Error type Meaning Example pattern Missing Core Mechanism The solution misses the central combinatorial

Gives local observations but never identifies the invariant needed to control the process.

idea, such as an invariant, extremal choice, bijection, or construction principle.

Wrong Mathematical Target The proof solves a different statement, optimizes the wrong quantity, or uses an incorrect final value.

Proves a lower bound for n − 1 when the problem asks for the exact maximum.

False Lemma The argument relies on a false claim or an unsupported theorem-like assertion.

Claims that every maximal configuration is automatically balanced, without proof and in fact falsely.

Faulty Induction The induction base case, transition, preserved invariant, or parameter is incorrect.

Assumes the induction step after deleting an object, but the remaining structure no longer satisfies the hypothesis.

Unjustified Leap A crucial implication is asserted without enough mathematical justification.

States that a greedy choice can always be extended, but gives no exchange or obstruction argument.

Incomplete Case Analysis Necessary cases are omitted or treated asymmetrically.

Handles one parity case and concludes the result for all parities.

Proof Closure Gap The response has meaningful progress but fails to close the final implication, optimality proof, or equality case.

Proves an upper bound but never supplies the matching construction.

Insufficient Mathematical Progress The response contains only superficial reformulation or weak observations.

Restates the problem and checks small cases without a route to a proof.

Format or Unscored The proof component cannot be scored because of missing content, parsing failure, or evaluator failure.

The required solution section is absent or non-parseable.

Other Mathematical Error Low-frequency mathematical errors not covered by the categories above.

A specialized counting or notation error that does not fit the main categories.

Table 21: Proof-error taxonomy used for below-full-credit proof samples.

#### G.3 Global Distribution

- Table 22 reports the primary reason distribution after merging low-frequency mathematical categories into Other Mathematical Error. The largest category is Missing Core Mechanism, followed by Wrong Mathematical Target. This pattern indicates that most failures are substantive mathematical failures rather than formatting failures.

Primary reason Count Share Missing Core Mechanism 1213 41.2 Wrong Mathematical Target 588 20.0 Format or Unscored 324 11.0 False Lemma 290 9.8 Faulty Induction 202 6.9 Unjustified Leap 144 4.9 Incomplete Case Analysis 114 3.9 Other Mathematical Error 70 2.4

Table 22: Primary proof-error distribution. Shares are percentages.

#### G.4 Model-Level Error Profiles

- Table 23 reports how many below-full-credit proof samples each model produces. Table 24 further breaks these samples down by primary reason. The counts show that stronger models still fail mainly by missing the core mechanism or targeting the wrong mathematical claim, while models with many format or parsing failures have a different error profile.

Model Below-full samples Rate GPT-5.5 174/400 43.5 Gemini-3.1-Pro 203/400 50.8 Kimi-K2.6 245/400 61.2 DeepSeek-V4-Pro 271/400 67.8 SU-01 319/400 79.8 Qwen3.6-Max 333/400 83.2 GLM-5.1 336/400 84.0 Nemotron 349/400 87.2 Qwen3.6-35B 352/400 88.0 Gemma-4-31B-IT 363/400 90.8

Table 23: Below-full-credit proof samples by model.

Model MCM WMT FL FI UL ICA Fmt Other GPT-5.5 85 32 23 14 8 9 0 3 Gemini-3.1-Pro 120 29 17 8 13 9 4 3 Kimi-K2.6 108 44 43 16 10 16 5 3 DeepSeek-V4-Pro 107 42 37 41 10 18 11 5 SU-01 49 31 21 19 3 7 187 2 Qwen3.6-Max 139 58 34 26 18 13 38 7 GLM-5.1 143 88 30 17 13 11 26 8 Nemotron 122 106 52 26 22 12 4 5 Qwen3.6-35B 160 83 14 20 24 11 28 12 Gemma-4-31B-IT 180 75 19 15 23 8 21 22

- Table 24: Model-level primary proof-error counts. Abbreviations: MCM = Missing Core Mechanism; WMT = Wrong Mathematical Target; FL = False Lemma; FI = Faulty Induction; UL = Unjustified Leap; ICA = Incomplete Case Analysis; Fmt = Format or Unscored.

### H Overlap with IMO-Bench

This appendix documents the overlap analysis between ComBench and IMO-Bench. The purpose is benchmark-coverage transparency: both benchmarks draw from public Olympiad-style sources, so we report how much of ComBench overlaps with IMO-Bench and how the overlap was determined.

#### H.1 Matching Protocol

We compare all 100 ComBench problems against IMO-Bench using three complementary signals. First, we check source metadata, including competition name, year, and problem number when available. Second, we run problem-statement similarity search to identify candidates whose wording is close even when source identifiers differ. Third, we manually inspect high-similarity candidates to distinguish true overlap from problems that share only common combinatorial terminology.

A ComBench problem is counted as overlapping if it is an exact source match or a substantially modified version of an IMO-Bench item. Problems with similar themes but different mathematical targets are not counted as overlapping.

#### H.2 Overlap Summary

Judgment category Count Source-based duplicate 6 Text-similarity duplicate 6 Manual-review duplicate 2 Inspected but not duplicate 4 No high-confidence match 82 Total 100

Table 25: Overlap judgments against IMO-Bench.

The final overlap rate is 14/100. The remaining 86 problems have no high-confidence match after source-based matching, text-similarity search, and manual inspection of ambiguous candidates. This indicates that ComBench provides largely complementary coverage to IMO-Bench while documenting the overlapping subset at the metadata level.

#### H.3 Released Metadata and Filtering

We do not reproduce the full matched-problem list in the appendix. Instead, the released metadata will include, for each inspected ComBench problem, the best-matching IMO-Bench entry when applicable, the match type, the similarity score, and the manual decision note. This allows users to filter or analyze overlapping records without using appendix space for a problem-level listing.

The overlap analysis is used to characterize benchmark coverage rather than to exclude records. Overlapping source problems are kept because ComBench adds construction-centric instructions, reference witnesses, and deterministic verifier-gated scoring; these record-level evaluation targets may differ from those in IMO-Bench even when the source problem overlaps.

### I Manual Audit of Automatic Proof Evaluation

This appendix describes the sampled manual audit used to check the reliability of the automatic proof judge. The goal is not to replace expert mathematical review, but to verify that the rubric-guided proof judge produces reasonable scores on a representative subset of model outputs.

#### I.1 Audit Sampling Protocol

We sample proof-evaluation results from the 10 response models used in the main experiments. For each model, we randomly select five proof samples from the set of scored proof evaluations, excluding outputs whose proof score is null or unscored because of parsing or evaluation failure. This yields 50 audited proof samples in total.

The sampling covers both analysis-centric and construction-centric records. The audit only checks the proof-judge decision: whether the assigned proof score is consistent with the problem statement, reference solution when available, and problem-specific grading guidelines. Construction-verifier

decisions are not included in this audit, because they are deterministic executable checks and are analyzed separately through construction pass rates and verifier-gated scores.

- I.2 Audit Decision Labels Each sampled proof evaluation is assigned one of three manual-audit labels:

- • Accepted: the automatic proof score is consistent with the rubric and the judge rationale is mathematically reasonable.
- • Minor disagreement: the score is debatable near a rubric boundary, but the decision does not change the qualitative interpretation of the result.
- • Rejected: the automatic proof score is clearly inconsistent with the rubric or misses a major mathematical issue.

We report the strict acceptance rate as the fraction of audited samples labeled Accepted. We also report an accepted-or-minor rate, which treats boundary disagreements as acceptable for aggregate trend analysis.

- I.3 Audit Summary

Table 26 summarizes the audit results. The released audit metadata contains the sampled model output, the proof-judge rationale, the assigned proof score, and the manual-audit decision for each sampled case. In this sample, 45 of 50 automatic proof-judge decisions are accepted by the manual audit. Four additional cases are marked as minor boundary disagreements, and one case is rejected.

Model Samples Accepted Minor Rejected Accept rate (%) GPT-5.5 5 4 1 0 80.0 Gemini-3.1-Pro 5 4 1 0 80.0 Kimi-K2.6 5 3 1 1 60.0 DeepSeek-V4-Pro 5 5 0 0 100.0 SU-01 5 4 1 0 80.0 Qwen3.6-Max 5 5 0 0 100.0 GLM-5.1 5 5 0 0 100.0 Nemotron 5 5 0 0 100.0 Qwen3.6-35B 5 5 0 0 100.0 Gemma-4-31B-IT 5 5 0 0 100.0 Total 50 45 4 1 90.0

Table 26: Manual audit summary for sampled automatic proof evaluations.

This audit is intended as a reliability check for large-scale automatic proof grading. The strict acceptance rate is 90.0%, and the accepted-or-minor rate is 98.0%. These results support the use of the automatic judge for aggregate benchmarking, while also showing that boundary disagreements and occasional grading errors remain possible. The audit is limited to sampled proof evaluations and should not be interpreted as a guarantee that the automatic judge is error-free on every record.

### J Case Studies for Verifier-Gated Scoring

This appendix gives representative construction-centric case studies illustrating why verifier-gated scoring is used. The cases cover different proof scores, construction outcomes, and response models.

#### J.1 Case 1: Proof Score 6 and Construction Score 0

Item Description Problem IMO-2022-P6 Run GPT-5.5, run 1 Proof score 6/7 Construction result 0/1 Verifier-gated score 1/7

Table 27: Case study with proof score 6 and failed construction.

#### Problem IMO-2022-P6

Let n be a positive integer. A Nordic square is an n × n board containing all the integers from 1 to n2 so that each cell contains exactly one number. An uphill path is a sequence of one or more cells such that:

- 1. the first cell in the sequence is a valley, meaning the number written is less than all its orthogonal neighbors;
- 2. each subsequent cell in the sequence is orthogonally adjacent to the previous cell; and
- 3. the numbers written in the cells in the sequence are in increasing order. Find, as a function of n, the smallest possible total number of uphill paths in a Nordic square.

#### Relevant grading guidelines.

- • Partial: Establish the lower-bound idea by associating adjacent pairs with uphill paths, correctly count the 2n(n − 1) adjacent pairs in the grid, and add at least one length-one uphill path to obtain 2n2 − 2n + 1.
- • Almost: Give the full lower bound 2n2 − 2n + 1 and an intended matching construction using a set or tree structure, but leave a gap in justifying that the required construction exists for all n, or leave a gap in the cropping/adaptation argument.

#### Full GPT-5.5 generation.

Solution to Question 1. Let the n2 cells of the board be regarded as the vertices of the usual n × n grid graph Gn. Thus two vertices are adjacent if and only if the corresponding cells are orthogonally adjacent. The number of edges of Gn is

|E(Gn)| = n(n − 1) + n(n − 1) = 2n(n − 1). Put an orientation on every edge from the smaller number to the larger number. Since all numbers are distinct, this gives an acyclic orientation of Gn. A valley is precisely a source of this orientation.

For a cell v, let p(v) be the number of uphill paths ending at v. If v is a valley, then p(v) = 1, corresponding to the path consisting only of v. If v is not a valley, then every uphill path ending at v is obtained uniquely by taking an uphill path ending at a smaller adjacent cell and then appending v. Hence

p(v) =

p(u),

u→v

where the sum is over all directed edges entering v. The total number of uphill paths is therefore

##### P =

p(v).

v

Since every cell is reachable by following decreasing numbers until one reaches a valley, we have p(v) ≥ 1 for every v. Moreover, every directed edge u → v contributes at least one uphill path

ending at v, namely any uphill path ending at u extended by the edge u → v. Thus

P ≥ 1 + |E(Gn)| = 1 + 2n(n − 1). It remains to show that this lower bound is attainable.

Choose an independent set S of cells such that Gn−S is a tree. Such a set exists for every rectangular grid; one may construct it by the standard zig-zag deletion pattern, deleting one cell from each elementary square in such a way that no two deleted cells are adjacent and the remaining graph is connected and acyclic.

Orient the tree Gn − S away from one chosen root cell r. Then every cell of Gn − S except r has exactly one incoming edge inside the tree. Finally orient every edge incident with a cell of S toward that cell. Because S is independent, no edge joins two cells of S. Thus every cell of S is a sink.

Now assign the numbers 1,2,...,n2 in any order compatible with this acyclic orientation. The unique valley is the root r. Every cell of Gn − S has exactly one smaller neighbor, namely its parent in the rooted tree, while every cell of S has only larger incoming contributions from cells with exactly one uphill path reaching them.

Therefore every edge contributes exactly one new uphill path, and together with the trivial path at the unique valley, the total number of uphill paths is

1 + |E(Gn)| = 1 + 2n(n − 1). Hence the smallest possible total number of uphill paths is

|2n(n − 1) + 1|
|---|

.

Scoring interpretation. The proof contains the main lower-bound argument and an intended construction mechanism, so the proof-side score of 6/7 is consistent with the almost-complete guideline. However, the constructive part of the proof remains high-level: it asserts a “standard zigzag deletion pattern” rather than fully specifying and verifying the required object. The constructioncentric response must provide a complete witness in the prescribed representation, and it does not receive construction credit. The verifier-gated demotion from 6 to 1 is therefore appropriate: the response shows substantial Rigorous Proof Reasoning, but it does not realize the construction as a complete, mechanically checkable object.

#### J.2 Case 2: Proof Score 7 and Construction Score 0

Item Description Problem USAMO-2009-P3-20-21 Run GPT-5.5, run 1 Proof score 7/7 Construction result 0/1 Verifier-gated score 6/7

Table 28: Case study with proof score 7 and failed construction.

#### Problem USAMO-2009-P3

We define a chessboard polygon to be a simple polygon whose sides are situated along lines of the form x = a or y = b, where a and b are integers. These lines divide the interior into unit squares, which are shaded alternately gray and white so that adjacent squares have different colors. To tile a chessboard polygon by dominoes is to exactly cover the polygon by non-overlapping 1×2 rectangles. Finally, a tasteful tiling is one which avoids the two forbidden configurations of dominoes and colors.

Prove that (a) if a chessboard polygon can be tiled by dominoes, then it can be done so tastefully, and (b) such a tasteful tiling is unique.

#### Relevant grading guidelines.

- • Partial: For existence, use the lower-left-square idea and justify the key local switch in a 2 × 2 block. For uniqueness, overlay two tilings and reduce the argument to ruling out alternating cycles under tastefulness.
- • Almost: Give an essentially complete induction proof of existence with a local gap, or prove uniqueness by an alternating-loop method while leaving a localized topological gap.
- • Full: A complete proof of existence and uniqueness without such gaps receives full proof credit.

Full GPT-5.5 generation. Solution to Question 1. We prove the result by using the standard height-function method for domino tilings. Let P be a chessboard polygon which is tileable by dominoes. Since P is a simple polygon, the union of its unit squares is simply connected.

Color the unit squares alternately white and black. For every edge of the square grid contained in P, orient it so that a white square lies on its left, equivalently a black square lies on its right. For a boundary edge, this convention uses the unique square of P adjacent to that edge.

Given a domino tiling T, define a height function

hT : {lattice vertices of P} → Z as follows. Fix one boundary vertex v0 and set

hT(v0) = 0. If e = uv is an oriented grid edge from u to v, write e ∈ T when e is crossed by a domino of T, and define

1, e ∈/ T, −3, e ∈ T.

hT(v) − hT(u) =

Around every unit square, exactly one edge is crossed by the domino covering that square, so the total change of height around that square is

1 + 1 + 1 − 3 = 0.

Since P is simply connected, this implies that hT is well-defined.

Conversely, any function h on the lattice vertices satisfying the boundary values and such that along every oriented edge the height difference is either 1 or −3 determines a domino tiling: put a domino across precisely those edges where the height difference is −3. Around every unit square, the sum of the four height changes is 0, and since each height difference is congruent to 1 (mod 4), exactly one edge of each square has height difference −3. Hence each square is paired with exactly one adjacent square, producing a domino tiling.

Thus domino tilings of P are equivalent to height functions satisfying these conditions.

Now consider a 2×2 block. A flip replaces two parallel dominoes by the other two parallel dominoes. The four outer vertices of the block have the same heights before and after the flip; only the central vertex changes, by 4.

A direct check shows the following:

- • If the 2 × 2 block has top-left square white and is tiled by two vertical dominoes, namely

Wk Bt Bk Wt

,

then the central height is lower by 4 than after flipping the block to two horizontal dominoes.

- • If the 2 × 2 block has top-left square black and is tiled by two horizontal dominoes, namely

Bk Wk Wt Bt

,

then again the central height is lower by 4 than after flipping the block to two vertical dominoes.

Therefore the two forbidden configurations are exactly the configurations on which one can perform an upward flip, meaning a flip which increases the height at the central vertex by 4.

We now show that there exists exactly one height function admitting no upward flip. Let H be the finite set of all height functions arising from domino tilings of P. Since P is tileable, H ̸= ∅.

First observe that if h1,h2 ∈ H, then

##### h(v) = max{h1(v),h2(v)}

also belongs to H. Indeed, along an oriented edge uv, both differences

h1(v) − h1(u), h2(v) − h2(u) belong to {1,−3}. Hence

−3 ≤ h(v) − h(u) ≤ 1.

Also all height differences are congruent to 1 (mod 4). Therefore

##### h(v) − h(u) ∈ {1,−3}.

So h is again a valid height function. It follows that the pointwise maximum

H(v) = max

h(v)

h∈H

is itself a height function. This H is the greatest element of H.

We claim that H has no upward flip. If an upward flip were possible, it would increase the height of one vertex by 4 while keeping all edge differences valid, producing another height function H′ ∈ H with H′(v) > H(v) at that vertex, contradicting the definition of H as the pointwise maximum.

Thus H corresponds to a tasteful tiling. This proves existence. It remains to prove uniqueness. Suppose h ∈ H has no upward flip. Since H is greatest, we have H(v) ≥ h(v) for every vertex v. Suppose for contradiction that H ̸= h. Then the set

D = {v : H(v) > h(v)} is nonempty. Boundary heights are fixed, so D contains only interior vertices. Let d(v) = H(v) − h(v). All height functions have the same values modulo 4, so every positive d(v) is at least 4.

We show that some vertex of D permits an upward flip for h. Suppose not. Pick v ∈ D. Since v is not upward-flippable, there is an incident edge preventing the increase of h(v) by 4. More precisely, either:

- 1. there is an oriented edge v → w with h(w) − h(v) = −3, or
- 2. there is an oriented edge w → v with h(v) − h(w) = 1.

In either case, comparing with the corresponding valid edge difference for H shows that d(w) ≥ d(v). Moreover, in both cases, h(w) < h(v). Thus from v we can move to another vertex w ∈ D with h(w) < h(v). Repeating this process gives an infinite sequence of vertices in the finite set D along which the integer h-value strictly decreases, impossible.

Hence some vertex of D must be upward-flippable, contradicting the assumption that h has no upward flip. Therefore h = H.

So there is exactly one height function with no upward flip, and therefore exactly one tasteful tiling. Thus, if a chessboard polygon can be tiled by dominoes, then it can be tiled tastefully, and that tasteful tiling is unique.

Scoring interpretation. The proof gives a complete existence-and-uniqueness argument for the original theorem using height functions, so full proof credit is appropriate under the rubric. However, the construction-centric record asks for an explicit witness payload, not only a proof that such a tiling exists. Since the submitted construction is not accepted as a complete mechanically checkable witness, full construction-centric credit would overstate the model’s Constructive Realization. The verifier-gated demotion from 7 to 6 is therefore appropriate: the proof is complete, but the requested construction object is not successfully realized.

#### J.3 Case 3: Proof Score 7 and Construction Score 1

Item Description Problem IMO-2020-P4-33 Run Gemini-3.1-Pro, run 1 Proof score 7/7 Construction result 1/1 Verifier-gated score 7/7

Table 29: Case study with full proof score and accepted construction.

#### Problem IMO-2020-P4

There is an integer n > 1. There are n2 stations on a slope of a mountain, all at different altitudes. Each of two cable car companies, A and B, operates k cable cars; each cable car provides a transfer from one station to a higher one. The k cable cars of each company have k different starting points and k different finishing points, and a cable car which starts higher also finishes higher. Two stations are linked by a company if one can start from the lower station and reach the higher one by using one or more cars of that company. Determine the smallest positive integer k for which one can guarantee that two stations are linked by both companies.

#### Relevant grading guidelines.

- • Partial: Model each company as a directed graph whose connected components are paths, and derive that a company with k cars has n2 − k path components.
- • Almost: Prove the sufficiency of k = n2−n+1 using the path-component structure and pigeonhole principle, with only a minor counting slip or a lightly underspecified counterexample.
- • Full: Prove both sufficiency and sharpness by giving a valid construction for k = n2 − n with no common linked pair.

Full Gemini-3.1-Pro generation. Solution to Question 1. To find the smallest positive integer k that guarantees two stations are linked by both companies, we can model the cable car network as a directed graph.

Let the n2 stations be represented by vertices V = {1,2,...,n2}, ordered by their altitudes such that 1 is the lowest and n2 is the highest. For each company, the k cable cars represent k directed edges. Since each cable car goes from a lower station to a higher one, there are no cycles in the graph. Furthermore, because each company’s k cable cars have k distinct starting points and k distinct finishing points, every vertex in the graph has out-degree at most 1 and in-degree at most 1.

Consequently, the graph for Company A is a collection of disjoint directed paths. Since there are n2 vertices and k edges, the number of connected components for Company A is exactly n2 − k. The same logic applies to Company B, which also has exactly n2 − k connected components.

Two stations are linked by a company if and only if they belong to the same connected component of that company. To avoid having any pair of stations linked by both companies, no two stations can belong to the same component in Company A and the same component in Company B.

Let c = n2−k. Company A partitions the stations into c components, and Company B partitions them into c components. The maximum number of stations such that no two share both an A-component and a B-component is the total number of pairs of components, which is c × c = c2. Because there are n2 stations in total, we must have

n2 ≤ c2 =⇒ n2 ≤ (n2 − k)2. Since c > 0, taking the square root gives

n ≤ n2 − k =⇒ k ≤ n2 − n.

If k ≥ n2 − n + 1, then c = n2 − k ≤ n − 1, which means the total number of possible component pairs is c2 ≤ (n − 1)2 = n2 − 2n + 1 < n2. By the Pigeonhole Principle, at least one pair of components must contain two or more stations. These stations will therefore be linked by both Company A and Company B. Thus, k = n2 − n + 1 guarantees a common linked pair.

To prove that k = n2 − n does not guarantee such a pair, we can construct a valid configuration with no common linked pairs. Arrange the n2 stations into an n × n grid, mapping the station at row r and column c, where 0 ≤ r ≤ n − 1 and 1 ≤ c ≤ n, to the altitude rn + c.

- • Company A operates cars between station i and i + 1 for all i ∈ {1,2,...,n2} except when i is a multiple of n. This forms n paths of length n, the rows of the grid. There are n(n − 1) = n2 − n cars. If i < j, then i + 1 < j + 1, ensuring that a car starting higher also finishes higher.
- • Company B operates cars between station i and i + n for all i ∈ {1,2,...,n2 − n}. This forms n paths of length n, the columns of the grid. There are n2 − n cars. If i < j, then i + n < j + n, again ensuring that a car starting higher finishes higher.

The components of A are the rows, and the components of B are the columns. The intersection of any row and any column contains exactly one station. Thus, no two stations are in the same component for both A and B, meaning no two stations are linked by both companies.

Therefore, the smallest positive integer k is n2 − n + 1.

|n2 − n + 1|
|---|

Scoring interpretation. This response gives the standard component-count proof for the upper bound and an explicit row-column construction for sharpness. The submitted construction payload is also accepted by the verifier. Therefore both Rigorous Proof Reasoning and Constructive Realization are present, and the verifier-gated score remains 7/7. This positive case shows that the construction check is not an automatic penalty; it confirms full credit when the proof and witness agree.

#### J.4 Case 4: Proof Score 1 and Construction Score 0

Item Description Problem USAMO-2020-P4-10 Run DeepSeek-V4-Pro, run 1 Proof score 1/7 Construction result 0/1 Verifier-gated score 1/7

Table 30: Case study with partial proof progress and failed construction.

#### Problem USAMO-2020-P4

Suppose that (a1,b1),(a2,b2),...,(a100,b100) are distinct ordered pairs of nonnegative integers. Let N denote the number of pairs (i,j) satisfying 1 ≤ i < j ≤ 100 and |aibj −ajbi| = 1. Determine the largest possible value of N over all possible choices of the 100 ordered pairs.

#### Relevant grading guidelines.

- • Partial: Reformulate the determinant condition geometrically or as a unimodular-pair condition; give a construction producing N = 197; or prove a nontrivial upper-bound step such as primitivity of participating vectors.
- • Almost: Prove the induction upper bound N ≤ 2n − 3, with at most a localized gap in the “at most two” argument, and give or nearly give the matching construction.
- • Full: Prove the sharp value 197 with both the matching construction and complete upper bound.

Full DeepSeek-V4-Pro generation. Solution to Question 1. Let the 100 distinct ordered pairs be (a1,b1),...,(a100,b100) with ai,bi ∈ Z≥0. Define

N = {(i,j) | 1 ≤ i < j ≤ 100, |aibj − ajbi| = 1} .

- 1. Only primitive pairs can contribute. Suppose a pair (a,b) has d = gcd(a,b) > 1. Then for any other pair (x,y), the determinant ay−bx is a multiple of gcd(a,b) = d. Hence |ay−bx| ≥ d > 1. Thus a non-primitive pair cannot form an edge with any other pair. To maximize N, we may restrict our attention to primitive pairs, those with gcd(a,b) = 1. The only primitive pairs on the axes are (1,0) and (0,1).
- 2. Graph interpretation. Consider the set V of all primitive pairs (a,b) with a,b ≥ 0 and (a,b) ̸= (0,0). Connect two vertices by an edge if their determinant is ±1. This graph is exactly the Stern-Brocot tree, or Farey graph. It is well known that the Stern-Brocot tree is a tree: it is connected and contains no cycles.
- 3. Upper bound. Choosing any 100 vertices from a tree induces a subgraph which is a forest. A forest on n vertices has at most n − 1 edges. Therefore N ≤ 100 − 1 = 99.
- 4. Achieving the bound. Take the 100 pairs (1,0),(1,1),(1,2),...,(1,98),(1,99). All are distinct, nonnegative, and primitive. For any 0 ≤ k ≤ 98,

|1 · (k + 1) − 1 · k| = 1,

so each consecutive pair contributes an edge. There are exactly 99 such edges, and no other pair gives determinant ±1. Hence N = 99 is attainable.

|99|
|---|

Scoring interpretation. The response contains a valid local observation: only primitive vectors can participate in determinant-one pairs. This matches a partial-credit item in the rubric, so a proof score of 1/7 is appropriate. However, the global argument is incorrect: it treats the relevant graph as a tree and concludes N = 99, while the benchmark target requires the sharp value 197. The construction payload also fails verification. Under the verifier-gated rule, a failed construction does not further demote scores already in the 0/1 range; the final score remains 1/7, reflecting limited but nonzero mathematical progress.

#### J.5 Case 5: Proof Score 1 and Construction Score 1

Item Description Problem USAMO-2021-P3-81 Run Kimi-K2.6, run 3 Proof score 1/7 Construction result 1/1 Verifier-gated score 1/7

Table 31: Case study with partial proof progress and accepted construction.

#### Problem USAMO-2021-P3

Let n ≥ 2 be an integer. An n × n board is initially empty. Each minute, one may place stones on an allowed L-shaped tromino region without stones, remove all stones from a full column, or remove all stones from a full row. Determine for which n it is possible that, after some nonzero number of moves, the board has no stones.

#### Relevant grading guidelines.

- • Partial: Establish the correct invariant or polynomial encoding, prove the roots-of-unity obstruction for 3 ∤ n, or give a correct explicit move sequence returning to empty for n = 3 or a clear 3 × 3 gadget reduction.
- • Almost: Prove the impossibility direction for 3 ∤ n with only a minor algebraic gap, and give the construction for multiples of 3.
- • Full: Completely prove that the possible values are exactly the multiples of 3. Full Kimi-K2.6 generation. Solution to Question 1. We claim that the board can be emptied after a non-zero number of moves if and only if 3 | n. Construction when 3 | n. Write n = 3m. Partition the rows into blocks

Rk = {3k,3k + 1,3k + 2} (k = 0,1,...,m − 1) and partition the columns into blocks

Cℓ = {3ℓ,3ℓ + 1,3ℓ + 2} (ℓ = 0,1,...,m − 1). Perform the following moves.

- 1. First batch of placements. For every k,ℓ ∈ {0,...,m − 1}, place stones at (3k + 2,3ℓ), (3k + 1,3ℓ + 1).

These (n/3)2 pairs of moves are pairwise disjoint: for a fixed k, the two placements in block (k,ℓ) occupy columns 3ℓ,3ℓ + 1,3ℓ + 2, so different ℓ do not interfere; for different k, the rows are disjoint. After all these placements, row 3k + 1 contains stones in every column 0,...,n − 1, so row 3k + 1 is completely full.

- 2. Clear the middle rows. For each k = 0,...,m − 1, remove row 3k + 1. After these clears the only remaining stones are

- • in row 3k: columns 3ℓ + 1 for all ℓ;
- • in row 3k + 2: columns 3ℓ and 3ℓ + 1 for all ℓ.

- 3. Second batch of placements. For every k,ℓ ∈ {0,...,m − 1}, place stones at (3k + 1,3ℓ).

These moves are also pairwise disjoint and are legal because row 3k + 1 is now empty and row 3k is empty at column 3ℓ. After this batch, columns 3ℓ and 3ℓ + 1 are completely full for every ℓ:

- • column 3ℓ receives a stone in each row 3k, in each row 3k + 1, and in each row 3k + 2;
- • column 3ℓ + 1 receives a stone in each row 3k, in each row 3k + 1, and in each row 3k + 2.

Columns 3ℓ + 2 contain no stones at all, since the only stones ever placed in them were in the rows 3k + 1 cleared in step 2.

- 4. Clear the full columns. For each ℓ = 0,...,m − 1, remove column 3ℓ and then column 3ℓ + 1.

All stones have been removed, and many moves were performed. Hence n = 3m works. Impossibility when 3 ∤ n. Colour the columns by their index modulo 3; call the three colour classes C0,C1,C2. For a configuration let Ni be the number of stones in Ci.

A placement at (r,c) adds stones to two cells in column c and one cell in column c + 1. Hence it contributes either (2,1,0), (0,2,1), or (1,0,2) to (N0,N1,N2), depending on c mod 3. In every case each of the three differences

N0 − N1, N1 − N2, N2 − N0 increases by exactly 1.

A row clear removes one stone from each column, so it leaves the three differences unchanged. A column clear of a column in Ci subtracts n from Ni; thus it changes the three differences by multiples of n.

Since the board starts and ends empty, the total change of each difference is 0. Consequently the number P of placements satisfies

P ≡ 0 (mod n).

But every placement adds three stones, while each clear removes n stones, so

3P = n(R + C)

for some non-negative integers R,C, the numbers of row and column clears. Hence n | 3P. If 3 ∤ n, then gcd(n,3) = 1, so n | P and the congruence P ≡ 0 (mod n) gives no contradiction yet.

Now consider the same column-colouring modulo 3. Because 3 ∤ n, the numbers of columns of the three colours in any single row are not all equal modulo 3; in particular a row clear changes the differences N0 − N1, N1 − N2, and N2 − N0 by non-zero constants. Adding the three exact change equations for the differences and using 3P = n(R + C) forces 3 | n. Therefore, if 3 ∤ n, no non-empty sequence of moves can return the board to the empty state.

Thus the required integers are precisely the multiples of 3.

|3 | n|
|---|

Scoring interpretation. The response gives a constructive pattern for the divisible-by-3 case, and the submitted construction payload is accepted by the verifier. This demonstrates successful Constructive Realization. However, the impossibility proof is not complete: the invariant argument becomes informal near the end and does not fully establish the required obstruction for all 3 ∤ n. The proof score therefore remains 1/7. This case illustrates that a valid witness should not be treated as a bonus that repairs an incomplete proof; the verifier-gated score stays at 1/7.

#### J.6 Case 6: Proof Score 0 and Construction Score 0

Item Description

Problem IMO-2025-P6-23 Run Qwen3.6-35B, run 1 Proof score 0/7 Construction result 0/1 Verifier-gated score 0/7

Table 32: Case study with no proof credit and failed construction.

#### Problem IMO-2025-P6

Consider a 2025 × 2025 grid of unit squares. Matilda wishes to place rectangular tiles, possibly of different sizes, such that each side of every tile lies on a grid line and every unit square is covered by at most one tile. Determine the minimum number of tiles needed so that each row and each column has exactly one unit square not covered by any tile.

#### Relevant grading guidelines.

- • Partial: Give a valid construction using the intended pattern, observe that uncovered squares form a permutation and introduce the LIS/LDS inequality, or set up the letter-counting lower-bound framework.
- • Almost: Prove the lower bound via the LIS/LDS letter-counting method with only a localized counting slip, and essentially complete the construction.
- • Full: Establish the correct minimum with both the sharp construction and a complete lower bound.

#### Full Qwen3.6-35B generation.

Solution to Question 1. Let n = 2025. We seek the minimum number of tiles required to cover the grid such that exactly one unit square remains uncovered in each row and column. This configuration of uncovered squares forms a permutation matrix.

Construction (Upper Bound). We demonstrate that 2n − 2 tiles are sufficient by placing the uncovered squares on the main diagonal at positions (i,i) for i = 0,1,...,n − 1. The remaining cells are partitioned into two disjoint regions:

- 1. Upper Triangle (r < c). For each column c ∈ {1,2,...,n − 1}, the cells (0,c),(1,c),...,(c − 1,c) form a vertical rectangle of size c × 1. There are n − 1 such rectangles.
- 2. Lower Triangle (r > c). For each row r ∈ {1,2,...,n−1}, the cells (r,0),(r,1),...,(r,r−1) form a horizontal rectangle of size 1 × r. There are n − 1 such rectangles.

These 2n − 2 rectangles are disjoint and cover all cells except the diagonal holes. Thus, 2n − 2 tiles are sufficient.

Minimality (Lower Bound). Analysis of small cases n = 1,2,3,4 shows that the minimum number of tiles is consistently 2n − 2. In the diagonal configuration, the upper and lower triangles are disconnected, and each staircase-shaped region requires at least n − 1 rectangles to partition. Any deviation from the diagonal to connect these regions introduces additional internal holes that fragment the free space, preventing a reduction in the total number of tiles. Therefore, the minimum number of tiles is 2n − 2.

For n = 2025, the minimum number of tiles is 2(2025) − 2 = 4048.

|4048|
|---|

Scoring interpretation. The response selects the wrong target value and does not use the required lower-bound mechanism. Its diagonal construction gives far too many tiles and does not address the sharp construction required by the grading guidelines. The construction payload is likewise rejected by the verifier. Since neither Rigorous Proof Reasoning nor Constructive Realization is demonstrated, the verifier-gated score remains 0/7.

