## Solving Inequality Proofs with Large Language Models

#### Pan Lu∗ α, Jiayi Sheng∗ β, Luna Lyu∗ α, Jikai Jinα, Tony Xiaα, Alex Guγ, James Zouα α Stanford University β UC Berkeley γ Massachusetts Institute of Technology

[Figure 1]

Website: https://ineqmath.github.io/ Code Dataset Leaderboard

[Figure 2]

[Figure 3]

[Figure 4]

# arXiv:2506.07927v3[cs.AI]15Dec2025

### Abstract

Inequality proving, crucial across diverse scientific and mathematical fields, tests advanced reasoning skills such as discovering tight bounds and strategic theorem application. This makes it a distinct, demanding frontier for large language models (LLMs), offering insights beyond general mathematical problem-solving. Progress in this area is hampered by existing datasets that are often scarce, synthetic, or rigidly formal. We address this by proposing an informal yet verifiable task formulation, recasting inequality proving into two automatically checkable subtasks: bound estimation and relation prediction. Building on this, we release IneqMath, an expert-curated dataset of Olympiad-level inequalities, including a test set and training corpus enriched with step-wise solutions and theorem annotations. We also develop a novel LLM-as-judge evaluation framework, combining a final-answer judge with four step-wise judges designed to detect common reasoning flaws. A systematic evaluation of 29 leading LLMs on IneqMath reveals a surprising reality: even top models like o1 achieve less than 10% overall accuracy under step-wise scrutiny; this is a drop of up to 65.5% from their accuracy considering only final answer equivalence. This discrepancy exposes fragile deductive chains and a critical gap for current LLMs between merely finding an answer and constructing a rigorous proof. Scaling model size and increasing test-time computation yield limited gains in overall proof correctness. Instead, our findings highlight promising research directions such as theorem-guided reasoning and self-refinement.

Proprietary reasoning LLMs Open-source reasoning LLMs Proprietary chat LLMs Open-source chat LLMs

| |24.5 35.0<br><br>54.5<br><br>65.5| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |44.5 47.5| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| |51.0 46.0| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| |39.5 38.0| | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

Gemini 2.5 Pro o3 o1 Grok 3 mini DeepSeek-R1 QwQ-32B Grok 3 Gemini 2.0 Flash Qwen2.5-72B Llama-4-Maverick

Figure 1: Final-answer accuracy versus overall accuracy for leading LLMs across different categories on the IneqMath benchmark of Olympiad-level inequality problems. Overall accuracy, measuring both answer correctness and step soundness, is substantially lower than final-answer accuracy for all model types. This highlights a critical gap: while LLMs may find correct final answers to these inequality problems, their reasoning is often unsound. Each model used its optimal maximal tokens.

* Co-first authors. Corresponding authors: {panlu,jamesz}@stanford.edu

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

### 1 Introduction

Mathematical inequalities are fundamental to diverse fields such as analysis, optimization, and probability theory, with applications spanning scientific modeling, economics, and competitive mathematics. Proving an inequality is a complex endeavor, demanding not just calculation but a sophisticated blend of intuition for discovering tight bounds, strategic insight for selecting and applying classical theorems (e.g., AM-GM, Cauchy-Schwarz), and precise symbolic transformations. These skills are hallmarks of advanced mathematical reasoning, distinguishing inequality proving from general math problem-solving. Automating this process would therefore have broad impact: it could supply automated theorem provers (ATPs) with missing lemmas, accelerate formal verification processes, and serve as a demanding testbed for general-purpose reasoners. However, despite impressive advancements in LLMs like DeepSeek-R1 [14] and OpenAI o3 [48], as well as in ATPs themselves [16, 18, 26, 34, 50, 75], automating inequality proving remains a challenging frontier.

A major bottleneck in advancing LLM capabilities for inequality proving is the scarcity of suitable benchmarks. Existing resources fall short in several ways: general ATP collections like MiniF2F [82] and ProofNet [7] contain few inequalities; synthetic datasets such as INT [64] and AIPS [63] offer scale but may lack structural diversity due to template-based generation; and curated collections like ChenNEQ [8] are often too small for extensive training. More fundamentally, most existing datasets adopt a fully formal representation, where problems and proofs are encoded in systems like Lean [11] or Isabelle [42]. While formal mathematical reasoning offers correctness guarantees and is a vital research direction, LLMs, trained on vast corpora of natural language, often exhibit strong informal reasoning capabilities. This suggests LLMs might solve problems informally even when struggling with the exacting syntax of formal provers. Our work, therefore, aims to explore and benchmark these informal abilities, complementing formal mathematical AI by focusing on a mode of reasoning closer to human intuition and the preliminary, often less structured, stages of mathematical discovery.

To bridge this gap between formal rigor and intuitive problem-solving, we propose an informal yet verifiable formulation (§2). Rather than requiring fully machine-checkable proofs within formal systems, we reformulate inequality problems into two concrete, automatically verifiable subtasks: (i) Bound estimation—determine the largest (or smallest) constant C that preserves the inequality; and (ii) Relation prediction—identify which relation (>, ≥, =, ≤, or <) holds between two expressions. Both tasks can be presented in natural language and LATEX, solved step-by-step by an LLM, and their final answers (a constant or a relation symbol) can be automatically checked. This preserves the creative essence of inequality proving while avoiding the heavy overhead of formal proof assistants. Building on this formulation, we present IneqMath (§3), the first large-scale dataset of Olympiad-level inequalities written entirely in informal language. The test set comprises 200 original problems, each crafted and reviewed by IMO-level medalists to ensure both originality and difficulty. The training corpus includes 1,252 problems sourced from advanced textbooks, automatically rephrased by LLMs into our subtasks and then meticulously reviewed by human experts. A key feature is that each training problem is accompanied by up to four step-wise solution paths, providing rich data for training LLMs on fine-grained reasoning. Additionally, 76.8% of the training problems are annotated with 83 named theorems across 29 categories relevant to their solutions. As shown in Table 2, IneqMath surpasses prior resources in scale, diversity, and alignment with human-like, informal problem-solving approaches.

However, producing the correct final answer is insufficient; the reasoning process itself must be sound. To rigorously assess this, we introduce an LLM-as-judge evaluation framework (§4). This framework comprises a high-precision final-answer judge to verify the answer equivalence, complemented by four specialized step-wise judges for step soundness. These step-wise judges are designed to detect the frequent reasoning flaws identified in our pilot studies: inappropriate reliance on toy case examples, unaddressed logical gaps, unjustified numeric approximations, and numeric calculation errors. Validated on manually labeled development set solutions, these judges demonstrate high reliability (F1 > 0.9 on average) and offer a scalable method to scrutinize the deductive integrity of LLM-generated proofs.

We evaluate 29 leading LLMs ranging from chat models to advanced reasoning LLMs, both opensource and proprietary (§5). As key results highlighted in Figure 1, several key findings emerge. While specialized reasoning LLMs (e.g., o1 [45]) achieve higher final-answer accuracy than general-purpose chat models (e.g., GPT-4o [43]), this advantage often collapses under step-wise scrutiny. Once our judges inspect every reasoning step, overall accuracy plummets by up to 65.5%. Indeed, even

top-performing models like o1 achieve less than 10% overall accuracy (Table 4), exposing fragile deductive chains and a significant gap between finding an answer and constructing a rigorous proof.

Our in-depth study (§5.3) reveals that while larger model sizes correlate with improved final-answer accuracy, their impact on overall accuracy is limited (e.g., o1 achieves only 8.0% overall accuracy). Similarly, extending test-time computation through longer reasoning chains offers diminishing returns in overall correctness (e.g., o1’s 8.0% overall accuracy remains unchanged when scaling maximum completion tokens from 5K to 40K, while o3 [48] saturates around 31%). These findings suggest that current scaling approaches are insufficient for robust deductive reasoning in IneqMath. Instead, we explore promising improvement strategies, demonstrating potential gains from methods such as theorem-guided reasoning—by providing golden theorems (improving overall accuracy by up to 11% for o3-mini [47] ) and critic-guided self-refinement (e.g., a 5% absolute increase in overall accuracy for Gemini 2.5 Pro [22]).

In summary, our work makes four key contributions: 1) We introduce an informal reformulation of inequality proving, decomposing the task into two verifiable subtasks (§2). 2) We release IneqMath, an expert-curated benchmark of Olympiad-level inequalities and a training corpus enriched with step-wise solutions and theorem annotations (§3). 3) We develop a modular LLM-as-judge framework that rigorously evaluates both final answers and proof step soundness (§4). 4) We conduct a systematic empirical study (§5) that exposes a pronounced gap between LLM performance and mathematical rigor, highlighting avenues for future research.

### 2 Task Formalization: An Informal Perspective

Inequality proof problems require demonstrating that a specified inequality holds under given conditions, such as proving a + b ≥ 2

√

ab for all positive real numbers a and b. Traditionally, these problems are formalized in proof assistants like Lean or Isabelle, represented as a tuple (S0,I,P), where S0 is the initial state, I is the inequality, and P is a set of premises. The proof process, often modeled as a Markov Decision Process, constructs a step-by-step solution verified by the system. However, this formal approach demands expertise in specialized tools, while informal proofs in natural language, though more intuitive, are difficult to verify automatically due to their unstructured nature.

To address these challenges, we propose an informal perspective that reformulates inequality proof problems into two verifiable subtasks: bound estimation and relation prediction.

IneqMath Training Example 1: Bound Problem Question: Find the maximal constant C such that for all real numbers a, b, c, the inequality holds:

a2 + (1 − b)2 + b2 + (1 − c)2 + c2 + (1 − a)2 ≥ C Solution: Applying Minkowsky’s Inequality to the left-hand side we have

a2 + (1 − b)2 + b2 + (1 − c)2 + c2 + (1 − a)2 ≥ (a + b + c)2 + (3 − a − b − c)2 By denoting a + b + c = x, we get

|3√2 2<br><br>|
|---|

2

3 2

9 2 ≥

9 2

(a + b + c)2 + (3 − a − b − c)2 = 2 x −

+

=

.

Minkowsky’s Inequality Theorem: For any real number r ≥ 1 and any positive real numbers a1, a2, . . . , an, b1, b2, . . . , bn

1 r

1 r

1 r

n

n

n

(ai + bi)r

ari

bri

≤

+

i=1

i=1

i=1

This bound estimation task involves finding an optimal constant for a given inequality. For example, in a + b ≥ C

√

ab for ∀a,b > 0, the objective is to find the largest C. Formally, a bound estimation problem instance is a triple:

Πbound = f(x), g(x), D , where D ⊆ Rn. Here, f,g : D → R are two expressions involving variables x = (x1,...,xn) within a specified domain D (e.g., xi > 0, xi = 1), and g(x) > 0,∀x ∈ D. The goal is to determine the extremal: C⋆ = sup{C ∈ R : f(x) ≥ Cg(x),∀x ∈ D} or C⋆ = inf{C ∈ R : f(x) ≤ Cg(x),∀x ∈ D}.

The relation prediction task requires determining the correct relationship between two expressions. For instance, given expressions f(x) = a + b and g(x) = 2

√

ab, the goal is to identify the relation

(in this case, ≥) that holds for ∀a,b > 0. Formally, a relation prediction problem instance is a triple: Πrel = f(x), g(x), D ,

where f(x) and g(x) are expressions over variables x in domain D ⊆ Rn. The goal is to find the relation between f(x) and g(x) (i.e. >, ≥, =, ≤, <, or none of the above).

IneqMath Training Example 2: Relation Problem Question: Let a, b, c be positive real numbers such that abc = 1. Consider the following expressions:

√

( ) √a +

b + √c + 3

b + c √a

c + a √

a + b √c

+

+

b

Determine the correct inequality relation to fill in the blank. Options: (A) ≤ (B) ≥ (C) = (D) < (E) > (F) None of the above Solution: From the AM-GM Inequality, we have

b + c √a

c + a √

a + b √c ≥ 2

bc a

ca b

ab c

+

+

+

+

b

bc a

ca b

ca b

ab c

ab c

bc a

+

+

+

+

+

=

√

√

√

√

≥ 2(√a +

b + √c + 3. AM-GM Inequality Theorem: If a1, a2, . . . , an are nonnegative real numbers, then

b + √c)

√a +

b + √c + 3 6

abc = √a +

|≥|
|---|

n

1 n

ai ≥ (a1a2 . . . an)n1

i=1

with equality if and only if a1 = a2 = . . . = an. This is a special case of the Power Mean Inequality.

These subtasks are chosen because they frequently appear in mathematical problem-solving, simplify the evaluation process, and crucially, retain the core reasoning challenges inherent in original inequality proof problems. An ideal LLM solution should not only produce the correct final answer but also present a clear, logically sound, and complete derivation. This includes strategic application of theorems, accurate symbolic manipulations and calculations, and justification of all critical steps.

### 3 IneqMath: The Inequality Problem Dataset

This section describes the data curation process and key statistics of IneqMath, a novel collection of inequality problems designed to support the informal perspective on solving and proving inequalities.

Test data curation. To mitigate contamination from common sources (textbooks, contests and online resources) that may be present in LLM training corpora, we commissioned IMO-level medalists to design novel inequality problems. These underwent rigorous review by a separate expert group and were validated only upon unanimous confirmation of solvability, soundness, and ground truth correctness. Problems identified as easier by experts were excluded from the test set (repurposed for development) to ensure a high level of challenge. To further illustrate the modest contamination, we also conduct a memorization probe on the test set in §C.8. See the developed curation tool in §A.2. We host an online evaluation website1, providing a fair evaluation platform for the community.

Key statistics. As shown in Table 1, the IneqMath dataset comprises 200 test problems for benchmarking, 100 development problems with public ground truth, and 1,252 training problems split evenly between bound estimation and relation prediction tasks. Each training problem includes step-wise solutions, with up to four solutions per problem, and 76.8% (962 problems) are annotated with relevant theorems. The dataset features 83 named theorems across 29 categories, with their distribution illustrated in Figure 2. Test problem examples are provided in §A.4.

1https://huggingface.co/spaces/AI4Math/IneqMath-Leaderboard

###### Statistic Number Bnd. Rel.

Theorem categories 29 - Named theorems 83 - -

Training problems (for training) 1252 626 626

- - With theorem annotations 962 482 480
- - With solution annotations 1252 626 626
- - Avg. solutions per problem 1.05 1.06 1.05
- - Max solutions per problem 4 4 4

Dev problems (for development) 100 50 50 Test problems (for benchmarking) 200 96 104

- Table 1: Statistics of the IneqMath dataset.

13.3%

10.8%

7.2%

7.2%

6.0% 6.0% 6.0%

4.8%

3.6%

3.6%

3.6%

2.4%

2.4% 2.4%

2.4%

21.7%

13.3 Inequality Between Means

Cauchy-Schwarz Inequality

10.8

7.2 Chebyshev's Inequality

7.2 Schur's Inequality

Convexity, Jensen's Inequality

6.0

6.0 Hölder's Inequality

6.0 Minkowski's Inequality

4.8 Rearrangement Inequality

3.6 Abstract Concreteness

3.6 Bernoulli's Inequality

3.6 Differential Calculus

2.4 Maclaurin's Inequality

2.4 Suranyi's Inequality

2.4 Trigonometry

2.4 Popoviciu's Inequality

21.7 Others

Figure 2: Distribution of theorem categories.

Comparison to existing datasets. As summarized in Table 2, IneqMath stands out for: (1) providing expert-curated training and test sets, (2) offering rich annotations with step-wise solutions and 83 grounded theorems, and (3) adopting an informal, accessible format for inequality proving through bound estimation and relation prediction, evaluated via LLM-as-judge. This design bridges the gap between formal proof systems and intuitive mathematical reasoning, making IneqMath a unique resource for advancing LLM capabilities in problem solving and theorem proving.

Data Source Data Annotation Problem and Evaluation Datasets Training Test / Dev #Theorem Solution Category Format Evaluation INT [64] Synthesized Synthesized 35 ✓ Proof Formal Symbolic DSL AIPS [63] Synthesized ✗ 8 ✓ Proof Formal Symbolic DSL MO-INT [63] ✗ Data compilation ✗ ✗ Proof Formal Symbolic DSL MINIF2F [82] ✗ Autoformalization ✗ ✗ Proof Formal ProofNet [7] ✗ Autoformalization ✗ ✗ Proof Formal FormalMATH [77] ✗ Autoformalization ✗ ✗ Proof Formal leanWorkbook [76] Autoformalization Autoformalization ✗ ✗ Proof Formal Proof or Bluff [49] ✗ Data compilation ✗ ✗ Proof Informal Human judge CHAMP [39] ✗ Autoformalization ✗ ✗ Open Informal Human judge Putnam Axiom [23] ✗ Data compilation ✗ ✗ Open Informal Answer checking LiveMathBench [37] ✗ Data compilation ✗ ✗ Open Informal Answer checking IneqMath (Ours) Expert annotated Expert annotated 83 ✓ MC, Open Informal LLM-as-judge

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

- Table 2: Comparison of datasets for inequality and theorem proving. IneqMath provides expert-annotated training and test/dev sets, featuring high-quality named theorems and step-wise solutions for model development. Unlike prior datasets using synthesis or autoformalization, IneqMath presents problems in informal language across multiple-choice (MC) and open-ended (Open) formats, and employs LLM-as-judge for evaluation.

Potential contamination statement. To ensure rigorous evaluation, the IneqMath test set was commissioned from IMO-level medalists to feature novel problems, minimizing prior LLM pretraining exposure. The poor performance across models (§5.2), particularly in overall accuracy (which demands step-wise correctness), strongly suggests that the benchmark poses a significant reasoning challenge, regardless of any potential familiarity with the underlying mathematical concepts. We therefore believe the IneqMath test set effectively probes novel problem-solving capabilities, and our conclusions on current LLM limitations in rigorous inequality proving remain robust.

### 4 Fine-grained Informal Judges for Inequality Solving

The test split of the IneqMath dataset serves as our benchmark, comprising 200 Olympiad-level inequality problems that challenge both humans and current LLMs. Traditional evaluation methods fall short in this setting: expert annotation is accurate but prohibitively labor-intensive, while automated techniques such as string matching or value equivalence fail to capture step-by-step correctness—an essential aspect of inequality problem solving. To address this, we propose a fine-grained LLMas-judge framework as illustrated in Figure 3, consisting of a final-answer judge for verifying the predicted answer (§4.1) and four specialized step-wise judges targeting common reasoning flaws (§4.2). A solution is considered correct overall only if it passes all five judges. As shown in Table 3, these judges achieve strong alignment with human annotations (F1 = 0.93), providing a scalable yet reliable alternative to manual evaluation.

- 4.1 Final Answer Judge LLM-generated solutions to IneqMath problems typically involve multiple reasoning steps followed by a concluding answer statement. However, the final answer may vary in phrasing, structure, or

√2 2

numeric format, especially for bound estimation problems. For example, C = √12 and C =

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Final Answer Judge Toy Case Judge Logical Gap Judge Approximation Judge Computation Judge

The answer is

Then, we have ( 2 + 1) = 3 + 2 2

Since 1+ 2 ≥ 2 x 1 x 2, 𝑎 + 𝑏 ≥ 2𝑎𝑏 holds for every 𝑎,𝑏 > 0.

For simplicity, replace all 𝜋 with 3.14 and complete the proof.

Based on intuition, the maximum value of 𝑓 𝑎,𝑏,𝑐 is 2.

###### C = 0.5

|Ground Truth:<br><br>C = 12<br><br>|
|---|

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Generate Code

[Figure 23]

Correct? No

Correct?

Correct? No

[Figure 24]

Equivalent? Yes

No Used the approximation of 𝜋 to provide a nonrigorous proof.

[Figure 25]

[Figure 26]

[Figure 27]

Execute

Correct? Yes

Logical gap exists as no clear steps are given to derive the extremum.

[Figure 28]

Used a toy case to make a general conclusion.

[Figure 29]

Final Answer Correct.

Computation Correct.

Figure 3: Illustration of the fine-grained LLM-as-judge framework. The framework combines a Final Answer Judge with four step-wise judges: Toy Case Judge, Logical Gap Judge, Numerical Approximation Judge (shown as Approximation Judge), and Numerical Computation Judge (shown as Computation Judge). A solution is considered correct only if it passes all five judges.

LLM-as-Judge Judge type Accuracy Precision Recall F1 score

Final Answer Judge Answer checking 1.00 1.00 1.00 1.00 Toy Case Judge Step soundness 0.91 0.86 0.97 0.91 Logical Gap Judge Step soundness 0.96 0.95 0.98 0.96 Numerical Approximation Judge Step soundness 0.96 0.95 0.98 0.96 Numerical Computation Judge Step soundness 0.71 0.68 0.98 0.80 Average - 0.91 0.89 0.98 0.93

Table 3: Performance metrics of LLM-as-judge framework on development set.

are mathematically equivalent but differ in form. Recent work [38] evaluates LLM outputs via format normalization and exact string matching, without accounting for mathematical equivalence. To address this, we propose a two-stage Final Answer Judge: it first identifies the concluding sentence containing the predicted answer, and then performs robust equivalence checking to assess mathematical correctness, even when the form differs from the reference. Prompt details and examples are in §B.1.

#### 4.2 Four Step-wised Judges

Toy Case Judge. Inequality problems in IneqMath often require reasoning over continuous domains (e.g., all a,b,c > 0), where specific numerical examples alone are insufficient for a valid proof. LLM frequently generalizes incorrectly from such examples—e.g., claiming an inequality holds universally because it holds for a = 1,b = 2,c = 3. Prior work [17] flags these under a broad “logical flaw” category, lacking granularity for targeted analysis. Our Toy Case Judge addresses this by detecting unjustified generalization from toy examples. It prompts an LLM to flag conclusions based solely on specific instances without broader justification. See §B.2 for prompts and examples.

Logical Gap Judge. IneqMath inequality problems often involve multi-step derivations (e.g., algebraic manipulation, constrained optimization, functional transformations) needing explicit justification. LLMs, however, often skip key reasoning steps or assert conclusions without support (e.g., stating an optimal bound without derivation). Existing step-level evaluations [68] assess validity and redundancy but lack granularity for such logical omissions. Our Logical Gap Judge addresses this by flagging missing transitions, unjustified claims, and vague derivations, especially in steps involving inequality transformations or bound estimation (see §B.3 for details).

Numerical Approximation Judge. Inequality problems in IneqMath often demand exact symbolic reasoning, where the use of numeric approximations—e.g., replacing √2 with 1.414—can compromise mathematical rigor. However, many LLM-generated solutions resort to such approximations in intermediate steps, leading to inaccurate or non-generalizable conclusions. To address this, we introduce a Numerical Approximation Judge that flags inappropriate use of numeric approximations—specifically when they affect derivations or final answers. Approximations used solely for intuition or side remarks are permitted. See §B.4 for prompt details and examples.

ActualTrueActualFalse

ActualTrueActualFalse

ActualTrueActualFalse

ActualTrueActualFalse

ActualTrueActualFalse

40

40

40

40

40

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

40 0

37 1

39 1

40 1

46 1

30

30

30

30

30

20

20

20

20

20

10

10

10

10

10

0 40

6 36

2 38

2 37

23 10

0

0

0

0

0

Pred. True Pred. False

Pred. True Pred. False

Pred. True Pred. False

Pred. True Pred. False

Pred. True Pred. False

(a) Final answer

(b) Toy case

(c) Logical gap

(d) Numerical approximation

(e) Numerical computation

- Figure 5: Confusion matrices for five judges, which exhibit strong agreement with human labels.

Numerical Computation Judge. Many IneqMath problems require explicit numerical computations after variable assignment (e.g., evaluating 272 or summing rational terms). While symbolic reasoning is vital, arithmetic accuracy is equally crucial for overall correctness. Prior work (e.g., EIC-Math [30]) categorizes broad error types but often overlooks subtle miscalculations in multi-step derivations. Our Numerical Computation Judge addresses this by verifying arithmetic steps once variables are instantiated. It prompts an LLM to extract numerical expressions, convert them into Python code, and evaluate using floating-point arithmetic within a small tolerance. This enables high-precision checking of both intermediate and final results. To further improve precision and mitigate floating-point issues, we encourage the use of symbolic mathematics packages such as SymPy, particularly for handling fractions and decimal numbers. Additional details are provided in §B.5.

#### 4.3 Effectiveness Verification of Judges

A holistic LLM judge baseline. To motivate our specialized judging system, we first evaluate a heuristic LLM-as-judge baseline. This prompts a single, general-purpose LLM to holistically assess IneqMath solution correctness, based on both final answer accuracy and step-wise soundness across the four reasoning categories in §4.2. As shown in the confusion matrix (Figure 4) using 80 human-annotated development examples, this naive approach exhibits poor agreement with human labels, underscoring its unreliability for rigorous evaluation in this domain.

ActualTrueActualFalse

40

[Figure 35]

17 24

30

20

10

4 35

0

Pred. True Pred. False

Figure 4: Confusion matrix for the judge baseline.

Performance of our fine-grained judges. In contrast, our proposed suite of five specialized judges exhibits strong alignment with human evaluations. Figure 5 presents the confusion matrices for each judge on the same development set. The final answer judge (using GPT-4o-mini) achieves near-perfect agreement, while the four step-wise judges (chosen for a balance of performance and cost as detailed in §B.6) also demonstrate high fidelity. This confirms that decomposing the complex evaluation task into targeted sub-problems allows LLMs to serve as reliable evaluators.

Qualitative analysis of judge disagreements. Despite the strong aggregate performance (overall F1 = 0.93, Table 3), LLM-as-judge evaluations are not perfect. Acknowledging the skepticism surrounding LLM-based evaluation, we conducted a qualitative analysis of failure cases where our judges’ assessments diverged from human annotations. Detailed examples are provided in §B.7. These instances underscore that while highly effective, our LLM judges can still struggle with the deep, nuanced understanding that characterizes expert-level mathematical reasoning.

### 5 Experiments in IneqMath

#### 5.1 Experimental Setups

We conduct a systematic evaluation of 29 leading LLMs on the inequality problems in the IneqMath test set. The evaluated models span two categories: general-purpose chat models (both open-source and proprietary) and specialized reasoning LLMs designed for complex, multi-step problem-solving. All models are prompted in a zero-shot setting with the problem statement and the instruction: “Please solve the problem with clear, rigorous, and logically sound steps” to encourage detailed reasoning. Model responses are assessed using our LLM-as-judge framework (§4). We report three key metrics:

- • Answer Acc: Measures the predicted answer correctness, verified by the final-answer judge (§4.1).
- • Step Acc: Aggregates the correctness of individual reasoning steps as determined by our four specialized step-wise judges (§4.2), which target common flaws.
- • Overall Acc: The primary metric, which deems a solution correct only if it achieves both a correct final answer and flawless step-wise reasoning (i.e., passes all five judges).

A response is thus considered fully correct (Overall Acc) only if it produces a correct final answer through logically valid steps, passing scrutiny from all judges. Additional setup details are in §C.1.

Step Acc (↑) Overall Acc (↑) Answer Acc (↑) No Toy Case No Logic. Gap No Approx. Error No Comp. Error

Models All Bnd. Rel. All Bnd. Rel. All Bnd. Rel. All Bnd. Rel. All Bnd. Rel. All Bnd. Rel. Heuristic Methods

Random Guess - - - 8.5 0.0 16.3 - - - - - - - - - - - Frequent Guess - - - 18.0 9.4 26.0 - - - - - - - - - - - -

###### Open-source Chat LLMs

Qwen2.5-Coder-32B [27] 1.5↓39.0 1.0↓50.0 1.9↓28.9 40.5 51.0 30.8 36.0 27.1 44.2 3.0 2.1 3.8 90.5 96.9 84.6 88.5 89.6 87.5 Llama-4-Scout [41] 1.5↓32.0 2.1↓44.8 1.0↓20.2 33.5 46.9 21.2 30.5 15.6 44.2 3.5 4.2 2.9 93.0 94.8 91.3 92.5 92.7 92.3 Qwen2.5-72B [51] 2.5↓39.5 3.1↓47.9 1.9↓31.8 42.0 51.0 33.7 54.5 53.1 55.8 5.0 4.2 5.8 91.0 94.8 87.5 95.0 94.8 95.2 Llama-4-Maverick [40] 2.5↓38.0 2.1↓43.7 2.9↓32.7 40.5 45.8 35.6 42.5 28.1 55.8 4.0 4.2 3.8 89.0 91.7 86.5 95.0 92.7 97.1 Qwen2.5-7B [52] 3.0↓32.0 2.1↓38.5 3.8↓26.0 35.0 40.6 29.8 44.5 32.3 55.8 4.5 3.1 5.8 92.5 96.9 88.5 93.0 92.7 93.3

###### Proprietary Chat LLMs

Gemini 2.0 Flash-Lite [20] 1.5↓31.5 2.1↓41.7 1.0↓22.1 33.0 43.8 23.1 11.5 11.5 11.5 3.5 3.1 3.8 73.0 77.1 69.2 90.5 87.5 93.3 GPT-4o mini [44] 2.0↓37.5 1.0↓41.7 2.9↓33.6 39.5 42.7 36.5 29.0 11.5 45.2 2.5 2.1 2.9 90.0 91.7 88.5 93.0 92.7 93.3 GPT-4.1 [46] 2.5↓38.0 0.0↓31.3 4.8↓44.2 40.5 31.3 49.0 16.0 12.0 19.0 10.0 8.3 11.5 59.5 66.7 52.9 93.5 92.7 94.2 GPT-4o [43] 3.0↓34.5 2.1↓38.5 3.8↓30.8 37.5 40.6 34.6 32.0 21.9 43.0 3.5 3.1 3.8 92.5 93.8 91.4 94.0 93.8 94.2 Gemini 2.0 Flash [19] 3.0↓46.0 3.1↓56.3 2.9↓36.5 49.0 59.4 39.4 15.5 13.5 17.3 13.5 7.3 19.2 55.5 60.4 51.0 94.5 94.8 94.2 Grok 3 [66] 3.5↓51.0 4.2↓62.5 2.9↓40.4 54.5 66.7 43.3 17.0 13.7 20.2 16.0 11.6 20.2 36.0 42.1 30.8 93.0 96.8 90.4

###### Open-source Reasoning LLMs

QwQ-32B [5] 2.0↓47.5 2.1↓52.1 1.9↓43.3 49.5 54.2 45.2 26.0 25.0 26.9 29.5 20.1 37.5 21.0 20.8 21.2 87.0 82.3 91.3 Deepseek-R1 (Llama-70B) [12] 3.5↓50.0 5.2↓53.1 1.9↓47.1 53.5 58.3 49.0 23.0 24.0 22.1 26.0 20.9 30.8 35.5 38.5 32.7 87.0 89.6 84.6 Deepseek-R1 (Qwen-14B) [13] 5.0↓35.5 6.3↓36.4 3.8↓34.7 40.5 42.7 38.5 21.0 18.8 23.1 21.0 19.8 22.1 35.5 38.5 32.7 85.0 91.7 78.8 Deepseek-R1 [14] 5.0↓44.5 4.2↓63.5 5.8↓26.9 49.5 67.7 32.7 57.0 53.1 60.9 17.5 6.3 27.9 81.0 95.8 67.3 95.0 99.0 91.3 Qwen3-235B-A22B [53] 6.0↓35.0 3.1↓32.3 8.7↓37.5 41.0 35.4 46.2 35.0 30.2 39.4 36.0 26.0 45.2 31.0 28.1 33.7 92.5 93.8 91.3

###### Proprietary Reasoning LLMs

Claude 3.7 Sonnet [6] 2.0↓40.0 2.1↓44.8 1.9↓35.6 42.0 46.9 37.5 49.0 36.5 60.6 4.0 3.1 4.8 93.5 95.8 91.3 93.0 90.6 95.2 Gemini 2.5 Flash [21] 4.5↓1.0 3.1↓1.1 5.8↓0.9 5.5 4.2 6.7 88.0 84.4 91.3 13.5 7.3 19.2 100.0 100.0 100.0 100.0 100.0 100.0 Grok 3 mini [67] 6.0↓65.5 4.2↓68.7 7.7↓62.5 71.5 72.9 70.2 24.0 16.7 30.8 19.5 11.5 26.9 53.5 63.5 44.2 91.0 94.8 87.5 Gemini 2.5 Pro [22] 6.0↓1.0 7.3↓1.0 4.8↓1.0 7.0 8.3 5.8 88.5 83.3 93.3 19.0 12.5 25.0 100.0 100.0 100.0 99.5 100.0 99.0

- o1 [45] 8.0↓54.5 7.3↓55.2 8.7↓53.8 62.5 62.5 62.5 34.5 37.5 31.7 17.5 12.5 22.1 86.5 99.0 75.0 99.5 100.0 99.0

- o3-mini [47] 9.5↓53.0 7.3↓62.5 11.5↓44.3 62.5 69.8 55.8 37.0 34.4 39.4 22.0 17.7 26.0 77.5 92.7 63.5 95.0 96.9 93.3

- o4-mini [48] 15.5↓49.5 14.6↓48.9 16.3↓50.0 65.0 63.5 66.3 62.0 58.3 65.4 26.0 25.0 26.9 86.5 90.6 82.7 93.0 92.7 93.3

- o3 [48] 21.0↓16.0 18.8↓11.4 23.1↓20.2 37.0 30.2 43.3 93.5 91.7 95.2 39.5 28.1 50.0 91.5 99.0 84.6 97.0 96.9 97.1 Average Accuracy (↑) 5.0↓38.0 4.5 ↓42.9 5.5↓33.5 43.0 47.4 39.0 40.3 34.8 45.5 15.0 11.0 18.7 73.1 77.9 68.8 93.2 93.7 92.8 Average Error Rate (↓) 95.0↑38.0 95.5 ↑42.9 94.5↑33.5 57.0 52.6 61.0 59.7 65.2 54.5 85.0 89.0 81.3 26.9 22.1 31.2 6.8 6.3 7.2

Table 4: Evaluation performance of chat and reasoning LLMs on the IneqMath benchmark (the test set). Bnd. denotes bound problems and Rel. denotes relation ones. We report: (1) Overall Acc, which reflects the correctness

- of both the final answer and intermediate steps; (2) Answer Acc, which measures final answer correctness alone; and (3) Step Acc, which evaluates the accuracy of intermediate steps across four error categories—Toy Case, Logical Gap, Numerical Approximation, and Numerical Computation. Blue superscripts ↓ indicate accuracy drop (Overall Acc - Answer Acc) from step-wise errors. Underlining denotes best result within each model category; boldface highlights best overall performance. Default max token limit for reasoning LLMs is 10K.

- 5.2 Main Evaluation Results Table 4 presents the performance of the evaluated LLMs on IneqMath. Our analysis reveals several critical insights into current LLM capabilities for inequality proving:

- 1) Reasoning LLMs achieve higher final-answer accuracy. Models like o1 (62.5% Answer Acc) and Grok 3 mini (71.5% Answer Acc) significantly outperform their general-purpose chat counterparts (e.g., GPT-4o at 37.5%, Grok 3 at 54.5%) in identifying the correct final answer. This suggests that specialized architectures or training techniques improve their search ability to find final answers.
- 2) Step-wise scrutiny reveals a dramatic performance drop. The advantage in Answer Acc often masks underlying reasoning flaws. Overall Acc plummets when steps are evaluated. For instance, Grok 3 mini’s accuracy drops by 65.5% (from 71.5% Answer Acc to 6.0% Overall Acc), and o3-mini by 53.0%. This stark discrepancy underscores the fragility of LLM-generated deductive chains.
- 3) Robust proof construction remains a major challenge. Even top models like o1 achieve low Overall Acc (8.0%). Many large models, despite moderate Answer Acc, also score poorly (e.g., Grok

- 3 at 3.5% Overall Acc). This indicates a fundamental gap between finding a plausible answer and constructing a mathematically rigorous, step-by-step derivation.

- 5.3 In-depth Study Failure solution analysis. As shown in Table 4, where we report average error rates for overall accuracy, final-answer accuracy, and step-wise accuracy across four categories, the most common step-wise errors in LLM-generated solutions are logical gaps (85.0% average failure rate across models) and unjustified generalization from toy cases (59.7%). Less frequent, but still significant, are errors from numerical approximations (26.9%) and miscalculations (6.8%). A detailed inspection of incorrect solutions (see examples in §C.2.1-§C.2.4) highlights these prevalent error patterns, which often undermine proofs even when LLMs produce the correct final answer. Beyond these step-wise

Gemini 2.5 Pro o1

DeepSeek-R1 (Llama-70B)

Grok 3 DeepSeek-R1

o1

Llama-4-Maverick

QwQ-32B-Preview

DeepSeek-R1 (Qwen-14B)

Qwen2.5-72B

Gemini 2.5 Pro DeepSeek-R1 Grok 3

DeepSeek-R1 (Llama-70B)

Qwen2.5-Coder-32B

Qwen2.5-7B

Lamma-4-Scout

Llama-4-Maverick

DeepSeek-R1 (Qwen-1.5B)

QwQ-32B

DeepSeek-R1 (Qwen-1.5B)

Qwen2.5-72B

Llama-3.1-8B

Llama-3.1-8B

Gemma-2B

Qwen2.5-Coder-32B

Gemma-2-9B

Llama-4-Scout

Llama-3.2-3B

Gemma-2-9B

Gemma-2B

Llama-3.2-3B

- Figure 6: Model-size scaling law (Answer Acc).

Figure 7: Model-size scaling law (Overall Acc).

errors, LLMs also struggle to derive correct final answers on complex problems (§C.2.5), indicating deeper challenges in theorem application and symbolic manipulation.

Scaling law in model size. Figure 6 shows how final-answer accuracy (which evaluates only the correctness of the final predicted answer) scales with model size for LLMs. As model size increases, we observe a steady improvement in answer accuracy, reflecting an empirical scaling law that larger models are better at inferring correct bounds and inequality relationships. However, this trend does not hold well when considering overall accuracy—which requires both a correct answer and valid intermediate reasoning steps—as shown in Figure 7. In this latter case, the scaling curve flattens, indicating that increased model size alone is insufficient to eliminate step-by-step reasoning errors.

Scaling law in test-time computation. Extended test-time computation, allowing longer reasoning chains, is a common strategy for complex problem-solving [14]. We investigated its impact on overall accuracy in IneqMath by varying the maximum completion tokens for reasoning LLMs. Figure 8 shows that while models like Gemini 2.5 Pro and o3 initially improve with more tokens, performance gains saturate (e.g., beyond 20K tokens). This indicates that merely increasing computational budget offers diminishing returns for achieving rigorous, step-wise correct proofs, highlighting the need for more than just longer thought processes.

Gemini 2.5 Pro

50

- o1

- o3

# Max Completion Tokens OverallAccuracy(%)

Gemini 2.5 Flash

40

30

20

10

5K 10K 20K 30K 40K

Figure 8: Scaling law in test-time computation for reasoning LLMs.

#### 5.4 Exploring Improvement Strategies

Retrieving relevant theorems as hints. To assess theorembased hints, we provide models with the top-k most frequent theorems from our IneqMath training corpus when solving a 40-problem test subset. As shown in Figure 9, providing one or two such theorems decreases overall accuracy for weaker models (e.g., Grok 3 mini, o3-mini, o4-mini), likely due to misapplication or distraction by potentially irrelevant information. Conversely, stronger models like Gemini 2.5 Pro benefit from these hints, suggesting advanced reasoning is crucial to effectively use such guidance. These results underscorethepotentialoftheorem-guidedreasoningbutalso highlight the critical need for more sophisticated theoremretrieval mechanisms (e.g., RAG [28, 24]) to reliably enhance LLM performance in inequality proving. Detailed experiments are available in §C.4.

60

No theorem (baseline) 53

50

- 1 theorem (experiment)

| |
|---|

- 2 theorems (experiment)

50

| |
|---|

OverallAccuracy(%)

43

40

30

20

20

18

20

10 10

10

8

10

5

5

0

Grok3minio3-mini(30K)o4-mini(30K)Gemini2.5Pro(30K)

Figure 9: Model performance with retrieved theorems as hints.

Self-improvement via critic as feedback. Allowing an LLM to critique and revise its own reasoning has been shown to improve performance on complex tasks [78, 57]. To explore whether this holds for inequality proving, we randomly sampled 40 test problems from IneqMath and ran one round of self-critique. As Figure 10 shows, self-critique consistently improves performance—e.g., Gemini 2.5 Pro’s overall accuracy rises from 43% to 48%. This upward trend underscores self-critique as a promising, supervision-free method to enhance logical rigor and solution quality of LLMs in inequality reasoning. More details are in §C.5.

No critic (baseline) With critic (experiment)

48

50

OverallAccuracy(%)

43

| |
|---|

40

30

23

20

20

15

10 10

10

10

0

Grok3mini o3-minio4-mini(30K)Gemini2.5Pro(30K)

Figure 10: Model performance via self-critic as feedback.

### 6 Related Work

Datasets for inequality and theorem proving. One of the major bottlenecks in advancing LLM capabilities for inequality proving is the scarcity of suitable datasets. Existing resources fall short in several ways: general ATP collections like MiniF2F [82] and ProofNet [7] contain few inequalities; synthetic datasets such as INT [64] and AIPS [63] offer scale but often lack structural diversity due to their template-based generation; and curated collections like ChenNEQ [8] are often too small for extensive training. More fundamentally, most existing datasets [80, 59, 73, 29, 58, 26] adopt a fully formal representation, where problems and proofs are encoded in systems such as Lean [11] or Isabelle [42]. While formal mathematical reasoning offers correctness guarantees and is a vital research direction, LLMs, trained on vast corpora of natural language, often exhibit strong informal reasoning capabilities. Therefore, our IneqMath adopts an informal perspective, reformulating inequality proof problems into two verifiable subtasks—bound estimation and relation prediction. These problems within IneqMath were crafted and reviewed by IMO-level medalist experts. Other informal reasoning datasets [49, 39, 23, 37] typically lack annotated solutions, theorem references, or corresponding training data. To address these gaps, IneqMath introduces 1,252 inequality problems for training, each annotated with theorems relevant to its solution, which comprises up to four steps.

Methods for inequality and theorem proving. Proving inequalities is complex, requiring intuition to identify tight bounds, strategic use of theorems, and precise symbolic manipulation. Traditional automated theorem provers (ATPs) primarily operate within formal systems like Lean [11] or Isabelle [42], requiring problems and proofs to be encoded in specialized languages. Inspired by the mathematical reasoning capabilities of LLMs [81], a significant body of recent work has focused on integrating LLMs with these formal ATPs. These approaches often model theorem proving as a Markov Decision Process (MDP), training LLMs to select appropriate tactics and premises to construct proofs within the formal system [1, 9, 62, 16, 18, 26, 34, 61, 72]. For instance, systems like Goedel-Prover [35] leverage large Lean corpora to train models for tactic prediction, enabling end-to-end formal proof generation. Other methods incorporate tree-search techniques to navigate the search space of premises within formal frameworks [65, 31, 70, 71].

LLMsaretrainedonvastnaturallanguagecorpora, givingthemstrengthsininformalreasoning—closer to how humans solve problems. This reveals an opportunity for methods that harness these informal abilities. Our work departs from formal paradigms by introducing an informal yet verifiable framework for inequality proving, designed to benchmark and enhance LLM performance in human-like problem solving, while exploring improvements such as theorem-guided reasoning and self-refinement.

LLM-as-judge for math problem solving. Reliable evaluation of mathematical problem-solving necessitates assessing not only the correctness of the final answer but also the logical soundness of each reasoning step, a significant challenge for automated systems. Traditional methods are often inadequate: expert annotation is labor-intensive and unscalable for large-scale evaluation [49, 39], while automated techniques such as string matching or value equivalence overlook crucial step-by-step proof correctness [25, 23, 37, 38]. While LLMs have shown promise as evaluators (LLM-as-judge), their capacity for detailed, step-wise mathematical judgment is still developing. For instance, existing step-level LLM judges [68, 17] may assess general step validity but often lack the granularity to identify nuanced reasoning flaws. Similarly, frameworks like EIC-Math [30] provide broad error categories but can miss subtle yet critical issues in multi-step derivations. To address these limitations and assess informal mathematical proofs like inequality solving, our LLM-as-judge framework combines a final-answer judge with four step-wise judges targeting common errors: toy case overgeneralization, logical gaps, unjustified numeric approximations, and numeric calculation mistakes.

### 7 Conclusion

In summary, we introduce an informal yet verifiable task formulation for inequality proving, decomposing it into bound estimation and relation prediction. Building on this, we release IneqMath, an expert-curated benchmark of Olympiad-level inequalities with a training corpus featuring step-wise solutions and theorem annotations. Our novel LLM-as-judge evaluation framework, comprising a final-answer judge and four step-wise judges, enables a rigorous assessment. Our comprehensive evaluation of diverse leading LLMs reveals a critical gap: while LLMs may achieve high final-answer accuracy, this often plummets by up to 65.5% under step-wise scrutiny, with top models like o1 achieving less than 10% overall accuracy. This discrepancy exposes fragile deductive chains for current LLMs in constructing rigorous proofs. We further find that scaling model size or increasing test-time computation yields limited gains in overall proof correctness. Instead, our findings highlight promising research directions such as theorem-guided reasoning and self-refinement.

### Acknowledgments

This work is partially supported by the Hoffman-Yee Research Grants program at Stanford HAI and the AI for Math Fund by Renaissance Philanthropy. We thank Yu (Bryan) Zhou for early discussion and feedback.

### References

- [1] Tudor Achim, Alex Best, Kevin Der, Mathïs Fédérico, Sergei Gukov, Daniel Halpern-Leister, Kirsten Henningsgard, Yury Kudryashov, Alexander Meiburg, Martin Michelsen, et al. Aristotle: Imo-level automated theorem proving. arXiv preprint arXiv:2510.01346, 2025.
- [2] Meta AI. Llama 3.1 8b. https://huggingface.co/meta-llama/Llama-3.1-8B, 2024. Accessed: 2025-05-15.
- [3] Meta AI. Llama 3.2 3b. https://huggingface.co/meta-llama/Llama-3.2-3B, 2024. Accessed: 2025-05-15.
- [4] AI-MO / Project Numina & Kimi teams. Kimina-prover-distill-8b. https://huggingface. co/AI-MO/Kimina-Prover-Distill-8B, 2025.
- [5] AlibabaQwenTeam. QwQ-32B. https://huggingface.co/Qwen/QwQ-32B, 2025. Hugging Face model card.
- [6] Anthropic. Claude 3.7 Sonnet. https://www.anthropic.com/claude/sonnet, 2025. Anthropic model card.
- [7] Zhangir Azerbayev, Bartosz Piotrowski, Hailey Schoelkopf, Edward William Ayers, and Dragomir Radev. Proofnet: Autoformalizing and formally proving undergraduate-level mathematics, 2024.
- [8] Evan Chen. A brief introduction to olympiad inequalities, 2014. Accessed: 2025-03-19.
- [9] Luoxin Chen, Jinming Gu, Liankai Huang, Wenhao Huang, Zhicheng Jiang, Allan Jie, Xiaoran Jin, Xing Jin, Chenggang Li, Kaijing Ma, et al. Seed-prover: Deep and broad reasoning for automated theorem proving. arXiv preprint arXiv:2507.23726, 2025.
- [10] Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. TheoremQA: A theorem-driven question answering dataset. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7889–7901, Singapore, December 2023. Association for Computational Linguistics.
- [11] Leonardo Mendonça de Moura, Soonho Kong, Jeremy Avigad, Floris van Doorn, and Jakob von Raumer. The lean theorem prover (system description). In CADE, 2015.
- [12] DeepSeek-AI. DeepSeek-R1-Distill-Llama-70B. https://huggingface.co/deepseek-ai/ DeepSeek-R1-Distill-Llama-70B, 2025. Hugging Face model card.
- [13] DeepSeek-AI. DeepSeek-R1-Distill-Qwen-14B. https://huggingface.co/deepseek-ai/ DeepSeek-R1-Distill-Qwen-14B, 2025. Hugging Face model card.
- [14] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
- [15] Kefan Dong and Tengyu Ma. Beyond limited data: Self-play llm theorem provers with iterative conjecturing and proving. arXiv preprint arXiv:2502.00212, 2025.
- [16] Kefan Dong, Arvind V. Mahankali, and Tengyu Ma. Formal theorem proving by rewarding LLMs to decompose proofs hierarchically. In The 4th Workshop on Mathematical Reasoning and AI at NeurIPS’24, 2024.

- [17] Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. Omni-MATH: A universal olympiad level mathematic benchmark for large language models. In The Thirteenth International Conference on Learning Representations, 2025.
- [18] Fabian Gloeckle, Jannis Limperg, Gabriel Synnaeve, and Amaury Hayat. ABEL: Sample efficient online reinforcement learning for neural theorem proving. In The 4th Workshop on Mathematical Reasoning and AI at NeurIPS’24, 2024.
- [19] Google DeepMind. Gemini 2.0 Flash. https://cloud.google.com/vertex-ai/ generative-ai/docs/models/gemini/2-0-flash, 2025. Vertex AI model card.
- [20] Google DeepMind. Gemini 2.0 Flash-Lite. https://cloud.google.com/vertex-ai/ generative-ai/docs/models/gemini/2-0-flash-lite, 2025. Vertex AI model card.
- [21] Google DeepMind. Gemini 2.5 Flash. https://cloud.google.com/vertex-ai/ generative-ai/docs/models/gemini/2-5-flash, 2025. Vertex AI model card.
- [22] Google DeepMind. Gemini 2.5 Pro. https://deepmind.google/technologies/gemini/ pro/, 2025. Google DeepMind model card.
- [23] Aryan Gulati, Brando Miranda, Eric Chen, Emily Xia, Kai Fronsdal, Bruno de Moraes Dumont, and Sanmi Koyejo. Putnam-AXIOM: A functional and static benchmark for measuring higher level mathematical reasoning. In The 4th Workshop on Mathematical Reasoning and AI at NeurIPS’24, 2024.
- [24] Shailja Gupta, Rajesh Ranjan, and Surya Narayan Singh. A comprehensive survey of retrievalaugmented generation (rag): Evolution, current landscape and future directions. arXiv preprint arXiv:2410.12837, 2024.
- [25] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.
- [26] Jiewen Hu, Thomas Zhu, and Sean Welleck. miniCTX: Neural theorem proving with (long)contexts. In The Thirteenth International Conference on Learning Representations, 2025.
- [27] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.
- [28] PatrickLewis, EthanPerez, AleksandraPiktus, FabioPetroni, VladimirKarpukhin, NamanGoyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 9459–9474. Curran Associates, Inc., 2020.
- [29] Wenda Li, Lei Yu, Yuhuai Wu, and Lawrence C Paulson. Isarstep: a benchmark for high-level mathematical reasoning. In International Conference on Learning Representations, 2021.
- [30] Xiaoyuan Li, Wenjie Wang, Moxin Li, Junrong Guo, Yang Zhang, and Fuli Feng. Evaluating mathematical reasoning of large language models: A focus on error identification and correction. arXiv preprint arXiv:2406.00755, 2024.
- [31] Yang Li, Dong Du, Linfeng Song, Chen Li, Weikang Wang, Tao Yang, and Haitao Mi. Hunyuanprover: A scalable data synthesis framework and guided tree search for automated theorem proving. arXiv preprint arXiv:2412.20735, 2024.
- [32] Zenan Li, Zhaoyu Li, Wen Tang, Xian Zhang, Yuan Yao, Xujie Si, Fan Yang, Kaiyu Yang, and Xiaoxing Ma. Proving olympiad inequalities by synergizing LLMs and symbolic reasoning. In The Thirteenth International Conference on Learning Representations, 2025.

- [33] Zhenwen Liang, Linfeng Song, Yang Li, Tao Yang, Feng Zhang, Haitao Mi, and Dong Yu. Mps-prover: Advancing stepwise theorem proving by multi-perspective search and data curation. arXiv preprint arXiv:2505.10962, 2025.
- [34] Haohan Lin, Zhiqing Sun, Sean Welleck, and Yiming Yang. Lean-STar: Learning to interleave thinking and proving. In The Thirteenth International Conference on Learning Representations, 2025.
- [35] Yong Lin, Shange Tang, Bohan Lyu, Jiayun Wu, Hongzhou Lin, Kaiyu Yang, Jia Li, Mengzhou Xia, Danqi Chen, Sanjeev Arora, et al. Goedel-prover: A frontier model for open-source automated theorem proving. arXiv preprint arXiv:2502.07640, 2025.
- [36] Yong Lin, Shange Tang, Bohan Lyu, Ziran Yang, Jui-Hui Chung, Haoyu Zhao, Lai Jiang, Yihan Geng, Jiawei Ge, Jingruo Sun, Jiayun Wu, Jiri Gesi, Ximing Lu, David Acuna, Kaiyu Yang, Hongzhou Lin, Yejin Choi, Danqi Chen, Sanjeev Arora, and Chi Jin. Goedel-prover-v2: Scaling formal theorem proving with scaffolded data synthesis and self-correction, 2025.
- [37] Junnan Liu, Hongwei Liu, Linchen Xiao, Ziyi Wang, Kuikun Liu, Songyang Gao, Wenwei Zhang, Songyang Zhang, and Kai Chen. Are your llms capable of stable reasoning? arXiv preprint arXiv:2412.13147, 2024.
- [38] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024.
- [39] Yujun Mao, Yoon Kim, and Yilun Zhou. CHAMP: A competition-level dataset for fine-grained analyses of LLMs’ mathematical reasoning capabilities. In Findings of the Association for Computational Linguistics: ACL 2024, pages 13256–13274, Bangkok, Thailand, August 2024. Association for Computational Linguistics.
- [40] Meta Platforms, Inc. Llama-4-Maverick-17B-128E. https://huggingface.co/ meta-llama/Llama-4-Maverick-17B-128E, 2025. Hugging Face model card.
- [41] Meta Platforms, Inc. Llama-4-Scout-17B-16E. https://huggingface.co/meta-llama/ Llama-4-Scout-17B-16E, 2025. Hugging Face model card; accessed 2025-05-12.
- [42] Tobias Nipkow, Markus Wenzel, and Lawrence C. Paulson. Isabelle/HOL: A Proof Assistant for Higher-Order Logic. Springer, 2002.
- [43] OpenAI. GPT-4o. https://openai.com/index/hello-gpt-4o/, 2024. OpenAI blog post.
- [44] OpenAI. GPT-4o mini. https://openai.com/index/ gpt-4o-mini-advancing-cost-efficient-intelligence/, 2024. OpenAI blog post.
- [45] OpenAI. OpenAI o1. https://openai.com/o1/, 2024. OpenAI official announcement.
- [46] OpenAI. GPT-4.1. https://openai.com/index/gpt-4-1/, 2025. OpenAI model announcement.
- [47] OpenAI. Openai o3-mini system card, January 2025. Accessed: 2025-03-19.
- [48] OpenAI. OpenAI o4-mini. https://openai.com/index/ introducing-o3-and-o4-mini/, 2025. OpenAI official announcement.
- [49] Ivo Petrov, Jasper Dekoninck, Lyuben Baltadzhiev, Maria Drencheva, Kristian Minchev, Mislav Balunović, Nikola Jovanović, and Martin Vechev. Proof or bluff? evaluating llms on 2025 usa math olympiad. arXiv preprint arXiv:2503.21934, 2025.
- [50] Gabriel Poesia, David Broman, Nick Haber, and Noah Goodman. Learning formal mathematics from intrinsic motivation. Advances in Neural Information Processing Systems, 37:43032–43057, 2024.

- [51] Qwen Team. Qwen2.5-72B. https://huggingface.co/Qwen/Qwen2.5-72B, 2024. Hugging Face model card.
- [52] Qwen Team. Qwen2.5-7B. https://huggingface.co/Qwen/Qwen2.5-7B, 2024. Hugging Face model card.
- [53] Qwen Team. Qwen3-235B-A22B. https://huggingface.co/Qwen/Qwen3-235B-A22B,

2025. Hugging Face model card.

- [54] Z. Z. Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, Z. F. Wu, Zhibin Gou, Shirong Ma, Hongxuan Tang, Yuxuan Liu, Wenjun Gao, Daya Guo, and Chong Ruan. Deepseek-prover-v2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition, April 2025.
- [55] Z. Z. Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, Z. F. Wu, Zhibin Gou, Shirong Ma, Hongxuan Tang, Yuxuan Liu, Wenjun Gao, Daya Guo, and Chong Ruan. Deepseek-prover-v2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition, 2025.
- [56] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.
- [57] Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Lei Han, Haitao Mi, and Dong Yu. Toward self-improvement of LLMs via imagination, searching, and criticizing. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.
- [58] George Tsoukalas, Jasper Lee, John Jennings, Jimmy Xin, Michelle Ding, Michael Jennings, Amitayush Thakur, and Swarat Chaudhuri. Putnambench: Evaluating neural theorem-provers on the putnam mathematical competition. arXiv preprint arXiv:2407.11214, 2024.
- [59] Nguyen Duy Tung. 567 nice and hard inequalities, April 2012. Accessed: 2025-03-19.
- [60] Haiming Wang, Mert Unsal, Xiaohan Lin, Mantas Baksys, Junqi Liu, Marco Dos Santos, Flood Sung, Marina Vinyes, Zhenzhe Ying, Zekai Zhu, Jianqiao Lu, Hugues de Saxc’e, Bolton Bailey, Chendong Song, Chenjun Xiao, Dehao Zhang, Ebony Zhang, Frederick Pu, Han Zhu, Jiawei Liu, Jonas Bayer, Julien Michel, Longhui Yu, Léo Dreyfus-Schmidt, Lewis Tunstall, Luigi Pagani, Moreira Machado, Pauline Bourigault, Ran Wang, Stanislas Polu, Thibaut Barroyer, Wen-Ding Li, Yazhe Niu, Yann Fleureau, Yangyang Hu, Zhouliang Yu, Zihan Wang, Zhilin Yang, Zhengying Liu, and Jia Li. Kimina-prover preview: Towards large formal reasoning models with reinforcement learning. arXiv preprint arXiv:2504.11354, 2025.
- [61] Haiming Wang, Huajian Xin, Chuanyang Zheng, Zhengying Liu, Qingxing Cao, Yinya Huang, Jing Xiong, Han Shi, Enze Xie, Jian Yin, Zhenguo Li, and Xiaodan Liang. LEGO-prover: Neural theorem proving with growing libraries. In The Twelfth International Conference on Learning Representations, 2024.
- [62] Ruida WANG, Rui Pan, Yuxin Li, Jipeng Zhang, Yizhen Jia, Shizhe Diao, Renjie Pi, Junjie Hu, and Tong Zhang. MA-lot: Model-collaboration lean-based long chain-of-thought reasoning enhances formal theorem proving. In Forty-second International Conference on Machine Learning, 2025.
- [63] Chenrui Wei, Mengzhou Sun, and Wei Wang. Proving olympiad algebraic inequalities without human demonstrations. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.
- [64] Yuhuai Wu, Albert Jiang, Jimmy Ba, and Roger Baker Grosse. {INT}: An inequality benchmark for evaluating generalization in theorem proving. In International Conference on Learning Representations, 2021.
- [65] Zijian Wu, Suozhi Huang, Zhejian Zhou, Huaiyuan Ying, Jiayu Wang, Dahua Lin, and Kai Chen. Internlm2. 5-stepprover: Advancing automated theorem proving via expert iteration on large-scale lean problems. arXiv preprint arXiv:2410.15700, 2024.

- [66] xAI. Grok 3. https://x.ai/news/grok-3, 2025. xAI official announcement.
- [67] xAI. Grok 3 Mini. https://x.ai/news/grok-3, 2025. xAI official announcement.
- [68] Shijie Xia, Xuefeng Li, Yixin Liu, Tongshuang Wu, and Pengfei Liu. Evaluating mathematical reasoning beyond accuracy. CoRR, abs/2404.05692, 2024.
- [69] Huajian Xin, Daya Guo, Zhihong Shao, Z.Z. Ren, Qihao Zhu, Bo Liu, Chong Ruan, Wenda Li, and Xiaodan Liang. Advancing theorem proving in LLMs through large-scale synthetic data. In The 4th Workshop on Mathematical Reasoning and AI at NeurIPS’24, 2024.
- [70] Ran Xin, Chenguang Xi, Jie Yang, Feng Chen, Hang Wu, Xia Xiao, Yifan Sun, Shen Zheng, and Kai Shen. Bfs-prover: Scalable best-first tree search for llm-based automatic theorem proving. arXiv preprint arXiv:2502.03438, 2025.
- [71] Ran Xin, Zeyu Zheng, Yanchen Nie, Kun Yuan, and Xia Xiao. Scaling up multi-turn off-policy rl and multi-agent tree search for llm step-provers. arXiv preprint arXiv:2509.06493, 2025.
- [72] Yu Xuejun, Jianyuan Zhong, Zijin Feng, Pengyi Zhai, Roozbeh Yousefzadeh, Wei Chong Ng, Haoxiong Liu, Ziyi Shou, Jing Xiong, Yudong Zhou, et al. Mathesis: Towards formal theorem proving from natural languages. arXiv preprint arXiv:2506.07047, 2025.
- [73] Kaiyu Yang and Jia Deng. Learning to prove theorems via interacting with proof assistants. In International Conference on Machine Learning, pages 6984–6994. PMLR, 2019.
- [74] Kaiyu Yang, Aidan M Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan Prenger, and Anima Anandkumar. Leandojo: Theorem proving with retrieval-augmented language models. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.
- [75] Ziyu Ye, Jiacheng Chen, Jonathan Light, Yifei Wang, Jiankai Sun, Guohao Li, Mac Schwager, Philip Torr, Yuxin Chen, Kaiyu Yang, Yisong Yue, and Ziniu Hu. Reasoning in reasoning: A hierarchical framework for (better and faster) neural theorem proving. In The 4th Workshop on Mathematical Reasoning and AI at NeurIPS’24, 2024.
- [76] Huaiyuan Ying, Zijian Wu, Yihan Geng, JIayu Wang, Dahua Lin, and Kai Chen. Lean workbook: A large-scale lean problem set formalized from natural language math problems. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.
- [77] Zhouliang Yu, Ruotian Peng, Keyi Ding, Yizhe Li, Zhongyuan Peng, Minghao Liu, Yifan Zhang, Zheng Yuan, Huajian Xin, Wenhao Huang, Yandong Wen, Ge Zhang, and Weiyang Liu. Formalmath: Benchmarking formal mathematical reasoning of large language models, 2025.
- [78] Mert Yuksekgonul, Federico Bianchi, Jack Boen, et al. Optimizing generative ai by backpropagating language model feedback. Nature, 639:609–616, 2025. Published 19 March 2025.
- [79] Ziyin Zhang, Jiahao Xu, Zhiwei He, Tian Liang, Qiuzhi Liu, Yansi Li, Linfeng Song, Zhengwen Liang, Zhuosheng Zhang, Rui Wang, et al. Deeptheorem: Advancing llm reasoning for theorem proving through natural language and reinforcement learning. arXiv preprint arXiv:2505.23754, 2025.
- [80] Haoyu Zhao, Yihan Geng, Shange Tang, Yong Lin, Bohan Lyu, Hongzhou Lin, Chi Jin, and Sanjeev Arora. Ineq-Comp: Benchmarking human-intuitive compositional reasoning in automated theorem proving on inequalities. arXiv preprint arXiv:2505.12680, 2025.
- [81] Zhiyu Zhao, Yongcheng Zeng, Ning Yang, and Guoqing Liu. Enhancing mathematical reasoning in language models through focused differentiation training, 2025.
- [82] Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. minif2f: a cross-system benchmark for formal olympiad-level mathematics. In International Conference on Learning Representations, 2022.

## Supplementary Materials for Solving Inequality Proofs with Large Language Models

### Appendix Contents

- A Dataset Curation Details 17

- A.1 Training data curation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.2 Data Annotation Tool . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.3 Prompts for Rephrasing Problems . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.4 Benchmark Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- B Fine-grained Informal Judge Details 22

- B.1 Final Answer Judge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- B.2 Toy Case Judge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- B.3 Logical Gap Judge . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- B.4 Numerical Approximation Judge . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- B.5 Numerical Computation Judge . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- B.6 Development Performance of Judges . . . . . . . . . . . . . . . . . . . . . . . . . 29
- B.7 Judge Failure Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30

- C Experimental Details for Inequality Solving 34

- C.1 Experimental Setups . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- C.2 Model Failure Solution Examples . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- C.3 Taking Annotated Theorems as Hints . . . . . . . . . . . . . . . . . . . . . . . . . 40
- C.4 Retrieval as Augmentation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- C.5 Self-improvement via Critic as Feedback . . . . . . . . . . . . . . . . . . . . . . . 45
- C.6 Few-shot Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
- C.7 Evaluation on the Formalized IneqMath . . . . . . . . . . . . . . . . . . . . . . . 47
- C.8 Memorization Probe . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48

- D Limitations 49
- E Broader Impacts 50

### A Dataset Curation Details

#### A.1 Training data curation.

Training problems were sourced from two advanced textbooks featuring graduate-level and Olympiadstyle inequality proof problems. We parsed these textbooks to extract proof problems, their step-wise solutions, and relevant theorems. We developed two LLM-based rephrasers to transform each source problem into two sub-tasks defined in §2: bound estimation and relation prediction. For instance, a source problem like “Prove a+b ≥ 2

√

ab for ∀a,b ∈ R+” would be rephrased into a bound estimation task (e.g., “Determine the maximal constant C such that a + b ≥ C

√

ab holds for ∀a,b ∈ R+”) and a relation prediction task (e.g., “Determine the inequality relation in the expression a + b ( ) 2

√

ab that holds for ∀a,b ∈ R+”).

Crucially, while rephrased problems are altered from the source proof problem in the format, they preserve the core mathematical reasoning and solution steps—such as applying relevant theorems, determining boundary conditions, and verifying inequalities. An annotation tool (see §A.2) was developed to facilitate human expert review and correction of the LLM-rephrased problems. Extracted theorems were curated, each including its name, a natural language definition, and a list of training problems where it is applicable.

#### A.2 Data Annotation Tool

[Figure 36]

- Figure 11: The interface of our developed tool for checking and editing the bound problems.

[Figure 37]

- Figure 12: The interface of our developed tool for checking and editing the relation problems.

- A.3 Prompts for Rephrasing Problems

Prompt for Rephrasing Proofs to Bound Problems

Task: Transform the given inequality problem into a bound prediction problem by introducing a constant C and determining its optimal value.

###### Instructions:

- 1. Analyze the original problem, focusing on its structure and potential for transformation.
- 2. Introduce a constant C by either replacing an existing constant or creating a new relationship between expressions.
- 3. Determine whether to find the minimal or maximal value of C that satisfies the inequality for all relevant variables.
- 4. Consider factors such as homogeneity, existing constraints, and the domain of variables (e.g., positive reals, all reals).
- 5. Ensure the rephrased problem maintains the mathematical essence and constraints of the original.

Output Format: Provide your response in the following structure: <Analysis>: Concise explanation of key features and transformation approach. <Conclusion>: YES or NO, followed by a brief summary of the transformation. <Rephrased Problem>: Transformed problem statement, focusing on finding the optimal C. <Answer>: C =< value >.

###### Key Considerations:

- 1. For double inequalities, attempt to rephrase as a single bound prediction problem when possible.
- 2. In homogeneous inequalities, focus on the ratios between variables rather than their absolute values.
- 3. Incorporate any existing constraints into the rephrased version of the problem.
- 4. Clearly specify the domain of the variables in the rephrased problem statement.
- 5. Ensure that the rephrased problem is logically equivalent to the original. Example:

Original problem: Let a, b, c ∈ R+. Prove the inequality

abc (1 + a)(a + b)(b + c)(c + 16) ≤

1 81

<Analysis>: To turn this into a bound prediction problem, we can focus on the following steps:

- 1. The left side is a rational expression that is always positive for a, b, c ∈ R+.
- 2. The right side is a fixed constant 811 .

- 3. We replace the constant 811 with a variable C and ask: What is the smallest C such that the inequality holds for all positive a, b, c?

- 4. This approach allows us to determine the tightest possible upper bound for the left-hand expression.
- 5. If we find the smallest C that works, we prove the original inequality and show it’s the best possible.

<Conclusion>: YES, the inequality can be rephrased as a bound prediction problem. By replacing the constant 811 with a variable C, we can determine the tightest upper bound for the given rational expression, effectively proving the original inequality and demonstrating its optimality.

<Rephrased problem>: Coll Determine the minimal constant C such that the following inequality holds for all a, b, c:

abc (1 + a)(a + b)(b + c)(c + 16) ≤ C.

<Answer>: C = 811 . Now, please rewrite the following problem: Original problem: {problem}

#### Prompt for Rephrasing Proofs to Relation Problems

Task: Transform the given inequality proof problem into a relation prediction problem. Instructions:

- 1. Analyze the original problem, identifying key components such as variables, domains, conditions, and the main inequality.
- 2. Rephrase the problem by maintaining the original expressions and replacing the relation symbol with a blank to be filled.
- 3. Preserve any additional conditions or constraints from the original problem in your rephrased version.
- 4. Change the task from “Prove” to “Determine the correct inequality relation to fill in the blank.”
- 5. Provide a set of options for the relation, always including ≤, ≥, =, <, >, and “None of the above”.
- 6. Determine the correct answer based on your modification and analysis.

Output Format: Provide your response in the following structure:

<Analysis>: Detailed step-by-step analysis of the original problem and your approach to rephrasing it. <Conclusion>: YES or NO, followed by a brief explanation of whether and how the problem can be effectively rephrased. <Rephrased Problem>:

Transformed problem statement. Options: (A) ≤ (B) ≥ (C) = (D) < (E) > (F) None of the above

<Answer>: Option letter. Key Considerations:

- 1. Maintain the original mathematical expressions and any given conditions as much as possible.
- 2. Ensure the rephrased problem captures the essence and complexity of the original problem.
- 3. For problems with multiple inequalities, focus on one main inequality for the relation prediction task.
- 4. When dealing with complex fractions or expressions, keep them intact to maintain the problem’s difficulty level.
- 5. If the relation depends on specific values of the variables or cannot be definitively determined, consider using “None of the above” as the correct answer.

Example: Original problem: Let a, b, c ∈ R+. Prove the inequality

1 81

abc (1 + a)(a + b)(b + c)(c + 16) ≤

<Analysis>: To rephrase it to a relation prediction problem, we can focus on the following steps:

- 1. The original problem is a proof task for an inequality involving positive real numbers a, b, and c.
- 2. The left side of the inequality is a complex fraction (1+a)(a+babc)(b+c)(c+16).

- 3. The right side is a constant fraction 811 .

- 4. The original inequality uses the “less than or equal to” (≤) relation, which needs to hold for all positive real values of a, b, and c.
- 5. We can transform the proof task into determining the correct relation between the left and right sides of the inequality. <Conclusion>: YES, the inequality can be effectively rephrased as a relation prediction problem.

<Rephrased problem>: Let a, b, c ∈ R+. Consider the following inequality:

abc (1 + a)(a + b)(b + c)(c + 16)

1 81

( )

.

Determine the correct inequality relation to fill in the blank. Options: (A) ≤ (B) ≥ (C) = (D) < (E) > (F) None of the above

<Answer>: A Now, please rewrite the following problem: Original problem: {problem}

- A.4 Benchmark Examples

- IneqMath Testing Example 1: Bound Problem

Question: Let x,y,z > 0 such that x + y + z = 1. Determine the minimal constant C such that the following inequality holds for all x,y,z:

xy(y + 4z) + yz(z + 4x) + zx(x + 4y) ≤ C.

- IneqMath Testing Example 2: Bound Problem

Question: Let a1,a2,...,an be real numbers and S be a non-empty subset of {1,2,...,n}. Find the largest constant C such that the following inequality holds for all a1,a2,...,an and S:

2C

i∈S

ai

2

≤

1≤i≤j≤n

(ai + ··· + aj)2 .

- IneqMath Testing Example 3: Bound Problem

Question: Let a1,a2,...,an > 0 such that a1 + a2 + ... + an < 1. Determine the minimal constant C such that the following inequality holds for all a1,a2,...,an:

a1 · a2 ...an (1 − a1 − a2 − ... − an) (a1 + a2 + ... + an)(1 − a1)(1 − a2)...(1 − an) ≤ C

3 nn−1.

#### IneqMath Testing Example 4: Relation Problem

Question: Let a,b,c,x,y,z ∈ R be real numbers such that a+b+c = 1 and x2 +y2 +z2 = 1. Consider the following expression:

a(x + b) + b(y + c) + c(z + a) ( ) 1. Determine the correct inequality relation to fill in the blank. Options: (A) ≤ (B) ≥ (C) = (D) < (E) > (F) None of the above

#### IneqMath Testing Example 5: Relation Problem

Question: In the plane of the acute-angled triangle △ABC, let L be a line such that u,v,w are the lengths of the perpendiculars from A,B,C respectively to L. Consider the following inequality:

u2 tanA + v2 tanB + w2 tanC ( ) ∆. where ∆ is the area of the triangle. Determine the correct inequality relation to fill in the blank. Options: (A) ≤ (B) ≥ (C) = (D) < (E) > (F) None of the above

###### IneqMath Testing Example 6: Relation Problem Question: Let a,b,c be the sides of any triangle. Consider the following inequality:

(c2 + ab(1 + 2cos(c)))(b2 + ac(1 + cos(b))) .

3

ab(1 + 2cos(c)) ( ) 2

cyc

cyc

Determine the correct inequality relation to fill in the blank. Options: (A) ≤ (B) ≥ (C) = (D) < (E) > (F) None of the above

### B Fine-grained Informal Judge Details

- B.1 Final Answer Judge

Prompt for Final Answer Judge: Answer Extraction for Bound problems

You are an expert in extracting numbers from answer sentences. Below are examples of sentences and the corresponding numbers:

- Example 1: answer is C = 2. Answer: C = 2
- Example 2: answer is C = √12. Answer: C = √12

- Example 3: answer is

|C = 2|
|---|

.

Answer: C = 2 Now, extract the number from the following sentence: {answer_sentence}. Make sure to return the answer in the format as “C=<extracted_answer>”, where <extracted_answer> is the extracted number or expression.

#### Prompt for Final Answer Judge: Answer Extraction for Relation Problems

You are an expert in extracting option letters (A, B, C, D, E, F) from answer sentences. The options are given below:

- A: (A) ≤
- B: (B) ≥
- C: (C) =
- D: (D) <
- E: (E) >
- F: (F) None of the above Below are examples of sentences and the corresponding option letters:

- Example 1: answer is (B) ≥. Answer: B
- Example 2: answer is (E) >. Answer: E
- Example 3: answer is:

.

|≤|
|---|

Answer: A Now, extract the option letter from the following sentence: {answer_sentence}. Make sure to return the option letter only, without any other characters.

#### Prompt for Final Answer Judge: Answer Equivalence Verification

You are an expert in verifying mathematical expression equivalence. Analyze if two expressions are exactly equivalent by following these strict rules:

###### Required Analysis Steps:

- 1. Check if both expressions are valid mathematical forms.
- 2. If either expression is not mathematical (e.g., text or undefined), return False.
- 3. For numerical expressions:

- - Direct equality (e.g., 2 = 2) → True.

- - Different representations of same value (e.g., 1/2 = 0.5, √4 = 2) → True.

- - Decimal approximations vs exact values (e.g., 2π ̸= 6.28318) → False.

- 4. For algebraic expressions:

- - Must have clear, valid transformation path between forms.
- - If transformation requires multiple non-obvious steps → False.
- - Verify equivalence through algebraic proof when possible.
- - For complex expressions, use techniques like squaring or substitution to verify.

###### Equivalence Criteria:

- - Must have exactly same deterministic value.
- - Must be provably equivalent through valid mathematical operations.
- - Different notations of same exact value are equivalent.
- - Decimal approximations are NOT equivalent to exact expressions.
- - No rounding or approximations allowed.
- - If equivalence cannot be conclusively proven → False.

Example Pairs and their Analysis: Ground truth: C = 2 Prediction: C = 2 Analysis: The expressions are identical in both form and value, representing the same integer 2. Equivalent: True

- Ground truth: C = 1.5 Prediction: C = 23 Analysis: The decimal 1.5 and fraction 32 are different representations of the same number (1.5 = 32). Equivalent: True

- Ground truth: C = 2π Prediction: C = 6.28318530718 Analysis: While 6.28318530718 is a decimal approximation of 2π, they are not symbolically equivalent expressions. Equivalent: False

Ground truth: C = 16 Prediction: C = √16 Analysis: These are equivalent through the property ab =

√a √

b when a, b > 0. Equivalent: True

Ground truth: C = 32 Prediction: C = 2√32 Analysis: The expressions differ as proven when squared: ( 32)2 = 23 ̸= 98 = (2√32)2. Equivalent: False

Now analyze these expressions: Ground truth: {ground_truth} Prediction: {prediction}

- B.2 Toy Case Judge

Prompt for Toy Case Judge

Task: Evaluate the logical rigor of a solution to an inequality problem, focusing specifically on whether the direction of the inequality was justified using toy cases or special value substitution.

###### Instructions:

- 1. Carefully read the reasoning process used to solve the inequality.
- 2. Identify whether the direction of the inequality was determined by testing special values, trying toy cases, or relying on extreme-case analysis, rather than providing a general proof valid over the entire domain.

- 3. If the model uses a toy case (e.g., setting a variable to 0, 1, or choosing symmetric/equal values) or considers a variable tending to 0 or ∞ (extreme-case reasoning) to conclude the inequality direction, this should be flagged as logically unsound unless it is later supported by a rigorous or general argument.
- 4. Substituting special values for the purpose of verifying equality or testing sharpness is acceptable and should not be flagged.
- 5. If a toy case is used to show that the inequality does not hold (i.e., the two sides are incomparable), this is acceptable and should not be flagged.
- 6. Trying toy cases or substituting special values for the purpose of exploring or analyzing the problem—without using them to directly conclude the inequality direction—is acceptable and should not be flagged.
- 7. The goal is to confirm whether the final conclusion is justified for all variables in the given domain by using sound and formal reasoning.

Output Format: <Analysis>: Brief explanation of whether toy cases, special values, or extreme-case reasoning were used to justify inequality direction, and whether this undermines the rigor of the proof. <Flagged Reasoning Step (if applicable)>: Quote or summarize the step(s) where toy cases or extreme cases were used to justify the inequality.

<Answer>: True or False. It indicates whether the solution’s reasoning is logically valid (True) or contains flaws due to unjustified use of toy cases, special value substitution, or extreme-case reasoning (False).

###### Key Considerations:

- 1. Check which steps are used to justify the final general conclusion.
- 2. Toy case examples should not be used to conclude general inequality directions.
- 3. These techniques are allowed for illustration or equality testing, but not as a replacement for a full justification.
- 4. Extreme-case analysis used as a substitute for proof to justify the final general conclusion should be flagged.
- 5. Do not provide improvement suggestions—simply judge whether the logic used is valid or flawed.
- 6. Only reasoning steps that fail to generalize and are used to justify the final general conclusion should be flagged.

Examples of Inputs and Outputs: {examples}

Now analyze the following problem and solution: Original Problem: {query} Solution: {response}

Judge Example 1: Toy Case Response: Let’s test the specific values.

- Case 1: If a = b = c, then: 2

- a

- b + c

+

- b

- c + a

+

c a + b

2

=

1 4

+

- 1

- 2

+

1 4

= 1.

- Case 2: If a = 0 and b = c = 1, then:

2

2

- a

- b + c

- b

- c + a

c a + b

+

+

= 0 + 1 + 1 = 2.

From the test cases, we can conclude that the minimum value of the expression was 1. Therefore, the expression is always at least 1. The answer is

.

|C = 1|
|---|

Judge Output: <Analysis>: The minimum of the expression is derived from two special cases(i.e. a = b = c and a = 0, b = c = 1). This is incorrect as it substitutes generality with selected toy cases.

<Flagged Reasoning Step (if applicable)>: From the test cases, we can conclude that the minimum value of the expression is 1. Therefore, the expression is always at least 1.

|False|
|---|

<Answer>:

- B.3 Logical Gap Judge

Prompt for Logical Gap Judge

Task: Evaluate the logical rigor of a proposed solution to a mathematical inequality problem. Focus on whether the reasoning includes non-trivial claims made without justification, logical leaps, or unsupported assertions, while allowing for valid optimization-based, algebraic, or analytic analysis when properly demonstrated. This prompt does not evaluate whether the direction of an inequality was justified using toy cases, special values, or asymptotic behavior; that aspect is handled separately.

###### Instructions:

- 1. Carefully read the entire reasoning process used to solve the inequality.
- 2. Identify whether the solution includes:

- - Any non-obvious (non-trivial) claims or transformations without justification or explanation.
- - Any logical gaps or skipped steps that lead to intermediate or final conclusions.

- 3. All significant transformations—especially involving inequalities, bounds, or extremal behavior—must be supported by: algebraic manipulation, well-known identities or theorems, valid analytical tools (e.g., convexity, derivatives, limits) or step-by-step numeric or symbolic reasoning.
- 4. Optimization methods (e.g., Lagrange multipliers, derivative-based analysis) are valid only if the analysis is explicitly shown:

- - If a solution invokes optimization or analytical techniques, it must demonstrate key steps, derivative conditions, or critical point verification.
- - Statements such as “solving the constrained optimization problem confirms...” without any derivation or argument are considered unjustified.
- - You do not need to assess whether toy cases, special values, or extreme behavior were used to infer the inequality direction. That responsibility lies outside the scope of this Judge.

- 5. Simple algebra or widely known transformations (e.g., AM-GM, factoring identities) may be used without full derivation.
- 6. The goal is to assess whether each important conclusion within the reasoning—not just the final answer—is logically supported and rigorously justified.

###### Output Format:

<Analysis>: Step-by-step explanation of whether the reasoning is logically sound. Highlight any unjustified claims or skipped steps, unless they are supported by valid asymptotic, numeric, or analytic reasoning.

<Flagged Reasoning Step (if applicable)>: Quote or summarize the specific step(s) where an unjustified claim or logical leap occurred. <Answer>: True or False. True if the reasoning is valid; False if it contains unjustified steps or unsupported claims.

###### Key Considerations:

- 1. Focus on whether each major step (not just the final answer) is logically justified.
- 2. Non-trivial algebraic identities or inequality steps must be explained unless they are well-known.
- 3. Minor simplifications and standard techniques are acceptable without proof.
- 4. Do not flag steps involving toy cases, extreme values, or special substitutions used to infer inequality direction—those are out of scope here.
- 5. Claims like “a numerical check shows” must include specific values, results, or graphs to be valid.
- 6. Optimization-based arguments (e.g., Lagrange multipliers, critical point methods) must include demonstrated steps or analytic structure. If only the method is named but not applied, the reasoning should be flagged.
- 7. Do not flag steps used solely for equality verification, sharpness testing, or illustration.
- 8. Do not provide improvement suggestions—simply judge whether the logic is valid or flawed as presented.

###### Examples of Inputs and Outputs: {examples}

Now analyze the following problem and solution: Original Problem: {query} Solution: {response}

Judge Example 2: Logical Gap Response:

... A numerical check confirms that the minimum occurs when x = 1, and the expression equals 4. Therefore, the largest constant C such that the inequality holds for all positive a,b,c is:

|C = 4|
|---|

Judge Output:

<Analysis>: The statement “A numerical check confirms that the minimum occurs when x = 1” is made without showing any actual numerical data, evaluations, or visual/analytic evidence to support this claim.

<Flagged Reasoning Step (if applicable)>: A numerical check confirms that the minimum occurs when x = 1.

|False|
|---|

<Answer>:

- B.4 Numerical Approximation Judge

Prompt for Numerical Approximation Judge

Task: Evaluate the logical rigor of a solution to an inequality problem, specifically checking whether approximate numerical substitutions (e.g., replacing √2 with 1.414) were improperly used in the reasoning process.

###### Instructions:

- 1. Carefully read the entire reasoning process used to solve the inequality.
- 2. Identify whether the solution includes:

- Any replacement of exact expressions (such as radicals, fractions, or constants like π) with approximate decimal values.

- 3.Strict rules for use of approximate values:

- - If approximated values are directly involved in any operations (such as addition, subtraction, multiplication, or division), it must immediately be considered invalid, regardless of whether the operation is for comparing sizes or for further reasoning!
- - Examples of invalid actions: Approximating √5 ≈ 2.236 and then using it to compute √5 + 3 approximately, or Approximating π ≈ 3.14 and then evaluating π/2 based on 3.14.

- 4. Approximate substitutions are allowed only under the following conditions: If approximate numerical comparison is used between simple numbers (e.g., √4, 12, √2) that humans can readily estimate, it is acceptable.

- 5. Approximate substitution is invalid and must be flagged in these cases:

- - If an approximate value is introduced for a complex irrational number (e.g., √17, √23) where human mental estimation is impractical, even for comparison purposes.

- - If any approximation alters the rigor of the argument.

- 6. You do not need to judge whether the final inequality direction is correct—only whether improper approximation substitution occurred.

###### Output Format:

<Analysis>: Step-by-step explanation of whether approximate numerical values were improperly substituted for exact expressions. Clarify whether approximations were used only illustratively or improperly incorporated into reasoning.

<Flagged Reasoning Step (if applicable)>: Quote or summarize the specific step(s) where inappropriate approximations were made. <Answer>: True or False. True if the reasoning maintains acceptable rigor regarding approximations;

False if it violates the rules. Key Considerations:

- 1. If approximated values are directly involved in any operations (such as addition, subtraction, multiplication, or division), it must immediately be considered invalid, regardless of whether the operation is for comparing sizes or for further reasoning.
- 2. Comparing simple, common values via approximation (e.g., √2 ≈ 1.414 vs 1.5) is acceptable if human estimation is reasonable.

- 3. Approximate values of complicated irrational numbers (e.g., √17, √23, π5/4) are invalid even for comparison.

- 4. Any use of approximate values for calculations (such as adding, subtracting, multiplying, or dividing approximate numbers) is strictly invalid, even if the final result seems close. Only comparing two simple exact numbers by approximation is acceptable; calculating further with approximated values is never allowed.
- 5. Widely known exact simplifications (e.g., √4 = 2, 12 = 0.5) are acceptable.

- 6. Do not suggest improvements—simply judge whether the solution follows the rules.

Examples of Inputs and Outputs: {examples}

Now analyze the following problem and solution: Original Problem: {query} Solution: {response}

Judge Example 3: Numerical Approximation Response:

... When A = 89◦, B = C = 45.5◦:

cos89◦ ≈ 0.01745 cos45.5◦ ≈ 0.7009

Compute:

cosA cosB cosC

0.01745 0.70092 ≈

0.01745 0.4913 ≈ 0.0355

=

- cosB

- cosC cosA

0.7009 0.7009 · 0.01745

0.7009

0.01223 ≈ 57.33 cosC cosAcosB

=

=

0.7009 0.01745 · 0.7009

0.7009

0.01223 ≈ 57.33 So the sum:

=

=

S ≈ 0.0355 + 57.33 + 57.33 ≈ 114.7 > 114

which concludes that S > 114.

#### ... Judge Output:

<Analysis>: The above computation relies on approximate decimal values for trigonometric functions. Using such coarse approximations—especially in the denominator—can lead to significant magnification of numerical error. So comparing the approximation of S with 114 to conclude the relationship with S and 114 is not mathematically rigorous.

<Flagged Reasoning Step (if applicable)>: So the sum S ≈ 0.0355 + 57.33 + 57.33 ≈ 114.7 > 114, which concludes that S > 114.

|False|
|---|

<Answer>:

- B.5 Numerical Computation Judge

Prompt for Numerical Computation Judge

Task: Evaluate the correctness of numerical computations in a solution to a mathematical inequality problem. Focus on verifying whether each calculation step is numerically valid, allowing for some error tolerance.

###### Instructions:

- 1. Carefully read through the entire solution.
- 2. Identify all numerically verifiable expressions, including:

- Exact value computations when variables are assigned specific numbers, allowing for floating-point operations.

- 3. Do not extract:

- - Symbolic manipulations or transformations.
- - Expressions involving symbolic variables or operations.
- - Inequalities; only extract equations.
- - Approximate equations (e.g., using “approximately equal to”).

- 4. Python validation rules:

- - Convert all operations to floating-point calculations.
- - Allow a small tolerance for numerical comparisons (e.g., “abs(lhs - rhs) < 1e-2”).
- - Set the final result of each check to a variable “answer”, where “answer = True” if the verification

succeeds, otherwise False.

- 5. Final output format:

- - Extracted expressions: List all selected expressions in natural math form.
- - Analysis: Clearly state why every expression meets the criteria for verification or doesn’t.
- - Validation Codes: Provide Python code snippets to verify each expression.

Output Format: <Analysis>: Reason why the expression needs to be verified and extracted or not. <Expressions>: List(Expression 1, Expression 2, ...) <Codes>:

- # Validation for Expression 1 import sympy as sp # your checking code here
- # Validation for Expression 2 import sympy as sp # your checking code here

###### Key Considerations:

- - Only extract expressions involving explicit numerical computations.
- - All selected expressions must be verifiable numerically, allowing for some error tolerance.
- - Use floating-point operations for all checks.
- - Do not extract approximate equations.
- - Do not extract expressions involving symbolic manipulations or transformations.

Examples: {examples}

Now analyze the following problem and solution: Original problem: {query} Solution: {response}

Judge Example 4: Numerical Computation Response:

... Based on the previous steps,

27 t3

- 2

- 3

φ(t) = t +

+

,

81 t4

φ′(t) = 1 −

,

So φ′(t) ≥ 0 for t ≥ 3 with equality only at t = 3. Hence φ is increasing on [3,∞) and attains its minimum at t = 3:

27 27

- 2

- 3

φ(3) = 3 +

+

= 4.

Therefore F(a,b,c) ≥ 4 for every positive a,b,c, and equality occurs when t = 3.

...

<Analysis>: φ(3) = 3 + 2727 + 23 = 4 is the only numerical expression without any variables. So this is the only expression that should be extracted.

<Expressions>: List(φ(3) = 3 + 2727 + 23 = 4) <Codes>:

# Validation for Expression 1: φ(3) = 3 + 27/27 = 4 from sympy import Rational phi_val = Rational(3) + Rational(27, 27) answer = (phi_val == 4)

#### B.6 Development Performance of Judges

Metrics GPT-4o mini GPT-4o GPT-4.1 mini GPT-4.1 o3-mini o4-mini o1 o3 Final Answer Judge

Accuracy (%) 100.0 - - - - 100.0 - Precision (%) 100.0 - - - - 100.0 - Recall (%) 100.0 - - - - 100.0 - F1 score 1.0 - - - - 1.0 - -

###### Toy Case Judge

Accuracy (%) 80.0 86.3 88.8 90.0 91.3 91.3 80.0 91.3 Precision (%) 89.3 84.6 82.2 87.5 87.8 86.0 71.2 90.0 Recall (%) 65.8 86.8 97.4 92.1 94.7 97.4 97.4 92.1 F1 score 0.76 0.86 0.89 0.90 0.91 0.91 0.82 0.91

###### Logical Gap Judge

- Accuracy (%) 64.6 71.3 78.8 75.0 80.0 96.3 72.2 90.8 Precision (%) 82.4 63.5 71.0 67.9 75.0 95.1 64.4 97.0 Recall (%) 35.9 100.0 97.5 95.0 90.0 97.5 97.4 84.2 F1 score 0.50 0.78 0.82 0.79 0.82 0.96 0.78 0.90

Numerical Approximation Judge

Accuracy (%) 80.0 71.3 60.0 87.5 72.5 87.5 96.3 96.3 Precision (%) 82.1 63.9 56.2 80.4 65.1 83.0 95.2 95.2 Recall (%) 78.0 97.5 100.0 100.0 100.0 95.1 97.6 97.6 F1 score 0.80 0.77 0.72 0.89 0.79 0.89 0.96 0.96

Numerical Calculation Judge

- Accuracy (%) 65.0 62.5 70.9 61.3 58.2 73.7 69.4 71.3 Precision (%) 64.2 62.0 67.6 60.3 58.2 69.2 67.2 67.1 Recall (%) 91.5 93.6 97.9 100.0 100.0 100.0 100.0 100.0 F1 score 0.75 0.75 0.80 0.75 0.74 0.82 0.80 0.80

Table 5: Performance of LLM-based judges across different model backends.

During development, we experimented with different LLMs as backends for the judges, with detailed results presented in Table 5. By balancing F1 score, inference cost, and latency, we select GPT-4o mini for the final-answer judge due to its perfect F1 score (1.0) and the lowest cost among candidates. For the four step-wise judges, we use o4-mini for the Toy Case Judge and Logical Gap Judge, o1 for the Numerical Approximation Judge, and GPT-4.1 mini for the Numerical Calculation Judge, which attains F1 scores above 0.9 on average.

- B.7 Judge Failure Examples

- Judge Failure Example 1: Toy Case

Problem: Let a,b,c be three non-negative real numbers. Find the largest constant C such that the following inequality holds for all a,b,c ≥ 0:

a2 + 1 b2 + 1 c2 + 1 ≥ (ab + bc + ca − abc)2 + C · abc

Response Snippet: Let a = b = c = t, the inequality becomes:

(t2 + 1)3 ≥ (3t2 − t3)2 + Ct3

After expanding the both sides and dividing both sides by t3 (for t > 0):

6t2 − 6t +

3 t

+

1 t3 ≥ C

Now, let’s minimize f(t) = 6t2 − 6t + 3t + t13 for t > 0.

... (Checking the minimum) So, at t = 1, f(1) = 4 is indeed the minimum. From the above, for all other values of t, f(t) > 4, so the final answer is

|C = 4|
|---|

. Expected Evaluation for the Judge:

The solution applies a toy-case simplification by assuming a = b = c = t, and analyzes f(t) to draw conclusions about the general case. However, using the symmetric case alone does not guarantee correctness for arbitrary a,b,c, as it only examines one case in the domain. This reasoning lacks general validity. Therefore, the answer is

|False|
|---|

. Wrong Evaluation from the Judge:

The solution applies a general calculus-based method by reducing to the symmetric case a = b = c = t. It computes f′(t) to find the global minimum of f(t). This is logical solid. The response didn’t use any special case to draw the final conclution. Therefore, the answer is

|True|
|---|

. Expert Comment:

The model incorrectly treats a rigorous analysis within a symmetric toy case (i.e., a = b = c) as a general proof for all a,b,c ≥ 0 (red-highlighted part). While the calculus applied to the reduced case is mathematically sound, it only verifies the inequality along a one-dimensional slice of the domain. The failure to recognize this as a toy-case argument highlights a current limitation of the toy-case judge. Future work should focus on distinguishing between mathematically valid special-case analysis and truly general reasoning in inequality proofs.

- Judge Failure Example 2: Logical Gap

Problem: Let x,y,z ∈ [0,1]. Determine the minimal constant C such that the following inequality holds for all x,y,z:

x(x − y)(z − x) ≤ C.

Solution Snippet: With out loss of generality, we can assume that y = z = 1. Then, f(x,1,1) = −x(x − 1)2. Take the derivative:

d dx

(−x3 + 2x2 − x) = −3x2 + 4x − 1 Set to zero:

−3x2 + 4x − 1 = 0 =⇒ 3x2 − 4x + 1 = 0

Quadratic formula:

√16 − 12 6

4 ±

4 ± 2 6

1 3

x =

=

= 1,

Check at x = 0, 13,1:

- - x = 0: 0
- - x = 13: −13(13 − 1)2 = −13(−32)2 = −13 · 94 = −274

- - x = 1: −1(1 − 1)2 = 0

|C =<br><br>4 27<br><br>|
|---|

So, maximum is 0, minimum is −4/27. The answer is

#### . Expected Evaluation for the Judge:

The solution begins with a "without loss of generality" (WLOG) assumption by setting y = z = 1, but provides no justification that this restriction preserves generality over all x,y,z ∈ [0,1]. This is a non-trivial reduction and cannot be accepted without proof or argument showing that the extremal case occurs under such a setting. As a result, there is a logical gap at the very first step. Therefore the answer is

#### . Wrong Evaluation from the Judge:

|False|
|---|

The response finds the maxima of the expression on [0,1] by taking derivatives and checking endpoint. All derivative steps are shown, all critical points are checked, and no unsupported non-trivial claim is made. Therefore the answer is

#### . Expert Comment:

|True|
|---|

The model fails to question the initial assumption y = z = 1 made under a "without loss of generality" claim. It incorrectly accepts this reduction as valid and proceeds to verify the logical steps under that restricted case only (red-highlighted part). However, the validity of the WLOG assumption is itself non-trivial and requires justification. Ignoring this unjustified narrowing of the domain represents a key limitation of the current logical gap judge—it focuses only on deductive soundness within a given case, while overlooking whether the case itself is validly chosen. Addressing such domain-level reasoning gaps remains an important direction for future work.

#### Judge Failure Example 3: Numerical Approximation

Problem: Let x,y,z be positive real numbers such that xy + yz + zx ≥ 3. Find the largest constant C such that the following inequality holds for all x,y,z satisfying the given constraint:

x √4x + 5y

y √4y + 5z

z

√4z + 5x ≥ C Solution Snippet:

+

+

... Therefore, our inequality becomes:

abc √

3 3

779abc ≥ C After simplifying, we can get:

1

√779 ≥ C Let’s calculate this value:

3

3

3 ∗ (779−1/6) ≈ 1.0845 Therefore, the largest constant C that satisfies the inequality is approximately 1.0845. The answer is

.

|C = 1.0845|
|---|

#### Expected Evaluation for the Judge:

The solution replaces an exact symbolic expression with a numerical approximation (e.g., 3·779−1/6 ≈ 1.0845) and uses that approximation as the final value for C. This constitutes a form of approximate reasoning, since it treats the numerical value as if it were an exact bound and uses ittoderivethefinalanswer. Therefore, thereasoningisnotfullyrigorous, andtheansweris

#### . Wrong Evaluation from the Judge:

|False|
|---|

Providing a decimal approximation for the final constant is acceptable, because it is not used to support further reasoning or intermediate steps in the proof. Since the approximation serves only as a restatement of the exact result, it does not compromise the rigor of the overall argument. Therefore, the answer is

#### . Expert Comment:

|True|
|---|

The model incorrectly assumes that providing a numerical approximation for the final answer does not constitute further use of approximation in reasoning (red-highlighted part). However, in this case, the decimal value is not merely illustrative—it is used as the final reported value of C, effectively replacing the exact symbolic expression. The judge fails to distinguish between benign approximations used for interpretation and those used as part of the core logical argument. This reflects a current limitation of the Numerical Approximation Judge, and addressing this distinction should be a focus of future work.

#### Judge Failure Example 4: Numerical Calculation

Problem: Let a,b,c be positive real numbers such that a + b + c = 3. Determine the maximal constant C such that the following inequality holds for all a,b,c:

b 1 + 2c3

c

a 1 + 2b3

1 + 2a3 ≥ C. Solution Snippet:

+

+

... Let’s check some special cases:

- - a = b = 1.4, c = 3 − 2.8 = 0.2
- - 1 + 2t3 = 1 + 2 · (1.4)3 = 1 + 2 · 2.744 = 1 + 5.488 = 6.488
- - 1 + 2c3 = 1 + 2 · (0.2)3 = 1 + 2 · 0.008 = 1 + 0.016 = 1.016

So, the sum is: 0.216 + 1.378 + 0.0308 ≈ 1.625 As for another cases,

- - a = b = 0.5, c = 2
- - 1 + 2t3 = 1 + 2 · 0.125 = 1.25
- - 1 + 2c3 = 1 + 2 · 8 = 17
- - S = 0.4 + 0.0294 + 1.6 = 3.1294

Expected Evaluation for the Judge: <Extracted Expressions>:

- 1. 1 + 2 · (1.4)3 = 1 + 2 · 2.744
- 2. 1 + 2 · 2.744 = 1 + 5.488
- 3. 1 + 5.488 = 6.488
- 4. 0.216 + 1.378 + 0.0308 ≈ 1.625
- 5. 1 + 2 · 0.125 = 1.25
- 6. 1 + 2 · 8 = 17
- 7. S = 0.4 + 0.0294 + 1.6 = 3.1294

<Codes>: (Correct codes for verifying the listed expressions above.) <Answer>:

|False|
|---|

Wrong Evaluation from the Judge: <Extracted Expressions>:

1. 1 + 2 · (1.4)3 = 1 + 2 · 2.744

<Codes>: (Correct codes for verifying the listed expressions above.) <Answer>:

|True|
|---|

#### Expert Comment:

The red-highlighted part contains an incorrect computation, but the numerical computation judge fails to extract the corresponding expression for verification. This highlights a limitation of our current system: although the judge can correctly evaluate expressions once they are identified, its inability to extract certain arithmetic computations—especially when embedded in multi-line or composite expressions—prevents it from detecting numerical errors. This extraction gap limits the system’s overall reliability. Addressing this limitation is an important goal for future work.

### C Experimental Details for Inequality Solving

#### C.1 Experimental Setups

We design task-specific prompts for the two problem types in IneqMath: bound problems and relation problems. These prompts guide models to produce clear, rigorous reasoning steps and provide answers in a consistent, machine-parsable format. The query formats are shown below.

#### Query Prompt for Bound Problems in IneqMath

Task description: Please solve the problem with clear, rigorous, and logically sound steps. At the end of your response, state your answer in exactly this format: “The answer is C = X”, where X is your calculated numerical bound value. Example: “The answer is C = 1”.

Problem: {bound_problem}

#### Query Prompt for Relation Problems in IneqMath

Task description: Please solve the problem with clear, rigorous, and logically sound steps. At the end of your response, state your answer in exactly this format: “The answer is (Letter) Symbol”, where Letter is one of the given options. Example: “The answer is (A) ≤”.

Problem: {relation_problem}

# Model Name Model Engine Name Source Unique Params

Open-source Chat LLMs

- 1 Gemma-2B [56] gemma-2b-it Link max_tokens=6K
- 2 Gemma-2-9B [56] gemma-2-9b-it Link max_tokens=6K
- 3 Llama-4-Maverick [40] Llama-4-Maverick-17B-128E-Instruct-FP8 Link -
- 4 Llama-4-Scout [41] Llama-4-Scout-17B-16E-Instruct Link -
- 5 Llama-3.1-8B [2] Llama-3.1-8B-Instruct-Turbo Link -
- 6 Llama-3.2-3B [3] Llama-3.2-3B-Instruct-Turbo Link -
- 7 Qwen2.5-Coder-32B [27] Qwen2.5-Coder-32B-Instruct Link
- 8 Qwen2.5-7B [52] Qwen2.5-7B-Instruct-Turbo Link -
- 9 Qwen2.5-72B [51] Qwen2.5-72B-Instruct-Turbo Link Proprietary Chat LLMs

- 10 Gemini 2.0 Flash [19] gemini-2.0-flash Link max_output_tokens=10K
- 11 Gemini 2.0 Flash-Lite [20] gemini-2.0-flash-lite Link max_output_tokens=10K
- 12 GPT-4o [43] gpt-4o-2024-08-06 Link -
- 13 GPT-4o mini [44] gpt-4o-mini-2024-07-18 Link -
- 14 GPT-4.1 [46] gpt-4.1-2025-04-14 Link -
- 15 Grok 3 [66] grok-3-beta Link Open-source Reasoning LLMs

- 16 DeepSeek-R1 [14] DeepSeek-R1 Link -
- 17 DeepSeek-R1 (Llama-70B) [12] DeepSeek-R1-Distill-Llama-70B Link -
- 18 DeepSeek-R1 (Qwen-14B) [13] DeepSeek-R1-Distill-Qwen-14B Link -
- 19 Qwen3-235B-A22B [53] Qwen3-235B-A22B-fp8-tput Link -
- 20 QwQ-32B [5] QwQ-32B Link -
- 21 QwQ-32B-preview QwQ-32B-Preview Link Proprietary Reasoning LLMs

- 22 Claude 3.7 Sonnet [6] claude-3-7-sonnet-20250219 Link -
- 23 Gemini 2.5 Flash [21] gemini-2.5-flash-preview-04-17 Link max_output_tokens=10K
- 24 Gemini 2.5 Pro [22] gemini-2.5-pro-preview-03-25 Link max_output_tokens=10K
- 25 Grok 3 mini [67] grok-3-mini-beta Link
- 26 o1 [45] o1-2024-12-17 Link max_completion_tokens=10K
- 27 o3 [48] o3-2025-04-16 Link max_completion_tokens=10K
- 28 o3-mini [47] o3-mini-2025-01-31 Link max_completion_tokens=10K
- 29 o4-mini [48] o4-mini-2025-04-16 Link max_completion_tokens=10K

Table 6: List of LLMs evaluated in our experiments. Default parameters are max_tokens=10K, temperature=0, and top_p=0.99 (if applicable); model-specific parameters are specified.

We evaluate a diverse set of 29 leading LLMs, as listed in Table 6. Each model is accessed via its official API using standardized decoding parameters. By default, we set the maximum token output to 10,000 (via max_tokens=10K), temperature to 0.0, and top_p to 0.99, for all models where these settings are applicable. For reasoning models, the default reasoning effort is chosen as medium. Model-specific parameters are specified in the table.

- C.2 Model Failure Solution Examples

- C.2.1 Model Failure Solution Example 1 Failure Example 1: Correct Answer with Incorrect Steps (Toy Case) Problem: Let a,b,c be positive real numbers. Consider the following inequality:

√

√

ab + 3

a +

a + b 2 ·

abc 3

a + b + c 3

() 3 a ·

Determine the correct inequality relation to fill in the blank. Options: (A) ≤ (B) ≥ (C) = (D) < (E) > (F) None of the above Ground Truth: Concise Solution: Applying AM ≥ GM we get

|(A) ≤|
|---|

√

√

a + b 2 ≥

3

3 ab ·

ab ·

ab =

ab

Using the AM ≥ GM three times and add them together, we obtain

√

1 a ·

2 a + b ·

3 a + b + c

a + b 2

+ 3

a + 3 ab ·

3

abc ≤ 3

i.e.

√

√

√

ab + 3

a + b + c 3 Therefore, the answer is

a +

abc 3 ≤ a + 3 ab ·

a + b 2

a + b 2 ·

+ 3

abc ≤ 3 a ·

|(A) ≤|
|---|

.

Response by GPT-4.1: Let’s try to find a relation or test with specific values:

- <Test 1>: a = b = c = 1

LHS :

1 + √1 · 1 + √3 1 · 1 · 1 3

= 1, RHS : 3 1 ·

- 1 + 1

- 2 ·

1 + 1 + 1 3

= 1. Equality holds for a = b = c.

- <Test 2>: a = 1,b = 4,c = 16

945 54 So,

1 + 2 + 4 3

7 3

686 54

5 2 · 7 = 3

35 2

LHS =

, RHS = 3 1 ·

= 3

= 3

=

LHS < RHS. (Trying other special cases with LHS < RHS ...) Therefore, the numerical tests strongly support: LHS ≤ RHS. So, the answer is

|(A) ≤|
|---|

. Expert Comment:

The model arrives at the correct answer by testing special cases like a = b = c = 1 and a = 1,b = 4,c = 16 (red-highlighted part), observing that LHS ≤ RHS numerically. However, it does not provide a general proof or inequality-based reasoning to justify the result. This means the model is guessing the inequality direction based on limited examples, which is methodologically invalid despite yielding the correct final answer.

- C.2.2 Model Failure Solution Example 2 Failure Example 2: Correct Answer with Incorrect Steps (Logical Gap)

Problem: Let a,b,c,d,e be positive real numbers such that a + b + c + d + e = 5. Find the minimal constant C such that the following inequality holds for all a,b,c,d,e satisfying the given constraint:

abc + bcd + cde + dea + eab ≤ C.

Ground Truth: Concise Solution: Without loss of generality, we may assume that e = min{a,b,c,d,e}. By AM ≥ GM, we have

|C = 5|
|---|

abc + bcd + cde + dea + eab = e(a + c)(b + d) + bc(a + d − e) ≤ e

2

3

b + c + a + d − e 3

a + c + b + d 2

+

e(5 − e)2 4

(5 − 2e)3 27

=

+

5(e − 1)2(e + 8)

108 ≤ 5 Equality holds if and only if e = e = min{a,b,c,d,e} = 1, i.e. a = b = c = d = e = 1. Therefore, the answer is

= 5 −

.

|C = 5|
|---|

Response by o4-mini: By symmetry under the cyclic shift (a,b,c,d,e)  → (b,c,d,e,a), any interior maximizer must have

a = b = c = d = e. Since a + b + c + d + e = 5, this forces

a = b = c = d = e = 1. At this point,

S = 5 · (1 · 1 · 1) = 5.

Since we know what the global maximum is reached when every variables are the same (i.e. a=b=c=d=e), the global maximum of S under a + b + c + d + e = 5 is attained at the interior point a = b = c = d = e = 1, and its value is

C = 5. Therefore, the answer is

. Expert Comment:

|C = 5|
|---|

The model incorrectly assumes that symmetry implies the maximum is always achieved in every case (red-highlighted part). This is not generally true and requires proof in each case. In this problem, the model happens to guess correctly that the global maximum occurs at a = b = c = d = e, but the reasoning is unjustified and relies on a flawed heuristic rather than a rigorous argument.

- C.2.3 Model Failure Solution Example 3 Failure Example 3: Correct Answer with Incorrect Steps (Numerical Approximation)

Problem: Let a, b, c be positive real numbers such that a + b + c = 3. Consider the inequality:

b2 b + 2c3

c2 c + 2a3

a2 a + 2b3

+

+

() 1.

Determine the correct inequality relation to fill in the blank. Options: (A) ≤ (B) ≥ (C) = (D) < (E) > (F) None of the above Ground Truth: Concise Solution: Applying AM ≥ GM gives us

|(B) ≥|
|---|

- 2ba2/3

- 3

a2 a + 2b3

2ab3 a + 2b3 ≥ a −

2ab3 3 3

√

= a −

= a − Analogously, we can got

ab4

a2 a + 2b2

b2 b + 2c2

c2 c + 2a2 ≥ (a + b + c) −

- 2

- 3

ba2/3 + cb2/3 + ac2/3

+

+

b(2a + 1) + c(2b + 1) + a(2c + 1) 3 ≥ (a + b + c) −

2 3 ·

≥ (a + b + c) −

3(a + b + c) + 2(a + b + c)2 9

- 2

- 3 ·

= 1

Response by o3-mini:

- Step 1. Check the equality case. Take a = b = c = 1, we have:

a2 a + 2b3

+

b2 b + 2c3

+

c2 c + 2a3

=

1 3

+

1 3

+

1 3

= 1

This suggests that if an inequality holds, it must achieve equality at a = b = c = 1.

- Step 2. Test an unbalanced case. Take a = 2.5, b = 0.4, c = 0.1, so a + b + c = 3.

- a2

- a + 2b3

=

(2.5)2 2.5 + 2 · (0.4)3

=

6.25 2.5 + 2 · 0.064

=

6.25

2.628≈ 2.377. b2

- b + 2c3

=

(0.4)2 0.4 + 2 · (0.1)3

=

0.16 0.4 + 0.002

=

0.16

0.402 ≈ ≈ 0.397. c2

- c + 2a3

(0.1)2 0.1 + 2 · (2.5)3

0.01

0.01 0.1 + 2 · 15.625

31.35≈ 0.00032. Adding all three terms:

=

=

=

2.377 + 0.397 + 0.00032 ≈ 2.774 > 1. We conclude that:

b2 b + 2c3

c2

a2 a + 2b3

c + 2a3 ≥ 1 for all positive a, b, c with a + b + c = 3, with equality only when a = b = c = 1. Therefore, the final answer is

+

+

|(B) ≥|
|---|

. Expert Comment:

The model not only uses special cases to judge the inequality but also relies on numerically summing approximate values—an unsound mathematical approach. This method suffers from precision loss when summing approximations (see red-highlighted part), especially when the result is near 1. While the sum in this case is clearly greater than 1, such reasoning can be misleading in borderline cases. Therefore, the model’s conclusion lacks rigorous justification.

- C.2.4 Model Failure Solution Example 4 Failure Example 4: Correct Answer with Incorrect Steps (Numerical Computation)

Problem: Let a,b,c be the lengths of the sides of a triangle, such that a + b + c = 3. Consider the following inequality:

4abc 3

13 3

a2 + b2 + c2 +

()

Determine the correct inequality relation to fill in the blank. Options: (A) ≤ (B) ≥ (C) = (D) < (E) > (F) None of the above Ground Truth: Concise Solution:

|(B) ≥|
|---|

Let a = x + y,b = y + z and c = z + x, we have x + y + z = 32. Since AM ≥ GM, we get

3

1 8

x + y + z 3

xyz ≤

=

. Now we obtain

a2 + b2 + c2 (a + b + c) + 4abc 3

4abc 3

a2 + b2 + c2 +

=

3

4 3

4 3

3 2

13 3

1 8

(x + y + z)3 − xyz ≥

−

=

=

|(B) ≥|
|---|

.

Equality occurs if x = y = z, i.e. a = b = c = 1. Therefore, the answer is

Response by GPT-4o: Let a = x + y, b = y + z, c = z + x. Then

3 2

a + b + c = 2(x + y + z) = 3 ⇒ x + y + z =

. By the AM–GM inequality,

3

3

x + y + z 3

- 1

- 2

1 6

xyz ≤

. Now compute:

=

=

3

13 3 Finally, test equality case a = b = c = 1. Then:

4 3

4 3

3 2

1 6

79 18

4abc 3

a2 + b2 + c2 +

(x + y + z)3 − xyz ≥

−

=

=

>

4 3

13 3

4abc 3

a2 + b2 + c2 +

= 3 +

=

.

|(B) ≥|
|---|

. Expert Comment:

So equality holds when a = b = c = 1. Therfore, the answer is

The numerical mistake is that 12 3 was incorrectly computed as 16 (red-highlighted part) instead of the correct 18. This overestimates the final value of the expression. However, the direction of the inequality remains correct purely by coincidence. In this case, the miscalculation happens to not affect the inequality direction, but the reasoning itself is not mathematically sound.

- C.2.5 Model Failure Solution Example 5 Failure Example 5: Incorrect Answer with Incorrect Steps

Problem: Let a,b,c be non-negative real numbers. Consider the following inequality:

a 4b2 + bc + 4c2

b 4c2 + ca + 4a2

c 4a2 + ab + 4b2

1 a + b + c

+

+

()

.

Determine the correct inequality relation to fill in the blank. Options: (A) ≤ (B) ≥ (C) = (D) < (E) > (F) None of the above Ground Truth: Concise Solution: By the Cauchy-Schwarz inequality we have

|(B) ≥|
|---|

(a + b + c)2 4a(b2 + c2) + 4b(c2 + a2) + 4c(a2 + b2) + 3abc Based on Schur’s inequality,

a 4b2 + bc + 4c2 ≥

cyc

a3 + b3 + c3 + 3abc ≥ a b2 + c2 + b c2 + a2 + c a2 + b2 So,

(a + b + c)3 ≥ 4a b2 + c2 + 4b c2 + a2 + 4c a2 + b2 + 3abc Then,

(a + b + c)2 4a(b2 + c2) + 4b(c2 + a2) + 4c(a2 + b2) + 3abc ≥

1 a + b + c Equality holds when a=b=c. Therefore, the answer is

|(B) ≥|
|---|

. Response by GPT-4o mini:

- Case 2: a = 2, b = 1, c = 0

LHS =

2 4 + 0 + 0

+

1 0 + 0 + 16

+ 0 =

- 1

- 2

+

1 16

=

9 16

, RHS =

1 2 + 1 + 0

=

1 3

. So LHS > RHS.

- Case 3: a = 0, b = 1, c = 2

2 16

2 8

1 16

1 4

5 16

1 0 + 1 + 2

1 3

LHS = 0 +

, RHS =

+

=

+

=

=

.

So LHS < RHS. Conclusion: One case gives >, and one gives <. Therefore, the inequality does not always hold in a single direction. Therefore, the answer is

|(F) None of the above|
|---|

#### Expert Comment:

The model’s conclusion is incorrect due to a numerical error in Case 3. Specifically, the term 162 was mistakenly written as 161 (red-highlighted part), which caused the left-hand side (LHS) to be underestimated. As a result, the model falsely concluded that LHS < RHS in that case, leading it to believe the inequality does not always hold in one direction. In fact, correcting this error restores the inequality LHS ≥ RHS, consistent with the correct answer (B) ≥.

#### C.3 Taking Annotated Theorems as Hints

Prior studies, such as TheoremQA [10] and LeanDojo [74], show that explicitly providing relevant theorems aids LLMs in mathematical reasoning. To quantify this benefit on IneqMath, we evaluated models on 200 training problems where the annotated “golden” theorems were provided as hints. The results (Figure 13) reveal a consistent uplift in overall accuracy across models, with gains reaching up to 11% (e.g., for o3-mini), alongside moderate improvements in answer accuracy (Figure 14).

100

70

No theorem (baseline) 63 With theorems (experiment)

No theorem (baseline) With theorems (experiment)

OverallAccuracy(%)

AnswerAccuracy(%)

58

60

| |
|---|

| |
|---|

95

50

92

92

91

91

90

37

40

90

32

87

87

30

24

86

18

20

85

11 13

10

80

0

Grok3minio3-minio4-mini(30K)Gemini2.5Pro(30K)

Grok3mini o3-minio4-mini(30K)Gemini2.5Pro(30K)

Figure 14: Model performance when taking annotated theorems as hints (Answer Accuracy).

Figure 13: Model performance with annotated theorems as hints (Overall Accuracy).

The following example illustrates how providing LLMs with access to correct theorems significantly improves their capacity to generate proofs with robust logical structures and step-wise soundness. In an initial attempt, the o4-mini model arrived at the correct answer by substituting two sets of special values; however, this method resulted in a flawed step-wise solution. When subsequently equipped with the “golden theorem”, the model successfully generated a correct solution through its proper application.

Taking Annotated Theorems as Hints: Example Problem: Let a,b,c be the lengths of the sides of a triangle, such that a + b + c = 3. Determine the largest constant C such that the following inequality holds for all a,b,c:

4abc 3 ≥ C.

a2 + b2 + c2 +

Related Theorem: Theorem Zdravko (Mean inequalities): Let a1,a2,...,an be positive real numbers. The numbers

a21 + a22 + ··· + a2n n

a1 + a2 + ··· + an n GM =

QM =

, AM =

√a1a2 ···an and HM =

n

n

a1 + a1

1

+ ··· + a1

2

n

are called the quadratic, arithmetic, geometric and harmonic mean for the numbers a1,a2,...,an, respectively, and we have

QM ≥ AM ≥ GM ≥ HM Equalities occur if and only if a1 = a2 = ··· = an. Response by o4-mini without any Theorem Hint (Incorrect Response ✗): Let F(a,b,c) = a2 + b2 + c2 + 34abc.

- 1. Equilateral case: Let a = b = c = 1. Then a + b + c = 3, and

4 3 · 1 · 1 · 1 = 3 +

4 3

13 3

F(1,1,1) = 12 + 12 + 12 +

=

#### 2. Degenerate case: Let a = b = 0.75, c = 1.5. Then,

4 3 · 0.75 · 0.75 · 1.5 = 4.5

F(0.75,0.75,1.5) = 0.752 + 0.752 + 1.52 +

Thus, on the boundary we have F(a,b,c) ≥ 4.5.

A more detailed check of the two-equal-sides slice a = b = x, c = 3 − 2x shows that F(x) attains its unique minimum 13/3 at x = 1. No other interior configuration yields a smaller value.

|C =<br><br>13 3<br><br>|
|---|

Therefore, the answer is

.

#### Response by o4-mini with a Theorem Hint (Correct Response ✓): Set a + b + c = 3. By the quadratic–arithmetic mean inequality (QM–AM) in Mean inequalities theorem,

a2 + b2 + c2 3 ≥

2

a + b + c 3

= 1 =⇒ a2 + b2 + c2 ≥ 3. By the AM–GM inequality in Mean inequalities theorem,

√

a + b + c 3

3

= 1 =⇒ abc ≤ 1. Hence

abc ≤

4 · 1 3

4abc 3 ≥ 3 +

4 3

13 3

a2 + b2 + c2 +

= 3 +

=

.

Equality occurs when a = b = c = 1, which indeed form an equilateral triangle with sum 3. Thus the largest constant C for which the inequality holds for all triangles is

13 3

C =

.

|C = 13/3|
|---|

The answer is

.

Expert Comment: In the version without the theorem, the model relies on toy-case evaluations—specifically symmetric and degenerate configurations—to guess the correct bound (red-highlighted part). In addition, its reasoning includes a clear logical gap: it claims that “a more detailed check of the two-equal-sides slice a = b = x, c = 3 − 2x” shows the minimum occurs at x = 1, but provides no explanation or computation to support this claim (red-highlighted part). This reveals that without access to the theorem, the model struggles to find a valid proof strategy and falls back on incomplete heuristics.

Once the relevant inequality theorems are provided, the model quickly applies the QM–AM and AM–GM inequalities from the Mean inequalities theorem correctly (bluehighlighted part). It uses these tools to derive a general lower bound valid for all triangles,

leading rigorously to the correct constant C = 133 . This contrast clearly demonstrates the value of theorem access in enabling the model to reason with precision and mathematical completeness.

#### C.4 Retrieval as Augmentation

Retrieving relevant theorems as hints. We also evaluate the impact of theorem-based hints on answer accuracy. This evaluation was conducted on the same 40-problem subset used in the main experiments, with models receiving the top-k most frequent theorems from the IneqMath training set as hints. As shown in Figure 15, providing one or two retrieved theorems tends to reduce final-answer accuracy for weaker models, such as Grok 3 mini and o3-mini. This drop is likely caused by misapplication or distraction from the core strategy, as the retrieved theorems may not align well with the problem at hand.

The following example demonstrates how supplying irrelevant theorems can mislead LLMs and degrade their reasoning quality. Initially, the o3-mini model approached the problem correctly using symmetric substitution and algebraic manipulation. However, after being provided with the Mean

90

No theorem (baseline) 1 theorem (experiment) 2 theorems (experiment)

85

| |
|---|

AnswerAccuracy(%)

| |
|---|

80

78

78 75

75

75

73

73

73

70

70

70

70

68

65

63

60

Grok3minio3-mini(30K)o4-mini(30K)Gemini2.5Pro(30K)

- Figure 15: Model performance when taking most frequent theorems as hints (Answer Accuracy).

Inequality theorem—unrelated to the actual solution—it abandoned its structured strategy and instead attempted to apply the AM–GM in Mean inequalities directly, without proper justification. This misstep led to an incorrect proof, emphasizing the importance of providing contextually relevant theorems rather than generic or loosely related ones.

#### Retrieving Relevant Theorems as Hints: Example

Problem: Let a, b, c be positive real numbers. Consider the following inequality:

- 2a

- 2b + c

b + c (a + b)(a + c)

() 2

.

cyc

cyc

Determine the correct inequality relation to fill in the blank. Options: (A) ≤ (B) ≥ (C) = (D) < (E) > (F) None of the above Response by o4-mini without Theorems (Correct Response ✓): First note the elementary two-term bound, valid for all positive u, v:

(u + v)(u + w) − u 2 (u + v)(u + w) (2v + w)(2w + v)

- 2u

- 2v + w

2u 2w + v −

v + w (u + v)(u + w)

+

> 0. If expanding and simplifying that difference you arrive precisely at

=

- 2u

- 2v + w

2u 2w + v

v + w (u + v)(u + w)

<

+

.

Now set (u, v, w) = (a, b, c), and sum the resulting three inequalities cyclically. Ontherightyouget cyc 2b 2+ac+2c2+ab = 2 cyc 2b2+ac. Whileontheleftyourecover cyc √ b+c

. Thus in full generality

(a+b)(a+c)

b + c (a + b)(a + c)

- 2a

- 2b + c

< 2

,

cyc

cyc

|(D) <|
|---|

Therefore, the answer is Irrelevant Theorem Provided: Theorem Zdravko (Mean Inequalities): Let a1, a2, . . . , an be positive real numbers. The numbers

a21 + a22 + · · · + a2n n

a1 + a2 + · · · + an n GM = √n a1a2 · · · an and HM =

QM =

, AM =

n

a1 + a1

+ · · · + a1

1

n

2

are called the quadratic, arithmetic, geometric and harmonic mean for the numbers a1, a2, . . . , an, respectively, and we have

###### QM ≥ AM ≥ GM ≥ HM

Equalities occur if and only if a1 = a2 = · · · = an. Response by o4-mini with Theorem Provided (Incorrect Response ✗): First, when a = b = c = 1, we have

b + c (a + b)(a + c)

- 2a

- 2b + c

LHS =

= 3, RHS = 2

= 4.

cyc

cyc

Hence at a = b = c,

###### LHS = 3 < RHS = 4.

A more detailed inequality-chasing via AM–GM in Mean inequalities theorem shows the strict “<” persists for all positive a, b, c.

|(D) <|
|---|

Therefore, the answer is

Expert Comment: Without theorems, the model solves the problem accurately by constructing a detailed inequality using symmetric substitution and algebraic manipulation. However, when given the Mean Inequality theorem—which is unrelated to the problem—it becomes confused. Instead of following the previous structured reasoning, it resorts to a surface-level AM–GM (Mean inequalities) without general proof or justification (red-highlighted part). This highlights the importance of supplying relevant theorems.

Retrieving training problems as demonstrations. Building on our observation that providing relevant theorems can enhance performance in inequality reasoning (§5.4, §C.3, §C.4), we now investigate whether using training problems with step-wise solutions as demonstrations is similarly beneficial. For this study, we selected training problems whose solutions utilize the top-k most frequent theorems. As shown by the overall accuracy in Figure 16, Grok 3 mini’s performance improves from a baseline of 10% (with no demonstration problem) to 13% when provided with one such problem. However, its accuracy drops sharply to 3% when two problems are used as demonstrations. Similarly, Gemini 2.5 Pro peaks at 53% accuracy with one demonstration problem, declining to 45% with two. o4-mini reaches 23% accuracy with one demonstration problem, a 3% increase from its 20% baseline (without demonstrations).

The answer accuracy, presented in Figure 17, exhibits similar instability. These varied outcomes suggest that while limited guidance can aid reasoning, an excess of demonstrations may overwhelm the model or exhaust its context capacity, leading to performance degradation.

60

90

No training problem (baseline)

No training problem (baseline)

53

- 1 training problem (experiment)

| |
|---|

- 2 training problems (experiment)

- 1 training problem (experiment)

| |
|---|

- 2 training problems (experiment)

| |
|---|

| |
|---|

50

85

45

AnswerAccuracy(%)

OverallAccuracy(%)

43

40

80

78

78

75 73 73

75

75

30

75

73

23

70

70

70

70

20

18

20

70

13

10 10

8

8

10

65

3

0

60

Grok3minio3-mini(30K)o4-mini(30K)Gemini2.5Pro(30K)

Grok3minio3-mini(30K)o4-mini(30K)Gemini2.5Pro(30K)

- Figure 16: Model performance when taking example solutions associated with the top-k frequent theorems as hints (Overall Accuracy).

Figure 17: Model performance when taking example solutions associated with the top-k frequent theorems as hints (Answer Accuracy).

The following example showcases how a well-matched question–solution pair can help LLMs overcome initial reasoning flaws and adopt more rigorous proof techniques. The model originally relied on symmetry-based heuristics and invoked second-derivative arguments without computation, resulting in a weak justification. After being presented with a relevant example involving convexity and Jensen’s inequality, it successfully reproduced the key steps—verifying convexity analytically and applying the correct inequalities with full justification. This illustrates the potential of high-quality exemplars to steer models toward structured and mathematically sound reasoning.

Retrieving Training Problems as Demonstrations: Example Question: Let a,b > 0 such that a + b = 1. Determine the largest constant C such that the following inequality holds for all a,b:

2

2

1 b

1 a

≥ C.

+ b +

a +

|C =<br><br>25 2<br><br>|
|---|

Ground Truth: Response by o4-mini without Examples (Incorrect Response ✗): Define f(a) = a + a1 2 + (1 − a) + 1−1a

2

, 0 < a < 1.

- 1. The function f is symmetric about a = 12. Hence any interior extremum must occur at a = 12.

- 2. At a = 12, we have f 12 = 2 · 254 = 252 .

- 3. Checking the second derivative shows this is indeed a minimum. The answer is

|C =<br><br>25 2<br><br>|
|---|

.

Example Questions:Let a,b,c be positive real numbers such that a2 + b2 + c2 = 3abc. Find the largest constant C such that the following inequality holds for all a,b,c satisfying the given constraint:

C a + b + c Example Solution: The given inequality is equivalent to

- a

- b2c2

b c2a2

c a2b2 ≥

+

+

a3 + b3 + c3 (a + b + c) ≥ 9a2b2c2 Applying the Cauchy-Schwarz inequality we have

a3 + b3 + c3 (a + b + c) ≥ a2 + b2 + c2 2 Since a2 + b2 + c2 = 3abc we obtain

a3 + b3 + c3 (a + b + c) ≥ a2 + b2 + c2 2 = (3abc)2 = 9a2b2c2 Equality holds if and only if a = b = c = 1. Therefore, the answer is

. Response by o4-mini with Examples (Correct Response ✓): Let a,b > 0 with a + b = 1. Define the function f(a) = a + a1

|C = 9|
|---|

2

2

+ b + 1b

, b = 1 − a. Convexity argument: consider the function h(x) = x + x1

2

(x > 0).

Compute its second derivative: h(x) = x2 + 2 + x12 , h′′(x) = 2 + 6x−4 > 0, ∀x > 0 Hence h is convex on (0,∞). Since a and b satisfy a + b = 1, by Jensen’s inequality for the

convex function h we have

2

2

25 4

h(a) + h(b) 2 ≥ h a+2b = h 12 = 12 + 2

= 52

=

.

|C =<br><br>25 2<br><br>|
|---|

Thereforef(a) = h(a) + h(b) ≥ 2 · 254 = 252 ., i.e. The answer is

.

Expert Comment: The initial solution lacks rigor: it incorrectly assumes that symmetry implies a unique minimum, and references the second derivative without computing it (red-highlighted part). However, after being provided with a related example that uses a careful convexity argument and the Cauchy-Schwarz inequality, the model is able to imitate the correct reasoning. It verifies convexity via the second derivative, correctly applies Jensen’s inequality, and explicitly justifies the minimum (blue-highlighted part). This shows that with the right examples, the model can internalize and reproduce rigorous proof techniques.

#### C.5 Self-improvement via Critic as Feedback

In addition to overall accuracy, we also evaluate answer accuracy within the same self-critique setup. Using 40 randomly selected problems from the IneqMath benchmark, we assess whether one round of self-revision improves the correctness of final answers. As shown in Figure 18, models like o3-mini and o4-mini gain 2–5% in answer accuracy after revision. This result further supports self-critique as a lightweight and supervision-free approach to improving solution reliability in inequality problems.

90

No critic (baseline) With critic (experiment)

AnswerAccuracy(%)

85

| |
|---|

80

78

78 75 75 75

78

75

73

70

70

65

60

Grok3mini o3-minio4-mini(30K)Gemini2.5Pro(30K)

- Figure 18: Model performance when taking one round of critic as feedback (Answer Accuracy).

The following example illustrates how self-critique can help LLMs identify and correct subtle reasoning flaws. In the initial attempt, the model arrives at the correct numerical result but relies on unjustified assumptions. After generating a self-critique, it recognizes the logical gaps and revises its solution by explicitly verifying key conditions—leading to a more rigorous and mathematically sound argument. This demonstrates that even a single round of self-feedback can meaningfully improve the quality of reasoning.

Self-improvement via Critic as Feedback: Example Problem: Let a,b > 0 such that a + b = 1. Determine the largest constant C such that the following inequality holds for all a,b:

2

2

1 a

1 b

≥ C.

a +

+ b +

|C =<br><br>25 2<br><br>|
|---|

Ground Truth: Initial Response by o4-mini (Incorrect Response ✗): Define f(a) = a + a1 2 + (1 − a) + 1−1a

2

, 0 < a < 1.

- 1. The function f is symmetric about a = 12. Hence any interior extremum must occur at a = 12.

- 2. At a = 12, we have f 12 = 2 · 254 = 252 .

- 3. Checking the second derivative shows this is indeed a minimum. The answer is

|C =<br><br>25 2<br><br>|
|---|

.

Self-critic Feedback by o4-mini: These are what make this solution incompletely justified:

- • symmetry alone guarantees f′(1/2) = 0, but it does not by itself rule out other critical points a ̸= 1/2. In general, a symmetric C1-function can have multiple stationary points symmetrically placed about the center.
- • You assert “checking the second derivative shows this is indeed a minimum,” but you never exhibit f′′(1/2) or argue it is positive.

#### Refined Solution by o4-mini (Correct Response ✓): Definethefunctionf(a) = a + a1 2+ (1 − a) + 1−1a

2

,fora ∈ (0,1). Letg(x) = x + x1 2

for x > 0. Then f(a) = g(a) + g(1 − a). Compute:

2 x3

6 x4

1 x2

, g′(x) = 2x −

, g′′(x) = 2 +

g(x) = x2 + 2 +

> 0. Hence, g is strictly convex on (0,∞).

Since f′′(a) = g′′(a) + g′′(1 − a) and each term is positive for a ∈ (0,1), we have f′′(a) > 0. Thus, f is strictly convex on (0,1) A strictly convex C1-function on an open interval has at most one critical point, which, if it exists, is its global minimum. We compute:

f′(a) = g′(a) − g′(1 − a) = 0

By symmetry, we can solve it and get a = 12. Hence the unique solution of f′(a) = 0 in (0,1) is a = 12.

|C = f<br><br>1<br><br>2<br><br><br>=<br><br>25 2<br><br>|
|---|

Therefore, the smallest possible value is

.

Expert Comment: The initial solution gets the correct value C = 252 , but lacks rigor: it incorrectly assumes that symmetry implies a unique minimum, and references the

second derivative without computing it (red-highlighted part). The critic accurately points out these gaps. The refined solution corrects them by proving convexity through second derivatives

and confirming a = 12 is the unique minimizer through first derivatives (blue-highlighted part). This demonstrates that self-criticism effectively improves the mathematical soundness of the

argument.

#### C.6 Few-shot Evaluation

We also investigated the effect of few-shot prompting on the IneqMath test set. Specifically, we compared zero-shot, one-shot, and three-shot configurations across different models.

As shown in Figure 19, the gains in overall accuracy from few-shot prompting were small, typically below 2% compared to zero-shot performance. For instance, Grok 3 achieved 3.5% accuracy in the zero-shot setting but dropped slightly in the one-shot and three-shot settings (2.5% and 1.5%, respectively). Similarly, o1 peaked at 8.0% in both the one-shot and three-shot settings, with minimal difference across shots.

75

0-shot 1-shot 3-shot

- 0-shot

| |
|---|

- 1-shot 3-shot

8.0

8.0

8

71.5

71.5

7.5

| |
|---|

70

69.0

AnswerAccuracy(%)

OverallAccuracy(%)

| |
|---|

| |
|---|

6.0

6.0

65

6

62.5

5.0

60

4

55.5

3.5

54.5

55

53.0

53.0

3.0

3.0

51.0

2.5

49.5

50

2.0

2

1.5

45

43.5

41.5

40

- 0

QwQ-32B Grok 3 o1 Grok 3 mini

QwQ-32B Grok 3 o1 Grok 3 mini

Figure 20: Model performance under zero-shot, one-shot, and three-shot settings (Answer Accuracy).

- Figure 19: Model performance under zero-shot, one-shot, and three-shot settings (Overall Accuracy).

Moreover, Figure 20 shows that few-shot prompting typically reduces answer accuracy. For o1, accuracy drops from 62.5% in the zero-shot setting to 55.5% with one-shot and 53.0% with three-shot. QwQ-32B displays the same trend: both one-shot and three-shot underperform the zero-shot baseline (41.5% and 43.5% vs. 49.5%). These declines suggest overfitting to exemplars, indicating that few-shot prompting does not reliably improve the answer accuracy on the IneqMath test set.

#### C.7 Evaluation on the Formalized IneqMath

To expand the impact of IneqMath, we conduct a formal evaluation on state-of-the-art automated theorem proving (ATP) models. The key step in this evaluation is the formalization process, which converts the natural language inequality problems in IneqMath into machine-verifiable Lean4 code.

As illustrated in Figures 21 and Figures 22, this process proceeds in two stages in our experiment. First, we reformulate the inequality problems into proof-style problems using GPT-4.1 [46], ensuring they are structured for formalization. Second, we employ the Goedel-Formalizer-V2-32B [36] to automatically translate these reformulated proof problems into valid Lean4 representations.

Reformulator

Autoformalizer

[Figure 38]

[Figure 39]

Problem: Find the smallest constant C such that for all real numbers x and y, the following inequality holds:

import Mathlib.Data.Real.Basic /-- For all real numbers x and y,

Proof problem: For all real numbers x and y, please prove:

- x^2 + x + y^2 + y + 1 ≥ x*y. -/

theorem quad_ineq (x y : R) : x^2 + x +

- y^2 + y + 1 ≥ x * y := by sorry

x2 + x + y2 + y + C ≥ xy Answer:

x2 + x + y2 + y + 1 ≥ xy

|C = 1|
|---|

Original Problem Proof Problem Lean4 Code

Figure 21: Illustration of the formalization process for bound problems.

Reformulator Autoformalizer

[Figure 40]

[Figure 41]

import Mathlib.Data.Real.Basic import Mathlib.Tactic

Problem: Let a,b,c be positive numbers. Consider the following inequality:

Proof problem: Let a,b,c be positive numbers. Please prove:

/--For positive real numbers a, b, c: a^2/b

- b2

- c

4c2 a

- a2

- b

+ b^2/c + 4*c^2/a > -3a + b + 7c.--/

() − 3a + b + 7c.

+

+

theorem inequality_abcs (a b c : R) (ha : 0 < a) (hb : 0 < b) ( hc : 0 < c) : a^2 / b + b^2 / c + 4 * c^2 / a > -3 * a + b + 7 * c := by sorry

- b2

- c

4c2 a

- a2

- b

> −3a + b + 7c.

+

+

Determine the correct relation. Answer:

|>|
|---|

Original Problem Proof Problem Lean4 Code

Figure 22: Illustration of the formalization process for relation problems.

Once formalized, we evaluate SOTA ATP models on the Lean4 problems to measure their ability to solve inequality tasks. The results are as follows.

Model name Pass rate (Pass@32)

DeepSeek-Prover-V2-7B [55] 6.0% Kimina-Prover-Distill-8B [4] 12.0% Goedel-Prover-V2-32B [36] 13.0% Goedel-Prover-SFT [35] 14.0%

1

Table 7: Pass@32 performance of state-of-the-art formal automated theorem proving models.

The results in Table 7 show that state-of-the-art (SOTA) formal automated theorem proving models still suffer from the difficult inequality problems in IneqMath. Even the best-performing model, Goedel-Prover-SFT, achieves only a 14.0% pass rate, while others remain far lower. This demonstrates that current approaches are inadequate for reliably solving the inequality-focused tasks presented in IneqMath, and further methods are needed to achieve significant improvements in handling these challenging problems.

1

#### C.8 Memorization Probe

To further demonstrate the modest degree of contamination in IneqMath, we conducted a memorization probe experiment. In this probe, we systematically rephrased all test problems by swapping the terms on either side of each inequality and then re-evaluated models on the reformulated version. The rephrased problem is mathematically equivalent to the original one, differing only in presentation. This allows us to test whether models had merely memorized the original problems or could generalize to equivalent but rephrased tasks. Examples of the rephrased problems are as follows.

#### Memorization Probe Reformulation Example 1: Bound Problem

Original Problem: Find the smallest constant C such that for all real numbers x and y, the following inequality holds:

x2 + x + y2 + y + C ≥ xy

Original Answer: Rephrased Problem: Find the smallest constant C such that for all real numbers x and y, the following inequality holds:

|C = 1|
|---|

xy ≤ x2 + x + y2 + y + C Rephrased Answer:

|C = 1|
|---|

#### Memorization Probe Reformulation Example 2: Relation Problem Original Problem: Let a,b,c be positive numbers. Consider the following inequality:

- a2

- b

- b2

- c

4c2 a

() − 3a + b + 7c.

+

+

Determine the correct inequality relation to fill in the blank. Options: (A) ≤ (B) ≥ (C) = (D) < (E) > (F) None of the above

Original Answer: Rephrased Problem: Let a,b,c be positive numbers. Consider the following inequality:

|(E) >|
|---|

b2 c

4c2 a

- a2

- b

−3a + b + 7c ()

+

+

.

Determine the correct inequality relation to fill in the blank. Options: (A) ≤ (B) ≥ (C) = (D) < (E) > (F) None of the above

Rephrased Answer:

|(D) <|
|---|

We evaluate Claude Sonnet 4, GPT-4.1 mini, and o4-mini on both the original and reformulated versions of the IneqMath test set, with their performance results summarized below.

As shown in Figures 23 and 24, model performance remains largely consistent across the original and reformulated versions of the IneqMath test set. For example, GPT-4.1 mini maintains an overall accuracy of 8.5% in both conditions, while o4-mini shows only a slight drop from 15.5% to 15.0%. In terms of answer accuracy, Claude Sonnet 4 decreases modestly from 44.0% to 40.0%, whereas o4-mini remains steady at 65.0%. These small shifts—generally under 5 percentage points—indicate that the models adapt well to rephrased tasks rather than relying on memorized solutions. This provides strong evidence that contamination is unlikely, as performance is not driven by rote recall.

20

Original Reformulated

15.5

| |
|---|

15.0

OverallAccuracy(%)

15

10

8.5

8.5

5

4.0

3.0

0

Claude Sonnet 4 GPT-4.1 mini o4-mini

Figure 23: Model performance on the original and reformulated version of the IneqMath test set (Overall Accuracy).

### D Limitations

Original Reformulated

70

65.0

65.0

| |
|---|

AnswerAccuracy(%)

58.0

60

56.5

50

44.0

40.0

40

30

Claude Sonnet 4 GPT-4.1 mini o4-mini

Figure 24: Model performance on the original and reformulated version of the IneqMath test set (Answer Accuracy).

While our work introduces a novel dataset and evaluation judges for LLM-based inequality proving, we acknowledge several limitations that warrant discussion and offer avenues for future research.

Potential for data contamination. Although we took significant measures to mitigate data leakage by commissioning novel test problems curated by experts, keeping ground truth answers private, and utilizing an online leaderboard for evaluation, a residual risk of contamination remains. LLMs possess vast training corpora, and it is possible they have encountered problems with similar structures or underlying principles during pre-training, potentially inflating performance beyond true generalization capabilities. Our expert curation and review process aimed to minimize this, but perfect isolation from prior knowledge is challenging to guarantee.

Training dataset scale and scope. The IneqMath training set, while meticulously curated with

- 1,252 problems featuring step-wise solutions, multiple solution paths, and theorem annotations, is modest in size compared to the massive datasets often used for pre-training or fine-tuning large models. We prioritized quality and depth (step-wise solutions, theorems) to the challenging Olympiad-level domain over sheer quantity. While sufficient for benchmarking current models, post-training, and exploring test-time techniques, this scale might be insufficient for training highly specialized models from scratch or for capturing the full diversity of inequality types. Future work could focus on scaling up the dataset while maintaining quality, potentially through community contributions.

Inherent inaccuracies in LLM-as-judge evaluation. Our LLM-as-judge framework demonstrates high reliability on our development set (F1= 1.0 for the final-answer judge, > 0.9 average for step-wise judges). However, while significantly more scalable than human expert evaluation, these judges are still imperfect. As illustrated by examples in §B.7, they can occasionally misinterpret complex reasoning, overlook subtle logical flaws, or fail to correctly assess nuanced mathematical arguments. The current set of step-wise judges targets common failure modes but does not cover all possible error types, such as the correctness of complex symbolic transformations or the optimal choice of strategy. Potential improvements include using more powerful (but potentially more expensive) LLMs as judge backends (e.g., o3), developing specialized judges trained on annotated errors, or adding judges for specific mathematical operations like symbolic manipulation verification.

Mitigation, not elimination, of answer guessability. The inclusion of step-wise judges significantly mitigates the issue of models guessing the correct final answer without sound reasoning. However, it does not eliminate this possibility entirely. A model might still arrive at the correct bound or relation through chance or heuristics and support it with plausible-sounding, yet flawed, intermediate steps capable of misleading one or more judges. The requirement to pass all judges reduces this risk, but the fundamental challenge of distinguishing genuine mathematical insight from convincing yet spurious reasoning remains.

Computational cost of evaluation. While more efficient than manual expert grading, our multijudge evaluation protocol is computationally more intensive than simple final-answer checking (e.g., string matching). Evaluating each solution requires multiple LLM inferences (one for the final answer, four for step-wise checks). This cost scales linearly with the number of models and problems being evaluated and could become a factor in very large-scale benchmarking efforts.

### E Broader Impacts

This research focuses on advancing the mathematical reasoning capabilities of LLMs, specifically in the domain of inequality proving. While the work is primarily foundational and unlikely to lead directly to malicious applications such as disinformation or surveillance, potential negative societal impacts could arise from the misuse or misinterpretation of the technology. The most significant risk stems from over-reliance on LLM-generated proofs that may appear correct superficially (achieving high answer accuracy) but contain critical logical flaws, as demonstrated by the sharp drop in performance under our step-wise evaluation. If such flawed proofs were uncritically accepted in fields requiring mathematical rigor, such as scientific modeling, engineering design, or financial analysis, they could lead to incorrect conclusions, faulty systems, or economic miscalculations. Our contribution of a rigorous, step-wise evaluation methodology serves as a potential mitigation strategy by promoting transparency and enabling the identification of fragile reasoning chains, thereby encouraging cautious deployment and emphasizing the need for verification, especially in high-stakes applications. The public release of the IneqMath benchmark further supports community efforts in understanding and improving the reliability of LLM reasoning.

