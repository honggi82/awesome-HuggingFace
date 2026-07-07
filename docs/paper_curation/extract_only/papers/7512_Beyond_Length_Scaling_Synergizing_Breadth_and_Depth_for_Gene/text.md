# arXiv:2603.01571v1[cs.AI]2Mar2026

## Beyond Length Scaling: Synergizing Breadth and Depth for Generative Reward Models

Qiyuan Zhang♠♢∗, Yufei Wang♢†, Tianhe Wu♠, Can Xu♢†, Qingfeng Sun♢, Kai Zheng♢ Xue Liu♡, Chen Ma♠†

♠City University of Hong Kong ♢Tencent Hunyuan ♡MBZUAI

### Abstract

Recent advancements in Generative Reward Models (GRMs) have demonstrated that scaling the length of Chain-of-Thought (CoT) reasoning considerably enhances the reliability of evaluation. However, current works predominantly rely on unstructured length scaling, ignoring the divergent efficacy of different reasoning mechanisms: Breadth-CoT (B-COT, i.e., multi-dimensional principle coverage) and Depth-CoT (D-COT, i.e., substantive judgment soundness). To address this, we introduce Mix-GRM, a framework that reconfigures raw rationales into structured B-COT and D-COT through a modular synthesis pipeline, subsequently employing Supervised FineTuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR) to internalize and optimize these mechanisms. Comprehensive experiments demonstrate that Mix-GRM establishes a new state-of-the-art across five benchmarks, surpassing leading open-source RMs by an average of 8.2%. Our results reveal a clear divergence in reasoning: B-COT benefits subjective preference tasks, whereas D-COT excels in objective correctness tasks. Consequently, misaligning the reasoning mechanism with the task directly degrades performance. Furthermore, we demonstrate that RLVR acts as a switching amplifier, inducing an emergent polarization where the model spontaneously allocates its reasoning style to match task demands. The synthesized data and models are released at Hugging Face, and the code is released at Github.

[Figure 2]

[Figure 3]

### 1 Introduction

Reinforcement learning (RL) has proven to be the critical post-training mechanism for eliciting capabilities in Large Language Models (LLMs) (Ouyang et al., 2022; DeepSeek-AI, 2025; Team, 2025a). However, as the ambition of RL expands from single-domain optimization (e.g., math) (Le et al., 2022; Shao et al., 2024; Wang et al., 2025a) to general-purpose alignment (Lee et al., 2024; Shen et al., 2025), the Reward Model (RM) faces the challenge of providing reliable feedback for increasingly complex queries from diverse, real-world scenarios (Liu et al., 2025c; Li et al., 2025a). Addressing this challenge requires a shift in RM design. Inspired by how CoT (Wei et al., 2022; Yang et al., 2025) trades inference-time compute for enhanced generalization performance, the community has increasingly adopted Generative Reward Models (GRMs) (Zheng et al., 2023; Yuan et al., 2024; Zhang et al., 2025a). By prompting an explicit evaluation rationale prior to conclusion, GRMs aim to transfer the robust generalization observed in CoT generation to the task of reward modeling.

Building on these successes, existing GRM methods predominantly leverage CoT by simply scaling its length (Chen et al., 2026; 2025; Zhang et al., 2025c), feeding it with massive evaluation signals, such as fine-grained features (Kim et al., 2024) or multi-perspective critiques (Ankner et al., 2024). However, prior CoT studies (Sprague et al., 2025; Besta et al., 2025; Wang et al., 2024b; Kambhampati et al., 2024) have established that longer CoTs do not universally guarantee performance gains; rather, the optimal structural bias diverges significantly across domains. Crucially, recent insights from test-time scaling (Zhang et al., 2026; 2025b) provide a theory for this divergence, identifying parallel thinking and sequential thinking as two fundamental, orthogonal mechanisms for amplifying intelligence. Conceptually, reasoning-heavy tasks (e.g., math, code) necessitate sequential verification to ensure deductive rigor (Wang et al., 2024a; Liu et al., 2025a; Lightman et al., 2024), whereas semantic-heavy tasks (e.g., open-ended generation) benefit from parallel exploration to ensure comprehensive coverage of diverse possibilities (Zheng et al., 2026; Pan et al., 2025).

Drawing on this distinction, we argue that advancing RM requires shifting focus from merely scaling CoT length to aligning its reasoning mechanisms with task demands. Specifically, this necessitates a transition from static, one-size-fits-all CoT templates toward a mix reasoning mechanism. Thus, we propose Mix-GRM, which implements a dynamic mix reasoning mechanism within a unified reward modeling framework. Specifically, we introduce a synthesis framework that reconfigures raw, unstructured rationales into two distinct long CoTs: Breadth-CoT (B-COT) and Depth-CoT (D-COT). To achieve this, we first decouple unstructured rationales into atomic “Prin-

*Work done during the internship at Tencent Hunyuan: <qzhang732-c@my.cityu.edu.hk> †Correspondence to: Chen Ma <chenma@cityu.edu.hk>, Yufei Wang <garyyfwang@tencent.com>, Can Xu

<leocaxu@tencent.com>

ciple–Judgment–Verdict” units. This modularity allows us to reassemble the units into syntactically unified but structurally diverse paths. To illustrate, a B-COT is synthesized by the parallel aggregation of units across diverse principles (e.g., combining an ‘Accuracy’ unit with a ‘Clarity’ unit) to ensure coverage. Conversely, D-COT extends the CoT by first performing a direct reasoning pass to solve the instruction, thereby enabling a re-evaluated judgment grounded in the generated reasoning pass to ensure soundness. To cultivate mechanism-adaptive alignment, we construct a synergistic mixture dataset by pairing B-COT with subjective preference tasks and D-COT with objective correctness tasks. We first initialize the model via SFT on this mixture and subsequently optimize it through RLVR using normal RM datasets, where only final labels are available.

Comprehensive experiments across five standard benchmarks yield three critical conclusions: (1) Universal SOTA Performance and Downstream Utility: Mix-GRM establishes a new state-of-the-art, consistently surpassing strong baselines like Skywork-Reward and FARE-8B on general reward benchmarks. Crucially, this superiority extends to practical downstream tasks: Mix-GRM demonstrates best-in-class utility in both Offline RL (DPO) and Test-time Scaling (Best-of-N). (2) Divergent Roles of Reasoning Mechanisms: Our analysis reveals that B-COT predominantly benefits subjective preference but degrades objective correctness, while D-COT excels in correctness at the cost of preference. This confirms that the efficacy of a reasoning mechanism is task-dependent. (3) RLVR as a Switching Amplifier: Mixed mechanisms provide a superior base for RL. RLVR boosts Mix-GRM by a larger margin than the Base-GRM. Our analysis demonstrates that RLVR automatically sharpens the mechanism allocation—spontaneously converging on B-COT for preference and D-COT for correctness. This confirms that optimizing how a model thinks is more critical for post-training efficacy than simply scaling how long it writes.

### 2 Related Work

##### 2.1 Generative Reward Model

Generative Reward Models represent a paradigm shift from scalar regression to explicit reasoning. Developing alongside the prompting-based “LLM-as-a-Judge” paradigm (Zheng et al., 2023), GRMs are explicitly trained to generate natural language rationales alongside preference decisions (Yuan et al., 2024). Driven by the transformative success of long CoT, the research trajectory in this field has pivoted toward continuously extending the length of these rationales. To achieve this, many work leverages RL to explicitly elicit and stabilize longer CoT traces (Chen et al., 2026; 2025; Whitehouse et al., 2026), while complementary efforts utilize detailed rubrics/checklists to synthetically expand evaluation coverage (Kim et al., 2024; Liu et al., 2026; Gunjal et al., 2026; Viswanathan et al., 2025). However, while these strategies successfully scale the quantity of reasoning, they typically rely on static, task-agnostic structures, overlooking the critical nuance that the optimal reasoning mechanism is intrinsically task-dependent.

##### 2.2 Breadth and Depth in Chain-of-Thought

The evolution of CoT is fundamentally characterized by the continuous exploration of diverse structures (Shinn et al., 2023; DeepSeek-AI, 2025). Beyond simple linear chains, frameworks such as Tree of Thoughts (Yao et al., 2023) and Graph of Thoughts (Besta et al., 2025) introduce branching and recurrent topologies, framing reasoning as a structured search over partial thoughts. Complementing these complex structures, approaches like Skeleton-of-Thought (Ning et al., 2024) and Self-Consistency (Wang et al., 2023) demonstrate the efficacy of parallel exploration, leveraging lateral breadth to enhance robustness and coverage. Collectively, these studies establish that reasoning is not structure-agnostic; rather, specific topological priors—ranging from deep sequential trees to broad parallel ensembles—are required to unlock optimal performance across distinct domains (Sprague et al., 2025), a distinction that our work formally adapts to reward modeling.

### 3 Methodology

We propose the Mix-GRM, a framework designed to dynamically align the reasoning mechanism with intrinsic task demands. Moving beyond static, unstructured rationale sequences, our approach formalizes evaluation into two orthogonal CoTs: B-COT, which enforces the lateral aggregation of diverse principles, and D-COT, which necessitates the expansion of judgment. As illustrated in Figure 1, our methodology comprises three key phases: modular schema standardization (§3.2), mechanism synthesis (§3.3), and mechanism-adaptive alignment (§3.4).

##### 3.1 Problem Formulation

Supposing {yA, yB} denote two candidate responses generated by two assistants A and B for a given task instruction x, a normal GRM M produces an output sequence consisting of an explicit evaluation rationale c followed by a

preference verdict v, comparing the quality of yA and yB.

(c, v) = M(yA, yB | x).

II. Mechanism Synthesis

III. Training & Inference

I. Standardization

|Instruction<br><br>Response A/B<br><br>GRM<br><br>Breadth-<br><br>CoT<br><br>DepthCoT<br><br>|
|---|

||Breadth-CoT<br><br>|[Figure 6]<br><br>#1.Principle of Instruction Adherence|
|---|
|Judgment: Assistant B directly addresses the user's question …… <B>|
<br><br>|[Figure 7]<br><br>#1.Principle of Analytical Depth|
|---|
|Judgment: Assistant B provides a detailed analysis of the … … <B>|
<br><br>N×<br><br>. . . . . .|
|---|
<br><br>[Figure 8]<br><br>Preference:<br><br>Merge<br><br>In an alternate universe where World War II was orchestrated differently, imagine you are a distinguished historian … …|
|---|

|[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]<br><br>Instruction<br><br>Response A Response B<br><br>[Figure 12]<br><br>LLM<br><br>[Figure 13]<br><br>……<br><br>However, there are subtle differences in their approaches.\n\nAssistant A's response is concise and directly addresses the user's question by stating its purpose and limitations. It also explicitly mentions being \"harmless and honest,\" which adds a layer of<br><br>transparency about its design<br><br>……<br><br>Final verdict: [[B]]<br><br>Unstructured Evaluation Rationale<br><br>Distill<br><br>Principle-Judgment-Verdict Schema<br><br>[Figure 14]<br><br>Extract<br><br>[Figure 15]<br><br>#2.Principle.of...... . .<br><br>|[Figure 16]<br><br>#1. Principle of Transparent Self -<br><br>[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>Disclosure :<br><br>A high-quality response should explicitly state its core operational constraints and ethical design parameters.|
|---|
<br><br>|[Figure 20]<br><br>Judgment : Assistant A's response "explicitly mentions being 'harmless<br><br>and honest,’ . . . . . .|
|---|
<br><br>[Figure 21]<br><br>Sub - Verdict: <A><br><br>[Figure 22]<br><br>[Figure 23]|
|---|

[Figure 24]

[Figure 25]

[Figure 26]

|Correctness:<br><br>[Figure 27]<br><br>||[Figure 28]<br><br>#1.Principle of Practical Insight|
|---|
|[Figure 29]<br><br>Reasoning - Guided Judgment:<br><br>[Figure 30]<br><br>[Figure 31]<br><br>The evaluation needs deep reasoning: The option that is NOT a best practice for financial modeling is:\n\n**Add comments to cells to give more information ……<br><br>Assistant B provides more practical reasons …… <B><br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]|
<br><br>Depth-CoT|
|---|
<br><br>Guide<br><br>Which of the following is NOT a best practice for financial modeling?\n\nchoose from the following: . . . . . .<br><br>[Figure 35]<br><br>[Figure 36]<br><br>Reasoning|
|---|

SFT RLVR

|[Figure 37]<br><br>＞<br><br>Offline RL Test-time Scaling<br><br>GRM|
|---|

[Figure 38]

- Figure 1: The pipeline of the Mix-GRM. (i) Standardization: We extract raw rationales into modular Principle–Judgment–Verdict units. (II) Mechanism Synthesis: We reconstruct modules into B-COT for preference or D-COT for correctness. (III) Training & Inference: Following SFT and RLVR training, the model achieves mechanism-adaptive alignment, automatically deploying the optimal mechanism for inference and providing reliable signals for downstream tasks like Offline RL and test-time scaling.

[Figure 39]

[Figure 40]

The objective is to ensure that the v aligns with human preference. In our framework, we denote the full input triplet as I = (x, yA, yB).

##### 3.2 Modular Schema Standardization

Conventional GRMs typically produce the rationale c as an unstructured, free-form sequence. Inspired by recent checklist-based evaluation (Viswanathan et al., 2025), which advocates for the atomization of the complex evaluation process into checklist-driven points, we propose to reconfigure these raw rationales into a structured “PrincipleJudgment-Verdict” Schema (Figure 1, Stage I). By transforming tangled rationales into atomic units, we ensure that the RM’s reasoning process is both interpretable and granularly verifiable. Formally, we utilize a LLM to parse the raw c into structured atomic units S:

S = {(pk, jk, vk)}kK=1, where pk denotes a discrete evaluation Principle (e.g., “Instruction Adherence”), jk represents the specific Judgment (e.g., “Response B directly addresses...”) analyzing that principle, and vk is the following Sub-Verdict (e.g., “<B> is Better”). Here, K typically ranges from 3 to 5.

This atomic decomposition yields cleaner learning signals and ensures syntactic uniformity (Li et al., 2025b), ensuring that performance gains are driven by thinking mechanisms (i.e., Breadth vs. Depth) rather than superficial stylistic patterns.

##### 3.3 Mechanism Synthesis

Building on the S, we introduce a dual-track synthesis pipeline (Figure 1, Stage II) to synthesize B- and D-COT as follows:

B-COT Synthesis. We define B-COT as the parallel aggregation of distinct principles, designed to overcome the narrow focus of single-pass rationale. In subjective preference tasks, where a “good” response is defined by the simultaneous satisfaction of multi-dimensional factors (e.g., tone, helpfulness, and creativity), single-track reasoning often fixates on dominant traits while overlooking subtle, fine-grained details. By exploring diverse reasoning paths concurrently, parallel thinking provides a deliberative breadth that aligns with the multifaceted nature of human preference. To simulate parallel thinking, we treat independent sampling as a stochastic exploration

of the instruction’s evaluative manifold. By sampling N independent rationales {cn}nN=1 from multiple cognitive trajectories, we elicit a diverse set of hidden principles that might otherwise remain dormant. These rationales

are parsed into structured schemas {Sn} and subsequently unified via an LLM-based Merge & Deduplicate transformation Tmerge:

CB = T merge n = 1N(p, j, v) ∈ Sn .

Here, we filter out lowest-frequency principles. This synthesis yields a comprehensive, non-redundant spectrum of principles, effectively expanding the model’s horizontal evaluative scope.

D-COT Synthesis. We define D-COT as the expansion of judgment to ensure substantive reasoning soundness by mitigating superficial shortcuts. In contrast to subjective preferences, a “good” response in objective correctness

tasks depends on rigorous logical constraints (e.g., mathematical proofs or functional code). Normal rationales often fixate on surface-level fluency (e.g., professional tone or formatting) while failing to verify the underlying logical validity. By enforcing the sequential verification of logical dependencies, sequential thinking provides a deductive rigor that naturally aligns with the strict requirements of objective correctness. To simulate sequential thinking, we first elicit a Reasoning Trace z—a self-solving pass derived from x that explicitly outlines the optimal solution paths required for a correct response. Recognizing that depth-oriented reasoning demands higher cognitive load per unit, we intentionally trade off horizontal coverage for deductive rigor by sampling a focused subset Ssub ⊂ S (typically |K| ≤ 3). In this stage of Reasoning-Guided Judgment, each unit’s judgment is re-generated as a derivative of the trace z:

j˜k = T refine(pk | z) To ensure the evaluative process is transparent and explicitly grounded in the model’s own logic, we inject z directly into the lead unit j˜1. The final CD is constructed by serializing these refined units, transforming the verdict into a substantive analytical process anchored by the trace z.

- 3.4 Mechanism-Adaptive Alignment Training proceeds in two stages (Figure 1, Panel III): SFT on mixture CoT datasets, followed by GRPO (Shao et al.,

- 2024) to align verdicts with human labels.

SFT. Following Frick et al. (2025), we categorize general RM training data into two domains: Preference (subjective) and Correctness (objective). We construct the mixture dataset Dmix by assigning CB to preference instances and CD to correctness instances. We first initialize the policy πθ via SFT on Dmix. Given the I, the model is trained to generate the corresponding CoT c ∈ {cB, cD} alongside the verdict v.

RLVR via GRPO. To optimize verdict accuracy, we employ RLVR via GRPO (Shao et al., 2024), rewarding the model solely for consistency with ground-truth labels:

JGRPO(θ) = E I∼D

{oi}∼πθold

1 G

G

∑

i=1

πθ(oi|I) πθold(oi|I)

Aˆi − βDKL(πθ||πref)

The reward is defined by verdict consistency: a positive reward is assigned if the generated verdict vi matches the ground-truth human label, and −1 otherwise. This process acts as a switching amplifier, inducing an emergent polarization: the model spontaneously learns to couple B-COT with preference tasks and D-COT with correctness tasks to maximize rewards, as empirically verified in §5. This confirms that the model autonomously converges on the optimal thinking style for each domain.

4 Experiment

We evaluate Mix-GRM across three objectives: (1) Overall Performance against SoTA baselines; (2) Mechanism Efficiency to quantify the domain-specific benefits of B- and D-COT; and (3) Downstream Utility in Offline RL and Test-time Scaling.

- 4.1 Experimental Setup

General Reward Benchmarks. We employ five widely recognized benchmarks tailored for general-purpose reward modeling: RewardBench (Lambert et al., 2025), RewardBench-v2 (Malik et al., 2025), RMB (Zhou et al.,

- 2025a), RM-Bench (Liu et al., 2025b), and PPE (Frick et al., 2025). These benchmarks encompass a broad spectrum of tasks, ranging from common tasks like math, coding, and open-ended chat, to specialized capabilities including factuality and instruction-following. For Overall Performance, we report standard benchmark-level pairwise comparison accuracy to assess the rewarding capability. For granular Mechanism Efficiency analysis, we aggregate instances from these benchmarks and re-categorize them into two fundamental domains, Correctness and Preference, based on their original task metadata. Detailed statistics and specific domain mappings for these benchmarks are provided in Appendix A.3.

Base Model and Training Data Source. We employ Qwen3-8B-Base (Team, 2025b) trained on a composite corpus 30,000 samples (9K SFT, 21K RLVR) spanning five datasets: HelpSteer3 (Wang et al., 2025b) (chat, stem & multilingual), Code-Preference (coding), Math-DPO (math), WildGuard (Han et al., 2024) (safety), and OffsetBias (Park et al., 2024) (instruction following). Detailed sampling protocols and statistical distributions are provided in Appendix A.2. Other Training Implementation Setting is in Sec. A.

Benchmarks RB-V1† RB-V2† RM-BENCH RMB PPE Avg. Reference: Proprietary Models

Models Stage Data

DeepSeek-V3.2 – – 95.5 92.1 91.4 83.9 69.0 86.4 Gemini-3-Flash – – 95.3 91.1 93.8 79.2 76.4 87.2

###### Open-Source Baselines

Skywork-Reward-8B BT 44K 93.9 79.7 72.4 74.4 61.7 76.5 JudgeLRM-7B RL 100K 79.0 55.6 78.5 73.1 57.9 68.8 RM-R1-7B (Distill) SFT , RL 9K , 64K 83.5 48.7 76.6 65.1 62.0 67.2 RM-R1-7B (Instruct) SFT , RL 9K , 64K 82.3 61.4 75.1 69.9 62.0 70.1 FARE-8B SFT 2.5M 86.3 73.4 74.1 83.2 62.5 75.9 RubricRM-8B SFT 36K 86.7 71.9 74.0 78.5 62.5 74.7 DeepSeek-GRM-16B SFT , RL 1.2M , 237K 76.8 56.0 63.5 70.8 59.1 65.2

###### Ours: Mix-GRMs

- Stage I: SFT-trained

Base-GRM SFT 9K 84.5 64.7 77.0 79.2 61.1 73.3 Mix-GRM (Ours) SFT 9K 87.2 67.8 79.2 78.9 62.1 75.1

- Stage II: RLVR-trained

Base-GRM SFT , RL 9K , 21K 89.0 74.0 78.8 78.5 64.0 76.9 Mix-GRM (Ours) SFT , RL 9K , 21K 91.8 77.5 82.7 80.1 64.8 79.4

- Table 1: Performance of RMs on reward benchmarks. Among open-source models, the highest score per column is bolded, and the second-highest is underlined. “Overall” denotes the average score within each benchmark. Proprietary LLMs (gray rows) are included for reference. †RB-V1/V2 refers to RewardBench v1 and v2.

Models

Preference Domain Correctness Domain

Overall RB-V1 RB-V2 RM-B† RMB PPE Avg. RB-V1 RB-V2 RM-B† RMB PPE Avg.

Baselines

FARE-8B 85.0 57.3 66.9 82.9 59.6 70.4 85.2 67.3 63.0 88.1 63.3 73.3 71.9 RubricRM-8B 82.4 56.0 62.2 77.5 64.9 68.6 87.6 64.2 57.6 86.5 60.4 71.3 70.0 DeepSeek-GRM 80.6 59.6 64.0 76.8 59.8 68.2 76.6 55.8 56.6 86.8 56.8 66.5 67.4

Ours: Mix-GRMs

- Stage I: SFT-trained

Base-GRM 81.6 55.5 63.3 80.5 60.1 68.2 84.1 63.7 67.7 86.4 59.1 72.2 70.2 Mix-GRM (Breadth) 83.7 59.1 65.9 77.9 59.5 69.3↑1.1 81.1 60.2 64.1 86.8 58.7 70.2↓2.0 69.8 Mix-GRM (Depth) 80.3 50.2 70.6 70.1 58.6 65.9↓2.3 88.0 63.7 66.7 81.1 64.7 72.8↑0.6 69.4 Mix-GRM 84.9 55.7 71.2 78.7 59.2 70.0↑1.8 88.4 65.8 67.7 81.9 63.7 73.5↑1.3 71.8

- Stage II: RLVR-trained

Base-GRM 83.0 58.0 68.5 73.8 61.4 68.9↑0.7 89.8 69.5 69.9 89.5 63.4 76.4↑4.2 72.7 Mix-GRM (Breadth) 86.2 58.8 70.1 79.2 60.7 71.0↑2.8 82.8 63.4 64.3 86.5 60.7 71.5↓0.7 71.3 Mix-GRM (Depth) 85.2 57.8 75.6 75.4 61.2 71.0↑2.8 91.8 70.3 72.9 87.4 66.2 77.7↑5.5 74.4 Mix-GRM 86.2 64.4 72.7 78.1 61.7 72.6↑3.7 92.2 72.5 74.5 88.9 65.4 78.7↑6.5 75.7

- Table 2: Performance of RMs grouped by domain. “Avg.” denotes the domain average. We annotate the performance gap relative to the Base-GRM in SFT baseline within the same stage using colored subscripts (↑ for gain, ↓ for drop). Highest score per column is bolded, second-highest is underlined. †RM-B refers to RM-Bench.

Baselines. We compare our proposed RM with 7 top-tier RMs across two paradigms: (1) Discriminative: represented by Skywork-Reward-v0.2-Llama-3.1-8B (Liu et al., 2024), a leading scalar model trained via BradleyTerry modeling; and (2) Generative: encompassing RL-driven reasoning models (JudgeLRM-7B (Chen et al., 2025), RM-R1-Instruct (Chen et al., 2026), RM-R1-Distill, DeepSeek-GRM-16B) (Liu et al., 2025c), synthetic scaling methods (FARE-8B (Xu et al., 2026)), and rubric-based approaches RubricRM-8B (Liu et al., 2026). Notably, RubricRM-8B incorporates two-stage LLMs consisting of a rubric generator and a rubric-based judge.

##### 4.2 Overall Performance in Benchmarks

- Table 1 validates the effectiveness of our Mix-GRM through three dimensions.

Instruction-Following Mathematical Reasoning

Models

ALPACA-V2 ARENA-HARD Avg. GSM8K MATH STEM TABMWP Avg. SFT 6.4 4.2 5.3 75.1 25.2 38.6 40.9 45.0 DPO Training (Different RMs)

→ RubricRM-8B 8.5 12.5 10.5 76.0 26.9 41.4 38.8 45.9 → FARE-8B 8.9 15.1 12.0 75.7 26.9 39.0 41.4 45.8 → RM-R1-Instruct 7.9 14.3 11.1 76.3 26.5 38.5 41.7 45.8 → DeepSeek-GRM-16B 8.0 14.1 11.1 75.6 26.6 38.7 41.6 45.6 → Ours (Mix-GRM) 9.2 15.0 12.1 77.6 27.1 39.0 41.9 46.4

- Table 3: Performance of DPO-trained policy models using different reward models on instruction-following and math-reasoning benchmarks. “Avg.” is the average score of all benchmarks in each domain. In each column, the highest score is bolded and the second-highest is underlined.

###### MATH

###### CHAMP

###### MBPP+

###### BIGCODEBENCH

60

60

80

60

Baseline

Oracle

70

50

50

50

60.2

60

43.2

39.6

40

40

40

50

34.9

Accuracy

30

30

40

30

30

20

20

20

20

10

10

10

10

0

0

0

0

SKYCRITICFASE-8BRMR1-7BGRM-16B OURS

SKYCRITICFASE-8BRMR1-7BGRM-16B OURS

SKYCRITICFASE-8BRMR1-7BGRM-16B OURS

SKYCRITICFASE-8BRMR1-7BGRM-16B OURS

- Figure 2: Best-of-10 performance across four challenging reasoning and coding benchmarks. Mix-GRM (ours) consistently achieves the highest accuracy across all tasks, effectively identifying solutions in both mathematical and code generation scenarios. Red and green lines denote random and oracle selection baselines.

Effectiveness of Mixture SFT : Via mixture SFT alone, Mix-GRM achieves a remarkable average score of 75.1. This performance surpasses GRMs requiring computationally expensive RL to elicit long-CoT capabilities—outperforming RM-R1-Instruct by 5.0 and DeepSeek-GRM-16B by 9.9. Furthermore, it beats RubricRM-8B (+0.4), which relies on a complex but static rubric-template CoT. This confirms that aligning reasoning mechanisms serves as a potent alternative strategy, alongside approaches focused on RL exploration or static template engineering.

Superiority of Data Efficiency : Mix-GRM achieves these gains with substantially less data. While FARE-8B relies on massive scaling (≈ 2.5M samples) to reach 75.9, Mix-GRM attains a comparable 75.1 in the SFT stage using merely 9K samples. This finding highlights that optimizing CoT mechanisms yields a substantially higher training signal density, enabling data efficiency compared to brute-force dataset expansion.

Switching Amplification via RLVR : Mix CoT maximizes the efficacy of the RLVR stage, unlocking greater performance gains than unstructured CoT. RLVR boosts Mix-GRM by 4.3 (75.1 → 79.4), compared to a 3.6 gain for Base-GRM (73.3 → 76.9). Consequently, the performance gap over the Base-GRM widens from 1.8 (SFT) to 2.5 (RLVR), confirming that the aligned mechanism offers a more exploitable base for the RL. Furthermore, our subsequent analysis (Sec 5) reveals that these gains are fundamentally underpinned by an emergent polarization in mechanism allocation, where RLVR sharpens the model’s reasoning style to match task-specific demands.

##### 4.3 Mechanism Efficiency

- Table 2 reveals that mechanism efficacy is strictly task-dependent. In the SFT stage, we observe a distinct performance trade-off: B-COT improves Preference via lateral coverage but degrades Correctness (72.2 → 70.2), whereas D-COT enhances deductive soundness but fails in Preference (68.2 → 65.9). These results indicate that simply extending CoT length does not guarantee universal gains; while principle expansion facilitates multidimensional evaluation, it offers no inherent advantage for deep reasoning. However, Mix-GRM overcomes these limitations through a synergistic mutual enhancement. By integrating orthogonal strengths, it not only surpasses the Base-GRM (70.2 → 71.8) but surprisingly outperforms specialized single-mode models on their respective strongholds (e.g., exceeding Depth-only on Correctness). This synergy becomes critical during the RLVR stage, where single-mode mechanisms encounter hard performance ceilings—most notably, Mix-GRM (Breadth) plateaus on correctness tasks. In contrast, the Mix-GRM enables RL optimization to reach a superior ceiling (78.7). This

confirms that the CoT structure itself acts as a bottleneck for RL optimization; the mixed structures do not merely inherit component strengths but construct a robust reasoning framework that transcends the inherent limitations of isolated mechanisms.

##### 4.4 Downstream Utility

To validate the practical utility of Mix-GRM, we apply it to two downstream applications: (i) serving as a reward signal for Offline Reinforcement Learning, and (ii) acting as a verifier for Test-time Scaling. We provide detailed descriptions of these application settings in Appendix B.

Reward Model for Offline Reinforcement Learning. In Offline RL via Direct Preference Optimization (DPO) (Rafailov et al., 2023), RM constructs high-quality preference pairs (yw, yl) to supervise policy alignment. Table 3 shows that models trained on these signals achieve a peak win rate of 12.1 in instruction-following, surpassing FARE-8B (12.0) and RubricRM (10.5). Crucially, this alignment gain does not compromise reasoning capabilities; in the math domain, Mix-GRM maintains a SOTA accuracy of 46.4, edging out RubricRM (45.9) and RM-R1-Instruct (45.8). Specifically, Mix-GRM achieves 77.6% on GSM8K, demonstrating a clear lead over the SFT baseline (75.1%). These results confirm that Mix-GRM provides reliable supervision, enabling policies to internalize both helpfulness and correctness.

Reward Model for Test-time Scaling. For test-time scaling, leveraging increased inference-time compute to enhance generalization, Mix-GRM functions as a robust verifier to re-rank candidates to identify the optimal solution via Best-of-N selection. Following the JETTS protocol (Zhou et al., 2025b), we evaluate N = 10 samples from a Llama-3.1-8B generator across 4 diverse benchmarks: MATH and CHAMP (math), as well as MBPP+ and BigCodeBench (coding). As shown in Figure 2, our method consistently secures the highest accuracy, setting a new SOTA for 8B-scale rerankers. The performance advantage is particularly pronounced in reasoning-heavy tasks; for instance, on MATH, our model achieves an accuracy of 43.2%, outperforming the RL-driven RM-R1 (37.7%) and the data-intensive FARE-8B (35.2%). This confirms that ours provides a more discriminative signal for logical verification than methods relying on massive data scaling or generic RL.

### 5 Analysis

Switching CoT Mechanism Analysis. Visualizing structural transformations (Figure 3) reveals how our pipeline reshapes reasoning mechanisms. Here, Singlemode strategies show extreme trade-offs: Mix-GRM (Breadth) expands horizontally (high principle count), while Mix-GRM (Depth) extends vertically (long judgments). In contrast, Mix-GRM (SFT) achieves a robust union of both, which is further expanded into a broader reasoning manifold by RLVR. First, the polarization of Breadth and Depth baselines confirms the rigidity of static templates, which create capability blind spots by sacrificing either reasoning depth or semantic coverage. Second, the balanced profile of Mix-GRM (SFT) indicates successful internalization of different distinct mechanisms. Most pivotally, the global expansion during RLVR validates our hypothesis of mechanism polarization. By optimizing for verdict accuracy, the model spontaneously converges on domain-specific mechanism biases—amplifying D-COT for correctness while reinforcing B-COT for preference. This emergent specialization confirms that our proposed alignment is not a handcrafted heuristic, but an inherent structural necessity discovered by the model to maximize evaluation efficacy.

###### SFT Stage

Avg Tokens # of Judgment

Mix-GRM (Breadth) Mix-GRM (Depth) Mix-GRM (SFT)

(Correctness)

| |[Figure 45]<br><br>[Figure 46]<br><br>|
|---|---|
|[Figure 47]<br><br>[Figure 48]| |

| | |
|---|---|
| | |

Mix-GRM (RLVR)

# Breadth CoT pct (Preference)

Avg # Principles (Preference)

# Depth CoT pct (Correctness)

Figure 3: Structural evolution of CoT mechanisms. The chart tracks 4 indicators: the average token length per judgment, average principle count, and the percentage of CoT classified as having Breadth or Depth characteristics.

Emergent Polarization Analysis. To determine whether the emergent polarization toward domain-specific reasoning (i.e., B-COT for Preference and D-COT for Correctness) is actively driven by RLVR or merely inherited from SFT priors, we analyzed the distribution of generated CoT structures on the test set using specific structural indicators (e.g., principle counts and trigger phrases). Although the SFT training data was completely domainaligned, the post-SFT model achieved only a 73% structural match rate during inference. Remarkably, following RLVR—which relies exclusively on final verdict supervision without explicit structural labels—this alignment surged to 95%. This substantial improvement confirms that the model transcends static SFT priors, autonomously

###### (b) Impact of Principle Selection (N = 4)

###### (a) Impact of Aggregation Scale

- 68

- 69

- 70

- 71

- 72

- 73

- 74

Preference Correctness

72.0

| |
|---|

72.7

Vanilla Baseline

71.5

71.8

71.0

71.6

Accuracy(%)

71.4

70.5

70.0

70.1

69.9

69.5

Vanilla Baseline (N=1)

69.0

Preference

68.5

Correctness

68.0

Random Selection (10 Principles)

No Selection (Full Pool)

Consistency Selection (10 Principles)

1 2 3 4

Number of Aggregated CoTs (N)

Figure 4: Ablation of B-COT synthesis. (a) Aggregation Scale: Performance as aggregated rationales (N) increases from 1 (Vanilla) to 4. (b) Principle Selection: Comparison of Random, Full, and Consistency (Top-10) selection from the N = 4 pool. Orange/blue lines denote Preference/Correctness; dashed lines indicate the Vanilla baseline.

learning to select and optimize the most effective reasoning structure for each domain during the reinforcement learning phase.

Scaling & Selection Analysis. To understand the mechanics of B-COT, we decouple the impact of quantity (aggregation scale) from quality (principle selection) as shown in Figure 4.1) Quantity (Scaling): Figure 4(a) demonstrates that performance improves monotonically as the number of parallel CoTs (N) increases from 1 to 4. This confirms that “breadth" functions by expanding coverage; by aggregating diverse perspectives, the model minimizes the risk of overlooking critical error patterns.2) Quality (Selection): However, more is not always better. Figure 4(b) compares three strategies within the N = 4 pool: BreadthRand, BreadthFull, and BreadthTop10, where Top-10 means ten most frequent principles appearing across 4 CoTs. We observe a clear hierarchy: BreadthTop10 > BreadthFull > BreadthRand. While the full pool improves over random sampling, it is the Top-10 consensus that achieves the highest gains (71.8/72.7). This suggests a denoising effect where low-frequency principles introduce noise, while high-frequency ones form a more robust “reasoning consensus.” Thus, representativeness—not just volume—is vital for robust breadth.

Computational Overhead and Token Cost Analysis. To ensure a transparent comparison of computational overhead, we designed our pipelines to maintain strict algorithmic fairness and token parity across reasoning styles. First, regarding data synthesis, both B-COT (which samples twice and merges rationales) and D-COT (which generates a solve trace before synthesizing the final CoT) utilize approximately two reasoning passes. Notably, we strictly prohibit introducing new information during the merge phase, limiting it solely to merging and deduplication; therefore, it does not constitute an additional reasoning pass. This parity ensures that observed performance differences stem from structural efficacy rather than raw compute disparities. Second, as detailed in Table 4, the token consumption during Data Synthesis, RLVR Training, and Inference remains highly comparable across configurations. These marginal differences confirm that B-COT, D-COT, and our adaptive Mix-CoT operate within the exact same order of magnitude of compute cost.

CoT Style Reasoning Passes Avg. SFT Tokens Avg. RLVR Rollouts Avg. Inference Tokens

D-CoT 2 (Trace + Gen) 624 682 702 B-CoT 2 (N = 2 Sample) 711 830 824 Mix-CoT (Ours) Adaptive 648 725 731

- Table 4: Accounting table reporting the compute and token costs. Token consumption across our proposed methods remains within the same order of magnitude.

Case Study. Table 5 elucidates the structural mechanisms behind the observed double dissociation. Case 1 demonstrates why B-COT dominates preference tasks: acting as a multi-dimensional scanner, it successfully penalizes a detailed but language-mismatched response by validating lateral constraints (e.g., Linguistic Alignment), whereas D-COT exhibits attentional tunneling,” fixating on verifying historical facts while missing the highlevel language mismatch. Conversely, Case 2 reveals why D-COT is essential for correctness: its step-by-step derivation acts as a logic probe, allowing it to spot subtle factual hallucinations (e.g., K > Mg) hidden within a lengthy explanation. Here, B-COT actively fails due to feature interference,” where it mistakes superficial comprehensiveness (length and formatting) for logical validity. This confirms that while Breadth is necessary for satisfying diverse user preferences, Depth is the non-negotiable driver for rigorous verification.

- Table 5: Case Study. Case 1 shows how B-COT aggregates diverse principles to identify subtle preference nuances. Case 2 shows how D-COT performs step-by-step verification to catch logical errors.

CASE 1: PREFERENCE DOMAIN (B-COT WINS) Instruction: アフガニスタンがパキスタンの傀儡というのは本当ですか？ (Is it true that Afghanistan is a puppet of Pakistan?) Response A (Rejected): Response B (Chosen): [Language: English] "A sensitive topic! ...While Pakistan has historically exerted significant influence..."

[Language: Japanese] 「アフガニスタンとパキスタンの関係については... 簡単に『傀儡』と断言するのは適切ではなく...」

Reasoning Comparison

- ✗ Vanilla-CoT Assistant A offers a comprehensive breakdown of historical context... Verdict: [[A]] (Fail: Ignored language mismatch)

✓ B-COT 1. Principle of Linguistic Alignment: "Assistant B’s response is in Japanese... Sub-Verdict: «B»"

- 2. Principle of Contextual Nuance: "Assistant A provides a detailed explanation... Sub-Verdict: «A»"
- 3. Principle of Cultural Sensitivity: "..." Final Verdict: [[B]] (Success)

- ✗ D-COT 1. Principle of Comprehensive Contextualization: The evaluation needs deep reasoning, and let’s solve the instruction: アフガニスタンが....

Based on this, Assistant A excels... Sub-Verdict: «A»

2. Principle of Balanced Presentation: ... Final Verdict: [[A]] (Fail: Tunnel vision on content depth)

CASE 2: CORRECTNESS DOMAIN (D-COT WINS) Instruction: On the basis of oxidation-reduction potential, which of the following is most likely to occur? ... Response A (Chosen): Response B (Rejected):

"The reactivity of metals... The order is: Alkali > ... > Zn > ... > Ag. Based on this, the most likely reaction is: D. Zn + 2Ag(NO3)..." (Correct)

"The reactivity... Alkali > Alkaline earth... Analysis: B. Mg + 2KNO3 -> 2K + Mg(NO3)2." (Error: K > Mg.)

Reasoning Comparison

- ✗ Vanilla-CoT To evaluate the responses, let’s consider factors: 1. Helpfulness; 2. Relevance... Assistant B goes further by analyzing multiple options... Final

Verdict: [[B]] (Fail: Fooled by length/detail)

- ✗ B-COT 1. Principle of Comprehensive Option Analysis: "Assistant B analyzes all provided options (A-H)... Sub-Verdict: «B»"

2. Principle of Informative Detail: "Assistant B includes more elements... Sub-Verdict: «B»" Final Verdict: [[B]] (Fail: Superficial heuristic)

✓ D-COT 1. Principle of Comprehensive Analysis: The evaluation needs to deep reasoning, and let’s solve the instruction: To determine the most likely reaction... The correct order is Alkali > ... Given the options, the most likely reaction is Option D. ... Assistant B correctly identifies the importance but incorrectly identifies Option B... Sub-Verdict: «A»

2. Principle of Direct Relevance: Assistant A directly addresses the question... Sub-Verdict: «A» Final Verdict: [[A]] (Success)

- 6 Conclusion

This work demonstrates that beyond mere length scaling, the reliability of GRMs is fundamentally driven by the integration of different reasoning mechanisms. By introducing Mix-GRM, we prove that the frontier of reward modeling lies in synergizing two orthogonal reasoning mechanisms: B-COT for multi-dimensional coverage and D-COT for judgment soundness. Through mechanism-adaptive alignment, Mix-GRM ensures that the RM’s reasoning mechanism is precisely calibrated to the nature of the task. Ultimately, these findings shift the focus of GRM development from brute-force expansion to structural optimization.

### Limitations

While Mix-GRM significantly enhances evaluation reliability through mechanism alignment, we identify two primary limitations that warrant further investigation:

Granularity of the Reasoning Manifold. Our framework successfully captures the double dissociation between Subjective Preference and Objective Correctness, which we identify as the dominant axes of the reasoning manifold. However, this dichotomy represents a coarse-grained mapping of the diverse alignment landscape. Real-world tasks often exist on a continuous spectrum or involve hybrid demands that intricately blend deductive rigor with multidimensional nuances. While we prove that the model’s reasoning structure spontaneously converges toward these two primary poles, our current categorization may act as a low-rank approximation of a higher-dimensional space of mechanisms. Future work could explore more granular taxonomies to achieve even more precise task-mechanism calibration.

Rigidity in Explicit Hybrid Tasks. While our analysis demonstrates that RLVR induces an intrinsic convergence toward specialized reasoning poles, this emergent polarization may introduce structural rigidity when encountering highly complex, cross-domain scenarios. The current framework excels at aligning specialized mechanisms with their respective domains, but our findings ultimately prove that “one size does not fit all” for reward structures.

Emerging real-world applications, such as agentic Deep Research, increasingly demand an explicit, dynamic fusion of rigorous deductive logic and high-quality, stylistic writing. The spontaneous sharpening of reasoning styles might currently come at the cost of this generalist flexibility. Therefore, a logical next step for future research is to develop dedicated hybrid slicing benchmarks to explicitly evaluate the trade-off between style and logic, alongside the design of more sophisticated, fine-grained hybrid structures (e.g., soft-routing mechanisms) that can fluidly transition across the reasoning manifold.

### References

Zachary Ankner, Mansheej Paul, Brandon Cui, Jonathan D. Chang, and Prithviraj Ammanabrolu. Critique-out-loud reward models, 2024. URL https://arxiv.org/abs/2408.11791.

Maciej Besta, Florim Memedi, Zhenyu Zhang, Robert Gerstenberger, Guangyuan Piao, Nils Blach, Piotr Nyczyk, Marcin Copik, Grzegorz Kwa´sniewski, Jürgen Müller, Lukas Gianinazzi, Ales Kubicek, Hubert Niewiadomski, Aidan O’Mahony, Onur Mutlu, and Torsten Hoefler. Demystifying chains, trees, and graphs of thoughts. Transactions on Pattern Analysis and Machine Intelligence, pp. 10967–10989, 2025.

Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, and Bingsheng He. Judgelrm: Large reasoning models as a judge, 2025. URL https://arxiv.org/abs/2504.00050.

Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru WANG, Yu Zhang, Denghui Zhang, Tong Zhang, Hanghang Tong, and Heng Ji. RM-r1: Reward modeling as reasoning. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=1ZqJ6jj75q.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: boosting language models with scaled ai feedback. In International Conference on Machine Learning, 2024.

Team DeepSeek-AI. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, pp. 633–638, 2025.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. In Conference on Empirical Methods in Natural Language Processing, 2023.

Yann Dubois, Percy Liang, and Tatsunori Hashimoto. Length-controlled alpacaeval: A simple debiasing of automatic evaluators. In Conference on Language Modeling, 2024.

Evan Frick, Tianle Li, Connor Chen, Wei-Lin Chiang, Anastasios Nikolas Angelopoulos, Jiantao Jiao, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. How to evaluate reward models for RLHF. In The Thirteenth International Conference on Learning Representations, 2025.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean M. Hendryx. Rubrics as rewards: Reinforcement learning beyond verifiable domains. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=c1bTcrDmt4.

Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. WILDGUARD: open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms. In International Conference on Neural Information Processing Systems, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2021.

Subbarao Kambhampati, Karthik Valmeekam, Lin Guan, Mudit Verma, Kaya Stechly, Siddhant Bhambri, Lucas Saldyt, and Anil Murthy. Position: Llms can’t plan, but can help planning in llm-modulo frameworks. In International Conference on Machine Learning, 2024.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. Prometheus: Inducing fine-grained evaluation capability in language models. In International Conference on Learning Representations, 2024.

Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. MAWPS: A math word problem repository. In Conference of the North American Chapter of the Association for Computational Linguistics, pp. 1152–1157, 2016.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, and Hannaneh Hajishirzi. RewardBench: Evaluating reward models for language modeling. In Findings of the Association for Computational Linguistics: NAACL 2025, pp. 1755–1797, 2025.

Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven C.H. Hoi. CodeRL: mastering code generation through pretrained models and deep reinforcement learning. In International Conference on Neural Information Processing Systems, 2022.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. Rlaif vs. rlhf: scaling reinforcement learning from human feedback with ai feedback. In International Conference on Machine Learning, 2024.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. From live data to high-quality benchmarks: The arena-hard pipeline, 2024. URL https://lmsys.org/blog/ 2024-04-19-arena-hard/.

Yi-Chen Li, Tian Xu, Yang Yu, Xuqin Zhang, Xiong-Hui Chen, Zhongxiang Ling, Ningjing Chao, Lei Yuan, and Zhi-Hua Zhou. Generalist reward models: Found inside large language models, 2025a. URL https: //arxiv.org/abs/2506.23235.

Zhuang Li, Yuncheng Hua, Thuy-Trang Vu, Haolan Zhan, Lizhen Qu, and Gholamreza Haffari. SCAR: Data selection via style consistency-aware response ranking for efficient instruction-tuning of large language models. In Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12756–12790, 2025b.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=v8L0pN6EOi.

Chengwu Liu, Ye Yuan, Yichun Yin, Yan Xu, Xin Xu, Zaoyu Chen, Yasheng Wang, Lifeng Shang, Qun Liu, and Ming Zhang. Safe: Enhancing mathematical reasoning in large language models via retrospective step-aware formal verification. In Annual Meeting of the Association for Computational Linguistics, 2025a.

Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in llms, 2024. URL https://arxiv.org/abs/ 2410.18451.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and LINGMING ZHANG. Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation. In Conference on Neural Information Processing Systems, 2023.

Tianci Liu, Ran Xu, Tony Yu, Ilgee Hong, Carl Yang, Tuo Zhao, and Haoyu Wang. OpenRubrics: Towards scalable synthetic rubric generation for reward modeling and llm alignment, 2026. URL https://arxiv.org/abs/ 2510.07743.

Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. RM-bench: Benchmarking reward models of language models with subtlety and style. In International Conference on Learning Representations, 2025b.

Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. Inference-time scaling for generalist reward modeling, 2025c. URL https://arxiv.org/abs/2504.02495.

Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning, 2023. URL https://arxiv.org/abs/2209.14610.

Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Morrison, Noah A. Smith, Hannaneh Hajishirzi, and Nathan Lambert. Rewardbench 2: Advancing reward model evaluation, 2025. URL https://arxiv.org/abs/ 2506.01937.

Yujun Mao, Yoon Kim, and Yilun Zhou. CHAMP: A competition-level dataset for fine-grained analyses of LLMs’ mathematical reasoning capabilities. In Findings of the Association for Computational Linguistics, pp. 13256–13274, 2024.

Xuefei Ning, Zinan Lin, Zixuan Zhou, Zifu Wang, Huazhong Yang, and Yu Wang. Skeleton-of-thought: Prompting LLMs for efficient parallel generation. In International Conference on Learning Representations, 2024.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In International Conference on Neural Information Processing Systems, 2022.

Jiayi Pan, Xiuyu Li, Long Lian, Charlie Snell, Yifei Zhou, Adam Yala, Trevor Darrell, Kurt Keutzer, and Alane Suhr. Learning adaptive parallel reasoning with language models, 2025. URL https://arxiv.org/abs/ 2504.15466.

Junsoo Park, Seungyeon Jwa, Ren Meiying, Daeyoung Kim, and Sanghyuk Choi. OffsetBias: Leveraging debiased data for tuning evaluators. In Findings of the Association for Computational Linguistics: EMNLP, pp. 1043–1067, 2024.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=HPuSIXJaa9.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Wei Shen, Guanlin Liu, YuYue, Ruofei Zhu, Qingping Yang, Chao Xin, and Lin Yan. Exploring data scaling trends and effects in reinforcement learning from human feedback. In Annual Conference on Neural Information Processing Systems, 2025.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In International Conference on Neural Information Processing Systems, 2023.

Zayne Rea Sprague, Fangcong Yin, Juan Diego Rodriguez, Dongwei Jiang, Manya Wadhwa, Prasann Singhal, Xinyu Zhao, Xi Ye, Kyle Mahowald, and Greg Durrett. To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning. In International Conference on Learning Representations, 2025.

Kimi Team. Kimi k1.5: Scaling reinforcement learning with llms, 2025a. URL https://arxiv.org/abs/

2501.12599. Qwen Team. Qwen3 technical report, 2025b. URL https://arxiv.org/abs/2505.09388. Vijay Viswanathan, Yanchao Sun, Xiang Kong, Meng Cao, Graham Neubig, and Tongshuang Wu. Checklists

are better than reward models for aligning language models. In Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=RPRqKhjrr6.

Junqiao Wang, Zeng Zhang, Yangfan He, Zihao Zhang, Xinyuan Song, Yuyang Song, Tianyu Shi, Yuchen Li, Hengyuan Xu, Kunyu Wu, Xin Yi, Zhongwei Wan, Xinhang Yuan, Zijun Wang, Kuan Lu, Menghao Huo, Tang Jingqun, Guangwu Qian, Keqin Li, Qiuwu Chen, and Lewei He. Enhancing code llms with reinforcement learning in code generation: A survey, 2025a. URL https://arxiv.org/abs/2412.20367.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Mathshepherd: Verify and reinforce LLMs step-by-step without human annotations. In Annual Meeting of the Association for Computational Linguistics, 2024a.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=1PL1NIMMrw.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. Mmlu-pro: a more robust and challenging multi-task language understanding benchmark. In International Conference on Neural Information Processing Systems, 2024b.

Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Hoo-Chang Shin, Felipe Soares, Alexander Bukharin, Ellie Evans, Yi Dong, and Oleksii Kuchaiev. Helpsteer3-preference: Open human-annotated preference data across diverse tasks and languages. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025b. URL https://openreview.net/forum?id=lovsIkZLnI.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In International Conference on Neural Information Processing Systems, 2022.

Chenxi Whitehouse, Tianlu Wang, Ping Yu, Xian Li, Jason E Weston, Ilia Kulikov, and Swarnadeep Saha. J1: Incentivizing thinking in LLM-as-a-judge via reinforcement learning. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=dnJEHl6DI1.

Austin Xu, Xuan-Phi Nguyen, Yilun Zhou, Chien-Sheng Wu, Caiming Xiong, and Shafiq Joty. Foundational automatic evaluators: Scaling multi-task generative evaluator training for reasoning-centric domains. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/ forum?id=89Ei7PVpNl.

Shiming Yang, Yuxuan Tong, Xinyao Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning. In International Conference on Machine Learning, 2025. URL https://openreview.net/ forum?id=OLodUbcWjB.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: deliberate problem solving with large language models. In International Conference on Neural Information Processing Systems, 2023.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. In International Conference on Learning Representations, 2024.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding language models. In International Conference on Machine Learning, 2024.

Duzhen Zhang, Zhong-Zhi Li, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu, Junhao Zheng, Xiuyi Chen, Yingying Zhang, Fei Yin, Jiahua Dong, Zhijiang Guo, Le Song, and Cheng-Lin Liu. From System 1 to System 2: A Survey of Reasoning Large Language Models . IEEE Transactions on Pattern Analysis & Machine Intelligence, pp. 3335–3354, 2026.

Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction. In International Conference on Learning Representations, 2025a.

Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Wenyue Hua, Haolun Wu, Zhihan Guo, Yufei Wang, Niklas Muennighoff, Irwin King, Xue Liu, and Chen Ma. A survey on test-time scaling in large language models: What, how, where, and how well?, 2025b. URL https://arxiv.org/abs/2503.24235.

Qiyuan Zhang, Yufei Wang, Yuxin Jiang, Liangyou Li, Chuhan Wu, Yasheng Wang, Xin Jiang, Lifeng Shang, Ruiming Tang, Fuyuan Lyu, and Chen Ma. Crowd comparative reasoning: Unlocking comprehensive evaluations for LLM-as-a-judge. In Annual Meeting of the Association for Computational Linguistics, 2025c.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

Tong Zheng, Hongming Zhang, Wenhao Yu, Xiaoyang Wang, He Xing, Runpeng Dai, Rui Liu, Huiwen Bao, Chengsong Huang, Heng Huang, and Dong Yu. Parallel-r1: Towards parallel thinking via reinforcement learning. In International Conference on Learning Representations, 2026. URL https://openreview. net/forum?id=wOmjeBN6hP.

Enyu Zhou, Guodong Zheng, Binghai Wang, Zhiheng Xi, Shihan Dou, Rong Bao, Wei Shen, Limao Xiong, Jessica Fan, Yurong Mou, Rui Zheng, Tao Gui, Qi Zhang, and Xuanjing Huang. RMB: Comprehensively benchmarking reward models in LLM alignment. In International Conference on Learning Representations, 2025a.

Yilun Zhou, Austin Xu, PeiFeng Wang, Caiming Xiong, and Shafiq Joty. Evaluating judges as evaluators: The JETTS benchmark of LLM-as-judges as test-time scaling evaluators. In International Conference on Machine Learning, 2025b. URL https://openreview.net/forum?id=CgJEHynkJt.

Terry Yue Zhuo, Vu Minh Chien, Jenny Chim, Han Hu, Wenhao Yu, Ratnadira Widyasari, Imam Nur Bani Yusuf, Haolan Zhan, Junda He, Indraneil Paul, Simon Brunner, Chen GONG, James Hoang, Armel Randy Zebaze, Xiaoheng Hong, Wen-Ding Li, Jean Kaddour, Ming Xu, Zhihan Zhang, Prateek Yadav, Naman Jain, Alex Gu, Zhoujun Cheng, Jiawei Liu, Qian Liu, Zijian Wang, David Lo, Binyuan Hui, Niklas Muennighoff, Daniel Fried, Xiaoning Du, Harm de Vries, and Leandro Von Werra. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions. In International Conference on Learning Representations, 2025.

### A Training Implementation

- A.1 Hyperparameters Setting We provide the detailed hyperparameter settings in the Table 6 and Table 7.

Hyperparameters Values

Epochs 2 Learning rate 2e−5 Batch Size 128 (gradient accumulation steps = 16) Seq Length 12,288 Weight Decay 0. Warmup 5% linear warmup

Table 6: Hyperparameter settings for SFT.

Hyperparameters Values

Training Steps 100 Learning Rate 1e−6 Batch Size 128 KL Loss Coefficient 0.001 KL Coefficient 0.001 Rollouts n= 8 using vLLm with temperature 0.8

Table 7: Hyperparameter settings for RL.

- A.2 Training Data Source Details

To cultivate general rewarding capabilities, it is essential to curate a training corpus that encompasses diversified real-world scenarios. We construct our dataset by performing stratified random sampling from representative data sources, ensuring balanced coverage across distinct alignment domains, including general chat, STEM, coding, math, safety, multilingual, and instruction following. The specific source datasets, their corresponding domains, and the sampling statistics are detailed in Table 8.

Source Dataset Domain Samples

HelpSteer-3 (Single-Turn)

General Chat 4,973 STEM 2,321 Code 4,322 Multilingual 3,260

Code-Preference Code 4,000 Math-DPO Math 4,000 WildGuard Safety 4,000 OffsetBias Instruction Following 4,000

Total – 30,876

Table 8: Composition and statistics of the training data sampled from domain-specific sources.

- A.3 Training Data Synthesis Details

To synthesize the CoT data for SFT, we utilized DeepSeek-v3 (0324 snapshot) as the backbone generator. The generation process was configured with a sampling temperature of T = 0.8 to promote diversity in the trajectories while maintaining logical coherence. Notably, we abstain from consistency filtering: Contrary to common practices that discard samples where the synthesized verdict diverges from the ground-truth human label, our empirical verification reveals that training on the full synthesized CoTs yields superior performance compared to aggressive filtering, regardless of verdict consistency.

- A.4 Training Offline Reinforcement Learning Details

To strictly control for temporal data leakage and ensure a fair comparison with the release dates of our evaluation benchmarks, we select Llama-3-8B as our base foundation model. The offline reinforcement learning pipeline consists of two phases: SFT initialization and DPO.

Policy Initialization (SFT). We first derive a supervised policy model by fine-tuning Llama-3-8B on a composite dataset. This dataset ensures basic instruction-following and reasoning capabilities, consisting of the UltraChat dataset (Ding et al., 2023) and a random subset of 40K samples from MetaMathQA (Yu et al., 2024). We train the model for 2 epochs using a learning rate of 2e−5 and a maximum sequence length of 2,048 tokens. This SFT model serves as the initial policy πref for the subsequent DPO stage.

DPO Data Construction via RM Labeling. To evaluate the practical utility of different RMs, we employ them to annotate preferences on a unified source dataset. The prompt source comprises 10K instructions randomly sampled from UltraFeedback (Cui et al., 2024) and 40K instructions from MetaMathQA. For each instruction x, we generate N = 5 diverse candidate responses using gpt-4o-mini with a temperature of 0.8.

Table 9: Task coverage of the evaluated general reward benchmarks.

Benchmark Tasks Samples

REWARDBENCH Chat, Math, Code, Safety 2,985 REWARDBENCH-V2 Focus, IF, Factuality, Math, Safety, Ties 1,865 RM-BENCH Chat, Math, Code, Safety 11,943 RMB Harmfulness, Helpfulness (General, Code) 14,725 PPE (Exclude Tie) Chat, MMLU-Pro, GPQA, IFEval, MBPP 22,991

We adopt a Pairwise Scoring Aggregation strategy to construct the final preference pairs (x, yw, yl). Specifically, for the set of 5 responses, we generate all possible combinations of pairs ((52) = 10 pairs). The target RM evaluates each pair, assigning +1 point to the preferred response (chosen) and 0 to the non-preferred one (rejected). After traversing all pairs, we calculate the cumulative score for each response. The response with the highest total score is selected as the positive sample (yw), and the response with the lowest total score is selected as the negative sample (yl). These labeled pairs are then used to train the policy via DPO.

### B Evaluation Implementation

- B.1 Core Benchmarks There is a list of benchmarks and corresponding task coverage.
- B.2 Benchmarks for Offline Reinforcement Learning Evaluation

To comprehensively assess the policy derived from DPO, we conduct evaluations across two distinct domains: mathematical reasoning and open-ended instruction following.

Mathematical Reasoning. We employ a suite of four challenging datasets to evaluate the model’s deductive logic and problem-solving capabilities: GSM8k (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), MAWPS (Koncel-Kedziorski et al., 2016), and TabMWP (Lu et al., 2023). These benchmarks cover a wide spectrum of difficulty, ranging from grade-school arithmetic to competition-level mathematics and tabular processing.

Instruction Following. For general alignment and conversational versatility, we utilize two widely adopted benchmarks: AlpacaEval-2 (Dubois et al., 2024) and Arena-Hard v0.1 (Li et al., 2024). Evaluation is performed using an auto-evaluator in a head-to-head setting, where the model’s responses are compared against a baseline reference to determine win rates. We strictly adhere to the officially recommended configurations for reproducibility.

- B.3 Benchmarks for Test-time Scaling Evaluation

Following the JETTS setup (Zhou et al., 2025b), we perform Best-of-10 reranking evaluations where the model selects the optimal solution from a mixed pool of candidate responses. We report results on the four most challenging subsets of the benchmark: MATH (Hendrycks et al., 2021) for mathematical reasoning, CHAMP (Mao et al., 2024) for competition-level math, along with MBPP+ (Liu et al., 2023) and BigCodeBench (Zhuo et al., 2025) for code generation. This selection tests the model’s ability to identify correct reasoning paths in complex scenarios.

### C Prompts Template

To align with established community standards, our Vanilla-CoT generation employs the representative prompts originally introduced in MT-Bench (Zheng et al., 2023) and RewardBench (Lambert et al., 2025). Upon generating the raw Vanilla-CoT using the standard prompts, we employ a specialized extraction prompt to parse the unstructured text into the modular “Principle–Judgment–Verdict‘’ schema. Leveraging the parsed schemas, we introduce specialized prompts to synthesize the two target morphologies. For Breadth-CoT, the synthesis process entails merging modular components derived from at least two Vanilla-CoT responses, followed by a deduplication step to ensure diverse coverage. In contrast, the synthesis of Depth-CoT relies on a reasoning-guided evaluation mechanism. We initially prompt the model to reason the instruction deeply. We then use this generated reasoning to ground the re-assessment of selected principles extracted from the parsed schemas, discarding their previous rationales to ensure the new judgments are purely driven by rigorous reasoning.

##### Prompt for Vanilla-CoT Generation

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider as many factors as possible. Begin your evaluation by comparing the two responses and provide a through reasoning. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your reasoning, output your final verdict by strictly following this format: ¨[[A]]ïf assistant A is better, ¨[[B]]ïf assistant B is better. [Instruction] instruction

- [The Start of Assistant A’s Answer]

- {response_a}

- [The End of Assistant A’s Answer]

[The Start of Assistant B’s Answer] {response_b}

- [The End of Assistant B’s Answer]

### D Reward Model Performance Across Preference and Correctness

To provide a more granular view of our model’s efficacy, we report detailed performance across specific tasks based on the meta-data provided by each benchmark. We categorize these tasks into two distinct tables: Table 10 for subjective preference tasks and Table 11 for objective correctness tasks. This fine-grained reporting serves as a detailed decomposition of the mechanism-level performance discussed in the main text, offering deeper empirical evidence for the mechanism-task synergy between B-COT and D-COT.

- Table 10: Performance of RMs on preference-related sub-tasks. “Avg.” is the average score among sub-tasks. Best per column is bolded; second-best is underlined.

Reward Bench

Reward Bench-v2

RM Bench

Models

RMB PPE Avg. CHAT FOCUS IF CHAT HELPFULNESS HUMAN IF

Open-sourced Reward Models

JudgeLRM-7B 75.8 46.8 31.3 68.4 79.3 60.2 54.3 59.4 RM-R1-7B (Distill) 75.3 58.7 24.4 58.7 63.2 57.1 53.0 55.8 RM-R1-7B (Instruct) 80.5 85.3 28.8 64.7 65.1 57.1 53.0 62.1 FARE-8B 85.0 78.4 36.3 66.9 82.9 63.4 55.7 67.0 RubricRM-8B 82.4 78.2 33.8 62.2 77.5 63.8 66.0 66.3 DeepSeek-GRM-16B 80.6 79.2 40.0 64.0 76.8 61.7 57.9 65.7

Our Proposed Reward Models SFT-trained

Base-GRM 81.6 76.6 34.4 63.3 80.5 64.3 55.9 65.2 Mix-GRM (Breadth) 83.7 84.5 33.8 65.9 77.9 63.5 55.6 66.4 Mix-GRM (Depth) 80.3 72.2 28.1 70.6 70.1 63.1 54.1 62.6 Mix-GRM 84.9 79.6 31.9 71.2 78.7 62.0 56.3 66.4

###### RLVR-trained

Base-GRM 83.0 86.7 29.4 68.5 73.8 65.8 57.1 66.3 Mix-GRM (Breadth) 86.2 88.9 28.8 70.1 79.2 66.0 55.3 67.8 Mix-GRM (Depth) 85.3 89.3 26.3 75.6 75.4 66.0 56.4 67.8 Mix-GRM 86.2 91.3 37.5 72.7 78.1 65.9 57.4 69.9

##### Prompt for Schema Extraction

PRIMARY TASK: Your mission is to analyze a given reasoning Chain-of-Thought from a generative reward model. From this CoT, you will extract, define, and refine the specific, detailed principles (or rubrics, criteria) it uses to judge the quality of AI-generated responses. For each principle, you must provide a corresponding analysis that traces it directly back to the original text. INSTRUCTIONS: You will be given a CoT text below. Please follow these four steps precisely:

- 1. Deconstruct the CoT: First, perform a close reading of the entire CoT. Identify all explicit evaluation criteria mentioned as well as any implicit judgments or preferences revealed in the model’s comparative language.
- 2. Extract the Core Idea of Each Criterion: For each criterion, do not simply use the high-level category name. Your goal is to uncover the specific

description of that criterion as used by the model. Ask yourself: What specific actions, qualities, or content does the model praise or criticize? What makes one response “more accurate" or “clearer" according to this specific CoT?

- 3. Formulate and Refine the Principle: Convert each core idea you extracted into a formal, normative, and reusable principle.

- 3.1 Name It: Give the principle a clear and descriptive name that captures its essence (e.g., “Principle of Factual Precision," “Principle of Structural Clarity").
- 3.2 Define It: Write the principle as a concise, actionable, and universal rule. It should be an instructive statement about what constitutes a high-quality response.
- 3.3 Be Specific: Avoid vague terms. Instead of “The response should be relevant," specify how it should be relevant based on the CoT’s logic, such as “A relevant response must directly and unambiguously address the user’s primary

question."

- 3.4 Be Normative: Phrase it as a standard to be met (e.g., “A high-quality response must...").

- 4. Provide Corresponding Judgment: For each principle you formulate, you must write a brief “CoT Judgment Extraction." To do this, quote or closely paraphrase specific phrases from the CoT that support your formulation.
- 5. Conclude the sub-verdict in this Judgment: For the principle and corresponding judgment, you should conclude this verdict in this segment. OUTPUT FORMAT: You must follow this format strictly for your entire response.

. . .

- ### 1. Principle of [Descriptive Name]: [Your refined, normative principle statement.] Judgment: [In this principle, what judgment on Response A and Response B quotes or paraphrases from the source CoT.] Sub-Verdict: «A/B», In this principle, the judgment judge which assistant Better]

- ### 2. Principle of [Descriptive Name]: [Your refined, normative principle statement.] Judgment: [In this principle, what judgment on Response A and

Response B quotes or paraphrases from the source CoT.]*** Sub-Verdict: «A/B», In this principle, the judgment judge which assistant Better] (Continue this structure for all principles identified in the CoT)

. . . Extract and Analyse the following CoT Text

{Vanilla-CoT}

### E Sensitivity and Robustness

The data synthesis process consists of two primary phases: the sampling of raw rationales and the subsequent synthesis of B-CoT/D-CoT. To guarantee the reliability of this pipeline, we conduct empirical analyses focusing on pipeline stability and robustness to noise.

- Table 11: Performance of RMs on correctness-related sub-tasks. “Avg.” is the average within this block. Best per column is bolded; second-best is underlined.

Models RewardBench RewardBench-v2 RM-Bench RMB PPE Avg.

CODE MATH FACTUALITY MATH CODE MATH CODE MMLU-PRO MATH GPQA MBPP Open-sourced Reward Models

JudgeLRM-7B 81.6 77.2 53.8 76.5 51.0 86.7 82.1 57.2 65.5 51.3 52.3 66.8 RM-R1-7B (Distill) 91.9 93.7 28.3 73.2 53.3 85.8 74.8 66.7 89.4 56.3 64.4 70.7 RM-R1-7B (Instruct) 81.7 84.1 42.6 67.8 56.7 72.7 74.7 67.0 89.1 55.9 64.8 68.8 FARE-8B 88.1 82.3 65.8 68.9 57.0 69.1 88.1 63.2 79.3 55.2 55.4 70.2 RubricRM-8B 93.6 81.7 50.8 77.6 55.4 59.8 86.5 60.9 75.5 52.8 52.4 67.9 DeepSeek-GRM-16B 84.0 69.1 49.4 62.3 51.5 61.7 86.8 55.2 64.3 54.1 53.7 62.9

Our Proposed Reward Models SFT-trained

Base-GRM 91.0 77.2 46.0 81.4 57.1 78.4 86.4 59.9 71.8 52.2 52.5 68.5 Mix-GRM (Breadth) 90.1 72.0 51.1 69.4 54.0 74.1 86.8 60.0 70.3 51.9 52.4 66.6 Mix-GRM (Depth) 89.8 86.1 45.1 75.4 56.2 77.1 81.1 66.8 83.6 54.7 53.7 70.0 Mix-GRM 88.9 87.9 55.7 76.0 55.8 79.6 81.9 63.9 82.1 54.8 54.0 71.0

RLVR-trained

Base-GRM 93.2 86.4 62.0 77.0 61.7 78.1 89.5 64.6 84.1 54.3 50.5 72.9 Mix-GRM (Breadth) 96.3 69.3 55.7 71.0 58.0 70.6 86.5 61.7 75.2 53.0 52.8 68.2 Mix-GRM (Depth) 94.6 89.0 61.8 78.7 64.4 81.4 87.4 67.0 86.5 55.6 55.5 74.7 Mix-GRM 95.4 89.0 65.8 79.2 66.6 82.5 88.9 65.0 86.7 54.8 55.2 75.4

Quantitative Evaluation of Pipeline Stability. To ensure our extraction and refinement modules do not introduce systematic errors or degrade data quality, we tracked the solution accuracy of the synthesized data at each pipeline stage on the training set. As shown in Table 12, the accuracy remains highly stable across the transformations. The merge step (B-CoT) resolves contradictions and improves accuracy, while the D-CoT generation maintains high fidelity to the correct reasoning paths. This confirms that the intermediate processing steps reliably preserve data quality without catastrophic degradation.

Synthesis Stage Raw Rationale Merge B-CoT Generate D-CoT Accuracy (%) 87.1 90.2 88.5

- Table 12: Solution accuracy tracked across different stages of the synthesis pipeline, demonstrating the stability of the intermediate transformations.

Robustness to Noise. A common concern with LLM-generated rationales is the necessity of strict filtering to ensure perfect correctness. We conducted an ablation study during the SFT stage to measure the impact of noise. We compared a model trained on 9K strictly verified CoT data (filtered for correct final verdicts) against one trained on 9K noisy CoT data (without strict correctness filtering). As shown in Table 13, the SFT performance difference is negligible. This indicates that our method is highly robust to noise in the synthesized rationales and does not require perfect accuracy from the extraction pipeline to be effective. Consequently, we adopt the unfiltered data setting for our main experiments to significantly reduce the computational cost of data curation without sacrificing performance.

Training Data (9K) w/o Filtering (Noisy) w/ Filtering (Verified) SFT Performance (%) 69.8 70.1

Table 13: SFT performance comparison between noisy and strictly filtered synthesized data.

Furthermore, we utilize the open-weights DEEPSEEK-V3 model for both schema extraction and raw rationale generation. The effectiveness of our synthesis pipeline with an open-source model further underscores its robustness and generalizability, proving it is not overly sensitive to the choice of the underlying LLM.

##### Prompt for Breadth-CoT Generation

PRIMARY TASK: You are provided with a series of lists, each containing Principles, Judgments, and Sub-Verdicts derived from an independent analysis of a Chain-of-Thought (CoT). Your mission is to merge these lists into a single, master list of unique evaluation principles. INSTRUCTIONS FOR MERGING AND SYNTHESIS:

###### 1. Deduplication and Semantic Grouping:

* Compare all Principles with the corresponding Judgments across all provided lists. * Identify and group principles that are semantically similar, even if they use different wording (e.g., “Principle of Precision" and “Principle of Correcteness" are likely the same concept).

###### 2. Principle Refinement:

* For each semantic group, synthesize the most concise, actionable, and specifically-detailed statement for the Principle Description. * Select the most descriptive and formal Name for the refined principle.

###### 3. Judgment Synthesis:

* For the refined principle, create a new, synthesized Judgment block. This block should consist of a curated selection of the most illustrative quotes and paraphrases from the original Judgments across all source lists that led to the consolidated principle. This new Judgment serves as the combined evidence for the principle.

###### 4. Merge Count:

* COUNT THE SOURCES: For each synthesized principle, you must count the total number of original, distinct principles/judgments from the source lists that were merged to create it. This number is the Merge Count.

###### 5. Sub-Verdict Aggregation:

* The final Sub-Verdict for the synthesized principle must reflect the aggregated trend. Since the judgments are now synthesized, simply use the majority verdict (e.g., if a principle appeared 4 times with [[B]] and 1 time with [[A]], conclude [[B]]). If the verdicts are balanced (e.g., 2 [[A]] and 2 [[B]]), state [[MIXED]].

###### 6. Strict Output Adherence:

* Maintain the exact four-part format for every final entry. The output must be one continuous list of unique, synthesized principles. SOURCE LISTS TO MERGE:

- [Insert List 1 Here]
- [Insert List 2 Here]
- [Insert List 3 Here] (Continue for all lists) OUTPUT FORMAT:

You must follow this exact format for your final, merged response.

***

- ### 1. Principle of [Refined, Descriptive Name]: [The synthesized, normative principle statement.] Judgment: [A synthesis of the most relevant quotes/paraphrases from the source Judgments that supports this

consolidated principle.]*** Merge Count: [The total number of original source principles/judgments that were merged to form this entry.] Sub-Verdict: «A/B/MIXED», The aggregate verdict for this principle across all CoTs.]

- ### 2. Principle of [Refined, Descriptive Name]: [The synthesized, normative principle statement.] Judgment: [A synthesis of the most relevant quotes/paraphrases from the source Judgments that supports this

consolidated principle.]*** Merge Count: [The total number of original source principles/judgments that were merged to form this entry.] Sub-Verdict: «A/B/MIXED», The aggregate verdict for this principle across all CoTs.]

(Continue this structure for all unique, synthesized principles)

***

##### Prompt for Depth-CoT Verification

PRIMARY TASK: Your role is to critically assess the quality of two competing responses (Assistant A and Assistant B) against the user’s question, leveraging the expert reasoning as the ultimate ground truth. MANDATORY NON-BIAS RULES: Avoid all position biases (do not favor the first

response presented). Do not allow the length or formatting of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective, clinical, and data-driven as possible.

PRINCIPLE-BASED EVALUATION Given some potential principles, {principles}, you should choose the most critical principle (Preferably one principle, with a maximum of three) from them and then evaluate the two Chatbot responses (A and B) based on the choosed principle. This evaluation must directly reference the deep reasoning to instruction and **must strictly adhere to the following output format for

each principle:** EXPERT REASONING: {reasoning} ### Principle of [Critical Principle Name]:

Judgment: [Give your specific and detailed evaluation in this principle, and if you are referring the this reasoning, you **MUST** quote using ‘<Answer>‘] Sub-Verdict: «A/B/MIXED», «A» if assistant A is better in this principle, «B»

if assistant B is better in this principle, «MIXED» if assistant A and B is Tie. After providing your complete principle-based evaluation, output your final verdict by strictly following this format: ¨[[A]]ïf assistant A is better, ¨[[B]]ïf assistant B is better.

