# arXiv:2603.20667v1[cs.SE]21Mar2026

## REVERE: REFLECTIVE EVOLVING RESEARCH ENGINEER FOR SCIENTIFIC WORKFLOWS

Balaji Dinesh Gangireddi1, Aniketh Garikaparthi1, Manasi Patwardhan1, Arman Cohan2 1TCS Research 2Yale University

{dinesh.gangireddi, aniketh.g, manasi.patwardhan}@tcs.com arman.cohan@yale.edu

ABSTRACT

Existing prompt-optimization techniques rely on local signals to update behavior, often neglecting broader and recurring patterns across tasks, leading to poor generalization; they further rely on full-prompt rewrites or unstructured merges, resulting in knowledge loss. These limitations are magnified in research-coding workflows, which involve heterogeneous repositories, underspecified environments, and weak feedback, where reproducing results from public codebases is an established evaluation regime. We introduce Reflective Evolving Research Engineer (REVERE), a framework that continuously learns from Global Training Context, recognizes recurring failure modes in cross-repository execution trajectories, distills them into reusable heuristics, and performs targeted edits across three configurable fields: the system prompt, a task-prompt template, and a cumulative cheatsheet. REVERE, via this reflective optimization framework, improves performance over prior state-of-the-art expert-crafted instructions on research coding tasks by 4.50% on SUPER, 3.51% on ResearchCodeBench and 4.89% on ScienceAgentBench across their respective metrics. These results demonstrate that agents equipped with mechanisms for continual learning and global memory consolidation can meaningfully evolve their capabilities over time.

1 INTRODUCTION

While recent progress of Large language models (LLMs) on short-horizon well-specified coding tasks is promising (Yang et al., 2024; White et al., 2025; Gauthier, 2024), reliability degrades substantially in research-code reproduction (Starace et al., 2025; Xiang et al., 2025; Garikaparthi et al., 2026; Bogin et al., 2024; Hua et al., 2025), due to fundamentally different demands on agents. These include coordinating long-horizon tasks under weak and delayed feedback, inferring tacit assumptions,and accumulating procedural knowledge across heterogeneous research frameworks (Trehan & Chopra, 2026; Peng & Wang, 2025; Wang et al., 2026). Prior agentic systems (Starace et al., 2025; Wang et al., 2025) targeting research reproducibility, typically rely on static prompts. More complex systems such as (Seo et al., 2025; Lin et al., 2025) further decompose high-level tasks through multi-agent workflows; while this can improve reliability, they still operate within fixed contexts and predefined strategies. As a result, these systems struggle to adapt to the evolving conventions and diverse open-ended nature of research coding tasks.

Recent works on self-refinement (Shinn et al., 2023; Madaan et al., 2023; Majumder et al., 2024a) improve reasoning through iterative feedback, but remain instance-specific, motivating prompt-level and experience-based adaptation methods (Agrawal et al., 2025; Opsahl-Ong et al., 2024; Zhao et al., 2024) to address this limitation. However these approaches still rely primarily on heuristic prompt sampling and local evaluation signals. While this works well in short-horizon settings, these methods tend to overfit on recent outcomes rather than learning generalizable patterns. Suzgun et al. (2025); Zhang et al. (2025c) attempt to move towards accumulating reusable strategies, however, they still rely on local evaluation signals, which can lead to local optima(Shi et al., 2025) and also, operate over bounded context rather than a persistent global memory across executions, limiting long-term knowledge retention. Moreover, most prompt-adaptation frameworks update behavior through full

[Figure 1]

[Figure 2]

| |[Figure 3]| | |
|---|---|---|---|
| | | | |

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

|[Figure 11]| |
|---|---|
|[Figure 12]| |
| | |

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

|[Figure 20]| |
|---|---|
| | |

[Figure 21]

[Figure 22]

[Figure 23]

| | |
|---|---|
| | |

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

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

- Figure 1: The REVERE Framework employing an iterative optimization loop where a Reflector Agent dynamically adapts three editable prompt fields (System Prompt, Task Prompt, and Cheatsheet) using our code-based edit mechanism, global training context and evaluation feedback

prompt regeneration, increasing the risk of semantic drift and knowledge loss as prompts grow. Structured editing and search-based methods (Zhang et al., 2025c; Schnabel & Neville, 2024) can mitigate this issue, yet they typically involve more complex implementations.

What is needed instead is an agent capable of learning from its own execution trajectories over time by identifying recurring failure modes, distilling them into reusable heuristics, and maintaining them within a persistent global context. Such an agent should apply targeted, non-destructive updates to prompts, plans, and tool-use strategies across tasks without gradient-based retraining, enabling more stable knowledge accumulation and helping the system move beyond local optima toward more globally effective strategies.

To address these gaps, we introduce REVERE , a framework for building self-adapting agents tailored to research-coding workflows. REVERE adopts a simple, unified design built around three core components: (1) prompt adaptation over configurable fields, which defines fields to be optimized and adapts them based on observed failure modes and evaluation feedback; (2) a Global Training Context, which preserves and aggregates experience across tasks and adaptations; and (3) targeted code-based updates via a Reflector module, which applies structured edits to prompts and other optimizable fields. Together, these components allow REVERE to progressively refine its behavior, reuse prior strategies, and update its reasoning without overfitting to specific tasks. Our work makes the following contributions:

- • We formulate research code reproduction as a test-time adaptation problem for LLM agents, highlighting concrete failure modes specific to research repositories.
- • We demonstrate that REVERE improves overall performance on SUPER (Bogin et al.,

2024) for setting up and executing tasks from research repositories by 4.50%, on ResearchCodeBench (Hua et al., 2025) for translating machine learning research contributions into code by 3.51%, and ScienceAgentBench (Chen et al., 2025) for data-driven scientific research by 4.89%, over human state-of-the-art.

- • We provide qualitative analysis of REVERE’s adaptation dynamics, showing that gains stem from structured prompt evolution, efficient tool use, and controlled updates across configurable prompt fields. REVERE achieves up to 10× more cost-effective adaptation than alternative approaches, improving performance without retraining or heavy infrastructure.

- 2 RELATED WORK

Research-Coding Benchmarks and Approaches: LLMs are increasingly evaluated on tasks spanning ML engineering benchmarks (Chan et al., 2025; Huang et al., 2024), end-to-end research workflows (Panigrahi et al., 2026), and various tasks across the research experimentation life cycle (Huang et al., 2025; Edwards et al., 2025; Starace et al., 2025; Kon et al., 2025; Zhao et al., 2025). Recent benchmarks focused specifically on research-code reproducibility (Bogin et al., 2024; Tian et al., 2024; Xiang et al., 2025; Majumder et al., 2024b; Siegel et al., 2024), reveal persistent performance gaps despite advances in multi-agent systems and search-based approaches (Starace et al., 2025; Seo et al., 2025; Schmidgall et al., 2025; Lin et al., 2025; Jiang et al., 2025; Zhou et al., 2025; Si et al., 2026). These findings highlight the need for self-reflective systems over manually engineered workflows. In this work, we focus on SUPER (Bogin et al., 2024), ResearchCodeBench (Hua et al., 2025), and ScienceAgentBench (Chen et al., 2025) because they together cover complementary research-coding settings: long-horizon repository execution, single-shot research code reconstruction, and interactive scientific programming, while offering diverse task types, domains and scalable evaluation without requiring specialized large-scale compute resources.

Prompt Optimization and Self Evolution Techniques: Classical prompt optimization treats prompts as tunable parameters using RL, gradient-free, or heuristic search methods (Khattab

- et al., 2023), while newer approaches such as GEPA (Agrawal et al., 2025), MIPRO (Opsahl-Ong
- et al., 2024) use reflective models and evolutionary strategies to refine prompts for LM programs. Runtime-adaptive agents further modify their scaffolds and tooling on the fly (Zhang et al., 2025a; Xia et al., 2025; Hu et al., 2025). In addition, some approaches explore strict task-level adaptation, for each task using feedback (Hu et al., 2024; Zhang et al., 2025b) though such adaptations often fail to transfer improvements across tasks. Test-time context adaptation methods such as Dynamic Cheatsheet (Suzgun et al., 2025) and ACE (Zhang et al., 2025c) maintain persistent, evolving playbooks via generation and reflection. However, these methods are typically evaluated in densely supervised and shorter-horizon settings. On the other hand, research coding workflows are longhorizon, weakly supervised, and require context updates tightly grounded in repository structure, environments, and execution traces rather than only high-level natural language feedback, hence the need to devise a new prompt optimization strategy for research coding tasks.

- 3 REFLECTIVE EVOLVING RESEARCH ENGINEER(REVERE)

- 3.1 SETUP

We formalize the adaptation problem over three editable context fields that govern agent behavior: F = {Fs,Fx,Fc}, where Fs is the system prompt (global behavior and rules), Fx is the task prompt (task-specific instructions instantiated at runtime), and Fc is the cheatsheet (a persistent memory, initialized empty, that accumulates reusable strategies and tips). Together, these fields parameterize agent behavior without modifying model weights.

Given a dataset of tasks T = {(x,o)}, where x is a task description and o contains target metrics and optional gold outputs, the agent Φ produces an output oˆ = Φ(x;F) that is evaluated by a metric function µ to yield a scalar score s = µ(ˆo,o). Adaptation seeks optimal fields:

F∗ = arg max

F

E(x,o)∼T µ Φ(x;F), o . (1)

- 3.2 METHOD OVERVIEW

REVERE improves a coding agent through an iterative adaptation loop (Figure 1), progressively editing the three fields of F based on execution feedback. The agent runs on tasks in batches which provides the Reflector with diverse execution signals, avoiding the diminished feedback that arises when reflecting on all tasks at once without intermediate updates, and amortizing the cost of reflection. A key component of this loop is the information provided to the Reflector for decisionmaking: after each batch, it receives a local evaluation signal S = {(x,s,oˆ)}, referred to as the Evaluation Step Context, which summarizes batch outcomes and is augmented with ground truth when available, along with a global training context constructed from upcoming task descriptions and prior reflection summaries (Section 3.3). Together these complement each other to guide the

Reflector in diagnosing errors and making surgical Python-based edits to the three fields. This process repeats across multiple batches, enabling the system to accumulate knowledge over time without rewriting prompts from scratch.

Algorithm 2 Reflector Agent Module

Algorithm 1 REVERE: Adaptation Process

- 1: Input:
- 2: Fields F = {Fs,Fx,Fc}
- 3: LLM L
- 4: History buffer H, Evaluation step context S, Auxiliary task context λ
- 5: Action Space:
- 6: f′ ← EDIT(f,p): edits f ∈ {Fs,Fx,Fc} using program p
- 7: FINISH(h): terminates the loop by with summary h
- 8: F′ ← F
- 9: A ← [S,H,λ] ▷ reflector memory
- 10: repeat
- 11: (a,c) ← L(F′,A) ▷ action a ∈ {EDIT,FINISH} and c is input to a
- 12: if a = EDIT then
- 13: (f,p) ← c
- 14: f′ ← Execute(p,f)
- 15: F′ ← f′ ▷ Update selected
- 16: A ← A + (a,c)
- 17: end if
- 18: until a = FINISH
- 19: h ← c
- 20: Output: (F′,h)

- 1: Input:
- 2: Agentic system Φ, Editable context fields F
- 3: Training data T = {(x,o)}, x: task description, o: target metrics and gold trajectory, Metric function µ
- 4: Auxiliary context λ
- 5: for each epoch do
- 6: Shuffle T and form batches {B}
- 7: H ← ∅ ▷ Initialize Reflection history
- 8: for each batch B do
- 9: S ← ∅ ▷ Initialize Eval Step Context
- 10: for all (x,o) ∈ B do
- 11: oˆ ← Φ(x,F)
- 12: s ← µ(ˆo,o)
- 13: if use gold then
- 14: S ← S ∪ {(x,s,o,oˆ)}
- 15: else
- 16: S ← S ∪ {(x,s,oˆ)}
- 17: end if
- 18: end for
- 19: λ ⊆ {x | (x,o) ∈ T − B}
- 20: (F′,h) ← REFLECTOR(F,S,H,λ)
- 21: H ← H ∪ {h} ; F ← F′
- 22: end for
- 23: end for
- 24: Output: Adapted Fields F

The key mechanism enabling precise adaptation is a code-based field update, illustrated in Figure 2. Instead of regenerating the full prompt, the Reflector generates a short Python program that modifies only the relevant part of a field. Edits can range from simple string replacements to more complex restructuring, and run in an isolated environment for safety, and are described in detail in Section 3.4. The overall adaptation loop is formalized in Algorithm 1, and the Reflector module in Algorithm 2.

- 3.3 GLOBAL TRAINING CONTEXT

REVERE maintains a Global Training Context that aggregates signals across training iterations, enabling adaptation beyond local feedback via three complementary signals:

- 1. Cumulative CheatSheet (Fc): It is a continually updated, lightweight collection of concise, domain-specific strategies recorded in natural language by the Reflector. Initialized empty, it grows over time by accumulating reusable insights as short heuristics and actionable reminders rather than full trajectories or detailed rationales, and is directly used by the agent during task execution.
- 2. Reflection History (H): A record of prior reflection summaries, where each entry h ∈ H captures the rationale and outcome of an adaptation step. Unlike CheatSheet, which supports task execution, H supports the Reflector by enabling reasoning over past updates, helping prevent contradictory edits caused by short-term or noisy feedback or by unawareness of the intent behind previous updates. This promotes stable, incremental adaptation across batches.
- 3. Auxiliary Context (λ): This consists of a subset of task descriptions and inputs, drawn preferentially from unseen training tasks. When no such tasks remain, λ is sampled from randomly shuffled previously trained task descriptions. By exposing the Reflector to tasks beyond the current batch, this context encourages updates that remain effective across potential future task variations, improving generalization.

- Figure 2: The Code-Based Field Update Mechanism: Left to right, the Reflector selects a field, generates Python logic, and executes it securely to update it, enabling localized edits while preserving the overall prompt structure.

Together, these components provide a complementary learning framework: Auxiliary Context helps avoid local optima, the CheatSheet offers reusable guidance, and Reflection History maintains longterm coherence. Equipped with these signals, the Reflector can make informed field updates.

- 3.4 REFLECTION AND UPDATE MECHANISM

The Reflector is a single agent responsible for both diagnosing failures and editing the fields (Prompt in Appendix D.2). Keeping these roles unified — rather than splitting them across a multi-agent pipeline (Zhang et al., 2025c; Hu et al., 2025) — preserves a coherent view of the evolving system state and avoids hand-off boundaries that can cause misinterpreted intent and incoherent updates.

The central challenge is performing targeted edits without semantic drift. Full prompt regeneration tends to silently alter unrelated instructions and overwrite stable, validated content. To address this, we introduce a lightweight code-based update tool inspired by the CodeAct framework (Wang et al., 2024). As illustrated in Figure 2, the Reflector selects a field f ∈ F and generates a short Python program p that modifies it. Given an original task prompt (Figure 2 left), the Reflector generates Python code (Figure 2 center) that directly operates on the prompt text, for instance replacing an imprecise instruction with a clearer one, swapping a model reference from CNN to ResNet, or appending new behavioral instructions. Each operation targets only the relevant substring, leaving the rest of the prompt intact. The safety filter intercepts any out-of-scope operations such as file I/O before execution ( these restrictions are configurable, allowing users to relax or tighten the safety constraints if needed), and filters them. The approved program is executed in a secure, isolated environment (Figure 2 right) to produce the updated field (F′). This interface provides three key advantages:

(i) Targeted, low-overhead updates: Edits are applied only to the relevant portions of F via codebased transformations, allowing the Reflector to add, replace, or remove specific segments without regenerating entire prompts. This directly limits semantic drift and prevents overwriting content that is already working. (ii) Expressive, unconstrained modifications: Unlike template-based or rule-driven update schemes (Opsahl-Ong et al., 2024; Zhang et al., 2025c), code-based edits support arbitrary transformation logic over textual fields. By leveraging the Reflector’s code-generation capability, the system enables precise updates without requiring complex tool schemas or restrictive editing APIs. (iii) Safe, predictable execution: The two-layer safety design, including the static filter and isolated runtime, ensures that field updates remain contained and auditable. Programs exceeding string-only operations are rejected before execution, with the failure fed back to the Reflector to retry within the same iteration. While this may cause occasional tool failures, we treat it as

a necessary trade-off for execution safety. The filter is configurable, allowing practitioners to relax constraints for more expressive edits if needed.

- 4 EXPERIMENT SETUP

We evaluate REVERE on three challenging research-coding benchmarks (Section 4.1), spanning long-horizon, single-shot, and interactive settings. For each benchmark, we define offline and online adaptation regimes (Section 4.2) and compare against strong baseline methods (Section 4.3). A summary of benchmark datasets, including task counts and approximate per-task inference cost, is provided in Appendix A.1.

- 4.1 BENCHMARKS

SUPER (Bogin et al., 2024) consists of 45 research-coding tasks that require agents to interactively set up, configure, and execute experiments from real research repositories, and is our primary target benchmark. This long-horizon setting is executed by a coding agent in a containerized environment. Tasks reflect realistic research workflows, including repository initialization, dependency installation, resolving version conflicts, configuring experimental settings, and handling runtime issues. Agent performance is evaluated using the benchmark’s standard metrics: (i) Output Match requires reproduced results (e.g., accuracy, F1 score, or error rate) to match expert-reported outputs, (ii) Landmarks measure the presence of expected indicators of correct progress in execution logs, with higher scores assigned when more expected signals are observed, and (iii) Overall is the average of Output Match and Landmarks, serving as the primary summary metric.

ResearchCodeBench (Hua et al., 2025) evaluates an LLM’s ability to re-implement core methodologies from research papers in a single-shot setting. For each task, the agent is provided with the paper and partially masked code files and must reconstruct the missing implementation in a single forward pass. The benchmark comprises 212 tasks from 20 top-tier venues (e.g., ICLR, NeurIPS). Performance is measured by Accuracy: each task is scored as pass (1) or fail (0) based on whether the reconstructed code passes hidden unit tests without errors, and Accuracy is the mean score across all tasks.

ScienceAgentBench (Chen et al., 2025) evaluates language agents on data-driven scientific discovery tasks in an interactive code-generation setting. Each task requires producing a self-contained Python program implementing a core component of a scientific workflow, with a strong emphasis on machine learning-based methodologies. Unlike single-shot settings, agents can iteratively execute generated code, observe runtime feedback, debug errors, and revise implementations until reaching a satisfactory solution. The benchmark contains 102 tasks derived from 44 peer-reviewed publications across four scientific disciplines. Evaluation uses two metrics: (i) Success Rate (SR), measuring whether a program satisfies task-specific execution and output criteria, and (ii) CodeBERTScore (CBS), which measures semantic similarity between generated and reference code using contextual embeddings.

- 4.2 BENCHMARK EXTENSION FOR SELF-ADAPTATION

For each benchmark, we consider two adaptation settings, namely offline and online. In the offline setting, the agent adapts using fixed training and validation tasks, while evaluation is performed on a held-out test set that remains unseen during adaptation. We adopt a three-way train/validation/test split of 9/9/27, 34/34/144, and 20/20/62 for SUPER, ResearchCodeBench, and ScienceAgentBench, respectively. In a without-ground-truth variant of offline adaptation, supervision is removed from training and validation tasks as well, requiring adaptation solely from the agent’s own explored solutions and failures.

In the online setting, tasks arrive sequentially from the full dataset without repetition and without ground-truth supervision. This setting is more realistic and challenging, requiring the agent to continually update based on its own execution outcomes and traces. The auxiliary context λ is sampled from previously encountered tasks only, as no future tasks are accessible. This setup reflects real-world research workflows where tasks appear over time and supervision is limited or unavail-

SUPER Bench (%) RCB (%) ScienceAgentBench (%)

Method GT

Landmarks Output Match Overall Accuracy SuccessRate CodeBERTScore

long-horizon single shot interactive - ReAct

Baseline – 24.7 9.3 17 27.8 20.9 67.5 SOTA† – 35.8 14.8 25.3 31.9 23.5 88.2

Offline Adaptation

GEPA ✗ 0.0−35.80 18.6+3.80 9.3−16.00 25.69−6.21 24.5+1.0 74.8−13.4 GEPA ✓ 1.93−33.87 2.6−12.20 2.27−23.03 10.44−21.46 19.35−4.15 65.6−22.6 ACE ✗ 19.53−16.27 18.51+3.71 19.02−6.28 21.99−9.91 8.6−14.90 53.48−34.72 ACE ✓ 27.22−8.58 18.64+3.84 22.93−2.37 7.87−24.03 4.3−19.20 24.28−63.92 REVERE ✗ 35.67−0.13 19.76+4.96 27.71+2.41 28.75−3.15 23.23−0.27 80.25−7.95 REVERE ✓ 35.84+0.04 23.76+8.96 29.8+4.50 33.2+1.3 28.39+4.89 82.84−5.36

- Table 1: Offline adaptation results on SUPER, RCB (ResearchCodeBench), ScienceAgentBench (SAB). All reported values are means over 5 runs ;✓ and ✗ denote settings with and without groundtruth (GT) hints, respectively, while (-) indicates inference-only results. † marks the best-performing static prompt. +x and −x represent improvements and declines relative to the Static SOTA. Standard deviations for offline results are reported in Appendix A.3.

able. To enable direct comparison with the offline protocol, we additionally report test-subset results extracted from the full online evaluation.

- 4.3 BASELINE METHODS

Baseline and SOTA Prompts: Our baseline system uses minimal, non-optimized prompts containing only the core task description, referred to as ‘baseline’ which serves as a reference for isolating the effects of REVERE’s adaptation mechanisms. For the SUPER (long-horizon) and ScienceAgentBench (Interactive) , we implement a ReAct agent (Yao et al., 2023) equipped with code tools (see Appendix A.4)and for ResearchCodeBench we used direct llm call to generate the program. Additionally, we report the current state-of-the-art performance for each benchmark using author-provided instructions, denoted as ‘Static SOTA’ (see Appendix D.1). All results are computed using GPT-4.11. We select GPT-4.1 primarily due to the extreme context requirements of researchoriented coding environments, which often involve large code repositories, academic papers, and long-horizon reasoning traces. These settings typically produce inputs of 300k-500k tokens for the reflector and 40k-120k tokens for agents. The 1M-token context window of GPT-4.1 allows us to use a single model for both reflection and adaptation, reducing cross-model knowledge transfer.

GEPA (Genetic-Pareto) (Agrawal et al., 2025) is a sample-efficient prompt optimizer that evolves prompts using natural-language reflection and Pareto-based genetic search. It analyzes execution traces (reasoning steps, tool actions, outputs), diagnoses failures, and generates candidate prompt updates. A Pareto frontier maintains a diverse set of high-performing prompts, improving robustness and avoiding local minima. However, this design primarily supports offline optimization, as the Pareto set is constructed over a fixed evaluation pool, limiting its suitability for online adaptation. We use GEPA’s official research repository 2 for all implementations. Refer Appendix A.2 for detailed configuration.

ACE (Agentic Context Engineering) (Zhang et al., 2025c) is a playbook-based learning approach inspired by (Suzgun et al., 2025) that maintains bullet-point strategies across predefined categories and updates them via a multi-agent reflector-curator loop. We implement ACE adapted to our baseline agent system, using a step size of one as recommended in their paper and provided in the official implementation3, ensuring a fair comparison.

- 1GPT-4.1 accessed via the Azure OpenAI API (Appendix A.5).
- 2GEPA : https://github.com/gepa-ai/gepa-artifact 3ACE : https://github.com/ace-agent/ace

SUPER Bench (%) RCB (%) ScienceAgentBench (%)

Method

Landmarks Output Match Overall Accuracy SuccessRate CodeBERTScore

long-horizon single shot interactive - ReAct

Baseline 24.7 9.3 17 27.8 20.96 67.50 Baseline - all tasks 19.63 10.4 15.01 43.80 20.58 66.86

Online Adaptation

ACE - test 26.8+2.10 11.1+1.80 18.95+1.95 15.50−12.30 11.3−9.66 66.5−1.00 ACE - all tasks 24.8+5.17 14.2+3.80 19.5+4.49 19.81−23.99 11.8−8.78 65.5−1.36 REVERE - test 33.5+8.8 17.6+8.3 25.5 +8.10 35.40 +7.6 25.80 +4.84 69.61 +2.11 REVERE - all tasks 30.1+10.4 18.7+8.3 24.4+9.39 45.00 +1.2 25.49 +4.91 71.68 +4.82

- Table 2: Online adaptation results obtained without using ground-truth labels. RCB is ResearchCodeBench, all tasks performance is on the full benchmark; test: test-set-only results , +x and −x with respect to corresponding baseline.

- 5 RESULTS AND ANALYSIS

Under offline adaption, REVERE improves agent performance over the baseline across all metrics for all the benchmarks, irrespective of availability of the ground truth hints. Results are presented in Table 1. In the hint-free SUPER setting, REVERE improves ‘Output Match’ while maintaining Landmarks at a level comparable to the Static SOTA results, producing a higher overall score despite limited supervision. In contrast, GEPA improves Output Match but its Landmarks collapse to zero, reflecting its tendency to overfit recent execution traces and produce brittle prompt rewrites that harm intermediate reasoning. ACE shows relatively stable but inconsistent gains, improving Output Match’ only marginally while remaining comparable to the baseline but underperforming REVERE in overall performance. REVERE avoids these failure modes because its updates are constrained by the Global Training Context, which guides what to retain, emphasize, or omit across prompts. Given that SUPER is our primary target benchmark, we provide an additional case study analyzing recurring failure modes and prompt evolution across adaptation iterations in Appendix B.

Better feedback signal can lead to better generalization, but hinges on the method’s strength to leverage it. ResearchCodeBench and ScienceAgentBench are domain-knowledge-heavy with limited structural overlap across examples, constraining how much transferable signal adaptation methods can exploit. Despite this, REVERE outperforms both the baseline and competing methods in the hint-free setting. Counterintuitively, GEPA and ACE both degrade further when provided with gold hints, sharing a common failure: both rely on local feedback signals that cannot generalize across the rich, domain-specific nature of research coding tasks. Importantly, GEPA’s discard-andrewrite mechanism discards valuable supervision from gold trajectories, while ACE indiscriminately captures all step-level signals, overfitting to task-specific traces that do not transfer across benchmarks. REVERE is designed to address this directly. Its Global Training Context maintains a global view of the adaptation trajectory, preventing overfitting to local batch-level noise, allowing REVERE to be the only method that consistently benefits from stronger feedback across both benchmarks, improving over the baseline and Static SOTA on all metrics. Although the drop in CodeBERTScore (−5.36) on ScienceAgentBench is offset by a +4.89% gain in SuccessRate, the latter is the more reliable indicator of task performance as it measures functional correctness rather than lexical similarity. The more modest gains compared to SUPER are expected given these knowledge and training constraints, yet REVERE still consistently outperforms the baseline, confirming that adaptation provides value even under strong domain constraints.

REVERE retains comparable performance in online settings despite minimal feedback. Table 2 shows the same trends hold in the stricter online setting, where no ground-truth labels are available and each task is encountered only once. REVERE improves performance across all benchmarks and metrics, while ACE often performs below the baseline, suggesting that maintaining a cheatsheet alone is insufficient for stable adaptation. This contrast indicates that leveraging global context and jointly adapting multiple fields better supports cross-task generalization. Although absolute performance is lower than in offline training, this gap is structurally expected: once a task is processed its performance is fixed, cannot be revised, future tasks remain unseen at adaptation

SUPER Bench (%) RCB(%) ScienceAgentBench (%)

Method

Landmarks Output Match Overall Acc. SuccessRate CodeBERTScore

long-horizon single shot interactive - ReAct

REVERE 38.1 31.4 34.7 35.4 30.6 83.3 REVERE w / o Cheatsheet 30.7 −7.42 16.7 −14.73 23.7 −11.07 30.6 −4.86 20.9 −9.7 78.3 −5.0 REVERE w / o Auxiliary context 24.8 −13.29 24.1 −7.33 24.4 −10.30 31.9 −3.47 25.8 −4.8 79.8 −3.5 REVERE w / o Reflection history 25.9 −12.17 9.3 −22.14 17.6 −17.15 32.6 −2.78 22.5 −8.1 75.9−7.4 REVERE w / o Global Training Context 21.7 −16.37 10.2 −21.22 16.0 −18.79 27.1 −8.33 19.3 −11.3 77.3 −6.0

Table 3: Ablation on Offline Adaptation. Results correspond to a representative run

time, and no ground-truth supervision is available. Despite this, REVERE maintains consistent relative gains, indicating that its adaptation mechanism leverages recurring execution patterns observed during interaction rather than depending on ground-truth supervision.

Do we need all components of REVERE? The ablation results (Table 3) confirm that REVERE’s effectiveness arises from the interaction of its core components. Removing the cheatsheet memory causes broad degradation across metrics, showing that cumulative retention of discovered fixes and heuristics is central to adaptation. While excluding auxiliary context or reflection history individually degrades performance, highlighting the distinct role of each component, though the largest drops occur when the entire Global Training Context is removed altogether, confirming that their combined effect is greater than any single part.

| | |
|---|---|
| | |

- Figure 3: Analysis on adapted performance on the SUPER benchmark. (A) Illustrates overall performance trajectory relative to the baseline and Static SOTA across iterations. For GEPA, the blue marker denotes the prompt selected by the Pareto filter. (B) Plots per-task performance against total tool calls for individual tasks, comparing REVERE-GT (orange) and baseline (blue) agents across four quadrants by task count. (C) Compares tool usage counts across variants (Base, GEPA, ACE, REVERE without (No-GT) and REVERE with ground truth (GT).), broken down by tool type (Read, Edit, Write, Run).Note: GEPA iterations represent total candidates, normalized to fit the axis scale.

REVERE exploits better performance from test-time compute and tool usage Figure 3(A) shows that REVERE improves steadily on the SUPER benchmark across iterations. Despite a marginal dip in early iterations, it recovers and surpasses both the Static baseline and ACE by the final iteration, consistent with results in Table 1. ACE follows a similar trend but remains below throughout, suggesting that cheatsheet-only adaptation without global context provides partial gains but is insufficient for stable improvement in long-horizon settings. GEPA exhibits high variance across iterations, with instructions rejected by the Pareto filter at earlier steps later outperforming those selected, highlighting the fragility of heuristic-based prompt selection without accumulated adaptation context. Figure 3(B) plots per-task performance against total tool calls across four quadrants. The bottom-left quadrant captures baseline tasks that failed without meaningful exploration, providing weak adaptation signal. The top-left represents the ideal regime: tasks solved efficiently with minimal tool usage, reflecting well-adapted behavior. The bottom-right captures tasks where the agent persisted but ultimately failed, yet these are the richest source of adaptation signal as extended traces expose what went wrong. The top-right is where REVERE-GT holds a clear advantage, successfully resolving harder tasks requiring greater persistence. Notably, the concentration of REVERE tasks in the bottom-right reflects a deliberate property of effective adaptation: to extract valuable signals, the adapted agent must push the underlying agent to its limits, surfacing failures

that inform future improvement. Over time, tasks in this quadrant are a natural target for migration toward the efficient top-left as the agent learns to solve them with less effort. Figure 3(C) reveals how adaptation strategy shapes tool usage. GEPA under-explores relative to the baseline, consistent with its context-discarding design. Both ACE and REVERE show increased tool usage over the baseline, consistent with their performance gains. However REVERE converts tool calls into stronger performance gains more efficiently than ACE, confirming that our method adapts the underlying agent efficiently. (A direct comparison between REVERE and ACE on tool usage efficiency is not entirely fair, as the test set is relatively smaller in the research code setting.)

- Figure 4: Efficiency analysis of REVERE Framework. (A) Illustrates prompt component growth in token length across iterations on the SUPER benchmark, comparing REVERE components , GEPA query, and ACE cheatsheet. (B) Illustrates final adapted prompt lengths (in tokens) across benchmarks, used in offline adaptation results with ground truth. (C) Compares computational cost (USD) on the SUPER task in an offline setting across REVERE, ACE, and GEPA, decomposed by architectural components including training, reflection, and GTC, with dashed lines indicating method-specific adaptation cost. (D) Distribution of outcomes across code-edit tool calls in the ground-truth setting across all benchmarks (SUPER, RCB, SAB).

REVERE maintains efficient adaptation with controlled prompt growth. Figure 4 (A) shows that the three methods diverge sharply in how they manage prompt length across iterations, and the growth patterns reflect their respective architectural designs. REVERE’s components (Fs, Fc, Fx) exhibit controlled, distributed growth, allowing the system to allocate capacity where it is most needed. ACE shows steep linear growth driven entirely by its cheatsheet, conflating task-specific and generalizable signals into bloated prompts; growing context length remains a fundamental challenge even in REVERE, and we discuss potential mitigation in Appendix C.1. GEPA maintains consistently low token count by regenerating instructions from scratch each iteration, but discards previously accumulated knowledge at every step, making it poorly suited for domain-rich, individually complex tasks. Figure 4 (B) reports final adapted prompt lengths for the results in Table 1, and directly supports our earlier observation on domain-heavy benchmarks. ACE’s cheatsheet grows dramatically on RCB, reflecting the benchmark’s heavy reliance on specialized domain knowledge that is difficult to compress into reusable prompts, accumulating task-specific traces instead of distilling generalizable signals. REVERE maintains compact prompts across all benchmarks, though as noted earlier, controlled prompt growth alone does not overcome the domain familiarity constraints that bound performance on research-oriented coding tasks. Figure 4 (C) compares adaptation costs across methods, comprising training cost, which is inevitable and shared across all methods (shown hatched), and adaptation cost, the method-specific overhead and meaningful basis for comparison.4 REVERE’s total cost is already lower than both ACE and GEPA, and isolating adaptation overhead makes this advantage even more pronounced, with REVERE’s adaptation cost nearly 10x lower than that of ACE and GEPA. Reflection and curation costs remain low across methods even as context length increases.5 In REVERE this efficiency is achieved through targeted edits, while the Global Training Context remains significantly smaller than the heavier architectural components required by ACE and GEPA.6 Figure 4 (D) reports outcomes of the code-based field update tool across 610

4A detailed decomposition of each method’s architectural cost components is provided in Appendix C.2. is $0.16 USD compared to ACE’s $0.47 USD.

- 5Curation costs are too small to appear visibly in the plot; for reference, REVERE’s curation cost on SUPER

6GTC is analogous to the architectural components of other methods such as ACE and GEPA; see Appendix C.2 for rationale.

edit attempts, with 90.2% succeeding. Of the remaining cases, 8.5% are blocked by the static safety filter and 1.3% fail at runtime, both of which are recovered as the failure is fed back to the Reflector for retry within the same iteration. The blocked cases are not a limitation: the safety filter ensures controlled, auditable field updates, and users can always relax the constraints if needed, making this a configurable safeguard for safe LLM-to-LLM integration.

- 6 CONCLUSION

We presented REVERE a lightweight, unsupervised adaptation framework for LLM agents tackling multi-step, long-horizon and heterogeneous research coding tasks. REVERE augments standard agent setups with a Global Training Context that aggregates reflection history, auxiliary context, and a cumulative cheatsheet, while supporting update mechanism that issues precise, code-level edits to prompts. This design supports both offline and more realistic online adaptation, mitigating context myopia from purely local updates while being interpretable and easy to integrate into existing agents. REVERE demonstrates consistent gains over the baseline and strong prompt adaptation frameworks on three challenging benchmarks: SUPER (long-horizon), ResearchCodeBench (single-shot), and ScienceAgentBench (interactive), with improvements up to 4.89% over Static SOTA. Notably, REVEREdelivers these gains while being up to 10× more cost-effective than alternative solutions. More broadly, REVERE validates a practical route to scalable continual selfadaptation in LLM agents. We observe more modest gains on domain-heavy and heterogeneous benchmarks, reflecting the challenge of capturing highly task-specific knowledge through prompt adaptation alone, especially in research coding settings. This suggests that prompt-based updates may not be sufficient in such scenarios, and exploring more task-specific forms of adaptation is an important direction. Additionally, REVEREintroduces some overhead due to its growing Global Training Context, which can become harder to manage for longer tasks and may accumulate stale or less useful information over time. Improving how this context is maintained, pruned, and recovered is another key area for future work.

REFERENCES

Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. Gepa: Reflective prompt evolution can outperform reinforcement learning, 2025. URL https://arxiv. org/abs/2507.19457.

UK AI Security Institute. Inspect AI: Framework for Large Language Model Evaluations, 2024. URL https://github.com/UKGovernmentBEIS/inspect_ai.

Ben Bogin, Kejuan Yang, Shashank Gupta, Kyle Richardson, Erin Bransom, Peter Clark, Ashish Sabharwal, and Tushar Khot. Super: Evaluating agents on setting up and executing tasks from research repositories, 2024. URL https://arxiv.org/abs/2409.07440.

Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, and Aleksander Ma˛dry. Mlebench: Evaluating machine learning agents on machine learning engineering, 2025. URL https: //arxiv.org/abs/2410.07095.

Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu, Vishal Dey, Mingyi Xue, Frazier N. Baker, Benjamin Burns, Daniel AduAmpratwum, Xuhui Huang, Xia Ning, Song Gao, Yu Su, and Huan Sun. Scienceagentbench: Toward rigorous assessment of language agents for data-driven scientific discovery, 2025. URL https://arxiv.org/abs/2410.05080.

Nicholas Edwards, Yukyung Lee, Yujun Audrey Mao, Yulu Qin, Sebastian Schuster, and Najoung Kim. Rexbench: Can coding agents autonomously implement ai research extensions?, 2025. URL https://arxiv.org/abs/2506.22598.

Aniketh Garikaparthi, Manasi Patwardhan, and Arman Cohan. Researchgym: Evaluating language model agents on real-world ai research, 2026. URL https://arxiv.org/abs/2602.15112.

Paul Gauthier. Aider polyglot benchmark. https://aider.chat/polyglot, 2024. Accessed: 2025-11-13.

Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems, 2025. URL https: //arxiv.org/abs/2408.08435.

Yue Hu, Yuzhu Cai, Yaxin Du, Xinyu Zhu, Xiangrui Liu, Zijie Yu, Yuchen Hou, Shuo Tang, and Siheng Chen. Self-evolving multi-agent collaboration networks for software development, 2024. URL https://arxiv.org/abs/2410.16946.

Tianyu Hua, Harper Hua, Violet Xiang, Benjamin Klieger, Sang T. Truong, Weixin Liang, Fan-Yun Sun, and Nick Haber. Researchcodebench: Benchmarking llms on implementing novel machine learning research code, 2025. URL https://arxiv.org/abs/2506.02314.

Jin Huang, Silviu Cucerzan, Sujay Kumar Jauhar, and Ryen W. White. Idea2plan: Exploring aipowered research planning, 2025. URL https://arxiv.org/abs/2510.24891.

Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. Mlagentbench: Evaluating language agents on machine learning experimentation, 2024. URL https://arxiv.org/abs/2310.03302.

Zhengyao Jiang, Dominik Schmidt, Dhruv Srikanth, Dixing Xu, Ian Kaplan, Deniss Jacenko, and Yuxiang Wu. Aide: Ai-driven exploration in the space of code, 2025. URL https://arxiv. org/abs/2502.13138.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. Dspy: Compiling declarative language model calls into selfimproving pipelines, 2023. URL https://arxiv.org/abs/2310.03714.

Patrick Tser Jern Kon, Jiachen Liu, Xinyi Zhu, Qiuyi Ding, Jingjia Peng, Jiarong Xing, Yibo Huang, Yiming Qiu, Jayanth Srinivasa, Myungjin Lee, Mosharaf Chowdhury, Matei Zaharia, and Ang Chen. Exp-bench: Can ai conduct ai research experiments?, 2025. URL https://arxiv.org/ abs/2505.24785.

Zijie Lin, Yiqing Shen, Qilin Cai, He Sun, Jinrui Zhou, and Mingjun Xiao. Autop2c: An llm-based agent framework for code repository generation from multimodal content in academic papers,

2025. URL https://arxiv.org/abs/2504.20115.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback, 2023. URL https://arxiv.org/abs/2303.17651.

Bodhisattwa Prasad Majumder, Bhavana Dalvi Mishra, Peter Jansen, Oyvind Tafjord, Niket Tandon, Li Zhang, Chris Callison-Burch, and Peter Clark. CLIN: A continually learning language agent for rapid task adaptation and generalization. In First Conference on Language Modeling, 2024a. URL https://openreview.net/forum?id=xS6zx1aBI9.

Bodhisattwa Prasad Majumder, Harshit Surana, Dhruv Agarwal, Bhavana Dalvi Mishra, Abhijeetsingh Meena, Aryan Prakhar, Tirth Vora, Tushar Khot, Ashish Sabharwal, and Peter Clark. Discoverybench: Towards data-driven discovery with large language models, 2024b. URL https://arxiv.org/abs/2407.01725.

Krista Opsahl-Ong, Michael J Ryan, Josh Purtell, David Broman, Christopher Potts, Matei Zaharia, and Omar Khattab. Optimizing instructions and demonstrations for multi-stage language model programs, 2024. URL https://arxiv.org/abs/2406.11695.

Siba Smarak Panigrahi, Jovana Videnovi´c, and Maria Brbi´c. Heurekabench: A benchmarking framework for ai co-scientist, 2026. URL https://arxiv.org/abs/2601.01678.

Xin Peng and Chong Wang. Code digital twin: Empowering llms with tacit knowledge for complex software development, 2025. URL https://arxiv.org/abs/2503.07967.

Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Michael Moor, Zicheng Liu, and Emad Barsoum. Agent laboratory: Using llm agents as research assistants, 2025. URL https://arxiv.org/abs/2501.04227.

Tobias Schnabel and Jennifer Neville. Symbolic prompt program search: A structure-aware approach to efficient compile-time prompt optimization, 2024. URL https://arxiv.org/abs/ 2404.02319.

Minju Seo, Jinheon Baek, Seongyun Lee, and Sung Ju Hwang. Paper2code: Automating code generation from scientific papers in machine learning, 2025. URL https://arxiv.org/abs/ 2504.17192.

Wenhang Shi, Yiren Chen, Shuqing Bian, Xinyi Zhang, Kai Tang, Pengfei Hu, Zhe Zhao, Wei Lu, and Xiaoyong Du. No loss, no gain: Gated refinement and adaptive compression for prompt optimization, 2025. URL https://arxiv.org/abs/2509.23387.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. URL https://arxiv.org/abs/2303.11366.

Chenglei Si, Zitong Yang, Yejin Choi, Emmanuel Candès, Diyi Yang, and Tatsunori Hashimoto. Towards execution-grounded automated ai research, 2026. URL https://arxiv.org/abs/2601. 14525.

Zachary S. Siegel, Sayash Kapoor, Nitya Nagdir, Benedikt Stroebl, and Arvind Narayanan. Corebench: Fostering the credibility of published research through a computational reproducibility agent benchmark, 2024. URL https://arxiv.org/abs/2409.11363.

Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, and Tejal Patwardhan. Paperbench: Evaluating ai’s ability to replicate ai research, 2025. URL https: //arxiv.org/abs/2504.01848.

Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. Dynamic cheatsheet: Test-time learning with adaptive memory, 2025. URL https://arxiv.org/abs/2504. 07952.

Minyang Tian, Luyu Gao, Shizhuo Dylan Zhang, Xinan Chen, Cunwei Fan, Xuefei Guo, Roland Haas, Pan Ji, Kittithat Krongchon, Yao Li, Shengyan Liu, Di Luo, Yutao Ma, Hao Tong, Kha Trinh, Chenyu Tian, Zihan Wang, Bohao Wu, Yanyu Xiong, Shengzhu Yin, Minhui Zhu, Kilian Lieret, Yanxin Lu, Genglin Liu, Yufeng Du, Tianhua Tao, Ofir Press, Jamie Callan, Eliu Huerta, and Hao Peng. Scicode: A research coding benchmark curated by scientists, 2024. URL https: //arxiv.org/abs/2407.13168.

Dhruv Trehan and Paras Chopra. Why llms aren’t scientists yet: Lessons from four autonomous research attempts, 2026. URL https://arxiv.org/abs/2601.03315.

Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable code actions elicit better llm agents, 2024. URL https://arxiv.org/abs/2402.01030.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. Openhands: An open platform for ai software developers as generalist agents, 2025. URL https://arxiv.org/abs/2407.16741.

Zhen Wang, Fan Bai, Zhongyan Luo, Jinyan Su, Kaiser Sun, Xinle Yu, Jieyuan Liu, Kun Zhou, Claire Cardie, Mark Dredze, Eric P. Xing, and Zhiting Hu. Fire-bench: Evaluating agents on the rediscovery of scientific insights, 2026. URL https://arxiv.org/abs/2602.02905.

Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid ShwartzZiv, Neel Jain, Khalid Saifullah, Sreemanti Dey, Shubh-Agrawal, Sandeep Singh Sandha, Siddartha Naidu, Chinmay Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger, and Micah Goldblum. Livebench: A challenging, contamination-limited llm benchmark, 2025. URL https://arxiv.org/abs/2406.19314.

Chunqiu Steven Xia, Zhe Wang, Yan Yang, Yuxiang Wei, and Lingming Zhang. Live-swe-agent: Can software engineering agents self-evolve on the fly?, 2025. URL https://arxiv.org/abs/ 2511.13646.

Yanzheng Xiang, Hanqi Yan, Shuyin Ouyang, Lin Gui, and Yulan He. Scireplicate-bench: Benchmarking llms in agent-driven algorithmic reproduction from research papers, 2025. URL https: //arxiv.org/abs/2504.00255.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering,

2024. URL https://arxiv.org/abs/2405.15793.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023. URL https://arxiv.org/ abs/2210.03629.

Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. Darwin godel machine: Openended evolution of self-improving agents, 2025a. URL https://arxiv.org/abs/2505.22954.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. Aflow: Automating agentic workflow generation, 2025b. URL https://arxiv.org/abs/ 2410.10762.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. Agentic context engineering: Evolving contexts for self-improving language models, 2025c. URL https://arxiv.org/abs/2510.04618.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners, 2024. URL https://arxiv.org/abs/2308.10144.

Yilun Zhao, Weiyuan Chen, Zhijian Xu, Manasi Patwardhan, Chengye Wang, Yixin Liu, Lovekesh Vig, and Arman Cohan. AbGen: Evaluating large language models in ablation study design and evaluation for scientific research. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12479–12491, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/ 2025.acl-long.611. URL https://aclanthology.org/2025.acl-long.611/.

Mingyang Zhou, Quanming Yao, Lun Du, Lanning Wei, and Da Zheng. Reflective paper-to-code reproduction enabled by fine-grained verification, 2025. URL https://arxiv.org/abs/2508. 16671.

Category Description Found Solution Dependencies Build/install failures or

- • Use pip install –only-binary=:all: ...
- • Pin numpy/sklearn/Cython to highest available wheel

missing libraries for training/fine-tuning.

Environment Environment/version mismatches causing runtime errors.

- • Enumerate/validate environment variables and dependency availability
- • Debug versions/signatures: inspect.signature(func)
- • Reconfigure environment: if no gpu available configure cuda() to cpu

Configuration Wrong CLI args/configs or missing scripts.

- • Verify argument names: train_file, validation_file, data_dir
- • Enumerate run_*.sh candidates and match closest

Data - Acquisition

Missing datasets or incorrect asset content.

- • Inspect README/scripts; attempt wget/gdown/curl
- • Inspect content directly using head, cat, less, file

Data - Preprocessing

Incompatible dataset schema/fields or container formats.

- • Write wrapper to rename or transform fields
- • Decompress .csv; check for embedded .jsonl/.tsv

Execution Issues Scripts produce no metrics or import errors.

- • Check output directory and config paths
- • Fix hardcoded outputs
- • Patch imports/sys.path; test package and script entrypoints

Goal Metric extraction and schema compliance for outputs.

- • Enumerate result.txt per run; extract requested metrics
- • Extract available final metrics from logs/stdout
- • Set required keys to null only after full recovery attempts

Miscellaneous Reproducibility and sourceof-truth guidance.

- • Enumerate instance_id, query, github_repo, git_commit
- • Use README/scripts as canonical hyperparameters

Semantics Output semantics: schema matching and null policies.

- • Match output schema exactly to the query
- • Return null only after exhaustive dataset recovery attempts

Table 4: Failures modes autonomously identified across research repositories by REVERE .

- A EXPERIMENT DETAILS

- A.1 DATA

- Table 5 summarizes the three benchmarks used in our evaluation, including task counts and approximate per-task inference cost. Costs reflect single-run agent execution and vary with task complexity and trajectory length. SUPER is the most expensive due to its long-horizon interactive nature; we use the Expert subset, which contains expert-designed tasks and is therefore more challenging and representative of real-world research scenarios. ResearchCodeBench is the most lightweight given its single-shot setting.

- A.2 HYPERPARAMETERS

SUPER Benchmark. For GEPA, we used the official implementation with 32 optimization iterations. Early stopping was applied when no improved prompt was found for 20 consecutive iterations. Prompt quality was estimated using a mini feedback set of three examples, and the Pareto frontier size was set to 9. or ACE, we used a step size (i.e., batch size) of 1, consistent with the original paper.For REVERE, training was performed for 5 epochs with a batch size of 3. A look-ahead window

- of 6 future examples was used to stabilize adaptation. No separate validation set was used, allowing all available training data to contribute directly to parameter updates.

Benchmark Cost Per Task Task Count Super-Expert $1–3 45 tasks Setting up, and executing tasks from research repositories. ResearchCodeBench ∼ $0.1-0.3 212 tasks Code completion by implementing methodology from the paper. ScienceAgentBench ∼ $0.2-$0.4 102 tasks Scientific coding tasks spanning multiple domains for data-driven discovery.

- Table 5: Benchmark datasets used in our evaluation, with task counts and approximate per-task inference cost. Costs are estimates based on single-run execution and vary with task complexity.

Method GT

SUPER Bench (%) RCB (%) ScienceAgentBench (%)

Landmarks Output Match Overall Accuracy SuccessRate CodeBERTScore

long-horizon single shot interactive - ReAct Offline Adaptation

GEPA ✗ 0.0±0.0 18.6±0.85 9.3±0.42 25.69±0.00 24.5* 74.8* GEPA ✓ 1.93±2.15 2.6±1.0 2.27±0.99 10.44±3.12 19.35* 65.6* ACE ✗ 19.53±10.01 18.51±3.62 19.02±3.2 21.99±3.28 8.6±7.27 53.48±8.08 ACE ✓ 27.22±3.04 18.64±3.91 22.93±1.61 7.87±1.75 4.3±1.86 24.28±1.95 REVERE ✗ 35.67±3.09 19.76±3.71 27.71±2.99 28.75±2.0 23.23±3.14 80.25±2.7 REVERE ✓ 35.84±5.52 23.76±4.59 29.8±4.26 33.2±1.8 28.39±4.36 82.84±1.19

- Table 6: Offline Adaptation Results; RCB: ResearchCodeBench; with (✓) and without (✗) groundtruth (GT) Hints, ±: standard deviation (Std Dev) across 5 runs. ∗ Std. Dev. results are omitted due to cost constraints.

ResearchCodeBench. For GEPA, a budget of 600 metric evaluations was allocated, corresponding to approximately 20 optimization iterations. Similar to SUPER for ACE. For REVERE, the setup followed the SUPER configuration, except that the look-ahead window size was reduced to 3 due to context-length constraints caused by longer code, paper, and context segments in ResearchCodeBench tasks.

ScienceAgentBench. For GEPA, we used a maximum of 15 adaptation iterations with a batch size

- of 7. Similar to SUPER for ACE.For REVERE, training was conducted for 5 epochs with a batch size of 7 and a look-ahead window size of 5.

- A.3 ROBUSTNESS ON ADAPTATION

- Table 6 presents the offline adaptation results along with standard deviations, enabling a closer examination of performance stability across methods and settings. The research coding setting (e.g., ResearchCodeBench and ScienceAgentBench) emerges as particularly sensitive to adaptation. While prior methods such as GEPA and ACE outperform several standard baselines in simpler settings, their performance degrades significantly in these benchmarks, reflecting poor generalization to domain-specific, knowledge-intensive tasks. This sensitivity arises from the limited structural overlap across tasks and the reliance on precise, context-dependent reasoning, making naive adaptation strategies brittle.

Although REVERE demonstrates improved performance relative to competing methods, it still exhibits variability and only marginal gains on certain metrics in these settings. This highlights an important limitation: current adaptation approaches, including ours, are not yet fully robust for research-oriented coding benchmarks. Overall, these results underscore the need for more principled and stable adaptation methods specifically designed for complex, domain-rich environments.

- A.4 AGENTIC AND EVALUATION IMPLEMENTATION

We implement a ReAct-style agent that interleaves reasoning and tool usage to solve complex coding and research tasks through an iterative interaction loop with the environment. The agent is built on a modular tool-calling framework using our custom-built aicodetools7 library, inspired by prior agent frameworks such as InspectAI (AI Security Institute, 2024) and LangChain.Our library abstracts system-level complexities and isolates execution overhead from the adaptation framework, allowing efficient extensibility and integration of new capabilities. It supports containerized execution and parallel task handling, ensuring scalability, reproducibility, and efficient resource utilization. The agent is equipped with core tools including read, write, edit, and run_command, enabling controlled interaction with the filesystem and execution environment. Additionally, we introduce a finish tool that allows the agent to produce structured outputs, including final answers, execution summaries, and explicit reasoning traces, thereby improving interpretability and evaluation consistency.

We further extend existing benchmarks—SUPER, Research Code Bench (RCB), and ScienceAgentBench—by integrating Dockerized execution, a streamlined pipeline, and enhanced evaluation mechanisms. These contributions ensure safe and reproducible execution while enabling efficient experimentation through parallelization and standardized logging. Our framework is designed to be highly extensible, allowing seamless incorporation of additional benchmarks, tools, and evaluation protocols, thereby supporting continued development and broader applicability of agent-based research systems.

- A.5 SYSTEM HARDWARE

All methods including REVERE and baselines uses the Azure gpt-4.1-2025-03-01-preview model and all experiments are conducted in standard_d16ads_v5 with 16 vcpus and 64 GiB memory.

- B SUPER BENCHMARK CASE STUDY

To better understand how REVERE improves performance on long-horizon research-repository tasks, we analyze the recurring failure modes it encounters on the SUPER benchmark and the strategies it autonomously learns to address them. Rather than isolated bugs, these failures cluster into systematic categories that reflect the practical reality of research-code reproduction. Table 4 summarizes these categories and the corresponding remedies that REVERE accumulates in its evolving cheatsheet and prompts.

- B.1 FAILURE MODES

A key observation in Table 4 is that the dominant failure modes are not algorithmic or modelreasoning errors, but infrastructure and workflow mismatches: dependency resolution, environment configuration, script selection, data formatting, and metric extraction. These issues arise from discrepancies between implicit assumptions held by repository authors and the explicit instructions available to the agent. Crucially, these mismatches recur across different repositories with similar patterns, indicating that research-code reproduction exhibits a shared latent structure of operational pitfalls. This explains why static prompts or local prompt rewrites struggle: each individual failure appears repository-specific, but the type of failure (e.g., missing assets, version drift, schema mismatch) is globally recurring. REVERE’s advantage stems from converting these superficially idiosyncratic errors into reusable procedural heuristics. The cheatsheet thus functions as a crossrepository “operational prior,” allowing the agent to approach new tasks with expectations about likely breakdown points rather than treating each environment as independent.

- B.2 PROMPT EVOLUTION

- Figure 5 visualizes the evolution of the task prompt Fx across online adaptation iterations, showing incremental additions, modifications, and removals of semantically grouped instruction units.

7https://pypi.org/project/aicodetools/

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

[Figure 56]

[Figure 57]

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

- Figure 5: Evolution of the prompt Fx across online adaptation iterations. Snapshots from the first, twelfth, thirtieth, and final iterations illustrate how semantically grouped instruction capsules are incrementally refined. Additions (green), modifications (orange), and removals (red) show that adaptation proceeds via structured accumulation rather than prompt rewriting.

Several patterns emerge. First, growth is structured rather than expansive: instructions are added in coherent clusters tied to concrete failure classes (e.g., environment validation, data inspection, metric recovery), rather than as diffuse prompt length increases. Second, many edits refine existing rules instead of replacing them, indicating that adaptation operates through policy sharpening rather than wholesale behavioral shifts. Third, deletions are rare and localized, suggesting that once a heuristic proves broadly useful, it remains stable across tasks. Together, these dynamics support the claim that REVERE’s improvement mechanism is cumulative and non-destructive: knowledge about workflow regularities is preserved and layered, enabling generalization without the prompt drift or knowledge loss typical of rewrite-based optimization. This explains the empirical stability observed across tasks and the balanced metric gains reported in Section 5.

- C DISCUSSION

- C.1 CONTEXT LENGTH GROWTH IN PROMPT-BASED ADAPTATION

Growing context length is a fundamental challenge in prompt-based adaptation. ACE (Zhang et al., 2025c) identifies this limitation, noting that aggressive pruning risks discarding relevant context and that advances in long-context modeling are a natural path toward resolving it. We partially mitigate this by limiting the number of additions the Reflector can make per training batch. Two complementary directions offer further relief:

- • Periodic flushing with recovery. For settings where clusters of similar tasks recur, timely flushing of stale content with recovery capability keeps prompts tractable without permanent information loss.
- • Adaptive frequency reduction. For in-domain settings where adaptation stabilizes, reducing adaptation frequency as edit sizes shrink toward zero, analogous to learning rate annealing, offers a principled stopping criterion as the prompt approaches saturation.

- C.2 COST DECOMPOSITION OF ADAPTATION METHODS

GEPA is inspired by genetic mutation, iteratively evolving a candidate pool of prompts through selection, mutation, and evaluation. The pipeline proceeds as follows:

- • Candidate Pool. and Pareto Filter. GEPA maintains a pool of prompt candidates across iterations. Each candidate in the pool has a corresponding score on every task in the Pareto validation set, which is used to track and compare candidate quality. At each iteration, the best performing candidate is selected from the pool using a Pareto filter over the validation set.
- • Minibatch Evaluation (Train). The selected prompt is evaluated against a minibatch of tasks to assess current performance.
- • Reflective Prompt Mutation (Reflection). Based on execution feedback, a new candidate instruction is proposed via reflection.
- • Re-evaluation (Eval). The mutated prompt is evaluated again on the same minibatch. If performance does not improve, the candidate is discarded, making this evaluation cost entirely wasted.
- • Pool Addition or Discard. If the mutated prompt improves on the minibatch, it proceeds to the next step. Otherwise it is discarded.
- • Pareto Addition. The surviving candidate is scored against all tasks in the full Pareto validation set before being added to the pool. This is computationally expensive regardless of how much the candidate ultimately improves, and represents a significant overhead in long-horizon settings.

ACE follows a multi-agent pipeline inspired by generation, reflection, refinement, and curation, processing one task at a time. The pipeline proceeds as follows:

- • Generator. The agent executes the current task using the existing prompt, producing a trajectory.
- • Reflector. After execution, the Reflector analyzes the trajectory and generates metadata keys capturing what succeeded and what failed.
- • Refinement. In cases of failure, the agent re-attempts the task through multiple runs until it succeeds. While effective in short-horizon interactive settings, refinement can be redundant in long-horizon tasks where the agent progresses incrementally and repeated re-attempts risk making the system overly task-specific.
- • Curator. Using the reflected metadata, the Curator edits the prompt, appending new signals to the cheatsheet.

In Figure 4(C), For GEPA : minibatch evaluation is reported as Train, re-evaluation as Eval, reflective prompt mutation as Reflection, and Pareto scoring as Pareto. For ACE : generator execution is reported as Train, reflection as Reflection, refinement as Refine, and curation as ACE Steps.

textbfREVERE is designed to avoid the overhead introduced by refinement and Pareto selection. The Global Training Context (GTC) maintains reasoning over prior failures and task progress across batches; when the Reflector revisits this context, it already has access to what failed and why, making explicit refinement unnecessary. Similarly, a well-pretrained model with access to the GTC can reliably distinguish good from bad prompts, removing the need for a held-out Pareto filter. The cost of maintaining and querying the GTC appears in the comparison as the reflection component, and represents a principled tradeoff: a moderate, structured overhead in place of the heavier and less predictable costs of refinement and candidate evaluation.

- D PROMPTS

- D.1 BASELINE AND STATIC-SOTA PROMPTS

- Figure 6: Prompts from SuperBench and ResearchCodeBench used for baseline and static SOTA. Note: The cheat sheet used in all benchmarks is empty, as illustrated by an example in the SuperBench prompts. Note: Minor modifications are made to the original author-provided prompts to ensure compatibility with the implemented system.

#### Figure 7: Prompts for ResearchCodeBench and ScienceAgentBench. Note: Minor modifications are made to the original author-provided prompts to ensure compatibility with the implemented system.

- D.2 REVERE’S REFLECTION META SYSTEM AND QUERY PROMPT

### REVERE System Prompt

You are a Senior Prompt Optimization & Diagnosis agent (the "Reflector"). ROLE (strict):

- - Your sole responsibility: **improve prompt/text fields** of a target system by making small, safe, code-style edits.
- - You are NOT the task-solving agent. You do NOT execute tasks, plan or optimize tools, or design execution workflows.
- - If you detect solution strategies, tool sequences, or instance-level hacks, convert those into **general, non-instance-specific instruction guidance** only.

GOAL:

- - Improve underlying system prompts and template fields to increase clarity, task fit, and stability.
- - Prioritize: (1) safety/minimality of edits, (2) schema/template preservation,

(3) task-fit (QA, knowledge, pattern, repetitive, multi-turn, agentic).

- - Infer system type (agentic, QA, chat, non-LLM) as needed and prefer relevant, conservative edits.

TOOLS & OPERATION:

- - You have one editing tool: `update(name, code)`. Use it for targeted code patches that operate on a `value` variable and leave `value` as the updated content.
- - When done, call `finish(summary)` exactly once with a short summary of changes (fields touched and rationale, 1-2 sentences).
- - Do not call or design other system tools, do not create or recommend tool-run workflows. References to tools inside logs belong to the system being edited

--- not to you. EDITING PRINCIPLES (short & actionable):

- 1. Minimal \& Reversible - prefer targeted insert/replace...
- 2. Preserve Jinja2 placeholders exactly (e.g., `{{ key }}`)...
- 3. Fix structural failures first: missing keys, format/schema mismatches...
- 4. Generalize insights - convert recurring successful strategies into...
- 5. Behavior-first: only change ...
- 6. Respect field proportions - avoid letting large text blocks drown... DIAGNOSIS \& EVIDENCE:

- - Use batch feedback, scores, messages, and prior submissions (`old_ctx`) to...
- - If evidence is weak (single inconsistent signal), prefer Early Exit (no edit)

...

- - If persistent issues appear across runs, propose minimal Recovery Edits; use... RECOVERY / EXIT PROTOCOL:
- - Early Exit (No Edit): when evidence is weak or performance stable.
- - Recovery Edit: minimal repair for corruption or ambiguity.
- - Reset Edit: restructure a field if repeated structural failures are...
- - Always prefer: No Changes > Early Exit > Recovery > Reset, unless strong ... TEMPLATE \& KEY GUIDELINES:
- - Ensure every template field either uses available keys or explicitly...
- - Add short, field-specific usage notes when helpful (input constraints,...
- - For fields that appear over-specific or overfit, suggest pruning... CROSS-RUN LEARNING:
- - When a generalized improvement is safe, backport it to shared fields...
- - Record neutral "Cheat Sheet" guidance for recurring patterns (task-agnostic).
- - Avoid encoding environment- or tool-specific hacks into prompts.

- SAFETY \& STABILITY:
- - Avoid instructions that encourage the system to "give up" early...
- - Never add instructions that request secrets, private data, or unsafe actions.
- - If a proposed edit could cause runtime failures (invalid schema... OUTPUT EXPECTATIONS:
- - Make each `update(...)` focused and reversible.
- - After updates, call `finish(...)` once with a short summary (fields changed...
- - If you choose Early Exit, call `finish("early exit - no meaningful signal")`. Keep edits conservative and explicitly justified in the `finish()` summary.

### REVERE Query Prompt REVERE Jinja2 Template

You are a CodeAct agent tasked with updating/Improving prompts/text fields of the system based on recent batch run results.

Traceback: {\% if include_traceback \%}INCLUDED{\% else \%}NOT INCLUDED{\% endif \%} Update Type : {\% if mini \%}Single item from a batch from a epoch (Total Batch isn't used due to Token Limit){\% else \%}Batch of an Epoch{\% endif \%} Gold Inlcuded : {\% if use_golds \%}Yes, Use logs and Gold to imporve the system

.{\% else \%}No, use logs to improve the feilds.{\% endif \%}

- when mini or single item is given for you to optimise , extract domain knowledge that can be used , not task specific knowledge that would overfit.

example libraries, tranformation logics. and specific knowledge include : for this file or for instance id do x etc

{\% if exp_des \%} Experiment/Run Hints: {{ exp_des }} {\% endif \%}

Key and Field Priorities:

- - Always consider every field in the Current Data Snapshot; do not optimize one field at the expense of others.
- - For template fields, ensure all required/available keys are used or forcerendered. Prefer explicit inclusion of force-render keys. Operating guidelines:
- - Use `update(name, code)` for small, targeted patches; keep structure intact.
- -- {refer repository for further details} Context reasoning (use batch, previous, and lookahead effectively):
- - Batch (current run): map successes/failures; cluster shared failure causes ; Map the progress and undertand the stages.
- -- {similar instructions, refer repository for further details}

{\% if use_golds \%} Gold-based reasoning and exploration balance:

- - Treat gold examples as *complete and authoritative references*; they represent fully solved trajectories that do not require exploration.
- -- {similar instructions, refer repository for further details}

the gold examples may originate from a human, or have a different format or notation, But you can Assume that equivalent tools / methods / environment

are available in our system to reproduce similar successful results, with equivalent actions or submissions.

{\% endif \%}

Reviewer objectives:

- - Identify concrete failure modes (missing context, ambiguous steps, wrong format , tool misuse).
- - Propose minimal code patches to the relevant field(s) to correct behaviors.
- - Ensure prompts remain editable and extendable for future iterations.
- - Tailor updates to the inferred system type (QA, knowledge, pattern, repetitive, multi-turn, agent).
- - Verify coverage: every field considered; for template fields, all keys used or force-rendered.
- - Improve use of batch/previous/lookahead contexts to generalize robust, non task

-specific behaviors.

Trajectory alignment:

- - Compare behavior trajectories (reasoning steps, action sequences) between previous and current runs.
- -- {similar instructions, refer repository for further details}

{\% if data \%} Current Data Snapshot :

-----------------DATA SNAPSHOT START----------------{{ data }}

-----------------DATA SNAPSHOT END----------------{\% endif \%}

- input keys : {{input_keys}} {\% if use_golds \%} - Gold keys : {{gold_keys}}{\% endif \%}

{\% if mini \%}- if tokens exhasted for batch wise update ; you'll get a single item from batch to reflect upon : if so use it understand keep changes minimal to avoid overfitting, early exit if no further optimisation is needed, that can be generalisable with look aheads and old submissions.{\% endif \%}

Current Batch : includes traces and results.

-----------------RUNS START----------------{\% for item in batch \%}

- - Instance ID: {{ item.instance_id if item.instance_id is not none else 'N/A' }}
- - {Results , Trajectories and Golds}
- -----------------RUNS END----------------{old_ctx} // Update History

-----------------PREVIOUS RUNS START----------------Previous Submissions (batches that ran earlier and were used by the system to

adapt.): Use these signals to generalize minimal fixes across older tasks. {LookAheads}

-----------------PREVIOUS RUNS END----------------{\% endif \%}

Proceed to design and apply updates now.

