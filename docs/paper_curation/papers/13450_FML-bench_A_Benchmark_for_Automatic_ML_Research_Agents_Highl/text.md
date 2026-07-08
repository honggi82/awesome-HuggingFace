## FML-bench: Benchmarking Machine Learning Agents for Scientific Research

Qiran Zou*1 Hou Hei Lam*12 Wenhao Zhao1 Yiming Tang1 Tingting Chen1 Samson Yu1 Tianyi Zhang3 Chang Liu2 Xiangyang Ji2 Dianbo Liu1

# arXiv:2510.10472v2[cs.CL]25Feb2026

### Abstract

Large language models (LLMs) have sparked growing interest in machine learning research agents that can autonomously propose ideas and conduct experiments. However, existing benchmarks predominantly adopt an engineeringoriented perspective: they emphasize applicationoriented tasks and evaluate primarily on final performance and computational cost, overlooking agents’ research processes and limiting assessment of their capabilities in scientific research settings. To more comprehensively evaluate agents in scientific research settings, we introduce FMLbench, a benchmark comprising 8 diverse and fundamental ML research tasks, and further propose complementary metrics, notably Exploration Diversity, which quantifies the variance of proposals across iterations and reveals how exploration patterns influence research outcomes. We evaluate state-of-the-art research agents on FMLbench, showing that agents employing broad exploration strategies exhibit higher exploration diversity and achieve superior performance, and that exploration diversity positively correlates with performance improvements across multiple tasks. We hope these findings and our benchmark inform future agent design and support the community in further investigating agent behavior. Our benchmark is available at: https: //github.com/qrzou/FML-bench.

### 1. Introduction

Large language models (LLMs) have catalyzed a resurgence of interest in machine learning (ML) research agents which assist or carry out parts of the scientific discovery workflow. These agents not only support hypothesis generation, cod-

*Equal contribution 1National University of Singapore, Singapore 2Tsinghua University, Beijing, China 3University of Minnesota, Minneapolis, MN, USA. Correspondence to: Qiran Zou <qiranzou@u.nus.edu>, Dianbo Liu <dianbo@nus.edu.sg>.

Preprint. February 26, 2026.

ing, and experiment management, but also increasingly act as collaborators in discovery by providing complementary perspectives that can accelerate machine learning research across domains. Within this landscape, agents that automatically propose ideas and run experiments are particularly compelling (Lu et al., 2024; Yamada et al., 2025). They close the loop from ideation to empirical validation to maximize automation and to speed up research cycles. Compared to settings that only elicit ideas and then use LLMs or humans to assess “novelty” and “feasibility” which often diverge from real-world utility, this approach evaluates agents based on actual experimental outcomes, providing objective and quantitative evidence of their effectiveness (Wang et al., 2024a; Baek et al., 2024; Si et al., 2024).

Despite rapid progress, existing benchmarks offer an incomplete picture of research competence, as shown in Tab. 1. These benchmarks predominantly adopt an engineeringoriented perspective rather than one aligned with scientific research settings, and this limitation manifests in two key aspects. First, regarding task construction, most benchmarks focus on Kaggle-style, application-oriented tasks that emphasize engineering execution (e.g., feature engineering, standardized model training, and optimization) while paying limited attention to evaluating an agent’s ability to tackle fundamental machine learning research problems, such as representation learning and generalization (Chan et al., 2024; Huang et al., 2023; Padigela et al., 2025; Jing et al., 2024). Second, regarding evaluation design, these benchmarks primarily assess final task performance metrics (e.g., accuracy, recall) and computational cost, while overlooking the characteristics of agents’ internal iterative processes, which limits analysis of how agent characteristics relate to research outcomes.

To address these gaps and more comprehensively evaluate agents in scientific research settings, we introduce FMLbench, a benchmark designed to assess automatic ML research agents on fundamental ML problems. FML-bench comprises 8 diverse tasks (Fig. 1) chosen to reflect bottlenecks that repeatedly surface in modern ML. The tasks span eight diverse and fundamental ML research problems, including generalization, data efficiency, representation learning, continual learning, causality, robustness, privacy, and fairness (see Section 3 for task design rationale and refer-

𝐶0

𝑅0

Task Specification

t

n

e

C

m

- r

o

- s

t

a

e

- s

-

d

o

m

a

i

n

- t

r

t

s

e

- o

u

- p

l

- a

- b

r

|Codebase|
|---|

|Baseline Results|
|---|

r

a

g

n

t

[Figure 1]

[Figure 2]

[Figure 3]

s

i

s

u

- r

o

- s

Task Description

f

e

q

r

c

E

a

{"splitMNIST": {"Accuracy": 0.2710,

GitHub_Repo_Sample/

You are working with the

|─ models/

Continual-Learning repo’s SI

...},

|─ train/ |─ eval/

baseline to improve accuracy and reduce forgetting on

n

}

- n

t

i

- o

I

n

|...

splitMNIST (5 tasks × 2

t e

e

- r e

a

- s o

t

e

r

|─ train.py |─ eval.py

classes, single-head over 10 classes). Your goal is ...

Target Metrics

e

[Figure 4]

v e

m

r

i

- n

t

i

- o n

- d

g

- e

t

Accuracy (Precision, …)

n

r

i

e

n g

v

e

[Figure 5]

o

l

[Figure 6]

Suggested Files for Editing

w

Protected Files

[Figure 7]

Experiment CMD List

[Figure 8]

- n

- o

GitHub_Repo_Sample/

- 1. python train.py \

--data=splitMNIST

- 2. python eval.py

GitHub_Repo_Sample/ |─ models/target_model.py

K

|─ eval/utils.py |─ eval.py

FML-bench

|─ train/trainer.py

|...

- m e a

- n

Input to Agent

D

Research Iterations

i

s

m

c

i

s

o

n

- n

g

f

r

- o

𝐻𝑡

𝑅𝑡

e

v

g

l

e

p

f

Hypothesis

Experiment Results

u

r

m

Propose Hypothesis

y o f

| |
|---|

l f

| |
|---|

a

i

e

n

s

a

r

w

a

- t

- u

e

e

[Figure 9]

- r e

- s

L

f

Generate Code Modification

𝑡→𝑡+1

Execute Experiment

| | | |
|---|---|---|
| | | |

P

r

o

𝑚𝑡

i

t

𝐶𝑡

- n

f

- o

e

n

o

c

t

o

t

i

e

o

i

r

c

t

Updated Codebase

Code Modification

n

m

p

n

Update Codebase

a

a

| |
|---|

u

a

g

- s

- t

t

r

a

i

i

r

s

i

o

n

o

e

- s

- t

n

R

c

l

l

e

a

a

i

r

k

a

a

- r

- s

g

e

- d

v

- e

a

- Figure 1. Overview of FML-bench. FML-bench includes 8 fundamental machine learning research tasks, designed to evaluate agents’ capabilities in solving machine learning research problems. Agents are assessed on their ability to solve machine learning problems through iterative research.

ences). Agents are expected to propose new or improved ML methods that deliver stronger empirical results than baselines across these tasks.

Moreover, to enable deeper analysis of agent behavior beyond final outcomes, we further propose complementary metrics that characterize the research process. Notably, we introduce Exploration Diversity, which quantifies the variance of proposals across iterations and reveals how exploration patterns influence research outcomes. Specifically, we compute code embeddings using GraphCodeBERT (Guo et al., 2020) and measure the dispersion of these embeddings around their centroid. Additionally, we propose Step Success Rate and Step Completion Rate to characterize agent reliability throughout the research process. We also report commonly used metrics, including Performance and Cost.

Beyond the two core contributions above, FML-bench addresses additional limitations of existing benchmarks. Some benchmarks provide only raw data without baseline code (Chan et al., 2024; Jing et al., 2024), making it difficult to systematically assess agents’ research capabilities while introducing coding barriers that can obscure academic merit (e.g., when sound ideas fail due to engineering pitfalls). Even when baseline codebases are provided, they are often handcrafted and tightly formatted (Huang et al., 2023; Padigela et al., 2025). In contrast, FML-bench tasks are constructed upon widely recognized ML research repositories with carefully designed task specifications, reflecting how researchers typically build upon established codebases to investigate new ideas. Furthermore, agents are not required to build entire codebases from scratch but can start from provided baselines, enabling them to focus on scientific ad-

vances in algorithms and architectures rather than purely engineering effort.

We evaluate several state-of-the-art automatic research agents on FML-bench. The results show that agents employing broad exploration strategies exhibit higher exploration diversity and achieve superior performance, and that exploration diversity positively correlates with performance improvements across multiple tasks.

We summarize our contributions as follows:

- • We construct FML-bench, a benchmark for scientific research centered on diverse fundamental ML problems, closing gaps left by application-oriented, engineering-heavy evaluations.
- • We propose complementary metrics for analyzing agent behavior throughout the scientific research process, notably Exploration Diversity, which quantifies the variance of proposals across iterations, along with Step Success Rate and Step Completion Rate for assessing agent reliability.
- • We report empirical findings suggesting that broad exploration strategies and high exploration diversity are important contributors to effective ML research agents.

### 2. Related Works

#### 2.1. Automatic Research Agents

With the emergence of large language models (LLMs), research agents have increasingly been explored as tools to

- Table 1. Comparison of ML agent benchmarks across key design goals. Repo refers to the repository, and Comp denotes Competition. ∗: In MLAgentBench, only part of the tasks meet this requirement; users must prepare baseline and evaluation code even when some tasks are based on real-world Kaggle repositories.

Design Goals Ours MLE–Bench MLAgentBench ML–Dev–Bench DSBench

Fundamental ML Problem Focus ✓ ✗ ✗ ✗ ✗ Process-level Evaluation ✓ ✗ ✗ ✗ ✗ Real-World Repo/Comp ✓ ✓ ✓∗ ✗ ✓ Low Coding Barrier ✓ ✗ ✓ ✓ ✗

support core components of the scientific workflow. These agents are capable of generating and prioritizing research ideas, retrieving and synthesizing literature, and simulating peer review processes. For instance, SciMON (Wang et al.,

- 2024a) and Nova (Hu et al., 2024) implemented frameworks for generating diverse and novel research ideas. AutoSurvey (Wang et al., 2024b) presented an automated literature review framework that performs retrieval over a large arXiv corpus, followed by outline planning and section drafting using specialized models. Meanwhile, AgentReview (Jin et al., 2024) employed LLM agents to simulate peer reviews, rebuttals, and committee discussions, offering insights into the dynamics of academic decision-making.

Recent efforts are moving beyond assistance toward fully automatic research agents. These systems aim not only to support researchers but to generate ideas, implement them, run experiments, and refine approaches without human supervision. One representative system is AIDE (Jiang et al.,

- 2025), a tree-search agent that optimizes user-defined metrics by iteratively editing and evaluating code, though it executes only one file and modifies a specific target file per iteration. TheAIScientist (Lu et al., 2024) represents an independent line of work, demonstrating end-to-end autonomy across the research process, including idea generation, implementation, experimentation, analysis, and manuscript drafting. Its improved version (Yamada et al., 2025) further reduces reliance on hand-crafted templates, enhancing generality across tasks. Similarly, the AgentLaboratory executes a full pipeline for automatic research, but its evaluation is limited to relatively simple research questions. Separately, AlphaEvolve (Novikov et al., 2025) adopts an evolutionary approach, iteratively refining and selecting promising ideas through variation and empirical evaluation. Beyond the computer science domain, a growing number of research agents have been developed for other fields, including chemistry, where they are used to investigate and optimize chemical processes (Boiko et al., 2023; M. Bran et al., 2024), and biomedical science, where they have been applied to the discovery of novel nanobodies (Swanson et al., 2024).

#### 2.2. Benchmarks for ML Agents

Existing benchmarks have begun to evaluate agents on codeintensive tasks, yet they remain limited in both scope and

flexibility. MLAgentBench (Huang et al., 2023) includes 13 machine learning engineering tasks, but most are implemented as single-file scripts, which is not practical for real-world scenarios. In addition, it requires to set individual evaluator for each task and lacks support, limiting its scalability to support more tasks. MLE-Bench (Chan et al., 2024) covers 75 Kaggle competitions and assesses whether agents can function as machine learning engineers. It emphasizes tasks such as data pipeline management, experiment orchestration, and submission formatting, which may shift focus away from core machine learning understanding. ML-DevBench (Padigela et al., 2025) places greater emphasis on engineering aspects such as dataset loading and API integration. It evaluates agents’ ability to improve existing baselines only in performance tests, which are relatively simple due to narrow task scopes like classification and segmentation, and the use of fixed starter files. In contrast, our benchmark includes tasks spanning diverse machine learning domains. DSBench (Jing et al., 2024) aggregates 466 data analysis tasks and 74 modeling tasks from ModelOff and Kaggle, focusing on problem-solving within data science workflows. By comparison, our benchmark focuses on 8 diverse and fundamental machine learning research tasks. It is built on real-world codebases, thereby providing practical challenges and extensibility by construction, while maintaining a low coding barrier.

### 3. Benchmark Task Design

Prior surveys have identified several critical dimensions that machine learning systems must address to achieve trustworthiness and practical reliability, including robustness, generalization, fairness, and privacy (Li et al., 2023; Mehrabi et al., 2021; Wang et al., 2022; Goodfellow et al., 2014; Abadi et al., 2016). Complementary work on learning systems emphasizes the importance of data efficiency, representation learning, causality, and continual adaptation for building capable and generalizable AI (Adadi, 2021; Bengio et al., 2013; Pearl, 2019; Chen & Liu, 2018). Guided by these foundational challenges, we constructed eight diverse tasks, each representing a distinct research challenge drawn from one of these domains.

For each task, we carefully selected benchmark datasets

and baseline methods according to two guiding principles. First, datasets should be both computationally tractable and sufficiently challenging: we required that baseline methods complete training and evaluation within 2 hours, while prioritizing datasets that present meaningful research challenges to differentiate agent capabilities. Second, baseline methods should be classic and widely recognized, yet remain sufficiently below theoretical optima to provide headroom for improvement. This design ensures that agents have sufficient room for exploration while enabling meaningful comparisons of their research capabilities across varying magnitudes of improvement.

improved causal inference strategies that minimize the mean absolute error in treatment effect estimation.

Robustness and Reliability. This task probes the ability to defend against adversarial data corruption. We constructed a poisoned MNIST dataset with diverse backdoor attacks including edge-based triggers and distributed attack patterns, and adopted dp-instahide (Borgnia et al., 2021) as the baseline defense. Agents must propose defenses that improve the defense score, which balances clean accuracy and resistance to backdoor attacks.

Generalization. This task evaluates the ability to develop algorithms that transfer across domains. We selected ColoredMNIST (Arjovsky et al., 2019), which introduces spurious correlations to create a controlled testbed for distribution shift, and adopted ERM (Vapnik, 1998) as the baseline. Agents are required to propose improved or novel algorithms that enhance out-of-domain accuracy on a held-out target domain while training only on the source domain.

Privacy. This task assesses the ability to protect against membership inference attacks. We selected CIFAR-10 (Krizhevsky et al., 2009) and adopted Wide-ResNet-28-2 (Zagoruyko & Komodakis, 2016) as the baseline, evaluated using both standard and robust membership inference attacks. Agents are required to design defense mechanisms that reduce attack AUC toward 0.5 while maintaining classification accuracy.

Data Efficiency. This task measures the ability to enhance few-shot learning under tight data constraints. We selected Mini-ImageNet (Vinyals et al., 2016), whose fine-grained inter-class similarities present meaningful challenges, and adopted Prototypical Networks (Snell et al., 2017) as the baseline. Agents must propose improved metric-based classification algorithms that boost few-shot accuracy while operating under a frozen backbone.

Representation Learning. This task tests the ability to learn meaningful features from unlabeled data via selfsupervised learning. We selected CIFAR-10 (Krizhevsky et al., 2009) for computational efficiency and adopted MoCo (He et al., 2020), a widely recognized contrastive method, as the baseline. Agents are required to improve the pretraining algorithm to achieve higher linear probing accuracy on frozen encoder features.

Continual Learning. This task evaluates long-term adaptability and the ability to mitigate catastrophic forgetting. We selected splitMNIST (Deng, 2012) under class-incremental learning, where models learn five sequential tasks with a shared output head, and adopted Synaptic Intelligence (Zenke et al., 2017) as the baseline. Agents must propose methods that improve average accuracy across all tasks while avoiding catastrophic forgetting.

Causality. This task assesses the capacity to estimate treatment effects for intervention reasoning. We selected the IHDP dataset (Hill, 2011), a semi-synthetic benchmark enabling ground-truth evaluation, and adopted DragonNet (Shi et al., 2019) as the baseline. Agents are required to develop

Fairness and Bias. This task measures the ability to balance equitable outcomes with model utility. We selected the COMPAS dataset (Angwin et al., 2016), which presents documented disparities across protected attributes, and adopted Adversarial Debiasing as the baseline. Agents must minimize absolute average odds difference toward parity while maintaining or improving classification accuracy.

### 4. Evaluation Protocol

#### 4.1. Evaluation Metrics

Existing benchmarks for automatic ML research agents primarily adopt an engineering-oriented evaluation, emphasizing final task performance while overlooking the characteristics of agents’ iterative research processes. To better capture agent behavior throughout scientific discovery, we go beyond standard metrics (Performance and Cost) and introduce additional process-oriented metrics, including Exploration Diversity, Step Success Rate, and Step Completion Rate.

To formally define these metrics, we first describe the iterative research process. Consider an agent conducting research over T iterations. At iteration t ∈ {1,...,T}, the agent generates a hypothesis ht, which is instantiated as a code modification mt applied to the codebase Ct−1, yielding Ct = Ct−1 ⊕ mt. Let et denote the code embedding of Ct extracted using GraphCodeBERT (Guo et al., 2020). After completing all iterations, the agent has produced a set of codebases {C1,...,CT} with corresponding embeddings {e1,...,eT}. Based on this formulation, we define the following metrics.

Performance. Performance serves as the primary objective metric for evaluating agent effectiveness. It measures the best task-specific result achieved across all iterations on the held-out test set. Let perf(Ct) denote the evaluation metric for codebase Ct. Performance is defined as:

P =

maxt∈{1,...,T} perf(Ct) if higher is better mint∈{1,...,T} perf(Ct) if lower is better

. (1)

This metric captures the agent’s ability to discover effective solutions through iterative refinement, evaluated using protected evaluation code to ensure consistent and fair comparison across agents.

Exploration Diversity. Exploration Diversity quantifies the variance of proposals across iterations, revealing how exploration patterns influence research outcomes. Let e¯ = n1 nt=1 et denote the centroid of all code embeddings, where n is the number of iterations with valid code outputs. Exploration Diversity is computed as:

1 n

D =

n

∥et − e¯∥2. (2)

t=1

GraphCodeBERT (Guo et al., 2020) leverages data-flow graphs and code structure rather than surface tokens alone, ensuring that code snippets with identical logic but different variable or function names receive high similarity scores. Such pairs are appropriately treated as low diversity. Greater dispersion indicates broader exploration of implementation strategies within a research run.

Intuitively, higher exploration diversity reflects broader coverage of the hypothesis space, increasing the likelihood of discovering effective solutions. This perspective aligns with classical results in stochastic search and multi-armed bandits (Lai & Robbins, 1985; Bubeck et al., 2012), as well as novelty search in evolutionary optimization (Lehman & Stanley, 2011; Pugh et al., 2016), where broader exploration improves the probability of finding high-reward regions. We verify this relationship empirically in Section 5.

Step Success Rate. Step Success Rate captures the fraction of iterations that produce valid experimental results without errors, reflecting an agent’s coding competence and ability to generate syntactically correct, semantically coherent code. It is defined as:

Nsuc Ncomp

, (3)

S =

where Nsuc is the number of iterations whose experiments execute without errors and yield valid results, and Ncomp is the number of iterations actually executed by the agent.

Step Completion Rate. Step Completion Rate measures the proportion of executed iterations relative to the assigned total, indicating whether agents can complete full experimental workflows without premature termination. It is defined as:

Ncomp Ntotal

, (4)

R =

where Ntotal is the total number of iterations assigned to the agent. Together with Step Success Rate, this metric characterizes overall agent reliability throughout the research process.

Cost. Cost accounts for computational and temporal expenditure required during the research process. We report four cost components: (1) total token consumption, the cumulative tokens used across all iterations; (2) step token consumption, the average tokens per iteration; (3) total time consumption, the wall-clock time for a complete experimental run; and (4) step time consumption, the average time per iteration. These components collectively characterize both computational and temporal efficiency.

#### 4.2. Unified Input-output Interface

In real-world scientific research, different ML projects often employ distinct codebases with varying execution pipelines, output formats, and evaluation protocols, making it challenging to construct a unified benchmark across diverse research problems. A core design of FML-bench addresses this challenge through unified input-output interfaces that accommodate repository diversity while preserving their inherent complexity. This design enables our benchmark to support diverse ML research tasks built upon existing repositories, bringing evaluation closer to real-world scientific research settings.

Input Existing benchmark designs struggle to handle different GitHub repositories. Data-based benchmarks accept only datasets and task descriptions as inputs. And benchmarks providing codebase assume unified training script names, single-stage training, no customizable arguments, or requiring to set an individual evaluator manually. In contrast, real repositories use different script names, multi-stage pipelines, diverse training arguments, and include their own evaluator already. Our solution treats the complete execution sequence (training and evaluation commands) as a single input unit so that the agent receives a command list for running experiments. Therefore, our benchmark provide following resources (as shown in Fig. 1) to agent as input: 1) task description with objectives and expected outputs, 2) complete repository code, 3) suggested files for modification, 4) protected code segments that cannot be modified to preserve evaluation integrity, 5) command list for running experiments, 6) baseline performance, and 7) target

TheAIScientist AIDE Claude Code

…

…

…

…

…

Parallel Exploration

Tree-based Exploration

Linear Refinement

Breadth: Wide Depth: Shallow

Breadth: Medium Depth: Medium

Breadth: Narrow Depth: Deep

- Figure 2. Comparison of research exploration strategies of different agents. TheAIScientist uses parallel exploration for broad coverage, AIDE employs hierarchical tree-based search balancing exploration and exploitation, while Claude Code follows linear refinement for sequential improvement.

improvement metrics.

assigned a fixed budget of total steps = 100. We select the best result achieved among the three rounds based on the target metric computed on the test set.

Output Repository outputs are various (e.g. from text files to JSON formats). However, most outputs share a common structure: performance metrics on specific datasets. We provide a post-processing module that converts diverse task outputs into a standardized format, enabling consistent metric extraction across all tasks while preserving native output mechanisms. This design bridges the gap between evaluation standardization and real-world repository diversity, enabling rigorous assessment of agents on practical code optimization tasks.

#### 5.2. Implementation Details

Agents To enable these agents to operate on our benchmark, several modifications were necessary. We adapted TheAIScientist by fixing compatibility issues and extending its functionality to support the requirements of our benchmark, such as executing experiments in real repositories and reporting results consistently. As for AIDE, we employed its cloud-based commercial variant, Weco, as the operational interface, adapting our benchmark to integrate with its workflow despite limited control over its internal mechanisms. For Claude Code, we designed a prompting scheme (see Appendix E) that enabled it to function as an automatic research agent, capable of reading code, generating hypotheses, and proposing modifications grounded in experimental feedback.

Evaluation Integrity Our benchmark implements protective measures to ensure evaluation integrity. Specifically, we protect evaluation files as read-only and prevent agent modification. Agents must utilize these protected evaluation files to assess their proposed methods.

### 5. Experiments

LLMs adopted for agents We employ GPT-5 (2025-0807) and Gemini-2.5-Pro (2025-06-17) for TheAIScientist and AIDE. For Claude Code, it is constrained to its native models and therefore we use Opus-4.1 (2025-08-05).

#### 5.1. Settings

Selections of Agents As shown in Fig. 2, we explore three automatic machine learning research agents, each adopting a distinct research strategy. TheAIScientist follows a broad exploration approach, generating and testing a wide range of hypotheses in parallel across multiple experimental directions. AIDE employs a hierarchical, tree-based search strategy, balancing the exploration of new possibilities with the exploitation of promising results. And we prompt Claude Code to employ a linear refinement strategy, sequentially improving its hypotheses and code implementations to address ML tasks.

#### 5.3. Results and Discoveries

5.3.1. COMPARISON OF AGENTS WITH DIFFERENT RESEARCH STRATEGIES

As shown in Tab. 2, the combination of TheAIScientist with Gemini-2.5-Pro achieved the best performance, ranking first in 4 out of 8 tasks. The combination of AIDE with Gemini2.5-Pro ranked second, securing top results in 2 out of the 8 tasks. These findings suggest that TheAIScientist performs better in discovering novel and effective machine learning methods, compared to AIDE and Claude Code.

Experimental Protocol Each agent is required to execute in three independent rounds. In each round, the agent is

- Table 2. Comparison of Performance among different agents. G2.5-Pro denotes Gemini-2.5-Pro. (↑) indicates higher is better, (↓) indicates lower is better.

ML Problems Baseline

TheAIScientist AIDE Claude Code GPT-5 G2.5-Pro GPT-5 G2.5-Pro Opus-4.1

Generalization (↑) 0.2254 0.5036 0.3252 0.2254 0.2254 0.5036 Data Eff. (↑) 0.6547 0.7689 0.8231 0.6547 0.6547 0.6571 Rep. Learn. (↑) 0.7562 0.7796 0.8597 0.8469 0.8466 0.7725 Cont. Learn. (↑) 0.2710 0.4281 0.7808 0.4369 0.3658 0.2337 Causality (↓) 1.1445 1.0063 0.9925 0.9683 0.9549 0.9840 Robust. & Rel.(↑) 0.4848 0.9311 0.9205 0.9174 0.9633 0.8921 Privacy (↓) 0.8114 0.4908 0.1750 0.4882 0.4814 0.4892 Fair. & Bias (↓) 0.3787 0.0603 0.1002 0.0385 0.0917 0.3787

- Table 3. Comparisons of different agents across Exploration Diversity, Step Success Rate, Step Completion Rate, Cost. M stands for Million, H for Hours, and Min for Minutes. ∗ AIDE does not support recording token usage.

TheAIScientist AIDE Claude Code GPT-5 G2.5-Pro GPT-5 G2.5-Pro Opus-4.1

Metrics

Exploration Diversity 28.46±9.97 20.86±4.42 28.60±13.32 18.41±12.68 8.75±6.07 Step Success Rate (↑) 0.83±0.27 0.80±0.26 0.64±0.30 0.66±0.34 0.83±0.32 Step Completion Rate (↑) 1.00±0.00 1.00±0.00 0.54±0.38 0.69±0.24 0.07±0.06 Total Token Cons. (M) (↓) 6.04±2.18 5.45±2.51 N/A∗ N/A∗ 9.06±3.64 Step Token Cons. (M) (↓) 0.06±0.02 0.05±0.03 N/A∗ N/A∗ 3.10±4.34 Total Time Cons. (H) (↓) 9.42±5.25 11.74±8.37 5.46±6.92 9.49±7.24 1.18±1.80 Step Time Cons. (Min) (↓) 11.05±10.71 14.23±14.55 12.55±12.17 10.48±8.29 13.26±11.29

As illustrated in Fig. 2, TheAIScientist adopts a research exploration strategy that is broad but shallow, while AIDE maintains both medium breadth and depth. In contrast, Claude Code exhibits a narrow yet deep exploration pattern. For a detailed explanation of the research strategies adopted by each agent, refer to Section 5.1. Considering the research exploration strategy together, the results suggest that a broader research exploration space adopted by TheAIScientist is more effective for discovering promising ideas. This insight offers practical guidance for machine learning research agent that broadly exploring diverse ideas could be more productive than focusing on a single direction.

5.3.2. ANALYSIS OF EXPLORATION DIVERSITY

As shown in Tab. 2 and Tab. 3, several key findings emerge from our experimental results. First, TheAIScientist and AIDE demonstrate significantly higher exploration diversity compared to Claude Code, while also exhibiting superior overall performance across the tasks. Second, when both systems employ GPT-5 as the underlying language model, TheAIScientist and AIDE achieve comparable diversity scores (28.46 vs. 28.60, respectively), and both methods attain one best result each. Third, when utilizing Gemini2.5-Pro, TheAIScientist exhibits higher diversity than AIDE

(20.85 vs. 18.41) and achieves a greater number of best results (4 vs. 2). These observations suggest that, with the LLM held fixed, agents with greater exploration diversity tend to achieve better outcome performance. In contrast, when the agent is held fixed, exploration diversity is not a decisive factor; performance also depends on other capabilities of different LLMs (e.g., reasoning and code-writing ability).

Furthermore, TheAIScientist achieves a mean exploration diversity of 24.66, which exceeds both AIDE’s 23.50 and Claude Code’s 8.75. This enhanced diversity can potentially be attributed to TheAIScientist’s more exploration breadthoriented search strategy, as illustrated in Sec. 5.3.1.

To further validate the relationship between exploration diversity and performance, we computed the correlation coefficients between exploration diversity and performance. As presented in Tab. 4, our findings reveal positive correlations across five tasks, with p-values indicating significant, borderline significant, or suggestive evidence of this relationship (p < 0.05 for significant; p < 0.10 for borderline significant; p < 0.15 for suggestive evidence; p ≥ 0.15 for not significant). Notably, for these five tasks, higher exploration diversity was associated with better performance.

- Table 4. Correlation analysis between exploration diversity and outcome. When computing the correlation, the outcome scores are negated for tasks where lower values indicate better performance, ensuring that positive correlations consistently represent improved performance with increased exploration diversity.

Metric Gen. Data Eff. Rep. Learn. Cont. Learn. Causality Robust. Privacy Fair.

p-value 0.1137 0.0121 0.1358 0.0002 0.4668 0.1944 0.6479 0.0517 Correlation 0.4418 0.6287 0.4036 0.8374 -0.2036 -0.3846 -0.1286 0.5292

- 5.3.3. TOKEN AND TIME CONSUMPTION

- 5.3.5. ADDITIONAL OBSERVATIONS

Shallow edits We found that AIDE sometimes misinterprets the structure and logic of target codebases. In certain cases, it generated new classes or components that were never integrated into the actual execution, resulting in no functional improvement over the baseline. As shown in Tab. 2, AIDE failed to improve the baseline in tasks related to Generalization and Data Efficiency. This may stem from the fact that AIDE only supports iterative modifications on a single file. However, real-world ML research codebases are often complex and span multiple files, making AIDE insufficient for addressing realistic research tasks.

Premature termination We encountered the issue of early termination of AIDE and Claude Code. For AIDE, the agent sometimes terminated prematurely due to its commercial version Weco which relies on cloud infrastructure that occasionally failed during execution. For Claude Code, early stopping was often triggered by the model’s internal reasoning, where the LLM would decide not to continue even when further actions are possible.

- 6. Conclusion

Tab. 3 reports both the average token and time consumption per step, as well as the total token and time usage for a complete experimental run. We observe that TheAIScientist consumes more tokens than AIDE, while Claude Code, despite its lower performance, uses the highest number of tokens among the three agents. This indicates that dedicated automatic ML research agents, such as TheAIScientist and AIDE, are more suitable for ML research problems in terms of both performance and token efficiency compared to general-purpose agents like Claude Code. In terms of time per step, all three agents show similar durations, with differences of around 2 minutes, which are not substantial. However, AIDE and Claude Code exhibit significantly shorter total execution times for a full experiment. This is primarily due to premature termination issues, as evidenced by the step completion rate, which leads to reduced overall time usage.

5.3.4. CASE ANALYSIS

Worst Case Analysis TheAIScientist strongly favors algorithmic modifications (56.2%) with zero implementation errors; AIDE presents balanced distribution but higher implementation issues (31.2%); while Claude Code shows elevated parameter configuration rates (37.5%) matching its algorithmic modification rate. Across LLMs, Gemini2.5-Pro exhibits extremely strong algorithmic modification tendency (68.8%), GPT-5 primarily focuses on structural modifications (50.0%), and Opus-4.1 maintains balanced distribution between parameter configuration and algorithmic modifications. See Appendix B for detailed results.

In this work, we introduce FML-bench, a benchmark designed to evaluate ML research agents in scientific research settings. Unlike existing benchmarks that adopt an engineering-oriented perspective through applicationoriented tasks and performance-focused evaluation, FMLbench comprises 8 diverse and fundamental ML research problems constructed upon widely recognized academic repositories. We further propose complementary metrics that characterize agent behavior throughout the research process, notably Exploration Diversity, along with Step Success Rate and Step Completion Rate for assessing agent reliability. Through systematic evaluation of state-of-the-art research agents on FML-bench, we find that agents employing broad exploration strategies exhibit higher exploration diversity and achieve superior performance, and that exploration diversity positively correlates with performance improvements across multiple tasks. We hope FML-bench and our findings provide a foundation for evaluating research agents from a scientific research perspective and offer insights for building more effective research agents.

Best Case Analysis TheAIScientist shows a strong algorithmic innovation focus, predominantly targeting core algorithmic logic; AIDE combines strong algorithmic innovation with hyperparameter tuning; while Claude Code uniquely favors architectural modifications including layer additions and normalization adjustments. Across LLMs, GPT-5 exclusively prioritizes algorithmic modifications (100.0%), Gemini-2.5-Pro exhibits the most diversified approach with unique data augmentation efforts (28.6%), and Opus-4.1 balances attention between architectural and algorithmic modifications. See Appendix B for detailed results.

### Impact Statement

This work impacts ML applications by shifting the evaluation of research agents from narrow, answer-matching benchmarks toward realistic, process-oriented scientific problem solving, which may lead to agents that generalize better and provide more reliable assistance in real research workflows. While research agents themselves may raise ethical concerns (e.g., misuse or over-automation of scientific judgment), FML-bench does not introduce inherent ethical risks; rather, it serves as a neutral tool for observing and analyzing agent behavior. Moreover, by grounding tasks in realistic research problems and emphasizing the investigation of agent behavior, FML-bench promotes more meaningful progress and greater transparency in research agent evaluation. In addition, our proposed unified input–output interfaces enable extensibility across repositories, datasets, and baselines, which we believe can help encourage diverse, reproducible, and behavior-focused evaluation.

### Acknowledgements

We gratefully acknowledge Zhengyao Jiang and Weco (https://www.weco.ai/) for their support and for providing access to their more general agent, which extended beyond the limitations of the original AIDE and enabled us to run AIDE as a baseline on our benchmark.

### References

Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, pp. 308–318, 2016.

Adadi, A. A survey on data-efficient algorithms in big data era. Journal of Big Data, 8(1):24, 2021.

Angwin, J., Larson, J., Mattu, S., and Kirchner, L. Machine bias: There’s software used across the country to predict future criminals. and it’s biased against blacks. ProPublica, May 2016. URL https://www.propublica .org/article/machine-bias-risk-asses sments-in-criminal-sentencing.

Arjovsky, M., Bottou, L., Gulrajani, I., and LopezPaz, D. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

Baek, J., Jauhar, S. K., Cucerzan, S., and Hwang, S. J. Researchagent: Iterative research idea generation over scientific literature with large language models. arXiv preprint arXiv:2404.07738, 2024.

Bellamy, R. K. E., Dey, K., Hind, M., Hoffman, S. C., Houde, S., Kannan, K., Lohia, P., Martino, J., Mehta, S.,

Mojsilovic, A., Nagar, S., Ramamurthy, K. N., Richards, J., Saha, D., Sattigeri, P., Singh, M., Varshney, K. R., and Zhang, Y. AI Fairness 360: An extensible toolkit for detecting, understanding, and mitigating unwanted algorithmic bias, October 2018. URL https://arxi v.org/abs/1810.01943.

Bengio, Y., Courville, A., and Vincent, P. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8): 1798–1828, 2013.

Boiko, D. A., MacKnight, R., Kline, B., and Gomes, G. Autonomous chemical research with large language models. Nature, 624(7992):570–578, 2023.

Borgnia, E., Geiping, J., Cherepanova, V., Fowl, L., Gupta, A., Ghiasi, A., Huang, F., Goldblum, M., and Goldstein, T. Dp-instahide: Provably defusing poisoning and backdoor attacks with differentially private data augmentations. arXiv preprint arXiv:2103.02079, 2021.

Bubeck, S., Cesa-Bianchi, N., et al. Regret analysis of stochastic and nonstochastic multi-armed bandit problems. Foundations and Trends® in Machine Learning, 5 (1):1–122, 2012.

Chan, J. S., Chowdhury, N., Jaffe, O., Aung, J., Sherburn, D., Mays, E., Starace, G., Liu, K., Maksin, L., Patwardhan, T., et al. Mle-bench: Evaluating machine learning agents on machine learning engineering. arXiv preprint arXiv:2410.07095, 2024.

Chen, H., Harinen, T., Lee, J.-Y., Yung, M., and Zhao, Z. Causalml: Python package for causal machine learning, 2020.

Chen, Z. and Liu, B. Lifelong machine learning. Morgan & Claypool Publishers, 2018.

Deng, L. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6):141–142, 2012.

Goodfellow, I. J., Shlens, J., and Szegedy, C. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.

Gulrajani, I. and Lopez-Paz, D. In search of lost domain generalization. arXiv preprint arXiv:2007.01434, 2020.

Guo, D., Ren, S., Lu, S., Feng, Z., Tang, D., Liu, S., Zhou, L., Duan, N., Svyatkovskiy, A., Fu, S., et al. Graphcodebert: Pre-training code representations with data flow. arXiv preprint arXiv:2009.08366, 2020.

He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation

learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9729–9738, 2020.

Hill, J. L. Bayesian nonparametric modeling for causal inference. Journal of Computational and Graphical Statistics, 20(1):217–240, 2011.

Hu, X., Fu, H., Wang, J., Wang, Y., Li, Z., Xu, R., Lu, Y., Jin, Y., Pan, L., and Lan, Z. Nova: An iterative planning and search approach to enhance novelty and diversity of llm generated ideas. arXiv preprint arXiv:2410.14255, 2024.

Huang, Q., Vora, J., Liang, P., and Leskovec, J. Mlagentbench: Evaluating language agents on machine learning experimentation. arXiv preprint arXiv:2310.03302, 2023.

Jiang, Z., Schmidt, D., Srikanth, D., Xu, D., Kaplan, I., Jacenko, D., and Wu, Y. Aide: Ai-driven exploration in the space of code. 2025. URL https://arxiv.or g/abs/2502.13138.

Jin, Y., Zhao, Q., Wang, Y., Chen, H., Zhu, K., Xiao, Y., and Wang, J. Agentreview: Exploring peer review dynamics with llm agents. arXiv preprint arXiv:2406.12708, 2024.

Jing, L., Huang, Z., Wang, X., Yao, W., Yu, W., Ma, K., Zhang, H., Du, X., and Yu, D. Dsbench: How far are data science agents from becoming data science experts? arXiv preprint arXiv:2409.07703, 2024.

Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. 2009.

Lai, T. L. and Robbins, H. Asymptotically efficient adaptive allocation rules. Advances in applied mathematics, 6(1): 4–22, 1985.

Lehman, J. and Stanley, K. O. Abandoning objectives: Evolution through the search for novelty alone. Evolutionary computation, 19(2):189–223, 2011.

Li, B., Qi, P., Liu, B., Di, S., Liu, J., Pei, J., Yi, J., and Zhou, B. Trustworthy ai: From principles to practices. ACM Computing Surveys, 55(9):1–46, 2023.

Lightly-AI. Lightly: A python library for self-supervised learning on images. https://github.com/light ly-ai/lightly, 2025.

Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., and Ha, D. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.

M. Bran, A., Cox, S., Schilter, O., Baldassari, C., White,

- A. D., and Schwaller, P. Augmenting large language models with chemistry tools. Nature Machine Intelligence, 6

(5):525–535, 2024.

Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., and Galstyan, A. A survey on bias and fairness in machine learning. ACM computing surveys (CSUR), 54(6):1–35, 2021.

Murakonda, S. K. and Shokri, R. Ml privacy meter: Aiding regulatory compliance by quantifying the privacy risks of machine learning. arXiv preprint arXiv:2007.09339, 2020.

Nicolae, M.-I., Sinn, M., Tran, M. N., Buesser, B., Rawat,

- A., Wistuba, M., Zantedeschi, V., Baracaldo, N., Chen,
- B., Ludwig, H., et al. Adversarial robustness toolbox v1. 0.0. arXiv preprint arXiv:1807.01069, 2018.

Novikov, A., V˜u, N., Eisenberger, M., Dupont, E., Huang, P.-S., Wagner, A. Z., Shirobokov, S., Kozlovskii, B., Ruiz, F. J., Mehrabian, A., et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.

Padigela, H., Shah, C., and Juyal, D. Ml-dev-bench: Comparative analysis of ai agents on ml development workflows. arXiv preprint arXiv:2502.00964, 2025.

Pearl, J. The seven tools of causal inference, with reflections on machine learning. Communications of the ACM, 62

(3):54–60, 2019.

Pugh, J. K., Soros, L. B., and Stanley, K. O. Quality diversity: A new frontier for evolutionary computation. Frontiers in Robotics and AI, 3:40, 2016.

Shi, C., Blei, D., and Veitch, V. Adapting neural networks for the estimation of treatment effects. Advances in neural information processing systems, 32, 2019.

Si, C., Yang, D., and Hashimoto, T. Can llms generate novel research ideas? a large-scale human study with 100+ nlp researchers. arXiv preprint arXiv:2409.04109, 2024.

Sicara. Easy few-shot learning: ready-to-use code and tutorial notebooks for few-shot image classification. ht tps://github.com/sicara/easy-few-sho t-learning, 2024.

Snell, J., Swersky, K., and Zemel, R. Prototypical networks for few-shot learning. Advances in neural information processing systems, 30, 2017.

Swanson, K., Wu, W., Bulaong, N. L., Pak, J. E., and Zou, J. The virtual lab: Ai agents design new sars-cov-2 nanobodies with experimental validation. bioRxiv, pp. 2024–11, 2024.

van de Ven, G. M., Tuytelaars, T., and Tolias, A. S. Three types of incremental learning. Nature Machine Intelligence, 4:1185–1197, 2022.

Vapnik, V. N. Statistical Learning Theory. WileyInterscience, New York, 1998.

Vinyals, O., Blundell, C., Lillicrap, T., Wierstra, D., et al. Matching networks for one shot learning. Advances in neural information processing systems, 29, 2016.

Wang, J., Lan, C., Liu, C., Ouyang, Y., Qin, T., Lu, W., Chen, Y., Zeng, W., and Yu, P. S. Generalizing to unseen domains: A survey on domain generalization. IEEE transactions on knowledge and data engineering, 35(8): 8052–8072, 2022.

Wang, Q., Downey, D., Ji, H., and Hope, T. Scimon: Scientific inspiration machines optimized for novelty. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 279–299, 2024a.

Wang, Y., Guo, Q., Yao, W., Zhang, H., Zhang, X., Wu, Z., Zhang, M., Dai, X., Wen, Q., Ye, W., et al. Autosurvey: Large language models can automatically write surveys. Advances in neural information processing systems, 37: 115119–115145, 2024b.

Yamada, Y., Lange, R. T., Lu, C., Hu, S., Lu, C., Foerster, J., Clune, J., and Ha, D. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025.

Zagoruyko, S. and Komodakis, N. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.

Zenke, F., Poole, B., and Ganguli, S. Continual learning through synaptic intelligence. In International conference on machine learning, pp. 3987–3995. PMLR, 2017.

### A. Task Design Details

Generalization. This task evaluates the ability to develop algorithms that transfer effectively across domains, which is critical for robust real-world deployment. We selected the ColoredMNIST dataset (Arjovsky et al., 2019), which introduces spurious correlations between color and class labels, creating a controlled yet challenging testbed for distribution shift. As the baseline, we adopted Empirical Risk Minimization (ERM) (Vapnik, 1998), the most fundamental training paradigm that provides a clear reference point for measuring generalization improvements. We constructed this task using the DomainBed repository (Gulrajani & Lopez-Paz, 2020), a widely used framework for domain generalization research. Agents are required to train on a source domain and are evaluated on a held-out target domain under distribution shift. The evaluation metric is out-of-domain accuracy, with the objective of maximizing generalization performance while maintaining in-domain accuracy.

Data Efficiency. This task measures the ability to enhance few-shot learning performance by optimizing decision rules under tight data constraints. We selected the Mini-ImageNet dataset (Vinyals et al., 2016), a standard few-shot learning benchmark whose fine-grained inter-class similarities and limited training examples per class present meaningful challenges for metric-based classification. As the baseline, we adopted Prototypical Networks (Snell et al., 2017), a classic and widely recognized metric-based approach that computes class prototypes in the embedding space. We constructed this task using the Easy-Few-Shot-Learning repository (Sicara, 2024). Agents must operate under a frozen backbone and propose improved algorithms for metric-based classification in the embedding space. The evaluation metric is few-shot classification accuracy, with the objective of improving the classifier’s ability to generalize to novel classes.

Representation Learning. This task tests the ability to learn generalizable and semantically meaningful features from unlabeled data through self-supervised learning. We selected the CIFAR-10 dataset (Krizhevsky et al., 2009), which provides a computationally efficient yet sufficiently challenging benchmark for evaluating learned representations. As the baseline, we adopted MoCo (Momentum Contrast) (He et al., 2020), a widely recognized and effective contrastive learning method that maintains a momentum-updated encoder for consistent representations. We constructed this task using the Lightly repository (Lightly-AI, 2025), a modular framework for self-supervised learning. Agents pretrain encoders on the CIFAR-10 training split without labels using a ResNet-18 backbone, and evaluation is conducted via linear probing accuracy with the encoder frozen. The objective is to improve representation quality such that a linear classifier trained on frozen features achieves higher Top-1 accuracy on the test split, while keeping parameter count and FLOPs comparable to the baseline.

Continual Learning. This task evaluates long-term adaptability in non-stationary environments and the ability to mitigate catastrophic forgetting when learning sequentially across multiple tasks. We selected the splitMNIST dataset (Deng, 2012) under the class-incremental learning scenario, where the model must learn five sequential tasks (each introducing two new digit classes) using a shared ten-way classification head—a setting that rigorously tests knowledge retention. As the baseline, we adopted Synaptic Intelligence (SI) (Zenke et al., 2017), an established regularization-based method that estimates parameter importance online to protect previously learned knowledge. We constructed this task using the Continual-Learning repository (van de Ven et al., 2022). The evaluation metric is average accuracy across all five tasks, with the objective of maximizing performance while avoiding unfair advantages in model size or computational budget.

Causality. This task assesses the capacity to reason about interventions and estimate treatment effects, which is essential for decision-making in high-stakes environments. We selected the IHDP (Infant Health and Development Program) dataset (Hill, 2011), a semi-synthetic benchmark built from a real randomized controlled trial with simulated treatment effects, enabling ground-truth evaluation of causal inference methods. As the baseline, we adopted DragonNet (Shi et al., 2019), an established neural network architecture designed for treatment effect estimation that jointly models outcome prediction and propensity scoring. We constructed this task using the CausalML repository (Chen et al., 2020). The evaluation metric is mean absolute error in estimating individual treatment effects, with the objective of improving the precision of causal effect estimation.

Robustness and Reliability. This task probes the ability to improve model robustness against adversarial data corruption while preserving performance on clean data. We constructed a poisoned MNIST dataset incorporating diverse backdoor attack methods, including edge-based triggers and various distributed attack patterns, to create a challenging defense scenario. As the baseline, we adopted dp-instahide (Borgnia et al., 2021), a privacy-preserving defense that mixes inputs with public data and applies differential privacy noise to hinder inversion and poisoning attacks. We constructed this task using the

Adversarial Robustness Toolbox (ART) repository (Nicolae et al., 2018). Agents are tasked with proposing defenses that reduce the effectiveness of poisoning attacks while maintaining high clean accuracy. The evaluation metric is a defense score defined as the harmonic mean of clean accuracy and resistance accuracy against backdoor attacks.

Privacy. This task is critical for assessing the ability to enhance privacy protections against membership inference attacks, which attempt to determine whether specific data points were used in model training. We selected the CIFAR-10 dataset (Krizhevsky et al., 2009) as the training corpus, providing a standard image classification setting for privacy evaluation. As the baseline, we adopted Wide-ResNet-28-2 (Zagoruyko & Komodakis, 2016), a widely used architecture that provides a representative target model for privacy auditing. We constructed this task using the PrivacyMeter repository (Murakonda & Shokri, 2020), which implements both standard Membership Inference Attacks (MIA) and Robust MIA (RMIA) that leverages reference models and population data for more rigorous privacy assessment. The evaluation metric is the attack AUC, with the objective of driving it toward 0.5 (random guessing) while maintaining classification accuracy, thereby reducing information leakage about training data membership.

Fairness and Bias. This task measures the ability to balance equitable outcomes across demographic groups with overall model utility in classification settings. We selected the COMPAS dataset (Angwin et al., 2016), a recidivism prediction benchmark that presents meaningful fairness challenges due to documented disparities across protected attributes such as race and sex. As the baseline, we adopted Adversarial Debiasing, which employs a classifier-adversary architecture to learn fair representations by preventing the adversary from predicting sensitive attributes. We constructed this task using the AIF360 repository (Bellamy et al., 2018). The evaluation metric is absolute average odds difference, with the objective of minimizing this disparity toward parity (zero) while maintaining or improving classification accuracy across protected subgroups.

### B. Case Analysis

#### B.1. Worst Case Analysis

Modification Distribution by Agents As shown in Tab. 5, TheAIScientist strongly favors algorithmic modifications (56.2%) with zero implementation errors, indicating robust execution. AIDE presents a more balanced modification distribution but experiences higher implementation and execution issues (31.2%). Claude Code shows elevated parameter configuration rates (37.5%), equivalent to its algorithmic modification rate.

Modification Distribution by LLMs As shown in Tab. 6, Gemini-2.5-Pro strongly favors algorithmic modifications (68.8%), preferring core algorithm changes; GPT-5 primarily focuses on structural modifications (50.0%) with emphasis on architecture adjustments; while Opus-4.1 maintains balanced distribution between parameter configuration and algorithmic modifications.

#### B.2. Best Case Analysis

Modification Distribution by Agents As shown in Tab. 7, TheAIScientist shows strong algorithmic innovation focus, predominantly targeting core algorithmic logic; AIDE combines strong algorithmic innovation with hyperparameter tuning; while Claude Code uniquely favors architectural modifications, including layer additions and normalization adjustments.

Modification Distribution by LLMs As shown in Tab. 8, GPT-5 exclusively prioritizes algorithmic modifications (100.0%); Gemini-2.5-Pro exhibits the most diversified approach, uniquely incorporating substantial data augmentation efforts (28.6%); while Opus-4.1 balances attention between architectural and algorithmic modifications.

- Table 5. Worst Case Modification Distribution by Agents

Modification Type TheAIScientist AIDE Claude Code

Algorithmic Modification 56.2% (9) 37.5% (6) 37.5% (3) Structural Modification 31.2% (5) 25.0% (4) 12.5% (1) Parameter Configuration 12.5% (2) 6.2% (1) 37.5% (3) Implementation & Execution 0.0% (0) 31.2% (5) 12.5% (1)

- Table 6. Worst Case Modification Distribution by LLMs

Modification Type GPT-5 Gemini-2.5-pro Opus-4.1 Algorithmic Modification 25.0% (4) 68.8% (11) 37.5% (3) Structural Modification 50.0% (8) 6.2% (1) 12.5% (1) Parameter Configuration 6.2% (1) 12.5% (2) 37.5% (3) Implementation & Execution 18.8% (3) 12.5% (2) 12.5% (1)

- Table 7. Best Case Modification Distribution by Agents

Modification Type TheAIScientist AIDE Claude Code

Algorithmic Modification 75.0% (12) 83.3% (10) 37.5% (3) Structural Modification 12.5% (2) 0.0% (0) 37.5% (3) Data Processing 12.5% (2) 16.7% (2) 0.0% (0) Parameter Configuration 0.0% (0) 0.0% (0) 25.0% (2)

- Table 8. Best Case Modification Distribution by LLMs

Modification Type GPT-5 Gemini-2.5-pro Opus-4.1 Algorithmic Modification 14 (100.0%) 8 (57.1%) 3 (37.5%) Structural Modification 0 (0.0%) 2 (14.3%) 3 (37.5%) Data Processing 0 (0.0%) 4 (28.6%) 0 (0.0%) Parameter Configuration 0 (0.0%) 0 (0.0%) 2 (25.0%)

### C. Protecting Evaluation Integrity

A crucial aspect of our implementation involved protecting evaluation files from inadvertent modification by the agents. We employed agent-specific protection strategies: For TheAIScientist, we explicitly instructed the agent through prompting to avoid modifying evaluation files and implemented a systematic refresh mechanism that restores evaluation files before each evaluation cycle. For AIDE, protection is inherently ensured by its single-file modification constraint, which guarantees that when the target code and evaluation code are separated into different files, the evaluation files remain naturally protected. For Claude Code, we implemented a two-pronged approach in which the evaluation files were first restricted to read and execute permissions before running the agent, and then the --disallowedTools argument was employed to explicitly prevent permission modification operations during execution.

With hard task

FML-bench: Benchmarking Machine Learning Agents for Scientific Research

### D. Additional Results

[Figure 10]

[Figure 11]

(a) Generalization (↑) (b) Data Efficiency (↑)

[Figure 12]

[Figure 13]

(c) Representation Learning (↑) (d) Continual Learning (↑)

[Figure 14]

[Figure 15]

(e) Causality (↓)

(f) Robustness and Reliability (↑)

[Figure 16]

[Figure 17]

(g) Privacy (↓) (h) Fairness and Bias (↓)

TheAIScientist (GPT-5)

[Figure 18]

[Figure 19]

TheAIScientist (Gemini-2.5-Pro) Claude Code AIDE (GPT-5) Baseline

[Figure 20]

[Figure 21]

AIDE (Gemini-2.5-Pro)

[Figure 22]

[Figure 23]

Figure 3. Agents’ performance improvement curves across 8 tasks.

This section provides a comprehensive comparative analysis of automated AI research systems, focusing on both their performance and operational efficiency. We evaluate TheAIScientist, AIDE, and Claude Code across eight core machine

learning research problems, considering not only their final outcomes but also their operational behavior over time.

Additional analysis results covers multiple key dimensions: detailed exploration diversity (Tab. 9), operational reliability measured by step success rate (Tab. 10) and step completion rate (Tab. 11), computational efficiency in terms of total token usage (Tab. 12) and step-wise token usage (Tab. 13), runtime efficiency in terms of total time (Tab. 14) and step-wise time (Tab. 15). Together, these results provide detailed insights into the trade-offs between different agentic research designs, illustrating how they influence both the quality of research outcomes and the efficiency of resource utilization.

- Table 9. Exploration Diversity Comparison of Best Performance Round. G2.5-Pro denotes Gemini-2.5-Pro. Bold numbers indicate the best (highest) performance in each row.

ML Problems

TheAIScientist∗ AIDE∗ Claude Code GPT-5 G2.5-Pro GPT-5 G2.5-Pro Opus-4.1

Generalization 39.32 18.60 4.36 48.23 19.50 Data Eff. 28.18 20.19 18.03 10.92 10.08 Rep. Learn. 15.28 11.55 18.91 9.71 5.30 Cont. Learn. 18.86 24.83 36.72 18.80 2.71 Causality 24.03 21.37 38.61 11.65 6.43 Robust. & Rel. 45.44 24.04 43.03 21.23 16.19 Privacy 29.66 21.17 33.16 13.67 5.97 Fair. & Bias 26.88 25.14 35.97 13.10 3.84

Average ± Std 28.46±9.97 20.86±4.42 28.60±13.32 18.41±12.68 8.75±6.07

- Table 10. Step Success Rate Comparison. G2.5-Pro denotes Gemini-2.5-Pro. Bold numbers indicate the best (highest) performance in each row.

ML Problems

TheAIScientist∗ AIDE∗ Claude Code GPT-5 G2.5-Pro GPT-5 G2.5-Pro Opus-4.1

Generalization 0.77 0.99 0.50 0.70 1.00 Data Eff. 0.96 0.65 0.23 0.29 0.94 Rep. Learn. 0.83 0.84 0.80 0.60 0.55 Cont. Learn. 0.96 0.94 0.82 0.92 1.00 Causality 0.98 0.89 0.88 0.98 1.00 Robust. & Rel. 0.98 0.89 0.87 0.79 1.00 Privacy 0.97 0.95 0.87 0.96 1.00 Fair. & Bias 0.18 0.21 0.18 0.04 0.13

Average ± Std 0.83±0.27 0.80±0.26 0.64±0.30 0.66±0.34 0.83±0.32

- Table 11. Step Completion Rate (SCR) Comparison. G2.5-Pro denotes Gemini-2.5-Pro. Bold numbers indicate the best (highest) performance in each row.

TheAIScientist∗ AIDE∗ Claude Code GPT-5 G2.5-Pro GPT-5 G2.5-Pro Opus-4.1

ML Problems

Generalization 1.00 1.00 0.02 0.36 0.03 Data Eff. 1.00 1.00 0.30 1.00 0.16 Rep. Learn. 1.00 1.00 0.45 1.00 0.11 Cont. Learn. 1.00 1.00 0.28 0.70 0.02 Causality 1.00 1.00 1.00 0.52 0.11 Robust. & Rel. 1.00 1.00 1.00 0.56 0.01 Privacy 1.00 1.00 0.70 0.69 0.03 Fair. & Bias 1.00 1.00 0.39 0.7 0.08

Average ± Std 1.00±0.00 1.00±0.00 0.54±0.38 0.69±0.24 0.07±0.06

- Table 12. Total Token Consumption Comparison (in millions). G2.5-Pro denotes Gemini-2.5-Pro. Bold numbers indicate the best (lowest) performance in each row.

ML Problems

TheAIScientist∗ Claude Code GPT-5 G2.5-Pro Opus-4.1

Generalization 9.34 9.88 5.98 Data Eff. 3.09 2.55 7.98 Rep. Learn. 4.69 3.99 8.89 Cont. Learn. 7.81 7.74 2.99 Causality 3.72 3.40 10.86 Robust. & Rel. 7.62 7.04 13.36 Privacy 6.64 4.45 13.88 Fair. & Bias 5.41 4.58 8.53

Average ± Std 6.04±2.18 5.45±2.51 9.06±3.64

- Table 13. Step Token Consumption Comparison (in millions). G2.5-Pro denotes Gemini-2.5-Pro. Bold numbers indicate the best (lowest) performance in each row.

ML Problems

TheAIScientist∗ Claude Code GPT-5 G2.5-Pro Opus-4.1

Generalization 0.09 0.10 1.99 Data Eff. 0.03 0.03 0.50 Rep. Learn. 0.05 0.04 0.81 Cont. Learn. 0.08 0.08 1.49 Causality 0.04 0.03 0.99 Robust. & Rel. 0.08 0.07 13.36 Privacy 0.07 0.04 4.63 Fair. & Bias 0.05 0.05 1.07

Average ± Std 0.06±0.02 0.05±0.03 3.10±4.34

- Table 14. Total Time Consumption Comparison (in hours). G2.5-Pro denotes Gemini-2.5-Pro. Bold numbers indicate the best (lowest) performance in each row.

TheAIScientist∗ AIDE∗ Claude Code GPT-5 G2.5-Pro GPT-5 G2.5-Pro Opus-4.1

ML Problems

Generalization 9.91 17.42 0.20 6.07 0.81 Data Eff. 4.47 4.33 1.02 2.40 0.47 Rep. Learn. 7.50 23.96 4.45 14.49 5.56 Cont. Learn. 14.68 15.17 3.70 7.08 0.28 Causality 4.97 3.51 4.50 3.43 0.40 Robust. & Rel. 19.40 6.60 22.02 17.84 0.32 Privacy 9.24 20.03 2.43 21.04 1.29 Fair. & Bias 5.20 2.87 5.32 3.60 0.30

Average ± Std 9.42±5.25 11.74±8.37 5.46±6.92 9.49±7.24 1.18±1.80

- Table 15. Step Time Consumption Comparison (in minutes). G2.5-Pro denotes Gemini-2.5-Pro. Bold numbers indicate the best (lowest) performance in each row.

TheAIScientist∗ AIDE∗ Claude Code GPT-5 G2.5-Pro GPT-5 G2.5-Pro Opus-4.1

ML Problems

Generalization 5.95 10.43 5.97 9.83 16.13 Data Eff. 2.68 2.60 2.03 1.43 1.77 Rep. Learn. 33.28 43.15 37.92 22.87 30.30 Cont. Learn. 8.80 9.08 7.93 5.98 8.33 Causality 2.98 2.10 2.68 3.88 2.20 Robust. & Rel. 11.63 18.37 13.08 18.78 19.20 Privacy 19.93 26.42 22.63 18.03 25.87 Fair. & Bias 3.12 1.72 8.18 3.05 2.25

Average ± Std 11.05±10.71 14.23±14.55 12.55±12.17 10.48±8.29 13.26±11.29

### E. Prompts Details

This section presents the detailed prompt specifications that form the foundation of our automatic research agent framework. The prompts serve as the primary interface between human researchers and AI agents, translating high-level research objectives into actionable instructions that can guide systematic scientific inquiry across diverse machine learning domains. The prompt design philosophy centers on creating a structured yet flexible research environment that balances autonomy with scientific rigor. Rather than providing overly prescriptive instructions that limit creative exploration, these prompts establish clear boundaries, evaluation criteria, and operational constraints while encouraging the agent to develop and test novel hypotheses within established research paradigms.

This section is organized into two complementary components. First, we present the Task Description Prompts that define specific research challenges across 8 fundamental areas of machine learning, each grounded in established benchmarks and methodologies. These prompts simulate realistic research scenarios where an AI agent must navigate complex technical requirements while pursuing meaningful improvements to existing methods.

Second, we detail the Automatic Research Agent Framework that governs how agents interact with research codebases to conduct iterative experimentation. This operational framework transforms the conceptual research challenges into executable workflows, ensuring that agent behavior follows sound scientific methodology while maintaining reproducibility and experimental integrity. Together, these prompt specifications create a comprehensive research environment where automatic agents can contribute meaningfully to advancing machine learning across multiple disciplines, providing both the research contexts and the methodological framework necessary for systematic scientific progress.

Task Description Prompts The following task descriptions establish comprehensive research contexts spanning 8 fundamental areas of machine learning. Each prompt follows a structured format that defines: 1) the researcher’s identity and expertise, 2) the specific technical setup including datasets and baseline methods, 3) clear optimization objectives and constraints, and 4) fairness criteria to ensure meaningful comparisons.

These prompts span a wide range of machine learning challenges (e.g., generalization, data efficiency, privacy, fairness, and robustness). Each task is anchored in established benchmarks and frameworks, such as DomainBed for domain generalization, EasyFSL for few-shot learning, and AIF360 for fairness-aware learning, providing realistic experimental environments that closely mirror actual research workflows.

The prompts are carefully crafted to encourage both incremental improvements to existing methods and the exploration of novel algorithmic approaches, while maintaining scientific rigor through controlled experimental conditions. This design enables systematic evaluation of how automatic agents can contribute to advancing the state-of-the-art across multiple ML disciplines simultaneously.

#### Generalization

System: You are an ambitious AI PhD student focused on improving the generalization performance of machine learning methods using the DomainBed benchmark.

Task Description: You are working with DomainBed’s ERM (Empirical Risk Minimization) method as the baseline on ColoredMNIST to evaluate generalization under distribution shifts. Your goal is to enhance test-time domain generalization accuracy beyond standard ERM. You should improve the algorithm based on ERM, but you may also propose entirely new algorithms if they can better support cross-domain generalization. You are also allowed to refine the backbone model, as long as your modifications are fair compared to the original architecture. The priority is to improve the average accuracy on unseen test domains while maintaining accuracy on in-domain tests, along with ensuring efficiency and low complexity.

#### Data Efficiency

System: You are an ambitious AI PhD student focused on data-efficient learning, specializing in few-shot learning and meta-learning.

Task Description: You are working with the EasyFSL framework to enhance the FewShotClassifier on the Mini-ImageNet dataset. The Mini-ImageNet dataset presents a challenging few-shot learning scenario due to its fine-grained inter-class similarities and limited training examples per class. Your goal is to improve the classifier’s ability to generalize to novel classes.

#### Representation Learning

System: You are an ambitious AI PhD student focused on improving representation learning on CIFAR-10 using the Lightly self-supervised learning framework.

Task Description: You are working with Lightly’s MoCo baseline on CIFAR-10, evaluated strictly by linear probing Top-1 accuracy. Your goal is to improve representation learning at pretrain stage to improve linear-probe accuracy on the CIFAR-10 test set beyond standard MoCo as much as you can under the same compute and data (no external data). You may modify MoCo or propose new self-supervised methods if they can yield better representations, as long as your modifications are fair compared to the original architecture. You are also allowed to refine the ResNet-18 backbone as long as parameter count and FLOPs remain comparable to the baseline. Pretrain on the CIFAR-10 train split without labels, fit the linear classifier on the same train split, and report Top-1 on the test split with priority on improving representation learning performance.

#### Continual Learning

System: You are an ambitious AI PhD student focused on improving continual learning based on Synaptic Intelligence (SI) on splitMNIST under the class-incremental scenario.

Task Description: You are working with the Continual-Learning repo’s SI baseline to improve accuracy and reduce forgetting on splitMNIST (5 tasks × 2 classes, single-head over 10 classes). Your goal is to improve average accuracy over all 5 contexts on splitMNIST without unfair model size or compute advantages. You should improve SI method, but are also allowed to add lightweight fair components , or propose new methods, as long as your modifications are fair (stay within fairness computation budgets). The priority is to improve the average accuracy.

#### Causality

System: You are an ambitious AI PhD student focused on advancing machine learning for causal inference, reasoning, and interpretable modeling.

Task Description: You are working with the Dragonnet framework to estimate individual treatment effects (ITEs) in IHDP. The Infant Health and Development Program (IHDP) is a semi-synthetic benchmark dataset commonly used in causal inference, built from a real randomized trial with simulated treatment effects to enable ground-truth evaluation. Your goal is to improve the precision of treatment effect estimation across IHDP.

#### Robustness and Reliability

System: You are an ambitious AI PhD student focused on improving robust learning under data poisoning and privacy constraints.

Task Description: You are given the Adversarial Robustness Toolbox (ART) codebase with a focus on the dp instahide defense. dp instahide mixes inputs with public data and applies differential privacy noise to hinder inversion and poisoning. While designed for privacy-preserving training, its structure offers headroom to harden against both clean-label and trigger/backdoor poisons. Your goal is to improve defense performance against diverse poisoning attacks while maintaining high clean accuracy. You may tune dp instahide, compose it with other defenses, or propose a new method if it outperforms baselines.

#### Privacy

System: You are an ambitious AI PhD student focused on improving model privacy and security against membership inference attacks.

Task Description: You are working with PrivacyMeter’s MIA (for information leakage through training points) and Robust MIA (RMIA, which refines the Likelihood Ratio Test with a tighter null hypothesis and leverages reference models and population data) to evaluate and reduce the model’s privacy risk. Your goal is to drive the auditor’s AUC toward 0.5 and keep TPR@0.1%FPR and TPR@0.0%FPR near zero while preserving task accuracy. Focus only on defense-side strategies rather than modifying the attack algorithms.

#### Fairness and Bias

System: You are an ambitious AI PhD student focused on improving fairness-aware learning with AIF360’s Adversarial Debiasing on the COMPAS dataset.

Task Description: You are working with AIF360’s Adversarial Debiasing (classifier–adversary) as the baseline on the COMPAS dataset to evaluate the fairness–accuracy trade-off. Your goal is to minimize absolute Average Odds Difference toward parity (=0) while maintaining or improving Balanced Accuracy on held-out test splits and across protected subgroups (e.g., sex/race). You should enhance the baseline Adversarial Debiasing algorithm, but you may also propose entirely new fairness methods if they better support reduced absolute Average Odds Difference without sacrificing Balanced Accuracy. You are allowed to refine the classifier and adversary networks and the training pipeline, provided comparisons remain fair to the original setup (similar capacity, training budget, and data access). The priority is minimizing absolute Average Odds Difference while preserving or improving Balanced Accuracy.

Prompting Claude Code as an Automatic Research Agent The following comprehensive prompt specification defines how an AI agent operates within established research codebases to achieve meaningful scientific progress. Unlike traditional one-shot code generation, this framework establishes a iterative research loop that mirrors authentic research methodology: hypothesis formation, implementation, experimentation, analysis, and refinement.

Your Role You are an autonomous coding agent that:

- • understands the task,
- • proposes a concrete idea/plan/solution,
- • edits the code (respecting read-only constraints),
- • executes a fixed command list,
- • handles errors by diagnosing and fixing them,
- • records each step’s modifications and results, and
- • iterates until the iteration limit is reached.

Repository Access You are given starter files, STARTER FILE PATHS, and may read other files as needed to complete the task. Hard constraint: Do not modify any file whose path matches READONLY PATHS. If a necessary change would touch a read-only file, propose an alternative (e.g., wrapper, config flag, adapter module) instead.

Loop Initialization Initialize:

- • count = 0
- • Record a snapshot/baseline of the original code (the repository state before any of your edits). All ”modifications” below are defined relative to this original baseline.
- • Read original baseline results for reference. The results are provided in ORIGINAL BASELINE RESULTS PATH.

Step 1-3: Understanding and Planning

#### Step 1 — Understand the task

- • Read the repo and TASK DESCRIPTION.

- • If helpful, quickly inventory key entry points, configs, data paths, and any training/eval scripts.

#### Step 2 — Generate a plan

• Produce a brief idea/plan/solution describing what you will change and why.

#### Step 3 — Modify the code

- • Implement your plan with minimal, focused edits.
- • Respect READONLY PATHS at all times (no renames, moves, or edits under those paths).

- • Keep changes atomic and well-commented.

Step 4-6: Execution and Error Handling

#### Step 4 — Execute commands

- • Run every command in COMMAND LIST sequentially.

- • Capture stdout/stderr and exit codes for each command.
- • After the command list completes (whether fully or interrupted by an error), do: count += 1.

##### Step 5 — Error handling If any command raised an exception or returned a non-zero exit code:

- • Diagnose the exception concisely (root cause + where it occurred).
- • Propose a specific fix.
- • Apply the fix by editing the code (still respecting READONLY PATHS).

- • Proceed to the next iteration (go back to Step 4 for another execution after modifications).

#### Step 6 — Iteration limit

• If count > MAX ITERS, stop and produce a final summary.

Step 7-10: Backup and Results Management

##### Step 7 — Per-step backup (always) For each iteration (each time you execute the command list), create a directory:

./ agent runs/step {COUNT}/ Store in that directory:

- • modified/ → only the files that differ from the original baseline (preserve their relative paths).
- • logs/ → command outputs (one file per command, including exit codes).

##### Step 8 — Successful run artifacts If all commands in COMMAND LIST completed successfully:

- • In ./ agent runs/step {COUNT}/results/, copy: ${RESULT DIR}/final info.json

- • Confirm that final info.json exists in the step results.

#### Step 9 — Read & reset results directory

- • Read and summarize the key contents of final info.json for guidance.

- • Then delete the entire RESULT DIR to avoid conflicts with future iterations.

#### Step 10 — Improve the plan

- • Based on the results, your current idea/plan/solution, and the code modifications so far: Generate a new idea/plan/solution to further improve the outcome.
- • Continue the loop, unless the iteration limit has been reached.

Additional Rules & Conventions Command semantics: Treat any non-zero exit code as an error. If a command expects env variables or paths, set them explicitly and document them in the logs. Diff discipline: When storing modified/ files for a step, include only files that differ from the original baseline (not from the previous step). Execution discipline:

- • Execute COMMAND LIST verbatim: do not change the order, arguments, flags, prefixes (e.g., env vars), or wrap the commands.

- • Resolve failures by modifying code or configuration (outside READONLY PATHS) instead of altering the commands.

#### Other rules:

- • Atomic changes: Prefer small, testable edits.
- • Evaluation integrity: Never alter evaluation logic, datasets, or scripts inside READONLY PATHS.

- • Idempotence: If a prior step succeeded, avoid regressing it.

Important constraint: Under no circumstances may you halt while the current step count ≤ MAX ITERS; you must continue the modify → execute COMMAND LIST → diagnose/fix loop.

Optimization Directions

Optimization directions: The global setting, OPTIMIZATION DIRECTION, defines the default direction for all metrics and accepts either ”higher” or ”lower”. This global direction is applied to all metrics unless explicitly overridden. For more granular control, PER METRIC DIRECTION provides a way to override the global setting by specifying a mapping of individual metric names to their desired optimization direction.

Optimization goal filtering: If optimization target metrics (TARGET METRICS) and dataset (TARGET DATASETS) names are provided, treat them as strict optimization goals: focus exclusively on improving the specified metrics on the specified datasets, and ignore all other metrics and datasets.

Runtime Logging Rule: You must log the start and end time of the entire process. Record wall-clock timestamps when you begin and exit the loop. Save timestamps to ./ agent runs/process time log.txt with format:

start_time: 2025-08-14 13:04:22 end_time: 2025-08-14 14:37:55 duration_seconds: 5573

CRITICAL EXECUTION REQUIREMENT

For ALL commands in COMMAND LIST that are expected to take more than 5 minutes (especially training commands with epochs > 10), you MUST:

- 1. ALWAYS use run in background=True when executing these commands with the Bash tool

- 2. Monitor the background process using BashOutput tool with the returned bash id

- 3. Wait for completion by periodically checking BashOutput until the process finishes
- 4. Only proceed to the next command after confirming the previous background process has completed successfully

Example execution pattern: result = Bash(

command="python main.py ... --num_epochs 100 ...", run_in_background=True, # MANDATORY for long commands description="Training in background"

) shell_id = result.split(

"Background shell started with ID: " )[1].split("\n")[0] # Monitor until completion while True:

output = BashOutput(bash_id=shell_id) if "still running" not in output:

break time.sleep(30) # Check every 30 seconds

FAILURE TO USE BACKGROUND EXECUTION FOR LONG-RUNNING COMMANDS WILL BE CONSIDERED A CRITICAL ERROR. Specifically for this task, the training command with 100 epochs MUST be run in background.

Reporting After each iteration, output a short report with:

- • count
- • Idea/Plan/Solution (current) — 3–6 bullet points
- • Changes Made — list of files edited with 1-line rationale each
- • Command Results — success/failure per command with exit code
- • If success: brief summary of final info.json key metrics

- • Next Steps — what you’ll try next (or stop if count > MAX ITERS)

Begin with Template Variables

TASK DESCRIPTION: “” STARTER FILE PATHS: [ ] READONLY PATHS: [“”] ORIGINAL BASELINE RESULTS PATH: “” TARGET METRICS: [“”] TARGET DATASETS: [“”] OPTIMIZATION DIRECTION: “” PER METRIC DIRECTION = { } COMMAND LIST: [ ] MAX ITERS: RESULT DIR: “”

