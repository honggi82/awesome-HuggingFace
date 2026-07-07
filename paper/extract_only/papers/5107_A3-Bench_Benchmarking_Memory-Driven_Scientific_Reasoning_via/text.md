# arXiv:2601.09274v1[cs.AI]14Jan2026

## A3-Bench: Benchmarking Memory-Driven Scientific Reasoning via Anchor and Attractor Activation

Jian Zhang1, Yu He1, Zhiyuan Wang1, Zhangqi Wang1, Kai He2, Fangzhi Xu1, Qika Lin2, Jun Liu1* 1Xi’an Jiaotong University 2National University of Singapore zhangjian062422@stu.xjtu.edu.cn, liukeen@xjtu.edu.cn

##### Abstract

Scientific reasoning relies not only on logical inference but also on activating prior knowledge and experiential structures. Memory can efficiently reuse knowledge and enhance reasoning consistency and stability. However, existing benchmarks mainly evaluate final answers or step-by-step coherence, overlooking the memory-driven mechanisms that underlie human reasoning, which involves activating anchors and attractors, then integrating them into multi-step inference. To address this gap, we propose A3-Bench 1, a benchmark to evaluate scientific reasoning through dual-scale memory-driven activation, grounded in Anchor and Attractor Activation. First, we annotate 2,198 science reasoning problems across domains using the SAPM process(subject, anchor & attractor, problem, and memory developing). Second, we introduce a dual-scale memory evaluation framework utilizing anchors and attractors, along with the AAUI (Anchor– Attractor Utilization Index) metric to measure memory activation rates. Finally, through experiments with various base models and paradigms, we validate A3-Bench and analyze how memory activation impacts reasoning performance, providing insights into memorydriven scientific reasoning.

##### 1 Introduction

Scientific reasoning tasks (Zhang et al., 2026b, 2025a), covering disciplines like math, physics, and chemistry, are essential for evaluating the ability of Large Language Models (LLMs) to integrate complex cognitive operations. Unlike traditional language tasks, scientific reasoning requires not only knowledge access but also the construction of reasoning trajectories, dynamic strategy adjustment, and validation of final results (Zhang et al., 2025b). For effective scientific reasoning, models need not

*Corresponding author 1https://a3-bench.github.io/

[Figure 1]

Figure 1: Comparison of reasoning paths on OlympiadBench. Activating anchors and attractors corrects the derivation path relative to no memory.

only to reason with available knowledge but also to incorporate key memory patterns. As shown in Figure 1, an example from OlympiadBench (He

- et al., 2024), GPT-5 (Leon, 2025) without memory fails to consider the kinetic energy theorem, leading to an incorrect reasoning result. However, when memory mechanisms such as the kinetic energy theorem, belt acceleration, and power scenarios are incorporated, the reasoning becomes correct. This illustrates the critical role of memory in enhancing reasoning accuracy and reliability.

Existing memory-driven works (Wang et al., 2024a) primarily offer advantages such as memory storage and fast retrieval, efficient knowledge reuse, and reasoning consistency (Xie et al., 2024; Cui et al., 2024) and stability (Liu et al., 2025; Du

- et al., 2025). However, current scientific reasoning benchmarks primarily emphasize final-answer correctness and process consistency, without directly evaluating memory activation ability. As a result, they do not reveal whether failures arise from flawed logical inference or from inadequate retrieval and activation of the necessary memory

[Figure 2]

- Figure 2: Performance and token analysis across ten LLMs and three memory paradigms. The three color-coded groups represent the experimental paradigms: vanilla, anchors & attractors, and annotated anchors & attractors.

during reasoning.

Human scientific reasoning is closely tied to how memory is organized and accessed. Memory is hierarchically structured, ranging from concrete experiences to abstract schemas (Bein and Niv, 2025), and relevant knowledge can be selectively activated by contextual cues during problem solving (Liu et al., 2012). These properties motivate the construction of benchmarks that align with human memory mechanisms. Such datasets should explicitly represent reusable knowledge units and structured, experience-based templates, and should require context-dependent activation across multiple reasoning steps. Such a benchmark enables finegrained evaluation of whether models precisely activate the appropriate knowledge and templates at the appropriate time during reasoning, and offers actionable signals to guide the development of more reliable, memory-driven large language models.

To this end, we introduce A3-Bench, a benchmark grounded in Anchor and Attractor Activation and designed to evaluate memory-driven scientific reasoning. Specifically, first, inspired by hierarchical human memory (Bein and Niv, 2025), we model scientific reasoning memory at two scales: anchors (foundational knowledge units) and attractors (experience-based templates). Using the SAPM process (subject, anchor & attractor, problem, and memory developing), we annotate 2,198 problems across domains and map each question to its anchor–attractor set. Second, motivated by context-dependent activation in human episodic memory (Liu et al., 2012), we introduce a dual-scale memory evaluation framework that leverages anchors and attractors, and propose the AAUI (Anchor–Attractor Utilization Index) metric to quantify memory activation rates. Third, we conduct experiments on A3-Bench across base models

and paradigms, validating its ability to evaluate anchor–attractor memory activation and utilization during reasoning. As shown in Figure 2, this enhancement improves accuracy while keeping token costs controllable, supporting cognitively aligned evaluation and model development. Our main contributions are as follows:

- • We create a dataset of 2,198 scientific reasoning problems via the SAPM process. Each instance is annotated with dual-scale memory signals, anchors and attractors, reflecting hierarchical human memory across domains.
- • A3-Bench is proposed as the first benchmark for memory-driven scientific reasoning. We further propose the AAUI metric, which quantifies memory activation rates by leveraging human-like contextdependent episodic recall.
- • Experiments validate A3-Bench and show how memory activation shapes multi-step reasoning. This enables fine-grained evaluation of precise memory activation in LLM inference and supports reliable model development.

##### 2 Preliminaries

This section introduces the foundational concepts of memory-driven scientific reasoning: anchor and attractor, memory activation, and memoryaugmented reasoning, which form the theoretical basis for how memory structures guide reasoning.

Definition 1: Anchor and Attractor. In reasoning, the anchor constrains the initial state and focus the system on relevant knowledge, while the attractor represents knowledge structures that guide reasoning along specific paths. Together, the activated anchors and attractors (Zhou and Knierim, 2025; Siegenthaler et al., 2025) form the state space of the Attractor Basin, unifying these two memory types and describing their collaborative role in reasoning.

[Figure 3]

- Figure 3: The four-step annotation process SAPM. First, subject benchmarking defines subdomains for each discipline. Next, experts develop anchors and attractors for each subdomain and define their relations. Then, a new set of questions is refined from existing datasets. Finally, memory mapping associates questions with relevant anchors and attractors.

Let Z ⊆ Rd be a neural or semantic state space, and let f : Z → Z be the dynamical update operator. A state z∗ is an attractor if:

f(t)(z0) = z∗. (1)

lim

t→∞

The basin of attraction associated with z∗ is defined as:

B(z∗) = z0 ∈ Z | lim

f(t)(z0) = z∗ . (2)

t→∞

- Definition 2: Memory Activation. Memory activation (Friston, 2010) is modeled as minimizing the formula:

F(z;x) = −log p(x | z)+DKL q(z)∥p(z) , (3)

where x denotes the input query, q(z) the posterior representation, and p(z) the prior knowledge distribution. The internal state evolves according to gradient descent:

zt+1 = zt − η∇zF(zt;x), (4)

driving the system toward an attractor z∗ that best explains the input, corresponding to the activation of memory structures.

- Definition 3: Memory-Augmented Reasoning. In memory-augmented reasoning (Ko et al., 2024), given an input query s0, we map it to an initial internal state z0 = ϕ(s0) and identify a set of candidate

attractors: A = {z∗k}Kk=1. Memory activation is formalized as a minimization problem:

z∗ = arg min z∗k∈A

F(z∗k;z0). (5)

Reasoning steps are then guided by the evolving internal representation:

si ∼ πθ(· | s0,s≤i−1,zt), (6)

and the final output is expressed as: sn = Ψ(z∗,s0), representing the stable inference outcome after the system settles within the attractor basin associated with the activated knowledge structure. The proof of memory activation and memoryaugmented reasoning is shown in App. A.

##### 3 The A3-Bench Dataset

This section introduces SAPM, a four-step annotation process for the A3-Bench dataset (Figure 3). § 3.1 describes subject benchmarking and hierarchical subject standards; § 3.2 develops Anchors and Attractors; § 3.3 reconstructs problems from existing datasets; and § 3.4 performs Memory Mapping by linking questions to anchor–attractor sets. An example data point is shown in Figure 4, and detailed guidance is provided in App. B.

###### 3.1 Subject Benchmarking

Scientific reasoning spans math, physics, and chemistry. For each discipline, we reference authoritative classification systems: math follows the American Mathematical Society (AMS) (Dunne and Hulek, 2020a), physics adopts international standards from the physics community (IP) (Smith, 2020), and chemistry follows the International Union of Pure and Applied Chemistry (IUPAC) (Heller et al., 2013). We then fine-tune and

[Figure 4]

Figure 4: A piece of math problem in A3-Bench.

integrate these systems, resulting in 8 subdomains for math, 5 for physics, and 5 for chemistry. Details are provided in App. C.

###### 3.2 Anchors & Attractors Developing

For each subdomain, we invited three subject experts to label anchors and attractors based on the established subdomains, following our memory development guidelines. Anchors include concepts, principles, and formulas, which set initial conditions and guide reasoning. Attractors, including abstract schemas and specific exemplars, ensure reasoning unfolds within a predefined framework.

###### 3.3 Problem Reconstructing

In this section, we construct a new problem set from existing datasets in the following stages:

Examination Given the varying difficulty levels and task focus of the four datasets MathVista (Lu

- et al., 2023), OlympiadBench, EMMA (Hao et al., 2025), and Humanity’s Last Exam (Phan et al., 2025), we begin by examining the questions Q =

{q1,q2,...,qn}. Each question qi is answered by three LLMs: GPT-5, Deepseek-V3.2 (Liu et al., 2024), and Qwen-30B (Yang et al., 2025). The diversity in model parameters, capabilities, and balance between open-source and closed-source models ensures varied responses. A question is passed to the next phase if any model answers incorrectly, and discarded if all models answer correctly.

Refinement For questions moving to the second stage, three LLMs M perform a cross-analysis. Each model Mj solves the problem, while the other two models Mk and Ml evaluate its reasoning and identify errors. Let Aj(qi) be the answer from model Mj for question qi. The evaluation

[Figure 5]

Figure 5: Schema of the A3-Bench dataset and its usage within a HybridRAG framework. (a) memory twinneedle activator. (b) context fabric composer.

function Ek,l represents the errors detected:

Ek,l(Aj,qi) =

m

Errore(Aj,qi), (7)

e=1

where Errore(Aj,qi) returns indictor if error e is found. After this, three subject experts revise the

question qi to qi′, integrating multi-step reasoning. The revision function is qi′ = f(qi,Ek,l,R), where R represents the reasoning steps. A standard an-

swer A′(qi′) is provided for each revised question. Assessment Once revisions are complete, the new problems are evaluated by the three LLMs. The same three models M answer each question qi 10 times, resulting in 30 answers per question. The difficulty is based on the number of correct answers. Let C = Correct(Aj,qi) return 1 for a correct answer and 0 for an incorrect one. A question is classified as "Easy" if 15 ≤ j C ≤ 30, "Medium" if 5 ≤ j C ≤ 14, and "Difficult" if 0 ≤ j C ≤ 4. The overall difficulty is determined by the majority of correct answers.

3.4 Memory Mapping Based on the question pool and anchor–attractor library, the process proceeds as follows:

First, three LLMs M recommend a subdomain for each question qi. If at least two models agree, the subdomain is finalized through a voting mechanism: Vote(M1(qi),M2(qi),M3(qi)).

Next, human experts review the recommendations and manually annotate relevant anchors and

Math Physics Chemistry

Method

Avg. AAUI Tokens

Easy Medium Hard Avg. Easy Medium Hard Avg. Easy Medium Hard Avg. Vanilla

DeepSeek-V3.2 46.37 39.46 26.33 38.28 64.58 53.33 31.67 51.33 60.42 53.33 36.67 51.17 45.36 0 7.04 × 105 Gemini-2.5-Flash 24.81 31.44 22.67 26.15 3.75 5.00 5.00 4.50 5.00 3.89 12.78 7.00 15.01 0 1.30 × 106 Claude-Haiku-4.5 43.11 38.46 27.67 37.07 50.42 36.11 21.11 37.33 64.58 54.44 33.33 52.17 41.26 0 9.57 × 105 Grok-4-Fast 51.38 44.82 33.33 43.99 62.50 58.33 30.00 51.50 58.75 48.33 37.22 49.17 47.45 0 1.18 × 106 GPT-5-Mini 31.83 25.42 21.00 26.65 12.92 11.11 12.22 12.17 26.67 26.11 18.33 24.00 21.97 0 1.35 × 106 Qwen3-4B 47.62 44.48 35.00 42.89 29.58 20.56 17.78 23.33 46.67 44.44 38.89 43.67 37.76 0 1.91 × 106 Qwen3-30B 55.64 52.17 36.67 48.90 56.25 41.11 28.33 43.33 60.83 56.11 38.33 52.67 48.41 0 1.81 × 106 Llama-3.1-70B 34.21 27.40 17.53 26.94 27.03 16.04 11.81 18.90 28.21 24.39 26.67 26.83 23.96 0 5.74 × 105 GLM-4-32B 37.59 26.76 18.33 28.56 27.08 21.67 14.44 21.67 30.00 23.33 13.89 23.17 25.20 0 4.40 × 105 GPT-OSS-120B 49.12 38.13 30.33 40.18 47.50 30.00 26.67 36.00 50.42 43.89 43.89 46.50 40.76 0 4.40 × 105

###### + Anchor & Attractor Activation

DeepSeek-V3.2 59.40 54.52 30.00 49.10 62.92 58.89 35.00 53.33 44.17 38.89 29.44 38.17 47.27 0.22 1.94 × 106 Gemini-2.5-Flash 30.58 36.45 23.67 30.26 10.00 3.89 6.67 7.17 23.75 18.33 22.78 21.83 21.66 0.14 2.77 × 106 Claude-Haiku-4.5 64.66 56.86 33.00 52.81 60.42 56.11 31.11 50.33 62.50 47.22 33.89 49.33 51.18 0.46 2.58 × 106 Grok-4-Fast 68.92 57.19 39.33 56.51 72.50 63.89 41.67 60.67 65.00 56.11 33.89 53.00 56.69 0.66 3.17 × 106 GPT-5-Mini 26.32 25.08 21.00 24.35 11.67 12.22 11.67 11.83 18.33 18.33 11.67 16.33 18.74 0.09 2.74 × 106 Qwen3-4B 59.15 46.15 30.33 46.59 47.50 43.89 23.33 39.17 30.83 20.00 15.56 23.00 38.13 0.27 1.87 × 106 Qwen3-30B 64.16 51.84 33.33 51.20 60.42 46.67 27.78 46.50 39.17 28.33 21.67 30.67 44.31 0.36 1.97 × 106 Llama-3.1-70B 44.11 39.13 27.00 37.47 33.33 24.44 13.89 24.83 20.83 18.33 17.33 19.00 28.98 0.33 1.88 × 106 GLM-4-32B 59.90 50.84 30.33 48.30 52.50 43.33 18.33 39.50 29.58 21.11 21.11 24.50 39.40 0.41 1.82 × 106 GPT-OSS-120B 56.14 50.17 36.67 48.50 57.08 46.11 35.00 47.17 52.08 43.33 37.78 45.17 47.22 0.44 2.48 × 106

###### + Annotated Anchor & Attractor Activation

DeepSeek-V3.2 65.66 57.19 32.33 53.11 72.50 60.00 35.00 57.50 73.75 59.44 39.44 59.17 55.96 0.88 1.65 × 106 Gemini-2.5-Flash 37.84 39.46 27.00 35.07 1.25 3.89 5.56 3.33 8.33 8.89 15.56 10.67 19.75 0.69 2.34 × 106 Claude-Haiku-4.5 70.93 58.53 34.33 56.21 63.75 52.78 34.44 51.67 74.58 62.22 42.78 61.33 56.37 0.77 2.26 × 106 Grok-4-Fast 75.94 59.20 40.00 60.12 80.00 79.44 55.00 72.33 78.75 70.56 45.00 66.17 65.10 0.97 2.64 × 106 GPT-5-Mini 36.59 28.09 31.67 32.57 16.67 16.11 15.00 16.00 22.92 22.78 22.22 22.67 25.34 0.74 2.33 × 106 Qwen3-4B 72.18 63.21 45.67 61.52 60.83 53.33 34.44 50.67 69.17 66.11 51.11 62.83 58.92 0.92 2.68 × 106 Qwen3-30B 73.18 62.88 47.33 62.32 67.08 62.22 42.78 58.33 71.67 63.33 41.11 60.00 60.60 0.95 2.73 × 106 Llama-3.1-70B 56.64 48.83 33.00 47.19 45.83 41.11 21.11 37.00 55.00 51.11 37.22 48.50 44.77 0.96 1.69 × 106 GLM-4-32B 63.91 53.18 34.67 51.90 55.83 45.56 30.00 45.00 55.42 43.33 30.56 44.33 47.95 0.92 1.62 × 106 GPT-OSS-120B 59.40 46.49 34.00 47.90 47.08 38.89 28.33 39.00 63.33 53.89 42.22 54.17 47.18 0.68 2.05 × 106

Table 1: Main results on A3-Bench under different memory paradigms across ten LLMs.

attractors, ensuring they are strongly related to the reasoning process and belong to the same subdomain. For each question, the anchors and attractors are denoted as {Anchori}ni=1a and {Attractori}ni=1t , where na ≤ 6 and nt ≤ 4.

Finally, the A3-Bench problem set is created and associated with the anchors and attractors. The statistics of A3-Bench are shown in Table 2.

##### 4 Experiments

This section outlines the experimental framework for focusing on the accurate activation of memory underlying scientific reasoning. § 4.1 describes the benchmarking method, § 4.2 introduces memory paradigms, and § 4.3 presents the evaluation synergy metric.

###### 4.1 Benchmarking Method

We instantiate the proposed memory-activation method by adapting HybridRAG (Sarmah et al.,

2024). As shown in Figure 5, the framework consists of two core components: the Memory TwinNeedle Activator and the Context Fabric Composer.

Memory Twin-Needle Activator. We build a hybrid memory substrate: Anchors/Attractors are indexed in a dense store Ivec and organized in a knowledge graph G = (Vanc ∪ Vattr,Erel). Instantiated with HybridRAG, the Vector Needle retrieves top-k nodes by semantic similarity, while the Graph Needle traverses Erel to recover their logical links:

z∗ ≈ Φhybrid(x) ≜ V(x) ⊕ G V(x) . (8)

Context Fabric Composer. We compose the final context by weaving the query x with the activated state z∗:

Cfinal = W(x,z∗) ≜ I ⊕ x ▷◁ S(z∗) , (9)

where I is a fixed instruction prefix, S(·) serializes z∗ into an LLM-readable form.

Statistics Number Total Problems 2,198 By Subject

Math 998(45.40%) Physics 600(27.30%) Chemistry 600(27.30%)

###### By Difficulty

Easy 879(40.00%) Medium 659(29.98%) Hard 660(30.02%)

###### Anchors/Attractors (per problem)

Average Anchor Count 2.79 Average Attractor Count 2.33 Max Anchor Count 6 Max Attractor Count 4

Table 2: Statistics of A3-Bench.

###### 4.2 Memory Paradigms

Paradigms. We evaluate memory-driven scientific reasoning under three paradigms: (i) No memory, where the model answers from parametric knowledge A = M(Q); (ii) Full memory, where it conditions on activated evidence from the full library A = M Q; Activate(Q,KTotal) ; and (iii) Gold memory, which restricts activation to the human-labeled subset A = M Q; Activate(Q,KGold) .

Base Models. We choose 10 LLMs spanning scales, architectures, and access types (open vs. proprietary): DeepSeek-V3.2, Gemini-2.5Flash (Comanici et al., 2025), Claude-Haiku-

- 4.5 (Anthropic, 2025), Grok-4-Fast (xAI, 2025), GPT-5-Mini, Qwen3-4B, Qwen3-30B, Llama-3.170B (Grattafiori et al., 2024), GLM-4-32B (GLM

et al., 2024), and GPT-OSS-120B.

###### 4.3 Evaluation Metrics

Accuracy (Acc). We report Acc by matching the model’s final answer to the ground truth.

AAUI. We propose AAUI (Anchor–Attractor Utilization Index) to measure how well a model activates expert-annotated Anchors/Attractors during reasoning. For annotated Anchors and Attractors sets Ai,Ti and response yi, define AUi = |A1

i| a∈Ai ⊮A(a,yi), TUi =

1 |Ti| t∈Ti ⊮T(t,yi), where ⊮ indicates semantic presence in yi. We compute

- 1

- 2

AAUIi =

AUi + TUi 2

+ AUi · TUi , (10)

Method M.C. M.E. P.C. P.E. Avg. Tokens Vanilla

DeepSeek-V3.2 15.34 27.02 11.02 35.65 21.25 1.12 × 106 Gemini-2.5-Flash 6.47 20.32 1.27 26.09 13.28 1.74 × 106 Claude-Haiku-4.5 32.44 49.27 21.61 55.65 40.29 1.58 × 106 Grok-4-Fast 27.54 26.61 28.39 35.65 27.53 2.41 × 106 GPT-5-Mini 18.11 13.95 5.93 4.35 14.52 1.63 × 106 Qwen3-4B 7.02 13.39 1.69 14.78 9.84 1.14 × 106 Qwen3-30B 9.15 20.40 4.66 25.22 14.67 1.03 × 106 Llama-3.1-70B 3.33 6.85 2.97 10.43 5.24 1.32 × 106 GLM-4-32B 8.69 15.32 3.39 18.26 11.71 9.85 × 105 GPT-OSS-120B 11.09 22.10 8.90 29.57 16.80 1.61 × 106

###### Chain of Thought

DeepSeek-V3.2 20.61 33.47 14.83 41.74 26.97 2.28 × 106 Gemini-2.5-Flash 10.91 24.27 5.08 31.30 17.47 3.61 × 106 Claude-Haiku-4.5 38.63 55.08 26.69 60.87 46.17 3.83 × 106 Grok-4-Fast 33.92 31.13 33.47 40.87 32.88 5.21 × 106 GPT-5-Mini 22.92 17.82 10.17 6.09 18.71 3.72 × 106 Qwen3-4B 10.53 18.47 3.81 20.00 14.03 2.46 × 106 Qwen3-30B 12.94 26.37 8.90 31.30 19.60 2.18 × 106 Llama-3.1-70B 5.91 10.40 5.93 14.78 8.38 2.63 × 106 GLM-4-32B 12.85 20.81 5.93 25.22 16.46 2.39 × 106 GPT-OSS-120B 16.17 27.74 13.56 35.65 22.15 3.47 × 106

###### Anchor & Attractor Activation

DeepSeek-V3.2 27.08 40.24 21.19 48.70 33.60 2.35 × 106 Gemini-2.5-Flash 17.74 29.11 10.59 38.26 23.27 3.53 × 106 Claude-Haiku-4.5 45.38 61.69 32.63 67.83 52.79 3.57 × 106 Grok-4-Fast 40.66 39.11 39.41 47.83 40.14 4.98 × 106 GPT-5-Mini 29.39 23.87 16.53 12.17 24.95 3.46 × 106 Qwen3-4B 16.91 24.76 7.63 26.96 20.16 2.31 × 106 Qwen3-30B 19.13 32.66 15.25 38.26 25.89 2.33 × 106 Llama-3.1-70B 11.46 16.05 11.44 20.87 13.99 2.37 × 106 GLM-4-32B 19.22 27.50 11.86 32.17 22.97 2.25 × 106 GPT-OSS-120B 22.73 33.95 20.34 42.61 28.58 3.15 × 106

Table 3: Generalized experiments for OlympiadBench.

where AAUIi ∈ [0,1]. AAUI = N1 Ni=1 AAUIi, which combines anchor/attractor recall with an in-

teraction term to reward simultaneous activation.

##### 5 Analysis

This section presents the main results (§ 5.1), generalized analysis (§ 5.2), memory gains (§ 5.3), inference-time analysis (§ 5.4), and error-type distribution (§ 5.5). App. D evaluates anchor-only and attractor-only activation, App. E reports significance tests and noise interference, and App. F provides successful and failure cases.

###### 5.1 Main Results

Table 1 reports the performance of ten LLMs under three paradigms. The results provide four main findings about memory-driven scientific reasoning.

Memory augmentation consistently improves scientific reasoning across LLMs and subjects. Across subjects and difficulty levels, all ten LLMs achieve higher accuracy under Annotated Activation than in the Vanilla, improving the average from 34.71% to 48.19% (+13.48). The gains are model-dependent, ranging from modest increases (e.g., GPT-5-Mini +3.37; Gemini-2.5-Flash +4.74)

[Figure 6]

Figure 6: Heatmap analysis of performance gains and memory utility across subjects and difficulties.

[Figure 7]

Figure 7: Acc. (%) vs. avg. inference time (s) per question. Gray: vanilla; Red: annotated anchors & attractors. Arrows indicate the performance shift.

to substantial boosts (e.g., GLM-4-32B +22.75; Qwen3-4B +21.16; Llama-3.1-70B +20.81), indicating heterogeneous ability to leverage activated Anchors and Attractors.

Memory activation is most beneficial on hard problems and reduces the difficulty gap. In the Vanilla paradigm, Hard subsets remain difficult for most models (e.g., Physics-Hard: Grok4-Fast 30.00%, GLM-4-32B 14.44%, Qwen3-4B 17.78%, Llama-3.1-70B 11.81%). With annotated memory, Hard performance improves substantially, especially in Physics (e.g., Grok-4-Fast +25.00, GLM-4-32B +15.56), and also in other domains (e.g., Qwen3-4B: Math-Hard +10.67, ChemistryHard +12.22). These gains suggest that many hard problems fail due to missing or misselected solution templates (Attractors); activating the right template makes multi-step reasoning more tractable.

AAUI correlates with accuracy and diagnoses reasoning fidelity under memory activation. Under the Anchor & Attractor Activation paradigm, higher AAUI generally aligns with higher accuracy. For example, Grok-4-Fast reaches AAUI= 0.66 with Avg.= 56.69% and Claude-Haiku-4.5 reaches AAUI= 0.46 with Avg.= 51.18%, while GPT-5Mini has AAUI= 0.09 with Avg.= 18.74%. This pattern suggests that AAUI captures whether a model co-activates compatible Anchors and Attractors and converts them into correct reasoning.

###### 5.2 Generalized Analysis

To assess the transferability of our memory mechanism, we evaluate the same models on the OlympiadBench, which includes competition-level (M.C., P.C.) and entrance-exam-level (M.E., P.E.)

problems. Table 3 summarizes the results and supports two observations.

Anchor–Attractor activation generalizes beyond the source dataset. For all ten models, Anchor– Attractor Activation consistently outperforms both the Vanilla baseline and CoT prompting. Overall, Anchor & Attractor Activation improves the average score across models by 11.12 points over Vanilla and by 6.35 points over CoT. For example, DeepSeek-V3.2 reaches 33.60% with activation (+12.35 vs. Vanilla; +6.63 vs. CoT). This suggests our Anchors and Attractors capture reusable scientific concepts and solution patterns that transfer to unseen high-difficulty problems.

Gains are largest on competition-level subsets. Improvements are most pronounced on the hardest competition subsets. On Physics Competition (P.C.), where Vanilla performance is extremely low (e.g., Qwen3-4B: 1.69%), activation raises the score to 7.63% (∼4.5×). Similarly, Claude-Haiku-

- 4.5 improves from 21.61% (Vanilla) to 32.63% (Activated) on P.C. These results indicate that CoT may help derivations but often misses the right starting principles; activating Attractors and supporting Anchors helps recover viable solution paths.
- 5.3 Gains of Memory

Overall, the increase in memory utility (∆AAUI) improves accuracy (∆Accuracy), especially under high difficulty conditions. As shown in Figure 6, the heatmap shows that math performs best under Hard difficulty with a ∆Accuracy increase of +10.67, alongside a notable increase in ∆AAUI. Physics and Chemistry also show improvements, especially in Hard difficulty, where the increase in

[Figure 8]

Figure 8: Evolution of error distributions across five paradigms.

∆AAUI is closely tied to gains in ∆Accuracy. In contrast, changes in ∆AAUI have a smaller effect on accuracy at Easy and Medium levels.

- 5.4 Inference Time Analysis Overall, under the Annotated Anchors & Attractors paradigm, the average inference time decreases by 2.1 seconds, with a performance improvement of 13.5%. As shown in Figure 7, the plot demonstrates that nearly all models show varying degrees of accuracy improvement with reduced inference times when switching to the Annotated Anchors & Attractors paradigm. Notably, larger models such as Llama-3.1-70B and Grok-4-Fast exhibit substantial performance gains and a reduction in inference time. Overall, this paradigm not only enhances model accuracy but also demonstrates a clear advantage in reducing inference latency.
- 5.5 Error Type Distribution

Overall, the models show improvements in error type distribution across the five experimental modes, particularly in Reasoning and Knowledge errors. As shown in Figure 8, the trend line chart indicates that, as the paradigms change (from Vanilla to Annotated), Reasoning and Knowledge errors decrease substantially, especially in the +Both and +Annotated paradigms, where the error rates are notably lower than in the Vanilla and +Anchors paradigms. Calculation and Formatting errors show minimal changes across the paradigms, with a slight reduction in error rates overall. Comprehension errors remain relatively stable across different paradigms, with only minor fluctuations.

- 6 Related Works This section reviews two strands relevant to our study: (i) memory methods for LLMs, and (ii) scientific reasoning benchmarks.

Memory. Memory can take multiple forms, with mechanisms tailored to different needs. A common line treats memory as external, writable, and retrievable: RAG links models to non-parametric stores via indices, enabling updates at inference time (Wang et al., 2024b; Oche et al., 2025). Another line targets long-horizon interaction, e.g., MemGPT’s “virtual memory” that swaps short- and long-term storage to mitigate context limits (Packer et al., 2024; Wang et al., 2025; Kang et al., 2025). Agentic frameworks (Chhikara et al., 2025; Li et al., 2025; Zhang et al., 2026a) store experience traces as episodic memory and improve behavior via reflection. LoCoMo (Maharana et al., 2024) benchmarks long-term conversational memory.

Scientific Reasoning Benchmarks. OlympiadBench (He et al., 2024) provides Olympiad-level bilingual multimodal problems (notably math and physics) with expert annotations. EMMA (Hao et al., 2025) targets multimodal reasoning across math, physics, chemistry, and coding. Humanity’s Last Exam (HLE) (Phan et al., 2025) is an expert-level, broad-coverage benchmark with a multimodal portion. MathVista (Lu et al., 2023) evaluates math reasoning in visual contexts, emphasizing fine-grained perception and compositional reasoning. ScienceBoard (Sun et al., 2025) benchmarks the scientific discovery tasks. However, these benchmarks do not measure memory utilization during scientific reasoning.

##### 7 Conclusion

We present A3-Bench, a memory-driven benchmark for scientific reasoning grounded in the activation of Anchors and Attractors. Consistent with human memory organization and retrieval, our design reflects hierarchical knowledge and contextdependent activation during problem solving. Using the SAPM process, we annotate 2,198 problems across math, physics, and chemistry with structured anchor units and attractor schemas. We further develop a dual-scale memory evaluation framework and the AAUI metric to quantify memory activation during reasoning. Extensive experiments show that activation improves accuracy, keeps token consumption controllable, and exposes substantial differences in how models utilize memory. Overall, A3-Bench provides a memorycentric, interpretable, and cognitively aligned evaluation paradigm that supports progress toward more human-like, memory-driven scientific reasoning.

##### References

Anthropic. 2025. Claude 4.5 haiku.

Susanne Arndt, Patrick Ion, Mila Runnwerth, Moritz Schubotz, and Olaf Teschke. 2021. 10 years later: The mathematics subject classification and linked open data. In Proceedings of the Conference on Intelligent Computer Mathematics (CICM 2021), pages 153–158. Springer / TIB.

Oded Bein and Yael Niv. 2025. Schemas, reinforcement learning and the medial prefrontal cortex. Nature Reviews Neuroscience, 26(3):141–157.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Preprint, arXiv:2507.06261.

Wendi Cui, Zhuohang Li, Damien Lopez, Kamalika Das, Bradley A Malin, Sricharan Kumar, and Jiaxin Zhang. 2024. Divide-conquer-reasoning for consistency evaluation and automatic improvement of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 334–361.

Xia Du, Shuhan Sun, Pengyuan Liu, and Dong Yu. 2025. Investigating value-reasoning reliability in small large language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 7757–7797.

- Edward Dunne and Klaus Hulek. 2020a. Mathematics subject classification 2020. Not. Am. Math. Soc, 67(3):410–411.
- Edward Dunne and Klaus Hulek. 2020b. Mathematics subject classification 2020. Not. Am. Math. Soc, 67(3):410–411.

Kehua Feng, Keyan Ding, Weijie Wang, Xiang Zhuang, Zeyuan Wang, Ming Qin, Yu Zhao, Jianhua Yao, Qiang Zhang, and Huajun Chen. 2024. Sciknoweval: Evaluating multi-level scientific knowledge of large language models. Preprint, arXiv:2406.09098.

Karl Friston. 2010. The free-energy principle: a unified brain theory? Nature reviews neuroscience, 11(2):127–138.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. 2025. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark. arXiv preprint arXiv:2501.05444.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. 2024. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850.

Stephen Heller, Alan McNaught, Stephen Stein, Dmitrii Tchekhovskoi, and Igor Pletnev. 2013. Inchi-the worldwide chemical structure identifier standard. Journal of cheminformatics, 5(1):7.

Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. 2021a. Aligning ai with shared human values. Proceedings of the International Conference on Learning Representations (ICLR).

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021b. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR).

Muye Huang, Lingling Zhang, Jie Ma, Han Lai, Fangzhi Xu, Yifei Li, Wenjun Wu, Yaqiang Wu, and Jun Liu. 2025. Chartsketcher: Reasoning with multimodal feedback and reflection for chart understanding. arXiv preprint arXiv:2505.19076.

Patrick Ion and Olaf Teschke. 2016. Mathematics subject classification and related classifications in the digital world. European Mathematical Society Magazine.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. 2025. Memory os of ai agent. arXiv preprint arXiv:2506.06326.

Ching-Yun Ko, Sihui Dai, Payel Das, Georgios Kollias, Subhajit Chaudhury, and Aurelie Lozano. 2024. Memreasoner: A memory-augmented llm architecture for multi-hop reasoning. In The First Workshop on System-2 Reasoning at Scale, NeurIPS’24.

Maikel Leon. 2025. Gpt-5 and open-weight large language models: Advances in reasoning, transparency, and control. Information Systems, page 102620.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023. Camel: Communicative agents for "mind"

exploration of large scale language model society. Preprint, arXiv:2303.17760.

Zhiyu Li, Shichao Song, Chenyang Xi, Hanyu Wang, Chen Tang, Simin Niu, Ding Chen, Jiawei Yang, Chunyu Li, Qingchen Yu, et al. 2025. Memos: A memory os for ai system. arXiv preprint arXiv:2507.03724.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Junnan Liu, Hongwei Liu, Linchen Xiao, Ziyi Wang, Kuikun Liu, Songyang Gao, Wenwei Zhang, Songyang Zhang, and Kai Chen. 2025. Are your llms capable of stable reasoning? In Findings of the Association for Computational Linguistics: ACL 2025, pages 17594–17632.

Xu Liu, Steve Ramirez, Petti T Pang, Corey B Puryear, Arvind Govindarajan, Karl Deisseroth, and Susumu Tonegawa. 2012. Optogenetic stimulation of a hippocampal engram activates fear memory recall. Nature, 484(7394):381–385.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, KaiWei Chang, Michel Galley, and Jianfeng Gao. 2023. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of llm agents. arXiv preprint arXiv:2402.17753.

Agada Joseph Oche, Ademola Glory Folashade, Tirthankar Ghosal, and Arpan Biswas. 2025. A systematic review of key retrieval-augmented generation (rag) systems: Progress, gaps, and future directions. arXiv preprint arXiv:2507.18910.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2024. Memgpt: Towards llms as operating systems. Preprint, arXiv:2310.08560.

Matilda QR Pembury Smith and Graeme D Ruxton.

2020. Effective use of the mcnemar test. Behavioral Ecology and Sociobiology, 74(11):133.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. 2025. Humanity’s last exam. arXiv preprint arXiv:2501.14249.

Nicholas T. Runcie, Charlotte M. Deane, and Fergus Imrie. 2025. Assessing the chemical intelligence of large language models. arXiv preprint arXiv:2505.07735.

Bhaskarjit Sarmah, Benika Hall, Rohan Rao, Sunil Patel, Stefano Pasquali, and Dhagash Mehta. 2024. Hybridrag: Integrating knowledge graphs and vector retrieval augmented generation for efficient information extraction. Preprint, arXiv:2408.04948.

Dominique Siegenthaler, Henry Denny, Sofía Skromne Carrasco, Johanna Luise Mayer, Daniel Levenstein, Adrien Peyrache, Stuart Trenholm, and Émilie Macé. 2025. Visual objects refine head direction coding. Science, 389(6765):eadu9828.

Arthur Smith. 2020. Physics subject headings (physh). KO KNOWLEDGE ORGANIZATION, 47(3):257– 266.

Qiushi Sun, Zhoumianze Liu, Chang Ma, Zichen Ding, Fangzhi Xu, Zhangyue Yin, Haiteng Zhao, Zhenyu Wu, Kanzhi Cheng, Zhaoyang Liu, et al. 2025. Scienceboard: Evaluating multimodal autonomous agents in realistic scientific workflows. In ICML 2025 Workshop on Computer Use Agents.

Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, et al. 2024a. Memoryllm: Towards self-updatable large language models. arXiv preprint arXiv:2402.04624.

Zheng Wang, Shu Teo, Jieer Ouyang, Yongjun Xu, and Wei Shi. 2024b. M-rag: Reinforcing large language model performance through retrieval-augmented generation with multiple partitions. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1966–1978.

Zixuan Wang, Bo Yu, Junzhe Zhao, Wenhao Sun, Sai Hou, Shuai Liang, Xing Hu, Yinhe Han, and Yiming Gan. 2025. Karma: Augmenting embodied ai agents with long-and-short term memory systems. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 1–8. IEEE.

Wenjun Wu, Lingling Zhang, Bo Zhao, Muye Huang, QianYing Wang, and Jun Liu. 2025. Causal-r: A causal-reasoning geometry problem solver for optimized solution exploration. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

xAI. 2025. Grok 4 fast: Pushing the frontier of costefficient intelligence.

Zhihui Xie, Jizhou Guo, Tong Yu, and Shuai Li. 2024. Calibrating reasoning in language models with internal consistency. Advances in Neural Information Processing Systems, 37:114872–114901.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Jian Zhang, Zhangqi Wang, Haiping Zhu, Jun Liu, Qika Lin, and Erik Cambria. 2026a. Mars: A multi-agent framework incorporating socratic guidance for automated prompt optimization. In Proceedings of AAAI.

Jian Zhang, Zhiyuan Wang, Zhangqi Wang, Xinyu Zhang, Fangzhi Xu, Qika Lin, Rui Mao, Erik Cambria, and Jun Liu. 2026b. Maps: A multi-agent framework based on big seven personality and socratic guidance for multimodal scientific problem solving. In Proceedings of AAAI.

Xinyu Zhang, Yuxuan Dong, Yanrui Wu, Jiaxing Huang, Chengyou Jia, Basura Fernando, Mike Zheng Shou, Lingling Zhang, and Jun Liu. 2025a. Physreason: A comprehensive benchmark towards physics-based reasoning. arXiv preprint arXiv:2502.12054.

Xinyu Zhang, Yuxuan Dong, Lingling Zhang, Chengyou Jia, Zhuohang Dang, Basura Fernando, Jun Liu, and Mike Zheng Shou. 2025b. Cofft: Chain of foresight-focus thought for visual language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Zehua Zhao, Zhixian Huang, Junren Li, Siyu Lin, Junting Zhou, Fengqi Cao, Kun Zhou, Rui Ge, Tingting Long, Yuexiang Zhu, Yan Liu, Jie Zheng, Junnian Wei, Rong Zhu, Peng Zou, Wenyu Li, Zekai Cheng, Tian Ding, Yaxuan Wang, Yizhao Yan, Tingru Wei, Haowei Ming, Weijie Mao, Chen Sun, Yiming Liu, Zichen Wang, Zuo Zhang, Tong Yang, Hao Ma, Zhen Gao, and Jian Pei. 2025. Superchem: A multimodal reasoning benchmark in chemistry. Preprint, arXiv:2512.01274.

Yue-Qing Zhou and James J Knierim. 2025. Neural compass in the sky. Science, 390(6770):239–240.

##### A Free-Energy-Driven Memory Activation

To better elucidate the preliminaries discussed in § 2, specifically the mechanisms of anchors and attractors (Zhou and Knierim, 2025; Siegenthaler

- et al., 2025), the dynamics of memory activation (Friston, 2010), and the paradigm of memoryaugmented reasoning (Ko et al., 2024), we formally define the memory process as “Anchor-Induced Attractor Dynamics.” By characterizing cognitive evolution as a trajectory within a potential energy landscape, we establish the following proposition:

Proposition (SAAM Dynamics). Let the Subject define a manifold S anchored by A. For a reconstructed question q, the memory state m∗ obeys m∗ = arg minm∈S DKL(q ∥p(m|A)) + H(m) . Reasoning follows the trajectory m˙ = −∇F towards stable Attractors; minimizing Free Energy thus yields Lyapunov-like stability and facilitates memory mapping.

The proof of proposition. We aim to prove that the memory activation state m∗ in a Subject domain S converges to a stable equilibrium under the guidance of Anchors A and Attractors:

m∗ = arg min m∈S

F(m,A,q), (11)

where F is the Variational Free Energy defined by the divergence between the question q and the anchor-parameterized distribution p(m|A).

- Step 1: Definition of the Energy Landscape. Let the Subject manifold S be equipped with a potential energy function U(m;A). The Anchors

A define a set of local minima {ak}Nk=1 ⊂ S, s.t. ∇U(ak) = 0. We define the Variational Free Energy F as:

F(m,A,q) = DKL(q ∥p(m|A)) + H(m), (12)

where DKL represents the informational "distance" (prediction error) and H represents the system entropy (uncertainty).

- Step 2: Variational Minimization. To find the optimal memory state m∗, we apply the variational principle. The first-order necessary condition for a minimum at m∗ is the vanishing of the functional derivative:

δF δm m∗

= 0. (13)

Expanding this, we seek a state where the drive to match the question q (accuracy) is perfectly balanced by the pull of the Anchors A (prior knowledge) and the constraint of entropy H.

- Step 3: Attractor-Driven Trajectory. We define the reasoning process as a gradient flow in the Subject space. The temporal evolution of the mental state m˙ follows:

m˙ = −η∇mF(m,A,q), (14)

where η is the learning rate or cognitive plasticity. In this dynamical system, the Anchors A act as **Attractors**, creating basins of attraction that pull the trajectory m˙ toward the nearest stable fixed point.

- Step 4: Convergence and Stability. By constructing F as a Lyapunov function, we observe that:

dF dt

= ⟨∇mF,m˙ ⟩ = −η∥∇mF∥2 ≤ 0. (15)

The strictly non-positive derivative ensures that the system trajectory is dissipative and must converge to a fixed point m∗.

Conclusion. Thus, the reasoning process:

[DKL(q ∥p(m|A)) + H(m)] (16)

mˆ = arg min

m

formally links the external query q with internal anchor-based structures, ensuring that memory activation is an emergent property of Attractor Dynamics and Free Energy Minimization.

##### B Annotation Guidance

This section presents the manual annotation guidelines for each step in § 3.

###### B.1 Anchor & Attractor Developing

- • Subdomain Definitions and Frameworks: Before annotation, experts should delineate each subdomain’s scope and key reasoning framework, so that the boundaries and core concepts are explicit. This provides a stable reference for selecting anchors and attractors without drifting across subdomains.
- • Anchors Identification: Anchors are the foundational reasoning primitives (e.g., core concepts, theorems, formulas). Experts should extract representative anchors from the

- subdomain’s canonical content, prioritizing items that are broadly reusable and frequently invoked as starting principles in problem solving.
- • Attractors Identification: Attractors connect abstract principles to actionable solution pathways. Each attractor contains an Abstract Schema (a reusable solution template grounded in anchors) and Episodic Exemplars (concrete instantiations). Experts should curate attractors that reliably operationalize anchors for typical scientific tasks.
- • Anchor–Attractor Relations and Consistency: After extraction, experts should specify which anchors support each attractor and how they interact during reasoning, ensuring the mapping is coherent, non-overlapping, and free of redundancy. The role of each unit should be unambiguous in the reasoning chain.
- • Library Construction: Finally, all anchors and attractors are organized into an Anchor Library and an Attractor Library in JSON format. Each entry includes an identifier, a concise definition, and explicit relations to enable scalable management, retrieval, and traceability.

###### B.2 Problem Reconstructing

- • Error Diagnosis: Experts review the original question alongside LLM answers to identify common failure modes (e.g., missing steps, wrong assumptions, incomplete derivations), analyze their causes, and revise the question accordingly.
- • Cross-Model Refinement: For second-stage questions, experts compare outputs from three LLMs to surface systematic discrepancies and uncovered knowledge gaps, then refine the problem to elicit essential reasoning steps and broader scientific coverage.
- • Multi-hop Enforcement: Reconstructed questions require at least two coupled knowledge points with an explicit stepwise dependency, preventing one-shot solutions and promoting multi-stage reasoning.
- • Reference Solution Writing: Each revised problem is paired with a correct, complete

standard answer that includes the final result and the key intermediate steps.

###### B.3 Memory Mapping

- • Subdomain Assignment: For each question, three LLMs propose candidate subdomains and a voting rule selects the final label; experts then verify the assignment and confirm the secondary discipline to ensure accurate classification.
- • Anchor–Attractor Annotation: Given the confirmed discipline, experts manually annotate Anchors (e.g., core concepts, theorems, formulas) and Attractors (e.g., abstract schemas with episodic exemplars) following the annotation principles, where anchors support key reasoning steps and attractors operationalize them into solution pathways.
- • Quantity Control and Consistency: Each question is capped at no more than 6 annotated units in total (anchors + attractors) to keep complexity moderate; experts also check that selected items are necessary, non-redundant, and tightly aligned with the intended reasoning process.
- • Rationale Logging: Experts record brief justifications for the chosen anchors and attractors and summarize the annotation process per question, enabling traceability and efficient future review.

C Subject Taxonomy

As shown in Table 4, this section outlines the dataset taxonomy across three subject domains: Math, Physics, and Chemistry. For each domain, we first state the primary classification criteria and then introduce the corresponding second-level subdomains. This design emphasizes the hierarchical organization of disciplinary knowledge and clarifies the distinctions across domains.

###### C.1 Math

The math subset follows the Mathematics Subject Classification (MSC) system (Dunne and Hulek, 2020b; Arndt et al., 2021; Ion and Teschke, 2016), jointly developed and maintained by the American Mathematical Society (AMS) and zbMATH Open (Dunne and Hulek, 2020a). As a globally adopted indexing scheme for journals, scholarly databases, and university curricula, MSC provides

Subject Subdomain Count

Algebra 158 Geometry 120 Number Theory 108 Calculus & Analysis 132 Discrete Math 132 Logic & Set Theory 96 Statistics & Probability 120 Computational Math 132

Math

Mechanics 120 Thermodynamics 120 Optics 120 Electromagnetism 120 Modern Physics 120

Physics

Inorganic Chemistry 120 Organic Chemistry 120 Physical Chemistry 120 Analytical Chemistry 120 Biochemistry 120

Chemistry

Grand Total 2,198

- Table 4: Subdomain composition of the A3-Bench.

a stable and internationally comparable foundation for benchmarking LLMs (Huang et al., 2025; Wu et al., 2025). Building on MSC primary classes (00–99) and modern mathematical organization, we group problems into eight subdomains: Algebra, Geometry, Number Theory, Mathematical Analysis, Discrete Math, Logic & Set Theory, Statistics & Decision Sciences, and Computational Math.

In brief, these subdomains cover core algebraic structures, spatial reasoning and invariants, integer arithmetic and congruences, limits/calculus and infinite processes, combinatorics and graph structures, formal foundations, probabilistic inference under uncertainty, and numerical/scientific computing. This taxonomy enables systematic assessment of mathematically grounded retrieval, abstraction across concept hierarchies, and domain-specific reasoning strategies.

###### C.2 Physics

The Physics taxonomy follows standard highereducation curricula and internationally recognized physics classification standards (Smith, 2020), aligned with prior work on broad LLM evaluation and alignment (Hendrycks et al., 2021b,a) and recent benchmarks for complex scientific reasoning and agentic exploration (He et al., 2024; Li et al., 2023). We organize the dataset into five canonical subdomains: Mechanics, Thermodynamics, Optics, Electromagnetism, and Modern Physics. This di-

vision supports systematic evaluation of whether models can retrieve appropriate physical principles (anchors) and apply scenario-specific solution patterns (attractors) across abstraction levels.

Mechanics covers kinematics, dynamics, and conservation laws, emphasizing force analysis and motion under constraints. Thermodynamics focuses on heat, work, and state variables, including entropy-driven reasoning and statisticalmechanical interpretations. Optics spans geometric optics (imaging, lenses, ray tracing) and wave optics (interference, diffraction, polarization), requiring careful treatment of limiting assumptions. Electromagnetism addresses charges, fields, and circuits, involving vector-field reasoning and circuitlevel modeling. Modern Physics extends beyond classical theory to relativity and quantum phenomena, where counter-intuitive effects demand strict adherence to formal principles.

###### C.3 Chemistry

The Chemistry taxonomy follows the canonical organization of chemical science education and advanced evaluation frameworks (Hendrycks et al., 2021b,a), aligned with internationally recognized IUPAC standards (Heller et al., 2013) and recent benchmarks for chemical intelligence (Runcie et al., 2025; Li et al., 2023; Feng et al., 2024; Zhao et al., 2025). We group problems into five subdomains: Inorganic Chemistry, Organic Chemistry, Physical Chemistry, Analytical Chemistry, and Biochemistry, enabling fine-grained evaluation of knowledge retrieval and structured reasoning across chemical contexts.

Inorganic Chemistry covers the synthesis, structure, and reactivity of inorganic and organometallic compounds. Organic Chemistry focuses on carbon-based molecules, emphasizing functional groups, reaction mechanisms, and synthesis planning. Physical Chemistry studies chemical systems through thermodynamics, kinetics, and quantum principles. Analytical Chemistry concerns qualitative and quantitative determination via methods such as spectroscopy and chromatography. Biochemistry examines biomolecular structure and function, linking chemical mechanisms to metabolic pathways and cellular processes.

Math Physics Chemistry

Method

Avg. AAUI Tokens

Easy Medium Hard Avg. Easy Medium Hard Avg. Easy Medium Hard Avg. Vanilla + Anchor-Only

DeepSeek-V3.2 60.90 49.16 28.00 47.49 64.58 58.33 43.89 56.50 69.58 56.67 37.22 56.00 52.27 0.44 1.07 × 106 Gemini-2.5-Flash 31.33 29.10 26.00 29.06 2.50 2.78 5.00 3.33 5.42 5.00 10.56 6.83 15.97 0.08 1.71 × 106 Claude-Haiku-4.5 62.66 52.51 30.67 50.00 57.92 45.00 29.44 45.50 70.00 57.22 37.22 56.33 50.50 0.43 1.49 × 106 Grok-4-Fast 63.91 53.51 36.67 52.61 60.42 56.67 30.56 50.33 67.50 49.44 38.33 53.33 52.18 0.47 1.80 × 106 GPT-5-Mini 31.08 27.76 29.00 29.46 13.75 10.56 14.44 13.00 23.75 22.78 18.89 22.00 22.93 0.26 1.73 × 106 Qwen3-4B 51.63 51.51 36.67 47.09 39.17 31.67 20.00 31.17 54.58 47.78 35.56 46.83 42.68 0.33 1.87 × 106 Qwen3-30B 53.38 48.16 33.67 45.89 52.50 36.67 28.33 40.50 47.50 47.78 27.78 41.67 43.27 0.38 1.83 × 106 Llama-3.1-70B 47.12 39.80 26.67 38.78 37.50 28.89 17.78 29.00 48.33 41.11 28.33 40.17 36.49 0.29 1.22 × 106 GLM-4-32B 57.89 45.48 28.33 45.29 39.58 33.33 18.89 31.50 41.67 34.44 21.11 33.33 38.26 0.36 9.33 × 105

- GPT-OSS-120B 55.64 47.49 34.67 46.89 50.00 38.33 32.78 41.33 55.83 52.78 47.22 52.33 46.86 0.42 1.45 × 106 Vanilla + Attractor-Only

DeepSeek-V3.2 62.16 57.19 30.33 51.10 74.17 53.89 38.33 57.33 70.83 65.00 43.33 60.83 55.46 0.46 1.24 × 106 Gemini-2.5-Flash 35.84 36.79 24.33 32.67 2.92 4.44 3.89 3.67 8.33 10.00 12.78 10.17 18.61 0.22 1.94 × 106 Claude-Haiku-4.5 66.42 59.20 30.33 53.41 61.67 48.33 30.00 48.17 72.92 63.33 40.56 60.33 53.87 0.31 1.74 × 106 Grok-4-Fast 72.93 55.85 42.33 58.62 70.42 57.78 43.89 58.67 67.92 53.33 38.89 54.83 57.60 0.49 2.17 × 106 GPT-5-Mini 26.82 26.42 28.33 27.15 18.75 12.22 17.22 16.33 20.00 22.22 18.89 20.33 22.34 0.36 1.95 × 106 Qwen3-4B 47.37 45.15 36.33 43.39 32.08 24.44 17.22 25.33 50.00 50.00 33.33 45.00 38.90 0.40 1.88 × 106 Qwen3-30B 58.15 51.17 38.67 50.20 58.33 38.89 34.44 45.33 58.33 51.67 30.56 48.00 48.27 0.42 1.85 × 106 Llama-3.1-70B 54.64 44.15 29.67 43.99 43.75 30.00 21.11 32.83 55.00 47.22 32.22 45.83 41.45 0.42 1.24 × 106 GLM-4-32B 62.66 54.52 31.00 50.70 47.92 32.78 26.11 36.83 48.33 43.89 26.11 40.33 44.09 0.40 1.19 × 106

- GPT-OSS-120B 56.64 46.15 30.33 45.59 47.08 33.89 27.22 37.17 57.08 56.67 48.33 54.33 45.68 0.41 1.66 × 106

- Table 5: Performance comparison of ten LLMs under two partial activation paradigms: vanilla + anchor-only versus vanilla + attractor-only.

##### D Different Memory Paradigms

To isolate the contributions of Anchors and Attractors, we conduct two different paradigms that activate only anchors or only attractors. Table 5 reports the results, leading to four findings.

Attractors tend to contribute more than Anchors when only one memory type is available. For most models (7/10), Attractor-only activation yields higher overall accuracy than Anchoronly activation. For example, Grok-4-Fast improves from 52.18% (Anchor-only) to 57.60% (Attractor-only), and GLM-4-32B increases from 38.26% to 44.09%. Similar trends hold for Llama-3.1-70B (36.49%→41.45%) and Qwen330B (43.27%→48.27%). This suggests that, under partial memory, access to procedural schemas is often more directly useful for completing multi-step scientific derivations.

Both memory types are needed to reach the best performance. Despite the strength of Attractoronly activation, it remains consistently below the dual annotated paradigm. For example, Grok-4Fast increases from 57.60% (Attractor-only) to 65.10% (Annotated dual), and Qwen3-30B increases from 48.27% to 60.60%. The gap is especially large for Qwen3-4B (38.90%→58.92%), indicating that procedural templates alone are insuffi-

cient without the supporting conceptual grounding. Overall, Anchors and Attractors play complementary roles: templates provide the solution pathway, while definitions and constraints help instantiate the pathway correctly for the specific problem.

Subjects show different sensitivities to memory types. The relative advantage of Attractors over Anchors varies by domain. In Chemistry, the difference is often small (e.g., Grok-4-Fast: 53.33% vs. 54.83%; Claude-Haiku-4.5: 56.33% vs. 60.33%), consistent with the need for precise property definitions alongside procedures. In contrast, Math and Physics more frequently favor Attractoronly activation, reflecting the procedural nature of theorem selection and template-based derivations (e.g., Grok-4-Fast: Math 52.61%→58.62%, Physics 50.33%→58.67%).

Dependence on complete memory support differs substantially across models. Removing either component can cause large drops for some models but only modest changes for others. For instance, Qwen3-4B falls from 58.92% (Annotated dual) to 42.68% (Anchor-only) and 38.90% (Attractor-only), and Qwen3-30B drops from 60.60% to 43.27% and 48.27%. By contrast, GPT-OSS-120B changes only slightly (47.18% to 46.86% and 45.68%). These results indicate that

### Comparison p-value Significance

#### Q vs. A < 0.001 ✓ Q vs. T < 0.001 ✓ A vs. T 0.120 × A vs. AT < 0.001 ✓ T vs. AT < 0.001 ✓ AT vs. AT∗ < 0.001 ✓

- Table 6: Results of McNemar’s Test for statistical significance. Q: question only; A: +anchors; T: +attractors; AT: +both; AT∗: +annotated . Significance level α = 0.05.

robustness to partial memory is model-dependent, and that the ability to integrate both Anchors and Attractors is a key limitation exposed by the dataset.

##### E Other Analysis

We conduct additional experiments, including a task-appropriate McNemar’s test for statistical significance and an analysis of robustness under noisy memory interference.

###### E.1 Significance Test

To quantitatively assess the necessity of memory augmentation for scientific reasoning, we apply McNemar’s test (Pembury Smith and Ruxton, 2020) to the prediction outcomes. As summarized in Table 6, the results provide statistical evidence for the reliance of scientific reasoning on external context.

First, the comparison between the question-only baseline and single-memory activation (Q vs. A/T, p < 0.001) shows that parametric knowledge alone is insufficient for solving complex scientific problems. Introducing external memory, whether declarative or procedural, leads to a statistically significant improvement in performance.

Second, the consistent advantage of combined activation (AT) over either component alone (p < 0.001) indicates that scientific reasoning operates as a dual-process mechanism. Anchors and Attractors each contribute essential information, and effective reasoning requires their joint activation rather than isolated use.

Finally, the absence of a significant difference between Anchor-only and Attractor-only activation (p = 0.120) suggests that the two memory types play complementary and comparably important roles. Neither dominates the other; instead,

[Figure 9]

Figure 9: Impact of memory relevance vs. Noise on model performance.This radial bar chart illustrates the degradation of accuracy in Grok-4-Fast as high-quality annotated memory (anchors & attractors) is progressively replaced by irrelevant noise memory. The concentric rings correspond to increasing noise replacement ratios (from 0% to 100%).

they function as parallel cognitive supports whose integration yields the strongest reasoning performance.

###### E.2 Noise Interference

As illustrated in Figure 9, increasing the noise replacement ratio consistently reduces model accuracy, highlighting that performance depends not only on having memory but on the relevance of the activated memory.

For Grok-4-Fast, accuracy declines monotonically from 65.1% with 100% annotated memory to 58.5% (20% noise), 51.2% (40%), 44.8% (60%), 38.4% (80%), and 32.5% under full noise replacement. Notably, the drop becomes pronounced once noise exceeds 40–60%, suggesting that irrelevant memory increasingly dominates the context, distracts the model from key principles, and disrupts the anchor–attractor alignment needed to initiate correct solution paths. The final performance at 100% noise approaches a near-memoryless regime, indicating that low-quality memory can effectively negate the benefits of retrieval and even harm reasoning by introducing misleading cues.

##### F Case Study

To illustrate how memory activation influences reasoning outcomes, we present two representative cases involving the Pitman–Yor process, shown in Figures 10 and 11.

Successful activation (Figure 10). In the successful case, the Memory Twin-Needle Activator retrieves both the core definition of the Pitman–Yor process (Anchor anc_105) and the relevant closedform expectation formula (Anchor anc_107). Importantly, it also activates the correct Attractor (attr_080), which encodes an abstract reasoning schema for manipulating Gamma-function ratios, along with an episodic exemplar for computing E[Kn]. With access to both the conceptual grounding (definitions) and the procedural guidance (schema), the model (Grok-4-Fast) successfully completes the multi-step derivation and obtains the correct result, 187/64. This case demonstrates that Attractors function as procedural guides, enabling reasoning trajectories that would be correct and efficiency.

Failure activation (Figure 11). The failure case highlights the fragility of reasoning under imprecise memory activation. Although the correct subject definition is retrieved (anc_105), the activation result contains substantial noise. The Vector Needle selects a distracting Anchor (anc_108), a complex closed-form expression that shifts the reasoning away from the intended recurrence-based approach. Meanwhile, the Graph Needle activates an irrelevant Attractor (attr_150) associated with Dirichlet series and multiplicative number theory.

As a consequence, the model (GLM-4-32B) follows an incompatible procedure and fails to produce a valid derivation, yielding an incorrect value (2.5). This example shows that identifying the correct topic alone is insufficient. Reliable reasoning requires coherent alignment between declarative content (Anchors) and procedural guidance (Attractors).

[Figure 10]

###### Figure 10: Successful case of Grok-4-Fast on a TheoremQA problem under anchor & attractor activation. Using HybridRAG, the model activates relevant anchors and attractors from the full memory repositories, composes them with the question, and produces the correct answer (187/64).

[Figure 11]

###### Figure 11: Failure case of GLM-4-32B on a TheoremQA problem under anchor & attractor activation. Although HybridRAG is used, the retrieved anchors/attractors are irrelevant and provide little support; after composition with the question, the model outputs an incorrect answer (2.5).

