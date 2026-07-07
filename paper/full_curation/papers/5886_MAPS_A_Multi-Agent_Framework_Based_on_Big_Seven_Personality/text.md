# arXiv:2503.16905v2[cs.AI]12Nov2025

## MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning

### Jian Zhang*1,2, Zhiyuan Wang*1,3, Zhangqi Wang*1,3, Fangzhi Xu1,3, Qika Lin4, Lingling Zhang1,3, Rui Mao5, Erik Cambria5, Jun Liu1,3†

1School of Computer Science and Technology, Xi’an Jiaotong University, China 2MOE KLINNS Lab, Xi’an Jiaotong University, China 3Shaanxi Province Key Laboratory of Big Data Knowledge Engineering, China 4Saw Swee Hock School of Public Health, National University of Singapore, Singapore 5College of Computing and Data Science, Nanyang Technological University, Singapore zhangjian062422@stu.xjtu.edu.cn, liukeen@xjtu.edu.cn

##### Abstract

Collaborative reasoning with multiple agents offers the potential for more robust and diverse problem-solving. However, existing approaches often suffer from homogeneous agent behaviors and lack of reflective and rethinking capabilities. We propose Multi-Agent Personality Shaping ((MAPS), a novel framework that enhances reasoning through agent diversity and internal critique. Inspired by the Big Five personality theory, MAPS assigns distinct personality traits to individual agents, shaping their reasoning styles and promoting heterogeneous collaboration. To enable deeper and more adaptive reasoning, MAPS introduces a Critic agent that reflects on intermediate outputs, revisits flawed steps, and guides iterative refinement. This integration of personality-driven agent design and structured collaboration improves both reasoning depth and flexibility. Empirical evaluations across three benchmarks demonstrate the strong performance of MAPS, with further analysis confirming its generalizability across different large language models and validating the benefits of multi-agent collaboration.

Code — https://github.com/exoskeletonzj/MAPS

### Introduction

Solving complex reasoning problems (Bhattacharya et al. 2024; Li et al. 2024d; He et al. 2024) often requires more than just accurate perception and factual recall—it demands nuanced interpretation, multi-step inference, and the ability to adapt when initial attempts fail (Fu et al. 2024; Li et al. 2024b). While large language models (LLMs) demonstrate promising capabilities in solving such problems, they frequently fall short in scenarios requiring sustained reasoning, internal verification, and flexible strategy revision (Anand et al. 2024; Alasadi and Baiz 2024; Gao et al. 2024; Wang et al. 2024b; Xu et al. 2025). A key challenge lies in how to effectively combine diverse reasoning strategies and incorporate mechanisms for intermediate reflection. Prior work has explored both single-agent solutions and simple collaborative setups (e.g., paired discussion or voting) (Kaesberg

*These authors contributed equally. †Corresponding Author

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

[Figure 1]

Figure 1: An example of a multimodal scientific multiplechoice problem. The correct answer is derived based on the reasoning over inputs that include context, question, and diagram.

et al. 2025), yet these approaches often suffer from rigid behaviors and limited capacity for self-correction. This raises an important research question: how can we enhance the depth and adaptability of reasoning by enabling more flexible and reflective solution processes? Figure 1 illustrates a representative scenario that embodies these challenges, where successful problem solving requires interpreting multimodal input and applying domain-specific reasoning.

As illustrated in Figure 2, existing methods (Wang et al. 2024a; Landau, P´aez, and Bordeianu 2024; Hardiansyah et al. 2024; Zhang et al. 2024; Caffagni et al. 2024; Qiu, Yuan, and Lam 2024) for complex reasoning problems often adopt single-agent solutions or simple two-agent collaborations. Although effective to some extent, these setups suffer from homogeneous agent behaviors—reasoning steps tend to repeat similar patterns, leading to redundancy and premature convergence. The lack of diversity limits exploration and reduces the chance of identifying alternative perspectives or correcting errors, especially in multi-turn settings where roles and strategies remain undifferentiated.

Another issue is the lack of reflective and rethinking capabilities in existing approaches. As shown in Figure 2, the interaction between two agents is often linear and lacks

[Figure 2]

Figure 2: Comparison of reasoning paradigms. Single-agent and two-agent approaches offer limited adaptability. MAPS enables dynamic collaborative reasoning. Right: Built on GPT-4o, MAPS achieves the best performance across three benchmarks.

mechanisms for feedback or revision. Even with multiple turns, agents rarely revisit earlier reasoning or correct initial misconceptions. In contrast, human reasoning is inherently iterative: People reflect, reassess, and adjust their thinking over time. Without structured reflection, current methods risk premature convergence and fail to recover from earlystage errors.

To address these two issues, we propose MAPS (MultiAgent Personality Shaping), a collaborative reasoning framework that enhances both diversity and adaptability in complex problem solving. Inspired by the Big Five personality theory (Almagor, Tellegen, and Waller 1995; Benet and Waller 1995; Simms 2007), MAPS shapes the reasoning behaviors of a set of role-specialized agents through distinct personality traits, promoting heterogeneous collaboration and mitigating behavioral homogeneity. To enable reflective thinking and iterative refinement, MAPS further introduces a Critic agent that revisits intermediate outputs, identifies flawed steps, and provides structured feedback. This integration of personality-guided agent design and internal critique supports deeper, more flexible reasoning aligned with human cognitive processes.

As shown in Figure 3, the interpreter (Openness) explores information from multiple angles, the aligner (Agreeableness) reconciles visual and textual cues, the scholar (Conscientiousness) ensures precision and factual grounding, the solver (Extraversion) drives goal-oriented conclusions, and the critic (Neuroticism) questions assumptions and detects flaws. The critic is further inspired by Socratic questioning (Elder and Paul 1998; Paul and Elder 2019), providing reflective feedback throughout the multi-stage reasoning process. Together, these roles enable a structured yet flexible collaboration that supports deeper and more reliable problem solving.

We conduct extensive experiments on three challenging benchmarks: MathVista (Lu et al. 2023), OlympiadBench (He et al. 2024), and EMMA (Hao et al. 2025). These datasets cover a wide range of complex reasoning tasks. MAPS consistently outperforms baseline methods across all datasets, confirming its effectiveness in enhancing both rea-

[Figure 3]

Figure 3: The corresponding relation between the Big Five Personality theory and the five function-specific agents.

soning depth and adaptability. We further evaluate MAPS under different base LLMs, and results consistently show its superiority across model backbones. Additional analyses examine the impact of the feedback mechanism and the overall efficiency of the framework.

##### Our main contributions are as follows:

- • We propose MAPS, a multi-agent reasoning framework. To the best of our knowledge, this is the first work that incorporates personality shaping based on the Big Five theory into collaborative reasoning.
- • MAPS addresses two key challenges in existing methods: behavioral homogeneity and lack of reflection. It assigns distinct personality traits to agents and introduces a Critic inspired by Socratic questioning.
- • We conduct extensive experiments on three scientific reasoning benchmarks. MAPS achieves consistent performance gains (up to 15.84%) and generalizes well across different tasks and base models.

### Methodology

This section introduces MAPS in four key components: Preliminaries, Agentic Interaction Logic, the Four-Step Reasoning process, and the Critic and Feedback mechanism.

[Figure 4]

- Figure 4: The overall architecture of MAPS. The framework consists of five functional agents inspired by the Big Five personality theory. The core reasoning process is carried out by four specialized agents—Interpreter, Aligner, Scholar, and Solver—each responsible for a distinct stage in solving complex reasoning problems. Finally, the Critic agent provides reflective feedback and correction to enhance accuracy and interpretability.

#### Preliminaries

Task Definition. We define the complex reasoning task as a mapping from structured multimodal inputs to a target answer space. Let D, C, and Q denote the input spaces of diagrams, contexts, and questions, respectively, and let A denote the answer space. Each instance is a triplet (di,ci,qi) ∈ D × C × Q, and the goal is to predict the corresponding answer ai ∈ A.

The task is thus modeled as a function:

M : D × C × Q → A, ai = M(di,ci,qi), (1)

where M denotes the reasoning system responsible for processing visual-textual input and generating the output.

Agent-Based Modeling. In MAPS, the reasoning process M is decomposed into a sequence of collaborative stages executed by a set of role-specialized agents {A1,A2,...,AK}, where each agent Ak performs a specific function conditioned on both the original input and intermediate reasoning states.

Let S0 = (di,ci,qi) be the initial input state. The overall reasoning unfolds as a staged transformation:

Sk = Ak(Sk−1), for k = 1,2,...,K, (2)

where Sk denotes the intermediate reasoning state after the k-th agent’s operation. The final answer is extracted as ai = Extract(SK). This formulation reflects the staged reasoning in MAPS, where specialized agents collaboratively refine the solution.

#### Agents Interaction Logic

As illustrated in Algorithm 1, this section introduces the interaction logic among the five personality-driven agents

in MAPS. Complex problem reasoning requires multimodal semantic integration and structured inference over multiple steps. MAPS models this process as a functional composition of role-specialized reasoning agents, each shaped by a distinct personality trait.

Let x = (di,ci,qi) be the input, and let pk ∈ Rm denote the personality embedding associated with agent Ak. The entire collaborative reasoning process is represented as:

ai = F(x;p1,...,p4) = A4 ◦ A3 ◦ A2 ◦ A1(x), (3)

where each agent Ak(·;pk) executes its stage conditioned on its personality vector pk, contributing a unique reasoning perspective.

After the initial reasoning trajectory T =

{S1,S2,S3,S4} is generated, the Critic agent performs a reflective evaluation by applying a feedback function:

f = Reflect(T ), f = {f1,f2,f3,f4} ∈ R4, (4)

where each score fk reflects the confidence of the kth stage. If any fk < τ, the corresponding stage is revised. This feedback loop complements the forward reasoning path, allowing MAPS to emulate deliberative human cognition through structured collaboration and critique.

Proposition 1 (Monotonic Free-Energy Descent). For the free energy F(t) = Eq(t)[−log p(x,ai | θ)] + KL q(t)∥ p , each Critic-triggered update satisfies

F(t+1) ≤ F(t). (5)

Thus the MAPS iteration produces a non-increasing freeenergy sequence that converges to a stationary point. The full proof is provided in Appendix A.1.

Algorithm 1: MAPS Collaborative Reasoning with Reflective Feedback

- 1: Input: Diagram di, Context ci, Question qi, Personality vectors {p1,...,p4}
- 2: Initialize: S0 = (di,ci,qi), t ← 0
- 3: repeat
- 4: Interpreter: pi = A1(S0;p1)
- 5: Aligner: li = A2(pi,ci,qi;p2)
- 6: Scholar: si = A3(li,pi,ci,qi;p3)
- 7: Solver: ai = A4(si,li,pi;p4)
- 8: Reasoning Trajectory: T (t) = {pi,li,si,ai}
- 9: Critic: f(t) = Reflect(T (t)), fk ∈ [0,1]
- 10: if all fk ≥ τ then
- 11: break
- 12: else
- 13: Identify stage k∗ = arg mink fk
- 14: Rerun agent Ak∗ with updated input
- 15: t ← t + 1
- 16: end if
- 17: until convergence
- 18: return Final answer ai

#### Four-Step Reasoning

Given input x = (di,ci,qi), MAPS conducts a structured reasoning process across four personality-driven agents: In-

terpreter, Aligner, Scholar, and Solver. Each agent Ak is parameterized by a personality embedding pk ∈ Rm, shaping its reasoning style and focus. The entire inference pipeline can be expressed as a function composition:

ai = A4 ◦ A3 ◦ A2 ◦ A1(x;p1,p2,p3,p4), (6)

where each intermediate state Sk is defined recursively as Sk = Ak(Sk−1;pk), with S0 = x.

Interpreter. The Interpreter agent aims to extract structured visual semantics from diagram di, translating them into a caption representation pi that can be consumed by downstream language agents. Let ϕvis(di) be a visual encoder, and ψlang(·) a caption generator. Then the agent performs:

pi = A1(di;p1) = ψlang ϕvis(di) + W1p1 , (7)

where W1 ∈ Rd×m projects the personality embedding into the encoder space, modulating its attention to attributes like spatiality, color, or topology.

Aligner. To resolve semantic mismatches across modalities, the Aligner fuses the interpreted diagram caption pi with the textual context ci and question qi. The process outputs an alignment representation li, optimized to preserve shared semantics and suppress modality conflict. Formally,

li = A2(pi,ci,qi;p2) = CrossFuse pi,ci,qi;p2 , (8) where CrossFuse(·) denotes a multi-head attention-based

fusion operator, adaptively weighted by p2 to emphasize visual or linguistic cues based on agent bias.

Scholar. While li captures semantic consistency, complex reasoning often requires external knowledge supplementation. The Scholar agent retrieves and integrates domainspecific knowledge K(li), such as physics principles or mathematical theorems. We define:

si = A3(li,pi,ci,qi;p3) = KnowAug li,K(li);p3 , (9)

where KnowAug(·) augments contextual embeddings with retrieved tuples K(li) from a structured knowledge memory, and p3 biases the agent toward formal rigor or heuristic reasoning.

Solver. The Solver agent aggregates all upstream outputs and executes logical composition to generate the final answer ai. Let Hi = {pi,li,si} be the hybrid reasoning state. The solver computes:

ai = A4(Hi;p4) = Deduct pi,li,si;p4 , (10)

where Deduct(·) is a constrained generation module that synthesizes the inputs under logical, numerical, or symbolic rules. The final prediction ai ∈ A may be a selected option or a free-form answer. Together, these four stages construct a trajectory T = {pi,li,si,ai}, on which the Critic agent operates for reflection and feedback.

Proposition 2 (Collaborative Information Bottleneck). Let Sk be the intermediate output of the k-th agent, given input x = (di,ci,qi) and target answer ai. Then the MAPS reasoning process optimizes

4

I(x;Sk) s.t. I(Sk;ai) ≥ ε, (11)

min

k=1

where I(·;·) denotes mutual information. The Critic monitors constraint violations and reactivates stages with insufficient task-relevant information.

A derivation and discussion are provided in Appendix A.2.

#### Critic and Feedback

The Critic agent evaluates the internal reasoning trajectory T = {pi,li,si,ai} without relying on ground-truth answers. Inspired by Socratic questioning, it examines each stage’s logic and justification to identify flawed assumptions and incomplete inferences.

We define the feedback vector as:

fi = Mcrit(pi,li,si,ai), fi ∈ [0,1]4, (12)

where each element fi(k) represents the Critic’s confidence in the correctness and completeness of stage k.

The weakest stage is selected by:

∗)

fi(k), if f(k

k∗ = arg min

i < τ, (13)

k

which triggers a targeted revision. This reflection-driven loop promotes iterative self-correction and deepens reasoning reliability.

MathVista EMMA OlympiadBench

Models CoT

Avg. Gen. Math Avg. Math Phy. Chem. Avg. MECO MZCE MZCO PECO PZCE Avg.

Random Choice - 26.09 22.78 24.30 13.00 23.00 27.00 21.00 0.67 0.33 0.00 1.75 0.33 0.87 16.06 Human Expert - 56.09 55.74 55.90 75.00 64.50 86.00 75.17 48.00 34.67 30.36 54.17 12.33 37.80 52.73 Claude 3.5 Sonnet - 68.04 63.15 65.40 23.00 34.00 44.00 33.67 20.67 13.00 10.71 10.75 14.00 13.23 37.43 Gemini 2.0 Flash - 70.65 70.93 70.80 20.00 40.00 36.00 32.00 8.00 5.67 7.14 3.07 7.00 5.39 36.06 GPT-4o - 65.22 61.30 63.10 30.00 38.00 33.00 33.67 23.33 20.33 19.64 22.15 21.00 21.47 39.41 Qwen2.5-VL-72B - 70.65 67.41 68.90 42.00 42.00 38.00 40.67 18.00 12.33 5.36 7.24 3.67 8.80 39.45 InternVL2.5-8B-MPO - 64.78 60.74 62.60 30.00 40.00 38.00 36.00 10.67 6.67 10.71 1.10 0.67 3.88 34.16 LLaVA-Onevision-72B - 62.83 58.52 60.50 25.00 32.00 24.00 27.00 6.67 7.33 3.57 3.29 9.67 6.18 31.23 Claude 3.5 Sonnet ✓ 71.74 64.26 67.70 30.00 38.00 41.00 36.33 24.00 11.00 16.07 12.72 10.33 13.23 39.09 Gemini 2.0 Flash ✓ 70.22 75.56 73.10 24.00 41.00 36.00 33.67 12.67 6.33 3.57 4.61 2.33 5.39 37.38 GPT-4o ✓ 65.22 62.59 63.80 27.00 44.00 35.00 35.33 25.33 21.67 12.50 24.12 20.33 22.27 40.47 Qwen2.5-VL-72B ✓ 71.09 77.96 74.80 38.00 36.00 37.00 37.00 23.33 13.00 10.71 8.11 1.33 9.59 40.46 InternVL2.5-8B-MPO ✓ 60.87 67.41 64.40 31.00 36.00 24.00 30.33 12.00 8.33 1.79 2.85 0.99 4.75 33.16 LLaVA-Onevision-72B ✓ 71.09 64.44 67.50 23.00 26.00 23.00 24.00 11.33 8.67 5.36 4.82 3.33 6.18 32.56 MAPS (GPT-4obase) - 75.87 83.15 79.80 52.00 71.00 51.00 58.00 46.00 30.33 32.14 28.51 28.33 31.14 56.31

Table 1: Performance across 10 subtasks. Gen. = General (MathVista), Phy./Chem. = Physics/Chemistry (EMMA), MECO/ MZCO/ MZCE = English/Chinese COMP & CEE Math (OlympiadBench), PECO/ PZCE = English/Chinese Physics (OlympiadBench).

### Experiments

#### Datasets and Baselines

We evaluate on three benchmarks for complex problem reasoning: MathVista, OlympiadBench, and EMMA. MathVista covers math and general science, OlympiadBench focuses on high-level math and physics, and EMMA includes math, physics, and chemistry. See Appendix B for details.

We use GPT-4o (Achiam et al. 2023) as the base model. For comparison, we include leading multimodal large language models (MLLMs), both proprietary and open-source, tested under both direct and Chain-of-Thought (CoT) settings. Experimental details and baseline setups are provided in Appendix C.

#### Main Results

MAPS achieves a new state-of-the-art (SOTA), surpassing human-level performance for the first time. As shown in Table 1, MAPS outperforms previous SOTA models by 15.84% and exceeds human expert performance by 3.58% across all tasks, highlighting its strength in solving complex multimodal problems. MAPS demonstrates robust performance across mathematical, physical, chemical, and general tasks, showcasing strong interdisciplinary reasoning. Its multi-agent design, based on the Big Five personality theory, enables effective collaboration and contributes to the SOTA results. The system excels in multimodal semantic integration and multi-step reasoning by jointly leveraging diagrams, contexts, and questions. Furthermore, the Critic agent applies Socratic feedback to refine responses, enhancing both accuracy and reliability on challenging tasks.

MAPS exhibits strong adaptability and robustness across diverse reasoning tasks. The evaluation datasets span a wide range of question types, modalities, and reasoning difficulties. MathVista features judgment, multiplechoice, and open-ended fill-in-the-blank questions with varied answer formats, requiring accurate intent understanding and response generation. OlympiadBench emphasizes challenging open-ended problems demanding multi-step symbolic reasoning, where small errors can lead to divergent outcomes. EMMA introduces multimodal complexity with diagrams embedded in both questions and answer choices. Through feedback-driven multi-agent collaboration and Socratic questioning, MAPS effectively handles these challenges, achieving SOTA and demonstrating strong generalization across heterogeneous reasoning scenarios.

#### Analysis of Critic Agent

Critic schema includes scoring and Socratic feedback. Each reasoning step is rated from 0–5, guiding whether the system should backtrack or proceed. Feedback is heuristic, encouraging rethinking rather than offering direct answers. The Critic uses these scores to decide whether to regenerate specific steps, ensuring robustness in the final answer.

The Critic enhances reasoning via Socratic feedback without using gold labels. As shown in Figure 5, the top illustrates the feedback schema, while the bottom shows feedback proportions across three datasets. The Critic prompts reflection and correction by encouraging agents to question assumptions rather than passively accept reasoning.

Solver receives the most feedback in EMMA and OlympiadBench. As shown in the lower half of Figure 5,

[Figure 5]

- Figure 5: The schema of the Critic agent, as well as the feedback and backtracking situations of the Critic agent across different datasets.

feedback varies by dataset. For MathVista, most steps need no regeneration, aligning with our superior 5.0% SOTA improvement in Table 1. This reflects strong baseline reasoning. In contrast, EMMA and OlympiadBench show the highest feedback for the Solver, especially in the interpretation, alignment, and integration steps. These are the most complex and error-prone stages. Other agents receive comparable and lower feedback, indicating better relative performance in their sub-tasks.

### Supplementary Analysis

Due to space limitations, generalization experiments on other datasets, such as DiagramQG (Zhang et al. 2025d), are discussed in Appendix D. The case study and complete process of MAPS is outlined in Appendix E, and the prompts for agents are provided in Appendix F.

#### Ablation Studies

Ablating the Interpreter results in the greatest loss of performance. We conduct ablation experiments on the OlympiadBench dataset to evaluate the impact of each module on the overall performance. Table 2 presents the effects of removing the Interpreter, Aligner, Scholar, and Critic modules from the MAPS framework. The results show that removing the Interpreter agent causes the largest performance degradation, at 16.09%. This is because, in complex problem reasoning, diagrams contain a wealth of valuable information, which serves as an important supplement to the text. Understanding diagrams plays a crucial role in problem-solving.

The removal of the Critic agent causes the smallest performance loss. It results in only a 7.05% decrease, underscoring its role in providing feedback and corrections. While this mechanism allows MAPS to backtrack and refine its reasoning, its impact is less than that of other agents. Removing the Scholar agent results in 11.49% performance drops, highlighting the importance of searching and integrating domain-specific knowledge. Finally, the removal of the Aligner agent causes a 10.86% drop, indicating that

Variation MECO MZCE MZCO PECO PZCE Avg. MAPS 46.00 30.33 32.14 28.51 28.33 31.14 w/oInterpreter 25.33 16.67 10.71 21.05 11.62 15.05

∆ (-20.67) (-13.66) (-21.43) (-7.46) (-16.71) (-16.09) w/oAligner 28.00 17.67 16.07 20.83 19.00 20.28

∆ (-18.00) (-12.66) (-16.07) (-7.68) (-9.33) (-10.86) w/oScholar 28.00 16.33 30.36 19.96 16.33 19.65

∆ (-18.00) (-14.00) (-1.78) (-8.55) (-12.00) (-11.49) w/oCritic 34.67 21.67 30.36 23.03 21.67 24.09

∆ (-11.33) (-8.66) (-2.42) (-5.48) (-6.66) (-7.05)

Table 2: Performance under different ablation settings are analyzed. We perform ablation experiments on the solving module w/oInterpreter, w/oAligner, w/oScholar or w/oCritic modules to evaluate the impact of removing these components.

while diagram and context alignment is valuable, its effect is smaller compared to other components.

#### Base Model Generalization

MAPS improves performance across diverse base models. We conduct experiments to verify whether our MAPS framework demonstrates robust generalization across various base LLMs. The results confirm MAPS’s robustness and transferability, highlighting its adaptability and consistent performance across different foundation models. To further validate its generalization, we evaluate both Qwen2.5-VL72B and Gemini 2.0 Flash, showing that MAPS performs well across models of varying scales and capabilities. Figure 6 presents results for three sets of base models. In each set, we compare MLLMs and MAPS on mathematical, physical, and chemical sub-tasks. MAPS consistently outperforms the base models. For example, MAPSQwen improves Qwen2.5-VL-72B by 12.4% in physics, while MAPSGemini improves Gemini by 4.2%. Similar gains are observed in math and chemistry, demonstrating MAPS’s effectiveness on both open-source and closed-source MLLMs.

#### Time Efficiency

Simpler formats and lower difficulty yield faster solving times. Solving time efficiency varies by question type, answer type, category, and difficulty, with multiple-choice and integer-type questions being the fastest, while higher difficulties and complex formats require more time. Figure 7 illustrates the solving time efficiency across various dimensions—question types, answer formats, subject categories, and difficulty levels—with all times normalized to a 100s benchmark.

Predefined structure and conceptual simplicity reduce reasoning time. Multiple-choice questions are solved more quickly thanks to predefined answer options that limit the need for extensive reasoning or exploration. Integertype answers also show high efficiency, often tied to simpler arithmetic or structured formats requiring minimal inference. General category questions are faster on average, likely due to lower conceptual and reasoning complexity

[Figure 6]

- Figure 6: Performance Comparison of MAPS on Math, Physics, and Chemistry Subtasks in the EMMA Dataset with GPT-4o, Gemini, and Qwen2.5-VL-72B as Bases.

compared to domain-specific tasks. In contrast, open-ended questions demand deeper analysis and justification, leading to longer solving times. Finally, solving efficiency declines with increased difficulty: as question complexity rises, so does the required reasoning time, reflecting greater cognitive and computational demands.

### Related Works

The related work is structured into two main aspects: Firstly, an introduction to complex problem reasoning; Secondly, an exploration of multi-agent techniques.

Complex Problem Reasoning. The research of complex problem reasoning spans across multiple fields, including mathematics, physics, and chemistry, with each area focusing on enhancing problem-solving abilities. In mathematics, studies (Didolkar et al. 2024; Fitriana and Waswa 2024; Tong et al. 2025) explore various methods to improve mathematical problem-solving, such as algorithm optimization, educational strategies, and the use of artificial intelligence. These approaches aim to boost the efficiency, accuracy, and depth of mathematical reasoning. In the field of physics, the papers (Mustofa, Bilad, and Grendis 2024; Kapuriya et al. 2024; Anand et al. 2024; Wu et al. 2024) emphasize the integration of different information types, such as images and text, through multimodal learning to enhance the efficiency and precision of problem solving. In chemistry, three articles (Alasadi and Baiz 2024; Kiernan, Manches, and Seery 2024; Li et al. 2024c; Lin et al. 2025; Dang et al. 2025) investigate the role of multimodal learning in solving chemical problems. By combining diverse information sources, including images and text, and employing techniques such as generative models and molecular geometry reasoning, they aim to improve both the efficiency and accuracy of solving chemistry problems.

Multi-Agent. Multi-agent systems, built on LLMs, consist of multiple AI agents that specialize in specific tasks,

[Figure 7]

Figure 7: An analysis of the solving time efficiency across different question types, answer types, question categories, and question difficulties.

working together to solve complex problems (Richards 2023; Yang, Yue, and He 2023; Wu and et al 2023; Sun

- et al. 2023; Zhang et al. 2025a,b,c; Yan et al. 2025). When presented with a problem, these agents decompose it into smaller, manageable subtasks and utilize various tools, such as internet data retrieval, to solve them through iterative steps. Several studies (Poldrack, Lu, and Beguˇs 2023; Wang
- et al. 2024c; Xi et al. 2025; Ni and Gao 2021) have employed multi-agent systems to tackle challenges like problem identification, code writing and debugging, data visualization, and providing interactive feedback to human users. In their work, Ni and Buehler (2024) highlights the potential of AIdriven multi-agent teams in solving mechanical problems autonomously, demonstrating an enhanced capability for understanding, formulating, and validating engineering solutions through self-correction and collaborative refinement. Inspired by the research, we developed the MAPS method, which leverages multi-agent collaborative learning and stepwise problem-solving to provide innovative solutions for complex problem reasoning. By combining the strengths of AI agents, complex problems can be broken down into subtasks and solved step by step through collaboration, improving efficiency and accuracy.

### Conclusion

This study presents MAPS, a multi-agent framework grounded in the Big Five Personality Theory and guided by Socratic principles, designed to address the challenges of multimodal comprehensive reasoning and enhance reflective capabilities. The framework involves five agents, each specializing in distinct aspects of problem-solving. To address the first challenge, a four-agent strategy is proposed, where each agent focuses on specific stages of the reasoning process. Additionally, the Critic agent addresses the second

challenge through Socratic reflection and critical feedback. Extensive experiments on the EMMA, OlympiadBench, and MathVista datasets validate MAPS’s effectiveness in overcoming these issues and enhancing performance across various reasoning tasks. Meanwhile, we perform additional analytical experiments to assess the model’s advancement as well as its generalization.

### Acknowledgments

This work was supported by the National Natural Science Foundation of China (No. 62137002, 62277042, 62293553, 62450005, 62437002, 62477036, 62477037, 62176209, 62192781, 62306229), the “LENOVO-XJTU” Intelligent Industry Joint Laboratory Project, the Shaanxi Provincial Social Science Foundation Project (No. 2024P041), the Natural Science Basic Research Program of Shaanxi (No. 2023JC-YB-593), and the Youth Innovation Team of Shaanxi Universities “Multi-modal Data Mining and Fusion”.

### References

Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Alasadi, E. A.; and Baiz, C. R. 2024. Multimodal generative artificial intelligence tackles visual problems in chemistry. Journal of Chemical Education, 101(7): 2716–2729.

Almagor, M.; Tellegen, A.; and Waller, N. G. 1995. The Big Seven model: A cross-cultural replication and further exploration of the basic dimensions of natural language trait descriptors. Journal of personality and social psychology, 69(2): 300.

Anand, A.; Kapuriya, J.; Singh, A.; Saraf, J.; Lal, N.; Verma, A.; Gupta, R.; and Shah, R. 2024. Mm-phyqa: Multimodal physics question-answering with multi-image cot prompting. In Pacific-Asia Conference on Knowledge Discovery and Data Mining, 53–64. Springer.

Bai, J.; Bai, S.; Chu, Y.; Cui, Z.; Dang, K.; Deng, X.; Fan,

- Y.; Ge, W.; Han, Y.; Huang, F.; et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Benet, V.; and Waller, N. G. 1995. The Big Seven factor model of personality description: Evidence for its crosscultural generality in a Spanish sample. Journal of Personality and Social Psychology, 69(4): 701.

Bhattacharya, M.; Pal, S.; Chatterjee, S.; Lee, S.-S.; and Chakraborty, C. 2024. Large language model to multimodal large language model: A journey to shape the biological macromolecules to biological sciences and medicine. Molecular Therapy-Nucleic Acids, 35(3).

Caffagni, D.; Cocchi, F.; Barsellotti, L.; Moratelli, N.; Sarto, S.; Baraldi, L.; Cornia, M.; and Cucchiara, R. 2024. The revolution of multimodal large language models: a survey.

- arXiv preprint arXiv:2402.12451.

Chen, Z.; Wang, W.; Cao, Y.; Liu, Y.; Gao, Z.; Cui, E.; Zhu, J.; Ye, S.; Tian, H.; Liu, Z.; et al. 2024. Expanding

performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271.

Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang, D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Dang, Z.; Luo, M.; Wang, J.; Jia, C.; Han, H.; Wan, H.; Dai, G.; Chang, X.; and Wang, J. 2025. Disentangled noisy correspondence learning. IEEE Transactions on Image Processing.

Didolkar, A.; Goyal, A.; Ke, N. R.; Guo, S.; Valko, M.; Lillicrap, T.; Rezende, D.; Bengio, Y.; Mozer, M.; and Arora, S. 2024. Metacognitive capabilities of llms: An exploration in mathematical problem solving. arXiv preprint arXiv:2405.12205.

Elder, L.; and Paul, R. 1998. The role of Socratic questioning in thinking, teaching, and learning. The Clearing House, 71(5): 297–301.

Fitriana, H.; and Waswa, A. N. 2024. The influence of a realistic mathematics education approach on students’ mathematical problem solving ability. Interval: Indonesian Journal of Mathematical Education, 2(1): 29–35.

Fu, D.; Guo, R.; Khalighinejad, G.; Liu, O.; Dhingra, B.; Yogatama, D.; Jia, R.; and Neiswanger, W. 2024. Isobench: Benchmarking multimodal foundation models on isomorphic representations. arXiv preprint arXiv:2404.01266.

Gao, T.; Chen, P.; Zhang, M.; Fu, C.; Shen, Y.; Zhang, Y.; Zhang, S.; Zheng, X.; Sun, X.; Cao, L.; et al. 2024. Cantor: Inspiring multimodal chain-of-thought of mllm. In Proceedings of the 32nd ACM International Conference on Multimedia, 9096–9105.

Hao, Y.; Gu, J.; Wang, H. W.; Li, L.; Yang, Z.; Wang, L.; and Cheng, Y. 2025. Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark. arXiv preprint arXiv:2501.05444.

Hardiansyah, F.; Armadi, A.; AR, M. M.; and Wardi, M. 2024. Analysis of field dependent and field independent cognitive styles in solving science problems in elementary schools. Jurnal Penelitian Pendidikan IPA, 10(3): 1159– 1166.

He, C.; Luo, R.; Bai, Y.; Hu, S.; Thai, Z. L.; Shen, J.; Hu, J.; Han, X.; Huang, Y.; Zhang, Y.; et al. 2024. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008.

Kaesberg, L. B.; Becker, J.; Wahle, J. P.; Ruas, T.; and Gipp, B. 2025. Voting or consensus? Decision-making in multiagent debate. arXiv preprint arXiv:2502.19130.

Kapuriya, J.; Kirtani, C.; Singh, A.; Saraf, J.; Lal, N.; Kumar, J.; Shivam, A. R.; Verma, A.; Anand, A.; and Shah, R. R. 2024. Mm-phyrlhf: Reinforcement learning framework for multimodal physics question-answering. arXiv preprint arXiv:2404.12926.

Kevian, D.; Syed, U.; Guo, X.; Havens, A.; Dullerud, G.; Seiler, P.; Qin, L.; and Hu, B. 2024. Capabilities of large language models in control engineering: A benchmark study on gpt-4, claude 3 opus, and gemini 1.0 ultra. arXiv preprint arXiv:2404.03647.

Kiernan, N. A.; Manches, A.; and Seery, M. K. 2024. Resources for reasoning of chemistry concepts: multimodal molecular geometry. Chemistry Education Research and Practice, 25(2): 524–543.

Landau, R. H.; P´aez, M. J.; and Bordeianu, C. C. 2024. Computational physics: Problem solving with Python. John Wiley & Sons.

Li, B.; Zhang, Y.; Guo, D.; Zhang, R.; Li, F.; Zhang, H.; Zhang, K.; Zhang, P.; Li, Y.; Liu, Z.; et al. 2024a. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Li, J.; Lu, W.; Fei, H.; Luo, M.; Dai, M.; Xia, M.; Jin, Y.; Gan, Z.; Qi, D.; Fu, C.; et al. 2024b. A survey on benchmarks of multimodal large language models. arXiv preprint arXiv:2408.08632.

Li, J.; Zhang, D.; Wang, X.; Hao, Z.; Lei, J.; Tan, Q.; Zhou, C.; Liu, W.; Yang, Y.; Xiong, X.; et al. 2024c. Chemvlm: Exploring the power of multimodal large language models in chemistry area. arXiv preprint arXiv:2408.07246.

Li, L.; Wang, Y.; Xu, R.; Wang, P.; Feng, X.; Kong, L.; and Liu, Q. 2024d. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models.

- arXiv preprint arXiv:2403.00231. Lin, Q.; Zhu, Y.; Pu, B.; Huang, L.; Luo, H.; Ma, J.; Peng,

- Z.; Zhao, T.; Xu, F.; Zhang, J.; et al. 2025. A Foundation Model for Chest X-ray Interpretation with Grounded Reasoning via Online Reinforcement Learning. arXiv preprint arXiv:2509.03906. Lu, P.; Bansal, H.; Xia, T.; Liu, J.; Li, C.; Hajishirzi, H.; Cheng, H.; Chang, K.-W.; Galley, M.; and Gao, J.

2023. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255.

Mustofa, H. A.; Bilad, M. R.; and Grendis, N. W. B. 2024. Utilizing AI for physics problem solving: a literature review and ChatGPT experience. Lensa: Jurnal Kependidikan Fisika, 12(1): 78–97.

Ni, B.; and Buehler, M. J. 2024. MechAgents: Large language model multi-agent collaborations can solve mechanics problems, generate new data, and integrate knowledge. Extreme Mechanics Letters, 67: 102131.

Ni, B.; and Gao, H. 2021. A deep learning approach to the inverse problem of modulus identification in elasticity. Mrs Bulletin, 46: 19–25.

Paul, R.; and Elder, L. 2019. The thinker’s guide to Socratic questioning. Rowman & Littlefield.

Poldrack, R. A.; Lu, T.; and Beguˇs, G. 2023. AIassisted coding: Experiments with GPT-4. arXiv preprint arXiv:2304.13187.

Qiu, J.; Yuan, W.; and Lam, K. 2024. The application of multimodal large language models in medicine. The Lancet Regional Health–Western Pacific, 45.

Richards, T. B. 2023. Auto-GPT: An experimental opensource attempt to make GPT-4 fully autonomous.

Simms, L. J. 2007. The Big Seven model of personality and its relevance to personality pathology. Journal of Personality, 75(1): 65–94.

Sun, Q.; Yin, Z.; Li, X.; Wu, Z.; Qiu, X.; and Kong, L. 2023. Corex: Pushing the boundaries of complex reasoning through multi-model collaboration. arXiv preprint arXiv:2310.00280.

Tishby, N.; Pereira, F. C.; and Bialek, W. 2000. The information bottleneck method. arXiv:physics/0004057.

Tong, Y.; Zhang, X.; Wang, R.; Wu, R.; and He, J. 2025. Dart-math: Difficulty-aware rejection tuning for mathematical problem-solving. Advances in Neural Information Processing Systems, 37: 7821–7846.

- Wang, K. D.; Burkholder, E.; Wieman, C.; Salehi, S.; and Haber, N. 2024a. Examining the potential and pitfalls of ChatGPT in science and engineering problem-solving. In Frontiers in Education, volume 8, 1330486. Frontiers Media SA.
- Wang, L.; Hu, Y.; He, J.; Xu, X.; Liu, N.; Liu, H.; and Shen, H. T. 2024b. T-sciq: Teaching multimodal chain-of-thought reasoning via large language model signals for science question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, 17, 19162–19170.

Wang, L.; Ma, C.; Feng, X.; Zhang, Z.; Yang, H.; Zhang, J.; Chen, Z.; Tang, J.; Chen, X.; Lin, Y.; et al. 2024c. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6): 186345.

Wu, Q.; and et al. 2023. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155.

Wu, W.; Zhang, L.; Liu, J.; Tang, X.; Wang, Y.; Wang, S.; and Wang, Q. 2024. E-gps: Explainable geometry problem solving via top-down solver and bottom-up generator. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13828–13837.

Xi, Z.; Chen, W.; Guo, X.; He, W.; Ding, Y.; Hong, B.; Zhang, M.; Wang, J.; Jin, S.; Zhou, E.; et al. 2025. The rise and potential of large language model based agents: A survey. Science China Information Sciences, 68(2): 121101.

Xu, F.; Lin, Q.; Han, J.; Zhao, T.; Liu, J.; and Cambria, E. 2025. Are large language models really good logical reasoners? a comprehensive evaluation and beyond. IEEE Transactions on Knowledge and Data Engineering.

Yan, H.; Xu, F.; Xu, R.; Li, Y.; Zhang, J.; Luo, H.; Wu, X.; Tuan, L. A.; Zhao, H.; Lin, Q.; et al. 2025. Mur: Momentum uncertainty guided reasoning for large language models. arXiv preprint arXiv:2507.14958.

Yang, H.; Yue, S.; and He, Y. 2023. Auto-gpt for online decision making: Benchmarks and additional opinions. arXiv preprint arXiv:2306.02224.

Zhang, D.; Yu, Y.; Dong, J.; Li, C.; Su, D.; Chu, C.; and Yu, D. 2024. Mm-llms: Recent advances in multimodal large language models. arXiv preprint arXiv:2401.13601.

Zhang, J.; Wang, Z.; Wang, Z.; Zhang, X.; Xu, F.; Lin, Q.; Mao, R.; Cambria, E.; and Liu, J. 2025a. MAPS: A MultiAgent Framework Based on Big Seven Personality and Socratic Guidance for Multimodal Scientific Problem Solving. arXiv preprint arXiv:2503.16905.

Zhang, J.; Wang, Z.; Zhu, H.; Liu, J.; Lin, Q.; and Cambria, E. 2025b. MARS: A Multi-Agent Framework Incorporating Socratic Guidance for Automated Prompt Optimization. arXiv preprint arXiv:2503.16874.

Zhang, J.; Wei, B.; Qi, S.; Liu, J.; Lin, Q.; et al. 2025c. GKG-LLM: A Unified Framework for Generalized Knowledge Graph Construction. arXiv preprint arXiv:2503.11227. Zhang, X.; Zhang, L.; Wu, Y.; Huang, M.; Wu, W.; Li, B.; Wang, S.; Fernando, B.; and Liu, J. 2025d. Diagram-Driven Course Questions Generation. arXiv:2411.17771.

### Proof of Proposition

#### Proof of Proposition 1 (Monotonic Free-Energy Descent)

Statement recap. Let F(t) denote the variational free energy after the t-th MAPS iteration:

F(t) = Eq(t)(θ)[−log p(x,ai | θ)] + KL(q(t)(θ)∥p(θ)),

(14) where q(t)(θ) is the approximate posterior induced by the agent outputs at iteration t. The proposition claims that each Critic-guided update ensures

F(t+1) ≤ F(t), (15)

which implies that {F(t)}∞t=0 is a non-increasing sequence that converges.

Step 1: Free Energy as a Variational Upper Bound. By standard variational inference, the marginal log-likelihood satisfies

log p(x,ai) = L(q) + KL(q(θ)∥p(θ | x,ai)), (16) where the ELBO is defined as

L(q) = Eq(θ)[log p(x,ai | θ)] − KL(q(θ)∥p(θ)). (17)

This leads to the variational free energy F(q) := −L(q), which upper-bounds the negative log marginal likelihood:

F(q) ≥ −log p(x,ai). (18)

Step 2: Critic-Guided Local Update. Suppose the Critic identifies the weakest stage k∗ = arg mink fk(t) and reexecutes Ak∗, yielding updated output Sk(t∗+1). The updated trajectory leads to a refined posterior:

q(t+1)(θ) = G(q(t)(θ),Sk(t∗+1)), (19)

where G represents the update mechanism that propagates changes forward through dependent modules.

As the re-computed stage corrects suboptimal reasoning or improves knowledge alignment, it leads to a strictly better or equal ELBO, and thus

F(t+1) = F(q(t+1)) ≤ F(q(t)) = F(t). (20)

Step 3: Convergence. Since F(t) is lower-bounded and non-increasing, the sequence converges:

F(t) = F(∞). (21)

lim

t→∞

#### Proof of Proposition 3 (Collaborative Information Bottleneck)

Statement recap. Given input x = (di,ci,qi) and target answer ai, let Sk denote the intermediate output of the k-th agent in the MAPS reasoning process. Then the objective is to find agents that minimize cumulative information compression from input while preserving task-relevant content:

4

I(x;Sk) s.t. I(Sk;ai) ≥ ε, (22)

min

{Ak}

k=1

where I(·;·) denotes mutual information, and ε > 0 is a threshold for information sufficiency.

Step 1: Motivation from the Information Bottleneck (IB) Principle. In standard IB theory (Tishby, Pereira, and Bialek 2000), one seeks a compressed representation S of input x that retains information about target ai. The tradeoff objective is:

minI(x;S) subject to I(S;ai) ≥ ε. (23)

MAPS generalizes this to a multi-agent, multi-stage setting, where each agent Ak produces Sk, and the full reasoning trajectory is T = {S1,...,S4}. The objective thus becomes collaborative:

4

I(x;Sk) s.t. ∀k, I(Sk;ai) ≥ ε. (24)

min

k=1

This reflects two goals: (1) compress redundant input features per stage, and (2) preserve task-relevant evidence at each step.

Step 2: Role of the Critic as a Constraint Monitor. The Critic observes the reasoning trajectory T and computes scores f = {f1,f2,f3,f4}, where each fk implicitly estimates I(Sk;ai). If fk < τ, we assume the sufficiency constraint is violated:

I(Sk;ai) < ε ⇒ Critic triggers Ak to revise. (25) The revision process updates Sk to Sk′ , ideally increas-

ing I(Sk′ ;ai) while not increasing I(x;Sk′ ). Since the Critic only intervenes when constraints are violated, each update

maintains feasibility and possibly reduces the objective.

##### Step 3: Convergence and Validity. As each update ei-

ther preserves or improves the sufficiency I(Sk;ai), and the overall input compression is bounded below by zero, the

process converges to a locally optimal set of agents {A∗k} satisfying:

∀k, I(Sk∗;ai) ≥ ε, with minimal

I(x;Sk∗).

k

(26)

This completes the proof that the MAPS four-step process can be viewed as a constrained collaborative information bottleneck optimization.

### Datasets

This study utilizes the latest three multimodal reasoning datasets, namely MathVista, OlympiadBench, and EMMA.

MathVista. MathVista is a large-scale scientific reasoning dataset that spans two subdomains: mathematics and general, aiming to assess the comprehensive capabilities of machine learning models in solving complex scientific problems. The dataset contains 1,000 data points covering various issues across multiple disciplines, designed with varying difficulty levels to help researchers evaluate model reasoning abilities. The release of MathVista supports interdisciplinary scientific research.

Dataset ABBR. Size MathVista

General Gen. 460 Mathematics Math 540

OlympiadBench Math En COMP MECO 150 Math Zh COMP MZCO 56 Math Zh CEE† MZCE 300 Physics En COMP PECO 456 Physics Zh CEE† PZCE 300

##### EMMA

Math 100 Physics Phy. 100 Chemistry Chem. 100

Table 3: The data distribution for the MathVista, OlympiadBench, and EMMA datasets is as follows: The symbol † indicates a sample size of 300 data points. The EMMA dataset uses its MINI version. The ‘ABBR.’ column represents the abbreviations for all the tasks.

OlympiadBench. OlympiadBench consists of two subdomains, mathematics and physics, and is specifically designed for Mathematical and Physical Olympiads, featuring a wide range of challenging problems to assess models’ performance on high-level scientific tasks. The mathematics subdomain contains three difficulty levels: English competition level, Chinese competition level, and college level. The physics subdomain includes two difficulty levels: English competition level and Chinese college level. To ensure data balance, 300 samples were taken from both the Chinese college-level mathematics and physics subsets.

EMMA. EMMA is a multimodal scientific reasoning dataset covering three subdomains: mathematics, physics, and chemistry. By integrating mathematical expressions, physical formulas, and chemical symbols with natural language descriptions, it focuses on testing models’ abilities in interdisciplinary scientific reasoning. This version uses the EMMA dataset, which contains 100 data points from each subdomain (mathematics, physics, and chemistry).

### Experiment Settings and Baselines

We select GPT-4o (Achiam et al. 2023), a powerful MLLM, as our primary agent for complex problem reasoning. GPT4o not only demonstrates strong reasoning and generation capabilities across a wide range of multimodal processing tasks, but also excels in efficiently exploring multiple perspectives when faced with complex scientific domain requirements. This makes it well-suited for adaptation to various tasks and datasets within the MAPS process. We use accuracy as our primary evaluation metric to comprehensively assess the performance of different methods across diverse task scenarios. The experimental results offer a thorough comparison of the performance of MAPS against all

baseline methods. The experiments are primarily conducted on three datasets in Appendix , where we provide a detailed comparison of MAPS and the baseline models across four types of tasks: mathematics, physics, chemistry, and general tasks. Our approach achieves a new SOTA performance. To further strengthen the comparison, Appendix includes a generalization experiment conducted on the physics data subset of the DiagramQG (Zhang et al. 2025d), which further demonstrates the robustness and effectiveness of our model.

We compare MAPS with three categories of baseline methods: original baselines, direct approach-based strong baselines, and CoT-enhanced strong baselines. Specifically, (1) the original category refers to two methods: random selection and human expert selection. These methods provide two distinct original baselines—one based on randomness and the other based on authority. (2) The direct approach-based strong baselines include some of the most powerful closed-source and open-source large language models (MLLMs) currently available worldwide. These include Claude 3.5 Sonnet (Kevian et al. 2024), Gemini 2.0 Flash (Comanici et al. 2025), GPT-4o, Qwen2.5-VL72B (Bai et al. 2023), InternVL2.5-8B-MPO (Chen et al. 2024), and LLaVA-Onevision-72B (Li et al. 2024a). (3) To ensure a fair comparison, the third category of baselines builds on the second by adding Chain-of-Thought (CoT) reasoning, which aims to enhance the capabilities of the strong MLLMs from the second category.

### Generalization Experiments

To further validate the generalization ability of MAPS, we conducted detailed experiments on the physical subset of the DiagramQG dataset. The main goal of these experiments was to compare the performance of MAPS with its base model, GPT-4o, particularly focusing on how it performed across different question categories. The experimental results, presented in Figure 8, show that MAPS outperformed GPT-4o in multiple aspects. These experiments clearly highlight the strong adaptability of MAPS across different subsets of the dataset.

Across various question categories in the DiagramQG physical dataset, MAPS achieved a performance boost, with the maximum improvement reaching 19.51% and an overall improvement of 7.71%. These results not only demonstrate the superiority of MAPS but also indicate that its Five Personality Agents architecture has a strong generalization ability, enabling it to maintain excellent performance across different datasets and tasks. This provides strong support for the practical application of MAPS, showcasing its potential in tackling complex tasks.

### Full Process of MAPS

Figure 9 illustrates the four-step solving process along with the feedback process from the Interpreter, Aligner, Scholar, and Critic agents, using a multimodal physics problem from the EMMA dataset. In this specific example, we can observe that the Interpreter first interprets the diagram, followed by the Aligner aligning the diagram with the context, question,

[Figure 8]

Figure 8: The generalization experiments conducted on the DiagramQG physical dataset, which are based on the GPT-4o base model and the incremental part of MAPS.

and options, ensuring consistency and completeness of the information. Then, the Scholar agent retrieves and supplements domain-specific knowledge to fill in the necessary expertise. Finally, the Solver completes the solving process, and the Critic agent provides feedback and corrections to ensure the accuracy and effectiveness of each step. Each step is closely connected, from understanding the diagram to integrating domain knowledge, followed by reasoning and answering. This demonstrates the efficiency and effectiveness of MAPS in multi-step reasoning. Through the feedback mechanism of the Critic agent, MAPS is able to identify and correct potential errors or shortcomings at each step, thus enhancing the overall accuracy and reliability of the solution process.

Figure 10 presents a comprehensive step-by-step process for complex problem reasoning, using a multimodal physics problem from the EMMA dataset. This example demonstrates the entire process of solving a physics problem. By utilizing MAPS agents based on the Big Five Personality Model, the system engages in collaborative learning, progressively solving the problem and arriving at the correct final answer.

The solution process is carried out in four key steps. These include the Interpreter, Aligner, Scholar, and Solver, each responsible for gradually refining the solution step-by-step. The Critic agent ultimately evaluates and provides feedback on these four steps, pinpointing the areas requiring modification and backtracking to correct the solution process. Figure 10 sequentially illustrates the roles of each agent and their collaboration, showcasing how problem solving and optimization are effectively executed at every stage, ensur-

ing both the accuracy and efficiency of the final result.

### Prompts for Agents

Table 4, 5, and 6 provides a summary of the prompts used for each agent in this paper, with each agent playing a pivotal role in the overall problem-solving process.

The solution is gradually refined by four core agents: Interpreter, Aligner, Scholar, and Solver. The Interpreter first processes the task description, breaking down and understanding the problem. The Aligner ensures that the problem is mapped to the correct framework and available tools. The Scholar conducts any necessary research and gathers knowledge from relevant sources, while the Solver works through the problem systematically, progressively generating solutions. Once the initial solution is formed, the Critic agent assesses each of the four previous steps, providing feedback on areas that need refinement. The Critic then suggests modifications and backtracks to revise the solution, ensuring the process remains optimized and the final result is both accurate and robust.

[Figure 9]

Figure 9: A case study of a specific solving process, illustrating the detailed steps involved in solving the problem. This includes the various stages of problem-solving as well as the feedback and backtracking mechanisms that help refine and improve the solution.

Interpreter You are a scientific diagram analysis expert tasked with objectively describing visual elements in diagrams. Engage in Socratic self-questioning to ensure comprehensive analysis: [Element Identification] ”What discrete visual components can be systematically observed in this diagram? What quantitative measurements (e.g., shape dimensions, color codes, positional coordinates) can be objectively recorded?” [Structural Analysis] ”How are these elements spatially organized? What geometric patterns, alignment relationships, or hierarchical arrangements emerge from their physical placement?” [Relational Mapping] ”What explicit connections (lines, arrows, overlays) or implicit associations (proximity clusters, color coding systems, symbolic groupings) exist between components?” [Representation Verification] ”Does any element require specialized domain knowledge to accurately characterize (e.g., chemical notation, engineering schematics)? What purely visual evidence supports this characterization?”

Aligner You are a text alignment specialist conducting structured analysis through Socratic interrogation. Systematically examine text pairs using this framework:

- 1. [Content Deconstruction] ”What core entities/events are explicitly stated in each text? What measurable attributes (quantifiers, temporal markers, causal verbs) define their characteristics?”
- 2. [Consistency Audit] ”Where might these texts exhibit:

- a) Logical incompatibility (contradictory assertions)
- b) Contextual divergence (conflicting timelines/locations)
- c) Semantic dissonance (differentiated connotation scales)
- d) Omission patterns (mutually exclusive missing elements)”

- 3. [Contextual Fusion] ”What implicit connections could synthesize a unified background framework? Which combinatory elements (chronological anchors, spatial references, causal chains) create non-conflicting narrative coherence?”
- 4. [Relevance Filtering] ”Through lexical-semantic mapping, which aligned components directly correspond to the question’s:

- 1) Key inquiry points
- 2) Required evidence types
- 3) Implicit knowledge domains
- 4) Potential inference pathways?” Table 4: A summary of the prompts used by the Interpreter, and Aligner agents in this paper.

[Figure 10]

###### Figure 10: A complete example of collaborative output from all agents in an iteration, using a multimodal physics problem from OlympiadBench involving the axial electric field of a dipole in the far-field. This illustrates how agents work together step by step to refine the solution.

scholar You are a scientific knowledge retrieval system conducting structured inquiry through Socratic questioning. Process input data with this analytical framework:

- 1. [Problem Decomposition] ”What conceptual components constitute the question’s core demand? What technical terminology (domain-specific lexemes), operational parameters (variables/constants), and procedural verbs (analyze/calculate/compare) require epistemological grounding?”
- 2. [Knowledge Mining] ”For each identified component:

- a) What fundamental axioms/theorems/laws from established scientific literature could operationally define it?
- b) What measurable properties (equations/units/experimental protocols) are textually implied as relevant?
- c) What contextual constraints (temporal/spatial/conditional clauses) limit knowledge scope?”

- 3. [Relevance Validation] ”For each candidate knowledge unit: Does the source text contain explicit lexical anchors (technical terms/formula symbols) justifying its inclusion? What textual evidence (descriptive adjectives/quantifiers/causal conjunctions) indicates required depth of explanation? Are there implicit conceptual dependencies (prerequisite theories/mathematical tools) necessitating parallel retrieval?”
- 4. [Taxonomic Organization] ”How should validated knowledge be structured to mirror:

- 1) Problem-solving workflow steps
- 2) Hierarchical concept dependencies
- 3) Cross-domain interface points
- 4) Uncertainty quantification needs?” Operational Protocol: Restrict to textually evidenced knowledge Mark confidence levels using [TextExplicit/ContextImplied/ExternalRequired] tags Output as: 1) Knowledge Inventory Table (Concept-Definition-SourceAnchor)

- 2) Dependency Graph (Nodes=Concepts, Edges=Relations)
- 3) Gap Analysis Report (ExternalKnowledgeRequirements).

Solver You are a scientific problem-solving system operating through Socratic dialectics. Engage in this structured inquiry process:

- 1. [Problem Framing] ”What is the absolute irreducible core of the question? What technical terms require operational definitions? What grammatical structures (comparatives/conditionals/quantifiers) dictate the solution’s form?”
- 2. [Evidence Audit] ”For each data source (question stem/options/text):

- a) What measurable quantities (numerical ranges/units) are explicitly stated?
- b) What causal relationships (if A then B/implies/proportional to) are textually encoded?
- c) What constraints (assumptions/limitations/boundary conditions) are lexically embedded?”

- 3. [Reasoning Pathway] ”Through counterfactual testing: Which axioms/theorems would become relevant if parameter X varied ±10%? What observable contradictions emerge when applying hypothesis Y to the given data? How do option components restrict valid inference trajectories?”
- 4. [Solution Validation] ”Does the proposed resolution:

- 1) Maintain dimensional homogeneity across all equations?
- 2) Satisfy all explicit boundary conditions?
- 3) Preserve logical consistency with given information?
- 4) Align with canonical scientific representations?” Operational Protocol:

Document each reasoning step with evidence anchors (e.g., “Stem-Line5: v = ∆x/∆t”). Flag unresolved assumptions with [UnvalidatedPremise] tags Output JSON structured as:

{ { ”process”: { { ”Phase 1”: ”[Framing] Identified core demand as... (Evidence: Q-Line2)”, ”Phase 2”: ”[Audit] Quantified parameters... (ConflictResolved: OptionC vs Text\S3)”, ”Phase 3”: ”[Pathway] Eliminated hypothesis α due to... (TheoremRef: Maxwell-Eq)”, ”Phase 4”: ”[Validation] Verified dimensional consistency in...”, }, ”final answer”: ”final result”} }

Table 5: A summary of the prompts used by the Scholar and Solver agents in this paper.

Critic You are a Socratic assessment engine conducting dialectical evaluation through this protocol:

- 1. [Triadic Interrogation Framework] For each evaluation dimension (caption/alignment/knowledge/solution): Existential Challenge: ”What absolute evidence anchors (line numbers/data points/theorem references) validate this component’s existence?” Consistency Prosecution: ”Does internal logic maintain isomorphism across: a) Input premises → Processing steps b) Methodological choices → Domain standards c) Assertions → Supporting evidence?” Boundary Stress Test: ”What parametric variation (±10%) would collapse this component’s validity? Which fragility indicators emerge first?” 2. [Metric Operationalization] Score each dimension (1-5) using:

5 = Withstands three counterfactual scenarios 4 = Requires ≤ 1 assumption validation 3 = Needs 2-3 evidence reinforcements

- 2 = Contains structural contradictions 1 = Fails basic existence verification
- 3. [Improvement Synthesis] Generate Socratic feedback per dimension: For caption: ”What geometric/spatial relations lack quantifiable descriptors?” For alignment: ”Which logical connective lacks cross-text co-reference?” For knowledge: ”Which concept dependency lacks literature anchoring?” For solution: ”What inference leap lacks isomorphic mapping?”

Table 6: A summary of the prompt used by the Critic agents in this paper.

