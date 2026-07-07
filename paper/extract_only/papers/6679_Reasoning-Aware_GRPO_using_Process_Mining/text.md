# arXiv:2510.25065v3[cs.AI]24May2026

## Rewarding Structural Conformance of Reasoning using Process Mining

##### Yongjae Lee∗

Dept. of Industrial Engineering Pusan National University Busan, Republic of Korea yongzzai1102@pusan.ac.kr

##### Taekhyun Park∗

Dept. of Data Science Pusan National University Busan, Republic of Korea pthpark1@pusan.ac.kr

Sunghyun Sim Dept. of Data Science Changwon National University Changwon, Republic of Korea ssh@changwon.ac.kr

Hyerim Bae† Dept. of Industrial Engineering Pusan National University Busan, Republic of Korea hrbae@pusan.ac.kr

#### Abstract

Recent advances in sparse reward policy gradient methods have enabled effective reinforcement learning (RL)-based language model post-training. However, for reasoning tasks such as mathematical problem solving, binarized outcome rewards provide limited feedback on intermediate reasoning steps. While some studies have attempted to address this issue by estimating overall reasoning quality, it remains unclear whether these rewards are reliable proxies for the quality of stepwise reasoning. In this study, we consider reasoning as a structured process and propose TACReward, the reward model that can be seamlessly integrated into sparse reward policy gradient methods without additional human annotation costs or architectural modifications. TACReward aggregates stepwise structural deviations between teacher and policy reasoning using process mining techniques, producing a scalar output reward range of [0,1] to indicate reasoning quality. Experiments on multiple mathematical reasoning benchmarks demonstrate that integrating the TACReward into sparse reward frameworks encourages the policy model to improve the structural quality of reasoning. Consequently, this leads to consistent performance improvements over existing sparse reward frameworks. Our code and checkpoints are publicly available at GitHub and HuggingFace.

#### 1 Introduction

Recent advances in post-training methods based on RL [1] have established a new paradigm for enhancing the alignment and reasoning capabilities of Large Reasoning Models (LRMs) [2]. Traditional policy gradient methods, such as Proximal Policy Optimization (PPO) [3] and its variants, have achieved significant success. However, PPO incurs substantial computational and memory overhead from maintaining a separate critic network and often suffers from training instability when scaled to large reasoning models. To improve efficiency, Reinforced Leave-One-Out (RLOO) [4] introduced a leave-one-out baseline to reduce the variance without a dedicated critic. GRPO [5] and GSPO [6] have shifted this paradigm by adopting sparse reward signals and removing the critic network and the Generalized Advantage Estimation (GAE).

∗Equal contribution. †Corresponding author.

Preprint.

| |
|---|

| |
|---|

| |
|---|

- Figure 1: TACReward separates correct from incorrect reasoning. Distributions of TACReward, computed against the reasoning of DeepSeek-R1 (671B) [14] as the reference, for correct (green) and incorrect (purple) solutions on DeepMath-103k [15]. Four models ranging from 1.5B to 70B parameters are evaluated. Correct responses consistently receive a higher degree of alignment than incorrect ones, and the separation is statistically significant for all models (U-test [16], p < 10−10).

Despite these advances, sparse reward policy gradient methods remain challenging to apply to reasoning tasks, such as mathematical problem solving due to their dependence on verifiable outcome rewards (i.e., reward sparsity) [7]. The reliance on binary and rule-based signals often produces uniform reward signals when responses within a problem group exhibit similar correctness. Consequently, this yields minimal differentiation and weakened learning gradients [8]. One potential solution involves providing step-level labels such as Process Reward Models (PRMs) [9]. However, adopting such dense reward within a sparse reward framework is challenging without architectural modifications, potentially diluting the advantage of stability [10]. Furthermore, defining step-level labels is inherently challenging, and human annotation remains costly [9].

Recent studies have proposed sparse reward frameworks that consider reasoning quality without annotation [7, 11–13], most of which are based on GRPO. However, these methods often rely on an indirect estimation or approximation of the overall reasoning process quality. This makes it challenging to consider the resulting reward signal as indicator of the quality of the individual reasoning steps. Consequently, it remains unclear whether improvements in cumulative rewards reflect genuine gains in logical maturity or mere over-optimization of surrogate metrics. Therefore, while ensuring that step-level logical integrity is crucial for authentic reasoning, an inherent dilemma arises from sparse reward nature.

We start from the observation that solutions to a reasoning problem can differ substantially at the surface level yet share a common structural skeleton of cognitive activities. In mathematical problem solving, for instance, one solution may proceed by substitution and another by direct computation, but both typically follow a sequence such as Formulate Strategy → Apply Formula → Simplify Expression → Verify. This observation suggests that reasoning quality can be characterized structurally, in terms of the sequence and composition of such activities, rather than the specific content of individual reasoning steps.

Building on this view, we propose Trace, Alignment, and Check Reward (TACReward), a novel reasoning-aware reward model that can be seamlessly integrated into sparse reward policy gradient methods for language model post-training (i.e., the final reward signal is in the range of 0 to 1) with no additional human annotation cost. In TACReward, the reasoning of LRMs is treated as a structured activity sequence (i.e., process) rather than a monolithic output and the policy model is encouraged to improve its degree of alignment with the teacher model’s reasoning, which is considered more mature than the policy model. To this end, process mining [17], a set of techniques specialized in analyzing data recording the execution of processes, is adapted and extended. Our preliminary investigation shows that TACReward, computed against the reference, assigns systematically higher scores to responses with correct final answers than to those with incorrect ones, with a statistically significant separation between the two distributions (Figure 1).

#### 2 Preliminaries

##### 2.1 Event Log & Process Mining

An event log [18] is hierarchically structured data that records the execution of a process. The formal definitions of events, traces, and logs are as follows:

Definition (Event, Trace, and Log). E is the event universe. An event is a tuple e = (c,a,t,d1,...,dm) ∈ E, where c is the case

- id, a is the activity, t is the timestamp, and
- d1,...,dm are the additional attributes. A trace σ = ⟨e1,e2,...,en⟩ ∈ E∗ is a finite, nonempty sequence of events such that a trace contains all and only the events in the same case id. An event log L is a multiset of traces.

Process mining [17] is a set of techniques that uses event logs to discover process models, check the conformance between event log and process model [19], and enhance or extend the models. Figure 2 presents an example of an event log and the process model derived from it. Various discovery algorithms have been proposed ranging from foundational Alpha Miner [18] to more robust approaches, such as inductive miners [20].

| |Case ID|Activity|Timestamp|
|---|---|---|---|
|0|Question 1|Formulate Strategy|07.06.10:00:00.542|
|1|Question 1|Recall Definition|07.06.10:00:01.004|
|2|Question 1|Apply Formula|07.06.10:00:01.014|
|3|Question 1|Identify Contradiction|07.06.10:00:04.251|
|4|Question 1|Explore Edge Cases|07.06.10:00:08.369|
|5|Question 1|Simplify Expression|07.06.10:00:11.224|
|6|Question 1|Apply Formula|07.06.10:00:17.016|
|7|Question 1|Identify Contradiction|07.06.10:00:18.482|
|8|Question 1|Verify Answer|07.06.10:00:21.581|
|…|…|…|…|

| |
|---|

| | |
|---|---|
| | |

| |
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| |
|---|

Figure 2: Example of mathematical reasoning event log with a single case id and the corresponding process model inferred from it. The boxes in the map represent activities, and the arrows indicate the flow. The circles denote the places, which are used to model the state of the process.

##### 2.2 Notations

In this study, an autoregressive LRM as a stochastic policy over sequences parameterized by θ

is denoted by policy πθ. The teacher model is denoted by πϕ, where ϕ is the set of parameters. Given an input query x ∼ D sampled from the dataset D, the model generates a response y =

y1,y2,...,y|y| .

##### 2.3 Sparse Reward Policy Gradient Methods for LRMs

The policy gradient methods [21] with sparse rewards maximizes expected reward for the responses generated by the policy LRM πθ:

J (θ) = Ex∼D,y∼π

θ(·|x) [R(x,y)] (1)

where R(x,y) is the reward function that provides a scalar reward signal based on the quality of the response y to query x. Using the score function estimator [22], the policy gradient can be expressed as

∇θJ (θ) = E ∇θ log πθ(y|x) R(x,y) − b(x) (2)

where b(x) is a baseline independent of y, introduced to reduce the variance of the estimator. Such a baseline acts as a control variate, subtracting any function independent of y, and preserving the unbiasedness of the gradient estimator. For autoregressive language models, the log-likelihood factorizes over the tokens as follows:

log πθ(y|x) =

|y|

log πθ yt | x,y<t (3)

t=1

It is often convenient to define the (Monte Carlo) advantage estimator [23] as: Aˆ(x,y) ≜ R(x,y) − b(x), such that Equation (2) can be expressed as Aˆ(x,y).

##### 2.4 Group-based Policy Gradient Methods for LRMs

Recent sparse-reward post-training methods often sample multiple responses for the same query x [5, 6]. For each x, we draw a group of G responses

{yi}Gi=1 ∼ πθ(· | x) (4)

and compute their corresponding outcome rewards {ri}Gi=1 ≜ {R x,yi }Gi=1, where ri is typically sparse (e.g., binary correctness) in reasoning tasks. A common baseline choice in this setting is:

Aˆi = ri − mean {ri}Gi=1 (5)

#### 3 Method

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

|Prompt| |
|---|---|
| | |

𝑦1 𝑦𝐺 <think>…</think>\boxed{…}

<think>…</think>\boxed{…}

|…|
|---|

Policy Model

###### Teacher Model

[Figure 12]

𝑦𝑟𝑒𝑓 <think>…</think>\boxed{…}

?

…

…

{System Role:...} {Question:...}

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

### TAC Reward TraceExtraction

[Figure 18]

[Figure 19]

###### Trace Extraction

Trace, Align and Check

|𝐺|
|---|

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

| |
|---|

[Figure 26]

|𝐴|𝐷|𝐵|≫|𝐸|𝐺|
|---|---|---|---|---|---|
|𝐴|≫|𝐵|𝐷|𝐸|𝐺|

- 𝑟1𝑇𝐴𝐶

- 𝑟2𝑇𝐴𝐶

[Figure 27]

Case ID Activity Step

[Figure 28]

Case ID Activity Step

Case ID Activity Step Q1 Formulate Strategy 0 Q1 ApplyFormula 1 Q1 RecallDefinition 2 Q1 Explore Edge Cases 3

Q1 Formulate Strategy 0

[Figure 29]

ℳ1

[Figure 30]

| | |
|---|---|
| | |

| |
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Q1 Formulate Strategy 0

[Figure 31]

[Figure 32]

Q1 Recall Definition 1

Process Discovery

Case ID Activity Step Q1 Formulate Strategy 0 Q1 ApplyFormula 1 Q1 RecallDefinition 2 Q1 Explore Edge Cases 3

𝐴 ≫ 𝐷 𝐵 𝐸 𝐺 𝐴 𝐶 𝐷 ≫ 𝐸 𝐺

Q1 Recall Definition 1

Q1 Apply Formula 2

…

…

Q1 Explore Edge Cases 4

Q1 Apply Formula 2

…

…

Algorithm

[Figure 33]

… … …

ℳ𝐺

[Figure 34]

[Figure 35]

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Q1 Explore Edge Cases 3

… … …

𝑟𝐺𝑇𝐴𝐶

𝐴 𝐷 𝐵 ≫ 𝐸 𝐺 𝐴 ≫ 𝐵 𝐷 𝐸 𝐺

… … …

… … …

Policy Traces Reference Trace

Process Models

Alignments Rewards

Figure 3: Overview of the proposed TACReward. Given a query x, responses {yi}Gi=1 are sampled from the policy model πθ and converts them into policy traces, from which G process models

{Mi}Gi=1 are discovered. The teacher’s reasoning is converted into a reference trace, and each process model Mi is evaluated against it to obtain the scalar rewards riTAC Gi=1. In cases where multiple responses are not generated for the same query, G = 1.

The proposed TACReward treats the reasoning of an LRM as a structured process rather than a monolithic output and aims to encourage a policy model to improve the degree of alignment with the teacher model’s reasoning process. To this end, techniques of process mining are leveraged and extended. In process mining literature, conformance checking aims to evaluate how a process model can explain an entire event log that contains multiple traces [17]. We extend this paradigm to operate on individual traces. Specifically, TACReward can be categorized into three main steps: 1) Trace: Formalizing the reasoning trace of both the policy and teacher models. 2) Align: Examines the alignment between the policy and reference traces. 3) Check: Evaluate how well the reasoning process of the policy model conforms to that of the teacher model.

The proposed TACReward is illustrated in Figure 3.

##### 3.1 Trace: Formalize Reasoning

The computation of the TACReward begins by prompting the policy model to generate a reasoning process for a given problem x. For each problem, the model produces G candidate responses, {yi}Gi=1, each containing a sequence of reasoning steps (see Section D.1 for the generation prompt). Subsequently, these responses were formalized into reasoning traces by segmenting them into discrete steps and formatting them to conform to an event log schema. Each structural units of reasoning represent a distinct activity. The taxonomy of the 20 activities is adopted from related studies and mathematical problem-decomposition frameworks [24–27]. A complete list of the activities is provided in Table 1. Table 8 provides the ablation study on the taxonomy.

To extract the traces, DeepSeek-V3.2 [28], a general-purpose language model, is employed. Let πψ denote a general-purpose model, the policy traces are formalized as:

{σi}Gi=1 = πψ (πθ(· | x)) (6)

where σi is the trace corresponding to the i-th response. Similarly, the reference trace is:

σref = πψ (πϕ(· | x)) (7)

where the teacher model generates a single response for each problem. The prompt used to formalize the trace is provided in Section D.2.

##### 3.2 Align: Examine the Alignment

In this step, each policy trace is transformed into a process model using the Inductive Miner (IM) algorithm [20]. Given the policy traces {σi}Gi=1 and reference trace σref, we discover a process model for each policy trace by treating {σi} as a single-trace event log:

Table 1: Taxonomy of 20 mathematical reasoning activities

# Activity # Activity

- 1 Start Problem 11 Justify Step
- 2 Recall Definition 12 Explore Edge Cases
- 3 Identify Known Results 13 Identify Contradiction
- 4 Formulate Strategy 14 Interpret Result
- 5 Apply Known Formula 15 Check Validity
- 6 Simplify Expression 16 Verify With Example
- 7 Change of Variable 17 Refine or Change Strategy
- 8 Evaluate Limit or Integral 18 Conclude Final Result
- 9 Perform Comparison 19 Recheck Original Question
- 10 Apply Theorem 20 End Problem

Mi = IM({σi}), i = 1,...,G (8)

where Mi denotes the discovered process model corresponding to the ith policy trace.

To establish an alignment [29] between a policy process model Mi and the reference trace σref, we relate the moves in the model to the moves in the trace. Let Σ be the set of activity labels and let ≫ denote a "no-move" (i.e., misalignment) symbol. Given model Mi, let β(Mi) ⊆ Σ∗ be the set of all possible complete executions sequences in Mi.

Definition (Alignment). The alignment of σref and Mi is a sequence: γ = ⟨(aref1 ,am1 ),...,(arefK ,amK)⟩ where (arefk ,amk ) ∈ (Σ ∪ {≫}) × (Σ ∪ {≫})\,{(≫,≫)} such that the top row yields σref and the bottom row yields σβ ∈ β(Mi). The set of all these alignments is denoted by Γ(σref,Mi).

To illustrate this, consider the reference trace σref = ⟨A,D,B,E,G⟩, and the process model in

- Figure 2 as the policy process model Mi. For clarity, two example alignments γ1 and γ2 are provided in Figure 4. In an alignment, ≫ indicates a misalignment (a log-only move or model-only move), where each column represents a vertical movement. When the same activity is executed in both the reference trace and the model, it is called a synchronous move.

To quantify the discrepancy between the reference traces and the behavior allowed by the policy process model Mi, a finite cost is assigned to each move.

- γ1 =

|A|D|B<br><br>|≫<br><br>|E|G|
|---|---|---|---|---|---|
|A<br><br>|≫|B<br><br>|D<br><br>|E<br><br>|G|

- γ2 =

|A<br><br>|≫|D<br><br>|B<br><br>|E|G|
|---|---|---|---|---|---|
|A|C<br><br>|D|≫<br><br>|E|G|

Definition (Cost Function). Let (aref,am) be a move in alignment γ ∈ Γ(σref,Mi). The move cost function δm is defined as:

Figure 4: Two example alignments γ1 and γ2. In γ1, the first move (A,A) is a synchronous move. The second move (D,≫) indicates that the reference trace executes D whereas the model does not. In γ2, the second move (≫,C) indicates that the model executes C that is absent from the reference trace.

 

0, aref = am ̸=≫, wL(aref), am =≫, wM(am), aref =≫

δm(aref,am) =



(9) where wL(·) and wM(·) are non-negative weights and wL(·) = wM(·) = 1 unless stated otherwise.

Finally, the optimal alignment γi∗ for the i-th model is obtained by minimizing the total cost δ(γ), which is the sum of the costs over all moves in the alignment:

γi∗ = argmin

δ(γ) (10)

γ∈Γ(σref,Mi)

Section C provides the computational complexity analysis of finding γi∗.

##### 3.3 Check: Evaluate the Conformance

Through the alignment step, an optimal alignment γi∗ is established between each policy process model Mi and the reference trace σref. Based on this alignment, we quantify the conformance score riTAC, which serves as the TACReward output for each policy response. We compute the conformance by using the F1-score of fitness and precision [29].

Fitness. Fitness measures how well the process model explains the observed behavior in the reference traces [17]. To obtain stable normalization, the finite worst-case deviation cost is defined as:

wM(e′) (11)

δworst(σref,Mi) =

wL(e) + min

σβ∈β(Mi)

e′∈σβ

e∈σref

where the cost of σref with only log moves, plus the minimum cost to complete the model using only the model moves. Subsequently, the fitness is computed as

δ(γi∗) δworst(σref,Mi)

sfit(σref,Mi) = 1 −

(12)

Precision. The precision assesses how often actions that are not present in the trace occur in a process model [17]. Let enM(e) be the set of enabled activities in model Mi in the state immediately before event e occurs, and let enL(e) be the set of activities observed in σref in the same context (i.e., the same prefix immediately before e). The precision is computed as:

|enL(e)| |enM(e)|

1 |E| e∈E

sprec(σref,Mi) =

, (13) where E denotes the collection of events in the reference trace. A precision value close to 1 indicates that the model does not allow significantly more behavior than that observed in the reference trace. Output of TACReward. Based on the computed fitness and precision, the output reward riTAC is:

2 · sfit(σref,Mi) · sprec(σref,Mi) sfit(σref,Mi) + sprec(σref,Mi)

riTAC =

(14)

where riTAC ∈ [0,1] denotes the degree of conformance between the reasoning process of the policy model and that of the teacher model for the ith response. TACReward can be seamlessly integrated

into sparse-reward policy gradient methods by directly adding its output as the original task reward. Detailed integration procedures are provided in Section A.

#### 4 Experiments

In this section, we present the experimental results of integrating TACReward into sparse reward policy gradient methods. The details of experiments are described in Section B.

##### 4.1 Foundational Policy Gradients Methods with TACReward

Table 2 compares four representative sparse-reward policy optimization methods—PPO, RLOO, GRPO, and GSPO, with and without TACReward-under a fixed training budget of 240 optimization steps on Qwen2.5-7B-Instruct. This setting characterizes early stage optimization rather than converged performance. To better assess the performance, we compute the average score (Avg. *) by excluding MATH500 and AIME 2024 and instead focus on the for the remaining five benchmarks.

Strong Synergy with GSPO. The most significant improvement was observed when TACReward was combined with GSPO. Although GSPO alone underperforms in this early stage regime, GSPO + TAC increases Avg. * from 17.2 to 32.5 (+89.2%). We attribute this synergy to a granularity match between the optimization objective and reward signal. GSPO defines the importance ratio and updates it at the sequence level, effectively treating each sampled response as a single optimization unit. TACReward aggregates the step-level deviations revealed by the trace alignment into a single sequence-level conformance score that represents the reasoning quality of one response. This allows TACReward to provide more informative and well-aligned training signal for GSPO and can be exploited more effectively than in token-level or step-mismatched objectives.

Table 2: Optimization behavior of sparse-reward RL methods with TACReward (240 training steps)

KSAT 2025

LiveMath Bench Avg.*

MATH 500 MINERVA Olympiad

AIME 2024

AIME 2025

Qwen2.5-7B-Instruct Baseline (SFT) 63.4 19.9 33.3 10.0 10.0 26.7 7.0 24.3 PPO 43.2 25.4 35.6 10.0 6.7 36.7 7.0 22.3 PPO + TAC 49.4 24.6 37.6 13.3 6.7 23.3 12.0 20.8 (-6.5%) RLOO 57.4 25.4 37.6 10.0 0.0 30.0 9.0 20.4 RLOO + TAC 57.2 22.8 35.4 13.3 10.0 30.0 10.0 21.6 (+6.1%) GRPO 57.4 26.1 33.3 13.3 3.3 30.0 7.0 19.9 GRPO + TAC 63.4 28.7 38.1 13.3 3.3 33.3 9.0 22.5 (+12.7%) GSPO 61.4 19.9 33.3 10.0 10.0 16.7 6.0 17.2 GSPO + TAC 64.2 37.1 38.2 10.0 10.0 53.3 15.0 32.5 (+89.2%)

Compatibility with PPO, RLOO, and GRPO. TACReward provides consistent improvements for RLOO and GRPO, with relative gains of +6.1% and +12.7%, respectively, in Avg.*, suggesting broad compatibility with variance-reduced and group-based objectives. In contrast, adding TACReward to PPO yields limited gains and slightly reduces the overall average score despite improvements on some benchmarks. We attribute this asymmetry to PPO’s reliance on dense reward signals: under the sparse reward regime considered in this study, PPO becomes more sensitive to reward variance and credit assignment, whereas RLOO and GRPO are inherently designed for variance reduction and thus absorb the additional signal from TACReward more effectively. Section E provides visualizations of reasoning processes with and without TACReward during GSPO training.

##### 4.2 Comparison with State-of-the-Art Models

- Table 3: Comparison with baseline RL methods using GSPO + TACReward (3000 training steps)

LiveMath Bench DeepSeek-R1-Distill-Qwen-7B

KSAT 2025

MATH 500 MINERVA Olympiad

AIME 2024

AIME 2025

BASELINE [5] 80.6 40.4 53.8 43.3 30.0 66.7 13 GSPO + TAC (Ours) 90.4 49.3 67.7 50.0 36.7 83.3 24 DeepMath [15] 83.4 42.6 48.7 34.2 30.0 63.3 17 SkyWork [30] 87.0 43.4 55.7 46.7 36.7 66.7 12 GRPO-LEAD [8] 84.6 47.4 52.3 40.0 26.7 76.7 13 DRGRPO [31] 80.2 10.3 41.0 40.0 6.7 66.7 21 ExGRPO [13] 82.8 41.0 47.9 23.3 16.7 63.3 13 Eurus-2 [7] 79.2 38.6 42.1 26.7 16.7 20.0 2 VeriThinker [32] 80.2 34.9 46.4 30.0 13.3 76.7 14

DeepSeek-R1-Distill-Qwen-1.5B

BASELINE [5] 69.4 26.5 40.0 23.3 23.3 60.0 11 GSPO + TAC (Ours) 74.0 27.2 44.0 20.0 20.0 63.3 14 DRA-GRPO [33] 71.0 26.1 40.4 20.0 23.3 43.3 5 Open-RS3 [34] 69.8 26.8 39.7 20.0 20.0 50.0 6 STILL-3 [35] 72.4 23.2 42.8 20.0 16.7 53.3 3 ExGRPO [13] 71.2 30.9 34.4 10.0 10.0 40.0 10

Based on the controlled comparison in Table 2, we selected GSPO + TACReward as the most effective configuration and optimized it further for evaluation. Subsequently, we compared this configuration with the recent state-of-the-art RL methods. Table 3 compares GSPO + TAC with the recent state-ofthe-art RL methods for two distilled backbone models of different sizes. Training was conducted for 3000 training steps. For both model sizes, GSPO + TAC exhibited strong or state-of-the-art

- Figure 5: Reward dynamics during GSPO + TAC training. Left: Comparison of accuracy reward among GRPO, GSPO, and GSPO + TAC on the 1.5B policy model. Right: TACReward (left axis) and accuracy reward (right axis) over training steps for the 7B (top) and 1.5B (bottom) policy models, smoothed with a 15-step moving average.

performance across the benchmarks. With the 7B backbone, it outperformed all baselines, particularly Olympiad and KSAT 2025, and remained competitive (often better) at 1.5B.

Reward Dynamics during Training. The comparison of reward dynamics among GRPO, GSPO, and GSPO + TAC in the 1.5B policy model in Figure 5 (Left) illustrates that GSPO + TAC achieves stable and high reward trajectories throughout training. Moreover, TACReward steadily increases during training, indicating a progressive better conformance between the policy’s reasoning process and the teacher’s traces. For both model sizes, the model learns to produce more accurate answers while more closely aligning the reasoning process with the teacher (Right). Section F provides visualizations of reasoning processes as training progresses, indicating that the model’s reasoning steps became more structured and aligned with those of the teacher over time.

##### 4.3 Analysis

- Table 4: Performance comparison of distilled models against the DeepSeek R1 teacher across reasoning benchmarks. Relative degradation from the teacher is shown in parentheses.

LiveMath Bench DeepSeek-R1-Distill-Qwen-1.5B

KSAT 2025

MATH 500

AIME 2024

AIME 2025

Teacher Size Teacher Acc.

MINERVA Olympiad

DeepSeek R1 685B 94.17% 49.8 19.7 20.6 0 0 30 12 R1-Distill-LLaMA 70B 86% (−8.7%) 45.8 19.1 18.4 0 0 30 10 R1-Distill-Qwen-2.5 32B 68% (−21.8%) 44.8 17.6 17.3 0 0 26.7 12 R1-Distill-Qwen-2.5 14B 67.8% (−28.0%) 44.2 15.4 17.3 3.3 0 23.3 12

Sensitivity to Teacher Quality. A natural concern is whether TACReward depends on access to a frontier-scale teacher. Table 4 compares the performance by replacing DeepSeek-R1 (671B) with progressively smaller and less accurate teachers, while keeping all other components fixed. Despite teacher accuracy on DeepMath-103k dropping by up to 28%, the resulting policy’s performance degrades only marginally on most benchmarks. This indicates that TACReward does not require the teacher to be correct on every problem; what matters is the structural pattern of the reasoning trace, which remains informative even when the teacher’s final answer is wrong.

Table 5: Policy performance when varying the trace extractor model.

KSAT 2025

LiveMath Bench

MATH 500 MINERVA Olympiad

AIME 2024

AIME 2025

Extractor

DeepSeek-V3.2 49.8 19.7 20.6 0.0 0.0 30.0 12 Gemini-2.5-Flash 49.8 19.7 20.2 3.3 0.0 26.7 12 GPT-5.4-nano 50.0 20.3 20.8 3.3 0.0 30.0 12

Sensitivity to Trace Extractor. In TACReward, a general-purpose model is utilized to convert raw reasoning responses into activity sequences (Section 3). To assess whether this step is bottlenecked by extractor capacity, we replace DeepSeek-V3.2 [28] with two alternatives spanning different model families: Gemini-2.5-Flash and GPT-5.4-nano. The latter is the smallest member of the GPT-5.4 family. As shown in Table 5, all three extractors yield nearly identical policy performance, with GPT-5.4-nano matching or slightly exceeding our default on every benchmark. Trace formalization is thus robust to the choice of extractor as long as the model can follow the 20-activity taxonomy.

Proof of Concept for Using Process Mining. Based on the definitions in [17], fitness penalizes the missing essential steps, whereas precision penalizes excessive or unjustified behavior. Through Figure 6, we found that this definition aligns exactly with the expected behavior patterns that we want to encourage during training.

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

- Figure 6: Changes in Entropy, Mean Length, and TACReward for Fitness vs. Precision vs. Both (240 training steps)

When only fitness was used, we expected the model to generate as many reasoning steps as possible and to employ a wide variety of activities to avoid penalization for missing essential steps. This is confirmed in Figure 6, where using only fitness (green) results in longer reasoning chains (mean length). However, when only precision is used (purple), the model becomes conservative, generating fewer reasoning steps and using fewer activities to minimize excessive behavior. This is indicated by the shorter mean length and higher entropy. Both components (red) balance these two tendencies, yielding the highest confidence (lowest entropy) and moderate length. Section G visualizes this behavior, and Table 7 confirms that using either component alone degrades performance across benchmarks.

Limitations TACReward requires a task-specific taxonomy: the current 20-activity scheme is tailored to mathematical problem solving, and other reasoning domains may demand a new taxonomy together. We further note that our experiments use high-cost API-based teacher and extractor models to showcase the full potential. In practice, one can use smaller models for both roles, and the sensitivity analyses suggest that this would not significantly degrade performance. However, the exact performance trade-offs and optimal configurations when using smaller models remain to be explored.

For additional analyses, Section C provides further results and discussions.

#### 5 Conclusion

Sparse reward policy gradient methods are challenging to incorporate into reasoning tasks, particularly when learning which reasoning steps to perform and in which order. Moreover, providing such annotations is costly, and their quality can significantly affect training outcomes. In this study, we proposed TACReward, a reasoning-aware reward model that can be seamlessly integrated into sparse reward policy gradient methods for LRM post-training without additional annotations or architectural changes. Instead of teaching the exact intermediate actions, TACReward measures the structural alignment between the reasoning processes of a policy model and that of a more logically mature teacher. The results showed that accounting for structural similarity improves the sparse reward performance and optimization behaviors induced by fitness and precision match their intended effects.

For future works, the underlying mechanism of TACReward is general and we expect it to apply to a broader range of reasoning domains such as code generation. We argue that a particularly promising

direction is to extend TACReward to domains where the underlying process is well-defined. In such settings, a teacher model is not strictly necessary: the reference trace can be specified directly from domain knowledge, and TACReward can serve as a structural guardrail that evaluates whether the policy’s reasoning conforms to the prescribed procedure. For instance, in loan assessment, the reference trace can encode the standard sequence of steps a credit analyst is expected to perform, and TACReward can guide the policy to internalize a reasoning process appropriate for the task.

#### References

- [1] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [2] Kaiyan Zhang, Yuxin Zuo, Bingxiang He, Youbang Sun, Runze Liu, Che Jiang, Yuchen Fan, Kai Tian, Guoli Jia, Pengfei Li, et al. A survey of reinforcement learning for large reasoning models. arXiv preprint arXiv:2509.08827, 2025.
- [3] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [4] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.
- [5] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [6] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.
- [7] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Yuchen Zhang, Jiacheng Chen, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.
- [8] Jixiao Zhang and Chunsheng Zuo. Grpo-lead: A difficulty-aware reinforcement learning approach for concise mathematical reasoning in language models. arXiv preprint arXiv:2504.09696, 2025.
- [9] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.
- [10] Michael Sullivan. Grpo is secretly a process reward model. arXiv preprint arXiv:2509.21154, 2025.
- [11] Lishui Fan, Yu Zhang, Mouxiang Chen, and Zhongxin Liu. Posterior-grpo: Rewarding reasoning processes in code generation. arXiv preprint arXiv:2508.05170, 2025.
- [12] Zhicheng Yang, Zhijiang Guo, Yinya Huang, Xiaodan Liang, Yiwei Wang, and Jing Tang. Treerpo: Tree relative policy optimization. arXiv preprint arXiv:2506.05183, 2025.
- [13] Runzhe Zhan, Yafu Li, Zhi Wang, Xiaoye Qu, Dongrui Liu, Jing Shao, Derek F Wong, and Yu Cheng. Exgrpo: Learning to reason from experience. arXiv preprint arXiv:2510.02245, 2025.
- [14] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [15] Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025.
- [16] Henry B Mann and Donald R Whitney. On a test of whether one of two random variables is stochastically larger than the other. The annals of mathematical statistics, pages 50–60, 1947.
- [17] Wil Van Der Aalst. Data science in action. In Process mining: Data science in action, pages 3–23. Springer, 2016.

- [18] Wil Van Der Aalst, Arya Adriansyah, Ana Karla Alves De Medeiros, Franco Arcieri, Thomas Baier, Tobias Blickle, Jagadeesh Chandra Bose, Peter Van Den Brand, Ronald Brandtjen, Joos Buijs, et al. Process mining manifesto. In International conference on business process management, pages 169–194. Springer, 2011.
- [19] Josep Carmona, Boudewijn van Dongen, and Matthias Weidlich. Conformance checking: foundations, milestones and challenges. In Process mining handbook, pages 155–190. Springer, 2022.
- [20] Sander JJ Leemans, Dirk Fahland, and Wil MP Van Der Aalst. Discovering block-structured process models from event logs-a constructive approach. In International conference on applications and theory of Petri nets and concurrency, pages 311–329. Springer, 2013.
- [21] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.
- [22] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.
- [23] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.
- [24] Tian Qin, Core Francisco Park, Mujin Kwun, Aaron Walsman, Eran Malach, Nikhil Anand, Hidenori Tanaka, and David Alvarez-Melis. Decomposing elements of problem solving: What" math" does rl teach? arXiv preprint arXiv:2505.22756, 2025.
- [25] Alessandro Berti, Humam Kourani, Gyunam Park, and Wil MP Van Der Aalst. Configuring large reasoning models using process mining: A benchmark and a case study. arXiv preprint arXiv:2501.00000, 2025.
- [26] George Polya. How to solve it: A new aspect of mathematical method. In How to solve it. Princeton university press, 2014.
- [27] Frank E Ritter, Farnaz Tehranchi, and Jacob D Oury. Act-r: A cognitive architecture for modeling cognition. Wiley Interdisciplinary Reviews: Cognitive Science, 10(3):e1488, 2019.
- [28] Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.
- [29] Wil Van der Aalst, Arya Adriansyah, and Boudewijn Van Dongen. Replaying history on process models for conformance checking and performance analysis. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 2(2):182–192, 2012.
- [30] Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, et al. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025.
- [31] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.
- [32] Zigeng Chen, Xinyin Ma, Gongfan Fang, Ruonan Yu, and Xinchao Wang. Verithinker: Learning to verify makes reasoning model efficient. arXiv preprint arXiv:2505.17941, 2025.
- [33] Xiwen Chen, Wenhui Zhu, Peijie Qiu, Xuanzhao Dong, Hao Wang, Haiyu Wu, Huayu Li, Aristeidis Sotiras, Yalin Wang, and Abolfazl Razi. Dra-grpo: Exploring diversity-aware reward adjustment for r1-zero-like training of large language models. arXiv preprint arXiv:2505.09655, 2025.
- [34] Quy-Anh Dang and Chris Ngo. Reinforcement learning for reasoning in small llms: What works and what doesn’t. arXiv preprint arXiv:2503.16219, 2025.
- [35] Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng Wang, Xiaoxue Cheng, Huatong Song, et al. Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. arXiv preprint arXiv:2412.09413, 2024.
- [36] Chi Liu. Rethinking gspo: The perplexity-entropy equivalence. arXiv preprint arXiv:2510.23142, 2025.

- [37] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.
- [38] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=d7KBjmI3GmQ.
- [39] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 3843–3857. Curran Associates, Inc., 2022.
- [40] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In LunWei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.211. URL https://aclanthology.org/2024.acl-long.211/.
- [41] Junnan Liu, Hongwei Liu, Linchen Xiao, Ziyi Wang, Kuikun Liu, Songyang Gao, Wenwei Zhang, Songyang Zhang, and Kai Chen. Are your llms capable of stable reasoning?, 2025. URL https: //arxiv.org/abs/2412.13147.
- [42] KICE. Mathematics section (calculus) of the 2025 korean college scholastic ability test, 2025. Administered on November 14, 2025.
- [43] MAA. American invitational mathematics examination (AIME). Mathematics Competition Series, n.d. URL https://maa.org/math-competitions/aime.
- [44] Mingqi Wu, Zhihao Zhang, Qiaole Dong, Zhiheng Xi, Jun Zhao, Senjie Jin, Xiaoran Fan, Yuhao Zhou, Huijie Lv, Ming Zhang, Yanwei Fu, Qin Liu, Songyang Zhang, and Qi Zhang. Reasoning or memorization? unreliable results of reinforcement learning due to data contamination, 2025. URL https://arxiv.org/ abs/2507.10532.
- [45] Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. Flashinfer: Efficient and customizable attention engine for llm inference serving, 2025. URL https://arxiv.org/abs/2501.01005.
- [46] Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. Trl: Transformer reinforcement learning. https: //github.com/huggingface/trl, 2020.
- [47] Robert A Wagner and Michael J Fischer. The string-to-string correction problem. Journal of the ACM (JACM), 21(1):168–173, 1974.
- [48] Yewei Song, Cedric Lothritz, Xunzhu Tang, Tegawendé Bissyandé, and Jacques Klein. Revisiting code similarity evaluation with abstract syntax tree edit distance. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 38–46, 2024.

#### A Integrating TACReward with Sparse Reward Policy Gradients Methods

In this section, we demonstrate how TACReward can be easily integrated into the foundational sparse reward policy gradient methods: PPO, RLOO, GRPO, and GSPO.

Notations TACReward is denoted as rTAC(x,yi) Gi=1. We set {racc(x,yi)}Gi=1 as the accuracy reward and rfmt(x,yi) Gi=1 as follows: thinking format reward for each response {yi}Gi=1. The total reward for each query-response pair is denoted as {R(x,yi)}Gi=1 as described in Section 2. In cases where multiple responses are not generated for the same query G = 1 and are simply denoted as R(x,y).

##### A.1 Proximal Policy Optimization (PPO) with TACReward

PPO [3] generates a single response for each input query (i.e., G = 1) and update the policy model based on the rewards obtained from the response. For an input x ∼ D and a sampled response y = y1,y2,...,y|y| ∼ πθ

(·|x), TACReward can be incorporated into PPO by simply adding it as follows:

old

|y|

πθ(yt|xt) πref(yt|xt)

R(x,y) = racc(x,y) + rfmt(x,y) + rTAC(x,y) − β

, (15)

log

t=1

where πref is a fixed reference policy and β > 0 controls the KL regularization strength. We define a sequence-level advantage with baseline b(x) and broadcast it to all timesteps:

At ≜ A(x,y) = R(x,y) − b(x), t = 1,...,|y|, (16) where b(x) is a baseline (e.g., a learned value function Vϕ(x)) to reduce variance. Let ωt(θ) =

πθ(yt|xt)

πθold(yt|xt). Subsequently, the PPO clipped objective is:

 

 . (17)

|y|

LPPO(θ) = E

min ωt(θ)At, clip(ωt(θ),1 − ϵ,1 + ϵ)At

t=1

##### A.2 REINFORCE with Leave-One-Out (RLOO) with TACReward

RLOO [4] generates a group of G responses for each input query (i.e., G > 1) and updates the policy using a leave-one-out baseline. Given an input x ∼ D and set of sampled responses {yi = ⟨yi,1,...,yi,|y

(·|x), TACReward can be integrated into RLOO by adding it to the reward for each response as follows:

i|⟩}Gi=1 ∼ πθ

old

|yi|

πθ(yi,t|xi,t) πref(yi,t|xi,t)

R(x,yi) = racc(x,yi)+rfmt(x,yi)+rTAC(x,yi)−β

log

, i = 1,...,G,

t=1

(18)

where πref is a fixed reference policy and β > 0 controls the KL regularization strength. RLOO uses a leave-one-out baseline computed from other samples in the group:

1 G − 1 j̸=i

R(x,yj), Ai ≜ A(x,yi) = R(x,yi) − bi(x). (19)

bi(x) =

Using the score function estimator, the RLOO policy gradient is

∇θJ (θ) = E

G

1 G

i=1

∇θ log πθ(yi|x) Ai , where log πθ(yi|x) =

|yi|

log πθ(yi,t|xi,t).

t=1

(20)

##### A.3 Group-based Policy Optimization with TACReward

GRPO and GSPO bypass the need for the value model by computing the group-relative advantage of each response within a response group for the same query. Partially borrowed from [36], the

group-relative advantages for the responses {y}Gi=1 are computed as:

R(x,yi) − mean R(x,yi)Gi=1 std R(x,yi)Gi=1

Aˆi,t = Aˆi =

(21)

where all the tokens in yi share the same advantage as Aˆi. GRPO then defines the objective as

|yi|

G

1 G

1 |yi|

min ωi,t(θ)Aˆi,t,clip(ωi,t(θ),1 − ϵ,1 + ϵ)Aˆi,t

JGRPO(θ) = E

t=1

i=1

(22) where ϵ is a hyperparameter for the clipping range, and the importance ratio ωi,t(θ) is ππθ(yi,t|x,yi,<t)

θold(yi,t|x,yi,<t)

The GSPO simplifies this objective as follows:

G

1 G

min si(θ)Aˆi,clip(si(θ),1 − ϵ,1 + ϵ)Aˆi (23)

JGSPO(θ) = E

i=1

1 |yi|

θ(yi|x) πθold(yi|x)

where the importance ratio si(θ) is π

In both GRPO and GSPO, TACReward can be integrated by simply adding it to the reward for each response as follows:

R(x,yi) = racc(x,yi) + rfmt(x,yi) + rTAC(x,yi) i = 1,...,G (24)

#### B Details of Experiments

##### B.1 Hyperparameters Setting

We trained the policy model using the TRL framework with DeepSpeed ZeRO-3 optimization. Table 6 summarizes the key hyperparameters and distributed training configuration used in experiments 4. All other parameters followed the default settings in TRL GRPOTrainer.

##### B.2 Experimental Settings

Models & Baselines We consider two categories of base models. For non-reasoning base models, we used Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct [37], which did not fine-tuned with RL for reasoning capabilities. For the reasoning-enhanced base models, we use DeepSeek-R1-Distill-Qwen1.5B and DeepSeek-R1-Distill-Qwen-7B [14], which already possess reasoning capabilities based on distillation.

Benchmarks & Evaluation Metric We train on the DeepMath-103k [15] dataset and evaluated seven challenging mathematical reasoning benchmarks: MATH-500 [38] MINERVA [39], OlympiadBench [40], LiveMathBench [41] and Korean CSAT Math Calculus [42], and AIME 2024–2025 [43]. However, we did not address this problem MATH500 and AIME 2024 as primary evidence because of potential contamination [44]. Therefore, the results for MATH-500 and AIME 2024 are reported for reference only, and the system prompts that was previously mentioned, was not used. Instead, all of the main quantitative comparisons and conclusions are obtained from the remaining benchmarks. All evaluations used the Pass@1 metric temperature = 0.6, top_p = 0.95, and max_tokens = 16384, following the flashinfer [45] framework. The answers are extracted using the DeepMath Evaluation Library [15].

RL Settings All experiments are conducted using the TRL framework [46] on 4× NVIDIA H200 GPUs. We employed the DeepSeek API [14] as the reward model that provides both accuracy and conformance reward signals. We used the AdamW optimizer for optimization at a constant learning rate of 1 × 10−6. The training was performed with a global batch size of 128, utilizing micro-batch sizes of 8 for the 1.5B models and 4 for 7B models, respectively. We set the KL divergence coefficient to zero for unconstrained policy updates. The details of hyperparameters are provided in Section B.

Table 6: Hyperparameters and Training Configuration for TAC Reward with GSPO

Module Parameter Value Description

dataset_name DeepMath-103k Training dataset. max_prompt_length 1024 Maximum input prompt length. max_completion_length 16384 Maximum response length. per_device_train_batch_size 2 Batch size per device.

Data

torch_dtype bfloat16 Model precision. attn_implementation flash_attention_2 Attention implementation. use_vllm True Enable vLLM for inference. vllm_mode colocate vLLM execution mode.

Model

learning_rate 1 × 10−6 Learning rate for policy optimizer. max_steps 1000 Total number of training steps. gradient_accumulation_steps 4 Gradient accumulation steps. optimizer AdamW Optimizer type (TRL default).

Optimizer

num_generations 8 Number of rollouts per prompt. importance_sampling_level sequence Importance sampling granularity. beta 0.0 KL penalty coefficient. epsilon 0.2 Clipping parameter (default).

GRPO

distributed_type DEEPSPEED Distributed training backend. zero_stage 3 ZeRO optimization stage. zero3_save_16bit_model True Save model in 16-bit precision. mixed_precision bf16 Mixed precision training. num_processes 4 Number of GPU processes.

DeepSpeed

#### C Additional Analysis

##### C.1 Computational Complexity of the optimal alignment.

We analyze the computational cost of TACReward as the length of the reasoning increases. Let n = |σref| denote the length of the reference trace, m = |σi| the length of the i-th policy trace, and |Σ| the size of the activity set. In our setting, |Σ| ≤ 20 (Section 3).

Optimal alignment. Computing the optimal alignment γi∗ between a reference trace σref and a policy process model Mi can be cast as an A∗ search over the synchronous product of the two [29]. In the general case this search is exponential in the trace length, but three properties of our setting reduce it substantially: (i) the activity taxonomy is fixed, so |Σ| is a constant; (ii) the number of reachable intermediate states in each process model is bounded by a constant, since each Mi is discovered from a single trace; and (iii) the search graph contains no branching beyond local model moves. Under these conditions, the cost of finding γi∗ reduces to O(nlog n).

Output reward. Once γi∗ is obtained, the fitness score requires summing local move costs along the alignment, which takes O(n + m). The precision score iterates over events in the reference trace

and inspects the set of enabled activities at each step, costing O(n · |Σ|). Since |Σ| is constant, both scores reduce to O(n), and the harmonic mean in Equation 14 adds only constant overhead. The overall cost of computing riTAC is therefore dominated by the alignment step at O(nlog n).

##### C.2 TAC as a Proxy for Step-level Reward

While riTAC is delivered as a single scalar value per response, riTAC is computed from a fine-grained step-level comparison of reasoning processes via the alignment γi∗. Consequently, TACReward provides a proxy for step-level supervision without requiring explicit dense annotations.

The alignment γi∗ decomposes the discrepancy between the policy and teacher reasoning traces into individual moves, each incurring local cost δm(·). The total deviation cost performs implicit credit assignment at the reasoning transition level, even though the final reward is scalar. A log-only move (aref,≫) indicates that the policy model omits the reasoning activity that is present in the teacher’s

trace and corresponds to a missing logical step. Conversely, the model-only move (≫,am) reflects redundant or irrelevant reasoning activities that are not supported by the teacher’s process.

Under this decomposition, the fitness penalizes missing essential steps, whereas the precision penalizes excessive or unjustified behavior. Their harmonic mean captures both completeness and selectivity and prevents premature conclusions and unsupported over generations.

##### C.3 Why Single-Trace Can Be Effective.

Although process discovery is traditionally applied to multitrace event logs, our use case differs fundamentally from classical approaches. We aimed to infer a global reasoning process and construct a local structural abstraction that captures the permissible reasoning transitions implied by a single policy trace. For each single trace, a minimal process model was constructed that could be generalized beyond the observed sequence, encoding the ordering constraints, optional branches, and loops in the reasoning trace. This abstraction enabled conformance checking, allowing us to penalize structurally invalid reasoning behaviors.

##### C.4 Why Response-Level Comparison Is Insufficient.

A natural question is whether trace formalization can be skipped, and whether reasoning responses can be directly compared as sequences. Given two token sequences, y and y′, the classical edit distance [47] is

 

ded(i − 1,j) + 1, ded(i,j − 1) + 1, ded(i − 1,j − 1) + I[yi ̸= yj′]

(25)

ded(i,j) = min



that measures surface-level similarity via insertions, deletions, and substitutions. However, ded measures only surface edit operations on raw text, and cannot reliably capture the structural properties of reasoning (e.g., missing or spurious transitions) that are distinct from benign re-orderings [48]. Trace formalization separates reasoning structure from linguistic realization, enabling a meaningful structural comparison.

##### C.5 On Imperfect Reference Traces.

In this study, our approach does not assume that the reference model is correct. This assumption is shared using several existing post training methods, including GRPO and GSPO. These methods rely on reward models or KL-divergence regularization with respect to a reference policy, despite these references being imperfect [1]. The reference model must be more mature than the policy being optimized, providing a relatively stable and coherent reasoning process. Under this assumption, conformance-based rewards suppress early stage randomness and structurally invalid reasoning, whereas outcome-based rewards continue to guide convergence as the training progresses.

##### C.6 Role of Reward Components.

TACReward consists of two components: fitness and precision. To investigate the contribution of each component, we conducted an ablation study by training the policy model with only fitness, only precision, and both components. The results are summarized in Table 7.

- Table 7: Ablation Study on Reward Components: Fitness vs. Precision vs. Both (240 training steps)

KSAT 2025

LiveMath Bench

AIME 2025

Minerva Olympiad

w/ Fit 30.9 30.5 6.7 43.3 10 w/ Prec 35.7 35.7 3.3 46.7 12 w/ Both 37.1 38.2 10.0 53.3 15

##### C.7 Role of Mathematical Taxonomy.

The Mathematical Reasoning Taxonomy (MRT) provides a fixed set of reasoning activities for formalizing raw reasoning responses into comparable traces.

- Table 8: Ablation Study on the Effect of Mathematical Reasoning Taxonomy (MRT) (240 training steps)

KSAT 2025

LiveMath Bench

AIME 2025

Minerva Olympiad

w/o MRT 37.1 35.6 3.3 43.3 9 w/ MRT 37.1 38.2 10.0 53.3 15

Table 8 indicates that removing MRT degrades performance, with larger drops on multistep benchmarks (e.g., AIME 2025 and KSAT 2025). In this setting, the rewards become less sensitive to structural differences and are dominated by surface variation without an explicit taxonomy.

#### D Prompts for Model Training

- D.1 Prompt for Generation

System Prompt for Generation

A conversation between a user and an assistant. The user asks a question, and the assistant solves it. The assistant MUST first think through the solution inside < think > ··· < /think >, and this block MUST contain ONLY step-by-step reasoning with no final answer or filler text. Immediately after < /think >, on a new line, the assistant MUST present the final answer exactly once: no duplicate answers, no alternatives, and no rephrasings. The entire final result MUST be enclosed in a single \boxed{ }, and there MUST be no additional text outside that single \boxed{ } in the final answer. Format template: < think >Step-bystep reasoning goes here.< /think > Final answer here with the result in \boxed{ }.

Example Input Format:

User: This is the problem: {Question} Assistant: < think >

- D.2 Prompt for Formalizing Trace

Prompt Template for Trace Formalization

System Role: You are an Expert Mathematical Problem Solver and Process Mining Assistant. Your goal is to convert unstructured logical reasoning steps into a structured, generalized CSV event log ready for process mining analysis.

Guidelines for Generalization: You must select the Activity label that best describes the intent of the reasoning step. Use the definitions below to ensure consistency:

- 11. Justify Step
- 12. Explore Edge Cases
- 13. Identify Contradiction
- 14. Interpret Result
- 15. Check Validity
- 16. Verify With Example
- 17. Refine or Change Strategy
- 18. Conclude Final Result
- 19. Recheck Original Question
- 20. End Problem

- 1. Start Problem
- 2. Recall Definition
- 3. Identify Known Results
- 4. Formulate Strategy
- 5. Apply Known Formula
- 6. Simplify Expression
- 7. Change of Variable
- 8. Evaluate Limit or Integral
- 9. Perform Comparison
- 10. Apply Theorem

##### Absolute Constraints:

- • NEVER invent new activities. Use ONLY the 20 labels listed above.
- • ALWAYS select exactly one of the above labels for the Activity column.
- • Use each activity label according to its abstract logical purpose.
- • Each step should reflect one unit of reasoning with a concise description.
- • Process the input line-by-line: output exactly one CSV row per non-empty line.

##### Example Output Format:

Case ID Step Activity Description

- 1 01 Start Problem Problem: Find function g(x) such that...
- 1 02 Formulate Strategy Break down the problem into constraints
- 1 03 Recall Definition Remember that xne−x is bounded...
- 1 04 Perform Comparison Compare integrals...
- 1 05 Apply Known Formula Use integral of e−x from 0 to 1
- 1 06 Simplify Expression Solve for c such that total = 1
- 1 07 Check Validity Ensure inequality is strict
- 1 08 Conclude Final Result Final answer: g(x) = e−x + 1/e
- 1 09 End Problem Problem solved with valid solution

Input: {problem} Return your output only in CSV format starting with the header.

#### E Reasoning Process Visualization with and without TACReward

###### Case Study: LiveMathBench (No.95)

Problem

Alice and Bob play a game. Initially, Clara writes the integers 2,3,…,30 on a board. Players take turns erasing integers, with Clara choosing a non-empty set 𝑆1 for Alice to erase in the first turn. For 𝑘 ≥ 2, the player choosing 𝑆𝑘 must ensure that for each 𝑥 ∈ 𝑆𝑘, there exists 𝑦 ∈ 𝑆𝑘−1 such that gcd(x,y) ≠ 1. The first player to choose an empty set loses. How many non-empty sets S1 allow Alice to win regardless of Bob’s choices?",

w/o TAC

Fitness: 0.78, Precision: 0.20, F1: 0.32

w/ TAC

###### Fitness: 0.54, Precision: 0.67, F1: 0.60

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

###### Activity

A: Apply Known Formula, B: Apply Theorem, C: Change of Variable, D: Check Validity, E: Conclude Final Result, F: End Problem, G: Evaluate Limit or Integral, H: Formulate Strategy, I: Identify Known Results, J: Justify Step, K: Perform Comparison, L: Recall Definition, M: Simplify Expression, N: Start Problem, O: Verify With Example

#### F Reasoning Process Visualization across Training Progress

Case Study: AIME 2025 (No.9) Problem

Let k be a real number such that the system 25 + 20i - z = 5 & z − 3i − k = z − 3i − k has exactly one complex solution z. The sum of all possible

values of k can be written as mn, where m and n are relatively prime positive integers. Find m + n. Here i = -1 .

Step : 100

Wrong F1: 0.45

Step: 300

Wrong F1: 0.32

Step: 500 Correct F1: 0.52

Step: 700 Wrong F1: 47

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

| |
|---|

| |
|---|

| |
|---|

Step: 1000 Correct F1: 0.55

#### G Reasoning Process Visualization under Fitness, Precision, and F1-Score

|Case Study: Minerva-Math (No.147) Problem| |
|---|---|
|For red light of wavelength 𝜆 = 6.7102 × 10−5𝑐𝑚, emitted by excited lithium atoms, calculate:<br><br>Subproblem 0: 𝑐 = 𝜆𝑣 and 𝑣 = 𝜆𝑐 where 𝑣 is the frequency of radiation (number of waves/s).<br><br>Subproblem 1: the wave number 𝑣 in 𝑐𝑚−1. Please format your answer as 𝑛 × 10𝑥, where 𝑛 is to 4 decimal places.<br>Subproblem 2: the wavelength 𝜆 in nm, to 2 decimal places.<br>| |
|Fitness Fitness: 0.83, Precision: 0.30, F1: 0.44, Tokens:1615<br><br>| |
|Precision Fitness: 0.67, Precision: 0.67, F1: 0.67, Tokens: 190<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|F1 (TAC) Fitness: 0.91, Precision: 0.50, F1: 0.65, Tokens: 340<br><br>| |
|Activity<br><br>A: Apply Known Formula, B: Change of Variable, C: Check Validity, D: Conclude Final Result, E: End Problem, F: Evaluate Limit or Integral, G: Formulate Strategy, H: Identify Known Results, I: Recall Definition, J: Simplify Expression, K: Start Problem| |

ҧ

###### Case Study: KSAT (No.9) Problem

닫힌구간 [0,2𝜋]에서 정의된 함수 𝑓 𝑥 = 𝑎 cos 𝑏𝑥 + 3이

𝑥 = 𝜋3 에서 최댓값 13을 갖도록 하는 두 자연수 𝑎,𝑏의 순서쌍 (𝑎,𝑏)에 대하여 𝑎 + 𝑏의 최솟값은? [4점]

Fitness

Fitness: 0.75, Precision: 0.30, F1: 0.43, Tokens: 373

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

| |
|---|

| |
|---|

Precision

Fitness: 0.63, Precision: 0.75, F1: 0.68, Tokens: 108

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|
| | |

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

F1 (TAC)

Fitness: 0.63, Precision: 0.44, F1: 0.52, Tokens: 454

Activity

A: Apply Known Formula, B: Check Validity, C: Conclude Final Result, D: End Problem, E: Evaluate Limit or Integral, F: Formulate Strategy, G: Identify Known Results, H: Interpret Result, I: Recall Definition, J: Simplify Expression, K: Start Problem, L: Verify With Example

###### Case Study: LiveMathBench (No.39) Problem

Donald sits at the origin (0,0) in the 𝑥𝑦-plane. Huey sits at the intersection of the lines 𝑦 = 3𝑥 − 5 and 𝑦 = 4𝑥 − 8. Dewey sits at the lowest point of the parabola 𝑦 = 𝑥2 + 5$. Louie sits at the intersection of the circle 𝑥2 + 𝑦2 = 30 and the line 𝑥 = 2𝑦 in the first quadrant. Which among Huey, Dewey, and Louie sit closest to Donald?

Fitness

Fitness: 0.44, Precision: 0.50, F1: 0.47, Tokens: 747

Precision

Fitness: 0.52, Precision: 0.75, F1: 0.61, Tokens: 635

F1 (TAC)

Fitness: 0.76, Precision: 0.33, F1: 0.46, Tokens: 845

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

###### Activity

A: Apply Known Formula, B: Change of Variable, C: Check Validity , D:

Conclude Final Result, E: End Problem, F: valuate Limit or Integral, G: Formulate Strategy, H: Identify Known Results, I: Interpret Result, J: Perform Comparison, K: Recall Definition, L: Simplify Expression, M: Start Problem

